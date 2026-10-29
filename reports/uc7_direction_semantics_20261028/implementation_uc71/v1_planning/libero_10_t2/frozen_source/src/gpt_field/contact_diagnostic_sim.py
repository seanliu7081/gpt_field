"""Evaluator-only short contact diagnostics with a separate simulator process.

Privileged contacts, object positions and articulation coordinates are returned
ONLY as ``evaluator_metrics``. Normal reset/step still return PolicyObservation
with the existing RGB/state allowlist. No policy, model or network is imported.
The local engagement criterion is an evaluation proxy, never LIBERO completion.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import sys

from .libero import (LiberoConfig, SimProcess, SimulatorError, _LiberoWorker,
    _TaskManipulationCollisionMonitor, _collision_task_key, _validate_action)
from .observation import OBSERVATION_KEYS, PolicyObservation


CONTACT_DIAGNOSTIC_VERSION = "contact-engagement-evaluator-v2"
FINGERPRINT_VERSION = "policy-observation-fingerprint-v1"
MAX_REPLAY_ACTIONS = 5000


def _state_value(value):
    """Deterministic controller-state serialization, with no memory addresses.

    The simulator itself is fingerprinted separately. Unsupported controller
    members fail closed rather than silently yielding an incomplete identity.
    """
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if hasattr(value, "dtype") and hasattr(value, "tobytes"):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "bytes_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, (tuple, list)):
        return [_state_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Controller state dictionary keys must be strings")
        return {key: _state_value(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {"class": type(value).__module__+"."+type(value).__qualname__,
                "members": {key: _state_value(item) for key, item in vars(value).items()
                            if key != "sim"}}
    raise ValueError("Unsupported controller state member: "+type(value).__qualname__)


def _physics_fingerprint(worker):
    """Evaluator-only exact replay identity including controller hidden state."""
    import random
    import numpy as np
    sim = worker.env.sim
    core = getattr(worker.env, "env", worker.env)
    arrays = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
              "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")
    physics = {key: _state_value(getattr(sim.data, key)) for key in arrays}
    physics["time"] = _state_value(float(sim.data.time))
    physics["flattened_state"] = _state_value(sim.get_state().flatten())
    controllers = [{"controller": _state_value(robot.controller),
                    "gripper_current_action": _state_value(robot.gripper.current_action)}
                   for robot in core.robots]
    model = sim.model.get_xml() if hasattr(sim.model, "get_xml") else core.model.get_xml()
    if not isinstance(model, bytes):
        model = model.encode("utf-8")
    components = {"physics": physics, "controllers": controllers,
                  "config": asdict(worker.config),
                  "model_xml_sha256": hashlib.sha256(model).hexdigest(),
                  "environment_clock": {key: _state_value(getattr(core, key, None))
                                        for key in ("timestep", "cur_time", "done")},
                  "python_random": _state_value(random.getstate()),
                  "numpy_random": _state_value(np.random.get_state())}
    hashes = {key+"_sha256": hashlib.sha256(_canonical(value).encode()).hexdigest()
              for key, value in components.items()}
    return {"protocol": "evaluator-physics-controller-fingerprint-v1",
            "scope": "evaluator_only_never_policy_input", "component_hashes": hashes,
            "combined_sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(),
            "coverage": "MuJoCo state, warmstart/control/applied forces/mocap/userdata, all controller members except separately fingerprinted sim, gripper command, config, XML, env clock and Python/NumPy RNG",
            "observation_cache_included": False}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation_fingerprint(observation):
    """Exact observable checkpoint identity; encoded RGB and full state values."""
    data = observation.to_dict() if isinstance(observation, PolicyObservation) else copy.deepcopy(observation)
    if not isinstance(data, dict) or set(data) != OBSERVATION_KEYS:
        raise ValueError("Diagnostic checkpoint requires only the two RGB views and state8")
    checked = PolicyObservation.from_dict(data)
    if len(checked.state) != 8:
        raise ValueError("Diagnostic checkpoint requires state8")
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {"protocol": FINGERPRINT_VERSION,
        "observation_sha256": digest(_canonical(data)),
        "state8_sha256": digest(_canonical(data["observation/state"])),
        "external_image_sha256": digest(data["observation/image"]),
        "wrist_image_sha256": digest(data["observation/wrist_image"])}


def _validate_fingerprint(value):
    keys = {"protocol", "observation_sha256", "state8_sha256", "external_image_sha256", "wrist_image_sha256"}
    if not isinstance(value, dict) or set(value) != keys or value["protocol"] != FINGERPRINT_VERSION:
        raise ValueError("Invalid diagnostic checkpoint fingerprint")
    for key in keys - {"protocol"}:
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("Checkpoint hashes must be lowercase SHA256")
    return copy.deepcopy(value)


@dataclass(frozen=True)
class EngagementThresholds:
    minimum_bilateral_frames: int = 3
    minimum_motion_m: float = .01
    minimum_tool_motion_m: float = .005
    maximum_relative_drift_m: float = .015

    def __post_init__(self):
        if type(self.minimum_bilateral_frames) is not int or self.minimum_bilateral_frames < 3:
            raise ValueError("Engagement requires at least three bilateral-contact action endpoints")
        for name in ("minimum_motion_m", "minimum_tool_motion_m", "maximum_relative_drift_m"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= .1:
                raise ValueError(f"{name} must be a finite positive metric threshold <=10cm")


def _target(value):
    if value not in ("bowl", "drawer_handle"):
        raise ValueError("Diagnostic target must be bowl or drawer_handle")
    return value


def _limit(value):
    if type(value) is not int or not 1 <= value <= 200:
        raise ValueError("Diagnostic action limit must be an integer in [1,200]")
    return value


def _point(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must be three finite numbers")
    return [float(x) for x in value]


def _measurement(value, target):
    """Copy and validate the minimum evaluator measurement contract."""
    if not isinstance(value, dict):
        raise ValueError("Contact measurement must be an object")
    required = {"target_xyz_m", "tool_xyz_m", "articulation_q_m", "left_contact", "right_contact",
                "left_pad_contact", "right_pad_contact", "other_support_contact", "nonfinger_robot_contact"}
    if not required.issubset(value):
        raise ValueError("Contact measurement is incomplete")
    result = copy.deepcopy(value)
    for key in ("target_xyz_m", "tool_xyz_m"):
        result[key] = _point(value[key], key)
    for key in required - {"target_xyz_m", "tool_xyz_m", "articulation_q_m"}:
        if type(value[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    q = value["articulation_q_m"]
    if target == "drawer_handle":
        if type(q) not in (float, int) or not math.isfinite(q):
            raise ValueError("Articulated diagnostic requires finite metric slide position")
    elif q is not None:
        raise ValueError("Free-object diagnostic has no articulation coordinate")
    return result


def _delta(a, b):
    return [a[i] - b[i] for i in range(3)]


def _norm(a):
    return math.sqrt(sum(x*x for x in a))


class EngagementEvidence:
    """Pure evaluator: simultaneous two-sided contact plus sustained motion.

    A capture epoch starts on a CLOSED-command endpoint with simultaneous
    left/right target contact and resets immediately if either is absent. The
    qualifying co-motion interval lies wholly within that epoch; an earlier push
    or supported seating cannot substitute for demonstrated free-object lift.
    """

    def __init__(self, target, thresholds=None, *, diagnostic_action_limit=100):
        self.target = _target(target)
        self.thresholds = thresholds or EngagementThresholds()
        if not isinstance(self.thresholds, EngagementThresholds):
            raise ValueError("thresholds must be EngagementThresholds")
        self.limit = _limit(diagnostic_action_limit)
        self.armed = False
        self.actions = 0
        self.origin = self.capture = self.latest = None
        self.bilateral_streak = self.bilateral_frames = 0
        self.success_ever = False
        self.first_success = None
        self._last_step = None
        self._motion_window = []
        self._latest = self._snapshot("not_armed", False, {})

    def begin(self, measurement):
        if self.armed:
            raise RuntimeError("A diagnostic may begin only once per reset")
        measured = _measurement(measurement, self.target)
        self.origin = self.latest = measured
        self.armed = True
        self._latest = self._snapshot("ready", False, {})
        return self.snapshot()

    def _snapshot(self, status, success, deltas):
        return {"protocol": CONTACT_DIAGNOSTIC_VERSION,
            "scope": "evaluator_only_never_policy_input", "target": self.target,
            "armed": self.armed, "criterion_ready": self.armed,
            "diagnostic_actions": self.actions, "diagnostic_action_limit": self.limit,
            "diagnostic_limit_reached": self.actions >= self.limit,
            "engagement_success": success, "engagement_success_ever": self.success_ever,
            "first_engagement_step": self.first_success,
            "evidence_status": status, "bilateral_streak": self.bilateral_streak,
            "bilateral_frames": self.bilateral_frames,
            "measurements": copy.deepcopy(self.latest), "deltas": copy.deepcopy(deltas),
            "thresholds": asdict(self.thresholds),
            "criterion_provenance": {
                "contact_sampling": "after every completed 20Hz action; simultaneous contacts with distance<=0",
                "threshold_selection": "predeclared diagnostic criteria, not fitted to trial outcomes",
                "motion_origin": "a contiguous closed-command bilateral-contact co-motion interval; free-object intervals additionally exclude support/nonfinger contact throughout",
                "qualification_revision": "v2 separates initial contact seating from later unsupported co-motion; identified on an archived positive control before any paired policy trials",
                "free_object": "positive Z displacement and tool motion exceed declared thresholds over a continuous bilateral interval; bounded drift and no other support throughout that interval",
                "articulated_object": "absolute slide displacement and tool motion exceed declared thresholds; sustained bilateral handle contact and bounded relative drift",
                "native_grasp_check": "robosuite _check_grasp tests contact in both pad groups only; it does not test motion, force closure, stability or task completion",
            },
            "limitations": [
                "This is a conservative contact-and-motion proxy, not force-closure certification.",
                "Only completed-action contacts are sampled here; transient substep contacts may be missed.",
                "A successful unilateral hook may move an articulation but does not satisfy this bilateral criterion.",
                "Articulation displacement is absolute; signed opening/closing motion is reported separately.",
                "A past successful capture does not establish current engagement after contact is lost.",
            ],
            "task_success_certified": False, "libero_success_replaced": False}

    def observe(self, measurement, action, step):
        if not self.armed:
            raise RuntimeError("begin_diagnostic is required before evaluating contact actions")
        if self.actions >= self.limit:
            raise RuntimeError("Short diagnostic action limit reached")
        values = _validate_action(action)
        measured = _measurement(measurement, self.target)
        if type(step) is not int or step < 0 or (self._last_step is not None and step != self._last_step+1):
            raise ValueError("Diagnostic completed-action steps must be consecutive")
        self._last_step = step
        self.actions += 1
        self.latest = measured
        closed = values[6] > 0
        bilateral = measured["left_contact"] and measured["right_contact"]
        if closed and bilateral:
            if self.capture is None:
                self.capture = copy.deepcopy(measured)
            self.bilateral_streak += 1
            self.bilateral_frames += 1
        else:
            self.capture = None
            self.bilateral_streak = 0
        unsupported = not (measured["other_support_contact"] or measured["nonfinger_robot_contact"])
        if closed and bilateral and (self.target != "bowl" or unsupported):
            self._motion_window.append((step, copy.deepcopy(measured)))
        else:
            self._motion_window = []
        deltas = {"target_from_checkpoint_xyz_m": _delta(measured["target_xyz_m"], self.origin["target_xyz_m"]),
                  "tool_from_checkpoint_xyz_m": _delta(measured["tool_xyz_m"], self.origin["tool_xyz_m"])}
        if self.target == "drawer_handle":
            dq = measured["articulation_q_m"] - self.origin["articulation_q_m"]
            deltas.update(articulation_from_checkpoint_m=dq,
                         opening_motion_from_checkpoint_m=-dq)
        success = False
        if self.capture is not None:
            target_delta = _delta(measured["target_xyz_m"], self.capture["target_xyz_m"])
            tool_delta = _delta(measured["tool_xyz_m"], self.capture["tool_xyz_m"])
            drift = _norm(_delta(target_delta, tool_delta))
            movement = (target_delta[2] if self.target == "bowl" else
                        abs(measured["articulation_q_m"]-self.capture["articulation_q_m"]))
            deltas.update(target_from_capture_xyz_m=target_delta,
                          tool_from_capture_xyz_m=tool_delta, relative_translation_drift_m=drift,
                          criterion_motion_m=movement, tool_motion_m=_norm(tool_delta))
            threshold = self.thresholds
            # Initial finger contact can seat or rotate an object on its support.
            # Qualify subsequent continuous co-motion rather than preserving that
            # seating displacement forever as an apparent slip. Every endpoint
            # in the selected interval must remain within the same drift bound.
            for index in range(len(self._motion_window)-threshold.minimum_bilateral_frames+1):
                anchor_step, anchor = self._motion_window[index]
                target_move = _delta(measured["target_xyz_m"], anchor["target_xyz_m"])
                tool_move = _delta(measured["tool_xyz_m"], anchor["tool_xyz_m"])
                amount = (target_move[2] if self.target == "bowl" else
                          abs(measured["articulation_q_m"]-anchor["articulation_q_m"]))
                maximum_drift = max(_norm(_delta(
                    _delta(sample["target_xyz_m"], anchor["target_xyz_m"]),
                    _delta(sample["tool_xyz_m"], anchor["tool_xyz_m"])))
                    for _, sample in self._motion_window[index:])
                if (amount >= threshold.minimum_motion_m
                        and _norm(tool_move) >= threshold.minimum_tool_motion_m
                        and maximum_drift <= threshold.maximum_relative_drift_m):
                    success = True
                    deltas.update(qualifying_interval_start_step=anchor_step,
                        qualifying_interval_frames=len(self._motion_window)-index,
                        qualifying_interval_motion_m=amount,
                        qualifying_interval_tool_motion_m=_norm(tool_move),
                        qualifying_interval_maximum_relative_drift_m=maximum_drift)
                    break
            status = "engagement_demonstrated" if success else "bilateral_contact_without_validated_motion"
        elif measured["left_contact"] or measured["right_contact"]:
            status = "one_sided_contact_or_push" if not bilateral else "bilateral_contact_without_close_intent"
        elif closed:
            status = "close_command_without_target_contact"
        else:
            status = "no_bilateral_capture"
        if success:
            self.success_ever = True
            if self.first_success is None:
                self.first_success = step
        self._latest = self._snapshot(status, success, deltas)
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy(self._latest)


class _ContactInspector:
    """Read named geometry and simulator truth only inside the evaluator worker."""

    def __init__(self, worker, target):
        self.target = _target(target)
        self.env = worker.env
        self.core = getattr(self.env, "env", self.env)
        self.sim = self.env.sim
        task_key = _collision_task_key(worker.config.suite, worker.status["task_id"])
        if self.target == "drawer_handle" and task_key != "drawer_bowl":
            raise ValueError("Selected task has no supported drawer handle")
        monitor = getattr(worker, "_collision_monitor", None)
        if not isinstance(monitor, _TaskManipulationCollisionMonitor):
            monitor = _TaskManipulationCollisionMonitor(self.env, task_key=task_key, stop_on_collision=False)
        self.geometry = monitor._geometry
        robots = self.core.robots
        if len(robots) != 1 or isinstance(robots[0].gripper, dict):
            raise ValueError("Diagnostic supports one Panda gripper")
        gripper = self.gripper = robots[0].gripper
        self.groups = {}
        for side in ("left", "right"):
            pad = set(gripper.important_geoms.get(side+"_fingerpad", []))
            finger = set(gripper.important_geoms.get(side+"_finger", [])) | pad
            self.groups[side] = {i for i, d in self.geometry.items() if d["name"] in finger}
            self.groups[side+"_pad"] = {i for i, d in self.geometry.items() if d["name"] in pad}
            if not self.groups[side] or not self.groups[side+"_pad"]:
                raise ValueError("Both named finger and finger-pad groups are required")
        if self.groups["left"] & self.groups["right"]:
            raise ValueError("Left and right contact groups must be disjoint")
        role = "target_bowl" if target == "bowl" else "drawer_handle"
        self.target_geoms = {i for i, record in self.geometry.items() if record["role"] == role}
        if not self.target_geoms:
            raise ValueError("Target contact geometry could not be identified")
        self.body_name = (self.core.objects_dict["akita_black_bowl_1"].root_body if target == "bowl"
                          else "wooden_cabinet_1_cabinet_top")
        self.body_id = int(self.sim.model.body_name2id(self.body_name))
        self.joint_name = None if target == "bowl" else "wooden_cabinet_1_top_level"
        if self.joint_name is not None:
            joint = int(self.sim.model.joint_name2id(self.joint_name))
            if int(self.sim.model.jnt_type[joint]) != 2:  # MuJoCo slide, metric metres.
                raise ValueError("Articulation criterion requires a metric slide joint")
        self.tool_site = gripper.important_sites["grip_site"]

    def sample(self):
        contacts = []
        contact_groups = {key: False for key in self.groups}
        support = nonfinger = False
        for i in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[i]
            distance = float(contact.dist)
            if not math.isfinite(distance):
                raise ValueError("Nonfinite contact distance")
            if distance > 0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in self.target_geoms) == (second in self.target_geoms):
                continue
            target_geom, other = (first, second) if first in self.target_geoms else (second, first)
            for key, ids in self.groups.items():
                contact_groups[key] |= other in ids
            other_record = self.geometry.get(other)
            role = None if other_record is None else other_record["role"]
            if other not in self.groups["left"] | self.groups["right"]:
                nonfinger |= role in ("robot", "robot_finger")
                support |= role not in ("robot", "robot_finger")
            contacts.append({"target_geom": self.sim.model.geom_id2name(target_geom),
                "other_geom": self.sim.model.geom_id2name(other), "distance_m": distance,
                "other_role": role})
        native = None
        native_error = None
        if hasattr(self.core, "_check_grasp"):
            try:
                native = bool(self.core._check_grasp(self.gripper,
                    [self.sim.model.geom_id2name(i) for i in sorted(self.target_geoms)]))
            except Exception as exc:
                native_error = type(exc).__name__
        result = {"target_xyz_m": [float(x) for x in self.sim.data.body_xpos[self.body_id]],
            "tool_xyz_m": [float(x) for x in self.sim.data.get_site_xpos(self.tool_site)],
            "articulation_q_m": (None if self.joint_name is None else
                                   float(self.sim.data.get_joint_qpos(self.joint_name))),
            "left_contact": contact_groups["left"], "right_contact": contact_groups["right"],
            "left_pad_contact": contact_groups["left_pad"], "right_pad_contact": contact_groups["right_pad"],
            "other_support_contact": bool(support), "nonfinger_robot_contact": bool(nonfinger),
            "native_pad_grasp_check": native, "native_grasp_check_error": native_error,
            "contacts": contacts,
            "target_body": self.body_name, "articulation_joint": self.joint_name,
            "source": "MuJoCo_contact_geometry_body_position_and_slide_joint_evaluator_only"}
        return _measurement(result, self.target)


class ContactDiagnosticSimProcess(SimProcess):
    """Evaluator client; privileged metrics never enter returned observations."""

    def __init__(self, config: LiberoConfig, *, target, diagnostic_action_limit=100,
                 thresholds=None, command=None):
        self.target = _target(target)
        self.diagnostic_action_limit = _limit(diagnostic_action_limit)
        self.thresholds = thresholds or EngagementThresholds()
        if not isinstance(self.thresholds, EngagementThresholds):
            raise ValueError("thresholds must be EngagementThresholds")
        if config.include_arm_joints or config.stop_on_collision:
            raise ValueError("Contact diagnostic requires state8 and record-only collisions")
        if command is None:
            command = [config.python_executable or sys.executable, "-u", "-m", "gpt_field.contact_diagnostic_sim",
                "--worker", "--config-json", _canonical(asdict(config)), "--diagnostic-json", _canonical({
                    "target": self.target, "diagnostic_action_limit": self.diagnostic_action_limit,
                    "thresholds": asdict(self.thresholds)})]
        self.evaluator_metrics = {}
        self.latest_observation = None
        self.checkpoint_check = None
        super().__init__(config, command=command)

    def _request(self, request):
        result = super()._request(request)
        if "evaluator_metrics" in result:
            self.evaluator_metrics = copy.deepcopy(result["evaluator_metrics"])
        return result

    def _observation(self, data):
        observation = super()._observation(data)
        if set(data) != OBSERVATION_KEYS:
            raise SimulatorError("Diagnostic observation has unexpected privileged fields")
        self.latest_observation = observation
        return observation

    def replay_checkpoint(self, task_id, episode_index, actions, expected_fingerprint, *, initial_fingerprint=None):
        expected = _validate_fingerprint(expected_fingerprint)
        initial_expected = None if initial_fingerprint is None else _validate_fingerprint(initial_fingerprint)
        if not isinstance(actions, (list, tuple)) or len(actions) > MAX_REPLAY_ACTIONS:
            raise ValueError("Checkpoint prefix must contain at most 5000 recorded actions")
        # Validate the entire immutable prefix before resetting or executing it.
        checked = [_validate_action(action) for action in actions]
        observation = self.reset(task_id, episode_index)
        initial_actual = observation_fingerprint(observation)
        if initial_expected is not None and initial_actual != initial_expected:
            raise SimulatorError("Initial reset fingerprint differs before any checkpoint replay action")
        for action in checked:
            observation, _ = self.step(action)
        actual = observation_fingerprint(observation)
        self.checkpoint_check = {"matched": actual == expected, "expected": expected, "actual": actual,
            "replayed_actions": len(checked), "method": "fresh_official_reset_then_recorded_actions",
            "initial_actual": initial_actual, "initial_expected": initial_expected,
            "initial_matched": None if initial_expected is None else initial_actual == initial_expected}
        if actual != expected:
            raise SimulatorError("Replayed checkpoint RGB/state fingerprint differs from frozen archive")
        return observation

    def begin_diagnostic(self, expected_fingerprint):
        expected = _validate_fingerprint(expected_fingerprint)
        if self.latest_observation is None or observation_fingerprint(self.latest_observation) != expected:
            raise SimulatorError("The exact observable checkpoint must match before diagnostic execution")
        result = self._request({"op": "begin_diagnostic", "expected_fingerprint": expected})
        self.status = result["status"]
        return copy.deepcopy(self.evaluator_metrics)


class _ContactDiagnosticWorker(_LiberoWorker):
    def __init__(self, config, *, target, diagnostic_action_limit=100, thresholds=None):
        self.target = _target(target)
        self.diagnostic_action_limit = _limit(diagnostic_action_limit)
        self.thresholds = EngagementThresholds(**(thresholds or {}))
        if config.include_arm_joints or config.stop_on_collision:
            raise ValueError("Diagnostic worker requires state8 and record-only collision protocol")
        self.inspector = self.evidence = None
        self.latest_observation = None
        self.checkpoint_physics_fingerprint = None
        super().__init__(config)
        self.metadata["contact_diagnostic"] = {"version": CONTACT_DIAGNOSTIC_VERSION,
            "target": target, "diagnostic_action_limit": diagnostic_action_limit,
            "thresholds": asdict(self.thresholds), "privileged_metrics_are_policy_inputs": False,
            "completion_is_local_contact_diagnostic_not_full_task": True}

    def reset(self, task_id, episode_index):
        self.inspector = self.evidence = None
        self.latest_observation = None
        self.checkpoint_physics_fingerprint = None
        result = super().reset(task_id, episode_index)
        self.inspector = _ContactInspector(self, self.target)
        self.evidence = EngagementEvidence(self.target, self.thresholds,
                                          diagnostic_action_limit=self.diagnostic_action_limit)
        self.latest_observation = copy.deepcopy(result["observation"])
        result["evaluator_metrics"] = self.evidence.snapshot()
        return result

    def begin_diagnostic(self, expected_fingerprint):
        expected = _validate_fingerprint(expected_fingerprint)
        if self.evidence is None or self.status.get("terminated"):
            raise RuntimeError("An active checkpoint is required")
        if observation_fingerprint(self.latest_observation) != expected:
            raise ValueError("Worker checkpoint fingerprint mismatch")
        self.checkpoint_physics_fingerprint = _physics_fingerprint(self)
        metrics = self.evidence.begin(self.inspector.sample())
        metrics["checkpoint_physics_fingerprint"] = copy.deepcopy(self.checkpoint_physics_fingerprint)
        return {"status": self.status.copy(), "evaluator_metrics": metrics}

    def step(self, action):
        if self.evidence is None:
            raise RuntimeError("Reset is required before stepping")
        if self.evidence.armed and self.evidence.actions >= self.diagnostic_action_limit:
            raise RuntimeError("Short diagnostic action limit reached")
        if not self.evidence.armed and self.status["steps"] >= MAX_REPLAY_ACTIONS:
            raise RuntimeError("Recorded checkpoint prefix limit reached")
        result = super().step(action)
        self.latest_observation = copy.deepcopy(result["observation"])
        if self.evidence.armed:
            metrics = self.evidence.observe(self.inspector.sample(), action, self.status["steps"]-1)
            if metrics["diagnostic_limit_reached"]:
                self.status.update(terminated=True, diagnostic_limit_reached=True)
                result["status"] = self.status.copy()
        else:
            metrics = self.evidence.snapshot()
        metrics["checkpoint_physics_fingerprint"] = copy.deepcopy(self.checkpoint_physics_fingerprint)
        result["evaluator_metrics"] = metrics
        return result


def _worker_main(config, diagnostic):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if worker is None:
                    worker = _ContactDiagnosticWorker(config, **diagnostic)
                op = request["op"]
                if op == "metadata":
                    result = worker.metadata
                elif op == "tasks":
                    result = {"tasks": worker.tasks()}
                elif op == "reset":
                    result = worker.reset(request["task_id"], request["episode_index"])
                elif op == "step":
                    result = worker.step(request["action"])
                elif op == "begin_diagnostic":
                    result = worker.begin_diagnostic(request["expected_fingerprint"])
                else:
                    raise ValueError("Unsupported diagnostic simulator operation")
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response, allow_nan=False)+"\n")
            protocol.flush()
    finally:
        if worker is not None:
            worker.close()
        protocol.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--diagnostic-json", required=True)
    args = parser.parse_args()
    return _worker_main(LiberoConfig(**json.loads(args.config_json)), json.loads(args.diagnostic_json))


if __name__ == "__main__":
    raise SystemExit(main())
