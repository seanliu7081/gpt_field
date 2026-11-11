"""Geometry-conditioned execution of model waypoints, with measured handoffs.

The outer v2 actuator/target limits remain hard bounds. Geometry can shorten
commands and the observation horizon; it cannot certify physical clearance.
The committed prefix is fixed at entry and may only shrink without new vision.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .action import ActionAdapter, ROTATION_SCALE_RAD, TRANSLATION_SCALE_M
from .agent_speedup_v2_user_check_fragments import (
    FragmentSequencePrimitive, MOTION_SKILLS, PROFILES, fragment_capabilities,
    STALL_MIN_STEPS, STALL_WINDOW_STEPS, STALL_PROGRESS_M,
)
from .agent_speedup_v2_user_check_motion import LocalMotionPrimitive
from .astranomy_agentic_motion import (
    ShortTargetConfig, ShortTargetExecutor, measured_pose, rotation_error_rad,
    OSC_ORIENTATION_REFRESH_RAD,
)

FRAGMENT_VERSION = "agent-speedup-v3-adaptive-fragments-10"


def adaptive_fragment_capabilities():
    result = fragment_capabilities()
    result.update(version=FRAGMENT_VERSION, shared_geometry=True,
                  planning="goal-local measured execution; coarse geometry advises candidates and checks robust conflicts; missing free boxes alone never veto motion",
                  unknown_continuation={"policy": "current public RGB, bounded controller target and finite progress watchdog; unknown is never labelled free"},
                  supported_continuation="same permitted goal may renew local chunks with normal tracking and measured progress; a once-estimated intent watchdog is never reset",
                  commitment="fixed at segment entry; only shrinks until a new model observation",
                  adaptive_speed="current coupled command is checked independently of persistent profile caps; supported fixed prefixes may continue to their frontier",
                  collision_scope="separate palm/fingers and estimated payload; not full arm",
                  native_macro_scope="native controllers retain their existing planning; geometry memory is shared",
                  physical_clearance_verified=False)
    result["semantics"].update(
        interruption="robust model conflict, observed change, tracking anomaly, missing payload shape or exhausted progress budget returns control",
        geometry_horizon="goal-local permission is separate from unknown free-space evidence; strict historical geometry retains its finite unknown horizon",
        tracking="per-channel drive/hold diagnostics; three valid drive samples and persistent anomalies are required; normal proprioception does not prove scene clearance",
        adaptive_handoff="geometry pause preserves the absolute goal and watchdog remainder; explicit continue_intent can supply fresh geometry without rewriting the action")
    return result


def _slerp(a, b, fraction):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1., max(-1., dot))
    if dot > .9995:
        q = a + fraction * (b - a)
        return (q / np.linalg.norm(q)).tolist()
    angle = math.acos(dot)
    return ((math.sin((1-fraction)*angle)*a + math.sin(fraction*angle)*b)/math.sin(angle)).tolist()


def _prefix(view, xyz, wxyz, fraction):
    pose = measured_pose(view.state)
    return (view.position + fraction*(np.asarray(xyz)-view.position),
            _slerp(pose[3:7], wxyz, fraction))


def _bounded_deadline(deadline, elapsed, suggested_steps):
    """Keep an existing deadline, or shorten it from this observation onward.

    Supported geometry has no additional horizon (None). A finite suggestion
    counts remaining steps, so evidence that degrades mid-segment gets a bounded
    continuation without either an immediate historical cutoff or rolling renewal.
    """
    if suggested_steps is None:
        return deadline
    return min(deadline, elapsed + int(suggested_steps))


class _TrackingEvidence:
    """Local response diagnostics, not a scene observation or a calibrated model."""
    def __init__(self):
        self.samples = []
        self.count = 0
        self.mismatch_count = 0
        self.anomaly_counts = {"translation": 0, "rotation": 0}
        self.normal_responses = {"translation": [], "rotation": []}
        self.evidence = {"status": "unobserved", "sample_count": 0, "window_steps": 0,
                         "consecutive_mismatch_steps": 0,
                         "commanded_translation_m": 0., "measured_translation_m": 0.,
                         "commanded_rotation_rad": 0., "measured_rotation_rad": 0.,
                         "position_progress_m": 0., "orientation_progress_rad": 0.,
                         "translation_response_ratio": None, "rotation_response_ratio": None,
                         "source": "measured_robot_proprioception"}

    def observe(self, before, action, after, xyz, wxyz, *, position_tolerance_m=.004,
                rotation_tolerance_rad=.06):
        command_xyz = np.asarray(action[:3], float) * TRANSLATION_SCALE_M
        command_rot = np.asarray(action[3:6], float) * ROTATION_SCALE_RAD
        actual_xyz = after.position - before.position
        actual_rot = rotation_error_rad(measured_pose(after.state)[3:7], before.state[3:6])
        commanded_translation = float(np.linalg.norm(command_xyz))
        commanded_rotation = float(np.linalg.norm(command_rot))
        position_before = float(np.linalg.norm(np.asarray(xyz)-before.position))
        position_after = float(np.linalg.norm(np.asarray(xyz)-after.position))
        rotation_before = float(np.linalg.norm(rotation_error_rad(wxyz, before.state[3:6])))
        rotation_after = float(np.linalg.norm(rotation_error_rad(wxyz, after.state[3:6])))
        position_progress = position_before-position_after
        orientation_progress = rotation_before-rotation_after
        sample = {"commanded_translation_m": commanded_translation,
                  "measured_translation_m": float(np.linalg.norm(actual_xyz)),
                  "commanded_rotation_rad": commanded_rotation,
                  "measured_rotation_rad": float(np.linalg.norm(actual_rot)),
                  "position_progress_m": position_progress,
                  "orientation_progress_rad": orientation_progress,
                  "position_error_m": position_after, "orientation_error_rad": rotation_after,
                  "translation_mode": "hold" if max(position_before, position_after) <= position_tolerance_m else "drive",
                  "rotation_mode": "hold" if max(rotation_before, rotation_after) <= rotation_tolerance_rad else "drive",
                  "translation_along_command_m": float(np.dot(actual_xyz, command_xyz))/max(commanded_translation, 1e-12),
                  "rotation_along_command_rad": float(np.dot(actual_rot, command_rot))/max(commanded_rotation, 1e-12)}
        self.count += 1
        self.samples = (self.samples + [sample])[-3:]
        channels = {}
        for axis, command, along, progress, minimum, tolerance in (
                ("translation", "commanded_translation_m", "translation_along_command_m", "position_progress_m", .0005, .0001),
                ("rotation", "commanded_rotation_rad", "rotation_along_command_rad", "orientation_progress_rad", .005, .002)):
            mode = sample[axis+"_mode"]
            rows = [s for s in self.samples if s[axis+"_mode"] == "drive" and s[command] >= minimum]
            response = sum(s[along] for s in rows)/sum(s[command] for s in rows) if rows else None
            # A holding channel inside its completion tolerance is not a drive
            # experiment. Tiny corrections and cross-axis lag cannot veto the
            # other channel or train its response model. Once outside tolerance,
            # it must accumulate its own valid drive evidence again.
            valid = mode == "drive" and len(rows) == 3
            improvement = sum(s[progress] for s in rows)
            anomalous = bool(valid and (response < -.05 or response > 2.5
                or improvement < -tolerance or (response < .03 and improvement <= tolerance)))
            self.anomaly_counts[axis] = self.anomaly_counts[axis]+1 if anomalous else 0
            channel_status = ("normal" if mode == "hold" else
                              "mismatch" if self.anomaly_counts[axis] >= 2 else
                              "normal" if valid and not anomalous else "unobserved")
            channels[axis] = {"mode": mode, "status": channel_status,
                              "valid_drive_samples": len(rows),
                              "consecutive_anomalous_windows": self.anomaly_counts[axis],
                              "response_ratio": response if mode == "drive" else None}
            if channel_status == "normal" and mode == "drive" and sample[progress] > 0.:
                self.normal_responses[axis] = (self.normal_responses[axis] + [min(1.5, max(.03, response))])[-12:]
        statuses = [channel["status"] for channel in channels.values()]
        status = "mismatch" if "mismatch" in statuses else "unobserved" if "unobserved" in statuses else "normal"
        self.mismatch_count = self.mismatch_count + 1 if status == "mismatch" else 0
        self.evidence = {**sample, "status": status, "sample_count": self.count,
                         "window_steps": len(self.samples),
                         "consecutive_mismatch_steps": self.mismatch_count,
                         "translation_response_ratio": channels["translation"]["response_ratio"],
                         "rotation_response_ratio": channels["rotation"]["response_ratio"],
                         "channels": channels,
                         "source": "measured_robot_proprioception"}

    def summary(self):
        result = deepcopy(self.evidence)
        for axis, values in self.normal_responses.items():
            result[f"{axis}_response_estimate"] = float(np.median(values)) if values else None
            result[f"{axis}_response_bounds"] = [min(values), max(values)] if values else None
        result["normal_response_samples"] = {axis: len(values) for axis, values in self.normal_responses.items()}
        result["response_bounds_are_empirical"] = True
        return result

    def progress_per_step(self):
        return (sum(s["position_progress_m"] + .05*s["orientation_progress_rad"]
                    for s in self.samples)/len(self.samples)) if self.samples else 0.

    def snapshot(self):
        return deepcopy({"samples": self.samples, "count": self.count,
                         "mismatch_count": self.mismatch_count, "evidence": self.evidence,
                         "anomaly_counts": self.anomaly_counts,
                         "normal_responses": self.normal_responses})

    def restore(self, state, *, same_measurement):
        self.normal_responses = deepcopy(state["normal_responses"])
        # Valid normal evidence survives a geometry-only patch. A changed robot
        # state or prior mismatch needs new short response observations instead.
        if same_measurement and state["evidence"]["status"] == "normal":
            self.samples = deepcopy(state["samples"])
            self.count = int(state["count"])
            self.mismatch_count = int(state["mismatch_count"])
            self.anomaly_counts = deepcopy(state.get("anomaly_counts", self.anomaly_counts))
            self.evidence = deepcopy(state["evidence"])


class _IntentWatchdog:
    """One finite estimate for this goal, preserved through geometry handoffs."""
    def __init__(self, initial_steps, *, learned=False):
        self.limit = int(initial_steps)
        self.learned = bool(learned)

    def learn_once(self, elapsed, distance, angle, tracking):
        progress = tracking.progress_per_step()
        if not self.learned and tracking.evidence["status"] == "normal" and progress > STALL_PROGRESS_M/STALL_WINDOW_STEPS:
            estimate = math.ceil(4.*(distance + .05*angle)/progress)
            self.limit = max(self.limit, int(elapsed) + max(1, estimate))
            self.learned = True


def _same_intent(saved, current):
    keys = ("intent", "target_id", "held_object_id", "allowed_contact_ids")
    return all(saved.get(key) == current.get(key) for key in keys)


def _validate_continuation(saved, kind, motion_context, keep_gripper):
    if saved.get("kind") != kind or not _same_intent(saved["motion_context"], motion_context):
        raise ValueError("continuation must preserve the authorized intent/target/held/contact semantics")
    ceiling = 120 if kind == "fragment" else 5
    for key in ("chunk_steps", "remaining_chunk_steps", "remaining_watchdog_steps"):
        if type(saved.get(key)) is not int or saved[key] <= 0:
            raise ValueError("continuation has no valid remaining authorized budget")
    if (saved["chunk_steps"] > ceiling or saved["remaining_chunk_steps"] > saved["chunk_steps"]
            or saved["remaining_chunk_steps"] > saved["remaining_watchdog_steps"]
            or type(saved.get("consumed_steps")) is not int or saved["consumed_steps"] < 0):
        raise ValueError("continuation budgets are inconsistent")
    if keep_gripper is not None and keep_gripper != saved["gripper_command"]:
        raise ValueError("continuation cannot change the maintained gripper command")


def _execution_status(assessment):
    return assessment.get("execution_status", assessment["status"])


def _execution_allowed(assessment):
    """Execution permission is distinct from knowledge of collision-free space."""
    return assessment.get("execution_allowed", _execution_status(assessment) == "supported")


def _assessed_prefix(view, xyz, wxyz, assessment):
    prefix = assessment.get("execution_prefix")
    if isinstance(prefix, dict):
        return np.asarray(prefix["xyz_m"], float), list(prefix["quaternion_wxyz"])
    return _prefix(view, xyz, wxyz, float(assessment["allowed_fraction"]))


def _scope_summary(assessment):
    return deepcopy({key: assessment[key] for key in (
        "status", "reason", "allowed_fraction", "execution_status", "execution_prefix",
        "coverage_witnesses", "missing_evidence_requests", "limiting_pair",
        "motion_support_revoked") if key in assessment})


def _assess_route(geometry, view, original_xyz, original_q, committed, context, step):
    """Keep whole-goal diagnostics while checking the immutable local frontier."""
    context = {**context, "original_goal_xyz_m": list(original_xyz),
               "original_goal_quaternion_wxyz": list(original_q), "assessment_scope": "whole_goal"}
    whole = geometry.assess_motion(view, original_xyz, original_q, context, step)
    if committed is None or (np.allclose(committed[0], original_xyz, atol=1e-12, rtol=0.)
                             and abs(float(np.dot(committed[1], original_q))) >= 1.-1e-12):
        return whole
    prefix = geometry.assess_motion(view, committed[0], committed[1],
                                   {**context, "assessment_scope": "committed_prefix"}, step)
    result = deepcopy(prefix)
    result.update(status=whole["status"], execution_status=_execution_status(prefix),
                  whole_goal_assessment=deepcopy(whole.get("whole_goal_assessment", _scope_summary(whole))),
                  committed_prefix_assessment=deepcopy(prefix.get("whole_goal_assessment", _scope_summary(prefix))))
    if hasattr(geometry, "last_motion"):
        # The shared summary must retain both scopes too. verify_command uses
        # a preview memory and cannot replace this original-goal assessment.
        geometry.last_motion = deepcopy(result)
    return result


def _risk_limits(assessment):
    # Current residual-clipped increments are not persistent actuator limits.
    # The fallback keeps older geometry test doubles compatible.
    return (float(assessment.get("translation_limit_m", assessment["translation_step_m"])),
            float(assessment.get("rotation_limit_rad", assessment["rotation_step_rad"])))


def _command_limits(assessment, translation_cap, rotation_cap, unknown_limits=None):
    translation = min(translation_cap, float(assessment["translation_step_m"]))
    rotation = min(rotation_cap, float(assessment["rotation_step_rad"]))
    if unknown_limits is not None:
        translation, rotation = min(translation, unknown_limits[0]), min(rotation, unknown_limits[1])
    return translation, rotation


def _pose_action(view, xyz, wxyz, translation_limit, rotation_limit, gripper):
    delta = np.asarray(xyz, float)-view.position
    length = float(np.linalg.norm(delta))
    if length > translation_limit:
        delta *= translation_limit/length
    action = list(ActionAdapter(max_rotation_rad=rotation_limit).adapt(
        delta, [*wxyz[1:], wxyz[0]], gripper, view.state))
    if not any(action[3:6]):
        action[3] = OSC_ORIENTATION_REFRESH_RAD/ROTATION_SCALE_RAD
    return tuple(float(value) for value in action)


def _verify_action(geometry, view, action, context, step, allow_unknown):
    verifier = getattr(geometry, "verify_command", None)
    if verifier is None:
        return None  # Compatibility for isolated controller test doubles.
    return verifier(view, action, context, step, allow_unknown=allow_unknown)


def _observe_geometry_transition(geometry, before, action, after, step, maintained_gripper):
    observer = getattr(geometry, "observe_transition", None)
    if observer is not None:
        gripper = "close" if action[-1] > 0 else "open" if action[-1] < 0 else maintained_gripper
        # Geometry deduplicates this acknowledged step when policy also records
        # it. Contact/release/FK and visual changes must precede the next guard.
        observer(before, after, step, gripper)
    else:
        watcher = getattr(geometry, "watch_observation", None)
        if watcher is not None:
            watcher(after, step)


def _settling_authorization(assessment, step, max_steps):
    reference = assessment.get("gripper_settling_reference")
    if (not isinstance(reference, dict) or not _execution_allowed(assessment)
            or float(assessment["allowed_fraction"]) < 1.-1e-9
            or not assessment.get("final_interaction_guard")
            or assessment.get("missing_evidence_requests")):
        return None
    return {**deepcopy(reference), "entry_step": int(step), "max_steps": int(max_steps)}


def _adaptive_contact_cap(geometry, view, context, target, profile, tracking):
    """Raise only the motion-command cap while estimated contact is distant.

    The contact profile's endpoint tolerance/envelope do not change. Target
    clearance uses padded public hand/object extents; the regular route and
    exact emitted-command checks still decide whether any increment can move.
    """
    base = PROFILES[profile]["action_translation_m"]
    if (profile != "contact" or not context.get("adaptive_contact_speed")
            or tracking.get("status") == "mismatch"):
        return base
    entity = getattr(geometry, "entities", {}).get(context.get("target_id"))
    if (not entity or entity.get("stale") or entity.get("monitoring_unknown")
            or entity.get("attachment") == "held" or entity.get("role") not in ("object", "graspable_part")):
        return base
    from .agent_speedup_v3_geometry import _box, obb_separation
    bodies, payload_unknown = geometry._local_bodies(view, context)
    if payload_unknown:
        return base
    targets = geometry._entity_boxes(entity, view)
    gaps = []
    for body in bodies:
        if body["owner"] != "robot_hand":
            continue
        world = _box(view.position+view.R @ body["center"], view.R @ body["rotation"],
                     body["half"], body["owner"], body["name"], body.get("uncertainty", 0.))
        gaps.extend(obb_separation(world, box, padding_a=world["uncertainty"], padding_b=box["uncertainty"])
                    for box in targets)
    if not gaps:
        return base
    fast = PROFILES["transit"]["action_translation_m"]
    distance = min(min(gaps), float(np.linalg.norm(np.asarray(target)-view.position)))
    # Keep at least two maximum command increments of contact approach reserve.
    # This is a controller-scale margin, not a task-dependent motion distance.
    reserve = 2*fast + 2*float(entity.get("uncertainty_m", 0.))
    fraction = float(np.clip((distance-reserve)/max(reserve, 1e-9), 0., 1.))
    return base+fraction*(fast-base)


class AdaptiveFragmentSequencePrimitive(FragmentSequencePrimitive):
    def __init__(self, view, execution, geometry, motion_context, *, step=0, keep_gripper="open", operation_runtime=None, _continuation=None):
        self.geometry = geometry
        self.operation_runtime = operation_runtime
        self.motion_context = deepcopy(motion_context)
        self.start_step = int(step)
        self._commitment = None
        self._continuation = deepcopy(_continuation)
        super().__init__(view, execution, keep_gripper=keep_gripper)

    def _geometry_context(self):
        context = {**self.motion_context, "commanded_gripper": self.gripper_command,
                   "gripper_action": self._current["skill"] == "set_gripper",
                   "tracking": self._tracking.summary(),
                   "controller_translation_limit_m": self._motion_translation_cap,
                   "controller_rotation_limit_rad": .1,
                   "position_tolerance_m": PROFILES[self._current["request"]["profile"]]["position_tolerance_m"],
                   "orientation_tolerance_rad": PROFILES[self._current["request"]["profile"]]["rotation_tolerance_rad"],
                   "gripper_settling_authorization": deepcopy(self._gripper_settling_authorization)}
        if self._current["skill"] == "set_gripper":
            context["commanded_gripper"] = self._current["request"]["gripper"]
        if self.operation_runtime is not None and hasattr(self.operation_runtime, "geometry_context"):
            context = self.operation_runtime.geometry_context(context, self.start_step+self.actions_executed,
                max(0, self._watchdog.limit-self._current["actions"]),
                allow_grasp_probe=self._current["skill"] in MOTION_SKILLS)
        return context

    def _assessment(self, view):
        profile = self._current["request"]["profile"]
        self._motion_translation_cap = (_adaptive_contact_cap(self.geometry, view, self.motion_context,
            self._target_xyz, profile, self._tracking.summary()) if self._current["skill"] in MOTION_SKILLS
            else PROFILES[profile]["action_translation_m"])
        c = self._commitment
        committed = (c["target_xyz_m"], c["target_quaternion_wxyz"]) if c else None
        return _assess_route(self.geometry, view, self._target_xyz, self._target_wxyz,
                             committed, self._geometry_context(), self.start_step+self.actions_executed)

    def _enter_segment(self, view):
        self._commitment = None
        self._prepared_action = None
        self._tracking = _TrackingEvidence()
        self._gripper_settling_authorization = None
        super()._enter_segment(view)
        saved = self._continuation if self._segment_index == 0 else None
        if saved:
            self._tracking.restore(saved["tracking_state"],
                                   same_measurement=np.allclose(view.state, saved["measured_state"], atol=1e-9, rtol=0.))
        self._chunk_steps = int(saved["chunk_steps"] if saved else self._current["request"]["max_steps"])
        self._prior_segment_steps = int(saved["consumed_steps"] if saved else 0)
        self._watchdog = _IntentWatchdog(saved["remaining_watchdog_steps"] if saved else self._chunk_steps,
                                        learned=saved["watchdog_learned"] if saved else False)
        self._local_continuations = 0
        if self.done or self._current["skill"] == "wait":
            return
        assessment = self._assessment(view)
        if self._current["skill"] == "set_gripper":
            self._gripper_settling_authorization = _settling_authorization(
                assessment, self.start_step + self.actions_executed, self._current["request"]["max_steps"])
        fraction = float(assessment["allowed_fraction"])
        xyz, q = _assessed_prefix(view, self._target_xyz, self._target_wxyz, assessment)
        motion = self._current["skill"] in MOTION_SKILLS
        # Gap settling still needs its original minimum and stability predicate.
        horizon = self._current["request"]["max_steps"]
        if motion:
            horizon = _bounded_deadline(horizon, 0, assessment["suggested_steps"])
        self._commitment = {"target_xyz_m": xyz.tolist(), "target_quaternion_wxyz": q,
                            "horizon_steps": horizon, "entry_fragment_step": self.actions_executed,
                            "geometry_horizon_steps": None if assessment["suggested_steps"] is None else horizon,
                            "original_goal_truncated": bool(motion and fraction < 1.-1e-9),
                            "controller_translation_limit_m": self._motion_translation_cap,
                            "controller_rotation_limit_rad": .1,
                            "unknown_limits": _risk_limits(assessment) if assessment["suggested_steps"] is not None else None,
                            "translation_step_m": min(self._motion_translation_cap, float(assessment["translation_step_m"])),
                            "rotation_step_rad": min(.1, float(assessment["rotation_step_rad"])),
                            "tracking": self._tracking.summary(),
                            "initial_assessment": assessment, "last_assessment": assessment}
        self._current["geometry_commitment"] = self._commitment
        if _execution_status(assessment) == "blocked" and fraction <= 1e-9:
            self._pause("geometry_predicted_block")
        else:
            self._prepare_action(view)

    def _pause(self, reason):
        self._current.update(status="paused", reason=reason, end_at_fragment_step=self.actions_executed)
        self._end("succeeded", reason, finalize_current=False,
                  detail="Observation handoff only. Original target remains pending; no arrival, contact or task success is claimed.")

    def _end(self, status, reason, *, detail=None, finalize_current=True):
        # v2's global 240-step cap is replaced by per-goal finite estimates and
        # measured no-progress checks; renewing a chunk never resets either.
        if reason == "execution_budget_exhausted":
            return
        if reason == "segment_budget_exhausted" and getattr(self, "_watchdog", None) is not None:
            if self._watchdog.learned and self._current["actions"] >= self._watchdog.limit:
                reason = "intent_watchdog_exhausted"
        return super()._end(status, reason, detail=detail, finalize_current=finalize_current)

    def _renew_chunk(self, view, assessment):
        if self._current["skill"] not in MOTION_SKILLS:
            return
        c = self._commitment
        elapsed = self._current["actions"] + 1
        distance, angle = self._errors(view)
        self._watchdog.learn_once(elapsed, distance, angle, self._tracking)
        eligible = (_execution_allowed(assessment) and assessment["suggested_steps"] is None
                    and c["geometry_horizon_steps"] is None
                    and self._tracking.evidence["status"] == "normal"
                    and self._tracking.progress_per_step() > STALL_PROGRESS_M/STALL_WINDOW_STEPS)
        deadline = self._current["request"]["max_steps"]
        if elapsed >= deadline and eligible and elapsed < self._watchdog.limit:
            new_deadline = min(deadline + self._chunk_steps, self._watchdog.limit)
            self._current["request"]["max_steps"] = new_deadline
            c["horizon_steps"] = new_deadline
            self._local_continuations += 1

    def _check_progress_watchdog(self):
        if self._current["skill"] not in MOTION_SKILLS or self._current["actions"] < STALL_MIN_STEPS:
            return
        samples = self._current["measurements"][-STALL_WINDOW_STEPS:]
        errors = [s["position_error_m"] + .05*s["orientation_error_rad"] for s in samples]
        if errors[0] - min(errors[1:]) < STALL_PROGRESS_M:
            self._end("failed", "segment_stall", detail="measured goal progress stopped; no local chunk renewal")

    def _next(self, view):
        if self._commitment is None:
            action = super()._next(view)
        else:
            if not np.array_equal(view.state, self._last_state):
                raise ValueError("fragment action state differs from the latest acknowledged measurement")
            if self._current["skill"] == "set_gripper":
                self.gripper_command = self._current["request"]["gripper"]
            action = self._prepared_action
        if self.operation_runtime is not None:
            command = "close" if action[-1] > 0 else "open" if action[-1] < 0 else self.gripper_command
            self.operation_runtime.before_action(view, self.motion_context, command, self.start_step+self.actions_executed)
        return action

    def _prepare_action(self, view):
        c = self._commitment
        c["controller_translation_limit_m"] = self._motion_translation_cap
        c["translation_step_m"], c["rotation_step_rad"] = _command_limits(
            c["last_assessment"], c["controller_translation_limit_m"], .1, c["unknown_limits"])
        action = _pose_action(view, c["target_xyz_m"], c["target_quaternion_wxyz"],
                              c["translation_step_m"], c["rotation_step_rad"],
                              self._geometry_context()["commanded_gripper"])
        context = {**self._geometry_context(), "original_goal_xyz_m": self._target_xyz.tolist(),
                   "original_goal_quaternion_wxyz": list(self._target_wxyz),
                   "assessment_scope": "emitted_command"}
        verification = _verify_action(self.geometry, view, action, context,
            self.start_step+self.actions_executed, c["geometry_horizon_steps"] is not None)
        c["current_command"] = list(action)
        c["command_verification"] = verification
        if verification is not None and not verification["command_allowed"]:
            self._pause("geometry_predicted_block")
        else:
            self._prepared_action = action

    def _observe(self, before, action, after):
        # Primitive.observe_transition has already validated the acknowledgement.
        _observe_geometry_transition(self.geometry, before, action, after,
                                     self.start_step + self.actions_executed, self.gripper_command)
        if self.operation_runtime is not None:
            command = "close" if action[-1] > 0 else "open" if action[-1] < 0 else self.gripper_command
            self.operation_runtime.observe_transition(before, after, self.motion_context, command,
                self.start_step+self.actions_executed)
        previous = self._current
        assessment = None
        if self._commitment is not None:
            c = self._commitment
            if previous["skill"] in MOTION_SKILLS:
                profile = PROFILES[previous["request"]["profile"]]
                self._tracking.observe(before, action, after, self._target_xyz, self._target_wxyz,
                    position_tolerance_m=profile["position_tolerance_m"],
                    rotation_tolerance_rad=profile["rotation_tolerance_rad"])
            c["tracking"] = self._tracking.summary()
            # Assess the acknowledged step before base completion can enter a
            # following segment. A mismatch can invalidate its shared evidence.
            assessment = self._assessment(after)
            c["last_assessment"] = assessment
            self._renew_chunk(after, assessment)
        super()._observe(before, action, after)
        if self.done or self._current is not previous or self._commitment is None:
            return {"fragment": self.progress()}
        c = self._commitment
        self._check_progress_watchdog()
        if self.done:
            return {"fragment": self.progress()}
        fraction = float(assessment["allowed_fraction"])
        if _execution_status(assessment) == "blocked" and fraction <= 1e-9:
            self._pause("geometry_predicted_block")
        elif self._current["skill"] in MOTION_SKILLS:
            if fraction < 1.-1e-9:
                xyz, q = _assessed_prefix(after, c["target_xyz_m"], c["target_quaternion_wxyz"], assessment)
                c.update(target_xyz_m=xyz.tolist(), target_quaternion_wxyz=q, original_goal_truncated=True)
            c["horizon_steps"] = _bounded_deadline(
                c["horizon_steps"], self._current["actions"], assessment["suggested_steps"])
            if assessment["suggested_steps"] is not None:
                if c["unknown_limits"] is None:
                    c["unknown_limits"] = _risk_limits(assessment)
                c["geometry_horizon_steps"] = _bounded_deadline(
                    c["geometry_horizon_steps"] if c["geometry_horizon_steps"] is not None else self._watchdog.limit,
                    self._current["actions"], assessment["suggested_steps"])
            profile = PROFILES[self._current["request"]["profile"]]
            c["translation_step_m"], c["rotation_step_rad"] = _command_limits(
                assessment, self._motion_translation_cap, .1, c["unknown_limits"])
            prefix_reached = (np.linalg.norm(np.asarray(c["target_xyz_m"])-after.position) <= profile["position_tolerance_m"]
                              and np.linalg.norm(rotation_error_rad(c["target_quaternion_wxyz"], after.state[3:6])) <= profile["rotation_tolerance_rad"])
            if c["original_goal_truncated"] and prefix_reached:
                self._pause("geometry_prefix_observed")
            elif self._current["actions"] >= c["horizon_steps"]:
                self._pause("geometry_observation_horizon")
        if not self.done:
            self._prepare_action(after)
        return {"fragment": self.progress()}

    def progress(self):
        result = super().progress()
        result.update(version=FRAGMENT_VERSION, geometry_commitment=deepcopy(self._commitment),
                      goal_reached=self._current["status"] == "succeeded")
        if hasattr(self, "_watchdog"):
            result.update(local_chunk_steps=self._chunk_steps, local_continuations=self._local_continuations,
                          intent_watchdog_steps=self._watchdog.limit,
                          remaining_watchdog_steps=max(0, self._watchdog.limit-self._current["actions"]))
        if self._current["status"] == "paused":
            start = self._segment_index
            result.update(status="observation_required", sequence_complete=False,
                          handoff_kind="geometry_observation_checkpoint", observation_requested=True,
                          pending_segment_indices=list(range(start, len(self.execution["segments"]))),
                          pending_segments=deepcopy(self.execution["segments"][start:]),
                          remaining_current_goal={"xyz_m": self._target_xyz.tolist(),
                                                  "quaternion_wxyz": list(self._target_wxyz),
                                                  "frame": "world"})
        return result

    def export_continuation(self):
        if not self.done or self._current["status"] != "paused":
            return None
        remaining = self._watchdog.limit-self._current["actions"]
        if remaining <= 0:
            return None
        pending = deepcopy(self.execution["segments"][self._segment_index:])
        chunk_left = max(1, min(self._chunk_steps, self._current["request"]["max_steps"]-self._current["actions"], remaining))
        pending[0]["max_steps"] = chunk_left
        if pending[0]["skill"] in MOTION_SKILLS:
            pending[0].update(skill="move_to_pose", xyz_m=self._target_xyz.tolist(),
                              quaternion_wxyz=list(self._target_wxyz))
        return {"kind": "fragment", "motion_context": deepcopy(self.motion_context),
                "gripper_command": self.gripper_command, "pending_segments": pending,
                "current_target": {"xyz_m": self._target_xyz.tolist(), "quaternion_wxyz": list(self._target_wxyz)},
                "chunk_steps": self._chunk_steps, "remaining_chunk_steps": chunk_left,
                "consumed_steps": self._prior_segment_steps+self._current["actions"],
                "remaining_watchdog_steps": remaining, "watchdog_learned": self._watchdog.learned,
                "tracking_state": self._tracking.snapshot(), "measured_state": self._last_state.tolist()}

    @classmethod
    def from_continuation(cls, view, saved, geometry, motion_context, *, step=0, keep_gripper=None, operation_runtime=None):
        _validate_continuation(saved, "fragment", motion_context, keep_gripper)
        first = saved["pending_segments"][0]
        if first["max_steps"] != saved["remaining_chunk_steps"]:
            raise ValueError("continuation segment budget differs from its committed remainder")
        if first["skill"] in MOTION_SKILLS and (first["skill"] != "move_to_pose"
                or first["xyz_m"] != saved["current_target"]["xyz_m"]
                or first["quaternion_wxyz"] != saved["current_target"]["quaternion_wxyz"]):
            raise ValueError("continuation must retain the original absolute target")
        return cls(view, {"segments": deepcopy(saved["pending_segments"])}, geometry, motion_context,
                   step=step, keep_gripper=saved["gripper_command"], operation_runtime=operation_runtime, _continuation=saved)


class AdaptiveLocalMotionPrimitive(LocalMotionPrimitive):
    """Preserve direct target validation, but execute only its geometric prefix."""
    def __init__(self, view, motion, geometry, motion_context, *, step=0, keep_gripper="open", operation_runtime=None, _continuation=None):
        self.geometry, self.motion_context, self.start_step = geometry, deepcopy(motion_context), int(step)
        self.operation_runtime = operation_runtime
        self.assessment = None
        self._tracking = _TrackingEvidence()
        if _continuation:
            self._tracking.restore(_continuation["tracking_state"],
                                   same_measurement=np.allclose(view.state, _continuation["measured_state"], atol=1e-9, rtol=0.))
        self._unknown_limits = None
        self._prepared_action = None
        self._command_verification = None
        self._gripper_settling_authorization = None
        self._chunk_steps = int(_continuation["chunk_steps"] if _continuation else motion["steps"])
        self._prior_segment_steps = int(_continuation["consumed_steps"] if _continuation else 0)
        self._watchdog = _IntentWatchdog(_continuation["remaining_watchdog_steps"] if _continuation else motion["steps"],
                                        learned=_continuation["watchdog_learned"] if _continuation else False)
        self._local_continuations = 0
        self._geometry_deadline = None
        self._original_goal_truncated = False
        self._arrival_count = 0
        super().__init__(view, motion, keep_gripper=keep_gripper)
        if self.done:
            return
        self.original_target = list(self.executor.target)
        # An explicit observation is a hold, not a scene-space continuation.
        self.observation_hold = (motion_context.get("intent") == "observe" and
                                 np.linalg.norm(np.asarray(self.original_target[:3])-view.position) < 1e-9 and
                                 np.linalg.norm(rotation_error_rad(self.original_target[3:7], view.state[3:6])) < 1e-9 and
                                 motion["gripper"] == "keep")
        if self.observation_hold:
            return
        self.assessment = self._assess(view)
        if (motion["gripper"] != "keep"
                and np.linalg.norm(np.asarray(self.original_target[:3])-view.position) < 1e-9
                and np.linalg.norm(rotation_error_rad(self.original_target[3:7], view.state[3:6])) < 1e-9):
            self._gripper_settling_authorization = _settling_authorization(
                self.assessment, self.start_step, motion["steps"])
        fraction = float(self.assessment["allowed_fraction"])
        self._original_goal_truncated = fraction < 1.-1e-9
        if self.assessment["suggested_steps"] is not None:
            self._geometry_deadline = int(self.assessment["suggested_steps"])
            self._unknown_limits = _risk_limits(self.assessment)
        if _execution_status(self.assessment) == "blocked" and fraction <= 1e-9:
            self._finish("failed", "geometry_predicted_block", self._evidence_at(view))
            return
        xyz, q = _assessed_prefix(view, self.original_target[:3], self.original_target[3:7], self.assessment)
        config = ShortTargetConfig(max_step_translation_m=min(.02, self.assessment["translation_step_m"]),
                                   max_step_rotation_rad=min(.1, self.assessment["rotation_step_rad"]))
        self.executor = ShortTargetExecutor(view.state, [*xyz, *q, self.original_target[7]],
            _bounded_deadline(motion["steps"], 0, self.assessment["suggested_steps"]), config=config)
        self._prepare_action(view)

    def _geometry_context(self):
        context = {**self.motion_context, "commanded_gripper": self.gripper_command,
             "gripper_action": self.motion["gripper"] != "keep",
             "tracking": self._tracking.summary(),
             "controller_translation_limit_m": .02, "controller_rotation_limit_rad": .1,
             "position_tolerance_m": .004, "orientation_tolerance_rad": .06,
             "gripper_settling_authorization": deepcopy(self._gripper_settling_authorization)}
        if self.operation_runtime is not None and hasattr(self.operation_runtime, "geometry_context"):
            context = self.operation_runtime.geometry_context(context, self.start_step+self.actions_executed,
                max(0, self._watchdog.limit-self.executor.executed_steps),
                allow_grasp_probe=self.motion["gripper"] == "keep")
        return context

    def _assess(self, view):
        return _assess_route(self.geometry, view, self.original_target[:3], self.original_target[3:7],
            (self.executor.target[:3], self.executor.target[3:7]), self._geometry_context(),
            self.start_step+self.actions_executed)

    def _prepare_action(self, view):
        action = _pose_action(view, self.executor.target[:3], self.executor.target[3:7],
            self.executor.config.max_step_translation_m, self.executor.config.max_step_rotation_rad,
            self.gripper_command)
        context = {**self._geometry_context(), "original_goal_xyz_m": self.original_target[:3],
                   "original_goal_quaternion_wxyz": self.original_target[3:7],
                   "assessment_scope": "emitted_command"}
        self._command_verification = _verify_action(self.geometry, view, action, context,
            self.start_step+self.actions_executed, self._geometry_deadline is not None)
        if self._command_verification is not None and not self._command_verification["command_allowed"]:
            self.executor.interrupt("geometry_predicted_block; emitted coupled command was not authorized")
            self._finish("succeeded", "geometry_predicted_block", self._evidence_at(view))
        else:
            self._prepared_action = action

    def _next(self, view):
        action = super()._next(view)
        if not self.observation_hold and not np.allclose(action, self._prepared_action, atol=1e-12, rtol=0.):
            raise RuntimeError("direct action differs from the coupled command checked at the acknowledged state")
        if self.operation_runtime is not None:
            command = "close" if action[-1] > 0 else "open" if action[-1] < 0 else self.gripper_command
            self.operation_runtime.before_action(view, self.motion_context, command, self.start_step+self.actions_executed)
        return action

    def _evidence_at(self, view):
        return {**self.executor.feedback(view.state), "original_requested_target": self.original_target,
                "geometry_assessment": deepcopy(self.assessment),
                "current_command": list(self._prepared_action) if self._prepared_action is not None else None,
                "command_verification": deepcopy(self._command_verification),
                "goal_reached": self.motion["gripper"] == "keep" and not self.observation_hold and self._arrival_count >= 2,
                "tracking": self._tracking.summary(),
                "local_chunk_steps": self._chunk_steps, "local_continuations": self._local_continuations,
                "remaining_watchdog_steps": max(0, self._watchdog.limit-self.executor.executed_steps),
                "automatic_resume": False, "physical_clearance_verified": False}

    def _observe(self, before, action, after):
        self.executor.observe_transition(before.state, action, after.state)
        _observe_geometry_transition(self.geometry, before, action, after,
                                     self.start_step + self.actions_executed, self.gripper_command)
        if self.operation_runtime is not None:
            command = "close" if action[-1] > 0 else "open" if action[-1] < 0 else self.gripper_command
            self.operation_runtime.observe_transition(before, after, self.motion_context, command,
                self.start_step+self.actions_executed)
        if not self.observation_hold:
            self._tracking.observe(before, action, after, self.original_target[:3], self.original_target[3:7])
            self.assessment = self._assess(after)
            fraction = float(self.assessment["allowed_fraction"])
            distance = float(np.linalg.norm(np.asarray(self.original_target[:3])-after.position))
            angle = float(np.linalg.norm(rotation_error_rad(self.original_target[3:7], after.state[3:6])))
            self._arrival_count = self._arrival_count+1 if distance <= .004 and angle <= .06 else 0
            if self.motion["gripper"] == "keep":
                self._watchdog.learn_once(self.executor.executed_steps, distance, angle, self._tracking)
            self._original_goal_truncated |= fraction < 1.-1e-9
            if self.assessment["suggested_steps"] is not None:
                if self._unknown_limits is None:
                    self._unknown_limits = _risk_limits(self.assessment)
                self._geometry_deadline = _bounded_deadline(
                    self._geometry_deadline if self._geometry_deadline is not None else self._watchdog.limit,
                    self.executor.executed_steps, self.assessment["suggested_steps"])
            translation, rotation = _command_limits(self.assessment, .02, .1, self._unknown_limits)
            self.executor.config = ShortTargetConfig(
                max_step_translation_m=translation, max_step_rotation_rad=rotation)
            self.executor.adapter = ActionAdapter(max_rotation_rad=self.executor.config.max_step_rotation_rad)
            self.executor.requested_steps = _bounded_deadline(
                self.executor.requested_steps, self.executor.executed_steps,
                self.assessment["suggested_steps"])
            if fraction < 1.-1e-9:
                xyz, q = _assessed_prefix(after, self.executor.target[:3], self.executor.target[3:7], self.assessment)
                self.executor.target = tuple([*xyz, *q, self.executor.target[7]])
            prefix_reached = (self._original_goal_truncated
                and np.linalg.norm(np.asarray(self.executor.target[:3])-after.position) <= .004
                and np.linalg.norm(rotation_error_rad(self.executor.target[3:7], after.state[3:6])) <= .06)
            if _execution_status(self.assessment) == "blocked" and fraction <= 1e-9:
                self.executor.interrupt("geometry_predicted_block; request new observation")
            elif prefix_reached:
                self.executor.interrupt("geometry_prefix_observed; original goal remains pending")
            elif self.executor.executed_steps >= self.executor.requested_steps and not self.executor.done:
                self.executor.interrupt("geometry_observation_horizon; request new observation")
            eligible = (self.motion["gripper"] == "keep" and _execution_allowed(self.assessment)
                        and self._geometry_deadline is None
                        and self._tracking.evidence["status"] == "normal"
                        and self._tracking.progress_per_step() > STALL_PROGRESS_M/STALL_WINDOW_STEPS)
            if (self.executor.done and self.executor.status == "segment_complete" and eligible
                    and self._arrival_count < 2 and self.executor.executed_steps < self._watchdog.limit):
                self.executor.requested_steps = min(self.executor.requested_steps+self._chunk_steps, self._watchdog.limit)
                self.executor.done = False
                self.executor.status = "running"
                self.executor.detail = "same supported goal continues in another bounded local chunk"
                self._local_continuations += 1
            if not self.executor.done:
                self._prepare_action(after)
        if self.executor.done:
            reason = (self.executor.detail.split(";")[0] if self.executor.status == "interrupted"
                      else "adaptive_local_segment_complete")
            if self.executor.status != "interrupted" and self._geometry_deadline is not None and self._arrival_count < 2:
                reason = "geometry_observation_horizon"
            elif self._watchdog.learned and self.executor.executed_steps >= self._watchdog.limit and self._arrival_count < 2:
                reason = "intent_watchdog_exhausted"
            self._finish("succeeded", reason, self._evidence_at(after))
        return {"local_motion": self._evidence_at(after)}

    def export_continuation(self):
        if not self.done or self.reason not in ("geometry_predicted_block", "geometry_observation_horizon", "geometry_prefix_observed"):
            return None
        remaining = self._watchdog.limit-self.executor.executed_steps
        if remaining <= 0:
            return None
        chunk_left = min(self._chunk_steps, max(1, self.executor.requested_steps-self.executor.executed_steps), remaining)
        motion = deepcopy(self.motion)
        motion.update(position_mode="absolute", xyz_m=self.original_target[:3],
                      quaternion_wxyz=self.original_target[3:7], steps=chunk_left)
        return {"kind": "motion", "motion_context": deepcopy(self.motion_context),
                "gripper_command": self.gripper_command, "motion": motion,
                "current_target": {"xyz_m": self.original_target[:3], "quaternion_wxyz": self.original_target[3:7]},
                "chunk_steps": self._chunk_steps, "remaining_chunk_steps": chunk_left,
                "consumed_steps": self._prior_segment_steps+self.executor.executed_steps,
                "remaining_watchdog_steps": remaining, "watchdog_learned": self._watchdog.learned,
                "tracking_state": self._tracking.snapshot(), "measured_state": self.executor.expected_state.tolist()}

    @classmethod
    def from_continuation(cls, view, saved, geometry, motion_context, *, step=0, keep_gripper=None, operation_runtime=None):
        _validate_continuation(saved, "motion", motion_context, keep_gripper)
        motion = saved["motion"]
        if (motion["steps"] != saved["remaining_chunk_steps"] or motion["position_mode"] != "absolute"
                or motion["xyz_m"] != saved["current_target"]["xyz_m"]
                or motion["quaternion_wxyz"] != saved["current_target"]["quaternion_wxyz"]):
            raise ValueError("continuation must retain the original absolute target and chunk remainder")
        return cls(view, deepcopy(saved["motion"]), geometry, motion_context, step=step,
                   keep_gripper=saved["gripper_command"], operation_runtime=operation_runtime, _continuation=saved)
