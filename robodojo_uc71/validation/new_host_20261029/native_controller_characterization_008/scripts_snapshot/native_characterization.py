"""Fixed, separately authorized robot-only native characterization; no launcher.

Thirty-seven actions are planned from recorded initial arm joints: one hold,
then +0.005, -0.005 and baseline for each joint of each arm. Both grippers remain
open. Measurements never alter a target or trigger a retry. This is not a
benchmark attempt, EpisodeLifecycle, controller calibration, or success test.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np

from .native_action_observer import observe_native_joint_action_once

ARMS = ("left", "right")
DELTA_RAD = .005
MAX_ACTIONS = 37


class CharacterizationError(RuntimeError):
    pass


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recorded_arms(inventory):
    if inventory.get("schema") != "robodojo-native-robot-inventory-1" or inventory.get("num_envs") != 1:
        raise CharacterizationError("native_initial_inventory_required")
    result = {}
    for robot in inventory["robots"]:
        arm = {"left_arm": "left", "right_arm": "right"}.get(robot["arm_name"])
        names = robot["joint_names"]
        ids = robot["native_arm_joint_indices"]
        finger_ids = robot["native_gripper_joint_indices"]
        q = np.asarray(robot["joint_positions"]["values"], dtype=float)
        limits = np.asarray(robot["joint_position_limits_lower_upper"]["values"], dtype=float)
        declared = robot["declared_metadata"]
        scale = declared["gripper_scale"]["values"]
        if (arm is None or arm in result or len(names) != 8 or len(set(names)) != 8
                or len(ids) != 6 or len(finger_ids) != 2 or set(ids + finger_ids) != set(range(8))
                or [names[i] for i in ids] != [f"joint{i}" for i in range(1, 7)]
                or [names[i] for i in finger_ids] != ["joint7", "joint8"]
                or q.shape != (1, 8) or limits.shape != (1, 8, 2)
                or not np.isfinite(q).all() or not np.isfinite(limits).all()
                or declared["gripper_move"] != {"base": "joint7", "sign": 1.0, "mimic": ["joint8", 1.0, 0.0]}
                or scale != [-.01, .044] or np.max(np.abs(q[0, finger_ids] - .044)) > 1e-4):
            raise CharacterizationError("unsupported_or_nonopen_initial_robot")
        arm_q, arm_limits = q[0, ids], limits[0, ids]
        if np.any(arm_q - DELTA_RAD <= arm_limits[:, 0]) or np.any(arm_q + DELTA_RAD >= arm_limits[:, 1]):
            raise CharacterizationError("probe_would_exceed_recorded_joint_limits")
        result[arm] = {"joint_positions": arm_q.tolist(), "finger_positions": q[0, finger_ids].tolist(),
                       "joint_limits": arm_limits.tolist(), "joint_names": [names[i] for i in ids]}
    if set(result) != set(ARMS) or len(inventory["robots"]) != 2:
        raise CharacterizationError("exact_dual_arm_inventory_required")
    return result


def build_probe_plan(initial_inventory_path, geometry_candidate_path, preparation_path):
    paths = [Path(p).resolve(strict=True) for p in (initial_inventory_path, geometry_candidate_path, preparation_path)]
    inventory, geometry, preparation = [json.loads(p.read_text()) for p in paths]
    if (geometry.get("schema") != "robodojo-physical-tool-candidate-1"
            or geometry.get("calibration_status") != "uncalibrated_candidate"
            or geometry.get("model_input_eligible") is not False or geometry.get("full_test_authorized") is not False
            or preparation.get("full_test_authorized") is not False
            or preparation.get("binding", {}).get("kit_index") != 0):
        raise CharacterizationError("bound_uncalibrated_geometry_and_preparation_required")
    if not geometry.get("source_files") or any(file_hash(row["path"]) != row["sha256"] for row in geometry["source_files"]):
        raise CharacterizationError("physical_tool_candidate_source_changed")
    initial = _recorded_arms(inventory)
    baseline = {f"{arm}_arm_joint_state": initial[arm]["joint_positions"] for arm in ARMS}
    baseline.update({f"{arm}_ee_joint_state": [1.] for arm in ARMS})
    actions = [{"index": 1, "probe": "baseline_hold", "action": deepcopy(baseline)}]
    for arm in ARMS:
        for joint in range(6):
            for label, delta in (("positive", DELTA_RAD), ("negative", -DELTA_RAD), ("return", 0.)):
                action = deepcopy(baseline)
                action[f"{arm}_arm_joint_state"][joint] += delta
                actions.append({"index": len(actions) + 1, "probe": f"{arm}_joint{joint+1}_{label}", "action": action})
    return {"schema": "robodojo-native-characterization-plan-1", "scope": "diagnostic_native_actions",
        "initial": initial, "actions": actions, "maximum_native_actions": MAX_ACTIONS,
        "perturbation_rad": DELTA_RAD, "physics_dt_s": .004, "substeps_per_native_action": 10,
        "maximum_simulated_time_s": 1.48, "binding": preparation["binding"],
        "layout_path": preparation["layout_path"], "layout_sha256": preparation["layout_sha256"],
        "initial_inventory_path": str(paths[0]), "geometry_candidate_path": str(paths[1]),
        "preparation_path": str(paths[2]),
        "source_sha256": {str(p): file_hash(p) for p in paths}, "geometry_applied_to_targets": False,
        "initial_joint_tolerance_rad": 2e-4, "measured_joint_envelope_rad": .02,
        "measured_velocity_limit_rad_s": 1., "gripper_commands": {"left": 1., "right": 1.},
        "benchmark_attempt_started": False, "full_test_authorized": False, "model_calls": 0,
        "task_success_assessed": False, "standard_ready": False, "calibration_status": "uncalibrated_candidate",
        "automatic_retry": False, "gripper_timing_actions_included": False,
        "limitations": ["Fixed transient probes do not establish settled tracking or full controller calibration",
            "Offline FK/Jacobian validation must use actual measured joints, never commanded joints as ground truth",
            "No object pose, contact, reward targeting, grasp success, or inner-pad clearance inference",
            "Empty-gripper close/open timing requires a separate bounded diagnostic scope"]}


def validate_probe_plan(plan):
    if (plan.get("schema") != "robodojo-native-characterization-plan-1"
            or plan.get("scope") != "diagnostic_native_actions" or plan.get("maximum_native_actions") != MAX_ACTIONS
            or plan.get("perturbation_rad") != DELTA_RAD or plan.get("benchmark_attempt_started") is not False
            or plan.get("full_test_authorized") is not False or plan.get("model_calls") != 0
            or plan.get("automatic_retry") is not False or len(plan.get("actions", [])) != MAX_ACTIONS):
        raise CharacterizationError("invalid_fixed_characterization_plan")
    rebuilt = build_probe_plan(plan["initial_inventory_path"], plan["geometry_candidate_path"], plan["preparation_path"])
    if plan != rebuilt:
        raise CharacterizationError("plan_differs_from_recorded_robot_sources")
    baseline = {f"{arm}_arm_joint_state": plan["initial"][arm]["joint_positions"] for arm in ARMS}
    baseline.update({f"{arm}_ee_joint_state": [1.] for arm in ARMS})
    expected = [deepcopy(baseline)]
    for arm in ARMS:
        for joint in range(6):
            for delta in (DELTA_RAD, -DELTA_RAD, 0.):
                target = deepcopy(baseline); target[f"{arm}_arm_joint_state"][joint] += delta
                expected.append(target)
    if any(row.get("index") != index or row.get("action") != action
           for index, (row, action) in enumerate(zip(plan["actions"], expected), 1)):
        raise CharacterizationError("fixed_targets_or_order_changed")
    for path, expected_hash in plan["source_sha256"].items():
        if file_hash(path) != expected_hash:
            raise CharacterizationError("plan_source_changed")
    if file_hash(plan["layout_path"]) != plan["layout_sha256"]:
        raise CharacterizationError("layout_binding_changed")


def authorization_template(plan):
    return {"schema": "robodojo-native-characterization-authorization-1", "plan_sha256": content_hash(plan),
        "gpu_uuid": plan["binding"]["gpu_uuid"], "maximum_native_actions": MAX_ACTIONS,
        "diagnostic_native_actions_authorized": False, "benchmark_attempt_started": False,
        "full_test_authorized": False, "model_calls_authorized": False, "automatic_retry": False,
        "gripper_timing_actions_authorized": False, "user_message": "",
        "authorization_basis": "Execution scope must be recorded separately; this template grants no execution permission."}


def verify_diagnostic_authorization(path, plan, index):
    value = json.loads(Path(path).read_text())
    expected = authorization_template(plan)
    expected.pop("user_message"); expected.pop("authorization_basis")
    expected["diagnostic_native_actions_authorized"] = True
    if (not 1 <= index <= MAX_ACTIONS or not value.get("user_message") or not value.get("authorization_basis")
            or any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in expected.items())):
        raise CharacterizationError("separate_exact_diagnostic_authorization_required")
    return file_hash(path)


def capture_characterization_state(env):
    """Read only initialized robot getters, retaining raw and COM-corrected data."""
    from .native_robot_inventory import capture_robot_inventory
    before = float(env.sim.sim.current_time)
    counters_before = (env.sim._sim_step_counter, tuple(env.take_action_cnt))
    inventory = capture_robot_inventory(env)
    arms = {}
    for robot in inventory["robots"]:
        arm = {"left_arm": "left", "right_arm": "right"}[robot["arm_name"]]
        ids = robot["native_arm_joint_indices"]
        body = robot["body_names"].index("link6")
        pose = np.asarray(robot["body_link_pose_w"]["values"], float)[0, body]
        com = np.asarray(robot["body_com_pos_b"]["values"], float)[0, body]
        raw = np.asarray(robot["root_physx_view_jacobians"]["values"], float)
        if not robot["is_fixed_base"] or body < 1 or raw.shape != (1, len(robot["body_names"])-1, 6, 8):
            raise CharacterizationError("unsupported_raw_jacobian_layout")
        quat = pose[3:]; norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or abs(norm-1) > 1e-3:
            raise CharacterizationError("invalid_link_orientation")
        w, x, y, z = quat / norm
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        r = rotation @ com; x, y, z = r
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        jacobian = raw[0, body-1][:, ids].copy()
        jacobian[:3] += skew @ jacobian[3:]
        arms[arm] = {"joint_positions": np.asarray(robot["joint_positions"]["values"])[0, ids].tolist(),
            "joint_velocities": np.asarray(robot["joint_velocities"]["values"])[0, ids].tolist(),
            "finger_positions": np.asarray(robot["joint_positions"]["values"])[0, robot["native_gripper_joint_indices"]].tolist(),
            "link6_world_pose_wxyz": pose.tolist(), "link6_actor_world_jacobian": jacobian.tolist(),
            "jacobian_note": "J_actor_linear = J_COM_linear + skew(R_world_link6 * COM_local) * J_angular"}
    if (before != float(env.sim.sim.current_time) or set(arms) != set(ARMS)
            or counters_before != (env.sim._sim_step_counter, tuple(env.take_action_cnt))):
        raise CharacterizationError("robot_capture_advanced_physics_or_missing_arm")
    return {"schema": "robodojo-native-characterization-measurement-1", "scope": "diagnostic_native_actions",
        "native_physics_time_s": before, "native_physics_counter": counters_before[0],
        "native_action_counter": list(counters_before[1]), "benchmark_attempt_started": False,
        "full_test_authorized": False, "model_calls": 0,
        "arms": arms, "raw_robot_inventory": inventory, "calibration_status": "uncalibrated_candidate",
        "model_input_eligible": False, "standard_ready": False}


class FixedRobotCharacterization:
    """One irreversible bounded diagnostic sequence; no automatic dispatch loop."""
    def __init__(self, env, plan, *, authorization_path, verify_runtime_binding):
        validate_probe_plan(plan)
        if not callable(verify_runtime_binding) or env.take_action_cnt != [0]:
            raise CharacterizationError("fresh_reset_and_runtime_binding_verifier_required")
        self.env, self.plan = env, deepcopy(plan)
        self._plan_hash = content_hash(self.plan)
        self._authorization = Path(authorization_path).resolve(strict=True)
        self._verify_runtime = verify_runtime_binding
        self._next, self._poisoned, self._stopped = 1, False, False
        self._lock = Lock()

    def _measurement(self, initial=False):
        value = capture_characterization_state(self.env)
        for arm in ARMS:
            actual = np.asarray(value["arms"][arm]["joint_positions"])
            baseline = np.asarray(self.plan["initial"][arm]["joint_positions"])
            bound = self.plan["initial_joint_tolerance_rad"] if initial else self.plan["measured_joint_envelope_rad"]
            if (np.max(np.abs(actual-baseline)) > bound
                    or np.max(np.abs(value["arms"][arm]["joint_velocities"])) > self.plan["measured_velocity_limit_rad_s"]
                    or np.max(np.abs(np.asarray(value["arms"][arm]["finger_positions"])
                        - np.asarray(self.plan["initial"][arm]["finger_positions"]))) > (1e-4 if initial else 1e-3)):
                raise CharacterizationError("measured_robot_envelope_exceeded")
        return value

    def dispatch_next(self):
        if not self._lock.acquire(blocking=False):
            self._poisoned = True
            raise CharacterizationError("concurrent_diagnostic_dispatch_forbidden")
        try:
            if self._poisoned or self._stopped or self._next > MAX_ACTIONS:
                raise CharacterizationError("diagnostic_stopped_no_retry")
            if content_hash(self.plan) != self._plan_hash:
                raise CharacterizationError("diagnostic_plan_mutated")
            validate_probe_plan(self.plan)
            index = self._next
            self._next += 1  # Failure consumes this local diagnostic slot; never retry.
            source = self.plan["actions"][index-1]
            before = self._measurement(initial=index == 1)
            authorization = {}

            def verify(action, native_counter):
                if (native_counter != index-1 or action != source["action"]
                        or self._verify_runtime(deepcopy(self.plan)) is not True):
                    raise CharacterizationError("diagnostic_runtime_or_target_binding_changed")
                authorization["sha256"] = verify_diagnostic_authorization(self._authorization, self.plan, index)
                return True

            accounting = observe_native_joint_action_once(self.env, source["action"], verify_authorization=verify)
            after = self._measurement()
            self._stopped = accounting["native_end_observed"] or index == MAX_ACTIONS
            return {"schema": "robodojo-native-characterization-action-1", "scope": "diagnostic_native_actions",
                "diagnostic_native_action_index": index, "planned_probe": source["probe"],
                "plan_sha256": self._plan_hash, "authorization_sha256": authorization["sha256"],
                "target": deepcopy(source["action"]), "before": before, "after": after,
                "native_accounting": accounting, "benchmark_attempt_started": False,
                "stop_after_this_action": self._stopped,
                "task_success_assessed": False, "model_calls": 0, "full_test_authorized": False,
                "standard_ready": False, "calibration_passed": False, "settled_tracking_assessed": False}
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()
