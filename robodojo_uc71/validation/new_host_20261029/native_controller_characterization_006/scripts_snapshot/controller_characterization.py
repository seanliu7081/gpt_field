"""Offline second-phase controller diagnostic design and trace analysis.

This module starts no simulator, lifecycle, dispatch, approval, or retry. Its
commands are candidates for a separately scoped diagnostic. The physical tool
offset is explicit and source-bound; no object state or task reward is accepted.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .controller import (ArmState, ControllerConfig, DualArmController,
    measured_tcp_pose, orientation_error_world, tcp_world_jacobian)
from .native_characterization import content_hash, file_hash, validate_probe_plan

ARMS = ("left", "right")


def _rotation(quaternion):
    q = np.asarray(quaternion, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1e-3:
        raise ValueError("Invalid measured quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _world_z_rotation(q, angle):
    w, x, y, z = q
    a, b = np.cos(angle/2), np.sin(angle/2)
    return [a*w-b*z, a*x-b*y, a*y+b*x, a*z+b*w]


def _physical_offset(geometry):
    transform = np.asarray(geometry["link6_from_tool"], float)
    if (geometry.get("schema") != "robodojo-physical-tool-candidate-1"
            or geometry.get("calibration_status") != "uncalibrated_candidate"
            or geometry.get("model_input_eligible") is not False or geometry.get("full_test_authorized") is not False
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.array_equal(transform[:3, :3], np.eye(3))
            or not np.array_equal(transform[3], [0., 0., 0., 1.])
            or any(file_hash(row["path"]) != row["sha256"] for row in geometry["source_files"])):
        raise ValueError("Source-bound identity-oriented physical tool candidate required")
    return transform[:3, 3]


def build_controller_probe_proposal(fixed_probe_plan_path):
    """Build the reviewed 960-action proposal; it grants no execution."""
    fixed_path = Path(fixed_probe_plan_path).resolve(strict=True)
    fixed = json.loads(fixed_path.read_text()); validate_probe_plan(fixed)
    inventory = json.loads(Path(fixed["initial_inventory_path"]).read_text())
    geometry = json.loads(Path(fixed["geometry_candidate_path"]).read_text())
    offset = _physical_offset(geometry)
    home = {}
    for robot in inventory["robots"]:
        arm = {"left_arm": "left", "right_arm": "right"}[robot["arm_name"]]
        pose = np.asarray(robot["body_link_pose_w"]["values"], float)[0, robot["body_names"].index("link6")]
        home[arm] = {"tool_pose_wxyz": [*(pose[:3]+_rotation(pose[3:])@offset), *pose[3:]],
                     "joint_positions": fixed["initial"][arm]["joint_positions"]}
    stages = []
    candidate = ControllerConfig(transit_translation_speed_m_s=.2, contact_translation_speed_m_s=.05,
        angular_speed_rad_s=.5, joint_target_lead_rad=.05, damping=.02)

    def stage(label, arm, kind, count, position=None, quaternion=None, profile="transit", speed=None, angular=None, grip=1.):
        config = asdict(candidate)
        if speed is not None: config[profile+"_translation_speed_m_s"] = speed
        if angular is not None: config["angular_speed_rad_s"] = angular
        stages.append({"index": len(stages), "label": label, "active_arm": arm, "kind": kind,
            "native_actions": count, "target_tool_pose_wxyz": None if position is None else [*position, *quaternion],
            "profile": profile, "controller_config": config, "active_gripper_command": grip,
            "other_gripper_command": 1., "rgb_checkpoint_after": True,
            "operator_clearance_required_before": kind == "gripper_close"})

    stage("baseline_hold", None, "joint_hold", 20)
    for arm in ARMS:
        p = np.asarray(home[arm]["tool_pose_wxyz"][:3]); q = home[arm]["tool_pose_wxyz"][3:]
        for profile, rates, distance in (("contact", (.025, .05), .01), ("transit", (.1, .2), .03)):
            for speed in rates:
                for suffix, target in (("up", p+[0., 0., distance]), ("return", p)):
                    stage(f"{arm}_{profile}_{speed}_{suffix}", arm, "translation", 25,
                          target.tolist(), q, profile, speed)
        raised = p+[0., 0., .03]
        stage(f"{arm}_orientation_raise", arm, "translation", 25, raised.tolist(), q, speed=.1)
        for angular in (.5,):
            for suffix, target_q in (("pulse", _world_z_rotation(q, .08)), ("return", q)):
                stage(f"{arm}_worldZ_rotation_{angular}_{suffix}", arm, "orientation", 20,
                      raised.tolist(), target_q, angular=angular)
        stage(f"{arm}_orientation_lower", arm, "translation", 25, p.tolist(), q, speed=.1)
    # Both arms' motion traces precede any conditional gripper test, so ambiguous
    # visual clearance cannot erase already collected controller measurements.
    for arm in ARMS:
        p = np.asarray(home[arm]["tool_pose_wxyz"][:3]); q = home[arm]["tool_pose_wxyz"][3:]
        raised = p+[0., 0., .03]
        retreated = raised - .04*_rotation(q)[:, 0]
        stage(f"{arm}_gripper_raise", arm, "translation", 25, raised.tolist(), q, speed=.1)
        stage(f"{arm}_generic_negative_toolX_retreat", arm, "translation", 40,
              retreated.tolist(), q, "contact", .05)
        stage(f"{arm}_visual_clearance_checkpoint", arm, "visual_checkpoint", 0, retreated.tolist(), q)
        stage(f"{arm}_gripper_close", arm, "gripper_close", 25, retreated.tolist(), q, grip=0.)
        stage(f"{arm}_gripper_open", arm, "gripper_open", 25, retreated.tolist(), q, grip=1.)
        stage(f"{arm}_return_from_retreat_open", arm, "translation", 40,
              raised.tolist(), q, "contact", .05)
        stage(f"{arm}_gripper_lower_open", arm, "translation", 25, p.tolist(), q, speed=.1)
    total = sum(row["native_actions"] for row in stages)
    if total != 960 or len(stages) != 39 or total >= 1400:
        raise ValueError("Unexpected diagnostic action/stage ceiling")
    return {"schema": "robodojo-controller-characterization-proposal-1", "scope": "diagnostic_native_actions",
        "fixed_probe_plan_path": str(fixed_path), "fixed_probe_plan_file_sha256": file_hash(fixed_path),
        "geometry_candidate_path": fixed["geometry_candidate_path"], "source_sha256": fixed["source_sha256"],
        "binding": fixed["binding"], "home": home, "stages": stages,
        "maximum_native_actions": total, "maximum_physics_substeps": total*10,
        "maximum_simulated_time_s": total*.04, "physics_dt_s": .004, "control_hz": 25,
        "candidate_controller": asdict(candidate), "tool_offset_link6_m": offset.tolist(),
        "required_before_execution": ["Separate source-bound diagnostic scope; no benchmark approval is consumed",
            "Successful fixed37 native accounting and reviewed multi-pose FK/Jacobian evidence",
            "Synchronized physical-tool measurements bound to the exact geometry candidate",
            "Fresh setup home joints and tool poses must match recorded initial state",
            "Per-action authorization, native accounting, physics monitor and bounded parent supervision"],
        "visual_clearance_checkpoint": {"physics_paused": True, "maximum_wait_s": 120,
            "reviewer": "root/operator", "requires_fresh_three_RGB_hashes_and_robot_camera_measurements": True,
            "no_object_truth": True, "ambiguous_or_timeout": "stop_remaining_sequence_without_close_or_automatic_return",
            "new_user_or_full_test_permission_required": False},
        "candidate_abort_envelopes": {"joint_distance_from_home_rad": .35, "measured_joint_velocity_rad_s": 2.,
            "tool_distance_from_home_m": .065, "tool_orientation_from_home_rad": .12,
            "inactive_joint_distance_from_latched_hold_rad": .015,
            "observed_finger_limit_excess_m": .001},
        "native_gripper_contract": {"open_normalized": 1., "close_normalized": 0.,
            "native_open_target_m": .044, "native_close_target_m": -.01,
            "recorded_physical_limits_m": [0., .044], "target_clipping_or_replacement": False,
            "finger_sum_semantics": "joint-separation proxy; not physical inner-pad clearance or grasp success"},
        "benchmark_attempt_started": False, "full_test_authorized": False, "model_calls": 0,
        "execution_authorized": False, "automatic_retry": False, "automatic_tuning": False,
        "calibration_passed": False, "standard_ready": False,
        "limitations": ["Bounds are diagnostic stop candidates, not validated controller settings",
            "Contact profile is characterized in free space only; contact-loaded behavior remains unmeasured",
            "25Hz sampled speeds do not bound continuous within-action motion",
            "Only local home/raised poses are exercised; no workspace-wide safety or tracking guarantee",
            "Empty-gripper timing requires a fresh visual clearance determination at the declared retreat",
            "A failed higher-speed candidate does not authorize tuning or rerunning it",
            "The real FragmentExecutor remains approval-gated; this proposal uses only its DualArmController primitive"]}


def validate_controller_probe_proposal(plan):
    if plan != build_controller_probe_proposal(plan["fixed_probe_plan_path"]):
        raise ValueError("Controller diagnostic proposal differs from its fixed robot-only sources")


def physical_arm_states(measurement, geometry):
    """Explicit physical offset adapter for candidate diagnostic measurements."""
    offset = _physical_offset(geometry)
    states = {}
    for raw in measurement["raw_robot_inventory"]["robots"]:
        arm = {"left_arm": "left", "right_arm": "right"}[raw["arm_name"]]
        item = measurement["arms"][arm]; ids = raw["native_arm_joint_indices"]
        limits = np.asarray(raw["joint_position_limits_lower_upper"]["values"], float)[0, ids]
        velocity = np.asarray(raw["joint_velocity_limits"]["values"], float)[0, ids]
        pose = item["link6_world_pose_wxyz"]
        state = ArmState(item["joint_positions"], item["joint_velocities"], pose[:3], pose[3:],
            item["link6_actor_world_jacobian"], limits[:, 0], limits[:, 1], velocity, offset)
        measured_tcp_pose(state)
        if not np.array_equal(state.joint_positions, np.asarray(raw["joint_positions"]["values"])[0, ids]):
            raise ValueError("Derived arm joints differ from raw measurement")
        states[arm] = state
    if set(states) != set(ARMS): raise ValueError("Both physical arm states required")
    return states


def stage_controller(stage, states):
    """Construct a candidate controller; caller owns every execution gate."""
    if stage["kind"] in ("visual_checkpoint", "joint_hold"):
        raise ValueError("This stage does not use a Cartesian command")
    controller = DualArmController(gripper_commands={arm: 1. for arm in ARMS},
        config=ControllerConfig(**stage["controller_config"]))
    controller.handoff(stage["active_arm"], states)
    return controller


def stage_joint_command(stage, controller, states):
    """Calculate only: this function cannot dispatch, approve, or advance time."""
    if stage["kind"] in ("visual_checkpoint", "joint_hold"):
        raise ValueError("No Cartesian dispatch command for this stage")
    pose = stage["target_tool_pose_wxyz"]
    result = controller.command(states, pose[:3], pose[3:], profile=stage["profile"],
                                gripper_commands={stage["active_arm"]: stage["active_gripper_command"]})
    # The native boundary accepts exact built-in numbers, not NumPy scalars.
    action = {f"{arm}_arm_joint_state": [float(value) for value in result.joint_positions[arm]] for arm in ARMS}
    action.update({f"{arm}_ee_joint_state": [result.gripper_commands[arm]] for arm in ARMS})
    return action, asdict(result.active_result)


def describe_sample(physics_time_s, states, finger_positions):
    arms = {}
    for arm in ARMS:
        state = states[arm]; p, q = measured_tcp_pose(state)
        arms[arm] = {"tool_pose_wxyz": [*p, *q], "joint_positions": list(state.joint_positions),
            "joint_velocities": list(state.joint_velocities), "finger_positions": list(finger_positions[arm]),
            "world_tool_twist_from_measured_joints": (tcp_world_jacobian(state)@state.joint_velocities).tolist()}
    return {"physics_time_s": float(physics_time_s), "arms": arms}


def summarize_stage(stage, samples):
    """Report observations and threshold sweeps; never select a calibrated value."""
    if len(samples) != stage["native_actions"]+1 or stage["native_actions"] < 1:
        raise ValueError("Exactly one pre-state and every acknowledged native action state are required")
    t = np.array([row["physics_time_s"] for row in samples], float)
    if not np.isfinite(t).all() or not np.allclose(np.diff(t), .04, atol=1e-7, rtol=0):
        raise ValueError("Diagnostic measurement cadence differs from native25Hz")
    summaries = {}
    for arm in ARMS:
        values = [row["arms"][arm] for row in samples]
        poses = np.array([v["tool_pose_wxyz"] for v in values], float)
        joints = np.array([v["joint_positions"] for v in values], float)
        fingers = np.array([v["finger_positions"] for v in values], float)
        if (poses.shape != (len(samples), 7) or joints.shape != (len(samples), 6)
                or fingers.shape != (len(samples), 2)
                or not all(np.isfinite(x).all() for x in (poses, joints, fingers))):
            raise ValueError("Finite per-arm robot trace required")
        speed = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)/np.diff(t)
        angular = np.array([np.linalg.norm(orientation_error_world(a[3:], b[3:])) for a, b in zip(poses, poses[1:])])/np.diff(t)
        value = {"sampled_translation_speed_peak_m_s": float(max(speed)),
            "sampled_translation_speed_p95_m_s": float(np.quantile(speed, .95)),
            "sampled_angular_speed_peak_rad_s": float(max(angular)),
            "maximum_joint_drift_from_stage_entry_rad": float(np.max(np.abs(joints-joints[0]))),
            "tail5_tool_position_range_m": np.ptp(poses[-5:, :3], axis=0).tolist(),
            "initial_finger_positions_m": fingers[0].tolist(), "final_finger_positions_m": fingers[-1].tolist(),
            "tail5_finger_range_m": np.ptp(fingers[-5:], axis=0).tolist(),
            "tail5_finger_sum_proxy_range_m": float(np.ptp(fingers[-5:].sum(axis=1)))}
        if arm == stage["active_arm"]:
            target = stage["target_tool_pose_wxyz"]
            distance = np.linalg.norm(poses[:, :3]-target[:3], axis=1)
            angle = np.array([np.linalg.norm(orientation_error_world(p[3:], target[3:])) for p in poses])
            value.update(final_position_error_m=float(distance[-1]), final_orientation_error_rad=float(angle[-1]))
            value["settling_threshold_sweep"] = []
            for pos_tol in (.0005, .001, .002):
                for rot_tol in (.005, .01, .02):
                    for stable in (3, 5):
                        accepted = (distance <= pos_tol) & (angle <= rot_tol)
                        first = next((i for i in range(1, len(samples)-stable+1) if accepted[i:].all()), None)
                        value["settling_threshold_sweep"].append({"position_tolerance_m": pos_tol,
                            "orientation_tolerance_rad": rot_tol, "minimum_stable_samples": stable,
                            "first_stable_native_action": first,
                            "settling_time_s": None if first is None else float(t[first]-t[0])})
            value["stall_progress_observations"] = []
            for weight in (.05, .15):
                error = distance + weight*angle
                for window in (5, 10):
                    progress = [float(error[i-window+1]-min(error[i-window+2:i+1])) for i in range(window, len(error))]
                    value["stall_progress_observations"].append({"orientation_weight_m_per_rad": weight,
                        "window_actions": window, "rolling_progress_m": progress,
                        "interpretation": "Low progress after arrival is expected; genuine-obstruction detection is untested"})
        summaries[arm] = value
    return {"schema": "robodojo-controller-characterization-stage-analysis-1", "stage": stage["label"],
        "scope": "diagnostic_native_actions", "arms": summaries, "calibration_passed": False,
        "standard_ready": False, "benchmark_attempt_started": False, "task_success_assessed": False,
        "automatic_configuration_selection": False, "grasp_or_inner_pad_clearance_inferred": False}
