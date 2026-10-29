#!/usr/bin/env python3
"""Offline replay of three immutable blocked-state captures; no simulator RPC."""
import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]
sys.path[:0] = [str(BASE / "src"), str(BASE / "scripts")]
import numpy as np
from analyze_joint_probe import URDF, pose_matrix, rotation_error, skew, THRESHOLDS
from robodojo_uc71.controller import (ArmState, ControllerConfig, ControllerError,
    joint_target, measured_tcp_pose, tcp_world_jacobian, orientation_error_world)

def read(path):
    return json.loads(Path(path).read_text())

def main():
    root = OUT.parent
    attempt = root / "standard_attempt_001"
    parent = attempt / "parent"
    sources = {}
    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or actual == expected, str(path)
        assert sources.get(str(path), actual) == actual
        sources[str(path)] = actual
        return path
    geometry = read(bind(root / "tool_geometry_001/candidate.json"))
    for row in geometry["source_files"]:
        bind(row["path"], row["sha256"])
    tool = np.asarray(geometry["link6_from_tool"])
    urdf_entry = next(row for row in geometry["source_files"] if row["path"].endswith(".urdf"))
    urdf = URDF(bind(urdf_entry["path"], urdf_entry["sha256"]))
    bundle = read(bind(root / "runtime_bundle_001/bundle.json"))
    config = ControllerConfig(**bundle["parameters"]["controller"])
    bind(__file__)
    bind(BASE / "scripts/analyze_joint_probe.py")
    bind(BASE / "src/robodojo_uc71/controller.py", "c246c6dcbfe1561656b60e629b4336ef3b53ad8934dc420e0056e9579372bd63")
    bind(BASE / "src/robodojo_uc71/fragment_executor.py")
    bind(BASE / "src/robodojo_uc71/native_worker_client.py")
    bind(BASE / "src/robodojo_uc71/native_measurements.py")
    rows = []
    for turn, action_index in ((0, 1), (1, 2), (2, 3)):
        directory = parent / f"model_turn_{turn:06d}"
        outcome = read(bind(directory / "fragment_outcome.json"))
        decision = read(bind(directory / "accepted_decision.json"))
        bind(directory / "public_observation.json")
        assert outcome["confirmed_steps"] == outcome["submitted_steps"] == 1
        feedback = outcome["feedback"]
        assert feedback["status"] == "blocked" and feedback["reason"] == "ik_unavailable"
        receipt = read(bind(parent / f"action_{action_index:04d}.json"))
        bind(receipt["receipt"]["path"], receipt["receipt"]["sha256"])
        frame = read(bind(parent / f"video_frames/frame_{action_index:04d}.json"))
        assert frame["native_action_index"] == action_index and frame["capture_is_readonly"] is True
        for image in frame["images"].values():
            bind(image["path"], image["sha256"])
        capture = Path(frame["images"]["head"]["path"]).parent
        inventory = read(bind(capture / "robot_inventory.json"))
        paused = read(bind(capture / "native_paused_state.json"))
        bind(capture / "camera_inventory.json")
        assert paused["native_action_counter"] == [action_index]
        arm = feedback["arm_id"]
        raw_robot = next(r for r in inventory["robots"] if r["arm_name"] == arm + "_arm")
        ids, names, bodies = (raw_robot[key] for key in
            ("native_arm_joint_indices", "joint_names", "body_names"))
        body = bodies.index("link6")
        q = np.asarray(raw_robot["joint_positions"]["values"])[0]
        dq = np.asarray(raw_robot["joint_velocities"]["values"])[0]
        poses = np.asarray(raw_robot["body_link_pose_w"]["values"])[0]
        actual = pose_matrix(poses[body])
        base = pose_matrix(np.asarray(raw_robot["root_link_pose_w"]["values"])[0])
        links, frames = urdf.forward(dict(zip(names, q)), base)
        expected = links["link6"]
        com = np.asarray(raw_robot["body_com_pos_b"]["values"])[0, body]
        raw_j = np.asarray(raw_robot["root_physx_view_jacobians"]["values"])[0, body-1][:, ids]
        actor_j = raw_j.copy()
        actor_j[:3] += skew(actual[:3, :3] @ com) @ raw_j[3:]
        expected_j = urdf.jacobian(expected[:3, 3], frames, [names[i] for i in ids])
        limits = np.asarray(raw_robot["joint_position_limits_lower_upper"]["values"])[0, ids]
        velocity = np.asarray(raw_robot["joint_velocity_limits"]["values"])[0, ids]
        state = ArmState(q[ids], dq[ids], poses[body, :3], poses[body, 3:], actor_j,
            limits[:, 0], limits[:, 1], velocity, tool[:3, 3])
        target = feedback["target_tool_pose_wxyz"]
        profile = decision["execution"]["segments"][0]["profile"]
        position, quaternion = measured_tcp_pose(state)
        assert np.allclose(np.r_[position, quaternion], feedback["last_tool_pose_wxyz"], atol=1e-9, rtol=0)
        try:
            joint_target(state, target[:3], target[3:], profile=profile, config=config)
        except ControllerError as error:
            guard = str(error)
        else:
            raise AssertionError("Recorded block was not reproduced")
        assert guard == "no_cartesian_progress"
        position_error = np.asarray(target[:3]) - position
        orientation_error = orientation_error_world(quaternion, target[3:])
        speed = config.transit_translation_speed_m_s if profile == "transit" else config.contact_translation_speed_m_s
        translation_cap, angular_cap = speed/config.control_hz, config.angular_speed_rad_s/config.control_hz
        desired = np.r_[position_error*min(1., translation_cap/max(np.linalg.norm(position_error), 1e-30)),
            orientation_error*min(1., angular_cap/max(np.linalg.norm(orientation_error), 1e-30))]
        jacobian = tcp_world_jacobian(state)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + config.damping**2*np.eye(6), desired)
        lead = np.minimum(config.joint_target_lead_rad, velocity/config.control_hz)
        lead_clipped = np.clip(delta, -lead, lead)
        position_clipped = np.clip(q[ids] + lead_clipped, limits[:, 0], limits[:, 1]) - q[ids]
        predicted = jacobian @ position_clipped
        scale = min(1., translation_cap/max(np.linalg.norm(predicted[:3]), 1e-30),
                    angular_cap/max(np.linalg.norm(predicted[3:]), 1e-30))
        predicted *= scale
        fk_position = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
        fk_rotation = rotation_error(expected[:3, :3], actual[:3, :3])
        j_linear = float(np.max(abs(actor_j[:3]-expected_j[:3])))
        j_angular = float(np.max(abs(actor_j[3:]-expected_j[3:])))
        assert fk_position <= THRESHOLDS["fk_translation_m"] and fk_rotation <= THRESHOLDS["fk_rotation_rad"]
        assert j_linear <= THRESHOLDS["jacobian_linear_max_abs"] and j_angular <= THRESHOLDS["jacobian_angular_max_abs"]
        before_dot, after_dot = float(desired @ (jacobian @ delta)), float(desired @ predicted)
        assert before_dot > 0 and after_dot < 0
        rows.append({"model_turn": turn, "acknowledged_native_action_index": action_index,
            "recorded_feedback": feedback["reason"], "replayed_controller_guard": guard,
            "target_tool_pose_wxyz": target, "profile": profile,
            "fk_position_error_m": fk_position, "fk_rotation_error_rad": fk_rotation,
            "actor_jacobian_linear_max_abs": j_linear, "actor_jacobian_angular_max_abs": j_angular,
            "qualified_kinematic_thresholds_passed": True,
            "tool_jacobian_singular_values": np.linalg.svd(jacobian, compute_uv=False).tolist(),
            "unclipped_descent_dot": before_dot, "clipped_and_scaled_descent_dot": after_dot,
            "guard_threshold": 1e-14, "unclipped_joint_delta_rad": delta.tolist(),
            "lead_limits_rad": lead.tolist(), "clipped_scaled_joint_delta_rad": (position_clipped*scale).tolist(),
            "lead_clipped_joint_indices": np.flatnonzero(abs(delta-lead_clipped)>1e-12).tolist(),
            "native_position_limit_clipped_joint_indices": np.flatnonzero(abs(position_clipped-lead_clipped)>1e-12).tolist()})
    result = {"schema": "robodojo-standard-ik-guard-replay-1",
        "scope": "offline_diagnosis_of_existing_approved_attempt", "controller_configuration": dict(bundle["parameters"]["controller"]),
        "cases": rows, "all_three_recorded_blocks_reproduced": True,
        "runtime_kinematics_match_qualified_source_thresholds": True, "qualification_thresholds": THRESHOLDS,
        "interpretation": "Independent joint lead clipping reverses the local desired Cartesian descent in these saved states. The unchanged no_cartesian_progress guard rejects the next command before dispatch. This is not evidence of a driver/VRAM or kinematic-frame failure.",
        "limitations": ["Only the first three blocked states are replayed; no alternative command was sent or validated.",
            "The approved attempt remains live and unchanged. This analysis neither changes native outcome nor provides feedback to the policy.",
            "A positive unbounded local linearized descent does not establish reachable, collision-free or dynamically safe task motion."],
        "analysis_simulator_calls": 0, "analysis_model_calls": 0, "analysis_native_actions": 0,
        "runtime_or_source_changed": False, "policy_feedback_injected": False,
        "source_files": [{"path": p, "sha256": h} for p,h in sorted(sources.items())]}
    destination = OUT / "ik_guard_replay.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                      "sources": len(sources), "cases": len(rows)}))

if __name__ == "__main__":
    main()
