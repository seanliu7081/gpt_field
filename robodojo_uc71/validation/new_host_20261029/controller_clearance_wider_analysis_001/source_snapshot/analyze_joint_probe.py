#!/usr/bin/env python3
"""Offline source-bound FK/Jacobian analysis of the fixed37 native joint probes.

Uses actual recorded joints and body/COM transforms, never commanded joints as
kinematic ground truth. Passing geometry does not validate speed, settled
tracking, gripper timing, a full calibration, or benchmark readiness.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

BASE = Path(__file__).resolve().parents[1]
THRESHOLDS = {"fk_translation_m": 5e-6, "fk_rotation_rad": 1e-5,
    "jacobian_linear_max_abs": 1e-5, "jacobian_angular_max_abs": 1e-5,
    "symmetric_pad_tcp_consistency_m": 5e-6}
CHECKPOINTS = (5, 14, 23, 32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Invalid recorded finite numeric shape")
    return value


def skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(axis, angle):
    axis = finite(axis, (3,)); norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("Zero joint axis")
    k = skew(axis/norm)
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def quaternion_rotation(q):
    q = finite(q, (4,)); norm = np.linalg.norm(q)
    if abs(norm-1) > 1e-5:
        raise ValueError("Invalid recorded quaternion")
    w, x, y, z = q/norm; k = skew([x, y, z])
    return np.eye(3)+2*w*k+2*k@k


def pose_matrix(pose):
    pose = finite(pose, (7,)); result = np.eye(4)
    result[:3, :3] = quaternion_rotation(pose[3:]); result[:3, 3] = pose[:3]
    return result


def rotation_error(a, b):
    delta = a @ b.T
    return float(np.arctan2(np.linalg.norm([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0],
                                          delta[1, 0]-delta[0, 1]])/2,
                           np.clip((np.trace(delta)-1)/2, -1, 1)))


class URDF:
    def __init__(self, path):
        self.joints = {}
        for joint in ET.parse(path).getroot().findall("joint"):
            origin = joint.find("origin")
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            roll, pitch, yaw = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
            transform = np.eye(4); transform[:3, 3] = xyz
            transform[:3, :3] = (axis_rotation([0, 0, 1], yaw) @ axis_rotation([0, 1, 0], pitch)
                                     @ axis_rotation([1, 0, 0], roll))
            axis = joint.find("axis")
            self.joints[joint.attrib["name"]] = {"type": joint.attrib["type"],
                "parent": joint.find("parent").attrib["link"], "child": joint.find("child").attrib["link"],
                "origin": transform, "axis": np.array([float(v) for v in axis.attrib["xyz"].split()]) if axis is not None else None}

    def forward(self, q, root):
        links, frames, pending = {"base_link": root}, {}, dict(self.joints)
        while pending:
            progressed = False
            for name, joint in list(pending.items()):
                if joint["parent"] not in links:
                    continue
                frame = links[joint["parent"]] @ joint["origin"]
                motion = np.eye(4)
                if joint["type"] in ("revolute", "continuous"):
                    motion[:3, :3] = axis_rotation(joint["axis"], q[name])
                elif joint["type"] == "prismatic":
                    motion[:3, 3] = joint["axis"]*q[name]
                elif joint["type"] != "fixed":
                    raise ValueError("Unsupported source joint type")
                frames[name] = frame
                links[joint["child"]] = frame @ motion
                pending.pop(name); progressed = True
            if not progressed:
                raise ValueError("Disconnected/cyclic source robot tree")
        return links, frames

    def jacobian(self, point, frames, arm_names):
        columns = []
        for name in arm_names:
            joint, frame = self.joints[name], frames[name]
            if joint["type"] != "revolute":
                raise ValueError("Expected six revolute arm joints")
            axis = frame[:3, :3] @ (joint["axis"]/np.linalg.norm(joint["axis"]))
            columns.append(np.r_[np.cross(axis, point-frame[:3, 3]), axis])
        return np.column_stack(columns)


def analyze_state(state, geometry, urdf, label):
    if state.get("schema") != "robodojo-native-characterization-measurement-1" or state.get("model_calls") != 0:
        raise ValueError("Expected an actual recorded native characterization state")
    results = []
    for robot in state["raw_robot_inventory"]["robots"]:
        arm = {"left_arm": "left", "right_arm": "right"}[robot["arm_name"]]
        names, bodies = robot["joint_names"], robot["body_names"]
        ids, fingers = robot["native_arm_joint_indices"], robot["native_gripper_joint_indices"]
        if (not robot["is_fixed_base"] or [names[i] for i in ids] != [f"joint{i}" for i in range(1, 7)]
                or [names[i] for i in fingers] != ["joint7", "joint8"]):
            raise ValueError("Unsupported live fixed-base joint mapping")
        q = finite(robot["joint_positions"]["values"], (1, 8))[0]
        poses = finite(robot["body_link_pose_w"]["values"], (1, len(bodies), 7))[0]
        root = pose_matrix(finite(robot["root_link_pose_w"]["values"], (1, 7))[0])
        links, frames = urdf.forward(dict(zip(names, q)), root)
        link_id = bodies.index("link6"); measured = pose_matrix(poses[link_id]); expected = links["link6"]
        com = finite(robot["body_com_pos_b"]["values"], (1, len(bodies), 3))[0, link_id]
        raw = finite(robot["root_physx_view_jacobians"]["values"], (1, len(bodies)-1, 6, 8))[0, link_id-1][:, ids]
        corrected = raw.copy(); corrected[:3] += skew(measured[:3, :3] @ com) @ raw[3:]
        recorded = finite(state["arms"][arm]["link6_actor_world_jacobian"], (6, 6))
        if not np.allclose(corrected, recorded, atol=1e-12, rtol=0):
            raise ValueError("Recorded corrected Jacobian differs from independently corrected raw data")
        expected_j = urdf.jacobian(expected[:3, 3], frames, [names[i] for i in ids])
        tool = finite(geometry["link6_from_tool"], (4, 4))
        if not np.array_equal(tool[:3, :3], np.eye(3)):
            raise ValueError("Physical candidate changes the current translation-only tool axes")
        expected_tool = expected @ tool; measured_tool = measured @ tool
        tool_j = corrected.copy(); tool_j[:3] -= skew(measured[:3, :3] @ tool[:3, 3]) @ corrected[3:]
        expected_tool_j = urdf.jacobian(expected_tool[:3, 3], frames, [names[i] for i in ids])
        actual_centers, source_centers, asymmetry = [], [], np.zeros(3)
        finger_fk_errors = []
        for link, pad in geometry["pad_geometry"].items():
            local = finite(pad["centroid_in_finger_link_m"], (3,))
            actual_pose, source_pose = pose_matrix(poses[bodies.index(link)]), links[link]
            actual_centers.append((actual_pose @ np.r_[local, 1])[:3])
            source_centers.append((source_pose @ np.r_[local, 1])[:3])
            asymmetry += finite(pad["joint_axis"], (3,))*q[names.index(pad["joint_name"])]/2
            finger_fk_errors.append(float(np.linalg.norm(actual_pose[:3, 3]-source_pose[:3, 3])))
        midpoint = np.mean(actual_centers, axis=0)
        symmetric_midpoint = midpoint-measured[:3, :3]@asymmetry
        row = {"state": label, "arm": arm, "native_physics_time_s": state["native_physics_time_s"],
            "measured_arm_joint_positions_rad": q[ids].tolist(),
            "measured_finger_positions_m": q[fingers].tolist(),
            "fk_translation_m": float(np.linalg.norm(expected[:3, 3]-measured[:3, 3])),
            "fk_rotation_rad": rotation_error(expected[:3, :3], measured[:3, :3]),
            "tool_fk_translation_m": float(np.linalg.norm(expected_tool[:3, 3]-measured_tool[:3, 3])),
            "jacobian_linear_max_abs": float(np.max(np.abs(corrected[:3]-expected_j[:3]))),
            "jacobian_angular_max_abs": float(np.max(np.abs(corrected[3:]-expected_j[3:]))),
            "tool_jacobian_linear_max_abs": float(np.max(np.abs(tool_j[:3]-expected_tool_j[:3]))),
            "raw_uncorrected_linear_max_abs": float(np.max(np.abs(raw[:3]-expected_j[:3]))),
            "finger_body_fk_translation_max_m": max(finger_fk_errors),
            "pad_centroid_fk_translation_max_m": float(np.max(np.linalg.norm(np.array(actual_centers)-source_centers, axis=1))),
            "symmetric_pad_tcp_consistency_m": float(np.linalg.norm(measured_tool[:3, 3]-symmetric_midpoint)),
            "nominal_to_actual_pad_midpoint_m": float(np.linalg.norm(measured_tool[:3, 3]-midpoint)),
            "measured_asymmetric_midpoint_shift_m": float(np.linalg.norm(asymmetry))}
        row["passed"] = (all(row[key] <= limit for key, limit in THRESHOLDS.items())
            and row["tool_fk_translation_m"] <= THRESHOLDS["fk_translation_m"]
            and row["tool_jacobian_linear_max_abs"] <= THRESHOLDS["jacobian_linear_max_abs"]
            and row["finger_body_fk_translation_max_m"] <= THRESHOLDS["fk_translation_m"]
            and row["pad_centroid_fk_translation_max_m"] <= THRESHOLDS["fk_translation_m"])
        results.append(row)
    if {row["arm"] for row in results} != {"left", "right"} or len(results) != 2:
        raise ValueError("Both measured arms are required")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); attempt = args.attempt.resolve(strict=True); output = args.output.resolve()
    if output.exists() or not output.is_relative_to(BASE / "validation"):
        parser.error("Use a fresh task validation output directory")
    sources = {}
    def bind(path, expected=None):
        path = Path(path).resolve(strict=True); actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Source artifact changed: {path}")
        sources[str(path)] = actual
        return path
    parent_path = bind(attempt / "launch_result.json"); parent = read(parent_path)
    child = attempt / "characterization"
    result = read(bind(child / "characterization_result.json", parent["characterization_result_sha256"]))
    if (parent.get("diagnostic_completed") is not True or parent.get("returncode") != 0 or parent.get("termination_signals")
            or result.get("diagnostic_native_actions") != 37 or result.get("diagnostic_physics_substeps") != 370
            or result.get("full_test_authorized") is not False or result.get("model_calls") != 0
            or result.get("benchmark_attempt_started") is not False or len(result["actions"]) != 37):
        raise ValueError("Completed fixed37 model-free diagnostic required")
    bind(child / "preparation.json", result["preparation_sha256"])
    plan = read(bind(attempt / "plan.json", result["plan_file_sha256"]))
    program = read(bind(attempt / "program_binding.json"))
    if program != parent["source_file_sha256"]:
        raise ValueError("Parent program bindings differ")
    for path, expected in program.items():
        bind(path, expected); bind(attempt / "scripts_snapshot" / Path(path).name, expected)
    for path, expected in plan["source_sha256"].items(): bind(path, expected)
    geometry_path = bind(plan["geometry_candidate_path"]); geometry = read(geometry_path)
    if geometry["schema"] != "robodojo-physical-tool-candidate-1" or geometry["model_input_eligible"] is not False:
        raise ValueError("Source-bound physical candidate required")
    for row in geometry["source_files"]: bind(row["path"], row["sha256"])
    urdf_path = next(Path(row["path"]) for row in geometry["source_files"] if Path(row["path"]).suffix == ".urdf")
    urdf = URDF(urdf_path)
    all_rows, captures, checkpoints, tracking = [], {}, [], []
    def capture(spec):
        path = bind(child/spec["robot_measurement"]["path"], spec["robot_measurement"]["sha256"])
        state = read(path)
        camera_path = bind(child/spec["camera_inventory"]["path"], spec["camera_inventory"]["sha256"])
        camera = read(camera_path)
        if (spec["physics_unchanged_during_capture"] is not True
                or camera["native_physics_time_s"] != state["native_physics_time_s"]
                or spec["native_physics_time_s"] != state["native_physics_time_s"]):
            raise ValueError("Synchronized capture clocks differ")
        images = [{**row, "path": str(bind(child/row["path"], row["sha256"]))} for row in spec["images"]]
        captures[spec["label"]] = state
        all_rows.extend(analyze_state(state, geometry, urdf, spec["label"]))
        return {"label": spec["label"], "robot_measurement": {"path": str(path), "sha256": sha(path)},
            "camera_inventory": {"path": str(camera_path), "sha256": sha(camera_path)}, "images": images}
    capture(result["captures"][0])
    previous_time = captures["before"]["native_physics_time_s"]
    for index, row in enumerate(result["actions"], 1):
        receipt = read(bind(child/row["receipt"]["path"], row["receipt"]["sha256"]))
        if (row["index"] != index or receipt["diagnostic_native_action_index"] != index
                or receipt["target"] != plan["actions"][index-1]["action"]
                or receipt["before"]["native_physics_time_s"] != previous_time
                or receipt["native_accounting"]["native_sim_step_calls"] != 10):
            raise ValueError("Fixed action/receipt/time sequence differs")
        after_capture = capture(row["after_capture"])
        after_time = captures[row["after_capture"]["label"]]["native_physics_time_s"]
        if receipt["after"]["native_physics_time_s"] != after_time:
            raise ValueError("Receipt differs from synchronized post-action capture")
        previous_time = after_time
        for phase in ("before", "after"):
            all_rows.extend(analyze_state(receipt[phase], geometry, urdf, f"receipt_{index:03d}_{phase}"))
        if index in CHECKPOINTS:
            if len(after_capture["images"]) != 3: raise ValueError("Missing fixed RGB checkpoint")
            checkpoints.append({"index": index, **after_capture})
        for arm in ("left", "right"):
            before = receipt["before"]["arms"][arm]; after = receipt["after"]["arms"][arm]
            target = np.asarray(receipt["target"][f"{arm}_arm_joint_state"])
            q0, q1 = np.asarray(before["joint_positions"]), np.asarray(after["joint_positions"])
            tracking.append({"action_index": index, "planned_probe": receipt["planned_probe"], "arm": arm,
                "duration_s": receipt["after"]["native_physics_time_s"]-receipt["before"]["native_physics_time_s"],
                "commanded_target_rad": target.tolist(), "measured_before_rad": q0.tolist(), "measured_after_rad": q1.tolist(),
                "post_action_target_error_rad": (target-q1).tolist(),
                "max_abs_post_action_target_error_rad": float(np.max(np.abs(target-q1))),
                "max_abs_measured_displacement_rad": float(np.max(np.abs(q1-q0))),
                "measured_after_velocity_rad_s": after["joint_velocities"],
                "max_abs_after_velocity_rad_s": float(np.max(np.abs(after["joint_velocities"])))})
    metric_names = [key for key in all_rows[0] if key not in ("state", "arm", "native_physics_time_s", "measured_arm_joint_positions_rad", "measured_finger_positions_m", "passed")]
    maxima = {key: max(row[key] for row in all_rows) for key in metric_names}
    passed = all(row["passed"] for row in all_rows)
    report = {"schema": "robodojo-native-joint-probe-analysis-1", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_launch_result_sha256": sha(parent_path), "geometry_candidate_sha256": sha(geometry_path),
        "source_launch_result_path": str(parent_path), "geometry_candidate_path": str(geometry_path),
        "source_binding_verified": True, "all_recorded_pose_checks_passed": passed, "kinematics_geometry_passed": passed,
        "full_calibration_passed": False, "controller_tracking_validated": False, "standard_ready": False,
        "full_test_authorized": False, "model_input_eligible": False, "simulator_starts": 0, "model_calls": 0,
        "scope": "offline_actual_measured_robot_kinematics_and_transient_joint_tracking",
        "thresholds": THRESHOLDS, "thresholds_predeclared_before_analysis": True,
        "threshold_source": "Parent task specification; thresholds unchanged after reading measurements",
        "diagnostic_native_actions_analyzed": 37, "unique_capture_count": len(captures),
        "state_records_analyzed_including_receipt_duplicates": len(all_rows)//2,
        "arm_pose_checks": len(all_rows), "maximum_errors": maxima,
        "pose_checks": all_rows, "transient_tracking": tracking,
        "transient_tracking_summary": {"max_abs_post_action_target_error_rad": max(r["max_abs_post_action_target_error_rad"] for r in tracking),
            "max_abs_after_velocity_rad_s": max(r["max_abs_after_velocity_rad_s"] for r in tracking),
            "settled_tracking_assessed": False, "full_speed_tracking_assessed": False, "gripper_timing_assessed": False},
        "fixed_projection_checkpoint_indices": list(CHECKPOINTS), "checkpoint_inputs": checkpoints,
        "projection_visual_review_completed": False,
        "pad_metric_definition": "Subtract R_world_link6 * (axis7*q7+axis8*q8)/2 from actual pad-centroid midpoint before comparing with the fixed nominal tool point. Raw nominal-to-instantaneous-midpoint difference is separately reported.",
        "source_files": [{"path": path, "sha256": expected} for path, expected in sorted(sources.items())],
        "script_sha256": sha(Path(__file__)), "limitations": [
            "All poses remain near reset; successful FK/Jacobian checks are not broad-workspace tracking calibration",
            "Each action is only10x0.004s; remaining velocity and target lag are transients, not settled errors",
            "Both grippers stayed open; no empty-gripper close/open timing or grip stability was calibrated",
            "No model, benchmark attempt, object/contact/reward targeting or native simulator call occurred in this analysis"]}
    output.mkdir(parents=True, exist_ok=False)
    (output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "passed": passed, "maximum_errors": maxima,
                      "transient_tracking_summary": report["transient_tracking_summary"],
                      "analysis_sha256": sha(output/"analysis.json")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
