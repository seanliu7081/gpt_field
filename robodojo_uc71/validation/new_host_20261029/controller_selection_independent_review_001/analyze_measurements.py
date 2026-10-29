"""Independent offline metrics from the completed native 960-action records."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(BASE / "scripts"), str(BASE / "src")]
from analyze_joint_probe import URDF, THRESHOLDS, analyze_state, pose_matrix

V = BASE / "validation/new_host_20261029"
HERE = Path(__file__).resolve().parent
ATTEMPT = V / "native_controller_characterization_010"
CHILD = ATTEMPT / "worker/controller"
sources = {}

def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected: raise ValueError("changed_source:" + str(path))
    sources[str(path)] = digest
    return path

def read(path, expected=None):
    return json.loads(bind(path, expected).read_text())

def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    # Quaternion chord angle avoids acos loss of precision near identity.
    chord = min(np.linalg.norm(a-b), np.linalg.norm(a+b))
    return float(4 * np.arcsin(min(1., chord/2)))

def pose(state, arm, offset):
    p = state["arms"][arm]["link6_world_pose_wxyz"]
    return pose_matrix(p)[:3, 3] + pose_matrix(p)[:3, :3] @ offset, np.asarray(p[3:])

def first_suffix(values, stable=5):
    return next((i for i in range(1, len(values)-stable+1) if all(values[i:])), None)

def main():
    output = HERE / "measurements.json"
    if output.exists(): raise ValueError("fresh_output_required")
    for p in (Path(__file__), BASE/"scripts/analyze_joint_probe.py", BASE/"src/robodojo_uc71/controller.py"):
        bind(p)
    launch = read(ATTEMPT/"launch_result.json")
    for flag in ("diagnostic_completed", "operator_clearances_verified", "gpu_processes_released",
                 "gpu_memory_released", "shutdown_contract_satisfied", "worker_isolation_verified"):
        assert launch[flag] is True
    assert launch["returncode"] == 0 and launch["termination_signals"] == []
    result = read(CHILD/"controller_result.json", launch["controller_result_sha256"])
    preparation = read(CHILD/"preparation.json", result["preparation_sha256"])
    plan = read(preparation["plan_path"], result["plan_file_sha256"])
    assert result["diagnostic_native_actions"] == 960 and result["diagnostic_physics_substeps"] == 9600
    assert len(plan["stages"]) == len(result["stages"]) == 39
    for original, digest in preparation["source_file_sha256"].items():
        bind(ATTEMPT/"scripts_snapshot"/Path(original).name, digest)
    geometry = read(plan["geometry_candidate_path"])
    for row in geometry["source_files"]: bind(row["path"], row["sha256"])
    urdf = URDF(next(row["path"] for row in geometry["source_files"] if Path(row["path"]).suffix == ".urdf"))
    offset = np.asarray(geometry["link6_from_tool"], float)[:3, 3]
    receipts, checks = [], []
    max_lag = {arm: 0. for arm in ("left", "right")}
    for index, entry in enumerate(result["actions"], 1):
        receipt = read(CHILD/entry["receipt"]["path"], entry["receipt"]["sha256"])
        assert entry["index"] == receipt["native_action_index"] == index
        assert receipt["before"]["native_action_counter"] == [index-1]
        assert receipt["after"]["native_action_counter"] == [index]
        assert receipt["measured_envelope_passed"] is True and receipt["measurement_validation_error"] is None
        accounting = receipt["native_accounting"]
        assert accounting["native_sim_step_calls"] == 10
        assert accounting["native_reward_callback_calls"] == accounting["native_terminal_callback_calls"] == 1
        if receipts:
            before, previous = receipt["before"], receipts[-1]["after"]
            assert before["arms"] == previous["arms"]
            assert before["native_physics_time_s"] == previous["native_physics_time_s"]
            assert before["native_physics_counter"] == previous["native_physics_counter"]
        for phase in ("before", "after"):
            checks.extend(analyze_state(receipt[phase], geometry, urdf, f"action_{index:04d}_{phase}"))
        for arm in max_lag:
            q = np.asarray(receipt["after"]["arms"][arm]["joint_positions"])
            max_lag[arm] = max(max_lag[arm], float(max(abs(q-receipt["target"][arm+"_arm_joint_state"]))))
        receipts.append(receipt)
    assert len(receipts) == 960
    for capture in result["captures"]:
        state = read(CHILD/capture["robot_measurement"]["path"], capture["robot_measurement"]["sha256"])
        camera = read(CHILD/capture["camera_inventory"]["path"], capture["camera_inventory"]["sha256"])
        assert capture["physics_unchanged_during_capture"] is True
        assert capture["native_physics_time_s"] == camera["native_physics_time_s"] == state["native_physics_time_s"]
        assert capture["native_action_counter"] == state["native_action_counter"]
        for image in capture["images"]: bind(CHILD/image["path"], image["sha256"])
        checks.extend(analyze_state(state, geometry, urdf, capture["label"]))
    stages = []
    for stage, entry in zip(plan["stages"], result["stages"]):
        if not stage["native_actions"]: continue
        rows = [r for r in receipts if r["stage_index"] == stage["index"]]
        assert len(rows) == stage["native_actions"]
        states = [rows[0]["before"], *[r["after"] for r in rows]]
        time = np.array([s["native_physics_time_s"] for s in states])
        assert np.allclose(np.diff(time), .04, rtol=0, atol=1e-7)
        original = read(CHILD/entry["analysis"]["path"], entry["analysis"]["sha256"])
        report = {"stage_index": stage["index"], "label": stage["label"], "kind": stage["kind"],
            "active_arm": stage["active_arm"], "native_actions": len(rows), "arms": {}}
        for arm in ("left", "right"):
            positions, quats = zip(*(pose(s, arm, offset) for s in states))
            positions, quats = np.asarray(positions), np.asarray(quats)
            joints = np.asarray([s["arms"][arm]["joint_positions"] for s in states])
            fingers = np.asarray([s["arms"][arm]["finger_positions"] for s in states])
            velocities = np.asarray([s["arms"][arm]["joint_velocities"] for s in states])
            speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)/np.diff(time)
            angular = np.asarray([angle(a, b) for a,b in zip(quats, quats[1:])])/np.diff(time)
            values = {"sampled_translation_speed_peak_m_s": float(max(speed)),
                "sampled_angular_speed_peak_rad_s": float(max(angular)),
                "maximum_joint_velocity_rad_s": float(np.max(abs(velocities))),
                "maximum_joint_drift_from_stage_entry_rad": float(np.max(abs(joints-joints[0]))),
                "maximum_tool_displacement_from_stage_entry_m": float(max(np.linalg.norm(positions-positions[0], axis=1))),
                "tail5_finger_sum_proxy_range_m": float(np.ptp(fingers[-5:].sum(axis=1))),
                "tail5_finger_position_range_m": np.ptp(fingers[-5:], axis=0).tolist(),
                "initial_finger_positions_m": fingers[0].tolist(), "final_finger_positions_m": fingers[-1].tolist()}
            for name in ("sampled_translation_speed_peak_m_s", "maximum_joint_drift_from_stage_entry_rad",
                         "tail5_finger_sum_proxy_range_m"):
                assert np.isclose(values[name], original["arms"][arm][name], atol=1e-12, rtol=1e-10)
            if stage["active_arm"] == arm:
                target = stage["target_tool_pose_wxyz"]
                distances = np.linalg.norm(positions-np.asarray(target[:3]), axis=1)
                angles = np.asarray([angle(q, target[3:]) for q in quats])
                values.update(final_position_error_m=float(distances[-1]), final_orientation_error_rad=float(angles[-1]),
                    tail5_max_position_error_m=float(max(distances[-5:])), tail5_max_orientation_error_rad=float(max(angles[-5:])),
                    position_error_m=distances.tolist(), orientation_error_rad=angles.tolist())
                values["settling"] = [{"position_tolerance_m": p, "orientation_tolerance_rad": q,
                    "minimum_stable_samples": n, "first_stable_native_action": first_suffix((distances<=p)&(angles<=q), n)}
                    for p in (.0005,.001,.002) for q in (.005,.01,.02) for n in (3,5)]
                if stage["kind"] in ("gripper_close", "gripper_open"):
                    endpoint = .044 if stage["kind"] == "gripper_open" else 0.
                    sums = fingers.sum(axis=1)
                    values["finger_positions_m"] = fingers.tolist()
                    values["gap_sum_proxy_m"] = sums.tolist()
                    values["first_suffix_each_finger_within1mm_of_endpoint_action"] = first_suffix(np.max(abs(fingers-endpoint), axis=1)<=.001)
                    values["stable_gap_sweeps"] = [{"samples": n, "range_m": tol,
                        "first_stable_window_ending_action": next((i for i in range(n,len(sums))
                            if np.ptp(sums[i-n+1:])<=tol),None)} for n in (3,5) for tol in (.0001,.0005,.001)]
            report["arms"][arm] = values
        stages.append(report)
    keys = [k for k in checks[0] if k not in ("state","arm","native_physics_time_s",
        "measured_arm_joint_positions_rad","measured_finger_positions_m","passed")]
    maxima = {k: max(row[k] for row in checks) for k in keys}
    report = {"schema":"robodojo-controller-selection-independent-measurements-1",
        "source_launch_result":{"path":str(ATTEMPT/"launch_result.json"),"sha256":sources[str(ATTEMPT/"launch_result.json")]},
        "native_actions_analyzed":960,"physics_substeps_analyzed":9600,
        "capture_count":len(result["captures"]),"arm_pose_checks":len(checks),
        "geometry_thresholds":THRESHOLDS,"all_source_bound_geometry_checks_passed":all(r["passed"] for r in checks),
        "maximum_geometry_errors":maxima,"maximum_command_next_state_joint_lag_rad":max_lag,
        "continuous_actual_measurements_verified":True,"stage_metrics":stages,
        "geometry_checks":checks,"controller_selection_issued":False,"runtime_bundle_issued":False,
        "loaded_contact_validated":False,"workspace_wide_tracking_validated":False,
        "full_test_authorized":False,"model_calls":0,"simulator_starts":0,
        "limitations":["Sampled speed and timing describe this single native free-space sequence, not guaranteed caps or loaded contact behavior.",
            "Finger joint sum is a robot-state gap proxy; empty-jaw settling does not establish object grasp or slip performance.",
            "Source-bound geometry uses actual measured q; a fixed nominal tool point does not move with asymmetric finger transients."],
        "source_files":[{"path":p,"sha256":digest} for p,digest in sorted(sources.items())]}
    with output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False);stream.write("\n")
    output.chmod(0o444)
    print(json.dumps({"path":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "geometry_passed":report["all_source_bound_geometry_checks_passed"],"maxima":maxima,
        "arm_pose_checks":len(checks),"max_joint_lag_rad":max_lag,
        "gripper_stages":[s for s in stages if s["kind"] in ("gripper_close","gripper_open")]},indent=2))

if __name__ == "__main__": main()
