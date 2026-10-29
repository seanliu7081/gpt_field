#!/usr/bin/env python3
"""Offline, manually selected monitoring replay over completed native010.

No simulator/model imports or actions. Arrival replay evaluates the measured
suffix only; it does not claim the alternate early-yield trajectory was run.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(BASE / "src"), str(BASE / "scripts")]
import numpy as np
from robodojo_uc71.controller import ControllerConfig, orientation_error_world
from robodojo_uc71.controller_characterization import _physical_offset, _rotation
from robodojo_uc71.fragment_executor import MonitoringSettings
from robodojo_uc71.schema import ExecutionLimits
from prepare_runtime_bundle import completed_controller

ROOT = BASE / "validation/new_host_20261029"
OUT = Path(__file__).resolve().parent
ATTEMPT = ROOT / "native_controller_characterization_010"
CHILD = ATTEMPT / "worker/controller"

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bound(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    return {"path": str(path), "sha256": digest}

def write(name, value):
    with (OUT / name).open("x") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def main():
    launch = completed_controller(ATTEMPT / "launch_result.json")
    result = read(CHILD / "controller_result.json")
    prep = read(CHILD / "preparation.json")
    plan = read(prep["plan_path"])
    geometry = read(plan["geometry_candidate_path"])
    offset = _physical_offset(geometry)
    controller = ControllerConfig(**plan["candidate_controller"])
    monitoring = MonitoringSettings(
        position_tolerance_m={"transit": .002, "contact": .001},
        orientation_tolerance_rad={"transit": .02, "contact": .01},
        arrival_stable_samples=5, gap_stable_samples=5,
        gap_stable_range_m=.0005, gap_rise_yield_m=.002, gap_rise_samples=2,
        stall_window_steps=10, stall_min_steps=10, stall_progress_m=.0001,
        orientation_error_weight_m_per_rad=.05)
    limits = ExecutionLimits(min_open_steps=20, min_close_steps=15,
        transit_translation_m=.03, transit_rotation_rad=.08,
        contact_translation_m=.01, contact_rotation_rad=.08,
        direct_translation_m=.03, direct_rotation_rad=.08)
    controller.validate(); monitoring.checked()
    selected = {"controller": asdict(controller), "monitoring": asdict(monitoring),
                "execution_limits": asdict(limits)}
    sources = [bound(__file__), bound(ATTEMPT / "launch_result.json"),
        bound(CHILD / "controller_result.json", launch["controller_result_sha256"]),
        bound(CHILD / "preparation.json", launch["preparation_sha256"]),
        bound(prep["plan_path"], prep["plan_file_sha256"]), bound(plan["geometry_candidate_path"])]
    for name in ("controller.py", "controller_characterization.py", "fragment_executor.py", "schema.py"):
        sources.append(bound(BASE / "src/robodojo_uc71" / name))
    actions = []
    for index, item in enumerate(result["actions"], 1):
        spec = item["receipt"]; path = CHILD / spec["path"]
        sources.append(bound(path, spec["sha256"]))
        action = read(path)
        assert action["native_action_index"] == index and action["measured_envelope_passed"]
        assert action["measurement_validation_error"] is None
        accounting = action["native_accounting"]
        assert accounting["native_sim_step_calls"] == 10
        assert accounting["native_counter_after"] == index
        assert accounting["native_terminal_callback_calls"] == accounting["native_reward_callback_calls"] == 1
        actions.append(action)
    assert len(actions) == 960

    def pose(measurement, arm):
        raw = np.asarray(measurement["arms"][arm]["link6_world_pose_wxyz"], float)
        return np.r_[raw[:3] + _rotation(raw[3:]) @ offset, raw[3:]]

    maxima = {key: {arm: 0. for arm in ("left", "right")} for key in (
        "joint_home_excursion_rad", "tool_home_distance_m", "tool_home_rotation_rad",
        "measured_joint_speed_rad_s", "command_lead_rad", "next_state_joint_lag_rad")}
    for action in actions:
        for arm in ("left", "right"):
            home = plan["home"][arm]
            for position in ("before", "after"):
                item = action[position]["arms"][arm]; p = pose(action[position], arm)
                values = {
                    "joint_home_excursion_rad": np.max(np.abs(np.asarray(item["joint_positions"]) - home["joint_positions"])),
                    "tool_home_distance_m": np.linalg.norm(p[:3] - home["tool_pose_wxyz"][:3]),
                    "tool_home_rotation_rad": np.linalg.norm(orientation_error_world(p[3:], home["tool_pose_wxyz"][3:])),
                    "measured_joint_speed_rad_s": np.max(np.abs(item["joint_velocities"]))}
                for key, value in values.items():
                    maxima[key][arm] = max(maxima[key][arm], float(value))
            for key, position in (("command_lead_rad", "before"), ("next_state_joint_lag_rad", "after")):
                value = np.max(np.abs(np.asarray(action["target"][arm + "_arm_joint_state"]) - action[position]["arms"][arm]["joint_positions"]))
                maxima[key][arm] = max(maxima[key][arm], float(value))

    replays = []; earliest_progress = []
    for entry in result["stages"]:
        stage = plan["stages"][entry["stage_index"]]
        if not stage["native_actions"]:
            continue
        rows = actions[entry["first_action_index"]-1:entry["last_action_index"]]
        spec = entry["analysis"]; sources.append(bound(CHILD / spec["path"], spec["sha256"]))
        summary = read(CHILD / spec["path"])
        arm = stage["active_arm"]
        if arm is None:
            continue
        target = stage["target_tool_pose_wxyz"]; profile = stage["profile"]
        distances = []; angles = []; gaps = []; arrival_count = 0; first_complete = None
        stall_values = []; false_stalls = []
        for step, action in enumerate(rows, 1):
            p = pose(action["after"], arm)
            distance = float(np.linalg.norm(p[:3] - target[:3]))
            angle = float(np.linalg.norm(orientation_error_world(p[3:], target[3:])))
            distances.append(distance); angles.append(angle)
            gaps.append(sum(action["after"]["arms"][arm]["finger_positions"]))
            arrived = distance <= monitoring.position_tolerance_m[profile] and angle <= monitoring.orientation_tolerance_rad[profile]
            arrival_count = arrival_count + 1 if arrived else 0
            pose_arrived = arrival_count >= monitoring.arrival_stable_samples
            if first_complete is not None:
                continue
            if stage["kind"] in ("gripper_close", "gripper_open"):
                minimum = limits.min_close_steps if stage["kind"] == "gripper_close" else limits.min_open_steps
                window = gaps[-monitoring.gap_stable_samples:]
                if step >= minimum and len(window) == monitoring.gap_stable_samples and max(window)-min(window) <= monitoring.gap_stable_range_m and pose_arrived:
                    first_complete = step
            elif pose_arrived:
                first_complete = step
            elif step >= monitoring.stall_min_steps:
                errors = np.asarray(distances) + monitoring.orientation_error_weight_m_per_rad*np.asarray(angles)
                window = errors[-monitoring.stall_window_steps:]
                progress = float(window[0] - min(window[1:]))
                stall_values.append({"step": step, "progress_m": progress})
                earliest_progress.append(progress)
                if progress < monitoring.stall_progress_m:
                    false_stalls.append(step)
        assert first_complete is not None and not false_stalls, stage["label"]
        record = {"stage_index": stage["index"], "label": stage["label"], "profile": profile,
            "actions_recorded": len(rows), "monitor_first_completion_action": first_complete,
            "monitor_first_completion_time_s": first_complete*.04,
            "pre_completion_stall_checks": stall_values, "false_stall_steps": false_stalls,
            "last5_maximum_position_error_m": max(distances[-5:]),
            "last5_maximum_orientation_error_rad": max(angles[-5:]),
            "sampled_translation_speed_peak_m_s": summary["arms"][arm]["sampled_translation_speed_peak_m_s"],
            "sampled_angular_speed_peak_rad_s": summary["arms"][arm]["sampled_angular_speed_peak_rad_s"]}
        if stage["kind"] in ("gripper_close", "gripper_open"):
            fingers = np.asarray([x["after"]["arms"][arm]["finger_positions"] for x in rows])
            endpoint = 0. if stage["kind"] == "gripper_close" else .044
            record["first_both_fingers_within_0_5mm_action"] = next(i+1 for i, values in enumerate(fingers) if max(abs(values-endpoint)) <= .0005)
            record["final_finger_positions_m"] = fingers[-1].tolist()
            record["finger_endpoint_error_at_monitor_completion_m"] = float(np.max(np.abs(fingers[first_complete-1]-endpoint)))
            record["last5_gap_range_m"] = max(gaps[-5:])-min(gaps[-5:])
        replays.append(record)
    assert len(replays) == 36
    waits = {}
    for arm in ("left", "right"):
        path = CHILD / f"clearance_{arm}/wait_result.json"; sources.append(bound(path))
        wait = read(path)
        elapsed = wait["end_monotonic_s"] - wait["start_monotonic_s"]
        assert wait["accepted"] and wait["physics_unchanged"] and elapsed <= 90
        assert wait["review_sha256"] == wait["review_consumed_sha256"] and wait["simulation_updates_during_wait"] == 0
        waits[arm] = {"consumed_after_s": elapsed, "review_sha256": wait["review_sha256"], "physics_unchanged": True}
    analysis = {"schema": "robodojo-controller-selection-monitor-replay-1",
        "scope": "measured_robot_geometry_and_free_space_characterization",
        "manually_selected_parameters": selected, "native_actions_analyzed": 960,
        "physics_substeps_analyzed": 9600, "stage_monitor_replays": replays,
        "all_recorded_stages_complete_under_selected_monitors": True,
        "minimum_precompletion_10_sample_progress_m": min(earliest_progress),
        "measured_maxima": maxima, "operator_clearances": waits,
        "source_files": sorted(sources, key=lambda x: x["path"]),
        "full_test_authorized": False, "benchmark_attempt_started": False, "model_calls": 0,
        "loaded_contact_validated": False, "workspace_wide_tracking_validated": False,
        "genuine_obstruction_response_validated": False, "loaded_grip_or_slip_response_validated": False,
        "automatic_configuration_selection": False,
        "limitations": [
            "The selected monitors are replayed over actual recorded measurements; early-yield and chained-policy trajectories were not re-executed.",
            "Arrival tolerances and five-sample checks use the predeclared measurement sweep. Stable-gap/rise/stall levels are conservative manual monitoring choices, not a claim of validated loaded-contact response.",
            "A 10-sample stall window spans9 intervals (0.36s). Observed progressing stages avoid false stalls; actual blocked motion was not induced.",
            "Gap is the sum of measured finger coordinates, not inner-pad clearance or grasp success. The2mm rise trigger and2-sample persistence were not exercised with a loaded grasp.",
            "Execution duration/count budgets are software ceilings. Translation/rotation caps are restricted to measured primitive amplitudes; arbitrary directions and workspace-wide behavior remain unvalidated.",
            "Measured speeds are25Hz samples and do not bound substep peaks. Published missing normal texture is unchanged; asset fidelity and Standard READY remain false."]}
    write("selected_parameters.json", selected)
    write("monitor_replay.json", analysis)
    print(json.dumps({"replay": str(OUT / "monitor_replay.json"), "sha256": sha(OUT / "monitor_replay.json"),
        "minimum_progress_m": min(earliest_progress), "maximum_observed": maxima,
        "gripper_completion": [{"stage": r["label"], "action": r["monitor_first_completion_action"]} for r in replays if "first_both_fingers_within_0_5mm_action" in r]}))

if __name__ == "__main__":
    main()
