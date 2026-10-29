"""Persisted diagnostic evidence tests; synthetic temporary files, no launch."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import provision_native_characterization as parent
import test_native_preview_supervisor as previous_tests


@pytest.fixture
def evidence(monkeypatch):
    old = previous_tests.NativePreviewEvidenceTests()
    old.setUp()
    try:
        baseline = {"left_arm_joint_state": [0.]*6, "right_arm_joint_state": [0.]*6,
                    "left_ee_joint_state": [1.], "right_ee_joint_state": [1.]}
        targets = [deepcopy(baseline)]
        for arm in ("left", "right"):
            for joint in range(6):
                for delta in (.005, -.005, 0.):
                    action = deepcopy(baseline); action[arm+"_arm_joint_state"][joint] += delta
                    targets.append(action)
        plan = {"initial": {arm: {"joint_positions": [0.]*6} for arm in ("left", "right")},
            "initial_joint_tolerance_rad": .0002, "measured_joint_envelope_rad": .02,
            "measured_velocity_limit_rad_s": 1., "actions": [{"action": target, "probe": str(i)} for i,target in enumerate(targets)]}
        candidate = {"calibration_status": "uncalibrated_candidate", "standard_ready": False, "model_input_eligible": False}
        def put(name, value):
            path = old.path / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return {"path": name, "sha256": parent.digest(path)}
        def state(index):
            target = baseline if index == 0 else targets[index-1]
            return {**candidate, "schema": "robodojo-native-characterization-measurement-1",
                "native_physics_time_s": round(1.2 + .04*index, 9), "native_physics_counter": 300+10*index,
                "native_action_counter": [index], "raw_robot_inventory": {"num_envs": 1, "robots": [{},{}]},
                "arms": {arm: {"joint_positions": target[arm+"_arm_joint_state"], "joint_velocities": [0.]*6,
                    "finger_positions": [.044,.044], "link6_world_pose_wxyz": [0.,0.,0.,1.,0.,0.,0.],
                    "link6_actor_world_jacobian": [[0.]*6 for _ in range(6)]} for arm in ("left","right")}}
        def capture(index):
            label = "before" if index == 0 else f"action_{index:03d}"
            robot = state(index)
            camera = {**candidate, "schema": "robodojo-native-camera-inventory-1",
                "native_physics_time_s": robot["native_physics_time_s"], "cameras": dict.fromkeys(("head","left_wrist","right_wrist"), {})}
            images = []
            for source in old.images if index in parent.RGB_ACTIONS else []:
                item = deepcopy(source); item["path"] = label + "/" + source["path"]
                path = old.path / item["path"]; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((old.path/source["path"]).read_bytes()); images.append(item)
            return {"label": label, "physics_unchanged_during_capture": True, "native_action_counter": [index],
                "native_physics_time_s": robot["native_physics_time_s"], "native_physics_counter": 300+10*index,
                "native_sim_step_calls": 300+10*index, "camera_inventory": put(label+"/camera_inventory.json", camera),
                "robot_measurement": put(label+"/robot_measurement.json", robot), "images": images}
        scope = {"model_calls": 0, "benchmark_attempt_started": False, "full_test_authorized": False,
                 "standard_ready": False, "calibration_passed": False, "task_success_assessed": False}
        captures, records, receipts = [capture(0)], [], []
        previous = captures[0]
        for index in range(1,38):
            after = capture(index)
            receipt = {**scope, "schema": "robodojo-native-characterization-action-1", "scope": "diagnostic_native_actions",
                "diagnostic_native_action_index": index, "plan_sha256": parent.content_hash(plan),
                "authorization_sha256": old.authorization["sha256"], "target": targets[index-1],
                "planned_probe": plan["actions"][index-1]["probe"], "before": state(index-1), "after": state(index),
                "stop_after_this_action": index == 37,
                "native_accounting": {"native_counter_before": index-1, "native_counter_after": index,
                    "native_sim_step_calls": 10, "physics_dt_s": .004, "native_terminal_callback_calls": 1,
                    "native_reward_callback_calls": 1, "native_reward_step_calls": 1, "native_end_observed": False,
                    "task_success_assessed": False, "benchmark_attempt_started": False, "model_calls": 0,
                    "native_physics_counter_before": 300+10*(index-1), "native_physics_counter_after": 300+10*index,
                    "physics_time_before_s": previous["native_physics_time_s"], "physics_time_after_s": after["native_physics_time_s"]}}
            receipts.append(receipt)
            records.append({"index": index, "receipt": put(f"native_action_{index:03d}.json", receipt),
                "before_capture": previous["label"], "after_capture": after})
            if after["images"]: captures.append(after)
            previous = after
        old.preparation.update(plan_sha256=parent.content_hash(plan), plan_file_sha256="synthetic-plan-file-sha", prior_preview={})
        old.result.update(scope, schema="robodojo-native-characterization-result-1", diagnostic_completed=True,
            diagnostic_native_actions=37, diagnostic_physics_substeps=370, diagnostic_simulated_time_s=1.48,
            image_count=18, native_take_action_count=[37], plan_sha256=parent.content_hash(plan),
            plan_file_sha256="synthetic-plan-file-sha", authorization_sha256=old.authorization["sha256"],
            captures=captures, actions=records, rgb_checkpoint_action_indices=list(parent.RGB_ACTIONS))
        old.ready.update(schema="robodojo-native-characterization-preclose-1", diagnostic_native_actions=37,
            benchmark_attempt_started=False, full_test_authorized=False, image_count=18)
        monkeypatch.setattr(parent.subprocess, "check_output", lambda *a,**kw: json.dumps([
            {"format":"PNG","mode":"RGB","shape":[480,640,3],"dtype":"uint8","std":1.}] * 18))
        def verify():
            for index, receipt in enumerate(receipts, 1):
                records[index-1]["receipt"] = put(f"native_action_{index:03d}.json", receipt)
            put("preparation.json", old.preparation)
            old.result["preparation_sha256"] = parent.digest(old.path/"preparation.json")
            put("characterization_result.json", old.result)
            old.ready.update(preparation_sha256=old.result["preparation_sha256"],
                characterization_result_sha256=parent.digest(old.path/"characterization_result.json"))
            put("characterization_preclose_ready.json", old.ready)
            return parent.verify_characterization_evidence(old.path, old.binding, old.identity, old.expected,
                old.helpers, old.log, Path(sys.executable), old.authorization, plan, {})
        yield old, receipts, verify
    finally:
        old.doCleanups()


def test_complete_fixed_diagnostic_evidence(evidence):
    _, _, verify = evidence
    value = verify()
    assert value["diagnostic_actions_verified"] is True and value["diagnostic_native_actions"] == 37
    assert len(value["independent_image_decoding"]) == 18 and value["image_visual_review_required"] is True


@pytest.mark.parametrize("mutation", ["scope", "target", "extra_reward", "substep", "terminal", "hidden_action",
    "camera_time", "missing_checkpoint", "asset_fidelity", "close_error", "helper_hash", "model_call"])
def test_tampered_or_expanded_evidence_rejected(evidence, mutation):
    old, receipts, verify = evidence
    row = receipts[4]
    if mutation == "scope": row["scope"] = "benchmark"
    if mutation == "target": row["target"] = {**row["target"], "left_ee_joint_state": [0.]}
    if mutation == "extra_reward": row["native_accounting"]["native_reward_callback_calls"] = 2
    if mutation == "substep": row["native_accounting"]["native_sim_step_calls"] = 11
    if mutation == "terminal": row["native_accounting"]["native_end_observed"] = True
    if mutation == "hidden_action": old.result["actions"][4]["after_capture"]["native_action_counter"] = [6]
    if mutation == "camera_time": old.result["actions"][4]["after_capture"]["native_physics_time_s"] += .004
    if mutation == "missing_checkpoint": old.result["captures"].pop()
    if mutation == "asset_fidelity": old.ready["dependency_closure_complete"] = True
    if mutation == "close_error": old.ready["cleanup_errors"] = [{"phase":"synthetic_failure"}]
    if mutation == "helper_hash": old.preparation["source_file_sha256"] = {}
    if mutation == "model_call": old.result["model_calls"] = 1
    with pytest.raises(ValueError): verify()
