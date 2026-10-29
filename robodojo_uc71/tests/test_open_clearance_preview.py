"""Synthetic preview boundaries; no simulator imports, launches or approvals."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import prepare_controller_characterization as child
from robodojo_uc71.controller import ArmState
from robodojo_uc71 import controller_characterization as helper
from robodojo_uc71.native_action_observer import clean_joint_action
from test_controller_characterization import proposal
from test_native_characterization import Env


def preview(tmp_path, monkeypatch):
    full = proposal(tmp_path, monkeypatch)
    source = tmp_path/"synthetic_prediction.json"
    source.write_text('{"synthetic_only":true}')
    candidate = {"schema":"robodojo-open-clearance-candidate-1",
        "joint_home_excursion_rad":.35,
        "parameters":{"lift_m":.03,"negative_tool_x_retreat_m":.025,"outward_m":.015,"outward_yaw_rad":0.},
        "fixed_probe_plan_file_sha256":full["fixed_probe_plan_file_sha256"],
        "source_files":[{"path":str(source),"sha256":helper.file_hash(source)}],
        "native_execution_authorized":False,"full_test_authorized":False,"model_calls":0,
        "selection_basis":"Synthetic fixture; never an execution candidate"}
    path = tmp_path/"candidate.json"; path.write_text(json.dumps(candidate))
    return helper.build_open_clearance_preview_proposal(full["fixed_probe_plan_path"], path), full


def states_for(plan):
    states = {}
    for arm in helper.ARMS:
        home = plan["home"][arm]; pose = home["tool_pose_wxyz"]
        offset = np.asarray(plan["tool_offset_link6_m"])
        states[arm] = ArmState(np.asarray(home["joint_positions"], dtype=float), np.zeros(6),
            np.asarray(pose[:3])-helper._rotation(pose[3:])@offset, pose[3:], np.eye(6),
            -np.ones(6)*10, np.ones(6)*10, np.ones(6)*5, offset)
    return states


def test_preview_does_not_change_or_qualify_as_full_plan(tmp_path, monkeypatch):
    plan, full = preview(tmp_path, monkeypatch)
    child.validate_diagnostic_plan(plan, "open-clearance-preview")
    child.validate_diagnostic_plan(full)
    assert full == helper.build_controller_probe_proposal(full["fixed_probe_plan_path"])
    assert len(plan["stages"]) == 8
    assert [s["native_actions"] for s in plan["stages"]] == [25,40,40,25]*2
    assert plan["maximum_native_actions"] == 260 and plan["maximum_physics_substeps"] == 2600
    assert plan["candidate_abort_envelopes"] == full["candidate_abort_envelopes"]
    assert plan["candidate_controller"] == full["candidate_controller"]
    assert plan["execution_authorized"] is False and plan["controller_characterization_completed"] is False
    with pytest.raises(ValueError): child.validate_diagnostic_plan(plan)
    with pytest.raises(ValueError): child.validate_diagnostic_plan(full, "open-clearance-preview")
    assert child.diagnostic_contract()["actions"] == 960
    assert child.diagnostic_contract("open-clearance-preview")["authorized_child_env"] != child.diagnostic_contract()["authorized_child_env"]


@pytest.mark.parametrize("mutation", ["close", "count", "target", "tolerance", "bounds", "candidate"])
def test_preview_cannot_change_exact_candidate_targets_bounds_or_fixed_count(tmp_path, monkeypatch, mutation):
    plan, _ = preview(tmp_path, monkeypatch)
    if mutation == "close": plan["stages"][1]["active_gripper_command"] = 0.
    if mutation == "count": plan["stages"][1]["native_actions"] += 1
    if mutation == "target": plan["stages"][1]["target_tool_pose_wxyz"][0] += .001
    if mutation == "tolerance": plan["endpoint_validation"]["samples"] = 1
    if mutation == "bounds": plan["candidate_abort_envelopes"]["joint_distance_from_home_rad"] = .5
    if mutation == "candidate": Path(plan["candidate"]["path"]).write_text('{}')
    with pytest.raises(ValueError): child.validate_diagnostic_plan(plan, "open-clearance-preview")


def test_actual_controller_outputs_are_open_and_native_normalized_for_all_stages(tmp_path, monkeypatch):
    plan, _ = preview(tmp_path, monkeypatch); states = states_for(plan)
    for stage in plan["stages"]:
        action, _ = helper.stage_joint_command(stage, helper.stage_controller(stage, states), states)
        actual = clean_joint_action(action)
        child.require_open_only_action(actual)
        assert actual == action
        assert actual["left_ee_joint_state"] == actual["right_ee_joint_state"] == [1.]


def test_wider_preview_bound_is_explicit_and_does_not_mutate_full_default(tmp_path, monkeypatch):
    plan, full = preview(tmp_path, monkeypatch)
    path = Path(plan["candidate"]["path"]); candidate = json.loads(path.read_text())
    candidate["joint_home_excursion_rad"] = .5; path.write_text(json.dumps(candidate))
    wider = helper.build_open_clearance_preview_proposal(full["fixed_probe_plan_path"], path)
    expected = {**full["candidate_abort_envelopes"], "joint_distance_from_home_rad":.5}
    assert wider["candidate_abort_envelopes"] == expected
    assert helper.build_controller_probe_proposal(full["fixed_probe_plan_path"]) == full
    assert full["candidate_abort_envelopes"]["joint_distance_from_home_rad"] == .35
    for invalid in (.51, .49, True, None):
        candidate["joint_home_excursion_rad"] = invalid; path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError): helper.build_open_clearance_preview_proposal(full["fixed_probe_plan_path"],path)


@pytest.mark.parametrize("closing_arm", ["left", "right"])
def test_final_native_callback_blocks_close_even_if_generated_command_is_corrupt(tmp_path, monkeypatch, closing_arm):
    plan, _ = preview(tmp_path, monkeypatch); states = states_for(plan); env = Env()
    current = {"native_physics_time_s":0.,"native_physics_counter":0,"native_sim_step_calls":0,
        "arms":{arm:{"finger_positions":[.044,.044]} for arm in helper.ARMS},"images":[]}
    monkeypatch.setattr(child, "capture_stage", lambda *a,**k:deepcopy(current))
    monkeypatch.setattr(child, "capture_characterization_state", lambda e:deepcopy(current))
    monkeypatch.setattr(child, "physical_arm_states", lambda *a:states)
    monkeypatch.setattr(child, "measured_envelope", lambda *a,**k:None)
    action = {f"{arm}_arm_joint_state":[0.]*6 for arm in helper.ARMS}
    action.update({f"{arm}_ee_joint_state":[0. if arm == closing_arm else 1.] for arm in helper.ARMS})
    monkeypatch.setattr(child, "stage_joint_command", lambda *a:(action,None))
    checked = []
    with pytest.raises(RuntimeError, match="cannot dispatch a closing"):
        child.execute_stages(env, {"native_sim_step_calls":0}, tmp_path, plan, {},
            nullcontext, lambda *a,**k:None, lambda:checked.append(True), diagnostic_kind="open-clearance-preview")
    assert checked and env.action_log == [] and env.take_action_cnt == [0]
    assert env.sim._sim_step_counter == 0 and env.reward_manager.calls == 0


@pytest.mark.parametrize("mutation", [None,"early_position","early_rotation","inactive_fingers"])
def test_last_five_measured_states_and_both_open_fingers_are_required(tmp_path, monkeypatch, mutation):
    plan, _ = preview(tmp_path, monkeypatch); stage = plan["stages"][1]
    sample = {"arms":{arm:{"tool_pose_wxyz":list(stage["target_tool_pose_wxyz"]),
        "finger_positions":[.044,.044]} for arm in helper.ARMS}}
    samples = [deepcopy(sample) for _ in range(6)]
    samples[0]["arms"]["left"]["tool_pose_wxyz"][0] += 1.  # The unqualified initial state is not one of the final five.
    if mutation == "early_position": samples[1]["arms"]["left"]["tool_pose_wxyz"][0] += .003
    if mutation == "early_rotation": samples[1]["arms"]["left"]["tool_pose_wxyz"][3:] = helper._world_z_rotation(stage["target_tool_pose_wxyz"][3:], .03)
    if mutation == "inactive_fingers": samples[2]["arms"]["right"]["finger_positions"][1] = .042
    if mutation:
        with pytest.raises(RuntimeError, match="five-state stable/open"): child.verify_preview_endpoint(samples, stage, plan)
    else:
        evidence = child.verify_preview_endpoint(samples, stage, plan)
        assert evidence["passed"] and evidence["samples_checked"] == 5 and evidence["extra_settling_actions"] == 0


def test_preview_authorization_cannot_use_full_environment_or_action_ceiling(tmp_path, monkeypatch):
    program = tmp_path/"synthetic.py"; program.write_text("synthetic_only=True\n")
    monkeypatch.setattr(child,"program_files",lambda:[program])
    binding = {"gpu_uuid":"synthetic","pci_bus_id":"0000:01:00.0","kit_index":0,"cuda_ordinal":0}
    plan = {"binding":binding,"candidate":{"sha256":"synthetic-candidate"},
        "candidate_abort_envelopes":{"joint_distance_from_home_rad":.5}}
    first = {"launch_result_sha256":"probe","analysis_sha256":"analysis"}
    auth = {"schema":"robodojo-open-clearance-preview-authorization-1","plan_sha256":child.content_hash(plan),
        "gpu_uuid":"synthetic","pci_bus_id":binding["pci_bus_id"],"attempt_output":str(tmp_path/"attempt"),
        "diagnostic_native_actions_authorized":True,"maximum_native_actions":260,"maximum_wall_time_s":900,
        "model_calls_authorized":False,"benchmark_attempt_started":False,"full_test_authorized":False,
        "automatic_retry":False,"automatic_tuning":False,"shutdown_mode":"stock-fast",
        "operator_visual_clearance_required":False,"operator_wait_timeout_s":0,
        "checkpoint_position_tolerance_m":.002,"checkpoint_orientation_tolerance_rad":.02,"open_finger_tolerance_m":.001,
        "first_probe_result_sha256":"probe","first_probe_analysis_sha256":"analysis",
        "program_sha256":{str(program):child.digest(program)},"authorization_basis":"Synthetic only","user_message":"Synthetic",
        "diagnostic_kind":"open-clearance-preview","open_only":True,"endpoint_stability_samples":5,
        "operator_review_after_exit":True,"candidate_sha256":"synthetic-candidate",
        "candidate_abort_envelopes":plan["candidate_abort_envelopes"]}
    path = tmp_path/"scope.json"; path.write_text(json.dumps(auth))
    def verify(): return child.check_authorization(path,plan,binding,first_probe=first,launch=True,
        output_parent=tmp_path/"attempt",diagnostic_kind="open-clearance-preview")
    monkeypatch.setenv("UC71_CONTROLLER_CHARACTERIZATION_AUTHORIZED_CHILD","1")
    monkeypatch.delenv("UC71_OPEN_CLEARANCE_PREVIEW_AUTHORIZED_CHILD",raising=False)
    with pytest.raises(ValueError,match="bounded parent"): verify()
    monkeypatch.setenv("UC71_OPEN_CLEARANCE_PREVIEW_AUTHORIZED_CHILD","1")
    assert verify()["maximum_native_actions"] == 260
    path.write_text(json.dumps({**auth,"maximum_native_actions":960}))
    with pytest.raises(ValueError,match="Exact separate"): verify()
    path.write_text(json.dumps({**auth,"candidate_abort_envelopes":{"joint_distance_from_home_rad":.35}}))
    with pytest.raises(ValueError,match="Exact separate"): verify()
