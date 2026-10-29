"""Synthetic completed-preview bindings; no simulator, execution or approvals."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from robodojo_uc71 import controller_characterization as helper
from test_open_clearance_preview import preview


def write(path, value):
    path.write_text(json.dumps(value)); return path


def ref(path):
    return {"path":str(path),"sha256":helper.file_hash(path)}


def completed_basis(tmp_path, monkeypatch):
    plan, full = preview(tmp_path,monkeypatch)
    candidate_path=Path(plan["candidate"]["path"]);candidate=json.loads(candidate_path.read_text())
    candidate["joint_home_excursion_rad"]=.5
    candidate["parameters"].update(negative_tool_x_retreat_m=.02,outward_m=.012)
    write(candidate_path,candidate)
    plan=helper.build_open_clearance_preview_proposal(full["fixed_probe_plan_path"],candidate_path)
    plan_path=write(tmp_path/"preview_plan.json",plan)
    attempt=tmp_path/"synthetic_attempt";child=attempt/"worker/controller";child.mkdir(parents=True)
    snapshot=attempt/"scripts_snapshot";snapshot.mkdir()
    program=tmp_path/"synthetic_program.py";program.write_text("synthetic_only=True\n")
    (snapshot/program.name).write_bytes(program.read_bytes())
    images=[];stages=[{} for _ in range(8)]
    for index in (1,5):
        directory=child/f"stage_{index:03d}";directory.mkdir()
        rows=[]
        for camera in ("cam_head","cam_left_wrist","cam_right_wrist"):
            path=directory/(camera+".png");path.write_bytes(b"synthetic binding only; not a rendered image")
            images.append(ref(path));rows.append({"camera":camera,"path":str(path.relative_to(child)),"sha256":helper.file_hash(path)})
        stages[index]={"capture":{"images":rows}}
    result_path=write(child/"clearance_preview_result.json",{"plan_sha256":helper.content_hash(plan),
        "candidate":plan["candidate"],"stages":stages})
    launch={"schema":"robodojo-open-clearance-preview-launch-1","clearance_preview_completed":True,
        "controller_characterization_completed":False,"diagnostic_completed":False,
        "diagnostic_native_actions":260,"diagnostic_physics_substeps":2600,
        "open_only_actions_verified":True,"stable_open_endpoints_verified":True,
        "bindings_unchanged_after_clearance_preview":True,"clean_shutdown_confirmed":True,
        "gpu_processes_released":True,"gpu_memory_released":True,"returncode":0,
        "termination_signals":[],"remaining_owned_pids":[],"benchmark_attempt_started":False,
        "full_test_authorized":False,"model_calls":0,"argv":["synthetic","--output",str(child),"--plan",str(plan_path)],
        "clearance_preview_result_sha256":helper.file_hash(result_path),"plan_sha256":helper.content_hash(plan),
        "plan_file_sha256":helper.file_hash(plan_path),"source_file_sha256":{str(program):helper.file_hash(program)}}
    launch_path=write(attempt/"launch_result.json",launch)
    visual_path=write(tmp_path/"synthetic_visual_review.json",{"schema":"robodojo-open-clearance-visual-review-1",
        "reviewer":"root","all_six_actual_endpoint_images_viewed":True,
        "supports_candidate_for_fresh_characterization":True,"full_test_authorized":False,
        "controller_characterization_completed":False,"model_calls":0,"source_files":[ref(launch_path),*images]})
    source=tmp_path/"synthetic_geometry_analysis_input.json";source.write_text('{"synthetic_only":true}')
    geometry_path=write(tmp_path/"synthetic_geometry_review.json",{
        "schema":"robodojo-native-open-preview-geometry-analysis-1","geometry_and_endpoints_passed":True,
        "native_actions_analyzed":260,"arm_pose_checks":1058,
        "thresholds_unchanged_from_predeclared37_action_analysis":True,
        "all260_actual_commands_both_grippers_open":True,"native_clearance_approved_by_this_analysis":False,
        "gripper_timing_validated":False,"full_controller_characterization_completed":False,
        "full_test_authorized":False,"standard_ready":False,"model_calls":0,
        "source_launch_result":ref(launch_path),"source_files":[ref(source)]})
    options={"clearance_preview_path":launch_path,"clearance_visual_review_path":visual_path,
        "clearance_geometry_review_path":geometry_path}
    return full,plan,options,program,snapshot,images,source


def test_completed_preview_changes_only_eight_poses_bound_and_explicit_basis(tmp_path,monkeypatch):
    full,preview_plan,options,*_=completed_basis(tmp_path,monkeypatch)
    revised=helper.build_controller_probe_proposal(full["fixed_probe_plan_path"],**options)
    helper.validate_controller_probe_proposal(revised)
    changed=[i for i,(a,b) in enumerate(zip(full["stages"],revised["stages"])) if a!=b]
    assert changed==[26,27,28,29,33,34,35,36]
    restored=deepcopy(revised);restored.pop("clearance_preview_basis")
    restored["candidate_abort_envelopes"]["joint_distance_from_home_rad"]=.35
    for i in changed:
        assert {k for k in full["stages"][i] if full["stages"][i][k]!=revised["stages"][i][k]}=={"target_tool_pose_wxyz"}
        restored["stages"][i]=full["stages"][i]
    assert restored==full==helper.build_controller_probe_proposal(full["fixed_probe_plan_path"])
    assert revised["clearance_preview_basis"]["closing_authorized_by_preview"] is False
    assert revised["visual_clearance_checkpoint"]["maximum_wait_s"]==120
    assert sum(s["native_actions"] for s in revised["stages"])==960


@pytest.mark.parametrize("broken",["incomplete","signals","visual","image","geometry","source","snapshot"])
def test_incomplete_or_changed_preview_evidence_cannot_change_full_proposal(tmp_path,monkeypatch,broken):
    full,plan,options,program,snapshot,images,source=completed_basis(tmp_path,monkeypatch)
    if broken in ("incomplete","signals"):
        p=options["clearance_preview_path"];v=json.loads(p.read_text())
        if broken=="incomplete":v["clearance_preview_completed"]=False
        else:v["termination_signals"]=[15]
        write(p,v)
    if broken=="visual":
        p=options["clearance_visual_review_path"];v=json.loads(p.read_text());v["supports_candidate_for_fresh_characterization"]=False;write(p,v)
    if broken=="image":Path(images[0]["path"]).write_bytes(b"changed")
    if broken=="geometry":
        p=options["clearance_geometry_review_path"];v=json.loads(p.read_text());v["geometry_and_endpoints_passed"]=False;write(p,v)
    if broken=="source":source.write_text("changed")
    if broken=="snapshot":(snapshot/program.name).write_text("changed")
    with pytest.raises(ValueError):helper.build_controller_probe_proposal(full["fixed_probe_plan_path"],**options)


def test_historical_snapshot_allows_later_program_edit_but_basis_still_binds_exact_hashes(tmp_path,monkeypatch):
    full,plan,options,program,*_=completed_basis(tmp_path,monkeypatch)
    program.write_text("a_later_authorized_revision=True\n")
    revised=helper.build_controller_probe_proposal(full["fixed_probe_plan_path"],**options)
    helper.validate_controller_probe_proposal(revised)
    revised["clearance_preview_basis"]["launch_result"]["sha256"]="0"*64
    with pytest.raises(ValueError):helper.validate_controller_probe_proposal(revised)


def test_all_three_completed_preview_reviews_are_required_together(tmp_path,monkeypatch):
    full,plan,options,*_=completed_basis(tmp_path,monkeypatch)
    options.pop("clearance_geometry_review_path")
    with pytest.raises(ValueError,match="all required"):
        helper.build_controller_probe_proposal(full["fixed_probe_plan_path"],**options)


def test_full_scope_explicitly_binds_new_preview_reviews_and_envelope(tmp_path,monkeypatch):
    import prepare_controller_characterization as child
    program=tmp_path/"synthetic.py";program.write_text("synthetic=True\n")
    monkeypatch.setattr(child,"program_files",lambda:[program])
    binding={"gpu_uuid":"synthetic","pci_bus_id":"0000:01:00.0"}
    plan={"binding":binding,"candidate_abort_envelopes":{"joint_distance_from_home_rad":.5},
        "clearance_preview_basis":{"launch_result":{"sha256":"preview"},"visual_review":{"sha256":"visual"},
            "geometry_analysis":{"sha256":"geometry"}}}
    first={"launch_result_sha256":"first","analysis_sha256":"first-analysis"}
    auth={"schema":"robodojo-controller-characterization-authorization-1","plan_sha256":helper.content_hash(plan),
        "gpu_uuid":"synthetic","pci_bus_id":binding["pci_bus_id"],"attempt_output":str(tmp_path/"attempt"),
        "diagnostic_native_actions_authorized":True,"maximum_native_actions":960,"maximum_wall_time_s":1800,
        "model_calls_authorized":False,"benchmark_attempt_started":False,"full_test_authorized":False,
        "automatic_retry":False,"automatic_tuning":False,"shutdown_mode":"stock-fast",
        "operator_visual_clearance_required":True,"operator_wait_timeout_s":120,"checkpoint_position_tolerance_m":.002,
        "checkpoint_orientation_tolerance_rad":.02,"open_finger_tolerance_m":.001,
        "first_probe_result_sha256":"first","first_probe_analysis_sha256":"first-analysis",
        "program_sha256":{str(program):helper.file_hash(program)},"authorization_basis":"Synthetic only","user_message":"Synthetic"}
    path=write(tmp_path/"scope.json",auth)
    def check():return child.check_authorization(path,plan,binding,first_probe=first,launch=False,output_parent=tmp_path/"attempt")
    with pytest.raises(ValueError):check()
    auth.update(candidate_abort_envelopes=plan["candidate_abort_envelopes"],clearance_preview_result_sha256="preview",
        clearance_visual_review_sha256="visual",clearance_geometry_review_sha256="geometry")
    write(path,auth);assert check()["maximum_native_actions"]==960
    auth["candidate_abort_envelopes"]={"joint_distance_from_home_rad":.35};write(path,auth)
    with pytest.raises(ValueError):check()
