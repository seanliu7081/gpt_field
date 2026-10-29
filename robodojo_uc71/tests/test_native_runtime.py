"""Offline bootstrap guard/policy/capture tests. No native startup or model."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robodojo_uc71 import native_runtime as runtime
from robodojo_uc71.approval import canonical_sha256, file_sha256
from robodojo_uc71.native_config import ResetOnlyClient
from test_native_worker_protocol import Case
from test_native_episode import action


def root_record(path, value):
    path.write_text(json.dumps(value)); os.chown(path,0,65534); path.chmod(0o640)
    return path


def test_private_root_records_reject_worker_writable_or_wrong_owner(tmp_path):
    path = root_record(tmp_path/"manifest.json",{"synthetic_only":True})
    assert runtime.root_readonly_record(path) == {"synthetic_only":True}
    path.chmod(0o660)
    with pytest.raises(runtime.NativeRuntimeError): runtime.root_readonly_record(path)
    path.chmod(0o640); os.chown(path,65534,65534)
    with pytest.raises(runtime.NativeRuntimeError): runtime.root_readonly_record(path)


def test_private_tokens_use_separate_portable_argument_and_actual_resolution(tmp_path,monkeypatch):
    expected = runtime.private_runtime_paths(tmp_path)
    arguments = runtime.private_kit_arguments(tmp_path)
    assert arguments[:3] == ["--portable","--portable-root",expected["portable_root"]]
    assert len(expected["tokens"]) == 9
    assert len([arg for arg in arguments if arg.startswith("--/app/tokens/")]) == 9
    monkeypatch.setattr(runtime.tempfile,"gettempdir",lambda:expected["tmpdir"])
    result = runtime.verify_effective_runtime_paths(tmp_path,lambda token:expected["tokens"][token[2:-1]])
    assert result["resolved_tokens"] == expected["tokens"]
    with pytest.raises(runtime.NativeRuntimeError):
        runtime.verify_effective_runtime_paths(tmp_path,lambda _:"/root-owned-installed-kit/cache")


def test_private_temp_preparation_requires_owner_mode_and_real_write(tmp_path,monkeypatch):
    expected = runtime.private_runtime_paths(tmp_path); temporary = Path(expected["tmpdir"])
    temporary.mkdir(mode=0o700); os.chown(temporary,65534,65534)
    for name in set(expected["tokens"].values()):
        path = Path(name); path.mkdir(parents=True,exist_ok=True); os.chown(path,65534,65534)
    monkeypatch.setattr(runtime.os,"geteuid",lambda:65534)
    monkeypatch.setattr(runtime.tempfile,"gettempdir",lambda:expected["tmpdir"])
    monkeypatch.setenv("TMPDIR",expected["tmpdir"])
    assert runtime.prepare_private_runtime_paths(tmp_path)["temporary_file_write_passed"] is True
    temporary.chmod(0o750)
    with pytest.raises(runtime.NativeRuntimeError): runtime.prepare_private_runtime_paths(tmp_path)


def test_actual_native_file_handler_must_stay_under_private_temp(tmp_path):
    import logging
    expected = Path(runtime.private_runtime_paths(tmp_path)["isaaclab_log_directory"])
    expected.mkdir(parents=True)
    good = logging.FileHandler(expected/"isaaclab_synthetic.log")
    bad = logging.FileHandler(tmp_path/"shared.log")
    try:
        os.chown(good.baseFilename,65534,65534)
        proof = runtime.verify_native_logging_paths(tmp_path,[good,logging.NullHandler()])
        assert proof["file_paths"] == [good.baseFilename]
        for handlers in ([],[bad],[good,bad]):
            with pytest.raises(runtime.NativeRuntimeError): runtime.verify_native_logging_paths(tmp_path,handlers)
    finally: good.close(); bad.close()


def test_spec_is_exact_existing_path_and_safe_boot_nonce(tmp_path):
    names = {f.name for f in runtime.fields(runtime.WorkerRuntimeSpec)}
    item = tmp_path/"input"; item.write_text("synthetic")
    spec = {name:str(item) for name in names-{"gpu_binding","boot_nonce"}}
    spec.update(schema="robodojo-native-worker-spec-1",boot_nonce="synthetic-worker-nonce",
        gpu_binding={"gpu_uuid":"GPU-synthetic", "pci_bus_id":"0000:01:00.0", "kit_index":0,"cuda_ordinal":0})
    path = root_record(tmp_path/"spec.json",spec)
    assert runtime.WorkerRuntimeSpec.load(path).boot_nonce == "synthetic-worker-nonce"
    for key,value in (("boot_nonce","../synthetic-unsafe"),("unexpected",True)):
        changed = {**spec,key:value}; root_record(path,changed)
        with pytest.raises(runtime.NativeRuntimeError): runtime.WorkerRuntimeSpec.load(path)


def policy_case(tmp_path, monkeypatch):
    preparation = {"asset_fidelity_passed":False,"dependency_closure_complete":False,
        "asset_manifest_sha256":"a"*64,"asset_diagnostic_assessment":{"sha256":"b"*64,
            "known_missing_reference":"/synthetic/missing-normal.png"}}
    manifest = {"attempt_dir":str(tmp_path),"manifest_sha256":"c"*64,"preview_sha256":"d"*64,
        "configuration":{"task_prompt":runtime.TASK_INSTRUCTION,"asset_policy":runtime.published_asset_policy(preparation)}}
    approval = {"schema":"robodojo-uc71-human-approval-1","approval_mode":"manual_chat_review",
        "manual_browser_review":False,"manifest_sha256":"c"*64,"preview_sha256":"d"*64,
        "attempt_dir":str(tmp_path),"attempts":1,"user_message":"SYNTHETIC approval fixture only"}
    root_record(tmp_path/"approval.json",approval)
    monkeypatch.setattr(runtime,"verify_manifest",lambda _:None)
    return preparation,manifest,approval


def test_full_preparation_requires_human_bound_explicit_published_defect_policy(tmp_path, monkeypatch):
    prep,manifest,_ = policy_case(tmp_path,monkeypatch)
    proof = runtime.verify_human_preparation_approval(manifest,prep)
    assert proof["human_approval_verified"] is True and proof["asset_fidelity_passed"] is False
    manifest["configuration"]["asset_policy"]["accept_published_asset_defects"] = False
    with pytest.raises(runtime.NativeRuntimeError): runtime.verify_human_preparation_approval(manifest,prep)


@pytest.mark.parametrize("kind",["diagnostic","prompt","manifest","upgrade","missing_approval"])
def test_diagnostic_or_unbound_assets_cannot_authorize_full_worker(tmp_path, monkeypatch, kind):
    prep,manifest,approval = policy_case(tmp_path,monkeypatch)
    if kind == "diagnostic": approval["schema"] = "robodojo-native-characterization-authorization-1"
    if kind == "prompt": manifest["configuration"]["task_prompt"] = "Different task"
    if kind == "manifest": approval["manifest_sha256"] = "e"*64
    if kind == "upgrade": manifest["configuration"]["asset_policy"]["asset_fidelity_passed"] = True
    root_record(tmp_path/"approval.json",approval)
    if kind == "missing_approval": (tmp_path/"approval.json").unlink()
    with pytest.raises((runtime.NativeRuntimeError,FileNotFoundError)):
        runtime.verify_human_preparation_approval(manifest,prep)


def guarded_case(tmp_path):
    case = Case(tmp_path)
    env = case.env
    env.reset = lambda seed: env.sim_step()
    env.take_action_batch = lambda *a,**kw: None
    env.step = lambda *a,**kw: None
    env.run_eval = lambda: pytest.fail("Task loop executed")
    guard = runtime.NativeExecutionGuard(env)
    guard.reset_once(); guard.bind(case.session)
    return case,guard


def test_guard_allows_one_setup_reset_then_only_real_session_reserved_action(tmp_path):
    case,guard = guarded_case(tmp_path)
    try:
        assert guard.reset_used and guard.sim_step_calls == 1
        with pytest.raises(runtime.NativeRuntimeError): guard.reset_once()
        with pytest.raises(runtime.NativeRuntimeError): case.env.reset(seed=[0])
        with pytest.raises(runtime.NativeRuntimeError): case.env.take_action(action())
        with pytest.raises(runtime.NativeRuntimeError): case.env.sim_step()
        with pytest.raises(runtime.NativeRuntimeError): case.env.step()
        with pytest.raises(runtime.NativeRuntimeError): case.env.take_action_batch([])
        with pytest.raises(runtime.NativeRuntimeError): case.env.run_eval()
        case.begin()
        case.rpc("action",{"ticket":1,"joint_action":action()})
        assert case.env.take_action_cnt == [1] and guard.sim_step_calls == 11
        assert case.env.reward_manager.calls == case.env.terminal_calls == 1
        with pytest.raises(runtime.NativeRuntimeError): case.env.take_action(action())
        with pytest.raises(runtime.NativeRuntimeError): case.env.sim_step()
    finally:
        case.close_sockets()


def callback_case(tmp_path, monkeypatch):
    case = Case(tmp_path)
    env = case.env
    env.model_client = ResetOnlyClient(); env.model_client.call(func_name="reset")
    env.gen_instruction = lambda _: [runtime.TASK_INSTRUCTION]
    env.obs_manager.render_for_capture = lambda: None
    env.obs_manager.get_obs = lambda: [{"vision":"synthetic-images-only","privileged_reward":99}]
    guard = NS(sim_step_calls=300)
    manifest = {"attempt_dir":str(case.attempt),"synthetic_only":True}
    manifest_path = root_record(tmp_path/"manifest.json",manifest)
    approval_path = root_record(case.attempt/"approval.json",{"synthetic_only":True})
    approval = {"path":str(approval_path),"sha256":file_sha256(approval_path)}
    output = tmp_path/"bootstrap"; output.mkdir()
    monitor = NS(is_fatal=lambda:False,get_broken_envs=lambda:[])
    camera,robot = {"robot_only_camera_metadata":True},{"robot_only":True}
    bundle = NS(verify=lambda:True,measure=lambda _:case.measure(),reset_reference={"frozen":"reference"},
        sensor_signature=lambda c:{"actual_camera":c},public_calibration=lambda _: {})
    def save(obs,directory):
        assert set(obs) == {"vision","privileged_reward"}
        rows=[]
        for name in runtime.CAMERAS:
            path = directory/("cam_"+name+".png"); path.write_bytes(b"synthetic-rgb-fixture")
            rows.append({"camera":"cam_"+name,"path":path.name})
        return rows
    def write(path,value): path.write_text(json.dumps(value))
    monkeypatch.setattr(runtime,"capture_robot_inventory",lambda _:deepcopy(robot))
    monkeypatch.setattr(runtime,"capture_camera_inventory",lambda _:deepcopy(camera))
    monkeypatch.setattr(runtime,"capture_paused_physics_state",lambda env,count:{"native_physics_time_s":env.sim.sim.current_time,"native_sim_step_calls":count})
    monkeypatch.setattr(runtime,"validate_native_preview_config",lambda _:None)
    def verify(reference,actual_robot,actual_camera,images):
        assert reference == {"frozen":"reference"} and actual_robot == robot and actual_camera == camera
        assert set(images) == set(runtime.CAMERAS)
        return {"passed":True,"start_state_signature":{"actual_reset_images_checked":True}}
    monkeypatch.setattr(runtime,"verify_fresh_reset",verify)
    callback = runtime.RuntimeCallbacks(env=env,bundle=bundle,manifest=manifest,
        preparation={"manifest_path":str(manifest_path)},approval=approval,output=output,guard=guard,
        monitor=monitor,config=None,save_rgb=save,write_json=write)
    return case,callback,bundle,output


def test_ready_captures_fresh_images_and_exact_paused_state_without_advancing(tmp_path, monkeypatch):
    case,callback,bundle,output = callback_case(tmp_path,monkeypatch)
    try:
        status = callback.verify("ready")
        assert status.sensor_signature == {"actual_camera":{"robot_only_camera_metadata":True}}
        assert status.start_state_signature == {"actual_reset_images_checked":True}
        paused = json.loads((output/"fresh_reset_01_ready/native_paused_state.json").read_text())
        assert paused["native_physics_counter"] == 300 and paused["native_action_counter"] == [0]
        assert not case.env.actions and case.env.sim.sim.current_time == 1.2
    finally: case.close_sockets()


@pytest.mark.parametrize("kind",["bundle","instruction","physics","approval_write"])
def test_readiness_cannot_ignore_failed_bundle_instruction_or_readonly_contract(tmp_path, monkeypatch,kind):
    case,callback,bundle,_ = callback_case(tmp_path,monkeypatch)
    try:
        if kind == "bundle": bundle.verify = lambda:False
        if kind == "instruction": case.env.gen_instruction = lambda _: ["Different instruction"]
        if kind == "physics": case.env.obs_manager.render_for_capture = lambda:setattr(case.env.sim.sim,"current_time",1.204)
        if kind == "approval_write": (case.attempt/"approval.json").chmod(0o660)
        with pytest.raises(runtime.NativeRuntimeError): callback.verify("ready")
        assert case.env.actions == []
    finally: case.close_sockets()
