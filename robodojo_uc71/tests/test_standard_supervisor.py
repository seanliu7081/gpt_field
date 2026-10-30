import importlib.util
import json
import os
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def supervisor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("standard_supervisor_test", scripts / "run_standard_attempt.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def good_infrastructure():
    return {"returncode":0,"termination_signals":[],"remaining_owned_pids":[],
        "gpu_processes_released":True,"gpu_memory_released":True,"frozen_inputs_unchanged":True,
        "post_reap_reconciliation":{"owned_worker_reaped_verified":True,
            "reconciled_terminal":{"official_success":True}},
        "worker_shutdown":{"verified":True},"video":{"available":True},"trajectory":{"available":True},
        "monitor_error":None}


@pytest.mark.parametrize("field,value", [("gpu_processes_released",False),
    ("gpu_memory_released",False),("frozen_inputs_unchanged",False),("returncode",1),
    ("termination_signals",[15]),("remaining_owned_pids",[123]),("monitor_error","TimeoutError"),
    ("video",{"available":False}),("worker_shutdown",{"verified":False}),
    ("post_reap_reconciliation",{})])
def test_infrastructure_failure_cannot_report_clean_completion(supervisor, field, value):
    record = good_infrastructure()
    assert supervisor.infrastructure_completed(record)
    record[field] = value
    assert not supervisor.infrastructure_completed(record)
    if field != "post_reap_reconciliation":
        assert record["post_reap_reconciliation"]["reconciled_terminal"]["official_success"] is True


def test_unmanaged_entry_point_never_spawns(supervisor, monkeypatch, tmp_path):
    monkeypatch.delenv("UC71_STANDARD_SUPERVISOR", raising=False)
    spawn = Mock(); monkeypatch.setattr(supervisor.subprocess, "Popen", spawn)
    with pytest.raises(ValueError, match="private_root_supervisor"):
        supervisor.run(tmp_path / "absent_spec.json")
    spawn.assert_not_called()


@pytest.mark.parametrize("failure",["busy_gpu","disconnected","invalid_token"])
def test_gpu_and_auth_failures_do_not_claim_or_consume_attempt(supervisor,monkeypatch,tmp_path,failure):
    import gpt_field.chatgpt_auth as auth
    monkeypatch.setattr(supervisor.os,"getuid",lambda:0)
    monkeypatch.setenv("UC71_STANDARD_SUPERVISOR","1")
    manifest = {"attempt_dir":str(tmp_path),"configuration":{"runtime":{"minimum_free_mib":16384}}}
    monkeypatch.setattr(supervisor,"preflight",lambda _,**kw:({"gpu_binding":{}},manifest,{}))
    gpu = {"processes":[{"pid":123}] if failure=="busy_gpu" else [],"memory_free":"24000 MiB"}
    monkeypatch.setattr(supervisor,"gpu_snapshot",lambda:[gpu])
    monkeypatch.setattr(supervisor,"validate_single_gpu",lambda *_:gpu)
    credentials = Mock()
    credentials.status.return_value = {"connected":failure!="disconnected"}
    credentials.access_token.side_effect = ValueError("invalid_token_fixture")
    monkeypatch.setattr(auth,"SubscriptionCredentials",Mock(return_value=credentials))
    spawn = Mock(); monkeypatch.setattr(supervisor.subprocess,"Popen",spawn)
    with pytest.raises(ValueError):
        supervisor.run(tmp_path/"spec.json")
    assert not (tmp_path/"launch_claim.json").exists()
    assert not (tmp_path/"consumed.json").exists()
    spawn.assert_not_called()


def test_approval_must_match_exact_manifest_before_acceptance(supervisor, tmp_path):
    attempt = tmp_path / "attempt"; attempt.mkdir()
    manifest = {"attempt_dir":str(attempt),"manifest_sha256":"a"*64,"preview_sha256":"b"*64}
    with pytest.raises(FileNotFoundError): supervisor.verify_approval(manifest)
    approval = {"schema":"robodojo-uc71-human-approval-1","approval_mode":"manual_chat_review",
        "manual_browser_review":False,"attempt_dir":str(attempt),"attempts":1,
        "user_message":"offline fixture only","manifest_sha256":"c"*64,"preview_sha256":"b"*64}
    path = attempt / "approval.json"; path.write_text(json.dumps(approval)); os.chown(path,0,65534); path.chmod(0o640)
    with pytest.raises(ValueError, match="does_not_match"): supervisor.verify_approval(manifest)
    approval["manifest_sha256"] = "a"*64; path.write_text(json.dumps(approval))
    assert len(supervisor.verify_approval(manifest)) == 64


def test_private_runtime_directory_modes_checked_before_spawn(supervisor, tmp_path):
    attempt = tmp_path / "attempt"
    runtime = attempt / "runtime"
    attempt.mkdir(); runtime.mkdir()
    paths = [attempt / "worker", *(runtime / name for name in ("home","cache","config","data","tmp"))]
    for path in paths: path.mkdir()
    for path in [runtime,*paths]: os.chown(path,65534,65534); path.chmod(0o700)
    supervisor.verify_private_directories(attempt,runtime)
    (runtime / "tmp").chmod(0o777)
    with pytest.raises(ValueError, match="private_worker_directories"):
        supervisor.verify_private_directories(attempt,runtime)


def test_existing_empty_log_is_group_readable_under_supervisor_umask0077(supervisor,tmp_path):
    path = tmp_path/"native.log"; path.touch(); os.chown(path,0,65534); path.chmod(0o640)
    mask = os.umask(0o077)
    try:
        with supervisor.open_worker_log(path) as stream:
            assert stat.S_IMODE(os.fstat(stream.fileno()).st_mode) == 0o640
            stream.write("synthetic native log\n")
    finally: os.umask(mask)
    assert path.read_text() == "synthetic native log\n"
    with pytest.raises(ValueError,match="empty_worker_log"):
        supervisor.open_worker_log(path)
    assert path.read_text() == "synthetic native log\n"


def test_worker_log_error_closes_fd_and_rejects_links(supervisor,tmp_path,monkeypatch):
    path = tmp_path/"native.log"; path.touch(); os.chown(path,0,65534); path.chmod(0o640)
    link = tmp_path/"link"; link.symlink_to(path)
    with pytest.raises(ValueError,match="canonical"): supervisor.open_worker_log(link)
    opened = []; original = os.open
    def open_fd(*args,**kwargs):
        fd = original(*args,**kwargs); opened.append(fd); return fd
    monkeypatch.setattr(supervisor.os,"open",open_fd)
    def fail(*_): raise OSError("synthetic permission update failure")
    monkeypatch.setattr(supervisor.os,"fchmod",fail)
    with pytest.raises(OSError,match="synthetic"): supervisor.open_worker_log(path)
    assert len(opened) == 1
    with pytest.raises(OSError): os.fstat(opened[0])


def test_runtime_paths_fail_before_launch_and_preserve_venv_python_symlink(supervisor,tmp_path):
    import sys
    python = tmp_path/"venv-python"; python.symlink_to(sys.executable)
    runtime = {"python":str(python),"worker_script":str(supervisor.BASE/"scripts/run_native_worker.py")}
    for name in ("library_path","overlay_target","compat_target"):
        path = tmp_path/name; path.mkdir(); runtime[name] = str(path)
    icd = tmp_path/"icd.json"; icd.write_text("{}"); runtime["icd"] = str(icd)
    supervisor.verify_runtime_executables(runtime)
    runtime["worker_script"] = str(tmp_path/"unreviewed_worker.py")
    with pytest.raises(ValueError,match="reviewed_native_worker"):
        supervisor.verify_runtime_executables(runtime)


def test_conda_record_is_required_in_the_exact_approval_file_closure(supervisor,tmp_path,monkeypatch):
    runtime = {name:"unused-fixture" for name in
        ("python","library_path","icd","overlay_target","compat_target","worker_script")}
    runtime.update(maximum_wall_time_s=10800,ready_timeout_s=600,minimum_free_mib=16384,
                   environment_record=str(tmp_path/"install_status.json"))
    frozen = {}
    for name in ("install_status.json","requirements.lock","package_inventory.json"):
        path=tmp_path/name; path.write_text("synthetic fixture"); frozen[str(path)]=supervisor.file_sha256(path)
    verifier=Mock(); monkeypatch.setattr(supervisor,"verify_runtime_executables",verifier)
    supervisor.verify_runtime_configuration(runtime,frozen)
    verifier.assert_called_once_with(runtime)
    for omitted in frozen:
        with pytest.raises(ValueError,match="conda_environment_must_be_bound"):
            supervisor.verify_runtime_configuration(runtime,{key:value for key,value in frozen.items() if key!=omitted})


def shutdown_fixture(supervisor,tmp_path,monkeypatch):
    """Synthetic on-disk producer-shaped proof; no native bootstrap is run."""
    from robodojo_uc71 import native_runtime
    attempt = tmp_path/"attempt"; root = attempt/"runtime/bootstrap"; root.mkdir(parents=True)
    (attempt/"worker").mkdir()
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value)); return path
    def digest(path): return supervisor.file_sha256(path)
    spec = {"runtime_dir":str(attempt/"runtime"),"manifest_path":str(attempt/"manifest.json"),
        "native_root":str(tmp_path/"native"),"boot_nonce":"synthetic-worker-nonce",
        "gpu_binding":{"gpu_uuid":"GPU-synthetic","pci_bus_id":"0000:01:00.0","kit_index":0,"cuda_ordinal":0}}
    for name in ("source_manifest_path","asset_manifest_path","asset_assessment_path"):
        spec[name] = str(save(tmp_path/(name+".json"),{"synthetic_only":True}))
    baseline = tmp_path/"baseline"; basechild = baseline/"worker/controller"
    base_module = save(basechild/"kit-portable/data/kit/module.py",{"synthetic_module":True})
    current_module = save(root/"kit-portable/data/kit/module.py",{"synthetic_module":True})
    modules = {"synthetic_only":{"file":str(base_module),"version":"test-only"}}
    module_path = save(basechild/"loaded_module_provenance.json",modules)
    result_path = save(basechild/"controller_result.json",{"loaded_module_provenance_sha256":digest(module_path)})
    library = tmp_path/"libcuda.synthetic.so"; library.write_text("not an actual loaded library")
    launch_path = save(baseline/"launch_result.json",{"controller_result_sha256":digest(result_path),
        "child_loaded_graphics_library_paths":[str(library)]})
    spec["runtime_bundle_path"] = str(save(tmp_path/"bundle.json",{"evidence":{"controller_launch":{
        "path":str(launch_path),"sha256":digest(launch_path)}}}))
    expected = {"asset_fidelity_passed":False,"dependency_closure_complete":False,
        "source_manifest_sha256":digest(Path(spec["source_manifest_path"])),
        "asset_manifest_sha256":digest(Path(spec["asset_manifest_path"])),
        "asset_diagnostic_assessment":{"sha256":digest(Path(spec["asset_assessment_path"])),
            "known_missing_reference":"/synthetic/missing.png"}}
    monkeypatch.setattr(supervisor,"expected_native_preparation",lambda _:expected.copy())
    program = tmp_path/"program.py"; program.write_text("# synthetic fixture only\n")
    monkeypatch.setattr(native_runtime,"runtime_program_files",lambda:[program])
    required = [program,*[Path(spec[name]) for name in
        ("runtime_bundle_path","source_manifest_path","asset_manifest_path","asset_assessment_path")]]
    source_hashes = {str(path):digest(path) for path in required}
    manifest = {"attempt_dir":str(attempt),"manifest_sha256":"a"*64,"preview_sha256":"b"*64,
        "configuration":{"asset_policy":native_runtime.published_asset_policy(expected)},
        "files":{str(i):{"path":str(path),"sha256":digest(path)} for i,path in enumerate(required)}}
    approval_path = save(attempt/"approval.json",{"schema":"robodojo-uc71-human-approval-1",
        "approval_mode":"manual_chat_review","manual_browser_review":False,"manifest_sha256":"a"*64,
        "preview_sha256":"b"*64,"attempt_dir":str(attempt),"attempts":1,"user_message":"SYNTHETIC FIXTURE ONLY"})
    os.chown(approval_path,0,65534); approval_path.chmod(0o640)
    preparation = {**expected,"schema":"robodojo-native-worker-preparation-1","manifest_path":spec["manifest_path"],
        "manifest_sha256":"a"*64,"worker_identity":{"pid":123,"boot_nonce":spec["boot_nonce"]},
        "runtime_bundle_sha256":digest(Path(spec["runtime_bundle_path"])),"source_file_sha256":source_hashes,
        "asset_policy":manifest["configuration"]["asset_policy"],"human_approval":{
            "path":str(approval_path),"sha256":digest(approval_path),"manifest_sha256":"a"*64,
            "human_approval_verified":True,"asset_fidelity_passed":False,"dependency_closure_complete":False}}
    prep_path = save(root/"preparation.json",preparation)
    isolation = save(root/"worker_isolation.json",{"pid":123,"uid":65534,"gid":65534,
        "supplementary_groups":[],"no_new_privs":True,"effective_permitted_inheritable_ambient_capabilities":0,
        "access_denied":{"credential_directory":True,"parent_environment":True}})
    terminal = save(attempt/"worker/terminal.json",{"synthetic_only":True})
    paths = native_runtime.private_runtime_paths(attempt/"runtime")
    for name in set(paths["tokens"].values())|{paths["tmpdir"]}:
        directory = Path(name); directory.mkdir(parents=True,exist_ok=True); os.chown(directory,65534,65534)
    log = save(Path(paths["isaaclab_log_directory"])/"isaaclab_synthetic.log",{"synthetic_only":True})
    os.chown(log,65534,65534)
    support = {
        "private_path_preparation":{**paths,"schema":"robodojo-private-runtime-path-preparation-1",
            "pid":123,"uid":65534,"temporary_file_write_passed":True,"kit_runtime_resolution_checked":False},
        "runtime_paths":{**paths,"schema":"robodojo-private-runtime-paths-1","pid":123,"uid":65534,
            "resolved_tokens":paths["tokens"],"verified_before_native_reset":True,"monotonic_s":11.},
        "native_logging_paths":{"schema":"robodojo-private-native-logging-1","pid":123,"uid":65534,
            "directory":paths["isaaclab_log_directory"],"file_paths":[str(log)],"verified_after_native_reset":True,"monotonic_s":12.},
        "loaded_module_provenance":{"synthetic_only":{"file":str(current_module),"version":"test-only"}},
        "loaded_graphics_library_paths":{"pid":123,"source":"/proc/self/maps","paths":[str(library)]}}
    preclose = {"schema":"robodojo-native-worker-preclose-1","ready_for_shutdown":True,
        "no_pending_exception":True,"native_env_close_returned":True,"physx_monitor_healthy":True,
        "cleanup_errors":[],"child_pid":123,"binding":spec["gpu_binding"],"shutdown_mode":"stock-fast",
        "full_teardown_returned":False,"full_plugin_cleanup_proven":False,"model_calls":0,
        "benchmark_attempt_started":True,"asset_fidelity_passed":False,"dependency_closure_complete":False,
        "native_action_count":[1],"monotonic_s":15.,"terminal_sha256":digest(terminal),
        "preparation_sha256":digest(prep_path),"worker_isolation_sha256":digest(isolation)}
    for name,value in support.items(): preclose[name+"_sha256"] = digest(save(root/(name+".json"),value))
    close_path = save(root/"worker_preclose_ready.json",preclose)
    record = {"worker_identity":{"pid":123},"started_monotonic_s":10.,"elapsed_attempt_s":6.,
        "post_reap_reconciliation":{"verified_native_actions":1,"policy_timing":{
            "available":True,"policy_begin_monotonic_s":13.,"last_native_callback_monotonic_s":14.}}}
    return spec,manifest,record,root,close_path


def test_preclose_binds_private_paths_modules_graphics_and_actual_callback_timing(supervisor,tmp_path,monkeypatch):
    spec,manifest,record,_,_ = shutdown_fixture(supervisor,tmp_path,monkeypatch)
    result = supervisor.verify_shutdown(spec,manifest,record)
    assert result["verified"] and result["private_runtime_paths_verified"]
    assert result["runtime_bundle_module_and_graphics_provenance_verified"]
    assert len(result["supporting_artifacts"]) == 5 and result["full_plugin_cleanup_proven"] is False


@pytest.mark.parametrize("kind",["hash","pid","token","logger","module","graphics","approval","timing","failure"])
def test_preclose_rejects_changed_or_inconsistent_supporting_evidence(supervisor,tmp_path,monkeypatch,kind):
    spec,manifest,record,root,close_path = shutdown_fixture(supervisor,tmp_path,monkeypatch)
    name = {"hash":"runtime_paths","pid":"runtime_paths","token":"runtime_paths",
        "logger":"native_logging_paths","module":"loaded_module_provenance","graphics":"loaded_graphics_library_paths",
        "approval":"preparation","timing":"runtime_paths"}.get(kind)
    if kind == "failure": (root/"worker_close_failure.json").write_text("{}"); name = None
    if name:
        path = root/(name+".json"); value = json.loads(path.read_text())
        if kind == "hash": value["changed"] = True
        if kind == "pid": value["pid"] = 124
        if kind == "token": value["resolved_tokens"]["omni_global_cache"] = "/installed/root/cache"
        if kind == "logger": value["file_paths"] = [str(tmp_path/"shared.log")]
        if kind == "module": value["synthetic_only"]["version"] = "unexpected"
        if kind == "graphics": value["pid"] = 124
        if kind == "approval": value["human_approval"]["sha256"] = "f"*64
        if kind == "timing": value["monotonic_s"] = 16.
        path.write_text(json.dumps(value))
        if kind != "hash":
            close = json.loads(close_path.read_text()); close[name+"_sha256"] = supervisor.file_sha256(path)
            close_path.write_text(json.dumps(close))
    with pytest.raises((ValueError,FileNotFoundError)): supervisor.verify_shutdown(spec,manifest,record)
