#!/usr/bin/env python3
"""Parent main-thread runner for ONE existing, human-approved Standard attempt.

Default invocation verifies inputs only. --launch requires an existing exact
approval and a private supervisor; this program never creates approval. There
is no whole-attempt retry, resume, alternate layout, API-key fallback or worker
restart. Native and parent terminal records remain separate after any failure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.approval import canonical_sha256, consume_authorization, file_sha256, verify_manifest
from robodojo_uc71.ipc import JsonLineChannel
from robodojo_uc71.native_worker_client import NativeWorkerClient, write_private_record, _record
from robodojo_uc71.worker_isolation import popen_identity, share_approval_record
from provision_new_host_camera import (child_environment, gpu_process_aliases, gpu_snapshot,
    owned_processes, process_record, terminate_owned, validate_single_gpu)


def read_shared(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_gid != 65534
            or info.st_nlink != 1 or info.st_size > 16*1024*1024 or stat.S_IMODE(info.st_mode) != 0o640):
        raise ValueError("root_owned_readonly_bound_record_required")
    return json.loads(path.read_text())


def verify_approval(manifest):
    attempt = Path(manifest["attempt_dir"])
    value = read_shared(attempt / "approval.json")
    required = {"schema": "robodojo-uc71-human-approval-1", "approval_mode": "manual_chat_review",
        "manual_browser_review": False, "manifest_sha256": manifest["manifest_sha256"],
        "preview_sha256": manifest["preview_sha256"], "attempt_dir": str(attempt), "attempts": 1}
    if (any(type(value.get(k)) is not type(v) or value.get(k) != v for k,v in required.items())
            or not isinstance(value.get("user_message"), str) or not value["user_message"].strip()):
        raise ValueError("explicit_human_approval_does_not_match")
    return file_sha256(attempt / "approval.json")


def verify_private_directories(attempt, runtime):
    attempt, runtime = Path(attempt), Path(runtime)
    if runtime != attempt / "runtime":
        raise ValueError("canonical_attempt_runtime_required")
    for path in [attempt / "worker", runtime, *(runtime / name for name in ("home", "cache", "config", "data", "tmp"))]:
        info = path.lstat()
        if (path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != 65534 or info.st_gid != 65534 or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("private_worker_directories_required")


def open_worker_log(path):
    """Open the reviewed empty log without truncation; preserve group-read under0077."""
    path = Path(path)
    if path.resolve(strict=True) != path:
        raise ValueError("canonical_worker_log_required")
    before = path.lstat()
    fd = os.open(path,os.O_WRONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_gid != 65534
                or info.st_nlink != 1 or info.st_size != 0
                or (info.st_dev,info.st_ino) != (before.st_dev,before.st_ino)):
            raise ValueError("fresh_root_owned_empty_worker_log_required")
        os.fchmod(fd,0o640)
        stream = os.fdopen(fd,"w")
        fd = None
        return stream
    finally:
        if fd is not None:
            try: os.close(fd)
            except OSError: pass


def expected_native_preparation(spec):
    from prepare_native_preview import verify_manifests
    from provision_native_common import pins
    return verify_manifests(Path(spec["native_root"]),Path(spec["source_manifest_path"]),
        Path(spec["asset_manifest_path"]),pins(),Path(spec["asset_assessment_path"]))


def worker_source_hashes(spec):
    from robodojo_uc71.native_runtime import runtime_program_files
    required = [*runtime_program_files(),*[Path(spec[name]) for name in
        ("runtime_bundle_path","source_manifest_path","asset_manifest_path","asset_assessment_path")]]
    return {str(path):file_sha256(path) for path in required}


def verify_runtime_executables(runtime):
    python = Path(runtime["python"])
    # Preserve the venv executable symlink; resolving it loses its environment.
    if not python.is_absolute() or not python.is_file() or not os.access(python,os.X_OK):
        raise ValueError("existing_absolute_runtime_python_required")
    worker = Path(runtime["worker_script"])
    if worker != BASE/"scripts/run_native_worker.py" or not worker.is_file():
        raise ValueError("reviewed_native_worker_entry_point_required")
    for name in ("library_path","overlay_target","compat_target","icd"):
        path = Path(runtime[name])
        if (not path.is_absolute() or path.resolve(strict=True) != path
                or (not path.is_file() if name == "icd" else not path.is_dir())):
            raise ValueError("existing_canonical_runtime_dependencies_required")


def _artifact(path):
    path = Path(path)
    if path.resolve(strict=True) != path or path.stat().st_nlink != 1:
        raise ValueError("canonical_single_link_evidence_required")
    return _record(path,maximum=16*1024*1024)


def _exact(value,expected):
    return all(type(value.get(key)) is type(item) and value.get(key) == item for key,item in expected.items())


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def verify_supporting_runtime_evidence(spec,manifest,record,preclose,preparation):
    from robodojo_uc71.native_runtime import private_runtime_paths, published_asset_policy
    root = Path(spec["runtime_dir"])/"bootstrap"; pid = record["worker_identity"]["pid"]
    expected_paths = private_runtime_paths(Path(spec["runtime_dir"]))
    records,bindings = {},{}
    for name in ("private_path_preparation","runtime_paths","native_logging_paths",
                 "loaded_module_provenance","loaded_graphics_library_paths"):
        path = root/(name+".json"); item,digest = _artifact(path)
        if preclose.get(name+"_sha256") != digest:
            raise ValueError("worker_supporting_evidence_hash_changed")
        records[name] = item; bindings[name] = {"path":str(path),"sha256":digest}
    prepared,paths,logs = (records[name] for name in ("private_path_preparation","runtime_paths","native_logging_paths"))
    if (not _exact(prepared,{**expected_paths,"schema":"robodojo-private-runtime-path-preparation-1",
            "pid":pid,"uid":65534,"temporary_file_write_passed":True,"kit_runtime_resolution_checked":False})
            or not _exact(paths,{**expected_paths,"schema":"robodojo-private-runtime-paths-1",
                "pid":pid,"uid":65534,"resolved_tokens":expected_paths["tokens"],"verified_before_native_reset":True})
            or not _exact(logs,{"schema":"robodojo-private-native-logging-1","pid":pid,"uid":65534,
                "directory":expected_paths["isaaclab_log_directory"],"verified_after_native_reset":True})):
        raise ValueError("worker_private_runtime_path_proof_invalid")
    times = [record.get("started_monotonic_s"),paths.get("monotonic_s"),logs.get("monotonic_s"),preclose.get("monotonic_s")]
    elapsed = record.get("elapsed_attempt_s")
    if (not all(_finite(t) and t > 0 for t in times) or times != sorted(times)
            or not _finite(elapsed) or elapsed < 0 or times[-1] > times[0]+elapsed+1e-6):
        raise ValueError("worker_preparation_and_shutdown_timing_invalid")
    timing = record["post_reap_reconciliation"].get("policy_timing",{})
    if timing.get("available") is True and not (logs["monotonic_s"] <= timing["policy_begin_monotonic_s"]
            <= timing["last_native_callback_monotonic_s"] <= preclose["monotonic_s"]):
        raise ValueError("native_callback_timing_outside_verified_runtime")
    files = logs.get("file_paths",[])
    if (not isinstance(files,list) or not files or len(files) != len(set(files))
            or any(Path(p).parent != Path(expected_paths["isaaclab_log_directory"])
                or Path(p).resolve(strict=True) != Path(p) or not Path(p).is_file()
                or Path(p).stat().st_uid != 65534 or not Path(p).name.startswith("isaaclab_") for p in files)):
        raise ValueError("worker_native_logger_files_invalid")
    for name in set(expected_paths["tokens"].values())|{expected_paths["tmpdir"]}:
        path = Path(name); info = path.stat()
        if path.resolve(strict=True) != path or not path.is_dir() or info.st_uid != 65534:
            raise ValueError("worker_runtime_directory_ownership_changed")
    expected = expected_native_preparation(spec)
    if any(preparation.get(k) != v for k,v in expected.items()):
        raise ValueError("worker_native_source_asset_preparation_changed")
    approval = {"path":str(Path(manifest["attempt_dir"])/"approval.json"),"sha256":verify_approval(manifest),
        "manifest_sha256":manifest["manifest_sha256"],"human_approval_verified":True,
        "asset_fidelity_passed":False,"dependency_closure_complete":False}
    source_hashes = worker_source_hashes(spec)
    frozen = {row["path"]:row["sha256"] for row in manifest["files"].values()}
    if (preparation.get("human_approval") != approval or preparation.get("manifest_path") != spec["manifest_path"]
            or preparation.get("source_file_sha256") != source_hashes
            or any(frozen.get(path) != digest for path,digest in source_hashes.items())
            or preparation.get("asset_policy") != published_asset_policy(expected)
            or preparation["asset_policy"] != manifest["configuration"].get("asset_policy")):
        raise ValueError("worker_approved_preparation_closure_changed")
    # Compare against the actual controller diagnostic bound by the runtime bundle.
    bundle,_ = _artifact(Path(spec["runtime_bundle_path"]))
    baseline = bundle["evidence"]["controller_launch"]
    launch,launch_hash = _artifact(Path(baseline["path"]))
    if launch_hash != baseline["sha256"]:
        raise ValueError("runtime_bundle_controller_launch_changed")
    baseline_root = Path(baseline["path"]).parent/"worker/controller"
    result,result_hash = _artifact(baseline_root/"controller_result.json")
    modules,module_hash = _artifact(baseline_root/"loaded_module_provenance.json")
    if result_hash != launch.get("controller_result_sha256") or module_hash != result.get("loaded_module_provenance_sha256"):
        raise ValueError("runtime_bundle_module_provenance_unbound")
    actual_modules = records["loaded_module_provenance"]
    if not modules or set(modules) != set(actual_modules):
        raise ValueError("native_loaded_module_set_changed")
    for name,entry in modules.items():
        old = Path(entry["file"]); relative = old.relative_to(baseline_root/"kit-portable") if old.is_relative_to(baseline_root/"kit-portable") else None
        target = root/"kit-portable"/relative if relative is not None else old
        if (actual_modules[name] != {"file":str(target),"version":entry["version"]}
                or not target.is_file() or target.resolve(strict=True) != target
                or file_sha256(target) != file_sha256(old)):
            raise ValueError("native_loaded_module_provenance_changed")
    graphics = records["loaded_graphics_library_paths"]
    if (not _exact(graphics,{"pid":pid,"source":"/proc/self/maps"}) or not graphics.get("paths")
            or graphics["paths"] != launch.get("child_loaded_graphics_library_paths")
            or graphics["paths"] != sorted(set(graphics["paths"]))
            or any(not Path(p).is_absolute() or not Path(p).is_file() for p in graphics["paths"])):
        raise ValueError("native_graphics_library_provenance_changed")
    return bindings


def preflight(spec_path, *, require_approval=False):
    from robodojo_uc71.native_runtime_bundle import VerifiedRuntimeBundle
    from robodojo_uc71.native_episode_runner import execution_configuration
    from robodojo_uc71.native_runtime import WorkerRuntimeSpec, TASK_INSTRUCTION, published_asset_policy
    WorkerRuntimeSpec.load(spec_path)
    spec = read_shared(spec_path)
    manifest = read_shared(spec["manifest_path"])
    verify_manifest(manifest)
    if not any(row == {"path": str(Path(spec_path).resolve()), "sha256": file_sha256(spec_path)}
               for row in manifest["files"].values()):
        raise ValueError("worker_spec_not_in_manifest")
    bundle = VerifiedRuntimeBundle.load(spec["runtime_bundle_path"], spec["native_root"])
    bundle.verify()
    configuration = manifest["configuration"]
    preparation = expected_native_preparation(spec)
    frozen = {row["path"]:row["sha256"] for row in manifest["files"].values()}
    if (configuration.get("task_prompt") != TASK_INSTRUCTION
            or configuration.get("asset_policy") != published_asset_policy(preparation)
            or bundle.document["evidence"]["asset_assessment"] != {
                "path":spec["asset_assessment_path"],"sha256":file_sha256(spec["asset_assessment_path"])}
            or any(frozen.get(path) != digest for path,digest in worker_source_hashes(spec).items())):
        raise ValueError("approved_worker_task_assets_and_program_closure_required")
    expected = execution_configuration(controller_config=bundle.controller_config,
        monitoring=bundle.monitoring_settings, limits=bundle.execution_limits,
        initial_gripper_commands={"left": "open", "right": "open"},
        controller_capabilities=bundle.controller_capabilities)
    if configuration.get("execution") != expected:
        raise ValueError("approved_execution_settings_differ_from_characterization")
    if configuration.get("model") != {"name": "gpt-6-astra", "reasoning_effort": "high", "request_deadline_s": 180}:
        raise ValueError("approved_subscription_model_contract_required")
    runtime = configuration["runtime"]
    if (set(runtime) != {"python", "library_path", "icd", "overlay_target", "compat_target", "worker_script",
            "maximum_wall_time_s", "ready_timeout_s", "minimum_free_mib"}
            or type(runtime["maximum_wall_time_s"]) is not int or not 600 <= runtime["maximum_wall_time_s"] <= 10800
            or runtime["ready_timeout_s"] != 600 or runtime["minimum_free_mib"] != 16384):
        raise ValueError("bounded_runtime_configuration_required")
    verify_runtime_executables(runtime)
    attempt = Path(manifest["attempt_dir"])
    meta = attempt.lstat()
    worker_meta = (attempt / "worker").lstat()
    if (attempt.resolve(strict=True) != attempt or not stat.S_ISDIR(meta.st_mode)
            or meta.st_uid != 0 or meta.st_gid != 65534 or stat.S_IMODE(meta.st_mode) != 0o750
            or worker_meta.st_uid != 65534 or stat.S_IMODE(worker_meta.st_mode) != 0o700
            or any((attempt / "worker").iterdir()) or (attempt / "parent").exists()
            or (attempt / "consumed.json").exists() or (attempt / "launch_claim.json").exists()):
        raise ValueError("fresh_unconsumed_prepared_attempt_required")
    verify_private_directories(attempt, spec["runtime_dir"])
    if Path(spec["parent_log"]) != attempt / "native_stdout_stderr.log":
        raise ValueError("unexpected_worker_log_path")
    log = Path(spec["parent_log"]).lstat()
    if (not stat.S_ISREG(log.st_mode) or log.st_uid != 0 or log.st_gid != 65534
            or log.st_nlink != 1 or log.st_size != 0 or stat.S_IMODE(log.st_mode) != 0o640):
        raise ValueError("fresh_root_owned_empty_worker_log_required")
    if require_approval:
        verify_approval(manifest)
    return spec, manifest, bundle


def verify_shutdown(spec, manifest, record):
    root = Path(spec["runtime_dir"]) / "bootstrap"
    if any((root/name).exists() for name in ("worker_failure.json","worker_close_failure.json")):
        raise ValueError("worker_recorded_execution_or_cleanup_failure")
    path = root / "worker_preclose_ready.json"
    value,_ = _artifact(path)
    expected = {"schema":"robodojo-native-worker-preclose-1", "ready_for_shutdown":True,
        "no_pending_exception":True, "native_env_close_returned":True, "physx_monitor_healthy":True,
        "cleanup_errors":[], "child_pid":record["worker_identity"]["pid"], "binding":spec["gpu_binding"],
        "shutdown_mode":"stock-fast", "full_teardown_returned":False, "full_plugin_cleanup_proven":False,
        "model_calls":0, "benchmark_attempt_started":True, "asset_fidelity_passed":False,
        "dependency_closure_complete":False}
    if any(type(value.get(k)) is not type(v) or value.get(k) != v for k,v in expected.items()):
        raise ValueError("worker_preclose_failed")
    terminal = Path(manifest["attempt_dir"]) / "worker/terminal.json"
    for key, target in (("terminal_sha256", terminal), ("preparation_sha256", root / "preparation.json"),
                        ("worker_isolation_sha256", root / "worker_isolation.json")):
        if value.get(key) != file_sha256(target): raise ValueError("worker_preclose_evidence_changed")
    preparation,_ = _artifact(root / "preparation.json")
    if (preparation.get("schema") != "robodojo-native-worker-preparation-1"
            or preparation["manifest_sha256"] != manifest["manifest_sha256"]
            or preparation["worker_identity"] != {"pid":record["worker_identity"]["pid"], "boot_nonce":spec["boot_nonce"]}
            or preparation["runtime_bundle_sha256"] != file_sha256(spec["runtime_bundle_path"])):
        raise ValueError("worker_preparation_binding_changed")
    proof = json.loads((root / "worker_isolation.json").read_text())
    if (proof.get("pid") != record["worker_identity"]["pid"] or proof.get("uid") != 65534
            or proof.get("gid") != 65534 or proof.get("supplementary_groups") != []
            or proof.get("no_new_privs") is not True
            or proof.get("effective_permitted_inheritable_ambient_capabilities") != 0
            or proof.get("access_denied") != {"credential_directory":True,"parent_environment":True}):
        raise ValueError("worker_privilege_boundary_unverified")
    if value["native_action_count"] != [record["post_reap_reconciliation"]["verified_native_actions"]]:
        raise ValueError("worker_preclose_action_count_differs")
    supporting = verify_supporting_runtime_evidence(spec,manifest,record,value,preparation)
    return {"path":str(path), "sha256":file_sha256(path), "verified":True,
            "supporting_artifacts":supporting,"private_runtime_paths_verified":True,
            "runtime_bundle_module_and_graphics_provenance_verified":True,
            "full_plugin_cleanup_proven":False}


def infrastructure_completed(record):
    """Process success is distinct from native task success."""
    return bool(record.get("returncode") == 0 and record.get("termination_signals") == []
        and record.get("remaining_owned_pids") == [] and record.get("gpu_processes_released") is True
        and record.get("gpu_memory_released") is True and record.get("frozen_inputs_unchanged") is True
        and record.get("post_reap_reconciliation", {}).get("owned_worker_reaped_verified") is True
        and record.get("worker_shutdown", {}).get("verified") is True
        and all(record.get(name, {}).get("available") is True for name in ("video", "trajectory"))
        and not any(key.endswith("_error") or key.endswith("_error_type") or key == "error_type"
                    for key,value in record.items() if value is not None))


class ProcessMonitor:
    """Own only the launched PID tree. Never signal unknown GPU processes."""
    def __init__(self, proc, identity, binding, output, before, deadline):
        self.proc, self.identity, self.binding = proc, identity, binding
        self.known = {proc.pid: identity["start_ticks"]}
        self.output, self.deadline = Path(output), deadline
        self.stop = threading.Event()
        self.error = None
        self.peak = int(before["memory_used"].split()[0])
        self.minimum_free = int(before["memory_free"].split()[0])
        self.namespace_candidate = None
        self.thread = threading.Thread(target=self._run, name="native-process-monitor", daemon=True)

    def _run(self):
        try:
            with (self.output / "gpu_monitor.jsonl").open("x") as stream:
                while not self.stop.is_set():
                    members = owned_processes(self.proc.pid, self.known)
                    snapshot = gpu_snapshot(timeout_s=10)
                    device = validate_single_gpu(snapshot, self.binding)
                    code = self.proc.poll()
                    stream.write(json.dumps({"monotonic_s": time.monotonic(), "child_returncode": code,
                        "owned_processes": members, "gpus": snapshot}) + "\n"); stream.flush()
                    self.peak = max(self.peak, int(device["memory_used"].split()[0]))
                    self.minimum_free = min(self.minimum_free, int(device["memory_free"].split()[0]))
                    if code is not None: return
                    aliases = gpu_process_aliases(self.known, child_identity=self.identity, child_unreaped=True)
                    unknown = [row for row in device["processes"] if row["pid"] not in aliases]
                    if unknown:
                        ids = {row["pid"] for row in device["processes"]}
                        candidate = unknown[0]["pid"]
                        if (len(ids) != 1 or process_record(candidate) is not None
                                or self.namespace_candidate not in (None, candidate)):
                            raise RuntimeError("unexpected_or_ambiguous_gpu_occupancy")
                        self.namespace_candidate = candidate
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError("approved_attempt_wall_time_reached")
                    self.stop.wait(.5)
        except BaseException as error:
            self.error = type(error).__name__
            # Wake the parent's main thread even during bounded model I/O.
            os.kill(os.getpid(), signal.SIGTERM)

    def tracked(self, expected):
        record = process_record(self.proc.pid)
        return bool(not self.error and expected["pid"] == self.proc.pid and self.proc.poll() is None
                    and record and record["start_ticks"] == self.identity["start_ticks"])

    def finish(self):
        self.stop.set(); self.thread.join(timeout=15)
        if self.thread.is_alive(): raise RuntimeError("gpu_monitor_did_not_stop")


def run(spec_path):
    if os.getuid() != 0 or os.environ.get("UC71_STANDARD_SUPERVISOR") != "1":
        raise ValueError("private_root_supervisor_required")
    spec, manifest, bundle = preflight(spec_path, require_approval=True)
    attempt, runtime = Path(manifest["attempt_dir"]), manifest["configuration"]["runtime"]
    # Exclusive launch claim prevents a pre-BEGIN crash being silently retried.
    write_private_record(attempt / "launch_claim.json", {"manifest_sha256": manifest["manifest_sha256"],
        "approval_sha256": verify_approval(manifest), "parent_pid": os.getpid(), "automatic_retry": False})
    before = gpu_snapshot(); device = validate_single_gpu(before, spec["gpu_binding"])
    if device["processes"] or int(device["memory_free"].split()[0]) < runtime["minimum_free_mib"]:
        raise ValueError("approved_gpu_must_be_idle_with_required_headroom")
    write_private_record(attempt / "gpu_before.json", {"devices": before})
    environment = child_environment(Path(runtime["python"]), Path(spec["runtime_dir"]),
        spec["gpu_binding"], Path(runtime["icd"]), Path(runtime["library_path"]), "stock-fast")
    environment = {k:v for k,v in environment.items() if not k.startswith("UC71_CAMERA_")}
    environment.update(PYTHONPATH=runtime["compat_target"] + os.pathsep + runtime["overlay_target"],
        GIT_OPTIONAL_LOCKS="0", PYTHONUNBUFFERED="1", TMPDIR=str(Path(spec["runtime_dir"]) / "tmp"))
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    argv = [runtime["python"], "-B", "-X", "faulthandler", "-u", runtime["worker_script"],
        "--launch", "--spec", str(Path(spec_path).resolve()), "--ipc-fd", str(right.fileno()),
        "--parent-pid", str(os.getpid())]
    started = time.monotonic(); proc = monitor = client = runner = identity = model = None
    consumed_binding = expected_worker = None
    record = {"schema": "robodojo-standard-parent-launch-1", "manifest_sha256": manifest["manifest_sha256"],
        "argv": argv, "sanitized_worker_environment": environment, "automatic_retry": False,
        "started_monotonic_s":started,
        "model": manifest["configuration"]["model"], "full_plugin_cleanup_proven": False}
    handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError("parent_stop_signal_" + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupted)
    try:
        with open_worker_log(spec["parent_log"]) as log:
            proc = subprocess.Popen(argv, cwd=attempt, env=environment, stdout=log,
                stderr=subprocess.STDOUT, pass_fds=(right.fileno(),), **popen_identity())
            right.close()
            identity = process_record(proc.pid)
            if not identity: raise RuntimeError("owned_worker_identity_unavailable")
            expected_worker = {"pid": proc.pid, "boot_nonce": spec["boot_nonce"]}
            record["worker_identity"] = identity
            write_private_record(attempt / "launcher_pid.json", {"launcher_pid": os.getpid(), "child": identity})
            monitor = ProcessMonitor(proc, identity, spec["gpu_binding"], attempt, device,
                                     started + runtime["maximum_wall_time_s"])
            monitor.thread.start()
            channel = JsonLineChannel(left, role="parent", timeout_s=runtime["ready_timeout_s"])
            client = NativeWorkerClient(channel, manifest=manifest, expected_worker=expected_worker,
                verify_tracked_worker=monitor.tracked, initial_gripper_commands={"left": "open", "right": "open"})
            ready = client.ready(); channel.timeout_s = 30.
            if not monitor.tracked(expected_worker): raise RuntimeError("worker_exited_before_consumption")
            consumed = consume_authorization(manifest, attempt, ready, expected_worker=expected_worker)
            share_approval_record(attempt, "consumed.json")
            consumed_binding = {"approval_sha256": consumed["approval_sha256"],
                "consumed_sha256": file_sha256(attempt / "consumed.json"), "claim_nonce": consumed["claim_nonce"],
                "ready_sha256": consumed["ready_sha256"]}
            from robodojo_uc71.client import NativeExecutiveClient
            from robodojo_uc71.native_episode_runner import NativeEpisodeRunner
            from gpt_field.astra import AstraConfig
            from gpt_field.chatgpt_auth import SubscriptionCredentials
            configuration = manifest["configuration"]
            credentials = SubscriptionCredentials('/workspace/gpt_field/.private/chatgpt_plan')
            model = NativeExecutiveClient(AstraConfig(model="gpt-6-astra", timeout_s=180.),
                task_prompt=configuration["task_prompt"], token_provider=credentials.access_token,
                approved_plan=configuration["approved_plan"], direction_frame=configuration["direction_frame"],
                limits=bundle.execution_limits, controller_capabilities=bundle.controller_capabilities,
                log_dir=client.output / "subscription_logs", reasoning_effort="high", request_deadline_s=180)
            runner = NativeEpisodeRunner(client, model_client=model, monitoring=bundle.monitoring_settings,
                controller_config=bundle.controller_config, limits=bundle.execution_limits,
                stop_requested=lambda: (attempt / "STOP").exists() or monitor.error is not None)
            runner.begin(consumed_binding)
            while runner.lifecycle.state == "RUNNING":
                outcome = runner.run_next()
                print(json.dumps({"phase": "decision_finished", "state": outcome["state"],
                    "confirmed_native_actions": runner.lifecycle.confirmed_actions,
                    "elapsed_s": time.monotonic() - started}), flush=True)
            record["parent_terminal"] = runner.lifecycle.terminal_record()
            proc.wait(timeout=60)
    except BaseException as error:
        record["error_type"] = type(error).__name__
        # Exceptions may contain external/model text; raw transport evidence
        # stays in private logs rather than being sent to the robot or model.
    finally:
        for sig in handlers: signal.signal(sig, signal.SIG_IGN)
        right.close()
        if runner is not None:
            try: record["parent_terminal"] = runner.close()
            except BaseException as error: record["parent_close_error"] = type(error).__name__
        elif client is not None:
            client.abort()
        left.close()
        # Let EOF/close finish the single in-flight action and durable cleanup
        # before escalating to owned-process signals. There is no resend/reset.
        if proc is not None and proc.poll() is None:
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: record["graceful_worker_exit_timed_out"] = True
        if monitor is not None:
            try: monitor.finish()
            except BaseException as error: record["monitor_close_error"] = type(error).__name__
        known = monitor.known if monitor else ({proc.pid: identity["start_ticks"]} if proc and identity else {})
        if proc is not None:
            try:
                record["termination_signals"] = terminate_owned(proc, known) if proc.poll() is None or owned_processes(proc.pid, known) else []
            except BaseException as error: record["termination_error"] = type(error).__name__
            try: record["returncode"] = proc.wait(timeout=15)
            except subprocess.TimeoutExpired: record["reap_failed"] = True
            record["remaining_owned_pids"] = [row["pid"] for row in owned_processes(proc.pid, known)]
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    record["elapsed_attempt_s"] = time.monotonic() - started
    record["benchmark_attempt_started"] = consumed_binding is not None
    if model is not None: record["model_usage"] = model.budget_snapshot()
    if monitor:
        record.update(sampled_peak_gpu_used_mib=monitor.peak, sampled_minimum_free_mib=monitor.minimum_free,
            monitor_error=monitor.error, nvml_namespace_candidate=monitor.namespace_candidate)
    try:
        after = gpu_snapshot(); deadline = time.monotonic() + 10
        while any(row["processes"] for row in after) and time.monotonic() < deadline:
            time.sleep(.5); after = gpu_snapshot(timeout_s=3)
        final = validate_single_gpu(after, spec["gpu_binding"])
        record["gpu_processes_released"] = not final["processes"]
        record["gpu_memory_released"] = int(final["memory_used"].split()[0]) - int(device["memory_used"].split()[0]) <= 256
        write_private_record(attempt / "gpu_after.json", {"devices": after})
    except BaseException as error: record["gpu_release_check_error"] = type(error).__name__
    if consumed_binding and proc and proc.returncode is not None and not record.get("remaining_owned_pids"):
        try:
            from robodojo_uc71.native_reconciliation import reconcile_after_reap
            record["post_reap_reconciliation"] = reconcile_after_reap(manifest=manifest,
                expected_worker=expected_worker, expected_consumed_binding=consumed_binding,
                verify_owned_worker_gone=lambda _: proc.returncode is not None and not owned_processes(proc.pid, known))
        except BaseException as error: record["reconciliation_error"] = type(error).__name__
    try: record["worker_shutdown"] = verify_shutdown(spec, manifest, record)
    except BaseException as error: record["worker_shutdown_error"] = type(error).__name__
    finalized = attempt / "finalized"
    finalized.mkdir(mode=0o700)
    from robodojo_uc71.attempt_evidence import finalize_video, finalize_trajectory
    for name, function in (("video", finalize_video), ("trajectory", finalize_trajectory)):
        try: record[name] = function(attempt, finalized)
        except BaseException as error: record[name] = {"available": False, "error_type": type(error).__name__}
    try:
        verify_manifest(manifest); bundle.verify(); record["frozen_inputs_unchanged"] = True
    except BaseException as error: record["frozen_input_check_error"] = type(error).__name__
    record["infrastructure_completed"] = infrastructure_completed(record)
    write_private_record(attempt / "launch_result.json", record)
    print(json.dumps({"phase": "attempt_finished", "result": str(attempt / "launch_result.json"),
                      "automatic_retry": False}), flush=True)
    return 0 if record["infrastructure_completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.launch:
        spec, manifest, bundle = preflight(args.spec)
        print(json.dumps({"inputs_verified": True, "manifest_sha256": manifest["manifest_sha256"],
                          "simulator_starts": 0, "model_calls": 0, "full_test_authorized": False}))
        return 0
    return run(args.spec)


if __name__ == "__main__":
    raise SystemExit(main())
