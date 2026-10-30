#!/usr/bin/env python3
"""One bounded, isolated new-host camera check; never a benchmark launcher.

Explicit bindings select one GPU even on a multi-GPU host. This preserves old A0/A1 evidence and
requires a fresh output directory. No automatic retry, task import or model call.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

from new_host_camera import normalize_pci

BASE = Path(__file__).resolve().parents[1]


def gpu_snapshot(*, timeout_s: float = 15) -> list[dict]:
    raw = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=timeout_s)
    root = ET.fromstring(raw)
    devices = []
    for index, item in enumerate(root.findall("gpu")):
        processes = []
        for proc in item.findall("processes/process_info"):
            mem = proc.findtext("used_memory", "")
            processes.append({"pid": int(proc.findtext("pid")), "type": proc.findtext("type"),
                              "process_name": proc.findtext("process_name"),
                              "used_memory_mib": int(mem.split()[0]) if mem.split() and mem.split()[0].isdigit() else None})
        devices.append({"index": index, "uuid": item.findtext("uuid"), "pci_bus_id": item.findtext("pci/pci_bus_id"),
                        "driver_version": root.findtext("driver_version"),
                        "memory_total": item.findtext("fb_memory_usage/total"),
                        "memory_free": item.findtext("fb_memory_usage/free"),
                        "memory_used": item.findtext("fb_memory_usage/used"), "processes": processes})
    return devices


def process_record(pid: int) -> dict | None:
    """Use kernel start ticks to avoid confusing a reused PID with our child."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        ns = next((line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")), [])
        return {"pid": pid, "ppid": int(tail[1]), "pgrp": int(tail[2]), "start_ticks": int(tail[19]),
                "nspid": [int(n) for n in ns], "state": tail[0]}
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
        return None


def owned_processes(group: int, known: dict[int, int]) -> list[dict]:
    """Find descendants even when a debugger gives the inferior another pgrp."""
    records = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            record = process_record(int(entry.name))
            if record is not None:
                records[record["pid"]] = record
    owned = {pid for pid, ticks in known.items() if pid in records and records[pid]["start_ticks"] == ticks}
    # The group itself is only meaningful while a known process still belongs to it.
    group_alive = any(records[pid]["pgrp"] == group for pid in owned)
    changed = True
    while changed:
        additions = {pid for pid, rec in records.items() if rec["ppid"] in owned or (group_alive and rec["pgrp"] == group)} - owned
        owned.update(additions)
        changed = bool(additions)
    for pid in owned:
        known[pid] = records[pid]["start_ticks"]
    return [records[pid] for pid in sorted(owned) if records[pid]["state"] != "Z"]


def gpu_process_aliases(known: dict[int, int], *, child_identity: dict,
                        child_unreaped: bool, process_lookup=process_record) -> set[int]:
    """Retain verified identities through exit, while rejecting actual PID reuse.

    Cleanup enumerates live processes only, but GPU accounting may outlive that
    live state. A same-start-tick zombie remains ours. If /proc has disappeared,
    only our direct unreaped child retains its original aliases: its PID cannot
    be reused until waitpid reaps it. Never extend a disappeared descendant.
    """
    aliases = set()
    for pid, ticks in known.items():
        current = process_lookup(pid)
        if current is not None:
            if current["start_ticks"] == ticks:
                aliases.update((pid, *current["nspid"]))
        elif (child_unreaped and pid == child_identity["pid"]
              and ticks == child_identity["start_ticks"]):
            aliases.update((pid, *child_identity["nspid"]))
    return aliases


def terminate_owned(proc: subprocess.Popen, known: dict[int, int]) -> list[dict]:
    sent = []
    def signal_current(sig):
        current = owned_processes(proc.pid, known)
        # Stop descendants before their parent; retained identities survive reparenting.
        for rec in sorted(current, key=lambda r: r["pid"] == proc.pid):
            verified = process_record(rec["pid"])
            if verified is not None and verified["start_ticks"] == rec["start_ticks"]:
                try:
                    os.kill(rec["pid"], sig)
                    sent.append({"pid": rec["pid"], "start_ticks": rec["start_ticks"],
                                 "signal": int(sig), "monotonic_s": time.monotonic()})
                except ProcessLookupError:
                    pass
    if not known and proc.poll() is None:
        # Popen is our unreaped direct child; no PID can be reused during this lifetime.
        # Preserve descendants in its freshly created session when /proc is unavailable.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if proc.poll() is not None:
                break
            try:
                if os.getpgid(proc.pid) == proc.pid:
                    os.killpg(proc.pid, sig)
                else:
                    proc.send_signal(sig)
                sent.append({"pid": proc.pid, "start_ticks": None, "signal": int(sig),
                             "basis": "unreaped directly-owned Popen child", "monotonic_s": time.monotonic()})
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                break
        return sent
    signal_current(signal.SIGTERM)
    end = time.monotonic() + 10
    while owned_processes(proc.pid, known) and time.monotonic() < end:
        proc.poll()
        time.sleep(0.1)
    signal_current(signal.SIGKILL)
    proc.wait(timeout=10)
    return sent


def classify_terminal_exit(returncode: int, monitor_reason: str | None, sent_signals: list[dict], *, child_pid: int) -> dict:
    """An already-dead child takes precedence over stale NVML bookkeeping."""
    child_signals = [event for event in sent_signals if event["pid"] == child_pid]
    value = {"returncode": returncode, "signal": -returncode if returncode < 0 else None,
             "guard_signals_sent": bool(sent_signals), "child_signals_sent": bool(child_signals)}
    if returncode < 0 and not child_signals:
        value["kind"] = "child_signal_exit"
        value["post_exit_monitor_observation"] = monitor_reason
    elif child_signals:
        value["kind"] = "monitor_initiated_shutdown"
        value["monitor_reason"] = monitor_reason
    else:
        value["kind"] = "child_completed" if returncode == 0 else "child_error_exit"
    return value


def validate_single_gpu(snapshot: list[dict], binding: dict, *, require_idle: bool = False) -> dict:
    # NVML enumerates the whole host; renderer/CUDA identity is checked separately.
    # Keep the historical function name for callers and archived records.
    matches = [gpu for gpu in snapshot if gpu["uuid"] == binding["gpu_uuid"]]
    if len(matches) != 1 or len({gpu["uuid"] for gpu in snapshot}) != len(snapshot):
        raise ValueError("Exactly one matching GPU UUID is required")
    gpu = matches[0]
    if (gpu["index"] != 0
            or normalize_pci(gpu["pci_bus_id"]) != normalize_pci(binding["pci_bus_id"])
            or binding["kit_index"] != 0 or binding["cuda_ordinal"] != 0):
        raise ValueError("Explicit single-GPU identity, PCI address, or ordinal binding differs")
    if require_idle and gpu["processes"]:
        raise ValueError("The diagnostic requires the selected GPU to have no active processes")
    return gpu


def child_environment(python: Path, output: Path, binding: dict, icd: Path,
                      library_path: Path | None, shutdown_mode: str = "full") -> dict[str, str]:
    # Explicit allowlist. No provider credentials, inherited CUDA mask or Python path.
    environment = {key: os.environ[key] for key in ("LANG", "LC_ALL", "TZ") if key in os.environ}
    environment.update({"HOME": str(output / "home"),
        "PATH": str(python.parent) + ":/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VK_ICD_FILENAMES": str(icd), "OMNI_KIT_ACCEPT_EULA": "YES",
        "UC71_CAMERA_AUTHORIZED_CHILD": "1", "UC71_CAMERA_OUTPUT": str(output),
        "UC71_CAMERA_BINDING": json.dumps(binding), "UC71_CAMERA_SHUTDOWN_MODE": shutdown_mode,
        "XDG_CACHE_HOME": str(output / "cache"), "XDG_CONFIG_HOME": str(output / "config"),
        "XDG_DATA_HOME": str(output / "data"), "CUDA_CACHE_PATH": str(output / "cache/cuda"),
        "WARP_CACHE_PATH": str(output / "cache/warp")})
    if library_path is not None:
        environment["LD_LIBRARY_PATH"] = str(library_path)
    return environment


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def shutdown_evidence(mode: str, result: dict, progress: dict, ready: dict, *,
                      returncode, termination_signals: list, rgb_sha256: str | None,
                      camera_result_sha256: str | None, child_pid: int | None, gpu_uuid: str) -> dict:
    """Exit zero in stock-fast mode cannot replace successful, durable preclose evidence."""
    expected = {"schema": "robodojo-camera-preclose-ready-1", "shutdown_mode": mode,
        "camera_passed": True, "capture_requests": 8, "synchronous_captures_completed": 8,
        "wait_for_render": True, "no_pending_exception": True, "annotator_detached": True,
        "render_product_destroyed": True, "cleanup_errors": [],
        "durability": "file_and_parent_directory_fsync_before_close", "child_pid": child_pid,
        "gpu_uuid": gpu_uuid, "model_calls": 0, "benchmark_actions": 0}
    preclose_ok = bool(rgb_sha256 and camera_result_sha256
        and all(type(ready.get(k)) is type(v) and ready.get(k) == v for k, v in expected.items())
        and ready.get("rgb_sha256") == rgb_sha256
        and ready.get("camera_result_sha256") == camera_result_sha256
        and result.get("passed") is True and result.get("capture_requests") == 8
        and result.get("synchronous_captures_completed") == 8 and result.get("wait_for_render") is True)
    close_returned = progress.get("phase") == "stock_simulation_app_close_returned"
    full_teardown_returned = mode == "full" and close_returned
    valid = (preclose_ok if mode == "stock-fast" else full_teardown_returned if mode == "full" else False)
    valid = bool(valid and returncode == 0 and not termination_signals)
    return {"shutdown_mode": mode, "preclose_ready_verified": preclose_ok,
            "simulation_app_close_returned": close_returned,
            "full_teardown_returned": full_teardown_returned,
            "full_plugin_cleanup_proven": False,
            "shutdown_contract_satisfied": valid}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--kit-index", type=int, required=True)
    parser.add_argument("--cuda-ordinal", type=int, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--install-record", type=Path, required=True)
    parser.add_argument("--eula-record", type=Path, default=BASE / "validation/provisioning/vendor_authorization.json")
    parser.add_argument("--vk-icd", type=Path, default=Path("/etc/vulkan/icd.d/nvidia_icd.json"))
    parser.add_argument("--library-path", type=Path)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--shutdown-mode", choices=("full", "stock-fast"), default="full",
                        help="full attempts plugin teardown; stock-fast uses NVIDIA default process exit after durable preclose validation")
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    args = parser.parse_args()
    # Keep the venv path itself: resolving its interpreter symlink would lose the venv.
    python = args.python.absolute()
    output = args.output.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error("--python must name an existing executable interpreter")
    if not 30 <= args.timeout_s <= 1200 or args.minimum_free_mib < 16384:
        parser.error("Diagnostic deadline must be 30..1200s; memory minimum must be at least 16384MiB")
    if not args.vk_icd.is_file():
        parser.error("The explicitly selected Vulkan ICD file is missing")
    library_path = args.library_path.resolve() if args.library_path else None
    if library_path is not None and (not library_path.is_dir() or not library_path.is_relative_to(BASE / "runtime")):
        parser.error("--library-path must be an existing task-local runtime directory")
    binding = {"gpu_uuid": args.gpu_uuid, "pci_bus_id": args.pci_bus_id,
               "kit_index": args.kit_index, "cuda_ordinal": args.cuda_ordinal}
    auth = json.loads(args.authorization.read_text())
    eula = json.loads(args.eula_record.read_text())
    expected = {"diagnostic_authorized": True, "full_test_authorized": False,
                "model_calls_authorized": False, "gpu_uuid": args.gpu_uuid}
    if any(type(auth.get(k)) is not type(v) or auth.get(k) != v for k, v in expected.items()) or not auth.get("user_message"):
        parser.error("A new-host setup scope record bound to this GPU is required; full tests/model calls must remain false")
    if eula.get("nvidia_eula_accepted") is not True:
        parser.error("The preserved NVIDIA licence acceptance is missing")
    installed = json.loads(args.install_record.read_text())
    if installed.get("schema") == "robodojo-conda-runtime-1":
        from robodojo_uc71.host_runtime import verify_conda_record
        verify_conda_record(python, args.install_record)
    lock = args.install_record.with_name("requirements.lock")
    inventory_path = args.install_record.with_name("package_inventory.json")
    if (installed.get("ready_for_diagnostic") is not True or Path(installed["python"]).absolute() != python
            or hashlib.sha256(lock.read_bytes()).hexdigest() != installed.get("lock_sha256")):
        parser.error("Completed install record, exact interpreter and pinned lock binding are required")
    metadata_command = [str(python), "-I", "-c",
        "import importlib.metadata as m,json,sys; print(json.dumps({'python_version':list(sys.version_info[:3]),'packages': [{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()]}))"]
    metadata = json.loads(subprocess.check_output(metadata_command, text=True, timeout=20,
        env={"PATH": str(python.parent) + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))
    normalize = lambda rows: {r["name"].lower().replace("_", "-"): r["version"] for r in rows}
    actual_packages = normalize(metadata["packages"])
    if (metadata["python_version"] != [3, 11, 16]
            or actual_packages != normalize(json.loads(inventory_path.read_text())["packages"])
            or actual_packages.get("isaacsim") != "5.1.0.0"
            or actual_packages.get("torch") != "2.7.0+cu128"
            or any(v != "5.1.0.0" for k, v in actual_packages.items() if k.startswith("isaacsim"))):
        parser.error("Current Python/package inventory differs from the recorded pinned installation")
    before = gpu_snapshot()
    gpu = validate_single_gpu(before, binding, require_idle=True)
    free_mib = int(gpu["memory_free"].split()[0])
    if free_mib < args.minimum_free_mib:
        parser.error(f"Only {free_mib}MiB free, below requested diagnostic minimum")
    kernel = Path("/proc/driver/nvidia/version").read_text()
    if not re.search(r"\b" + re.escape(gpu["driver_version"]) + r"\b", kernel):
        parser.error("NVML driver version does not match the injected kernel driver")
    output.mkdir(parents=False, exist_ok=False)  # Never reuse an attempt directory.
    os.chmod(output, 0o700)
    for directory in ("home", "cache", "config", "data"):
        (output / directory).mkdir(mode=0o700)
    for source, name in ((args.authorization, "setup_authorization.json"),
                         (args.eula_record, "vendor_authorization.json"),
                         (Path(__file__), "provision_new_host_camera.py"),
                         (Path(__file__).with_name("new_host_camera.py"), "new_host_camera.py")):
        (output / name).write_bytes(source.read_bytes())
    write_json(output / "gpu_before.json", before)
    write_json(output / "package_inventory.json", metadata)
    (output / "install_status.json").write_bytes(args.install_record.read_bytes())
    (output / "requirements.lock").write_bytes(lock.read_bytes())
    (output / "kernel_driver.txt").write_text(kernel)
    environment = child_environment(python, output, binding, args.vk_icd.resolve(), library_path, args.shutdown_mode)
    child = output / "new_host_camera.py"
    argv = [str(python), "-X", "faulthandler", "-u", str(child)]
    record = {"schema": "robodojo-new-host-camera-1",
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": argv, "environment": environment, "binding": binding,
        "wall_time_limit_s": args.timeout_s, "minimum_free_mib": args.minimum_free_mib,
        "shutdown_mode": args.shutdown_mode,
        "shutdown_mode_limitation": "Stock fast shutdown skips full plugin/Python teardown; preclose evidence and process/GPU release are required, not proof of full teardown.",
        "setup_authorization_sha256": hashlib.sha256(args.authorization.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "install_record_sha256": hashlib.sha256(args.install_record.read_bytes()).hexdigest(),
        "lock_sha256": installed["lock_sha256"],
        "precreated_primary_cuda_context": False,
        "identity_method": "NVML/kernel driver, CUDA metadata UUID/PCI, Kit renderer index/UUID prefix/PCI bus; no context bootstrap",
        "nvml_namespace_note": "A lone inaccessible NVML PID may be a host-namespace candidate; temporal association is not PID attestation. Final empty GPU process list is required.",
        "automatic_retry": False, "full_test_authorized": False,
        "model_calls": 0, "benchmark_actions": 0,
        "limitations": "Environment filtering is not filesystem or network isolation. Camera diagnostic does not exercise native RoboDojo."}
    write_json(output / "launch.json", record)
    started = time.monotonic()
    known: dict[int, int] = {}
    termination_signals = []
    observed_libraries = set()
    previous_handlers = {}

    def interrupt(signum, _frame):
        raise InterruptedError(f"Launcher received signal {signum}")

    proc = None
    namespace_candidate = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        with (output / "stdout_stderr.log").open("x") as log, (output / "gpu_monitor.jsonl").open("x") as monitor:
            proc = subprocess.Popen(argv, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            record["pid"] = proc.pid
            initial = process_record(proc.pid)
            if initial is None:
                raise RuntimeError("Child exited before its kernel process identity could be recorded")
            known[proc.pid] = initial["start_ticks"]
            write_json(output / "launcher_pid.json", {"launcher_pid": os.getpid(), "child": initial})
            while proc.poll() is None:
                if time.monotonic() - started >= args.timeout_s:
                    raise TimeoutError("Bounded diagnostic wall-time deadline reached")
                members = owned_processes(proc.pid, known)
                snapshot = gpu_snapshot(timeout_s=min(10, max(0.1, args.timeout_s - (time.monotonic() - started))))
                exit_code = proc.poll()
                monitor.write(json.dumps({"elapsed_s": time.monotonic() - started,
                    "child_returncode": exit_code, "owned_processes": members, "gpus": snapshot}) + "\n")
                monitor.flush()
                if exit_code is not None:
                    break  # NVML entries may drain after process death; preserve actual exit.
                current_gpu = validate_single_gpu(snapshot, binding)
                aliases = gpu_process_aliases(known, child_identity=initial,
                    child_unreaped=proc.returncode is None)
                unknown = [p for p in current_gpu["processes"] if p["pid"] not in aliases]
                if unknown and proc.poll() is None:
                    # NVML can expose C and C+G rows for the same process.
                    if (len({p["pid"] for p in current_gpu["processes"]}) != 1
                            or len({p["pid"] for p in unknown}) != 1
                            or process_record(unknown[0]["pid"]) is not None):
                        raise RuntimeError(f"Unexpected or ambiguous GPU process occupancy: {unknown}")
                    candidate = unknown[0]["pid"]
                    if namespace_candidate is not None and namespace_candidate != candidate:
                        raise RuntimeError("The inaccessible NVML process candidate changed during the same attempt")
                    namespace_candidate = candidate
                    record["nvml_namespace_candidate"] = {
                        "pid": candidate, "binding_proven": False,
                        "basis": "single newly active inaccessible NVML PID on initially idle sole GPU; final release required"}
                for member in members:
                    try:
                        maps = Path(f"/proc/{member['pid']}/maps").read_text()
                    except (OSError, ProcessLookupError):
                        continue
                    for line in maps.splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and re.search(r"/(?:libcuda|libnvidia|libGLX|libEGL|libvulkan)", fields[5]):
                            observed_libraries.add(fields[5])
                time.sleep(0.5)
            record["returncode"] = proc.wait(timeout=5)
    except BaseException as error:
        record["termination"] = type(error).__name__
        record["termination_detail"] = str(error)
    finally:
        # During cleanup, ignore repeated termination requests so they cannot orphan Kit.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if proc is not None:
            if proc.poll() is None or owned_processes(proc.pid, known):
                try:
                    termination_signals.extend(terminate_owned(proc, known))
                except Exception as error:
                    record["cleanup_error"] = repr(error)
            record["returncode"] = proc.poll()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    record["elapsed_s"] = time.monotonic() - started
    record["termination_signals"] = termination_signals
    record["remaining_owned_pids"] = [p["pid"] for p in owned_processes(proc.pid, known)] if proc else []
    record["loaded_graphics_library_paths"] = sorted(observed_libraries)
    try:
        after = gpu_snapshot()
        # Permit a short NVML drain interval after a clean child exit, never a restart.
        drain_deadline = time.monotonic() + 10
        while any(g["processes"] for g in after) and time.monotonic() < drain_deadline:
            time.sleep(0.5)
            after = gpu_snapshot(timeout_s=3)
        write_json(output / "gpu_after.json", after)
        final_gpu = validate_single_gpu(after, binding)
        record["gpu_processes_released"] = not final_gpu["processes"]
        record["gpu_memory_used_delta_mib"] = int(final_gpu["memory_used"].split()[0]) - int(gpu["memory_used"].split()[0])
        record["gpu_memory_released"] = record["gpu_memory_used_delta_mib"] <= 256
    except Exception as error:
        record["gpu_cleanup_check_error"] = repr(error)
        record["gpu_processes_released"] = False
    evidence = {}
    record["result_read_errors"] = {}
    for filename in ("camera_result.json", "camera_progress.json", "active_renderer_identity.json", "cuda_identity.json", "camera_preclose_ready.json"):
        try:
            value = json.loads((output / filename).read_text())
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            evidence[filename] = value
        except (OSError, ValueError) as error:
            evidence[filename] = {}
            record["result_read_errors"][filename] = str(error)
    result = evidence["camera_result.json"]
    progress = evidence["camera_progress.json"]
    renderer = evidence["active_renderer_identity.json"]
    cuda_identity = evidence["cuda_identity.json"]
    record["camera_render_passed"] = result.get("passed") is True and (output / "rgb.png").is_file()
    def artifact_hash(name):
        try:
            return hashlib.sha256((output / name).read_bytes()).hexdigest()
        except OSError:
            return None
    record.update(shutdown_evidence(args.shutdown_mode, result, progress, evidence["camera_preclose_ready.json"],
        returncode=record.get("returncode"), termination_signals=termination_signals,
        rgb_sha256=artifact_hash("rgb.png"), camera_result_sha256=artifact_hash("camera_result.json"),
        child_pid=record.get("pid"), gpu_uuid=binding["gpu_uuid"]))
    record["clean_shutdown_confirmed"] = bool(record["shutdown_contract_satisfied"]
        and record["gpu_processes_released"] and record.get("gpu_memory_released") is True
        and not record["remaining_owned_pids"])
    record["renderer_identity_verified"] = renderer.get("uuid_prefix_and_pci_bus_match") is True
    record["cuda_identity_verified"] = bool(cuda_identity.get("gpu_uuid") == binding["gpu_uuid"]
        and cuda_identity.get("cuda_ordinal") == binding["cuda_ordinal"]
        and cuda_identity.get("child_pid") == record.get("pid")
        and cuda_identity.get("primary_context_created") is False)
    record["diagnostic_passed"] = bool(record.get("returncode") == 0 and "termination" not in record
        and record["camera_render_passed"] and record["clean_shutdown_confirmed"]
        and not termination_signals
        and record["renderer_identity_verified"] and record["cuda_identity_verified"] and record["gpu_processes_released"]
        and record.get("gpu_memory_released") is True
        and not record["remaining_owned_pids"] and "cleanup_error" not in record)
    record["image_visual_review_required"] = record["camera_render_passed"]
    if proc is not None and record.get("returncode") is not None:
        record["terminal_exit"] = classify_terminal_exit(record["returncode"], record.get("termination"),
                                                         termination_signals, child_pid=proc.pid)
    write_json(output / "launch_result.json", record)
    print(json.dumps({"output": str(output), "diagnostic_passed": record["diagnostic_passed"],
                      "returncode": record.get("returncode"), "termination": record.get("termination")}), flush=True)
    return 0 if record["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
