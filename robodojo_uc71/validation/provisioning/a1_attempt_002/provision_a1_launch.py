#!/usr/bin/env python3
"""One bounded stock camera attempt; no automatic restart or benchmark task."""
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

BASE = Path(__file__).resolve().parents[1]
RUNTIME = BASE / "runtime"
REPORT = BASE / "validation/provisioning"
GPU_UUID = "GPU-e1a460e9-e272-b227-222e-0dbb77c248c5"


def gpu_snapshot() -> list[dict]:
    raw = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=15)
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


def terminate_owned(proc: subprocess.Popen, known: dict[int, int]) -> None:
    def signal_current(sig):
        current = owned_processes(proc.pid, known)
        # Stop descendants before their parent; retained identities survive reparenting.
        for rec in sorted(current, key=lambda r: r["pid"] == proc.pid):
            verified = process_record(rec["pid"])
            if verified is not None and verified["start_ticks"] == rec["start_ticks"]:
                try:
                    os.kill(rec["pid"], sig)
                except ProcessLookupError:
                    pass
    signal_current(signal.SIGTERM)
    end = time.monotonic() + 10
    while owned_processes(proc.pid, known) and time.monotonic() < end:
        proc.poll()
        time.sleep(0.1)
    signal_current(signal.SIGKILL)
    proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=["a1_attempt_001", "a1_attempt_002"], required=True)
    parser.add_argument("--gdb", action="store_true", help="Capture native stacks during this same attempt")
    args = parser.parse_args()
    auth_path = REPORT / "a1_authorization.json"
    auth = json.loads(auth_path.read_text())
    eula = json.loads((REPORT / "vendor_authorization.json").read_text())
    expected = {"diagnostics_scope_reopened": True, "gpu_index": 4, "gpu_uuid": GPU_UUID,
                "host_driver_changes_authorized": False, "model_calls_authorized": False,
                "full_make_toast_test_authorized": False}
    if any(auth.get(k) != v for k, v in expected.items()) or eula.get("nvidia_eula_accepted") is not True:
        raise SystemExit("Scoped renewed diagnostic and prior vendor authorization required")
    if args.attempt == "a1_attempt_002":
        repair = json.loads((REPORT / "a1_attempt_002_authorization.json").read_text())
        if repair.get("authorized_attempt") != args.attempt or repair.get("automatic_retry") is not False or args.gdb:
            raise SystemExit("Attempt002 requires the reviewed direct-launch ownership repair; no debugger")
    install = json.loads((REPORT / "a1_install_status.json").read_text())
    if install.get("ready_for_diagnostic") is not True:
        raise SystemExit("Stock isolated install has not passed dependency verification")
    python = RUNTIME / "a1_env/bin/python"
    before = gpu_snapshot()
    selected = [gpu for gpu in before if gpu["uuid"] == GPU_UUID]
    if len(selected) != 1 or selected[0]["index"] != 4 or selected[0]["processes"]:
        raise SystemExit("Authorized physical GPU4 is unavailable or already in use")
    # Unmasked Kit CUDA/Vulkan interop creates contexts on enumerated GPUs.
    # This bounded diagnostic is permitted only while every GPU is idle.
    if any(gpu["processes"] for gpu in before):
        raise SystemExit("Unmasked stock diagnostic requires all GPUs idle to avoid interfering with other work")
    output = REPORT / args.attempt
    output.mkdir()  # Exact one-attempt guard; never overwrite a preceding attempt.
    for source in [auth_path, REPORT / "vendor_authorization.json", REPORT / "a1_requirements.in",
                   REPORT / "a1_requirements.lock", REPORT / "a1_install_status.json",
                   REPORT / "a1_wheel_sources.json", REPORT / "a1_installed_package_inventory.json",
                   BASE / "scripts/provision_a1_camera.py", Path(__file__)]:
        (output / source.name).write_bytes(source.read_bytes())
    if args.attempt == "a1_attempt_002":
        (output / "a1_attempt_002_authorization.json").write_bytes((REPORT / "a1_attempt_002_authorization.json").read_bytes())
    (output / "gpu_before.json").write_text(json.dumps(before, indent=2) + "\n")
    environment = {k: os.environ[k] for k in ("LANG", "LC_ALL", "TZ") if k in os.environ}
    # Dedicated HOME and caches also keep inherited provider config out of reach by default.
    home = RUNTIME / "a1_home"
    home.mkdir(exist_ok=True)
    environment.update({"HOME": str(home), "PATH": str(python.parent) + ":/usr/local/bin:/usr/bin:/bin",
                        "LD_LIBRARY_PATH": str(RUNTIME / "sysroot/usr/lib/x86_64-linux-gnu") + ":/usr/lib/x86_64-linux-gnu",
                        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "OMNI_KIT_ACCEPT_EULA": "YES",
                        "UC71_A1_AUTHORIZED_CHILD": "1", "UC71_A1_OUTPUT": str(output),
                        "XDG_CACHE_HOME": str(RUNTIME / "a1_cache"), "XDG_CONFIG_HOME": str(RUNTIME / "a1_config"),
                        "XDG_DATA_HOME": str(RUNTIME / "a1_data"), "CUDA_CACHE_PATH": str(RUNTIME / "a1_cache/cuda"),
                        "WARP_CACHE_PATH": str(RUNTIME / "a1_cache/warp")})
    child = [str(python), "-X", "faulthandler", "-u", str(BASE / "scripts/provision_a1_camera.py")]
    argv = child
    if args.gdb:
        argv = ["gdb", "--nx", "--batch", "--return-child-result", "-q", "-ex", "set pagination off",
                "-ex", "set confirm off", "-ex", "set print thread-events off", "-ex", "set debuginfod enabled off",
                "-ex", "set print frame-arguments none", "-ex", "set print elements 20",
                "-ex", "run", "-ex", f"set logging file {output / 'native_backtrace.txt'}",
                "-ex", "set logging enabled on", "-ex", "thread apply all bt 16",
                "-ex", "set logging enabled off", "--args", *child]
    record = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "argv": argv,
              "environment": environment, "gpu_uuid": GPU_UUID, "wall_time_limit_s": 600,
              "non_target_process_allocation_abort_mib": 768,
              "non_target_context_allowance": "At most 768 MiB per idle non-target GPU for enumerated CUDA/Vulkan contexts; render and physics settings remain GPU4. Abort on any GPU process whose ownership cannot be established.",
              "authorization_sha256": hashlib.sha256(auth_path.read_bytes()).hexdigest(),
              "automatic_retry": False, "model_calls": 0, "benchmark_actions": 0}
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    seen_libs = set()
    monitoring_error = None
    with (output / "stdout_stderr.log").open("x") as log, (output / "gpu_monitor.jsonl").open("x") as gpu_log:
        proc = subprocess.Popen(argv, cwd=RUNTIME, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = proc.pid
        initial = process_record(proc.pid)
        if initial is None:
            proc.wait()
            raise RuntimeError("Child exited before its stable process identity could be recorded")
        known = {proc.pid: initial["start_ticks"]}
        (output / "launcher_pid.json").write_text(json.dumps({"launcher_pid": os.getpid(), "child_group": proc.pid}) + "\n")
        try:
            while proc.poll() is None:
                owned_processes(proc.pid, known)
                snapshot = gpu_snapshot()
                members = owned_processes(proc.pid, known)
                aliases = {alias for rec in members for alias in [rec["pid"], *rec["nspid"]]}
                gpu_log.write(json.dumps({"elapsed_s": time.monotonic() - started, "owned_processes": members, "gpus": snapshot}) + "\n")
                gpu_log.flush()
                for member in members:
                    pid = member["pid"]
                    try:
                        maps = Path(f"/proc/{pid}/maps").read_text()
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
                    (output / f"process_{pid}_maps.txt").write_text(maps)
                    for line in maps.splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and re.search(r"/(?:libcuda|libnvidia|libGLX|libEGL|libvulkan)", fields[5]):
                            seen_libs.add(fields[5])
                wrong = []
                for gpu in snapshot:
                    if gpu["uuid"] == GPU_UUID:
                        continue
                    owned = [p for p in gpu["processes"] if p["pid"] in aliases]
                    if any(p["used_memory_mib"] is None for p in owned) or sum(p["used_memory_mib"] or 0 for p in owned) > 768:
                        wrong.append({"gpu_uuid": gpu["uuid"], "owned_processes": owned})
                external = [{"gpu_uuid": gpu["uuid"], **p} for gpu in snapshot
                            for p in gpu["processes"] if p["pid"] not in aliases]
                if external:
                    record["termination"] = "gpu_process_ownership_unresolved"
                    record["unmapped_gpu_processes"] = external
                    terminate_owned(proc, known)
                    break
                if wrong:
                    record["termination"] = "non_target_gpu_allocation"
                    record["unexpected_allocations"] = wrong
                    terminate_owned(proc, known)
                    break
                if time.monotonic() - started > 600:
                    record["termination"] = "bounded_diagnostic_timeout"
                    terminate_owned(proc, known)
                    break
                time.sleep(1)
        except BaseException as error:
            monitoring_error = type(error).__name__
            record["termination"] = "monitoring_error"
            terminate_owned(proc, known)
        finally:
            record["returncode"] = proc.wait()
    # A crash reporter/debugger must not leave an owned child running.
    remaining = owned_processes(proc.pid, known)
    if remaining:
        terminate_owned(proc, known)
    libraries = []
    for filename in sorted(seen_libs):
        path = Path(filename)
        entry = {"path": filename}
        try:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entry.update({"resolved_path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest})
        except OSError:
            entry["hash_unavailable"] = True
        libraries.append(entry)
    (output / "loaded_graphics_libraries.json").write_text(json.dumps(libraries, indent=2) + "\n")
    record.update({"elapsed_s": time.monotonic() - started, "monitoring_error": monitoring_error,
                   "owned_process_identities": known,
                   "remaining_owned_pids": [r["pid"] for r in owned_processes(proc.pid, known)]})
    try:
        camera_result = json.loads((output / "camera_result.json").read_text())
        progress = json.loads((output / "camera_progress.json").read_text())
    except (OSError, ValueError):
        camera_result, progress = {}, {}
    record["camera_render_passed"] = camera_result.get("passed") is True
    record["clean_shutdown_confirmed"] = progress.get("phase") == "stock_simulation_app_close_returned"
    record["diagnostic_passed"] = bool(record["returncode"] == 0 and "termination" not in record
                                       and record["camera_render_passed"] and record["clean_shutdown_confirmed"]
                                       and not record["remaining_owned_pids"])
    (output / "gpu_after.json").write_text(json.dumps(gpu_snapshot(), indent=2) + "\n")
    (output / "launch_result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return 0 if record["diagnostic_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
