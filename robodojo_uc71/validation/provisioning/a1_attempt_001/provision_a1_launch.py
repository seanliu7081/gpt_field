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
                              "used_memory_mib": int(mem.split()[0]) if mem.split() and mem.split()[0].isdigit() else None})
        devices.append({"index": index, "uuid": item.findtext("uuid"), "pci_bus_id": item.findtext("pci/pci_bus_id"),
                        "memory_used": item.findtext("fb_memory_usage/used"), "processes": processes})
    return devices


def group_members(group: int) -> list[int]:
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if os.getpgid(int(entry.name)) == group:
                    result.append(int(entry.name))
            except (ProcessLookupError, PermissionError):
                pass
    return result


def terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    end = time.monotonic() + 10
    while group_members(proc.pid) and time.monotonic() < end:
        proc.poll()
        time.sleep(0.1)
    if group_members(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=["a1_attempt_001"], required=True)
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
              "non_target_context_allowance": "At most 768 MiB per idle non-target GPU for enumerated CUDA/Vulkan contexts; render and physics settings remain GPU4. Abort if another job appears.",
              "authorization_sha256": hashlib.sha256(auth_path.read_bytes()).hexdigest(),
              "automatic_retry": False, "model_calls": 0, "benchmark_actions": 0}
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    seen_libs = set()
    monitoring_error = None
    with (output / "stdout_stderr.log").open("x") as log, (output / "gpu_monitor.jsonl").open("x") as gpu_log:
        proc = subprocess.Popen(argv, cwd=RUNTIME, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = proc.pid
        (output / "launcher_pid.json").write_text(json.dumps({"launcher_pid": os.getpid(), "child_group": proc.pid}) + "\n")
        try:
            while proc.poll() is None:
                members = group_members(proc.pid)
                snapshot = gpu_snapshot()
                gpu_log.write(json.dumps({"elapsed_s": time.monotonic() - started, "group_members": members, "gpus": snapshot}) + "\n")
                gpu_log.flush()
                for pid in members:
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
                    owned = [p for p in gpu["processes"] if p["pid"] in members]
                    if any(p["used_memory_mib"] is None for p in owned) or sum(p["used_memory_mib"] or 0 for p in owned) > 768:
                        wrong.append({"gpu_uuid": gpu["uuid"], "owned_processes": owned})
                external = [{"gpu_uuid": gpu["uuid"], **p} for gpu in snapshot
                            for p in gpu["processes"] if p["pid"] not in members]
                if external:
                    record["termination"] = "external_gpu_job_appeared"
                    record["external_allocations"] = external
                    terminate_group(proc)
                    break
                if wrong:
                    record["termination"] = "non_target_gpu_allocation"
                    record["unexpected_allocations"] = wrong
                    terminate_group(proc)
                    break
                if time.monotonic() - started > 600:
                    record["termination"] = "bounded_diagnostic_timeout"
                    terminate_group(proc)
                    break
                time.sleep(1)
        except BaseException as error:
            monitoring_error = type(error).__name__
            record["termination"] = "monitoring_error"
            terminate_group(proc)
        finally:
            record["returncode"] = proc.wait()
    # A crash reporter/debugger must not leave an owned child running.
    remaining = group_members(proc.pid)
    if remaining:
        terminate_group(proc)
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
                   "remaining_owned_pids": group_members(proc.pid)})
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
