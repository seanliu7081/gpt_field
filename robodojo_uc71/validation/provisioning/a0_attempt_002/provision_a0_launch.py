#!/usr/bin/env python3
"""Single-use, explicitly authorized GPU-4 render diagnostic (never a test run)."""
from __future__ import annotations

import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
RUNTIME = BASE / "runtime"
VALIDATION = BASE / "validation" / "provisioning"
GPU_UUID = "GPU-e1a460e9-e272-b227-222e-0dbb77c248c5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=("a0_attempt_001", "a0_attempt_002"), default="a0_attempt_001")
    arguments = parser.parse_args()
    if arguments.attempt == "a0_attempt_002":
        repair = json.loads((VALIDATION / "setup_repair_authorization.json").read_text())
        if repair.get("authorized_next_attempt") != arguments.attempt or repair.get("automatic_retry") is not False:
            raise SystemExit("The separately reviewed setup repair is not authorized")
    authorization_path = VALIDATION / "vendor_authorization.json"
    authorization = json.loads(authorization_path.read_text())
    required = {
        "nvidia_eula_accepted": True,
        "stage_a0_launch_authorized": True,
        "approved_gpu_uuid": GPU_UUID,
        "full_task_authorized": False,
    }
    if any(authorization.get(k) != v for k, v in required.items()):
        raise SystemExit("Explicit vendor/Stage A0 authorization is absent or mismatched")
    if not str(authorization.get("user_message", "")).strip():
        raise SystemExit("Authorization must preserve the actual user acceptance message")
    python = RUNTIME / "a0_env" / "bin" / "python"
    if not python.is_file():
        raise SystemExit("Isolated Python environment is not provisioned")
    query = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used", "--format=csv,noheader,nounits"]
    before = subprocess.check_output(query, text=True)
    selected = [r.split(",") for r in before.splitlines() if GPU_UUID in r]
    if len(selected) != 1 or selected[0][0].strip() != "4" or int(selected[0][3].strip()) >= 256:
        raise SystemExit("GPU 4 identity changed or is no longer idle")
    output = VALIDATION / arguments.attempt
    output.mkdir()  # Fail on reuse. No automatic launch or retry.
    (output / "gpu_before.csv").write_text(before)
    (output / "authorization_snapshot.json").write_bytes(authorization_path.read_bytes())
    for name in ("provision_a0_camera.py", "provision_a0_launch.py"):
        (output / name).write_bytes((BASE / "scripts" / name).read_bytes())
    for name in ("a0_camera.kit", "a0_requirements.in", "a0_requirements.lock"):
        (output / name).write_bytes((VALIDATION / name).read_bytes())
    if arguments.attempt == "a0_attempt_002":
        (output / "setup_repair_authorization.json").write_bytes((VALIDATION / "setup_repair_authorization.json").read_bytes())
    # Do not inherit model/provider credentials or unrelated training settings.
    environment = {k: os.environ[k] for k in ("HOME", "LANG", "LC_ALL", "TZ") if k in os.environ}
    environment.update({
        "PATH": str(python.parent) + ":/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(RUNTIME / "sysroot/usr/lib/x86_64-linux-gnu")
                           + ":/opt/nvidia-drivers/lib:/usr/lib/x86_64-linux-gnu",
        "PYTHONNOUSERSITE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "OMNI_KIT_ACCEPT_EULA": "YES",  # Only after the explicit record check.
        "UC71_A0_AUTHORIZED_CHILD": "1", "UC71_A0_OUTPUT": str(output),
        "XDG_CACHE_HOME": str(RUNTIME / "cache"),
        "XDG_CONFIG_HOME": str(RUNTIME / "config"),
        "XDG_DATA_HOME": str(RUNTIME / "data"),
        "CUDA_CACHE_PATH": str(RUNTIME / "cache" / "cuda"),
        "WARP_CACHE_PATH": str(RUNTIME / "cache" / "warp"),
    })
    argv = [str(python), "-X", "faulthandler", "-u", str(BASE / "scripts" / "provision_a0_camera.py")]
    record = {"argv": argv, "gpu_uuid": GPU_UUID, "automatic_retry": False,
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "authorization_sha256": hashlib.sha256(authorization_path.read_bytes()).hexdigest()}
    (output / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    with (output / "stdout_stderr.log").open("w") as logfile:
        proc = subprocess.Popen(argv, cwd=RUNTIME, env=environment, stdout=logfile,
                                stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = proc.pid
        try:
            returncode = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            returncode = 124
            record["termination"] = "diagnostic_timeout_no_retry"
    record.update(returncode=returncode, elapsed_s=time.monotonic() - started)
    (output / "gpu_after.csv").write_text(subprocess.check_output(query, text=True))
    (output / "launch_result.json").write_text(json.dumps(record, indent=2) + "\n")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
