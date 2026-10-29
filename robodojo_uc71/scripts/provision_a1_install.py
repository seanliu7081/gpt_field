#!/usr/bin/env python3
"""Install the hash-pinned stock Isaac Sim diagnostic in a new isolated venv."""
from __future__ import annotations

import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parents[1]
RUNTIME = BASE / "runtime"
REPORT = BASE / "validation/provisioning"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-reviewed-setup", action="store_true")
    args = parser.parse_args()
    auth = json.loads((REPORT / "a1_authorization.json").read_text())
    eula = json.loads((REPORT / "vendor_authorization.json").read_text())
    if auth.get("diagnostics_scope_reopened") is not True or eula.get("nvidia_eula_accepted") is not True:
        raise SystemExit("Explicit renewed diagnostics and vendor authorization are required")
    lock = REPORT / "a1_requirements.lock"
    if not all(pin in lock.read_text() for pin in ("torch-2.7.0%2Bcu128", "torchvision-0.22.0%2Bcu128", "torchaudio-2.7.0%2Bcu128")):
        raise SystemExit("CUDA 12.8 torch pins missing from resolved lock")
    free = shutil.disk_usage(BASE).free
    if free < 60 * 1024**3:
        raise SystemExit("At least 60 GiB free is required for bounded isolated setup")
    prefix = RUNTIME / "a1_env"
    if prefix.exists() and not args.resume_reviewed_setup:
        raise SystemExit("a1_env already exists; inspect prior setup, do not automatically reinstall")
    if args.resume_reviewed_setup:
        prior = json.loads((REPORT / "a1_install_status_failed_backend_resolution.json").read_text())
        if prior.get("ready_for_diagnostic") is not False or not prefix.is_dir():
            raise SystemExit("Only the preserved reviewed dependency-resolution failure may be resumed")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = REPORT / f"a1_install_{stamp}.log"
    python = prefix / "bin/python"
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TZ") if k in os.environ}
    env.update({"UV_CACHE_DIR": str(RUNTIME / "a1_cache/uv"), "PYTHONNOUSERSITE": "1"})
    interpreter = RUNTIME / "python-builds/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
    commands = [] if prefix.exists() else [["uv", "venv", "--python", str(interpreter), str(prefix)]]
    commands += [
        ["uv", "pip", "install", "--python", str(python), "--require-hashes",
         "--extra-index-url", "https://pypi.nvidia.com", "--link-mode", "copy", "-r", str(lock)],
        ["uv", "pip", "check", "--python", str(python)],
        ["uv", "pip", "freeze", "--python", str(python)],
    ]
    status = {"scope": "stock Isaac Sim 5.1 model-free diagnostic setup only", "started_at_utc": stamp,
              "python": str(python), "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
              "log": str(log), "free_bytes_before": free, "events": [], "kit_started": False,
              "model_calls": 0, "benchmark_actions": 0,
              "reviewed_setup_continuation": args.resume_reviewed_setup,
              "source_bindings": str(REPORT / "a1_wheel_sources.json")}
    with log.open("x") as output:
        for command in commands:
            event = {"argv": command, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            output.write(json.dumps(event) + "\n"); output.flush()
            result = subprocess.run(command, env=env, cwd=BASE, stdout=output, stderr=subprocess.STDOUT)
            event["returncode"] = result.returncode
            status["events"].append(event)
            status["ready_for_diagnostic"] = result.returncode == 0 and len(status["events"]) == len(commands)
            status["free_bytes_after"] = shutil.disk_usage(BASE).free
            (REPORT / "a1_install_status.json").write_text(json.dumps(status, indent=2) + "\n")
            if result.returncode:
                print(json.dumps(status), flush=True)
                return result.returncode
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
