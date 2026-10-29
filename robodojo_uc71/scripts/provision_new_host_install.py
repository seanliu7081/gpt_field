#!/usr/bin/env python3
"""Rebuild the preserved stock Isaac Sim lock without changing old evidence.

This installs packages only. It never imports Isaac Sim or starts a task.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = BASE / "runtime"
    prefix = runtime / "new_host_env"
    interpreter = runtime / "python-builds/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
    lock = BASE / "validation/provisioning/a1_requirements.lock"
    licence = BASE / "validation/provisioning/vendor_authorization.json"
    if json.loads(licence.read_text()).get("nvidia_eula_accepted") is not True:
        raise SystemExit("Preserved NVIDIA licence acceptance is required")
    if prefix.exists():
        raise SystemExit("Refusing to overwrite an existing environment; review setup evidence first")
    if not interpreter.is_file():
        raise SystemExit("Bootstrap the task-local pinned Python 3.11.16 first")
    if shutil.disk_usage(BASE).free < 60 * 1024**3:
        raise SystemExit("At least 60 GiB free is required for this isolated installation")
    if not all(pin in lock.read_text() for pin in (
        "isaacsim==5.1.0.0", "torch-2.7.0%2Bcu128", "torchvision-0.22.0%2Bcu128",
    )):
        raise SystemExit("Expected preserved Isaac Sim / CUDA 12.8 package pins are missing")
    python = prefix / "bin/python"
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ") if key in os.environ}
    env.update(UV_CACHE_DIR=str(runtime / "new_host_cache/uv"), PYTHONNOUSERSITE="1")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required")
    commands = [
        [uv, "venv", "--python", str(interpreter), str(prefix)],
        [uv, "pip", "install", "--python", str(python), "--require-hashes",
         "--default-index", "https://pypi.org/simple", "--extra-index-url", "https://pypi.nvidia.com",
         "--link-mode", "copy", "-r", str(lock)],
        [uv, "pip", "check", "--python", str(python)],
    ]
    status = {
        "scope": "New-host isolated package setup; full test not approved",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "licence_provenance": str(licence.relative_to(BASE)),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python": str(python), "events": [], "ready_for_diagnostic": False,
        "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
        "full_test_approved": False,
    }
    shutil.copyfile(lock, output / "requirements.lock")
    with (output / "install.log").open("x") as log:
        for command in commands:
            event = {"argv": command, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            print(json.dumps(event), flush=True)
            log.write(json.dumps(event) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
            event["returncode"] = result.returncode
            status["events"].append(event)
            status["free_bytes_after"] = shutil.disk_usage(BASE).free
            (output / "install_status.json").write_text(json.dumps(status, indent=2) + "\n")
            if result.returncode:
                print(json.dumps(status), flush=True)
                return result.returncode
    inventory = subprocess.check_output([
        str(python), "-c", "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda x:x['name'].lower())},indent=2))",
    ], env=env, text=True)
    (output / "package_inventory.json").write_text(inventory)
    status["ready_for_diagnostic"] = True
    status["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "install_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
