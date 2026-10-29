#!/usr/bin/env python3
"""Install the pinned render-only subset into a task-local Python environment."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
RUNTIME = BASE / "runtime"
REPORT = BASE / "validation" / "provisioning"


def main() -> int:
    authorization = json.loads((REPORT / "vendor_authorization.json").read_text())
    if authorization.get("nvidia_eula_accepted") is not True:
        raise SystemExit("NVIDIA EULA not explicitly accepted; vendor install prohibited")
    interpreter = RUNTIME / "python-builds/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11"
    prefix = RUNTIME / "a0_env"
    python = prefix / "bin/python"
    lock = REPORT / "a0_requirements.lock"
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "OMNI_KIT_ACCEPT_EULA", "PRIVACY_CONSENT"):
        environment.pop(key, None)
    environment.update({"UV_CACHE_DIR": str(RUNTIME / "cache/uv"),
                        "UV_PYTHON_INSTALL_DIR": str(RUNTIME / "python-builds"),
                        "PIP_CACHE_DIR": str(RUNTIME / "cache/pip"),
                        "PYTHONNOUSERSITE": "1"})
    commands = []
    if not prefix.exists():
        commands.append(["uv", "venv", "--python", str(interpreter), str(prefix)])
    commands.extend([
        ["uv", "pip", "compile", "--python", str(python), "--generate-hashes",
         "--extra-index-url", "https://pypi.nvidia.com", str(REPORT / "a0_requirements.in"),
         "--output-file", str(lock)],
        ["uv", "pip", "install", "--python", str(python), "--require-hashes",
         "--extra-index-url", "https://pypi.nvidia.com", "-r", str(lock)],
        ["uv", "pip", "check", "--python", str(python)],
        ["uv", "pip", "freeze", "--python", str(python)],
    ])
    events = []
    # Preserve logs across manual invocations, including resolution failures.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = REPORT / f"install_{stamp}.log"
    with log.open("x") as stream:
        for argv in commands:
            event = {"argv": argv, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            stream.write(json.dumps(event) + "\n"); stream.flush()
            process = subprocess.run(argv, cwd=BASE, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            event["returncode"] = process.returncode
            events.append(event)
            (REPORT / "install_status.json").write_text(json.dumps({"events": events, "log": str(log),
                "kit_started": False, "full_task_started": False}, indent=2) + "\n")
            if process.returncode:
                return process.returncode
    manifest = {"python": str(python), "python_version": subprocess.check_output([str(python), "--version"], text=True).strip(),
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(), "installer_log": str(log),
                "vendor_package_version": "5.1.0.0", "kit_started": False, "full_task_started": False}
    (REPORT / "environment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
