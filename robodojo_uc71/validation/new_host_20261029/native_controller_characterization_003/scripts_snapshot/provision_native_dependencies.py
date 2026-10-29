#!/usr/bin/env python3
"""Install only native reset/camera dependencies into a new task-local overlay.

No simulator, Torch, planner, or model code is imported. The existing Isaac Sim
environment is used only for interpreter selection and distribution metadata.
An existing overlay or evidence directory is never reused.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

BASE = Path(__file__).resolve().parents[1]
PINS = {
    "omegaconf": "2.3.0",
    "antlr4-python3-runtime": "4.9.3",
    "transforms3d": "0.4.2",
    "shapely": "2.1.2",
    "gymnasium": "1.2.1",
    "cloudpickle": "3.1.2",
    "farama-notifications": "0.0.4",
    "prettytable": "3.3.0",
    "wcwidth": "0.2.14",
    "flatdict": "4.0.0",
    "h5py": "3.15.1",
    "tqdm": "4.67.1",
}
METADATA_CODE = """import importlib.metadata as m,json,sys
distributions = m.distributions(path=[sys.argv[1]]) if len(sys.argv)>1 else m.distributions()
print(json.dumps({'python':sys.version,'python_version':list(sys.version_info[:3]),
 'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in distributions],
 key=lambda row:row['name'].lower())},indent=2))
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_map(inventory: dict) -> dict[str, str]:
    result = {}
    for row in inventory["packages"]:
        name, version = canonical(row["name"]), row["version"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
            raise ValueError("Invalid distribution metadata")
        if name in result:
            raise ValueError(f"Duplicate distribution metadata: {name}")
        result[name] = version
    return result


def filter_overlay_lock(compiled: str, base: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Accept only exact hashed requirements; remove identical base packages."""
    blocks, current = [], []
    for line in compiled.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        current.append(line)
        if not line.rstrip().endswith("\\"):
            blocks.append(current)
            current = []
    if current:
        raise ValueError("Incomplete lock continuation")
    selected, seen, missing = [], {}, {}
    for block in blocks:
        tokens = " ".join(line.rstrip().removesuffix("\\").strip() for line in block).split()
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", tokens[0])
        if match is None or len(tokens) < 2 or any(not re.fullmatch(r"--hash=sha256:[a-f0-9]{64}", token) for token in tokens[1:]):
            raise ValueError("Lock must contain only exact pins and SHA256 hashes")
        name, version = canonical(match[1]), match[2]
        if name in seen:
            raise ValueError(f"Duplicate lock requirement: {name}")
        seen[name] = version
        if name in base:
            if version != base[name]:
                raise ValueError(f"Resolver changed preserved base version: {name}")
        elif PINS.get(name) != version:
            raise ValueError(f"Unexpected overlay dependency or version: {name}=={version}")
        else:
            selected.append("\n".join(block))
            missing[name] = version
    if any(seen.get(name) != version for name, version in PINS.items()):
        raise ValueError("Compiled lock omits or changes a requested pin")
    return "\n".join(selected) + "\n", missing


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-record", type=Path, required=True,
                        help="Completed base install_status.json with adjacent lock and inventory")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args()
    output = args.output.resolve()
    runtime = BASE / "runtime"
    prefix = runtime / "new_host_env"
    target = runtime / "native_overlay"
    python = prefix / "bin/python"
    record_path = args.install_record.resolve()
    inventory_path = record_path.with_name("package_inventory.json")
    lock_path = record_path.with_name("requirements.lock")
    record = json.loads(record_path.read_text())
    recorded_inventory = json.loads(inventory_path.read_text())
    base = package_map(recorded_inventory)
    essential = {"isaacsim": "5.1.0.0", "torch": "2.7.0+cu128", "numpy": "1.26.0",
                 "torchvision": "0.22.0+cu128", "torchaudio": "2.7.0+cu128", "pillow": "11.3.0"}
    if (record.get("ready_for_diagnostic") is not True
            or Path(record.get("python", "")).absolute() != python
            or sha256(lock_path) != record.get("lock_sha256")
            or len(base) != 140 or any(base.get(k) != v for k, v in essential.items())
            or any(v != "5.1.0.0" for k, v in base.items() if k.startswith("isaacsim"))):
        parser.error("Expected completed, lock-bound 140-package Isaac Sim 5.1 / Torch cu128 installation")
    if not python.is_file() or not os.access(python, os.X_OK):
        parser.error("The recorded task-local interpreter is missing")
    if target.exists() or target.is_symlink():
        parser.error("Refusing to reuse an existing native_overlay, including a partial installation")
    if output.exists() or not output.is_relative_to(BASE / "validation"):
        parser.error("--output must be a new directory inside this task's validation directory")
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("home", "tmp", "uv-cache"):
        (output / directory).mkdir()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output / "home"),
           "TMPDIR": str(output / "tmp"), "UV_CACHE_DIR": str(output / "uv-cache"),
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    status = {"schema": "robodojo-native-dependencies-1", "started_at_utc": now(),
              "scope": "Package setup for native reset/camera preview only",
              "python": str(python), "target": str(target), "events": [],
              "base_install_record": str(record_path), "base_lock_sha256": sha256(lock_path),
              "base_inventory_sha256": sha256(inventory_path),
              "script_sha256": sha256(Path(__file__)), "pins": PINS,
              "base_unchanged": False, "ready_for_native_preview_dependencies": False,
              "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
              "full_test_approved": False,
              "limitations": ["Package resolution and metadata checks only; native runtime remains untested",
                              "Source-only flatdict/ANTLR builds use uv-isolated build dependencies"]}

    def save() -> None:
        write_json(output / "install_status.json", status)

    def run(argv: list[str], timeout: int = 900) -> Path:
        index = len(status["events"])
        log_path = output / f"command_{index:02d}.log"
        event = {"argv": argv, "started_at_utc": now(), "log": log_path.name}
        status["events"].append(event)
        save()
        print(json.dumps(event), flush=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(argv, cwd=output, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                event["returncode"] = process.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                event["interrupted_or_timed_out"] = True
                raise
            finally:
                event["completed_at_utc"] = now()
                save()
        if event["returncode"]:
            raise RuntimeError(f"Command {index} failed; inspect {log_path}")
        return log_path

    def inventory(name: str, overlay: bool = False) -> dict:
        argv = [str(python), "-I", "-B", "-c", METADATA_CODE]
        if overlay:
            argv.append(str(target))
        value = json.loads(run(argv, timeout=30).read_text())
        write_json(output / name, value)
        return value

    def verify_base(value: dict) -> None:
        if (value.get("python_version") != [3, 11, 16]
                or value.get("python") != recorded_inventory.get("python")
                or package_map(value) != base):
            raise RuntimeError("Current Python/base distribution metadata differs from preserved inventory")

    save()
    for source, name in ((record_path, "base_install_status.json"),
                         (inventory_path, "base_recorded_inventory.json"),
                         (lock_path, "base_requirements.lock"),
                         (Path(__file__), "provision_native_dependencies.py")):
        shutil.copyfile(source, output / name)
    try:
        verify_base(inventory("base_before.json"))
        run([uv, "--version"], timeout=30)
        constraints = output / "base_constraints.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(base.items())))
        requirements = output / "requirements.in"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(PINS.items())))
        resolved = output / "resolved_requirements.lock"
        run([uv, "--no-config", "pip", "compile", "--python", str(python),
             "--default-index", "https://pypi.org/simple", "--no-sources", "--generate-hashes",
             "--no-annotate", "--no-header", "--constraints", str(constraints),
             "--output-file", str(resolved), str(requirements)])
        filtered, expected_overlay = filter_overlay_lock(resolved.read_text(), base)
        overlay_lock = output / "requirements.lock"
        overlay_lock.write_text(filtered)
        status["lock_sha256"] = sha256(overlay_lock)
        status["resolved_lock_sha256"] = sha256(resolved)
        status["expected_overlay_packages"] = expected_overlay
        save()
        target.mkdir(exist_ok=False)
        run([uv, "--no-config", "pip", "install", "--python", str(python),
             "--target", str(target), "--require-hashes", "--no-deps", "--no-sources",
             "--default-index", "https://pypi.org/simple", "--link-mode", "copy",
             "--requirements", str(overlay_lock)])
        overlay = inventory("overlay_inventory.json", overlay=True)
        actual_overlay = package_map(overlay)
        if actual_overlay != expected_overlay or set(actual_overlay) & set(base):
            raise RuntimeError("Overlay inventory differs from its lock or shadows a base distribution")
        after = inventory("base_after.json")
        verify_base(after)
        status["base_unchanged"] = True
        combined = dict(base, **actual_overlay)
        write_json(output / "combined_inventory.json", {"python": after["python"],
            "packages": [{"name": name, "version": version} for name, version in sorted(combined.items())]})
        status["artifacts_sha256"] = {name: sha256(output / name) for name in (
            "base_before.json", "base_after.json", "base_constraints.txt", "requirements.in",
            "requirements.lock", "resolved_requirements.lock", "overlay_inventory.json", "combined_inventory.json")}
        status["ready_for_native_preview_dependencies"] = True
        status["completed_at_utc"] = now()
        save()
        print(json.dumps(status), flush=True)
        return 0
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        try:
            verify_base(inventory("base_after_failure.json"))
            status["base_unchanged"] = True
        except Exception as audit_error:
            status["base_audit_error"] = f"{type(audit_error).__name__}: {audit_error}"
        status["completed_at_utc"] = now()
        save()
        print(json.dumps(status), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
