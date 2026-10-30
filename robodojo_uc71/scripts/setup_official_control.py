#!/usr/bin/env python3
"""Provision only the pinned official IK dependencies in a task-local overlay.

--install builds a git-archive snapshot and installs a hash-locked wheel set;
it does not upgrade the base interpreter, native source, or existing assets.
--prepare-assets derives the native X5 curobo.yml from its pinned template by
the upstream path-substitution rule. Default operation records/verifies an
already installed overlay. No simulator, inference or robot motion is launched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.official_control_setup import SCHEMA, digest, verify_setup_record

TEMPLATE_SHA256 = "5eea08bf7ef276bf9dfd078ef9f69221e9a67c8cf6cd19fce5ef66d42e96b3f2"


def file_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def generated_asset(native_root, *, prepare=False):
    template = native_root / "Assets/Robots/x5/curobo_tmp.yml"
    output = template.with_name("curobo.yml")
    if digest(template) != TEMPLATE_SHA256:
        raise ValueError("Pinned X5 cuRobo asset template differs")
    value = template.read_text().replace("${ASSETS_PATH}", str(native_root)).replace("$ASSETS_PATH", str(native_root))
    if output.exists():
        if output.read_text() != value:
            raise ValueError("Existing native curobo.yml differs; refusing to overwrite")
    elif prepare:
        with output.open("x") as stream:
            stream.write(value)
    else:
        raise ValueError("Native curobo.yml absent; use --prepare-assets")
    return {"template": file_record(template), "generated": file_record(output),
            "rule": "upstream ${ASSETS_PATH}/$ASSETS_PATH substitution only",
            "generator": file_record(native_root / "utils/update_embodiment_config_path.py")}


def install_overlay(native_root, output, curobo_commit):
    overlay = output / "site-packages"
    if overlay.exists() and any(overlay.iterdir()):
        raise ValueError("Use a fresh output directory for installation; existing overlay is preserved")
    source = output / "source"
    source.mkdir()
    raw = subprocess.check_output(["git", "-C", str(native_root / "third_party/curobo"), "archive", curobo_commit])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(source, filter="data")
    wheels = output / "wheelhouse"
    wheels.mkdir(exist_ok=True)
    env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="0.0.0+g" + curobo_commit[:12])
    commands = [
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(source)],
        [sys.executable, "-m", "pip", "download", "--no-deps", "--require-hashes", "-r",
         str(BASE / "configs/official_control_dependencies.lock"), "--dest", str(wheels)],
    ]
    for i, command in enumerate(commands):
        with (output / f"install_{i}.log").open("x") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / "install_overlay.log").open("x") as log:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--no-compile", "--target", str(overlay),
                        *[str(p) for p in sorted(wheels.glob("*.whl"))]], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def complete_package_data(native_root, overlay, *, prepare=False):
    """Preserve the data available in the upstream recommended editable install.

    This pinned wheel omits content/ YAML and mesh files; copy exact source bytes
    without changing package code or substituting another cuRobo release.
    """
    source = native_root / "third_party/curobo/curobo/content"
    rows = []
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        target = overlay / "curobo/content" / item.relative_to(source)
        if target.exists():
            if digest(target) != digest(item):
                raise ValueError("Existing official cuRobo package data differs")
        elif prepare:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
        else:
            raise ValueError("Official wheel omits package data; use --complete-package-data")
        rows.append(file_record(item))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=BASE / "runtime/native/RoboDojo")
    parser.add_argument("--output", type=Path, default=BASE / "runtime/official_control")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--complete-package-data", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--record-name", default="setup.json")
    args = parser.parse_args()
    native, output = args.native_root.resolve(strict=True), args.output.resolve()
    if Path(args.record_name).name != args.record_name or not args.record_name.endswith(".json"):
        raise ValueError("record-name must be a JSON basename")
    record_path = output / args.record_name
    if args.verify_only:
        record = verify_setup_record(record_path)
        print(json.dumps({"verified": True, "overlay_target": record["overlay_target"]}))
        return
    output.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise ValueError("Existing setup record is immutable; use --verify-only or a fresh output directory")
    pins = json.loads((BASE / "configs/source_pins.json").read_text())
    native_commit = pins["native_repository"]["commit"]
    curobo_commit = next(x["commit"] for x in pins["gitlinks"] if x["path"] == "third_party/curobo")
    for path, expected in ((native, native_commit), (native / "third_party/curobo", curobo_commit)):
        if subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip() != expected:
            raise ValueError("Native/cuRobo checkout differs from source pins")
        subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--"],
                       stdout=subprocess.DEVNULL, check=True)
    asset = generated_asset(native, prepare=args.prepare_assets)
    if args.install:
        install_overlay(native, output, curobo_commit)
    overlay = output / "site-packages"
    if not (overlay / "curobo/inverse_kinematics.py").is_file():
        raise ValueError("Official cuRobo overlay missing; use --install")
    package_data = complete_package_data(native, overlay, prepare=args.install or args.complete_package_data)
    # Import validation uses a child process with no GPU/SimulationApp creation.
    import_check = "from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg; from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg; from curobo.motion_planner import MotionPlanner; from curobo.types import ToolPoseCriteria; print('official cuRobo imports passed')"
    env = dict(os.environ, PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1")
    with (output / "imports.log").open("w") as log:
        subprocess.run([sys.executable, "-c", import_check], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    source = [BASE / "configs/source_pins.json", BASE / "configs/official_control_dependencies.lock",
              Path(__file__), BASE / "scripts/check_official_ik.py",
              BASE / "src/robodojo_uc71/official_control_setup.py",
              native / "env/planner_manager/curobo_planner.py",
              native / "env/robot_manager/robot_manager.py",
              native / "Assets/Robots/x5/X5A.urdf", native / "Assets/Robots/x5/robot_config.yml",
              *[Path(asset[k]["path"]) for k in ("template", "generated", "generator")]]
    # The complete local X5 asset closure is bound, including URDF mesh dependencies.
    source.extend(p for p in (native / "Assets/Robots/x5").rglob("*") if p.is_file())
    source.extend(Path(row["path"]) for row in package_data)
    base = ["torch", "numpy", "scipy", "trimesh", "networkx", "omegaconf", "setuptools", "packaging"]
    record = {"schema": SCHEMA, "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "verified": True, "overlay_target": str(overlay), "python": sys.executable,
              "native_commit": native_commit, "curobo_commit": curobo_commit,
              "source_files": [file_record(p) for p in sorted(set(source))],
              "installed_files": [file_record(p) for p in sorted(overlay.rglob("*"))
                                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"],
              "base_requirements": [{"name": n, "version": metadata.version(n)} for n in base],
              "wheels": [file_record(p) for p in sorted((output / "wheelhouse").glob("*.whl"))],
              "asset_derivation": asset, "imports_passed": True,
              "package_data_source": "exact pinned source content/, as available in upstream editable install",
              "native_motion_qualified": False, "model_calls": 0, "benchmark_actions": 0}
    with record_path.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    verify_setup_record(record_path)
    print(json.dumps({"verified": True, "overlay_target": str(overlay), "record": str(record_path)}))


if __name__ == "__main__":
    main()
