#!/usr/bin/env python3
"""Fetch detached native/gitlink pins only after a reviewed camera gate.

Does not run installers, import simulation code, fetch task assets or start tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

from provision_native_common import BASE, PINS, digest, exclusive_json, pins, require_camera_gate, require_native_checkout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-result", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--camera-assessment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = require_camera_gate(args.camera_result, args.visual_review, args.camera_assessment)
    pin_data = pins()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = BASE / "runtime/native/RoboDojo"
    if root.exists():
        raise SystemExit("Native checkout exists; verify it rather than overwriting or refetching automatically")
    root.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "robodojo-native-source-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "gate": gate, "pins_sha256": digest(PINS), "root": str(root), "commands": [],
              "complete": False, "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
              "full_test_authorized": False}
    environment = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
    environment.update(GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    try:
        with (output / "checkout.log").open("x") as log:
            def run(argv):
                record["commands"].append(argv)
                log.write(json.dumps(argv) + "\n"); log.flush()
                subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            repo = pin_data["native_repository"]
            run(["git", "init", str(root)])
            run(["git", "-C", str(root), "remote", "add", "origin", repo["url"]])
            run(["git", "-C", str(root), "fetch", "--depth=1", "origin", repo["commit"]])
            run(["git", "-C", str(root), "checkout", "--detach", repo["commit"]])
            for entry in pin_data["gitlinks"]:
                row = subprocess.check_output(["git", "-C", str(root), "ls-tree", "HEAD", entry["path"]], text=True).strip()
                if row.split("\t", 1)[0] != f"160000 commit {entry['commit']}":
                    raise ValueError(f"Unexpected gitlink pin: {entry['path']}")
                url = subprocess.check_output(["git", "-C", str(root), "config", "--file", ".gitmodules",
                                               "--get", f"submodule.{entry['path']}.url"], text=True).strip()
                if url != entry["url"]:
                    raise ValueError(f"Unexpected gitlink source URL: {entry['path']}")
            run(["git", "-C", str(root), "-c", "protocol.file.allow=never", "submodule", "update", "--init", "--recursive", "--depth=1"])
            record["verification"] = require_native_checkout(root, pin_data)
            record["recursive_submodule_status"] = subprocess.check_output(
                ["git", "-C", str(root), "submodule", "status", "--recursive"], text=True)
            if any(line and line[0] != " " for line in record["recursive_submodule_status"].splitlines()):
                raise ValueError("A recursive submodule is missing or differs from its gitlink")
            record["complete"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        exclusive_json(output / "source_manifest.json", record)
    print(json.dumps({"complete": record["complete"], "root": str(root), "manifest": str(output / "source_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
