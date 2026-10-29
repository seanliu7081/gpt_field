#!/usr/bin/env python3
"""Verify the isolated core; does not import it, run a model or start a simulator."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
core = ROOT / "vendor" / "uc71"
manifest = json.loads((core / "release_manifest.json").read_text())
failures = []
for name, expected in manifest["files"].items():
    path = core / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
print(json.dumps({"core_files": len(manifest["files"]), "mismatches": failures,
                  "model_calls": 0, "simulator_starts": 0}, indent=2))
sys.exit(bool(failures))
