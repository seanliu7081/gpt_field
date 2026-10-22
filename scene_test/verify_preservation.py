"""Verify that pre-existing project files remain byte-for-byte unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path, baseline: Path) -> dict:
    records = json.loads(baseline.read_text())["files"]
    changed, missing = [], []
    for name, expected in records.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    return {"status": "passed" if not changed and not missing else "failed",
            "existing_files_checked": len(records), "changed": changed,
            "missing": missing, "baseline": str(baseline)}


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(here.parent, here / "source_preservation_before.json")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
