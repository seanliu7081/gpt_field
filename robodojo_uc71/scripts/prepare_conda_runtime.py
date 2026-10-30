#!/usr/bin/env python3
"""Record and verify the installed Conda runtime; no simulator or inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robodojo_uc71.host_runtime import prepare_conda_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_conda_record(args.python, args.output))
