#!/usr/bin/env python3
"""Isolated UC7.1 transit50 entry point; preflight is the only setup operation."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    # Must precede ALL gpt_field imports: action.py reads this at import time.
    existing = os.environ.get("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M")
    if existing is not None and float(existing) != .05:
        raise SystemExit("This entry point requires transit50 = 0.05m/action; conflicting environment value")
    os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.05"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT / "scripts")]
    from rc365.runner import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
