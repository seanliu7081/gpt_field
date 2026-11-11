#!/usr/bin/env python3
"""Isolated RestockPantry foundation-vision entry point; preserves mobile transit50."""
import os
from pathlib import Path
import sys


def main():
    existing = os.environ.get("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M")
    if existing is not None and float(existing) != .05:
        raise SystemExit("Foundation RoboCasa requires transit50; conflicting environment value")
    os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.05"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    project = Path(__file__).resolve().parent
    root = project.parent
    sys.path[:0] = [str(project), str(root / "src"), str(root / "scripts")]
    from rc365.foundation_vision import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
