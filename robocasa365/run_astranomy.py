#!/usr/bin/env python3
"""One explicitly authorized Astranomy Direct RoboCasa365 retest."""
import os
from pathlib import Path
import sys


def main():
    here = Path(__file__).resolve().parent
    # This fixes the shared native OSC normalization, not the Direct step cap.
    # ShortTargetExecutor independently retains its 0.02 m / 0.1 rad bounds.
    value = os.environ.get("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M", "0.05")
    if float(value) != .05:
        raise SystemExit("The matched RoboCasa native controller requires scale 0.05")
    os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.05"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path[:0] = [str(here), str(here.parent / "src"), str(here.parent / "scripts")]
    from rc365.astranomy_runner import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
