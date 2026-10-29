"""Frozen subscription/history dependency, isolated to this adapter process."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


FROZEN_SOURCE = Path(__file__).resolve().parents[2] / "vendor" / "uc71" / "src"


def require_frozen_core():
    """Reject an already imported shared-tree core; never monkey-patch it."""
    module = sys.modules.get("gpt_field")
    if module is None:
        if not (FROZEN_SOURCE / "gpt_field" / "__init__.py").is_file():
            raise RuntimeError("The frozen uc7.1 snapshot is missing")
        sys.path.insert(0, str(FROZEN_SOURCE))
        module = importlib.import_module("gpt_field")
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(FROZEN_SOURCE):
        raise RuntimeError("This adapter must import gpt_field from vendor/uc71")
    return module


require_frozen_core()

from gpt_field.direct_history import _trim_images  # noqa: E402


def trim_visual_sets(messages, keep=6):
    """The frozen function trims whole turns, including all three camera views."""
    if type(keep) is not int or keep != 6:
        raise ValueError("The pilot retains exactly six synchronized visual sets")
    return _trim_images(messages, keep)

