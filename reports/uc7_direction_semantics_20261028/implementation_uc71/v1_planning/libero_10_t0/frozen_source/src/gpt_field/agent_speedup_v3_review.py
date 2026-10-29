"""Version-isolated, single-use plan approval for agent SpeedUp v3.

Manual browser review is the default. An explicit, immutable user-delegation
note can authorize Codex to review one new draft before approving it. Delegation
never approves a draft automatically. Reset/code/config binding and one-use
claiming are inherited unchanged; old-version approvals cannot run this version.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

VERSION = "agent-speedup-v3-11"
DISPLAY_NAME = "agent speedup v3"
MANUAL_REVIEW = "manual_browser_review"
DELEGATED_REVIEW = "user_delegated_agent_review"
_DELEGATION_SCOPE = "this_initial_plan_and_one_execution"

_path = Path(__file__).with_name("agent_speedup_v2_user_check.py")
_spec = importlib.util.spec_from_file_location("gpt_field._speedup_v3_review_state", _path)
_state = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _state
_spec.loader.exec_module(_state)
_state.VERSION = VERSION
_state.DISPLAY_NAME = DISPLAY_NAME

ReviewError = _state.ReviewError
canonical_sha256 = _state.canonical_sha256
initial_observation_sha256 = _state.initial_observation_sha256
validate_review_plan = _state.validate_review_plan
plan_validation = _state.plan_validation


def delegated_review_preparation(note):
    """Describe user authorization, without marking any plan reviewed/approved."""
    if not isinstance(note, str) or not note.strip() or len(note) > 4096:
        raise ReviewError("Delegated review requires a nonempty user-authorization note of at most 4096 characters")
    return {
        "approval_mode": DELEGATED_REVIEW,
        "manual_review": False,
        "approval_provenance": {
            "reviewer": "codex", "authorized_by": "user", "authorization_note": note,
            "scope": _DELEGATION_SCOPE, "automatically_approved": False,
        },
    }


def approval_metadata(preparation=None):
    """Validate review provenance and return consistent reporting fields."""
    preparation = preparation or {}
    mode = preparation.get("approval_mode", MANUAL_REVIEW)
    if mode == MANUAL_REVIEW:
        if preparation.get("manual_review", True) is not True:
            raise ReviewError("Manual browser review must retain manual_review=true")
        provenance = preparation.get("approval_provenance")
        if provenance is not None and provenance != {
                "reviewer": "user", "authorized_by": "user", "authorization_note": None,
                "scope": _DELEGATION_SCOPE, "automatically_approved": False}:
            raise ReviewError("Manual review cannot carry delegated approval provenance")
        return {"approval_mode": mode, "manual_review": True, "reviewer": "user",
                "authorized_by": "user", "authorization_note": None,
                "approval_scope": _DELEGATION_SCOPE, "automatically_approved": False}
    if mode != DELEGATED_REVIEW:
        raise ReviewError("Only manual review or explicit user-delegated agent review is supported")
    if preparation.get("manual_review") is not False:
        raise ReviewError("Delegated agent review must retain manual_review=false")
    provenance = preparation.get("approval_provenance")
    if not isinstance(provenance, dict):
        raise ReviewError("Delegated review requires immutable user authorization and reviewer provenance")
    expected = delegated_review_preparation(provenance.get("authorization_note"))["approval_provenance"]
    if provenance != expected:
        raise ReviewError("Delegated reviewer must be Codex, authorized by the user for this one review")
    return {"approval_mode": mode, "manual_review": False, "reviewer": "codex",
            "authorized_by": "user", "authorization_note": provenance["authorization_note"],
            "approval_scope": _DELEGATION_SCOPE, "automatically_approved": False}


class ReviewStore(_state.ReviewStore):
    """Manual by default; delegated review still needs an explicit approve call."""

    @classmethod
    def create(cls, path, **kwargs):
        preparation = deepcopy(kwargs.get("preparation") or {})
        meta = approval_metadata(preparation)
        preparation.update(approval_mode=meta["approval_mode"], manual_review=meta["manual_review"])
        preparation.setdefault("approval_provenance", {
            "reviewer": meta["reviewer"], "authorized_by": meta["authorized_by"],
            "authorization_note": meta["authorization_note"], "scope": _DELEGATION_SCOPE,
            "automatically_approved": False,
        })
        kwargs["preparation"] = preparation
        return super().create(path, **kwargs)

    def _load(self):
        data = super()._load()
        approval_metadata(data.get("preparation"))
        return data
