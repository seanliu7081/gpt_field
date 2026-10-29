"""Single-episode action accounting and immutable terminal truth; no simulator imports."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping


class LifecycleError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeResult:
    """A result of the registered native reward callback, after this action.

    Construction belongs to the trusted worker/evaluator, never the model.
    native_reward is full-goal reward (0/1), not the partial process score.
    """
    action_index: int
    native_reward: float | None
    end_flag: bool
    checks_registered: bool
    callback_verified: bool
    evidence_id: str
    partial_score: float | None = None


class EpisodeLifecycle:
    def __init__(self, *, action_limit: int = 1400):
        if type(action_limit) is not int or action_limit < 1:
            raise ValueError("Action limit must be a positive integer")
        self.action_limit = action_limit
        self.state = "CREATED"
        self.submitted_actions = 0
        self.confirmed_actions = 0
        self._in_flight: int | None = None
        self._terminal: dict | None = None
        self._claim: dict | None = None
        self.cleanup_errors: list[str] = []

    def ready(self) -> None:
        if self.state != "CREATED":
            raise LifecycleError("READY is only valid once, before starting")
        self.state = "READY"

    def start(self, consumed_evidence: Mapping) -> None:
        if self.state != "READY":
            raise LifecycleError("An already-ready worker and a new claim are required")
        if (consumed_evidence.get("schema") != "robodojo-uc71-consumed-1"
                or consumed_evidence.get("attempts") != 1
                or not consumed_evidence.get("claim_nonce")
                or not consumed_evidence.get("manifest_sha256")
                or not consumed_evidence.get("ready_sha256")):
            raise LifecycleError("Consumed approval evidence is required")
        self._claim = deepcopy(dict(consumed_evidence))
        self.state = "RUNNING"

    def reserve_action(self) -> int:
        """Reserve before dispatch; an unconfirmed action can never be resent."""
        if self.state != "RUNNING" or self._in_flight is not None:
            raise LifecycleError("Action requires RUNNING with no unconfirmed dispatch")
        if self.submitted_actions >= self.action_limit:
            raise LifecycleError("Native action horizon reached")
        self.submitted_actions += 1
        self._in_flight = self.submitted_actions
        return self._in_flight

    def complete_action(self, ticket: int, result: NativeResult) -> dict:
        if self.state != "RUNNING" or self._in_flight is None or type(ticket) is not int or ticket != self._in_flight:
            raise LifecycleError("Action completion does not match the pending dispatch")
        if not isinstance(result, NativeResult) or type(result.action_index) is not int or result.action_index != ticket:
            raise LifecycleError("Native result action index does not match")
        if (result.callback_verified is not True or result.checks_registered is not True
                or not isinstance(result.evidence_id, str) or not result.evidence_id.strip()
                or type(result.end_flag) is not bool
                or type(result.native_reward) not in (int, float)
                or not 0 <= result.native_reward <= 1
                or not math.isfinite(result.native_reward)):
            raise LifecycleError("Verified registered native callback evidence is required")
        if result.partial_score is not None and (type(result.partial_score) not in (int, float)
                or not 0 <= result.partial_score <= 100 or not math.isfinite(result.partial_score)):
            raise LifecycleError("Invalid native partial score")
        self.confirmed_actions = ticket
        self._in_flight = None
        evidence = {
            "source": "registered_native_reward_callback", "evidence_id": result.evidence_id,
            "action_index": ticket, "native_reward": result.native_reward,
            "native_end_flag": result.end_flag, "checks_registered": True,
            "callback_verified": True, "partial_score": result.partial_score,
        }
        if result.end_flag and result.native_reward > 1 - 1e-3:
            self._finish("success", True, evidence)
        elif self.confirmed_actions >= self.action_limit:
            self._finish("horizon_exhausted", False, evidence)
        elif result.end_flag:
            self._finish("native_failure", False, evidence)
        return self.snapshot()

    def stop(self, reason: str = "user_stop") -> dict:
        if reason not in {"user_stop", "infrastructure_error"}:
            raise LifecycleError("Unsupported external termination reason")
        if self._terminal is not None:
            return self.snapshot()
        if reason == "user_stop" and self._in_flight is not None:
            raise LifecycleError("User stop must wait for a complete feedback boundary")
        self._finish(reason, False if reason == "user_stop" else None, None)
        return self.snapshot()

    def _finish(self, status: str, success: bool | None, evidence: dict | None) -> None:
        if self._terminal is not None:
            raise LifecycleError("Terminal outcome cannot be replaced")
        self.state = "TERMINAL"
        self._terminal = {
            "schema": "robodojo-uc71-terminal-1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "official_success": success,
            "submitted_actions": self.submitted_actions, "confirmed_actions": self.confirmed_actions,
            "unconfirmed_action": self._in_flight, "native_action_limit": self.action_limit,
            "native_evidence": deepcopy(evidence), "completion_time_eligible": success is True,
            "completion_time_s": None,
        }

    def record_cleanup_failure(self, error_code: str) -> None:
        """Record an allowlisted/sanitized code, without changing the outcome."""
        if not isinstance(error_code, str) or not error_code or len(error_code) > 160:
            raise ValueError("Cleanup error code must be a short string")
        self.cleanup_errors.append(error_code)

    def snapshot(self) -> dict:
        return {"state": self.state, "submitted_actions": self.submitted_actions,
                "confirmed_actions": self.confirmed_actions, "in_flight": self._in_flight,
                "terminal": deepcopy(self._terminal), "cleanup_errors": list(self.cleanup_errors)}

    def persist_terminal(self, path: str | Path) -> None:
        """Write once and fsync before cleanup; never overwrite prior evidence."""
        record = self.terminal_record()
        path = Path(path)
        raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def terminal_record(self) -> dict:
        if self._terminal is None:
            raise LifecycleError("No terminal result exists")
        return deepcopy(self._terminal)


def load_terminal_or_crash(path: str | Path) -> dict:
    """Never turn process exit 0 or malformed terminal evidence into success."""
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict) or value.get("schema") != "robodojo-uc71-terminal-1":
            raise ValueError("Unknown terminal record")
        status = value.get("status")
        if status not in {"success", "horizon_exhausted", "native_failure", "user_stop", "infrastructure_error"}:
            raise ValueError("Unknown terminal status")
        submitted, confirmed, limit = (value.get("submitted_actions"), value.get("confirmed_actions"), value.get("native_action_limit"))
        if any(type(n) is not int for n in [submitted, confirmed, limit]) or not 0 <= confirmed <= submitted <= limit or limit < 1:
            raise ValueError("Invalid terminal action accounting")
        expected_success = True if status == "success" else None if status == "infrastructure_error" else False
        if value.get("official_success") is not expected_success:
            raise ValueError("Terminal status and success disagree")
        if value.get("completion_time_eligible") is not (status == "success"):
            raise ValueError("Terminal timing eligibility disagrees")
        elapsed = value.get("completion_time_s")
        if elapsed is not None and (status != "success" or type(elapsed) not in (int, float)
                                    or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Invalid completion time")
        pending = value.get("unconfirmed_action")
        if submitted == confirmed:
            if pending is not None:
                raise ValueError("Unexpected pending action")
        elif not (status == "infrastructure_error" and submitted == confirmed + 1
                  and type(pending) is int and pending == submitted):
            raise ValueError("Unresolved action accounting")
        if status in {"success", "horizon_exhausted", "native_failure"}:
            evidence = value.get("native_evidence")
            if not isinstance(evidence, dict):
                raise ValueError("Native evidence is missing")
            reward = evidence.get("native_reward")
            if not (evidence.get("callback_verified") is True and evidence.get("checks_registered") is True
                    and evidence.get("source") == "registered_native_reward_callback"
                    and isinstance(evidence.get("evidence_id"), str) and evidence["evidence_id"].strip()
                    and type(evidence.get("action_index")) is int and evidence["action_index"] == confirmed
                    and type(evidence.get("native_end_flag")) is bool
                    and type(reward) in (int, float) and 0 <= reward <= 1 and math.isfinite(reward)
                    and confirmed > 0 and confirmed == submitted and pending is None):
                raise ValueError("Unverified native evidence")
            native_success = evidence["native_end_flag"] and reward > 1 - 1e-3
            if status == "success" and not native_success:
                raise ValueError("Unsupported success evidence")
            if status != "success" and native_success:
                raise ValueError("Native success was relabeled as failure")
            if status == "horizon_exhausted" and confirmed != limit:
                raise ValueError("Invalid horizon outcome")
            if status == "native_failure" and (not evidence["native_end_flag"] or confirmed >= limit):
                raise ValueError("Invalid native failure")
        elif value.get("native_evidence") is not None:
            raise ValueError("External stop contains unexpected native evidence")
        return value
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        return {"schema": "robodojo-uc71-terminal-1", "status": "infrastructure_error",
                "reason": "crash_without_valid_terminal_record", "official_success": None,
                "confirmed_actions": None, "native_evidence": None,
                "completion_time_eligible": False, "completion_time_s": None}
