"""Review artifacts for the separate ``speed up v2 user check`` entry point.

No robot, simulator, evaluator or model is imported here.  A review is bound to
one task reset, one code inventory, and one immutable model draft.  The browser
may select only from that draft's parts and effect descriptions.  Approval is a
single-use transition, not a boolean supplied to the test runner.

Checksums detect stale/corrupted artifacts; they are not signatures against a
local user who can rewrite every file in the review directory.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping
import uuid

from .agent_speedup_v2_plan import PlanParseError
from .agent_speedup_v2_user_check_plan import parse_plan, validate_plan
from .agent_speedup_v2_types import SLICE_DO_KINDS


VERSION = "agent-speedup-v2-user-check-7.1"
DISPLAY_NAME = "speed up v2 user check"
SCHEMA_VERSION = 1
_HASH = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_FIELDS = (
    "schema_version", "version", "id", "created_at", "task_key",
    "episode_index", "task_prompt", "initial_observation_sha256",
    "runtime_sha256", "draft_plan", "preparation",
)
_REVIEW_FIELDS = set(_CONTEXT_FIELDS) | {
    "context_sha256", "plan", "revision", "status", "reviewed_steps",
    "approved", "claim", "state_sha256",
}


class ReviewError(ValueError):
    """Invalid, stale, unapproved, or already consumed review."""

    def __init__(self, message: str, *, issues: list[dict] | None = None):
        super().__init__(message)
        self.issues = issues or []


def _json_bytes(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ReviewError("Artifact must contain finite JSON values") from exc


def canonical_sha256(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def initial_observation_sha256(value) -> str:
    """The existing v2 simulator's reset hash, without importing its worker."""
    return canonical_sha256(value)


def _copy(value):
    return json.loads(_json_bytes(value))


def semantic_plan(raw: Mapping) -> dict:
    """Copy a native plan, omitting only viewpoint-dependent part grid cells."""
    result = _copy(raw)
    for part in result["parts"]:
        part.pop("w", None)
        part.pop("x", None)
    return result


def _parse_lossless(raw: Mapping) -> dict:
    try:
        plan = parse_plan(raw)
    except (PlanParseError, ValueError, TypeError) as exc:
        raise ReviewError(str(exc)) from exc
    parsed = plan.to_json()
    if parsed != raw:
        raise ReviewError(
            "Plan text must round-trip unchanged through v2; use printable ASCII "
            "rather than text that the executor would silently sanitize"
        )
    return _copy(parsed)


def plan_validation(raw: Mapping, *, require_native: bool = False) -> dict:
    """Validate task intents; require macro support only for native execution."""
    clean = _parse_lossless(raw)
    issues = [issue.to_dict() for issue in validate_plan(parse_plan(clean))]
    from .agent_speedup_v2_user_check_spatial import validate_spatial_spec
    from .agent_speedup_v2_user_check_constraints import constraint_conflicts
    for issue in validate_spatial_spec(clean):
        issues.append({"code": "spatial_spec", "step": None, "pid": None,
                       "text": f"{issue['field']}[{issue['index']}]: {issue['reason']}"[:120]})
    for index, step in enumerate(clean["steps"]):
        for conflict in constraint_conflicts(step):
            issues.append({"code": "constraint_conflict", "step": index, "pid": conflict["part"],
                           "text": "Active clear_of forbids the contact required by this skill"})
    if not clean["steps"]:
        issues.append({"code": "empty_steps", "step": None, "pid": None,
                       "text": "At least one task intent must be reviewed"})
    for i, step in enumerate(clean["steps"]):
        if require_native and step["do"] not in SLICE_DO_KINDS:
            issues.append({"code": "relation_not_implemented", "step": i,
                           "pid": step["obj"],
                           "text": f"Step {i}: {step['do']} is not implemented in v2"})
            continue
        ignored = []
        if step["do"] != "slide" and step["sense"] != "any":
            ignored.append("sense must be any except for slide")
        if step["do"] not in ("place_on", "place_in", "push_to") and step["ref"] is not None:
            ignored.append("ref must be null except for place_on, place_in or push_to")
        if step["do"] == "release" and step["grip"]:
            ignored.append("release must have an empty grip list")
        for criterion in step["until"]:
            if criterion["o"] not in ("effect", "fits") and criterion["p"] is not None:
                ignored.append("stop/angle criteria must have p null")
            if criterion["o"] != "effect" and criterion["txt"] is not None:
                ignored.append("only effect criteria accept descriptive text")
        for text in ignored:
            issues.append({"code": "ignored_selection", "step": i,
                           "pid": step["obj"], "text": text})
    return {"valid": not issues, "issues": issues}


def validate_review_plan(raw: Mapping) -> dict:
    """Validate exact task-intent JSON independently of optional macro availability."""
    clean = _parse_lossless(raw)
    validation = plan_validation(clean)
    if not validation["valid"]:
        raise ReviewError("; ".join(issue["text"] for issue in validation["issues"]),
                          issues=validation["issues"])
    return clean


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime(value) -> dict:
    if not isinstance(value, dict) or not value:
        raise ReviewError("runtime_sha256 must be a nonempty code hash mapping")
    if any(not isinstance(key, str) or not key or not isinstance(digest, str)
           or not _HASH.fullmatch(digest) for key, digest in value.items()):
        raise ReviewError("runtime_sha256 requires file names and lowercase SHA-256 values")
    return _copy(value)


def _indices(values, count: int, *, complete: bool = False) -> list[int]:
    if not isinstance(values, (list, tuple)) or any(
            type(i) is not int or not 0 <= i < count for i in values):
        raise ReviewError("reviewed_steps must contain zero-based step indices")
    if len(set(values)) != len(values):
        raise ReviewError("reviewed_steps contains duplicate step indices")
    result = sorted(values)
    if complete and result != list(range(count)):
        raise ReviewError("Every step must be checked before approval")
    return result


def _selector_edits(draft: dict, plan: dict) -> None:
    if draft["scene"] != plan["scene"]:
        raise ReviewError("Scene text is model provenance and cannot be rewritten")
    for part in plan["parts"]:
        if not any(all(part[key] == old[key] for key in ("id", "text", "w", "x"))
                   for old in draft["parts"]):
            raise ReviewError("Select existing model parts; part text and image cells are immutable")
    descriptions = {None}
    for step in draft["steps"]:
        descriptions.update(item["txt"] for item in step["until"])
    if any(item["txt"] not in descriptions for step in plan["steps"] for item in step["until"]):
        raise ReviewError("Select an effect description from the model draft")


class ReviewStore:
    """Locked review state.  ``create`` returns a store; transitions return data."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.review_path = self.path / "review.json"

    @contextmanager
    def _lock(self):
        if not self.path.is_dir():
            raise ReviewError("Review directory does not exist")
        with (self.path / ".review.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self, name: str):
        try:
            with (self.path / name).open(encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError) as exc:
            raise ReviewError(f"Cannot read valid review artifact: {name}") from exc

    def _write(self, name: str, value) -> None:
        material = _json_bytes(value)
        descriptor, temporary = tempfile.mkstemp(prefix=".review-write-", dir=self.path)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(material + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path / name)
            descriptor = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _save(self, data: dict) -> None:
        data["state_sha256"] = canonical_sha256(
            {key: value for key, value in data.items() if key != "state_sha256"})
        self._write("review.json", data)

    @staticmethod
    def _context(data: dict) -> dict:
        return {field: data[field] for field in _CONTEXT_FIELDS}

    @staticmethod
    def _approval_binding(data: dict) -> str:
        return canonical_sha256({
            "context_sha256": data["context_sha256"],
            "plan": data["plan"],
            "revision": data["revision"],
            "reviewed_steps": data["reviewed_steps"],
        })

    def _load(self) -> dict:
        data = self._read("review.json")
        if not isinstance(data, dict) or set(data) != _REVIEW_FIELDS:
            raise ReviewError("Review schema fields do not match this version")
        if (type(data["schema_version"]) is not int
                or data["schema_version"] != SCHEMA_VERSION or data["version"] != VERSION):
            raise ReviewError("Review schema/version is not supported")
        if type(data["revision"]) is not int or data["revision"] < 0:
            raise ReviewError("Invalid review revision")
        state = {key: value for key, value in data.items() if key != "state_sha256"}
        if canonical_sha256(state) != data["state_sha256"]:
            raise ReviewError("Review integrity check failed")
        context_hash = canonical_sha256(self._context(data))
        seal = self._read(".review-context.json")
        if (context_hash != data["context_sha256"]
                or seal != {"id": data["id"], "context_sha256": context_hash}):
            raise ReviewError("Immutable review context was changed")
        if initial_observation_sha256(self._read("initial_observation.json")) != data["initial_observation_sha256"]:
            raise ReviewError("Initial observation was changed after plan preparation")
        _runtime(data["runtime_sha256"])
        _parse_lossless(data["draft_plan"])
        _parse_lossless(data["plan"])
        _selector_edits(data["draft_plan"], data["plan"])
        _indices(data["reviewed_steps"], len(data["plan"]["steps"]))
        if data["status"] not in ("draft", "approved", "consumed"):
            raise ReviewError("Invalid review status")
        if data["status"] == "draft":
            if data["approved"] is not None or data["claim"] is not None:
                raise ReviewError("Draft review cannot carry approval or claim")
        else:
            approved = data["approved"]
            if not isinstance(approved, dict) or set(approved) != {
                    "at", "revision", "plan_sha256", "binding_sha256"}:
                raise ReviewError("Missing or malformed approval")
            if (approved["revision"] != data["revision"]
                    or approved["plan_sha256"] != canonical_sha256(data["plan"])
                    or approved["binding_sha256"] != self._approval_binding(data)):
                raise ReviewError("Approved plan/context was changed")
            _indices(data["reviewed_steps"], len(data["plan"]["steps"]), complete=True)
            validate_review_plan(data["plan"])
        claimed = self.path / ".review-claimed.json"
        if claimed.exists():
            marker = self._read(".review-claimed.json")
            if data["status"] != "consumed" or marker != data["claim"]:
                raise ReviewError("Approval has already been consumed; review cannot be replayed")
        elif data["status"] == "consumed" or data["claim"] is not None:
            raise ReviewError("Consumed review claim is missing or inconsistent")
        return data

    @staticmethod
    def _public(data: dict) -> dict:
        result = _copy(data)
        result["validation"] = plan_validation(result["plan"])
        return result

    @classmethod
    def create(cls, path, *, task_key: str, episode_index: int, task_prompt: str,
               initial_observation: dict, plan: Mapping, runtime_sha256: dict,
               preparation=None) -> "ReviewStore":
        clean = _parse_lossless(plan)
        if not isinstance(task_key, str) or not task_key:
            raise ReviewError("task_key must be nonempty")
        if type(episode_index) is not int or episode_index < 0:
            raise ReviewError("episode_index must be a nonnegative integer")
        if not isinstance(task_prompt, str) or not task_prompt:
            raise ReviewError("task_prompt must be nonempty")
        runtime = _runtime(runtime_sha256)
        initial = _copy(initial_observation)
        if not isinstance(initial, dict):
            raise ReviewError("initial_observation must be an object")
        store = cls(path)
        store.path.mkdir(parents=True, exist_ok=True)
        with store._lock():
            if any((store.path / name).exists() for name in
                   ("review.json", ".review-context.json", ".review-claimed.json", "initial_observation.json")):
                raise ReviewError("Review already exists; prepare a new review for each test")
            data = {
                "schema_version": SCHEMA_VERSION, "version": VERSION,
                "id": str(uuid.uuid4()), "created_at": _now(),
                "task_key": task_key, "episode_index": episode_index,
                "task_prompt": task_prompt,
                "initial_observation_sha256": initial_observation_sha256(initial),
                "runtime_sha256": runtime, "draft_plan": clean,
                "preparation": _copy(preparation),
                "plan": _copy(clean), "revision": 0, "status": "draft",
                "reviewed_steps": [], "approved": None, "claim": None,
            }
            data["context_sha256"] = canonical_sha256(store._context(data))
            store._write(".review-context.json", {
                "id": data["id"], "context_sha256": data["context_sha256"]})
            store._write("initial_observation.json", initial)
            store._save(data)
        return store

    def load(self) -> dict:
        with self._lock():
            return self._public(self._load())

    @staticmethod
    def _editable(data: dict, expected_revision: int) -> None:
        if type(expected_revision) is not int or data["revision"] != expected_revision:
            raise ReviewError("Stale review revision; reload before saving or approving")
        if data["status"] == "consumed":
            raise ReviewError("Approval has already been consumed; prepare a new task review")

    def update(self, plan: Mapping, *, expected_revision: int, reviewed_steps) -> dict:
        clean = _parse_lossless(plan)
        with self._lock():
            data = self._load()
            self._editable(data, expected_revision)
            _selector_edits(data["draft_plan"], clean)
            checked = _indices(reviewed_steps, len(clean["steps"]))
            if clean != data["plan"]:
                checked = []
            data.update(plan=clean, revision=data["revision"] + 1, status="draft",
                        reviewed_steps=checked, approved=None, claim=None)
            self._save(data)
            return self._public(data)

    def approve(self, *, expected_revision: int, reviewed_steps) -> dict:
        with self._lock():
            data = self._load()
            self._editable(data, expected_revision)
            validate_review_plan(data["plan"])
            data["reviewed_steps"] = _indices(
                reviewed_steps, len(data["plan"]["steps"]), complete=True)
            data["status"] = "approved"
            data["approved"] = {
                "at": _now(), "revision": data["revision"],
                "plan_sha256": canonical_sha256(data["plan"]),
                "binding_sha256": self._approval_binding(data),
            }
            self._save(data)
            return self._public(data)

    def claim(self, *, output, task_key: str, episode_index: int,
              initial_observation_sha256: str, runtime_sha256: dict) -> dict:
        """Consume one approval before the first test action, under the same lock."""
        runtime = _runtime(runtime_sha256)
        with self._lock():
            data = self._load()
            if data["status"] != "approved":
                raise ReviewError("A fresh user-approved review is required before every test")
            if (task_key != data["task_key"] or type(episode_index) is not int
                    or episode_index != data["episode_index"]):
                raise ReviewError("Approved task/episode does not match this test")
            if initial_observation_sha256 != data["initial_observation_sha256"]:
                raise ReviewError("Initial observation does not match the reviewed task reset")
            if runtime != data["runtime_sha256"]:
                raise ReviewError("Runtime code changed; prepare and review the plan again")
            claim = {
                "at": _now(), "id": data["id"],
                "approval_binding_sha256": data["approved"]["binding_sha256"],
                "output": str(Path(output).expanduser().resolve()),
            }
            # Commit the single-use marker first: a crash must fail closed, never
            # make an approval reusable after a test might have started.
            self._write(".review-claimed.json", claim)
            data.update(status="consumed", claim=claim)
            self._save(data)
            return self._public(data)
