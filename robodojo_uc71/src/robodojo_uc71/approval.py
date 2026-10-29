"""File-bound, single-use approval. Calling authorize requires prior human approval.

The caller owns the child process and must obtain expected_worker identity from
that process's authenticated local channel, never from the READY payload alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid


class ApprovalError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifiedFileHashCache:
    """Cryptographically verify once per unchanged regular-file identity.

    Evidence/code is protected from the unprivileged worker. Cache reuse requires
    the same device, inode, size, mtime_ns and ctime_ns. A metadata change forces
    hashing, including after same-size writes with restored mtime. A new cache
    entry requires ctime at least one second old: this host's roughly 4ms kernel
    timestamp ticks can otherwise hide an immediate same-size write. First
    verification waits at most 1.25s for metadata to settle, then fails closed.
    Pre/post stat disagreement rejects hashing races. This does not defend against privileged
    host access rewriting file data and kernel metadata together. Public
    file_sha256 remains uncached. A forked child starts with an empty cache.
    """
    def __init__(self):
        self._pid = os.getpid()
        self._records = {}

    def verify(self, filename, expected):
        if self._pid != os.getpid():
            self._records.clear()
            self._pid = os.getpid()
        _digest(expected, "bound file")
        try:
            path = Path(filename)
            if not path.is_absolute() or path.resolve(strict=True) != path:
                raise ApprovalError("Bound file must be canonical and nonsymlinked")
            def identity():
                s = path.lstat()
                if not stat.S_ISREG(s.st_mode):
                    raise ApprovalError("Bound file must remain regular")
                return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
            before = identity()
            cached = self._records.get(str(path))
            if cached is None or cached[0] != before:
                deadline = time.monotonic() + 1.25
                while time.time_ns() - before[4] < 1_000_000_000:
                    if time.monotonic() >= deadline:
                        raise ApprovalError("Bound file metadata did not stabilize")
                    time.sleep(.05)
                    before = identity()
                digest = file_sha256(path)
                if identity() != before or path.resolve(strict=True) != path:
                    raise ApprovalError("Bound file changed during verification")
                cached = (before, digest)
                self._records[str(path)] = cached
            if cached[1] != expected:
                raise ApprovalError("Bound file changed")
        except (OSError, TypeError, ValueError) as error:
            raise ApprovalError("Bound file unavailable") from error


_manifest_file_cache = VerifiedFileHashCache()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ApprovalError(f"Invalid {name} digest")
    return value


def _identity(value: Mapping) -> dict:
    pid, nonce = value.get("pid"), value.get("boot_nonce")
    if type(pid) is not int or pid <= 0 or not isinstance(nonce, str) or len(nonce) < 16:
        raise ApprovalError("Invalid worker identity")
    return {"pid": pid, "boot_nonce": nonce}


def _exclusive_json(path: Path, value: Mapping) -> None:
    """A partial file still consumes the name; failures must never enable reuse."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ApprovalError(f"Single-use record already exists: {path.name}") from error
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def build_manifest(*, task_key: str, attempt_dir: str | Path, bindings: Mapping[str, str | Path],
                   configuration: Mapping, sensor_sha256: str,
                   start_state_sha256: str, preview_sha256: str,
                   preparation_worker: Mapping,
                   worker_mode: str = "fresh", idle_check_sha256: str | None = None) -> dict:
    """Bind all relevant code/core/config/layout/assets and the reviewed start.

    bindings must include the complete frozen inventory, not just its filenames.
    This helper cannot discover which source/asset dependencies are relevant.
    """
    if not isinstance(task_key, str) or not task_key or not bindings:
        raise ApprovalError("Task and nonempty file bindings are required")
    if worker_mode != "fresh" or idle_check_sha256 is not None:
        raise ApprovalError("Preview-worker reuse is disabled pending native idle validation")
    frozen_files = {}
    for label, filename in sorted(bindings.items()):
        if not isinstance(label, str) or not label:
            raise ApprovalError("File binding needs a label")
        path = Path(filename).resolve(strict=True)
        if not path.is_file():
            raise ApprovalError("Only regular files may be bound")
        frozen_files[label] = {"path": str(path), "sha256": file_sha256(path)}
    manifest = {
        "schema": "robodojo-uc71-approval-manifest-1", "id": str(uuid.uuid4()),
        "created_at_utc": _now(), "task_key": task_key, "attempts": 1,
        "attempt_dir": str(Path(attempt_dir).resolve()),
        "files": frozen_files, "configuration": json.loads(json.dumps(configuration, allow_nan=False)),
        "sensor_sha256": _digest(sensor_sha256, "sensor"),
        "start_state_sha256": _digest(start_state_sha256, "start state"),
        "preview_sha256": _digest(preview_sha256, "preview"),
        "preparation_worker": _identity(preparation_worker), "worker_mode": worker_mode,
        "idle_check_sha256": idle_check_sha256,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_manifest(manifest: Mapping, *, verify_files: bool = True) -> None:
    if manifest.get("schema") != "robodojo-uc71-approval-manifest-1" or manifest.get("attempts") != 1:
        raise ApprovalError("Unsupported approval manifest")
    if manifest.get("worker_mode") != "fresh" or manifest.get("idle_check_sha256") is not None:
        raise ApprovalError("Preview-worker reuse is disabled pending native idle validation")
    expected = _digest(manifest.get("manifest_sha256"), "manifest")
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if canonical_sha256(content) != expected:
        raise ApprovalError("Manifest content changed")
    if verify_files:
        for entry in manifest["files"].values():
            _manifest_file_cache.verify(entry["path"], entry["sha256"])


def authorize(manifest: Mapping, attempt_dir: str | Path, *, user_message: str,
              approved_manifest_sha256: str, approved_preview_sha256: str) -> dict:
    """Record explicit human approval; never call as part of prepare/preflight."""
    verify_manifest(manifest)
    if not isinstance(user_message, str) or not user_message.strip():
        raise ApprovalError("Explicit approval message is required")
    if approved_manifest_sha256 != manifest["manifest_sha256"] or approved_preview_sha256 != manifest["preview_sha256"]:
        raise ApprovalError("Approval does not identify the displayed manifest and preview")
    directory = Path(attempt_dir)
    if str(directory.resolve()) != manifest.get("attempt_dir"):
        raise ApprovalError("Attempt directory differs from the frozen manifest")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "consumed.json").exists():
        raise ApprovalError("Attempt was already consumed")
    record = {
        "schema": "robodojo-uc71-human-approval-1", "approved_at_utc": _now(),
        "attempt_dir": str(directory.resolve()),
        "manifest_sha256": manifest["manifest_sha256"], "preview_sha256": manifest["preview_sha256"],
        "user_message": user_message, "approval_mode": "manual_chat_review",
        "manual_browser_review": False, "attempts": 1,
    }
    _exclusive_json(directory / "approval.json", record)
    return record


def validate_ready(manifest: Mapping, ready: Mapping, *, expected_worker: Mapping) -> None:
    """Validate READY against a live child identity tracked independently by caller."""
    expected = _identity(expected_worker)
    actual = _identity(ready)
    if actual != expected:
        raise ApprovalError("READY is not from the tracked live worker")
    if ready.get("state") != "READY" or ready.get("reset_validated") is not True or type(ready.get("policy_actions")) is not int or ready["policy_actions"] != 0:
        raise ApprovalError("Worker is not reset-ready before policy actions")
    for key in ("manifest_sha256", "sensor_sha256", "start_state_sha256"):
        if ready.get(key) != manifest[key]:
            raise ApprovalError(f"READY {key} differs from approval binding")
    prepared = _identity(manifest["preparation_worker"])
    if manifest["worker_mode"] == "fresh":
        if actual["boot_nonce"] == prepared["boot_nonce"]:
            raise ApprovalError("Fresh worker required; preview worker cannot be reused")
    else:
        raise ApprovalError("Preview-worker reuse is disabled pending native idle validation")


def consume_authorization(manifest: Mapping, attempt_dir: str | Path, ready: Mapping,
                          *, expected_worker: Mapping) -> dict:
    """Verify a READY worker first, then atomically claim exactly one attempt.

    Caller must verify that its subprocess is alive immediately before this call.
    No request/action may be sent before this returns. A crash after the claim
    cannot be retried with the same directory or approval.
    """
    verify_manifest(manifest)
    validate_ready(manifest, ready, expected_worker=expected_worker)
    directory = Path(attempt_dir)
    if str(directory.resolve()) != manifest.get("attempt_dir"):
        raise ApprovalError("Attempt directory differs from the frozen manifest")
    try:
        approval = json.loads((directory / "approval.json").read_text())
    except (OSError, ValueError) as error:
        raise ApprovalError("Valid explicit approval is missing") from error
    if (approval.get("schema") != "robodojo-uc71-human-approval-1"
            or approval.get("manifest_sha256") != manifest["manifest_sha256"]
            or approval.get("preview_sha256") != manifest["preview_sha256"]
            or approval.get("attempt_dir") != str(directory.resolve())
            or approval.get("attempts") != 1 or not approval.get("user_message", "").strip()):
        raise ApprovalError("Approval record does not match the attempt")
    record = {
        "schema": "robodojo-uc71-consumed-1", "consumed_at_utc": _now(),
        "manifest_sha256": manifest["manifest_sha256"],
        "approval_sha256": file_sha256(directory / "approval.json"),
        "worker": _identity(expected_worker), "ready_sha256": canonical_sha256(ready),
        "claim_nonce": str(uuid.uuid4()), "attempts": 1,
    }
    _exclusive_json(directory / "consumed.json", record)
    return record
