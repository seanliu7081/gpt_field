"""Static integrity checks for the task-local official cuRobo installation.

No imports of cuRobo, Isaac Sim or torch and no package installation occur here.
Passing this check establishes file identity, not controller qualification.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from .approval import ApprovalError, VerifiedFileHashCache


SCHEMA = "robodojo-official-control-setup-1"
_files = VerifiedFileHashCache()
_record_cache = {}


def _identity(path):
    info = os.lstat(path)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _unchanged(entry):
    """Check every file, plus directories that can introduce importable files."""
    if entry["pid"] != os.getpid():
        return False
    try:
        if any(_identity(name) != identity for name, identity in entry["files"].items()):
            return False
        for name, identity in entry["directories"].items():
            current = _identity(name)
            # Inside the overlay, changed directory contents require reinventory.
            # Ancestors only guard path replacement; unrelated siblings may change.
            if (current if name in entry["inventory_dirs"] else current[:3]) != identity:
                return False
        # A short timestamp tick must not hide newly added importable code.
        # Listing names is inexpensive; file contents still use their identities.
        for directory, expected in entry["children"].items():
            with os.scandir(directory) as contents:
                if frozenset(item.name for item in contents) != expected:
                    return False
    except OSError:
        return False
    return True


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_setup_record(path: str | Path) -> dict:
    """Verify the exact overlay and native input closure before importing it."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    record = json.loads(raw)
    key = (str(path), hashlib.sha256(raw).hexdigest())
    cached = _record_cache.get(key)
    if cached is not None and _unchanged(cached):
        _verify_base_requirements(record)
        return record
    if (record.get("schema") != SCHEMA or record.get("verified") is not True
            or not all(re.fullmatch(r"[0-9a-f]{40}", record.get(k, ""))
                       for k in ("native_commit", "curobo_commit"))):
        raise ValueError("official_control_setup_record_invalid")
    overlay = Path(record["overlay_target"])
    if not overlay.is_absolute() or overlay.is_symlink() or not overlay.is_dir():
        raise ValueError("official_control_overlay_invalid")
    overlay = overlay.resolve(strict=True)
    expected = set()
    file_identities = {}
    ancestors = set(overlay.parents)
    for group in ("source_files", "installed_files"):
        rows = record.get(group)
        if not isinstance(rows, list) or not rows:
            raise ValueError("official_control_file_closure_missing")
        seen = set()
        for row in rows:
            item = Path(row["path"])
            if (not item.is_absolute() or item.is_symlink() or not item.is_file()
                    or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
                    or str(item) in seen):
                raise ValueError("official_control_file_changed")
            try:
                before = _identity(item)
                _files.verify(str(item), row["sha256"])
                after = _identity(item)
                if before != after:
                    raise ApprovalError("File identity changed during verification")
            except ApprovalError as error:
                raise ValueError("official_control_file_changed") from error
            seen.add(str(item))
            file_identities[str(item)] = after
            ancestors.update(item.parents)
            if group == "installed_files":
                if not item.resolve().is_relative_to(overlay):
                    raise ValueError("official_control_overlay_file_escape")
                expected.add(item.resolve())
    # Unexpected code in an import directory is also an integrity failure.
    # Bytecode is deliberately absent: even PYTHONDONTWRITEBYTECODE reads existing
    # .pyc files, so accepting unbound bytecode would permit different loaded code.
    actual = set()
    directories = {}
    children = {}
    for directory, names, files in os.walk(overlay, followlinks=False):
        children[directory] = frozenset(names + files)
        for name in (directory, *(os.path.join(directory, n) for n in names)):
            identity = _identity(name)
            if not stat.S_ISDIR(identity[2]):
                raise ValueError("official_control_overlay_inventory_changed")
            directories[name] = identity
        for name in files:
            item = Path(directory) / name
            if not stat.S_ISREG(_identity(item)[2]):
                raise ValueError("official_control_overlay_inventory_changed")
            actual.add(item)
    if actual != expected:
        raise ValueError("official_control_overlay_inventory_changed")
    inventory_dirs = set(directories)
    for parent in ancestors:
        identity = _identity(parent)
        if not stat.S_ISDIR(identity[2]):
            raise ValueError("official_control_file_changed")
        directories.setdefault(str(parent), identity[:3])
    _verify_base_requirements(record)
    entry = {"pid": os.getpid(), "files": file_identities,
             "directories": directories, "inventory_dirs": inventory_dirs,
             "children": children}
    if not _unchanged(entry):
        raise ValueError("official_control_files_changed_during_verification")
    # Bound growth when callers verify successive immutable setup generations.
    _record_cache.clear()
    _record_cache[key] = entry
    return record


def _verify_base_requirements(record):
    for entry in record.get("base_requirements", []):
        from importlib.metadata import version
        if version(entry["name"]) != entry["version"]:
            raise ValueError("official_control_base_runtime_changed")


def activate_setup_record(path: str | Path, cache_root: str | Path) -> dict:
    """Activate verified dependencies and redirect caches into the worker tree.

    Call before importing Isaac Sim or cuRobo so that an unverified bundled
    module cannot win import precedence. This does not initialize CUDA or IK.
    """
    record = verify_setup_record(path)
    overlay = Path(record["overlay_target"]).resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"curobo", "warp", "cuda"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename and not Path(filename).resolve().is_relative_to(overlay):
            raise ValueError("official_control_module_already_loaded_from_other_source")
    cache = Path(cache_root).resolve()
    if cache.is_relative_to(overlay):
        raise ValueError("official_control_cache_must_be_outside_overlay")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    for variable, directory in (("WARP_CACHE_PATH", "warp"), ("CUDA_CACHE_PATH", "cuda"),
                                ("XDG_CACHE_HOME", "xdg")):
        target = cache / directory
        target.mkdir(exist_ok=True, mode=0o700)
        os.environ[variable] = str(target)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if str(overlay) not in sys.path:
        sys.path.insert(0, str(overlay))
    # Upstream's runtime output path is independent of XDG_CACHE_HOME. These
    # pinned configuration modules import only pathlib; no CUDA/solver starts.
    curobo_cache = cache / "curobo"
    curobo_cache.mkdir(exist_ok=True, mode=0o700)
    installed = {row["path"]: row["sha256"] for row in record["installed_files"]}
    for name in ("curobo.runtime", "curobo._src.runtime"):
        module = importlib.import_module(name)
        filename = str(Path(module.__file__).resolve(strict=True))
        if filename not in installed:
            raise ValueError("official_control_module_loaded_from_other_source")
        try:
            _files.verify(filename, installed[filename])
        except ApprovalError as error:
            raise ValueError("official_control_module_loaded_from_other_source") from error
        module.cache_dir = str(curobo_cache)
    return record
