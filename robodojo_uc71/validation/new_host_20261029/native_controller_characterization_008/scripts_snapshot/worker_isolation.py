"""Separate simulator UID and root-owned approval/credential boundaries on Linux.

No service or process is launched by this module. The parent prepares only fresh
directories, shares approval records read-only and uses the returned Popen user
arguments. The child must enforce no_new_privs before importing native plugins.
This is Unix-user isolation, not a filesystem namespace or a network sandbox.
"""
import ctypes
import errno
import os
from pathlib import Path
import pwd
import stat


WORKER_UID = 65534
WORKER_GID = 65534


class WorkerIsolationError(RuntimeError):
    pass


def _parent():
    if os.getuid() != 0 or os.geteuid() != 0:
        raise WorkerIsolationError("root_supervisor_required_for_separate_worker_uid")
    account = pwd.getpwuid(WORKER_UID)
    if account.pw_name != "nobody" or account.pw_gid != WORKER_GID:
        raise WorkerIsolationError("expected_existing_worker_account_unavailable")


def _plain_dir(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise WorkerIsolationError("real_directory_required")
    return info


def prepare_access(attempt_dir, runtime_dir):
    """Create fresh parent/worker directories without changing existing trees."""
    _parent()
    attempt, runtime = Path(attempt_dir).absolute(), Path(runtime_dir).absolute()
    if attempt.exists() or runtime.exists() or attempt == runtime:
        raise WorkerIsolationError("fresh_distinct_attempt_and_runtime_required")
    # Existing parents must be real; no recursive chown of the shared workspace.
    for parent in {attempt.parent, runtime.parent}:
        _plain_dir(parent)
        if parent.resolve() != parent:
            raise WorkerIsolationError("symlink_parent_rejected")
    attempt.mkdir(mode=0o750)
    os.chown(attempt, 0, WORKER_GID)
    os.chmod(attempt, 0o750)
    worker = attempt / "worker"
    worker.mkdir(mode=0o700)
    os.chown(worker, WORKER_UID, WORKER_GID)
    runtime.mkdir(mode=0o700)
    os.chown(runtime, WORKER_UID, WORKER_GID)
    return {"schema": "robodojo-worker-unix-isolation-1", "worker_uid": WORKER_UID,
            "worker_gid": WORKER_GID, "supplementary_groups": [],
            "attempt_dir": str(attempt), "worker_output": str(worker), "runtime_dir": str(runtime),
            "namespace_isolation": False, "network_isolation": False,
            "full_test_authorized": False}


def share_approval_record(attempt_dir, name):
    """Grant only group-read on an existing root-owned single-link approval file."""
    _parent()
    if name not in ("approval.json", "consumed.json"):
        raise WorkerIsolationError("only_approval_records_may_be_shared")
    directory = Path(attempt_dir).absolute()
    info = _plain_dir(directory)
    if info.st_uid != 0 or info.st_gid != WORKER_GID or stat.S_IMODE(info.st_mode) != 0o750:
        raise WorkerIsolationError("root_owned_readonly_attempt_parent_required")
    fd = os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        entry = os.fstat(fd)
        if (not stat.S_ISREG(entry.st_mode) or entry.st_uid != 0 or entry.st_nlink != 1
                or stat.S_IMODE(entry.st_mode) not in (0o600, 0o640)):
            raise WorkerIsolationError("unsafe_approval_file_ownership_or_mode")
        os.fchown(fd, 0, WORKER_GID)
        os.fchmod(fd, 0o640)
        os.fsync(fd)
    finally:
        os.close(fd)


def popen_identity():
    _parent()
    return {"user": WORKER_UID, "group": WORKER_GID, "extra_groups": [], "start_new_session": True}


def enforce_child_boundary(*, credential_store, parent_pid):
    """Reject a privileged worker and prove credential/proc access is denied.

Only attempts to open the protected directory and parent environ are made;
credential values are never read, returned or logged.
"""
    if (os.getuid() != WORKER_UID or os.geteuid() != WORKER_UID
            or os.getgid() != WORKER_GID or os.getegid() != WORKER_GID or os.getgroups()):
        raise WorkerIsolationError("separate_uid_gid_and_empty_supplementary_groups_required")
    if (type(parent_pid) is not int or parent_pid <= 1 or parent_pid != os.getppid()
            or Path(f"/proc/{parent_pid}").stat().st_uid != 0):
        raise WorkerIsolationError("actual_root_supervisor_pid_required")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise WorkerIsolationError("no_new_privs_failed")
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    if (status.get("NoNewPrivs") != "1" or any(int(status[k], 16) != 0 for k in
                                             ("CapEff", "CapPrm", "CapInh", "CapAmb"))):
        raise WorkerIsolationError("privilege_or_capability_remains")
    denied = {}
    paths = {"credential_directory": (Path(credential_store), os.O_RDONLY | os.O_DIRECTORY),
             "parent_environment": (Path(f"/proc/{parent_pid}/environ"), os.O_RDONLY)}
    for label, (path, flags) in paths.items():
        try:
            fd = os.open(path, flags | os.O_NOFOLLOW)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EPERM):
                raise WorkerIsolationError("protected_resource_unavailable_without_permission_proof") from None
            denied[label] = True
        else:
            os.close(fd)
            raise WorkerIsolationError("worker_can_open_protected_parent_resource")
    return {"schema": "robodojo-worker-unix-isolation-proof-1", "pid": os.getpid(),
            "uid": os.getuid(), "gid": os.getgid(), "supplementary_groups": os.getgroups(),
            "no_new_privs": True, "effective_permitted_inheritable_ambient_capabilities": 0,
            "access_denied": denied, "namespace_isolation": False, "network_isolation": False}
