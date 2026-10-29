"""Precreated restricted-worker output checks; temporary synthetic fixtures."""
import os

import pytest

from robodojo_uc71.approval import file_sha256
from robodojo_uc71.native_worker_protocol import NativeWorkerSession, WorkerSessionError
from test_native_worker_protocol import Case


def replace_session(case, *, enabled=True):
    layout = case.session._layout
    case.session = NativeWorkerSession(case.worker, env=case.env, manifest=case.manifest,
        worker_identity=case.identity, verify_runtime_readiness=case.readiness,
        measure_robot=case.measure, capture_rgb=case.capture, expected_layout_path=layout,
        expected_layout_sha256=file_sha256(layout), registration_evidence_id="synthetic-registration",
        initial_gripper_commands={"left":"open", "right":"open"}, precreated_worker_output=enabled)


def test_parent_precreated_private_worker_directory_is_accepted(tmp_path):
    case = Case(tmp_path)
    try:
        case.session.output.chmod(0o700)
        replace_session(case)
        assert case.rpc("ready")["state"] == "READY"
        assert not (case.attempt/"approval.json").exists()
    finally:
        case.close_sockets()


@pytest.mark.parametrize("kind", ["mode", "owner", "nonempty", "symlink", "file", "option", "default"])
def test_unsafe_or_reused_worker_directory_is_rejected(tmp_path, monkeypatch, kind):
    case = Case(tmp_path)
    try:
        output = case.session.output
        output.chmod(0o700)
        if kind == "mode": output.chmod(0o750)
        if kind == "owner": monkeypatch.setattr(os, "geteuid", lambda: output.stat().st_uid + 1)
        if kind == "nonempty": (output/"old-attempt.json").write_text("{}")
        if kind == "symlink":
            output.rmdir(); target = tmp_path/"outside"; target.mkdir(mode=0o700); output.symlink_to(target)
        if kind == "file": output.rmdir(); output.write_text("not a directory")
        with pytest.raises((WorkerSessionError, FileExistsError)):
            replace_session(case, enabled=1 if kind == "option" else False if kind == "default" else True)
    finally:
        case.close_sockets()
