from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os

import pytest

from robodojo_uc71.approval import (ApprovalError, authorize, build_manifest,
    consume_authorization, file_sha256, verify_manifest)
import robodojo_uc71.approval as approval_module


def setup_case(tmp_path, *, reuse=False):
    code = tmp_path / "code.py"
    code.write_text("frozen code\n")
    idle = tmp_path / "idle.json"
    idle.write_text('{"passed":true,"duration_s":300}\n')
    prep = {"pid": 100, "boot_nonce": "preview-worker-nonce-0001"}
    worker = prep if reuse else {"pid": 101, "boot_nonce": "fresh-worker-nonce-000002"}
    directory = tmp_path / "attempt"
    manifest = build_manifest(task_key="make_toast", attempt_dir=directory, bindings={"code": code, "idle": idle},
        configuration={"native_limit": 1400}, sensor_sha256="a"*64,
        start_state_sha256="b"*64, preview_sha256="c"*64,
        preparation_worker=prep,
        worker_mode="reuse_preview_after_idle_check" if reuse else "fresh",
        idle_check_sha256=file_sha256(idle) if reuse else None)
    ready = {**worker, "state": "READY", "reset_validated": True, "policy_actions": 0,
             **{k: manifest[k] for k in ["manifest_sha256", "sensor_sha256", "start_state_sha256"]}}
    if reuse:
        ready["idle_check_sha256"] = manifest["idle_check_sha256"]
    directory = tmp_path / "attempt"
    return code, manifest, ready, worker, directory


def approve(manifest, directory):
    return authorize(manifest, directory, user_message="approve this preview",
        approved_manifest_sha256=manifest["manifest_sha256"], approved_preview_sha256=manifest["preview_sha256"])


def test_ready_mismatch_does_not_consume_approval(tmp_path):
    _, manifest, ready, worker, directory = setup_case(tmp_path)
    approve(manifest, directory)
    wrong = {**ready, "start_state_sha256": "d"*64}
    with pytest.raises(ApprovalError):
        consume_authorization(manifest, directory, wrong, expected_worker=worker)
    assert not (directory / "consumed.json").exists()
    consume_authorization(manifest, directory, ready, expected_worker=worker)
    with pytest.raises(ApprovalError):
        consume_authorization(manifest, directory, ready, expected_worker=worker)


def test_concurrent_claim_is_exactly_once(tmp_path):
    _, manifest, ready, worker, directory = setup_case(tmp_path)
    approve(manifest, directory)
    def claim():
        try:
            consume_authorization(manifest, directory, ready, expected_worker=worker)
            return "claimed"
        except ApprovalError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: claim(), range(2))) == ["claimed", "rejected"]
    assert json.loads((directory / "consumed.json").read_text())["worker"] == worker


def test_frozen_input_mutation_rejected_before_claim(tmp_path):
    code, manifest, ready, worker, directory = setup_case(tmp_path)
    approve(manifest, directory)
    code.write_text("changed after approval")
    with pytest.raises(ApprovalError, match="Bound file changed"):
        consume_authorization(manifest, directory, ready, expected_worker=worker)
    assert not (directory / "consumed.json").exists()


def test_manifest_cannot_be_rewritten_or_approval_replaced(tmp_path):
    _, manifest, _, _, directory = setup_case(tmp_path)
    changed = deepcopy(manifest)
    changed["configuration"]["native_limit"] = 2000
    with pytest.raises(ApprovalError, match="Manifest content changed"):
        verify_manifest(changed)
    approve(manifest, directory)
    with pytest.raises(ApprovalError):
        approve(manifest, directory)


@pytest.mark.parametrize("change", [
    {"pid": 999}, {"boot_nonce": "wrong-process-nonce"}, {"state": "RUNNING"},
    {"policy_actions": 1}, {"policy_actions": False}, {"reset_validated": False},
])
def test_worker_cannot_claim_using_untrusted_identity_or_unready_state(tmp_path, change):
    _, manifest, ready, worker, directory = setup_case(tmp_path)
    approve(manifest, directory)
    with pytest.raises(ApprovalError):
        consume_authorization(manifest, directory, {**ready, **change}, expected_worker=worker)
    assert not (directory / "consumed.json").exists()


def test_preview_worker_reuse_disabled_until_native_idle_gate_exists(tmp_path):
    # Even a claimed passed300s file cannot enable an unimplemented live gate.
    with pytest.raises(ApprovalError, match="reuse is disabled"):
        setup_case(tmp_path, reuse=True)


def test_fresh_worker_rule_rejects_preview_identity(tmp_path):
    _, manifest, ready, _, directory = setup_case(tmp_path)
    worker = manifest["preparation_worker"]
    approve(manifest, directory)
    with pytest.raises(ApprovalError, match="Fresh worker required"):
        consume_authorization(manifest, directory, {**ready, **worker}, expected_worker=worker)


def test_authorize_requires_exact_preview_and_nonempty_human_message(tmp_path):
    _, manifest, _, _, directory = setup_case(tmp_path)
    for message, preview in [("", manifest["preview_sha256"]), ("approve", "f"*64)]:
        with pytest.raises(ApprovalError):
            authorize(manifest, directory, user_message=message,
                approved_manifest_sha256=manifest["manifest_sha256"], approved_preview_sha256=preview)
    assert not (directory / "approval.json").exists()


def test_copied_approval_cannot_start_another_output_directory(tmp_path):
    _, manifest, ready, worker, directory = setup_case(tmp_path)
    approve(manifest, directory)
    copied = tmp_path / "copied_attempt"
    copied.mkdir()
    (copied / "approval.json").write_bytes((directory / "approval.json").read_bytes())
    with pytest.raises(ApprovalError, match="directory differs"):
        consume_authorization(manifest, copied, ready, expected_worker=worker)
    with pytest.raises(ApprovalError, match="directory differs"):
        approve(manifest, copied)
    assert not (copied / "consumed.json").exists()


def test_manifest_file_hashes_are_cached_but_canonical_content_is_always_checked(tmp_path, monkeypatch):
    _, manifest, _, _, _ = setup_case(tmp_path)
    calls = []
    def counted(path):
        calls.append(str(path)); return file_sha256(path)
    monkeypatch.setattr(approval_module, "file_sha256", counted)
    monkeypatch.setattr(approval_module, "_manifest_file_cache", approval_module.VerifiedFileHashCache())
    verify_manifest(manifest)
    assert len(calls) == 2
    verify_manifest(manifest); verify_manifest(manifest)
    assert len(calls) == 2
    manifest["configuration"]["native_limit"] = 1401
    with pytest.raises(ApprovalError, match="Manifest content changed"):
        verify_manifest(manifest)


def test_manifest_cache_rejects_same_size_write_with_restored_mtime(tmp_path):
    code, manifest, _, _, _ = setup_case(tmp_path)
    verify_manifest(manifest)
    before, expected = code.stat(), file_sha256(code)
    code.write_bytes(code.read_bytes().replace(b"frozen", b"Frozen"))
    os.utime(code, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert code.stat().st_size == before.st_size
    assert file_sha256(code) != expected  # Public helper is still uncached.
    with pytest.raises(ApprovalError, match="Bound file changed"):
        verify_manifest(manifest)


@pytest.mark.parametrize("same_bytes", [False, True])
def test_inode_replacement_is_rehashed_even_with_identical_size_and_mtime(tmp_path, monkeypatch, same_bytes):
    code, manifest, _, _, _ = setup_case(tmp_path)
    verify_manifest(manifest)
    before = code.stat()
    replacement = tmp_path / "replacement"
    content = code.read_bytes()
    replacement.write_bytes(content if same_bytes else content.replace(b"frozen", b"Frozen"))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(code)
    calls = []
    def counted(path):
        calls.append(str(path)); return file_sha256(path)
    monkeypatch.setattr(approval_module, "file_sha256", counted)
    if same_bytes:
        verify_manifest(manifest)
    else:
        with pytest.raises(ApprovalError, match="Bound file changed"):
            verify_manifest(manifest)
    assert calls == [str(code)]


def test_pre_post_hash_stat_race_is_rejected_even_if_hasher_returns_expected_digest(tmp_path, monkeypatch):
    code, manifest, _, _, _ = setup_case(tmp_path)
    def raced(path):
        digest = file_sha256(path)
        if str(path) == str(code):
            code.write_bytes(code.read_bytes().replace(b"frozen", b"Frozen"))
        return digest
    monkeypatch.setattr(approval_module, "file_sha256", raced)
    monkeypatch.setattr(approval_module, "_manifest_file_cache", approval_module.VerifiedFileHashCache())
    with pytest.raises(ApprovalError, match="changed during verification"):
        verify_manifest(manifest)


def test_forked_cache_identity_forces_first_full_hash(tmp_path, monkeypatch):
    code, _, _, _, _ = setup_case(tmp_path)
    cache = approval_module.VerifiedFileHashCache()
    expected = file_sha256(code)
    cache.verify(code, expected)
    cache._pid = -1
    calls = []
    def counted(path):
        calls.append(str(path)); return file_sha256(path)
    monkeypatch.setattr(approval_module, "file_sha256", counted)
    cache.verify(code, expected)
    assert calls == [str(code)]
