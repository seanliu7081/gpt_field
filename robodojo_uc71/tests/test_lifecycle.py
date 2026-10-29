import json

import pytest

from robodojo_uc71.lifecycle import EpisodeLifecycle, LifecycleError, NativeResult, load_terminal_or_crash


def running(limit=1400):
    run = EpisodeLifecycle(action_limit=limit)
    run.ready()
    run.start({"schema": "robodojo-uc71-consumed-1", "attempts": 1,
               "claim_nonce": "unique", "manifest_sha256": "a"*64, "ready_sha256": "b"*64})
    return run


def result(index, *, success=False, end=False, verified=True, checks=True):
    return NativeResult(index, 1.0 if success else 0.0, end or success, checks, verified,
                        f"native-evidence-{index}", 50)


def test_exact_native_horizon_never_dispatches_1401():
    run = running()
    for n in range(1, 1401):
        ticket = run.reserve_action()
        assert ticket == n
        run.complete_action(ticket, result(n))
    terminal = run.terminal_record()
    assert terminal["status"] == "horizon_exhausted"
    assert terminal["official_success"] is False
    assert terminal["confirmed_actions"] == terminal["submitted_actions"] == 1400
    assert terminal["completion_time_eligible"] is False
    with pytest.raises(LifecycleError):
        run.reserve_action()
    with pytest.raises(LifecycleError):
        run.start({})


def test_success_at_horizon_takes_precedence_and_cleanup_cannot_erase_it(tmp_path):
    run = running(2)
    run.complete_action(run.reserve_action(), result(1))
    run.complete_action(run.reserve_action(), result(2, success=True))
    saved = run.terminal_record()
    path = tmp_path / "terminal.json"
    run.persist_terminal(path)
    run.record_cleanup_failure("worker_shutdown_failed")
    run.stop("infrastructure_error")
    assert run.terminal_record() == saved
    assert load_terminal_or_crash(path)["status"] == "success"
    with pytest.raises(FileExistsError):
        run.persist_terminal(path)


def test_no_default_success_before_verified_native_callback():
    run = running()
    assert run.snapshot()["terminal"] is None
    ticket = run.reserve_action()
    with pytest.raises(LifecycleError, match="Verified"):
        run.complete_action(ticket, result(ticket, success=True, verified=False))
    run.stop("infrastructure_error")
    record = run.terminal_record()
    assert record["official_success"] is None
    assert record["unconfirmed_action"] == 1 and record["confirmed_actions"] == 0


def test_unregistered_native_checks_cannot_succeed():
    run = running()
    ticket = run.reserve_action()
    with pytest.raises(LifecycleError):
        run.complete_action(ticket, result(ticket, success=True, checks=False))
    assert run.snapshot()["terminal"] is None


def test_pending_action_cannot_repeat_or_stop_before_feedback():
    run = running()
    ticket = run.reserve_action()
    with pytest.raises(LifecycleError):
        run.reserve_action()
    with pytest.raises(LifecycleError):
        run.stop("user_stop")
    with pytest.raises(LifecycleError):
        run.complete_action(True, result(1))
    run.complete_action(ticket, result(ticket))
    run.stop("user_stop")
    assert run.terminal_record()["official_success"] is False


def test_native_failure_is_not_horizon_or_infrastructure():
    run = running()
    run.complete_action(run.reserve_action(), result(1, end=True))
    assert run.terminal_record()["status"] == "native_failure"


def test_missing_truncated_and_unsupported_terminal_are_unknown(tmp_path):
    path = tmp_path / "terminal.json"
    assert load_terminal_or_crash(path)["official_success"] is None
    path.write_text('{"status":')
    assert load_terminal_or_crash(path)["status"] == "infrastructure_error"
    path.write_text(json.dumps({"schema": "robodojo-uc71-terminal-1", "status": "success", "official_success": True}))
    assert load_terminal_or_crash(path)["official_success"] is None


def test_returned_snapshot_cannot_mutate_final_evidence():
    run = running()
    run.complete_action(run.reserve_action(), result(1, success=True))
    snapshot = run.snapshot()
    snapshot["terminal"]["official_success"] = False
    assert run.terminal_record()["official_success"] is True


def test_unknown_terminal_status_and_false_horizon_are_rejected(tmp_path):
    path = tmp_path / "terminal.json"
    for status, confirmed in [("not_an_outcome", 3), ("horizon_exhausted", 3)]:
        path.write_text(json.dumps({"schema":"robodojo-uc71-terminal-1", "status":status,
            "official_success":False, "submitted_actions":3, "confirmed_actions":confirmed,
            "native_action_limit":1400}))
        assert load_terminal_or_crash(path)["official_success"] is None


@pytest.mark.parametrize("bad_evidence", [[], True, 7, "", "   "])
def test_terminal_loader_requires_nonempty_string_evidence(tmp_path,bad_evidence):
    run=running()
    run.complete_action(run.reserve_action(),result(1,success=True))
    record=run.terminal_record()
    record["native_evidence"]["evidence_id"]=bad_evidence
    path=tmp_path/"terminal.json";path.write_text(json.dumps(record))
    assert load_terminal_or_crash(path)["official_success"] is None


@pytest.mark.parametrize("status", ["user_stop","native_failure","horizon_exhausted","infrastructure_error"])
def test_terminal_loader_rejects_status_success_disagreement(tmp_path,status):
    run=running()
    run.stop("user_stop")
    record=run.terminal_record();record["status"]=status;record["official_success"]=True
    path=tmp_path/"terminal.json";path.write_text(json.dumps(record))
    assert load_terminal_or_crash(path)["official_success"] is None


def test_terminal_loader_preserves_valid_native_failure_and_horizon(tmp_path):
    for status in ["native_failure","horizon_exhausted"]:
        run=running(1 if status=="horizon_exhausted" else 1400)
        run.complete_action(run.reserve_action(),result(1,end=status=="native_failure"))
        path=tmp_path/(status+".json");run.persist_terminal(path)
        loaded=load_terminal_or_crash(path)
        assert loaded["status"]==status and loaded["official_success"] is False


@pytest.mark.parametrize("field", ["completion_time_s", "native_reward"])
def test_huge_json_integer_in_terminal_is_infrastructure_error(tmp_path, field):
    run = running()
    run.complete_action(run.reserve_action(), result(1, success=True))
    record = run.terminal_record()
    target = record if field == "completion_time_s" else record["native_evidence"]
    target[field] = 10**400
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(record))
    assert load_terminal_or_crash(path)["official_success"] is None


def test_huge_native_reward_is_rejected_without_confirming_action():
    from dataclasses import replace
    run = running()
    ticket = run.reserve_action()
    with pytest.raises(LifecycleError):
        run.complete_action(ticket, replace(result(ticket), native_reward=10**400))
    assert run.confirmed_actions == 0
