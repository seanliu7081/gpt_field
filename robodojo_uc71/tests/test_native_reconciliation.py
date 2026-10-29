"""Post-reap callback truth tests; synthetic native sessions and tmp approval only."""
from copy import deepcopy
import json

import pytest

from robodojo_uc71.lifecycle import EpisodeLifecycle
from robodojo_uc71.native_reconciliation import NativeReconciliationError, reconcile_after_reap
from test_native_worker_protocol import Case
from test_native_episode import action


def begun(tmp_path):
    case = Case(tmp_path)
    case.session._registration_id = case.manifest["manifest_sha256"]+":make_toast-registered-native-predicates"
    ready = case.rpc("ready"); claim = case.claim(ready); case.rpc("begin",claim)
    return case,claim


def reconcile(case,claim,callback=lambda _:True):
    # This explicitly synthetic verifier stands in for a parent's real reap proof.
    return reconcile_after_reap(manifest=case.manifest,expected_worker=case.identity,
        expected_consumed_binding=claim,verify_owned_worker_gone=callback)


def mutate(path,change):
    record = json.loads(path.read_text()); change(record); path.write_text(json.dumps(record))


def test_late_success_after_parent_timeout_preserves_original_parent_terminal(tmp_path):
    case,claim = begun(tmp_path)
    try:
        parent = EpisodeLifecycle(); parent.ready()
        parent.start(json.loads((case.attempt/"consumed.json").read_text()))
        parent.reserve_action(); parent.stop("infrastructure_error")
        directory = case.attempt/"parent"; directory.mkdir()
        parent.persist_terminal(directory/"terminal.json")
        original = (directory/"terminal.json").read_bytes()
        case.env.reward_manager.reward = 1.
        case.rpc("action",{"ticket":1,"joint_action":action()})
        case.close_sockets()
        calls = case.env.reward_manager.calls
        result = reconcile(case,claim)
        assert result["reconciled_terminal"]["official_success"] is True
        assert result["verified_native_actions"] == 1 and result["verified_native_substeps"] == 10
        assert result["last_native_result"]["evidence_id"] == claim["claim_nonce"]+":native-action:1"
        assert result["native_terminal_certified_from_receipts"] is True
        timing = json.loads((case.session.output/"action_0001.json").read_text())["timing"]
        assert result["reconciled_terminal"]["completion_time_s"] == (
            timing["native_callback_completed_monotonic_s"]-timing["policy_begin_monotonic_s"])
        assert (directory/"terminal.json").read_bytes() == original
        assert calls == case.env.reward_manager.calls == 1
    finally: case.close_sockets()


def test_success_receipt_is_authoritative_even_if_terminal_commit_was_lost(tmp_path):
    case,claim = begun(tmp_path)
    try:
        case.env.reward_manager.reward = 1.
        case.rpc("action",{"ticket":1,"joint_action":action()})
        (case.session.output/"terminal.json").unlink()
        result = reconcile(case,claim)
        assert result["reconciled_terminal"]["status"] == "success"
        assert result["worker_terminal"] is None and result["worker_terminal_present"] is False
        assert result["unrecorded_dispatches_not_excluded"] is False
    finally: case.close_sockets()


def test_nonterminal_prefix_missing_terminal_remains_unknown(tmp_path):
    case,claim = begun(tmp_path)
    try:
        case.rpc("action",{"ticket":1,"joint_action":action()})
        result = reconcile(case,claim)
        assert result["reconciled_terminal"]["official_success"] is None
        assert result["unrecorded_dispatches_not_excluded"] is True
        assert result["policy_timing"]["elapsed_policy_wall_time_through_last_callback_s"] >= 0
        assert result["policy_timing"]["completion_time_s"] is None
    finally: case.close_sockets()


def test_real_external_stop_with_zero_actions_does_not_create_native_result(tmp_path):
    case,claim = begun(tmp_path)
    try:
        case.rpc("close")
        result = reconcile(case,claim)
        assert result["reconciled_terminal"]["status"] == "user_stop"
        assert result["verified_native_actions"] == 0 and result["last_native_result"] is None
    finally: case.close_sockets()


def test_mandatory_owned_process_reap_verifier_precedes_evidence_reads(tmp_path):
    case,claim = begun(tmp_path)
    try:
        (case.attempt/"consumed.json").unlink()
        with pytest.raises(NativeReconciliationError,match="reaped"):
            reconcile(case,claim,lambda _:False)
    finally: case.close_sockets()


@pytest.mark.parametrize("field,value",[
    ("episode_evidence_id","another-claim"),("layout_sha256","0"*64),
    ("registration_evidence_id","unbound-registration"),("native_sim_step_calls",9),
    ("native_reward_callback_calls",2),("authorization_verified_for_ticket",True),
    ("native_physics_counter_after",311),("physics_time_after_s",1.25),
])
def test_wrong_episode_layout_authorization_or_native_cadence_rejected(tmp_path,field,value):
    case,claim = begun(tmp_path)
    try:
        case.rpc("action",{"ticket":1,"joint_action":action()})
        mutate(case.session.output/"action_0001.json",lambda row:row.update({field:value}))
        with pytest.raises(NativeReconciliationError): reconcile(case,claim)
    finally: case.close_sockets()


def test_no_hidden_physics_between_contiguous_actions(tmp_path):
    case,claim = begun(tmp_path)
    try:
        for ticket in (1,2): case.rpc("action",{"ticket":ticket,"joint_action":action()})
        def change(row):
            row["native_physics_counter_before"] += 1; row["native_physics_counter_after"] += 1
            row["physics_time_before_s"] += .004; row["physics_time_after_s"] += .004
        mutate(case.session.output/"action_0002.json",change)
        with pytest.raises(NativeReconciliationError,match="between"): reconcile(case,claim)
    finally: case.close_sockets()


def test_terminal_success_without_its_callback_receipt_is_rejected(tmp_path):
    case,claim = begun(tmp_path)
    try:
        case.env.reward_manager.reward = 1.
        case.rpc("action",{"ticket":1,"joint_action":action()})
        (case.session.output/"action_0001.json").unlink()
        with pytest.raises(NativeReconciliationError,match="without_matching"): reconcile(case,claim)
    finally: case.close_sockets()


def test_receipt_gap_and_consumed_binding_mutation_are_rejected(tmp_path):
    case,claim = begun(tmp_path)
    try:
        for ticket in (1,2): case.rpc("action",{"ticket":ticket,"joint_action":action()})
        path = case.session.output/"action_0001.json"; original=path.read_bytes(); path.unlink()
        with pytest.raises(NativeReconciliationError,match="contiguous"): reconcile(case,claim)
        path.write_bytes(original)
        mutate(case.attempt/"consumed.json",lambda row:row.update(worker={**case.identity,"pid":999}))
        with pytest.raises(NativeReconciliationError,match="hash_changed"): reconcile(case,claim)
    finally: case.close_sockets()


def test_commit_before_lifecycle_update_race_preserves_native_callback_success(tmp_path):
    case,claim = begun(tmp_path)
    try:
        case.env.reward_manager.reward = 1.
        case.rpc("action",{"ticket":1,"joint_action":action()})
        terminal_path = case.session.output/"terminal.json"; terminal_path.unlink()
        old = EpisodeLifecycle(); old.ready(); old.start(json.loads((case.attempt/"consumed.json").read_text()))
        old.reserve_action(); old.stop("infrastructure_error"); old.persist_terminal(terminal_path)
        result = reconcile(case,claim)
        assert result["worker_terminal"]["record"]["official_success"] is None
        assert result["reconciled_terminal"]["official_success"] is True
    finally: case.close_sockets()


@pytest.mark.parametrize("kind",["begin_changed","overlap","callback_out_of_order"])
def test_consistent_ordered_real_worker_timing_required(tmp_path,kind):
    case,claim = begun(tmp_path)
    try:
        for ticket in (1,2): case.rpc("action",{"ticket":ticket,"joint_action":action()})
        first = json.loads((case.session.output/"action_0001.json").read_text())["timing"]
        def change(row):
            timing = row["timing"]
            if kind == "begin_changed": timing["policy_begin_monotonic_s"] -= 1
            if kind == "overlap": timing["dispatch_started_monotonic_s"] = first["dispatch_started_monotonic_s"]
            if kind == "callback_out_of_order": timing["native_callback_completed_monotonic_s"] = timing["dispatch_completed_monotonic_s"]+1
        mutate(case.session.output/"action_0002.json",change)
        with pytest.raises(NativeReconciliationError,match="timing"): reconcile(case,claim)
    finally: case.close_sockets()
