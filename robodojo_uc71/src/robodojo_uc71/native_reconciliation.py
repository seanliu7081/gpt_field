"""Parent-only, post-reap verification of durable native callback evidence.

The mandatory caller verifier must establish that this *owned* process has been
reaped (including its tracked generation), not merely that its PID is absent.
This function reads existing approval/consumption/READY and committed receipts;
it never launches, resets, sends, retries, evaluates rewards, writes approval,
or replaces parent/worker terminal files. A fresh in-memory lifecycle replays the
already-consumed evidence only to validate accounting; it is not a new attempt.

The concrete native bootstrap registers predicates with evidence id
``<manifest_sha256>:make_toast-registered-native-predicates``. Episode evidence
ids use the consumed claim nonce, and result ids append ``:native-action:N``.
A native terminal result can therefore survive a lost RPC response or missing
terminal-file commit. An unsupported terminal boolean can never certify it.
All returned data is evaluator/operator-only and must stay out of model input.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields
import math
from pathlib import Path
import re
import uuid

from .approval import canonical_sha256, validate_ready, verify_manifest
from .lifecycle import EpisodeLifecycle, NativeResult, load_terminal_or_crash
from .native_worker_client import _keys, _record


class NativeReconciliationError(RuntimeError):
    pass


def _require(ok, code):
    if not ok:
        raise NativeReconciliationError(code)


def _equal_fields(value, expected):
    return all(type(value.get(k)) is type(v) and value.get(k) == v for k,v in expected.items())


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def reconcile_after_reap(*, manifest: Mapping, expected_worker: Mapping,
                         expected_consumed_binding: Mapping, verify_owned_worker_gone) -> dict:
    """Return verified late native truth; raise on inconsistent or unbound data.

    The caller retains its original terminal separately and persists this return
    value as new evidence. ``reconciled_terminal`` is derived from the contiguous
    callback receipts plus a validated external stop, never from an RPC timeout
    or an unchecked success flag. Missing final terminal evidence is disclosed.
    """
    identity = deepcopy(dict(expected_worker)); binding = deepcopy(dict(expected_consumed_binding))
    _keys(identity,{"pid","boot_nonce"},"invalid_reconciliation_worker")
    _require(type(identity["pid"]) is int and identity["pid"] > 0 and
        isinstance(identity["boot_nonce"],str) and len(identity["boot_nonce"]) >= 16,
        "invalid_reconciliation_worker")
    _require(callable(verify_owned_worker_gone) and verify_owned_worker_gone(deepcopy(identity)) is True,
        "owned_worker_must_be_reaped")
    _keys(binding,{"approval_sha256","consumed_sha256","claim_nonce","ready_sha256"},"invalid_consumed_binding")
    for name in ("approval_sha256","consumed_sha256","ready_sha256"):
        _require(isinstance(binding[name],str) and re.fullmatch(r"[0-9a-f]{64}",binding[name]) is not None,
            "invalid_consumed_digest")
    _require(str(uuid.UUID(binding["claim_nonce"])) == binding["claim_nonce"],"invalid_claim_nonce")
    manifest = deepcopy(dict(manifest)); verify_manifest(manifest)
    attempt = Path(manifest["attempt_dir"])
    _require(attempt.is_absolute() and attempt.resolve(strict=True) == attempt,"invalid_attempt_directory")
    worker = attempt/"worker"
    _require(worker.is_dir() and worker.resolve(strict=True) == worker,"invalid_worker_directory")
    approval,approval_hash = _record(attempt/"approval.json")
    consumed,consumed_hash = _record(attempt/"consumed.json")
    ready,ready_hash = _record(worker/"ready.json")
    _keys(ready,{"pid","boot_nonce","state","reset_validated","policy_actions","manifest_sha256",
        "sensor_sha256","start_state_sha256"},"invalid_stored_ready")
    validate_ready(manifest,ready,expected_worker=identity)
    _require(approval_hash == binding["approval_sha256"] and consumed_hash == binding["consumed_sha256"]
        and canonical_sha256(ready) == binding["ready_sha256"],"claim_artifact_hash_changed")
    _require(_equal_fields(approval,{"schema":"robodojo-uc71-human-approval-1",
        "approval_mode":"manual_chat_review","manual_browser_review":False,
        "manifest_sha256":manifest["manifest_sha256"],"preview_sha256":manifest["preview_sha256"],
        "attempt_dir":str(attempt),"attempts":1}) and isinstance(approval.get("user_message"),str)
        and bool(approval["user_message"].strip()),"human_approval_binding_invalid")
    _require(_equal_fields(consumed,{"schema":"robodojo-uc71-consumed-1","attempts":1,
        "manifest_sha256":manifest["manifest_sha256"],"approval_sha256":approval_hash,
        "worker":identity,"ready_sha256":binding["ready_sha256"],"claim_nonce":binding["claim_nonce"]}),
        "consumed_attempt_binding_invalid")
    layouts = {entry["path"]:entry["sha256"] for entry in manifest["files"].values()
        if entry["path"].endswith("/arx_x5/0/make_toast_0.json")}
    _require(len(layouts) == 1,"unique_manifest_bound_layout0_required")
    layout_path,layout_hash = next(iter(layouts.items()))
    registration_id = manifest["manifest_sha256"]+":make_toast-registered-native-predicates"
    paths = sorted(worker.glob("action_*.json"))
    _require(len(paths) <= 1400,"native_horizon_exceeded")
    _require(all(path.name == f"action_{i:04d}.json" for i,path in enumerate(paths,1)),
        "native_receipts_not_contiguous")
    replay = EpisodeLifecycle(action_limit=1400)
    replay.ready(); replay.start(consumed)
    receipts = []; previous = None; final_result = None; first_timing = last_timing = None
    for ticket,path in enumerate(paths,1):
        receipt,digest = _record(path)
        expected = {"schema":"robodojo-native-action-receipt-1","scope":"evaluator_only",
            "model_input_eligible":False,"calibration_status":"uncalibrated_candidate",
            "episode_evidence_id":binding["claim_nonce"],"registration_evidence_id":registration_id,
            "layout_sha256":layout_hash,"authorization_verified_for_ticket":ticket,"native_action_index":ticket,
            "physics_dt_s":.004,"native_sim_step_calls":10,
            "native_terminal_callback_calls":1,"native_reward_callback_calls":1}
        _require(_equal_fields(receipt,expected),"native_receipt_identity_or_cadence_invalid")
        raw = receipt.get("native_result")
        _keys(raw,{field.name for field in fields(NativeResult)},"invalid_native_result")
        result = NativeResult(**raw)
        _require(type(result.action_index) is int and result.action_index == ticket
            and result.callback_verified is True and result.checks_registered is True
            and type(result.end_flag) is bool and _finite(result.native_reward) and 0 <= result.native_reward <= 1
            and (result.partial_score is None or _finite(result.partial_score) and 0 <= result.partial_score <= 100),
            "invalid_registered_native_result")
        _require(result.evidence_id == f"{binding['claim_nonce']}:native-action:{ticket}","native_result_episode_mismatch")
        callback = receipt.get("native_callback_state")
        _keys(callback,{"reward","partial_score","end_flag","native_success_flag","returned_all_ended"},
            "invalid_native_callback_state")
        _require(_equal_fields(callback,{"reward":result.native_reward,"partial_score":result.partial_score,
            "end_flag":result.end_flag,"returned_all_ended":result.end_flag})
            and type(callback["native_success_flag"]) is bool
            and (result.native_reward <= 1-1e-3 or result.end_flag and callback["native_success_flag"])
            and (callback["native_success_flag"] or result.end_flag)
            and (ticket != 1400 or result.end_flag),"native_callback_state_disagrees")
        before,after = receipt.get("native_physics_counter_before"),receipt.get("native_physics_counter_after")
        start,end = receipt.get("physics_time_before_s"),receipt.get("physics_time_after_s")
        _require(type(before) is int and type(after) is int and before >= 0 and after-before == 10
            and _finite(start) and _finite(end) and start >= 0 and math.isclose(end-start,.04,abs_tol=1e-7,rel_tol=0),
            "native_physics_accounting_invalid")
        if previous is not None:
            _require(before == previous["native_physics_counter_after"]
                and math.isclose(start,previous["physics_time_after_s"],abs_tol=1e-7,rel_tol=0),
                "physics_advanced_between_native_receipts")
        timing = receipt.get("timing")
        names = ("policy_begin_monotonic_s","dispatch_started_monotonic_s",
            "native_callback_completed_monotonic_s","dispatch_completed_monotonic_s")
        _keys(timing,{"clock",*names},"invalid_native_wall_timing")
        values = [timing[name] for name in names]
        _require(timing["clock"] == "time.monotonic" and all(_finite(value) and value >= 0 for value in values)
            and values == sorted(values),"invalid_native_wall_timing")
        if last_timing is not None:
            _require(timing[names[0]] == first_timing[names[0]] and timing[names[1]] >= last_timing[names[3]],
                "native_wall_timing_discontinuous")
        else:
            first_timing = timing
        last_timing = timing
        _require(replay.state == "RUNNING","native_receipt_after_terminal")
        _require(replay.reserve_action() == ticket,"native_ticket_mismatch")
        replay.complete_action(ticket,result)
        receipts.append({"path":str(path),"sha256":digest,"native_action_index":ticket,
            "native_result":asdict(result),"timing":deepcopy(timing)})
        previous,final_result = receipt,result

    terminal_path = worker/"terminal.json"; terminal_evidence = None
    count = len(receipts)
    if terminal_path.exists() or terminal_path.is_symlink():
        terminal,terminal_hash = _record(terminal_path)
        _require(load_terminal_or_crash(terminal_path) == terminal and terminal["native_action_limit"] == 1400,
            "invalid_worker_terminal")
        terminal_evidence = {"path":str(terminal_path),"sha256":terminal_hash,"record":terminal}
        if terminal["status"] in ("success","native_failure","horizon_exhausted"):
            _require(replay.state == "TERMINAL","terminal_without_matching_callback_receipts")
            derived = replay.terminal_record()
            _require(all(terminal[k] == derived[k] for k in derived
                if k not in ("recorded_at_utc","completion_time_s")),"worker_terminal_receipt_disagreement")
        else:
            submitted,confirmed,pending = (terminal[k] for k in ("submitted_actions","confirmed_actions","unconfirmed_action"))
            ordinary = confirmed == count and submitted in (count,count+1)
            commit_race = count > 0 and confirmed == count-1 and submitted == pending == count
            _require(ordinary or commit_race,"external_terminal_receipt_count_disagreement")
            if replay.state == "RUNNING":
                if ordinary and submitted == count+1: replay.reserve_action()
                replay.stop(terminal["status"])
    if replay.state != "TERMINAL":
        replay.stop("infrastructure_error")
    final = replay.terminal_record()
    elapsed = (None if last_timing is None else last_timing["native_callback_completed_monotonic_s"]
               - first_timing["policy_begin_monotonic_s"])
    # Set only the new reconciled record; original terminal files are untouched.
    if final["official_success"] is True:
        final["completion_time_s"] = elapsed
    _require(verify_owned_worker_gone(deepcopy(identity)) is True,"owned_worker_reap_proof_changed")
    return {"schema":"robodojo-native-post-reap-reconciliation-1","scope":"evaluator_only",
        "model_input_eligible":False,"manifest_sha256":manifest["manifest_sha256"],"worker":identity,
        "owned_worker_reaped_verified":True,"existing_consumed_binding":binding,
        "ready_record_sha256":ready_hash,"layout":{"path":layout_path,"sha256":layout_hash},
        "registration_evidence_id":registration_id,"episode_evidence_id":binding["claim_nonce"],
        "native_action_receipts":receipts,"verified_native_actions":count,
        "verified_native_substeps":count*10,"verified_policy_simulation_time_s":count*.04,
        "last_native_result":None if final_result is None else asdict(final_result),
        "worker_terminal":terminal_evidence,"worker_terminal_present":terminal_evidence is not None,
        "unrecorded_dispatches_not_excluded":terminal_evidence is None and final["status"] == "infrastructure_error",
        "native_terminal_certified_from_receipts":final["status"] in ("success","native_failure","horizon_exhausted"),
        "reconciled_terminal":final,"parent_terminal_modified":False,
        "policy_timing":{"source":"actual_worker_native_callback_receipts","clock":"time.monotonic",
            "available":last_timing is not None,
            "policy_begin_monotonic_s":None if first_timing is None else first_timing["policy_begin_monotonic_s"],
            "last_native_callback_monotonic_s":None if last_timing is None else last_timing["native_callback_completed_monotonic_s"],
            "elapsed_policy_wall_time_through_last_callback_s":elapsed,
            "completion_time_s":elapsed if final["official_success"] is True else None},
        "additional_runtime_actions":0,"additional_reward_evaluations":0}
