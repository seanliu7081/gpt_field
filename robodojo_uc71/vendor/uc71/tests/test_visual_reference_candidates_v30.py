"""Typed repair uses paid RGB evidence and does not issue control permission."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v21 import RoleAwareVisualMonitorV21
from gpt_field.visual_reference_candidates_v30 import (
    CausalReferenceCandidatesV30, ReferenceReviewEvent, ReferenceReviewKind as Kind)
from gpt_field.visual_reference_review_v30 import (
    TypedReferenceReviewPolicyMixinV30, reference_review_event)
from test_visual_reference_candidates_v24 import setup, advance


def plant():
    _, monitor, context, obs = setup(moving=True)
    bank = CausalReferenceCandidatesV30()
    bank.configure(obs, monitor, context)
    return bank, monitor, context, obs


def lost_event(reason="never_seen_before_control_reason"):
    return ReferenceReviewEvent(Kind.CONTROLLED_REFERENCE_LOST, detail_reason=reason)


def test_dynamic_support_loss_extracts_paid_history_without_static_trigger_or_debt_change():
    bank, monitor, _, obs = plant()
    obs = advance(bank, monitor, obs)
    before = (bank.path_m, bank.sequence, copy.deepcopy(monitor._active_reference))
    assert bank.summary()["support_max_changed_fraction"] > .05
    assert bank.summary()["candidate_extraction_count"] == 0
    monitor.tracks["moving"].reliable = False
    event = reference_review_event(monitor, detail_reason="new_stop_reason_v999")
    assert event.kind is Kind.CONTROLLED_REFERENCE_LOST
    report = bank.review(obs, monitor, event)
    assert report["extraction_status"] == "available"
    assert report["candidate_extraction_count"] == 1 and report["candidates"]
    assert report["review_required"] and report["review_event"]["detail_reason"] == "new_stop_reason_v999"
    assert not monitor.tracks["moving"].reliable
    assert (bank.path_m, bank.sequence, monitor._active_reference) == before
    assert not report["automatic_reference_promotion"] and not report["control_debt_modified"]


def test_repeated_loss_handoffs_and_holds_keep_one_ticket_and_one_extraction():
    bank, monitor, _, obs = plant()
    obs = advance(bank, monitor, obs)
    monitor.tracks["moving"].reliable = False
    first = bank.review(obs, monitor, lost_event())
    for i in range(4):
        bank.observe(obs, obs, monitor)
        report = bank.review(obs, monitor, lost_event(f"differently_worded_{i}"))
        assert report["review_id"] == first["review_id"]
        assert report["candidate_extraction_count"] == first["candidate_extraction_count"] == 1
        assert report["measured_path_m"] == first["measured_path_m"]


def test_target_anchor_loss_is_explicitly_ineligible_not_an_empty_search():
    bank, monitor, _, obs = plant()
    obs = advance(bank, monitor, obs)
    monitor.tracks["target"].reliable = False
    event = reference_review_event(monitor, handoff_present=True)
    assert event.kind is Kind.TARGET_REFERENCE_LOST
    report = bank.review(obs, monitor, event)
    assert report["extraction_status"] == "ineligible"
    assert report["eligibility_reason"] == "stationary_anchor_review_required"
    assert report["candidate_extraction_count"] == 0 and not report["candidates"]
    assert not report["review_required"] and bank.image(obs) is None


@pytest.mark.parametrize("fault,expected", [
    ("stale_observation", "observation_binding_mismatch"),
    ("stale_monitor", "monitor_state_binding_mismatch"),
    ("different_track", "controlled_correspondence_binding_changed"),
    ("role", "controlled_role_is_not_tool"),
    ("orientation", "orientation_domain_changed"),
    ("fingers", "finger_domain_changed"),
    ("short_history", "insufficient_acknowledged_history"),
    ("missing_historical_anchor", "stationary_anchor_missing_in_retained_history"),
    ("changed_anchor_raster", "stationary_anchor_observation_binding_mismatch"),
])
def test_typed_request_does_not_bypass_domain_history_or_current_binding(fault, expected):
    bank, monitor, _, obs = plant()
    old = copy.deepcopy(obs)
    obs = advance(bank, monitor, obs, steps=2 if fault == "short_history" else 4)
    if fault == "stale_observation": obs = old
    elif fault == "stale_monitor": monitor.state = monitor.state + np.r_[.001, np.zeros(7)]
    elif fault == "different_track": monitor.tracks["moving"] = copy.deepcopy(monitor.tracks["moving"])
    elif fault == "role": bank._moving_kind = "payload"
    elif fault in ("orientation", "fingers"):
        index = 3 if fault == "orientation" else 6
        state = list(obs["observation/state"]); state[index] += .08
        obs["observation/state"] = state
        bank.current.state = np.array(state); monitor.state = np.array(state)
    elif fault == "missing_historical_anchor": bank._frames[0].anchor_reliable = False
    elif fault == "changed_anchor_raster": monitor.tracks["target"].previous_image = np.zeros_like(monitor.tracks["target"].previous_image)
    report = bank.review(obs, monitor, lost_event())
    assert report["extraction_status"] == "ineligible"
    assert report["eligibility_reason"] == expected
    assert report["candidate_extraction_count"] == 0
    assert not report["review_required"]


def test_unchanged_rasters_distinguish_extracted_none_from_not_requested():
    bank, monitor, _, obs = plant()
    assert bank.summary()["extraction_status"] == "not_requested"
    # Under4mm, no inherited early-static extraction; enough paid handoff data.
    obs = advance(bank, monitor, obs, static=True)
    report = bank.review(obs, monitor, ReferenceReviewEvent(Kind.CONTROL_RISK_HANDOFF))
    assert report["extraction_status"] == "extracted_none"
    assert report["candidate_extraction_count"] == 1 and not report["candidates"]
    again = bank.review(obs, monitor, ReferenceReviewEvent(Kind.CONTROL_RISK_HANDOFF))
    assert again["extraction_status"] == "extracted_none"
    assert again["candidate_extraction_count"] == 1


def test_stop_text_is_never_a_bank_dispatch_key_and_normal_handoff_does_not_extract():
    bank, monitor, _, obs = plant()
    obs = advance(bank, monitor, obs)
    with pytest.raises(TypeError): bank.handoff(obs, monitor, "external_reference_unavailable")
    with pytest.raises(TypeError): ReferenceReviewEvent("controlled_reference_lost")
    monitor.summary = lambda: {"reference_health": {"repair_required": False}}
    assert reference_review_event(monitor, handoff_present=True, detail_reason="measured_local_goal_arrival") is None
    assert bank.summary()["candidate_extraction_count"] == 0
    event = reference_review_event(monitor, handoff_present=True,
                                  failed_control_handoff=True, detail_reason="new_geometry_admission_failure")
    assert event.kind is Kind.CONTROL_RISK_HANDOFF
    monitor.summary = lambda: {"reference_health": {"repair_required": True}}
    assert reference_review_event(monitor, handoff_present=True, detail_reason="normal_arrival") is None
    event = reference_review_event(monitor, handoff_present=True, failed_control_handoff=True,
                                   detail_reason="new_evidence_reason")
    assert event.kind is Kind.CONTROLLED_RESPONSE_UNRELIABLE


def test_candidate_resolution_remains_observation_bound_and_retrospective():
    bank, monitor, context, obs = plant()
    old = copy.deepcopy(obs); obs = advance(bank, monitor, obs)
    report = bank.review(obs, monitor, lost_event())
    candidate = report["candidates"][0]
    with pytest.raises(ValueError): bank.resolve_candidate(candidate["candidate_id"], old)
    resolved = bank.resolve_candidate(candidate["candidate_id"], obs)
    assert not resolved["evidence"]["validation_is_prospective"]
    assert resolved["evidence"]["fresh_held_out_confirmation_required"]
    from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21
    pixels = _gray(np.array(decode_image(obs["observation/image"])))
    monitor.tracks["moving"] = PartFeatureTrackV21(pixels, resolved["current_point_uv"],
                                                   "chosen", roi_uv=resolved["current_roi_uv"])
    path = bank.path_m
    bank.configure(obs, monitor, context)
    assert bank.summary()["extraction_status"] == "not_requested"
    assert not bank.summary()["candidates"] and bank.path_m == path
    assert not bank.raw_support_evidence(monitor.state, monitor)["causal_supported"]


def test_policy_mixin_uses_actual_failure_occurrence_not_control_reason_membership():
    class Base:
        def reset(self): self.failed_hypotheses = []
        def _failure(self, reason): self.failed_hypotheses.append({"reason": reason})
    class Policy(TypedReferenceReviewPolicyMixinV30, Base): pass
    policy = Policy(); policy.reset()
    bank, monitor, _, obs = plant(); obs = advance(bank, monitor, obs)
    monitor.summary = lambda: {"reference_health": {"repair_required": False}}
    policy.reference_candidates = bank; policy.monitor = monitor
    policy._requested_reference_reviews = set(); policy.events = []
    policy.refresh_reason = "new_geometry_failure"; policy.step_index = 4
    policy._review_paid_reference(obs, policy.refresh_reason)
    assert bank.summary()["extraction_status"] == "not_requested"
    policy._failure("any_future_failure_wording")
    policy._review_paid_reference(obs, policy.refresh_reason)
    assert bank.summary()["review_event"]["kind"] == "control_risk_handoff"
    assert bank.summary()["extraction_status"] == "available"
    assert len(policy.events) == 1 and len(policy.failed_hypotheses) == 1
    policy._review_paid_reference(obs, policy.refresh_reason)
    assert len(policy.events) == 1
    policy.reset(); assert not policy._typed_reference_control_failure


ARCHIVE = Path(__file__).resolve().parents[1]/"runs/visual_field_v29_zero_shot_001/trials/bowl_original_zero_hybrid_v29/trace.jsonl"


def replay_recorded_handoffs():
    """Passive actual-RGB replay; no policy action selection or simulator runs."""
    rows = [json.loads(line) for line in ARCHIVE.read_text().splitlines()]
    actions = {r["step"]: r for r in rows if r["event"] == "action"}
    afters = {r["step"]: PolicyObservation.from_dict(r["observation"])
              for r in rows if r["event"] == "executed"}
    feedback = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    obs = PolicyObservation.from_dict(next(r["observation"] for r in rows if r["event"] == "reset"))
    bank, monitor = CausalReferenceCandidatesV30(), RoleAwareVisualMonitorV21()
    def images(o): return {"external": np.array(decode_image(o.image)), "wrist": np.array(decode_image(o.wrist_image))}
    records = []
    for step in range(61):
        prediction = actions[step]["diagnostics"].get("prediction")
        if prediction:
            monitor.configure(images(obs), prediction["visual_references"], obs.state, intent_id=step)
            bank.configure(obs, monitor, prediction["motion_context"])
        after = afters[step]
        monitor.observe(images(after), after.state, actions[step]["action"])
        bank.observe(obs, after, monitor)
        if step in (39, 41, 60):
            actual = feedback[step]
            moving = monitor.summary()["tracking"]["moving"]
            assert moving["reliable"] == actual["visual"]["tracking"]["moving"]["reliable"]
            assert moving["failure_reason"] == actual["visual"]["tracking"]["moving"]["failure_reason"]
            reason = actual["last_field_handoff"]["reason"]
            event = reference_review_event(monitor, handoff_present=True, detail_reason=reason)
            before = (bank.path_m, bank.sequence, copy.deepcopy(monitor.summary()))
            started = time.perf_counter()
            report = bank.review(after, monitor, event)
            review_s = time.perf_counter() - started
            assert (bank.path_m, bank.sequence, monitor.summary()) == before
            records.append({"action_step": step, "next_request_step": step+1,
                            "recorded_handoff_reason": reason, "moving_failure": moving["failure_reason"],
                            "legacy_extraction_count": actual["reference_repair"]["candidate_extraction_count"],
                            "review_runtime_s": review_s, "review": report})
        obs = after
    return bank, monitor, obs, records


def test_actual_v29_all_three_lost_track_handoffs_run_current_paid_history_extraction():
    if not ARCHIVE.exists(): pytest.skip("Optional recorded V29 RGB archive unavailable")
    bank, monitor, obs, records = replay_recorded_handoffs()
    assert [r["legacy_extraction_count"] for r in records] == [0, 0, 0]
    assert [r["review"]["candidate_extraction_count"] for r in records] == [1, 2, 3]
    for row in records:
        report = row["review"]
        assert report["review_event"]["kind"] == "controlled_reference_lost"
        assert report["extraction_status"] in ("available", "extracted_none")
        assert report["eligibility_reason"] is None
        assert report["retained_history_frames"] == 31
        assert not report["control_debt_modified"] and not report["automatic_reference_promotion"]
    assert any(r["review"]["candidates"] for r in records)
    assert bank._matches(obs) and bank.sequence == 61
