"""Focused contract checks; no model, simulator, historical fixtures or runs."""

from copy import deepcopy
from types import SimpleNamespace

from gpt_field.agent_speedup_v2_user_check_framework import (
    ExecutionFeedback, RecoveryMemory, StateSnapshot, TaskContract, intent_key,
)


def reviewed_plan():
    return {
        "scene": "An object, a support and a switch.",
        "parts": [
            {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1},
            {"id": "B", "text": "object: handle", "move": "free", "w": 2, "x": 2},
            {"id": "C", "text": "support: top", "move": "fixed", "w": 3, "x": 3},
            {"id": "D", "text": "switch: knob", "move": "hinged", "w": 4, "x": 4},
            {"id": "E", "text": "indicator: light", "move": "fixed", "w": 5, "x": 5},
        ],
        "hold_true": [{"rel": "effect", "a": "E", "b": None}],
        "steps": [
            {"do": "place_on", "obj": "A", "grip": ["B"], "ref": "C", "sense": "any",
             "keep": [{"c": "upright", "p": "A"}], "until": []},
            {"do": "turn", "obj": "D", "grip": [], "ref": None, "sense": "any", "keep": [],
             "until": [{"o": "effect", "p": "E", "txt": "indicator illuminated"}]},
        ],
    }


def controller_outcome(*, status="failed", steps=0, evidence=None):
    return SimpleNamespace(
        status=status, relation="supported_on", reason="precondition_unmet", steps=steps,
        pids={"obj": "A", "ref": "C"}, evidence=evidence or {}, thresholds=[], evidence_text="",
    )


def test_replanning_can_reorder_change_grip_and_reword_without_changing_reviewed_intent():
    original = reviewed_plan()
    contract = TaskContract.from_plan(original, "Activate the switch and put the object on the support.")
    replanned = deepcopy(original)
    replanned["steps"].reverse()
    replanned["steps"][1]["grip"] = ["A"]
    replanned["parts"][0]["text"] = "object: visible body surface"
    assert contract.validate_replan(replanned) == []
    # The approved record is detached from both input and returned JSON.
    original["steps"].clear()
    exported = contract.to_dict()
    exported["reviewed_intents"].clear()
    assert len(contract.reviewed_intents) == 2
    assert contract.entity_bindings["A"]["reviewed_description"] == "object: body"


def test_dropped_goal_needs_verified_completion_and_changed_effect_is_not_equivalent():
    plan = reviewed_plan()
    contract = TaskContract.from_plan(plan)
    omitted = deepcopy(plan)
    omitted["steps"] = omitted["steps"][:1]
    issues = contract.validate_replan(omitted)
    assert [(issue["code"], issue["pid"]) for issue in issues] == [("reviewed_intent_missing", "D")]
    assert contract.validate_replan(omitted, completed_intents=[intent_key(plan["steps"][1])]) == []
    changed = deepcopy(plan)
    changed["steps"][1]["until"][0]["txt"] = "indicator dark"
    assert contract.validate_replan(changed)[0]["code"] == "reviewed_intent_missing"


def test_rewording_and_time_do_not_reset_failure_memory_but_observed_state_changes_do():
    records = {"C": {"pid": "C", "text": "support: top", "status": "registered",
                      "registration_step": 12, "xyz": [0.1, 0.2, 0.3], "sigma_h_m": 0.001,
                      "region_xy": [], "region_cells": [], "roles": ["ref"]}}
    before = StateSnapshot.from_records(records, step=20, holding="A")
    relabeled = deepcopy(records)
    relabeled["C"].update(text="support: currently visible top", registration_step=70)
    after = StateSnapshot.from_records(relabeled, step=80, holding="A")
    assert before.fingerprint() == after.fingerprint()
    assert after.to_dict()["entities"]["C"]["observed_at_step"] == 70
    feedback = ExecutionFeedback("blocked", "place_on", "precondition_unmet", False,
                                 unmet_conditions=({"predicate": "empty_hand_for_observation", "part": "C"},),
                                 evidence={"pids": {"obj": "A", "ref": "C"}})
    memory = RecoveryMemory()
    memory.record(feedback, before, context={"plan_revision": 1})
    repeated = memory.record(feedback, after, context={"plan_revision": 2})
    assert repeated["same_state_failures"] == repeated["consecutive_unchanged"] == 2
    released = StateSnapshot.from_records(relabeled, step=81, holding=None)
    changed = memory.record(feedback, released)
    assert changed["same_state_failures"] == changed["consecutive_unchanged"] == 1
    assert changed["recurring_condition_count"] == 3
    assert memory.for_model()["total_outcomes"] == 3


def test_survey_motion_does_not_mean_blocked_manipulation_executed():
    outcome = controller_outcome(steps=12, evidence={"execution_feedback": {
        "status": "blocked", "executed": False,
        "unmet_conditions": [{"predicate": "destination_cells_nonempty", "part": "C", "value": False}],
        "observation": {"status": "succeeded", "geometry_ready": False},
    }})
    feedback = ExecutionFeedback.from_outcome(outcome, policy_step=42)
    assert feedback.status == "blocked"
    assert feedback.executed is False
    assert feedback.evidence["controller_steps"] == 12
    assert feedback.unmet_conditions[0]["predicate"] == "destination_cells_nonempty"
    assert feedback.observed_effects == ()


def test_pending_verification_and_failed_observation_cannot_be_promoted_to_success():
    metadata_pending = controller_outcome(status="succeeded", steps=4, evidence={
        "execution_feedback": {"status": "pending_verification", "executed": True, "geometry_ready": False}})
    unknown_verification = controller_outcome(status="succeeded", steps=4, evidence={
        "verification_status": "unknown", "execution_feedback": {"status": "succeeded", "executed": True}})
    for outcome in (metadata_pending, unknown_verification):
        feedback = ExecutionFeedback.from_outcome(outcome)
        assert feedback.status == "pending_verification"
        assert feedback.observed_effects == ()
    contradictory = controller_outcome(status="failed", evidence={
        "execution_feedback": {"status": "succeeded", "executed": False}})
    assert ExecutionFeedback.from_outcome(contradictory).status == "failed"
    # Passive observation can fail without any new robot action.
    assert ExecutionFeedback.from_outcome(controller_outcome(status="failed")).status == "failed"


def test_keep_constraints_remain_scoped_and_hold_true_claims_survive_replanning():
    plan = reviewed_plan()
    contract = TaskContract.from_plan(plan)
    weakened = deepcopy(plan)
    weakened["steps"][0]["keep"] = []
    assert contract.validate_replan(weakened)[0]["code"] == "reviewed_constraint_missing"
    completed = deepcopy(plan)
    completed["steps"] = completed["steps"][1:]
    # Upright applies to pursuing the reviewed placement, not an unrelated turn.
    completion = [intent_key(plan["steps"][0])]
    assert contract.validate_replan(completed, completed_intents=completion) == []
    completed["hold_true"] = []
    assert contract.validate_replan(completed, completed_intents=completion)[0]["code"] == "reviewed_hold_true_missing"
