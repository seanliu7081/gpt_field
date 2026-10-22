"""Deterministic interaction memory uses only prior accepted observable history."""
import copy
import json

import pytest

from gpt_field.attempt_memory import (
    ATTEMPT_MEMORY_VERSION, MAX_MEMORY_CHARS, attempt_memory_instructions,
    attempt_review_schema, build_attempt_memory, validate_attempt_review,
)

STATE = [.1, .2, 1., 0., 0., 0., .04, -.04]


def message(value, role="user", **updates):
    return {"role": role, "content": [{"type": "input_text" if role == "user" else "output_text",
                                      "text": json.dumps(value)}], **updates}


def observed(index):
    result = message({"type": "observation", "observation_index": index, "observation/state": STATE})
    result["content"].append({"type": "input_image", "image_url": "never-decode-this"})
    return result


def review(attempt=None, outcome="uncertain", **changes):
    value = {"previous_attempt": None if attempt is None else {
        "attempt_id": attempt, "outcome": outcome, "evidence": "Object occluded; fingers stable alone are insufficient."},
        "next_change": "Change approach orientation using current views.", "prerequisites": []}
    value.update(changes)
    return value


def intent(command=None, phase="grasp", target="object", assessment=None):
    result = {"controller": "contact_direct", "target_region": target,
              "control_context": {"phase": phase, "visual_reference": None},
              "gripper_operation": None if command is None else {"command": command}}
    if assessment is not None:
        result["interaction_review"] = assessment
    return message(result, "assistant")


def feedback(step, command=1., dz=0.):
    before = STATE.copy()
    after = STATE.copy()
    after[2] += dz
    after[6:] = [.02, -.02]
    return message({"type": "execution_feedback", "step": step,
                    "before_state": before, "after_state": after, "gripper_command": command})


def close_turn(index, first, count, assessment=None):
    return [observed(index), intent("close", assessment=assessment),
            *[feedback(i) for i in range(first, first + count)],
            message({"type": "local_gripper_result", "step": first + count, "command": "close",
                     "actions_executed": count, "stop_reason": "budget", "fingers_stable": False,
                     "finger_motion_observed": True, "pose_error_m": .0002,
                     "grasp_success_certified": False, "task_success_certified": False})]


def test_schema_matches_validator_and_contract_is_presentation_only():
    value = review("attempt-0001")
    assert validate_attempt_review(value) == value
    assert validate_attempt_review(value) is not value
    assert set(attempt_review_schema()["required"]) == set(value)
    assert ATTEMPT_MEMORY_VERSION in attempt_memory_instructions()
    assert "no mandatory probe" in attempt_memory_instructions()
    assert "not object attachment" in attempt_memory_instructions()


@pytest.mark.parametrize("change", [
    {"unexpected": 1}, {"next_change": ""}, {"next_change": " "},
    {"next_change": "a" * 241}, {"prerequisites": None},
    {"previous_attempt": {"attempt_id": "future", "outcome": "success", "evidence": "No proof"}},
    {"previous_attempt": {"attempt_id": "x", "outcome": "uncertain", "evidence": "x", "verified": True}},
    {"prerequisites": [{"key": "door", "state": "supported", "evidence": "Seen"}] * 4},
    {"prerequisites": [{"key": "door", "state": "supported", "evidence": "Seen"}] * 2},
])
def test_review_rejects_unbounded_or_unexpected_claims(change):
    value = review()
    value.update(change)
    with pytest.raises(ValueError):
        validate_attempt_review(value)


def test_105_actions_are_one_attempt_with_four_legitimate_close_continuations():
    transcript, first = [], 0
    for index, count in enumerate([30, 30, 30, 15]):
        transcript += close_turn(index, first, count)
        first += count
    before = copy.deepcopy(transcript)
    memory = build_attempt_memory(transcript, 4, STATE)
    assert transcript == before
    assert memory["attempts_seen"] == 1
    record = memory["attempts"][0]
    assert record["close_operation_requests"] == 4
    assert record["continuation_close_requests"] == 3
    assert record["close_operation_actions"] == record["closed_command_actions"] == 105
    assert record["executed_actions"] == 105
    assert record["last_feedback_step"] == 104
    assert record["latest_gripper_result"]["fingers_stable"] is False
    assert record["latest_outcome_claim"] is None
    assert "grasp_success_certified" not in json.dumps(memory)
    assert record["first_close_state8"] == STATE
    assert record["latest_close_state8"][-2:] == [.02, -.02]


def test_full_open_feedback_is_retained_then_reclose_starts_new_attempt():
    transcript = close_turn(0, 0, 15)
    transcript += [observed(1), intent("open"), *[feedback(i, -1.) for i in range(15, 35)],
                   message({"type": "local_gripper_result", "step": 35, "command": "open",
                            "actions_executed": 20, "stop_reason": "settled", "fingers_stable": True})]
    transcript += close_turn(2, 35, 15)
    memory = build_attempt_memory(transcript, 3)
    assert memory["attempts_seen"] == 2
    old, current = memory["attempts"]
    assert old["end_reason"] == "executed_open_command"
    assert old["executed_actions"] == 35 and old["last_feedback_step"] == 34
    assert old["latest_gripper_result"]["command"] == "open"
    assert memory["active_attempt_id"] == current["attempt_id"] == "attempt-0002"
    assert current["close_operation_actions"] == 15
    assert old["latest_outcome_claim"] is None  # Opening is not inferred success/failure.


def test_destination_transport_and_alignment_keep_held_object_attempt():
    transcript = close_turn(0, 0, 15)
    transcript += [observed(1), intent(phase="verify"), feedback(15, dz=.012),
                   observed(2), intent(phase="transport", target="destination"), feedback(16, dz=.1),
                   observed(3), intent(phase="align", target="destination"), feedback(17, dz=-.1),
                   observed(4), intent(phase="approach", target="destination"), feedback(18)]
    memory = build_attempt_memory(transcript, 5)
    assert memory["attempts_seen"] == 1
    record = memory["attempts"][0]
    assert record["target_region_model_label"] == "object"
    assert record["post_close_measured_translation_m"] == [0., 0., .012]
    assert record["latest_verification_motion"]["measured_translation_m"] == [0., 0., .012]
    assert record["latest_verification_motion"]["source"] == "measured_robot_feedback"
    assert record["latest_verification_motion"]["observation_index"] == 1
    assert record["latest_outcome_claim"] is None


def test_reported_loss_is_dated_claim_and_changes_do_not_certify_object_motion():
    transcript = close_turn(0, 0, 15)
    transcript += [observed(1), intent(phase="transport", target="destination"), feedback(15, dz=.02),
                   observed(2), intent(phase="recover", assessment=review("attempt-0001", "lost_engagement")),
                   feedback(16), observed(3), intent("close")]
    memory = build_attempt_memory(transcript, 4)
    assert memory["attempts_seen"] == 2
    old = memory["attempts"][0]
    claim = old["latest_outcome_claim"]
    assert claim["outcome"] == "lost_engagement"
    assert claim["source"] == "model" and claim["verified"] is False
    assert claim["observation_index"] == 2 and claim["age_observations"] == 2
    assert old["end_reason"] == "model_reported_retry"
    assert memory["attempts"][1]["latest_outcome_claim"] is None


def test_unknown_outcome_and_invented_reference_never_become_measured_fact():
    transcript = close_turn(0, 0, 15)
    transcript += [observed(1), intent(assessment=review("attempt-0001")), feedback(15),
                   observed(2), intent(assessment=review("attempt-9999", "possibly_engaged")), feedback(16)]
    memory = build_attempt_memory(transcript, 3)
    assert memory["attempts"][0]["latest_outcome_claim"]["outcome"] == "uncertain"
    assert memory["unmatched_attempt_claim"]["reference_status"] == "unmatched_not_applied"
    assert memory["unmatched_attempt_claim"]["verified"] is False


def test_prerequisites_update_by_key_and_preserve_observation_age():
    first = {"key": "destination_accessible", "state": "supported", "evidence": "Appears open in current RGB."}
    changed = {**first, "state": "contradicted", "evidence": "Appears retracted in current RGB."}
    transcript = [observed(0), intent(assessment=review(prerequisites=[first])),
                  observed(1), intent(assessment=review(prerequisites=[changed])),
                  observed(2), intent(assessment=review())]
    memory = build_attempt_memory(transcript, 3)
    assert len(memory["prerequisites"]) == 1
    claim = memory["prerequisites"][0]
    assert claim["state"] == "contradicted" and claim["observation_index"] == 1
    assert claim["age_observations"] == 2 and claim["source"] == "model"


def test_current_future_rejected_or_incomplete_responses_do_not_enter_memory():
    transcript = close_turn(0, 0, 15)
    rejected = intent("close", assessment=review("attempt-0001", "possibly_engaged"))
    rejected["accepted"] = False
    incomplete = intent("close", assessment=review("attempt-0001", "not_engaged"))
    incomplete["status"] = "incomplete"
    transcript += [observed(1), rejected, feedback(15), observed(2), incomplete, feedback(16)]
    prior = build_attempt_memory(transcript, 3, STATE)
    transcript += [observed(3), intent("close", assessment=review("attempt-0001", "lost_engagement")), feedback(17),
                   observed(4), intent("close"), feedback(18)]
    memory = build_attempt_memory(transcript, 3, STATE)
    assert memory == prior
    assert memory["attempts"][0]["executed_actions"] == 15
    assert memory["attempts"][0]["latest_outcome_claim"] is None


def test_feedback_summary_aggregates_exact_counts_without_reading_hidden_fields():
    transcript = [observed(0), intent("close"), message({
        "type": "execution_block_summary", "first_step": 0, "last_step": 19, "executed_actions": 20,
        "gripper_close_steps": 20, "before_state": STATE, "after_state": STATE,
        "oracle_object_pose": "hidden secret", "success": True})]
    memory = build_attempt_memory(transcript, 1)
    assert memory["attempts"][0]["close_operation_actions"] == 20
    assert memory["attempts"][0]["latest_outcome_claim"] is None
    assert "hidden secret" not in json.dumps(memory)
    malformed = copy.deepcopy(transcript)
    value = json.loads(malformed[-1]["content"][0]["text"])
    value["executed_actions"] = 99
    malformed[-1] = message(value)
    assert build_attempt_memory(malformed, 1)["attempts"][0]["executed_actions"] == 0


def test_long_history_memory_stays_bounded_and_never_drops_active_attempt():
    transcript, step = [], 0
    for index in range(30):
        assessment = review(next_change="界" * 240, prerequisites=[
            {"key": f"state-{index}-{j}", "state": "uncertain", "evidence": "界" * 160} for j in range(3)])
        transcript += close_turn(index * 2, step, 15, assessment)
        step += 15
        if index < 29:
            transcript += [observed(index * 2 + 1), intent("open"), feedback(step, -1.)]
            step += 1
    memory = build_attempt_memory(transcript, 60, STATE)
    assert len(memory["attempts"]) <= 3
    assert memory["attempts_seen"] == 30
    assert memory["active_attempt_id"] == "attempt-0030"
    assert memory["attempts"][-1]["attempt_id"] == memory["active_attempt_id"]
    assert len(json.dumps(memory, separators=(",", ":"), ensure_ascii=True)) <= MAX_MEMORY_CHARS
    assert memory == build_attempt_memory(copy.deepcopy(transcript), 60, STATE)


def test_ordinary_direct_regrasp_after_reported_failure_is_new_attempt_without_local_expansion():
    transcript = close_turn(0, 0, 15)
    transcript += [observed(1), intent(phase="verify"), feedback(15, dz=.01), observed(2)]
    value = json.loads(intent(phase="grasp", assessment=review("attempt-0001", "not_engaged"))["content"][0]["text"])
    value["direct"] = {"actions": [[0., 0., 0., 0., 0., 0., 1.]] * 5}
    transcript += [message(value, "assistant"), *[feedback(i) for i in range(16, 21)]]
    memory = build_attempt_memory(transcript, 3)
    assert memory["attempts_seen"] == 2
    new = memory["attempts"][-1]
    assert new["executed_actions"] == new["closed_command_actions"] == 5
    assert new["close_operation_requests"] == new["close_operation_actions"] == 0


def test_gripper_result_is_allowlisted_bounded_and_requires_matching_feedback_boundary():
    transcript = close_turn(0, 0, 15)
    value = json.loads(transcript[-1]["content"][0]["text"])
    value.update(stop_reason="r" * 10000, hidden_pose="not allowed", fingers_stable=["not bool"] * 1000)
    transcript[-1] = message(value)
    memory = build_attempt_memory(transcript, 1)
    result = memory["attempts"][0]["latest_gripper_result"]
    assert len(result["stop_reason"]) == 80
    assert "fingers_stable" not in result
    assert "not allowed" not in json.dumps(memory)
    value["step"] = 999
    transcript[-1] = message(value)
    assert build_attempt_memory(transcript, 1)["attempts"][0]["latest_gripper_result"] is None


@pytest.mark.parametrize("commands,preclosed,ended", [
    ([1., -1., 1.], True, True), ([-1., 1.], False, False),
    ([1., -1., 1.], False, True), ([-1., 1., -1.], False, True),
])
def test_mixed_direct_raw_and_summary_have_same_boundaries_without_invented_internal_pose(commands, preclosed, ended):
    prefix = close_turn(0, 0, 15) if preclosed else []
    index, step = (1, 15) if preclosed else (0, 0)
    prefix += [observed(index), intent()]
    raw = [feedback(step + j, c) for j, c in enumerate(commands)]
    summary = message({"type": "execution_block_summary", "first_step": step,
        "last_step": step + len(commands) - 1, "executed_actions": len(commands),
        "gripper_close_steps": sum(c > 0 for c in commands), "before_state": STATE,
        "after_state": STATE[:6] + [.02, -.02],
        "gripper_command_runs": [{"command": c, "actions": 1} for c in commands]})
    left, right = [build_attempt_memory(prefix + suffix, index + 1) for suffix in (raw, [summary])]
    assert left["active_attempt_id"] == right["active_attempt_id"]
    for value in (left, right):
        record = value["attempts"][0]
        assert (record["status"] == "ended") is ended
        assert record["mixed_gripper_block_seen"] is True
        assert record["executed_actions"] == step + len(commands)
        assert record["closed_command_actions"] == step + sum(c > 0 for c in commands)
        if not preclosed:
            assert record["first_close_state8"] is None
            assert record["post_close_measured_translation_m"] is None
            assert record["post_close_displacement_ambiguous"] is True
