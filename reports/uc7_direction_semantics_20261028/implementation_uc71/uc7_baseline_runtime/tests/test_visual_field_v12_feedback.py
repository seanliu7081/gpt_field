"""Feedback aliasing preserves exact evidence and raw observation/history."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gpt_field.visual_feedback_compact_v11 import compact_payload_feedback as compact_v11
from gpt_field.visual_feedback_compact_v12 import compact_payload_feedback
from test_visual_field_v11_feedback import feedback


def record(value):
    return {"role": "user", "content": [{"type": "input_text", "text": json.dumps(value)}]}


def payload():
    evidence = feedback()
    evidence["translation_guard"]["observation_travel_remaining_m"] = .002
    evidence["translation_guard"]["risk_progress"] = {"state": "worsening", "per_neighbor": [{"id": "neighbor", "gap_change_px": -2.}]}
    evidence["robot_translation_response"] = {"state": "supported", "actual_displacement_m": [.002,0,0]}
    robot = [0.,0.,0.,0.,0.,0.,.04,-.04]
    return {"instructions": "unchanged", "input": [
        {"role": "assistant", "content": [{"type": "output_text", "text": '{"raw":"model","reason":"unchanged"}'}]},
        record({"type": "visual_execution_block", "before_state": robot, "after_state": robot,
                "steps": 3, "refresh_reason": "translation_budget_exhausted", "end_visual_feedback": copy.deepcopy(evidence)}),
        record({"type": "observation", "observation_index": 8, "observation/state": robot,
                "observation/joint_position": [1.,2.,3.,4.,5.,6.,7.],
                "observable_context": {"visual_feedback": evidence, "refresh_reason": "translation_budget_exhausted"}})]}


def parsed(value, index):
    return json.loads(value["input"][index]["content"][0]["text"])


def test_only_exact_duplicate_is_referenced_and_full_live_hazards_robot_state_and_raw_outputs_survive():
    raw = payload(); original = copy.deepcopy(raw)
    before = compact_v11(raw)
    packed = compact_payload_feedback(raw)
    assert raw == original and packed["instructions"] == raw["instructions"]
    assert packed["input"][0] == raw["input"][0]
    block, observation = parsed(packed, 1), parsed(packed, 2)
    assert block["end_visual_feedback"]["feedback_ref"]["observation_index"] == 8
    assert block["end_visual_feedback"]["feedback_ref"]["path"] == "observable_context.visual_feedback"
    assert observation == parsed(before, 2)
    for key in ("before_state", "after_state", "steps", "refresh_reason"):
        assert block[key] == parsed(raw, 1)[key]
    actual = observation["observable_context"]["visual_feedback"]
    assert actual["translation_guard"] == feedback_union(raw)["translation_guard"]
    assert actual["robot_translation_response"] == feedback_union(raw)["robot_translation_response"]
    assert actual["visual"]["attachment"]["uncertain_travel_remaining_m"] == .0012
    assert actual["subgoal_ledger"] == feedback_union(raw)["subgoal_ledger"]
    assert compact_payload_feedback(packed) == packed


def feedback_union(value):
    return parsed(value, 2)["observable_context"]["visual_feedback"]


@pytest.mark.parametrize("change", ["budget", "warning", "identity", "number_type", "missing_following", "no_observation_index"])
def test_even_small_evidence_difference_or_missing_reference_keeps_full_block_feedback(change):
    raw = payload()
    observation = parsed(raw, 2)
    current = observation["observable_context"]["visual_feedback"]
    if change == "budget": current["translation_guard"]["observation_travel_remaining_m"] -= 1e-12
    elif change == "warning": current["translation_guard"]["risk_progress"]["state"] = "inconclusive"
    elif change == "identity": current["translation_guard"]["risk_progress"]["per_neighbor"][0]["id"] = "other"
    elif change == "number_type": current["translation_guard"]["accepted"] = 0
    elif change == "no_observation_index": observation.pop("observation_index")
    raw["input"][2] = record(observation)
    if change == "missing_following": raw["input"].pop()
    packed = compact_payload_feedback(raw)
    assert "feedback_ref" not in parsed(packed, 1)["end_visual_feedback"]


def test_archived_v11_request_duplicate_reduction_preserves_original_and_exact_observation():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "runs/visual_field_v11_zero_shot_001/trials/bowl_original_zero_hybrid_v11/api").glob("*.request.json"))
    if not paths: pytest.skip("Immutable V11 request archive unavailable")
    path = paths[-1]; digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = json.loads(path.read_text())["payload"]; original = copy.deepcopy(raw)
    packed = compact_payload_feedback(raw)
    assert raw == original and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert packed["text"] == raw["text"] and packed["instructions"] == raw["instructions"]
    before = compact_v11(raw)
    full = {}
    for old, new in zip(before["input"], packed["input"]):
        if old.get("role") != "user": assert old == new
        for a, b in zip(old.get("content", []), new.get("content", [])):
            if a.get("type") != "input_text": assert a == b; continue
            try: source, current = json.loads(a["text"]), json.loads(b["text"])
            except ValueError: assert a == b; continue
            if source.get("type") == "observation":
                assert source == current
                full[current["observation_index"]] = current["observable_context"]["visual_feedback"]
    references = 0
    for message in packed["input"]:
        for part in message.get("content", []):
            if part.get("type") != "input_text": continue
            try: row = json.loads(part["text"])
            except ValueError: continue
            ref = row.get("end_visual_feedback", {}).get("feedback_ref")
            if ref:
                references += 1
                assert ref["observation_index"] in full
    assert references >= 1
    assert len(json.dumps(packed)) < len(json.dumps(raw))
