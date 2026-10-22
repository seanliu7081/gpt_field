"""Causal operation feedback and wire validation, without model/simulator calls."""
import copy
import json

import pytest

from gpt_field.operation_contract import (
    MAX_LEDGER_CHARS, OPERATION_VERSION, OperationEvidenceLedger, legacy_context,
    operation_instructions, operation_schema, validate_operation,
    validate_visual_intent, visual_intent_schema,
)


def operation(kind="move", **changes):
    return {"id": "op-1", "kind": kind, "target_id": "black_bowl",
        "intent": "Approach the visible grasp region with the hand open.",
        "expected_result": "The hand approaches without moving the target.",
        "local_action_budget": 30, "allow_local_continuation": False,
        "prerequisites": [{"key": "destination_accessible", "state": "uncertain",
            "evidence": "The opening is partly occluded in the current views.",
            "source": "current_rgb_and_robot_state"}],
        "verify_motion": None, **changes}


def visual(kind="move", **changes):
    return {"mode": {"pregrasp": "pregrasp", "place_align": "held_placement"}.get(kind, "none"),
        "pregrasp_reference": None, "placement_reference": None, "tracking_reference": None,
        "unavailable_reasons": {"external": "Target contact is occluded.",
                                "wrist": "Insufficient visible material texture."}, **changes}


def tracking_reference():
    return {"reference_id": "regions-1", "wrist": None,
        "external": {"object_roi_uv": [.15, .2, .3, .4], "tool_roi_uv": [.4, .5, .6, .7],
                     "target_roi_uv": None}}


def pregrasp_reference():
    return {"reference_id": "contact-1", "target_id": "black_bowl", "wrist": None,
        "external": {"tool_contact_uv": [.2, .3], "target_contact_uv": [.5, .3],
            "tool_roi_uv": [.15, .25, .25, .35], "target_roi_uv": [.45, .25, .55, .35],
            "desired_offset_uv": [-.02, .01], "tool_reference_mode": "tracked_material"}}


def state(x=0., finger=.03):
    return [x, 0., .2, 0., 0., 0., finger, -finger]


def test_schema_objects_are_strict_and_all_properties_required():
    def check(schema):
        if schema.get("type") == "object":
            assert schema["additionalProperties"] is False
            assert set(schema["properties"]) == set(schema["required"])
        for value in schema.values():
            if isinstance(value, dict):
                check(value)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        check(child)
    check(operation_schema())
    check(visual_intent_schema())


@pytest.mark.parametrize("change", [
    {"kind": "align"}, {"local_action_budget": True}, {"local_action_budget": 0},
    {"local_action_budget": 61}, {"local_action_budget": 31},
    {"allow_local_continuation": 1}, {"target_id": " "}, {"expected_result": ""},
    {"verify_motion": {"offset_xyz_m": [0., 0., .01], "keep_orientation": True}},
])
def test_operation_rejects_invalid_explicit_intent(change):
    with pytest.raises(ValueError):
        validate_operation(operation(**change))


def test_continuation_is_explicit_and_does_not_mutate_input():
    raw = operation(local_action_budget=60, allow_local_continuation=True)
    accepted = validate_operation(raw)
    accepted["prerequisites"][0]["state"] = "supported"
    assert raw["prerequisites"][0]["state"] == "uncertain"


def test_verify_motion_checks_norm_not_only_per_axis_and_never_infers_direction():
    good = operation("grasp_close", local_action_budget=60, allow_local_continuation=True,
        verify_motion={"offset_xyz_m": [0., -.02, 0.], "keep_orientation": True})
    assert validate_operation(good)["verify_motion"]["offset_xyz_m"] == [0., -.02, 0.]
    for offset in ([.03, .03, 0.], [0., 0., 0.], [False, 0., .01], [0., float("nan"), .01]):
        bad = copy.deepcopy(good)
        bad["verify_motion"]["offset_xyz_m"] = offset
        with pytest.raises(ValueError):
            validate_operation(bad)
    bad = copy.deepcopy(good)
    bad["verify_motion"]["keep_orientation"] = False
    with pytest.raises(ValueError):
        validate_operation(bad)


def test_prerequisite_source_and_unknown_are_not_certification():
    raw = operation()
    assert validate_operation(raw)["prerequisites"][0]["state"] == "uncertain"
    raw["prerequisites"][0]["source"] = "simulator_truth"
    with pytest.raises(ValueError):
        validate_operation(raw)
    raw = operation()
    raw["prerequisites"] *= 2
    with pytest.raises(ValueError):
        validate_operation(raw)


def test_pregrasp_applicability_retained_when_all_references_unavailable():
    checked = validate_visual_intent(visual("pregrasp"), operation("pregrasp"))
    assert checked["mode"] == "pregrasp"
    assert checked["pregrasp_reference"] is None
    with pytest.raises(ValueError):
        validate_visual_intent(visual(), operation("pregrasp"))
    with pytest.raises(ValueError):
        validate_visual_intent(visual("pregrasp"), operation())


def test_unavailable_views_need_reasons_without_fabricating_reference():
    raw = visual("pregrasp")
    raw["unavailable_reasons"]["external"] = None
    with pytest.raises(ValueError):
        validate_visual_intent(raw, operation("pregrasp"))
    raw["unavailable_reasons"]["external"] = "Both fingertips are occluded."
    assert validate_visual_intent(raw, operation("pregrasp"))["pregrasp_reference"] is None


def test_generic_body_tracking_valid_without_exact_contact_points():
    raw = visual(tracking_reference=tracking_reference())
    raw["unavailable_reasons"]["external"] = None
    checked = validate_visual_intent(raw, operation())
    assert checked["tracking_reference"] == tracking_reference()
    assert checked["pregrasp_reference"] is None
    raw["unavailable_reasons"]["external"] = "Contradictory unavailable claim."
    with pytest.raises(ValueError):
        validate_visual_intent(raw, operation())


def test_pregrasp_reference_target_and_explicit_tool_mode_are_checked():
    raw = visual("pregrasp", pregrasp_reference=pregrasp_reference())
    raw["unavailable_reasons"]["external"] = None
    assert validate_visual_intent(raw, operation("pregrasp"))["pregrasp_reference"] is not None
    raw["pregrasp_reference"]["target_id"] = "different_object"
    with pytest.raises(ValueError):
        validate_visual_intent(raw, operation("pregrasp"))
    raw["pregrasp_reference"] = pregrasp_reference()
    del raw["pregrasp_reference"]["external"]["tool_reference_mode"]
    with pytest.raises(ValueError):
        validate_visual_intent(raw, operation("pregrasp"))


@pytest.mark.parametrize("kind,expected", [
    ("move", "approach"), ("pregrasp", "align"), ("place_align", "align"),
    ("transport", "transport"), ("grasp_close", "grasp"), ("verify_grasp", "verify"),
    ("articulate", "drawer_open"), ("release", "release"), ("recover", "recover"),
])
def test_old_phase_mapping_is_execution_only(kind, expected):
    assert legacy_context(operation(kind), visual(kind), "drawer_bowl")["phase"] == expected
    assert "phase" not in operation_schema()["properties"]


def test_standalone_prompt_keeps_task_geometry_and_does_not_append_old_phase_logic():
    text = operation_instructions("drawer_bowl", robot_geometry_guidance=True,
        target_description="A visible patterned bowl.")
    assert OPERATION_VERSION in text
    assert "open the top drawer and put the bowl inside" in text
    assert "A visible patterned bowl." in text
    assert "Do not add another 0.097 m" in text
    assert "phase=align" not in text
    assert "control_context contains" not in text
    assert "kind=pregrasp" in text and "Direct" in text
    assert "never a simulator pose" in text
    assert "No direction is inferred" in text


def test_ledger_claims_become_stale_after_actions_without_becoming_verified():
    ledger = OperationEvidenceLedger()
    ledger.accept(operation(), visual(), 0, state())
    initial = ledger.presentation(0)
    assert initial["prerequisites"][0]["stale"] is False
    ledger.record_feedback(state(), state(.01, .025), 1)
    value = ledger.presentation(1, state(.01, .025))
    claim = value["prerequisites"][0]
    assert claim["stale"] is True
    assert claim["age_observations"] == 1
    assert claim["executed_actions_since_claim"] == 1
    assert claim["independently_verified"] is False
    assert claim["usable_as_verified_condition"] is False
    motion = value["recent_operations"][-1]
    assert motion["measured_translation_m"] == [.01, 0., 0.]
    assert motion["measured_path_m"] == .01
    assert value["object_attachment_verified"] is False
    assert value["task_success_certified"] is False


def test_null_initial_state_and_unobserved_conditions_remain_unknown():
    ledger = OperationEvidenceLedger()
    initial = ledger.presentation(0)
    assert initial["current_measured_state8"] is None
    assert not initial["prerequisites"] and not initial["recent_operations"]
    ledger.accept(operation(), visual(), 0, None)
    assert ledger.presentation(1)["current_measured_state8"] is None
    assert ledger.presentation(1)["prerequisites"][0]["state"] == "uncertain"


def test_validation_is_atomic_and_causal():
    ledger = OperationEvidenceLedger()
    ledger.accept(operation(), visual(), 0, state())
    before = ledger.presentation(0)
    bad = operation("pregrasp")
    with pytest.raises(ValueError):
        ledger.accept(bad, visual(), 1, state())
    assert ledger.presentation(0) == before
    with pytest.raises(ValueError):
        ledger.accept(operation(), visual(), 0, state())
    with pytest.raises(ValueError):
        ledger.record_feedback(state(), state()[:-1], 1)
    assert ledger.presentation(0) == before
    ledger.record_feedback(state(), state(.01), 1)
    before = ledger.presentation(1)
    with pytest.raises(ValueError):
        ledger.record_feedback(state(.01), state(.02), 1)
    assert ledger.presentation(1) == before


def test_same_operation_id_preserves_motion_and_new_id_starts_new_operation():
    ledger = OperationEvidenceLedger()
    ledger.accept(operation(), visual(), 0, state())
    ledger.record_feedback(state(), state(.01), 1)
    ledger.accept(operation(), visual(), 1, state(.01))
    ledger.record_feedback(state(.01), state(.02), 2)
    current = ledger.presentation(2)["recent_operations"][-1]
    assert current["total_operation_actions"] == 2
    assert current["request_action_count"] == 1
    assert current["measured_translation_m"] == [.02, 0., 0.]
    with pytest.raises(ValueError):
        ledger.accept(operation("transport"), visual("transport"), 2, state(.02))
    ledger.accept(operation("transport", id="op-2"), visual("transport"), 2, state(.02))
    assert ledger.presentation(2)["recent_operations"][-1]["total_operation_actions"] == 0


def test_missing_feedback_is_marked_not_filled_with_invented_object_motion():
    ledger = OperationEvidenceLedger()
    ledger.accept(operation(), visual(), 0, state())
    ledger.record_feedback(state(), state(.01), 1)
    ledger.record_feedback(state(.03), state(.04), 3)
    record = ledger.presentation(1)["recent_operations"][-1]
    assert record["feedback_chain_contiguous"] is False
    assert record["total_operation_actions"] == 2
    assert record["measured_translation_m"][0] == pytest.approx(.02)


def test_visual_diagnostics_are_bounded_attributed_and_never_certify_success():
    ledger = OperationEvidenceLedger()
    ledger.accept(operation(), visual(), 0, state())
    ledger.record_feedback(state(), state(.01), 1,
        {"status": "co_motion_support", "task_success_certified": True,
         "grasp_certified": True, "object_attachment_verified": True,
         "verification_evidence": "supporting", "verification_reason": "fixed_view_co_motion",
         "depth": [1.]*100, "sim_object_pose": [0.]*7, "image": "secret-data"})
    data = ledger.presentation(1)
    local = data["latest_visual_evidence"]
    assert local["status"] == "co_motion_support"
    assert local["task_success_certified"] is False and local["grasp_certified"] is False
    assert local["object_attachment_verified"] is False
    assert local["verification_evidence"] == "supporting"
    assert local["verification_reason"] == "fixed_view_co_motion"
    assert local["completed_action_step"] == 1
    assert "sim_object_pose" not in local and "image" not in local and "depth" not in local


def test_ledger_is_bounded_without_recursively_accumulating_presentations():
    ledger = OperationEvidenceLedger()
    for index in range(30):
        op = operation(id=f"op-{index}", intent="i"*240, expected_result="e"*240,
            prerequisites=[{"key": f"condition-{index}", "state": "supported", "evidence": "v"*160,
                            "source": "current_rgb_and_robot_state"}])
        ledger.accept(op, visual(), index, state(index*.001))
        ledger.record_feedback(state(index*.001), state((index+1)*.001), index+1,
            {"status": "tracking", "views": {"external": {"evidence": "x"*1000}}})
        result = ledger.presentation(index+1)
        assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= MAX_LEDGER_CHARS
        result["recent_operations"].clear()
        assert ledger.presentation(index+1)["recent_operations"]
    assert len(ledger._operations) <= 4
    assert len(ledger._claims) <= 8


def test_old_claim_action_age_survives_operation_history_eviction():
    ledger = OperationEvidenceLedger()
    for index in range(8):
        op = operation(id=f"op-{index}", prerequisites=[] if index else operation()["prerequisites"])
        ledger.accept(op, visual(), index, state(index*.001))
        ledger.record_feedback(state(index*.001), state((index+1)*.001), index+1)
    assert ledger.presentation(8)["prerequisites"][0]["executed_actions_since_claim"] == 8


def test_rotation_feedback_uses_physical_angle_not_axis_angle_subtraction():
    ledger = OperationEvidenceLedger()
    initial, after = state(), state()
    initial[3] = 3.141592653589793
    after[3] = -3.141592653589793
    ledger.accept(operation(), visual(), 0, initial)
    ledger.record_feedback(initial, after, 1)
    angle = ledger.presentation(1)["recent_operations"][-1]["measured_orientation_change_rad"]
    assert angle == pytest.approx(0., abs=1e-8)
