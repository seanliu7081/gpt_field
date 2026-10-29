"""Explicit RGB obstacle review and task motion purpose without clearance claims."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v8 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV8, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v6_protocol import prediction as prediction_v6
from test_visual_field_v7_protocol import stream_error, current_record


def motion_context(*, present=False, purpose="task_translation"):
    return {"purpose": purpose, "tool_external_roi_uv": [.1, .2, .35, .5],
            "nearby_obstacles": [{"id": "neighbor", "external_roi_uv": [.5, .5, .7, .7]}] if present else [],
            "obstacle_assessment": "present" if present else "none_relevant"}


def prediction(*, controller="direct", phase="approach", present=False):
    value = prediction_v6(controller=controller, phase=phase)
    value["motion_context"] = motion_context(present=present)
    if controller == "visual_field" and present:
        value["field_intent"]["obstacles"] = [{"id": "neighbor", "center_offset_m": [.02, .03, -.04],
            "half_extents_m": [.01, .015, .02], "uncertainty_m": .005}]
    return value


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}, id=f"v8_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV8(token_provider=lambda:"offline-oauth-fixture",
                                            transport=transport, **kwargs), transport


def test_present_field_neighbors_have_concrete_matching_metric_hypotheses_and_raw_history():
    value = prediction(controller="visual_field", present=True)
    before = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    result = instance.predict(observation().to_dict())
    assert value == before
    assert result["motion_context"] == value["motion_context"]
    box = result["field_intent"]["obstacles"][0]
    assert box["id"] == "neighbor"
    assert box["lower_offset_m"] == pytest.approx([.01, .015, -.06])
    assert box["upper_offset_m"] == pytest.approx([.03, .045, -.02])
    assert "orientation_target_xyzw" in result["field_intent"]
    instance.predict(observation(1).to_dict())
    raw = [json.loads(part["text"]) for message in transport.calls[-1]["input"]
           if message.get("role") == "assistant" for part in message.get("content", [])
           if part.get("type") == "output_text"]
    assert raw == [before]


def test_present_neighbor_can_be_an_explicit_unknown_metric_region():
    value = prediction(controller="visual_field", present=True)
    value["field_intent"]["unknown_regions"] = value["field_intent"]["obstacles"]
    value["field_intent"]["obstacles"] = []
    instance, _ = client()
    assert instance.validate_response(value, None)["field_intent"]["unknown_regions"][0]["id"] == "neighbor"


@pytest.mark.parametrize("purpose", ["reference_repair", "task_reorientation", "contact_correction", "articulation"])
@pytest.mark.parametrize("roi", [None, [.1, .2, .35, .5]])
def test_rotation_hypotheses_reach_runtime_unchanged_for_auditable_veto(purpose, roi):
    value = prediction()
    value["direct_actions"] = [[0., 0., 0., 0., 0., .15, 0.]]*5
    value["motion_context"].update(purpose=purpose, tool_external_roi_uv=roi)
    instance, transport = client(value)
    assert instance.predict(observation().to_dict()) == value
    assert len(transport.calls) == 1  # A risk veto is not a malformed-wire retry.


@pytest.mark.parametrize("neighbors", [False, True])
def test_unobservable_assessment_keeps_uncertainty_explicit_and_known_neighbors_optional(neighbors):
    value = prediction(controller="visual_field", present=neighbors)
    value["motion_context"]["obstacle_assessment"] = "unobservable"
    value["motion_context"]["tool_external_roi_uv"] = None
    instance, _ = client()
    result = instance.validate_response(value, None)
    assert result["motion_context"]["obstacle_assessment"] == "unobservable"
    assert result["motion_context"]["tool_external_roi_uv"] is None


def test_unobservable_other_geometry_cannot_drop_a_declared_known_field_neighbor():
    value = prediction(controller="visual_field", present=True)
    value["motion_context"]["obstacle_assessment"] = "unobservable"
    box = value["field_intent"]["obstacles"].pop()
    instance, _ = client()
    with pytest.raises(SchemaError, match="Every listed image neighbor"):
        instance.validate_response(value, None)
    value["field_intent"]["unknown_regions"] = [box]
    result = instance.validate_response(value, None)
    assert result["field_intent"]["unknown_regions"][0]["id"] == "neighbor"


@pytest.mark.parametrize("case", ["missing_context", "null_context", "extra_property", "bad_purpose",
    "bad_assessment", "reversed_tool", "zero_tool", "reversed_neighbor", "outside_neighbor", "nan",
    "present_without_neighbors", "none_with_neighbors", "empty_id", "repeated_id", "unmapped_field_id",
    "second_unmapped_id", "too_many_neighbors", "missing_tool_roi"])
def test_malformed_geometry_or_inconsistent_obstacle_review_is_rejected(case):
    value = prediction(controller="visual_field", present=True)
    context = value["motion_context"]
    if case == "missing_context": value.pop("motion_context")
    elif case == "null_context": value["motion_context"] = None
    elif case == "extra_property": context["clearance_certified"] = True
    elif case == "bad_purpose": context["purpose"] = "automatic_yaw"
    elif case == "bad_assessment": context["obstacle_assessment"] = "certified_clear"
    elif case == "reversed_tool": context["tool_external_roi_uv"] = [.3, .2, .1, .5]
    elif case == "zero_tool": context["tool_external_roi_uv"] = [.1, .2, .1, .5]
    elif case == "reversed_neighbor": context["nearby_obstacles"][0]["external_roi_uv"] = [.7, .5, .5, .7]
    elif case == "outside_neighbor": context["nearby_obstacles"][0]["external_roi_uv"] = [.5, .5, 1.1, .7]
    elif case == "nan": context["tool_external_roi_uv"][0] = float("nan")
    elif case == "present_without_neighbors": context["nearby_obstacles"] = []
    elif case == "none_with_neighbors": context["obstacle_assessment"] = "none_relevant"
    elif case == "empty_id": context["nearby_obstacles"][0]["id"] = " "
    elif case == "repeated_id": context["nearby_obstacles"] *= 2
    elif case == "unmapped_field_id": value["field_intent"]["obstacles"] = []
    elif case == "second_unmapped_id":
        context["nearby_obstacles"].append({"id":"other", "external_roi_uv":[.7, .7, .9, .9]})
    elif case == "too_many_neighbors":
        context["nearby_obstacles"] = [{"id":f"n{i}", "external_roi_uv":[.5, .5, .7, .7]} for i in range(7)]
    elif case == "missing_tool_roi": context.pop("tool_external_roi_uv")
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_obstacle_contract_error_uses_one_bounded_semantic_repair_without_reordering(tmp_path):
    bad = prediction(controller="visual_field", present=True)
    bad["motion_context"]["tool_external_roi_uv"] = [.35, .2, .1, .5]
    valid = prediction(controller="visual_field", present=True)
    instance, transport = client(bad, valid, log_dir=tmp_path)
    original = copy.deepcopy(bad)
    result = instance.predict(observation().to_dict())
    assert result["motion_context"] == valid["motion_context"] and bad == original
    assert len(transport.calls) == 2
    repair = current_record(transport.calls[1])["observable_context"]["output_repair"]
    assert "ordered positive-area" in repair["validation_error"]
    assert json.loads(repair["rejected_model_output_json"]) == original
    assert len(list(tmp_path.glob("*.response.json"))) == 2
    assert instance.history_stats()["semantic_repair_requests"] == 1


def test_stream_and_semantic_repair_remain_shared_and_accounted():
    bad = prediction(controller="visual_field", present=True)
    bad["field_intent"]["obstacles"] = []
    valid = prediction(controller="visual_field", present=True)
    instance, transport = client(stream_error(), bad, valid)
    result = instance.predict(observation().to_dict())
    assert len(transport.calls) == 3 and result["motion_context"] == valid["motion_context"]
    assert instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["unknown_usage_calls"] == 1
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.history_stats()["stream_retry_requests"] == 1


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_generic_v8_guidance_preserves_hard_controls_and_current_obstacle_review(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    payload = transport.calls[-1]
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort":"medium"}
    assert payload["text"]["format"]["schema"] == output_schema()
    assert_unique_required(output_schema())
    assert instance.prompt_version == VERSION+"-hybrid"
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert next(x for x in input_texts(payload) if x["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    text = shared_guidance(resolve_distance_task(task))
    assert len(text.split()) <= 1050
    for phrase in ("response_evidence", "inconclusive", "Do not rotate just to repair tracking",
                   "outside the wrist crop", "not a tracking patch", "metric box", "unknown_regions",
                   "risk heuristic", "zero XYZ", "Field release/z", "exactly five", "no completion flag"):
        assert phrase in text
    with pytest.raises(ValueError): shared_guidance(resolve_distance_task(task), demonstration=True)


def test_gripper_and_direct_release_rules_still_apply_with_motion_context():
    value = prediction(controller="gripper", phase="grasp")
    instance, _ = client()
    assert instance.validate_response(value, None) == value
    invalid = prediction(phase="release")
    invalid["direct_actions"][0][0] = .1
    with pytest.raises(SchemaError, match="zero XYZ"):
        instance.validate_response(invalid, None)


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_v8_context_and_images_do_not_admit_privileged_information(name):
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback": {name:"not-policy-input"}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls
