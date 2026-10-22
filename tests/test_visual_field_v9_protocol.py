"""V9 geometry meaning changes without relaxing the V8 wire or observation contract."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v8 import output_schema as schema_v8, shared_guidance as guidance_v8
from gpt_field.visual_field_subscription_v9 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV9, geometry_semantics, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v7_protocol import current_record, stream_error
from test_visual_field_v8_protocol import prediction


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}, id=f"v9_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV9(token_provider=lambda:"offline-oauth-fixture",
                                            transport=transport, **kwargs), transport


@pytest.mark.parametrize("assessment", ["present", "unobservable"])
@pytest.mark.parametrize("group", ["obstacles", "unknown_regions"])
def test_known_neighbor_mapping_remains_mandatory_for_both_geometry_classes(assessment, group):
    value = prediction(controller="visual_field", present=True)
    value["motion_context"]["obstacle_assessment"] = assessment
    box = value["field_intent"]["obstacles"].pop()
    instance, _ = client()
    with pytest.raises(SchemaError, match="Every listed image neighbor"):
        instance.validate_response(value, None)
    value["field_intent"][group] = [box]
    result = instance.validate_response(value, None)
    assert result["field_intent"][group][0]["id"] == "neighbor"
    assert result["field_intent"][group][0]["uncertainty_m"] == .005
    assert result["motion_context"] == value["motion_context"]


@pytest.mark.parametrize("case", ["duplicate_across_geometry", "unknown_id_mismatch", "second_neighbor_missing",
    "none_relevant_with_neighbors", "negative_extent", "negative_uncertainty", "reversed_tool_roi",
    "missing_motion_context", "unrequested_geometry_mode", "nonzero_locked_axis", "direct_release_translation",
    "six_direct_actions", "gripper_in_transport"])
def test_new_field_meaning_does_not_relax_existing_control_and_geometry_constraints(case):
    value = prediction(controller="visual_field", present=True)
    intent = value["field_intent"]
    if case == "duplicate_across_geometry": intent["unknown_regions"] = copy.deepcopy(intent["obstacles"])
    elif case == "unknown_id_mismatch":
        intent["unknown_regions"] = intent["obstacles"]; intent["obstacles"] = []
        intent["unknown_regions"][0]["id"] = "not_named_neighbor"
    elif case == "second_neighbor_missing":
        value["motion_context"]["nearby_obstacles"].append({"id":"forgotten", "external_roi_uv":[.7, .7, .8, .8]})
    elif case == "none_relevant_with_neighbors": value["motion_context"]["obstacle_assessment"] = "none_relevant"
    elif case == "negative_extent": intent["obstacles"][0]["half_extents_m"][0] = -.01
    elif case == "negative_uncertainty": intent["obstacles"][0]["uncertainty_m"] = -.01
    elif case == "reversed_tool_roi": value["motion_context"]["tool_external_roi_uv"] = [.5, .5, .1, .1]
    elif case == "missing_motion_context": value.pop("motion_context")
    elif case == "unrequested_geometry_mode": value["scene_geometry_mode"] = "ignore_obstacles"
    elif case == "nonzero_locked_axis": intent["motion_axes"] = "z"
    elif case == "direct_release_translation":
        value = prediction(phase="release"); value["direct_actions"][0][0] = .1
    elif case == "six_direct_actions":
        value = prediction(); value["direct_actions"].append([0.]*7)
    elif case == "gripper_in_transport": value = prediction(controller="gripper", phase="transport")
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_wire_history_keeps_nominal_geometry_and_uncertainty_without_client_shrink_or_reclassification():
    value = prediction(controller="visual_field", present=True)
    value["field_intent"]["obstacles"][0]["uncertainty_m"] = .028
    value["field_intent"]["unknown_regions"] = [{"id":"occluded_gap", "center_offset_m":[.04, -.03, .01],
        "half_extents_m":[.02, .01, .015], "uncertainty_m":.021}]
    original = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    context = {"visual_feedback":{"geometry_semantics":geometry_semantics()}}
    instance.observable_context = context
    result = instance.predict(observation().to_dict())
    assert value == original and instance.observable_context is context
    assert result["field_intent"]["obstacles"][0]["uncertainty_m"] == .028
    assert result["field_intent"]["unknown_regions"][0]["uncertainty_m"] == .021
    assert result["field_intent"]["obstacles"][0]["lower_offset_m"] == pytest.approx([.01, .015, -.06])
    assert current_record(transport.calls[0])["observable_context"] == context
    instance.predict(observation(1).to_dict())
    accepted = [json.loads(part["text"]) for message in transport.calls[-1]["input"]
                if message.get("role") == "assistant" for part in message.get("content", [])
                if part.get("type") == "output_text"]
    assert accepted == [original]


@pytest.mark.parametrize("case,soft", [("matched", True), ("unobservable", False),
    ("missing_tool", False), ("unmatched_extra_box", False), ("none_relevant_with_box", False),
    ("unknown", True)])
def test_wire_scene_support_controls_executor_uncertainty_without_forgetting_any_region(case, soft):
    from gpt_field.visual_field_v9 import MetricFieldExecutorV9

    value = prediction(controller="visual_field", present=True)
    context, intent = value["motion_context"], value["field_intent"]
    if case == "unobservable": context["obstacle_assessment"] = "unobservable"
    elif case == "missing_tool": context["tool_external_roi_uv"] = None
    elif case == "unmatched_extra_box":
        extra = copy.deepcopy(intent["obstacles"][0]); extra["id"] = "unlocalized_extra"
        intent["obstacles"].append(extra)
    elif case == "none_relevant_with_box":
        context["obstacle_assessment"] = "none_relevant"; context["nearby_obstacles"] = []
    elif case == "unknown":
        intent["unknown_regions"] = intent["obstacles"]; intent["obstacles"] = []
    original = copy.deepcopy(value)
    instance, transport = client(value)
    observed = observation()
    result = instance.predict(observed.to_dict())
    executor = MetricFieldExecutorV9()
    report = executor.configure(observed.state, result["field_intent"],
                                scene_assessment=result["motion_context"])
    assert report["soft_uncertainty_enabled"] is soft
    assert len(transport.calls) == 1 and value == original
    assert len(executor.regions) == len(intent["obstacles"]) + len(intent["unknown_regions"])
    for group in ("obstacles", "unknown_regions"):
        for raw in intent[group]:
            region = next(item for item in executor.regions if item["id"] == raw["id"])
            is_soft = soft and group == "obstacles"
            inflation = intent["tool_radius_m"] + (0. if is_soft else intent["uncertainty_m"] + raw["uncertainty_m"])
            assert region["inflation_m"] == pytest.approx(inflation)
            assert region["kind"] == group
            expected = "hard_tool_core_soft_uncertainty" if is_soft else (
                "hard_full_unknown" if group == "unknown_regions" else "hard_full_strict_assessment")
            assert region["geometry_mode"] == expected


def test_bounded_subscription_and_semantic_retry_are_inherited_with_real_accounting(tmp_path):
    bad = prediction(controller="visual_field", present=True)
    bad["field_intent"]["obstacles"] = []
    value = prediction(controller="visual_field", present=True)
    instance, transport = client(stream_error(), bad, value, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert result["motion_context"] == value["motion_context"]
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.budget_snapshot()["unknown_usage_calls"] == 1
    assert instance.history_stats()["semantic_repair_requests"] == 1
    assert instance.history_stats()["stream_retry_requests"] == 1
    assert instance.history_stats()["observations"] == 1
    assert len(list(tmp_path.glob("*.request.json"))) == 3
    assert json.loads(current_record(transport.calls[-1])["observable_context"]["output_repair"]
                      ["rejected_model_output_json"]) == bad


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_prompt_replaces_all_hard_visible_uncertainty_with_explicit_nominal_core_semantics(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    payload = transport.calls[-1]
    assert output_schema() == schema_v8()
    assert_unique_required(output_schema())
    assert payload["text"]["format"]["schema"] == schema_v8()
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v9"
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort":"medium"}
    assert payload["stream"] is True and payload["store"] is False
    assert instance.prompt_version == VERSION+"-hybrid"
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert next(x for x in input_texts(payload) if x["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    text = shared_guidance(resolve_distance_task(task))
    assert len(text.split()) <= 1250
    for phrase in ("approximate NOMINAL metric boxes", "core inflated by tool/payload radius is HARD excluded",
                   "NONNEGATIVE traversal cost", "not calibrated collision probability",
                   "genuinely occluded, unseen or unlocalized", "remain fully HARD inflated",
                   "Lack of calibrated depth alone", "Never reclassify, shrink or delete",
                   "visual_feedback.geometry_semantics", "current whole-tool ROI", "matching EVERY obstacles",
                   "retains FULL hard uncertainty inflation", "EVERY listed neighbor", "0.015", "0.005",
                   "appropriate bounded Field translation", "Request explicit gripper", "endless static holds",
                   "zero XYZ", "Field release/z", "exactly five", "no completion flag"):
        assert phrase in text
    assert "Local planning inflates by radius/uncertainty" not in text
    assert "Local planning inflates by radius/uncertainty" in guidance_v8(resolve_distance_task(task))


def test_geometry_feedback_is_fresh_observable_metadata_not_a_clearance_claim():
    first, second = geometry_semantics(), geometry_semantics()
    first["unobservable_scene_limits"]["max_travel_m"] = 2.
    assert second["unobservable_scene_limits"] == {"max_travel_m":.015, "command_limit_m":.005}
    assert second["clearance_certified"] is False
    assert "hard" in second["obstacles"] and "nonnegative" in second["visible_uncertainty"]
    assert "matching every obstacles entry" in second["soft_geometry_permission"]
    assert "full hard uncertainty inflation" in second["soft_geometry_permission"]


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_v9_does_not_admit_privileged_evaluator_or_geometry_inputs(name):
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback":{"geometry_semantics":{name:"not-policy-input"}}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls


def test_demonstrations_and_direct_baseline_are_still_not_v9_inputs():
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV9(controller="direct")
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV9(demonstration=object())
    with pytest.raises(ValueError): shared_guidance(resolve_distance_task("bowl_plate"), demonstration=True)
