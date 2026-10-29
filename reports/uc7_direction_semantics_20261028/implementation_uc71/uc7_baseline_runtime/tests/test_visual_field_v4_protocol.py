"""V4 contracts and subscription/privacy checks, using fake transport only."""
import copy
import json

import pytest

from gpt_field.astra import APIError, MissingCredentials, SchemaError
from gpt_field.visual_field_subscription_v4 import (
    VisualFieldSubscriptionClientV4, VERSION, HISTORY_VERSION, output_schema,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, prediction as prior_prediction, input_texts
from test_visual_field_v2_protocol import assert_unique_required


def external():
    return {"moving_id": "tool_tip", "moving_kind": "tool", "moving_point_uv": [.3, .4],
            "target_id": "task_target", "target_anchor_uv": [.7, .6], "target_point_uv": [.6, .5],
            "tool_point_uv": None}


def wrist():
    return {"target_id": "task_target", "target_anchor_uv": [.7, .6], "desired_target_uv": [.5, .5]}


def box(name="obstacle_a"):
    return {"id": name, "center_offset_m": [.02, .03, -.035],
            "half_extents_m": [.01, .01, .015], "uncertainty_m": .005}


def prediction(*, field=False, phase="approach", actions=None):
    value = prior_prediction(field=field, phase=phase, actions=actions)
    value["visual_references"] = {"external": external(), "wrist": None}
    if field:
        value["field_intent"].update(goal_tolerance_m=.005, max_actions=30,
            max_travel_m=.12, command_limit_m=.01, tool_radius_m=.015, uncertainty_m=.005,
            obstacles=[], unknown_regions=[])
    return value


def client(*values, **kwargs):
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort": "medium"},
                                         id=f"v4_{index}") for index, value in enumerate(values)))
    return VisualFieldSubscriptionClientV4(token_provider=lambda: "offline-oauth-fixture",
                                           transport=transport, **kwargs), transport


def test_v4_separate_camera_contract_accepts_missing_wrist_without_a_tool_footprint():
    instance, _ = client()
    value = prediction(field=True)
    assert instance.validate_response(value, None) == value
    value["visual_references"]["wrist"] = wrist()
    assert instance.validate_response(value, None) == value
    assert_unique_required(output_schema())
    text = json.dumps(output_schema())
    assert "moving_footprint_uv" not in text and "blocked_boxes_uv" not in text
    from gpt_field.visual_field_subscription_v3 import output_schema as schema_v3
    assert "moving_footprint_uv" in json.dumps(schema_v3())
    assert "command_limit_m" not in json.dumps(schema_v3())


@pytest.mark.parametrize("case", ["direct_field", "field_direct", "missing_external", "grasp_field",
    "field_close", "bad_quaternion", "slightly_nonunit", "fractional_budget", "bool_budget", "zero_budget", "large_budget",
    "goal_outside_grid", "huge_command", "zero_tolerance", "nan", "transport_tool", "transport_no_tool",
    "duplicate_box", "blank_box", "negative_extent", "zero_volume", "nan_extent", "bool_extent",
    "huge_extent", "huge_center", "legacy_corners", "huge_uncertainty", "unknown_extra",
    "six_direct_actions", "wrist_tool_footprint", "mismatched_target", "same_landmark", "blank_hypothesis"])
def test_invalid_hypothesis_never_passes_execution_contract(case):
    value = prediction(field=case not in ("direct_field", "six_direct_actions"))
    if case == "direct_field": value["field_intent"] = prediction(field=True)["field_intent"]
    elif case == "field_direct": value["direct_actions"] = [[0.] * 7] * 5
    elif case == "missing_external": value["visual_references"]["external"] = None
    elif case == "grasp_field": value["phase"] = "grasp"
    elif case == "field_close": value["field_intent"]["gripper_command"] = "close"
    elif case == "bad_quaternion": value["field_intent"]["orientation_target_xyzw"] = [0., 0., 0., 0.]
    elif case == "slightly_nonunit": value["field_intent"]["orientation_target_xyzw"] = [0., 0., .1, 1.]
    elif case in ("fractional_budget", "bool_budget", "zero_budget", "large_budget"):
        value["field_intent"]["max_actions"] = {"fractional_budget": 30.5, "bool_budget": True,
                                                "zero_budget": 0, "large_budget": 61}[case]
    elif case == "goal_outside_grid": value["field_intent"]["goal_offset_m"][0] = .081
    elif case == "huge_command": value["field_intent"]["command_limit_m"] = .021
    elif case == "zero_tolerance": value["field_intent"]["goal_tolerance_m"] = 0.
    elif case == "nan": value["field_intent"]["goal_offset_m"][0] = float("nan")
    elif case in ("transport_tool", "transport_no_tool"):
        value["phase"] = "transport"
        if case == "transport_no_tool": value["visual_references"]["external"]["moving_kind"] = "payload"
    elif case in ("duplicate_box", "blank_box", "negative_extent", "zero_volume", "nan_extent", "bool_extent",
                 "huge_extent", "huge_center", "legacy_corners", "huge_uncertainty"):
        region = box(); value["field_intent"]["obstacles"] = [region]
        if case == "duplicate_box": value["field_intent"]["unknown_regions"] = [box()]
        elif case == "blank_box": region["id"] = " "
        elif case in ("negative_extent", "zero_volume", "nan_extent", "bool_extent", "huge_extent"):
            region["half_extents_m"][0] = {"negative_extent": -.01, "zero_volume": 0.,
                "nan_extent": float("nan"), "bool_extent": True, "huge_extent": .501}[case]
        elif case == "huge_center": region["center_offset_m"][0] = -.501
        elif case == "legacy_corners":
            region.pop("center_offset_m"); region.pop("half_extents_m")
            region.update(lower_offset_m=[-.1, -.33, -.1], upper_offset_m=[.1, -.34, .1])
        elif case == "huge_uncertainty": region["uncertainty_m"] = .101
    elif case == "unknown_extra": value["task_complete"] = True
    elif case == "six_direct_actions": value["direct_actions"].append([0.] * 7)
    elif case == "wrist_tool_footprint":
        value["visual_references"]["wrist"] = {**wrist(), "moving_footprint_uv": [[0., 0.], [1., 1.]]}
    elif case == "mismatched_target": value["visual_references"]["external"]["target_id"] = "other_object"
    elif case == "same_landmark": value["visual_references"]["external"]["moving_id"] = "task_target"
    elif case == "blank_hypothesis": value["hypothesis_id"] = " "
    instance, transport = client()
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)
    assert not transport.calls


def test_center_extents_preserve_geometry_uncertainty_and_all_region_ids():
    value = prediction(field=True)
    value["field_intent"]["obstacles"] = [box(f"obstacle_{i}") for i in range(8)]
    value["field_intent"]["unknown_regions"] = [box(f"unknown_{i}") for i in range(8)]
    # A negative-coordinate thin region cannot produce reversed corners with
    # positive half extents, unlike the old failing right-cabinet response.
    value["field_intent"]["obstacles"][0].update(
        center_offset_m=[0., -.335, 0.], half_extents_m=[.01, .005, .02])
    original = copy.deepcopy(value)
    instance, _ = client()
    result = instance.validate_response(value, None)
    assert value == original
    for name in ("obstacles", "unknown_regions"):
        assert len(result["field_intent"][name]) == 8
        for raw, converted in zip(value["field_intent"][name], result["field_intent"][name]):
            assert converted["id"] == raw["id"]
            assert converted["uncertainty_m"] == raw["uncertainty_m"]
            assert converted["lower_offset_m"] == pytest.approx([
                c-h for c, h in zip(raw["center_offset_m"], raw["half_extents_m"])])
            assert converted["upper_offset_m"] == pytest.approx([
                c+h for c, h in zip(raw["center_offset_m"], raw["half_extents_m"])])
    cabinet = result["field_intent"]["obstacles"][0]
    assert cabinet["lower_offset_m"][1] == pytest.approx(-.34)
    assert cabinet["upper_offset_m"][1] == pytest.approx(-.33)


def test_center_extents_contract_has_bounded_valid_corners_at_numeric_limits():
    value = prediction(field=True)
    value["field_intent"]["obstacles"] = [dict(box(), center_offset_m=[-.5, .5, 0.],
                                                half_extents_m=[.5, .5, .0001])]
    instance, _ = client()
    converted = instance.validate_response(value, None)["field_intent"]["obstacles"][0]
    assert converted["lower_offset_m"] == [-1., 0., -.0001]
    assert converted["upper_offset_m"] == [0., 1., .0001]
    schema = json.dumps(output_schema())
    assert "lower_offset_m" not in schema and "upper_offset_m" not in schema
    assert "center_offset_m" in schema and "half_extents_m" in schema


def test_converted_execution_corners_never_replace_raw_wire_history_or_logs(tmp_path):
    value = prediction(field=True)
    value["field_intent"]["obstacles"] = [box()]
    original = copy.deepcopy(value)
    instance, transport = client(value, prediction(), log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert "lower_offset_m" in result["field_intent"]["obstacles"][0]
    instance.predict(observation(1).to_dict())
    messages = transport.calls[1]["input"]
    outputs = [json.loads(part["text"]) for message in messages if message.get("role") == "assistant"
               for part in message["content"] if part["type"] == "output_text"]
    assert outputs == [original]
    raw = json.loads(sorted(tmp_path.glob("*.response.json"))[0].read_text())
    logged = [json.loads(part["text"]) for message in raw["output"] if message.get("type") == "message"
              for part in message["content"] if part["type"] == "output_text"]
    assert logged == [original]
    assert value == original


def test_transport_has_payload_and_independent_tool_but_no_wrist_tool_requirement():
    value = prediction(field=True, phase="transport")
    value["visual_references"]["external"].update(moving_kind="payload", tool_point_uv=[.28, .4])
    instance, _ = client()
    assert instance.validate_response(value, None) == value


@pytest.mark.parametrize("kwargs", [{"controller": "direct"}, {"demonstration": object()}])
def test_v4_scope_is_zero_shot_hybrid(kwargs):
    with pytest.raises(ValueError, match="Hybrid only"):
        client(**kwargs)


@pytest.mark.parametrize("hidden", ["object_poses", "depth", "contacts", "success", "camera_calibration",
                                      "task_complete", "libero_success"])
def test_privileged_observation_or_nested_context_never_reaches_model(hidden):
    instance, transport = client(prediction())
    obs = observation().to_dict(); obs[hidden] = "private"
    with pytest.raises(ValueError): instance.predict(obs)
    instance.observable_context = {"field_feedback": {hidden: "private"}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    with pytest.raises(ValueError): instance.record_block({"type": "visual_execution_block", hidden: "private"})
    assert not transport.calls


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_medium_subscription_new_prompt_and_compacted_history(task, tmp_path):
    instance, transport = client(*(prediction() for _ in range(5)), task_key=task, log_dir=tmp_path)
    for index in range(5): instance.predict(observation(index, task=task).to_dict())
    request = transport.calls[-1]
    assert request["model"] == "gpt-6-astra" and request["reasoning"] == {"effort": "medium"}
    assert request["text"]["format"]["name"] == "visual_field_rgb_joints_v4"
    assert request["text"]["format"]["schema"] == output_schema()
    assert "ZERO-SHOT" in request["instructions"] and "bounded cold-start" in request["instructions"]
    assert "V3 FEASIBILITY" not in request["instructions"]
    assert "Do not repeatedly request Direct calibration" in request["instructions"]
    assert "a complete tool footprint" not in request["instructions"]
    records = input_texts(request)
    assert sum(record["type"] == "observation" for record in records) == 2
    assert next(record for record in records if record["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    assert len(list(tmp_path.glob("*.request.json"))) == 5
    budget = instance.budget_snapshot()
    assert budget["billing"] == "chatgpt_subscription" and budget["api_key_calls"] == 0
    assert instance.prompt_version == VERSION + "-hybrid"


def test_subscription_auth_failure_never_uses_available_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    transport = FakeTransport(envelope(prediction(), model="gpt-6-astra", reasoning={"effort": "medium"}))
    def unavailable():
        raise MissingCredentials("offline deliberate auth failure")
    instance = VisualFieldSubscriptionClientV4(token_provider=unavailable, transport=transport)
    with pytest.raises(MissingCredentials): instance.predict(observation().to_dict())
    assert not transport.calls and instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("change", [{"model": "other-model"}, {"reasoning": {"effort": "high"}}])
def test_provider_model_or_effort_mismatch_is_terminal(change):
    instance, transport = client(prediction())
    transport.responses[0].update(copy.deepcopy(change))
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1
