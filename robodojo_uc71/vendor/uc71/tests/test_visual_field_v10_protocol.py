"""Explicit lifecycle wire validation and inherited subscription constraints."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v9 import output_schema as schema_v9
from gpt_field.visual_field_subscription_v10 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV10, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v7_protocol import current_record, stream_error
from test_visual_field_v8_protocol import prediction as prediction_v8


def prediction(*, controller="direct", phase="approach", present=False, action=None):
    value = prediction_v8(controller=controller, phase=phase, present=present)
    action = action or ("new" if controller == "visual_field" else "none")
    value["subgoal_update"] = {"action": action, "subgoal_id": None if action == "none" else "approach_goal",
                               "reason": "Current visible task evidence."}
    return value


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}, id=f"v10_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV10(token_provider=lambda:"offline-oauth-fixture",
                                            transport=transport, **kwargs), transport


@pytest.mark.parametrize("controller,action", [("visual_field", x) for x in ("new", "retain", "replace")]
    + [(c, a) for c in ("direct", "gripper") for a in ("retain", "finish", "none")])
def test_explicit_valid_lifecycle_operations_preserve_raw_wire(controller, action):
    value = prediction(controller=controller, phase="grasp" if controller == "gripper" else "approach", action=action)
    before = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    result = instance.predict(observation().to_dict())
    assert result["subgoal_update"] == before["subgoal_update"] and value == before
    instance.predict(observation(1).to_dict())
    raw = [json.loads(part["text"]) for message in transport.calls[-1]["input"]
           if message.get("role") == "assistant" for part in message["content"] if part["type"] == "output_text"]
    assert raw == [before]


@pytest.mark.parametrize("case", ["missing", "extra", "null_id", "blank_id", "none_with_id",
    "replace_without_reason", "replace_blank_reason", "field_finish", "field_none", "direct_new", "gripper_replace",
    "forgotten_neighbor", "direct_six", "release_translation"])
def test_malformed_lifecycle_and_inherited_control_contracts_are_rejected(case):
    value = prediction(controller="visual_field", present=True)
    update = value["subgoal_update"]
    if case == "missing": value.pop("subgoal_update")
    elif case == "extra": update["world_goal_xyz"] = [0, 0, 0]
    elif case == "null_id": update["subgoal_id"] = None
    elif case == "blank_id": update["subgoal_id"] = "  "
    elif case == "none_with_id": value = prediction(); value["subgoal_update"]["subgoal_id"] = "wrong"
    elif case == "replace_without_reason": update["action"] = "replace"; update.pop("reason")
    elif case == "replace_blank_reason": update.update(action="replace", reason="  ")
    elif case == "field_finish": update["action"] = "finish"
    elif case == "field_none": update.update(action="none", subgoal_id=None)
    elif case == "direct_new": value = prediction(action="new")
    elif case == "gripper_replace": value = prediction(controller="gripper", phase="grasp", action="replace")
    elif case == "forgotten_neighbor": value["field_intent"]["obstacles"] = []
    elif case == "direct_six": value = prediction(); value["direct_actions"].append([0.]*7)
    elif case == "release_translation": value = prediction(phase="release"); value["direct_actions"][0][0] = .1
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_retained_field_relative_proposal_is_kept_for_ledger_and_history_not_silently_rewritten():
    value = prediction(controller="visual_field", action="retain")
    value["field_intent"]["goal_offset_m"] = [.02, .01, .04]
    instance, transport = client(value)
    context = {"visual_feedback":{"subgoal_ledger":{"active_id":"approach_goal", "goal_xyz":[.11, .2, 1.03]}}}
    instance.observable_context = context
    result = instance.predict(observation().to_dict())
    assert result["field_intent"]["goal_offset_m"] == [.02, .01, .04]
    assert current_record(transport.calls[0])["observable_context"] == context
    assert instance.observable_context is context


def test_semantic_and_missing_stream_retries_remain_bounded_and_accounted(tmp_path):
    invalid = prediction(controller="visual_field"); invalid["subgoal_update"]["subgoal_id"] = None
    valid = prediction(controller="visual_field")
    instance, transport = client(invalid, stream_error(), valid, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert result["subgoal_update"] == valid["subgoal_update"]
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.budget_snapshot()["unknown_usage_calls"] == 1
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["semantic_repair_requests"] == instance.history_stats()["stream_retry_requests"] == 1
    assert len(list(tmp_path.glob("*.request.json"))) == 3
    assert json.loads(current_record(transport.calls[-1])["observable_context"]["output_repair"]
                      ["rejected_model_output_json"]) == invalid
    assert instance.observable_context == {}


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_prompt_explains_persistent_subgoals_without_relaxing_geometry_or_observation_contract(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    payload = transport.calls[-1]
    assert_unique_required(output_schema())
    stripped = output_schema(); stripped["properties"].pop("subgoal_update"); stripped["required"].remove("subgoal_update")
    assert stripped == schema_v9()
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v10"
    assert payload["text"]["format"]["schema"] == output_schema()
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort":"medium"}
    assert payload["stream"] is True and payload["store"] is False
    assert instance.prompt_version == VERSION+"-hybrid"
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert next(v for v in input_texts(payload) if v["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    text = shared_guidance(resolve_distance_task(task))
    for phrase in ("SAME stored world goal", "logged but ignored", "finish", "measured arrival", "SAME active ID",
                   "visual_feedback.subgoal_ledger", "diagnostic translation/test lift", "Direct5", "NOT task completion",
                   "retains FULL hard uncertainty inflation", "matching EVERY obstacles", "0.015", "0.005",
                   "no completion flag", "zero XYZ", "Field release/z"):
        assert phrase in text
    assert "After closing, use a short Direct test" not in text


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_ledger_feedback_never_admits_privileged_inputs(name):
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback":{"subgoal_ledger":{name:"not-policy-input"}}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls


def test_zero_shot_hybrid_only():
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV10(controller="direct")
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV10(demonstration=object())
