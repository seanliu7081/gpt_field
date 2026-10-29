"""Translation mode labels preserve all V10 goal and V9 geometry constraints."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v10 import output_schema as schema_v10
from gpt_field.visual_field_subscription_v11 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV11, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v7_protocol import current_record, stream_error
from test_visual_field_v10_protocol import prediction as prediction_v10


def prediction(*, controller="direct", phase="approach", present=False, action=None, mode="transit"):
    value = prediction_v10(controller=controller, phase=phase, present=present, action=action)
    value["motion_context"].update(translation_mode=mode,
        intended_contact_id=value["target_id"] if mode == "interaction" else None)
    return value


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}, id=f"v11_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV11(token_provider=lambda:"offline-oauth-fixture",
                                            transport=transport, **kwargs), transport


@pytest.mark.parametrize("controller", ["visual_field", "direct", "gripper"])
@pytest.mark.parametrize("mode", ["transit", "separate", "interaction"])
def test_modes_are_explicit_metadata_for_all_controllers_and_preserve_raw_history(controller, mode):
    value = prediction(controller=controller, phase="grasp" if controller == "gripper" else "approach", mode=mode)
    original = copy.deepcopy(value)
    instance, transport = client(value, prediction())
    result = instance.predict(observation().to_dict())
    assert result["motion_context"] == value["motion_context"] and value == original
    assert result["subgoal_update"] == value["subgoal_update"]
    if controller == "direct": assert result["direct_actions"] == value["direct_actions"]
    instance.predict(observation(1).to_dict())
    raw = [json.loads(part["text"]) for message in transport.calls[-1]["input"]
           if message.get("role") == "assistant" for part in message["content"] if part["type"] == "output_text"]
    assert raw == [original]


@pytest.mark.parametrize("case", ["missing_mode", "missing_contact", "unknown_mode", "extra_permission",
    "transit_contact", "separate_contact", "interaction_null", "interaction_wrong", "interaction_blank",
    "forgotten_neighbor", "replace_blank_reason", "retain_blank_id", "field_finish", "direct_six",
    "field_release_xyz", "direct_release_translation"])
def test_new_metadata_never_relaxes_old_control_contract(case):
    value = prediction(controller="visual_field", present=True)
    context = value["motion_context"]
    if case == "missing_mode": context.pop("translation_mode")
    elif case == "missing_contact": context.pop("intended_contact_id")
    elif case == "unknown_mode": context["translation_mode"] = "ignore_obstacles"
    elif case == "extra_permission": context["clearance_certified"] = True
    elif case in ("transit_contact", "separate_contact"):
        context.update(translation_mode=case.split('_')[0], intended_contact_id=value["target_id"])
    elif case.startswith("interaction_"):
        context.update(translation_mode="interaction", intended_contact_id={
            "interaction_null":None, "interaction_wrong":"wrong_neighbor", "interaction_blank":" "}[case])
    elif case == "forgotten_neighbor": value["field_intent"]["obstacles"] = []
    elif case == "replace_blank_reason": value["subgoal_update"].update(action="replace", reason=" ")
    elif case == "retain_blank_id": value["subgoal_update"].update(action="retain", subgoal_id=" ")
    elif case == "field_finish": value["subgoal_update"]["action"] = "finish"
    elif case == "direct_six": value = prediction(); value["direct_actions"].append([0.]*7)
    elif case == "field_release_xyz": value["phase"] = "release"
    elif case == "direct_release_translation": value = prediction(phase="release"); value["direct_actions"][0][0] = .1
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_separate_does_not_silently_shrink_geometry_rewrite_goal_or_override_runtime_conflict():
    value = prediction(controller="visual_field", present=True, action="retain", mode="separate")
    value["motion_context"]["tool_external_roi_uv"] = [.45, .45, .65, .65]
    original = copy.deepcopy(value)
    instance, transport = client(value)
    context = {"visual_feedback":{"translation_guard":{"accepted":False,
        "reason":"projected_neighbor_conflict", "conflicting_neighbor_ids":["neighbor"]}}}
    instance.observable_context = context
    result = instance.predict(observation().to_dict())
    assert result["motion_context"] == original["motion_context"]
    assert result["field_intent"]["goal_offset_m"] == original["field_intent"]["goal_offset_m"]
    assert result["field_intent"]["obstacles"][0]["uncertainty_m"] == .005
    assert result["subgoal_update"] == original["subgoal_update"]
    supplied = current_record(transport.calls[0])["observable_context"]["visual_feedback"]
    assert supplied["translation_guard"] == context["visual_feedback"]["translation_guard"]
    assert instance.observable_context is context
    assert value == original and len(transport.calls) == 1  # Runtime screening is not a schema retry.


def test_stream_and_semantic_repairs_share_same_observation_and_existing_accounting(tmp_path):
    bad = prediction(mode="interaction"); bad["motion_context"]["intended_contact_id"] = "wrong"
    good = prediction()
    instance, transport = client(stream_error(), bad, good, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == good
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["unknown_usage_calls"] == 1 and instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.history_stats()["semantic_repair_requests"] == instance.history_stats()["stream_retry_requests"] == 1
    assert instance.history_stats()["observations"] == 1
    assert len(list(tmp_path.glob("*.request.json"))) == 3
    assert json.loads(current_record(transport.calls[-1])["observable_context"]["output_repair"]["rejected_model_output_json"]) == bad


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_v11_wire_delta_is_only_two_context_fields_and_guidance_is_generic(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    assert_unique_required(output_schema())
    stripped = output_schema(); context = stripped["properties"]["motion_context"]
    for key in ("translation_mode", "intended_contact_id"):
        context["properties"].pop(key); context["required"].remove(key)
    assert stripped == schema_v10()
    payload = transport.calls[-1]
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v11"
    assert payload["text"]["format"]["schema"] == output_schema()
    assert instance.prompt_version == VERSION+"-hybrid" and instance.history_stats()["protocol"] == HISTORY_VERSION
    assert next(v for v in input_texts(payload) if v["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort":"medium"}
    assert payload["stream"] and not payload["store"]
    text = shared_guidance(resolve_distance_task(task))
    for phrase in ("visual_feedback.translation_guard", "separate is not a bypass", "same line",
                   "z-only path cannot detour", "unsupported escape remains finite", "neighbor under the jaws",
                   "target-matched wrist observer", "Do not invent a centered", "accepted commands unchanged",
                   "SAME stored world goal", "retains FULL hard uncertainty inflation"):
        assert phrase in text


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_projection_guard_feedback_does_not_admit_privileged_inputs(name):
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback":{"translation_guard":{name:"not-policy-input"}}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls


def test_zero_shot_hybrid_only():
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV11(controller="direct")
    with pytest.raises(ValueError): VisualFieldSubscriptionClientV11(demonstration=object())
