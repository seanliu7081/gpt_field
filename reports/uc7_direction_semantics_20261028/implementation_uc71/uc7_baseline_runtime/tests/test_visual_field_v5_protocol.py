"""V5 semantic contracts; fake transport exercises the real OAuth/history path."""
import copy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from gpt_field.action import _rotvec_quaternion
from gpt_field.astra import APIError, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.visual_field_subscription_v5 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV5, output_schema,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v4_protocol import box, prediction as prediction_v4, wrist


def prediction(*, field=False, phase="approach", actions=None):
    result = prediction_v4(field=field, phase=phase, actions=actions)
    result["visual_references"]["external"].update(
        moving_roi_uv=[.25, .35, .35, .45], target_roi_uv=[.65, .55, .75, .65], tool_roi_uv=None)
    if field:
        result["field_intent"].pop("orientation_target_xyzw")
        result["field_intent"].update(orientation_command="hold", max_actions=80, command_limit_m=.015)
    return result


def client(*values, **kwargs):
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort": "medium"},
                                         id=f"v5_{index}") for index, value in enumerate(values)))
    return VisualFieldSubscriptionClientV5(token_provider=lambda: "offline-oauth-fixture",
                                           transport=transport, **kwargs), transport


def test_new_contract_requires_explicit_nullable_regions_and_hold_command():
    instance, _ = client()
    value = prediction(field=True)
    value["visual_references"]["wrist"] = dict(wrist(), target_roi_uv=[.6, .5, .8, .7])
    assert instance.validate_response(value, None) == value
    for ref in value["visual_references"].values():
        for name in list(ref):
            if name.endswith("roi_uv"):
                ref[name] = None
    assert instance.validate_response(value, None) == value
    assert_unique_required(output_schema())
    wire = json.dumps(output_schema())
    assert "orientation_command" in wire and "orientation_target_xyzw" not in wire
    assert "lower_offset_m" not in wire


@pytest.mark.parametrize("case", ["missing_roi", "reversed_roi", "zero_roi", "roi_outside_image",
    "point_outside_roi", "tool_roi_without_point", "wrist_anchor_outside_roi", "nan_roi", "bool_roi",
    "legacy_quaternion", "rotate_command", "close_command", "too_many_actions", "float_actions",
    "bool_actions", "too_much_travel", "too_large_command", "goal_outside_grid", "missing_external",
    "field_in_contact", "transport_without_payload", "transport_without_tool", "duplicate_box",
    "negative_extent", "empty_hypothesis", "direct_and_field", "six_direct", "complete_claim"])
def test_invalid_or_ambiguous_model_output_is_rejected_without_motion(case):
    value = prediction(field=case not in ("six_direct",))
    external = value["visual_references"]["external"]
    intent = value["field_intent"]
    if case == "missing_roi": external.pop("moving_roi_uv")
    elif case == "reversed_roi": external["moving_roi_uv"] = [.35, .35, .25, .45]
    elif case == "zero_roi": external["moving_roi_uv"] = [.3, .4, .3, .4]
    elif case == "roi_outside_image": external["moving_roi_uv"][0] = -.01
    elif case == "point_outside_roi": external["moving_point_uv"] = [.9, .9]
    elif case == "tool_roi_without_point": external["tool_roi_uv"] = [.2, .3, .3, .4]
    elif case == "wrist_anchor_outside_roi":
        value["visual_references"]["wrist"] = dict(wrist(), target_roi_uv=[.1, .1, .2, .2])
    elif case == "nan_roi": external["target_roi_uv"][0] = float("nan")
    elif case == "bool_roi": external["moving_roi_uv"][0] = False
    elif case == "legacy_quaternion":
        intent.pop("orientation_command"); intent["orientation_target_xyzw"] = [0, 0, 0, 1]
    elif case == "rotate_command": intent["orientation_command"] = "rotate"
    elif case == "close_command": intent["gripper_command"] = "close"
    elif case == "too_many_actions": intent["max_actions"] = 121
    elif case == "float_actions": intent["max_actions"] = 80.0
    elif case == "bool_actions": intent["max_actions"] = True
    elif case == "too_much_travel": intent["max_travel_m"] = .161
    elif case == "too_large_command": intent["command_limit_m"] = .021
    elif case == "goal_outside_grid": intent["goal_offset_m"][0] = .081
    elif case == "missing_external": value["visual_references"]["external"] = None
    elif case == "field_in_contact": value["phase"] = "grasp"
    elif case == "transport_without_payload": value["phase"] = "transport"
    elif case == "transport_without_tool":
        value["phase"] = "transport"; external["moving_kind"] = "payload"
    elif case == "duplicate_box":
        intent["obstacles"] = [box()]; intent["unknown_regions"] = [box()]
    elif case == "negative_extent":
        intent["obstacles"] = [box()]; intent["obstacles"][0]["half_extents_m"][0] = -.01
    elif case == "empty_hypothesis": value["hypothesis_id"] = " "
    elif case == "direct_and_field": value["direct_actions"] = [[0.] * 7] * 5
    elif case == "six_direct": value["direct_actions"].append([0.] * 7)
    elif case == "complete_claim": value["task_complete"] = True
    instance, transport = client()
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)
    assert not transport.calls


def test_payload_references_allow_independent_tool_roi_and_extended_budget():
    value = prediction(field=True, phase="transport")
    value["visual_references"]["external"].update(
        moving_kind="payload", tool_point_uv=[.2, .4], tool_roi_uv=[.15, .35, .25, .45])
    value["field_intent"]["max_actions"] = 120
    instance, _ = client()
    assert instance.validate_response(value, None) == value


@pytest.mark.parametrize("rotvec", [(0., 0., 0.), (math.pi-.0001, 0., 0.),
                                     (-math.pi+.0001, 0., 0.), (.7, -.4, 1.2)])
def test_hold_uses_current_measured_orientation_without_rewriting_wire_history(tmp_path, rotvec):
    value = prediction(field=True)
    value["field_intent"]["obstacles"] = [box()]
    untouched = copy.deepcopy(value)
    instance, transport = client(value, prediction(), log_dir=tmp_path)
    obs = observation()
    obs = replace(obs, state=(*obs.state[:3], *rotvec, *obs.state[6:]))
    result = instance.predict(obs.to_dict())
    intent = result["field_intent"]
    assert intent["orientation_target_xyzw"] == pytest.approx(_rotvec_quaternion(np.asarray(rotvec)))
    assert "orientation_command" not in intent
    assert intent["obstacles"][0]["lower_offset_m"] == pytest.approx([.01, .02, -.05])
    instance.predict(observation(1).to_dict())
    outputs = [json.loads(part["text"]) for message in transport.calls[1]["input"]
               if message.get("role") == "assistant" for part in message.get("content", [])
               if part.get("type") == "output_text"]
    assert outputs == [untouched]
    raw = json.loads(sorted(tmp_path.glob("*.response.json"))[0].read_text())
    logged = [json.loads(part["text"]) for message in raw["output"] if message.get("type") == "message"
              for part in message["content"] if part["type"] == "output_text"]
    assert logged == [untouched] and value == untouched


def test_direct_contact_actions_pass_through_with_no_orientation_conversion():
    actions = [[.05*i, -.1, 0., .03*i, 0., 0., 1.] for i in range(5)]
    value = prediction(phase="grasp", actions=actions)
    instance, _ = client(value)
    assert instance.predict(observation().to_dict()) == value


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_medium_oauth_history_and_task_general_handoff_protocol(task):
    instance, transport = client(*(prediction() for _ in range(5)), task_key=task)
    for index in range(5): instance.predict(observation(index, task=task).to_dict())
    payload = transport.calls[-1]
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["text"]["format"]["schema"] == output_schema()
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v5"
    assert instance.prompt_version == VERSION+"-hybrid"
    text = payload["instructions"]
    assert "ZERO-SHOT" in text and "switch to Direct5 terminal" in text
    assert "max_actions=80" in text and "orientation_command=hold" in text
    assert "model waiting does not advance physics" in text
    assert "V2 ANCHOR CONTRACT" not in text and "V4 HYBRID" not in text
    records = input_texts(payload)
    assert sum(record["type"] == "observation" for record in records) == 2
    assert next(record for record in records if record["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    assert instance.budget_snapshot()["billing"] == "chatgpt_subscription"
    assert instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("hidden", ["depth", "camera_calibration", "object_poses", "contacts", "libero_success"])
def test_privileged_observations_and_feedback_remain_rejected(hidden):
    instance, transport = client(prediction())
    obs = observation().to_dict(); obs[hidden] = "private"
    with pytest.raises(ValueError): instance.predict(obs)
    instance.observable_context = {"field_feedback": {hidden: "private"}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    with pytest.raises(ValueError): instance.record_block({"type": "visual_execution_block", hidden: "private"})
    assert not transport.calls


def test_authentication_failure_never_falls_back_to_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    def unavailable(): raise MissingCredentials("offline deliberate auth failure")
    instance, transport = client(prediction())
    instance.token_provider = unavailable
    with pytest.raises(MissingCredentials): instance.predict(observation().to_dict())
    assert not transport.calls and instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("provider_change", [{"model": "other-model"}, {"reasoning": {"effort": "high"}}])
def test_changed_provider_model_or_effort_stops_without_retry(provider_change):
    instance, transport = client(prediction())
    transport.responses[0].update(provider_change)
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1


@pytest.mark.parametrize("kwargs", [{"controller": "direct"}, {"demonstration": object()}])
def test_v5_scope_remains_zero_shot_hybrid(kwargs):
    with pytest.raises(ValueError, match="Hybrid only"):
        client(**kwargs)


def invalid_region_prediction():
    value = prediction(field=True)
    value["visual_references"]["external"]["moving_roi_uv"] = [.35, .35, .25, .45]
    return value


def test_one_semantic_repair_uses_same_observation_and_keeps_both_raw_results(tmp_path):
    invalid, valid = invalid_region_prediction(), prediction(phase="grasp")
    instance, transport = client(invalid, valid, prediction(), log_dir=tmp_path)
    base_context = {"refresh_reason": "external_reference_unavailable", "task_success_certified": False}
    instance.observable_context = base_context
    obs = observation()
    result = instance.predict(obs.to_dict())
    assert result == valid
    assert len(transport.calls) == 2 and instance.budget_snapshot()["api_calls"] == 2
    first = [item for item in input_texts(transport.calls[0]) if item["type"] == "observation"][-1]
    second = [item for item in input_texts(transport.calls[1]) if item["type"] == "observation"][-1]
    assert first["observation_index"] == second["observation_index"] == 0
    assert first["observation/state"] == second["observation/state"]
    for key in ("observation/joint_position", "observation/joint_velocity", "prompt"):
        assert first[key] == second[key]
    image_sets = [[part["image_url"] for message in payload["input"] for part in message.get("content", [])
                   if part["type"] == "input_image"] for payload in transport.calls]
    assert image_sets[0] == image_sets[1]
    repair = second["observable_context"]["output_repair"]
    assert repair["attempt"] == 1
    assert repair["validation_error"] == "Visual ROIs require ordered positive-area unit bounds"
    assert json.loads(repair["rejected_model_output_json"]) == invalid
    assert "NO actions" in repair["instruction"] and "untrusted" in repair["instruction"]
    assert instance.observable_context is base_context
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["semantic_repair_requests"] == 1
    raw_results = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.result.json"))]
    assert [item["status"] for item in raw_results] == ["error", "valid"]
    raw_responses = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.response.json"))]
    logged = [json.loads(part["text"]) for raw in raw_responses for message in raw["output"]
              if message.get("type") == "message" for part in message["content"]
              if part["type"] == "output_text"]
    assert logged == [invalid, valid]
    instance.predict(observation(1).to_dict())
    latest = [item for item in input_texts(transport.calls[-1]) if item["type"] == "observation"][-1]
    assert "output_repair" not in latest["observable_context"]


def test_second_invalid_output_is_terminal_without_third_request(tmp_path):
    instance, transport = client(invalid_region_prediction(), invalid_region_prediction(), prediction(),
                                 log_dir=tmp_path)
    original_context = {"refresh_reason": "new_intent"}
    instance.observable_context = original_context
    with pytest.raises(SchemaError, match="positive-area"):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 2 and instance.budget_snapshot()["api_calls"] == 2
    assert len(transport.responses) == 1
    assert instance.history_stats()["observations"] == 0
    assert instance.history_stats()["semantic_repair_requests"] == 1
    assert instance.observable_context is original_context and instance._v5_repair_record is None
    assert len(list(tmp_path.glob("*.response.json"))) == 2


@pytest.mark.parametrize("error", [APIError("deliberate provider failure"),
                                    BudgetExceeded("deliberate budget exhaustion"),
                                    SchemaError("untrusted provider body that was not locally validated")])
def test_transport_or_uncaptured_error_never_triggers_semantic_repair(error):
    instance, transport = client(prediction())
    transport.responses = [error, envelope(prediction())]
    with pytest.raises(type(error)):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.history_stats()["semantic_repair_requests"] == 0
    assert instance.observable_context == {}


def test_failed_repair_transport_does_not_retry_or_persist_repair_context():
    instance, transport = client(invalid_region_prediction())
    transport.responses.extend([APIError("repair request failed"), envelope(prediction())])
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 2 and len(transport.responses) == 1
    assert instance.observable_context == {} and instance._v5_repair_record is None


def test_extra_model_generated_key_is_quoted_for_repair_not_trusted_as_feedback():
    invalid = prediction()
    invalid["task_complete"] = True  # A rejected claim, never evaluator information.
    instance, transport = client(invalid, prediction())
    result = instance.predict(observation().to_dict())
    assert "task_complete" not in result
    record = [item for item in input_texts(transport.calls[1]) if item["type"] == "observation"][-1]
    repair = record["observable_context"]["output_repair"]
    assert isinstance(repair["rejected_model_output_json"], str)
    assert "task_complete" not in repair and json.loads(repair["rejected_model_output_json"]) == invalid


def real_grasp_reference_prediction(request):
    """Minimal replay of V5_001 rejected 18/22/28; model output only, no hidden state."""
    value = prediction(phase="grasp")
    value.update(target_id="black_bowl", hypothesis_id="observed_attachment_test",
                 reason="Hold closure and observe independent object/tool motion before transport.")
    value["visual_references"] = {
        "external": {"moving_id": "black_bowl", "moving_kind": "payload",
            "moving_point_uv": [.245, .584], "moving_roi_uv": [.19, .56, .31, .63],
            "target_id": "plate", "target_anchor_uv": [.218, .718],
            "target_point_uv": [.245, .56], "target_roi_uv": [.14, .665, .31, .79],
            "tool_point_uv": [.23, .495], "tool_roi_uv": [.09, .445, .35, .54]},
        "wrist": {"target_id": "black_bowl", "target_anchor_uv": [.51, .77],
            "desired_target_uv": [.51, .77], "target_roi_uv": [.26, .59, .75, .83]},
    }
    external = value["visual_references"]["external"]
    hold, lift = [0., 0., 0., 0., 0., 0., 1.], [0., 0., .1, 0., 0., 0., 1.]
    value["direct_actions"] = copy.deepcopy([hold, hold, lift, lift, lift])
    if request == 18:
        external.update(moving_point_uv=[.245, .585], moving_roi_uv=[.195, .56, .307, .625],
            target_anchor_uv=[.222, .721], target_point_uv=[.245, .55],
            target_roi_uv=[.143, .652, .317, .79], tool_point_uv=[.231, .497],
            tool_roi_uv=[.09, .455, .35, .545])
        value["visual_references"]["wrist"]["target_roi_uv"] = [.27, .59, .74, .8]
        value["direct_actions"] = copy.deepcopy([hold]*5)
    elif request == 22:
        external.update(moving_point_uv=[.245, .593], target_anchor_uv=[.225, .724],
            target_point_uv=[.245, .55], target_roi_uv=[.139, .652, .313, .791],
            tool_point_uv=[.23, .505], tool_roi_uv=[.09, .455, .35, .545])
        value["direct_actions"] = [[0., 0., .15, 0., 0., 0., 1.]]*5
    elif request != 28:
        raise ValueError("Only the three independent-anchor rejected outputs are represented")
    return value


@pytest.mark.parametrize("request_ordinal", [18, 22, 28])
def test_real_rejected_attachment_references_now_execute_without_role_rewrite_or_repair(request_ordinal):
    value = real_grasp_reference_prediction(request_ordinal)
    instance, transport = client(value)
    assert instance.predict(observation().to_dict()) == value
    assert len(transport.calls) == 1
    assert instance.history_stats()["semantic_repair_requests"] == 0
    assert value["target_id"] == value["visual_references"]["wrist"]["target_id"] == "black_bowl"
    assert value["visual_references"]["external"]["moving_kind"] == "payload"
    assert value["visual_references"]["external"]["target_id"] == "plate"


@pytest.mark.parametrize("case", ["wrong_wrist", "unrelated_field_anchor", "self_anchor", "no_tool"])
def test_contact_reference_exception_does_not_weaken_navigation_or_attachment_roles(case):
    value = real_grasp_reference_prediction(28)
    if case == "wrong_wrist":
        value["visual_references"]["wrist"]["target_id"] = "plate"
    elif case == "unrelated_field_anchor":
        value.update(controller="visual_field", phase="transport", direct_actions=None,
                     field_intent=prediction(field=True)["field_intent"])
    elif case == "self_anchor":
        # The actual second-invalid request 29 corrected the anchor to the
        # payload itself. This still cannot manufacture an independent track.
        ext = value["visual_references"]["external"]
        ext.update(target_id="black_bowl", target_anchor_uv=ext["moving_point_uv"],
                   target_roi_uv=ext["moving_roi_uv"])
    elif case == "no_tool":
        value["visual_references"]["external"].update(tool_point_uv=None, tool_roi_uv=None)
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_grasp_then_test_lift_then_field_transport_preserves_independent_payload_roles():
    grasp = real_grasp_reference_prediction(18)
    lift = real_grasp_reference_prediction(22)
    transport = prediction(field=True, phase="transport")
    transport.update(target_id="plate", visual_references=copy.deepcopy(lift["visual_references"]))
    transport["visual_references"]["wrist"] = None  # Destination not yet visible in this view.
    instance, requests = client(grasp, lift, transport)
    outputs = [instance.predict(observation(index).to_dict()) for index in range(3)]
    assert len(requests.calls) == 3 and instance.history_stats()["semantic_repair_requests"] == 0
    assert [item["controller"] for item in outputs] == ["direct", "direct", "visual_field"]
    assert [item["target_id"] for item in outputs] == ["black_bowl", "black_bowl", "plate"]
    for item in outputs:
        external = item["visual_references"]["external"]
        assert external["moving_id"] == "black_bowl" and external["moving_kind"] == "payload"
        assert external["target_id"] == "plate" and external["tool_point_uv"] is not None
    # The real local monitor, rather than phase labels/schema validity, must
    # still establish attachment from measured independent visual motion.
    assert all("attachment_evidence" not in item for item in outputs)
