"""Persistent-reference, constrained spatial motion and gripper wire contracts."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v6 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV6, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required
from test_visual_field_v5_protocol import prediction as prediction_v5


def prediction(*, controller="direct", phase="approach", retain=False):
    value = prediction_v5(field=controller == "visual_field", phase=phase)
    old = value["visual_references"]["external"]
    value["visual_references"]["external"] = {
        "update": "retain" if retain else "seed", "moving_id": old["moving_id"],
        "moving_kind": old["moving_kind"], "target_id": old["target_id"],
        "goal_offset_uv": [a-b for a, b in zip(old["target_point_uv"], old["target_anchor_uv"])],
        "seed": None if retain else {name: copy.deepcopy(old[name]) for name in (
            "moving_point_uv", "moving_roi_uv", "target_anchor_uv", "target_roi_uv", "tool_point_uv", "tool_roi_uv")},
    }
    value["controller"] = controller
    value["gripper_intent"] = {"command": "close", "max_actions": 20} if controller == "gripper" else None
    if controller == "gripper": value["direct_actions"] = None
    if controller == "visual_field": value["field_intent"]["motion_axes"] = "xyz"
    return value


def client(*values, **kwargs):
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort": "medium"},
                                         id=f"v6_{i}") for i, value in enumerate(values)))
    return VisualFieldSubscriptionClientV6(token_provider=lambda: "offline-oauth-fixture",
                                           transport=transport, **kwargs), transport


def wrist(*, retain=False):
    return {"update": "retain" if retain else "seed", "target_id": "task_target",
            "desired_target_uv": [.5, .5], "seed": None if retain else {
                "target_anchor_uv": [.7, .6], "target_roi_uv": [.65, .55, .75, .65]}}


def test_seed_and_retain_are_explicit_without_mutating_or_inventing_coordinates():
    instance, _ = client()
    for retain in (False, True):
        value = prediction(controller="visual_field", retain=retain)
        value["visual_references"]["wrist"] = wrist(retain=retain)
        before = copy.deepcopy(value)
        assert instance.validate_response(value, None) == before and value == before
        if retain:
            assert value["visual_references"]["external"]["seed"] is None
    # The client has no live tracks: state-dependent retain resolution belongs
    # to the monitor, never model/transport validation or fabricated UVs.
    assert_unique_required(output_schema())


@pytest.mark.parametrize("case", ["retain_with_seed", "seed_without_seed", "wrist_retain_with_seed",
    "missing_goal_offset", "legacy_flat_point", "self_anchor", "wrong_wrist", "wrong_field_anchor",
    "payload_seed_without_tool", "bad_roi", "point_outside_roi", "two_branches", "gripper_wrong_phase",
    "gripper_float_budget", "gripper_bool_budget", "gripper_huge_budget", "gripper_tiny_budget",
    "gripper_hold", "xy_with_z_goal", "z_with_xy_goal", "release_xyz", "field_grasp", "direct_release_xyz",
    "missing_axis_mode", "legacy_completion", "six_direct"])
def test_invalid_v6_output_never_reaches_execution(case):
    kind = "gripper" if case.startswith("gripper_") else "visual_field"
    phase = "grasp" if kind == "gripper" else "approach"
    if case in ("direct_release_xyz", "six_direct"): kind = "direct"
    value = prediction(controller=kind, phase=phase)
    ext, intent = value["visual_references"]["external"], value["field_intent"]
    if case == "retain_with_seed": ext["update"] = "retain"
    elif case == "seed_without_seed": ext["seed"] = None
    elif case == "wrist_retain_with_seed":
        value["visual_references"]["wrist"] = wrist(); value["visual_references"]["wrist"]["update"] = "retain"
    elif case == "missing_goal_offset": ext.pop("goal_offset_uv")
    elif case == "legacy_flat_point": ext["moving_point_uv"] = [.2, .3]
    elif case == "self_anchor": ext["moving_id"] = ext["target_id"]
    elif case == "wrong_wrist":
        value["visual_references"]["wrist"] = wrist(); value["visual_references"]["wrist"]["target_id"] = "different"
    elif case == "wrong_field_anchor": ext["target_id"] = "different"
    elif case == "payload_seed_without_tool": ext["moving_kind"] = "payload"
    elif case == "bad_roi": ext["seed"]["moving_roi_uv"] = [.4, .4, .3, .5]
    elif case == "point_outside_roi": ext["seed"]["moving_point_uv"] = [.9, .9]
    elif case == "two_branches": value["gripper_intent"] = {"command": "open", "max_actions": 20}
    elif case == "gripper_wrong_phase": value["phase"] = "transport"
    elif case == "gripper_float_budget": value["gripper_intent"]["max_actions"] = 20.0
    elif case == "gripper_bool_budget": value["gripper_intent"]["max_actions"] = True
    elif case == "gripper_huge_budget": value["gripper_intent"]["max_actions"] = 31
    elif case == "gripper_tiny_budget": value["gripper_intent"]["max_actions"] = 9
    elif case == "gripper_hold": value["gripper_intent"]["command"] = "hold"
    elif case == "xy_with_z_goal": intent.update(motion_axes="xy", goal_offset_m=[.02, 0., .001])
    elif case == "z_with_xy_goal": intent.update(motion_axes="z", goal_offset_m=[.001, 0., -.02])
    elif case == "release_xyz": value["phase"] = "release"
    elif case == "field_grasp": value["phase"] = "grasp"
    elif case == "direct_release_xyz":
        value["phase"] = "release"; value["direct_actions"][0][0] = .1
    elif case == "missing_axis_mode": intent.pop("motion_axes")
    elif case == "legacy_completion": value["task_complete"] = True
    elif case == "six_direct": value["direct_actions"].append([0.]*7)
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_explicit_generic_placement_sequence_keeps_translation_in_field_and_release_model_selected():
    align = prediction(controller="visual_field", phase="align")
    align["field_intent"].update(motion_axes="xy", goal_offset_m=[.01, -.02, 0.])
    descend = prediction(controller="visual_field", phase="release", retain=True)
    descend["field_intent"].update(motion_axes="z", goal_offset_m=[0., 0., -.03])
    release = prediction(controller="gripper", phase="release", retain=True)
    release["gripper_intent"] = {"command": "open", "max_actions": 20}
    instance, requests = client(align, descend, release)
    outputs = [instance.predict(observation(i).to_dict()) for i in range(3)]
    assert len(requests.calls) == 3
    assert outputs[0]["field_intent"]["motion_axes"] == "xy"
    assert outputs[1]["field_intent"]["motion_axes"] == "z"
    assert outputs[1]["field_intent"]["gripper_command"] == "hold"
    assert outputs[2] == release
    assert outputs[1]["visual_references"]["external"]["seed"] is None


def test_direct_reorientation_during_release_retains_exact_five_zero_translation_actions():
    value = prediction(phase="release")
    value["direct_actions"] = [[0., 0., 0., .02*i, 0., 0., 0.] for i in range(5)]
    instance, transport = client(value)
    assert instance.predict(observation().to_dict()) == value and len(transport.calls) == 1


def test_gripper_and_direct_attachment_can_reference_separate_stationary_anchor():
    for controller in ("direct", "gripper"):
        value = prediction(controller=controller, phase="grasp")
        ext = value["visual_references"]["external"]
        ext.update(moving_id="task_target", moving_kind="payload", target_id="stationary_support")
        ext["seed"].update(tool_point_uv=[.2, .4], tool_roi_uv=[.15, .35, .25, .45])
        value["visual_references"]["wrist"] = wrist()
        instance, _ = client()
        assert instance.validate_response(value, None) == value


def test_semantic_repair_and_raw_persistent_reference_history_are_inherited(tmp_path):
    invalid = prediction(controller="visual_field", phase="release", retain=True)
    valid = copy.deepcopy(invalid)
    valid["field_intent"].update(motion_axes="z", goal_offset_m=[0., 0., -.02])
    instance, requests = client(invalid, valid, prediction(), log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert len(requests.calls) == 2 and instance.history_stats()["semantic_repair_requests"] == 1
    assert "orientation_target_xyzw" in result["field_intent"]
    assert result["visual_references"] == valid["visual_references"]
    instance.predict(observation(1).to_dict())
    prior_outputs = [json.loads(p["text"]) for m in requests.calls[-1]["input"] if m.get("role") == "assistant"
                     for p in m.get("content", []) if p["type"] == "output_text"]
    assert prior_outputs == [valid]
    assert len(list(tmp_path.glob("*.response.json"))) == 3


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_medium_subscription_history_and_explicit_constraints(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    payload = transport.calls[-1]
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["text"]["format"]["schema"] == output_schema()
    assert instance.prompt_version == VERSION+"-hybrid"
    records = input_texts(payload)
    assert next(x for x in records if x["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    text = shared_guidance(resolve_distance_task(task))
    assert len(text.split()) <= 1100
    assert "zero XYZ translation" in text and "Field align/xy" in text
    assert "Retain cannot resurrect a lost track" in text
    assert instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_new_contract_keeps_privileged_observations_out(name):
    instance, transport = client(prediction())
    obs = observation().to_dict(); obs[name] = "unavailable-private-data"
    with pytest.raises(ValueError): instance.predict(obs)
    instance.observable_context = {"visual_feedback": {name: "unavailable-private-data"}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls


def test_provider_failure_still_does_not_retry_or_fallback():
    instance, transport = client(prediction())
    transport.responses = [APIError("deliberate transport failure"), envelope(prediction())]
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.budget_snapshot()["api_key_calls"] == 0
