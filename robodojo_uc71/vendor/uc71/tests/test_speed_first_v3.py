import copy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.speed_first_v3 import (ChatGPTPlanSpeedFirstV3Client, SpeedFirstV3Policy,
    FINE_GRID, speed_first_v3_schema, speed_first_v3_instructions)
from test_speed_first_v2 import prediction, operation, tick
from test_field_history import FakeTransport, envelope, observation
from test_speed_first_visual import observation as visual_observation, reference, PATH


def controller(*responses):
    transport = FakeTransport(*(envelope(raw, id=f"v3-{i}", model="gpt-6-astra",
                                        reasoning={"effort": "medium"})
                                for i, raw in enumerate(responses)))
    client = ChatGPTPlanSpeedFirstV3Client(token_provider=lambda: "offline-oauth-v3", transport=transport)
    return SpeedFirstV3Policy(client), transport


def test_fine_field_direct_and_gripper_contract_no_dense_values(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("V3 must not expand a dense field")
    monkeypatch.setattr("gpt_field.compact_subscription.expand_compact_field", forbidden)
    direct = prediction(direct=True, phase="grasp")
    policy, transport = controller(prediction(), direct, operation())
    obs, decision, _ = tick(policy, observation())
    assert policy.grid == FINE_GRID and policy.grid.count == 4173281
    assert decision.diagnostics["plan"]["search"]["method"] == "analytic_uniform_goal_distance"
    assert not decision.diagnostics["field_prediction"]["dense_values_materialized"]
    assert len(json.dumps(decision.diagnostics)) < 25000
    assert policy.pending_field_event["reason"] == "goal_reached"
    for expected in direct["direct"]["actions"]:
        obs, decision, _ = tick(policy, obs)
        assert decision.action == tuple(expected)
    for _ in range(15):
        obs, decision, _ = tick(policy, obs, fingers=(.015, -.015))
        assert decision.diagnostics["execution_kind"] == "local_gripper_settle"
    assert len(transport.calls) == 3
    assert policy.gripper_operation.done
    assert policy.client.history_stats()["feedback_records"] == 21
    assert all(p["model"] == "gpt-6-astra" and p["reasoning"] == {"effort": "medium"}
               for p in transport.calls)
    assert "0.005 m spacing" not in transport.calls[0]["instructions"]
    assert "161 cubed" in transport.calls[0]["instructions"]


def test_visual_fit_from_unchanged_direct_actions_drives_fine_field_and_reduces_error():
    direct = prediction(direct=True, phase="align")
    direct["control_context"]["visual_reference"] = reference()
    path = [(0., 0.), *PATH, PATH[-1]]
    direct["direct"]["actions"] = [[*(np.subtract(b, a)/.05), 0., 0., 0., 0., 1.]
                                    for a, b in zip(path[:-1], path[1:])]
    field = prediction(goal=(.01, .005, 0.), phase="align")
    field["field"]["gripper_command"] = "hold"
    field["control_context"]["visual_reference"] = reference(PATH[-1])
    policy, transport = controller(direct, field, prediction(direct=True))
    policy.adapter._gripper = 1.
    obs = replace(visual_observation(), prompt=policy.client.task_prompt)
    for xy, expected in zip(path[1:], direct["direct"]["actions"]):
        decision = policy.act(obs)
        assert decision.action == tuple(expected)
        after = replace(visual_observation(xy), prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert policy.local_feedback["calibration_valid"]
    previous_error = policy.local_feedback["pixel_error_norm"]
    # Use partial measured tracking response, with independently rendered RGB.
    for _ in range(3):
        decision = policy.act(obs)
        xy = np.asarray(obs.state[:2]) - [.1, .2] + .01*np.asarray(decision.action[:2])
        after = replace(visual_observation(xy), prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert len(transport.calls) == 2
    assert policy.local_goal_updates > 0
    assert policy.local_feedback["pixel_error_norm"] < previous_error
    assert policy.local_visual_total_s > 0 and policy.local_update_total_s > 0
    assert not policy.local_feedback["task_success_certified"]
    assert policy._field_cache.effective_compact_field["goal_offset_m"] != field["field"]["goal_offset_m"]
    assert policy.client.last_compact_prediction == field


def test_unavailable_reference_does_not_gate_normal_field():
    field = prediction(goal=(.06, 0., 0.), phase="align")
    field["control_context"]["visual_reference"] = reference()
    field["field"]["gripper_command"] = "close"
    policy, _ = controller(field)
    decision = policy.act(observation())
    assert np.linalg.norm(decision.action[:3]) > 0
    assert not decision.diagnostics["local_visual_feedback"]["calibration_valid"]
    assert decision.diagnostics["local_goal_updates"] == 0


@pytest.mark.parametrize("mutate", [
    lambda r: r["control_context"].update(visual_reference={"hidden_object_pose": [1,2,3]}),
    lambda r: r["control_context"].update(phase="grasp", visual_reference=reference()),
    lambda r: r["field"].update(goal_reference={"kind": "held_object"}),
    lambda r: r.update(task_complete=True),
])
def test_malformed_output_rejected_before_any_action(mutate):
    raw = prediction()
    mutate(raw)
    client = ChatGPTPlanSpeedFirstV3Client(token_provider=lambda: "offline")
    with pytest.raises(SchemaError):
        client.validate_response(raw, FINE_GRID)


def test_prompt_schema_and_quaternion_provenance():
    raw = prediction()
    raw["field"]["orientation_target_xyzw"] = [0., 0., 0., 1.003]
    client = ChatGPTPlanSpeedFirstV3Client(token_provider=lambda: "offline")
    result = client.validate_response(raw, FINE_GRID)
    assert result["field"]["orientation_target_xyzw"] == [0., 0., 0., 1.]
    assert client.last_compact_prediction == raw
    assert client.last_quaternion_normalization["raw_norm"] == 1.003
    assert "control_context contains phase and visual_reference=null" not in speed_first_v3_instructions()
    assert speed_first_v3_schema()["properties"]["control_context"]["properties"]["visual_reference"]["anyOf"]


def test_open_direct_action_invalidates_reference_before_finger_motion():
    raw = prediction(direct=True, phase="align")
    raw["control_context"]["visual_reference"] = reference()
    raw["direct"]["actions"] = [[0.] * 6 + [-1.]] * 5
    policy, _ = controller(raw)
    policy.adapter._gripper = 1.
    obs = replace(visual_observation(), prompt=policy.client.task_prompt)
    decision = policy.act(obs)
    assert decision.action == tuple(raw["direct"]["actions"][0])
    assert policy.local_feedback["reliable"]
    policy.observe_transition(obs, decision, obs)
    assert policy.local_feedback["status"] == "executed_open_command"
    assert not policy.local_feedback["calibration_valid"]


def test_rotating_field_keeps_model_motion_but_does_not_use_planar_mapping():
    from test_speed_first_visual import calibrated
    raw = prediction(goal=(.02, .01, 0.), phase="align")
    raw["field"].update(gripper_command="hold", orientation_target_xyzw=[0., 0., math.sin(.1), math.cos(.1)])
    raw["control_context"]["visual_reference"] = reference(PATH[-1])
    policy, _ = controller(raw)
    policy.adapter._gripper = 1.
    policy.visual, _ = calibrated()
    obs = replace(visual_observation(PATH[-1]), prompt=policy.client.task_prompt)
    decision = policy.act(obs)
    policy.observe_transition(obs, decision, obs)
    assert policy.local_feedback["status"] == "nominal_goal_changes_orientation"
    assert policy.local_goal_updates == 0
    assert np.linalg.norm(decision.action[:3]) > 0
