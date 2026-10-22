"""Offline behavior checks for navigation/interaction, without model or sim I/O."""
import copy

import numpy as np
import pytest

from gpt_field.astra import AstraConfig, SchemaError
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.speed_first_hybrid import (
    ChatGPTPlanSpeedFirstClient, SpeedFirstHybridPolicy, speed_first_output_schema,
)
from test_distance_policy import intent, tick
from test_field_history import FakeTransport, envelope, observation


def controller(*responses, task_key="bowl_plate"):
    transport = FakeTransport(*(envelope(x, id=f"speed-{i}", reasoning={"effort": "medium"})
                                for i, x in enumerate(responses)))
    client = ChatGPTPlanSpeedFirstClient(token_provider=lambda: "offline-oauth", transport=transport,
                                         task_key=task_key)
    return SpeedFirstHybridPolicy(client), transport


def test_field_arrival_hands_to_five_unchanged_grasp_actions():
    direct = intent(direct=True, phase="grasp")
    direct["direct"]["actions"] = [[.01 * i, 0., -.02, 0., 0., 0., 1.] for i in range(5)]
    policy, transport = controller(intent(), direct, intent(goal=(.04, 0., 0.), phase="transport"))
    obs, move = tick(policy, observation())
    assert move.diagnostics["plan"]["search"]["method"] == "dijkstra_distance_field"
    assert policy.pending_field_event["reason"] == "goal_reached"
    for index in range(5):
        obs, decision = tick(policy, obs)
        assert decision.action == tuple(direct["direct"]["actions"][index])
        assert decision.diagnostics["refreshed"] is (index == 0)
        assert not decision.diagnostics["task_success_certified"]
    assert len(transport.calls) == 2
    _, decision = tick(policy, obs)
    assert decision.diagnostics["controller"] == "coarse_field"
    assert len(transport.calls) == 3


def test_field_runs_thirty_actions_without_visual_training_then_refreshes():
    policy, transport = controller(intent(goal=(.075, 0., 0.)), intent(direct=True, phase="grasp"))
    obs = observation()
    for index in range(30):
        obs, decision = tick(policy, obs, fraction=.025)
        assert decision.diagnostics["refreshed"] is (index == 0)
        assert decision.action[0] > 0
    assert len(transport.calls) == 1
    assert policy.pending_field_event["reason"] == "field_budget_exhausted"
    _, decision = tick(policy, obs)
    assert decision.diagnostics["controller"] == "contact_direct"
    assert len(transport.calls) == 2
    assert policy.local_visual_total_s == 0


def test_no_motion_requests_model_recovery_without_a_local_lock():
    recovery = intent(direct=True, phase="recover")
    recovery["direct"]["actions"] = [[0., .05, .03, 0., 0., 0., -1.]] * 5
    policy, _ = controller(intent(goal=(.06, 0., 0.)), recovery)
    obs = observation()
    for _ in range(5):
        obs, _ = tick(policy, obs, fraction=0.)
    assert policy.pending_field_event["reason"] == "no_progress"
    for action in recovery["direct"]["actions"]:
        obs, decision = tick(policy, obs)
        assert decision.action == tuple(action)


def test_hold_does_not_trigger_automatic_upward_view_recovery_or_completion():
    policy, transport = controller(intent(phase="verify", hold=True), intent(direct=True, phase="recover"))
    obs = observation()
    for _ in range(5):
        obs, decision = tick(policy, obs)
        assert decision.action[:3] == (0., 0., 0.)
    assert policy.pending_field_event["reason"] == "field_budget_exhausted"
    assert policy._completion is None
    _, decision = tick(policy, obs)
    assert decision.diagnostics["controller"] == "contact_direct"
    assert len(transport.calls) == 2


@pytest.mark.parametrize("goal", [(.3, 0., 0.), (.04, 0., 0.)])
def test_invalid_field_refreshes_without_disabling_direct(goal):
    raw = intent(goal=goal)
    if goal[0] < .1:
        raw["field"]["blocked_boxes"] = [{"lower_offset_m": [-.01] * 3,
                                          "upper_offset_m": [.01] * 3}]
    recovery = intent(direct=True, phase="recover")
    policy, _ = controller(raw, recovery)
    obs, decision = tick(policy, observation())
    assert decision.action[:3] == (0., 0., 0.)
    assert policy.pending_field_event["reason"] in ("field_boundary_reached", "invalid_direction")
    _, decision = tick(policy, obs)
    assert decision.action == tuple(recovery["direct"]["actions"][0])


def test_payload_is_zero_shot_medium_subscription_with_no_visual_goal_contract():
    policy, _ = controller()
    payload = policy.client.build_payload(observation().to_dict())
    assert payload["model"] == "gpt-6-astra"
    assert payload["reasoning"]["effort"] == "medium"
    assert "ZERO-SHOT" in payload["instructions"]
    assert "collisions are recorded separately" in payload["instructions"]
    assert "exactly five" in payload["instructions"]
    assert "30 actions" in payload["instructions"]
    assert payload["text"]["format"]["schema"] == speed_first_output_schema()
    schema = payload["text"]["format"]["schema"]
    assert schema["properties"]["control_context"]["properties"]["visual_reference"] == {"type": "null"}
    assert "task_complete" not in schema["properties"]
    assert policy.client.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("kind", ["visual", "goal"])
def test_unimplemented_visual_goals_are_rejected_not_silently_ignored(kind):
    raw = intent()
    if kind == "visual":
        raw["control_context"]["visual_reference"] = {"anything": "unsupported"}
    else:
        raw["field"]["goal_reference"] = {"anything": "unsupported"}
    policy, _ = controller(raw)
    with pytest.raises(SchemaError, match="must be null"):
        policy.act(observation())


@pytest.mark.parametrize("kwargs", [
    {"reasoning_effort": "high"}, {"stop_on_model_complete": True},
    {"chunk_size": 30}, {"demonstration": object()}, {"zero_shot_collision_free": True},
    {"planner_horizon": 3}, {"field_execution_steps": 1},
    {"config": AstraConfig(model="different-model")},
])
def test_profile_rejects_changes_to_experiment_contract(kwargs):
    with pytest.raises((ValueError, TypeError)):
        ChatGPTPlanSpeedFirstClient(token_provider=lambda: "offline", **kwargs)


def test_drawer_uses_same_algorithm_without_fixed_pose_or_automatic_lift():
    policy, _ = controller(task_key="drawer_bowl")
    from gpt_field.distance_tasks import resolve_distance_task
    from dataclasses import replace
    obs = replace(observation(), prompt=resolve_distance_task("drawer_bowl").prompt)
    payload = policy.client.build_payload(obs.to_dict())
    assert "drawer_handle" in payload["instructions"]
    assert "not a lifting test" in payload["instructions"]
    assert policy.grid == HYBRID_GRID
    assert policy.field_execution_steps == policy.planner.config.horizon == 10


def test_integer_and_float_quaternion_arrival_have_same_actions():
    raw = intent()
    raw["field"]["orientation_target_xyzw"] = [0, 0, 0, 1]
    floating = copy.deepcopy(raw)
    floating["field"]["orientation_target_xyzw"] = [0., 0., 0., 1.]
    left, _ = controller(raw)
    right, _ = controller(floating)
    _, a = tick(left, observation())
    _, b = tick(right, observation())
    np.testing.assert_array_equal(a.action, b.action)
    assert left.pending_field_event == right.pending_field_event
