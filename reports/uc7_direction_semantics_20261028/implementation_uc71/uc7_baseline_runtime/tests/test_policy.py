"""Policy orchestration checks with deterministic predictions and no API calls."""

from copy import deepcopy

import numpy as np
import pytest

from gpt_field.observation import OBSERVATION_KEYS, PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.policy import FieldPolicy
from gpt_field.schema import GridSpec


IMAGE = image_to_data_url(np.zeros((4, 4, 3), dtype=np.uint8), size=4)
GRID = GridSpec(size=5, spacing_m=0.05)


def observation(x=0.0):
    return PolicyObservation(IMAGE, IMAGE, (x, 0, 0.5, 0, 0, 0, 0.04, -0.04), "move toward the mug")


def prediction(stage="approach", gripper="hold", motion="move"):
    coordinates = GRID.coordinates((0, 0, 0))
    return {"stage": stage, "orientation_target_xyzw": [0, 0, 0, 1],
            "gripper_command": gripper, "motion_mode": motion,
            "values": (-coordinates[:, 0] + coordinates[:, 1] ** 2).tolist(),
            "blocked_indices": [0], "unknown_indices": [GRID.count - 1]}


class FakePredictor:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, observation, grid, mode="field"):
        assert set(observation) == OBSERVATION_KEYS
        self.calls.append({"observation": deepcopy(observation), "grid": grid, "mode": mode})
        return deepcopy(self.predictions[min(len(self.calls) - 1, len(self.predictions) - 1)])


def test_five_step_refresh_keeps_world_center_frozen_and_uses_measured_state():
    client = FakePredictor([prediction(), prediction(stage="grasp", gripper="close")])
    policy = FieldPolicy(client, grid=GRID, planner_config=PlannerConfig(horizon=1))
    for step in range(5):
        measured_x = step * 0.003
        decision = policy.act(observation(measured_x))
        assert decision.diagnostics["refreshed"] is (step == 0)
        assert policy.field.center == (0.0, 0.0, 0.5)
        assert decision.diagnostics["field_center"] == [0.0, 0.0, 0.5]
        # Replanning uses each measured position, not the cached grid center.
        candidates = decision.diagnostics["plan"]["candidates"]
        assert candidates[0]["positions"][0] == [measured_x, 0.0, 0.5]
    assert len(client.calls) == 1
    refreshed = policy.act(observation(0.015))
    assert len(client.calls) == 2
    assert refreshed.diagnostics["refreshed"]
    assert refreshed.diagnostics["stage_changed"]
    assert policy.field.center == (0.015, 0.0, 0.5)
    assert refreshed.action[-1] == 1.0
    assert client.calls[-1]["observation"]["observation/state"][0] == 0.015


@pytest.mark.parametrize("method", ["field", "field_one", "affine", "shuffled", "minimum"])
def test_field_ablations_preserve_auxiliary_outputs_and_masks(method):
    raw = prediction(stage="close_gripper", gripper="close", motion="hold")
    raw["orientation_target_xyzw"] = [0.0, 0.0, np.sin(0.01), np.cos(0.01)]
    client = FakePredictor([raw])
    policy = FieldPolicy(client, method=method, grid=GRID)
    decision = policy.act(observation())
    converted = policy.field.prediction.to_dict()
    for key in ("stage", "orientation_target_xyzw", "gripper_command", "motion_mode", "blocked_indices", "unknown_indices"):
        assert converted[key] == raw[key]
    assert client.calls[0]["mode"] == "field"
    assert decision.action[:3] == (0.0, 0.0, 0.0)
    assert decision.action[-1] == 1.0


def test_direct_baseline_executes_exactly_five_cached_actions():
    first = [[i / 10, 0, 0, 0, 0, 0, -1] for i in range(5)]
    second = [[-i / 10, 0, 0, 0, 0, 0, 1] for i in range(5)]
    client = FakePredictor([{"actions": first}, {"actions": second}])
    policy = FieldPolicy(client, method="direct", grid=GRID)
    for i, action in enumerate(first + second):
        decision = policy.act(observation(i * 0.001))
        assert decision.action == tuple(action)
        assert decision.diagnostics["refreshed"] is (i % 5 == 0)
        assert len(client.calls) == i // 5 + 1
    assert [call["mode"] for call in client.calls] == ["direct", "direct"]
    with pytest.raises(ValueError, match="five-action"):
        FieldPolicy(client, method="direct", update_steps=4)


def test_policy_revalidates_observation_allowlist_before_client_call():
    class LeakyObservation(PolicyObservation):
        def to_dict(self):
            data = super().to_dict()
            data["object_pose"] = [1, 2, 3]
            return data

    client = FakePredictor([prediction()])
    policy = FieldPolicy(client, grid=GRID)
    clean = observation()
    leaky = LeakyObservation(clean.image, clean.wrist_image, clean.state, clean.prompt)
    with pytest.raises(ValueError, match="four allowlisted"):
        policy.act(leaky)
    assert not client.calls


def test_reset_clears_cached_prediction_stage_and_gripper_command():
    client = FakePredictor([prediction(stage="grasp", gripper="close", motion="hold"),
                            prediction(stage="approach", gripper="hold", motion="hold")])
    policy = FieldPolicy(client, grid=GRID)
    assert policy.act(observation()).action[-1] == 1
    policy.reset()
    assert policy.field is None and policy.prediction is None and policy.last_stage is None
    assert policy.step_index == 0
    decision = policy.act(observation())
    assert len(client.calls) == 2
    assert decision.action[-1] == -1
    assert decision.diagnostics["stage_changed"] is False


def test_waypoint_prediction_uses_its_own_mode_and_supplied_auxiliary_commands():
    raw = prediction(gripper="close", motion="hold")
    del raw["values"]
    raw["waypoint_xyz"] = [0.3, 0, 0.5]
    client = FakePredictor([raw])
    policy = FieldPolicy(client, grid=GRID, method="waypoint")
    decision = policy.act(observation())
    assert client.calls[0]["mode"] == "waypoint"
    assert decision.action[-1] == 1
    assert policy.field.prediction.blocked_indices == (0,)
    assert policy.field.prediction.unknown_indices == (GRID.count - 1,)
