"""Worker dispatch and evaluator isolation without a simulator or model."""
from types import SimpleNamespace

import pytest

from gpt_field import agent_speedup_v2_sim as generic_sim
from gpt_field import astranomy_agentic_geometry as geometry
from gpt_field import astranomy_agentic_sim as sim


def test_worker_selection_preserves_config_and_legacy_path(monkeypatch):
    calls = []
    generic, legacy = object(), object()

    def generic_factory(config):
        calls.append(("generic", config))
        return generic

    def legacy_factory(config):
        calls.append(("legacy", config))
        return legacy

    monkeypatch.setattr(generic_sim, "V2LiberoWorker", generic_factory)
    monkeypatch.setattr(sim, "_LiberoWorker", legacy_factory)
    new_config = sim.LiberoConfig(suite="libero_10", collision_check_mode="none",
                                task_bddl_override="custom.bddl", initial_state_override="state.npy")
    old_config = sim.LiberoConfig(suite="libero_spatial", collision_check_mode="task_manipulation_v1")
    assert sim._make_worker(new_config) is generic
    assert sim._make_worker(old_config) is legacy
    assert calls == [("generic", new_config), ("legacy", old_config)]
    assert calls[0][1] is new_config and calls[1][1] is old_config


def test_observations_geometry_and_truth_remain_separate(monkeypatch):
    reset_reply = {"observation": {"observation/state": [0., 1., 2., 3., 4., 5., 6., 7.],
                                  "observation/image": "unchanged reset RGB"},
                   "status": {"collision": {"mode": "generic_contact_v2"}}}
    step_reply = {"observation": {"observation/state": [8., 9., 10., 11., 12., 13., 14., 15.],
                                 "observation/image": "unchanged step RGB"},
                  "status": {"success": True}}
    truth_reply = {"raw_conjunction": True, "strict_conjunction": False,
                   "atoms": [{"why_not": ["not_held"]}]}
    calls = []
    worker = SimpleNamespace(env=object(), metadata={"controller": {"output_max": [.05] * 3},
                             "evaluator_ops": ["truth", "settle_truth", "oracle"]})

    def reset(task_id, episode_index):
        calls.append(("reset", task_id, episode_index))
        return reset_reply

    def step(action):
        calls.append(("step", action))
        return step_reply

    def truth():
        calls.append(("truth",))
        return truth_reply

    def calibration(env, *, image_size, observation_state):
        calls.append(("geometry", env, image_size, observation_state.copy()))
        return {"camera": "calibrated"}

    worker.reset, worker.step, worker.truth = reset, step, truth
    monkeypatch.setattr(geometry, "extract_geometry_metadata", calibration)
    config = SimpleNamespace(collision_check_mode="none", image_size=512)
    metadata, state = sim._dispatch(worker, {"op": "metadata"}, config, None)
    assert state is None and calls == []
    assert metadata["evaluator_ops"] == ["truth"] and metadata["evaluator_truth_available"] is True
    assert metadata["agentic_worker_backend"] == "V2LiberoWorker"
    assert metadata["controller"]["output_max"] == [.05] * 3
    assert worker.metadata["evaluator_ops"] == ["truth", "settle_truth", "oracle"]

    result, state = sim._dispatch(worker, {"op": "reset", "task_id": 8, "episode_index": 0}, config, state)
    assert result is reset_reply
    assert state == reset_reply["observation"]["observation/state"]
    assert state is not reset_reply["observation"]["observation/state"]
    action = [0., 0., .1, 0., 0., 0., -1.]
    result, state = sim._dispatch(worker, {"op": "step", "action": action}, config, state)
    assert result is step_reply and calls[-1] == ("step", action)
    result, state = sim._dispatch(worker, {"op": "geometry"}, config, state)
    assert result == {"camera": "calibrated"}
    assert calls[-1] == ("geometry", worker.env, 512, step_reply["observation"]["observation/state"])
    assert not any(call[0] == "truth" for call in calls)
    previous_state = state
    result, state = sim._dispatch(worker, {"op": "truth"}, config, state)
    assert result is truth_reply and state is previous_state
    assert calls[-1] == ("truth",)
    assert "atoms" not in step_reply["observation"] and "atoms" not in reset_reply["observation"]
    with pytest.raises(ValueError, match="unknown operation"):
        sim._dispatch(worker, {"op": "settle_truth"}, config, state)


def test_legacy_worker_does_not_gain_truth_or_require_generic_geometry(monkeypatch):
    worker = SimpleNamespace(env=None, metadata={"collision_check_mode": "task_manipulation_v1"})
    config = SimpleNamespace(collision_check_mode="task_manipulation_v1", image_size=512)
    metadata, state = sim._dispatch(worker, {"op": "metadata"}, config, None)
    assert metadata["collision_check_mode"] == "task_manipulation_v1"
    assert metadata["agentic_worker_backend"] == "_LiberoWorker"
    assert metadata["evaluator_ops"] == [] and metadata["evaluator_truth_available"] is False
    with pytest.raises(ValueError, match="truth is unavailable"):
        sim._dispatch(worker, {"op": "truth"}, config, state)
    with pytest.raises(ValueError, match="Reset is required"):
        sim._dispatch(worker, {"op": "geometry"}, config, state)
