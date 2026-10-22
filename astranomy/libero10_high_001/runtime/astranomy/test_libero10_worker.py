"""Offline adapter and monitor regressions: no environment or model is started."""
import copy
from dataclasses import replace
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.libero import LiberoConfig
from . import libero10_worker as adapter


def configuration():
    return LiberoConfig(suite="libero_10", image_size=512, render_size=512,
                        max_steps=None, stop_on_success=True, stop_on_collision=False,
                        collision_check_mode="none", include_arm_joints=True)


class FakeWorker:
    instances = []

    def __init__(self, config, monitor_options):
        self.config, self.options = config, monitor_options
        self.env = None
        self.closed = False
        self.resets, self.actions = [], []
        self.metadata = {"evaluator_ops": ["truth", "settle_truth", "oracle"],
                         "oracle_enabled": False, "collision_policy": "record_only"}
        self.state = [0., 0., 1., 0., 0., 0., .04, -.04]
        self.instances.append(self)

    def tasks(self):
        return [{"task_id": 2, "prompt": "official language"}]

    def reset(self, task_id, episode_index):
        self.resets.append((task_id, episode_index))
        self.env = object()
        return self.response()

    def step(self, action):
        self.actions.append(action)
        self.state[0] += .01
        return self.response()

    def response(self):
        return {"observation": {"observation/state": list(self.state),
                                "observation/image": "paired-rgb"},
                "status": {"libero_success": False, "success": False,
                           "terminated": False, "collision": {"detected": True}}}

    def geometry(self):
        pytest.fail("V2 privileged geometry must never be exposed")

    def close(self):
        self.closed = True


def exchange(messages, *, factory=FakeWorker, config=None):
    protocol = io.StringIO()
    adapter.serve(config or configuration(),
                  [json.dumps(message) + "\n" for message in messages],
                  protocol, worker_factory=factory)
    return [json.loads(line) for line in protocol.getvalue().splitlines()]


def test_protocol_uses_cached_observation_for_geometry_and_records_contacts(monkeypatch):
    geometry_states = []

    def extract(env, image_size, observation_state):
        assert env is FakeWorker.instances[-1].env and image_size == 512
        geometry_states.append(list(observation_state))
        return {"sensor_geometry": list(observation_state)}

    monkeypatch.setattr(adapter, "extract_geometry_metadata", extract)
    action = [0.] * 6 + [-1.]
    replies = exchange([{"op": "metadata"}, {"op": "tasks"},
                        {"op": "reset", "task_id": 6, "episode_index": 0},
                        {"op": "geometry"}, {"op": "step", "action": action},
                        {"op": "geometry"}])
    worker = FakeWorker.instances[-1]
    assert all(reply["ok"] for reply in replies)
    assert worker.options == {"goal_aware": True, "truth_window": False}
    assert worker.resets == [(6, 0)] and worker.actions == [action] and worker.closed
    assert replies[0]["result"]["evaluator_ops"] == []
    assert replies[0]["result"]["oracle_enabled"] is False
    assert geometry_states[0][0] == 0. and geometry_states[1][0] == .01
    # Status contacts remain evaluator-only and cannot terminate a non-success.
    for index in (2, 4):
        result = replies[index]["result"]
        assert result["status"]["collision"]["detected"]
        assert not result["status"]["terminated"]
        assert set(result["observation"]) == {"observation/state", "observation/image"}


@pytest.mark.parametrize("operation", ["truth", "settle_truth", "answer_key", "oracle", "close"])
def test_privileged_or_unknown_ops_rejected_without_constructing_worker(operation):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported operations must not initialize a simulator")
    reply, = exchange([{"op": operation}], factory=forbidden)
    assert not reply["ok"] and "unsupported operation" in reply["error"]


@pytest.mark.parametrize("operation", ["geometry", "step"])
def test_reset_required_before_sensor_or_action(operation):
    reply, = exchange([{"op": operation, "action": [0.] * 7}])
    worker = FakeWorker.instances[-1]
    assert not reply["ok"] and "Reset is required" in reply["error"]
    assert not worker.actions and worker.closed


@pytest.mark.parametrize("task_id,episode_index", [(0, 0), (2, 1), (True, 0), (2, False)])
def test_reset_rejects_unapproved_task_or_state(task_id, episode_index):
    reply, = exchange([{"op": "reset", "task_id": task_id, "episode_index": episode_index}])
    assert not reply["ok"]
    assert FakeWorker.instances[-1].resets == []


@pytest.mark.parametrize("change", [
    {"suite": "libero_goal"}, {"max_steps": 520}, {"stop_on_collision": True},
    {"stop_on_success": False}, {"include_arm_joints": False},
    {"initial_state_override": "/tmp/not-an-official-state.npy"},
])
def test_config_fails_before_worker_construction(change):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration must not initialize a simulator")
    with pytest.raises(ValueError):
        exchange([{"op": "metadata"}], factory=forbidden,
                 config=replace(configuration(), **change))


def test_v2_contact_records_do_not_change_base_success_or_termination(monkeypatch):
    """Exercise the real V2 reset/step methods around a synthetic base worker."""
    from gpt_field import agent_speedup_v2_sim as v2
    from gpt_field import agent_speedup_v2_truth as truth
    events = []

    class Monitor:
        def __init__(self, env, *, goal_aware):
            assert goal_aware is True

        def inspect(self, **kwargs):
            events.append("inspect")

        def attach(self):
            events.append("attach")

        def begin_action(self, index):
            events.append(("begin", index))

        def finish_action(self, *, refresh):
            assert refresh is True
            events.append("finish")

        def snapshot(self):
            return {"detected": True}

    def base_init(self, config):
        self.config, self.metadata = config, {}
        self._collision_monitor = None
        self.status = {}
        self.raw = False
        self.env = SimpleNamespace(sim=object(), check_success=lambda: self.raw)

    def base_reset(self, task_id, episode_index):
        self.status = {"steps": 0, "success": False, "terminated": False,
                       "horizon_reached": False}
        return {"observation": {"unchanged": True}, "status": self.status.copy()}

    def base_step(self, action):
        events.append("base_step")
        self.status.update(steps=self.status["steps"] + 1,
                           success=self.raw, terminated=self.raw)
        return {"observation": {"unchanged": True}, "status": self.status.copy()}

    monkeypatch.setattr(v2._LiberoWorker, "__init__", base_init)
    monkeypatch.setattr(v2._LiberoWorker, "reset", base_reset)
    monkeypatch.setattr(v2._LiberoWorker, "step", base_step)
    monkeypatch.setattr(v2, "GenericContactMonitor", Monitor)
    monkeypatch.setattr(v2, "forward_keeping_warmstart", lambda sim: events.append("forward"))
    monkeypatch.setattr(truth, "Scene", lambda env: object())
    monkeypatch.setattr(truth, "Window", lambda: pytest.fail("truth window must remain disabled"))
    worker = v2.V2LiberoWorker(configuration(), monitor_options=adapter.MONITOR_OPTIONS)
    reset = worker.reset(2, 0)
    assert reset["status"]["collision"]["detected"]
    assert reset["status"]["terminated"] is False and worker.window is None
    assert worker.oracle_enabled is False
    result = worker.step([0.] * 7)
    assert result["observation"] == {"unchanged": True}
    assert not result["status"]["success"] and not result["status"]["terminated"]
    worker.raw = True
    result = worker.step([0.] * 7)
    assert result["status"]["libero_success"] and result["status"]["success"]
    assert result["status"]["terminated"] and not result["status"]["collision_free_success"]
    assert events.count("base_step") == 2


def test_contact_endpoint_refresh_restores_warmstart():
    from gpt_field.agent_speedup_v2_contacts import forward_keeping_warmstart
    state = SimpleNamespace(qacc_warmstart=np.array([.3, -.2]), qpos=np.array([1., 2.]),
                            qvel=np.array([.4, .5]), time=1.)
    before = copy.deepcopy(state)

    def forward():
        state.qacc_warmstart[:] = 99.

    forward_keeping_warmstart(SimpleNamespace(data=state, forward=forward))
    np.testing.assert_array_equal(state.qacc_warmstart, before.qacc_warmstart)
    np.testing.assert_array_equal(state.qpos, before.qpos)
    np.testing.assert_array_equal(state.qvel, before.qvel)
    assert state.time == before.time
