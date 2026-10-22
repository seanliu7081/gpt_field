"""Task-routing regressions using fake Codex and simulator, no inference."""
from dataclasses import replace
import json

import pytest

from . import policy, run
from .client import CodexPolicyClient, ProtocolError
from .libero10_tasks import LIBERO10_TASK_KEYS, RESET_SHA256
from .offline_check import fixture_request
from .test_policy import FakeRPC, FakeSim, runner
from gpt_field.distance_tasks import resolve_distance_task


@pytest.mark.parametrize("key", LIBERO10_TASK_KEYS)
def test_only_task_sentence_changes_in_policy(key):
    original = policy.policy_instructions("drawer_bowl")
    old = resolve_distance_task("drawer_bowl")
    spec = resolve_distance_task(key)
    assert policy.policy_instructions(key) == original.replace(old.prompt, spec.prompt)
    assert spec.suite == "libero_10" and spec.task_id == int(key[-1])
    config = run.sim_configuration(key)
    assert config.suite == "libero_10"
    assert config.seed == 7 and config.settle_steps == 10
    assert config.image_size == config.render_size == 512
    assert config.include_arm_joints and config.max_steps is None
    assert config.stop_on_success and not config.stop_on_collision
    assert config.collision_check_mode == "none"  # New worker owns the generic monitor.
    assert len(RESET_SHA256[(key, 0)]) == 64


@pytest.mark.parametrize("key", LIBERO10_TASK_KEYS)
def test_requested_task_and_high_reach_codex_thread(tmp_path, key):
    client = CodexPolicyClient(task_key=key, token_provider=lambda: "offline-fixture",
                              rpc_factory=FakeRPC, binary="fake", log_dir=tmp_path / key)
    try:
        client.predict(fixture_request())
        start = next(p for method, p in client.rpc.calls if method == "thread/start")
        turn = next(p for method, p in client.rpc.calls if method == "turn/start")
        assert start["baseInstructions"] == policy.policy_instructions(key)
        assert start["model"] == turn["model"] == "gpt-6-astra"
        assert start["config"]["model_reasoning_effort"] == turn["effort"] == "high"
        assert "Task: open the top drawer" not in start["baseInstructions"]
        assert client.validator.task_key == key
        assert len([i for i in turn["input"] if i["type"] == "image"]) == 2
    finally:
        client.close()


class TaskSim(FakeSim):
    def __init__(self, config, key):
        super().__init__(config)
        self.spec = resolve_distance_task(key)
        self.obs = replace(self.obs, prompt=self.spec.prompt)

    def reset(self, task_id, episode_index):
        assert task_id == self.spec.task_id and episode_index == 0
        self.status = {"terminated": False, "libero_success": False}
        return self.obs


@pytest.mark.parametrize("key", LIBERO10_TASK_KEYS)
def test_completed_run_records_actual_task_high_and_raw_success(tmp_path, runner, monkeypatch, key):
    sim = TaskSim(None, key)
    monkeypatch.setattr(runner, "expected_reset", lambda *args: runner.initial_identity(sim.obs))
    clients = []
    def factory(config, **kwargs):
        client = CodexPolicyClient(config, rpc_factory=FakeRPC, binary="fake", **kwargs)
        clients.append(client)
        return client
    output = tmp_path / key
    result = runner.execute(output, task_key=key, token_provider=lambda: "offline-fixture",
                            sim_factory=lambda config: sim, client_factory=factory, video=False)
    assert result["status"] == "success" and result["libero_success"]
    assert result["task_id"] == sim.spec.task_id and result["suite"] == "libero_10"
    assert result["reasoning_effort"] == "high" and result["steps"] == 2
    assert result["collision"]["detected"] and result["cleanup_errors"] == []
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["collision_monitor"] == "generic_contact_v2"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["runtime_harness"] == "persistent_codex_app_server_thread"
    assert json.loads((output / "initial_observation_check.json").read_text())["matched"]
    assert sim.closed and clients[0].rpc.closed


@pytest.mark.parametrize("key", LIBERO10_TASK_KEYS)
def test_reset_mismatch_prevents_model_and_motion(tmp_path, runner, monkeypatch, key):
    sim = TaskSim(None, key)
    monkeypatch.setattr(runner, "expected_reset", lambda *args: "0" * 64)
    class NoModel:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, request):
            pytest.fail("A mismatched reset must never contact the model")
        def budget_snapshot(self):
            return {"api_calls": 0, "api_key_calls": 0}
        def close(self):
            pass
    result = runner.execute(tmp_path / key, task_key=key, token_provider=lambda: "offline-fixture",
                            sim_factory=lambda config: sim, client_factory=NoModel, video=False)
    assert result["status"] == "infrastructure_error"
    assert result["api_calls"] == 0 and result["steps"] == 0 and not sim.actions


def test_unrequested_task_rejected_before_connection(tmp_path):
    with pytest.raises(ValueError):
        CodexPolicyClient(task_key="libero_10_t7", token_provider=lambda: "offline-fixture",
                          rpc_factory=FakeRPC, binary="fake", log_dir=tmp_path)
