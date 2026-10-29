"""Runner plumbing tests use mock API responses and a mock simulator only."""

import importlib.util
import json
import os
from pathlib import Path
import signal

import numpy as np
import pytest

from gpt_field.astra import APIError
from gpt_field.libero import LiberoConfig
from gpt_field.observation import PolicyObservation, image_to_data_url


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_bowl_direct_history.py"
spec = importlib.util.spec_from_file_location("evaluate_bowl_direct_history", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def sim_config(**changes):
    return LiberoConfig(**{"suite": "libero_spatial", "seed": 7, "max_steps": None,
                          "libero_root": runner.DEFAULT_LIBERO_ROOT,
                          "python_executable": runner.DEFAULT_SIM_PYTHON, **changes})


@pytest.fixture
def observation():
    image = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    return PolicyObservation(image, image, (0, 0, 1, 0, 0, 0, 0.04, -0.04), runner.TASK)


class FakeRecorder:
    instances = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs, self.events = path, kwargs, []
        self.instances.append(self)
        path.write_bytes(b"MOCK VIDEO FOR UNIT TEST ONLY")

    def observe(self, obs, **kwargs):
        self.events.append(("observe", kwargs))

    def decision(self, obs, decision, **kwargs):
        self.events.append(("decision", kwargs))

    def executed(self, obs, **kwargs):
        self.events.append(("executed", kwargs))

    def close(self, **kwargs):
        self.events.append(("close", kwargs))


def fake_simulator(observation, *, success_at=None, fail_at=None, before_step=None):
    instances = []

    class FakeSim:
        def __init__(self, config):
            assert os.environ.get("OPENAI_API_KEY") is None
            assert config.max_steps is None
            self.config, self.metadata = config, {"unit_test_mock_simulator": True}
            self.status, self.actions, self.closed = {}, [], False
            self.obs = observation
            instances.append(self)

        def start(self):
            return self

        def list_tasks(self):
            return [{"task_id": 0, "prompt": runner.TASK}]

        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (0, 0)
            self.status = {"success": False, "terminated": False, "steps": 0,
                           "evaluator_only_marker": "MUST_NOT_ENTER_POLICY"}
            return self.obs

        def step(self, action):
            if before_step:
                before_step(len(self.actions), action)
            if fail_at is not None and len(self.actions) == fail_at:
                raise RuntimeError("simulator communication failed")
            self.actions.append(list(action))
            state = list(self.obs.state)
            state[:3] = [v + a * 0.05 for v, a in zip(state[:3], action[:3])]
            self.obs = PolicyObservation(self.obs.image, self.obs.wrist_image, tuple(state), self.obs.prompt)
            success = success_at is not None and len(self.actions) == success_at
            self.status = {**self.status, "steps": len(self.actions), "success": success, "terminated": success}
            return self.obs, success

        def close(self):
            self.closed = True

    return FakeSim, instances


def response(actions=None):
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"actions": actions or [[0, 0, 0, 0, 0, 0, 1]] * 5})}]}],
        "usage": {"input_tokens": 100, "output_tokens": 100}}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_preflight_is_read_only_and_never_checks_credentials_or_starts_sim(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not call API, simulator, or credentials check")

    monkeypatch.setenv("OPENAI_API_KEY", "SECRET_FOR_UNIT_TEST")
    monkeypatch.setattr(runner.DirectHistoryAstraClient, "preflight", forbidden)
    monkeypatch.setattr(runner, "SimProcess", forbidden)
    monkeypatch.setattr(runner, "urllib_transport", forbidden)
    output = tmp_path / "must_not_exist"
    result = runner.preflight(runner.configuration(), sim_config(), output)
    assert result["status"] == "preflight_only"
    assert result["synthetic_observation"] and result["files_written"] is False
    assert result["api_calls"] == 0 and result["simulator_started"] is False
    assert result["live_budget_required"] and not result["old_field_episode_budget_inherited"]
    assert result["payload"]["initial_image_count"] == 2
    assert not output.exists()
    assert "SECRET_FOR_UNIT_TEST" not in json.dumps(result)


def test_live_budget_is_new_explicit_and_no_action_or_call_or_wall_limit():
    cfg = runner.configuration(75)
    runner.validate_config(cfg, sim_config())
    assert cfg.max_cost_usd == 75
    assert cfg.max_api_calls is cfg.max_wall_time_s is None
    with pytest.raises(ValueError, match="own explicit"):
        runner.validate_config(runner.configuration(), sim_config())
    with pytest.raises(ValueError, match="Offline smoke"):
        runner.validate_config(cfg, sim_config(), offline_smoke=True)


@pytest.mark.parametrize("changes", [
    {"suite": "libero_10"}, {"seed": 8}, {"max_steps": 520}, {"settle_steps": 5},
    {"render_size": 224}, {"image_size": 256},
])
def test_runner_rejects_sim_protocol_drift(changes):
    with pytest.raises(ValueError, match="official libero_spatial"):
        runner.validate_config(runner.configuration(75), sim_config(**changes))


def test_cli_requires_own_live_budget_before_credentials_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.getpass, "getpass", lambda *a: pytest.fail("No credential prompt expected"))
    with pytest.raises(SystemExit) as caught:
        runner.main(["--output", str(tmp_path / "not_created")])
    assert caught.value.code == 2
    assert not (tmp_path / "not_created").exists()


def test_cli_preserves_venv_symlink_and_preflight_writes_nothing(tmp_path, monkeypatch):
    base = tmp_path / "base-python"
    base.touch()
    venv = tmp_path / "venv-python"
    venv.symlink_to(base)
    captured = {}

    def preview(config, simulation, output):
        captured.update(config=config, simulation=simulation, output=output)
        return {"preflight": True}

    monkeypatch.setattr(runner, "preflight", preview)
    assert runner.main(["--preflight", "--sim-python", str(venv)]) == 0
    assert captured["simulation"].python_executable == str(venv)
    assert captured["simulation"].max_steps is None
    assert captured["config"].max_cost_usd is None


def test_offline_smoke_executes_ten_actions_and_feedback_but_never_scores_success(tmp_path, observation, monkeypatch):
    output = tmp_path / "smoke"
    seen_flushes = []

    def before_step(step, action):
        # Intent must be durable before simulator execution begins.
        trace = rows(output / "trace.jsonl")
        assert trace[-1]["event"] == "action" and trace[-1]["step"] == step
        seen_flushes.append(step)

    simulator, instances = fake_simulator(observation, before_step=before_step)
    monkeypatch.setenv("OPENAI_API_KEY", "SECRET_FOR_UNIT_TEST")
    monkeypatch.setattr(runner, "urllib_transport", lambda *args: pytest.fail("Offline smoke used network"))
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=simulator, recorder_factory=FakeRecorder)
    assert result["status"] == "diagnostic_complete"
    assert result["steps"] == result["measured_feedback_count"] == 10
    assert result["validated_action_chunks"] == result["mock_response_calls"] == 2
    assert result["api_calls"] == 0 and result["budget"]["cost_usd_upper_bound"] == 0
    assert result["gripper_close_actions"] == result["gripper_open_actions"] == 5
    assert result["evaluation_completed"] is False and result["success_rate"] is None
    assert result["scored_episodes"] == 0 and result["libero_success"] is None
    assert instances[0].closed and len(instances[0].actions) == 10
    assert seen_flushes == list(range(10))
    video = FakeRecorder.instances[-1]
    assert video.kwargs["offline_smoke"] is True
    assert [event for event, _ in video.events].count("executed") == 10
    assert video.events[-1][0] == "close"
    trace = rows(output / "trace.jsonl")
    assert [row["event"] for row in trace].count("executed") == 10
    assert [row["event"] for row in trace].count("feedback") == 10
    requests = sorted((output / "api").glob("*.request.json"))
    assert len(requests) == 2
    second = json.loads(requests[1].read_text())
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(second)
    assert "SECRET_FOR_UNIT_TEST" not in "".join(p.read_text(errors="ignore") for p in output.rglob("*") if p.is_file())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["kind"] == "fresh_direct_history_baseline"
    assert manifest["configuration"]["evaluation_scored"] is False
    assert len(manifest["configuration"]["runner"]["sha256"]) == 64


def test_only_evaluator_success_completes_live_episode(tmp_path, observation):
    simulator, instances = fake_simulator(observation, success_at=2)
    result = runner.execute_episode(tmp_path / "live_mock", config=runner.configuration(75),
        sim_config=sim_config(), api_key="TEST_SECRET", sim_factory=simulator,
        recorder_factory=FakeRecorder, transport=lambda *args: response())
    assert result["status"] == "success" and result["steps"] == 2
    assert result["evaluation_completed"] and result["libero_success"] is True
    assert result["success_rate"] == 1.0 and result["gripper_close_actions"] == 2
    assert result["api_calls"] == 1
    assert instances[0].closed


def test_budget_exhaustion_is_unscored_and_dispatches_nothing(tmp_path, observation):
    simulator, instances = fake_simulator(observation)
    result = runner.execute_episode(tmp_path / "budget", config=runner.configuration(0.1),
        sim_config=sim_config(), api_key="TEST_SECRET", sim_factory=simulator,
        recorder_factory=FakeRecorder, transport=lambda *a: pytest.fail("Budget must block transport"))
    assert result["status"] == "budget_exceeded" and result["steps"] == 0
    assert result["api_calls"] == 0 and result["success_rate"] is None
    assert result["libero_success"] is None and not result["evaluation_completed"]
    assert instances[0].closed


def test_inflight_api_interrupt_finalizes_and_keeps_full_unknown_reservation(tmp_path, observation):
    simulator, instances = fake_simulator(observation)

    def interrupted(*args):
        raise KeyboardInterrupt

    output = tmp_path / "interrupted"
    result = runner.execute_episode(output, config=runner.configuration(75), sim_config=sim_config(),
        api_key="TEST_SECRET", sim_factory=simulator, recorder_factory=FakeRecorder, transport=interrupted)
    assert result["status"] == "user_stop" and result["success_rate"] is None
    assert result["api_calls"] == result["budget"]["unknown_usage_calls"] == 1
    assert result["budget"]["cost_usd_upper_bound"] == result["budget"]["reservation_per_call_usd"]
    assert instances[0].closed and FakeRecorder.instances[-1].events[-1][0] == "close"
    assert json.loads((output / "run_state.json").read_text())["status"] == "user_stop"


def test_transport_failure_redacts_error_and_retains_unknown_cost(tmp_path, observation):
    simulator, _ = fake_simulator(observation)
    secret = "TEST_SECRET"

    def denied(*args):
        raise APIError(f"Rejected {secret}", response_body=json.dumps({"error": secret}))

    output = tmp_path / "failed"
    result = runner.execute_episode(output, config=runner.configuration(75), sim_config=sim_config(),
        api_key=secret, sim_factory=simulator, recorder_factory=FakeRecorder, transport=denied)
    assert result["status"] == "infrastructure_error" and result["success_rate"] is None
    assert result["budget"]["unknown_usage_calls"] == 1
    assert secret not in "".join(p.read_text(errors="ignore") for p in output.rglob("*") if p.is_file())


def test_simulator_failure_keeps_pending_intent_and_does_not_claim_execution(tmp_path, observation):
    simulator, _ = fake_simulator(observation, fail_at=0)
    output = tmp_path / "sim_failed"
    result = runner.execute_episode(output, config=runner.configuration(75), sim_config=sim_config(),
        api_key="TEST_SECRET", sim_factory=simulator, recorder_factory=FakeRecorder,
        transport=lambda *args: response())
    assert result["status"] == "infrastructure_error" and result["steps"] == 0
    assert result["pending_action_index"] == 0
    assert not any(row["event"] == "executed" for row in rows(output / "trace.jsonl"))


def test_fresh_directory_guard_runs_before_any_api_or_simulator(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Fresh-directory guard must run first")

    monkeypatch.setattr(runner, "DirectHistoryAstraClient", forbidden)
    with pytest.raises(FileExistsError, match="Fresh baseline"):
        runner.execute_episode(tmp_path, config=runner.configuration(75), sim_config=sim_config(),
                               api_key="TEST_SECRET")


def test_sigterm_is_converted_to_interrupt_and_handler_restored():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt, match="signal"):
        with runner.termination_as_interrupt():
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is previous

