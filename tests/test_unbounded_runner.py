import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_field.astra import APIError, AstraConfig
from gpt_field.libero import LiberoConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_bowl_until_success.py"
spec = importlib.util.spec_from_file_location("evaluate_bowl_until_success", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def configuration(**changes):
    values = json.loads((ROOT / "configs/astra_unbounded_episode.json").read_text())
    return AstraConfig(**{**values, **changes})


def record(output, *, status="success", steps=221):
    output.mkdir(parents=True, exist_ok=True)
    row = {"suite": "libero_spatial", "task_id": 0, "episode_index": 0,
           "status": status, "steps": steps, "video": "episode.mp4"}
    (output / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    return row


def test_configuration_has_no_action_call_or_time_limit_but_retains_cost_budget():
    config = configuration()
    runner.validate_config(config)
    assert config.max_api_calls is config.max_wall_time_s is None
    assert config.max_cost_usd == 50
    assert config.timeout_s == 180 and config.max_output_tokens == 8192
    assert (config.input_price_per_million, config.cached_input_price_per_million,
            config.cache_write_price_per_million, config.output_price_per_million) == (10, 1, 12.5, 50)
    assert runner.reference_metadata(config)["episode_limits"] == {
        "max_steps": None, "max_api_calls": None, "max_wall_time_s": None, "max_cost_usd": 50,
    }


@pytest.mark.parametrize("name,value", [
    ("max_api_calls", 88), ("max_wall_time_s", 18000),
    ("timeout_s", 60), ("max_output_tokens", 4096), ("output_price_per_million", 0),
])
def test_unbounded_runner_rejects_changed_protocol(name, value):
    with pytest.raises(ValueError):
        runner.validate_config(configuration(**{name: value}))


@pytest.mark.parametrize("cost", [None, 0, -1, float("inf"), float("nan"), True])
def test_cost_budget_must_remain_finite_and_positive(cost):
    with pytest.raises(ValueError):
        runner.validate_config(configuration(max_cost_usd=cost))


def test_reference_metadata_records_configured_cost_budget():
    config = configuration(max_cost_usd=25)
    runner.validate_config(config)
    assert runner.reference_metadata(config)["episode_limits"]["max_cost_usd"] == 25
    assert "API budget reached" in runner.reference_metadata(config)["stopping_rule"]


def test_transport_redacts_error_message_and_response_body():
    def denied(*args):
        raise APIError("Rejected SECRET", response_body='{"error":"SECRET"}')

    with pytest.raises(APIError) as caught:
        runner.redacting_transport("SECRET", denied)("fake", {}, {}, 180)
    assert "SECRET" not in str(caught.value)
    assert "SECRET" not in caught.value.response_body
    assert caught.value.__suppress_context__


def test_transport_redacts_success_and_unexpected_exception():
    response = {"SECRET": {"nested": ["SECRET", 12]}}
    send = runner.redacting_transport("SECRET", lambda *args: response)
    assert send("fake", {}, {}, 180) == {"[REDACTED]": {"nested": ["[REDACTED]", 12]}}

    def unexpected(*args):
        raise RuntimeError("SECRET in transport")

    with pytest.raises(APIError, match="Transport RuntimeError") as caught:
        runner.redacting_transport("SECRET", unexpected)("fake", {}, {}, 180)
    assert "SECRET" not in str(caught.value)


def test_prompt_guard_rejects_before_policy_or_api(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.FieldPolicy, "act", lambda self, obs: calls.append(obs.prompt))
    policy = runner.BowlFieldPolicy(object())
    with pytest.raises(ValueError, match="Task text"):
        policy.act(SimpleNamespace(prompt="pick a different bowl"))
    assert calls == []
    policy.act(SimpleNamespace(prompt=runner.TASK))
    assert calls == [runner.TASK]


def test_single_episode_is_unbounded_and_keeps_key_out_of_simulator_and_artifacts(tmp_path, monkeypatch):
    seen = []
    secret = "transient-dummy-secret"

    def fake_run(output, policy, sim_config, **kwargs):
        assert runner.os.environ.get("OPENAI_API_KEY") is None
        assert sim_config.suite == "libero_spatial" and sim_config.seed == 7
        assert sim_config.max_steps is None
        assert kwargs["task_ids"] == [0] and kwargs["episode_indices"] == [0]
        assert kwargs["save_video"] is True and kwargs["resume"] is False
        assert kwargs["client"]._api_key == secret
        assert policy.grid.size == 9 and policy.grid.spacing_m == 0.02
        assert policy.planner.config.horizon == 3 and policy.planner.config.step_m == 0.01
        assert policy.update_steps == 5 and policy.seed == 7
        metadata = kwargs["reference_metadata"]
        assert metadata["task"] == runner.TASK
        assert len(metadata["runner"]["sha256"]) == 64
        assert metadata["policy"]["exact_task_prompt_guard"] is True
        seen.append(kwargs)
        record(output, steps=221)

    monkeypatch.setattr(runner, "run_libero", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    output = tmp_path / "episode"
    summary = runner.execute_episode(output, config=configuration(),
                                     sim_config=LiberoConfig(suite="libero_spatial", max_steps=None),
                                     api_key=secret)
    assert len(seen) == 1
    assert summary["status"] == "success" and summary["libero_success"] is True
    assert summary["steps"] == 221
    assert not any(secret.encode() in path.read_bytes() for path in output.rglob("*") if path.is_file())


@pytest.mark.parametrize("status", ["api_failure", "sim_error", "schema_failure", "timeout", "policy_error"])
def test_infrastructure_or_unexpected_timeout_is_not_a_scored_failure(tmp_path, status):
    record(tmp_path, status=status)
    summary = runner.episode_summary(tmp_path, 0)
    assert summary["status"] == "infrastructure_error"
    assert summary["libero_success"] is None
    assert not summary["evaluation_completed"]
    assert "success_rate" not in summary


def test_budget_exhaustion_is_an_incomplete_episode(tmp_path):
    record(tmp_path, status="budget_exceeded", steps=225)
    summary = runner.episode_summary(tmp_path, 0)
    assert summary["status"] == "budget_exceeded"
    assert summary["libero_success"] is None
    assert not summary["evaluation_completed"]
    assert "success_rate" not in summary


def test_cli_returns_distinct_budget_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "execute_episode", lambda *args, **kwargs: {"status": "budget_exceeded"})
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert runner.main(["--output", str(tmp_path / "episode")]) == 3


def test_resume_validation_error_cannot_be_reported_as_old_success(tmp_path):
    record(tmp_path)
    summary = runner.episode_summary(tmp_path, 0, error="resume rejected: source changed")
    assert summary["status"] == "infrastructure_error"
    assert summary["libero_success"] is None


def test_user_stop_reports_incomplete_episode_and_monitor_exits(tmp_path, monkeypatch):
    monitors = []
    original_monitor = runner.ProgressMonitor

    def monitor(*args, **kwargs):
        instance = original_monitor(*args, interval_s=0.01, **kwargs)
        monitors.append(instance)
        return instance

    def interrupt(output, *args, **kwargs):
        output.mkdir()
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "ProgressMonitor", monitor)
    monkeypatch.setattr(runner, "run_libero", interrupt)
    summary = runner.execute_episode(tmp_path / "episode", config=configuration(),
                                     sim_config=LiberoConfig(suite="libero_spatial", max_steps=None),
                                     api_key="dummy")
    assert summary["status"] == "user_stop" and summary["libero_success"] is None
    assert not summary["evaluation_completed"]
    assert monitors[0].stop_event.is_set() and not monitors[0].thread.is_alive()


def test_progress_uses_persisted_step_stage_and_raw_request_count(tmp_path):
    (tmp_path / "run_state.json").write_text(json.dumps({
        "status": "running", "step": 225, "stage": "deciding", "budget": {"api_calls": 45},
    }))
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "000046-pending.request.json").write_text("{}")
    progress = runner.read_progress(tmp_path)
    assert progress["step"] == 225 and progress["stage"] == "deciding"
    assert progress["api_calls_recorded"] == 45 and progress["raw_request_count"] == 1


def test_cli_preserves_virtualenv_symlink_and_selects_only_initial_state_zero(tmp_path, monkeypatch):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)
    captured = {}

    def fake_execute(output, **kwargs):
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(runner, "execute_episode", fake_execute)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert runner.main(["--output", str(tmp_path / "episode"), "--sim-python", str(venv_python)]) == 0
    assert captured["sim_config"].python_executable == str(venv_python)
    assert captured["sim_config"].max_steps is None
    assert captured["episode_index"] == 0
    assert captured["api_key"] == "dummy" and runner.os.environ.get("OPENAI_API_KEY") is None


def test_cli_default_uses_compatible_simulator_environment(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "execute_episode", lambda output, **kwargs: captured.update(kwargs) or {"status": "success"})
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert runner.main(["--output", str(tmp_path / "episode")]) == 0
    assert captured["sim_config"].python_executable == "/workspace/venvs/libero-plus/bin/python"
