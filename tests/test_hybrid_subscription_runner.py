"""Hybrid full-task runner boundaries; all model/simulator calls are fixtures."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path

import pytest

from gpt_field.astra import MissingCredentials
from gpt_field.chatgpt_plan import _protocol_error
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows, sim_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_bowl_hybrid_subscription",
    ROOT / "scripts/evaluate_bowl_hybrid_subscription.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def reply(controller="contact_direct", gripper="close", ident=0):
    direct = controller == "contact_direct"
    envelope = {"controller": controller, "target_region": "black_bowl", "near_target": direct,
        "routing_reason": "Synthetic fixture visual routing explanation.",
        "field": None if direct else {"stage": "fixture_field", "orientation_target_xyzw": [0, 0, 0, 1],
            "gripper_command": gripper, "motion_mode": "hold", "values": [0] * 729,
            "blocked_indices": [], "unknown_indices": []},
        "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1 if gripper == "close" else -1] for _ in range(5)]} if direct else None}
    return {"id": f"response-{ident}", "model": "gpt-6-astra", "status": "completed",
        "output": [{"id": f"message-{ident}", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(envelope)}]}],
        "usage": {"input_tokens": 10, "output_tokens": 10}}


def execute(output, obs, *, simulator=None, transport=None, **kwargs):
    if simulator is None:
        simulator, _ = fake_simulator(obs, success_at=7)
    return runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport or (lambda *args: reply()),
        sim_factory=simulator, recorder_factory=FakeRecorder, **kwargs)


def test_preflight_has_hybrid_schema_and_never_authenticates_or_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *args: pytest.fail("No simulator in preflight"))
    monkeypatch.setattr(runner.ChatGPTPlanHybridClient, "predict", lambda *args: pytest.fail("No inference in preflight"))
    output = tmp_path / "unused"
    result = runner.preflight(runner.configuration(), sim_config(), output)
    assert not output.exists()
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not any(result[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    assert result["payload"]["store"] is False and result["payload"]["stream"] is True
    assert "controller" in result["payload"]["text"]["format"]["schema"]["properties"]
    assert result["max_policy_steps"] is result["max_api_calls"] is result["max_wall_time_s"] is None


@pytest.mark.parametrize("change", [{"max_api_calls": 1}, {"max_wall_time_s": 2}, {"max_cost_usd": 300},
                                     {"model": "different-model"}, {"input_price_per_million": 10}])
def test_live_configuration_rejects_horizons_model_fallback_and_api_prices(change):
    with pytest.raises(ValueError):
        runner.validate_config(replace(runner.configuration(), **change), sim_config())


def test_live_continues_through_close_and_scores_only_actual_task_success(tmp_path, observation):
    sim, instances = fake_simulator(observation, success_at=7)
    result = execute(tmp_path / "success", observation, simulator=sim)
    assert result["status"] == "success" and result["steps"] == 7
    assert result["first_close_event"]["action_index"] == 0
    assert result["gripper_close_actions"] == result["contact_direct_actions"] == 7
    assert result["validated_direct_chunks"] == result["validated_hybrid_refreshes"] == 2
    assert result["local_plan_count"] == result["validated_field_refreshes"] == 0
    assert result["measured_feedback_count"] == 7
    assert result["evaluation_completed"] and result["scored_episodes"] == 1 and result["success_rate"] == 1
    assert len(instances[0].actions) == 7 and instances[0].closed
    assert FakeRecorder.instances[-1].events[-1][1]["status"] == "success"


def test_live_has_no_fifteen_step_smoke_horizon_and_uses_all_feedback(tmp_path, observation):
    sim, instances = fake_simulator(observation, success_at=26)
    payloads = []
    def transport(endpoint, payload, headers, timeout):
        payloads.append(payload)
        return reply(ident=len(payloads))
    result = execute(tmp_path / "long", observation, simulator=sim, transport=transport)
    assert result["steps"] == 26 and result["api_calls"] == 6
    for ordinal, payload in enumerate(payloads):
        feedback = []
        for message in payload["input"]:
            for part in message.get("content", []):
                if part.get("type") == "input_text" and part["text"].startswith("{"):
                    value = json.loads(part["text"])
                    if value.get("type") == "execution_feedback":
                        feedback.append(value)
        assert [item["step"] for item in feedback] == list(range(ordinal * 5))
        assert "MUST_NOT_ENTER_POLICY" not in json.dumps(payload)
    assert len(instances[0].actions) == 26


def test_scripted_smoke_runs_all_three_modes_and_retains_closed_gripper(tmp_path, observation):
    sim, instances = fake_simulator(observation)
    output = tmp_path / "smoke"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=sim, recorder_factory=FakeRecorder)
    assert result["status"] == "diagnostic_complete" and result["diagnostic_complete"]
    assert result["steps"] == result["measured_feedback_count"] == 15
    assert result["coarse_field_actions"] == result["local_plan_count"] == 10
    assert result["contact_direct_actions"] == 5
    assert result["controller_switches"] == 2
    assert result["validated_hybrid_refreshes"] == result["mock_response_calls"] == 3
    assert result["validated_field_refreshes"] == 2 and result["validated_direct_chunks"] == 1
    assert result["api_calls"] == result["api_key_calls"] == result["budget"]["subscription_inference_calls"] == 0
    assert not result["evaluation_completed"] and result["success_rate"] is None
    assert [a[6] for a in instances[0].actions] == [-1] * 5 + [1] * 10
    trace = rows(output / "trace.jsonl")
    switches = [r for r in trace if r["event"] == "controller_switch"]
    assert [(r["step"], r["from_controller"], r["to_controller"]) for r in switches] == [
        (5, "coarse_field", "contact_direct"), (10, "contact_direct", "coarse_field")]
    routing = [r for r in trace if r["event"] == "routing_decision"]
    assert [r["step"] for r in routing] == [0, 5, 10]
    assert all(r["source"] == "model_prediction" for r in routing + switches)
    for file in (output / "requests").glob("*.json"):
        record = json.loads(file.read_text())
        assert record["billing"] == "offline_mock"
        if "payload" in record:
            assert record["prompt_version"] == runner.HYBRID_PROMPT_VERSION
    assert instances[0].closed


def test_scripted_success_remains_unscored(tmp_path, observation):
    sim, _ = fake_simulator(observation, success_at=2)
    result = runner.execute_episode(tmp_path / "smoke_success", config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=sim, recorder_factory=FakeRecorder)
    assert result["libero_success_observed"] and result["status"] == "diagnostic_sim_terminated"
    assert not result["evaluation_completed"] and result["success_rate"] is None and result["scored_episodes"] == 0


def test_failed_sim_step_preserves_pending_action_without_fabricated_feedback(tmp_path, observation):
    sim, instances = fake_simulator(observation, fail_at=0)
    result = execute(tmp_path / "sim_error", observation, simulator=sim)
    assert result["status"] == "infrastructure_error" and result["steps"] == 0
    assert result["pending_action_index"] == 0 and result["first_close_event"] is None
    assert result["measured_feedback_count"] == 0 and result["success_rate"] is None
    assert instances[0].closed and not instances[0].actions


def test_feedback_failure_cannot_report_task_completion_or_next_action(tmp_path, observation, monkeypatch):
    sim, instances = fake_simulator(observation, success_at=1)
    monkeypatch.setattr(runner.HybridFieldDirectPolicy, "observe_transition",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("fixture feedback failure")))
    result = execute(tmp_path / "feedback_error", observation, simulator=sim)
    assert result["status"] == "infrastructure_error" and result["steps"] == 1
    assert result["libero_success_observed"] and not result["evaluation_completed"]
    assert result["measured_feedback_count"] == 0 and len(instances[0].actions) == 1
    assert result["scored_episodes"] == 0


def test_quota_failure_does_not_retry_or_score_a_failure(tmp_path, observation):
    calls = []
    def quota(*args):
        calls.append(1)
        raise _protocol_error("fixture private detail", http_status=429,
                              error_code="subscription_sharing_usage_limit_exceeded")
    result = execute(tmp_path / "quota", observation, transport=quota)
    assert len(calls) == 1 and result["status"] == "subscription_error"
    assert result["steps"] == 0 and result["api_key_calls"] == 0
    assert result["error"]["http_status"] == 429
    assert not result["evaluation_completed"] and result["success_rate"] is None
    assert "fixture private detail" not in json.dumps(result)


def test_auth_failure_cannot_fallback_to_api_key(tmp_path, observation, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unused-fixture")
    sim, _ = fake_simulator(observation)
    def denied():
        raise RuntimeError("synthetic-private-auth-error")
    result = runner.execute_episode(tmp_path / "auth", config=runner.configuration(), sim_config=sim_config(),
        token_provider=denied, sim_factory=sim, recorder_factory=FakeRecorder,
        transport=lambda *args: pytest.fail("No network after failed authentication"))
    assert result["status"] == "subscription_error" and result["api_calls"] == 0
    assert not result["api_key_fallback"] and result["api_key_calls"] == 0
    assert "synthetic-private-auth-error" not in json.dumps(result)
    assert os.environ["OPENAI_API_KEY"] == "sk-unused-fixture"


def test_user_interrupt_finalizes_and_is_unscored(tmp_path, observation):
    def interrupted(*args):
        raise KeyboardInterrupt
    result = execute(tmp_path / "interrupted", observation, transport=interrupted)
    assert result["status"] == "user_stop" and result["budget"]["unknown_usage_calls"] == 1
    assert not result["evaluation_completed"] and FakeRecorder.instances[-1].events[-1][0] == "close"


def test_fresh_directory_and_explicit_subscription_provider_required(tmp_path, observation):
    with pytest.raises(FileExistsError):
        execute(tmp_path, observation)
    with pytest.raises(MissingCredentials):
        runner.execute_episode(tmp_path / "unauthorized", config=runner.configuration(), sim_config=sim_config())
    assert not (tmp_path / "unauthorized").exists()


def test_smoke_cannot_accept_custom_transport_or_credentials(tmp_path, observation):
    with pytest.raises(ValueError):
        runner.execute_episode(tmp_path / "no", config=runner.configuration(), sim_config=sim_config(),
            offline_smoke=True, token_provider=lambda: "unused")


def test_cli_preserves_simulator_venv_symlink(tmp_path, monkeypatch):
    actual = tmp_path / "python"
    actual.touch()
    alias = tmp_path / "venv_python"
    alias.symlink_to(actual)
    seen = []
    monkeypatch.setattr(runner, "preflight", lambda config, sim, output: seen.append(sim.python_executable) or {})
    assert runner.main(["--preflight", "--sim-python", str(alias)]) == 0
    assert seen == [str(alias)]
