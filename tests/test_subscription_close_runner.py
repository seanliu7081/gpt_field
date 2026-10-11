"""Subscription close-stop behavior; all inference and simulation are fixtures."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path

import pytest

from gpt_field.action import ActionAdapter
from gpt_field.astra import APIError, MissingCredentials
from gpt_field.chatgpt_plan import _protocol_error
from test_field_history_runner import (
    FakeRecorder, fake_simulator, observation, response, rows, sim_config,
)


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_bowl_subscription_close",
    ROOT / "scripts/evaluate_bowl_subscription_close.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def field_response(gripper="close"):
    value = response()
    field = json.loads(value["output"][0]["content"][0]["text"])
    field["gripper_command"] = gripper
    value["output"][0]["content"][0]["text"] = json.dumps(field)
    value["model"] = "gpt-6-astra"
    return value


def execute(output, observation, *, transport=None, simulator=None, **kwargs):
    if simulator is None:
        simulator, _ = fake_simulator(observation)
    return runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "oauth-fixture-subscription", sim_factory=simulator,
        recorder_factory=FakeRecorder, transport=transport or (lambda *args: field_response()), **kwargs)


def test_preflight_is_no_auth_no_network_no_sim_and_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("No simulator expected"))
    monkeypatch.setattr(runner.ChatGPTPlanFieldClient, "predict", lambda *a: pytest.fail("No inference expected"))
    result = runner.preflight(runner.configuration(), sim_config(), tmp_path / "unused")
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert result["credentials_checked"] is result["simulator_started"] is result["files_written"] is False
    assert result["payload"]["store"] is False and result["payload"]["stream"] is True
    assert "max_output_tokens" not in result["payload"]
    assert result["max_policy_steps"] is result["max_api_calls"] is result["max_wall_time_s"] is None
    assert result["billing"] == "chatgpt_subscription" and not (tmp_path / "unused").exists()


@pytest.mark.parametrize("changes", [
    {"model": "other"}, {"max_api_calls": 1}, {"max_wall_time_s": 100}, {"max_cost_usd": 300},
    {"input_price_per_million": 10}, {"output_price_per_million": 50}, {"timeout_s": 20},
])
def test_configuration_rejects_model_fallback_caps_and_api_key_pricing(changes):
    with pytest.raises(ValueError):
        runner.validate_config(replace(runner.configuration(), **changes), sim_config())


@pytest.mark.parametrize("changes", [{"max_steps": 10}, {"suite": "libero_10"}, {"seed": 8},
    {"settle_steps": 0}, {"render_size": 224}, {"image_size": 256}])
def test_same_simulator_protocol_required(changes):
    with pytest.raises(ValueError):
        runner.validate_config(runner.configuration(), sim_config(**changes))


def test_first_close_is_executed_once_then_feedback_video_and_final_state_saved(tmp_path, observation):
    output = tmp_path / "live"
    checked = []

    def before_step(index, action):
        assert action[6] > 0 and index == 0
        assert rows(output / "trace.jsonl")[-1]["event"] == "action"
        state = json.loads((output / "run_state.json").read_text())
        assert state["status"] == "running" and state["pending_action_index"] == 0
        assert state["close_event"] is None
        checked.append(index)

    simulator, instances = fake_simulator(observation, before_step=before_step)
    result = execute(output, observation, simulator=simulator)
    assert checked == [0] and len(instances[0].actions) == 1
    assert result["status"] == "first_gripper_close_executed"
    assert result["steps"] == result["measured_feedback_count"] == result["gripper_close_actions"] == 1
    assert result["close_event"]["action_index"] == 0 and result["close_event"]["action_ordinal"] == 1
    assert result["close_event"]["gripper_joints_before"] == list(observation.state[6:8])
    assert result["close_event"]["gripper_joint_delta"] == [0, 0]
    assert not result["close_event"]["full_closure_verified"] and not result["close_event"]["grasp_verified"]
    assert result["diagnostic_complete"] and result["close_execution_acknowledged"]
    assert not result["evaluation_completed"] and result["success_rate"] is None and result["scored_episodes"] == 0
    assert [row["event"] for row in rows(output / "trace.jsonl")] == ["reset", "action", "executed", "feedback"]
    assert instances[0].closed and result["pending_action_index"] is None
    video = FakeRecorder.instances[-1]
    assert [event for event, _ in video.events] == ["observe", "decision", "executed", "close"]
    assert video.events[-2][1]["history_stats"]["feedback_records"] == 1
    assert json.loads((output / "run_state.json").read_text())["status"] == result["status"]


def test_stop_uses_actual_action_not_prediction_text(tmp_path, observation, monkeypatch):
    original = ActionAdapter.adapt
    count = 0

    def delayed_actual_close(self, *args):
        nonlocal count
        count += 1
        action = original(self, *args)
        return (*action[:6], -1.0 if count < 3 else 1.0)

    monkeypatch.setattr(ActionAdapter, "adapt", delayed_actual_close)
    result = execute(tmp_path / "delayed", observation)
    assert result["steps"] == result["measured_feedback_count"] == 3
    assert result["gripper_open_actions"] == 2 and result["gripper_close_actions"] == 1
    assert result["api_calls"] == 1 and result["close_event"]["action_index"] == 2


def test_no_action_horizon_and_feedback_precedes_each_field_refresh(tmp_path, observation):
    payloads = []

    def transport(endpoint, payload, headers, timeout):
        payloads.append(payload)
        return field_response("open" if len(payloads) < 5 else "close")

    result = execute(tmp_path / "long", observation, transport=transport)
    assert result["steps"] == 21 and result["measured_feedback_count"] == 21
    assert result["api_calls"] == result["validated_field_refreshes"] == 5
    assert result["local_plan_count"] == 21
    assert result["budget"]["max_api_calls"] is result["budget"]["max_wall_time_s"] is None
    for index, payload in enumerate(payloads):
        records = [json.loads(part["text"]) for message in payload["input"]
                   for part in message.get("content", []) if part["type"] == "input_text" and part["text"].startswith("{")]
        assert [record["step"] for record in records if record.get("type") == "execution_feedback"] == list(range(index * 5))
        assert "MUST_NOT_ENTER_POLICY" not in json.dumps(payload)
        assert payload["model"] == "gpt-6-astra" and payload["stream"] is True
        assert "max_output_tokens" not in payload


def test_smoke_uses_no_credentials_and_stops_after_six_actions(tmp_path, observation, monkeypatch):
    simulator, instances = fake_simulator(observation)
    monkeypatch.setattr(runner.ChatGPTPlanFieldClient, "preflight", lambda self: None)
    result = runner.execute_episode(tmp_path / "smoke", config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=simulator, recorder_factory=FakeRecorder)
    assert result["status"] == runner.STOP_REASON and result["steps"] == 6
    assert result["gripper_open_actions"] == 5 and result["gripper_close_actions"] == 1
    assert result["api_calls"] == result["api_key_calls"] == result["budget"]["subscription_inference_calls"] == 0
    assert result["mock_response_calls"] == 2 and result["billing"] == "offline_mock"
    assert len(instances[0].actions) == 6 and FakeRecorder.instances[-1].kwargs["offline_smoke"]
    assert all(json.loads(path.read_text())["billing"] == "offline_mock"
               for path in (tmp_path / "smoke/requests").glob("*.json"))


def test_success_before_close_remains_unscored(tmp_path, observation):
    simulator, instances = fake_simulator(observation, success_at=2)
    result = execute(tmp_path / "success_before", observation, simulator=simulator,
                     transport=lambda *args: field_response("open"))
    assert result["status"] == "libero_success_before_close" and result["steps"] == 2
    assert result["libero_success_observed"] and not result["close_execution_acknowledged"]
    assert not result["diagnostic_complete"] and result["success_rate"] is None
    assert "before any close action" in result["interpretation"]
    assert instances[0].closed


def test_success_on_close_still_reports_diagnostic_not_task_score(tmp_path, observation):
    simulator, _ = fake_simulator(observation, success_at=1)
    result = execute(tmp_path / "success_on_close", observation, simulator=simulator)
    assert result["status"] == runner.STOP_REASON and result["libero_success_observed"]
    assert not result["evaluation_completed"] and result["success_rate"] is None and result["scored_episodes"] == 0


def test_failed_close_step_never_claims_execution_and_keeps_pending_marker(tmp_path, observation):
    simulator, instances = fake_simulator(observation, fail_at=0)
    output = tmp_path / "step_failure"
    result = execute(output, observation, simulator=simulator)
    assert result["status"] == "infrastructure_error" and result["steps"] == 0
    assert result["pending_action_index"] == 0 and not result["close_execution_acknowledged"]
    assert result["gripper_close_actions"] == result["measured_feedback_count"] == 0
    assert [row["event"] for row in rows(output / "trace.jsonl")] == ["reset", "action"]
    assert not instances[0].actions and instances[0].closed


def test_quota_error_has_no_fallback_retry_or_close_claim(tmp_path, observation):
    attempts = []

    def limited(*args):
        attempts.append(True)
        raise _protocol_error("private server detail must not leak", http_status=429,
                              error_code="subscription_sharing_usage_limit_exceeded")

    output = tmp_path / "quota"
    result = execute(output, observation, transport=limited)
    assert len(attempts) == 1 and result["api_calls"] == 1
    assert result["status"] == "subscription_error" and result["steps"] == 0
    assert result["error"]["http_status"] == 429
    assert result["error"]["error_code"] == "subscription_sharing_usage_limit_exceeded"
    assert not result["close_execution_acknowledged"] and not result["diagnostic_complete"]
    assert "private server detail" not in json.dumps(result)
    assert "before any close action" in result["interpretation"]


def test_auth_failure_never_falls_back_to_env_api_key(tmp_path, observation, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-use-this")
    simulator, instances = fake_simulator(observation)

    def denied():
        raise RuntimeError("oauth-private-failure")

    result = runner.execute_episode(tmp_path / "auth", config=runner.configuration(), sim_config=sim_config(),
        token_provider=denied, sim_factory=simulator, recorder_factory=FakeRecorder,
        transport=lambda *args: pytest.fail("Authentication failed; no inference allowed"))
    assert result["status"] == "subscription_error" and result["api_calls"] == result["steps"] == 0
    assert result["api_key_calls"] == 0 and not result["api_key_fallback"]
    assert instances[0].closed
    assert "oauth-private-failure" not in json.dumps(result) and "sk-never-use-this" not in json.dumps(result)
    assert os.environ["OPENAI_API_KEY"] == "sk-never-use-this"


def test_interrupt_finalizes_video_and_marks_unknown_usage(tmp_path, observation):
    def interrupt(*args):
        raise KeyboardInterrupt

    result = execute(tmp_path / "interrupt", observation, transport=interrupt)
    assert result["status"] == "user_stop" and result["steps"] == 0
    assert result["budget"]["unknown_usage_calls"] == 1 and result["budget"]["cost_usd_upper_bound"] is None
    assert FakeRecorder.instances[-1].events[-1][0] == "close"


def test_feedback_failure_after_close_preserves_ack_without_false_diagnostic_completion(tmp_path, observation, monkeypatch):
    monkeypatch.setattr(runner.FieldHistoryPolicy, "observe_transition",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("feedback failed")))
    result = execute(tmp_path / "feedback_error", observation)
    assert result["status"] == "infrastructure_error" and result["steps"] == 1
    assert result["close_execution_acknowledged"] and not result["diagnostic_complete"]
    assert result["measured_feedback_count"] == 0
    assert len([row for row in rows(tmp_path / "feedback_error/trace.jsonl") if row["event"] == "executed"]) == 1


def test_fresh_output_and_explicit_subscription_provider_required(tmp_path, observation):
    with pytest.raises(FileExistsError):
        execute(tmp_path, observation)
    with pytest.raises(MissingCredentials):
        runner.execute_episode(tmp_path / "noauth", config=runner.configuration(), sim_config=sim_config())
    assert not (tmp_path / "noauth").exists()


def test_cli_preflight_preserves_simulator_venv_symlink(tmp_path, monkeypatch):
    executable = tmp_path / "real_python"
    executable.touch()
    alias = tmp_path / "venv_python"
    alias.symlink_to(executable)
    captured = {}

    def preview(config, simulation, output):
        captured.update(config=config, simulation=simulation)
        return {}

    monkeypatch.setattr(runner, "preflight", preview)
    assert runner.main(["--preflight", "--sim-python", str(alias)]) == 0
    assert captured["simulation"].python_executable == str(alias)
    assert captured["config"].max_cost_usd is None
