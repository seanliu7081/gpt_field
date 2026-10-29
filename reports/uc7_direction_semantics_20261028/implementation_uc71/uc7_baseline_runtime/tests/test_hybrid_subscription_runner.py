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
            "gripper_command": gripper, "motion_mode": "hold", "values": [0] * runner.HYBRID_GRID.count,
            "blocked_indices": [], "unknown_indices": []},
        "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1 if gripper == "close" else -1] for _ in range(runner.HYBRID_CHUNK_SIZE)]} if direct else None}
    return {"id": f"response-{ident}", "model": "gpt-6-astra", "status": "completed",
        "output": [{"id": f"message-{ident}", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(envelope)}]}],
        "usage": {"input_tokens": 10, "output_tokens": 10}}


def execute(output, obs, *, simulator=None, transport=None, **kwargs):
    if simulator is None:
        simulator, _ = fake_simulator(obs, success_at=7)
    return runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport or (lambda *args: reply()),
        sim_factory=simulator, recorder_factory=FakeRecorder, field_representation="dense", **kwargs)


def test_preflight_has_hybrid_schema_and_never_authenticates_or_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *args: pytest.fail("No simulator in preflight"))
    monkeypatch.setattr(runner.ChatGPTPlanHybridClient, "predict", lambda *args: pytest.fail("No inference in preflight"))
    output = tmp_path / "unused"
    result = runner.preflight(runner.configuration(), sim_config(), output, field_representation="dense")
    assert not output.exists()
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not any(result[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    assert result["payload"]["store"] is False and result["payload"]["stream"] is True
    assert result["payload"]["reasoning"] == {"effort": "high"}
    assert result["policy"]["reasoning_effort"] == "high"
    properties = result["payload"]["text"]["format"]["schema"]["properties"]
    assert "controller" in properties
    field_values = properties["field"]["anyOf"][0]["properties"]["values"]
    direct_actions = properties["direct"]["anyOf"][0]["properties"]["actions"]
    assert field_values["minItems"] == field_values["maxItems"] == 35937
    assert direct_actions["minItems"] == direct_actions["maxItems"] == 5
    assert result["policy"]["grid"] == runner.HYBRID_GRID.to_dict()
    assert result["policy"]["controller_update_steps"] == result["policy"]["field_update_steps"] == 30
    assert result["policy"]["direct_action_chunks"] == result["policy"]["direct_action_steps"] == 5
    assert result["policy"]["planner"]["horizon"] == 10
    assert result["policy"]["field_execution_steps"] == 10
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


def test_live_has_no_sixty_five_step_smoke_horizon_and_uses_all_feedback(tmp_path, observation):
    sim, instances = fake_simulator(observation, success_at=76)
    payloads = []
    def transport(endpoint, payload, headers, timeout):
        payloads.append(payload)
        return reply(ident=len(payloads))
    result = execute(tmp_path / "long", observation, simulator=sim, transport=transport)
    assert result["steps"] == 76 and result["api_calls"] == 16
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
    assert len(instances[0].actions) == 76


def test_scripted_smoke_runs_all_three_modes_and_retains_closed_gripper(tmp_path, observation):
    sim, instances = fake_simulator(observation)
    output = tmp_path / "smoke"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=sim, recorder_factory=FakeRecorder)
    assert result["status"] == "diagnostic_complete" and result["diagnostic_complete"]
    assert result["steps"] == result["measured_feedback_count"] == 65
    assert result["coarse_field_actions"] == 60
    assert result["local_plan_count"] == 6
    assert result["contact_direct_actions"] == 5
    assert result["controller_switches"] == 2
    assert result["validated_hybrid_refreshes"] == result["mock_response_calls"] == 3
    assert result["validated_field_refreshes"] == 2 and result["validated_direct_chunks"] == 1
    assert result["api_calls"] == result["api_key_calls"] == result["budget"]["subscription_inference_calls"] == 0
    assert not result["evaluation_completed"] and result["success_rate"] is None
    assert [a[6] for a in instances[0].actions] == [-1] * 30 + [1] * 35
    trace = rows(output / "trace.jsonl")
    switches = [r for r in trace if r["event"] == "controller_switch"]
    assert [(r["step"], r["from_controller"], r["to_controller"]) for r in switches] == [
        (30, "coarse_field", "contact_direct"), (35, "contact_direct", "coarse_field")]
    routing = [r for r in trace if r["event"] == "routing_decision"]
    assert [r["step"] for r in routing] == [0, 30, 35]
    assert all(r["source"] == "model_prediction" for r in routing + switches)
    for file in (output / "requests").glob("*.json"):
        record = json.loads(file.read_text())
        assert record["billing"] == "offline_mock"
        if "payload" in record:
            assert record["prompt_version"] == runner.HYBRID_PROMPT_VERSION
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["policy"]["reasoning_effort"] == "high"
    assert manifest["offline_diagnostic_steps"] == 65
    assert manifest["policy"]["grid"] == runner.HYBRID_GRID.to_dict()
    assert manifest["policy"]["update_steps"] == manifest["policy"]["field_update_steps"] == 30
    assert manifest["policy"]["direct_action_chunks"] == manifest["policy"]["direct_action_steps"] == 5
    assert manifest["policy"]["planner"]["horizon"] == 10
    assert manifest["policy"]["field_execution_steps"] == 10
    assert manifest["policy"]["local_plans_per_field_action"] == .1
    actions = [r for r in trace if r["event"] == "action"]
    first_plan = actions[0]["diagnostics"]["plan"]
    selected = first_plan["candidates"][first_plan["selected_index"]]
    assert len(selected["primitives"]) == 10
    assert len(selected["positions"]) == 11
    assert sum(abs(value) for value in actions[0]["action"][:3]) == pytest.approx(.2)
    field_actions = [r for r in actions if r["diagnostics"]["controller"] == "coarse_field"]
    assert [r["step"] for r in field_actions if r["diagnostics"]["plan_refreshed"]] == [0, 10, 20, 35, 45, 55]
    assert [r["diagnostics"]["plan_step"] for r in field_actions] == list(range(10)) * 6
    for row in field_actions:
        details = row["diagnostics"]
        assert details["plan_execution_steps"] == 10
        assert details["plan_start_step"] == row["step"] - details["plan_step"]
        assert details["plan"] == actions[details["plan_start_step"]]["diagnostics"]["plan"]
        assert [value * .05 for value in row["action"][:3]] == pytest.approx(details["planned_delta_xyz"])
    # The first ten commands execute every planned segment, including turns.
    for index, row in enumerate(actions[:10]):
        expected = [b - a for a, b in zip(selected["positions"][index], selected["positions"][index + 1])]
        assert [value * .05 for value in row["action"][:3]] == pytest.approx(expected)
    assert [r["diagnostics"]["chunk_step"] for r in actions] == list(range(30)) + list(range(5)) + list(range(30))
    assert [r["diagnostics"]["update_steps"] for r in actions] == [30] * 30 + [5] * 5 + [30] * 30
    assert [r["diagnostics"]["chunk_index"] for r in actions] == [0] * 30 + [1] * 5 + [2] * 30
    assert [r["diagnostics"]["routing_decision_step"] for r in actions] == [0] * 30 + [30] * 5 + [35] * 30
    feedback = [r["feedback"] for r in trace if r["event"] == "feedback"]
    assert [r["step"] for r in feedback] == list(range(65))
    assert [r["chunk_index"] for r in feedback] == [0] * 30 + [1] * 5 + [2] * 30
    assert [r["chunk_step"] for r in feedback] == list(range(30)) + list(range(5)) + list(range(30))
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
    monkeypatch.setattr(runner, "preflight", lambda config, sim, output, **kwargs: seen.append(sim.python_executable) or {})
    assert runner.main(["--preflight", "--sim-python", str(alias)]) == 0
    assert seen == [str(alias)]


class CompletionRecorder(FakeRecorder):
    def model_completion(self, obs, **kwargs):
        self.events.append(("model_completion", kwargs))


def compact_reply(*, complete=False, controller="contact_direct", ident=0):
    from test_compact_subscription import hybrid
    prediction = hybrid(contact=controller == "contact_direct")
    if complete:
        prediction["target_region"] = "plate"
        if prediction["field"] is not None:
            prediction["field"]["motion_mode"] = "hold"
            prediction["field"]["gripper_command"] = "open"
        else:
            prediction["direct"]["actions"] = [[0, 0, 0, 0, 0, 0, -1] for _ in range(5)]
    prediction.update(task_complete=complete,
        completion_reason="Bowl rests on the plate with the gripper open and clear." if complete else "")
    response = reply(ident=ident)
    response["reasoning"] = {"effort": "medium"}
    response["output"][0]["content"][0]["text"] = json.dumps(prediction)
    return response


def test_medium_completion_preflight_does_not_authenticate_and_preserves_horizons(tmp_path):
    result = runner.preflight(runner.configuration(), sim_config(), tmp_path / "preview",
        reasoning_effort="medium", stop_on_model_complete=True)
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["policy"]["reasoning_effort"] == "medium"
    assert result["stop_on_model_complete"] and not result["model_completion_is_libero_success"]
    assert "Astra declares task_complete=true" in result["stopping_rule"]
    assert "LIBERO success" in result["stopping_rule"]
    assert result["simulator"]["stop_on_success"] is True
    assert result["policy"]["field_update_steps"] == 30
    assert result["policy"]["direct_action_steps"] == 5
    assert result["policy"]["planner"]["horizon"] == result["policy"]["field_execution_steps"] == 10
    assert not (tmp_path / "preview").exists()
    assert result["api_calls"] == 0
    schema = result["payload"]["text"]["format"]["schema"]
    assert {"task_complete", "completion_reason"} <= set(schema["required"])


@pytest.mark.parametrize("controller", ["coarse_field", "contact_direct"])
def test_model_completion_stops_before_action_planning_and_never_scores_libero(tmp_path, observation, controller):
    sim, instances = fake_simulator(observation)
    requests = []
    def transport(endpoint, payload, headers, timeout):
        requests.append(payload)
        return compact_reply(complete=len(requests) == 2,
            controller=controller if len(requests) == 2 else "contact_direct", ident=len(requests))
    output = tmp_path / controller
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport, sim_factory=sim,
        recorder_factory=CompletionRecorder, reasoning_effort="medium", stop_on_model_complete=True)
    assert result["status"] == "model_declared_complete"
    assert result["model_declared_complete"] and result["benchmark_completed"]
    assert not result["evaluation_completed"] and not result["libero_success_observed"]
    assert result["libero_success"] is result["success_rate"] is None
    assert result["scored_episodes"] == 0 and result["incomplete_episodes"] == 1
    assert result["steps"] == result["measured_feedback_count"] == len(instances[0].actions) == 5
    assert result["pending_action_index"] is None and instances[0].closed
    assert result["api_calls"] == result["validated_hybrid_refreshes"] == 2
    assert result["validated_field_refreshes"] == int(controller == "coarse_field")
    assert result["validated_direct_chunks"] == 1 + int(controller == "contact_direct")
    assert result["local_plan_count"] == result["controller_switches"] == 0
    assert result["final_controller"] == controller
    assert result["api_key_calls"] == 0 and result["billing"] == "chatgpt_subscription"
    assert all(payload["reasoning"] == {"effort": "medium"} for payload in requests)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(requests)
    trace = rows(output / "trace.jsonl")
    assert [row["step"] for row in trace if row["event"] == "action"] == list(range(5))
    assert [row["step"] for row in trace if row["event"] == "executed"] == list(range(5))
    assert [row["event"] for row in trace].count("routing_decision") == 1
    terminal = trace[-1]
    assert terminal["event"] == "model_completion" and terminal["step"] == 5
    assert terminal["diagnostics"]["task_complete"] is True
    assert terminal["prediction"]["controller"] == controller
    assert not terminal["action_proposed"] and not terminal["action_executed"]
    assert result["model_completion"]["completion_reason"] == terminal["completion_reason"]
    assert result["control_elapsed_s"] == terminal["control_elapsed_s"]
    assert result["elapsed_s"] >= terminal["elapsed_s"] >= terminal["control_elapsed_s"] >= 0
    assert result["startup_to_initial_observation_s"] >= 0
    events = CompletionRecorder.instances[-1].events
    assert events[-2][0] == "model_completion" and events[-1][1]["status"] == "model_declared_complete"
    state = json.loads((output / "run_state.json").read_text())
    assert state["stage"] == "finished" and state["model_completion"] == result["model_completion"]
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["policy"]["reasoning_effort"] == "medium"
    assert manifest["stop_on_model_complete"] is True
    assert manifest["simulator"]["stop_on_success"] is True


def test_immediate_model_completion_proposes_and_executes_zero_actions(tmp_path, observation):
    sim, instances = fake_simulator(observation)
    result = runner.execute_episode(tmp_path / "immediate", config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=lambda *args: compact_reply(complete=True),
        sim_factory=sim, recorder_factory=CompletionRecorder,
        reasoning_effort="medium", stop_on_model_complete=True)
    assert result["status"] == "model_declared_complete" and result["steps"] == 0
    assert result["api_calls"] == 1 and result["measured_feedback_count"] == 0
    assert result["pending_action_index"] is None and instances[0].actions == []
    assert result["model_completion"]["step"] == 0


def test_optimized_forced_direct_preserves_five_actions_and_terminal_stop(tmp_path, observation):
    sim, instances = fake_simulator(observation)
    requests = []
    def transport(endpoint, payload, headers, timeout):
        requests.append(payload)
        complete = len(requests) == 2
        response = reply(ident=len(requests))
        response["reasoning"] = {"effort": "medium"}
        response["output"][0]["content"][0]["text"] = json.dumps({
            "actions": [[0, 0, 0, 0, 0, 0, -1 if complete else 1]] * 5,
            "task_complete": complete, "completion_reason": "Bowl released on plate." if complete else ""})
        return response
    output = tmp_path / "direct_summary"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport, sim_factory=sim,
        recorder_factory=CompletionRecorder, reasoning_effort="medium", stop_on_model_complete=True,
        controller_mode="direct", feedback_history="block_summary")
    assert result["status"] == "model_declared_complete"
    assert result["steps"] == len(instances[0].actions) == 5
    assert result["coarse_field_actions"] == result["local_plan_count"] == 0
    assert result["api_calls"] == 2 and result["api_key_calls"] == 0
    assert result["feedback_history"] == "block_summary"
    assert result["timing_components"]["local_planning_s"] == 0
    assert result["timing_components"]["decision_s"] >= result["timing_components"]["model_pipeline_s"] > 0
    assert (output / "frozen_source/compact_subscription.py").is_file()
    assert len([row for row in rows(output / "trace.jsonl") if row["event"] == "feedback"]) == 5
    assert '"type":"execution_feedback"' not in json.dumps(requests[-1]["input"])


def test_optimized_tracking_settings_are_explicit_in_preflight():
    result = runner.preflight(runner.configuration(), sim_config(), reasoning_effort="medium",
        stop_on_model_complete=True, execution_mode="tracking", feedback_history="block_summary")
    assert result["policy"]["execution_mode"] == "tracking"
    assert result["policy"]["tracking_lookahead_m"] == .03
    assert result["policy"]["tracking_max_command_m"] == .05
    assert result["policy"]["feedback_history"] == "block_summary"
    assert result["api_calls"] == 0


@pytest.mark.parametrize("grasp_guidance", [False, True])
def test_centered_direct_holds_continue_until_actual_libero_success(tmp_path, observation, grasp_guidance):
    sim, instances = fake_simulator(observation, success_at=7)
    requests = []

    def transport(endpoint, payload, headers, timeout):
        requests.append(payload)
        response = reply(ident=len(requests))
        response["reasoning"] = {"effort": "medium"}
        response["output"][0]["content"][0]["text"] = json.dumps({
            "actions": [[0, 0, 0, 0, 0, 0, -1]] * 5})
        return response

    output = tmp_path / "centered_direct"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport, sim_factory=sim,
        recorder_factory=FakeRecorder, reasoning_effort="medium", controller_mode="direct",
        feedback_history="block_summary", centered_placement_guidance=True,
        grasp_settle_guidance=grasp_guidance)
    assert result["status"] == "success" and result["libero_success_observed"]
    assert result["steps"] == len(instances[0].actions) == 7
    assert result["api_calls"] == 2 and not result["stop_on_model_complete"]
    assert result["centered_placement_guidance"] and result["local_plan_count"] == 0
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["policy"]["centered_placement_guidance"]
    assert manifest["policy"]["grasp_settle_guidance"] is grasp_guidance
    assert result["grasp_settle_guidance"] is grasp_guidance
    version = "-centered-placement-grasp-settle-v5" if grasp_guidance else "-centered-placement-v4"
    assert manifest["policy"]["prompt_version"].endswith(version)
    for request in requests:
        assert request["text"]["format"]["schema"]["required"] == ["actions"]
        assert "MUST_NOT_ENTER_POLICY" not in json.dumps(request)


@pytest.mark.parametrize("options", [
    {"controller_mode": "hybrid", "stop_on_model_complete": True},
    {"controller_mode": "direct", "stop_on_model_complete": True},
    {"controller_mode": "direct", "field_representation": "dense"},
])
def test_centered_placement_guidance_rejects_incompatible_runner_modes(options):
    with pytest.raises(ValueError, match="Centered placement guidance"):
        runner.preflight(runner.configuration(), sim_config(), centered_placement_guidance=True, **options)


def test_grasp_guidance_requires_centered_protocol():
    with pytest.raises(ValueError, match="Grasp settling guidance"):
        runner.preflight(runner.configuration(), sim_config(), controller_mode="direct", grasp_settle_guidance=True)


def test_fair_hybrid_and_direct_preflight_share_operational_guidance():
    from gpt_field.compact_subscription import compact_operational_guidance
    guidance = compact_operational_guidance(centered_placement_guidance=True, grasp_settle_guidance=True)
    for mode in ("hybrid", "direct"):
        result = runner.preflight(runner.configuration(), sim_config(), reasoning_effort="medium",
            controller_mode=mode, execution_mode="tracking" if mode == "hybrid" else "fixed",
            feedback_history="block_summary", centered_placement_guidance=True, grasp_settle_guidance=True)
        assert result["payload"]["instructions"].count(guidance) == 1
        assert result["policy"]["direct_action_steps"] == 5
        assert result["policy"]["field_update_steps"] == 30
        assert not result["stop_on_model_complete"]
        assert "task_complete" not in result["payload"]["text"]["format"]["schema"]["properties"]
        assert result["payload"]["reasoning"] == {"effort": "medium"}
        assert result["api_calls"] == result["api_key_calls"] == 0


@pytest.mark.parametrize("success_at_reset", [False, True])
def test_opt_in_stops_on_libero_success_before_astra_completion(tmp_path, observation, success_at_reset):
    base_sim, instances = fake_simulator(observation, success_at=None if success_at_reset else 1)
    output = tmp_path / "early_success"
    observed_statuses = []
    class NonterminalSuccessSim(base_sim):
        def reset(self, *args):
            obs = super().reset(*args)
            assert self.config.stop_on_success is True
            if success_at_reset:
                self.status["success"] = True
            return obs

        def step(self, action):
            observed_statuses.append(json.loads((output / "run_state.json").read_text())["status"])
            obs, success = super().step(action)
            self.status["terminated"] = False
            return obs, success

    calls = []
    def transport(endpoint, payload, headers, timeout):
        calls.append(payload)
        return compact_reply(complete=len(calls) == 2, ident=len(calls))
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=transport,
        sim_factory=NonterminalSuccessSim, recorder_factory=CompletionRecorder,
        reasoning_effort="medium", stop_on_model_complete=True)
    expected_steps = 0 if success_at_reset else 1
    assert result["status"] == "success"
    assert result["steps"] == len(instances[0].actions) == expected_steps
    assert len(calls) == expected_steps
    assert observed_statuses == ["running"] * expected_steps
    assert not result["model_declared_complete"] and result["benchmark_completed"]
    assert result["libero_success_observed"] and result["libero_success"] is True
    assert result["evaluation_completed"] and result["success_rate"] == 1.0
    assert result["scored_episodes"] == 1 and result["incomplete_episodes"] == 0
    assert result["model_completion"] is None
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(calls)
    assert "libero_success" not in json.dumps(calls)


def test_model_completion_cleanup_failure_is_not_a_completed_benchmark(tmp_path, observation):
    sim, _ = fake_simulator(observation)
    class BrokenRecorder(CompletionRecorder):
        def close(self, **kwargs):
            raise RuntimeError("Synthetic recorder failure")
    result = runner.execute_episode(tmp_path / "cleanup", config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=lambda *args: compact_reply(complete=True),
        sim_factory=sim, recorder_factory=BrokenRecorder,
        reasoning_effort="medium", stop_on_model_complete=True)
    assert result["status"] == "infrastructure_error"
    assert not result["benchmark_completed"] and not result["evaluation_completed"]
    assert result["model_completion"]["task_complete"] is True


@pytest.mark.parametrize("flags", [
    ["--offline-smoke", "--reasoning-effort", "medium"],
    ["--offline-smoke", "--stop-on-model-complete"],
    ["--field-representation", "dense", "--reasoning-effort", "medium"],
    ["--field-representation", "dense", "--stop-on-model-complete"],
    ["--timing-comparison", "--reasoning-effort", "medium"],
    ["--timing-comparison", "--stop-on-model-complete"],
])
def test_unsupported_completion_modes_fail_before_authentication(tmp_path, monkeypatch, flags):
    import gpt_field.chatgpt_auth as auth
    monkeypatch.setattr(auth, "SubscriptionCredentials", lambda **kwargs: pytest.fail("No credentials for invalid options"))
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--output", str(output), *flags])
    assert exc.value.code == 2 and not output.exists()


def test_cli_model_completion_is_successful_exit_without_libero_score(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(auth, "SubscriptionCredentials", lambda **kwargs: SimpleNamespace(access_token=lambda: "unused"))
    def execute_fixture(output, **kwargs):
        calls.append(kwargs)
        return {"evaluation_completed": False, "diagnostic_complete": False, "model_declared_complete": True}
    monkeypatch.setattr(runner, "execute_episode", execute_fixture)
    result = runner.main(["--output", str(tmp_path / "finished"), "--reasoning-effort", "medium", "--stop-on-model-complete"])
    assert result == 0 and len(calls) == 1
    assert calls[0]["reasoning_effort"] == "medium" and calls[0]["stop_on_model_complete"] is True
