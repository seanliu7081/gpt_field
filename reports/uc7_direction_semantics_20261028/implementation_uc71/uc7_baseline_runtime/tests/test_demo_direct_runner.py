"""Matched one-demo Direct runner integration; model and simulator are fixtures."""
from dataclasses import replace
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.planner import LocalPlanner
from test_demo_conditioning import reference
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


DRAWER = resolve_distance_task("drawer_bowl")
DIRECT_OPTIONS = dict(task_key=DRAWER.key, controller_mode="direct", reasoning_effort="medium",
    execution_mode="fixed", feedback_history="block_summary", centered_placement_guidance=True,
    grasp_settle_guidance=True, field_improvements=False, distance_field_control=False,
    stop_on_model_complete=False)


def direct_response(offset=0, *, phase="drawer_open", target="drawer_handle", reason=None):
    return {"actions": [[round(.025 + offset + i * .013, 6), -.017, .004,
                         .003, -.002, .001, -1. if i < 2 else 1.] for i in range(5)],
        "target_region": target,
        "routing_reason": reason or "Fixture decision from current images, independent of reference actions.",
        "control_context": {"phase": phase, "visual_reference": None}}


def mock_drawer(observation, output, *, success_at=None, fail_at=None):
    obs = replace(observation, prompt=DRAWER.prompt)
    base, instances = fake_simulator(obs, success_at=success_at, fail_at=fail_at)
    resets = []

    class DrawerSimulator(base):
        def list_tasks(self):
            return [{"task_id": 0, "prompt": "unrelated task"},
                    {"task_id": DRAWER.task_id, "prompt": DRAWER.prompt}]

        def reset(self, task_id, episode_index):
            resets.append((task_id, episode_index))
            assert (task_id, episode_index) == (3, 0)
            assert self.config.suite == "libero_goal" and self.config.seed == 7
            assert self.config.max_steps is None
            assert (output / "demonstration.json").is_file()
            result = super().reset(0, 0)
            self.status.update(task_id=task_id, episode_index=episode_index)
            return result

    return DrawerSimulator, instances, resets


def forbid_field_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure Direct invoked field planning or the distance policy")
    monkeypatch.setattr(LocalPlanner, "plan", forbidden)
    monkeypatch.setattr(LocalPlanner, "plan_analytic", forbidden)
    monkeypatch.setattr(runner, "DistanceHybridPolicy", forbidden)


def test_direct_demo_cli_selects_matched_defaults_and_keeps_preflight_read_only(tmp_path, capsys, monkeypatch):
    path, demo = reference(tmp_path)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **kw: pytest.fail("Preflight started simulator"))
    monkeypatch.setattr(runner.ChatGPTPlanDemoDirectClient, "predict",
                        lambda *a, **kw: pytest.fail("Preflight invoked model"))
    assert runner.main(["--task", DRAWER.key, "--demonstration", str(path),
                        "--controller-mode", "direct", "--output", str(output), "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["task"] == DRAWER.prompt and result["task_id"] == 3
    assert result["episode_index"] == 0
    assert result["simulator"]["suite"] == "libero_goal" and result["simulator"]["seed"] == 7
    assert result["demonstration"] == {"artifact": "demonstration.json", **demo.metadata()}
    assert result["billing"] == "chatgpt_subscription"
    assert result["payload"]["model"] == "gpt-6-astra"
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    policy = result["policy"]
    assert policy["controller_mode"] == "direct" and policy["execution_mode"] == "fixed"
    assert policy["feedback_history"] == "block_summary"
    assert policy["centered_placement_guidance"] and policy["grasp_settle_guidance"]
    assert not policy["field_improvements"] and not policy["distance_field_control"]
    assert policy["direct_action_steps"] == policy["direct_action_chunks"] == 5
    assert policy["demo_direct_control"] and policy["method"] == "demonstration_direct"
    assert policy["controllers"] == ["contact_direct"]
    assert policy["local_path_strategy"] is policy["field_refresh_policy"] is None
    assert not policy["field_generation"] and not policy["local_field_planning"]
    schema = result["payload"]["text"]["format"]["schema"]["properties"]
    assert schema["actions"]["minItems"] == schema["actions"]["maxItems"] == 5
    assert not {"field", "controller", "task_complete"}.intersection(schema)
    assert result["max_policy_steps"] is result["max_api_calls"] is result["max_wall_time_s"] is None
    assert result["simulator"]["max_steps"] is None and not result["stop_on_model_complete"]
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not any(result[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    assert not result["api_key_fallback"] and not result["model_fallback"]
    assert not output.exists()


def test_direct_model_chunks_execute_unchanged_and_stop_at_actual_success(tmp_path, observation, monkeypatch):
    path, demo = reference(tmp_path)
    output = tmp_path / "direct-success"
    simulator, instances, resets = mock_drawer(observation, output, success_at=7)
    forbid_field_calls(monkeypatch)
    first, second = direct_response(), direct_response(.1, phase="verify", target="drawer_interior")
    transport = FakeTransport(*(envelope(raw, id=f"direct-demo-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i, raw in enumerate((first, second))))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(suite="libero_goal"), demonstration_path=path,
        token_provider=lambda: "fixture-oauth", transport=transport,
        sim_factory=simulator, recorder_factory=FakeRecorder, **DIRECT_OPTIONS)
    assert result["status"] == "success", result.get("error")
    assert resets == [(3, 0)]
    assert result["steps"] == result["measured_feedback_count"] == 7
    assert result["api_calls"] == result["budget"]["subscription_inference_calls"] == 2
    assert result["api_key_calls"] == 0 and not result["api_key_fallback"]
    assert result["controller_mode"] == "direct" and not result["distance_field_control"]
    assert result["demo_direct_control"]
    assert result["mode"] == "live_chatgpt_subscription_demo_direct"
    assert result["contact_direct_actions"] == 7 and result["coarse_field_actions"] == 0
    assert result["validated_direct_chunks"] == 2
    assert result["local_plan_count"] == result["validated_field_refreshes"] == 0
    assert result["timing_components"]["local_planning_s"] == 0
    assert result["timing_components"]["field_expansion_s"] == 0
    assert result["benchmark_completed"] and result["libero_success"] is True
    assert result["evaluation_completed"] and result["scored_episodes"] == 1
    assert not result["stop_on_model_complete"] and not result["model_declared_complete"]
    # Success on the second action of the second chunk permits the terminal
    # chunk to end early. Every action actually executed remains model output.
    expected = first["actions"] + second["actions"][:2]
    assert instances[0].actions == expected and instances[0].closed
    reference_actions = json.loads(path.read_text())["steps"]
    assert all(action != step["normalized_action"] for action in expected for step in reference_actions)
    assert (output / "demonstration.json").read_bytes() == path.read_bytes()
    assert result["demonstration"] == {"artifact": "demonstration.json", **demo.metadata()}
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["demonstration"] == result["demonstration"]
    assert manifest["task"] == DRAWER.prompt and manifest["task_id"] == 3 and manifest["episode_index"] == 0
    assert manifest["simulator"]["suite"] == "libero_goal" and manifest["simulator"]["seed"] == 7
    assert manifest["policy"]["class"] == "gpt_field.hybrid_policy.HybridFieldDirectPolicy"
    assert not manifest["policy"]["field_generation"] and not manifest["policy"]["local_field_planning"]
    trace = rows(output / "trace.jsonl")
    recorded = [r for r in trace if r["event"] == "action"]
    assert [r["action"] for r in recorded] == expected
    assert all(r["diagnostics"]["controller"] == "contact_direct" for r in recorded)
    assert sum(r["event"] == "executed" for r in trace) == 7
    assert [r["step"] for r in trace if r["event"] == "executed" and r["success"]] == [6]
    assert all(payload["input"][0] == demo.message() for payload in transport.calls)
    assert all(payload["reasoning"] == {"effort": "medium"} for payload in transport.calls)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(transport.calls)


def test_visual_completion_claim_and_verify_phase_do_not_score_or_stop_without_libero(tmp_path, observation, monkeypatch):
    path, _ = reference(tmp_path)
    output = tmp_path / "direct-incomplete"
    simulator, instances, _ = mock_drawer(observation, output, fail_at=7)
    forbid_field_calls(monkeypatch)
    decision = direct_response(phase="verify", target="drawer_interior",
        reason="The current image appears to show a completed placement; fixture claim only.")
    transport = FakeTransport(*(envelope(decision, id=f"unverified-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i in range(2)))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(suite="libero_goal"), demonstration_path=path,
        token_provider=lambda: "fixture-oauth", transport=transport,
        sim_factory=simulator, recorder_factory=FakeRecorder, **DIRECT_OPTIONS)
    assert result["status"] != "success" and len(instances[0].actions) == 7
    assert result["api_calls"] == 2
    assert not result["benchmark_completed"] and not result["evaluation_completed"]
    assert result["scored_episodes"] == 0 and result["incomplete_episodes"] == 1
    assert result["libero_success"] is None and result["success_rate"] is None
    assert not result["libero_success_observed"] and not result["model_declared_complete"]
    assert instances[0].closed


@pytest.mark.parametrize("change", [
    {"reasoning_effort": "high"}, {"execution_mode": "tracking"}, {"feedback_history": "full"},
    {"stop_on_model_complete": True}, {"field_improvements": True}, {"distance_field_control": True},
])
def test_matched_direct_rejects_protocol_drift_before_output_or_credentials(tmp_path, change):
    path, _ = reference(tmp_path)
    output = tmp_path / "not-created"
    options = {**DIRECT_OPTIONS, **change}
    with pytest.raises(ValueError):
        runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(suite="libero_goal"), demonstration_path=path,
            token_provider=lambda: pytest.fail("Invalid configuration requested credentials"), **options)
    assert not output.exists()


def test_drawer_direct_requires_reference_before_output_or_credentials(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(suite="libero_goal"),
            token_provider=lambda: pytest.fail("Missing reference requested credentials"), **DIRECT_OPTIONS)
    assert not output.exists()
