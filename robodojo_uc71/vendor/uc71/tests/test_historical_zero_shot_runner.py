"""Old-protocol zero-shot Direct routing, reset matching, and raw action execution."""
from dataclasses import replace
import hashlib
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from test_demo_direct_runner import direct_response, forbid_field_calls
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation
from test_hybrid_subscription_runner import runner, sim_config


OPTIONS = dict(controller_mode="direct", historical_zero_shot_direct=True,
    reasoning_effort="medium", execution_mode="fixed", feedback_history="block_summary",
    centered_placement_guidance=True, grasp_settle_guidance=True)


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_cli_no_demo_protocol_is_explicit_and_has_no_limits(task, capsys):
    assert runner.main(["--historical-zero-shot-direct", "--controller-mode", "direct",
                        "--task", task, "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "demonstration" not in result
    assert result["policy"]["historical_zero_shot_direct_control"]
    assert result["policy"]["direct_action_steps"] == 5
    assert not result["policy"]["local_field_planning"]
    assert not result["simulator"]["include_arm_joints"]
    assert result["simulator"]["collision_check_mode"] == "none"
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["max_policy_steps"] is result["max_wall_time_s"] is None
    assert not result["stop_on_model_complete"]


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_zero_shot_raw_actions_stop_on_libero_and_exact_reset(tmp_path, observation, monkeypatch, task):
    spec = resolve_distance_task(task)
    obs = replace(observation, prompt=spec.prompt)
    base, instances = fake_simulator(obs, success_at=7)

    class Simulator(base):
        def list_tasks(self):
            return [{"task_id": spec.task_id, "prompt": spec.prompt}]

        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (spec.task_id, 0)
            return super().reset(0, 0)

    forbid_field_calls(monkeypatch)
    raw = direct_response(phase="grasp", target="black_bowl")
    transport = FakeTransport(*(envelope(raw, id=f"direct-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i in range(2)))
    digest = hashlib.sha256(json.dumps(obs.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = tmp_path / "run"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(suite=spec.suite),
        task_key=task, token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=Simulator, recorder_factory=FakeRecorder, expected_initial_observation_sha256=digest, **OPTIONS)
    assert result["status"] == "success", result["error"]
    assert result["libero_success"] is True and result["steps"] == 7
    assert result["api_key_calls"] == 0 and result["api_calls"] == 2
    assert result["local_plan_count"] == result["coarse_field_actions"] == 0
    assert instances[0].actions == raw["actions"] + raw["actions"][:2]
    assert not (output / "demonstration.json").exists()
    assert json.loads((output / "initial_observation_check.json").read_text())["matched"] is True
    assert all("external_demonstration" not in json.dumps(p["input"]) for p in transport.calls)


def test_mismatched_reset_aborts_before_credentials_or_inference(tmp_path, observation):
    simulator, instances = fake_simulator(observation, success_at=1)
    transport = FakeTransport()
    result = runner.execute_episode(tmp_path / "mismatch", config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: pytest.fail("Mismatched reset requested credentials"), transport=transport,
        sim_factory=simulator, recorder_factory=FakeRecorder,
        expected_initial_observation_sha256="0" * 64, **OPTIONS)
    assert result["status"] == "infrastructure_error"
    assert result["steps"] == result["api_calls"] == 0
    assert not transport.calls and instances[0].closed
