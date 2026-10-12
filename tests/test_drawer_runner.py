"""Drawer runner task identity and stopping contracts; no live model/simulator."""
from dataclasses import replace
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from test_distance_runner import SETTINGS
from test_drawer_policy import drawer_response
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


DRAWER = resolve_distance_task("drawer_bowl")


def test_drawer_cli_preflight_selects_goal3_uncapped_medium_subscription(capsys, monkeypatch):
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **kw: pytest.fail("Preflight started simulator"))
    monkeypatch.setattr(runner.ChatGPTPlanDistanceClient, "predict",
                        lambda *a, **kw: pytest.fail("Preflight made inference request"))
    assert runner.main(["--task", DRAWER.key, "--distance-field-control", "--preflight"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (result["simulator"]["suite"], result["task_id"], result["task"]) == (
        "libero_goal", 3, DRAWER.prompt)
    assert result["distance_task"] == DRAWER.key
    assert result["task_spec"]["allow_upward_reobservation"] is False
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not result["credentials_checked"] and not result["simulator_started"]
    assert result["billing"] == "chatgpt_subscription"
    assert result["payload"]["model"] == "gpt-6-astra"
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["policy"]["direct_action_steps"] == 5
    assert result["max_policy_steps"] is result["max_api_calls"] is result["max_wall_time_s"] is None
    assert result["simulator"]["max_steps"] is None
    assert not result["stop_on_model_complete"]
    assert not result["api_key_fallback"] and not result["model_fallback"]
    assert "plate" not in result["payload"]["instructions"].lower()
    assert result["payload"]["text"]["format"]["schema"]["properties"]["target_region"]["enum"] == list(DRAWER.target_regions)


def test_nondefault_task_requires_distance_mode_before_output_or_credentials(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--task", DRAWER.key, "--preflight", "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_drawer_rejects_wrong_simulator_suite_before_creating_run(tmp_path):
    output = tmp_path / "wrong-suite"
    with pytest.raises(ValueError, match="suite must match"):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
            task_key=DRAWER.key, token_provider=lambda: pytest.fail("Unexpected credentials"), **SETTINGS)
    assert not output.exists()


def test_mock_drawer_run_resets_actual_task3_and_records_consistent_identity(tmp_path, observation):
    obs = replace(observation, prompt=DRAWER.prompt)
    base, instances = fake_simulator(obs, success_at=6)
    reset_calls = []

    class DrawerSimulator(base):
        def list_tasks(self):
            # Deliberately include task0 first: selecting the first list entry is
            # a real regression for the new task profile.
            return [{"task_id": 0, "prompt": "unrelated fixture"},
                    {"task_id": DRAWER.task_id, "prompt": DRAWER.prompt}]

        def reset(self, task_id, episode_index):
            reset_calls.append((task_id, episode_index))
            assert (task_id, episode_index) == (3, 0)
            assert self.config.suite == "libero_goal"
            result = super().reset(0, 0)
            self.status.update(task_id=task_id, episode_index=episode_index)
            return result

    opening = drawer_response(phase="drawer_open", direct=True)
    opening["direct"]["actions"] = [[.02 * i, 0., 0., 0., 0., 0., 1.] for i in range(5)]
    verify = drawer_response(target="drawer_interior", phase="verify", hold=True)
    transport = FakeTransport(*(envelope(raw, id=f"drawer-runner-{i}", model="gpt-6-astra",
                                        reasoning={"effort": "medium"}) for i, raw in enumerate((opening, verify))))
    output = tmp_path / "drawer-run"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(suite="libero_goal"),
        task_key=DRAWER.key, token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=DrawerSimulator, recorder_factory=FakeRecorder, **SETTINGS)
    assert result["status"] == "success", result.get("error")
    assert reset_calls == [(3, 0)]
    assert (result["suite"], result["task_id"], result["task"], result["distance_task"]) == (
        "libero_goal", 3, DRAWER.prompt, DRAWER.key)
    assert result["steps"] == result["measured_feedback_count"] == 6
    assert result["api_key_calls"] == 0 and result["budget"]["subscription_inference_calls"] == 2
    assert instances[0].actions[:5] == opening["direct"]["actions"]
    assert instances[0].actions[5][:3] == [0., 0., 0.]
    assert instances[0].closed
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert (manifest["task_id"], manifest["simulator"]["suite"], manifest["task"]) == (3, "libero_goal", DRAWER.prompt)
    assert manifest["distance_task"] == DRAWER.key
    assert manifest["task_spec"]["allow_upward_reobservation"] is False
    assert manifest["max_policy_steps"] is manifest["max_api_calls"] is manifest["max_wall_time_s"] is None
    assert "libero_goal_t03_i00" in FakeRecorder.instances[-1].kwargs["episode_id"]
    trace = rows(output / "trace.jsonl")
    actions = [row for row in trace if row["event"] == "action"]
    assert [row["diagnostics"]["routing"]["target_region"] for row in actions] == ["drawer_handle"] * 5 + ["drawer_interior"]
    assert all(not row["diagnostics"]["local_recovery_active"] for row in actions)
    assert "MUST_NOT_ENTER_POLICY" not in json.dumps(transport.calls)
    assert all(call["reasoning"] == {"effort": "medium"} for call in transport.calls)
