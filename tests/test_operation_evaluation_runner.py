"""Offline end-to-end runner boundaries for the opt-in operation protocol."""
import json

import pytest

from gpt_field.speed_first_v4 import ChatGPTPlanSpeedFirstV4Client
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation
from test_speed_first_v4 import intent


OPTS = dict(OPTIONS, operation_control=True, pregrasp_feedback=True, attempt_memory=True)


@pytest.mark.parametrize("planner", ["field", "waypoint"])
def test_operation_preflight_preserves_approved_protocol(planner, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV4Client, "predict",
                        lambda *a, **k: pytest.fail("called model"))
    output = tmp_path / "absent"
    assert runner.main(["--operation-control", "--operation-planner", planner,
        "--task", "drawer_bowl", "--preflight", "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert result["operation_control"]["planner_mode"] == planner
    assert result["speed_first_protocol"] == "operation-field-hybrid-v4"
    assert result["policy"]["prompt_version"].startswith("operation-field-hybrid-v4")
    assert result["policy"]["field_reuse_limit"] == 60
    assert result["policy"]["direct_action_steps"] == 5
    assert result["payload"]["reasoning"] == {"effort": "medium"}
    assert result["payload"]["model"] == "gpt-6-astra"
    assert result["task_id"] == 3 and result["episode_index"] == 0
    assert result["simulator"]["seed"] == 7
    assert result["simulator"]["render_size"] == result["simulator"]["image_size"] == 512
    assert not result["simulator"]["stop_on_collision"]
    assert result["simulator"]["stop_on_success"] and not result["stop_on_model_complete"]
    assert result["success_qualification"] == "raw_libero_goal"
    assert all(result[k] is None for k in ("max_wall_time_s", "max_api_calls", "max_policy_steps"))
    assert result["billing"] == "chatgpt_subscription" and not result["api_key_fallback"]
    assert result["api_calls"] == result["api_key_calls"] == 0
    assert not any(result[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    schema = result["payload"]["text"]["format"]["schema"]["properties"]
    assert "operation" in schema and "visual_intent" in schema and "control_context" not in schema


@pytest.mark.parametrize("flags", [["--controller-mode", "direct"], ["--offline-smoke"],
    ["--stop-on-model-complete"], ["--reasoning-effort", "high"], ["--demonstration", "absent.json"]])
def test_operation_wrong_protocol_rejected_without_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("configuration reached"))
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--operation-control", "--output", str(output), *flags])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("planner", ["field", "waypoint"])
def test_real_runner_preserves_direct_five_raw_success_and_timing(planner, tmp_path, highres_observation):
    raw = intent("direct", direct=True)
    actions = [[.01*(i+1), -.02, .03, 0., 0., 0., 1.] for i in range(5)]
    raw["direct"]["actions"] = actions
    simulator, instances = monitored_simulator(highres_observation, success_at=5, collision_at=1)
    fake = FakeTransport(envelope(raw, id="offline-operation", model="gpt-6-astra",
                                  reasoning={"effort": "medium"}))
    result = runner.execute_episode(tmp_path / planner, config=runner.configuration(),
        sim_config=sim_config(render_size=512, image_size=512,
            collision_check_mode="task_manipulation_v1", stop_on_collision=False),
        token_provider=lambda: "OFFLINE_OAUTH", transport=fake, sim_factory=simulator,
        recorder_factory=FakeRecorder, operation_planner=planner, **OPTS)
    assert result["status"] == "success" and result["libero_success"]
    assert result["steps"] == 5 and instances[0].actions == actions
    assert len(fake.calls) == 1 and result["api_key_calls"] == 0 and instances[0].closed
    assert result["collision"]["detected"] and result["failure_reason"] is None
    assert result["control_without_video_s"] == pytest.approx(
        result["control_elapsed_s"] - result["timing_components"]["video_render_s"])
    assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)
    manifest = json.loads((tmp_path / planner / "manifest.json").read_text())["configuration"]
    state = json.loads((tmp_path / planner / "run_state.json").read_text())
    for record in (manifest, manifest["policy"], state, result):
        assert record["operation_control"]["planner_mode"] == planner
    assert "operation_diagnostics" in result and "operation_diagnostics" in state
    events = [json.loads(x) for x in (tmp_path / planner / "trace.jsonl").read_text().splitlines()]
    assert sum(e.get("event") == "operation_feedback" for e in events) == 5
