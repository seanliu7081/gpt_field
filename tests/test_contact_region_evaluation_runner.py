"""Exercise the real evaluation entry point without inference or a simulator."""
import json

import pytest

from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation
from test_speed_first_v4 import intent


OPTS = dict(OPTIONS, operation_control=True, pregrasp_feedback=True, attempt_memory=True)


def test_apply_and_shadow_preflight_use_identical_model_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("started simulator"))
    payloads = []
    for mode in ("apply", "shadow"):
        output = tmp_path / mode
        assert runner.main(["--contact-region-feedback", mode, "--task", "drawer_bowl",
            "--preflight", "--output", str(output)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert not output.exists()
        assert not any(result[k] for k in ("credentials_checked", "simulator_started", "files_written"))
        assert result["operation_control"]["contact_region_mode"] == mode
        assert result["pregrasp_feedback"]["region_anchors_are_contact_measurements"] is False
        assert result["policy"]["field_reuse_limit"] == 60
        assert result["policy"]["direct_action_steps"] == 5
        assert result["payload"]["reasoning"] == {"effort": "medium"}
        assert result["payload"]["model"] == "gpt-6-astra"
        assert result["billing"] == "chatgpt_subscription" and not result["api_key_fallback"]
        assert result["success_qualification"] == "raw_libero_goal"
        assert result["simulator"]["render_size"] == result["simulator"]["image_size"] == 512
        assert all(result[k] is None for k in ("max_wall_time_s", "max_api_calls", "max_policy_steps"))
        payloads.append(result["payload"])
    assert payloads[0] == payloads[1]


@pytest.mark.parametrize("flags", [["--controller-mode", "direct"], ["--offline-smoke"],
    ["--stop-on-model-complete"], ["--reasoning-effort", "high"], ["--demonstration", "absent.json"]])
def test_region_protocol_rejects_incompatible_flags_before_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("configuration reached"))
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--contact-region-feedback", "apply", "--output", str(output), *flags])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("mode", [False, True, "off", "invalid", [], 1])
def test_bad_region_mode_rejected(mode):
    with pytest.raises(ValueError):
        runner._validate_contact_region_mode(mode, True)
    with pytest.raises(ValueError):
        runner._validate_contact_region_mode("apply", False)


@pytest.mark.parametrize("mode", ["apply", "shadow"])
def test_real_runner_preserves_direct_actions_success_and_timing(mode, tmp_path, highres_observation):
    raw = intent("direct", direct=True)
    actions = [[.01*(i+1), -.02, .03, 0., 0., 0., 1.] for i in range(5)]
    raw["direct"]["actions"] = actions
    simulator, instances = monitored_simulator(highres_observation, success_at=5, collision_at=1)
    fake = FakeTransport(envelope(raw, id="offline-region", model="gpt-6-astra",
        reasoning={"effort": "medium"}))
    result = runner.execute_episode(tmp_path / mode, config=runner.configuration(),
        sim_config=sim_config(render_size=512, image_size=512,
            collision_check_mode="task_manipulation_v1", stop_on_collision=False),
        token_provider=lambda: "OFFLINE_OAUTH", transport=fake, sim_factory=simulator,
        recorder_factory=FakeRecorder, contact_region_mode=mode, **OPTS)
    assert result["status"] == "success" and result["libero_success"]
    assert result["steps"] == 5 and instances[0].actions == actions
    assert len(fake.calls) == 1 and result["api_key_calls"] == 0 and instances[0].closed
    assert result["collision"]["detected"] and result["failure_reason"] is None
    assert result["control_without_video_s"] == pytest.approx(
        result["control_elapsed_s"] - result["timing_components"]["video_render_s"])
    assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(fake.calls)
    manifest = json.loads((tmp_path / mode / "manifest.json").read_text())["configuration"]
    state = json.loads((tmp_path / mode / "run_state.json").read_text())
    assert manifest["policy"]["class"] == "gpt_field.contact_region_policy.ContactRegionPolicy"
    for record in (manifest, manifest["policy"], state, result):
        assert record["operation_control"]["contact_region_mode"] == mode
    assert "contact_region_diagnostics" in result and "contact_region_diagnostics" in state
    events = [json.loads(line) for line in (tmp_path / mode / "trace.jsonl").read_text().splitlines()]
    assert sum(e.get("event") == "contact_region_feedback" for e in events) == 5
