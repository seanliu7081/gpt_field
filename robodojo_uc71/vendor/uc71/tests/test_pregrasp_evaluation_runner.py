"""Pregrasp evaluation wiring using fake responses/feedback only."""
import copy
import json

import pytest

from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation, v3_intent


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_pregrasp_preflight_is_opt_in_and_retains_protocol(task, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "predict",
                        lambda *a, **k: pytest.fail("called inference"))
    output = tmp_path / "must_not_exist"
    args = ["--speed-first-v3", "--task", task, "--preflight", "--output", str(output)]
    assert runner.main(args) == 0
    original = json.loads(capsys.readouterr().out)
    assert runner.main([*args, "--pregrasp-feedback"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    metadata = runner._pregrasp_metadata(True)["pregrasp_feedback"]
    assert enabled["pregrasp_feedback"] == enabled["policy"]["pregrasp_feedback"] == metadata
    assert metadata["extra_model_calls"] == 0 and not metadata["automatic_probe"]
    assert not metadata["automatic_close"] and not metadata["task_success_certified"]
    assert not output.exists()
    assert enabled["api_calls"] == enabled["api_key_calls"] == 0
    assert not any(enabled[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    context = enabled["payload"]["text"]["format"]["schema"]["properties"]["control_context"]
    assert "pregrasp_reference" in context["properties"]
    assert "pregrasp_reference" in context["required"]
    # The deliberate differences are opt-in prompt/schema/version/metadata.
    normalized = copy.deepcopy(enabled)
    normalized.pop("pregrasp_feedback")
    normalized["policy"].pop("pregrasp_feedback")
    normalized["policy"]["prompt_version"] = original["policy"]["prompt_version"]
    normalized["payload"]["instructions"] = original["payload"]["instructions"]
    normalized["payload"]["text"] = original["payload"]["text"]
    assert normalized == original


@pytest.mark.parametrize("flags", [[], ["--speed-first"], ["--speed-first-v2"],
    ["--speed-first-v3", "--controller-mode", "direct"],
    ["--speed-first-v3", "--offline-smoke"]])
def test_pregrasp_rejects_wrong_cli_protocol_before_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("configuration reached"))
    output = tmp_path / "must_not_exist"
    with pytest.raises(SystemExit) as error:
        runner.main(["--pregrasp-feedback", "--output", str(output), *flags])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_pregrasp_requires_boolean_before_episode_creation(value, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_validate_collision_configuration",
                        lambda *a, **k: pytest.fail("invalid flag escaped validation"))
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="Pregrasp feedback"):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
                               pregrasp_feedback=value, **OPTIONS)
    assert not output.exists()


def test_enabled_null_reference_keeps_direct_actions_and_records_zero_use(tmp_path, highres_observation):
    raw = v3_intent(direct=True, phase="verify")
    raw["direct"]["actions"] = [[.01 * (i + 1), -.02, .03, 0., 0., 0., 1.] for i in range(5)]
    results = []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        simulator, instances = monitored_simulator(highres_observation, success_at=5, collision_at=1)
        response = copy.deepcopy(raw)
        if enabled:
            response["control_context"]["pregrasp_reference"] = None
        transport = FakeTransport(envelope(response, id="offline-pregrasp-fixture", model="gpt-6-astra",
                                           reasoning={"effort": "medium"}))
        result = runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(render_size=512, image_size=512,
                collision_check_mode="task_manipulation_v1", stop_on_collision=False),
            token_provider=lambda: "OFFLINE_FIXTURE_OAUTH", transport=transport,
            sim_factory=simulator, recorder_factory=FakeRecorder,
            pregrasp_feedback=enabled, **OPTIONS)
        assert result["status"] == "success" and result["steps"] == 5
        assert result["libero_success"] and not result["model_declared_complete"]
        assert instances[0].actions == raw["direct"]["actions"]
        assert len(transport.calls) == 1 and result["api_key_calls"] == 0 and instances[0].closed
        assert result["collision"]["detected"] and result["failure_reason"] is None
        assert result["control_without_video_s"] == pytest.approx(
            result["control_elapsed_s"] - result["timing_components"]["video_render_s"])
        manifest = json.loads((output / "manifest.json").read_text())["configuration"]
        state = json.loads((output / "run_state.json").read_text())
        for document in (manifest, manifest["policy"], state, result):
            if enabled:
                assert document["pregrasp_feedback"] == runner._pregrasp_metadata(True)["pregrasp_feedback"]
            else:
                assert "pregrasp_feedback" not in document
        for document in (state, result):
            if enabled:
                assert document["pregrasp_diagnostics"]["references_requested"] == 0
                assert document["pregrasp_diagnostics"]["applied_goals"] == 0
            else:
                assert "pregrasp_diagnostics" not in document
        assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(transport.calls)
        results.append(instances[0].actions)
    assert results[0] == results[1]
