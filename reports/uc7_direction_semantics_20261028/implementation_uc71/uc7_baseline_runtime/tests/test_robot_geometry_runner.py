"""Geometry-only stage: protocol guards and fake execution, never a live trial."""
import copy
import hashlib
import json

import pytest

from gpt_field.robot_geometry_guidance import (
    ROBOT_GEOMETRY_GUIDANCE_VERSION,
    robot_geometry_instructions,
    robot_geometry_metadata,
)
from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client, SPEED_FIRST_V3_VERSION
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation, v3_intent


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_geometry_preflight_changes_only_prompt_and_additive_metadata(task, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("preflight started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "predict",
                        lambda *a, **k: pytest.fail("preflight called model"))
    output = tmp_path / "not_created"
    args = ["--speed-first-v3", "--task", task, "--preflight", "--output", str(output)]
    assert runner.main(args) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert runner.main([*args, "--robot-geometry-guidance"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    metadata = robot_geometry_metadata()
    assert metadata["version"] == ROBOT_GEOMETRY_GUIDANCE_VERSION
    assert metadata["source"] == "known_robot_assets_only"
    assert metadata["text_sha256"] == hashlib.sha256(robot_geometry_instructions().encode()).hexdigest()
    assert enabled["robot_geometry_guidance"] == enabled["policy"]["robot_geometry_guidance"] == metadata
    assert "robot_geometry_guidance" not in baseline and "robot_geometry_guidance" not in baseline["policy"]
    assert enabled["policy"]["prompt_version"] == SPEED_FIRST_V3_VERSION + "+" + ROBOT_GEOMETRY_GUIDANCE_VERSION
    assert enabled["payload"]["instructions"].count(robot_geometry_instructions()) == 1
    assert not output.exists()
    assert enabled["api_calls"] == enabled["api_key_calls"] == 0
    assert not any(enabled[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    # The configured sensor interface, schema, planner, execution, history,
    # completion rule, model and effort are identical to the existing v3 profile.
    comparable = copy.deepcopy(enabled)
    comparable.pop("robot_geometry_guidance")
    comparable["policy"].pop("robot_geometry_guidance")
    comparable["policy"]["prompt_version"] = baseline["policy"]["prompt_version"]
    comparable["payload"]["instructions"] = baseline["payload"]["instructions"]
    assert comparable == baseline


@pytest.mark.parametrize("flags", [
    [], ["--speed-first"], ["--speed-first-v2"],
    ["--controller-mode", "direct"],
    ["--historical-zero-shot-direct", "--controller-mode", "direct"],
    ["--speed-first-v3", "--controller-mode", "direct"],
    ["--speed-first-v3", "--offline-smoke"],
])
def test_geometry_cli_rejects_unsupported_protocol_before_side_effects(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid protocol reached configuration"))
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("invalid protocol started simulator"))
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--robot-geometry-guidance", "--output", str(output), *flags])
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_geometry_api_requires_boolean_before_any_execution(value, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_validate_collision_configuration",
                        lambda *a, **k: pytest.fail("invalid option passed validation"))
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="Robot geometry guidance"):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
                               robot_geometry_guidance=value, **OPTIONS)
    assert not output.exists()


def test_geometry_fake_episode_preserves_direct_actions_and_records_provenance(tmp_path, highres_observation):
    raw = v3_intent(direct=True, phase="verify")
    raw["direct"]["actions"] = [[.01 * (i + 1), -.02, .03, 0., 0., 0., 1.] for i in range(5)]
    records = []
    for enabled in (False, True):
        output = tmp_path / ("geometry" if enabled else "baseline")
        simulator, instances = monitored_simulator(highres_observation, success_at=5, collision_at=1)
        transport = FakeTransport(envelope(raw, id="offline-geometry-fixture", model="gpt-6-astra",
                                           reasoning={"effort": "medium"}))
        result = runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(render_size=512, image_size=512,
                collision_check_mode="task_manipulation_v1", stop_on_collision=False),
            token_provider=lambda: "OFFLINE_FIXTURE_OAUTH", transport=transport,
            sim_factory=simulator, recorder_factory=FakeRecorder,
            robot_geometry_guidance=enabled, **OPTIONS)
        assert result["status"] == "success" and result["steps"] == 5
        assert result["libero_success"] is True and result["model_declared_complete"] is False
        assert result["ordinary_direct_actions"] == 5 and result["local_gripper_actions"] == 0
        assert result["coarse_field_actions"] == 0 and result["api_key_calls"] == 0
        assert len(transport.calls) == 1 and instances[0].closed
        assert instances[0].actions == raw["direct"]["actions"]
        assert result["collision"]["detected"] and result["failure_reason"] is None
        assert result["control_without_video_s"] == pytest.approx(
            result["control_elapsed_s"] - result["timing_components"]["video_render_s"])
        manifest = json.loads((output / "manifest.json").read_text())["configuration"]
        state = json.loads((output / "run_state.json").read_text())
        for document in (manifest, manifest["policy"], state, result):
            if enabled:
                assert document["robot_geometry_guidance"] == robot_geometry_metadata()
            else:
                assert "robot_geometry_guidance" not in document
        trace = rows(output / "trace.jsonl")
        assert [r["step"] for r in trace if r["event"] == "feedback"] == list(range(5))
        request = transport.calls[0]
        assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(request)
        records.append(request)
    changed = copy.deepcopy(records[1])
    changed["instructions"] = records[0]["instructions"]
    assert changed == records[0]
