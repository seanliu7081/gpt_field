"""Optional identity prompts are validated with preflight and fake execution only."""
import copy
import hashlib
import json

import pytest

from gpt_field.robot_geometry_guidance import ROBOT_GEOMETRY_GUIDANCE_VERSION
from gpt_field.speed_first_v3 import (
    ChatGPTPlanSpeedFirstV3Client, SPEED_FIRST_V3_VERSION, speed_first_v3_instructions,
)
from gpt_field.target_identity_guidance import (
    TARGET_IDENTITY_VERSION, normalize_target_description,
    target_identity_instructions, target_identity_metadata,
)
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_speed_first_runner import monitored_simulator
from test_speed_first_v3_runner import OPTIONS, highres_observation, v3_intent


DESCRIPTION = (
    "The target is the separate round bowl with a visible rim, concave opening, and light grey "
    "patterned interior. Distinguish it from the stove, heating ring, and dark appliance components."
)


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_target_preflight_only_adds_prompt_suffix_and_metadata(geometry, task, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("preflight started simulator"))
    monkeypatch.setattr(ChatGPTPlanSpeedFirstV3Client, "predict",
                        lambda *a, **k: pytest.fail("preflight called model"))
    output = tmp_path / "absent"
    args = ["--speed-first-v3", "--task", task, "--preflight", "--output", str(output)]
    if geometry:
        args.append("--robot-geometry-guidance")
    assert runner.main(args) == 0
    baseline = json.loads(capsys.readouterr().out)
    assert runner.main([*args, "--target-description", "  " + DESCRIPTION + "\n"]) == 0
    enabled = json.loads(capsys.readouterr().out)
    metadata = target_identity_metadata(DESCRIPTION)
    assert metadata == {
        "version": TARGET_IDENTITY_VERSION, "description": DESCRIPTION,
        "text_sha256": hashlib.sha256(target_identity_instructions(DESCRIPTION).encode()).hexdigest(),
        "source": "human_RGB_description",
    }
    assert enabled["target_identity_guidance"] == enabled["policy"]["target_identity_guidance"] == metadata
    assert "target_identity_guidance" not in baseline and "target_identity_guidance" not in baseline["policy"]
    assert enabled["payload"]["instructions"] == baseline["payload"]["instructions"] + "\n\n" + target_identity_instructions(DESCRIPTION)
    assert enabled["policy"]["prompt_version"] == baseline["policy"]["prompt_version"] + "+" + TARGET_IDENTITY_VERSION
    assert not output.exists()
    assert enabled["api_calls"] == enabled["api_key_calls"] == 0
    assert not any(enabled[k] for k in ("credentials_checked", "simulator_started", "files_written"))
    comparable = copy.deepcopy(enabled)
    comparable.pop("target_identity_guidance")
    comparable["policy"].pop("target_identity_guidance")
    comparable["policy"]["prompt_version"] = baseline["policy"]["prompt_version"]
    comparable["payload"]["instructions"] = baseline["payload"]["instructions"]
    assert comparable == baseline


@pytest.mark.parametrize("geometry", [False, True])
def test_client_none_keeps_old_payload_and_description_changes_only_instructions(geometry, highres_observation):
    kwargs = {"robot_geometry_guidance": geometry, "token_provider": lambda: pytest.fail("credentials read")}
    baseline = ChatGPTPlanSpeedFirstV3Client(**kwargs)
    explicit_none = ChatGPTPlanSpeedFirstV3Client(target_description=None, **kwargs)
    described = ChatGPTPlanSpeedFirstV3Client(target_description="\t" + DESCRIPTION + "  ", **kwargs)
    original = baseline.build_payload(highres_observation.to_dict())
    assert explicit_none.build_payload(highres_observation.to_dict()) == original
    changed = described.build_payload(highres_observation.to_dict())
    assert changed["instructions"] == original["instructions"] + "\n\n" + target_identity_instructions(DESCRIPTION)
    changed["instructions"] = original["instructions"]
    assert changed == original
    assert described.target_description == DESCRIPTION
    expected = SPEED_FIRST_V3_VERSION + ("+" + ROBOT_GEOMETRY_GUIDANCE_VERSION if geometry else "")
    assert baseline.prompt_version == explicit_none.prompt_version == expected
    assert described.prompt_version == expected + "+" + TARGET_IDENTITY_VERSION


@pytest.mark.parametrize("flags", [
    [], ["--speed-first"], ["--speed-first-v2"], ["--distance-field-control"],
    ["--controller-mode", "direct"],
    ["--historical-zero-shot-direct", "--controller-mode", "direct"],
    ["--speed-first-v3", "--controller-mode", "direct"],
])
def test_target_cli_rejects_non_v3_before_side_effects(flags, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid protocol reached configuration"))
    with pytest.raises(SystemExit) as exc:
        runner.main(["--target-description", DESCRIPTION, "--preflight", *flags])
    assert exc.value.code == 2


@pytest.mark.parametrize("value", ["", " \n\t ", "x" * 1501])
def test_target_cli_rejects_invalid_description_before_side_effects(value, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("invalid description reached configuration"))
    with pytest.raises(SystemExit) as exc:
        runner.main(["--speed-first-v3", "--target-description", value, "--preflight"])
    assert exc.value.code == 2


@pytest.mark.parametrize("value", [False, True, 0, 1, [], {}, "", "\t", "x" * 1501])
def test_invalid_description_rejected_by_helper_client_prompt_and_runner(value, tmp_path, monkeypatch):
    for build in (normalize_target_description, target_identity_instructions, target_identity_metadata):
        with pytest.raises(ValueError, match="target_description"):
            build(value)
    with pytest.raises(ValueError, match="target_description"):
        ChatGPTPlanSpeedFirstV3Client(target_description=value)
    with pytest.raises(ValueError, match="target_description"):
        speed_first_v3_instructions(target_description=value)
    monkeypatch.setattr(runner, "_validate_collision_configuration",
                        lambda *a, **k: pytest.fail("invalid description passed validation"))
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="target_description"):
        runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
                               target_description=value, **OPTIONS)
    assert not output.exists()


def test_description_boundary_and_source_semantics():
    assert normalize_target_description(None) is None
    assert target_identity_instructions(None) == ""
    assert normalize_target_description("  " + "x" * 1500 + "\n") == "x" * 1500
    text = target_identity_instructions(DESCRIPTION)
    for phrase in ("initial RGB observation", "not a live position estimate", "shape", "single color",
                   "same object instance", "temporary occlusion", "unsuccessful grasp",
                   "within the existing model calls", "no mandatory check", "no fixed coordinates",
                   "task-required interactions with other objects"):
        assert phrase in text
    assert text.count(DESCRIPTION) == 1
    with pytest.raises(ValueError, match="enabled description"):
        target_identity_metadata(None)


def test_target_fake_episode_preserves_actions_and_records_metadata(tmp_path, highres_observation):
    raw = v3_intent(direct=True, phase="verify")
    raw["direct"]["actions"] = [[.01 * (i + 1), -.02, .03, 0., 0., 0., 1.] for i in range(5)]
    requests = []
    for description in (None, DESCRIPTION):
        output = tmp_path / ("described" if description else "baseline")
        simulator, instances = monitored_simulator(highres_observation, success_at=5, collision_at=1)
        transport = FakeTransport(envelope(raw, id="offline-target-fixture", model="gpt-6-astra",
                                           reasoning={"effort": "medium"}))
        result = runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(render_size=512, image_size=512,
                collision_check_mode="task_manipulation_v1", stop_on_collision=False),
            token_provider=lambda: "OFFLINE_FIXTURE_OAUTH", transport=transport,
            sim_factory=simulator, recorder_factory=FakeRecorder,
            robot_geometry_guidance=True, target_description=description, **OPTIONS)
        assert result["status"] == "success" and result["steps"] == 5
        assert result["libero_success"] is True and not result["model_declared_complete"]
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
            assert document["robot_geometry_guidance"]["version"] == ROBOT_GEOMETRY_GUIDANCE_VERSION
            if description is not None:
                assert document["target_identity_guidance"] == target_identity_metadata(DESCRIPTION)
            else:
                assert "target_identity_guidance" not in document
        assert [r["step"] for r in rows(output / "trace.jsonl") if r["event"] == "feedback"] == list(range(5))
        assert "CONTACT_MUST_NOT_ENTER_POLICY" not in json.dumps(transport.calls[0])
        requests.append(transport.calls[0])
    changed = copy.deepcopy(requests[1])
    changed["instructions"] = requests[0]["instructions"]
    assert changed == requests[0]
