"""Freeze changed assets and evaluator provenance before subscription inference."""

import hashlib
import json

import numpy as np
import pytest

from gpt_field.libero import validate_task_bddl_override
from test_bowl_demo_runner import DIRECT_OPTIONS, HYBRID_OPTIONS, bowl_reference
from test_demo_direct_subscription import prediction as direct_prediction
from test_distance_policy import intent
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


def scene(tmp_path):
    text = """(define (problem libero_tabletop_manipulation) (:domain robosuite)
      (:language Put bowl on plate) (:objects bowl - bowl plate - plate)
      (:goal (On bowl plate)))"""
    original = tmp_path / "official.bddl"
    original.write_text(text)
    custom = tmp_path / "custom.bddl"
    custom.write_text(text.replace("plate - plate)", "plate - plate milk - milk)"))
    state = tmp_path / "state.npy"
    np.save(state, np.zeros(12, dtype=np.float64))
    return original, custom, state


@pytest.mark.parametrize("direct", [False, True])
def test_custom_scene_preflight_is_read_only(tmp_path, capsys, monkeypatch, direct):
    original, custom, state = scene(tmp_path)
    demo_path, _ = bowl_reference(tmp_path)
    output = tmp_path / "absent"
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("Preflight launched simulator"))
    mode = ["--controller-mode", "direct"] if direct else ["--distance-field-control"]
    assert runner.main(["--preflight", "--demonstration", str(demo_path), "--task-bddl-override",
        str(custom), "--initial-state-override", str(state), "--output", str(output), *mode]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["benchmark_variant"] == "custom_scene"
    assert result["task_bddl_override_source"] == validate_task_bddl_override(custom)
    assert not result["credentials_checked"] and not result["simulator_started"]
    assert not result["files_written"] and not output.exists()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("tamper", [None, "custom", "original"])
def test_scene_snapshot_provenance_and_privacy(tmp_path, observation, direct, tamper):
    original, custom, state = scene(tmp_path)
    original_bytes, custom_bytes, state_bytes = original.read_bytes(), custom.read_bytes(), state.read_bytes()
    source = validate_task_bddl_override(custom)
    demo_path, demo = bowl_reference(tmp_path)
    output = tmp_path / "run"
    base, instances = fake_simulator(observation, success_at=1)

    class SceneSimulator(base):
        def __init__(self, config):
            assert config.task_bddl_override == str((output / "task_override.bddl").resolve())
            assert (output / "task_override.bddl").read_bytes() == custom_bytes
            assert (output / "initial_state_override.npy").read_bytes() == state_bytes
            custom.write_bytes(b"source changed after snapshot")
            super().__init__(config)

        def reset(self, task_id, episode_index):
            super().reset(task_id, episode_index)
            self.status.update(original_bddl_file=str(original),
                original_bddl_sha256=("0" * 64 if tamper == "original" else hashlib.sha256(original_bytes).hexdigest()),
                bddl_sha256=("0" * 64 if tamper == "custom" else source["sha256"]),
                task_bddl_override_sha256=source["sha256"])
            return self.obs

    raw = direct_prediction(phase="grasp", target="black_bowl") if direct else intent(direct=True, phase="grasp")
    transport = FakeTransport(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(task_bddl_override=str(custom), initial_state_override=str(state)),
        demonstration_path=demo_path, token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=SceneSimulator, recorder_factory=FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert instances[0].closed
    if tamper:
        assert result["status"] == "infrastructure_error"
        assert not transport.calls and not instances[0].actions
        return
    assert result["status"] == "success"
    assert (output / "original_task.bddl").read_bytes() == original_bytes
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    for metadata in (result, manifest):
        assert metadata["benchmark_variant"] == "custom_scene"
        assert metadata["task_bddl_override_source"] == source
        assert metadata["task_bddl_override"] == {"artifact": "task_override.bddl",
            **{key: source[key] for key in ("sha256", "byte_count")}}
    assert FakeRecorder.instances[-1].kwargs["episode_id"].endswith(" / CUSTOM SCENE")
    assert transport.calls[0]["input"][0] == demo.message()
    payload = json.dumps(transport.calls)
    for private in (source["sha256"], str(custom), "task_bddl_override", "MUST_NOT_ENTER_POLICY"):
        assert private not in payload


def test_invalid_scene_options_fail_before_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("Invalid scene reached configuration"))
    for options in (["--task-bddl-override", "scene.bddl"],
                    ["--task-bddl-override", "scene.bddl", "--initial-state-override", "state.npy", "--timing-comparison"]):
        with pytest.raises(SystemExit) as stopped:
            runner.main([*options, "--output", str(tmp_path / "absent")])
        assert stopped.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_scene_source_race_rejected_before_any_output(tmp_path, monkeypatch):
    _, custom, state = scene(tmp_path)
    actual = runner.validate_task_bddl_override

    def changed(path):
        metadata = actual(path)
        custom.write_text(custom.read_text() + "\n; changed\n")
        return metadata

    monkeypatch.setattr(runner, "validate_task_bddl_override", changed)
    with pytest.raises(ValueError, match="changed during validation"):
        runner.execute_episode(tmp_path / "absent", config=runner.configuration(),
            sim_config=sim_config(task_bddl_override=str(custom), initial_state_override=str(state)),
            token_provider=lambda: pytest.fail("Invalid snapshot requested credentials"))
    assert not (tmp_path / "absent").exists()
