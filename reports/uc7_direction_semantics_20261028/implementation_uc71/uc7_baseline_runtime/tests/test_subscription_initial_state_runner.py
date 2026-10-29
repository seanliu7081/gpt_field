"""Selected LIBERO layouts and frozen overrides stay evaluator-side only."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from gpt_field.libero import validate_initial_state_override
from test_bowl_demo_runner import BOWL, DIRECT_OPTIONS, HYBRID_OPTIONS, bowl_reference
from test_demo_direct_subscription import prediction as direct_prediction
from test_distance_policy import intent
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("episode_index", [0, 7, 49])
def test_cli_preflight_reports_selected_initial_state_without_starting_anything(tmp_path, capsys, monkeypatch,
                                                                               direct, episode_index):
    path, demo = bowl_reference(tmp_path)
    output = tmp_path / "not-created"
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("Preflight launched simulator"))
    mode = ["--controller-mode", "direct"] if direct else ["--distance-field-control"]
    assert runner.main(["--demonstration", str(path), "--episode-index", str(episode_index),
                        "--preflight", "--output", str(output), *mode]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["episode_index"] == episode_index and preview["task_id"] == BOWL.task_id
    assert preview["demonstration"] == {"artifact": "demonstration.json", **demo.metadata()}
    assert not preview["credentials_checked"] and not preview["simulator_started"]
    assert preview["api_calls"] == 0 and not preview["files_written"] and not output.exists()
    assert "initial_state_override" not in preview


@pytest.mark.parametrize("episode_index", [-1, 50, True, False, 1.0, "7", None])
@pytest.mark.parametrize("preflight", [False, True])
def test_programmatic_invalid_indices_fail_before_output_credentials_or_simulator(tmp_path, monkeypatch,
                                                                                 episode_index, preflight):
    output = tmp_path / "not-created"
    monkeypatch.setattr(runner, "SimProcess", lambda *a, **k: pytest.fail("Invalid index launched simulator"))
    if preflight:
        call = lambda: runner.preflight(runner.configuration(), sim_config(), output, episode_index=episode_index)
    else:
        call = lambda: runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
            episode_index=episode_index, token_provider=lambda: pytest.fail("Invalid index requested credentials"))
    with pytest.raises(ValueError, match="episode_index"):
        call()
    assert not output.exists()


@pytest.mark.parametrize("value", ["-1", "50", "1.5", "true"])
def test_cli_invalid_index_exits_before_credentials_configuration_or_output(tmp_path, monkeypatch, value):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("Invalid index reached configuration"))
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--episode-index", value, "--output", str(output)])
    assert stopped.value.code == 2 and not output.exists()


@pytest.mark.parametrize("direct", [False, True])
def test_both_methods_reset_selected_index_and_keep_metadata_video_and_observation_consistent(
        tmp_path, observation, direct):
    demo_path, demo = bowl_reference(tmp_path)
    output = tmp_path / "selected-index"
    base, instances = fake_simulator(observation, success_at=1)
    resets = []
    selected = replace(observation, state=(.137, .021, 1., 0., 0., 0., .04, -.04))

    class SelectedSimulator(base):
        def reset(self, task_id, episode_index):
            resets.append((task_id, episode_index))
            assert (task_id, episode_index) == (0, 17)
            assert self.config.initial_state_override is None
            super().reset(0, 0)
            self.obs = selected
            self.status.update(task_id=task_id, episode_index=episode_index)
            return self.obs

    raw = direct_prediction(phase="grasp", target="black_bowl") if direct else intent(direct=True, phase="grasp")
    transport = FakeTransport(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}))
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        episode_index=17, demonstration_path=demo_path, token_provider=lambda: "offline-oauth",
        transport=transport, sim_factory=SelectedSimulator, recorder_factory=FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert result["status"] == "success" and result["episode_index"] == 17
    assert resets == [(0, 17)] and instances[0].closed
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    assert manifest["episode_index"] == 17 and manifest["task_id"] == 0
    for filename in ("summary.json", "runner_summary.json", "run_state.json"):
        assert json.loads((output / filename).read_text())["episode_index"] == 17
    assert rows(output / "episodes.jsonl")[0]["episode_index"] == 17
    reset = rows(output / "trace.jsonl")[0]
    assert (reset["task_id"], reset["episode_index"]) == (0, 17)
    assert reset["sim_status"]["episode_index"] == 17
    assert reset["observation"] == selected.to_dict()
    assert "libero_spatial_t00_i17" in FakeRecorder.instances[-1].kwargs["episode_id"]
    payload = transport.calls[0]
    assert payload["input"][0] == demo.message()
    live = json.loads(payload["input"][1]["content"][0]["text"])
    assert live["observation/state"] == list(selected.state)
    assert "episode_index" not in live and "MUST_NOT_ENTER_POLICY" not in json.dumps(payload)


def test_default_index_and_explicit_zero_have_identical_preflight_payloads():
    implicit = runner.preflight(runner.configuration(), sim_config())
    explicit = runner.preflight(runner.configuration(), sim_config(), episode_index=0)
    assert implicit == explicit and implicit["episode_index"] == 0


def test_legacy_timing_comparison_rejects_unthreaded_nondefault_layouts_before_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "configuration", lambda: pytest.fail("Unsupported layout reached configuration"))
    for extra in (["--episode-index", "7"], ["--initial-state-override", str(tmp_path / "state.npy")]):
        with pytest.raises(SystemExit) as stopped:
            runner.main(["--timing-comparison", "--output", str(tmp_path / "not-created"), *extra])
        assert stopped.value.code == 2
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("direct", [False, True])
def test_custom_override_is_validated_and_frozen_before_simulator_construction(tmp_path, observation, direct):
    demo_path, demo = bowl_reference(tmp_path)
    source_path = tmp_path / "selected_custom.npy"
    np.save(source_path, np.arange(18, dtype=np.float64) / 100.)
    source_bytes = source_path.read_bytes()
    source = validate_initial_state_override(source_path)
    output = tmp_path / "custom-run"
    base, instances = fake_simulator(observation, success_at=1)

    class OverrideSimulator(base):
        def __init__(self, config):
            frozen = output / "initial_state_override.npy"
            assert config.initial_state_override == str(frozen.resolve())
            assert frozen.read_bytes() == source_bytes
            # An external source change after freezing cannot affect this run.
            source_path.write_bytes(b"source changed after validated snapshot")
            super().__init__(config)

        def reset(self, task_id, episode_index):
            assert (task_id, episode_index) == (0, 9)
            super().reset(0, 0)
            self.status.update(task_id=task_id, episode_index=episode_index, initial_state_source="custom_override",
                initial_state_override_file=self.config.initial_state_override,
                initial_state_override_sha256=source["sha256"])
            return self.obs

    raw = direct_prediction(phase="grasp", target="black_bowl") if direct else intent(direct=True, phase="grasp")
    transport = FakeTransport(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}))
    result = runner.execute_episode(output, config=runner.configuration(),
        sim_config=sim_config(initial_state_override=str(source_path)), episode_index=9,
        demonstration_path=demo_path, token_provider=lambda: "offline-oauth", transport=transport,
        sim_factory=OverrideSimulator, recorder_factory=FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    assert result["status"] == "success" and instances[0].closed
    frozen = output / "initial_state_override.npy"
    record = {"artifact": frozen.name, **{key: source[key] for key in ("sha256", "shape", "dtype")}}
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    for metadata in (result, manifest):
        assert metadata["benchmark_variant"] == "custom_layout"
        assert metadata["initial_state_override"] == record
        assert metadata["initial_state_override_source"] == source
    assert manifest["simulator"]["initial_state_override"] == str(frozen.resolve())
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == source["sha256"]
    reset = rows(output / "trace.jsonl")[0]
    assert reset["episode_index"] == 9 and reset["sim_status"]["initial_state_source"] == "custom_override"
    assert reset["sim_status"]["initial_state_override_sha256"] == source["sha256"]
    assert "i09" in FakeRecorder.instances[-1].kwargs["episode_id"]
    assert FakeRecorder.instances[-1].kwargs["episode_id"].endswith(" / CUSTOM LAYOUT")
    assert transport.calls[0]["input"][0] == demo.message()
    assert source["sha256"] not in json.dumps(transport.calls)
    assert "initial_state_override" not in json.dumps(transport.calls)


def test_override_cli_preflight_checks_source_but_creates_no_snapshot(tmp_path, capsys):
    path = tmp_path / "state.npy"
    np.save(path, np.zeros(12, dtype=np.float32))
    source = validate_initial_state_override(path)
    output = tmp_path / "not-created"
    assert runner.main(["--preflight", "--episode-index", "11", "--initial-state-override", str(path),
                        "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["episode_index"] == 11 and result["benchmark_variant"] == "custom_layout"
    assert result["initial_state_override_source"] == source
    assert result["initial_state_override"]["sha256"] == source["sha256"]
    assert not output.exists() and not result["files_written"]


def test_invalid_override_never_creates_output_or_requests_credentials(tmp_path, monkeypatch):
    source = tmp_path / "invalid.npy"
    source.write_bytes(b"not a state array")
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        runner.execute_episode(output, config=runner.configuration(),
            sim_config=sim_config(initial_state_override=str(source)),
            token_provider=lambda: pytest.fail("Invalid override requested credentials"))
    import gpt_field.chatgpt_auth as auth
    monkeypatch.setattr(auth, "SubscriptionCredentials", lambda *a, **k: pytest.fail("Invalid override checked credentials"))
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--initial-state-override", str(source), "--output", str(output)])
    assert stopped.value.code == 2 and not output.exists()


def test_override_changed_between_validation_and_snapshot_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "state.npy"
    np.save(path, np.zeros(12, dtype=np.float64))
    actual = runner.validate_initial_state_override

    def mutate_after_validation(candidate):
        result = actual(candidate)
        np.save(path, np.ones(12, dtype=np.float64))
        return result

    monkeypatch.setattr(runner, "validate_initial_state_override", mutate_after_validation)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="changed during validation"):
        runner.preflight(runner.configuration(), sim_config(initial_state_override=str(path)), output)
    assert not output.exists()
