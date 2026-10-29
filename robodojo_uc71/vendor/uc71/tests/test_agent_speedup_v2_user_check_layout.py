"""Custom-layout approval boundaries; fake RGB only, no simulator or network."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_user_check import ReviewError
from test_agent_speedup_v2_user_check_runner import DraftClient, TASK, runner, settings


def custom_settings(tmp_path):
    config, sim_config, monitor = settings()
    path = tmp_path / "layout.npy"
    np.save(path, np.arange(16, dtype=float), allow_pickle=False)
    return config, replace(sim_config, initial_state_override=str(path)), monitor


def prepare_layout(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    fake = FakeV2Sim(prompt=TASK.prompt)
    expected = runner.initial_observation_sha256(fake.wrist.observation().to_dict())
    DraftClient.requests = []
    store = runner.prepare_review(
        tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, client_factory=DraftClient,
        sim_factory=lambda _: fake, token_provider=lambda: "offline",
        expected_initial_observation_sha256=expected, video=False)
    return store, fake, config, sim_config, monitor


def execution_kwargs(store, config, sim_config, monitor):
    return dict(review_dir=store.path, config=config, sim_config=sim_config,
                task_key=TASK.key, episode_index=0, monitor_options=monitor,
                token_provider=lambda: "offline", video=False)


def test_base_stays_official_only_and_opt_in_does_not_allow_bddl(tmp_path):
    config, sim_config, _ = custom_settings(tmp_path)
    with pytest.raises(ValueError, match="official LIBERO"):
        runner.base.validate_configuration(config, sim_config, TASK)
    runner.base.validate_configuration(config, sim_config, TASK, allow_initial_state_override=True)
    with pytest.raises(ValueError, match="official LIBERO"):
        runner.base.validate_configuration(
            config, replace(sim_config, task_bddl_override="not-supported.bddl"), TASK,
            allow_initial_state_override=True)


def test_missing_custom_preview_hash_fails_before_preparation_side_effects(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    constructed = []
    with pytest.raises(ReviewError, match="from its preview"):
        runner.prepare_review(
            tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
            episode_index=0, monitor_options=monitor,
            sim_factory=lambda _: constructed.append(True))
    assert not constructed
    assert not (tmp_path / "review").exists()


def test_base_custom_execute_requires_preview_hash_even_with_injected_sim(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    constructed = []
    with pytest.raises(ValueError, match="explicit expected_initial_observation_sha256"):
        runner.base.execute(
            tmp_path / "run", config=config, sim_config=sim_config, task_key=TASK.key,
            episode_index=0, monitor_options=monitor, allow_initial_state_override=True,
            sim_factory=lambda _: constructed.append(True))
    assert not constructed
    assert not (tmp_path / "run").exists()


def test_custom_execution_archives_exact_layout_and_labels_manifest(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    fake = FakeV2Sim(prompt=TASK.prompt)
    expected = runner.initial_observation_sha256(fake.wrist.observation().to_dict())
    output = tmp_path / "run"
    runner.base.execute(
        output, config=config, sim_config=sim_config, task_key=TASK.key, episode_index=0,
        monitor_options=monitor, allow_initial_state_override=True, sim_factory=lambda _: fake,
        expected_initial_observation_sha256=expected, smoke=True, smoke_steps=1, video=False)
    manifest = json.loads((output / "manifest.json").read_text())["configuration"]
    archived = output / "frozen_inputs/initial_state.npy"
    assert archived.read_bytes() == Path(sim_config.initial_state_override).read_bytes()
    assert manifest["benchmark_variant"] == "custom_initial_layout"
    assert manifest["initial_layout"]["sha256"] == hashlib.sha256(archived.read_bytes()).hexdigest()
    assert manifest["expected_initial_observation_sha256"] == expected


def test_preparation_freezes_layout_without_policy_actions_and_content_changes_invalidate(tmp_path):
    store, fake, config, sim_config, monitor = prepare_layout(tmp_path)
    review = store.load()
    execution = review["preparation"]["execution"]
    source = Path(sim_config.initial_state_override)
    frozen = store.path / "frozen_inputs/initial_state.npy"
    assert frozen.read_bytes() == source.read_bytes()
    assert execution["initial_layout"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert execution["benchmark_variant"] == "custom_initial_layout"
    assert review["preparation"]["policy_actions"] == 0
    assert fake.actions == []
    store.approve(expected_revision=review["revision"], reviewed_steps=[0])
    np.save(source, np.arange(16, dtype=float) + 1, allow_pickle=False)
    with pytest.raises(ReviewError, match="Runtime code changed"):
        runner.execute_reviewed(tmp_path / "run", **execution_kwargs(store, config, sim_config, monitor))
    assert store.load()["status"] == "approved"
    assert not (tmp_path / "run").exists()


def test_changed_frozen_layout_is_rejected_before_claim(tmp_path):
    store, _, config, sim_config, monitor = prepare_layout(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    (store.path / "frozen_inputs/initial_state.npy").write_bytes(b"changed")
    with pytest.raises(ReviewError, match="Frozen custom initial state changed"):
        runner.execute_reviewed(tmp_path / "run", **execution_kwargs(store, config, sim_config, monitor))
    assert store.load()["status"] == "approved"
    assert not (tmp_path / "run").exists()


def test_custom_reset_mismatch_never_requests_model_or_action(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    fake = FakeV2Sim(prompt=TASK.prompt)
    DraftClient.requests = []
    with pytest.raises(ReviewError, match="no model call performed"):
        runner.prepare_review(
            tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
            episode_index=0, monitor_options=monitor, client_factory=DraftClient,
            sim_factory=lambda _: fake, expected_initial_observation_sha256="0" * 64)
    assert DraftClient.requests == []
    assert fake.actions == [] and fake.closed


def test_reviewed_custom_execution_preserves_reset_binding_and_single_use(tmp_path, monkeypatch):
    store, _, config, sim_config, monitor = prepare_layout(tmp_path)
    approved = store.approve(expected_revision=0, reviewed_steps=[0])
    calls = []
    monkeypatch.setattr(runner.base, "execute", lambda output, **kw: calls.append(kw) or {"stub": True})
    monkeypatch.setattr(runner, "_normalize_artifacts", lambda output, summary, *args, **kw: summary)
    assert runner.execute_reviewed(
        tmp_path / "run", **execution_kwargs(store, config, sim_config, monitor)) == {"stub": True}
    assert len(calls) == 1
    assert calls[0]["sim_config"].initial_state_override == sim_config.initial_state_override
    assert calls[0]["expected_initial_observation_sha256"] == approved["initial_observation_sha256"]
    assert calls[0]["allow_initial_state_override"] is True
    assert store.load()["status"] == "consumed"
    with pytest.raises(ReviewError, match="fresh user-approved review"):
        runner.execute_reviewed(tmp_path / "another", **execution_kwargs(store, config, sim_config, monitor))
    assert len(calls) == 1


def test_run_cli_restores_custom_layout_from_review_without_new_flags(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth

    store, _, _, sim_config, _ = prepare_layout(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    gates, calls = [], []
    monkeypatch.setattr(runner, "_live_gate", lambda *args: gates.append(args))

    class NoCredentials:
        def __init__(self, **kwargs):
            pass

        def access_token(self):
            pytest.fail("CLI wiring test must not read credentials")

    monkeypatch.setattr(auth, "SubscriptionCredentials", NoCredentials)

    def execute(output, **kwargs):
        calls.append(kwargs)
        return {"status": "stub", "output": str(output), "evaluation_completed": True,
                "user_check": {"review_id": store.load()["id"]}}

    monkeypatch.setattr(runner, "execute_reviewed", execute)
    assert runner.main(["run", "--review-dir", str(store.path),
                        "--output", str(tmp_path / "run")]) == 0
    assert calls[0]["sim_config"].initial_state_override == sim_config.initial_state_override
    assert calls[0]["video"] is False
    assert gates[0][-1] == store.load()["initial_observation_sha256"]


def test_custom_preflight_command_keeps_layout_and_preview_hash(tmp_path):
    config, sim_config, monitor = custom_settings(tmp_path)
    result = runner.preflight(
        config, sim_config, task_key=TASK.key, episode_index=0, monitor_options=monitor,
        expected_initial_observation_sha256="a" * 64)
    assert result["benchmark_variant"] == "custom_initial_layout"
    assert result["frozen_reset_sha256"] == "a" * 64
    assert "--initial-state-override" in result["live_command"]
    assert sim_config.initial_state_override in result["live_command"]
    assert "--expected-initial-observation-sha256 " + "a" * 64 in result["live_command"]
    assert result["simulator_started"] is False
    assert result["api_calls"] == 0
