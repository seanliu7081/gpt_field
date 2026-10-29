"""Bounded local paired tests must not become full-task or privileged policies."""
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gpt_field.libero import LiberoConfig
from test_field_history import FakeTransport, envelope, observation
from test_field_history_runner import FakeRecorder
from test_speed_first_v4 import intent

spec = importlib.util.spec_from_file_location("contact_runner",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_contact_routing_diagnostic.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def wire(index):
    raw = intent("pregrasp", direct=True, identifier=f"pregrasp-{index}")
    return {"decision": {"scope": "contact_alignment", "scope_reason": "Visible hand alignment",
                         "response": raw}}


def checkpoint(tmp_path):
    prefix = tmp_path / "prefix.json"
    prefix.write_text("[]")
    return {"id": "bowl_a", "target": "bowl", "task_key": "bowl_plate",
        "task_id": 0, "episode_index": 0, "prefix_actions_path": str(prefix),
        "expected_fingerprint": {"observation_sha256": "fixture"},
        "sim_config": asdict(LiberoConfig(suite="libero_spatial", image_size=512,
            render_size=512, max_steps=None, collision_check_mode="task_manipulation_v1",
            stop_on_collision=False))}


def simulation(*, success_at=None, raw_success=False, replay_fails=False):
    instances = []
    class Sim:
        def __init__(self, config, **kwargs):
            instances.append(self)
            self.obs = observation()
            self.status = {"terminated": False}
            self.metadata = {"sentinel": "HIDDEN_SIMULATOR_TRUTH"}
            self.checkpoint_check = {"matched": True}
            self.evaluator_metrics = {"criterion_ready": True, "engagement_success": False,
                                      "sentinel": "HIDDEN_SIMULATOR_TRUTH"}
            self.actions = []
            self.closed = False
        def replay_checkpoint(self, task, episode, actions, expected_fingerprint, initial_fingerprint=None):
            if replay_fails:
                raise ValueError("Fingerprint mismatch")
            return self.obs
        def begin_diagnostic(self, fingerprint):
            return self.evaluator_metrics
        def step(self, action):
            self.actions.append(tuple(action))
            self.evaluator_metrics = dict(self.evaluator_metrics,
                engagement_success=success_at is not None and len(self.actions) >= success_at)
            self.obs = replace(self.obs, state=(self.obs.state[0] + .001, *self.obs.state[1:]))
            return self.obs, raw_success
        def close(self):
            self.closed = True
    return Sim, instances


def test_model_budget_finishes_authorized_five_actions_without_extra_query(tmp_path):
    fake = FakeTransport(*(envelope(wire(i), id=f"diag-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i in range(2)))
    sim, instances = simulation(raw_success=True)
    result = runner.run_trial(tmp_path / "trial", checkpoint(tmp_path), "direct5",
        limits={"actions": 100, "model_requests": 2, "completed_close_check_attempts": 2},
        sim_factory=sim, recorder_factory=FakeRecorder, token_provider=lambda: "OFFLINE", transport=fake)
    assert result["status"] == "diagnostic_request_limit", result
    assert result["steps"] == result["feedback_count"] == 10
    assert len(fake.calls) == result["model_requests"] == 2
    assert result["execution_counts"] == {"direct_five_actions": 10}
    assert result["libero_success"] and not result["local_engagement_success"]
    assert not result["task_completion_time_eligible"]
    assert "HIDDEN_SIMULATOR_TRUTH" not in json.dumps(fake.calls)
    assert instances[0].closed
    assert result["control_without_video_s"] == pytest.approx(
        result["control_elapsed_s"] - result["timing_components"]["video_render_s"])


def test_engagement_stop_is_independent_and_does_not_execute_remaining_actions(tmp_path):
    fake = FakeTransport(envelope(wire(0), id="diag", model="gpt-6-astra", reasoning={"effort": "medium"}))
    sim, instances = simulation(success_at=2)
    result = runner.run_trial(tmp_path / "trial", checkpoint(tmp_path), "current",
        sim_factory=sim, recorder_factory=FakeRecorder, token_provider=lambda: "OFFLINE", transport=fake)
    assert result["status"] == "local_engagement_confirmed", result
    assert result["local_engagement_success"] and not result["libero_success"]
    assert result["steps"] == result["feedback_count"] == len(instances[0].actions) == 2
    assert len(fake.calls) == 1


def test_replay_mismatch_prevents_all_inference_and_actions(tmp_path):
    fake = FakeTransport()
    sim, instances = simulation(replay_fails=True)
    result = runner.run_trial(tmp_path / "trial", checkpoint(tmp_path), "current",
        sim_factory=sim, recorder_factory=FakeRecorder, token_provider=lambda: "OFFLINE", transport=fake)
    assert result["status"] == "diagnostic_error"
    assert result["model_requests"] == result["steps"] == 0
    assert result["control_without_video_s"] is None and instances[0].closed
    assert not fake.calls and not instances[0].actions


def test_hidden_physics_mismatch_blocks_second_arm_even_with_same_rgb(tmp_path):
    fake = FakeTransport()
    sim, instances = simulation()
    start = checkpoint(tmp_path)
    start["paired_physics_fingerprint"] = {"combined_sha256": "first-arm-state"}
    result = runner.run_trial(tmp_path / "trial", start, "direct5", sim_factory=sim,
        recorder_factory=FakeRecorder, token_provider=lambda: "OFFLINE", transport=fake)
    assert result["status"] == "diagnostic_error"
    assert result["steps"] == result["model_requests"] == 0
    assert not fake.calls and instances[0].closed
    check = json.loads((tmp_path / "trial/paired_state_check.json").read_text())
    assert check["physics_matches_first_arm"] is False


def test_paired_order_and_diagnostic_limits():
    starts = [{"id": target + suffix, "target": target}
              for target in ("handle", "bowl") for suffix in ("a", "b")]
    assert [row["arm"] for row in runner.trial_order(starts)] == [
        "current", "direct5", "direct5", "current"] * 2
    args = dict(steps=1, requests=8, completed_attempts=0, engagement=False)
    assert runner.stop_reason(**args, needs_model_request=False) is None
    assert runner.stop_reason(**args, needs_model_request=True) == "diagnostic_request_limit"
    args["completed_attempts"] = 2
    assert runner.stop_reason(**args, needs_model_request=False) == "diagnostic_attempt_limit"
    args["engagement"] = True
    assert runner.stop_reason(**args, needs_model_request=True) == "local_engagement_confirmed"


def test_suite_entry_wires_subscription_provider_and_never_falls_back(tmp_path, monkeypatch):
    start = checkpoint(tmp_path)
    manifest = {"checkpoints": [start], "trial_order": runner.trial_order([start])}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "verify_manifest", lambda value: None)
    provider = lambda: "OFFLINE"
    calls = []
    def run(output, checkpoint, arm, *, token_provider):
        calls.append((checkpoint["id"], arm, token_provider))
        return {"status": "diagnostic_error"}
    monkeypatch.setattr(runner, "run_trial", run)
    result = runner.execute(tmp_path, token_provider=provider)
    assert calls == [("bowl_a", "current", provider)]
    assert len(result) == 1


def test_missing_provider_rejected_before_any_output_or_simulator(tmp_path):
    with pytest.raises(ValueError, match="subscription token provider"):
        runner.run_trial(tmp_path / "trial", checkpoint(tmp_path), "current")
    assert not (tmp_path / "trial").exists()


def test_simulator_environment_excludes_tokens_then_restores(monkeypatch):
    monkeypatch.setenv("CHATGPT_ACCESS_TOKEN", "OFFLINE_SENTINEL")
    with runner.simulator_without_credentials():
        assert "CHATGPT_ACCESS_TOKEN" not in runner.os.environ
    assert runner.os.environ["CHATGPT_ACCESS_TOKEN"] == "OFFLINE_SENTINEL"
