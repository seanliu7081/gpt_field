"""Finite benchmark orchestration and timing ledger; no network or real simulator."""
import importlib.util
import os
from pathlib import Path

import pytest

from gpt_field.chatgpt_plan import _access_token, _protocol_error
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.observation import PolicyObservation
from test_field_history_runner import observation, sim_config
from test_hybrid_subscription_runner import runner

spec = importlib.util.spec_from_file_location("compact_timing_test", Path(__file__).resolve().parents[1] / "scripts/benchmark_compact_subscription.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class Clock:
    def __init__(self):
        self.value = 0.
    def __call__(self):
        return self.value
    def advance(self, value):
        self.value += value


def fixtures(observation, *, success_at=None, fail_transport=False, fail_close=False, drift_reset=False):
    clock, clients, sims = Clock(), [], []

    class Client:
        grid = HYBRID_GRID
        chunk_size = 5
        field_update_steps = 30
        planner_horizon = field_execution_steps = 10
        reasoning_effort = "high"
        field_representation = "analytic-goal-rbf-v1"
        prompt_version = "offline-fixture"
        last_request_key = "offline-fixture"
        last_log_path = None
        routing_is_model_prediction = False
        last_field_expansion_s = 0.
        last_compact_prediction = None

        def __init__(self, config, *, controller_mode, token_provider, transport=None, log_dir=None):
            self.controller_mode, self.token_provider = controller_mode, token_provider
            self.calls, self.feedback = 0, []
            self.transport = transport or self.fixture_transport
            clients.append(self)

        def fixture_transport(self, *args):
            clock.advance(2.)
            if fail_transport:
                raise _protocol_error("private failure", http_status=429, error_code="subscription_usage_limit")
            return {}

        def reset_history(self):
            self.feedback.clear()

        def history_stats(self):
            return {"feedback_records": len(self.feedback)}

        def record_transition(self, feedback):
            clock.advance(.05)
            self.feedback.append(feedback)

        def budget_snapshot(self):
            return {"api_calls": self.calls, "input_tokens": self.calls * 100, "output_tokens": self.calls * 10}

        def predict(self, obs, grid, mode):
            assert grid == self.grid and mode == "hybrid"
            _access_token(self.token_provider)
            self.calls += 1
            self.transport("fixture", {}, {}, 180)
            field = self.controller_mode == "field"
            self.last_field_expansion_s = .3 if field else 0.
            clock.advance(self.last_field_expansion_s)
            self.last_compact_prediction = {"offline_fixture": True}
            return {"controller": "coarse_field" if field else "contact_direct",
                "target_region": "black_bowl", "near_target": not field,
                "routing_reason": "Fixed benchmark mode.",
                "field": {"stage": "fixture", "orientation_target_xyzw": [0, 0, 0, 1],
                    "gripper_command": "open", "motion_mode": "hold", "values": [0.] * grid.count,
                    "blocked_indices": [], "unknown_indices": []} if field else None,
                "direct": None if field else {"actions": [[0, 0, 0, 0, 0, 0, 0]] * 5}}

    class Sim:
        metadata = {"offline_fixture": True}
        def __init__(self, config):
            assert os.environ.get("OPENAI_API_KEY") is None
            self.closed, self.resets, self.status = False, [], {}
            sims.append(self)
        def start(self):
            return self
        def list_tasks(self):
            return [{"task_id": 0, "prompt": benchmark.TASK}]
        def reset(self, task, state):
            self.resets.append((task, state))
            self.steps, self.obs = 0, observation
            if drift_reset and len(self.resets) > 1:
                state = list(observation.state)
                state[0] += .001
                self.obs = PolicyObservation(observation.image, observation.wrist_image, tuple(state), observation.prompt)
            self.status = {"success": False, "terminated": False}
            clock.advance(100.)  # Explicitly excluded initialization/reset time.
            return self.obs
        def step(self, action):
            clock.advance(.1)
            self.steps += 1
            success = self.steps == success_at
            self.status = {"success": success, "terminated": success}
            return self.obs, success
        def close(self):
            self.closed = True
            if fail_close:
                raise RuntimeError("private cleanup failure")

    return clock, Client, Sim, clients, sims


def run(output, observation, *, blocks=3, actions=30, token="oauth-fixture", **fixture_kwargs):
    clock, client, sim, clients, sims = fixtures(observation, **fixture_kwargs)
    result = benchmark.run_benchmark(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: token, blocks=blocks, actions_per_block=actions,
        client_factory=client, sim_factory=sim, clock=clock)
    return result, clients, sims


def test_paired_order_fresh_resets_equal_actions_and_separate_timing(tmp_path, observation, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unused-private-fixture")
    result, clients, sims = run(tmp_path / "comparison", observation)
    assert result["status"] == "completed" and result["comparison"]["complete"]
    assert [a["controller_mode"] for a in result["arms"]] == ["field", "direct", "direct", "field", "field", "direct"]
    assert len(clients) == 6 and sims[0].resets == [(0, 0)] * 6 and sims[0].closed
    assert os.environ["OPENAI_API_KEY"] == "sk-unused-private-fixture"
    for arm in result["arms"]:
        field = arm["controller_mode"] == "field"
        assert arm["actions"] == arm["translation_hold_actions"] == 30
        assert arm["reset_s"] == pytest.approx(100.)
        assert arm["initial_observation_digest"] == result["arms"][0]["initial_observation_digest"]
        assert arm["local_plans"] == (3 if field else 0)
        assert arm["api_calls"] == (1 if field else 6)
        t = arm["timing"]
        assert t["model_transport_s"] == pytest.approx(2 if field else 12)
        assert t["field_expansion_s"] == pytest.approx(.3 if field else 0.)
        assert t["policy_decision_s"] == pytest.approx(2.3 if field else 12)
        assert t["feedback_s"] == pytest.approx(1.5)
        assert t["simulator_step_s"] == pytest.approx(3.)
        assert t["control_loop_s"] == pytest.approx(6.8 if field else 16.5)
        assert t["policy_total_s"] == pytest.approx(t["policy_decision_s"] + t["feedback_s"])
    assert result["simulator_startup_s"] == 0.
    assert result["wall_elapsed_s"] > 600.
    assert (tmp_path / "comparison/comparison.md").is_file()
    assert (tmp_path / "comparison/run_state.json").is_file()


def test_partial_termination_stops_and_cannot_claim_matched_comparison(tmp_path, observation):
    result, clients, sims = run(tmp_path / "partial", observation, success_at=2)
    assert result["status"] == "incomplete_simulator_termination"
    assert len(result["arms"]) == len(clients) == 1
    assert result["arms"][0]["actions"] == 2 and sims[0].closed
    assert not result["comparison"]["complete"] and result["comparison"]["aggregates"] == {}
    assert "No overall speed comparison" in (tmp_path / "partial/comparison.md").read_text()


def test_transport_failure_records_elapsed_attempt_without_retry_or_secret(tmp_path, observation):
    result, clients, sims = run(tmp_path / "failed", observation, fail_transport=True)
    assert result["status"] == "subscription_error" and not result["comparison"]["complete"]
    arm = result["arms"][0]
    assert arm["actions"] == 0 and arm["api_calls"] == 1 and len(clients) == 1
    assert arm["timing"]["model_transport_s"] == arm["timing"]["policy_decision_s"] == 2.
    assert arm["transport_calls"] == [{"transport_s": 2., "completed": False}]
    assert sims[0].closed and "private failure" not in str(result)


def test_api_key_token_is_rejected_before_transport(tmp_path, observation):
    result, clients, _ = run(tmp_path / "auth", observation, token="sk-not-oauth")
    assert result["status"] == "subscription_error" and result["api_key_calls"] == 0
    assert result["arms"][0]["transport_calls"] == []
    assert result["arms"][0]["actions"] == 0 and clients[0].calls == 0


@pytest.mark.parametrize("blocks, actions", [(0, 30), (True, 30), (1, 0), (1, 2.5)])
def test_finite_positive_integer_bounds_required(tmp_path, observation, blocks, actions):
    with pytest.raises(ValueError):
        run(tmp_path / "invalid", observation, blocks=blocks, actions=actions)
    assert not (tmp_path / "invalid").exists()


def test_default_hybrid_preflight_uses_compact_client_without_auth_or_execution(tmp_path):
    result = runner.preflight(runner.configuration(), sim_config(), tmp_path / "not_created")
    assert result["api_calls"] == 0 and not result["files_written"]
    assert result["policy"]["field_representation"] == "analytic-goal-rbf-v1"
    assert result["policy"]["controller_mode"] == "hybrid"
    assert "compact" in result["policy"]["prompt_version"]
    field_schema = result["payload"]["text"]["format"]["schema"]["properties"]["field"]["anyOf"][0]
    assert "values" not in field_schema["properties"]


def test_cleanup_failure_cannot_complete_the_comparison(tmp_path, observation):
    result, clients, sims = run(tmp_path / "cleanup", observation, blocks=1, actions=1, fail_close=True)
    assert all(arm["status"] == "completed" for arm in result["arms"])
    assert result["status"] == "infrastructure_error"
    assert not result["comparison"]["complete"] and result["comparison"]["aggregates"] == {}
    assert sims[0].closed and "private cleanup failure" not in str(result)


def test_different_reset_observation_stops_before_next_inference(tmp_path, observation):
    result, clients, _ = run(tmp_path / "reset", observation, blocks=1, actions=1, drift_reset=True)
    assert result["status"] == "infrastructure_error" and not result["comparison"]["complete"]
    assert len(clients) == 1 and [arm["actions"] for arm in result["arms"]] == [1, 0]
    assert result["arms"][0]["initial_observation_digest"] != result["arms"][1]["initial_observation_digest"]


def test_cli_rejects_dense_timing_comparison_before_credentials(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--timing-comparison", "--field-representation", "dense", "--output", str(tmp_path / "unused")])
    assert exc.value.code == 2 and not (tmp_path / "unused").exists()
