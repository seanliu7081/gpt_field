"""Offline lifecycle regressions; no model, credentials, or simulator is started."""
import importlib.util
import hashlib
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import PolicyObservation, image_to_data_url


@pytest.fixture
def runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_astranomy_agentic.py"
    spec = importlib.util.spec_from_file_location("_test_astranomy_agentic_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "freeze_sources", lambda output: None)
    monkeypatch.setattr(module, "source_manifest", lambda: {})
    monkeypatch.setattr(module.life, "DEFAULT_EXPECTED_SHA256", {})
    monkeypatch.setattr(module.life, "termination_as_interrupt", nullcontext)
    monkeypatch.setattr(module.life, "_simulator_without_credentials", nullcontext)
    return module


@pytest.fixture
def observation():
    url = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), size=512, rotate=False)
    return PolicyObservation(url, url, (0., 0., 1., np.pi, 0., 0., .04, -.04),
                             resolve_distance_task("bowl_plate").prompt,
                             tuple([0.] * 7), tuple([0.] * 7))


class FakeSim:
    def __init__(self, observation, *, success_after):
        self.observation = observation
        self.success_after = success_after
        self.metadata = {"offline_fake": True}
        self.status = {}
        self.actions = []
        self.geometry_calls = 0
        self.closed = False
        self.started = False

    def start(self):
        self.started = True
        return self

    def reset(self, task_id, episode_index):
        self.status = {"success": False, "libero_success": False, "terminated": False, "steps": 0}
        return self.observation

    def geometry(self):
        self.geometry_calls += 1
        # The runner passes this through; calibration internals belong to their own tests.
        return {"offline_geometry_fixture": True}

    def step(self, action):
        assert not self.status["terminated"], "No action after evaluator termination"
        self.actions.append(list(action))
        success = len(self.actions) == self.success_after
        self.status.update(success=success, libero_success=success, terminated=success,
                           steps=len(self.actions))
        # No motion dynamics are mocked: an unchanged measurement deliberately tests that
        # finishing a selected segment does not imply arrival or semantic task completion.
        return self.observation, success

    def close(self):
        self.closed = True

    def truth(self):
        assert self.status["terminated"] and not self.closed
        return {"offline_truth_fixture": True, "steps": len(self.actions)}


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.results = []
        self.requests = []

    def budget_snapshot(self):
        return {"api_calls": self.calls, "api_key_calls": 0}

    def record_result(self, result):
        self.results.append(result)

    def request(self, request):
        assert set(request) == {"external_image", "wrist_image", "robot_state",
                                "joint_positions", "geometry", "step"}
        assert not {"success", "libero_success", "reward", "collision"}.intersection(request)
        self.calls += 1
        self.requests.append(request)
        target = list(request["robot_state"])
        target[2] += .01
        target[7] = 1
        return {"name": "execute_eef_target", "arguments": {"target": target, "steps": 5}}

    def predict(self, request):
        return self.request(request)


def execute(runner, tmp_path, sim, client, **kwargs):
    return runner.execute(tmp_path / "attempt", task_key="bowl_plate", video=False,
                          token_provider=lambda: "unused-offline-fixture",
                          sim_factory=lambda config: sim,
                          client_factory=lambda *args, **kwargs: client, **kwargs)


@pytest.mark.parametrize("limit", ["translation", "rotation"])
def test_target_limit_rejects_without_motion_then_next_decision_continues(
        runner, tmp_path, observation, limit):
    sim = FakeSim(observation, success_after=5)

    class ReplanningClient(FakeClient):
        def predict(self, request):
            tool = self.request(request)
            if self.calls == 1:
                if limit == "translation":
                    tool["arguments"]["target"][0] += .1
                else:
                    # Identity orientation is pi radians from the downward reset pose.
                    tool["arguments"]["target"][3:7] = [1., 0., 0., 0.]
            else:
                assert self.calls == 2
                assert sim.actions == []
                assert len(self.results) == 1
                assert self.results[0]["status"] == "rejected"
                assert self.results[0]["executed_steps"] == 0
                assert request["step"] == 0
            return tool

    client = ReplanningClient()
    result = execute(runner, tmp_path, sim, client)
    assert result["status"] == "success" and result["error"] is None
    assert result["api_calls"] == result["model_responses"] == 2
    assert result["moves"] == 2 and result["rejected_moves"] == 1
    assert result["steps"] == len(sim.actions) == 5
    assert [feedback["status"] for feedback in client.results] == ["rejected", "segment_complete"]
    assert client.results[0]["simulation_time_s"] == 0
    assert client.results[1]["position_error_m"] > 0
    assert all(action[6] == 1. for action in sim.actions)
    assert json.loads((tmp_path / "attempt/tool_results.json").read_text()) == client.results
    assert sim.closed


def test_initial_hash_guard_precedes_geometry_and_model(runner, tmp_path, monkeypatch, observation):
    monkeypatch.setattr(runner.life, "DEFAULT_EXPECTED_SHA256", {("bowl_plate", 0): "wrong"})
    sim = FakeSim(observation, success_after=1)
    client = FakeClient()
    result = execute(runner, tmp_path, sim, client)
    assert result["status"] == "infrastructure_error"
    assert "no inference performed" in result["error"]["message"]
    assert result["api_calls"] == result["model_responses"] == result["steps"] == 0
    assert client.requests == client.results == sim.actions == []
    assert sim.geometry_calls == 0 and sim.closed
    check = json.loads((tmp_path / "attempt/initial_observation_check.json").read_text())
    assert check["matched"] is False
    assert result["control_elapsed_s"] is None


@pytest.mark.parametrize("failure,status", [
    (RuntimeError("Subscription unavailable"), "infrastructure_error"),
    (KeyboardInterrupt(), "user_stop"),
])
def test_failed_request_wait_is_accounted_without_a_model_response(
        runner, tmp_path, monkeypatch, observation, failure, status):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock.now)
    sim = FakeSim(observation, success_after=1)

    class FailingClient(FakeClient):
        def predict(self, request):
            self.request(request)
            clock.now += 8.0
            raise failure

    client = FailingClient()
    result = execute(runner, tmp_path, sim, client)
    assert result["status"] == status
    assert result["api_calls"] == 1 and result["model_responses"] == result["steps"] == 0
    assert result["timing_components"]["model_pipeline_s"] == 8.0
    assert result["control_without_video_s"] == 8.0
    assert not result["evaluation_completed"] and not result["completion_time_eligible"]
    assert client.results == sim.actions == [] and sim.closed
    progress = json.loads((tmp_path / "attempt/run_state.json").read_text())
    assert progress["stage"] == "complete" and progress["status"] == status
    assert progress["api_calls"] == 1 and progress["model_responses"] == 0


def test_raw_success_interrupts_selected_segment_and_returns_actual_count(
        runner, tmp_path, observation):
    sim = FakeSim(observation, success_after=2)
    client = FakeClient()
    result = execute(runner, tmp_path, sim, client)
    assert result["status"] == "success" and result["libero_success"] is True
    assert result["evaluation_completed"] and result["completion_time_eligible"]
    assert result["api_calls"] == result["model_responses"] == result["moves"] == 1
    assert result["steps"] == len(sim.actions) == 2
    assert result["rejected_moves"] == 0
    assert len(client.results) == 1
    feedback = client.results[0]
    assert feedback["status"] == "interrupted"
    assert feedback["requested_steps"] == 5 and feedback["executed_steps"] == 2
    assert feedback["simulation_time_s"] == .1
    assert result["last_feedback"] == feedback
    assert not {"success", "libero_success", "reward", "collision"}.intersection(feedback)
    assert sim.closed
    assert result["reasoning_effort"] == "medium"
    assert result["collision_policy"] == "record_only"


def test_existing_output_directory_is_never_overwritten(runner, tmp_path):
    marker = tmp_path / "preserve.txt"
    marker.write_text("existing result")
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, task_key="bowl_plate", token_provider=lambda: "unused",
                       video=False, client_factory=lambda *args, **kwargs: pytest.fail("client constructed"),
                       sim_factory=lambda config: pytest.fail("simulator constructed"))
    assert marker.read_text() == "existing result"


def test_operator_stop_records_partial_segment_without_another_action(
        runner, tmp_path, observation):
    output = tmp_path / "attempt"

    class StopAfterFirstStepSim(FakeSim):
        def step(self, action):
            transition = super().step(action)
            assert len(self.actions) == 1
            (output / "operator_stop.request.json").write_text("{}")
            return transition

    sim = StopAfterFirstStepSim(observation, success_after=5)
    client = FakeClient()
    result = execute(runner, tmp_path, sim, client)
    assert result["status"] == "user_stop" and result["error"] is None
    assert result["steps"] == len(sim.actions) == 1
    assert result["api_calls"] == result["model_responses"] == 1
    assert not result["libero_success"] and not result["evaluation_completed"]
    assert len(client.results) == 1
    feedback = client.results[0]
    assert feedback["status"] == "interrupted"
    assert feedback["requested_steps"] == 5 and feedback["executed_steps"] == 1
    assert feedback["simulation_time_s"] == .05
    assert result["last_feedback"] == feedback
    assert json.loads((output / "tool_results.json").read_text()) == [feedback]
    progress = json.loads((output / "run_state.json").read_text())
    assert progress["stage"] == "complete" and progress["last_feedback"] == feedback
    records = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert [item["feedback"] for item in records if item["event"] == "tool_result"] == [feedback]
    assert sim.closed


def test_high_custom_scene_uses_frozen_inputs_and_explicit_raw_reset_hash(
        runner, tmp_path, monkeypatch, observation):
    prompt = "put both moka pots on the stove"
    observation = replace(observation, prompt=prompt)
    bddl = tmp_path / "scene.bddl"
    bddl.write_text("(define (problem test) (:domain robosuite) "
                    "(:language put both moka pots on the stove) "
                    "(:objects moka_pot_1 moka_pot_2 - moka_pot) "
                    "(:goal (and (on moka_pot_1 stove) (on moka_pot_2 stove))))")
    initial_state = tmp_path / "state.npy"
    np.save(initial_state, np.arange(60, dtype=float))
    expected = runner.initial_identity(observation)
    monkeypatch.setattr(runner.life, "DEFAULT_EXPECTED_SHA256", {("libero_10_t8", 0): "wrong"})
    sim = FakeSim(observation, success_after=1)
    client = FakeClient()
    seen = {}

    def make_client(config, **kwargs):
        seen["client"] = kwargs
        return client

    def make_sim(config):
        seen["sim"] = config
        assert Path(config.task_bddl_override).read_bytes() == bddl.read_bytes()
        assert Path(config.initial_state_override).read_bytes() == initial_state.read_bytes()
        return sim

    result = runner.execute(tmp_path / "attempt", task_key="libero_10_t8", video=False,
        token_provider=lambda: "unused-offline-fixture", sim_factory=make_sim,
        client_factory=make_client, reasoning_effort="high", task_bddl_override=bddl,
        initial_state_override=initial_state, expected_initial_observation_sha256=expected,
        sim_python="/fixture/python", libero_root="/fixture/LIBERO")
    assert result["status"] == "success" and result["reasoning_effort"] == "high"
    assert seen["client"]["reasoning_effort"] == "high"
    assert seen["sim"].suite == "libero_10" and seen["sim"].max_steps is None
    assert seen["sim"].python_executable == "/fixture/python"
    assert seen["sim"].libero_root == "/fixture/LIBERO"
    assert seen["sim"].include_arm_joints and seen["sim"].collision_check_mode == "none"
    assert result["collision_policy"] == "record_only"
    assert result["collision_monitor"] == "generic_contact_v2"
    output = tmp_path / "attempt"
    identity = json.loads((output / "manifest.json").read_text())["configuration"]
    from gpt_field.astranomy_agentic_motion import ShortTargetConfig
    assert identity["motion"] == ShortTargetConfig().to_dict()
    assert identity["reasoning_effort"] == "high" and identity["custom_scene"] is True
    provenance = json.loads((output / "input_provenance.json").read_text())
    for field, source in (("task_bddl_override", bddl), ("initial_state_override", initial_state)):
        record = provenance[field]
        assert record["source"] == str(source)
        assert Path(record["frozen_path"]).parent == output / "frozen_inputs"
        assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    check = json.loads((output / "initial_observation_check.json").read_text())
    assert check["matched"] is True and check["expected"] == expected
    assert check["source"] == "explicit_override"
    assert result["physical_truth"] == {"offline_truth_fixture": True, "steps": 1}
    assert json.loads((output / "physical_truth.json").read_text()) == result["physical_truth"]
    assert client.calls == len(sim.actions) == 1 and sim.closed


def test_explicit_reset_mismatch_precedes_geometry_and_model(runner, tmp_path, observation):
    sim, client = FakeSim(observation, success_after=1), FakeClient()
    result = execute(runner, tmp_path, sim, client,
                     expected_initial_observation_sha256="0" * 64)
    assert result["status"] == "infrastructure_error"
    assert sim.geometry_calls == client.calls == len(sim.actions) == 0
    assert sim.closed


@pytest.mark.parametrize("overrides", [
    {"initial_state_override": "state.npy"},
    {"task_bddl_override": "scene.bddl"},
    {"initial_state_override": "state.npy", "task_bddl_override": "scene.bddl"},
])
def test_custom_scene_requires_pair_and_explicit_identity(runner, tmp_path, overrides):
    with pytest.raises(ValueError, match="Custom scene requires"):
        runner.execute(tmp_path / "attempt", task_key="libero_10_t8",
            token_provider=lambda: "unused", video=False, **overrides)
    assert not (tmp_path / "attempt").exists()


def test_cli_forwards_high_custom_scene_options_without_running(runner, tmp_path, monkeypatch):
    from gpt_field import chatgpt_auth
    seen = {}
    monkeypatch.setattr(chatgpt_auth, "SubscriptionCredentials",
                        lambda **kwargs: SimpleNamespace(access_token=lambda: "unused"))

    def capture(output, **kwargs):
        seen.update(output=output, **kwargs)
        return {"evaluation_completed": True}

    monkeypatch.setattr(runner, "execute", capture)
    assert runner.main(["--task", "libero_10_t8", "--output", str(tmp_path / "attempt"),
        "--reasoning-effort", "high", "--task-bddl-override", "scene.bddl",
        "--initial-state-override", "state.npy", "--expected-initial-observation-sha256", "a" * 64,
        "--sim-python", "/fixture/python", "--libero-root", "/fixture/LIBERO", "--no-video"]) == 0
    assert seen["task_key"] == "libero_10_t8" and seen["reasoning_effort"] == "high"
    assert seen["task_bddl_override"] == Path("scene.bddl")
    assert seen["initial_state_override"] == Path("state.npy")
    assert seen["expected_initial_observation_sha256"] == "a" * 64
    assert seen["sim_python"] == Path("/fixture/python")
    assert seen["libero_root"] == Path("/fixture/LIBERO")
    assert seen["video"] is False
