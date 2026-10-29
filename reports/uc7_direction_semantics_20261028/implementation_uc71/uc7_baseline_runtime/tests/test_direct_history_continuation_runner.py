"""Continuation control-flow tests: all simulator and API objects are mocked."""

import copy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.astra import BudgetExceeded
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.policy import Decision


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/continue_bowl_direct_history.py"
spec = importlib.util.spec_from_file_location("continue_bowl_direct_history", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "parent"
    source.mkdir()
    camera = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    observations = tuple(PolicyObservation(camera, camera,
        (i * 0.00001, 0, 1, 0, 0, 0, 0.04, -0.04), runner.base.TASK) for i in range(451))
    statuses = tuple({"steps": i, "success": False, "terminated": False,
                      "task_id": 0, "episode_index": 0, "initial_state_sha256": "fixed"}
                     for i in range(451))
    config = runner.base.configuration(300)
    old_config = runner.base.configuration(150)
    budget = {"api_calls": 90, "unknown_usage_calls": 0,
              "cost_usd_upper_bound": 123.52010000000006,
              "max_cost_usd": 300, "configuration": asdict(config)}
    old_budget = {**budget, "max_cost_usd": 150, "configuration": asdict(old_config)}
    simulator = asdict(runner.LiberoConfig(suite="libero_spatial", max_steps=None, seed=7,
        libero_root=runner.base.DEFAULT_LIBERO_ROOT, python_executable=runner.base.DEFAULT_SIM_PYTHON))
    metadata = {"mock": True, "suite": "libero_spatial"}
    tasks = [{"task_id": 0, "prompt": runner.base.TASK}]
    (source / "simulator.json").write_text(json.dumps({"metadata": metadata, "tasks": tasks}))
    fresh_runner = "scripts/evaluate_bowl_direct_history.py"
    manifest = {"identity_hash": "mock-validated-parent", "configuration": {
        "simulator": simulator, "software": {"source_sha256": {}},
        "runner": {"path": fresh_runner, "sha256": runner.sha(ROOT / fresh_runner)},
        "policy": {"method": "direct_history", "update_steps": 5}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    summary = {"budget": old_budget, "validated_action_chunks": 90,
               "gripper_close_actions": 82, "gripper_open_actions": 368,
               "gripper_neutral_actions": 0, "path_length_m": 1.3904897899101252}
    return SimpleNamespace(source=source, steps=450, api_calls=90, config=config, budget=budget,
        observations=observations, actions=tuple((0, 0, 0, 0, 0, 0, -1) for _ in range(450)),
        sim_statuses=statuses, final_observation=observations[-1], manifest=manifest, summary=summary,
        history_digest="full-prior-history-digest", next_request_key="verified-next-request-key",
        verification={"offline_only": True},
        source_hashes={name: runner.sha(source / name) for name in ("manifest.json", "simulator.json")},
        metadata=metadata, tasks=tasks)


class FakeRecorder:
    instances = []

    def __init__(self, path, **kwargs):
        self.kwargs, self.events, self.refresh_count = kwargs, [], 0
        path.write_bytes(b"TEST VIDEO ONLY")
        self.instances.append(self)

    def observe(self, obs, **kwargs):
        self.events.append(("observe", kwargs))

    def decision(self, obs, decision, **kwargs):
        self.events.append(("decision", {"step": decision.diagnostics["step"]}))

    def executed(self, obs, **kwargs):
        self.events.append(("executed", kwargs))

    def close(self, **kwargs):
        self.events.append(("close", kwargs))


def install_mocks(monkeypatch, prepared, *, mismatch=None, success_at=452, fail_second_api=False,
                  interrupt_api=False, restore_mutation=False):
    instances, restorations, requests = [], [], []
    monkeypatch.setattr(runner, "load_resume", lambda source, **kwargs: prepared)

    class Simulator:
        def __init__(self, config):
            assert "OPENAI_API_KEY" not in os.environ
            assert config.max_steps is None
            self.metadata, self.status = prepared.metadata, {}
            self.actions, self.closed, self.reset_calls = [], False, 0
            self.obs = prepared.observations[0]
            instances.append(self)

        def start(self):
            return self

        def list_tasks(self):
            return [] if mismatch == "metadata" else prepared.tasks

        def reset(self, task, init):
            assert (task, init) == (0, 0)
            self.reset_calls += 1
            self.status = dict(prepared.sim_statuses[0])
            return self.obs

        def step(self, action):
            self.actions.append(action)
            step = len(self.actions)
            if step <= 450:
                self.obs = prepared.observations[step]
                self.status = dict(prepared.sim_statuses[step])
                if step == 25:
                    if mismatch == "state":
                        values = list(self.obs.state)
                        values[0] += 1e-15
                        self.obs = PolicyObservation(self.obs.image, self.obs.wrist_image,
                                                     tuple(values), self.obs.prompt)
                    elif mismatch == "image":
                        pixels = np.zeros((224, 224, 3), dtype=np.uint8)
                        pixels[0, 0, 0] = 1
                        image = image_to_data_url(pixels, rotate=False)
                        self.obs = PolicyObservation(image, self.obs.wrist_image, self.obs.state, self.obs.prompt)
                    elif mismatch == "status":
                        self.status["initial_state_sha256"] = "different"
                if step == 450 and mismatch == "parent_hash":
                    (prepared.source / "simulator.json").write_text("{}")
            else:
                values = list(self.obs.state)
                values[0] += 0.001
                self.obs = PolicyObservation(self.obs.image, self.obs.wrist_image, tuple(values), self.obs.prompt)
                self.status = {**self.status, "steps": step,
                               "success": step == success_at, "terminated": step == success_at}
            return self.obs, self.status["success"]

        def close(self):
            self.closed = True

    def restore(resume, *, api_key, transport, log_dir):
        assert len(instances[0].actions) == 450
        assert instances[0].status == prepared.sim_statuses[-1]
        assert not instances[0].closed
        restorations.append(resume)
        ledger = copy.deepcopy(prepared.budget)

        class Client:
            def budget_snapshot(self):
                return dict(ledger)

            def preflight(self):
                pass

            def history_stats(self):
                return {"messages": 665, "observations": 90, "feedback_records": 450}

            def build_payload(self, observation, grid, mode):
                return {"input": ["all prior outputs and feedback", observation], "cap": 300}

            def request_key(self, payload, grid, mode):
                return "wrong-next-request-key" if restore_mutation else prepared.next_request_key

        class Policy:
            def __init__(self):
                self.client, self.step_index, self.chunk, self.grid = Client(), 450, [], object()

            def reset(self):
                pytest.fail("Restored policy must NEVER be reset")

            def act(self, observation):
                step = self.step_index
                refresh = step % 5 == 0
                if refresh:
                    if fail_second_api and ledger["api_calls"] == 91:
                        raise BudgetExceeded("test next request reservation blocked")
                    ledger["api_calls"] += 1
                    requests.append({"ordinal": ledger["api_calls"], "prior_steps": len(instances[0].actions),
                                     "max_cost_usd": ledger["max_cost_usd"]})
                    if interrupt_api:
                        ledger["cost_usd_upper_bound"] += 26.8644
                        ledger["unknown_usage_calls"] += 1
                        raise KeyboardInterrupt
                    transport("mock", self.client.build_payload(observation.to_dict(), self.grid, "direct"), {}, 180)
                    ledger["cost_usd_upper_bound"] += 0.1
                self.step_index += 1
                return Decision((0.02, 0, 0, 0, 0, 0, 1), {"step": step, "refreshed": refresh})

            def observe_transition(self, before, decision, after):
                assert before != after
                return {"executed_step": decision.diagnostics["step"]}

        return Policy()

    monkeypatch.setattr(runner, "restore_policy", restore)
    return Simulator, instances, restorations, requests


def run(tmp_path, prepared, simulator, **kwargs):
    return runner.execute_episode(prepared.source, tmp_path / "child", max_cost_usd=300,
        api_key="UNIT_TEST_SECRET", sim_factory=simulator, recorder_factory=FakeRecorder,
        transport=kwargs.pop("transport", lambda *args: {"mock": True}), **kwargs)


def test_exact_gate_then_same_policy_continues_at_step450_and_ordinal91(tmp_path, prepared, monkeypatch):
    sim, instances, restores, requests = install_mocks(monkeypatch, prepared)
    monkeypatch.setenv("OPENAI_API_KEY", "UNIT_TEST_SECRET")
    result = run(tmp_path, prepared, sim)
    assert result["status"] == "success" and result["replay_verified"]
    assert result["base_steps"] == 450 and result["new_steps"] == 2
    assert result["steps"] == result["cumulative_steps"] == 452
    assert result["api_calls"] == 91 and result["new_api_calls"] == 1
    assert result["budget"]["cost_usd_upper_bound"] == pytest.approx(123.6201)
    assert result["max_cost_usd"] == 300 and result["requested_episodes"] == 1
    assert result["gripper_close_actions"] == 84 and result["new_gripper_close_actions"] == 2
    assert result["measured_feedback_count"] == 452
    assert len(restores) == 1 and requests == [{"ordinal": 91, "prior_steps": 450, "max_cost_usd": 300}]
    assert instances[0].closed and instances[0].reset_calls == 1
    trace = [json.loads(line) for line in (tmp_path / "child/trace.jsonl").read_text().splitlines()]
    assert trace[0]["event"] == "resume" and trace[0]["step"] == 450
    assert [row["step"] for row in trace if row["event"] == "executed"] == [450, 451]
    assert not any(row["event"] == "reset" for row in trace)
    replay = json.loads((tmp_path / "child/replay.json").read_text())
    assert replay["completed_actions"] == 450
    assert replay["state_checks"] == replay["image_pairs_checked"] == replay["status_checks"] == 451
    assert replay["max_state_error"] == replay["max_pixel_error"] == replay["astra_requests_during_reconstruction"] == 0
    video = FakeRecorder.instances[-1]
    assert video.refresh_count == 90 and video.events[0] == ("observe", {"step": 450})
    assert video.events[-1][0] == "close"


@pytest.mark.parametrize("mismatch", ["metadata", "state", "image", "status", "parent_hash"])
def test_replay_or_parent_evidence_mismatch_blocks_all_new_api(tmp_path, prepared, monkeypatch, mismatch):
    sim, instances, restores, requests = install_mocks(monkeypatch, prepared, mismatch=mismatch)
    result = run(tmp_path, prepared, sim, transport=lambda *a: pytest.fail("Gate failure called API"))
    assert result["status"] == "infrastructure_error" and not result["replay_verified"]
    assert result["new_steps"] == result["new_api_calls"] == 0
    assert result["cumulative_steps"] == 450 and result["api_calls"] == 90
    assert result["budget"]["cost_usd_upper_bound"] == pytest.approx(123.5201)
    assert requests == restores == [] and instances[0].closed
    assert result["success_rate"] is None and result["video"] is None


def test_next_request_history_mismatch_blocks_api_after_replay(tmp_path, prepared, monkeypatch):
    sim, instances, restores, requests = install_mocks(monkeypatch, prepared, restore_mutation=True)
    result = run(tmp_path, prepared, sim)
    assert result["status"] == "infrastructure_error" and result["new_api_calls"] == 0
    assert "next request" in result["error"] and len(restores) == 1 and requests == []
    assert instances[0].closed


def test_next_chunk_budget_stop_keeps_single_unscored_episode_and_cumulative_counts(tmp_path, prepared, monkeypatch):
    sim, instances, _, requests = install_mocks(monkeypatch, prepared, success_at=None, fail_second_api=True)
    result = run(tmp_path, prepared, sim)
    assert result["status"] == "budget_exceeded"
    assert result["new_steps"] == 5 and result["cumulative_steps"] == 455
    assert result["api_calls"] == 91 and len(requests) == 1
    assert result["evaluation_completed"] is False and result["scored_episodes"] == 0
    assert result["success_rate"] is None and result["libero_success"] is None
    assert instances[0].closed


def test_api_interrupt_retains_cumulative_reservation_and_finalizes(tmp_path, prepared, monkeypatch):
    sim, instances, _, _ = install_mocks(monkeypatch, prepared, interrupt_api=True)
    result = run(tmp_path, prepared, sim)
    assert result["status"] == "user_stop" and result["new_steps"] == 0
    assert result["api_calls"] == 91 and result["budget"]["unknown_usage_calls"] == 1
    assert result["budget"]["cost_usd_upper_bound"] == pytest.approx(150.3845)
    assert instances[0].closed and FakeRecorder.instances[-1].events[-1][0] == "close"


def test_terminal_transport_error_never_leaks_credential(tmp_path, prepared, monkeypatch):
    sim, instances, _, _ = install_mocks(monkeypatch, prepared)

    def failure(*args):
        raise RuntimeError("UNIT_TEST_SECRET in a transport error")

    result = run(tmp_path, prepared, sim, transport=failure)
    assert result["status"] == "infrastructure_error" and instances[0].closed
    contents = "".join(path.read_text(errors="ignore") for path in (tmp_path / "child").rglob("*") if path.is_file())
    assert "UNIT_TEST_SECRET" not in contents


def test_preflight_does_not_start_sim_restore_live_policy_or_write(tmp_path, prepared, monkeypatch):
    monkeypatch.setattr(runner, "load_resume", lambda *a, **kw: prepared)
    monkeypatch.setattr(runner, "SimProcess", lambda *a: pytest.fail("Preflight started simulator"))
    monkeypatch.setattr(runner, "restore_policy", lambda *a, **kw: pytest.fail("Preflight restored live client"))
    child = tmp_path / "child"
    result = runner.preflight(prepared.source, child, max_cost_usd=300)
    assert result["new_api_calls"] == 0 and result["next_request_ordinal"] == 91
    assert result["files_written"] is False and not child.exists()


def test_path_and_existing_descendant_guards_refuse_ledger_forks(tmp_path, prepared):
    with pytest.raises(ValueError, match="separate sibling"):
        runner.validate_output(prepared.source, prepared.source / "child")
    with pytest.raises(ValueError, match="separate sibling"):
        runner.validate_output(prepared.source, prepared.source)
    other = tmp_path / "prior_child"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"configuration": {
        "kind": runner.KIND, "parent_run": str(prepared.source)}}))
    with pytest.raises(ValueError, match="ledger may be newer"):
        runner.validate_output(prepared.source, tmp_path / "next_child")


def test_authorization_must_match_cap_parent_and_historical_spending(tmp_path, prepared):
    path = tmp_path / "authorization.json"
    record = {"max_cost_usd_cumulative": 300, "parent_run": str(prepared.source), "parent_steps": 450,
              "parent_api_calls": 90, "parent_unknown_usage_calls": 0,
              "parent_cost_usd_upper_bound": 123.5201,
              "parent_artifact_sha256": prepared.source_hashes}
    path.write_text(json.dumps(record))
    assert runner.authorization_record(path, 300, prepared)["record"] == record
    with pytest.raises(ValueError, match="cap differs"):
        runner.authorization_record(path, 301, prepared)
    path.write_text(json.dumps({**record, "parent_steps": 445}))
    with pytest.raises(ValueError, match="exact parent"):
        runner.authorization_record(path, 300, prepared)

