import copy
from dataclasses import asdict
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.astra import AstraClient, AstraConfig, BudgetExceeded
from gpt_field.experiment import source_manifest
from gpt_field.libero import LiberoConfig
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.schema import GridSpec


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/continue_bowl_episode.py"
spec = importlib.util.spec_from_file_location("continue_bowl_episode", SCRIPT)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def observation(step, pixel=0):
    image = image_to_data_url(np.full((2, 2, 3), pixel, dtype=np.uint8), size=2)
    return {"observation/image": image, "observation/wrist_image": image,
            "observation/state": [step / 100, 0, 0, 0, 0, 0, 0.04, 0.04], "prompt": continuation.base.TASK}


def status(step, *, success=False):
    return {"task_id": 0, "episode_index": 0, "steps": step, "success": success,
            "terminated": success, "horizon_reached": False,
            "initial_state_file": "/fake/init0", "initial_state_sha256": "initial-state-hash", "bddl_sha256": "bddl-hash"}


def prediction():
    return {"stage": "approach", "orientation_target_xyzw": [0, 0, 0, 1], "gripper_command": "open",
            "motion_mode": "hold", "values": [1.0] * 729, "blocked_indices": [], "unknown_indices": []}


def envelope():
    return {"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": json.dumps(prediction())}]}],
        "usage": {"input_tokens": 1000, "output_tokens": 4000,
                  "input_tokens_details": {"cached_tokens": 20}, "output_tokens_details": {"reasoning_tokens": 30}}}


def save_rows(source, rows):
    with gzip.open(source / continuation.recovery.TRACE_NAME, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def make_source(source):
    source.mkdir()
    api = AstraConfig(max_cost_usd=27, input_price_per_million=10, cached_input_price_per_million=1,
                       cache_write_price_per_million=12.5, output_price_per_million=50)
    client = AstraClient(api, api_key="offline-dummy", transport=lambda *args: envelope(), log_dir=source / "api")
    client.predict(observation(0), GridSpec())
    budget = client.budget_snapshot()
    sim = LiberoConfig(suite="libero_spatial", max_steps=None,
                        python_executable="/workspace/venvs/libero-plus/bin/python", libero_root="/fake/libero")
    identity = {"simulator": asdict(sim), "task_ids": [0], "episode_indices": [0], "method": "field",
                "grid": GridSpec().to_dict(), "planner": asdict(PlannerConfig()), "update_steps": 5,
                "policy_seed": 7, "api": asdict(api), "reference": continuation.base.reference_metadata(api),
                "software": source_manifest()}
    (source / "manifest.json").write_text(json.dumps({"configuration": identity,
        "identity_hash": continuation.recovery.canonical_hash(identity)}))
    (source / "simulator.json").write_text(json.dumps({"metadata": {"task_order_index": 0},
        "tasks": [{"task_id": 0, "prompt": continuation.base.TASK}]}))
    (source / "run_state.json").write_text(json.dumps({"status": "budget_exceeded", "budget": budget}))
    episode = {"status": "budget_exceeded", "steps": 5, "seed": 7, "suite": "libero_spatial",
        "method": "field", "task_id": 0, "episode_index": 0, "task_prompt": continuation.base.TASK,
        "budget_after": budget, "log": continuation.recovery.TRACE_NAME, "video": "parent.mp4"}
    (source / "episodes.jsonl").write_text(json.dumps(episode) + "\n")
    rows = [{"event": "reset", "observation": observation(0), "sim_status": status(0)}]
    for step in range(5):
        details = {"method": "field", "step": step, "refreshed": step == 0, "stage": "approach"}
        if step == 0:
            details.update(prediction=prediction(), field_prediction=prediction())
        rows.append({"step": step, "action": [0.1, 0, 0, 0, 0, 0, -1],
                     "state": observation(step)["observation/state"], "diagnostics": details})
        rows.append({"event": "executed", "step": step, "next_state": observation(step + 1)["observation/state"],
                     "observation": observation(step + 1), "sim_status": status(step + 1), "success": False})
    save_rows(source, rows)
    return rows, budget


class FakeSimulator:
    def __init__(self, config, *, mismatch=False, state_error=0, keep_running=False):
        assert config.python_executable == "/workspace/venvs/libero-plus/bin/python"
        self.metadata = {"task_order_index": 0}
        self.status, self.actions, self.closed = {}, [], False
        self.mismatch, self.keep_running = mismatch, keep_running
        self.state_error = state_error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def list_tasks(self):
        return [{"task_id": 0, "prompt": continuation.base.TASK}]

    def reset(self, task, index):
        assert (task, index) == (0, 0)
        self.status = status(0)
        return PolicyObservation.from_dict(observation(0))

    def step(self, action):
        self.actions.append(action)
        step = len(self.actions)
        success = step > 5 and not self.keep_running
        self.status = status(step, success=success)
        data = observation(step, pixel=1 if self.mismatch and step == 1 else 0)
        if step == 1:
            data["observation/state"][0] += self.state_error
        return PolicyObservation.from_dict(data), success


class FakeRecorder:
    def __init__(self, path, **kwargs):
        self.path, self.closed = path, False

    def observe(self, obs, *, step):
        assert step == 5

    def decision(self, obs, decision, **kwargs):
        assert decision.diagnostics["step"] >= 5

    def executed(self, obs, **kwargs):
        pass

    def close(self, **kwargs):
        self.closed = True
        self.path.write_bytes(b"offline-fake-video")


def test_budget_handoff_carries_every_counter_cost_and_elapsed_without_source_mutation(tmp_path):
    source = tmp_path / "source"
    _, original = make_source(source)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    prepared = continuation.prepare_source(source, 50)
    expected = copy.deepcopy(original)
    expected["configuration"]["max_cost_usd"] = 50
    expected["max_cost_usd"] = 50
    assert prepared["original_budget"] == original
    assert prepared["handoff_budget"] == expected
    assert prepared["handoff_budget"]["api_calls"] == 1
    assert prepared["handoff_budget"]["cost_usd_upper_bound"] > 0
    identity = continuation.continuation_identity(prepared)
    assert identity["cost_cap_change"]["old_total_usd"] == 27
    assert identity["cost_cap_change"]["new_total_usd"] == 50
    assert identity["continuation_offset"] == 5
    assert identity["original_budget_snapshot"] == original
    assert {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("cap", [None, 0, -1, float("inf"), float("nan"), True, 27, 26, 27.01])
def test_invalid_nonincreasing_or_insufficient_total_budgets_fail_closed(tmp_path, cap):
    source = tmp_path / "source"
    make_source(source)
    with pytest.raises(ValueError):
        continuation.prepare_source(source, cap)


@pytest.mark.parametrize("change", ["running", "boundary", "gripper", "prediction", "ledger", "raw_ledger", "missing_result", "source"])
def test_source_invariants_fail_closed_before_any_simulation_or_paid_call(tmp_path, change):
    source = tmp_path / "source"
    rows, _ = make_source(source)
    if change == "running":
        path = source / "run_state.json"
        data = json.loads(path.read_text())
        data["status"] = "running"
        path.write_text(json.dumps(data))
    elif change == "boundary":
        path = source / "episodes.jsonl"
        data = json.loads(path.read_text())
        data["steps"] = 4
        path.write_text(json.dumps(data) + "\n")
    elif change in ("gripper", "prediction"):
        if change == "gripper":
            rows[1]["action"][-1] = 1
        else:
            rows[1]["diagnostics"]["prediction"]["gripper_command"] = "close"
        save_rows(source, rows)
    elif change == "ledger":
        path = source / "run_state.json"
        data = json.loads(path.read_text())
        data["budget"]["cost_usd_upper_bound"] = 0
        path.write_text(json.dumps(data))
    elif change == "raw_ledger":
        path = next((source / "api").glob("*.result.json"))
        data = json.loads(path.read_text())
        data["budget"]["api_calls"] = 0
        path.write_text(json.dumps(data))
    elif change == "missing_result":
        next((source / "api").glob("*.result.json")).unlink()
    else:
        path = source / "manifest.json"
        data = json.loads(path.read_text())
        data["configuration"]["software"]["source_sha256"]["action.py"] = "changed"
        data["identity_hash"] = continuation.recovery.canonical_hash(data["configuration"])
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        continuation.prepare_source(source, 50)


@pytest.mark.parametrize("mismatch", ["image", "state"])
def test_replay_gate_blocks_paid_transport_on_image_or_state_mismatch(tmp_path, mismatch):
    source = tmp_path / "source"
    make_source(source)
    prepared = continuation.prepare_source(source, 50)
    simulations = []

    def sim_factory(config):
        sim = FakeSimulator(config, mismatch=mismatch == "image", state_error=1e-6 if mismatch == "state" else 0)
        simulations.append(sim)
        return sim

    def client_factory(config, **kwargs):
        kwargs["transport"] = lambda *args: pytest.fail("no inference before full replay agreement")
        return AstraClient(config, **kwargs)

    result = continuation.execute_continuation(prepared, tmp_path / "child", api_key="dummy",
        simulator_factory=sim_factory, client_factory=client_factory, recorder_factory=FakeRecorder)
    assert result["status"] == "infrastructure_error"
    assert result["new_evaluated_actions"] == result["additional_api_calls"] == 0
    assert result["libero_success"] is None and not result["evaluation_completed"]
    assert simulations[0].closed
    assert json.loads((tmp_path / "child/replay_validation.json").read_text())["status"] == "validation_failed"


def test_same_simulator_handoff_preserves_cadence_global_api_ordinals_and_spend(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _, original = make_source(source)
    prepared = continuation.prepare_source(source, 50)
    output = tmp_path / "child"
    simulations, calls = [], []

    def sim_factory(config):
        sim = FakeSimulator(config)
        simulations.append(sim)
        return sim

    def client_factory(config, **kwargs):
        def transport(*args):
            assert len(simulations) == 1 and len(simulations[0].actions) == 5
            assert not simulations[0].closed
            assert json.loads((output / "replay_validation.json").read_text())["status"] == "validated"
            assert continuation.os.environ.get("OPENAI_API_KEY") is None
            calls.append(args)
            return envelope()
        kwargs["transport"] = transport
        return AstraClient(config, **kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    result = continuation.execute_continuation(prepared, output, api_key="dummy",
        simulator_factory=sim_factory, client_factory=client_factory, recorder_factory=FakeRecorder)
    assert result["status"] == "success" and result["libero_success"] is True
    assert result["reconstruction_actions"] == 5 and result["new_evaluated_actions"] == 1
    assert result["cumulative_actions"] == 6 and result["additional_api_calls"] == len(calls) == 1
    assert result["budget_after"]["api_calls"] == original["api_calls"] + 1
    assert result["budget_after"]["cost_usd_upper_bound"] > original["cost_usd_upper_bound"]
    request = json.loads(next((output / "api").glob("*.request.json")).read_text())
    assert request["request_ordinal"] == 2 and request["configuration"]["max_cost_usd"] == 50
    with gzip.open(output / "continuation.jsonl.gz", "rt") as stream:
        trace = [json.loads(line) for line in stream]
    assert trace[0]["event"] == "continuation_start"
    assert trace[1]["step"] == trace[1]["diagnostics"]["step"] == 5
    assert trace[1]["diagnostics"]["refreshed"] is True
    assert trace[1]["diagnostics"]["stage_changed"] is False
    assert simulations[0].closed


def test_continued_budget_stop_is_incomplete_and_retains_prior_spend(tmp_path):
    source = tmp_path / "source"
    _, original = make_source(source)
    prepared = continuation.prepare_source(source, 50)

    class StopsAtBudget(AstraClient):
        def predict(self, *args, **kwargs):
            raise BudgetExceeded("simulated finite budget reached")

    result = continuation.execute_continuation(prepared, tmp_path / "child", api_key="dummy",
        simulator_factory=FakeSimulator, client_factory=StopsAtBudget, recorder_factory=FakeRecorder)
    assert result["status"] == "budget_exceeded" and result["libero_success"] is None
    assert not result["evaluation_completed"] and "success_rate" not in result
    assert result["budget_after"]["cost_usd_upper_bound"] == original["cost_usd_upper_bound"]
    assert result["additional_api_calls"] == result["new_evaluated_actions"] == 0


def test_cli_requires_explicit_total_cap():
    with pytest.raises(SystemExit) as caught:
        continuation.main(["--source", "unused", "--output", "unused-child"])
    assert caught.value.code == 2
