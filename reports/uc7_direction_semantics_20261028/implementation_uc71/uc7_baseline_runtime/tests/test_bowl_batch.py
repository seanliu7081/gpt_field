import importlib.util
import json
from pathlib import Path
import threading

import pytest

from gpt_field.astra import APIError, AstraConfig
from gpt_field.libero import LiberoConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_bowl10.py"
spec = importlib.util.spec_from_file_location("evaluate_bowl10", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def configuration():
    return AstraConfig(**json.loads((SCRIPT.parents[1] / "configs/astra_bowl10.json").read_text()))


def row(root, index, status, steps=0):
    directory = batch.episode_directory(root, index)
    directory.mkdir(parents=True, exist_ok=True)
    record = {"suite": "libero_spatial", "task_id": 0, "episode_index": index,
              "status": status, "steps": steps, "method": "field"}
    (directory / "episodes.jsonl").write_text(json.dumps(record) + "\n")
    return record


def test_infrastructure_errors_never_become_zero_success_rate(tmp_path):
    row(tmp_path, 0, "api_failure")
    summary = batch.aggregate(tmp_path)
    assert summary["intended_denominator"] == 10
    assert summary["success_rate"] is None
    assert summary["observed_completed_success_rate"] is None
    assert summary["completed_evaluations"] == 0
    assert summary["infrastructure_failures"] == 1
    assert summary["pending_or_running"] == 9
    assert summary["status"] == "provisional"
    batch.write_gallery(tmp_path, summary)
    assert "provisional" in (tmp_path / "index.html").read_text()


def test_final_denominator_requires_all_ten_proper_terminal_outcomes(tmp_path):
    for index in range(10):
        row(tmp_path, index, "success" if index < 3 else "timeout", 100 if index < 3 else 220)
    summary = batch.aggregate(tmp_path)
    assert summary["status"] == "complete"
    assert summary["success_rate"] == 0.3
    assert summary["task_failures"] == 7
    assert summary["infrastructure_failures"] == 0
    assert summary["wilson_95"][0] < 0.3 < summary["wilson_95"][1]
    row(tmp_path, 9, "timeout", 100)
    provisional = batch.aggregate(tmp_path)
    assert provisional["success_rate"] is None
    assert provisional["completed_evaluations"] == 9
    assert provisional["infrastructure_failures"] == 1


@pytest.mark.parametrize("message,body", [
    ("OpenAI HTTP 401", "invalid_api_key SECRET"),
    ("OpenAI HTTP 429", '{"error":{"code":"insufficient_quota","message":"SECRET"}}'),
])
def test_account_failure_stops_later_dispatch_and_redacts_secret(message, body):
    gate = batch.DispatchGate()
    calls = []

    def denied(*args):
        calls.append(args)
        raise APIError(message + " SECRET", response_body=body)

    send = gate.transport("SECRET", denied)
    with pytest.raises(APIError) as caught:
        send("fake", {}, {"Authorization": "Bearer SECRET"}, 1)
    assert "SECRET" not in str(caught.value)
    assert "SECRET" not in caught.value.response_body
    assert gate.stopped.is_set()
    with pytest.raises(APIError, match="dispatch stopped"):
        send("fake", {}, {}, 1)
    assert len(calls) == gate.dispatched == 1


def test_ordinary_rate_limit_does_not_trigger_global_account_stop():
    gate = batch.DispatchGate()

    def limited(*args):
        raise APIError("OpenAI HTTP 429", response_body='{"error":{"code":"rate_limit_exceeded"}}')

    with pytest.raises(APIError):
        gate.transport("key", limited)("fake", {}, {}, 1)
    assert not gate.stopped.is_set()


def test_prompt_guard_runs_before_model_request():
    class Observation:
        prompt = "a different task"

    class Client:
        def predict(self, *args, **kwargs):
            pytest.fail("model must not be called for the wrong task")

    policy = batch.BowlFieldPolicy(Client())
    with pytest.raises(ValueError, match="Task text"):
        policy.act(Observation())


def test_parallel_coordinator_runs_exact_indices_and_resume_keeps_outcomes(tmp_path, monkeypatch):
    calls = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def fake_run(directory, policy, sim_config, *, task_ids, episode_indices, client, phase, resume, save_video):
        assert task_ids == [0]
        assert phase == "frozen" and save_video
        assert policy.update_steps == 5
        assert policy.planner.config.horizon == 3
        assert policy.planner.config.step_m == 0.01
        assert sim_config.max_steps == 220
        assert batch.os.environ.get("OPENAI_API_KEY") is None
        assert client._api_key == "transient-test-secret"
        index = episode_indices[0]
        with lock:
            calls.append(index)
        if index < 2:
            barrier.wait(timeout=5)
        row(directory.parent, index, "success", 1)

    monkeypatch.setattr(batch, "run_libero", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "transient-test-secret")
    output = tmp_path / "batch"
    sim = LiberoConfig(suite="libero_spatial", seed=7, max_steps=220)
    summary = batch.execute_batch(output, config=configuration(), sim_config=sim,
                                  api_key="transient-test-secret", workers=2)
    assert sorted(calls) == list(range(10))
    assert summary["completed_evaluations"] == 10
    assert summary["success_rate"] == 1
    assert summary["http_dispatches_this_process"] == 0
    assert not any(b"transient-test-secret" in p.read_bytes() for p in output.rglob("*") if p.is_file())
    again = batch.execute_batch(output, config=configuration(), sim_config=sim,
                                api_key="transient-test-secret", workers=1, resume=True)
    assert again["completed_evaluations"] == 10
    assert len(calls) == 10


def test_dispatch_abort_keeps_queued_episode_visible(tmp_path):
    gate = batch.DispatchGate()
    gate.stop("authentication or quota failure")
    result = batch._run_episode(tmp_path, 0, configuration(),
                               LiberoConfig(suite="libero_spatial", max_steps=220),
                               "unused", gate, False)
    assert result["status"] == "aborted"
    assert not result["evaluation_completed"]
    summary = batch.aggregate(tmp_path, gate=gate)
    assert summary["infrastructure_failures"] == 1
    assert summary["success_rate"] is None


def test_invalid_worker_count_and_accounting_configuration_fail_early(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        batch.execute_batch(tmp_path / "bad", config=configuration(),
                            sim_config=LiberoConfig(), api_key="unused", workers=11)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="accounting rates"):
        config = configuration()
        batch.validate_config(AstraConfig(**{**batch.asdict(config), "output_price_per_million": 0}))


def test_progress_exposes_current_simulator_step_and_stage(tmp_path):
    directory = batch.episode_directory(tmp_path, 0)
    directory.mkdir()
    (directory / "run_state.json").write_text(json.dumps({
        "status": "running", "step": 25, "stage": "deciding", "budget": {"api_calls": 5},
    }))
    summary = batch.aggregate(tmp_path)
    assert summary["episodes"][0]["current_step"] == 25
    assert summary["episodes"][0]["current_stage"] == "deciding"
    assert summary["api_calls_reserved"] == 5


def test_main_preserves_virtualenv_interpreter_symlink(tmp_path, monkeypatch):
    underlying = tmp_path / "base-python"
    underlying.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(underlying)
    captured = {}

    def fake_batch(output, **kwargs):
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(batch, "execute_batch", fake_batch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert batch.main(["--output", str(tmp_path / "run"), "--sim-python", str(interpreter)]) == 0
    assert captured["sim_config"].python_executable == str(interpreter)
    assert captured["sim_config"].python_executable != str(underlying)
    assert batch.os.environ.get("OPENAI_API_KEY") is None
