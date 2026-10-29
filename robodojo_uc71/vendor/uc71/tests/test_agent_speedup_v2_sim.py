"""agent_SpeedUp_v2 simulator worker / process client without a simulator, plus recorded replay regressions."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gpt_field import agent_speedup_v2_sim as v2sim
from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_option_validation():
    config, monitor = build_v2_sim_config("libero_10_t2")
    assert v2sim.validate_v2_sim_config(config) is config
    with pytest.raises(ValueError):
        v2sim.validate_v2_sim_config(replace(config, stop_on_collision=True))
    chained, _ = build_v2_sim_config("drawer_bowl")
    assert v2sim.validate_v2_sim_config(chained).collision_check_mode == "task_manipulation_v1"
    assert v2sim.validate_monitor_options(None) == {"goal_aware": True, "truth_window": True}
    for bad in ({"goal_aware": 1}, {"unknown": True}):
        with pytest.raises(ValueError):
            v2sim.validate_monitor_options(bad)


def test_worker_command_and_own_session(monkeypatch):
    config, monitor = build_v2_sim_config("libero_10_t2")
    command = v2sim.worker_command(config, monitor)
    assert command[1:4] == ["-u", "-m", "gpt_field.agent_speedup_v2_sim"] and "--monitor-json" in command
    assert json.loads(command[command.index("--monitor-json") + 1]) == monitor
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            seen.update(cmd=cmd, **kwargs)
            raise OSError("not starting a simulator in a unit test")
    monkeypatch.setattr(v2sim.subprocess, "Popen", FakePopen)
    process = v2sim.V2SimProcess(config, monitor_options=monitor)
    with pytest.raises(OSError):
        process.start()
    assert seen["start_new_session"] is True and seen["cmd"] == command
    assert str(ROOT / "src") in seen["env"]["PYTHONPATH"].split(":")


class _FakeGeneric:
    def snapshot(self):
        return {"mode": "generic_contact_v2", "detected": True}


class _FakeEnv:
    def __init__(self, success):
        self.success, self.calls = success, 0

    def check_success(self):
        self.calls += 1
        return self.success


def test_worker_status_update_contract():
    """_update_status: the generic record replaces the base status record; a chained v1 record is kept apart;
    libero_success is the worker's own check_success() evaluation (PF-4), not a copy of the base success."""
    worker = object.__new__(v2sim.V2LiberoWorker)
    worker.generic = _FakeGeneric()
    worker._collision_monitor = object()
    worker.env = _FakeEnv(True)
    worker.status = {"success": True, "libero_success": True, "collision": {"mode": "task_manipulation_v1"}}
    worker._update_status(worker._raw_success())
    assert worker.env.calls == 1
    assert worker.status["collision"]["mode"] == "generic_contact_v2"
    assert worker.status["collision_v1_record_only"] == {"mode": "task_manipulation_v1"}
    assert worker.status["libero_success"] is True and worker.status["collision_free_success"] is False
    worker._collision_monitor = None
    worker.env = _FakeEnv(False)
    worker.status = {"success": True}      # the worker's evaluation is independent; the runner compares the two
    worker._update_status(worker._raw_success())
    assert "collision_v1_record_only" not in worker.status and worker.status["libero_success"] is False


def test_chained_raw_success_must_match_the_base_worker():
    worker = object.__new__(v2sim.V2LiberoWorker)
    worker._collision_monitor = object()
    worker.env = _FakeEnv(True)
    worker.status = {"success": False, "libero_success": False}
    with pytest.raises(RuntimeError, match="differs"):
        worker._raw_success()


def test_process_in_flight_flag(monkeypatch):
    """PF-1: the flag stays True only when an interrupt leaves a request's reply unread."""
    config, monitor = build_v2_sim_config("libero_10_t2")
    process = v2sim.V2SimProcess(config, monitor_options=monitor)
    assert process._in_flight is False
    behaviour = {}

    def fake_request(self, request):
        if behaviour.get("raise"):
            raise behaviour["raise"]
        return {"ok": True}
    monkeypatch.setattr(v2sim.SimProcess, "_request", fake_request)
    assert process._request({"op": "x"}) == {"ok": True} and process._in_flight is False
    behaviour["raise"] = RuntimeError("worker error reply consumed")
    with pytest.raises(RuntimeError):
        process._request({"op": "x"})
    assert process._in_flight is False
    behaviour["raise"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        process._request({"op": "step"})
    assert process._in_flight is True


def test_worker_main_reports_errors_as_json(tmp_path):
    """The JSONL worker answers malformed and unknown ops with ok=false (no simulator is constructed for them
    only if construction succeeds; here the LIBERO root is invalid, so the error text comes back)."""
    config, monitor = build_v2_sim_config("libero_10_t2", libero_root=str(tmp_path))
    proc = subprocess.run(v2sim.worker_command(replace(config, python_executable=sys.executable), monitor),
                          input='{"op": "metadata"}\n', capture_output=True, text=True, timeout=120,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    reply = json.loads(proc.stdout.splitlines()[0])
    assert reply["ok"] is False and "LIBERO" in reply["error"]


def test_initial_observation_hash_recipe():
    observation = {"b": [1.0, 2.0], "a": "x"}
    import hashlib
    expected = hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert v2sim.initial_observation_sha256(observation) == expected


@pytest.mark.parametrize("name, first, v1_steps, generic_steps, both", [
    ("bowl002", 340, 41, 48, 41),
    ("drawer003", 129, 101, 99, 99),
])
def test_recorded_monitor_regression(name, first, v1_steps, generic_steps, both):
    """scripts/agent_speedup_v2_replay.py on the v1 runs: first events and per-step agreement (spec §6.4)."""
    path = ROOT / f"runs/agent_speedup_v2_m0verify_fix_replay_{name}/replay.json"
    if not path.is_file():
        pytest.skip("M0 replay record not present")
    data = json.loads(path.read_text())
    current = hashlib.sha256((ROOT / "src/gpt_field/agent_speedup_v2_contacts.py").read_bytes()).hexdigest()
    assert data["contacts_module_sha256"] == current, "stale replay record: rerun scripts/agent_speedup_v2_replay.py"
    assert data["bitwise"] is True and data["max_abs_state8_deviation"] == 0.0
    assert data["render_mismatch_max_abs_grey"] <= 1
    comparison = data["monitor_comparison"]
    assert comparison["v1_first_event_step"] == comparison["generic_first_event_step"] == first
    assert (comparison["v1_forbidden_steps"], comparison["generic_forbidden_steps"], comparison["both"]) == (
        v1_steps, generic_steps, both)
