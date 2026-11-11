"""Autonomous runner/recording checks, using no model, simulator or encoder."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import pytest

from gpt_field.agent_speedup_v2_sim import initial_observation_sha256
from gpt_field.foundation_vision_harness_V1.recording import AsyncVideoRecorder
from gpt_field.foundation_vision_harness_V1.runner import execute_episode, OPERATOR_STOP_FILE
from gpt_field.policy import Decision
from test_fvh_v1_client import view


TASK = SimpleNamespace(task_id=4, key="fixture", prompt="Grasp the selected object.")


class FakeSim:
    def __init__(self, observation):
        self.observation = observation
        self.config = SimpleNamespace(stop_on_success=False)
        self.metadata = {"oracle_enabled": False}
        self.status = {"libero_success": False, "terminated": False}
        self.actions = []
        self.closed = False

    def start(self):
        return self

    def list_tasks(self):
        return [{"task_id": TASK.task_id, "prompt": TASK.prompt}]

    def reset(self, task_id, episode_index):
        assert task_id == TASK.task_id
        return self.observation

    def step(self, action):
        self.actions.append(tuple(action))
        self.status["libero_success"] = True
        return self.observation, True

    def truth(self):
        return {"strict_conjunction": True, "check_success": self.status["libero_success"],
                "atoms": [{"strict": True, "fixture_evaluator_only": True}]}

    def close(self):
        self.closed = True


class FakePolicy:
    gripper_command = "close"

    def __init__(self, *, finish_after=3, stop_path=None, fail=False):
        self.calls = self.transitions = 0
        self.terminal = None
        self.finish_after = finish_after
        self.stop_path = stop_path
        self.fail = fail
        self.closed = False

    def act(self, obs):
        self.calls += 1
        assert not hasattr(obs, "strict_conjunction")
        assert not hasattr(obs, "libero_success")
        if self.fail:
            raise RuntimeError("offline policy failure")
        if self.stop_path is not None:
            self.stop_path.write_text("operator stop")
        if self.calls > self.finish_after:
            self.terminal = {"kind": "finish", "reason": "public fixture evidence", "evidence": []}
        return Decision((.05, 0., 0., 1e-8, 0., 0., 1.), {"step": self.transitions})

    def observe_transition(self, before, decision, after):
        self.transitions += 1

    def planning_snapshot(self):
        return {"status": "active", "step": self.transitions}

    def agent_stats(self):
        return {"calls": self.calls}

    def close(self):
        self.closed = True


class FakeClient:
    closed = False

    def close(self):
        self.closed = True


def launch(tmp_path, *, policy=None, expected=None):
    obs = view().observation
    sim = FakeSim(obs)
    client = FakeClient()
    policy = policy or FakePolicy()
    result = execute_episode(tmp_path / "trial", task=TASK, episode_index=0,
        sim_factory=lambda: sim, client_factory=lambda **kwargs: client,
        policy_factory=lambda supplied: policy,
        expected_initial_observation_sha256=expected or initial_observation_sha256(obs.to_dict()))
    return result, sim, client, policy


def test_raw_success_does_not_stop_and_finish_is_checked_before_dispatch(tmp_path):
    result, sim, client, policy = launch(tmp_path)
    assert result["status"] == "finish"
    assert result["first_raw_success"]["step"] == 1
    assert result["steps"] == 3 and policy.calls == 4 and policy.transitions == 3
    assert len(sim.actions) == 13  # Three online actions, ten evaluator holds.
    assert all(action[-1] == 1 for action in sim.actions)
    assert all(action[:3] == (0., 0., 0.) for action in sim.actions[-10:])
    assert result["strict_success"] and result["completion_time_eligible"]
    assert result["completion_wall_s"] >= result["online_wall_s"]
    assert not result["video_time_subtracted"]
    assert sim.closed and policy.closed and client.closed
    rows = [json.loads(line) for line in (tmp_path / "trial/trace.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "raw_success_first_reached" for row in rows) == 1
    assert (tmp_path / "trial/truth_after_same_gripper_hold.json").is_file()


def test_stop_requested_while_deciding_withholds_action_and_hold(tmp_path):
    policy = FakePolicy(stop_path=tmp_path / "trial" / OPERATOR_STOP_FILE)
    result, sim, client, policy = launch(tmp_path, policy=policy)
    assert result["status"] == "user_stop"
    assert result["withheld_decisions_after_stop"] == 1
    assert sim.actions == [] and policy.transitions == 0
    assert result["same_gripper_hold_steps"] == 0 and not result["strict_success"]
    assert sim.closed and client.closed and policy.closed


def test_bad_reset_prevents_client_policy_and_actions(tmp_path):
    result, sim, client, policy = launch(tmp_path, expected="0"*64)
    assert result["status"] == "error"
    assert "frozen reset" in result["error"]
    assert sim.actions == [] and policy.calls == 0
    assert sim.closed


def test_policy_error_is_not_success_and_cleans_up(tmp_path):
    result, sim, client, policy = launch(tmp_path, policy=FakePolicy(fail=True))
    assert result["status"] == "error" and not result["strict_success"]
    assert result["completion_wall_s"] is None and not sim.actions
    assert sim.closed and client.closed and policy.closed


def test_async_recorder_drops_backlog_and_captures_immutable_plan(tmp_path):
    entered, release = Event(), Event()
    captured = []

    class SlowRenderer:
        def __init__(self, path, *, planning_provider, **kwargs):
            self.provider = planning_provider

        def executed(self, observation, **kwargs):
            captured.append(deepcopy(self.provider()))
            entered.set()
            assert release.wait(2.)

        def close(self, **kwargs):
            pass

    recorder = AsyncVideoRecorder(tmp_path / "unused.mp4", queue_size=1, renderer_factory=SlowRenderer)
    plan = {"stage": "contact", "keyposes": [{"position": [1., 2., 3.]}]}
    try:
        recorder.submit(view().observation, step=0, planning=plan)
        assert entered.wait(2.)
        started = time.monotonic()
        for step in range(1, 20):
            recorder.submit(None, step=step, planning=plan)
        elapsed = time.monotonic()-started
        plan["keyposes"][0]["position"][0] = 99.
        assert elapsed < .5
    finally:
        release.set()
        summary = recorder.close(status="finish")
    assert summary["dropped_events"] >= 18
    assert not summary["worker_alive"] and summary["error"] is None
    assert captured[-1]["capture"]["step"] == 19
    assert all(item["keyposes"][0]["position"][0] == 1. for item in captured)


def test_cli_does_not_offer_trial_caps_or_retry_and_requires_explicit_run_flag(tmp_path):
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_foundation_vision_harness_V1.py"
    spec = importlib.util.spec_from_file_location("fvh_v1_script_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["run", "--task", "fixture", "--sam-checkpoint", "unused",
                                            "--da3-model", "unused", "--output", str(tmp_path / "never")])
    with pytest.raises(ValueError, match="explicit user approval"):
        module.run(args, None, None, None, None, None, None, {})
    assert not (tmp_path / "never").exists()
    assert not hasattr(args, "max_steps") and not hasattr(args, "retry")
