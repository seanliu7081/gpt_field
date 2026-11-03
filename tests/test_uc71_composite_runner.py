"""Ordered completion cannot be replaced by an unordered final arrangement."""
import copy
import json

import pytest

from agent_speedup_v2_fakes import FakeV2Sim, truth_result, v2_status
from test_agent_speedup_v2_runner import runner
from gpt_field.agent_speedup_v2_tasks import COMPOSITE_TASK_KEY, build_v2_sim_config, resolve_v2_task

TASK = resolve_v2_task(COMPOSITE_TASK_KEY)


def qualify(status, complete):
    status.update(success_qualification="ordered_custom_goal",
                  ordered_task={"enabled": True, "task_key": TASK.key,
                                "initial_state_valid": True, "ordered_complete": complete,
                                "completed_stage_count": 5 if complete else 0,
                                "observation_index": status["steps"], "events": []},
                  ordered_success=bool(status["libero_success"] and complete))
    status["success"] = status["terminated"] = status["ordered_success"]
    return status


def test_raw_goal_without_order_is_valid_but_not_terminal():
    status = qualify(v2_status(steps=1, success=True), False)
    assert runner._collision_record_v2(status, ordered_task_key=TASK.key)
    assert not status["success"] and not status["terminated"]
    with pytest.raises(ValueError, match="explicitly registered"):
        runner._collision_record_v2(status)


@pytest.mark.parametrize("change", [
    {"ordered_success": True}, {"success": True}, {"terminated": True},
    {"ordered_task": None}, {"success_qualification": "raw_libero_goal"},
])
def test_inconsistent_ordered_status_fails_closed(change):
    status = qualify(v2_status(steps=1, success=True), False)
    status.update(change)
    with pytest.raises(ValueError):
        runner._collision_record_v2(status, ordered_task_key=TASK.key)


class OrderedFake(FakeV2Sim):
    def __init__(self, complete_at):
        super().__init__(prompt=TASK.prompt, task_id=TASK.task_id, success_at=1,
                         terminal_truth=truth_result(True, True),
                         settled_truth=truth_result(True, True, state="after_release_settle"))
        self.complete_at = complete_at

    def reset(self, *args):
        obs = super().reset(*args)
        qualify(self.status, False)
        return obs

    def step(self, action):
        obs, _ = super().step(action)
        qualify(self.status, self.complete_at is not None and self.status["steps"] >= self.complete_at)
        return obs, self.status["success"]

    def settle_truth(self, *args):
        # Even a faulty diagnostic mutation cannot rewrite the frozen terminal order.
        qualify(self.status, True)
        return super().settle_truth(*args)


def execute(tmp_path, fake, steps=4):
    cfg, monitor = build_v2_sim_config(TASK.key)
    expected = runner.initial_observation_sha256(fake.wrist.observation().to_dict())
    return runner.execute(tmp_path / "out", config=runner.configuration(), sim_config=cfg,
                          task_key=TASK.key, episode_index=0, monitor_options=monitor,
                          sim_factory=lambda cfg: fake, video=False, smoke=True, smoke_steps=steps,
                          expected_initial_observation_sha256=expected,
                          allow_initial_state_override=True, allow_task_bddl_override=True)


def test_runner_continues_past_raw_goal_until_ordered_success(tmp_path):
    fake = OrderedFake(3)
    summary = execute(tmp_path, fake)
    assert len(fake.actions) == summary["steps"] == 3
    assert summary["status"] == "diagnostic_sim_terminated"
    assert summary["libero_success"] and summary["ordered_success"] and summary["v2_success"]
    assert summary["ordered_task"]["observation_index"] == 3
    result = json.loads((tmp_path / "out/task_evaluation.json").read_text())
    assert result["ordered_success"] and result["libero_success"]


def test_unordered_raw_success_and_diagnostic_cannot_score_completion(tmp_path):
    fake = OrderedFake(None)
    summary = execute(tmp_path, fake, steps=2)
    assert summary["steps"] == 2 and summary["status"] == "diagnostic_complete"
    assert summary["libero_success"] is True
    assert summary["ordered_success"] is False and summary["v2_success"] is False
    assert summary["ordered_task"]["ordered_complete"] is False
    assert summary["completion_time_eligible"] is False
    assert summary["physical_truth"]["ordered_failure_reason"] == "ordered_sequence_incomplete"


def test_only_composite_stopping_rule_changes():
    assert runner._task_stopping_rules(TASK, runner.STOPPING_RULE)[0] == "ordered task sequence AND final LIBERO goal"
    official = resolve_v2_task("libero_10_t2")
    assert runner._task_stopping_rules(official, runner.STOPPING_RULE) == runner.STOPPING_RULE
