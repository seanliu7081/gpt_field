"""Ordered completion, raw-goal separation and post-terminal safety; no model."""
from types import SimpleNamespace

import pytest

from gpt_field.agent_speedup_v2_user_check_sim import UserCheckLiberoWorker
from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
from gpt_field.agent_speedup_v2_truth import physical_summary
from gpt_field.ordered_task_evaluation import (
    FINAL_KEYS, OrderedProgress, qualify_status, qualify_truth,
)

KEY = "drawer_bowl_right_moka_stove_left_no_obstacles"
INITIAL = dict(bowl_on_cabinet_top=False, drawer_closed=True, stove_on=False,
               moka_on_stove=False, drawer_open=False, bowl_inside_drawer=True,
               stove_off=True)
OPEN = dict(INITIAL, drawer_closed=False, drawer_open=True)
BOWL_TOP = dict(OPEN, bowl_inside_drawer=False, bowl_on_cabinet_top=True)
CLOSED = dict(BOWL_TOP, drawer_open=False, drawer_closed=True)
STOVE_ON = dict(CLOSED, stove_off=False, stove_on=True)
FINAL = dict(STOVE_ON, moka_on_stove=True)
TRACE = (OPEN, BOWL_TOP, CLOSED, STOVE_ON, FINAL)


def tracker():
    result = OrderedProgress()
    assert result.reset(INITIAL)["initial_state_valid"]
    return result


def test_normal_order_and_final_state_alone():
    p = tracker()
    p.observe(FINAL)
    assert p.stage == 0 and not p.ordered_complete
    p.reset(INITIAL)
    for index, predicates in enumerate(TRACE, 1):
        result = p.observe(predicates)
        assert result["completed_stage_count"] == result["observation_index"] == index
    assert p.ordered_complete
    p.reset(FINAL)
    for predicates in TRACE:
        p.observe(predicates)
    assert not p.ordered_complete


@pytest.mark.parametrize("early", ["stove_on", "moka_on_stove"])
def test_early_action_must_be_repeated_after_prerequisite(early):
    p = tracker()
    p.observe(dict(INITIAL, **{early: True}))
    for predicates in TRACE:
        p.observe(dict(predicates, **{early: True}))
    assert not p.ordered_complete
    if early == "stove_on":
        p.observe(CLOSED)
        p.observe(STOVE_ON)
    else:
        p.observe(STOVE_ON)
    p.observe(FINAL)
    assert p.ordered_complete


def test_raw_goal_never_terminates_without_sequence_and_horizon_is_preserved():
    p = tracker()
    raw = dict(libero_success=True, success=True, terminated=True, horizon_reached=False,
               collision={"detected": False}, collision_free_success=True)
    result = qualify_status(raw, p, KEY, stop_on_success=True)
    assert result["libero_success"] and result["collision_free_success"]
    assert not result["success"] and not result["ordered_success"] and not result["terminated"]
    assert raw["success"] and raw["terminated"]  # no in-place status mutation
    assert qualify_status(dict(raw, horizon_reached=True), p, KEY, stop_on_success=True)["terminated"]
    for predicates in TRACE:
        p.observe(predicates)
    result = qualify_status(raw, p, KEY, stop_on_success=True)
    assert result["success"] and result["ordered_success"] and result["terminated"]
    assert not qualify_status(raw, p, KEY, stop_on_success=False)["terminated"]
    assert not qualify_status(dict(raw, libero_success=False), p, KEY, stop_on_success=True)["success"]


def truth(strict):
    return dict(raw_conjunction=True, check_success=True, strict_conjunction=strict,
                atoms=[dict(atom=["on", "moka_pot_1", "flat_stove_1_cook_region"],
                            strict=strict, why_not=[] if strict else ["not_held", "at_rest"])])


def test_diagnostic_release_cannot_rescue_missing_order_but_keeps_hold_only_rule():
    p = tracker()
    frozen = p.snapshot()
    terminal = qualify_truth(truth(False), p, KEY)
    settled = qualify_truth(truth(True), p, KEY)
    assert settled["raw_strict_conjunction"] and not settled["strict_conjunction"]
    assert settled["raw_conjunction"] and settled["check_success"]
    assert settled["ordered_completion_why_not"] == ["ordered_sequence_incomplete"]
    assert not physical_summary(terminal, settled, raw_libero_success=True)["v2_success"]
    assert p.snapshot() == frozen
    for predicates in TRACE:
        p.observe(predicates)
    terminal = qualify_truth(truth(False), p, KEY)
    settled = qualify_truth(truth(True), p, KEY)
    physical = physical_summary(terminal, settled, raw_libero_success=True)
    assert physical["terminal_failure_is_hold_only"] and physical["v2_success"]


class FakeEnv:
    def __init__(self):
        self.env = self
        self.predicates = dict(INITIAL)

    def check_success(self):
        return all(self.predicates[k] for k in FINAL_KEYS)

    def _eval_predicate(self, atom):
        op, obj, *target = atom
        key = {"close": "drawer_closed", "open": "drawer_open", "in": "bowl_inside_drawer",
               "turnon": "stove_on", "turnoff": "stove_off"}.get(op)
        if op == "on":
            key = "bowl_on_cabinet_top" if obj.startswith("akita") else "moka_on_stove"
        return self.predicates[key]


def fake_worker():
    worker = object.__new__(UserCheckLiberoWorker)
    worker.env = FakeEnv()
    worker._collision_monitor = None
    worker.status = dict(steps=0, success=False, libero_success=False, terminated=False,
                         horizon_reached=False, collision={"detected": False})
    worker.config = SimpleNamespace(stop_on_success=True)
    worker._ordered_task_key = KEY
    worker._ordered_progress = tracker()
    worker._ordered_observed_steps = 0
    worker.counterfactual_applied = False
    return worker


def test_worker_counts_each_real_step_even_before_raw_success_exactly_once():
    worker = fake_worker()
    for index, predicates in enumerate(TRACE, 1):
        worker.status["steps"] = index
        worker.env.predicates = predicates
        assert worker._raw_success() == (index == 5)
        assert worker._ordered_progress.stage == index
        assert worker._raw_success() == (index == 5)
        assert worker._ordered_progress.observation_index == index
    worker.counterfactual_applied = True
    worker.status["steps"] += 1
    worker.env.predicates = INITIAL
    worker._raw_success()
    assert worker._ordered_progress.observation_index == 5


def test_worker_refuses_missed_policy_step():
    worker = fake_worker()
    worker.status["steps"] = 2
    with pytest.raises(RuntimeError, match="missed an actual policy step"):
        worker._raw_success()


def test_worker_step_retains_observation_and_qualifies_status(monkeypatch):
    worker = fake_worker()
    observation = {"rgb": "only allowed observation channels"}

    def fake_step(self, predicates):
        self.env.predicates = predicates
        self.status["steps"] += 1
        raw = self._raw_success()
        self.status.update(libero_success=raw, success=raw, terminated=raw)
        return {"observation": observation, "status": self.status.copy()}

    monkeypatch.setattr(V2LiberoWorker, "step", fake_step)
    response = worker.step(FINAL)
    assert response["observation"] is observation
    assert response["status"]["libero_success"]
    assert not response["status"]["success"] and not response["status"]["terminated"]
    assert "ordered_task" not in response["observation"]


def test_official_worker_path_keeps_parent_response(monkeypatch):
    worker = fake_worker()
    worker._ordered_progress = None
    response = {"observation": {"native": True}, "status": {"libero_success": True, "success": True}}
    monkeypatch.setattr(V2LiberoWorker, "step", lambda self, action: response)
    assert worker.step([0] * 7) is response
