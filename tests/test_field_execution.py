"""Committed field primitives preserve direction order despite measured tracking lag."""

import copy

import numpy as np
import pytest

from gpt_field.planner import CandidateRecord, PlanResult, PlannerConfig
from gpt_field.policy import FieldPolicy
from test_policy import FakePredictor, GRID, observation, prediction


class ScriptedPlanner:
    def __init__(self, *, empty_status=None):
        self.config = PlannerConfig(horizon=10)
        self.calls = []
        self.empty_status = empty_status

    def plan(self, field, position):
        self.calls.append((tuple(field.center), tuple(position)))
        if self.empty_status is not None:
            return PlanResult((0., 0., 0.), self.empty_status, 0, 0,
                              reason="explicit_motion_hold" if self.empty_status == "hold" else "all_candidates_rejected")
        primitives = ("+x", "+y") * 5
        points = [tuple(position)]
        for primitive in primitives:
            point = list(points[-1])
            point[0 if primitive == "+x" else 1] += .01
            points.append(tuple(point))
        candidate = CandidateRecord(primitives, tuple(points), .1, True, 0., .5)
        return PlanResult((.01, 0., 0.), "move", 1, 1, (candidate,), 0, 1., 1.)


def committed_policy(*, update_steps=30, execution_steps=10, empty_status=None):
    client = FakePredictor([prediction()])
    policy = FieldPolicy(client, grid=GRID, planner_config=PlannerConfig(horizon=10),
                         update_steps=update_steps, execution_steps=execution_steps)
    planner = ScriptedPlanner(empty_status=empty_status)
    policy.planner = planner
    return policy, client, planner


def test_pending_fixed_deltas_ignore_tracking_lag_and_replan_from_actual_pose():
    policy, client, planner = committed_policy()
    plans = []
    for step in range(31):
        measured = observation(step * .0007)
        decision = policy.act(measured)
        expected = [.01, 0., 0.] if step % 2 == 0 else [0., .01, 0.]
        assert decision.diagnostics["planned_delta_xyz"] == expected
        assert np.asarray(decision.action[:3]) * .05 == pytest.approx(expected)
        assert decision.diagnostics["plan_refreshed"] is (step % 10 == 0)
        assert decision.diagnostics["plan_start_step"] == (step // 10) * 10
        assert decision.diagnostics["plan_step"] == step % 10
        assert decision.diagnostics["plan_execution_steps"] == 10
        assert ("plan_latency_s" in decision.diagnostics) == (step % 10 == 0)
        if step % 10 == 0:
            plans.append(copy.deepcopy(decision.diagnostics["plan"]))
        assert decision.diagnostics["plan"] == plans[-1]
        assert len(planner.calls) == step // 10 + 1
    assert [call[1] for call in planner.calls] == [observation(step * .0007).state[:3] for step in (0, 10, 20, 30)]
    assert len(client.calls) == 2


def test_nondivisible_field_interval_truncates_commit_and_refresh_discards_suffix():
    policy, client, planner = committed_policy(update_steps=25)
    decisions = [policy.act(observation(step * .001)) for step in range(31)]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["plan_refreshed"]] == [0, 10, 20, 25]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["refreshed"]] == [0, 25]
    assert decisions[20].diagnostics["plan_execution_steps"] == 5
    assert decisions[24].diagnostics["plan_step"] == 4
    assert decisions[25].diagnostics["plan_step"] == 0
    assert decisions[25].diagnostics["plan_execution_steps"] == 10
    assert decisions[25].diagnostics["planned_delta_xyz"] == [.01, 0., 0.]
    assert planner.calls[-1][0] == planner.calls[-1][1] == observation(.025).state[:3]
    assert len(client.calls) == 2


def test_orientation_adapter_recomputes_error_from_actual_state_during_commit():
    raw = prediction(gripper="close")
    raw["orientation_target_xyzw"] = [0., 0., np.sin(.1), np.cos(.1)]
    policy, client, planner = committed_policy()
    client.predictions = [raw]
    first = policy.act(observation())
    assert first.action[5] == pytest.approx(.15 / .5)
    current = observation(.07)
    state = list(current.state)
    state[5] = .2
    measured = type(current)(current.image, current.wrist_image, tuple(state), current.prompt)
    second = policy.act(measured)
    assert second.action[3:6] == pytest.approx((0., 0., 0.))
    assert second.action[:3] == pytest.approx((0., .2, 0.))
    assert first.action[6] == second.action[6] == 1.
    assert len(planner.calls) == 1


@pytest.mark.parametrize("status", ["hold", "no_valid_candidates"])
def test_no_selected_trajectory_commits_zero_translation_until_next_plan(status):
    policy, _, planner = committed_policy(empty_status=status)
    for step in range(11):
        decision = policy.act(observation(step * .001))
        assert decision.action[:3] == (0., 0., 0.)
        assert decision.diagnostics["planned_delta_xyz"] == [0., 0., 0.]
        assert decision.diagnostics["status"] == status
        assert decision.diagnostics["plan_refreshed"] is (step in (0, 10))
    assert len(planner.calls) == 2


def test_reset_and_explicit_clear_discard_suffix_without_automatic_actions():
    policy, client, planner = committed_policy()
    policy.act(observation())
    assert len(policy._plan_deltas) == 9
    assert policy.step_index == 1 and len(planner.calls) == 1
    policy.clear_pending_plan()
    assert policy._plan_deltas == [] and policy._plan_result is None
    assert policy.step_index == 1 and len(planner.calls) == 1
    next_decision = policy.act(observation(.02))
    assert next_decision.diagnostics["plan_start_step"] == 1
    policy.reset()
    assert policy._plan_deltas == [] and policy._plan_result is None
    reset_decision = policy.act(observation(.03))
    assert reset_decision.diagnostics["plan_start_step"] == reset_decision.diagnostics["plan_step"] == 0
    assert len(client.calls) == 2


def test_legacy_single_step_execution_keeps_action_and_diagnostic_schema():
    default = FieldPolicy(FakePredictor([prediction()]), grid=GRID)
    explicit = FieldPolicy(FakePredictor([prediction()]), grid=GRID, execution_steps=1)
    for step in range(6):
        first, second = default.act(observation(step * .002)), explicit.act(observation(step * .002))
        assert first.action == second.action
        for decision in (first, second):
            assert "plan_refreshed" not in decision.diagnostics
            assert "planned_delta_xyz" not in decision.diagnostics
            for key in ("api_latency_s", "plan_latency_s"):
                decision.diagnostics.pop(key, None)
        assert first.diagnostics == second.diagnostics


@pytest.mark.parametrize("execution_steps", [0, 11, True, 1.5])
def test_invalid_execution_lengths_are_rejected(execution_steps):
    with pytest.raises(ValueError, match="execution_steps"):
        committed_policy(execution_steps=execution_steps)


def test_direct_and_single_horizon_methods_reject_multi_step_field_execution():
    with pytest.raises(ValueError, match="execution_steps"):
        FieldPolicy(FakePredictor([]), method="direct", execution_steps=2)
    with pytest.raises(ValueError, match="execution_steps"):
        FieldPolicy(FakePredictor([]), method="field_one", execution_steps=2)
