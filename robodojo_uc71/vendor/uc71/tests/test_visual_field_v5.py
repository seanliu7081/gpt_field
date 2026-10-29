"""Persistence cannot weaken the inherited geometric or transition limits."""
import numpy as np
import pytest

from gpt_field.visual_field_v4 import MetricFieldExecutorV4
from gpt_field.visual_field_v5 import MetricFieldExecutorV5
from test_visual_field_v4 import Q, box, intent, state


def test_long_intent_uses_measured_motion_and_stops_at_bound():
    executor = MetricFieldExecutorV5()
    measured = state()
    executor.configure(measured, intent(max_actions=80))
    for _ in range(80):
        chosen = executor.select(measured, Q)
        assert chosen["accepted"]
        following = measured.copy()
        following[:3] += np.asarray(chosen["action"][:3])*.05*.02
        executor.observe(measured, chosen["action"], following)
        measured = following
    assert executor.summary()["executed_actions"] == 80
    assert executor.select(measured, Q)["status"] == "intent_action_budget"
    assert executor.summary()["measured_travel_m"] < .02
    with pytest.raises(ValueError):
        MetricFieldExecutorV4().configure(state(), intent(max_actions=80))


def test_extended_action_horizon_does_not_override_arrival_or_blocked_goal():
    executor = MetricFieldExecutorV5()
    executor.configure(state(), intent(max_actions=120, goal_offset_m=[.003, 0., 0.]))
    assert executor.select(state(), Q)["status"] == "measured_local_goal_arrival"
    executor.configure(state(), intent(max_actions=120,
        obstacles=[box(low=(.065, -.02, -.02), high=(.075, .02, .02))]))
    assert executor.select(state(), Q)["status"] == "invalid_goal_hypothesis"


@pytest.mark.parametrize("budget", [9, 121, True, 80.0])
def test_horizon_limit_is_strict(budget):
    with pytest.raises(ValueError):
        MetricFieldExecutorV5().configure(state(), intent(max_actions=budget))
