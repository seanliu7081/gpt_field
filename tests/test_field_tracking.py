"""Opt-in pose tracking: measured progress, command bounds and mask checks."""

from copy import deepcopy

import numpy as np
import pytest

from gpt_field.field import GridField
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import CandidateRecord, PlanResult, PlannerConfig
from gpt_field.policy import FieldPolicy
from gpt_field.schema import FieldPrediction, GridSpec
from gpt_field.tracking import PathTracker


IMAGE = image_to_data_url(np.zeros((4, 4, 3), dtype=np.uint8), size=4)
GRID = GridSpec(size=41, spacing_m=0.005)
CENTER = (0., 0., 0.5)


def observation(x=0., y=0., z=0.5):
    return PolicyObservation(IMAGE, IMAGE, (x, y, z, 0., 0., 0., 0.04, -0.04), "move to the target")


def prediction(motion="move", blocked=(), unknown=()):
    points = GRID.coordinates(CENTER)
    return {"stage": "approach", "orientation_target_xyzw": [0., 0., 0., 1.],
            "gripper_command": "close", "motion_mode": motion, "values": (-points[:, 0]).tolist(),
            "blocked_indices": list(blocked), "unknown_indices": list(unknown)}


class Client:
    def __init__(self, raw=None):
        self.raw = raw or prediction()
        self.calls = 0

    def predict(self, observation, grid, mode="field"):
        self.calls += 1
        return deepcopy(self.raw)


class StraightPlanner:
    config = PlannerConfig(horizon=10)

    def __init__(self):
        self.calls = 0

    def plan(self, field, position):
        self.calls += 1
        points = tuple(tuple(np.asarray(position) + (i * 0.01, 0., 0.)) for i in range(11))
        candidate = CandidateRecord(("+x",) * 10, points, .1, True, -0.1, -0.01)
        return PlanResult((.01, 0., 0.), "move", 1, 1, (candidate,), 0)


def policy(mode="tracking", **kwargs):
    client = Client()
    result = FieldPolicy(client, grid=GRID, planner_config=PlannerConfig(horizon=10), update_steps=30,
                         execution_steps=10, execution_mode=mode, **kwargs)
    result.planner = StraightPlanner()
    return result, client


def grid_field(blocked=(), unknown=()):
    return GridField(GRID, CENTER, FieldPrediction.from_dict(prediction(blocked=blocked, unknown=unknown), GRID))


def test_lagged_pose_does_not_consume_unreached_waypoints():
    tracked, client = policy()
    first = tracked.act(observation())
    lagged = tracked.act(observation(.002))
    assert first.action[0] == pytest.approx(.6)
    assert lagged.action[0] == pytest.approx(.6)
    assert lagged.diagnostics["tracking"]["progress_m"] == pytest.approx(.002)
    assert lagged.diagnostics["tracking"]["reference_xyz"] == pytest.approx([.032, 0., .5])
    assert lagged.diagnostics["planned_delta_xyz"] == [.01, 0., 0.]
    assert lagged.diagnostics["commanded_delta_xyz"] == pytest.approx([.03, 0., 0.])
    assert lagged.diagnostics["plan_step"] == 1
    assert not lagged.diagnostics["plan_refreshed"]
    assert tracked.planner.calls == client.calls == 1


def test_progress_is_measured_monotone_and_can_correct_backtracking():
    tracked, _ = policy()
    tracked.act(observation())
    tracked.act(observation(.02))
    decision = tracked.act(observation(.015))
    assert decision.diagnostics["tracking"]["progress_m"] == pytest.approx(.02)
    assert decision.diagnostics["tracking"]["reference_xyz"][0] == pytest.approx(.05)
    assert decision.action[0] * .05 == pytest.approx(.035)


def test_ten_action_window_and_thirty_action_field_refresh_are_unchanged():
    tracked, client = policy()
    decisions = [tracked.act(observation(i * .001)) for i in range(31)]
    assert [i for i, d in enumerate(decisions) if d.diagnostics["plan_refreshed"]] == [0, 10, 20, 30]
    assert [i for i, d in enumerate(decisions) if d.diagnostics["refreshed"]] == [0, 30]
    assert tracked.planner.calls == 4 and client.calls == 2


def test_command_norm_is_bounded_even_when_off_path():
    tracker = PathTracker(lookahead_m=.1, max_command_m=.025)
    tracker.reset(((0., 0., .5), (.08, 0., .5)))
    delta, details = tracker.command((0., -.02, .5), grid_field(), (-1., -1., 0.), (1., 1., 1.5))
    assert np.linalg.norm(delta) == pytest.approx(.025)
    assert details["reference_xyz"] == pytest.approx([.08, 0., .5])
    assert grid_field().segment_valid((0., -.02, .5), details["command_target_xyz"])


@pytest.mark.parametrize("mask_name", ["blocked", "unknown"])
def test_lookahead_does_not_cut_across_masked_corner(mask_name):
    index = int(np.ravel_multi_index((22, 22, 20), (GRID.size,) * 3))  # (1 cm, 1 cm, 0.5 m)
    field = grid_field(**{mask_name: [index]})
    tracker = PathTracker(lookahead_m=.04)
    tracker.reset(((0., 0., .5), (0., .02, .5), (.02, .02, .5)))
    assert not field.segment_valid(CENTER, (.02, .02, .5))
    delta, details = tracker.command(CENTER, field, (-1., -1., 0.), (1., 1., 1.5))
    assert delta == pytest.approx((0., .02, 0.))
    assert details["shortened_for_validity"]
    assert field.segment_valid(CENTER, details["command_target_xyz"])


def test_invalid_measured_start_holds_instead_of_crossing_unknown_space():
    field = grid_field(blocked=[int(np.ravel_multi_index((20, 20, 20), (GRID.size,) * 3))])
    tracker = PathTracker()
    tracker.reset((CENTER, (.05, 0., .5)))
    delta, details = tracker.command(CENTER, field, (-1., -1., 0.), (1., 1., 1.5))
    assert delta == (0., 0., 0.)
    assert details["status"] == "invalid_measured_start"


def test_workspace_bounds_reject_outside_reference():
    tracker = PathTracker()
    tracker.reset((CENTER, (.01, 0., .5), (.02, 0., .5), (.03, 0., .5)))
    delta, details = tracker.command(CENTER, grid_field(), (-1., -1., 0.), (.015, 1., 1.5))
    assert delta == pytest.approx((.01, 0., 0.))
    assert details["shortened_for_validity"]


def test_fixed_grid_boundary_does_not_allow_outside_command():
    tracker = PathTracker()
    tracker.reset(((.09, 0., .5), (.10, 0., .5), (.11, 0., .5), (.12, 0., .5)))
    delta, details = tracker.command((.09, 0., .5), grid_field(), (-1., -1., 0.), (1., 1., 1.5))
    assert delta == pytest.approx((.01, 0., 0.))
    assert details["command_target_xyz"][0] <= .1


def test_explicit_hold_keeps_gripper_and_orientation_with_zero_translation():
    client = Client(prediction(motion="hold"))
    tracked = FieldPolicy(client, grid=GRID, planner_config=PlannerConfig(horizon=10), update_steps=30,
                          execution_steps=10, execution_mode="tracking")
    for _ in range(10):
        decision = tracked.act(observation())
        assert decision.action[:3] == (0., 0., 0.)
        assert decision.action[-1] == 1.
        assert decision.diagnostics["status"] == "hold"
        assert decision.diagnostics["tracking"]["status"] == "no_selected_path"
    assert client.calls == 1


def test_path_endpoint_holds_without_advancing_new_path_by_action_count():
    tracked, _ = policy()
    tracked.act(observation())
    decision = tracked.act(observation(.1))
    assert decision.action[:3] == (0., 0., 0.)
    assert decision.diagnostics["tracking"]["status"] == "path_complete"


def test_default_fixed_execution_preserves_primitive_behavior_and_diagnostics():
    explicit, _ = policy(mode="fixed")
    default = FieldPolicy(Client(), grid=GRID, planner_config=PlannerConfig(horizon=10), update_steps=30,
                          execution_steps=10)
    default.planner = StraightPlanner()
    for x in (0., .002, -.005):
        expected = explicit.act(observation(x))
        actual = default.act(observation(x))
        assert actual.action == expected.action
        assert actual.action == pytest.approx((.2, 0., 0., 0., 0., 0., 1.))
        assert actual.diagnostics["planned_delta_xyz"] == [.01, 0., 0.]
        assert "tracking" not in actual.diagnostics
        assert "commanded_delta_xyz" not in actual.diagnostics


def test_reset_discards_tracking_progress_and_path():
    tracked, _ = policy()
    tracked.act(observation())
    tracked.act(observation(.02))
    tracked.reset()
    assert tracked.tracker.positions is None
    assert tracked.tracker.progress_m == 0.
    assert tracked.act(observation()).diagnostics["tracking"]["progress_m"] == 0.


@pytest.mark.parametrize("kwargs", [{"execution_mode": "unknown"}, {"tracking_lookahead_m": 0.},
                                    {"tracking_lookahead_m": float("nan")},
                                    {"tracking_max_command_m": .051}, {"tracking_max_command_m": True}])
def test_bad_tracking_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FieldPolicy(Client(), **kwargs)


def test_direct_cannot_enable_tracking():
    with pytest.raises(ValueError, match="direct baseline requires fixed"):
        FieldPolicy(Client(), method="direct", execution_mode="tracking")
