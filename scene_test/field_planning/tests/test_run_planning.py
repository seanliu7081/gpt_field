"""Independent runner checks; no scene state or model inference is needed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_planning as runner
from joint_planner import JointPlanner, PlannerSettings, interpolate_edge
from predicted_fields import QueryResult


class FakeRobot:
    def __init__(self, own_valid=True, self_collision=False):
        self.own_valid = own_valid
        self.self_collision = self_collision
        self.sample_metadata = [{'geom_id': 1, 'geom_name': 'native_arm'},
                                {'geom_id': 2, 'geom_name': 'native_finger'}]
        self.sample_calls = 0

    def check(self, q):
        return {'valid': self.own_valid, 'self_collision': self.self_collision,
                'joint_limits_valid': self.own_valid or self.self_collision}

    def samples(self, q):
        self.sample_calls += 1
        return np.array([[.1, .2, 1.], [.1, .3, 1.], [.1, .4, 1.]]), np.array([1, 1, 2])


class FakeField:
    def __init__(self, states):
        self.states = np.asarray(states, np.int8)
        self.calls = 0

    def query(self, points):
        self.calls += 1
        assert len(points) == len(self.states)
        q = QueryResult.empty(len(points))
        q.state[:] = self.states
        q.support[:] = self.states != 0
        q.conservative_clearance_m[q.state == 1] = .001
        return q


@pytest.mark.parametrize('states,valid,reason', [
    ([1, 1, 1], True, 'sampled_clear'),
    ([1, 0, 1], False, 'unknown_space'),
    ([1, -1, 1], False, 'predicted_occupied'),
    ([0, -1, 1], False, 'predicted_occupied_and_unknown'),
    ([0, 0, 0], False, 'unknown_space'),
])
def test_collision_callback_refuses_every_unknown_or_occupied_sample(states, valid, reason):
    robot, field = FakeRobot(), FakeField(states)
    check = runner.check_robot_field(robot, field, np.zeros(7), details=True)
    assert check['valid'] is valid
    assert check['reason'] == reason
    assert check['continuous_or_full_volume_certificate'] is False
    assert sum(row['points'] for row in check['per_geom']) == 3
    assert sum(row['occupied'] for row in check['per_geom']) == states.count(-1)
    assert sum(row['unknown'] for row in check['per_geom']) == states.count(0)
    if 1 not in states:
        assert check['minimum_free_sample_clearance_lower_bound_m'] is None


@pytest.mark.parametrize('self_collision,reason', [(True, 'self_collision'), (False, 'joint_limit')])
def test_robot_limits_and_self_collision_prevent_field_query(self_collision, reason):
    robot, field = FakeRobot(False, self_collision), FakeField([1, 1, 1])
    check = runner.check_robot_field(robot, field, np.zeros(7))
    assert check['valid'] is False and check['reason'] == reason
    assert field.calls == robot.sample_calls == 0


def test_cases_depend_only_on_joint_readout_and_frozen_offsets():
    accessed = []

    class OnlyJointReadout:
        def __getattribute__(self, name):
            accessed.append(name)
            if name == 'q_initial':
                return np.arange(7, dtype=float) / 10
            raise AssertionError('case generation requested unexpected input: ' + name)

    q0 = np.arange(7, dtype=float) / 10
    cases = runner.make_cases(OnlyJointReadout(), 'drawer003_closed_step60')
    assert len(cases) == len(runner.GOAL_OFFSETS) == 10
    assert set(accessed) == {'q_initial'}
    for case, (name, offset) in zip(cases, runner.GOAL_OFFSETS):
        assert case['case_id'] == name
        assert np.array_equal(case['start_q'], q0)
        assert np.array_equal(case['goal_q'], q0 + offset)
        assert case['offset_rad'] == list(offset)


def completed_fake_records():
    rows = []
    for snapshot in runner.SNAPSHOTS:
        for method in ('astra', 'depthpro', 'oracle'):
            for case_id in ('a', 'b'):
                planned = (case_id == 'a' and method != 'astra')
                rows.append({'snapshot': snapshot, 'method': method, 'case_id': case_id,
                             'seed': 17, 'status': 'planned' if planned else 'invalid_start',
                             'truth': {'valid_at_all_samples': method == 'oracle'} if planned else None,
                             'planning_seconds': .2, 'eef_path_length_m': .3 if planned else 0.})
    return rows


def test_summary_uses_returned_path_denominator_and_does_not_call_no_paths_safe():
    summary = runner.summarize(completed_fake_records())
    assert len(summary) == 6
    for row in summary:
        assert row['queries'] == 2
        if row['method'] == 'astra':
            assert row['returned_paths'] == 0
            assert row['returned_path_truth_collision_rate'] is None
            assert row['mean_returned_eef_length_m'] is None
            assert row['refusals_with_oracle_valid_path'] == 1
        elif row['method'] == 'depthpro':
            assert row['returned_paths'] == 1
            assert row['truth_invalid_returned_paths'] == 1
            assert row['returned_path_truth_collision_rate'] == 1.
            assert row['refusals_with_oracle_valid_path'] == 0
        else:
            assert row['returned_paths'] == row['truth_valid_returned_paths'] == 1
            assert row['returned_path_truth_collision_rate'] == 0.


@pytest.mark.parametrize('seed', [29, 44])
def test_swapped_tree_detours_preserve_requested_start_goal(seed):
    def valid(q):
        return not (.42 <= q[0] <= .58 and q[1] < .75)
    settings = PlannerSettings(max_iterations=1000, extension_rad=.13,
                               edge_max_joint_step_rad=.01, shortcut_attempts=15)
    start, goal = [.9, .2], [.1, .2]
    planned = JointPlanner([[0., 1.], [0., 1.]], valid, settings).plan(start, goal, seed)
    assert planned['status'] == 'planned'
    assert np.array_equal(planned['path_q'][0], start)
    assert np.array_equal(planned['path_q'][-1], goal)
    for a, b in zip(planned['path_q'], planned['path_q'][1:]):
        assert all(valid(q) for q in interpolate_edge(a, b, .002))


def test_empty_collision_sample_set_is_rejected():
    robot = FakeRobot()
    robot.samples = lambda q: (np.empty((0, 3)), np.empty(0, np.int32))
    with pytest.raises(ValueError):
        runner.check_robot_field(robot, FakeField([]), np.zeros(7))


def test_invalid_state_codes_and_wrong_query_shape_are_rejected():
    with pytest.raises(ValueError):
        runner.check_robot_field(FakeRobot(), FakeField([1, 42, 1]), np.zeros(7))
    bad = SimpleNamespace(query=lambda points: QueryResult.empty(2))
    with pytest.raises(ValueError):
        runner.check_robot_field(FakeRobot(), bad, np.zeros(7))


def test_missing_native_geom_samples_are_rejected():
    robot = FakeRobot()
    robot.geom_ids = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        runner.check_robot_field(robot, FakeField([1, 1, 1]), np.zeros(7))


def test_empty_summary_uses_null_instead_of_nan_or_false_zero_collision_rate():
    summary = runner.summarize([])
    for row in summary:
        assert row['queries'] == row['returned_paths'] == 0
        assert row['returned_path_truth_collision_rate'] is None
        assert row['mean_planning_seconds'] is None
        assert row['mean_returned_eef_length_m'] is None


def test_summary_refusal_comparison_requires_matching_oracle_case():
    records = completed_fake_records()
    records = [r for r in records if not (r['method'] == 'oracle' and r['case_id'] == 'a')]
    with pytest.raises(ValueError):
        runner.summarize(records)
