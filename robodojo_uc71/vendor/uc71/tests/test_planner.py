import json
import unittest
from dataclasses import replace

import numpy as np

from gpt_field.baselines import waypoint_field
from gpt_field.field import GridField
from gpt_field.planner import LocalPlanner, PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec


AUX = {
    "stage": "approach",
    "orientation_target_xyzw": [0, 0, 0, 1],
    "gripper_command": "open",
    "motion_mode": "move",
}


def make_field(function, grid=None, **changes):
    grid = grid or GridSpec()
    center = (0.0, 0.0, 0.5)
    data = {
        **AUX,
        "values": [float(function(point)) for point in grid.coordinates(center)],
        "blocked_indices": [],
        "unknown_indices": [],
        **changes,
    }
    return GridField(grid, center, FieldPrediction.from_dict(data, grid))


class PlannerTests(unittest.TestCase):
    def test_diagonal_distance_does_not_stall_on_old_triangle_inequality_bug(self):
        start = (0, 0, 0.5)
        target = (0.10, 0.10, 0.5)
        field = waypoint_field(GridSpec(), start, AUX, target)
        result = LocalPlanner().plan(field, start)
        self.assertEqual(result.status, "move")
        self.assertIn(result.delta_xyz, ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0)))
        self.assertGreater(result.predicted_improvement, 0)
        # Under the old length + exact-distance objective, these moving
        # candidates all lose to waiting at sqrt(.1**2 + .1**2).
        initial_distance = np.linalg.norm(np.asarray(target) - start)
        for candidate in result.candidates:
            if candidate.valid:
                exact_terminal = np.linalg.norm(np.asarray(target) - candidate.positions[-1])
                self.assertGreater(candidate.path_length_m + exact_terminal, initial_distance)

    def test_equal_length_nonzero_paths_and_no_immediate_reversals(self):
        field = make_field(lambda point: -point[0])
        result = LocalPlanner().plan(field, field.center)
        self.assertEqual(result.delta_xyz, (0.01, 0.0, 0.0))
        self.assertEqual(result.candidate_count, 216)
        self.assertEqual(result.valid_count, 150)
        opposite = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y", "+z": "-z", "-z": "+z"}
        for candidate in result.candidates:
            self.assertEqual(len(candidate.primitives), 3)
            self.assertAlmostEqual(candidate.path_length_m, 0.03)
            if candidate.valid:
                self.assertTrue(all(opposite[a] != b for a, b in zip(candidate.primitives[:-1], candidate.primitives[1:])))
                for a, b in zip(candidate.positions[:-1], candidate.positions[1:]):
                    self.assertAlmostEqual(np.linalg.norm(np.asarray(b) - a), 0.01)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_equal_terminal_ties_prefer_immediate_progress(self):
        grid = GridSpec(size=9, spacing_m=0.01)
        target = np.array([0.01, 0.02, 0.5])
        field = make_field(lambda point: np.sum((point - target) ** 2), grid=grid)
        result = LocalPlanner().plan(field, field.center)
        self.assertEqual(result.delta_xyz, (0.0, 0.01, 0.0))
        np.testing.assert_allclose(result.candidates[result.selected_index].positions[-1], target)

    def test_numerical_ties_do_not_depend_on_potential_unit(self):
        # Exact diagonal ties remain deterministic across many positive scales.
        for scale in (1e-20, 1.0, 1e20):
            with self.subTest(scale=scale):
                field = make_field(lambda point: -scale * (point[0] + point[1]))
                result = LocalPlanner().plan(field, field.center)
                self.assertEqual(result.delta_xyz, (0.01, 0.0, 0.0))
                self.assertEqual(result.status, "move")
        # Constant nonzero fields can interpolate with a few ulps of error.
        constant = make_field(lambda point: 3.141592653589793)
        result = LocalPlanner().plan(constant, constant.center)
        self.assertEqual(result.status, "plateau_move")
        self.assertEqual(result.delta_xyz, (0.01, 0.0, 0.0))

    def test_explicit_hold_invalid_start_and_no_valid_candidates(self):
        field = make_field(lambda point: -point[0], motion_mode="hold")
        result = LocalPlanner().plan(field, field.center)
        self.assertEqual(result.status, "hold")
        self.assertEqual(result.delta_xyz, (0.0, 0.0, 0.0))
        move_field = GridField(field.grid, field.center, replace(field.prediction, motion_mode="move"))
        invalid = LocalPlanner().plan(move_field, (2.0, 0.0, 0.5))
        self.assertEqual(invalid.status, "no_valid_candidates")
        self.assertEqual(invalid.reason, "invalid_start")
        # Only the center node is valid, so every nonzero interpolation fails.
        middle = 4 * 81 + 4 * 9 + 4
        blocked = tuple(i for i in range(field.grid.count) if i != middle)
        sealed = GridField(field.grid, field.center, replace(move_field.prediction, blocked_indices=blocked))
        result = LocalPlanner().plan(sealed, sealed.center)
        self.assertEqual(result.status, "no_valid_candidates")
        self.assertEqual(result.valid_count, 0)
        self.assertEqual(result.candidate_count, 216)

    def test_flat_field_reports_plateau_but_does_not_insert_wait(self):
        field = make_field(lambda point: 0.0)
        result = LocalPlanner().plan(field, field.center)
        self.assertEqual(result.status, "plateau_move")
        self.assertEqual(result.delta_xyz, (0.01, 0.0, 0.0))
        self.assertEqual(result.predicted_improvement, 0.0)
        self.assertEqual(result.reason, "no_terminal_improvement")

    def test_workspace_constrains_whole_candidate(self):
        field = make_field(lambda point: -point[0])
        config = PlannerConfig(workspace_max=(0.015, 1, 1.5))
        result = LocalPlanner(config).plan(field, field.center)
        self.assertGreater(result.valid_count, 0)
        self.assertTrue(any(candidate.invalid_reason == "outside_workspace" for candidate in result.candidates))
        for candidate in result.candidates:
            if candidate.valid:
                self.assertTrue(all(point[0] <= 0.015 for point in candidate.positions))


if __name__ == "__main__":
    unittest.main()
