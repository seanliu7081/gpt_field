import unittest
from dataclasses import replace

import numpy as np

from gpt_field.baselines import _affine_values, transform_field, waypoint_field
from gpt_field.field import GridField
from gpt_field.planner import LocalPlanner
from gpt_field.schema import FieldPrediction, GridSpec


def sample_field():
    grid = GridSpec(size=5)
    coords = grid.coordinates((0.1, 0.2, 0.5))
    values = 2 * coords[:, 0] - coords[:, 1] + coords[:, 2] ** 2
    values[0] = -1e6
    values[1] = 1e6
    prediction = FieldPrediction.from_dict({
        "stage": "carry",
        "orientation_target_xyzw": [0, 0, 0, 1],
        "gripper_command": "close",
        "motion_mode": "move",
        "values": values.tolist(),
        "blocked_indices": [0],
        "unknown_indices": [1],
    }, grid)
    return GridField(grid, (0.1, 0.2, 0.5), prediction)


class BaselineTests(unittest.TestCase):
    def test_all_invalid_masks_have_identical_hold_behavior_for_all_modes(self):
        original = sample_field()
        blocked = tuple(range(0, original.grid.count, 2))
        unknown = tuple(range(1, original.grid.count, 2))
        field = GridField(original.grid, original.center, replace(
            original.prediction, blocked_indices=blocked, unknown_indices=unknown,
        ))
        for method in ("field", "affine", "shuffled", "minimum"):
            with self.subTest(method=method):
                transformed = transform_field(field, method)
                self.assertIs(transformed, field)
                plan = LocalPlanner().plan(transformed, field.center)
                self.assertEqual(plan.status, "no_valid_candidates")
                self.assertEqual(plan.delta_xyz, (0.0, 0.0, 0.0))
                self.assertEqual(plan.valid_count, 0)
        with self.assertRaises(ValueError):
            transform_field(field, "unrecognized")

    def test_all_ablations_preserve_auxiliary_outputs_masks_and_center(self):
        field = sample_field()
        for method in ("field", "affine", "shuffled", "minimum"):
            with self.subTest(method=method):
                result = transform_field(field, method)
                original_dict = field.prediction.to_dict()
                changed_dict = result.prediction.to_dict()
                del original_dict["values"]
                del changed_dict["values"]
                self.assertEqual(changed_dict, original_dict)
                self.assertEqual(result.center, field.center)
                self.assertEqual(result.grid, field.grid)
                self.assertEqual(result.prediction.values[:2], field.prediction.values[:2])

    def test_shuffle_is_seeded_and_permutes_only_valid_values(self):
        field = sample_field()
        a = transform_field(field, "shuffled", seed=7)
        b = transform_field(field, "shuffled", seed=7)
        c = transform_field(field, "shuffled", seed=8)
        self.assertEqual(a.prediction.values, b.prediction.values)
        self.assertNotEqual(a.prediction.values, c.prediction.values)
        valid = list(field.valid_indices)
        self.assertEqual(
            sorted(a.prediction.values[i] for i in valid),
            sorted(field.prediction.values[i] for i in valid),
        )

    def test_affine_fit_removes_curvature_excludes_masks_and_is_cached(self):
        field = sample_field()
        before = _affine_values.cache_info().hits
        result = transform_field(field, "affine")
        again = transform_field(field, "affine")
        self.assertGreaterEqual(_affine_values.cache_info().hits, before + 1)
        self.assertEqual(result.prediction.values, again.prediction.values)
        coords = field.grid.coordinates(field.center)
        valid = list(field.valid_indices)
        design = np.column_stack((coords, np.ones(field.grid.count)))
        expected_coefficients = np.linalg.lstsq(design[valid], np.asarray(field.prediction.values)[valid], rcond=None)[0]
        np.testing.assert_allclose(np.asarray(result.prediction.values)[valid], design[valid] @ expected_coefficients, atol=1e-12)
        self.assertLess(max(abs(result.prediction.values[i]) for i in valid), 10)

    def test_minimum_target_excludes_invalid_low_placeholder(self):
        field = sample_field()
        valid = np.asarray(field.valid_indices)
        original = np.asarray(field.prediction.values)
        target_index = int(valid[np.argmin(original[valid])])
        result = transform_field(field, "minimum")
        coordinates = field.grid.coordinates(field.center)
        expected = np.linalg.norm(coordinates[valid] - coordinates[target_index], axis=1)
        np.testing.assert_allclose(np.asarray(result.prediction.values)[valid], expected)
        self.assertEqual(result.prediction.values[target_index], 0.0)

    def test_waypoint_uses_supplied_target_and_keeps_masks(self):
        original = sample_field()
        aux = original.prediction.to_dict()
        del aux["values"]
        target = (0.2, 0.3, 0.8)
        field = waypoint_field(original.grid, original.center, aux, target)
        expected = np.linalg.norm(original.grid.coordinates(original.center) - target, axis=1)
        np.testing.assert_allclose(field.prediction.values, expected)
        self.assertEqual(field.prediction.blocked_indices, original.prediction.blocked_indices)
        self.assertEqual(field.prediction.gripper_command, "close")
        # Live and synthetic callers can pass full validated output objects.
        full_field = waypoint_field(original.grid, original.center, original.prediction.to_dict(), target)
        full_waypoint = waypoint_field(original.grid, original.center, {**aux, "waypoint_xyz": target}, target)
        self.assertEqual(full_field.prediction, field.prediction)
        self.assertEqual(full_waypoint.prediction, field.prediction)
        with self.assertRaises(ValueError):
            transform_field(field, "unrecognized")


if __name__ == "__main__":
    unittest.main()
