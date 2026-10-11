import json
import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np

from gpt_field.field import GridField
from gpt_field.schema import FieldPrediction, GridSpec


def prediction(grid, values=None, **changes):
    data = {
        "stage": "approach",
        "orientation_target_xyzw": [0, 0, 0, 1],
        "gripper_command": "open",
        "motion_mode": "move",
        "values": [0.0] * grid.count if values is None else list(values),
        "blocked_indices": [],
        "unknown_indices": [],
    }
    data.update(changes)
    return FieldPrediction.from_dict(data, grid)


class SchemaTests(unittest.TestCase):
    def test_coordinate_order_and_negative_potential_round_trip(self):
        grid = GridSpec(size=3, spacing_m=0.02)
        coords = grid.coordinates((1, 2, 3))
        np.testing.assert_allclose(coords[1 * 9 + 2 * 3 + 0], (1, 2.02, 2.98))
        original = prediction(grid, [-5] * grid.count)
        parsed = FieldPrediction.from_dict(json.loads(json.dumps(original.to_dict(), allow_nan=False)), grid)
        self.assertEqual(original, parsed)

    def test_schema_rejects_invalid_values_masks_and_quaternions(self):
        grid = GridSpec(size=3)
        valid = prediction(grid).to_dict()
        bad_cases = [
            {"values": [0.0] * (grid.count - 1)},
            {"values": [float("nan")] + [0.0] * (grid.count - 1)},
            {"values": [float("inf")] + [0.0] * (grid.count - 1)},
            {"values": [True] + [0.0] * (grid.count - 1)},
            {"orientation_target_xyzw": [0, 0, 0, 2]},
            {"blocked_indices": [0, 0]},
            {"blocked_indices": [0], "unknown_indices": [0]},
            {"unknown_indices": [grid.count]},
            {"unknown_indices": [True]},
            {"motion_mode": "guess"},
            {"stage": ""},
            {"extra_truth": [1, 2, 3]},
        ]
        for changes in bad_cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    FieldPrediction.from_dict({**valid, **changes}, grid)
        missing = valid.copy()
        del missing["motion_mode"]
        with self.assertRaises(ValueError):
            FieldPrediction.from_dict(missing, grid)


class FieldTests(unittest.TestCase):
    def test_affine_interpolation_boundaries_and_fixed_center(self):
        grid = GridSpec()
        center = [0.0, 0.0, 0.5]
        coordinates = grid.coordinates(center)
        values = coordinates @ np.array([2.0, -3.0, 0.5]) + 7.0
        field = GridField(grid, center, prediction(grid, values))
        center[0] = 100.0
        self.assertEqual(field.center, (0.0, 0.0, 0.5))
        point = np.array([0.007, -0.013, 0.523])
        self.assertAlmostEqual(field.query(point), float(point @ np.array([2.0, -3.0, 0.5]) + 7.0))
        self.assertIsNotNone(field.query(field.lower))
        self.assertIsNotNone(field.query(field.upper))
        self.assertIsNone(field.query(field.upper + np.array([0.001, 0, 0])))
        self.assertIsNone(field.query([float("nan"), 0, 0.5]))
        with self.assertRaises(FrozenInstanceError):
            field.center = (1, 2, 3)

    def test_masks_exclude_only_active_interpolation_corners(self):
        grid = GridSpec(size=3, spacing_m=1)
        field = GridField(grid, (0, 0, 0), prediction(grid, blocked_indices=[0]))
        self.assertIsNone(field.query((-0.5, -0.5, -0.5)))
        self.assertIsNotNone(field.query((0, -1, -1)))
        self.assertIsNone(field.query((-1, -1, -1)))
        unknown = GridField(grid, (0, 0, 0), replace(field.prediction, blocked_indices=(), unknown_indices=(0,)))
        self.assertIsNone(unknown.query((-0.5, -0.5, -0.5)))

    def test_segment_rejects_invalid_interior_despite_valid_endpoints(self):
        grid = GridSpec(size=5, spacing_m=1)
        blocked_center = 2 * 25 + 2 * 5 + 2
        field = GridField(grid, (0, 0, 0), prediction(grid, blocked_indices=[blocked_center]))
        start, end = (-2, 0, 0), (2, 0, 0)
        self.assertIsNotNone(field.query(start))
        self.assertIsNotNone(field.query(end))
        self.assertFalse(field.segment_valid(start, end))
        self.assertTrue(field.segment_valid((-2, -2, 0), (2, -2, 0)))

    def test_segment_checks_tiny_cell_corner_crossing(self):
        grid = GridSpec(size=7, spacing_m=1)
        blocked_center = 3 * 49 + 3 * 7 + 3
        field = GridField(grid, (0, 0, 0), prediction(grid, blocked_indices=[blocked_center]))
        # The invalid active-corner region is crossed for only ~1e-6 meters.
        # A uniform 1 cm or 1 mm sampling rule would miss it.
        start, end = (-2, -0.000001, 0), (0, 1.999999, 0)
        self.assertIsNotNone(field.query(start))
        self.assertIsNotNone(field.query(end))
        self.assertFalse(field.segment_valid(start, end))


if __name__ == "__main__":
    unittest.main()
