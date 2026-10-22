"""ROI/no-cap, geometry clipping and failure metrics for the real-scene runner."""
from pathlib import Path
import csv
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reconstruct_libero as runner
import trimesh


def box(center, half, name):
    return {"center_m": center, "half_extents_m": half, "rotation_local_to_world": np.eye(3).tolist(), "object": name}


class ReconstructionTests(unittest.TestCase):
    def test_triangle_clipping_preserves_plane_and_adds_no_caps(self):
        mesh = trimesh.Trimesh(vertices=[[-2, -2, .25], [2, -2, .25], [0, 2, .25]], faces=[[0, 1, 2]], process=False)
        clipped = runner.clip_surface_to_roi(mesh, [-.5, -.5, 0], [.5, .5, 1])
        np.testing.assert_allclose(clipped.vertices[:, 2], .25, atol=1e-15)
        self.assertTrue(np.all(clipped.vertices[:, :2] >= -.5 - 1e-15))
        self.assertTrue(np.all(clipped.vertices[:, :2] <= .5 + 1e-15))
        self.assertAlmostEqual(clipped.area, 1.)
        self.assertFalse(clipped.is_watertight)

    def test_surface_clipping_empty_and_full(self):
        mesh = trimesh.creation.box(extents=[.1, .2, .3])
        self.assertIsNone(runner.clip_surface_to_roi(mesh, [1, 1, 1], [2, 2, 2]))
        result = runner.clip_surface_to_roi(mesh, [-1, -1, -1], [1, 1, 1])
        self.assertAlmostEqual(result.area, mesh.area)

    def test_drawer_rois_are_common_and_grid_aligned(self):
        snapshots = [{"name": "drawer_closed", "boxes": [box([0, 0, 1], [.1, .08, .1], "wooden_cabinet_1")]},
                     {"name": "drawer_open", "boxes": [box([0, .1, 1], [.1, .08, .1], "wooden_cabinet_1")]}]
        rois = runner.choose_rois(snapshots)
        for key in ("lower_m", "upper_m", "score_lower_m", "score_upper_m"):
            np.testing.assert_array_equal(rois["drawer_closed"][key], rois["drawer_open"][key])
        for roi in rois.values():
            np.testing.assert_allclose(roi["lower_m"] / .004, np.round(roi["lower_m"] / .004), atol=1e-12)
            self.assertLessEqual(roi["vertex_count_at_1mm"], 50_000_000)

    def test_line_crossings_measure_material_widths(self):
        query = lambda p: np.abs(p[:, 0]) - .0123
        line = {"start_m": [-.05, 0, 0], "end_m": [.05, 0, 0]}
        result = runner.line_profile(query, line)
        np.testing.assert_allclose(result["positions_m"], [.0377, .0623], atol=1e-11)
        np.testing.assert_allclose(result["interval_widths_m"], [.0246], atol=1e-11)

    def test_partial_failure_csv(self):
        result = {"snapshot": "s", "spacing_m": .001, "status": "failed", "error": "reference query failure",
                  "near_distance": {"p95_mm": .3}, "uniform_distance": {"p95_mm": .2}, "near_occupancy": {"iou": .9}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            runner.write_csv(path, [result])
            with path.open() as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["near_p95_mm"], "0.3")
        self.assertEqual(row["surface_mean_mm"], "")

    def test_empty_surface_retains_primary_object_failure(self):
        result, display = runner.evaluate_surface(None, None, {"primary_objects": ["bowl", "plate"]},
            {"reference_surface_object": np.array(["bowl", "bowl"])}, 10, 1)
        self.assertIsNone(display)
        self.assertEqual(result["per_object"]["bowl"]["f1"]["0.002"], 0)
        self.assertIsNone(result["per_object"]["plate"]["f1"]["0.002"])

    def test_csv_preserves_failure_before_occupancy_is_available(self):
        result = {"snapshot": "s", "spacing_m": .001, "status": "failed", "near_distance": {"p95_mm": .3}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            runner.write_csv(path, [result])
            with path.open() as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["near_p95_mm"], "0.3")
        self.assertEqual(row["near_iou"], "")

    def test_unsigned_distance_matches_independent_kernel(self):
        mesh = trimesh.creation.box(extents=[.1, .2, .3])
        points = np.array([[.08, 0, 0], [0, 0, 0], [0, .1, 0]])
        np.testing.assert_allclose(runner.unsigned_mesh_distance(mesh, points), [.03, .05, 0], atol=1e-12)
        self.assertTrue(runner.validate_mesh_distance(mesh, points)["passed"])
        for point, expected in zip(points, [.03, .05, 0]):
            distance, projected = runner.unsigned_mesh_distance(mesh, point[None], return_projection=True)
            self.assertEqual(distance.shape, (1,))
            self.assertEqual(projected.shape, (1, 3))
            self.assertAlmostEqual(distance[0], expected, places=12)


if __name__ == "__main__":
    unittest.main()
