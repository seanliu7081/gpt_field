"""Reference-query and failure-provenance regressions for the SDF runner."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_deps"))
import trimesh
import reconstruct
from geometry import make_scenes
from sdf_grid import SDFGrid
from mesh_queries import exhaustive_mesh_distance


class ReconstructionTests(unittest.TestCase):
    def test_json_checkpoint_preserves_previous_record_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            reconstruct.write_json(path, {"status": "running", "completed": 3})
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                reconstruct.write_json(path, {"unsupported": object()})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_integrity_failure_prevents_complete_status(self):
        results = [{"status": "ok"}]
        self.assertEqual(reconstruct.final_run_status(results, {"source.py": False}), "integrity_failed")
        self.assertEqual(reconstruct.final_run_status(results, {"source.py": True}), "complete")
        self.assertEqual(reconstruct.final_run_status([{"status": "failed"}], {"source.py": True}), "completed_with_failures")

    def test_strict_nearest_distance_ignores_face_normal_ties(self):
        # Library's normal tie heuristic chooses the farther of these surfaces.
        vertices = [[0, 0, 0], [0, .1, 0], [.1, 0, 0],
                    [0, 0, -.000001], [.1, 0, -.000001], [0, .1, -.000001]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [3, 4, 5]], process=False)
        points = np.array([[.02, .02, .001]])
        distance = reconstruct.MeshDistance(mesh)(points)
        np.testing.assert_allclose(distance, [.001], rtol=0, atol=1e-15)
        nearest = trimesh.proximity.closest_point_naive(mesh, points)[1]
        np.testing.assert_allclose(distance, nearest, rtol=0, atol=1e-15)

    def test_box_seed23_h1mm_reference_surface_regression(self):
        scene = next(s for s in make_scenes() if s.name == "rotated_box_seed23")
        rng = np.random.default_rng(scene.seed + 210_019)
        rng.uniform(-.12, .12, (100_000, 3))
        surface = scene.sample_surface(100_000, rng)
        mesh = SDFGrid.build(scene, .001).surface()
        query = reconstruct.MeshDistance(mesh)
        expected = exhaustive_mesh_distance(mesh.triangles, surface[:12])
        np.testing.assert_allclose(query(surface[:12]), expected, rtol=0, atol=1e-13)
        self.assertTrue(query.validate(surface)["passed"])
        # This on-face point exposed the metre-scale library tolerance bug:
        # old acceleration and naive routines both reported ~0.143 mm here.
        self.assertLess(query(surface[7:8])[0], 1e-7)

    def test_partial_failure_csv_keeps_available_metrics(self):
        record = {"scene": "a", "category": "b", "spacing_m": .001, "status": "failed",
                  "uniform_distance": {"mae_mm": .1, "p95_mm": .2},
                  "near_distance": {"mae_mm": .3, "p95_mm": .4, "sampled_max_mm": .5},
                  "error": "nearest surface failed"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            reconstruct.write_csv(path, [record])
            with path.open() as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["near_p95_mm"], "0.4")
        self.assertEqual(rows[0]["surface_symmetric_mean_mm"], "")
        self.assertEqual(rows[0]["error"], record["error"])

    def test_terminal_failure_is_saved_and_existing_run_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            def fail(args):
                out.mkdir()
                reconstruct.write_json(out / "manifest.json", {"status": "running"})
                reconstruct.write_json(out / "metrics.json", {"configurations": [{"status": "ok"}]})
                raise ValueError("deliberate diagnostic failure")
            with patch.object(reconstruct, "_run", fail):
                self.assertEqual(reconstruct.run(SimpleNamespace(output=str(out))), 1)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "aborted")
            self.assertIn("deliberate diagnostic failure", manifest["terminal_error"])
            self.assertEqual(len(json.loads((out / "metrics.json").read_text())["configurations"]), 1)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with patch.object(reconstruct, "_run", side_effect=FileExistsError(out)):
                self.assertEqual(reconstruct.run(SimpleNamespace(output=str(out))), 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
