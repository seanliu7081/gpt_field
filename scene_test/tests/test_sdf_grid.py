"""Meaningful numerical and coordinate-contract checks, without robot imports."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_deps"))
from sdf_grid import SDFGrid


class AffineScene:
    def sdf(self, p):
        return p @ np.array([1., 2., -3.]) + .01


class SphereScene:
    center = np.array([.013, -.009, .017])
    radius = .041

    def sdf(self, p):
        return np.linalg.norm(p - self.center, axis=1) - self.radius


class GridTests(unittest.TestCase):
    def test_affine_interpolation_xyz_and_boundary(self):
        scene = AffineScene()
        grid = SDFGrid.build(scene, .004)
        points = np.random.default_rng(101).uniform(-.12, .12, (4000, 3))
        points = np.vstack([points, grid.lower, grid.upper])
        np.testing.assert_allclose(grid.query(points), scene.sdf(points), atol=3e-8, rtol=0)

    def test_outside_is_unknown_not_free(self):
        grid = SDFGrid.build(AffineScene(), .004)
        result = grid.query([[.120001, 0, 0], [0, -.120001, 0], [np.nan, 0, 0]])
        self.assertTrue(np.isnan(result).all())

    def test_roundtrip_does_not_overwrite(self):
        grid = SDFGrid.build(SphereScene(), .004)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field.npz"
            grid.save(path)
            loaded = SDFGrid.load(path)
            np.testing.assert_array_equal(grid.values, loaded.values)
            np.testing.assert_array_equal(grid.upper, loaded.upper)
            with self.assertRaises(FileExistsError):
                grid.save(path)

    def test_marching_cube_axis_origin_sign_and_accuracy(self):
        scene = SphereScene()
        grid = SDFGrid.build(scene, .002)
        mesh = grid.surface()
        self.assertIsNotNone(mesh)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume, 0)
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), scene.center, atol=.00008)
        self.assertLess(np.max(np.abs(scene.sdf(mesh.vertices))), .00004)
        self.assertLess(grid.query(scene.center[None])[0], 0)
        self.assertGreater(grid.query([[.1, 0, 0]])[0], 0)

    def test_empty_surface(self):
        self.assertIsNone(SDFGrid(np.ones((5, 5, 5)), .01, [0, 0, 0]).surface())


if __name__ == "__main__":
    unittest.main()
