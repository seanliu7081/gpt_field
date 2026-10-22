"""Check the interpolation error allowance later used for volume queries."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import Sphere, make_scenes
from sdf_grid import SDFGrid


def test_lipschitz_interpolation_bound_includes_storage_error():
    for scene in make_scenes():
        grid = SDFGrid.build(scene, .004)
        random = np.random.default_rng(scene.seed+928).uniform(-.12, .12, (5000, 3))
        near = scene.sample_surface(1000, np.random.default_rng(scene.seed))
        points = np.concatenate((random, near))
        bound = np.sqrt(3)*grid.spacing/2 + grid.vertex_quantization_max_m + 1e-12
        assert np.max(np.abs(grid.query(points)-scene.sdf(points))) <= bound


def test_bound_is_needed_at_sphere_medial_axis():
    # A sphere centre at the centre of a grid cell reaches sqrt(3)*h/2 error.
    # Good surface averages must not replace a whole-domain error allowance.
    scene = Sphere(np.array([.0005, .0005, .0005]), .003)
    grid = SDFGrid.build(scene, .001, lower=(-.01,)*3, upper=(.01,)*3)
    error = abs(grid.query(scene.center[None])[0]-scene.sdf(scene.center[None])[0])
    ideal = np.sqrt(3)*grid.spacing/2
    assert error > .99*ideal
    assert error <= ideal+grid.vertex_quantization_max_m+1e-12
