"""Independent geometric checks for the exact reference used by the experiment."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import Box, OpenContainer, Sphere, make_scenes, validate_reference


def winding_inside(mesh, points):
    """Independent triangle solid-angle sign, without rtree or ray acceleration."""
    triangles = mesh.triangles
    inside = []
    for point in points:
        a, b, c = np.moveaxis(triangles-point, 1, 0)
        la, lb, lc = [np.linalg.norm(x, axis=1) for x in (a, b, c)]
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = la*lb*lc + np.einsum("ij,ij->i", a, b)*lc + np.einsum("ij,ij->i", b, c)*la + np.einsum("ij,ij->i", c, a)*lb
        angle = 2*np.arctan2(numerator, denominator)
        inside.append(abs(angle.sum()) > 2*np.pi)
    return np.array(inside)


def test_known_distances():
    sphere = Sphere(np.zeros(3), .03)
    np.testing.assert_allclose(sphere.sdf(np.array([[0, 0, 0], [.03, 0, 0], [.06, 0, 0]])), [-.03, 0, .03], atol=1e-15)
    box = Box(np.zeros(3), np.array([.01, .02, .03]), np.eye(3))
    np.testing.assert_allclose(box.sdf(np.array([[0, 0, 0], [.04, .06, .03], [.01, 0, 0]])), [-.01, .05, 0], atol=1e-15)


def test_container_true_corner_distance():
    container = OpenContainer([0, 0, 0], [.05, .04, .03], .005, .006, np.eye(3))
    # A material point near the concave cavity corner has sqrt(2)*2 mm
    # clearance to exposed cavity, rather than 2 mm from naive CSG min/max.
    p = np.array([[.047, .037, 0]])
    np.testing.assert_allclose(container.sdf(p), [-np.sqrt(2)*.002], atol=1e-15)
    np.testing.assert_allclose(container.sdf([[0, 0, 0], [0, 0, .05]]), [.024, np.hypot(.035, .02)], atol=1e-15)


@pytest.mark.parametrize("scene", make_scenes(), ids=lambda s: s.name)
def test_manifest_mesh_surface_and_probes(scene):
    mesh = scene.mesh()
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert np.all(mesh.bounds[0] >= -.10)
    assert np.all(mesh.bounds[1] <= .10)
    sample = scene.sample_surface(5000, np.random.default_rng(90))
    assert np.max(np.abs(scene.sdf(sample))) < 1e-14
    assert scene.feature_checks(scene.sdf)["all_passed"]
    assert scene.to_dict()["seed"] in (11, 23, 37)


@pytest.mark.parametrize("scene", [s for s in make_scenes() if s.category in ("rotated_box", "open_hollow_container", "thin_walls_and_narrow_gaps")], ids=lambda s: s.name)
def test_exact_reference_against_independent_triangles(scene):
    rng = np.random.default_rng(scene.seed+100)
    # Uniform plus boundary-offset points test air, cavity and thin material.
    points = np.concatenate([rng.uniform(-.09, .09, (120, 3)),
                             scene.sample_surface(120, rng)+rng.normal(scale=.001, size=(120, 3))])
    mesh = scene.mesh()
    _, unsigned, _ = trimesh.proximity.closest_point_naive(mesh, points)
    expected = unsigned.copy()
    expected[winding_inside(mesh, points)] *= -1
    np.testing.assert_allclose(scene.sdf(points), expected, atol=2e-14, rtol=1e-11)


def test_matrix_deterministic_and_area_sampling():
    scenes = make_scenes()
    assert len(scenes) == 15
    assert [s.to_dict() for s in scenes] == [s.to_dict() for s in make_scenes()]
    box = Box(np.zeros(3), np.array([1., 2., 3.]), np.eye(3))
    points = box.sample_surface(50000, np.random.default_rng(1))
    observed = np.isclose(np.abs(points), box.half_extents).mean(axis=0)
    np.testing.assert_allclose(observed, np.array([6, 3, 2])/11, atol=.009)


def test_reference_gate_and_width_measurements():
    report = validate_reference()
    assert report["passed"], report
    assert report["scene_count"] == 15
    for scene in make_scenes():
        features = scene.feature_checks(scene.sdf)
        for section in features["cross_sections"]:
            assert section["crossing_count_matches"]
            np.testing.assert_allclose(section["width_errors_m"], 0, atol=2e-12)
        assert scene.minimum_feature_size_m > 0


def test_feature_probes_detect_missing_material_and_closed_gap():
    scene = next(s for s in make_scenes() if s.category == "thin_walls_and_narrow_gaps")
    empty = scene.feature_checks(lambda p: np.full(len(p), .01))
    assert not empty["all_passed"]
    assert not empty["cross_sections"][0]["crossing_count_matches"]
    filled = scene.feature_checks(lambda p: np.full(len(p), -.01))
    assert not filled["all_passed"]
