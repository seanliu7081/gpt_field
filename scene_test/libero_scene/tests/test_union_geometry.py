"""Independent continuous-union tests, especially hidden overlap surfaces."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from union_geometry import UnionScene


def box(center=(0., 0., 0.), half=(.01, .02, .03), rotation=None, name="box"):
    return {"center_m": list(center), "half_extents_m": list(half),
            "rotation_local_to_world": (np.eye(3) if rotation is None else rotation).tolist(), "object": name}


def analytic_box_sdf(points, center, half, rotation):
    q = np.abs((points-center)@rotation)-half
    return np.linalg.norm(np.maximum(q, 0), axis=1)+np.minimum(np.max(q, axis=1), 0)


def test_single_rotated_box_against_analytic_truth():
    r = Rotation.from_euler("xyz", [.3, -.2, .4]).as_matrix()
    center, half = np.array([.05, -.03, .91]), np.array([.013, .021, .017])
    scene = UnionScene("single", [box(center, half, r)])
    points = np.random.default_rng(14).uniform(center-.04, center+.04, (3000, 3))
    np.testing.assert_allclose(scene.sdf(points), analytic_box_sdf(points, center, half, r), atol=2e-12)
    assert scene.validate(sample_count=300)["passed"]


def test_overlap_internal_faces_removed_and_duplicates_preserved():
    # Two overlapping boxes form one larger box. At the overlap center,
    # min(primitive SDF) would give -5 mm, while the true union gives -10 mm.
    left, right = box((-.005, 0, 0), (.01, .01, .01)), box((.005, 0, 0), (.01, .01, .01))
    scene = UnionScene("overlap", [left, right, dict(left)])
    assert scene.sdf([[0, 0, 0]])[0] == pytest.approx(-.01, abs=1e-13)
    points = np.random.default_rng(53).uniform(-.03, .03, (3000, 3))
    np.testing.assert_allclose(scene.sdf(points), analytic_box_sdf(points, np.zeros(3), np.array([.015, .01, .01]), np.eye(3)), atol=1e-13)
    assert scene.mesh().volume == pytest.approx(.03*.02*.02)
    assert scene.validate(sample_count=300)["passed"]


def test_concave_union_distance_not_naive_csg_minimum():
    # L-shaped union cross-section. Internal point (0.5,0.5) mm is closest
    # to a concave corner (1,1) mm, not to buried primitive faces x=1/y=1.
    scene = UnionScene("concave", [box((0, -.002, 0), (.004, .003, .01)),
                                    box((-.002, 0, 0), (.003, .004, .01))])
    assert scene.sdf([[.0005, .0005, 0]])[0] == pytest.approx(-np.sqrt(2)*.0005, abs=1e-13)
    assert scene.sdf([[.002, .002, 0]])[0] == pytest.approx(.001, abs=1e-13)


def test_disjoint_floor_sdf_and_roi_irrelevance_certificate():
    floor = {"center_m": [0, 0, 0], "rotation_local_to_world": np.eye(3).tolist()}
    scene = UnionScene("floor", [box((0, 0, 1), (.05, .05, .05))], [floor])
    np.testing.assert_allclose(scene.sdf([[0, 0, -.1], [0, 0, .1], [0, 0, 1]]), [-.1, .1, -.05], atol=1e-13)
    assert scene.prove_planes_irrelevant([-.1, -.1, .9], [.1, .1, 1.1])["passed"]
    assert not scene.prove_planes_irrelevant([-.1, -.1, -.1], [.1, .1, .1])["passed"]
    with pytest.raises(ValueError, match="overlap/touch"):
        UnionScene("invalid_floor", [box()], [floor])


def test_surface_sampling_and_labels():
    scene = UnionScene("two", [box((-.05, 0, 0), name="left"), box((.05, 0, 0), name="right")])
    points = scene.sample_surface(2000, np.random.default_rng(50))
    assert np.max(np.abs(scene.sdf(points))) < 2e-12
    labels = scene.surface_object_labels(points)
    assert np.all(labels[points[:, 0] < 0] == "left")
    assert np.all(labels[points[:, 0] > 0] == "right")
    assert scene.sdf(np.empty((0, 3))).shape == (0,)


def test_winding_weighted_values_are_not_used_for_distance():
    scene = UnionScene("outside", [box()])
    p = np.array([[.003, .004, .006], [.031, .031, .031]])
    np.testing.assert_allclose(scene.sdf(p), analytic_box_sdf(p, np.zeros(3), np.array([.01, .02, .03]), np.eye(3)), atol=1e-13)
    assert "unsigned" in scene.to_dict()["distance_backend"]
