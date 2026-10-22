"""Independent geometric and adversarial checks for stage-two reference proofs."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_reference import Body, ExactChecker, Part, Trajectory, box_box, continuous_check, sphere_box


def test_sphere_box_face_corner_inside_and_tangent():
    box = Part(np.zeros(3), np.array([.01, .02, .03]))
    outside = sphere_box([.04, .06, .03], .005, box)
    assert outside["status"] == "free"
    assert outside["signed_separation_m"] == pytest.approx(.045)
    assert sphere_box([0, 0, 0], .001, box)["status"] == "blocked"
    assert sphere_box([.015, 0, 0], .005, box)["status"] == "unknown"
    assert sphere_box([.0149, 0, 0], .005, box)["status"] == "blocked"
    rotation = Rotation.from_euler("xyz", [.3, -.4, .8]).as_matrix()
    transformed = Part([.03, -.02, .01], box.half, rotation)
    point = transformed.center+rotation @ np.array([.04, .06, .03])
    assert sphere_box(point, .005, transformed)["signed_separation_m"] == pytest.approx(.045)


def test_sat_axis_aligned_separation_overlap_containment_and_tangent():
    a = Part(np.zeros(3), [.01, .02, .03])
    assert box_box(a, Part([.05, 0, 0], [.01, .02, .03]))["clearance_lower_m"] == pytest.approx(.03)
    assert box_box(a, Part([.019, 0, 0], [.01, .02, .03]))["status"] == "blocked"
    assert box_box(a, Part([.02, 0, 0], [.01, .02, .03]))["status"] == "unknown"
    assert box_box(a, Part([0, 0, 0], [.001, .001, .001]))["status"] == "blocked"


def test_sat_cross_axis_only_separation():
    a = Part([0, 0, 0], [.04, .007, .005])
    b = Part([.025307975016927692, .0067356895749019555, -.03275904561538654],
             [.035, .006, .004],
             [[.1613024342648124, -.14780053279401514, -.975774834275844],
              [.44796318224862586, .8919648812353708, -.06105438553007567],
              [.8793807548780733, -.4272629789117891, .2100853036304391]])
    # Every face normal overlaps: omitting edge cross axes would falsely collide.
    face_axes = np.vstack((a.rotation.T, b.rotation.T))
    face_gaps = np.abs(face_axes@(b.center-a.center))-np.abs(face_axes@a.rotation)@a.half-np.abs(face_axes@b.rotation)@b.half
    assert face_gaps.max() < -.004
    result = box_box(a, b)
    assert result["status"] == "free"
    assert result["clearance_lower_m"] == pytest.approx(.005833134533815117, abs=1e-14)
    assert box_box(b, a)["clearance_lower_m"] == pytest.approx(result["clearance_lower_m"])


def test_near_parallel_axes_require_strict_overlap_witness():
    a = Part([0, 0, 0], [.01, .01, .01])
    rotation = Rotation.from_rotvec([0, 0, 5e-13]).as_matrix()
    b = Part([0, 0, 0], a.half, rotation)
    result = box_box(a, b)
    assert result["near_parallel_cross_axis"]
    assert result["status"] == "blocked"
    witness = np.asarray(result["strict_interior_witness_m"])
    for box in (a, b):
        assert np.min(box.half-np.abs(box.rotation.T@(witness-box.center))) > 1e-10
    assert box_box(a, Part([.0200001, 0, 0], a.half, rotation))["status"] == "free"
    assert box_box(a, Part([.02, 0, 0], a.half, rotation))["status"] == "unknown"


def test_compound_radius_covers_offset_parts_and_lattice_covers_volume():
    parts = [Part([.03, .004, 0], [.0033, .0071, .0024], Rotation.from_euler("z", .4).as_matrix()),
             Part([-.02, -.006, .003], [.0023, .0042, .0037])]
    body = Body("compound", parts=parts)
    vertices = np.concatenate([part.vertices() for part in parts])
    assert body.radius == pytest.approx(np.linalg.norm(vertices, axis=1).max())
    assert body.radius > max(np.linalg.norm(part.half) for part in parts)+.01
    volume, surface, rho = body.sample_lattice(.002)
    assert len(volume) > len(surface)
    expected = max(.5*np.linalg.norm(2*p.half/np.ceil(2*p.half/.002)) for p in parts)
    assert rho == pytest.approx(expected)
    rng = np.random.default_rng(523)
    interior = np.concatenate([rng.uniform(-1, 1, (3000, 3))*p.half @ p.rotation.T+p.center for p in parts])
    assert cKDTree(volume).query(interior)[0].max() <= rho+1e-14


def test_shortest_slerp_quaternion_sign_and_midpoint_displacement_bound():
    q = Rotation.from_euler("xyz", [.3, -.7, .2]).as_quat()
    same = Trajectory([0, 0, 0], [0, 0, 0], q, -q)
    assert same.angle < 1e-14
    np.testing.assert_allclose(same.pose(.37)[1], Rotation.from_quat(q).as_matrix(), atol=1e-14)
    q0, q1 = Rotation.from_euler("z", [170, -170], degrees=True).as_quat()
    path = Trajectory([-.01, 0, 0], [.02, .01, -.005], q0, q1)
    assert path.angle == pytest.approx(np.deg2rad(20))
    np.testing.assert_allclose(path.pose(.5)[1], Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-14)
    body = Body("offset_box", parts=[Part([.02, .004, 0], [.005, .006, .007])])
    vertices = body.parts[0].vertices()
    for a, b in [(0, 1), (.13, .79), (.31, .32)]:
        t_mid, r_mid = path.pose((a+b)/2)
        midpoint_vertices = vertices @ r_mid.T+t_mid
        bound = path.displacement_bound(a, b, body)
        for t in np.linspace(a, b, 21):
            translation, rotation = path.pose(t)
            assert np.linalg.norm(vertices @ rotation.T+translation-midpoint_vertices, axis=1).max() <= bound+1e-14


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_continuous_check_catches_between_free_endpoints(kind):
    body = Body("bar", parts=[Part([0, 0, 0], [.04, .003, .003])]) if kind == "rotation" else Body("box", parts=[Part([0, 0, 0], [.005, .005, .005])])
    obstacle = Part([.035, 0, 0], [.003, .003, .003]) if kind == "rotation" else Part([0, 0, 0], [.003, .003, .003])
    if kind == "rotation":
        q0, q1 = Rotation.from_euler("z", [-45, 45], degrees=True).as_quat()
        path = Trajectory([0, 0, 0], [0, 0, 0], q0, q1)
    else:
        path = Trajectory([-.03, 0, 0], [.03, 0, 0])
    checker = ExactChecker(body, [obstacle])
    assert checker.pose(*path.pose(0))["status"] == "free"
    assert checker.pose(*path.pose(1))["status"] == "free"
    result = continuous_check(checker, path, body)
    assert result["status"] == "blocked"
    assert result["witness_time"] == .5


def test_continuous_check_finds_quarter_collision_and_endpoint_collision():
    body = Body("sphere", sphere_radius=.002)
    checker = ExactChecker(body, [Part([-.02, 0, 0], [.001, .005, .005])])
    path = Trajectory([-.04, 0, 0], [.04, 0, 0])
    assert checker.pose(*path.pose(.5))["status"] == "free"
    result = continuous_check(checker, path, body)
    assert result["status"] == "blocked"
    assert result["witness_time"] == .25
    result = continuous_check(checker, Trajectory([-.02, 0, 0], [.04, 0, 0]), body)
    assert result["status"] == "blocked"
    assert result["witness_time"] == 0


def test_free_intervals_are_proven_and_budget_exhaustion_stays_unknown():
    body = Body("sphere", sphere_radius=.003)
    obstacles = [Part([0, 0, 0], [.003, .003, .003])]
    path = Trajectory([-.03, .02, 0], [.03, .02, 0])
    checker = ExactChecker(body, obstacles)
    for t in [0, .5, 1]:
        assert checker.pose(*path.pose(t))["status"] == "free"
    shallow = continuous_check(checker, path, body, max_depth=0)
    assert shallow["status"] == "unknown"
    assert "depth_limit" in shallow["uncertain_reasons"]
    limited = continuous_check(checker, path, body, node_budget=1)
    assert limited["status"] == "unknown"
    assert limited["pose_queries"] <= 3
    assert "pose_query_budget" in limited["uncertain_reasons"]
    complete = continuous_check(checker, path, body)
    assert complete["status"] == "free"
    assert complete["certified_interval_count"] >= 2


def test_geometrically_free_motion_with_roi_margin_is_not_certified():
    class ROIRestrictedChecker:
        scalar_queries = 0

        def pose(self, translation, rotation):
            self.scalar_queries += 1
            return {"status": "free", "clearance_lower_m": 1., "roi_margin_m": -.001,
                    "reason": "body_outside_roi"}

    body = Body("sphere", sphere_radius=.003)
    result = continuous_check(ROIRestrictedChecker(), Trajectory([0, 0, 0], [.01, 0, 0]), body, max_depth=1)
    assert result["status"] == "unknown"
    assert result["certified_interval_count"] == 0


def test_zero_motion_tangency_remains_unknown_without_recursion():
    body = Body("sphere", sphere_radius=.005)
    checker = ExactChecker(body, [Part([0, 0, 0], [.01, .01, .01])])
    path = Trajectory([.015, 0, 0], [.015, 0, 0])
    result = continuous_check(checker, path, body)
    assert result["status"] == "unknown"
    assert result["pose_queries"] == 3
    assert "motion_tolerance_limit" in result["uncertain_reasons"]
