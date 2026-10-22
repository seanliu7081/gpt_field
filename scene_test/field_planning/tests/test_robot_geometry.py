"""Native robot geometry, package isolation and bounded evaluator checks."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from robot_geometry import (RobotGeometry, GroundTruthEvaluator, ARM_JOINTS,
                            sample_triangle_mesh, native_collision_pair_eligible)

PROJECT = HERE.parents[1]
SNAPSHOT = "drawer003_closed_step60"


@pytest.fixture(scope="module")
def robot():
    return RobotGeometry(HERE / "preparation_001" / SNAPSHOT, sample_spacing_m=.01)


@pytest.fixture(scope="module")
def truth():
    return GroundTruthEvaluator(PROJECT / "scene_test/rgb_scene/references_004/snapshots" / SNAPSHOT / "evaluator")


def test_native_geometry_package_has_no_scene_state_or_objects(robot):
    root = ET.parse(robot.package / "robot.xml").getroot()
    bodies = [b.get("name", "") for b in root.iter("body")]
    assert all(not n or n.startswith(("robot0_", "gripper0_", "mount0_")) for n in bodies)
    assert robot.model.nq == 9 and robot.model.ngeom == 13
    kinds = robot.model.geom_type[robot.geom_ids]
    assert (kinds == int(mujoco.mjtGeom.mjGEOM_MESH)).sum() == 11
    assert (kinds == int(mujoco.mjtGeom.mjGEOM_BOX)).sum() == 2
    assert "focal" not in robot.metadata and "depth" not in robot.metadata
    assert [b.get("name") for b in root.find("worldbody")] == ["robot0_base"]


def test_original_robot_fk_matches_full_model_at_multiple_joint_poses(robot, truth):
    rng = np.random.default_rng(29083)
    poses = [robot.q_initial] + [rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1]) for _ in range(8)]
    for q in poses:
        robot.set_q(q)
        truth._set_q(q)
        for gid in robot.geom_ids:
            name = mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
            original_gid = mujoco.mj_name2id(truth.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            np.testing.assert_allclose(robot.data.geom_xpos[gid], truth.data.geom_xpos[original_gid], atol=2e-12, rtol=0)
            np.testing.assert_allclose(robot.data.geom_xmat[gid], truth.data.geom_xmat[original_gid], atol=2e-12, rtol=0)


def test_self_contact_native_semantics_and_eligibility(robot, truth):
    rng = np.random.default_rng(813)
    for q in [robot.q_initial] + [rng.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1]) for _ in range(18)]:
        a = robot.check(q)
        b = truth.check(q, distances=False)
        assert a["self_collision"] == b["self_collision"]
        for contact in truth.data.contact:
            assert native_collision_pair_eligible(truth.model, int(contact.geom1), int(contact.geom2))


def test_barycentric_samples_cover_original_triangles_and_edges():
    vertices = np.array([[0., 0., 0.], [.03, 0., 0.], [0., .02, 0.]])
    samples, info = sample_triangle_mesh(vertices, [[0, 1, 2]], .003)
    assert info["surface_cover_radius_m"] <= .003
    for vertex in vertices:
        assert np.min(np.linalg.norm(samples - vertex, axis=1)) < 1e-12
    rng = np.random.default_rng(2094)
    weights = rng.dirichlet(np.ones(3), 10000)
    distances = cKDTree(samples).query(weights @ vertices)[0]
    assert np.max(distances) <= info["surface_cover_radius_m"]


def test_samples_use_native_geometries_and_camera_transform(robot):
    q = robot.q_initial
    world, ids = robot.samples(q, frame="world")
    camera, camera_ids = robot.samples(q, frame="camera")
    expected = world @ robot.camera_from_world[:3, :3].T + robot.camera_from_world[:3, 3]
    np.testing.assert_allclose(camera, expected, atol=1e-12)
    np.testing.assert_array_equal(ids, camera_ids)
    assert set(ids) == set(robot.geom_ids)
    hand, hand_ids = robot.samples(q, gripper_only=True)
    assert len(set(hand_ids)) == 5 and len(hand) < len(world)


def test_fingers_and_environment_remain_frozen(robot, truth):
    q = robot.q_initial + np.array([.003, 0., 0., 0., 0., 0., 0.])
    robot.set_q(q)
    truth._set_q(q)
    np.testing.assert_array_equal(robot.data.qpos[robot.finger_addresses], robot.finger_q_initial)
    other = np.ones(truth.model.nq, bool)
    other[truth.arm_addresses] = False
    np.testing.assert_array_equal(truth.data.qpos[other], truth.frozen_qpos[other])


def test_ground_truth_distance_and_bounded_path_sampling(truth):
    q = truth.q_initial
    check = truth.check(q)
    assert check["valid"]
    assert 0 < check["environment_min_distance_m"] <= .05
    path = np.array([q, q + [.011, 0, 0, 0, 0, 0, 0]])
    report = truth.validate_path(path, max_joint_step_rad=.005, keep_samples=True)
    assert report["sample_count"] == 4
    assert report["maximum_observed_joint_step_rad"] <= .005 + 1e-14
    assert not report["continuous_collision_certificate"]
    assert len(report["samples"]) == 4


def test_out_of_limits_and_invalid_input_are_not_clipped(robot):
    q = robot.q_initial.copy()
    q[0] = robot.joint_limits[0, 1] + .1
    assert not robot.check(q)["valid"]
    with pytest.raises(ValueError):
        robot.set_q([1, 2, 3])


def test_package_mesh_tamper_rejected(tmp_path, robot):
    import shutil
    copy = tmp_path / "robot"
    shutil.copytree(robot.package, copy)
    asset = copy / robot.metadata["assets"][0]["file"]
    asset.write_bytes(asset.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        RobotGeometry(copy)
