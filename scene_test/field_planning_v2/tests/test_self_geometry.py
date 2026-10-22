from pathlib import Path
import copy
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from self_geometry import InitialRobotSelfGeometry
from robot_geometry import RobotGeometry


@pytest.fixture(scope="module")
def self_space():
    snapshot = "drawer003_closed_step60"
    robot = RobotGeometry(HERE.parent / "field_planning/preparation_001" / snapshot)
    return InitialRobotSelfGeometry(robot, rig_calibration_path=HERE / "preparation_001" / snapshot / "rig_calibration.json")


def test_all_original_material_samples_are_self_and_outside_is_not_free(self_space):
    points, gids = self_space.robot.samples(self_space.q_initial)
    inside, owners = self_space.contains_with_owner(points)
    assert len(points) == 172411 and len(set(gids)) == 13
    assert inside.all() and np.all(owners >= 0)
    assert not self_space.contains(np.array([[100, 100, 100], [np.nan, 0, 1], [0, 0, np.inf]])).any()
    assert self_space.provenance()["environment_geometry_inputs"] == []
    assert self_space.provenance()["intentional_dilation_m"] == 0.


def test_novel_motion_does_not_enlarge_known_initial_material(self_space):
    before = self_space.q_initial.copy()
    q = before.copy()
    q[0] += .1
    moved, _ = self_space.robot.samples(q)
    inside = self_space.contains(moved)
    assert 0 < inside.sum() < len(moved)
    np.testing.assert_array_equal(self_space.q_initial, before)
    original, _ = self_space.robot.samples(before)
    assert self_space.contains(original).all()


def test_native_finger_pad_ray_interval_matches_exact_box(self_space):
    space = copy.copy(self_space)
    hull = next(h for h in space.hulls if h["name"] == "gripper0_finger1_pad_collision")
    space.hulls = [hull]
    R, t = hull["rotation_camera_from_local"], hull["translation_camera_m"]
    local_origin = np.array([0., 0., -.1])
    camera_origin = local_origin @ R.T + t
    direction = np.array([[0., 0., 1.]]) @ R.T
    result = space.ray_intervals(direction, camera_origin)
    np.testing.assert_allclose(result["front_t"], [.092], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result["exit_t"], [.108], atol=2e-12, rtol=0)
    assert result["geom_id"][0] == hull["geom_id"]
    miss_origin = np.array([.1, 0., -.1]) @ R.T + t
    miss = space.ray_intervals(direction, miss_origin)
    assert np.isinf(miss["front_t"][0]) and miss["geom_id"][0] == -1


def test_ray_parameter_is_not_assumed_unit_length(self_space):
    direction = np.array([[0., 0., 1.]])
    first = self_space.ray_intervals(direction)
    twice = self_space.ray_intervals(2 * direction)
    finite = np.isfinite(first["front_t"])
    np.testing.assert_array_equal(finite, np.isfinite(twice["front_t"]))
    np.testing.assert_allclose(first["front_t"][finite], 2 * twice["front_t"][finite], atol=1e-12)


def test_start_inside_native_material_has_zero_entry(self_space):
    hull = self_space.hulls[-1]
    center = hull["vertices_local"].mean(axis=0) @ hull["rotation_camera_from_local"].T + hull["translation_camera_m"]
    result = self_space.ray_intervals([[0, 0, 1]], center)
    assert result["front_t"][0] == 0 and result["exit_t"][0] > 0


def test_no_macroscopic_boundary_dilation(self_space):
    space = copy.copy(self_space)
    hull = next(h for h in space.hulls if h["name"] == "gripper0_finger1_pad_collision")
    space.hulls = [hull]
    local = np.array([[.008, 0, 0], [.008 + 1e-7, 0, 0], [.008 - 1e-7, 0, 0]])
    points = local @ hull["rotation_camera_from_local"].T + hull["translation_camera_m"]
    np.testing.assert_array_equal(space.contains(points), [True, False, True])
    with pytest.raises(ValueError, match="numerical"):
        InitialRobotSelfGeometry(self_space.robot, tolerance_m=.001)


def test_pixel_rays_and_generic_direction_queries_agree_and_keep_K_separate(self_space):
    K = self_space.rig_K.copy()
    K[:2] *= .125
    image = self_space.ray_image(K, (64, 64))
    y, x = np.mgrid[:64, :64]
    pixel = np.stack([x.ravel(), y.ravel(), np.ones(64 * 64)], axis=1)
    directions = pixel @ np.linalg.inv(K).T
    rays = self_space.ray_intervals(directions)
    np.testing.assert_array_equal(image["front_depth_m"].ravel(), rays["front_t"])
    np.testing.assert_array_equal(image["geom_id"].ravel(), rays["geom_id"])
    assert np.isfinite(image["front_depth_m"]).any()
    K_other = K.copy()
    K_other[0, 0] *= .7
    other = self_space.ray_image(K_other, (64, 64))
    assert not np.array_equal(image["geom_id"], other["geom_id"])
    assert not image["front_depth_m"].flags.writeable


def test_rig_calibration_source_binding_and_mismatch_rejection(self_space, tmp_path):
    import json
    provenance = self_space.provenance()
    rig = provenance["rig_calibration"]
    assert rig["sha256"] == provenance["consumed_sha256"][rig["path"]]
    assert rig["source_calibration_sha256"]
    metadata = json.loads(Path(rig["path"]).read_text())
    metadata["T_camera_cv_from_world"][0][3] += .001
    wrong = tmp_path / "wrong_rig.json"
    wrong.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="frames disagree"):
        InitialRobotSelfGeometry(self_space.robot, rig_calibration_path=wrong)
