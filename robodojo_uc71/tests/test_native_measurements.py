"""Offline geometry/reference and paused-state tests; no simulator imports."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
from robodojo_uc71.native_measurements import (NativeMeasurementError, NativeMeasurements,
    capture_camera_inventory, capture_paused_physics_state, native_link6_tool_candidate)
from prepare_native_preview import run_pause_validation


def candidate(tmp_path):
    for name in ("Assets/Robots/x5/robot_config.yml", "env/robot_manager/robot_manager.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic provenance fixture, not native calibration\n")
    return native_link6_tool_candidate(tmp_path)


def robot_env():
    # Deliberately permute live joints and robot-list order. The body is not the
    # first fixed-base Jacobian row, and its frame is rotated90deg about worldZ.
    names = ["joint7", "joint3", "joint8", "joint1", "joint5", "joint2", "joint6", "joint4"]
    arm_ids = [3, 5, 1, 7, 4, 6]
    body_names = ["base_link", "link1", "link6", "camera"]
    actor_j = np.arange(36, dtype=float).reshape(6, 6) / 17
    actor_j[3:] += np.eye(3, 6)
    com = np.array([.2, .1, -.03])
    world_com = np.array([-.1, .2, -.03])
    x, y, z = world_com
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    raw_j = actor_j.copy()
    raw_j[:3] -= skew @ actor_j[3:]
    robots, articulations = [], []
    for arm in ("right", "left"):
        q = np.array([[.02, .3, .04, .1, .5, .2, .6, .4]])
        poses = np.zeros((1, 4, 7)); poses[:, :, 3] = 1
        poses[0, 2] = [1, 2, 3, math.sqrt(.5), 0, 0, math.sqrt(.5)]
        coms = np.zeros((1, 4, 3)); coms[0, 2] = com
        jac = np.zeros((1, 3, 6, 8)); jac[0, 1][:, arm_ids] = raw_j
        data = NS(joint_pos=q, joint_vel=q*.01, joint_pos_limits=np.tile([-10., 10.], (1, 8, 1)),
            joint_vel_limits=np.ones((1, 8))*5, body_link_pose_w=poses, body_com_pos_b=coms)
        articulation = NS(data=data, joint_names=names, body_names=body_names, num_joints=8,
            num_bodies=4, is_fixed_base=True,
            root_physx_view=NS(get_jacobians=lambda jac=jac: jac,
                              get_dof_positions=lambda q=q: q.copy()))
        robot = NS(arm_name=arm+"_arm", ee_link_name="link6", base_link="base_link",
            arm_joint_indices=arm_ids, gripper_joint_indices=[0, 2],
            arm_joints_name=[f"joint{i}" for i in range(1, 7)], gripper_joints_name=["joint7", "joint8"])
        robots.append(robot); articulations.append(articulation)
    env = NS(num_envs=1, sim=NS(sim=NS(current_time=2.)), robot_manager=NS(
        num_envs=1, robot_list=robots, robot_key=articulations))
    return env, actor_j


def test_com_to_actor_correction_uses_live_rotation_rows_columns_and_fingers(tmp_path):
    env, expected = robot_env()
    sample = NativeMeasurements(env, tool_candidate=candidate(tmp_path)).sample()
    for arm in ("left", "right"):
        assert np.allclose(sample.arms[arm].link6_world_jacobian, expected, atol=1e-12)
        assert sample.arms[arm].joint_positions == pytest.approx([.1, .2, .3, .4, .5, .6])
        assert sample.arms[arm].link6_position == pytest.approx([1, 2, 3])
        assert sample.finger_position_m[arm] == pytest.approx((.02, .04))
        assert sample.provenance["robots"][arm]["jacobian_body_row"] == 1
        assert not np.allclose(sample.provenance["robots"][arm]["raw_com_world_jacobian"], expected)
    assert sample.provenance["standard_ready"] is False
    assert sample.provenance["model_input_eligible"] is False


@pytest.mark.parametrize("change", ["floating_base", "wrong_indices", "nan_com", "wrong_shape", "missing_arm"])
def test_measurements_fail_closed_on_unsupported_native_state(tmp_path, change):
    env, _ = robot_env(); robot = env.robot_manager.robot_list[0]; art = env.robot_manager.robot_key[0]
    if change == "floating_base": art.is_fixed_base = False
    if change == "wrong_indices": robot.arm_joint_indices = [0, 1, 2, 3, 4, 5]
    if change == "nan_com": art.data.body_com_pos_b[0, 2, 0] = np.nan
    if change == "wrong_shape": art.root_physx_view.get_jacobians = lambda: np.zeros((1, 24, 8))
    if change == "missing_arm": robot.arm_name = "left_arm"
    with pytest.raises(NativeMeasurementError):
        NativeMeasurements(env, tool_candidate=candidate(tmp_path)).sample()


@pytest.mark.parametrize("change", ["offset", "rotation", "ready_claim", "source_changed"])
def test_native_reference_candidate_cannot_claim_physical_tcp_or_ready(tmp_path, change):
    env, _ = robot_env(); spec = candidate(tmp_path)
    if change == "offset": spec["tools"]["left"]["link6_from_tool"][0][3] = .145
    if change == "rotation": spec["tools"]["left"]["link6_from_tool"] = np.diag([1, -1, -1, 1]).tolist()
    if change == "ready_claim": spec["standard_ready"] = True
    if change == "source_changed": Path(spec["source_artifacts"][0]["path"]).write_text("changed")
    with pytest.raises(NativeMeasurementError, match="tool_candidate"):
        NativeMeasurements(env, tool_candidate=spec)


def test_candidate_always_rejects_full_test_readiness(tmp_path):
    env, _ = robot_env(); adapter = NativeMeasurements(env, tool_candidate=candidate(tmp_path))
    with pytest.raises(NativeMeasurementError, match="calibration_not_validated"):
        adapter.require_ready()


class Camera:
    def __init__(self, name):
        self.prim_path = "/World/"+name
        self.k = np.array([[288., 0, 320], [0, 336., 240], [0, 0, 1.]])
        self.view = np.array([[0, -1., 0, 1.], [1., 0, 0, 2.], [0, 0, 1., 3.], [0, 0, 0, 1.]])
    def get_resolution(self): return (640, 480)
    def get_lens_distortion_model(self): return "pinhole"
    def get_intrinsics_matrix(self, *, device): assert device == "cpu"; return self.k
    def get_view_matrix_ros(self, *, device): assert device == "cpu"; return self.view


def camera_env():
    env, _ = robot_env()
    names = ["cam_right_wrist", "cam_head", "cam_left_wrist"]
    mounts = {"cam_head": None, "cam_right_wrist": "robot1/camera", "cam_left_wrist": "robot0/camera"}
    env.camera_manager = NS(num_envs=1, camera_names=[names], cameras=[[Camera(n) for n in names]],
        camera_config={n: {"camera": {"mount_link": mounts[n]}} for n in names})
    return env


def test_camera_inventory_preserves_anisotropic_k_and_inverts_actual_optical_view():
    env = camera_env(); result = capture_camera_inventory(env)
    head = result["cameras"]["head"]
    assert head["native_camera_index"] == 1
    assert head["intrinsics"][0][0] != head["intrinsics"][1][1]
    assert np.allclose(head["world_from_optical"], np.linalg.inv(env.camera_manager.cameras[0][1].view))
    assert head["fixed_world_frame_candidate"] is True
    assert result["cameras"]["left_wrist"]["fixed_world_frame_candidate"] is False
    assert result["rendered_landmark_validation_passed"] is False


@pytest.mark.parametrize("change", ["mount", "names", "nonrigid", "pixels", "error_text"])
def test_camera_inventory_rejects_unreviewed_mapping_or_bad_getters_without_leaking_text(change):
    env = camera_env(); manager = env.camera_manager
    if change == "mount": manager.camera_config["cam_head"]["camera"]["mount_link"] = "moving/object"
    if change == "names": manager.camera_names[0][0] = "cam_head"
    if change == "nonrigid": manager.cameras[0][0].view[0, 1] = -2
    if change == "pixels": manager.cameras[0][0].get_resolution = lambda: (800, 600)
    if change == "error_text":
        def broken(**kwargs): raise RuntimeError("SECRET_OBJECT_CONTACT")
        manager.cameras[0][0].get_intrinsics_matrix = broken
    with pytest.raises(NativeMeasurementError) as error:
        capture_camera_inventory(env)
    assert "SECRET_OBJECT_CONTACT" not in str(error.value)


def test_paused_capture_reads_raw_dofs_not_cached_joint_positions():
    env, _ = robot_env()
    env.robot_manager.robot_key[0].data.joint_pos[:] = 999
    env.robot_manager.robot_key[0].root_physx_view.get_dof_positions = lambda: np.zeros((1, 8))
    state = capture_paused_physics_state(env, 17)
    assert state["articulation_dof_positions"]["right"] == [[0.]*8]
    assert state["native_sim_step_calls"] == 17


@pytest.mark.parametrize("drift", [False, True])
def test_pause_wait_uses_only_sleep_and_log_and_detects_raw_state_drift(tmp_path, monkeypatch, drift):
    env, _ = robot_env(); now = [100.]; sleeps = []; progress = []
    def sleep(seconds):
        sleeps.append(seconds); now[0] += seconds
        if drift: env.robot_manager.robot_key[0].data.joint_pos[0, 0] += .001
    monkeypatch.setattr("prepare_native_preview.time.monotonic", lambda: now[0])
    monkeypatch.setattr("prepare_native_preview.time.sleep", sleep)
    monitor = NS(is_fatal=lambda: False, get_broken_envs=lambda: [])
    def record(phase, **kw): progress.append(phase)
    if drift:
        with pytest.raises(RuntimeError, match="paused-physics validation failed"):
            run_pause_validation(env, monitor, {"native_sim_step_calls": 7}, tmp_path, record, [])
    else:
        result = run_pause_validation(env, monitor, {"native_sim_step_calls": 7}, tmp_path, record, [])
        assert result["passed"] is True
    saved = json.loads((tmp_path/"pause_validation.json").read_text())
    assert saved["dof_positions_unchanged"] is not drift
    assert saved["elapsed_monotonic_s"] == 300
    assert saved["simulation_updates_during_wait"] == 0
    assert sleeps == [30.]*10
    assert progress.count("paused_physics_wait_heartbeat") == 10
