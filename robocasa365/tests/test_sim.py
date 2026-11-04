"""Pure fake checks for mobile/world-frame routing; no simulator/model calls."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import numpy as np
import pytest

from rc365.sim import (
    RoboCasa365Config, RoboCasa365Worker, RoboCasa365SimProcess,
    MobileObservation, MOBILE_SCHEMA, MOBILE_COMMAND_FRAME,
    validate_mobile_command, validate_mobile_state, expand_mobile_action,
    switch_arm_reference, validate_native_controllers,
)

INDEXES = {"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)}


def mobile_state():
    return dict(schema_version=MOBILE_SCHEMA, command_frame=MOBILE_COMMAND_FRAME,
                base_joint_position=[0, 0, 0], base_joint_velocity=[0, 0, 0],
                torso_joint_position=[.12], torso_joint_velocity=[0],
                base_position_world_m=[.2, -.4, 0], base_rotation_world=np.eye(3).tolist(),
                control_mode="arm", step=0)


def test_configuration_defaults_isolated_target_split_and_native_images():
    cfg = RoboCasa365Config()
    assert cfg.task == "RestockPantry" and cfg.split == "target" and cfg.seed == 7
    assert "/robocasa365/" in cfg.python_executable and "/robocasa365/" in cfg.robocasa_root
    assert cfg.mobile_enabled and cfg.stop_on_success and cfg.image_size == 512
    with pytest.raises(ValueError): replace(cfg, seed=True)
    with pytest.raises(ValueError): replace(cfg, split="invented")
    with pytest.raises(ValueError): replace(cfg, image_size=256)


@pytest.mark.parametrize("command", [
    {"base": [0, 0], "torso": 0}, {"base": [0, 0, True], "torso": 0},
    {"base": [0, float("nan"), 0], "torso": 0}, {"base": [.201, 0, 0], "torso": 0},
    {"base": [0, 0, 0], "torso": -.201}, {"base": [0, 0, 0], "torso": True},
    {"base": [0, 0, 0], "torso": 0, "target_obj": "secret"},
])
def test_mobile_command_rejects_malformed_or_out_of_range_before_action(command):
    with pytest.raises(ValueError): validate_mobile_command(command)


def test_native_mobile_mapping_uses_declared_indexes_and_keeps_gripper():
    command = {"base": [.2, -.1, .05], "torso": -.2}
    arm = [0] * 6 + [1]
    full = expand_mobile_action(arm, command, INDEXES, 12, 1)
    np.testing.assert_allclose(full, [0] * 6 + [1, .2, -.1, .05, -.2, 1])
    reordered = {"base": (0, 3), "torso": (3, 4), "right": (4, 10), "right_gripper": (10, 11)}
    np.testing.assert_allclose(expand_mobile_action(arm, command, reordered, 12, 1),
                               [.2, -.1, .05, -.2] + [0] * 6 + [1, 1])
    with pytest.raises(ValueError): expand_mobile_action([.1] + [0] * 5 + [1], command, INDEXES, 12, 1)
    with pytest.raises(ValueError): expand_mobile_action(arm, command, INDEXES, 12, -1)
    with pytest.raises(ValueError): expand_mobile_action(arm, command, {**INDEXES, "base": (8, 11)}, 12, 1)


def test_mobile_state_only_accepts_measured_robot_channels():
    original = mobile_state()
    clean = validate_mobile_state(original)
    original["base_joint_position"][0] = 99
    assert clean["base_joint_position"][0] == 0
    for change in ({"object_pose": [1, 2, 3]}, {"command_frame": "world"},
                   {"base_rotation_world": [[1, 0, 0], [0, 1, 0], [0, 0, -1]]},
                   {"torso_joint_velocity": [float("inf")]}, {"step": True}):
        with pytest.raises(ValueError): validate_mobile_state({**mobile_state(), **change})


class FakeArm:
    def __init__(self):
        self.input_ref_frame = "world"
        self.origin_pos = np.array([1., 2., .3])
        self.origin_ori = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.ref_pos = np.array([1.2, 2.4, 1.])
        self.ref_ori_mat = np.eye(3)
        self.goal_pos = np.array([99, 99, 99])
        self.goal_ori = -np.eye(3)
        self.calls = []

    def update(self, force=False): self.calls.append(("update", force))
    def update_origin(self, pos, ori):
        self.origin_pos, self.origin_ori = np.array(pos), np.array(ori)
    def set_goal_update_mode(self, mode): self.goal_update_mode = mode
    def set_goal(self, action):
        assert np.array_equal(action, np.zeros(6))
        assert self.goal_pos is None and self.goal_ori is None
        if self.input_ref_frame == "base":
            self.goal_pos = self.origin_ori.T @ (self.ref_pos - self.origin_pos)
            self.goal_ori = self.origin_ori.T @ self.ref_ori_mat
        else:
            self.goal_pos, self.goal_ori = self.ref_pos.copy(), self.ref_ori_mat.copy()
        self.calls.append(("goal", self.input_ref_frame))


def test_frame_switch_rebases_goal_from_measured_pose_including_rotated_base():
    arm = FakeArm()
    switch_arm_reference(arm, "base")
    assert arm.goal_update_mode == "desired"
    np.testing.assert_allclose(arm.goal_pos, [.4, -.2, .7])
    np.testing.assert_allclose(arm.origin_ori @ arm.goal_ori, arm.ref_ori_mat)
    arm.ref_pos += [.03, -.02, .01]
    switch_arm_reference(arm, "world")
    assert arm.goal_update_mode == "achieved"
    np.testing.assert_allclose(arm.goal_pos, arm.ref_pos)
    np.testing.assert_allclose(arm.goal_ori, np.eye(3))


def fake_worker():
    worker = RoboCasa365Worker(RoboCasa365Config())
    arm = FakeArm()
    torso = SimpleNamespace(input_type="absolute", qpos_index=[3],
                            set_goal=lambda value: None)
    comp = SimpleNamespace(get_controller_base_pose=lambda name: (arm.origin_pos, arm.origin_ori))
    robot = SimpleNamespace(composite_controller=comp)
    worker.env = SimpleNamespace(robots=[robot], action_dim=12,
                                 sim=SimpleNamespace(data=SimpleNamespace(qpos=np.array([0, 0, 0, .12]))))
    worker.arm_controller, worker.torso_controller = arm, torso
    worker.indexes, worker._mode, worker._last_gripper = INDEXES, "arm", -1
    worker.torso_position = np.array([.12])
    worker.status = {"steps": 0, "terminated": False}
    worker._advance = lambda full: np.array(full)
    return worker


def test_arm_mobile_arm_transition_holds_latest_torso_and_retains_mobile_goal():
    worker = fake_worker()
    command = {"base": [.1, 0, 0], "torso": .2}
    full = worker.mobile_step([0] * 6 + [-1], command)
    assert worker._mode == "mobile" and worker.arm_controller.input_ref_frame == "base"
    assert worker.torso_controller.input_type == "delta" and full[-1] == 1
    old_goal = worker.arm_controller.goal_pos.copy()
    worker.arm_controller.ref_pos += [.01, .02, .03]
    worker.env.sim.data.qpos[3] = .17
    worker.mobile_step([0] * 6 + [-1], command)
    np.testing.assert_allclose(worker.arm_controller.goal_pos, old_goal)
    full = worker.step([.1, 0, 0, 0, 0, 0, -1])
    assert worker._mode == "arm" and worker.arm_controller.input_ref_frame == "world"
    assert worker.torso_controller.input_type == "absolute" and full[-1] == -1
    assert full[10] == .17
    np.testing.assert_equal(full[7:10], [0, 0, 0])
    worker.env.sim.data.qpos[3] = .169
    full = worker.step([0] * 6 + [-1])
    assert full[10] == .17


def test_bad_mobile_request_changes_no_mode_or_controller_goal():
    worker = fake_worker()
    with pytest.raises(ValueError): worker.mobile_step([0] * 6 + [1], {"base": [.1, 0, 0], "torso": 0})
    assert worker._mode == "arm" and worker.arm_controller.calls == []
    worker.config = replace(worker.config, mobile_enabled=False)
    with pytest.raises(ValueError): worker.mobile_step([0] * 6 + [-1], {"base": [.1, 0, 0], "torso": 0})
    assert worker.arm_controller.calls == []


def test_action_archive_precedes_native_in_place_base_transform():
    worker = fake_worker()
    del worker._advance
    worker._response = lambda: dict(worker.status)
    worker.env.step = lambda action: action.__setitem__(7, 99)
    sent = np.zeros(12)
    sent[7] = .1
    result = worker._advance(sent)
    assert sent[7] == .1 and result["last_native_action"][7] == .1
    assert result["steps"] == 1


def test_observation_transport_excludes_evaluator_and_snapshot_truth():
    from gpt_field.observation import image_to_data_url
    from test_robocasa_user_check_geometry import STATE, metadata
    rgb = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), 512, rotate=False)
    obs = MobileObservation(rgb, rgb, tuple(STATE), "official language", joint_position=(0.,)*7,
                            joint_velocity=(0.,)*7, geometry=metadata(), mobile_state=mobile_state())
    assert set(obs.to_dict()) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    # The process reads only explicit allowlisted observation, geometry and robot sensors.
    process = RoboCasa365SimProcess()
    payload = dict(observation={**obs.to_dict(), "observation/joint_position": [0]*7,
                                "observation/joint_velocity": [0]*7}, geometry=obs.geometry,
                   mobile_state=obs.mobile_state, status={"robocasa_success": False, "hidden_objects": "secret"},
                   snapshot={"hidden_target": [1,2,3]})
    result = process._consume_observation(payload)
    assert result.mobile_state == obs.mobile_state
    assert not hasattr(result, "status") and not hasattr(result, "snapshot")
    assert "hidden" not in repr(result.mobile_state)


def native_robot_fake():
    arm = type("OperationalSpaceController", (), {})()
    arm.input_ref_frame, arm.input_type, arm.control_dim = "base", "delta", 6
    arm.input_min, arm.input_max = np.full(6, -1), np.ones(6)
    arm.output_max = np.array([.05] * 3 + [.5] * 3)
    arm.output_min = -arm.output_max
    arm.interpolator_pos = arm.interpolator_ori = None
    parts = {"right": arm}
    for name, kind, count, scale in (("base", "LegacyMobileBaseJointVelocityController", 3, 1),
                                     ("torso", "JointPositionController", 1, .05)):
        controller = type(kind, (), {})()
        controller.control_dim, controller.impedance_mode = count, "fixed"
        controller.input_min, controller.input_max = np.full(count, -1), np.ones(count)
        controller.output_min, controller.output_max = np.full(count, -scale), np.full(count, scale)
        controller.input_type = "delta"
        if name == "base":
            controller.actuator_min = np.array([-1., -1., -1.5])
            controller.actuator_max = np.array([1., 1., 1.5])
        parts[name] = controller
    return SimpleNamespace(part_controllers=parts, composite_controller=type("HybridMobileBase", (), {})())


def test_native_contract_checks_fixed_base_and_torso_scales_without_persistent_frame_mutation():
    robot = native_robot_fake()
    record = validate_native_controllers(robot, initial=True)
    assert record["arm"]["input_ref_frame"] == "base"
    assert record["base"]["output_max"] == [1, 1, 1]
    assert record["base"]["normalized_to_actuator_velocity_scale"] == [1., 1., 1.5]
    assert record["base"]["actuator_max"] == [1., 1., 1.5]
    assert record["torso"]["output_max"] == [.05]
    assert robot.part_controllers["right"].input_ref_frame == "base"
    robot.part_controllers["right"].output_max[0] = .1
    with pytest.raises(ValueError): validate_native_controllers(robot, initial=True)
    assert robot.part_controllers["right"].input_ref_frame == "base"


@pytest.mark.parametrize("part,field,value", [
    ("base", "output_max", [1, 1, 1.5]), ("base", "control_dim", 4),
    ("base", "actuator_max", [1, 1, 1]), ("base", "actuator_min", [-1, -1, -1]),
    ("torso", "output_max", [.1]), ("torso", "input_type", "absolute"),
    ("right", "input_ref_frame", "world"), ("right", "interpolator_pos", object()),
])
def test_native_contract_changes_refuse_before_policy_actions(part, field, value):
    robot = native_robot_fake()
    setattr(robot.part_controllers[part], field, value)
    with pytest.raises(ValueError): validate_native_controllers(robot, initial=True)


def test_native_base_actuator_scale_matches_the_installed_xml():
    from pathlib import Path
    import xml.etree.ElementTree as ET
    path = Path(__file__).resolve().parents[1] / "vendor/robosuite/robosuite/models/assets/bases/omron_mobile_base.xml"
    tree = ET.parse(path)
    wanted = ["joint_mobile_forward", "joint_mobile_side", "joint_mobile_yaw"]
    actual = {node.attrib["joint"]: [float(x) for x in node.attrib["ctrlrange"].split()]
              for node in tree.findall("actuator/velocity")}
    ranges = np.array([actual[key] for key in wanted])
    np.testing.assert_array_equal(ranges[:, 0], [-1, -1, -1.5])
    np.testing.assert_array_equal(ranges[:, 1], [1, 1, 1.5])
    native = native_robot_fake().part_controllers["base"]
    command = np.array([.2, -.2, .2])
    actual_velocity = .5 * (native.actuator_max + native.actuator_min) + .5 * (native.actuator_max - native.actuator_min) * command
    np.testing.assert_allclose(actual_velocity, [.2, -.2, .3])


def test_kitchen_native_override_matches_the_exact_checked_base_class():
    from pathlib import Path
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "vendor/robocasa/robocasa/environments/kitchen/kitchen.py").read_text())
    overrides = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Constant) and node.value.value == "JOINT_VELOCITY_LEGACY"]
    assert len(overrides) == 1
    assert ast.unparse(overrides[0].targets[0]) == "controller_configs['body_parts']['base']['type']"
    from robosuite.controllers.parts.mobile_base.joint_vel import LegacyMobileBaseJointVelocityController
    assert LegacyMobileBaseJointVelocityController.__name__ == native_robot_fake().part_controllers["base"].__class__.__name__
    # Exercise the pinned native set_goal on a sensor-only object, without calling
    # its constructor or creating any MuJoCo model/simulation.
    native = object.__new__(LegacyMobileBaseJointVelocityController)
    native.impedance_mode, native.qpos_index, native.interpolator = "fixed", [0, 1, 2], None
    native.update = lambda: None
    native.scale_action = lambda action: action.copy()
    native.init_ori = np.eye(3)
    native.get_base_pose = lambda: (np.zeros(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    native.joint_names = ["mobilebase0_joint_mobile_forward"]
    native.sim = SimpleNamespace(model=SimpleNamespace(joint_name2id=lambda name: 0, jnt_axis=np.eye(3)))
    native.set_goal(np.array([.2, 0, .2]))
    np.testing.assert_allclose(native.goal_qvel, [0, .2, .2], atol=1e-7)


def test_default_robocasa_wrist_override_matches_existing_geometry_contract():
    from pathlib import Path
    import ast
    from gpt_field.robocasa_user_check_geometry import WRIST_POSITION_HAND_M, HAND_BODY
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "vendor/robocasa/robocasa/utils/camera_utils.py").read_text())
    config = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CAM_CONFIGS" for target in node.targets))
    default = next(k.value for k in config.keywords if k.arg == "DEFAULT")
    wrist = next(k.value for k in default.keywords if k.arg == "robot0_eye_in_hand")
    fields = {k.arg: ast.literal_eval(k.value) for k in wrist.keywords}
    np.testing.assert_array_equal(fields["pos"], WRIST_POSITION_HAND_M)
    assert fields["parent_body"] == HAND_BODY
    q = np.array(fields["quat"])
    np.testing.assert_allclose(q / np.linalg.norm(q), [0, 2**-.5, 2**-.5, 0], atol=1e-9)
