"""Synthetic diagnostic dispatch only; no EpisodeLifecycle or simulator launch."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from robodojo_uc71 import native_characterization as module
from robodojo_uc71.native_action_observer import NativeActionObservationError, observe_native_joint_action_once


class Reward:
    def __init__(self):
        self.check_list, self.calls, self.steps = [["synthetic"]], 0, 0
    def step(self):
        self.steps += 1
    def get_reward(self):
        self.calls += 1
        return [.42]


class Env:
    def __init__(self):
        self.num_envs, self.task_name, self.config_name, self.env_seeds = 1, "make_toast", "arx_x5", [0]
        self.dt, self.interact = .004, False
        self.robot_manager = NS(planner={}, ik_solver={})
        self.obs_manager = NS(collect_interval=10)
        self.sim = NS(cfg=NS(decimation=1, sim=NS(dt=.004)), physics_dt=.004,
                      sim=NS(current_time=0.), _sim_step_counter=0)
        self.take_action_cnt, self.end_flag = [0], [False]
        self.reward_manager = Reward()
        self.positions = {arm: [0.] * 6 for arm in module.ARMS}
        self.action_log = []
        self.step_count, self.physics_delta, self.reward_copies = 10, .004, 1
        self.fail_step = None
        self.force_end = False
    def have_empty(self, indices):
        assert indices == [0]
        return True
    def sim_step(self):
        self.sim._sim_step_counter += 1
        self.sim.sim.current_time += self.physics_delta
        if self.fail_step == self.sim._sim_step_counter:
            raise RuntimeError("synthetic failure")
    def is_episode_end(self):
        for _ in range(self.reward_copies): self.reward_manager.get_reward()
        self.end_flag = [self.force_end]
        return self.force_end
    def take_action(self, action):
        self.action_log.append(deepcopy(action))
        self.take_action_cnt[0] += 1
        for _ in range(self.step_count): self.sim_step()
        self.reward_manager.step()
        self.is_episode_end()
        for arm in module.ARMS: self.positions[arm] = list(action[f"{arm}_arm_joint_state"])


def setup(tmp_path, monkeypatch, *, authorized=True):
    source = tmp_path / "synthetic_robot.urdf"; source.write_text("synthetic-only")
    layout = tmp_path / "make_toast_0.json"; layout.write_text('{"synthetic_only":true}')
    robots = []
    for arm in module.ARMS:
        robots.append({"arm_name": arm + "_arm", "joint_names": [f"joint{i}" for i in range(1, 9)],
            "native_arm_joint_indices": list(range(6)), "native_gripper_joint_indices": [6, 7],
            "joint_positions": {"values": [[0.] * 6 + [.044, .044]]},
            "joint_position_limits_lower_upper": {"values": [[[-2., 2.]] * 6 + [[0., .044], [0., .044]]]},
            "declared_metadata": {"gripper_scale": {"values": [-.01, .044]},
                "gripper_move": {"base": "joint7", "sign": 1., "mimic": ["joint8", 1., 0.]}}})
    values = [
        {"schema": "robodojo-native-robot-inventory-1", "num_envs": 1, "robots": robots},
        {"schema": "robodojo-physical-tool-candidate-1", "calibration_status": "uncalibrated_candidate",
            "model_input_eligible": False, "full_test_authorized": False,
            "source_files": [{"path": str(source), "sha256": module.file_hash(source)}]},
        {"full_test_authorized": False, "binding": {"kit_index": 0, "gpu_uuid": "synthetic-gpu"},
            "layout_path": str(layout), "layout_sha256": module.file_hash(layout)},
    ]
    paths = [tmp_path / name for name in ("inventory.json", "geometry.json", "preparation.json")]
    for path, value in zip(paths, values): path.write_text(json.dumps(value))
    plan = module.build_probe_plan(*paths)
    auth = module.authorization_template(plan)
    auth.update(diagnostic_native_actions_authorized=authorized, user_message="synthetic diagnostic only",
                authorization_basis="test fixture; never full-test approval")
    auth_path = tmp_path / "diagnostic_scope.json"; auth_path.write_text(json.dumps(auth))
    def measure(env):
        return {"arms": {arm: {"joint_positions": list(env.positions[arm]),
            "joint_velocities": [0.] * 6, "finger_positions": [.044, .044]} for arm in module.ARMS}}
    monkeypatch.setattr(module, "capture_characterization_state", measure)
    env = Env()
    helper = module.FixedRobotCharacterization(env, plan, authorization_path=auth_path, verify_runtime_binding=lambda p: True)
    return env, plan, helper, auth_path


def test_exact_37_fixed_actions_without_benchmark_lifecycle(tmp_path, monkeypatch):
    env, plan, helper, _ = setup(tmp_path, monkeypatch)
    for index in range(1, 38):
        receipt = helper.dispatch_next()
        assert receipt["diagnostic_native_action_index"] == index
        assert receipt["benchmark_attempt_started"] is False
        assert receipt["settled_tracking_assessed"] is False
        assert receipt["native_accounting"]["native_sim_step_calls"] == 10
        assert "native_reward" not in receipt["native_accounting"]
    assert env.sim._sim_step_counter == 370 and env.sim.sim.current_time == pytest.approx(1.48)
    assert env.reward_manager.calls == env.reward_manager.steps == 37
    assert env.action_log == [row["action"] for row in plan["actions"]]
    assert all(row["left_ee_joint_state"] == row["right_ee_joint_state"] == [1.] for row in env.action_log)
    with pytest.raises(module.CharacterizationError, match="no_retry"): helper.dispatch_next()


def test_unexecuted_authorization_template_cannot_dispatch(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch, authorized=False)
    with pytest.raises(module.CharacterizationError, match="authorization"): helper.dispatch_next()
    assert env.take_action_cnt == [0]
    with pytest.raises(module.CharacterizationError, match="no_retry"): helper.dispatch_next()


def test_authorization_is_reread_for_every_action(tmp_path, monkeypatch):
    env, _, helper, path = setup(tmp_path, monkeypatch)
    helper.dispatch_next()
    auth = json.loads(path.read_text()); auth["diagnostic_native_actions_authorized"] = False
    path.write_text(json.dumps(auth))
    with pytest.raises(module.CharacterizationError): helper.dispatch_next()
    assert env.take_action_cnt == [1]


def test_targets_cannot_be_changed_from_recorded_plan(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    helper.plan["actions"][1]["action"]["left_arm_joint_state"][0] = .006
    with pytest.raises(module.CharacterizationError, match="mutated"): helper.dispatch_next()
    assert env.take_action_cnt == [0]


def test_native_error_restores_hooks_and_permanently_stops(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    original = env.sim_step, env.is_episode_end, env.reward_manager.step, env.reward_manager.get_reward
    env.fail_step = 3
    with pytest.raises(RuntimeError): helper.dispatch_next()
    assert original == (env.sim_step, env.is_episode_end, env.reward_manager.step, env.reward_manager.get_reward)
    assert "sim_step" not in vars(env) and "get_reward" not in vars(env.reward_manager)
    with pytest.raises(module.CharacterizationError, match="no_retry"): helper.dispatch_next()


@pytest.mark.parametrize("field,value", [("step_count", 9), ("step_count", 11),
    ("physics_delta", .005), ("reward_copies", 2)])
def test_wrong_native_cadence_or_callback_count_rejected(tmp_path, monkeypatch, field, value):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    setattr(env, field, value)
    with pytest.raises(NativeActionObservationError): helper.dispatch_next()


def test_terminal_event_stops_without_success_claim(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    env.force_end = True
    receipt = helper.dispatch_next()
    assert receipt["native_accounting"]["native_end_observed"] is True
    assert receipt["task_success_assessed"] is False
    with pytest.raises(module.CharacterizationError, match="no_retry"): helper.dispatch_next()


def test_initial_live_joint_mismatch_prevents_action(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    env.positions["left"][0] = .001
    with pytest.raises(module.CharacterizationError, match="envelope"): helper.dispatch_next()
    assert env.take_action_cnt == [0]


def test_com_jacobian_corrected_in_world_frame_with_live_joint_mapping(monkeypatch):
    from robodojo_uc71 import native_robot_inventory as inventory_module
    robots = []
    for arm in module.ARMS:
        raw = np.zeros((1, 1, 6, 8)); raw[0, 0, 0, 5] = -.7; raw[0, 0, 5, 5] = .7
        robots.append({"arm_name": arm + "_arm", "native_arm_joint_indices": [5, 4, 3, 2, 1, 0],
            "native_gripper_joint_indices": [6, 7], "body_names": ["base", "link6"], "is_fixed_base": True,
            "body_link_pose_w": {"values": [[[0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 2**-.5, 0., 0., 2**-.5]]]},
            "body_com_pos_b": {"values": [[[0., 0., 0.], [1., 0., 0.]]]},
            "root_physx_view_jacobians": {"values": raw.tolist()},
            "joint_positions": {"values": [list(range(8))]},
            "joint_velocities": {"values": [[0.] * 8]}})
    monkeypatch.setattr(inventory_module, "capture_robot_inventory", lambda env: {"robots": robots})
    result = module.capture_characterization_state(Env())
    for arm in module.ARMS:
        assert result["arms"][arm]["joint_positions"] == [5, 4, 3, 2, 1, 0]
        assert np.asarray(result["arms"][arm]["link6_actor_world_jacobian"])[:, 0] == pytest.approx([0., 0., 0., 0., 0., .7])
    assert result["benchmark_attempt_started"] is False


def test_initial_source_cannot_be_relabelled_as_an_arbitrary_plan(tmp_path, monkeypatch):
    _, plan, _, _ = setup(tmp_path, monkeypatch)
    changed = deepcopy(plan)
    changed["initial"]["left"]["joint_positions"][0] = .5
    with pytest.raises(module.CharacterizationError, match="recorded_robot_sources"):
        module.validate_probe_plan(changed)


def test_runtime_binding_denial_prevents_dispatch(tmp_path, monkeypatch):
    env, _, helper, _ = setup(tmp_path, monkeypatch)
    helper._verify_runtime = lambda plan: False
    with pytest.raises(module.CharacterizationError, match="runtime_or_target"): helper.dispatch_next()
    assert env.take_action_cnt == [0]
