"""Synthetic native-shaped episode; no simulator/model import or launch."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from robodojo_uc71.lifecycle import EpisodeLifecycle
from robodojo_uc71.native_episode import (NativeEpisodeActionAdapter, NativeEpisodeError,
                                         validate_joint_action)
import robodojo_uc71.native_episode as native_episode_module


def running():
    run = EpisodeLifecycle()
    run.ready()
    run.start({"schema": "robodojo-uc71-consumed-1", "attempts": 1,
               "claim_nonce": "synthetic-only", "manifest_sha256": "a" * 64,
               "ready_sha256": "b" * 64})
    return run


def action():
    return {"left_arm_joint_state": [0., .1, .2, .3, .4, .5],
            "right_arm_joint_state": [0.] * 6,
            "left_ee_joint_state": [0.], "right_ee_joint_state": [1.]}


class RewardManager:
    def __init__(self):
        self.check_list = [[[("synthetic_registered_predicate", {})]]]
        self._gated_score_lst = None
        self.calls = 0
        self.reward = 0.
        self.final_checks = []

    def get_reward(self, final_check=True):
        self.calls += 1
        self.final_checks.append(final_check)
        self._gated_score_lst = [25.]
        return [self.reward]

    def get_score(self):
        pytest.fail("adapter must never query score")


class NativeEnv:
    def __init__(self, layout):
        self.num_envs, self.step_lim = 1, 1400
        self.task_name, self.config_name, self.env_seeds = "make_toast", "arx_x5", [0]
        self.dt, self.interact = .004, False
        self.seed_manager = NS(seed_info=[{"scene_layout": str(layout)}])
        robots = [NS(arm_name=f"{side}_arm", gripper_name=f"{side}_ee", type="target", ee_type="gripper")
                  for side in ("left", "right")]
        self.robot_manager = NS(planner={}, ik_solver={}, robot_list=robots)
        self.obs_manager = NS(collect_interval=10)
        self.sim = NS(cfg=NS(decimation=1, sim=NS(dt=.004)), physics_dt=.004,
                      sim=NS(current_time=1.2), _sim_step_counter=300)
        self.reward_manager = RewardManager()
        self.take_action_cnt, self.end_flag, self.success = [0], [False], [True]
        self.actions = []
        self.step_count = 10
        self.callback_count = 1
        self.counter_increment = 1
        self.physics_delta = .004
        self.queue_empty = True
        self.fail_at_step = None
        self.live_steps = 0
        self.terminal_calls = 0
        self.reward_evaluations = 1
        self.call_reward_outside = False

    def have_empty(self, env_idx_list):
        assert env_idx_list == [0]
        return self.queue_empty

    def sim_step(self, render=False):
        self.live_steps += 1
        self.sim._sim_step_counter += 1
        self.sim.sim.current_time += self.physics_delta
        if self.live_steps == self.fail_at_step:
            raise RuntimeError("sensitive native details must not escape")

    def is_episode_end(self):
        self.terminal_calls += 1
        for _ in range(self.reward_evaluations):
            rewards = self.reward_manager.get_reward(final_check=self.take_action_cnt[0] >= self.step_lim)
        if rewards[0] > 1 - 1e-3:
            self.end_flag[0], self.success[0] = True, True
        elif self.take_action_cnt[0] >= self.step_lim or not self.success[0]:
            self.end_flag[0], self.success[0] = True, False
        return all(self.end_flag)

    def take_action(self, joint_action):
        self.actions.append(deepcopy(joint_action))
        self.take_action_cnt[0] += self.counter_increment
        for _ in range(self.step_count):
            self.sim_step(render=False)
        if self.call_reward_outside:
            self.reward_manager.get_reward()
        for _ in range(self.callback_count):
            self.is_episode_end()


class OfficialNativeEnv(NativeEnv):
    """Synthetic native EEF dispatch, including native failed-IK behavior."""
    def __init__(self, layout):
        super().__init__(layout)
        planner = NS(robot_cfg={"synthetic_kinematics": True},
                     scene_model={"synthetic_table": True}, ik_solver=object())
        self.robot_manager.planner = {"x5": planner}
        self.robot_manager.ik_solver = {"x5": planner}
        self.robot_manager.solve_ik = self.solve_ik
        for robot in self.robot_manager.robot_list:
            robot.robot_name = "x5"
        self.ik_status = {"left": "Success", "right": "Success"}
        self.ik_calls = []
        self.skip_ik_arm = None
        self.joint_targets = {arm: [0.] * 6 for arm in self.ik_status}
        self.gripper_targets = {arm: [1.] for arm in self.ik_status}

    def solve_ik(self, *, target_pose, env_idx, robot):
        assert env_idx == 0 and self.take_action_cnt[0] == len(self.actions)
        arm = robot.arm_name.removesuffix("_arm")
        self.ik_calls.append((arm, deepcopy(target_pose)))
        return {"status": self.ik_status[arm], "joint_value": [.1] * 6,
                "native_private_reason": "must never cross the control boundary"}

    def take_action(self, ee_action):
        self.actions.append(deepcopy(ee_action))
        self.take_action_cnt[0] += self.counter_increment
        for robot in self.robot_manager.robot_list:
            arm = robot.arm_name.removesuffix("_arm")
            if arm != self.skip_ik_arm:
                result = self.robot_manager.solve_ik(target_pose=ee_action[f"{arm}_ee_pose"],
                    env_idx=0, robot=robot)
                if result["status"] == "Success":
                    self.joint_targets[arm] = result["joint_value"]
            # Just like the pinned native dispatcher, Fail leaves the previous
            # arm command in force while the gripper command still executes.
            self.gripper_targets[arm] = deepcopy(ee_action[f"{arm}_ee_joint_state"])
        for _ in range(self.step_count):
            self.sim_step(render=False)
        for _ in range(self.callback_count):
            self.is_episode_end()


def ee_action():
    return {"left_ee_pose": [.1, .2, .3, 1., 0., 0., 0.],
            "right_ee_pose": [.4, .5, .6, 1., 0., 0., 0.],
            "left_ee_joint_state": [0.], "right_ee_joint_state": [1.]}


def setup(tmp_path, verifier=None, configure=None, *, control_backend="legacy_joint"):
    layout = tmp_path / "arx_x5" / "0" / "make_toast_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{"synthetic_only": true}\n')
    env_type = OfficialNativeEnv if control_backend == "official_ee" else NativeEnv
    env, lifecycle = env_type(layout), running()
    if configure is not None:
        configure(env)
    verifications = []
    def authorized(ticket, normalized):
        verifications.append((ticket, deepcopy(normalized)))
        return True
    adapter = NativeEpisodeActionAdapter(env, lifecycle=lifecycle,
        verify_authorization=authorized if verifier is None else verifier,
        expected_layout_path=layout, expected_layout_sha256=hashlib.sha256(layout.read_bytes()).hexdigest(),
        registration_evidence_id="synthetic-registration", episode_evidence_id="synthetic-episode",
        control_backend=control_backend)
    return env, lifecycle, adapter, verifications


@pytest.mark.parametrize("failed_arm", [None, "left", "right"])
def test_official_ee_delegates_once_and_preserves_native_failed_ik_and_grippers(tmp_path, failed_arm):
    env, lifecycle, adapter, approvals = setup(tmp_path, control_backend="official_ee")
    if failed_arm:
        env.ik_status[failed_arm] = "Fail"
        env.joint_targets[failed_arm] = [.3] * 6
    original_ik = env.robot_manager.solve_ik
    ticket = lifecycle.reserve_action()
    receipt = adapter.dispatch(ticket, ee_action())
    lifecycle.complete_action(ticket, receipt.native_result)
    assert env.actions == [ee_action()] and len(approvals) == 1
    assert env.ik_calls == [(arm, ee_action()[f"{arm}_ee_pose"]) for arm in ("left", "right")]
    assert env.robot_manager.solve_ik == original_ik
    assert env.take_action_cnt == [1] and env.live_steps == 10
    assert env.reward_manager.calls == env.terminal_calls == lifecycle.confirmed_actions == 1
    assert env.gripper_targets == {"left": [0.], "right": [1.]}
    if failed_arm:
        assert env.joint_targets[failed_arm] == [.3] * 6
    assert receipt.ik_status == env.ik_status
    assert receipt.evaluator_receipt["ik_status"] == env.ik_status
    assert receipt.evaluator_receipt["control_backend"] == "official_ee"
    assert "native_private_reason" not in json.dumps(receipt.evaluator_receipt)


@pytest.mark.parametrize("change", ["unknown_key", "joint_key", "quaternion", "gripper"])
def test_official_ee_invalid_targets_never_dispatch(tmp_path, change):
    env, lifecycle, adapter, approvals = setup(tmp_path, control_backend="official_ee")
    invalid = ee_action()
    if change == "unknown_key": invalid["scene_object"] = [1.]
    elif change == "joint_key": invalid["left_arm_joint_state"] = [0.] * 6
    elif change == "quaternion": invalid["left_ee_pose"][3:] = [0.] * 4
    else: invalid["left_ee_joint_state"] = [.5]
    with pytest.raises(NativeEpisodeError):
        adapter.dispatch(lifecycle.reserve_action(), invalid)
    assert not env.actions and not env.ik_calls and not approvals


def test_official_ee_missing_native_ik_call_cannot_create_confirmed_action(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path, control_backend="official_ee")
    env.skip_ik_arm = "right"
    ticket = lifecycle.reserve_action()
    with pytest.raises(NativeEpisodeError, match="native_ik_calls_incomplete"):
        adapter.dispatch(ticket, ee_action())
    assert lifecycle.confirmed_actions == 0 and env.live_steps == 0
    assert env.take_action_cnt == [1]
    with pytest.raises(NativeEpisodeError, match="adapter_failed_no_retry"):
        adapter.dispatch(ticket, ee_action())
    assert len(env.actions) == 1


def test_official_ee_changed_solver_is_rejected_before_dispatch(tmp_path):
    env, lifecycle, adapter, approvals = setup(tmp_path, control_backend="official_ee")
    env.robot_manager.ik_solver["x5"] = object()
    with pytest.raises(NativeEpisodeError, match="official_native_ik_binding_changed"):
        adapter.dispatch(lifecycle.reserve_action(), ee_action())
    assert not env.actions and not approvals


@pytest.mark.parametrize("change", ["scene", "solver"])
def test_official_ee_native_planner_mutation_is_rejected_before_dispatch(tmp_path, change):
    env, lifecycle, adapter, approvals = setup(tmp_path, control_backend="official_ee")
    planner = env.robot_manager.planner["x5"]
    if change == "scene":
        planner.scene_model["synthetic_table"] = False
    else:
        planner.ik_solver = object()
    with pytest.raises(NativeEpisodeError, match="official_native_planner_configuration_changed"):
        adapter.dispatch(lifecycle.reserve_action(), ee_action())
    assert not env.actions and not approvals


def test_receipt_wall_clock_uses_actual_begin_dispatch_and_callback_boundaries(tmp_path,monkeypatch):
    ticks = iter((100.,110.,112.,113.,120.,122.,123.))
    monkeypatch.setattr(native_episode_module,"monotonic",lambda:next(ticks))
    env,lifecycle,adapter,_ = setup(tmp_path)
    for ticket,expected in ((1,(110.,112.,113.)),(2,(120.,122.,123.))):
        assert lifecycle.reserve_action() == ticket
        receipt = adapter.dispatch(ticket,action())
        lifecycle.complete_action(ticket,receipt.native_result)
        timing = receipt.evaluator_receipt["timing"]
        assert timing == {"clock":"time.monotonic","policy_begin_monotonic_s":100.,
            "dispatch_started_monotonic_s":expected[0],"native_callback_completed_monotonic_s":expected[1],
            "dispatch_completed_monotonic_s":expected[2]}
    assert env.reward_manager.calls == 2 and env.terminal_calls == 2


def dispatch(lifecycle, adapter):
    ticket = lifecycle.reserve_action()
    receipt = adapter.dispatch(ticket, action())
    lifecycle.complete_action(ticket, receipt.native_result)
    return receipt


def test_exact_native_callback_and_ten_physics_steps_without_duplicate_reward(tmp_path):
    env, lifecycle, adapter, verifications = setup(tmp_path)
    originals = env.take_action, env.is_episode_end, env.sim_step, env.reward_manager.get_reward
    receipt = dispatch(lifecycle, adapter)
    assert env.actions == [action()]
    assert env.live_steps == 10 and env.terminal_calls == env.reward_manager.calls == 1
    assert env.reward_manager.final_checks == [False]
    assert env.sim.sim.current_time == pytest.approx(1.24)
    assert lifecycle.confirmed_actions == 1 and lifecycle.state == "RUNNING"
    assert len(verifications) == 1 and verifications[0][0] == 1
    assert originals == (env.take_action, env.is_episode_end, env.sim_step, env.reward_manager.get_reward)
    assert "is_episode_end" not in vars(env) and "get_reward" not in vars(env.reward_manager)
    assert receipt.native_result.native_reward == 0. and receipt.native_result.partial_score == 25.
    assert receipt.evaluator_receipt["scope"] == "evaluator_only"
    assert receipt.evaluator_receipt["model_input_eligible"] is False
    assert receipt.evaluator_receipt["calibration_status"] == "uncalibrated_candidate"
    json.dumps(receipt.evaluator_receipt, allow_nan=False)


def test_authorization_is_required_on_every_dispatch_and_denial_poisoned(tmp_path):
    approvals = []
    def verify(ticket, normalized):
        approvals.append(ticket)
        normalized["left_arm_joint_state"][0] = 999.  # cannot mutate the actual native action
        return ticket == 1
    env, lifecycle, adapter, _ = setup(tmp_path, verify)
    dispatch(lifecycle, adapter)
    ticket = lifecycle.reserve_action()
    with pytest.raises(NativeEpisodeError, match="dispatch_not_authorized"):
        adapter.dispatch(ticket, action())
    assert approvals == [1, 2]
    assert env.actions[0]["left_arm_joint_state"][0] == 0.
    with pytest.raises(NativeEpisodeError, match="adapter_failed_no_retry"):
        adapter.dispatch(ticket, action())
    assert len(env.actions) == 1 and lifecycle.snapshot()["in_flight"] == 2


def test_mandatory_callable_authorization_cannot_be_replaced_with_boolean(tmp_path):
    with pytest.raises(NativeEpisodeError, match="authorization_verifier_required"):
        setup(tmp_path, True)


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(left_ee_pose=[0.] * 7),
    lambda x: x.pop("right_ee_joint_state"),
    lambda x: x.update(left_arm_joint_state=[0.] * 5),
    lambda x: x.update(left_arm_joint_state=[0.] * 5 + [float("nan")]),
    lambda x: x.update(left_arm_joint_state=[0.] * 5 + [float("inf")]),
    lambda x: x.update(left_ee_joint_state=[.5]),
    lambda x: x.update(left_ee_joint_state=[True]),
])
def test_strict_joint_only_boundary_rejects_before_native_dispatch(tmp_path, mutate):
    env, lifecycle, adapter, approvals = setup(tmp_path)
    invalid = action()
    mutate(invalid)
    with pytest.raises(NativeEpisodeError):
        adapter.dispatch(lifecycle.reserve_action(), invalid)
    assert not env.actions and env.live_steps == env.reward_manager.calls == 0 and approvals == []


def test_monotonically_reserved_ticket_required_not_just_an_integer(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path)
    with pytest.raises(NativeEpisodeError, match="monotonic_reserved_ticket_required"):
        adapter.dispatch(1, action())
    assert not env.actions


def test_native_success_is_captured_only_from_real_callback_then_no_more_actions(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path)
    assert env.success == [True]  # reset viability flag is not completion
    env.reward_manager.reward = 1.
    receipt = dispatch(lifecycle, adapter)
    assert receipt.native_result.end_flag and receipt.native_result.native_reward == 1.
    assert lifecycle.terminal_record()["official_success"] is True
    with pytest.raises(NativeEpisodeError, match="native_episode_already_ended"):
        adapter.dispatch(2, action())
    assert len(env.actions) == 1 and env.reward_manager.calls == 1


@pytest.mark.parametrize("setting,value,expected_steps", [
    ("step_count", 9, 9), ("step_count", 11, 10),
    ("callback_count", 0, 10), ("callback_count", 2, 10),
    ("counter_increment", 2, 0), ("physics_delta", .008, 1),
    ("reward_evaluations", 2, 10), ("call_reward_outside", True, 10),
])
def test_native_accounting_deviation_fails_without_retry_or_duplicate_callback(tmp_path, setting, value, expected_steps):
    env, lifecycle, adapter, _ = setup(tmp_path)
    setattr(env, setting, value)
    ticket = lifecycle.reserve_action()
    with pytest.raises(NativeEpisodeError):
        adapter.dispatch(ticket, action())
    assert env.live_steps == expected_steps
    assert env.reward_manager.calls <= 1 and env.terminal_calls <= 1
    with pytest.raises(NativeEpisodeError, match="adapter_failed_no_retry"):
        adapter.dispatch(ticket, action())
    assert len(env.actions) == 1
    assert lifecycle.confirmed_actions == 0 and lifecycle.snapshot()["in_flight"] == 1
    assert "sim_step" not in vars(env) and "is_episode_end" not in vars(env)


def test_native_exception_is_sanitized_hooks_restored_and_reserved_action_not_resent(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path)
    env.fail_at_step = 3
    ticket = lifecycle.reserve_action()
    with pytest.raises(NativeEpisodeError, match="^native_dispatch_failed$"):
        adapter.dispatch(ticket, action())
    assert env.live_steps == 3 and env.reward_manager.calls == 0
    assert "sim_step" not in vars(env) and "get_reward" not in vars(env.reward_manager)
    with pytest.raises(NativeEpisodeError, match="adapter_failed_no_retry"):
        adapter.dispatch(ticket, action())
    lifecycle.stop("infrastructure_error")
    assert lifecycle.terminal_record()["unconfirmed_action"] == 1


@pytest.mark.parametrize("alter", [
    lambda e: setattr(e.sim.cfg, "decimation", 2),
    lambda e: setattr(e.obs_manager, "collect_interval", 9),
    lambda e: setattr(e, "dt", .005),
    lambda e: setattr(e, "step_lim", 1401),
    lambda e: e.robot_manager.planner.update(implicit=True),
    lambda e: setattr(e, "queue_empty", False),
    lambda e: setattr(e.reward_manager, "check_list", [[[]]]),
    lambda e: setattr(e, "env_seeds", [1]),
])
def test_changed_native_configuration_or_reset_rejected_before_authorization(tmp_path, alter):
    env, lifecycle, adapter, approvals = setup(tmp_path)
    alter(env)
    with pytest.raises(NativeEpisodeError):
        adapter.dispatch(lifecycle.reserve_action(), action())
    assert not env.actions and not approvals


def test_authorization_cannot_replace_bound_layout_between_checks(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path)
    def change_layout(ticket, normalized):
        (tmp_path / "arx_x5/0/make_toast_0.json").write_text("changed")
        return True
    adapter._authorize = change_layout
    with pytest.raises(NativeEpisodeError, match="selected_layout_binding_changed"):
        adapter.dispatch(lifecycle.reserve_action(), action())
    assert not env.actions


def test_authorization_cannot_end_lifecycle_then_dispatch_a_reserved_action(tmp_path):
    env, lifecycle, adapter, _ = setup(tmp_path)
    def stop_lifecycle(ticket, normalized):
        lifecycle.stop("infrastructure_error")
        return True
    adapter._authorize = stop_lifecycle
    with pytest.raises(NativeEpisodeError, match="monotonic_reserved_ticket_required"):
        adapter.dispatch(lifecycle.reserve_action(), action())
    assert not env.actions


@pytest.mark.parametrize("configure", [
    lambda e: setattr(e.reward_manager, "check_list", [[]]),
    lambda e: setattr(e, "take_action_cnt", [1]),
    lambda e: setattr(e, "end_flag", [True]),
])
def test_constructor_requires_registered_fresh_unended_episode(tmp_path, configure):
    with pytest.raises(NativeEpisodeError):
        setup(tmp_path, configure=configure)


def test_exact_standard_horizon_1400_keeps_single_native_callback_each_action(tmp_path):
    env, lifecycle, adapter, approvals = setup(tmp_path)
    for _ in range(1400):
        receipt = dispatch(lifecycle, adapter)
    assert len(env.actions) == len(approvals) == env.terminal_calls == env.reward_manager.calls == 1400
    assert env.live_steps == 14000
    assert env.reward_manager.final_checks[-1] is True
    assert all(not value for value in env.reward_manager.final_checks[:-1])
    assert receipt.native_result.end_flag and receipt.native_result.native_reward == 0.
    assert lifecycle.terminal_record()["status"] == "horizon_exhausted"
    with pytest.raises(NativeEpisodeError):
        adapter.dispatch(1401, action())
    assert len(env.actions) == 1400
