"""Observe one real native joint dispatch without creating an episode lifecycle.

Authorization belongs to the caller and is checked immediately before dispatch.
Native reward/terminal callbacks are delegated and counted, never queried again;
their reward values and task-success flags are deliberately not returned.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import math

KEYS = frozenset({"left_arm_joint_state", "left_ee_joint_state",
                  "right_arm_joint_state", "right_ee_joint_state"})
_ABSENT = object()


class NativeActionObservationError(RuntimeError):
    pass


def clean_joint_action(action):
    if type(action) is not dict or set(action) != KEYS:
        raise NativeActionObservationError("exact_native_joint_keys_required")
    clean = {}
    for key in sorted(KEYS):
        values = action[key]
        size = 6 if "_arm_" in key else 1
        if (not isinstance(values, (list, tuple)) or len(values) != size
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)):
            raise NativeActionObservationError("finite_native_joint_values_required")
        clean[key] = [float(v) for v in values]
        if size == 1 and clean[key][0] not in (0., 1.):
            raise NativeActionObservationError("native_binary_gripper_command_required")
    return clean


@contextmanager
def _observe(owner, name, wrapper):
    previous = vars(owner).get(name, _ABSENT)
    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        if previous is _ABSENT:
            delattr(owner, name)
        else:
            setattr(owner, name, previous)


def observe_native_joint_action_once(env, action, *, verify_authorization):
    """Return a neutral accounting receipt; caller must poison its run on failure.

The verifier receives a copy of the exact action and the native counter before
dispatch. It must return literal True. This helper creates no approval, lifecycle,
retry, reward query, reset, rendering loop, or benchmark attempt.
"""
    action = clean_joint_action(action)
    if not callable(verify_authorization):
        raise NativeActionObservationError("per_dispatch_verifier_required")
    manager = env.reward_manager
    original = {"take_action": env.take_action, "sim_step": env.sim_step,
                "is_episode_end": env.is_episode_end, "get_reward": manager.get_reward,
                "reward_step": manager.step}

    def counter():
        value = env.take_action_cnt
        if not isinstance(value, (list, tuple)) or len(value) != 1 or type(value[0]) is not int or value[0] < 0:
            raise NativeActionObservationError("invalid_native_action_counter")
        return value[0]

    def clock():
        value = env.sim.sim.current_time
        if type(value) not in (float, int) or not math.isfinite(value):
            raise NativeActionObservationError("invalid_native_clock")
        return float(value)

    def physics_counter():
        value = env.sim._sim_step_counter
        if type(value) is not int or value < 0:
            raise NativeActionObservationError("invalid_native_physics_counter")
        return value

    def contract():
        if (env.num_envs != 1 or env.task_name != "make_toast" or env.config_name != "arx_x5"
                or list(env.env_seeds) != [0] or env.robot_manager.planner or env.robot_manager.ik_solver
                or getattr(env, "interact", False) or env.obs_manager.collect_interval != 10
                or env.sim.cfg.decimation != 1 or env.dt != .004 or env.sim.physics_dt != .004
                or env.sim.cfg.sim.dt != .004 or env.end_flag != [False] or env.have_empty([0]) is not True
                or env.take_action != original["take_action"] or env.sim_step != original["sim_step"]
                or env.is_episode_end != original["is_episode_end"] or env.reward_manager is not manager
                or manager.get_reward != original["get_reward"] or manager.step != original["reward_step"]):
            raise NativeActionObservationError("native_joint_dispatch_contract_changed")
        checks = manager.check_list
        if not isinstance(checks, list) or len(checks) != 1 or not checks[0]:
            raise NativeActionObservationError("native_predicates_not_registered")

    contract()
    before, time_before, ticks_before = counter(), clock(), physics_counter()
    if verify_authorization(deepcopy(action), before) is not True:
        raise NativeActionObservationError("diagnostic_dispatch_not_authorized")
    contract()
    if (counter(), clock(), physics_counter()) != (before, time_before, ticks_before):
        raise NativeActionObservationError("authorization_changed_native_state")
    counts = {"physics": 0, "terminal": 0, "reward": 0, "reward_step": 0}
    inside_end = False

    def step(*args, **kwargs):
        if counts["physics"] >= 10 or counts["terminal"] or counter() != before + 1:
            raise NativeActionObservationError("unexpected_native_substep")
        t, n = clock(), physics_counter()
        counts["physics"] += 1
        value = original["sim_step"](*args, **kwargs)
        if physics_counter() != n + 1 or not math.isclose(clock() - t, .004, rel_tol=0., abs_tol=1e-7):
            raise NativeActionObservationError("native_substep_cadence_mismatch")
        return value

    def reward_step(*args, **kwargs):
        if counts["physics"] != 10 or counts["reward_step"] or counts["terminal"]:
            raise NativeActionObservationError("unexpected_native_reward_step")
        counts["reward_step"] += 1
        return original["reward_step"](*args, **kwargs)

    def reward(*args, **kwargs):
        if not inside_end or counts["reward"]:
            raise NativeActionObservationError("unexpected_native_reward_callback")
        counts["reward"] += 1
        return original["get_reward"](*args, **kwargs)

    def terminal(*args, **kwargs):
        nonlocal inside_end
        if counts["physics"] != 10 or counts["reward_step"] != 1 or counts["terminal"]:
            raise NativeActionObservationError("unexpected_native_terminal_callback")
        counts["terminal"] += 1
        inside_end = True
        try:
            value = original["is_episode_end"](*args, **kwargs)
        finally:
            inside_end = False
        if type(value) is not bool or env.end_flag != [value] or counts["reward"] != 1:
            raise NativeActionObservationError("native_terminal_callback_mismatch")
        return value

    with ExitStack() as hooks:
        for owner, name, wrapper in ((env, "sim_step", step), (env, "is_episode_end", terminal),
                                     (manager, "step", reward_step), (manager, "get_reward", reward)):
            hooks.enter_context(_observe(owner, name, wrapper))
        original["take_action"](deepcopy(action))
    if (counts != {"physics": 10, "terminal": 1, "reward": 1, "reward_step": 1}
            or counter() != before + 1 or physics_counter() != ticks_before + 10
            or not math.isclose(clock() - time_before, .040, rel_tol=0., abs_tol=1e-7)
            or env.have_empty([0]) is not True):
        raise NativeActionObservationError("native_action_accounting_mismatch")
    return {"schema": "robodojo-native-dispatch-observation-1", "scope": "diagnostic_native_actions",
        "native_counter_before": before, "native_counter_after": counter(), "native_sim_step_calls": 10,
        "native_physics_counter_before": ticks_before, "native_physics_counter_after": physics_counter(),
        "physics_time_before_s": time_before, "physics_time_after_s": clock(), "physics_dt_s": .004,
        "native_terminal_callback_calls": 1, "native_reward_callback_calls": 1, "native_reward_step_calls": 1,
        "native_end_observed": env.end_flag[0], "task_success_assessed": False,
        "benchmark_attempt_started": False, "model_calls": 0}
