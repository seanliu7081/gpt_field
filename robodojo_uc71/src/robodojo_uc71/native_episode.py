"""Single-episode native action adapter; no simulator/model launch API.

Construct only around an already reset Standard make_toast/arx_x5 layout-0 env,
after native run_reward() registered its predicates. Construction does not reset,
register/evaluate reward, consume approval, or authorize a test. The mandatory
caller verifier must validate the current approval/readiness and exact dispatch
on EVERY call; a bare truth value supplied to the constructor is not accepted.

The caller owns EpisodeLifecycle: reserve_action() first, pass that ticket to
dispatch(), persist the returned evaluator_receipt, then complete_action() with
native_result. On failure the adapter is permanently poisoned; the caller must
stop/persist infrastructure_error, with its reserved action left unconfirmed.
There is no retry, ticket cancellation, horizon extension, or reset. The explicit
official_ee backend delegates EEF targets to native take_action without replacing
its IK, failed-IK behavior, interpolation, or gripper handling.

Pinned EvalEnv.take_action invokes is_episode_end once after native interpolation
and reward_manager.step. is_episode_end internally calls get_reward once but
returns only all(end_flag). Temporary observers delegate those original calls
exactly once and capture the real reward return; this module never requests an
additional reward evaluation, score calculation, or terminal check. Native
callback side effects, including final RGB capture, remain native behavior.

Each of ten env.sim_step delegations must advance one native physics counter and
0.004 seconds (decimation=1), with exactly one action-counter increment. Runtime
method hooks are restored after dispatch even on error. No source files change.

NativeActionReceipt is EVALUATOR-ONLY. It has no robot observation/control feedback
payload. A separate, calibrated reader must provide robot measurements to the
fragment executor. No TCP, joint ordering, Jacobian convention, or calibration is
inferred here; the upstream native gripper mapping is preserved without clipping.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
from time import monotonic

from .lifecycle import EpisodeLifecycle, NativeResult
from .official_control import (OFFICIAL_EE, validate_control_action,
                               validate_control_backend, validate_ik_status)
from .schema import vector


ACTION_KEYS = frozenset({"left_arm_joint_state", "left_ee_joint_state",
                         "right_arm_joint_state", "right_ee_joint_state"})
ACTION_LIMIT = 1400
SUBSTEPS = 10
PHYSICS_DT = .004
_MISSING = object()


class NativeEpisodeError(RuntimeError):
    """Sanitized contract failure; no native exception text is exposed."""


@dataclass(frozen=True)
class NativeActionReceipt:
    native_result: NativeResult
    evaluator_receipt: dict
    ik_status: dict | None = None


def validate_joint_action(action: dict) -> dict:
    if not isinstance(action, dict) or set(action) != ACTION_KEYS:
        raise NativeEpisodeError("exact_joint_mode_keys_required")
    clean = {}
    try:
        for key in sorted(ACTION_KEYS):
            size = 6 if "_arm_" in key else 1
            clean[key] = vector(action[key], size, "native joint action")
            if size == 1 and clean[key][0] not in (0., 1.):
                raise ValueError("Gripper command is not binary")
    except (TypeError, ValueError, OverflowError):
        raise NativeEpisodeError("invalid_native_joint_values") from None
    return clean


def _one(value, kind, code):
    if not isinstance(value, (list, tuple)) or len(value) != 1 or type(value[0]) is not kind:
        raise NativeEpisodeError(code)
    return value[0]


def _finite(value, code):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise NativeEpisodeError(code)
    return float(value)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _planner_signature(planner) -> str:
    """Bind native robot/scene settings without exposing them as feedback."""
    names = ("yml_path", "robot_cfg", "scene_model", "frame_bias", "tool_frames",
             "ee_link", "active_joints_name", "cspace_joint_names", "cspace_retract_config",
             "use_cuda_graph", "use_graph_planner", "dt")

    def serializable(value):
        if isinstance(value, Path):
            return str(value)
        if callable(getattr(value, "tolist", None)):
            return value.tolist()
        raise TypeError("unsupported_native_planner_setting")

    payload = {name: getattr(planner, name, None) for name in names}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                     default=serializable).encode()
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _observe_method(owner, name, wrapper):
    prior = vars(owner).get(name, _MISSING)
    setattr(owner, name, wrapper)
    try:
        yield
    finally:
        if prior is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, prior)


class NativeEpisodeActionAdapter:
    def __init__(self, env, *, lifecycle: EpisodeLifecycle, verify_authorization,
                 expected_layout_path: str | Path, expected_layout_sha256: str,
                 registration_evidence_id: str, episode_evidence_id: str,
                 control_backend: str = "legacy_joint"):
        # NativeWorkerSession constructs this immediately after consumed BEGIN.
        # This is evaluator-only wall time; it neither advances nor times replay.
        self._policy_begin_monotonic = monotonic()
        self._last_dispatch_completed = self._policy_begin_monotonic
        if not callable(verify_authorization):
            raise NativeEpisodeError("per_dispatch_authorization_verifier_required")
        if not isinstance(lifecycle, EpisodeLifecycle) or lifecycle.action_limit != ACTION_LIMIT:
            raise NativeEpisodeError("standard_caller_lifecycle_required")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_layout_sha256):
            raise NativeEpisodeError("pinned_layout_hash_required")
        for value in (registration_evidence_id, episode_evidence_id):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,200}", value):
                raise NativeEpisodeError("evaluator_evidence_identity_required")
        self.env, self.lifecycle = env, lifecycle
        try:
            self.control_backend = validate_control_backend(control_backend)
        except ValueError:
            raise NativeEpisodeError("invalid_control_backend") from None
        self._authorize = verify_authorization
        self._layout = Path(expected_layout_path).resolve(strict=True)
        if self._layout.name != "make_toast_0.json" or self._layout.parent.name != "0":
            raise NativeEpisodeError("explicit_numeric_layout_zero_required")
        self._layout_hash = expected_layout_sha256
        self._registration_id, self._episode_id = registration_evidence_id, episode_evidence_id
        self._last_ticket = 0
        self._poisoned = False
        self._lock = Lock()
        try:
            self._reward_manager = env.reward_manager
            checks = self._reward_manager.check_list
            if (not isinstance(checks, list) or len(checks) != 1
                    or not isinstance(checks[0], list) or not checks[0]):
                raise NativeEpisodeError("native_reward_predicates_not_registered")
            self._registered_check_container = checks
            self._take_action = env.take_action
            self._native_end = env.is_episode_end
            self._native_reward = self._reward_manager.get_reward
            self._native_step = env.sim_step
            self._planner_binding = self._ik_binding = self._native_ik = None
            self._planner_signatures = self._planner_children = None
            if self.control_backend == OFFICIAL_EE:
                robot_manager = env.robot_manager
                if (not isinstance(robot_manager.planner, dict) or not robot_manager.planner
                        or not isinstance(robot_manager.ik_solver, dict) or not robot_manager.ik_solver
                        or not callable(getattr(robot_manager, "solve_ik", None))):
                    raise NativeEpisodeError("official_native_ik_required")
                self._planner_binding = dict(robot_manager.planner)
                self._ik_binding = dict(robot_manager.ik_solver)
                self._native_ik = robot_manager.solve_ik
                self._planner_signatures = {name: _planner_signature(planner)
                    for name, planner in self._planner_binding.items()}
                self._planner_children = {name: tuple(getattr(planner, field, None)
                    for field in ("motion_planner", "motion_planner_batch", "ik_solver"))
                    for name, planner in self._planner_binding.items()}
            self._check_live()
            if (self._counter() != 0 or lifecycle.submitted_actions != 0
                    or lifecycle.confirmed_actions != 0 or lifecycle.snapshot()["in_flight"] is not None):
                raise NativeEpisodeError("fresh_reset_episode_required")
        except NativeEpisodeError:
            raise
        except Exception:
            raise NativeEpisodeError("native_episode_setup_unavailable") from None

    def _counter(self):
        count = _one(self.env.take_action_cnt, int, "invalid_native_action_counter")
        if not 0 <= count <= ACTION_LIMIT:
            raise NativeEpisodeError("invalid_native_action_counter")
        return count

    def _clock(self):
        return _finite(self.env.sim.sim.current_time, "invalid_native_physics_time")

    def _physics_counter(self):
        count = self.env.sim._sim_step_counter
        if type(count) is not int or count < 0:
            raise NativeEpisodeError("invalid_native_physics_counter")
        return count

    def _check_live(self):
        env = self.env
        if self._poisoned:
            raise NativeEpisodeError("adapter_failed_no_retry")
        if self.lifecycle.action_limit != ACTION_LIMIT:
            raise NativeEpisodeError("standard_caller_lifecycle_required")
        if (env.num_envs != 1 or env.step_lim != ACTION_LIMIT or env.task_name != "make_toast"
                or env.config_name != "arx_x5" or list(env.env_seeds) != [0]
                or getattr(env, "interact", False)):
            raise NativeEpisodeError("standard_episode_configuration_changed")
        robot_manager = env.robot_manager
        if self.control_backend == OFFICIAL_EE:
            names = {robot.robot_name for robot in robot_manager.robot_list}
            if (set(robot_manager.planner) != names or set(robot_manager.ik_solver) != names
                    or set(self._planner_binding) != names or set(self._ik_binding) != names
                    or any(robot_manager.planner[name] is not self._planner_binding[name]
                           or robot_manager.ik_solver[name] is not self._ik_binding[name]
                           or robot_manager.planner[name] is not robot_manager.ik_solver[name]
                           for name in names)
                    or robot_manager.solve_ik != self._native_ik):
                raise NativeEpisodeError("official_native_ik_binding_changed")
            for name in names:
                planner = robot_manager.planner[name]
                children = tuple(getattr(planner, field, None)
                    for field in ("motion_planner", "motion_planner_batch", "ik_solver"))
                if (_planner_signature(planner) != self._planner_signatures[name]
                        or any(current is not frozen for current, frozen in
                               zip(children, self._planner_children[name]))):
                    raise NativeEpisodeError("official_native_planner_configuration_changed")
        elif robot_manager.planner or robot_manager.ik_solver:
            raise NativeEpisodeError("standard_episode_configuration_changed")
        if (env.reward_manager is not self._reward_manager
                or self._reward_manager.check_list is not self._registered_check_container):
            raise NativeEpisodeError("registered_reward_manager_or_episode_replaced")
        selected = Path(env.seed_manager.seed_info[0]["scene_layout"]).resolve(strict=True)
        if selected != self._layout or _hash(selected) != self._layout_hash:
            raise NativeEpisodeError("selected_layout_binding_changed")
        if (env.obs_manager.collect_interval != SUBSTEPS or env.sim.cfg.decimation != 1
                or _finite(env.dt, "invalid_native_dt") != PHYSICS_DT
                or _finite(env.sim.physics_dt, "invalid_native_dt") != PHYSICS_DT
                or _finite(env.sim.cfg.sim.dt, "invalid_native_dt") != PHYSICS_DT):
            raise NativeEpisodeError("standard_native_action_cadence_changed")
        identities = {(r.arm_name, r.gripper_name) for r in env.robot_manager.robot_list}
        if (len(env.robot_manager.robot_list) != 2
                or identities != {("left_arm", "left_ee"), ("right_arm", "right_ee")}
                or any(r.type != "target" or r.ee_type != "gripper" for r in env.robot_manager.robot_list)):
            raise NativeEpisodeError("exact_dual_target_grippers_required")
        if _one(env.end_flag, bool, "invalid_native_end_flag") or self._counter() >= ACTION_LIMIT:
            raise NativeEpisodeError("native_episode_already_ended")
        if env.have_empty([0]) is not True:
            raise NativeEpisodeError("native_control_queue_not_empty")
        if (env.take_action != self._take_action or env.is_episode_end != self._native_end
                or env.sim_step != self._native_step or self._reward_manager.get_reward != self._native_reward):
            raise NativeEpisodeError("native_callback_identity_changed")

    def _check_reserved(self, ticket):
        snapshot = self.lifecycle.snapshot()
        if (snapshot["state"] != "RUNNING" or snapshot["in_flight"] != ticket
                or snapshot["submitted_actions"] != ticket or snapshot["confirmed_actions"] != ticket - 1
                or self._counter() != ticket - 1):
            raise NativeEpisodeError("monotonic_reserved_ticket_required")

    def dispatch(self, ticket: int, action: dict) -> NativeActionReceipt:
        if not self._lock.acquire(blocking=False):
            self._poisoned = True
            raise NativeEpisodeError("concurrent_dispatch_forbidden")
        try:
            dispatch_started = monotonic()
            self._check_live()
            if type(ticket) is not int or ticket != self._last_ticket + 1:
                raise NativeEpisodeError("monotonic_reserved_ticket_required")
            self._check_reserved(ticket)
            self._last_ticket = ticket  # consumed locally even if authorization or native execution fails
            try:
                clean = validate_control_action(action, self.control_backend)
            except ValueError:
                raise NativeEpisodeError("invalid_native_control_action") from None
            if self._authorize(ticket, deepcopy(clean)) is not True:
                raise NativeEpisodeError("dispatch_not_authorized")
            self._check_live()  # authorization cannot silently alter environment/configuration
            self._check_reserved(ticket)
            return self._dispatch_once(ticket, clean, dispatch_started)
        except NativeEpisodeError:
            self._poisoned = True
            raise
        except Exception:
            self._poisoned = True
            raise NativeEpisodeError("native_dispatch_failed") from None
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()

    def _dispatch_once(self, ticket: int, action: dict, dispatch_started: float) -> NativeActionReceipt:
        env, manager = self.env, self._reward_manager
        time_before, physics_before = self._clock(), self._physics_counter()
        state = {"step_calls": 0, "end_calls": 0, "reward_calls": 0, "inside_end": False}
        callback = {}
        ik_status = {}

        def observed_ik(*args, **kwargs):
            # The pinned evaluator calls solve_ik exactly once per target arm,
            # with these keyword arguments before interpolation. Delegate the
            # original call unchanged and retain only its public status enum.
            if (args or set(kwargs) != {"target_pose", "env_idx", "robot"}
                    or kwargs["env_idx"] != 0 or state["step_calls"] or self._counter() != ticket):
                raise NativeEpisodeError("unexpected_native_ik_call")
            robot = kwargs["robot"]
            if not any(robot is entry for entry in env.robot_manager.robot_list):
                raise NativeEpisodeError("unexpected_native_ik_robot")
            arm = {"left_arm": "left", "right_arm": "right"}.get(robot.arm_name)
            if arm is None or arm in ik_status or kwargs["target_pose"] != action[f"{arm}_ee_pose"]:
                raise NativeEpisodeError("unexpected_native_ik_target")
            result = self._native_ik(*args, **kwargs)
            if not isinstance(result, dict) or result.get("status") not in ("Success", "Fail"):
                raise NativeEpisodeError("invalid_native_ik_status")
            ik_status[arm] = result["status"]
            return result

        def observed_step(*args, **kwargs):
            if state["step_calls"] >= SUBSTEPS or state["end_calls"] or self._counter() != ticket:
                raise NativeEpisodeError("unexpected_native_physics_step")
            if self.control_backend == OFFICIAL_EE and set(ik_status) != {"left", "right"}:
                raise NativeEpisodeError("native_ik_calls_incomplete")
            before, counter = self._clock(), self._physics_counter()
            state["step_calls"] += 1
            result = self._native_step(*args, **kwargs)
            if (self._physics_counter() != counter + 1
                    or not math.isclose(self._clock() - before, PHYSICS_DT, rel_tol=0., abs_tol=1e-7)):
                raise NativeEpisodeError("native_substep_cadence_mismatch")
            return result

        def observed_reward(*args, **kwargs):
            if not state["inside_end"] or state["reward_calls"]:
                raise NativeEpisodeError("unexpected_native_reward_evaluation")
            state["reward_calls"] += 1
            rewards = self._native_reward(*args, **kwargs)
            if not isinstance(rewards, (list, tuple)) or len(rewards) != 1:
                raise NativeEpisodeError("invalid_native_reward_result")
            reward = _finite(rewards[0], "invalid_native_reward_result")
            if not 0 <= reward <= 1:
                raise NativeEpisodeError("invalid_native_reward_result")
            callback["reward"] = reward
            cached = manager._gated_score_lst
            if cached is not None and (not isinstance(cached, (list, tuple)) or len(cached) != 1):
                raise NativeEpisodeError("invalid_cached_native_score")
            partial = None if cached is None else _finite(cached[0], "invalid_cached_native_score")
            if partial is not None and not 0 <= partial <= 100:
                raise NativeEpisodeError("invalid_cached_native_score")
            callback["partial_score"] = partial
            return rewards

        def observed_end(*args, **kwargs):
            if state["end_calls"] or state["step_calls"] != SUBSTEPS or self._counter() != ticket:
                raise NativeEpisodeError("unexpected_native_terminal_callback")
            state["end_calls"] += 1
            state["inside_end"] = True
            try:
                returned = self._native_end(*args, **kwargs)
                state["callback_completed_monotonic"] = monotonic()
            finally:
                state["inside_end"] = False
            ended = _one(env.end_flag, bool, "invalid_native_end_flag")
            success = _one(env.success, bool, "invalid_native_success_flag")
            if type(returned) is not bool or returned != ended or state["reward_calls"] != 1:
                raise NativeEpisodeError("native_terminal_callback_not_verified")
            if ((callback["reward"] > 1 - 1e-3 and (not ended or not success))
                    or (ticket == ACTION_LIMIT and not ended) or (not success and not ended)):
                raise NativeEpisodeError("native_terminal_state_inconsistent")
            callback.update(end_flag=ended, native_success_flag=success, returned_all_ended=returned)
            return returned

        with ExitStack() as hooks:
            hooks.enter_context(_observe_method(env, "sim_step", observed_step))
            hooks.enter_context(_observe_method(env, "is_episode_end", observed_end))
            hooks.enter_context(_observe_method(manager, "get_reward", observed_reward))
            if self.control_backend == OFFICIAL_EE:
                hooks.enter_context(_observe_method(env.robot_manager, "solve_ik", observed_ik))
            self._take_action(deepcopy(action))
        if (state["step_calls"] != SUBSTEPS or state["end_calls"] != 1 or state["reward_calls"] != 1
                or self._counter() != ticket or self._physics_counter() != physics_before + SUBSTEPS
                or not math.isclose(self._clock() - time_before, SUBSTEPS * PHYSICS_DT, rel_tol=0., abs_tol=1e-7)):
            raise NativeEpisodeError("native_action_accounting_mismatch")
        result = NativeResult(ticket, callback["reward"], callback["end_flag"], True, True,
                              f"{self._episode_id}:native-action:{ticket}", callback["partial_score"])
        timing = {"clock":"time.monotonic","policy_begin_monotonic_s":self._policy_begin_monotonic,
            "dispatch_started_monotonic_s":dispatch_started,
            "native_callback_completed_monotonic_s":state["callback_completed_monotonic"],
            "dispatch_completed_monotonic_s":monotonic()}
        values = [timing[key] for key in timing if key != "clock"]
        if (any(type(value) not in (int,float) or not math.isfinite(value) or value < 0 for value in values)
                or values != sorted(values) or dispatch_started < self._last_dispatch_completed):
            raise NativeEpisodeError("invalid_native_policy_wall_timing")
        self._last_dispatch_completed = timing["dispatch_completed_monotonic_s"]
        receipt = {"schema": "robodojo-native-action-receipt-1", "scope": "evaluator_only",
            "model_input_eligible": False, "calibration_status": "uncalibrated_candidate",
            "episode_evidence_id": self._episode_id, "registration_evidence_id": self._registration_id,
            "layout_sha256": self._layout_hash, "authorization_verified_for_ticket": ticket,
            "native_action_index": ticket, "physics_dt_s": PHYSICS_DT, "native_sim_step_calls": state["step_calls"],
            "native_physics_counter_before": physics_before, "native_physics_counter_after": self._physics_counter(),
            "physics_time_before_s": time_before, "physics_time_after_s": self._clock(),
            "native_terminal_callback_calls": state["end_calls"], "native_reward_callback_calls": state["reward_calls"],
            "native_callback_state": callback, "native_result": asdict(result),"timing":timing}
        if self.control_backend == OFFICIAL_EE:
            statuses = validate_ik_status(ik_status)
            receipt.update(control_backend=self.control_backend, ik_status=statuses)
            return NativeActionReceipt(result, receipt, deepcopy(statuses))
        return NativeActionReceipt(result, receipt)
