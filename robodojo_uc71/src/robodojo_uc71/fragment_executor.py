"""Pure, candidate ARX executor for model-selected generic fragments.

Integration contract (no simulator/model imports or launch authorization here):
* The caller owns a RUNNING EpisodeLifecycle, started only after the separate
  approval gate. Standard uses its unchanged 1400-action horizon. This module
  never creates approval, resets the episode, extends the horizon, or retries.
* measure() returns synchronized robot-only ArmState and the two measured finger
  joint positions per arm, in metres. Finger gap is their declared sum proxy,
  never an object/contact/grasp measurement. Jacobian/TCP calibration belongs to
  the caller and must be independently validated before native execution.
* send_native_joint_action(ticket, action), whose legacy callback name is kept
  for compatibility, performs exactly ONE acknowledged
  native env.take_action, including native interpolation, and returns the
  post-action RobotMeasurement plus the registered evaluator's NativeResult.
  Legacy actions use {left,right}_arm_joint_state (six radians); official_ee
  actions use {left,right}_ee_pose (world link6 xyz+wxyz). Both use
  {left,right}_ee_joint_state (one normalized scalar: open=1, close=0).
  NativeResult is passed only to lifecycle accounting; no reward or object
  truth enters target generation or the allowlisted model-facing feedback.
* Required MonitoringSettings have no guessed ARX defaults. Supplying values
  does not calibrate or validate this candidate. ControllerConfig remains the
  existing uncalibrated candidate; live readiness is an external gate.
* Targets are resolved once from each stage's measured entry. Legacy dispatch
  recomputes differential control from the latest acknowledged measurement;
  official_ee sends the complete target through native cuRobo/interpolation.
  The inactive arm holds its measured joint reference in legacy mode and its
  measured link6 pose in official mode; the latter does not guarantee identical
  joint angles. Official mode does not use the legacy DLS speed limits. Grippers
  latch. Only explicit model-selected stages run. Wait/checkpoints yield; pending
  stages are discarded, with no automatic resume, retreat, lift, or release.
  Direct motion yields on stable measured arrival or its requested allowance.
* observe/finish issue zero native actions and request an observation. Local
  'succeeded' means a measured predicate/allowance completed, never task success.
  Only feedback is suitable for model input; retain lifecycle/operator counters
  separately. Exceptions are sanitized codes; failed dispatches remain reserved
  and terminate infrastructure accounting, with no resend.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .controller import (ArmState, ControllerConfig, ControllerError,
                         DualArmController, measured_tcp_pose, orientation_error_world)
from .lifecycle import EpisodeLifecycle, LifecycleError, NativeResult
from .official_control import (LEGACY_JOINT, OFFICIAL_EE, OfficialEEController,
                               validate_control_backend, validate_ik_status)
from .schema import (ARMS, ExecutionLimits, validate_controller_capabilities,
                     validate_decision, validate_feedback, vector)


@dataclass(frozen=True)
class RobotMeasurement:
    arms: Mapping[str, ArmState]
    finger_position_m: Mapping[str, object]


@dataclass(frozen=True)
class ActionReceipt:
    measurement: RobotMeasurement
    native_result: NativeResult
    ik_status: Mapping[str, str] | None = None


@dataclass(frozen=True)
class MonitoringSettings:
    """Explicit robot-only thresholds; synthetic values are not calibration."""
    position_tolerance_m: Mapping[str, float]
    orientation_tolerance_rad: Mapping[str, float]
    arrival_stable_samples: int
    gap_stable_samples: int
    gap_stable_range_m: float
    gap_rise_yield_m: float
    gap_rise_samples: int
    stall_window_steps: int
    stall_min_steps: int
    stall_progress_m: float
    orientation_error_weight_m_per_rad: float

    def checked(self) -> dict:
        values = validate_controller_capabilities(deepcopy(vars(self)))
        if self.stall_window_steps < 2 or self.stall_min_steps < self.stall_window_steps:
            raise ValueError("Stall monitoring requires at least two samples within its minimum steps")
        return values


@dataclass(frozen=True)
class FragmentOutcome:
    feedback: dict
    submitted_steps: int
    confirmed_steps: int
    observation_required: bool = True
    calibration_status: str = "uncalibrated_candidate"


class FragmentExecutionError(RuntimeError):
    """Sanitized infrastructure code; never embed worker exceptions in feedback."""


_MOTION = frozenset({"move_to_pose", "move_relative", "approach", "motion"})
_GRIP = {"open": 1.0, "close": 0.0}


def _measurement(value: RobotMeasurement) -> RobotMeasurement:
    if (not isinstance(value, RobotMeasurement) or not isinstance(value.arms, Mapping)
            or set(value.arms) != set(ARMS) or not isinstance(value.finger_position_m, Mapping)
            or set(value.finger_position_m) != set(ARMS)):
        raise ValueError("Both measured arms and finger pairs are required")
    result = deepcopy(value)
    for arm in ARMS:
        measured_tcp_pose(result.arms[arm])  # validates the complete ArmState
        vector(result.finger_position_m[arm], 2, "measured finger joints")
    return result


def _pose(sample: RobotMeasurement, arm: str) -> np.ndarray:
    xyz, quat = measured_tcp_pose(sample.arms[arm])
    return np.concatenate((xyz, quat))


class FragmentExecutor:
    def __init__(self, *, lifecycle: EpisodeLifecycle,
                 measure: Callable[[], RobotMeasurement],
                 send_native_joint_action: Callable[[int, dict], ActionReceipt],
                 gripper_commands: Mapping[str, str], monitoring: MonitoringSettings,
                 controller_config: ControllerConfig, limits: ExecutionLimits | None = None,
                 stop_requested: Callable[[], bool] | None = None,
                 control_backend: str = LEGACY_JOINT):
        if not isinstance(lifecycle, EpisodeLifecycle):
            raise TypeError("Caller-owned EpisodeLifecycle is required")
        if set(gripper_commands) != set(ARMS) or any(v not in _GRIP for v in gripper_commands.values()):
            raise ValueError("Both last requested binary gripper commands are required")
        self.lifecycle = lifecycle
        self._measure, self._send = measure, send_native_joint_action
        self._stop_requested = stop_requested or (lambda: False)
        self._settings = monitoring.checked()
        self._limits = limits or ExecutionLimits()
        if not isinstance(self._limits, ExecutionLimits):
            raise TypeError("ExecutionLimits is required")
        self._grippers = dict(gripper_commands)
        self.control_backend = validate_control_backend(control_backend)
        latches = {a: _GRIP[g] for a, g in gripper_commands.items()}
        self._controller = (OfficialEEController(gripper_commands=latches)
            if self.control_backend == OFFICIAL_EE else
            DualArmController(gripper_commands=latches, config=controller_config))
        self._active_arm = None
        self._busy = False

    def run(self, decision: dict) -> FragmentOutcome:
        """Execute one fully validated decision; no retained continuation queue."""
        clean = validate_decision(decision, self._limits)
        if self._busy:
            raise FragmentExecutionError("executor_already_running")
        if clean["type"] in ("observe", "finish"):
            kind = clean["type"]
            return FragmentOutcome(validate_feedback({"status": "pending_verification",
                "reason": "observation_only" if kind == "observe" else "model_finish_observation",
                "skill": kind, "executed_steps": 0}), 0, 0)
        if self.lifecycle.state != "RUNNING":
            raise LifecycleError("Physical execution requires the caller's RUNNING lifecycle")
        self._busy = True
        try:
            return self._run(clean)
        except (FragmentExecutionError, LifecycleError, KeyboardInterrupt):
            self.lifecycle.stop("infrastructure_error")
            raise
        except Exception:
            self.lifecycle.stop("infrastructure_error")
            raise FragmentExecutionError("executor_infrastructure_error") from None
        finally:
            self._busy = False

    def _read(self) -> RobotMeasurement:
        try:
            return _measurement(self._measure())
        except Exception:
            raise FragmentExecutionError("measurement_unavailable") from None

    def _run(self, decision: dict) -> FragmentOutcome:
        arm = decision["arm_id"]
        direct = decision["type"] == "motion"
        if direct:
            motion = decision["motion"]
            segments = [{**motion, "skill": "motion", "profile": "transit",
                         "max_steps": motion["steps"], "observation_after": True}]
        else:
            segments = decision["execution"]["segments"]
        sample = self._read()
        if self._active_arm != arm:
            self._controller.handoff(arm, sample.arms)
            self._active_arm = arm
        submitted = confirmed = 0
        completed = []
        settings = self._settings
        last = {}

        def outcome(status, reason):
            feedback = {"status": status, "reason": reason, "arm_id": arm,
                "skill": segment["skill"], "executed_steps": confirmed,
                "gripper_command": self._grippers[arm], **last}
            if not direct:
                feedback.update(completed_segment_indices=list(completed),
                    pending_segment_indices=[i for i in range(len(segments)) if i not in completed])
            return FragmentOutcome(validate_feedback(feedback), submitted, confirmed)

        for index, segment in enumerate(segments):
            skill, profile = segment["skill"], segment["profile"]
            entry = _pose(sample, arm)
            target = entry.copy()
            if skill in _MOTION:
                absolute = skill == "move_to_pose" or (direct and segment["position_mode"] == "absolute")
                with np.errstate(over="ignore", invalid="ignore"):
                    target[:3] = segment["xyz_m"] if absolute else entry[:3] + segment["xyz_m"]
                if segment["quaternion_wxyz"] is not None:
                    target[3:] = segment["quaternion_wxyz"]
            distance_limit = (self._limits.direct_translation_m if direct else
                              getattr(self._limits, f"{profile}_translation_m"))
            angle_limit = (self._limits.direct_rotation_rad if direct else
                           getattr(self._limits, f"{profile}_rotation_rad"))
            last = {"entry_tool_pose_wxyz": entry.tolist(), "last_tool_pose_wxyz": entry.tolist()}
            if not np.isfinite(target).all():
                return outcome("failed", "target_envelope_rejected")
            last["target_tool_pose_wxyz"] = target.tolist()
            entry_distance = math.dist(target[:3], entry[:3])
            entry_angle = float(np.linalg.norm(orientation_error_world(entry[3:], target[3:])))
            if (not math.isfinite(entry_distance) or entry_distance > distance_limit + 1e-12
                    or entry_angle > angle_limit + 1e-12):
                return outcome("failed", "target_envelope_rejected")

            def errors():
                pose = _pose(sample, arm)
                distance = math.dist(target[:3], pose[:3])
                angle = float(np.linalg.norm(orientation_error_world(pose[3:], target[3:])))
                if not math.isfinite(distance):
                    raise FragmentExecutionError("measurement_unavailable")
                last.update(last_tool_pose_wxyz=pose.tolist(), position_error_m=distance,
                            orientation_error_rad=angle,
                            finger_gap_m=float(sum(sample.finger_position_m[arm])))
                return distance, angle

            def arrived(distance, angle):
                return (distance <= settings["position_tolerance_m"][profile]
                        and angle <= settings["orientation_tolerance_rad"][profile])

            distance, angle = errors()
            arrival_count = int(arrived(distance, angle))
            gap_floor = last["finger_gap_m"]
            gap_rise_count = 0
            gap_history, error_history, measurements = [], [], []
            for step in range(1, segment["max_steps"] + 1):
                stop = self._stop_requested()
                if type(stop) is not bool:
                    raise FragmentExecutionError("invalid_stop_request")
                if stop and self.lifecycle.state == "RUNNING":
                    self.lifecycle.stop("user_stop")
                if self.lifecycle.state != "RUNNING":
                    return outcome("blocked", "interrupted")
                grip = segment["gripper"]
                updates = {} if grip == "keep" else {arm: _GRIP[grip]}
                try:
                    command = self._controller.command(sample.arms, target[:3], target[3:],
                                                        profile=profile, gripper_commands=updates)
                except ControllerError as exc:
                    reason = "joint_limit" if str(exc) in ("no_joint_motion_available", "inactive_hold_outside_live_limits") else "ik_unavailable"
                    return outcome("blocked", reason)
                if self.control_backend == OFFICIAL_EE:
                    action = command
                else:
                    action = {}
                    for name in ARMS:
                        action[f"{name}_arm_joint_state"] = list(command.joint_positions[name])
                        action[f"{name}_ee_joint_state"] = [command.gripper_commands[name]]
                ticket = self.lifecycle.reserve_action()
                submitted += 1
                if grip != "keep":
                    self._grippers[arm] = grip
                try:
                    receipt = self._send(ticket, action)
                    if not isinstance(receipt, ActionReceipt):
                        raise ValueError("Invalid receipt")
                    self.lifecycle.complete_action(ticket, receipt.native_result)
                    confirmed += 1
                    sample = _measurement(receipt.measurement)
                    ik_status = (validate_ik_status(receipt.ik_status)
                                 if self.control_backend == OFFICIAL_EE else None)
                except Exception:
                    raise FragmentExecutionError("native_action_or_measurement_unavailable") from None
                distance, angle = errors()
                measurements.append({"tool_pose_wxyz": last["last_tool_pose_wxyz"],
                    "position_error_m": distance, "orientation_error_rad": angle,
                    "finger_gap_m": last["finger_gap_m"],
                    "finger_position_m": list(sample.finger_position_m[arm])})
                last["measurements"] = measurements[-32:]
                if self.lifecycle.state != "RUNNING":
                    return outcome("blocked", "interrupted")
                # Native IK failure has already consumed this action and may
                # have applied the gripper or retained prior joint commands.
                # Report it only after the actual dispatch/measurement; do not
                # replace native behavior, claim zero movement or retry.
                if ik_status is not None and "Fail" in ik_status.values():
                    return outcome("blocked", "ik_unavailable")
                arrival_count = arrival_count + 1 if arrived(distance, angle) else 0
                gap = last["finger_gap_m"]
                gap_history.append(gap)
                error_history.append(distance + settings["orientation_error_weight_m_per_rad"] * angle)
                gap_floor = min(gap_floor, gap)
                if skill != "set_gripper" and self._grippers[arm] == "close":
                    gap_rise_count = gap_rise_count + 1 if gap - gap_floor >= settings["gap_rise_yield_m"] else 0
                    if gap_rise_count >= settings["gap_rise_samples"]:
                        return outcome("failed", "maintained_grip_gap_changed")
                pose_arrived = arrival_count >= settings["arrival_stable_samples"]
                complete = False
                if direct:
                    if pose_arrived or step == segment["max_steps"]:
                        return outcome("succeeded" if pose_arrived else "pending_verification",
                                       "pose_arrived" if pose_arrived else "segment_budget_exhausted")
                elif skill in _MOTION:
                    complete = pose_arrived
                elif skill == "set_gripper":
                    gaps = gap_history[-settings["gap_stable_samples"]:]
                    stable = (len(gaps) == settings["gap_stable_samples"]
                              and max(gaps) - min(gaps) <= settings["gap_stable_range_m"])
                    complete = (step >= getattr(self._limits, f"min_{grip}_steps") and stable and pose_arrived)
                elif skill == "wait":
                    complete = step == segment["max_steps"]
                if complete:
                    completed.append(index)
                    if index == len(segments) - 1:
                        return outcome("succeeded", "fragment_complete")
                    if segment["observation_after"] or skill == "wait":
                        return outcome("succeeded", "observation_required")
                    break
                if step == segment["max_steps"]:
                    return outcome("failed", "segment_budget_exhausted")
                if skill in _MOTION and step >= settings["stall_min_steps"]:
                    window = error_history[-settings["stall_window_steps"]:]
                    if window[0] - min(window[1:]) < settings["stall_progress_m"]:
                        return outcome("failed", "segment_stall")
        raise FragmentExecutionError("unexpected_fragment_state")
