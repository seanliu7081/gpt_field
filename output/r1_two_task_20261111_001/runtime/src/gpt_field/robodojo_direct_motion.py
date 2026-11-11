"""Absolute pose tool executed through unchanged 20 Hz LIBERO OSC_POSE.

The model selects only a world grip-site pose and optional aperture. A static
Panda-only IK model checks the straight Cartesian / shortest-rotation path before
any action. Measured-pose tracking then supplies bounded OSC deltas. There is no
scene collision planner, visual servo, grasp test, task skill or automatic retry.
This preserves LIBERO physics and is explicitly not CuRobo joint-path execution.

RoboDojo's angle names are nonstandard: R = Rz(yaw) Ry(roll) Rx(pitch) Rx(pi),
where all three inputs are degrees and zero is a downward Panda hand. Omitted
pose components retain their measured call-start value. Aperture 0 means closed,
1 means open, and its new command is applied only after measured arm arrival.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import numpy as np

from .action import ActionAdapter
from .robodojo_direct_sim import PandaKinematics, matrix_quaternion, rotation_matrix, rotation_vector, vector

MOTION_VERSION = "robodojo-direct-absolute-osc-1"
CONTROL_HZ = 20
POSE_KEYS = ("x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg", "gripper")
DOWNWARD = rotation_matrix([math.pi, 0., 0.])


def orientation_matrix(*, roll_deg=0., pitch_deg=0., yaw_deg=0.):
    """Match RoboDojo pitch-about-X / roll-about-Y naming, with Panda reference."""
    radians = np.radians([pitch_deg, roll_deg, yaw_deg])
    return (rotation_matrix([0., 0., radians[2]]) @ rotation_matrix([0., radians[1], 0.])
            @ rotation_matrix([radians[0], 0., 0.]) @ DOWNWARD)


def robot_state(state8):
    """Only measured robot proprioception; no evaluator/scene state accepted."""
    state = vector(state8, 8, "state8")
    relative = rotation_matrix(state[3:6]) @ DOWNWARD.T
    # Conventional XYZ decomposition, then relabel X=pitch and Y=roll.
    cosine_y = math.hypot(float(relative[0, 0]), float(relative[1, 0]))
    y = math.atan2(float(-relative[2, 0]), cosine_y)
    if cosine_y > 1e-8:
        x = math.atan2(relative[2, 1], relative[2, 2])
        z = math.atan2(relative[1, 0], relative[0, 0])
    else:
        x = math.atan2(-relative[1, 2], relative[1, 1])
        z = 0.
    return {"x": float(state[0]), "y": float(state[1]), "z": float(state[2]),
            "roll_deg": math.degrees(y), "pitch_deg": math.degrees(x), "yaw_deg": math.degrees(z),
            "gripper": float(np.clip((state[6]-state[7])/0.08, 0., 1.))}


def resolve_target(state8, arguments):
    if not isinstance(arguments, dict) or set(arguments)-set(POSE_KEYS)-{"note"}:
        raise ValueError("move_eef accepts only pose components, gripper and note")
    target = robot_state(state8)
    present = 0
    for key in POSE_KEYS:
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be a finite real number or null")
        if key == "gripper" and not 0 <= value <= 1:
            raise ValueError("gripper must be in [0,1]")
        target[key] = float(value)
        present += 1
    if not present:
        raise ValueError("move_eef needs at least one target component")
    note = arguments.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError("note must be text or null")
    return target


@dataclass(frozen=True)
class MotionConfig:
    translation_speed_m_s: float = 0.15
    rotation_speed_deg_s: float = 45.
    max_translation_command_m: float = 0.012
    max_rotation_command_deg: float = 4.
    position_tolerance_m: float = 0.004
    orientation_tolerance_deg: float = 2.
    stable_arrival_steps: int = 3
    tracking_extra_steps: int = 80
    max_arm_steps: int = 170
    gripper_steps: int = 30
    ik_waypoint_m: float = 0.02
    ik_waypoint_deg: float = 10.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("stable_arrival_steps", "tracking_extra_steps", "max_arm_steps", "gripper_steps"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if self.max_translation_command_m > .05 or self.max_rotation_command_deg > math.degrees(.5):
            raise ValueError("OSC command bounds exceed the unchanged physical interface")


def _cap_norm(v, maximum):
    norm = float(np.linalg.norm(v))
    return v if norm <= maximum else v * (maximum/norm)


class AbsoluteMotionExecutor:
    """One checked tool invocation, driven one observed physical step at a time.

    ``previous_gripper`` and ``current_gripper`` use aperture targets in [0,1],
    separately from the measured aperture. Retaining a closed target during a
    later arm move maintains squeeze even when an object prevents full closure.
    ``done`` may be true immediately on IK rejection; callers must not step.
    """

    def __init__(self, state8, arguments, joint_positions, kinematics: PandaKinematics,
                 *, config=None, previous_gripper=None):
        self.config = config or MotionConfig()
        self.start_state = vector(state8, 8, "state8").copy()
        self.target = resolve_target(state8, arguments)
        self.position = np.array([self.target[k] for k in ("x", "y", "z")])
        # Preserve exact measured orientation when all angles were omitted;
        # Euler decomposition/recomposition at singularities need not be exact.
        if all(arguments.get(k) is None for k in ("roll_deg", "pitch_deg", "yaw_deg")):
            self.orientation = rotation_matrix(self.start_state[3:6])
        else:
            self.orientation = orientation_matrix(**{k: self.target[k] for k in ("roll_deg", "pitch_deg", "yaw_deg")})
        self.start_orientation = rotation_matrix(self.start_state[3:6])
        self.rotation_delta = rotation_vector(self.orientation @ self.start_orientation.T)
        self.gripper_requested = arguments.get("gripper") is not None
        initial_gripper = robot_state(state8)["gripper"] if previous_gripper is None else previous_gripper
        if (isinstance(initial_gripper, bool) or not isinstance(initial_gripper, Real)
                or not math.isfinite(initial_gripper) or not 0 <= initial_gripper <= 1):
            raise ValueError("previous_gripper must be a finite aperture target in [0,1]")
        self.gripper_command = float(initial_gripper)
        if not self.gripper_requested:
            self.target["gripper"] = self.gripper_command
        self.gripper_applied = False
        self.executed_steps = self.arm_steps = self.finger_steps = self.stable_steps = 0
        self.done = False
        self.status = "incomplete"
        self.detail = "arm trajectory pending"
        self.phase = "arm"
        self.pending = None
        self.adapter = ActionAdapter(max_rotation_rad=math.radians(self.config.max_rotation_command_deg))
        distance = float(np.linalg.norm(self.position-self.start_state[:3]))
        angle = math.degrees(float(np.linalg.norm(self.rotation_delta)))
        self.trajectory_steps = max(1, math.ceil(distance/self.config.translation_speed_m_s*CONTROL_HZ),
                                    math.ceil(angle/self.config.rotation_speed_deg_s*CONTROL_HZ))
        self.arm_step_budget = min(self.config.max_arm_steps,
                                   self.trajectory_steps+self.config.tracking_extra_steps)
        self.preflight = {"method": "robot_only_DLS_IK_along_cartesian_path", "collision_checking": False,
                          "checked_waypoints": 0, "planned_joint_waypoints": []}
        if self.trajectory_steps > self.config.max_arm_steps:
            self._finish("rejected", "requested trajectory exceeds per-motion execution budget; no action executed")
            return
        # Kinematics reads only fixed ROBOT assets plus measured robot state.
        model = kinematics.bind(state8, joint_positions)
        q = vector(joint_positions, 7, "joint_positions")
        count = max(1, math.ceil(distance/self.config.ik_waypoint_m), math.ceil(angle/self.config.ik_waypoint_deg))
        for i in range(1, count+1):
            fraction = i/count
            p = self.start_state[:3] + fraction*(self.position-self.start_state[:3])
            R = rotation_matrix(fraction*self.rotation_delta) @ self.start_orientation
            result = model.solve(p, R, q)
            self.preflight["checked_waypoints"] += 1
            if not result.ok:
                self.preflight["failed_waypoint"] = i
                self.preflight["ik_position_error_m"] = result.position_error_m
                self.preflight["ik_orientation_error_deg"] = result.orientation_error_deg
                self._finish("rejected", f"no robot-only IK solution found at path waypoint {i}/{count}; "
                             "no arm or gripper action executed; collision safety was not evaluated")
                return
            q = np.asarray(result.joints)
            self.preflight["planned_joint_waypoints"].append(q.tolist())

    @property
    def current_gripper(self):
        return self.gripper_command

    def _finish(self, status, detail):
        self.done, self.status, self.detail, self.phase = True, status, detail, "done"

    def _errors(self, state):
        return (float(np.linalg.norm(self.position-state[:3])),
                math.degrees(float(np.linalg.norm(rotation_vector(self.orientation @ rotation_matrix(state[3:6]).T)))))

    def next_action(self, state8):
        if self.done:
            raise RuntimeError("motion is already finished")
        if self.pending is not None:
            raise RuntimeError("observe_transition must acknowledge the preceding action")
        state = vector(state8, 8, "state8")
        if self.phase == "arm":
            fraction = min(1., (self.arm_steps+1)/self.trajectory_steps)
            desired_position = self.start_state[:3] + fraction*(self.position-self.start_state[:3])
            desired_rotation = rotation_matrix(fraction*self.rotation_delta) @ self.start_orientation
        else:
            desired_position, desired_rotation = self.position, self.orientation
        delta = _cap_norm(desired_position-state[:3], self.config.max_translation_command_m)
        action = list(self.adapter.adapt(delta, matrix_quaternion(desired_rotation), "hold", state))
        # The unchanged OSC controller otherwise retains a stale orientation goal
        # when given exactly zero rotation; an infinitesimal refresh rebases it.
        if not any(action[3:6]):
            action[3] = 2e-9
        measured_gap = float(state[6]-state[7])
        desired_gap = .08*self.gripper_command
        if self.gripper_command == 0.:
            action[6] = 1.
        elif self.gripper_command == 1.:
            action[6] = -1.
        else:
            # Generic aperture actuator servo; no contact/grasp classifier.
            action[6] = float(np.clip((measured_gap-desired_gap)/.004, -1., 1.))
        action = tuple(float(v) for v in action)
        self.pending = (state.copy(), action, self.phase)
        return action

    def observe_transition(self, before_state8, action, after_state8):
        if self.pending is None:
            raise RuntimeError("no action awaits acknowledgement")
        before = vector(before_state8, 8, "before_state8")
        after = vector(after_state8, 8, "after_state8")
        issued_state, issued_action, issued_phase = self.pending
        if not np.allclose(before, issued_state, atol=1e-9, rtol=0.) or not np.allclose(
                vector(action, 7, "action"), issued_action, atol=1e-12, rtol=0.):
            raise ValueError("acknowledgement does not match the issued state and action")
        self.pending = None
        self.executed_steps += 1
        pe, re = self._errors(after)
        arrived = pe <= self.config.position_tolerance_m and re <= self.config.orientation_tolerance_deg
        if issued_phase == "arm":
            self.arm_steps += 1
            self.stable_steps = self.stable_steps+1 if arrived else 0
            if self.stable_steps >= self.config.stable_arrival_steps:
                if self.gripper_requested:
                    self.phase = "gripper"
                    self.gripper_command = self.target["gripper"]
                else:
                    self._finish("reached", "arm reached measured tolerance; no new gripper target requested; "
                                 "no task or grasp success is inferred")
            elif self.arm_steps >= self.arm_step_budget:
                self._finish("incomplete", "arm tracking budget exhausted before measured arrival; "
                             "new gripper target skipped; inspect actual pose and choose the next motion")
        else:
            self.gripper_applied = True
            self.finger_steps += 1
            if self.finger_steps >= self.config.gripper_steps:
                self._finish("reached" if arrived else "incomplete", "gripper command applied after arm arrival; "
                             "aperture may be constrained by contact; no task or grasp success is inferred" +
                             ("; arm drifted outside arrival tolerance" if not arrived else ""))
        return {"phase": issued_phase, "position_error_m": pe, "orientation_error_deg": re,
                "executed_steps": self.executed_steps, "done": self.done}

    def feedback(self, state8):
        state = vector(state8, 8, "state8")
        measured = robot_state(state)
        pe, re = self._errors(state)
        return {"status": self.status, "requested_target": dict(self.target), "measured_state": measured,
                "position_error_m": pe, "orientation_error_deg": re, "executed_steps": self.executed_steps,
                "detail": self.detail, "gripper_applied": self.gripper_applied,
                "gripper_error": abs(measured["gripper"]-self.gripper_command)}

    def interrupt(self, detail="user stopped during motion"):
        self.pending = None
        self._finish("user_stop", detail)
