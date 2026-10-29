"""Measured, task-independent action fragments for the user-check executive.

Targets are supplied by the model in the world grip_site/right_hand convention.
Each control step uses robot proprioception and the existing OSC action adapter.
There is no scene oracle, grasp certificate, collision planner, implicit retry,
release, lift, or claim of visual/task success in this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .action import ActionAdapter
from .astranomy_agentic_motion import measured_pose, rotation_error_rad
from .contact_skills import Primitive, StepView

FRAGMENT_VERSION = "user-check-measured-fragments-1"
MAX_SEGMENTS = 6
MAX_SEGMENT_STEPS = 120
MAX_EXECUTION_STEPS = 240
PROFILES = {
    "transit": {"target_translation_m": .30, "target_rotation_rad": 1.2,
                "action_translation_m": .02, "position_tolerance_m": .004,
                "rotation_tolerance_rad": .06},
    "contact": {"target_translation_m": .08, "target_rotation_rad": .5,
                "action_translation_m": .005, "position_tolerance_m": .002,
                "rotation_tolerance_rad": .04},
}
MOTION_SKILLS = ("move_to_pose", "move_relative", "approach")
SKILLS = (*MOTION_SKILLS, "set_gripper", "wait")
MIN_GRIPPER_STEPS = {"close": 12, "open": 8}
ARRIVAL_STABLE_SAMPLES = 2
GAP_STABLE_SAMPLES = 5
GAP_STABLE_RANGE_MM = .6
GAP_RISE_YIELD_MM = 6.
GAP_RISE_SAMPLES = 3
STALL_WINDOW_STEPS = 12
STALL_MIN_STEPS = 16
STALL_PROGRESS_M = .0008


def execution_schema() -> dict:
    number = {"type": "number"}
    def nullable_vector(size):
        return {"anyOf": [{"type": "array", "items": number,
                           "minItems": size, "maxItems": size}, {"type": "null"}]}
    fields = {
        "skill": {"type": "string", "enum": list(SKILLS)},
        "xyz_m": nullable_vector(3),
        "quaternion_wxyz": nullable_vector(4),
        "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        "profile": {"type": "string", "enum": list(PROFILES)},
        "max_steps": {"type": "integer", "minimum": 1, "maximum": MAX_SEGMENT_STEPS},
        "observation_after": {"type": "boolean"},
    }
    return {"type": "object", "properties": {
        "segments": {"type": "array", "minItems": 1, "maxItems": MAX_SEGMENTS,
                     "items": {"type": "object", "properties": fields,
                               "required": list(fields), "additionalProperties": False}}},
            "required": ["segments"], "additionalProperties": False}


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
        or not math.isfinite(float(v)) for v in value
    ):
        raise ValueError(f"{name} must contain {size} finite real numbers")
    return [float(v) for v in value]


def _norm(vector):
    return math.hypot(*(float(v) for v in vector))


def validate_execution(value) -> dict:
    """Check every field combination before execution, returning a detached copy.

    Relative translation lengths can be checked here. Absolute distances and
    rotation bounds are checked at each segment's measured entry pose.
    """
    if not isinstance(value, Mapping) or set(value) != {"segments"}:
        raise ValueError("execution must contain only segments")
    segments = value["segments"]
    if not isinstance(segments, (list, tuple)) or not 1 <= len(segments) <= MAX_SEGMENTS:
        raise ValueError(f"execution requires 1 through {MAX_SEGMENTS} segments")
    required = set(execution_schema()["properties"]["segments"]["items"]["required"])
    normalized = []
    for index, raw in enumerate(segments):
        prefix = f"execution.segments[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(f"{prefix}: fields must exactly match the segment schema")
        segment = deepcopy(dict(raw))
        skill, profile, grip = segment["skill"], segment["profile"], segment["gripper"]
        if not isinstance(skill, str) or skill not in SKILLS:
            raise ValueError(f"{prefix}.skill: unsupported skill")
        if not isinstance(profile, str) or profile not in PROFILES:
            raise ValueError(f"{prefix}.profile: must be transit or contact")
        if not isinstance(grip, str) or grip not in ("keep", "open", "close"):
            raise ValueError(f"{prefix}.gripper: must be keep, open, or close")
        if type(segment["max_steps"]) is not int or not 1 <= segment["max_steps"] <= MAX_SEGMENT_STEPS:
            raise ValueError(f"{prefix}.max_steps: must be an integer from 1 through {MAX_SEGMENT_STEPS}")
        if type(segment["observation_after"]) is not bool:
            raise ValueError(f"{prefix}.observation_after: must be boolean")
        if segment["xyz_m"] is not None:
            segment["xyz_m"] = _finite_vector(segment["xyz_m"], 3, prefix + ".xyz_m")
        if segment["quaternion_wxyz"] is not None:
            q = np.asarray(_finite_vector(segment["quaternion_wxyz"], 4,
                                           prefix + ".quaternion_wxyz"))
            scale = float(np.max(np.abs(q)))
            if scale == 0:
                raise ValueError(f"{prefix}.quaternion_wxyz: quaternion must be nonzero")
            q = q / scale
            segment["quaternion_wxyz"] = (q / _norm(q)).tolist()
        if skill in MOTION_SKILLS:
            if segment["xyz_m"] is None or grip != "keep":
                raise ValueError(f"{prefix}: arm motion requires xyz_m and gripper keep")
            if skill == "approach" and profile != "contact":
                raise ValueError(f"{prefix}: approach requires profile contact")
            if skill != "move_to_pose" and _norm(segment["xyz_m"]) > PROFILES[profile]["target_translation_m"] + 1e-12:
                raise ValueError(f"{prefix}: relative translation exceeds the {profile} envelope")
        else:
            if segment["xyz_m"] is not None or segment["quaternion_wxyz"] is not None:
                raise ValueError(f"{prefix}: {skill} requires null xyz_m and quaternion_wxyz")
            if skill == "set_gripper":
                if grip not in MIN_GRIPPER_STEPS:
                    raise ValueError(f"{prefix}: set_gripper requires open or close")
                if segment["max_steps"] < MIN_GRIPPER_STEPS[grip]:
                    raise ValueError(f"{prefix}: {grip} requires max_steps >= {MIN_GRIPPER_STEPS[grip]}")
            elif grip != "keep":
                raise ValueError(f"{prefix}: wait requires gripper keep")
        normalized.append(segment)
    if sum(segment["max_steps"] for segment in normalized) > MAX_EXECUTION_STEPS:
        raise ValueError(f"execution total max_steps must not exceed {MAX_EXECUTION_STEPS}")
    return {"segments": normalized}


def fragment_capabilities() -> dict:
    return {
        "version": FRAGMENT_VERSION, "frame": "world", "control_hz": 20,
        "position_reference": "robot grip_site", "orientation_reference": "robot right_hand",
        "skills": list(SKILLS), "profiles": deepcopy(PROFILES),
        "max_action_rotation_rad": .1, "max_segments": MAX_SEGMENTS,
        "max_segment_steps": MAX_SEGMENT_STEPS, "max_execution_steps": MAX_EXECUTION_STEPS,
        "minimum_gripper_steps": deepcopy(MIN_GRIPPER_STEPS),
        "arrival_stable_samples": ARRIVAL_STABLE_SAMPLES,
        "semantics": {
            "move_to_pose": "absolute world xyz_m and optional absolute world quaternion_wxyz",
            "move_relative": "world xyz_m displacement resolved once from measured segment entry pose",
            "approach": "world xyz_m displacement from measured entry, contact profile; stall is not a contact certificate",
            "orientation_null": "retain measured segment entry orientation",
            "gripper": "motion keeps the actual command; use separate set_gripper with null pose fields",
            "set_gripper": "hold entry pose, apply command for at least the minimum, require stable measured gap; no grasp claim",
            "wait": "hold entry pose and command for max_steps, then return for observation",
            "completion": "only measured pose arrival or measured gap stability; object/task success remains unknown",
            "observation_after": "yield after this segment; pending segments do not autoplay or resume after intervention",
            "continuation": "submit a new execution from the current observation; relative targets use new measured entry poses",
            "interruption": "stall, target envelope rejection, budget, or sustained gap rise returns control; no implicit retry",
            "preparation": "uses current robot proprioception and known OSC conversion; no scene registration or calibration sweep",
        },
        "unknowns": ["object identity", "object pose", "grasp success", "collision-free path", "task success"],
    }


class FragmentSequencePrimitive(Primitive):
    """Run a bounded sequence until measured completion or the first handoff.

    Full field validation precedes any action. Geometric bounds are additionally
    enforced from measured entry state for every segment, including later ones.
    """

    def __init__(self, view: StepView, execution, keep_gripper="open"):
        super().__init__("execute_fragment")
        if not isinstance(view, StepView):
            raise TypeError("fragment execution requires a StepView snapshot")
        self.execution = validate_execution(execution)
        if keep_gripper not in ("open", "close"):
            raise ValueError("keep_gripper requires the actual open/close runtime command")
        self.gripper_command = keep_gripper
        self.commanded_reorientation = False
        self.adapter = ActionAdapter(max_rotation_rad=.1)
        self._last_state = view.state.copy()
        self._last_pose = measured_pose(view.state)
        self._last_gap_mm = float(view.gap_mm)
        self._segment_index = 0
        self._records = []
        self._current = None
        self._target_xyz = None
        self._target_wxyz = None
        self._arrival_count = 0
        self._gap_rise_count = 0
        self._gap_floor_mm = float(view.gap_mm)
        self._enter_segment(view)

    def _errors(self, view):
        return (_norm(self._target_xyz - view.position),
                _norm(rotation_error_rad(self._target_wxyz, view.state[3:6])))

    def _arrived(self, position_error, orientation_error):
        profile = PROFILES[self._current["request"]["profile"]]
        return (position_error <= profile["position_tolerance_m"] and
                orientation_error <= profile["rotation_tolerance_rad"])

    def _enter_segment(self, view):
        segment = self.execution["segments"][self._segment_index]
        pose = measured_pose(view.state)
        self._target_xyz = np.asarray(pose[:3], dtype=float)
        if segment["skill"] in MOTION_SKILLS:
            value = np.asarray(segment["xyz_m"], dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                self._target_xyz = value if segment["skill"] == "move_to_pose" else self._target_xyz + value
        self._target_wxyz = list(pose[3:7] if segment["quaternion_wxyz"] is None
                                 else segment["quaternion_wxyz"])
        self._current = {
            "index": self._segment_index, "skill": segment["skill"], "request": deepcopy(segment),
            "status": "running", "reason": None, "entry_at_fragment_step": self.actions_executed,
            "entry_pose_wxyz": pose, "entry_gap_mm": float(view.gap_mm),
            "target_xyz_m": self._target_xyz.tolist(), "target_quaternion_wxyz": self._target_wxyz,
            "requested_gripper": segment["gripper"], "actions": 0, "measurements": [],
            "object_grasp_verified": False,
        }
        self._records.append(self._current)
        self._arrival_count = 0
        self._gap_rise_count = 0
        self._gap_floor_mm = float(view.gap_mm)
        if not np.isfinite(self._target_xyz).all():
            self._end("failed", "target_envelope_rejected", detail="target arithmetic is not finite")
            return
        distance, angle = self._errors(view)
        profile = PROFILES[segment["profile"]]
        self._current.update(position_error_m=distance, orientation_error_rad=angle)
        if distance > profile["target_translation_m"] + 1e-12 or angle > profile["target_rotation_rad"] + 1e-12:
            self._end("failed", "target_envelope_rejected",
                      detail="target exceeds the profile envelope from measured segment entry; this segment issued no action")
            return
        self.commanded_reorientation |= angle > 1e-9
        # This initial measurement can count toward arrival, but completion is
        # observed after an actual acknowledged action, never inside _next.
        if segment["skill"] in MOTION_SKILLS and self._arrived(distance, angle):
            self._arrival_count = 1

    def _next(self, view):
        if not np.array_equal(view.state, self._last_state):
            raise ValueError("fragment action state differs from the latest acknowledged measurement")
        segment = self._current["request"]
        if segment["skill"] == "set_gripper":
            self.gripper_command = segment["gripper"]
        delta = self._target_xyz - view.position
        length = _norm(delta)
        limit = PROFILES[segment["profile"]]["action_translation_m"]
        if length > limit:
            delta = delta * (limit / length)
        q = self._target_wxyz
        return self.adapter.adapt(delta, [*q[1:], q[0]], self.gripper_command, view.state)

    def _observe(self, before, action, after):
        self._last_state = after.state.copy()
        self._last_pose = measured_pose(after.state)
        self._last_gap_mm = float(after.gap_mm)
        record = self._current
        segment = record["request"]
        record["actions"] += 1
        distance, angle = self._errors(after)
        record.update(position_error_m=distance, orientation_error_rad=angle,
                      last_pose_wxyz=self._last_pose, last_gap_mm=self._last_gap_mm)
        sample = {"at_fragment_step": self.actions_executed, "xyz_m": self._last_pose[:3],
                  "quaternion_wxyz": self._last_pose[3:7], "position_error_m": distance,
                  "orientation_error_rad": angle, "gap_mm": self._last_gap_mm,
                  "gripper_command": self.gripper_command}
        record["measurements"].append(sample)
        arrived = self._arrived(distance, angle)
        self._arrival_count = self._arrival_count + 1 if arrived else 0
        self._gap_floor_mm = min(self._gap_floor_mm, self._last_gap_mm)
        if segment["skill"] != "set_gripper" and self.gripper_command == "close":
            self._gap_rise_count = self._gap_rise_count + 1 if self._last_gap_mm - self._gap_floor_mm >= GAP_RISE_YIELD_MM else 0
            if self._gap_rise_count >= GAP_RISE_SAMPLES:
                record["gap_change_evidence"] = {
                    "minimum_gap_mm": self._gap_floor_mm,
                    "current_gap_mm": self._last_gap_mm,
                    "rise_mm": self._last_gap_mm - self._gap_floor_mm,
                    "consecutive_samples": self._gap_rise_count,
                }
                self._end("failed", "maintained_grip_gap_changed",
                          detail="measured finger gap rose while close was maintained; cause and object holding remain unknown")
                return {"fragment": self.progress()}
        complete_reason = None
        if segment["skill"] in MOTION_SKILLS:
            if self._arrival_count >= ARRIVAL_STABLE_SAMPLES:
                complete_reason = "pose_arrived"
        elif segment["skill"] == "set_gripper":
            recent = record["measurements"][-GAP_STABLE_SAMPLES:]
            gaps = [item["gap_mm"] for item in recent]
            gap_stable = len(gaps) == GAP_STABLE_SAMPLES and max(gaps) - min(gaps) <= GAP_STABLE_RANGE_MM
            record["gap_stable"] = gap_stable
            if (record["actions"] >= MIN_GRIPPER_STEPS[self.gripper_command] and gap_stable
                    and self._arrival_count >= ARRIVAL_STABLE_SAMPLES):
                complete_reason = "gap_stable"
        elif record["actions"] >= segment["max_steps"]:
            complete_reason = "wait_elapsed"
        if complete_reason:
            self._complete_segment(after, complete_reason)
        elif record["actions"] >= segment["max_steps"]:
            self._end("failed", "segment_budget_exhausted",
                      detail="requested control budget ended before the measured completion predicate")
        elif self.actions_executed >= MAX_EXECUTION_STEPS:
            self._end("failed", "execution_budget_exhausted")
        elif segment["skill"] in MOTION_SKILLS and record["actions"] >= STALL_MIN_STEPS:
            window = record["measurements"][-STALL_WINDOW_STEPS:]
            errors = [item["position_error_m"] + .05 * item["orientation_error_rad"] for item in window]
            if errors[0] - min(errors[1:]) < STALL_PROGRESS_M:
                record["stall_evidence"] = {
                    "window_steps": len(window),
                    "position_error_start_m": window[0]["position_error_m"],
                    "position_error_end_m": window[-1]["position_error_m"],
                    "orientation_error_start_rad": window[0]["orientation_error_rad"],
                    "orientation_error_end_rad": window[-1]["orientation_error_rad"],
                    "weighted_error_improvement_m": errors[0] - min(errors[1:]),
                    "weighted_error_definition": "position_error_m + 0.05 * orientation_error_rad",
                }
                self._end("failed", "segment_stall",
                          detail="pose error did not measurably improve over the local window; obstruction/contact cause is unknown")
        return {"fragment": self.progress()}

    def _complete_segment(self, view, reason):
        self._current.update(status="succeeded", reason=reason,
                             end_at_fragment_step=self.actions_executed)
        segment = self._current["request"]
        pending = self._segment_index + 1 < len(self.execution["segments"])
        if not pending:
            self._end("succeeded", "fragment_complete", finalize_current=False)
        elif segment["observation_after"] or segment["skill"] == "wait":
            self._end("succeeded", "observation_required", finalize_current=False,
                      detail="completed the current predicate; remaining segments were not executed and require a new model decision")
        else:
            self._segment_index += 1
            self._enter_segment(view)

    def _end(self, status, reason, *, detail=None, finalize_current=True):
        if finalize_current:
            self._current.update(status=status, reason=reason,
                                 end_at_fragment_step=self.actions_executed)
        if detail:
            self._current["detail"] = detail
        # Set status before collecting evidence so progress reflects the handoff.
        self._finish(status, reason, {})
        evidence = self.progress()
        evidence["segment_trace"] = deepcopy(self._records)
        evidence["detail"] = detail
        self._evidence = evidence

    def progress(self) -> dict:
        completed = [r["index"] for r in self._records if r["status"] == "succeeded"]
        pending_start = self._segment_index + 1
        current = self._current
        summaries = [{key: deepcopy(value) for key, value in record.items() if key != "measurements"}
                     for record in self._records]
        return {
            "version": FRAGMENT_VERSION,
            "status": self.status or "running", "reason": self.reason,
            "observed_at_fragment_step": self.actions_executed,
            "executed_steps": self.actions_executed,
            "budget_steps": sum(s["max_steps"] for s in self.execution["segments"]),
            "measured_pose_wxyz": deepcopy(self._last_pose), "measured_gap_mm": self._last_gap_mm,
            "gripper_command": self.gripper_command,
            "current_segment_index": self._segment_index,
            "current_skill": current["skill"], "current_segment_status": current["status"],
            "current_segment_actions": current["actions"],
            "target_xyz_m": deepcopy(current["target_xyz_m"]),
            "target_quaternion_wxyz": deepcopy(current["target_quaternion_wxyz"]),
            "position_error_m": current.get("position_error_m"),
            "orientation_error_rad": current.get("orientation_error_rad"),
            "completed_segment_indices": completed,
            "sequence_complete": len(completed) == len(self.execution["segments"]),
            "handoff_kind": ("completed" if self.reason == "fragment_complete" else
                             "observation_checkpoint" if self.reason == "observation_required" else
                             "budget" if self.reason in ("segment_budget_exhausted", "execution_budget_exhausted") else
                             "blocked_or_uncertain" if self.done else "executing"),
            "pending_segment_indices": list(range(pending_start, len(self.execution["segments"]))),
            "pending_segments": deepcopy(self.execution["segments"][pending_start:]),
            "segments": summaries,
            "observation_requested": bool(self.done and (self.reason == "observation_required"
                or current["request"]["observation_after"] or current["skill"] == "wait")),
            "grasp_success": "unknown", "task_success": "unknown",
            "contact_or_collision_cause": "unknown", "automatic_resume": False,
        }
