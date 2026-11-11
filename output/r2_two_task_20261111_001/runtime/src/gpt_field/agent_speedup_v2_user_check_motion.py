"""Task-independent short motion proposals for the user-check variant only.

The model selects world-frame targets from current images and proprioception.
The unchanged Astranomy executor supplies the 1--5-step measured-error control;
this module adds no scene geometry, object names, paths or task success claims.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .agent_speedup_v2_plan import (
    PlanParseError, conform, parse_repair, repair_schema, sanitize_text,
)
from .astranomy_agentic_motion import (
    ShortTargetConfig, ShortTargetExecutor, TargetValidationError,
    measured_pose, rejected_feedback, rotation_error_rad,
)
from .contact_skills import Primitive, StepView

LOCAL_MOTION_VERSION = "user-check-local-motion-2"

REVIEWED_RECOVERY_INSTRUCTIONS = """
Local recovery has an explicit measured subgoal and a separate macro handoff.
local_motion.step.recovery_goal is null to keep the active subgoal (or use the
segment's absolute target when starting), or an object with kind pose/height/
open_hand/observe, xyz_m, height_m and tolerance_m. Set unused fields null.
pose uses an absolute world grip_site xyz; height requires world z >= height_m;
tolerance_m is a positive metric tolerance <= 0.02. observe requests a new view,
not task success. Short segments do not establish arrival or a grasp. Look at
the measured subgoal verdict and current images before choosing the next action.
Use verify_grasp with part=object ID and step={grip:grip-part ID} after local
closing. The executor verifies identity by measured object co-motion, load and
held-frame geometry; a model claim or finger closure cannot create a held state.
Use resume_macro with part=null,step=null only after the subgoal is met and the
handoff has no blockers. The current skill recomputes its targets using measured
state; an already verified grasp is reused. At a skill checkpoint, inspect the
images and checkpoint evidence, then resume_macro to continue or local_motion
to correct the pose. Repoint/replan/insert_step requests exit the checkpoint and
run through the ordinary repair executor. A final visual correction is still
subject to measured whole-object placement verification.
""".strip()


def recovery_goal_schema():
    number = {"type": "number"}
    nullable_number = {"anyOf": [number, {"type": "null"}]}
    properties = {
        "kind": {"type": "string", "enum": ["pose", "height", "open_hand", "observe"]},
        "xyz_m": {"anyOf": [{"type": "array", "items": number, "minItems": 3, "maxItems": 3},
                              {"type": "null"}]},
        "height_m": nullable_number,
        "tolerance_m": nullable_number,
    }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def local_motion_schema() -> dict:
    """All motion fields are explicit; null orientation retains the current one."""
    number = {"type": "number"}
    properties = {
        "frame": {"type": "string", "enum": ["world"]},
        "position_mode": {"type": "string", "enum": ["absolute", "relative"]},
        "xyz_m": {"type": "array", "items": number, "minItems": 3, "maxItems": 3},
        "quaternion_wxyz": {"anyOf": [
            {"type": "array", "items": number, "minItems": 4, "maxItems": 4},
            {"type": "null"},
        ]},
        "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        "steps": {"type": "integer", "minimum": 1, "maximum": 5},
        "recovery_goal": {"anyOf": [recovery_goal_schema(), {"type": "null"}]},
    }
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def reviewed_repair_schema() -> dict:
    """Extend the repair/review wire format without changing the native schema."""
    schema = repair_schema()
    schema["properties"]["action"]["enum"].extend(["local_motion", "verify_grasp", "resume_macro"])
    schema["properties"]["step"]["anyOf"].append(local_motion_schema())
    schema["properties"]["step"]["anyOf"].append({
        "type": "object", "properties": {"grip": deepcopy(schema["properties"]["part"]["anyOf"][0])},
        "required": ["grip"], "additionalProperties": False})
    return schema


def _conform_motion(value: Mapping) -> None:
    schema = local_motion_schema()
    if not isinstance(value, Mapping):
        raise PlanParseError("$.step: local_motion requires a motion object")
    # Old audited segments remain readable; new structured outputs include the
    # nullable subgoal explicitly.
    if set(value) not in (set(schema["required"]), set(schema["required"]) - {"recovery_goal"}):
        raise PlanParseError("$.step: local_motion fields must match the motion schema")
    for name, field in schema["properties"].items():
        item = value.get(name)
        if name == "recovery_goal":
            _conform_goal(item)
            continue
        if name not in ("xyz_m", "quaternion_wxyz"):
            conform(item, field, f"$.step.{name}")
            continue
        if name == "quaternion_wxyz" and item is None:
            continue
        size = 3 if name == "xyz_m" else 4
        if not isinstance(item, (list, tuple)) or len(item) != size or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not math.isfinite(float(v)) for v in item
        ):
            raise PlanParseError(f"$.step.{name}: must contain {size} finite real numbers")
        if name == "quaternion_wxyz" and not any(float(v) for v in item):
            raise PlanParseError("$.step.quaternion_wxyz: quaternion must have nonzero norm")


def _conform_goal(goal):
    if goal is None:
        return
    fields = recovery_goal_schema()["properties"]
    if not isinstance(goal, Mapping) or set(goal) != set(fields):
        raise PlanParseError("$.step.recovery_goal: fields must match the recovery goal schema")
    conform(goal["kind"], fields["kind"], "$.step.recovery_goal.kind")
    xyz = goal["xyz_m"]
    if xyz is not None and (not isinstance(xyz, (list, tuple)) or len(xyz) != 3 or any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(float(x)) for x in xyz)):
        raise PlanParseError("$.step.recovery_goal.xyz_m: requires three finite numbers or null")
    for name in ("height_m", "tolerance_m"):
        value = goal[name]
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                                  or not math.isfinite(float(value))):
            raise PlanParseError(f"$.step.recovery_goal.{name}: requires a finite number or null")
    if goal["kind"] == "pose" and xyz is None:
        raise PlanParseError("$.step.recovery_goal: pose needs absolute world xyz_m")
    if goal["kind"] == "height" and goal["height_m"] is None:
        raise PlanParseError("$.step.recovery_goal: height needs absolute world height_m")
    tolerance = goal["tolerance_m"]
    if tolerance is not None and not 0 < tolerance <= .02:
        raise PlanParseError("$.step.recovery_goal.tolerance_m: must be positive and <= 0.02")


def conform_reviewed_repair(data: Mapping) -> None:
    """Validate numeric motion fields, which the original plan checker lacks.

    The wire union is additionally checked against its action discriminator.
    Native actions retain their original structural parser and sanitization.
    """
    if isinstance(data, Mapping) and data.get("action") in ("verify_grasp", "resume_macro"):
        outer = repair_schema()
        outer["properties"]["action"] = {"type": "string", "enum": [data["action"]]}
        if data["action"] == "resume_macro":
            outer["properties"]["part"] = {"type": "null"}
            outer["properties"]["step"] = {"type": "null"}
        else:
            pid = deepcopy(outer["properties"]["part"]["anyOf"][0])
            outer["properties"]["part"] = pid
            outer["properties"]["step"] = {"type": "object", "properties": {"grip": pid},
                                             "required": ["grip"], "additionalProperties": False}
        conform(data, outer)
        return
    if not isinstance(data, Mapping) or data.get("action") != "local_motion":
        parse_repair(data)
        return
    outer = repair_schema()
    outer["properties"]["action"] = {"type": "string", "enum": ["local_motion"]}
    outer["properties"]["part"] = {"type": "null"}
    outer["properties"]["step"] = {"type": "null"}
    conform({**data, "step": None}, outer)
    if "step" not in data:
        raise PlanParseError("$.step: local_motion requires a motion object")
    _conform_motion(data["step"])
    if not sanitize_text(data["why"], outer["properties"]["why"]["maxLength"]):
        raise PlanParseError("$.why: fewer than 1 readable ASCII characters")


@dataclass(frozen=True)
class LocalMotionAnswer:
    """Repair-answer shape with a numeric motion payload, never a PlanStep."""
    why: str
    action: str
    part: str | None
    step: dict

    def __post_init__(self):
        if self.action != "local_motion":
            raise PlanParseError("LocalMotionAnswer.action must be local_motion")
        conform_reviewed_repair(self.to_json())
        object.__setattr__(self, "step", deepcopy(self.step))

    def to_json(self) -> dict:
        return {"why": self.why, "action": self.action, "part": self.part,
                "step": deepcopy(self.step)}


@dataclass(frozen=True)
class HandoffAnswer:
    why: str
    action: str
    part: str | None
    step: dict | None

    def __post_init__(self):
        if self.action not in ("verify_grasp", "resume_macro"):
            raise PlanParseError("HandoffAnswer requires verify_grasp or resume_macro")
        conform_reviewed_repair(self.to_json())
        object.__setattr__(self, "step", deepcopy(self.step))

    def to_json(self):
        return {"why": self.why, "action": self.action, "part": self.part, "step": deepcopy(self.step)}


def parse_reviewed_repair(data: Mapping):
    """Return a native RepairAnswer or the isolated LocalMotionAnswer extension."""
    conform_reviewed_repair(data)
    if data["action"] in ("verify_grasp", "resume_macro"):
        why = sanitize_text(data["why"], repair_schema()["properties"]["why"]["maxLength"])
        return HandoffAnswer(why, data["action"], data["part"], data["step"])
    if data["action"] != "local_motion":
        return parse_repair(data)
    why = sanitize_text(data["why"], repair_schema()["properties"]["why"]["maxLength"])
    return LocalMotionAnswer(why, "local_motion", None, dict(data["step"]))


def motion_context(view: StepView, *, contract=False) -> dict:
    """Expose robot measurements and the public motion contract, without truth state."""
    pose = measured_pose(view.state)
    result = {
        "version": LOCAL_MOTION_VERSION,
        "frame": "world",
        "xyz_m": pose[:3],
        "quaternion_wxyz": pose[3:7],
        "measured_gripper_closure": pose[7],
    }
    if not contract:
        return result
    result.update({
        "position_reference": "robot grip_site",
        "orientation_reference": "robot right_hand in world, quaternion wxyz",
        "units": {"position": "metres", "rotation": "radians"},
        "limits": ShortTargetConfig().to_dict(),
        "semantics": {
            "relative": "xyz_m is one world-frame displacement from the segment's initial measured pose",
            "absolute": "xyz_m is one world-frame target; z is absolute world height",
            "orientation_null": "retain the segment's initial measured orientation",
            "gripper_keep": "retain the runtime gripper command, not inferred closure or grasp state",
            "completion": "selected steps executed; arrival, grasp and task success remain unverified",
            "feedback": "request a new segment only after its result and fresh images are available",
        },
    })
    return result


def compact_local_feedback(feedback):
    """Single model-facing motion result; detailed evidence remains in audit."""
    if not feedback:
        return None
    keys = ("action", "status", "reason", "observed_at_step", "executed_steps", "measured_translation_m",
            "state_changed", "subgoal", "grasp_effect", "task_effect", "handoff", "cause")
    return {key: deepcopy(feedback[key]) for key in keys if key in feedback}


class LocalMotionPrimitive(Primitive):
    """Adapt Astranomy's bounded, measured-error executor to the v2 protocol."""

    def __init__(self, view: StepView, motion: Mapping, *, keep_gripper=None):
        super().__init__("local_motion")
        if not isinstance(view, StepView):
            raise TypeError("local motion requires a StepView snapshot")
        _conform_motion(motion)
        self.motion = deepcopy(dict(motion))
        pose = measured_pose(view.state)
        position = np.asarray(motion["xyz_m"], dtype=float)
        if motion["position_mode"] == "relative":
            with np.errstate(over="ignore", invalid="ignore"):
                position = np.asarray(pose[:3]) + position
        orientation = pose[3:7] if motion["quaternion_wxyz"] is None else motion["quaternion_wxyz"]
        gripper = motion["gripper"]
        if gripper == "keep":
            if keep_gripper not in ("open", "close"):
                raise ValueError("gripper keep requires an explicit open/close runtime command")
            gripper = keep_gripper
        self.gripper_command = gripper
        self.commanded_reorientation = bool(np.linalg.norm(
            rotation_error_rad(orientation, view.state[3:6])) > 1e-9)
        target = [*position, *orientation, int(gripper == "close")]
        self.executor = None
        try:
            self.executor = ShortTargetExecutor(view.state, target, motion["steps"])
        except TargetValidationError as exc:
            try:
                evidence = rejected_feedback(view.state, target, motion["steps"], str(exc))
            except TargetValidationError:
                evidence = {"status": "rejected", "requested_motion": self.motion,
                            "measured_pose": pose, "requested_steps": motion["steps"],
                            "executed_steps": 0,
                            "detail": str(exc) + "; no arm or gripper action was executed"}
            self._finish("failed", "local_motion_target_rejected", evidence)

    def _next(self, view: StepView):
        return self.executor.next_action(view.state)

    def _observe(self, before: StepView, action, after: StepView) -> dict:
        feedback = self.executor.observe_transition(before.state, action, after.state)
        if self.executor.done:
            self._finish("succeeded", "local_motion_segment_complete", feedback)
        return {"local_motion": feedback}
