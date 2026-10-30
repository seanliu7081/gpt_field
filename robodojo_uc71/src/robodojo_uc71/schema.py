"""Strict dual-ARX model boundary; evaluator records have no schema here."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
import re

import numpy as np

ARMS = ("left", "right")
CAMERAS = ("head", "left_wrist", "right_wrist")
SCHEMA_VERSION = "robodojo-uc71-dual-arm-1"
CALIBRATION_VERSION = "robodojo-uc71-calibration-1"
SKILLS = ("move_to_pose", "move_relative", "approach", "set_gripper", "wait")
DENIED_KEYS = frozenset({
    "reward", "rewards", "score", "partial_score", "success", "task_success",
    "official_success", "end_flag", "remaining_steps", "remaining_actions",
    "native_action_count", "take_action_cnt", "step_lim", "action_budget",
    "object_poses", "object_positions", "object_state", "scene_state", "sim_state",
    "depth", "segmentation", "point_cloud", "contacts", "contact_ids",
    "contact_identities", "support_truth", "held_truth", "grasp_truth",
    "reward_predicates", "grasp_perfect_direction", "task_recipe",
})


@dataclass(frozen=True)
class ExecutionLimits:
    """Explicit local command limits; these are not an episode budget.

    Candidate defaults preserve fragment/gripper durations at native 25 Hz.
    Direct motion intentionally retains the frozen one-to-five action count.
    Controller characterization must approve a replacement or these candidates
    before any pilot; schema validation is not controller calibration.
    """

    control_hz: int = 25
    max_segments: int = 6
    max_segment_steps: int = 150
    max_execution_steps: int = 300
    direct_max_steps: int = 5
    min_open_steps: int = 10
    min_close_steps: int = 15
    transit_translation_m: float = .30
    transit_rotation_rad: float = 1.2
    contact_translation_m: float = .08
    contact_rotation_rad: float = .5
    direct_translation_m: float = .05
    direct_rotation_rad: float = .35

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid execution limit: {name}")
        for name in ("control_hz", "max_segments", "max_segment_steps", "max_execution_steps",
                     "direct_max_steps", "min_open_steps", "min_close_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.control_hz != 25:
            raise ValueError("The native pilot uses 25 Hz")
        if max(self.min_open_steps, self.min_close_steps) > self.max_segment_steps:
            raise ValueError("Gripper minimum exceeds segment limit")


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{where}: missing or unexpected fields")


def _number(value, where):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{where}: expected finite number")
    return float(value)


def vector(value, size, where):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError(f"{where}: expected {size} values")
    return [_number(x, where) for x in value]


def quaternion(value, where, *, normalize=False):
    q = np.asarray(vector(value, 4, where))
    scale = np.max(np.abs(q))
    if not scale:
        raise ValueError(f"{where}: zero quaternion")
    scaled = q / scale
    scaled_norm = np.linalg.norm(scaled)
    if normalize:
        return (scaled / scaled_norm).tolist()
    norm = scale * scaled_norm
    if not normalize and not math.isclose(float(norm), 1., abs_tol=1e-5):
        raise ValueError(f"{where}: measured quaternion must be unit length")
    return (scaled / scaled_norm).tolist()


def _matrix(value, rows, columns, where):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != rows:
        raise ValueError(f"{where}: wrong matrix shape")
    return np.asarray([vector(row, columns, where) for row in value])


def _transform(value, where):
    matrix = _matrix(value, 4, 4, where)
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-5)):
        raise ValueError(f"{where}: expected rigid world transform")
    return matrix.tolist()


def validate_calibration(value):
    _keys(value, {"version", "cameras", "tools"}, "calibration")
    if value["version"] != CALIBRATION_VERSION:
        raise ValueError("Unsupported calibration version")
    _keys(value["cameras"], CAMERAS, "cameras")
    _keys(value["tools"], ARMS, "tools")
    clean = {"version": CALIBRATION_VERSION, "cameras": {}, "tools": {}}
    for name in CAMERAS:
        camera = value["cameras"][name]
        _keys(camera, {"width", "height", "intrinsics", "world_from_optical", "pose_source"}, name)
        if type(camera["width"]) is not int or type(camera["height"]) is not int or (camera["width"], camera["height"]) != (640, 480):
            raise ValueError("The pilot delivers native uncropped 640x480 RGB")
        k = _matrix(camera["intrinsics"], 3, 3, name + ".intrinsics")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError("Invalid pinhole intrinsics")
        if not isinstance(camera["pose_source"], str) or not camera["pose_source"].strip() or len(camera["pose_source"]) > 100:
            raise ValueError("Camera pose_source must identify synchronized robot/static calibration")
        clean["cameras"][name] = {**camera, "intrinsics": k.tolist(),
            "world_from_optical": _transform(camera["world_from_optical"], name)}
    for name in ARMS:
        tool = value["tools"][name]
        _keys(tool, {"link_name", "link6_from_tool", "finger_gap_formula", "finger_gap_semantics", "source_sha256"}, name)
        if (tool["link_name"] != "link6"
                or tool["finger_gap_formula"] != "sum_of_measured_finger_joint_positions"
                or tool["finger_gap_semantics"] != "joint_separation_proxy"
                or not isinstance(tool["source_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", tool["source_sha256"]) is None):
            raise ValueError("Tool calibration must have explicit physical provenance and gap semantics")
        clean["tools"][name] = {**tool, "link6_from_tool": _transform(tool["link6_from_tool"], name)}
    return deepcopy(clean)


def validate_robot(value):
    _keys(value, ARMS, "robot")
    result = {}
    required = {"tool_pose_wxyz", "joint_position_rad", "joint_velocity_rad_s",
                "finger_position_m", "finger_gap_m", "gripper_command"}
    for name in ARMS:
        arm = value[name]
        _keys(arm, required, name)
        pose = vector(arm["tool_pose_wxyz"], 7, name + ".tool_pose_wxyz")
        pose[3:] = quaternion(pose[3:], name)
        fingers = vector(arm["finger_position_m"], 2, name + ".finger_position_m")
        gap = _number(arm["finger_gap_m"], name + ".finger_gap_m")
        if not math.isclose(gap, sum(fingers), abs_tol=1e-9, rel_tol=1e-7):
            raise ValueError("finger_gap_m must match the declared measured-joint proxy")
        if arm["gripper_command"] not in ("open", "close"):
            raise ValueError("gripper_command is the latched requested binary command")
        result[name] = {"tool_pose_wxyz": pose,
            "joint_position_rad": vector(arm["joint_position_rad"], 6, name),
            "joint_velocity_rad_s": vector(arm["joint_velocity_rad_s"], 6, name),
            "finger_position_m": fingers, "finger_gap_m": gap,
            "gripper_command": arm["gripper_command"]}
    return result


def public_json(value):
    """Reject known privileged channels, including in nested feedback/error data."""
    clean = json.loads(json.dumps(value, allow_nan=False))
    def visit(item):
        if isinstance(item, dict):
            if any(str(key).lower() in DENIED_KEYS for key in item):
                raise ValueError("Privileged evaluator/budget field in model input")
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(clean)
    return clean


FEEDBACK_REASONS = frozenset({"pose_arrived", "gap_stable", "wait_elapsed", "fragment_complete",
    "observation_required", "segment_budget_exhausted", "execution_budget_exhausted", "segment_stall",
    "maintained_grip_gap_changed", "target_envelope_rejected", "ik_unavailable", "joint_limit",
    "controller_unavailable", "observation_only", "model_finish_observation", "interrupted",
    "measurement_unavailable", "running"})
_MEASURED_SCALARS = frozenset({"position_error_m", "orientation_error_rad", "finger_gap_m"})
_MEASURED_POSES = frozenset({"entry_tool_pose_wxyz", "target_tool_pose_wxyz", "last_tool_pose_wxyz"})


def validate_feedback(value):
    """Typed robot feedback only: no arbitrary text, exception or debug payloads."""
    if value is None:
        return None
    allowed = {"status", "reason", "arm_id", "skill", "executed_steps", "gripper_command",
               "completed_segment_indices", "pending_segment_indices", "measurements"} | _MEASURED_SCALARS | _MEASURED_POSES
    if not isinstance(value, dict) or not {"status", "reason"} <= set(value) or set(value) - allowed:
        raise ValueError("Robot feedback has missing or non-allowlisted fields")
    if value["status"] not in ("succeeded", "failed", "blocked", "pending_verification", "running") or value["reason"] not in FEEDBACK_REASONS:
        raise ValueError("Robot feedback requires enumerated status/reason codes")
    clean = deepcopy(value)
    if "arm_id" in clean and clean["arm_id"] not in ARMS:
        raise ValueError("Invalid feedback arm")
    if "skill" in clean and clean["skill"] not in (*SKILLS, "motion", "observe", "finish"):
        raise ValueError("Invalid feedback skill")
    if "gripper_command" in clean and clean["gripper_command"] not in ("open", "close"):
        raise ValueError("Invalid requested gripper command")
    if "executed_steps" in clean and (type(clean["executed_steps"]) is not int or clean["executed_steps"] < 0):
        raise ValueError("executed_steps is a nonnegative local-fragment count")
    for key in _MEASURED_SCALARS & clean.keys():
        clean[key] = _number(clean[key], key)
    for key in _MEASURED_POSES & clean.keys():
        pose = vector(clean[key], 7, key)
        pose[3:] = quaternion(pose[3:], key)
        clean[key] = pose
    for key in ("completed_segment_indices", "pending_segment_indices"):
        if key in clean and (not isinstance(clean[key], list) or len(clean[key]) > 6 or
                any(type(x) is not int or not 0 <= x < 6 for x in clean[key])):
            raise ValueError("Invalid local segment indices")
    if "measurements" in clean:
        samples = clean["measurements"]
        if not isinstance(samples, list) or len(samples) > 32:
            raise ValueError("At most 32 explicit robot measurements per feedback snapshot")
        allowed_sample = _MEASURED_SCALARS | {"tool_pose_wxyz", "finger_position_m"}
        for sample in samples:
            if not isinstance(sample, dict) or not sample or set(sample) - allowed_sample:
                raise ValueError("Measurement has non-allowlisted fields")
            for key in _MEASURED_SCALARS & sample.keys():
                sample[key] = _number(sample[key], key)
            if "tool_pose_wxyz" in sample:
                pose = vector(sample["tool_pose_wxyz"], 7, "sample pose")
                pose[3:] = quaternion(pose[3:], "sample pose")
                sample["tool_pose_wxyz"] = pose
            if "finger_position_m" in sample:
                sample["finger_position_m"] = vector(sample["finger_position_m"], 2, "sample fingers")
    return clean


def validate_controller_capabilities(value):
    """Small control-limit allowlist; no full joint model or arbitrary metadata."""
    positive = {"action_rotation_rad", "max_joint_step_rad", "joint_target_lead_rad", "gap_stable_range_m",
                "gap_rise_yield_m", "stall_progress_m", "orientation_error_weight_m_per_rad"}
    counts = {"arrival_stable_samples", "gap_stable_samples", "gap_rise_samples", "stall_window_steps", "stall_min_steps"}
    profiles = {"action_translation_m", "position_tolerance_m", "orientation_tolerance_rad"}
    if not isinstance(value, dict) or set(value) - positive - counts - profiles - {"status", "version"}:
        raise ValueError("Controller capabilities have non-allowlisted fields")
    clean = deepcopy(value)
    if "version" in clean and clean["version"] not in (
            "robodojo-uc71-robot-only-controller-1", "robodojo-uc71-official-ee-controller-1"):
        raise ValueError("Unknown controller capability version")
    if clean.get("version") == "robodojo-uc71-official-ee-controller-1" and (
            set(clean) & {"action_translation_m", "action_rotation_rad", "max_joint_step_rad", "joint_target_lead_rad"}):
        raise ValueError("Official IK does not inherit the legacy differential-controller step limits")
    if "status" in clean and clean["status"] not in ("candidate_unvalidated", "validated"):
        raise ValueError("Invalid controller validation status")
    for key in positive & clean.keys():
        clean[key] = _number(clean[key], key)
        if clean[key] <= 0:
            raise ValueError("Controller physical limits must be positive")
    for key in counts & clean.keys():
        if type(clean[key]) is not int or clean[key] < 1:
            raise ValueError("Controller sample counts must be positive integers")
    for key in profiles & clean.keys():
        _keys(clean[key], {"transit", "contact"}, key)
        for profile in clean[key]:
            clean[key][profile] = _number(clean[key][profile], key)
            if clean[key][profile] <= 0:
                raise ValueError("Profile limits must be positive")
    return clean


def calibration_binding(value):
    """Only wrist extrinsics are dynamic; all other delivered calibration is fixed."""
    result = validate_calibration(value)
    for name in ("left_wrist", "right_wrist"):
        result["cameras"][name].pop("world_from_optical")
    return result


def validate_observation(value):
    if not isinstance(value, dict) or set(value) not in (
        {"step", "images", "robot", "calibration"},
        {"step", "images", "robot", "calibration", "direction_frame"},
    ):
        raise ValueError("Observation has missing or unexpected fields")
    if type(value["step"]) is not int or value["step"] < 0:
        raise ValueError("step is a nonnegative model-observation index, never native action count")
    _keys(value["images"], CAMERAS, "images")
    return {"step": value["step"], "images": dict(value["images"]),
            "robot": validate_robot(value["robot"]),
            "calibration": validate_calibration(value["calibration"]),
            **({"direction_frame": deepcopy(value["direction_frame"])} if "direction_frame" in value else {})}


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _vec_schema(size, nullable=False):
    value = {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}
    return {"anyOf": [value, {"type": "null"}]} if nullable else value


def execution_schema(limits=ExecutionLimits()):
    fields = {"skill": {"type": "string", "enum": list(SKILLS)},
              "xyz_m": _vec_schema(3, True), "quaternion_wxyz": _vec_schema(4, True),
              "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
              "profile": {"type": "string", "enum": ["transit", "contact"]},
              "max_steps": {"type": "integer", "minimum": 1, "maximum": limits.max_segment_steps},
              "observation_after": {"type": "boolean"}}
    return _object({"segments": {"type": "array", "minItems": 1, "maxItems": limits.max_segments,
                                  "items": _object(fields)}})


def motion_schema(limits=ExecutionLimits()):
    return _object({"frame": {"type": "string", "enum": ["world"]},
        "position_mode": {"type": "string", "enum": ["absolute", "relative"]},
        "xyz_m": _vec_schema(3), "quaternion_wxyz": _vec_schema(4, True),
        "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        "steps": {"type": "integer", "minimum": 1, "maximum": limits.direct_max_steps}})


def decision_schema(limits=ExecutionLimits()):
    return _object({"belief": {"type": "string", "minLength": 1, "maxLength": 1800},
        "plan": {"type": "string", "minLength": 1, "maxLength": 1800},
        "why": {"type": "string", "minLength": 1, "maxLength": 600},
        "type": {"type": "string", "enum": ["execute", "motion", "observe", "finish"]},
        "arm_id": {"anyOf": [{"type": "string", "enum": list(ARMS)}, {"type": "null"}]},
        "motion": {"anyOf": [motion_schema(limits), {"type": "null"}]},
        "execution": {"anyOf": [execution_schema(limits), {"type": "null"}]}})


def validate_execution(value, limits=ExecutionLimits()):
    _keys(value, {"segments"}, "execution")
    if not isinstance(value["segments"], list) or not 1 <= len(value["segments"]) <= limits.max_segments:
        raise ValueError("Invalid segment count")
    result = deepcopy(value)
    for segment in result["segments"]:
        _keys(segment, execution_schema(limits)["properties"]["segments"]["items"]["required"], "segment")
        skill, profile, grip = segment["skill"], segment["profile"], segment["gripper"]
        if skill not in SKILLS or profile not in ("transit", "contact") or grip not in ("keep", "open", "close"):
            raise ValueError("Invalid segment enum")
        if type(segment["max_steps"]) is not int or not 1 <= segment["max_steps"] <= limits.max_segment_steps:
            raise ValueError("Invalid segment action allowance")
        if type(segment["observation_after"]) is not bool:
            raise ValueError("observation_after must be boolean")
        if segment["xyz_m"] is not None:
            segment["xyz_m"] = vector(segment["xyz_m"], 3, "xyz_m")
        if segment["quaternion_wxyz"] is not None:
            segment["quaternion_wxyz"] = quaternion(segment["quaternion_wxyz"], "quaternion_wxyz", normalize=True)
        if skill in SKILLS[:3]:
            if segment["xyz_m"] is None or grip != "keep" or (skill == "approach" and profile != "contact"):
                raise ValueError("Motion requires xyz_m/gripper keep; approach requires contact")
            envelope = getattr(limits, profile + "_translation_m")
            if skill != "move_to_pose" and math.hypot(*segment["xyz_m"]) > envelope + 1e-12:
                raise ValueError("Relative target exceeds profile envelope")
        else:
            if segment["xyz_m"] is not None or segment["quaternion_wxyz"] is not None:
                raise ValueError("Gripper/wait has no pose target")
            if skill == "wait" and grip != "keep":
                raise ValueError("wait preserves gripper command")
            if skill == "set_gripper" and (grip == "keep" or segment["max_steps"] < getattr(limits, f"min_{grip}_steps")):
                raise ValueError("set_gripper requires open/close and its minimum allowance")
    if sum(s["max_steps"] for s in result["segments"]) > limits.max_execution_steps:
        raise ValueError("Fragment exceeds execution allowance")
    return result


def validate_decision(value, limits=ExecutionLimits()):
    schema = decision_schema(limits)
    _keys(value, schema["required"], "decision")
    result = deepcopy(value)
    for name in ("belief", "plan", "why"):
        text = result[name]
        if not isinstance(text, str) or not text.strip() or len(text) > schema["properties"][name]["maxLength"]:
            raise ValueError("Invalid bounded reasoning text")
    kind = result["type"]
    if kind not in ("execute", "motion", "observe", "finish"):
        raise ValueError("Unsupported action; native Panda macros are unavailable")
    if result["arm_id"] not in (ARMS if kind in ("execute", "motion") else (None,)):
        raise ValueError("Select exactly one arm for motion; observe/finish use null")
    if kind == "execute":
        result["execution"] = validate_execution(result["execution"], limits)
    elif result["execution"] is not None:
        raise ValueError("Inactive execution must be null")
    if kind == "motion":
        motion = result["motion"]
        _keys(motion, motion_schema(limits)["required"], "motion")
        if motion["frame"] != "world" or motion["position_mode"] not in ("absolute", "relative") or motion["gripper"] not in ("keep", "open", "close"):
            raise ValueError("Invalid short-motion enum")
        motion["xyz_m"] = vector(motion["xyz_m"], 3, "motion.xyz_m")
        if motion["quaternion_wxyz"] is not None:
            motion["quaternion_wxyz"] = quaternion(motion["quaternion_wxyz"], "motion.quaternion", normalize=True)
        if type(motion["steps"]) is not int or not 1 <= motion["steps"] <= limits.direct_max_steps:
            raise ValueError("Invalid short-motion allowance")
        if motion["position_mode"] == "relative" and math.hypot(*motion["xyz_m"]) > limits.direct_translation_m + 1e-12:
            raise ValueError("Short relative target exceeds envelope")
    elif result["motion"] is not None:
        raise ValueError("Inactive motion must be null")
    return result


def capabilities(limits=ExecutionLimits()):
    return {"version": SCHEMA_VERSION, "execution_owner": "model", "active_arms_per_decision": 1,
            "frame": "world", "control_hz": limits.control_hz, "skills": list(SKILLS),
            "limits": asdict(limits), "native_macros": {"available": False,
                "reason": "Panda-specific macro machinery is outside this explicit ARX pilot adaptation"},
            "completion": "measured robot pose or finger settling only; object/task completion remains unknown",
            "inactive_arm": "hold measured handoff reference and retain requested gripper command"}
