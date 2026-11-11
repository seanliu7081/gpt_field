"""RGB/joint-only semantic intent contract for the visual-field experiment.

Uses the existing audited OAuth transport, with no credential or model fallback.
Shared tracking feedback and task guidance are identical in both comparison arms.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .astra import SchemaError, _canonical
from .demonstration import DemoReference
from .direct_history import _trim_images
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient
from .observation import PolicyObservation

VERSION = "visual-field-rgb-joints-v1"
PHASES = ("approach", "grasp", "transport", "align", "release", "verify", "recover",
          "handle_grasp", "drawer_open", "handle_release")


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _array(n, lo=None, hi=None):
    item = {"type": "number"}
    if lo is not None:
        item["minimum"] = lo
    if hi is not None:
        item["maximum"] = hi
    return {"type": "array", "items": item, "minItems": n, "maxItems": n}


def output_schema(controller="hybrid"):
    identifier = {"type": "string", "minLength": 1, "maxLength": 80}
    box = _object({"object_id": identifier, "lower_uv": _array(2, 0, 1),
                   "upper_uv": _array(2, 0, 1)})
    boxes = {"type": "array", "items": box, "maxItems": 8}
    reference = _object({"moving_id": identifier, "target_id": identifier,
        "moving_kind": {"type": "string", "enum": ["tool", "payload"]},
        "tool_point_uv": {"anyOf": [_array(2, 0, 1), {"type": "null"}]},
        "moving_point_uv": _array(2, 0, 1), "target_point_uv": _array(2, 0, 1),
        "moving_radius_uv": {"type": "number", "minimum": 0, "maximum": .3},
        "blocked_boxes_uv": boxes, "unknown_boxes_uv": boxes})
    nullable = lambda schema: {"anyOf": [schema, {"type": "null"}]}
    intent = _object({"goal_offset_m": _array(3, -.16, .16),
                      "orientation_target_xyzw": _array(4, -1, 1),
                      "gripper_command": {"type": "string", "enum": ["open", "close", "hold"]}})
    actions = {"type": "array", "minItems": 5, "maxItems": 5, "items": _array(7, -1, 1)}
    return _object({"controller": {"type": "string", "enum":
                        ["visual_field", "direct"] if controller == "hybrid" else ["direct"]},
        "phase": {"type": "string", "enum": list(PHASES)}, "target_id": identifier,
        "hypothesis_id": identifier,
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "visual_references": _object({"external": nullable(reference), "wrist": nullable(reference)}),
        "field_intent": nullable(intent), "direct_actions": nullable(actions)})


def _validate(value, schema, path="response"):
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate(value, option, path)
                return
            except SchemaError as exc:
                errors.append(exc)
        raise SchemaError(f"{path} has an invalid nullable value")
    kind = schema["type"]
    if kind == "null":
        valid = value is None
    elif kind == "object":
        valid = isinstance(value, dict) and set(value) == set(schema["properties"])
        if valid:
            for key, child in schema["properties"].items():
                _validate(value[key], child, path + "." + key)
    elif kind == "array":
        valid = isinstance(value, list) and schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 999)
        if valid:
            for item in value:
                _validate(item, schema["items"], path + "[]")
    elif kind == "number":
        valid = type(value) in (int, float) and math.isfinite(value)
        valid = valid and schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf)
    else:
        valid = isinstance(value, str) and schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", 999)
    if not valid or "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path} violates the visual control schema")


def shared_guidance(task, demonstration):
    return (
        f"Task: {task.prompt}. Use only current external and wrist RGB, measured eight-value "
        "EEF xyz/axis-angle/finger state, seven arm joint positions/velocities and observable history. "
        "No depth, supplied camera/environment calibration, object ground truth, contacts, reward or "
        "benchmark success is available. Joint positions are separate from the eight-value state. "
        "Complete the task without unintended collision or obstacle displacement. Necessary finger "
        "contact with the task object or articulated handle and object support on the intended "
        "surface are allowed; other robot/payload contact is not. Consider fingers, payload and arm "
        "extent. Occluded space is uncertain, not certified free. Image overlap is not a 3D collision "
        "test. Preserve object identities across views/time; never substitute a nearby similar object "
        "after occlusion. Ground identities and motion in current images, not guessed world positions. "
        "Provide visual_references for BOTH views whenever identifiable: normalized UV, top-left origin, "
        "u right, v down in the images shown. moving_point_uv is a textured tracked point on the tool "
        "or rigidly held object; target_point_uv is the desired projection of that SAME point, accounting "
        "for grasp offset/orientation. Choose stable moving_id/target_id; don't rename unchanged points. "
        "Use conservative moving_radius_uv for visible tool/payload extent. blocked_boxes_uv bound "
        "relevant visible obstacles, unknown_boxes_uv bound uncertain/occluded regions, with stable "
        "object_id. Do not label the intended grasp/support contact region as an unintended obstacle. "
        "moving_kind=tool for a tool point, payload for an object point. When tracking payload, supply "
        "tool_point_uv on an independently visible gripper feature to check co-motion; use null if "
        "unobservable. During a test lift, track payload and gripper independently where possible. "
        "References may be null only when genuinely unobservable/ambiguous; explain this in reason. "
        "Local patch matches are fallible and cannot certify task success or 3D clearance. "
        "Use a stable hypothesis_id for one grasp/interaction hypothesis. After failed motion/grasp, "
        "change approach, alignment or observation with visible evidence; don't merely rename the "
        "same failed attempt. Check object co-motion during a short test lift before transport, "
        "continued attachment during transport, and intended support after release. Closure alone "
        "does not prove grasp. Stage changes are hypotheses, not evaluator truth. Avoid repetitive "
        "hold/verify when observable placement remains ambiguous; gather information and correct it. "
        "All OSC actions are seven normalized numbers [dx,dy,dz,rx,ry,rz,grip], [-1,1]. Translation "
        "1 means 0.05 m delta target; rotation 1 means 0.5 rad world axis-angle delta. +1 closes, "
        "-1 opens, 0 is zero incremental gripper motion. Commands execute at 20 Hz relative to the "
        "actual measured pose, not the preceding requested target. Actual movement can be smaller. "
        "Five actions are only 0.25 s: allow closure to settle, test attachment, then transport. "
        "Infer articulated handle motion from visual evidence; opening a drawer does not imply bowl "
        "placement. Choose visibly informative recovery; no hard-coded task-specific recovery exists. "
        + ("ONE-SHOT: exactly one fixed external demonstration precedes live observations. Learn roles, "
           "relative alignments, contact intent and stage evidence; its layout/obstacles may differ. "
           "Re-localize and replan from current RGB. Never replay its coordinates/actions. " if demonstration else
           "ZERO-SHOT: no external demonstration is supplied. Live history is execution feedback, "
           "not a reference demonstration. Infer the task from its instruction and current views. ")
    )


class VisualFieldSubscriptionClient(ChatGPTPlanHybridClient):
    def __init__(self, config=None, *, controller="hybrid", task_key="bowl_plate",
                 demonstration: DemoReference | None = None, **kwargs):
        if controller not in ("hybrid", "direct"):
            raise ValueError("controller must be hybrid or direct")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("This comparison requires gpt-6-astra")
        self.controller_mode = controller
        self.task_spec = resolve_distance_task(task_key)
        self.demonstration = demonstration
        if demonstration is not None and demonstration.metadata()["task_key"] != task_key:
            raise ValueError("Demonstration task differs")
        self.observable_context = {}
        super().__init__(config, reasoning_effort="medium", task_prompt=self.task_spec.prompt, **kwargs)

    @property
    def prompt_version(self):
        return VERSION + "-" + self.controller_mode

    def build_payload(self, observation, grid=None, mode="hybrid"):
        obs = PolicyObservation.from_dict(observation)
        if obs.prompt != self.task_spec.prompt or obs.joint_position is None:
            raise ValueError("Visual field requires task-matched RGB and measured joint feedback")
        text = {"type": "observation", "observation_index": self._observation_count,
            "prompt": obs.prompt, "observation/state": list(obs.state),
            "observation/joint_position": list(obs.joint_position),
            "observation/joint_velocity": list(obs.joint_velocity),
            "observable_context": copy.deepcopy(self.observable_context)}
        live = _trim_images(self._history + [{"role": "user", "content": [
            {"type": "input_text", "text": _canonical(text)},
            {"type": "input_text", "text": "CURRENT EXTERNAL RGB"},
            {"type": "input_image", "image_url": obs.image},
            {"type": "input_text", "text": "CURRENT WRIST RGB"},
            {"type": "input_image", "image_url": obs.wrist_image}]}], 2)
        instructions = shared_guidance(self.task_spec, self.demonstration is not None)
        if self.controller_mode == "hybrid":
            instructions += (
                "HYBRID: select visual_field for free motion with persistent local intent, direct for "
                "contact/precise operations. visual_field supplies field_intent and direct_actions=null. "
                "Its goal_offset_m is a bounded local Cartesian prior relative to current EEF; provide "
                "absolute orientation_target_xyzw (unit quaternion) and gripper_command. Local image "
                "Dijkstra fields and learned OSC/joint/image response score short candidate translations "
                "every action, with conservative bootstrap before sufficient evidence. Reuse up to 30 "
                "actions, extend to 120 only with reliable progress; refresh earlier on lost tracks, "
                "arrival, model mismatch or failed progress. This is not a metric 3D obstacle map. "
                "For direct, field_intent=null and exactly five direct_actions execute unchanged. ")
        else:
            instructions += ("DIRECT: controller=direct, field_intent=null; exactly five direct_actions "
                "execute unchanged. Shared visual tracking summaries inform the next request, but no "
                "local field planner, motion correction or automatic recovery alters your actions. ")
        instructions += "Return only strict JSON. Do not output a task-completion flag."
        return {"model": self.config.model, "store": False, "stream": True,
                "reasoning": {"effort": "medium"}, "include": ["reasoning.encrypted_content"],
                "instructions": instructions,
                "input": ([self.demonstration.message()] if self.demonstration else []) + live,
                "text": {"format": {"type": "json_schema", "strict": True,
                    "name": "visual_field_rgb_joints_v1", "schema": output_schema(self.controller_mode)}}}

    def _history_input(self, payload):
        return payload["input"][1:] if self.demonstration else payload["input"]

    def validate_response(self, data, grid):
        _validate(data, output_schema(self.controller_mode))
        field = data["controller"] == "visual_field"
        if (data["field_intent"] is not None) != field or (data["direct_actions"] is None) != field:
            raise SchemaError("Only the selected controller branch may be populated")
        if field:
            q = data["field_intent"]["orientation_target_xyzw"]
            if abs(sum(x*x for x in q) - 1) > .03:
                raise SchemaError("Orientation target must be a unit quaternion")
        for ref in data["visual_references"].values():
            if ref is None:
                continue
            for box in ref["blocked_boxes_uv"] + ref["unknown_boxes_uv"]:
                if any(a >= b for a, b in zip(box["lower_uv"], box["upper_uv"])):
                    raise SchemaError("Visual boxes require ordered positive-area bounds")
        return copy.deepcopy(data)

    def request_key(self, payload, grid, mode):
        return hashlib.sha256(_canonical({"payload": payload, "version": self.prompt_version}).encode()).hexdigest()

    def record_block(self, record):
        # Only policy-created observable records enter history; never pass simulator status here.
        with self._history_lock:
            self._history.append({"role": "user", "content": [
                {"type": "input_text", "text": _canonical(record)}]})
            self._feedback_count += 1
