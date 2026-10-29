"""Opt-in persistent metric hypothesis fields, RGB feedback, and subscription transport.

Geometry in this contract is a model hypothesis, never measured depth or a
collision certificate. The protocol deliberately gives the two cameras distinct
roles and does not require a complete tool footprint in the wrist image.
"""
from __future__ import annotations

import copy
import math

from .astra import SchemaError, _canonical
from .visual_field_subscription import PHASES, _array, _object, _validate
from .visual_field_subscription_v2 import VisualFieldSubscriptionClientV2, _text_record

VERSION = "visual-field-rgb-joints-v4-center-extents-v2"
HISTORY_VERSION = "visual-v4-persistent-intent-recent-two-turns-v1"
FIELD_PHASES = ("approach", "transport", "align", "recover")


def output_schema(controller="hybrid"):
    if controller != "hybrid":
        raise ValueError("Visual v4 is zero-shot Hybrid only")
    identifier = {"type": "string", "minLength": 1, "maxLength": 80}
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    external = _object({
        "moving_id": identifier, "moving_kind": {"type": "string", "enum": ["tool", "payload"]},
        "moving_point_uv": _array(2, 0, 1), "target_id": identifier,
        "target_anchor_uv": _array(2, 0, 1), "target_point_uv": _array(2, 0, 1),
        "tool_point_uv": nullable(_array(2, 0, 1)),
    })
    wrist = _object({"target_id": identifier, "target_anchor_uv": _array(2, 0, 1),
                     "desired_target_uv": _array(2, 0, 1)})
    # The wire contract expresses volume directly, so corner ordering is not a
    # separate arithmetic task for the model. These limits imply local corners
    # in [-1, 1] m; the execution validator independently checks those bounds.
    box = _object({"id": identifier, "center_offset_m": _array(3, -.5, .5),
                   "half_extents_m": _array(3, .0001, .5),
                   "uncertainty_m": {"type": "number", "minimum": 0, "maximum": .1}})
    intent = _object({
        "goal_offset_m": _array(3, -.08, .08),
        "orientation_target_xyzw": _array(4, -1, 1),
        "gripper_command": {"type": "string", "enum": ["hold"]},
        "goal_tolerance_m": {"type": "number", "minimum": .003, "maximum": .01},
        "max_actions": {"type": "integer", "minimum": 10, "maximum": 60},
        "max_travel_m": {"type": "number", "minimum": .005, "maximum": .16},
        "command_limit_m": {"type": "number", "minimum": .002, "maximum": .02},
        "tool_radius_m": {"type": "number", "minimum": .005, "maximum": .08},
        "uncertainty_m": {"type": "number", "minimum": .001, "maximum": .04},
        "obstacles": {"type": "array", "items": copy.deepcopy(box), "maxItems": 8},
        "unknown_regions": {"type": "array", "items": copy.deepcopy(box), "maxItems": 8},
    })
    actions = {"type": "array", "items": _array(7, -1, 1), "minItems": 5, "maxItems": 5}
    return _object({
        "controller": {"type": "string", "enum": ["direct", "visual_field"]},
        "phase": {"type": "string", "enum": list(PHASES)}, "target_id": identifier,
        "hypothesis_id": identifier, "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "visual_references": _object({"external": nullable(external), "wrist": nullable(wrist)}),
        "field_intent": nullable(intent), "direct_actions": nullable(actions),
    })


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v4 is zero-shot Hybrid only")
    return (
        f"Task: {task.prompt}. ZERO-SHOT: no external demonstration is supplied. Use current external "
        "and wrist RGB, measured eight-value EEF xyz/axis-angle/finger state, seven arm joint positions "
        "and velocities, and observable execution history. No depth, supplied camera/environment "
        "calibration, object truth, contacts, reward, or evaluator success is provided. Live feedback "
        "is not a reference demonstration. Preserve task-object identity across views and occlusion. "
        "Complete the task without unintended collision or obstacle displacement. Necessary finger "
        "contact with the task object/handle and object support on the intended surface are allowed; "
        "other robot/payload contact is not. Account for fingers, payload and arm extent. Image overlap "
        "alone is not 3D collision evidence; missing depth is not evidence of clearance. "
        "V4 HYBRID: use visual_field for sustained noncontact approach, transport, alignment or recovery. "
        "Output one persistent field_intent, normally max_actions=30, and direct_actions=null. Its "
        "goal_offset_m is a local EEF translation hypothesis in robot/world axes relative to the CURRENT "
        "measured EEF, bounded to +/-0.08 m per axis. It remains fixed while local planning and measured "
        "pose feedback track its path. A 33-cubed grid at 0.005 m spacing represents a weighted "
        "goal-distance field; the controller plans around your hypothesized boxes and follows the "
        "route without another model request for every action. Supply all intent bounds explicitly: "
        "goal_tolerance_m, max_actions (10..60), max_travel_m, command_limit_m, tool_radius_m and "
        "uncertainty_m. These are local intent budgets, not episode timeouts. Local checks may refresh "
        "early on arrival, disagreement, lost identity, insufficient measured motion or budget exhaustion. "
        "Do not repeatedly request Direct calibration before any Field use: a bounded cold-start "
        "translation can collect response evidence while following a valid hypothesis route. "
        "OBSTACLE HYPOTHESES: obstacles and unknown_regions are world-axis-aligned boxes. Express "
        "center_offset_m in meters from the SAME current EEF, each component in [-0.5,0.5], and "
        "half_extents_m as three POSITIVE half-sizes in [0.0001,0.5] m (half the full width along "
        "each world axis). Do not output lower/upper corners or signed sizes. The controller derives "
        "lower=center-half_extents and upper=center+half_extents without changing your geometry. "
        "Their positions/extents and "
        "uncertainty_m are RGB-based hypotheses, not calibrated scene measurements. Identify relevant "
        "visible obstacles and genuinely uncertain route regions; maintain stable box IDs. The local "
        "planner inflates them by tool/payload radius and uncertainty and excludes them. Do not carve "
        "out the robot start, goal or contact target to force a route; revise a contradicted hypothesis "
        "or use a visibly justified contact/observation action. Avoid projecting every occluded image "
        "pixel into an invented solid 3D wall. Neither an empty list nor a feasible route certifies "
        "physical clearance. Field holds orientation and gripper: gripper_command must be hold; "
        "orientation_target_xyzw must match the measured orientation. Use Direct for contact, grasp, "
        "release, articulated interaction or reorientation before returning to Field. "
        "CAMERA ROLES: normalized UV has top-left origin, u right and v down. External moving_point_uv "
        "is a currently visible textured point on the actually moving tool or payload, never a static "
        "robot base or background patch. target_anchor_uv is a real visible target landmark; "
        "target_point_uv is the desired projection of that SAME moving point and can be virtual. "
        "Preserve their physical identities. External reference is required for Field. Wrist reference "
        "tracks only a real target_anchor_uv and desired_target_uv bearing: it does not require the "
        "whole gripper, a footprint or a tool landmark to be in view. Wrist may be null if unavailable "
        "without globally disabling Field. Its image is still available for your contact decisions. "
        "The controller validates external feature response using measured robot movement; a stationary "
        "patch match is not semantic evidence of the correct moving point. "
        "ATTACHMENT: closure alone does not prove grasp. After grasp, check object/tool co-motion with "
        "a short Direct test before Field transport; use moving_kind=payload with independently visible "
        "tool_point_uv. Placement tracks the payload relative to its target, accounting for grasp "
        "offset. Missing attachment or ambiguous post-release support calls for informative observation "
        "or correction, not repeated completion claims. Keep a stable hypothesis_id; changing its name "
        "or relearning an old response is not progress. Stage labels are hypotheses, not task success. "
        "DIRECT: output field_intent=null and exactly five unchanged normalized actions "
        "[dx,dy,dz,rx,ry,rz,grip] in [-1,1]. Translation 1 requests 0.05 m delta; rotation 1 requests "
        "0.5 rad world axis-angle delta; +1 grip closes, -1 opens and 0 means zero incremental motion. "
        "Commands are at 20 Hz relative to actual measured pose, so actual displacement may be smaller. "
        "Five actions last 0.25 s; allow closure to settle and test attachment before transport. "
        "There is no task-specific automatic upward recovery. History retains two complete recent "
        "request turns and compact observable block memory; full raw records remain local. Return "
        "strict JSON only, without a task-completion flag or invented success/clearance claim."
    )


def _runtime_schema(schema):
    """The shared validator predates integer schemas; preserve strict integer checks below."""
    if isinstance(schema, dict):
        result = {key: _runtime_schema(value) for key, value in schema.items()}
        if result.get("type") == "integer":
            result["type"] = "number"
        return result
    if isinstance(schema, list):
        return [_runtime_schema(value) for value in schema]
    return schema


class VisualFieldSubscriptionClientV4(VisualFieldSubscriptionClientV2):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        # Reuse audited OAuth/history handling, replacing the old method contract entirely.
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v4", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def validate_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        field = data["controller"] == "visual_field"
        if (data["field_intent"] is not None) != field or (data["direct_actions"] is None) != field:
            raise SchemaError("Only the selected V4 controller branch may be populated")
        if any(not data[key].strip() for key in ("target_id", "hypothesis_id", "reason")):
            raise SchemaError("Intent identities and reason must be nonempty")
        external = data["visual_references"]["external"]
        for ref in data["visual_references"].values():
            if ref is not None and (not ref["target_id"].strip() or ref["target_id"] != data["target_id"]):
                raise SchemaError("References must retain the response target identity")
        if external is not None and (not external["moving_id"].strip()
                                    or external["moving_id"] == external["target_id"]):
            raise SchemaError("Moving and target landmarks must be nonempty and distinct")
        if field:
            intent = data["field_intent"]
            if data["phase"] not in FIELD_PHASES or external is None:
                raise SchemaError("Field requires noncontact phase and an observable external reference")
            if data["phase"] == "transport" and (external["moving_kind"] != "payload"
                                                    or external["tool_point_uv"] is None):
                raise SchemaError("Field transport requires independent payload/tool references")
            if abs(math.sqrt(sum(x*x for x in intent["orientation_target_xyzw"]))-1) > .001:
                raise SchemaError("Orientation target must be a unit quaternion")
            if type(intent["max_actions"]) is not int:
                raise SchemaError("Field max_actions must be an integer")
            regions = intent["obstacles"] + intent["unknown_regions"]
            ids = [box["id"] for box in regions]
            if any(not ident.strip() for ident in ids) or len(ids) != len(set(ids)):
                raise SchemaError("Hypothesis box IDs must be nonempty and unique across both lists")
        result = copy.deepcopy(data)
        if field:
            # Only the execution representation changes. The base subscription
            # client keeps raw model output in logs and conversation history.
            # No sorting, clipping, dropping, or uncertainty reduction occurs.
            for name in ("obstacles", "unknown_regions"):
                result["field_intent"][name] = [
                    {"id": box["id"],
                     "lower_offset_m": [center-half for center, half in
                                        zip(box["center_offset_m"], box["half_extents_m"])],
                     "upper_offset_m": [center+half for center, half in
                                        zip(box["center_offset_m"], box["half_extents_m"])],
                     "uncertainty_m": box["uncertainty_m"]}
                    for box in data["field_intent"][name]
                ]
        return result

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_VERSION,
                      text_history="recent_two_complete_turns_plus_observable_block_memory")
        return result
