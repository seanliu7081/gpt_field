"""Explicit persistent RGB references and generic model-directed gripper settling.

Retain requests name existing correspondences; only the live monitor can resolve
them. The wire contains neither guessed replacement coordinates nor simulator
state. V5's single semantic repair and measured-orientation conversion are reused.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v5 import (
    VisualFieldSubscriptionClientV5, _validate_region, output_schema as schema_v5,
)

VERSION = "visual-field-rgb-joints-v6-persistent-references-axis-gripper-v1"
HISTORY_VERSION = "visual-v6-persistent-reference-recent-two-turns-v1"
FIELD_PHASES = ("approach", "transport", "align", "release", "recover")


def output_schema(controller="hybrid"):
    if controller != "hybrid":
        raise ValueError("Visual v6 is zero-shot Hybrid only")
    result = schema_v5()
    props = result["properties"]
    identifier = {"type": "string", "minLength": 1, "maxLength": 80}
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    update = {"type": "string", "enum": ["seed", "retain"]}
    external_seed = _object({
        "moving_point_uv": _array(2, 0, 1), "moving_roi_uv": nullable(_array(4, 0, 1)),
        "target_anchor_uv": _array(2, 0, 1), "target_roi_uv": nullable(_array(4, 0, 1)),
        "tool_point_uv": nullable(_array(2, 0, 1)), "tool_roi_uv": nullable(_array(4, 0, 1)),
    })
    external = _object({
        "update": copy.deepcopy(update), "moving_id": copy.deepcopy(identifier),
        "moving_kind": {"type": "string", "enum": ["tool", "payload"]},
        "target_id": copy.deepcopy(identifier), "goal_offset_uv": _array(2, -1, 1),
        "seed": nullable(external_seed),
    })
    wrist = _object({
        "update": copy.deepcopy(update), "target_id": copy.deepcopy(identifier),
        "desired_target_uv": _array(2, 0, 1),
        "seed": nullable(_object({"target_anchor_uv": _array(2, 0, 1),
                                  "target_roi_uv": nullable(_array(4, 0, 1))})),
    })
    props["visual_references"] = _object({"external": nullable(external), "wrist": nullable(wrist)})
    intent = props["field_intent"]["anyOf"][0]
    intent["properties"]["motion_axes"] = {"type": "string", "enum": ["xyz", "xy", "z"]}
    intent["required"].append("motion_axes")
    props["controller"]["enum"] = ["visual_field", "direct", "gripper"]
    props["gripper_intent"] = nullable(_object({
        "command": {"type": "string", "enum": ["open", "close"]},
        "max_actions": {"type": "integer", "minimum": 10, "maximum": 30},
    }))
    result["required"].append("gripper_intent")
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v6 is zero-shot Hybrid only")
    return (
        f"Task: {task.prompt}. ZERO-SHOT: no external demonstration. Use external/wrist RGB, measured "
        "EEF xyz/axis-angle/two-finger state, seven arm joint positions/velocities and live feedback. "
        "No depth, supplied camera/environment calibration, object truth, contacts, reward or evaluator "
        "success is provided. Preserve object identities. Complete the task without unintended "
        "robot/payload collision or obstacle displacement. Necessary finger contact with the task "
        "object/handle and intended support are allowed. Account for fingers, payload and arm extent. "
        "Image overlap is not metric clearance or task success. "
        "ROUTING: visual_field handles substantial noncontact spatial movement, including approach, "
        "transport, alignment and the last bounded approach to support. Direct supplies exactly five "
        "unchanged actions for complex contact, a short attachment test or reorientation. Gripper "
        "supplies one explicit open/close settling intent at the current pose. Select exactly one "
        "populated branch: field_intent, direct_actions or gripper_intent; the other two must be null. "
        "A tracking refresh is not proof of physical misalignment or a need to retreat. Use current "
        "views to choose progress; avoid repeated calibration, unchanged tiny approaches and holds. "
        "FIELD: goal_offset_m is a robot/world-axis EEF translation from the CURRENT measured EEF, "
        "within +/-0.08 m per axis. A 33-cubed grid at 0.005 m spacing represents local goal distance; "
        "the controller plans around your obstacle hypotheses and follows the route from measured "
        "poses without calling you each action. orientation_command=hold and gripper_command=hold. "
        "motion_axes=xyz permits a spatial route; xy holds the initial height and requires zero goal "
        "z offset; z holds the initial XY and requires zero goal x/y offsets. Locked coordinates are "
        "measured-pose constraints, not additional guessed target coordinates. Obstacles and measured "
        "drift still constrain execution; the controller must not leave an axis corridor to force a route. "
        "Use max_actions=80 and command_limit_m=0.012..0.02 in visibly clear free space, shorter "
        "bounded intents and 0.003..0.006 near interaction. max_actions is 10..120; max_travel_m is at "
        "most 0.16 m. Arrival, contradictory observations or budgets end reuse early. Supply tolerance, "
        "tool/payload radius and uncertainty honestly. The allowed Field phases are approach, "
        "transport, align, release and recover; Field release requires motion_axes=z and HOLDS grip. "
        "PLACEMENT: first align the PAYLOAD with the intended support using both views and the observed "
        "grasp offset. Then preserve horizontal alignment while using a bounded z Field to approach "
        "support. Use Field align/xy for needed horizontal correction FIRST. Direct phase release "
        "must have zero XYZ translation in all five actions; it can reorient if needed, with "
        "opening/closing handled by the gripper branch. Do not keep adding lateral Direct "
        "corrections during descent after horizontal "
        "alignment appears adequate. If new evidence contradicts alignment, explicitly return to "
        "align and revise the goal. Field never opens the fingers. Request gripper open only when "
        "the current views justify release; reassess payload alignment and support after separation. "
        "Visible touching or partial overlap alone does not establish centered/stable placement. "
        "GEOMETRY: obstacles/unknown_regions are world-axis-aligned RGB hypotheses with stable IDs, "
        "center_offset_m from this SAME measured EEF, positive half_extents_m and uncertainty_m. "
        "Local planning inflates them by tool/payload radius and uncertainty. Preserve static world "
        "positions when the EEF moves. Do not remove obstacles, lower uncertainty or invent solid "
        "walls for every occluded pixel to force an answer. A feasible route is not a collision certificate. "
        "PERSISTENT REFERENCES: normalized UV uses top-left origin, u right and v down. Each external "
        "or wrist reference explicitly selects update=seed or retain. Seed requires a seed object "
        "with currently visible physical points and nullable tight [u0,v0,u1,v1] ROIs; every nonnull "
        "ROI has positive area and contains its point. Prefer a textured rigid part around the seed "
        "over a tiny edge. External moving_point_uv is on the actual tool/payload; target_anchor_uv "
        "is on a separate reference landmark. tool_point_uv independently observes the gripper when "
        "moving_kind=payload. Retain requires seed=null and EXACT existing identities/moving_kind; "
        "the monitor reuses its currently tracked points and features, not re-guessed coordinates. "
        "Choose retain while those correspondences remain reliable. Seed only when you intentionally "
        "change a physical correspondence or recover an unavailable track; it resets related evidence. "
        "Retain cannot resurrect a lost track. External goal_offset_uv explicitly specifies desired "
        "moving-point projection minus target-anchor projection, including on retain; it may be a "
        "virtual offset. Wrist desired_target_uv is the desired bearing of its target landmark. "
        "Top-level target_id names the current interaction/navigation target. Wrist target_id must "
        "match it; Field external target_id must also match it. During Direct/gripper interaction, "
        "the external stationary anchor may be a distinct support/background landmark while the "
        "moving payload is the grasp target. Never use that payload itself as its stationary anchor. "
        "A seeded payload reference requires independent tool_point_uv; a retained one requires its "
        "reliable existing tool track. Keep payload/tool identities through a destination change; "
        "update or omit the wrist reference independently. Field requires external reference; wrist "
        "may be null and never requires a tool footprint. Null references do not certify clearance. "
        "GRIPPER: only phases grasp/release permit controller=gripper. gripper_intent contains "
        "command=open|close and max_actions=10..30; local measured-pose hold repeats that explicit "
        "command until finger settling or the bound. This does not infer a grasp, destination or "
        "completion. Model waiting does not advance physics. After closing, use a short Direct test "
        "movement with independent payload/tool references before Field transport. Closure alone "
        "does not establish attachment; lost or contradicted co-motion needs a targeted test/correction. "
        "DIRECT: exactly five normalized [dx,dy,dz,rx,ry,rz,grip] actions in [-1,1], at 20 Hz. "
        "Translation 1 requests 0.05 m relative to measured pose; rotation 1 requests 0.5 rad world "
        "axis-angle. Grip +1 closes, -1 opens and 0 holds remembered intent. Five actions last 0.25 s; "
        "actual motion can be smaller. Return strict JSON with concise reasons and no completion flag. "
        "Two recent full turns plus observable block memory are supplied; raw records remain local."
    )


class VisualFieldSubscriptionClientV6(VisualFieldSubscriptionClientV5):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v6", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        """V5 repair hook; V6 shape/role checks never resolve live references."""
        _validate(data, _runtime_schema(output_schema()))
        controller = data["controller"]
        branch = {"visual_field": "field_intent", "direct": "direct_actions", "gripper": "gripper_intent"}[controller]
        if any((data[name] is not None) != (name == branch)
               for name in ("field_intent", "direct_actions", "gripper_intent")):
            raise SchemaError("Only the selected V6 controller branch may be populated")
        if any(not data[key].strip() for key in ("target_id", "hypothesis_id", "reason")):
            raise SchemaError("Intent identities and reason must be nonempty")
        field = controller == "visual_field"
        references = data["visual_references"]
        external = references["external"]
        for view, ref in references.items():
            if ref is None:
                continue
            if not ref["target_id"].strip():
                raise SchemaError("Reference landmark identities must be nonempty")
            if ref["target_id"] != data["target_id"] and (view == "wrist" or field):
                raise SchemaError("Wrist and Field references must retain the response target identity")
            if (ref["seed"] is not None) != (ref["update"] == "seed"):
                raise SchemaError("Reference seed mode requires seed data; retain requires seed=null")
            if ref["seed"] is not None:
                _validate_region(ref["seed"], "target_roi_uv", "target_anchor_uv")
        if external is not None:
            if not external["moving_id"].strip() or external["moving_id"] == external["target_id"]:
                raise SchemaError("Moving and stationary landmarks must be nonempty and distinct")
            seed = external["seed"]
            if seed is not None:
                _validate_region(seed, "moving_roi_uv", "moving_point_uv")
                _validate_region(seed, "tool_roi_uv", "tool_point_uv")
                if external["moving_kind"] == "payload" and seed["tool_point_uv"] is None:
                    raise SchemaError("Seeded payload observation requires an independent visible tool reference")
        if controller == "gripper":
            if data["phase"] not in ("grasp", "release"):
                raise SchemaError("Gripper settling is only available for grasp or release")
            if type(data["gripper_intent"]["max_actions"]) is not int:
                raise SchemaError("Gripper max_actions must be an integer")
        if controller == "direct" and data["phase"] == "release":
            if any(any(action[:3]) for action in data["direct_actions"]):
                raise SchemaError("Direct release actions must have zero XYZ; use Field align/xy then release/z")
        if field:
            intent = data["field_intent"]
            if data["phase"] not in FIELD_PHASES or external is None:
                raise SchemaError("Field requires a spatial-motion phase and an external reference")
            if data["phase"] == "transport" and external["moving_kind"] != "payload":
                raise SchemaError("Field transport requires independently observed payload/tool references")
            if type(intent["max_actions"]) is not int:
                raise SchemaError("Field max_actions must be an integer")
            axes, delta = intent["motion_axes"], intent["goal_offset_m"]
            if (axes == "xy" and delta[2] != 0) or (axes == "z" and any(delta[:2])):
                raise SchemaError("Field goal offsets must be zero on locked axes")
            if data["phase"] == "release" and axes != "z":
                raise SchemaError("Field release approach requires z-only motion and held grip")
            ids = [box["id"] for group in ("obstacles", "unknown_regions") for box in intent[group]]
            if any(not ident.strip() for ident in ids) or len(ids) != len(set(ids)):
                raise SchemaError("Hypothesis box IDs must be nonempty and unique across both lists")
        result = copy.deepcopy(data)
        if field:
            for name in ("obstacles", "unknown_regions"):
                result["field_intent"][name] = [
                    {"id": box["id"],
                     "lower_offset_m": [c-h for c, h in zip(box["center_offset_m"], box["half_extents_m"])],
                     "upper_offset_m": [c+h for c, h in zip(box["center_offset_m"], box["half_extents_m"])],
                     "uncertainty_m": box["uncertainty_m"]}
                    for box in data["field_intent"][name]]
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
