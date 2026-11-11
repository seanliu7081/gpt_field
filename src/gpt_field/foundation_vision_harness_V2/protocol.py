"""Compact model decisions normalized into measured execution contracts.

Bookkeeping belongs here: stable per-response IDs, frame lineage, primitive
budgets, contact defaults and checkpoint references. The model still chooses
objects, contact strategy, geometry references, motion and alternatives.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json

from ..astra import SchemaError
from ..agent_speedup_v2_user_check_fragments import validate_execution
from ..foundation_vision_harness_V1.client import (
    _object, _nullable, _string, _vector, _conform,
    harness_schema as _v1_schema, validate_contract as _validate_targets,
)


class ProtocolRepairError(SchemaError):
    """A completed model proposal needs semantic repair, not episode abort."""

    def __init__(self, message, raw_plan=None):
        super().__init__(message)
        draft = "" if raw_plan is None else json.dumps(raw_plan, ensure_ascii=False, default=str)
        self.evidence = {"reason": str(message), "source": "rejected_model_proposal",
                         "draft_json": draft[:12000], "draft_truncated": len(draft) > 12000,
                         "no_candidate_executed": True}


def grounding_schema():
    return _nullable(_object({
        "region": _string(32), "view": {"type": "string", "enum": ["external", "wrist"]},
        "pixel": _vector(2), "offset_hand_m": _vector(3), "standoff_m": {"type": "number"},
        "reference": {"type": "string", "enum": ["tool", "payload_support"]},
    }))


def harness_schema():
    """Only the compact protocol is sent; archived expanded plans remain readable."""
    motion = _object({
        "skill": {"type": "string", "enum": ["move_to_pose", "move_relative", "approach"]},
        "xyz_m": _nullable(_vector(3)), "quaternion_wxyz": _nullable(_vector(4)),
        "profile": {"type": "string", "enum": ["transit", "contact"]},
    })
    grip = _object({"skill": {"type": "string", "enum": ["open", "close"]}})
    wait = _object({"skill": {"type": "string", "enum": ["wait"]},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 120}})
    checkpoint = _nullable(_object({
        "state": {"type": "string", "enum": ["observe", "held", "released"]},
        "object": _nullable(_string(32)),
    }))
    segment = _object({
        "object": _nullable(_string(32)), "anchor": _nullable(_string(32)),
        "contact_mode": {"type": "string", "enum": ["free", "grasp", "articulated", "articulated_tool", "place"]},
        "keep_upright": {"type": "boolean"}, "checkpoint": checkpoint,
        "grounding": grounding_schema(),
        "candidates": {"type": "array", "items": {"anyOf": [motion, grip, wait]}, "minItems": 1, "maxItems": 3},
    })
    targets = deepcopy(_v1_schema()["properties"]["targets"])
    targets["items"]["properties"].pop("frame_id")
    targets["items"]["required"].remove("frame_id")
    return _object({
        "status": {"type": "string", "enum": ["plan", "observe", "finish", "blocked"]},
        "why": _string(500), "targets": targets,
        "segments": {"type": "array", "items": segment, "maxItems": 24},
        "completion_conditions": {"type": "array", "items": _string(160), "maxItems": 12},
    })


def _expanded_schema():
    grounding = grounding_schema()
    grounding["anyOf"][0]["properties"]["frame_id"] = {"type": "integer", "minimum": 0}
    grounding["anyOf"][0]["required"].append("frame_id")
    candidate = _object({
        "skill": {"type": "string", "enum": ["move_to_pose", "move_relative", "approach", "set_gripper", "wait"]},
        "xyz_m": _nullable(_vector(3)), "quaternion_wxyz": _nullable(_vector(4)),
        "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        "profile": {"type": "string", "enum": ["transit", "contact"]},
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 120},
    })
    segment = _object({
        "id": _string(48), "object": _nullable(_string(32)), "anchor": _nullable(_string(32)),
        "contact_mode": {"type": "string", "enum": ["free", "grasp", "articulated", "place"]},
        "contact_parts": {"type": "array", "items": {"type": "string", "enum": ["finger", "pad", "tool", "payload"]}, "maxItems": 4},
        "keep_upright": {"type": "boolean"},
        "checkpoint": {"type": "string", "enum": ["none", "observe", "held", "released"]},
        "checkpoint_object": _nullable(_string(32)), "grounding": grounding,
        "candidates": {"type": "array", "items": candidate, "minItems": 1, "maxItems": 3},
    })
    return _object({
        "status": {"type": "string", "enum": ["plan", "observe", "finish", "blocked"]},
        "why": _string(500), "targets": deepcopy(_v1_schema()["properties"]["targets"]),
        "segments": {"type": "array", "items": segment, "maxItems": 24},
        "completion_conditions": {"type": "array", "items": _string(160), "maxItems": 12},
    })


def _expand(data, frame_id):
    if not isinstance(data, dict):
        raise ValueError("proposal must be an object")
    result = deepcopy(data)
    compact = (any("id" not in s for s in result.get("segments", []))
               or any("frame_id" not in t for t in result.get("targets", [])))
    if compact:
        _conform(result, harness_schema())
        if result["targets"] and frame_id is None:
            raise ValueError("compact targets require the current frame_id")
        for target in result["targets"]:
            target["frame_id"] = frame_id
        for index, segment in enumerate(result["segments"]):
            segment["id"] = f"segment_{index+1:02d}"
            check = segment["checkpoint"]
            segment["checkpoint"] = "none" if check is None else check["state"]
            segment["checkpoint_object"] = None if check is None else check["object"]
            mode = segment["contact_mode"]
            segment["contact_parts"] = {"free": [], "grasp": ["finger", "pad"],
                "articulated": ["finger", "pad"], "articulated_tool": ["tool"],
                "place": ["finger", "pad", "payload"]}[mode]
            if mode == "articulated_tool":
                segment["contact_mode"] = "articulated"
            for candidate in segment["candidates"]:
                skill = candidate["skill"]
                if skill in {"open", "close"}:
                    candidate.update(skill="set_gripper", xyz_m=None, quaternion_wxyz=None,
                        gripper=skill, profile="contact", max_steps=16 if skill == "open" else 20)
                elif skill == "wait":
                    candidate.update(xyz_m=None, quaternion_wxyz=None, gripper="keep", profile="contact",
                                     max_steps=candidate.pop("steps"))
                else:
                    candidate.update(gripper="keep", max_steps=120 if candidate["profile"] == "transit" else 80)
    else:
        for segment in result.get("segments", []):
            segment.setdefault("checkpoint_object", None)
            segment.setdefault("grounding", None)
    for segment in result["segments"]:
        grounding = segment["grounding"]
        if grounding is not None:
            grounding.setdefault("frame_id", frame_id)
            if frame_id is not None and grounding["frame_id"] != frame_id:
                raise ValueError("grounding pixel frame differs from the request frame")
    _conform(result, _expanded_schema())
    return result


def _normalize_robot_parent_hints(targets, known):
    """Discard redundant robot-only hierarchy, never material-object lineage.

    Robot masks are independent self-observation queries. Their parent hints do
    not grant contact, remove geometry or identify a carried object. Resolve the
    complete chain first so unknown, external and cyclic links remain errors.
    """
    records = {**known, **{target["id"]: target for target in targets}}
    redundant = []
    for target in targets:
        if target["role"] != "robot" or target["parent_id"] is None:
            continue
        seen, parent = {target["id"]}, target["parent_id"]
        while parent is not None:
            ancestor = records.get(parent)
            if parent in seen or not isinstance(ancestor, Mapping) or ancestor.get("role") != "robot":
                break
            seen.add(parent)
            parent = ancestor.get("parent_id")
        else:
            redundant.append(target)
    for target in redundant:
        target["parent_id"] = None


def validate_contract(data, *, known_ids=(), frame_id=None, image_shape=None,
                      held_object_id=None, released_object_id=None):
    """Normalize the full proposal atomically; return no executable prefix on error."""
    try:
        result = _expand(data, frame_id)
        known = dict(known_ids) if isinstance(known_ids, Mapping) else {key: {} for key in known_ids}
        _normalize_robot_parent_hints(result["targets"], known)
        _validate_targets({"status": "observe", "why": result["why"], "targets": result["targets"],
            "operations": [], "completion_conditions": result["completion_conditions"]},
            known_ids=known, frame_id=frame_id, image_shape=image_shape)
        records = {**known, **{target["id"]: target for target in result["targets"]}}
        if (result["status"] == "plan") != bool(result["segments"]):
            raise ValueError("only status=plan has a nonempty segment queue")
        ids = [s["id"] for s in result["segments"]]
        if len(set(ids)) != len(ids):
            raise ValueError("segment ids must be unique within a response")
        # These references describe plan dependencies, never actual grasp proof.
        held, released = held_object_id, released_object_id
        for segment in result["segments"]:
            for key in ("object", "anchor", "checkpoint_object"):
                ident = segment[key]
                if ident is not None and ident not in records:
                    raise ValueError(f"segment {key} requires a known target")
                if ident is not None and records[ident].get("role") == "robot":
                    raise ValueError("robot segmentation is not an action target")
            mode, parts, source = segment["contact_mode"], segment["contact_parts"], segment["object"]
            if len(parts) != len(set(parts)):
                raise ValueError("contact_parts must be unique")
            if mode == "free" and parts:
                raise ValueError("free segment cannot permit contact")
            if mode != "free" and source is None:
                raise ValueError("contact mode requires an identified object")
            if "tool" in parts and mode != "articulated":
                raise ValueError("tool contact requires an articulated contact strategy")
            if "payload" in parts and (mode != "place" or segment["anchor"] is None):
                raise ValueError("payload contact requires a placement support anchor")
            if mode == "articulated" and segment["checkpoint"] == "held":
                raise ValueError("articulated fixture is not a carried payload")
            families, grippers = set(), set()
            for index, candidate in enumerate(segment["candidates"]):
                projected_position = (candidate["skill"] == "move_to_pose" and candidate["xyz_m"] is None
                                      and segment["grounding"] is not None)
                checked = {**candidate, "xyz_m": [0., 0., 0.] if projected_position else candidate["xyz_m"]}
                normalized = validate_execution({"segments": [{**checked,
                    "observation_after": False, "route": None}]})["segments"][0]
                if projected_position:
                    # Validation-only placeholder. The runtime must resolve this
                    # null from the explicit material reference before dispatch.
                    normalized["xyz_m"] = None
                segment["candidates"][index] = {key: normalized[key] for key in candidate}
                families.add(candidate["skill"] if candidate["skill"] in ("set_gripper", "wait") else "motion")
                grippers.add(candidate["gripper"])
            if len(families) != 1 or len(grippers) != 1:
                raise ValueError("alternatives must preserve motion/gripper/wait and gripper semantics")
            candidate = segment["candidates"][0]
            if candidate["skill"] == "set_gripper":
                if candidate["gripper"] == "close" and mode != "articulated":
                    if source is not None and held is not None and held != source:
                        raise ValueError("close changes payload identity without a preceding release")
                    # A free gripper close can be an empty-hand action. It does
                    # not invent a payload identity or attachment proof.
                    held, released = source or held, None
                elif candidate["gripper"] == "open":
                    if held is not None and source is not None and held != source:
                        raise ValueError("release object conflicts with the preceding held object")
                    released, held = source or held, None
            kind = segment["checkpoint"]
            if kind in {"held", "released"}:
                expected = held if kind == "held" else released
                explicit = segment["checkpoint_object"]
                identity = explicit or expected or source
                if identity is None:
                    raise ValueError(f"{kind} checkpoint object is unresolved; name it or provide a unique preceding {kind} transition")
                if expected is not None and explicit is not None and expected != explicit:
                    raise ValueError(f"{kind} checkpoint object conflicts with the preceding transition")
                if identity not in records or records[identity].get("role") == "robot":
                    raise ValueError(f"{kind} checkpoint requires a known nonrobot object")
                segment["checkpoint_object"] = identity
            elif segment["checkpoint_object"] is not None:
                raise ValueError("only held/released checkpoints have an object reference")
            grounding = segment["grounding"]
            if grounding is not None:
                region = records.get(grounding["region"])
                if region is None or region.get("role") == "robot":
                    raise ValueError("grounding requires a known material region or support")
                if any(c["skill"] != "move_to_pose" for c in segment["candidates"]):
                    raise ValueError("grounding requires absolute move_to_pose candidates; relative motion keeps its explicit displacement")
                if not 0. <= grounding["standoff_m"] <= .20:
                    raise ValueError("grounding standoff must be within 0 to 0.20 m")
                if image_shape is not None:
                    u, v = grounding["pixel"]
                    if not (0 <= u < image_shape[1] and 0 <= v < image_shape[0]):
                        raise ValueError("grounding pixel lies outside the current image")
                if grounding["reference"] == "payload_support" and mode != "place":
                    raise ValueError("payload support grounding requires place contact mode")
        return result
    except ProtocolRepairError:
        raise
    except (SchemaError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ProtocolRepairError(f"candidate queue: {exc}", data) from None


def compact_plan(expanded):
    """Render a normalized plan for offline protocol size comparison/review.

    Local max_steps are deliberately replaced by documented executor defaults;
    this helper never executes or sends a plan.
    """
    result = deepcopy(expanded)
    for target in result["targets"]:
        target.pop("frame_id", None)
    for segment in result["segments"]:
        segment.pop("id", None)
        parts = segment.pop("contact_parts")
        if "tool" in parts:
            if set(parts) != {"tool"}:
                raise ValueError("compact articulated_tool does not imply extra finger contact")
            segment["contact_mode"] = "articulated_tool"
        kind = segment["checkpoint"]
        identity = segment.pop("checkpoint_object", None)
        segment["checkpoint"] = None if kind == "none" else {"state": kind, "object": identity}
        segment.setdefault("grounding", None)
        if segment["grounding"] is not None:
            segment["grounding"].pop("frame_id", None)
        for index, candidate in enumerate(segment["candidates"]):
            if candidate["skill"] == "set_gripper":
                compact = {"skill": candidate["gripper"]}
            elif candidate["skill"] == "wait":
                compact = {"skill": "wait", "steps": candidate["max_steps"]}
            else:
                compact = {key: candidate[key] for key in ("skill", "xyz_m", "quaternion_wxyz", "profile")}
            segment["candidates"][index] = compact
    return result
