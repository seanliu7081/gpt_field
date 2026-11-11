"""Astra owns metric actions; optional measurements never supply missing poses."""
from __future__ import annotations

from copy import deepcopy
import json
import math

from ..astra import SchemaError
from ..agent_speedup_v2_user_check_fragments import execution_schema, validate_execution, validate_route


class ProtocolRepairError(SchemaError):
    def __init__(self, message, raw_plan=None):
        super().__init__(message)
        draft = "" if raw_plan is None else json.dumps(raw_plan, ensure_ascii=False, default=str)
        self.evidence = {"reason": str(message), "source": "rejected_model_proposal",
                         "draft_json": draft[:12000], "draft_truncated": len(draft) > 12000,
                         "no_candidate_executed": True}


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def _string(limit=64):
    return {"type": "string", "minLength": 1, "maxLength": limit}


def _vector(size):
    return {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}


def harness_schema():
    motion = _object({
        "skill": {"type": "string", "enum": ["move_to_pose", "move_relative", "approach"]},
        "xyz_m": _vector(3), "quaternion_wxyz": _nullable(_vector(4)),
        "profile": {"type": "string", "enum": ["transit", "contact"]},
    })
    grip = _object({"skill": {"type": "string", "enum": ["open", "close"]}})
    wait = _object({"skill": {"type": "string", "enum": ["wait"]},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 120}})
    checkpoint = _nullable(_object({
        "state": {"type": "string", "enum": ["observe", "held", "released"]},
        "object": _nullable(_string(64)),
    }))
    segment = _object({"object": _nullable(_string(64)), "checkpoint": checkpoint,
        "candidates": {"type": "array", "items": {"anyOf": [motion, grip, wait]},
                       "minItems": 1, "maxItems": 3}})
    measurement = _object({"id": _string(64),
        "kind": {"type": "string", "enum": ["segment", "point", "surface"]},
        "query": _nullable(_string(160)), "view": {"type": "string", "enum": ["external", "wrist"]},
        "region": _nullable(_vector(4)), "pixel": _nullable(_vector(2))})
    route = deepcopy(execution_schema()["properties"]["segments"]["items"]["properties"]["route"])
    return _object({
        "status": {"type": "string", "enum": ["execute", "measure", "finish", "blocked"]},
        "why": _string(500), "obstacle_map": route,
        "segments": {"type": "array", "items": segment, "minItems": 0, "maxItems": 24},
        "measurements": {"type": "array", "items": measurement, "minItems": 0, "maxItems": 8},
    })


def _conform(value, schema, path="response"):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _conform(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"invalid {path}")
    kind = schema["type"]
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str) and bool(value.strip()),
             "number": lambda: type(value) in (int, float) and math.isfinite(value),
             "integer": lambda: type(value) is int,
             "null": lambda: value is None}[kind]()
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise ValueError(f"invalid {path}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"missing or unexpected fields in {path}")
        for key, item in value.items():
            _conform(item, schema["properties"][key], f"{path}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 100):
            raise ValueError(f"invalid length of {path}")
        for index, item in enumerate(value):
            _conform(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", len(value)):
            raise ValueError(f"invalid length of {path}")
    elif kind in ("number", "integer"):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"invalid range for {path}")


def validate_contract(data, *, frame_id=None, image_shape=None):
    """Normalize one complete response before any prefix can execute."""
    try:
        _conform(data, harness_schema())
        result = deepcopy(data)
        status = result["status"]
        if (status == "execute") != bool(result["segments"]):
            raise ValueError("only execute has a nonempty segment queue")
        if (status == "measure") != bool(result["measurements"]):
            raise ValueError("only measure has nonempty measurements")
        if status != "execute" and result["obstacle_map"] is not None:
            raise ValueError("obstacle_map updates belong to execute responses")
        result["obstacle_map"] = validate_route(result["obstacle_map"])
        for index, segment in enumerate(result["segments"]):
            segment["id"] = f"segment_{index+1:02d}"
            check = segment["checkpoint"]
            segment["checkpoint"] = "none" if check is None else check["state"]
            segment["checkpoint_object"] = None if check is None else check["object"]
            families, grippers = set(), set()
            for number, compact in enumerate(segment["candidates"]):
                skill = compact["skill"]
                if skill in ("open", "close"):
                    candidate = {"skill": "set_gripper", "xyz_m": None, "quaternion_wxyz": None,
                        "gripper": skill, "profile": "contact", "max_steps": 16 if skill == "open" else 20}
                elif skill == "wait":
                    candidate = {"skill": "wait", "xyz_m": None, "quaternion_wxyz": None,
                        "gripper": "keep", "profile": "contact", "max_steps": compact["steps"]}
                else:
                    candidate = {**compact, "gripper": "keep",
                                 "max_steps": 120 if compact["profile"] == "transit" else 80}
                candidate.update(observation_after=False, route=None)
                normalized = validate_execution({"segments": [candidate]})["segments"][0]
                segment["candidates"][number] = normalized
                families.add(normalized["skill"] if normalized["skill"] in ("set_gripper", "wait") else "motion")
                grippers.add(normalized["gripper"])
            if len(families) != 1 or len(grippers) != 1:
                raise ValueError("alternatives must preserve motion/gripper/wait and open/close semantics")
        ids = [item["id"] for item in result["measurements"]]
        if len(ids) != len(set(ids)):
            raise ValueError("measurement ids must be unique within a response")
        if ids and (type(frame_id) is not int or frame_id < 0):
            raise ValueError("measurements require the current nonnegative frame_id")
        for item in result["measurements"]:
            if item["kind"] == "segment" and item["query"] is None:
                raise ValueError("segment measurement requires a query")
            if item["kind"] == "surface" and all(item[key] is None for key in ("query", "region", "pixel")):
                raise ValueError("surface measurement requires query, region or pixel")
            if item["kind"] == "point" and item["pixel"] is None:
                raise ValueError("point measurement requires an explicit pixel")
            if item["kind"] == "segment" and item["pixel"] is not None:
                raise ValueError("segment measurement uses a region, not a point pixel")
            region, pixel = item["region"], item["pixel"]
            if region is not None:
                x0, y0, x1, y1 = region
                if not (0 <= x0 < x1 and 0 <= y0 < y1):
                    raise ValueError("measurement region must be an ordered positive image box")
                if image_shape is not None and (x1 > image_shape[1] or y1 > image_shape[0]):
                    raise ValueError("measurement region extends outside the source image")
            if pixel is not None:
                if min(pixel) < 0 or (image_shape is not None and
                    not (pixel[0] <= image_shape[1]-1 and pixel[1] <= image_shape[0]-1)):
                    raise ValueError("measurement pixel extends outside the source image")
                if region is not None and not (region[0] <= pixel[0] < region[2] and region[1] <= pixel[1] < region[3]):
                    raise ValueError("measurement pixel is outside its requested region")
            item["frame_id"] = frame_id
        return result
    except ProtocolRepairError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ProtocolRepairError(f"V3 contract: {exc}", data) from None
