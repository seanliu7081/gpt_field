"""A single visible-object inventory feeds all internal scene geometry roles.

The legacy projection below validates inherited non-scene contracts. Its list
caps apply ONLY to that compatibility check. Returned inventory/geometry are
complete; the policy's persistent inventory materializes every remembered entry.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v8 import _ordered_roi
from .visual_field_subscription_v15 import (
    VisualFieldSubscriptionClientV15, output_schema as schema_v15,
    shared_guidance as guidance_v15, geometry_semantics,
)

VERSION = "visual-field-rgb-joints-v16-persistent-scene-inventory-v1"
HISTORY_VERSION = "visual-v16-inventory-recent-two-turns-v1"


def output_schema(controller="hybrid"):
    result = schema_v15(controller)
    props = result["properties"]
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    geometry = _object({"center_offset_m": _array(3, -.5, .5),
        "half_extents_m": _array(3, .0001, .5),
        "uncertainty_m": {"type": "number", "minimum": 0, "maximum": .1}})
    props["scene_inventory"] = {"type": "array", "maxItems": 12, "items": _object({
        "id": {"type": "string", "minLength": 1, "maxLength": 80},
        "external_roi_uv": nullable(_array(4, 0, 1)),
        "wrist_roi_uv": nullable(_array(4, 0, 1)), "geometry": nullable(geometry)})}
    props.pop("target_geometry")
    props["target_contact"] = {"type": "string", "enum": ["none", "fingers", "payload_support"]}
    result["required"].remove("target_geometry")
    result["required"].extend(("scene_inventory", "target_contact"))
    for obj, names in ((props["motion_context"], ("nearby_obstacles", "obstacle_assessment")),
                       (props["field_intent"]["anyOf"][0], ("obstacles",))):
        for name in names:
            obj["properties"].pop(name)
            obj["required"].remove(name)
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v15(task, demonstration)
    begin, end = text.index("GEOMETRY."), text.index("\n\nRECOVERY.")
    text = text[:begin] + """SCENE INVENTORY. Every response lists ALL currently visible nonrobot physical objects, including the target, distractors and fixtures, regardless of guessed relevance to the current goal. Do not omit an object because it lies outside the wrist crop or planned corridor. Use stable physical IDs and whole-object current external_roi_uv and/or wrist_roi_uv, null only in a view where unavailable. At least one ROI must be current for each listed entry; small texture patches are not object extents. Geometry is one approximate world-axis {center_offset_m,half_extents_m,uncertainty_m} from the SAME CURRENT measured EEF, or null when honestly unlocalized. Preserve static world placement as EEF moves. Geometry-null retains a remembered box; for a new object it means image-only uncertainty, never clear space. The local persistent inventory retains omitted/occluded objects and reports age/support; omission, target changes and renaming never delete them. This is a generic list capped at12 entries, not an expected scene count. Never invent object identities, geometry, calibrated depth or precision.

SINGLE GEOMETRY SOURCE. Do not output field_intent.obstacles, motion_context.nearby_obstacles/obstacle_assessment or target_geometry. They are derived from the inventory and measured state. target_contact alone states none|fingers|payload_support for the exact top-level target_id. A visible/current target requires its inventory entry and grounded approximate geometry for approach or interaction; an omitted target may retain its remembered geometry, never fabricated fallback. Keep field_intent.unknown_regions only for genuinely unseen/unlocalized spatial regions with distinct IDs, not duplicates of listed physical objects. Known visible object uncertainty does not make it a solid unknown wall. Locally relevant nominal tool-inflated object cores stay hard, with nonnegative uncertainty traversal cost under adequate current image support. Genuine unknown regions retain hard inflation. No box/ROI deletion, shrinkage or uncertainty reduction merely to bypass rejection.

MOTION CONTEXT. Supply actual purpose and whole-tool external ROI including open fingers/payload; this is different from a texture patch. transit/separate require intended_contact_id=null; interaction requires exact target_id. Physical IDs remain stable across neighbor/target/payload roles. Current/remembered inventory geometry and projected support constrain every controller; image-only or stale coverage is explicit uncertainty, not proof of collision or clearance. Read scene_inventory feedback and choose meaningful bounded Field progress or informative task correction. Unobservable scene handling retains the existing15mm travel/5mm command caps; it does not fabricate occupancy everywhere. Read current translation_guard, reference_support and recovery_options; earlier snapshots are not new evidence.""" + text[end:]
    begin, end = text.index("TARGET/PART CONTACT."), text.index("\n\nCONTACT ALIGNMENT.")
    text = text[:begin] + """TARGET/PART CONTACT. target_contact=none permits no intended target contact. fingers exempts only known fingers/pads against the exact interaction target, NEVER hand/palm. payload_support permits only independently observed carried-object support on a DIFFERENT destination during release/interaction and exempts no robot part. Whole target extent comes only from its current/remembered inventory record, independent of distractor lists. Unavailable target geometry permits finite observation/hold/recovery or held opening, never approach/closure. Preserve carried-object constraints through uncertain attachment and destination changes; opening alone does not prove separation. All RGB geometry remains hypothetical, not a collision certificate.""" + text[end:]
    text = text.replace("CONTACT ALIGNMENT.", "CONTACT ALIGNMENT. When a Field approach moves closer to the target core and enters the local handoff band max(30mm, target uncertainty), supply current actual two-view target-contact and pinch descriptors. A learned image response is not required just to provide those visible descriptors. A guessed metric waypoint or wrist image center is insufficient. A justified lateral/away recovery remains available under the other scene checks; the handoff is not an instruction to freeze or descend blindly.")
    return text


def _memory_geometry(context, identity):
    entries = (((context or {}).get("visual_feedback") or {}).get("scene_inventory") or {}).get("entries", [])
    for entry in entries:
        if entry.get("id") == identity:
            geometry = copy.deepcopy(entry.get("current_geometry"))
            if geometry is not None:
                # Current model offsets are bounded by the wire schema. A
                # remembered world object may rebase beyond that wire range.
                _validate(geometry, _object({"center_offset_m": _array(3),
                    "half_extents_m": _array(3, .0001, .5),
                    "uncertainty_m": {"type": "number", "minimum": 0, "maximum": .1}}),
                    "observable_scene_inventory.current_geometry")
            return geometry
    return None


def _validation_only_proxy(box):
    """Legacy shape gate only; never used for planning, output or raw history."""
    result = copy.deepcopy(box)
    if result is not None and any(abs(x) > .5 for x in result["center_offset_m"]):
        result["center_offset_m"] = [0., 0., 0.]
    return result


def _internal_box(box):
    return {"id": box["id"],
        "lower_offset_m": [c-h for c, h in zip(box["center_offset_m"], box["half_extents_m"])],
        "upper_offset_m": [c+h for c, h in zip(box["center_offset_m"], box["half_extents_m"])],
        "uncertainty_m": box["uncertainty_m"]}


class VisualFieldSubscriptionClientV16(VisualFieldSubscriptionClientV15):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v16", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        inventory = data["scene_inventory"]
        ids = [entry["id"] for entry in inventory]
        if any(not identity.strip() for identity in ids) or len(set(ids)) != len(ids):
            raise SchemaError("Scene inventory IDs must be nonempty and unique")
        for entry in inventory:
            for view in ("external_roi_uv", "wrist_roi_uv"):
                _ordered_roi(entry[view])
            if entry["external_roi_uv"] is entry["wrist_roi_uv"] is None:
                raise SchemaError("Each current inventory object requires a visible ROI in at least one view; omit occluded entries to retain memory")
        field = data["field_intent"]
        unknown = [] if field is None else field["unknown_regions"]
        unknown_ids = [box["id"] for box in unknown]
        remembered = (((self.observable_context or {}).get("visual_feedback") or {}).get("scene_inventory") or {}).get("entries", [])
        physical_ids = set(ids) | {entry.get("id") for entry in remembered}
        if len(set(unknown_ids)) != len(unknown_ids) or physical_ids.intersection(unknown_ids):
            raise SchemaError("Unknown spatial region IDs must be unique and distinct from physical inventory objects")
        resolved = [(entry, copy.deepcopy(entry["geometry"]) or _memory_geometry(self.observable_context, entry["id"]))
                    for entry in inventory]
        target = next((geometry for entry, geometry in resolved if entry["id"] == data["target_id"]), None)
        if target is None:
            target = _memory_geometry(self.observable_context, data["target_id"])
        target_geometry = None if target is None else dict(id=data["target_id"], **target, allowed_contact=data["target_contact"])
        if target_geometry is None and data["target_contact"] != "none":
            raise SchemaError("Unavailable target geometry cannot grant target contact permission")
        boxes = [dict(id=entry["id"], **geometry) for entry, geometry in resolved
                 if entry["id"] != data["target_id"] and geometry is not None]
        neighbors = [{"id": entry["id"], "external_roi_uv": copy.deepcopy(entry["external_roi_uv"])}
                     for entry in inventory if entry["id"] != data["target_id"] and entry["external_roi_uv"] is not None]
        incomplete = any(geometry is None or entry["external_roi_uv"] is None for entry, geometry in resolved)
        assessment = "unobservable" if incomplete else "present" if neighbors else "none_relevant"
        legacy = copy.deepcopy(data)
        for key in ("scene_inventory", "target_contact"):
            legacy.pop(key)
        legacy["target_geometry"] = _validation_only_proxy(target_geometry)
        # Compatibility-only projection: all unlocalized/extra entries remain
        # in the returned prediction for persistent runtime coverage admission.
        localized = {box["id"] for box in boxes}
        shadow_neighbors = [entry for entry in neighbors if entry["id"] in localized][:6]
        needed = {entry["id"] for entry in shadow_neighbors}
        shadow_boxes = ([box for box in boxes if box["id"] in needed] +
                        [box for box in boxes if box["id"] not in needed])[:8]
        legacy["motion_context"].update(nearby_obstacles=shadow_neighbors,
            obstacle_assessment="unobservable" if incomplete else "present" if shadow_neighbors else "none_relevant")
        if legacy["field_intent"] is not None:
            legacy["field_intent"]["obstacles"] = [_validation_only_proxy(box) for box in shadow_boxes]
        result = super()._validate_v5_response(legacy, grid)
        result["target_geometry"] = copy.deepcopy(target_geometry)
        result["motion_context"].update(nearby_obstacles=neighbors, obstacle_assessment=assessment)
        if result["field_intent"] is not None:
            result["field_intent"]["obstacles"] = [_internal_box(box) for box in boxes]
        result.update(scene_inventory=copy.deepcopy(inventory), target_contact=data["target_contact"],
            raw_wire_prediction=copy.deepcopy(data))
        return result

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
