"""Model-authorized semantic stages executed against public estimated geometry.

No stage contains a Cartesian waypoint. Targets are derived from object/support
OBBs and the robot's collision geometry. Only accepted local pose observations
can revise a saved goal; all continuations retain cumulative finite budgets.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .agent_speedup_v3_fragments import AdaptiveFragmentSequencePrimitive, _IntentWatchdog, _same_intent, _slerp
from .agent_speedup_v2_user_check_fragments import PROFILES
from .agent_speedup_v3_geometry import _rotation
from .agent_speedup_v3_observation import propose_evidence_views
from .astranomy_agentic_motion import measured_pose, rotation_error_rad
from .contact_skills import Primitive
from .robot_tool_geometry import _local_parts, FINGER_JOINT_RANGES_M

PROGRAM_VERSION = "agent-speedup-v3-semantic-program-4"
GOAL_KINDS = ("approach", "grasp", "lift_clear", "transfer", "place", "hold", "retreat")
MAX_LOCAL_RECOVERIES = 2
MAX_LOCAL_GOAL_UPDATES = 4


def _object(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def stages_program_schema():
    # The client imports this module, so the shared context contract is fetched
    # only when a schema is requested after module initialization.
    from .agent_speedup_v3_client import motion_context_schema
    ident = {"type": "string", "minLength": 1, "maxLength": 80}
    goal = _object({"kind": {"type": "string", "enum": list(GOAL_KINDS)},
        "reference_id": {"anyOf": [ident, {"type": "null"}]},
        "quaternion_wxyz": {"anyOf": [{"type": "array", "items": {"type": "number"},
                                        "minItems": 4, "maxItems": 4}, {"type": "null"}]}})
    stage = _object({"motion_context": motion_context_schema(), "goal": goal,
        "gripper": {"type": "string", "enum": ["keep", "open", "close"]},
        "profile": {"type": "string", "enum": ["transit", "contact"]},
        "verification": {"type": "string", "enum": ["arrival", "visual_hold", "visual_release"]},
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 120}})
    return _object({"stages": {"type": "array", "items": stage, "minItems": 1, "maxItems": 24}})


def compact_program_schema():
    """A bounded composition of generic operations, without Cartesian moves."""
    from .agent_speedup_v3_client import motion_context_schema
    ident = {"type": "string", "minLength": 1, "maxLength": 80}
    ids = {"type": "array", "items": ident, "maxItems": 12}
    inventory = motion_context_schema()["properties"]["scene_inventory"]
    operation = _object({"id": ident, "kind": {"type": "string", "enum": ["pick_place"]},
        "object_id": ident, "destination_id": ident, "after": ids,
        "quaternion_wxyz": {"anyOf": [{"type": "array", "items": {"type": "number"},
            "minItems": 4, "maxItems": 4}, {"type": "null"}]},
        "source_inventory": inventory, "destination_inventory": inventory,
        "route_obstacle_ids": ids,
        "contact_ids": _object({key: ids for key in ("grasp", "carry", "place", "retreat")}),
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 120}})
    return _object({"operations": {"type": "array", "items": operation, "minItems": 1, "maxItems": 3}})


def program_schema():
    return {"anyOf": [stages_program_schema(), compact_program_schema()]}


def _operation_order(operations):
    """Stable topological order; dependencies are semantic operation IDs."""
    ids = [item["id"] for item in operations]
    if len(set(ids)) != len(ids):
        raise ValueError("operation IDs must be unique")
    for item in operations:
        if item["id"] in item["after"] or not set(item["after"]) <= set(ids):
            raise ValueError("operation dependencies must reference other operations in this contract")
    ordered, done = [], set()
    while len(ordered) < len(operations):
        ready = next((item for item in operations if item["id"] not in done and set(item["after"]) <= done), None)
        if ready is None:
            raise ValueError("operation dependencies contain a cycle")
        ordered.append(ready)
        done.add(ready["id"])
    return ordered


def validate_compact_program(value):
    from .agent_speedup_v3_client import validate_motion_context
    if not isinstance(value, dict) or set(value) != {"operations"}:
        raise ValueError("compact program requires exactly operations")
    operations = value["operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= 3:
        raise ValueError("compact program requires 1..3 operations")
    clean = deepcopy(value)
    fields = set(compact_program_schema()["properties"]["operations"]["items"]["required"])
    def identifier(item):
        return isinstance(item, str) and bool(item.strip()) and len(item) <= 80
    def identifiers(items, name):
        if not isinstance(items, list) or len(items) > 12 or any(not identifier(item) for item in items):
            raise ValueError(name+" must be a bounded list of IDs")
        if len(set(items)) != len(items):
            raise ValueError(name+" must not contain duplicates")
    for item in clean["operations"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("operation fields differ from the strict semantic contract")
        if item["kind"] != "pick_place":
            raise ValueError("unsupported operation kind; use explicit stages or a native route")
        if any(not identifier(item[name]) for name in ("id", "object_id", "destination_id")):
            raise ValueError("operation IDs must be nonempty bounded strings")
        if item["object_id"] == item["destination_id"]:
            raise ValueError("operation object and destination must differ")
        identifiers(item["after"], "operation.after")
        identifiers(item["route_obstacle_ids"], "operation.route_obstacle_ids")
        contacts = item["contact_ids"]
        if not isinstance(contacts, dict) or set(contacts) != {"grasp", "carry", "place", "retreat"}:
            raise ValueError("contact_ids requires exactly grasp, carry, place, retreat")
        for name, entries in contacts.items():
            identifiers(entries, "contact_ids."+name)
        for name in ("source_inventory", "destination_inventory"):
            context = {"intent": "transit", "target_id": item["object_id"], "held_object_id": None,
                "allowed_contact_ids": [], "scene_inventory": item[name]}
            item[name] = validate_motion_context(context)["scene_inventory"]
        if len(set(item["source_inventory"]["support_ids"]+item["destination_inventory"]["support_ids"])) > 12:
            raise ValueError("combined route inventory exceeds the bounded support list")
        q = item["quaternion_wxyz"]
        if q is not None:
            if (not isinstance(q, list) or len(q) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in q)
                    or np.linalg.norm(q) < 1e-12):
                raise ValueError("operation quaternion must contain four finite nonzero numbers")
            item["quaternion_wxyz"] = (np.asarray(q)/np.linalg.norm(q)).tolist()
        if type(item["max_steps"]) is not int or not 1 <= item["max_steps"] <= 120:
            raise ValueError("operation max_steps must be a 1..120 local chunk")
    _operation_order(clean["operations"])
    return clean


def compile_operations(value):
    """Compile authorized contacts and predicates; geometry still derives goals."""
    contract = validate_compact_program(value)
    stages, stage_operation_ids, ranges = [], [], []
    for item in _operation_order(contract["operations"]):
        obj, destination = item["object_id"], item["destination_id"]
        source, dest = item["source_inventory"], item["destination_inventory"]
        route = {"reviewed": source["reviewed"] and dest["reviewed"],
            "support_ids": list(dict.fromkeys(source["support_ids"]+dest["support_ids"])),
            "nearby_obstacle_ids": item["route_obstacle_ids"], "unresolved": source["unresolved"] or dest["unresolved"]}
        # Possible payload extent is retained through the release probe. The
        # executor clears held_object_id only after geometry accepts release.
        specs = [
            ("approach", "approach", obj, None, source, [], "keep", "arrival", "transit"),
            ("grasp", "grasp", obj, None, source, item["contact_ids"]["grasp"], "keep", "arrival", "contact"),
            ("hold", "grasp", obj, None, source, item["contact_ids"]["grasp"], "close", "arrival", "contact"),
            ("lift_clear", "grasp", obj, obj, source, item["contact_ids"]["carry"], "keep", "visual_hold", "contact"),
            ("transfer", "transit", destination, obj, route, item["contact_ids"]["carry"], "keep", "arrival", "transit"),
            ("place", "place", destination, obj, dest, item["contact_ids"]["place"], "keep", "arrival", "contact"),
            ("hold", "place", destination, obj, dest, item["contact_ids"]["place"], "open", "arrival", "contact"),
            ("retreat", "retreat", destination, obj, dest, item["contact_ids"]["retreat"], "keep", "visual_release", "transit"),
        ]
        start = len(stages)
        for kind, intent, target, held, inventory, contacts, gripper, verification, profile in specs:
            stages.append({"motion_context": {"intent": intent, "target_id": target, "held_object_id": held,
                "allowed_contact_ids": deepcopy(contacts), "scene_inventory": deepcopy(inventory)},
                "goal": {"kind": kind, "reference_id": target,
                    "quaternion_wxyz": deepcopy(item["quaternion_wxyz"]) if kind in ("approach", "grasp") else None},
                "gripper": gripper, "profile": profile, "verification": verification, "max_steps": item["max_steps"]})
            stage_operation_ids.append(item["id"])
        ranges.append({"id": item["id"], "object_id": obj, "destination_id": destination,
            "after": deepcopy(item["after"]), "first_stage": start, "last_stage": len(stages)-1})
    return {"program": validate_program({"stages": stages}), "operation_contract": contract,
        "stage_operation_ids": stage_operation_ids, "operation_ranges": ranges}


def validate_program(value):
    from .agent_speedup_v3_client import validate_motion_context
    if isinstance(value, dict) and set(value) == {"operations"}:
        return validate_compact_program(value)
    if not isinstance(value, dict) or set(value) != {"stages"}:
        raise ValueError("program requires exactly stages")
    if not isinstance(value["stages"], list) or not 1 <= len(value["stages"]) <= 24:
        raise ValueError("program requires 1..24 stages")
    clean = deepcopy(value)
    for stage in clean["stages"]:
        if not isinstance(stage, dict) or set(stage) != {"motion_context", "goal", "gripper", "profile", "verification", "max_steps"}:
            raise ValueError("program stage fields differ from the strict contract")
        stage["motion_context"] = validate_motion_context(stage["motion_context"])
        goal = stage["goal"]
        if not isinstance(goal, dict) or set(goal) != {"kind", "reference_id", "quaternion_wxyz"} or goal["kind"] not in GOAL_KINDS:
            raise ValueError("program goal must be semantic geometry, without coordinates or offsets")
        ref = goal["reference_id"]
        if ref is not None and (not isinstance(ref, str) or not ref.strip() or len(ref) > 80):
            raise ValueError("program reference_id must be a bounded ID or null")
        if goal["kind"] in ("approach", "grasp", "transfer", "place") and ref is None:
            raise ValueError("this semantic goal needs a geometry reference")
        q = goal["quaternion_wxyz"]
        if q is not None:
            if (not isinstance(q, list) or len(q) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in q)
                    or np.linalg.norm(q) < 1e-12):
                raise ValueError("program quaternion must contain four finite nonzero numbers")
            goal["quaternion_wxyz"] = (np.asarray(q)/np.linalg.norm(q)).tolist()
        if stage["gripper"] not in ("keep", "open", "close") or stage["profile"] not in PROFILES:
            raise ValueError("unsupported program gripper/profile")
        if stage["verification"] not in ("arrival", "visual_hold", "visual_release"):
            raise ValueError("unsupported program verification")
        if type(stage["max_steps"]) is not int or not 1 <= stage["max_steps"] <= 120:
            raise ValueError("program max_steps must be a 1..120 local chunk")
        if goal["kind"] in ("transfer", "place") and not stage["motion_context"]["held_object_id"]:
            raise ValueError("transfer/place require an explicit held_object_id")
    return clean


class MissingGeometry(ValueError):
    pass


def _goal_local(geometry):
    return getattr(geometry, "execution_policy", None) == "goal_local"


def _conditional_shape_usable(entity):
    """Expected contact/occlusion can retain a hypothesis, never refresh it."""
    monitor = entity.get("visual_monitoring") or {}
    if (entity.get("nonsettling_invalidation") or monitor.get("status") == "changed"
            or entity.get("invalidation_kind") in ("visual_appearance_changed", "local_observed_displacement")
            or entity.get("stale_reason") in ("visual_appearance_changed", "local_observed_displacement")):
        return False
    contact = (entity.get("invalidation_kind") == "possible_contact"
               and entity.get("invalidation_local_contact") is True)
    occluded = monitor.get("self_occluded") is True or monitor.get("reason") == "expected_self_occlusion"
    occlusion_stale = (occluded and entity.get("stale_reason") in (None, "expected_self_occlusion")
                       and entity.get("invalidation_kind") in (None, "expected_self_occlusion"))
    return ((not entity.get("stale") or contact or occlusion_stale)
            and (not entity.get("monitoring_unknown") or contact or occluded))


def _entity(geometry, ident, *, roles=None, allow_probe=False):
    entity = geometry.entities.get(ident)
    if entity is None or not entity.get("boxes"):
        raise MissingGeometry("geometry_entity:"+str(ident))
    if (_goal_local(geometry) and (entity.get("stale") or entity.get("monitoring_unknown"))
            and (entity.get("nonsettling_invalidation")
                 or (entity.get("visual_monitoring") or {}).get("status") == "changed"
                 or entity.get("invalidation_kind") in ("visual_appearance_changed", "local_observed_displacement")
                 or entity.get("stale_reason") in ("visual_appearance_changed", "local_observed_displacement"))):
        raise MissingGeometry("unresolved_observed_geometry_change:"+str(ident))
    if ((entity.get("stale") or entity.get("monitoring_unknown"))
            and not (allow_probe and entity.get("attachment_probe_shape_usable"))
            and not (_goal_local(geometry) and _conditional_shape_usable(entity))):
        raise MissingGeometry("fresh_geometry:"+str(ident))
    if roles and entity["role"] not in roles:
        raise MissingGeometry("geometry_role:"+str(ident))
    return entity


def _bounds(box):
    extent = np.abs(box["rotation"]) @ box["half"] + box.get("uncertainty", 0.)
    return box["center"]-extent, box["center"]+extent


def _preview_goal(geometry, view, xyz, q, stage, command):
    preview = getattr(geometry, "preview_motion", None)
    if preview is None:
        return None
    profile = PROFILES[stage["profile"]]
    return preview(view, xyz, q, {**stage["motion_context"], "commanded_gripper": command,
        "gripper_action": False, "original_goal_xyz_m": list(xyz),
        "controller_translation_limit_m": profile["action_translation_m"], "controller_rotation_limit_rad": .1,
        "position_tolerance_m": profile["position_tolerance_m"], "orientation_tolerance_rad": profile["rotation_tolerance_rad"]})


def _route_allowed(preview):
    """Execution permission is separate from completeness of scene evidence."""
    if not isinstance(preview, dict):
        return False
    if "execution_allowed" in preview:
        return preview["execution_allowed"] is True
    return preview.get("status") == "supported" and preview.get("allowed_fraction", 0.) >= 1.-1e-9


def _nominal_bounds(box):
    extent = np.abs(box["rotation"]) @ box["half"]
    return box["center"]-extent, box["center"]+extent


def _grasp_support_constraints(chosen, supports, rotation, parts, tolerance):
    """Build relevant nominal support inequalities from public hand vertices.

    Model uncertainty is recorded separately. Expanding a support by its
    uncalibrated uncertainty must not lift a grasp above the target itself.
    """
    vertices = np.concatenate([vertices for _, vertices, _ in parts]) @ rotation.T
    constraints = []
    for support in supports:
        axis = int(np.argmax(np.abs(support["rotation"][2])))
        normal = support["rotation"][:, axis].copy()
        if normal[2] < 0:
            normal *= -1
        tangent = [index for index in range(3) if index != axis]
        center = support["rotation"].T @ (chosen["center"]-support["center"])
        extent = np.abs(support["rotation"].T @ chosen["rotation"]) @ chosen["half"]
        if any(abs(center[index]) > support["half"][index]+extent[index] for index in tangent):
            continue
        plane = float(np.dot(normal, support["center"])+support["half"][axis])
        target_top = float(np.dot(normal, chosen["center"])+np.dot(np.abs(normal @ chosen["rotation"]), chosen["half"]))
        if plane > target_top+tolerance:
            continue  # An overhead shelf is an obstacle, not this object's support.
        constraints.append({"entity_id": support["owner"], "normal": normal,
            "plane_offset_m": plane, "tool_min_offset_m": float(np.min(vertices @ normal)),
            "support_uncertainty_m": float(support["uncertainty"])})
    return constraints


def _support_consistent_grasp(point, chosen, rotation, open_parts, constraints, tolerance):
    """Intersect support clearance with the actual pad-closing contact region."""
    point = np.asarray(point).copy()
    correction = max([0., *[(item["plane_offset_m"]+tolerance-np.dot(item["normal"], point)
        -item["tool_min_offset_m"])/item["normal"][2] for item in constraints]])
    point[2] += correction
    pads = [vertices for name, vertices, _ in open_parts if "pad" in name]
    center = rotation.T @ (chosen["center"]-point)
    half = np.abs(rotation.T @ chosen["rotation"]) @ chosen["half"]
    lower, upper = center-half, center+half
    inner_left, inner_right = float(pads[0][:, 1].min()), float(pads[1][:, 1].max())
    if lower[1] < inner_right-1e-9 or upper[1] > inner_left+1e-9:
        return None, {"reason": "object_outside_open_jaws"}
    # Each pad must reach the object by closing along local Y. Its X/Z contact
    # area must still intersect after the support correction, on both sides.
    overlaps = [[float(min(vertices[:, axis].max(), upper[axis])-max(vertices[:, axis].min(), lower[axis]))
                 for axis in (0, 2)] for vertices in pads]
    if any(min(overlap) <= 1e-9 for overlap in overlaps):
        return None, {"reason": "support_clearance_and_pad_contact_do_not_intersect"}
    # Closing to q=0 shifts the left/right pads by their public slider ranges.
    closed_left = inner_left-FINGER_JOINT_RANGES_M[0][1]
    closed_right = inner_right-FINGER_JOINT_RANGES_M[1][0]
    if upper[1] < closed_left or lower[1] > closed_right:
        return None, {"reason": "object_outside_bilateral_closing_range"}
    checks = [{"entity_id": item["entity_id"], "normal_world": item["normal"].tolist(),
        "nominal_tool_clearance_m": float(np.dot(item["normal"], point)+item["tool_min_offset_m"]-item["plane_offset_m"]),
        "support_uncertainty_m": item["support_uncertainty_m"]} for item in constraints]
    return point, {"source": "nominal_model_support_and_public_whole_hand",
        "support_correction_world_m": [0., 0., float(correction)], "support_constraints": checks,
        "pad_object_overlap_xz_m": overlaps, "object_uncertainty_m": float(chosen["uncertainty"]),
        "uncertainty_is_advisory": True, "physical_contact_verified": False}


def _credible_pose_revision(entity):
    evidence = entity.get("local_pose_evidence") or {}
    revision = entity.get("local_pose_revision", 0)
    if (type(revision) is int and revision > 0 and not entity.get("stale")
            and not entity.get("monitoring_unknown") and evidence.get("accepted") is True
            and evidence.get("status") == "supported" and evidence.get("pose_updated") is True
            and evidence.get("source") == "public_two_view_feature_displacement"
            and evidence.get("geometry_revision") == entity.get("geometry_revision")):
        return revision
    return 0


def derive_program_goal(view, stage, geometry, *, excluded_targets=()):
    """Return a deterministic estimated goal plus its geometric derivation."""
    goal, context = stage["goal"], stage["motion_context"]
    kind, ref = goal["kind"], goal["reference_id"]
    q = goal["quaternion_wxyz"] or measured_pose(view.state)[3:7]
    rotation = _rotation(q)
    if kind == "hold":
        return {"xyz_m": view.position.tolist(), "quaternion_wxyz": list(q), "evidence_ids": [], "derivation": "measured_current_pose"}
    inventory, local = context["scene_inventory"], _goal_local(geometry)
    if not inventory["support_ids"] or (not local and (not inventory["reviewed"] or inventory["unresolved"])):
        raise MissingGeometry("reviewed_support_and_obstacle_inventory")
    probe = kind == "lift_clear" and stage["verification"] == "visual_hold"
    release_probe = kind == "retreat" and stage["verification"] == "visual_release"
    reference = _entity(geometry, ref, allow_probe=(probe or release_probe) and ref == context["held_object_id"]) if ref else None
    ids = set(inventory["support_ids"] + inventory["nearby_obstacle_ids"])
    if reference is not None:
        ids.add(ref)
    boxes, supports, advisories = [], [], []
    if local and (not inventory["reviewed"] or inventory["unresolved"]):
        advisories.append({"reason": "scene_inventory_incomplete", "target_geometry_still_required": True})
    for ident in sorted(ids):
        entity = _entity(geometry, ident, allow_probe=(probe or release_probe) and ident == context["held_object_id"])
        if local and entity["uncertainty_m"] > 0:
            advisories.append({"entity_id": ident, "reason": "nominal_goal_uses_uncertain_model_shape",
                "uncertainty_m": entity["uncertainty_m"], "uncertainty_reduced": False})
        if local and (entity.get("stale") or entity.get("monitoring_unknown")):
            advisories.append({"entity_id": ident, "reason": "conditional_shape_after_contact_or_occlusion",
                "stale_reason": entity.get("stale_reason"), "monitoring_unknown": bool(entity.get("monitoring_unknown")),
                "uncertainty_m": entity["uncertainty_m"], "observation_refreshed": False})
        if entity["attachment"] != "held" and entity["role"] != "destination_region":
            entity_boxes = geometry._entity_boxes(entity, view)
            boxes.extend(entity_boxes)
            if entity["role"] == "support_surface":
                supports.extend(entity_boxes)
    if not boxes:
        raise MissingGeometry("support_and_obstacle_bounds")
    bodies, payload_unknown = geometry._local_bodies(view, context)
    if payload_unknown and not ((release_probe or (local and probe)) and context["held_object_id"] and
            geometry.entities.get(context["held_object_id"], {}).get("attachment_probe_shape_usable")):
        raise MissingGeometry("conservative_payload_extent:"+str(context["held_object_id"]))
    local_bottom = min(float((rotation @ body["center"])[2] -
        np.dot(np.abs((rotation @ body["rotation"])[2]), body["half"]) - body.get("uncertainty", 0.)) for body in bodies)
    if local:
        local_bottom = min(local_bottom, min(float(np.min((vertices @ rotation.T)[:, 2]))
            for _, vertices, _ in _local_parts(np.asarray(view.state), True)))
    obstacle_top = max(float((_nominal_bounds(box) if local else _bounds(box))[1][2]) for box in boxes)
    tolerance = PROFILES[stage["profile"]]["position_tolerance_m"]
    clearance_z = obstacle_top-local_bottom+tolerance
    xyz = view.position.copy()
    goal_candidates = []
    goal_consistency = None
    if kind in ("approach", "grasp"):
        _entity(geometry, ref, roles=("object", "graspable_part"))
        open_state = np.asarray(view.state).copy()
        open_state[6:8] = [FINGER_JOINT_RANGES_M[0][1], FINGER_JOINT_RANGES_M[1][0]]
        open_parts = _local_parts(open_state, False)
        pads = [vertices for name, vertices, _ in open_parts if "pad" in name]
        pad_center = np.mean([vertices.mean(axis=0) for vertices in pads], axis=0)
        free_jaw_span = float(pads[0][:, 1].min()-pads[1][:, 1].max())
        candidates = []
        for box in geometry._entity_boxes(reference, view):
            width = 2*(np.dot(np.abs(box["rotation"].T @ rotation[:, 1]), box["half"])
                       +(0. if local else box["uncertainty"]))
            if width <= free_jaw_span:
                candidates.append((width, box))
        if not candidates:
            raise MissingGeometry("graspable_part_fitting_public_jaw_span:"+str(ref))
        ranked = []
        rejected_contact = False
        for width, chosen in candidates:
            axis = int(np.argmax(chosen["half"]))
            constraints = (_grasp_support_constraints(chosen, supports, rotation,
                _local_parts(open_state, True), tolerance) if local else [])
            if local and not constraints:
                raise MissingGeometry("support_beneath_grasp_target:"+str(ref))
            for fraction in (0., -.5, .5):
                point = chosen["center"]+fraction*chosen["half"][axis]*chosen["rotation"][:, axis]-rotation @ pad_center
                consistency = None
                if local:
                    point, consistency = _support_consistent_grasp(point, chosen, rotation, open_parts, constraints, tolerance)
                    if point is None:
                        rejected_contact = True
                        continue
                if kind == "approach":
                    point[2] = max(clearance_z, point[2])
                if any(np.linalg.norm(point-np.asarray(old)) <= tolerance for old in excluded_targets):
                    continue
                preview = _preview_goal(geometry, view, point, q, stage, "open")
                if (preview is not None and (preview.get("execution_allowed") is False or
                        ("execution_allowed" not in preview and preview.get("status") == "blocked" and preview["allowed_fraction"] <= 1e-9))):
                    continue
                permitted = _route_allowed(preview)
                clearance = (preview or {}).get("clearance_estimate_m")
                ranked.append(((permitted, (preview or {}).get("allowed_fraction", 1.),
                                clearance if clearance is not None else 0., -np.linalg.norm(point-view.position)), point, consistency))
        if not ranked:
            if local and rejected_contact:
                raise MissingGeometry("support_clearance_and_grasp_contact_inconsistent:"+str(ref))
            raise MissingGeometry("supported_grasp_candidate:"+str(ref))
        ranked.sort(key=lambda row: row[0], reverse=True)
        xyz = ranked[0][1]
        goal_candidates = [point.tolist() for _, point, _ in ranked]
        goal_consistency = ranked[0][2]
    elif kind in ("lift_clear", "retreat"):
        xyz[2] = max(view.position[2], clearance_z)
        if kind == "lift_clear" and stage["verification"] == "visual_hold":
            payload = _entity(geometry, context["held_object_id"], allow_probe=True)
            # A visible motion baseline is derived from the estimated body's
            # size and uncertainty, never an object/task-specific lift distance.
            baseline = max(2*payload["uncertainty_m"], min(min(box["half_extents_m"]) for box in payload["boxes"]))
            xyz[2] = max(xyz[2], view.position[2]+baseline)
    elif kind in ("transfer", "place"):
        _entity(geometry, ref, roles=("destination_region",))
        payload = _entity(geometry, context["held_object_id"])
        payload_boxes = [body for body in bodies if body["owner"] == payload["id"]]
        if not payload_boxes:
            raise MissingGeometry("attached_payload_bounds:"+payload["id"])
        offset = np.asarray(payload.get("attachment_offset_hand_m", view.R.T @ (np.asarray(payload["position_m"])-view.position)))
        regions = geometry._entity_boxes(reference, view)
        region = max(regions, key=lambda box: float(np.prod(box["half"])))
        xyz = region["center"]-rotation @ offset
        if kind == "transfer":
            xyz[2] = max(clearance_z, view.position[2])
        else:
            supports = [box for ident in inventory["support_ids"] for box in geometry._entity_boxes(
                _entity(geometry, ident, roles=("support_surface",)), view)]
            bounds = _nominal_bounds if local else _bounds
            lo, hi = bounds(region)
            supports = [box for box in supports if np.all(bounds(box)[1][:2] >= lo[:2]) and np.all(bounds(box)[0][:2] <= hi[:2])]
            if not supports:
                raise MissingGeometry("destination_support_overlap:"+str(ref))
            payload_bottom = min(float((rotation @ body["center"])[2]-np.dot(
                np.abs((rotation @ body["rotation"])[2]), body["half"])-(0. if local else body["uncertainty"])) for body in payload_boxes)
            minimum_z = max(lo[2], max(bounds(box)[1][2] for box in supports))-payload_bottom
            payload_top = max(float((rotation @ body["center"])[2]+np.dot(
                np.abs((rotation @ body["rotation"])[2]), body["half"])+(0. if local else body["uncertainty"])) for body in payload_boxes)
            maximum_z = min(clearance_z, hi[2]-payload_top)
            accepted = None
            for height in np.linspace(minimum_z, max(minimum_z, maximum_z), 7):
                candidate = xyz.copy()
                candidate[2] = height
                contained = True
                for body in payload_boxes:
                    center = region["rotation"].T @ (candidate+rotation @ body["center"]-region["center"])
                    half = (np.abs(region["rotation"].T @ rotation @ body["rotation"]) @ body["half"]
                            +(0. if local else body["uncertainty"] + region["uncertainty"]))
                    contained &= bool(np.all(np.abs(center)+half <= region["half"]+1e-9))
                if not contained:
                    continue
                preview = _preview_goal(geometry, view, candidate, q, stage, "open")
                if preview is None or _route_allowed(preview):
                    accepted = candidate
                    break
            if accepted is None:
                raise MissingGeometry("contained_supported_release_pose:"+str(ref))
            xyz = accepted
    return {"xyz_m": xyz.tolist(), "quaternion_wxyz": list(q), "evidence_ids": sorted(ids),
            "derivation": kind+"_from_estimated_OBBs_and_public_robot_parts", "clearance_height_m": clearance_z,
            "candidate_targets": goal_candidates,
            "goal_consistency": goal_consistency, "geometry_advisories": advisories,
            "pose_evidence_revisions": {ident: _credible_pose_revision(geometry.entities[ident]) for ident in sorted(ids)},
            "physical_clearance_verified": False}


class AdaptiveProgramPrimitive(Primitive):
    """Advance only model-authorized stages after their local predicates hold."""
    def __init__(self, view, program, geometry, *, step=0, keep_gripper="open", interaction_monitor=None, operation_runtime=None, _saved=None):
        super().__init__("semantic_program")
        self.program = validate_program(program)
        if "operations" in self.program:
            compiled = compile_operations(self.program)
            self.program = compiled["program"]
        else:
            compiled = _saved or {}
        self.operation_contract = deepcopy(compiled.get("operation_contract"))
        self.stage_operation_ids = deepcopy(compiled.get("stage_operation_ids", []))
        self.operation_ranges = deepcopy(compiled.get("operation_ranges", []))
        self.completed_operations = deepcopy(_saved.get("completed_operations", [])) if _saved else []
        self.payload_release_events = deepcopy(_saved.get("payload_release_events", [])) if _saved else []
        self.geometry, self.start_step = geometry, int(step)
        self._prior_actions = int(_saved.get("consumed_steps", 0)) if _saved else 0
        self.operation_runtime = operation_runtime
        if operation_runtime is not None:
            interaction_monitor = operation_runtime.monitor
        elif interaction_monitor is None:
            from .agent_speedup_v3_interaction import InteractionEvidence
            interaction_monitor = InteractionEvidence(geometry)
        self.interaction_monitor = interaction_monitor
        self.gripper_command = keep_gripper
        self.stage_index = int(_saved["stage_index"]) if _saved else 0
        self.completed_stages = deepcopy(_saved["completed_stages"]) if _saved else []
        self.interaction_starts = deepcopy(_saved["interaction_starts"]) if _saved else {}
        self.confirmed_interactions = set(_saved.get("confirmed_interactions", [])) if _saved else set()
        self._confirmation_signatures = deepcopy(_saved.get("confirmation_signatures", {})) if _saved else {}
        self.interaction_object_ids = deepcopy(_saved.get("interaction_object_ids", {})) if _saved else {}
        self.recovery_events = deepcopy(_saved.get("recovery_events", [])) if _saved else []
        self.recovery_keys = set(_saved.get("recovery_keys", [])) if _saved else set()
        self.recovery_attempts = deepcopy(_saved.get("recovery_attempts", {})) if _saved else {}
        self.goal_updates = deepcopy(_saved.get("goal_updates", [])) if _saved else []
        self._breadcrumbs = deepcopy(_saved.get("breadcrumbs", [])) if _saved else []
        self._recovery = deepcopy(_saved.get("recovery")) if _saved else None
        self._excluded_targets = deepcopy(_saved.get("excluded_targets", [])) if _saved else []
        self.child = None
        self.stage_goal = None
        self.stage_actions = 0
        self.verification_evidence = None
        self.missing_evidence_requests = []
        self._last_state = np.asarray(view.state).copy()
        self._phase = "move"
        self._pending_visual_revalidation = False
        self._enter_stage(view, _saved)

    @property
    def motion_context(self):
        return self.program["stages"][self.stage_index]["motion_context"]

    @property
    def current_context(self):
        return self.motion_context

    @property
    def _tracking(self):
        return getattr(self.child, "_tracking", None)

    def _current_operation(self):
        if not self.stage_operation_ids:
            return None
        ident = self.stage_operation_ids[self.stage_index]
        return next(item for item in self.operation_ranges if item["id"] == ident)

    def _operation_entry_allowed(self):
        current = self._current_operation()
        if current is None or self.stage_index != current["first_stage"]:
            return True
        completed = {item["id"] for item in self.completed_operations}
        prior = {item["id"] for item in self.operation_ranges if item["last_stage"] < self.stage_index}
        if not (set(current["after"]) | prior) <= completed:
            self._pause("program_previous_operation_unverified", ["accepted_release:"+ident for ident in sorted(
                (set(current["after"]) | prior)-completed)])
            return False
        return True

    def _interaction_id(self, kind=None):
        stage = self.program["stages"][self.stage_index]
        if kind == "release":
            if self.operation_runtime is not None:
                return self.operation_runtime.bound_object_id
            return self.interaction_object_ids.get("release")
        if kind == "hold":
            explicit = self.motion_context["held_object_id"]
            if explicit is not None:
                return explicit
            if self.operation_runtime is not None:
                return self.operation_runtime.bound_object_id
            return self.interaction_object_ids.get("hold")
        return self.motion_context["held_object_id"] or self.motion_context["target_id"] or stage["goal"]["reference_id"]

    def _interaction(self, kind, view):
        ident = self._interaction_id(kind)
        if self.operation_runtime is not None:
            return self.operation_runtime.evidence(kind, view, self.start_step+self.actions_executed, target_id=ident)
        if ident is None:
            return {"status": "unknown", "kind": kind, "reason": "explicit_interaction_object_required"}
        since = self.interaction_starts.get(kind+":"+str(ident), self.start_step)
        if kind == "hold" and self.interaction_starts.get("release:"+str(ident), -1) >= since:
            return {"status": "unknown", "reason": "release_command_revoked_prior_hold_evidence", "target_id": ident}
        return self.interaction_monitor.evidence(kind, ident, view, self.start_step+self.actions_executed, since)

    def _accept_visual(self, kind, view, evidence):
        if evidence.get("status") != "supported":
            return False
        ident = evidence.get("target_id")
        operation = self._current_operation()
        if operation is not None and ident != operation["object_id"]:
            evidence.update(status="unknown", reason="operation_object_evidence_required")
            return False
        key = kind+":"+str(ident)
        def signature():
            entity = self.geometry.entities.get(ident, {})
            return {"reference_step": evidence.get("reference_step"), **{name: deepcopy(entity.get(name)) for name in
                ("geometry_revision", "stale", "stale_reason", "monitoring_unknown", "attachment",
                 "attachment_source", "release_pending", "attachment_probe_unconfirmed")}}
        if self.operation_runtime is not None:
            accepted = (evidence.get("geometry_confirmation") or {}).get("accepted") is True
        elif key in self.confirmed_interactions and self._confirmation_signatures.get(key) == signature():
            accepted = True
        elif hasattr(self.geometry, "confirm_attachment_evidence"):
            confirmation = self.geometry.confirm_attachment_evidence(evidence, view, self.start_step+self.actions_executed)
            evidence["geometry_confirmation"] = deepcopy(confirmation)
            accepted = isinstance(confirmation, dict) and confirmation.get("accepted") is True
        else:
            accepted = False
        if accepted:
            self.confirmed_interactions.add(key)
            self._confirmation_signatures[key] = signature()
        else:
            self.confirmed_interactions.discard(key)
            self._confirmation_signatures.pop(key, None)
            evidence.update(status="unknown", reason="geometry_confirmation_rejected")
        return accepted

    def _release_payload_binding(self, evidence):
        """A release intent alone never removes the possible payload envelope."""
        stage = self.program["stages"][self.stage_index]
        held = stage["motion_context"]["held_object_id"]
        if (stage["verification"] != "visual_release" or not held or evidence.get("target_id") != held
                or evidence.get("status") != "supported"
                or (evidence.get("geometry_confirmation") or {}).get("accepted") is not True):
            return False
        stage["motion_context"]["held_object_id"] = None
        self.payload_release_events.append({"stage_index": self.stage_index, "object_id": held,
            "step": self.start_step+self.actions_executed,
            "reference_step": evidence.get("reference_step"),
            "geometry_confirmation": deepcopy(evidence["geometry_confirmation"]),
            "remaining_watchdog_steps": max(0, self._watchdog.limit-self.stage_actions)})
        self._pending_visual_revalidation = False
        return True

    def _pause(self, reason, missing=()):
        self.missing_evidence_requests = list(missing)
        self._finish("succeeded", reason, {})
        self._evidence = self.progress()

    def _enter_stage(self, view, saved=None):
        stage = self.program["stages"][self.stage_index]
        self.stage_actions = 0
        self._prior_stage_actions = int(saved.get("stage_consumed_steps", 0)) if saved else 0
        self._pending_visual_revalidation = False
        self._phase = saved["phase"] if saved else "move"
        if (not saved and stage["goal"]["kind"] == "hold" and stage["gripper"] != "keep"
                and (stage["goal"]["quaternion_wxyz"] is None or np.linalg.norm(rotation_error_rad(
                    stage["goal"]["quaternion_wxyz"], view.state[3:6])) <= PROFILES[stage["profile"]]["rotation_tolerance_rad"])):
            self._phase = "gripper"
        self._watchdog = _IntentWatchdog(saved["remaining_watchdog_steps"] if saved else stage["max_steps"],
                                        learned=saved["watchdog_learned"] if saved else False)
        if not self._operation_entry_allowed():
            return
        if stage["goal"]["kind"] in ("transfer", "place"):
            self.verification_evidence = self._interaction("hold", view)
            if not self._accept_visual("hold", view, self.verification_evidence):
                self._pause("program_visual_hold_required", ["visual_common_motion:"+str(self._interaction_id("hold"))])
                return
        try:
            self.stage_goal = deepcopy(saved["stage_goal"]) if saved and saved["stage_goal"] else derive_program_goal(view, stage, self.geometry)
        except MissingGeometry as error:
            self._pause("program_geometry_required", [str(error)])
            return
        distance = float(np.linalg.norm(np.asarray(self.stage_goal["xyz_m"])-view.position))
        angle = float(np.linalg.norm(rotation_error_rad(self.stage_goal["quaternion_wxyz"], view.state[3:6])))
        limits = PROFILES[stage["profile"]]
        pieces = max(1, math.ceil(distance/limits["target_translation_m"]), math.ceil(angle/limits["target_rotation_rad"]))
        self._watchdog = _IntentWatchdog(saved["remaining_watchdog_steps"] if saved else stage["max_steps"]*(pieces+(stage["gripper"] != "keep")),
                                        learned=saved["watchdog_learned"] if saved else False)
        self._remember_pose(view)
        refreshed = self._refresh_goal(view, rebuild=False) if saved and self._phase == "move" and self._recovery is None else False
        if self.done:
            return
        if saved and saved.get("child_continuation") and not refreshed:
            self.child = AdaptiveFragmentSequencePrimitive.from_continuation(view, saved["child_continuation"], self.geometry,
                self._child_context(), step=self.start_step+self.actions_executed, keep_gripper=self.gripper_command,
                operation_runtime=self.operation_runtime)
        elif self._phase == "verify":
            self._verify_stage(view)
            return
        else:
            self._new_child(view)
        self._check_child_entry(view)

    def _child_context(self):
        context = deepcopy(self.motion_context)
        if self._recovery is not None:
            context["intent"] = "observe" if self._recovery["kind"] == "view" else "retreat"
            if self._recovery["kind"] == "view":
                context["allowed_contact_ids"] = []
        context["adaptive_contact_speed"] = (self._phase == "move" and self._recovery is None
            and self.program["stages"][self.stage_index]["goal"]["kind"] in ("approach", "grasp"))
        stage = self.program["stages"][self.stage_index]
        if (self.operation_runtime is not None and hasattr(self.operation_runtime, "probe_context")
                and stage["goal"]["kind"] == "lift_clear" and stage["verification"] == "visual_hold"
                and self._recovery is None and self._phase == "move"):
            context = self.operation_runtime.probe_context(context, self.start_step+self.actions_executed,
                max(0, self._watchdog.limit-self.stage_actions))
        return context

    def _remember_pose(self, view):
        pose = measured_pose(view.state)
        tolerance = PROFILES[self.program["stages"][self.stage_index]["profile"]]["position_tolerance_m"]
        if not self._breadcrumbs or np.linalg.norm(np.asarray(self._breadcrumbs[-1]["pose"][:3])-view.position) >= tolerance:
            self._breadcrumbs.append({"pose": list(pose), "step": self.start_step+self.actions_executed})
            self._breadcrumbs = self._breadcrumbs[-96:]

    def _refresh_goal(self, view, *, rebuild=True):
        stage = self.program["stages"][self.stage_index]
        if (self._phase != "move" or self._recovery is not None or self.stage_goal is None
                or stage["goal"]["kind"] not in ("approach", "grasp", "transfer", "place")):
            return False
        previous = self.stage_goal.get("pose_evidence_revisions", {})
        revisions = {ident: _credible_pose_revision(self.geometry.entities.get(ident, {}))
                     for ident in self.stage_goal.get("evidence_ids", [])}
        fresh = [ident for ident, revision in revisions.items() if revision > previous.get(ident, 0)]
        if not fresh:
            return False
        try:
            updated = derive_program_goal(view, stage, self.geometry, excluded_targets=self._excluded_targets)
        except MissingGeometry:
            return False
        limits = PROFILES[stage["profile"]]
        distance = float(np.linalg.norm(np.asarray(updated["xyz_m"])-self.stage_goal["xyz_m"]))
        angle = 2*math.acos(min(1., abs(float(np.dot(updated["quaternion_wxyz"], self.stage_goal["quaternion_wxyz"])))))
        if distance <= limits["position_tolerance_m"] and angle <= limits["rotation_tolerance_rad"]:
            self.stage_goal["pose_evidence_revisions"] = revisions
            return False
        if sum(item["stage_index"] == self.stage_index for item in self.goal_updates) >= MAX_LOCAL_GOAL_UPDATES:
            self._pause("program_local_goal_update_budget_exhausted", ["stable_target_pose:"+str(stage["goal"]["reference_id"])])
            return False
        self.goal_updates.append({"stage_index": self.stage_index, "step": self.start_step+self.actions_executed,
            "previous_target": deepcopy(self.stage_goal), "updated_target": deepcopy(updated),
            "fresh_pose_evidence_ids": fresh, "remaining_watchdog_steps": self._watchdog.limit-self.stage_actions})
        self.stage_goal = updated
        if rebuild:
            preview = _preview_goal(self.geometry, view, updated["xyz_m"], updated["quaternion_wxyz"], stage, self.gripper_command)
            if not _route_allowed(preview):
                self._pause("program_geometry_after_pose_update", (preview or {}).get("missing_evidence_requests", []))
            else:
                self._new_child(view)
                self._check_child_entry(view)
        return True

    def _recovery_object_id(self):
        stage = self.program["stages"][self.stage_index]
        if stage["goal"]["kind"] in ("approach", "grasp"):
            return stage["goal"]["reference_id"]
        return self.motion_context["held_object_id"] or self._interaction_id("hold") or self.motion_context["target_id"]

    def _claim_recovery(self, reason, kind, view):
        ident = self._recovery_object_id()
        ids = self.stage_goal.get("evidence_ids", []) if self.stage_goal else []
        signature = [(key, {name: self.geometry.entities.get(key, {}).get(name) for name in
            ("position_m", "quaternion_wxyz", "boxes", "uncertainty_m", "stale", "local_pose_revision")}) for key in ids]
        key = repr((ident, self.stage_index, reason, kind, signature))
        if key in self.recovery_keys or self.recovery_attempts.get(str(ident), 0) >= MAX_LOCAL_RECOVERIES:
            return False
        if self.operation_runtime is not None:
            grant = self.operation_runtime.claim_recovery(reason, key, self.start_step+self.actions_executed,
                max_attempts=MAX_LOCAL_RECOVERIES, object_id=ident)
            if not grant.get("accepted"):
                return False
        self.recovery_keys.add(key)
        self.recovery_attempts[str(ident)] = self.recovery_attempts.get(str(ident), 0)+1
        self.recovery_events.append({"stage_index": self.stage_index, "step": self.start_step+self.actions_executed,
            "object_id": ident, "kind": kind, "trigger": reason, "status": "started",
            "remaining_watchdog_steps": self._watchdog.limit-self.stage_actions,
            "tracking_at_trigger": self.child._tracking.summary() if self.child is not None else None})
        return True

    def _start_recovery(self, view, kind, execution, reason):
        if not self._claim_recovery(reason, kind, view):
            return False
        self._recovery = {"kind": kind, "reason": reason, "original_goal": deepcopy(self.stage_goal)}
        remaining = self._watchdog.limit-self.stage_actions
        sequence = deepcopy(execution)
        for segment in sequence["segments"]:
            segment.update(max_steps=min(segment["max_steps"], remaining), observation_after=False)
        self.child = AdaptiveFragmentSequencePrimitive(view, sequence, self.geometry, self._child_context(),
            step=self.start_step+self.actions_executed, keep_gripper=self.gripper_command,
            operation_runtime=self.operation_runtime)
        if self.child.done:
            self.recovery_events[-1].update(status="rejected", reason=self.child.reason)
            self._recovery = None
            return False
        return True

    def _try_local_recovery(self, view, reason):
        if (self._recovery is not None or self.stage_goal is None or self._phase != "move"
                or self._watchdog.limit-self.stage_actions <= 0 or not hasattr(self.geometry, "preview_motion")):
            return False
        if not (str(reason).startswith("geometry_") or reason in ("segment_stall", "measured_tracking_mismatch")):
            return False
        stage = self.program["stages"][self.stage_index]
        observer = getattr(self.geometry, "observe_local_geometry", None)
        if observer is not None:
            observer(view, self.start_step+self.actions_executed, self.stage_goal.get("evidence_ids", []))
        refreshed = self._refresh_goal(view, rebuild=False)
        if self.done:
            return False
        if refreshed:
            preview = _preview_goal(self.geometry, view, self.stage_goal["xyz_m"], self.stage_goal["quaternion_wxyz"], stage, self.gripper_command)
            if (_route_allowed(preview) and self._claim_recovery(reason, "fresh_pose_retarget", view)):
                self._new_child(view)
                if not self.child.done:
                    self.recovery_events[-1]["status"] = "resumed"
                    return True
        # A stall may reflect physical contact not represented in model geometry.
        # Retreat to a previously observed pose and recheck that whole route now;
        # neither the old traversal nor the old target is a clearance certificate.
        if stage["goal"]["kind"] in ("approach", "grasp"):
            entity = self.geometry.entities.get(stage["goal"]["reference_id"], {})
            scale = max(2*PROFILES[stage["profile"]]["position_tolerance_m"], 2*float(entity.get("uncertainty_m", 0.)))
            toward = np.asarray(self.stage_goal["xyz_m"])-view.position
            recovery_stage = deepcopy(stage)
            recovery_stage["motion_context"]["intent"] = "retreat"
            for item in reversed(self._breadcrumbs):
                pose = item["pose"]
                delta = np.asarray(pose[:3])-view.position
                if np.linalg.norm(delta) < scale or np.dot(delta, toward) >= 0:
                    continue
                preview = _preview_goal(self.geometry, view, pose[:3], pose[3:7], recovery_stage, self.gripper_command)
                if _route_allowed(preview):
                    execution = {"segments": [{"skill": "move_to_pose", "xyz_m": pose[:3], "quaternion_wxyz": pose[3:7],
                        "gripper": "keep", "profile": stage["profile"], "max_steps": stage["max_steps"], "observation_after": False}]}
                    return self._start_recovery(view, "retreat", execution, reason)
        assessment = getattr(self.geometry, "last_motion", None) or {}
        assessment = assessment.get("whole_goal_assessment") or assessment
        proposals = propose_evidence_views(self.geometry, view, self._child_context(), assessment,
            self.start_step+self.actions_executed,
            attempted=[item["target_xyz_m"] for item in self.recovery_events if "target_xyz_m" in item])
        if proposals.get("candidates"):
            selected = proposals["candidates"][0]
            if self._start_recovery(view, "view", selected["execution"], reason):
                self.recovery_events[-1]["target_xyz_m"] = selected["xyz_m"]
                return True
        return False

    def _complete_recovery(self, view):
        recovery = self._recovery
        if self.child.reason != "fragment_complete":
            self.recovery_events[-1].update(status="paused", reason=self.child.reason)
            if self.child.export_continuation() is None:
                self._recovery = None
            self._pause("program_local_recovery_"+str(self.child.reason))
            return
        self._recovery = None
        stage = self.program["stages"][self.stage_index]
        observer = getattr(self.geometry, "observe_local_geometry", None)
        if observer is not None:
            observer(view, self.start_step+self.actions_executed, self.stage_goal.get("evidence_ids", []))
        if recovery["kind"] == "view":
            if not self._refresh_goal(view, rebuild=False):
                self.recovery_events[-1]["status"] = "new_view_requires_interpretation"
                self._pause("program_local_view_requires_evidence", ["fresh_local_geometry:"+str(self._recovery_object_id())])
                return
        else:
            self._excluded_targets.append(recovery["original_goal"]["xyz_m"])
            try:
                self.stage_goal = derive_program_goal(view, stage, self.geometry, excluded_targets=self._excluded_targets)
            except MissingGeometry as error:
                self.recovery_events[-1]["status"] = "no_alternative"
                self._pause("program_local_recovery_requires_geometry", [str(error)])
                return
        preview = _preview_goal(self.geometry, view, self.stage_goal["xyz_m"], self.stage_goal["quaternion_wxyz"], stage, self.gripper_command)
        if not _route_allowed(preview):
            self.recovery_events[-1]["status"] = "alternative_not_supported"
            self._pause("program_local_recovery_requires_geometry", (preview or {}).get("missing_evidence_requests", []))
            return
        self.recovery_events[-1].update(status="resumed", resumed_goal=deepcopy(self.stage_goal))
        self._new_child(view)
        self._check_child_entry(view)

    def _new_child(self, view):
        stage = self.program["stages"][self.stage_index]
        goal, profile = self.stage_goal, PROFILES[stage["profile"]]
        remaining = self._watchdog.limit-self.stage_actions
        if remaining <= 0:
            self._pause("program_watchdog_exhausted")
            return
        if self._phase == "gripper":
            kind = "hold" if stage["gripper"] == "close" else "release"
            ident, step = self._interaction_id(), self.start_step+self.actions_executed
            if kind == "release":
                # The next physical opening starts a new release reference.
                # Legacy monitoring must bind it to this operation's held
                # object, not reuse the preceding operation's release ID.
                ident = ((self.operation_runtime.bound_object_id if self.operation_runtime is not None else None)
                         or self.motion_context["held_object_id"] or self.interaction_object_ids.get("hold")
                         or self.interaction_object_ids.get("release"))
            if self.operation_runtime is None:
                self.interaction_monitor.begin(ident, view, step, kind=kind)
                self.interaction_starts[kind+":"+str(ident)] = step
                self.interaction_object_ids[kind] = ident
                self.confirmed_interactions.discard(kind+":"+str(ident))
            if kind == "release" and hasattr(self.geometry, "begin_release_probe"):
                self.geometry.begin_release_probe(ident, view, step)
            segment = {"skill": "set_gripper", "xyz_m": None, "quaternion_wxyz": None,
                       "gripper": stage["gripper"]}
        else:
            delta = np.asarray(goal["xyz_m"])-view.position
            angle = float(np.linalg.norm(rotation_error_rad(goal["quaternion_wxyz"], view.state[3:6])))
            fraction = min(1., profile["target_translation_m"]/max(np.linalg.norm(delta), 1e-12),
                           profile["target_rotation_rad"]/max(angle, 1e-12))
            xyz = view.position+fraction*delta
            q = _slerp(measured_pose(view.state)[3:7], goal["quaternion_wxyz"], fraction)
            segment = {"skill": "move_to_pose", "xyz_m": xyz.tolist(), "quaternion_wxyz": q, "gripper": "keep"}
        segment.update(profile=stage["profile"], max_steps=min(stage["max_steps"], remaining), observation_after=False)
        self.child = AdaptiveFragmentSequencePrimitive(view, {"segments": [segment]}, self.geometry,
            self._child_context(), step=self.start_step+self.actions_executed, keep_gripper=self.gripper_command,
            operation_runtime=self.operation_runtime)

    def _check_child_entry(self, view=None):
        if self.child is not None and self.child.done:
            if view is None or not self._try_local_recovery(view, self.child.reason):
                self._pause("program_"+str(self.child.reason))

    def _verify_stage(self, view):
        stage = self.program["stages"][self.stage_index]
        verification = stage["verification"]
        if verification != "arrival":
            kind = "hold" if verification == "visual_hold" else "release"
            self.verification_evidence = self._interaction(kind, view)
            if not self._accept_visual(kind, view, self.verification_evidence):
                self._pause("program_visual_verification_required", [verification+":"+str(self._interaction_id(kind))])
                return
            if kind == "release":
                self._release_payload_binding(self.verification_evidence)
        self.completed_stages.append({"index": self.stage_index, "goal": deepcopy(self.stage_goal),
            "verification": verification, "visual_evidence": deepcopy(self.verification_evidence),
            "actions": self._prior_stage_actions+self.stage_actions})
        operation = self._current_operation()
        if operation is not None and self.stage_index == operation["last_stage"]:
            self.completed_operations.append({"id": operation["id"], "object_id": operation["object_id"],
                "destination_id": operation["destination_id"], "completed_at_step": self.start_step+self.actions_executed,
                "release_evidence": deepcopy(self.verification_evidence)})
        if self.stage_index+1 == len(self.program["stages"]):
            self._finish("succeeded", "program_complete", {})
            self._evidence = self.progress()
        else:
            self.stage_index += 1
            self.child, self.stage_goal, self.verification_evidence = None, None, None
            self._recovery, self._excluded_targets = None, []
            self._enter_stage(view)

    def _next(self, view):
        return self.child.next_action(view)

    def _observe(self, before, action, after):
        feedback = self.child.observe_transition(before, action, after)
        self._last_state = np.asarray(after.state).copy()
        self.stage_actions += 1
        self.gripper_command = self.child.gripper_command
        if self.operation_runtime is None:
            self.interaction_monitor.observe(after, self.start_step+self.actions_executed)
        else:
            self.operation_runtime.observe_transition(before, after, self._child_context(), self.gripper_command,
                self.start_step+self.actions_executed)
        self._remember_pose(after)
        if self._recovery is not None:
            if self.stage_actions >= self._watchdog.limit:
                self._pause("program_watchdog_exhausted")
            elif self.child.done:
                self._complete_recovery(after)
            return {"program": self.progress(), "stage_feedback": feedback}
        stage = self.program["stages"][self.stage_index]
        evidence = None
        released_binding = False
        if stage["verification"] in ("visual_hold", "visual_release"):
            kind = "hold" if stage["verification"] == "visual_hold" else "release"
            key = kind+":"+str(self._interaction_id(kind))
            evidence = self._interaction(kind, after)
            already_confirmed = key in self.confirmed_interactions
            if self._accept_visual(kind, after, evidence):
                if not already_confirmed:
                    self._pending_visual_revalidation = True
                if kind == "release":
                    released_binding = self._release_payload_binding(evidence)
        distance = float(np.linalg.norm(np.asarray(self.stage_goal["xyz_m"])-after.position))
        angle = float(np.linalg.norm(rotation_error_rad(self.stage_goal["quaternion_wxyz"], after.state[3:6])))
        self._watchdog.learn_once(self.stage_actions, distance, angle, self.child._tracking)
        if released_binding and not (self.child.done and self.child.reason == "fragment_complete"
                and distance <= PROFILES[stage["profile"]]["position_tolerance_m"]
                and angle <= PROFILES[stage["profile"]]["rotation_tolerance_rad"]):
            # The physical payload role changed with accepted evidence. Recheck
            # the same goal without a payload, preserving the parent watchdog;
            # protected child continuations must not silently change held ID.
            self._new_child(after)
            self._check_child_entry(after)
            return {"program": self.progress(), "stage_feedback": feedback}
        if (not self.child.done or self.child.reason == "fragment_complete") and self._refresh_goal(after):
            return {"program": self.progress(), "stage_feedback": feedback}
        if self.done:
            return {"program": self.progress(), "stage_feedback": feedback}
        if self.child.done:
            if self.child.reason != "fragment_complete":
                saved = self.child.export_continuation()
                if (self._pending_visual_revalidation and evidence is not None
                        and evidence.get("status") == "supported" and saved is not None):
                    self._pending_visual_revalidation = False
                    self.child = AdaptiveFragmentSequencePrimitive.from_continuation(after, saved, self.geometry,
                        self._child_context(), step=self.start_step+self.actions_executed, keep_gripper=self.gripper_command,
                        operation_runtime=self.operation_runtime)
                    self._check_child_entry(after)
                elif self._try_local_recovery(after, self.child.reason):
                    pass
                else:
                    self._pause("program_"+str(self.child.reason))
            elif self._phase == "move" and (distance > PROFILES[stage["profile"]]["position_tolerance_m"] or
                                             angle > PROFILES[stage["profile"]]["rotation_tolerance_rad"]):
                self._new_child(after)
                self._check_child_entry(after)
            elif self._phase == "move" and stage["gripper"] != "keep":
                self._phase = "gripper"
                self._new_child(after)
                self._check_child_entry(after)
            else:
                if (self.operation_runtime is None and self._phase == "gripper" and stage["gripper"] == "close"
                        and hasattr(self.geometry, "begin_attachment_probe")):
                    self.geometry.begin_attachment_probe(self._interaction_id(), after, self.start_step+self.actions_executed)
                self._phase = "verify"
                self._verify_stage(after)
        elif self.stage_actions >= self._watchdog.limit:
            self._pause("program_watchdog_exhausted")
        return {"program": self.progress(), "stage_feedback": feedback}

    def progress(self):
        status = "pending_verification" if self.done and self.reason != "program_complete" else self.status or "running"
        operation = self._current_operation()
        completed = {item["id"] for item in self.completed_operations}
        return {"version": PROGRAM_VERSION, "status": status, "primitive_status": self.status or "running", "reason": self.reason,
            "program_complete": len(self.completed_stages) == len(self.program["stages"]),
            "stage_index": self.stage_index, "stage_phase": self._phase,
            "stage_goal": deepcopy(self.stage_goal), "stage_actions": self.stage_actions,
            "stage_consumed_steps": getattr(self, "_prior_stage_actions", 0)+self.stage_actions,
            "remaining_watchdog_steps": max(0, self._watchdog.limit-self.stage_actions) if hasattr(self, "_watchdog") else None,
            "completed_stages": deepcopy(self.completed_stages), "pending_stages": deepcopy(self.program["stages"][self.stage_index:]),
            "motion_context": deepcopy(self.motion_context), "missing_evidence_requests": list(self.missing_evidence_requests),
            "verification_evidence": deepcopy(self.verification_evidence), "gripper_command": self.gripper_command,
            "interaction_object_ids": deepcopy(self.interaction_object_ids),
            "local_recovery": deepcopy(self._recovery), "recovery_events": deepcopy(self.recovery_events),
            "goal_updates": deepcopy(self.goal_updates),
            "operation_contract": deepcopy(self.operation_contract), "current_operation": deepcopy(operation),
            "completed_operations": deepcopy(self.completed_operations),
            "pending_operations": [deepcopy(item) for item in (self.operation_contract or {}).get("operations", [])
                if item["id"] not in completed], "payload_release_events": deepcopy(self.payload_release_events),
            "child": self.child.progress() if self.child else None, "task_success": "unknown"}

    def export_continuation(self):
        if not self.done or self.reason == "program_complete" or self._watchdog.limit <= self.stage_actions:
            return None
        child_saved = self.child.export_continuation() if self.child else None
        # A stall has no child continuation, but the authorized semantic goal
        # and the parent's finite remaining budget still survive the handoff.
        whole = deepcopy((self.geometry.last_motion or {}).get("whole_goal_assessment", {})) if hasattr(self.geometry, "last_motion") else {}
        if self.missing_evidence_requests:
            whole.update(status="unknown", reason=self.reason, missing_evidence_requests=list(self.missing_evidence_requests),
                         semantic_goal=deepcopy(self.program["stages"][self.stage_index]["goal"]))
        return {"kind": "program", "program": deepcopy(self.program), "stage_index": self.stage_index,
            "operation_contract": deepcopy(self.operation_contract),
            "stage_operation_ids": deepcopy(self.stage_operation_ids), "operation_ranges": deepcopy(self.operation_ranges),
            "completed_operations": deepcopy(self.completed_operations), "payload_release_events": deepcopy(self.payload_release_events),
            "stage_goal": deepcopy(self.stage_goal), "phase": self._phase,
            "completed_stages": deepcopy(self.completed_stages), "interaction_starts": deepcopy(self.interaction_starts),
            "confirmed_interactions": sorted(self.confirmed_interactions),
            "confirmation_signatures": deepcopy(self._confirmation_signatures),
            "interaction_object_ids": deepcopy(self.interaction_object_ids),
            "recovery_events": deepcopy(self.recovery_events), "recovery_keys": sorted(self.recovery_keys),
            "recovery_attempts": deepcopy(self.recovery_attempts), "recovery": deepcopy(self._recovery),
            "goal_updates": deepcopy(self.goal_updates), "breadcrumbs": deepcopy(self._breadcrumbs),
            "excluded_targets": deepcopy(self._excluded_targets),
            "remaining_watchdog_steps": self._watchdog.limit-self.stage_actions, "watchdog_learned": self._watchdog.learned,
            "child_continuation": child_saved, "gripper_command": self.gripper_command,
            "motion_context": deepcopy(self.motion_context), "current_target": deepcopy(
                child_saved["current_target"] if self._recovery is not None and child_saved is not None else self.stage_goal),
            "whole_goal_assessment": whole, "missing_evidence_requests": list(self.missing_evidence_requests),
            "remaining_stages": deepcopy(self.program["stages"][self.stage_index:]),
            "stage_consumed_steps": self._prior_stage_actions+self.stage_actions,
            "consumed_steps": self._prior_actions+self.actions_executed}

    @classmethod
    def from_continuation(cls, view, saved, geometry, motion_context=None, *, step=0, keep_gripper=None, interaction_monitor=None, operation_runtime=None):
        if saved.get("kind") != "program" or type(saved.get("remaining_watchdog_steps")) is not int or saved["remaining_watchdog_steps"] <= 0:
            raise ValueError("program continuation has no remaining authorized watchdog")
        current = motion_context or saved["motion_context"]
        if not _same_intent(saved["motion_context"], current) or keep_gripper not in (None, saved["gripper_command"]):
            raise ValueError("program continuation cannot change current intent or gripper")
        program = deepcopy(saved["program"])
        program["stages"][saved["stage_index"]]["motion_context"] = deepcopy(current)
        return cls(view, program, geometry, step=step, keep_gripper=saved["gripper_command"],
                   interaction_monitor=interaction_monitor, operation_runtime=operation_runtime, _saved=saved)
