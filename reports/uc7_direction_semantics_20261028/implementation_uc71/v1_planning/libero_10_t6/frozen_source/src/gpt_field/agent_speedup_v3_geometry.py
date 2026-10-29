"""Persistent, explicitly uncertain geometry for agent_speedup_v3.

Only executive RGB estimates, synchronized camera calibration, known robot parts,
measured state and existing native visual evidence enter this module. Predictions
are never physical clearance certificates. No simulator/evaluator access occurs.
"""
from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict
from copy import copy, deepcopy
import math
from numbers import Real

import numpy as np

from .astranomy_agentic_motion import measured_pose
from .action import TRANSLATION_SCALE_M, ROTATION_SCALE_RAD, _rotvec_quaternion, _quaternion_product
from .robot_tool_geometry import _local_parts

GEOMETRY_VERSION = "agent-speedup-v3-geometry-8"
MAX_SWEEP_INTERVALS = 192
MAX_SWEEP_REFINEMENTS = 3
MAX_COVERAGE_WITNESSES = 6
MAX_UNION_CELLS = 64
ROLES = ("obstacle", "object", "graspable_part", "support_surface", "destination_region", "free_region")
COVERAGE = ("partial", "visible_surfaces", "model_completion")
MAX_ENTITIES = 24
MAX_UPDATE_ENTITIES = 12
MAX_BOXES = 8
_CONTACT_INTENTS = {"approach", "grasp", "place", "manipulate"}
_SIGNS = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


def _object(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def scene_update_schema():
    def vec(n):
        return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}
    ident = {"type": "string", "minLength": 1, "maxLength": 80}
    ids = {"type": "array", "items": ident, "maxItems": MAX_UPDATE_ENTITIES}
    box = _object({"center_local_m": vec(3), "half_extents_m": vec(3), "quaternion_wxyz": vec(4)})
    anchors = _object({"external_uv": vec(2), "wrist_uv": vec(2), "same_feature": {"type": "boolean"},
                       "anchor_local_m": {"anyOf": [vec(3), {"type": "null"}]}})
    entity = _object({
        "id": ident, "role": {"type": "string", "enum": list(ROLES)},
        "parent_id": {"anyOf": [ident, {"type": "null"}]},
        "frame": {"type": "string", "enum": ["world"]}, "position_m": vec(3),
        "quaternion_wxyz": vec(4),
        "boxes": {"type": "array", "items": box, "minItems": 1, "maxItems": MAX_BOXES},
        "uncertainty_m": {"type": "number", "minimum": 0, "maximum": 2},
        "coverage": {"type": "string", "enum": list(COVERAGE)},
        "attachment": {"type": "string", "enum": ["none", "held"]},
        "related_ids": ids, "anchors": {"anyOf": [anchors, {"type": "null"}]},
    })
    return _object({"entities": {"type": "array", "items": entity, "maxItems": MAX_UPDATE_ENTITIES},
                    "invalidate_ids": ids, "remove_ids": ids})


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value, length, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(f"{name} must have {length} entries")
    result = [_number(v, name) for v in value]
    if max(map(abs, result), default=0) > 1e6:
        raise ValueError(f"{name} exceeds numeric bounds")
    return result


def _quaternion(value):
    q = np.asarray(_vector(value, 4, "quaternion_wxyz"))
    length = float(np.linalg.norm(q))
    if length < 1e-12:
        raise ValueError("quaternion_wxyz must be nonzero")
    return q / length


def _ids(value, name):
    if not isinstance(value, (list, tuple)) or len(value) > MAX_UPDATE_ENTITIES:
        raise ValueError(f"{name} requires at most {MAX_UPDATE_ENTITIES} identifiers")
    if any(not isinstance(v, str) or not 1 <= len(v) <= 80 for v in value) or len(set(value)) != len(value):
        raise ValueError(f"{name} identifiers must be unique nonempty strings")
    return list(value)


def validate_scene_update(value):
    """Strict, detached JSON contract; uncertainty is a model estimate, not sigma."""
    schema = scene_update_schema()
    if not isinstance(value, Mapping) or set(value) != set(schema["required"]):
        raise ValueError("scene_update fields must match the schema")
    entities = value["entities"]
    if not isinstance(entities, (list, tuple)) or len(entities) > MAX_UPDATE_ENTITIES:
        raise ValueError("scene_update.entities must have at most 12 entries")
    required = set(schema["properties"]["entities"]["items"]["required"])
    result = {"entities": [], "invalidate_ids": _ids(value["invalidate_ids"], "invalidate_ids"),
              "remove_ids": _ids(value["remove_ids"], "remove_ids")}
    seen = set()
    for raw in entities:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("scene entity fields must match the schema")
        entity = deepcopy(dict(raw))
        _ids([entity["id"]], "id")
        if entity["id"] in seen:
            raise ValueError("duplicate entity id")
        seen.add(entity["id"])
        if entity["parent_id"] is not None:
            _ids([entity["parent_id"]], "parent_id")
            if entity["parent_id"] == entity["id"]:
                raise ValueError("entity cannot be its own parent")
        if entity["frame"] != "world" or entity["role"] not in ROLES or entity["coverage"] not in COVERAGE:
            raise ValueError("unsupported scene frame, role or coverage")
        if entity["attachment"] not in ("none", "held"):
            raise ValueError("attachment must be none or held")
        if entity["attachment"] == "held" and entity["role"] in ("free_region", "destination_region", "support_surface"):
            raise ValueError("regions and support surfaces cannot be attached payloads")
        entity["position_m"] = _vector(entity["position_m"], 3, "position_m")
        entity["quaternion_wxyz"] = _quaternion(entity["quaternion_wxyz"]).tolist()
        entity["related_ids"] = _ids(entity["related_ids"], "related_ids")
        uncertainty = _number(entity["uncertainty_m"], "uncertainty_m")
        if not 0 <= uncertainty <= 2:
            raise ValueError("uncertainty_m must be between zero and two metres")
        entity["uncertainty_m"] = uncertainty
        boxes = entity["boxes"]
        if not isinstance(boxes, (list, tuple)) or not 1 <= len(boxes) <= MAX_BOXES:
            raise ValueError("entity requires 1 through 8 separate solid boxes")
        for box in boxes:
            if not isinstance(box, dict) or set(box) != {"center_local_m", "half_extents_m", "quaternion_wxyz"}:
                raise ValueError("box fields must match the schema")
            box["center_local_m"] = _vector(box["center_local_m"], 3, "center_local_m")
            box["half_extents_m"] = _vector(box["half_extents_m"], 3, "half_extents_m")
            if any(not 0 < v <= 5 for v in box["half_extents_m"]):
                raise ValueError("box half extents must be positive and at most five metres")
            box["quaternion_wxyz"] = _quaternion(box["quaternion_wxyz"]).tolist()
        anchors = entity["anchors"]
        if anchors is not None:
            if (not isinstance(anchors, dict) or set(anchors) not in (
                    {"external_uv", "wrist_uv", "same_feature"},
                    {"external_uv", "wrist_uv", "same_feature", "anchor_local_m"})
                    or type(anchors["same_feature"]) is not bool):
                raise ValueError("anchors must identify a corresponding pair of pixels")
            for label in ("external_uv", "wrist_uv"):
                anchors[label] = _vector(anchors[label], 2, label)
            if anchors.get("anchor_local_m") is not None:
                anchors["anchor_local_m"] = _vector(anchors["anchor_local_m"], 3, "anchor_local_m")
        result["entities"].append(entity)
    if seen & set(result["remove_ids"]) or set(result["remove_ids"]) & set(result["invalidate_ids"]):
        raise ValueError("conflicting scene update/remove/invalidate identifiers")
    return result


def _rotation(q):
    w, x, y, z = _quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _matrix_quaternion(matrix):
    # Stable eigenvector form, including rotations arbitrarily close to pi.
    r = np.asarray(matrix, float)
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def _slerp(a, b, fraction):
    a, b = _quaternion(a), _quaternion(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        return _quaternion(a + fraction*(b-a))
    theta = math.acos(float(np.clip(dot, -1, 1)))
    return (math.sin((1-fraction)*theta)*a + math.sin(fraction*theta)*b) / math.sin(theta)


def _box(center, rotation, half, owner, name, uncertainty=0.):
    return {"center": np.asarray(center), "rotation": np.asarray(rotation), "half": np.asarray(half),
            "owner": owner, "name": name, "uncertainty": float(uncertainty)}


def _corners(box):
    return (_SIGNS * box["half"]) @ box["rotation"].T + box["center"]


def obb_separation(a, b, padding_a=0., padding_b=0.):
    """Maximum normalized SAT gap; positive is a lower bound for these OBBs.

    Negative means estimated OBB overlap, NOT measured physical penetration.
    Solid intersection and containment are covered, unlike vertex-only checks.
    """
    axes = [*a["rotation"].T, *b["rotation"].T]
    axes += [np.cross(x, y) for x in a["rotation"].T for y in b["rotation"].T]
    axes = np.asarray([axis / np.linalg.norm(axis) for axis in axes if np.linalg.norm(axis) > 1e-10])
    delta = b["center"] - a["center"]
    ra = np.abs(axes @ a["rotation"]) @ (a["half"] + padding_a)
    rb = np.abs(axes @ b["rotation"]) @ (b["half"] + padding_b)
    return float(np.max(np.abs(axes @ delta) - ra - rb))


def _separations(a, centers, rotations, halves, padding_a=0.):
    """Vectorized exact SAT gaps against all declared obstacle OBBs."""
    count = len(centers)
    if not count:
        return np.empty(0)
    left = np.broadcast_to(a["rotation"].T, (count, 3, 3))
    right = rotations.transpose(0, 2, 1)
    cross = np.cross(left[:, :, None, :], right[:, None, :, :]).reshape(count, 9, 3)
    axes = np.concatenate((left, right, cross), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    valid = lengths > 1e-10
    axes = axes / np.maximum(lengths[:, :, None], 1e-10)
    ra = np.abs(axes @ a["rotation"]) @ (a["half"] + padding_a)
    rb = np.einsum("nki,ni->nk", np.abs(np.einsum("nki,nij->nkj", axes, rotations)), halves)
    gaps = np.abs(np.einsum("nki,ni->nk", axes, centers-a["center"])) - ra - rb
    return np.max(np.where(valid, gaps, -np.inf), axis=1)


def _contained(inner, outer, padding=0.):
    half = outer["half"] - outer["uncertainty"]
    if np.any(half <= 0):
        return False
    center = (inner["center"] - outer["center"]) @ outer["rotation"]
    extent = np.abs(outer["rotation"].T @ inner["rotation"]) @ (inner["half"] + padding)
    return bool(np.all(np.abs(center) + extent <= half))


def _containment_margin(inner, outer, padding=0.):
    """Minimum face margin of a fully enclosed uncertain OBB."""
    center = (inner["center"]-outer["center"]) @ outer["rotation"]
    extent = np.abs(outer["rotation"].T @ inner["rotation"]) @ (inner["half"]+padding)
    return float(np.min(outer["half"]-outer["uncertainty"]-np.abs(center)-extent))


def _union_coverage(inner, regions, padding=0.):
    """Prove volume coverage by bounded OBB subdivision, never corner voting.

    Each accepted cell is entirely inside one uncertainty-eroded region. If
    subdivision runs out of work, the unresolved cells remain unknown.
    """
    regions = [region for region in regions if np.all(region["half"] > region["uncertainty"])]
    if not regions:
        return False, set(), 0
    corners = _corners({**inner, "half": inner["half"]+padding})
    inside_union = np.zeros(len(corners), bool)
    for region in regions:
        inside_union |= np.all(np.abs((corners-region["center"]) @ region["rotation"])
                               <= region["half"]-region["uncertainty"], axis=1)
    if not inside_union.all():
        return False, set(), 0
    pending = [(inner["center"], inner["half"]+padding)]
    owners, cells = set(), 0
    while pending and cells < MAX_UNION_CELLS:
        center, half = pending.pop()
        cell = {**inner, "center": center, "half": half}
        cells += 1
        containing = next((r for r in regions if _contained(cell, r)), None)
        if containing is not None:
            owners.add(containing["owner"])
            continue
        # A separating axis proves that this whole cell is outside every region;
        # partitioning it further cannot produce complete coverage.
        if all(obb_separation(cell, r, padding_b=-r["uncertainty"]) > 0 for r in regions):
            return False, owners, cells
        nearest = max(regions, key=lambda r: _containment_margin(cell, r))
        relative = (center-nearest["center"]) @ nearest["rotation"]
        extent = np.abs(nearest["rotation"].T @ inner["rotation"]) @ half
        face = int(np.argmin(nearest["half"]-nearest["uncertainty"]-np.abs(relative)-extent))
        axis = int(np.argmax(np.abs(inner["rotation"].T @ nearest["rotation"][:, face])*half))
        if half[axis] <= 1e-5:
            return False, owners, cells
        child = half.copy(); child[axis] *= .5
        offset = inner["rotation"][:, axis]*child[axis]
        pending.extend(((center-offset, child), (center+offset, child)))
    return not pending, owners, cells


def _box_key(box):
    return (box["owner"], box["name"], tuple(box["center"]), tuple(box["rotation"].ravel()),
            tuple(box["half"]), box["uncertainty"], box.get("alternative_for"))


def _coverage_witness(body, regions, body_uncertainty, padding, fraction):
    """Describe a failed model containment check, never an observed collision."""
    witness = {"body_id": body["owner"], "body_part": body["name"], "path_fraction": float(fraction),
               "world_point_m": body["center"].tolist(), "region_id": None, "region_part": None,
               "axis": None, "face": None, "uncertainty_m": None,
               "body_uncertainty_m": float(body_uncertainty), "interpolation_padding_m": float(padding),
               "nominal_margin_m": None, "uncertainty_margin_m": None, "padded_margin_m": None,
               "coverage_kind": "no_free_region", "source": "model_geometry_containment_prediction",
               "physical_clearance_verified": False}
    choices = []
    for region in regions:
        relative = (body["center"]-region["center"]) @ region["rotation"]
        axes = np.abs(region["rotation"].T @ body["rotation"])
        nominal = region["half"]-np.abs(relative)-axes @ body["half"]
        uncertain = nominal-region["uncertainty"]-axes @ np.full(3, body_uncertainty)
        padded = uncertain-axes @ np.full(3, padding)
        choices.append((float(padded.min()), region, relative, nominal, uncertain, padded, axes))
    if not choices:
        return witness
    _, region, relative, nominal, uncertain, padded, axes = max(choices, key=lambda row: row[0])
    axis = int(np.argmin(padded)); sign = 1. if relative[axis] >= 0 else -1.
    corners = _corners({**body, "half": body["half"]+body_uncertainty+padding})
    outside_union = np.ones(len(corners), bool)
    for other in regions:
        outside_union &= ~np.all(np.abs((corners-other["center"]) @ other["rotation"])
                                <= other["half"]-other["uncertainty"], axis=1)
    directional = ((corners-region["center"]) @ region["rotation"])[:, axis]*sign
    possible = np.flatnonzero(outside_union)
    corner = int(possible[np.argmax(directional[possible])]) if len(possible) else int(np.argmax(directional))
    kind = ("nominal_boundary" if nominal[axis] < 0 else
            "uncertainty_boundary" if uncertain[axis] < 0 else "interpolation_bound")
    if not len(possible):
        kind = "single_region_fit_union_not_proven"
    witness.update(world_point_m=corners[corner].tolist(), region_id=region["owner"], region_part=region["name"],
                   axis="xyz"[axis], face="positive" if sign > 0 else "negative",
                   region_axis_world=region["rotation"][:, axis].tolist(), uncertainty_m=region["uncertainty"],
                   nominal_margin_m=float(nominal[axis]), uncertainty_margin_m=float(uncertain[axis]),
                   padded_margin_m=float(padded[axis]), interpolation_axis_margin_m=float(axes[axis].sum()*padding),
                   deficit_m=max(0., -float(padded[axis])), coverage_kind=kind,
                   point_outside_eroded_region_union=bool(len(possible)))
    return witness


def _bounded_witnesses(witnesses):
    # Preserve the worst witness for each distinct body/boundary. Route onset is
    # reported separately; repeated samples must not crowd out another part.
    grouped = {}
    for witness in witnesses:
        key = tuple(witness.get(k) for k in ("body_part", "region_part", "axis", "face"))
        if key not in grouped or (witness.get("deficit_m") or 0) > (grouped[key].get("deficit_m") or 0):
            grouped[key] = witness
    return sorted(grouped.values(), key=lambda row: (-(row.get("deficit_m") or 0), row["path_fraction"]))[:MAX_COVERAGE_WITNESSES]


class _PreviewVisualWatch:
    """Read existing monitoring evidence without loading RGB or resetting it."""
    def __init__(self, entities):
        self.reports = {ident: deepcopy(entity.get("visual_monitoring") or {}) for ident, entity in entities.items()}

    def capture(self, ident, boxes, view, step, *, occluder_boxes=()):
        report = {"status": "unverified", "reason": "preview_model_update_no_new_visual_baseline",
                  "geometry_verified": False, "free_space_verified": False}
        self.reports[ident] = report
        return deepcopy(report)

    def assess(self, ident, boxes, view, step, *, occluder_boxes=()):
        return deepcopy(self.reports.get(ident, {"status": "unverified", "reason": "preview_no_reference",
                                                "geometry_verified": False, "free_space_verified": False}))

    def forget(self, ids):
        for ident in ids:
            self.reports.pop(ident, None)


def _triangulation(anchors, view):
    if anchors is None or not anchors["same_feature"]:
        return {"status": "unavailable", "reason": "no_asserted_correspondence"}
    geometry = getattr(view.observation, "geometry", None)
    if not isinstance(geometry, Mapping):
        return {"status": "unavailable", "reason": "no_synchronized_camera_metadata"}
    rays = []
    try:
        for name in ("external", "wrist"):
            cam = geometry["cameras"][name]
            uv = anchors[name + "_uv"]
            if not 0 <= uv[0] < geometry["image"]["width"] or not 0 <= uv[1] < geometry["image"]["height"]:
                return {"status": "unavailable", "reason": "pixel_outside_image"}
            fx, fy = cam["focal_length_px"]
            cx, cy = cam["principal_point_px"]
            local = np.array([(uv[0]-cx)/fx, -(uv[1]-cy)/fy, -1.])
            ray = np.asarray(cam["rotation_world_from_camera"]) @ local
            rays.append((np.asarray(cam["position_world_m"]), ray/np.linalg.norm(ray)))
        (a, da), (b, db) = rays
        matrix = np.column_stack((da, -db))
        cosine = float(np.clip(np.dot(da, db), -1, 1))
        angle = math.acos(abs(cosine))
        if angle < math.radians(2):
            return {"status": "degenerate", "ray_angle_deg": math.degrees(angle), "reason": "insufficient_ray_angle"}
        distances = np.linalg.lstsq(matrix, b-a, rcond=None)[0]
        pa, pb = a+distances[0]*da, b+distances[1]*db
        gap = float(np.linalg.norm(pa-pb))
        usable = bool(np.all(distances > 0) and gap <= .02)
        return {"status": "conditional_estimate" if usable else "inconsistent", "ray_gap_m": gap,
                "ray_angle_deg": math.degrees(angle), "position_world_m": ((pa+pb)/2).tolist(),
                "source": "model_pixel_correspondence_plus_camera_rays", "verified_correspondence": False,
                "used_to_replace_entity_pose": False}
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return {"status": "unavailable", "reason": "invalid_camera_or_correspondence_data"}


class GeometryMemory:
    """Bounded episode memory; no per-step reconstruction or extra model call."""
    def __init__(self, visual_watch=None, local_perception=None):
        if visual_watch is None:
            from .agent_speedup_v3_visual_watch import VisualGeometryWatch
            visual_watch = VisualGeometryWatch()
        self.visual_watch = visual_watch
        if local_perception is None:
            from .agent_speedup_v3_perception import LocalRGBGeometryEvidence
            local_perception = LocalRGBGeometryEvidence()
        self.local_perception = local_perception
        self.entities = {}
        self.step = 0
        self.revision = 0
        self.events = []
        self.last_motion = None
        self._last_gripper_command = None
        self._last_view = None
        self._last_watch_step = None
        self._watched_ids = set()
        self._last_transition_step = None
        self._last_transition_result = None
        self._last_motion_revoke_step = None
        self._entity_box_cache = {}
        self._robot_body_cache = OrderedDict()
        self._sweep_cache = OrderedDict()
        self._route_certificates = OrderedDict()
        self._geometry_epoch = 0
        self._occluder_cache = None
        self.compute_counts = {"entity_box_builds": 0, "entity_box_cache_hits": 0,
                               "sweep_evaluations": 0, "exact_sweep_cache_hits": 0,
                               "supported_route_reuses": 0, "obstacles_pruned": 0}

    def _event(self, kind, ids, step, **details):
        self.events.append({"kind": kind, "ids": sorted(set(ids)), "step": int(step), **details})
        self.events[:] = self.events[-24:]

    def _invalidate(self, ids, step, reason, event_kind=None, *, local_contact=False):
        origins = sorted(set(ids))
        affected = set(ids)
        for _ in range(len(self.entities)):
            previous = len(affected)
            for entity in self.entities.values():
                # parent_id expresses a physical pose dependency, parent ->
                # child. related_ids is semantic association, never a causal
                # edge: touching an object does not move its supporting table.
                if entity["parent_id"] in affected:
                    affected.add(entity["id"])
            if len(affected) == previous:
                break
        # A changed object invalidates intersecting free-space hypotheses, not
        # unrelated background. These free volumes were never measured empty.
        # Expected finger-target contact is a local uncertainty about that
        # object, not evidence that an entire aerial corridor changed. Its stale
        # solid envelope remains checked by every non-exempt hand/payload pair.
        # A free-volume appearance change also cannot move other free volumes.
        changed_boxes = [box for ident in affected if ident in self.entities
                         and self.entities[ident]["role"] != "free_region"
                         for box in self._entity_boxes(self.entities[ident])]
        for entity in self.entities.values():
            if entity["role"] == "free_region" and any(
                obb_separation(region, changed, padding_b=changed["uncertainty"]) <= 0
                for region in self._entity_boxes(entity) for changed in changed_boxes):
                if local_contact:
                    entity["local_uncertain_solid_ids"] = sorted(set(entity.get("local_uncertain_solid_ids", [])) | affected)
                else:
                    affected.add(entity["id"])
        for ident in affected:
            if ident in self.entities:
                entity = self.entities[ident]
                kind = event_kind or reason
                entity.update(stale=True, stale_reason=reason, invalidated_step=int(step),
                              invalidation_kind=kind, invalidation_origin_ids=origins,
                              invalidation_local_contact=bool(local_contact),
                              invalidation_origins=sorted(set(entity.get("invalidation_origins", [])) | set(origins)),
                              nonsettling_invalidation=bool(entity.get("nonsettling_invalidation") or
                                                           kind not in ("release_command", "possible_contact")))
        if affected:
            self._geometry_epoch += 1
            self._route_certificates.clear()
            self._event(reason, affected, step, origin_ids=origins, event_kind=event_kind or reason,
                        propagation=("physical_parent_and_local_solid_exclusions" if local_contact else
                                     "physical_parent_then_intersecting_free_regions"))

    def update_from_model(self, update, view, step):
        clean = validate_scene_update(update)
        self.step, self._last_view = int(step), view
        # Actual opening and its evidence baseline survive replacement of a
        # model hypothesis. Snapshot before model revision invalidation mutates
        # the old estimate; no timestamp is reconstructed from the new model.
        pending_releases = {ident: deepcopy(entity) for ident, entity in self.entities.items()
                            if entity.get("release_pending")}
        self._invalidate(clean["invalidate_ids"], step, "model_invalidation")
        self._invalidate(clean["remove_ids"], step, "dependency_removed")
        self.visual_watch.forget(clean["remove_ids"])
        if self.local_perception is not None:
            self.local_perception.forget(clean["remove_ids"])
        for ident in clean["remove_ids"]:
            self.entities.pop(ident, None)
            self._entity_box_cache.pop(ident, None)
        # A revised solid estimate may invalidate dependent corridors even when
        # the model omitted invalidate_ids. This is a changed hypothesis, not a
        # claim that RGB measured object motion. Apply before the whole batch so
        # a simultaneously refreshed corridor receives its own new baseline.
        changed = []
        for entity in clean["entities"]:
            prior = self.entities.get(entity["id"])
            if prior is not None and (prior["role"] != "free_region" or entity["role"] != "free_region") and (
                any(prior[key] != entity[key] for key in
                    ("position_m", "quaternion_wxyz", "boxes", "attachment", "parent_id", "role"))
                or entity["uncertainty_m"] > prior["uncertainty_m"]):
                changed.append(entity["id"])
        self._invalidate(changed, step, "model_geometry_revision")
        pose = measured_pose(view.state)
        hand_r, hand_p = _rotation(pose[3:7]), np.asarray(pose[:3])
        for entity in clean["entities"]:
            ident = entity["id"]
            prior = self.entities.get(ident)
            entity.update(source="model_visual_estimate", observation_step=int(step), updated_step=int(step),
                          stale=False, stale_reason=None, uncertainty_kind="uncalibrated_model_extent_estimate",
                          triangulation=_triangulation(entity["anchors"], view), physically_verified=False,
                          visual_monitoring=None, monitoring_unknown=False,
                          geometry_revision=1 if prior is None else prior.get("geometry_revision", 0)+1,
                          local_pose_revision=0 if prior is None else prior.get("local_pose_revision", 0),
                          local_pose_evidence=None)
            if entity["attachment"] == "held":
                entity["attachment_offset_hand_m"] = (hand_r.T @ (np.asarray(entity["position_m"])-hand_p)).tolist()
                entity["attachment_rotation_hand"] = (hand_r.T @ _rotation(entity["quaternion_wxyz"])).tolist()
                entity["attachment_source"] = "model_attachment_hypothesis"
            if prior is not None:
                entity["previous_observation_step"] = prior["observation_step"]
            self.entities[ident] = entity
            self._entity_box_cache.pop(ident, None)
            if ident in pending_releases:
                self._preserve_pending_release(pending_releases[ident], entity, view)
        while len(self.entities) > MAX_ENTITIES:
            candidates = [e for e in self.entities.values() if e["attachment"] != "held"]
            if not candidates:
                candidates = list(self.entities.values())
            victim = min(candidates, key=lambda e: (not e["stale"], e["observation_step"]))
            self.visual_watch.forget([victim["id"]])
            del self.entities[victim["id"]]
        for raw in clean["entities"]:
            entity = self.entities.get(raw["id"])
            if entity is None:
                continue
            if (self.local_perception is not None and entity["attachment"] != "held"
                    and entity["role"] not in ("free_region", "destination_region")):
                anchor = self.local_perception.anchor_pose(entity["id"], entity, view, step)
                entity["local_anchor_evidence"] = deepcopy(anchor)
                if anchor.get("status") == "supported":
                    self.apply_visual_displacement(anchor, view, step)
                entity["local_perception_reference"] = self.local_perception.capture(
                    entity["id"], self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                    entity["geometry_revision"], occluder_boxes=self._visual_occluders(view, entity["id"]))
            if entity["attachment"] == "held":
                # Static appearance correspondence is invalid for moving payloads.
                self.visual_watch.forget([entity["id"]])
                entity["visual_monitoring"] = {"status": "unverified", "reason": "moving_attachment_not_tracked",
                                                "geometry_verified": False, "free_space_verified": False}
            else:
                entity["visual_monitoring"] = self.visual_watch.capture(
                    entity["id"], self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                    occluder_boxes=self._visual_occluders(view, entity["id"]))
            entity["visual_checked_step"] = int(step)
            self._watched_ids.discard(entity["id"])
        self.revision += 1
        if clean["entities"] or clean["invalidate_ids"] or clean["remove_ids"]:
            self._geometry_epoch += 1
            self._route_certificates.clear()
        self._event("model_update", [e["id"] for e in clean["entities"]], step)
        return {"accepted": len(clean["entities"]), "revision": self.revision, "entity_count": len(self.entities)}

    def _preserve_pending_release(self, prior, entity, view):
        """Keep actual command provenance independent of replaceable estimates.

        Fresh model geometry stays the current hypothesis. Conservative old/new
        carried and stationary alternatives remain until accepted visual release;
        attachment='none' alone cannot erase a possibly carried object.
        """
        old_carried = self._entity_boxes(prior, view)
        old_world = self._entity_boxes({**prior, "attachment": "none",
            "position_m": prior.get("release_position_estimate_m", prior["position_m"]),
            "quaternion_wxyz": prior.get("release_quaternion_estimate_wxyz", prior["quaternion_wxyz"])})
        new_world = self._entity_boxes(entity, view)

        def envelope(boxes, old, hand_local):
            corners = np.concatenate([_corners(box) for box in boxes])
            if hand_local:
                corners = (corners-view.position)@view.R
            low, high = corners.min(axis=0), corners.max(axis=0)
            uncertainty = max(box["uncertainty"] for box in boxes)
            if old is not None:
                low = np.minimum(low, np.asarray(old["center"])-old["half"])
                high = np.maximum(high, np.asarray(old["center"])+old["half"])
                uncertainty = max(uncertainty, old["uncertainty"])
            return {"center": ((low+high)/2).tolist(), "half": ((high-low)/2).tolist(),
                    "uncertainty": uncertainty, "source": "pending_actual_release_old_and_new_estimates"}

        entity["release_pending_carried_envelope"] = envelope(old_carried+new_world,
            prior.get("release_pending_carried_envelope"), True)
        entity["release_pending_world_envelope"] = envelope(old_world+new_world,
            prior.get("release_pending_world_envelope"), False)
        entity["release_current_world_estimate_m"] = list(entity["position_m"])
        entity["release_current_quaternion_estimate_wxyz"] = list(entity["quaternion_wxyz"])
        entity["release_model_attachment_estimate"] = entity["attachment"]
        for key in ("release_requested_step", "release_actual_command_step", "release_probe_started",
                    "release_position_estimate_m", "release_quaternion_estimate_wxyz",
                    "release_probe_shape_usable", "attachment_probe_shape_usable"):
            if key in prior:
                entity[key] = deepcopy(prior[key])
        for key in ("attachment_offset_hand_m", "attachment_rotation_hand"):
            if key not in entity and key in prior:
                entity[key] = deepcopy(prior[key])
        entity.update(release_pending=True, attachment_source="uncertain_after_release_command")
        self._entity_box_cache.pop(entity["id"], None)

    def apply_visual_displacement(self, evidence, view, step):
        """Consume fresh metric material motion without shrinking shape risk.

        Existing extents/orientation assignment remain hypotheses. Old-to-new
        occupancy becomes a local uncertain exclusion; unrelated free volumes
        retain their original evidence, and no newly vacated volume becomes free.
        """
        if not isinstance(evidence, Mapping):
            return {"accepted": False, "reason": "local_rgb_displacement_required"}
        ident = evidence.get("entity_id")
        entity = self.entities.get(ident)
        if (entity is None or entity["attachment"] == "held"
                or entity["role"] in ("free_region", "destination_region")
                or evidence.get("status") != "supported"
                or evidence.get("source") != "public_two_view_feature_displacement"
                or evidence.get("observed_at_step") != int(step)
                or evidence.get("reference_geometry_revision") != entity.get("geometry_revision")
                or evidence.get("reference_step", int(step)+1) > int(step)
                or any((evidence.get("views", {}).get(name) or {}).get("status") != "supported"
                       for name in ("external", "wrist"))):
            return {"accepted": False, "reason": "fresh_matching_metric_two_view_evidence_required"}
        if entity.get("stale") and entity.get("invalidation_kind") not in (
                "visual_appearance_changed", "possible_contact", "release_command", "local_observed_displacement"):
            return {"accepted": False, "reason": "unresolved_nonpose_invalidation"}
        try:
            rotation = np.asarray(evidence["world_rotation_delta"], float)
            translation = np.asarray(_vector(evidence["world_translation_delta_m"], 3, "observed_delta"))
            position_margin = _number(evidence["position_uncertainty_m"], "position_uncertainty_m")
            angular_margin = _number(evidence["orientation_uncertainty_rad"], "orientation_uncertainty_rad")
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)
                    or not 0 <= position_margin <= .10 or not 0 <= angular_margin <= .5):
                raise ValueError("unusable observed transform or uncertainty")
        except (KeyError, ValueError, TypeError):
            return {"accepted": False, "reason": "invalid_or_unbounded_observed_displacement"}
        old_boxes = self._entity_boxes(entity, suppress_stale_padding=True)
        old_position = np.asarray(entity["position_m"])
        new_position = rotation@old_position+translation
        if np.linalg.norm(new_position-old_position) > .25:
            return {"accepted": False, "reason": "observed_displacement_exceeds_local_identity_window"}
        radius = max(np.linalg.norm(box["center"]-old_position)+np.linalg.norm(box["half"]) for box in old_boxes)
        additional = position_margin+radius*angular_margin+(.02 if entity.get("stale") else 0.)
        # Only physical descendants lose their unobserved pose. A semantic
        # relationship or a large free-box intersection does not stale a room.
        descendants = set()
        for _ in range(len(self.entities)):
            new = {e["id"] for e in self.entities.values() if e["parent_id"] in descendants | {ident}}
            if new <= descendants:
                break
            descendants.update(new)
        self._invalidate(descendants, step, "local_observed_displacement", local_contact=True)
        prior_uncertainty = entity["uncertainty_m"]
        entity.update(position_m=new_position.tolist(),
                      quaternion_wxyz=_matrix_quaternion(rotation@_rotation(entity["quaternion_wxyz"])).tolist(),
                      uncertainty_m=min(2., prior_uncertainty+additional),
                      stale=False, stale_reason=None, monitoring_unknown=False,
                      nonsettling_invalidation=False, geometry_revision=entity["geometry_revision"]+1,
                      local_pose_revision=entity.get("local_pose_revision", 0)+1,
                      pose_observation_step=int(step), updated_step=int(step), physically_verified=False)
        entity["local_pose_evidence"] = {**deepcopy(dict(evidence)), "accepted": True, "pose_updated": True,
            "geometry_revision": entity["geometry_revision"], "uncertainty_m": entity["uncertainty_m"],
            "shape_source": "retained_model_extent", "model_observation_step": entity["observation_step"]}
        self._entity_box_cache.pop(ident, None)
        # A metric anchor corrects a hypothesis; it does not show physical travel
        # through the former erroneous pose. Temporal motion keeps swept local
        # occupancy unknown until an explicit new scene estimate resolves it.
        if evidence.get("reason") != "explicit_local_anchor_triangulated":
            new_boxes = self._entity_boxes(entity, suppress_stale_padding=True)
            corners = np.concatenate([_corners(box) for box in old_boxes+new_boxes])
            low, high = corners.min(axis=0), corners.max(axis=0)
            previous = entity.get("local_motion_exclusion_boxes", [])
            if previous:
                low = np.minimum(low, np.min([np.asarray(b["center"])-b["half"] for b in previous], axis=0))
                high = np.maximum(high, np.max([np.asarray(b["center"])+b["half"] for b in previous], axis=0))
            entity["local_motion_exclusion_boxes"] = [{"center": ((low+high)/2).tolist(),
                "half": ((high-low)/2).tolist(), "uncertainty": entity["uncertainty_m"],
                "source": "unobserved_local_swept_occupancy"}]
        for free in self.entities.values():
            if free["role"] == "free_region" and any(
                    obb_separation(region, changed, padding_b=changed["uncertainty"]) <= 0
                    for region in self._entity_boxes(free) for changed in old_boxes+self._entity_boxes(entity)):
                free["local_uncertain_solid_ids"] = sorted(set(free.get("local_uncertain_solid_ids", [])) | {ident})
        self._geometry_epoch += 1
        self.revision += 1
        self._route_certificates.clear()
        self._event("local_pose_updated", [ident], step, evidence_id=evidence.get("evidence_id"),
                    observation_step_preserved=True, free_volume_verified=False)
        entity["visual_monitoring"] = self.visual_watch.capture(ident,
            self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
            occluder_boxes=self._visual_occluders(view, ident))
        entity["visual_checked_step"] = int(step)
        return {"accepted": True, "entity_id": ident, "local_pose_revision": entity["local_pose_revision"],
                "geometry_revision": entity["geometry_revision"], "evidence_id": evidence.get("evidence_id"),
                "pose_observation_step": int(step), "physical_clearance_verified": False}

    def observe_local_geometry(self, view, step, entity_ids):
        """Explicit, bounded local perception; no global stereo reconstruction."""
        results = {}
        if self.local_perception is None:
            return results
        for ident in list(dict.fromkeys(entity_ids))[:MAX_UPDATE_ENTITIES]:
            entity = self.entities.get(ident)
            if entity is None or entity["attachment"] == "held" or entity["role"] in ("free_region", "destination_region"):
                continue
            report = self.local_perception.assess(ident,
                self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                entity["geometry_revision"], occluder_boxes=self._visual_occluders(view, ident))
            entity["local_perception_evidence"] = deepcopy(report)
            accepted = self.apply_visual_displacement(report, view, step) if report.get("status") == "supported" else {"accepted": False}
            results[ident] = {**deepcopy(report), **accepted}
            if accepted.get("accepted"):
                entity["local_perception_reference"] = self.local_perception.capture(ident,
                    self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                    entity["geometry_revision"], occluder_boxes=self._visual_occluders(view, ident))
        return results

    def _visual_occluders(self, view, excluded_id=None):
        key = (id(view), self._geometry_epoch, tuple((e["id"], tuple(e["position_m"]), e["uncertainty_m"])
                                                    for e in self.entities.values() if e["attachment"] == "held"))
        if self._occluder_cache is not None and self._occluder_cache[0] == key:
            return [b for b in self._occluder_cache[1] if b["owner"] != excluded_id]
        bodies, _ = self._local_bodies(view, {})
        boxes = [_box(view.position+view.R @ body["center"], view.R @ body["rotation"],
                     body["half"], body["owner"], body["name"], body["uncertainty"])
                 for body in bodies]
        self._occluder_cache = (key, boxes)
        return [b for b in boxes if b["owner"] != excluded_id]

    def watch_observation(self, view, step, evidence_ids=None):
        """Watch only active supporting RGB evidence, before primitive assessment.

        Consistent appearance does not refresh a model estimate, measure free
        volume, prove contact, or clear an earlier invalidation. Expected robot
        occlusion can retain a conditional pose prediction; loss of an already
        available object watch makes that object's evidence temporarily unknown.
        """
        self.step, self._last_view = int(step), view
        if self._last_watch_step != int(step):
            self._last_watch_step = int(step)
            self._watched_ids.clear()
        relevant = set((self.last_motion or {}).get("evidence_ids", []) if evidence_ids is None else evidence_ids)
        checked, changed, unavailable = [], [], []
        for ident in sorted(relevant-self._watched_ids):
            entity = self.entities.get(ident)
            if entity is None or entity["attachment"] == "held":
                continue
            prior = entity.get("visual_monitoring") or {}
            report = self.visual_watch.assess(ident, self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                                              occluder_boxes=self._visual_occluders(view, ident))
            self._watched_ids.add(ident)
            entity["visual_monitoring"] = deepcopy(report)
            entity["visual_checked_step"] = int(step)
            checked.append(ident)
            status = report.get("status")
            if status == "changed":
                local = self.observe_local_geometry(view, step, [ident]).get(ident, {})
                if local.get("accepted"):
                    continue
                changed.append(ident)
                entity["monitoring_unknown"] = True
            elif status in ("consistent", "reference"):
                entity["monitoring_unknown"] = False
                entity["visual_reference_seen"] = True
            else:
                unavailable.append(ident)
                had_reference = entity.get("visual_reference_seen", False) or prior.get("status") in ("reference", "consistent")
                reason = str(report.get("reason", ""))
                self_occluded = bool(report.get("self_occluded", False)) or "self_occlu" in reason
                # A missing/featureless initial watch is explicit diagnostic
                # uncertainty, not an invented change in a fresh model estimate.
                # Unexpected monitoring loss after a usable reference is an
                # event; ordinary known self-occlusion is not a pose change.
                # Known robot occlusion changes observability, not object pose.
                # Keep the conditional prediction without refreshing its age;
                # actual possible contact still invalidates the object below.
                entity["monitoring_unknown"] = bool(had_reference and not self_occluded)
        self._invalidate(changed, step, "visual_appearance_changed")
        return {"checked_ids": checked, "changed_ids": changed, "unavailable_ids": unavailable,
                "source": "public_rgb_change_watch", "observation_age_refreshed": False,
                "physical_clearance_verified": False}

    def _entity_boxes(self, entity, view=None, suppress_stale_padding=False):
        key = (entity.get("geometry_revision"), tuple(entity["position_m"]), tuple(entity["quaternion_wxyz"]),
               entity["uncertainty_m"], entity["stale"], suppress_stale_padding, repr(entity["boxes"]),
               None if view is None or entity["attachment"] != "held" else tuple(np.asarray(view.state)[:6]),
               repr(entity.get("attachment_offset_hand_m")), repr(entity.get("attachment_rotation_hand")))
        cached = self._entity_box_cache.get(entity["id"])
        if cached is not None and cached[0] == key:
            self.compute_counts["entity_box_cache_hits"] += 1
            return cached[1]
        rotation = _rotation(entity["quaternion_wxyz"])
        position = np.asarray(entity["position_m"])
        if entity["attachment"] == "held" and view is not None and "attachment_offset_hand_m" in entity:
            rotation = view.R @ np.asarray(entity["attachment_rotation_hand"])
            position = view.position + view.R @ np.asarray(entity["attachment_offset_hand_m"])
        uncertainty = entity["uncertainty_m"] + (.02 if entity["stale"] and not suppress_stale_padding else 0.)
        boxes = [_box(position + rotation @ np.asarray(b["center_local_m"]), rotation @ _rotation(b["quaternion_wxyz"]),
                     b["half_extents_m"], entity["id"], f'{entity["id"]}:{index}', uncertainty)
                for index, b in enumerate(entity["boxes"])]
        self.compute_counts["entity_box_builds"] += 1
        self._entity_box_cache[entity["id"]] = (key, boxes)
        return boxes

    def _local_bodies(self, view, context, settling_ids=(), probe_ids=()):
        bodies = []
        state = np.asarray(view.state)
        robot_key = (tuple(state[6:8]), context.get("commanded_gripper"))
        cached_robot = self._robot_body_cache.get(robot_key)
        if cached_robot is not None:
            bodies.extend(cached_robot)
        endpoint_parts = {}
        if cached_robot is None and context.get("commanded_gripper") in ("open", "close"):
            endpoint = state.copy()
            endpoint[6:8] = [.04, -.04] if context["commanded_gripper"] == "open" else [0., 0.]
            endpoint_parts = {name: vertices for name, vertices, _ in _local_parts(endpoint, False)}
        for name, vertices, _ in (() if cached_robot is not None else _local_parts(state, False)):
            if name.startswith("finger") and name in endpoint_parts:
                # Sliders move affinely; each separate finger/pad box encloses
                # the measured-to-command interval without filling the palm gap.
                vertices = np.concatenate((vertices, endpoint_parts[name]))
            lower, upper = vertices.min(axis=0), vertices.max(axis=0)
            bodies.append(_box((lower+upper)/2, np.eye(3), (upper-lower)/2, "robot_hand", name))
        if cached_robot is None:
            self._robot_body_cache[robot_key] = tuple(bodies)
            while len(self._robot_body_cache) > 128:
                self._robot_body_cache.popitem(last=False)
        held_id = context.get("held_object_id")
        for entity in self.entities.values():
            pending = entity.get("release_pending_carried_envelope") if entity.get("release_pending") else None
            if pending is not None:
                bodies.append(_box(pending["center"], np.eye(3), pending["half"], entity["id"],
                                   "pending_release_carried_alternative", pending["uncertainty"]))
            if entity["attachment"] != "held":
                continue
            # Retain all explicitly attached payload hypotheses; inconsistent ids
            # lower evidence quality but never silently shrink the carried shape.
            for box in self._entity_boxes(entity, view, entity["id"] in settling_ids):
                bodies.append(_box(view.R.T @ (box["center"]-view.position), view.R.T @ box["rotation"],
                                   box["half"], entity["id"], box["name"], box["uncertainty"]))
        payload_unknown = bool(held_id and (held_id not in self.entities or self.entities[held_id]["attachment"] != "held"))
        payload_unknown |= any(entity.get("release_pending", False) for entity in self.entities.values())
        payload_unknown |= any(entity.get("attachment_probe_unconfirmed", False) and entity["id"] not in probe_ids
                               for entity in self.entities.values())
        return bodies, payload_unknown

    def observe_transition(self, before_view, after_view, step, gripper_command):
        # A validated primitive observes before assessing/entering the next
        # segment; the policy also calls this as a fallback for native skills.
        # Watch/assessment writes self.step too, so it cannot deduplicate the
        # physical transition. Do not grow payload uncertainty or invalidate
        # dependencies twice when both paths report the same control sample.
        step = int(step)
        if self._last_transition_step == step:
            return deepcopy(self._last_transition_result)
        self.step, self._last_view = int(step), after_view
        distance = float(np.linalg.norm(after_view.position-before_view.position))
        angle = math.acos(float(np.clip((np.trace(before_view.R.T @ after_view.R)-1)/2, -1, 1)))
        touched = []
        releases, expected_releases, slips, contacts, expected_contacts, contact_pairs = [], [], [], [], [], []
        gap_change = abs(float(after_view.gap_mm-before_view.gap_mm))
        prior_motion = self.last_motion or {}
        allowed = set(prior_motion.get("allowed_contact_ids", []))
        checked_gripper = bool(prior_motion.get("gripper_settling_reference") or prior_motion.get("gripper_settling_expected_ids"))
        hand = []
        if distance > 1e-6 or gap_change > .2:
            local, _ = self._local_bodies(after_view, {})
            hand = [_box(after_view.position+after_view.R @ b["center"], after_view.R @ b["rotation"],
                         b["half"], b["owner"], b["name"]) for b in local if b["owner"] == "robot_hand"]
        for entity in self.entities.values():
            if entity["attachment"] == "held":
                entity["position_m"] = (after_view.position + after_view.R @ np.asarray(entity["attachment_offset_hand_m"])).tolist()
                entity["quaternion_wxyz"] = _matrix_quaternion(after_view.R @ np.asarray(entity["attachment_rotation_hand"])).tolist()
                entity["updated_step"] = int(step)
                entity["uncertainty_m"] = min(2., entity["uncertainty_m"] + .01*distance + .002*angle)
                if gripper_command == "open":
                    # Opening is not a measured release. Retain the payload
                    # envelope through this local block so the next assessment
                    # cannot abruptly shrink the hand or block settling against
                    # its own just-released object. A fresh executive estimate
                    # explicitly resolves the attached/detached alternatives.
                    new_release = not entity.get("release_pending")
                    if new_release:
                        entity["release_actual_command_step"] = int(step)
                        if not entity.get("release_probe_started"):
                            entity["release_requested_step"] = int(step)
                        entity["release_position_estimate_m"] = list(entity["position_m"])
                        entity["release_quaternion_estimate_wxyz"] = list(entity["quaternion_wxyz"])
                    entity["release_pending"] = True
                    entity["attachment_source"] = "uncertain_after_release_command"
                    if new_release:
                        touched.append(entity["id"])
                        (expected_releases if checked_gripper else releases).append(entity["id"])
                elif gap_change > 6.:
                    touched.append(entity["id"])
                    slips.append(entity["id"])
            elif entity["role"] in ("object", "graspable_part", "obstacle") and hand:
                # Selectively stale only geometry intersecting the measured tool;
                # moving the robot elsewhere preserves unaffected background.
                pairs = [(h["name"], o["name"]) for h in hand for o in self._entity_boxes(entity)
                         if obb_separation(h, o, padding_b=o["uncertainty"]) <= 0]
                if pairs:
                    touched.append(entity["id"])
                    expected = entity["id"] in allowed and all(name.startswith("finger") for name, _ in pairs)
                    contact_pairs.extend({"body_part": name, "entity_id": entity["id"], "entity_part": part,
                                          "expected_finger_contact": expected, "measured_contact": False}
                                         for name, part in pairs)
                    if not (entity["stale"] and entity.get("invalidation_kind") == "possible_contact"):
                        (expected_contacts if expected else contacts).append(entity["id"])
        for ids, kind in ((releases, "release_command"), (contacts, "possible_contact"), (slips, "possible_slip")):
            self._invalidate(ids, step, "release_slip_or_possible_contact", event_kind=kind)
        for ids, kind in ((expected_contacts, "possible_contact"), (expected_releases, "release_command")):
            self._invalidate(ids, step, "release_slip_or_possible_contact", event_kind=kind, local_contact=True)
        self._last_gripper_command = gripper_command
        visual = self.watch_observation(after_view, step)
        result = {"invalidated_ids": sorted(set(touched)), "measured_translation_m": distance,
                  "measured_rotation_rad": angle, "source": "robot_state_not_object_measurement",
                  "visual_watch": visual, "predicted_contact_pairs": contact_pairs}
        self._last_transition_step = step
        self._last_transition_result = deepcopy(result)
        return result

    def import_native(self, scene, held, view, step):
        """Adopt an already available native visual held frame without calibration.

        This does not create native holding, inspect simulator data or manufacture
        payload shape when the model/native scene did not provide an extent.
        """
        if held is None or scene is None:
            return {"imported": False, "reason": "native_held_frame_unavailable"}
        ident = getattr(held, "pid", None)
        entity = self.entities.get(ident)
        frame = getattr(scene, "user_check_body_frames", {}).get(ident, {})
        if entity is None or not frame.get("verified_extent"):
            return {"imported": False, "reason": "matching_entity_and_native_extent_required"}
        offset = np.asarray(getattr(held, "offset_h", []), float)
        sigma = getattr(held, "sigma_offset_m", None)
        if offset.shape != (3,) or not np.isfinite(offset).all() or not isinstance(sigma, Real) or not math.isfinite(float(sigma)):
            return {"imported": False, "reason": "invalid_native_offset"}
        source_step = frame.get("step")
        source_step = int(source_step) if isinstance(source_step, (int, np.integer)) else None
        if (entity.get("attachment_source") == "native_visual_held_offset"
                and entity.get("native_offset_step") == source_step
                and entity.get("attachment_offset_hand_m") == offset.tolist()
                and entity.get("native_offset_uncertainty_m") == float(sigma)):
            return {"imported": False, "reason": "native_offset_unchanged", "id": ident}
        entity.update(attachment="held", attachment_offset_hand_m=offset.tolist(),
                      attachment_rotation_hand=(view.R.T @ _rotation(entity["quaternion_wxyz"])).tolist(),
                      attachment_source="native_visual_held_offset", native_offset_step=source_step,
                      native_offset_uncertainty_m=float(sigma))
        entity["uncertainty_m"] = max(entity["uncertainty_m"], max(0., float(sigma)))
        self._event("native_held_offset", [ident], step)
        return {"imported": True, "id": ident, "shape_source": entity["source"], "task_success": False}

    def begin_attachment_probe(self, target_id, view, step):
        """Carry an existing uncertain extent through a bounded visual probe."""
        entity = self.entities.get(target_id)
        if entity is None or entity["role"] not in ("object", "graspable_part"):
            return {"status": "unknown", "reason": "existing_target_extent_required"}
        usable = not entity["stale"] or bool(entity.get("invalidation_local_contact")
            and entity.get("invalidation_kind") == "possible_contact"
            and not entity.get("nonsettling_invalidation")
            and set(entity.get("invalidation_origins", [])) == {target_id})
        if entity["attachment"] != "held":
            entity["attachment_offset_hand_m"] = (view.R.T @ (np.asarray(entity["position_m"])-view.position)).tolist()
            entity["attachment_rotation_hand"] = (view.R.T @ _rotation(entity["quaternion_wxyz"])).tolist()
        entity["attachment_probe_position_estimate_m"] = list(entity["position_m"])
        entity["attachment_probe_quaternion_estimate_wxyz"] = list(entity["quaternion_wxyz"])
        entity.update(attachment="held", attachment_source="unverified_attachment_probe",
                      attachment_probe_unconfirmed=True, attachment_probe_shape_usable=usable,
                      attachment_probe_step=int(step))
        self._geometry_epoch += 1
        self._route_certificates.clear()
        self.visual_watch.forget([target_id])
        self._event("attachment_probe_started", [target_id], step, physically_verified=False)
        return {"status": "unknown", "target_id": target_id, "attachment_probe_unconfirmed": True,
                "attachment_probe_shape_usable": usable, "observation_age_refreshed": False}

    def _attachment_probe_ids(self, context, step):
        ident = context.get("attachment_probe_target_id")
        entity = self.entities.get(ident)
        start, budget = context.get("attachment_probe_start_step"), context.get("attachment_probe_max_steps")
        if (context.get("attachment_probe_authorized") is not True or entity is None
                or ident != context.get("held_object_id") or entity["attachment"] != "held"
                or not entity.get("attachment_probe_unconfirmed") or not entity.get("attachment_probe_shape_usable")
                or entity.get("release_pending") or context.get("commanded_gripper") != "close"
                or (entity.get("stale") and (not entity.get("invalidation_local_contact")
                    or entity.get("invalidation_kind") != "possible_contact" or entity.get("nonsettling_invalidation")
                    or set(entity.get("invalidation_origins", [])) != {ident}))
                or type(start) is not int or type(budget) is not int or not 1 <= budget <= 120
                or not entity.get("attachment_probe_step", step+1) <= start <= step < start+budget):
            return set()
        return {ident}

    def begin_release_probe(self, target_id, view, step):
        entity = self.entities.get(target_id)
        if entity is None or entity["attachment"] != "held":
            return {"status": "unknown", "reason": "existing_payload_extent_required"}
        usable = not entity["stale"] or bool(entity.get("invalidation_local_contact")
            and entity.get("invalidation_kind") in ("release_command", "possible_contact")
            and not entity.get("nonsettling_invalidation")
            and set(entity.get("invalidation_origins", [])) == {target_id})
        if not entity.get("release_pending"):
            world_position = view.position+view.R @ np.asarray(entity["attachment_offset_hand_m"])
            entity.update(release_position_estimate_m=world_position.tolist(),
                          release_quaternion_estimate_wxyz=_matrix_quaternion(
                              view.R @ np.asarray(entity["attachment_rotation_hand"])).tolist(),
                          release_requested_step=int(step))
        # Register the observation baseline before opening; do not pretend that
        # the physical release command has already happened or poison its entry
        # inventory. observe_transition sets release_pending on actual opening.
        entity.update(release_probe_started=True, release_probe_shape_usable=usable,
                      attachment_probe_shape_usable=usable)
        if self.local_perception is not None:
            entity["local_perception_reference"] = self.local_perception.capture(target_id,
                self._entity_boxes(entity, view, suppress_stale_padding=True), view, step,
                entity["geometry_revision"], occluder_boxes=self._visual_occluders(view, target_id))
        return {"status": "unknown", "target_id": target_id, "release_pending": bool(entity.get("release_pending")),
                "payload_extent_retained": True, "release_verified": False,
                "release_probe_shape_usable": usable}

    def confirm_attachment_evidence(self, evidence, view, step):
        """Consume explicit two-view motion support, not gap or model belief.

        RGB motion does not re-estimate absolute depth, shape, attachment offset,
        placement success or contact safety. Existing extent uncertainty survives.
        """
        if not isinstance(evidence, Mapping):
            return {"accepted": False, "reason": "two_view_motion_evidence_required"}
        ident, kind = evidence.get("target_id"), evidence.get("kind")
        entity = self.entities.get(ident)
        views = evidence.get("views") or {}
        observed, reference = evidence.get("observed_at_step"), evidence.get("reference_step")
        if (entity is None or kind not in ("hold", "release") or evidence.get("status") != "supported"
                or evidence.get("source") != "public_two_view_feature_motion"
                or any((views.get(name) or {}).get("status") != "supported" for name in ("external", "wrist"))
                or not isinstance(observed, int) or not isinstance(reference, int)
                or not reference <= observed == int(step)
                or observed < (entity.get("attachment_probe_step", int(step)+1) if kind == "hold" else
                               entity.get("release_actual_command_step", entity.get("release_requested_step", int(step)+1)))):
            return {"accepted": False, "reason": "current_matching_two_view_motion_support_required"}
        if kind == "hold":
            if entity["attachment"] != "held" or not entity.get("attachment_probe_shape_usable", False):
                return {"accepted": False, "reason": "usable_attachment_probe_required"}
            if entity["stale"]:
                if (entity.get("nonsettling_invalidation") or not entity.get("invalidation_local_contact")
                        or entity.get("invalidation_kind") != "possible_contact"
                        or set(entity.get("invalidation_origins", [])) != {ident}):
                    return {"accepted": False, "reason": "unresolved_shape_or_neighbor_change"}
                # Preserve the exact extra extent previously supplied by stale
                # padding while resolving only the expected-contact pose event.
                entity["uncertainty_m"] = min(2., entity["uncertainty_m"]+.02)
                entity.update(stale=False, stale_reason=None, nonsettling_invalidation=False)
            entity.update(attachment_probe_unconfirmed=False, attachment_source="conditional_two_view_comotion",
                          conditional_attachment_supported=True, release_pending=False)
        else:
            if not entity.get("release_pending"):
                return {"accepted": False, "reason": "release_probe_required"}
            # The release reference is a conservative world-pose hypothesis;
            # feature stationarity provides no replacement metric object pose.
            entity.update(attachment="none", attachment_probe_unconfirmed=False, release_pending=False,
                          position_m=deepcopy(entity.get("release_current_world_estimate_m", entity["release_position_estimate_m"])),
                          quaternion_wxyz=deepcopy(entity.get("release_current_quaternion_estimate_wxyz",
                              entity.get("release_quaternion_estimate_wxyz", entity["quaternion_wxyz"]))),
                          attachment_source="conditional_two_view_world_stationarity",
                          stale=True, stale_reason="released_world_pose_remains_uncertain")
        if kind == "release":
            retained = entity.pop("release_pending_world_envelope", None)
            entity.pop("release_pending_carried_envelope", None)
            if retained is not None:
                entity["local_motion_exclusion_boxes"] = [retained]
        if kind == "release" and self.local_perception is not None:
            # Co-motion/stationarity decides attachment only. Independently
            # triangulated current material can additionally locate the released
            # hypothesis; absent that evidence, the conservative stale pose stays.
            self.observe_local_geometry(view, step, [ident])
        entity["interaction_evidence"] = deepcopy(dict(evidence))
        entity["physically_verified"] = False
        self._geometry_epoch += 1
        self._route_certificates.clear()
        self._event("conditional_visual_"+kind, [ident], step, source=evidence["source"], absolute_pose_reestimated=False)
        return {"accepted": True, "kind": kind, "target_id": ident,
                "conditional_attachment_supported": kind == "hold", "physical_clearance_verified": False,
                "observation_age_refreshed": False, "task_success": None}

    def _settling_ids(self, view, target, q1, context, step):
        """Narrow runtime authority to finish the same checked gripper segment."""
        ref = context.get("gripper_settling_authorization")
        if not isinstance(ref, Mapping) or not context.get("gripper_action"):
            return set()
        try:
            inventory = context.get("scene_inventory") or {}
            if (ref.get("source") != "runtime_validated_stationary_gripper"
                    or ref["target_id"] != context.get("target_id")
                    or ref["held_object_id"] != context.get("held_object_id")
                    or ref["gripper_command"] != context.get("commanded_gripper")
                    or not ref["entry_step"] <= step <= ref["entry_step"]+ref["max_steps"]
                    or sorted(ref["allowed_contact_ids"]) != sorted(context.get("allowed_contact_ids", []))
                    or sorted(ref["support_ids"]) != sorted(inventory.get("support_ids", []))
                    or sorted(ref["nearby_obstacle_ids"]) != sorted(inventory.get("nearby_obstacle_ids", []))
                    or any(self.entities.get(ident, {}).get("geometry_revision") != revision
                           for ident, revision in ref["entity_revisions"].items())):
                return set()
            anchor = np.asarray(_vector(ref["hold_position_m"], 3, "hold_position_m"))
            anchor_q = _quaternion(ref["hold_quaternion_wxyz"])
            pose_q = _quaternion(measured_pose(view.state)[3:7])
            position_tolerance = min(.01, max(1e-6, float(ref.get("position_tolerance_m", .004))))
            orientation_tolerance = min(.10, max(.001, float(ref.get("orientation_tolerance_rad", .08))))
            if (np.linalg.norm(target-anchor) > 1e-6 or np.linalg.norm(view.position-anchor) > position_tolerance
                    or abs(np.dot(q1, anchor_q)) < math.cos(.001/2)
                    or abs(np.dot(pose_q, anchor_q)) < math.cos(orientation_tolerance/2)):
                return set()
        except (KeyError, TypeError, ValueError):
            return set()
        expected = {ident for ident in (ref["target_id"], ref["held_object_id"]) if ident in self.entities}
        relaxed = set()
        for ident in expected:
            entity = self.entities[ident]
            monitor = entity.get("visual_monitoring") or {}
            if entity.get("monitoring_unknown") and not monitor.get("self_occluded"):
                continue
            if entity["stale"] and (entity.get("nonsettling_invalidation")
                    or not entity.get("invalidation_origins")
                    or not set(entity.get("invalidation_origins", [])) <= expected
                    or entity.get("invalidated_step", -1) < ref["entry_step"]
                    or entity.get("invalidation_kind") not in (("release_command", "possible_contact")
                        if ref["gripper_command"] == "open" else ("possible_contact",))
                    or (ref["gripper_command"] == "open" and entity.get("invalidation_kind") == "possible_contact"
                        and not entity.get("invalidation_local_contact"))):
                continue
            relaxed.add(ident)
        return relaxed

    def _interaction_inventory(self, context, target_entity, final_interaction, settling_ids=(), *, stationary_release=False, departure_probe=False):
        """Require explicit model inventory before final grasp/place interaction.

        The inventory is an executive audit, never an independent completeness
        certificate. Support must be an actual declared solid near the target;
        a free-volume hypothesis cannot substitute for a table/rim/wall model.
        """
        if not final_interaction:
            return [], set()
        inventory = context.get("scene_inventory") or {}
        missing, relevant = [], set()
        if inventory.get("reviewed") is not True:
            missing.append("scene_inventory.reviewed")
        if inventory.get("unresolved") is not False:
            missing.append("scene_inventory.unresolved=false")
        target_id = context.get("target_id")
        if target_id in self.entities:
            relevant.add(target_id)
        if (not target_entity or target_entity.get("role") == "free_region"
                or (target_id not in settling_ids and
                    (target_entity.get("stale") or target_entity.get("monitoring_unknown")))):
            missing.append("current_target_envelope:" + str(target_id or "unspecified"))
        support_ids = inventory.get("support_ids", [])
        obstacle_ids = inventory.get("nearby_obstacle_ids", [])
        valid_supports = []
        for label, ids in (("support_ids", support_ids), ("nearby_obstacle_ids", obstacle_ids)):
            for ident in ids:
                entity = self.entities.get(ident)
                if entity is not None:
                    relevant.add(ident)
                roles = {"support_surface"} if label == "support_ids" else {"obstacle", "object", "graspable_part", "support_surface"}
                expected_target = label == "nearby_obstacle_ids" and ident in settling_ids
                if (entity is None or entity["role"] not in roles or (not expected_target and
                        (entity["stale"] or entity.get("monitoring_unknown")))):
                    missing.append("current_" + label + ":" + str(ident))
                    continue
                if label == "support_ids":
                    valid_supports.append(entity)
        if target_entity:
            target_boxes = self._entity_boxes(target_entity)
            valid_supports = [entity for entity in valid_supports if any(
                obb_separation(support, target, padding_b=target["uncertainty"]) <= .06
                for support in self._entity_boxes(entity) for target in target_boxes)]
        # Opening a stationary hand releases its payload; being supported is
        # not a precondition for opening. This does not authorize descent or
        # claim placement. Declared neighbors/supports and the exact sweep still
        # require valid evidence, and closing/approach keep the support guard.
        if not valid_supports and not stationary_release and not departure_probe:
            missing.append("nearby_support_surface_for:" + str(target_id or "unspecified"))
        return sorted(set(missing)), relevant

    def preview_scene_update(self, update, view, step):
        """Stage model hypotheses on an isolated memory, without fresh RGB claims."""
        preview = copy(self)
        preview.entities = deepcopy(self.entities)
        preview.events = deepcopy(self.events)
        preview.last_motion = deepcopy(self.last_motion)
        preview._last_transition_result = deepcopy(self._last_transition_result)
        preview._watched_ids = set(self._watched_ids)
        preview._entity_box_cache = self._entity_box_cache.copy()
        preview._robot_body_cache = self._robot_body_cache.copy()
        preview._sweep_cache = self._sweep_cache.copy()
        preview._route_certificates = self._route_certificates.copy()
        preview.compute_counts = self.compute_counts.copy()
        preview.visual_watch = _PreviewVisualWatch(preview.entities)
        preview.local_perception = None  # Pure previews never consume/rebaseline public image tracks.
        if update is not None:
            preview.update_from_model(update, view, step)
        return preview

    def preview_motion(self, view, target_xyz, target_wxyz=None, context=None, step=None):
        """Evaluate a candidate without changing live memory, watches or events."""
        step = self.step if step is None else int(step)
        preview = self.preview_scene_update(None, view, step)
        result = preview.assess_motion(view, target_xyz, target_wxyz, context, step)
        result["preview_only"] = True
        return result

    def verify_command(self, view, normalized_action, context=None, step=None, *, allow_unknown=False):
        """Check the exact coupled OSC increment after runtime clipping.

        A short explicitly authorized unknown continuation can remain unknown;
        the method never relabels it supported. No live evidence is mutated.
        """
        action = np.asarray(_vector(normalized_action, 7, "normalized_action"))
        if np.any(np.abs(action) > 1.+1e-9):
            raise ValueError("normalized action exceeds controller bounds")
        delta = action[:3]*TRANSLATION_SCALE_M
        rotation = _quaternion_product(_rotvec_quaternion(action[3:6]*ROTATION_SCALE_RAD),
                                       _rotvec_quaternion(np.asarray(view.state[3:6])))
        target_q = rotation[[3, 0, 1, 2]]
        ctx = {**(context or {}), "_verify_exact_command": True}
        if abs(action[6]) > 1e-9:
            ctx["commanded_gripper"] = "close" if action[6] > 0 else "open"
        step = self.step if step is None else int(step)
        preview = self.preview_scene_update(None, view, step)
        result = preview.assess_motion(view, view.position+delta, target_q, ctx, step)
        result["preview_only"] = True
        supported = result["status"] == "supported"
        observation_hold = (result["reason"] == "stationary_observation_no_clearance_claim"
                            and not ctx.get("gripper_action") and np.linalg.norm(delta) <= 1e-9
                            and np.linalg.norm(action[3:6]*ROTATION_SCALE_RAD) <= 1e-8)
        # Contact/release can invalidate the interacted object and its free
        # hypothesis during an already checked stationary gripper segment.
        # Its runtime token permits only that segment's settling, never a new
        # pose or a newly changed neighbor/free-space hypothesis.
        settling = False
        reference = ctx.get("gripper_settling_authorization")
        expected = set(result.get("gripper_settling_expected_ids", []))
        if (isinstance(reference, Mapping) and expected and result["status"] == "unknown"
                and not result["inventory_missing_evidence_requests"]
                and not result["coverage_witnesses"]
                and result["allowed_fraction"] >= 1.-1e-9
                and np.linalg.norm(delta) <= min(.01, float(reference.get("position_tolerance_m", .004)))
                and np.linalg.norm(action[3:6]*ROTATION_SCALE_RAD) <= min(.1, float(reference.get("orientation_tolerance_rad", .08)))):
            settling = True
            kinds = ("release_command", "possible_contact") if ctx.get("commanded_gripper") == "open" else ("possible_contact",)
            for ident in reference.get("entity_revisions", {}):
                entity = preview.entities.get(ident)
                if entity is None:
                    settling = False
                    break
                if entity.get("stale") and (entity.get("nonsettling_invalidation")
                        or entity.get("invalidation_kind") not in kinds
                        or (ctx.get("commanded_gripper") == "open" and entity.get("invalidation_kind") == "possible_contact"
                            and not entity.get("invalidation_local_contact"))
                        or not entity.get("invalidation_origins")
                        or not set(entity["invalidation_origins"]) <= expected
                        or entity.get("invalidated_step", -1) < reference.get("entry_step", 0)):
                    settling = False
                    break
                if entity.get("monitoring_unknown") and not (entity.get("visual_monitoring") or {}).get("self_occluded"):
                    settling = False
                    break
        result.update(command_supported=supported,
                      command_settling_authorized=settling,
                      command_observation_hold_allowed=observation_hold,
                      command_allowed=bool(supported or settling or observation_hold or (allow_unknown and result["status"] == "unknown"
                           and result["allowed_fraction"] >= 1.-1e-9)),
                      exact_command_xyz_m=(view.position+delta).tolist(),
                      exact_command_quaternion_wxyz=target_q.tolist(),
                      command_scope="exact_normalized_translation_rotation_and_gripper")
        return result

    def _sweep_geometry(self, view, target, q0, q1, bodies, obstacles, free, allowed, radius):
        """Bounded adaptive numerical enclosure of the declared rigid sweep.

        Refinement reduces only interpolation overbounds. Model extent and free
        region uncertainty are retained. A collision or a genuinely uncovered
        model boundary cannot be cleared by merely reducing those uncertainties.
        """
        distance = float(np.linalg.norm(target-view.position))
        angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0, 1)))
        cache_key = (tuple(view.position), tuple(target), tuple(q0), tuple(q1),
                     tuple(_box_key(b) for b in bodies), tuple(_box_key(b) for b in obstacles),
                     tuple(_box_key(b) for b in free), tuple(sorted(allowed)), radius)
        if cache_key in self._sweep_cache:
            self.compute_counts["exact_sweep_cache_hits"] += 1
            self._sweep_cache.move_to_end(cache_key)
            return {**self._sweep_cache[cache_key], "exact_cache_hit": True}
        self.compute_counts["sweep_evaluations"] += 1
        intervals = max(1, min(96, math.ceil((distance+radius*angle)/.006)))
        # The centre line plus a bounding sphere encloses every translated and
        # rotated tool/payload OBB. The extra radius retains all <5cm relevant
        # obstacle evidence plus numerical interval padding.
        broad_margin = radius+.05+distance/(2*intervals)+2*radius*math.sin(angle/(4*intervals))
        lower = np.minimum(view.position, target)-broad_margin
        upper = np.maximum(view.position, target)+broad_margin
        original_obstacle_count = len(obstacles)
        obstacles = [box for box in obstacles if np.all(box["center"]+np.abs(box["rotation"]) @
                     (box["half"]+box["uncertainty"]) >= lower) and
                     np.all(box["center"]-np.abs(box["rotation"]) @ (box["half"]+box["uncertainty"]) <= upper)]
        self.compute_counts["obstacles_pruned"] += original_obstacle_count-len(obstacles)
        centers = np.asarray([b["center"] for b in obstacles])
        rotations = np.asarray([b["rotation"] for b in obstacles])
        halves = np.asarray([b["half"]+b["uncertainty"] for b in obstacles])
        contact_mask = np.asarray([b["owner"] in allowed for b in obstacles], bool)
        free_centers = np.asarray([b["center"] for b in free])
        free_rotations = np.asarray([b["rotation"] for b in free])
        free_halves = np.asarray([b["half"]-b["uncertainty"] for b in free])
        attempts = []
        for refinement in range(MAX_SWEEP_REFINEMENTS+1):
            padding = distance/(2*intervals)+2*radius*math.sin(angle/(4*intervals))
            rows, witnesses, unexpected, start_overlaps = [], [], [], {}
            evidence, supporting_regions = set(), set()
            min_gap, coverage_reserve, limiting_pair, refinable, union_cells = math.inf, math.inf, None, False, 0
            for index in range(intervals+1):
                fraction = index/intervals
                rotation = _rotation(q0) if angle < 1e-12 else _rotation(_slerp(q0, q1, fraction))
                position = view.position+fraction*(target-view.position)
                row_gap, overlaps, covered = math.inf, {}, True
                for body in bodies:
                    world = _box(position+rotation @ body["center"], rotation @ body["rotation"],
                                 body["half"], body["owner"], body["name"])
                    margin = padding+body["uncertainty"]
                    region, margins = None, np.empty(0)
                    if free:
                        relative = np.einsum("ni,nij->nj", world["center"]-free_centers, free_rotations)
                        extent = np.abs(np.einsum("nji,jk->nik", free_rotations, world["rotation"])) @ (world["half"]+margin)
                        margins = np.min(free_halves-np.abs(relative)-extent, axis=1)
                        best = int(np.argmax(margins))
                        if margins[best] >= 0:
                            region = free[best]
                            coverage_reserve = min(coverage_reserve, float(margins[best]))
                    if region is None:
                        union, owners, cells = _union_coverage(world, free, margin) if len(free) > 1 else (False, set(), 0)
                        union_cells += cells
                        if union:
                            supporting_regions.update(owners)
                            coverage_reserve = 0.
                        else:
                            covered = False
                            witnesses.append(_coverage_witness(world, free, body["uncertainty"], padding, fraction))
                            refinable |= padding > 0 and any(_contained(world, region, body["uncertainty"]) for region in free)
                    else:
                        supporting_regions.add(region["owner"])
                    gaps = _separations(world, centers, rotations, halves, margin)
                    # Held and world-stationary release alternatives describe
                    # the same payload; they cannot collide with one another.
                    # The real hand and every other payload still check both.
                    for oi, obstacle in enumerate(obstacles):
                        if obstacle.get("alternative_for") == body["owner"]:
                            gaps[oi] = np.inf
                    finger = body["owner"] == "robot_hand" and body["name"].startswith("finger")
                    if finger:
                        gaps[contact_mask] = np.inf
                    if len(gaps):
                        row_gap = min(row_gap, float(gaps.min()))
                        nearest = int(np.argmin(gaps))
                        if index == 0 and math.isfinite(float(gaps[nearest])) and (
                                limiting_pair is None or gaps[nearest] < limiting_pair["clearance_estimate_m"]):
                            obstacle = obstacles[nearest]
                            limiting_pair = {"body_id": body["owner"], "body_part": body["name"],
                                "obstacle_id": obstacle["owner"], "obstacle_part": obstacle["name"],
                                "clearance_estimate_m": float(gaps[nearest]),
                                "combined_extent_uncertainty_m": body["uncertainty"]+obstacle["uncertainty"],
                                "interval_padding_m": padding, "uncertainty_already_applied": True}
                        if padding > 0 and np.any(gaps <= 0):
                            without_interpolation = _separations(world, centers, rotations, halves, body["uncertainty"])
                            refinable |= bool(np.any((gaps <= 0) & (without_interpolation > 0)))
                    for oi in np.flatnonzero(gaps < .05):
                        obstacle, gap = obstacles[int(oi)], float(gaps[int(oi)])
                        evidence.add(obstacle["owner"])
                        if gap <= 0:
                            overlaps[(body["name"], obstacle["name"])] = gap
                min_gap = min(min_gap, row_gap)
                if index == 0:
                    start_overlaps = overlaps
                new_overlap = set(overlaps)-set(start_overlaps)
                worsening = any(gap < start_overlaps[key]-1e-6 for key, gap in overlaps.items() if key in start_overlaps)
                if index and (new_overlap or worsening or (overlaps and not start_overlaps)):
                    unexpected.append(index)
                rows.append((fraction, row_gap, covered, overlaps))
            unknown_count = sum(not row[2] for row in rows)
            attempts.append({"samples": len(rows), "interval_padding_m": padding,
                             "uncovered_samples": unknown_count, "overlap_samples": sum(bool(row[3]) for row in rows),
                             "union_subdivision_cells": union_cells})
            if not refinable or refinement == MAX_SWEEP_REFINEMENTS or intervals >= MAX_SWEEP_INTERVALS:
                break
            intervals = min(MAX_SWEEP_INTERVALS, intervals*2)
        escape = bool(start_overlaps and not unexpected and all(
            rows[-1][3].get(key, 0.) > initial+1e-5 for key, initial in start_overlaps.items()))
        first_uncovered = next((row[0] for row in rows if not row[2]), None)
        first_unsafe = next((index for index, row in enumerate(rows) if not row[2] or row[3]), intervals+1)
        prefix = 1. if first_unsafe > intervals else max(0., (first_unsafe-1)/intervals)
        result = {"rows": rows, "intervals": intervals, "padding": padding, "min_gap": min_gap,
                "coverage_reserve_m": coverage_reserve, "exact_cache_hit": False,
                "obstacle_boxes_input": original_obstacle_count, "obstacle_boxes_checked": len(obstacles),
                "limiting_pair": limiting_pair, "evidence": evidence, "supporting_regions": supporting_regions,
                "start_overlaps": start_overlaps, "unexpected": unexpected, "escape": escape,
                "unknown_count": unknown_count, "first_uncovered_fraction": first_uncovered,
                "supported_prefix_fraction": prefix, "coverage_witnesses": _bounded_witnesses(witnesses),
                "refinement": {"attempts": attempts, "refinements": len(attempts)-1,
                               "interval_limit": MAX_SWEEP_INTERVALS, "model_uncertainty_reduced": False,
                               "still_interpolation_limited": bool(refinable)}}
        self._sweep_cache[cache_key] = result
        while len(self._sweep_cache) > 64:
            self._sweep_cache.popitem(last=False)
        return result

    def _goal_sweep(self, view, target, q0, q1, bodies, obstacles, free, allowed, radius, step):
        """Reuse only an already supported constant-orientation route tube.

        The new centre-line suffix and every current body vertex must remain
        within the old route's unused face/separation margins. Rotation routes,
        changed evidence, unknown coverage and exhausted margins are recomputed.
        Exact emitted commands are independently checked by verify_command.
        """
        key = (tuple(target), tuple(q1), tuple(sorted(allowed)), self._geometry_epoch,
               tuple(_box_key(b) for b in obstacles), tuple(_box_key(b) for b in free))
        certificate = self._route_certificates.get(key)
        if certificate is not None and len(bodies) == len(certificate["bodies"]):
            start, direction = certificate["start"], target-certificate["start"]
            length2 = float(np.dot(direction, direction))
            progress = float(np.dot(view.position-start, direction)/length2) if length2 > 1e-12 else -1.
            if 0 <= progress <= 1:
                shape_error = 0.
                for current, original in zip(bodies, certificate["bodies"]):
                    if (current["owner"], current["name"]) != (original["owner"], original["name"]):
                        shape_error = math.inf
                        break
                    angle = math.acos(float(np.clip((np.trace(current["rotation"].T @ original["rotation"])-1)/2, -1, 1)))
                    shape_error = max(shape_error, float(np.linalg.norm(current["center"]-original["center"]))
                        +float(np.linalg.norm(current["half"]-original["half"]))
                        +math.sqrt(3)*abs(current["uncertainty"]-original["uncertainty"])
                        +2*float(np.linalg.norm(original["half"]+original["uncertainty"]))*math.sin(angle/2))
                angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0, 1)))
                displacement = float(np.linalg.norm(view.position-(start+progress*direction)))
                displacement += 2*max(radius, certificate["radius"])*math.sin(angle/2)+shape_error
                reserve = certificate["reserve"]-displacement
                if reserve > 1e-8:
                    self.compute_counts["supported_route_reuses"] += 1
                    source = certificate["sweep"]
                    gap = source["min_gap"]-displacement
                    return {**source, "rows": [(0., gap, True, {}), (1., gap, True, {})],
                            "intervals": 1, "padding": 0., "min_gap": gap,
                            "coverage_reserve_m": source["coverage_reserve_m"]-displacement,
                            "supported_route_reuse": {"basis_step": certificate["step"],
                                "vertex_displacement_bound_m": displacement, "remaining_margin_m": reserve,
                                "new_sample_evaluations": 0, "observation_age_refreshed": False},
                            "refinement": {"attempts": [], "refinements": 0, "interval_limit": MAX_SWEEP_INTERVALS,
                                           "model_uncertainty_reduced": False, "supported_route_reused": True}}
        result = self._sweep_geometry(view, target, q0, q1, bodies, obstacles, free, allowed, radius)
        reserve = min(result["coverage_reserve_m"], result["min_gap"])
        if (result["unknown_count"] == 0 and not any(row[3] for row in result["rows"])
                and reserve > 1e-8 and abs(np.dot(q0, q1)) >= 1.-1e-12
                and np.linalg.norm(target-view.position) > .01):
            self._route_certificates[key] = {"start": view.position.copy(), "bodies": deepcopy(bodies),
                                             "radius": radius, "reserve": reserve, "sweep": result, "step": int(step)}
            while len(self._route_certificates) > 8:
                self._route_certificates.popitem(last=False)
        return result

    def _command_candidates(self, view, target, q0, q1, bodies, obstacles, free, allowed, context, radius):
        """Check bounded OSC error-increment candidates, not velocity promises.

        Each independently clipped translation/rotation command gets its own
        short swept-volume check. Shape uncertainty is applied once in those
        checks. Empirical response predicts progress; it never turns an unknown
        volume into free space or inversely amplifies a weak/contact response.
        """
        distance = float(np.linalg.norm(target-view.position))
        angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0, 1)))
        tracking = context.get("tracking") or {}
        translation_cap = min(.02, max(0., float(context.get("controller_translation_limit_m", .02))))
        rotation_cap = min(.1, max(0., float(context.get("controller_rotation_limit_rad", .1))))
        if tracking.get("status") != "normal":
            translation_cap, rotation_cap = min(.01, translation_cap), min(.06, rotation_cap)

        def response(label):
            value = tracking.get(label+"_response_estimate")
            return float(value) if isinstance(value, Real) and math.isfinite(float(value)) and value > 0 else None

        translation_response, rotation_response = response("translation"), response("rotation")
        # An empirical upper response above one suggests possible overshoot;
        # reduce a candidate at the goal. Responses below one never enlarge it.
        def upper_response(label):
            bounds = tracking.get(label+"_response_bounds")
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                value = bounds[1]
                if isinstance(value, Real) and math.isfinite(float(value)):
                    return max(1., float(value))
            return 1.

        candidates, seen = [], set()
        for scale in (1., .5, .25, .1):
            command_m = min(translation_cap*scale, distance/upper_response("translation"))
            command_rad = min(rotation_cap*scale, angle/upper_response("rotation"))
            key = (round(command_m, 12), round(command_rad, 12))
            if key in seen:
                continue
            seen.add(key)
            endpoint = view.position + (target-view.position)*(command_m/max(distance, 1e-12))
            endpoint_q = _slerp(q0, q1, command_rad/max(angle, 1e-12))
            sweep = self._sweep_geometry(view, endpoint, q0, endpoint_q, bodies, obstacles, free, allowed, radius)
            intervals, padding = sweep["intervals"], sweep["padding"]
            clearance = sweep["min_gap"]
            covered = sweep["unknown_count"] == 0
            collision = any(row[3] for row in sweep["rows"])
            predicted_m = None if translation_response is None else command_m*translation_response
            predicted_rad = None if rotation_response is None else command_rad*rotation_response
            candidates.append({"translation_step_m": command_m if distance > 1e-12 else translation_cap*scale,
                               "rotation_step_rad": command_rad if angle > 1e-12 else rotation_cap*scale,
                               "translation_limit_m": translation_cap*scale,
                               "rotation_limit_rad": rotation_cap*scale,
                               "command_translation_m": command_m, "command_rotation_rad": command_rad,
                               "xyz_m": endpoint.tolist(), "quaternion_wxyz": endpoint_q.tolist(),
                               "translation_delta_m": (endpoint-view.position).tolist(),
                               "controller_translation_limit_m": translation_cap,
                               "controller_rotation_limit_rad": rotation_cap,
                               "feasible": bool(covered and not collision),
                               "reason": "estimated_overlap" if collision else "model_uncovered" if not covered else "predicted_sweep_supported",
                               "clearance_estimate_m": None if not math.isfinite(clearance) else clearance,
                               "interval_padding_m": padding, "sweep_samples": intervals+1,
                               "expected_translation_progress_m": predicted_m,
                               "expected_rotation_progress_rad": predicted_rad,
                               "progress_score_m": (command_m if predicted_m is None else predicted_m)
                                                   +radius*(command_rad if predicted_rad is None else predicted_rad)})
            # Scales descend monotonically and empirical responses are positive;
            # the first feasible candidate has the largest progress score.
            if covered and not collision:
                break
        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        return (max(feasible, key=lambda candidate: candidate["progress_score_m"]) if feasible else None), candidates

    def assess_motion(self, view, target_xyz, target_wxyz, context=None, step=None):
        """Check nominal hand/payload sweep; return evidence and execution advice.

        Sampling includes interval displacement inflation. It bounds the declared
        rigid prediction, not actual OSC arm motion, unmodelled obstacles or RGB
        error. Unknown permits a short explicitly uncertain model-directed move.
        A supported whole goal or conservative contiguous supported prefix has
        no geometry observation timer: suggested_steps is None and the caller
        retains its original goal, fixed prefix and execution watchdog budget.
        """
        context = context or {}
        step = self.step if step is None else int(step)
        target = np.asarray(_vector(target_xyz, 3, "target_xyz"))
        pose = measured_pose(view.state)
        q0, q1 = _quaternion(pose[3:7]), _quaternion(pose[3:7] if target_wxyz is None else target_wxyz)
        distance = float(np.linalg.norm(target-view.position))
        angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0, 1)))
        settling_ids = self._settling_ids(view, target, q1, context, step)
        probe_ids = self._attachment_probe_ids(context, step)
        bodies, payload_unknown = self._local_bodies(view, context, settling_ids, probe_ids)
        radius = max(float(np.linalg.norm(b["center"])+np.linalg.norm(b["half"]+b["uncertainty"])) for b in bodies)
        obstacles, free = [], []
        for entity in self.entities.values():
            pending = entity.get("release_pending_world_envelope") if entity.get("release_pending") else None
            if pending is not None:
                box = _box(pending["center"], np.eye(3), pending["half"], entity["id"],
                           "pending_release_world_alternative", pending["uncertainty"])
                obstacles.append({**box, "alternative_for": entity["id"]})
            if entity["attachment"] == "held":
                if entity["id"] in probe_ids and entity.get("attachment_probe_position_estimate_m"):
                    stationary = {**entity, "attachment": "none",
                        "position_m": entity["attachment_probe_position_estimate_m"],
                        "quaternion_wxyz": entity["attachment_probe_quaternion_estimate_wxyz"]}
                    obstacles.extend({**box, "alternative_for": entity["id"]} for box in self._entity_boxes(stationary))
                if entity.get("release_pending") and entity.get("release_position_estimate_m"):
                    alternative = {**entity, "attachment": "none", "position_m": entity["release_position_estimate_m"],
                                   "quaternion_wxyz": entity.get("release_quaternion_estimate_wxyz", entity["quaternion_wxyz"])}
                    obstacles.extend({**box, "alternative_for": entity["id"]} for box in
                                     self._entity_boxes(alternative, suppress_stale_padding=entity["id"] in settling_ids))
                continue
            for index, excluded in enumerate(entity.get("local_motion_exclusion_boxes", [])):
                obstacles.append(_box(excluded["center"], np.eye(3), excluded["half"], entity["id"],
                                      f"local_motion_exclusion_{index}", excluded["uncertainty"]))
            if entity["role"] == "destination_region":
                continue
            boxes = self._entity_boxes(entity, suppress_stale_padding=entity["id"] in settling_ids)
            if entity.get("release_pending"):
                boxes = [{**box, "alternative_for": entity["id"]} for box in boxes]
            if entity["role"] == "free_region":
                if not entity["stale"]:
                    free.extend(boxes)
            else:
                obstacles.extend(boxes)
        target_id = context.get("target_id")
        target_entity = self.entities.get(target_id, {})
        related = set(target_entity.get("related_ids", [])) | {target_id}
        if target_entity.get("parent_id"):
            related.add(target_entity["parent_id"])
        allowed = set(context.get("allowed_contact_ids", [])) & related if context.get("intent") in _CONTACT_INTENTS else set()
        stationary_release = bool(context.get("gripper_action") and context.get("commanded_gripper") == "open"
                                  and ((distance <= 1e-6 and angle <= .001) or settling_ids))
        if stationary_release:
            # The token only authorizes the same checked release segment. Its
            # stationary payload alternative may contact the opening fingers;
            # palms and all neighboring objects remain collision checked.
            bound_id = context.get("held_object_id")
            if (bound_id in self.entities and self.entities[bound_id]["role"] in ("object", "graspable_part")
                    and (self.entities[bound_id]["attachment"] == "held"
                         or bound_id in context.get("allowed_contact_ids", []))):
                allowed.add(bound_id)
            allowed.update(settling_ids & {bound_id})
        allowed.update(probe_ids & set(context.get("allowed_contact_ids", [])))
        # Select one current supporting hypothesis per body sample; an older
        # redundant region does not independently invalidate a covered route.
        free.sort(key=lambda region: (-self.entities[region["owner"]]["observation_step"],
                                      region["uncertainty"], region["name"]))
        payload_ids = {body["owner"] for body in bodies if body["owner"] != "robot_hand"}
        sweep = (self._sweep_geometry(view, target, q0, q1, bodies, obstacles, free, allowed, radius)
                 if context.get("_verify_exact_command") else
                 self._goal_sweep(view, target, q0, q1, bodies, obstacles, free, allowed, radius, step))
        rows, intervals, padding = sweep["rows"], sweep["intervals"], sweep["padding"]
        evidence = set(payload_ids) | (allowed & self.entities.keys()) | sweep["evidence"]
        supporting_regions = sweep["supporting_regions"]
        min_gap, limiting_pair = sweep["min_gap"], sweep["limiting_pair"]
        unexpected, start_overlaps, escape = sweep["unexpected"], sweep["start_overlaps"], sweep["escape"]
        unknown_count = len(rows) if payload_unknown else sweep["unknown_count"]
        hold_only = (distance < 1e-9 and angle < 1e-9 and context.get("intent") == "observe"
                     and not context.get("gripper_action", False))
        blocked = bool(not hold_only and (unexpected or (start_overlaps and not escape)))
        first_block = unexpected[0] if unexpected else (0 if blocked else intervals+1)
        predicted_prefix = max(0., (first_block-1)/intervals) if blocked else 1.
        unknown_fraction = unknown_count / len(rows)
        evidence.update(supporting_regions)
        # Final interaction needs the target/support/neighbor audit in addition
        # to free-volume coverage. It cannot creep through missing inventory via
        # repeated short unknown continuations.
        gripper_closing = bool(context.get("gripper_action") and context.get("commanded_gripper") == "close")
        interaction_goal = np.asarray(_vector(context.get("original_goal_xyz_m", target), 3, "original_goal_xyz_m"))
        interaction_distance = float(np.linalg.norm(interaction_goal-view.position))
        final_interaction = (context.get("intent") in ("approach", "grasp", "place") and
                             (interaction_distance <= .05 or context.get("gripper_action", False))) or gripper_closing or stationary_release
        missing_evidence, inventory_evidence = self._interaction_inventory(context, target_entity, final_interaction, settling_ids | probe_ids,
                                                        stationary_release=stationary_release, departure_probe=bool(probe_ids))
        evidence.update(inventory_evidence)
        # New segments can depend on previously unused cached entities. Watch
        # those IDs at this same observation before accepting their support;
        # per-ID dedup keeps the policy's earlier watch call inexpensive.
        epoch_before_watch = self._geometry_epoch
        self.watch_observation(view, step, evidence)
        if self._geometry_epoch != epoch_before_watch:
            return self.assess_motion(view, target_xyz, target_wxyz, context, step)
        tracking = context.get("tracking") or {}
        tracking_status = tracking.get("status", "unobserved")
        motion_revoked = tracking_status in ("mismatch", "lost") and not hold_only
        if motion_revoked and self._last_motion_revoke_step != step:
            self._event("motion_support_revoked", evidence, step, scope="current_motion_only",
                        reason="measured_tracking_mismatch", scene_entities_invalidated=False)
            self._last_motion_revoke_step = step
        # RGB invalidation reaches the inventory guard in this observation;
        # tracking separately revokes the current motion, never scene poses.
        settling_ids = self._settling_ids(view, target, q1, context, step)
        missing_evidence, inventory_evidence = self._interaction_inventory(context, target_entity, final_interaction, settling_ids | probe_ids,
                                                        stationary_release=stationary_release, departure_probe=bool(probe_ids))
        evidence.update(inventory_evidence)
        stale_ids = [ident for ident in evidence if self.entities[ident]["stale"] and ident not in probe_ids]
        monitoring_unknown = [ident for ident in evidence if self.entities[ident].get("monitoring_unknown")]
        fresh_age = max((max(0, step-self.entities[ident]["observation_step"]) for ident in evidence), default=0)
        uncertain = bool(unknown_fraction or stale_ids or monitoring_unknown or payload_unknown or escape or hold_only
                         or tracking_status in ("mismatch", "lost"))
        if (missing_evidence and not hold_only) or motion_revoked:
            blocked, predicted_prefix = True, 0.
        status = "blocked" if blocked else "unknown" if uncertain else "supported"
        whole_status = status
        # This is the longest prefix supported by this bounded subdivision,
        # not a claim about the unobserved remainder of the original goal.
        supported_prefix = sweep["supported_prefix_fraction"]
        if missing_evidence or motion_revoked or stale_ids or monitoring_unknown or payload_unknown or escape or hold_only:
            supported_prefix = 0.
        prefix_supported = supported_prefix > 1e-9 and (distance+radius*angle > 1e-9 or status == "supported")
        execution_status = "supported" if prefix_supported else status
        candidate_target = view.position+supported_prefix*(target-view.position) if prefix_supported else target
        candidate_q = _slerp(q0, q1, supported_prefix) if prefix_supported else q1
        contact = context.get("intent") in _CONTACT_INTENTS
        # Pose-error commands and measured progress are different quantities.
        # A finite candidate sweep, rather than .25*gap or a second uncertainty
        # penalty, determines which existing controller commands are supported.
        local_gap = rows[0][1]
        translation_step = min(.005, max(0., float(context.get("controller_translation_limit_m", .02))))
        rotation_step = min(.04, max(0., float(context.get("controller_rotation_limit_rad", .1))))
        translation_limit, rotation_limit = translation_step, rotation_step
        selected_candidate, candidates, candidate_failure = None, [], False
        if execution_status == "supported" and not context.get("_verify_exact_command", False):
            selected_candidate, candidates = self._command_candidates(
                view, candidate_target, q0, candidate_q, bodies, obstacles, free, allowed, context, radius)
            if selected_candidate is None:
                status, execution_status, blocked, predicted_prefix, candidate_failure = "blocked", "blocked", True, 0., True
                supported_prefix, prefix_supported = 0., False
            else:
                translation_step = selected_candidate["translation_step_m"]
                rotation_step = selected_candidate["rotation_step_rad"]
                translation_limit = selected_candidate["translation_limit_m"]
                rotation_limit = selected_candidate["rotation_limit_rad"]
        if execution_status != "supported":
            horizon_m, horizon_rad = (.015, .10) if contact else (.02, .12)
            translation_step, rotation_step = min(.005, translation_step), min(.04, rotation_step)
            suggested = 4 if contact else 6
            fraction = min(1., horizon_m/max(distance, 1e-12), horizon_rad/max(angle, 1e-12), predicted_prefix)
        else:
            suggested, fraction = None, supported_prefix

        reason = ("stationary_observation_no_clearance_claim" if hold_only else
                  "measured_tracking_mismatch" if motion_revoked else
                  "geometry_inventory_required" if missing_evidence else
                  "local_command_sweep_not_supported" if candidate_failure else
                  "estimated_obstacle_overlap" if blocked else "escape_estimated_initial_overlap" if escape else
                  "payload_geometry_unknown" if payload_unknown else "geometry_stale" if stale_ids else
                  "visual_monitoring_unavailable" if monitoring_unknown else
                  "unobserved_or_model_uncovered_sweep" if unknown_fraction else "bounded_model_geometry_support")
        coverage_requests = sorted({"sweep_coverage:"+witness["body_part"]+":"+
                                    str(witness["region_id"] or "free_region_missing")+":"+
                                    str(witness["axis"] or "all")+":"+str(witness["face"] or "unspecified")
                                    for witness in sweep["coverage_witnesses"]})
        all_missing = sorted(set(missing_evidence+coverage_requests))
        prefix_xyz = view.position+fraction*(target-view.position)
        prefix_q = _slerp(q0, q1, fraction)
        execution_prefix = {"status": execution_status, "fraction": fraction, "xyz_m": prefix_xyz.tolist(),
                            "quaternion_wxyz": prefix_q.tolist(),
                            "reason": ("supported_model_sweep_frontier" if prefix_supported and fraction < 1.-1e-9 else
                                       "whole_goal_supported" if execution_status == "supported" else reason),
                            "supported_fraction": supported_prefix,
                            "physical_clearance_verified": False}
        whole_goal = {"status": whole_status, "reason": reason if not candidate_failure else "local_command_sweep_not_supported",
                      "xyz_m": target.tolist(), "quaternion_wxyz": q1.tolist(),
                      "unknown_fraction": unknown_fraction, "clearance_estimate_m": None if not math.isfinite(min_gap) else min_gap,
                      "first_uncovered_fraction": sweep["first_uncovered_fraction"],
                      "coverage_witnesses": deepcopy(sweep["coverage_witnesses"]),
                      "missing_evidence_requests": all_missing, "inventory_missing_evidence_requests": missing_evidence,
                      "evidence_ids": sorted(evidence), "stale_evidence_ids": sorted(stale_ids),
                      "monitoring_unknown_ids": sorted(monitoring_unknown),
                      "sweep_samples": len(rows), "at_step": step,
                      "scope": context.get("assessment_scope", "original_requested_goal"),
                      "physical_clearance_verified": False}
        result = {"status": status, "clearance_estimate_m": None if not math.isfinite(min_gap) else min_gap,
                  "unknown_fraction": unknown_fraction, "suggested_steps": suggested,
                  "translation_step_m": translation_step, "rotation_step_rad": rotation_step,
                  "translation_limit_m": translation_limit, "rotation_limit_rad": rotation_limit,
                  "allowed_fraction": fraction, "reason": reason, "evidence_ids": sorted(evidence),
                  "support_region_ids": sorted(supporting_regions), "payload_evidence_ids": sorted(payload_ids),
                  "stale_evidence_ids": sorted(stale_ids), "max_relevant_evidence_age_steps": fresh_age,
                  "monitoring_unknown_ids": sorted(monitoring_unknown),
                  "missing_evidence_requests": all_missing, "inventory_missing_evidence_requests": missing_evidence,
                  "coverage_witnesses": sweep["coverage_witnesses"],
                  "whole_goal_assessment": whole_goal, "execution_prefix": execution_prefix,
                  "execution_status": execution_status, "supported_prefix_fraction": supported_prefix,
                  "first_uncovered_fraction": sweep["first_uncovered_fraction"],
                  "sweep_refinement": sweep["refinement"], "final_interaction_guard": final_interaction,
                  "geometry_computation": {"exact_sweep_cache_hit": sweep.get("exact_cache_hit", False),
                      "supported_route_reuse": sweep.get("supported_route_reuse"),
                      "obstacle_boxes_input": sweep["obstacle_boxes_input"],
                      "obstacle_boxes_checked": sweep["obstacle_boxes_checked"], "counts": dict(self.compute_counts)},
                  "gripper_settling_expected_ids": sorted(settling_ids),
                  "validity_policy": "event_driven_not_age_expiry", "tracking_status": tracking_status,
                  "motion_support_revoked": bool(motion_revoked), "scene_invalidated_by_tracking": False,
                  "limiting_pair": limiting_pair, "command_candidates": candidates,
                  "selected_command_candidate": selected_candidate,
                  "speed_policy": "bounded_command_candidates_checked_in_padded_geometry",
                  "speed_evidence": {"current_clearance_estimate_m": None if not math.isfinite(local_gap) else local_gap,
                                     "remaining_translation_m": distance, "remaining_rotation_rad": angle,
                                     "limiting_pair_extent_uncertainty_m": None if limiting_pair is None else limiting_pair["combined_extent_uncertainty_m"],
                                     "uncertainty_applied_once_to_shapes": True,
                                     "command_units": "OSC_pose_error_increment_not_promised_physical_displacement",
                                     "response_estimate_is_physical_certificate": False},
                  "sweep_samples": len(rows), "interval_padding_m": padding, "payload_geometry_unknown": payload_unknown,
                  "allowed_contact_ids": sorted(allowed), "source": "predicted_scene_and_measured_hand",
                  "clearance_is_physical_certificate": False, "full_arm_checked": False,
                  "scene_revision": self.revision, "at_step": step, "estimated_overlap_escape": escape,
                  "commanded_gripper_sweep": context.get("commanded_gripper") if context.get("commanded_gripper") in ("open", "close") else None}
        result["conditional_probe_geometry_supported"] = bool(probe_ids and status == "supported")
        result["attachment_state_unknown"] = any(e.get("attachment_probe_unconfirmed") or e.get("release_pending")
                                                 for e in self.entities.values())
        result["attachment_probe_hypotheses"] = ["world_stationary", "hand_following"] if probe_ids else []
        result["attachment_verified"] = False
        result["gripper_settling_reference"] = None
        if (status == "supported" and final_interaction and context.get("gripper_action")
                and distance <= 1e-6 and angle <= .001 and context.get("commanded_gripper") in ("open", "close")):
            inventory = context.get("scene_inventory") or {}
            result["gripper_settling_reference"] = {
                "source": "runtime_validated_stationary_gripper", "target_id": context.get("target_id"),
                "held_object_id": context.get("held_object_id"), "gripper_command": context["commanded_gripper"],
                "allowed_contact_ids": list(context.get("allowed_contact_ids", [])),
                "support_ids": list(inventory.get("support_ids", [])),
                "nearby_obstacle_ids": list(inventory.get("nearby_obstacle_ids", [])),
                "entity_revisions": {ident: self.entities[ident]["geometry_revision"] for ident in evidence},
                "hold_position_m": target.tolist(), "hold_quaternion_wxyz": q1.tolist()}
            result["gripper_settling_reference"].update(
                position_tolerance_m=min(.01, max(1e-6, float(context.get("position_tolerance_m", .004)))),
                orientation_tolerance_rad=min(.1, max(.001, float(context.get("orientation_tolerance_rad", .08)))))
        self.last_motion = deepcopy(result)
        return result

    def summary(self, view=None, context=None):
        view = self._last_view if view is None else view
        context = context or {}
        entities = []
        for entity in self.entities.values():
            row = {k: deepcopy(entity[k]) for k in ("id", "role", "parent_id", "position_m", "quaternion_wxyz",
                "boxes", "uncertainty_m", "coverage", "attachment", "related_ids", "source", "observation_step",
                "updated_step", "stale", "stale_reason", "triangulation")}
            row["age_control_steps"] = max(0, self.step-entity["observation_step"])
            row["age_invalidates_evidence"] = False
            row["visual_monitoring"] = deepcopy(entity.get("visual_monitoring"))
            row["visual_checked_step"] = entity.get("visual_checked_step")
            row["monitoring_unknown"] = bool(entity.get("monitoring_unknown", False))
            row["geometry_revision"] = entity.get("geometry_revision", 0)
            for key in ("invalidation_kind", "invalidation_origin_ids", "invalidation_origins", "invalidated_step"):
                row[key] = deepcopy(entity.get(key))
            for key in ("invalidation_local_contact", "local_uncertain_solid_ids", "attachment_probe_unconfirmed",
                        "attachment_probe_shape_usable", "release_probe_shape_usable", "conditional_attachment_supported",
                        "local_pose_revision", "pose_observation_step", "local_pose_evidence", "local_anchor_evidence",
                        "local_perception_evidence", "local_motion_exclusion_boxes",
                        "release_requested_step", "release_actual_command_step", "release_model_attachment_estimate",
                        "release_pending_carried_envelope", "release_pending_world_envelope"):
                row[key] = deepcopy(entity.get(key))
            row["release_pending"] = bool(entity.get("release_pending", False))
            row["attachment_status"] = ("unknown_after_open; carried_extent_retained_until_visual_update"
                                        if row["release_pending"] else "model_or_native_estimate")
            if "attachment_source" in entity:
                row["attachment_source"] = entity["attachment_source"]
                row["attachment_offset_hand_m"] = deepcopy(entity["attachment_offset_hand_m"])
            entities.append(row)
        relations = {}
        if view is not None:
            target = self.entities.get(context.get("target_id"))
            if target:
                delta = np.asarray(target["position_m"])-view.position
                relations = {"target_id": target["id"], "target_from_grip_world_m": delta.tolist(),
                             "target_from_grip_hand_m": (view.R.T @ delta).tolist(),
                             "source": target["source"], "not_a_verified_grasp_pose": True}
        return {"version": GEOMETRY_VERSION, "frame": "world", "quaternion_order": "wxyz",
                "camera_axes": "OpenGL_x_right_y_up_minus_z_forward", "revision": self.revision,
                "entities": entities, "relations": relations, "recent_changes": deepcopy(self.events[-6:]),
                "last_motion_evidence": deepcopy(self.last_motion), "physical_clearance_verified": False,
                "validity_policy": "event_driven_not_age_expiry",
                "dependency_policy": "physical_parent_to_child; related_ids_are_semantic_only",
                "geometry_computation_counts": dict(self.compute_counts),
                "local_perception_stats": self.local_perception.performance_stats() if self.local_perception is not None else None,
                "unknown_space_is_free": False, "native_holding_created_from_model": False}

    def snapshot(self):
        return {"version": GEOMETRY_VERSION, "step": self.step, "revision": self.revision,
                "entities": deepcopy(self.entities), "events": deepcopy(self.events),
                "last_motion": deepcopy(self.last_motion)}
