"""Measured spatial regions and whole-object placement for the user-check variant.

Region specifications are semantic bindings, never simulator object names or an
oracle map. A plane region selects only cells already measured or visually filled
by the scene's observed plane. Unknown and obstacle-fill cells stay excluded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from . import agent_speedup_v2_observers as observers
from . import agent_speedup_v2_planner as planner
from . import agent_speedup_v2_relations as relations
from .agent_speedup_v2_scene import HeldFrameEstimator
from .agent_speedup_v2_types import HF_SOURCES, PIDS, HandPose, ObsReport, jsonable

SPATIAL_VERSION = "user-check-spatial-3"
DIRECTION_SEMANTICS = "external_view_v1"
DIRECTION_RELATIONS = ("left_of", "right_of", "front_of", "behind")
REGION_RELATIONS = ("on", "left_of", "right_of", "front_of", "behind", "near")
PLACEMENT_RELATIONS = ("supported_on", "centered_on", "inside")
_NUM = {"type": "number"}
_NULL = {"type": "null"}
_PAIR = {"type": "array", "items": _NUM, "minItems": 2, "maxItems": 2}
# Match the extension parser's existing 32-item limit on the wire, too.
# These are plan representation limits, not episode/action/time budgets.
SPATIAL_MAX_ITEMS = 32
_PART_ID = {"type": "string", "enum": list(PIDS)}
SPATIAL_PLAN_PROPERTIES = {
    "regions": {"type": "array", "maxItems": SPATIAL_MAX_ITEMS,
        "items": {"type": "object", "additionalProperties": False,
        "properties": {"id": _PART_ID,
            "support": {"type": "string", "enum": [*PIDS, "observed_plane"]},
            "anchor": {"anyOf": [_PART_ID, _NULL]},
            "relation": {"type": "string", "enum": list(REGION_RELATIONS)},
            "frame": {"type": "string", "enum": ["world"]},
            "bounds_xy_m": {"anyOf": [{"type": "array", "items": _PAIR, "minItems": 2,
                                          "maxItems": 2}, _NULL]}},
        "required": ["id", "support", "anchor", "relation", "frame", "bounds_xy_m"]}},
    "placement_goals": {"type": "array", "maxItems": SPATIAL_MAX_ITEMS,
        "items": {"type": "object", "additionalProperties": False,
        "properties": {"obj": _PART_ID, "ref": _PART_ID,
            "relation": {"type": "string", "enum": list(PLACEMENT_RELATIONS)},
            "tolerance_m": {"anyOf": [_NUM, _NULL]}},
        "required": ["obj", "ref", "relation", "tolerance_m"]}},
}
SPATIAL_INSTRUCTIONS = """
A physical object and a surface region are distinct. Both regions and
placement_goals are required arrays; use [] when an extension is unnecessary.
Do not invent or repeat entries to fill an array. Every region id is a distinct
existing part id A-H; each placement (obj,ref) pair appears at most once.
Put a bare surface region in
regions as {id,support,anchor,relation,frame,bounds_xy_m}; keep its id in parts so a
place_on ref can reference it. support is a measured part id or observed_plane;
anchor is a measured part id or null. left_of/right_of/front_of/behind (and these
words in the task) are as seen in image 1, the external camera: right/left = toward
image 1's right/left; behind/front_of = farther from/nearer to that camera along
its viewing direction (toward the top/bottom of image 1). Not the robot's own
left/right; never judged in the wrist images. Use the same viewer words in plan
text; never restate them as world axes. Vertical words mean world up/down. A word
naming part of an object (back compartment, left plate) selects that part by
pointing; it is not a region. frame "world" describes metric bounds only; keep
bounds_xy_m null for directional relations. Bounds are
[[xmin,xmax],[ymin,ymax]] in metres, relative to the anchor's measured centre when
anchor is set, otherwise absolute world coordinates. null bounds use observed
support extent. near requires explicit bounds. Never manufacture safe geometry:
unobserved region cells require observation, while a region already grounded in
observed support may be refreshed with an occupied hand. Entries in placement_goals
specify supported_on, centered_on or inside for an existing place_on (obj,ref).
centered_on asks for the WHOLE OBJECT's centre, not a handle or gripper point;
tolerance_m=null derives tolerance from measured uncertainty and available space.
Only request centered_on/inside when intended by the task. Missing explicit goals
use supported_on. Post-release placement remains unverified until fresh geometry
confirms the whole-object relation. Repoint an incomplete body region or use local
motion to reveal it; a model assertion alone never verifies the goal.
Parts used only as a region support or anchor are legitimate plan dependencies;
do not invent manipulation steps solely to make those parts appear used.
""".strip()


def validate_spatial_spec(plan: dict) -> list[dict]:
    """Return semantic issues; native validation owns ordinary plan fields."""
    issues = []
    ids = {p.get("id") for p in plan.get("parts", []) if isinstance(p, dict)}
    region_ids = set()
    for index, spec in enumerate(plan.get("regions", [])):
        prefix = {"field": "regions", "index": index}
        if not isinstance(spec, dict):
            issues.append(dict(prefix, reason="region_must_be_object")); continue
        pid, support, anchor = spec.get("id"), spec.get("support"), spec.get("anchor")
        if pid not in ids or pid in region_ids:
            issues.append(dict(prefix, reason="region_id_missing_or_duplicate", part=pid))
        region_ids.add(pid)
        if support != "observed_plane" and support not in ids:
            issues.append(dict(prefix, reason="unknown_region_support", part=support))
        if support == pid or anchor == pid or (anchor is not None and anchor not in ids):
            issues.append(dict(prefix, reason="invalid_region_dependency"))
        if spec.get("frame") != "world" or spec.get("relation") not in REGION_RELATIONS:
            issues.append(dict(prefix, reason="invalid_region_frame_or_relation"))
        if spec.get("relation") not in ("on",) and anchor is None:
            issues.append(dict(prefix, reason="relative_region_requires_anchor"))
        bounds = spec.get("bounds_xy_m")
        if spec.get("relation") == "near" and bounds is None:
            issues.append(dict(prefix, reason="near_requires_metric_bounds"))
        if bounds is not None:
            try:
                array = np.asarray(bounds, dtype=float)
                valid = array.shape == (2, 2) and np.isfinite(array).all() and bool((array[:, 0] < array[:, 1]).all())
            except (TypeError, ValueError):
                valid = False
            if not valid:
                issues.append(dict(prefix, reason="invalid_region_bounds"))
    # Regions cannot depend on other regions: support and anchor have physical identity.
    for index, spec in enumerate(plan.get("regions", [])):
        if isinstance(spec, dict) and (spec.get("support") in region_ids or spec.get("anchor") in region_ids):
            issues.append({"field": "regions", "index": index, "reason": "region_requires_physical_support_and_anchor"})
    seen = set()
    for index, goal in enumerate(plan.get("placement_goals", [])):
        if not isinstance(goal, dict):
            issues.append({"field": "placement_goals", "index": index, "reason": "goal_must_be_object"}); continue
        pair = (goal.get("obj"), goal.get("ref"))
        tol = goal.get("tolerance_m")
        if pair[0] not in ids or pair[1] not in ids or pair in seen or pair[0] in region_ids:
            issues.append({"field": "placement_goals", "index": index, "reason": "invalid_placement_binding"})
        seen.add(pair)
        if goal.get("relation") not in PLACEMENT_RELATIONS or (tol is not None and
                (not isinstance(tol, (int, float)) or isinstance(tol, bool) or not math.isfinite(tol) or tol <= 0)):
            issues.append({"field": "placement_goals", "index": index, "reason": "invalid_placement_goal"})
    return issues


def install_spatial_spec(scene, plan: dict) -> None:
    scene.user_check_regions = {r["id"]: copy.deepcopy(r) for r in plan.get("regions", [])}
    scene.user_check_placement_goals = {(g["obj"], g["ref"]): copy.deepcopy(g)
                                        for g in plan.get("placement_goals", [])}
    if not hasattr(scene, "user_check_region_evidence"):
        scene.user_check_region_evidence = {}
    if not hasattr(scene, "user_check_body_frames"):
        scene.user_check_body_frames = {}



def _frame_json(value):
    """Detach finite JSON data with the same canonical form as review artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def build_view_frame(geometry) -> dict:
    """Freeze horizontal viewer directions from validated external calibration.

    R describes the displayed upright, uncropped, non-mirrored OpenGL image; this
    function never flips pixels or uses the wrist camera. World +z is support up.
    At zero roll, projected camera +x defines viewer right, and up cross right
    defines the horizontal viewing direction (image-up for a top-down view).
    This is a global horizontal convention, not a local perspective Jacobian.
    Off-centre image-up can differ even at zero roll; rolled cameras can also
    disagree with local pixel directions. Recorded LIBERO (0 degree) and RoboCasa
    (about 0.9 degree) views are within the intended gravity-upright use. The
    report's sampled table stayed word-consistent through 15 degrees of roll;
    that empirical result is not a guarantee for arbitrary cameras or scenes.
    Large-roll/local-plane semantics and inverted-view policy remain deferred:
    inverted views retain the same formula and are diagnosed, never vetoed.

    Camera numerical validity is checked by the observation geometry boundary.
    Missing calibration is an unavailable fact, including for old/fake reviews.
    The sole geometric availability case is a vertical camera x axis (1e-9 is
    the fixed numerical epsilon specified for this frame, not a tuning threshold).
    The digest includes both full external calibration and image conventions.
    """
    frame = {"direction_semantics": DIRECTION_SEMANTICS, "available": False,
             "reason": "no_external_calibration", "external_camera": None,
             "image": None, "calibration_sha256": None, "right_xy": None,
             "away_xy": None, "roll_deg": None, "inverted": None,
             "relation_directions": {}}
    cameras = geometry.get("cameras") if isinstance(geometry, Mapping) else None
    external = cameras.get("external") if isinstance(cameras, Mapping) else None
    if not isinstance(external, Mapping) or "rotation_world_from_camera" not in external:
        return frame
    # No robot/task/object channels are copied into the frozen semantic frame.
    calibration = {"external_camera": copy.deepcopy(dict(external)),
                   "image": copy.deepcopy(geometry.get("image"))}
    encoded = _frame_json(calibration)
    calibration = json.loads(encoded)
    frame.update(calibration, calibration_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    rotation = np.asarray(calibration["external_camera"]["rotation_world_from_camera"], dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("External camera rotation must be a 3 by 3 matrix")
    right = rotation[:2, 0]
    length = float(np.linalg.norm(right))
    up_components = rotation[2, :2]
    frame.update(inverted=bool(rotation[2, 1] < 0),
                 roll_deg=(None if not np.any(up_components) else
                           math.degrees(math.atan2(float(up_components[0]), float(up_components[1])))))
    if length < 1e-9:
        frame["reason"] = "camera_x_vertical"
        return frame
    right = right / length
    away = np.array([-right[1], right[0]])
    frame.update(available=True, reason=None, right_xy=right.tolist(), away_xy=away.tolist(),
                 relation_directions={"right_of": right.tolist(), "left_of": (-right).tolist(),
                                      "behind": away.tolist(), "front_of": (-away).tolist()})
    return frame


def validate_view_frame(value) -> dict:
    """Return a detached frozen frame, checking its self-contained provenance.

    This checks serialized review data only; it never compares a live camera or
    changes the review's frame. Absence remains unavailable for legacy/fake use.
    """
    if value is None:
        return build_view_frame(None)
    try:
        clean = json.loads(_frame_json(value))
        if not isinstance(clean, dict) or clean.get("direction_semantics") != DIRECTION_SEMANTICS:
            raise ValueError("Unsupported view-frame semantics")
        expected = build_view_frame({"cameras": {"external": clean.get("external_camera")},
                                     "image": clean.get("image")})
        if _frame_json(clean) != _frame_json(expected):
            raise ValueError("View frame does not match its frozen calibration")
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise ValueError("Invalid frozen view frame") from exc
    return clean


def region_directions(view_frame, regions) -> dict:
    """Project reviewed directional labels through the one frozen direction table.

    All consumers (executive, resolver, approval display) use this pure mapping.
    Empty region lists still carry the viewer basis for directional part names.
    Missing frames yield null vectors, never legacy world-axis interpretations.
    """
    frozen = view_frame if isinstance(view_frame, Mapping) else {}
    available = bool(frozen.get("available")) and frozen.get("direction_semantics") == DIRECTION_SEMANTICS
    table = frozen.get("relation_directions", {}) if available else {}
    return {"direction_semantics": DIRECTION_SEMANTICS, "available": available,
            "reason": frozen.get("reason") if available else frozen.get("reason") or "no_external_calibration",
            "frame": {"right_xy": copy.deepcopy(frozen.get("right_xy")) if available else None,
                      "away_xy": copy.deepcopy(frozen.get("away_xy")) if available else None},
            "regions": {spec["id"]: {"relation": spec["relation"], "anchor": spec.get("anchor"),
                                      "direction_world_xy": copy.deepcopy(table.get(spec["relation"]))}
                        for spec in regions if spec.get("relation") in DIRECTION_RELATIONS}}


def _polygon(record):
    raw = getattr(record, "region_xy", None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(-1, 2)
    return arr if len(arr) >= 3 and np.isfinite(arr).all() else None


def _centre(poly):
    contour = np.asarray(poly, dtype=np.float32)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-12:
        return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return np.asarray(poly, dtype=float).mean(axis=0)


def _mask_for_poly(hf, poly, margin=0.0):
    mask = np.zeros(hf.shape, dtype=np.uint8)
    cells = hf.cells_in_polygon(poly)
    if len(cells):
        mask[cells[:, 0], cells[:, 1]] = 1
    radius = int(math.ceil(max(0.0, margin) / hf.cell_m))
    if radius:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return mask.astype(bool)


def _ground_observed_region(ctx, pid: str) -> dict:
    """Resolve a region from observed support without asking object segmentation for a plane."""
    scene, hf = ctx.scene, ctx.scene.hf
    spec = getattr(scene, "user_check_regions", {}).get(pid)
    result = {"part": pid, "kind": "region", "ready": False, "reason": None, "cells": 0,
              "source": "observed_support_geometry", "step": int(ctx.step)}
    if spec is None:
        return dict(result, reason="not_a_spatial_region")
    record = scene.parts.get(pid)
    if record is None or hf is None:
        return dict(result, reason="support_geometry_unavailable")
    support_pid = spec["support"]
    support = scene.parts.get(support_pid)
    held = getattr(ctx, "holding", None)
    held_parts = ({held.pid, held.grip_pid} if held is not None else set())
    if support_pid != "observed_plane" and (not relations._registered(support) or support_pid in held_parts):
        return dict(result, reason="support_current_pose_unmeasured", dependency=support_pid)
    surface = relations._base_plane(ctx) if support_pid == "observed_plane" else getattr(support, "top_z", None)
    if surface is None:
        return dict(result, reason="support_height_unmeasured", dependency=support_pid)
    # plane fill is grounded by an observed plane and visible colour evidence;
    # obstacle fill and unknown cells are never promoted into a destination.
    mask = hf.known & (hf.measured | (hf.source == HF_SOURCES.index("plane")))
    sigma = np.nan_to_num(hf.sigma, nan=0.0)
    tolerance = np.maximum(hf.cell_m, 3.0 * np.hypot(sigma, float(surface.sigma)))
    mask &= np.isfinite(hf.z) & (np.abs(hf.z - float(surface.value)) <= tolerance)
    if support_pid != "observed_plane":
        poly = _polygon(support)
        if poly is None:
            return dict(result, reason="support_extent_unmeasured", dependency=support_pid)
        mask &= _mask_for_poly(hf, poly)
    gy, gx = np.indices(hf.shape)
    xs, ys = hf.origin_xy[0] + gx * hf.cell_m, hf.origin_xy[1] + gy * hf.cell_m
    origin = np.zeros(2)
    anchor_pid = spec.get("anchor")
    if anchor_pid is not None:
        anchor = scene.parts.get(anchor_pid)
        if not relations._registered(anchor) or anchor_pid in held_parts:
            return dict(result, reason="anchor_current_pose_unmeasured", dependency=anchor_pid)
        poly = _polygon(anchor)
        if poly is None:
            return dict(result, reason="anchor_extent_unmeasured", dependency=anchor_pid)
        origin = _centre(poly)
        relation = spec["relation"]
        if relation in DIRECTION_RELATIONS:
            directions = region_directions(getattr(ctx, "user_check_view_frame", None), [spec])
            direction = directions["regions"][pid]["direction_world_xy"]
            result.update(view_frame=directions["frame"], direction_semantics=DIRECTION_SEMANTICS,
                          direction_world_xy=direction)
            if not directions["available"] or direction is None:
                return dict(result, reason="view_frame_unavailable")
            d = np.asarray(direction, dtype=float)
            mask &= xs * d[0] + ys * d[1] > np.max(poly @ d)
        if relation == "on": mask &= _mask_for_poly(hf, poly)
    bounds = spec.get("bounds_xy_m")
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        mask &= (xs >= origin[0] + b[0, 0]) & (xs <= origin[0] + b[0, 1])
        mask &= (ys >= origin[1] + b[1, 0]) & (ys <= origin[1] + b[1, 1])
    excluded = []
    regions = getattr(scene, "user_check_regions", {})
    for other, part in scene.parts.items():
        if other in held_parts or other in regions or other == support_pid or (other == anchor_pid and spec["relation"] == "on"):
            continue
        poly = _polygon(part)
        if poly is None or getattr(part, "top_z", None) is None:
            continue
        if float(part.top_z.value) <= float(surface.value) + max(hf.cell_m, 3 * surface.sigma):
            continue
        uncertainty = float(getattr(part, "sigma_h_m", 0.0) or 0.0)
        mask &= ~_mask_for_poly(hf, poly, margin=max(hf.cell_m, uncertainty))
        excluded.append(other)
    cells = np.argwhere(mask)
    result.update(cells=int(len(cells)), support=support_pid, anchor=anchor_pid, excluded_objects=excluded,
                  surface_z_m=float(surface.value), observed_only=True)
    if len(cells) < 3:
        result["reason"] = "observed_region_empty"
        scene.user_check_region_evidence[pid] = result
        return result
    xy = hf.origin_xy + cells[:, ::-1] * hf.cell_m
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        result["reason"] = "observed_region_collapsed"
        scene.user_check_region_evidence[pid] = result
        return result
    centre = xy.mean(axis=0)
    # region_cells, not this display hull, define every permitted landing cell;
    # holes and unobserved space inside the hull remain unavailable to aim_target.
    scene.parts[pid] = replace(record, xyz=np.r_[centre, float(surface.value)],
        cov=np.diag([hf.cell_m ** 2, hf.cell_m ** 2, float(surface.sigma) ** 2]),
        top_z=surface, support_z=surface, region_xy=hull, region_cells=cells,
        sigma_h_m=float(surface.sigma), registration_step=int(ctx.step), status="registered",
        registered_empty_hand=True, provisional=False, partial=False, top_bound=None,
        top_upper_z=None, top_band=None, axis_check=None, elongation=None, width_by_yaw=(),
        reason=None)
    if not hasattr(hf, "user_check_region_ids"):
        hf.user_check_region_ids = set()
    hf.user_check_region_ids.add(pid)
    result.update(ready=True, reason=None, centre_xy_m=centre.tolist(),
                  registration_basis="observed_support_not_hand_dependent_segmentation")
    scene.user_check_region_evidence[pid] = result
    return result


def ground_region(ctx, pid: str) -> dict:
    """Publish measured region geometry, invalidating it if a dependency is lost."""
    result = _ground_observed_region(ctx, pid)
    scene = ctx.scene
    if pid not in getattr(scene, "user_check_regions", {}):
        return result
    scene.user_check_region_evidence[pid] = result
    if not result["ready"]:
        record = scene.parts.get(pid)
        if record is not None:
            scene.parts[pid] = replace(record, status="seeded", reason=None, xyz=None, cov=None,
                top_z=None, support_z=None, region_xy=None, region_cells=None, sigma_h_m=None,
                registered_empty_hand=False, registration_step=None, provisional=False, partial=False,
                top_bound=None, top_upper_z=None, top_band=None, width_by_yaw=None, elongation=None,
                axis_check=None, track=None, track_pose=None, external=None, rigid_patch=None)
        if scene.hf is not None:
            getattr(scene.hf, "user_check_region_ids", set()).discard(pid)
        # A stale spatial-region volume must not remain active after its
        # physical support or anchor can no longer be located.
        for key, keepout in list(getattr(scene, "keepouts", {}).items()):
            if getattr(keepout, "pid", None) == pid:
                del scene.keepouts[key]
    return result


def region_precondition(ctx, record):
    evidence = ground_region(ctx, record.pid)
    return {"ok": evidence["ready"], "reason": None if evidence["ready"] else "geometry_unmeasured",
            "values": {"region_cells": evidence["cells"], "surface_z_m": evidence.get("surface_z_m")},
            "thresholds": [], "support_source": "observed_region", "cue": "measured_surface_cells",
            "region": evidence}


def aim_target(ref, held, hf, keepouts, pose):
    """Keep the actual landing point in an observed, permitted region cell.

    Native centre-first aiming can select the arithmetic centre of a disconnected
    region, which may be an unobserved hole. Preserve its ordinary result when
    usable; otherwise choose an observed free cell with measured clearance.
    """
    result = planner.aim_target(ref, held, hf, keepouts, pose)
    if not result.get("ok") or ref.pid not in getattr(hf, "user_check_region_ids", set()):
        return result
    region = planner._region_cells(ref, hf)
    region_set = {(int(i), int(j)) for i, j in region}
    candidate = hf.index(result["target_obj_xy"])
    if candidate is not None and candidate in region_set:
        return result
    field = planner._Field(hf, keepouts, exclude_pids={held.pid, held.grip_pid, ref.pid},
                           exclude_cells=held.vacated_cells)
    iy, ix = region[:, 0], region[:, 1]
    z = np.where(hf.known[iy, ix] & ~field.skip[iy, ix], hf.z[iy, ix], -np.inf)
    fills = field.unverified[iy, ix]
    z = np.where(fills, z - (planner.D7B_FILL_SIGMA_K - planner.PV_PARENT_SIGMA_K) *
                 np.nan_to_num(hf.sigma[iy, ix], nan=0.0), z)
    band = max(planner.M8_FLOOR_M, planner.M8_SIGMA_K * float(ref.top_z.sigma))
    free = region[np.maximum(z, field.overlay[iy, ix]) <= float(ref.top_z.value) + band]
    if not len(free):
        return dict(result, ok=False, reason="support_above_destination", target_obj_xy=None,
                    target_grip_xy=None, target_rule="observed_region_has_no_free_cell")
    mask = np.zeros(hf.shape, dtype=np.uint8)
    mask[free[:, 0], free[:, 1]] = 1
    padded = np.pad(mask, 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    clearance = np.maximum(distance[free[:, 0], free[:, 1]] * hf.cell_m - hf.cell_m / 2, 0.0)
    xy = hf.origin_xy + free[:, ::-1] * hf.cell_m
    centre = planner._centre_xy(ref, hf)
    to_centre = np.linalg.norm(xy - centre, axis=1)
    need = held.r_obj_m + max(planner.M16_FLOOR_M, held.sigma_offset_m)
    fits = clearance >= need
    candidates = np.flatnonzero(fits if fits.any() else clearance >= np.max(clearance) - 1e-9)
    index = int(candidates[np.argmin(to_centre[candidates])])
    target, radius = xy[index], float(clearance[index])
    margin = radius - held.r_obj_m
    tolerance = planner.th_max("M4_aim_tol_m",
        planner.Term("DD", planner.M4_FLOOR_M, "M4_FLOOR_M"),
        planner.Term("ONLINE", planner.M4_FRACTION * max(margin, 0.0),
                     "M4_FRACTION * observed region clearance"), unit="m")
    tol = tolerance.value
    thresholds = [t for t in result.get("thresholds", []) if t.name != "M4_aim_tol_m"] + [tolerance]
    return dict(result, target_obj_xy=target.tolist(),
                target_grip_xy=(target - (pose.R @ held.offset_h)[:2]).tolist(),
                tol_m=(tol, tol), m_free_m=margin, r_free_m=radius, fits=bool(fits[index]),
                free_spot=True, target_rule="observed_free_cell_nearest_centre",
                target_to_centre_m=float(to_centre[index]),
                flags=[] if fits[index] else ["tight_fit"], thresholds=thresholds)


class WholeObjectHeldFrameEstimator(HeldFrameEstimator):
    """Preserve measured body/hand geometry separately from a contact patch.

    Native HS-H1 already knows how to select a current whole-body polygon. This
    variant retains its provenance and local outline through every disturbance;
    no body extent is invented when only a grasp point is measured.
    """
    def prior_at_close(self, obj_pid, grip_pid, view, **kwargs):
        body = self.scene.body_record(obj_pid)
        body = None if body is None else copy.deepcopy(body)
        held = super().prior_at_close(obj_pid, grip_pid, view, **kwargs)
        pose = HandPose.from_view(view)
        poly = self.scene._held_local.get(f"{obj_pid}:polygon")
        log = self.last_prior.get("body", {}) if isinstance(self.last_prior, dict) else {}
        usable = (body is not None and poly is not None and len(poly) >= 3 and
                  log.get("why") not in ("not_adjacent_to_grip", "grip_outside_body", "overlaps_other_object",
                                         "record_invariant", "no_body_record"))
        picked = self.scene.parts.get(obj_pid)
        height = (None if picked is None or picked.top_z is None else
                  float(picked.top_z.value) - float(held.support_z_pick.value))
        height_sigma = (None if picked is None or picked.top_z is None else
                        math.hypot(float(picked.top_z.sigma), float(held.support_z_pick.sigma)))
        frame = {"obj": obj_pid, "grip": grip_pid, "verified_extent": bool(usable),
                 "pick_body_height_m": height, "body_height_sigma_m": height_sigma,
                 "source": "current_epoch_whole_body" if usable else "contact_patch_only",
                 "step": int(getattr(self.scene, "step", 0)), "offset_h": held.offset_h.copy(),
                 "radius_m": float(held.r_obj_m), "sigma_offset_m": float(held.sigma_offset_m),
                 "outline_h": None if poly is None else np.asarray(poly, dtype=float).copy(),
                 "registration": log}
        if usable:
            # Explicitly form centre offset from the complete attached outline;
            # subsequent native aim_target consumes this exact held.offset_h.
            world_outline = pose.grip[None, :] + np.asarray(poly) @ pose.R.T
            centre = _centre(world_outline[:, :2])
            bottom = float(held.lowest_point_z(pose))
            held.offset_h = pose.R.T @ (np.r_[centre, bottom] - pose.grip)
            held.r_obj_m = float(np.max(np.linalg.norm(world_outline[:, :2] - centre, axis=1)))
            frame.update(offset_h=held.offset_h.copy(), radius_m=held.r_obj_m)
        if not hasattr(self.scene, "user_check_body_frames"):
            self.scene.user_check_body_frames = {}
        self.scene.user_check_body_frames[obj_pid] = frame
        return held


def refresh_whole_object_frame(ctx, held=None):
    """Expose current measured hand/object transform for grasp handoff and feedback."""
    held = ctx.holding if held is None else held
    if held is None:
        return {"verified": False, "reason": "nothing_held"}
    frame = getattr(ctx.scene, "user_check_body_frames", {}).get(held.pid, {})
    return jsonable({"verified": bool(frame.get("verified_extent")), "obj": held.pid,
                    "grip": held.grip_pid, "centre_world_m": held.object_ref_point(relations._pose(ctx)),
                    "offset_h_m": held.offset_h, "radius_m": held.r_obj_m,
                    "sigma_offset_m": held.sigma_offset_m, "source": frame.get("source", "unmeasured_body")})


def _goal(ctx, obj, ref):
    return getattr(ctx.scene, "user_check_placement_goals", {}).get((obj, ref),
            {"obj": obj, "ref": ref, "relation": "supported_on", "tolerance_m": None})


def _body_snapshot(ctx, obj):
    record = ctx.scene.parts.get(obj)
    held = ctx.holding
    frame = getattr(ctx.scene, "user_check_body_frames", {}).get(obj, {})
    poly = _polygon(record)
    outline_h = getattr(ctx.scene, "_held_local", {}).get(f"{obj}:polygon", frame.get("outline_h"))
    body_height = frame.get("pick_body_height_m")
    if held is not None and outline_h is not None:
        pose = relations._pose(ctx)
        world = pose.grip[None, :] + np.asarray(outline_h) @ pose.R.T
        body_height = float(np.max(world[:, 2]) - held.lowest_point_z(pose))
    return {"record": record, "step": int(ctx.step), "body_height_m": body_height,
            "body_height_sigma_m": frame.get("body_height_sigma_m"),
            "polygon": None if poly is None else poly.copy(),
            "frame": copy.deepcopy(frame), "held": held,
            "centre": None if held is None else held.object_ref_point(relations._pose(ctx)).copy()}


def _measure_goal(ctx, obj, ref, pre, support, record):
    goal, dest = _goal(ctx, obj, ref), ctx.scene.parts.get(ref)
    poly, dest_poly = _polygon(record), _polygon(dest)
    report = {"goal": goal, "verified": False, "reason": None, "source": "fresh_post_release_geometry",
              "observation_step": int(ctx.step), "release_step": int(pre.get("step", -1))}
    if record is None or record.registration_step is None or record.registration_step <= pre.get("step", -1):
        return dict(report, reason="fresh_object_geometry_required")
    if poly is None or dest_poly is None or record.partial or record.provisional:
        return dict(report, reason="whole_object_extent_unmeasured")
    prior = pre.get("whole_object", {})
    prior_frame = prior.get("frame", {})
    prior_poly = prior.get("polygon")
    area = abs(float(cv2.contourArea(poly.astype(np.float32))))
    # A contracted contact patch cannot establish final whole-object placement.
    expected_area = 0.0 if prior_poly is None else abs(float(cv2.contourArea(np.asarray(prior_poly, np.float32))))
    area_ratio = None if expected_area <= 1e-12 else area / expected_area
    body = ctx.scene.body_record(obj)
    fresh_body = body is not None and int(body.get("step", -1)) > pre.get("step", -1)
    if not fresh_body or (prior_frame.get("verified_extent") and area_ratio is not None and area_ratio < 0.5):
        return dict(report, reason="fresh_registration_is_contact_patch", area_ratio=area_ratio,
                    measured_area_m2=area, expected_area_m2=expected_area)
    centre, target = _centre(poly), planner._centre_xy(dest, ctx.scene.hf)
    hf = ctx.scene.hf
    dest_cells = planner._region_cells(dest, hf)
    body_cells = hf.cells_in_polygon(poly)
    available = {(int(i), int(j)) for i, j in dest_cells}
    share = (sum((int(i), int(j)) in available for i, j in body_cells) / len(body_cells)) if len(body_cells) else 0.0
    centre_ij = hf.index(centre)
    centre_inside = centre_ij is not None and tuple(centre_ij) in available
    offset = float(np.linalg.norm(centre - target))
    sigma_xy = max(hf.cell_m, float(getattr(record, "sigma_h_m", 0.0) or 0.0),
                   float(getattr(dest, "sigma_h_m", 0.0) or 0.0))
    uncertainty = 3.0 * sigma_xy
    requested = goal.get("tolerance_m")
    # Default centering precision follows measured grid/pose uncertainty; it is
    # not selected by object type or task id.
    tolerance = float(requested) if requested is not None else uncertainty
    centre_pass = offset <= tolerance
    relation_pass = centre_inside and share > 0
    if goal["relation"] == "centered_on": relation_pass &= centre_pass
    if goal["relation"] == "inside": relation_pass &= share >= 1.0 - 1.0 / max(len(body_cells), 1)
    expected = prior.get("record")
    s_dest = (support or {}).get("S_dest", None if dest.top_z is None else dest.top_z.value)
    height_ok = False
    dz, height_tolerance = None, None
    body_height = prior.get("body_height_m")
    if (record.top_z is not None and expected is not None and expected.top_z is not None and
            body_height is not None and s_dest is not None):
        expected_top = float(s_dest) + body_height
        dz = float(record.top_z.value) - expected_top
        body_sigma = prior.get("body_height_sigma_m")
        body_sigma = float(expected.top_z.sigma) if body_sigma is None else float(body_sigma)
        surface_sigma = 0.0 if dest.top_z is None else float(dest.top_z.sigma)
        expected_sigma = math.hypot(body_sigma, surface_sigma)
        height_tolerance = max(hf.cell_m, 3 * math.hypot(record.top_z.sigma, expected_sigma))
        height_ok = abs(dz) <= height_tolerance
    report.update(centre_xy_m=centre.tolist(), target_centre_xy_m=target.tolist(), offset_m=offset,
                  tolerance_m=tolerance, position_uncertainty_m=uncertainty,
                  centre_in_destination=bool(centre_inside), footprint_in_destination_share=float(share),
                  fresh_body=True, area_ratio=area_ratio, top_error_m=dz, top_tolerance_m=height_tolerance,
                  support_height_confirmed=bool(height_ok), verified=bool(relation_pass and height_ok))
    report["reason"] = None if report["verified"] else ("placement_relation_not_met" if not relation_pass
                                                        else "placement_height_unverified")
    return report


def install_spatial_layer(layer):
    """Install private relation hooks; base modules and their globals are untouched."""
    namespace = getattr(layer, "USER_CHECK_NAMESPACE", None)
    if namespace is None:
        return layer
    planner_view = SimpleNamespace(**vars(namespace["planner"]))
    planner_view.aim_target = aim_target
    namespace["planner"] = planner_view
    layer.planner = planner_view
    original_capture = namespace["_pre_release_capture"]
    original_reacquire = namespace["_reacquire_at_rest"]
    original_register = namespace["register_regions"]
    original_survey = namespace["survey_hover"]
    original_place_on = namespace["place_on"]

    def capture(ctx, obj):
        result = original_capture(ctx, obj)
        result["whole_object"] = _body_snapshot(ctx, obj)
        return result

    def post_release(ctx, obj, ref, pre_release, support):
        expected = ctx.scene.parts.get(obj)
        surveyed = yield from original_reacquire(ctx, obj, expected)
        report = _measure_goal(ctx, obj, ref, pre_release or {}, support, surveyed.get("record"))
        report["survey_reason"] = surveyed.get("reason")
        ctx.memory["user_check_placement_validation"] = jsonable(report)
        if not report["verified"]:
            callback = ctx.memory.get("user_check_checkpoint")
            step = ctx.memory.get("user_check_step")
            if callable(callback) and step is not None:
                handoff = yield from callback("placement_validation", step, report)
                # Local recovery changes viewpoints/physical pose; it never
                # substitutes for remeasurement of the released object's goal.
                if handoff.get("resume") and handoff.get("changed"):
                    surveyed = yield from original_reacquire(ctx, obj, ctx.scene.parts.get(obj))
                    report = _measure_goal(ctx, obj, ref, pre_release or {}, support, surveyed.get("record"))
                    report["survey_reason"] = surveyed.get("reason")
                    ctx.memory["user_check_placement_validation"] = jsonable(report)
        return {"ok": bool(report["verified"]), "evidence": jsonable(report), "text_items": [],
                "flag": None if report["verified"] else "post_release_unverified"}

    def register_regions(ctx, plan, roles):
        region_specs = getattr(ctx.scene, "user_check_regions", {})
        requested = [pid for pid in region_specs if roles.get(pid)]
        physical_roles = {pid: values for pid, values in roles.items() if pid not in region_specs}
        # A semantic region's dependencies need geometry even if not themselves
        # an object of a manipulation step.
        for pid in requested:
            spec = region_specs[pid]
            for dep in (spec.get("support"), spec.get("anchor")):
                if dep and dep != "observed_plane":
                    physical_roles[dep] = frozenset(set(physical_roles.get(dep, ())) | {"ref"})
        outcomes = yield from original_register(ctx, plan, physical_roles)
        for pid in requested:
            result = ground_region(ctx, pid)
            outcomes.append(relations._outcome(ctx, "registered",
                "succeeded" if result["ready"] else "failed",
                "registered" if result["ready"] else "geometry_unmeasured", pids=relations._pids(pid),
                evidence={"spatial_region": result, "execution_feedback": {
                    "status": "succeeded" if result["ready"] else "blocked",
                    "executed": False, "geometry_ready": result["ready"],
                    "unmet_conditions": [] if result["ready"] else [{"predicate": "observed_spatial_region",
                        "part": pid, "value": False, "reason": result.get("reason"),
                        "dependency": result.get("dependency")}]}}, start=int(ctx.step)))
        return outcomes

    def survey_hover(ctx, pid, *args, **kwargs):
        if pid in getattr(ctx.scene, "user_check_regions", {}):
            from .agent_speedup_v2_user_check_skills import observe_spatial_region
            return (yield from observe_spatial_region(ctx, pid, relation_layer=layer))
        return (yield from original_survey(ctx, pid, *args, **kwargs))

    def place_on(ctx, step):
        result = yield from original_place_on(ctx, step)
        if result.status != "succeeded":
            return result
        validation = copy.deepcopy(ctx.memory.get("user_check_placement_validation", {}))
        # A persistent spatial relation depends on BOTH bodies. Tracking only
        # the released object would miss a support/anchor moving underneath it.
        spec = getattr(ctx.scene, "user_check_regions", {}).get(step.ref)
        dependencies = [step.obj]
        if spec is None:
            dependencies.append(step.ref)
        else:
            dependencies.extend(p for p in (spec.get("support"), spec.get("anchor"))
                                if p and p != "observed_plane")
        checks, unavailable = {}, []
        for pid in dict.fromkeys(dependencies):
            region = namespace["_external_region"](ctx, pid)
            if region is None:
                unavailable.append(pid)
            else:
                checks[pid] = observers.RegionStaysCheck(region=region,
                    baseline_rgb=np.asarray(ctx.view.external_rgb_upright).copy(),
                    projector_mask=namespace["_hand_mask_fn"](ctx),
                    sigma_px=namespace["_sigma_px"](ctx, "external"), false_rate=0.0, kind="unchanged")

        def stays(view):
            reports = {pid: check(view) for pid, check in checks.items()}
            verdicts = [r.verdict for r in reports.values()]
            verdict = ("violated" if "violated" in verdicts else
                       "pending" if unavailable or "pending" in verdicts else "met")
            evidence = {"placement_goal": validation.get("goal"), "verified_at_step": validation.get("observation_step"),
                        "dependencies": {pid: {"verdict": r.verdict, "reason": r.reason} for pid, r in reports.items()},
                        "unobserved_dependencies": unavailable}
            return ObsReport("whole_object_placement_stays", verdict, None,
                None if verdict == "met" else "spatial_relation_requires_revalidation", evidence, (),
                int(getattr(ctx, "step", 0)))

        ctx.stays.add(f"stays:supported_on:{step.obj}:{step.ref}", stays, step_i=step.i, kind="step")
        return result

    namespace["place_on"] = place_on
    layer.place_on = place_on
    namespace["register_regions"] = register_regions
    namespace["survey_hover"] = survey_hover
    layer.register_regions = register_regions
    layer.survey_hover = survey_hover
    namespace["_pre_release_capture"] = capture
    namespace["_post_release_check"] = post_release
    layer._pre_release_capture = capture
    layer._post_release_check = post_release
    return layer
