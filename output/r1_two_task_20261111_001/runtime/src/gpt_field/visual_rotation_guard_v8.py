"""Screen declared RGB neighborhoods before model-commanded Direct rotation.

Image rectangles are model hypotheses, not segmentation or calibrated geometry.
The expanded rectangle is a projected-risk heuristic, not a swept-volume bound.
Translations, contact validity and full-arm clearance are not certified here.
"""
from __future__ import annotations

import math
import numpy as np


def _box(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (result.shape != (4,) or not np.isfinite(result).all()
            or np.any(result < 0.) or np.any(result > 1.)
            or result[0] >= result[2] or result[1] >= result[3]):
        return None
    return result


def projected_rotation_risk(context, actions, *, image_size=(224, 224)):
    """Return a transparent whole-Direct-block rotation screening decision.

    Accepted commands are never clipped or rewritten. A rejected block should
    execute no requested action and ask for a changed, task-directed decision.
    Local measured-orientation holding is a separate, bounded Field executor.
    """
    commands = np.asarray(actions, dtype=float)
    if commands.shape != (5, 7) or not np.isfinite(commands).all() or np.any(abs(commands) > 1.):
        raise ValueError("Rotation screening requires five finite normalized actions")
    size = np.asarray(image_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 1.):
        raise ValueError("image_size must contain width and height greater than one")
    angle = float(.5*np.linalg.norm(commands[:, 3:6], axis=1).sum())
    result = {"version": "rgb-projected-rotation-risk-v8", "accepted": True,
        "reason": "no_requested_rotation", "requested_angular_bound_rad": angle,
        "intersecting_obstacle_ids": [], "clearance_certified": False,
        "geometry_source": "model-declared current external RGB rectangles",
        "limitations": ["projected risk heuristic, not a 3D swept-volume bound",
            "unlisted objects and incorrect rectangles are not detected",
            "no translation or full-arm collision certificate"]}
    if angle <= 1e-12:
        return result

    def reject(reason):
        result.update(accepted=False, reason=reason)
        return result

    if not isinstance(context, dict):
        return reject("rotation_context_missing")
    if context.get("purpose") == "reference_repair":
        return reject("rotation_for_reference_repair_not_allowed")
    assessment = context.get("obstacle_assessment")
    if assessment == "unobservable":
        return reject("rotation_neighborhood_unobservable")
    tool = _box(context.get("tool_external_roi_uv"))
    if tool is None:
        return reject("rotation_tool_envelope_unavailable")
    neighbors = context.get("nearby_obstacles")
    if (not isinstance(neighbors, list) or assessment not in ("present", "none_relevant")
            or (assessment == "present" and not neighbors)
            or (assessment == "none_relevant" and neighbors)):
        return reject("rotation_neighborhood_inconsistent")
    # A declared rectangle's diagonal provides an image scale only. No depth,
    # camera model or hidden object/robot geometry is inferred from this margin.
    scale = size-1.
    extent_px = (tool[2:]-tool[:2])*scale
    margin_px = 2.+float(np.linalg.norm(extent_px))*min(angle, math.pi)
    expanded = tool+np.r_[-margin_px/scale, margin_px/scale]
    result.update(tool_external_roi_uv=tool.tolist(), projected_margin_px=margin_px,
                  projected_envelope_uv=expanded.tolist())
    if np.any(expanded < 0.) or np.any(expanded > 1.):
        return reject("rotation_projected_envelope_outside_view")
    for neighbor in neighbors:
        if not isinstance(neighbor, dict):
            return reject("rotation_obstacle_rectangle_invalid")
        obstacle = _box(neighbor.get("external_roi_uv"))
        identity = neighbor.get("id")
        if obstacle is None or not isinstance(identity, str) or not identity.strip():
            return reject("rotation_obstacle_rectangle_invalid")
        if (expanded[0] <= obstacle[2] and obstacle[0] <= expanded[2]
                and expanded[1] <= obstacle[3] and obstacle[1] <= expanded[3]):
            result["intersecting_obstacle_ids"].append(identity)
    if result["intersecting_obstacle_ids"]:
        return reject("rotation_projected_obstacle_risk")
    result.update(reason="no_declared_projected_rotation_conflict")
    return result
