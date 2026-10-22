"""Empirical sensitivity of a declared outline transported by local RGB points.

The tracker fits a similarity to a small set of correspondences. A good local
fit does not imply that a distant edge of a model-declared whole-tool outline
is equally well measured. This module propagates observed correspondence
residuals through that exact similarity model and also inspects leave-one-out
fits. The resulting interval is a sensitivity set under explicitly stated
pixel-perturbation assumptions, not calibrated confidence or physical clearance.

No point, outline, tracker state, obstacle geometry or motion budget is changed.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


DESCRIPTOR_RESOLUTION_PX = .08


def _design(points):
    points = np.asarray(points, float)
    result = np.zeros((2*len(points), 4))
    result[::2, 0] = result[1::2, 1] = points[:, 0]
    result[::2, 1] = -points[:, 1]
    result[1::2, 0] = points[:, 1]
    result[::2, 2] = result[1::2, 3] = 1.
    return result


def _bounds(points):
    return np.r_[np.min(points, axis=0), np.max(points, axis=0)]


def _failure(reason, representation_key=None):
    return {"available": False, "reason": reason,
            "representation_key": copy.deepcopy(representation_key),
            "clearance_certified": False,
            "empirical_sensitivity_not_calibrated_confidence": True}


def capture_boundary_evidence(track, source_corners, *, representation_key):
    """Snapshot one persistent track and its full declared source-frame outline.

    ``source_corners`` are the same four pre-transform corners used by the guard,
    e.g. ``tool_source`` or a neighbor record's ``source``. The caller supplies a
    measurement-epoch key; it must change when the meaning of the outline does.
    A track-instance token also prevents comparing unrelated reseeded tracks.
    """
    if track is None or not getattr(track, "reliable", False):
        return _failure("unreliable_or_missing_track", representation_key)
    try:
        # Check serializability before retaining an unbounded or opaque key.
        json.dumps(representation_key, allow_nan=False)
        initial = np.asarray(track.initial_points, float)
        active_raw = np.asarray(track.active)
        active = active_raw.astype(int)
        destination = np.asarray(track.points, float)
        corners = np.asarray(source_corners, float)
        matrix = np.asarray(track.matrix, float)
        translation = np.asarray(track.translation, float)
        if (initial.ndim != 2 or initial.shape[1] != 2
                or active.ndim != 1 or len(active) < 3
                or not np.array_equal(active, active_raw)
                or len(set(active.tolist())) != len(active)
                or np.any(active < 0) or np.any(active >= len(initial))
                or destination.shape != (len(active), 2)
                or corners.shape != (4, 2) or matrix.shape != (2, 2)
                or translation.shape != (2,)
                or not all(np.isfinite(x).all() for x in
                           (initial, destination, corners, matrix, translation))):
            return _failure("invalid_correspondence_or_outline", representation_key)
        source = initial[active]
        center = source.mean(axis=0)
        centered = source-center
        design = _design(centered)
        if np.linalg.matrix_rank(design) < 4 or np.sum(centered*centered) < 1e-8:
            return _failure("degenerate_similarity_support", representation_key)
        if np.max(np.ptp(corners, axis=0)) < 1e-8:
            return _failure("degenerate_declared_outline", representation_key)
        inverse = np.linalg.pinv(design)
        fit = inverse@destination.reshape(-1)
        fitted_points = (design@fit).reshape(-1, 2)
        actual_points = source@matrix+translation
        if not np.allclose(fitted_points, actual_points, atol=1e-6, rtol=0.):
            return _failure("track_transform_does_not_match_current_correspondences", representation_key)
        residuals = np.linalg.norm(destination-fitted_points, axis=1)
        perturbations = np.maximum(DESCRIPTOR_RESOLUTION_PX, residuals)
        corner_design = _design(corners-center)
        coefficients = corner_design@inverse
        # Worst component displacement under the declared independent bounded
        # descriptor perturbations; no Gaussian/noise-independence claim.
        radii = (np.abs(coefficients)@np.repeat(perturbations, 2)).reshape(4, 2)
        nominal_corners = corners@matrix+translation
        alternatives = [nominal_corners]
        for excluded in range(len(source)):
            keep = np.ones(2*len(source), dtype=bool)
            keep[2*excluded:2*excluded+2] = False
            reduced = design[keep]
            if np.linalg.matrix_rank(reduced) < 4:
                return _failure("leave_one_out_support_degenerate", representation_key)
            alternative = np.linalg.lstsq(reduced, destination.reshape(-1)[keep], rcond=None)[0]
            alternatives.append((corner_design@alternative).reshape(4, 2))
        alternatives = np.asarray(alternatives)
        lower = np.minimum(nominal_corners-radii, np.min(alternatives, axis=0))
        upper = np.maximum(nominal_corners+radii, np.max(alternatives, axis=0))
        # Intervals for [left, top, right, bottom], retaining the full outline.
        bounds_lower = np.r_[lower.min(axis=0), lower.max(axis=0)]
        bounds_upper = np.r_[upper.min(axis=0), upper.max(axis=0)]
        return {"available": True, "reason": "empirical_similarity_boundary_sensitivity",
            "representation_key": copy.deepcopy(representation_key),
            "track_instance_token": id(track), "identity": str(track.ident),
            "initial_feature_bank_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "source_corners_px": corners.tolist(),
            "active_feature_ids": active.tolist(), "feature_count": len(active),
            "support_center_px": center.tolist(),
            "support_radius_rms_px": float(np.sqrt(np.mean(np.sum(centered*centered, axis=1)))),
            "local_fit_residual_rms_px": float(np.sqrt(np.mean(residuals*residuals))),
            "descriptor_resolution_floor_px": DESCRIPTOR_RESOLUTION_PX,
            "point_perturbation_radius_px": perturbations.tolist(),
            "corner_coordinate_sensitivity_l1": np.abs(coefficients).sum(axis=1).reshape(4, 2).tolist(),
            "corner_coordinate_radius_px": radii.tolist(),
            "leave_one_out_corner_spread_px": np.ptp(alternatives, axis=0).tolist(),
            "nominal_corners_px": nominal_corners.tolist(),
            "nominal_bounds_px": _bounds(nominal_corners).tolist(),
            "bounds_interval_px": [bounds_lower.tolist(), bounds_upper.tolist()],
            "uncertainty_assumption": "Each correspondence coordinate perturbed by at most max(0.08px, its observed 2D fit residual); union with leave-one-out fits.",
            "empirical_sensitivity_not_calibrated_confidence": True,
            "declared_outline_physically_verified": False, "clearance_certified": False}
    except (AttributeError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return _failure("invalid_correspondence_or_outline", representation_key)


def _same_representation(first, second):
    return (first["representation_key"] == second["representation_key"]
        and first["track_instance_token"] == second["track_instance_token"]
        and first["initial_feature_bank_sha256"] == second["initial_feature_bank_sha256"]
        and first["identity"] == second["identity"]
        and np.allclose(first["source_corners_px"], second["source_corners_px"], atol=1e-7, rtol=0.))


def _gap(a, b):
    return float(max(b[0]-a[2], a[0]-b[2], b[1]-a[3], a[1]-b[3]))


def _gap_interval(tool, neighbor):
    tlo, thi = np.asarray(tool["bounds_interval_px"], float)
    nlo, nhi = np.asarray(neighbor["bounds_interval_px"], float)
    lower = max(nlo[0]-thi[2], tlo[0]-nhi[2], nlo[1]-thi[3], tlo[1]-nhi[3])
    upper = max(nhi[0]-tlo[2], thi[0]-nlo[2], nhi[1]-tlo[3], thi[1]-nlo[3])
    return np.array([lower, upper], float)


def compare_gap_evidence(previous_tool, current_tool, previous_neighbor, current_neighbor):
    """Compare a gap only within both unchanged measurement representations.

    Classification is conditional on the snapshots' empirical perturbation sets.
    A narrowing interval alone is precision change, not another executed motion;
    callers must separately account actual transitions and directional response.
    """
    result = {"comparable": False, "classification": "unavailable",
              "gap_change_interval_px": None, "nominal_gap_change_px": None,
              "empirical_sensitivity_not_calibrated_confidence": True,
              "clearance_certified": False,
              "interval_narrowing_is_new_motion": False}
    snapshots = (previous_tool, current_tool, previous_neighbor, current_neighbor)
    if any(not isinstance(row, dict) or not row.get("available") for row in snapshots):
        return {**result, "reason": "missing_or_unusable_boundary_support"}
    try:
        if not (_same_representation(previous_tool, current_tool)
                and _same_representation(previous_neighbor, current_neighbor)):
            return {**result, "reason": "measurement_representation_changed"}
        previous = _gap_interval(previous_tool, previous_neighbor)
        current = _gap_interval(current_tool, current_neighbor)
        interval = np.array([current[0]-previous[1], current[1]-previous[0]])
        old_gap = _gap(previous_tool["nominal_bounds_px"], previous_neighbor["nominal_bounds_px"])
        new_gap = _gap(current_tool["nominal_bounds_px"], current_neighbor["nominal_bounds_px"])
        if (not np.isfinite(np.r_[previous, current, interval, old_gap, new_gap]).all()
                or previous[0] > previous[1] or current[0] > current[1]):
            return {**result, "reason": "invalid_empirical_boundary_interval"}
        classification = "closing" if interval[1] < 0 else "opening" if interval[0] > 0 else "inconclusive"
        return {**result, "comparable": True, "classification": classification,
            "reason": "same_representation_empirical_gap_change",
            "previous_gap_interval_px": previous.tolist(), "current_gap_interval_px": current.tolist(),
            "previous_nominal_gap_px": old_gap, "current_nominal_gap_px": new_gap,
            "nominal_gap_change_px": new_gap-old_gap, "gap_change_interval_px": interval.tolist()}
    except (KeyError, TypeError, ValueError, IndexError):
        return {**result, "reason": "invalid_boundary_snapshot"}
