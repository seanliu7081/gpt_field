"""Coherent observed surface patches for local planning and motion feedback.

The returned cloud is a derived subset of the original observations. Density and
normal consistency reject unsupported depth edges and isolated patches; they do
not reconstruct unseen surfaces or infer an object's complete collision volume.
All thresholds below are dimensionless neighbourhood/consensus choices. There
are no object classes, task identifiers, expected object sizes, or metre-scale
shape cutoffs.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import blake2b
import json
from threading import RLock
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SURFACE_VERSION = "coherent-observed-surface-1"
_NEIGHBOURS = 8
_LINK_NEIGHBOURS = 16
_DENSITY_MAD_MULTIPLIER = 3.
_DENSITY_MIN_MULTIPLIER = 2.
_LINK_SCALE_MULTIPLIER = 2.
_NORMAL_AGREEMENT = .5
_DOMINANT_MASS_FRACTION = .60
_SECOND_COMPONENT_RATIO = .50
_MAX_CACHE_ENTRIES = 32
_MAX_CACHE_POINTS = 1_000_000
_CACHE = OrderedDict()
_CACHE_POINTS = 0
_CACHE_LOCK = RLock()


def _empty():
    points = np.empty((0, 3), dtype=float)
    points.setflags(write=False)
    return points


def _metadata(raw_count):
    return {"version": SURFACE_VERSION, "status": "insufficient_geometry",
            "raw_point_count": int(raw_count), "planning_point_count": 0,
            "raw_geometry_preserved": True, "selection": "derived_observed_subset",
            "complete_object_shape": False, "unseen_surfaces": "unknown",
            "collision_volume_certified": False, "error_bound_m": None,
            "filtering_does_not_prove_removed_points_are_false": True,
            "rules": {"neighbours": _NEIGHBOURS, "link_neighbours": _LINK_NEIGHBOURS,
                      "density_mad_multiplier": _DENSITY_MAD_MULTIPLIER,
                      "density_min_multiplier": _DENSITY_MIN_MULTIPLIER,
                      "link_scale_multiplier": _LINK_SCALE_MULTIPLIER,
                      "normal_agreement_cosine": _NORMAL_AGREEMENT,
                      "dominant_mass_fraction": _DOMINANT_MASS_FRACTION,
                      "second_component_ratio": _SECOND_COMPONENT_RATIO,
                      "thresholds_are_calibrated_probabilities": False}}


def _support_labels(count, ranges):
    """Recover source-view and support labels without modifying provenance."""
    source = np.full(count, -1, dtype=np.int32)
    supported = np.zeros(count, dtype=bool)
    names = {}
    for item in ranges:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start"), item.get("end")
        name = item.get("view")
        if (type(start) is not int or type(end) is not int or start < 0 or end > count
                or start >= end or not isinstance(name, str)):
            continue
        label = names.setdefault(name, len(names))
        source[start:end] = label
        if item.get("support") == "crossview_supported":
            supported[start:end] = True
    return source, supported, {value: key for key, value in names.items()}


def _cache_key(points, normals, ranges, status):
    digest = blake2b(digest_size=20)
    digest.update(SURFACE_VERSION.encode())
    digest.update(str(status).encode())
    for array in (points, normals):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.tobytes(order="C"))
    digest.update(json.dumps(ranges, sort_keys=True, separators=(",", ":"), default=str).encode())
    return digest.digest()


def planning_surface(obj):
    """Return ``(points[N,3], evidence)`` for a coherent observed surface.

    Empty/ambiguous inputs return an empty cloud and an explicit reason. The
    caller decides how an unavailable payload estimate affects execution. Input
    coordinates, normals, masks and provenance are never changed. Content-based
    caching also notices an accidental in-place input revision, and lets the
    executive summary, tracker and planner share the same geometric work.
    """
    global _CACHE_POINTS
    if obj is None:
        return _empty(), {**_metadata(0), "reason": "object_unavailable"}
    try:
        points = np.asarray(obj.points, dtype=float)
    except (TypeError, ValueError, AttributeError):
        return _empty(), {**_metadata(0), "reason": "object_points_unavailable"}
    if points.ndim != 2 or points.shape[1] != 3:
        return _empty(), {**_metadata(0), "reason": "object_points_not_n_by_3"}
    normals = np.asarray(getattr(obj, "normals", np.empty((0, 3))), dtype=float)
    if normals.shape != points.shape:
        normals = np.zeros_like(points)
    ranges = (getattr(obj, "evidence", {}) or {}).get("point_support_ranges", [])
    status = getattr(obj, "status", "unknown")
    key = _cache_key(points, normals, ranges, status)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached[0], deepcopy(cached[1])
    started = perf_counter()
    result, evidence = _extract(points, normals, ranges, status)
    evidence["extraction_s"] = perf_counter() - started
    result = np.ascontiguousarray(result, dtype=float)
    result.setflags(write=False)
    with _CACHE_LOCK:
        previous = _CACHE.pop(key, None)
        if previous is not None:
            _CACHE_POINTS -= len(previous[0])
        _CACHE[key] = (result, evidence)
        _CACHE_POINTS += len(result)
        while len(_CACHE) > _MAX_CACHE_ENTRIES or _CACHE_POINTS > _MAX_CACHE_POINTS:
            _, (removed, _) = _CACHE.popitem(last=False)
            _CACHE_POINTS -= len(removed)
    return result, deepcopy(evidence)


def _extract(points, normals, ranges, status):
    evidence = _metadata(len(points))
    if status in {"missing", "ambiguous", "conflict", "stale"}:
        return _empty(), {**evidence, "reason": "object_geometry_" + status}
    finite = np.isfinite(points).all(axis=1)
    normal_length = np.linalg.norm(normals, axis=1)
    valid_normal = np.isfinite(normals).all(axis=1) & (normal_length > .5)
    unit = np.zeros_like(normals)
    unit[valid_normal] = normals[valid_normal] / normal_length[valid_normal, None]
    source, supported, source_names = _support_labels(len(points), ranges)
    eligible = finite & (valid_normal | supported)
    # A surface may legitimately lack local tangent estimates. Keep that case
    # explicit and use density only, rather than inventing normals or a solid.
    normal_evidence = bool(np.any(valid_normal & finite))
    if not np.any(eligible) and np.any(finite):
        eligible = finite.copy()
    indices = np.flatnonzero(eligible)
    evidence.update({"finite_point_count": int(finite.sum()),
                     "valid_normal_point_count": int((valid_normal & finite).sum()),
                     "crossview_supported_input_points": int((supported & finite).sum()),
                     "normal_evidence_available": normal_evidence,
                     "source_views": sorted(source_names.values()),
                     "normal_or_support_eligible_points": len(indices)})
    if len(indices) < 4:
        return _empty(), {**evidence, "reason": "too_few_supported_local_points"}

    cloud = points[indices]
    labels = source[indices]
    density_ok = np.zeros(len(indices), dtype=bool)
    normal_ok = np.ones(len(indices), dtype=bool)
    scales = np.zeros(len(indices))
    density_reports = []
    # Each camera has its own point spacing. A sparse external view is not
    # rejected simply because the wrist camera observes many more pixels.
    for label in np.unique(labels):
        local = np.flatnonzero(labels == label)
        if len(local) < 4:
            density_reports.append({"view": source_names.get(int(label), "unspecified"),
                                    "points": len(local), "status": "too_few_local_samples"})
            continue
        samples = cloud[local]
        k = min(_NEIGHBOURS + 1, len(local))
        distances, neighbours = cKDTree(samples).query(samples, k=k)
        radii = distances[:, -1]
        positive = radii[np.isfinite(radii) & (radii > np.finfo(float).eps)]
        if not len(positive):
            density_reports.append({"view": source_names.get(int(label), "unspecified"),
                                    "points": len(local), "status": "degenerate_coincident_points"})
            continue
        typical = float(np.median(positive))
        mad = float(np.median(np.abs(positive - typical)))
        cutoff = max(_DENSITY_MIN_MULTIPLIER * typical,
                     typical + _DENSITY_MAD_MULTIPLIER * 1.4826 * mad)
        density_ok[local] = np.isfinite(radii) & (radii <= cutoff)
        scales[local] = np.maximum(radii, typical * .25)
        full_indices = indices[local]
        other = full_indices[neighbours[:, 1:]]
        usable_pairs = valid_normal[full_indices, None] & valid_normal[other]
        agreement = np.abs(np.sum(unit[full_indices, None] * unit[other], axis=2)) >= _NORMAL_AGREEMENT
        usable_count = usable_pairs.sum(axis=1)
        agreeing_count = (usable_pairs & agreement).sum(axis=1)
        normal_ok[local] = (usable_count < 3) | (2 * agreeing_count >= usable_count) | supported[full_indices]
        density_reports.append({"view": source_names.get(int(label), "unspecified"),
                                "points": len(local), "status": "neighbourhood_scale_estimated",
                                "typical_neighbour_radius_m": typical,
                                "density_radius_cutoff_m": float(cutoff),
                                "density_retained": int(density_ok[local].sum()),
                                "normal_consistent": int(normal_ok[local].sum())})
    keep = density_ok & normal_ok
    evidence["density_by_view"] = density_reports
    evidence["density_and_normal_retained_points"] = int(keep.sum())
    if keep.sum() < 4:
        return _empty(), {**evidence, "reason": "no_coherent_local_neighbourhood"}
    indices, cloud, scales = indices[keep], cloud[keep], scales[keep]

    k = min(_LINK_NEIGHBOURS + 1, len(cloud))
    distances, neighbours = cKDTree(cloud).query(cloud, k=k)
    distances, neighbours = distances[:, 1:], neighbours[:, 1:]
    rows = np.broadcast_to(np.arange(len(cloud))[:, None], neighbours.shape)
    local_limit = _LINK_SCALE_MULTIPLIER * np.minimum(scales[:, None], scales[neighbours])
    paired_normals = valid_normal[indices, None] & valid_normal[indices[neighbours]]
    agreement = np.abs(np.sum(unit[indices, None] * unit[indices[neighbours]], axis=2)) >= _NORMAL_AGREEMENT
    edges = (distances <= local_limit) & (~paired_normals | agreement)
    graph = coo_matrix((np.ones(int(edges.sum()), dtype=np.uint8),
                        (rows[edges], neighbours[edges])), shape=(len(cloud), len(cloud)))
    count, component = connected_components(graph, directed=False)
    # Cross-view support strengthens a component; it is not a probability and
    # does not turn its single-view remainder into cross-view-supported shape.
    weights = 1. + supported[indices].astype(float)
    masses = np.bincount(component, weights=weights, minlength=count)
    sizes = np.bincount(component, minlength=count)
    order = np.argsort(-masses, kind="stable")
    best = int(order[0])
    mass_fraction = float(masses[best] / masses.sum())
    second_ratio = float(masses[order[1]] / masses[best]) if len(order) > 1 else 0.
    reports = []
    for label in order[:8]:
        selected = component == label
        patch = cloud[selected]
        reports.append({"component": int(label), "points": int(sizes[label]),
                        "support_weighted_mass": float(masses[label]),
                        "crossview_supported_points": int(supported[indices[selected]].sum()),
                        "bounds_m": [patch.min(0).tolist(), patch.max(0).tolist()]})
    evidence.update({"component_count": int(count), "largest_components": reports,
                     "components_omitted_from_summary": max(0, int(count) - len(reports)),
                     "selected_component": best, "selected_mass_fraction": mass_fraction,
                     "second_component_mass_ratio": second_ratio})
    if sizes[best] < 4:
        return _empty(), {**evidence, "reason": "only_isolated_surface_samples"}
    if mass_fraction < _DOMINANT_MASS_FRACTION or second_ratio >= _SECOND_COMPONENT_RATIO:
        return _empty(), {**evidence, "status": "ambiguous_components",
                          "reason": "multiple_comparable_surface_components",
                          "selected_component": None}
    selected = component == best
    result = cloud[selected]
    supported_count = int(supported[indices[selected]].sum())
    evidence.update({"status": "coherent_observed_patch",
                     "reason": "dominant_density_and_normal_consistent_component",
                     "planning_point_count": len(result),
                     "removed_point_count": int(len(points) - len(result)),
                     "planning_crossview_supported_points": supported_count,
                     "planning_single_view_points": int(len(result) - supported_count),
                     "planning_centroid_m": result.mean(0).tolist(),
                     "planning_bounds_m": [result.min(0).tolist(), result.max(0).tolist()],
                     "planning_bounds_are_complete_object_extent": False})
    return result, evidence
