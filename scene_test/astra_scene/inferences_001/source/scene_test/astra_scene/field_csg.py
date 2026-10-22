"""Bounded, non-executable camera-coordinate CSG fields supplied by a model.

Primitive values are Euclidean signed distances in metres. Boolean composition
preserves the intended zero set but is NOT generally an exact Euclidean SDF.
Quaternion rotations map local coordinates to camera coordinates (xyzw order).
No images, cameras, simulator state, depth, or ground-truth data are read here.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

FRAME = "camera_cv_xright_ydown_zforward"
PRIMITIVES = frozenset(("sphere", "box", "cylinder"))
OPERATIONS = frozenset(("union", "intersection", "difference"))
QUATERNION_NORM_TOLERANCE = 1e-3
_TOP_KEYS = frozenset(("schema_version", "units", "coordinate_frame",
                      "estimated_focal_px", "nodes", "root", "uncertainty_notes"))
_NODE_KEYS = frozenset(("name", "op", "center_m", "rotation_xyzw", "size_m", "left", "right"))


def field_schema() -> dict[str, Any]:
    """Strict transport schema; cross-node and dimension rules are validated below."""
    def vector(length):
        return {"type": "array", "items": {"type": "number"},
                "minItems": length, "maxItems": length}
    nullable = lambda value: {"anyOf": [value, {"type": "null"}]}
    node = {"type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 160},
                "op": {"type": "string", "enum": sorted(PRIMITIVES | OPERATIONS)},
                "center_m": nullable(vector(3)),
                "rotation_xyzw": nullable(vector(4)),
                "size_m": nullable(vector(3)),
                "left": nullable({"type": "integer", "minimum": 0, "maximum": 63}),
                "right": nullable({"type": "integer", "minimum": 0, "maximum": 63})},
            "required": sorted(_NODE_KEYS)}
    return {"type": "object", "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "integer", "enum": [1]},
                "units": {"type": "string", "enum": ["m"]},
                "coordinate_frame": {"type": "string", "enum": [FRAME]},
                "estimated_focal_px": {"type": "number", "minimum": 1., "maximum": 10000.},
                "nodes": {"type": "array", "items": node, "minItems": 1, "maxItems": 64},
                "root": {"type": "integer", "minimum": 0, "maximum": 63},
                "uncertainty_notes": {"type": "string", "maxLength": 8000}},
            "required": sorted(_TOP_KEYS)}


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: expected a finite number")
    return float(value)


def _vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where}: expected {length} numbers")
    return [_number(item, where) for item in value]


def _index(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: expected an integer index")
    return value


def rotation_matrix(quaternion):
    """Local-to-camera proper rotation for a validated xyzw unit quaternion."""
    x, y, z, w = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float64)


def validate_field(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate model JSON and return a copied canonical specification.

    Norm errors <= 1e-3 are normalized and explicitly logged under _validation.
    Larger errors are rejected. Revalidating canonical JSON is supported and
    preserves its normalization history. The transport schema excludes this
    local validation record; the model never generates it.
    """
    if not isinstance(spec, dict) or set(spec) - {"_validation"} != _TOP_KEYS:
        raise ValueError("field has missing or unexpected keys")
    result = copy.deepcopy(spec)
    history = []
    if "_validation" in result:
        record = result.pop("_validation")
        if not isinstance(record, dict) or set(record) != {"normalizations"} or not isinstance(record["normalizations"], list):
            raise ValueError("invalid local validation record")
        for item in record["normalizations"]:
            if not isinstance(item, dict) or set(item) != {"node", "original_norm", "action"}:
                raise ValueError("invalid quaternion normalization record")
            _index(item["node"], "normalization node")
            norm = _number(item["original_norm"], "original quaternion norm")
            if abs(norm - 1) > QUATERNION_NORM_TOLERANCE or item["action"] != "normalize_xyzw":
                raise ValueError("invalid quaternion normalization history")
            history.append(item)
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if result["units"] != "m" or result["coordinate_frame"] != FRAME:
        raise ValueError("only camera CV coordinates in metres are supported")
    focal = _number(result["estimated_focal_px"], "estimated_focal_px")
    if not 1 <= focal <= 10000:
        raise ValueError("estimated focal length outside generic 1..10000 px limit")
    result["estimated_focal_px"] = focal
    if not isinstance(result["uncertainty_notes"], str) or len(result["uncertainty_notes"]) > 8000:
        raise ValueError("uncertainty_notes must be a string of at most 8000 characters")
    nodes = result["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 64:
        raise ValueError("expected 1..64 nodes")
    primitive_count = operation_count = 0
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != _NODE_KEYS:
            raise ValueError(f"node {i}: missing or unexpected keys")
        if not isinstance(node["name"], str) or not 1 <= len(node["name"]) <= 160:
            raise ValueError(f"node {i}: invalid name")
        op = node["op"]
        if not isinstance(op, str):
            raise ValueError(f"node {i}: operation must be a string")
        if op in PRIMITIVES:
            primitive_count += 1
            if node["left"] is not None or node["right"] is not None:
                raise ValueError(f"primitive {i}: children must be null")
            center = _vector(node["center_m"], 3, f"node {i} center")
            size = _vector(node["size_m"], 3, f"node {i} size")
            q = np.asarray(_vector(node["rotation_xyzw"], 4, f"node {i} quaternion"))
            if max(map(abs, center)) > 10:
                raise ValueError(f"node {i}: center exceeds generic +/-10 m safety bounds")
            active = 3 if op == "box" else 1 if op == "sphere" else 2
            if any(not 1e-5 <= s <= 10 for s in size[:active]) or any(s != 0 for s in size[active:]):
                raise ValueError(f"node {i}: active dimensions must be 1e-5..10 m; unused dimensions must be zero")
            norm = float(np.linalg.norm(q))
            if abs(norm - 1) > QUATERNION_NORM_TOLERANCE:
                raise ValueError(f"node {i}: quaternion norm error exceeds 1e-3")
            if abs(norm - 1) > 8*np.finfo(float).eps:
                history.append({"node": i, "original_norm": norm, "action": "normalize_xyzw"})
                q = q/norm
            node.update(center_m=center, size_m=size, rotation_xyzw=q.tolist())
        elif op in OPERATIONS:
            operation_count += 1
            if any(node[key] is not None for key in ("center_m", "rotation_xyzw", "size_m")):
                raise ValueError(f"operation {i}: geometric arrays must be null")
            for key in ("left", "right"):
                child = _index(node[key], f"node {i} {key}")
                if not 0 <= child < i:
                    raise ValueError(f"node {i}: children must reference prior nodes")
        else:
            raise ValueError(f"node {i}: unsupported operation")
    if primitive_count > 32 or operation_count > 32:
        raise ValueError("at most 32 primitives and 32 operations are allowed")
    if _index(result["root"], "root") != len(nodes)-1:
        raise ValueError("root must be the last positional node index")
    reached, stack = set(), [result["root"]]
    while stack:
        index = stack.pop()
        if index in reached:
            continue
        reached.add(index)
        node = nodes[index]
        if node["op"] in OPERATIONS:
            stack.extend((node["left"], node["right"]))
    if len(reached) != len(nodes):
        raise ValueError("every node must be reachable from root")
    if any(not 0 <= item["node"] < len(nodes) or nodes[item["node"]]["op"] not in PRIMITIVES for item in history):
        raise ValueError("normalization record points to a non-primitive node")
    result["_validation"] = {"normalizations": history}
    lower, upper = _bounds(result)
    if np.any(upper <= lower):
        raise ValueError("root has an empty or zero-volume bounding box")
    return result


def _canonical(spec):
    # Public entry points deliberately validate even previously canonical JSON.
    # This is tiny compared with voxel or ray queries and avoids trusting a marker.
    return validate_field(spec)


def evaluate_field(spec, points):
    """Evaluate scalar CSG values at Nx3 camera-coordinate points, in metres."""
    spec = _canonical(spec)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx3")
    values = []
    for node in spec["nodes"]:
        op = node["op"]
        if op in PRIMITIVES:
            local = (points - node["center_m"]) @ rotation_matrix(node["rotation_xyzw"])
            size = node["size_m"]
            if op == "sphere":
                value = np.linalg.norm(local, axis=1) - size[0]
            elif op == "box":
                delta = np.abs(local)-size
                value = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            else:
                delta = np.column_stack((np.linalg.norm(local[:, :2], axis=1)-size[0], np.abs(local[:, 2])-size[1]))
                value = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
        else:
            a, b = values[node["left"]], values[node["right"]]
            value = np.minimum(a, b) if op == "union" else np.maximum(a, b if op == "intersection" else -b)
        values.append(value)
    return values[spec["root"]]


def _bounds(spec):
    bounds = []
    for node in spec["nodes"]:
        op = node["op"]
        if op in PRIMITIVES:
            center = np.asarray(node["center_m"])
            rotation = rotation_matrix(node["rotation_xyzw"])
            size = node["size_m"]
            if op == "sphere":
                extent = np.full(3, size[0])
            elif op == "box":
                extent = np.abs(rotation) @ np.asarray(size)
            else:
                extent = size[0]*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+size[1]*np.abs(rotation[:, 2])
            bounds.append((center-extent, center+extent))
        else:
            a, b = bounds[node["left"]], bounds[node["right"]]
            if op == "union":
                bounds.append((np.minimum(a[0], b[0]), np.maximum(a[1], b[1])))
            elif op == "intersection":
                bounds.append((np.maximum(a[0], b[0]), np.minimum(a[1], b[1])))
            else:
                bounds.append(a)
    return bounds[spec["root"]]


def field_bounds(spec):
    """Conservative finite root AABB, without any grid margin or snapping."""
    return tuple(array.copy() for array in _bounds(_canonical(spec)))


def _quadratic_interval(origin, direction, radius):
    a = np.einsum("ij,ij->i", direction, direction)
    b = np.einsum("ij,ij->i", origin, direction)
    c = np.einsum("ij,ij->i", origin, origin) - radius*radius
    lower, upper = np.full(len(a), np.inf), np.full(len(a), -np.inf)
    tolerance = 64*np.finfo(float).eps*np.maximum(np.einsum("ij,ij->i", origin, origin)+radius*radius, 1)
    parallel = a == 0
    inside_parallel = parallel & (c <= tolerance)
    lower[inside_parallel], upper[inside_parallel] = -np.inf, np.inf
    boundary_parallel = parallel & (np.abs(c) <= tolerance)
    discriminant = b*b-a*c
    discriminant_tolerance = 64*np.finfo(float).eps*np.maximum(b*b+np.abs(a*c), np.finfo(float).tiny)
    valid = ~parallel & (discriminant >= -discriminant_tolerance)
    if valid.any():
        av, bv, cv = a[valid], b[valid], c[valid]
        sqrt_disc = np.sqrt(np.maximum(discriminant[valid], 0))
        q = -bv-np.copysign(sqrt_disc, bv)
        first = np.divide(q, av)
        second = np.divide(cv, q, out=-bv/av, where=q != 0)
        lower[valid], upper[valid] = np.minimum(first, second), np.maximum(first, second)
    return lower, upper, boundary_parallel


def _slab_interval(origin, direction, half_size):
    lower, upper = np.full(len(origin), -np.inf), np.full(len(origin), np.inf)
    wall = np.zeros(len(origin), dtype=bool)
    for axis, half in enumerate(half_size):
        o, d = origin[:, axis], direction[:, axis]
        parallel = d == 0
        tolerance = 64*np.finfo(float).eps*np.maximum(np.maximum(np.abs(o), half), 1)
        outside = parallel & (np.abs(o) > half+tolerance)
        wall |= parallel & (np.abs(np.abs(o)-half) <= tolerance)
        a = np.divide(-half-o, d, out=np.full(len(o), -np.inf), where=~parallel)
        b = np.divide(half-o, d, out=np.full(len(o), np.inf), where=~parallel)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        lo[outside], hi[outside] = np.inf, -np.inf
        lower, upper = np.maximum(lower, lo), np.minimum(upper, hi)
    return lower, upper, wall


def _primitive_interval(node, origins, directions):
    rotation = rotation_matrix(node["rotation_xyzw"])
    origin = (origins-node["center_m"]) @ rotation
    direction = directions @ rotation
    size, op = node["size_m"], node["op"]
    if op == "sphere":
        return _quadratic_interval(origin, direction, size[0])
    if op == "box":
        return _slab_interval(origin, direction, size)
    radial = _quadratic_interval(origin[:, :2], direction[:, :2], size[0])
    axial = _slab_interval(origin[:, 2:], direction[:, 2:], [size[1]])
    return np.maximum(radial[0], axial[0]), np.minimum(radial[1], axial[1]), radial[2] | axial[2]


def analytic_first_hits(spec, origins, directions):
    """Exact first positive zero-set boundary from primitive analytic intervals.

    Returned t is the ray parameter in origin + t*direction (not necessarily
    Euclidean range). Misses are NaN. CSG interval membership is composed with
    signed Boolean min/max, retaining tangent boundaries rather than losing
    them through ordinary closed-interval coalescing. No sphere tracing, sampled
    field, or decoded mesh is used. Boundary-coincident rays return the first
    positive finite primitive endpoint; the uncountable interior zeros have no
    mathematically smallest positive t.
    """
    spec = _canonical(spec)
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must have identical Nx3 shapes")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=1) == 0):
        raise ValueError("rays must be finite and directions nonzero")
    result = np.full(len(origins), np.nan)
    nodes = spec["nodes"]
    for start in range(0, len(origins), 4096):
        stop = min(start+4096, len(origins))
        origin, direction = origins[start:stop], directions[start:stop]
        intervals = {i: _primitive_interval(node, origin, direction)
                     for i, node in enumerate(nodes) if node["op"] in PRIMITIVES}
        candidates = np.stack([endpoint for interval in intervals.values() for endpoint in interval[:2]], axis=1)
        candidate_valid = np.isfinite(candidates) & (candidates > 0)
        safe_candidates = np.where(candidate_valid, candidates, 0)
        values = []
        for i, node in enumerate(nodes):
            op = node["op"]
            if op in PRIMITIVES:
                lower, upper, wall = intervals[i]
                lower, upper = lower[:, None], upper[:, None]
                valid = lower <= upper
                scale = np.maximum(np.abs(safe_candidates), 1)
                # Parameter tolerance compensates floating-point endpoint error,
                # scaled so direction magnitude does not change physical tolerance.
                tolerance = 128*np.finfo(float).eps*scale
                inside = valid & (safe_candidates > lower+tolerance) & (safe_candidates < upper-tolerance)
                boundary = valid & (((np.abs(safe_candidates-lower) <= tolerance) |
                                     (np.abs(safe_candidates-upper) <= tolerance)) |
                                    (wall[:, None] & (safe_candidates >= lower-tolerance) & (safe_candidates <= upper+tolerance)))
                value = np.ones(candidates.shape, dtype=np.int8)
                value[inside] = -1
                value[boundary] = 0
            else:
                a, b = values[node["left"]], values[node["right"]]
                value = np.minimum(a, b) if op == "union" else np.maximum(a, b if op == "intersection" else -b)
            values.append(value)
        admissible = candidate_valid & (values[spec["root"]] == 0)
        hits = np.min(np.where(admissible, candidates, np.inf), axis=1)
        result[start:stop] = np.where(np.isfinite(hits), hits, np.nan)
    return result
