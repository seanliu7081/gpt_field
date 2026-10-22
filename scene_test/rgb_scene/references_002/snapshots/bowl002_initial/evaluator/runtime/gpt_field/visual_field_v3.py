"""Opt-in feasibility-first visual fields, with separately represented uncertainty.

All costs and exclusions are image-space hypotheses. No depth, camera calibration,
privileged object state, or certified scene clearance is available to this module.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .visual_field import ImageDistanceField, VIEWS, _clear_segment, _gray, _uv, _vector
from .visual_field_v2 import FeatureBankTrack, VisualFieldControllerV2, _response

VISUAL_FIELD_VERSION = "feasibility-local-image-distance-v3"
STRUCTURAL_CODES = frozenset({"footprint_out_of_view", "footprint_missing_moving_point",
    "start_obstacle_conflict", "start_unknown_conflict", "goal_obstacle_conflict",
    "goal_unknown_conflict", "goal_out_of_view", "self_occlusion_outside_footprint"})


def _bounds(value, *, outside=False):
    if not isinstance(value, dict) or not {"lower_uv", "upper_uv"} <= set(value):
        raise ValueError("region requires lower_uv and upper_uv")
    low, high = [_vector(value[k], 2, k) for k in ("lower_uv", "upper_uv")]
    if np.any(low >= high) or (not outside and (np.any(low < 0) or np.any(high > 1))):
        raise ValueError("region bounds must be ordered, and scene boxes inside [0,1]")
    return low, high


def _inside(point, low, high):
    return bool(np.all(point >= low-1e-12) and np.all(point <= high+1e-12))


def _block(code, ident, evidence, remedy):
    return {"code": code, "category": code, "region_id": ident, "feature_id": ident,
            "evidence": evidence, "remedy": remedy}


def _canonical(ref):
    required = {"moving_id", "target_id", "moving_point_uv", "target_anchor_uv", "target_point_uv",
                "moving_footprint_uv", "blocked_boxes_uv", "unknown_boxes_uv"}
    if not isinstance(ref, dict) or not required <= set(ref) or set(ref)-required-{"moving_kind", "tool_point_uv"}:
        raise ValueError("V3 reference requires the canonical footprint, identities, landmarks and typed regions")
    for key in ("moving_id", "target_id"):
        VisualFieldControllerV2._identity(ref[key])
    if ref["moving_id"] == ref["target_id"] or ref.get("moving_kind", "tool") not in ("tool", "payload"):
        raise ValueError("invalid moving/target identity or kind")
    for key in ("moving_point_uv", "target_anchor_uv", "target_point_uv"):
        _uv(ref[key], key)
    if ref.get("tool_point_uv") is not None:
        _uv(ref["tool_point_uv"], "tool_point_uv")
    if set(ref["moving_footprint_uv"]) != {"lower_uv", "upper_uv"}:
        raise ValueError("footprint requires only lower_uv and upper_uv")
    _bounds(ref["moving_footprint_uv"], outside=True)
    seen = set()
    for group, label, allowed in (("blocked_boxes_uv", "role", {"obstacle", "contact_target"}),
                                  ("unknown_boxes_uv", "source", {"occluded", "out_of_view", "self_occlusion"})):
        if not isinstance(ref[group], (list, tuple)) or len(ref[group]) > 8:
            raise ValueError("at most eight regions per category")
        for region in ref[group]:
            if not isinstance(region, dict) or set(region) != {"object_id", "lower_uv", "upper_uv", label}:
                raise ValueError("regions require identity, bounds and explicit role/source")
            ident = VisualFieldControllerV2._identity(region["object_id"])
            if ident in seen or region[label] not in allowed:
                raise ValueError("region identities must be unique and roles/sources explicit")
            seen.add(ident)
            _bounds(region)


def preflight_reference(ref, image_shape, grid_size=40, *, _validate=True):
    """Pure semantic/coverage checks; no texture, physics or motion samples needed.

    The footprint uses absolute image coordinates. Its offsets from the moving
    point give an asymmetric Minkowski exclusion [box.low-offset.high-margin,
    box.high-offset.low+margin]. No footprint or region is clipped or shrunk.
    """
    if type(grid_size) is not int or not 8 <= grid_size <= 96:
        raise ValueError("grid_size must be an integer in [8,96]")
    if len(image_shape) < 2 or min(image_shape[:2]) < 2:
        raise ValueError("image_shape must contain height and width")
    if ref is None:
        return {"available": False, "blockers": [], "semantic_repair_required": False,
                "start_valid": None, "goal_valid": None, "terminal_contact_regions": []}
    if _validate:
        _canonical(ref)
    moving, goal = np.asarray(ref["moving_point_uv"]), np.asarray(ref["target_point_uv"])
    low, high = _bounds(ref["moving_footprint_uv"], outside=True)
    offset_low, offset_high, margin = low-moving, high-moving, 1/(grid_size-1)
    blockers, contacts = [], []
    if np.any(low < 0) or np.any(high > 1):
        blockers.append(_block("footprint_out_of_view", ref["moving_id"],
            {"lower_uv": low.tolist(), "upper_uv": high.tolist()}, "reobserve the full moving footprint; do not clip its extent"))
    if not _inside(moving, low, high):
        blockers.append(_block("footprint_missing_moving_point", ref["moving_id"], {}, "correct the landmark or observed footprint"))
    if np.any(goal+offset_low < 0) or np.any(goal+offset_high > 1):
        blockers.append(_block("goal_out_of_view", ref["target_id"],
            {"goal_footprint_lower_uv": (goal+offset_low).tolist(), "goal_footprint_upper_uv": (goal+offset_high).tolist()},
            "choose an observable intermediate region or reobserve; do not shrink the footprint"))
    self_visibility = []
    for group in ("blocked_boxes_uv", "unknown_boxes_uv"):
        for region in ref[group]:
            a, b = _bounds(region, outside=not _validate)
            ident = region["object_id"]
            if region.get("source") == "self_occlusion":
                if np.any(a < low-margin) or np.any(b > high+margin):
                    blockers.append(_block("self_occlusion_outside_footprint", ident,
                        {"region_lower_uv": a.tolist(), "region_upper_uv": b.tolist()},
                        "split observed body self-occlusion from the unobserved external area; outside area remains unknown"))
                else:
                    self_visibility.append(ident)
                continue
            forbidden_low, forbidden_high = a-offset_high-margin, b-offset_low+margin
            if region.get("role") == "contact_target":
                if _inside(goal, forbidden_low, forbidden_high):
                    contacts.append(ident)
                if _inside(moving, forbidden_low, forbidden_high):
                    blockers.append(_block("start_contact_region", ident, {}, "hand off intended contact to Direct; no local body-contact permission"))
                continue
            category = "unknown" if group == "unknown_boxes_uv" else "obstacle"
            for endpoint, point in (("start", moving), ("goal", goal)):
                if _inside(point, forbidden_low, forbidden_high):
                    blockers.append(_block(endpoint+"_"+category+"_conflict", ident,
                        {"point_uv": point.tolist(), "forbidden_lower_uv": forbidden_low.tolist(), "forbidden_upper_uv": forbidden_high.tolist()},
                        "correct the region role/extent, footprint or intermediate goal using new visual evidence; additional calibration cannot repair this contradiction"))
    return {"available": True, "blockers": blockers,
        "semantic_repair_required": any(b["code"] in STRUCTURAL_CODES for b in blockers),
        "start_valid": not any(b["code"].startswith("start_") or b["code"].startswith("footprint_") or b["code"] == "self_occlusion_outside_footprint" for b in blockers),
        "goal_valid": not any(b["code"].startswith("goal_") or b["code"] == "self_occlusion_outside_footprint" for b in blockers),
        "terminal_contact_regions": contacts, "body_self_occlusion_regions": self_visibility,
        "footprint_offsets_uv": {"lower_uv": offset_low.tolist(), "upper_uv": offset_high.tolist()},
        "grid_margin_uv": margin, "clearance_certified": False}


class _FootprintField(ImageDistanceField):
    """Same 8-neighbor solve, with asymmetric footprints and unclipped masks."""
    def __init__(self, shape, goal, offsets, boxes, size):
        self.size, self.margin_uv = size, 0.
        self.scale = np.array([shape[1]-1, shape[0]-1], float)
        self.target_uv = np.asarray(goal, float)
        self.offset_low, self.offset_high = [np.asarray(v, float) for v in offsets]
        self.boxes, self.unknown_boxes = boxes, []
        xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        self.points = np.stack((xx, yy), axis=-1)
        self.valid = np.all((self.points+self.offset_low >= 0) & (self.points+self.offset_high <= 1), axis=-1)
        self.unknown, self.risk = ~self.valid.copy(), np.zeros((size, size))
        for low, high in boxes:
            self.valid &= ~np.all((self.points >= low) & (self.points <= high), axis=-1)
            separation = np.maximum(np.maximum(low-self.points, self.points-high), 0.)
            self.risk += 2*np.exp(-np.sum(separation**2, axis=-1)/(2*(2/(size-1))**2))
        self.distance = np.full((size, size), np.inf)
        self.successor = np.full((size, size, 2), -1, int)
        self._solve()

    def segment_valid(self, start_uv, end_uv):
        points = np.asarray([start_uv, end_uv], float)
        return bool(np.isfinite(points).all() and np.all(points+self.offset_low >= -1e-12)
                    and np.all(points+self.offset_high <= 1+1e-12)
                    and _clear_segment(points[0], points[1], self.boxes))


class VisualFieldControllerV3(VisualFieldControllerV2):
    """Independent per-view observations; candidate-local risk and probe checks."""
    def __init__(self, *, grid_size=40, joint_lower=None, joint_upper=None):
        super().__init__(grid_size=grid_size, joint_lower=joint_lower, joint_upper=joint_upper)
        self._next_epoch = 0

    def _epoch(self):
        self._next_epoch += 1
        return self._next_epoch

    def _view_features(self, view):
        tracks = view.get("tracks", {})
        if not view.get("provided") or not all(k in tracks and tracks[k].reliable for k in ("moving", "target")):
            return None
        return tracks["moving"].position-self._goal(view)

    def _make_view(self, pixels, ref, old, state):
        if ref is None:
            if old and old.get("ref") is not None:
                old["provided"] = False
                old["status"] = "reference_unavailable_retained_constraints"
                old["reliable"] = False
                return old
            return {"status": "reference_unavailable", "provided": False, "reliable": False,
                    "geometry_reliable": False, "shape": pixels.shape, "tracks": {}, "boxes": [], "ref": None,
                    "tool": None, "preflight": preflight_reference(None, pixels.shape), "field": None,
                    "landmark_epoch": self._epoch(), "goal_epoch": self._epoch(), "model_epoch": self._epoch(),
                    "feature_model": _response("image"), "anchor_model": _response("image"),
                    "projection_state": state.copy(), "missing_age": 0}
        _canonical(ref)
        result = {"status": "tracking", "provided": True, "shape": pixels.shape, "ref": copy.deepcopy(ref),
            "tracks": {}, "tool": None, "boxes": [], "seed_ids": (ref["moving_id"], ref["target_id"]),
            "moving_kind": ref.get("moving_kind", "tool"), "missing_age": 0, "field": None,
            "field_key": None, "goal_field": None, "goal_key": None, "projection_state": state.copy()}
        scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], float)
        moving = np.asarray(ref["moving_point_uv"])
        result["goal_offset"] = (np.asarray(ref["target_point_uv"])-ref["target_anchor_uv"])*scale
        low, high = _bounds(ref["moving_footprint_uv"], outside=True)
        result["offsets"] = low-moving, high-moving
        for key, ident, point in (("moving", ref["moving_id"], moving), ("target", ref["target_id"], ref["target_anchor_uv"])):
            previous = (old or {}).get("tracks", {}).get(key) if (old or {}).get("shape") == pixels.shape else None
            try:
                result["tracks"][key] = self._track(pixels, point, ident, previous)
            except ValueError:
                pass
        if ref.get("tool_point_uv") is not None:
            try:
                tool_uv = np.asarray(ref["tool_point_uv"])
                if np.linalg.norm((tool_uv-moving)*scale) >= 10:
                    result["tool"] = self._track(pixels, tool_uv, "independent_tool", (old or {}).get("tool"))
            except ValueError:
                pass
        same = bool(old and len(result["tracks"]) == 2 and all(result["tracks"].get(k) is old.get("tracks", {}).get(k) for k in ("moving", "target")))
        same_pose = bool(old and np.linalg.norm(state[3:6]-old["projection_state"][3:6]) <= .12
                         and max(abs(state[6:]-old["projection_state"][6:])) <= .004)
        result["landmark_epoch"] = old["landmark_epoch"] if same else self._epoch()
        result["goal_epoch"] = old["goal_epoch"] if same and np.allclose(result["goal_offset"], old["goal_offset"], atol=1e-8) else self._epoch()
        preserve = same and same_pose
        result["model_epoch"] = old["model_epoch"] if preserve else self._epoch()
        result["feature_model"] = old["feature_model"] if preserve else _response("image")
        result["anchor_model"] = old["anchor_model"] if preserve else _response("image")
        if preserve:
            result["projection_state"] = old["projection_state"]
        previous_regions = {b["id"]: b for b in (old or {}).get("boxes", [])}
        present = set()
        for group in ("blocked_boxes_uv", "unknown_boxes_uv"):
            for raw in ref[group]:
                ident = raw["object_id"]; present.add(ident)
                previous = previous_regions.get(ident)
                low, high = _bounds(raw)
                patch = None
                if group == "blocked_boxes_uv":
                    try:
                        patch = self._track(pixels, (low+high)/2, ident,
                                            previous.get("patch") if previous else None, bounds=(low, high))
                    except ValueError:
                        pass
                response = previous["response"] if preserve and previous and patch is previous.get("patch") else _response("image")
                result["boxes"].append({"id": ident, "kind": "blocked" if group == "blocked_boxes_uv" else "unknown",
                    "role": raw.get("role"), "source": raw.get("source"), "low": low, "high": high,
                    "patch": patch, "anchor": None if patch is None else patch.position.copy(),
                    "response": response, "age": 0, "growth_px": 0., "omitted": False,
                    "self_offsets": (low-moving, high-moving) if raw.get("source") == "self_occlusion" else None})
        # An omitted region is not newly observed free space. Retain its last
        # coordinates until the same explicit identity is revised by fresh RGB.
        for ident, previous in previous_regions.items():
            if ident not in present:
                carried = copy.copy(previous)
                carried["omitted"] = True
                result["boxes"].append(carried)
        result["reliable"] = self._view_features(result) is not None
        self._build_view(result)
        return result

    def _point(self, view, key):
        track = view.get("tracks", {}).get(key)
        if track is not None:
            return track.uv.copy()
        return np.asarray(view["ref"]["moving_point_uv" if key == "moving" else "target_anchor_uv"], float)

    def _region_bounds(self, view, region, *, uncertainty=True):
        if region.get("self_offsets") is not None:
            point = self._point(view, "moving")
            low, high = [point+offset for offset in region["self_offsets"]]
        else:
            shift = np.zeros(2)
            patch = region.get("patch")
            if patch is not None:
                shift = (patch.position-region["anchor"])/patch.scale
            low, high = region["low"]+shift, region["high"]+shift
        if uncertainty and region["source"] != "self_occlusion":
            scale = np.array([view["shape"][1]-1, view["shape"][0]-1])
            growth = region["growth_px"]/scale
            low, high = low-growth, high+growth
        return low, high

    def _forbidden(self, view, region, *, uncertainty=True):
        low, high = self._region_bounds(view, region, uncertainty=uncertainty)
        flo, fhi = view["offsets"]
        margin = 1/(self.grid_size-1)
        return low-fhi-margin, high-flo+margin

    def _current_reference(self, view):
        ref = copy.deepcopy(view["ref"])
        point = self._point(view, "moving")
        anchor = self._point(view, "target")
        scale = np.array([view["shape"][1]-1, view["shape"][0]-1])
        ref.update(moving_point_uv=point.tolist(), target_anchor_uv=anchor.tolist(),
                   target_point_uv=(anchor+view["goal_offset"]/scale).tolist(),
                   moving_footprint_uv={"lower_uv": (point+view["offsets"][0]).tolist(),
                                        "upper_uv": (point+view["offsets"][1]).tolist()},
                   blocked_boxes_uv=[], unknown_boxes_uv=[])
        for region in view["boxes"]:
            low, high = self._region_bounds(view, region, uncertainty=False)
            item = {"object_id": region["id"], "lower_uv": low.tolist(), "upper_uv": high.tolist()}
            if region["kind"] == "blocked":
                item["role"] = region["role"]; ref["blocked_boxes_uv"].append(item)
            else:
                item["source"] = region["source"]; ref["unknown_boxes_uv"].append(item)
        return ref

    def _build_view(self, view):
        if view.get("ref") is None:
            return
        ref = self._current_reference(view)
        view["preflight"] = preflight_reference(ref, view["shape"], self.grid_size, _validate=False)
        point, goal = np.asarray(ref["moving_point_uv"]), np.asarray(ref["target_point_uv"])
        goal_key = tuple(goal)
        if goal_key != view.get("goal_key"):
            view["goal_field"] = _FootprintField(view["shape"], goal, (np.zeros(2), np.zeros(2)), [], self.grid_size)
            view["goal_key"] = goal_key
        # Goal distance remains a separate diagnostic even if risk exclusions
        # make a requested route infeasible. No infeasible route is executable.
        risk = [self._forbidden(view, b) for b in view["boxes"] if b["source"] != "self_occlusion" and b["role"] != "contact_target"]
        key = (goal_key, tuple(tuple(x) for x in view["offsets"]), tuple((tuple(a), tuple(b)) for a, b in risk))
        if view["preflight"]["semantic_repair_required"]:
            view["field"] = None
        elif key != view.get("field_key") or view.get("field") is None:
            view["field"] = _FootprintField(view["shape"], goal, view["offsets"], risk, self.grid_size)
            view["field_key"] = key
        view["geometry_reliable"] = not view["preflight"]["semantic_repair_required"]
        if not view.get("reliable"):
            view["status"] = "reference_unavailable"
        elif view["preflight"]["semantic_repair_required"]:
            view["status"] = "semantic_correction_required"
        else:
            view["status"] = "candidate_checks_required"

    def configure(self, images, seeds, q, *, state8):
        if not isinstance(images, dict) or set(images) != set(VIEWS) or not isinstance(seeds, dict) or set(seeds) != set(VIEWS):
            raise ValueError("images and seeds must have exactly external and wrist keys")
        joints, state = self._joints(q), _vector(state8, 8, "state8")
        previous = self.views
        self.views = {name: self._make_view(_gray(images[name]), seeds[name], previous.get(name), state) for name in VIEWS}
        changed = not all(self.views[v].get("tracks", {}).get("moving") is previous.get(v, {}).get("tracks", {}).get("moving")
                          and self.views[v].get("tool") is previous.get(v, {}).get("tool") for v in VIEWS)
        if changed:
            self._attachment = []; self._attachment_epoch += 1
        self.q, self.state8 = joints, state.copy()
        self._epochs += 1; self._status = "configured"
        return self.summary()

    def observe(self, images, q, *, state8, executed_action):
        if self.q is None:
            raise RuntimeError("configure before observe")
        if not isinstance(images, dict) or set(images) != set(VIEWS):
            raise ValueError("images must have exactly external and wrist keys")
        joints, state, action = self._joints(q), _vector(state8, 8, "state8"), _vector(executed_action, 7, "executed_action")
        if np.any(abs(action) > 1):
            raise ValueError("actions must be normalized")
        delta, dq = state-self.state8, joints-self.q
        effective_gripper = float(action[6]) if abs(action[6]) > 1e-8 else self._last_gripper
        previous_pairs = self._pairs()
        before_moving = {v: self._point(view, "moving").copy() for v, view in self.views.items() if view.get("ref") is not None}
        attachment_lost = False
        for name, view in self.views.items():
            pixels = _gray(images[name])
            if pixels.shape != view["shape"]:
                raise ValueError("image dimensions changed")
            before = self._view_features(view)
            before_anchor = view.get("tracks", {}).get("target")
            before_anchor = None if before_anchor is None else before_anchor.position.copy()
            before_regions = {b["id"]: b["patch"].position.copy() for b in view["boxes"] if b.get("patch") is not None and b["patch"].reliable}
            for key, track in list(view.get("tracks", {}).items())+[("tool", view.get("tool"))]:
                if track is not None:
                    was_reliable = track.reliable
                    track.track(pixels)
                    attachment_lost |= bool(was_reliable and not track.reliable and key in ("moving", "tool"))
            view["reliable"] = self._view_features(view) is not None
            view["missing_age"] = 0 if view["reliable"] else view.get("missing_age", 0)+1
            for region in view["boxes"]:
                region["age"] += 1
                patch = region.get("patch")
                tracked = patch is not None and patch.track(pixels)
                # Untracked/visibility regions remain present and grow by one
                # pixel per observation. This is a cautious image heuristic,
                # not a statistical or physical upper bound on scene motion.
                if not tracked and region["source"] != "self_occlusion":
                    current_anchor = view.get("tracks", {}).get("target")
                    drift = (float(np.linalg.norm(current_anchor.position-before_anchor))
                             if current_anchor is not None and current_anchor.reliable and before_anchor is not None else 0.)
                    region["growth_px"] += 1.+drift
            after = self._view_features(view)
            stable = (np.linalg.norm(action[3:6]) <= .08 and np.linalg.norm(delta[3:6]) <= .04
                and np.linalg.norm(state[3:6]-view["projection_state"][3:6]) <= .12
                and max(abs(delta[6:])) <= .004
                and (self._last_gripper is None or effective_gripper is None
                     or (effective_gripper >= 0) == (self._last_gripper >= 0)))
            if not stable:
                view["feature_model"], view["anchor_model"] = _response("image"), _response("image")
                view["model_epoch"] = self._epoch(); view["projection_state"] = state.copy()
                for region in view["boxes"]:
                    region["response"] = _response("image")
            elif before is not None and after is not None:
                view["feature_model"].add(delta[:3], after-before)
                anchor_delta = view["tracks"]["target"].position-before_anchor
                view["anchor_model"].add(delta[:3], anchor_delta)
                for region in view["boxes"]:
                    patch = region.get("patch")
                    if patch is not None and patch.reliable and region["id"] in before_regions:
                        region["response"].add(delta[:3], patch.position-before_regions[region["id"]]-anchor_delta)
            self._build_view(view)
        if np.linalg.norm(action[3:6]) <= .08 and np.linalg.norm(delta[3:6]) <= .08:
            self.action_model.add(action[:3], delta[:3]); self.joint_model.add(action[:3], dq)
        if attachment_lost:
            self._attachment_epoch += 1
        pairs = self._pairs()
        if (pairs is not None and previous_pairs is not None and action[6] >= 0
                and np.linalg.norm(delta[3:6]) <= .04 and max(abs(delta[6:])) <= .004):
            displacement = max(float(np.linalg.norm((self._point(view, "moving")-before_moving[name])*
                                np.array([view["shape"][1]-1, view["shape"][0]-1]))) for name, view in self.views.items())
            self._attachment = (self._attachment+[(float(np.linalg.norm(delta[:3])), pairs.copy(), displacement)])[-5:]
        else:
            self._attachment = []
        self.motion = (self.motion+[(float(np.linalg.norm(action[:3])*.05), float(np.linalg.norm(delta[:3])))])[-5:]
        self.q, self.state8, self._last_gripper = joints, state.copy(), effective_gripper
        self._status = "observed"
        return self.summary()

    def summary(self):
        views, blockers, models = {}, [], {}
        for name, view in self.views.items():
            tracks = view.get("tracks", {})
            entry = {"status": view["status"], "reliable": bool(view.get("reliable")),
                "geometry_reliable": bool(view.get("geometry_reliable")),
                "moving_id": view.get("seed_ids", (None, None))[0], "target_id": view.get("seed_ids", (None, None))[1],
                "moving_kind": view.get("moving_kind"), "moving_track_reliable": bool(tracks.get("moving") and tracks["moving"].reliable),
                "target_anchor_track_reliable": bool(tracks.get("target") and tracks["target"].reliable),
                "tool_track_reliable": bool(view.get("tool") and view["tool"].reliable),
                "tool_point_uv": view["tool"].uv.tolist() if view.get("tool") and view["tool"].reliable else None,
                "image_shape_hw": list(view["shape"]), "image_scale_px": [view["shape"][1]-1, view["shape"][0]-1],
                "preflight": copy.deepcopy(view["preflight"]), "distance_cost_px": None, "route_cost_px": None,
                "landmark_epoch": view["landmark_epoch"], "model_epoch": view["model_epoch"], "goal_epoch": view["goal_epoch"],
                "missing_age": view.get("missing_age", 0), "constraints_retained": not view.get("provided"),
                "obstacle_track_count": len(view["boxes"])}
            for blocker in view["preflight"]["blockers"]:
                blockers.append({"view": name, **copy.deepcopy(blocker)})
            if not view.get("reliable"):
                for key, code in (("moving", "moving_reference_unavailable"), ("target", "target_reference_unavailable")):
                    if not view.get("provided") or key not in tracks or not tracks[key].reliable:
                        blockers.append({"view": name, **_block(code, entry["moving_id" if key == "moving" else "target_id"],
                            {"last_constraints_retained": view.get("ref") is not None}, "reobserve the same landmark; retain other-view learning")})
            models[name] = {"eef_to_image": view["feature_model"].summary(), "eef_to_target_anchor": view["anchor_model"].summary()}
            entry.update(models[name])
            if view.get("reliable"):
                for model_name, model_info in models[name].items():
                    if not model_info["reliable"]:
                        blockers.append({"view": name, **_block("response_unsupported", model_name,
                            {"rank": model_info["rank"], "validation_count": model_info["validation_count"]},
                            "use only a direction with empirical image support; a fresh unknown direction requires model-directed Direct measurement")})
            if view.get("ref") is not None:
                ref = self._current_reference(view)
                point, goal = np.asarray(ref["moving_point_uv"]), np.asarray(ref["target_point_uv"])
                entry.update(moving_point_uv=point.tolist(), target_anchor_uv=ref["target_anchor_uv"], target_point_uv=goal.tolist(),
                    moving_footprint_uv=ref["moving_footprint_uv"], goal_error_px=float(np.linalg.norm((point-goal)*entry["image_scale_px"])),
                    goal_descriptor={"moving_id": entry["moving_id"], "target_id": entry["target_id"], "target_offset_px": view["goal_offset"].tolist()})
                if view.get("goal_field") is not None:
                    entry["distance_cost_px"] = view["goal_field"].cost(point)
                if view.get("field") is not None:
                    entry["route_cost_px"] = view["field"].cost(point)
            entry["obstacle_tracks"] = [{"object_id": b["id"], "kind": b["kind"], "role": b["role"], "source": b["source"],
                "track_reliable": bool(b.get("patch") and b["patch"].reliable), "interior_track_required": b["kind"] == "blocked",
                "age": b["age"], "uncertainty_growth_px": b["growth_px"], "retained_after_omission": b["omitted"],
                "response_reliable": b["response"].summary()["reliable"]} for b in view["boxes"]]
            views[name] = entry
        relative_change = float(np.max(np.linalg.norm(np.array([v[1] for v in self._attachment])-self._attachment[0][1], axis=-1))) if self._attachment else None
        comotion = bool(len(self._attachment) >= 3 and sum(v[0] for v in self._attachment) >= .005
                       and sum(v[2] for v in self._attachment) >= 1 and relative_change <= 2.)
        available = [v for v in VIEWS if views.get(v, {}).get("reliable")]
        structural = [b for b in blockers if b["code"] in STRUCTURAL_CODES]
        return {"version": VISUAL_FIELD_VERSION, "status": self._status, "views": views, "view_models": models,
            "blockers": blockers, "feasibility": {"semantic_repair_required": bool(structural),
                "usable_views": available, "all_seeded_endpoints_valid": not structural,
                "candidate_checks_required": True, "clearance_certified": False},
            "tracks_reliable": bool(available), "both_views_reliable": len(available) == 2,
            "geometry_reliable": bool(available) and not structural,
            "action_to_eef": self.action_model.summary(), "action_to_joint": self.joint_model.summary(),
            "eef_to_image": {"reliable": any(m["eef_to_image"]["reliable"] for m in models.values()),
                             "sample_count": sum(m["eef_to_image"]["sample_count"] for m in models.values()), "per_view": models},
            "co_motion_evidence": comotion, "co_motion_sample_count": len(self._attachment),
            "object_tool_relative_change_px": relative_change, "attachment_epoch": self._attachment_epoch,
            "prediction_failure": None, "no_progress": bool(len(self.motion) >= 5 and sum(v[0] for v in self.motion) >= .005 and sum(v[1] for v in self.motion) <= .0005),
            "recovery": {"missing_views": [v for v in VIEWS if v not in available],
                "suggested_action": "semantic_correction_required" if structural else "resolve the named evidence blocker; preserve healthy per-view fits"},
            "clearance_certified": False, "task_success_certified": False,
            "limitations": ["image-space hypotheses, no metric scene depth or calibration", "unknown masks remain unknown",
                "untracked-mask expansion is a heuristic, not a physical motion bound", "one-view control has missing-view uncertainty",
                "no full-arm scene collision or identity/grasp/success certificate"]}

    feedback = summary

    @staticmethod
    def _predict(model, x, validated):
        if model.last_error is not None and model.last_error > model.residual_limit:
            return None
        return model.predict(x, validated=validated)

    def _candidate(self, command, *, validated, corridors=None, information=False):
        eef = self._predict(self.action_model, command, validated)
        dq = self._predict(self.joint_model, command, validated)
        if eef is None or (self.joint_lower is not None and dq is None):
            return None, "response_unsupported"
        if self.joint_lower is not None and (np.any(self.q+dq < self.joint_lower+.005) or np.any(self.q+dq > self.joint_upper-.005)):
            return None, "predicted_joint_limit"
        total, current, usable, signal = 0., 0., [], 0.
        missing = []
        for name, view in self.views.items():
            if view.get("ref") is None:
                # No last-known footprint or masks exists to bound this view.
                return None, "unobserved_view_scope"
            delta = self._predict(view["feature_model"], eef, validated)
            anchor_delta = self._predict(view["anchor_model"], eef, validated)
            if delta is None or anchor_delta is None:
                return None, "response_unsupported"
            reliable = view.get("reliable")
            if not reliable:
                missing.append(name)
                if view.get("missing_age", 0) > 3 or np.linalg.norm(command) > .05+1e-10:
                    return None, "missing_view_trust_region"
            else:
                usable.append(name)
            scale = np.array([view["shape"][1]-1, view["shape"][0]-1])
            point = self._point(view, "moving")
            end_relative = point+delta/scale
            absolute_end = end_relative+anchor_delta/scale
            flo, fhi = view["offsets"]
            # Missing-view pixels are bounded only by recent empirical response;
            # this is an explicit trust-region assumption, not certified geometry.
            uncertainty = (view.get("missing_age", 0)*(np.linalg.norm(delta)+1.))/scale if not reliable else np.zeros(2)
            path_low = np.minimum(point, absolute_end)+flo-uncertainty
            path_high = np.maximum(point, absolute_end)+fhi+uncertainty
            if np.any(path_low < 0) or np.any(path_high > 1):
                return None, "candidate_coverage_unknown"
            if corridors is not None and reliable and corridors.get(name) is None:
                return None, "probe_corridor_missing"
            if corridors is not None and corridors.get(name) is not None:
                low, high = _bounds(corridors[name])
                if np.any(path_low < low) or np.any(path_high > high):
                    return None, "probe_corridor_exceeded"
            for region in view["boxes"]:
                if region["source"] == "self_occlusion":
                    # Validated body-owned mask stays in the visibility layer.
                    # It licenses no background clearance or contact; any portion
                    # outside the observed body was rejected by preflight.
                    continue
                low, high = self._forbidden(view, region)
                drift = self._predict(region["response"], eef, validated) if region["kind"] == "blocked" else None
                if drift is not None:
                    relative_end = end_relative-drift/scale
                    padding = uncertainty
                else:
                    # Keep every mask. Near a candidate, include motion/response
                    # uncertainty; a distant untracked object is not a global veto.
                    relative_end = absolute_end
                    padding = uncertainty+(np.linalg.norm(anchor_delta)+2.)/scale
                clear = _clear_segment(point, relative_end, [(low-padding, high+padding)])
                if not clear:
                    if region["role"] == "contact_target":
                        return None, "contact_handoff_required"
                    return None, "candidate_unknown_intersection" if region["kind"] == "unknown" else "candidate_obstacle_intersection"
            if reliable:
                field = view.get("goal_field") if information else view.get("field")
                before = field.cost(point) if field is not None else None
                after = field.cost(end_relative) if field is not None else None
                if before is None or after is None:
                    return None, "route_unavailable"
                current += before; total += after
            signal += float(np.linalg.norm(delta)+np.linalg.norm(anchor_delta))
        if not usable:
            return None, "visual_reference_unavailable"
        return {"score": float(total), "current": float(current), "predicted_eef_delta_m": eef.tolist(),
            "prediction_residual_margin_m": 2*max(self.action_model.residual or 0., self.action_model.last_error or 0., .0001),
            "predicted_information_signal_px": signal, "usable_views": usable, "missing_views": missing}, None

    @staticmethod
    def _grant(value):
        if value is None:
            return None
        keys = {"blocker_codes", "directions", "max_steps", "max_displacement_m", "corridors_uv"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("observation_intent must use the explicit V3 grant schema")
        if (not isinstance(value["blocker_codes"], list) or not value["blocker_codes"]
                or any(not isinstance(v, str) or not v for v in value["blocker_codes"])):
            raise ValueError("grant requires explicit blocker codes")
        if not isinstance(value["directions"], list) or not 1 <= len(value["directions"]) <= 6:
            raise ValueError("grant requires one to six actual normalized directions")
        for direction in value["directions"]:
            norm = np.linalg.norm(_vector(direction, 3, "grant direction"))
            if not 1e-8 < norm <= .1+1e-10:
                raise ValueError("probe direction norm must be in (0,.1]")
        if type(value["max_steps"]) is not int or not 1 <= value["max_steps"] <= 10:
            raise ValueError("grant max_steps must be in [1,10]")
        limit = value["max_displacement_m"]
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0 < limit <= .02:
            raise ValueError("grant displacement must be in (0,.02]")
        corridors = value["corridors_uv"]
        if not isinstance(corridors, dict) or set(corridors) != set(VIEWS):
            raise ValueError("grant corridors must name both views")
        for corridor in corridors.values():
            if corridor is not None:
                if set(corridor) != {"lower_uv", "upper_uv"}:
                    raise ValueError("corridor requires only bounds")
                _bounds(corridor)
        return value

    def select_action(self, proposed_action, *, q=None, observation_intent=None):
        proposed = _vector(proposed_action, 7, "proposed_action")
        if np.any(abs(proposed) > 1):
            raise ValueError("actions must be normalized")
        if q is not None and (self.q is None or not np.allclose(self._joints(q), self.q, atol=1e-10, rtol=0)):
            raise ValueError("selection requires last acknowledged joints")
        grant = self._grant(observation_intent)
        info, held = self.summary(), np.zeros(7, dtype=float)

        def result(action, status, accepted=False, *, count=0, checked=None, reasons=None):
            details = {"candidate_count": count, "risk_unknown": True, "clearance_certified": False,
                "feedback": info, "rejection_reasons": reasons or {},
                "rotation_deadband_applied": bool(accepted and 0 < np.linalg.norm(proposed[3:6]) <= .02),
                "rotation_gripper_preserved": bool(np.array_equal(action[3:], proposed[3:]))}
            if checked is not None:
                details.update(checked)
            return {"action": action.tolist(), "accepted": accepted, "status": status, "horizon": 1,
                    "need_refresh": not accepted, "diagnostics": details}

        if self.q is None:
            return result(held, "unconfigured")
        if info["feasibility"]["semantic_repair_required"]:
            return result(held, "semantic_correction_required")
        if any(b["code"] == "start_contact_region" for b in info["blockers"]):
            return result(held, "contact_handoff_required")
        if self.joint_lower is not None and (np.any(self.q < self.joint_lower) or np.any(self.q > self.joint_upper)):
            return result(held, "measured_joint_limit")
        if np.linalg.norm(proposed[3:6]) > .02:
            return result(held, "rotation_requires_direct_probe")
        if not info["tracks_reliable"]:
            return result(held, "visual_reference_unavailable")
        if info["no_progress"]:
            return result(held, "no_progress")
        cap = .2 if info["both_views_reliable"] else .05
        norm = float(np.linalg.norm(proposed[:3])); magnitude = min(cap, norm)
        candidates = []
        if norm > 1e-8:
            bounded = proposed[:3]*min(1., cap/norm)
            candidates = [bounded, .5*bounded]
            candidates += [np.eye(3)[axis]*magnitude*sign for axis in range(3) for sign in (-1, 1)]
            if self.action_model.basis is not None:
                candidates += [row*magnitude*sign for row in self.action_model.basis for sign in (-1, 1)]
        unique = list({tuple(np.round(c, 10)): c for c in candidates}.values())
        best, reasons = None, {}
        for command in unique:
            checked, reason = self._candidate(command, validated=True)
            if reason:
                reasons[reason] = reasons.get(reason, 0)+1
            elif checked["score"] < checked["current"]-.05:
                objective = checked["score"]+.1*float(np.linalg.norm(command-proposed[:3]))
                if best is None or objective < best[0]:
                    best = objective, command, checked
        if best is not None:
            action = proposed.copy(); action[:3] = best[1]; action[3:6] = 0.
            return result(action, "visual_distance_descent", True, count=len(unique), checked=best[2], reasons=reasons)
        live_codes = {b["code"] for b in info["blockers"]}|set(reasons)
        # A grant must name an actually unresolved information/response blocker.
        informative = {"response_unsupported", "moving_reference_unavailable", "target_reference_unavailable",
                       "candidate_unknown_intersection", "missing_view_trust_region", "route_unavailable"}
        matches = set(grant["blocker_codes"]) & live_codes & informative if grant else set()
        if matches:
            probe_best = None
            for direction in grant["directions"]:
                command = np.asarray(direction, float)
                if np.linalg.norm(command) > min(cap, .1)+1e-10:
                    reasons["probe_trust_region_exceeded"] = reasons.get("probe_trust_region_exceeded", 0)+1
                    continue
                checked, reason = self._candidate(command, validated=False, corridors=grant["corridors_uv"], information=True)
                if reason:
                    reasons[reason] = reasons.get(reason, 0)+1
                    continue
                if not any(grant["corridors_uv"][v] is not None for v in checked["usable_views"]):
                    reasons["probe_corridor_missing"] = 1
                    continue
                movement = np.linalg.norm(checked["predicted_eef_delta_m"])+checked["prediction_residual_margin_m"]
                if movement > grant["max_displacement_m"]:
                    reasons["probe_displacement_exceeded"] = 1
                    continue
                if checked["predicted_information_signal_px"] < .25:
                    reasons["probe_no_predicted_information"] = 1
                    continue
                gain = checked["predicted_information_signal_px"]
                if probe_best is None or gain > probe_best[0]:
                    probe_best = gain, command, checked
            if probe_best is not None:
                action = np.r_[probe_best[1], 0., 0., 0., 0.]
                probe_best[2]["grant_blocker_codes"] = sorted(matches)
                probe_best[2]["model_grant_is_clearance_certificate"] = False
                return result(action, "local_information_probe", True, count=len(unique)+len(grant["directions"]), checked=probe_best[2], reasons=reasons)
        status = next((k for k in ("candidate_unknown_intersection", "contact_handoff_required",
            "candidate_obstacle_intersection", "missing_view_trust_region", "unobserved_view_scope",
            "response_unsupported", "route_unavailable", "candidate_coverage_unknown", "predicted_joint_limit",
            "probe_corridor_missing", "probe_corridor_exceeded", "probe_displacement_exceeded",
            "probe_trust_region_exceeded", "probe_no_predicted_information") if k in reasons),
            next(iter(reasons), "no_supported_progress"))
        return result(held, status, count=len(unique), reasons=reasons)
