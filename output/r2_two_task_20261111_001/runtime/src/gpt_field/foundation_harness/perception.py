"""Independent, dated SAM/DA3 perception for the integrated manipulation loop.

Candidate selection and local surface support are separate decisions. A rejected
cross-view patch does not erase another supported patch. Missing surfaces are
never completed into a solid object, and every raw depth surface remains in the
scene obstacle hypotheses. This module neither commands the robot nor reads
simulator object state, contacts, rewards or task predicates.
"""
from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..foundation_vision.bridge import frame_from_step_view
from ..foundation_vision.geometry import (
    build_geometry, _independent, _mask_at_world, _mask_on_depth, _sample,
    _project, _pixel_indices,
)
from ..foundation_vision.types import Detection, TargetSpec
from .types import ObjectGeometry, SceneSnapshot, jsonable


def _empty():
    return np.empty((0, 3), dtype=float)


def _histogram(rgb, mask):
    pixels = rgb[mask]
    if not len(pixels):
        return np.zeros(24)
    hist = np.concatenate([np.histogram(pixels[:, i], bins=8, range=(0, 256))[0]
                           for i in range(3)]).astype(float)
    return hist / max(1., hist.sum())


def _iou(first, second):
    if first.shape != second.shape:
        return 0.
    union = np.count_nonzero(first | second)
    return np.count_nonzero(first & second) / union if union else 0.


def _normals(view):
    """Camera-facing local normals; zero means no usable local tangent plane.

    Central differences only use nearby valid depth samples. A depth edge does
    not become a fabricated side wall. Normals are model estimates, not measured
    contact normals or a grasp stability certificate.
    """
    height, width = view.valid.shape
    grid = np.full((height, width, 3), np.nan)
    grid[view.rows, view.cols] = view.points_world
    normals = np.zeros_like(grid)
    if min(height, width) < 3:
        return normals[view.rows, view.cols]
    center = grid[1:-1, 1:-1]
    dx = grid[1:-1, 2:] - grid[1:-1, :-2]
    dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
    normal = np.cross(dx, dy)
    size = np.linalg.norm(normal, axis=2)
    depth = view.estimate.depth_m[1:-1, 1:-1]
    # A numerical edge filter, not a calibrated uncertainty bound.
    span = .01 + .02 * np.abs(depth)
    usable = (np.isfinite(center).all(2) & np.isfinite(normal).all(2)
              & (size > 1e-12) & (np.linalg.norm(dx, axis=2) < span)
              & (np.linalg.norm(dy, axis=2) < span))
    normal[usable] /= size[usable, None]
    normal[~usable] = 0.
    toward_camera = view.center - center
    normal[np.sum(normal * toward_camera, axis=2) < 0] *= -1
    normals[1:-1, 1:-1] = normal
    return normals[view.rows, view.cols]


def _target_records(targets):
    result = []
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError("perception targets must be objects")
        object_id, query = item.get("id"), item.get("query")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 256
               for value in (object_id, query)):
            raise ValueError("perception target id/query must be nonempty bounded strings")
        selections = item.get("selections") or {}
        if not isinstance(selections, dict) or any(
                not isinstance(view, str) or (candidate is not None and not isinstance(candidate, str))
                for view, candidate in selections.items()):
            raise ValueError("selections maps camera names to candidate IDs or null")
        regions = item.get("region_pixels") or {}
        if not isinstance(regions, dict):
            raise ValueError("region_pixels maps camera names to rectangles or null")
        parsed_regions = {}
        for name, rectangle in regions.items():
            if rectangle is None:
                continue
            values = np.asarray(rectangle, float)
            if (not isinstance(name, str) or values.shape != (4,) or not np.isfinite(values).all()
                    or np.any(values < 0) or values[0] >= values[2] or values[1] >= values[3]):
                raise ValueError("region rectangle must be finite [x0,y0,x1,y1] with positive extent")
            parsed_regions[name] = tuple(float(value) for value in values)
        parent_id = item.get("parent_id")
        if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip() or parent_id == object_id):
            raise ValueError("parent_id must name a different logical object")
        result.append({"id": object_id, "query": query.strip(),
                       "selections": {view: candidate for view, candidate in selections.items() if candidate},
                       "region_pixels": parsed_regions, "parent_id": parent_id})
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("perception target IDs must be unique")
    parents = {item["id"]: item["parent_id"] for item in result}
    for object_id in parents:
        seen, current = set(), object_id
        while current is not None:
            if current not in parents or current in seen:
                raise ValueError("parent_id relationships must form an acyclic known object hierarchy")
            seen.add(current)
            current = parents[current]
    return result


class HarnessPerception:
    """Refresh vision independently of expensive executive requests.

    ``engine`` supplies the existing resident ``segmenter``, ``depth_estimator``
    and geometry configuration; its old all-or-nothing target fusion is not
    called. Refresh cadence counts new public observation objects. Reusing an
    observation, including after a slow Astra response, preserves its original
    timestamp. A selection change can reuse that frame's model predictions.
    """

    def __init__(self, engine, output_dir=None, refresh_steps=10, max_age_s=5.):
        if type(refresh_steps) is not int or refresh_steps < 1:
            raise ValueError("refresh_steps must be a positive integer")
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.engine = engine
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.refresh_steps, self.max_age_s = refresh_steps, float(max_age_s)
        self.reset()

    def reset(self):
        self.latest = None
        self._frame = None
        self._depths = None
        self._geometry = None
        self._seen_observation = None
        self._seen_at = None
        self._observation_index = -1
        self._last_refresh_index = -1
        self._frame_id = 0
        self._queries = ()
        self._targets = ()
        self._candidates = []
        self._candidate_records = {}
        self._candidate_frames = deque(maxlen=3)
        self._archive = {}
        self._bindings = {}
        self._selection_tokens = {}
        self._regions = {}
        self._normal_maps = {}
        self._log_revision = 0
        self._timings = {}

    def update(self, view, geometry, targets, force=False):
        targets = _target_records(targets)
        queries = tuple((item["id"], item["query"]) for item in targets)
        selection_key = tuple((item["id"], item["query"], tuple(sorted(item["selections"].items())),
                               tuple(sorted(item["region_pixels"].items())), item["parent_id"])
                              for item in targets)
        observation = getattr(view, "observation", view)
        now = perf_counter()
        if observation is not self._seen_observation:
            self._seen_observation = observation
            self._seen_at = now
            self._observation_index += 1
        same_observation = self._frame is not None and self._seen_at == self._frame.captured_at_s
        old = self.latest is None or now - self.latest.captured_at_s > self.max_age_s * .5
        due = self._observation_index - self._last_refresh_index >= self.refresh_steps
        refresh = self.latest is None or force or old or due or queries != self._queries
        if not refresh and selection_key == self._targets:
            return self.latest

        started = perf_counter()
        self._timings = {"segmentation_s": 0., "depth_s": 0., "scene_geometry_s": 0.}
        try:
            if refresh and not same_observation:
                self._frame_id += 1
                self._frame = frame_from_step_view(view, geometry, self._frame_id, self._seen_at)
                self._depths = None
                self._geometry = None
                self._normal_maps.clear()
            if self._frame is None:
                raise RuntimeError("perception did not receive a public observation")
            need_segment = (not same_observation or queries != self._queries or not self._candidate_records)
            if need_segment:
                specs = [TargetSpec(item["id"], item["query"]) for item in targets]
                before = perf_counter()
                detections = tuple(self.engine.segmenter.segment(self._frame, specs))
                self._timings["segmentation_s"] = perf_counter() - before
                self._record_candidates(detections, targets)
            if self._depths is None:
                before = perf_counter()
                self._depths = self.engine.depth_estimator.estimate(self._frame)
                self._timings["depth_s"] = perf_counter() - before
            if self._geometry is None:
                before = perf_counter()
                # Preserve all scene surfaces; per-target fusion is performed below.
                self._geometry = build_geometry(self._frame, self._depths, {}, self.engine.config.geometry)
                self._normal_maps = {name: _normals(item) for name, item in self._geometry._views.items()}
                self._timings["scene_geometry_s"] = perf_counter() - before
            snapshot = self._assemble(targets)
        except Exception:
            self.latest = None
            self._bindings.clear()
            self._selection_tokens.clear()
            self._regions.clear()
            self._frame = None
            self._depths = None
            self._geometry = None
            self._candidate_records = {}
            raise
        self._queries, self._targets = queries, selection_key
        self._last_refresh_index = self._observation_index
        self._timings["update_s"] = perf_counter() - started
        snapshot.evidence["timings_s"] = dict(self._timings)
        snapshot.generated_at_s = perf_counter()
        snapshot.evidence["stale_at_generation"] = snapshot.generated_at_s - snapshot.captured_at_s > self.max_age_s
        self.latest = snapshot
        self._save(snapshot)
        return snapshot

    def _record_candidates(self, detections, targets):
        queries = {item["id"]: item["query"] for item in targets}
        cameras = {camera.name: camera for camera in self._frame.cameras}
        candidates, records = [], {}
        for detection in detections:
            if (not isinstance(detection, Detection) or detection.prompt_key not in queries
                    or detection.view not in cameras):
                raise ValueError("unexpected segmentation target/view")
            if detection.mask.shape != cameras[detection.view].rgb.shape[:2]:
                raise ValueError("candidate masks must use the input RGB raster")
            yy, xx = np.nonzero(detection.mask)
            if len(xx) < self.engine.config.min_region_pixels:
                continue
            digest = sha256((detection.prompt_key + "\0" + detection.view).encode()
                            + np.packbits(detection.mask).tobytes()).hexdigest()[:16]
            candidate_id = f"f{self._frame.frame_id}:{digest}"
            if candidate_id in records:
                continue
            description = {"id": candidate_id, "candidate_id": candidate_id,
                           "frame_id": self._frame.frame_id, "target_id": detection.prompt_key,
                           "query": queries[detection.prompt_key], "view": detection.view,
                           "score": float(detection.score), "score_is_probability": False,
                           "pixels": int(len(xx)),
                           "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                           "shape_hw": list(detection.mask.shape)}
            candidates.append(description)
            records[candidate_id] = {"description": description, "mask": detection.mask,
                                     "frame": self._frame,
                                     "histogram": _histogram(cameras[detection.view].rgb, detection.mask)}
        self._candidates = sorted(candidates, key=lambda item: (item["target_id"], item["view"], item["bbox_xyxy"]))
        self._candidate_records = records
        # Explicit decisions may refer to the saved frame sent to Astra. Keep a
        # small frame archive; bound selections retain their own region evidence.
        if self._frame.frame_id not in self._candidate_frames:
            self._candidate_frames.append(self._frame.frame_id)
        self._archive.update(records)
        keep = set(self._candidate_frames)
        self._archive = {key: record for key, record in self._archive.items()
                         if record["description"]["frame_id"] in keep}

    def _binding(self, record, query=None):
        name = record["description"]["view"]
        source = self._geometry._views.get(name)
        same_frame = record["description"]["frame_id"] == self._frame.frame_id
        source = source if same_frame else record.get("depth_view")
        points = source.points_world[_mask_on_depth(record["mask"], source)] if source is not None else _empty()
        if len(points) > 2048:
            points = points[np.linspace(0, len(points) - 1, 2048, dtype=int)]
        camera = next(camera for camera in record["frame"].cameras if camera.name == name)
        return {"record": record, "points": points, "camera_pose": camera.world_to_camera.copy(),
                "query": record["description"]["query"] if query is None else query}

    def _matches_binding(self, binding, record):
        old = binding["record"]
        if binding["query"] != record["description"]["query"]:
            return False
        histogram_distance = float(np.abs(old["histogram"] - record["histogram"]).sum() / 2)
        if histogram_distance > self.engine.config.max_appearance_distance:
            return False
        name = record["description"]["view"]
        camera = next(camera for camera in self._frame.cameras if camera.name == name)
        fixed = np.allclose(binding["camera_pose"], camera.world_to_camera, atol=1e-6, rtol=0)
        if fixed and _iou(old["mask"], record["mask"]) >= self.engine.config.min_mask_iou:
            return True
        points = binding["points"]
        current = self._geometry._views.get(name)
        if current is None or not len(points):
            return False
        _, _, surface, _, _, _ = _sample(points, current, self._geometry.config)
        matches = surface & _mask_at_world(points, record["mask"], current)
        return np.count_nonzero(matches) >= 3 and np.count_nonzero(matches) / len(points) >= .2

    def _select(self, target):
        object_id, query = target["id"], target["query"]
        grouped = defaultdict(list)
        for record in self._candidate_records.values():
            if record["description"]["query"] == query:
                grouped[record["description"]["view"]].append(record)
        # Identical query prompts under aliases can produce identical masks.
        for name, values in grouped.items():
            unique = {}
            for record in values:
                unique[np.packbits(record["mask"]).tobytes()] = record
            grouped[name] = list(unique.values())
        masks, reasons, selected_ids, chosen_records = {}, {}, {}, {}
        camera_names = {camera.name for camera in self._frame.cameras}
        if set(target["selections"]) - camera_names:
            raise ValueError("selection refers to an unknown camera")
        if set(target["region_pixels"]) - camera_names:
            raise ValueError("region rectangle refers to an unknown camera")
        for name in sorted(camera_names):
            key = (object_id, name)
            requested = target["selections"].get(name)
            binding = self._bindings.get(key)
            if binding is not None and binding["query"] != query:
                binding = None
                self._bindings.pop(key, None)
                self._selection_tokens.pop(key, None)
            chosen = None
            if requested and self._selection_tokens.get(key) != requested:
                reference = self._archive.get(requested)
                if reference is None or reference["description"]["view"] != name:
                    reasons[name] = "unknown_or_mismatched_selection"
                    continue
                binding = self._binding(reference, query)
                self._bindings[key] = binding
                self._selection_tokens[key] = requested
                if reference["description"]["frame_id"] == self._frame.frame_id:
                    chosen = reference
                    reasons[name] = "astra_selected_saved_candidate"
            if chosen is None and binding is not None:
                matches = [record for record in grouped[name] if self._matches_binding(binding, record)]
                if len(matches) == 1:
                    chosen = matches[0]
                    reasons[name] = "unique_region_reassociation_not_material_tracking"
                else:
                    reasons[name] = "reassociation_ambiguous" if matches else "selected_region_lost"
            elif chosen is None and len(grouped[name]) == 1:
                chosen = grouped[name][0]
                reasons[name] = "unique_query_candidate_identity_hypothesis"
            elif chosen is None:
                reasons[name] = "multiple_candidates_require_selection" if grouped[name] else "no_candidate"
            if chosen is not None:
                masks[name] = chosen["mask"]
                selected_ids[name] = chosen["description"]["candidate_id"]
                chosen_records[name] = chosen
                self._bindings[key] = self._binding(chosen, query)
        if target["region_pixels"]:
            masks, region_reasons = self._crop_regions(target, masks, chosen_records)
            reasons.update({name: reasons.get(name, "") + ";" + reason
                            for name, reason in region_reasons.items()})
        return masks, reasons, selected_ids

    def _crop_regions(self, target, masks, chosen_records):
        """Bind image regions to observed surface patches, then reproject them.

        The patch is a world-anchored estimate. It is suitable for a stationary
        destination or pre-contact part; object motion invalidates its depth
        agreement and requires a new semantic region selection.
        """
        patches, reasons = [], {}
        for name, rectangle in target["region_pixels"].items():
            key = (target["id"], name)
            token = (target["query"], rectangle, target["selections"].get(name))
            entry = self._regions.get(key)
            if entry is None or entry["token"] != token:
                reference_id = target["selections"].get(name)
                reference = self._archive.get(reference_id) if reference_id else chosen_records.get(name)
                if reference is None:
                    reasons[name] = "region_has_no_selected_reference"
                    continue
                source = reference.get("depth_view")
                if source is None:
                    reasons[name] = "region_reference_depth_unavailable"
                    continue
                raw_mask = reference["mask"]
                height, width = raw_mask.shape
                x0, y0, x1, y1 = rectangle
                if x1 > width or y1 > height:
                    raise ValueError("region rectangle lies outside its referenced RGB raster")
                yy, xx = np.indices(raw_mask.shape)
                region_mask = raw_mask & (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
                selected = _mask_on_depth(region_mask, source)
                points = source.points_world[selected & (source.contradictions == 0)]
                entry = {"token": token, "points": points,
                         "frame_id": reference["description"]["frame_id"]}
                self._regions[key] = entry
            patches.append(entry["points"])
        points = np.concatenate(patches) if patches else _empty()
        if not len(points):
            return {}, {**reasons, "region": "no_observed_region_surface"}
        result = {}
        for name, mask in masks.items():
            current = self._geometry._views.get(name)
            if current is None:
                continue
            uv, z = _project(points, current.camera.intrinsics, current.camera.world_to_camera)
            yy, xx, inside = _pixel_indices(uv, mask.shape)
            _, _, surface, _, _, _ = _sample(points, current, self._geometry.config)
            valid = inside & (z > 0) & surface
            projected = np.zeros(mask.shape, dtype=bool)
            projected[yy[valid], xx[valid]] = True
            # One-pixel rasterization padding; never fill occluded geometry.
            padded = np.pad(projected, 1)
            expanded = np.zeros_like(projected)
            height, width = projected.shape
            for dy in range(3):
                for dx in range(3):
                    expanded |= padded[dy:dy + height, dx:dx + width]
            clipped = mask & expanded
            if np.any(clipped):
                result[name] = clipped
                reasons[name] = "world_anchored_observed_region_patch"
            else:
                reasons[name] = "region_patch_no_longer_observed"
        return result, reasons

    def _object(self, target, masks, reasons, selected_ids):
        views, config = self._geometry._views, self._geometry.config
        selected, local, supported, partial, pairs = {}, {}, {}, {}, []
        for name, mask in masks.items():
            if name not in views:
                continue
            view = views[name]
            selected[name] = np.flatnonzero(_mask_on_depth(mask, view))
            indices = selected[name]
            points = view.points_world[indices]
            n = len(points)
            matches_any = np.zeros(n, dtype=bool)
            mask_conflict = np.zeros(n, dtype=bool)
            for other_name, other_mask in masks.items():
                if other_name == name or other_name not in views or not _independent(view, views[other_name], config):
                    continue
                other = views[other_name]
                _, _, surface, occluded, front, valid = _sample(points, other, config)
                in_mask = _mask_at_world(points, other_mask, other)
                matches = surface & in_mask
                matches_any |= matches
                mask_conflict |= surface & ~in_mask
                pairs.append({"source_view": name, "other_view": other_name,
                              "comparable_points": int(surface.sum()), "mask_matches": int(matches.sum()),
                              "mask_disagreements": int((surface & ~in_mask).sum()),
                              "depth_contradictions": int(front.sum()), "occluded": int(occluded.sum()),
                              "not_comparable": int((~valid).sum())})
            clean = (view.contradictions[indices] == 0) & ~mask_conflict
            local[name] = indices[clean]
            supported[name] = indices[clean & matches_any]
            partial[name] = indices[clean & ~matches_any]
        supported_count = sum(map(len, supported.values()))
        # A shared supported patch can associate complementary views of the
        # selected instance. Their other clean surfaces remain single-view
        # evidence, but are useful for opposing finger contacts. Never join
        # disconnected groups just because they share a text query.
        neighbours = {name: set() for name in selected}
        for pair in pairs:
            first, second = pair["source_view"], pair["other_view"]
            if (pair["mask_matches"] >= config.min_target_points
                    and len(supported.get(first, ())) >= config.min_target_points):
                neighbours[first].add(second)
                neighbours[second].add(first)
        components, remaining = [], set(neighbours)
        while remaining:
            pending, component = [min(remaining)], set()
            while pending:
                name = pending.pop()
                if name in component:
                    continue
                component.add(name)
                pending.extend(neighbours[name] - component)
            remaining -= component
            components.append(component)
        associated = max(components, key=lambda component: sum(len(supported[name]) for name in component),
                         default=set())
        ambiguity = any("ambiguous" in reason or "multiple_candidates" in reason
                        or "mismatched_selection" in reason for reason in reasons.values())
        if len(associated) > 1 and sum(len(supported[name]) for name in associated) >= config.min_target_points:
            accepted = {name: np.concatenate((supported[name], partial[name])) for name in sorted(associated)}
            status = "partial" if any(len(partial[name]) for name in associated) else "supported"
        elif supported_count >= config.min_target_points:
            accepted, status = supported, "supported"
        elif len(selected) == 1:
            accepted = {name: np.concatenate((supported[name], partial[name])) for name in selected}
            status = "partial"
        elif len(selected) > 1:
            # Unassociated views remain separate hypotheses. Prefer an explicit
            # selection as the local anchor; never average disconnected objects.
            names = sorted(selected, key=lambda name: (name not in target["selections"], name != "external", name))
            anchor = next((name for name in names if len(local[name]) >= config.min_target_points), None)
            accepted = {anchor: np.concatenate((supported[anchor], partial[anchor]))} if anchor else {}
            status = "partial" if anchor else "conflict"
        else:
            accepted, status = {}, "ambiguous" if ambiguity else "missing"
        points = np.concatenate([views[name].points_world[index] for name, index in accepted.items()]) if accepted else _empty()
        normal_parts = []
        for name, index in accepted.items():
            view = views[name]
            normal = self._normal_maps[name][index].copy()
            # Do not let a normal straddle the selected object's mask boundary.
            region = np.zeros(view.valid.shape, dtype=bool)
            region[view.rows[selected[name]], view.cols[selected[name]]] = True
            interior = np.zeros_like(region)
            if min(region.shape) >= 3:
                interior[1:-1, 1:-1] = (region[1:-1, 1:-1] & region[1:-1, :-2]
                                       & region[1:-1, 2:] & region[:-2, 1:-1] & region[2:, 1:-1])
            normal[~interior[view.rows[index], view.cols[index]]] = 0.
            normal_parts.append(normal)
        normals = np.concatenate(normal_parts) if normal_parts else _empty()
        if len(points) < config.min_target_points:
            points, normals, accepted = _empty(), _empty(), {}
            if status in {"partial", "supported"}:
                status = "conflict" if pairs else "missing"
        support_ranges, point_offset = [], 0
        accepted_support, accepted_partial = {}, {}
        for name, index in accepted.items():
            count = int(np.isin(index, supported[name]).sum())
            accepted_support[name], accepted_partial[name] = count, len(index) - count
            if count:
                support_ranges.append({"start": point_offset, "end": point_offset + count,
                                       "view": name, "support": "crossview_supported"})
            if len(index) > count:
                support_ranges.append({"start": point_offset + count, "end": point_offset + len(index),
                                       "view": name, "support": "single_view_clean_patch"})
            point_offset += len(index)
        evidence = {"selection": reasons, "selected_candidate_ids": selected_ids,
                    "parent_id": target.get("parent_id"),
                    "crossview_pairs": pairs, "crossview_supported_points": supported_count,
                    "unfused_partial_points_by_view": {
                        name: len(index) - accepted_partial.get(name, 0) for name, index in partial.items()},
                    "accepted_points_by_view": {name: len(index) for name, index in accepted.items()},
                    "accepted_crossview_supported_points_by_view": accepted_support,
                    "accepted_single_view_points_by_view": accepted_partial,
                    "point_support_ranges": support_ranges,
                    "associated_view_components": [sorted(component) for component in components],
                    "normal_supported_points": int((np.linalg.norm(normals, axis=1) > .5).sum()),
                    "identity": "selected_or_uniquely_associated_region_hypothesis",
                    "contact_and_attachment": "not_measured", "complete_object_extent": False,
                    "unseen_surfaces": "unknown", "error_bound_m": None}
        return ObjectGeometry(target["id"], target["query"], points, normals, status, masks, evidence), accepted

    def _assemble(self, targets):
        # Archives retain the actual source rays so a delayed model selection
        # can be bound to the frame it saw, rather than to today's pixel values.
        for record in self._candidate_records.values():
            record["depth_view"] = self._geometry._views.get(record["description"]["view"])
        objects, ownership = {}, {}
        for target in targets:
            masks, reasons, selected_ids = self._select(target)
            obj, accepted = self._object(target, masks, reasons, selected_ids)
            objects[target["id"]] = obj
            ownership[target["id"]] = accepted
        clouds, slices, offset = [], {}, 0
        for name, view in self._geometry._views.items():
            clouds.append(view.points_world)
            slices[name] = [offset, offset + len(view.points_world)]
            offset += len(view.points_world)
        points = np.concatenate(clouds) if clouds else _empty()
        owners = np.full(len(points), "", dtype=f"<U{max([1] + [len(key) for key in objects])}")
        ambiguous = np.zeros(len(points), dtype=bool)
        parents = {item["id"]: item.get("parent_id") for item in targets}
        def ancestors(object_id):
            result = set()
            while parents.get(object_id) is not None:
                object_id = parents[object_id]
                result.add(object_id)
            return result
        ancestor_sets = {object_id: ancestors(object_id) for object_id in parents}
        for object_id, accepted in ownership.items():
            for name, indices in accepted.items():
                positions = slices[name][0] + indices
                existing = owners[positions].copy()
                assign = (existing == "") & ~ambiguous[positions]
                for owner in np.unique(existing[existing != ""]):
                    same = existing == owner
                    if owner == object_id or owner in ancestor_sets[object_id]:
                        continue
                    if object_id in ancestor_sets[owner]:
                        assign |= same
                    else:
                        ambiguous[positions[same]] = True
                owners[positions[assign]] = object_id
        owners[ambiguous] = ""
        evidence = {"scene_point_view_slices": slices,
                    "scene_points_include_conflicting_depth": True,
                    "scene_points_include_robot_surfaces": True,
                    "unseen_and_occluded_space": "unknown",
                    "point_ownership": "selected_supported_or_single_view_patch_only; ambiguous_overlap_unassigned",
                    "refresh_steps": self.refresh_steps, "max_age_s": self.max_age_s,
                    "observation_index": self._observation_index,
                    "identity_tracking": "region_and_geometry_association_not_material_tracking",
                    "scores_are_calibrated": False}
        return SceneSnapshot(self._frame.frame_id, self._frame.captured_at_s, perf_counter(),
                             objects, points, self._geometry, self._frame,
                             list(self._candidates), evidence, scene_point_object_ids=owners)

    def _save(self, snapshot):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_revision += 1
        prefix = f"frame_{snapshot.frame_id:06d}_revision_{self._log_revision:06d}"
        path = self.output_dir / (prefix + ".npz")
        if path.exists():
            raise FileExistsError(f"refusing to replace perception artifact: {path}")
        arrays, metadata = {}, {"cameras": [], "candidates": []}
        for index, camera in enumerate(snapshot.frame.cameras):
            key = f"camera_{index}"
            arrays[key + "_rgb"] = camera.rgb
            arrays[key + "_intrinsics"] = camera.intrinsics
            arrays[key + "_world_to_camera"] = camera.world_to_camera
            estimate = self._depths.get(camera.name)
            if estimate is not None:
                arrays[key + "_depth_m"] = estimate.depth_m.astype(np.float32)
                arrays[key + "_depth_intrinsics"] = estimate.intrinsics
            metadata["cameras"].append({"view": camera.name, "array_prefix": key})
        for index, candidate in enumerate(self._candidates):
            key = f"candidate_{index}_mask_bits"
            arrays[key] = np.packbits(self._candidate_records[candidate["id"]]["mask"])
            metadata["candidates"].append({**candidate, "mask_bits_key": key})
        for index, obj in enumerate(snapshot.objects.values()):
            arrays[f"object_{index}_points"] = obj.points.astype(np.float32)
            arrays[f"object_{index}_normals"] = obj.normals.astype(np.float32)
        np.savez_compressed(path, **arrays)
        record = {"snapshot": snapshot.summary(perf_counter()), "arrays": path.name,
                  "array_metadata": metadata,
                  "object_array_order": list(snapshot.objects)}
        with (self.output_dir / "snapshots.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
