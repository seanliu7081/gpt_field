"""Explicit, scoped SAM/DA3 measurements; never a robot control admission gate.

Construction and capabilities use only the standard library. Backends are loaded
only by an explicit ``measure`` request. RGB uses the existing StepView upright
raster and its matching public camera calibration; no simulator state is read.
An injected zero-argument engine_factory may supply segmenter/depth_estimator.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import monotonic


class OnDemandVisionTools:
    def __init__(self, mode="on-demand", sam_checkpoint=None, da3_model=None,
                 device="cuda:0", output_dir=None, max_age_s=60.,
                 engine_factory=None, clock=monotonic):
        if mode not in {"on-demand", "off"}:
            raise ValueError("vision mode must be on-demand or off")
        if not math.isfinite(float(max_age_s)) or float(max_age_s) <= 0:
            raise ValueError("max_age_s must be positive")
        self.mode, self.device, self.max_age_s = mode, str(device), float(max_age_s)
        self.sam_checkpoint = None if sam_checkpoint is None else str(sam_checkpoint)
        self.da3_model = None if da3_model is None else str(da3_model)
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.engine_factory, self.clock = engine_factory, clock
        self._engine = None
        self._backends = {}
        self._frames = OrderedDict()
        self._closed = False
        self._counters = {"calls": 0, "requests": 0, "result_cache_hits": 0,
                          "segment_backend_calls": 0, "depth_backend_calls": 0,
                          "factory_calls": 0, "unavailable_results": 0,
                          "expired_results": 0, "backend_errors": 0,
                          "logging_errors": 0, "elapsed_s": 0.}

    def capabilities(self):
        def availability(path, directory=False):
            if self.mode == "off" or self._closed:
                return "disabled"
            if self.engine_factory is not None:
                return "injected_loaded" if self._engine is not None else "injected_unloaded"
            if path is None:
                return "unavailable_not_configured"
            try:
                exists = Path(path).is_dir() if directory else Path(path).is_file()
            except OSError:
                exists = False
            return "configured_local" if exists else "unavailable_path"
        return {"mode": self.mode, "enabled": self.mode != "off" and not self._closed,
                "kinds": ["segment", "point", "surface"],
                "sam": availability(self.sam_checkpoint),
                "da3": availability(self.da3_model, directory=True),
                "sam_checkpoint": self.sam_checkpoint, "da3_model": self.da3_model,
                "max_age_s": self.max_age_s, "loading": "only_on_explicit_measure_request",
                "scope": "requested visible region or material pixel; no occupancy or collision certificate",
                "coordinates": "processed upright RGB pixels (u,v); world metres; supplied public calibration",
                "metric_predictions_are_verified": False,
                "point_requires_pixel": True, "query_requests_mask_evidence": True,
                "surface_requires_query_region_or_pixel": True,
                "action_gate": False, "coordinate_overrides": False}

    def stats(self):
        return {**self._counters, "mode": self.mode, "closed": self._closed,
                "initialized_backends": sorted(self._backends),
                "cached_frames": len(self._frames), "automatic_calls": 0}

    @staticmethod
    def _unavailable(request, reason, **details):
        frame = request.get("frame_id")
        return {"id": request.get("id") if isinstance(request.get("id"), str) else None,
                "kind": request.get("kind") if isinstance(request.get("kind"), str) else None,
                "frame_id": frame if type(frame) is int else None, "status": "unavailable",
                "reason": reason, **details}

    @staticmethod
    def _json_safe(value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(key): OnDemandVisionTools._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [OnDemandVisionTools._json_safe(item) for item in value]
        if hasattr(value, "tolist"):
            return OnDemandVisionTools._json_safe(value.tolist())
        return str(value)[:240]

    @staticmethod
    def _error(error):
        return {"type": type(error).__name__, "message": str(error)[:240]}

    def _backend(self, name):
        if name in self._backends:
            return self._backends[name]
        if self.engine_factory is not None:
            if self._engine is None:
                self._counters["factory_calls"] += 1
                self._engine = self.engine_factory()
            backend = (self._engine.get(name) if isinstance(self._engine, dict)
                       else getattr(self._engine, name, None))
            if backend is None:
                raise ValueError(f"{name}_backend_unavailable")
        elif name == "segmenter":
            if self.sam_checkpoint is None or not Path(self.sam_checkpoint).is_file():
                raise ValueError("sam_checkpoint_unavailable")
            from ..foundation_vision.backends import Sam31Segmenter
            backend = Sam31Segmenter(checkpoint_path=self.sam_checkpoint, device=self.device,
                                     local_files_only=True)
        else:
            if self.da3_model is None or not Path(self.da3_model).is_dir():
                raise ValueError("da3_model_directory_unavailable")
            from ..foundation_vision.backends import DA3DepthEstimator
            backend = DA3DepthEstimator(model_id=self.da3_model, device=self.device,
                                       local_files_only=True)
        self._backends[name] = backend
        return backend

    def _frame(self, view, frame_id, received_at, captured_at_s=None):
        import numpy as np
        from ..foundation_vision.bridge import frame_from_step_view
        geometry = getattr(getattr(view, "observation", None), "geometry", None)
        if geometry is None:
            geometry = getattr(view, "geometry", None)
        if geometry is None:
            raise ValueError("public_camera_geometry_unavailable")
        # A repeated tool call on exactly the same public observation must not
        # manufacture a new timestamp, even when its query or request id changes.
        digest = hashlib.sha256()
        digest.update(str(frame_id).encode())
        digest.update(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        digest.update(np.asarray(view.state, dtype=np.float64).tobytes())
        rgb_hashes = {}
        for name, rgb in (("external", view.external_rgb_upright), ("wrist", view.wrist_rgb_upright)):
            array = np.asarray(rgb)
            rgb_hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
            digest.update(str(array.shape).encode())
            digest.update(rgb_hashes[name].encode())
        key = digest.hexdigest()
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        captured = captured_at_s
        if captured is None:
            captured = getattr(view, "captured_at_s", None)
        if captured is None:
            captured = getattr(getattr(view, "observation", None), "captured_at_s", None)
        timestamp_source = "caller_observation_time" if captured is not None else "tool_input_receipt_before_loading"
        captured = received_at if captured is None else float(captured)
        if not math.isfinite(captured) or captured < 0 or captured > received_at + 1e-6:
            raise ValueError("invalid_observation_timestamp")
        frame = frame_from_step_view(view, geometry, frame_id, captured)
        entry = {"key": key, "frame": frame, "captured_at_s": captured,
                 "timestamp_source": timestamp_source, "rgb_sha256": rgb_hashes,
                 "segments": {}, "depths": None, "depth_attempted": False,
                 "depth_error": None, "results": {}}
        self._frames[key] = entry
        while len(self._frames) > 2:
            self._frames.popitem(last=False)
        return entry

    @staticmethod
    def _request(value, frame_id, cameras):
        if not isinstance(value, dict):
            raise ValueError("measurement_request_must_be_object")
        request = {key: value.get(key) for key in ("id", "kind", "query", "view", "region", "pixel", "frame_id")}
        if not isinstance(request["id"], str) or not request["id"]:
            raise ValueError("measurement_id_required")
        if request["frame_id"] != frame_id or isinstance(request["frame_id"], bool):
            raise ValueError("measurement_source_frame_mismatch")
        if request["kind"] not in {"segment", "point", "surface"} or request["view"] not in cameras:
            raise ValueError("unknown_measurement_kind_or_view")
        query = request["query"]
        if query is not None and (not isinstance(query, str) or not query.strip() or len(query) > 256):
            raise ValueError("query_must_be_nonempty_text_or_null")
        height, width = cameras[request["view"]].rgb.shape[:2]
        for key, length in (("pixel", 2), ("region", 4)):
            values = request[key]
            if values is None:
                continue
            if (not isinstance(values, (list, tuple)) or len(values) != length
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)):
                raise ValueError(f"invalid_{key}")
            request[key] = list(map(float, values))
        pixel, region = request["pixel"], request["region"]
        if pixel is not None and not (0 <= pixel[0] <= width-1 and 0 <= pixel[1] <= height-1):
            raise ValueError("pixel_outside_requested_image")
        if region is not None and not (0 <= region[0] < region[2] <= width and 0 <= region[1] < region[3] <= height):
            raise ValueError("region_outside_requested_image")
        if request["kind"] == "point" and pixel is None:
            raise ValueError("point_requires_visible_material_pixel")
        if request["kind"] == "segment" and query is None:
            raise ValueError("segment_requires_query")
        if request["kind"] == "surface" and query is None and region is None and pixel is None:
            raise ValueError("surface_scope_required")
        return request

    def _infer(self, entry, requests):
        from ..foundation_vision.types import TargetSpec
        missing = sorted({r["query"] for r in requests if r["query"] is not None
                          and r["query"] not in entry["segments"]})
        if missing:
            keys = {q: "v3_"+hashlib.sha256(q.encode()).hexdigest()[:20] for q in missing}
            try:
                backend = self._backend("segmenter")
                self._counters["segment_backend_calls"] += 1
                detections = tuple(backend.segment(entry["frame"], [TargetSpec(keys[q], q) for q in missing]))
                for query in missing:
                    entry["segments"][query] = {"detections": tuple(d for d in detections if d.prompt_key == keys[query]),
                                                "error": None}
            except Exception as error:
                self._counters["backend_errors"] += 1
                for query in missing:
                    entry["segments"][query] = {"detections": (), "error": self._error(error)}
        if any(r["kind"] in {"point", "surface"} for r in requests) and not entry["depth_attempted"]:
            entry["depth_attempted"] = True
            try:
                backend = self._backend("depth_estimator")
                self._counters["depth_backend_calls"] += 1
                entry["depths"] = backend.estimate(entry["frame"])
            except Exception as error:
                self._counters["backend_errors"] += 1
                entry["depth_error"] = self._error(error)

    @staticmethod
    def _region_mask(request, shape):
        import numpy as np
        mask = np.ones(shape, bool)
        if request["region"] is not None:
            x0, y0, x1, y1 = request["region"]
            mask[:] = False
            mask[int(math.floor(y0)):int(math.ceil(y1)), int(math.floor(x0)):int(math.ceil(x1))] = True
        elif request["kind"] == "surface" and request["pixel"] is not None:
            u, v = (int(math.floor(x+.5)) for x in request["pixel"])
            mask[:] = False
            mask[max(0, v-3):v+4, max(0, u-3):u+4] = True
        return mask

    @staticmethod
    def _mask_summary(mask, score):
        import numpy as np
        rows, cols = np.nonzero(mask)
        return {"bbox_pixels": [int(cols.min()), int(rows.min()), int(cols.max()+1), int(rows.max()+1)],
                "area_pixels": len(rows), "score": float(score), "score_calibrated": False}

    @staticmethod
    def _depth_camera(camera, estimate):
        import numpy as np
        if estimate is None:
            raise ValueError("requested_view_depth_unavailable")
        h, w = estimate.depth_m.shape
        rh, rw = camera.rgb.shape[:2]
        affine = np.array([[w/rw, 0, (w/rw-1)/2], [0, h/rh, (h/rh-1)/2], [0, 0, 1.]])
        if (not np.allclose(estimate.world_to_camera, camera.world_to_camera, atol=1e-7, rtol=0)
                or not np.allclose(estimate.intrinsics, affine @ camera.intrinsics, atol=1e-7, rtol=1e-7)):
            raise ValueError("depth_camera_calibration_mismatch")
        return estimate

    @staticmethod
    def _crossview(point, source, entry, query):
        import numpy as np
        from ..foundation_vision.geometry import GeometryConfig, _project, _pixel_indices, _tolerance
        config = GeometryConfig()
        origin = -source.world_to_camera[:3, :3].T @ source.world_to_camera[:3, 3]
        observations = []
        for camera in entry["frame"].cameras:
            if camera.name == source.name:
                continue
            other_origin = -camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3]
            estimate = (entry["depths"] or {}).get(camera.name)
            if estimate is None:
                observations.append({"view": camera.name, "status": "unavailable"})
                continue
            try:
                OnDemandVisionTools._depth_camera(camera, estimate)
            except (ValueError, TypeError, AttributeError):
                observations.append({"view": camera.name, "status": "unavailable",
                                     "reason": "other_view_calibration_unavailable"})
                continue
            uv, z = _project(point[None], estimate.intrinsics, estimate.world_to_camera)
            rows, cols, inside = _pixel_indices(uv, estimate.depth_m.shape)
            observed = float(estimate.depth_m[rows[0], cols[0]])
            valid = bool(inside[0] and np.isfinite(z[0]) and z[0] > 0 and math.isfinite(observed) and observed > 0)
            evidence = {"view": camera.name, "independent": bool(np.linalg.norm(origin-other_origin) > config.min_baseline_m)}
            if not valid:
                evidence["status"] = "not_comparable"
            else:
                delta = float(z[0]-observed)
                tolerance = float(_tolerance(z[0], observed, config))
                evidence.update(status="front_conflict" if delta < -tolerance else
                                "occluded" if delta > tolerance else "surface_agreement",
                                projected_depth_m=float(z[0]), predicted_depth_m=observed,
                                difference_m=delta, tolerance_m=tolerance)
                rgb_uv, _ = _project(point[None], camera.intrinsics, camera.world_to_camera)
                evidence["pixel"] = rgb_uv[0].tolist()
                if query is not None:
                    rr, cc, in_rgb = _pixel_indices(rgb_uv, camera.rgb.shape[:2])
                    detections = entry["segments"][query]["detections"]
                    evidence["requested_mask_support"] = bool(in_rgb[0] and any(
                        d.view == camera.name and d.mask[rr[0], cc[0]] for d in detections))
            observations.append(evidence)
        agreeing = [e["view"] for e in observations if e.get("independent") and e["status"] == "surface_agreement"]
        conflicting = [e["view"] for e in observations if e.get("independent") and e["status"] == "front_conflict"]
        return {"supporting_views": [source.name]+agreeing, "conflicting_views": conflicting,
                "status": "conflicting" if conflicting else "consistent" if agreeing else "single_view_unverified",
                "observations": observations, "agreement_is_accuracy_certificate": False}

    @staticmethod
    def _normal(points):
        import numpy as np
        if len(points) < 6:
            return None
        centered = points-np.mean(points, axis=0)
        try:
            _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if len(singular) < 3 or singular[1] <= 1e-8 or singular[2] > .2*singular[1]:
            return None
        return {"normal_world": axes[-1].tolist(), "sign": "unoriented",
                "fit": "local_predicted_points_plane", "flatness_ratio": float(singular[2]/singular[1])}

    def _measurement(self, request, entry, cameras):
        import numpy as np
        camera = cameras[request["view"]]
        scope = self._region_mask(request, camera.rgb.shape[:2])
        masks, summaries = [], []
        if request["query"] is not None:
            segmentation = entry["segments"][request["query"]]
            if segmentation["error"]:
                return self._unavailable(request, "segmentation_unavailable", error=segmentation["error"])
            for detection in segmentation["detections"]:
                if detection.view != camera.name or detection.mask.shape != scope.shape:
                    continue
                mask = detection.mask & scope
                if np.any(mask):
                    masks.append(mask)
                    summaries.append(self._mask_summary(mask, detection.score))
            if not masks:
                return self._unavailable(request, "requested_mask_unobserved")
        else:
            masks = [scope]
        base = {"id": request["id"], "kind": request["kind"], "frame_id": request["frame_id"],
                "view": camera.name, "query": request["query"], "region": request["region"],
                "source": "SAM3.1_query_masks" if request["kind"] == "segment" else "DA3_predicted_depth_on_requested_rays",
                "uncertainty_calibrated": False, "mask_candidates": summaries,
                "mask_candidate_count": len(summaries), "identity_verified": False}
        if request["kind"] == "segment":
            return {**base, "status": "ok", "complete_object_extent": False}
        if entry["depth_error"]:
            return self._unavailable(request, "depth_unavailable", error=entry["depth_error"])
        estimate = self._depth_camera(camera, (entry["depths"] or {}).get(camera.name))
        selected = np.logical_or.reduce(masks)
        if request["kind"] == "point":
            u, v = (int(math.floor(x+.5)) for x in request["pixel"])
            if not selected[v, u]:
                return self._unavailable(request, "selected_pixel_outside_requested_material")
            matches = sum(bool(mask[v, u]) for mask in masks)
            local = np.zeros_like(selected)
            local[max(0, v-2):v+3, max(0, u-2):u+3] = True
            selected &= local
            base.update(pixel=request["pixel"], mask_matches_at_pixel=matches if request["query"] is not None else None)
        rows, cols = np.nonzero(selected)
        raw_h, raw_w = camera.rgb.shape[:2]
        h, w = estimate.depth_m.shape
        dr = np.floor((rows+.5)*h/raw_h).astype(int).clip(0, h-1)
        dc = np.floor((cols+.5)*w/raw_w).astype(int).clip(0, w-1)
        depths = estimate.depth_m[dr, dc]
        good = np.isfinite(depths) & (depths > 0)
        rows, cols, depths = rows[good], cols[good], depths[good]
        if not len(depths):
            return self._unavailable(request, "requested_depth_unobserved")
        rays = np.column_stack((cols, rows, np.ones(len(rows)))) @ np.linalg.inv(camera.intrinsics).T
        points = (rays*depths[:, None]-camera.world_to_camera[:3, 3]) @ camera.world_to_camera[:3, :3]
        sample = points if len(points) <= 2048 else points[np.linspace(0, len(points)-1, 2048, dtype=int)]
        base.update(status="uncertain", verified=False, complete_object_extent=False,
                    observed_samples=len(points), local_normal=self._normal(sample),
                    depth_range_m=[float(depths.min()), float(depths.max())],
                    depth_q10_q90_m=np.quantile(depths, [.1, .9]).tolist(),
                    geometry_is_prediction=True, action_authorized_by_measurement=False)
        if request["kind"] == "point":
            pixel = np.asarray(request["pixel"], float)
            dp = (pixel+.5)*[w/raw_w, h/raw_h]-.5
            cu, rv = np.floor(dp+.5).astype(int)
            depth = float(estimate.depth_m[rv, cu])
            if not math.isfinite(depth) or depth <= 0:
                return self._unavailable(request, "selected_pixel_depth_unavailable")
            # The selected material ray owns the point: no object centroid and
            # no patch-median substitution can silently change its position.
            point = (np.linalg.solve(camera.intrinsics, np.r_[pixel, 1])*depth-camera.world_to_camera[:3, 3]) @ camera.world_to_camera[:3, :3]
            base.update(point_world_m=point.tolist(), depth_m=depth,
                        crossview=self._crossview(point, camera, entry, request["query"]),
                        position_source="selected_material_pixel", whole_object_centroid_used=False)
        else:
            base.update(visible_bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
                        scope_identity="union_of_requested_visible_masks" if request["query"] is not None else "requested_image_region",
                        identity_ambiguous=len(masks) > 1,
                        crossview={"status": "not_evaluated_for_surface_extent", "verified": False})
        return base

    def _finish(self, frame_id, captured, received, results, entry=None, reason=None):
        now = self.clock()
        age = None if captured is None else max(0., now-captured)
        expired = age is not None and age > self.max_age_s
        final = []
        for item in results:
            value = deepcopy(item)
            measured_at = value.get("measured_at_s", now)
            cache_hit = value.get("cache_hit")
            if expired and value.get("status") != "unavailable":
                value = self._unavailable(value, "observation_expired", prediction_discarded=True)
            self._counters["expired_results"] += value.get("reason") == "observation_expired"
            value.update(captured_at_s=captured, measured_at_s=measured_at,
                         generated_at_s=measured_at, served_at_s=now, age_s=age)
            if cache_hit is not None:
                value["cache_hit"] = cache_hit
            self._counters["unavailable_results"] += value.get("status") == "unavailable"
            final.append(value)
        available = sum(item["status"] != "unavailable" for item in final)
        report = {"frame_id": frame_id, "captured_at_s": captured, "generated_at_s": now,
                  "age_s": age, "mode": self.mode,
                  "status": "ok" if final and available == len(final) else "partial" if available else "unavailable",
                  "results": final, "elapsed_s": max(0., now-received),
                  "measurement_is_control_gate": False, "unknown_space_is_free": False}
        if reason:
            report["reason"] = reason
        if entry is not None:
            report["source"] = {"public_input_sha256": entry["key"], "rgb_sha256": entry["rgb_sha256"],
                                "timestamp_source": entry["timestamp_source"],
                                "raster": "StepView upright RGB; raw transport image horizontally mirrored",
                                "coordinate_convention": "OpenCV x-right y-down z-forward; world metres"}
        provenance = {}
        for name, backend in self._backends.items():
            method = getattr(backend, "provenance", None)
            if callable(method):
                try:
                    provenance[name] = method()
                except Exception:
                    provenance[name] = {"available": False}
        report["backend_provenance"] = provenance
        report = self._json_safe(report)
        self._counters["elapsed_s"] += report["elapsed_s"]
        if self.output_dir is not None:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with (self.output_dir / "measurements.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report, ensure_ascii=False, allow_nan=False)+"\n")
            except Exception:
                self._counters["logging_errors"] += 1
        return report

    def measure(self, view, requests, frame_id, *, captured_at_s=None):
        received = self.clock()
        self._counters["calls"] += 1
        values = list(requests) if isinstance(requests, (list, tuple)) else []
        self._counters["requests"] += len(values)
        stubs = [value if isinstance(value, dict) else {"id": None, "kind": None, "frame_id": frame_id} for value in values]
        if self.mode == "off" or self._closed:
            return self._finish(frame_id, None, received,
                [self._unavailable(value, "vision_disabled") for value in stubs], reason="vision_disabled")
        if not values:
            return self._finish(frame_id, None, received, [], reason="no_measurement_requests")
        entry = None
        try:
            if type(frame_id) is not int or frame_id < 0:
                raise ValueError("invalid_frame_id")
            entry = self._frame(view, frame_id, received, captured_at_s)
            cameras = {camera.name: camera for camera in entry["frame"].cameras}
            requests_valid, errors = {}, {}
            for index, value in enumerate(values):
                try:
                    requests_valid[index] = self._request(value, frame_id, cameras)
                except Exception as error:
                    errors[index] = self._unavailable(stubs[index], "invalid_measurement_request", error=self._error(error))
            if received-entry["captured_at_s"] > self.max_age_s:
                results = [errors.get(index) or self._unavailable(value, "observation_expired")
                           for index, value in enumerate(stubs)]
                return self._finish(frame_id, entry["captured_at_s"], received, results, entry)
            self._infer(entry, list(requests_valid.values()))
            results = []
            for index, stub in enumerate(stubs):
                if index in errors:
                    results.append(errors[index])
                    continue
                request = requests_valid[index]
                key = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if key in entry["results"]:
                    self._counters["result_cache_hits"] += 1
                    result = deepcopy(entry["results"][key])
                    result["cache_hit"] = True
                else:
                    try:
                        result = self._measurement(request, entry, cameras)
                    except Exception as error:
                        result = self._unavailable(request, "measurement_unavailable", error=self._error(error))
                    result["measured_at_s"] = self.clock()
                    entry["results"][key] = deepcopy(result)
                    result["cache_hit"] = False
                results.append(result)
            return self._finish(frame_id, entry["captured_at_s"], received, results, entry)
        except Exception as error:
            return self._finish(frame_id, None if entry is None else entry["captured_at_s"], received,
                [self._unavailable(value, "measurement_input_unavailable", error=self._error(error)) for value in stubs], entry)

    def close(self):
        if self._closed:
            return
        seen = set()
        for backend in (*self._backends.values(), self._engine):
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            method = getattr(backend, "close", None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
        self._frames.clear()
        self._backends.clear()
        self._engine = None
        self._closed = True
