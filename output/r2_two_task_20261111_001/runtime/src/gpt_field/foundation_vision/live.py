"""Vision-conditioned executive; no automatic depth action gate or robot control.

Only public StepView RGB/state and synchronized camera calibration enter vision.
The executive still chooses its own model-estimated obstacle_map. Direct/native
actions and the full robot are not checked by this adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ..agent_speedup_v2_user_check_agent_client import (
    AGENT_INSTRUCTIONS, AGENT_KIND, AgentFirstV2Client,
)
from .bridge import file_sha256, frame_from_step_view
from .types import TargetSpec


LIVE_VERSION = "foundation-vision-executive-1"
MAX_TARGETS = 12
MAX_SCENE_POINTS = 64
MAX_VIEW_CANDIDATES = 3
_OLD_SENSOR_CLAIM = (
    "No object poses, depth, segmentation, contact identities,\n"
    "reward or simulator goal predicates are supplied."
)
VISION_INSTRUCTIONS = AGENT_INSTRUCTIONS.replace(_OLD_SENSOR_CLAIM,
    "Context.foundation_vision supplies SAM 3.1 segmentation and DA3 RGB-derived\n"
    "partial surface geometry predictions using public camera calibration.\n"
    "No simulator object poses, contact identities, reward or goal predicates are supplied.") + """

FOUNDATION VISION. Use context.foundation_vision as auxiliary evidence for target
selection, target geometry and path selection, checking it against both RGB views.
Its coordinates are predictions in world metres, not measured object poses.
Respect status, age_s, max_age_s, geometry_usable and candidate ambiguity. Unknown,
missing and stale evidence do not establish empty space or clearance. A surface
centroid is not a grasp point; partial bounds are not the whole object's extent.
Normals describe fitted visible surfaces, not a verified grasp orientation.
Sampled scene points are sparse context only, never a complete obstacle map.
Segmentation scores are uncalibrated. No output proves attachment, slip, contact,
release, task success, or collision freedom. Verify manipulation outcomes in RGB.
Use fresh supported predictions to inform your own obstacle_map when relevant;
retain source=model_estimate and account for unobserved extents and uncertainty.
This is a vision-conditioned executive. It does not automatically convert DA3
points to avoid boxes, gate actions, or check direct/native motions or the full
robot. Existing transit routing checks only the boxes you choose. You remain
responsible for selecting relevant obstacles and requesting new observations.
"""
if _OLD_SENSOR_CLAIM not in AGENT_INSTRUCTIONS:
    raise RuntimeError("Executive sensor description changed; review vision instructions")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _compact_numbers(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _compact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_numbers(item) for item in value]
    return value


def targets_from_plan(plan):
    """All reviewed part descriptions, with no task-specific text or ROI seeds."""
    parts = plan.get("parts") if isinstance(plan, Mapping) else None
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_TARGETS:
        raise ValueError("foundation vision requires 1..12 reviewed parts")
    targets = tuple(TargetSpec(part["id"], part["text"]) for part in parts)
    if len({target.key for target in targets}) != len(targets):
        raise ValueError("reviewed target IDs must be unique")
    return targets


class _RecordingSegmenter:
    def __init__(self, backend):
        self.backend = backend
        self.detections = ()

    def segment(self, frame, targets):
        self.detections = ()
        self.detections = tuple(self.backend.segment(frame, targets))
        return self.detections

    def __getattr__(self, name):
        return getattr(self.backend, name)


def _candidate(detection, index):
    y, x = np.nonzero(detection.mask)
    return {"array_key": f"mask_{index:04d}", "target_key": detection.prompt_key,
            "view": detection.view, "score": float(detection.score),
            "bbox_xyxy_exclusive": ([int(x.min()), int(y.min()), int(x.max()) + 1,
                                      int(y.max()) + 1] if len(x) else None),
            "pixels": int(len(x)), "shape_hw": list(detection.mask.shape)}


def _summary(data, candidates, now_s):
    """Bound prompt size while preserving missing/ambiguous/stale evidence."""
    age = max(float(data["age_s"]), now_s - data["captured_at_s"])
    stale = data["status"] == "stale" or age > data["max_age_s"]
    objects = []
    for obj in data["objects"]:
        geom = obj.get("geometry", {})
        usable = (not stale and obj["status"] in
                  {"new_candidate", "reacquired_candidate", "tracked_estimate"}
                  and geom.get("status") == "supported_surface_hypothesis"
                  and obj.get("centroid_world_m") is not None)
        views = {}
        for view, count in obj["candidate_counts"].items():
            found = sorted((item for item in candidates
                            if item["target_key"] == obj["key"] and item["view"] == view),
                           key=lambda item: item["score"], reverse=True)
            views[view] = {"candidate_count": count, "summary_count": min(len(found), MAX_VIEW_CANDIDATES),
                "candidates": [{key: item[key] for key in ("score", "bbox_xyxy_exclusive", "pixels")}
                               for item in found[:MAX_VIEW_CANDIDATES]],
                "candidates_omitted": max(0, len(found) - MAX_VIEW_CANDIDATES)}
        objects.append({"key": obj["key"], "text": obj["text"],
            "status": "stale" if stale else obj["status"],
            "geometry_status": "stale" if stale else geom.get("status", "unknown"),
            "geometry_usable": usable,
            "centroid_world_m": None if stale else obj.get("centroid_world_m"),
            "bounds_world_m": None if stale else geom.get("bounds_world_m"),
            "bounds_are_complete_object_extent": False,
            "normal_world": None if stale else geom.get("normal_world"),
            "normal_kind": geom.get("normal_kind"),
            "supported_point_count": geom.get("supported_point_count", 0),
            "candidate_counts": deepcopy(obj["candidate_counts"]), "views": views,
            "identity_evidence": "stale_observation" if stale else obj["identity_evidence"],
            "interaction": "unknown"})
    geometry = data["geometry"]
    points = geometry.get("surface_points_world", [])
    indices = np.linspace(0, len(points) - 1, min(len(points), MAX_SCENE_POINTS), dtype=int) if points else []
    quality_keys = ("valid_depth_pixels", "invalid_depth_pixels", "crossview_supported_points",
                    "contradicted_points", "occluded_in_other_views_points")
    scene = {"status": "stale" if stale else "partial_surface_hypotheses",
        "surface_hypothesis_count": geometry.get("surface_hypothesis_count", 0),
        "views": {name: {key: value.get(key) for key in quality_keys}
                  for name, value in geometry.get("views", {}).items()},
        "surface_points_world": [] if stale else [points[i] for i in indices],
        "sampled_context_only": True, "complete_obstacle_map": False,
        "error_bound_m": None, "unseen_and_occluded_space": "unknown"}
    for source, target in (("surface_crossview_support_counts", "sample_crossview_support_counts"),
                           ("surface_contradiction_counts", "sample_contradiction_counts")):
        values = geometry.get(source, [])
        scene[target] = [] if stale or len(values) != len(points) else [values[i] for i in indices]
    result = {"version": LIVE_VERSION, "mode": "vision_conditioned_executive",
        "frame_id": data["frame_id"], "captured_at_s": data["captured_at_s"],
        "generated_at_s": data["generated_at_s"], "summary_at_s": now_s,
        "age_s": age, "max_age_s": data["max_age_s"],
        "status": "stale" if stale else data["status"],
        "source": "rgb_foundation_model_predictions", "coordinate_frame": "world", "length_unit": "metre",
        "objects": objects, "scene": scene,
        "automatic_action_gate": False, "obstacle_map_source": "executive_model_estimate",
        "full_robot_collision_check": False, "direct_native_geometry_check": False,
        "grasp_contact_attachment_evidence": "unknown", "collision_free_certified": False}
    result = _compact_numbers(result)
    if len(_json(result)) > 28000:
        raise ValueError("bounded foundation vision summary exceeded 28000 characters")
    return result


class FoundationVisionAgentClient(AgentFirstV2Client):
    prompt_version = LIVE_VERSION

    def __init__(self, *args, vision_engine, vision_output_dir, vision_extra_targets=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_engine = vision_engine
        reviewed = targets_from_plan(self.plan_lock.plan())
        if isinstance(vision_extra_targets, str):
            raise ValueError("vision_extra_targets must be a sequence of user-specified query strings")
        extras = tuple(TargetSpec(f"extra_{index}", text)
                       for index, text in enumerate(vision_extra_targets))
        self.vision_targets = reviewed + extras
        if (len(self.vision_targets) > MAX_TARGETS
                or len({target.key for target in self.vision_targets}) != len(self.vision_targets)):
            raise ValueError("reviewed parts plus extra queries require unique IDs and at most 12 total targets")
        self.vision_output_dir = Path(vision_output_dir)
        self.vision_output_dir.mkdir(parents=True, exist_ok=False)
        self._vision_segmenter = _RecordingSegmenter(vision_engine.segmenter)
        vision_engine.segmenter = self._vision_segmenter
        self._instruction_text[AGENT_KIND] = VISION_INSTRUCTIONS
        self._vision_cache = None
        self._vision_pending = None
        self._vision_received_at_s = None
        self._vision_request_record = None
        self._vision_stats = {"version": LIVE_VERSION, "mode": "vision_conditioned_executive",
            "executive_requests": 0, "vision_requests": 0, "vision_inferences_completed": 0,
            "cache_reuses": 0, "payloads_built": 0, "consumed_requests": 0,
            "supported_centroids": 0, "nonempty_returned_obstacle_map": 0,
            "vision_elapsed_s": 0., "request_elapsed_s": 0., "errors": 0,
            "automatic_action_gate": False, "full_robot_collision_check": False,
            "direct_native_geometry_check": False, "collision_free_certified": False,
            "consumed_definition": "vision summary in request with an accepted executive response; not proof of reasoning use",
            "clock": "perf_counter; capture timestamp assigned when public StepView enters adapter",
            "engine_config": asdict(vision_engine.config),
            "targets": [{"key": target.key, "text": target.text,
                         "source": "reviewed_plan_part" if target in reviewed else "user_extra_query"}
                        for target in self.vision_targets]}
        self._write_stats()

    def _write_stats(self):
        temporary = self.vision_output_dir / "summary.json.tmp"
        temporary.write_text(_json(self._vision_stats) + "\n", encoding="utf-8")
        temporary.replace(self.vision_output_dir / "summary.json")

    def vision_stats(self):
        return deepcopy(self._vision_stats)

    def _append(self, name, record):
        with (self.vision_output_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")

    def _save_candidates(self, step):
        detections = self._vision_segmenter.detections
        rows = [_candidate(item, index) for index, item in enumerate(detections)]
        path = self.vision_output_dir / f"frame_{step:06d}_sam_candidates.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **{row["array_key"]: item.mask
                                          for row, item in zip(rows, detections)})
        metadata = {"frame_id": step, "candidate_count": len(rows), "candidates": rows,
                    "masks": {"path": path.name, "sha256": file_sha256(path)},
                    "scores_are_calibrated": False}
        (self.vision_output_dir / f"frame_{step:06d}_sam_candidates.json").write_text(
            _json(metadata) + "\n", encoding="utf-8")
        return rows

    def _get_snapshot(self, request):
        view = self._vision_pending
        fields = request["fields"]
        step, geometry = fields["step"], fields["geometry"]
        captured = self._vision_received_at_s
        digest = hashlib.sha256(_json({"geometry": geometry, "state8": view.state.tolist()}).encode())
        for image in (view.external_rgb_upright, view.wrist_rgb_upright):
            digest.update(np.ascontiguousarray(image).tobytes())
        fingerprint = digest.hexdigest()
        cached = self._vision_cache
        if cached is not None and step <= cached["step"]:
            if step < cached["step"] or fingerprint != cached["input_sha256"]:
                raise ValueError("vision frame IDs must increase; repeated step must contain identical public inputs")
            self._vision_stats["cache_reuses"] += 1
            return cached
        frame = frame_from_step_view(view, geometry, step, captured)
        self._vision_stats["vision_requests"] += 1
        self._vision_segmenter.detections = ()
        started = perf_counter()
        try:
            snapshot = self.vision_engine.process(frame, self.vision_targets, now_s=perf_counter())
            data = snapshot.to_dict()
        except Exception:
            self._save_candidates(step)
            raise
        finally:
            self._vision_stats["vision_elapsed_s"] += perf_counter() - started
        rows = self._save_candidates(step)
        path = self.vision_output_dir / f"frame_{step:06d}_snapshot.json"
        path.write_text(_json(data) + "\n", encoding="utf-8")
        self._vision_cache = {"step": step, "input_sha256": fingerprint,
                             "data": data, "candidates": rows,
                             "snapshot_path": path.name, "snapshot_sha256": file_sha256(path)}
        self._vision_stats["vision_inferences_completed"] += 1
        return self._vision_cache

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != AGENT_KIND:
            return super().build_payload(request, grid, mode)
        if self._vision_pending is None or self._vision_request_record is None:
            raise RuntimeError("vision executive payload must originate from validated request_agent")
        # The inherited request_agent already checked public sensors, geometry,
        # allowed caller context, task contract, images, and plan authority.
        cached = self._get_snapshot(request)
        summary = _summary(cached["data"], cached["candidates"], perf_counter())
        trusted = deepcopy(request)
        trusted["fields"]["context"]["foundation_vision"] = summary
        if len(_json(trusted["fields"])) > 56000:
            raise ValueError("executive observation including foundation vision exceeds 56000 characters")
        payload = super().build_payload(trusted, grid, mode)
        # The parent uses its module constant, so changing _instruction_text alone
        # would leave a false sensor claim in the actual wire request.
        payload["instructions"] = VISION_INSTRUCTIONS
        self._vision_stats["payloads_built"] += 1
        self._vision_request_record.update(summary=summary,
            snapshot_path=cached["snapshot_path"], snapshot_sha256=cached["snapshot_sha256"],
            input_sha256=cached["input_sha256"], payload_built=True)
        self._append("payloads.jsonl", {"request_index": self._vision_request_record["request_index"],
            "step": cached["step"], "summary": summary,
            "payload_sha256": hashlib.sha256(_json(payload).encode()).hexdigest(),
            "consumed": False, "meaning": "payload_built; see requests.jsonl for accepted response"})
        return payload

    def request_agent(self, view, context):
        if self._vision_pending is not None:
            raise RuntimeError("concurrent foundation vision requests are unsupported")
        started = perf_counter()
        self._vision_stats["executive_requests"] += 1
        record = {"request_index": self._vision_stats["executive_requests"],
                  "step": context.get("step") if isinstance(context, Mapping) else None,
                  "payload_built": False, "consumed": False, "status": "pending"}
        self._vision_pending, self._vision_request_record = view, record
        self._vision_received_at_s = started
        try:
            result = super().request_agent(view, context)
            if not record["payload_built"]:
                raise RuntimeError("executive response returned without a foundation vision payload")
            obstacle_map = result.get("obstacle_map")
            nonempty = bool(obstacle_map and obstacle_map.get("avoid_boxes"))
            supported = sum(obj["geometry_usable"] for obj in record["summary"]["objects"])
            record.update(status="accepted", consumed=True, supported_centroids=supported,
                          nonempty_returned_obstacle_map=nonempty,
                          response_type=result["type"], request_key=self.last_request_key)
            self._vision_stats["consumed_requests"] += 1
            self._vision_stats["supported_centroids"] += supported
            self._vision_stats["nonempty_returned_obstacle_map"] += int(nonempty)
            return result
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error_message=str(exc)[:1000])
            self._vision_stats["errors"] += 1
            raise
        finally:
            record["elapsed_s"] = perf_counter() - started
            self._vision_stats["request_elapsed_s"] += record["elapsed_s"]
            self._vision_pending = None
            self._vision_received_at_s = None
            self._vision_request_record = None
            self._append("requests.jsonl", record)
            self._write_stats()
