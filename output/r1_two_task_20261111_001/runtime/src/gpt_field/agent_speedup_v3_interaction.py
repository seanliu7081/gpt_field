"""Conditional interaction evidence from public RGB, calibration and robot motion.

Tracked material must distinguish hand-rigid from world-stationary predictions
in both views. Finger gap, the model's attachment flag and FK alone never pass.
No simulator object pose, depth or contact information is consumed.
"""
from __future__ import annotations

from copy import deepcopy
from time import monotonic

import cv2
import numpy as np

VERSION = "agent-speedup-v3-visual-interaction-2"
_SIGNS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def _project(points, camera):
    local = (np.asarray(points)-camera["origin"]) @ camera["rotation"]
    depth = -local[:, 2]
    uv = np.column_stack((camera["principal"][0]+camera["focal"][0]*local[:, 0]/np.maximum(depth, 1e-8),
                          camera["principal"][1]-camera["focal"][1]*local[:, 1]/np.maximum(depth, 1e-8)))
    return uv, depth


def _frame(view, name):
    metadata = view.observation.geometry
    raw = metadata["cameras"][name]
    rgb = getattr(view, name+"_rgb_upright")
    h, w = rgb.shape[:2]
    scale = min(1., 256/max(h, w))
    size = (round(w*scale), round(h*scale))
    gray = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    camera = {"origin": np.asarray(raw["position_world_m"], float),
              "rotation": np.asarray(raw["rotation_world_from_camera"], float),
              "focal": np.asarray(raw["focal_length_px"], float)*scale,
              "principal": np.asarray(raw["principal_point_px"], float)*scale}
    return gray, camera


def _mask(boxes, camera, shape):
    mask = np.zeros(shape, np.uint8)
    for box in boxes:
        corners = (_SIGNS*np.asarray(box["half"])) @ np.asarray(box["rotation"]).T+box["center"]
        uv, depth = _project(corners, camera)
        if np.all(depth > 1e-5):
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 255)
    return mask


def _unproject(uv, depth, camera):
    local = np.column_stack(((uv[:, 0]-camera["principal"][0])*depth/camera["focal"][0],
                            -(uv[:, 1]-camera["principal"][1])*depth/camera["focal"][1],
                            np.full(len(uv), -depth)))
    return local @ camera["rotation"].T+camera["origin"]


class InteractionEvidence:
    """Short-lived, two-view feature tracks; unavailable images remain unknown."""
    def __init__(self, geometry):
        self.geometry = geometry
        self.active = None
        self.records = {}
        self.compute_s = 0.
        self.observations = 0

    def adopt_reference(self, prepared):
        """Commit an observed baseline prepared before an acknowledged action.

        The caller owns the preparation timer. The stable monitor instance
        retains its accumulated counters and unrelated historical evidence.
        """
        if prepared.geometry is not self.geometry or prepared.active is None:
            raise ValueError("interaction reference must use the same geometry")
        self.active = prepared.active
        self.records.pop((self.active["kind"], self.active["target_id"]), None)

    def _capture_view(self, session, view, name, step):
        entity = self.geometry.entities.get(session["target_id"])
        if entity is None:
            raise ValueError("target_extent_unavailable")
        if session["kind"] == "release" and entity.get("release_position_estimate_m") is not None:
            # A reacquired release reference stays around the last released
            # world hypothesis, rather than following the retreating hand.
            entity = {**entity, "attachment": "none", "position_m": entity["release_position_estimate_m"],
                      "quaternion_wxyz": entity.get("release_quaternion_estimate_wxyz", entity.get("quaternion_wxyz"))}
        boxes = self.geometry._entity_boxes(entity, view, suppress_stale_padding=True)
        if not boxes:
            raise ValueError("target_extent_unavailable")
        gray, camera = _frame(view, name)
        mask = cv2.erode(_mask(boxes, camera, gray.shape), np.ones((3, 3), np.uint8))
        occlusion = _mask(self.geometry._visual_occluders(view, session["target_id"]), camera, gray.shape)
        mask[cv2.dilate(occlusion, np.ones((3, 3), np.uint8)) > 0] = 0
        points = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=.04, minDistance=3, mask=mask)
        if points is None or len(points) < 4:
            raise ValueError("insufficient_visible_material_features")
        points = points.reshape(-1, 2)
        center = np.mean([b["center"] for b in boxes], axis=0)
        depth = -float(((center-camera["origin"]) @ camera["rotation"])[2])
        if depth <= 0:
            raise ValueError("target_behind_camera")
        return {"gray": gray, "camera": camera, "points": points,
                "start_points": points.copy(), "world": _unproject(points, depth, camera), "depth": depth,
                "reference_position": np.asarray(view.position).copy(), "reference_rotation": view.R.copy(),
                "reference_step": int(step), "stable_samples": 0,
                "report": {"status": "unknown", "reason": "baseline_only", "reference_step": int(step)}}

    def begin(self, target_id, view, step, kind="hold"):
        if kind not in ("hold", "release"):
            raise ValueError("Unsupported interaction evidence kind")
        began = monotonic()
        self.records.pop((kind, target_id), None)
        session = {"kind": kind, "target_id": target_id, "reference_step": int(step),
                   "start_position": np.asarray(view.position).copy(), "start_rotation": view.R.copy(),
                   "last_step": int(step), "views": {}, "supported_samples": 0}
        for name in ("external", "wrist"):
            try:
                session["views"][name] = self._capture_view(session, view, name, step)
            except (AttributeError, KeyError, TypeError, ValueError, cv2.error):
                continue
        self.active = session
        self.compute_s += monotonic()-began
        return self.evidence(kind, target_id, view, step, step)

    def observe(self, view, step):
        session = self.active
        if session is None or step <= session["last_step"]:
            return
        began = monotonic()
        session["last_step"] = int(step)
        translation = float(np.linalg.norm(view.position-session["start_position"]))
        for name in ("external", "wrist"):
            state = session["views"].get(name)
            try:
                if state is None or state.get("needs_reference"):
                    # A newly visible view contributes only future motion from
                    # this observed reference. Previous motion is never replayed
                    # as evidence for a camera that could not see the material.
                    session["views"][name] = self._capture_view(session, view, name, step)
                    continue
                gray, camera = _frame(view, name)
                old = state["points"].reshape(-1, 1, 2).astype(np.float32)
                tracked, ok, _ = cv2.calcOpticalFlowPyrLK(state["gray"], gray, old, None,
                    winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .02))
                if tracked is None:
                    raise ValueError("no_tracks")
                back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, state["gray"], tracked, None,
                    winSize=(15, 15), maxLevel=2)
                if back is None:
                    raise ValueError("no_backward_tracks")
                points = tracked.reshape(-1, 2)
                good = (ok.reshape(-1) > 0) & (back_ok.reshape(-1) > 0)
                good &= np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1) <= 1.
                good &= (points[:, 0] >= 1) & (points[:, 0] < gray.shape[1]-1) & (points[:, 1] >= 1) & (points[:, 1] < gray.shape[0]-1)
                occlusion = _mask(self.geometry._visual_occluders(view, session["target_id"]), camera, gray.shape)
                idx = np.round(points).astype(int)
                idx[:, 0] = np.clip(idx[:, 0], 0, gray.shape[1]-1)
                idx[:, 1] = np.clip(idx[:, 1], 0, gray.shape[0]-1)
                good &= occlusion[idx[:, 1], idx[:, 0]] == 0
                if int(good.sum()) < 4:
                    raise ValueError("insufficient_visible_material_tracks")
                points, previous = points[good], state["points"][good]
                world = state["world"][good]
                local = (world-state["reference_position"]) @ state["reference_rotation"]
                held_world = local @ view.R.T+view.position
                held_uv, _ = _project(held_world, camera)
                fixed_uv, _ = _project(world, camera)
                separation = float(np.median(np.linalg.norm(held_uv-fixed_uv, axis=1)))
                held_error = float(np.median(np.linalg.norm(points-held_uv, axis=1)))
                fixed_error = float(np.median(np.linalg.norm(points-fixed_uv, axis=1)))
                threshold = max(1.25, min(3., .35*separation))
                supported = separation >= 3. and held_error <= threshold and fixed_error > held_error+1.5
                report = {"status": "supported" if supported else "unknown", "tracked_features": len(points),
                          "reference_step": state["reference_step"],
                          "hypothesis_separation_px": separation, "held_residual_px": held_error,
                          "stationary_residual_px": fixed_error}
                if session["kind"] == "release":
                    # Release support is recent world stability after opening,
                    # not agreement with the pre-release (possibly falling) pose.
                    old_world = _unproject(previous, state["depth"], state["camera"])
                    expected_fixed, _ = _project(old_world, camera)
                    recent_error = float(np.median(np.linalg.norm(points-expected_fixed, axis=1)))
                    state["stable_samples"] = state["stable_samples"]+1 if recent_error <= 1.25 else 0
                    supported = state["stable_samples"] >= 3 and separation >= 3. and held_error > threshold
                    report.update(status="supported" if supported else "unknown", recent_world_stability_error_px=recent_error,
                                  stable_samples=state["stable_samples"])
                state.update(gray=gray, camera=camera, points=points, world=world,
                             start_points=state["start_points"][good], report=report)
            except (AttributeError, KeyError, TypeError, ValueError, cv2.error) as exc:
                if state is not None:
                    state["report"] = {"status": "unknown", "reason": str(exc)[:120]}
                    state["needs_reference"] = True
        reports = {name: state["report"] for name, state in session["views"].items()}
        supported = all(reports.get(name, {}).get("status") == "supported" for name in ("external", "wrist"))
        session["supported_samples"] = session["supported_samples"]+1 if supported else 0
        result = {"version": VERSION, "status": "supported" if session["supported_samples"] >= 2 else "unknown",
                  "kind": session["kind"], "target_id": session["target_id"], "reference_step": session["reference_step"],
                  "observed_at_step": int(step), "translation_m": translation, "views": reports,
                  "source": "public_two_view_feature_motion", "attachment_verified": False,
                  "conditional_attachment_supported": session["kind"] == "hold" and session["supported_samples"] >= 2,
                  "task_success": "not_available_to_policy", "geometry_pose_updated": False}
        self.records[(session["kind"], session["target_id"])] = result
        self.observations += 1
        self.compute_s += monotonic()-began

    def evidence(self, kind, target_id, view, step, since_step=0):
        self.observe(view, step)
        result = self.records.get((kind, target_id))
        if result and result["reference_step"] >= since_step and result["observed_at_step"] == step:
            return deepcopy(result)
        return {"version": VERSION, "status": "unknown", "kind": kind, "target_id": target_id,
                "source": "public_two_view_feature_motion", "reason": "fresh_two_view_material_motion_required",
                "observed_at_step": int(step), "attachment_verified": False,
                "conditional_attachment_supported": False, "task_success": "not_available_to_policy"}

    def performance_stats(self):
        return {"visual_interaction_s": self.compute_s, "visual_interaction_observations": self.observations,
                "scope": "subset of policy act/observe timing"}
