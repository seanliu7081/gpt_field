"""Fast material tracking and payload lifecycle using public RGB/proprioception.

Image evidence is conditional on rigid material correspondence. No contact,
object pose or reward from the evaluator is consumed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..foundation_harness.feedback import grip_pose, transform_points, project


@dataclass
class Track:
    gray: np.ndarray
    pixels: np.ndarray
    local: np.ndarray


class MaterialTrack:
    """Track a selected material region; never silently substitute object centre."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.object_id = None
        self.tracks = {}
        self.reference = None
        self.pose = None
        self.last_pose = None
        self.evidence = {"status": "unavailable", "reason": "not_seeded"}
        self.last_frame = None

    def seed(self, scene, object_id):
        import cv2
        self.reset()
        obj = scene.objects.get(object_id)
        if obj is None or obj.status in {"missing", "ambiguous", "conflict", "unobserved", "stale"}:
            return self.evidence
        points = np.asarray(obj.points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(1)]
        if len(points) < 8:
            return self.evidence
        # Keep all supported patches: no dominant-component veto.
        self.reference = np.eye(4)
        self.reference[:3, 3] = np.median(points, axis=0)
        local = points - self.reference[:3, 3]
        for camera in scene.frame.cameras:
            mask = obj.masks.get(camera.name)
            if mask is None or np.shape(mask) != camera.rgb.shape[:2]:
                continue
            gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
            features = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=.02,
                                              minDistance=4, mask=np.uint8(mask)*255)
            if features is None:
                continue
            uv, depth = project(points, camera)
            valid = np.isfinite(uv).all(1) & (depth > 0)
            uv, xyz = uv[valid], local[valid]
            if not len(uv):
                continue
            pixels, refs, used = [], [], set()
            for pixel in features[:, 0]:
                distances = np.linalg.norm(uv-pixel, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= 3 and index not in used:
                    pixels.append(pixel); refs.append(xyz[index]); used.add(index)
            if len(refs) >= 6:
                self.tracks[camera.name] = Track(gray, np.float32(pixels), np.float64(refs))
        self.object_id, self.pose = object_id, self.reference.copy()
        self.evidence = {"status": "seeded" if self.tracks else "unavailable",
                         "object_id": object_id, "views": len(self.tracks)}
        return self.evidence

    def update(self, frame):
        import cv2
        if self.reference is None or self.last_frame == frame.frame_id:
            return self.evidence
        self.last_frame = frame.frame_id
        estimates, tracks = [], {}
        for camera in frame.cameras:
            old = self.tracks.get(camera.name)
            if old is None or len(old.pixels) < 6:
                continue
            gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
            previous = old.pixels.reshape(-1, 1, 2)
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(old.gray, gray, previous, None,
                                               winSize=(21, 21), maxLevel=3)
            if nxt is None or ok is None:
                continue
            back, bok, _ = cv2.calcOpticalFlowPyrLK(gray, old.gray, nxt, None,
                                                winSize=(21, 21), maxLevel=3)
            if back is None or bok is None:
                continue
            valid = ok[:, 0].astype(bool) & bok[:, 0].astype(bool)
            valid &= np.linalg.norm(back[:, 0]-previous[:, 0], axis=1) < 1.5
            valid &= np.isfinite(nxt[:, 0]).all(1)
            pixels, local = nxt[valid, 0], old.local[valid]
            h, w = gray.shape
            valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < w)
                     & (pixels[:, 1] >= 0) & (pixels[:, 1] < h))
            pixels, local = pixels[valid], local[valid]
            if len(local) < 6:
                continue
            try:
                success, rv, tv, inliers = cv2.solvePnPRansac(
                    np.float64(local), np.float64(pixels), camera.intrinsics, None,
                    iterationsCount=40, reprojectionError=3., flags=cv2.SOLVEPNP_EPNP)
            except cv2.error:
                continue
            if not success or inliers is None or len(inliers) < 6:
                continue
            ids = inliers[:, 0]
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(rv)[0]
            pose[:3, 3] = tv[:, 0]
            pose = np.linalg.inv(camera.world_to_camera) @ pose
            predicted, z = project(transform_points(pose, local[ids]), camera)
            residual = float(np.median(np.linalg.norm(predicted-pixels[ids], axis=1)))
            if residual > 3 or np.any(z <= 0):
                continue
            sigma = max(.002, residual*float(np.median(z))/float(camera.intrinsics[0, 0]))
            tracks[camera.name] = Track(gray, np.float32(pixels[ids]), local[ids])
            estimates.append((residual, sigma, pose, camera.name, len(ids)))
        self.tracks = tracks
        if not estimates:
            self.evidence = {"status": "unavailable", "object_id": self.object_id,
                             "reason": "material_correspondence_lost"}
            return self.evidence
        estimates.sort(key=lambda item: item[0])
        _, sigma, pose, _, _ = estimates[0]
        if any(np.linalg.norm(item[2][:3, 3]-pose[:3, 3]) > max(.02, 4*(sigma+item[1]))
               for item in estimates[1:]):
            self.evidence = {"status": "unavailable", "object_id": self.object_id,
                             "reason": "views_disagree"}
            return self.evidence
        self.last_pose, self.pose = self.pose, pose
        self.evidence = {"status": "tracked", "object_id": self.object_id,
                         "position_world_m": pose[:3, 3].tolist(), "sigma_m": sigma,
                         "delta_from_seed_m": (pose[:3, 3]-self.reference[:3, 3]).tolist(),
                         "views": [{"name": e[3], "points": e[4], "residual_px": e[0]} for e in estimates],
                         "source": "RGB_LK_PnP", "physical_contact_measured": False}
        return self.evidence


class PayloadState:
    """Full observed object geometry survives occlusion and identity-preserving attachment."""

    def __init__(self):
        self.held = None
        self.released = None
        self.track = MaterialTrack()
        self.votes = 0
        self.stationary_votes = 0
        self.close_grip = None
        self.close_object = None
        self.last_gap = None
        self.report = {"state": "free"}

    def attach_candidate(self, scene, object_id, view, *, keep_upright=False):
        obj = scene.objects.get(object_id)
        points = np.empty((0, 3)) if obj is None else np.asarray(obj.points, float).reshape(-1, 3)
        points = points[np.isfinite(points).all(1)]
        explicit_region = bool(obj and obj.evidence.get("parent_id"))
        unknown = (len(points) < 8 or explicit_region or obj is None
                   or obj.status in {"missing", "ambiguous", "conflict", "unobserved", "stale"})
        pose = grip_pose(view.state)
        sigma = .01 if obj is None else float(np.max(np.asarray(obj.evidence.get("sigma_m", .01))))
        local = transform_points(np.linalg.inv(pose), points)
        if len(local) > 2048:
            # Preserve all extrema, then a bounded sample of the whole object.
            extrema = np.unique(np.r_[np.argmin(local, axis=0), np.argmax(local, axis=0)])
            indices = np.unique(np.r_[extrema, np.linspace(0, len(local)-1, 2048, dtype=int)])
            local = local[indices]
        self.held = {"object_id": object_id, "points_in_gripper": local,
                     "owner_ids": [object_id] + [key for key, value in scene.objects.items()
                                                  if value.evidence.get("parent_id") == object_id],
                     "sigma_m": sigma, "confirmed": False, "extent_unknown": unknown,
                     "source": "whole_object_observed_geometry_at_close", "geometry_complete": False,
                     "reference_in_gripper": np.eye(4)}
        if keep_upright:
            self.held["upright_in_gripper"] = pose[:3, :3].T @ np.array([0., 0., 1.])
        self.track.seed(scene, object_id)
        self.close_grip = pose
        self.close_object = None if self.track.pose is None else self.track.pose.copy()
        self.last_gap, self.votes = view.gap_mm, 0
        self.released = None
        self.report = {"state": "held_candidate", "object_id": object_id,
                       "extent_unknown": unknown, "gap_mm": view.gap_mm}
        return self.report

    def released_at(self, view):
        if self.held is not None:
            self.released = {**self.held,
                "predicted_world_points": transform_points(grip_pose(view.state), self.held["points_in_gripper"]),
                "released_grip": grip_pose(view.state),
                "tracked_pose_at_release": None if self.track.pose is None else self.track.pose.copy()}
        self.held = None
        self.stationary_votes = 0
        self.report = {"state": "released_unobserved",
                       "object_id": (self.released or {}).get("object_id")}

    def observe(self, view, frame, command):
        evidence = self.track.update(frame)
        if self.held is not None and command == "close":
            grip = grip_pose(view.state)
            moved = np.linalg.norm(grip[:3, 3]-self.close_grip[:3, 3])
            attached_error, stationary_error = None, None
            if evidence.get("status") == "tracked" and self.close_object is not None:
                expected = grip @ np.linalg.inv(self.close_grip) @ self.close_object
                attached_error = float(np.linalg.norm(self.track.pose[:3, 3]-expected[:3, 3]))
                stationary_error = float(np.linalg.norm(self.track.pose[:3, 3]-self.close_object[:3, 3]))
                sigma = float(evidence.get("sigma_m", .005))
                good = (moved >= .008 and attached_error <= max(.01, 3*sigma)
                        and stationary_error > attached_error+.003 and view.gap_mm > 3.)
                self.votes = self.votes+1 if good else 0
                if self.votes >= 2:
                    self.held["confirmed"] = True
                if self.held["confirmed"] and attached_error > max(.015, 4*sigma):
                    self.held["confirmed"] = False
                    self.report = {"state": "suspected_slip", "object_id": self.held["object_id"],
                                   "attached_error_m": attached_error, "tracking": evidence}
                    return self.report
            if view.gap_mm < 3. and moved > .004:
                self.held["confirmed"] = False
                self.report = {"state": "object_absent", "object_id": self.held["object_id"],
                               "gap_mm": view.gap_mm, "tracking": evidence}
                return self.report
            self.report = {"state": "held_confirmed" if self.held["confirmed"] else "held_candidate",
                           "object_id": self.held["object_id"], "tracking": evidence,
                           "attached_error_m": attached_error, "stationary_error_m": stationary_error,
                           "extent_unknown": self.held["extent_unknown"]}
        elif self.released is not None and command == "open":
            at_release = self.released.get("tracked_pose_at_release")
            hand_delta = view.position-self.released["released_grip"][:3, 3]
            stationary = False
            if evidence.get("status") == "tracked" and at_release is not None:
                object_delta = self.track.pose[:3, 3]-at_release[:3, 3]
                stationary_error = float(np.linalg.norm(object_delta))
                attached_error = float(np.linalg.norm(object_delta-hand_delta))
                stationary = (np.linalg.norm(hand_delta) >= .012 and stationary_error < .008
                              and attached_error > stationary_error+.005)
            self.stationary_votes = self.stationary_votes+1 if stationary else 0
            self.report = {"state": "released_stationary" if self.stationary_votes >= 3 else "released_unobserved",
                           "object_id": self.released["object_id"], "tracking": evidence,
                           "support_contact_measured": False}
        return self.report

    def summary(self):
        held = None if self.held is None else {k: v for k, v in self.held.items()
                                             if k not in {"points_in_gripper", "reference_in_gripper"}}
        return {"held": held, "feedback": self.report}
