"""Material-point feedback from RGB and measured robot pose, never contact truth.

PnP estimates a rigid reference from tracked image features. Its output is
conditional visual evidence, not a physical attachment certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..robot_tool_geometry import _rotation_matrix
from .types import jsonable
from .surface import planning_surface


def grip_pose(state):
    pose = np.eye(4)
    pose[:3, :3] = _rotation_matrix(np.asarray(state)[3:6])
    pose[:3, 3] = np.asarray(state)[:3]
    return pose


def transform_points(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def project(points, camera):
    xyz = transform_points(camera.world_to_camera, points)
    uvw = xyz @ camera.intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, xyz[:, 2]


@dataclass
class _Track:
    gray: np.ndarray
    pixels: np.ndarray
    local: np.ndarray


class InteractionTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}
        self.object_id = None
        self.reference_pose = None
        self.reference_points = None
        self.last_pose = None
        self.last_grip = None
        self.held = None
        self.contact_mode = "free_object"
        self.keep_upright = False
        self.attachment_votes = 0
        self.release_votes = 0
        self.last_observation = None
        self.report = {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                       "reason": "no_material_tracks"}

    def seed(self, scene, object_id, view):
        import cv2
        self.reset()
        obj = scene.objects.get(object_id)
        if obj is None or len(obj.points) < 8 or obj.status in {"missing", "ambiguous", "conflict"}:
            return self.report
        self.object_id = object_id
        self.reference_pose = grip_pose(view.state)
        self.last_grip = self.reference_pose.copy()
        points, surface_evidence = planning_surface(obj)
        if len(points) < 8:
            self.report = {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                           "object_id": object_id, "reason": "tracking_surface_unavailable",
                           "surface": surface_evidence}
            return self.report
        self.reference_points = transform_points(np.linalg.inv(self.reference_pose), points)
        for camera in scene.frame.cameras:
            mask = obj.masks.get(camera.name)
            if mask is None:
                continue
            gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
            features = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=.02,
                                              minDistance=5, mask=np.uint8(mask)*255)
            if features is None:
                continue
            uv, depth = project(points, camera)
            usable = np.isfinite(uv).all(1) & (depth > 0)
            visible_uv, visible_local = uv[usable], self.reference_points[usable]
            if not len(visible_uv):
                continue
            selected_uv, selected_local, used = [], [], set()
            for pixel in features[:, 0]:
                distances = np.linalg.norm(visible_uv-pixel, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= 3 and index not in used:
                    # Track the actual RGB feature, retain its nearby predicted
                    # surface position; the residual remains part of PnP quality.
                    selected_uv.append(pixel); selected_local.append(visible_local[index]); used.add(index)
            if len(selected_uv) >= 6:
                self.tracks[camera.name] = _Track(gray, np.float32(selected_uv), np.float64(selected_local))
        self.last_observation = view.observation
        self.report = {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                       "object_id": object_id, "views": len(self.tracks),
                       "reason": "awaiting_observed_relative_motion"}
        return self.report

    def observe(self, view, frame, gripper_command):
        import cv2
        if self.object_id is None or view.observation is self.last_observation:
            return self.report
        self.last_observation = view.observation
        grip = grip_pose(view.state)
        estimates, detail, next_tracks = [], [], {}
        for camera in frame.cameras:
            track = self.tracks.get(camera.name)
            if track is None or len(track.pixels) < 6:
                continue
            gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
            old = track.pixels.reshape(-1, 1, 2)
            new, ok, _ = cv2.calcOpticalFlowPyrLK(track.gray, gray, old, None,
                                                winSize=(21, 21), maxLevel=3)
            if new is None or ok is None:
                continue
            back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, track.gray, new, None,
                                                      winSize=(21, 21), maxLevel=3)
            if back is None or back_ok is None:
                continue
            good = ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool)
            good &= np.linalg.norm(back[:, 0]-old[:, 0], axis=1) < 1.5
            good &= np.isfinite(new[:, 0]).all(1)
            pixels, local = new[good, 0], track.local[good]
            h, w = gray.shape
            inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
            pixels, local = pixels[inside], local[inside]
            if len(pixels) < 6:
                continue
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.float64(local), np.float64(pixels), camera.intrinsics, None,
                iterationsCount=60, reprojectionError=3., flags=cv2.SOLVEPNP_EPNP)
            if not success or inliers is None or len(inliers) < 6:
                continue
            indices = inliers[:, 0]
            local, pixels = local[indices], pixels[indices]
            next_tracks[camera.name] = _Track(gray, np.float32(pixels), local)
            camera_from_reference = np.eye(4)
            camera_from_reference[:3, :3] = cv2.Rodrigues(rvec)[0]
            camera_from_reference[:3, 3] = tvec[:, 0]
            observed_pose = np.linalg.inv(camera.world_to_camera) @ camera_from_reference
            predicted, z = project(transform_points(observed_pose, local), camera)
            residual = float(np.median(np.linalg.norm(predicted-pixels, axis=1)))
            if residual > 3 or np.any(z <= 0):
                continue
            attached_pose = grip if self.held is None else grip @ self.held["reference_in_gripper"]
            attached_pixels, _ = project(transform_points(attached_pose, local), camera)
            stationary_pixels, _ = project(transform_points(self.reference_pose, local), camera)
            detail[camera.name] = {"material_points": len(local), "pnp_residual_px": residual,
                                  "attached_error_px": float(np.median(np.linalg.norm(attached_pixels-pixels, axis=1))),
                                  "stationary_error_px": float(np.median(np.linalg.norm(stationary_pixels-pixels, axis=1)))}
            estimates.append((residual, observed_pose, camera.name))
        self.tracks = next_tracks
        if not estimates:
            self.attachment_votes = self.release_votes = 0
            if self.held is not None:
                self.held["confirmed"] = False
            self.report = {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                           "object_id": self.object_id, "reason": "material_tracking_unavailable"}
            return self.report
        estimates.sort(key=lambda item: item[0])
        pose = estimates[0][1]
        # Contradictory view estimates are not averaged into a confident pose.
        if any(np.linalg.norm(other[1][:3, 3]-pose[:3, 3]) > .02 for other in estimates[1:]):
            if self.held is not None:
                self.held["confirmed"] = False
            self.report = {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                           "object_id": self.object_id, "reason": "material_pose_views_disagree", "views": detail}
            return self.report
        relative = np.linalg.inv(grip) @ pose
        movement = np.linalg.norm(grip[:3, 3]-self.reference_pose[:3, 3])
        evidence = detail[estimates[0][2]]
        attached = movement >= .008 and evidence["attached_error_px"] <= 4
        attached &= evidence["stationary_error_px"] >= evidence["attached_error_px"] + 2
        self.attachment_votes = self.attachment_votes + 1 if attached and gripper_command == "close" else 0
        slip = "unknown"
        if self.held is not None and gripper_command == "close":
            previous_relative = self.held["reference_in_gripper"]
            drift = np.linalg.norm(relative[:3, 3]-previous_relative[:3, 3])
            rotation = np.arccos(np.clip((np.trace(relative[:3, :3] @ previous_relative[:3, :3].T)-1)/2, -1, 1))
            slip = "suspected" if drift > .012 or rotation > .15 else "not_observed"
            self.held["confirmed"] = slip == "not_observed"
        if self.contact_mode == "free_object" and (self.attachment_votes >= 2 or (self.held is not None and self.held["confirmed"])):
            self.held = {"object_id": self.object_id, "confirmed": True,
                         "points_in_gripper": transform_points(relative, self.reference_points),
                         "reference_in_gripper": relative, "source": "RGB_material_tracks_PnP",
                         "captured_at_s": frame.captured_at_s}
            if self.keep_upright:
                reference_up = self.reference_pose[:3, :3].T @ np.array([0., 0., 1.])
                self.held["upright_in_gripper"] = relative[:3, :3] @ reference_up
        stationary = self.last_pose is not None and np.linalg.norm(pose[:3, 3]-self.last_pose[:3, 3]) < .004
        if gripper_command == "open" and stationary:
            self.release_votes += 1
        else:
            self.release_votes = 0
        released = self.release_votes >= 3
        self.report = jsonable({"object_id": self.object_id,
                               "attachment": "supported" if self.held and self.held["confirmed"] and gripper_command == "close" else "unknown",
                               "slip": slip, "release": "visually_stationary_after_open" if released else "unknown",
                               "engagement": "visually_supported" if self.contact_mode == "articulated" and self.attachment_votes >= 2 else "unknown",
                               "reference_pose_world": pose, "reference_in_gripper": relative,
                               "views": detail, "hand_translation_since_seed_m": movement,
                               "reason": "conditional_rigid_material_correspondence",
                               "physical_contact_measured": False})
        self.last_pose, self.last_grip = pose, grip
        return self.report
