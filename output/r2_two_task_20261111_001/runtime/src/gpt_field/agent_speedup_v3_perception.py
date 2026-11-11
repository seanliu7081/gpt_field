"""Event-driven rigid displacement evidence from public calibrated stereo RGB.

Model boxes select material to track; they never supply feature depth. Metric
points require reciprocal descriptors, calibrated triangulation, temporal
forward/backward tracking in both views, and a spatially distributed rigid fit.
This observes a relative surface transform, not object completeness, emptiness,
contact, attachment, or task success. Unavailable evidence never repairs a pose.
"""
from __future__ import annotations

from copy import deepcopy
from time import monotonic

import cv2
import numpy as np

from .agent_speedup_v3_interaction import _frame, _mask, _project

VERSION = "agent-speedup-v3-local-rgb-geometry-1"
SOURCE = "public_two_view_feature_displacement"


def _ray(points, camera):
    points = np.asarray(points, float)
    directions = np.column_stack(((points[:, 0]-camera["principal"][0])/camera["focal"][0],
                                  -(points[:, 1]-camera["principal"][1])/camera["focal"][1],
                                  -np.ones(len(points)))) @ camera["rotation"].T
    return directions/np.linalg.norm(directions, axis=1)[:, None]


def _triangulate(a, b, cameras):
    """Closest calibrated rays; reject weak baseline and reprojection mismatch."""
    ca, cb = cameras
    da, db = _ray(a, ca), _ray(b, cb)
    cosine = np.sum(da*db, axis=1)
    sine = np.sqrt(np.maximum(0., 1.-cosine*cosine))
    baseline = cb["origin"]-ca["origin"]
    ad, bd = da@baseline, db@baseline
    denom = np.maximum(1e-12, 1.-cosine*cosine)
    ta, tb = (ad-cosine*bd)/denom, (cosine*ad-bd)/denom
    world = (ca["origin"]+ta[:, None]*da+cb["origin"]+tb[:, None]*db)/2
    pa, za = _project(world, ca)
    pb, zb = _project(world, cb)
    error = np.maximum(np.linalg.norm(pa-a, axis=1), np.linalg.norm(pb-b, axis=1))
    good = (sine > np.sin(np.deg2rad(2.))) & (ta > 0) & (tb > 0) & (za > 0) & (zb > 0) & (error <= 1.25)
    focal = min(*ca["focal"], *cb["focal"])
    uncertainty = np.maximum(.25, error)*np.maximum(ta, tb)/focal/np.maximum(sine, 1e-8)
    good &= np.isfinite(world).all(axis=1) & np.isfinite(uncertainty)
    return world, good, uncertainty, error


def _rigid(a, b):
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a-ac).T@(b-bc))
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T@u.T)
    rotation = vt.T@fix@u.T
    return rotation, bc-rotation@ac


def _fit_rigid(a, b, threshold):
    """Deterministic bounded consensus; planar material is sufficient, a line isn't."""
    if len(a) < 6:
        raise ValueError("insufficient_material_correspondences")
    rng = np.random.default_rng(0)
    best = None
    for _ in range(32):
        sample = rng.choice(len(a), 3, replace=False)
        if np.linalg.norm(np.cross(a[sample[1]]-a[sample[0]], a[sample[2]]-a[sample[0]])) < 1e-6:
            continue
        rotation, translation = _rigid(a[sample], b[sample])
        residual = np.linalg.norm(a@rotation.T+translation-b, axis=1)
        inliers = residual <= threshold
        score = (int(inliers.sum()), -float(np.median(residual)))
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[0][0] < max(6, int(np.ceil(.70*len(a)))):
        raise ValueError("nonrigid_or_ambiguous_feature_motion")
    keep = best[1]
    spread = np.linalg.svd(a[keep]-a[keep].mean(axis=0), compute_uv=False)/np.sqrt(keep.sum())
    if spread[1] < .004:
        raise ValueError("insufficient_spatial_material_extent")
    rotation, translation = _rigid(a[keep], b[keep])
    residual = np.linalg.norm(a[keep]@rotation.T+translation-b[keep], axis=1)
    return rotation, translation, keep, float(np.quantile(residual, .9)), float(spread[1])


class LocalRGBGeometryEvidence:
    """Sparse local references, only evaluated after a relevant visual event."""
    def __init__(self):
        self.references = {}
        self.results = {}
        self.compute_s = 0.
        self.assessments = 0
        self.captures = 0
        self._frames = {}

    def forget(self, ids):
        for ident in ids:
            self.references.pop(ident, None)
            self.results.pop(ident, None)

    def _frames_for(self, view):
        if self._frames.get("view") is view:
            return self._frames["frames"]
        frames = {}
        for name in ("external", "wrist"):
            raw = np.asarray(getattr(view, name+"_rgb_upright"))
            dimensions = view.observation.geometry["image"]
            if raw.dtype != np.uint8 or raw.shape != (dimensions["height"], dimensions["width"], 3):
                raise ValueError("image_calibration_mismatch")
            gray, camera = _frame(view, name)
            if (not np.isfinite(np.concatenate((camera["origin"], camera["rotation"].ravel(),
                                                camera["focal"], camera["principal"]))).all()
                    or np.any(camera["focal"] <= 0)
                    or not np.allclose(camera["rotation"].T@camera["rotation"], np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(camera["rotation"]), 1., atol=1e-6)):
                raise ValueError("invalid_calibration")
            frames[name] = gray, camera
        self._frames = {"view": view, "frames": frames, "features": {}}
        return frames

    @staticmethod
    def _report(ident, step, status, reason, **fields):
        return {"version": VERSION, "source": SOURCE, "entity_id": ident,
                "observed_at_step": int(step), "status": status, "reason": reason,
                "pose_updated": False, "physical_clearance_verified": False,
                "free_space_verified": False, "attachment_verified": False,
                "uncertainty_calibrated": False, **fields}

    def capture(self, ident, boxes, view, step, geometry_revision, *, occluder_boxes=()):
        began = monotonic()
        self.forget([ident])
        try:
            frames = self._frames_for(view)
            features = {}
            for name, (gray, camera) in frames.items():
                mask = cv2.erode(_mask(boxes, camera, gray.shape), np.ones((3, 3), np.uint8))
                occlusion = _mask(occluder_boxes, camera, gray.shape)
                mask[cv2.dilate(occlusion, np.ones((3, 3), np.uint8)) > 0] = 0
                if np.count_nonzero(mask) < 48:
                    raise ValueError("occluded_or_insufficient_visible_material")
                cached_features = self._frames["features"].get(name)
                if cached_features is None:
                    keypoints, descriptors = cv2.SIFT_create(nfeatures=500, contrastThreshold=.02).detectAndCompute(gray, None)
                    points = np.asarray([k.pt for k in keypoints]).reshape(-1, 2)
                    cached_features = points, descriptors
                    self._frames["features"][name] = cached_features
                points, descriptors = cached_features
                if descriptors is None or len(points) < 6:
                    raise ValueError("insufficient_texture")
                indices = np.clip(np.round(points).astype(int), [0, 0], [gray.shape[1]-1, gray.shape[0]-1])
                visible = mask[indices[:, 1], indices[:, 0]] > 0
                if visible.sum() < 6:
                    raise ValueError("insufficient_texture")
                features[name] = points[visible], descriptors[visible]
            ap, ad = features["external"]
            bp, bd = features["wrist"]
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            forward = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(ad, bd, k=2)
                       if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance}
            backward = {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(bd, ad, k=2)
                        if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance}
            pairs = [(a, b) for a, b in forward.items() if backward.get(b) == a]
            if len(pairs) < 6:
                raise ValueError("ambiguous_stereo_correspondence")
            a, b = np.array([ap[a] for a, _ in pairs]), np.array([bp[b] for _, b in pairs])
            world, good, uncertainty, _ = _triangulate(a, b, [frames[n][1] for n in ("external", "wrist")])
            inside = np.zeros(len(world), bool)
            for box in boxes:
                local = (world-box["center"])@box["rotation"]
                inside |= np.all(np.abs(local) <= np.asarray(box["half"])+float(box.get("uncertainty", 0.))+.003, axis=1)
            good &= inside
            if good.sum() < 6:
                raise ValueError("insufficient_triangulated_material_in_declared_object")
            self.references[ident] = {"step": int(step), "geometry_revision": int(geometry_revision),
                "frames": frames, "points": {"external": a[good], "wrist": b[good]},
                "world": world[good], "uncertainty": uncertainty[good],
                "center": np.mean([box["center"] for box in boxes], axis=0)}
            result = self._report(ident, step, "reference", "triangulated_material_reference",
                                  reference_step=int(step), reference_geometry_revision=int(geometry_revision),
                                  tracked_features=int(good.sum()))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, cv2.error) as exc:
            result = self._report(ident, step, "unavailable", str(exc)[:160])
        self.captures += 1
        self.compute_s += monotonic()-began
        return result

    def anchor_pose(self, ident, entity, view, step):
        """Translate a declared shape from an explicitly assigned material point.

        The local assignment remains a model hypothesis. Pixel similarity plus
        epipolar agreement checks the two-view match, not that assignment's truth.
        A single anchor cannot estimate shape, orientation or free space.
        """
        anchors = entity.get("anchors") or {}
        if anchors.get("anchor_local_m") is None or anchors.get("same_feature") is not True:
            return self._report(ident, step, "unavailable", "explicit_local_material_anchor_required")
        try:
            frames = self._frames_for(view)
            local = np.asarray(anchors["anchor_local_m"], float)
            # Imported lazily to avoid a geometry/perception import cycle.
            from .agent_speedup_v3_geometry import _rotation
            if not any(np.all(np.abs((local-np.asarray(box["center_local_m"])) @
                                         _rotation(box["quaternion_wxyz"])) <=
                                  np.asarray(box["half_extents_m"])+entity["uncertainty_m"])
                       for box in entity["boxes"]):
                raise ValueError("local_anchor_outside_declared_shape")
            points, patches = [], []
            width = view.observation.geometry["image"]["width"]
            height = view.observation.geometry["image"]["height"]
            for name in ("external", "wrist"):
                gray, camera = frames[name]
                pixel = np.asarray(anchors[name+"_uv"], float)*[gray.shape[1]/width, gray.shape[0]/height]
                if np.any(pixel < 5) or np.any(pixel > np.array(gray.shape[::-1])-6):
                    raise ValueError("anchor_out_of_view")
                patch = cv2.getRectSubPix(gray, (11, 11), tuple(map(float, pixel))).astype(float)
                patch -= patch.mean()
                if patch.std() < 8:
                    raise ValueError("anchor_insufficient_texture")
                points.append(pixel[None])
                patches.append(patch)
            world, good, uncertainty, error = _triangulate(*points, [frames[n][1] for n in ("external", "wrist")])
            if not good[0]:
                raise ValueError("anchor_triangulation_unavailable")
            rotation = _rotation(entity["quaternion_wxyz"])
            normals = []
            for box in entity["boxes"]:
                box_rotation = _rotation(box["quaternion_wxyz"])
                box_local = (local-np.asarray(box["center_local_m"]))@box_rotation
                half = np.asarray(box["half_extents_m"])
                if np.all(np.abs(box_local) <= half+entity["uncertainty_m"]):
                    for axis in range(3):
                        if abs(abs(box_local[axis])-half[axis]) <= entity["uncertainty_m"]+.003:
                            normals.append(rotation@box_rotation[:, axis])
            if not normals:
                raise ValueError("anchor_local_surface_orientation_unavailable")
            # Compare the same material under a calibrated plane homography.
            # A wrist-camera roll or perspective change does not imply an
            # appearance mismatch. The declared face normal is only a warp
            # hypothesis, accepted only when the actual material agrees.
            external_gray, ca = frames["external"]
            wrist_gray, cb = frames["wrist"]
            yy, xx = np.indices((35, 35), dtype=float)
            external_pixels = np.column_stack((xx.ravel()-17, yy.ravel()-17))+points[0][0]
            rays = _ray(external_pixels, ca)
            template = cv2.getRectSubPix(external_gray, (11, 11), tuple(map(float, points[0][0])))
            confirmed = []
            for normal in normals:
                denominator = rays@normal
                if np.min(np.abs(denominator)) < .1:
                    continue
                distance = float((world[0]-ca["origin"])@normal)/denominator
                warped_world = ca["origin"]+distance[:, None]*rays
                uv, depth = _project(warped_world, cb)
                if (np.any(distance <= 0) or np.any(depth <= 0) or np.any(uv < 1)
                        or np.any(uv > np.array(wrist_gray.shape[::-1])-2)):
                    continue
                search = cv2.remap(wrist_gray, uv[:, 0].reshape(35, 35).astype(np.float32),
                                   uv[:, 1].reshape(35, 35).astype(np.float32), cv2.INTER_LINEAR)
                scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
                _, best, _, peak = cv2.minMaxLoc(scores)
                center = np.array(scores.shape[::-1])/2-.5
                if best < .85 or np.linalg.norm(np.asarray(peak)-center) > 2.:
                    continue
                competing = scores.copy()
                px, py = peak
                competing[max(0, py-2):py+3, max(0, px-2):px+3] = -1
                # Retain the original strict uniqueness test after rectifying;
                # rotation support must not turn repeated texture into depth.
                if competing.max() <= .94*best:
                    confirmed.append(float(best))
            if not confirmed:
                raise ValueError("anchor_rectified_correspondence_unverified_or_ambiguous")
            similarity = max(confirmed)
            delta = world[0]-(np.asarray(entity["position_m"])+rotation@local)
            if np.linalg.norm(delta) > .25:
                raise ValueError("anchor_correction_exceeds_local_identity_window")
            return self._report(ident, step, "supported", "explicit_local_anchor_triangulated",
                reference_step=int(step), reference_geometry_revision=entity["geometry_revision"],
                evidence_id=f"{ident}:{entity['geometry_revision']}:{int(step)}:anchor",
                world_rotation_delta=np.eye(3).tolist(), world_translation_delta_m=delta.tolist(),
                center_displacement_m=delta.tolist(), position_uncertainty_m=float(uncertainty[0]),
                orientation_uncertainty_rad=0., anchor_local_m=local.tolist(),
                observed_anchor_world_m=world[0].tolist(), local_assignment_source="model_declared_material_point",
                orientation_observed=False, anchor_patch_similarity=similarity,
                anchor_correspondence_method="calibrated_local_surface_rectification_with_unique_material_peak",
                views={name: {"status": "supported", "reprojection_error_px": float(error[0])}
                       for name in ("external", "wrist")})
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, cv2.error) as exc:
            return self._report(ident, step, "tracking_unavailable", str(exc)[:160])

    def assess(self, ident, boxes, view, step, geometry_revision, *, occluder_boxes=()):
        cached = self.results.get(ident)
        if cached is not None and cached["observed_at_step"] == int(step):
            return deepcopy(cached)
        began = monotonic()
        ref = self.references.get(ident)
        fields = {} if ref is None else {"reference_step": ref["step"],
                                         "reference_geometry_revision": ref["geometry_revision"]}
        try:
            if ref is None:
                raise ValueError("no_triangulated_material_reference")
            if ref["geometry_revision"] != geometry_revision:
                raise ValueError("geometry_revision_changed")
            frames = self._frames_for(view)
            good = np.ones(len(ref["world"]), bool)
            tracked = {}
            fb_errors = []
            view_reports = {}
            for name, (gray, camera) in frames.items():
                old_gray = ref["frames"][name][0]
                old = ref["points"][name].astype(np.float32).reshape(-1, 1, 2)
                now, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, old, None, winSize=(21, 21), maxLevel=3)
                if now is None:
                    raise ValueError("material_tracking_lost")
                back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, now, None, winSize=(21, 21), maxLevel=3)
                if back is None:
                    raise ValueError("material_tracking_lost")
                points = now.reshape(-1, 2)
                error = np.linalg.norm(back.reshape(-1, 2)-old.reshape(-1, 2), axis=1)
                valid = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (error <= .75)
                valid &= (points[:, 0] >= 2) & (points[:, 0] < gray.shape[1]-2)
                valid &= (points[:, 1] >= 2) & (points[:, 1] < gray.shape[0]-2)
                occlusion = cv2.dilate(_mask(occluder_boxes, camera, gray.shape), np.ones((3, 3), np.uint8))
                index = np.clip(np.round(points).astype(int), [0, 0], [gray.shape[1]-1, gray.shape[0]-1])
                occluded = occlusion[index[:, 1], index[:, 0]] > 0
                valid &= ~occluded
                if valid.sum() < 6:
                    raise ValueError("occluded_material" if occluded.sum() > len(occluded)/2 else "material_tracking_lost")
                # Forward/backward flow can agree on repeated texture. Retain
                # descriptor identity through a normalized material-patch check.
                patch_ok = np.zeros(len(points), bool)
                for i in np.flatnonzero(valid):
                    before = cv2.getRectSubPix(old_gray, (9, 9), tuple(map(float, old[i, 0]))).astype(float)
                    after = cv2.getRectSubPix(gray, (9, 9), tuple(map(float, points[i]))).astype(float)
                    before -= before.mean(); after -= after.mean()
                    norm = np.linalg.norm(before)*np.linalg.norm(after)
                    patch_ok[i] = norm > 50 and float(np.sum(before*after)/norm) >= .70
                valid &= patch_ok
                good &= valid
                tracked[name] = points
                fb_errors.append(error)
                view_reports[name] = {"status": "tracked", "visible_material_features": int(valid.sum())}
            world, stereo_good, uncertainty, _ = _triangulate(tracked["external"], tracked["wrist"],
                                                             [frames[n][1] for n in ("external", "wrist")])
            good &= stereo_good
            if good.sum() < 6:
                raise ValueError("appearance_changed_without_rigid_correspondence")
            # Pixel triangulation uncertainty is already carried into the output
            # risk margin. Rigid-fit tolerance rejects inconsistent material.
            threshold = max(.002, min(.012, float(np.median(uncertainty[good]))))
            rotation, translation, fit_inliers, residual, spread = _fit_rigid(ref["world"][good], world[good], threshold)
            movement = rotation@ref["center"]+translation-ref["center"]
            angle = float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))
            if np.linalg.norm(movement) > .25 or angle > .6:
                raise ValueError("displacement_exceeds_local_identity_window")
            margin = float(np.quantile(np.maximum(uncertainty[good], ref["uncertainty"][good])[fit_inliers], .9))+residual
            angular_margin = min(.5, 2*residual/max(spread, .004))
            meaningful = np.linalg.norm(movement) > max(.001, 2*residual) or angle > max(.015, angular_margin)
            result = self._report(ident, step, "supported" if meaningful else "consistent",
                "two_view_rigid_material_displacement" if meaningful else "no_resolved_material_displacement",
                **fields, evidence_id=f"{ident}:{ref['geometry_revision']}:{ref['step']}:{int(step)}",
                world_rotation_delta=rotation.tolist(), world_translation_delta_m=translation.tolist(),
                center_displacement_m=movement.tolist(), position_uncertainty_m=margin,
                orientation_uncertainty_rad=angular_margin, rigid_residual_m=residual,
                inlier_features=int(fit_inliers.sum()), material_spread_m=spread,
                views={name: {**report, "status": "supported"} for name, report in view_reports.items()})
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
            reason = str(exc)[:160]
            status = "occluded" if "occlud" in reason else ("appearance_changed" if "appearance" in reason or "nonrigid" in reason else "tracking_unavailable")
            result = self._report(ident, step, status, reason, **fields)
        self.results[ident] = deepcopy(result)
        self.assessments += 1
        self.compute_s += monotonic()-began
        return result

    def performance_stats(self):
        return {"local_rgb_geometry_s": self.compute_s, "local_rgb_geometry_captures": self.captures,
                "local_rgb_geometry_assessments": self.assessments,
                "scope": "local event-driven perception only; no model or simulator calls"}
