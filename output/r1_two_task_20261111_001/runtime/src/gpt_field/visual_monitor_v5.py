"""Persistent local RGB feature bundles for opt-in v5 field execution.

The model supplies an identity and an image seed, not a calibrated projection.
We track a *fixed* set of corners in a bounded neighborhood of that seed. No
corner is inserted after initialization, and lost bundles cannot reacquire a
nearby object. A consensus transform transports the original seed, rather than
silently replacing it with the strongest nearby corner. This is numerical
correspondence evidence, not object identity or obstacle-clearance certification.
"""
from __future__ import annotations

import itertools
import numpy as np

from .visual_field import _uv, _gray
from .visual_monitor_v4 import RoleAwareVisualMonitor, _orientation_distance


def _sample(pixels, points):
    """Bilinear samples; callers reject points outside the image first."""
    xy = np.asarray(points, float)
    x, y = xy[..., 0], xy[..., 1]
    if np.any(x < 0) or np.any(y < 0) or np.any(x >= pixels.shape[1]-1) or np.any(y >= pixels.shape[0]-1):
        return None
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = x-ix, y-iy
    return ((1-fx)*(1-fy)*pixels[iy, ix] + fx*(1-fy)*pixels[iy, ix+1]
            + (1-fx)*fy*pixels[iy+1, ix] + fx*fy*pixels[iy+1, ix+1])


def _ncc(first, second):
    if first is None or second is None:
        return -1.
    a, b = first-first.mean(), second-second.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator > 1e-8 else -1.


def _similarity(source, destination):
    """Least-squares oriented 2-D similarity, written as row-vector matrix."""
    src_mean, dst_mean = source.mean(axis=0), destination.mean(axis=0)
    src, dst = source-src_mean, destination-dst_mean
    energy = float(np.sum(src*src))
    if energy < 1e-8:
        return None
    a = float(np.sum(src*dst)/energy)
    b = float(np.sum(src[:, 0]*dst[:, 1]-src[:, 1]*dst[:, 0])/energy)
    matrix = np.array([[a, b], [-b, a]])
    return matrix, dst_mean-src_mean@matrix


class LocalFeatureBundle:
    """Fixed seed-local corners with FB matching and bounded robust consensus.

    Search is local to each previous correspondence. The support radius is ten
    source-image pixels by default, independent of the target/object size. This
    footprint is exposed in diagnostics so it must not be mistaken for a mask.
    Only initial features can survive or return: there is no automatic feature
    reseeding. An occluded initial feature may return only inside its predicted
    location, with FB agreement and its original appearance revalidated.
    """
    def __init__(self, pixels, uv, ident, *, support_radius_px=10., maximum_features=12, roi_uv=None):
        pixels = np.asarray(pixels, float)
        self.ident = ident
        self.scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], float)
        self.seed = _uv(uv)*self.scale
        self.position = self.seed.copy()
        self.roi = None
        if roi_uv is not None:
            roi = np.asarray(roi_uv, float)
            if (roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi < 0)
                    or np.any(roi > 1) or np.any(roi[:2] >= roi[2:])):
                raise ValueError("ROI must be an ordered image box")
            self.roi = roi.reshape(2, 2)*self.scale
            if np.any(self.seed < self.roi[0]) or np.any(self.seed > self.roi[1]):
                raise ValueError("seed must lie inside its declared ROI")
        self.support_radius_px = float(support_radius_px)
        if not 4 <= self.support_radius_px <= 16:
            raise ValueError("feature support radius must be within 4..16 pixels")
        self.offsets = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4)), axis=-1)
        self.initial_points = self._corners(pixels, maximum_features)
        if len(self.initial_points) < 3:
            raise ValueError("insufficient_seed_local_corners")
        self.initial_templates = [_sample(pixels, point+self.offsets) for point in self.initial_points]
        self.points = self.initial_points.copy()
        self.active = np.arange(len(self.points))
        self.previous_image = pixels.copy()
        self.matrix, self.translation = np.eye(2), np.zeros(2)
        self.reliable, self.score = True, 1.
        self.failure_reason = None
        self.last_inliers = len(self.points)
        self.last_residual_px = 0.

    def _corners(self, pixels, count):
        # Shi-Tomasi response over 3x3 gradients, requiring texture in both axes.
        gy, gx = np.gradient(pixels)
        def summed(values):
            return np.lib.stride_tricks.sliding_window_view(np.pad(values, 1), (3, 3)).sum(axis=(-1, -2))
        xx, yy, xy = summed(gx*gx), summed(gy*gy), summed(gx*gy)
        score = .5*(xx+yy-np.sqrt(np.maximum((xx-yy)**2+4*xy*xy, 0)))
        rows, cols = np.indices(pixels.shape)
        distance2 = (cols-self.seed[0])**2+(rows-self.seed[1])**2
        valid = ((distance2 <= self.support_radius_px**2) & (cols >= 4) & (rows >= 4)
                 & (cols < pixels.shape[1]-5) & (rows < pixels.shape[0]-5))
        if self.roi is not None:
            # Keep the entire descriptor inside the declared part at seeding.
            valid &= ((cols-3 >= self.roi[0, 0]) & (rows-3 >= self.roi[0, 1])
                      & (cols+3 <= self.roi[1, 0]) & (rows+3 <= self.roi[1, 1]))
        score = np.where(valid, score, 0.)
        threshold = max(70., float(score.max())*.035)
        selected = []
        for _ in range(count):
            row, col = np.unravel_index(np.argmax(score), score.shape)
            if score[row, col] < threshold:
                break
            selected.append([float(col), float(row)])
            score[(cols-col)**2+(rows-row)**2 < 3.**2] = 0.
        return np.asarray(selected, float).reshape(-1, 2)

    def _match(self, source, destination, point, *, search=7):
        template = _sample(source, point+self.offsets)
        if template is None or template.std() < 4:
            return None
        x, y = np.rint(point).astype(int)
        left, right = max(3, x-search), min(destination.shape[1]-5, x+search)
        top, bottom = max(3, y-search), min(destination.shape[0]-5, y+search)
        if right <= left or bottom <= top:
            return None
        patches = np.lib.stride_tricks.sliding_window_view(destination[top-3:bottom+4, left-3:right+4], (7, 7))
        normalized = template-template.mean()
        energy = np.maximum(np.sum(patches*patches, axis=(-1, -2))-patches.mean(axis=(-1, -2))**2*49, 0.)
        denominator = np.sqrt(energy)*np.linalg.norm(normalized)
        scores = np.divide(np.einsum("ijkl,kl->ij", patches, normalized), denominator,
                           out=np.full(energy.shape, -1.), where=denominator > 1e-8)
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[row, col])
        second = scores.copy()
        second[max(0, row-2):row+3, max(0, col-2):col+3] = -1.
        if score < .75 or score-second.max() < .025:
            return None
        # Subpixel localization removes the integer staircase which otherwise
        # corrupts the response learned from submillimeter observed robot steps.
        refined = np.array([left+col, top+row], float)
        for axis, coordinate in enumerate((col, row)):
            line = scores[row, :] if axis == 0 else scores[:, col]
            if 0 < coordinate < len(line)-1:
                a, b, c = line[coordinate-1:coordinate+2]
                curvature = a-2*b+c
                if curvature < -1e-8:
                    refined[axis] += np.clip(.5*(a-c)/curvature, -.5, .5)
        if np.max(np.abs(refined-point)) >= search-1:
            return None
        return refined, score

    def _fail(self, reason):
        self.reliable = False
        self.failure_reason = reason
        return False

    def track(self, pixels, search=7):
        # A failed identity cannot become reliable by matching a nearby object.
        if not self.reliable:
            return False
        pixels = np.asarray(pixels, float)
        if pixels.shape != self.previous_image.shape:
            return self._fail("image_shape_changed")
        matched, indices, scores = [], [], []
        predicted_points = self.initial_points@self.matrix+self.translation
        predicted_points[self.active] = self.points
        for index, point in enumerate(predicted_points):
            forward = self._match(self.previous_image, pixels, point, search=search)
            if forward is None:
                continue
            backward = self._match(pixels, self.previous_image, forward[0], search=search)
            if backward is None or np.linalg.norm(backward[0]-point) > .9:
                continue
            indices.append(index)
            matched.append(forward[0])
            scores.append(forward[1])
        if len(matched) < 3:
            return self._fail("insufficient_forward_backward_correspondences")
        indices, destination = np.asarray(indices), np.asarray(matched)
        source = self.initial_points[indices]
        best, best_error = None, float("inf")
        # At most 66 deterministic hypotheses. No random RANSAC state or new
        # points; all candidates refer to the initial declared seed footprint.
        for first, second in itertools.combinations(range(len(source)), 2):
            if np.linalg.norm(source[first]-source[second]) < 3:
                continue
            fitted = _similarity(source[[first, second]], destination[[first, second]])
            if fitted is None:
                continue
            matrix, shift = fitted
            length = np.linalg.norm(matrix[0])
            if not .7 <= length <= 1.4:
                continue
            error = np.linalg.norm(source@matrix+shift-destination, axis=1)
            mask = error <= 1.5
            count = int(mask.sum())
            residual = float(error[mask].sum())
            if count >= 3 and (best is None or count > best.sum() or count == best.sum() and residual < best_error):
                best, best_error = mask, residual
        if best is None or best.sum() < max(3, int(np.ceil(len(source)*.6))):
            return self._fail("inconsistent_feature_motion")
        matrix, shift = _similarity(source[best], destination[best])
        relative = np.linalg.solve(self.matrix, matrix)
        if not .88 <= np.linalg.norm(relative[0]) <= 1.12 or abs(np.arctan2(relative[0, 1], relative[0, 0])) > .15:
            return self._fail("abrupt_feature_transform")
        # Check each survivor against its ORIGINAL descriptor sampled through
        # the shared similarity transform. Updating only frame-to-frame would
        # gradually drift onto a different foreground/background feature.
        keep = best.copy()
        for n in np.flatnonzero(best):
            original = self.initial_templates[indices[n]]
            warped = _sample(pixels, (source[n]+self.offsets)@matrix+shift)
            if _ncc(original, warped) < .6:
                keep[n] = False
        if keep.sum() < 3:
            return self._fail("original_appearance_changed")
        matrix, shift = _similarity(source[keep], destination[keep])
        new_position = self.seed@matrix+shift
        if np.any(new_position < 0) or np.any(new_position > self.scale):
            return self._fail("seed_left_image")
        self.active, self.points = indices[keep], destination[keep]
        self.matrix, self.translation = matrix, shift
        self.position, self.previous_image = new_position, pixels.copy()
        self.score = float(np.median(np.asarray(scores)[keep]))
        self.last_residual_px = float(np.sqrt(np.mean(np.sum((source[keep]@matrix+shift-destination[keep])**2, axis=1))))
        self.last_inliers = int(keep.sum())
        self.failure_reason = None
        return True

    @property
    def uv(self):
        return self.position/self.scale

    def summary(self):
        return {"identity": self.ident, "reliable": self.reliable,
                "support_radius_px": self.support_radius_px,
                "seed_roi_uv": None if self.roi is None else (self.roi/self.scale).reshape(4).tolist(),
                "initial_features": len(self.initial_points), "surviving_features": len(self.active),
                "inliers": self.last_inliers, "residual_px": self.last_residual_px,
                "failure_reason": self.failure_reason, "automatic_reseeding": False,
                "semantic_correspondence_certified": False}


class RoleAwareVisualMonitorV5(RoleAwareVisualMonitor):
    """V4 evidence and SO(3)/budget checks, using persistent RGB bundles."""
    feature_bundle_class = LocalFeatureBundle
    def __init__(self):
        super().__init__()
        self._held_out_motion = []

    def configure(self, images, references, state8, *, intent_id=None):
        self._patch_references = references
        self._patch_external_image = _gray(images["external"])
        self._patch_wrist_image = _gray(images["wrist"])
        external = references.get("external") or {}
        for name, region in (("moving", "moving_roi_uv"), ("target", "target_roi_uv"),
                             ("tool", "tool_roi_uv")):
            track = self.tracks.get(name)
            roi = external.get(region)
            if track is not None and track.reliable and roi is not None:
                box = np.asarray(roi, float).reshape(2, 2)*track.scale
                # A new model-declared part cannot silently inherit feature
                # correspondences that now lie outside that part.
                if np.any(track.points < box[0]) or np.any(track.points > box[1]):
                    track._fail("updated_roi_excludes_persistent_features")
        previous_epoch = self.epoch
        super().configure(images, references, state8, intent_id=intent_id)
        if self.epoch != previous_epoch:
            self._held_out_motion = []
        return self.summary()

    def observe(self, images, state8, action):
        state = np.asarray(state8, float)
        delta = state[:3]-self.state[:3]
        domain_changed = (_orientation_distance(state[3:6], self.state[3:6]) > .035
                          or np.max(np.abs(state[6:8]-self.state[6:8])) > .002)
        prediction = None if domain_changed else self.response.predict(delta)
        previous = {name: self.tracks[name].position.copy() for name in ("moving", "target")
                    if self.tracks.get(name) is not None and self.tracks[name].reliable}
        if domain_changed:
            self._held_out_motion = []
        super().observe(images, state8, action)
        # The prediction was made BEFORE the current transition entered the fit.
        # Beating a zero-image-motion predictor prevents low-error static tracks
        # from being accepted solely by the permissive absolute residual limit.
        if (prediction is not None and len(previous) == 2 and self._external_reliable()
                and np.linalg.norm(delta) > .0001):
            observed = ((self.tracks["moving"].position-previous["moving"])
                        -(self.tracks["target"].position-previous["target"]))
            self._held_out_motion.append((float(np.dot(observed, observed)),
                                          float(np.sum((observed-prediction)**2))))
            self._held_out_motion = self._held_out_motion[-12:]
        if self.status == "response_validated" and not self.response_ready:
            self.status = "awaiting_predictive_motion_evidence"
        elif self.status == "observing" and self.response_ready:
            self.status = "response_validated"
        return self.summary()

    @property
    def feature_motion_threshold_px(self):
        track = self.tracks.get("moving")
        if track is None or not track.reliable:
            return 1.
        # Consensus scatter is per-feature; divided by sqrt(inliers) it gives
        # a local anchor precision estimate. The floor avoids fictitious exact
        # pixel measurements on nearly identical frames.
        noise = max(.08, track.last_residual_px/np.sqrt(max(3, track.last_inliers)))
        return max(.35, 3*noise)

    @property
    def response_ready(self):
        if not self._external_reliable() or not super().response_ready or len(self._held_out_motion) < 3:
            return False
        null_error, prediction_error = np.sum(self._held_out_motion, axis=0)
        return bool(null_error > 1e-8 and prediction_error <= .7*null_error)

    def _patch(self, pixels, point, identity):
        if point is None:
            return None
        candidates = []
        for view, expected in (("external", self._patch_external_image), ("wrist", self._patch_wrist_image)):
            ref = self._patch_references.get(view) or {}
            if not np.array_equal(pixels, expected):
                continue
            for seed, region, ident in (("moving_point_uv", "moving_roi_uv", ref.get("moving_id")),
                                       ("target_anchor_uv", "target_roi_uv", ref.get("target_id")),
                                       ("tool_point_uv", "tool_roi_uv", "independent_tool")):
                if identity == ident and ref.get(seed) is not None and np.array_equal(ref[seed], point) and ref.get(region) is not None:
                    candidates.append(np.asarray(ref[region], float))
        roi = None
        if candidates:
            # Identical synthetic images/seeds can match both views. Their
            # intersection respects both declared footprints rather than
            # accidentally selecting one view's region for the other.
            roi = np.r_[np.max([v[:2] for v in candidates], axis=0),
                        np.min([v[2:] for v in candidates], axis=0)]
        try:
            return self.feature_bundle_class(pixels, point, identity, roi_uv=roi,
                                             support_radius_px=16 if roi is not None else 10)
        except ValueError:
            return None

    def summary(self):
        result = super().summary()
        result["version"] = "role-aware-rgb-bundles-v5"
        result["tracking"] = {key: None if track is None else track.summary()
                              for key, track in self.tracks.items()}
        result["feature_motion_threshold_px"] = self.feature_motion_threshold_px
        result["held_out_motion_samples"] = len(self._held_out_motion)
        errors = np.sum(self._held_out_motion, axis=0) if self._held_out_motion else [0., 0.]
        result["held_out_zero_motion_squared_error"] = float(errors[0])
        result["held_out_prediction_squared_error"] = float(errors[1])
        return result
