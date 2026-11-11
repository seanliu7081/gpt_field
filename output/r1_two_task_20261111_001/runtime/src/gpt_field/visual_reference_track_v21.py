"""Bounded appearance maintenance for a fixed, model-selected RGB support.

This is an opt-in correspondence tracker, not a robot-part detector. Appearance
anchors may only be taken from frames that independently match the immutable
seed. Inactive original corners may return only through the legacy immutable
seed check, never an adaptive template. No physical evidence/budget changes.
"""
from __future__ import annotations

import itertools
import numpy as np

from .visual_monitor_v5 import LocalFeatureBundle, _ncc, _sample, _similarity


class PartFeatureTrackV21(LocalFeatureBundle):
    """Fixed identities, temporal FB geometry, and finite seed-verified anchors.

    A recent frame can explain modest shading/view changes without requiring
    every descriptor to retain its first-frame appearance. It cannot create an
    indefinitely self-validating template chain: every stored anchor descriptor
    individually passes the unchanged seed NCC test, and anchors expire by age
    and image displacement. Semantic identity remains a model hypothesis.
    """

    maximum_keyframes = 2
    minimum_keyframe_interval = 4
    maximum_anchor_age = 48
    maximum_anchor_displacement_px = 12.
    maximum_keyframe_updates = 16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_templates = tuple(x.copy() for x in self.initial_templates)
        for template in self.seed_templates:
            template.setflags(write=False)
        self.last_good_templates = {
            i: x.copy() for i, x in enumerate(self.seed_templates)}
        self.appearance_keyframes = []
        self.accepted_image_transitions = 0
        self.keyframe_updates = 0
        self.last_keyframe_transition = 0
        self.last_keyframe_position = self.position.copy()
        self.appearance_assisted_transitions = 0
        self.last_seed_inliers = self.last_inliers
        self.last_anchor_inliers = self.last_inliers
        self.last_temporal_inliers = self.last_inliers
        self.seed_verified_reactivations = 0

    def track(self, pixels, search=7):
        if not self.reliable:
            return False
        pixels = np.asarray(pixels, float)
        if pixels.shape != self.previous_image.shape:
            return self._fail("image_shape_changed")
        # configure/retain can present the identical observation repeatedly.
        # This creates neither a new appearance anchor nor extra support age.
        if np.array_equal(pixels, self.previous_image):
            return True
        transition = self.accepted_image_transitions + 1
        matched, indices, scores = [], [], []
        # Preserve legacy recovery of ORIGINAL inactive points, but only through
        # immutable seed verification below. Adaptive templates cannot recover
        # them or transfer support to new corners/nearby objects.
        previous_active = set(map(int, self.active))
        predicted_points = self.initial_points@self.matrix+self.translation
        predicted_points[self.active] = self.points
        for index, point in enumerate(predicted_points):
            forward = self._match(self.previous_image, pixels, point, search=search)
            if forward is None:
                continue
            backward = self._match(pixels, self.previous_image, forward[0], search=search)
            if backward is None or np.linalg.norm(backward[0]-point) > .9:
                continue
            indices.append(int(index))
            matched.append(forward[0])
            scores.append(forward[1])
        if len(matched) < 3:
            return self._fail("insufficient_forward_backward_correspondences")
        indices, destination = np.asarray(indices), np.asarray(matched)
        source = self.initial_points[indices]
        best, best_error = None, float("inf")
        for first, second in itertools.combinations(range(len(source)), 2):
            if np.linalg.norm(source[first]-source[second]) < 3:
                continue
            fitted = _similarity(source[[first, second]], destination[[first, second]])
            if fitted is None:
                continue
            matrix, shift = fitted
            if not .7 <= np.linalg.norm(matrix[0]) <= 1.4:
                continue
            error = np.linalg.norm(source@matrix+shift-destination, axis=1)
            mask = error <= 1.5
            count, residual = int(mask.sum()), float(error[mask].sum())
            if count >= 3 and (best is None or count > best.sum()
                              or count == best.sum() and residual < best_error):
                best, best_error = mask, residual
        if best is None or best.sum() < max(3, int(np.ceil(len(source)*.6))):
            return self._fail("inconsistent_feature_motion")
        matrix, shift = _similarity(source[best], destination[best])
        relative = np.linalg.solve(self.matrix, matrix)
        if (not .88 <= np.linalg.norm(relative[0]) <= 1.12
                or abs(np.arctan2(relative[0, 1], relative[0, 0])) > .15):
            return self._fail("abrupt_feature_transform")
        projected_seed = self.seed@matrix+shift
        anchors = [anchor for anchor in self.appearance_keyframes
                   if transition-anchor["transition"] <= self.maximum_anchor_age
                   and np.linalg.norm(projected_seed-anchor["position"])
                   <= self.maximum_anchor_displacement_px]
        keep, seed_keep = best.copy(), np.zeros(len(best), bool)
        temporal_count = 0
        for n in np.flatnonzero(best):
            index = int(indices[n])
            warped = _sample(pixels, (source[n]+self.offsets)@matrix+shift)
            seed_ok = _ncc(self.seed_templates[index], warped) >= .6
            seed_keep[n] = seed_ok
            # Last-good templates are canonicalized using the same immutable
            # geometry. They may be refreshed only after full acceptance.
            temporal_ok = (_ncc(self.last_good_templates.get(index), warped) >= .6
                           if index in previous_active else seed_ok)
            temporal_count += int(temporal_ok)
            anchor_ok = seed_ok or (index in previous_active and any(
                _ncc(anchor["templates"].get(index), warped) >= .6 for anchor in anchors))
            keep[n] = bool(temporal_ok and anchor_ok)
        self.last_seed_inliers = int((best & seed_keep).sum())
        self.last_anchor_inliers = int(keep.sum())
        self.last_temporal_inliers = temporal_count
        if keep.sum() < 3:
            return self._fail("bounded_appearance_support_lost")
        matrix, shift = _similarity(source[keep], destination[keep])
        new_position = self.seed@matrix+shift
        if np.any(new_position < 0) or np.any(new_position > self.scale):
            return self._fail("seed_left_image")
        # Recompute accepted descriptors after final consensus refinement. A
        # keyframe stores only descriptors directly verified against the seed.
        new_templates = {int(indices[n]): _sample(
            pixels, (source[n]+self.offsets)@matrix+shift) for n in np.flatnonzero(keep)}
        if any(value is None for value in new_templates.values()):
            return self._fail("descriptor_left_image")
        directly_verified = {index: value.copy() for index, value in new_templates.items()
                             if index in previous_active
                             and _ncc(self.seed_templates[index], value) >= .6}
        if (len(directly_verified) >= 3 and self.keyframe_updates < self.maximum_keyframe_updates
                and transition-self.last_keyframe_transition >= self.minimum_keyframe_interval
                and np.linalg.norm(new_position-self.last_keyframe_position) >= .5):
            anchors.append({"transition": transition, "position": new_position.copy(),
                            "templates": directly_verified})
            anchors = anchors[-self.maximum_keyframes:]
            self.keyframe_updates += 1
            self.last_keyframe_transition = transition
            self.last_keyframe_position = new_position.copy()
        self.appearance_keyframes = anchors
        self.seed_verified_reactivations += len(set(map(int, indices[keep]))-previous_active)
        self.active, self.points = indices[keep], destination[keep]
        self.matrix, self.translation = matrix, shift
        self.position, self.previous_image = new_position, pixels.copy()
        self.last_good_templates = {
            i: (x if i in previous_active else self.seed_templates[i]).copy()
            for i, x in new_templates.items()}
        self.score = float(np.median(np.asarray(scores)[keep]))
        self.last_residual_px = float(np.sqrt(np.mean(np.sum(
            (source[keep]@matrix+shift-destination[keep])**2, axis=1))))
        self.last_inliers = int(keep.sum())
        self.accepted_image_transitions = transition
        self.appearance_assisted_transitions += int(self.last_seed_inliers < 3)
        self.failure_reason = None
        return True

    def summary(self):
        result = super().summary()
        result["appearance_maintenance"] = {
            "version": "bounded-seed-verified-appearance-v21",
            "seed_templates_immutable": True,
            "new_features": False,
            "inactive_recovery_requires_immutable_seed": True,
            "accepted_image_transitions": self.accepted_image_transitions,
            "keyframe_count": len(self.appearance_keyframes),
            "keyframe_updates": self.keyframe_updates,
            "maximum_keyframe_updates": self.maximum_keyframe_updates,
            "maximum_anchor_age": self.maximum_anchor_age,
            "maximum_anchor_displacement_px": self.maximum_anchor_displacement_px,
            "seed_inliers": self.last_seed_inliers,
            "bounded_anchor_inliers": self.last_anchor_inliers,
            "last_good_inliers": self.last_temporal_inliers,
            "appearance_assisted_transitions": self.appearance_assisted_transitions,
            "immutable_seed_verified_reactivations": self.seed_verified_reactivations,
            "independent_robot_motion_or_semantic_evidence": False,
            "physical_budgets_modified": False,
        }
        return result
