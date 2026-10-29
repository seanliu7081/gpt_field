"""Current visible RGB support is distinct from complete object visibility.

V16's full unsnapped projection, world hypothesis and explicit identities are
preserved. A successfully tracked visible part can support object continuity
when the declared whole-object rectangle crosses an image edge. Such support
is neither observation of the missing extent nor a collision certificate.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_scene_inventory_v16 import VisualSceneInventoryV16


class VisualSceneInventoryV17(VisualSceneInventoryV16):
    version = "persistent-scene-visible-support-v17"
    minimum_support_inliers = 3
    descriptor_half_width_px = 3.
    minimum_visible_span_px = 7.

    @staticmethod
    def _new_view():
        record = VisualSceneInventoryV16._new_view()
        record.update(current_visible_roi_uv=None, partial_current_support=False,
                      support_status="unavailable", support_inlier_count=0,
                      support_observed_at=None, support_points_px=[],
                      support_is_complete_extent_observation=False,
                      support_is_clearance_certificate=False,
                      extent_visibility="unavailable", extent_touches_image_boundary=False,
                      declared_extent_touches_image_boundary=False)
        return record

    def _declare_view(self, record, image, roi, identity, view):
        super()._declare_view(record, image, roi, identity, view)
        record["declared_extent_touches_image_boundary"] = bool(roi is not None and
            (np.any(np.asarray(roi)[:2] <= 0.) or np.any(np.asarray(roi)[2:] >= 1.)))

    def _update_support(self, record, image):
        record.update(current_visible_roi_uv=None, partial_current_support=False,
                      support_status="unavailable", support_inlier_count=0,
                      support_observed_at=None, support_points_px=[],
                      extent_visibility="unavailable", extent_touches_image_boundary=False)
        track = record["track"]
        status = record["status"]
        if status not in ("current_model_declaration", "current_affine_tracking_hypothesis",
                          "whole_rectangle_not_fully_in_view"):
            record["support_status"] = "no_current_correspondence"
            return
        projection = record.get("projected_roi_uv")
        if projection is None:
            record["support_status"] = "projected_extent_unavailable"
            return
        full = np.asarray(projection, float)
        if full.shape != (4,) or not np.isfinite(full).all() or np.any(full[:2] >= full[2:]):
            record["support_status"] = "projected_extent_invalid"
            return
        touches = bool(np.any(full[:2] <= 0.) or np.any(full[2:] >= 1.))
        boundary_limited = touches or record["declared_extent_touches_image_boundary"]
        record.update(extent_touches_image_boundary=touches,
            extent_visibility="boundary_limited" if boundary_limited else "within_image")
        # Even a fresh untextured boundary declaration has incomplete extent;
        # a legacy nonnull current_roi must not grant full-support admission.
        if track is None or not track.reliable:
            record["support_status"] = "no_current_correspondence"
            return
        scale = np.array([image.shape[1]-1, image.shape[0]-1], float)
        # This is a visible INTERSECTION only. The full projection is never
        # overwritten, and this rectangle must not become a smaller world box.
        visible = np.r_[np.maximum(full[:2], 0.), np.minimum(full[2:], 1.)]
        span = (visible[2:]-visible[:2])*scale
        if np.any(span < self.minimum_visible_span_px):
            record["support_status"] = "visible_intersection_unusable"
            return
        points = np.asarray(track.points, float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            record["support_status"] = "current_feature_points_invalid"
            return
        lower, upper = visible[:2]*scale, visible[2:]*scale
        radius = self.descriptor_half_width_px
        supported = np.all(points-radius >= lower, axis=1) & np.all(points+radius <= upper, axis=1)
        count = min(int(supported.sum()), int(track.last_inliers))
        record["support_inlier_count"] = count
        record["support_points_px"] = points[supported].tolist()
        if count < self.minimum_support_inliers:
            record["support_status"] = "insufficient_current_visible_descriptor_support"
            return
        partial = boundary_limited or record["current_roi_uv"] is None
        record.update(current_visible_roi_uv=visible.tolist(), partial_current_support=partial,
            support_status=("partial_current_support_hypothesis" if partial else "current_visible_support_hypothesis"),
            support_observed_at=self.observation_index)

    def configure(self, observation, entries):
        super().configure(observation, entries)
        for item in self.objects.values():
            for view, record in item["views"].items():
                self._update_support(record, self._images[view])
        return self.summary()

    def observe(self, before, after):
        prior_index = self.observation_index
        super().observe(before, after)
        if self.observation_index == prior_index:
            # No acknowledged transition: do not describe old pixels as
            # current support for the supplied after-observation.
            for item in self.objects.values():
                for record in item["views"].values():
                    record.update(current_visible_roi_uv=None, partial_current_support=False,
                        support_status="unacknowledged_transition_gap", support_inlier_count=0,
                        support_observed_at=None, support_points_px=[],
                        extent_visibility="unavailable", extent_touches_image_boundary=False)
            return self.summary()
        for item in self.objects.values():
            for view, record in item["views"].items():
                self._update_support(record, self._images[view])
        return self.summary()

    def materialize(self, state8, target_id, *, allowed_contact="none", controlled_payload_id=None):
        result = super().materialize(state8, target_id, allowed_contact=allowed_contact,
                                     controlled_payload_id=controlled_payload_id)
        entries = {row["id"]: row for row in result["objects"]}
        result["partial_external_support_ids"] = []
        result["missing_current_external_support_ids"] = []
        for identity, row in entries.items():
            view = row["views"]["external"]
            if view["partial_current_support"]:
                result["partial_external_support_ids"].append(identity)
            if view["current_visible_roi_uv"] is None:
                result["missing_current_external_support_ids"].append(identity)
        for region in result["nearby_obstacles"]:
            row = entries[region["id"]]
            for name in ("external", "wrist"):
                view = row["views"][name]
                if view["partial_current_support"]:
                    region[name+"_roi_uv"] = copy.deepcopy(view["current_visible_roi_uv"])
                region[name+"_full_projected_roi_uv"] = copy.deepcopy(view["projected_roi_uv"])
                region[name+"_partial_current_support"] = view["partial_current_support"]
                region[name+"_support_status"] = view["support_status"]
                region[name+"_extent_visibility"] = view["extent_visibility"]
        result["visible_support_changes_world_geometry"] = False
        result["partial_support_requires_sweep_coverage_check"] = True
        return result

    def summary(self, state8=None):
        result = super().summary(state8)
        result["visible_support_semantics"] = {
            "current_roi_uv": "unchanged V16 whole-rectangle availability",
            "current_visible_roi_uv": "current in-frame intersection supported by persistent RGB descriptors",
            "projected_roi_uv": "full unsnapped projected extent; never replaced by the visible intersection",
            "extent_visibility": "boundary-limited declared extent stays incomplete until an explicit current interior declaration; successful feature tracking does not observe missing extent",
            "minimum_inliers": self.minimum_support_inliers,
            "minimum_visible_span_px": self.minimum_visible_span_px,
            "partial_support_observes_missing_extent": False,
            "partial_support_certifies_clearance": False,
            "automatic_reseeding": False,
            "changes_world_geometry_or_motion_allowance": False}
        return result
