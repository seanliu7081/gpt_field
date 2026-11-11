"""Persistent RGB-declared scene hypotheses, independent of planner relevance.

No detector, depth, calibration or simulator object state is used. A model
revision is a hypothesis anchored to the measured EEF at that observation.
Omission, lost visibility, changed references and changed targets never delete
remembered geometry. Tracked rectangles remain uncertain identity claims.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np

from .observation import decode_image
from .visual_field import _gray
from .visual_monitor_v5 import LocalFeatureBundle


class VisualSceneInventoryV16:
    version = "persistent-observable-scene-inventory-v16"
    maximum_entries_per_update = 12
    # Pixel-coordinate quantization only; never a geometry/clearance margin.
    boundary_quantization_tolerance_px = .25

    def __init__(self):
        self.objects = {}
        self.state = None
        self._images = None
        self.observation_index = 0
        self.updates = 0
        self.transition_gaps = 0
        self.last_status = "not_configured"

    @staticmethod
    def _state(value):
        state = np.asarray(value, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Inventory needs finite measured state8")
        return state

    @classmethod
    def _data(cls, observation):
        if isinstance(observation, dict):
            state = observation["observation/state"]
            external, wrist = observation["observation/image"], observation["observation/wrist_image"]
        else:
            state, external, wrist = observation.state, observation.image, observation.wrist_image
        return {"external": np.asarray(decode_image(external)),
                "wrist": np.asarray(decode_image(wrist))}, cls._state(state)

    @staticmethod
    def _vector(value, length, low, high, name):
        if (not isinstance(value, (list, tuple)) or len(value) != length or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
            raise ValueError(f"Invalid {name}")
        array = np.asarray(value, float)
        if not np.isfinite(array).all() or np.any(array < low) or np.any(array > high):
            raise ValueError(f"Invalid {name}")
        return array

    @classmethod
    def _validate(cls, entries):
        if not isinstance(entries, list) or len(entries) > cls.maximum_entries_per_update:
            raise ValueError("Inventory update must contain at most twelve current declarations")
        identities = set()
        for item in entries:
            if not isinstance(item, dict) or set(item) != {"id", "external_roi_uv", "wrist_roi_uv", "geometry"}:
                raise ValueError("Inventory entry has missing or unexpected fields")
            identity = item["id"]
            if not isinstance(identity, str) or not identity.strip() or len(identity) > 80 or identity in identities:
                raise ValueError("Inventory identities must be unique nonempty stable IDs")
            identities.add(identity)
            visible = False
            for view in ("external", "wrist"):
                value = item[view+"_roi_uv"]
                if value is not None:
                    box = cls._vector(value, 4, 0., 1., view+" ROI")
                    if np.any(box[:2] >= box[2:]):
                        raise ValueError("Inventory ROI must have positive area")
                    visible = True
            if not visible:
                raise ValueError("A listed inventory revision requires a current visible view; omit unseen IDs to retain them")
            geometry = item["geometry"]
            if geometry is not None:
                if not isinstance(geometry, dict) or set(geometry) != {"center_offset_m", "half_extents_m", "uncertainty_m"}:
                    raise ValueError("Inventory geometry fields invalid")
                cls._vector(geometry["center_offset_m"], 3, -.5, .5, "center_offset_m")
                cls._vector(geometry["half_extents_m"], 3, .0001, .5, "half_extents_m")
                cls._vector([geometry["uncertainty_m"]], 1, 0., .1, "uncertainty_m")

    @staticmethod
    def _new_view():
        return {"track": None, "source": None, "current_roi_uv": None, "last_known_roi_uv": None,
            "status": "never_observed", "reliable": False, "last_declared_observation": None,
            "last_observed_observation": None, "projected_out_of_view_roi_uv": None,
            "projected_roi_uv": None, "boundary_quantization": None}

    @staticmethod
    def _corners(box):
        return np.array([box[:2], [box[2], box[1]], box[2:], [box[0], box[3]]])

    @staticmethod
    def _seed(pixels, roi, identity):
        candidates = []
        roi = np.asarray(roi, float)
        for fraction in ((.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            try:
                candidates.append(LocalFeatureBundle(pixels, roi[:2]+(roi[2:]-roi[:2])*fraction,
                    identity, support_radius_px=12., maximum_features=8, roi_uv=roi))
            except ValueError:
                pass
        return max(candidates, key=lambda track: track.last_inliers) if candidates else None

    def _declare_view(self, record, image, roi, identity, view):
        if roi is None:
            # Explicit current invisibility cannot silently preserve a claimed
            # current rectangle. Its last rectangle and metric risk remain.
            record.update(track=None, source=None, current_roi_uv=None, reliable=False,
                          status="not_visible_in_current_declaration", boundary_quantization=None)
            return
        pixels = _gray(image)
        scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1])
        box = np.asarray(roi)*np.tile(scale, 2)
        track = record["track"]
        compatible = (track is not None and track.reliable and
            np.all(track.points-3 >= box[:2]) and np.all(track.points+3 <= box[2:]))
        if not compatible:
            track = self._seed(pixels, roi, f"inventory:{view}:{identity}")
        source = None if track is None else (self._corners(box)-track.translation)@np.linalg.inv(track.matrix)
        record.update(track=track, source=source, current_roi_uv=list(roi), last_known_roi_uv=list(roi),
            reliable=bool(track is not None and track.reliable), status="current_model_declaration",
            last_declared_observation=self.observation_index, last_observed_observation=self.observation_index,
            projected_out_of_view_roi_uv=None, projected_roi_uv=list(roi), boundary_quantization=None)

    def configure(self, observation, entries):
        self._validate(entries)  # Entire update is checked before any mutation.
        images, state = self._data(observation)
        continuous = (self.state is None or (np.allclose(state, self.state, atol=1e-9, rtol=0.) and
            all(np.array_equal(images[v], self._images[v]) for v in images)))
        if not continuous:
            self.transition_gaps += 1
            for item in self.objects.values():
                for record in item["views"].values():
                    record.update(track=None, source=None, current_roi_uv=None, reliable=False,
                                  status="unacknowledged_observation_gap", boundary_quantization=None)
        self.state, self._images = state.copy(), {k: v.copy() for k, v in images.items()}
        self.updates += 1
        for item in self.objects.values():
            item["omitted_from_latest_inventory"] = True
        for declaration in entries:
            identity = declaration["id"]
            if identity not in self.objects:
                self.objects[identity] = {"id": identity, "world_geometry": None,
                    "views": {v: self._new_view() for v in images}, "geometry_revisions": deque(maxlen=4),
                    "geometry_revision_count": 0, "first_seen_observation": self.observation_index}
            item = self.objects[identity]
            item.update(omitted_from_latest_inventory=False, last_declared_update=self.updates)
            geometry = declaration["geometry"]
            if geometry is not None:
                old = copy.deepcopy(item["world_geometry"])
                world = {"center_xyz_m": (state[:3]+geometry["center_offset_m"]).tolist(),
                    "half_extents_m": list(geometry["half_extents_m"]), "uncertainty_m": geometry["uncertainty_m"]}
                if old != world:
                    item["geometry_revision_count"] += 1
                    item["geometry_revisions"].append({"observation_index": self.observation_index,
                        "update_index": self.updates, "anchored_eef_xyz_m": state[:3].tolist(),
                        "previous_world_geometry": old, "new_world_geometry": copy.deepcopy(world),
                        "evidence": "current model-declared visible RGB rectangle; metric revision remains unverified"})
                item["world_geometry"] = world
            for view in images:
                self._declare_view(item["views"][view], images[view], declaration[view+"_roi_uv"], identity, view)
        self.last_status = "current_inventory_merged_without_deletion"
        return self.summary()

    def observe(self, before, after):
        before_images, before_state = self._data(before)
        images, state = self._data(after)
        if (self.state is None or not np.allclose(before_state, self.state, atol=1e-9, rtol=0.) or
                any(not np.array_equal(before_images[v], self._images[v]) for v in images)):
            self.last_status = "unacknowledged_transition_gap"
            return self.summary()
        self.observation_index += 1
        pixels = {v: _gray(im) for v, im in images.items()}
        for item in self.objects.values():
            for view, record in item["views"].items():
                record["boundary_quantization"] = None
                track = record["track"]
                if track is None or not track.reliable:
                    record.update(current_roi_uv=None, reliable=False)
                    if record["last_known_roi_uv"] is not None:
                        record["status"] = "last_known_rectangle_without_current_track"
                    continue
                if images[view].shape != before_images[view].shape or not track.track(pixels[view]):
                    record.update(current_roi_uv=None, reliable=False, status="visual_correspondence_lost")
                    continue
                points = record["source"]@track.matrix+track.translation
                scale = np.array([images[view].shape[1]-1, images[view].shape[0]-1])
                pixel_box = np.r_[points.min(axis=0), points.max(axis=0)]
                pixel_scale = np.tile(scale, 2)
                box = pixel_box/pixel_scale
                record["projected_roi_uv"] = box.tolist()  # Full unsnapped extent.
                excess = np.maximum(np.maximum(-pixel_box, pixel_box-pixel_scale), 0.)
                finite = bool(np.isfinite(box).all())
                within_quantization = finite and bool(np.all(excess <= self.boundary_quantization_tolerance_px))
                if not within_quantization:
                    record.update(current_roi_uv=None, reliable=False, status="whole_rectangle_not_fully_in_view",
                                  projected_out_of_view_roi_uv=box.tolist())
                    continue
                snapped = bool(np.any(excess > 0.))
                current = np.clip(box, 0., 1.) if snapped else box
                if snapped:
                    # Only normalize a <=quarter-pixel numerical edge overrun
                    # AFTER successful RGB tracking. Keep the full projected
                    # rectangle and all world geometry; this neither observes
                    # unseen surfaces nor certifies that the object is clear.
                    record["boundary_quantization"] = {
                        "applied": True, "tolerance_px": self.boundary_quantization_tolerance_px,
                        "coordinate_excess_px": excess.tolist(),
                        "maximum_excess_px": float(excess.max()),
                        "current_support_coordinates_snapped_to_image_boundary": True,
                        "full_projected_extent_preserved": True,
                        "world_geometry_changed": False, "clearance_certified": False}
                record.update(current_roi_uv=current.tolist(), last_known_roi_uv=current.tolist(), reliable=True,
                    status="current_affine_tracking_hypothesis", last_observed_observation=self.observation_index,
                    projected_out_of_view_roi_uv=box.tolist() if snapped else None)
        self.state, self._images = state.copy(), {v: im.copy() for v, im in images.items()}
        self.last_status = "acknowledged_rgb_robot_transition"
        return self.summary()

    def _entries(self, state):
        result = []
        for item in self.objects.values():
            row = {k: copy.deepcopy(v) for k, v in item.items() if k not in ("views", "geometry_revisions")}
            row["geometry_revisions"] = copy.deepcopy(list(item["geometry_revisions"]))
            geometry = row["world_geometry"]
            row["current_geometry"] = None if geometry is None else {
                "center_offset_m": (np.asarray(geometry["center_xyz_m"])-state[:3]).tolist(),
                "half_extents_m": copy.deepcopy(geometry["half_extents_m"]),
                "uncertainty_m": geometry["uncertainty_m"]}
            row["views"] = {v: {k: copy.deepcopy(value) for k, value in view.items() if k not in ("track", "source")}
                            for v, view in item["views"].items()}
            row.update(geometry_source="model_RGB_hypothesis_anchored_to_measured_EEF", semantic_identity_certified=False,
                       observed_missing_does_not_mean_absent=True)
            result.append(row)
        return result

    def materialize(self, state8, target_id, *, allowed_contact="none", controlled_payload_id=None):
        state = self._state(state8)
        if allowed_contact not in ("none", "fingers", "payload_support"):
            raise ValueError("Unknown target contact permission")
        entries = self._entries(state)
        result = {"objects": entries, "obstacles": [], "nearby_obstacles": [], "target_geometry": None,
            "missing_geometry_ids": [], "image_only_ids": [], "stale_external_ids": [], "lost_external_ids": [],
            "outside_legacy_center_domain_ids": [], "outside_legacy_bounds_domain_ids": [],
            "controlled_payload_excluded_id": controlled_payload_id,
            "controlled_payload_exclusion_requires_caller_observed_evidence": True,
            "target_id": target_id, "all_remembered_ids": [row["id"] for row in entries],
            "objects_deleted": False, "relevance_filter_applied": False, "clearance_certified": False}
        for row in entries:
            identity, geometry = row["id"], row["current_geometry"]
            external, wrist = row["views"]["external"], row["views"]["wrist"]
            ext_box = external["current_roi_uv"] or external["last_known_roi_uv"]
            wrist_box = wrist["current_roi_uv"] or wrist["last_known_roi_uv"]
            if external["current_roi_uv"] is None:
                result["stale_external_ids"].append(identity)
            if external["last_known_roi_uv"] is not None and not external["reliable"]:
                result["lost_external_ids"].append(identity)
            if ext_box is not None and identity != controlled_payload_id:
                result["nearby_obstacles"].append({"id": identity, "external_roi_uv": copy.deepcopy(ext_box),
                    "wrist_roi_uv": copy.deepcopy(wrist_box)})
            if geometry is None:
                result["missing_geometry_ids"].append(identity)
                result["image_only_ids"].append(identity)
                continue
            center, half = np.asarray(geometry["center_offset_m"]), np.asarray(geometry["half_extents_m"])
            low, high = center-half, center+half
            if np.any(abs(center) > .5):
                result["outside_legacy_center_domain_ids"].append(identity)
            if np.any(abs(low) > 1.) or np.any(abs(high) > 1.):
                result["outside_legacy_bounds_domain_ids"].append(identity)
            if identity == target_id:
                result["target_geometry"] = {"id": identity, **copy.deepcopy(geometry), "allowed_contact": allowed_contact}
            elif identity != controlled_payload_id:
                result["obstacles"].append({"id": identity, "lower_offset_m": low.tolist(),
                    "upper_offset_m": high.tolist(), "uncertainty_m": geometry["uncertainty_m"]})
        return result

    def summary(self, state8=None):
        state = self.state if state8 is None else self._state(state8)
        return {"version": self.version, "status": self.last_status,
            "entries": [] if state is None else self._entries(state), "update_count": self.updates,
            "observation_index": self.observation_index, "unacknowledged_observation_gaps": self.transition_gaps,
            "omission_semantics": "retain identity, metric hypothesis and last image risk; omission never means empty space",
            "revision_semantics": "nonnull geometry plus a current RGB declaration revises the measured-EEF-anchored model hypothesis; null retains it",
            "automatic_identity_merge": False, "automatic_deletion": False, "environment_inputs": [],
            "clearance_certified": False, "task_success_certified": False}
