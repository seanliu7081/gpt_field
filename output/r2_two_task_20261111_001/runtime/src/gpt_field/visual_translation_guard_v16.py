"""Persistent projected neighbors keyed by explicit scene-inventory identity.

Overlap is never an identity test: a target and an adjacent object can occupy
nearly the same image rectangle. V15's tracking, finite measured travel and
admission semantics are preserved; only the legacy neighbor-update adapter is
replaced. Declared identities and rectangles remain model hypotheses.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_field import _gray
from .visual_translation_guard_v11 import _box, _corners
from .visual_translation_guard_v15 import ObservedRiskTranslationGuardV15


class ObservedRiskTranslationGuardV16(ObservedRiskTranslationGuardV15):
    version = "persistent-explicit-scene-neighbors-v16"

    @staticmethod
    def _validate_context(context, target_id):
        context = context if isinstance(context, dict) else {}
        if _box(context.get("tool_external_roi_uv")) is None:
            return "translation_tool_envelope_unavailable", []
        mode, assessment = context.get("translation_mode"), context.get("obstacle_assessment")
        neighbors = context.get("nearby_obstacles")
        if (mode not in ("transit", "separate", "interaction")
                or assessment not in ("present", "none_relevant")
                or not isinstance(neighbors, list)
                or (assessment == "present" and not neighbors)
                or (assessment == "none_relevant" and neighbors)):
            return "translation_neighborhood_unobservable_or_inconsistent", []
        if ((mode == "interaction" and context.get("intended_contact_id") != target_id)
                or (mode != "interaction" and context.get("intended_contact_id") is not None)):
            return "translation_contact_role_inconsistent", []
        incoming, identities = [], set()
        for record in neighbors:
            if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                    or not record["id"].strip()):
                return "translation_neighbor_invalid", []
            identity = record["id"]
            box = _box(record.get("external_roi_uv"))
            if box is None:
                return "translation_neighbor_invalid", []
            if identity in identities:
                return "translation_neighbor_identity_duplicate", []
            identities.add(identity)
            incoming.append((identity, box))
        return None, incoming

    def configure(self, images, motion_context, state8, *, target_id, monitor=None):
        state = np.asarray(state8, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state8 must contain eight finite values")
        pixels = _gray(images["external"])
        scale = np.array([pixels.shape[1]-1, pixels.shape[0]-1], float)
        actual_context = copy.deepcopy(motion_context)
        invalid, incoming = self._validate_context(actual_context, target_id)
        if invalid is not None:
            # Reject the full declaration before updating any scene record.
            # A malformed refresh cannot erase risk or buy new travel.
            self.context, self.target_id = actual_context, target_id
            self.context_invalid = invalid
            self.state, self.scale = state.copy(), scale
            self._pending_cold = False
            self._experiment_pending = None
            self._motion_classification_pending = None
            self._regressing_escape = False
            return self.summary()

        self.scale = scale
        for identity, box in incoming:
            pixel_box = box*np.tile(scale, 2)
            old = next((row for row in self.neighbors if row["id"] == identity), None)
            if old is not None and old["track"] is not None and old["track"].reliable:
                prior = self._transport(old)
                combined = np.r_[np.minimum(prior[:2], pixel_box[:2]),
                                 np.maximum(prior[2:], pixel_box[2:])]
                old["box"] = combined
                old["source"] = ((_corners(combined)-old["track"].translation)
                                 @np.linalg.inv(old["track"].matrix))
                old["aliases"] = {identity}
                continue
            track = self._seed_neighbor(pixels, box, identity)
            record = {"id": identity, "aliases": {identity}, "box": pixel_box,
                      "track": track, "source": _corners(pixel_box)}
            if old is None:
                self.neighbors.append(record)
            else:
                # Lost texture does not prove that the old object extent vanished.
                prior = old["box"]
                record["box"] = np.r_[np.minimum(prior[:2], pixel_box[:2]),
                                      np.maximum(prior[2:], pixel_box[2:])]
                record["source"] = _corners(record["box"])
                index = next(i for i, row in enumerate(self.neighbors) if row is old)
                self.neighbors[index] = record

        # Skip only V11's IoU-based *update*. All real records remain installed
        # for inherited snapshots, risk assessment, tracking and debt accounting.
        adapter_context = copy.deepcopy(actual_context)
        adapter_context["nearby_obstacles"] = []
        adapter_context["obstacle_assessment"] = "none_relevant"
        try:
            super().configure(images, adapter_context, state, target_id=target_id,
                              monitor=monitor)
        finally:
            self.context = actual_context
        return self.summary()

    def summary(self):
        result = super().summary()
        result["neighbor_identity_contract"] = {
            "key": "exact_explicit_scene_inventory_id",
            "projected_overlap_merges_objects": False,
            "omission_deletes_objects": False,
            "context_refresh_replenishes_travel": False,
            "semantic_identity_certified": False,
        }
        return result


ProjectedTranslationGuardV16 = ObservedRiskTranslationGuardV16
