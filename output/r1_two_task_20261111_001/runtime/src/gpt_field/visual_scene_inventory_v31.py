"""Explicit, current-observation revisions of remembered RGB scene hypotheses.

A revision is an auditable model claim, not a measurement of object geometry.
All declarations and revision records are checked before changing inventory
state. Omission/null geometry, tracker updates and target changes retain the
world hypothesis and do not make its metric estimate newer.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from .visual_scene_inventory_v17 import VisualSceneInventoryV17

_USE_PROVIDER = object()


class VisualSceneInventoryV31(VisualSceneInventoryV17):
    version = "explicit-scene-hypothesis-revisions-v31"
    geometry_equality_tolerance_m = 1e-9
    revision_evidence_kinds = frozenset({
        "current_two_view", "measured_motion_conflict", "observed_object_motion"})

    def __init__(self, revision_context_provider=None):
        super().__init__()
        self.revision_context_provider = revision_context_provider
        self.revision_log = []

    @staticmethod
    def _world(geometry, state):
        return {"center_xyz_m": (state[:3]+geometry["center_offset_m"]).tolist(),
                "half_extents_m": list(geometry["half_extents_m"]),
                "uncertainty_m": geometry["uncertainty_m"]}

    @classmethod
    def _same_geometry(cls, first, second):
        return all(np.max(np.abs(np.asarray(first[key])-np.asarray(second[key])))
                   <= cls.geometry_equality_tolerance_m
                   for key in ("center_xyz_m", "half_extents_m", "uncertainty_m"))

    def _revisions(self, entries, state, records=_USE_PROVIDER):
        if records is _USE_PROVIDER:
            records = ([] if self.revision_context_provider is None else
                       self.revision_context_provider())
        if not isinstance(records, list) or len(records) > self.maximum_entries_per_update:
            raise ValueError("scene_revisions must be a list of at most twelve records")
        revisions = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"id", "reason", "evidence_kind"}:
                raise ValueError("Scene revision fields invalid")
            identity, reason = record["id"], record["reason"]
            if (not isinstance(identity, str) or not identity.strip() or len(identity) > 80
                    or identity in revisions):
                raise ValueError("Scene revision identities must be unique stable inventory IDs")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
                raise ValueError("Scene revision needs a nonempty reason of at most 500 characters")
            if (not isinstance(record["evidence_kind"], str) or
                    record["evidence_kind"] not in self.revision_evidence_kinds):
                raise ValueError("Unknown scene revision evidence kind")
            revisions[identity] = copy.deepcopy(record)
        changed, unchanged, declared = {}, {}, {}
        for entry in entries:
            identity, geometry = entry["id"], entry["geometry"]
            if geometry is None:
                continue
            world = self._world(geometry, state)
            old = self.objects.get(identity, {}).get("world_geometry")
            declared[identity] = world
            if old is None:
                continue
            if self._same_geometry(old, world):
                unchanged[identity] = copy.deepcopy(old)
            else:
                changed[identity] = (copy.deepcopy(old), world)
                if identity not in revisions:
                    raise ValueError(f"Explicit scene revision required for changed geometry: {identity}")
                if revisions[identity]["evidence_kind"] == "current_two_view" and any(
                        entry[view+"_roi_uv"] is None for view in ("external", "wrist")):
                    raise ValueError("current_two_view revision requires both current declared RGB views")
        if set(revisions) != set(changed):
            raise ValueError("Scene revision is orphaned, stale, unchanged, or lacks current nonnull geometry")
        return revisions, changed, unchanged, declared

    def validate_revision_update(self, observation, entries, revisions):
        """Pure current-response preflight for the client's bounded repair path.

        This never consults the policy's previous-response provider, declares
        views, decodes images, or changes tracker/metric history. Geometry
        remains a self-reported hypothesis after successful validation.
        """
        self._validate(entries)
        state = self._state(observation["observation/state"] if isinstance(observation, dict)
                            else observation.state)
        self._revisions(entries, state, records=revisions)
        return True

    def configure(self, observation, entries):
        # Includes visibility, scalar finiteness and every member of a batch;
        # a later invalid revision must not partially apply an earlier entry.
        self._validate(entries)
        images, state = self._data(observation)
        revisions, changed, unchanged, declared = self._revisions(entries, state)
        old_geometry = {identity: copy.deepcopy(self.objects.get(identity, {}).get("world_geometry"))
                        for identity in declared}
        unchanged_history = {identity: (
            self.objects[identity]["geometry_revision_count"],
            copy.deepcopy(self.objects[identity]["geometry_revisions"])) for identity in unchanged}
        # Sub-nanometre serialization/rebasing differences cannot accumulate a
        # sequence of undeclared geometry edits: retain the last canonical box.
        super().configure(observation, copy.deepcopy(entries))
        for identity, world in unchanged.items():
            item = self.objects[identity]
            item["world_geometry"] = world
            item["geometry_revision_count"], item["geometry_revisions"] = unchanged_history[identity]
        binding = {"state8": state.tolist(), "observation_index": self.observation_index,
                   "update_index": self.updates,
                   "rgb_sha256": {view: hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                                  for view, image in images.items()}}
        declarations = {entry["id"]: entry for entry in entries}
        for identity in declared:
            item = self.objects[identity]
            item.update(estimated_at_observation=self.observation_index,
                        metric_declared_at_update=self.updates)
            if old_geometry[identity] is None or identity in changed:
                revision = revisions.get(identity)
                log = {"id": identity,
                       "kind": "initial_metric_declaration" if revision is None else "explicit_metric_revision",
                       "previous_world_geometry": old_geometry[identity],
                       "new_world_geometry": copy.deepcopy(item["world_geometry"]),
                       "revision": copy.deepcopy(revision), "observation_binding": copy.deepcopy(binding),
                       "current_declared_regions": {view+"_roi_uv": copy.deepcopy(
                           declarations[identity][view+"_roi_uv"]) for view in images},
                       "evidence_status": "model_reported_hypothesis_not_independently_verified",
                       "geometry_measured_or_certified": False,
                       "motion_allowances_or_evidence_changed": False}
                self.revision_log.append(log)
                item.update(metric_changed_at_observation=self.observation_index,
                            last_explicit_revision=copy.deepcopy(log) if revision is not None else None)
                # The inherited bounded history remains useful to old readers;
                # retain full revision metadata in the separate complete log.
                item["geometry_revisions"][-1]["explicit_revision"] = copy.deepcopy(revision)
        return self.summary()

    def _entries(self, state):
        rows = super()._entries(state)
        for row in rows:
            estimated = row.get("estimated_at_observation")
            declared = row.get("metric_declared_at_update")
            row["metric_age_observations"] = None if estimated is None else self.observation_index-estimated
            row["metric_age_updates"] = None if declared is None else self.updates-declared
            current = [v["last_observed_observation"] for v in row["views"].values()
                       if v["last_observed_observation"] is not None]
            row["last_seen_observation"] = max(current) if current else None
            row["metric_estimate_freshness_is_not_accuracy"] = True
        return rows

    def summary(self, state8=None):
        result = super().summary(state8)
        result.update(revision_log=copy.deepcopy(self.revision_log),
                      revision_log_complete=True,
                      geometry_equality_tolerance_m=self.geometry_equality_tolerance_m,
                      revision_semantics=("Changed remembered metric geometry requires a current visible "
                          "declaration and explicit revision record; evidence_kind is a model claim, "
                          "not a calibration, object measurement, or clearance certificate. Null/omission "
                          "retains the estimate and its age."))
        return result
