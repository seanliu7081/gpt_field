"""Current Field geometry contract for finite observed-motion reversal."""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

from .visual_field_v20 import MetricFieldExecutorV20
from .visual_scene_guard_v17 import SceneCoverageGuardV17


class MetricFieldExecutorV21(MetricFieldExecutorV20):
    def nominal_regions_for_segment(self, start, end):
        records = super().nominal_regions_for_segment(start, end)
        guard, monitor = self.translation_guard, self.visual_monitor
        if guard is None or guard.state is None:
            return records
        state = np.asarray(guard.state, float)
        contract = self._retreat_geometry_contract(state, np.asarray(start, float), np.asarray(end, float))
        for row in records:
            row["directional_retreat_contract"] = copy.deepcopy(contract)
        return records

    def _retreat_geometry_contract(self, state, start, end):
        material = self._transit_materialization or {}
        objects = material.get("objects") or []
        target = self.target_configuration or {}
        carried = self.contact_geometry.carried_summary(state) if self.contact_geometry else {}
        canonical = [{"id": row.get("id"), "world_geometry": row.get("world_geometry")}
                     for row in objects]
        signature = hashlib.sha256(json.dumps({"objects": canonical,
            "intent_uncertainty_m": (self.intent or {}).get("uncertainty_m"),
            "tool_radius_m": (self.intent or {}).get("tool_radius_m"),
            "unknown_regions": (self.intent or {}).get("unknown_regions")},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        contract = {"version": "v21-current-empty-tool-retreat", "state8": state.tolist(),
            "start_xyz_m": start.tolist(), "end_xyz_m": end.tolist(),
            "observation_epoch": getattr(self.translation_guard, "sweep_observation_epoch", None),
            "geometry_sha256": signature, "scene_ids": [r.get("id") for r in objects],
            "target_id": target.get("target_id"), "target_contact": target.get("effective_allowed_contact"),
            "empty_tool": not carried.get("active") and not carried.get("candidate_ids"),
            "contact_or_visual_goal_dependency": bool(self.contact_refinement_enabled or self.goal_revisions),
            "full_scene_sweep_accepted": False, "outside_target_contact_band": False,
            "field_controller": True, "clearance_certified": False,
            "remaining_actions": 0 if self.intent is None else self.intent["max_actions"]-self.actions,
            "remaining_measured_travel_m": 0 if self.intent is None else self.intent["max_travel_m"]-self.measured_travel_m-self.observed_overshoot_margin_m}
        if (self.scene_assessor is None or not self._same(start, state[:3], (3,))
                or not objects or material.get("controlled_payload_excluded_id") is not None):
            return contract
        admission = self.scene_assessor(state, end-start)
        data = admission.get("diagnostics") or {}
        checks = data.get("sweep_checks") or []
        ids, checked_ids = contract["scene_ids"], [r.get("id") for r in checks]
        contract["full_scene_sweep_accepted"] = bool(admission.get("accepted")
            and data.get("coverage_scope") == "actual_commanded_whole_tool_sweep"
            and data.get("full_finger_range") is True
            and data.get("orientation_margin_rad", 0.) >= .04
            and self._same(data.get("start_xyz_m"), start, (3,))
            and self._same(data.get("end_xyz_m"), end, (3,))
            and len(ids) == len(set(ids)) and set(ids) == set(checked_ids)
            and len(checked_ids) == len(set(checked_ids))
            and set(data.get("checked_metric_ids", [])) == set(ids)
            and not data.get("coverage_issues") and not data.get("unresolved_far_image_ids")
            and all(r.get("world_geometry") is not None for r in objects)
            and all(r.get("full_uncertainty_sweep_disjoint") is True
                    and r.get("configuration_part_count", 0) >= 6 for r in checks))
        record = next((r for r in objects if r.get("id") == target.get("target_id")), None)
        if record is None or record.get("world_geometry") is None or not target.get("available"):
            return contract
        bare = copy.deepcopy(record); bare["world_geometry"]["uncertainty_m"] = 0.
        intent = copy.deepcopy(self.intent); intent["uncertainty_m"] = 0.
        boxes = SceneCoverageGuardV17._sweep_boxes(bare, state, {"field_intent": intent}, .04)
        if boxes:
            low, high = np.minimum(start, end), np.maximum(start, end)
            distance = min(float(np.linalg.norm(np.maximum(np.maximum(
                box["lower_xyz"]-high, low-box["upper_xyz"]), 0.))) for box in boxes)
            band = max(.03, float(target["sigma_m"]))
            contract.update(target_band_m=band, target_core_distance_lower_bound_m=distance,
                outside_target_contact_band=distance > band+1e-10)
        return contract

    def summary(self):
        result = super().summary()
        result["version"] = "candidate-bound-directional-retreat-field-v21"
        return result
