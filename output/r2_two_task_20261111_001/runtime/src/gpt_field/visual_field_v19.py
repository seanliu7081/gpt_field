"""Attach exact current whole-tool sweep evidence to each Field proposal."""
from __future__ import annotations

import numpy as np

from .visual_field_v17 import MetricFieldExecutorV17


class MetricFieldExecutorV19(MetricFieldExecutorV17):
    def nominal_regions_for_segment(self, start, end):
        records = super().nominal_regions_for_segment(start, end)
        guard = self.translation_guard
        if guard is None or self.scene_assessor is None or guard.state is None:
            return records
        state = np.asarray(guard.state, float).copy()
        start, end = np.asarray(start, float), np.asarray(end, float)
        if not np.allclose(start, state[:3], atol=1e-10, rtol=0.):
            return records
        admission = self.scene_assessor(state, end-start)
        details = admission.get("diagnostics") or {}
        try:
            checked_start = np.asarray(details.get("start_xyz_m"), float)
            checked_end = np.asarray(details.get("end_xyz_m"), float)
        except (ValueError, TypeError):
            return records
        if (details.get("coverage_scope") != "actual_commanded_whole_tool_sweep"
                or checked_start.shape != (3,) or checked_end.shape != (3,)
                or not np.allclose(checked_start, start, atol=1e-10, rtol=0.)
                or not np.allclose(checked_end, end, atol=1e-10, rtol=0.)):
            return records
        checks = {row["id"]: row for row in details.get("sweep_checks", [])}
        for record in records:
            check = checks.get(record["id"])
            if check is None or record.get("kind") == "unknown_regions":
                continue
            record["model_uncertainty_sweep"] = {
                "version": "v19-current-full-uncertainty-union", "object_id": record["id"],
                "observation_epoch": getattr(guard, "sweep_observation_epoch", None),
                "state8": state.tolist(), "start_xyz_m": start.tolist(), "end_xyz_m": end.tolist(),
                "scene_coverage_accepted": admission.get("accepted") is True,
                "coverage_scope": details["coverage_scope"],
                "full_finger_range": details.get("full_finger_range"),
                "orientation_margin_rad": details.get("orientation_margin_rad"),
                "configuration_part_count": check.get("configuration_part_count"),
                "full_uncertainty_sweep_disjoint": check.get("full_uncertainty_sweep_disjoint"),
                "image_support": check.get("image_support"),
                "carried_payload_hypothesis_in_sweep": details.get("carried_payload_hypothesis_in_sweep"),
                "clearance_certified": False,
            }
        return records

    def summary(self):
        result = super().summary()
        result.update(version="current-model-uncertainty-sweep-field-v19",
            model_uncertainty_admission="exact current known-tool segment against every declared additive-margin region",
            uncertainty_is_statistical_sigma=False, clearance_certified=False)
        return result
