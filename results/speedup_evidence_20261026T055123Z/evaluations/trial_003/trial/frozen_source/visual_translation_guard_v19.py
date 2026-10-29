"""Bounded execution under exact current model-uncertainty sweep hypotheses.

The model's additive uncertainty margin is not a statistical standard
deviation. A current whole-tool segment outside that complete margin need not
also prove image-plane separation from a projected overlapping object. This
is conditional model geometry, never measured or certified free space.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v17 import ProjectedTranslationGuardV17
from .visual_translation_guard_v11 import _gap


class ProjectedTranslationGuardV19(ProjectedTranslationGuardV17):
    version = "current-model-uncertainty-sweep-admission-v19"
    model_sweep_command_limit_m = .005

    def __init__(self):
        super().__init__()
        self.sweep_observation_epoch = 0
        self._admission_sweep_ids = None
        self._last_sweep_admission = None

    def configure(self, *args, **kwargs):
        self.sweep_observation_epoch += 1
        self._admission_sweep_ids = None
        self._last_sweep_admission = None
        return super().configure(*args, **kwargs)

    def observe(self, *args, **kwargs):
        # No prior command's geometric proof is used for a new image, including
        # zero commands or veto holds. Legacy RGB worsening monitoring remains.
        self._admission_sweep_ids = None
        self.sweep_observation_epoch += 1
        return super().observe(*args, **kwargs)

    def _near_metric(self, identity):
        if (self._admission_sweep_ids is not None
                and identity in self._admission_sweep_ids
                and identity not in self.worsening):
            return False
        return super()._near_metric(identity)

    def _proof_status(self, record, command, rotation_rad):
        proof = record.get("model_uncertainty_sweep")
        if not isinstance(proof, dict):
            return "unavailable"
        try:
            state = np.asarray(proof["state8"], float)
            start, end = np.asarray(proof["start_xyz_m"], float), np.asarray(proof["end_xyz_m"], float)
            valid = (proof["version"] == "v19-current-full-uncertainty-union"
                and proof["object_id"] == record["id"]
                and record.get("kind") != "unknown_regions"
                and proof["observation_epoch"] == self.sweep_observation_epoch
                and state.shape == (8,) and start.shape == (3,) and end.shape == (3,)
                and np.isfinite(np.r_[state, start, end]).all()
                and np.allclose(state, self.state, atol=1e-10, rtol=0.)
                and np.allclose(start, self.state[:3], atol=1e-10, rtol=0.)
                and np.allclose(end, start+command, atol=1e-10, rtol=0.)
                and proof["scene_coverage_accepted"] is True
                and proof["full_finger_range"] is True
                and proof["coverage_scope"] == "actual_commanded_whole_tool_sweep"
                and np.isfinite(proof["orientation_margin_rad"])
                and proof["orientation_margin_rad"] >= max(.04, abs(rotation_rad))-1e-12
                and proof["configuration_part_count"] >= 6
                and isinstance(proof["full_uncertainty_sweep_disjoint"], bool))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            return "invalid_or_stale"
        return "disjoint" if proof["full_uncertainty_sweep_disjoint"] else "intersects_no_exemption"

    def assess(self, delta_xyz, **kwargs):
        command = np.asarray(delta_xyz, float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command translation must be three finite metres")
        records = kwargs.get("nominal_regions") or []
        status = {row["id"]: self._proof_status(row, command, float(kwargs.get("rotation_rad", 0.)))
                  for row in records}
        # Nominal hard-core/unknown/axis validity is independent and mandatory.
        disjoint = {identity for identity, value in status.items() if value == "disjoint"}
        eligible = disjoint if kwargs.get("nominal_clear") is True else set()
        # Preserve the throughput of already-unambiguous V18 proposals: apply
        # the conditional 5mm branch only when it removes an actual legacy
        # metric-near AND image-near restriction for this exact command.
        implicated = set()
        metrics = {row["id"]: row for row in records}
        if eligible and self.tool_box is not None:
            flow, _ = self._image_prediction(command, kwargs.get("translation_response"))
            for risk in self._risks():
                identity = risk["id"]
                if identity not in eligible or risk["intended_contact"]:
                    continue
                row = metrics[identity]
                try:
                    distance, sigma = float(row["core_distance_m"]), float(row["uncertainty_m"])
                    legacy_near = (np.isfinite([distance, sigma]).all() and min(distance, sigma) >= 0
                                   and distance <= 3*sigma) or identity in self.worsening
                except (ValueError, TypeError, KeyError):
                    legacy_near = True
                threshold = max(risk["margin_px"], 2.+self._gap_noise(risk))
                projected = risk["gap_px"] if flow is None else _gap(
                    self.tool_box+np.tile(flow, 2), risk["box_px"])
                if legacy_near and (min(risk["gap_px"], projected) <= threshold or identity in self.worsening):
                    implicated.add(identity)
        previous = self._admission_sweep_ids
        self._admission_sweep_ids = implicated
        try:
            result = super().assess(delta_xyz, **kwargs)
        finally:
            self._admission_sweep_ids = previous
        exercised = bool(result["accepted"] and implicated
                         and result["reason"] == "projected_risk_no_near_conflict"
                         and np.linalg.norm(command) > 1e-12)
        if exercised:
            cap = min(float(np.linalg.norm(command)), self.model_sweep_command_limit_m)
            if result.get("cap_m") is not None:
                cap = min(cap, result["cap_m"])
            result.update(reason="projected_model_uncertainty_sweep_disjoint", cap_m=cap)
            if self._motion_classification_pending is not None:
                self._motion_classification_pending = {
                    **self._motion_classification_pending, "command_cap_m": cap}
            self._v12_last_result = {key: copy.deepcopy(value) for key, value in result.items()
                                   if key != "diagnostics"}
            self._recovery_reason = result["reason"]
        self._last_sweep_admission = {
            "observation_epoch": self.sweep_observation_epoch,
            "per_object": status, "eligible_disjoint_ids": sorted(eligible),
            "materially_overridden_near_ids": sorted(implicated-self.worsening),
            "branch_exercised": exercised, "command_limit_m": self.model_sweep_command_limit_m,
            "proof_scope": "current_assess_only_not_observe_or_next_proposal",
            "uncertainty_semantics": "model_additive_margin_not_statistical_standard_deviation",
            "observed_worsening_overrides": sorted(eligible & self.worsening),
            "clearance_certified": False, "physical_budget_replenished": False,
        }
        result.setdefault("diagnostics", {})["model_uncertainty_sweep_admission"] = copy.deepcopy(self._last_sweep_admission)
        return result

    def summary(self):
        result = super().summary()
        result["model_uncertainty_sweep_admission"] = copy.deepcopy(getattr(self, "_last_sweep_admission", None))
        result["sweep_observation_epoch"] = getattr(self, "sweep_observation_epoch", 0)
        return result


ObservedRiskTranslationGuardV19 = ProjectedTranslationGuardV19
