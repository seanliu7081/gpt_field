"""Physical-direction recovery accounting and transactional admission previews.

V14's RGB/robot evidence, model-geometry checks and finite measured travel
allowance are retained. A repaired visual correspondence does not create a new
physical direction or replenish its allowance. Predictions remain conditional
evidence; neither a preview nor an admitted experiment certifies clearance.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v14 import ObservedRiskTranslationGuardV14


class ObservedRiskTranslationGuardV15(ObservedRiskTranslationGuardV14):
    version = "physical-direction-recovery-preview-v15"

    def __init__(self):
        super().__init__()
        # These identities are diagnostics, never reasons to grant allowance.
        # They are committed only after an acknowledged, matching command.
        self._experiment_support_tracks = {}
        self._support_revision_events = []

    def _active_experiment(self):
        if self._last_charged_experiment is not None:
            return self._last_charged_experiment
        return len(self.experiments)-1 if self.experiments else None

    def _experiment_for(self, direction, *, nominal_away):
        """Pure physical-heading lookup; only observe creates/updates a row."""
        proposal = {"index": None, "direction": direction.tolist(),
                    "support_generation": self._support_generation}
        if not self.experiments:
            return proposal, None
        angles = [self._angle(direction, row["direction"]) for row in self.experiments]
        matched = int(np.argmin(angles))
        active = self._active_experiment()
        separate = self.context.get("translation_mode") == "separate" and nominal_away
        if angles[matched] <= self.compatible_direction_rad+1e-9:
            row = self.experiments[matched]
            if row["measured_travel_m"] >= self.experiment_limit_m-1e-9:
                return None, "translation_recovery_direction_exhausted"
            if matched != active and not separate:
                return None, "translation_direction_resume_requires_nominal_separation"
            # An earlier command-scale fit failure is not a permanent ban:
            # assess recomputes the cap using the CURRENT measured response.
            return {**proposal, "index": matched}, None
        if not separate:
            return None, "translation_new_experiment_requires_nominal_separation"
        if min(angles) < self.fresh_direction_rad-1e-9:
            return None, "translation_recovery_direction_not_distinct"
        if len(self.experiments) >= self.maximum_experiments:
            return None, "translation_recovery_experiment_limit"
        previous = self.experiments[active]
        if (previous["measured_travel_m"] < self.information_checkpoint_m-1e-9
                and not previous.get("support_loss_observed", False)):
            return None, "translation_recovery_direction_change_requires_review"
        return proposal, None

    def observe(self, images, state8, action, *, monitor=None):
        pending = self._experiment_pending
        classification = self._motion_classification_pending
        command = np.asarray(action[:3], float)*.05
        norm = float(np.linalg.norm(command))
        matched = bool(classification is not None and norm > 1e-8
                       and self._angle(command/norm, classification["direction"]) < 1e-5
                       and norm <= classification["command_cap_m"]+1e-9)
        super().observe(images, state8, action, monitor=monitor)
        if matched and pending is not None:
            index = self._last_charged_experiment
            row = self.experiments[index]
            if self._experiment_support_tracks.get(index) is not self.tool_track:
                event = {
                    "revision": len(self._support_revision_events),
                    "experiment_index": index,
                    "support_generation": self._support_generation,
                    "measured_direction_travel_m": row["measured_travel_m"],
                    "acknowledged_actions": row["acknowledged_actions"],
                    "is_observed_motion_of_reference_revision": False,
                    "allowance_replenished": False,
                }
                self._support_revision_events.append(event)
                row.setdefault("support_revisions", []).append(copy.deepcopy(event))
                self._experiment_support_tracks[index] = self.tool_track
            row["latest_executed_support_generation"] = self._support_generation
            # A successful CURRENT cap fit supersedes the historical failure.
            self._command_exhausted_experiments.discard(index)
        active = self._active_experiment()
        tool = self._tool(monitor) if monitor is not None else self.tool_track
        lost = (tool is None or tool is not self.tool_track or not tool.reliable
                or self.domain_invalid == "translation_tool_correspondence_lost")
        if active is not None and lost:
            row = self.experiments[active]
            row["support_loss_observed"] = True
            row.setdefault("support_loss_first_observed_at_travel_m", row["measured_travel_m"])
        return self.summary()

    def preview_assess(self, delta_xyz, *, prospective_translation_mode=None, **kwargs):
        """Return the exact current admission decision without mutating state.

        A prospective ``separate`` context is a proposed future MODEL CHOICE,
        not a live mode change or execution permission. All other context,
        tracking, worsening, nominal geometry and measured-response checks are
        identical to assess. The result must be reassessed after that choice.

        assess replaces its provisional dictionaries/scalars; its only nested
        in-place mutation is the command-fit failure set. A shallow transaction
        plus an isolated copy of that set preserves the identity of live
        monitor tracks and response models, unlike a deepcopy of the guard.
        The original dictionary is restored even on an exception. This object
        is used serially by the policy; previews are not concurrent operations.
        """
        if prospective_translation_mode not in (None, "separate"):
            raise ValueError("prospective_translation_mode must be None or 'separate'")
        original = self.__dict__
        working = original.copy()
        working["_command_exhausted_experiments"] = set(self._command_exhausted_experiments)
        if prospective_translation_mode is not None:
            working["context"] = dict(self.context or {})
            working["context"]["translation_mode"] = prospective_translation_mode
            working["context"]["intended_contact_id"] = None
        try:
            self.__dict__ = working
            result = copy.deepcopy(self.assess(delta_xyz, **kwargs))
        finally:
            self.__dict__ = original
        result["preview"] = True
        result["requires_explicit_model_choice"] = prospective_translation_mode is not None
        result["prospective_translation_mode"] = prospective_translation_mode
        return result

    def summary(self):
        result = super().summary()
        if not hasattr(self, "_support_revision_events"):
            return result
        ledger = result["recovery_experiments"]
        for row in ledger["trials"]:
            row["exhausted"] = row["measured_travel_m"] >= self.experiment_limit_m-1e-9
            row["last_command_fit_failed"] = row.pop("remaining_cannot_fit_useful_command")
            row["command_fit_must_be_reassessed"] = not row["exhausted"]
        ledger.update({
            "active_experiment_index": self._active_experiment(),
            "slot_identity": "physical_command_heading_not_visual_reference",
            "reference_revision_events": copy.deepcopy(self._support_revision_events),
            "new_direction_requires": "explicit nominal-away separation and >=45deg from all prior headings; after 4mm measured checkpoint or observed support loss",
            "existing_direction_resume_requires": "remaining allowance; explicit nominal-away separation if switching back from another heading",
            "preview_changes_state": False,
        })
        return result


ProjectedTranslationGuardV15 = ObservedRiskTranslationGuardV15
