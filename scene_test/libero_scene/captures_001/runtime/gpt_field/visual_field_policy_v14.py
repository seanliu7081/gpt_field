"""Directional evidence recovery with persistent Field task control.

Recovery options are proposals from existing model hypotheses, not permission
to execute. The model still selects a subgoal and every actual planned segment
passes the normal geometry, reference, attachment and translation checks.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v13 import VisualFieldPolicyV13
from .visual_field_v6 import _AxisConstrainedField
from .visual_translation_guard_v14 import ProjectedTranslationGuardV14
from .visual_reference_diagnostic_v14 import MovingReferenceDiagnosticV14


def recovery_options(local, guard, state8):
    """Read-only nominal separation proposals; never call stateful assess()."""
    summary = guard.summary()
    ledger = summary.get("recovery_experiments", {})
    conflicts = summary.get("conflicting_obstacle_ids", [])
    result = {"diagnostic_required": bool(conflicts and ledger.get("trials")),
        "candidates": [], "shared_remaining_m": ledger.get("shared_remaining_m"),
        "requires_current_model_review": True, "clearance_certified": False,
        "actual_segment_must_be_rechecked": True}
    if local.intent is None or not conflicts:
        return result
    length = min(.008, max(0., ledger.get("shared_remaining_m", 0.)))
    if length < .001:
        result["reason"] = "no_useful_unvalidated_measured_allowance"
        return result
    field = local.field
    while isinstance(field, _AxisConstrainedField):
        field = field._obstacle_field
    point = np.asarray(state8[:3], float)
    for axis in range(3):
        for sign in (-1., 1.):
            direction = np.eye(3)[axis]*sign
            delta = direction*length
            # Only prospective axis restrictions are removed. Actual obstacle,
            # target-contact, unknown masks and continuous boxes all remain.
            valid = local.planner._valid_segment(field, point, point+delta, local.boxes)
            if not valid or not local._nominal_away(point, delta):
                continue
            result["candidates"].append({"direction_xyz": direction.tolist(),
                "checked_nominal_segment_m": length,
                "motion_axes": "z" if axis == 2 else "xy",
                "requires_explicit_subgoal_update": True,
                "required_translation_mode": "separate",
                "image_separation_observed": False})
    result["reason"] = ("nominal_candidates_require_live_admission" if result["candidates"]
                        else "no_nominal_separating_candidate_in_current_hypotheses")
    return result


class VisualFieldPolicyV14(VisualFieldPolicyV13):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV14()
        self.reference_support = MovingReferenceDiagnosticV14()
        self.recovery_options = {}
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        support = getattr(self, "reference_support", None)
        result["reference_support"] = {} if support is None else support.summary()
        result["reference_support"]["diagnostic_required"] = bool(
            result["reference_support"].get("review_recommended"))
        result["recovery_options"] = copy.deepcopy(getattr(self, "recovery_options", {}))
        return result

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        start = time.monotonic()
        self.reference_support.configure(obs, self.monitor)
        self.recovery_options = recovery_options(self.local, self.translation_guard, obs.state)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-start
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v14"
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        start = time.monotonic()
        self.reference_support.observe(before, after, self.monitor)
        # Build proposed alternatives only when the existing controller hands
        # back to the model. The diagnostic alone never adds an inference call.
        if self.refresh_reason:
            self.recovery_options = recovery_options(self.local, self.translation_guard, after.state)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-start
        return row
