"""Persistent physical recovery directions and causal RGB motion diagnostics."""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v13 import VisualFieldPolicyV13
from .visual_field_policy_v14 import VisualFieldPolicyV14
from .visual_field_v6 import _AxisConstrainedField
from .visual_translation_guard_v15 import ProjectedTranslationGuardV15
from .visual_motion_diagnostic_v15 import VisualMotionDiagnosticV15


def recovery_options(local, guard, state8, *, monitor=None, robot_response=None):
    """Propose segments checked by geometry AND a side-effect-free guard preview.

The mode change and axis change still require an explicit model subgoal. These
are current translation-admission results, not future scene/contact guarantees.
Every actually selected path segment is independently checked again.
"""
    summary = guard.summary()
    ledger = summary.get("recovery_experiments", {})
    conflicts = summary.get("conflicting_obstacle_ids", [])
    result = {"diagnostic_required": bool(conflicts and ledger.get("trials")),
        "candidates": [], "rejected_candidates": [],
        "shared_remaining_m": ledger.get("shared_remaining_m"),
        "requires_current_model_review": True, "clearance_certified": False,
        "actual_segment_must_be_rechecked": True,
        "preview_scope": "current geometry and translation guard under prospective separate mode"}
    if local.intent is None or not conflicts:
        return result
    field = local.field
    while isinstance(field, _AxisConstrainedField):
        field = field._obstacle_field
    point = np.asarray(state8[:3], float)
    for axis in range(3):
        for sign in (-1., 1.):
            direction = np.eye(3)[axis]*sign
            # Long enough to review a useful local corridor, not an instruction
            # to execute eight millimetres without closed-loop feedback.
            extent = .008
            endpoint = point+direction*extent
            geometry_clear = local.planner._valid_segment(field, point, endpoint, local.boxes)
            nominal_away = geometry_clear and local._nominal_away(point, direction*extent)
            if not nominal_away:
                result["rejected_candidates"].append({"direction_xyz": direction.tolist(),
                    "reason": "nominal_segment_blocked" if not geometry_clear else "not_nominally_separating"})
                continue
            command = direction*.002
            converged = False
            for _ in range(3):
                preview = guard.preview_assess(command, monitor=monitor,
                    nominal_clear=True, nominal_away=local._nominal_away(point, command),
                    nominal_regions=local.nominal_regions_for_segment(point, point+command),
                    translation_response=robot_response, rotation_rad=0., orientation_hold=True,
                    prospective_translation_mode="separate")
                cap = preview.get("cap_m")
                if not preview["accepted"] or cap is not None and cap < .001-1e-9:
                    break
                size = float(np.linalg.norm(command))
                if cap is None or size <= cap+1e-12:
                    converged = True
                    break
                # Recompute all magnitude-dependent admission evidence for the
                # actual proposed command, rather than offering an unchecked cap.
                command = direction*cap
            cap = preview.get("cap_m")
            if not converged:
                result["rejected_candidates"].append({"direction_xyz": direction.tolist(),
                    "reason": preview["reason"] if not preview["accepted"] or cap is not None and cap < .001-1e-9
                        else "preview_command_cap_not_converged", "command_cap_m": cap})
                continue
            result["candidates"].append({"direction_xyz": direction.tolist(),
                "checked_nominal_segment_m": extent,
                "preview_first_command_delta_m": command.tolist(),
                "preview_reason": preview["reason"],
                "eligible_under_current_references": True,
                "motion_axes": "z" if axis == 2 else "xy",
                "requires_explicit_subgoal_update": True,
                "required_translation_mode": "separate",
                "recommended_local_max_travel_m": .012,
                "local_limit_does_not_override_persistent_guard_allowance": True,
                "image_separation_observed": False})
    result["reason"] = ("current_translation_preview_candidates" if result["candidates"]
                        else "no_candidate_passes_current_geometry_and_translation_admission")
    return result


class VisualFieldPolicyV15(VisualFieldPolicyV14):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV15()
        self.motion_diagnostic = VisualMotionDiagnosticV15()
        self.client.motion_diagnostic_provider = self.motion_diagnostic.image
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        motion = getattr(self, "motion_diagnostic", None)
        result["motion_evidence"] = {} if motion is None else motion.summary()
        return result

    def _refresh(self, obs):
        # V13 owns actual task/contact control. Avoid generating the obsolete
        # V14 geometry-only alternatives before replacing them with V15 previews.
        latency = VisualFieldPolicyV13._refresh(self, obs)
        start = time.monotonic()
        self.reference_support.configure(obs, self.monitor)
        self.motion_diagnostic.configure(obs, self.monitor, self.prediction["motion_context"])
        self.recovery_options = recovery_options(self.local, self.translation_guard, obs.state,
            monitor=self.monitor, robot_response=self.robot_response)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-start
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v15"
        return decision

    def observe_transition(self, before, decision, after):
        row = VisualFieldPolicyV13.observe_transition(self, before, decision, after)
        start = time.monotonic()
        self.reference_support.observe(before, after, self.monitor)
        self.motion_diagnostic.observe(before, after, self.monitor)
        if self.refresh_reason:
            self.recovery_options = recovery_options(self.local, self.translation_guard, after.state,
                monitor=self.monitor, robot_response=self.robot_response)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-start
        return row
