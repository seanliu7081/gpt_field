"""Persistent visual objective evidence and checked Field recovery."""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v20 import VisualFieldPolicyV20
from .visual_field_v21 import MetricFieldExecutorV21
from .visual_translation_guard_v21 import ProjectedTranslationGuardV21
from .visual_monitor_v21 import RoleAwareVisualMonitorV21
from .visual_task_progress_v21 import VisualTaskProgressV21
from .visual_field_subscription_v21 import geometry_semantics
from .visual_field_v6 import _AxisConstrainedField


class VisualFieldPolicyV21(VisualFieldPolicyV20):
    def reset(self):
        super().reset()
        self.monitor = RoleAwareVisualMonitorV21()
        self.translation_guard = ProjectedTranslationGuardV21()
        self.local = MetricFieldExecutorV21()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.task_progress = VisualTaskProgressV21()

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result["geometry_semantics"] = geometry_semantics()
        progress = getattr(self, "task_progress", None)
        result["task_progress"] = None if progress is None else progress.summary()
        result["local_reference_maintenance"] = {
            "version": "seed-anchored-appearance-maintenance-v21",
            "new_features_or_semantic_identity_inferred": False,
            "response_validation_and_motion_budgets_unchanged": True,
            "details": "Current surviving features may use bounded directly seed-verified keyframes; inactive originals require original seed appearance. Loss still requires model review."}
        return result

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        started = time.monotonic()
        self.task_progress.configure(obs.state, self.monitor, self.prediction, self.goal, step=self.step_index)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def _screen_recovery_options(self, state8):
        super()._screen_recovery_options(state8)
        # Axis-only proposals cannot reverse a measured diagonal movement.
        # A guard-offered reverse is a prospective model choice, never an
        # automatically selected action or a claim of metric separation.
        directions = getattr(self.translation_guard, "recovery_directions", lambda: [])()
        if self.local.intent is None or not directions:
            return
        point = np.asarray(state8[:3], float)
        field = self.local.field
        while isinstance(field, _AxisConstrainedField):
            field = field._obstacle_field
        for offer in directions:
            direction = np.asarray(offer["direction_xyz"], float)
            endpoint = point + direction*float(offer["maximum_measured_extent_m"])
            if not self.local.planner._valid_segment(field, point, endpoint, self.local.boxes):
                self.recovery_options.setdefault("rejected_candidates", []).append({
                    **copy.deepcopy(offer), "reason": "reverse_model_segment_blocked"})
                continue
            command = direction*.002
            preview = None
            converged = False
            for _ in range(3):
                preview = self.translation_guard.preview_assess(command, monitor=self.monitor,
                    nominal_clear=True, nominal_away=self.local._nominal_away(point, command),
                    nominal_regions=self.local.nominal_regions_for_segment(point, point+command),
                    translation_response=self.robot_response, rotation_rad=0., orientation_hold=True,
                    prospective_translation_mode="separate")
                cap = preview.get("cap_m")
                if not preview["accepted"] or cap is not None and cap < .001-1e-9:
                    break
                if cap is None or np.linalg.norm(command) <= cap+1e-12:
                    converged = True
                    break
                command = direction*cap
            if not converged:
                self.recovery_options.setdefault("rejected_candidates", []).append({
                    **copy.deepcopy(offer), "reason": (preview or {}).get("reason", "reverse_preview_unavailable")})
                continue
            scene = self._scene_assess(state8, command)
            if not scene["accepted"]:
                self.recovery_options.setdefault("rejected_candidates", []).append({
                    **copy.deepcopy(offer), "reason": scene["reason"]})
                continue
            self.recovery_options["candidates"].append({**copy.deepcopy(offer),
                "checked_nominal_segment_m": float(offer["maximum_measured_extent_m"]),
                "preview_first_command_delta_m": command.tolist(),
                "preview_reason": preview["reason"], "preview_admission": preview,
                "scene_coverage_preview": scene, "motion_axes": "xyz",
                "required_translation_mode": "separate", "requires_explicit_subgoal_update": True,
                "recommended_goal_offset_m": (direction*.008).tolist(),
                "recommended_goal_tolerance_m": .003,
                "recommended_local_max_travel_m": float(offer["maximum_measured_extent_m"]),
                "local_limit_does_not_override_persistent_guard_allowance": True,
                "image_separation_observed": False,
                "actual_segment_must_be_rechecked": True})
        if self.recovery_options["candidates"]:
            self.recovery_options["reason"] = "current_candidate_checks_with_bounded_observed_motion_reverse"

    def _field_control_limits(self, obs):
        active = self.task_progress.active
        if active is not None and active["review_required"]:
            self.local.proprioceptive_transit_request = None
            return 0., "task_goal_hypothesis_requires_revision"
        return super()._field_control_limits(obs)

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v21"
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        selection = decision.diagnostics.get("selection") or {}
        accepted = self.field_configured and selection.get("accepted") is True
        straight = bool(accepted and self.goal is not None
            and self.local.checked.segment_valid(after.state[:3], self.goal))
        reason = self.task_progress.observe(before.state, after.state, decision.action,
            monitor=self.monitor, prediction=self.prediction, goal_xyz=self.goal,
            accepted_field=accepted, straight_route=straight, step=self.step_index-1)
        if reason and not self.refresh_reason:
            self.refresh_reason = reason
            self.events.append({"step": self.step_index-1, "reason": reason,
                "task_success_certified": False, "automatic_goal_retarget": False})
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
