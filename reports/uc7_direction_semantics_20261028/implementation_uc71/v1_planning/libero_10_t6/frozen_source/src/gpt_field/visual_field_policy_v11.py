"""Persistent Field execution with robot envelopes and live RGB boundaries."""
from __future__ import annotations

import copy
import time
import numpy as np

from .policy import Decision
from .visual_field_policy import images
from .visual_field_policy_v10 import VisualFieldPolicyV10
from .visual_field_v8 import measured_current_pose_hold_action
from .visual_field_v11 import MetricFieldExecutorV11
from .visual_translation_guard_v11 import ProjectedTranslationGuardV11


class VisualFieldPolicyV11(VisualFieldPolicyV10):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV11()
        self.local = MetricFieldExecutorV11()
        self.local.bind_translation_guard(self.translation_guard, self.monitor)
        self.translation_vetoes = 0

    @property
    def diagnostic_counters(self):
        result = super().diagnostic_counters
        result["projected_translation_vetoes"] = getattr(self, "translation_vetoes", 0)
        return result

    def _observable_feedback(self):
        result = super()._observable_feedback()
        guard = getattr(self, "translation_guard", None)
        result["translation_guard"] = guard.summary() if guard is not None else None
        result["controller_handoff"]["decision_rule"] += (
            " A weighted path through scene uncertainty does not establish clearance. "
            "Use the current whole-tool/neighbor projection check to revise an unsafe corridor. "
            "A separating movement must improve the conflicting-neighbor image gap; changing "
            "stage labels or repeating the same descent does not resolve it. Keep a visible "
            "wrist target correspondence during near approach, and verify the intended object "
            "rather than a nearby object is in the interaction corridor. Robot geometry is "
            "known locally, but obstacle depth and 3D placement remain uncertain.")
        return result

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        started = time.monotonic()
        self.translation_guard.configure(images(obs), self.prediction["motion_context"],
            obs.state, target_id=self.prediction["target_id"], monitor=self.monitor)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        started = time.monotonic()
        selected = decision.diagnostics.get("selection")
        if decision.diagnostics["controller"] == "direct" and not (selected and not selected.get("accepted")):
            delta = np.asarray(decision.action[:3])*.05
            admission = self.translation_guard.assess(delta, monitor=self.monitor,
                nominal_away=False, rotation_rad=float(np.linalg.norm(decision.action[3:6])*.5))
            cap = admission.get("cap_m")
            if not admission["accepted"] or cap is not None and np.linalg.norm(delta) > cap+1e-12:
                reason = admission["reason"] if not admission["accepted"] else "direct_translation_exceeds_observed_allowance"
                self.chunk = []
                self._failure(reason)
                self.events[-1]["step"] = self.step_index-1
                hold = measured_current_pose_hold_action()
                diagnostics = copy.deepcopy(decision.diagnostics)
                diagnostics.update(selection={"action": hold, "accepted": False,
                    "status": reason, "need_refresh": True, "diagnostics": copy.deepcopy(admission)},
                    rejected_model_action=list(decision.action), refresh_reason=reason,
                    rejection_hold={"purpose": "reset_persistent_osc_goal_to_current_measured_orientation",
                        "orientation_refresh_rad": self.local.OSC_GOAL_REFRESH_RAD,
                        "requested_motion_executed": False})
                decision = Decision(tuple(hold), diagnostics)
                self.pending = obs, decision
            decision.diagnostics["translation_admission"] = copy.deepcopy(admission)
        selected = decision.diagnostics.get("selection") or {}
        if selected and not selected.get("accepted") and "translation" in selected.get("status", ""):
            self.translation_vetoes += 1
        decision.diagnostics.update(policy_version="visual-field-policy-v11",
                                    feedback=self._observable_feedback())
        elapsed = time.monotonic()-started
        self.local_total_s += elapsed
        decision.diagnostics["local_planning_s"] += elapsed
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self.translation_guard.observe(images(after), after.state, decision.action, monitor=self.monitor)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
