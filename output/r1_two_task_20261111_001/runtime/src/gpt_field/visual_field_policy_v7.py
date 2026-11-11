"""Field reuse with persistent correspondence health and explicit repair feedback."""
from __future__ import annotations

from .visual_field_policy_v6 import VisualFieldPolicyV6
from .visual_field_policy_v5 import REFERENCE_FAILURES
from .visual_monitor_v7 import RoleAwareVisualMonitorV7


class VisualFieldPolicyV7(VisualFieldPolicyV6):
    def reset(self):
        super().reset()
        self.monitor = RoleAwareVisualMonitorV7()

    def _observable_feedback(self):
        result = super()._observable_feedback()
        health = result["visual"].get("reference_health") or {}
        result["controller_handoff"]["reference_repair_required"] = bool(health.get("repair_required"))
        result["controller_handoff"]["decision_rule"] += (
            " Numerical tracking reliability does not prove the feature belongs to the moving "
            "tool or payload. When reference repair is required, inspect the marked M in the "
            "derived external image and seed a genuinely different visible controlled part; "
            "retaining, renaming or slightly moving the same stationary reference does not "
            "renew its unvalidated motion budget. A changed informative Direct5 view is an "
            "option if the controlled part is not visible. Do not equate navigation-target "
            "centering or stalled descent with grasp readiness: inspect a graspable surface "
            "between the jaws, and use a bounded Direct5 terminal contact correction where "
            "needed before requesting gripper settling. Robot waiting does not create contact.")
        return result

    def _failure(self, reason):
        health = self.monitor.summary().get("reference_health") or {}
        if reason not in REFERENCE_FAILURES and health.get("repair_required"):
            self.reference_failure_events.append({"step": self.step_index,
                "target_id": self.prediction["target_id"], "phase": self.prediction["phase"],
                "reason": reason})
            self.reference_failure_events = self.reference_failure_events[-8:]
        super()._failure(reason)

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v7"
        return decision
