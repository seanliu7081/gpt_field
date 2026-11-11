"""Explicit model-risk distance planning with the measured V8 execution loop."""
from __future__ import annotations

import copy
import time

from .visual_field_policy_v8 import VisualFieldPolicyV8
from .visual_field_v9 import MetricFieldExecutorV9
from .visual_field_subscription_v9 import geometry_semantics


class VisualFieldPolicyV9(VisualFieldPolicyV8):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV9()

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result["geometry_semantics"] = geometry_semantics()
        result["controller_handoff"]["decision_rule"] += (
            " Read field feasibility before replacing a goal. A hard-core or genuinely unknown "
            "start conflict cannot be solved by changing the destination. Visible-object pose "
            "uncertainty is an explicit path cost only when the current RGB assessment supports "
            "localizing that object; this never establishes obstacle-free motion. Do not delete "
            "or relabel a rejected unknown region just to obtain a route. Waiting without a "
            "changed view provides no new geometric evidence. Preserve valid observations and "
            "choose a bounded task-directed translation or necessary contact correction when "
            "supported by the current views."
        )
        return result

    def _refresh(self, obs):
        # The inherited refresh first configures a strict, conservative field.
        # Only this response's explicit scene assessment may then enable the
        # uncertainty-cost contract. No old assessment survives a refresh.
        latency = super()._refresh(obs)
        started = time.monotonic()
        if self.prediction["controller"] == "visual_field":
            effective = copy.deepcopy(self.local.intent)
            self.local.configure(obs.state, effective,
                scene_assessment=copy.deepcopy(self.prediction.get("motion_context")))
            self.goal = self.local.goal.copy()
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v9"
        return decision
