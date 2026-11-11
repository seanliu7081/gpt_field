"""Field execution with comparable visual-risk measurements across refreshes.

Reference and envelope revisions update the current scene representation.
They do not represent robot motion or replenish any physical travel allowance.
"""
from __future__ import annotations

from .visual_field_policy_v25 import VisualFieldPolicyV25
from .visual_translation_guard_v26 import ProjectedTranslationGuardV26


class VisualFieldPolicyV26(VisualFieldPolicyV25):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV26()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        # Retain the same planner, measured-response model and fixed-goal
        # ledger. Only the provenance of temporal risk measurements changes.
        self.local.bind_translation_guard(
            self.translation_guard, self.monitor, self.robot_response)

    def _observable_feedback(self):
        from .visual_field_subscription_v26 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v26'
        return decision
