"""Opt-in constraint-aware tracking along a retained Field route."""
from __future__ import annotations

from .visual_field_policy_v28 import VisualFieldPolicyV28
from .visual_field_v29 import MetricFieldExecutorV29
from .visual_translation_guard_v29 import ProjectedTranslationGuardV29


class VisualFieldPolicyV29(VisualFieldPolicyV28):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV29()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV29()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _observable_feedback(self):
        from .visual_field_subscription_v29 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v29'
        return decision
