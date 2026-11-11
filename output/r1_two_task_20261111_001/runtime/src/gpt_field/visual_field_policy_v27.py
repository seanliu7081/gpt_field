"""Field planning with bounded reconciliation of image and metric hypotheses."""
from __future__ import annotations

from .visual_field_policy_v26 import VisualFieldPolicyV26
from .visual_field_v27 import MetricFieldExecutorV27
from .visual_translation_guard_v27 import ProjectedTranslationGuardV27


class VisualFieldPolicyV27(VisualFieldPolicyV26):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV27()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV27()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _observable_feedback(self):
        from .visual_field_subscription_v27 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v27'
        return decision
