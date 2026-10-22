"""Explicitly reviewed finite reversal can abort an adverse acquisition heading."""
from __future__ import annotations

from .visual_field_policy_v32 import VisualFieldPolicyV32
from .visual_field_v33 import MetricFieldExecutorV33
from .visual_translation_guard_v33 import ProjectedTranslationGuardV33


class VisualFieldPolicyV33(VisualFieldPolicyV32):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV33()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV33()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _bind_egress_context(self):
        result = super()._bind_egress_context()
        guard = getattr(self, 'translation_guard', None)
        if hasattr(guard, 'bind_reverse_abort_context'):
            guard.bind_reverse_abort_context(self.prediction, self.model_requests,
                fixed_goal_xyz=getattr(self.local, 'goal', None))
        return result

    def _observable_feedback(self):
        result = super()._observable_feedback()
        from .visual_field_subscription_v33 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v33'
        return decision
