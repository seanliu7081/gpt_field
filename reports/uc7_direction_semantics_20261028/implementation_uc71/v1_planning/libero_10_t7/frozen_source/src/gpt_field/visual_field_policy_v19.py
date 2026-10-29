"""Use the same full-margin scene hypothesis for planning and admission.

Exact current-command sweeps resolve model-remote image overlaps. They do not
certify physical free space or replenish any remaining uncertain-motion debt.
"""
from .visual_field_policy_v18 import VisualFieldPolicyV18
from .visual_field_v19 import MetricFieldExecutorV19
from .visual_translation_guard_v19 import ProjectedTranslationGuardV19
from .visual_field_subscription_v19 import geometry_semantics


class VisualFieldPolicyV19(VisualFieldPolicyV18):
    def _observable_feedback(self):
        result = super()._observable_feedback()
        result["geometry_semantics"] = geometry_semantics()
        return result

    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV19()
        self.translation_guard = ProjectedTranslationGuardV19()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v19"
        return decision
