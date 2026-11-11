"""Bounded heading evidence and paid-history reference review for Field control."""
from __future__ import annotations

from .visual_field_policy_v27 import VisualFieldPolicyV27
from .visual_reference_candidates_v28 import CausalReferenceCandidatesV28
from .visual_translation_guard_v28 import ProjectedTranslationGuardV28


class VisualFieldPolicyV28(VisualFieldPolicyV27):
    def reset(self):
        super().reset()
        self.reference_candidates = CausalReferenceCandidatesV28()
        self.client.reference_candidate_provider = self.reference_candidates.image
        self.client.reference_candidate_resolver = self._resolve_reference_candidate
        self.translation_guard = ProjectedTranslationGuardV28()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        # V27's measured fixed-goal executor, scene and contact gates persist.
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _observable_feedback(self):
        from .visual_field_subscription_v28 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v28'
        return decision
