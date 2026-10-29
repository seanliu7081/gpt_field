"""Typed paid-history reference review and explicit target approach roles."""
from __future__ import annotations

from .visual_field_policy_v29 import VisualFieldPolicyV29
from .visual_reference_candidates_v30 import CausalReferenceCandidatesV30
from .visual_reference_review_v30 import TypedReferenceReviewPolicyMixinV30
from .visual_translation_guard_v30 import ProjectedTranslationGuardV30


class VisualFieldPolicyV30(TypedReferenceReviewPolicyMixinV30, VisualFieldPolicyV29):
    def reset(self):
        super().reset()
        self.reference_candidates = CausalReferenceCandidatesV30()
        self.client.reference_candidate_provider = self.reference_candidates.image
        self.client.reference_candidate_resolver = self._resolve_reference_candidate
        self.translation_guard = ProjectedTranslationGuardV30()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _bind_egress_context(self):
        # This hook follows current guard/executor configuration, before both
        # preview and actual selection. Direct/gripper must replace stale roles.
        prediction = self.prediction or {}
        self.translation_guard.bind_target_approach_context({
            'controller': prediction.get('controller'),
            'phase': prediction.get('phase'),
            'target_id': prediction.get('target_id'),
        })
        return super()._bind_egress_context()

    def _observable_feedback(self):
        from .visual_field_subscription_v30 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v30'
        return decision
