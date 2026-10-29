"""Reuse checked visual goals and paid observations with one final feedback row."""
from __future__ import annotations

import copy

from .visual_feedback_transaction_v34 import FinalTransitionFeedbackMixinV34
from .visual_field_policy_v33 import VisualFieldPolicyV33
from .visual_field_proposals_v34 import FieldProposalBankV34
from .visual_field_v34 import MetricFieldExecutorV34
from .visual_translation_guard_v34 import ProjectedTranslationGuardV34
from .visual_reference_review_v34 import TypedReferenceReviewPolicyMixinV34


class VisualFieldPolicyV34(FinalTransitionFeedbackMixinV34,
                          TypedReferenceReviewPolicyMixinV34,
                          VisualFieldPolicyV33):
    def reset(self):
        self._transition_feedback_deferred = False
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV34()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV34()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.field_proposals = FieldProposalBankV34()
        self.client.field_proposal_resolver = self.field_proposals.resolve

    def _observable_feedback(self):
        if getattr(self, '_transition_feedback_deferred', False):
            return {}
        result = super()._observable_feedback()
        from .visual_field_subscription_v34 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        result['reference_review_dispatch'] = copy.deepcopy(
            getattr(self, 'reference_review_dispatch', None))
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v34'
        return decision
