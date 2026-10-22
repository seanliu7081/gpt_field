"""Separate observed wrist navigation from anatomy, with stable boundary evidence."""
from __future__ import annotations

import copy

from .visual_field_policy_v34 import VisualFieldPolicyV34
from .visual_field_proposals_v35 import FieldProposalBankV35
from .visual_navigation_fields_v35 import VisualNavigationFieldsV35
from .visual_translation_guard_v35 import ProjectedTranslationGuardV35


class VisualFieldPolicyV35(VisualFieldPolicyV34):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV35()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.visual_goal_field = VisualNavigationFieldsV35()
        self.field_proposals = FieldProposalBankV35()
        self.client.field_proposal_resolver = self.field_proposals.resolve

    def _observable_feedback(self):
        if getattr(self, '_transition_feedback_deferred', False):
            return {}
        result = super()._observable_feedback()
        from .visual_field_subscription_v35 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        local = getattr(self, 'local', None)
        intent = (getattr(local, 'intent', None) or {})
        declared = (self.prediction or {}).get('field_intent') or {}
        scene_limits = copy.deepcopy(getattr(self, 'scene_execution_limits', None))
        result['execution_limit_sources'] = {
            'model_declared_max_travel_m': declared.get('max_travel_m'),
            'model_declared_command_limit_m': declared.get('command_limit_m'),
            'effective_intent_max_travel_m': intent.get('max_travel_m'),
            'effective_intent_command_limit_m': intent.get('command_limit_m'),
            'scene_execution_limits': scene_limits,
            'scene_limit_active': scene_limits is not None,
            'interpretation': ('Null scene_execution_limits means no additional scene-level '
                '15mm cap was imposed. The current intent, geometry, reference, contact and '
                'measured travel/debt checks still apply. Partial visibility alone is not '
                'a requirement to choose 15mm. Changing a model declaration refunds no debt.'),
        }
        return result

    def _review_scene(self, obs, report):
        super()._review_scene(obs, report)
        self.scene_review['visual_objective_source'] = report.get('objective_source')
        self.scene_review['wrist_bearing_is_depth_or_contact_evidence'] = False
        if report.get('objective_source') == 'wrist_bearing':
            for row in self.scene_review['reasons']:
                if row['reason'] == 'image_improving_heading_opposes_metric_target_centre':
                    row['interpretation'] = (
                        'Review target estimate and declared wrist desired bearing against '
                        'current RGB. Wrist bearing has no target range or pinch alignment.')

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v35'
        return decision
