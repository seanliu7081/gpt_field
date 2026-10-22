"""Explicit wrist-navigation offers with unchanged physical action checks."""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_proposals_v31 import FieldProposalBankV31
from .visual_field_proposals_v34 import FieldProposalBankV34
from .visual_objective_v32 import uses_rigid_proxy


class FieldProposalBankV35(FieldProposalBankV34):
    def _configure_preview_guard(self, guard, obs, context, state, prediction, monitor):
        super()._configure_preview_guard(guard, obs, context, state, prediction, monitor)
        # configure clears the prior role binding. Bind the exact prospective
        # response AFTER configuration, on this clone alone. Actual admission
        # still requires the model response's own binding and all live checks.
        guard.bind_reference_role_context(prediction)

    def _preview(self, policy, obs, offset, *, recovery=False):
        prospective = copy.copy(policy)
        prospective.prediction = copy.deepcopy(policy.prediction)
        prediction = prospective.prediction
        context = copy.deepcopy(prediction.get('motion_context') or {})
        context.update(translation_mode='separate' if recovery else 'transit',
                       intended_contact_id=None, purpose='task_translation')
        prediction.update(controller='visual_field',
            phase='recover' if recovery else prediction['phase'],
            recovery_route='normal', target_contact='none', contact_alignment=None,
            motion_context=context,
            field_intent=self._intent(policy, np.asarray(obs.state, float), offset,
                                      recovery=recovery))
        return super()._preview(prospective, obs, offset, recovery=recovery)

    def build(self, policy, obs, visual_report):
        if visual_report.get('objective_source') != 'wrist_bearing':
            return super().build(policy, obs, visual_report)
        self.phase_transition = None
        prediction = policy.prediction or {}
        external = (prediction.get('visual_references') or {}).get('external') or {}
        eligible = (
            visual_report.get('accepted') is True
            and prediction.get('phase') in ('approach', 'align', 'recover')
            and prediction.get('controller') in ('visual_field', 'direct')
            and prediction.get('target_contact') == 'none'
            and prediction.get('contact_alignment') is None
            and external.get('moving_kind') == 'tool'
            and (policy.local.intent or {}).get('gripper_command') == 'hold'
            and policy.local.pending is None and getattr(policy, 'pending', None) is None
            and not policy.target_contact.summary().get('carried_payload', {}).get('active')
        )
        if not eligible:
            unavailable = copy.deepcopy(visual_report)
            unavailable.update(accepted=False, candidates=[])
            return super().build(policy, obs, unavailable)
        fresh = policy.visual_goal_field.propose(obs.state,
            motion_axes=policy.local.intent.get('motion_axes', 'xyz'),
            metric_goal_xyz=policy.goal, max_goal_distance_m=.030,
            max_candidates=6, joint_position=obs.joint_position)
        rows = fresh.get('candidates', []) if (fresh.get('objective_source') == 'wrist_bearing'
            and fresh.get('accepted') is True) else []
        qualified = [copy.deepcopy(row) for row in rows
                     if row in visual_report.get('candidates', [])]
        if not qualified:
            unavailable = copy.deepcopy(visual_report)
            unavailable.update(accepted=False, candidates=[])
            super().build(policy, obs, unavailable)
            self.rejected.extend({'goal_offset_m': copy.deepcopy(row.get('goal_offset_m')),
                'reason': 'proposal_wrist_goal_not_currently_qualified', 'source': 'wrist_bearing'}
                for row in visual_report.get('candidates', []))
            return self.summary()
        prospective = copy.copy(policy)
        prospective.prediction = copy.deepcopy(prediction)
        if prediction['phase'] == 'recover':
            prospective.prediction['phase'] = 'approach'
            self.phase_transition = {
                'from_phase': 'recover', 'offered_phase': 'approach',
                'objective_source': 'wrist_bearing',
                'qualified_visual_candidates': len(qualified),
                'explicit_model_choice_required': True,
                'live_phase_changed': False, 'physical_debt_replenished': False,
                'full_route_and_first_command_checks_unchanged': True,
            }
        report = copy.deepcopy(visual_report)
        report['candidates'] = qualified
        # V32 excludes anatomical goals under a proxy. A separate observed
        # wrist target supplies navigation evidence, never proxy pinch anatomy.
        # The original full route and actual first-command checks still apply.
        return FieldProposalBankV31.build(self, prospective, obs, report)

    def summary(self):
        result = super().summary()
        result['version'] = 'checked-separate-wrist-navigation-offers-v35'
        result['wrist_bearing_is_contact_or_depth_evidence'] = False
        return result
