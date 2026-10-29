"""Checked visual-goal offers may explicitly leave a recovery stage.

The phase change exists only on a prospective policy view until the model
selects the observation-bound offer. All V32 role exclusions and V31 full
route/first-command preview checks remain in force.
"""
from __future__ import annotations

import copy

from .visual_objective_v32 import FieldProposalBankV32, uses_rigid_proxy


class FieldProposalBankV34(FieldProposalBankV32):
    def __init__(self):
        super().__init__()
        self.phase_transition = None

    def build(self, policy, obs, visual_report):
        self.phase_transition = None
        prediction = policy.prediction or {}
        external = (prediction.get('visual_references') or {}).get('external') or {}
        eligible = (
            prediction.get('phase') == 'recover'
            and prediction.get('controller') in ('visual_field', 'direct')
            and prediction.get('target_contact') == 'none'
            and external.get('moving_kind') == 'tool'
            and not uses_rigid_proxy(prediction)
            and policy.local.intent is not None
            and policy.local.pending is None
            and getattr(policy, 'pending', None) is None
            and not policy.target_contact.summary().get('carried_payload', {}).get('active')
            and visual_report.get('accepted') is True
            and bool(visual_report.get('candidates'))
        )
        if not eligible:
            return super().build(policy, obs, visual_report)

        # The argument is presentation, not an authority to reuse old visual
        # evidence. Recompute the same pure local query and admit only exact
        # current candidates; this does not train, move or change any budget.
        fresh = policy.visual_goal_field.propose(
            obs.state, motion_axes=policy.local.intent.get('motion_axes', 'xyz'),
            metric_goal_xyz=policy.goal, max_goal_distance_m=.030,
            max_candidates=6, joint_position=obs.joint_position)
        current_rows = fresh.get('candidates', []) if fresh.get('accepted') is True else []
        supplied_rows = visual_report['candidates']
        qualified = [copy.deepcopy(row) for row in current_rows if row in supplied_rows]
        stale = [row for row in supplied_rows if row not in current_rows]
        if not qualified:
            result = super().build(policy, obs, visual_report)
            self.rejected.extend({'goal_offset_m': copy.deepcopy(row.get('goal_offset_m')),
                'reason': 'proposal_visual_goal_not_currently_qualified',
                'source': 'visual_goal_field'} for row in stale)
            return self.summary() if stale else result

        prospective = copy.copy(policy)
        prospective.prediction = copy.deepcopy(prediction)
        prospective.prediction['phase'] = 'approach'
        report = copy.deepcopy(visual_report)
        report['candidates'] = qualified
        self.phase_transition = {
            'from_phase': 'recover', 'offered_phase': 'approach',
            'qualified_visual_candidates': len(qualified),
            'selection_priority': 'current_ranked_visual_goals_before_short_normal_recovery',
            'explicit_model_choice_required': True,
            'live_phase_changed': False, 'physical_debt_replenished': False,
            'full_route_and_first_command_checks_unchanged': True,
        }
        super().build(prospective, obs, report)
        # Preserve the supplied report for diagnosis even if some stale rows
        # were excluded. Exact fresh rows alone reach the inherited preview.
        self.visual_report = copy.deepcopy(visual_report)
        self.rejected.extend({'goal_offset_m': copy.deepcopy(row.get('goal_offset_m')),
            'reason': 'proposal_visual_goal_not_currently_qualified',
            'source': 'visual_goal_field'} for row in stale)
        return self.summary()

    def summary(self):
        result = super().summary()
        result['version'] = 'observation-bound-explicit-phase-transition-proposals-v34'
        result['phase_transition'] = copy.deepcopy(self.phase_transition)
        return result
