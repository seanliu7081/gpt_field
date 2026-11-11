"""Ordinary Field alternatives at an early visual-intent review.

These are explicit model-choice hypotheses, not validated image-goal offers.
Cloned previews keep actual geometry and accumulated physical limits. No
alternative is selected or executed here, and each actual action is rechecked.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_proposals_v31 import observation_binding
from .visual_field_proposals_v35 import FieldProposalBankV35


class OrdinaryIntentPreviewV36(FieldProposalBankV35):
    @staticmethod
    def _intent(policy, state, offset, *, recovery=False):
        result = FieldProposalBankV35._intent(policy, state, offset, recovery=recovery)
        # An explicit replacement may change axes. The prospective version
        # leaves the live goal, axis corridor, model response and debt untouched.
        result['motion_axes'] = 'xyz'
        return result


def ordinary_replan_previews(policy, obs, *, distance_m=.020):
    result = {'version': 'ordinary-intent-replan-previews-v36',
        'observation_sha256': observation_binding(obs), 'candidates': [],
        'explicit_model_choice_required': True, 'automatic_action': False,
        'validated_visual_improvement': False, 'clearance_certified': False,
        'physical_debt_replenished': False,
        'interpretation': ('These ordinary noncontact goals pass cloned full-route and '
            'first-command checks for this observation, retained references and scene. '
            'Select a task-relevant direction from current RGB using an ordinary Field '
            'response and explicit subgoal replacement; field_proposal_id stays null. '
            'They are not calibrated free space, proven visual corrections, contact '
            'alignment or permission after a reference/geometry change. Actual checks '
            'still run. Unlisted intents are not automatically forbidden.')}
    prediction = policy.prediction or {}
    external = (prediction.get('visual_references') or {}).get('external') or {}
    if (prediction.get('controller') != 'visual_field'
            or prediction.get('phase') not in ('approach', 'align')
            or prediction.get('target_contact') != 'none'
            or prediction.get('contact_alignment') is not None
            or external.get('moving_kind') != 'tool'
            or policy.local.intent is None
            or policy.local.intent.get('gripper_command') != 'hold'
            or policy.local.pending is not None or policy.pending is not None
            or policy.target_contact.summary().get('carried_payload', {}).get('active')):
        result['reason'] = 'ordinary_noncontact_approach_context_unavailable'
        return result
    if not np.isfinite(distance_m) or not .006 <= distance_m <= .030:
        raise ValueError('Replan preview distance must be between6 and30mm')
    preview = OrdinaryIntentPreviewV36()
    active = policy.subgoals.active
    update = {'action': 'replace' if active and not active['local_complete'] else 'new',
              'subgoal_id': active['subgoal_id'] if active and not active['local_complete'] else None,
              'reason_required': 'Explain the changed task direction from current RGB and ongoing objective review.'}
    for axis in range(3):
        for sign in (-1, 1):
            offset = np.zeros(3); offset[axis] = sign * distance_m
            offer, reason = preview._preview(policy, obs, offset)
            row = {'goal_offset_m': offset.tolist(), 'accepted': offer is not None,
                   'reason': reason, 'motion_axes': 'xyz',
                   'visual_effect_validated': False}
            if offer is not None:
                row.update(field_intent=copy.deepcopy(offer['field_intent']),
                    motion_context_overrides=copy.deepcopy(offer['motion_context_overrides']),
                    preview_action=copy.deepcopy(offer['preview_action']),
                    preview_status=offer['preview_status'],
                    full_route_geometry_sha256=offer['full_route_geometry_sha256'],
                    subgoal_update=copy.deepcopy(update))
            result['candidates'].append(row)
    result['reason'] = 'current_ordinary_intent_previews' if any(
        row['accepted'] for row in result['candidates']) else 'no_tested_axis_intent_admitted'
    return result
