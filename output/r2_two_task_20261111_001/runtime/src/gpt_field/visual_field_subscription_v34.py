"""Explicit task resumption and distinct paid reference choices, same wire."""
from __future__ import annotations

from .visual_field_subscription_v33 import (
    VisualFieldSubscriptionClientV33, output_schema,
    shared_guidance as guidance_v33, geometry_semantics as semantics_v33,
    HISTORY_VERSION, FEEDBACK_VERSION,
)

VERSION = 'visual-field-rgb-joints-v34-explicit-task-resumption-v1'
TASK_RESUMPTION_GUIDANCE = (
    'TASK RESUMPTION. Local recovery arrival is a handoff, not task completion. '
    'A current qualified visual-goal offer can explicitly change recover to noncontact approach '
    'under the same full-route and first-command checks. When an appropriate checked task path '
    'is available, select it explicitly rather than repeating short recovery subgoals by habit. '
    'Copy the current offer contract exactly, including its phase and new/replace subgoal update; '
    'no automatic phase change, action, clearance or travel credit follows from an offer. '
    'Field motion offers (field_proposal_id) and paid visual-reference candidates '
    '(reference_candidate_id) are different choices and cannot be selected together. '
    'A board offered during weak-response review contains possible image correspondences, '
    'not verified identities: inspect current RGB and make an explicit semantic choice within '
    'the existing anatomy/proxy contract. A reference choice alone is not a motion offer. '
    'Setting external or wrist references to null drops that active support; it does not signal '
    'completion or repair a lost track. Retain still-current references, or explicitly select '
    'a current candidate for a feature you can identify; do not pretend unavailable support is '
    'current. Reobserve honestly when support cannot be identified. All geometry, contact, '
    'reference-domain and measured-budget checks remain in force.'
)


def shared_guidance(task, demonstration=False):
    return guidance_v33(task, demonstration) + '\n\n' + TASK_RESUMPTION_GUIDANCE


def geometry_semantics():
    result = semantics_v33()
    result['version'] = 'explicit-task-resumption-and-paid-reference-choice-v34'
    result['recovery_to_approach_proposals'] = {
        'from_phase': 'recover', 'offered_phase': 'approach',
        'current_qualified_visual_goal_required': True,
        'explicit_model_choice_required': True,
        'full_route_and_first_command_checks_unchanged': True,
        'automatic_stage_change': False, 'physical_budget_replenished': False,
        'clearance_certified': False,
    }
    result['reference_choice_contract'] = {
        'motion_offer_and_reference_candidate_are_distinct': True,
        'current_observation_binding_required': True,
        'explicit_semantic_choice_required': True,
        'null_reference_means_task_complete': False,
        'automatic_identity_promotion': False,
    }
    result['final_feedback_contract'] = {
        'read_only_summary': True, 'prepares_new_motion_offers': False,
        'requests_model_inference': False,
        'observer_and_control_updates_keep_original_order': True,
        'only_intermediate_diagnostic_materializations_suppressed': True,
        'complete_current_row_rendered_once_after_observers': True,
        'across_observation_cache': False,
        'control_timing_excludes_only_video': True,
    }
    return result


class VisualFieldSubscriptionClientV34(VisualFieldSubscriptionClientV33):
    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format']['name'] = 'visual_field_rgb_joints_v34'
        return payload
