"""Same-wire Field control with bounded constraint-aware route tracking."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v28 import (
    VisualFieldSubscriptionClientV28, output_schema, shared_guidance as guidance_v28,
    geometry_semantics as semantics_v28, FEEDBACK_VERSION, CANDIDATE_LABEL,
    compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v29-constrained-route-tracking-v1'
HISTORY_VERSION = 'visual-v29-constrained-route-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v28()
    result['version'] = 'constraint-aware-retained-route-field-v29'
    result['constraint_aware_tracking'] = {
        'scope': 'normal_empty_tool_noncontact_axis_constrained_field_with_warned_objects',
        'fixed_world_goal_and_intent_limits_preserved': True,
        'only_conflicting_locked_axis_recentering_may_be_omitted': True,
        'original_and_replacement_exact_candidate_proofs_required': True,
        'exact_selected_candidate_rechecked': True,
        'full_scene_current_reference_and_contact_checks_retained': True,
        'warned_object_model_gap_nondecrease_retained': True,
        'physical_budget_replenished': False,
        'arrival_is_actual_measured_state_not_projected_progress': True,
        'clearance_certified': False,
    }
    result['projected_warning_transit']['ambiguous_continuation_spends_existing_allowance'] = False
    result['projected_warning_transit']['unvalidated_acquisition_spends_existing_allowance'] = True
    result['validated_transit_lifecycle'] = {
        'scope': 'normal_empty_tool_no_contact_fixed_goal_field',
        'fresh_per_object_held_out_response_required': True,
        'responsive_robot_and_current_exact_model_constraints_required': True,
        'current_contrary_evidence_stops': True,
        'maximum_command_m': .003,
        'supported_travel_accounted_separately': True,
        'fixed_intent_action_and_travel_limits_retained': True,
        'old_acquisition_debt_preserved': True,
        'support_loss_falls_back_to_remaining_old_allowance': True,
        'physical_budget_replenished': False,
        'clearance_certified': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v28(task, demonstration).replace(
        'Admitted commands are at most 3 mm; measured travel spends existing 8 mm heading and 20 mm shared allowances plus intent remaining, without new credit.',
        'Admitted commands are at most 3 mm. Unvalidated acquisition spends existing 8 mm heading and 20 mm shared allowances plus intent remaining, without new credit. A distinct supported-transit state requires fresh held-out image response for every warned object, responsive robot motion, current exact model constraints and no contrary evidence. Supported travel is counted separately under the same fixed intent limits; old acquisition debt remains unchanged. Losing support falls back to whatever old allowance remains, never a fresh allowance. Read translation_guard.validated_intent_execution for this current execution state.')
    position = text.index('ROUTE SELECTION.')
    return text[:position] + """CONSTRAINT-AWARE FIELD TRACKING. For a normal empty-tool noncontact Field route with axis constraints and retained warnings, local tracking can omit a conflicting locked-axis recentering correction while making useful progress along the existing planned route. Read field.axis_tracking_constraint for the original and replacement command checks. The fixed world goal, axis corridor, arrival test and action/travel limits remain; every selected command still needs current geometry, reference, contact and physical-budget admission. Neither a model gap nor a tracking adjustment certifies clearance. Keep a valid retained subgoal when local progress continues. If no useful candidate satisfies the current constraints, inspect the reported conflict and explicitly revise the route or goal from current observations; do not repeat the same failed correction or rename objects to erase risk. A projected route position is not measured arrival or task completion.

""" + text[position:]


class VisualFieldSubscriptionClientV29(VisualFieldSubscriptionClientV28):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v29', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        return payload

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        return result
