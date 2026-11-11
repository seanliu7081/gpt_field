"""Explicit route-family selection over the V24 scene/reference protocol.

The selector dispatches a planner; it grants neither motion permission nor a
fresh physical allowance. Raw responses preserve the exact model choice.
"""
from __future__ import annotations

import copy

from .astra import SchemaError, _canonical
from .visual_field_subscription import _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v24 import (
    VisualFieldSubscriptionClientV24, output_schema as schema_v24,
    shared_guidance as guidance_v24, geometry_semantics as semantics_v24,
    FEEDBACK_VERSION, CANDIDATE_LABEL, compact_older_contexts,
)

VERSION = 'visual-field-rgb-joints-v25-explicit-recovery-route-v1'
HISTORY_VERSION = 'visual-v25-explicit-route-recent-two-turns-v1'
RECOVERY_ROUTES = ('normal', 'reverse_observed', 'initial_uncertainty_egress')


def output_schema(controller='hybrid'):
    result = schema_v24(controller)
    result['properties']['recovery_route'] = {'type': 'string', 'enum': list(RECOVERY_ROUTES)}
    result['required'].append('recovery_route')
    return result


def geometry_semantics():
    result = semantics_v24()
    result['version'] = 'explicit-recovery-route-current-geometry-v25'
    result['recovery_route_selection'] = {
        'wire_field': 'recovery_route', 'values': list(RECOVERY_ROUTES),
        'required_for_every_response': True, 'phase_or_purpose_dispatch': False,
        'preview_and_actual_same_family': True,
        'retained_world_goal_preserves_field_route': True,
        'direct_or_gripper_does_not_clear_retained_field_route': True,
        'current_candidate_geometry_and_observation_rechecked': True,
        'physical_budget_reset': False, 'clearance_certified': False,
    }
    result['finite_reverse_route'] = {
        'explicit_route': 'reverse_observed', 'original_route_travel_limit_m': .008,
        'per_direction_limit_m': .008, 'shared_uncertain_travel_limit_m': .020,
        'maximum_command_m': .003, 'positive_gap_required_at_4mm': False,
        'four_mm_observation_is_telemetry_not_new_travel_credit': True,
        'original_fixed_goal_geometry_and_current_reference_binding_required': True,
        'full_scene_unknown_target_contact_checks_retained': True,
        'further_worsening_lost_reference_or_domain_change_still_stops': True,
        'previous_measured_motion_and_hold_drift_charged': True,
        'rename_reseed_replace_or_refresh_renews_allowance': False,
        'conditional_model_geometry_exposure': True, 'clearance_certified': False,
    }
    result['recent_response_evidence'] = {
        'contiguous_positive_nondeclining_samples': 2,
        'pair_aggregate_must_be_distinguishable': True,
        'unchanged_full_history_empirical_envelope': True,
        'requested_command_amplified': False, 'physical_allowance_added_m': 0.,
        'responsiveness_is_clearance': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v24(task, demonstration)
    old = 'existing shared/heading allowance and a 4 mm observed-improvement checkpoint.'
    assert old in text
    text = text.replace(old, 'the original at-most-8 mm route travel, existing 8 mm per-direction debt and shared 20 mm uncertain-motion allowance. For a fully bound short reverse, the 4 mm observation reports improvement/inconclusive/worsening without requiring positive gap improvement to unlock its already bounded route. This is conditional model-geometry exposure, not clearance: current further worsening, lost/contradicted references, domain changes, unknowns and target/contact restrictions still stop it. Every measured move and hold drift consumes the same allowance; reseeding, renaming, refreshing or replacing a goal cannot renew it.')
    begin = text.index('UNCERTAINTY EGRESS.')
    text = text[:begin] + '''ROUTE SELECTION. Every response requires recovery_route=normal|reverse_observed|initial_uncertainty_egress. normal uses ordinary Field planning, including ordinary task recovery; Direct and gripper always specify normal. The other two routes require Field with phase=recover and translation_mode=separate. A current offer sourced from reverse_observed_worsening_motion uses reverse_observed: copy its checked goal/limits and required_recovery_route; purpose may describe task_translation or reference_repair without changing dispatch. Only an initial_uncertainty_egress offer uses that route with purpose=reference_repair. Phase/purpose labels alone never select a route. A route selector is not clearance or permission: actual current-state geometry, contact, image support and persistent budgets still apply. Read recovery_route_binding. Retaining an unfinished Field subgoal also retains its route, including across a temporary Direct/gripper repair; choose new/replace with an explicit reason to change route family. A blocked short reverse is not a reason to relabel it initial egress or repeatedly resend an unchanged goal.

''' + text[begin:]
    text = text.replace('Explicitly select Field recover/separate/reference_repair with target_contact=none',
        'Explicitly select recovery_route=initial_uncertainty_egress and Field recover/separate/reference_repair with target_contact=none')
    return text


class VisualFieldSubscriptionClientV25(VisualFieldSubscriptionClientV24):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v25', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        route = data['recovery_route']
        if route != 'normal':
            context = data['motion_context']
            if (data['controller'] != 'visual_field' or data['phase'] != 'recover'
                    or context['translation_mode'] != 'separate'):
                raise SchemaError('Non-normal recovery_route requires visual_field, phase recover and translation_mode separate')
            if route == 'initial_uncertainty_egress' and context['purpose'] != 'reference_repair':
                raise SchemaError('initial_uncertainty_egress requires purpose reference_repair')
        legacy = copy.deepcopy(data)
        legacy.pop('recovery_route')
        result = super()._validate_v5_response(legacy, grid)
        result['recovery_route'] = route
        result['raw_wire_prediction'] = copy.deepcopy(data)
        return result

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        return result
