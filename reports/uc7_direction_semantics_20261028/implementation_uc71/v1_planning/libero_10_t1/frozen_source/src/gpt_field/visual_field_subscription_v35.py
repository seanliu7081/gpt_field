"""Explicit wrist-bearing navigation with lossless current-feedback factoring."""
from __future__ import annotations

import json

from .astra import _canonical
from .visual_feedback_compact_v35 import (
    compact_current_feedback, VERSION as FEEDBACK_VERSION, SEMANTICS,
)
from .visual_field_subscription_v34 import (
    VisualFieldSubscriptionClientV34, output_schema, HISTORY_VERSION,
    shared_guidance as guidance_v34, geometry_semantics as semantics_v34,
)

VERSION = 'visual-field-rgb-joints-v35-separate-wrist-navigation-v1'
WRIST_GUIDANCE = (
    'WRIST NAVIGATION. A current field offer may name objective_source=wrist_bearing. '
    'Its goal reduces the actual tracked target pixel error to your declared desired_target_uv '
    'using acknowledged measured robot response, prospectively checked predictions and a '
    'bounded observed motion subspace. The desired UV is a goal coordinate, not an observed '
    'anchor or a pinch center. This objective can coexist with a rigid_transit_proxy because '
    'the wrist target is independently observed; no proxy-target anatomy error is used. '
    'A qualified wrist offer is only noncontact approach/navigation: it estimates neither '
    'depth nor grasp readiness, and grants no contact, attachment or success evidence. '
    'Inspect the current views and intended bearing before selecting it. Copy a current '
    'offer contract exactly; all actual route, first-command and subsequent checks remain. '
    'Retain the still-current wrist reference and desired bearing across a mere external '
    'reference repair; change them explicitly when identity, tracking or intended objective '
    'changes. Do not use a convenient image point as an invented observed feature. '
    'execution_limit_sources separates your declared limits from the effective intent and '
    'runtime scene cap. When the scene cap is null, partial object visibility alone does '
    'not mandate a 15mm subgoal; choose useful travel within current physical checks. '
    'An empty optional offer list does not prove that every ordinary Field intent is blocked. '
    'Use current refusal reasons, RGB and measured response to propose a justified ordinary '
    'intent or scene revision. Report blockage when no justified motion remains; repeating '
    'static confirmations creates no evidence or travel allowance.'
)


def shared_guidance(task, demonstration=False):
    text = guidance_v34(task, demonstration)
    # Correct inherited overbroad language in this opt-in version only.
    text = text.replace('Unobservable scene caps remain15mm travel/5mm command.',
        'A runtime scene cap, when active, remains15mm travel/5mm command; '
        'read execution_limit_sources for whether that cap is active.')
    text = text.replace(
        'No executable offer means report the current blockage honestly; no extra confirmation '
        'motion, new perception input or automatic recovery is authorized.',
        'No executable reverse offer means that reverse route is unavailable. Ordinary '
        'Field remains subject to its own current admission; no extra confirmation motion, '
        'new perception input or automatic recovery is authorized by an empty offer list.')
    return text + '\n\n' + WRIST_GUIDANCE + '\n\nCURRENT FEEDBACK. ' + SEMANTICS


def geometry_semantics():
    result = semantics_v34()
    result['version'] = 'separate-wrist-navigation-and-stable-boundary-evidence-v35'
    result['wrist_navigation_contract'] = {
        'objective': 'squared tracked-target pixel error to model-declared desired bearing',
        'training': 'acknowledged measured motion; prediction before training on its outcome',
        'independent_of_external_reference_role': True,
        'current_target_reference_and_robot_domain_required': True,
        'observed_subspace_and_same_heading_validation_required': True,
        'maximum_proposed_distance_m': .030,
        'explicit_model_selection_required': True,
        'full_route_and_actual_action_checks_unchanged': True,
        'desired_coordinate_is_observed_anchor': False,
        'depth_or_contact_alignment_estimated': False,
        'automatic_motion_or_task_success': False,
    }
    result['boundary_representation_contract'] = {
        'same_current_hand_track_and_extent_reuses_original_source_corners': True,
        'changed_track_extent_state_or_image_keeps_inherited_revision': True,
        'adverse_history_and_physical_budget_preserved': True,
        'boundary_continuity_is_clearance': False,
    }
    result['current_feedback_presentation'] = {
        'version': FEEDBACK_VERSION, 'exact_reversible_same_observation': True,
        'numerical_values_and_rejections_preserved': True,
        'raw_history_and_controller_input_unchanged': True,
    }
    return result


class VisualFieldSubscriptionClientV35(VisualFieldSubscriptionClientV34):
    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format']['name'] = 'visual_field_rgb_joints_v35'
        # Only the last/current observation. No stored controller feedback,
        # historical raw assistant response, image or earlier observation changes.
        for message in reversed(payload['input']):
            if message.get('role') != 'user':
                continue
            for part in message.get('content', []):
                if part.get('type') != 'input_text':
                    continue
                try:
                    record = json.loads(part['text'])
                except (KeyError, TypeError, ValueError):
                    continue
                if not isinstance(record, dict) or record.get('type') != 'observation':
                    continue
                context = record.get('observable_context') or {}
                feedback = context.get('visual_feedback')
                if isinstance(feedback, dict):
                    context['visual_feedback'] = compact_current_feedback(feedback)
                    part['text'] = _canonical(record)
                return payload
        return payload
