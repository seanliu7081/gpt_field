"""V22 preserves the wire and makes recovery proposals match local planning."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v21 import (
    VisualFieldSubscriptionClientV21, output_schema, CANDIDATE_LABEL,
    shared_guidance as guidance_v21, geometry_semantics as semantics_v21,
)

VERSION = 'visual-field-rgb-joints-v22-heading-constrained-recovery-v1'
HISTORY_VERSION = 'visual-v22-heading-constrained-recovery-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v21()
    result['version'] = 'candidate-bound-heading-constrained-recovery-v22'
    result['heading_constrained_recovery'] = {
        'model_selected_goal_required': True,
        'actual_proposal_direction_checked': True,
        'full_scene_contact_reference_checks_retained': True,
        'debt_reset': False,
        'clearance_certified': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v21(task, demonstration)
    old = ('For an offered reversal, use its exact goal_offset_m direction with the offered 8 mm goal '
           'and goal_tolerance_m=.003; a long diagonal goal may leave the admissible cone. Prior worsening remains.')
    new = ('For an offered reversal, explicitly select its recommended_goal_offset_m and '
           'recommended_goal_tolerance_m (normally 8 mm and .003 m), with separate mode. '
           'A model-selected short reverse goal uses a continuous straight checked Field route; normal grid planning stays unchanged. '
           'Recovery constrains the ACTUAL candidate heading; an admissible endpoint alone '
           'does not authorize a different grid step. Read heading_recovery for the fixed route and current admission. '
           'Every executed segment still needs current full-scene, '
           'contact and reference checks. No direction or travel allowance is reset; prior worsening remains.')
    assert old in text
    return text.replace(old, new)


class VisualFieldSubscriptionClientV22(VisualFieldSubscriptionClientV21):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v22', schema=output_schema())
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
