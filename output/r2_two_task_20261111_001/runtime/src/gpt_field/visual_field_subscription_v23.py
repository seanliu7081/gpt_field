"""V23 retains the wire and describes bounded model-selected uncertainty egress."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v22 import (
    VisualFieldSubscriptionClientV22, output_schema, CANDIDATE_LABEL,
    shared_guidance as guidance_v22, geometry_semantics as semantics_v22,
)

VERSION = 'visual-field-rgb-joints-v23-bounded-uncertainty-egress-v1'
HISTORY_VERSION = 'visual-v23-uncertainty-egress-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v22()
    result['version'] = 'candidate-bound-uncertainty-egress-v23'
    result['initial_uncertainty_egress'] = {
        'explicit_model_selection_required': True,
        'nominal_tool_cores_must_be_clear': True,
        'empty_tool_required': True,
        'target_contact_must_be_none': True,
        'outside_target_contact_band_required': True,
        'all_scene_hypotheses_retained': True,
        'new_halo_entry_allowed': False,
        'unknown_region_exemption': False,
        'contact_exemption': False,
        'route_extent_limit_m': .04,
        'maximum_command_m': .003,
        'initial_measured_probe_limit_m': .004,
        'fresh_favorable_image_evidence_required_after_probe': True,
        'previous_shared_uncertain_debt_included': True,
        'contextual_budget_reset_on_rename_or_reseed': False,
        'clearance_certified': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v22(task, demonstration)
    cue = '''UNCERTAINTY EGRESS. With an empty tool, target_contact=none and outside the target contact band, if nominal tool cores are clear but existing non-target uncertainty covers the origin, consider an executable recovery_options candidate with source=initial_uncertainty_egress. Explicitly choose Field phase=recover, motion_context.translation_mode=separate and purpose=reference_repair, with subgoal_update new or replace as required. Copy its locally previewed goal, tolerance, command and travel limits. No obstacle deletion/shrink, new halo entry, unknown-region or contact exemption is allowed. The contextual route is <=40 mm, commands <=3 mm: only the first <=4 mm measured probe may bootstrap; further travel needs fresh rolling favorable image evidence. The total 40 mm includes prior shared uncertain travel, never a fresh allowance from a renamed ID, reseed, refresh or new goal. Checks can refuse or end the route early. These are model hypotheses, not measured depth or clearance. Prefer a valid offered candidate to repeated static retries; if none is executable, preserve the refusal honestly. On Field arrival, resume the actual task; leaving uncertainty is not task completion. Normal weighted Field planning remains unchanged.

'''
    return text.replace('LEDGER/HISTORY.', cue + 'LEDGER/HISTORY.', 1)


class VisualFieldSubscriptionClientV23(VisualFieldSubscriptionClientV22):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v23', schema=output_schema())
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
