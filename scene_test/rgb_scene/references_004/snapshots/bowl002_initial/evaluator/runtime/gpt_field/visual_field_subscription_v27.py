"""Same-wire task Field planning with bounded projected-warning reconciliation.

Model geometry and empirical image evidence remain distinct hypotheses. A
conditional normal-transit admission does not erase warnings or certify safety.
"""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v26 import (
    VisualFieldSubscriptionClientV26, output_schema, shared_guidance as guidance_v26,
    geometry_semantics as semantics_v26, FEEDBACK_VERSION, CANDIDATE_LABEL,
    compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v27-bounded-projected-warning-transit-v1'
HISTORY_VERSION = 'visual-v27-bounded-transit-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v26()
    result['version'] = 'bounded-projected-warning-reconciliation-v27'
    result['projected_warning_transit'] = {
        'scope': 'normal_empty_tool_no_contact_fixed_goal_field_transit_only',
        'translation_mode': 'transit',
        'current_full_scene_margin_sweep_required': True,
        'warned_object_model_clearance_must_not_decrease': True,
        'fresh_heading_specific_image_response_required': True,
        'per_warned_object_prospective_validation_required': True,
        'boundary_evidence_must_be_ambiguous_not_resolved_closing': True,
        'empirical_boundary_evidence_is_not_calibrated_confidence': True,
        'warnings_history_and_measured_debt_retained': True,
        'ambiguous_continuation_spends_existing_allowance': True,
        'maximum_command_m': .003,
        'existing_heading_measured_limit_m': .008,
        'existing_shared_measured_limit_m': .020,
        'intent_remaining_still_bounds_execution': True,
        'new_budget_or_clearance_certificate': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v26(task, demonstration)
    position = text.index('ROUTE SELECTION.')
    return text[:position] + """PROJECTED WARNINGS AND TRANSIT. A projected rectangle getting closer can disagree with the current model geometry. Neither view is a clearance certificate. For useful empty-tool translation, keep recovery_route=normal, translation_mode=transit, target_contact=none and the fixed world goal when justified. Local Field may reconcile a retained warning only with a current full-scene, full-margin candidate sweep, nondecreasing model clearance for every warned object, fresh prospectively validated heading-specific relative image flow for each object, and empirical boundary sensitivity that leaves closing ambiguous. Resolved closing still stops. Read translation_guard.projected_warning_transit and field.transit_reconciliation_geometry. Admitted commands are at most 3 mm; measured travel spends existing 8 mm heading and 20 mm shared allowances plus intent remaining, without new credit. Warnings/history/current geometry stay active. Ambiguous 2-D gap alone does not require automatic reversal. Stale/lost/contradictory references, unknown occupancy, target/contact constraints and exhausted budgets still require handoff. Do not shrink/delete obstacles, rename a route, or invent a safe image boundary to obtain permission. Prefer useful Field task motion when its actual checked selection is accepted; Direct five-step contact correction and explicit gripper control retain their existing roles. Local arrival and an admitted segment are not task completion.

""" + text[position:]


class VisualFieldSubscriptionClientV27(VisualFieldSubscriptionClientV26):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v27', schema=output_schema())
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
