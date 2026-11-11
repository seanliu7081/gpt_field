"""Same-wire Field control with explicit temporal measurement semantics.

Model edits change a representation, not observed robot motion. Current risk,
prior comparable evidence and all physical allowances remain authoritative.
"""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v25 import (
    VisualFieldSubscriptionClientV25, output_schema, shared_guidance as guidance_v25,
    geometry_semantics as semantics_v25, FEEDBACK_VERSION, CANDIDATE_LABEL,
    compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v26-comparable-measurement-epochs-v1'
HISTORY_VERSION = 'visual-v26-comparable-measurement-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v25()
    result['version'] = 'comparable-temporal-risk-measurement-v26'
    result['measurement_representation'] = {
        'accepted_feature_or_extent_revision_is_observed_motion': False,
        'temporal_comparison_requires_comparable_representation': True,
        'current_revised_extent_and_all_scene_hypotheses_retained': True,
        'prior_valid_worsening_retained': True,
        'current_conflict_and_unknown_contact_checks_retained': True,
        'new_epoch_grants_motion_permission': False,
        'new_epoch_replenishes_physical_allowance': False,
        'clearance_certified': False,
    }
    result['neighbor_descriptor_matching'] = {
        'scope': 'opt_in_projected_neighbor_tracker_only',
        'equality_space': 'exact_bilinear_sampled_grayscale_descriptor_not_whole_RGB',
        'legacy_texture_uniqueness_forward_backward_and_consensus_checks_retained': True,
        'exact_same_coordinate_sampled_descriptor_may_correct_local_subpixel_bias': True,
        'static_object_or_camera_assumed': False,
        'tool_payload_and_inventory_trackers_changed': False,
        'risk_thresholds_or_physical_allowances_changed': False,
        'clearance_certified': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v25(task, demonstration)
    marker = 'ROUTE SELECTION.'
    position = text.index(marker)
    return text[:position] + """MEASUREMENT EPOCHS. Changing a selected feature or a tool/neighbor rectangle revises the image representation; it is not robot motion. Compare temporal gap changes only within a comparable representation. An accepted revision starts a new measurement segment without erasing prior valid worsening or refunding route, heading or shared travel. The revised current extent remains active: an enlarged rectangle can reveal current risk even when no motion occurred. Read translation_guard.measurement_representation for the epoch, changed identities and comparable reverse segments; a new epoch or inconclusive gap alone grants neither clearance nor execution permission. Retain accurate scene geometry and a valid fixed world goal through reference-only repair. Do not shrink extents, repeatedly reseed or request holds merely to clear a historical warning. Genuine fresh closing motion, lost/contradictory support, unknown regions and target/contact constraints still stop execution.

""" + text[position:]


class VisualFieldSubscriptionClientV26(VisualFieldSubscriptionClientV25):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v26', schema=output_schema())
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
