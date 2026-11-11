"""Same-wire bounded Field evidence acquisition and paid-history handoff review."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v27 import (
    VisualFieldSubscriptionClientV27, output_schema, shared_guidance as guidance_v27,
    geometry_semantics as semantics_v27, FEEDBACK_VERSION, CANDIDATE_LABEL,
    compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v28-paid-evidence-handoff-v1'
HISTORY_VERSION = 'visual-v28-paid-evidence-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v27()
    result['version'] = 'bounded-heading-evidence-and-paid-reference-review-v28'
    result['heading_evidence_acquisition'] = {
        'scope': 'current_checked_normal_empty_tool_no_contact_field_candidate',
        'fresh_full_scene_geometry_required': True,
        'finite_unvalidated_acquisition_is_not_validated_response': True,
        'existing_heading_measured_limit_m': .008,
        'existing_shared_measured_limit_m': .020,
        'physical_budget_replenished': False,
        'warnings_and_current_risk_retained': True,
        'clearance_certified': False,
    }
    result['paid_handoff_reference_review'] = {
        'source': 'already_acknowledged_RGB_state_history',
        'maximum_frames': 32,
        'explicit_current_candidate_selection_required': True,
        'automatic_reference_promotion': False,
        'new_motion_for_board': False,
        'physical_budget_replenished': False,
    }
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v27(task, demonstration).replace(
        'Local Field may reconcile a retained warning only with',
        'The supported-response Field branch reconciles a retained warning only with')
    position = text.index('ROUTE SELECTION.')
    return text[:position] + """PAID EVIDENCE AT A HANDOFF. Read reference_repair at projected-response, geometry or reverse-risk interruptions, including heading-evidence handoffs. A current numbered board may use motion already executed; inspecting it adds no calibration motion. The interruption does not prove the old feature is wrong. Choose reference_candidate_id only when the original two RGB views identify a rigid distal tool feature; keep the full-tool ROI truthful and use the existing explicit seed contract. If only the correspondence changes, retain the fixed Field subgoal and keep unchanged inventory geometry null. Selection never clears risk, resets physical debt or validates the new response. Missing or ambiguous candidates are not permission to invent one. In addition to that supported-response branch, local Field may gather a finite amount of heading evidence from an otherwise checked normal empty-tool path under the existing 8 mm heading/20 mm shared measured allowances. Unvalidated acquisition is distinct from supported prediction and stops on current contrary evidence or its remaining budget; it is not certified clearance. Prefer a useful permitted Field continuation to repeated zero holds; if genuinely blocked, return a valid selected controller branch with a clear reason. Direct still requires exactly five actions; never select direct while leaving direct_actions null.

""" + text[position:]


class VisualFieldSubscriptionClientV28(VisualFieldSubscriptionClientV27):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v28', schema=output_schema())
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
