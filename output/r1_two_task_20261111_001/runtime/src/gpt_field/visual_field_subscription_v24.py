"""V24: bounded uncertain route exposure and concise historical results.

The V23 wire, observation-bound candidate resolution and subscription retries are
unchanged. Only older user engineering-feedback presentation is summarized; the
current context, recent RGB/state/joints and raw assistant outputs stay exact.
"""
from __future__ import annotations

import copy
from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v23 import (
    VisualFieldSubscriptionClientV23, output_schema, CANDIDATE_LABEL,
    shared_guidance as guidance_v23, geometry_semantics as semantics_v23,
)
from .visual_feedback_compact_v21 import expanded_observation_feedback, restore_feedback

VERSION = 'visual-field-rgb-joints-v24-bounded-route-exposure-v1'
HISTORY_VERSION = 'visual-v24-previous-action-result-recent-two-turns-v1'
FEEDBACK_VERSION = 'current-full-older-action-result-v24-v1'


def geometry_semantics():
    result = semantics_v23()
    result['version'] = 'candidate-bound-model-route-exposure-v24'
    egress = result['initial_uncertainty_egress']
    egress['initial_measured_probe_limit_m'] = .004
    egress['qualified_causal_reference_required_after_probe'] = True
    egress['single_reference_confirmation_limit_m'] = .004
    egress['fresh_favorable_image_evidence_required_after_probe'] = False
    egress['bounded_model_geometry_exposure'] = True
    egress['inconclusive_observations_consume_persistent_allowance'] = True
    egress['absence_of_worsening_certifies_clearance'] = False
    result['reference_review_at_existing_handoff'] = {
        'extra_motion_or_budget_granted': False,
        'candidate_selection_is_semantic_model_claim': True,
        'reference_only_repair_preserves_world_geometry_and_goal': True,
    }
    result['older_feedback_presentation'] = FEEDBACK_VERSION
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v23(task, demonstration)
    begin, end = text.index('UNCERTAINTY EGRESS.'), text.index('LEDGER/HISTORY.')
    text = text[:begin] + '''UNCERTAINTY EGRESS. A current recovery_options offer with source=initial_uncertainty_egress is a bounded model-geometry route, NOT observed clearance. Explicitly select Field recover/separate/reference_repair with target_contact=none and an empty tool; copy its previewed goal/tolerance/command/travel limits. Nominal tool cores, unknown regions, target/contact restrictions, no new halo entry and current-segment checks remain. The fixed route must leave the initial uncertainty region; the total <=40 mm allowance includes previously spent uncertain travel and commands are <=3 mm. After the initial <=4 mm reference probe, continuation needs a qualified causal controlled-feature witness, not favorable obstacle-gap changes every4mm. An explicitly selected candidate can receive one <=4 mm confirmation within the SAME total allowance; repeated selection cannot renew it. Read controlled_reference_witness and route_reference_selection. Inconclusive obstacle vision consumes finite allowance; absence of worsening is NOT evidence of free space. Strong adverse evidence, lost required support or a current constraint can stop the route. No geometry shrink/deletion/reclassification, ID change, reseed, refresh or new goal refunds travel. Local arrival ends only this subgoal; resume the task without claiming completion.

REFERENCE HANDOFF. At an existing interruption, inspect reference_repair and its current observation-bound candidate board even when small-motion evidence is inconclusive. This review grants no extra motion or allowance. Prefer a candidate visibly on the distal rigid gripper/finger over a fixed base, proximal link, shadow or merely moving patch; otherwise state that no candidate can be identified. Supply a freshly inspected whole-tool ROI and follow exact candidate seed/null-coordinate rules. During reference-only repair, keep each unchanged remembered scene_inventory object's geometry=null instead of re-estimating its metric box. Current visible ROIs may be updated honestly; never invent a geometry revision to bypass a refusal. Preserve the existing fixed world subgoal with subgoal_update=retain when its phase/target/axes remain appropriate; do not replace a task goal merely because its visual reference changed.

''' + text[end:]
    text = text.replace('Latest feedback is self-contained; older snapshots may be exact patches against it.',
        'Latest feedback is complete and authoritative. Older observation contexts contain explicitly historical previous-action/result summaries, not complete past engineering state; full original records remain local.')
    return text


def _pick(value, keys):
    return {key: copy.deepcopy(value[key]) for key in keys if isinstance(value, dict) and key in value}


def historical_result_summary(feedback, *, observation_index):
    """Lossy historical presentation, never used by policy/validators or latest state."""
    value = restore_feedback(feedback)
    summary = {
        'type': 'historical_action_result_summary', 'version': FEEDBACK_VERSION,
        'source_observation_index': observation_index,
        'semantics': 'Historical previous-action/result only. Use the latest full context for all current constraints, scene hypotheses and budgets. Full historical feedback remains in local logs.',
    }
    summary.update(_pick(value, ('type', 'step', 'action', 'before_state', 'after_state',
        'commanded_translation_m', 'measured_translation_m', 'same_failure_repeats',
        'task_success_certified', 'last_field_handoff')))
    summary['type'] = 'historical_action_result_summary'
    selections = {
        'field': ('status', 'stop_reason', 'actions', 'actions_executed', 'goal_xyz', 'goal_xyz_m'),
        'visual': ('status', 'reason', 'need_refresh', 'refresh_reason', 'attachment_state'),
        'gripper_execution': ('status', 'reason', 'command', 'executed_actions', 'grasp_certified'),
        'scene_coverage': ('accepted', 'reason', 'need_refresh'),
        'task_progress': ('status', 'reason', 'review_required', 'goal_review_required'),
        'uncertainty_egress': ('reason', 'spent_measured_m', 'egress_measured_m',
            'total_remaining_m', 'observation_tranche_remaining_m', 'fresh_progress_grants'),
        'reference_support': ('status', 'reason', 'review_recommended', 'diagnostic_required'),
        'reference_repair': ('status', 'reason', 'review_required', 'observation_binding'),
        'subgoal_ledger': ('status', 'reason', 'active_id', 'active_goal_xyz_m'),
    }
    for name, keys in selections.items():
        selected = _pick(value.get(name), keys)
        if selected:
            summary[name] = selected
    guard = value.get('translation_guard') or {}
    admission = _pick(guard.get('last_admission'), ('accepted', 'reason', 'cap_m', 'need_refresh'))
    if admission:
        summary['translation_admission'] = admission
    visual = value.get('visual') or {}
    attachment = _pick(visual.get('attachment'), ('state', 'status', 'reason', 'observed', 'attached'))
    if attachment:
        summary['attachment'] = attachment
    return summary


def compact_older_contexts(payload):
    """Keep latest text byte-exact; resolve same-request patches before summarizing."""
    result = copy.deepcopy(payload)
    snapshots = expanded_observation_feedback(result)
    for part, record, feedback in snapshots[:-1]:
        context = copy.deepcopy(record['observable_context'])
        # Unknown context fields, output/transport repair and request metadata
        # survive. Only historical visual engineering feedback is summarized.
        context['visual_feedback'] = historical_result_summary(feedback,
            observation_index=record.get('observation_index'))
        record['observable_context'] = context
        part['text'] = _canonical(record)
    return result


class VisualFieldSubscriptionClientV24(VisualFieldSubscriptionClientV23):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v24', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        return compact_older_contexts(payload)

    def _history_input(self, payload):
        # The base transport settles history from its sent payload. Restore the
        # original retained observation messages here so this request-only
        # summary never overwrites full in-memory history or raw assistant turns.
        originals = {}
        for message in self._history:
            record = _text_record(message)
            if message.get('role') == 'user' and record and record.get('type') == 'observation':
                originals[record.get('observation_index')] = message
        result = copy.deepcopy(super()._history_input(payload))
        for index, message in enumerate(result):
            record = _text_record(message)
            if message.get('role') == 'user' and record and record.get('type') == 'observation':
                original = originals.get(record.get('observation_index'))
                if original is not None:
                    result[index] = copy.deepcopy(original)
        return result

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_VERSION, older_feedback_presentation=FEEDBACK_VERSION)
        return result
