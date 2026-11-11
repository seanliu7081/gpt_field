"""V14 presentation-only removal of repeated diagnostic detail.

Current guard, recovery options, reference evidence, scene hypotheses, target
permissions, attachment and persistent goals remain authoritative. Unknown keys
are retained. Raw observations, model outputs and local traces are untouched.
"""
from __future__ import annotations

import copy
import json
from .astra import _canonical
from .visual_feedback_compact_v12 import compact_payload_feedback as compact_v12

VERSION = "observable-feedback-actionable-v14-v1"


def _admission(value):
    result = copy.deepcopy(value)
    if not isinstance(result, dict) or not isinstance(result.get('diagnostics'), dict):
        return result
    diag = result['diagnostics']
    # This is the earlier decision's snapshot. Preserve its actual veto, caps,
    # debts and risk outcome; the full CURRENT response/geometry is supplied in
    # visual_feedback.translation_guard, not fabricated as equal old evidence.
    for name in ('version', 'geometry_source', 'budget_semantics', 'limitations',
                 'tool_box_px', 'envelope_revision', 'nominal_regions', 'response', 'cumulative_response'):
        diag.pop(name, None)
    if isinstance(diag.get('neighbors'), list):
        diag['neighbors'] = [{key:copy.deepcopy(item) for key,item in neighbor.items()
                             if key not in ('box_px',)} for neighbor in diag['neighbors']]
    diag['snapshot_scope'] = 'decision-time veto/budget evidence; current full geometry and response are in translation_guard'
    return result


def compact_feedback(value):
    if not isinstance(value, dict) or 'feedback_ref' in value:
        return copy.deepcopy(value)
    result = copy.deepcopy(value)
    field = result.get('field')
    if isinstance(field, dict):
        envelope = field.get('robot_tool_envelope')
        if isinstance(envelope, dict) and isinstance(envelope.get('parts'), list):
            envelope['part_names'] = [part.get('name') for part in envelope.pop('parts')]
            envelope['part_geometry_in_local_diagnostics'] = True
        if 'translation_admission' in field:
            field['translation_admission'] = _admission(field['translation_admission'])
        # The field's static target snapshot may be a strict subset of the
        # current target record, which additionally describes payload lifecycle.
        target, current = field.get('target_contact_geometry'), result.get('target_contact_geometry')
        if isinstance(target, dict) and isinstance(current, dict) and all(
                key in current and _canonical(item)==_canonical(current[key]) for key,item in target.items()):
            field['target_contact_geometry'] = {'feedback_ref': 'target_contact_geometry',
                'meaning': 'same target geometry; current record additionally includes payload lifecycle'}
        for key in ('orientation_hold',):
            if isinstance(field.get(key), dict):
                for name in ('error_frame','gain','target_source','target_xyzw','osc_goal_refresh_epsilon_rad'):
                    field[key].pop(name, None)
    alignment = result.get('contact_alignment')
    if isinstance(alignment, dict) and alignment.get('active') is False:
        # Null coordinates and empty fits are not useful when never configured.
        # Actual nonempty errors/reasons/counters and unknown future keys stay.
        for key in ('limitations','maximum_goal_change_m'):
            alignment.pop(key, None)
        for key in ('image_error_px','error_component_uncertainty_px','last_correction','current_points_px','point_uncertainty_px'):
            if alignment.get(key) is None or alignment.get(key)=={}:
                alignment.pop(key, None)
    result['feedback_representation'] = VERSION
    return result


def _record(value):
    if isinstance(value, list):
        return [_record(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {key:compact_feedback(item) if key in ('visual_feedback','end_visual_feedback') else _record(item)
            for key,item in value.items()}


def compact_payload_feedback(payload):
    result = compact_v12(payload)
    for message in result.get('input', []):
        if message.get('role')!='user':
            continue
        for part in message.get('content', []):
            if part.get('type')!='input_text':
                continue
            try:
                record=json.loads(part['text'])
            except (ValueError,TypeError,KeyError):
                continue
            if isinstance(record, dict) and record.get('type') in ('observation','visual_execution_block','visual_history_summary'):
                compact=_record(record)
                if compact!=record:
                    part['text']=_canonical(compact)
    return result
