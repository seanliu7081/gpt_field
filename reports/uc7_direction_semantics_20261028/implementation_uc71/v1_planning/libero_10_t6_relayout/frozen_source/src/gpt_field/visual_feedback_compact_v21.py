"""Lossless request presentation for repeated known feedback structures.

Only user feedback is rewritten. Current scene geometry remains directly in
inventory entries. Named column tables preserve every per-view value, including
unknown extensions, exact coordinates, failure flags and observation bindings.
The inverse is used for offline equivalence checks, never to modify controls.
"""
from __future__ import annotations

import copy
import json
from .astra import _canonical
from .visual_feedback_compact_v17 import (
    restore_feedback_delta, _deduplicate_older_feedback,
)

VERSION = 'observable-feedback-columnar-v21-v1'
_TABLE = 'presentation_table_v21'
_ROW = 'presentation_view_row_v21'


def _table(records, *, first=()):
    if not records or not all(isinstance(item, dict) for item in records):
        return None
    keys = set().union(*(item.keys() for item in records))
    columns = [key for key in first if key in keys]
    columns += sorted(keys - set(columns))
    rows, missing = [], []
    for row_index, item in enumerate(records):
        rows.append([copy.deepcopy(item.get(key)) for key in columns])
        missing.extend([row_index, col] for col, key in enumerate(columns) if key not in item)
    result = {'columns': columns, 'rows': rows}
    if missing:
        result['missing_cells'] = missing
    return result


def _untable(table):
    missing = {tuple(cell) for cell in table.get('missing_cells', [])}
    return [{key: copy.deepcopy(value) for col, (key, value) in enumerate(zip(table['columns'], row))
             if (index, col) not in missing} for index, row in enumerate(table['rows'])]


def compact_feedback(value):
    result = copy.deepcopy(value)
    if not isinstance(result, dict) or 'feedback_ref' in result or result.get('presentation_v21'):
        return result
    inventory = result.get('scene_inventory')
    if isinstance(inventory, dict) and isinstance(inventory.get('entries'), list):
        records, destinations = [], []
        for entry in inventory['entries']:
            if not isinstance(entry, dict) or not isinstance(entry.get('views'), dict):
                continue
            for view, status in entry['views'].items():
                # Unknown view names/fields are preserved, not interpreted.
                if not isinstance(status, dict) or _ROW in status:
                    continue
                records.append({'object_id': entry.get('id'), 'view': view, **status})
                destinations.append((entry['views'], view))
        # Identity columns use separate names; refuse a hypothetical extension
        # collision rather than overwrite that extension.
        if records and not any('object_id' in views[view] or 'view' in views[view]
                               for views, view in destinations):
            table = _table(records, first=('object_id', 'view', 'current_roi_uv',
                'current_visible_roi_uv', 'reliable', 'status', 'extent_visibility',
                'partial_current_support', 'projected_roi_uv', 'last_known_roi_uv'))
            encoded = {'view_table_v21': table,
                       'view_table_semantics': 'Each row is the exact named object/view snapshot. Entry views reference row index; missing_cells means absent key, not null. No inferred visibility.'}
            original_size = sum(len(_canonical(views[view])) for views, view in destinations)
            if len(_canonical(encoded)) + 50 * len(destinations) < original_size:
                inventory.update(encoded)
                for index, (views, view) in enumerate(destinations):
                    views[view] = {_ROW: index}
    field = result.get('field')
    guard = result.get('translation_guard')
    for parent, key in ((field, 'region_feasibility'), (guard, 'neighbors')):
        if not isinstance(parent, dict) or not isinstance(parent.get(key), list):
            continue
        original = parent[key]
        table = _table(original, first=('id', 'kind', 'geometry_mode'))
        if table is not None:
            replacement = {_TABLE: table}
            if len(_canonical(replacement)) + 100 < len(_canonical(original)):
                parent[key] = replacement
    result['presentation_v21'] = VERSION
    return result


def restore_feedback(value):
    """Exact inverse to the *already V17-presented* source feedback snapshot."""
    result = copy.deepcopy(value)
    if not isinstance(result, dict) or not result.get('presentation_v21'):
        return result
    inventory = result.get('scene_inventory')
    if isinstance(inventory, dict) and 'view_table_v21' in inventory:
        rows = _untable(inventory.pop('view_table_v21'))
        inventory.pop('view_table_semantics', None)
        for entry in inventory.get('entries', []):
            for view, status in (entry.get('views') or {}).items():
                if isinstance(status, dict) and set(status) == {_ROW}:
                    row = copy.deepcopy(rows[status[_ROW]])
                    if row.pop('object_id') != entry.get('id') or row.pop('view') != view:
                        raise ValueError('V21 feedback view reference identity mismatch')
                    entry['views'][view] = row
    for parent, key in ((result.get('field'), 'region_feasibility'),
                        (result.get('translation_guard'), 'neighbors')):
        if isinstance(parent, dict) and isinstance(parent.get(key), dict) and set(parent[key]) == {_TABLE}:
            parent[key] = _untable(parent[key][_TABLE])
    result.pop('presentation_v21')
    return result


def _records(payload):
    records = []
    for message in payload.get('input', []):
        if message.get('role') != 'user':
            continue
        for part in message.get('content', []):
            if part.get('type') != 'input_text':
                continue
            try:
                record = json.loads(part['text'])
            except (ValueError, TypeError, KeyError):
                continue
            if isinstance(record, dict) and record.get('type') in (
                    'observation', 'visual_execution_block', 'visual_history_summary'):
                records.append((part, record))
    return records


def expanded_observation_feedback(payload):
    """Read exact same-request V17 patches before modifying their base snapshot."""
    observations = []
    for part, record in _records(payload):
        if record.get('type') == 'observation':
            feedback = (record.get('observable_context') or {}).get('visual_feedback')
            if isinstance(feedback, dict):
                observations.append((part, record, feedback))
    by_index = {record.get('observation_index'): feedback for _, record, feedback in observations}
    result = []
    for part, record, feedback in observations:
        if 'feedback_patch' in feedback:
            ref = feedback.get('feedback_ref') or {}
            base = by_index.get(ref.get('observation_index'))
            if ref.get('path') != 'observable_context.visual_feedback' or not isinstance(base, dict) or 'feedback_patch' in base:
                raise ValueError('V21 feedback patch requires a full same-request base')
            feedback = restore_feedback_delta(feedback, base)
        result.append((part, record, feedback))
    return result


def _record(value):
    if isinstance(value, list):
        return [_record(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {key: compact_feedback(item) if key in ('visual_feedback', 'end_visual_feedback') else _record(item)
            for key, item in value.items()}


def compact_payload_feedback(payload):
    result = copy.deepcopy(payload)
    # Expand *all* old patches before touching the current base. This avoids
    # cross-request references or interpreting old deltas against a changed base.
    for part, record, feedback in expanded_observation_feedback(result):
        record['observable_context']['visual_feedback'] = feedback
        part['text'] = _canonical(record)
    for part, record in _records(result):
        part['text'] = _canonical(_record(record))
    return _deduplicate_older_feedback(result)
