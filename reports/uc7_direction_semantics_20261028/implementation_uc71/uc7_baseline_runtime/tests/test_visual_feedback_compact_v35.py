import copy
import json
from pathlib import Path

import pytest

from gpt_field.astra import _canonical
from gpt_field.visual_feedback_compact_v35 import (
    VERSION, compact_current_feedback, expand_current_feedback,
)


def _feedback():
    evidence = {
        'current_state8': [.12345678901234567, -.042, 1.2, 3.14, 0., -0., .04, -.04],
        'reference_id': 'current-external-epoch-3', 'observation_binding': 'current-37',
        'uncertainty_m': .0321592653589793, 'remaining_measured_m': .00123456789012345,
        'reason': 'inconclusive_boundary_not_clearance', 'clearance_certified': False,
        'adverse_evidence': {'worsening_ids': ['same-obstacle'], 'reference_lost': True},
        'future_extension': {'unknowns_remain_hard': True, 'null_differs_from_absence': None},
    }
    regions = [{'id': f'object-{i}', 'lower_xyz': [i*.001, -.02, 1.],
                'upper_xyz': [i*.001+.02, .03, 1.1], 'unknown': i == 4,
                'uncertainty_m': .04, 'nominal_clearance_is_physical_clearance': False,
                'unrecognized_extension': {'keep_this_value': i}}
               for i in range(6)]
    candidates = []
    for i in range(5):
        current = copy.deepcopy(evidence)
        current['remaining_measured_m'] += i*.00000000001
        current['candidate_id'] = f'candidate-{i}'
        current['first_command_m'] = [i*.0001, -.002, 0.]
        candidates.append({'requires_explicit_model_choice': True,
            'accepted': i == 0, 'reason': 'accepted' if i == 0 else 'unknown_region',
            'contract': current, 'geometry': copy.deepcopy(regions)})
    return {'scene_inventory': {'entries': regions}, 'translation_guard': evidence,
            'controlled_reference_witness': copy.deepcopy(evidence),
            'recovery_options': {'candidates': candidates},
            'field': {'diagnostics': copy.deepcopy(evidence), 'regions': copy.deepcopy(regions)},
            'task_success_certified': False, 'scene_execution_limits': None}


def test_round_trip_preserves_exact_geometry_risk_limits_bindings_and_candidate_contracts():
    source = _feedback(); before = _canonical(source)
    packed = compact_current_feedback(source)
    assert packed['presentation_v35'] == VERSION
    assert len(_canonical(packed)) < len(before)
    assert _canonical(expand_current_feedback(packed)) == before
    assert _canonical(source) == before
    assert compact_current_feedback(packed) == packed
    # Every first-command value, rejection, future field and tiny numeric
    # difference is compared; no float tolerance or semantic equality shortcut.
    for candidate, restored in zip(source['recovery_options']['candidates'],
            expand_current_feedback(packed)['recovery_options']['candidates']):
        assert _canonical(candidate) == _canonical(restored)


def test_compacted_and_expanded_results_do_not_alias_the_input_or_each_other():
    source = _feedback(); original = copy.deepcopy(source)
    packet = compact_current_feedback(source); encoded_before = _canonical(packet)
    restored = expand_current_feedback(packet)
    restored['field']['diagnostics']['current_state8'][0] = 999
    assert restored['translation_guard']['current_state8'][0] != 999
    assert source == original
    assert _canonical(packet) == encoded_before
    packet['shared_facts'].clear()
    assert source == original


def test_table_distinguishes_absent_keys_and_null_and_preserves_original_row_order():
    rows = []
    for i in range(8):
        row = {f'long_numeric_coordinate_{k}': i + k*.037 for k in range(12)}
        if i != 3:
            row['optional_null_not_absent'] = None
        rows.append(row)
    source = {'rows': rows}
    packed = compact_current_feedback(source)
    assert '$table_v35' in _canonical(packed)
    restored = expand_current_feedback(packed)
    assert _canonical(restored) == _canonical(source)
    assert 'optional_null_not_absent' not in restored['rows'][3]
    assert restored['rows'][4]['optional_null_not_absent'] is None


def test_unknown_marker_literals_are_not_interpreted_or_changed():
    source = _feedback()
    source['future_unknown_field'] = {'$same_v35': 'literal-not-a-reference'}
    packed = compact_current_feedback(source)
    assert packed == source and packed is not source
    assert expand_current_feedback(packed) == source


def test_reserved_packet_marker_source_is_not_rewritten_and_malformed_claim_is_rejected():
    source = {'presentation_v35': VERSION, 'unknown_future_fact': {'must_remain': 3}}
    assert compact_current_feedback(source) == source
    with pytest.raises(ValueError, match='packet keys'):
        expand_current_feedback(source)


def test_sparse_recursive_record_shapes_do_not_create_shared_definition_cycles():
    nodes = []
    for i in range(7):
        node = {'label': 'same', 'reason': 'a long explanation that must stay literal '*5,
                'unknown': False, 'clearance_certified': False, 'next': None}
        for j in range(i):
            node = {**node, 'next': copy.deepcopy(node)}
        nodes.append(node)
    source = {'recursive_shape': nodes, 'other': copy.deepcopy(nodes)}
    assert _canonical(expand_current_feedback(compact_current_feedback(source))) == _canonical(source)


def test_unknown_extension_names_cannot_collide_with_common_fact_namespace():
    repeated = 'unknown region remains hard; '*40
    common = {f'unchanged_field_{i}': 'exact common value '*30 for i in range(5)}
    source = {'common_foo': repeated, 'duplicate': copy.deepcopy(repeated),
              'foo': {**common, 'different_id': 'A'},
              'bar': {**common, 'different_id': 'B'}}
    packed = compact_current_feedback(source)
    assert any(name.startswith('common_foo_') for name in packed['shared_facts'])
    assert any(name.startswith('same_common_foo_') for name in packed['shared_facts'])
    assert _canonical(expand_current_feedback(packed)) == _canonical(source)


@pytest.mark.parametrize('source', [None, [], 4, 'text', {'one': 1}])
def test_small_or_non_dictionary_values_remain_detached_and_unchanged(source):
    assert compact_current_feedback(source) == source
    assert expand_current_feedback(compact_current_feedback(source)) == source


def _packet(current, shared=None):
    return {'presentation_v35': VERSION, 'semantics': 'test',
            'shared_facts': {} if shared is None else shared, 'current': current}


@pytest.mark.parametrize('packet,match', [
    (_packet({'$same_v35': 'missing'}), 'same-observation'),
    (_packet({'$same_v35': 'a'}, {'a': {'$same_v35': 'b'}, 'b': {'$same_v35': 'a'}}), 'Cyclic'),
    (_packet({'$with_v35': 'a', 'different_values': {}}, {'a': [1, 2]}), 'common-fields'),
    (_packet({'$table_v35': {'columns': ['x', 'x'], 'rows': [[1, 2]]}}), 'shape'),
    (_packet({'$table_v35': {'columns': ['x'], 'rows': [[1, 2]]}}), 'shape'),
    (_packet({'$table_v35': {'columns': ['x'], 'rows': [[1]], 'absent_cells': [[2, 0]]}}), 'absent cells'),
])
def test_decoder_rejects_missing_cross_observation_cyclic_and_malformed_data(packet, match):
    before = copy.deepcopy(packet)
    with pytest.raises(ValueError, match=match):
        expand_current_feedback(packet)
    assert packet == before


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = [path for suffix in ('001', '002') for path in sorted(
    (ROOT / f'runs/visual_field_v34_zero_shot_{suffix}/trials/bowl_original_zero_hybrid_v34/api').glob('*.request.json'))]


def _current(payload):
    observations = []
    for message in payload['input']:
        if message.get('role') != 'user':
            continue
        for part in message.get('content', []):
            if part.get('type') != 'input_text':
                continue
            try:
                record = json.loads(part['text'])
            except (ValueError, TypeError, KeyError):
                continue
            if isinstance(record, dict) and record.get('type') == 'observation':
                observations.append(record)
    return observations[-1]['observable_context']['visual_feedback']


@pytest.mark.parametrize('path', ARCHIVES, ids=lambda p: p.parents[3].name+'-'+p.name[:6])
def test_all_real_v34_requests_restore_all_current_values_without_mutating_raw_input(path):
    raw = path.read_bytes(); payload = json.loads(raw)['payload']
    before_payload = _canonical(payload)
    source = _current(payload); before = _canonical(source)
    packed = compact_current_feedback(source)
    assert _canonical(expand_current_feedback(packed)) == before
    assert len(_canonical(packed)) <= len(before)
    assert _canonical(payload) == before_payload
    assert compact_current_feedback(packed) == packed
    assert path.read_bytes() == raw
