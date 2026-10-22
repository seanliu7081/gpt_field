"""Explicit transit proxy contract; only fake subscription transport is used."""
import copy
import json
from pathlib import Path

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v15 import MOTION_LABEL as OLD_MOTION_LABEL
from gpt_field.visual_field_subscription_v18 import CANDIDATE_LABEL as OLD_CANDIDATE_LABEL
from gpt_field.visual_field_subscription_v31 import output_schema as schema_v31, shared_guidance as guidance_v31
from gpt_field.visual_field_subscription_v32 import (
    VisualFieldSubscriptionClientV32, output_schema, shared_guidance, geometry_semantics,
    CANDIDATE_LABEL, MOTION_LABEL, VERSION, HISTORY_VERSION, REFERENCE_ROLES,
    compact_guard_debug_payload, restore_guard_debug_snapshot, FEEDBACK_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v7_protocol import current_record, current_images, stream_error
from test_visual_field_v18_protocol import candidate_result, resolver_for
from test_visual_field_v31_protocol import prediction as old_prediction, proposed as old_proposed, attach


def prediction(*, role='tool_anatomy', **kwargs):
    result = old_prediction(**kwargs)
    ext = result['visual_references']['external']
    if ext is not None:
        ext['reference_role'] = role
        if role == 'rigid_transit_proxy' and ext['seed'] is not None:
            ext['seed']['tool_point_uv'] = [.8, .8]
            ext['seed']['tool_roi_uv'] = [.76, .76, .84, .84]
            result['motion_context']['tool_external_roi_uv'] = [.7, .7, .95, .95]
    return result


def client(*responses, **kwargs):
    transport = FakeTransport(*(r if isinstance(r, BaseException) else envelope(
        r, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v32_{i}')
        for i, r in enumerate(responses)))
    return VisualFieldSubscriptionClientV32(token_provider=lambda: 'offline-oauth',
        transport=transport, **kwargs), transport


def test_only_external_reference_role_changes_schema_and_prompt_is_shorter():
    schema = output_schema()
    ext = schema['properties']['visual_references']['properties']['external']['anyOf'][0]
    assert ext['properties'].pop('reference_role')['enum'] == list(REFERENCE_ROLES)
    ext['required'].remove('reference_role')
    assert schema == schema_v31()
    task = resolve_distance_task('bowl_plate')
    guidance = shared_guidance(task)
    assert len(guidance) < len(guidance_v31(task))
    assert 'rigid_transit_proxy' in guidance and 'independent actual-hand' in guidance
    assert 'NOT an articulated proximal link' in guidance
    assert 'not pinch position' in guidance and 'exactly FIVE' in guidance
    assert 'cabinet' not in guidance.lower()
    assert not geometry_semantics()['reference_role_contract']['proxy_support_is_hand_collision_envelope']


def test_proxy_candidate_outside_hand_keeps_clearance_and_evidence_exact():
    wire = prediction(role='rigid_transit_proxy', candidate=True, controller='visual_field')
    original = copy.deepcopy(wire)
    instance, transport = client(wire, prediction())
    obs = observation().to_dict(); calls = []
    instance.reference_candidate_resolver = resolver_for(obs, calls)
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire == original
    assert all(result['motion_context'][key] == value for key, value in wire['motion_context'].items())
    ext = result['visual_references']['external']
    assert ext['seed']['moving_point_uv'] == candidate_result()['current_point_uv']
    assert ext['seed']['moving_roi_uv'] == candidate_result()['current_roi_uv']
    assert ext['seed']['tool_point_uv'] == [.8, .8]
    assert ext['reference_role'] == 'rigid_transit_proxy'
    assert result['reference_candidate_resolution'] == candidate_result()
    assert calls == [(wire['reference_candidate_id'], obs)]
    assert instance._candidate_observation is None
    instance.predict(observation(1).to_dict())
    raw = [json.loads(c['text']) for m in transport.calls[-1]['input']
        if m.get('role') == 'assistant' for c in m['content'] if c['type'] == 'output_text']
    assert raw == [wire]
    assert raw[0]['visual_references']['external']['seed']['moving_point_uv'] is None
    assert instance.prompt_version == VERSION + '-hybrid'
    assert instance.history_stats()['protocol'] == HISTORY_VERSION


def test_anatomy_candidate_outside_whole_hand_remains_invalid():
    wire = prediction(candidate=True, controller='visual_field')
    wire['motion_context']['tool_external_roi_uv'] = [.7, .7, .95, .95]
    instance, _ = client()
    obs = observation().to_dict(); instance._candidate_observation = obs
    instance.reference_candidate_resolver = resolver_for(obs, [])
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_ordinary_role_preserves_existing_payload_transport_semantics():
    wire = prediction(controller='visual_field', phase='transport')
    ext = wire['visual_references']['external']
    ext.update(moving_kind='payload', moving_id='held-object')
    ext['seed'].update(tool_point_uv=[.8,.8], tool_roi_uv=[.75,.75,.85,.85])
    instance, _ = client()
    result = instance.validate_response(wire, None)
    assert result['visual_references']['external']['moving_kind'] == 'payload'
    assert result['visual_references']['external']['reference_role'] == 'tool_anatomy'
    assert result['raw_wire_prediction'] == wire


@pytest.mark.parametrize('case', ['missing_role', 'unknown_role', 'payload', 'grip', 'contact',
    'alignment', 'interaction', 'transport', 'release', 'moving_direct', 'rotating_direct',
    'closing_direct', 'missing_hand_point', 'missing_hand_roi', 'hand_outside_envelope',
    'missing_envelope', 'missing_proxy_roi', 'same_point'])
def test_proxy_cannot_be_used_as_anatomy_or_escape_current_geometric_contract(case):
    wire = prediction(role='rigid_transit_proxy', controller='visual_field')
    ext = wire['visual_references']['external']
    if case == 'missing_role': ext.pop('reference_role')
    elif case == 'unknown_role': ext['reference_role'] = 'anything_moving'
    elif case == 'payload': ext['moving_kind'] = 'payload'
    elif case == 'grip': wire['controller'] = 'gripper'
    elif case == 'contact': wire['target_contact'] = 'fingers'
    elif case == 'alignment': wire['contact_alignment'] = {
        'update': 'retain', 'target_id': wire['target_id'], 'external': None, 'wrist': None}
    elif case == 'interaction': wire['motion_context']['translation_mode'] = 'interaction'
    elif case in ('transport', 'release'): wire['phase'] = case
    elif case.endswith('_direct'):
        wire.update(controller='direct', field_intent=None, direct_actions=[[0.]*7 for _ in range(5)])
        index = {'moving_direct': 0, 'rotating_direct': 3, 'closing_direct': 6}[case]
        wire['direct_actions'][0][index] = .001
    elif case == 'missing_hand_point': ext['seed']['tool_point_uv'] = None
    elif case == 'missing_hand_roi': ext['seed']['tool_roi_uv'] = None
    elif case == 'hand_outside_envelope': wire['motion_context']['tool_external_roi_uv'] = [.01,.01,.1,.1]
    elif case == 'missing_envelope': wire['motion_context']['tool_external_roi_uv'] = None
    elif case == 'missing_proxy_roi': ext['seed']['moving_roi_uv'] = None
    elif case == 'same_point':
        ext['seed']['moving_point_uv'] = [.8,.8]; ext['seed']['moving_roi_uv'] = [.76,.76,.84,.84]
    instance, _ = client()
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_proxy_exact_zero_direct_hold_and_retain_preserve_roles():
    wire = prediction(role='rigid_transit_proxy', controller='direct', phase='recover')
    wire['direct_actions'] = [[0.]*7 for _ in range(5)]
    wire['visual_references']['external'].update(update='retain', seed=None)
    instance, _ = client()
    result = instance.validate_response(wire, None)
    assert result['direct_actions'] == wire['direct_actions']
    assert result['visual_references']['external']['reference_role'] == 'rigid_transit_proxy'
    assert result['raw_wire_prediction'] == wire


def test_bound_field_proposal_preserves_role_and_resolution_without_callback_leak():
    wire, resolved, context = old_proposed()
    for value in (wire, resolved):
        value['visual_references']['external']['reference_role'] = 'rigid_transit_proxy'
    instance, _ = client(wire)
    obs = observation().to_dict(); attach(instance, resolved, context, obs)
    original_resolver = instance.field_proposal_resolver
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire
    assert result['field_proposal_resolution'] == resolved
    assert result['visual_references']['external']['reference_role'] == 'rigid_transit_proxy'
    assert instance.field_proposal_resolver is original_resolver
    resolved['visual_references']['external']['reference_role'] = 'tool_anatomy'
    instance, _ = client(); attach(instance, resolved, context, obs)
    instance._candidate_observation = obs
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_actual_board_and_motion_labels_are_replaced_preserving_images_and_raw_assistant():
    wire = prediction()
    instance, _ = client(wire)
    obs = observation().to_dict(); instance.predict(obs)
    instance.observable_context = {'visual_feedback': {'reference_repair': {'review_required': True}}}
    instance.reference_candidate_provider = lambda actual: actual['observation/image']
    payload = instance.build_payload(observation(1).to_dict())
    labels = [c['text'] for m in payload['input'] if m.get('role') == 'user'
        for c in m.get('content', []) if c['type'] == 'input_text']
    assert CANDIDATE_LABEL in labels and OLD_CANDIDATE_LABEL not in labels
    assert len(current_images(payload)) == 3
    assert current_images(payload)[:2] == [observation(1).to_dict()[k]
        for k in ('observation/image', 'observation/wrist_image')]
    instance.observable_context = {'visual_feedback': {'reference_support': {'diagnostic_required': True}}}
    instance.motion_diagnostic_provider = lambda actual: actual['observation/image']
    payload = instance.build_payload(observation(1).to_dict())
    labels = [c['text'] for m in payload['input'] if m.get('role') == 'user'
        for c in m.get('content', []) if c['type'] == 'input_text']
    assert MOTION_LABEL in labels and OLD_MOTION_LABEL not in labels
    assert payload['text']['format']['name'] == 'visual_field_rgb_joints_v32'


@pytest.mark.parametrize('order', ['stream_first', 'schema_first'])
def test_proxy_repair_and_stream_retry_keep_same_observation_and_medium_budget(order):
    wire = prediction(role='rigid_transit_proxy', candidate=True, controller='visual_field')
    invalid = copy.deepcopy(wire); invalid['visual_references']['external']['seed']['tool_point_uv'] = None
    failures = [stream_error(), invalid] if order == 'stream_first' else [invalid, stream_error()]
    instance, transport = client(*failures, wire)
    obs = observation().to_dict(); instance.reference_candidate_resolver = resolver_for(obs, [])
    assert instance.predict(obs)['raw_wire_prediction'] == wire
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    assert all(p['model']=='gpt-6-astra' and p['reasoning']=={'effort':'medium'} for p in transport.calls)
    assert len({current_record(p)['observation_index'] for p in transport.calls}) == 1
    assert all(current_images(p)==current_images(transport.calls[0]) for p in transport.calls)
    assert json.loads(current_record(transport.calls[-1])['observable_context']['output_repair']['rejected_model_output_json']) == invalid


def test_quota_error_is_not_retried():
    instance, transport = client(stream_error(code='insufficient_quota',status=429),prediction())
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.budget_snapshot()['api_key_calls'] == 0


def _expanded_snapshot(snapshot, guard):
    if isinstance(snapshot, dict) and set(snapshot) == {'feedback_ref', 'overrides'}:
        reference = snapshot['feedback_ref']
        if reference.get('path') == 'translation_guard':
            result = {key: copy.deepcopy(guard[key]) for key in reference['keys']}
            result.update(copy.deepcopy(snapshot['overrides']))
            return result
    return snapshot


def test_debug_delta_is_exact_for_changes_removals_unknown_fields_without_touching_current():
    guard = {'configured': True, 'clearance_certified': False, 'translation_mode': 'separate',
        'physical_remaining_m': .005, 'large_known_list': [{'value': i} for i in range(300)],
        'remove_this': {'unknown': 1}, 'newly_added_runtime_field': {'important': 'current'}}
    snapshot = copy.deepcopy(guard)
    snapshot['physical_remaining_m'] = .004
    snapshot.pop('remove_this')
    snapshot['newly_added_runtime_field']['important'] = 'prospective warning'
    snapshot['future_unknown_evidence'] = {'must_survive': [1,2,3]}
    context = {'visual_feedback': {'translation_guard': guard,
        'scene_inventory': {'entries': [{'id': 'visible', 'current_geometry': {'center_offset_m': [.1,.2,.3]}}]},
        'field': {'translation_admission': {'accepted': False, 'reason': 'actual_block', 'cap_m': .003,
            'diagnostics': snapshot}},
        'recovery_options': {'candidates': [{'exact_wire': {'preserve': 4}}]},
        'visual': {'attachment': {'state': 'not_attached'}, 'current_rois': [[.1,.1,.2,.2]]}}}
    instance, _ = client(); instance.observable_context = copy.deepcopy(context)
    original = copy.deepcopy(context)
    payload = compact_guard_debug_payload({'input': [{'role': 'user', 'content': [
        {'type': 'input_text', 'text': json.dumps({'type': 'observation', 'observable_context': context})}]}]})
    feedback = current_record(payload)['observable_context']['visual_feedback']
    debug = feedback['field']['translation_admission']['diagnostics']
    assert debug['presentation'] == FEEDBACK_VERSION
    assert restore_guard_debug_snapshot(debug, feedback['translation_guard']) == snapshot
    assert feedback['translation_guard'] == guard
    assert feedback['field']['translation_admission']['reason'] == 'actual_block'
    assert feedback['recovery_options'] == context['visual_feedback']['recovery_options']
    assert instance.observable_context == original


def test_all_nine_archived_packets_decode_exactly_and_keep_raw_images_and_outputs():
    directory = Path(__file__).parents[1] / 'runs/visual_field_v31_zero_shot_001/trials/bowl_original_zero_hybrid_v31/api'
    paths = sorted(directory.glob('*.request.json'))
    if len(paths) != 9:
        pytest.skip('The nine audited V31 archived packets are not available')
    totals = [0, 0]; encoded = 0
    for path in paths:
        payload = json.loads(path.read_text())['payload']
        frozen = copy.deepcopy(payload)
        compact = compact_guard_debug_payload(payload)
        totals[0] += len(json.dumps(payload)); totals[1] += len(json.dumps(compact))
        for old_message, new_message in zip(payload['input'], compact['input']):
            if old_message.get('role') != 'user':
                assert old_message == new_message
                continue
            for old, new in zip(old_message['content'], new_message['content']):
                if old['type'] != 'input_text':
                    assert old == new
                    continue
                try:
                    before, after = json.loads(old['text']), json.loads(new['text'])
                except ValueError:
                    assert old == new
                    continue
                if not isinstance(before, dict) or before.get('type') != 'observation':
                    assert before == after
                    continue
                guard = before['observable_context']['visual_feedback'].get('translation_guard')
                if not isinstance(guard, dict):
                    assert before == after
                    continue
                assert after['observable_context']['visual_feedback']['translation_guard'] == guard
                def equivalent(a, b):
                    nonlocal encoded
                    if isinstance(b, dict) and b.get('presentation') == FEEDBACK_VERSION:
                        assert restore_guard_debug_snapshot(b, guard) == _expanded_snapshot(a, guard)
                        encoded += 1
                    elif isinstance(a, dict):
                        assert a.keys() == b.keys()
                        for key in a: equivalent(a[key], b[key])
                    elif isinstance(a, list):
                        assert len(a) == len(b)
                        for x,y in zip(a,b): equivalent(x,y)
                    else:
                        assert a == b
                equivalent(before, after)
        assert payload == frozen
    assert encoded >= 8
    assert totals[1] < totals[0]


def test_payload_debug_compression_preserves_stored_history_and_assistant_wire():
    wire = prediction()
    instance, transport = client(wire, wire)
    guard = {'configured': True, 'clearance_certified': False, 'translation_mode': 'transit',
        'debug_samples': [{'error': x} for x in range(300)]}
    context = {'visual_feedback': {'translation_guard': guard,
        'field': {'translation_admission': {'accepted': False, 'reason': 'blocked',
            'diagnostics': copy.deepcopy(guard)}}}}
    instance.observable_context = copy.deepcopy(context)
    instance.predict(observation().to_dict())
    saved = copy.deepcopy(instance.history_snapshot())
    history_record = next(_text for m in saved for c in m.get('content',[])
        if c.get('type')=='input_text' and (_text:=json.loads(c['text'])).get('type')=='observation')
    debug = history_record['observable_context']['visual_feedback']['field']['translation_admission']['diagnostics']
    assert debug.get('presentation') != FEEDBACK_VERSION
    instance.predict(observation(1).to_dict())
    assert instance.observable_context == context
    raw = [json.loads(c['text']) for m in transport.calls[-1]['input'] if m.get('role')=='assistant'
        for c in m['content'] if c['type']=='output_text']
    assert raw == [wire]
