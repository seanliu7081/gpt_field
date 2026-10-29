"""Explicit scene revisions and exact paid-history proposal selection, offline."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v30 import output_schema as schema_v30
from gpt_field.visual_field_subscription_v31 import (
    VisualFieldSubscriptionClientV31, output_schema, shared_guidance,
    geometry_semantics, SCENE_REVISION_EVIDENCE_KINDS, VERSION, HISTORY_VERSION,
    FEEDBACK_VERSION, compact_scene_revision_payload,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v25_protocol import prediction as prediction_v25
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def prediction(**kwargs):
    value = prediction_v25(**kwargs)
    value.update(scene_revisions=[], field_proposal_id=None)
    return value


def client(*responses, **kwargs):
    transport = FakeTransport(*(r if isinstance(r, BaseException) else envelope(
        r, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v31_{i}')
        for i, r in enumerate(responses)))
    return VisualFieldSubscriptionClientV31(token_provider=lambda: 'offline-oauth',
                                            transport=transport, **kwargs), transport


def proposed():
    wire = prediction(controller='visual_field')
    field = copy.deepcopy(wire['field_intent'])
    remembered = copy.deepcopy(wire['scene_inventory'][0]['geometry'])
    wire['scene_inventory'][0]['geometry'] = None
    wire['field_proposal_id'] = 'field-current-1'
    wire['field_intent'] = None
    wire['visual_references']['external'].update(update='retain', seed=None)
    resolved = {'proposal_id': wire['field_proposal_id'], 'controller': 'visual_field',
                'phase': wire['phase'], 'target_id': wire['target_id'],
                'recovery_route': wire['recovery_route'], 'field_intent': field,
                'motion_context_overrides': copy.deepcopy(wire['motion_context']),
                'subgoal_update': copy.deepcopy(wire['subgoal_update']),
                'scene_declarations': copy.deepcopy(wire['scene_inventory']),
                'visual_references': copy.deepcopy(wire['visual_references']),
                'binding': {'observation_sha256': 'synthetic-current-observation'}}
    context = {'visual_feedback': {'scene_inventory': {'entries': [
        {'id': wire['target_id'], 'current_geometry': remembered,
         'estimated_at_observation': 0, 'metric_declared_at_update': 1,
         'metric_changed_at_observation': 0, 'metric_age_observations': 5,
         'metric_age_updates': 3, 'last_seen_observation': 5,
         'scene_review': {'reason': 'measured_motion_conflict', 'review_required': True}}]}}}
    return wire, resolved, context


def attach(instance, resolved, context, expected, calls=None):
    instance.observable_context = copy.deepcopy(context)
    def resolve(identity, actual):
        if calls is not None:
            calls.append((identity, copy.deepcopy(actual)))
        if actual != expected or identity != resolved['proposal_id']:
            raise ValueError('private callback details must not reach repair prompt')
        return copy.deepcopy(resolved)
    instance.field_proposal_resolver = resolve


def test_only_two_new_required_fields_and_positive_generic_revision_guidance():
    schema = output_schema()
    assert schema['properties']['scene_revisions']['items']['properties']['evidence_kind']['enum'] == list(SCENE_REVISION_EVIDENCE_KINDS)
    for key in ('scene_revisions', 'field_proposal_id'):
        schema['properties'].pop(key)
        schema['required'].remove(key)
    assert schema == schema_v30()
    text = shared_guidance(resolve_distance_task('bowl_plate'))
    for phrase in ('explicitly CORRECT', 'measured_motion_conflict', 'age of a retained metric estimate',
                   'Nominal center/extent and uncertainty margin are different claims',
                   'field_intent=null', 'scene_declarations', 'contact_alignment=null',
                   'no extra confirmation call', 'exactly FIVE', 'every NON-TARGET check remains unchanged'):
        assert phrase in text
    assert 'cabinet' not in text.lower()
    assert not geometry_semantics()['scene_estimate_revision']['null_geometry_or_new_roi_refreshes_metric_estimate']
    assert not geometry_semantics()['field_proposal_selection']['permission_or_physical_allowance_reset']
    with pytest.raises(ValueError):
        shared_guidance(resolve_distance_task('bowl_plate'), True)


@pytest.mark.parametrize('kind', SCENE_REVISION_EVIDENCE_KINDS)
def test_revision_claim_is_preserved_without_altering_other_geometry(kind):
    wire = prediction(controller='visual_field')
    wire['scene_inventory'][0]['wrist_roi_uv'] = [.3, .3, .7, .7]
    wire['scene_revisions'] = [{'id': wire['target_id'], 'reason': 'Current view disagrees with the retained estimate.',
                                'evidence_kind': kind}]
    before = copy.deepcopy(wire)
    instance, _ = client()
    result = instance.validate_response(wire, None)
    assert wire == before == result['raw_wire_prediction']
    assert result['scene_revisions'] == wire['scene_revisions']
    assert result['scene_inventory'] == wire['scene_inventory']
    assert result['field_proposal_resolution'] is None


@pytest.mark.parametrize('case', ['duplicate', 'unknown', 'null_geometry', 'blank_reason', 'blank_id',
                                  'unsupported_evidence', 'two_view_missing', 'missing_revision_list',
                                  'missing_proposal_id'])
def test_malformed_revision_claim_never_reaches_inventory(case):
    wire = prediction(controller='visual_field')
    wire['scene_revisions'] = [{'id': wire['target_id'], 'reason': 'Visible discrepancy.',
                                'evidence_kind': 'measured_motion_conflict'}]
    if case == 'duplicate': wire['scene_revisions'] *= 2
    elif case == 'unknown': wire['scene_revisions'][0]['id'] = 'unlisted'
    elif case == 'null_geometry': wire['scene_inventory'][0]['geometry'] = None
    elif case == 'blank_reason': wire['scene_revisions'][0]['reason'] = ' '
    elif case == 'blank_id': wire['scene_revisions'][0]['id'] = ' '
    elif case == 'unsupported_evidence': wire['scene_revisions'][0]['evidence_kind'] = 'simulator_truth'
    elif case == 'two_view_missing': wire['scene_revisions'][0]['evidence_kind'] = 'current_two_view'
    elif case == 'missing_revision_list': wire.pop('scene_revisions')
    else: wire.pop('field_proposal_id')
    instance, _ = client()
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_exact_current_proposal_resolves_internal_intent_and_preserves_raw_history():
    wire, resolved, context = proposed()
    obs = observation().to_dict()
    original = copy.deepcopy((wire, resolved, context, obs))
    instance, transport = client(wire, prediction())
    calls = []
    attach(instance, resolved, context, obs, calls)
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire and wire['field_intent'] is None
    assert result['field_proposal_resolution'] == resolved
    assert result['field_intent']['goal_offset_m'] == resolved['field_intent']['goal_offset_m']
    assert result['field_intent']['uncertainty_m'] == resolved['field_intent']['uncertainty_m']
    assert result['field_intent']['orientation_target_xyzw'] is not None
    assert (wire, resolved, context, obs) == original
    assert calls == [(wire['field_proposal_id'], obs)]
    assert instance._candidate_observation is None
    instance.predict(observation(1).to_dict())
    payload = transport.calls[-1]
    assistants = [json.loads(p['text']) for m in payload['input'] if m.get('role') == 'assistant'
                  for p in m.get('content', []) if p.get('type') == 'output_text']
    assert assistants == [wire]
    assert assistants[0]['field_intent'] is None
    assert payload['text']['format']['name'] == 'visual_field_rgb_joints_v31'
    assert instance.prompt_version == VERSION + '-hybrid'
    assert instance.history_stats()['protocol'] == HISTORY_VERSION


@pytest.mark.parametrize('case', ['stale', 'wrong_resolved_id', 'missing_binding', 'wrong_phase', 'wrong_target',
    'wrong_route', 'context_margin_change', 'unknown_context_key', 'changed_scene_roi', 'scene_estimate',
    'reference_candidate', 'external_seed', 'wrist_seed', 'contact_descriptor', 'contact_permission',
    'changed_reference_objective', 'subgoal_retain', 'nonnull_wire_intent', 'direct', 'missing_resolver',
    'missing_native_intent', 'malformed_native_intent', 'legacy_obstacles'])
def test_selected_proposal_cannot_override_wire_or_current_binding(case):
    wire, resolved, context = proposed()
    obs = observation().to_dict()
    if case == 'stale': wire['field_proposal_id'] = 'old-proposal'
    elif case == 'wrong_resolved_id': resolved['proposal_id'] = 'other'
    elif case == 'missing_binding': resolved.pop('binding')
    elif case == 'wrong_phase': resolved['phase'] = 'align'
    elif case == 'wrong_target': resolved['target_id'] = 'other'
    elif case == 'wrong_route': resolved['recovery_route'] = 'reverse_observed'
    elif case == 'context_margin_change': wire['motion_context']['tool_external_roi_uv'][0] += .01
    elif case == 'unknown_context_key': resolved['motion_context_overrides']['secret_permission'] = True
    elif case == 'changed_scene_roi': wire['scene_inventory'][0]['external_roi_uv'][0] += .01
    elif case == 'scene_estimate': wire['scene_inventory'][0]['geometry'] = context['visual_feedback']['scene_inventory']['entries'][0]['current_geometry']
    elif case == 'reference_candidate': wire['reference_candidate_id'] = 'current-candidate'
    elif case == 'external_seed': wire['visual_references']['external']['update'] = 'seed'
    elif case == 'wrist_seed': wire['visual_references']['wrist'] = {
        'update': 'seed', 'target_id': wire['target_id'], 'desired_target_uv': [.5, .5],
        'seed': {'target_anchor_uv': [.5, .5], 'target_roi_uv': [.4, .4, .6, .6]}}
    elif case == 'contact_descriptor': wire['contact_alignment'] = {
        'update': 'retain', 'target_id': wire['target_id'], 'external': None, 'wrist': None}
    elif case == 'contact_permission': wire['target_contact'] = 'fingers'
    elif case == 'changed_reference_objective': wire['visual_references']['external']['goal_offset_uv'] = [.1, .1]
    elif case == 'subgoal_retain': wire['subgoal_update']['action'] = 'retain'
    elif case == 'nonnull_wire_intent': wire['field_intent'] = copy.deepcopy(resolved['field_intent'])
    elif case == 'direct': wire['controller'] = 'direct'
    elif case == 'missing_native_intent': resolved.pop('field_intent')
    elif case == 'malformed_native_intent': resolved['field_intent']['goal_offset_m'] = [.2, 0, 0]
    elif case == 'legacy_obstacles': resolved['field_intent']['obstacles'] = []
    instance, _ = client()
    attach(instance, resolved, context, obs)
    instance._candidate_observation = copy.deepcopy(obs)
    if case == 'missing_resolver': instance.field_proposal_resolver = None
    if case == 'wrong_resolved_id': instance.field_proposal_resolver = lambda *_: resolved
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_ordinary_field_still_requires_intent_and_does_not_call_proposal_resolver():
    instance, _ = client()
    instance.field_proposal_resolver = lambda *_: pytest.fail('ordinary Field must not resolve a proposal')
    wire = prediction(controller='visual_field')
    assert instance.validate_response(wire, None)['field_proposal_resolution'] is None
    wire['field_intent'] = None
    with pytest.raises(SchemaError): instance.validate_response(wire, None)


def test_current_freshness_review_and_offer_contract_survive_presentation_exactly():
    wire, resolved, context = proposed()
    context['visual_feedback'].update(
        scene_review={'review_required': True, 'nominal_estimate_observation': 0,
                      'current_observation': 278, 'reason': 'retained_geometry_image_progress_conflict'},
        field_proposals={'offers': [resolved], 'clearance_certified': False})
    instance, _ = client()
    instance.observable_context = copy.deepcopy(context)
    payload = instance.build_payload(observation().to_dict())
    actual = current_record(payload)['observable_context']['visual_feedback']
    assert actual['scene_inventory']['entries'] == context['visual_feedback']['scene_inventory']['entries']
    assert actual['scene_review'] == context['visual_feedback']['scene_review']
    assert actual['field_proposals'] == context['visual_feedback']['field_proposals']
    assert instance.observable_context == context
    assert len(current_images(payload)) == 2


@pytest.mark.parametrize('order', ['stream_first', 'schema_first'])
def test_new_contract_repair_is_bounded_same_observation_medium_subscription(order):
    wire, resolved, context = proposed()
    invalid = copy.deepcopy(wire)
    invalid['motion_context']['tool_external_roi_uv'][0] += .01
    failures = [stream_error(), invalid] if order == 'stream_first' else [invalid, stream_error()]
    instance, transport = client(*failures, wire)
    obs = observation().to_dict()
    attach(instance, resolved, context, obs)
    assert instance.predict(obs)['raw_wire_prediction'] == wire
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    assert all(p['model'] == 'gpt-6-astra' and p['reasoning'] == {'effort': 'medium'} for p in transport.calls)
    assert len({current_record(p)['observation_index'] for p in transport.calls}) == 1
    assert all(current_images(p) == current_images(transport.calls[0]) for p in transport.calls)
    repair = current_record(transport.calls[-1])['observable_context']['output_repair']
    assert json.loads(repair['rejected_model_output_json']) == invalid
    assert instance._candidate_observation is None


def test_resolver_errors_are_sanitized_and_quota_never_retries():
    wire, resolved, context = proposed()
    instance, transport = client(wire, wire)
    instance.observable_context = context
    def stale(*_): raise ValueError('private callback details')
    instance.field_proposal_resolver = stale
    with pytest.raises(SchemaError, match='unknown, stale or unavailable'):
        instance.predict(observation().to_dict())
    repair = current_record(transport.calls[-1])['observable_context']['output_repair']
    assert 'private callback details' not in json.dumps(repair)
    assert instance.history_snapshot() == []
    instance, transport = client(stream_error(code='insufficient_quota', status=429), prediction())
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.budget_snapshot()['api_key_calls'] == 0


def test_pure_inventory_revision_callback_uses_bounded_semantic_repair_before_history_commit():
    invalid = prediction(controller='visual_field')
    corrected = copy.deepcopy(invalid)
    corrected['scene_revisions'] = [{'id': corrected['target_id'],
        'reason': 'The new measured view contradicts the retained nominal offset.',
        'evidence_kind': 'measured_motion_conflict'}]
    obs = observation().to_dict()
    calls = []
    instance, transport = client(invalid, corrected)
    def validate(actual, entries, revisions):
        calls.append((copy.deepcopy(actual), copy.deepcopy(entries), copy.deepcopy(revisions)))
        assert actual == obs
        # This fake pure validator represents the inventory's authoritative
        # changed-estimate detection; it never executes or accepts a revision.
        if not revisions:
            assert instance.history_snapshot() == []
            raise ValueError('private model geometry contents must not leak')
        actual['observation/state'][0] += .1
        entries[0]['geometry']['center_offset_m'][0] += .1
        revisions[0]['reason'] = 'mutated callback copy'
    instance.scene_revision_validator = validate
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == corrected
    assert len(calls) == len(transport.calls) == 2
    assert calls[0][0] == calls[1][0] == obs
    assert calls[0][2] == [] and calls[1][2] == corrected['scene_revisions']
    repair = current_record(transport.calls[-1])['observable_context']['output_repair']
    assert 'private model geometry contents' not in json.dumps(repair)
    assert json.loads(repair['rejected_model_output_json']) == invalid
    assert instance._candidate_observation is None
    assert instance.budget_snapshot()['api_key_calls'] == 0


def test_many_revision_bindings_compact_only_in_request_current_facts_and_history_remain():
    _, _, context = proposed()
    inventory = context['visual_feedback']['scene_inventory']
    current = inventory['entries'][0]
    records = []
    for index in range(40):
        records.append({'id': current['id'], 'kind': 'explicit_metric_revision',
            'previous_world_geometry': {'center_xyz_m': [.1 + index*.001, .2, .3],
                'half_extents_m': [.03]*3, 'uncertainty_m': .02},
            'new_world_geometry': {'center_xyz_m': [.101 + index*.001, .2, .3],
                'half_extents_m': [.03]*3, 'uncertainty_m': .02},
            'revision': {'id': current['id'], 'reason': 'Corrected estimate from acknowledged image change.',
                'evidence_kind': 'measured_motion_conflict'},
            'observation_binding': {'state8': [.1]*8, 'observation_index': index,
                'update_index': index+1, 'rgb_sha256': {'external': 'a'*64, 'wrist': 'b'*64}},
            'current_declared_regions': {'external_roi_uv': [.1,.2,.3,.4], 'wrist_roi_uv': [.2,.3,.4,.5]},
            'evidence_status': 'model_reported_hypothesis_not_independently_verified',
            'geometry_measured_or_certified': False, 'motion_allowances_or_evidence_changed': False})
    inventory.update(revision_log=records, revision_log_complete=True)
    current['last_explicit_revision'] = copy.deepcopy(records[-1])
    context['visual_feedback']['scene_review'] = {'required': True, 'reason': 'current_geometry_conflict'}
    original = copy.deepcopy(context)
    raw = prediction()
    instance, transport = client(raw, raw)
    instance.observable_context = copy.deepcopy(context)
    result = instance.predict(observation().to_dict())
    feedback = current_record(transport.calls[0])['observable_context']['visual_feedback']
    shown = feedback['scene_inventory']
    assert 'revision_log' not in shown and 'revision_log_complete' not in shown
    assert shown['revision_log_summary']['record_count'] == 40
    assert shown['revision_log_summary']['explicit_revision_count'] == 40
    assert shown['revision_presentation'] == FEEDBACK_VERSION
    assert 'rgb_sha256' not in json.dumps(shown) and 'state8' not in json.dumps(shown)
    latest = shown['entries'][0]['last_explicit_revision']
    assert latest['revision'] == records[-1]['revision']
    assert latest['observation_index'] == 39 and latest['update_index'] == 40
    assert latest['previous_center_xyz_m'] == records[-1]['previous_world_geometry']['center_xyz_m']
    assert latest['changed_estimate_components'] == ['center_xyz_m']
    for key, value in current.items():
        if key != 'last_explicit_revision': assert shown['entries'][0][key] == value
    assert feedback['scene_review'] == context['visual_feedback']['scene_review']
    assert len(json.dumps(shown)) < .15 * len(json.dumps(inventory))
    assert result['raw_wire_prediction'] == raw
    assert instance.observable_context == context == original
    # The full feedback is still in local memory; a later payload summary
    # must not overwrite it or any raw assistant output.
    prior = copy.deepcopy(instance._history)
    stored = current_record({'input': prior})['observable_context']['visual_feedback']['scene_inventory']
    assert stored['revision_log'] == records
    instance.predict(observation(1).to_dict())
    assert instance._history[:len(prior)] == prior
    assistants = [json.loads(p['text']) for m in transport.calls[-1]['input'] if m.get('role') == 'assistant'
        for p in m.get('content', []) if p.get('type') == 'output_text']
    assert assistants == [raw]
    assert instance._v31_history_payload is None


def test_revision_presentation_preserves_unknown_extensions_and_raw_assistant_message():
    log = {'id': 'obj', 'kind': 'explicit_metric_revision',
        'observation_binding': {'state8': [0]*8, 'rgb_sha256': {'external': 'old'}, 'new_warning': 'preserve'},
        'previous_world_geometry': {'center_xyz_m': [1,2,3], 'new_geometry_fact': 7},
        'revision': {'id': 'obj', 'reason': 'old', 'evidence_kind': 'current_two_view', 'unknown_evidence': True},
        'unknown_fact': {'blocking': True}}
    record = {'type': 'observation', 'observable_context': {'visual_feedback': {'scene_inventory': {
        'entries': [{'id': 'obj', 'last_explicit_revision': log}], 'revision_log': [log]}}}}
    payload = {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': json.dumps(record)},
        {'type': 'input_image', 'image_url': 'data:image/png;base64,unchanged'}]},
        {'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps({'scene_inventory': [log]})}]}]}
    before = copy.deepcopy(payload)
    result = compact_scene_revision_payload(payload)
    shown = current_record(result)['observable_context']['visual_feedback']['scene_inventory']
    extension = shown['revision_log_extensions'][0]
    assert extension['unknown_fact'] == log['unknown_fact']
    assert extension['observation_binding']['new_warning'] == 'preserve'
    assert extension['previous_world_geometry']['new_geometry_fact'] == 7
    assert extension['revision']['unknown_evidence'] is True
    assert shown['entries'][0]['last_explicit_revision']['revision'] == log['revision']
    assert result['input'][1] == payload['input'][1]
    assert result['input'][0]['content'][1] == payload['input'][0]['content'][1]
    assert payload == before
