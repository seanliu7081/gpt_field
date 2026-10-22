"""V33 guidance only: fake transport, unchanged wire and no robot execution."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v32 import (
    output_schema as schema_v32, shared_guidance as guidance_v32,
    geometry_semantics as semantics_v32, HISTORY_VERSION as HISTORY_V32,
    FEEDBACK_VERSION as FEEDBACK_V32, restore_guard_debug_snapshot,
)
from gpt_field.visual_field_subscription_v33 import (
    VisualFieldSubscriptionClientV33, output_schema, shared_guidance,
    geometry_semantics, HISTORY_VERSION, FEEDBACK_VERSION, VERSION, RECOVERY_GUIDANCE,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v7_protocol import current_record, current_images, stream_error
from test_visual_field_v18_protocol import candidate_result, resolver_for
from test_visual_field_v32_protocol import prediction, _expanded_snapshot


def client(*responses):
    transport = FakeTransport(*(response if isinstance(response, BaseException) else envelope(
        response, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v33_{index}')
        for index, response in enumerate(responses)))
    return VisualFieldSubscriptionClientV33(
        token_provider=lambda: 'offline-oauth', transport=transport), transport


def test_only_guidance_changes_and_never_creates_permission_or_credit():
    task = resolve_distance_task('bowl_plate')
    assert output_schema() == schema_v32()
    assert geometry_semantics() == semantics_v32()
    assert HISTORY_VERSION == HISTORY_V32 and FEEDBACK_VERSION == FEEDBACK_V32
    assert shared_guidance(task) == guidance_v32(task) + '\n\n' + RECOVERY_GUIDANCE
    for phrase in ('current accepted', 'reverse_observed', 'incomplete', 'old heading',
                   'does not erase', 'refund', 'current geometry', 'Rejected or stale',
                   'not execution permission', 'clearance certificate'):
        assert phrase in RECOVERY_GUIDANCE
    assert 'bowl' not in RECOVERY_GUIDANCE.lower()
    assert 'cabinet' not in RECOVERY_GUIDANCE.lower()
    with pytest.raises(ValueError):
        shared_guidance(task, demonstration=True)


def test_actual_payload_preserves_raw_history_current_rgb_and_resolved_candidate():
    wire = prediction(candidate=True, controller='visual_field')
    original = copy.deepcopy(wire)
    instance, transport = client(wire, prediction())
    obs = observation().to_dict(); before = copy.deepcopy(obs); calls = []
    instance.reference_candidate_resolver = resolver_for(obs, calls)
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire == original
    assert result['reference_candidate_resolution'] == candidate_result()
    assert calls == [(wire['reference_candidate_id'], obs)]
    assert obs == before and instance._candidate_observation is None
    instance.predict(observation(1).to_dict())
    payload = transport.calls[-1]
    raw = [json.loads(part['text']) for message in payload['input']
           if message.get('role') == 'assistant' for part in message['content']
           if part['type'] == 'output_text']
    assert raw == [original]
    assert payload['text']['format']['schema'] == schema_v32()
    assert payload['text']['format']['name'] == 'visual_field_rgb_joints_v33'
    assert payload['instructions'].endswith(RECOVERY_GUIDANCE)
    assert instance.prompt_version == VERSION + '-hybrid'
    assert instance.history_stats()['protocol'] == HISTORY_V32
    assert current_images(transport.calls[0])[:2] == [obs['observation/image'], obs['observation/wrist_image']]
    assert instance.budget_snapshot()['api_key_calls'] == 0


@pytest.mark.parametrize('order', ['stream_first', 'schema_first'])
def test_bounded_repairs_still_share_observation_medium_route_and_budget(order):
    valid = prediction(phase='recover')
    valid['direct_actions'] = [[0.] * 7 for _ in range(5)]
    invalid = copy.deepcopy(valid); invalid['direct_actions'] = None
    errors = [stream_error(), invalid] if order == 'stream_first' else [invalid, stream_error()]
    instance, transport = client(*errors, valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction'] == valid
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    assert len({current_record(payload)['observation_index'] for payload in transport.calls}) == 1
    assert all(current_images(payload) == current_images(transport.calls[0]) for payload in transport.calls)
    assert all(payload['model'] == 'gpt-6-astra' and payload['reasoning'] == {'effort': 'medium'}
               for payload in transport.calls)


def test_authentication_failure_does_not_retry_or_fallback():
    instance, transport = client(stream_error(code='invalid_api_key', status=401), prediction())
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1
    assert instance.budget_snapshot()['api_key_calls'] == 0


def test_invalid_recovery_wire_does_not_gain_permission_from_guidance():
    wire = prediction(controller='visual_field', phase='approach')
    wire['recovery_route'] = 'reverse_observed'
    instance, _ = client()
    with pytest.raises(SchemaError):
        instance.validate_response(wire, None)


def test_current_checkpoint_feedback_choices_and_exact_debug_delta_survive():
    instance, transport = client(prediction())
    guard = {'configured': True, 'clearance_certified': False, 'translation_mode': 'separate',
             'recovery_experiments': {'shared_remaining_m': .007, 'old_heading_travel_m': .003},
             'precheckpoint_reverse_abort': {'old_checkpoint_abandoned': True, 'paid_debt_retained': True},
             'debug_samples': [{'residual': value} for value in range(300)]}
    snapshot = copy.deepcopy(guard)
    snapshot['precheckpoint_reverse_abort']['old_checkpoint_abandoned'] = False
    snapshot['future_guard_extension'] = {'must_survive': 'exact rejected preview'}
    choices = {'candidates': [{'accepted': True, 'goal_offset_m': [.005, 0., 0.],
                              'recovery_route': 'reverse_observed'}],
               'rejected_candidates': [{'accepted': False, 'reason': 'stale_current_binding'}]}
    context = {'visual_feedback': {'translation_guard': guard, 'recovery_options': choices,
        'field': {'translation_admission': {'accepted': False,
            'reason': 'prior_heading_checkpoint_incomplete', 'diagnostics': snapshot}}}}
    instance.observable_context = copy.deepcopy(context)
    instance.predict(observation().to_dict())
    feedback = current_record(transport.calls[0])['observable_context']['visual_feedback']
    assert feedback['translation_guard'] == guard
    assert feedback['recovery_options'] == choices
    admission = feedback['field']['translation_admission']
    assert admission['reason'] == 'prior_heading_checkpoint_incomplete'
    assert admission['accepted'] is False
    restored = restore_guard_debug_snapshot(_expanded_snapshot(admission['diagnostics'], guard), guard)
    assert restored.pop('snapshot_scope', None) == (
        'decision-time veto/budget evidence; current full geometry and response are in translation_guard')
    assert restored == snapshot
    assert instance.observable_context == context
