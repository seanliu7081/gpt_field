"""Task-resumption guidance with unchanged native wire and fake transport."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v33 import output_schema as schema_v33
from gpt_field.visual_field_subscription_v34 import (
    VisualFieldSubscriptionClientV34, output_schema, shared_guidance,
    geometry_semantics, HISTORY_VERSION, FEEDBACK_VERSION, VERSION, TASK_RESUMPTION_GUIDANCE,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v7_protocol import current_record, current_images, stream_error
from test_visual_field_v18_protocol import resolver_for, candidate_result
from test_visual_field_v32_protocol import prediction
from test_visual_field_v31_protocol import proposed, attach


def client(*responses):
    transport = FakeTransport(*(response if isinstance(response, BaseException) else envelope(
        response, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v34_{index}')
        for index, response in enumerate(responses)))
    return VisualFieldSubscriptionClientV34(
        token_provider=lambda: 'offline-oauth', transport=transport), transport


def test_generic_resumption_guidance_changes_no_schema_or_physical_contract():
    assert output_schema() == schema_v33()
    text = shared_guidance(resolve_distance_task('bowl_plate'))
    assert text.endswith(TASK_RESUMPTION_GUIDANCE)
    for phrase in ('not task completion', 'noncontact approach', 'first-command checks',
                   'field_proposal_id', 'reference_candidate_id', 'cannot be selected together',
                   'weak-response review', 'not verified identities', 'null drops',
                   'Retain still-current', 'measured-budget checks'):
        assert phrase in TASK_RESUMPTION_GUIDANCE
    assert 'cabinet' not in TASK_RESUMPTION_GUIDANCE.lower()
    assert 'bowl' not in TASK_RESUMPTION_GUIDANCE.lower()
    assert 'precheckpoint_reverse_abort' in text
    semantics = geometry_semantics()
    assert semantics['recovery_to_approach_proposals']['full_route_and_first_command_checks_unchanged']
    assert not semantics['recovery_to_approach_proposals']['physical_budget_replenished']
    assert not semantics['reference_choice_contract']['automatic_identity_promotion']
    assert not semantics['final_feedback_contract']['prepares_new_motion_offers']
    assert semantics['final_feedback_contract']['observer_and_control_updates_keep_original_order']
    assert semantics['final_feedback_contract']['complete_current_row_rendered_once_after_observers']
    assert not semantics['final_feedback_contract']['across_observation_cache']
    assert semantics['final_feedback_contract']['control_timing_excludes_only_video']
    assert HISTORY_VERSION.startswith('visual-v32-') and FEEDBACK_VERSION.endswith('v32-v1')
    with pytest.raises(ValueError):
        shared_guidance(resolve_distance_task('bowl_plate'), demonstration=True)


def test_candidate_selection_keeps_current_coordinates_raw_history_and_original_views():
    wire = prediction(candidate=True, controller='visual_field')
    before = copy.deepcopy(wire)
    instance, transport = client(wire, prediction())
    obs = observation().to_dict(); calls = []
    instance.reference_candidate_resolver = resolver_for(obs, calls)
    result = instance.predict(obs)
    assert result['reference_candidate_resolution'] == candidate_result()
    assert result['raw_wire_prediction'] == wire == before
    assert calls == [(wire['reference_candidate_id'], obs)]
    instance.predict(observation(1).to_dict())
    payload = transport.calls[-1]
    assert payload['text']['format']['name'] == 'visual_field_rgb_joints_v34'
    assert payload['text']['format']['schema'] == schema_v33()
    assert payload['instructions'].endswith(TASK_RESUMPTION_GUIDANCE)
    assert instance.prompt_version == VERSION + '-hybrid'
    raw = [json.loads(p['text']) for m in payload['input'] if m.get('role') == 'assistant'
           for p in m['content'] if p['type'] == 'output_text']
    assert raw == [before]
    assert current_images(transport.calls[0])[:2] == [obs['observation/image'], obs['observation/wrist_image']]


def test_explicit_approach_offer_resolves_and_does_not_allow_a_reference_selection_together():
    wire, resolved, context = proposed()
    for value in (wire, resolved):
        value['phase'] = 'approach'
        value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    context['visual_feedback']['field_proposals'] = {'offers': [copy.deepcopy(resolved)],
        'phase_transition': {'from_phase': 'recover', 'offered_phase': 'approach',
                             'explicit_model_choice_required': True, 'live_phase_changed': False}}
    obs = observation().to_dict()
    instance, transport = client(wire)
    attach(instance, resolved, context, obs)
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire and wire['field_intent'] is None
    assert result['field_proposal_resolution'] == resolved
    assert result['phase'] == 'approach'
    sent = current_record(transport.calls[0])['observable_context']['visual_feedback']['field_proposals']
    assert sent == context['visual_feedback']['field_proposals']
    invalid = copy.deepcopy(wire); invalid['reference_candidate_id'] = 'simultaneous-not-allowed'
    instance, _ = client(); attach(instance, resolved, context, obs)
    instance._candidate_observation = obs
    with pytest.raises(SchemaError):
        instance.validate_response(invalid, None)


def test_same_observation_bounded_repair_preserves_medium_subscription_without_fallback():
    valid = prediction(); invalid = copy.deepcopy(valid); invalid['direct_actions'] = None
    instance, transport = client(stream_error(), invalid, valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction'] == valid
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    assert all(p['reasoning'] == {'effort': 'medium'} and p['model'] == 'gpt-6-astra' for p in transport.calls)
    assert len({current_record(p)['observation_index'] for p in transport.calls}) == 1
    assert all(current_images(p) == current_images(transport.calls[0]) for p in transport.calls)
    blocked, attempts = client(stream_error(code='insufficient_quota', status=429), valid)
    with pytest.raises(APIError): blocked.predict(observation().to_dict())
    assert len(attempts.calls) == 1 and blocked.budget_snapshot()['api_key_calls'] == 0
