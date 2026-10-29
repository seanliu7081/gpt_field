"""V36 paid review handoff: real validators and offline subscription transport."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_feedback_compact_v35 import expand_current_feedback
from gpt_field.visual_field_subscription_v35 import (
    VisualFieldSubscriptionClientV35, output_schema as schema35,
    geometry_semantics as semantics35, HISTORY_VERSION as HISTORY35,
)
from gpt_field.visual_field_subscription_v36 import (
    VisualFieldSubscriptionClientV36, VERSION, HISTORY_VERSION, output_schema,
    geometry_semantics, shared_guidance, OBJECTIVE_REVIEW_GUIDANCE,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v7_protocol import current_record, current_images, stream_error
from test_visual_field_v18_protocol import candidate_result, resolver_for
from test_visual_field_v31_protocol import proposed, attach
from test_visual_field_v32_protocol import prediction


def client(cls=VisualFieldSubscriptionClientV36, *responses):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v36_{index}')
        for index, value in enumerate(responses)))
    return cls(token_provider=lambda: 'offline-oauth', transport=transport), transport


def review_context():
    event = {'reason': 'wrist_target_visibility_declining_under_current_heading',
             'measured_path_m': .013456789012345, 'heading_xyz': [0., 0., -1.],
             'predicted_uv': [.97345123456789, .135], 'target_id': 'object-A',
             'clearance_certified': False, 'physical_budget_changed': False}
    return {'refresh_reason': event['reason'], 'visual_feedback': {
        'ongoing_objective_review': {
            'active': True, 'new_event': True, 'review_required': True,
            'status': 'review_requested', 'current_evidence': copy.deepcopy(event),
            'event': copy.deepcopy(event), 'archive': [copy.deepcopy(event)],
        },
        'translation_guard': {
            'allowed': False, 'reason': 'preserved_current_adverse_evidence',
            'shared_remaining_m': 0.0000123456789,
            'obstacles': [{'id': 'fixture-B', 'unknown': True, 'margin_m': .04321}],
        },
        'field_proposals': {'offers': [], 'rejected': [{'reason': 'no_validated_heading'}]},
        'ordinary_replan_previews': {'automatic_action': False, 'candidates': [
            {'goal_offset_m': [.02, 0., 0.], 'accepted': True,
             'field_intent': {'max_travel_m': .0301234567},
             'visual_improvement_certified': False, 'clearance_certified': False},
            {'goal_offset_m': [-.02, 0., 0.], 'accepted': False,
             'reason': 'preserved_current_obstacle_refusal'},
        ]},
        'execution_limit_sources': {'effective_intent_max_travel_m': .0167891234},
    }}


def test_review_payload_changes_only_guidance_and_schema_name_not_facts_or_history():
    wires = [prediction(), prediction()]
    old, old_transport = client(VisualFieldSubscriptionClientV35, *wires)
    new, new_transport = client(VisualFieldSubscriptionClientV36, *wires)
    context = review_context()
    before = copy.deepcopy(context)
    observations = [observation(index).to_dict() for index in range(2)]
    original_observations = copy.deepcopy(observations)
    for obs in observations:
        for instance in (old, new):
            instance.observable_context = copy.deepcopy(context)
            instance.predict(obs)
    for previous, current in zip(old_transport.calls, new_transport.calls):
        expected = copy.deepcopy(previous)
        expected['instructions'] = shared_guidance(new.task_spec)
        expected['text']['format']['name'] = 'visual_field_rgb_joints_v36'
        assert expected == current
        feedback = expand_current_feedback(current_record(current)['observable_context']['visual_feedback'])
        assert feedback['ongoing_objective_review'] == context['visual_feedback']['ongoing_objective_review']
        assert feedback['translation_guard'] == context['visual_feedback']['translation_guard']
        assert feedback['field_proposals'] == context['visual_feedback']['field_proposals']
        assert feedback['ordinary_replan_previews'] == context['visual_feedback']['ordinary_replan_previews']
        assert current_images(previous) == current_images(current)
        assert current['text']['format']['schema'] == schema35() == output_schema()
    raw = [json.loads(part['text']) for message in new_transport.calls[-1]['input']
           if message.get('role') == 'assistant' for part in message['content']
           if part['type'] == 'output_text']
    assert raw == wires[:1]
    assert context == before == new.observable_context
    assert observations == original_observations
    assert new.prompt_version == VERSION + '-hybrid'
    assert new.history_stats()['protocol'] == HISTORY_VERSION == HISTORY35


def test_native_ordinary_replacement_can_share_reference_repair_without_granting_admission():
    wire = prediction(controller='visual_field', candidate=True)
    wire['subgoal_update'] = {'action': 'replace', 'subgoal_id': 'unfinished-goal',
        'reason': 'Current observed target moves toward the image edge under prior heading.'}
    original = copy.deepcopy(wire)
    instance, transport = client(VisualFieldSubscriptionClientV36, wire)
    obs = observation().to_dict()
    calls = []
    instance.reference_candidate_resolver = resolver_for(obs, calls)
    instance.observable_context = review_context()
    result = instance.predict(obs)
    assert result['raw_wire_prediction'] == wire == original
    assert result['subgoal_update'] == wire['subgoal_update']
    assert result['field_proposal_resolution'] is None
    assert result['reference_candidate_resolution'] == candidate_result()
    assert calls == [(wire['reference_candidate_id'], obs)]
    for key in ['goal_offset_m', 'max_travel_m', 'command_limit_m', 'tool_radius_m', 'uncertainty_m']:
        assert result['field_intent'][key] == wire['field_intent'][key]
    assert result['scene_inventory'] == wire['scene_inventory']
    assert result['target_contact'] == 'none'
    assert instance.observable_context == review_context()
    assert len(transport.calls) == 1


def current_offer():
    wire, resolved, context = proposed()
    for value in (wire, resolved):
        value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    return wire, resolved, context


def test_exact_offer_and_same_observation_resolution_remain_unchanged():
    wire, resolved, context = current_offer()
    instance, _ = client(VisualFieldSubscriptionClientV36, wire)
    obs = observation().to_dict()
    calls = []
    attach(instance, resolved, context, obs, calls)
    result = instance.predict(obs)
    assert result['field_proposal_resolution'] == resolved
    assert result['raw_wire_prediction'] == wire
    assert calls == [(wire['field_proposal_id'], obs)]


@pytest.mark.parametrize('change', ['reseed_wrist', 'revise_scene', 'different_contract', 'stale_id'])
def test_review_guidance_does_not_relax_exact_offer_contract(change):
    wire, resolved, context = current_offer()
    if change == 'reseed_wrist':
        wire['visual_references']['wrist'] = {'target_id': wire['target_id'],
            'update': 'seed', 'desired_target_uv': [.5, .5],
            'seed': {'target_anchor_uv': [.4, .3], 'target_roi_uv': [.3, .2, .5, .4]}}
    elif change == 'revise_scene':
        wire['scene_inventory'][0]['geometry'] = copy.deepcopy(context['visual_feedback']['scene_inventory']['entries'][0]['current_geometry'])
        wire['scene_revisions'] = [{'id': wire['target_id'],
            'reason': 'Revised hypothesis', 'evidence_kind': 'measured_motion_conflict'}]
    elif change == 'different_contract':
        wire['motion_context']['tool_external_roi_uv'][0] += .001
    else:
        wire['field_proposal_id'] = 'not-the-current-offer'
    instance, _ = client()
    obs = observation().to_dict()
    attach(instance, resolved, context, obs)
    instance._candidate_observation = obs
    with pytest.raises(SchemaError):
        instance.validate_response(wire, None)


@pytest.mark.parametrize('order', ['stream_first', 'schema_first'])
def test_bounded_native_repairs_keep_review_current_images_medium_and_oauth(order):
    valid = prediction()
    invalid = copy.deepcopy(valid)
    invalid['direct_actions'] = None
    errors = [stream_error(), invalid] if order == 'stream_first' else [invalid, stream_error()]
    instance, transport = client(VisualFieldSubscriptionClientV36, *errors, valid)
    instance.observable_context = review_context()
    assert instance.predict(observation().to_dict())['raw_wire_prediction'] == valid
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    first = current_record(transport.calls[0])
    for payload in transport.calls:
        assert payload['model'] == 'gpt-6-astra'
        assert payload['reasoning'] == {'effort': 'medium'}
        current = current_record(payload)
        # Existing transport/schema repair metadata is intentionally added;
        # the observation, review evidence, limits and choices must not change.
        for key in ('transport_retry', 'output_repair'):
            current['observable_context'].pop(key, None)
        assert current == first
        assert current_images(payload) == current_images(transport.calls[0])


@pytest.mark.parametrize('code,status', [('invalid_api_key', 401), ('insufficient_quota', 429)])
def test_authorization_and_quota_do_not_retry_or_fallback(code, status):
    instance, transport = client(VisualFieldSubscriptionClientV36,
        stream_error(code=code, status=status), prediction())
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1
    assert instance.budget_snapshot()['api_key_calls'] == 0


def test_guidance_is_task_general_zero_shot_with_explicit_non_authority_contract():
    first = resolve_distance_task('bowl_plate')
    second = resolve_distance_task('drawer_bowl')
    assert shared_guidance(first).replace(first.prompt, '<TASK>') == shared_guidance(second).replace(second.prompt, '<TASK>')
    with pytest.raises(ValueError):
        shared_guidance(first, demonstration=True)
    assert output_schema() == schema35()
    before = semantics35()
    semantics = geometry_semantics()
    contract = semantics.pop('ongoing_objective_review_contract')
    semantics['version'] = before['version']
    assert semantics == before
    assert contract['review_requires_explicit_model_decision'] is True
    assert contract['ordinary_field_replacement_remains_available_subject_to_admission'] is True
    for key in ['unobserved_heading_is_predictively_validated',
                'declared_desired_bearing_is_grasp_or_target_pose',
                'review_is_collision_or_clearance_evidence',
                'review_completes_goal_or_changes_budget',
                'ordinary_replan_preview_certifies_visual_improvement_or_clearance',
                'changed_bindings_preserve_preview_authority',
                'automatic_action_or_stage_change', 'wire_schema_changed']:
        assert contract[key] is False
    assert 'SAME active subgoal_id' in OBJECTIVE_REVIEW_GUIDANCE
    assert 'retain ignores' in OBJECTIVE_REVIEW_GUIDANCE
    assert 'field_proposal_id=null' in OBJECTIVE_REVIEW_GUIDANCE
    assert 'not an observed feature, target pose or grasp/pinch location' in OBJECTIVE_REVIEW_GUIDANCE
    assert 'no clearance' in OBJECTIVE_REVIEW_GUIDANCE
    assert 'bowl' not in OBJECTIVE_REVIEW_GUIDANCE.lower()
    assert 'cabinet' not in OBJECTIVE_REVIEW_GUIDANCE.lower()
