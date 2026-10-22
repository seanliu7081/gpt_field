"""Explicit planner-family selection; fake transport only, no robot/model calls."""
import copy
import json
import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v24 import output_schema as schema_v24
from gpt_field.visual_field_subscription_v25 import (
    VisualFieldSubscriptionClientV25, output_schema, shared_guidance, geometry_semantics,
    VERSION, HISTORY_VERSION, FEEDBACK_VERSION, RECOVERY_ROUTES,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v18_protocol import prediction as old_prediction, candidate_result, resolver_for
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def prediction(*, route='normal', **kwargs):
    value = old_prediction(**kwargs)
    value['recovery_route'] = route
    if route != 'normal':
        value['motion_context'].update(translation_mode='separate', intended_contact_id=None,
            purpose='reference_repair')
    return value


def client(*responses):
    transport = FakeTransport(*(r if isinstance(r, BaseException) else envelope(
        r, model='gpt-6-astra', reasoning={'effort':'medium'}, id=f'v25_{i}')
        for i, r in enumerate(responses)))
    return VisualFieldSubscriptionClientV25(token_provider=lambda:'offline-oauth', transport=transport), transport


def test_only_one_new_required_enum_and_guidance_describes_real_dispatch():
    schema = output_schema(); old = copy.deepcopy(schema)
    assert old['properties'].pop('recovery_route') == {'type':'string','enum':list(RECOVERY_ROUTES)}
    old['required'].remove('recovery_route'); assert old == schema_v24()
    text = shared_guidance(resolve_distance_task('bowl_plate'))
    for phrase in ('Phase/purpose labels alone never select a route', 'Direct and gripper always specify normal',
                   'purpose may describe task_translation or reference_repair', 'also retains its route',
                   'absence of worsening is NOT evidence of free space', 'geometry=null'):
        assert phrase in text
    assert 'cabinet' not in text.lower()
    semantics = geometry_semantics()['recovery_route_selection']
    assert semantics['preview_and_actual_same_family'] and not semantics['physical_budget_reset']
    assert not semantics['phase_or_purpose_dispatch'] and not semantics['clearance_certified']
    with pytest.raises(ValueError):shared_guidance(resolve_distance_task('bowl_plate'), True)


@pytest.mark.parametrize('route', RECOVERY_ROUTES)
def test_each_field_family_validates_with_exact_wire_and_no_mutation(route):
    value = prediction(route=route, controller='visual_field', phase='recover')
    original = copy.deepcopy(value); instance, _ = client()
    result = instance.validate_response(value, None)
    assert value == original == result['raw_wire_prediction']
    assert result['recovery_route'] == route


@pytest.mark.parametrize('purpose', ['task_translation', 'reference_repair'])
def test_reverse_family_is_independent_of_semantic_purpose(purpose):
    value = prediction(route='reverse_observed', controller='visual_field', phase='recover')
    value['motion_context']['purpose'] = purpose
    instance, _ = client()
    assert instance.validate_response(value, None)['recovery_route'] == 'reverse_observed'


@pytest.mark.parametrize('case', ['missing', 'unknown', 'direct_reverse', 'gripper_egress', 'approach_reverse', 'transit_reverse', 'egress_task_purpose'])
def test_malformed_or_incompatible_dispatch_never_reaches_controller(case):
    value = prediction(route='reverse_observed', controller='visual_field', phase='recover')
    if case == 'missing':value.pop('recovery_route')
    elif case == 'unknown':value['recovery_route'] = 'auto'
    elif case == 'direct_reverse':value['controller'] = 'direct'
    elif case == 'gripper_egress':value['controller'] = 'gripper'; value['recovery_route'] = 'initial_uncertainty_egress'
    elif case == 'approach_reverse':value['phase'] = 'approach'
    elif case == 'transit_reverse':value['motion_context']['translation_mode'] = 'transit'
    else:value['recovery_route'] = 'initial_uncertainty_egress'; value['motion_context']['purpose'] = 'task_translation'
    instance, _ = client()
    with pytest.raises(SchemaError):instance.validate_response(value, None)


def test_candidate_resolver_and_raw_history_keep_explicit_route_and_null_wire_coordinates():
    value = prediction(route='reverse_observed', candidate=True, controller='visual_field', phase='recover')
    valid_next = prediction(); instance, transport = client(value, valid_next)
    obs = observation().to_dict(); instance.reference_candidate_resolver = resolver_for(obs, [])
    result = instance.predict(obs)
    assert result['reference_candidate_resolution'] == candidate_result()
    assert result['recovery_route'] == 'reverse_observed' and result['raw_wire_prediction'] == value
    assert value['visual_references']['external']['seed']['moving_point_uv'] is None
    history = copy.deepcopy(instance._history)
    instance.observable_context = {'visual_feedback':{'current_unknown_extension':{'do_not_drop':True},
        'recovery_route_binding':{'recovery_route':'reverse_observed','accepted':False,'reason':'current_reason'}}}
    instance.predict(observation(1).to_dict())
    assert instance._history[:len(history)] == history
    payload = transport.calls[-1]
    records = [json.loads(p['text']) for m in payload['input'] if m.get('role')=='assistant'
        for p in m.get('content',[]) if p.get('type')=='output_text']
    assert records == [value]
    assert current_record(payload)['observable_context']['visual_feedback']['current_unknown_extension']=={'do_not_drop':True}
    assert payload['text']['format']['name']=='visual_field_rgb_joints_v25'
    assert instance.prompt_version==VERSION+'-hybrid' and instance.history_stats()['protocol']==HISTORY_VERSION
    assert instance.budget_snapshot()['api_key_calls']==0


@pytest.mark.parametrize('order',['stream_first','schema_first'])
def test_new_selector_repair_shares_same_observation_bounded_retry(order):
    invalid=prediction();invalid.pop('recovery_route');good=prediction()
    failures=[stream_error(),invalid] if order=='stream_first' else [invalid,stream_error()]
    instance,transport=client(*failures,good)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==good
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index'] for p in transport.calls})==1
    assert all(current_images(p)==current_images(transport.calls[0]) for p in transport.calls)
    assert all(p['model']=='gpt-6-astra' and p['reasoning']=={'effort':'medium'} for p in transport.calls)
    repair=current_record(transport.calls[-1])['observable_context']['output_repair']
    assert json.loads(repair['rejected_model_output_json'])==invalid


def test_no_quota_retry_or_model_api_fallback():
    instance,transport=client(stream_error(code='insufficient_quota',status=429),prediction())
    with pytest.raises(APIError):instance.predict(observation().to_dict())
    assert len(transport.calls)==1 and instance.budget_snapshot()['api_key_calls']==0


def test_finite_reverse_guidance_keeps_existing_egress_and_all_adverse_checks():
    from gpt_field.visual_field_subscription_v24 import geometry_semantics as semantics_v24
    semantic = geometry_semantics(); finite = semantic['finite_reverse_route']
    assert finite['original_route_travel_limit_m'] == finite['per_direction_limit_m'] == .008
    assert finite['shared_uncertain_travel_limit_m'] == .020 and finite['maximum_command_m'] == .003
    assert finite['positive_gap_required_at_4mm'] is False
    assert finite['four_mm_observation_is_telemetry_not_new_travel_credit']
    assert finite['full_scene_unknown_target_contact_checks_retained']
    assert finite['further_worsening_lost_reference_or_domain_change_still_stops']
    assert finite['previous_measured_motion_and_hold_drift_charged']
    assert not finite['rename_reseed_replace_or_refresh_renews_allowance']
    assert not finite['clearance_certified']
    assert semantic['initial_uncertainty_egress'] == semantics_v24()['initial_uncertainty_egress']
    response = semantic['recent_response_evidence']
    assert response['contiguous_positive_nondeclining_samples'] == 2
    assert response['pair_aggregate_must_be_distinguishable'] and response['unchanged_full_history_empirical_envelope']
    assert not response['requested_command_amplified'] and response['physical_allowance_added_m'] == 0.
    text = shared_guidance(resolve_distance_task('bowl_plate'))
    assert '4 mm observed-improvement checkpoint' not in text
    for phrase in ('already bounded route', 'current further worsening', 'hold drift consumes',
                   'initial <=4 mm reference probe', 'SAME total allowance'):
        assert phrase in text
