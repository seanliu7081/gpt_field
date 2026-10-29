"""V22-only presentation delta, preserving native wire and causal references."""
import copy
import json

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v21 import output_schema as schema_v21
from gpt_field.visual_field_subscription_v22 import (
    VisualFieldSubscriptionClientV22, output_schema, shared_guidance, geometry_semantics,
    VERSION, HISTORY_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v18_protocol import prediction, candidate_result, resolver_for
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport=FakeTransport(*(r if isinstance(r,BaseException)else envelope(
        r,model='gpt-6-astra',reasoning={'effort':'medium'},id=f'v22_{i}')
        for i,r in enumerate(responses)))
    return VisualFieldSubscriptionClientV22(token_provider=lambda:'offline-oauth',transport=transport),transport


def test_new_guidance_only_changes_checked_recovery_and_never_grants_permission():
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    assert output_schema()==schema_v21()
    for phrase in ('recommended_goal_offset_m','recommended_goal_tolerance_m','ACTUAL candidate heading',
                   'endpoint alone','Every executed segment','No direction or travel allowance is reset',
                   'NEVER hand/palm','exactly FIVE','reference_candidate_id','task_progress',
                   'ALL visible nonrobot objects'):
        assert phrase in text
    assert '36.54'not in text and '232'not in text
    semantics=geometry_semantics()['heading_constrained_recovery']
    assert semantics['model_selected_goal_required']is True
    assert semantics['actual_proposal_direction_checked']is True
    assert semantics['full_scene_contact_reference_checks_retained']is True
    assert semantics['debt_reset']is semantics['clearance_certified']is False


def test_real_client_keeps_raw_wire_two_views_history_and_exact_candidate_binding():
    raw=prediction(candidate=True);ordinary=prediction();original=copy.deepcopy(raw)
    instance,transport=client(raw,ordinary);obs=observation().to_dict();before=copy.deepcopy(obs)
    calls=[];instance.reference_candidate_resolver=resolver_for(obs,calls)
    instance.observable_context={'visual_feedback':{'reference_repair':{'review_required':True}}}
    result=instance.predict(obs)
    assert result['reference_candidate_resolution']==candidate_result()
    assert result['raw_wire_prediction']==raw==original and obs==before
    assert result['visual_references']['external']['seed']['moving_point_uv']==candidate_result()['current_point_uv']
    assert raw['visual_references']['external']['seed']['moving_point_uv']is None
    instance.observable_context={};instance.predict(observation(1).to_dict())
    payload=transport.calls[-1]
    histories=[json.loads(p['text'])for m in payload['input']if m.get('role')=='assistant'
        for p in m.get('content',[])if p['type']=='output_text']
    assert histories==[original]
    assert payload['text']['format']['schema']==schema_v21()
    assert payload['text']['format']['name']=='visual_field_rgb_joints_v22'
    assert instance.prompt_version==VERSION+'-hybrid'
    assert instance.history_stats()['protocol']==HISTORY_VERSION
    assert current_images(transport.calls[0])[:2]==[obs['observation/image'],obs['observation/wrist_image']]
    assert instance.budget_snapshot()['api_key_calls']==0


def test_mixed_bounded_transport_and_schema_retry_keeps_one_observation_and_route():
    invalid=prediction();invalid['direct_actions']=None
    valid=prediction(phase='recover');valid['direct_actions']=[[0.]*7 for _ in range(5)]
    instance,transport=client(stream_error(),invalid,valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==valid
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}for p in transport.calls)
    assert all(current_images(p)==current_images(transport.calls[0])for p in transport.calls)


def test_inherited_compaction_keeps_new_current_route_admission_and_recovery_contract():
    instance,transport=client(prediction())
    heading={'route_id':'observed-reverse-1','goal_xyz_m':[.1,.2,.3],
        'geometry_sha256':'exact-model-geometry','actual_direction_valid':False,
        'reason':'current_scene_refusal','measured_remaining_m':.002}
    contract={'explicit_model_choice_required':True,'physical_budgets_reset':False}
    instance.observable_context={'visual_feedback':{'field':{'heading_recovery':heading},
        'recovery_execution_contract':contract}}
    instance.predict(observation().to_dict())
    feedback=current_record(transport.calls[0])['observable_context']['visual_feedback']
    assert feedback['field']['heading_recovery']==heading
    assert feedback['recovery_execution_contract']==contract
