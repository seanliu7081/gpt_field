"""Selected IDs ground exact current points; no model/robot calls in this suite."""
import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v18 import (
    VisualFieldSubscriptionClientV18, output_schema, shared_guidance, CANDIDATE_LABEL,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v16_protocol import prediction as old_prediction
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def prediction(*, candidate=False, **kwargs):
    value=old_prediction(**kwargs);value['reference_candidate_id']='board4.c2' if candidate else None
    if candidate:
        ref=value['visual_references']['external'];ref['update']='seed';ref['moving_kind']='tool'
        for key in ('moving_point_uv','moving_roi_uv','tool_point_uv','tool_roi_uv'):ref['seed'][key]=None
        value['motion_context']['tool_external_roi_uv']=[.1,.1,.6,.6]
    return value


def client(*responses, **kwargs):
    transport=FakeTransport(*(r if isinstance(r,BaseException)else envelope(r,model='gpt-6-astra',
        reasoning={'effort':'medium'},id=f'v18_{i}')for i,r in enumerate(responses)))
    return VisualFieldSubscriptionClientV18(token_provider=lambda:'offline-oauth',transport=transport,**kwargs),transport


def candidate_result():
    return {'candidate_id':'board4.c2','current_point_uv':[.23123456789,.34123456789],
            'current_roi_uv':[.2,.3,.29,.4],'evidence':[{'step':3,'measured_xyz':[.1,.2,.3]}],
            'observation_binding':'fixture-current','semantic_identity_certified':False}


def resolver_for(expected, calls):
    def resolve(identity,obs):
        calls.append((identity,copy.deepcopy(obs)))
        if identity!='board4.c2' or obs!=expected:raise ValueError('arbitrary callback body must remain private')
        return candidate_result()
    return resolve


def test_exact_candidate_resolution_preserves_wire_and_assistant_history_and_observation():
    raw=prediction(candidate=True,controller='visual_field');original=copy.deepcopy(raw)
    instance,transport=client(raw,prediction());obs=observation().to_dict();before=copy.deepcopy(obs);calls=[]
    instance.reference_candidate_resolver=resolver_for(obs,calls)
    result=instance.predict(obs)
    seed=result['visual_references']['external']['seed']
    assert seed['moving_point_uv']==candidate_result()['current_point_uv']
    assert seed['moving_roi_uv']==candidate_result()['current_roi_uv']
    assert result['reference_candidate_resolution']==candidate_result()
    assert result['reference_candidate_id']=='board4.c2'
    assert result['raw_wire_prediction']==raw==original and obs==before
    assert instance._candidate_observation is None
    assert calls==[('board4.c2',obs)]
    instance.predict(observation(1).to_dict())
    history=[json.loads(p['text'])for m in transport.calls[-1]['input']if m.get('role')=='assistant'
             for p in m.get('content',[])if p.get('type')=='output_text']
    assert history==[original] and history[0]['visual_references']['external']['seed']['moving_point_uv'] is None


@pytest.mark.parametrize('case',['retain','payload','manual_point','manual_roi','auxiliary_tool','missing_whole_tool','point_outside_whole_tool','missing_resolver'])
def test_candidate_cannot_ambiguously_override_manual_coordinates_or_roles(case):
    raw=prediction(candidate=True);ext=raw['visual_references']['external']
    if case=='retain':ext['update']='retain';ext['seed']=None
    elif case=='payload':ext['moving_kind']='payload'
    elif case=='manual_point':ext['seed']['moving_point_uv']=[.2,.3]
    elif case=='manual_roi':ext['seed']['moving_roi_uv']=[.2,.3,.4,.5]
    elif case=='auxiliary_tool':ext['seed']['tool_point_uv']=[.2,.3]
    elif case=='missing_whole_tool':raw['motion_context']['tool_external_roi_uv']=None
    elif case=='point_outside_whole_tool':raw['motion_context']['tool_external_roi_uv']=[.7,.7,.9,.9]
    instance,transport=client(raw,raw);obs=observation().to_dict()
    if case!='missing_resolver':instance.reference_candidate_resolver=resolver_for(obs,[])
    with pytest.raises(SchemaError):instance.predict(obs)
    assert len(transport.calls)==2 and instance._candidate_observation is None
    assert instance.history_snapshot()==[] and instance.budget_snapshot()['api_key_calls']==0


def test_null_candidate_does_not_relax_ordinary_seed_point_validation():
    value=prediction();value['visual_references']['external']['seed']['moving_point_uv']=None
    instance,_=client()
    with pytest.raises(SchemaError):instance.validate_response(value,None)
    value=prediction();result=instance.validate_response(value,None)
    assert result['reference_candidate_id']is None and result['reference_candidate_resolution']is None
    assert result['raw_wire_prediction']==value


@pytest.mark.parametrize('order',['stream_first','schema_first'])
def test_unknown_candidate_and_stream_share_existing_bounded_retry_exact_observation(order,tmp_path):
    invalid=prediction(candidate=True);invalid['reference_candidate_id']='unknown'
    good=prediction(candidate=True)
    sequence=[stream_error(),invalid]if order=='stream_first'else[invalid,stream_error()]
    instance,transport=client(*sequence,good,log_dir=tmp_path)
    obs=observation().to_dict();calls=[];instance.reference_candidate_resolver=resolver_for(obs,calls)
    assert instance.predict(obs)['raw_wire_prediction']==good
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0 and instance._candidate_observation is None
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(current_images(p)==current_images(transport.calls[0])for p in transport.calls)
    assert all(p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}for p in transport.calls)
    repair=current_record(transport.calls[-1])['observable_context']['output_repair']
    assert 'arbitrary callback body'not in json.dumps(repair)
    assert json.loads(repair['rejected_model_output_json'])==invalid
    result_logs=[json.loads(p.read_text())for p in sorted(tmp_path.glob('*.result.json'))]
    assert [r['status']for r in result_logs]==['error','error','valid']


def test_stale_candidate_cannot_roll_over_to_new_observation_or_survive_failure():
    value=prediction(candidate=True);instance,transport=client(value,value,value)
    first=observation().to_dict();instance.reference_candidate_resolver=resolver_for(first,[])
    instance.predict(first)
    with pytest.raises(SchemaError,match='unknown, stale or unavailable'):
        instance.predict(observation(1).to_dict())
    assert len(transport.calls)==3 and instance._candidate_observation is None
    assert instance.history_stats()['observations']==1


@pytest.mark.parametrize('error',[stream_error(code='invalid_api_key',status=401),stream_error(code='insufficient_quota',status=429),TimeoutError('network timeout')])
def test_other_transport_errors_never_retry_candidate_or_fallback(error):
    instance,transport=client(error,prediction(candidate=True));obs=observation().to_dict();calls=[]
    instance.reference_candidate_resolver=resolver_for(obs,calls)
    with pytest.raises(APIError):instance.predict(obs)
    assert len(transport.calls)==1 and not calls and instance._candidate_observation is None
    assert instance.budget_snapshot()['api_key_calls']==0


def test_current_bound_candidate_board_replaces_only_diagnostic_and_does_not_cache():
    instance,transport=client(prediction(),prediction());first=observation().to_dict();second=observation(1).to_dict()
    board=first['observation/wrist_image'];calls=[]
    def provider(obs):
        calls.append(copy.deepcopy(obs));return board if obs==first else None
    instance.reference_candidate_provider=provider
    instance.observable_context={'visual_feedback':{'reference_repair':{'review_required':True,'candidates':[{'candidate_id':'board4.c2'}]}}}
    instance.predict(first);instance.predict(second)
    assert calls==[first,second]
    assert current_images(transport.calls[0])==[first['observation/image'],first['observation/wrist_image'],board]
    assert current_images(transport.calls[1])==[second['observation/image'],second['observation/wrist_image']]
    assert CANDIDATE_LABEL in json.dumps(transport.calls[0])
    assert transport.calls[0]['text']['format']['schema']==output_schema()


def test_guidance_resolves_weak_tracking_vs_static_support_review_without_false_certainty():
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    assert 'review even if reference_health still says weakly_observable' in text
    assert 'does not require retaining the same disputed physical feature' in text
    assert 'not merely the most moving patch' in text and 'no action is automatically added' in text
    assert 'same-direction measured debt remains' in text
    assert 'max(30mm, target uncertainty)'in text and 'NEVER hand/palm'in text
