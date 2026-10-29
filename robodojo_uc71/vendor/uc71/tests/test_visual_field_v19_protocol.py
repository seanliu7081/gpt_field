"""V19 clarifies geometry admission; V18 wire and local grounding remain exact."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v18 import output_schema as schema_v18
from gpt_field.visual_field_subscription_v19 import (
    VisualFieldSubscriptionClientV19, output_schema, shared_guidance,
    geometry_semantics, VERSION, HISTORY_VERSION, CANDIDATE_LABEL,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v18_protocol import prediction, candidate_result, resolver_for
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport = FakeTransport(*(r if isinstance(r, BaseException) else envelope(
        r, model='gpt-6-astra', reasoning={'effort':'medium'}, id=f'v19_{i}')
        for i, r in enumerate(responses)))
    return VisualFieldSubscriptionClientV19(token_provider=lambda:'offline-oauth', transport=transport), transport


def test_wire_schema_raw_history_rgb_state_and_subscription_route_unchanged():
    value=prediction(controller='visual_field');before=copy.deepcopy(value)
    instance,transport=client(value,prediction());obs=observation().to_dict();original=copy.deepcopy(obs)
    result=instance.predict(obs);instance.predict(observation(1).to_dict())
    assert output_schema()==schema_v18()
    assert result['raw_wire_prediction']==value==before and obs==original
    p=transport.calls[-1]
    histories=[json.loads(c['text'])for m in p['input']if m.get('role')=='assistant'
               for c in m.get('content',[])if c['type']=='output_text']
    assert histories==[before]
    assert p['text']['format']['name']=='visual_field_rgb_joints_v19'
    assert p['text']['format']['schema']==schema_v18()
    assert p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}
    assert current_images(transport.calls[0])==[obs['observation/image'],obs['observation/wrist_image']]
    assert current_record(transport.calls[0])['observation/state']==obs['observation/state']
    assert instance.prompt_version==VERSION+'-hybrid'
    assert instance.history_stats()['protocol']==HISTORY_VERSION
    assert instance.budget_snapshot()['api_key_calls']==0


def test_candidate_board_binding_and_exact_local_coordinates_remain_inherited():
    raw=prediction(candidate=True);instance,transport=client(raw);obs=observation().to_dict();calls=[]
    instance.reference_candidate_resolver=resolver_for(obs,calls)
    instance.reference_candidate_provider=lambda current: obs['observation/wrist_image']if current==obs else None
    instance.observable_context={'visual_feedback':{'reference_repair':{'review_required':True}}}
    result=instance.predict(obs)
    assert result['reference_candidate_resolution']==candidate_result()
    assert result['raw_wire_prediction']==raw
    assert result['visual_references']['external']['seed']['moving_point_uv']==candidate_result()['current_point_uv']
    assert raw['visual_references']['external']['seed']['moving_point_uv']is None
    assert len(calls)==1 and instance._candidate_observation is None
    assert CANDIDATE_LABEL in json.dumps(transport.calls[0])
    assert len(current_images(transport.calls[0]))==3


@pytest.mark.parametrize('violation',['negative_uncertainty','missing_inventory','unknown_duplicate','near_contact_no_target'])
def test_guidance_does_not_relax_inherited_geometry_and_target_contract(violation):
    raw=prediction(controller='visual_field')
    if violation=='negative_uncertainty':raw['scene_inventory'][0]['geometry']['uncertainty_m']=-.01
    elif violation=='missing_inventory':raw.pop('scene_inventory')
    elif violation=='unknown_duplicate':
        raw['field_intent']['unknown_regions']=[{'id':raw['scene_inventory'][0]['id'],
            'center_offset_m':[.1,.1,.1],'half_extents_m':[.01]*3,'uncertainty_m':.01}]
    else:raw['scene_inventory']=[]
    instance,_=client()
    with pytest.raises(SchemaError):instance.validate_response(raw,None)


def test_bounded_semantic_and_stream_repair_same_observation_no_extra_model_route():
    invalid=prediction();invalid['direct_actions']=None
    good=prediction(phase='recover');good['direct_actions']=[[0.]*7 for _ in range(5)]
    instance,transport=client(stream_error(),invalid,good)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==good
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(current_images(p)==current_images(transport.calls[0])for p in transport.calls)


def test_margin_guidance_preserves_task_generic_progress_and_all_retained_checks():
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    for phrase in ['NOT a calibrated statistical standard deviation','full-uncertainty',
                   'never physical clearance certification','never shrink, delete or rename',
                   'shared/directional travel allowance','persistent Field goal',
                   'genuine near, unknown or unsupported','exactly FIVE','NEVER hand/palm',
                   'review even if reference_health still says weakly_observable']:
        assert phrase in text
    assert '66.9'not in text and '73.9'not in text
    semantics=geometry_semantics();semantics['clearance_certified']=True
    assert geometry_semantics()['clearance_certified']is False
    assert 'standard deviation' in geometry_semantics()['uncertainty_margin_semantics']
