"""V24 changes only guidance and older engineering-feedback presentation."""
import copy
import json
from pathlib import Path
import pytest

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v23 import output_schema as schema_v23
from gpt_field.visual_field_subscription_v24 import (
    VisualFieldSubscriptionClientV24, output_schema, shared_guidance, geometry_semantics,
    compact_older_contexts, VERSION, HISTORY_VERSION, FEEDBACK_VERSION,
)
from gpt_field.visual_feedback_compact_v21 import expanded_observation_feedback
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v18_protocol import prediction, candidate_result, resolver_for
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport=FakeTransport(*(r if isinstance(r,BaseException)else envelope(
        r,model='gpt-6-astra',reasoning={'effort':'medium'},id=f'v24_{i}')
        for i,r in enumerate(responses)))
    return VisualFieldSubscriptionClientV24(token_provider=lambda:'offline-oauth',transport=transport),transport


def test_unchanged_wire_and_honest_bounded_exposure_guidance():
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    assert output_schema()==schema_v23()
    for phrase in ('NOT observed clearance', 'absence of worsening is NOT evidence of free space',
                   'geometry=null', 'subgoal_update=retain', 'grants no extra motion',
                   'distal rigid gripper/finger', 'full original records remain local'):
        assert phrase in text
    assert 'further travel needs fresh rolling favorable image evidence'not in text
    assert 'cabinet'not in text.lower()
    e=geometry_semantics()['initial_uncertainty_egress']
    assert e['bounded_model_geometry_exposure']and e['all_scene_hypotheses_retained']
    for key in ('fresh_favorable_image_evidence_required_after_probe','absence_of_worsening_certifies_clearance',
                'contextual_budget_reset_on_rename_or_reseed','clearance_certified','unknown_region_exemption'):
        assert e[key]is False
    with pytest.raises(ValueError): shared_guidance(resolve_distance_task('bowl_plate'),True)


def test_actual_client_preserves_latest_warnings_unknown_keys_candidate_and_raw_history():
    raw=prediction(candidate=True);ordinary=prediction();obs=observation().to_dict()
    instance,transport=client(raw,ordinary);calls=[]
    instance.reference_candidate_resolver=resolver_for(obs,calls)
    old={'refresh_reason':'old_actual_refusal','visual_feedback':{'step':3,'action':[0.]*7,
        'measured_translation_m':.004,'field':{'stop_reason':'old_budget','giant_old_samples':[1]*1000},
        'reference_repair':{'review_required':True},'old_unknown_extension':{'raw':'preserve-locally'}}}
    instance.observable_context=copy.deepcopy(old)
    first=instance.predict(obs)
    assert first['reference_candidate_resolution']==candidate_result()
    assert first['raw_wire_prediction']==raw and raw['visual_references']['external']['seed']['moving_point_uv']is None
    history=copy.deepcopy(instance._history)
    current={'refresh_reason':'current_refusal','visual_feedback':{
        'field':{'stop_reason':'active_obstacle','remaining_m':.003},
        'scene_inventory':{'entries':[{'id':'held-id','current_geometry':{'center_offset_m':[.1,.2,.3]},'uncertainty_m':.02}]},
        'unknown_current_extension':{'exact':[.001234567890123,False,None]},
        'reference_repair':{'review_required':True,'observation_binding':'current-only'}}}
    instance.observable_context=copy.deepcopy(current)
    instance.predict(observation(1).to_dict())
    payload=transport.calls[-1]
    assert current_record(payload)['observable_context']['visual_feedback']['unknown_current_extension']==current['visual_feedback']['unknown_current_extension']
    assert current_record(payload)['observable_context']['visual_feedback']['field']['stop_reason']=='active_obstacle'
    snapshots=expanded_observation_feedback(payload)
    old_summary=snapshots[0][2]
    assert old_summary['type']=='historical_action_result_summary'
    assert old_summary['field']['stop_reason']=='old_budget'
    assert 'giant_old_samples'not in old_summary['field']
    assert instance._history[:len(history)]==history
    assert old=={'refresh_reason':'old_actual_refusal','visual_feedback':{'step':3,'action':[0.]*7,
        'measured_translation_m':.004,'field':{'stop_reason':'old_budget','giant_old_samples':[1]*1000},
        'reference_repair':{'review_required':True},'old_unknown_extension':{'raw':'preserve-locally'}}}
    assistants=[json.loads(p['text'])for m in payload['input']if m.get('role')=='assistant'
        for p in m.get('content',[])if p['type']=='output_text']
    assert assistants==[raw]
    assert current_images(transport.calls[0])[:2]==[obs['observation/image'],obs['observation/wrist_image']]
    assert payload['text']['format']['name']=='visual_field_rgb_joints_v24'
    assert instance.prompt_version==VERSION+'-hybrid'
    assert instance.history_stats()['protocol']==HISTORY_VERSION
    assert instance.budget_snapshot()['api_key_calls']==0


@pytest.mark.parametrize('order',['stream_first','schema_first'])
def test_same_observation_retries_remain_bounded_medium_subscription_only(order):
    invalid=prediction();invalid['direct_actions']=None
    failures=[stream_error(),invalid]if order=='stream_first'else[invalid,stream_error()]
    valid=prediction(phase='recover');valid['direct_actions']=[[0.]*7 for _ in range(5)]
    instance,transport=client(*failures,valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==valid
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}for p in transport.calls)
    assert all(current_images(p)==current_images(transport.calls[0])for p in transport.calls)


def test_all_archived_v22_requests_keep_latest_exact_and_rgb_state_joints_assistants_unchanged():
    api=Path(__file__).resolve().parents[1]/'runs/visual_field_v22_zero_shot_001/trials/bowl_original_zero_hybrid_v22/api'
    paths=sorted(api.glob('*.request.json'));assert len(paths)==7
    smaller=0
    for path in paths:
        raw_bytes=path.read_bytes();before=json.loads(raw_bytes)['payload'];original=copy.deepcopy(before)
        after=compact_older_contexts(before)
        old,new=expanded_observation_feedback(before),expanded_observation_feedback(after)
        assert len(old)==len(new)
        assert old[-1][0]['text']==new[-1][0]['text']
        for index,((_,a,fa),(_,b,fb))in enumerate(zip(old,new)):
            ca=a.pop('observable_context');cb=b.pop('observable_context')
            assert a==b # exact observation pixels are separate parts; state/q/index/prompt unchanged
            if index==len(old)-1:assert ca==cb
            else:
                assert fb['type']=='historical_action_result_summary'and fb['version']==FEEDBACK_VERSION
                ca.pop('visual_feedback');cb.pop('visual_feedback');assert ca==cb
        for a,b in zip(before['input'],after['input']):
            if a.get('role')!='user':assert a==b
            for pa,pb in zip(a.get('content',[]),b.get('content',[])):
                if pa.get('type')!='input_text':assert pa==pb
        assert before==original and path.read_bytes()==raw_bytes
        smaller+=len(json.dumps(after))<len(json.dumps(before))
    assert smaller==6
