import copy
import json
from pathlib import Path

from gpt_field.visual_feedback_compact_v14 import compact_feedback, compact_payload_feedback
from gpt_field.visual_field_subscription_v14 import DIAGNOSTIC_LABEL
from test_visual_field_v14_protocol import client, prediction
from test_visual_field_v7_protocol import current_record
from test_visual_field_protocol import observation


def context():
    return {'visual_feedback': {'visual': {'reference_health': {'repair_required':False},
        'tracking': {'moving': {'tracked_point_uv':[.4,.3], 'reliable':True},
                     'target': {'tracked_point_uv':[.7,.6], 'reliable':True}}},
        'reference_support': {'status':'reference_support_stasis','diagnostic_required':True,'interpretation':'inconclusive'},
        'recovery_options': {'shared_remaining_m':.004,'actual_segment_must_be_rechecked':True},
        'translation_guard': {'recovery_experiments':{'remaining_m':.004},'conflicting_obstacle_ids':['neighbor'],
            'last_admission':{'accepted':False,'reason':'budget_exhausted'}}}}


def test_diagnostic_adds_one_claim_label_without_changing_false_repair_flag_or_original_views():
    instance, transport=client(prediction())
    ctx=context();original=copy.deepcopy(ctx);instance.observable_context=ctx
    instance.predict(observation().to_dict())
    payload=transport.calls[0]
    assert ctx==original
    record=current_record(payload)
    assert record['observable_context']['visual_feedback']['visual']['reference_health']['repair_required'] is False
    parts=payload['input'][-1]['content'];images=[p['image_url']for p in parts if p['type']=='input_image']
    obs=observation().to_dict()
    assert images[:2]==[obs['observation/image'],obs['observation/wrist_image']] and len(images)==3
    assert sum(p.get('text')==DIAGNOSTIC_LABEL for p in parts)==1
    assert 'not verified object identity' in DIAGNOSTIC_LABEL
    assert record['observable_context']['visual_feedback']['reference_support']==original['visual_feedback']['reference_support']


def test_healthy_diagnostic_flag_does_not_add_an_image():
    instance,transport=client(prediction());ctx=context();ctx['visual_feedback']['reference_support']['diagnostic_required']=False
    instance.observable_context=ctx;instance.predict(observation().to_dict())
    assert sum(p.get('type')=='input_image'for m in transport.calls[0]['input']for p in m['content'])==2


def test_current_authoritative_facts_and_unknown_warning_survive_presentation_compaction():
    fb=context()['visual_feedback']
    fb.update(target_contact_geometry={'target_id':'obj','nominal':{'lower_xyz':[0,0,0],'upper_xyz':[1,1,1]},'effective_allowed_contact':'fingers'},
        contact_alignment={'active':True,'image_error_px':[8,0,1,2],'status':'known_mismatch'},
        subgoal_ledger={'active':{'goal_xyz':[.2,.3,.4],'id':'goal'},'local_complete':False},
        last_target_admission={'accepted':False,'reason':'palm_contact'},
        unknown_new_hazard={'reason':'must_survive','limit_m':.001})
    fb['visual']['attachment']={'state':'temporarily_uncertain','remaining_m':.002}
    fb['field']={'goal_xyz':[.2,.3,.4], 'target_contact_geometry':copy.deepcopy(fb['target_contact_geometry']),
        'region_feasibility':[{'id':'obj','nominal':{'lower_xyz':[0,0,0],'upper_xyz':[1,1,1]},'tool_core':{'contains_current':False,'current_distance_to_union_m':.002}}],
        'translation_admission':{'accepted':False,'reason':'budget_exhausted','cap_m':0,'diagnostics':{'observation_travel_remaining_m':0,'neighbors':[{'id':'neighbor','box_px':[1,2,3,4],'gap_px':.1,'margin_px':2}], 'unrecognized_warning':'keep'}}}
    before=copy.deepcopy(fb);packed=compact_feedback(fb)
    for key in ('translation_guard','reference_support','recovery_options','visual','target_contact_geometry','contact_alignment','subgoal_ledger','last_target_admission','unknown_new_hazard'):
        assert packed[key]==fb[key]
    assert packed['field']['region_feasibility']==fb['field']['region_feasibility']
    assert packed['field']['translation_admission']['reason']=='budget_exhausted'
    assert packed['field']['translation_admission']['diagnostics']['unrecognized_warning']=='keep'
    assert packed['field']['translation_admission']['diagnostics']['observation_travel_remaining_m']==0
    assert fb==before and compact_feedback(packed)==packed


def test_saved_v13_request_compaction_is_effective_idempotent_and_leaves_images_outputs_and_state_exact():
    directory=Path(__file__).resolve().parents[1]/'runs/visual_field_v13_zero_shot_002/trials/bowl_original_zero_hybrid_v13/api'
    sources=sorted(directory.glob('*.request.json'))
    assert sources
    payload=json.loads(sources[-1].read_text())['payload'];before=copy.deepcopy(payload)
    result=compact_payload_feedback(payload)
    assert result==compact_payload_feedback(result) and payload==before
    for old,new in zip(payload['input'],result['input']):
        if old.get('role')!='user':assert old==new
        for a,b in zip(old.get('content',[]),new.get('content',[])):
            if a['type']!='input_text':assert a==b
            elif a['text'].startswith('{'):
                x,y=json.loads(a['text']),json.loads(b['text'])
                for key in ('observation/state','observation/joint_position','observation/joint_velocity','prompt'):
                    if key in x:assert x[key]==y[key]
                if x.get('type')=='observation':
                    xf=x['observable_context']['visual_feedback'];yf=y['observable_context']['visual_feedback']
                    for key in ('translation_guard','visual','target_contact_geometry','contact_alignment','subgoal_ledger'):
                        if key=='contact_alignment':
                            assert xf[key]['status']==yf[key]['status'] and xf[key]['active']==yf[key]['active']
                        else: assert xf[key]==yf[key]
    length=lambda p:sum(len(c.get('text',''))for m in p['input']for c in m.get('content',[]))
    assert length(result)<.92*length(payload)
