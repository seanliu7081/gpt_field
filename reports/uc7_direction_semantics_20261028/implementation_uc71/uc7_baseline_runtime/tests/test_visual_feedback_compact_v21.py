import copy
import json
from pathlib import Path

import pytest
from gpt_field.astra import _canonical
from gpt_field.visual_feedback_compact_v17 import _deduplicate_older_feedback
from gpt_field.visual_feedback_compact_v21 import (
    compact_feedback, restore_feedback, compact_payload_feedback,
    expanded_observation_feedback,
)


def feedback():
    entries=[]
    for i in range(8):
        views={}
        for view in ('external','wrist'):
            views[view]={'current_roi_uv':[.1,.2,.3,.4] if i%2 else None,
                'current_visible_roi_uv':[.1,.2,.3,.4], 'projected_roi_uv':[-.1,.2,.3,1.1],
                'last_known_roi_uv':[.05,.15,.4,.5], 'extent_visibility':'boundary_limited',
                'partial_current_support':True, 'reliable':bool(i%2),
                'status':'current_affine_tracking_hypothesis',
                'support_status':'current_visible_support_hypothesis',
                'support_is_clearance_certificate':False,'support_is_complete_extent_observation':False,
                'support_observed_at':3+i,'boundary_quantization':None,
                'future_unknown_warning':{'must_keep':i,'binding':'causal-current-image'},
                'future_optional':None}
            if i==2:views[view].pop('future_optional')
        entries.append({'id':f'object_{i}', 'current_geometry':{
            'center_offset_m':[.9+i*.1,-.2,.3],'half_extents_m':[.1,.2,.3],'uncertainty_m':.07},
            'views':views})
    return {'scene_inventory':{'entries':entries},
        'task_progress':{'requires_goal_review':True,'reason':'metric_better_image_worse'},
        'visual':{'attachment':{'state':'unobserved','transport_command_cap_m':0.},
                  'reference_health':{'unvalidated_travel_remaining_m':0.,'repair_required':True}},
        'translation_guard':{'directional_retreat':{'remaining_m':.002,'reason':'worsening'},
                             'new_future_key':{'preserve':True}},
        'reference_repair':{'observation_binding':'exact-binding','candidates':[{'candidate_id':'A','current_point_uv':[.4,.5]}]}}


def test_known_table_exactly_preserves_current_geometry_flags_unknown_keys_and_bindings():
    original=feedback();before=copy.deepcopy(original);packed=compact_feedback(original)
    assert 'view_table_v21' in packed['scene_inventory']
    assert restore_feedback(packed)==original==before
    assert len(_canonical(packed))<len(_canonical(original))
    for a,b in zip(original['scene_inventory']['entries'],packed['scene_inventory']['entries']):
        assert b['current_geometry']==a['current_geometry']
        assert b['id']==a['id']
    for key in ('task_progress','visual','translation_guard','reference_repair'):
        assert packed[key]==original[key]
    assert compact_feedback(packed)==packed


def payload_for(old,current):
    return {'input':[
        {'role':'user','content':[{'type':'input_text','text':_canonical({'type':'observation','observation_index':2,
            'observation/state':[1,2,3],'observation/joint_position':[4],
            'observable_context':{'visual_feedback':old}})}, {'type':'input_image','image_url':'exact-old-rgb'}]},
        {'role':'assistant','content':[{'type':'output_text','text':'{"raw_wire_prediction":"untouched"}'}]},
        {'role':'user','content':[{'type':'input_text','text':_canonical({'type':'observation','observation_index':3,
            'observation/state':[7,8,9],'observation/joint_position':[10],
            'observable_context':{'visual_feedback':current,'semantic_repair':{'error':'retain this error'}}})},
            {'type':'input_image','image_url':'exact-current-rgb'}]}]}


def test_decode_existing_older_patch_before_repacking_preserves_removed_and_changed_keys():
    old=feedback();new=copy.deepcopy(old);new['task_progress']['reason']='new_reason'
    new['visual']['reference_health']['unvalidated_travel_remaining_m']=.006
    old['old_unknown_field']={'must_survive':True}
    raw=payload_for(old,new);archived=_deduplicate_older_feedback(copy.deepcopy(raw));before=copy.deepcopy(archived)
    packed=compact_payload_feedback(archived)
    assert archived==before
    snapshots=expanded_observation_feedback(packed)
    assert restore_feedback(snapshots[0][2])==old
    assert restore_feedback(snapshots[1][2])==new
    assert packed['input'][1]==raw['input'][1]
    for a,b in zip(archived['input'],packed['input']):
        assert [c for c in a['content']if c['type']!='input_text']==[c for c in b['content']if c['type']!='input_text']
    assert 'feedback_ref' not in snapshots[-1][2]
    assert compact_payload_feedback(packed)==packed


def test_view_extension_identity_collision_is_not_overwritten():
    value=feedback();value['scene_inventory']['entries'][0]['views']['external']['object_id']='future extension'
    assert restore_feedback(compact_feedback(value))==value
    assert 'view_table_v21'not in compact_feedback(value)['scene_inventory']


def test_invalid_older_patch_does_not_read_another_request_or_guess_missing_base():
    p=payload_for(feedback(),feedback())
    record=json.loads(p['input'][0]['content'][0]['text'])
    record['observable_context']['visual_feedback']={'feedback_ref':{'observation_index':999,'path':'observable_context.visual_feedback'},'feedback_patch':[]}
    p['input'][0]['content'][0]['text']=json.dumps(record)
    with pytest.raises(ValueError,match='same-request base'):compact_payload_feedback(p)


ARCHIVE=Path(__file__).resolve().parents[1]/'runs/visual_field_v20_zero_shot_001/trials/bowl_original_zero_hybrid_v20/api'
@pytest.mark.parametrize('path',sorted(ARCHIVE.glob('*.request.json')),ids=lambda p:p.name[:6])
def test_all_real_v20_requests_restore_every_observation_and_keep_raw_messages(path):
    raw=path.read_bytes();payload=json.loads(raw)['payload'];before=copy.deepcopy(payload)
    packed=compact_payload_feedback(payload)
    old=expanded_observation_feedback(payload);new=expanded_observation_feedback(packed)
    assert len(old)==len(new)
    for (_,orecord,ofeedback),(_,nrecord,nfeedback)in zip(old,new):
        assert restore_feedback(nfeedback)==ofeedback
        orecord['observable_context'].pop('visual_feedback');nrecord['observable_context'].pop('visual_feedback')
        assert orecord==nrecord
    for a,b in zip(payload['input'],packed['input']):
        if a.get('role')!='user':assert a==b
        for c,d in zip(a.get('content',[]),b.get('content',[])):
            if c.get('type')!='input_text':assert c==d
    assert raw==path.read_bytes()and payload==before
