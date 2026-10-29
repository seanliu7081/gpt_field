"""Seven saved real requests plus synthetic extension/partial-support preservation."""
import copy
import json
from pathlib import Path

import pytest

from gpt_field.visual_feedback_compact_v17 import compact_feedback, compact_payload_feedback, restore_feedback_delta

ARCHIVE=Path(__file__).resolve().parents[1]/'runs/visual_field_v16_zero_shot_002/trials/bowl_original_zero_hybrid_v16/api'


def feedbacks(payload):
    found=[]
    def visit(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('visual_feedback','end_visual_feedback'):found.append(v)
                else:visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for message in payload.get('input',[]):
        if message.get('role')!='user':continue
        for part in message.get('content',[]):
            if part.get('type')!='input_text':continue
            try:visit(json.loads(part['text']))
            except ValueError:pass
    bases={}
    for message in payload.get('input',[]):
        for part in message.get('content',[]):
            if part.get('type')!='input_text':continue
            try:record=json.loads(part['text'])
            except ValueError:continue
            if isinstance(record,dict)and record.get('type')=='observation':
                f=(record.get('observable_context')or{}).get('visual_feedback')
                if isinstance(f,dict)and 'feedback_ref'not in f:bases[record['observation_index']]=f
    return [restore_feedback_delta(v,bases[v['feedback_ref']['observation_index']])
            if isinstance(v,dict)and 'feedback_patch'in v else v for v in found]


def expand(value,root):
    if isinstance(value,list):return [expand(v,root)for v in value]
    if not isinstance(value,dict):return copy.deepcopy(value)
    ref=value.get('feedback_ref')
    if isinstance(ref,dict) and 'keys' in ref:
        original=root
        for part in ref['path'].split('.'):original=original[part]
        result={key:copy.deepcopy(original[key])for key in ref['keys']}
        result.update(copy.deepcopy(value['overrides']))
        return expand(result,root)
    return {k:expand(v,root)for k,v in value.items()}


@pytest.mark.parametrize('number',range(1,8))
def test_each_saved_request_retains_actionable_feedback_and_exact_raw_inputs(number):
    files=sorted(ARCHIVE.glob(f'{number:06d}-*.request.json'))
    if not files:pytest.skip('Saved V16 archive not present')
    payload=json.loads(files[0].read_text())['payload'];original=copy.deepcopy(payload)
    packed=compact_payload_feedback(payload)
    assert payload==original
    for old,new in zip(payload['input'],packed['input']):
        if old.get('role')!='user':assert old==new
        else:
            assert [p for p in old.get('content',[])if p.get('type')=='input_image']==[p for p in new.get('content',[])if p.get('type')=='input_image']
            for a,b in zip(old.get('content',[]),new.get('content',[])):
                if a.get('type')!='input_text':continue
                try:oa,ob=json.loads(a['text']),json.loads(b['text'])
                except ValueError:continue
                if isinstance(oa,dict) and oa.get('type')=='observation':
                    assert {k:v for k,v in oa.items()if k!='observable_context'}=={k:v for k,v in ob.items()if k!='observable_context'}
                    for e in (oa.get('observable_context',{}).get('visual_feedback',{}).get('scene_inventory',{})or{}).get('entries',[]):
                        current,world=e.get('current_geometry'),e.get('world_geometry')
                        if current is not None and world is not None:
                            assert [x+y for x,y in zip(current['center_offset_m'],oa['observation/state'][:3])]==pytest.approx(world['center_xyz_m'],abs=1e-14)
                            assert current['half_extents_m']==world['half_extents_m'] and current['uncertainty_m']==world['uncertainty_m']
    for old,new in zip(feedbacks(payload),feedbacks(packed)):
        if not isinstance(old,dict)or'feedback_ref'in old:assert old==new;continue
        restored=expand(new,new)
        for key in old:
            if key not in ('scene_inventory','feedback_representation'):
                assert restored[key]==old[key],key
        before=(old.get('scene_inventory')or{}).get('entries',[])
        after=(new.get('scene_inventory')or{}).get('entries',[])
        assert [e['id']for e in before]==[e['id']for e in after]
        for a,b in zip(before,after):
            for key in a:
                if key not in ('world_geometry','geometry_revisions'):assert a[key]==b[key],key
            assert a.get('current_geometry')==b.get('current_geometry')
    assert compact_payload_feedback(packed)==packed


def test_partial_status_unknown_extensions_warnings_and_changed_snapshot_not_erased():
    geometry={'center_offset_m':[.2,.1,.3],'half_extents_m':[.03]*3,'uncertainty_m':.04}
    world={'center_xyz_m':[.4,.5,.7],'half_extents_m':[.03]*3,'uncertainty_m':.04,'future_metric_warning':'keep'}
    view={'current_roi_uv':None,'projected_roi_uv':[-.2,.1,.3,.4],'partial_current_support':True,
          'current_visible_roi_uv':[0.,.1,.3,.4],'support_status':'current_partial_texture',
          'support_points_px':[[2,33],[4,44]],'support_inlier_count':2,'support_observed_at':9,'future_support_warning':'keep'}
    f={'scene_inventory':{'entries':[{'id':'all_ids_stay','current_geometry':geometry,'world_geometry':world,
       'geometry_revision_count':2,'geometry_revisions':[{'update_index':1,'evidence':'old',
       'new_world_geometry':world,'future_revision_warning':'keep'}],'views':{'external':view}}]},
       'translation_guard':{'reason':'current_stop','remaining_m':.003,'large':[1]*100},
       'field':{'translation_admission':{'accepted':False,'reason':'old_stop',
       'diagnostics':{'reason':'prior_stop','remaining_m':.006,'large':[1]*100,'new_admission_warning':'keep'}}},
       'new_runtime_warning':{'reason':'must_not_drop','budget':.001}}
    original=copy.deepcopy(f);packed=compact_feedback(f);restored=expand(packed,packed)
    assert f==original and restored['field']==f['field']
    assert packed['new_runtime_warning']==f['new_runtime_warning']
    entry=packed['scene_inventory']['entries'][0]
    assert entry['views']['external']==view and entry['current_geometry']==geometry
    assert entry['world_geometry']=={'future_metric_warning':'keep'}
    assert entry['geometry_revision_extensions'][0]['future_revision_warning']=='keep'
    assert entry['geometry_revision_extensions'][0]['new_world_geometry']['future_metric_warning']=='keep'


def test_unknown_added_removed_keys_and_list_lengths_restore_exactly_without_cycles():
    old={'unknown_warning':{'value':1,'only_old':[1,2]},'list':[{'a':1},{'a':2}],
         'repeated_payload':{'text':'same'*1000}}
    new={'unknown_warning':{'value':2,'only_new':False},'list':[{'a':3}],
         'repeated_payload':{'text':'same'*1000}}
    def message(index,feedback):return {'role':'user','content':[{'type':'input_text','text':json.dumps(
        {'type':'observation','observation_index':index,'observation/state':[0]*8,
         'observable_context':{'visual_feedback':feedback}})}]}
    payload={'input':[message(11,old),message(12,new)]}
    packed=compact_payload_feedback(payload)
    a,b=[json.loads(m['content'][0]['text'])for m in packed['input']]
    prior=a['observable_context']['visual_feedback'];current=b['observable_context']['visual_feedback']
    assert prior['feedback_ref']['observation_index']==12 and 'feedback_ref'not in current
    assert restore_feedback_delta(prior,current)==compact_feedback(old)
    assert feedbacks(packed)==[compact_feedback(old),compact_feedback(new)]
    assert compact_payload_feedback(packed)==packed


def test_known_part_coordinates_only_are_omitted_unknown_nested_details_survive():
    part={'name':'palm','lower_xyz':[1,2,3],'upper_xyz':[4,5,6],
          'new_veto':{'reason':'keep_this','distance':.01}}
    value={'target_contact_geometry':{'core_parts':[part],'exempt_parts':['finger'],
        'union_contains_current':False,'union_distance_m':.04,'uncertainty_m':.05}}
    packed=compact_feedback(value)['target_contact_geometry']
    assert 'core_parts'not in packed
    assert packed['core_parts_summary']['parts']==[{'name':'palm','new_veto':part['new_veto']}]
    assert packed['exempt_parts']==['finger'] and packed['union_distance_m']==.04
