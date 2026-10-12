import copy
import json
import numpy as np
from gpt_field.distance_video import DistanceHybridVideoRecorder
from gpt_field.policy import Decision
from gpt_field.schema import GridSpec
from test_field_history_runner import observation
from test_hybrid_video import memory_writer


def test_distance_map_revision_raw_guidance_and_invalid_clear(tmp_path, observation, memory_writer):
    grid=GridSpec(size=3, spacing_m=.005)
    raw={'stage':'align','orientation_target_xyzw':[0,0,0,1],'gripper_command':'close','motion_mode':'move',
         'values':[17.]*grid.count,'blocked_indices':[],'unknown_indices':[]}
    display={**raw,'values':list(np.arange(grid.count,dtype=float)/100)}
    center=list(observation.state[:3])
    goal=center.copy(); goal[0]+=.005
    routing={'controller':'coarse_field','target_region':'plate','near_target':False,'routing_reason':'fixture'}
    details={'step':0,'controller':'coarse_field','routing':routing,'refreshed':True,
             'field_center':center,'field_prediction':raw,'distance_field_display':display,
             'plan_refreshed':True,'status':'move','plan':{'status':'move','candidate_count':0,'valid_count':0,
             'candidates':[],'selected_index':None,'search':{'method':'dijkstra_distance_field','cache_hit':False,
             'goal_xyz':goal,'cost_unit':'m'}}}
    recorder=DistanceHybridVideoRecorder(tmp_path/'fixture.mp4',grid=grid,offline_smoke=True)
    recorder.observe(observation)
    recorder.decision(observation,Decision((0,0,0,0,0,0,1),details))
    assert recorder.field.prediction.values==tuple(display['values'])
    assert details['field_prediction']['values']==[17.]*grid.count
    revised=copy.deepcopy(details)
    revised.update(step=1,refreshed=False,local_field_updated=True,local_field_revision=1)
    revised['distance_field_display']['values']=[.2]*grid.count
    recorder.decision(observation,Decision((0,0,0,0,0,0,1),revised))
    assert recorder.field.prediction.values==(.2,)*grid.count
    assert recorder.field_step==1
    cached=copy.deepcopy(revised);cached.pop('distance_field_display')
    cached.update(step=2,local_field_updated=False)
    cached['plan']['search']['cache_hit']=True
    recorder.decision(observation,Decision((0,0,0,0,0,0,1),cached))
    assert recorder.field.prediction.values==(.2,)*grid.count
    invalid=copy.deepcopy(cached)
    invalid.update(step=3,refreshed=True)
    invalid['plan']['status']='needs_observation'
    invalid['plan']['search']['cache_hit']=False
    recorder.decision(observation,Decision((0,0,0,0,0,0,1),invalid))
    assert recorder.field is None
    recorder.close(status='fixture_complete')
    records=[json.loads(line) for line in (tmp_path/'fixture.events.jsonl').read_text().splitlines()]
    fresh=[row for row in records if row['event']=='field_refresh']
    assert fresh[0]['field_prediction']==raw
    assert fresh[0]['distance_field_display']==display
    assert fresh[0]['field_display_source']=='local_dijkstra_cost_to_go'
    assert fresh[0]['local_metadata_scope']=='pre_action_decision'
    assert fresh[-1]['field_display_source']=='unavailable'
    local=[row for row in records if row['event']=='local_replan' and row['step']==1][0]
    assert local['local_field_revision']==1
    assert local['distance_field_display']['values']==[.2]*grid.count
    assert 'field_prediction' not in local
