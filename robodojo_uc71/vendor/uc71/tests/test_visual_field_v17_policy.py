"""Whole-inventory Field integration with causal RGB motion; no inference."""
import copy
import numpy as np
import pytest

from gpt_field.visual_field_policy_v17 import VisualFieldPolicyV17
from gpt_field.visual_field_v16 import InventoryTargetGeometryV16
from gpt_field.visual_field_v17 import MetricFieldExecutorV17
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import prediction as aligned_prediction, observation, transition


def prediction():
    p = aligned_prediction()
    g = copy.deepcopy(p['target_geometry'])
    p['target_contact'] = g.pop('allowed_contact')
    identity = g.pop('id')
    p['scene_inventory'] = [{'id':identity,
        'external_roi_uv':p['visual_references']['external']['seed']['target_roi_uv'],
        'wrist_roi_uv':p['visual_references']['wrist']['seed']['target_roi_uv'], 'geometry':g}]
    p['raw_wire_prediction'] = {'offline_fixture':True,'scene_inventory':copy.deepcopy(p['scene_inventory'])}
    return p


def test_current_inventory_allows_real_field_reuse_and_contact_refinement_without_extra_calls():
    p = prediction();client = Client(p);policy = VisualFieldPolicyV17(client);obs = observation()
    accepted = 0
    for _ in range(30):
        obs, decision, row = transition(policy,obs)
        accepted += bool(decision.diagnostics['selection']['accepted'])
        assert policy.pending is None and policy.local.pending is None
        assert policy.prediction == p and client.value == p
        if policy.refresh_reason:break
    assert accepted > 10 and client.calls == 1
    assert policy.visual_corrections > 0
    assert decision.diagnostics['policy_version'] == 'visual-field-policy-v17'
    assert [x['id'] for x in policy.local.region_geometry] == ['part']
    assert any(x['id'] == 'part' for x in policy.translation_guard.neighbors)
    assert len(row['scene_inventory']['entries']) == 1
    assert not row['task_success_certified']


def test_omitted_distractor_keeps_world_position_and_both_constraint_consumers():
    p = prediction();distractor = copy.deepcopy(p['scene_inventory'][0]);distractor['id']='different_object'
    distractor['geometry']['center_offset_m']=[.35,.35,.35]
    # Separate visible patch so identity is independent of the primary target.
    distractor['external_roi_uv']=[.02,.02,.15,.15]
    p['scene_inventory'].append(distractor)
    client=Client(p);policy=VisualFieldPolicyV17(client);obs=observation()
    obs,d,row=transition(policy,obs)
    old=copy.deepcopy(policy.inventory.summary()['entries'][1]['world_geometry'])
    kept=copy.deepcopy(p);kept['scene_inventory']=[]
    kept['subgoal_update']['action']='retain'
    kept['visual_references']['external'].update(update='retain',seed=None)
    kept['visual_references']['wrist'].update(update='retain',seed=None)
    kept['contact_alignment'].update(update='retain',external=None,wrist=None)
    client.value=kept;policy.refresh_reason='offline_refresh'
    d=policy.act(obs)
    assert policy.inventory.summary()['entries'][1]['world_geometry']==old
    assert 'different_object' in policy.local.inventory_applied_ids
    assert 'different_object' in [n['id'] for n in policy.translation_guard.neighbors]
    assert d.diagnostics['scene_materialization']['objects_deleted'] is False


def test_executor_keeps_more_than_eight_objects_and_large_rebased_coordinates():
    p=prediction();obs=observation();executor=MetricFieldExecutorV17()
    objects=[];obstacles=[]
    for index in range(12):
        identity=f'fixture_{index}'
        center=np.asarray(obs.state[:3])+[1.2+index*.05,.1,0.]
        objects.append({'id':identity,'world_geometry':{'center_xyz_m':center.tolist(),
            'half_extents_m':[.01]*3,'uncertainty_m':.02},
            'views':{'external':{'current_roi_uv':[.1,.1,.2,.2]}}})
        obstacles.append({'id':identity,'lower_offset_m':(center-np.asarray(obs.state[:3])-.01).tolist(),
            'upper_offset_m':(center-np.asarray(obs.state[:3])+.01).tolist(),'uncertainty_m':.02})
    executor.configure(obs.state,p['field_intent'],scene_assessment=p['motion_context'],
        inventory_materialization={'objects':objects,'obstacles':obstacles})
    assert len(executor.region_geometry)==12 and len(executor.inventory_applied_ids)==12
    assert executor.region_geometry[0]['nominal']['lower_xyz'][0]==pytest.approx(objects[0]['world_geometry']['center_xyz_m'][0]-.01)
    assert executor.checked.segment_valid(np.asarray(obs.state[:3]), np.asarray(obs.state[:3])+[.005,0.,0.])


def test_internal_rebased_target_keeps_exact_world_geometry_and_palm_constraints():
    target=InventoryTargetGeometryV16();obs=observation()
    geometry={'id':'far_target','center_offset_m':[.71,-.82,.93],
        'half_extents_m':[.01]*3,'uncertainty_m':.03,'allowed_contact':'fingers'}
    target.configure(obs.state,geometry,target_id='far_target')
    info=target.summary()
    expected=np.asarray(obs.state[:3])+geometry['center_offset_m']-.01
    assert info['nominal']['lower_xyz']==pytest.approx(expected)
    assert any(p['tool_part']=='hand_collision' for p in info['core_parts'])
    assert 'finger1_collision' in info['exempt_parts']


def test_unlocalized_neighbor_near_tool_blocks_gripper_close():
    from test_visual_field_v13_payload_policy import prediction as grip_prediction
    from test_visual_field_v6_policy import observation as grip_observation
    p=grip_prediction();g=copy.deepcopy(p['target_geometry']);permission=g.pop('allowed_contact');identity=g.pop('id')
    p.update(target_contact=permission,scene_inventory=[{'id':identity,'external_roi_uv':[.68,.61,.8,.74],
        'wrist_roi_uv':None,'geometry':g},{'id':'unlocalized','external_roi_uv':[.24,.28,.31,.38],
        'wrist_roi_uv':None,'geometry':None}],raw_wire_prediction={'offline_fixture':True})
    policy=VisualFieldPolicyV17(Client(p));obs=grip_observation()
    d=policy.act(obs)
    assert not d.diagnostics['selection']['accepted']
    assert d.action[6] == 0.
    assert d.diagnostics['selection']['status']=='scene_visible_region_needs_metric_localization'
    assert policy.gripper.pending is None


def test_partial_extent_plans_with_full_uncertainty_without_changing_inventory():
    p=prediction();obs=observation();executor=MetricFieldExecutorV17()
    center=np.asarray(obs.state[:3])+[.25,.25,.25]
    material={'objects':[{'id':'edge_fixture',
        'world_geometry':{'center_xyz_m':center.tolist(),'half_extents_m':[.02]*3,'uncertainty_m':.04},
        'views':{'external':{'current_roi_uv':[.8,.2,1.,.6],
            'current_visible_roi_uv':[.8,.2,1.,.6], 'extent_visibility':'boundary_limited',
            'partial_current_support':True,'projected_roi_uv':[.8,.2,1.1,.6]}}}],
        'obstacles':[{'id':'edge_fixture','lower_offset_m':[.23]*3,
            'upper_offset_m':[.27]*3,'uncertainty_m':.04}]}
    original=copy.deepcopy(material)
    executor.configure(obs.state,p['field_intent'],scene_assessment=p['motion_context'],
        inventory_materialization=material)
    assert material == original
    region=executor.region_geometry[0]
    assert region['geometry_mode']=='persistent_parts_hard_unobserved_uncertainty'
    assert executor.summary()['partial_visibility_hard_region_ids']==['edge_fixture']
    assert len(executor.boxes)==len(region['full_halo_parts'])
    for (lo,hi),part in zip(executor.boxes,region['full_halo_parts']):
        assert lo==pytest.approx(part['lower_xyz'])
        assert hi==pytest.approx(part['upper_xyz'])
    assert executor.checked.segment_valid(np.asarray(obs.state[:3]),np.asarray(obs.state[:3])+[.005,0,0])


def test_partial_primary_target_has_same_hard_coverage_as_candidate_admission():
    from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17
    p=prediction();obs=observation();executor=MetricFieldExecutorV17()
    contact=InventoryTargetGeometryV16()
    contact.configure(obs.state,p['target_geometry'],target_id=p['target_id'])
    executor.contact_geometry=contact
    geometry=p['scene_inventory'][0]['geometry']
    target={'id':p['target_id'],'world_geometry':{
        'center_xyz_m':(np.asarray(obs.state[:3])+geometry['center_offset_m']).tolist(),
        'half_extents_m':geometry['half_extents_m'],'uncertainty_m':geometry['uncertainty_m']},
        'views':{'external':{'current_roi_uv':[.8,.2,1.,.6],
            'extent_visibility':'boundary_limited','partial_current_support':True}}}
    material={'objects':[target],'obstacles':[]};original=copy.deepcopy(material)
    executor.configure(obs.state,p['field_intent'],scene_assessment=p['motion_context'],
        inventory_materialization=material)
    expected=SceneCoverageGuardV17._sweep_boxes(target,np.asarray(obs.state),
        p,executor.MAX_DRIFT_RAD)
    actual=executor.summary()['partial_target_coverage_parts']
    assert len(actual)==len(expected)>1 and material==original
    for first,second in zip(actual,expected):
        assert first['lower_xyz']==pytest.approx(second['lower_xyz'])
        assert first['upper_xyz']==pytest.approx(second['upper_xyz'])
    assert contact.summary()['effective_allowed_contact']==p['target_contact']
    assert p['target_id'] not in [r['id'] for r in executor.risk_regions]

