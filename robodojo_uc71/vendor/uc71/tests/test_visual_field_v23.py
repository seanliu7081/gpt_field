"""Real archived perception boundary exercises conditional Field egress."""
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from gpt_field.observation import PolicyObservation,decode_image
from gpt_field.visual_field_v23 import MetricFieldExecutorV23
from gpt_field.visual_field_v16 import InventoryTargetGeometryV16
from gpt_field.visual_scene_inventory_v17 import VisualSceneInventoryV17
from gpt_field.visual_scene_guard_v23 import SceneCoverageGuardV23
from gpt_field.visual_translation_guard_v23 import ProjectedTranslationGuardV23
from gpt_field.visual_monitor_v21 import RoleAwareVisualMonitorV21


def archive_field():
    root=Path(__file__).resolve().parents[1]/'runs/visual_field_v22_zero_shot_001/trials/bowl_original_zero_hybrid_v22'
    if not root.exists():pytest.skip('Optional V22 archive unavailable')
    obs=PolicyObservation.from_dict(json.loads((root/'initial_observation.json').read_text()))
    row=next(json.loads(l)for l in(root/'trace.jsonl').read_text().splitlines()if json.loads(l)['event']=='action')
    pred=copy.deepcopy(row['diagnostics']['prediction'])
    images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    monitor=RoleAwareVisualMonitorV21();monitor.configure(images,pred['visual_references'],obs.state,intent_id=1)
    inventory=VisualSceneInventoryV17();inventory.configure(obs,pred['scene_inventory'])
    material=inventory.materialize(obs.state,pred['target_id'],allowed_contact='none')
    context=copy.deepcopy(pred['motion_context']);context['nearby_obstacles']=material['nearby_obstacles']
    guard=ProjectedTranslationGuardV23();guard.configure(images,context,obs.state,target_id=pred['target_id'],monitor=monitor)
    guard.bind_egress_runtime(inventory,monitor)
    field=MetricFieldExecutorV23();field.bind_translation_guard(guard,monitor)
    contact=InventoryTargetGeometryV16();contact.configure(obs.state,material['target_geometry'],target_id=pred['target_id'],tool_radius_m=pred['field_intent']['tool_radius_m'])
    field.contact_geometry=contact
    field.configure(obs.state,pred['field_intent'],scene_assessment=context,inventory_materialization=material)
    material['applied_metric_ids']=[r['id']for r in field.region_geometry]
    material['applied_image_ids']=[r['id']for r in guard.neighbors]
    scene=SceneCoverageGuardV23()
    def assessor(prospective,state,delta):
        value=copy.deepcopy(pred);value.update(controller='visual_field',phase='recover',target_contact='none',field_intent=prospective.intent)
        value['motion_context']=dict(context,translation_mode='separate',purpose='reference_repair',intended_contact_id=None)
        return scene.assess(state,delta,materialization=material,field=prospective,monitor=monitor,prediction=value)
    field.scene_assessor=lambda state,delta:assessor(field,state,delta)
    return field,guard,monitor,inventory,obs,pred,assessor


def test_real_v22_blocked_initial_halo_has_actual_previewed_egress_without_debt_reset():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    assert not field.checked.segment_valid(obs.state[:3],obs.state[:3])
    candidates=field.egress_candidates(obs.state)
    assert candidates
    candidate=candidates[0]
    before=(field.__dict__,guard.__dict__,pickle.dumps(guard.egress_evidence.__dict__),pickle.dumps(field.contact_geometry.__dict__))
    selected=field.preview_egress(obs.state,obs.joint_position,candidate['goal_offset_m'],scene_assessor=assessor)
    assert selected['accepted'],selected['status']
    assert before[0] is field.__dict__ and before[1] is guard.__dict__
    assert before[2]==pickle.dumps(guard.egress_evidence.__dict__)
    assert before[3]==pickle.dumps(field.contact_geometry.__dict__)
    assert not field.checked.segment_valid(obs.state[:3],obs.state[:3])
    proof=selected['diagnostics']['egress_contract']
    assert proof['candidate_endpoint_full_halo_disjoint'] is False
    assert proof['exit_goal_full_halo_disjoint'] is True
    assert proof['goal_full_halo_clearance_m']>.003
    assert len(proof['scene_ids'])==8
    scene=selected['diagnostics']['scene_coverage_admission']
    assert scene['accepted'] and scene['diagnostics']['full_uncertainty_sweeps_not_reclassified']
    cabinet=next(x for x in scene['diagnostics']['sweep_checks']if x['id']=='drawer_cabinet')
    assert not cabinet['full_uncertainty_sweep_disjoint']
    assert guard.observation_travel_m==0 and guard.egress_evidence.spent_m==0


def test_actual_selection_matches_preview_and_rejects_wrong_acknowledgement():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    goal=field.egress_candidates(obs.state)[0]['goal_offset_m']
    preview=field.preview_egress(obs.state,obs.joint_position,goal,scene_assessor=assessor)
    intent=copy.deepcopy(field.intent);intent.update(goal_offset_m=goal,goal_tolerance_m=.003,command_limit_m=.003,max_travel_m=.04,max_actions=120)
    material=copy.deepcopy(field._transit_materialization)
    field.configure(obs.state,intent,scene_assessment=guard.context,inventory_materialization=material)
    field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    guard.context.update(translation_mode='separate',purpose='reference_repair',intended_contact_id=None)
    chosen=field.select(obs.state,obs.joint_position)
    assert chosen['accepted'],chosen['status']
    assert np.allclose(chosen['action'],preview['action'],atol=1e-12)
    assert np.linalg.norm(np.array(chosen['action'][:3])*.05)<=.003+1e-12
    bad=chosen['action'].copy();bad[0]+=.01
    with pytest.raises(ValueError):field.observe(obs.state,bad,obs.state)
    after=np.array(obs.state);after[:3]+=np.array(chosen['action'][:3])*.05*.25
    field.observe(obs.state,chosen['action'],after)
    assert field.actions==1 and field.pending is None


def test_partial_or_insufficient_arrival_margin_goal_is_not_licensed():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    for goal in([0,.004,.008],[0,.003,.030]):
        chosen=field.preview_egress(obs.state,obs.joint_position,goal,scene_assessor=assessor)
        assert not chosen['accepted']
        assert chosen['status']=='egress_fixed_goal_does_not_complete_exit_with_arrival_margin'


def test_without_exact_model_optin_normal_initial_start_stays_blocked():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    field.egress_context={'phase':'approach','purpose':'reference_repair','translation_mode':'separate'}
    selected=field.select(obs.state,obs.joint_position)
    assert not selected['accepted'] and selected['status']=='invalid_measured_start'


def test_preview_exception_preserves_guard_ledger_and_original_masks():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field();goal=field.egress_candidates(obs.state)[0]['goal_offset_m']
    before=(field.__dict__,guard.__dict__,pickle.dumps(guard.egress_evidence.__dict__))
    def broken(*args):raise RuntimeError('deliberate read-only preview error')
    with pytest.raises(RuntimeError,match='deliberate'):
        field.preview_egress(obs.state,obs.joint_position,goal,scene_assessor=broken)
    assert field.__dict__ is before[0] and guard.__dict__ is before[1]
    assert pickle.dumps(guard.egress_evidence.__dict__)==before[2]
    assert not field.checked.segment_valid(obs.state[:3],obs.state[:3])


def test_same_world_geometry_refresh_keeps_original_egress_origin_and_exemptions():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    original=field.egress_geometry;origin=original.state.copy();parts=set(original.initial_parts)
    after=np.array(obs.state);after[:3]+=[0,.001,.004]
    intent=copy.deepcopy(field.intent)
    # A newly centered intent must retain the physical scene and episode origin.
    intent['goal_offset_m']=[0,.005,.030]
    material=copy.deepcopy(field._transit_materialization)
    field.configure(after,intent,scene_assessment=guard.context,inventory_materialization=material)
    assert field.egress_geometry is original
    assert np.array_equal(original.state,origin) and original.initial_parts==parts
    # A short subsequent step cannot escape the old contextual40mm limit.
    current=origin[:3]+[0,.003,.039]
    result=original.check(current,current+[0,0,.003])
    assert not result['accepted'] and result['reason']=='egress_context_extent_exceeded'
    # Neither a refresh nor an old exemption allows reentry after measured exit.
    outside=origin[:3]+[0,.006,.032]
    result=original.check(outside,origin[:3])
    assert not result['accepted'] and result['reason']=='egress_new_halo_entry_or_reentry'


def test_unacknowledged_jump_to_goal_cannot_be_called_measured_arrival():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    goal=field.egress_candidates(obs.state)[0]['goal_offset_m']
    intent=copy.deepcopy(field.intent);intent.update(goal_offset_m=goal,goal_tolerance_m=.003,max_travel_m=.04)
    field.configure(obs.state,intent,scene_assessment=guard.context,inventory_materialization=field._transit_materialization)
    field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    jumped=np.array(obs.state);jumped[:3]=field.goal
    result=field.select(jumped,obs.joint_position)
    assert not result['accepted'] and result['status']=='egress_stale_current_robot_state'
    assert field.actions==0


def test_committed_egress_cannot_rebase_initial_halos_after_model_geometry_revision():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    goal=field.egress_candidates(obs.state)[0]['goal_offset_m']
    intent=copy.deepcopy(field.intent);intent.update(goal_offset_m=goal,goal_tolerance_m=.003,command_limit_m=.003,max_travel_m=.04,max_actions=120)
    material=copy.deepcopy(field._transit_materialization)
    field.configure(obs.state,intent,scene_assessment=guard.context,inventory_materialization=material)
    field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    guard.context.update(translation_mode='separate',purpose='reference_repair',intended_contact_id=None)
    selected=field.select(obs.state,obs.joint_position);assert selected['accepted']
    original=field.egress_geometry
    assert field._committed_egress_geometry is None
    after=np.array(obs.state);after[:3]+=np.array(selected['action'][:3])*.05*.2
    field.observe(obs.state,selected['action'],after)
    assert field._committed_egress_geometry is original
    # Realistic tiny orientation evolution preserves the original conservative
    # .04-radian robot envelope and world identity, rather than rebasing it.
    refreshed=after.copy();refreshed[3]+=.0001
    from gpt_field.action import _rotvec_quaternion
    rotated_intent=copy.deepcopy(intent);rotated_intent['orientation_target_xyzw']=_rotvec_quaternion(refreshed[3:6]).tolist()
    field.configure(refreshed,rotated_intent,scene_assessment=guard.context,inventory_materialization=material)
    assert field.egress_geometry is original
    assert field._egress_world_signature==field._committed_egress_world_signature
    revised=copy.deepcopy(material)
    cabinet=next(r for r in revised['objects']if r['id']=='drawer_cabinet')
    cabinet['world_geometry']['center_xyz_m'][2]+=.003
    field.configure(after,intent,scene_assessment=guard.context,inventory_materialization=revised)
    guard.state=after.copy();field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    assert field.egress_geometry is original and field.egress_candidates(after)==[]
    proof=field.egress_contract(after,[0,.0003,.002])
    assert not proof['accepted'] and proof['reason']=='egress_committed_world_geometry_changed'
    assert np.array_equal(original.state,np.array(obs.state))


def synthetic_egress_field():
    """Only synthetic RGB and declared model boxes; actual classes throughout."""
    from test_visual_egress_evidence_v23 import Plant
    from gpt_field.observation import image_to_data_url
    from gpt_field.robot_translation_response import MeasuredTranslationResponse
    _,_,_,_,archive,pred,_=archive_field()
    plant=Plant();state=np.array(archive.state)
    def observation():
        rgb=[np.repeat(np.clip(im,0,255).astype(np.uint8)[...,None],3,axis=2)for im in plant.images()]
        return PolicyObservation(image_to_data_url(rgb[0],rotate=False),image_to_data_url(rgb[1],rotate=False),
            tuple(state),'Move the empty tool away from the visible object',archive.joint_position,(0.,)*7)
    obs=observation();images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    roi=lambda box:(np.array(box)/223).tolist()
    refs={'external':{'update':'seed','moving_id':'distal-tool','moving_kind':'tool','target_id':'target',
        'goal_offset_uv':[0.,0.],'seed':{'moving_point_uv':roi([100,100]),'moving_roi_uv':roi([83,83,117,117]),
            'target_anchor_uv':roi([60,175]),'target_roi_uv':roi([43,158,77,192]),'tool_point_uv':None,'tool_roi_uv':None}},'wrist':None}
    monitor=RoleAwareVisualMonitorV21();monitor.configure(images,refs,state,intent_id=1)
    declarations=[{'id':'cab','external_roi_uv':roi([142,92,178,128]),'wrist_roi_uv':roi([72,77,108,113]),
        'geometry':copy.deepcopy(next(r['geometry']for r in pred['scene_inventory']if r['id']=='drawer_cabinet'))},
        {'id':'target','external_roi_uv':roi([42,157,78,193]),'wrist_roi_uv':None,
        'geometry':copy.deepcopy(next(r['geometry']for r in pred['scene_inventory']if r['id']=='black_bowl'))}]
    inventory=VisualSceneInventoryV17();inventory.configure(obs,declarations)
    material=inventory.materialize(state,'target',allowed_contact='none')
    context={'purpose':'reference_repair','translation_mode':'separate','intended_contact_id':None,
        'tool_external_roi_uv':roi([82,82,118,118]),'obstacle_assessment':'present','nearby_obstacles':material['nearby_obstacles']}
    guard=ProjectedTranslationGuardV23();guard.configure(images,context,state,target_id='target',monitor=monitor)
    guard.bind_egress_runtime(inventory,monitor)
    field=MetricFieldExecutorV23();field.bind_translation_guard(guard,monitor)
    response=MeasuredTranslationResponse();field.robot_response=response
    contact=InventoryTargetGeometryV16();contact.configure(state,material['target_geometry'],target_id='target',tool_radius_m=.045)
    field.contact_geometry=contact
    intent=copy.deepcopy(pred['field_intent']);intent.update(goal_offset_m=[0,.006,.035],goal_tolerance_m=.003,
        command_limit_m=.003,max_travel_m=.04,max_actions=120)
    field.configure(state,intent,scene_assessment=context,inventory_materialization=material)
    field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    scene=SceneCoverageGuardV23()
    def assess(current,delta):
        material=inventory.materialize(current,'target',allowed_contact='none')
        material['applied_metric_ids']=[r['id']for r in field.region_geometry]
        material['applied_image_ids']=[r['id']for r in guard.neighbors]
        return scene.assess(current,delta,materialization=material,field=field,monitor=monitor,
            prediction={'controller':'visual_field','phase':'recover','target_id':'target','target_contact':'none',
                'motion_context':context,'field_intent':field.intent})
    field.scene_assessor=assess
    return field,guard,monitor,inventory,response,plant,state,observation,obs


def test_actual_executor_guard_scene_rgb_plant_reuses_one_fixed_goal_beyond30mm_at_slow_gain():
    field,guard,monitor,inventory,response,plant,state,observation,obs=synthetic_egress_field()
    original=field.egress_geometry;goal=field.goal.copy();commands=[]
    for step in range(120):
        selected=field.select(state,obs.joint_position)
        assert selected['accepted'],(step,selected['status'],guard.egress_evidence.summary())
        action=np.array(selected['action']);delta=action[:3]*.05
        assert np.linalg.norm(delta)<=.003+1e-10 and action[6]==0
        before=state.copy();state[:3]+=delta*.2;plant.travel+=float(np.linalg.norm(delta*.2))
        after=observation();images={'external':np.asarray(decode_image(after.image)),'wrist':np.asarray(decode_image(after.wrist_image))}
        field.observe(before,action,state)
        response.observe(before,action,state,before_joints=obs.joint_position,after_joints=after.joint_position)
        monitor.observe(images,state,action)
        inventory.observe(obs,after)
        guard.observe(images,state,action,monitor=monitor)
        guard.observe_egress(before,action,state)
        obs=after;commands.append(np.linalg.norm(delta))
        assert field.pending is None and field.egress_geometry is original
        assert field.goal==pytest.approx(goal)
        if plant.travel>.031:break
    assert plant.travel>.031 and field.actions==len(commands)>50
    assert field.plan_calls==1 and field.goal_revisions==0
    assert guard.egress_evidence.grants>=7
    assert guard.egress_evidence.egress_travel_m==pytest.approx(plant.travel)
    assert guard.observation_travel_m>=plant.travel-1e-10
    assert field._committed_egress_geometry is original
    assert not guard.egress_evidence.summary()['clearance_certified']


def test_recorded_rejected_hold_drift_uses_current_acknowledged_offer_boundary():
    from gpt_field.visual_field_policy_v23 import VisualFieldPolicyV23
    from test_visual_field_v23_policy import OfferClient
    root=Path(__file__).resolve().parents[1]/'runs/visual_field_v23_design_001/controller_diagnostic_002'
    if not root.exists():pytest.skip('Optional deterministic controller diagnostic unavailable')
    rows=list(map(json.loads,(root/'trace.jsonl').read_text().splitlines()))
    before=PolicyObservation.from_dict(next(r['observation']for r in rows if r['event']=='reset'))
    after=PolicyObservation.from_dict(next(r['observation']for r in rows if r['event']=='executed'))
    pred=json.loads((root/'archived_prediction.json').read_text())
    class CurrentPoseOfferClient(OfferClient):
        def predict(self,observation):
            from gpt_field.action import _rotvec_quaternion
            value=super().predict(observation)
            # The saved dictionary is a parsed internal prediction. Reproduce
            # the subscription parser's current-pose orientation construction.
            value['field_intent']['orientation_target_xyzw']=_rotvec_quaternion(np.array(observation['observation/state'][3:6])).tolist()
            return value
    policy=VisualFieldPolicyV23(CurrentPoseOfferClient(pred))
    held=policy.act(before)
    assert held.diagnostics['selection']['status']=='invalid_measured_start'
    assert held.action[:3]==(0.,0.,0.)
    policy.observe_transition(before,held,after)
    assert np.linalg.norm(np.array(after.state[:3])-before.state[:3])==pytest.approx(1.6292605447429793e-5)
    assert not np.allclose(policy.local.expected_state,after.state,atol=1e-9,rtol=0.)
    assert np.array_equal(policy.translation_guard.state,after.state)
    assert np.array_equal(policy.inventory.state,after.state)
    original=policy.local.egress_geometry
    offers=[c for c in policy.recovery_options['candidates']if c.get('source')=='initial_uncertainty_egress']
    assert offers,policy.recovery_options
    assert policy.local.egress_geometry is original and policy.local.pending is None
    assert policy.translation_guard.egress_evidence.spent_m==0
    # Stale observation offers are refused, and proposal generation has not
    # silently synchronized the stopped executor or licensed an actual command.
    assert policy.local.egress_candidates(before.state)==[]
    policy.local.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    policy.local.stop_reason=None
    refused=policy.local.select(after.state,after.joint_position)
    assert not refused['accepted'] and refused['status']=='egress_stale_current_robot_state'
    policy.local.stop_reason='invalid_measured_start'
    selected=policy.act(after)
    assert selected.diagnostics['selection']['accepted'],selected.diagnostics['selection']
    assert policy.client.calls==2
    assert np.allclose(np.array(selected.action[:3])*.05,
        policy.client.offer['preview_first_command_delta_m'],atol=1e-10)
    stale_preview=policy.local.preview_egress(before.state,before.joint_position,offers[0]['recommended_goal_offset_m'])
    assert not stale_preview['accepted']
