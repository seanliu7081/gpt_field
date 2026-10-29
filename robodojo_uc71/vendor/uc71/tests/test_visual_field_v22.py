"""Continuous recovery offers are executable across headings and grid phases."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.action import _rotvec_quaternion
from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_field_v22 import MetricFieldExecutorV22
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17
from test_visual_field_v4 import Q
from test_visual_translation_guard_v21 import plant, archive_trigger, setup


def promote(field):
    # The already-configured real fixture keeps exact tracks and checked-owner
    # identities. Only the class changes to exercise the new opt-in executor.
    field.__class__=MetricFieldExecutorV22
    field.heading_recovery=None
    return field


def bind_history(guard,field,direction,measured):
    contract=field.nominal_regions_for_segment(measured[:3],measured[:3]+direction*.002)[0]['directional_retreat_contract']
    identity=next(iter(guard.neighbors))['id']
    guard.worsening={identity}
    guard.worsening_motion[identity]={'measured_direction_xyz':(-direction).tolist(),
        'measured_net_translation_m':.012,'state8':measured.tolist(),
        'geometry_sha256':contract['geometry_sha256'],'risk_evidence':{'state':'worsening'}}
    guard.context['translation_mode']='separate'


def configure_reverse(field,guard,measured,direction):
    intent=copy.deepcopy(field.intent)
    intent.update(goal_offset_m=(direction*.008).tolist(),goal_tolerance_m=.003,
        max_travel_m=.008,command_limit_m=.002,
        orientation_target_xyzw=_rotvec_quaternion(measured[3:6]).tolist())
    field.configure(measured,intent,scene_assessment=guard.context,
                    inventory_materialization=field._transit_materialization)


def snapshot(field,guard):
    return (field.__dict__,pickle.dumps(guard.__dict__),
        pickle.dumps(field.contact_geometry.__dict__),pickle.dumps(field.tracker.__dict__),
        json.dumps(field._transit_materialization,sort_keys=True),field.planner,
        guard.tool_track,field.visual_monitor.tracks['moving'])


def assert_unchanged(before,field,guard):
    after=snapshot(field,guard)
    assert before[0] is after[0]
    assert before[1:5]==after[1:5]
    assert all(a is b for a,b in zip(before[5:],after[5:]))


HEADINGS=[np.array(v,float)/np.linalg.norm(v) for v in
    [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
     (1,1,1),(-1,-1,1),(1,-2,3),(-.3163442896,-.5044369216,.8034112786),
     (.03,-1,.8),(-.77,.2,-.1)]]


@pytest.mark.parametrize('direction',HEADINGS)
def test_same_actual_preview_and_execution_across_headings(direction):
    guard,monitor,field,measured=plant();promote(field)
    bind_history(guard,field,direction,measured)
    saved=snapshot(field,guard)
    preview=field.preview_recovery(measured,Q,direction*.008)
    assert preview['accepted'],preview['status']
    assert_unchanged(saved,field,guard)
    configure_reverse(field,guard,measured,direction)
    selected=field.select(measured,Q)
    assert selected['accepted'],selected['status']
    assert np.allclose(selected['action'],preview['action'],atol=1e-12,rtol=0)
    delta=np.asarray(selected['action'][:3])*.05
    assert np.allclose(delta/np.linalg.norm(delta),direction,atol=1e-11)
    assert selected['diagnostics']['heading_recovery']['route_id']==preview['diagnostics']['heading_recovery']['route_id']
    assert field.last_plan.search_metadata['method']=='heading_constrained_continuous_distance'
    assert guard.experiments==[] and guard.observation_travel_m==0
    wrong=list(selected['action']);wrong[0]+=.01
    with pytest.raises(ValueError):field.observe(measured,wrong,measured)
    after=measured.copy();after[:3]+=delta*.3
    field.observe(measured,selected['action'],after)
    assert field.actions==1 and field.pending is None


@pytest.mark.parametrize('offset',[[-.0013,.0024,.0007],[.00249,-.0006,-.0011],[.0048,.0037,.002]])
def test_subgrid_current_pose_preserves_heading_and_fixed_route(offset):
    guard,monitor,field,measured=plant();promote(field)
    direction=HEADINGS[9]
    # Explicit initial measurement phase relative to the old field grid. A
    # prospective new intent must not inherit any old grid-corner tangent.
    measured=measured.copy();measured[:3]+=offset
    guard.state=measured.copy();monitor.state=measured.copy()
    bind_history(guard,field,direction,measured)
    preview=field.preview_recovery(measured,Q,direction*.008)
    assert preview['accepted'],preview['status']
    assert np.allclose(np.asarray(preview['action'][:3])*.05,direction*.002,atol=1e-11)


def archive_v21_failure():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v21_zero_shot_001/trials/bowl_original_zero_hybrid_v21/trace.jsonl'
    if not path.exists():pytest.skip('Optional real V21 archive not present')
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    actions={r['step']:r for r in rows if r['event']=='action'}
    feedback={r['step']:r['feedback'] for r in rows if r['event']=='feedback'}
    obs=PolicyObservation.from_dict(next(r['observation'] for r in rows if r['event']=='executed' and r['step']==227))
    pred=copy.deepcopy(actions[228]['diagnostics']['prediction']);material=copy.deepcopy(actions[228]['diagnostics']['scene_materialization'])
    refs=copy.deepcopy(pred['visual_references']);tracking=feedback[227]['visual']['tracking']
    refs['external']['update']='seed'
    refs['external']['seed']={'moving_point_uv':tracking['moving']['tracked_point_uv'],
        'moving_roi_uv':tracking['moving']['current_roi_uv'],
        'target_anchor_uv':tracking['target']['tracked_point_uv'],
        'target_roi_uv':tracking['target']['current_roi_uv'],'tool_point_uv':None,'tool_roi_uv':None}
    images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    context=copy.deepcopy(actions[228]['diagnostics']['feedback']['effective_motion_context'])
    guard,monitor,field=setup(obs.state,images,refs,context,pred['field_intent'],material)
    promote(field)
    guard.worsening={'other_bowl'}
    guard.worsening_motion=copy.deepcopy(feedback[227]['translation_guard']['directional_retreat']['actual_worsening_motion'])
    return guard,monitor,field,np.asarray(obs.state),pred,feedback[228]


def test_actual_v21_rejected_vertical_route_now_tracks_exact_offered_reverse():
    guard,monitor,field,measured,pred,feedback=archive_v21_failure()
    recorded=feedback['translation_guard']['directional_retreat']['last_admission']
    assert not recorded['accepted']
    assert recorded['diagnostics']['candidate_delta_xyz_m']==[0.,0.,.002]
    direction=np.asarray(pred['field_intent']['goal_offset_m']);direction/=np.linalg.norm(direction)
    saved=snapshot(field,guard)
    preview=field.preview_recovery(measured,Q,pred['field_intent']['goal_offset_m'])
    assert preview['accepted'],preview['status']
    assert_unchanged(saved,field,guard)
    selected=field.select(measured,Q)
    assert selected['accepted'],selected['status']
    assert np.allclose(np.asarray(selected['action'][:3])*.05,direction*.002,atol=1e-11)
    assert np.allclose(selected['action'],preview['action'],atol=1e-12)
    assert len(selected['diagnostics']['heading_recovery']['scene_ids'])==8
    assert guard.worsening=={'other_bowl'} and guard.experiments==[]


@pytest.mark.parametrize('fault',['state','domain','geometry','payload','contact','unknown','scene_missing'])
def test_changed_constraints_cannot_be_licensed_by_continuous_route(fault):
    guard,monitor,field,measured=plant();promote(field);direction=HEADINGS[9]
    bind_history(guard,field,direction,measured)
    if fault=='state':measured=measured.copy();measured[0]+=.001
    elif fault=='domain':guard.worsening_motion['neighbor']['state8'][6]-=.01
    elif fault=='geometry':guard.worsening_motion['neighbor']['geometry_sha256']='changed'
    elif fault=='payload':field.contact_geometry._grasp_candidates['payload']={'attachment_observed':False}
    elif fault=='contact':field.contact_geometry._record['geometry']['allowed_contact']='fingers'
    elif fault=='unknown':
        field.intent['unknown_regions']=[{'id':'unseen','lower_offset_m':[-.002,-.002,.001],
            'upper_offset_m':[.002,.002,.003],'uncertainty_m':0.}]
    else:field.scene_assessor=lambda s,d:{'accepted':True,'diagnostics':{}}
    before=snapshot(field,guard)
    chosen=field.preview_recovery(measured,Q,direction*.008)
    assert not chosen['accepted'],chosen
    assert_unchanged(before,field,guard)
    assert guard.experiments==[]


def test_preview_exception_and_pending_action_restore_every_original_object():
    guard,monitor,field,measured=plant();promote(field);direction=HEADINGS[9]
    bind_history(guard,field,direction,measured)
    saved=snapshot(field,guard)
    def raises(prospective,state,delta):
        assert prospective is field
        assert np.allclose(prospective.goal,state[:3]+direction*.008)
        raise RuntimeError('synthetic scene failure')
    with pytest.raises(RuntimeError,match='synthetic scene failure'):
        field.preview_recovery(measured,Q,direction*.008,scene_assessor=raises)
    assert_unchanged(saved,field,guard)
    configure_reverse(field,guard,measured,direction);selection=field.select(measured,Q)
    assert selection['accepted']
    pending=field.pending
    result=field.preview_recovery(measured,Q,direction*.008)
    assert not result['accepted'] and field.pending is pending


def test_no_recovery_request_keeps_weighted_graph_planner():
    guard,monitor,field,measured=plant();promote(field)
    selected=field.select(measured,Q)
    assert selected['accepted']
    assert field.heading_recovery is None
    assert field.last_plan.search_metadata['method']=='dijkstra_distance_field'


def test_policy_measured_cap_is_still_applied_in_actual_preview():
    from gpt_field.robot_translation_response import MeasuredTranslationResponse
    guard,monitor,field,measured=plant();promote(field);direction=HEADINGS[9]
    field.robot_response=MeasuredTranslationResponse()
    bind_history(guard,field,direction,measured)
    chosen=field.preview_recovery(measured,Q,direction*.008,measured_cap_m=.0004)
    assert not chosen['accepted']
    assert 'allowance' in chosen['status']


def test_scene_proof_deduplication_is_exact_and_never_survives_next_proposal():
    from gpt_field.visual_field_v21 import MetricFieldExecutorV21
    guard,monitor,field,measured=plant();promote(field)
    original=field.scene_assessor;calls=[];changed=[False]
    def counting(state,delta):
        calls.append((state.copy(),delta.copy()))
        result=original(state,delta)
        if changed[0]:
            result['accepted']=False
            result['reason']='counterfactual_current_scene_changed'
        return result
    field.scene_assessor=counting
    end=measured[:3]+np.array([0.,0.,.002])
    expected=MetricFieldExecutorV21.nominal_regions_for_segment(field,measured[:3],end)
    assert len(calls)==2
    calls.clear()
    result=field.nominal_regions_for_segment(measured[:3],end)
    assert result==expected and len(calls)==1
    assert field.scene_assessor is counting
    changed[0]=True
    result=field.nominal_regions_for_segment(measured[:3],end)
    assert len(calls)==2
    assert result[0]['directional_retreat_contract']['full_scene_sweep_accepted'] is False
    assert field.scene_assessor is counting


def test_scene_proof_callback_is_restored_after_exception():
    guard,monitor,field,measured=plant();promote(field)
    def broken(state,delta):raise RuntimeError('scene callback exception')
    field.scene_assessor=broken
    with pytest.raises(RuntimeError,match='scene callback exception'):
        field.nominal_regions_for_segment(measured[:3],measured[:3]+[0,0,.002])
    assert field.scene_assessor is broken


def test_recovery_actual_joint_limit_remains_enforced():
    from gpt_field.visual_field_v4 import PANDA_JOINT_UPPER
    guard,monitor,field,measured=plant();promote(field);direction=HEADINGS[9]
    bind_history(guard,field,direction,measured)
    joints=np.asarray(Q).copy();joints[0]=PANDA_JOINT_UPPER[0]
    before=snapshot(field,guard)
    chosen=field.preview_recovery(measured,joints,direction*.008)
    assert not chosen['accepted'] and chosen['status']=='measured_joint_limit'
    assert_unchanged(before,field,guard)
