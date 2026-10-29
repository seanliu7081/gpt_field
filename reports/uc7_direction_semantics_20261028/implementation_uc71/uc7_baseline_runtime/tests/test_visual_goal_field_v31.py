"""Causal image-goal proposals; synthetic observations do not certify semantics."""
import copy
import json
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_goal_field_v31 import VisualGoalFieldV31


def fixture():
    def track(point):
        return SimpleNamespace(position=np.array(point, float), scale=np.array([223.,223.]),
            reliable=True, last_residual_px=.12, last_inliers=8)
    monitor = SimpleNamespace(tracks={'moving':track([125.,90.]), 'target':track([80.,90.])})
    monitor._external_reliable = lambda: all(t.reliable for t in monitor.tracks.values())
    prediction = {'target_id':'object', 'phase':'approach', 'visual_references':{'external':{
        'moving_kind':'tool', 'moving_id':'hand_texture', 'target_id':'object', 'goal_offset_uv':[0.,0.]}}}
    state = np.array([0.,0.,1.,0.,0.,0.,.04,-.04])
    helper = VisualGoalFieldV31();helper.configure(state, monitor, prediction)
    return helper, monitor, prediction, state


def advance(helper, monitor, prediction, state, delta=(0,.0035,0), flow=(-1.,0.),
            *, action=None, acknowledged=True, joint_position=None):
    after = state.copy();after[:3] += delta
    monitor.tracks['moving'].position += flow
    command = np.r_[np.asarray(delta)/.05, [0.,0.,0.,0.]] if action is None else action
    helper.observe(state, command, after, monitor=monitor, prediction=prediction,
        acknowledged=acknowledged, joint_position=joint_position)
    return after


def trained(count=9, *, flow=(-1.,0.), delta=(0,.0035,0)):
    h,m,p,s=fixture()
    for _ in range(count):s=advance(h,m,p,s,delta=delta,flow=flow)
    return h,m,p,s


def test_rank_one_descent_uses_measured_direction_and_fixed_finite_goals():
    h,m,p,s=trained();r=h.propose(s, motion_axes='xy')
    assert r['accepted'] and r['supported_rank']==1
    assert all(c['goal_offset_m'][1]>0 and abs(c['goal_offset_m'][0])<1e-10
        and c['goal_offset_m'][2]==0 for c in r['candidates'])
    assert all(.006 <= c['proposal_distance_m'] <= .03 for c in r['candidates'])
    assert all(c['predicted_error_reduction_px']>0 and c['trust_limit_m']<=c['measured_directional_span_m']
        for c in r['candidates'])
    assert not r['execution_authorized'] and not r['target_pose_estimated']
    assert len({c['proposal_id'] for c in r['candidates']})==len(r['candidates'])


def test_validation_is_before_training_and_fit_alone_cannot_offer():
    h,m,p,s=trained(3)
    assert h.model.matrix is not None and len(h.held_out)==0
    assert not h.propose(s)['accepted']
    s=advance(h,m,p,s);assert len(h.held_out)==1 and not h.propose(s)['accepted']
    s=advance(h,m,p,s);assert h.propose(s)['accepted']
    rows=h.summary()['held_out_records']
    assert [v['training_samples_before_prediction'] for v in rows]==[3,4]
    assert all(np.linalg.norm(np.asarray(v['prediction_px'])-v['flow_px'])<1e-4 for v in rows)


def test_measured_gain_not_command_scale_learns_same_image_field():
    h,m,p,s=fixture()
    for _ in range(35):
        s=advance(h,m,p,s,delta=(0,.0007,0),flow=(-.2,0),action=[0,.07,0,0,0,0,0])
    assert h.propose(s)['accepted']
    assert h.model.matrix[1,0]==pytest.approx(-1/.0035,rel=1e-4)
    assert h.measured_training_path_m <= .0245+1e-12


def test_opposite_objective_does_not_borrow_unobserved_reverse_validation():
    h,m,p,s=trained(flow=(1.,0.))
    assert not h.propose(s)['accepted']
    # Now acknowledged reverse motion independently earns reverse validation.
    for _ in range(8):s=advance(h,m,p,s,delta=(0,-.0035,0),flow=(-1.,0.))
    r=h.propose(s);assert r['accepted']
    assert all(c['goal_offset_m'][1]<0 for c in r['candidates'])


def test_metric_conflict_only_claims_observed_component_not_3d_pose():
    h,m,p,s=trained();r=h.propose(s, metric_goal_xyz=s[:3]+[.1,-.1,-.2])
    mismatch=r['metric_image_disagreement']
    assert mismatch['opposes_visual_descent_in_observed_subspace']
    assert mismatch['observed_subspace_component_m']==pytest.approx([0,-.1,0])
    assert mismatch['unobserved_component_m']==pytest.approx([.1,0,-.2])
    assert mismatch['not_a_target_pose_or_geometry_rejection']


def test_proposals_are_pure_and_repeated_configuration_keeps_paid_evidence():
    h,m,p,s=trained();state=pickle.dumps(h.__dict__);before=copy.deepcopy(h.summary())
    a=h.propose(s);b=h.propose(s)
    assert a==b and pickle.dumps(h.__dict__)==state
    h.configure(s,m,copy.deepcopy(p))
    assert h.summary()==before
    assert 'budget' not in h.__dict__ and not h.summary()['physical_budget_owned']


@pytest.mark.parametrize('change',['feature','offset','target_role','image_scale'])
def test_reference_semantic_or_representation_changes_cannot_transfer_validation(change):
    h,m,p,s=trained();epoch=h.epoch
    if change=='feature':m.tracks['moving']=copy.deepcopy(m.tracks['moving'])
    if change=='offset':p['visual_references']['external']['goal_offset_uv']=[.01,0.]
    if change=='target_role':p['visual_references']['external']['moving_kind']='payload'
    if change=='image_scale':
        for t in m.tracks.values():t.scale=np.array([255.,255.])
    h.configure(s,m,p)
    assert len(h.model.samples)==len(h.held_out)==0
    assert not h.propose(s)['accepted']
    if change!='target_role':assert h.epoch==epoch+1


@pytest.mark.parametrize('kind',['orientation','fingers','state_gap','pixel_revision','anchor_motion'])
def test_actual_domain_changes_require_new_causal_evidence(kind):
    h,m,p,s=trained();after=s.copy()
    if kind=='orientation':after[3]+=.04
    if kind=='fingers':after[6]+=.003
    if kind=='state_gap':after[0]+=.001
    if kind=='pixel_revision':m.tracks['moving'].position+=[2,0]
    if kind=='anchor_motion':
        m.tracks['target'].position+=[7,0]
        h.observe(s,np.zeros(7),after,monitor=m,prediction=p)
    else:h.configure(after,m,p)
    assert not h.propose(after)['accepted'] and not h.held_out


def test_joint_domain_changes_reset_estimator_without_any_control_effect():
    h,m,p,s=fixture();q=np.zeros(7);h.configure(s,m,p,joint_position=q)
    for _ in range(8):s=advance(h,m,p,s,joint_position=q)
    assert h.propose(s)['accepted']
    q[2]=.36;h.configure(s,m,p,joint_position=q)
    assert not h.propose(s)['accepted'] and h.status=='visual_objective_joint_domain_changed'


def test_proposal_binds_latest_acknowledged_joints_and_cannot_mutate_on_stale_query():
    h,m,p,s=fixture();q=np.zeros(7);h.configure(s,m,p,joint_position=q)
    for _ in range(8):
        q=q.copy();q[1]+=.01
        s=advance(h,m,p,s,joint_position=q)
    before=pickle.dumps(h.__dict__)
    assert h.propose(s,joint_position=q)['accepted']
    assert h.propose(s,joint_position=np.zeros(7))['reason']=='visual_objective_proposal_joint_feedback_stale'
    changed=q.copy();changed[1]+=.5
    assert not h.propose(s,joint_position=changed)['accepted']
    assert pickle.dumps(h.__dict__)==before
    candidate=h.propose(s)['candidates'][0]
    assert candidate['binding']['joint_position']==pytest.approx(q)
    assert h.summary()['current_snapshot']['joint_position']==pytest.approx(q)


def test_hold_and_unacknowledged_commands_never_produce_training_credit():
    h,m,p,s=fixture()
    for _ in range(8):s=advance(h,m,p,s,action=np.zeros(7))
    assert h.training_updates==0 and not h.held_out
    for _ in range(8):s=advance(h,m,p,s,acknowledged=False)
    assert h.training_updates==0 and not h.propose(s)['accepted']


def test_static_noisy_and_future_reference_cannot_claim_predictive_response():
    h,m,p,s=trained(flow=(0.,0.));assert not h.propose(s)['accepted']
    h,m,p,s=trained();m.tracks['moving'].reliable=False
    assert h.propose(s)['reason']=='visual_objective_proposal_reference_stale'
    m.tracks['moving'].reliable=True;m.tracks['moving'].position[0]+=.1
    assert not h.propose(s)['accepted']
    future=s.copy();future[1]+=.001
    assert h.propose(future)['reason']=='visual_objective_proposal_state_stale'


def test_unsupported_axis_has_no_fake_depth_solution_and_span_caps_short_history():
    h,m,p,s=trained(5)
    assert not h.propose(s,motion_axes='z')['accepted']
    result=h.propose(s,max_goal_distance_m=.5)
    assert result['accepted']
    assert all(c['proposal_distance_m']<=.0175+1e-12 for c in result['candidates'])
    assert not h.propose(s,max_goal_distance_m=.004)['accepted']


def test_new_noise_or_counter_response_removes_heading_validation():
    h,m,p,s=trained()
    for n in range(9):s=advance(h,m,p,s,flow=(3. if n%2 else -3.,0.))
    assert not h.propose(s)['accepted']


def test_real_v30_paid47step_observation_stream_offers_rank_one_positive_y():
    path=pathlib.Path(__file__).resolve().parents[1]/'runs/visual_field_v30_zero_shot_001/trials/bowl_original_zero_hybrid_v30/trace.jsonl'
    if not path.exists():pytest.skip('Optional original rollout artifact unavailable')
    rows=[];prediction=None
    for line in path.open():
        record=json.loads(line)
        if record.get('event')=='action':prediction=record['diagnostics'].get('prediction') or prediction
        if record.get('event')=='feedback' and 195<=record['step']<=241:
            rows.append((record['feedback'],copy.deepcopy(prediction)))
    assert len(rows)==47
    h,m,p,s=fixture();first=rows[0][0];active=first['task_progress']['active']
    p=rows[0][1];ext=p['visual_references']['external'];offset=np.asarray(ext['goal_offset_uv'])*223
    anchor=np.asarray(active['last_anchor_px']);m.tracks['target'].position=anchor.copy()
    m.tracks['moving'].position=anchor+offset+np.asarray(active['initial_error_vector_px'])
    s=np.asarray(first['before_state']);h.configure(s,m,p)
    for row,p in rows:
        active=row['task_progress']['active'];anchor=np.asarray(active['last_anchor_px'])
        m.tracks['target'].position=anchor.copy()
        m.tracks['moving'].position=anchor+offset+np.asarray(active['current_error_vector_px'])
        # Replay recorded tracked-pixel observations, not raw-RGB tracker or simulator.
        for track in m.tracks.values():track.last_residual_px=active['maximum_noise_px']*np.sqrt(8)/np.sqrt(2)
        h.observe(row['before_state'],row['action'],row['after_state'],monitor=m,prediction=p)
        s=np.asarray(row['after_state'])
    result=h.propose(s,motion_axes='xy',metric_goal_xyz=[-.19124162683394808,-.18109060188932152,.7992495176982402])
    assert result['accepted'] and result['supported_rank']==1
    assert result['metric_image_disagreement']['opposes_visual_descent_in_observed_subspace']
    assert h.training_updates>=12 and len(h.held_out)>=8
    assert all(c['goal_offset_m'][1]>0 and c['goal_offset_m'][2]==0 for c in result['candidates'])
    assert any(.005999 <= c['proposal_distance_m'] <= .006001 for c in result['candidates'])
    assert result['candidates'][0]['predicted_error_reduction_px']>5.
