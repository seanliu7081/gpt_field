"""A finite acquisition must learn through actual acknowledgements, not reseeds."""
import copy
import pickle

import numpy as np
import pytest

from gpt_field.robot_translation_response_v25 import MeasuredTranslationResponseV25
from gpt_field.visual_heading_evidence_v28 import HeadingEvidenceAcquisitionV28
from gpt_field.visual_translation_guard_v27 import _NeighborResponse


def state():
    return np.array([0., 0., 1., 0., 0., 0., .04, -.04])


def binding(s, delta, *, intent='first', representation=('tool1', 'object1'),
            goal=(.04, 0., 1.), geometry='world-boxes-original'):
    return {'eligible': True, 'route_family': 'normal', 'empty_tool': True,
        'target_contact': 'none', 'contact_or_visual_goal_dependency': False,
        'full_scene_sweep_accepted': True, 'hard_segment_clear': True,
        'outside_target_contact_band': True, 'state8': s.tolist(),
        'start_xyz_m': s[:3].tolist(), 'end_xyz_m': (s[:3]+delta).tolist(),
        'fixed_goal_xyz_m': list(goal), 'fixed_origin_xyz_m': [0., 0., 1.],
        'observation_epoch': 2, 'geometry_sha256': geometry,
        'model_intent_key': intent, 'representation_key': representation,
        'warning_ids': ['neighbor']}


def assess(h, r, s, d, *, used=0., shared=0., experiment=None, evidence=None,
           bind=None, remaining=.04, actions=120):
    return h.assess(s, d, binding=bind or binding(s,d), experiment=experiment,
        heading_used_m=used, shared_used_m=shared, intent_remaining_m=remaining,
        intent_remaining_actions=actions, robot_response=r,
        image_evidence={} if evidence is None else evidence)


def acknowledge(h, r, s, d, decision, *, gain=1., index=0):
    after=s.copy(); after[:3]+=d*gain
    action=np.r_[d/.05,0.,0.,0.,0.]
    r.observe(s,action,after)
    result=h.observe(s,action,after,pending=decision['pending'],experiment_index=index)
    return after,result


def snapshot(s, *, gain=500., representation=1, identity='neighbor'):
    return {'state': s[:3].copy(), 'tool': np.array([s[0]*gain,s[1]*gain]),
        'neighbors':{identity:{'position':np.zeros(2),'measurement_epoch':representation}}}


@pytest.mark.parametrize('gain', [1., .2])
def test_causal_new_heading_acquires_robot_and_two_object_holdouts_inside_eight_mm(gain):
    helper=HeadingEvidenceAcquisitionV28();robot=MeasuredTranslationResponseV25();s=state()
    witness=_NeighborResponse(1); witness.observe(snapshot(s),'neighbor')
    used=0.;commands=[];heldout_before=[]
    for i in range(60):
        # Construct the real candidate before asking for its full geometry proof.
        cap=helper.requested_cap(s,[.003,0,0],heading_used_m=used,shared_used_m=used,
            intent_remaining_m=.04-used,intent_remaining_actions=120-i,robot_response=robot)
        assert cap.get('usable'), cap
        d=np.array(cap['command_delta_m'])
        predicted=robot.predict(d,s)
        actual_query=np.asarray(predicted.get('mean_delta_m') or d.tolist())
        evidence=witness.evidence(actual_query,s[:3])
        previous_helper=pickle.dumps(helper); previous_robot=pickle.dumps(robot); previous_witness=pickle.dumps(witness)
        result=assess(helper,robot,s,d,used=used,shared=used,experiment=None if i==0 else 0,
            evidence={'neighbor':evidence})
        assert result['accepted'], result
        assert pickle.dumps(helper)==previous_helper
        assert pickle.dumps(robot)==previous_robot and pickle.dumps(witness)==previous_witness
        if result['qualification']['ready']:
            break
        heldout_before.append(len(witness.errors))
        commands.append(float(np.linalg.norm(d)))
        before=s.copy();s,observed=acknowledge(helper,robot,s,d,result,gain=gain)
        assert observed['matched']
        # Existing per-object model does the training, AFTER this action's
        # qualification/prediction. Helper never receives an estimator to fit.
        witness.observe(snapshot(s),'neighbor')
        used+=float(np.linalg.norm(s[:3]-before[:3]))
    else:
        pytest.fail('finite acquisition did not reach genuinely held-out support')
    assert result['reason'].endswith('validated_transit_ready')
    assert used < .008 and result['qualification']['robot_response_ready']
    assert witness.evidence(actual_query,s[:3])['held_out_count'] >= 2
    assert commands[:4] == pytest.approx([.001]*4)
    assert max(commands) <= .003+1e-12
    assert len(commands) == (5 if gain==1 else 13)
    assert used == pytest.approx(.0055 if gain==1 else .00575)
    assert heldout_before[:3] == [0,0,0]
    assert helper.summary()['physical_budget_allocated_m']==0.
    assert helper.rows[0]['measured_path_m']==pytest.approx(used)


def test_cap_reselection_does_not_license_longer_or_different_segment():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state()
    result=assess(h,r,s,np.array([.003,0,0]))
    assert not result['accepted'] and result['command_cap_m']==.001
    assert result['reason'].endswith('candidate_requires_cap_reselection')
    assert result['pending'] is None and h.rows=={}
    short=np.array([.001,0,0])
    assert assess(h,r,s,short)['accepted']
    stale=binding(s,np.array([.003,0,0]))
    assert not assess(h,r,s,short,bind=stale)['accepted']


@pytest.mark.parametrize('gain', [0., -.25, .005])
def test_persistent_observed_blockage_or_opposition_cannot_be_erased_by_reseed(gain):
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    for i in range(4):
        result=assess(h,r,s,d,experiment=None if i==0 else 0)
        assert result['accepted']
        s,_=acknowledge(h,r,s,d,result,gain=gain)
    assert h.rows[0]['adverse_response'] is not None
    # Changing all declaration keys does not remove the actual row's evidence.
    fresh=binding(s,d,intent='fresh-model-choice',representation=('new-tool','new-object'),geometry='new-world-boxes')
    refused=assess(h,r,s,d,experiment=0,bind=fresh)
    assert not refused['accepted'] and refused['reason'].endswith('observed_robot_nonresponsive')
    assert h.rows[0]['acknowledged_actions']==4


def test_residual_hold_stays_with_last_actual_experiment_not_pending_next_heading():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d);s,_=acknowledge(h,r,s,d,result,gain=.2)
    next_d=np.array([0,.001,0]);pending=assess(h,r,s,next_d,experiment=None)['pending']
    after=s.copy();after[0]+=.0001
    observed=h.observe(s,np.zeros(7),after,pending=pending,experiment_index=1)
    assert not observed['matched'] and observed['is_hold']
    assert list(h.rows)==[0] and h.last_experiment_index==0
    assert h.rows[0]['measured_path_m']==pytest.approx(.0003)
    assert h.rows[0]['residual_hold_path_m']==pytest.approx(.0001)
    assert h.rows[0]['acknowledged_actions']==1
    assert h.rows[0]['last_qualification']['ready'] is False


def test_hold_before_first_action_cannot_create_experiment_or_validation():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d)
    after=s.copy();after[0]+=.0001
    h.observe(s,np.zeros(7),after,pending=result['pending'],experiment_index=0)
    assert h.rows=={} and h.acknowledged_actions==0


@pytest.mark.parametrize('fault', ['before','action','experiment','oversize'])
def test_only_exact_acknowledgement_creates_record(fault):
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d,experiment=0);before=s.copy();index=0
    action=np.r_[d/.05,0.,0.,0.,0.]
    if fault=='before':before[0]+=.00001
    if fault=='action':action[:3]=[0,.02,0]
    if fault=='experiment':index=1
    if fault=='oversize':action[0]*=1.001
    after=before.copy();after[:3]+=action[:3]*.05
    observed=h.observe(before,action,after,pending=result['pending'],experiment_index=index)
    assert not observed['matched'] and h.rows=={}


def test_declared_evidence_context_change_preserves_history_debt_and_requires_new_holdouts():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d);s,_=acknowledge(h,r,s,d,result,gain=.2)
    old=copy.deepcopy(h.rows[0])
    changed=binding(s,d,representation=('tool2','object2'))
    result=assess(h,r,s,d,experiment=0,used=.0002,shared=.0002,bind=changed)
    assert result['accepted'] and result['evidence_segment_changed']
    assert not result['qualification']['ready']
    assert h.rows[0]==old # Declaration alone changes nothing.
    s,_=acknowledge(h,r,s,d,result,gain=.2)
    assert h.rows[0]['segments']==2 and h.rows[0]['measured_path_m']==pytest.approx(.0004)
    assert len(h.rows[0]['recent_robot_observations'])==2
    assert not h.rows[0]['last_qualification']['ready']
    # An explicit new fixed-goal model intent can use the same remaining
    # physical row; this is a new evidence context, not an extra allowance.
    changed=binding(s,d,intent='second',goal=(.03,.01,1.),geometry='new-model-geometry')
    result=assess(h,r,s,d,experiment=0,used=.0004,shared=.0191,bind=changed)
    assert not result['accepted'] and result['reason'].endswith('no_useful_command')
    result=assess(h,r,s,d,experiment=0,used=.0004,shared=.0004,bind=changed)
    assert result['accepted'] and not result['qualification']['ready']


def test_same_intent_goal_rounding_compares_immutable_origin_and_rejects_creep():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d);s,_=acknowledge(h,r,s,d,result,gain=.2)
    original=copy.deepcopy(h.rows[0]['context'])
    for offset in (1e-12,1.9e-12):
        b=binding(s,d,goal=(.04+offset,0,1))
        result=assess(h,r,s,d,experiment=0,bind=b)
        assert result['accepted']
        assert result['pending']['context']['fixed_goal_xyz_m']==original['fixed_goal_xyz_m']
    b=binding(s,d,goal=(.04+3e-12,0,1))
    assert assess(h,r,s,d,experiment=0,bind=b)['reason'].endswith('geometry_or_domain_changed')
    b=binding(s,d,geometry='changed')
    assert assess(h,r,s,d,experiment=0,bind=b)['reason'].endswith('geometry_or_domain_changed')


@pytest.mark.parametrize('fault', ['role','route','contact','endpoint','state','core','target','dependency'])
def test_current_binding_is_required_and_does_not_mutate_preview(fault):
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0]);b=binding(s,d)
    if fault=='role':b['empty_tool']=False
    if fault=='route':b['route_family']='reverse_observed'
    if fault=='contact':b['target_contact']='fingers'
    if fault=='endpoint':b['end_xyz_m'][1]=.001
    if fault=='state':b['state8'][0]=.001
    if fault=='core':b['hard_segment_clear']=False
    if fault=='target':b['outside_target_contact_band']=False
    if fault=='dependency':b['contact_or_visual_goal_dependency']=True
    before=pickle.dumps((h,r))
    assert not assess(h,r,s,d,bind=b)['accepted']
    assert pickle.dumps((h,r))==before


def test_all_warned_objects_need_current_heading_validation_and_contradiction_stops():
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    valid={'validated':True,'held_out_count':2,'current_supported_heading':True,'recent_measured_sample':True}
    b=binding(s,d);b['warning_ids']=['neighbor','second']
    result=assess(h,r,s,d,bind=b,evidence={'neighbor':valid})
    assert result['accepted'] and result['qualification']['awaiting_object_ids']==['second']
    assert not result['qualification']['ready'] # Robot direction is still cold too.
    result=assess(h,r,s,d,bind=b,evidence={'neighbor':valid,'second':{'contradicted':True}})
    assert not result['accepted'] and result['reason'].endswith('evidence_contradicted')


def test_real_old_heading_validation_cannot_be_copied_to_new_heading():
    witness=_NeighborResponse(1);s=state()
    for i in range(9):
        s[0]=i*.001;witness.observe(snapshot(s),'neighbor')
    assert witness.evidence(np.array([.001,0,0]),s[:3])['validated']
    evidence=witness.evidence(np.array([0,.001,0]),s[:3])
    assert not evidence['validated'] and evidence['held_out_count']==0
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();d=np.array([0,.001,0])
    old=pickle.dumps(witness)
    result=assess(h,r,s,d,evidence={'neighbor':evidence})
    assert result['accepted'] and not result['qualification']['ready']
    assert pickle.dumps(witness)==old and h.rows=={}


@pytest.mark.parametrize('kw', [dict(used=.008),dict(shared=.020),dict(remaining=0.),dict(actions=0)])
def test_no_debt_or_intent_allowance_is_replenished(kw):
    h=HeadingEvidenceAcquisitionV28();r=MeasuredTranslationResponseV25();s=state();d=np.array([.001,0,0])
    result=assess(h,r,s,d,**kw)
    assert not result['accepted'] and result['reason'].endswith('budget_exhausted')
    assert h.rows=={}


def test_actual_guard_cold_cap_reselection_preview_and_parent_ack(monkeypatch):
    """Inject only the legacy warning trigger; use actual geometry and tracking.

    This checks the helper/guard transaction, not whether a synthetic warning
    itself would have occurred in the fixture's preceding nonexistent motion.
    """
    from gpt_field.visual_translation_guard_v28 import ProjectedTranslationGuardV28
    from test_visual_translation_guard_v27 import setup
    from test_visual_translation_guard_v21 import pixels
    guard,field,s,command,kwargs=setup(monkeypatch)
    monitor=kwargs['monitor']
    del monitor.cumulative_response_evidence
    del monitor._external_reliable
    guard.__class__=ProjectedTranslationGuardV28
    for key,value in ProjectedTranslationGuardV28().__dict__.items():
        if key not in guard.__dict__:guard.__dict__[key]=value
    robot=MeasuredTranslationResponseV25();kwargs['translation_response']=robot
    images={'external':pixels(),'wrist':pixels()}
    guard._neighbor_responses={}  # Replace the older test's evidence stub with actual estimators.
    guard.configure(images,copy.deepcopy(guard.context),s,target_id=guard.target_id,monitor=monitor)
    current=guard._usable_snapshot()
    response_key=current['neighbors']['neighbor']['response_representation']
    guard._neighbor_responses['neighbor']=_NeighborResponse(response_key)
    guard._neighbor_responses['neighbor'].observe(current,'neighbor')
    kwargs['nominal_regions']=field.nominal_regions_for_segment(s[:3],s[:3]+command)
    first=guard.assess(command,**kwargs)
    assert first['accepted'] and first['cap_m']==pytest.approx(.001), first['reason']
    assert first['reason'].endswith('candidate_requires_cap_reselection')
    assert guard._heading_pending is None and guard._experiment_pending is None
    short=command/np.linalg.norm(command)*first['cap_m']
    kwargs['nominal_regions']=field.nominal_regions_for_segment(s[:3],s[:3]+short)
    before_helper=pickle.dumps(guard.heading_acquisition)
    before_pending=copy.deepcopy(guard._heading_pending)
    preview=guard.preview_assess(short,**kwargs)
    assert preview['accepted'] and preview['reason'].endswith('admitted_cold_command')
    assert pickle.dumps(guard.heading_acquisition)==before_helper
    assert guard._heading_pending==before_pending and guard.experiments==[]
    result=guard.assess(short,**kwargs)
    assert result['accepted'] and guard._heading_pending is not None
    assert guard.observation_travel_m==0 and guard.experiments==[]
    after=s.copy();after[:3]+=short*.2
    action=np.r_[short/.05,0.,0.,0.,0.]
    images={'external':pixels(-1),'wrist':pixels(-1)}
    monitor.observe(images,after,action)
    guard.observe(images,after,action,monitor=monitor)
    robot.observe(s,action,after)
    assert guard.heading_acquisition.last_observation['matched']
    assert guard.heading_acquisition.acknowledged_actions==1
    assert guard.heading_acquisition.rows[0]['measured_path_m']==pytest.approx(.0002)
    assert len(guard.experiments)==1
    assert guard.experiments[0]['measured_travel_m']==pytest.approx(.0002)
    assert guard.observation_travel_m==pytest.approx(.0002)
    assert guard.worsening=={'neighbor'}
