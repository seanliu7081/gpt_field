"""Measured-robot-only response counterexamples and saved V11 trace replay."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gpt_field.robot_translation_response import MeasuredTranslationResponse


def state():return np.array([0.,0.,1.,0.,0.,0.,.04,-.04])
JOINTS=np.array([0.,-.5,0.,-1.,0.,1.,0.])


def feed(model,before,command_m,measured_m,*,joints=None,after_joints=None):
    after=before.copy();after[:3]+=measured_m
    action=np.r_[np.asarray(command_m)/.05,0.,0.,0.,0.]
    result=model.observe(before,action,after,before_joints=joints,
                         after_joints=joints if after_joints is None else after_joints)
    return after,result


def calibrated(gain=.22,*,command_m=(.002,0.,0.),joints=None):
    model=MeasuredTranslationResponse();before=state();command=np.array(command_m)
    for _ in range(5):before,_=feed(model,before,command,gain*command,joints=joints)
    return model,before


def test_command_and_measured_units_are_distinct_and_supported_2mm_command_fits_650um_observed_step_cap():
    model,current=calibrated()
    prediction=model.predict([.002,0.,0.],current)
    assert prediction['supported'] and prediction['responsive']
    assert prediction['mean_delta_m']==pytest.approx([.00044,0.,0.])
    assert .00044<prediction['max_measured_norm_m']<.00065
    cap=model.command_cap_for_measured_limit([1.,0.,0.],.00065,.002,current)
    assert cap['supported'] and not cap['cold_fallback']
    assert cap['command_cap_m']==pytest.approx(.002)  # command remains above the 1mm action floor.
    assert cap['command_cap_m']>.00065  # These two numbers intentionally have different meanings.
    assert cap['prediction']['max_measured_norm_m']<=.00065
    assert not cap['guaranteed_bound'] and not cap['clearance_certified']


def test_measured_cap_inversion_reduces_only_as_needed_and_never_amplifies_request_or_extrapolates_scale():
    model,current=calibrated()
    result=model.command_cap_for_measured_limit([3.,0.,0.],.0004,.002,current)
    assert result['supported'] and .001<result['command_cap_m']<.002
    assert result['prediction']['max_measured_norm_m']<=.0004
    larger=model.command_cap_for_measured_limit([1.,0.,0.],.005,.015,current)
    assert larger['supported'] and larger['command_cap_m']==pytest.approx(.003)
    assert larger['prediction']['bounded_scale_probe']
    assert larger['prediction']['scale_probe_margin_m']>0.
    assert larger['command_cap_m']<=1.5*max(s['command_norm']for s in model.samples)
    tiny=model.command_cap_for_measured_limit([1.,0.,0.],.0001,.002,current)
    assert tiny['command_cap_m']==0. and not tiny['supported'] and not tiny['cold_fallback']
    assert tiny['reason']=='measured_limit_below_empirical_command_support'


def test_interval_covers_observed_variability_lateral_response_and_fresh_held_out_sample():
    model=MeasuredTranslationResponse();current=state()
    for gain,lateral in [(.17,.000012),(.21,-.000020),(.23,.000015),(.20,-.000014),(.22,.00001)]:
        current,_=feed(model,current,[.002,0.,0.],[.002*gain,lateral,0.])
    before_summary=copy.deepcopy(model.summary())
    predicted=model.predict([.002,0.,0.],current)
    future=np.array([.00045,.000019,-.000008])
    assert predicted['longitudinal_interval_m'][0]<=future[0]<=predicted['longitudinal_interval_m'][1]
    assert np.linalg.norm(future[1:])<=predicted['lateral_radius_m']
    assert np.linalg.norm(future)<=predicted['max_measured_norm_m']
    assert np.all(future>=predicted['measured_component_lower_m'])
    assert np.all(future<=predicted['measured_component_upper_m'])
    for _ in range(5):model.predict([.002,0.,0.],current)
    assert model.summary()==before_summary  # prediction cannot create calibration evidence.


@pytest.mark.parametrize('command',[[0.,.002,0.],[-.002,0.,0.],[.002*math.cos(.4),.002*math.sin(.4),0.]])
def test_unobserved_or_opposed_command_direction_cannot_borrow_calibration(command):
    model,current=calibrated()
    predicted=model.predict(command,current)
    assert not predicted['supported'] and predicted['sample_count']==0
    assert predicted['max_measured_norm_m'] is None
    inverse=model.command_cap_for_measured_limit(command,.0006,.002,current)
    assert not inverse['supported'] and inverse['cold_fallback']
    assert inverse['command_cap_m']<=.0006
    assert 'not certified' in inverse['cold_assumption']


@pytest.mark.parametrize('command',[[.0031,0.,0.],[.0004,0.,0.]])
def test_observed_direction_does_not_imply_arbitrary_command_scale_support(command):
    model,current=calibrated()
    predicted=model.predict(command,current)
    assert not predicted['supported'] and predicted['reason']=='robot_response_command_scale_unsupported'
    assert predicted['command_scale_range_m']==pytest.approx([.001,.003])


def test_stationary_and_drifting_holds_are_excluded_from_motion_fit_and_drift_increases_uncertainty():
    model,current=calibrated()
    original=model.predict([.002,0.,0.],current)
    for _ in range(4):
        current,out=feed(model,current,[0.,0.,0.],[0.,.00012,0.])
        assert out['is_hold'] and not out['sample_added']
    summary=model.summary()
    assert summary['samples']==5 and summary['holds_excluded_from_fit']==4
    assert summary['hold_drift_margin_m']==pytest.approx(.00012)
    after=model.predict([.002,0.,0.],current)
    assert after['mean_delta_m']==original['mean_delta_m']
    assert after['max_measured_norm_m']>original['max_measured_norm_m']
    held=model.predict([0.,0.,0.],current)
    assert not held['supported'] and held['reason']=='zero_command_not_response_evidence'


def test_recent_robot_blockage_or_reversed_response_never_inverts_to_larger_commands():
    model,current=calibrated()
    for _ in range(3):current,_=feed(model,current,[.002,0.,0.],[0.,0.,0.])
    stalled=model.predict([.002,0.,0.],current)
    assert not stalled['supported'] and stalled['responsive'] is False
    result=model.command_cap_for_measured_limit([1.,0.,0.],.01,.01,current)
    assert result['command_cap_m']==0. and not result['cold_fallback']
    opposed,current=calibrated(-.2)
    result=opposed.command_cap_for_measured_limit([1.,0.,0.],.01,.002,current)
    assert result['command_cap_m']==0. and result['reason']=='robot_response_unresponsive_or_opposed'


@pytest.mark.parametrize('change,reason',[
    ('orientation','robot_response_orientation_domain_changed'),
    ('fingers','robot_response_finger_domain_changed'),
    ('position','robot_response_position_domain_changed'),
    ('joints','robot_response_joint_domain_changed'),
])
def test_local_robot_configuration_changes_expire_response_without_transferring_old_evidence(change,reason):
    model,current=calibrated(joints=JOINTS)
    changed=current.copy();joints=JOINTS.copy()
    if change=='orientation':changed[5]+=.02
    elif change=='fingers':changed[6]-=.003
    elif change=='position':changed[1]+=.041
    else:joints[2]+=.121
    predicted=model.predict([.002,0.,0.],changed,joint_position=joints)
    assert not predicted['supported'] and predicted['reason']==reason
    assert len(model.samples)==5  # query is read-only; actual transition changes the domain.
    result=model.observe(current,[0.]*7,changed,before_joints=JOINTS,after_joints=joints)
    assert result['domain_reset_reason']==reason
    assert len(model.samples)==0 and model.summary()['domain_resets']==1
    assert not model.predict([.002,0.,0.])['supported']


def test_near_pi_axis_angle_sign_wrap_is_same_robot_orientation_not_domain_reset():
    model=MeasuredTranslationResponse();current=state();current[3]=math.pi-1e-7
    for _ in range(4):current,_=feed(model,current,[.002,0.,0.],[.0004,0.,0.])
    wrapped=current.copy();wrapped[3]=-math.pi-1e-7
    result=model.observe(current,[0.]*7,wrapped)
    assert result['domain_reset_reason'] is None
    assert model.summary()['domain_resets']==0 and model.predict([.002,0.,0.])['supported']


def test_missing_joint_feedback_and_observation_discontinuities_invalidate_old_model():
    model,current=calibrated(joints=JOINTS)
    # Dropping paired joint feedback is not interpreted as proving the same domain.
    after=current.copy();after[0]+=.0004
    result=model.observe(current,[.04,0.,0.,0.,0.,0.,0.],after)
    assert result['domain_reset_reason']=='robot_response_joint_feedback_domain_changed'
    assert len(model.samples)==1 and not model.predict([.002,0.,0.])['supported']
    jumped=after.copy();jumped[1]+=.005
    more=jumped.copy();more[0]+=.0004
    result=model.observe(jumped,[.04,0.,0.,0.,0.,0.,0.],more)
    assert result['domain_reset_reason']=='robot_response_observation_discontinuity'
    assert len(model.samples)==1


def test_cold_fallback_is_bounded_explicitly_uncalibrated_and_does_not_create_samples():
    model=MeasuredTranslationResponse()
    for remaining,requested in [(.01,.01),(.0007,.002),(0.,.002),(.01,0.)]:
        cap=model.command_cap_for_measured_limit([1.,1.,0.],remaining,requested,state())
        assert not cap['supported'] and cap['command_cap_m']<=min(remaining,requested,.002)
        assert not cap['guaranteed_bound'] and not cap['clearance_certified']
    assert model.summary()['samples']==0 and model.summary()['transitions']==0


def test_v11_archived_robot_transitions_support_2mm_command_within_observed_650um_response_scale():
    trace=Path(__file__).resolve().parents[1]/'runs/visual_field_v11_zero_shot_001/trials/bowl_original_zero_hybrid_v11/trace.jsonl'
    if not trace.exists():pytest.skip('Archived V11 trace unavailable')
    rows=[json.loads(x)for x in trace.read_text().splitlines()]
    reset=next(x for x in rows if x['event']=='reset')['observation']
    actions={x['step']:x for x in rows if x['event']=='action'}
    observations={x['step']:x['observation']for x in rows if x['event']=='executed'}
    model=MeasuredTranslationResponse();before=reset;moving=0;maximum_step=0.;predictions=[]
    for index,after in observations.items():
        action=actions[index]['action'];command=np.array(action[:3])*.05
        if np.linalg.norm(command)>1e-8:
            moving+=1
            prediction=model.predict(command)
            if prediction['supported']:
                actual=np.array(after['observation/state'][:3])-before['observation/state'][:3]
                predictions.append((np.linalg.norm(actual),prediction['max_measured_norm_m']))
        result=model.observe(before['observation/state'],action,after['observation/state'],
            before_joints=before['observation/joint_position'],after_joints=after['observation/joint_position'])
        maximum_step=max(maximum_step,result['measured_norm_m'] if result['sample_added'] else 0.)
        before=after
    assert moving==11 and model.summary()['nonzero_samples']==11
    assert model.summary()['domain_resets']==0
    assert len(predictions)>=4 and all(actual<=bound+1e-12 for actual,bound in predictions)
    # This is the actual V11 +Y/+Z separation direction, not a task-space pose.
    predicted=model.predict(np.array([0.,1.,1.])/math.sqrt(2)*.002)
    assert predicted['supported'],predicted
    assert predicted['max_measured_norm_m']<1.5*maximum_step
    cap=model.command_cap_for_measured_limit([0.,1.,1.],1.5*maximum_step,.002)
    assert cap['supported'] and cap['command_cap_m']==pytest.approx(.002)
    assert cap['command_cap_m']>=.001  # command floor and measured support are not conflated.
    assert model.summary()['environment_inputs']==[]


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,-1.])
def test_invalid_limits_are_not_silently_clamped_into_commands(bad):
    model=MeasuredTranslationResponse()
    with pytest.raises(ValueError):model.command_cap_for_measured_limit([1.,0.,0.],bad,.002,state())


def test_invalid_robot_feedback_and_action_ranges_reject_without_mutating_model():
    model,current=calibrated();saved=copy.deepcopy(model.summary())
    with pytest.raises(ValueError):model.observe(current,[2.,0.,0.,0.,0.,0.,0.],current)
    with pytest.raises(ValueError):model.observe(current,[0.]*7,current,before_joints=JOINTS)
    wrong=current.tolist();wrong[0]=True
    with pytest.raises(ValueError):model.predict([.002,0.,0.],wrong)
    with pytest.raises(ValueError):model.predict([.051,0.,0.],current)
    assert model.summary()==saved
    json.dumps(model.summary(),allow_nan=False)


def test_scale_probe_growth_requires_new_acknowledged_command_not_repeated_requests():
    model,current=calibrated()
    for _ in range(5):
        cap=model.command_cap_for_measured_limit([1.,0.,0.],.01,.005,current)
        assert cap['command_cap_m']==pytest.approx(.003)
    current,_=feed(model,current,[.003,0.,0.],[.00066,0.,0.])
    cap=model.command_cap_for_measured_limit([1.,0.,0.],.01,.005,current)
    assert cap['supported'] and cap['command_cap_m']==pytest.approx(.0045)
    current,_=feed(model,current,[.0045,0.,0.],[.00099,0.,0.])
    cap=model.command_cap_for_measured_limit([1.,0.,0.],.01,.005,current)
    assert cap['supported'] and cap['command_cap_m']==pytest.approx(.005)


def test_old_transient_hold_drift_does_not_remain_lifetime_uncertainty_after_fresh_robot_response():
    model,current=calibrated()
    current,_=feed(model,current,[0.,0.,0.],[0.,.0002,0.])
    initial=model.predict([.002,0.,0.])['max_measured_norm_m']
    for _ in range(8):current,_=feed(model,current,[.002,0.,0.],[.00044,0.,0.])
    assert model.summary()['hold_drift_margin_m']==0.
    assert model.summary()['all_time_hold_drift_m']==pytest.approx(.0002)
    assert model.predict([.002,0.,0.])['max_measured_norm_m']<initial


def test_small_finger_noise_and_natural_local_joint_motion_do_not_reset_response():
    model,current=calibrated(joints=JOINTS)
    joints=JOINTS.copy()
    for _ in range(6):
        after=current.copy();after[0]+=.00044;after[6:]+=[-.00001,.00001]
        moved=joints.copy();moved[1]+=.002
        model.observe(current,[.04,0.,0.,0.,0.,0.,0.],after,before_joints=joints,after_joints=moved)
        current,joints=after,moved
    assert model.summary()['domain_resets']==0
    assert model.predict([.002,0.,0.])['supported']
