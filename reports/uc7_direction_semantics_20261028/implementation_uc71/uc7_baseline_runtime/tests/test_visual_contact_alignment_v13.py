"""Contact-image Field goals using real persistent RGB trackers and a test plant."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle
from gpt_field.visual_contact_alignment_v13 import VisualContactAlignmentV13


HAND=np.array([60.,70.]);EXT_TARGET=np.array([106.,116.]);WRIST_TARGET=np.array([150.,100.])
TARGET_STATE=np.array([.018,.012,.010])
TEXTURES=[np.random.default_rng(880+i).integers(0,256,(31,31,3),dtype=np.uint8) for i in range(3)]
TOOL_ROI=np.array([40.,50.,126.,132.])/223


def external_delta(p):
    x,y,z=p
    return np.array([700*x+200*z,100*y+600*z])


def wrist_delta(p):
    x,y,z=p
    return np.array([-450*x+150*y,100*x-500*y+100*z])


def render(position):
    ext=np.full((224,224,3),35,dtype=np.uint8);wrist=np.full_like(ext,55)
    for image,point,texture in [(ext,HAND+external_delta(position),TEXTURES[0]),
                                (ext,EXT_TARGET,TEXTURES[1]),
                                (wrist,WRIST_TARGET+wrist_delta(position),TEXTURES[2])]:
        x,y=np.rint(point).astype(int)
        image[y-15:y+16,x-15:x+16]=texture
    return ext,wrist


def box(point):return np.r_[point-15,point+15]/223


def fixture():
    ext,wrist=render(np.zeros(3));ex=_gray(ext);wr=_gray(wrist)
    monitor=SimpleNamespace(refs={'external':{'moving_kind':'tool','target_id':'part'},
                                  'wrist':{'target_id':'part'}},tracks={
       'moving':LocalFeatureBundle(ex,HAND/223,'tool',roi_uv=box(HAND),support_radius_px=16),
       'target':LocalFeatureBundle(ex,EXT_TARGET/223,'part',roi_uv=box(EXT_TARGET),support_radius_px=16),
       'wrist_target':LocalFeatureBundle(wr,WRIST_TARGET/223,'part',roi_uv=box(WRIST_TARGET),support_radius_px=16)})
    descriptor={'update':'seed','target_id':'part',
       'external':{'target_contact_point_uv':(EXT_TARGET/223).tolist(),
                   'tool_pinch_point_uv':((EXT_TARGET-external_delta(TARGET_STATE))/223).tolist()},
       'wrist':{'target_contact_point_uv':(WRIST_TARGET/223).tolist(),
                'tool_pinch_point_uv':((WRIST_TARGET+wrist_delta(TARGET_STATE))/223).tolist()}}
    module=VisualContactAlignmentV13();state=np.r_[np.zeros(6),.04,-.04]
    module.configure(descriptor,monitor,state,tool_roi_uv=TOOL_ROI)
    assert module.active,module.summary()
    return module,monitor,state,descriptor


def advance(module,monitor,state,position,*,fixed_images=False):
    ext,wr=render(np.zeros(3) if fixed_images else position)
    assert monitor.tracks['moving'].track(_gray(ext))
    assert monitor.tracks['target'].track(_gray(ext))
    assert monitor.tracks['wrist_target'].track(_gray(wr))
    after=state.copy();after[:3]=position
    module.observe(monitor,after,[.04,0,0,0,0,0,0])
    return after


def train(module,monitor,state,axes=(0,1,2)):
    # Actual task-like, non-collinear travel, not externally supplied calibration.
    for repeat in range(max(4,int(np.ceil(6/len(axes))))):
        for axis in axes:
            point=state[:3].copy();point[axis]+=.002
            state=advance(module,monitor,state,point)
    return state


def test_contact_midpoint_uses_whole_tool_bound_and_propagates_extrapolation_uncertainty():
    module,monitor,state,descriptor=fixture()
    assert np.linalg.norm(module.current['points']['tool']-monitor.tracks['moving'].position)>30
    assert module.current['point_uncertainty_px']['tool']>.25
    assert module.summary()['image_error_px'] is not None
    assert not module.summary()['automatic_close']
    assert not module.summary()['grasp_certified']


def test_real_two_view_motion_estimates_contact_goal_and_reduces_error():
    module,monitor,state,descriptor=fixture();state=train(module,monitor,state)
    assert module.response_ready,module.summary()
    assert module.model.rank==3
    initial=np.linalg.norm(module.current['error'])
    goal=state[:3]+[.015,-.01,.012]  # An imperfect model-chosen XYZ endpoint.
    errors=[]
    for _ in range(14):
        result=module.correction(goal)
        if result['reason']=='contact_alignment_observed':break
        if result['accepted']:
            proposed=np.asarray(result['goal_xyz'])
            assert np.linalg.norm(proposed-goal)<=.006+1e-12
            goal=proposed
        delta=goal-state[:3]
        if np.linalg.norm(delta)<1e-6:break
        delta*=min(1.,.0015/np.linalg.norm(delta))
        state=advance(module,monitor,state,state[:3]+delta)
        errors.append(float(np.linalg.norm(module.current['error'])))
    assert errors and min(errors)<.6*initial,(initial,errors,module.summary())
    assert np.linalg.norm(state[:3]-TARGET_STATE)<.008
    assert not module.summary()['grasp_certified']


def test_supported_one_axis_can_correct_without_full_rank_or_probe_loop():
    module,monitor,state,descriptor=fixture();state=train(module,monitor,state,axes=(2,))
    assert module.response_ready,module.summary()
    assert module.model.rank==1
    goal=state[:3]+[.01,.02,.01]
    result=module.correction(goal,allowed_axes='z')
    assert result['accepted'],result
    assert np.asarray(result['goal_xyz'])[:2]==pytest.approx(goal[:2])
    assert module.correction(goal,allowed_axes='xy')['reason']=='contact_alignment_axes_not_observed'


def test_retain_preserves_points_fit_and_rejects_new_coordinates():
    module,monitor,state,descriptor=fixture();state=train(module,monitor,state)
    rows=len(module.model.samples);before=module.current['points']['tool'].copy()
    retained={'update':'retain','target_id':'part','external':None,'wrist':None}
    module.configure(retained,monitor,state,tool_roi_uv=TOOL_ROI)
    assert module.active and len(module.model.samples)==rows
    assert module.current['points']['tool']==pytest.approx(before)
    retained['external']=descriptor['external']
    with pytest.raises(ValueError):module.configure(retained,monitor,state)


@pytest.mark.parametrize('change',['finger','rotation','lost','replaced'])
def test_changed_contact_domain_never_reuses_previous_pad_goal(change):
    module,monitor,state,descriptor=fixture();state=train(module,monitor,state)
    if change=='finger':state[6]-=.004
    if change=='rotation':state[3]=.05
    if change=='lost':monitor.tracks['target'].reliable=False
    if change=='replaced':monitor.tracks['moving']=copy.copy(monitor.tracks['moving'])
    module.observe(monitor,state,[0]*7)
    assert not module.active
    assert not module.correction(state[:3])['accepted']
    assert not module.ready_for_model_review


def test_stationary_images_are_not_a_supported_measured_response():
    module,monitor,state,descriptor=fixture()
    for i in range(1,12):
        state=advance(module,monitor,state,[i*.002,0,0],fixed_images=True)
    assert not module.response_ready
    assert not module.correction(state[:3]+[0,0,.01])['accepted']


def test_target_contact_must_stay_inside_its_declared_object_support():
    module,monitor,state,descriptor=fixture()
    descriptor['external']['target_contact_point_uv']=[.95,.95]
    module.configure(descriptor,monitor,state,tool_roi_uv=TOOL_ROI)
    assert not module.active
    assert module.status=='contact_alignment_point_outside_declared_support'


def test_wrong_target_or_payload_role_is_not_grasp_alignment_evidence():
    module,monitor,state,descriptor=fixture()
    monitor.refs['external']['moving_kind']='payload'
    module.configure(descriptor,monitor,state,tool_roi_uv=TOOL_ROI)
    assert not module.active and module.status=='contact_alignment_role_mismatch'


def test_axis_intersection_does_not_invent_unobserved_coordinate_direction():
    basis=np.array([[1.,1.,0.]])/np.sqrt(2.)
    assert VisualContactAlignmentV13._intersection(basis,'x').shape==(3,0)
    assert VisualContactAlignmentV13._intersection(basis,'xy').shape==(3,1)


def test_fresh_two_view_alignment_hands_back_for_review_instead_of_deeper_goal():
    module,monitor,state,descriptor=fixture();state=train(module,monitor,state)
    while np.linalg.norm(TARGET_STATE-state[:3])>.0001:
        step=TARGET_STATE-state[:3];step*=min(1.,.0015/np.linalg.norm(step))
        state=advance(module,monitor,state,state[:3]+step)
    for _ in range(3):state=advance(module,monitor,state,state[:3])
    assert module.ready_for_model_review
    # An older model-selected depth cannot authorize additional descent after
    # the fresh image goal is reached. Root holds and requests model review.
    result=module.correction(state[:3]+[0,0,.01])
    assert not result['accepted'] and result['goal_xyz'] is None
    assert result['reason']=='contact_alignment_observed'
    assert not result['diagnostics']['automatic_close']
    assert not result['diagnostics']['grasp_certified']
