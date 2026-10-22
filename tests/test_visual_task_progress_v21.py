"""Objective-progress contracts, independent of visual Jacobian readiness.

These numerical unit fixtures supply explicit tracked pixel observations. They
do not stand in for tracker validation or simulator task success.
"""
from types import SimpleNamespace
import copy
import numpy as np

from gpt_field.visual_task_progress_v21 import VisualTaskProgressV21


def fixture():
    def track(x):
        return SimpleNamespace(position=np.array(x, float), scale=np.array([223.,223.]),
            reliable=True, last_residual_px=.15, last_inliers=6)
    monitor = SimpleNamespace(tracks={"moving":track([100.,80.]),"target":track([80.,80.])},
        response_ready=False, _external_reliable=lambda: True)
    pred = {"target_id":"object", "phase":"approach", "visual_references":{"external":{
        "moving_kind":"tool", "target_id":"object", "goal_offset_uv":[0.,0.]}}}
    state = np.array([0.,0.,1.,0.,0.,0.,.04,-.04]); goal=np.array([.06,0.,1.])
    ledger=VisualTaskProgressV21();ledger.configure(state,monitor,pred,goal,step=0)
    return ledger,monitor,pred,state,goal


def advance(ledger,monitor,pred,state,goal, *, shift=.35, straight=True, accepted=True, step=1):
    after=state.copy();after[0]+=.003
    monitor.tracks['moving'].position += [shift,0.]
    reason=ledger.observe(state,after,[.06,0,0,0,0,0,0],monitor=monitor,prediction=pred,
        goal_xyz=goal,accepted_field=accepted,straight_route=straight,step=step)
    return after,reason


def test_worsening_declared_visual_objective_reviews_without_claiming_response_or_task_validity():
    ledger,monitor,pred,state,goal=fixture();reasons=[]
    for i in range(8):
        state,r=advance(ledger,monitor,pred,state,goal,step=i+1)
        if r:reasons.append(r)
    assert reasons==['metric_goal_progress_disagrees_with_declared_image_objective']
    info=ledger.summary()['active']
    assert info['metric_improvement_m']>.02 and info['image_error_change_px']>2.
    assert not monitor.response_ready and not info['task_success_certified']


def test_retained_reference_and_world_goal_do_not_reset_measured_evidence():
    ledger,monitor,pred,state,goal=fixture()
    for i in range(4):state,_=advance(ledger,monitor,pred,state,goal,step=i+1)
    before=copy.deepcopy(ledger.active)
    ledger.configure(state,monitor,copy.deepcopy(pred),goal.copy(),step=4)
    assert ledger.active==before
    state,reason=advance(ledger,monitor,pred,state,goal,step=5)
    assert reason and ledger.active['samples']==5


def test_temporary_direct_hold_does_not_erase_pending_objective_review():
    ledger,monitor,pred,state,goal=fixture()
    for i in range(6):state,_=advance(ledger,monitor,pred,state,goal,step=i+1)
    before=copy.deepcopy(ledger.active)
    direct=copy.deepcopy(pred);direct['phase']='recover'
    ledger.configure(state,monitor,direct,None,step=6)
    ledger.configure(state,monitor,pred,goal,step=7)
    assert ledger.active==before and ledger.active['review_required']


def test_reference_change_archives_incompatible_objective_instead_of_summing_pixels():
    ledger,monitor,pred,state,goal=fixture()
    for i in range(6):state,_=advance(ledger,monitor,pred,state,goal,step=i+1)
    prior=copy.deepcopy(ledger.active)
    monitor.tracks['moving']=copy.deepcopy(monitor.tracks['moving'])
    monitor.tracks['moving'].position += [10.,-8.]
    ledger.configure(state,monitor,pred,goal,step=6)
    assert ledger.intervals[-1]==prior and ledger.active['samples']==0
    assert ledger.summary()['prior_conflicting_intervals']==1
    assert not ledger.summary()['pixel_errors_summed_across_references']


def test_approaching_visual_objective_does_not_request_repair():
    ledger,monitor,pred,state,goal=fixture()
    for i in range(10):
        state,reason=advance(ledger,monitor,pred,state,goal,shift=-.35,step=i+1)
        assert reason is None
    assert ledger.active['image_error_change_px']<0.


def test_detour_and_rejected_commands_do_not_create_conflict_evidence():
    for mode in ['detour','rejected']:
        ledger,monitor,pred,state,goal=fixture()
        for i in range(8):
            state,reason=advance(ledger,monitor,pred,state,goal,
                straight=mode!='detour',accepted=mode!='rejected',step=i+1)
            assert reason is None
        assert not ledger.active['review_required']


def test_lost_reference_noise_and_shifted_anchor_cannot_trigger_goal_revision():
    for mode in ['lost','noise','anchor']:
        ledger,monitor,pred,state,goal=fixture()
        for i in range(8):
            if mode=='lost':monitor.tracks['moving'].reliable=False
            elif mode=='noise':monitor.tracks['moving'].last_residual_px=4.
            else:monitor.tracks['target'].position += [7.,0.]
            state,reason=advance(ledger,monitor,pred,state,goal,step=i+1)
            assert reason is None
        assert not ledger.active['review_required']
