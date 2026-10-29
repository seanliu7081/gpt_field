"""Counterexamples for repeated relative goals and submillimeter cap starvation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from gpt_field.visual_field_v9 import MetricFieldExecutorV9
from gpt_field.visual_field_v10 import MetricFieldExecutorV10, SubgoalLedger
from test_visual_field_v4 import Q, box, state
from test_visual_field_v6 import axis_intent
from test_visual_field_v9 import assessment, halo_box


def prediction(action='new', *, ident='transport-one', controller='visual_field', value=None,
               phase='transport', target='destination', reason='Current RGB supports this goal'):
    return {'controller': controller, 'phase': phase, 'target_id': target,
            'field_intent': value or axis_intent('z', goal_offset_m=[0.,0.,.05]),
            'subgoal_update': {'action': action, 'subgoal_id': None if action=='none' else ident, 'reason': reason}}


def test_repeated_direct_and_gripper_repairs_resume_same_world_goal_not_another_relative_lift():
    ledger=SubgoalLedger('task'); current=state()
    created=ledger.apply(current,prediction());goal=created['feedback']['active']['world_goal_xyz']
    for controller,advance in [('direct',.009),('gripper',0.),('direct',.008)]:
        current[2]+=advance
        # Contact repair can concern the payload, even though the preserved
        # transport intent concerns the destination.
        result=ledger.apply(current,prediction('retain',controller=controller,phase='grasp',target='payload'))
        assert result['accepted'] and result['field_intent'] is None
        resumed=ledger.apply(current,prediction('retain'))
        assert resumed['accepted']
        assert np.asarray(resumed['field_intent']['goal_offset_m'])+current[:3] == pytest.approx(goal)
        assert resumed['field_intent']['goal_offset_m'][2] == pytest.approx(goal[2]-current[2])
        assert resumed['feedback']['last_update']['proposed_world_goal_xyz'][2] > goal[2]
        assert resumed['executor_context']['subgoal_origin_xyz']==state()[:3].tolist()
    assert ledger.summary()['explicit_revisions']==0


def test_direct_test_reaches_or_crosses_old_goal_and_does_not_authorize_another_lift():
    ledger=SubgoalLedger();ledger.apply(state(),prediction())
    near=state();near[2]+=.044;ledger.observe(near)
    after=state();after[2]+=.056
    ledger.apply(after,prediction('retain',controller='direct',phase='grasp',target='payload'))
    assert ledger.summary()['local_complete']
    assert ledger.summary()['active']['completion_reason']=='measured_transition_crossed_local_goal_region'
    resumed=ledger.apply(after,prediction('retain'))
    assert not resumed['accepted'] and resumed['status']=='subgoal_local_complete'
    assert resumed['field_intent'] is None
    assert not resumed['feedback']['task_success_certified']
    assert not resumed['feedback']['active']['task_success_certified']
    # Once reached, subsequent test motion does not silently reactivate it.
    after[2]+=.005;ledger.observe(after);assert ledger.summary()['local_complete']


def test_locked_axis_drift_does_not_count_as_local_arrival():
    ledger=SubgoalLedger();ledger.apply(state(),prediction())
    after=state();after[0]=.0031;after[2]+=.05
    ledger.observe(after)
    assert not ledger.summary()['local_complete']
    result=ledger.apply(after,prediction('retain'))
    assert not result['accepted'] and result['status']=='subgoal_locked_axis_drift_requires_replan'


def test_resume_holds_current_inactive_axis_but_preserves_original_corridor():
    ledger=SubgoalLedger();ledger.apply(state(),prediction())
    after=state();after[0]=.0028;after[2]+=.012
    resumed=ledger.apply(after,prediction('retain'))
    assert resumed['accepted']
    assert resumed['field_intent']['goal_offset_m'][:2]==[0.,0.]
    executor=MetricFieldExecutorV10()
    executor.configure(after,resumed['field_intent'],scene_assessment=assessment(),**resumed['executor_context'])
    selected=executor.select(after,Q);assert selected['accepted']
    assert selected['action'][:2]==pytest.approx([0.,0.],abs=1e-10)
    # This point is within the NEW pose-centered corridor, but outside the
    # retained original one; the continuous check must still exclude it.
    forbidden=after[:3].copy();forbidden[0]=.0032
    assert not executor.checked.segment_valid(after[:3],forbidden)
    drift=after.copy();drift[0]=.0031;drift[2]+=.001
    executor.observe(after,selected['action'],drift)
    assert executor.select(drift,Q)['status']=='locked_axis_pose_drift'


def test_out_of_grid_retain_does_not_clamp_or_replace_original_goal():
    ledger=SubgoalLedger();ledger.apply(state(),prediction())
    goal=copy.deepcopy(ledger.active['world_goal_xyz'])
    far=state();far[2]-=.04
    result=ledger.apply(far,prediction('retain'))
    assert not result['accepted'] and result['status']=='subgoal_outside_local_grid_requires_replan'
    assert ledger.active['world_goal_xyz']==goal and ledger.active['origin_xyz']==state()[:3].tolist()
    assert result['field_intent'] is None


def test_new_and_replace_require_explicit_stateful_contracts_and_are_auditable():
    ledger=SubgoalLedger();ledger.apply(state(),prediction())
    rejected=ledger.apply(state(),prediction('new',ident='another'))
    assert rejected['status']=='subgoal_active_requires_retain_or_explicit_replace'
    assert ledger.active['subgoal_id']=='transport-one'
    rejected=ledger.apply(state(),prediction('retain',phase='approach'))
    assert rejected['status']=='subgoal_semantics_changed_requires_explicit_replace'
    rejected=ledger.apply(state(),prediction('replace',ident='wrong'))
    assert rejected['status']=='subgoal_identity_mismatch'
    with pytest.raises(ValueError,match='explicit reason'):
        ledger.apply(state(),prediction('replace',reason=''))
    replaced=ledger.apply(state(),prediction('replace',phase='approach',target='new_target',
                         value=axis_intent('xy',goal_offset_m=[.02,.02,0.]),reason='Observed goal changed'))
    assert replaced['accepted'] and ledger.summary()['explicit_revisions']==1
    assert ledger.active['world_goal_xyz']==[.02,.02,1.]
    assert ledger.completed[-1]['world_goal_xyz']==[0.,0.,1.05]
    assert ledger.completed[-1]['superseded_reason']=='Observed goal changed'


def test_invalid_new_or_replace_intent_leaves_old_goal_unmodified():
    ledger=SubgoalLedger();ledger.apply(state(),prediction());before=ledger.summary()
    with pytest.raises(ValueError,match='16 cm grid'):
        ledger.apply(state(),prediction('replace',value=axis_intent('z',goal_offset_m=[0.,0.,.09])))
    assert ledger.summary()==before
    with pytest.raises(ValueError,match='locked motion axes'):
        ledger.apply(state(),prediction('replace',value=axis_intent('z',goal_offset_m=[.001,0.,.03])))
    assert ledger.summary()==before


def test_none_never_clears_active_goal_finish_requires_arrival_and_new_task_reset_clears():
    ledger=SubgoalLedger('task1');ledger.apply(state(),prediction())
    assert ledger.apply(state(),prediction('none',controller='direct'))['accepted']
    assert ledger.active is not None
    assert ledger.apply(state(),prediction('finish',controller='gripper'))['status']=='subgoal_finish_before_measured_arrival'
    reached=state();reached[2]+=.05
    assert ledger.apply(reached,prediction('finish',controller='direct'))['accepted']
    assert ledger.active is None and len(ledger.completed)==1
    result=ledger.apply(reached,prediction('new',ident='next_goal'))
    assert result['accepted']
    ledger.reset('task2')
    assert ledger.active is None and ledger.completed==[] and ledger.summary()['task_id']=='task2'
    assert ledger.apply(state(),prediction('retain'))['status']=='subgoal_missing'


@pytest.mark.parametrize('cap',[0.,.0001,.0002,.0009])
def test_submillimeter_cap_handoffs_without_rounding_up_or_counting_an_action(cap):
    executor=MetricFieldExecutorV10();executor.configure(state(),axis_intent(),scene_assessment=assessment())
    selected=executor.select(state(),Q,command_cap_m=cap)
    assert not selected['accepted'] and selected['status']=='field_command_below_useful_motion_floor'
    assert selected['action']==[0.]*7 and executor.pending is None
    assert executor.summary()['executed_actions']==0
    assert executor.summary()['orientation_hold']['osc_goal_refresh_commands']==0
    executor.observe(state(),measured_current_pose_hold_action(),state())
    assert executor.summary()['executed_actions']==0
    assert executor.select(state(),Q)['status']=='field_command_below_useful_motion_floor'


def test_one_millimeter_cap_is_respected_and_acknowledgement_remains_exact():
    executor=MetricFieldExecutorV10();executor.configure(state(),axis_intent(),scene_assessment=assessment())
    chosen=executor.select(state(),Q,command_cap_m=.001)
    assert chosen['accepted'] and np.linalg.norm(chosen['action'][:3])*.05<=.001+1e-12
    assert chosen['action'][3]==pytest.approx(2e-9,abs=1e-15)
    with pytest.raises(RuntimeError,match='acknowledge'):
        executor.select(state(),Q,command_cap_m=.0002)
    changed=chosen['action'].copy();changed[0]+=.001
    with pytest.raises(ValueError,match='match'):
        executor.observe(state(),changed,state())
    after=state();after[0]+=.00025;executor.observe(state(),chosen['action'],after)
    assert executor.summary()['executed_actions']==1


def test_budget_remainder_below_floor_stops_without_spending_remaining_distance():
    executor=MetricFieldExecutorV10();current=state()
    executor.configure(current,axis_intent(max_travel_m=.005,command_limit_m=.002),scene_assessment=assessment())
    for _ in range(3):
        selected=executor.select(current,Q);assert selected['accepted']
        after=current.copy();after[0]+=.0015;executor.observe(current,selected['action'],after);current=after
    stopped=executor.select(current,Q)
    assert not stopped['accepted'] and stopped['status']=='field_command_below_useful_motion_floor'
    assert executor.summary()['measured_travel_m']==pytest.approx(.0045)
    assert executor.summary()['executed_actions']==3


def test_local_arrival_has_priority_over_small_remaining_cap():
    executor=MetricFieldExecutorV10();current=state()
    executor.configure(current,axis_intent(goal_offset_m=[.002,0.,0.]),scene_assessment=assessment())
    assert executor.select(current,Q,command_cap_m=.0002)['status']=='measured_local_goal_arrival'


def test_low_response_transport_exits_before_recorded_long_starvation_tail():
    old,new=MetricFieldExecutorV9(),MetricFieldExecutorV10();a,b=state(),state()
    for executor in (old,new):executor.configure(state(),axis_intent(max_actions=120),scene_assessment=assessment())
    # Recorded command starvation can be replicated without inference or
    # simulator truth: a0.2mm cap and a small response below the evidence floor.
    first=new.select(b,Q,command_cap_m=.0002)
    assert not first['accepted']
    for _ in range(30):
        chosen=old.select(a,Q,command_cap_m=.0002);assert chosen['accepted']
        after=a.copy();after[:3]+=np.asarray(chosen['action'][:3])*.05*.002
        old.observe(a,chosen['action'],after);a=after
    assert old.summary()['executed_actions']==30
    assert old.summary()['measured_travel_m']<.0001
    assert new.summary()['executed_actions']==0


def test_v9_soft_costs_unknown_hard_boxes_axis_and_orientation_limits_stay_intact():
    executor=MetricFieldExecutorV10();value=axis_intent(obstacles=[halo_box()],uncertainty_m=.025,max_actions=120)
    executor.configure(state(),value,scene_assessment=assessment('visible'))
    chosen=executor.select(state(),Q);assert chosen['accepted']
    assert chosen['diagnostics']['plan']['search']['cost_unit']=='weighted_m'
    assert executor.summary()['grid_size']==33 and executor.summary()['max_actions']==120
    after=state();after[0]+=.002;after[5]=.02
    executor.observe(state(),chosen['action'],after)
    held=executor.select(after,Q);assert held['accepted'] and held['action'][5]<0.
    assert np.linalg.norm(held['action'][3:6])*.5<=.02+1e-12
    assert not executor.retarget(executor.goal+[0.,0.,.0001])
    executor.configure(state(),axis_intent(unknown_regions=[halo_box()],uncertainty_m=.025),scene_assessment=assessment('visible'))
    assert executor.select(state(),Q)['status']=='invalid_measured_start'
    executor.configure(state(),value)  # previous soft permission cannot persist
    assert not executor.summary()['soft_uncertainty_enabled']
    assert executor.select(state(),Q)['status']=='invalid_measured_start'


def test_archive_starvation_cap_stops_without_replaying_or_claiming_task_outcome():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v9_zero_shot_001/trials/bowl_original_zero_hybrid_v9/trace.jsonl'
    if not path.exists():pytest.skip('Optional immutable V9 trace unavailable')
    chosen=None
    for line in path.open():
        row=json.loads(line)
        if row.get('event')=='action' and row.get('step')==620:
            chosen=row;break
    assert chosen is not None
    cap=chosen['diagnostics']['feedback']['visual']['attachment']['transport_command_cap_m']
    assert .0002<cap<.00021
    executor=MetricFieldExecutorV10();executor.configure(state(),axis_intent(),scene_assessment=assessment())
    assert not executor.select(state(),Q,command_cap_m=cap)['accepted']
    assert not executor.summary()['task_success_certified']


def test_validated_visual_correction_persists_across_resume_without_changing_locked_corridor():
    ledger=SubgoalLedger(); initial=state()
    ledger.apply(initial,prediction(value=axis_intent('xy',goal_offset_m=[.05,0.,0.])))
    resumed_pose=state();resumed_pose[0]+=.01;resumed_pose[2]+=.002
    ledger.observe(resumed_pose)
    # Actual local goal uses the measured inactive coordinate; the ledger's
    # original plane remains z=1, with its unchanged3mm corridor.
    assert ledger.record_visual_correction([.05,.002,1.002])
    assert ledger.active['world_goal_xyz']==[.05,.002,1.]
    assert ledger.active['model_world_goal_xyz']==[.05,0.,1.]
    assert ledger.active['visual_correction_count']==1
    result=ledger.apply(resumed_pose,prediction('retain',value=axis_intent('xy',goal_offset_m=[.05,0.,0.])))
    assert result['accepted']
    assert result['field_intent']['goal_offset_m']==pytest.approx([.04,.002,0.])
    assert result['executor_context']['subgoal_origin_xyz']==[0.,0.,1.]
    assert not ledger.record_visual_correction([.05,.002,1.0025])
    assert not ledger.record_visual_correction([.05,.002,1.004])
    assert not ledger.record_visual_correction([.05,.006,1.002])
    assert not ledger.active['task_success_certified']


def test_visual_correction_total_budget_cannot_ratchet_across_repairs():
    ledger=SubgoalLedger();ledger.apply(state(),prediction(value=axis_intent()))
    for y in [.003,.006,.009]:
        assert ledger.record_visual_correction([.07,y,1.])
        assert ledger.apply(state(),prediction('retain',controller='direct'))['accepted']
    assert ledger.active['visual_correction_travel_m']==pytest.approx(.009)
    before=copy.deepcopy(ledger.active)
    assert not ledger.record_visual_correction([.07,.011,1.])
    assert ledger.active==before
    reached=state();reached[:3]=ledger.active['world_goal_xyz'];ledger.observe(reached)
    assert not ledger.record_visual_correction([.07,.0095,1.])


def test_exact_grid_edge_roundoff_is_normalized_without_clipping_invalid_goals():
    ledger=SubgoalLedger();result=ledger.apply(state(),prediction(value=axis_intent('z',goal_offset_m=[0.,0.,.08])))
    assert result['field_intent']['goal_offset_m']==[0.,0.,.08]
    executor=MetricFieldExecutorV10();executor.configure(state(),result['field_intent'])
    moved=state();moved[2]-=.000001
    assert ledger.apply(moved,prediction('retain'))['status']=='subgoal_outside_local_grid_requires_replan'


def test_zero_or_tiny_cap_never_reopens_rejection_or_skips_pending_state_contract():
    executor=MetricFieldExecutorV10();executor.configure(state(),axis_intent())
    drift=state();drift[0]+=.001
    assert executor.select(drift,Q,command_cap_m=0.)['status']=='unacknowledged_state_change'
    assert executor.select(state(),Q,command_cap_m=0.)['status']=='unacknowledged_state_change'
    executor.configure(state(),axis_intent())
    assert executor.select(state(),Q,command_cap_m=1e-8)['status']=='field_command_below_useful_motion_floor'
