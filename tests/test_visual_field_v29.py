"""Constraint-aware tracking: real saved RGB replay and synthetic geometry cases."""
import copy
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest

from gpt_field.tracking import PathTracker
from gpt_field.visual_field_v29 import MetricFieldExecutorV29, _AxisCandidateTracker
from gpt_field.visual_field_v27 import MetricFieldExecutorV27
from gpt_field.visual_field_v16 import InventoryTargetGeometryV16
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17
from test_visual_translation_guard_v21 import plant
from test_visual_field_v6 import axis_intent
from test_visual_field_v4 import Q


def setup(drift=.0003, axes='xy'):
    guard, monitor, _, state = plant()
    field = MetricFieldExecutorV29(); field.bind_translation_guard(guard, monitor)
    geometry = {'id':'neighbor', 'center_offset_m':[0,0,-.25],
        'half_extents_m':[.02]*3, 'uncertainty_m':.015, 'allowed_contact':'none'}
    contact = InventoryTargetGeometryV16()
    contact.configure(state, geometry, target_id='neighbor', tool_radius_m=.045)
    field.contact_geometry = contact
    material = {'target_id':'neighbor', 'target_geometry':geometry, 'objects':[{
        'id':'neighbor', 'world_geometry':{'center_xyz_m':(state[:3]+[0,0,-.25]).tolist(),
            'half_extents_m':[.02]*3, 'uncertainty_m':.015},
        'views':{'external':{'current_roi_uv':[98/223,53/223,122/223,77/223]}}}],
        'obstacles':[], 'applied_metric_ids':['neighbor'], 'applied_image_ids':['neighbor']}
    intent = axis_intent(axes, goal_offset_m=[.05,0,0], tool_radius_m=.045,
        uncertainty_m=.015, max_actions=120)
    field.configure(state, intent, scene_assessment=guard.context, inventory_materialization=material)
    assert field.bind_recovery_route('normal',state)['accepted']
    scene = SceneCoverageGuardV17()
    prediction = {'controller':'visual_field','phase':'approach','target_id':'neighbor','field_intent':intent}
    field.scene_assessor = lambda cur,delta:scene.assess(cur,delta,materialization=material,
        field=field,monitor=monitor,prediction=prediction)
    field.tracker.reset([state[:3],field.goal])
    state = state.copy(); state[0]+=.0003; state[2]+=drift
    field.expected_state=state.copy();guard.state=state.copy();monitor.state=state.copy()
    guard.worsening={'neighbor'}  # Isolated geometry selection; no invented RGB evidence.
    return field,guard,monitor,state


def candidate(field,state,cap=.003):
    tracker = _AxisCandidateTracker(lookahead_m=cap,max_command_m=cap,
                                   endpoint_tolerance_m=field.tracker.endpoint_tolerance_m)
    tracker.__dict__.update(field.tracker.__dict__)
    tracker.lookahead_m=tracker.max_command_m=cap
    tracker._candidate_selector=field._axis_candidate
    return tracker.command(state[:3],field.checked,field.planner.config.workspace_min,field.planner.config.workspace_max)


@pytest.mark.parametrize('drift',[.0002966939613922026,.000019969749155634986,
                                  .0000064693620702716,.000009044603126340256])
def test_observed_scale_locked_drift_uses_same_norm_active_direction_with_exact_geometry(drift):
    field,guard,_,state=setup(drift)
    goal=field.goal.copy(); path=field.tracker.positions.copy()
    debt=pickle.dumps((guard.observation_travel_m,guard.experiments,field.pending))
    delta,details=candidate(field,state)
    metadata=details['axis_tracking_constraint']
    assert metadata['reason']=='locked_axis_return_suppressed_before_admission'
    assert delta==pytest.approx([.003,0,0],abs=1e-12)
    proof=field._transit_reconciliation_contract(state,state[:3],state[:3]+delta)
    assert proof['eligible'] and proof['object_proofs']['neighbor']['gap_nondecreasing']
    assert proof['object_proofs']['neighbor']['full_uncertainty_sweep_disjoint']
    np.testing.assert_array_equal(field.goal,goal);np.testing.assert_array_equal(field.tracker.positions,path)
    assert pickle.dumps((guard.observation_travel_m,guard.experiments,field.pending))==debt


def test_one_mm_cold_cap_is_not_shrunk_below_floor_or_amplified():
    field,_,_,state=setup()
    delta,details=candidate(field,state,.001)
    assert np.linalg.norm(delta)==pytest.approx(.001,abs=1e-15)
    assert delta[2]==0.
    assert np.linalg.norm(delta)<=np.linalg.norm(details['axis_tracking_constraint']['nominal_delta_m'])+1e-15
    assert details['axis_tracking_constraint']['inherited_admission_still_required']


@pytest.mark.parametrize('fault',['outside_corridor','endpoint_offset','path_not_planar','scene',
                                  'contact','payload','unknown','stale_state','reference_corner'])
def test_projection_never_overrides_another_constraint(fault):
    field,guard,_,state=setup()
    if fault=='outside_corridor':state[2]=field.origin[2]+.0031
    elif fault=='endpoint_offset':state[2]=field.origin[2]+field.tracker.endpoint_tolerance_m
    elif fault=='path_not_planar':field.tracker.positions[-1,2]+=.001
    elif fault=='scene':field.scene_assessor=lambda *args:{'accepted':False,'diagnostics':{}}
    elif fault=='contact':field.target_configuration['effective_allowed_contact']='fingers'
    elif fault=='payload':field.contact_geometry.carried_summary=lambda s:{'active':True,'candidate_ids':[]}
    elif fault=='unknown':
        field.boxes.append((state[:3]+[.001,-.001,-.001],state[:3]+[.002,.001,.001]))
    elif fault=='stale_state':guard.state[0]+=.001
    elif fault=='reference_corner':
        # A selected earlier corner provides <1mm forward distance. Never
        # extend it to meet the command floor; the original command is retained.
        nominal=np.array([.0008,0.,-.0003])
        selected,metadata=field._axis_candidate(state[:3],nominal,field.tracker,
            field.checked,field.planner.config.workspace_min,field.planner.config.workspace_max,state[:3]+nominal)
        assert selected is None and metadata['reason']=='projected_candidate_below_useful_motion_floor'
        return
    # Evaluate the actual pre-admission selector, preserving legacy refusal.
    nominal=np.array([.003,0.,-.0003])
    selected,metadata=field._axis_candidate(state[:3],nominal,field.tracker,
        field.checked,field.planner.config.workspace_min,field.planner.config.workspace_max,state[:3]+nominal)
    assert selected is None,(fault,metadata)


def test_endpoint_arrival_remains_measured_euclidean_and_no_warning_has_exact_parity():
    field,guard,_,state=setup()
    field.tracker.endpoint_tolerance_m=.0002
    field.tracker.progress_m=field.tracker.arcs[-1]
    point=field.goal+[0,0,.0003]
    assert not field.tracker.endpoint_reached(point)
    assert field.tracker.endpoint_distance(point)>.0002
    # Neither no warning nor other route families install the new selector.
    for route in ('normal','reverse_observed','initial_uncertainty_egress'):
        field.recovery_route_binding['recovery_route']=route
        guard.worsening=set() if route=='normal' else {'neighbor'}
        assert not field._axis_tracking_scoped()
    from test_visual_field_v27 import upgraded
    new,gn,_,sn=upgraded();old,go,_,so=upgraded()
    new.__class__=MetricFieldExecutorV29;new.axis_tracking_constraint=None
    a=new.select(sn,Q);b=old.select(so,Q)
    assert a['accepted']==b['accepted'] and a['action']==b['action']
    assert new.axis_tracking_constraint is None


def test_actual_executor_reselects_exact_cap_and_acknowledges_only_final_candidate(monkeypatch):
    field,guard,_,state=setup();tracker=field.tracker
    field.last_plan=field.planner.plan(field.field,field.origin,goal_xyz=field.goal,
        compact_prediction=field.compact,goal_tolerance_m=field.tracker.endpoint_tolerance_m)
    checked=[]
    def admission(delta,**kwargs):
        delta=np.asarray(delta)
        proof=kwargs['nominal_regions'][0]['transit_reconciliation_contract']
        assert proof['eligible'] and proof['object_proofs']['neighbor']['gap_nondecreasing']
        np.testing.assert_allclose(proof['end_xyz_m'],state[:3]+delta,rtol=0,atol=1e-15)
        assert delta[2]==0.
        checked.append(delta.copy())
        return {'accepted':True,'cap_m':.001,'reason':'test_admission_cap','diagnostics':{}}
    # Admission spy tests exact re-selection plumbing; geometry is real. It
    # does not represent newly obtained visual evidence or authorize a robot.
    monkeypatch.setattr(guard,'assess',admission)
    debt=guard.observation_travel_m;goal=field.goal.copy();path=tracker.positions.copy()
    selected=field.select(state,Q,command_cap_m=.003)
    assert selected['accepted'],selected['status']
    assert len(checked)==2 and np.linalg.norm(checked[0])==pytest.approx(.003)
    assert np.linalg.norm(checked[1])==pytest.approx(.001)
    np.testing.assert_allclose(np.asarray(selected['action'][:3])*.05,checked[-1],rtol=0,atol=1e-15)
    assert field.tracker is tracker and guard.observation_travel_m==debt
    np.testing.assert_array_equal(field.goal,goal);np.testing.assert_array_equal(tracker.positions,path)
    wrong=np.asarray(selected['action']).copy();wrong[2]+=.01
    with pytest.raises(ValueError):field.observe(state,wrong,state)
    after=state.copy();after[:3]+=checked[-1]*.2
    field.observe(state,selected['action'],after)
    assert field.pending is None and field.actions==1
    assert field.measured_travel_m==pytest.approx(.0002,abs=1e-12)


def test_geometry_candidate_does_not_override_actual_inherited_image_refusal():
    field,guard,_,state=setup();tracker=field.tracker
    field.last_plan=field.planner.plan(field.field,field.origin,goal_xyz=field.goal,
        compact_prediction=field.compact,goal_tolerance_m=field.tracker.endpoint_tolerance_m)
    selected=field.select(state,Q)
    assert not selected['accepted']
    assert field.axis_tracking_constraint['reason']=='locked_axis_return_suppressed_before_admission'
    assert selected['status'].startswith('translation_')
    assert field.pending is None and field.actions==0 and field.tracker is tracker
    assert guard.observation_travel_m==0.


def test_actual_saved_207_transition_prefix_then_constrained_candidate_keeps_full_admission():
    root=Path(__file__).resolve().parents[1]
    helper=root/'runs/visual_field_v29_design_001/tracking_diagnostic.py'
    if not helper.exists():pytest.skip('Optional archived V28 prefix fixture unavailable')
    spec=importlib.util.spec_from_file_location('v29_tracking_test_fixture',helper)
    archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
    _,initial,decisions,actions,executed=archive.replay_data()
    from gpt_field.visual_field_policy_v28 import VisualFieldPolicyV28
    from gpt_field.observation import PolicyObservation
    policy=VisualFieldPolicyV28(archive.HeadingClient(decisions));obs=PolicyObservation.from_dict(initial)
    for step in range(207):
        selected=policy.act(obs)
        np.testing.assert_allclose(selected.action,actions[step],atol=1e-10,rtol=0.)
        after=PolicyObservation.from_dict(executed[step]);policy.observe_transition(obs,selected,after);obs=after
    local,guard=policy.local,policy.translation_guard
    local.__class__=MetricFieldExecutorV29;local.axis_tracking_constraint=None
    path=local.tracker.positions.copy();goal=local.goal.copy();tracker=local.tracker
    debt=guard.observation_travel_m; calls=policy.client.calls
    selected=policy.act(obs)
    result=selected.diagnostics['selection']
    assert result['accepted'],result['status']
    delta=np.asarray(selected.action[:3])*.05
    assert delta[0]<0 and delta[2]==0 and np.linalg.norm(delta)==pytest.approx(.001,abs=1e-12)
    assert result['diagnostics']['axis_tracking_constraint']['reason']=='locked_axis_return_suppressed_before_admission'
    proof=result['diagnostics']['transit_reconciliation_geometry']
    assert proof['eligible'] and set(proof['object_proofs'])==set(proof['scene_ids'])
    assert len(proof['scene_ids'])==8
    assert all(proof['object_proofs'][i]['gap_nondecreasing'] for i in guard.worsening)
    assert all(p['full_uncertainty_sweep_disjoint'] for p in proof['object_proofs'].values())
    np.testing.assert_array_equal(local.goal,goal);np.testing.assert_array_equal(local.tracker.positions,path)
    assert local.tracker is tracker and local.pending is not None
    assert guard.observation_travel_m==debt and policy.client.calls==calls
    wrong=np.asarray(selected.action).copy();wrong[0]+=.001
    with pytest.raises(ValueError):local.observe(obs.state,wrong,obs.state)
    # No counterfactual transition is fed back: this is an offline selection,
    # not robot continuation or evidence of task success.
    policy.finish()
