"""Candidate admission regressions; no simulator, inference, or truth inputs.

These executor-isolation tests reconstruct an exhausted reference ledger.
Actual causal RGB accumulation/one-request reuse is covered in policy tests.
"""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_field_policy_v20 import VisualFieldPolicyV20
from gpt_field.visual_field_v20 import MetricFieldExecutorV20
from gpt_field.observation import PolicyObservation
from gpt_field.robot_tool_geometry import world_aabbs
from test_visual_field_v6_policy import Client
from test_visual_field_v17_policy import prediction
from test_visual_field_v13_execution import observation
from test_visual_field_v4 import Q, box


def free_prediction():
    value = prediction()
    value.update(phase="approach", target_contact="none", contact_alignment=None)
    value["target_geometry"]["allowed_contact"] = "none"
    value["scene_inventory"][0]["geometry"].update(center_offset_m=[.3,.3,.3], uncertainty_m=.001)
    value["field_intent"].update(goal_offset_m=[.06,0.,0.], command_limit_m=.01,
        max_actions=120, tool_radius_m=.005, uncertainty_m=.001)
    value["motion_context"].update(translation_mode="transit", intended_contact_id=None)
    return value


def bound_request(executor, state):
    return {"version": executor.TRANSIT_REQUEST_VERSION, "reason": executor.WEAK_REFERENCE_REASON,
        "state8": list(state), "observation_epoch": executor.translation_guard.sweep_observation_epoch,
        "goal_xyz": executor.goal.tolist(), "reference_id": executor.visual_monitor._active_reference["id"],
        "target_id": executor.target_configuration["target_id"], "anchor_shifted":False,
        "raw_external_role":"tool", "target_contact":"none"}


def configured(value=None):
    policy = VisualFieldPolicyV20(Client(value or free_prediction()))
    obs = observation()
    policy._refresh(obs)
    monitor = policy.monitor
    monitor._active_reference["unvalidated_travel_m"] = .0301
    monitor._active_reference["blocked_reason"] = monitor.uncertainty_budget_reason
    monitor.status = monitor.uncertainty_budget_reason
    policy.local.proprioceptive_transit_request = bound_request(policy.local, obs.state)
    return policy, obs


def test_exhausted_reference_can_use_fresh_metric_candidate_without_resetting_ledger():
    policy, obs = configured(); executor=policy.local
    record=pickle.dumps(policy.monitor._active_reference)
    debt=policy.translation_guard.observation_travel_m
    goal=executor.goal.copy()
    selected=executor.select(obs.state,Q)
    assert selected["accepted"], selected
    report=selected["diagnostics"]["proprioceptive_transit_admission"]
    assert report["accepted"] and report["diagnostics"]["complete_scene_ids"]==["part"]
    assert 0 < np.linalg.norm(np.asarray(selected["action"][:3])*.05) <= .005+1e-12
    assert pickle.dumps(policy.monitor._active_reference)==record
    assert policy.translation_guard.observation_travel_m==debt
    assert np.array_equal(executor.goal,goal) and executor.goal_revisions==0
    assert not report["diagnostics"]["reference_validation_modified"]
    assert not report["diagnostics"]["clearance_certified"]
    after=np.asarray(obs.state).copy();after[:3]+=np.asarray(selected["action"][:3])*.05*.4
    executor.observe(obs.state,selected["action"],after)
    assert executor.pending is None and executor.actions==1


@pytest.mark.parametrize("changed", ["state", "orientation", "epoch", "goal", "reference", "anchor", "role"])
def test_provisional_request_is_bound_to_this_observation_and_fixed_goal(changed):
    policy,obs=configured();executor=policy.local;request=executor.proprioceptive_transit_request
    if changed=="state":request["state8"][0]+=.0001
    elif changed=="orientation":request["state8"][3]+=.001
    elif changed=="epoch":request["observation_epoch"]-=1
    elif changed=="goal":request["goal_xyz"][0]+=.001
    elif changed=="reference":request["reference_id"]="unbound-reference"
    elif changed=="anchor":request["anchor_shifted"]=True
    elif changed=="role":request["raw_external_role"]="payload"
    count=executor.orientation_goal_refresh_commands
    result=executor.select(obs.state,Q)
    assert not result["accepted"] and executor.pending is None
    assert executor.orientation_goal_refresh_commands==count and executor.actions==0


@pytest.mark.parametrize("fault", ["crosses_halo", "missing_id", "stale_endpoint", "incomplete_parts", "unlocalized_far"])
def test_post_selection_veto_rolls_back_pending_orientation_but_not_reference_debt(fault):
    policy,obs=configured();executor=policy.local
    original=executor.scene_assessor
    def malformed(state,delta):
        result=copy.deepcopy(original(state,delta));details=result["diagnostics"]
        if fault=="crosses_halo":details["sweep_checks"][0]["full_uncertainty_sweep_disjoint"]=False
        elif fault=="missing_id":details["sweep_checks"]=[]
        elif fault=="stale_endpoint":details["end_xyz_m"][0]+=.001
        elif fault=="incomplete_parts":details["sweep_checks"][0]["configuration_part_count"]=1
        elif fault=="unlocalized_far":details["unresolved_far_image_ids"]=["unlocalized"]
        return result
    executor.scene_assessor=malformed
    record=pickle.dumps(policy.monitor._active_reference)
    count=executor.orientation_goal_refresh_commands;rotation=executor.last_orientation_command_rad.copy()
    result=executor.select(obs.state,Q)
    assert not result["accepted"] and executor.pending is None
    assert executor.orientation_goal_refresh_commands==count
    assert np.array_equal(executor.last_orientation_command_rad,rotation)
    assert pickle.dumps(policy.monitor._active_reference)==record
    assert executor.actions==0


def test_target_contact_band_applies_even_to_a_clear_away_segment():
    value=free_prediction();obs=observation()
    parts=world_aabbs(obs.state,orientation_margin_rad=.04,full_finger_range=True)
    part=max(parts,key=lambda part:part["offset_max_xyz"][0])
    center=(np.asarray(part["offset_min_xyz"])+part["offset_max_xyz"])/2
    center[0]=part["offset_max_xyz"][0]+.004+.025
    value["scene_inventory"][0]["geometry"].update(
        center_offset_m=center.tolist(),half_extents_m=[.004]*3)
    value["field_intent"]["goal_offset_m"]=[-.06,0.,0.]
    policy,obs=configured(value)
    result=policy.local.select(obs.state,Q)
    assert not result["accepted"] and result["status"]=="proprioceptive_transit_near_target_contact_band"
    report=result["diagnostics"]["proprioceptive_transit_admission"]["diagnostics"]
    assert report["scene_coverage_admission"]["accepted"]
    assert report["target_contact_band"]["whole_segment_core_distance_lower_bound_m"]<.03
    assert policy.local.pending is None


def test_real_unknown_geometry_is_not_softened_by_transit_request():
    value=free_prediction()
    value["field_intent"]["unknown_regions"]=[box("unseen",low=(-.001,-.001,-.001), high=(.001,.001,.001))]
    policy,obs=configured(value)
    result=policy.local.select(obs.state,Q)
    assert not result["accepted"] and result["status"]=="invalid_measured_start"
    assert policy.local.intent["unknown_regions"]==value["field_intent"]["unknown_regions"]
    assert policy.local.pending is None


def test_original_action_and_travel_limits_remain_authoritative():
    policy,obs=configured();executor=policy.local
    executor.actions=executor.intent["max_actions"]
    result=executor.select(obs.state,Q)
    assert not result["accepted"] and "budget" in result["status"]
    assert executor.pending is None


def test_no_request_leaves_inherited_selection_unchanged():
    policy,obs=configured();executor=policy.local
    executor.proprioceptive_transit_request=None
    result=executor.select(obs.state,Q)
    assert result["accepted"]
    assert result["diagnostics"]["proprioceptive_transit_admission"] is None
    assert executor.last_proprioceptive_transit_admission is None


def test_recorded_v19_paid_motion_can_continue_same_world_goal_after_weak_budget():
    root=Path(__file__).resolve().parents[1]
    path=root/'runs/visual_field_v19_zero_shot_001/trials/bowl_original_zero_hybrid_v19/trace.jsonl'
    if not path.exists():pytest.skip('Optional recorded V19 prefix unavailable')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    actions={row['step']:row for row in rows if row['event']=='action'}
    observed={row['step']:PolicyObservation.from_dict(row['observation'])
              for row in rows if row['event']=='executed'}
    obs=PolicyObservation.from_dict(rows[0]['observation'])
    client=Client(copy.deepcopy(actions[0]['diagnostics']['prediction']))
    policy=VisualFieldPolicyV20(client)
    for step in range(28):
        decision=policy.act(obs)
        assert decision.diagnostics['selection']['accepted'], (step,decision.diagnostics['selection'])
        assert decision.action==pytest.approx(actions[step]['action'],abs=1e-10)
        after=observed[step]
        policy.observe_transition(obs,decision,after)
        obs=after
    assert client.calls==1 and not policy.refresh_reason
    monitor=policy.monitor
    assert monitor.measured_control_limit()[1]==monitor.uncertainty_budget_reason
    assert actions[28]['diagnostics']['selection']['status']==monitor.uncertainty_budget_reason
    before=pickle.dumps(monitor._active_reference)
    original_goal=policy.local.goal.copy()
    proposed=policy.act(obs)
    assert proposed.diagnostics['selection']['accepted'], proposed.diagnostics['selection']
    admission=proposed.diagnostics['selection']['diagnostics']['proprioceptive_transit_admission']
    assert admission['accepted'] and len(admission['diagnostics']['complete_scene_ids'])==8
    assert 0 < np.linalg.norm(np.array(proposed.action[:3])*.05) <= .005+1e-12
    assert pickle.dumps(monitor._active_reference)==before
    assert np.array_equal(policy.local.goal,original_goal) and policy.local.goal_revisions==0
    assert client.calls==1
    # This newly proposed command is NOT executed or assigned a task outcome.
