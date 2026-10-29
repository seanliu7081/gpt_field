"""Exact directed-box geometry; no simulator or environment-truth inputs."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from gpt_field.uncertainty_egress_v23 import UncertaintyEgressRoute


def archive():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v22_zero_shot_001/trials/bowl_original_zero_hybrid_v22/trace.jsonl'
    if not path.exists():pytest.skip('Optional archived V22 result unavailable')
    action=next(json.loads(l)for l in path.read_text().splitlines()if json.loads(l)['event']=='action')
    sd=action['diagnostics']['selection']['diagnostics']
    state=np.array([*sd['origin_xyz'],0.,0.,0.,.04,-.04])
    return UncertaintyEgressRoute(state,sd['region_feasibility'],target_id='black_bowl'),action


def test_real_v22_initial_halo_has_feasible_generic_full_exit_without_deleting_geometry():
    route,action=archive();original=copy.deepcopy(route.regions)
    assert route.initial_parts=={('drawer_cabinet','hand_collision'),('drawer_cabinet','finger1_collision')}
    geometry=route.check(route.state[:3],route.state[:3]+[0,.003,.030],require_progress=True)
    assert geometry['accepted'] and geometry['exit_goal_full_halo_disjoint']
    assert geometry['nominal_core_clear'] and geometry['no_new_halo_entry']
    candidates=route.candidates()
    assert candidates and candidates[0]['full_exit']
    best=candidates[0]
    assert best['route_length_m']<=.04 and best['goal_full_halo_clearance_m']>.003
    assert np.allclose(best['goal_offset_m'],[0,.0056703820201374227,.03159269376947704])
    assert route.regions==original and len(route.regions)==8


@pytest.mark.parametrize('delta',[[0,-.003,.03],[0,.003,-.03]])
def test_real_v22_deeper_halo_directions_refused(delta):
    route,_=archive();result=route.check(route.state[:3],route.state[:3]+delta,require_progress=True)
    assert not result['accepted'] and not result['monotone_egress']


def test_real_v22_partial_route_not_misreported_as_complete_exit():
    route,_=archive();result=route.check(route.state[:3],route.state[:3]+[0,.004,.008],require_progress=True)
    assert result['accepted'] and not result['exit_goal_full_halo_disjoint']
    assert not result['full_uncertainty_sweep_disjoint']


def test_closed_segment_new_halo_entry_is_refused_even_when_endpoints_are_clear():
    route,_=archive();regions=copy.deepcopy(route.regions)
    start=route.state[:3];middle=start+np.array([0,.0015,.015])
    part={'tool_part':'thin_counterexample','lower_xyz':(middle-.0002).tolist(),'upper_xyz':(middle+.0002).tolist()}
    regions.append({'id':'new_halo','kind':'obstacles','tool_core_parts':[], 'full_halo_parts':[part]})
    changed=UncertaintyEgressRoute(route.state,regions,target_id='black_bowl')
    result=changed.check(start,start+[0,.003,.030])
    assert not result['accepted'] and result['reason']=='egress_new_halo_entry_or_reentry'


def test_unknown_full_region_is_never_an_egress_exception():
    route,_=archive();regions=copy.deepcopy(route.regions)
    cabinet=next(r for r in regions if r['id']=='drawer_cabinet');cabinet['kind']='unknown_regions'
    changed=UncertaintyEgressRoute(route.state,regions,target_id='black_bowl')
    assert changed.invalid_reason=='egress_initial_target_or_unknown_overlap'
    assert not changed.check(route.state[:3],route.state[:3]+[0,.003,.03])['accepted']


def test_nominal_core_collision_cannot_be_exempted_by_halo_overlap():
    route,_=archive();regions=copy.deepcopy(route.regions);point=route.state[:3]
    next(r for r in regions if r['id']=='drawer_cabinet')['tool_core_parts'].append(
        {'tool_part':'counterexample_core','lower_xyz':(point-.001).tolist(),'upper_xyz':(point+.001).tolist()})
    changed=UncertaintyEgressRoute(route.state,regions,target_id='black_bowl')
    assert changed.invalid_reason=='egress_initial_nominal_core_overlap'


def test_once_outside_an_original_halo_cannot_reenter_it():
    route,_=archive();outside=route.state[:3]+[0,.003,.030]
    result=route.check(outside,route.state[:3])
    assert not result['accepted'] and result['reason']=='egress_new_halo_entry_or_reentry'


def test_context_bound_is_not_recentered_by_a_short_step():
    route,_=archive();start=route.state[:3]+[0,.003,.039]
    result=route.check(start,start+[0,0,.003])
    assert not result['accepted'] and result['reason']=='egress_context_extent_exceeded'


def test_zero_pose_has_no_progress_or_clearance_claim():
    route,_=archive();result=route.check(route.state[:3],route.state[:3])
    assert result['accepted'] and not result['exit_goal_full_halo_disjoint']
    assert not result['clearance_certified'] and not result['full_uncertainty_sweep_disjoint']
    assert not route.check(route.state[:3],route.state[:3],require_progress=True)['accepted']
