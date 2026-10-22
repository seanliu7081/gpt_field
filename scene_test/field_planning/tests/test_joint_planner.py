import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_planner import JointPlanner, PlannerSettings, interpolate_edge


def test_edges_include_endpoints_and_respect_all_joint_steps():
    a=np.array([0.,.1]);b=np.array([.13,-.2])
    q=interpolate_edge(a,b,.025)
    assert np.array_equal(q[0],a)
    assert np.allclose(q[-1],b)
    assert np.max(np.abs(np.diff(q,axis=0))) <= .025+1e-14


def test_straight_path_checks_interior_not_only_endpoints():
    seen=[]
    def check(q):
        seen.append(q.copy())
        return not (.45 < q[0] < .55)
    p=JointPlanner([[0.,1.]],check,PlannerSettings(max_iterations=40))
    r=p.plan([.1],[.9],1)
    assert r['status']=='no_path_within_budget'
    assert any(.45 < q[0] < .55 for q in seen)
    assert not r['path_q']


def test_bidirectional_search_detours_and_retains_start_goal_order():
    def valid(q):
        return not (.42 <= q[0] <= .58 and q[1] < .75)
    settings=PlannerSettings(max_iterations=1000,extension_rad=.13,edge_max_joint_step_rad=.01)
    p=JointPlanner([[0.,1.],[0.,1.]],valid,settings)
    r=p.plan([.1,.2],[.9,.2],17)
    assert r['status']=='planned' and r['reason']=='rrt_connect'
    assert np.array_equal(r['path_q'][0],[.1,.2])
    assert np.array_equal(r['path_q'][-1],[.9,.2])
    assert len(r['path_q'])>2
    for a,b in zip(r['path_q'],r['path_q'][1:]):
        assert all(valid(q) for q in interpolate_edge(a,b,.001))
    r2=p.plan([.1,.2],[.9,.2],17)
    assert r['path_q']==r2['path_q'] and r['counts']==r2['counts']


@pytest.mark.parametrize('where',['start','goal'])
def test_invalid_endpoints_never_create_fake_path(where):
    p=JointPlanner([[0.,1.]],lambda q:{'valid':q[0]>.2,'reason':'unknown_space'})
    r=p.plan([.1 if where=='start' else .5],[.1 if where=='goal' else .8],1)
    assert r['status']=='invalid_'+where
    assert r['reason']=='unknown_space' and r['path_q']==[]


def test_joint_limit_refusal_does_not_invoke_geometry():
    p=JointPlanner([[0.,1.]],lambda q:pytest.fail('must not query invalid joint'))
    r=p.plan([-1.],[.5],1)
    assert r['status']=='invalid_start' and r['reason']=='joint_limit'


def test_unknown_cannot_be_truthy_valid_by_accident():
    p=JointPlanner([[0.,1.]],lambda q:{'valid':'unknown'})
    with pytest.raises(ValueError):p.plan([.1],[.5],1)


def test_empty_motion_is_a_valid_zero_length_path():
    r=JointPlanner([[0.,1.]],lambda q:True).plan([.4],[.4],9)
    assert r['status']=='planned' and r['joint_path_length_rad']==0.
