"""Target insertion must preserve continuous axis connectors and real masks.

The first fixture records allowed robot feedback from V13 trial 001. These tests
reconstruct local mathematics only; no scene truth or simulator is queried.
"""
import numpy as np
import pytest

from gpt_field.contact_geometry_v13 import TargetContactGeometry
from gpt_field.visual_field_v6 import _AxisConstrainedField
from gpt_field.visual_field_v12 import MetricFieldExecutorV12
from gpt_field.visual_field_v13 import MetricFieldExecutorV13
from test_visual_field_v4 import Q, box, state
from test_visual_field_v6 import axis_intent
from test_visual_field_v9 import assessment


RECORDED_BEFORE=np.array([-0.21124162683394807,-0.011090601889321502,1.1742495176982402,
    -3.139837268246076,0.0028468604393764676,0.08770975201502634,
    0.03872336505859045,-0.0387222020954563])
RECORDED_AFTER=np.array([-0.2112311375788675,-0.01110483663063852,1.1740375257573,
    -3.1398780297366433,0.002385294166962738,0.08807978112182126,
    0.03884534700550173,-0.03884478178732438])


def obstacle_field(field):
    result=field.field
    while isinstance(result,_AxisConstrainedField):result=result._obstacle_field
    return result


def geometry(center=(.25,.25,-.2),half=(.02,.02,.02),sigma=.01,permission='none'):
    return {'id':'target','center_offset_m':list(center),'half_extents_m':list(half),
        'uncertainty_m':sigma,'allowed_contact':permission}


def configured(current,intent,*,target=None,context=None,origin=None):
    contact=TargetContactGeometry()
    contact.configure(current,target or geometry(),target_id='target',tool_radius_m=intent['tool_radius_m'])
    result=MetricFieldExecutorV13();result.contact_geometry=contact
    result.configure(current,intent,scene_assessment=context or assessment(),subgoal_origin_xyz=origin)
    return result


def test_recorded_18_micrometer_lateral_drift_does_not_fail_target_axis_reconstruction():
    target=geometry(center=(-.06,-.18,-.32),half=(.045,.045,.035),sigma=.05)
    intent=axis_intent('z',goal_offset_m=[0.,0.,-.08],goal_tolerance_m=.008,
        max_actions=100,max_travel_m=.09,command_limit_m=.012,tool_radius_m=.055,
        uncertainty_m=.035,orientation_target_xyzw=[-0.9996096147738875,
            0.0009063364830718683,0.027923584546988175,0.0002646352713947504])
    field=configured(RECORDED_BEFORE,intent,target=target)
    selected=field.select(RECORDED_BEFORE,Q,command_cap_m=.002)
    assert selected['accepted']
    assert selected['action']==pytest.approx([0.,0.,-.04,2e-9,0.,0.,0.],abs=1e-12)
    lateral=np.linalg.norm((RECORDED_AFTER-RECORDED_BEFORE)[:2])
    assert lateral==pytest.approx(.00001768197761868552)
    assert lateral<.003
    assert field.checked.segment_valid(RECORDED_BEFORE[:3],RECORDED_AFTER[:3])
    assert obstacle_field(field).query(RECORDED_AFTER[:3]) is not None
    field.observe(RECORDED_BEFORE,selected['action'],RECORDED_AFTER)
    assert field.stop_reason is None and field.actions==1
    following=field.select(RECORDED_AFTER,Q,command_cap_m=.002)
    assert following['accepted'],following['status']
    # The connector may correct the tiny measured drift back toward the
    # original Z line; it cannot expand the authorized XY corridor.
    next_xy=RECORDED_AFTER[:2]+np.asarray(following['action'][:2])*.05
    assert np.linalg.norm(next_xy-RECORDED_BEFORE[:2])<=lateral+1e-12
    assert field.checked.segment_valid(RECORDED_AFTER[:3],
        RECORDED_AFTER[:3]+np.array(following['action'][:3])*.05)


@pytest.mark.parametrize('axes,goal,drift_axis',[('z',[0.,0.,.06],0),('xy',[.06,0.,0.],2)])
@pytest.mark.parametrize('drift',[.000018,.0029])
def test_both_axis_modes_accept_subgrid_feedback_without_erasing_axis_graph(axes,goal,drift_axis,drift):
    before=state();field=configured(before,axis_intent(axes,goal_offset_m=goal))
    selected=field.select(before,Q,command_cap_m=.002)
    assert selected['accepted']
    after=before.copy();after[:3]+=np.array(selected['action'][:3])*.05*.25
    after[drift_axis]+=drift
    field.observe(before,selected['action'],after)
    assert field.stop_reason is None
    assert field.select(after,Q,command_cap_m=.002)['accepted']
    outside=after[:3].copy();outside[drift_axis]=before[drift_axis]+.0031
    assert obstacle_field(field).query(outside) is not None
    assert not field.checked.segment_valid(outside,outside)
    # Graph nodes remain constrained even though continuous connectors admit
    # small between-node measured drift.
    points=field.grid.coordinates(field.origin)
    valid=points[list(field.field.valid_indices)]
    assert np.max(abs(valid[:,drift_axis]-before[drift_axis]))<1e-12


@pytest.mark.parametrize('axes,goal,drift_axis',[('z',[0.,0.,.06],1),('xy',[.06,0.,0.],2)])
def test_larger_than_3mm_measured_axis_drift_still_stops(axes,goal,drift_axis):
    before=state();field=configured(before,axis_intent(axes,goal_offset_m=goal))
    selected=field.select(before,Q,command_cap_m=.002)
    assert selected['accepted']
    after=before.copy();after[drift_axis]+=.0031
    field.observe(before,selected['action'],after)
    assert field.stop_reason=='locked_axis_pose_drift'
    assert not field.select(after,Q)['accepted']
    assert field.subgoal_origin==pytest.approx(before[:3])


def test_original_axis_corridor_survives_refreshed_target_reconstruction():
    current=state();original=current[:3].copy();current[0]+=.002
    field=configured(current,axis_intent('z',goal_offset_m=[0.,0.,.06]),origin=original)
    inside=current[:3].copy();inside[0]=original[0]+.0029
    assert field.checked.segment_valid(inside,inside)
    outside=current[:3].copy();outside[0]=original[0]+.0031
    # Inside the refreshed corridor, but outside the persistent one.
    assert abs(outside[0]-current[0])<.003
    assert not field.checked.segment_valid(outside,outside)
    assert field.subgoal_origin==pytest.approx(original)


@pytest.mark.parametrize('axes,goal',[('z',[0.,0.,.06]),('xy',[.06,0.,0.])])
@pytest.mark.parametrize('strict',[False,True])
def test_target_neighbor_and_unknown_masks_all_survive_unwrapping(axes,goal,strict):
    current=state()
    neighbor=box('neighbor',low=(-.002,-.002,-.067),high=(.002,.002,-.063))
    unknown=box('unseen',low=(.060,-.003,.018),high=(.065,.003,.023))
    intent=axis_intent(axes,goal_offset_m=goal,obstacles=[neighbor],unknown_regions=[unknown],
        uncertainty_m=.012)
    context=assessment('neighbor','unseen')
    if strict:context['obstacle_assessment']='unobservable'
    baseline=MetricFieldExecutorV12();baseline.configure(current,intent,scene_assessment=context)
    initial=obstacle_field(baseline)
    field=configured(current,intent,target=geometry(center=(0.,0.,.04),half=(.003,.003,.003),sigma=.015),context=context)
    merged=obstacle_field(field)
    old_blocked=set(initial.prediction.blocked_indices)
    old_unknown=set(initial.prediction.unknown_indices)
    new_blocked=set(merged.prediction.blocked_indices)
    new_unknown=set(merged.prediction.unknown_indices)
    assert old_blocked and old_unknown
    assert old_blocked<=new_blocked
    assert old_unknown<=(new_unknown|new_blocked)
    assert new_unknown==old_unknown-new_blocked
    target=field.target_configuration
    hard=target['uncertainty_halo_parts'] if strict else target['core_parts']
    points=field.grid.coordinates(field.origin);target_mask=np.zeros(field.grid.count,bool)
    for part in hard:
        target_mask|=np.all((points>=part['lower_xyz'])&(points<=part['upper_xyz']),axis=1)
    target_indices=set(np.flatnonzero(target_mask).tolist())
    assert target_indices and target_indices<=new_blocked
    assert new_blocked==old_blocked|target_indices
    assert field.intent==intent  # No geometry/uncertainty reduction.
    for region in field.region_geometry:
        parts=(region['full_halo_parts'] if strict or region['kind']=='unknown_regions'
               else region['tool_core_parts'])
        for part in parts:
            center=(np.array(part['lower_xyz'])+part['upper_xyz'])/2
            if np.all(center>=merged.lower) and np.all(center<=merged.upper):
                assert not field.checked.segment_valid(center,center)
