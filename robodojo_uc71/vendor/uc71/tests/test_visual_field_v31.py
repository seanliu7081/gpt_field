"""Exact subgrid connectors never delete masks or unrepresented constraints."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpt_field.distance_planner import DistanceFieldPlanner
from gpt_field.field import GridField
from gpt_field.schema import GridSpec
from gpt_field.visual_field_v4 import _RegionPlanner
from gpt_field.visual_field_v6 import _AxisConstrainedField
from gpt_field.visual_field_v31 import (ExactBoxConnectorFieldV31, MetricFieldExecutorV31,
    exact_box_connector_field)
from test_field import prediction


def scene(*, unknown=False, center=(0.,0.,1.), size=5, boxes=None):
    grid = GridSpec(size=size, spacing_m=.005)
    points = grid.coordinates(center)
    boxes = [(np.asarray(center)+[.008,-.02,-.02], np.asarray(center)+[.02,.02,.02])]
    mask = np.zeros(grid.count,bool)
    for low, high in boxes: mask |= np.all((points>=low)&(points<=high),axis=1)
    masks = {'unknown_indices' if unknown else 'blocked_indices': np.flatnonzero(mask).tolist()}
    values = points@np.array([2.,-3.,.5])+7.
    base = GridField(grid,center,prediction(grid,values,**masks))
    return base, boxes


def compact():
    return {'stage':'approach','orientation_target_xyzw':[0,0,0,1],
        'gripper_command':'hold','motion_mode':'move','goal_offset_m':[.006,0,0],
        'attraction_weights':[1.,1.,1.],'repulsors':[], 'blocked_boxes':[], 'unknown_boxes':[]}


@pytest.mark.parametrize('unknown',[False,True])
def test_exact_clear_subgrid_point_uses_same_scalar_values_and_same_mask(unknown):
    base,boxes=scene(unknown=unknown);field,report=exact_box_connector_field(base,boxes)
    point=np.array([.006,.001,1.001])
    assert base.query(point) is None and field.query(point) is not None
    assert field.query(point)==pytest.approx(float(point@np.array([2.,-3.,.5])+7.))
    assert field.prediction==base.prediction
    assert np.array_equal(field._valid,base._valid) and np.array_equal(field._values,base._values)
    assert field.valid_indices==base.valid_indices and report['enabled']
    assert not report['clearance_certified']
    assert field.query([.008,0,1.]) is None and field.query([.009,0,1.]) is None
    assert field.segment_valid([0,0,1.],[.007,0,1.])
    assert not field.segment_valid([0,0,1.],[.009,0,1.])


@pytest.mark.parametrize('unknown',[False,True])
def test_unexplained_invalid_vertex_disables_entire_exception(unknown):
    base,boxes=scene();center=base.grid.count//2
    pred=replace(base.prediction,**({'unknown_indices':(center,)} if unknown else
        {'blocked_indices':tuple(sorted((*base.prediction.blocked_indices,center)))}))
    original=GridField(base.grid,base.center,pred)
    field,report=exact_box_connector_field(original,boxes)
    assert field is original and not report['enabled']
    assert field.query([.006,0,1.]) is None
    with pytest.raises(ValueError,match='not represented'):ExactBoxConnectorFieldV31(original,boxes)


def test_thin_box_with_no_masked_vertex_still_blocks_complete_crossing_and_tangent():
    grid=GridSpec(size=5,spacing_m=.005);base=GridField(grid,(0,0,1),prediction(grid))
    boxes=[(np.array([.001,-.002,.998]),np.array([.0011,.002,1.002]))]
    field,report=exact_box_connector_field(base,boxes)
    assert report['enabled'] and field.valid_indices==base.valid_indices
    assert field.query([0,0,1])==0 and field.query([.003,0,1])==0
    assert not field.segment_valid([0,0,1],[.003,0,1])
    assert not field.segment_valid([0,.002,1],[.003,.002,1])
    assert field.segment_valid([0,.0021,1],[.003,.0021,1])


@pytest.mark.parametrize('point',[[.010001,0,1],[-.010001,0,1],[0,0,1.010001],
    [float('nan'),0,1],[0,0],[0,float('inf'),1]])
def test_invalid_or_outside_grid_points_never_enter_connector(point):
    base,boxes=scene();field,_=exact_box_connector_field(base,boxes)
    assert field.query(point) is None and not field.segment_valid([0,0,1],point)


def test_two_axis_corridors_and_all_original_mask_indices_are_preserved():
    base,boxes=scene();wrapped=_AxisConstrainedField(base,(2,),[0,0,1.])
    wrapped=_AxisConstrainedField(wrapped,(2,),[0,0,1.001])
    result,report=exact_box_connector_field(wrapped,boxes)
    assert report['enabled'] and report['axis_layers_preserved']==2
    assert result.prediction==wrapped.prediction and result.valid_indices==wrapped.valid_indices
    assert result.query([.006,0,1.002]) is not None
    assert result.query([.006,0,.9975]) is None  # Outside second corridor.
    assert result.query([.006,0,1.0031]) is None  # Outside first corridor.
    assert not result.segment_valid([0,0,1],[.006,0,1.0031])
    assert result.segment_valid([0,0,1],[.006,0,1.002])


def test_unknown_field_subclass_is_not_unwrapped_or_assumed_represented():
    class ExtraConstraint(GridField):
        def query(self,point):return None
    base,boxes=scene();other=ExtraConstraint(base.grid,base.center,base.prediction)
    result,report=exact_box_connector_field(other,boxes)
    assert result is other and not report['enabled']


@pytest.mark.parametrize('box',[(np.zeros(3),np.zeros(3)),([0,0],[1,1]),
    ([0,0,0],[1,1,float('nan')])])
def test_malformed_box_cannot_justify_mask_override(box):
    base,_=scene();field,report=exact_box_connector_field(base,[box])
    assert field is base and not report['enabled']


def test_connector_copies_box_bounds_and_does_not_mutate_source():
    base,boxes=scene();original=base.to_dict();field,_=exact_box_connector_field(base,boxes)
    boxes[0][0][0]=-.01
    assert field.query([.006,0,1]) is not None
    assert base.to_dict()==original and base.query([.006,0,1]) is None


def test_dijkstra_preserves_invalid_vertices_and_checks_entire_extracted_route():
    base,boxes=scene();field,_=exact_box_connector_field(base,boxes)
    planner=_RegionPlanner(boxes);goal=np.array([.006,0,1.])
    old=_RegionPlanner(boxes).plan(base,[0,0,1.],goal_xyz=goal,compact_prediction=compact(),goal_tolerance_m=.001)
    assert old.status!='move'
    result=planner.plan(field,[0,0,1.],goal_xyz=goal,compact_prediction=compact(),goal_tolerance_m=.001)
    assert result.status=='move' and result.selected_index is not None
    route=np.asarray(result.candidates[result.selected_index].positions)
    assert np.linalg.norm(route[-1]-goal)<=.001+1e-10
    assert all(field.segment_valid(a,b) for a,b in zip(route,route[1:]))
    assert all(DistanceFieldPlanner._avoids_boxes(a,b,boxes)[0] for a,b in zip(route,route[1:]))
    assert np.all(np.isinf(planner.distance_grid_m.ravel()[list(base.prediction.blocked_indices)]))
    assert result.search_metadata['valid_vertex_count']==len(base.valid_indices)


def test_recorded_v30_cabinet_clear_6mm_connector_does_not_inherit_extra_grid_padding():
    path=Path(__file__).parents[1]/'runs/visual_field_v31_design_001/proposal6mm_grid_admission_audit.json'
    if not path.exists():pytest.skip('optional saved geometry-only audit unavailable')
    audit=json.loads(path.read_text());start=np.asarray(audit['start']);goal=np.asarray(audit['goal'])
    boxes=[(np.asarray(part['lower_xyz']),np.asarray(part['upper_xyz'])) for part in audit['cabinet']['full_halo_parts']]
    grid=GridSpec(size=33,spacing_m=.005);points=grid.coordinates(start);blocked=np.zeros(grid.count,bool)
    for low,high in boxes:blocked |=np.all((points>=low)&(points<=high),axis=1)
    base=GridField(grid,tuple(start),prediction(grid,np.linalg.norm(points-goal,axis=1),blocked_indices=np.flatnonzero(blocked).tolist()))
    axis=_AxisConstrainedField(base,(2,),start);field,report=exact_box_connector_field(axis,boxes)
    assert axis.query(goal) is None and report['enabled']
    assert field.query(goal) is not None and field.segment_valid(start,goal)
    assert field.prediction==axis.prediction
    planner=_RegionPlanner(boxes);result=planner.plan(field,start,goal_xyz=goal,compact_prediction=compact(),goal_tolerance_m=.0015)
    assert result.status=='move'
    route=np.asarray(result.candidates[result.selected_index].positions)
    assert all(field.segment_valid(a,b) for a,b in zip(route,route[1:]))
    assert np.linalg.norm(route[-1]-goal)<=.0015+1e-10


def test_executor_installs_only_opt_in_wrapper_and_keeps_guard_selection_authority():
    from test_visual_field_v29 import setup
    legacy,guard,monitor,state=setup(drift=0.)
    field=MetricFieldExecutorV31();field.__dict__.update(legacy.__dict__)
    # Configure a fresh identical known scene via the actual full inherited
    # geometry builder. No accepted guard state or pending action is injected.
    field.configure(state,copy.deepcopy(legacy.intent),scene_assessment=guard.context,
        inventory_materialization=copy.deepcopy(legacy._transit_materialization))
    assert field.connector_validity['enabled']
    assert field.translation_guard is guard and field.visual_monitor is monitor
    assert field.pending is None
    before=copy.deepcopy(guard.summary())
    # A current guard-domain veto must still block an exact-box-clear route.
    guard.domain_invalid='translation_tool_correspondence_lost'
    selected=field.select(state,[-.01,-.17,-.02,-2.45,0.,2.23,.77])
    assert not selected['accepted'] and field.pending is None
    assert guard.observation_travel_m==before['observation_travel_m']
