"""Counterexamples for V9 hard-core geometry and weighted uncertainty costs."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_field_v8 import MetricFieldExecutorV8
from gpt_field.visual_field_v9 import MetricFieldExecutorV9, RiskRegionDistancePlanner
from test_visual_field_v4 import Q, box, state
from test_visual_field_v6 import axis_intent


def assessment(*identities):
    return {"purpose": "task_translation", "obstacle_assessment": "present" if identities else "none_relevant",
            "tool_external_roi_uv": [.4, .3, .6, .5],
            "nearby_obstacles": [{"id": name, "external_roi_uv": [.65, .3, .8, .55]} for name in identities]}


def halo_box():
    # Nominal and tool-radius core do not contain the origin or straight path;
    # uncertainty inflation does. This is the exact distinction under test.
    return box("visible", low=(.015, .013, -.002), high=(.025, .023, .002))


def test_explicit_visible_halo_has_finite_cost_instead_of_invalid_start_without_carving():
    value = axis_intent(obstacles=[halo_box()], uncertainty_m=.025)
    original = copy.deepcopy(value)
    old, new = MetricFieldExecutorV8(), MetricFieldExecutorV9()
    old.configure(state(), value)
    new.configure(state(), value, scene_assessment=assessment("visible"))
    assert old.select(state(), Q)["status"] == "invalid_measured_start"
    result = new.select(state(), Q)
    assert result["accepted"]
    assert result["diagnostics"]["plan"]["search"]["cost_unit"] == "weighted_m"
    assert new.intent == original and value == original
    region = new.summary()["region_feasibility"][0]
    assert not region["nominal"]["contains_current"]
    assert not region["tool_core"]["contains_current"]
    assert region["full_halo"]["contains_current"]
    assert region["current_only_in_uncertainty_halo"]
    assert region["margins_m"]["combined_uncertainty"] == pytest.approx(.026)
    assert region["geometry_mode"] == "hard_tool_core_soft_uncertainty"
    assert not region["certified_occupied"]
    assert not result["diagnostics"]["clearance_certified"]
    # The same core blocks a point independent of whether it is the origin,
    # goal or an interior path point. There is no special origin exemption.
    inside_core = np.array([.02, .018, 1.])
    assert not new.checked.segment_valid(inside_core, inside_core)


@pytest.mark.parametrize("group", ["obstacles", "unknown_regions"])
def test_nominal_and_tool_core_collisions_stay_hard_even_with_supported_rgb(group):
    wall = box("wall", low=(-.002, -.002, -.002), high=(.002, .002, .002))
    executor = MetricFieldExecutorV9()
    executor.configure(state(), axis_intent(**{group: [wall]}), scene_assessment=assessment("wall"))
    chosen = executor.select(state(), Q)
    assert chosen["status"] == "invalid_measured_start" and not chosen["accepted"]
    detail = executor.summary()["region_feasibility"][0]
    assert detail["nominal"]["contains_current"] and detail["tool_core"]["contains_current"]


def test_unknown_halo_is_still_excluded_even_though_visible_object_halo_is_soft():
    executor = MetricFieldExecutorV9()
    value = axis_intent(unknown_regions=[halo_box()], uncertainty_m=.025)
    executor.configure(state(), value, scene_assessment=assessment("visible"))
    assert executor.select(state(), Q)["status"] == "invalid_measured_start"
    assert executor.summary()["risk_density"]["region_count"] == 0
    detail = executor.summary()["region_feasibility"][0]
    assert detail["geometry_mode"] == "hard_full_unknown"
    assert detail["current_only_in_uncertainty_halo"]
    assert executor.regions[0]["inflation_m"] == pytest.approx(.005+.025+.001)


@pytest.mark.parametrize("bad_context", [None, {}, [],
    {"obstacle_assessment": "unobservable", "tool_external_roi_uv": [.4, .3, .6, .5], "nearby_obstacles": []},
    {"obstacle_assessment": "present", "tool_external_roi_uv": None, "nearby_obstacles": []},
    {"obstacle_assessment": "none_relevant", "tool_external_roi_uv": [.4, .3, .6, .5], "nearby_obstacles": []},
    {"obstacle_assessment": "present", "tool_external_roi_uv": [.4, .3, .6, .5], "nearby_obstacles": [{"id": "visible", "external_roi_uv": None}]},
    {"obstacle_assessment": "present", "tool_external_roi_uv": [.4, .3, .6, .5], "nearby_obstacles": [{"id": "wrong", "external_roi_uv": [.7, .3, .8, .5]}]},
])
def test_missing_unobservable_or_inconsistent_assessment_retains_full_old_inflation(bad_context):
    executor = MetricFieldExecutorV9(); value = axis_intent(obstacles=[halo_box()], uncertainty_m=.025)
    executor.configure(state(), value, scene_assessment=bad_context)
    assert not executor.summary()["soft_uncertainty_enabled"]
    assert executor.select(state(), Q)["status"] == "invalid_measured_start"
    assert executor.summary()["risk_density"]["region_count"] == 0


def test_omitted_context_on_reconfigure_never_inherits_previous_soft_permission():
    executor = MetricFieldExecutorV9(); value = axis_intent(obstacles=[halo_box()], uncertainty_m=.025)
    executor.configure(state(), value, scene_assessment=assessment("visible"))
    assert executor.select(state(), Q)["accepted"]
    executor.configure(state(), value)
    assert not executor.summary()["soft_uncertainty_enabled"]
    assert executor.select(state(), Q)["status"] == "invalid_measured_start"


def test_cost_density_is_finite_nonnegative_continuous_and_decreases_away_from_core():
    executor = MetricFieldExecutorV9()
    executor.configure(state(), axis_intent(obstacles=[halo_box()], uncertainty_m=.01), scene_assessment=assessment("visible"))
    core = executor.risk_regions[0];high=np.asarray(core["upper_xyz"]);center=(high+core["lower_xyz"])*.5
    positions=np.array([center, [high[0],center[1],center[2]], [high[0]+1e-8,center[1],center[2]],
                        [high[0]+.011,center[1],center[2]], [high[0]+.1,center[1],center[2]]])
    density=executor.planner._density(positions, executor.field, executor.compact)
    assert np.isfinite(density).all() and np.all(density >= 1.)
    assert density[:2] == pytest.approx([9.,9.])
    assert density[2] == pytest.approx(density[1], abs=1e-10)
    assert density[2] > density[3] > density[4]
    executor.planner.risk_regions[0]["sigma_m"] *= 2
    wider = executor.planner._density(positions, executor.field, executor.compact)
    assert np.all(wider >= density)
    assert executor.summary()["risk_density"]["is_probability"] is False


def test_weighted_dijkstra_selects_longer_lower_risk_route_not_just_geometric_shortest():
    region=box("side_object",low=(.027,.018,-.002),high=(.043,.022,.002))
    value=axis_intent(obstacles=[region],uncertainty_m=.019)
    executor=MetricFieldExecutorV9();executor.configure(state(),value,scene_assessment=assessment("side_object"))
    zero_risk=RiskRegionDistancePlanner(executor.boxes,executor.risk_regions,risk_weight=0.)
    plain=zero_risk.plan(executor.field,state()[:3],goal_xyz=executor.goal,compact_prediction=executor.compact,goal_tolerance_m=.0015)
    weighted=executor.planner.plan(executor.field,state()[:3],goal_xyz=executor.goal,compact_prediction=executor.compact,goal_tolerance_m=.0015)
    assert plain.status==weighted.status=="move"
    straight=np.asarray(plain.candidates[0].positions);detour=np.asarray(weighted.candidates[0].positions)
    assert np.max(abs(straight[:,1])) < 1e-12
    assert np.min(detour[:,1]) < -.015
    assert weighted.search_metadata["shortest_graph_path_length_m"] > plain.search_metadata["shortest_graph_path_length_m"]+.005
    points=np.column_stack([np.linspace(0.,.0685,138),np.zeros(138),np.ones(138)])
    density=executor.planner._density(points,executor.field,executor.compact)
    straight_risk_cost=float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)*(density[:-1]+density[1:])*.5))
    assert weighted.start_potential < straight_risk_cost
    assert all(executor.checked.segment_valid(a,b)for a,b in zip(detour[:-1],detour[1:]))


def test_cache_reuses_same_cost_field_but_includes_uncertainty_weight_and_continuous_boxes():
    executor=MetricFieldExecutorV9();executor.configure(state(),axis_intent(obstacles=[halo_box()],uncertainty_m=.015),scene_assessment=assessment("visible"))
    def plan():return executor.planner.plan(executor.field,state()[:3],goal_xyz=executor.goal,compact_prediction=executor.compact)
    assert not plan().search_metadata["cache_hit"]
    assert plan().search_metadata["cache_hit"]
    executor.planner.risk_regions[0]["sigma_m"] += .002
    assert not plan().search_metadata["cache_hit"]
    assert plan().search_metadata["cache_hit"]
    executor.planner.risk_weight=4.
    assert not plan().search_metadata["cache_hit"]
    executor.planner.hard_boxes[0][0][0] += .0001
    assert not plan().search_metadata["cache_hit"]


def test_thin_continuous_hard_wall_cannot_be_crossed_even_when_no_grid_vertex_is_masked():
    executor=MetricFieldExecutorV9();executor.configure(state(),axis_intent(),scene_assessment=assessment())
    low=np.array([.0224,-.2,.999]);high=np.array([.0226,.2,1.001])
    executor.boxes=[(low,high)]
    executor.planner=RiskRegionDistancePlanner([(low,high)],[])
    # No scene-box mask was created in this fixture: only the continuous slab
    # test can detect a wall thinner than the 5mm lattice spacing.
    assert executor.field.query([.0225,0.,1.]) is not None
    assert not executor.checked.segment_valid([.02,0.,1.],[.025,0.,1.])
    assert executor.select(state(),Q)["status"]=="field_goal_unreachable_in_known_grid"


def test_axis_corridor_and_v8_orientation_hold_remain_active_with_soft_risk():
    executor=MetricFieldExecutorV9();executor.configure(state(),axis_intent(obstacles=[halo_box()],uncertainty_m=.025,max_actions=120),scene_assessment=assessment("visible"))
    selected=executor.select(state(),Q);assert selected["accepted"]
    assert selected["action"][2]==0.
    assert np.linalg.norm(selected["action"][3:6])*.5==pytest.approx(1e-9,abs=1e-15)
    after=state();after[0]+=.001;after[5]+=.01
    executor.observe(state(),selected["action"],after)
    corrected=executor.select(after,Q);assert corrected["action"][5]<0.
    assert not executor.retarget(executor.goal+[0.,0.,.0001])
    drift=after.copy();drift[2]+=.0031
    executor.observe(after,corrected["action"],drift)
    assert executor.select(drift,Q)["status"]=="locked_axis_pose_drift"
    assert executor.summary()["max_actions"]==120
    assert executor.summary()["grid_size"]==33 and executor.summary()["grid_spacing_m"]==.005


def test_one_supported_halo_intent_sustains_measured_motion_without_model_refresh():
    executor=MetricFieldExecutorV9();measured=state()
    value=axis_intent(obstacles=[halo_box()],uncertainty_m=.025,max_actions=120,command_limit_m=.003)
    executor.configure(measured,value,scene_assessment=assessment("visible"))
    for _ in range(35):
        selected=executor.select(measured,Q)
        assert selected["accepted"], selected["status"]
        after=measured.copy();after[:3]+=np.asarray(selected["action"][:3])*.05*.4
        assert executor.checked.segment_valid(measured[:3],after[:3])
        executor.observe(measured,selected["action"],after)
        measured=after
    report=executor.summary()
    assert report["effective_moving_actions"]==35
    assert report["measured_travel_m"]>.04
    assert report["stop_reason"] is None
    assert report["max_actions"]==120 and report["max_travel_m"]==.16
    assert not report["clearance_certified"] and not report["task_success_certified"]


def test_extra_visible_box_without_matching_rgb_neighbor_keeps_whole_configuration_strict():
    value=axis_intent(obstacles=[halo_box(),box('unmatched',low=(.055,.035,.01),high=(.060,.040,.02))],uncertainty_m=.025)
    executor=MetricFieldExecutorV9();executor.configure(state(),value,scene_assessment=assessment('visible'))
    assert not executor.summary()['soft_uncertainty_enabled']
    assert executor.summary()['scene_assessment_reason']=='visible_obstacle_without_localized_image_support'
    assert executor.select(state(),Q)['status']=='invalid_measured_start'


def test_archive_v8_four_unknown_hypotheses_remain_rejected_without_relabeling():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v8_zero_shot_001/trials/bowl_original_zero_hybrid_v8/trace.jsonl'
    if not path.exists():pytest.skip('Optional immutable V8 archive is unavailable')
    actions=[];feedback={}
    with path.open()as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='action'and row['diagnostics']['controller']=='visual_field':actions.append(row)
            if row['event']=='feedback':feedback[row['step']]=row['feedback']
    assert len(actions)==4
    for row in actions:
        d=row['diagnostics'];pred=d['prediction'];before=feedback[row['step']]['before_state'];original=copy.deepcopy(pred['field_intent'])
        executor=MetricFieldExecutorV9();executor.configure(before,pred['field_intent'],scene_assessment=pred['motion_context'])
        assert executor.select(before,Q)['status']=='invalid_measured_start'
        assert executor.intent==original
        assert all(r['geometry_mode']=='hard_full_unknown'for r in executor.summary()['region_feasibility']if r['kind']=='unknown_regions')
