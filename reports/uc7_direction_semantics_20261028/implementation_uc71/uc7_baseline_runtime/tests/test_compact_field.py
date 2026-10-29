"""Analytic fields preserve dense-grid ordering, mathematical shape, and masks."""

import copy
import json
import math

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.compact_field import (
    COMPACT_FIELD_VERSION, compact_field_schema, expand_compact_field, validate_compact_field,
)
from gpt_field.field import GridField
from gpt_field.planner import LocalPlanner
from gpt_field.schema import FieldPrediction, GridSpec


GRID = GridSpec(size=33, spacing_m=.005)


def compact(**updates):
    result = {"stage": "approach", "orientation_target_xyzw": [0, 0, 0, 1],
              "gripper_command": "open", "motion_mode": "move", "goal_offset_m": [.04, 0, 0],
              "attraction_weights": [1, 1, 1], "repulsors": [], "blocked_boxes": [], "unknown_boxes": []}
    result.update(updates)
    return result


def repulsor(**updates):
    result = {"center_offset_m": [0, 0, 0], "radius_m": .005, "weight": 3.}
    result.update(updates)
    return result


def box(lower=(0, 0, 0), upper=(0, 0, 0)):
    return {"lower_offset_m": list(lower), "upper_offset_m": list(upper)}


def index(i, j, k, size=33):
    return i * size * size + j * size + k


def test_schema_is_strict_bounded_and_requires_every_parameter():
    assert COMPACT_FIELD_VERSION == "analytic-goal-rbf-v1"
    schema = compact_field_schema()
    assert set(schema["required"]) == set(compact())

    def check_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check_objects(value)
        elif isinstance(node, list):
            for value in node:
                check_objects(value)

    check_objects(schema)
    props = schema["properties"]
    for name in ("repulsors", "blocked_boxes", "unknown_boxes"):
        assert props[name]["maxItems"] == 4
    assert props["goal_offset_m"]["items"] == {"type": "number", "minimum": -.5, "maximum": .5}
    assert props["attraction_weights"]["items"] == {"type": "number", "minimum": .1, "maximum": 10}
    assert props["repulsors"]["items"]["properties"]["weight"]["exclusiveMinimum"] == 0
    json.dumps(schema, allow_nan=False)


def test_expansion_matches_scalar_formula_and_flat_xyz_order_without_mutation():
    data = compact(goal_offset_m=[.035, -.02, .01], attraction_weights=[2, 3, .5],
                   repulsors=[repulsor(center_offset_m=[.01, -.005, .02], radius_m=.025, weight=1.7),
                              repulsor(center_offset_m=[-.03, .01, -.015], radius_m=.01, weight=.4)])
    before = copy.deepcopy(data)
    assert validate_compact_field(data) is data
    dense = expand_compact_field(data, GRID)
    assert data == before
    assert len(dense["values"]) == 35_937
    for i, j, k in ((0, 0, 0), (32, 32, 32), (16, 16, 16), (18, 11, 25), (9, 24, 15)):
        point = [(coordinate - 16) * .005 for coordinate in (i, j, k)]
        expected = math.sqrt(sum(weight * ((coordinate - goal) / .08) ** 2
                                 for coordinate, goal, weight in zip(point, data["goal_offset_m"], data["attraction_weights"])))
        for bump in data["repulsors"]:
            squared_distance = sum((coordinate - center) ** 2
                                   for coordinate, center in zip(point, bump["center_offset_m"]))
            expected += bump["weight"] * math.exp(-.5 * squared_distance / bump["radius_m"] ** 2)
        assert dense["values"][index(i, j, k)] == pytest.approx(expected, rel=1e-13)
    for key in ("stage", "orientation_target_xyzw", "gripper_command", "motion_mode"):
        assert dense[key] == data[key]
    assert FieldPrediction.from_dict(dense, GRID).to_dict() == dense
    json.dumps(dense, allow_nan=False)


def test_attraction_decreases_toward_goal_and_existing_planner_uses_world_offsets():
    data = compact(attraction_weights=[4, 1, 1])
    dense = expand_compact_field(data, GRID)
    samples = [dense["values"][index(i, 16, 16)] for i in range(16, 25)]
    assert all(left > right for left, right in zip(samples, samples[1:]))
    assert samples[-1] == pytest.approx(0.)
    center = (.17, -.23, .9)
    field = GridField(GRID, center, FieldPrediction.from_dict(dense, GRID))
    assert field.query(np.asarray(center) + [.04, 0, 0]) == pytest.approx(0., abs=1e-14)
    plan = LocalPlanner().plan(field, center)
    assert plan.delta_xyz == (.01, 0., 0.)
    assert plan.status == "move"


def test_gaussian_repulsors_add_soft_cost_without_synthesizing_masks():
    dense = expand_compact_field(compact(goal_offset_m=[0, 0, 0], repulsors=[repulsor()]), GRID)
    assert dense["values"][index(16, 16, 16)] == pytest.approx(3.)
    assert dense["values"][index(17, 16, 16)] == pytest.approx(.005 / .08 + 3 * math.exp(-.5))
    assert dense["blocked_indices"] == dense["unknown_indices"] == []
    baseline = expand_compact_field(compact(goal_offset_m=[0, 0, 0]), GRID)
    assert all(bumped >= base for bumped, base in zip(dense["values"], baseline["values"]))


def test_inclusive_degenerate_boxes_and_overlap_have_sorted_disjoint_masks():
    grid = GridSpec(size=5, spacing_m=.005)
    blocked_box = box((0, -.005, 0), (.005, .005, 0))
    data = compact(blocked_boxes=[blocked_box, copy.deepcopy(blocked_box)],
                   unknown_boxes=[box((-.005, -.005, 0), (.005, .005, 0)),
                                  box((.4, .4, .4), (.5, .5, .5))])
    dense = expand_compact_field(data, grid)
    assert dense["blocked_indices"] == [index(i, j, 2, 5) for i in (2, 3) for j in (1, 2, 3)]
    assert dense["unknown_indices"] == [index(1, j, 2, 5) for j in (1, 2, 3)]
    assert not set(dense["blocked_indices"]) & set(dense["unknown_indices"])
    assert np.isfinite(dense["values"]).all()
    unmasked = expand_compact_field(compact(), grid)
    assert dense["values"] == unmasked["values"]
    FieldPrediction.from_dict(dense, grid)


def test_parameter_limits_and_four_objects_are_accepted():
    data = compact(goal_offset_m=[-.5, .5, 0], attraction_weights=[.1, 10, 1],
                   repulsors=[repulsor(center_offset_m=[-.5, .5, 0], radius_m=radius, weight=weight)
                              for radius, weight in ((.005, 10), (.5, 1e-10), (.1, 1), (.02, 2))],
                   blocked_boxes=[box((-.5, -.5, -.5), (.5, .5, .5))] * 4,
                   unknown_boxes=[box()] * 4)
    assert validate_compact_field(data) is data
    dense = expand_compact_field(data, GRID)
    assert dense["blocked_indices"] == list(range(GRID.count))
    assert dense["unknown_indices"] == []


@pytest.mark.parametrize("updates", [
    {"stage": " "}, {"stage": False}, {"gripper_command": "grasp"}, {"motion_mode": "stop"},
    {"orientation_target_xyzw": [0, 0, 0, 0]}, {"orientation_target_xyzw": [0, 0, 0, True]},
    {"orientation_target_xyzw": [0, 0, 0, float("nan")]},
    {"goal_offset_m": [0, 0]}, {"goal_offset_m": [0, 0, .5001]},
    {"goal_offset_m": [-.5001, 0, 0]}, {"goal_offset_m": [0, True, 0]},
    {"goal_offset_m": [0, float("inf"), 0]}, {"goal_offset_m": [0, float("nan"), 0]},
    {"goal_offset_m": [10 ** 400, 0, 0]},
    {"attraction_weights": [0, 1, 1]}, {"attraction_weights": [.099, 1, 1]},
    {"attraction_weights": [10.01, 1, 1]}, {"attraction_weights": [True, 1, 1]},
    {"attraction_weights": [float("nan"), 1, 1]},
    {"repulsors": None}, {"repulsors": {}}, {"repulsors": [repulsor()] * 5},
    {"repulsors": [repulsor(radius_m=.0049)]}, {"repulsors": [repulsor(radius_m=.501)]},
    {"repulsors": [repulsor(radius_m=True)]}, {"repulsors": [repulsor(radius_m=float("nan"))]},
    {"repulsors": [repulsor(weight=0)]}, {"repulsors": [repulsor(weight=-1)]},
    {"repulsors": [repulsor(weight=10.01)]}, {"repulsors": [repulsor(weight=True)]},
    {"repulsors": [repulsor(weight=float("inf"))]},
    {"repulsors": [repulsor(center_offset_m=[.51, 0, 0])]},
    {"repulsors": [repulsor(center_offset_m=[0, float("nan"), 0])]},
    {"repulsors": [{**repulsor(), "extra": 1}]}, {"repulsors": [None]},
    {"blocked_boxes": [box()] * 5}, {"unknown_boxes": [box()] * 5},
    {"blocked_boxes": [box((.1, 0, 0), (0, 0, 0))]},
    {"unknown_boxes": [box((0, 0, 0), (0, -.1, 0))]},
    {"blocked_boxes": [box((-.51, 0, 0), (0, 0, 0))]},
    {"unknown_boxes": [box((0, 0, 0), (.51, 0, 0))]},
    {"blocked_boxes": [box((False, 0, 0), (0, 0, 0))]},
    {"unknown_boxes": [box((0, 0, 0), (float("nan"), 0, 0))]},
    {"unknown_boxes": [{"lower_offset_m": [0, 0, 0]}]},
    {"blocked_boxes": [{**box(), "extra": 0}]}, {"blocked_boxes": "all"},
])
def test_invalid_parameters_are_rejected_instead_of_clipped(updates):
    data = compact(**updates)
    with pytest.raises(SchemaError):
        validate_compact_field(data)
    with pytest.raises(SchemaError):
        expand_compact_field(data, GRID)


@pytest.mark.parametrize("data", [None, [], {}, {**compact(), "object_pose": [0, 0, 0]},
                                 {key: value for key, value in compact().items() if key != "goal_offset_m"}])
def test_top_level_shape_and_allowlist_are_strict(data):
    with pytest.raises(SchemaError):
        validate_compact_field(data)


def test_expansion_requires_a_grid_spec():
    with pytest.raises(ValueError, match="GridSpec"):
        expand_compact_field(compact(), None)
