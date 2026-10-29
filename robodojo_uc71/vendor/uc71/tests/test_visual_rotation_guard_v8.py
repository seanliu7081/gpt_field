import copy

import pytest

from gpt_field.visual_rotation_guard_v8 import projected_rotation_risk


def context():
    return {"purpose": "task_reorientation", "tool_external_roi_uv": [.3, .2, .5, .4],
            "nearby_obstacles": [{"id": "nearby_fixture", "external_roi_uv": [.57, .25, .67, .45]}],
            "obstacle_assessment": "present"}


def test_rotation_sweep_can_intersect_when_current_tool_rectangle_does_not():
    ctx = context()
    commands = [[0., 0., 0., 0., 0., .16, 0.]]*5
    original = copy.deepcopy(commands)
    result = projected_rotation_risk(ctx, commands)
    assert not result["accepted"]
    assert result["reason"] == "rotation_projected_obstacle_risk"
    assert result["intersecting_obstacle_ids"] == ["nearby_fixture"]
    assert result["requested_angular_bound_rad"] == pytest.approx(.4)
    assert commands == original and not result["clearance_certified"]


def test_task_rotation_with_clear_declared_projection_passes_unchanged():
    ctx = context()
    ctx["nearby_obstacles"][0]["external_roi_uv"] = [.8, .7, .9, .8]
    result = projected_rotation_risk(ctx, [[0., 0., 0., 0., 0., .04, 0.]]*5)
    assert result["accepted"] and not result["clearance_certified"]


def test_reference_repair_never_authorizes_rotation_even_with_empty_obstacle_list():
    ctx = context()
    ctx.update(purpose="reference_repair", nearby_obstacles=[], obstacle_assessment="none_relevant")
    result = projected_rotation_risk(ctx, [[0., 0., 0., 0., 0., .02, 0.]]*5)
    assert not result["accepted"]
    assert result["reason"] == "rotation_for_reference_repair_not_allowed"


@pytest.mark.parametrize("change,reason", [
    ({"tool_external_roi_uv": None}, "rotation_tool_envelope_unavailable"),
    ({"obstacle_assessment": "unobservable"}, "rotation_neighborhood_unobservable"),
    ({"tool_external_roi_uv": [.0, .1, .15, .3]}, "rotation_projected_envelope_outside_view"),
])
def test_unknown_or_clipped_rotation_footprint_requests_changed_action(change, reason):
    ctx = context(); ctx.update(change)
    result = projected_rotation_risk(ctx, [[0., 0., 0., 0., 0., .04, 0.]]*5)
    assert not result["accepted"] and result["reason"] == reason


def test_absent_geometry_does_not_globally_block_nonrotating_task_actions():
    commands = [[.02, -.01, 0., 0., 0., 0., -1.]]*5
    assert projected_rotation_risk(None, commands)["accepted"]


@pytest.mark.parametrize("actions", [[[0.]*7]*4, [[0.]*6+[float("nan")]]*5, [[2.]+[0.]*6]*5])
def test_invalid_commands_are_not_silently_sanitized(actions):
    with pytest.raises(ValueError):
        projected_rotation_risk(context(), actions)
