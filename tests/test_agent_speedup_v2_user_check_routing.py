"""Offline route behavior; no simulator, model calls, or scene truth."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_user_check_routing import (
    RouteUnavailable, TransitObstacleMap, TransitRoute,
)


def make_route(start, goal, boxes=(), clearance=.01):
    return TransitRoute(start, goal, [0., 1., 0., 0.],
                        {"avoid_boxes": list(boxes), "clearance_m": clearance},
                        max_command_m=.02, position_tolerance_m=.004)


def follow(route, start, goal):
    """Ideal measured tracking isolates routing from robot dynamics."""
    position, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    points = [position.copy()]
    for _ in range(100):
        if np.linalg.norm(position - goal) <= .004:
            break
        delta, _ = route.command(position)
        delta = np.asarray(delta)
        assert 0 < np.linalg.norm(delta) <= .02 + 1e-12
        position = position + delta
        points.append(position.copy())
    assert np.linalg.norm(position - goal) <= .004
    return np.asarray(points)


def test_transit_route_reaches_full_fifty_centimetre_goal():
    start, goal = [0., 0., .8], [.5, 0., .8]
    route = make_route(start, goal)
    points = follow(route, start, goal)
    assert points[-1, 0] > .49
    assert route.diagnostics()["path_xyz_m"][-1] == pytest.approx(goal)


@pytest.mark.parametrize("configured,expected", [(None, .02), ("0.10", .10)])
def test_configured_route_commands_reach_half_metre_without_hidden_caps(configured, expected):
    # Configuration is read at import by the adapter, route and tracker. A fresh
    # process checks their actual shared configuration rather than monkeypatching
    # only one layer and accidentally hiding a stale downstream limit.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if configured is None:
        environment.pop("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M", None)
    else:
        environment["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = configured
    result = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import json
        import numpy as np
        from gpt_field.action import USER_CHECK_TRANSIT_ACTION_M
        from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute

        start, goal = np.array([0., 0., .8]), np.array([.3, .4, .8])
        route = TransitRoute(start, goal, [0., 1., 0., 0.],
                             {"avoid_boxes": [], "clearance_m": 0.},
                             max_command_m=USER_CHECK_TRANSIT_ACTION_M,
                             position_tolerance_m=.004)
        position, norms = start.copy(), []
        for _ in range(100):
            delta, _ = route.command(position)
            delta = np.array(delta)
            norms.append(float(np.linalg.norm(delta)))
            assert route.field.segment_valid(position, position + delta)
            position += delta
            if np.linalg.norm(position - goal) <= .004:
                break
        assert np.linalg.norm(position - goal) <= .004
        if USER_CHECK_TRANSIT_ACTION_M > .05:
            # The larger lookahead must still follow a real detour without
            # cutting across the obstacle when selecting a command target.
            obstacle = {"lower_xyz_m": [.15, -.035, .755],
                        "upper_xyz_m": [.25, .035, .845]}
            detour = TransitRoute(start, [.4, 0., .8], [0., 1., 0., 0.],
                                  {"avoid_boxes": [obstacle], "clearance_m": .015},
                                  max_command_m=USER_CHECK_TRANSIT_ACTION_M,
                                  position_tolerance_m=.004)
            point = start.copy()
            for _ in range(100):
                delta, _ = detour.command(point)
                delta = np.array(delta)
                assert np.linalg.norm(delta) <= .10 + 1e-12
                assert detour.field.segment_valid(point, point + delta)
                point += delta
                if np.linalg.norm(point - [.4, 0., .8]) <= .004:
                    break
            assert np.linalg.norm(point - [.4, 0., .8]) <= .004
            assert detour.diagnostics()["planner"]["calls"] > 0
        print(json.dumps({"first_command_m": norms[0], "max_command_m": max(norms),
                          "commands": len(norms), "last_position": position.tolist(),
                          "goal": route.diagnostics()["path_xyz_m"][-1]}))
    """)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(result.stdout)
    assert measured["first_command_m"] == pytest.approx(expected)
    assert measured["max_command_m"] == pytest.approx(expected)
    assert measured["last_position"] == pytest.approx([.3, .4, .8], abs=.004)
    assert measured["goal"] == pytest.approx([.3, .4, .8])
    if configured:
        assert measured["commands"] == 5


def test_commanded_detour_clears_expanded_obstacle_and_keeps_original_goal():
    start, goal = [0., 0., .8], [.4, 0., .8]
    box = {"lower_xyz_m": [.15, -.035, .755],
           "upper_xyz_m": [.25, .035, .845]}
    clearance = .015
    route = make_route(start, goal, [box], clearance)
    points = follow(route, start, goal)
    # Check what would actually be commanded, including tracker shortcuts,
    # rather than checking only the planner's own reported vertices.
    low = np.asarray(box["lower_xyz_m"]) - clearance
    high = np.asarray(box["upper_xyz_m"]) + clearance
    for before, after in zip(points[:-1], points[1:]):
        count = int(np.ceil(np.linalg.norm(after - before) / .002)) + 1
        sampled = np.linspace(before, after, count)
        assert not np.any(np.all((sampled >= low) & (sampled <= high), axis=1))
    assert np.max(np.linalg.norm(points[:, 1:] - [0., .8], axis=1)) > .05
    assert route.diagnostics()["path_xyz_m"][-1] == pytest.approx(goal)


def test_obstructed_goal_returns_unavailable_without_straight_fallback():
    box = {"lower_xyz_m": [.17, -.02, .78],
           "upper_xyz_m": [.23, .02, .82]}
    with pytest.raises(RouteUnavailable) as failure:
        make_route([0., 0., .8], [.2, 0., .8], [box])
    assert failure.value.reason


def test_orientation_only_route_restores_measured_translation_drift():
    route = make_route([0., 0., .8], [0., 0., .8])
    delta, _ = route.command([.006, 0., .8])
    assert delta == pytest.approx([-.006, 0., 0.])
    assert np.linalg.norm(delta) <= .02
    assert route.diagnostics()["path_xyz_m"] == [[0., 0., .8]]


@pytest.mark.parametrize("drift_x", [.014, .025])
def test_orientation_only_drift_correction_never_crosses_shared_obstacle(drift_x):
    box = {"lower_xyz_m": [.012, -.01, .79], "upper_xyz_m": [.018, .01, .81]}
    route = make_route([0., 0., .8], [0., 0., .8], [box], clearance=0.)
    # The measured hand may be inside the obstacle or beyond it. Neither case
    # permits a straight correction through the obstacle to the original XYZ.
    with pytest.raises(RouteUnavailable):
        route.command([drift_x, 0., .8])


def test_obstacle_map_reuses_replaces_and_clears_estimates_without_aliasing():
    scene = TransitObstacleMap()
    initial = {"avoid_boxes": [{"lower_xyz_m": [.1, -.02, .78],
                               "upper_xyz_m": [.2, .02, .82]}], "clearance_m": .01}
    scene.update(initial, step=10)
    first = scene.snapshot()
    scene.update(None, step=20)
    assert scene.snapshot() == first
    assert first["source"] == "model_estimate"
    assert first["updated_at_step"] == 10

    # Editing either the accepted input or an exported snapshot must not move
    # an obstacle that future grants will reuse.
    initial["avoid_boxes"][0]["lower_xyz_m"][0] = -1.
    first["avoid_boxes"].clear()
    assert scene.snapshot()["avoid_boxes"][0]["lower_xyz_m"][0] == .1
    previous_revision = scene.snapshot()["revision"]
    replacement = {"avoid_boxes": [{"lower_xyz_m": [.3, -.02, .78],
                                   "upper_xyz_m": [.4, .02, .82]}], "clearance_m": .02}
    scene.update(replacement, step=30)
    replaced = scene.snapshot()
    assert replaced["avoid_boxes"] == replacement["avoid_boxes"]
    assert replaced["clearance_m"] == .02
    assert replaced["revision"] > previous_revision

    scene.update({"avoid_boxes": [], "clearance_m": 0.}, step=40)
    cleared = scene.snapshot()
    assert cleared["avoid_boxes"] == []
    assert cleared["source"] == "model_estimate"
    assert cleared["revision"] > replaced["revision"]
    assert cleared["updated_at_step"] == 40
