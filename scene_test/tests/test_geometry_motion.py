"""Adversarial field/volume/motion checks, independent of the formal matrix."""
from pathlib import Path
import json
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import geometry_motion as gm
from geometry import Box, Scene
from sdf_grid import SDFGrid
from motion_reference import Body, Part, Trajectory, ExactChecker, continuous_check


def grid_for(obstacles, spacing=.001, extent=.04):
    scene = Scene("test", "test", 0, [Box(o.center, o.half, o.rotation) for o in obstacles], [])
    return SDFGrid.build(scene, spacing, lower=[-extent] * 3, upper=[extent] * 3)


def test_containment_surface_sampling_is_not_a_collision_certificate():
    body = Body("solid", parts=[Part([0, 0, 0], [.020, .012, .010])])
    obstacles = [Part([.0041, .0022, -.0017], [.0018] * 3)]
    grid = grid_for(obstacles)
    assert ExactChecker(body, obstacles).pose(np.zeros(3), np.eye(3))["status"] == "blocked"
    assert gm.FieldChecker(grid, body, mode="naive_surface").pose(np.zeros(3), np.eye(3))["status"] == "free"
    assert gm.FieldChecker(grid, body).pose(np.zeros(3), np.eye(3))["status"] == "blocked"


def test_tiny_obstacle_between_volume_samples_cannot_be_certified_free():
    body = Body("solid", parts=[Part([0, 0, 0], [.006] * 3)])
    obstacles = [Part([.001] * 3, [.0002] * 3)]
    grid = grid_for(obstacles, spacing=.0005, extent=.012)
    checker = gm.FieldChecker(grid, body)
    assert grid.query(checker.volume_points).min() > 0
    assert ExactChecker(body, obstacles).pose(np.zeros(3), np.eye(3))["status"] == "blocked"
    assert checker.pose(np.zeros(3), np.eye(3))["status"] == "unknown"


def test_three_mm_clearance_resolves_with_finer_grid():
    body = Body("box", parts=[Part([0, 0, 0], [.014, .004, .006])])
    obstacles = [Part([0, sign * .017, 0], [.020, .010, .020]) for sign in (-1, 1)]
    assert ExactChecker(body, obstacles).pose(np.zeros(3), np.eye(3))["status"] == "free"
    assert gm.FieldChecker(grid_for(obstacles, .004), body).pose(np.zeros(3), np.eye(3))["status"] == "unknown"
    assert gm.FieldChecker(grid_for(obstacles, .001), body).pose(np.zeros(3), np.eye(3))["status"] == "free"


def test_sphere_uses_radius_not_only_center_sign():
    obstacle = Part([0, 0, 0], [.004] * 3)
    grid = grid_for([obstacle])
    body = Body("sphere", sphere_radius=.006)
    result = gm.FieldChecker(grid, body).pose(np.array([.008, 0, 0]), np.eye(3))
    assert grid.query(np.array([[.008, 0, 0]]))[0] > 0
    assert result["status"] == "blocked"


def test_body_outside_roi_is_unknown_even_if_center_is_inside():
    grid = grid_for([Part([-.01, -.01, -.01], [.002] * 3)], extent=.02)
    result = gm.FieldChecker(grid, Body("sphere", sphere_radius=.008)).pose(np.array([.015, 0, 0]), np.eye(3))
    assert result["status"] == "unknown"
    assert result["reason"] == "body_outside_grid_roi"


def test_missing_bound_metadata_is_never_treated_as_zero():
    grid = SDFGrid(np.ones((5, 5, 5)), .01, [-.02] * 3)
    checker = gm.FieldChecker(grid, Body("sphere", sphere_radius=.002))
    assert checker.pose(np.zeros(3), np.eye(3))["reason"] == "missing_vertex_error_bound"
    assert checker.scalar_queries == 0


def test_continuous_field_catches_midpath_collision_with_free_endpoints():
    body = Body("sphere", sphere_radius=.004)
    obstacles = [Part([0, 0, 0], [.006] * 3)]
    checker = gm.FieldChecker(grid_for(obstacles), body)
    trajectory = Trajectory([-.025, 0, 0], [.025, 0, 0])
    assert all(checker.pose(*trajectory.pose(t))["status"] == "free" for t in (0, 1))
    result = continuous_check(checker, trajectory, body)
    assert result["status"] == "blocked"
    assert result["witness_time"] == .5


def test_static_near_boundary_stays_unknown_without_exhausting_budget():
    body = Body("sphere", sphere_radius=.008)
    obstacles = [Part([0, sign * .0185, 0], [.02, .01, .02]) for sign in (-1, 1)]
    checker = gm.FieldChecker(grid_for(obstacles, .004), body)
    result = continuous_check(checker, Trajectory([0, 0, 0], [0, 0, 0]), body)
    assert result["status"] == "unknown"
    assert result["pose_queries"] == 3


def test_cases_fixed_and_references_resolve_expected_outcomes():
    cases = gm.make_cases()
    assert len(cases) == len({case["name"] for case in cases}) == 15
    for case in cases:
        gm.check_case_geometry(case)
        result = continuous_check(ExactChecker(case["body"], case["obstacles"]), case["trajectory"],
                                  case["body"], max_depth=16, node_budget=4097, motion_tolerance=1e-7)
        assert result["status"] == case["expected_reference_path"], case["name"]


def test_confusion_keeps_unknown_out_of_false_free_denominator():
    result = gm.confusion([("blocked", "free"), ("blocked", "unknown"), ("free", "free"), ("unknown", "free")])
    assert result["false_free_count"] == 1
    assert result["false_free_fraction_of_reference_blocked"] == .5
    assert result["false_free_fraction_of_resolved_reference_predicted_free"] == .5
    assert result["reference_unresolved_count"] == 1


def test_existing_output_refused_without_modification(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "manifest.json"
    marker.write_text('{"old":true}')
    args = gm.parser().parse_args(["--output", str(out), "--cases", "sphere_gap_loose", "--spacings", ".004"])
    with pytest.raises(FileExistsError):
        gm.run(args)
    assert marker.read_text() == '{"old":true}'


def test_failed_build_preserved_as_failed_configuration(tmp_path, monkeypatch):
    out = tmp_path / "failed"
    monkeypatch.setattr(gm, "plot_cases", lambda *args: None)
    monkeypatch.setattr(gm, "plot_results", lambda *args: None)
    def fail(*args, **kwargs):
        raise RuntimeError("injected build failure")
    monkeypatch.setattr(gm.SDFGrid, "build", fail)
    args = gm.parser().parse_args(["--output", str(out), "--cases", "sphere_gap_loose", "--spacings", ".004"])
    assert gm.run(args) == 1
    result = json.loads((out / "metrics.json").read_text())
    assert result["status"] == "completed_with_failures"
    assert result["configurations"][0]["status"] == "failed"
    assert result["aggregates"]["0.004"]["failed_configurations"] == 1
    assert "injected build failure" in (out / "metrics.csv").read_text()


def test_source_integrity_failure_prevents_success(tmp_path, monkeypatch):
    out = tmp_path / "integrity"
    monkeypatch.setattr(gm, "plot_cases", lambda *args: None)
    monkeypatch.setattr(gm, "plot_results", lambda *args: None)
    original = gm.sha256
    calls = {}
    def fake_hash(path):
        path = Path(path)
        value = original(path)
        if path.parent == gm.HERE:
            calls[path.name] = calls.get(path.name, 0) + 1
            if calls[path.name] >= 2:
                return "simulated-mismatch"
        return value
    monkeypatch.setattr(gm, "sha256", fake_hash)
    args = gm.parser().parse_args(["--output", str(out), "--cases", "sphere_gap_loose", "--spacings", ".02"])
    assert gm.run(args) == 1
    assert json.loads((out / "manifest.json").read_text())["status"] == "source_integrity_failed"
    assert json.loads((out / "metrics.json").read_text())["status"] == "source_integrity_failed"
