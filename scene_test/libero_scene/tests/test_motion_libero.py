from pathlib import Path
import json
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import motion_libero as motion
from motion_reference import Body, Part, Trajectory, ExactChecker, continuous_check
from sdf_grid import SDFGrid


def valid_integrity():
    return {"sources_unchanged": {"code": True}, "frozen_sources_unchanged": {"code": True},
            "artifacts_unchanged": {"settings": True}, "capture_unchanged": True}


def test_swept_filter_contains_compound_offsets_and_rotations():
    body = Body("offset", parts=[Part([.05, 0, 0], [.01, .003, .003])])
    trajectory = Trajectory([0, 0, 0], [0, 0, .02], q1=[0, 0, 1, 0])
    obstacles = [Part([0, .055, .01], [.005] * 3), Part([.2, 0, 0], [.01] * 3)]
    kept, proof = motion.swept_aabb_filter(body, trajectory, obstacles)
    assert len(kept) == 1
    assert proof["kept_indices"] == [0]
    assert proof["skipped"][0]["separation_lower_bound_m"] > 0
    for t in np.linspace(0, 1, 31):
        low, high = body.bounds(*trajectory.pose(t))
        assert np.all(low >= np.asarray(proof["swept_body_aabb_lower_m"]) - 1e-14)
        assert np.all(high <= np.asarray(proof["swept_body_aabb_upper_m"]) + 1e-14)


def test_filter_never_discards_touching_aabbs():
    body = Body("sphere", sphere_radius=.01)
    obstacle = Part([.02, 0, 0], [.01] * 3)
    kept, _ = motion.swept_aabb_filter(body, Trajectory([0, 0, 0], [0, 0, 0]), [obstacle])
    assert len(kept) == 1


def test_plane_exclusion_refuses_unsupported_plane_interaction():
    plane = {"geom_id": 0, "normal_world": [0, 0, 1], "center_m": [0, 0, 0]}
    body = Body("sphere", sphere_radius=.01)
    assert motion.plane_exclusion_proof(body, Trajectory([0, 0, 1], [0, 0, .2]), [plane])[0]["minimum_plane_clearance_bound_m"] > 0
    with pytest.raises(ValueError, match="could encounter"):
        motion.plane_exclusion_proof(body, Trajectory([0, 0, 1], [0, 0, .005]), [plane])


def test_extra_reference_allowance_does_not_modify_saved_quantization():
    grid = SDFGrid(np.ones((5, 5, 5)), .01, [-.02] * 3, 2e-8)
    checker = motion.guarded_field_checker(grid, Body("sphere", sphere_radius=.001), 1e-6)
    assert grid.vertex_quantization_max_m == 2e-8
    assert checker.error_bound == pytest.approx(np.sqrt(3) * .01 / 2 + 2e-8 + 1e-10 + 1e-6)


def test_incomplete_sdf_run_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="must finish"):
        motion.field_index(tmp_path)


def test_saved_field_hash_is_checked_before_motion(tmp_path):
    (tmp_path / "fields").mkdir()
    configurations = []
    paths = []
    for name in motion.SNAPSHOTS:
        for h in motion.SPACINGS:
            path = tmp_path / "fields" / f"{name}_h{h * 1000:g}mm.npz"
            path.write_bytes(b"test fixture")
            configurations.append({"snapshot": name, "spacing_m": h, "status": "ok",
                                   "field": str(path.relative_to(tmp_path)), "field_sha256": motion.sha256(path)})
            paths.append(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", **valid_integrity()}))
    (tmp_path / "metrics.json").write_text(json.dumps({"status": "complete", "configurations": configurations, **valid_integrity()}))
    assert len(motion.field_index(tmp_path)[0]) == 9
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="path/hash disagrees"):
        motion.field_index(tmp_path)


def test_capture_identity_must_match_sdf_frozen_capture(tmp_path):
    (tmp_path / "reference").mkdir()
    original = tmp_path / "reference" / "snapshots.json"
    original.write_text('{"geometry":"original"}')
    supplied = tmp_path / "supplied.json"
    supplied.write_text('{"geometry":"different"}')
    manifest = {"status": "complete", "capture_sha256": motion.sha256(original), **valid_integrity()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "metrics.json").write_text(json.dumps({"status": "complete", **valid_integrity()}))
    with pytest.raises(ValueError, match="does not match"):
        motion.field_index(tmp_path, supplied)


@pytest.mark.parametrize("key", ["sources_unchanged", "frozen_sources_unchanged", "artifacts_unchanged", "capture_unchanged"])
def test_actual_sdf_integrity_keys_are_required_and_must_pass(tmp_path, key):
    record = {"status": "complete", **valid_integrity()}
    record[key] = False if key == "capture_unchanged" else {"tampered": False}
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    (tmp_path / "metrics.json").write_text(json.dumps({"status": "complete", **valid_integrity()}))
    with pytest.raises(ValueError, match="integrity"):
        motion.field_index(tmp_path)


def test_nine_fixtures_are_geometry_only_and_filter_preserves_pose_truth():
    snapshots = json.loads((HERE / "captures_001" / "snapshots.json").read_text())["snapshots"]
    cases = motion.make_cases(snapshots)
    assert len(cases) == len({case["name"] for case in cases}) == 9
    for case in cases:
        body, trajectory = case["body"], case["trajectory"]
        filtered = ExactChecker(body, case["obstacles"])
        original = ExactChecker(body, case["all_obstacles"])
        for t in (0., .5, 1.):
            assert filtered.pose(*trajectory.pose(t))["status"] == original.pose(*trajectory.pose(t))["status"]
        assert case["filter_proof"]["kept_count"] < case["filter_proof"]["total_count"]
        if "drawer" in case["snapshot"]:
            low = np.array([-.148, -.392, .868])
            high = np.array([.188, .036, 1.168])
            for t in np.linspace(0, 1, 17):
                a, b = body.bounds(*trajectory.pose(t))
                assert np.all(a >= low) and np.all(b <= high)


def test_filtered_empty_scene_remains_free():
    body = Body("sphere", sphere_radius=.001)
    trajectory = Trajectory([0, 0, 0], [.01, 0, 0])
    filtered, _ = motion.swept_aabb_filter(body, trajectory, [Part([1, 1, 1], [.01] * 3)])
    assert not filtered
    assert continuous_check(ExactChecker(body, filtered), trajectory, body)["status"] == "free"
