"""Prescribed finite-body motions in captured static LIBERO scenes.

Run only after the three-scene SDF reconstruction matrix is complete. Paths
are fixed from captured geometry before any field values are inspected.
This evaluates geometric collision queries, not a robot controller or planner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import numpy as np
from scipy.spatial.transform import Rotation
from sdf_grid import SDFGrid
from motion_reference import Body, Part, Trajectory, ExactChecker, continuous_check, TOL
from geometry_motion import FieldChecker, sampled_path, aggregates, plot_results, write_csv, write_json, sha256

SPACINGS = [.004, .002, .001]
SNAPSHOTS = ["bowl002_initial", "drawer003_closed_step60", "drawer003_open_step552"]


def parts_from_snapshot(snapshot):
    return [Part(item["center_m"], item["half_extents_m"], item["rotation_local_to_world"])
            for item in snapshot["boxes"]]


def swept_aabb_filter(body, trajectory, obstacles):
    """Conservative broad phase over the complete translated/rotated solid.

    Every body point stays within body.radius of the linearly interpolated
    origin. Strict AABB separation therefore proves a skipped obstacle cannot
    collide at any time. Remaining obstacles are checked by independent SAT.
    """
    lower = np.minimum(trajectory.start, trajectory.end) - body.radius
    upper = np.maximum(trajectory.start, trajectory.end) + body.radius
    kept, kept_ids, skipped = [], [], []
    for i, obstacle in enumerate(obstacles):
        extent = np.abs(obstacle.rotation) @ obstacle.half
        lo, hi = obstacle.center - extent, obstacle.center + extent
        gaps = np.maximum(lo - upper, lower - hi)
        axis = int(np.argmax(gaps))
        if gaps[axis] > TOL:
            skipped.append({"obstacle_index": i, "separating_world_axis": axis,
                            "separation_lower_bound_m": float(gaps[axis])})
        else:
            kept.append(obstacle)
            kept_ids.append(i)
    return kept, {"swept_body_aabb_lower_m": lower.tolist(), "swept_body_aabb_upper_m": upper.tolist(),
                  "body_radius_about_motion_origin_m": body.radius, "kept_indices": kept_ids,
                  "kept_count": len(kept), "total_count": len(obstacles), "skipped": skipped,
                  "proof": "linear_origin_segment_inflated_by_entire_compound_body_radius; strict_AABB_separation"}


def plane_exclusion_proof(body, trajectory, planes):
    """All archived floors must stay below the entire swept body, or refuse."""
    records = []
    for plane in planes:
        normal = np.asarray(plane["normal_world"], float)
        center = np.asarray(plane["center_m"], float)
        if (normal.shape != (3,) or center.shape != (3,) or not np.all(np.isfinite(normal))
                or not np.all(np.isfinite(center)) or not np.isclose(np.linalg.norm(normal), 1., atol=1e-12, rtol=0)):
            raise ValueError("exported plane must have a finite unit normal and finite center")
        minimum = min(np.dot(normal, trajectory.start - center), np.dot(normal, trajectory.end - center)) - body.radius
        if minimum <= TOL:
            raise ValueError("trajectory could encounter an exported plane; box-only oracle is insufficient")
        records.append({"geom_id": plane["geom_id"], "minimum_plane_clearance_bound_m": float(minimum),
                        "proof": "minimum_endpoint_origin_signed_plane_distance_minus_entire_body_radius"})
    return records


def make_cases(snapshots):
    by_name = {snapshot["name"]: snapshot for snapshot in snapshots}
    if set(by_name) != set(SNAPSHOTS):
        raise ValueError("expected exactly the three declared LIBERO snapshots")
    cases = []
    for name in SNAPSHOTS:
        snapshot = by_name[name]
        if name == "bowl002_initial":
            bounds = snapshot["object_bounds"]["akita_black_bowl_1"]
            lower, upper = np.asarray(bounds["lower_m"]), np.asarray(bounds["upper_m"])
            center = (lower + upper) / 2
            start = np.array([center[0], center[1], upper[2] + .020])
            end = np.array([center[0], center[1], lower[2] + .55 * (upper[2] - lower[2])])
            bodies = [("sphere_6mm", Body("sphere_radius6mm", sphere_radius=.006), False),
                      ("small_box", Body("box_half8_8_4mm", parts=[Part([0, 0, 0], [.008, .008, .004])]), False),
                      ("wide_bar", Body("bar_half48_5_4mm", parts=[Part([0, 0, 0], [.048, .005, .004])]), False)]
            anchor = {"object": "akita_black_bowl_1", "object_bounds": bounds,
                      "start_rule": "bbox_xy_center_and_bbox_max_z_plus20mm",
                      "end_rule": "bbox_xy_center_and_min_z_plus0.55_times_height"}
        else:
            matches = [box for box in snapshot["boxes"] if box["name"] == "wooden_cabinet_1_g11"]
            if len(matches) != 1:
                raise ValueError("top drawer front-board anchor must identify one geometry")
            front = np.asarray(matches[0]["center_m"])
            start = np.array([front[0], front[1] - .050, 1.150])
            end = np.array([front[0], front[1] - .050, front[2]])
            gripper = Body("three_box_gripper_proxy", parts=[
                Part([-.014, 0, 0], [.004, .016, .005]),
                Part([.006, -.013, 0], [.016, .003, .005]),
                Part([.006, .013, 0], [.016, .003, .005])])
            bodies = [("sphere_6mm", Body("sphere_radius6mm", sphere_radius=.006), False),
                      ("small_box", Body("box_half10_8_4mm", parts=[Part([0, 0, 0], [.010, .008, .004])]), False),
                      ("rotating_gripper_proxy", gripper, True)]
            anchor = {"geom_id": matches[0]["geom_id"], "geom_name": matches[0]["name"],
                      "geom_center_m": front.tolist(),
                      "xy_rule": "front_board_center_x_and_center_y_minus50mm",
                      "z_rule": "1.150m_down_to_front_board_center_z"}
        obstacles = parts_from_snapshot(snapshot)
        for label, body, rotating in bodies:
            q1 = Rotation.from_euler("z", 180, degrees=True).as_quat() if rotating else None
            trajectory = Trajectory(start, end, q1=q1)
            filtered, filter_proof = swept_aabb_filter(body, trajectory, obstacles)
            cases.append({"name": f"{name}__{label}", "snapshot": name, "body": body,
                          "trajectory": trajectory, "obstacles": filtered, "all_obstacles": obstacles,
                          "anchor": anchor, "filter_proof": filter_proof,
                          "plane_exclusion": plane_exclusion_proof(body, trajectory, snapshot["planes"])})
    return cases


def serialize_case(case):
    return {"name": case["name"], "snapshot": case["snapshot"], "body": case["body"].to_dict(),
            "trajectory": case["trajectory"].to_dict(), "geometry_only_anchor": case["anchor"],
            "obstacle_broadphase": case["filter_proof"], "plane_exclusion": case["plane_exclusion"],
            "selection_note": "fixed geometric construction; no SDF values or outcomes used to select or tune this path"}


def guarded_field_checker(grid, body, guard_m):
    if not np.isfinite(guard_m) or guard_m < 0:
        raise ValueError("reference arithmetic allowance must be finite and nonnegative")
    checker = FieldChecker(grid, body, body_spacing=.002)
    if checker.error_bound is not None:
        checker.error_bound += guard_m
    return checker


def field_index(sdf_run, captures=None):
    """Accept only complete reconstruction data with all nine declared grids."""
    sdf_run = Path(sdf_run)
    manifest = json.loads((sdf_run / "manifest.json").read_text())
    metrics = json.loads((sdf_run / "metrics.json").read_text())
    if manifest.get("status") != "complete" or metrics.get("status") != "complete":
        raise ValueError("real-scene SDF reconstruction must finish successfully before motion evaluation")
    for record in (manifest, metrics):
        for key in ("sources_unchanged", "frozen_sources_unchanged", "artifacts_unchanged"):
            integrity = record.get(key)
            if not isinstance(integrity, dict) or not integrity or any(value is not True for value in integrity.values()):
                raise ValueError(f"SDF integrity check missing or failed: {key}")
        if record.get("capture_unchanged") is not True:
            raise ValueError("SDF capture integrity check missing or failed")
    if captures is not None:
        frozen_capture = sdf_run / "reference" / "snapshots.json"
        digest = manifest.get("capture_sha256")
        if (not isinstance(digest, str) or not frozen_capture.is_file()
                or sha256(captures) != digest or sha256(frozen_capture) != digest):
            raise ValueError("supplied capture geometry does not match the SDF's frozen capture hash")
    index = {}
    for name in SNAPSHOTS:
        for spacing in SPACINGS:
            path = sdf_run / "fields" / f"{name}_h{spacing * 1000:g}mm.npz"
            if not path.is_file():
                raise ValueError(f"required SDF field missing: {path}")
            matches = [row for row in metrics.get("configurations", [])
                       if row.get("snapshot") == name and row.get("spacing_m") == spacing]
            if len(matches) != 1 or matches[0].get("status") != "ok":
                raise ValueError("SDF configuration must have one successful matching record")
            record = matches[0]
            if record.get("field") != str(path.relative_to(sdf_run)) or record.get("field_sha256") != sha256(path):
                raise ValueError("SDF field path/hash disagrees with completed reconstruction")
            index[(name, spacing)] = path
    return index, manifest, metrics


def reference_results(cases):
    output = {}
    for case in cases:
        checker = ExactChecker(case["body"], case["obstacles"])
        started = time.perf_counter()
        path = continuous_check(checker, case["trajectory"], case["body"],
                                max_depth=16, node_budget=4097, motion_tolerance=1e-7)
        poses = [{"time": float(t), **checker.pose(*case["trajectory"].pose(t))} for t in np.linspace(0, 1, 9)]
        output[case["name"]] = {"path": path, "poses": poses, "elapsed_s": time.perf_counter() - started,
                               "reference_unresolved_is_reported_not_forced": True}
    return output


def run(args):
    if args.max_depth < 0 or args.node_budget < 1 or not np.isfinite(args.reference_guard) or args.reference_guard < 0:
        raise ValueError("invalid adaptive limits or reference arithmetic allowance")
    captures = Path(args.captures)
    capture_manifest = json.loads((captures.parent / "manifest.json").read_text())
    capture_bytes = captures.read_bytes()
    capture_data = json.loads(capture_bytes)
    loaded_capture_hash = hashlib.sha256(capture_bytes).hexdigest()
    if capture_manifest.get("status") != "complete" or capture_data.get("status") != "complete" or not capture_manifest.get("source_unchanged"):
        raise ValueError("static captures must be complete and unchanged")
    index, sdf_manifest, _ = field_index(args.sdf_run, captures)
    if loaded_capture_hash != sdf_manifest["capture_sha256"] or loaded_capture_hash != capture_manifest.get("snapshots_sha256"):
        raise ValueError("loaded capture bytes disagree with captured/SDF geometry provenance")
    cases = make_cases(capture_data["snapshots"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("source", "reference", "figures", "configuration_metrics"):
        (out / folder).mkdir()
    results, manifest = [], {"status": "initializing"}
    started = time.perf_counter()
    try:
        sources = [Path(__file__), BASE / "motion_reference.py", BASE / "geometry_motion.py", BASE / "sdf_grid.py"]
        source_hashes = {}
        for source in sources:
            destination = out / "source" / source.name
            shutil.copy2(source, destination)
            source_hashes[str(source)] = sha256(destination)
            if sha256(source) != source_hashes[str(source)]:
                raise ValueError("source changed during snapshot")
        inputs = [captures, captures.parent / "manifest.json", Path(args.sdf_run) / "manifest.json",
                  Path(args.sdf_run) / "metrics.json", Path(args.sdf_run) / "reference" / "snapshots.json"] + list(index.values())
        input_hashes = {str(path): sha256(path) for path in inputs}
        frozen_capture = out / "reference" / "snapshots.json"
        frozen_capture.write_bytes(capture_bytes)
        settings = {"cases": 9, "configuration_count": 27, "spacing_m": SPACINGS,
                    "body_lattice_spacing_m": .002, "max_depth": args.max_depth,
                    "midpoint_node_budget": args.node_budget, "motion_tolerance_m": .00001,
                    "extra_reference_arithmetic_allowance_m": args.reference_guard,
                    "reference_allowance_note": "declared conservative numerical allowance; conditional bound, not an empirical global proof",
                    "reference_max_depth": 16, "reference_node_budget": 4097,
                    "reference_motion_tolerance_m": 1e-7,
                    "baseline": "nine surface-sampled poses and two exact endpoints; neither is a continuous certificate",
                    "scope": "static archived collision geometry and virtual rigid probes; no robot actions or new model calls"}
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "settings": settings, "sources_sha256": source_hashes, "inputs_sha256": input_hashes,
                    "new_model_calls": 0, "new_robot_actions": 0,
                    "sdf_source": str(Path(args.sdf_run)), "capture_source": str(captures)}
        write_json(out / "manifest.json", manifest)
        write_json(out / "cases.json", [serialize_case(case) for case in cases])
        write_json(out / "settings.json", settings)
        artifact_hashes = {"cases.json": sha256(out / "cases.json"),
                           "settings.json": sha256(out / "settings.json"),
                           "reference/snapshots.json": loaded_capture_hash}
        references = reference_results(cases)
        write_json(out / "reference_checks.json", references)
        artifact_hashes["reference_checks.json"] = sha256(out / "reference_checks.json")
        manifest["artifacts_sha256"] = artifact_hashes
        write_json(out / "manifest.json", manifest)
        for snapshot in SNAPSHOTS:
            for spacing in SPACINGS:
                field_path = index[(snapshot, spacing)]
                grid = SDFGrid.load(field_path)
                if not np.isclose(grid.spacing, spacing, atol=1e-14, rtol=0):
                    raise ValueError("grid spacing disagrees with frozen field index")
                for case in [case for case in cases if case["snapshot"] == snapshot]:
                    name = f"{case['name']}_h{spacing * 1000:g}mm"
                    result = {"case": case["name"], "snapshot": snapshot, "spacing_m": spacing}
                    print(f"START {name}", flush=True)
                    try:
                        body, trajectory = case["body"], case["trajectory"]
                        bounded = guarded_field_checker(grid, body, args.reference_guard)
                        naive = FieldChecker(grid, body, body_spacing=.002, mode="naive_surface")
                        t0 = time.perf_counter()
                        path = continuous_check(bounded, trajectory, body, max_depth=args.max_depth,
                                                node_budget=args.node_budget, motion_tolerance=.00001)
                        elapsed = time.perf_counter() - t0
                        reference = references[case["name"]]
                        poses = []
                        for truth in reference["poses"]:
                            t = truth["time"]
                            poses.append({"time": t, "reference": truth,
                                          "bounded_sdf": bounded.pose(*trajectory.pose(t)),
                                          "naive_surface": naive.pose(*trajectory.pose(t))})
                        naive_path = sampled_path([{"time": p["time"], **p["naive_surface"]} for p in poses])
                        endpoint_path = sampled_path([reference["poses"][0], reference["poses"][-1]])
                        result.update(status="ok", reference_path=reference["path"], poses=poses,
                            paths={"bounded_sdf": path, "naive_surface_9_poses": naive_path,
                                   "exact_endpoints_only": endpoint_path},
                            interpolation_bound_m=bounded.error_bound, body_cover_radius_m=bounded.cover_radius,
                            vertex_quantization_max_m=grid.vertex_quantization_max_m,
                            added_reference_arithmetic_allowance_m=args.reference_guard,
                            body_volume_sample_count=len(bounded.volume_points),
                            timing={"build_s": 0., "bounded_path_s": elapsed},
                            field=str(field_path), field_sha256=input_hashes[str(field_path)],
                            field_bytes=field_path.stat().st_size,
                            field_note="read-only reuse from completed real-scene SDF reconstruction",
                            obstacle_broadphase=case["filter_proof"], plane_exclusion=case["plane_exclusion"])
                        print(f"DONE {name} reference={reference['path']['status']} bounded={path['status']}", flush=True)
                    except Exception:
                        result.update(status="failed", error=traceback.format_exc())
                        print(f"FAILED {name}: {result['error']}", flush=True)
                    results.append(result)
                    write_json(out / "configuration_metrics" / f"{name}.json", result)
                    write_json(out / "metrics.json", {"status": "running", "configurations": results})
                    write_csv(out / "metrics.csv", results)
                del grid
        plot_results(out / "figures" / "trajectory_results.png", results)
        source_integrity = {path: sha256(path) == digest for path, digest in source_hashes.items()}
        frozen_source_integrity = {Path(path).name: sha256(out / "source" / Path(path).name) == digest
                                   for path, digest in source_hashes.items()}
        input_integrity = {path: sha256(path) == digest for path, digest in input_hashes.items()}
        artifact_integrity = {path: sha256(out / path) == digest for path, digest in artifact_hashes.items()}
        status = "complete" if len(results) == 27 and all(r["status"] == "ok" for r in results) else "completed_with_failures"
        if not all(source_integrity.values()) or not all(input_integrity.values()) or not all(frozen_source_integrity.values()) or not all(artifact_integrity.values()):
            status = "source_or_input_integrity_failed"
        summary = {"status": status, "configurations": results, "aggregates": aggregates(results),
                   "sources_unchanged": source_integrity, "inputs_unchanged": input_integrity,
                   "frozen_sources_unchanged": frozen_source_integrity, "artifacts_unchanged": artifact_integrity,
                   "elapsed_s": time.perf_counter() - started,
                   "scope": "static known LIBERO geometry; prescribed virtual bodies only",
                   "unknown_interpretation": "unresolved under the bound or budget; neither free nor confirmed collision"}
        write_json(out / "metrics.json", summary)
        manifest.update(status=status, sources_unchanged=source_integrity, inputs_unchanged=input_integrity,
                        frozen_sources_unchanged=frozen_source_integrity, artifacts_unchanged=artifact_integrity,
                        cases_sha256=sha256(out / "cases.json"), settings_sha256=sha256(out / "settings.json"),
                        reference_checks_sha256=sha256(out / "reference_checks.json"),
                        elapsed_s=summary["elapsed_s"], finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        print(f"FINISHED {status} configurations={len(results)}", flush=True)
        return 0 if status == "complete" else 1
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "aborted"
        detail = traceback.format_exc()
        manifest.update(status=status, error=detail, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        write_json(out / "metrics.json", {"status": status, "configurations": results, "error": detail})
        print(detail, flush=True)
        return 130 if status == "interrupted" else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--sdf-run", required=True, help="completed three-scene SDF reconstruction run")
    p.add_argument("--captures", default=str(HERE / "captures_001" / "snapshots.json"))
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--node-budget", type=int, default=257)
    p.add_argument("--reference-guard", type=float, default=1e-6, help="extra reference arithmetic allowance in metres")
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
