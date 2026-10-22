"""Isolated geometry-motion benchmark following the SDF reconstruction stage.

Known static box geometry is rasterized; sphere/box/compound bodies move along
prescribed trajectories. This is collision-query evaluation, not robot planning.
Only a NEW output directory is written. No model calls or robot imports occur.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_deps"))
import numpy as np
from scipy.spatial.transform import Rotation
from geometry import Box, Scene
from sdf_grid import SDFGrid
from motion_reference import Body, Part, Trajectory, ExactChecker, box_box, continuous_check, TOL


class FieldChecker:
    """Conditional whole-solid certificates from sampled exact scene SDFs.

    The bound assumes a 1-Lipschitz exact SDF sampled at the grid vertices.
    Quantization metadata alone is not proof of that assumption: the runner
    builds every field from declared, disjoint analytical box obstacles.
    """
    def __init__(self, grid, body, body_spacing=.002, mode="bounded"):
        if mode not in ("bounded", "naive_surface"):
            raise ValueError("unsupported body checking method")
        self.grid, self.body, self.mode = grid, body, mode
        self.volume_points, self.surface_points, self.cover_radius = body.sample_lattice(body_spacing)
        q = grid.vertex_quantization_max_m
        self.error_bound = (np.sqrt(3) * grid.spacing / 2 + q + 1e-10
                            if q is not None and np.isfinite(q) and q >= 0 else None)
        self.scalar_queries = 0

    def pose(self, translation, rotation):
        lower, upper = self.body.bounds(translation, rotation)
        roi_margin = float(min(np.min(lower - self.grid.lower), np.min(self.grid.upper - upper)))
        if roi_margin < 0:
            return {"status": "unknown", "clearance_lower_m": None,
                    "roi_margin_m": roi_margin, "reason": "body_outside_grid_roi"}
        if self.mode == "bounded" and self.error_bound is None:
            return {"status": "unknown", "clearance_lower_m": None,
                    "roi_margin_m": roi_margin, "reason": "missing_vertex_error_bound"}
        local = self.volume_points if self.mode == "bounded" else self.surface_points
        values = self.grid.query(local @ rotation.T + translation)
        self.scalar_queries += len(values)
        if not np.all(np.isfinite(values)):
            return {"status": "unknown", "clearance_lower_m": None,
                    "roi_margin_m": roi_margin, "reason": "nonfinite_field_query"}
        minimum = float(values.min())
        sphere_radius = self.body.sphere_radius or 0.
        if self.mode == "naive_surface":
            signed = minimum - sphere_radius
            return {"status": "free" if signed > TOL else "blocked" if signed < -TOL else "unknown",
                    "clearance_lower_m": None, "roi_margin_m": roi_margin,
                    "minimum_sampled_sdf_m": minimum,
                    "reason": "sampled_heuristic_not_a_certificate"}
        # Sphere centers use exact morphological offset by the radius. For box
        # unions, every material point lies <= cover_radius from a volume node.
        lower_clearance = minimum - self.error_bound - self.cover_radius
        collision_upper = minimum + self.error_bound - sphere_radius
        if collision_upper < -TOL:
            status, reason = "blocked", "strict_collision_witness"
        elif lower_clearance > TOL:
            status, reason = "free", "whole_solid_clearance_bound"
        else:
            status, reason = "unknown", "field_or_body_sampling_uncertainty"
        witness_index = int(np.argmin(values))
        return {"status": status, "reason": reason, "clearance_lower_m": lower_clearance,
                "roi_margin_m": roi_margin, "minimum_sampled_sdf_m": minimum,
                "collision_witness_upper_m": collision_upper,
                "minimum_sample_position_m": (local[witness_index] @ rotation.T + translation).tolist(),
                "interpolation_error_bound_m": self.error_bound, "body_cover_radius_m": self.cover_radius}


def make_cases():
    """Frozen 15-case matrix; all dimensions are in metres."""
    cases = []
    offset = np.array([.0007, -.0003, .0005])

    def case(name, body, obstacles, trajectory, expected, purpose):
        scene = Scene(name, "geometry_motion", 0,
                      [Box(o.center, o.half, o.rotation) for o in obstacles], [])
        cases.append({"name": name, "body": body, "obstacles": obstacles,
                      "trajectory": trajectory, "scene": scene,
                      "expected_reference_path": expected, "purpose": purpose})

    def slit(gap):
        return [Part(offset + [0, sign * (gap / 2 + .025), 0], [.010, .025, .028]) for sign in (-1, 1)]

    sphere = Body("sphere_radius_8mm", sphere_radius=.008)
    box = Body("rigid_box_28x8x12mm", parts=[Part(np.zeros(3), [.014, .004, .006])])
    for kind, body, width in (("sphere", sphere, .016), ("box", box, .008)):
        for label, clearance in (("loose", .008), ("medium", .003), ("near", .0005), ("blocked", -.0005)):
            gap = width + 2 * clearance
            case(f"{kind}_gap_{label}", body, slit(gap),
                 Trajectory(offset + [-.060, 0, 0], offset + [.060, 0, 0]),
                 "blocked" if clearance < 0 else "free",
                 f"Slit with per-side nominal clearance {clearance * 1000:g}mm")
    q90 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    q180 = Rotation.from_euler("z", 180, degrees=True).as_quat()
    case("box_yaw90_blocked", box, slit(.018),
         Trajectory(offset + [-.060, 0, 0], offset + [.060, 0, 0], q90, q90), "blocked",
         "Same rigid body at a different orientation cannot pass the slit")
    case("box_rotation_midpath_collision", box, slit(.018),
         Trajectory(offset, offset, q1=q180), "blocked",
         "Endpoints are free but a 180-degree rotation collides in the middle")
    case("sphere_midpath_collision", Body("sphere_radius_6mm", sphere_radius=.006),
         [Part(offset, [.006, .006, .006])],
         Trajectory(offset + [-.060, 0, 0], offset + [.060, 0, 0]), "blocked",
         "Translation endpoints are free but the swept sphere hits the central obstacle")
    case("box_contains_tiny_obstacle", Body("containing_box_40x24x20mm", parts=[Part(np.zeros(3), [.020, .012, .010])]),
         [Part([.0041, .0022, -.0017], [.0018, .0018, .0018])],
         Trajectory(np.zeros(3), np.zeros(3)), "blocked",
         "A 3.6mm obstacle is fully inside the solid body; surface-only sampling misses it")
    gripper = Body("three_box_gripper_proxy", parts=[
        Part([-.014, 0, 0], [.004, .016, .005]),
        Part([.006, -.013, 0], [.016, .003, .005]),
        Part([.006, .013, 0], [.016, .003, .005])])
    case("gripper_opening_free", gripper, [Part([.008, 0, 0], [.006, .004, .004])],
         Trajectory([-.004, 0, 0], [.004, 0, 0]), "free",
         "Obstacle lies between separated fingers; a solid outer bounding box would overreject")
    case("gripper_finger_collision", gripper, [Part([.008, .013, 0], [.003, .002, .003])],
         Trajectory([-.030, 0, 0], [.030, 0, 0]), "blocked",
         "The reference-point path is clear while a finger collides")
    case("sphere_outside_roi", sphere, [Part([-.070, .060, .040], [.010, .010, .010])],
         Trajectory([.100, 0, 0], [.130, 0, 0]), "free",
         "Known analytic scene is free, but the body's sweep leaves the grid and must remain unknown")
    return cases


def case_description(case):
    return {"name": case["name"], "purpose": case["purpose"],
            "body": case["body"].to_dict(), "trajectory": case["trajectory"].to_dict(),
            "obstacles": [o.to_dict() for o in case["obstacles"]],
            "expected_reference_path": case["expected_reference_path"]}


def check_case_geometry(case):
    obstacles = case["obstacles"]
    pair_checks = []
    for i, a in enumerate(obstacles):
        if np.any(a.vertices() <= -.12) or np.any(a.vertices() >= .12):
            raise ValueError("obstacle leaves the fixed SDF ROI")
        for j in range(i):
            result = box_box(a, obstacles[j])
            if result["status"] != "free":
                raise ValueError("obstacles must be strictly disjoint for exact union SDF semantics")
            pair_checks.append({"obstacles": [j, i], **result})
    return {"passed": True, "nonoverlap_checks": pair_checks,
            "exact_sdf_basis": "minimum_of_exact_SDFs_for_strictly_disjoint_boxes"}


def sampled_path(rows):
    blocked = next((r for r in rows if r["status"] == "blocked"), None)
    return {"status": "blocked" if blocked else "free" if all(r["status"] == "free" for r in rows) else "unknown",
            "witness_time": blocked["time"] if blocked else None,
            "sample_count": len(rows), "method": "discrete_samples_only_no_continuous_certificate"}


def confusion(pairs):
    labels = ("free", "blocked", "unknown")
    table = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in pairs:
        table[truth][pred] += 1
    false_free = table["blocked"]["free"]
    blocked_count = sum(table["blocked"].values())
    predicted_free = table["free"]["free"] + false_free
    resolved = sum(table[truth][pred] for truth in ("free", "blocked") for pred in labels)
    return {"rows_reference_columns_prediction": table,
            "reference_resolved_count": resolved, "reference_unresolved_count": sum(table["unknown"].values()),
            "false_free_count": false_free,
            "false_free_fraction_of_reference_blocked": false_free / blocked_count if blocked_count else None,
            "false_free_fraction_of_resolved_reference_predicted_free": false_free / predicted_free if predicted_free else None,
            "false_blocked_count": table["free"]["blocked"],
            "prediction_unknown_fraction_on_resolved_reference":
                (table["free"]["unknown"] + table["blocked"]["unknown"]) / resolved if resolved else None}


def aggregates(results):
    output = {}
    for spacing in sorted({r["spacing_m"] for r in results}, reverse=True):
        selected = [r for r in results if r["spacing_m"] == spacing]
        successful = [r for r in selected if r["status"] == "ok"]
        methods = {}
        for method in ("bounded_sdf", "naive_surface_9_poses", "exact_endpoints_only"):
            methods[method] = confusion([(r["reference_path"]["status"], r["paths"][method]["status"]) for r in successful])
        poses = {method: confusion([(p["reference"]["status"], p[method]["status"])
                                    for r in successful for p in r["poses"]])
                 for method in ("bounded_sdf", "naive_surface")}
        output[f"{spacing:g}"] = {"spacing_m": spacing, "attempted_configurations": len(selected),
                                 "successful_configurations": len(successful),
                                 "failed_configurations": len(selected) - len(successful),
                                 "trajectory_classification": methods, "sampled_pose_classification": poses}
    return output


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(safe_json(data), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, results):
    names = ["case", "spacing_mm", "status", "reference_path", "bounded_sdf", "naive_surface_9_poses",
             "exact_endpoints_only", "interpolation_bound_mm", "body_cover_radius_mm", "build_s",
             "bounded_path_s", "bounded_pose_queries", "bounded_sdf_scalar_queries", "field_bytes", "error"]
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for result in results:
            row = {"case": result["case"], "spacing_mm": result["spacing_m"] * 1000,
                   "status": result["status"], "error": result.get("error", "")}
            if result["status"] == "ok":
                row.update(reference_path=result["reference_path"]["status"],
                           interpolation_bound_mm=result["interpolation_bound_m"] * 1000,
                           body_cover_radius_mm=result["body_cover_radius_m"] * 1000,
                           build_s=result["timing"]["build_s"], bounded_path_s=result["timing"]["bounded_path_s"],
                           bounded_pose_queries=result["paths"]["bounded_sdf"]["pose_queries"],
                           bounded_sdf_scalar_queries=result["paths"]["bounded_sdf"]["primitive_or_sdf_queries"],
                           field_bytes=result["field_bytes"])
                row.update({name: result["paths"][name]["status"] for name in
                            ("bounded_sdf", "naive_surface_9_poses", "exact_endpoints_only")})
            writer.writerow(row)
    temporary.replace(path)


def plot_cases(path, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    for ax, case in zip(axes.flat, cases):
        def draw_part(part, color, alpha):
            vertices = np.unique(part.vertices()[:, :2], axis=0) * 1000
            hull = ConvexHull(vertices)
            ax.add_patch(Polygon(vertices[hull.vertices], facecolor=color, edgecolor=color, alpha=alpha))
        for obstacle in case["obstacles"]:
            draw_part(obstacle, "firebrick", .5)
        trajectory, body = case["trajectory"], case["body"]
        for t, alpha in ((0., .18), (.5, .50), (1., .18)):
            translation, rotation = trajectory.pose(t)
            if body.sphere_radius is not None:
                ax.add_patch(Circle(translation[:2] * 1000, body.sphere_radius * 1000,
                                    facecolor="royalblue", edgecolor="navy", alpha=alpha))
            else:
                for part in body.world_parts(translation, rotation):
                    draw_part(part, "royalblue", alpha)
        ax.plot([trajectory.start[0] * 1000, trajectory.end[0] * 1000],
                [trajectory.start[1] * 1000, trajectory.end[1] * 1000], "k--", linewidth=.8)
        ax.set(title=case["name"].replace("_", " "), xlim=(-125, 145), ylim=(-85, 85),
               xlabel="X (mm)", ylabel="Y (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    fig.suptitle("Known box obstacles (red), moving solid at t=0, 0.5, 1 (blue)\nXY projections, not collision evidence; orientation and Z are checked in 3D")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    names = list(dict.fromkeys(r["case"] for r in results))
    spacings = sorted({r["spacing_m"] for r in results}, reverse=True)
    methods = ("bounded_sdf", "naive_surface_9_poses", "exact_endpoints_only")
    mapping = {"free": 0, "blocked": 1, "unknown": 2}
    cmap = ListedColormap(["#8bc8a2", "#e89c97", "#e4ce83", "#d0d0d0"])
    fig, axes = plt.subplots(1, 3, figsize=(14, max(5, len(names) * .43)))
    for ax, method in zip(axes, methods):
        values = np.full((len(names), len(spacings)), 3.)
        for result in results:
            if result["status"] == "ok":
                i, j = names.index(result["case"]), spacings.index(result["spacing_m"])
                predicted, reference = result["paths"][method]["status"], result["reference_path"]["status"]
                values[i, j] = mapping[predicted]
                text = predicted + (" !" if predicted == "free" and reference == "blocked" else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        ax.imshow(values, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(len(spacings)), [f"{h * 1000:g} mm" for h in spacings])
        ax.set_yticks(range(len(names)), names if method == methods[0] else [""] * len(names), fontsize=8)
        ax.set_title(method.replace("_", " "))
    fig.suptitle("Whole-trajectory classifications; ! = incorrectly free against independent geometry\nUnknown is unresolved, not blocked. Heuristic baselines provide no continuous guarantee.")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args):
    cases = make_cases()
    if args.cases:
        selected = set(args.cases.split(","))
        unknown = selected - {c["name"] for c in cases}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in selected]
    spacings = [float(h) for h in args.spacings.split(",")]
    if not cases or not spacings or min(spacings) <= 0 or args.body_spacing <= 0:
        raise ValueError("nonempty cases and positive spacings required")
    if len(set(spacings)) != len(spacings):
        raise ValueError("duplicate spacings would overwrite configuration artifacts")
    out = Path(args.output)
    # mkdir is outside the error-finalizer: a path not created by this process
    # is never modified, including concurrent creation races.
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("source", "fields", "configuration_metrics", "figures"):
        (out / folder).mkdir()
    results, manifest = [], {"status": "initializing"}
    started = time.perf_counter()
    try:
        source_names = ("geometry_motion.py", "motion_reference.py", "geometry.py", "sdf_grid.py")
        hashes = {}
        for name in source_names:
            destination = out / "source" / name
            shutil.copy2(HERE / name, destination)
            hashes[name] = sha256(destination)
            if sha256(HERE / name) != hashes[name]:
                raise RuntimeError("source changed during snapshot")
        settings = {"spacing_m": spacings, "body_lattice_max_spacing_m": args.body_spacing,
                    "max_depth": args.max_depth, "node_budget": args.node_budget,
                    "motion_tolerance_m": args.motion_tolerance,
                    "reference_max_depth": 16, "reference_node_budget": 4097,
                    "reference_motion_tolerance_m": 1e-7, "naive_time_samples": 9,
                    "roi_lower_m": [-.12] * 3, "roi_upper_m": [.12] * 3,
                    "arithmetic_guard_m": 1e-10,
                    "interpolation_error_bound": "sqrt(3)*h/2 + measured_vertex_quantization_max + 1e-10",
                    "bound_assumption": "exact 1-Lipschitz analytical scene SDF at all vertices",
                    "unknown_is": "unresolved; never counted as collision-free or as a collision",
                    "scope": "rigid prescribed geometric motion only; no robot, dynamics, contact forces, or planning",
                    "timing": "one build and one measured check per configuration; descriptive, no speed benchmark"}
        manifest = {"status": "validating_reference", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "formal_matrix": (len(cases) == 15 and spacings == [.004, .002, .001]
                                      and args.body_spacing == .002 and args.max_depth == 12
                                      and args.node_budget == 257 and args.motion_tolerance == .00001),
                    "configuration_count": len(cases) * len(spacings), "sources_sha256": hashes,
                    "model_calls": 0, "robotics": False, "settings": settings,
                    "python": sys.version, "command": sys.argv,
                    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "trimesh", "matplotlib")}}
        write_json(out / "manifest.json", manifest)
        write_json(out / "settings.json", settings)
        write_json(out / "cases.json", [case_description(case) for case in cases])
        manifest["settings_sha256"] = sha256(out / "settings.json")
        manifest["cases_sha256"] = sha256(out / "cases.json")
        references = {}
        for case in cases:
            geometry_validation = check_case_geometry(case)
            exact = ExactChecker(case["body"], case["obstacles"])
            reference_start = time.perf_counter()
            path = continuous_check(exact, case["trajectory"], case["body"], max_depth=16,
                                    node_budget=4097, motion_tolerance=1e-7)
            if path["status"] != case["expected_reference_path"]:
                raise ValueError(f"reference did not resolve declared case {case['name']}: {path['status']}")
            poses = [{"time": float(t), **exact.pose(*case["trajectory"].pose(t))} for t in np.linspace(0, 1, 9)]
            references[case["name"]] = {"geometry_validation": geometry_validation, "path": path,
                                         "poses": poses, "elapsed_s": time.perf_counter() - reference_start}
        write_json(out / "reference_checks.json", references)
        manifest["reference_checks_sha256"] = sha256(out / "reference_checks.json")
        manifest["status"] = "running"
        write_json(out / "manifest.json", manifest)
        plot_cases(out / "figures" / "case_geometry.png", cases)
        for case in cases:
            body, trajectory = case["body"], case["trajectory"]
            reference = references[case["name"]]
            for spacing in spacings:
                name = f"{case['name']}_h{spacing * 1000:g}mm"
                result = {"case": case["name"], "spacing_m": spacing}
                print(f"START {name}", flush=True)
                try:
                    t0 = time.perf_counter()
                    grid = SDFGrid.build(case["scene"], spacing)
                    build_time = time.perf_counter() - t0
                    bounded = FieldChecker(grid, body, args.body_spacing)
                    naive = FieldChecker(grid, body, args.body_spacing, mode="naive_surface")
                    t0 = time.perf_counter()
                    path = continuous_check(bounded, trajectory, body, max_depth=args.max_depth,
                                            node_budget=args.node_budget, motion_tolerance=args.motion_tolerance)
                    bounded_time = time.perf_counter() - t0
                    poses = []
                    for truth in reference["poses"]:
                        t = truth["time"]
                        pose = trajectory.pose(t)
                        poses.append({"time": t, "reference": truth, "bounded_sdf": bounded.pose(*pose),
                                      "naive_surface": naive.pose(*pose)})
                    naive_path = sampled_path([{"time": p["time"], **p["naive_surface"]} for p in poses])
                    endpoints = sampled_path([reference["poses"][0], reference["poses"][-1]])
                    field = out / "fields" / f"{name}.npz"
                    grid.save(field)
                    result.update(status="ok", reference_path=reference["path"], poses=poses,
                                  paths={"bounded_sdf": path, "naive_surface_9_poses": naive_path,
                                         "exact_endpoints_only": endpoints},
                                  interpolation_bound_m=bounded.error_bound,
                                  measured_vertex_quantization_max_m=grid.vertex_quantization_max_m,
                                  body_cover_radius_m=bounded.cover_radius,
                                  volume_sample_count=len(bounded.volume_points), surface_sample_count=len(naive.surface_points),
                                  timing={"build_s": build_time, "bounded_path_s": bounded_time},
                                  field=str(field.relative_to(out)), field_sha256=sha256(field), field_bytes=field.stat().st_size)
                    print(f"DONE {name} reference={reference['path']['status']} bounded={path['status']} naive={naive_path['status']}", flush=True)
                    del grid, bounded, naive
                except Exception:
                    result.update(status="failed", error=traceback.format_exc())
                    print(f"FAILED {name}: {result['error']}", flush=True)
                results.append(result)
                write_json(out / "configuration_metrics" / f"{name}.json", result)
                write_json(out / "metrics.json", {"status": "running", "configurations": results})
                write_csv(out / "metrics.csv", results)
        plot_results(out / "figures" / "trajectory_results.png", results)
        source_integrity = {name: sha256(HERE / name) == digest for name, digest in hashes.items()}
        status = "complete" if all(r["status"] == "ok" for r in results) else "completed_with_failures"
        if not all(source_integrity.values()):
            status = "source_integrity_failed"
        summary = {"status": status, "configurations": results, "aggregates": aggregates(results),
                   "sources_unchanged": source_integrity,
                   "elapsed_s": time.perf_counter() - started,
                   "limitations": ["Geometry known analytically; not reconstruction from images.",
                                   "All trajectories prescribed; no path search or robot kinematics.",
                                   "Whole-solid and continuous certificates conditional on declared numeric bounds.",
                                   "Naive surface and endpoint-only baselines are heuristics without certificates.",
                                   "Reference knows finite scene geometry globally; field treats outside ROI as unknown."]}
        write_json(out / "metrics.json", summary)
        manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(),
                        sources_unchanged=source_integrity, elapsed_s=summary["elapsed_s"])
        write_json(out / "manifest.json", manifest)
        print(f"FINISHED {status} {len(results)} configurations", flush=True)
        return 0 if status == "complete" else 1
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "aborted"
        detail = traceback.format_exc()
        manifest.update(status=status, terminal_error=detail, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / "manifest.json", manifest)
        write_json(out / "metrics.json", {"status": status, "configurations": results, "terminal_error": detail})
        print(detail, flush=True)
        return 130 if status == "interrupted" else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new output directory; existing paths are refused")
    p.add_argument("--spacings", default=".004,.002,.001", help="comma-separated grid spacings in metres")
    p.add_argument("--cases", help="comma-separated names for a labeled subset")
    p.add_argument("--body-spacing", type=float, default=.002, help="maximum filled body lattice spacing, metres")
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--node-budget", type=int, default=257, help="midpoint query budget, plus two endpoints")
    p.add_argument("--motion-tolerance", type=float, default=.00001, help="unresolved interval displacement cutoff, metres")
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
