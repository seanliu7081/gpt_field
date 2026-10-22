"""Known LIBERO collision geometry -> local scene SDF -> reconstructed surface.

All bounded scene geometry remains in the reference union. Only the field and
evaluation region are local. These are static collision-geometry measurements,
not reconstruction from RGB and not robot control.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(HERE / "_deps"), str(PARENT / "_deps"), str(PARENT), str(HERE)]
import numpy as np
import trimesh
from scipy.optimize import brentq
from sdf_grid import SDFGrid
from mesh_queries import MeshDistance
from reconstruct import (write_json, sha256, distance_statistics, occupancy_statistics,
                         sample_mesh, RSSSampler, jsonable)

ROI_PAD = .040
SCORE_GUARD = .012
NEAR_BAND = .005
TOLERANCES = (.001, .002, .005)


def box_bounds(box):
    center = np.asarray(box["center_m"], float)
    extent = np.abs(np.asarray(box["rotation_local_to_world"])) @ np.asarray(box["half_extents_m"])
    return center - extent, center + extent


def object_bounds(snapshot, names):
    boxes = [box for box in snapshot["boxes"] if box.get("object") in names]
    if not boxes:
        raise ValueError(f"snapshot {snapshot['name']} lacks required objects {names}")
    bounds = np.array([box_bounds(b) for b in boxes])
    return bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)


def choose_rois(snapshots):
    """ROI choice depends only on declared geometry, before examining errors."""
    drawers = [s for s in snapshots if s["name"].startswith("drawer")]
    common = None
    if drawers:
        bounds = np.array([object_bounds(s, {"wooden_cabinet_1"}) for s in drawers])
        common = (bounds[:, 0].min(0), bounds[:, 1].max(0))
    result = {}
    for snapshot in snapshots:
        if snapshot["name"].startswith("drawer"):
            bounds = common
            objects = ["wooden_cabinet_1"]
            purpose = "same cabinet-and-open-drawer ROI for both drawer snapshots"
        else:
            objects = ["akita_black_bowl_1", "plate_1"]
            bounds = object_bounds(snapshot, set(objects))
            purpose = "complete target bowl and plate plus40mm motion context"
        lower = np.floor((bounds[0] - ROI_PAD) / .004) * .004
        upper = np.ceil((bounds[1] + ROI_PAD) / .004) * .004
        shape = np.rint((upper - lower) / .001).astype(int) + 1
        if np.prod(shape) > 50_000_000:
            raise ValueError(f"declared 1mm ROI exceeds50M vertices: {snapshot['name']}: {shape}")
        result[snapshot["name"]] = {"lower_m": lower, "upper_m": upper,
            "score_lower_m": lower + SCORE_GUARD, "score_upper_m": upper - SCORE_GUARD,
            "padding_m": ROI_PAD, "score_guard_m": SCORE_GUARD,
            "primary_objects": objects, "purpose": purpose,
            "grid_shape_at_1mm": shape, "vertex_count_at_1mm": int(np.prod(shape)),
            "scope": "local task ROI; all scene boxes remain in global reference"}
    return result


def _clip_polygon_axis(polygon, axis, bound, keep_greater):
    if len(polygon) == 0:
        return polygon
    result = []
    for previous, current in zip(np.roll(polygon, 1, axis=0), polygon):
        previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
        current_inside = current[axis] >= bound if keep_greater else current[axis] <= bound
        if previous_inside != current_inside:
            ratio = (bound - previous[axis]) / (current[axis] - previous[axis])
            point = previous + ratio * (current - previous)
            point[axis] = bound
            result.append(point)
        if current_inside:
            result.append(current)
    return np.asarray(result, dtype=float).reshape(-1, 3)


def clip_surface_to_roi(mesh, lower, upper):
    """Clip existing triangles to a box WITHOUT adding artificial cap faces."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    tri = np.asarray(mesh.triangles)
    inside_vertices = np.all((tri >= lower) & (tri <= upper), axis=2)
    whole = inside_vertices.all(axis=1)
    possible = np.all((tri.max(axis=1) >= lower) & (tri.min(axis=1) <= upper), axis=1)
    pieces = [tri[whole]]
    for triangle in tri[possible & ~whole]:
        polygon = triangle
        for axis in range(3):
            polygon = _clip_polygon_axis(polygon, axis, lower[axis], True)
            polygon = _clip_polygon_axis(polygon, axis, upper[axis], False)
        if len(polygon) >= 3:
            pieces.append(np.array([[polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)]))
    triangles = np.concatenate(pieces)
    if not len(triangles):
        return None
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    triangles = triangles[areas2 > 1e-20]
    if not len(triangles):
        return None
    return trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(len(triangles) * 3).reshape(-1, 3), process=False)


def unsigned_mesh_distance(mesh, points, return_projection=False):
    """PCU unsigned only: its signed API can change distance magnitudes."""
    import point_cloud_utils as pcu
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    if count == 0:
        return (np.empty(0), np.empty((0, 3))) if return_projection else np.empty(0)
    # PCU0.34 singleton queries can return incorrect distances, not just a
    # squeezed shape; use its independently verified batched path.
    query = np.repeat(points, 2, axis=0) if count == 1 else points
    distance, faces, barycentric = pcu.closest_points_on_mesh(
        query, np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64))
    distance = np.asarray(distance, dtype=float).reshape(-1)[:count]
    faces = np.asarray(faces).reshape(-1)[:count]
    barycentric = np.asarray(barycentric).reshape(-1, 3)[:count]
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("invalid unsigned mesh distance")
    if return_projection:
        projection = np.einsum("ni,nij->nj", barycentric, np.asarray(mesh.triangles)[faces])
        return distance, projection
    return distance


def validate_mesh_distance(mesh, points):
    sample = np.asarray(points)[:16]
    actual = unsigned_mesh_distance(mesh, sample)
    independent = MeshDistance(mesh)(sample)
    error = float(np.max(np.abs(actual - independent)))
    if error > 2e-7:
        raise ValueError(f"PCU unsigned versus normalized-plane-edge query differs by {error}m")
    return {"passed": True, "samples": len(sample), "max_abs_difference_m": error,
            "tolerance_m": 2e-7, "reference": "independent_rtree_normalized_plane_edge_projection"}


def build_reference_samples(scene, roi, count, surface_count, seed):
    lower, upper = np.asarray(roi["score_lower_m"]), np.asarray(roi["score_upper_m"])
    mesh = scene.mesh()
    clipped = clip_surface_to_roi(mesh, lower, upper)
    if clipped is None:
        raise ValueError("ROI contains no exposed reference surface")
    rng = np.random.default_rng(seed)
    uniform = rng.uniform(lower, upper, (count, 3))
    surface = sample_mesh(clipped, surface_count, rng)
    near, have, proposed = [], 0, 0
    while have < count:
        n = min(max(count - have, 1000) * 2, 200_000)
        base = sample_mesh(clipped, n, rng)
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        points = base + direction * (NEAR_BAND * rng.random(n) ** (1 / 3))[:, None]
        distance = scene.sdf(points)
        keep = np.all((points >= lower) & (points <= upper), axis=1) & (np.abs(distance) <= NEAR_BAND)
        near.append(points[keep])
        have += int(keep.sum())
        proposed += n
    near = np.concatenate(near)[:count]
    labels = np.asarray(scene.surface_object_labels(surface)).astype(str)
    data = {"uniform_points_m": uniform, "uniform_reference_m": scene.sdf(uniform),
            "near_points_m": near, "near_reference_m": scene.sdf(near),
            "reference_surface_points_m": surface, "reference_surface_object": labels}
    return data, {"sampling": "exact_area_weighted_clipped_exposed_union_triangles; clipping adds no cap faces",
                  "score_surface_area_m2": float(clipped.area), "surface_object_counts": dict(zip(*np.unique(labels, return_counts=True))),
                  "near_sampling": "surface_area_then_uniform_3D_ball_radius5mm_filtered_by_exact_SDF_and_scoreROI",
                  "near_proposed": proposed, "near_distance_quantiles_m": np.quantile(data["near_reference_m"], [0, .01, .1, .5, .9, .99, 1])}


def line_definitions(snapshot, roi):
    lower, upper = np.asarray(roi["score_lower_m"]), np.asarray(roi["score_upper_m"])
    if snapshot["name"].startswith("drawer"):
        boxes = [b for b in snapshot["boxes"] if b.get("body", "").endswith("_cabinet_top")]
        if not boxes:
            return []
        bounds = np.array([box_bounds(b) for b in boxes])
        center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) / 2
        horizontal = [center.copy(), center.copy()]
        horizontal[0][1], horizontal[1][1] = lower[1], upper[1]
        vertical = [center.copy(), center.copy()]
        vertical[0][2], vertical[1][2] = lower[2], upper[2]
        return [{"name": "top_drawer_front_back_section", "start_m": horizontal[0], "end_m": horizontal[1]},
                {"name": "top_drawer_vertical_cavity_section", "start_m": vertical[0], "end_m": vertical[1]}]
    bounds = object_bounds(snapshot, {"akita_black_bowl_1"})
    center = (bounds[0] + bounds[1]) / 2
    center[2] = bounds[0][2] + .65 * (bounds[1][2] - bounds[0][2])
    a, b = center.copy(), center.copy()
    a[0], b[0] = bounds[0][0] - .010, bounds[1][0] + .010
    return [{"name": "bowl_wall_cavity_wall_section", "start_m": a, "end_m": b}]


def line_profile(query, line):
    start, end = np.asarray(line["start_m"]), np.asarray(line["end_m"])
    length = float(np.linalg.norm(end - start))
    t = np.linspace(0, 1, max(4097, int(np.ceil(length / .00005)) + 1))
    points = start + t[:, None] * (end - start)
    values = np.asarray(query(points))
    if not np.all(np.isfinite(values)):
        raise ValueError("feature line left valid query region")
    roots = list(t[values == 0])
    for i in np.flatnonzero(values[:-1] * values[1:] < 0):
        roots.append(brentq(lambda u: float(query((start + u * (end - start))[None])[0]), t[i], t[i + 1], xtol=1e-12))
    roots = np.unique(np.round(np.array(roots) * length, 12))
    return {"positions_m": roots, "interval_widths_m": np.diff(roots),
            "sample_count": len(t), "minimum_sdf_m": float(values.min()), "maximum_sdf_m": float(values.max())}


def feature_metrics(scene, grid, lines):
    rows = []
    for line in lines:
        truth, predicted = line_profile(scene.sdf, line), line_profile(grid.query, line)
        same = len(truth["positions_m"]) == len(predicted["positions_m"])
        rows.append({**line, "reference": truth, "prediction": predicted, "crossing_count_matches": same,
                     "max_crossing_error_m": float(np.max(np.abs(truth["positions_m"] - predicted["positions_m"]))) if same and len(truth["positions_m"]) else None,
                     "interval_width_errors_m": predicted["interval_widths_m"] - truth["interval_widths_m"] if same else None,
                     "limitation": "fixed line diagnosis, not a global topology certificate"})
    return rows


def distance_summary(values):
    if len(values) == 0:
        return None
    return {"count": len(values), "mean_mm": float(np.mean(values) * 1000),
            "p95_mm": float(np.quantile(values, .95) * 1000), "sampled_max_mm": float(np.max(values) * 1000)}


def surface_scores(pred_to_ref, ref_to_pred):
    metrics = {"prediction_to_reference": distance_summary(pred_to_ref), "reference_to_prediction": distance_summary(ref_to_pred),
               "symmetric_mean_mm": float((np.mean(pred_to_ref) + np.mean(ref_to_pred)) * 500) if len(pred_to_ref) and len(ref_to_pred) else None,
               "precision": {}, "recall": {}, "f1": {}}
    for tolerance in TOLERANCES:
        key = str(tolerance)
        precision = float(np.mean(pred_to_ref <= tolerance)) if len(pred_to_ref) else None
        recall = float(np.mean(ref_to_pred <= tolerance)) if len(ref_to_pred) else None
        metrics["precision"][key], metrics["recall"][key] = precision, recall
        metrics["f1"][key] = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0. if recall is not None else None
    return metrics


def evaluate_surface(scene, mesh, roi, reference, surface_count, seed):
    if mesh is None:
        empty = {"symmetric_mean_mm": None,
            "precision": {str(t): None for t in TOLERANCES}, "recall": {str(t): 0. for t in TOLERANCES},
            "f1": {str(t): 0. for t in TOLERANCES}}
        per_object = {}
        for name in roi["primary_objects"]:
            count = int(np.sum(reference["reference_surface_object"] == name))
            per_object[name] = {**empty, "reference_sample_count": count,
                "status": "empty_reconstruction" if count else "reference_object_unsampled"}
            if not count:
                per_object[name] = {**per_object[name], "recall": {str(t): None for t in TOLERANCES},
                                    "f1": {str(t): None for t in TOLERANCES}}
        return {"status": "empty_reconstruction", "global": empty, "per_object": per_object}, None
    clipped = clip_surface_to_roi(mesh, roi["score_lower_m"], roi["score_upper_m"])
    if clipped is None:
        return evaluate_surface(scene, None, roi, reference, surface_count, seed)
    predicted = sample_mesh(clipped, surface_count, np.random.default_rng(seed))
    pred_to_ref, projected = unsigned_mesh_distance(scene.mesh(), predicted, return_projection=True)
    ref_to_pred = unsigned_mesh_distance(mesh, reference["reference_surface_points_m"])
    validation = validate_mesh_distance(mesh, reference["reference_surface_points_m"])
    labels = np.asarray(scene.surface_object_labels(projected)).astype(str)
    per_object = {}
    for name in roi["primary_objects"]:
        a = pred_to_ref[labels == name]
        b = ref_to_pred[reference["reference_surface_object"] == name]
        per_object[name] = surface_scores(a, b)
        per_object[name]["prediction_label_rule"] = "object owning the nearest point on the exposed reference union surface"
    return {"status": "ok", "global": surface_scores(pred_to_ref, ref_to_pred), "per_object": per_object,
            "distance_validation": validation, "continuous_reference": "exposed global Boolean union, not overlapping primitive surfaces",
            "distance_backend": "PCU unsigned closest_points_on_mesh; checked against independent normalized plane-edge kernel",
            "predicted_vertices": len(mesh.vertices), "predicted_faces": len(mesh.faces),
            "predicted_roi_mesh_components": int(mesh.body_count),
            "component_warning": "ROI boundary truncates context geometry; count is diagnostic, not whole-scene topology"}, (predicted, pred_to_ref)


def plot_context(path, scene, roi, snapshot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    points = scene.sample_surface(35_000, np.random.default_rng(908))
    lower, upper = np.asarray(roi["lower_m"]), np.asarray(roi["upper_m"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (i, j), title in zip(axes, [(0, 1), (0, 2), (1, 2)], ["XY", "XZ", "YZ"]):
        ax.scatter(points[:, i] * 1000, points[:, j] * 1000, s=.12, color="gray", alpha=.5)
        ax.add_patch(Rectangle(lower[[i, j]] * 1000, (upper[i] - lower[i]) * 1000,
                               (upper[j] - lower[j]) * 1000, fill=False, color="crimson", linewidth=2, label="fine-grid ROI"))
        ax.set(title=title, xlabel=f"{'XYZ'[i]} (mm)", ylabel=f"{'XYZ'[j]} (mm)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    fig.suptitle(f"{snapshot['name']}: all bounded nonrobot collision geometry used; red box is the evaluated field region")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_result(path, scene, grid, roi, snapshot, surface_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = line_definitions(snapshot, roi)
    z = float(np.asarray(lines[0]["start_m"])[2]) if lines else float(np.mean([grid.lower[2], grid.upper[2]]))
    x = np.linspace(grid.lower[0], grid.upper[0], 350)
    y = np.linspace(grid.lower[1], grid.upper[1], 350)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    truth, values = scene.sdf(points).reshape(xx.shape), grid.query(points).reshape(xx.shape)
    extent = [grid.lower[0] * 1000, grid.upper[0] * 1000, grid.lower[1] * 1000, grid.upper[1] * 1000]
    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(131)
    image = ax.imshow(values * 1000, origin="lower", extent=extent, cmap="coolwarm", vmin=-15, vmax=15)
    if truth.min() < 0 < truth.max():
        ax.contour(xx * 1000, yy * 1000, truth, levels=[0], colors="black", linewidths=.8)
    if values.min() < 0 < values.max():
        ax.contour(xx * 1000, yy * 1000, values, levels=[0], colors="lime", linewidths=.6)
    ax.set(title=f"SDF at Z={z * 1000:.1f}mm\nblack=reference, green=reconstructed", xlabel="X (mm)", ylabel="Y (mm)")
    fig.colorbar(image, ax=ax, label="Signed surface distance (mm)")
    ax = fig.add_subplot(132)
    image = ax.imshow(np.abs(values - truth) * 1000, origin="lower", extent=extent, cmap="magma", vmin=0)
    ax.set(title="Absolute interpolation error, same slice", xlabel="X (mm)", ylabel="Y (mm)")
    fig.colorbar(image, ax=ax, label="Error (mm)")
    ax = fig.add_subplot(133, projection="3d")
    if surface_data is not None:
        points, errors = surface_data
        stride = max(1, len(points) // 15_000)
        image = ax.scatter(*(points[::stride] * 1000).T, c=errors[::stride] * 1000, s=.5, cmap="viridis", vmin=0)
        fig.colorbar(image, ax=ax, label="Surface error (mm)", shrink=.7)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title="Reconstructed exposed surfaces")
    ax.set_box_aspect(grid.upper - grid.lower)
    fig.suptitle(f"{snapshot['name']}, {grid.spacing * 1000:g}mm grid — collision geometry, not RGB reconstruction")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path, results):
    names = ["snapshot", "spacing_mm", "status", "near_p95_mm", "uniform_p95_mm", "near_iou",
             "surface_mean_mm", "surface_f1_1mm", "surface_f1_2mm", "surface_f1_5mm",
             "grid_vertices", "build_s", "peak_rss_bytes", "error"]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for result in results:
            row = {"snapshot": result["snapshot"], "spacing_mm": result["spacing_m"] * 1000,
                   "status": result["status"], "error": result.get("error", "")}
            if "near_distance" in result:
                row["near_p95_mm"] = result["near_distance"]["p95_mm"]
            if "uniform_distance" in result:
                row["uniform_p95_mm"] = result["uniform_distance"]["p95_mm"]
            if "near_occupancy" in result:
                row["near_iou"] = result["near_occupancy"]["iou"]
            if "surface" in result:
                row["surface_mean_mm"] = result["surface"]["global"]["symmetric_mean_mm"]
                for mm in (1, 2, 5):
                    row[f"surface_f1_{mm}mm"] = result["surface"]["global"]["f1"][str(mm / 1000)]
            if "performance" in result:
                row.update(grid_vertices=result["performance"]["grid_vertices"], build_s=result["performance"]["build_s"], peak_rss_bytes=result["performance"]["observed_peak_rss_bytes"])
            writer.writerow(row)


def run(args):
    from union_geometry import UnionScene
    source_input = Path(args.captures).resolve()
    captures = json.loads(source_input.read_text())
    if captures.get("status") != "complete" or captures.get("units") != "m":
        raise ValueError("capture must be complete and explicitly use metres")
    snapshots = captures["snapshots"]
    rois = choose_rois(snapshots)  # Common drawer ROI uses both snapshots even in smoke subsets.
    if args.snapshots:
        requested = set(args.snapshots.split(","))
        snapshots = [snapshot for snapshot in snapshots if snapshot["name"] in requested]
        if len(snapshots) != len(requested):
            raise ValueError("unknown snapshot selection")
    spacings = [float(x) for x in args.spacings.split(",")]
    if not snapshots or len(set(spacings)) != len(spacings) or min(spacings) <= 0 or min(args.samples, args.surface_samples) <= 0:
        raise ValueError("invalid nonempty unique configuration matrix")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("source", "reference", "fields", "meshes", "figures", "configuration_metrics"):
        (output / name).mkdir()
    manifest, results = {"status": "initializing"}, []
    try:
        files = [HERE / "reconstruct_libero.py", HERE / "union_geometry.py", PARENT / "sdf_grid.py",
                 PARENT / "reconstruct.py", PARENT / "mesh_queries.py"]
        hashes = {}
        for file in files:
            destination = output / "source" / file.name
            shutil.copy2(file, destination)
            hashes[str(file)] = sha256(destination)
            if sha256(file) != hashes[str(file)]:
                raise RuntimeError("source changed during freezing")
        shutil.copy2(source_input, output / "reference" / "snapshots.json")
        input_hash = sha256(output / "reference" / "snapshots.json")
        if sha256(source_input) != input_hash:
            raise RuntimeError("snapshot input changed during freezing")
        manifest = {"status": "validating_reference", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration_count": len(snapshots) * len(spacings),
                    "formal_matrix": len(snapshots) == 3 and spacings == [.004, .002, .001] and args.samples == args.surface_samples == 100_000,
                    "sources_sha256": hashes, "capture_path": str(source_input), "capture_sha256": input_hash,
                    "model_calls": 0, "robot_control": False, "scope": "known simulated collision geometry representation, not RGB reconstruction",
                    "roi": rois, "spacing_m": spacings, "uniform_points": args.samples, "near_points": args.samples,
                    "surface_samples_each_direction": args.surface_samples, "near_band_m": NEAR_BAND,
                    "timing": "one build/query/extraction per configuration; descriptive, not a speed benchmark",
                    "sign_boundary_exclusion_m": .00001, "unexcluded_sign_metrics_also_reported": True,
                    "surface_tolerances_m": TOLERANCES, "surface_reference": "continuous exposed Boolean-union boundary, all bounded nonrobot collision geometry",
                    "sampling": "score ROI eroded12mm identically for all spacings; samples fixed before field construction",
                    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "trimesh", "scikit-image", "manifold3d", "point-cloud-utils", "numba")},
                    "python": sys.version, "command": sys.argv}
        write_json(output / "manifest.json", manifest)
        settings = {key: manifest[key] for key in ("roi", "spacing_m", "uniform_points", "near_points", "surface_samples_each_direction", "near_band_m", "sign_boundary_exclusion_m", "surface_tolerances_m")}
        write_json(output / "settings.json", settings)
        manifest["settings_sha256"] = sha256(output / "settings.json")
        prepared, reference_hashes = [], {}
        for index, snapshot in enumerate(snapshots):
            print(f"REFERENCE {snapshot['name']}", flush=True)
            scene = UnionScene(snapshot["name"], snapshot["boxes"], snapshot.get("planes", []))
            roi = rois[snapshot["name"]]
            validation = scene.validate()
            if not validation.get("passed"):
                raise ValueError(f"invalid reference union: {snapshot['name']}")
            plane_certificate = scene.prove_planes_irrelevant(roi["lower_m"], roi["upper_m"])
            if not plane_certificate.get("passed"):
                raise ValueError("unbounded plane may affect ROI; finite mesh sampling would be invalid")
            reference, sampling = build_reference_samples(scene, roi, args.samples, args.surface_samples, 900_000 + index)
            path = output / "reference" / f"{snapshot['name']}_points.npz"
            np.savez_compressed(path, **reference)
            scene.mesh().export(output / "reference" / f"{snapshot['name']}_global_union.ply")
            record = {"geometry": scene.to_dict(), "validation": validation, "plane_irrelevance": plane_certificate,
                      "sampling": sampling, "sample_points_sha256": sha256(path),
                      "feature_lines": line_definitions(snapshot, roi)}
            write_json(output / "reference" / f"{snapshot['name']}_manifest.json", record)
            for artifact in (path, output / "reference" / f"{snapshot['name']}_global_union.ply",
                             output / "reference" / f"{snapshot['name']}_manifest.json"):
                reference_hashes[str(artifact.relative_to(output))] = sha256(artifact)
            plot_context(output / "figures" / f"{snapshot['name']}_context.png", scene, roi, snapshot)
            prepared.append((snapshot, scene, roi, reference, record))
        manifest["status"] = "running"
        manifest["reference_artifacts_sha256"] = reference_hashes
        write_json(output / "manifest.json", manifest)
        for index, (snapshot, scene, roi, reference, record) in enumerate(prepared):
            for h in spacings:
                name = f"{snapshot['name']}_h{h * 1000:g}mm"
                result = {"snapshot": snapshot["name"], "spacing_m": h}
                print(f"START {name}", flush=True)
                try:
                    with RSSSampler() as rss:
                        t0 = time.perf_counter()
                        grid = SDFGrid.build(scene, h, lower=roi["lower_m"], upper=roi["upper_m"], chunk_size=250_000)
                        build_time = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        for label in ("uniform", "near"):
                            prediction = grid.query(reference[f"{label}_points_m"])
                            truth = reference[f"{label}_reference_m"]
                            result[f"{label}_distance"] = distance_statistics(prediction, truth, reference[f"{label}_points_m"])
                            result[f"{label}_occupancy"] = occupancy_statistics(prediction, truth, .00001)
                            result[f"{label}_occupancy_unexcluded"] = occupancy_statistics(prediction, truth, None)
                        query_time = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        mesh = grid.surface()
                        extraction_time = time.perf_counter() - t0
                        result["surface"], display = evaluate_surface(scene, mesh, roi, reference, args.surface_samples, 910_000 + index)
                        result["features"] = feature_metrics(scene, grid, record["feature_lines"])
                    result["status"] = result["surface"]["status"]
                    field = output / "fields" / f"{name}.npz"
                    grid.save(field)
                    if mesh is not None:
                        mesh.export(output / "meshes" / f"{name}.ply")
                    result.update(field=str(field.relative_to(output)), field_sha256=sha256(field), field_bytes=field.stat().st_size,
                                  vertex_quantization_max_m=grid.vertex_quantization_max_m,
                                  performance={"build_s": build_time, "query_s": query_time, "extraction_s": extraction_time,
                                               "grid_vertices": grid.values.size, "grid_value_bytes": grid.values.nbytes,
                                               "observed_peak_rss_bytes": rss.peak, "memory_method": "process RSS sampled10ms"})
                    plot_result(output / "figures" / f"{name}.png", scene, grid, roi, snapshot, display)
                    print(f"DONE {name} nearP95={result['near_distance']['p95_mm']:.6f}mm F1@2={result['surface']['global']['f1']['0.002']}", flush=True)
                    del grid, mesh, display
                except Exception:
                    result.update(status="failed", error=traceback.format_exc())
                    print(f"FAILED {name}: {result['error']}", flush=True)
                results.append(result)
                write_json(output / "configuration_metrics" / f"{name}.json", result)
                write_json(output / "metrics.json", {"status": "running", "configurations": results})
                write_csv(output / "metrics.csv", results)
        integrity = {file: Path(file).is_file() and sha256(file) == digest for file, digest in hashes.items()}
        frozen_integrity = {file: sha256(output / "source" / Path(file).name) == digest for file, digest in hashes.items()}
        artifact_integrity = {file: sha256(output / file) == digest for file, digest in reference_hashes.items()}
        artifact_integrity["settings.json"] = sha256(output / "settings.json") == manifest["settings_sha256"]
        artifact_integrity["reference/snapshots.json"] = sha256(output / "reference" / "snapshots.json") == input_hash
        artifact_integrity.update({r["field"]: sha256(output / r["field"]) == r["field_sha256"] for r in results if "field" in r})
        input_unchanged = source_input.is_file() and sha256(source_input) == input_hash
        status = "complete" if all(r["status"] in ("ok", "empty_reconstruction") for r in results) else "completed_with_failures"
        if not all(integrity.values()) or not all(frozen_integrity.values()) or not all(artifact_integrity.values()) or not input_unchanged:
            status = "integrity_failed"
        summary = {"status": status, "configurations": results, "sources_unchanged": integrity,
                   "frozen_sources_unchanged": frozen_integrity, "artifacts_unchanged": artifact_integrity,
                   "capture_unchanged": input_unchanged, "limitations": [
                       "Accuracy measured only within declared task ROIs, not entire scene workspace.",
                       "Original RGBs provide context; no geometry is inferred from images.",
                       "Geometry is simulator collision geometry, which differs from visual meshes and physical objects.",
                       "Surface samples are area weighted; per-object counts are retained and undefined metrics remain null.",
                       "Feature lines and sampled maxima are diagnostics, not global topology/error certificates."]}
        write_json(output / "metrics.json", summary)
        manifest.update(status=status, sources_unchanged=integrity, capture_unchanged=input_unchanged,
                        frozen_sources_unchanged=frozen_integrity, artifacts_unchanged=artifact_integrity,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(f"FINISHED {status} {len(results)} configurations", flush=True)
        return 0 if status == "complete" else 1
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "aborted"
        manifest.update(status=status, terminal_error=traceback.format_exc(), finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        write_json(output / "metrics.json", {"status": status, "configurations": results, "terminal_error": manifest["terminal_error"]})
        print(manifest["terminal_error"], flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", required=True, help="immutable capture snapshots.json")
    parser.add_argument("--output", required=True, help="new directory only")
    parser.add_argument("--snapshots", help="comma-separated subset, explicitly nonformal")
    parser.add_argument("--spacings", default=".004,.002,.001")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--surface-samples", type=int, default=100_000)
    sys.exit(run(parser.parse_args()))
