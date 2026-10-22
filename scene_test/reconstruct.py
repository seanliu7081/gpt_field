"""Known geometry -> scene SDF -> continuous zero surface benchmark.

This standalone experiment imports no robot controller. Existing runs are never
overwritten. Formal runs use the counts in the preserved draft protocol; smoke
runs are explicitly labelled and cannot satisfy the formal test matrix.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_deps"))
import numpy as np
import psutil
import trimesh
from sdf_grid import SDFGrid
from mesh_queries import MeshDistance


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, data):
    path = Path(path)
    serialized = json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RSSSampler:
    """Observed process peak; not an allocator-exact or systemwide peak."""
    def __init__(self):
        self.stop = threading.Event()
        self.baseline = psutil.Process().memory_info().rss
        self.peak = self.baseline
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        process = psutil.Process()
        while not self.stop.wait(.01):
            self.peak = max(self.peak, process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.stop.set()
        self.thread.join()


def timed(function, repeats):
    """A full warmup then measured repeats; return the last computed result."""
    result = function()
    del result
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, samples


def distance_statistics(predicted, truth, points):
    errors = np.abs(np.asarray(predicted) - truth)
    if not np.all(np.isfinite(errors)):
        raise ValueError("evaluation query produced non-finite distances")
    index = int(np.argmax(errors))
    return {"count": len(errors), "mae_mm": float(errors.mean() * 1000),
            "rmse_mm": float(np.sqrt(np.mean(errors ** 2)) * 1000),
            "p95_mm": float(np.quantile(errors, .95) * 1000),
            "sampled_max_mm": float(errors[index] * 1000),
            "sampled_max_position_m": points[index].tolist(),
            "sampled_max_truth_m": float(truth[index]),
            "sampled_max_prediction_m": float(predicted[index])}


def occupancy_statistics(predicted, truth, exclude):
    mask = np.ones(len(truth), dtype=bool) if exclude is None else np.abs(truth) > exclude
    p, t = np.asarray(predicted)[mask] < 0, np.asarray(truth)[mask] < 0
    tp, fp, fn, tn = [int(x.sum()) for x in (p & t, p & ~t, ~p & t, ~p & ~t)]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "excluded_boundary_count": int((~mask).sum()), "boundary_exclusion_m": exclude,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def sample_mesh(mesh, count, rng):
    tri = mesh.triangles
    face = rng.choice(len(tri), count, p=mesh.area_faces / mesh.area)
    u, v = rng.random((2, count))
    s = np.sqrt(u)
    return (1 - s)[:, None] * tri[face, 0] + (s * (1 - v))[:, None] * tri[face, 1] + (s * v)[:, None] * tri[face, 2]


def surface_statistics(mesh, scene, reference_points, count, seed, tolerances):
    if mesh is None or len(mesh.faces) == 0:
        return {"status": "empty_reconstruction", "symmetric_mean_mm": None,
                "precision": {str(t): None for t in tolerances},
                "recall": {str(t): 0. for t in tolerances},
                "f1": {str(t): 0. for t in tolerances}}, None
    predicted_points = sample_mesh(mesh, count, np.random.default_rng(seed))
    pred_to_ref = np.abs(scene.sdf(predicted_points))
    distance = MeshDistance(mesh)
    validation = distance.validate(reference_points)
    ref_to_pred = distance(reference_points)
    precision = {str(t): float(np.mean(pred_to_ref <= t)) for t in tolerances}
    recall = {str(t): float(np.mean(ref_to_pred <= t)) for t in tolerances}
    f1 = {str(t): (2 * precision[str(t)] * recall[str(t)] / (precision[str(t)] + recall[str(t)])
                   if precision[str(t)] + recall[str(t)] else 0.) for t in tolerances}
    def describe(a):
        return {"mean_mm": float(a.mean() * 1000), "p95_mm": float(np.quantile(a, .95) * 1000),
                "sampled_max_mm": float(a.max() * 1000)}
    metrics = {"status": "ok", "samples_per_direction": count,
               "predicted_to_reference": describe(pred_to_ref),
               "reference_to_predicted": describe(ref_to_pred),
               "symmetric_mean_mm": float((pred_to_ref.mean() + ref_to_pred.mean()) * 500),
               "precision": precision, "recall": recall, "f1": f1,
               "reference_distance": "continuous_declared_geometry_via_abs_exact_sdf",
               "reconstruction_distance": distance.backend, "mesh_distance_validation": validation,
               "vertices": len(mesh.vertices), "faces": len(mesh.faces),
               "is_watertight": bool(mesh.is_watertight), "connected_components": int(mesh.body_count)}
    return metrics, (predicted_points, pred_to_ref)


def reference_samples(scene, count, surface_count, band):
    rng = np.random.default_rng(scene.seed + 210_019)
    uniform = rng.uniform(-.12, .12, (count, 3))
    surface_points = scene.sample_surface(surface_count, rng)
    near = []
    accepted = 0
    proposed = 0
    while accepted < count:
        n = min(max(count - accepted, 1000) * 2, 200_000)
        base = scene.sample_surface(n, rng)
        # Isotropic uniform-in-ball perturbation, filtered by true surface band.
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        p = base + direction * (band * rng.random(n) ** (1 / 3))[:, None]
        d = scene.sdf(p)
        keep = (np.abs(d) <= band) & np.all((p >= -.12) & (p <= .12), axis=1)
        near.append(p[keep])
        accepted += int(keep.sum())
        proposed += n
    near = np.concatenate(near)[:count]
    return {"uniform_points_m": uniform, "uniform_reference_m": scene.sdf(uniform),
            "near_points_m": near, "near_reference_m": scene.sdf(near),
            "reference_surface_points_m": surface_points}, {
                "method": "area_surface_sample_plus_uniform_3D_ball_perturbation_then_true_band_filter",
                "band_m": band, "proposed": proposed,
                "near_signed_distance_quantiles_m": np.quantile(scene.sdf(near), [0, .01, .1, .5, .9, .99, 1])}


def plot_config(out, scene, grid, surface_data, reference_points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    z = float(np.median(reference_points[:, 2]))
    a = np.linspace(-.12, .12, 241)
    x, y = np.meshgrid(a, a, indexing="xy")
    p = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])
    d, truth = grid.query(p).reshape(x.shape), scene.sdf(p).reshape(x.shape)
    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_subplot(131)
    im = ax.imshow(d * 1000, origin="lower", extent=(-120, 120, -120, 120), cmap="coolwarm", vmin=-20, vmax=20)
    if truth.min() < 0 < truth.max():
        ax.contour(x * 1000, y * 1000, truth, levels=[0], colors="black", linewidths=1)
    if d.min() < 0 < d.max():
        ax.contour(x * 1000, y * 1000, d, levels=[0], colors="lime", linewidths=.7)
    ax.set(title=f"SDF, z={z * 1000:.1f} mm\nblack: truth; green: reconstructed zero", xlabel="X (mm)", ylabel="Y (mm)")
    fig.colorbar(im, ax=ax, label="Signed surface distance (mm)", shrink=.8)
    ax = fig.add_subplot(132)
    im = ax.imshow(np.abs(d - truth) * 1000, origin="lower", extent=(-120, 120, -120, 120), cmap="magma", vmin=0)
    ax.set(title="Absolute distance error, same slice", xlabel="X (mm)", ylabel="Y (mm)")
    fig.colorbar(im, ax=ax, label="Absolute error (mm)", shrink=.8)
    ax = fig.add_subplot(133, projection="3d")
    stride = max(1, len(reference_points) // 2000)
    r = reference_points[::stride]
    ax.scatter(*(r * 1000).T, color="gray", s=.3, alpha=.2, label="reference surface")
    if surface_data is not None:
        points, error = surface_data
        stride = max(1, len(points) // 6000)
        im = ax.scatter(*(points[::stride] * 1000).T, c=error[::stride] * 1000, cmap="viridis", s=1, vmin=0)
        fig.colorbar(im, ax=ax, label="Reconstructed surface error (mm)", shrink=.7)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title="Continuous surfaces (sampled for display)")
    ax.set_box_aspect([1, 1, 1])
    fig.suptitle(f"{scene.name}: grid spacing {grid.spacing * 1000:g} mm", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def aggregate_metrics(results):
    successful = [r for r in results if r.get("status") in ("ok", "empty_reconstruction")]
    groups = {}
    for h in sorted({r["spacing_m"] for r in results}, reverse=True):
        for category in ["all"] + sorted({r["category"] for r in results}):
            group = [r for r in successful if r["spacing_m"] == h and (category == "all" or r["category"] == category)]
            expected = [r for r in results if r["spacing_m"] == h and (category == "all" or r["category"] == category)]
            metrics = {}
            values = {
                "near_p95_mm": [r["near_distance"]["p95_mm"] for r in group],
                "uniform_mae_mm": [r["uniform_distance"]["mae_mm"] for r in group],
                "near_occupancy_iou": [r["near_occupancy"]["iou"] for r in group],
                "surface_f1_at_2mm": [r["surface"]["f1"]["0.002"] for r in group],
                "symmetric_surface_mean_mm": [r["surface"]["symmetric_mean_mm"] for r in group]}
            for name, vals in values.items():
                valid = [v for v in vals if v is not None]
                metrics[name] = {"equal_scene_mean": float(np.mean(valid)) if valid else None,
                                 "min": min(valid) if valid else None, "max": max(valid) if valid else None,
                                 "defined_count": len(valid), "undefined_count": len(vals) - len(valid)}
            groups[f"{category}@{h:g}"] = {"category": category, "spacing_m": h,
                "successful_configs": len(group), "attempted_configs": len(expected),
                "failed_configs": len(expected) - len(group), "metrics": metrics}
    return groups


def final_run_status(results, source_integrity):
    if not all(source_integrity.values()):
        return "integrity_failed"
    if all(r["status"] in ("ok", "empty_reconstruction") for r in results):
        return "complete"
    return "completed_with_failures"


def write_csv(path, results):
    columns = ["scene", "category", "spacing_mm", "status", "uniform_mae_mm", "uniform_p95_mm",
               "near_mae_mm", "near_p95_mm", "near_sampled_max_mm", "near_occupancy_iou",
               "surface_symmetric_mean_mm", "surface_f1_at_1mm", "surface_f1_at_2mm", "surface_f1_at_5mm",
               "components", "build_median_s", "query_median_s", "extract_median_s", "field_bytes", "observed_peak_rss_bytes", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            row = {"scene": r["scene"], "category": r["category"], "spacing_mm": 1000 * r["spacing_m"], "status": r["status"]}
            for subset in ("uniform", "near"):
                if f"{subset}_distance" in r:
                    for key in ("mae_mm", "p95_mm"):
                        row[f"{subset}_{key}"] = r[f"{subset}_distance"][key]
            if "near_distance" in r:
                row["near_sampled_max_mm"] = r["near_distance"]["sampled_max_mm"]
            if "near_occupancy" in r:
                row["near_occupancy_iou"] = r["near_occupancy"]["iou"]
            if "surface" in r:
                row["surface_symmetric_mean_mm"] = r["surface"]["symmetric_mean_mm"]
                for mm in (1, 2, 5):
                    row[f"surface_f1_at_{mm}mm"] = r["surface"]["f1"][str(mm / 1000)]
                row["components"] = r["surface"].get("connected_components")
            if "performance" in r:
                for step in ("build", "query", "extract"):
                    row[f"{step}_median_s"] = np.median(r["performance"][f"{step}_seconds"])
                row["observed_peak_rss_bytes"] = r["performance"]["observed_peak_rss_bytes"]
            if "artifacts" in r:
                row["field_bytes"] = r["artifacts"]["field_bytes"]
            row["error"] = r.get("error", "")
            writer.writerow(row)


def plot_summary(path, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    categories = sorted({r["category"] for r in results})
    for category in categories:
        good = [r for r in results if r["category"] == category and r.get("status") in ("ok", "empty_reconstruction")]
        h = sorted({r["spacing_m"] for r in good})
        groups = [[r for r in good if r["spacing_m"] == spacing] for spacing in h]
        values = [
            [np.mean([r["near_distance"]["p95_mm"] for r in g]) for g in groups],
            [np.mean([r["surface"]["f1"]["0.002"] for r in g]) for g in groups],
            [np.mean([r["near_occupancy"]["iou"] for r in g if r["near_occupancy"]["iou"] is not None]) for g in groups]]
        for ax, v in zip(axes, values):
            ax.plot(np.array(h) * 1000, v, marker="o", label=category)
    for ax, title, ylabel in zip(axes, ["Near-surface distance error", "Surface recovery at 2 mm", "Near-surface occupancy"], ["P95 absolute error (mm)", "Surface F1", "IoU"]):
        ax.set(title=title, xlabel="Grid spacing (mm)", ylabel=ylabel)
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=.25)
    axes[1].set_ylim(0, 1.03)
    axes[2].set_ylim(0, 1.03)
    axes[0].legend(fontsize=7)
    fig.suptitle("Equal-instance category means; individual failures remain in metrics.json")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _run(args):
    import geometry
    protocol = json.loads((HERE / "protocol.json").read_text())
    scenes = geometry.make_scenes()
    if args.scenes:
        names = set(args.scenes.split(","))
        scenes = [s for s in scenes if s.name in names or s.category in names]
        if not scenes:
            raise ValueError("no scenes matched selection")
    spacings = [float(x) for x in args.spacings.split(",")] if args.spacings else protocol["field"]["spacing_m"]
    count = args.samples if args.samples else protocol["evaluation"]["uniform_roi_points_per_scene"]
    surface_count = args.surface_samples if args.surface_samples else protocol["evaluation"]["surface_samples_per_direction"]
    repeats = args.timing_repeats if args.timing_repeats is not None else protocol["performance"]["timing_repeats_after_warmup"]
    if min(count, surface_count, repeats) <= 0:
        raise ValueError("sample counts and timing repeats must be positive")
    formal = (len(scenes) == 15 and spacings == protocol["field"]["spacing_m"] and count == 100_000 and surface_count == 100_000 and repeats == 3)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for d in ("source", "reference", "fields", "meshes", "figures", "configuration_metrics"):
        (out / d).mkdir()
    sources = sorted(HERE.glob("*.py")) + [HERE / "README.txt", HERE / "protocol.json"]
    hashes = {}
    for source in sources:
        copied = out / "source" / source.name
        shutil.copy2(source, copied)
        hashes[source.name] = sha256(copied)
        if hashes[source.name] != sha256(source):
            raise RuntimeError(f"source changed while freezing: {source.name}")
    versions = {}
    for package in ("numpy", "scipy", "scikit-image", "trimesh", "rtree", "open3d", "matplotlib", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"status": "preparing_reference", "started_utc": datetime.now(timezone.utc).isoformat(),
                "formal_protocol_run": formal, "model_calls": 0, "robotics": False,
                "protocol_status_note": "Original protocol file is an unchanged draft snapshot; this manifest records actual execution.",
                "configuration_count": len(scenes) * len(spacings), "spacing_m": spacings,
                "uniform_and_near_points_each": count, "surface_points_each_direction": surface_count,
                "timing_repeats_after_full_warmup": repeats, "scene_geometry": [s.to_dict() for s in scenes],
                "sources_sha256": hashes, "versions": versions,
                "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                             "logical_cpu_count": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total},
                "python": sys.version, "command": sys.argv,
                "distance_truth": "known_continuous_geometry_not_RGB_reconstruction",
                "axis_order": "XYZ", "surface_algorithm": "skimage.measure.marching_cubes Lewiner level=0 gradient_direction=descent; positive-volume orientation validated",
                "peak_memory_method": "process RSS sampled every 10 ms across numeric evaluation; includes retained library caches; observed not guaranteed maximum",
                "near_surface_sampling": "surface area samples perturbed uniformly in a radius5mm ball then true-band filtered; not uniform band volume",
                "undefined_metrics": "null, never replaced with perfect scores",
                "sampled_max_warning": "maximum over saved evaluation points, not a global error bound"}
    write_json(out / "manifest.json", manifest)
    print(f"RUN {out} formal={formal} configs={manifest['configuration_count']}", flush=True)
    try:
        validation = geometry.validate_reference(scenes)
        write_json(out / "reference_validation.json", validation)
        if validation.get("passed") is not True:
            raise ValueError("independent geometry reference validation did not pass")
        print("REFERENCE VALIDATED", flush=True)
    except Exception:
        manifest.update(status="reference_invalid", error=traceback.format_exc())
        write_json(out / "manifest.json", manifest)
        raise
    references = {}
    for scene in scenes:
        samples, sampling = reference_samples(scene, count, surface_count, protocol["evaluation"]["near_surface_band_m"])
        path = out / "reference" / f"{scene.name}_points.npz"
        np.savez_compressed(path, **samples)
        scene.mesh().export(out / "reference" / f"{scene.name}_geometry.ply")
        write_json(out / "reference" / f"{scene.name}_geometry.json", scene.to_dict())
        references[scene.name] = {"points_sha256": sha256(path), "sampling": sampling}
    manifest.update(status="running", references=references)
    write_json(out / "manifest.json", manifest)
    results = []
    started = time.perf_counter()
    for scene in scenes:
        with np.load(out / "reference" / f"{scene.name}_points.npz") as data:
            samples = {k: data[k] for k in data.files}
        for h in spacings:
            name = f"{scene.name}_h{h * 1000:g}mm"
            result = {"scene": scene.name, "category": scene.category, "seed": scene.seed, "spacing_m": h}
            print(f"START {name}", flush=True)
            try:
                with RSSSampler() as rss:
                    grid, build_times = timed(lambda: SDFGrid.build(scene, h), repeats)
                    all_points = np.concatenate([samples["uniform_points_m"], samples["near_points_m"]])
                    predictions, query_times = timed(lambda: grid.query(all_points), repeats)
                    mesh, extract_times = timed(grid.surface, repeats)
                    for label, pred in zip(("uniform", "near"), np.split(predictions, 2)):
                        truth = samples[f"{label}_reference_m"]
                        result[f"{label}_distance"] = distance_statistics(pred, truth, samples[f"{label}_points_m"])
                        result[f"{label}_occupancy"] = occupancy_statistics(pred, truth, protocol["evaluation"]["sign_boundary_exclusion_m"])
                        result[f"{label}_occupancy_unexcluded"] = occupancy_statistics(pred, truth, None)
                    surface, surface_data = surface_statistics(mesh, scene, samples["reference_surface_points_m"], surface_count,
                                                              scene.seed + 310_019, protocol["evaluation"]["surface_tolerances_m"])
                    result["surface"] = surface
                    result["features"] = scene.feature_checks(grid.query)
                    result["reference_features"] = scene.feature_checks(scene.sdf)
                result["status"] = surface["status"]
                result["grid_vertex_quantization_max_m"] = grid.vertex_quantization_max_m
                result["performance"] = {"build_seconds": build_times, "query_seconds": query_times,
                    "extract_seconds": extract_times, "query_points": len(all_points),
                    "query_points_per_second_median": len(all_points) / float(np.median(query_times)),
                    "grid_value_bytes": grid.values.nbytes, "observed_peak_rss_bytes": rss.peak,
                    "rss_baseline_bytes": rss.baseline, "visualization_excluded_from_algorithm_timing": True}
                field_path = out / "fields" / f"{name}.npz"
                grid.save(field_path)
                if mesh is not None:
                    mesh.export(out / "meshes" / f"{name}.ply")
                result["artifacts"] = {"field": str(field_path.relative_to(out)), "field_sha256": sha256(field_path),
                                       "field_bytes": field_path.stat().st_size,
                                       "mesh": f"meshes/{name}.ply" if mesh is not None else None}
                result["proposed_target_at_1mm"] = {
                    "applicability": "ordinary categories with declared minimum feature >=4h; thin-wall/narrow-gap stress scenes evaluated separately",
                    "minimum_feature_size_m": scene.minimum_feature_size_m,
                    "applicable": bool(h == .001 and "thin" not in scene.category and scene.minimum_feature_size_m >= 4 * h),
                    "near_p95_le_1mm": result["near_distance"]["p95_mm"] <= 1,
                    "surface_f1_2mm_ge_095": surface["f1"]["0.002"] >= .95}
                plot_config(out / "figures" / f"{name}.png", scene, grid, surface_data, samples["reference_surface_points_m"])
                del grid, mesh, surface_data
                print(f"DONE {name} near_P95={result['near_distance']['p95_mm']:.6f}mm F1@2={surface['f1']['0.002']:.6f}", flush=True)
            except Exception:
                result.update(status="failed", error=traceback.format_exc())
                print(f"FAILED {name}: {result['error']}", flush=True)
            results.append(result)
            write_json(out / "configuration_metrics" / f"{name}.json", result)
            write_json(out / "metrics.json", {"status": "running", "configurations": results})
            write_csv(out / "metrics.csv", results)
    plot_summary(out / "figures" / "resolution_summary.png", results)
    integrity = {name: (HERE / name).is_file() and sha256(HERE / name) == digest for name, digest in hashes.items()}
    summary = {"status": final_run_status(results, integrity),
               "formal_protocol_run": formal, "configurations": results, "aggregates": aggregate_metrics(results),
               "sources_unchanged": integrity, "elapsed_evaluation_s": time.perf_counter() - started}
    write_json(out / "metrics.json", summary)
    manifest.update(status=summary["status"], finished_utc=datetime.now(timezone.utc).isoformat(),
                    sources_unchanged=integrity,
                    elapsed_evaluation_s=summary["elapsed_evaluation_s"])
    write_json(out / "manifest.json", manifest)
    print(f"FINISHED {manifest['status']} {len(results)} configurations", flush=True)
    return 0 if summary["status"] == "complete" else 1


def run(args):
    """Finalize unexpected errors without editing pre-existing run directories."""
    out = Path(args.output)
    already_existed = out.exists()
    try:
        return _run(args)
    except (Exception, KeyboardInterrupt) as exc:
        error = traceback.format_exc()
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted"
        if not already_existed and out.is_dir():
            for name in ("manifest.json", "metrics.json"):
                path = out / name
                record = json.loads(path.read_text()) if path.exists() else {}
                if record.get("status") == "reference_invalid":
                    record["terminal_status"] = status
                else:
                    record["status"] = status
                record.update(terminal_error=error, finished_utc=datetime.now(timezone.utc).isoformat())
                write_json(path, record)
            (out / "terminal_error.txt").write_text(error)
        print(f"{status.upper()} {error}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new directory; existing paths are refused")
    p.add_argument("--scenes", help="comma-separated scene names or categories for explicitly nonformal subset")
    p.add_argument("--spacings", help="comma-separated metres; default approved 0.004,0.002,0.001")
    p.add_argument("--samples", type=int, help="points in EACH uniform/near set; default100000")
    p.add_argument("--surface-samples", type=int, help="surface samples in EACH direction; default100000")
    p.add_argument("--timing-repeats", type=int, help="measured repeats after warmup; default3")
    sys.exit(run(p.parse_args()))
