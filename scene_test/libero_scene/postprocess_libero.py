"""Standalone visual report for a COMPLETED real-scene SDF experiment.

Only a new output directory is written. Metrics, field values, captures and
runtime sources are read-only inputs. Feature zooms are selected solely from
the smallest predeclared reference interval, never from reconstruction errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path[:0] = [str(HERE / "_deps"), str(PARENT / "_deps"), str(HERE), str(PARENT)]
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sdf_grid import SDFGrid
from reconstruct import sample_mesh, write_json, sha256
from reconstruct_libero import clip_surface_to_roi, unsigned_mesh_distance
from union_geometry import UnionScene

NAMES = {"bowl002_initial": "Bowl + plate", "drawer003_closed_step60": "Drawer closed", "drawer003_open_step552": "Drawer open"}
OBJECTS = {"akita_black_bowl_1": "Bowl", "plate_1": "Plate", "wooden_cabinet_1": "Cabinet"}
SPACING_COLOR = {.004: "#d97706", .002: "#2563eb", .001: "#059669"}
SCENE_COLOR = {"bowl002_initial": "#7c3aed", "drawer003_closed_step60": "#2563eb", "drawer003_open_step552": "#d97706"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.labelcolor": "#334155",
                         "xtick.color": "#475569", "ytick.color": "#475569"})


def metric_curves(path, results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    for name in NAMES:
        rows = sorted([r for r in results if r["snapshot"] == name], key=lambda r: r["spacing_m"])
        if not rows:
            continue
        spacing = [1000 * row["spacing_m"] for row in rows]
        color = SCENE_COLOR[name]
        axes[0, 0].plot(spacing, [r["near_distance"]["p95_mm"] for r in rows], "o-", color=color, label=NAMES[name], linewidth=2)
        axes[0, 1].plot(spacing, [r["near_occupancy"]["iou"] for r in rows], "o-", color=color, label=NAMES[name], linewidth=2)
        for object_name in rows[0]["surface"]["per_object"]:
            marker, line_style = ("s", "--") if object_name == "plate_1" else ("o", "-")
            label = f"{NAMES[name]} / {OBJECTS.get(object_name, object_name)}"
            for ax, tolerance in zip(axes[1], ("0.001", "0.002")):
                values = [r["surface"]["per_object"][object_name]["f1"][tolerance] for r in rows]
                ax.plot(spacing, values, marker=marker, linestyle=line_style, color=color, label=label, linewidth=2)
    settings = [(axes[0, 0], "Distance accuracy near actual surfaces", "P95 absolute SDF error (mm)"),
                (axes[0, 1], "Material occupancy near surfaces", "Occupancy IoU"),
                (axes[1, 0], "Object surface recovery: 1 mm tolerance", "Area-sampled surface F1"),
                (axes[1, 1], "Object surface recovery: 2 mm tolerance", "Area-sampled surface F1")]
    for ax, title, ylabel in settings:
        ax.set(title=title, xlabel="Grid spacing (mm)", ylabel=ylabel, xticks=[1, 2, 4])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylim(bottom=0)
    for ax in (axes[0, 1], *axes[1]):
        ax.set_ylim(0, 1.035)
    fig.suptitle("Real LIBERO collision geometry: representation accuracy", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .935, "Known geometry → sampled SDF → recovered surface • local task ROIs • same evaluation points at every resolution",
             ha="center", color="#475569", fontsize=10)
    fig.text(.5, .018, "High F1 at a tolerance does not guarantee that thin walls, narrow gaps or connectivity are preserved.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def load_grid(run, row):
    path = run / row["field"]
    if sha256(path) != row["field_sha256"]:
        raise ValueError(f"field hash mismatch: {path}")
    return SDFGrid.load(path)


def feature_sections(path, run, snapshots, results):
    names = [name for name in NAMES if any(r["snapshot"] == name for r in results)]
    fig, axes = plt.subplots(len(names), 3, figsize=(17, 4.1 * len(names)), squeeze=False)
    metadata = []
    for row_index, name in enumerate(names):
        rows = sorted([r for r in results if r["snapshot"] == name], key=lambda r: r["spacing_m"], reverse=True)
        feature = rows[0]["features"][0]
        start, end = np.asarray(feature["start_m"]), np.asarray(feature["end_m"])
        length = float(np.linalg.norm(end - start))
        positions = np.linspace(0, length, 8001)
        points = start + (positions / length)[:, None] * (end - start)
        scene = UnionScene.from_snapshot(snapshots[name])
        profiles = {"Reference": scene.sdf(points)}
        for row in rows:
            grid = load_grid(run, row)
            profiles[f"{row['spacing_m'] * 1000:g} mm"] = grid.query(points)
            del grid
        roots = np.asarray(feature["reference"]["positions_m"])
        intervals = np.diff(roots)
        if not len(intervals):
            raise ValueError("predeclared reference feature has no bounded interval")
        index = int(np.argmin(intervals))
        width = float(intervals[index])
        zoom_lower = max(0., roots[index] - max(.003, width * 1.5))
        zoom_upper = min(length, roots[index + 1] + max(.003, width * 1.5))
        ax_all, ax_zoom, ax_sign = axes[row_index]
        for ax in (ax_all, ax_zoom):
            ax.plot(positions * 1000, profiles["Reference"] * 1000, color="#0f172a", linewidth=2.5, label="Reference", zorder=5)
            for row in rows:
                spacing = row["spacing_m"]
                ax.plot(positions * 1000, profiles[f"{spacing * 1000:g} mm"] * 1000,
                        color=SPACING_COLOR[spacing], label=f"h = {spacing * 1000:g} mm", linewidth=1.7, alpha=.95)
            ax.axhline(0, color="#94a3b8", linewidth=.8, linestyle="--")
            ax.grid(alpha=.15)
            ax.set(xlabel="Distance along fixed reference line (mm)", ylabel="Signed surface distance (mm)")
        ax_all.set_title(f"{NAMES[name]}: full reference section", fontsize=11)
        ax_all.legend(fontsize=8, ncol=2)
        ax_zoom.axvspan(roots[index] * 1000, roots[index + 1] * 1000, color="#cbd5e1", alpha=.35)
        ax_zoom.set_xlim(zoom_lower * 1000, zoom_upper * 1000)
        mask = (positions >= zoom_lower) & (positions <= zoom_upper)
        values = np.concatenate([v[mask] for v in profiles.values()]) * 1000
        ymin, ymax = min(-.1, float(values.min())), max(.1, float(values.max()))
        margin = max(.15, (ymax - ymin) * .12)
        ax_zoom.set_ylim(ymin - margin, ymax + margin)
        predicted_widths = {}
        width_labels = []
        for row in rows:
            predicted_roots = np.asarray(row["features"][0]["prediction"]["positions_m"])
            predicted_width = float(predicted_roots[index + 1] - predicted_roots[index]) if len(predicted_roots) == len(roots) else None
            predicted_widths[str(row["spacing_m"])] = predicted_width
            width_labels.append(f"{row['spacing_m'] * 1000:g}mm grid: {predicted_width * 1000:.3f}mm" if predicted_width is not None
                                else f"{row['spacing_m'] * 1000:g}mm grid: crossings changed")
        ax_zoom.set_title(f"Reference interval: {width * 1000:.3f} mm", fontsize=11)
        ax_zoom.text(.02, .98, "\n".join(width_labels), transform=ax_zoom.transAxes,
                     va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
        signs = np.array([v < 0 for v in profiles.values()], dtype=float)
        ax_sign.imshow(signs, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#f1f5f9", "#334155"]),
                       vmin=0, vmax=1, extent=(0, length * 1000, len(profiles) - .5, -.5))
        ax_sign.set_yticks(range(len(profiles)), profiles.keys())
        ax_sign.set(xlabel="Distance along the same line (mm)", title="Material / air along the section\ndark = material; light = air")
        ax_sign.axvline(roots[index] * 1000, color="#dc2626", linewidth=.7, alpha=.7)
        ax_sign.axvline(roots[index + 1] * 1000, color="#dc2626", linewidth=.7, alpha=.7)
        metadata.append({"snapshot": name, "feature_name": feature["name"], "start_m": start, "end_m": end,
                         "zoom_selection": "smallest positive inter-crossing interval from reference only",
                         "reference_interval_m": [roots[index], roots[index + 1]], "reference_interval_width_m": width,
                         "predicted_interval_widths_m": predicted_widths,
                         "zoom_extent_m": [zoom_lower, zoom_upper],
                         "reference_crossings": len(roots),
                         "predicted_crossings": {str(row["spacing_m"]): len(row["features"][0]["prediction"]["positions_m"]) for row in rows}})
    fig.suptitle("Do the walls and openings survive the grid?", fontsize=18, fontweight="bold", y=.995)
    fig.text(.5, .963, "Predeclared geometry sections • zoom chosen from the reference's smallest interval • negative SDF means solid material",
             ha="center", fontsize=10, color="#475569")
    fig.text(.5, .012, "These fixed sections diagnose local details; they do not prove global topology preservation.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, .025, 1, .945))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return metadata


def scene_overview(path, run, manifest, snapshots, results, capture_directory):
    names = [name for name in NAMES if any(r["snapshot"] == name and r["spacing_m"] == .001 for r in results)]
    fig = plt.figure(figsize=(15, 4.6 * len(names)))
    rgb_hashes = {}
    normalization = Normalize(vmin=0, vmax=1)
    for i, name in enumerate(names):
        snapshot = snapshots[name]
        roi = manifest["roi"][name]
        row = next(r for r in results if r["snapshot"] == name and r["spacing_m"] == .001)
        rgb = snapshot["reference_rgb"]["agentview"]
        rgb_path = capture_directory / rgb["file"]
        if sha256(rgb_path) != rgb["sha256"]:
            raise ValueError("context RGB hash differs from preserved capture")
        rgb_hashes[str(rgb_path)] = rgb["sha256"]
        ax = fig.add_subplot(len(names), 3, 3 * i + 1)
        ax.imshow(plt.imread(rgb_path))
        ax.set_title(f"{NAMES[name]} — actual task image", fontsize=12)
        ax.axis("off")
        scene = UnionScene.from_snapshot(snapshot)
        reference = clip_surface_to_roi(scene.mesh(), roi["score_lower_m"], roi["score_upper_m"])
        reference_points = sample_mesh(reference, 28_000, np.random.default_rng(1720 + i))
        labels = scene.surface_object_labels(reference_points)
        target = np.isin(labels, roi["primary_objects"])
        ax = fig.add_subplot(len(names), 3, 3 * i + 2, projection="3d")
        ax.scatter(*(reference_points[~target] * 1000).T, s=.28, color="#cbd5e1", alpha=.6)
        ax.scatter(*(reference_points[target] * 1000).T, s=.4, color="#0369a1", alpha=.85)
        ax.set_title("Known collision surfaces in task ROI", fontsize=12)
        lower, upper = np.asarray(roi["score_lower_m"]), np.asarray(roi["score_upper_m"])
        axes3d = [ax]
        mesh_path = run / "meshes" / f"{name}_h1mm.ply"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        prediction = clip_surface_to_roi(mesh, lower, upper)
        points = sample_mesh(prediction, 28_000, np.random.default_rng(1820 + i))
        error = unsigned_mesh_distance(scene.mesh(), points) * 1000
        ax = fig.add_subplot(len(names), 3, 3 * i + 3, projection="3d")
        ax.scatter(*(points * 1000).T, c=error, norm=normalization, cmap="viridis", s=.45, alpha=.9)
        ax.set_title("1 mm reconstruction: surface error", fontsize=12)
        axes3d.append(ax)
        for ax in axes3d:
            ax.set(xlim=(lower[0] * 1000, upper[0] * 1000), ylim=(lower[1] * 1000, upper[1] * 1000),
                   zlim=(lower[2] * 1000, upper[2] * 1000), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
            ax.set_box_aspect(upper - lower)
            ax.view_init(elev=24, azim=65)
            ax.tick_params(labelsize=7)
        colorbar = fig.colorbar(ScalarMappable(norm=normalization, cmap="viridis"), ax=axes3d[-1], shrink=.6, pad=.01)
        colorbar.set_label("Unsigned surface error (mm)")
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("From the real task scene to a geometric distance field", fontsize=18, fontweight="bold", y=.995)
    fig.text(.5, .963, "RGB is context only; geometry comes from simulator collision shapes. 3D panels use a separate viewpoint and the local evaluation ROI.",
             ha="center", fontsize=10, color="#475569")
    fig.text(.5, .012, "Blue: task objects; gray: surrounding collision surfaces. Error color saturates at 1 mm; this is a display, not the metric sample set.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .025, 1, .95))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return rgb_hashes


def main(args):
    run = Path(args.run).resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    if manifest.get("status") != "complete" or metrics.get("status") != "complete":
        raise ValueError("postprocessing requires a completed, successful source run")
    results = metrics["configurations"]
    if any(row["status"] != "ok" for row in results):
        raise ValueError("this visualization requires nonempty successful reconstructed surfaces; metrics retain any failures")
    snapshot_file = run / "reference" / "snapshots.json"
    if sha256(snapshot_file) != manifest["capture_sha256"]:
        raise ValueError("frozen snapshot input hash mismatch")
    snapshots = {s["name"]: s for s in json.loads(snapshot_file.read_text())["snapshots"]}
    for file, digest in manifest["sources_sha256"].items():
        if sha256(file) != digest:
            raise ValueError(f"runtime source differs from measurement run: {file}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(__file__)
    shutil.copy2(__file__, output / "postprocess_used.py")
    input_hashes = {str(run / name): sha256(run / name) for name in ("manifest.json", "metrics.json", "settings.json")}
    report = {"status": "running", "source_run": str(run), "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_script_sha256": source_hash, "input_sha256": input_hashes,
              "outputs_are_postprocessing_only": True}
    write_json(output / "manifest.json", report)
    style()
    print("PLOT metric curves", flush=True)
    metric_curves(output / "metric_curves.png", results)
    print("PLOT fixed geometry feature sections", flush=True)
    features = feature_sections(output / "feature_sections.png", run, snapshots, results)
    write_json(output / "feature_zoom_definitions.json", features)
    print("PLOT real-scene overview", flush=True)
    capture_directory = Path(manifest["capture_path"]).parent
    rgb_hashes = scene_overview(output / "scene_overview.png", run, manifest, snapshots, results, capture_directory)
    summary = []
    for row in results:
        summary.append({"snapshot": row["snapshot"], "spacing_m": row["spacing_m"],
                        "near_distance_p95_mm": row["near_distance"]["p95_mm"], "near_occupancy_iou": row["near_occupancy"]["iou"],
                        "per_object": row["surface"]["per_object"], "features": row["features"]})
    write_json(output / "summary.json", summary)
    unchanged = {file: sha256(file) == digest for file, digest in input_hashes.items()}
    unchanged["postprocess_source"] = sha256(__file__) == source_hash
    report.update(status="complete" if all(unchanged.values()) else "integrity_failed",
                  finished_utc=datetime.now(timezone.utc).isoformat(), inputs_unchanged=unchanged,
                  rgb_inputs_sha256=rgb_hashes,
                  figures_sha256={p.name: sha256(p) for p in output.glob("*.png")})
    write_json(output / "manifest.json", report)
    print(f"FINISHED {report['status']}: {output}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True, help="new directory only")
    sys.exit(main(parser.parse_args()))
