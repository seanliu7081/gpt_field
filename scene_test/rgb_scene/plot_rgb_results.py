#!/usr/bin/env python3
"""Plot RGB -> serialized projective field -> field-decoded geometry.

No model inference, depth-to-mesh replacement, or geometry reconstruction is
performed here. The field panel reads saved scalar arrays and observed masks;
the geometry panel reads the already-decoded mesh artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from PIL import Image


NAMES = ("bowl002_initial", "drawer003_closed_step60", "drawer003_open_step552")
LABELS = {NAMES[0]: "Bowl · initial", NAMES[1]: "Drawer · closed", NAMES[2]: "Drawer · open"}
SPACING_M = .001
X_LIMITS = (-1.0, 1.0)
Z_LIMITS = (.35, 1.65)
DEPTH_LIMITS_M = (.4, 1.6)
TSDF_LIMIT_MM = 20.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checked(path, digest, consumed):
    path = Path(path).resolve()
    actual = sha256(path)
    if actual != digest:
        raise ValueError(f"Artifact hash mismatch: {path}")
    consumed[str(path)] = actual
    return path


def saved_field_slice(build, config, snapshot, consumed):
    """Merge the y=0 planes of actual saved fields, restricted to their cores."""
    spacing = float(config["spacing_m"])
    xs = np.arange(round((X_LIMITS[1]-X_LIMITS[0])/spacing)+1)*spacing+X_LIMITS[0]
    zs = np.arange(round((Z_LIMITS[1]-Z_LIMITS[0])/spacing)+1)*spacing+Z_LIMITS[0]
    mosaic = np.full((len(xs), len(zs)), np.nan, dtype=np.float32)
    records = {Path(row["file"]).stem: row for row in config["fields"]}
    used, overlap_max = [], 0.0
    for block in snapshot["blocks"]:
        low, high = np.asarray(block["lower_m"]), np.asarray(block["upper_m"])
        if low[1] > 0 or high[1] < 0:
            continue
        record = records[block["id"]]
        path = checked(build / record["file"], record["sha256"], consumed)
        with np.load(path, allow_pickle=False) as data:
            lower, h = data["lower_m"], float(data["spacing_m"])
            if not np.isclose(h, spacing, rtol=0, atol=1e-14):
                raise ValueError("Field spacing disagrees with configuration")
            yi_float = -lower[1]/h
            yi = round(yi_float)
            if abs(yi-yi_float) > 1e-7:
                raise ValueError("The fixed y=0 plane is not a saved grid plane")
            values = data["values"][:, yi, :].copy()
            valid = data["observed"][:, yi, :].copy()
            metadata = json.loads(str(data["metadata_json"]))
        x = lower[0]+h*np.arange(values.shape[0])
        z = lower[2]+h*np.arange(values.shape[1])
        X, Z = np.meshgrid(x, z, indexing="ij")
        points = np.stack((X, np.zeros_like(X), Z), axis=-1)
        for plane in np.asarray(metadata["clip_planes"]):
            valid &= points@plane[:3]+plane[3] >= -1e-10
        ix = np.rint((x-X_LIMITS[0])/spacing).astype(int)
        iz = np.rint((z-Z_LIMITS[0])/spacing).astype(int)
        valid &= (ix[:, None] >= 0) & (ix[:, None] < len(xs))
        valid &= (iz[None, :] >= 0) & (iz[None, :] < len(zs))
        a, b = np.nonzero(valid)
        old = mosaic[ix[a], iz[b]]
        overlap = np.isfinite(old)
        if np.any(overlap):
            error = float(np.max(np.abs(old[overlap]-values[a[overlap], b[overlap]])))
            overlap_max = max(overlap_max, error)
            if error > 2e-6:
                raise ValueError("Overlapping saved field slices disagree by more than 2 micrometers")
        mosaic[ix[a], iz[b]] = values[a, b]
        used.append({"file": record["file"], "sha256": record["sha256"]})
        if len(used) % 20 == 0:
            print(json.dumps({"slice": config["snapshot"], "loaded_fields": len(used)}), flush=True)
    if not np.any(np.isfinite(mosaic)):
        raise ValueError(f"No observed saved field vertices intersect y=0: {config['snapshot']}")
    return mosaic, {"slice_camera_y_m": 0.0, "spacing_m": spacing,
                    "x_limits_m": X_LIMITS, "z_limits_m": Z_LIMITS,
                    "values_source": "saved NPZ values; masked by saved observed vertices and saved block core planes",
                    "unknown_source": "saved unobserved vertices or no saved block; no depth-derived filling",
                    "observed_pixel_fraction_in_display_window": float(np.isfinite(mosaic).mean()),
                    "overlap_max_abs_difference_m": overlap_max, "fields": used}


@njit(cache=False)
def rasterize_triangles(projected, view_depth, native_depth, vertices, faces, palette, size):
    """Opaque two-sided triangle renderer: every saved face participates."""
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    zbuffer = np.full((size, size), -np.inf, dtype=np.float64)
    light = np.array([-.3, -.6, .8])
    light /= np.sqrt(np.sum(light*light))
    rasterized_faces = 0
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        x0, y0 = projected[a, 0], projected[a, 1]
        x1, y1 = projected[b, 0], projected[b, 1]
        x2, y2 = projected[c, 0], projected[c, 1]
        denominator = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denominator) < 1e-12:
            continue
        xmin = max(0, int(np.ceil(min(x0, x1, x2))))
        xmax = min(size-1, int(np.floor(max(x0, x1, x2))))
        ymin = max(0, int(np.ceil(min(y0, y1, y2))))
        ymax = min(size-1, int(np.floor(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue
        edge1, edge2 = vertices[b]-vertices[a], vertices[c]-vertices[a]
        normal = np.cross(edge1, edge2)
        length = np.sqrt(np.sum(normal*normal))
        shade = .45 if length <= 1e-20 else .45+.55*abs(np.dot(normal, light)/length)
        color_index = min(255, max(0, int(255*((native_depth[a]+native_depth[b]+native_depth[c])/3-.4)/1.2)))
        color = palette[color_index]*shade
        hit = False
        for yy in range(ymin, ymax+1):
            for xx in range(xmin, xmax+1):
                wa = ((y1-y2)*(xx-x2)+(x2-x1)*(yy-y2))/denominator
                wb = ((y2-y0)*(xx-x2)+(x0-x2)*(yy-y2))/denominator
                wc = 1-wa-wb
                if wa >= -1e-10 and wb >= -1e-10 and wc >= -1e-10:
                    depth = wa*view_depth[a]+wb*view_depth[b]+wc*view_depth[c]
                    if depth > zbuffer[yy, xx]:
                        zbuffer[yy, xx] = depth
                        for channel in range(3):
                            image[yy, xx, channel] = np.uint8(min(255, max(0, color[channel])))
                        hit = True
        if hit:
            rasterized_faces += 1
    return image, zbuffer, rasterized_faces


def draw_actual_mesh(ax, path):
    """Render every face from the artifact with a fixed oblique orthographic view."""
    with np.load(path, allow_pickle=False) as data:
        vertices, faces = data["vertices_m"], data["faces"]
    if not len(faces):
        raise ValueError("Cannot display an empty decoded mesh")
    size = 800
    world = vertices[:, [0, 2, 1]].copy()
    world[:, 2] *= -1  # CV y down -> figure vertical up.
    elevation, azimuth = np.deg2rad([18., -65.])
    eye = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    up = np.cross(eye, right)
    box = np.array([[x, z, height] for x in X_LIMITS for z in Z_LIMITS for height in (-1., 1.)])
    camera = np.stack((right, up), axis=1)
    box_uv = box@camera
    center = (box_uv.min(0)+box_uv.max(0))/2
    scale = (size-30)/np.max(np.ptp(box_uv, axis=0))
    projected = (world@camera-center)*scale+(size-1)/2
    projected[:, 1] = (size-1)-projected[:, 1]
    palette = (plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]*255).astype(np.float64)
    rendered, visible, rasterized_faces = rasterize_triangles(
        projected, world@eye, vertices[:, 2], world, faces, palette, size)
    ax.imshow(rendered)
    ax.axis("off")
    ax.text(.01, .02, f"All {len(faces):,} artifact faces submitted; fixed oblique view",
            transform=ax.transAxes, fontsize=8, color="#444444")
    return {"artifact_faces": len(faces), "submitted_faces": len(faces),
            "rasterized_faces_writing_pixels": int(rasterized_faces),
            "rendered_surface_pixel_fraction": float(np.isfinite(visible).mean()),
            "display_sampling": "none; all saved triangles rendered with z-buffer occlusion, no new geometry",
            "vertex_coordinate_permutation_for_display": "(x,z,-y)",
            "camera_depth_color_limits_m": DEPTH_LIMITS_M,
            "display_elevation_deg": 18., "display_azimuth_deg": -65., "render_size": [size, size]}


def representation_plot(metrics, output):
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for index, name in enumerate(NAMES):
        rows = sorted((c for c in metrics["configurations"] if c["snapshot"] == name),
                      key=lambda row: row["spacing_m"], reverse=True)
        h = np.array([r["spacing_m"]*1000 for r in rows])
        errors = [r["representation_B"]["symmetric_mean_m"] for r in rows]
        p95 = [max(r["representation_B"][direction]["p95_m"] for direction in
                    ("prediction_to_reference", "reference_to_prediction")) for r in rows]
        axes[0, index].plot(h, np.asarray(errors)*1000, "o-", label="B symmetric mean")
        axes[0, index].plot(h, np.asarray(p95)*1000, "s--", label="B larger directional P95")
        axes[0, index].set(title=LABELS[name], ylabel="Preservation error (mm)")
        for threshold in ("1mm", "2mm", "5mm"):
            axes[1, index].plot(h, [r["representation_B"]["thresholds"][threshold]["f1"]*100 for r in rows],
                                "o-", label=f"F1 @ {threshold}")
        axes[1, index].set(ylabel="Surface F1 (%)", ylim=(0, 101))
        retained_area = [100*r["representation_B"]["prediction_area_m2"] /
                         r["representation_B"]["reference_area_m2"] for r in rows]
        axes[2, index].plot(h, retained_area, "o-", color="#238b45", label="Decoded / source area")
        axes[2, index].set(ylabel="Decoded / source surface area (%)", ylim=(0, 105))
        for ax in axes[:, index]:
            ax.set(xlabel="Voxel spacing (mm)", xticks=[1, 2, 4], xlim=(4.2, .8))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    figure.suptitle("B: saved-field decode versus its input predicted surface\n"
                    "Representation loss, not scene accuracy. Area changes include smoothing and holes; F1 can pass nearby missing strips.", fontsize=12)
    figure.savefig(output / "representation_loss.png", dpi=180)
    plt.close(figure)


def evaluation_plots(evaluation, build, build_configs, output, consumed):
    """A/C use the same evaluator-only region; no GT value enters the field."""
    evaluation = Path(evaluation).resolve()
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or Path(manifest["fields_run"]).resolve() != build:
        raise ValueError("Evaluator must be complete and refer to this exact field build")
    consumed[str(manifest_path)] = sha256(manifest_path)
    metrics_path = checked(evaluation / "metrics.json", manifest["metrics_sha256"], consumed)
    metrics = json.loads(metrics_path.read_text())
    scenes = {row["name"]: row for row in metrics["snapshots"]}
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharey="row", constrained_layout=True)
    region = "nonrobot_visible"
    for index, name in enumerate(NAMES):
        A = scenes[name]["regions"][region]["perception_A_rays"]
        configs = sorted((row for row in metrics["configurations"] if row["snapshot"] == name),
                         key=lambda row: row["spacing_m"], reverse=True)
        h = np.array([row["spacing_m"]*1000 for row in configs])
        C = [row["ray_regions"][region]["end_to_end_C_rays"] for row in configs]
        axes[0, index].axhline(A["coverage"]*100, linestyle="--", color="#777777", label="A: before field")
        axes[0, index].plot(h, [row["coverage"]*100 for row in C], "o-", color="#8c3db3", label="C: decoded field")
        axes[0, index].set(title=LABELS[name], ylabel="First-hit coverage (%)", ylim=(0, 101))
        for key, label, style in (("mae_on_hits_m", "MAE", "-"), ("p95_on_hits_m", "P95", ":")):
            axes[1, index].axhline(A[key]*1000, linestyle=style, color="#777777", label=f"A {label}")
            axes[1, index].plot(h, [row[key]*1000 for row in C], marker="o", linestyle=style,
                                color="#8c3db3", label=f"C {label}")
        axes[1, index].set(ylabel="Absolute Z error on hits (mm)")
        axes[2, index].axhline(A["within_threshold_all_reference_pixels"]["5mm"]*100,
                              linestyle="--", color="#777777", label="A: before field")
        axes[2, index].plot(h, [row["within_threshold_all_reference_pixels"]["5mm"]*100 for row in C],
                            "o-", color="#8c3db3", label="C: decoded field")
        axes[2, index].set(ylabel="All reference pixels within 5 mm (%)", ylim=(0, 101))
        for ax in axes[:, index]:
            ax.set(xlabel="Voxel spacing (mm)", xticks=[1, 2, 4], xlim=(4.2, .8))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    figure.suptitle("A / C: first-hit geometry accuracy on the SAME GT-camera rays\n"
                    "Nonrobot visible pixels; native metric scale, no alignment. Misses remain failures in coverage / within-5mm rates.", fontsize=13)
    figure.savefig(output / "scene_accuracy_AC.png", dpi=180)
    plt.close(figure)

    artifact_by_name = {row["file"]: row for row in manifest["artifacts"]}
    figure, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
    comparison = []
    for index, name in enumerate(NAMES):
        gt_relative = f"distances/{name}_perception_ray_depth.npz"
        decoded_relative = f"distances/{build_configs[name]['name']}_field_ray_depth.npz"
        gt_record, decoded_record = artifact_by_name[gt_relative], artifact_by_name[decoded_relative]
        gt_path = checked(evaluation / gt_relative, gt_record["sha256"], consumed)
        decoded_path = checked(evaluation / decoded_relative, decoded_record["sha256"], consumed)
        with np.load(gt_path, allow_pickle=False) as data:
            gt = data["reference_z_m"]
            valid = data["valid_reference"].astype(bool)
        with np.load(decoded_path, allow_pickle=False) as data:
            decoded = data["decoded_gt_rays_z_m"]
        hit = valid & np.isfinite(decoded) & (decoded > 0)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d3d3d3")
        gt_view = axes[index, 0].imshow(np.ma.masked_where(~valid, gt), cmap=cmap,
                                      vmin=DEPTH_LIMITS_M[0], vmax=DEPTH_LIMITS_M[1])
        axes[index, 0].set_title(f"{LABELS[name]} · exact GT camera Z", fontsize=11)
        axes[index, 1].imshow(np.ma.masked_where(~hit, decoded), cmap=cmap,
                              vmin=DEPTH_LIMITS_M[0], vmax=DEPTH_LIMITS_M[1])
        axes[index, 1].set_title("Camera Z from saved-field mesh · h=1 mm", fontsize=11)
        error_cmap = plt.get_cmap("magma").copy()
        error_cmap.set_bad("#d3d3d3")
        error = np.abs(decoded-gt)*1000
        error_view = axes[index, 2].imshow(np.ma.masked_where(~hit, error), cmap=error_cmap, vmin=0, vmax=200)
        missing = np.zeros((*valid.shape, 4))
        missing[valid & ~hit] = [0., .75, .9, 1.]
        axes[index, 2].imshow(missing)
        axes[index, 2].set_title("Absolute error · cyan = missed reference ray", fontsize=11)
        for ax in axes[index]:
            ax.axis("off")
        figure.colorbar(gt_view, ax=axes[index, :2].tolist(), shrink=.6, label="Camera Z (m), fixed 0.4–1.6")
        figure.colorbar(error_view, ax=axes[index, 2], shrink=.6, label="Error (mm), capped at 200")
        comparison.append({"name": name, "reference_rays": gt_record, "decoded_rays": decoded_record,
                           "projection": "identical ground-truth camera K for reference and field-decoded mesh; evaluation only",
                           "valid_reference_pixels": int(valid.sum()), "decoded_hit_pixels": int(hit.sum()),
                           "decoded_mesh": build_configs[name]["decoded_mesh"],
                           "depth_color_limits_m": DEPTH_LIMITS_M, "error_color_limits_mm": [0, 200]})
    figure.suptitle("Exact visible reference versus the geometry decoded from the saved field\n"
                    "Same GT camera, pixel axes and color scale for both. No rescaling, camera refit or depth-to-mesh replacement.", fontsize=13)
    figure.savefig(output / "reference_comparison.png", dpi=180)
    plt.close(figure)
    return {"evaluation": str(evaluation), "A_C_region": region, "comparison": comparison}


def run(args):
    build, output = Path(args.build).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis output: {output}")
    build_manifest_path = build / "manifest.json"
    build_manifest = json.loads(build_manifest_path.read_text())
    if build_manifest.get("status") != "complete":
        raise ValueError("Plot only a complete saved-field run")
    consumed = {str(build_manifest_path): sha256(build_manifest_path)}
    metrics_path = checked(build / "metrics.json", build_manifest["metrics_sha256"], consumed)
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") != "complete":
        raise ValueError("Field metrics are not complete")
    predictions_path = Path(build_manifest["prediction_manifest"])
    frozen_prediction_record = next(row for row in build_manifest["inputs"] if row["path"] == str(predictions_path))
    checked(predictions_path, frozen_prediction_record["sha256"], consumed)
    predictions = json.loads(predictions_path.read_text())
    if predictions.get("status") != "complete":
        raise ValueError("Prediction run is not complete")
    rows = {row["name"]: row for row in predictions["inputs"]}
    snapshots = {row["name"]: row for row in metrics["snapshots"]}
    configs = {row["snapshot"]: row for row in metrics["configurations"]
               if np.isclose(row["spacing_m"], SPACING_M, atol=1e-14, rtol=0)}
    if set(configs) != set(NAMES) or any(configs[name]["status"] != "ok" for name in NAMES):
        raise ValueError("All three successful 1mm field configurations are required")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    source_hash = sha256(source)
    shutil.copy2(source, output / source.name)
    manifest = {"status": "running", "stage": "visualization_of_actual_saved_artifacts",
                "source": str(source), "source_sha256": source_hash,
                "build": str(build), "started_unix": time.time(),
                "consumed_sha256": consumed, "scenes": [],
                "primary_figure": "pipeline.png", "surface_error_figure": "representation_loss.png"}
    write_json(output / "manifest.json", manifest)
    try:
        fig = plt.figure(figsize=(15.5, 13.5), constrained_layout=True)
        grid = fig.add_gridspec(3, 3, width_ratios=(1, 1.1, 1.2))
        for row_index, name in enumerate(NAMES):
            rgb = checked(Path(rows[name]["rgb_path"]), rows[name]["rgb_sha256"], consumed)
            ax = fig.add_subplot(grid[row_index, 0])
            with Image.open(rgb) as image:
                ax.imshow(image)
            ax.set_title(f"{LABELS[name]}\nSingle upright RGB input", fontsize=11)
            ax.axis("off")
            field, info = saved_field_slice(build, configs[name], snapshots[name], consumed)
            np.savez_compressed(output / f"{name}_saved_field_slice.npz", values_m=field,
                                observed=np.isfinite(field), lower_xz_m=[X_LIMITS[0], Z_LIMITS[0]],
                                spacing_m=SPACING_M, camera_y_m=0.0)
            ax = fig.add_subplot(grid[row_index, 1])
            cmap = plt.get_cmap("RdBu_r").copy()
            cmap.set_bad("#d3d3d3")
            image = ax.imshow(np.ma.masked_invalid(field.T*1000), origin="upper",
                              extent=(*X_LIMITS, Z_LIMITS[1], Z_LIMITS[0]), interpolation="nearest",
                              aspect="equal", cmap=cmap, vmin=-TSDF_LIMIT_MM, vmax=TSDF_LIMIT_MM)
            ax.set(xlabel="camera x (m)", ylabel="camera z (m)",
                   title="Actual saved projective TSDF · h=1 mm\ny=0 slice; gray = unknown / outside saved support")
            fig.colorbar(image, ax=ax, shrink=.6, label="Axial residual (mm)")
            decoded = configs[name]["decoded_mesh"]
            mesh_path = checked(build / decoded["file"], decoded["sha256"], consumed)
            ax = fig.add_subplot(grid[row_index, 2])
            mesh_info = draw_actual_mesh(ax, mesh_path)
            ax.set_title("Actual field-decoded mesh · h=1 mm\nOpen visible surface; camera depth colors", fontsize=11)
            manifest["scenes"].append({"name": name, "field_slice": info,
                                       "decoded_mesh": {**decoded, **mesh_info},
                                       "rgb_sha256": rows[name]["rgb_sha256"]})
            print(json.dumps({"plot_scene_complete": name}), flush=True)
        fig.suptitle("Single RGB → saved 3D projective field → geometry decoded from that field\n"
                     "All three scenes share axis/color scales. Unknown space is not inferred free space. "
                     "This field is not a Euclidean SDF.", fontsize=14)
        fig.savefig(output / "pipeline.png", dpi=180)
        plt.close(fig)
        representation_plot(metrics, output)
        if args.evaluation:
            manifest["evaluation_visuals"] = evaluation_plots(
                args.evaluation, build, configs, output, consumed)
        manifest["inputs_unchanged"] = {path: sha256(path) == digest for path, digest in consumed.items()}
        manifest["source_unchanged"] = sha256(source) == source_hash == sha256(output / source.name)
        if not all(manifest["inputs_unchanged"].values()) or not manifest["source_unchanged"]:
            raise RuntimeError("Plot input/source integrity check failed")
        manifest["artifacts_sha256"] = {p.name: sha256(p) for p in output.iterdir()
                                        if p.suffix in (".png", ".npz")}
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_unix"] = time.time()
        write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation", help="Optional complete evaluator output directory for A/C and GT comparison")
    arguments = parser.parse_args()
    run(arguments)
