"""Plot actual Astra field blocks, saved-only decoded meshes, and frozen scores.

No geometry is inferred or regenerated for these figures. A saved scalar slice
and every triangle of the saved mesh are shown. The older RGB experiment's
triangle rasterizer is a read-only display dependency, never a geometry source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RASTER_SOURCE = HERE.parent / "rgb_scene" / "plot_rgb_results.py"
NAMES = ("bowl002_initial", "drawer003_closed_step60", "drawer003_open_step552")
LABELS = dict(zip(NAMES, ("Bowl / initial", "Drawer / closed", "Drawer / open")))
DEPTH_LIMITS_M = (0., 4.)
ERROR_LIMITS_M = (0., .5)
FIELD_LIMIT_MM = 20.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(root, record, consumed):
    path = (Path(root) / record["file"]).resolve()
    actual = sha(path)
    if actual != record["sha256"]:
        raise ValueError(f"Artifact SHA mismatch: {path}")
    consumed[str(path)] = actual
    return path


def saved_field_slice(build, config, snapshot, consumed):
    """Nearest saved y plane to zero inside the model AABB; no field evaluation."""
    allocation = config["allocation"]
    lower = np.asarray(allocation["grid_lower_m"], float)
    cells = np.asarray(allocation["grid_cells"], int)
    h = float(config["spacing_m"])
    geo_lower, geo_upper = np.asarray(snapshot["geometry_lower_m"]), np.asarray(snapshot["geometry_upper_m"])
    chosen_y = float(np.clip(0., geo_lower[1], geo_upper[1]))
    yindex = int(np.clip(np.rint((chosen_y-lower[1])/h), 0, cells[1]))
    plane_y = float(lower[1] + yindex*h)
    values = np.full((cells[0]+1, cells[2]+1), np.nan, dtype=np.float32)
    loaded, overlap_max = [], 0.
    for record in config["fields"]:
        block_lower = np.asarray(record["lower_m"], float)
        shape = np.asarray(record["shape"], int)
        coordinate = (plane_y-block_lower[1])/h
        yi = int(round(coordinate))
        if yi < 0 or yi >= shape[1]:
            continue
        if abs(coordinate-yi) > 1e-7:
            raise ValueError("Selected slice is not on a saved grid plane")
        path = checked(build, record, consumed)
        with np.load(path, allow_pickle=False) as saved:
            actual_lower = saved["lower_m"].astype(float)
            actual_h = float(saved["spacing_m"])
            if not np.array_equal(actual_lower, block_lower) or actual_h != h:
                raise ValueError("Saved block transform differs from its manifest")
            if not np.array_equal(saved["values"].shape, shape):
                raise ValueError("Saved block shape differs from its manifest")
            plane = saved["values"][:, yi, :].copy()
        starts = np.rint((block_lower-lower)/h).astype(int)
        ix, iz = starts[[0, 2]]
        target = values[ix:ix+shape[0], iz:iz+shape[2]]
        if target.shape != plane.shape:
            raise ValueError("Saved slice extends outside declared grid")
        overlap = np.isfinite(target)
        if overlap.any():
            error = float(np.max(np.abs(target[overlap]-plane[overlap])))
            overlap_max = max(overlap_max, error)
            if error > 2e-6:
                raise ValueError("Shared field vertices disagree by over 2 micrometres")
        target[:] = plane
        loaded.append(record)
    upper = lower + cells*h
    return values, {"camera_y_m": plane_y, "requested_y_m": chosen_y, "spacing_m": h,
                    "lower_xz_m": lower[[0, 2]].tolist(), "upper_xz_m": upper[[0, 2]].tolist(),
                    "slice_rule": "nearest saved grid plane to y=0 clamped to model geometry bounds",
                    "source": "actual saved numeric blocks only; no analytic field calls",
                    "gray_meaning": "no block saved here (certified no zero surface), not unknown space",
                    "display_saturation_mm": [-FIELD_LIMIT_MM, FIELD_LIMIT_MM],
                    "stored_plane_vertex_fraction": float(np.isfinite(values).mean()),
                    "overlap_max_difference_m": overlap_max,
                    "loaded_blocks": [{"file": r["file"], "sha256": r["sha256"]} for r in loaded]}


def mesh_projection(vertices, bounds, size=800):
    """Same model-derived orthographic view and scale for all three scene meshes."""
    vertices = np.asarray(vertices, float)
    low, high = (np.asarray(x, float) for x in bounds)
    world = vertices[:, [0, 2, 1]].copy(); world[:, 2] *= -1
    box = np.array([[x, z, -y] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    elevation, azimuth = np.deg2rad([18., -65.])
    eye = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    up = np.cross(eye, right)
    camera = np.column_stack((right, up))
    box_uv = box@camera
    center = .5*(box_uv.min(0)+box_uv.max(0))
    scale = (size-40)/np.max(np.ptp(box_uv, axis=0))
    projected = (world@camera-center)*scale+(size-1)/2
    projected[:, 1] = size-1-projected[:, 1]
    return projected, world@eye, world


def draw_mesh(ax, path, bounds, rasterizer):
    with np.load(path, allow_pickle=False) as saved:
        vertices, faces = saved["vertices_m"].copy(), saved["faces"].copy()
    if not len(faces):
        raise ValueError("Cannot display an empty mesh")
    projected, distance, world = mesh_projection(vertices, bounds)
    if np.any(projected < 0) or np.any(projected > 799):
        raise ValueError("Shared model-derived projection unexpectedly clips mesh vertices")
    lower, upper = bounds
    # The read-only rasterizer's palette indexing expects .4..1.6. Remap only
    # the independent COLOR attribute to those values; geometry is untouched.
    palette_coordinate = .4 + 1.2*(vertices[:, 2]-lower[2])/(upper[2]-lower[2])
    palette = 255*plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]
    rendered, zbuffer, drawn = rasterizer(projected, distance, palette_coordinate, world, faces, palette, 800)
    ax.imshow(rendered); ax.axis("off")
    ax.text(.01, .02, f"{len(faces):,} saved triangles; all submitted", transform=ax.transAxes, fontsize=8, color="#444444")
    return {"saved_vertices": len(vertices), "saved_faces": len(faces), "submitted_faces": len(faces),
            "rasterized_faces_writing_pixels": int(drawn), "display_geometry": "actual decoded mesh; no simplification",
            "rendered_pixel_fraction": float(np.isfinite(zbuffer).mean()),
            "shared_bounds_m": [np.asarray(x).tolist() for x in bounds],
            "shared_camera_z_color_limits_m": [float(lower[2]), float(upper[2])],
            "mesh_vertices_clipped_by_view": 0, "azimuth_deg": -65., "elevation_deg": 18.}


def error_image(decoded, gt, valid):
    hit = valid & np.isfinite(decoded) & (decoded > 0)
    error = np.abs(decoded-gt)
    masked = np.ma.masked_where(~hit, error)
    overlay = np.zeros((*valid.shape, 4))
    overlay[valid & ~hit] = [0., .75, .9, 1.]
    return masked, overlay, hit


def comparison_figure(output, scenes, configs, rgbs, evaluation, consumed):
    fig, axes = plt.subplots(3, 5, figsize=(20, 12), constrained_layout=True)
    report = []
    for index, name in enumerate(NAMES):
        source = checked(evaluation, scenes[name]["arrays"], consumed)
        decoded_source = checked(evaluation, configs[name]["arrays"], consumed)
        with np.load(source, allow_pickle=False) as saved:
            gt, original, valid = saved["gt_depth_m"], saved["analytic_depth_m"], saved["valid_depth"].astype(bool)
        with np.load(decoded_source, allow_pickle=False) as saved:
            decoded = saved["decoded_depth_m"]
        with Image.open(rgbs[name]) as rgb:
            axes[index, 0].imshow(rgb)
        axes[index, 0].set_title(f"{LABELS[name]}\nSingle RGB input")
        depth_cmap = plt.get_cmap("viridis").copy(); depth_cmap.set_bad("#00bfdc")
        for column, values, title in ((1, gt, "Reference: exact visible geometry"),
                                      (2, original, "A: raw Astra analytic field"),
                                      (3, decoded, f"C: saved-field mesh ({configs[name]['spacing_m']*1000:g} mm)")):
            mask = valid & np.isfinite(values) & (values > 0)
            depth_view = axes[index, column].imshow(np.ma.masked_where(~mask, values), cmap=depth_cmap,
                                                    vmin=DEPTH_LIMITS_M[0], vmax=DEPTH_LIMITS_M[1])
            invalid = np.zeros((*valid.shape, 4)); invalid[~valid] = [.83, .83, .83, 1.]
            axes[index, column].imshow(invalid)
            axes[index, column].set_title(title, fontsize=10)
        masked_error, missing, hit = error_image(decoded, gt, valid)
        error_cmap = plt.get_cmap("magma").copy(); error_cmap.set_bad("#d3d3d3")
        error_view = axes[index, 4].imshow(masked_error, cmap=error_cmap, vmin=ERROR_LIMITS_M[0], vmax=ERROR_LIMITS_M[1])
        axes[index, 4].imshow(missing)
        axes[index, 4].set_title(f"|C - reference|; cyan = miss\n{int((valid & ~hit).sum()):,} / {int(valid.sum()):,} reference rays missed", fontsize=10)
        for ax in axes[index]:
            ax.axis("off")
        fig.colorbar(depth_view, ax=axes[index, 1:4].tolist(), shrink=.55, label="Camera Z (m), display clipped to 0–4")
        fig.colorbar(error_view, ax=axes[index, 4], shrink=.55, label="Z error (m), display capped at 0.5")
        report.append({"name": name, "reference_and_analytic_arrays": scenes[name]["arrays"],
                       "decoded_arrays": configs[name]["arrays"], "reference_pixels": int(valid.sum()),
                       "decoded_hit_pixels": int(hit.sum()), "decoded_misses": int((valid & ~hit).sum()),
                       "decoded_hit_error_above_display_cap": int(np.sum(masked_error.compressed() > ERROR_LIMITS_M[1])),
                       "gt_depth_above_display_cap": int(np.sum(gt[valid] > DEPTH_LIMITS_M[1]))})
    fig.suptitle("Astra image → implicit field → saved scalar grid → decoded geometry\n"
                 "A / C use identical GT-camera rays; native scale, no alignment. Shared depth colors; cyan marks missing geometry.", fontsize=14)
    fig.savefig(output/"reference_comparison.png", dpi=170); plt.close(fig)
    return report


def metric_figure(output, metrics):
    scenes = {s["name"]: s for s in metrics["snapshots"]}
    fig, axes = plt.subplots(5, 3, figsize=(13, 17), constrained_layout=True)
    for index, name in enumerate(NAMES):
        region = "target_bowl_visible" if name == NAMES[0] else "target_cabinet_visible"
        A = scenes[name]["regions"][region]
        configs = sorted((c for c in metrics["configurations"] if c.get("snapshot") == name and c["status"] == "ok"), key=lambda c: c["spacing_m"], reverse=True)
        h = np.array([c["spacing_m"]*1000 for c in configs])
        C = [c["regions"][region] for c in configs]
        B = [c["representation_B_rays"] for c in configs]
        axes[0, index].axhline(A["coverage"]*100, color="#777777", linestyle="--", label="A: raw analytic")
        axes[0, index].plot(h, [c["coverage"]*100 for c in C], "o-", color="#8c3db3", label="C: saved-field mesh")
        axes[0, index].set(title=f"{LABELS[name]} / {'bowl' if index == 0 else 'cabinet'} pixels", ylabel="Target hit coverage (%)", ylim=(-2, 102))
        for key, label, style in (("mae_on_hits_m", "MAE", "-"), ("p95_on_hits_m", "P95", ":")):
            axes[1, index].axhline(A[key]*1000 if A[key] is not None else np.nan, color="#777777", linestyle=style, label=f"A {label}")
            axes[1, index].plot(h, [(c[key]*1000 if c[key] is not None else np.nan) for c in C], marker="o", linestyle=style, color="#8c3db3", label=f"C {label}")
            axes[3, index].plot(h, [(b[key]*1000 if b[key] is not None else np.nan) for b in B], marker="o", linestyle=style, label=f"B {label}")
        axes[1, index].set(ylabel="Target Z error on hits (mm)")
        axes[2, index].axhline(A["within_threshold_all_reference_pixels"]["5mm"]*100, color="#777777", linestyle="--", label="A: raw analytic")
        axes[2, index].plot(h, [c["within_threshold_all_reference_pixels"]["5mm"]*100 for c in C], "o-", color="#8c3db3", label="C: saved-field mesh")
        axes[2, index].set(ylabel="All target pixels within 5 mm (%)", ylim=(-2, 102))
        axes[3, index].set(ylabel="B: six-view Z error on hits (mm)")
        axes[4, index].plot(h, [b["coverage"]*100 for b in B], "o-", label="Retained analytic hits")
        axes[4, index].plot(h, [b["new_hit_fraction_all_test_rays"]*100 for b in B], "s--", label="New hits / all test rays")
        axes[4, index].set(ylabel="B: hit coverage / extra hits (%)", ylim=(-2, 102))
        for ax in axes[:, index]:
            ax.set(xlabel="Voxel spacing (mm)", xticks=h, xlim=(float(h.max()+.4), float(h.min()-.4)))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Geometry estimation A / final reconstruction C versus field preservation B\n"
                 "A/C: visible target GT rays; misses fail coverage and within-5mm. B: six external model-derived views, first-hit only.\n"
                 "Errors are ray-axis distances, not whole-surface Hausdorff errors. Accurate B does not imply accurate scene geometry.", fontsize=13)
    fig.savefig(output/"accuracy_and_preservation.png", dpi=170); plt.close(fig)


def run(args):
    build, evaluation, output = (Path(getattr(args, key)).resolve() for key in ("fields", "evaluation", "output"))
    if output.exists():
        raise FileExistsError(f"Refusing existing analysis directory: {output}")
    consumed = {}
    bm_path, em_path = build/"manifest.json", evaluation/"manifest.json"
    bm, em = json.loads(bm_path.read_text()), json.loads(em_path.read_text())
    consumed.update({str(bm_path): sha(bm_path), str(em_path): sha(em_path)})
    if bm.get("status") != "complete" or em.get("status") != "complete" or Path(em["fields"]).resolve() != build:
        raise ValueError("Plot requires completed matching field and evaluation runs")
    if not em.get("source_unchanged") or not em.get("inputs_unchanged"):
        raise ValueError("Evaluator integrity failed")
    if bm.get("integrity", {}).get("settings") is not True:
        raise ValueError("Field settings integrity failed")
    for group in ("sources", "inputs", "artifacts"):
        records = bm.get("integrity", {}).get(group)
        if not records or any(r.get("unchanged") is not True for r in records):
            raise ValueError(f"Field integrity failed: {group}")
    metrics_path = checked(evaluation, {"file": "metrics.json", "sha256": em["metrics_sha256"]}, consumed)
    metrics = json.loads(metrics_path.read_text())
    if metrics["status"] != "complete":
        raise ValueError("Metrics are not complete")
    snapshots = {s["name"]: s for s in bm["snapshots"]}
    scenes = {s["name"]: s for s in metrics["snapshots"]}
    if set(snapshots) != set(NAMES) or set(scenes) != set(NAMES):
        raise ValueError("Unexpected snapshot set")
    spacings = [{c["spacing_m"] for c in bm["configurations"] if c["snapshot"] == name and c["status"] == "ok"} for name in NAMES]
    h = min(set.intersection(*spacings))
    configs = {c["snapshot"]: c for c in bm["configurations"] if c["spacing_m"] == h}
    evaluation_configs = {c["snapshot"]: c for c in metrics["configurations"] if c.get("spacing_m") == h}
    if set(configs) != set(NAMES) or set(evaluation_configs) != set(NAMES):
        raise ValueError("Missing common decoded resolution")
    inference_path = Path(bm["inference_manifest"])
    frozen = next(r for r in bm["inputs"] if r.get("copy") == "inputs/inference_manifest.json")
    checked(Path("/"), {"file": str(inference_path), "sha256": frozen["sha256"]}, consumed)
    inference = json.loads(inference_path.read_text())
    rgb_rows = {p["name"]: p["rgb"] for p in inference["predictions"]}
    rgbs = {name: checked(inference_path.parent, rgb_rows[name], consumed) for name in NAMES}
    bounds = (np.min([c["allocation"]["grid_lower_m"] for c in configs.values()], axis=0),
              np.max([c["allocation"]["grid_upper_m"] for c in configs.values()], axis=0))
    output.mkdir(parents=True, exist_ok=False); (output/"source").mkdir()
    sources = []
    for source in (Path(__file__).resolve(), RASTER_SOURCE):
        copy_path = output/"source"/source.name
        shutil.copy2(source, copy_path)
        sources.append({"path": str(source), "sha256": sha(source), "copy": str(copy_path.relative_to(output)), "copy_sha256": sha(copy_path)})
    module_spec = importlib.util.spec_from_file_location("astra_readonly_mesh_renderer", output/"source"/RASTER_SOURCE.name)
    renderer = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(renderer)
    manifest = {"status": "running", "fields": str(build), "evaluation": str(evaluation), "sources": sources,
                "spacing_m_in_pipeline": h, "scenes": [], "consumed_sha256": consumed,
                "depth_display_limits_m": DEPTH_LIMITS_M, "error_display_limits_m": ERROR_LIMITS_M,
                "started_unix": time.time(), "geometry_substitution": False}
    dump(output/"manifest.json", manifest)
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
        for index, name in enumerate(NAMES):
            with Image.open(rgbs[name]) as rgb:
                axes[index, 0].imshow(rgb)
            axes[index, 0].set_title(f"{LABELS[name]}\nSingle RGB input", fontsize=11); axes[index, 0].axis("off")
            values, slice_info = saved_field_slice(build, configs[name], snapshots[name], consumed)
            np.savez_compressed(output/f"{name}_saved_field_slice.npz", values_m=values,
                                lower_xz_m=slice_info["lower_xz_m"], spacing_m=h, camera_y_m=slice_info["camera_y_m"])
            cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#d3d3d3")
            lower, upper = slice_info["lower_xz_m"], slice_info["upper_xz_m"]
            view = axes[index, 1].imshow(np.ma.masked_invalid(values.T*1000), extent=(lower[0], upper[0], upper[1], lower[1]),
                                          origin="upper", aspect="equal", interpolation="nearest", cmap=cmap,
                                          vmin=-FIELD_LIMIT_MM, vmax=FIELD_LIMIT_MM)
            axes[index, 1].set(title=f"Actual saved implicit field / h={h*1000:g} mm\ny={slice_info['camera_y_m']:.3f} m; gray = no saved block",
                               xlabel="Camera x (m)", ylabel="Camera z (m)")
            fig.colorbar(view, ax=axes[index, 1], shrink=.55, label="CSG field (mm), display clipped ±20")
            mesh = checked(build, configs[name]["decoded_mesh"], consumed)
            mesh_info = draw_mesh(axes[index, 2], mesh, bounds, renderer.rasterize_triangles)
            axes[index, 2].set_title("Actual saved-field decoded mesh\nShared oblique projection and depth colors", fontsize=11)
            manifest["scenes"].append({"name": name, "field_slice": slice_info, "decoded_mesh": {**configs[name]["decoded_mesh"], **mesh_info}, "rgb": rgb_rows[name]})
            print(json.dumps({"plotted_scene": name}), flush=True)
        fig.suptitle("Single RGB → Astra implicit field → saved numeric field → decoded geometry\n"
                     "Field sign: negative = material. Gray blocks were not saved; this is not an unknown-space mask. CSG values are not a global Euclidean SDF.", fontsize=13)
        fig.savefig(output/"pipeline.png", dpi=175); plt.close(fig)
        manifest["comparison"] = comparison_figure(output, scenes, evaluation_configs, rgbs, evaluation, consumed)
        metric_figure(output, metrics)
        manifest["inputs_unchanged"] = {p: sha(p) == digest for p, digest in consumed.items()}
        manifest["source_unchanged"] = all(sha(r["path"]) == r["sha256"] and sha(output/r["copy"]) == r["copy_sha256"] for r in sources)
        if not all(manifest["inputs_unchanged"].values()) or not manifest["source_unchanged"]:
            raise ValueError("Plot inputs or source changed during rendering")
        manifest["artifacts_sha256"] = {p.name: sha(p) for p in output.iterdir() if p.suffix in (".png", ".npz")}
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["finished_unix"] = time.time(); dump(output/"manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
