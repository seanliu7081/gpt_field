"""New slide-only figures from immutable experiment evidence; no inference."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

OUT = Path(__file__).resolve().parent
SCENE = OUT.parent.parent
consumed = {}


def consume(path):
    path = Path(path).resolve()
    consumed[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def color_image(values, valid, vmin, vmax, cmap, prediction=False):
    hit = valid & np.isfinite(values) & (values > 0)
    rgba = plt.get_cmap(cmap)(np.clip((np.nan_to_num(values)-vmin)/(vmax-vmin), 0, 1))
    rgba[~valid] = [0.83, 0.83, 0.83, 1]
    if prediction:
        rgba[valid & ~hit] = [0., .75, .9, 1.]
    return Image.fromarray((rgba[:, :, :3]*255).round().astype(np.uint8))


def build():
    rows = []
    for state, name in (("closed", "drawer003_closed_step60"), ("open", "drawer003_open_step552")):
        prefix = SCENE/"astra_scene/runs/evaluation_001/arrays"
        with np.load(consume(prefix/f"{name}_reference_and_analytic.npz")) as saved:
            gt = saved["gt_depth_m"].copy()
            valid = saved["valid_depth"].astype(bool)
            target = saved["target_cabinet_visible"].astype(bool)
        with np.load(consume(prefix/f"{name}_h2mm.npz")) as saved:
            decoded = saved["decoded_depth_m"].copy()
        rgb = Image.open(consume(SCENE/f"astra_scene/inferences_001/inputs/{name}.png")).convert("RGB")
        hit = valid & np.isfinite(decoded) & (decoded > 0)
        error = np.abs(decoded-gt)
        rgba = plt.get_cmap("magma")(np.clip(np.nan_to_num(error)/.5, 0, 1))
        rgba[~valid] = [.83, .83, .83, 1]
        rgba[valid & ~hit] = [0., .75, .9, 1.]
        err = Image.fromarray((rgba[:, :, :3]*255).round().astype(np.uint8))
        images = [rgb, color_image(gt, valid, 0, 4, "viridis"), color_image(decoded, valid, 0, 4, "viridis", True), err]
        strip = Image.new("RGB", (512*4+12*3, 512), "white")
        for k, (kind, img) in enumerate(zip(("rgb", "gt_depth", "decoded_depth", "absolute_error"), images)):
            img.save(OUT/f"{state}_{kind}.png")
            strip.paste(img, (k*(512+12), 0))
        strip.save(OUT/f"{state}_comparison_strip.png")
        target_hit = target & hit
        rows.append({"snapshot": name, "panels": [f"{state}_{kind}.png" for kind in ("rgb", "gt_depth", "decoded_depth", "absolute_error")],
                     "valid_reference_pixels": int(valid.sum()), "decoded_misses": int((valid & ~hit).sum()),
                     "target_cabinet_pixels": int(target.sum()), "target_hit_pixels": int(target_hit.sum()),
                     "target_mae_on_hits_mm": float(error[target_hit].mean()*1000),
                     "error_above_display_cap": int((error[hit]>.5).sum())})
    for name, limits, cmap, label in (("depth_colorbar", (0., 4.), "viridis", "Camera Z (m)"),
                                      ("error_colorbar", (0., .5), "magma", "|Predicted Z - reference Z| (m)")):
        fig = plt.figure(figsize=(5.5, .6))
        ax = fig.add_axes([.03, .55, .94, .30])
        cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=Normalize(*limits), orientation="horizontal")
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
        fig.savefig(OUT/f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Posthoc native scene rendering at the first colliding verification sample.
    sys.path.insert(0, str(SCENE/"field_planning"))
    import mujoco
    from robot_geometry import GroundTruthEvaluator
    report = json.loads(consume(SCENE/"field_planning_v2/runs/failure_diagnostics_001/report.json").read_text())
    row = next(r for r in report["failure_rows"] if r["snapshot"] == "drawer003_open_step552"
               and r["case_id"] == "wrist_negative" and r["seed"] == 17)
    source = SCENE/f"rgb_scene/references_004/snapshots/{row['snapshot']}/evaluator"
    truth = GroundTruthEvaluator(source)
    consumed.update(truth.consumed_sha256)
    truth._set_q(row["q7"])
    expected_contact = row["contact_queries"][0]
    expected_pair = {expected_contact["geom1"], expected_contact["geom2"]}
    native_matches = []
    for item in truth.data.contact[:truth.data.ncon]:
        names = {mujoco.mj_id2name(truth.model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                 for g in (item.geom1, item.geom2)}
        if names == expected_pair:
            native_matches.append({"distance_m": float(item.dist), "position_world_m": item.pos.tolist()})
    assert native_matches, "Saved failing native pair did not recur at the exact archived q."
    match = min(native_matches, key=lambda c: abs(c["distance_m"]-expected_contact["native_contact_distance_m"]))
    distance_error = abs(match["distance_m"]-expected_contact["native_contact_distance_m"])
    position_error = float(np.linalg.norm(np.asarray(match["position_world_m"])-
                                         expected_contact["native_contact_position_world_m"]))
    assert distance_error < 1e-10 and position_error < 1e-10, (distance_error, position_error)
    truth.model.vis.global_.offwidth = 640
    truth.model.vis.global_.offheight = 640
    renderer = mujoco.Renderer(truth.model, height=600, width=600)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.sitegroup[:] = 0
    renderer.update_scene(truth.data, camera="agentview", scene_option=option)
    raw = Image.fromarray(renderer.render())
    renderer.close()
    raw.save(OUT/"collision_native_scene.png")
    marked = raw.copy()
    draw = ImageDraw.Draw(marked)
    calibration = json.loads(consume(source/"calibration.json").read_text())
    K = np.asarray(calibration["K"]).copy()
    scale = 600/calibration["width"]
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2]+.5)*scale-.5
    K[1, 2] = (K[1, 2]+.5)*scale-.5
    T = np.asarray(calibration["T_camera_cv_from_world"])
    contact = row["contact_queries"][0]
    point = np.asarray(contact["native_contact_position_world_m"]) @ T[:3, :3].T + T[:3, 3]
    pixel = K @ point
    x, y = pixel[:2]/pixel[2]
    draw.ellipse((x-16, y-16, x+16, y+16), outline="white", width=7)
    draw.ellipse((x-16, y-16, x+16, y+16), outline="#E43D30", width=4)
    draw.line((x+18, y-15, x+100, y-72), fill="white", width=7)
    draw.line((x+18, y-15, x+100, y-72), fill="#E43D30", width=4)
    marked.save(OUT/"collision_native_scene_marked.png")
    box = [max(0, int(x-85)), max(0, int(y-85)), min(600, int(x+85)), min(600, int(y+85))]
    marked.crop(box).resize((510, 510)).save(OUT/"collision_contact_zoom.png")
    contact_record = {"failure_row": row, "render_pixels": [600, 600], "contact_pixel_xy": [float(x), float(y)],
                      "zoom_crop_box": box, "camera": "native agentview", "q_mode": "saved exact first native collision q7; no dynamics",
                      "display_only": "Hide geomgroup0 duplicate collision shapes and sites; original visual scene remains unchanged",
                      "native_contact_recheck_distance_difference_m": distance_error,
                      "native_contact_recheck_position_difference_m": position_error,
                      "purpose": "Posthoc illustration only; no ground-truth input to planning"}
    (OUT/"collision_render_data.json").write_text(json.dumps(contact_record, indent=2)+"\n")

    preview = consume(SCENE/"field_planning_v2/video_001/previews/22_030.png")
    img = Image.open(preview)
    img.crop((24, 112, 624, 712)).save(OUT/"rrt_native_scene.png")
    img.crop((656, 112, 1256, 712)).save(OUT/"rrt_field_panel.png")
    metadata = {"status": "complete", "reconstruction_panels": rows,
                "depth_color_limits_m": [0, 4], "error_color_limits_m": [0, .5],
                "gray": "invalid reference ray", "cyan": "valid reference ray with no positive decoded hit",
                "camera_depth_semantics": "camera Z; identical saved reference-camera rays; native predicted scale, no alignment",
                "rrt_panels": {"source": str(preview), "snapshot": "drawer003_open_step552", "case_id": "base_negative", "seed": 17,
                               "scene_crop_box": [24,112,624,712], "field_crop_box": [656,112,1256,712],
                               "source_kind": "cropped saved planning video preview", "frame_index": 30},
                "collision": contact_record, "consumed_sha256": consumed}
    metadata["sources_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == s for p, s in consumed.items())
    (OUT/"evidence_figures_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"rows": rows, "contact_pixel_xy": [float(x), float(y)], "source_files": len(consumed),
                      "sources_unchanged": metadata["sources_unchanged"]}, indent=2))


if __name__ == "__main__":
    build()
