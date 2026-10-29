#!/usr/bin/env python3
"""Offline scientific overlays for the four predeclared moving-robot checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from check_robot_projection import project_world, world_landmarks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); analysis_path = args.analysis.resolve(strict=True)
    analysis = json.loads(analysis_path.read_text()); output = args.output.resolve()
    if output.exists() or not output.is_relative_to(BASE / "validation"):
        parser.error("Use a fresh task validation directory")
    if (analysis["schema"] != "robodojo-native-joint-probe-analysis-1"
            or analysis["fixed_projection_checkpoint_indices"] != [5, 14, 23, 32]
            or analysis["kinematics_geometry_passed"] is not True):
        raise ValueError("Passing source-bound fixed-checkpoint analysis required")
    for row in analysis["source_files"]:
        if sha(row["path"]) != row["sha256"]: raise ValueError("Recorded source changed")
    geometry_path = Path(analysis["geometry_candidate_path"])
    if sha(geometry_path) != analysis["geometry_candidate_sha256"]: raise ValueError("Physical candidate changed")
    geometry = json.loads(geometry_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output/"plot_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    sources = {str(analysis_path): sha(analysis_path), str(geometry_path): sha(geometry_path)}
    records, figures = [], []
    colors = {"left_arm": "#e600ff", "right_arm": "#00ffff"}
    for checkpoint in analysis["checkpoint_inputs"]:
        for spec in [checkpoint["robot_measurement"], checkpoint["camera_inventory"], *checkpoint["images"]]:
            if sha(spec["path"]) != spec["sha256"]: raise ValueError("Checkpoint input changed")
            sources[spec["path"]] = spec["sha256"]
        state = json.loads(Path(checkpoint["robot_measurement"]["path"]).read_text())
        camera_inventory = json.loads(Path(checkpoint["camera_inventory"]["path"]).read_text())
        landmarks = world_landmarks(state["raw_robot_inventory"], geometry)
        image_paths = {row["camera"]: Path(row["path"]) for row in checkpoint["images"]}
        fig, axes = plt.subplots(3, 2, figsize=(12.8, 15.3), dpi=160)
        projections = {}
        for camera_index, name in enumerate(("head", "left_wrist", "right_wrist")):
            camera = camera_inventory["cameras"][name]
            rgb = np.asarray(Image.open(image_paths[camera["native_name"]]).convert("RGB"))
            if rgb.shape != (480, 640, 3): raise ValueError("Native RGB shape mismatch")
            k, view, world = (np.asarray(camera[key]) for key in ("intrinsics", "optical_from_world", "world_from_optical"))
            if not np.allclose(view@world, np.eye(4), atol=1e-6, rtol=0): raise ValueError("Optical transforms disagree")
            a, b = axes[camera_index]
            for ax in (a, b):
                ax.imshow(rgb, interpolation="nearest", origin="upper")
                ax.set(xlim=(-.5, 639.5), ylim=(479.5, -.5), xlabel="native u (pixels)", ylabel="native v (pixels)")
            a.set_title(name+": original unchanged RGB")
            b.set_title(name+": source geometry projected from recorded poses")
            projected_arms = {}
            for arm, robot in landmarks.items():
                pads = {}
                for link, pad in robot["pads"].items():
                    vertices = project_world(pad["polygon_world_m"], k, view)
                    centroid = project_world([pad["centroid_world_m"]], k, view)[0]
                    facing = bool(np.dot(pad["inward_normal_world"], world[:3, 3]-pad["centroid_world_m"]) > 0)
                    pads[link] = {"vertices": vertices, "centroid": centroid,
                                  "inward_face_toward_camera": facing, "occlusion_assessed": False}
                    if all(v["positive_depth"] and v["uv_pixels"] is not None for v in vertices):
                        uv = np.array([v["uv_pixels"] for v in vertices]); uv = np.r_[uv, uv[:1]]
                        b.plot(uv[:, 0], uv[:, 1], color=colors[arm], linewidth=1.0, linestyle="-" if facing else "--")
                    if centroid["inside_native_image"]:
                        b.plot(*centroid["uv_pixels"], marker="+", markersize=7, color=colors[arm])
                tool = project_world([robot["nominal_tool_point_world_m"]], k, view)[0]
                if tool["inside_native_image"]: b.plot(*tool["uv_pixels"], marker="x", markersize=7, color=colors[arm])
                projected_arms[arm] = {"pads": pads, "nominal_tool": tool}
            projections[name] = projected_arms
        fig.suptitle(f"Fixed checkpoint {checkpoint['index']} — actual measured robot poses\nMagenta:left; cyan:right; + pad centroid; x nominal tool; dashed inward faces point away\nOperator-only near-home consistency analysis; no depth/occlusion proof or full calibration", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        path = output/f"checkpoint_{checkpoint['index']:03d}_projection.png"
        fig.savefig(path); plt.close(fig)
        figures.append({"index": checkpoint["index"], "path": str(path), "sha256": sha(path)})
        records.append({"index": checkpoint["index"], "native_physics_time_s": state["native_physics_time_s"],
                        "world_landmarks": landmarks, "camera_projections": projections})
    for path, expected in sources.items():
        if sha(path) != expected: raise ValueError("Original input changed during analysis")
    report = {"schema": "robodojo-joint-probe-projection-analysis-1", "scope": "operator_only_four_recorded_near_home_robot_poses",
        "kinematics_analysis": {"path": str(analysis_path), "sha256": sha(analysis_path)},
        "source_files": [{"path": path, "sha256": expected} for path, expected in sorted(sources.items())],
        "script_sha256": sha(Path(__file__)), "projection_helper_sha256": sha(BASE/"scripts/check_robot_projection.py"),
        "fixed_checkpoint_indices": [5, 14, 23, 32], "original_rgb_unchanged": True,
        "figures": figures, "checkpoints": records, "visual_review_required": True,
        "model_input_eligible": False, "full_test_authorized": False, "standard_ready": False,
        "full_calibration_passed": False, "simulator_starts": 0, "model_calls": 0}
    (output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "figures": len(figures), "analysis_sha256": sha(output/"analysis.json")}))


if __name__ == "__main__":
    main()
