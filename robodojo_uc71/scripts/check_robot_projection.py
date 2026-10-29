#!/usr/bin/env python3
"""Offline operator-only robot landmark projections; never launch a simulator.

Original RGB and native assets remain unchanged. Matplotlib figures are separate
scientific annotations, not model input. Geometry agreement alone does not claim
image alignment; a subsequent visual review must distinguish occluded surfaces.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[1]
USD_INSPECTION = r'''
from pxr import Usd,UsdGeom,Gf
import json,sys
s=Usd.Stage.Open(sys.argv[1]);cache=UsdGeom.XformCache();result={}
if UsdGeom.GetStageMetersPerUnit(s)!=1 or UsdGeom.GetStageUpAxis(s)!="Z":
    raise ValueError("Unexpected robot asset units/axis")
for name,sign in (("link7",-1),("link8",1)):
    root=s.GetPrimAtPath('/X5A/'+name);inv=cache.GetLocalToWorldTransform(root).GetInverse();found=[]
    for p in Usd.PrimRange(root):
        if p.GetTypeName()!='Mesh' or '/visuals/' not in str(p.GetPath()):continue
        m=UsdGeom.Mesh(p);T=cache.GetLocalToWorldTransform(p)*inv
        points=[T.Transform(Gf.Vec3d(v)) for v in m.GetPointsAttr().Get()]
        ids=m.GetFaceVertexIndicesAttr().Get();counts=m.GetFaceVertexCountsAttr().Get();faces=[];at=0
        for index,count in enumerate(counts):
            face=[points[k] for k in ids[at:at+count]];at+=count
            if count!=3:raise ValueError("Nontriangular robot visual face")
            n=Gf.Cross(face[1]-face[0],face[2]-face[0]);area=n.GetLength()/2
            if area==0:continue
            n/=2*area;c=(face[0]+face[1]+face[2])/3
            if n[1]*sign>1-1e-6 and c[0]>.04:
                faces.append({'face_index':index,'vertices':[list(v) for v in face],
                              'normal':list(n),'area_m2':area,'centroid':list(c)})
        found.append({'prim':str(p.GetPath()),'orientation':m.GetOrientationAttr().Get(),'pad_faces':faces})
    result[name]=found
print(json.dumps(result))
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid finite recorded geometry")
    return result


def rotation(q):
    q = array(q, (4,)); norm = np.linalg.norm(q)
    if abs(norm-1) > 1e-5:
        raise ValueError("Nonunit recorded robot orientation")
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def project_world(points, k, optical_from_world):
    """Column-vector world→optical→pinhole, matching Camera's installed API."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 world points")
    optical = (array(optical_from_world, (4, 4)) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    homogeneous = (array(k, (3, 3)) @ optical.T).T
    result = []
    for xyz, uvz in zip(optical, homogeneous):
        uv = (uvz[:2]/uvz[2]).tolist() if abs(uvz[2]) > 1e-12 else None
        front = bool(xyz[2] > 0)
        inside = bool(front and uv is not None and -.5 <= uv[0] < 639.5 and -.5 <= uv[1] < 479.5)
        result.append({"optical_xyz_m": xyz.tolist(), "uv_pixels": uv,
                       "positive_depth": front, "inside_native_image": inside})
    return result


def bound_child(root, spec):
    path = root / spec["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root) or digest(path) != spec["sha256"]:
        raise ValueError("Preview artifact hash/path mismatch")
    return path


def verify_native_pad_faces(candidate, usd_python):
    sources = {Path(row["path"]).name: Path(row["path"]) for row in candidate["source_files"]}
    inspected = json.loads(subprocess.check_output([str(usd_python), "-I", "-B", "-c", USD_INSPECTION,
        str(sources["ARX.usd"])], text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))
    for link, pad in candidate["pad_geometry"].items():
        meshes = inspected[link]
        if len(meshes) != 1 or meshes[0]["orientation"] != "rightHanded" or meshes[0]["prim"] != pad["native_visual_prim"]:
            raise ValueError("Native finger visual identity/winding changed")
        faces = meshes[0]["pad_faces"]
        if len(faces) != 2 or [f["face_index"] for f in faces] != pad["triangle_indices"]:
            raise ValueError("Original USD opposed-pad topology differs from the candidate")
        areas = np.array([f["area_m2"] for f in faces])
        centroid = np.average([f["centroid"] for f in faces], axis=0, weights=areas)
        vertices = np.unique(np.array([f["vertices"] for f in faces]).reshape(-1, 3), axis=0)
        if (not np.isclose(areas.sum(), pad["area_m2"], rtol=0, atol=1e-12)
                or not np.allclose(centroid, pad["centroid_in_finger_link_m"], rtol=0, atol=1e-10)
                or not np.allclose(vertices, pad["pad_vertices_in_finger_link_m"], rtol=0, atol=1e-10)):
            raise ValueError("Native USD pad centroid/vertices differ from the STL-derived candidate")
    return inspected


def world_landmarks(robots, candidate):
    output = {}
    for robot in robots["robots"]:
        arm = robot["arm_name"]
        if arm not in ("left_arm", "right_arm") or arm in output:
            raise ValueError("Expected unique native arm identities")
        names = robot["body_names"]
        poses = array(robot["body_link_pose_w"]["values"], (1, len(names), 7))[0]
        pads = {}
        for link, pad in candidate["pad_geometry"].items():
            pose = poses[names.index(link)]; r = rotation(pose[3:])
            local = array(pad["pad_vertices_in_finger_link_m"], (4, 3))
            center = array(pad["centroid_in_finger_link_m"], (3,))
            order = np.argsort(np.arctan2(local[:, 2]-center[2], local[:, 0]-center[0]))
            world = (r @ local[order].T).T + pose[:3]
            normal = r @ np.array([0., -1. if link == "link7" else 1., 0.])
            pads[link] = {"polygon_world_m": world.tolist(), "centroid_world_m": (pose[:3]+r@center).tolist(),
                          "inward_normal_world": normal.tolist(), "body_pose_wxyz": pose.tolist()}
        ee = poses[names.index("link6")]
        tool = array(candidate["link6_from_tool"], (4, 4))
        point = ee[:3] + rotation(ee[3:]) @ tool[:3, 3]
        midpoint = np.mean([row["centroid_world_m"] for row in pads.values()], axis=0)
        output[arm] = {"pads": pads, "nominal_tool_point_world_m": point.tolist(),
            "measured_pad_centroid_midpoint_world_m": midpoint.tolist(),
            "nominal_minus_measured_midpoint_m": (point-midpoint).tolist()}
    if set(output) != {"left_arm", "right_arm"}:
        raise ValueError("Both measured native arms required")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview-root", type=Path, required=True)
    p.add_argument("--tool-candidate", type=Path, required=True)
    p.add_argument("--usd-python", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(BASE / "validation"):
        p.error("Use a fresh task validation directory")
    preview_root = args.preview_root.resolve(strict=True)
    parent_path = preview_root / "launch_result.json"; parent = read(parent_path)
    preview_path = preview_root / "preview/preview_result.json"; preview = read(preview_path); child = preview_path.parent
    if (parent.get("diagnostic_render_passed") is not True or parent.get("returncode") != 0
            or parent.get("termination_signals") or parent.get("preview_result_sha256") != digest(preview_path)
            or preview.get("model_calls") != 0 or preview.get("benchmark_actions") != 0
            or preview.get("full_test_authorized") is not False):
        raise ValueError("A completed clean model-free preview is required")
    records = {}
    for name in ("robot_inventory", "camera_inventory"):
        if preview[name]["status"] != "captured":
            raise ValueError("Both optional inventories must have been captured")
        records[name] = bound_child(child, preview[name])
    robots, cameras = read(records["robot_inventory"]), read(records["camera_inventory"])
    if (robots["schema"] != "robodojo-native-robot-inventory-1"
            or cameras["schema"] != "robodojo-native-camera-inventory-1"
            or cameras["native_physics_time_s"] != preview["capture_physics_time_after_s"]
            or preview["capture_physics_time_before_s"] != preview["capture_physics_time_after_s"]):
        raise ValueError("Camera/robot inventory or synchronized clock contract differs")
    candidate_path = args.tool_candidate.resolve(strict=True); candidate = read(candidate_path)
    if (candidate["schema"] != "robodojo-physical-tool-candidate-1"
            or candidate["calibration_status"] != "uncalibrated_candidate"
            or candidate["model_input_eligible"] is not False or candidate["full_test_authorized"] is not False
            or candidate["script_sha256"] != digest(BASE / "scripts/derive_tool_candidate.py")
            or candidate["inventory_sha256"] != digest(candidate["inventory_path"])
            or candidate["usd_vertices_sha256"] != digest(candidate_path.parent / "usd_finger_vertices.json")):
        raise ValueError("Tool candidate provenance/status changed")
    source_rows = candidate["source_files"]
    for row in source_rows:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Original native robot source changed")
    # Preserve the virtual-environment launcher path; resolving its interpreter
    # symlink would select the base Python and lose the usd-core environment.
    usd_python = args.usd_python.absolute()
    if not usd_python.is_file():
        raise ValueError("Explicit plain-USD Python executable is absent")
    usd_topology = verify_native_pad_faces(candidate, usd_python)
    landmarks = world_landmarks(robots, candidate)
    image_paths = {row["camera"]: bound_child(child, row) for row in preview["images"]}
    if set(image_paths) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
        raise ValueError("Exactly three original RGB images required")
    original_hashes = {str(path): digest(path) for path in image_paths.values()}
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / "plot_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    projections, figures = {}, []
    colors = {"left_arm": "#e600ff", "right_arm": "#00ffff"}
    for name, camera in cameras["cameras"].items():
        rgb = np.asarray(Image.open(image_paths[camera["native_name"]]).convert("RGB"))
        if rgb.shape != (480, 640, 3):
            raise ValueError("Original image dimensions differ from the camera contract")
        k = array(camera["intrinsics"], (3, 3)); view = array(camera["optical_from_world"], (4, 4))
        world = array(camera["world_from_optical"], (4, 4))
        if not np.allclose(view@world, np.eye(4), atol=1e-6, rtol=0):
            raise ValueError("Recorded camera transforms are not inverses")
        records_for_camera = {}
        for arm, body in landmarks.items():
            pads = {}
            for link, pad in body["pads"].items():
                center = np.array(pad["centroid_world_m"])
                pads[link] = {"vertices": project_world(pad["polygon_world_m"], k, view),
                    "centroid": project_world([center], k, view)[0],
                    "inward_face_toward_camera": bool(np.dot(pad["inward_normal_world"], world[:3, 3]-center) > 0),
                    "occlusion_assessed": False}
            records_for_camera[arm] = {"pads": pads,
                "nominal_tool": project_world([body["nominal_tool_point_world_m"]], k, view)[0]}
        projections[name] = records_for_camera

        def annotate(ax, only=None):
            for arm, body in records_for_camera.items():
                for link, pad in body["pads"].items():
                    if only is not None and only != (arm, link): continue
                    if all(v["positive_depth"] and v["uv_pixels"] is not None for v in pad["vertices"]):
                        uv = np.array([v["uv_pixels"] for v in pad["vertices"]]); uv = np.r_[uv, uv[:1]]
                        ax.plot(uv[:, 0], uv[:, 1], color=colors[arm], linewidth=1.1,
                                linestyle="-" if pad["inward_face_toward_camera"] else "--")
                    c = pad["centroid"]
                    if c["inside_native_image"]:
                        ax.plot(*c["uv_pixels"], marker="+", markersize=8, color=colors[arm], markeredgewidth=1.2)
                        if only is None:
                            ax.annotate(arm.replace("_arm", "")+" "+link, c["uv_pixels"], xytext=(4, 7),
                                        textcoords="offset points", fontsize=7, color=colors[arm])
                tool = body["nominal_tool"]
                if only is None and tool["inside_native_image"]:
                    ax.plot(*tool["uv_pixels"], marker="x", markersize=8, color=colors[arm], markeredgewidth=1.3)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=150)
        for ax in axes:
            ax.imshow(rgb, origin="upper", interpolation="nearest")
            ax.set(xlim=(-.5, 639.5), ylim=(479.5, -.5), xlabel="u (native pixels)", ylabel="v (native pixels)")
        axes[0].set_title("Original RGB (unchanged source)")
        axes[1].set_title("Projected pads (+centroid), nominal tool (x)")
        annotate(axes[1])
        fig.suptitle(name+" — magenta:left; cyan:right; dashed: inward face points away\nOperator analysis only; occlusion/alignment require visual review", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92))
        path = output / (name+"_projection.png"); fig.savefig(path); plt.close(fig)
        figures.append({"path": path.name, "sha256": digest(path), "kind": "full_original_and_projection"})
        for arm, body in records_for_camera.items():
            for link, pad in body["pads"].items():
                if not pad["centroid"]["inside_native_image"]: continue
                uv = np.array([v["uv_pixels"] for v in pad["vertices"]])
                middle = np.mean(uv, axis=0); half = np.maximum((uv.max(axis=0)-uv.min(axis=0))/2+14, 32)
                low, high = np.maximum(middle-half, [-.5, -.5]), np.minimum(middle+half, [639.5, 479.5])
                fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=160)
                for ax in axes:
                    ax.imshow(rgb, origin="upper", interpolation="nearest")
                    ax.set(xlim=(low[0], high[0]), ylim=(high[1], low[1]), xlabel="native u", ylabel="native v")
                axes[0].set_title("Original pixels"); axes[1].set_title("Projected inner-pad outline")
                annotate(axes[1], (arm, link))
                fig.suptitle(f"{name}: {arm} {link}; inward face toward camera={pad['inward_face_toward_camera']}\nA visible outline is not evidence that an occluded inner face is visible", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, .91))
                path = output / f"{name}_{arm}_{link}_zoom.png"; fig.savefig(path); plt.close(fig)
                figures.append({"path": path.name, "sha256": digest(path), "kind": "original_and_projection_zoom"})
    if any(digest(path) != sha for path, sha in original_hashes.items()):
        raise ValueError("Original RGB changed during offline analysis")
    source_paths = [parent_path, preview_path, candidate_path, *records.values(), *image_paths.values()]
    report = {"schema": "robodojo-robot-projection-analysis-1", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "operator_only_recorded_robot_geometry_and_rgb", "model_input_eligible": False,
        "standard_ready": False, "calibration_status": "uncalibrated_candidate",
        "rendered_landmarks_validated": False, "visual_review_required": True,
        "live_multi_pose_validation_passed": False, "full_test_authorized": False,
        "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
        "original_rgb_unchanged": True, "projection_convention": "uv=K*(optical_from_world*[world_xyz,1]) then divide by optical z; image origin upper left",
        "source_files": [{"path": str(p), "sha256": digest(p)} for p in source_paths],
        "original_robot_sources": source_rows, "script_sha256": digest(Path(__file__)),
        "usd_inspection_code_sha256": hashlib.sha256(USD_INSPECTION.encode()).hexdigest(),
        "usd_topology_crosscheck": usd_topology, "world_landmarks": landmarks,
        "projections": projections, "figures": figures,
        "limitations": ["No renderer depth/segmentation or scene-object truth accessed",
            "Positive depth, in-frame coordinates and face orientation do not prove visibility or pixel alignment",
            "Nominal tool point is a virtual geometry-defined point, not a directly visible marker",
            "No live perturbation, actuator tracking or multi-pose calibration performed"]}
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "figures": len(figures), "original_rgb_unchanged": True,
                      "visual_review_required": True, "standard_ready": False}))


if __name__ == "__main__":
    main()
