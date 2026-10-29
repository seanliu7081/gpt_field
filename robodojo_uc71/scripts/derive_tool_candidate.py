#!/usr/bin/env python3
"""Derive a robot-only pad-center candidate; never certify live calibration."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

BASE = Path(__file__).resolve().parents[1]
ROBOT = BASE / "runtime/native/RoboDojo/Assets/Robots/x5"
USD_INSPECTION = r'''
import json,sys
from pxr import Usd,UsdGeom,Gf
stage=Usd.Stage.Open(sys.argv[1]); cache=UsdGeom.XformCache(); result={}
for name in ("link7","link8"):
    link=stage.GetPrimAtPath(stage.GetDefaultPrim().GetPath().AppendChild(name))
    inverse=cache.GetLocalToWorldTransform(link).GetInverse(); meshes=[]
    for prim in Usd.PrimRange(link):
        if prim.GetTypeName()=="Mesh" and "/visuals/" in str(prim.GetPath()):
            matrix=cache.GetLocalToWorldTransform(prim)*inverse
            points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
            meshes.append({"prim":str(prim.GetPath()),"vertices_in_link_frame":
                [list(matrix.Transform(Gf.Vec3d(v))) for v in points]})
    result[name]=meshes
print(json.dumps(result))
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-5:
        raise ValueError("Invalid recorded actor quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--usd-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(BASE / "validation"):
        parser.error("Use a fresh task validation directory")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("schema") != "robodojo-native-robot-inventory-1" or inventory["num_envs"] != 1:
        raise ValueError("Native one-environment robot inventory required")
    tree = ET.parse(ROBOT / "X5A.urdf").getroot()
    joints = {j.attrib["name"]: j for j in tree.findall("joint")}
    usd = json.loads(subprocess.check_output([str(args.usd_python.absolute()), "-c", USD_INSPECTION,
        str(ROBOT / "ARX.usd")], text=True, timeout=60))
    pads, sources = {}, [ROBOT / "X5A.urdf", ROBOT / "ARX.usd", ROBOT / "robot_config.yml"]
    for link, joint_name, sign in (("link7", "joint7", -1), ("link8", "joint8", 1)):
        path = ROBOT / "meshes" / (link + ".STL")
        sources.append(path)
        mesh = trimesh.load_mesh(path, process=False)
        selected = np.flatnonzero((mesh.face_normals[:, 1] * sign > 1 - 1e-6)
                                  & (mesh.triangles_center[:, 0] > .04))
        if len(selected) != 2 or len(usd[link]) != 1:
            raise ValueError("Expected two triangles on each distal inward-facing pad and one native visual mesh")
        native_vertices = np.unique(np.round(usd[link][0]["vertices_in_link_frame"], 10), axis=0)
        stl_vertices = np.unique(np.round(mesh.vertices, 10), axis=0)
        if native_vertices.shape != stl_vertices.shape or not np.array_equal(native_vertices, stl_vertices):
            raise ValueError("Native USD visual vertices differ from the robot STL")
        joint = joints[joint_name]
        origin = np.array([float(n) for n in joint.find("origin").attrib["xyz"].split()])
        rpy = [float(n) for n in joint.find("origin").attrib["rpy"].split()]
        axis = np.array([float(n) for n in joint.find("axis").attrib["xyz"].split()])
        if (joint.attrib["type"] != "prismatic" or joint.find("parent").attrib["link"] != "link6"
                or rpy != [0., 0., 0.] or axis.tolist() != [0., -float(sign), 0.]):
            raise ValueError("Native finger kinematics differ from the reviewed symmetric translation")
        center = np.average(mesh.triangles_center[selected], axis=0, weights=mesh.area_faces[selected])
        pads[link] = {"joint_name": joint_name, "joint_origin_xyz": origin.tolist(),
            "joint_axis": axis.tolist(), "triangle_indices": selected.tolist(),
            "area_m2": float(mesh.area_faces[selected].sum()),
            "centroid_in_finger_link_m": center.tolist(),
            "centroid_in_link6_at_zero_joint_m": (origin + center).tolist(),
            "pad_vertices_in_finger_link_m": np.unique(mesh.triangles[selected].reshape(-1, 3), axis=0).tolist(),
            "native_visual_prim": usd[link][0]["prim"], "usd_stl_vertices_match": True}
    center = (np.array(pads["link7"]["centroid_in_link6_at_zero_joint_m"])
              + np.array(pads["link8"]["centroid_in_link6_at_zero_joint_m"])) / 2
    tool = np.eye(4)
    tool[:3, 3] = center
    measured = []
    if {r["arm_name"] for r in inventory["robots"]} != {"left_arm", "right_arm"}:
        raise ValueError("Both native arms are required")
    for robot in inventory["robots"]:
        poses = np.asarray(robot["body_link_pose_w"]["values"], dtype=float)[0]
        q = np.asarray(robot["joint_positions"]["values"], dtype=float)[0]
        reference = poses[robot["body_names"].index("link6")]
        world_rotation = rotation(reference[3:])
        for link, pad in pads.items():
            actual = poses[robot["body_names"].index(link)]
            expected = np.array(pad["joint_origin_xyz"]) + np.array(pad["joint_axis"]) * q[robot["joint_names"].index(pad["joint_name"])]
            observed = world_rotation.T @ (actual[:3] - reference[:3])
            position_error = float(np.linalg.norm(observed - expected))
            orientation_error = float(np.linalg.norm(world_rotation.T @ rotation(actual[3:]) - np.eye(3)))
            if position_error > 1e-6 or orientation_error > 1e-5:
                raise ValueError("Recorded native finger transforms disagree with the robot URDF")
            measured.append({"arm": robot["arm_name"], "link": link,
                "position_error_m": position_error, "rotation_matrix_error_norm": orientation_error})
    output.mkdir(parents=True, exist_ok=False)
    (output / "usd_finger_vertices.json").write_text(json.dumps(usd) + "\n")
    record = {"schema": "robodojo-physical-tool-candidate-1", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "robot_only_geometry_and_recorded_measurements", "calibration_status": "uncalibrated_candidate",
        "definition": "Fixed nominal midpoint of the two opposed distal inner-pad surface centroids under symmetric finger displacement",
        "axes": {"x": "along finger extension", "y": "from link8 toward link7", "z": "right-handed completion; unchanged link6 orientation"},
        "link6_from_tool": tool.tolist(), "pad_geometry": pads,
        "selection": {"normal_alignment_tolerance": 1e-6, "minimum_face_centroid_x_m": .04,
            "rationale": "Select the distal opposed planar pad surfaces, excluding the proximal sloped finger plates"},
        "recorded_pose_crosschecks": measured,
        "source_files": [{"path": str(p), "sha256": digest(p)} for p in sources],
        "script_sha256": digest(Path(__file__)), "inventory_path": str(args.inventory.resolve()),
        "inventory_sha256": digest(args.inventory), "usd_vertices_sha256": digest(output / "usd_finger_vertices.json"),
        "rendered_landmarks_validated": False, "live_multi_pose_validation_passed": False,
        "controller_tracking_validated": False, "model_input_eligible": False,
        "full_test_authorized": False, "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
        "limitations": ["One recorded home pose crosschecked; no live movement or image landmark validation",
            "Fixed nominal tool point is not the instantaneous midpoint when finger displacements differ",
            "Does not establish contact, grasp success, pad clearance, or task success"]}
    (output / "candidate.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(output), "link6_from_tool": tool.tolist(),
        "maximum_recorded_finger_position_error_m": max(r["position_error_m"] for r in measured),
        "calibration_status": record["calibration_status"]}))


if __name__ == "__main__":
    main()
