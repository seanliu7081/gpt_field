"""Original LIBERO Panda geometry, isolated from evaluator-only scene truth.

The exported planning package contains only the robot's native collision geoms,
kinematics, joint readout and fixed camera extrinsics. No scene object geometry,
depth, focal length, labels or object joint state is copied into that package.
MuJoCo mesh collision uses convex hulls; those native hulls are sampled without
substituting spheres, capsules or simplified gripper meshes. Surface sampling
and bounded joint interpolation are finite approximations, not CCD certificates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
SCHEMA_VERSION = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed_{int(index)}"


def _robot_geom_ids(model):
    result = []
    for gid in range(model.ngeom):
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name.startswith(("robot0_", "gripper0_")) and (
            model.geom_contype[gid] or model.geom_conaffinity[gid]
        ):
            result.append(gid)
    return np.asarray(result, dtype=int)


def _joint_addresses(model, names):
    ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
    if np.any(ids < 0):
        raise ValueError("missing original robot joints")
    return ids, np.asarray(model.jnt_qposadr[ids], int)


def export_robot_package(snapshot_evaluator, output):
    """Create a NEW, independently loadable robot-only package.

    This is the only preparation function allowed to read full-scene artifacts.
    RobotGeometry below never opens those paths. Camera EXTRINSICS are explicitly
    supplied calibration for the offline planning experiment, not RGB inference.
    """
    source, out = Path(snapshot_evaluator).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    xml_path = source / "render_model.xml"
    state_path = source / "camera_sample_state.npz"
    calibration_path = source / "calibration.json"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    with np.load(state_path, allow_pickle=False) as state:
        qpos = state["qpos"].copy()
        expected_positions = state["geom_xpos"].copy()
        expected_rotations = state["geom_xmat"].copy()
    arm_ids, arm_addresses = _joint_addresses(model, ARM_JOINTS)
    _, finger_addresses = _joint_addresses(model, FINGER_JOINTS)
    original_gids = _robot_geom_ids(model)
    expected_names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in original_gids]
    if len(original_gids) != 13:
        raise ValueError(f"expected original 13 Panda collision geoms, got {len(original_gids)}")
    root = ET.parse(xml_path).getroot()
    isolated = ET.Element("mujoco", {"model": "original_libero_robot_only"})
    for tag in ("compiler", "option", "size", "default"):
        value = root.find(tag)
        if value is not None:
            isolated.append(copy.deepcopy(value))
    compiler = isolated.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(isolated, "compiler")
    compiler.set("meshdir", "assets")
    world = ET.SubElement(isolated, "worldbody")
    candidates = [b for b in root.find("worldbody") if b.tag == "body" and b.get("name") == "robot0_base"]
    if len(candidates) != 1:
        raise ValueError("expected one original robot0_base tree")
    robot = copy.deepcopy(candidates[0])
    world.append(robot)
    # Retain native colliders and all kinematics/inertials/sites, remove visuals.
    for parent in list(robot.iter()):
        for child in list(parent):
            if child.tag in ("camera", "light") or (child.tag == "geom" and child.get("name") not in expected_names):
                parent.remove(child)
            elif child.tag == "geom":
                child.attrib.pop("material", None)
    meshes = {g.get("mesh") for g in robot.iter("geom") if g.get("mesh")}
    assets = ET.SubElement(isolated, "asset")
    out.mkdir(parents=True, exist_ok=False)
    (out / "assets").mkdir()
    copied_assets = []
    for asset in root.find("asset"):
        if asset.tag != "mesh" or asset.get("name") not in meshes:
            continue
        element = copy.deepcopy(asset)
        filename = asset.get("file")
        if filename:
            original = Path(filename)
            if not original.is_absolute():
                original = xml_path.parent / root.find("compiler").get("meshdir", "") / original
            digest = sha256(original)
            target = out / "assets" / (digest[:16] + original.suffix)
            if not target.exists():
                shutil.copyfile(original, target)
            element.set("file", target.name)
            copied_assets.append({"mesh_name": asset.get("name"), "file": str(target.relative_to(out)), "sha256": digest})
        assets.append(element)
    if len(assets) != len(meshes):
        raise ValueError("missing native robot mesh assets")
    # Original explicit robot exclusions/pairs, if present, are retained.
    body_names = {b.get("name") for b in robot.iter("body")}
    original_contacts = root.find("contact")
    if original_contacts is not None:
        contacts = ET.SubElement(isolated, "contact")
        for item in original_contacts:
            if (item.tag == "exclude" and item.get("body1") in body_names and item.get("body2") in body_names) or (
                item.tag == "pair" and item.get("geom1") in expected_names and item.get("geom2") in expected_names
            ):
                contacts.append(copy.deepcopy(item))
    ET.indent(isolated)
    ET.ElementTree(isolated).write(out / "robot.xml", encoding="unicode")
    camera = json.loads(calibration_path.read_text())["T_camera_cv_from_world"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "robot": "original LIBERO Panda, native MuJoCo collision geometry",
        "arm_joint_names": list(ARM_JOINTS), "finger_joint_names": list(FINGER_JOINTS),
        "q_initial": qpos[arm_addresses].tolist(), "finger_q_initial": qpos[finger_addresses].tolist(),
        "joint_limits": model.jnt_range[arm_ids].tolist(),
        "T_camera_cv_from_world": camera,
        "camera_transform_semantics": "explicit fixed rig extrinsics; no focal, depth or environment geometry",
        "collision_geom_names": expected_names,
        "native_mesh_semantics": "MuJoCo convex hull of original collision-mesh vertices; native pad boxes retained",
        "assets": copied_assets,
        "excluded_data": ["environment geometry", "object state", "depth", "segmentation", "camera focal", "task goal"],
        "robot_xml_sha256": sha256(out / "robot.xml"),
    }
    (out / "robot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return audit_robot_package(source, out)


def audit_robot_package(snapshot_evaluator, output):
    source, out = Path(snapshot_evaluator).resolve(), Path(output).resolve()
    xml_path = source / "render_model.xml"
    state_path = source / "camera_sample_state.npz"
    calibration_path = source / "calibration.json"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    with np.load(state_path, allow_pickle=False) as state:
        data.qpos[:] = state["qpos"]
        expected_positions = state["geom_xpos"].copy()
        expected_rotations = state["geom_xmat"].copy()
    mujoco.mj_forward(model, data)
    original_gids = _robot_geom_ids(model)
    expected_names = [_name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in original_gids]
    robot_only = RobotGeometry(out, sample_spacing_m=None)
    robot_only.set_q(robot_only.q_initial)
    errors = []
    for source_gid, name in zip(original_gids, expected_names):
        dest = mujoco.mj_name2id(robot_only.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        errors.append(max(float(np.max(np.abs(robot_only.data.geom_xpos[dest] - data.geom_xpos[source_gid]))),
                          float(np.max(np.abs(robot_only.data.geom_xmat[dest] - data.geom_xmat[source_gid])))))
    maximum = max(errors)
    if maximum > 2e-12:
        raise ValueError(f"robot-only FK mismatch with reloaded original model: {maximum}")
    archived_position_error = float(np.max(np.abs(data.geom_xpos[original_gids] - expected_positions[original_gids])))
    archived_rotation_error = float(np.max(np.abs(data.geom_xmat[original_gids] - expected_rotations[original_gids])))
    audit = {"robot_geom_count": len(original_gids), "maximum_fk_component_error": maximum,
             "source_artifact_sha256": {p.name: sha256(p) for p in (xml_path, state_path, calibration_path)},
             "robot_metadata_sha256": sha256(out / "robot.json"),
             "original_robot_geometry_preserved": True, "environment_geometry_exported": False,
             "archived_xml_roundtrip_maximum_position_component_error_m": archived_position_error,
             "archived_xml_roundtrip_maximum_rotation_component_error": archived_rotation_error,
             "xml_roundtrip_note": "Planner and truth use identical reloaded original XML; saved XML decimal precision differs slightly from pre-serialization camera-state geom transforms."}
    (out / "export_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def collision_mesh(model, gid):
    """Native collision hull, in MuJoCo's compiled geom-local coordinates."""
    kind = int(model.geom_type[gid])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        first, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[first:first + count], float).copy()
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        # Orient outward for consumers that need consistent normals.
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        vertices = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * model.geom_size[gid]
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        faces[reverse] = faces[reverse][:, [0, 2, 1]]
    else:
        raise ValueError("unexpected original robot geom; refusing surrogate replacement")
    return vertices, faces


def sample_triangle_mesh(vertices, faces, max_spacing_m):
    """Adaptive centroid cover with surface covering radius <= max_spacing.

    Every original triangle is covered. Refine its longest edge until all three
    vertices are within spacing of the centroid. By convexity, the entire
    triangle (including its edges) is then within spacing of that sample.
    Original mesh vertices are included as well. This is a conservative cover,
    not calibrated perception uncertainty or a simplification of the geometry.
    This covers the original hull SURFACE, not its whole solid volume.
    """
    spacing = float(max_spacing_m)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("positive finite sample spacing required")
    active = np.asarray(vertices)[np.asarray(faces)].copy()
    pieces = [np.asarray(vertices).copy()]
    area = float(np.linalg.norm(np.cross(active[:, 1] - active[:, 0], active[:, 2] - active[:, 0]), axis=1).sum() / 2)
    maximum_refined_edge = 0.
    maximum_cover_radius = 0.
    while len(active):
        lengths = np.stack([np.linalg.norm(active[:, a] - active[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], axis=1)
        longest = np.argmax(lengths, axis=1)
        maxima = lengths[np.arange(len(active)), longest]
        centroids = active.mean(axis=1)
        radii = np.linalg.norm(active - centroids[:, None], axis=2).max(axis=1)
        done = radii <= spacing - 2e-13
        if np.any(done):
            pieces.append(centroids[done])
            maximum_refined_edge = max(maximum_refined_edge, float(maxima[done].max()))
            maximum_cover_radius = max(maximum_cover_radius, float(radii[done].max()))
        active, longest = active[~done], longest[~done]
        if not len(active):
            break
        # Cyclic ordering puts the longest edge first without changing winding.
        ordered = np.stack([active[np.arange(len(active)), (longest + offset) % 3] for offset in range(3)], axis=1)
        midpoint = (ordered[:, 0] + ordered[:, 1]) * .5
        active = np.concatenate([np.stack([ordered[:, 0], midpoint, ordered[:, 2]], axis=1),
                                 np.stack([midpoint, ordered[:, 1], ordered[:, 2]], axis=1)])
    samples = np.unique(np.round(np.concatenate(pieces), 13), axis=0)
    return samples, {"sample_count": len(samples), "surface_cover_radius_m": maximum_cover_radius + 2e-13,
                     "maximum_refined_edge_m": maximum_refined_edge, "surface_area_m2": float(area),
                     "covers": "entire native collision-hull surface; not volume",
                     "sampling_method": "deterministic longest-edge refinement; centroid covers every subtriangle and its edges; original mesh vertices included"}


class RobotGeometry:
    """Planner-visible robot only; never reads environment/evaluator artifacts."""
    def __init__(self, package, sample_spacing_m=.005):
        self.package = Path(package).resolve()
        self.metadata = json.loads((self.package / "robot.json").read_text())
        if sha256(self.package / "robot.xml") != self.metadata["robot_xml_sha256"]:
            raise ValueError("robot XML integrity failure")
        for asset in self.metadata["assets"]:
            if sha256(self.package / asset["file"]) != asset["sha256"]:
                raise ValueError("native robot mesh integrity failure")
        self.model = mujoco.MjModel.from_xml_path(str(self.package / "robot.xml"))
        self.data = mujoco.MjData(self.model)
        self.arm_ids, self.arm_addresses = _joint_addresses(self.model, ARM_JOINTS)
        self.finger_ids, self.finger_addresses = _joint_addresses(self.model, FINGER_JOINTS)
        self.q_initial = np.asarray(self.metadata["q_initial"], float)
        self.finger_q_initial = np.asarray(self.metadata["finger_q_initial"], float)
        self.joint_limits = np.asarray(self.metadata["joint_limits"], float)
        self.camera_from_world = np.asarray(self.metadata["T_camera_cv_from_world"], float)
        self.geom_ids = _robot_geom_ids(self.model)
        if self.model.ngeom != 13 or len(self.geom_ids) != 13 or self.model.nq != 9:
            raise ValueError("planning package contains unexpected geometry or state")
        self.eef_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if self.eef_site < 0:
            raise ValueError("missing original grip site")
        self.local_meshes, self.local_samples, self.sample_metadata = {}, {}, []
        for gid in self.geom_ids:
            vertices, faces = collision_mesh(self.model, gid)
            self.local_meshes[int(gid)] = (vertices, faces)
            if sample_spacing_m is not None:
                points, info = sample_triangle_mesh(vertices, faces, sample_spacing_m)
                self.local_samples[int(gid)] = points
                info.update(geom_id=int(gid), geom_name=_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                            native_type=int(self.model.geom_type[gid]), vertices=len(vertices), faces=len(faces))
                self.sample_metadata.append(info)
        self.set_q(self.q_initial)

    def set_q(self, q):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("finite 7-DOF arm vector required")
        self.data.qpos[self.arm_addresses] = q
        self.data.qpos[self.finger_addresses] = self.finger_q_initial
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def within_limits(self, q):
        q = np.asarray(q, float)
        return bool(q.shape == (7,) and np.all(np.isfinite(q)) and np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    def transforms(self, q):
        self.set_q(q)
        return {int(gid): {"rotation_world_from_local": self.data.geom_xmat[gid].reshape(3, 3).copy(),
                           "translation_world_m": self.data.geom_xpos[gid].copy(),
                           "name": _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)} for gid in self.geom_ids}

    def samples(self, q, gripper_only=False, frame="camera"):
        self.set_q(q)
        points, ids = [], []
        for gid in self.geom_ids:
            if gripper_only and not _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid).startswith("gripper0_"):
                continue
            local = self.local_samples[int(gid)]
            world = local @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
            points.append(world)
            ids.append(np.full(len(world), gid, dtype=np.int32))
        xyz = np.concatenate(points)
        if frame == "camera":
            xyz = xyz @ self.camera_from_world[:3, :3].T + self.camera_from_world[:3, 3]
        elif frame != "world":
            raise ValueError("frame must be camera or world")
        return xyz, np.concatenate(ids)

    def eef_pose(self, q, frame="world"):
        self.set_q(q)
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.eef_site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.eef_site]
        return self.camera_from_world @ pose if frame == "camera" else pose

    def check(self, q, penetration_tolerance_m=1e-7):
        if not self.within_limits(q):
            return {"valid": False, "self_collision": False, "joint_limits_valid": False, "contacts": []}
        self.set_q(q)
        contacts = _contacts(self.model, self.data, set(map(int, self.geom_ids)), penetration_tolerance_m)
        return {"valid": not contacts, "self_collision": bool(contacts), "joint_limits_valid": True, "contacts": contacts}


def _contacts(model, data, robot_ids, tolerance):
    result = []
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if (a in robot_ids or b in robot_ids) and float(contact.dist) < -tolerance:
            result.append({"geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, a),
                           "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, b),
                           "distance_m": float(contact.dist),
                           "kind": "self" if a in robot_ids and b in robot_ids else "environment"})
    return result


def native_collision_pair_eligible(model, a, b):
    """Native dynamic pair filters for these snapshots (no explicit pairs).

    Matches contact bits, same welded body, parent-weld filter and exclusions.
    Explicit predefined pairs override these and are handled separately by the
    evaluator. Tests compare against native contacts over deterministic poses.
    """
    ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
    wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
    if wa == wb:
        return False
    if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or
            (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
        return False
    signature = (min(ba, bb) << 16) + max(ba, bb)
    if signature in model.exclude_signature:
        return False
    if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
        if wa != 0 and wb != 0 and (int(model.body_weldid[model.body_parentid[wa]]) == wb or
                                    int(model.body_weldid[model.body_parentid[wb]]) == wa):
            return False
    return True


class GroundTruthEvaluator:
    """Evaluator-only original static LIBERO scene; never pass this to planner."""
    def __init__(self, snapshot_evaluator, distance_cap_m=.05):
        self.source = Path(snapshot_evaluator).resolve()
        xml_path = self.source / "render_model.xml"
        root = ET.parse(xml_path).getroot()
        compiler = root.find("compiler")
        consumed = [xml_path, self.source / "camera_sample_state.npz", self.source / "calibration.json"]
        for asset in root.find("asset"):
            if not asset.get("file"):
                continue
            asset_path = Path(asset.get("file"))
            if not asset_path.is_absolute():
                directory = compiler.get("meshdir" if asset.tag == "mesh" else "texturedir", "") if compiler is not None else ""
                asset_path = self.source / directory / asset_path
            consumed.append(asset_path.resolve())
        self.consumed_sha256 = {str(path): sha256(path) for path in sorted(set(consumed))}
        self.model = mujoco.MjModel.from_xml_path(str(self.source / "render_model.xml"))
        self.data = mujoco.MjData(self.model)
        with np.load(self.source / "camera_sample_state.npz", allow_pickle=False) as state:
            self.frozen_qpos = state["qpos"].copy()
        self.arm_ids, self.arm_addresses = _joint_addresses(self.model, ARM_JOINTS)
        self.joint_limits = self.model.jnt_range[self.arm_ids].copy()
        self.q_initial = self.frozen_qpos[self.arm_addresses].copy()
        self.robot_ids = set(map(int, _robot_geom_ids(self.model)))
        self.distance_cap_m = float(distance_cap_m)
        if not np.isfinite(self.distance_cap_m) or self.distance_cap_m <= 0:
            raise ValueError("distance cap must be positive")
        explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(self.model.pair_geom1, self.model.pair_geom2)}
        self.environment_pairs, self.self_pairs = [], []
        collidable = [i for i in range(self.model.ngeom) if self.model.geom_contype[i] or self.model.geom_conaffinity[i]]
        for a in sorted(self.robot_ids):
            for b in collidable:
                if b in self.robot_ids and b <= a:
                    continue
                if a == b:
                    continue
                if tuple(sorted((a, b))) in explicit or native_collision_pair_eligible(self.model, a, b):
                    (self.self_pairs if b in self.robot_ids else self.environment_pairs).append((a, b))
        self._set_q(self.q_initial)

    def provenance(self):
        return {"consumed_sha256": dict(self.consumed_sha256),
                "environment_access": "evaluator only; never exported into RobotGeometry package",
                "native_engine": "MuJoCo " + mujoco.__version__,
                "original_collision_geometry": True,
                "eligible_robot_environment_pair_count": len(self.environment_pairs),
                "eligible_self_pair_count": len(self.self_pairs),
                "environment_state": "frozen original camera sample qpos; zero qvel; no dynamics step",
                "positive_distance_cap_m": self.distance_cap_m}

    def _set_q(self, q):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("finite 7-DOF arm vector required")
        self.data.qpos[:] = self.frozen_qpos
        self.data.qpos[self.arm_addresses] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _minimum_distance(self, pairs):
        best, closest = self.distance_cap_m, None
        for a, b in pairs:
            # Native bounding spheres are ONLY an exact broad-phase cull; all
            # retained distances use original geoms via mj_geomDistance.
            if int(self.model.geom_type[a]) != 0 and int(self.model.geom_type[b]) != 0:
                lower = np.linalg.norm(self.data.geom_xpos[a] - self.data.geom_xpos[b]) - self.model.geom_rbound[a] - self.model.geom_rbound[b]
                if lower >= best:
                    continue
            distance = float(mujoco.mj_geomDistance(self.model, self.data, a, b, self.distance_cap_m, None))
            if distance < best:
                best = distance
                closest = [_name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in (a, b)]
        return best, closest

    def check(self, q, penetration_tolerance_m=1e-7, distances=True):
        self._set_q(q)
        q = np.asarray(q, float)
        limits = bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))
        contacts = _contacts(self.model, self.data, self.robot_ids, penetration_tolerance_m)
        environment = [c for c in contacts if c["kind"] == "environment"]
        self_contacts = [c for c in contacts if c["kind"] == "self"]
        result = {"valid": limits and not contacts, "joint_limits_valid": limits,
                  "environment_collision": bool(environment), "self_collision": bool(self_contacts),
                  "contacts": contacts, "penetration_tolerance_m": penetration_tolerance_m}
        if distances:
            env_distance, env_pair = self._minimum_distance(self.environment_pairs)
            self_distance, self_pair = self._minimum_distance(self.self_pairs)
            result.update(environment_min_distance_m=env_distance, environment_closest_pair=env_pair,
                          self_min_distance_m=self_distance, self_closest_pair=self_pair,
                          distance_cap_m=self.distance_cap_m,
                          distance_semantics="native eligible geom pairs; positive distances capped; not mesh-surface sampling")
        return result

    def validate_path(self, path, max_joint_step_rad=.005, penetration_tolerance_m=1e-7, keep_samples=False):
        path = np.asarray(path, float)
        step = float(max_joint_step_rad)
        if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.all(np.isfinite(path)) or not np.isfinite(step) or step <= 0:
            raise ValueError("finite nonempty Nx7 path and positive step required")
        count, minimum, first, records = 0, self.distance_cap_m, None, []
        maximum_observed_step = 0.
        previous = None
        for segment in range(max(1, len(path) - 1)):
            a = path[segment]
            b = path[min(segment + 1, len(path) - 1)]
            n = max(1, int(math.ceil(np.max(np.abs(b - a)) / step)))
            for index in range(0 if segment == 0 else 1, n + 1):
                q = a + (b - a) * (index / n)
                if previous is not None:
                    maximum_observed_step = max(maximum_observed_step, float(np.max(np.abs(q - previous))))
                previous = q
                result = self.check(q, penetration_tolerance_m)
                minimum = min(minimum, result["environment_min_distance_m"])
                if not result["valid"] and first is None:
                    first = {"sample_index": count, "segment_index": segment, "segment_fraction": index / n,
                             "q": q.tolist(), **result}
                if keep_samples:
                    records.append({"sample_index": count, "segment_index": segment, "fraction": index / n, **result})
                count += 1
        summary = {"valid_at_all_samples": first is None, "sample_count": count,
                   "environment_min_distance_m": minimum, "first_invalid_sample": first,
                   "max_joint_step_rad": step, "maximum_observed_joint_step_rad": maximum_observed_step,
                   "finger_joints": "frozen at original snapshot readout",
                   "environment": "frozen at exact original camera sample state",
                   "continuous_collision_certificate": False,
                   "limitation": "finite joint-linear samples; collisions between samples can be missed",
                   "distance_cap_m": self.distance_cap_m}
        if keep_samples:
            summary["samples"] = records
        return summary
