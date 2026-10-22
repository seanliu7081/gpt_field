"""Export static LIBERO collision scenes without policy inference or task replay.

Bowl002 is initialized using its archived simulator implementation and reset
configuration (including the original ten neutral settling actions). Drawer003
uses already recorded geometry poses at pre-action steps60 and552. Its RGB
images come from the exact corresponding archived pre-action observations.
All outputs go into a newly created directory; archived runs are read-only.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

BOWL = Path("/workspace/gpt_field/runs/agent_speedup_v1_bowl_zero_shot_002/trial")
DRAWER = Path("/workspace/gpt_field/runs/agent_speedup_v1_drawer_zero_shot_003/trial")
ARCHIVED_GEOMETRY = Path("/workspace/agent_speedup_v1_drawer003_scene_fidelity/run_001")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def geometry_records(geometries, positions, rotations):
    """Convert collision-only metadata/poses into the shared world-space schema."""
    boxes, planes = [], []
    for geom in geometries:
        if not geom.get("in_scene_occupancy", geom["role"] not in ("robot", "robot_finger")):
            continue
        gid = geom["id"]
        center = np.asarray(positions[gid], float)
        rotation = np.asarray(rotations[gid], float).reshape(3, 3)
        size = np.asarray(geom["size"], float)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(rotation)):
            raise ValueError("nonfinite geometry pose")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-12, rtol=0):
            raise ValueError("nonorthogonal geometry orientation")
        item = {"geom_id": int(gid), "name": geom["name"], "body": geom["body"],
                "object": geom["object"], "role": geom["role"],
                "center_m": center.tolist(), "rotation_local_to_world": rotation.tolist(),
                "contype": int(geom["contype"]), "conaffinity": int(geom["conaffinity"]),
                "collision_margin_m": float(geom["margin_m"])}
        if geom["type"] == "box":
            if np.any(size <= 0):
                raise ValueError("nonpositive collision box size")
            item["half_extents_m"] = size.tolist()
            boxes.append(item)
        elif geom["type"] == "plane":
            item.update(normal_world=rotation[:, 2].tolist(), displayed_size_m=size.tolist(),
                        geometry="infinite_plane_halfspace_negative_local_z")
            planes.append(item)
        else:
            raise ValueError(f"unsupported scene collision shape; cannot silently omit {geom['type']}:{geom['name']}")
    return boxes, planes


def object_bounds(boxes):
    bounds = {}
    for item in boxes:
        c = np.asarray(item["center_m"])
        radius = np.abs(np.asarray(item["rotation_local_to_world"])) @ np.asarray(item["half_extents_m"])
        key = item["object"]
        if key not in bounds:
            bounds[key] = [c - radius, c + radius]
        else:
            bounds[key][0] = np.minimum(bounds[key][0], c - radius)
            bounds[key][1] = np.maximum(bounds[key][1], c + radius)
    return {key: {"lower_m": value[0].tolist(), "upper_m": value[1].tolist()} for key, value in bounds.items()}


def image_bytes(data_uri):
    if not data_uri.startswith("data:image/png;base64,"):
        raise ValueError("expected archived PNG data URI")
    raw = base64.b64decode(data_uri.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG":
            raise ValueError("image data is not PNG")
    return raw


def save_observation_images(observation, folder, root, provenance):
    records = {}
    for key, camera in (("observation/image", "agentview"), ("observation/wrist_image", "robot0_eye_in_hand")):
        raw = image_bytes(observation[key])
        path = folder / f"{camera}.png"
        path.write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as image:
            shape = [image.height, image.width, len(image.getbands())]
        records[camera] = {"file": str(path.relative_to(root)), "sha256": sha256(path),
                           "shape": shape, "provenance": provenance,
                           "orientation": "archived_policy_image_rotate180_then_centered_resize",
                           "warning": "RGB renders visual assets; evaluation reference uses collision geometry, which can differ"}
    return records


def compare_observation(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("reset observation keys differ")
    images = ("observation/image", "observation/wrist_image")
    for key in actual:
        if key not in images and actual[key] != expected[key]:
            raise ValueError(f"reset nonimage mismatch: {key}")
    comparison = {"canonical_observation_sha256": canonical_digest(actual),
                  "expected_canonical_observation_sha256": canonical_digest(expected),
                  "complete_observation_exact": actual == expected, "state8_exact": True, "images": {}}
    for key in images:
        a, b = image_bytes(actual[key]), image_bytes(expected[key])
        x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), np.int16)
        y = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), np.int16)
        if x.shape != y.shape:
            raise ValueError("reset image dimensions differ")
        difference = np.abs(x - y)
        comparison["images"][key] = {"png_bytes_exact": a == b,
            "pixels_differing": int(np.any(difference != 0, axis=2).sum()),
            "maximum_channel_difference": int(difference.max())}
    return comparison


def trace_preaction_observations(path, steps):
    """Return observations before specified actions, not the images after them."""
    steps, found = set(steps), {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "reset" and 0 in steps:
                found[0] = row["observation"]
            elif row.get("event") == "executed" and row["step"] + 1 in steps:
                found[row["step"] + 1] = row["observation"]
            if len(found) == len(steps):
                break
    if set(found) != steps:
        raise ValueError(f"trace missing pre-action observations: {sorted(steps-set(found))}")
    return found


def export_drawer(root):
    static = json.loads((ARCHIVED_GEOMETRY / "geometry_static.json").read_text())
    capture = json.loads((ARCHIVED_GEOMETRY / "capture_summary.json").read_text())
    if capture["status"] != "complete" or not capture["source_unchanged"] or capture["max_state_error"] != 0:
        raise ValueError("prior drawer geometry capture is not a validated replay")
    steps = [60, 552]
    observations = trace_preaction_observations(DRAWER / "trace.jsonl", steps)
    snapshots = []
    with np.load(ARCHIVED_GEOMETRY / "geometry_frames.npz", allow_pickle=False) as data:
        for step, state_label in zip(steps, ("closed", "open")):
            indices = np.flatnonzero(data["step"] == step)
            if len(indices) != 1:
                raise ValueError("drawer step must identify exactly one saved frame")
            index = int(indices[0])
            if not np.array_equal(data["state8"][index], observations[step]["observation/state"]):
                raise ValueError("drawer RGB observation and geometry frame state8 disagree")
            name = f"drawer003_{state_label}_step{step}"
            folder = root / "snapshots" / name
            folder.mkdir()
            boxes, planes = geometry_records(static, data["geom_xpos"][index], data["geom_xmat"][index])
            state_file = folder / "static_state.npz"
            np.savez_compressed(state_file, qpos=data["qpos"][index], qvel=data["qvel"][index],
                                geom_xpos=data["geom_xpos"][index], geom_xmat=data["geom_xmat"][index],
                                state8=data["state8"][index])
            image_provenance = f"original drawer003 trace executed action {step-1}; exact pre-action {step} observation"
            snapshot = {"name": name, "task": "drawer003", "task_language": observations[step]["prompt"],
                "boxes": boxes, "planes": planes, "object_bounds": object_bounds(boxes),
                "state": {"label": state_label, "pre_action_step": step, "archive_frame_index": index,
                    "file": str(state_file.relative_to(root)), "sha256": sha256(state_file),
                    "state8_matches_archived_observation": True,
                    "observation_sha256": canonical_digest(observations[step]),
                    "drawer_slide_qpos_index37_m": float(data["qpos"][index, 37]),
                    "selection_reason": "first stored closed snapshot" if step == 60 else
                        "first stored fully open snapshot; target bowl remains on table before transport"},
                "reference_rgb": save_observation_images(observations[step], folder, root, image_provenance),
                "source": {"trial": str(DRAWER), "geometry_capture": str(ARCHIVED_GEOMETRY),
                           "new_simulator_steps": 0, "new_policy_actions": 0, "new_model_calls": 0},
                "geometry_reference": "MuJoCo nonrobot collision geometry; plane exported separately"}
            write_json(folder / "geometry.json", snapshot)
            snapshots.append(snapshot)
            print(f"DRAWER {name} boxes={len(boxes)} planes={len(planes)}", flush=True)
    return snapshots


def export_bowl(root):
    configuration = json.loads((BOWL / "manifest.json").read_text())["configuration"]
    expected = json.loads((BOWL / "initial_observation.json").read_text())
    runtime = root / "runtime" / "gpt_field"
    runtime.mkdir(parents=True)
    copied = {}
    for source in sorted((BOWL / "frozen_source").glob("*.py")):
        destination = runtime / source.name
        shutil.copy2(source, destination)
        copied[source.name] = sha256(destination)
        if copied[source.name] != sha256(source):
            raise ValueError("archived runtime changed during copying")
    write_json(root / "frozen_runtime_hashes.json", copied)
    for key, value in configuration["render_environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    sys.path.insert(0, str(root / "runtime"))
    import mujoco
    from gpt_field.libero import LiberoConfig, _LiberoWorker
    worker = None
    name = "bowl002_initial"
    folder = root / "snapshots" / name
    folder.mkdir()
    try:
        worker = _LiberoWorker(LiberoConfig(**configuration["simulator"]))
        original_metadata = json.loads((BOWL / "simulator.json").read_text())["metadata"]
        for key in ("packages", "controller_sources", "libero_source", "controller"):
            if worker.metadata[key] != original_metadata[key]:
                raise ValueError(f"simulator implementation drift: {key}")
        response = worker.reset(configuration["task_id"], configuration["episode_index"])
        observation = response["observation"]
        comparison = compare_observation(observation, expected)
        write_json(folder / "initial_observation_comparison.json", comparison)
        # Require actual state and image byte identity for this reset capture.
        if not comparison["complete_observation_exact"]:
            write_json(folder / "actual_observation.json", observation)
            raise ValueError("bowl002 reset did not match the archived state/image hashes")
        model, data = worker.env.sim.model._model, worker.env.sim.data._data
        geometries = []
        for gid, record in sorted(worker._collision_monitor._geometry.items()):
            kind = mujoco.mjtGeom(int(model.geom_type[gid])).name.removeprefix("mjGEOM_").lower()
            geometries.append({**record, "type": kind, "size": model.geom_size[gid].tolist(),
                "contype": int(model.geom_contype[gid]), "conaffinity": int(model.geom_conaffinity[gid]),
                "margin_m": float(model.geom_margin[gid]),
                "in_scene_occupancy": record["role"] not in ("robot", "robot_finger")})
        boxes, planes = geometry_records(geometries, data.geom_xpos, data.geom_xmat)
        state_file = folder / "static_state.npz"
        np.savez_compressed(state_file, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                            sim_time=float(data.time), geom_xpos=data.geom_xpos.copy(),
                            geom_xmat=data.geom_xmat.copy(), state8=np.asarray(observation["observation/state"]))
        xml_file = folder / "model.xml"
        xml_file.write_text(worker.env.sim.model.get_xml())
        write_json(folder / "simulator_metadata.json", worker.metadata)
        write_json(folder / "collision_geometry_static.json", geometries)
        snapshot = {"name": name, "task": "bowl002", "task_language": observation["prompt"],
            "boxes": boxes, "planes": planes, "object_bounds": object_bounds(boxes),
            "state": {"label": "initial_after_original_settling", "pre_action_step": 0,
                "file": str(state_file.relative_to(root)), "sha256": sha256(state_file),
                "observation_sha256": canonical_digest(observation),
                "complete_observation_matches_archive": True,
                "model_xml": str(xml_file.relative_to(root)), "model_xml_sha256": sha256(xml_file)},
            "reference_rgb": save_observation_images(observation, folder, root, "new static reset; PNG bytes exactly match archived bowl002 initial observation"),
            "source": {"trial": str(BOWL), "suite": configuration["simulator"]["suite"],
                "task_id": configuration["task_id"], "episode_index": configuration["episode_index"],
                "seed": configuration["simulator"]["seed"],
                "setup_neutral_settling_actions": configuration["simulator"]["settle_steps"],
                "new_policy_actions": 0, "new_model_calls": 0,
                "initial_state_file": worker.status["initial_state_file"],
                "initial_state_sha256": worker.status["initial_state_sha256"],
                "bddl_sha256": worker.status["bddl_sha256"]},
            "geometry_reference": "MuJoCo nonrobot collision geometry; plane exported separately"}
        write_json(folder / "geometry.json", snapshot)
        print(f"BOWL {name} exact observation hash; boxes={len(boxes)} planes={len(planes)}", flush=True)
        return snapshot
    finally:
        if worker is not None:
            worker.close()


def run(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "snapshots").mkdir()
    shutil.copy2(__file__, root / "capture_used.py")
    sources = [Path(__file__), BOWL / "manifest.json", BOWL / "simulator.json", BOWL / "initial_observation.json",
               DRAWER / "manifest.json", DRAWER / "simulator.json", DRAWER / "trace.jsonl",
               ARCHIVED_GEOMETRY / "geometry_static.json", ARCHIVED_GEOMETRY / "geometry_frames.npz",
               ARCHIVED_GEOMETRY / "capture_summary.json"]
    sources.extend(sorted((BOWL / "frozen_source").glob("*.py")))
    before = {str(path): sha256(path) for path in sources}
    manifest = {"status": "running", "schema_version": 1, "units": "m",
        "started_utc": datetime.now(timezone.utc).isoformat(), "new_model_calls": 0,
        "new_policy_actions": 0, "source_hashes_before": before,
        "reference_warning": "Collision geometry is simulator ground truth for this test; RGB visual meshes may differ.",
        "union_warning": "Boxes can overlap; use their true geometric union boundary for signed Euclidean distance, not min-CSG interior distance."}
    write_json(root / "manifest.json", manifest)
    snapshots = []
    start = time.perf_counter()
    try:
        snapshots.extend(export_drawer(root))
        snapshots.insert(0, export_bowl(root))
        manifest.update(status="complete", snapshot_count=len(snapshots),
                        counts={s["name"]: {"boxes": len(s["boxes"]), "planes": len(s["planes"]),
                                           "objects": dict(Counter(b["object"] for b in s["boxes"]))} for s in snapshots})
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=traceback.format_exc())
        print(manifest["error"], flush=True)
    finally:
        unchanged = {path: sha256(path) == digest for path, digest in before.items()}
        if not all(unchanged.values()):
            manifest["status"] = "source_integrity_failed"
        manifest.update(source_unchanged=all(unchanged.values()), source_integrity=unchanged,
                        elapsed_s=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / "snapshots.json", {"status": manifest["status"], "schema_version": 1,
                                             "units": "m", "snapshots": snapshots})
        manifest["snapshots_sha256"] = sha256(root / "snapshots.json")
        write_json(root / "manifest.json", manifest)
    print(f"FINISHED {manifest['status']} snapshots={len(snapshots)}", flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new directory, never an existing archive")
    sys.exit(run(parser.parse_args().output))
