"""Prepare one observable-only LIBERO demonstration for an inference reference.

Official create_dataset.py stores each observation AFTER its paired action.
This producer preserves that relation; it never reads simulator state vectors,
model XML, rewards, dones, object poses, or joint-state observations. Choosing a
source episode distinct from an evaluation initialization is an offline audit,
not a policy input. The selected episode is explicit via --demo-key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task
from gpt_field.observation import image_to_data_url, quaternion_to_axisangle


def _text(value, name):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _array(dataset, shape, name):
    array = np.asarray(dataset[:])
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must have numeric shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.astype(float)


def _canonical_axisangle(value):
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.zeros(3)
    quaternion = np.r_[value / angle * math.sin(angle / 2), math.cos(angle / 2)]
    return np.asarray(quaternion_to_axisangle(quaternion))


def validate_source_metadata(data_group, task_key):
    task = resolve_distance_task(task_key)
    attrs = data_group.attrs
    problem = json.loads(_text(attrs["problem_info"], "problem_info"))
    if problem.get("language_instruction") != task.prompt:
        raise ValueError("HDF5 task language does not match selected task")
    expected_bddl = task.prompt.replace(" ", "_") + ".bddl"
    bddl = Path(_text(attrs["bddl_file_name"], "bddl_file_name"))
    if bddl.name != expected_bddl or bddl.parent.name != task.suite:
        raise ValueError("HDF5 BDDL identity does not match selected task and suite")
    if _text(attrs["macros_image_convention"], "macros_image_convention") != "opengl":
        raise ValueError("Only raw opengl RGB with the live rotate-180 protocol is supported")
    env = json.loads(_text(attrs["env_args"], "env_args"))
    config = env.get("env_kwargs", {}).get("controller_configs", {})
    if config.get("type") != "OSC_POSE" or config.get("control_delta") is not True:
        raise ValueError("HDF5 must use delta OSC_POSE actions")
    for key, expected in (("output_max", [.05, .05, .05, .5, .5, .5]),
                          ("output_min", [-.05, -.05, -.05, -.5, -.5, -.5])):
        actual = np.asarray(config.get(key, []), dtype=float)
        if actual.shape != (6,) or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError("HDF5 action scale differs from the live control protocol")
    if config.get("input_min") != -1 or config.get("input_max") != 1:
        raise ValueError("HDF5 controller must accept normalized [-1,1] commands")
    return task


def build_observable_demo(data_group, *, source_path, file_sha256, demo_key="demo_0", task_key="drawer_bowl"):
    """Read only allowlisted robot observations and actions from one HDF5 group."""
    from gpt_field.demonstration import select_keyframe_indices

    task = validate_source_metadata(data_group, task_key)
    demo = data_group[demo_key]
    count = len(demo["actions"])
    if count <= 0:
        raise ValueError("Demonstration must contain actions")
    actions = _array(demo["actions"], (count, 7), "actions")
    if (np.abs(actions) > 1).any():
        raise ValueError("Actions must be normalized to [-1,1]")
    obs = demo["obs"]
    positions = _array(obs["ee_pos"], (count, 3), "obs/ee_pos")
    rotations = _array(obs["ee_ori"], (count, 3), "obs/ee_ori")
    grippers = _array(obs["gripper_states"], (count, 2), "obs/gripper_states")
    canonical = np.stack([_canonical_axisangle(v) for v in rotations])
    states = np.concatenate([positions, canonical, grippers], axis=1)
    cameras = (obs["agentview_rgb"], obs["eye_in_hand_rgb"])
    for camera in cameras:
        if (len(camera.shape) != 4 or camera.shape[0] != count or camera.shape[-1] != 3
                or min(camera.shape[1:3]) < 1 or camera.dtype != np.uint8):
            raise ValueError("Demo cameras must be action-aligned N x H x W x 3 uint8 RGB")
    rounded_actions = np.round(actions, 6).tolist()
    rounded_states = np.round(states, 6).tolist()
    keyframes = []
    for index in select_keyframe_indices(rounded_actions, max_frames=24):
        external, wrist = (image_to_data_url(camera[index], size=224, rotate=True) for camera in cameras)
        keyframes.append({"after_action_index": index, "external_image": external, "wrist_image": wrist})
    return {
        "schema_version": "libero-observable-demo-v1",
        "task_key": task.key,
        "task_prompt": task.prompt,
        "source": {"format": "libero_hdf5", "file_name": Path(source_path).name,
                   "file_sha256": file_sha256, "demo_key": demo_key,
                   "suite": task.suite, "task_id": task.task_id,
                   "episode_step_count": count},
        "preprocess": {"observation_timing": "after_action",
                       "state_layout": "eef_xyz_axisangle_gripper_qpos",
                       "action_layout": "normalized_world_osc_pose_7",
                       "image_orientation": "rotate_180", "image_size": 224,
                       "orientation_conversion": "axisangle_shortest_equivalent",
                       "numeric_decimals": 6,
                       "keyframe_selection": "uniform_plus_gripper_transitions_v1"},
        "steps": [{"action_index": index, "normalized_action": action, "after_state": state}
                  for index, (action, state) in enumerate(zip(rounded_actions, rounded_states))],
        "keyframes": keyframes,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-key", default="demo_0")
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="drawer_bowl")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite demonstration bundle: {args.output}")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Use a Python environment with h5py to prepare the official HDF5 file") from exc
    digest = file_sha256(args.source)
    with h5py.File(args.source, "r") as hdf5:
        result = build_observable_demo(hdf5["data"], source_path=args.source,
                                      file_sha256=digest, demo_key=args.demo_key, task_key=args.task)
    from gpt_field.demonstration import validate_demonstration
    validate_demonstration(result, task_key=args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "file_sha256": digest,
                      "demo_key": args.demo_key, "steps": len(result["steps"]),
                      "keyframes": len(result["keyframes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
