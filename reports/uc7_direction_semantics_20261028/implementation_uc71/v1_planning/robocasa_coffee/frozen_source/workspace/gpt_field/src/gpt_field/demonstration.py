"""One immutable, observable-only LIBERO demonstration supplied as reference.

The reference is separate from the live episode. LIBERO's dataset builder saves
the robot state and both camera frames AFTER the indexed action. Every action
and after-state is retained; RGB is explicitly sampled into deterministic
keyframes. This module never loads simulator states, evaluates success, runs a
model, or replays an action.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .distance_tasks import resolve_distance_task
from .observation import decode_image


DEMONSTRATION_VERSION = "libero-observable-demo-v1"
KEYFRAME_SELECTION = "uniform_plus_gripper_transitions_v1"
MAX_DEMO_FRAMES = 24
MAX_DEMO_STEPS = 10000
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
_BUNDLE_KEYS = {"schema_version", "task_key", "task_prompt", "source", "preprocess", "steps", "keyframes"}
_SOURCE_KEYS = {"format", "file_name", "file_sha256", "demo_key", "suite", "task_id", "episode_step_count"}
_PREPROCESS = {
    "observation_timing": "after_action",
    "state_layout": "eef_xyz_axisangle_gripper_qpos",
    "orientation_conversion": "axisangle_shortest_equivalent",
    "action_layout": "normalized_world_osc_pose_7",
    "image_orientation": "rotate_180",
    "image_size": 224,
    "numeric_decimals": 6,
    "keyframe_selection": KEYFRAME_SELECTION,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_dict(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} contains missing or non-allowlisted fields")
    return value


def _vector(value: Any, size: int, label: str, *, bounded=False) -> list[float]:
    valid = isinstance(value, list) and len(value) == size
    if valid:
        try:
            valid = all(not isinstance(v, bool) and isinstance(v, (int, float))
                        and math.isfinite(v) for v in value)
        except OverflowError:
            valid = False
    if not valid:
        raise ValueError(f"{label} must contain {size} finite numbers")
    if bounded and any(abs(v) > 1 for v in value):
        raise ValueError(f"{label} must be normalized to [-1, 1]")
    if any(round(v, _PREPROCESS["numeric_decimals"]) != v for v in value):
        raise ValueError(f"{label} must match the declared six-decimal preprocessing")
    return value


def select_keyframe_indices(actions, max_frames: int = MAX_DEMO_FRAMES) -> tuple[int, ...]:
    """Keep endpoints and gripper-change neighborhoods; fill largest time gaps.

    Selection uses only the exported normalized actions. If gripper transitions
    alone exceed the budget, sample those candidate indices evenly, retaining
    both episode endpoints. Input may be a NumPy array or an ordinary list.
    """
    if type(max_frames) is not int or not 2 <= max_frames <= MAX_DEMO_FRAMES:
        raise ValueError("max_frames must be an integer from 2 through 24")
    try:
        values = np.asarray(actions, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("actions must be a finite N by 7 array") from exc
    if (values.ndim != 2 or values.shape[1] != 7 or not 1 <= len(values) <= MAX_DEMO_STEPS
            or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
        raise ValueError("actions must be a finite normalized N by 7 array")
    count, budget = len(values), min(max_frames, len(values))
    selected = {0, count-1}
    last_nonzero = 0
    for index, value in enumerate(values[:, 6]):
        sign = int(np.sign(value))
        if sign:
            if last_nonzero and sign != last_nonzero:
                selected.update((max(0, index-1), index))
            last_nonzero = sign
    if len(selected) > budget:
        candidates = sorted(selected)
        # Integer arithmetic makes the selection independent of np.linspace
        # rounding details; budget>=2 here, except the single-step case.
        selected = {candidates[i*(len(candidates)-1)//(budget-1)] for i in range(budget)}
    while len(selected) < budget:
        ordered = sorted(selected)
        # Break equal-size gap ties in favor of the earlier interval.
        left, right = max(zip(ordered[:-1], ordered[1:]), key=lambda pair: (pair[1]-pair[0], -pair[0]))
        selected.add((left+right)//2)
    return tuple(sorted(selected))


def _image(url: Any, label: str) -> dict:
    if not isinstance(url, str) or len(url) > 4*MAX_IMAGE_BYTES//3 + 128:
        raise ValueError(f"{label} is not a bounded PNG data URL")
    try:
        pixels = decode_image(url)
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{label} is not a valid RGB PNG data URL") from exc
    if pixels.size != (_PREPROCESS["image_size"], _PREPROCESS["image_size"]) or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label} must be a bounded 224 by 224 RGB PNG")
    return {"png_sha256": _hash(raw), "pixel_sha256": _hash(pixels.tobytes()), "png_bytes": len(raw)}


def _validate(bundle: Any, task_key: str) -> dict:
    task = resolve_distance_task(task_key)
    _exact_dict(bundle, _BUNDLE_KEYS, "demonstration")
    if bundle["schema_version"] != DEMONSTRATION_VERSION:
        raise ValueError("Unsupported demonstration schema_version")
    if bundle["task_key"] != task.key or bundle["task_prompt"] != task.prompt:
        raise ValueError("Demonstration task does not match the configured task")
    source = _exact_dict(bundle["source"], _SOURCE_KEYS, "demonstration source")
    if source["format"] != "libero_hdf5":
        raise ValueError("Demonstration source must identify LIBERO HDF5")
    name = source["file_name"]
    if (not isinstance(name, str) or not 1 <= len(name) <= 255
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.hdf5", name)):
        raise ValueError("Demonstration source file_name must be an HDF5 basename")
    if not isinstance(source["file_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", source["file_sha256"]) is None:
        raise ValueError("Demonstration source file_sha256 must be a lowercase SHA256 digest")
    if (not isinstance(source["demo_key"], str) or len(source["demo_key"]) > 64
            or re.fullmatch(r"demo_(0|[1-9][0-9]*)", source["demo_key"]) is None):
        raise ValueError("Demonstration must identify one demo_N episode")
    if source["suite"] != task.suite or type(source["task_id"]) is not int or source["task_id"] != task.task_id:
        raise ValueError("Demonstration benchmark source does not match the configured task")
    count = source["episode_step_count"]
    if type(count) is not int or not 1 <= count <= MAX_DEMO_STEPS:
        raise ValueError("Demonstration episode_step_count must be bounded and positive")
    preprocess = _exact_dict(bundle["preprocess"], set(_PREPROCESS), "demonstration preprocessing")
    if any(type(preprocess[key]) is not type(expected) or preprocess[key] != expected
           for key, expected in _PREPROCESS.items()):
        raise ValueError("Demonstration preprocessing does not match the supported after-action RGB/state convention")
    steps = bundle["steps"]
    if not isinstance(steps, list) or len(steps) != count:
        raise ValueError("Demonstration must include every source action and after-state")
    for index, step in enumerate(steps):
        _exact_dict(step, {"action_index", "normalized_action", "after_state"}, "demonstration step")
        if type(step["action_index"]) is not int or step["action_index"] != index:
            raise ValueError("Demonstration action indices must be contiguous and chronological from zero")
        _vector(step["normalized_action"], 7, "demonstration action", bounded=True)
        _vector(step["after_state"], 8, "demonstration after-state")
    frames = bundle["keyframes"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_DEMO_FRAMES:
        raise ValueError("Demonstration must contain at most 24 paired RGB keyframes")
    indices = select_keyframe_indices([step["normalized_action"] for step in steps])
    if len(frames) != len(indices):
        raise ValueError("Demonstration keyframes must match deterministic full-episode sampling")
    frame_hashes = []
    for frame, index in zip(frames, indices):
        _exact_dict(frame, {"after_action_index", "external_image", "wrist_image"}, "demonstration keyframe")
        if type(frame["after_action_index"]) is not int or frame["after_action_index"] != index:
            raise ValueError("Demonstration keyframes must be chronological deterministic selected indices")
        frame_hashes.append({"after_action_index": index,
            "external": _image(frame["external_image"], "external demo image"),
            "wrist": _image(frame["wrist_image"], "wrist demo image")})
    return {
        "schema_version": DEMONSTRATION_VERSION, "task_key": task.key, "task_prompt": task.prompt,
        "source_file_name": source["file_name"], "source_file_sha256": source["file_sha256"],
        "source_suite": source["suite"], "source_task_id": source["task_id"],
        "demo_key": source["demo_key"], "source_episode_step_count": count,
        "action_count": count, "state_count": count, "all_steps_included": True,
        "keyframe_indices": list(indices), "keyframe_count": len(indices), "image_count": 2*len(indices),
        "image_sampling": "keyframes_only", "preprocess": preprocess,
        "full_step_sha256": _hash(_canonical(steps).encode("utf-8")), "frame_hashes": frame_hashes,
    }


def validate_demonstration(bundle: Any, task_key: str = "drawer_bowl") -> dict:
    """Validate an export before writing; return independent content metadata.

    Raw bundle-byte identity is added by load_demonstration() after serialization.
    """
    return json.loads(_canonical(_validate(bundle, task_key)))


@dataclass(frozen=True, slots=True, init=False)
class DemoReference:
    """Validated immutable bytes; returned payload/metadata dictionaries are copies."""

    _bundle_bytes: bytes
    _metadata_json: str
    _message_json: str

    def __init__(self):
        raise TypeError("Use load_demonstration() to construct a validated reference")

    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    def message(self) -> dict:
        return json.loads(self._message_json)

    def write(self, path) -> None:
        """Write the exact validated source bundle bytes, preserving its hash."""
        Path(path).write_bytes(self._bundle_bytes)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Demonstration JSON contains duplicate keys")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Demonstration JSON contains nonfinite constant {value}")


def load_demonstration(path, task_key: str = "drawer_bowl") -> DemoReference:
    """Read one explicit bundle; reject hidden fields rather than filtering them.

    The HDF5 identity and episode length are provenance assertions by the
    exporter. This loader validates their form, the full exported sequence and
    image hashes; it does not independently read or replay the source HDF5.
    """
    resolve_distance_task(task_key)
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BUNDLE_BYTES + 1)
    if not raw or len(raw) > MAX_BUNDLE_BYTES:
        raise ValueError("Demonstration bundle must be nonempty and no larger than 32 MiB")
    try:
        bundle = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Demonstration bundle must be valid UTF-8 JSON") from exc
    metadata = {**_validate(bundle, task_key), "bundle_sha256": _hash(raw), "bundle_bytes": len(raw)}
    # This is one user reference message, never a live observation, execution
    # feedback record, or an assistant controller response.
    header = {
        "type": "external_demonstration", "reference": metadata,
        "not_current_episode": True, "all_actions_and_after_states_included": True,
        "images_are_sampled_keyframes": True,
        "trajectory_convention": "step t contains action[t] and the state AFTER action[t]; no before-state is invented",
        "trajectory": bundle["steps"],
    }
    content = [{"type": "input_text", "text": _canonical(header)}]
    for frame in bundle["keyframes"]:
        for camera, key in (("external", "external_image"), ("wrist", "wrist_image")):
            content.append({"type": "input_text", "text": _canonical({
                "type": "demonstration_keyframe", "after_action_index": frame["after_action_index"],
                "camera": camera, "not_current_episode": True})})
            content.append({"type": "input_image", "image_url": frame[key], "detail": "high"})
    result = object.__new__(DemoReference)
    object.__setattr__(result, "_bundle_bytes", raw)
    object.__setattr__(result, "_metadata_json", _canonical(metadata))
    object.__setattr__(result, "_message_json", _canonical({"role": "user", "content": content}))
    return result
