"""Allowlisted RGB/calibration bridge and strict offline observation manifests.

No simulator is constructed or imported. Existing transport images are mirrored
back to their upright raster exactly as StepView does. A manifest's explicit
camera matrices instead use the OpenCV convention documented in types.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping

import numpy as np
from PIL import Image

from .types import CameraFrame, TargetSpec, VisionFrame, readonly_array


MANIFEST_VERSION = "foundation-vision-input-1"
SOURCE_KINDS = frozenset({"saved_public_observations", "synthetic_offline_fixture", "user_rgb"})


def _keys(value, required, optional=(), *, name):
    if not isinstance(value, Mapping) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError(f"{name} requires {sorted(required)}; optional {sorted(optional)}; extra fields are forbidden")


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates,
                      parse_constant=reject_constant)


def frame_from_step_view(view, geometry, frame_id: int, captured_at_s: float) -> VisionFrame:
    """Convert only public, synchronized robot/RGB fields; never access env/sim.

The timestamp is supplied by the observation owner. This validates wrist
calibration against the accompanying measured state, but cannot establish that
an externally supplied timestamp really came from the sensor.
    """
    from ..astranomy_agentic_geometry import wrist_camera_pose_from_observation
    from ..robocasa_user_check_geometry import validate_geometry_metadata
    from ..robot_tool_geometry import _rotation_matrix

    metadata = validate_geometry_metadata(geometry)
    state = readonly_array(view.state, (8,), "public state8")
    wrist = metadata["cameras"]["wrist"]
    if wrist["pose_source"] != "observed_robot_pose_fixed_mount":
        raise ValueError("wrist calibration must use the same observed robot state")
    wrist_position, wrist_rotation = wrist_camera_pose_from_observation(state)
    if (not np.allclose(wrist["position_world_m"], wrist_position, atol=1e-6, rtol=0)
            or not np.allclose(wrist["rotation_world_from_camera"], wrist_rotation, atol=1e-6, rtol=0)):
        raise ValueError("wrist calibration does not match the measured state8")
    expected_shape = (metadata["image"]["height"], metadata["image"]["width"], 3)
    flip_axes = np.diag([1., -1., -1.])
    cameras = []
    for name, rgb in (("external", view.external_rgb_upright), ("wrist", view.wrist_rgb_upright)):
        pixels = np.asarray(rgb)
        if pixels.shape != expected_shape:
            raise ValueError("RGB raster dimensions do not match camera calibration")
        camera = metadata["cameras"][name]
        r_world_camera = np.asarray(camera["rotation_world_from_camera"], dtype=float)
        rotation = flip_axes @ r_world_camera.T
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = -rotation @ np.asarray(camera["position_world_m"], dtype=float)
        fx, fy = camera["focal_length_px"]
        cx, cy = camera["principal_point_px"]
        k = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        cameras.append(CameraFrame(name, pixels, k, extrinsic, captured_at_s))
    hand_world = np.eye(4)
    hand_world[:3, :3] = _rotation_matrix(state[3:6])
    # The public position is grip_site. Recover the actual right_hand origin.
    hand_world[:3, 3] = state[:3] - hand_world[:3, :3] @ np.asarray(metadata["gripper"]["hand_to_grip_site_m"])
    return VisionFrame(frame_id, captured_at_s, tuple(cameras), hand_world)


def frame_from_observation(observation, geometry, frame_id: int, captured_at_s: float) -> VisionFrame:
    """Read the existing four/six-field public observation wire format only."""
    from ..observation import PolicyObservation, decode_image

    if isinstance(observation, Mapping):
        observation = PolicyObservation.from_dict(observation)
    if not isinstance(observation, PolicyObservation):
        raise TypeError("observation must be a public PolicyObservation or its allowlisted dictionary")
    view = SimpleNamespace(state=observation.state,
        external_rgb_upright=np.ascontiguousarray(np.asarray(decode_image(observation.image))[:, ::-1]),
        wrist_rgb_upright=np.ascontiguousarray(np.asarray(decode_image(observation.wrist_image))[:, ::-1]))
    return frame_from_step_view(view, geometry, frame_id, captured_at_s)


@dataclass(frozen=True)
class LoadedManifest:
    frames: tuple[VisionFrame, ...]
    targets: tuple[TargetSpec, ...]
    source_kind: str
    source_hashes: dict[str, str]
    manifest_path: str


def load_manifest(path: str | Path) -> LoadedManifest:
    """Load/validate all inputs before model inference; files are never rewritten.

Two frame formats are supported:
1. Explicit OpenCV camera matrices with upright RGB file paths.
2. An existing public observation JSON and its separately saved calibration.
All file paths are explicit and relative to the manifest unless absolute.
    """
    path = Path(path).resolve()
    data = _read_json(path)
    _keys(data, {"schema_version", "source_kind", "targets", "frames"}, name="manifest")
    if data["schema_version"] != MANIFEST_VERSION or data["source_kind"] not in SOURCE_KINDS:
        raise ValueError("unsupported manifest schema or source_kind")
    if not isinstance(data["targets"], list):
        raise ValueError("targets must be a list (empty is allowed for depth-only evaluation)")
    targets = []
    for raw in data["targets"]:
        _keys(raw, {"key", "text"}, {"seed_pixels"}, name="target")
        targets.append(TargetSpec(**raw))
    if len({target.key for target in targets}) != len(targets):
        raise ValueError("target keys must be unique")
    if not isinstance(data["frames"], list) or not data["frames"]:
        raise ValueError("frames must be a nonempty list")
    hashes = {str(path): file_sha256(path)}

    def source(raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("input file path must be a nonempty string")
        resolved = (path.parent / raw).resolve()
        if not resolved.is_file():
            raise ValueError(f"input file is unavailable: {resolved}")
        hashes[str(resolved)] = file_sha256(resolved)
        return resolved

    frames = []
    for raw in data["frames"]:
        if not isinstance(raw, Mapping):
            raise ValueError("frame must be an object")
        if "cameras" in raw:
            _keys(raw, {"frame_id", "captured_at_s", "cameras"}, {"hand_world"}, name="frame")
            if not isinstance(raw["cameras"], list):
                raise ValueError("frame cameras must be a list")
            cameras = []
            for camera in raw["cameras"]:
                _keys(camera, {"name", "rgb_path", "intrinsics", "world_to_camera", "observed_at_s"}, name="camera")
                rgb_path = source(camera["rgb_path"])
                with Image.open(rgb_path) as image:
                    if image.mode != "RGB":
                        raise ValueError("input images must be RGB; implicit channel conversion is forbidden")
                    rgb = np.array(image)
                cameras.append(CameraFrame(camera["name"], rgb, camera["intrinsics"],
                                           camera["world_to_camera"], camera["observed_at_s"]))
            frame = VisionFrame(raw["frame_id"], raw["captured_at_s"], tuple(cameras), raw.get("hand_world"))
        else:
            _keys(raw, {"frame_id", "captured_at_s", "observation_path", "geometry_path"}, name="public-observation frame")
            observation = _read_json(source(raw["observation_path"]))
            geometry = _read_json(source(raw["geometry_path"]))
            frame = frame_from_observation(observation, geometry, raw["frame_id"], raw["captured_at_s"])
        if frames and (frame.frame_id <= frames[-1].frame_id or frame.captured_at_s <= frames[-1].captured_at_s):
            raise ValueError("frame IDs and sensor timestamps must be strictly increasing")
        if frames and tuple(c.name for c in frame.cameras) != tuple(c.name for c in frames[0].cameras):
            raise ValueError("camera identity/order must remain constant within a sequence")
        by_name = {c.name: c for c in frame.cameras}
        for target in targets:
            for name, (u, v) in target.seed_pixels.items():
                if name not in by_name:
                    raise ValueError("target seed references an unknown camera")
                height, width = by_name[name].rgb.shape[:2]
                if not (0 <= u < width and 0 <= v < height):
                    raise ValueError("target seed lies outside the RGB raster")
        frames.append(frame)
    return LoadedManifest(tuple(frames), tuple(targets), data["source_kind"], hashes, str(path))
