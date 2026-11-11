"""Validated public sensor and model-output contracts; distances are metres.

Camera coordinates use OpenCV (+x right, +y down, +z forward), integer pixel
centres, and world_to_camera transforms. Scores are uncalibrated model scores.
No class accepts simulator objects, contacts, reward, or task predicates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence
from types import MappingProxyType
import math

import numpy as np


def finite_scalar(value, name: str, *, minimum=None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"invalid {name}")
    return result


def readonly_array(value, shape, name, dtype=float):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=dtype, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    result.setflags(write=False)
    return result


def transform(value, name="transform"):
    result = readonly_array(value, (4, 4), name)
    r = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must be a rigid proper transform")
    return result


def intrinsics(value):
    result = readonly_array(value, (3, 3), "intrinsics")
    if (result[0, 0] <= 0 or result[1, 1] <= 0
            or not np.allclose(result[2], [0, 0, 1], atol=1e-8, rtol=0)
            or abs(result[1, 0]) > 1e-8 or abs(result[0, 1]) > 1e-8):
        raise ValueError("intrinsics must be a positive, zero-skew pinhole matrix")
    return result


@dataclass(frozen=True)
class CameraFrame:
    name: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    observed_at_s: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("camera name is required")
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("rgb must be H x W x 3 uint8 with dimensions >= 2")
        object.__setattr__(self, "rgb", readonly_array(rgb, rgb.shape, "rgb", np.uint8))
        object.__setattr__(self, "intrinsics", intrinsics(self.intrinsics))
        object.__setattr__(self, "world_to_camera", transform(self.world_to_camera))
        object.__setattr__(self, "observed_at_s", finite_scalar(self.observed_at_s, "observed_at_s", minimum=0))


@dataclass(frozen=True)
class VisionFrame:
    frame_id: int
    captured_at_s: float
    cameras: tuple[CameraFrame, ...]
    hand_world: np.ndarray | None = None

    def __post_init__(self):
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        at = finite_scalar(self.captured_at_s, "captured_at_s", minimum=0)
        cameras = tuple(self.cameras)
        if len(cameras) < 2 or any(not isinstance(c, CameraFrame) for c in cameras):
            raise ValueError("at least two calibrated camera frames are required")
        if len({c.name for c in cameras}) != len(cameras):
            raise ValueError("camera names must be unique")
        if any(abs(c.observed_at_s - at) > 1e-6 for c in cameras):
            raise ValueError("camera frames and calibration must share one sensor timestamp")
        object.__setattr__(self, "captured_at_s", at)
        object.__setattr__(self, "cameras", cameras)
        if self.hand_world is not None:
            object.__setattr__(self, "hand_world", transform(self.hand_world, "hand_world"))


@dataclass(frozen=True)
class TargetSpec:
    key: str
    text: str
    seed_pixels: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if any(not isinstance(s, str) or not s.strip() or len(s) > 256 for s in (self.key, self.text)):
            raise ValueError("target key and text must be nonempty strings of at most 256 characters")
        seeds = {}
        for view, uv in self.seed_pixels.items():
            if not isinstance(view, str) or not view:
                raise ValueError("seed camera name is required")
            seeds[view] = tuple(readonly_array(uv, (2,), "seed pixel"))
        object.__setattr__(self, "seed_pixels", MappingProxyType(seeds))


@dataclass(frozen=True)
class Detection:
    view: str
    prompt_key: str
    mask: np.ndarray
    score: float

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.view, self.prompt_key)):
            raise ValueError("detection view and prompt key are required")
        mask = np.asarray(self.mask)
        if mask.ndim != 2 or min(mask.shape) < 2 or mask.dtype != np.bool_:
            raise ValueError("detection mask must be a two-dimensional boolean array")
        object.__setattr__(self, "mask", readonly_array(mask, mask.shape, "mask", bool))
        object.__setattr__(self, "score", finite_scalar(self.score, "score"))


@dataclass(frozen=True)
class DepthEstimate:
    view: str
    depth_m: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    score_map: np.ndarray | None = None
    source: str = "model_prediction"

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.view, self.source)):
            raise ValueError("depth view and source are required")
        # Invalid pixels are retained as invalid evidence, never converted to free space.
        if np.iscomplexobj(self.depth_m):
            raise ValueError("depth must be real")
        depth = np.array(self.depth_m, dtype=float, copy=True)
        if depth.ndim != 2 or min(depth.shape) < 2:
            raise ValueError("depth must be a two-dimensional array")
        depth.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "intrinsics", intrinsics(self.intrinsics))
        object.__setattr__(self, "world_to_camera", transform(self.world_to_camera))
        if self.score_map is not None:
            if np.iscomplexobj(self.score_map):
                raise ValueError("score_map must be real")
            scores = np.array(self.score_map, dtype=float, copy=True)
            if scores.shape != depth.shape:
                raise ValueError("score_map shape must match depth")
            scores.setflags(write=False)
            object.__setattr__(self, "score_map", scores)


class Segmenter(Protocol):
    def segment(self, frame: VisionFrame, targets: Sequence[TargetSpec]) -> Sequence[Detection]: ...


class DepthEstimator(Protocol):
    def estimate(self, frame: VisionFrame) -> Mapping[str, DepthEstimate]: ...
