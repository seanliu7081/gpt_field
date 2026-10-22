"""The only information allowed to cross from simulator to policy."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image

OBSERVATION_KEYS = frozenset({
    "observation/image", "observation/wrist_image", "observation/state", "prompt"
})
# The original four-field protocol remains the default. Arm proprioception is
# an explicit paired extension, never a route for arbitrary simulator state.
JOINT_OBSERVATION_KEYS = OBSERVATION_KEYS | frozenset({
    "observation/joint_position", "observation/joint_velocity"
})


def image_to_data_url(image: np.ndarray, size: int = 224, rotate: bool = True) -> str:
    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("camera images must be H x W x 3 uint8 RGB arrays")
    if type(size) is not int or size <= 0:
        raise ValueError("image size must be a positive integer")
    if rotate:
        pixels = pixels[::-1, ::-1]
    frame = Image.fromarray(np.ascontiguousarray(pixels))
    # Match openpi-client resize_with_pad integer truncation and floor offsets.
    # Source: packages/openpi-client/src/openpi_client/image_tools.py.
    if frame.size != (size, size):
        ratio = max(frame.width / size, frame.height / size)
        width, height = int(frame.width / ratio), int(frame.height / ratio)
        if min(width, height) < 1:
            raise ValueError("image aspect ratio is too extreme for requested output size")
        resized = frame.resize((width, height), resample=Image.Resampling.BILINEAR)
        frame = Image.new("RGB", (size, size), 0)
        frame.paste(resized, (int((size-width)/2), int((size-height)/2)))
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image(url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("images must be PNG data URLs, not external URLs or paths")
    try:
        raw = base64.b64decode(url[len(prefix):], validate=True)
        image = Image.open(io.BytesIO(raw))
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("images must encode RGB PNGs")
        image.load()
        return image
    except Exception as exc:
        raise ValueError("invalid PNG image data") from exc


def quaternion_to_axisangle(xyzw) -> tuple[float, float, float]:
    q = np.asarray(xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite numbers")
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise ValueError("observed quaternion must be unit length")
    q = q / norm
    if q[3] < 0:
        q = -q
    magnitude = float(np.linalg.norm(q[:3]))
    if magnitude < 1e-10:
        return (0.0, 0.0, 0.0)
    angle = 2 * math.atan2(magnitude, float(q[3]))
    return tuple(float(v) for v in q[:3] * angle / magnitude)


@dataclass(frozen=True)
class PolicyObservation:
    image: str
    wrist_image: str
    state: tuple[float, ...]
    prompt: str
    joint_position: tuple[float, ...] | None = None
    joint_velocity: tuple[float, ...] | None = None

    def __post_init__(self):
        if len(self.state) != 8 or any(
            isinstance(x, bool) or not isinstance(x, (int, float, np.number))
            or not math.isfinite(float(x)) for x in self.state
        ):
            raise ValueError("state must contain 8 finite numbers: xyz, axis-angle, two gripper joints")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty task instruction")
        if (self.joint_position is None) != (self.joint_velocity is None):
            raise ValueError("joint_position and joint_velocity must be supplied together")
        for name in ("joint_position", "joint_velocity"):
            values = getattr(self, name)
            if values is None:
                continue
            try:
                valid = len(values) == 7 and all(
                    not isinstance(x, (bool, np.bool_))
                    and isinstance(x, (int, float, np.number))
                    and not isinstance(x, (complex, np.complexfloating))
                    and math.isfinite(float(x)) for x in values)
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must contain 7 finite real Panda arm joint values")
            object.__setattr__(self, name, tuple(float(x) for x in values))
        decode_image(self.image)
        decode_image(self.wrist_image)
        object.__setattr__(self, "state", tuple(float(x) for x in self.state))

    def to_dict(self) -> dict:
        data = {
            "observation/image": self.image,
            "observation/wrist_image": self.wrist_image,
            "observation/state": list(self.state),
            "prompt": self.prompt,
        }
        if self.joint_position is not None:
            data.update({"observation/joint_position": list(self.joint_position),
                         "observation/joint_velocity": list(self.joint_velocity)})
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "PolicyObservation":
        if set(data) not in (OBSERVATION_KEYS, JOINT_OBSERVATION_KEYS):
            raise ValueError("policy observations require the four allowlisted fields, optionally with both arm joint fields")
        joints = {}
        if set(data) == JOINT_OBSERVATION_KEYS:
            if data["observation/joint_position"] is None or data["observation/joint_velocity"] is None:
                raise ValueError("serialized arm joint fields must contain values, not null")
            joints = {"joint_position": data["observation/joint_position"],
                      "joint_velocity": data["observation/joint_velocity"]}
        return cls(data["observation/image"], data["observation/wrist_image"],
                   tuple(data["observation/state"]), data["prompt"], **joints)

    @classmethod
    def from_libero(cls, obs: Mapping, prompt: str, image_size: int = 224, *,
                    include_arm_joints: bool = False) -> "PolicyObservation":
        # Explicit indexing: additional object, contact, reward, or predicate data is discarded.
        position = np.asarray(obs["robot0_eef_pos"], dtype=float)
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
        if position.shape != (3,) or gripper.shape != (2,):
            raise ValueError("unexpected LIBERO position or gripper state shape")
        state = (*position, *quaternion_to_axisangle(obs["robot0_eef_quat"]), *gripper)
        if type(include_arm_joints) is not bool:
            raise ValueError("include_arm_joints must be a boolean")
        joints = {}
        if include_arm_joints:
            joints = {"joint_position": obs["robot0_joint_pos"],
                      "joint_velocity": obs["robot0_joint_vel"]}
        return cls(image_to_data_url(obs["agentview_image"], image_size),
                   image_to_data_url(obs["robot0_eye_in_hand_image"], image_size),
                   tuple(state), prompt, **joints)
