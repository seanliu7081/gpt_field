"""Three-view initial planning and lossless reviewed-goal contracts."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .history import require_frozen_core
from .schema import CAMERAS, validate_calibration

require_frozen_core()
from gpt_field.agent_speedup_v2_user_check_plan import variant_plan_schema, parse_plan
from gpt_field.agent_speedup_v2_user_check import plan_validation as _frozen_plan_validation
from gpt_field.agent_speedup_v2_user_check_framework import TaskContract
from gpt_field.agent_speedup_v2_user_check_spatial import build_view_frame, validate_view_frame

PLAN_VERSION = "robodojo-uc71-three-camera-plan-1"
PUBLIC_HOME_RULE = "A task counts as complete only when both arms are back at the pose they started the episode in."


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def native_plan_schema():
    schema = deepcopy(variant_plan_schema())
    part = schema["properties"]["parts"]["items"]
    part["properties"]["r"] = {"type": "integer", "minimum": 0, "maximum": 64}
    part["required"].append("r")
    return schema


def _projection(raw):
    if not isinstance(raw, dict) or set(raw) != set(native_plan_schema()["required"]):
        raise ValueError("Plan has missing or unexpected fields")
    clean = deepcopy(raw)
    if not isinstance(clean["parts"], list):
        raise ValueError("Plan parts must be a list")
    for part in clean["parts"]:
        expected = native_plan_schema()["properties"]["parts"]["items"]["required"]
        if not isinstance(part, dict) or set(part) != set(expected):
            raise ValueError("Part must contain the three declared camera cells")
        if type(part["r"]) is not int or not 0 <= part["r"] <= 64:
            raise ValueError("Right-wrist cell must be an integer in 0..64")
        del part["r"]
    if parse_plan(clean).to_json() != clean:
        raise ValueError("Plan must round-trip without text or intent changes")
    return clean


def plan_validation(raw):
    """Structural errors raise; semantic issues are retained as original evidence."""
    return _frozen_plan_validation(_projection(raw), require_native=False)


def validate_plan(raw, *, semantic=True):
    projected = _projection(raw)
    if semantic:
        check = _frozen_plan_validation(projected, require_native=False)
        if not check["valid"]:
            raise ValueError("; ".join(issue["text"] for issue in check["issues"]))
    return deepcopy(raw)


def semantic_plan(raw):
    clean = validate_plan(raw, semantic=False)
    for part in clean["parts"]:
        for name in ("x", "w", "r"):
            part.pop(name)
    return clean


class NativePlanLock:
    """Duck-typed frozen client lock; only grounding may change without review."""

    allow_auto_replan = False

    def __init__(self, plan, *, task_prompt, direction_frame):
        self._approved = validate_plan(plan)
        self._semantic_hash = canonical_sha256(semantic_plan(plan))
        self.task_contract = TaskContract.from_plan(self._approved, task_prompt=task_prompt)
        self.view_frame = validate_view_frame(direction_frame)
        if not self.view_frame["available"]:
            raise ValueError("The native pilot requires a reviewed head-camera direction frame")
        self.conflict = None

    def plan(self):
        return deepcopy(self._approved)

    def check_active(self):
        if self.conflict is not None:
            raise ValueError("Reviewed plan lock is stopped")

    def check_plan(self, plan, *, source):
        self.check_active()
        if canonical_sha256(semantic_plan(plan)) != self._semantic_hash:
            self.conflict = {"source": source, "reason": "approved_semantics_changed"}
            raise ValueError("Plan changes approved goal semantics")


def direction_frame_from_calibration(calibration):
    calibration = validate_calibration(calibration)
    head = calibration["cameras"]["head"]
    transform = np.asarray(head["world_from_optical"])
    # This conversion is only for the unchanged viewer-frame function. Delivered
    # images/calibration remain in the explicitly documented +Z optical frame.
    rotation_gl = transform[:3, :3] @ np.diag([1., -1., -1.])
    return build_view_frame({"cameras": {"external": {
        "camera_id": "head", "pose_source": head["pose_source"],
        "position_world_m": transform[:3, 3].tolist(),
        "rotation_world_from_camera": rotation_gl.tolist(),
        "intrinsics": head["intrinsics"],
    }}, "image": {"width": 640, "height": 480, "pixel_axes": "u_right_v_down",
        "pixel_centers": "integer", "source_transform": "native_isaac_rgb_unflipped_no_crop"}})


def rgb_image(value):
    if isinstance(value, str):
        prefix = "data:image/png;base64,"
        if not value.startswith(prefix):
            raise ValueError("RGB must be an inline PNG, never a remote URL")
        value = base64.b64decode(value[len(prefix):], validate=True)
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("Expected a three-channel RGB PNG")
            array = np.asarray(image).copy()
    else:
        array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape != (480, 640, 3):
        raise ValueError("Expected native RGB uint8[480,640,3]")
    return np.ascontiguousarray(array)


def png_url(value):
    array = rgb_image(value)
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _grid_url(array):
    image = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    width, height = image.size
    for col in range(1, 8):
        draw.line((col * width / 8, 0, col * width / 8, height), fill=(255, 255, 255), width=1)
    for row in range(1, 8):
        draw.line((0, row * height / 8, width, row * height / 8), fill=(255, 255, 255), width=1)
    for row in range(8):
        for col in range(8):
            xy = (col * width // 8 + 3, row * height // 8 + 2)
            label = str(row * 8 + col + 1)
            bounds = draw.textbbox(xy, label, font=font)
            draw.rectangle((bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1), fill=(0, 0, 0))
            draw.text(xy, label, font=font, fill=(255, 255, 255))
    return png_url(np.asarray(image))


def planning_images(images, *, wrist_quarter_turns=None):
    if not isinstance(images, dict) or set(images) != set(CAMERAS):
        raise ValueError("Initial planning requires exactly all three native cameras")
    turns = wrist_quarter_turns or {"left_wrist": 0, "right_wrist": 0}
    if turns != {"left_wrist": 0, "right_wrist": 0}:
        raise ValueError("This native rectangular pilot uses unrotated, explicitly labeled wrist images")
    result = []
    for index, name in enumerate(CAMERAS):
        array = rgb_image(images[name])
        label = name.replace("_", " ")
        result.extend([(f"Image {2 * index + 1}: {label}, clean native RGB, unrotated:", png_url(array)),
                       (f"Image {2 * index + 2}: {label}, same RGB with 8x8 grid, cells 1..64:", _grid_url(array))])
    return result
