"""Derived RGB diagnostics for a disputed visual correspondence.

Labels show the tracker's claims, never a semantic mask or privileged geometry.
Original camera inputs are preserved; healthy references add no image overhead.
"""
from __future__ import annotations

import base64
import io
import json
import math

from PIL import ImageDraw

from .observation import decode_image


REPAIR_REASONS = frozenset((
    "controlled_feature_unresponsive", "visual_response_budget_exhausted",
    "reference_motion_unresponsive", "reference_motion_budget_exhausted",
    "reference_repair_required", "rejected_reference_reused",
))
LABEL = ("Derived external RGB reference diagnostic. M=claimed controlled feature; "
         "T=claimed stationary anchor; G=independent tool when supplied. These "
         "labels are tracker estimates, not verified object identity. A cross is "
         "the last unreliable estimate. Inspect the original views too. If M is "
         "on a fixed base/background rather than the moving tool/payload, explicitly "
         "seed a different visible physical part; retaining or renaming it cannot "
         "repair the correspondence.")


def _visual(context):
    return (context or {}).get("visual_feedback", {}).get("visual", {})


def reference_diagnostic_needed(context):
    visual = _visual(context)
    health = visual.get("reference_health") or {}
    return bool(health.get("repair_required") or visual.get("reference_repair_required")
                or visual.get("external_status") in REPAIR_REASONS
                or (context or {}).get("refresh_reason") in REPAIR_REASONS)


def _point(value, width, height):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not 0 <= x <= 1 for x in value):
        return None
    return (round(value[0]*(width-1)), round(value[1]*(height-1)))


def reference_diagnostic_image(observation, context):
    """Return an annotated copy of current external RGB, or None when unneeded."""
    if not reference_diagnostic_needed(context):
        return None
    visual = _visual(context)
    tracks = visual.get("tracking") or {}
    rejected = (visual.get("reference_health") or {}).get("rejected_current_uv")
    if not tracks.get("moving") and rejected is not None:
        tracks = {**tracks, "moving": {"tracked_point_uv": rejected, "reliable": False}}
    frame = decode_image(observation["observation/image"]).copy()
    draw = ImageDraw.Draw(frame)
    drawn = False
    for key, letter, color in (("moving", "M", (255, 40, 210)),
                               ("target", "T", (30, 230, 255)),
                               ("tool", "G", (255, 225, 20))):
        track = tracks.get(key) or {}
        point = _point(track.get("tracked_point_uv"), frame.width, frame.height)
        if point is None:
            continue
        x, y = point
        if track.get("reliable"):
            draw.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2)
        else:
            draw.line((x-5, y-5, x+5, y+5), fill=color, width=2)
            draw.line((x-5, y+5, x+5, y-5), fill=color, width=2)
        tx, ty = min(frame.width-12, x+7), max(0, y-12)
        draw.rectangle((tx-1, ty-1, tx+10, ty+11), fill=(0, 0, 0))
        draw.text((tx, ty), letter, fill=color)
        drawn = True
    if not drawn:
        return None
    stream = io.BytesIO()
    frame.save(stream, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode("ascii")


def append_reference_diagnostic(payload, observation, context):
    """Append to this request's observation; preserve raw images and old history."""
    diagnostic = reference_diagnostic_image(observation, context)
    if diagnostic is None:
        return payload
    for message in reversed(payload["input"]):
        content = message.get("content", [])
        if message.get("role") != "user" or not content:
            continue
        try:
            record = json.loads(content[0].get("text", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict) and record.get("type") == "observation":
            if not any(part.get("text") == LABEL for part in content):
                content.extend([{"type": "input_text", "text": LABEL},
                                {"type": "input_image", "image_url": diagnostic}])
            return payload
    raise ValueError("Reference diagnostic requires the current observation message")
