"""External-camera image -> world direction map measured from the robot's own motion.

The external camera is uncalibrated: no pose, intrinsics or extrinsics are used.
At the episode start the hand makes a few short horizontal probe moves; dense
optical flow between the external frames before and after each move gives the
image motion of the hand (the fastest-moving part of the arm), and a 2x2
least-squares fit maps measured world xy displacement to image displacement.
Its inverse gives the world horizontal direction for image right/up. This is
the R-1 idea (an image Jacobian estimated online from proprioceptive motion)
applied to the external view.

Recorded check (live run 001, 83 decision-to-decision moves): median direction
error 5.5 deg for translations; moves that also re-oriented the hand fail
(the rotating gripper dominates the flow), so only pure-translation probes are
fitted.
"""

from __future__ import annotations

import math

import numpy as np

EXTERNAL_AXES_VERSION = "agent-speedup-v1-external-axes-1"
DIFF_THRESHOLD = 12            # grey levels; the static scene differs by < 3 between frames
TOP_FRACTION = 0.10            # fastest decile of changed pixels = the hand
MIN_CHANGED_PIXELS = 50
MIN_FLOW_PX = 1.0
MIN_SAMPLE_MOTION_M = 0.02
MIN_IMAGE_MOTION_PX = 3.0
MAX_RESIDUAL_DEG = 30.0
MIN_SAMPLE_ANGLE_DEG = 30.0    # the horizontal probe directions must span the plane


def _gray(rgb) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def hand_image_motion(before_rgb, after_rgb) -> dict | None:
    """Median flow (dx right, dy down, pixels) of the fastest-moving changed region, or None."""
    try:
        import cv2
    except ImportError:
        return None
    before, after = _gray(before_rgb), _gray(after_rgb)
    if before.shape != after.shape:
        raise ValueError("frames must have the same size")
    flow = cv2.calcOpticalFlowFarneback(before, after, None, 0.5, 5, 21, 5, 7, 1.5, 0)
    changed = np.abs(before.astype(int) - after.astype(int)) > DIFF_THRESHOLD
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    magnitude = np.linalg.norm(flow, axis=2)
    selected = changed & (magnitude > MIN_FLOW_PX)
    if int(selected.sum()) < MIN_CHANGED_PIXELS:
        return None
    threshold = float(np.quantile(magnitude[selected], 1.0 - TOP_FRACTION))
    pick = selected & (magnitude >= threshold)
    vector = np.median(flow[pick], axis=0)
    return {"image_shift_px": [float(vector[0]), float(vector[1])], "pixels": int(pick.sum()),
            "changed_pixels": int(selected.sum())}


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 180.0
    return math.degrees(math.acos(float(np.clip(a @ b / denominator, -1.0, 1.0))))


def _unit3(vector_xy) -> list[float]:
    vector = np.asarray(vector_xy, float)
    vector = vector / float(np.linalg.norm(vector))
    return [round(float(vector[0]), 3), round(float(vector[1]), 3), 0.0]


def fit_external_axes(samples) -> dict:
    """Fit image_shift = J @ world_xy over probe samples; return the published map or why it is unavailable.

    ``samples``: dicts with ``world_displacement_m`` (3) and ``image_shift_px`` (2, or None if unmeasured).
    """
    usable, rejected = [], []
    for sample in samples:
        world = np.asarray(sample.get("world_displacement_m"), float)[:2]
        shift = sample.get("image_shift_px")
        if shift is None:
            rejected.append("no_flow")
        elif float(np.linalg.norm(world)) < MIN_SAMPLE_MOTION_M:
            rejected.append("too_little_motion")
        elif float(np.linalg.norm(shift)) < MIN_IMAGE_MOTION_PX:
            rejected.append("no_image_motion")
        else:
            usable.append((world, np.asarray(shift, float)))
    result = {"version": EXTERNAL_AXES_VERSION, "available": False, "samples": len(usable), "rejected": rejected}
    if len(usable) < 2:
        return {**result, "reason": "fewer_than_two_usable_probe_moves"}
    spread = max(_angle_deg(a, b) if _angle_deg(a, b) <= 90 else 180.0 - _angle_deg(a, b)
                 for (a, _), (b, _) in ((p, q) for i, p in enumerate(usable) for q in usable[i + 1:]))
    if spread < MIN_SAMPLE_ANGLE_DEG:
        return {**result, "reason": "probe_directions_collinear"}
    W = np.stack([world for world, _ in usable], axis=1)        # 2 x n
    S = np.stack([shift for _, shift in usable], axis=1)        # 2 x n
    J = S @ np.linalg.pinv(W)                                    # pixels per metre
    residuals = [_angle_deg(J @ world, shift) for world, shift in usable]
    result.update(J_px_per_m=J.tolist(), residual_deg=[round(r, 1) for r in residuals])
    if abs(float(np.linalg.det(J))) < 1e-6 or max(residuals) > MAX_RESIDUAL_DEG:
        return {**result, "reason": "inconsistent_probe_flow"}
    inverse = np.linalg.inv(J)
    right, up = inverse @ np.array([1.0, 0.0]), inverse @ np.array([0.0, -1.0])
    return {**result, "available": True, "reason": "fitted",
            "image_right_world": _unit3(right), "image_up_world": _unit3(up)}


def image_shift_for(axes: dict | None, world_displacement_m) -> list[float] | None:
    """Predicted external-image shift (right, down px) of the hand for a measured world xy displacement."""
    if not axes or not axes.get("available"):
        return None
    shift = np.asarray(axes["J_px_per_m"], float) @ np.asarray(world_displacement_m, float)[:2]
    return [round(float(shift[0]), 1), round(float(shift[1]), 1)]
