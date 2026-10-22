"""agent_SpeedUp_v2 robot model: pinned ROBOT facts R1-R14, the focal length (R-2f) and the R-5 / R-5b base logic.

Spec §4.4 and §5.3. Every number here is a robot asset fact, a ruling fact or a labelled v1 design default; the
labels and derivations live in ``agent_speedup_v2_thresholds.CONSTANT_LABELS``. ``robot_facts_v2.json`` (beside
this module) is written by ``scripts/agent_speedup_v2_robot_facts.py`` from the simulator's own robot collision
geoms; the pinned values below are checked against it by ``tests/test_agent_speedup_v2_robot.py`` (R5 and R6
within 0.5 mm). Reading the robot's own asset is a ROBOT fact, not evaluator truth.

Frames: R0, R1, R2 are the columns of the right_hand rotation (state8[3:6]); R1 is the jaw (finger separation,
wrist image right), R0 is wrist image up, R2 is the approach. state8[:3] is the grip site.

R-5 (spec §5.3): the base is used only for the frontal stretch guard. At episode start base_est = start_grip -
the start offset (R14). Within the 4.5 cm gate of the verified mount (R13) the verified base is used exactly and
the v1 guards are unchanged (R-5). Otherwise, with R-5b granted, base_est is used, the shoulder is base_est +
0.333 m z, and the frontal guards are tightened by the 2.8 cm spread bound. The model never receives a base
direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .agent_speedup_v2_thresholds import Threshold

ROBOT_VERSION = "agent-speedup-v2-robot-v2"
FACTS_PATH = Path(__file__).resolve().with_name("robot_facts_v2.json")

RULINGS = {
    "R-1_online_image_jacobian": True,
    "R-1x_external_image_axes_from_own_motion": True,
    "R-1v_vertical_probe_moves_looming": True,
    "R-2_wrist_mount_facts": True,
    "R-2f_focal_length_as_robot_fact": True,
    "R-3_evaluator_tuned_constants": False,
    "R-4_episode_continuation_signal": False,
    "R-5_robot_base_position": True,
    "R-5b_start_pose_base_outside_verified_mount": True,
}
RULINGS_GRANTED_ON = {"R-1v_vertical_probe_moves_looming": "2026-10-18", "R-2f_focal_length_as_robot_fact": "2026-10-18",
                      "R-5b_start_pose_base_outside_verified_mount": "2026-10-18"}

# R1, R2, R3
GRIP_SITE_ALONG_R2_M = 0.097
WRIST_CAMERA_ALONG_R0_M = 0.05
FOCAL_PX_AT_512 = 333.6
FOCAL_REFERENCE_IMAGE_PX = 512
# R4, R5, R6, R9
PAD_SPAN_ALONG_APPROACH_MM = (-11.6, 4.4)
PAD_LENGTH_R0_MM = 16.0
PAD_THICKNESS_R1_MM = 8.0
FINGERTIP_REACH_MM = 9.3
PALM_UNDERSIDE_MM = 31.0
PAD_SEPARATION_OFFSET_MM = 1.0
# R10, R11
TRANSLATION_PER_UNIT_M = 0.05
ROTATION_PER_UNIT_RAD = 0.5
ROTATION_CAP_RAD_PER_STEP = 0.15
CONTROL_HZ = 20
GRIPPER_RAMP_PER_SUBSTEP = 0.01
# R12
JOINT7_LIMIT_RAD = 2.8973
NOMINAL_START_Q7_RAD = math.pi / 4
START_Q7_PERTURBATION_RAD = 0.15
# R13, R14
VERIFIED_MOUNT_BASE_XYZ = (-0.66, 0.0, 0.912)
SHOULDER_ABOVE_BASE_M = 0.333
WRIST_FROM_GRIP_M = 0.204
FULL_STRETCH_M = 0.804
START_OFFSET_FROM_BASE_M = (0.4537, 0.0002, 0.2655)
START_OFFSET_SPREAD_M = (0.011, 0.019, 0.018)
# R-5 / R-5b (spec §5.3)
VERIFIED_MOUNT_GATE_M = 0.045
FRONTAL_STRETCH_LIMIT_M = 0.77
FRONTAL_STRETCH_STOP_M = 0.785
UNVERIFIED_GUARD_TIGHTENING_M = 0.028
FRONTAL_MAX_VERTICAL = 0.5


def load_facts(path=FACTS_PATH):
    text = Path(path).read_text()
    return json.loads(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


ROBOT_FACTS, ROBOT_FACTS_SHA256 = load_facts()


def yaw_budget_rad():
    """R12: net hand-yaw budget about the approach from the start orientation."""
    return JOINT7_LIMIT_RAD - NOMINAL_START_Q7_RAD - START_Q7_PERTURBATION_RAD


def focal_length_px(image_px):
    """R3 under R-2f: the wrist camera focal length at ``image_px`` square images."""
    return FOCAL_PX_AT_512 * float(image_px) / FOCAL_REFERENCE_IMAGE_PX


def pad_world_z_band_top_down(grip_z_m):
    """R4: pad world z interval for a straight-down hand: [g - 4.4 mm, g + 11.6 mm]."""
    low, high = PAD_SPAN_ALONG_APPROACH_MM
    return (grip_z_m - high / 1000.0, grip_z_m - low / 1000.0)


def pad_separation_mm(finger_gap_mm):
    """R9: pad separation from the measured finger gap (q1 - q2)."""
    return float(finger_gap_mm) - PAD_SEPARATION_OFFSET_MM


def finger_landing_footprints():
    """P16 input: each finger's lowest-band footprint keyed by the PER-FINGER joint q in mm (finger gap = 2 q;
    see ``finger_landing_footprints_key_semantics`` and ``..._below_hard_stop`` in robot_facts_v2.json)."""
    return ROBOT_FACTS["asset"]["finger_landing_footprints_by_finger_q_mm"]


def palm_profile():
    """R6: [(xi along R1 mm, palm underside behind the grip mm)] from the hand collision hull."""
    return [tuple(row) for row in ROBOT_FACTS["asset"]["palm_profile_mm"]["xi_R1_mm_and_p_mm"]]


def palm_underside_at(xi_mm):
    """Linear interpolation of the palm profile p(xi); None outside the hand footprint (hand_footprint_R1_mm).
    Inside the footprint but beyond the profile's sampled ends the nearest end value is used."""
    rows = palm_profile()
    if not rows:
        return None
    low, high = ROBOT_FACTS["asset"]["hand_footprint_R1_mm"]
    xs = [row[0] for row in rows]
    if xi_mm < min(low, xs[0]) or xi_mm > max(high, xs[-1]):
        return None
    return float(np.interp(xi_mm, xs, [row[1] for row in rows]))     # np.interp clamps to the end values


def robot_thresholds():
    """The pinned ROBOT facts as Threshold records (the run's threshold_set trace event)."""
    from .agent_speedup_v2_thresholds import CONSTANT_LABELS
    out = []
    for name in ("GRIP_SITE_ALONG_R2_M", "WRIST_CAMERA_ALONG_R0_M", "FOCAL_PX_AT_512", "PAD_LENGTH_R0_MM",
                 "PAD_THICKNESS_R1_MM", "FINGERTIP_REACH_MM", "PALM_UNDERSIDE_MM", "PAD_SEPARATION_OFFSET_MM",
                 "ROTATION_CAP_RAD_PER_STEP", "JOINT7_LIMIT_RAD", "WRIST_FROM_GRIP_M", "FULL_STRETCH_M",
                 "VERIFIED_MOUNT_GATE_M", "FRONTAL_STRETCH_LIMIT_M", "FRONTAL_STRETCH_STOP_M"):
        entry = CONSTANT_LABELS["agent_speedup_v2_robot." + name]
        out.append(Threshold(name, globals()[name], entry["label"], entry["derivation"]))
    out.append(Threshold("YAW_BUDGET_RAD", yaw_budget_rad(), "ROBOT",
                         "R12: JOINT7_LIMIT_RAD - NOMINAL_START_Q7_RAD - START_Q7_PERTURBATION_RAD"))
    return out


@dataclass(frozen=True)
class MountDecision:
    """R-5 / R-5b decision for one episode (computed once from the start state8)."""
    ruling: str                      # "R-5" (verified mount) | "R-5b" (start-pose base) | "frontal_needs_r5b"
    base_xyz: tuple | None
    shoulder_xyz: tuple | None
    base_estimate_xyz: tuple
    distance_to_verified_m: float
    frontal_allowed: bool
    frontal_stretch_limit_m: float | None
    frontal_stretch_stop_m: float | None
    thresholds: tuple

    def to_dict(self):
        data = asdict(self)
        data["thresholds"] = [t.to_dict() if isinstance(t, Threshold) else t for t in self.thresholds]
        return data


def mount_decision(start_state8, *, rulings=None):
    rulings = RULINGS if rulings is None else rulings
    grip = np.asarray(start_state8[:3], float)
    estimate = grip - np.asarray(START_OFFSET_FROM_BASE_M, float)
    verified = np.asarray(VERIFIED_MOUNT_BASE_XYZ, float)
    distance = float(np.linalg.norm(estimate - verified))
    gate = Threshold("VERIFIED_MOUNT_GATE_M", VERIFIED_MOUNT_GATE_M, "RULING",
                     "half the distance to the nearest other LIBERO mount (spec §5.3)")
    est = tuple(round(float(v), 5) for v in estimate)
    if distance <= VERIFIED_MOUNT_GATE_M and rulings.get("R-5_robot_base_position"):
        shoulder = verified + [0.0, 0.0, SHOULDER_ABOVE_BASE_M]
        return MountDecision("R-5", tuple(float(v) for v in verified), tuple(round(float(v), 5) for v in shoulder),
                             est, round(distance, 5), True, FRONTAL_STRETCH_LIMIT_M, FRONTAL_STRETCH_STOP_M, (gate,))
    if rulings.get("R-5b_start_pose_base_outside_verified_mount"):
        shoulder = estimate + [0.0, 0.0, SHOULDER_ABOVE_BASE_M]
        tight = Threshold("UNVERIFIED_GUARD_TIGHTENING_M", UNVERIFIED_GUARD_TIGHTENING_M, "RULING",
                          "R-5b: guards tightened by the start-pose spread bound")
        return MountDecision("R-5b", est, tuple(round(float(v), 5) for v in shoulder), est, round(distance, 5), True,
                             round(FRONTAL_STRETCH_LIMIT_M - UNVERIFIED_GUARD_TIGHTENING_M, 4),
                             round(FRONTAL_STRETCH_STOP_M - UNVERIFIED_GUARD_TIGHTENING_M, 4), (gate, tight))
    return MountDecision("frontal_needs_r5b", None, None, est, round(distance, 5), False, None, None, (gate,))


def wrist_to_shoulder_m(position, approach_axis, shoulder_xyz):
    """Distance of the wrist point (grip - 0.204 m along the approach axis) from the shoulder (R13)."""
    wrist = np.asarray(position, float) - WRIST_FROM_GRIP_M * np.asarray(approach_axis, float)
    return float(np.linalg.norm(wrist - np.asarray(shoulder_xyz, float)))


def is_frontal(approach_axis):
    return abs(float(np.asarray(approach_axis, float)[2])) < FRONTAL_MAX_VERTICAL


def asset_vs_pins():
    """Differences (mm) between the pinned R5/R6 values and the asset measurement in robot_facts_v2.json."""
    asset = ROBOT_FACTS["asset"]
    return {"R5_fingertip_reach_mm": FINGERTIP_REACH_MM - asset["fingertip_reach_beyond_grip_mm"],
            "R6_palm_underside_mm": PALM_UNDERSIDE_MM - asset["palm_underside_behind_grip_mm"],
            "R9_pad_separation_offset_mm": PAD_SEPARATION_OFFSET_MM - asset["pad_separation_offset_mm"]["at_limit_q"],
            "R4_pad_span_mm": [a - b for a, b in zip(PAD_SPAN_ALONG_APPROACH_MM, asset["pad_span_along_approach_mm"])],
            "R1_grip_site_mm": (GRIP_SITE_ALONG_R2_M - asset["grip_site_along_R2_m"]) * 1000.0,
            "R2_camera_offset_mm": (WRIST_CAMERA_ALONG_R0_M - asset["wrist_camera"]["pos_hand_m"][0]) * 1000.0,
            "R3_focal_px": FOCAL_PX_AT_512 - asset["wrist_camera"]["f_px_at_512"]}


def robot_identity():
    """Manifest block: versions, rulings, facts file hash and the pinned facts."""
    return {"version": ROBOT_VERSION, "rulings": dict(RULINGS), "rulings_granted_on": dict(RULINGS_GRANTED_ON),
            "robot_facts_file": FACTS_PATH.name, "robot_facts_sha256": ROBOT_FACTS_SHA256,
            "robot_facts_version": ROBOT_FACTS.get("version"), "focal_px_at_512": FOCAL_PX_AT_512,
            "yaw_budget_rad": round(yaw_budget_rad(), 4), "asset_vs_pins": asset_vs_pins()}
