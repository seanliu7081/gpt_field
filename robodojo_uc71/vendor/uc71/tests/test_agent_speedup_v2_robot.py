"""agent_SpeedUp_v2 ROBOT facts: pins vs the simulator's own robot asset (robot_facts_v2.json), R-5 / R-5b."""

import math

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_robot as robot
from gpt_field import robot_tool_geometry as rtg

ASSET = robot.ROBOT_FACTS["asset"]


def test_r5_r6_pinned_within_half_a_millimetre_of_the_asset():
    pins = robot.asset_vs_pins()
    assert abs(pins["R5_fingertip_reach_mm"]) <= 0.5
    assert abs(pins["R6_palm_underside_mm"]) <= 0.5
    assert max(abs(v) for v in pins["R4_pad_span_mm"]) <= 0.05
    assert abs(pins["R1_grip_site_mm"]) < 1e-6 and abs(pins["R2_camera_offset_mm"]) < 1e-6
    assert abs(pins["R3_focal_px"]) < 0.1
    assert abs(pins["R9_pad_separation_offset_mm"]) <= 0.05          # TLR-7: R9 pinned to its asset value


def test_asset_facts_agree_with_the_frozen_stl_vertices():
    """robot_facts_v2.json (compiled sim model) vs robot_tool_geometry.py (embedded STL vertices)."""
    hand = np.asarray(rtg.HAND_COLLISION_VERTICES_M)
    finger = np.asarray(rtg.FINGER_COLLISION_VERTICES_M)
    palm = (rtg.GRIP_SITE_Z_M - hand[:, 2].max()) * 1000
    tip = (finger[:, 2].max() + rtg.FINGER_BODY_Z_M - rtg.GRIP_SITE_Z_M) * 1000
    assert abs(palm - ASSET["palm_underside_behind_grip_mm"]) < 0.05
    assert abs(tip - ASSET["fingertip_reach_beyond_grip_mm"]) < 0.05
    assert abs(hand[:, 1].min() * 1000 - ASSET["hand_footprint_R1_mm"][0]) < 0.05
    assert abs(hand[:, 1].max() * 1000 - ASSET["hand_footprint_R1_mm"][1]) < 0.05
    pad_z = rtg.FINGER_BODY_Z_M + rtg.PAD_CENTER_FROM_FINGER_M[2] - rtg.GRIP_SITE_Z_M
    assert abs((pad_z - rtg.PAD_HALF_SIZE_M[2]) * 1000 - ASSET["pad_span_along_approach_mm"][0]) < 0.05
    assert ASSET["grip_site_along_R2_m"] == rtg.GRIP_SITE_Z_M


def test_spec_inequalities_and_camera():
    """R8 (TLR-7): the spec's '>= 38.2 mm' is the inner face at the reset finger q; the limit q gives 39.49 mm."""
    inner, outer = ASSET["pad_inner_face_mm"], ASSET["pad_outer_face_mm"]
    assert inner["at_reset_q"] >= 38.2 and inner["at_limit_q"] == pytest.approx(39.49, abs=0.01)
    assert inner["limit_q_m"] == 0.04 and 0.038 < inner["reset_q_m"] < 0.04
    assert outer["at_limit_q"] - inner["at_limit_q"] == pytest.approx(8.0, abs=0.01)   # pad thickness R4
    assert inner["at_limit_q"] - inner["at_reset_q"] == pytest.approx((0.04 - inner["reset_q_m"]) * 1000, abs=0.01)
    assert abs(ASSET["hand_footprint_R0_mm"][0]) <= 31.7 + 0.1     # R7 conservative box
    assert ASSET["wrist_camera"]["fovy_deg"] == 75.0
    assert ASSET["wrist_camera"]["pos_hand_m"] == [0.05, 0.0, 0.0]
    assert ASSET["joint7_range_rad"] == [-2.8973, 2.8973]
    assert robot.focal_length_px(512) == robot.FOCAL_PX_AT_512 and robot.focal_length_px(256) == 166.8


def test_palm_profile_and_footprints():
    profile = robot.palm_profile()
    assert len(profile) > 50
    assert min(p for _, p in profile) == pytest.approx(ASSET["palm_underside_behind_grip_mm"], abs=0.01)
    assert robot.palm_underside_at(0.0) == pytest.approx(31.04, abs=0.05)
    assert robot.palm_underside_at(1000.0) is None
    # TLR-12: defined over the whole hand footprint (hull end points), None just outside it
    low, high = ASSET["hand_footprint_R1_mm"]
    assert profile[0][0] == low and profile[-1][0] == high
    for xi in (low, -103.5, 100.3, high):
        assert robot.palm_underside_at(xi) is not None
    assert robot.palm_underside_at(low - 0.5) is None and robot.palm_underside_at(high + 0.5) is None
    centre = [p for xi, p in profile if abs(xi) <= 80]
    assert max(centre) - min(centre) < 0.5          # flat palm between the finger slots
    feet = robot.finger_landing_footprints()
    assert sorted(feet, key=int) == ["0", "10", "20", "30", "40"]
    # TLR-8: keys are the per-finger joint q (mm); the pads touch at q = 0.506 mm, so "0" is below the hard stop
    assert "per-finger joint q" in ASSET["finger_landing_footprints_key_semantics"]
    assert ASSET["pad_contact_finger_q_mm"] == pytest.approx(0.506, abs=0.01)
    assert ASSET["finger_landing_footprints_below_hard_stop"] == ["0"]
    for aperture, fingers in feet.items():
        assert fingers["finger1"]["lowest_beyond_grip_mm"] == pytest.approx(ASSET["fingertip_reach_beyond_grip_mm"])
        lo1, hi1 = fingers["finger1"]["R1_mm"]
        lo2, hi2 = fingers["finger2"]["R1_mm"]
        assert lo1 == pytest.approx(-hi2, abs=0.2) and hi1 == pytest.approx(-lo2, abs=0.2)  # symmetric fingers
        assert lo1 == pytest.approx(int(aperture) - 0.1, abs=0.2)                        # moves with the slider


def test_yaw_budget_and_pad_band():
    assert robot.yaw_budget_rad() == pytest.approx(1.96, abs=0.01)
    low, high = robot.pad_world_z_band_top_down(1.0)
    assert low == pytest.approx(1.0 - 0.0044) and high == pytest.approx(1.0 + 0.0116)
    assert robot.pad_separation_mm(25.39) == pytest.approx(24.39)


@pytest.mark.parametrize("start, distance_cm", [
    ((-0.20201, -0.00851, 1.176), 0.98),      # drawer_bowl ep 0
    ((-0.21124, -0.01109, 1.17425), 1.27),    # bowl_plate ep 0
    ((-0.20137, 0.0187, 1.18932), 2.25),      # libero_10 task 2 ep 0
])
def test_r5_verified_mount(start, distance_cm):
    decision = robot.mount_decision([*start, 0, 0, 0, 0, 0])
    assert decision.ruling == "R-5" and decision.base_xyz == robot.VERIFIED_MOUNT_BASE_XYZ
    assert decision.shoulder_xyz == pytest.approx((-0.66, 0.0, 1.245))
    assert decision.distance_to_verified_m * 100 == pytest.approx(distance_cm, abs=0.02)
    assert (decision.frontal_stretch_limit_m, decision.frontal_stretch_stop_m) == (0.77, 0.785)


def test_r5b_outside_the_verified_mount():
    study = np.asarray((-0.75, 0.0, 0.912)) + robot.START_OFFSET_FROM_BASE_M
    decision = robot.mount_decision([*study, 0, 0, 0, 0, 0])
    assert decision.ruling == "R-5b" and decision.frontal_allowed
    assert decision.base_xyz == pytest.approx((-0.75, 0.0, 0.912), abs=1e-5)
    assert decision.shoulder_xyz == pytest.approx((-0.75, 0.0, 1.245), abs=1e-5)
    assert (decision.frontal_stretch_limit_m, decision.frontal_stretch_stop_m) == (0.742, 0.757)
    refused = robot.mount_decision([*study, 0, 0, 0, 0, 0],
                                   rulings={**robot.RULINGS, "R-5b_start_pose_base_outside_verified_mount": False})
    assert refused.ruling == "frontal_needs_r5b" and not refused.frontal_allowed
    assert robot.VERIFIED_MOUNT_GATE_M == pytest.approx(0.090 / 2)


def test_measured_start_poses_in_facts():
    for row in robot.ROBOT_FACTS["start_poses"]:
        assert row["base_xyz"] == [-0.66, 0.0, 0.912]
        assert row["shoulder_link2_xyz"] == [-0.66, 0.0, 1.245]
        offset = np.asarray(row["start_grip_minus_base_m"]) - robot.START_OFFSET_FROM_BASE_M
        assert np.all(np.abs(offset) <= np.asarray(robot.START_OFFSET_SPREAD_M) + 1e-6)
        assert robot.mount_decision([*row["start_grip_xyz"], 0, 0, 0, 0, 0]).ruling == "R-5"
    # TLR-11: the bound covers the measured maxima (1.08, 1.85, 1.72) cm over the 10 libero_10 resets
    assert np.all(np.asarray(robot.START_OFFSET_SPREAD_M) >= np.asarray((0.0108, 0.0185, 0.0172)))


def test_rulings_and_wrist_distance():
    assert robot.RULINGS["R-1v_vertical_probe_moves_looming"] and robot.RULINGS["R-2f_focal_length_as_robot_fact"]
    assert robot.RULINGS["R-5b_start_pose_base_outside_verified_mount"]
    assert not robot.RULINGS["R-3_evaluator_tuned_constants"] and not robot.RULINGS["R-4_episode_continuation_signal"]
    d = robot.wrist_to_shoulder_m((0.1, 0.0, 1.245), (1.0, 0.0, 0.0), (-0.66, 0.0, 1.245))
    assert d == pytest.approx(0.76 - 0.204)
    assert robot.is_frontal((1, 0, 0)) and not robot.is_frontal((0, 0, -1))


def test_robot_thresholds_are_labelled():
    rows = robot.robot_thresholds()
    assert {t.label for t in rows} <= {"ROBOT", "RULING", "DD"}
    names = {t.name for t in rows}
    assert {"FINGERTIP_REACH_MM", "PALM_UNDERSIDE_MM", "FOCAL_PX_AT_512", "YAW_BUDGET_RAD"} <= names
    assert math.isclose([t for t in rows if t.name == "FOCAL_PX_AT_512"][0].value, 333.6)
    identity = robot.robot_identity()
    assert identity["robot_facts_sha256"] == robot.ROBOT_FACTS_SHA256 and len(identity["robot_facts_sha256"]) == 64
