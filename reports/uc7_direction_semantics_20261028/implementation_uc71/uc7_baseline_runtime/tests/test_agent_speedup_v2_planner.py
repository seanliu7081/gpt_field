"""T1 tests of agent_speedup_v2_planner (contract §C7): truth tables on synthetic HeightFields and PartRecords.

No renderer, no simulator, no model. Every scene is built from geometry only (support plane, disks, ridges, walls,
blocks); the part records carry what the scene builder would measure (top, support, width per jaw yaw, elongation,
D11 circle, region). Numbers follow spec §7.1 and the contract's §C7 truth tables.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_planner as pl
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field.agent_speedup_v2_thresholds import Threshold
from gpt_field.agent_speedup_v2_types import (AXIS_INCONSISTENT, HeightField, HeldFrame, KeepOut, Meas, PartRecord,
                                              RobotModel, YAW_GRID_RAD, hf_code, rect_polygon)
from gpt_field.contact_skills import matrix_to_quaternion_xyzw

ROBOT = RobotModel()
SUPPORT = 0.900
T_F = ROBOT.t_f_m
LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
MODULE = Path(pl.__file__)


# ------------------------------------------------------------------------------------------------ builders
def rotvec_from_matrix(R):
    q = matrix_to_quaternion_xyzw(R)
    v = np.asarray(q[:3], float)
    s = float(np.linalg.norm(v))
    return np.zeros(3) if s < 1e-12 else v / s * (2.0 * math.atan2(s, float(q[3])))


def top_down_R(jaw_angle):
    jaw = np.array([math.cos(jaw_angle), math.sin(jaw_angle), 0.0])
    approach = np.array([0.0, 0.0, -1.0])
    return np.column_stack([np.cross(jaw, approach), jaw, approach])


def state8(grip, jaw_angle, gap_mm):
    q = gap_mm / 2000.0
    return [*grip, *rotvec_from_matrix(top_down_R(jaw_angle)), q, -q]


def inputs(*, jaw=-math.pi / 2, g_open=79.9, sigma_align=pl.SIGMA_ALIGN_PRIOR_M, budget=(1.96, 1.96),
           grip=(0.0, 0.0, 1.10)):
    pose = ROBOT.pose(state8(grip, jaw, g_open))
    return pl.PlannerInputs(ROBOT, g_open, sigma_align, {1: budget[0], -1: budget[1]}, pose, {}, None)


def field(*, z=SUPPORT, sigma=0.001, source="kappa", centre=(0.0, 0.0), half=0.3):
    hf = HeightField.empty(centre, half_extent_m=half)
    hf.z[:] = z
    hf.sigma[:] = sigma
    hf.known[:] = True
    hf.source[:] = hf_code(source)
    hf.stamp[:] = 0
    return hf


def stamp(hf, poly, z, *, sigma=0.001, source="kappa"):
    cells = hf.cells_in_polygon(poly)
    iy, ix = cells[:, 0], cells[:, 1]
    hf.z[iy, ix], hf.sigma[iy, ix], hf.known[iy, ix], hf.source[iy, ix] = z, sigma, True, hf_code(source)
    return cells


def unknown(hf, poly):
    cells = hf.cells_in_polygon(poly)
    iy, ix = cells[:, 0], cells[:, 1]
    hf.z[iy, ix] = hf.sigma[iy, ix] = np.nan
    hf.known[iy, ix], hf.source[iy, ix] = False, 0
    return cells


def disk(centre, radius, n=64):
    a = np.arange(n) * 2.0 * math.pi / n
    return np.asarray(centre, float) + radius * np.column_stack([np.cos(a), np.sin(a)])


def octagon(centre, across_flats):
    radius = across_flats / 2.0 / math.cos(math.pi / 8.0)
    a = math.pi / 8.0 + np.arange(8) * math.pi / 4.0
    return np.asarray(centre, float) + radius * np.column_stack([np.cos(a), np.sin(a)])


def widths_bar(across, along, axis_angle, sigma=0.001):
    rows = []
    for k in range(12):
        psi = k * YAW_GRID_RAD
        s, c = abs(math.sin(psi - axis_angle)), abs(math.cos(psi - axis_angle))
        w = min(across / s if s > 1e-9 else math.inf, along / c if c > 1e-9 else math.inf)
        rows.append((psi, Meas(w, sigma, "kappa")))
    return tuple(rows)


def widths_const(w, sigma):
    return tuple((k * YAW_GRID_RAD, Meas(w, sigma, "kappa")) for k in range(12))


def part(pid, *, xyz, top, sigma_h=0.002, support=SUPPORT, move="free", region=None, cells=None, widths=None,
         elongation=None, circle=None, roles=("grip",), status="registered", provisional=False, cov_sigma=0.002):
    return PartRecord(pid=pid, text=f"part {pid}", move_prior=move, roles=frozenset(roles), xyz=np.array(xyz, float),
                      cov=np.eye(3) * cov_sigma ** 2, top_z=Meas(top, sigma_h, "kappa"),
                      support_z=None if support is None else Meas(support, 0.001, "kappa"), region_xy=region,
                      region_cells=cells, width_by_yaw=widths, elongation=elongation, circle=circle, status=status,
                      registration_step=0 if (status == "registered" or provisional) else None, sigma_h_m=sigma_h,
                      provisional=provisional)


def lever_scene(sigma_h=0.0025):
    """An elongated rotary part (24.3 mm ridge along world x, top 0.955) over a 47.6 mm disk (top 0.918) on the
    support; the pointed point is 1 cm along the ridge from the disk centre's projection."""
    hf = field()
    stamp(hf, disk((0.0, 0.0), 0.0476), 0.918)
    ridge = rect_polygon((0.0, 0.004), (1.0, 0.0), 0.0475, 0.01215)
    cells = stamp(hf, ridge, 0.955)
    p = part("A", xyz=(0.010, 0.004, 0.955), top=0.955, sigma_h=sigma_h, support=0.918, move="hinged", region=ridge,
             cells=cells, widths=widths_bar(0.0243, 0.095, 0.0), elongation=(np.array([1.0, 0.0]), 0.9),
             circle=(np.array([0.0, 0.0]), 0.0476, 0.5))
    return hf, p


def thin_part_scene(sigma_h=0.001, *, plane_filled=True):
    """A thin part of a free object (13 mm thick along x, elongated along y, top 1.0345) beside a level body (top
    1.0331, 72 mm across, its edge 15 mm beyond the hand footprint); the support around is D7a plane-filled."""
    hf = field(source="plane" if plane_filled else "kappa")
    body_poly = disk((0.0, 0.0826), 0.036)
    body_cells = stamp(hf, body_poly, 1.0331)
    thin = rect_polygon((0.0, 0.016), (0.0, 1.0), 0.028, 0.0065)
    cells = stamp(hf, thin, 1.0345)
    c = part("C", xyz=(0.0, 0.0, 1.0345), top=1.0345, sigma_h=sigma_h, support=SUPPORT, move="free", region=thin,
             cells=cells, widths=widths_bar(0.013, 0.056, math.pi / 2.0), elongation=(np.array([0.0, 1.0]), 0.9))
    d = part("D", xyz=(0.0, 0.0826, 1.0331), top=1.0331, support=SUPPORT, move="free", region=body_poly,
             cells=body_cells, widths=widths_const(0.072, 0.001), roles=("carried",))
    return hf, c, d


def wall_scene(*, block=False, surround=False):
    """A shallow circular container: wall ring r 48-53 mm, rim top 0.930, floor 0.905; the seed on the rim at +x."""
    hf = field()
    outer = stamp(hf, disk((0.0, 0.0), 0.053), 0.930)
    inner = stamp(hf, disk((0.0, 0.0), 0.048), 0.905)
    inner_set = {tuple(c) for c in inner}
    ring = np.array([c for c in outer if tuple(c) not in inner_set])
    if block:
        stamp(hf, rect_polygon((0.135, 0.0), (1.0, 0.0), 0.015, 0.03), 0.955)     # beyond the outer column
    if surround:
        cells = hf.cells_in_polygon(disk((0.0, 0.0), 0.2))
        centres = hf.origin_xy + cells[:, ::-1] * hf.cell_m
        far = cells[np.linalg.norm(centres, axis=1) >= 0.12]
        hf.z[far[:, 0], far[:, 1]] = 0.955
    widths = tuple((k * YAW_GRID_RAD, Meas(0.005 if k == 0 else 0.06, 0.0005, "kappa")) for k in range(12))
    b = part("B", xyz=(0.0505, 0.0, 0.930), top=0.930, support=SUPPORT, move="free", cells=ring, widths=widths,
             circle=(np.array([0.0, 0.0]), 0.0505, 0.5))
    return hf, b


def body_scene():
    """A free body 66 mm across flats (top 1.033) with two protrusions along y (tops 1.032 and 1.033)."""
    hf = field()
    body_poly = octagon((0.0, 0.0), 0.066)
    cells = stamp(hf, body_poly, 1.033)
    stamp(hf, rect_polygon((0.0, 0.047), (0.0, 1.0), 0.013, 0.006), 1.032)
    stamp(hf, rect_polygon((0.0, -0.047), (0.0, 1.0), 0.013, 0.006), 1.033)
    d = part("D", xyz=(0.0, 0.0, 1.033), top=1.033, support=SUPPORT, move="free", region=body_poly, cells=cells,
             widths=widths_const(0.066, 0.00075))
    return hf, d


def held_frame(*, pid="D", grip_pid="C", offset_h=(0.08, 0.0, 0.0), b_low=0.1219, r_obj=0.04, sigma_offset=0.003,
               vacated=None):
    return HeldFrame(pid=pid, grip_pid=grip_pid, grasp_pose=tuple(state8((0.0, 0.0, 1.02), 0.0, 14.0)),
                     offset_h=np.array(offset_h, float), vacated_cells=vacated, sigma_offset_m=sigma_offset,
                     b_low=Meas(b_low, 0.001, "touch"), r_obj_m=r_obj, extent_axes=None, tilt_deg=None,
                     wrist_visible=True, silhouette=None, ext_template=None, gap_ref_mm=14.0, sigma_carry_mm=None,
                     support_z_pick=Meas(SUPPORT, 0.001, "kappa"), g_close=1.0219)


def as_json(value):
    return json.dumps(pl.jsonable_result(value), allow_nan=False)


# ------------------------------------------------------------------------------------------------ module rules
def test_module_lint_audit_and_lazy_geometry():
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(MODULE, vocab) == []
    assert th.literal_audit(MODULE) == []
    tree = ast.parse(MODULE.read_text())
    assert th.firewall_violations(tree, MODULE.stem) == []
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("geometry" in ast.unparse(n) for n in top_level)
    lazy = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            and "geometry" in ast.unparse(n)]
    assert lazy == ["from . import agent_speedup_v2_geometry as geometry"]
    assert "agent_speedup_v2_planner.py" in th.CONTROL_MODULES and th.LABEL_MERGE_ERRORS == {}
    assert set(th.module_labels(MODULE)) <= set(th.CONSTANT_LABELS)


def test_restated_v1_constants_equal_their_sources():
    from gpt_field import wrist_servo
    assert pl.SIGMA_ALIGN_PRIOR_M == pytest.approx(wrist_servo.Align.TOLERANCE_MM / 1000.0, abs=1e-15)
    assert pl.V5_AXIS_AGREE == wrist_servo.BAR_CHECK_MIN_AGREE
    assert pl.V5_COHERENCE == wrist_servo.BAR_CHECK_MIN_COHERENCE
    assert pl.V5_RMS == wrist_servo.BAR_CHECK_MIN_RMS


def test_contract_constants():
    assert (pl.P1_ALIGN_K, pl.P1_PAD_MARGIN_MM, pl.P2_SIGMA_K, pl.P2_INFEASIBLE_MARGIN_MM) == (2.0, 1.0, 2.0, 2.0)
    assert (pl.P3_MIN_OVERLAP_MM, pl.P4_MIN_KNOWN_FRACTION, pl.P5_SIGMA_K) == (8.0, 0.7, 3.0)
    assert (pl.P6_TARGET_FLOOR_MM, pl.P6_TARGET_SIGMA_K, pl.P6_PALM_FLOOR_MM, pl.P6_PALM_SIGMA_K,
            pl.P6_LANDING_FLOOR_MM, pl.P6_LANDING_SIGMA_K) == (2.0, 1.5, 5.0, 2.0, 3.0, 3.0)
    assert (pl.P7_FLOOR_MM, pl.P7_SIGMA_K, pl.P8_FLOOR_M, pl.P8_SIGMA_K, pl.P9_FLOOR_M, pl.P9_SIGMA_K) == (
        5.0, 3.0, 0.005, 2.0, 0.03, 3.0)
    assert (pl.P10_MAX_RETREAT_MM, pl.P11_FLOOR_M, pl.P11_SIGMA_K, pl.P11_OFFSET_M) == (8.0, 0.05, 3.0, 0.02)
    assert (pl.P12_FLOOR_M, pl.P12_SIGMA_K, pl.P17_BONUS) == (0.01, 3.0, 0.3)
    assert (pl.SCORE_TOUCH_BONUS, pl.SCORE_TURN_PENALTY_PER_QUARTER, pl.SCORE_MARGINAL_PENALTY) == (0.5, 0.2, 1.0)
    assert (pl.G13_FLOOR_MM, pl.G13_SIGMA_K, pl.M1_FLOOR_M, pl.M1_SIGMA_K, pl.M2_SEGMENT_M) == (3.0, 3.0, 0.02, 3.0,
                                                                                               0.075)
    assert (pl.M4_FLOOR_M, pl.M4_FRACTION, pl.M5_FLOOR_M, pl.M5_FRACTION, pl.M6_ABOVE_M, pl.M6_SIGMA_K) == (
        0.005, 0.5, 0.005, 0.5, 0.03, 3.0)
    assert (pl.M7_FLOOR_M, pl.M7_TOL_K, pl.M8_FLOOR_M, pl.M8_SIGMA_K, pl.M16_FLOOR_M) == (0.01, 2.0, 0.01, 3.0, 0.005)
    assert (pl.O5_FLOOR_M, pl.O5_SIGMA_K, pl.O5_TOP_FLOOR_M, pl.V5_GRASP_SIGMA_K) == (0.02, 3.0, 0.02, 3.0)
    assert pl.DEST_SEPARATION_FRACTION == 0.5 and pl.SIGMA_VC_M == 0.004
    assert pl.FAMILIES == ("top_down", "frontal", "frontal_pitched_45")
    assert pl.DEPTH_RULES == ("touch_stop", "face_stop", "height_stop", "palm_stop")


# ------------------------------------------------------------------------------------------------ 1 wall
def test_wall_touch_stop_inner_column_on_the_floor_outer_on_the_support():
    hf, b = wall_scene()
    plan = pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "touch_stop"
    grip = np.array(plan.grip_point_xyz)
    assert np.linalg.norm(grip[:2]) == pytest.approx(0.0505, abs=1e-9)          # on the circle
    jaw = np.array(plan.jaw_world[:2])
    assert abs(float(jaw @ grip[:2])) / 0.0505 == pytest.approx(1.0, abs=1e-6)   # radial jaw
    landings = sorted(c.landing_z for c in plan.columns)
    assert landings == pytest.approx([0.900, 0.905], abs=1e-9)                  # outer: support, inner: floor
    inner = min(plan.columns, key=lambda c: np.linalg.norm(c.polygon_xy.mean(axis=0)))
    assert inner.landing_z == pytest.approx(0.905)
    assert plan.expected[0].grip_z == pytest.approx(0.905 + T_F) and plan.expected[1].grip_z == pytest.approx(
        0.900 + T_F)
    assert plan.clearance_retreat_m == pytest.approx(0.0, abs=1e-12)
    assert plan.checks["P3_overlap_mm"] == pytest.approx(16.0)


def test_wall_rim_point_and_side_chosen_by_footprint_clearance():
    hf, b = wall_scene(block=True)
    result = pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry")
    assert isinstance(result, pl.ContactPlan)
    angle = math.atan2(result.grip_point_xyz[1], result.grip_point_xyz[0])
    assert abs(angle) >= math.radians(15.0) - 1e-9                               # turned away from the block
    assert result.checks["P7_clearance_mm"] > 0.0 and result.depth_rule == "touch_stop"
    toward = pl.evaluate_candidate(b, hf, inputs(jaw=0.0), "top_down", 0.0, (), purpose="carry")
    assert isinstance(toward, pl.CandidateRejection) and toward.reason == "footprint_blocked"
    assert toward.rule.endswith(":P7") and toward.values["by_rule"]["touch_stop"]["check"] == "P7"


def test_wall_overhang_all_around_is_footprint_blocked_before_descent():
    hf, b = wall_scene(surround=True)
    result = pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry")
    assert isinstance(result, pl.Infeasible) and result.reason == "footprint_blocked"
    # both finger assignments at every rim point (BP-8): 2 x 24 candidates
    assert len(result.rejections) == 2 * round(2.0 * math.pi / YAW_GRID_RAD)
    assert all(r.rim_angle_rad is not None for r in result.rejections)
    as_json(result)
    assert all("rim_angle_deg" in row for row in pl.trace_payload(result)["rejections"])


def test_wall_rim_twin_with_finger_one_inside_needs_no_turn():
    """BP-8: with the +x side blocked beyond the reach of a -x hand footprint and a +-1 rad budget, the -x rim point
    is reached at the start jaw with finger 1 inside (its finger-1-outside twin needs a half turn)."""
    hf, b = wall_scene()
    stamp(hf, rect_polygon((0.12, 0.0), (1.0, 0.0), 0.04, 0.12), 0.99)          # x 0.08..0.16 at 0.99
    plan = pl.plan_contact(b, hf, inputs(jaw=0.0, budget=(1.0, 1.0)), (), purpose="carry")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "touch_stop"
    grip = np.array(plan.grip_point_xyz[:2])
    assert grip[0] < 0.0 and np.linalg.norm(grip) == pytest.approx(0.0505, abs=1e-9)
    assert abs(plan.yaw_turn_rad) < 0.3
    jaw = np.array(plan.jaw_world[:2])
    assert float(jaw @ grip) < 0.0                                                # finger 1 (on +jaw) inside
    assert plan.rim_angle_rad is not None and abs(abs(plan.rim_angle_rad) - math.pi) < 0.3
    assert len(plan.key()) == 4 and plan.candidate_key()[2] is None
    # finger 1 outside at the same rim point (the old single assignment) needs a half turn: out of budget
    outside = pl.evaluate_candidate(b, hf, inputs(jaw=0.0, budget=(1.0, 1.0)), "top_down", plan.rim_angle_rad, (),
                                    purpose="carry")
    assert isinstance(outside, pl.CandidateRejection) and outside.reason == "yaw_budget"
    # excluding the candidate removes only this rim point's assignment, not the other end of the same jaw line
    again = pl.plan_contact(b, hf, inputs(jaw=0.0, budget=(1.0, 1.0)), (), purpose="carry",
                            exclude=[plan.candidate_key()])
    assert not (isinstance(again, pl.ContactPlan) and again.key()[:2] == plan.key()[:2]
                and again.rim_angle_rad is not None and pl._same_angle(again.rim_angle_rad, plan.rim_angle_rad))
    as_json(plan.to_dict())


# ------------------------------------------------------------------------------------------------ 2 rotary part
def test_rotary_part_on_a_disk_touch_stop_first():
    hf, a = lever_scene()
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "touch_stop"
    assert plan.family == "top_down" and plan.align_mode == "across_bar"
    assert np.allclose(plan.jaw_world, (0.0, -1.0, 0.0), atol=1e-9) and abs(plan.yaw_turn_rad) < 1e-9
    assert all(c.landing_z == pytest.approx(0.918) and c.straddles_edge for c in plan.columns)
    assert plan.expected[0].kind == "fingertip_landing"
    assert plan.expected[0].grip_z == pytest.approx(0.918 + 0.0093, abs=1e-9)
    assert plan.clearance_retreat_m * 1000.0 == pytest.approx(5.6, abs=0.5)
    assert plan.checks["P3_overlap_mm"] == pytest.approx(16.0)
    sigma_c = max(a.sigma_h_m, 0.001) + pl.SIGMA_VC_M
    assert plan.bound_grip_z == pytest.approx(0.918 + 0.0093 - max(0.03, 3.0 * sigma_c), abs=1e-9)
    assert plan.grip_point_xyz[:2] == pytest.approx((0.0, 0.004), abs=1e-9)    # circle centre on the axis
    assert plan.width["class"] == "feasible" and plan.width["w_mm"] == pytest.approx(24.3)
    assert plan.width["e_lat_mm"] == pytest.approx(2 * 3.0 + 1.0 + 1.0)
    assert plan.checks["P17"] is None
    assert plan.target_grip_z is None
    as_json(plan.to_dict())


def test_rotary_part_height_stop_second():
    hf, a = lever_scene()
    first = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    second = pl.plan_contact(a, hf, inputs(), (), purpose="turn", exclude=[first.key()])
    forced = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    for plan in (second, forced):
        assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
        assert plan.target_grip_z == pytest.approx(0.955 - 0.0116 - max(0.002, 1.5 * 0.0025), abs=1e-9)
        assert plan.target_grip_z == pytest.approx(0.939, abs=0.001)
        assert plan.expected[0].kind == "none_before_target"
    # the 180-deg twin needs a half turn, beyond the R12 budget: excluding the yaw leaves nothing
    nothing = pl.plan_contact(a, hf, inputs(), (), purpose="turn", exclude=[first.key()[:2]])
    assert isinstance(nothing, pl.Infeasible) and nothing.reason == "yaw_budget"


def test_purpose_decides_the_grip_point():
    hf, a = lever_scene()
    turn = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    hold = pl.plan_contact(a, hf, inputs(), (), purpose="hold")
    assert turn.grip_point_xyz == pytest.approx((0.0, 0.004, 0.955))
    assert hold.grip_point_xyz == pytest.approx((0.010, 0.004, 0.955))


def test_closing_sweep_blocks_both_rules():
    hf, a = lever_scene()
    ring = disk((0.0, 0.0), 0.028)
    inner = {tuple(c) for c in hf.cells_in_polygon(disk((0.0, 0.0), 0.022))}
    region = {tuple(c) for c in a.region_cells}
    cells = [c for c in hf.cells_in_polygon(ring) if tuple(c) not in inner and tuple(c) not in region]
    cells = np.array(cells)
    hf.z[cells[:, 0], cells[:, 1]] = 0.930                                        # a raised ring inside the sweep
    result = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    assert isinstance(result, pl.Infeasible) and result.reason in ("column_blocked", "yaw_budget")
    blocked = [r for r in result.rejections if r.reason == "column_blocked"]
    assert blocked and blocked[0].values["by_rule"]["touch_stop"]["check"] == "P5"


# ------------------------------------------------------------------------------------------------ 3 thin part
@pytest.mark.parametrize("sigma_h, target", [(0.001, 1.0209), (0.00133, 1.0209), (0.002, 1.0199)])
def test_thin_part_height_stop_by_p6(sigma_h, target):
    hf, c, d = thin_part_scene(sigma_h)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    assert plan.target_grip_z == pytest.approx(target, abs=1e-4)
    assert np.allclose(np.abs(plan.jaw_world), (1.0, 0.0, 0.0), atol=1e-9)        # jaw across the thin part
    assert plan.checks["P17"] is True and plan.jaw_world[0] > 0.0                 # R0 toward the body
    assert plan.checks["P7_clearance_mm"] > 0.0
    for column in plan.columns:                                                    # P4 counts the plane fills
        assert column.known_fraction == pytest.approx(1.0) and column.landing_z == pytest.approx(SUPPORT)


def test_thin_part_touch_stop_refused_and_no_palm_stop():
    hf, c, d = thin_part_scene(0.002)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert plan.depth_rule == "height_stop"
    touch = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, depth_rules=("touch_stop",))
    assert isinstance(touch, pl.Infeasible)
    # BP-6: the module refuses it by P6's palm clause (the band reads [support, top] without a measured band-low, so
    # P3 sees 16 mm of overlap); the exact reason is asserted so a change of the deciding check is caught
    reasons = {(r.values["by_rule"]["touch_stop"]["check"], r.values["by_rule"]["touch_stop"]["reason"])
               for r in touch.rejections}
    assert reasons == {("P6", "palm_clearance")}
    assert all(r.values["by_rule"]["touch_stop"]["values"]["overlap_mm"] == pytest.approx(16.0)
               for r in touch.rejections)
    for rejection in touch.rejections:                                             # the landing is the support
        assert rejection.values["by_rule"]["touch_stop"]["values"]["part_band_z"][1] == pytest.approx(1.0345)
    palm = pl.evaluate_candidate(c, hf, inputs(), "top_down", 0.0, (), purpose="carry", depth_rules=("palm_stop",))
    assert isinstance(palm, pl.CandidateRejection) and palm.rule.endswith("P13")
    assert "palm_stop" not in pl.SLICE_DEPTH_RULES


def test_thin_part_feasible_at_5_5_mm_palm_clearance_at_6_mm():
    hf, c, d = thin_part_scene(0.0055)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    assert plan.checks["P6_palm_margin_mm"] >= 0.0
    hf, c, d = thin_part_scene(0.006)
    result = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(result, pl.Infeasible) and result.reason == "palm_clearance"
    height = result.rejections[0].values["by_rule"]["height_stop"]
    assert height["reason"] == "palm_clearance" and height["check"] == "P6"
    assert height["values"]["target_grip_z"] == pytest.approx(1.0139, abs=1e-4)
    assert pl.needs_height_refinement(result)                                     # Appendix L 3
    assert result.rejections[0].values["sigma_h_limit_m"] == pytest.approx(0.0194 / 3.5, abs=1e-6)   # 5.54 mm
    assert pl.trace_payload(result)["refine_height"] is True


def test_touch_stop_palm_clause_at_the_contact_grip():
    """BP-2: the palm clause of a touch_stop is checked at the contact grip (the hand's lowest point), not at the
    retreated grip: a free 24 mm part 37 mm above its landing leaves the palm 3.3 mm above its top at contact."""
    hf = field()
    poly = disk((0.0, 0.0), 0.012)
    cells = stamp(hf, poly, 0.937)
    free = part("F", xyz=(0.0, 0.0, 0.937), top=0.937, sigma_h=0.001, support=SUPPORT, move="free", region=poly,
                cells=cells, widths=widths_const(0.024, 0.0005))
    touch = pl.evaluate_candidate(free, hf, inputs(jaw=0.0), "top_down", 0.0, (), purpose="carry",
                                  depth_rules=("touch_stop",))
    assert isinstance(touch, pl.CandidateRejection) and touch.reason == "palm_clearance" and touch.rule.endswith("P6")
    values = touch.values["by_rule"]["touch_stop"]["values"]
    assert values["palm_margin_m"] == pytest.approx(SUPPORT + T_F + ROBOT.p_min_m - 0.937 - 0.005, abs=1e-9)
    assert values["retreat_m"] > 0.0                                               # the retreated grip would pass
    plan = pl.plan_contact(free, hf, inputs(jaw=0.0), (), purpose="carry")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    hinged = PartRecord(**{**free.__dict__, "move_prior": "hinged"})                 # fixture: palm margin 0
    fixture = pl.plan_contact(hinged, hf, inputs(jaw=0.0), (), purpose="carry")
    assert fixture.depth_rule == "touch_stop"
    assert fixture.checks["P6_palm_margin_mm"] == pytest.approx((SUPPORT + T_F + ROBOT.p_min_m - 0.937) * 1000.0)


def test_thin_part_columns_unknown_without_plane_fill():
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    for x in (0.045, -0.045):
        unknown(hf, rect_polygon((x, 0.0), (1.0, 0.0), 0.02, 0.02))
    result = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"


def test_sigma_limited_footprint_refusal_asks_for_height_refinement():
    """Real task-2 geometry (BP probe): a structure level with the thin part's top beside it (inside the hand
    footprint, outside the part region and the finger slot) binds P7 before P6 once sigma_h grows. Gap round: the
    structure stands 5.5 mm above the part's top, so the IF-HS raise (8.6 mm) fails its P3 guard at sigma_h 5 mm."""
    for sigma_h, feasible in ((0.002, True), (0.005, False)):
        hf, c, d = thin_part_scene(sigma_h)
        stamp(hf, rect_polygon((0.014, 0.020), (1.0, 0.0), 0.006, 0.005), 1.040)
        result = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.ContactPlan) == feasible
        if not feasible:
            assert result.reason == "footprint_blocked" and pl.needs_height_refinement(result)
    hf, c, d = thin_part_scene(0.002)
    keep = KeepOut("E", "clear_of", rect_polygon((0.08, -0.02), (1.0, 0.0), 0.01, 0.005), SUPPORT, 1.06, ())
    keep_neg = KeepOut("E", "clear_of", rect_polygon((-0.08, -0.02), (1.0, 0.0), 0.01, 0.005), SUPPORT, 1.06, ())
    blocked = pl.plan_contact(c, hf, inputs(), (keep, keep_neg), purpose="carry", next_held=d)
    assert blocked.reason == "footprint_blocked" and not pl.needs_height_refinement(blocked)


def test_turned_hand_raises_the_hover_instead_of_refusing():
    """BP-10: a post swept only by the Orient's turn raises the hover to the P12 requirement; the P17 yaw is kept
    rather than refused in favour of the twin that turns the other way."""
    hf, c, d = thin_part_scene(0.002)
    stamp(hf, rect_polygon((0.055, 0.055), (1.0, 0.0), 0.006, 0.006), 1.12)
    plan = pl.plan_contact(c, hf, inputs(jaw=math.pi / 2.0), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.checks["P17"] is True
    assert abs(math.degrees(plan.yaw_rad)) == pytest.approx(0.0, abs=1e-6)
    assert plan.checks["P12_hover_raised_m"] > 0.0 and plan.checks["P12_clearance_m"] == pytest.approx(0.0)
    assert plan.hover_grip_z + ROBOT.p_min_m == pytest.approx(1.12 + pl.P12_FLOOR_M)
    hf, c, d = thin_part_scene(0.002)
    low = pl.plan_contact(c, hf, inputs(jaw=math.pi / 2.0), (), purpose="carry", next_held=d)
    assert low.checks["P12_hover_raised_m"] == 0.0 and low.hover_grip_z < plan.hover_grip_z


def test_keepout_volume_is_an_obstacle_for_the_footprint():
    hf, c, d = thin_part_scene(0.002)
    keep = KeepOut("E", "clear_of", rect_polygon((0.08, -0.02), (1.0, 0.0), 0.01, 0.005), SUPPORT, 1.06, ())
    keep_neg = KeepOut("E", "clear_of", rect_polygon((-0.08, -0.02), (1.0, 0.0), 0.01, 0.005), SUPPORT, 1.06, ())
    result = pl.plan_contact(c, hf, inputs(), (keep, keep_neg), purpose="carry", next_held=d)
    assert isinstance(result, pl.Infeasible) and result.reason == "footprint_blocked"
    # the grasped part's own keep-out is never its obstacle
    own = KeepOut("C", "clear_of", rect_polygon((0.08, -0.02), (1.0, 0.0), 0.01, 0.005), SUPPORT, 1.06, ())
    assert isinstance(pl.plan_contact(c, hf, inputs(), (own,), purpose="carry", next_held=d), pl.ContactPlan)


# ------------------------------------------------------------------------------------------------ 4 free body
def test_free_body_along_the_protrusion_axis_is_column_blocked():
    hf, d = body_scene()
    rejection = pl.evaluate_candidate(d, hf, inputs(jaw=0.0, g_open=78.5, sigma_align=0.002), "top_down",
                                      math.pi / 2.0, (), purpose="carry")
    assert isinstance(rejection, pl.CandidateRejection) and rejection.reason == "column_blocked"
    assert rejection.values["by_rule"]["touch_stop"]["reason"] == "column_blocked"
    assert rejection.values["by_rule"]["height_stop"]["reason"] == "column_blocked"


def test_own_protrusions_in_the_region_still_land_the_columns():
    """BP-1: the columns skip only the measured cross-section (+-w/2 along the jaw), so protrusions that the region
    includes (a same-band region grown over the whole object) still block the pinch along their axis."""
    hf, d = body_scene()
    extra = [hf.cells_in_polygon(rect_polygon((0.0, y), (0.0, 1.0), 0.013, 0.006)) for y in (0.047, -0.047)]
    whole = PartRecord(**{**d.__dict__, "region_cells": np.unique(np.vstack([d.region_cells, *extra]), axis=0)})
    probe = inputs(jaw=0.0, g_open=78.5, sigma_align=0.002)
    along = pl.evaluate_candidate(whole, hf, probe, "top_down", math.pi / 2.0, (), purpose="carry")
    assert isinstance(along, pl.CandidateRejection) and along.reason == "column_blocked"
    assert along.values["by_rule"]["touch_stop"]["reason"] == "column_blocked"
    assert along.values["by_rule"]["height_stop"]["reason"] == "column_blocked"
    assert sorted(c["landing_z"] for c in along.values["columns"]) == pytest.approx([1.032, 1.033])
    across = pl.evaluate_candidate(whole, hf, probe, "top_down", 0.0, (), purpose="carry")
    assert isinstance(across, pl.ContactPlan) and np.allclose(np.abs(across.jaw_world), (1.0, 0.0, 0.0), atol=1e-9)
    assert all(c.landing_z == pytest.approx(SUPPORT) for c in across.columns)
    # decision §2.4 P-C: this record's flat width function disagrees with its region grown along y (PCA aspect >= 1.5),
    # so plan_contact no longer offers the plain grid for the free part: the axis is refined first (BR-1)
    refine = pl.plan_contact(whole, hf, probe, (), purpose="carry")
    assert isinstance(refine, pl.Infeasible) and refine.refine_axis and refine.reason == "geometry_unmeasured"
    # the cross-section helper: the body's cells within +-33 mm along the jaw, none of the protrusions
    section = pl.section_cells(whole.region_cells, hf, (0.0, 0.0), math.pi / 2.0, 0.033)
    centres = hf.origin_xy + section[:, ::-1] * hf.cell_m
    assert len(section) and np.abs(centres[:, 1]).max() <= 0.033 + 1e-9
    assert {tuple(c) for c in d.region_cells} >= {tuple(c) for c in section}


@pytest.mark.parametrize("sigma_align, e_lat, cls", [(0.002, 5.75, "feasible"), (0.003, 7.75, "marginal")])
def test_free_body_across_feasible_or_marginal(sigma_align, e_lat, cls):
    hf, d = body_scene()
    plan = pl.plan_contact(d, hf, inputs(jaw=0.0, g_open=78.5, sigma_align=sigma_align), (), purpose="carry")
    assert isinstance(plan, pl.ContactPlan)
    assert plan.width["sep_mm"] == pytest.approx(77.5) and plan.width["e_lat_mm"] == pytest.approx(e_lat)
    assert plan.width["class"] == cls
    assert np.allclose(np.abs(plan.jaw_world), (1.0, 0.0, 0.0), atol=1e-9)
    assert plan.depth_rule == "height_stop"                                       # the palm rules out touch_stop
    result = pl.plan_contact(d, hf, inputs(jaw=0.0, g_open=78.5, sigma_align=sigma_align), (), purpose="carry")
    assert result.key() == plan.key()


# ------------------------------------------------------------------------------------------------ 5 cylinder
def cylinder_scene(widths=None, poly=None):
    hf = field()
    poly = disk((0.0, 0.0), 0.0315) if poly is None else poly
    cells = stamp(hf, poly, 0.930)
    return hf, part("F", xyz=(0.0, 0.0, 0.930), top=0.930, support=SUPPORT, move="free", region=poly, cells=cells,
                    widths=widths_const(0.063, 0.001) if widths is None else widths)


def test_63_mm_cylinder_is_marginal_not_infeasible():
    hf, f = cylinder_scene()
    plan = pl.plan_contact(f, hf, inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    assert isinstance(plan, pl.ContactPlan) and plan.width["class"] == "marginal"
    assert plan.width["e_lat_mm"] == pytest.approx(8.0) and plan.depth_rule == "touch_stop"
    widths = [t for t in plan.thresholds if t.name.startswith("P2_")]
    assert {t.name for t in widths} == {"P2_width_feasible_mm", "P2_width_infeasible_mm"}
    assert next(t for t in widths if t.name == "P2_width_feasible_mm").value == pytest.approx(77.5 - 16.0)
    too_wide = cylinder_scene(widths=widths_const(0.078, 0.001))                 # 78 - 2 >= 77.5 - 2
    result = pl.plan_contact(too_wide[1], too_wide[0], inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    assert isinstance(result, pl.Infeasible) and result.reason == "part_too_wide"


def test_feasible_candidates_rank_above_marginal_ones():
    rows = tuple((k * YAW_GRID_RAD, Meas(0.050 if k == 0 else 0.063, 0.001, "kappa")) for k in range(12))
    ellipse = np.column_stack([0.025 * np.cos(np.linspace(0, 2 * math.pi, 64, endpoint=False)),
                               0.0315 * np.sin(np.linspace(0, 2 * math.pi, 64, endpoint=False))])
    hf, f = cylinder_scene(widths=rows, poly=ellipse)
    plan = pl.plan_contact(f, hf, inputs(jaw=math.pi / 2.0, g_open=78.5), (), purpose="carry")
    assert plan.width["class"] == "feasible" and abs(abs(plan.yaw_turn_rad) - math.pi / 2.0) < 1e-6


# ------------------------------------------------------------------------------------------------ 5c hover
def test_hover_for_a_seeded_part():
    hf = field()
    stamp(hf, rect_polygon((0.02, 0.0), (1.0, 0.0), 0.006, 0.006), 0.98)          # within 3 sigma_xy of the seed
    stamp(hf, rect_polygon((0.0, 0.12), (1.0, 0.0), 0.01, 0.01), 1.05)            # beyond it, outside the hand
    seeded = PartRecord(pid="A", text="part A", move_prior="hinged", roles=frozenset({"grip"}),
                        xyz=np.array([0.0, 0.0, 0.93]), cov=np.eye(3) * 0.015 ** 2, sigma_h_m=0.015)
    out = pl.hover_for(seeded, hf, ROBOT, ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9)))
    assert out["z_ref"] == pytest.approx(0.98) and out["raised_m"] == pytest.approx(0.0)
    assert out["fingertip_z"] >= 0.98 + max(0.05, 3 * 0.015 + 0.02) - 1e-9
    assert out["hover_grip_xyz"][:2] == pytest.approx([0.0, 0.0])
    tall = field()
    stamp(tall, rect_polygon((0.08, 0.0), (1.0, 0.0), 0.01, 0.01), 1.10)            # under the hand along the jaw
    raised = pl.hover_for(seeded, tall, ROBOT, ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9)))
    assert raised["raised_m"] > 0.0
    assert raised["hover_grip_xyz"][2] + ROBOT.p_min_m >= 1.10 + 0.01 - 1e-9
    assert pl.hover_for(PartRecord(pid="A", text="", move_prior="free", roles=frozenset()), hf, ROBOT,
                        ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9)))["reason"] == "geometry_unmeasured"


def test_provisional_or_unregistered_parts_are_refused():
    hf, a = lever_scene()
    provisional = PartRecord(pid="A", text="part A", move_prior="hinged", roles=frozenset({"grip"}),
                             xyz=np.array([0.0, 0.0, 0.95]), cov=np.eye(3) * 1e-4, sigma_h_m=0.015, provisional=True,
                             registration_step=0)
    for record in (provisional, PartRecord(pid="A", text="", move_prior="free", roles=frozenset())):
        result = pl.plan_contact(record, hf, inputs(), (), purpose="turn")
        assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"


def test_yaw_budget_and_frontal_reserved():
    hf, a = lever_scene()
    result = pl.plan_contact(a, hf, inputs(jaw=0.0, budget=(0.1, 0.1)), (), purpose="turn")
    assert isinstance(result, pl.Infeasible) and result.reason == "yaw_budget"
    assert pl.families_for(a, inputs()) == ("top_down",)
    with pytest.raises(NotImplementedError):
        pl.evaluate_candidate(a, hf, inputs(), "frontal", 0.0, (), purpose="turn")
    with pytest.raises(ValueError):
        pl.plan_contact(a, hf, inputs(), (), purpose="lift")


def test_grid_yaws_within_budget_smaller_turns_first():
    hf, d = body_scene()
    yaws = pl.jaw_yaws(d, "top_down", inputs(jaw=0.0, g_open=78.5))
    turns = [abs(pl._wrap(y)) for y in yaws]
    assert all(b >= a - 1e-9 for a, b in zip(turns, turns[1:])) and turns[0] == pytest.approx(0.0)
    assert max(turns) <= 1.96 + 1e-9 and len(yaws) == 15
    hf, a = lever_scene()
    assert sorted(round(math.degrees(y)) for y in pl.jaw_yaws(a, "top_down", inputs())) == [-90, 90]


# ------------------------------------------------------------------------------------------------ 6 answer checks
def test_grasp_point_check_uses_the_finger_columns_not_every_neighbour():
    hf, c, d = thin_part_scene(0.002)
    ok = pl.check_point_answer("grasp_point", (256, 200), c, hf, inputs(), jaw_yaw=0.0)
    assert ok["ok"] and ok["reason"] is None and min(ok["values"]["above_m"]) > 0.1
    wrong = part("C", xyz=(0.08, -0.05, SUPPORT), top=SUPPORT, sigma_h=0.002)
    bad = pl.check_point_answer("grasp_point", (100, 100), wrong, hf, inputs(), jaw_yaw=0.0)
    assert not bad["ok"] and bad["reason"] == "pointed_not_grasp_point"
    on_body = part("C", xyz=(0.0, 0.0826, 1.0331), top=1.0331, sigma_h=0.002)
    assert not pl.check_point_answer("grasp_point", (1, 1), on_body, hf, inputs(), jaw_yaw=0.0)["ok"]
    assert all(isinstance(t, Threshold) and t.name == "V5_grasp_margin_m" for t in ok["thresholds"])


def test_grasp_point_check_needs_the_re_measured_sigma():
    hf, a = lever_scene(sigma_h=0.015)                                             # start-view sigma ~ 1.5 cm
    assert not pl.check_point_answer("grasp_point", (1, 1), a, hf, inputs())["ok"]
    hf, a = lever_scene(sigma_h=0.0025)                                            # after the hover probe
    assert pl.check_point_answer("grasp_point", (1, 1), a, hf, inputs())["ok"]


@pytest.mark.parametrize("sigma_h, ok", [(0.012, True), (0.0125, False)])
def test_grasp_point_margin_is_k_sigma_h(sigma_h, ok):
    """BP-3: the V5 grasp margin is V5_GRASP_SIGMA_K*sigma_h (contract §C7): the lever top stands 37 mm above its
    disk landings, so it passes at sigma_h 12 mm (36 mm) and fails at 12.5 mm (37.5 mm)."""
    hf, a = lever_scene(sigma_h=sigma_h)
    out = pl.check_point_answer("grasp_point", (1, 1), a, hf, inputs())
    assert out["values"]["margin_m"] == pytest.approx(pl.V5_GRASP_SIGMA_K * sigma_h, abs=1e-12)
    assert min(out["values"]["above_m"]) == pytest.approx(0.037, abs=1e-9)
    assert out["ok"] is ok and out["thresholds"][0].derivation.startswith("V5_GRASP_SIGMA_K*sigma_h")


def test_v5_plane_constants_restate_the_scene_d8():
    scene = pytest.importorskip("gpt_field.agent_speedup_v2_scene")
    assert pl.V5_PLANE_FLOOR_M == scene.D8_FLOOR_M and pl.V5_PLANE_SIGMA_K == scene.D8_SIGMA_K
    hf = field()
    stamp(hf, rect_polygon((0.0, 0.0), (1.0, 0.0), 0.095, 0.095), 0.9305, sigma=0.002)
    flat = part("B", xyz=(0.0, 0.0, 0.9305), top=0.9305, sigma_h=0.002)
    out = pl.check_point_answer("surface_point", (1, 1), flat, hf, inputs(), held=held_frame(r_obj=0.04))
    assert out["thresholds"][0].value == pytest.approx(pl.V5_PLANE_SIGMA_K * math.hypot(0.002, 0.002))   # hypot


def test_surface_and_elongated_checks():
    hf = field()
    stamp(hf, rect_polygon((0.0, 0.0), (1.0, 0.0), 0.095, 0.095), 0.9305)
    flat = part("B", xyz=(0.0, 0.0, 0.9305), top=0.9305, sigma_h=0.002)
    held = held_frame(r_obj=0.04)
    assert pl.check_point_answer("surface_point", (1, 1), flat, hf, inputs(), held=held)["ok"]
    edge = part("B", xyz=(0.09, 0.0, 0.9305), top=0.9305, sigma_h=0.002)
    out = pl.check_point_answer("surface_point", (1, 1), edge, hf, inputs(), held=held)
    assert not out["ok"] and out["reason"] == "pointed_not_surface"
    hf, a = lever_scene()
    gray = np.zeros((512, 512))
    # pose jaw -y: image right = R1 = -y, image up = R0 = +x, so the world-x ridge runs along the image v axis
    along_v = pl.check_point_answer("elongated", (256, 256), a, hf, inputs(), wrist_gray_upright=gray,
                                    structure_axis=lambda g, uv: (np.array([0.0, 1.0]), 0.9, 5.0))
    across = pl.check_point_answer("elongated", (256, 256), a, hf, inputs(), wrist_gray_upright=gray,
                                   structure_axis=lambda g, uv: (np.array([1.0, 0.0]), 0.9, 5.0))
    weak = pl.check_point_answer("elongated", (256, 256), a, hf, inputs(), wrist_gray_upright=gray,
                                 structure_axis=lambda g, uv: (np.array([0.0, 1.0]), 0.3, 5.0))
    assert along_v["ok"] and along_v["values"]["agree"] == pytest.approx(1.0)
    assert not across["ok"] and across["reason"] == "pointed_not_elongated"
    assert not weak["ok"]
    with pytest.raises(ValueError):
        pl.check_point_answer("cavity", (1, 1), a, hf, inputs())


def test_elongated_check_with_the_real_geometry_module():
    geometry = pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    if not hasattr(geometry, "structure_axis"):
        pytest.skip("geometry.structure_axis not available yet")
    hf, a = lever_scene()
    gray = np.full((512, 512), 200.0)
    gray[:, 244:268] = 30.0                                                        # a dark ridge along image v
    out = pl.check_point_answer("elongated", (256.0, 256.0), a, hf, inputs(), wrist_gray_upright=gray)
    assert out["values"]["checked"] and out["ok"], out


# ------------------------------------------------------------------------------------------------ 7 classify
def test_classify_contact_all_four_classes():
    hf, a = lever_scene()
    touch = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    z = touch.expected[0].grip_z
    band = touch.expected[0].band_m
    assert band == pytest.approx(max(0.005, 2.0 * (a.sigma_h_m + pl.SIGMA_VC_M)))
    assert pl.classify_contact(touch, z + 0.001, "contact_stall")["class"] == "expected"
    assert pl.classify_contact(touch, z + band + 0.005, "contact_touch")["class"] == "blocked_above_part"
    # P-B: on the hinged (fixture) part a stop below the band and above the bound is a measured landing; on a free
    # part the same stop stays no_contact_where_expected
    below = pl.classify_contact(touch, z - band - 0.005, "contact_stall")
    assert below["class"] == "expected" and below["landing_measured"] is True
    free = dataclasses.replace(touch, move_prior="free")
    assert pl.classify_contact(free, z - band - 0.005, "contact_stall")["class"] == "no_contact_where_expected"
    assert pl.classify_contact(touch, touch.bound_grip_z, "travel_reached")["class"] == "no_contact_within_bound"
    height = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    assert pl.classify_contact(height, height.target_grip_z, "travel_reached")["class"] == "expected"
    assert pl.classify_contact(height, None, "target_reached")["class"] == "expected"
    assert pl.classify_contact(height, height.target_grip_z + 0.03, "contact_stall")["class"] == "blocked_above_part"
    out = pl.classify_contact(touch, z, "contact_stall")
    assert out["matched"] == 0 and {t.name for t in out["thresholds"]} >= {"P8_band_m", "P9_bound_grip_z_m",
                                                                           "P3_close_limit_grip_z_m"}
    as_json(out)


def test_band_upper_side_capped_where_a_close_still_overlaps():
    """BP-4: sigma_vc widens the P8 band (Appendix L 11), but a stop above top + pad bottom - P3 (the pads would
    overlap the part top by less than 8 mm) is blocked_above_part, never closed on."""
    hf, c, d = thin_part_scene(0.002)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert plan.depth_rule == "height_stop" and plan.target_grip_z == pytest.approx(1.0199, abs=1e-4)
    e = plan.expected[0]
    cap = 1.0345 + 0.0044 - 0.008
    assert e.band_m == pytest.approx(2.0 * (0.002 + pl.SIGMA_VC_M)) and e.close_limit_grip_z == pytest.approx(cap)
    assert e.grip_z + e.band_m > cap and e.upper_z == pytest.approx(cap)
    assert pl.classify_contact(plan, cap - 0.0005, "contact_stall")["class"] == "expected"
    assert pl.classify_contact(plan, cap + 0.0005, "contact_stall")["class"] == "blocked_above_part"
    pads = ROBOT.pad_band_z(cap)
    assert 1.0345 - pads[0] == pytest.approx(pl.P3_MIN_OVERLAP_MM / 1000.0)        # P3 holds exactly at the cap
    # the lever's touch band stays uncapped (cap 0.9514 above 0.9273 + 13 mm)
    hf, a = lever_scene()
    touch = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    assert all(x.upper_z == pytest.approx(x.grip_z + x.band_m) for x in touch.expected)


def test_classify_non_contact_stops_and_handover_events():
    """BP-10: a non-contact stop short of the plan's end is stopped_early (not no_contact_within_bound); BP-7: a
    GuardedLower fast-phase reason handover_<cue> is classified by its cue."""
    hf, a = lever_scene()
    height = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    assert pl.classify_contact(height, height.target_grip_z + 0.02, "budget")["class"] == "stopped_early"
    assert pl.classify_contact(height, height.target_grip_z + 0.001, "budget")["class"] == "expected"
    touch = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    z = touch.expected[0].grip_z
    assert pl.classify_contact(touch, z + 0.02, "budget")["class"] == "stopped_early"
    assert pl.classify_contact(touch, z - touch.expected[0].band_m - 0.002, "budget")["class"] == \
        "no_contact_where_expected"
    assert pl.classify_contact(touch, None, "travel_reached")["class"] == "no_contact_within_bound"
    assert pl.classify_contact(touch, z, pl.HANDOVER_PREFIX + "contact_stall")["class"] == "expected"
    ref = part("B", xyz=(-0.046, 0.206, 0.9305), top=0.9305, sigma_h=0.002, move="fixed", roles=("ref",))
    held = held_frame(b_low=0.1219, sigma_offset=0.003)
    lp = pl.lowering_plan(ref, held, {"m_free_m": 0.055, "tol_m": (0.0275, 0.0275)}, sigma_touch_m=0.002)
    high = 0.9305 + 0.06 + 0.1219                                                 # an obstacle 6 cm above
    for event in ("contact_stall", pl.HANDOVER_PREFIX + "contact_stall", pl.HANDOVER_PREFIX + "gap_rise"):
        out = pl.support_classify(lp, high, held, 0.0, event=event)
        assert out["class"] == "blocked_above" and out["event"] == event and out["support_cue"] in pl.SUPPORT_EVENTS
    perched = pl.support_classify(lp, 0.9305 + 0.015 + 0.1219, held, 0.0, event=pl.HANDOVER_PREFIX + "gap_rise")
    assert perched["class"] == "perched"
    assert pl.support_classify(lp, high, held, 0.0, event=pl.HANDOVER_PREFIX + "travel_reached")["class"] == \
        "no_support_within_bound"


def test_clearance_retreat_and_width_consistency():
    hf, a = lever_scene()
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    retreat, th10 = pl.clearance_retreat_m(plan, 0.918, 0.955)
    assert retreat * 1000.0 == pytest.approx(5.6, abs=0.05) and th10.name == "P10_retreat_m"
    assert pl.clearance_retreat_m(plan, 0.900, 1.0345)[0] == pytest.approx(0.008)
    assert pl.clearance_retreat_m(plan, 0.905, 0.930)[0] == 0.0
    good = pl.width_consistency(plan, 25.4)
    assert good["consistent"] and good["sep_mm"] == pytest.approx(24.4) and good["band_mm"] == pytest.approx(3.0)
    assert not pl.width_consistency(plan, 31.0)["consistent"]
    assert not pl.width_consistency(plan, 18.0)["consistent"]


# ------------------------------------------------------------------------------------------------ 8 transit
def test_free_path_clears_a_tall_obstacle_and_a_keepout():
    hf = field(half=0.3)
    stamp(hf, rect_polygon((0.0, 0.047), (1.0, 0.0), 0.02, 0.006), SUPPORT + 0.15)   # under a finger column
    keep = KeepOut("E", "clear_of", disk((0.08, 0.0), 0.02, 16), SUPPORT, 1.12, ())   # under the palm only
    pose = ROBOT.pose(state8((-0.15, 0.0, 0.95), -math.pi / 2.0, 79.9))              # jaw -y: fingers at +-y
    out = pl.free_path((-0.15, 0.0, 0.95), (0.15, 0.0, 0.95), hf, (keep,), ROBOT, pose)
    assert out["transit_z"] >= SUPPORT + 0.15 + T_F + 0.02 - 1e-9                     # fingertips over it
    assert out["transit_z"] >= 1.12 - ROBOT.p_min_m + 0.02 - 1e-9                    # palm over the keep-out
    assert out["transit_z"] == pytest.approx(1.12 - ROBOT.p_min_m + 0.02)
    assert all(s <= pl.M2_SEGMENT_M + 1e-9 for s in out["segments"])
    assert out["waypoints"][0] == pytest.approx([-0.15, 0.0, 0.95])
    assert out["waypoints"][-1] == pytest.approx([0.15, 0.0, 0.95])
    zs = [p[2] for p in out["waypoints"]]
    assert max(zs) == pytest.approx(out["transit_z"])
    no_keep = pl.free_path((-0.15, 0.0, 0.95), (0.15, 0.0, 0.95), hf, (), ROBOT, pose)
    assert no_keep["transit_z"] == pytest.approx(SUPPORT + 0.15 + T_F + 0.02) and no_keep["transit_z"] < out[
        "transit_z"]
    assert out["goal_clear"] is False and out["waypoints"][1][:2] == pytest.approx([-0.15, 0.0])
    as_json(out)


def task2_like():
    """Support plane-filled; a 19 cm destination region (top 0.9305); a rotary part's articulated keep-out near it;
    the held object's pick-time cells (top 1.05 with fills) around (0.030, -0.007)."""
    hf = field(source="plane", centre=(-0.05, 0.05), half=0.35)
    region = rect_polygon((-0.046, 0.206), (1.0, 0.0), 0.095, 0.095)
    region_cells = stamp(hf, region, 0.9305)
    pot = stamp(hf, disk((0.030, -0.007), 0.045), 1.05, source="obstacle")
    rotary = part("A", xyz=(-0.185, 0.195, 0.955), top=0.955, support=SUPPORT, move="hinged",
                  circle=(np.array([-0.185, 0.195]), 0.0476, 0.5))
    keep = pl.keepout_for(rotary, "articulated", sigma_loc_m=0.002, sigma_h_m=0.002, theta_part_rad=0.4)
    ref = part("B", xyz=(-0.046, 0.206, 0.9305), top=0.9305, support=SUPPORT, move="fixed", region=region,
               cells=region_cells, roles=("ref",))
    return hf, keep, pot, ref


def test_transit_excludes_the_held_object_own_cells():
    hf, keep, pot, _ = task2_like()
    pose = ROBOT.pose(state8((0.0335, -0.0866, 1.047), 0.0, 14.0))                # jaw +x, R0 +y (object side)
    held = held_frame(vacated=pot)
    out = pl.transit_path(held, (0.0335, -0.0866, 1.047), (-0.046, 0.126), hf, (keep,), ROBOT, pose)
    assert out["ok"] and out["overhang_exit"] is None
    assert out["carry_z"] == pytest.approx(0.975 + 0.02 + 0.1219, abs=0.003)       # ~1.115, the keep-out is swept
    assert all(s <= pl.M2_SEGMENT_M + 1e-9 for s in out["segments"])
    assert out["waypoints"][-1][:2] == pytest.approx([-0.046, 0.126])
    assert out["waypoints"][-1][2] == pytest.approx(out["carry_z"])
    hf.z[pot[:, 0], pot[:, 1]] = 1.040                                              # not vacated (and not above
    not_excluded = pl.transit_path(held_frame(vacated=None), (0.0335, -0.0866, 1.047), (-0.046, 0.126), hf,
                                   (keep,), ROBOT, pose)                           # the grip): its own obstacle
    assert not_excluded["carry_z"] > 1.17
    as_json(out)


def test_transit_moves_out_from_under_an_overhang_first():
    hf = field(half=0.3)
    stamp(hf, rect_polygon((0.07, 0.08), (1.0, 0.0), 0.05, 0.08), 1.10)            # an overhang over the +x side
    pose = ROBOT.pose(state8((0.0, 0.0, 1.05), 0.0, 14.0))
    held = held_frame(offset_h=(0.08, 0.0, 0.0), r_obj=0.03)
    out = pl.transit_path(held, (0.0, 0.0, 1.05), (-0.15, -0.1), hf, (), ROBOT, pose)
    assert out["ok"] and out["overhang_exit"] is not None
    first = out["waypoints"][1]
    assert first[2] == pytest.approx(1.05) and first[0] < 0.0                    # horizontal, away from it
    assert out["overhang_exit"]["distance_m"] > 0.0
    assert out["waypoints"][-1][:2] == pytest.approx([-0.15, -0.1])
    clear = pl.transit_path(held, (-0.12, -0.08, 1.05), (-0.15, -0.1), hf, (), ROBOT, pose)
    assert clear["overhang_exit"] is None


# ------------------------------------------------------------------------------------------------ 9 aim
def test_aim_target_tolerance_and_free_spot():
    hf = field()
    region = rect_polygon((0.0, 0.0), (1.0, 0.0), 0.095, 0.095)
    cells = stamp(hf, region, 0.9305)
    ref = part("B", xyz=(0.0, 0.0, 0.9305), top=0.9305, sigma_h=0.002, move="fixed", region=region, cells=cells,
               roles=("ref",))
    pose = ROBOT.pose(state8((0.0, -0.1, 1.12), 0.0, 14.0))
    held = held_frame(r_obj=0.035, sigma_offset=0.003)
    aim = pl.aim_target(ref, held, hf, (), pose)
    assert aim["ok"] and not aim["free_spot"] and aim["fits"]
    assert aim["target_obj_xy"] == pytest.approx([0.0, 0.0], abs=0.003)
    assert aim["target_grip_xy"] == pytest.approx([0.0, -0.08], abs=0.003)
    assert aim["m_free_m"] == pytest.approx(aim["r_free_m"] - 0.035)
    assert aim["tol_m"][0] == pytest.approx(max(0.005, 0.5 * aim["m_free_m"])) and aim["tol_m"][0] > 0.02
    stamp(hf, rect_polygon((0.0475, 0.0), (1.0, 0.0), 0.0475, 0.095), 1.00)       # half occupied
    half = pl.aim_target(ref, held, hf, (), pose)
    assert half["ok"] and half["free_spot"] and half["fits"]
    assert half["target_obj_xy"][0] < -0.03
    assert half["occupied_fraction"] == pytest.approx(0.5, abs=0.1)


# ------------------------------------------------------------------------------------------------ 10 lowering
def test_lowering_plan_and_support_classes_with_cp_numbers():
    ref = part("B", xyz=(-0.046, 0.206, 0.9305), top=0.9305, sigma_h=0.002, move="fixed", roles=("ref",))
    held = held_frame(b_low=0.1219, sigma_offset=0.003)
    lp = pl.lowering_plan(ref, held, {"m_free_m": 0.055, "tol_m": (0.0275, 0.0275)}, sigma_touch_m=0.002)
    assert lp["S_dest"] == pytest.approx(0.9305) and lp["expected_support_grip_z"] == pytest.approx(1.0524)
    assert lp["tol_h_m"] == pytest.approx(0.01)
    assert lp["fast_until_grip_z"] == pytest.approx(1.0524 + 0.03 + 3.0 * math.hypot(0.002, 0.001))
    assert lp["bound_grip_z"] == pytest.approx(1.0524 - 0.02)
    assert lp["drift_abort_m"] == pytest.approx(0.0275)
    on = pl.support_classify(lp, 1.0516, held, 0.004, event="contact_stall")
    assert on["class"] == "on_surface" and on["bottom_z"] == pytest.approx(0.9297)
    raised = pl.support_classify(lp, 0.9305 + 0.04 + 0.1219, held, 0.004, event="contact_stall")
    assert raised["class"] in ("perched", "blocked_above")
    assert pl.support_classify(lp, 0.9305 + 0.015 + 0.1219, held, 0.004, event="contact_stall")["class"] == "perched"
    missed = pl.support_classify(lp, SUPPORT + 0.1219, held, 0.05, event="contact_stall")
    assert missed["class"] == "missed_surface"
    assert pl.support_classify(lp, lp["bound_grip_z"], held, 0.0, event="bound_reached")["class"] == \
        "no_support_within_bound"
    assert pl.support_classify(lp, 1.0516, held, 0.05, event="contact_stall")["class"] == "ambiguous"
    as_json(lp)
    as_json(on)


# ------------------------------------------------------------------------------------------------ 11 destination
def test_destination_sigma_rule():
    ok = part("B", xyz=(0, 0, 0.9305), top=0.9305, sigma_h=0.004, support=SUPPORT, roles=("ref",))
    bad = part("B", xyz=(0, 0, 0.9305), top=0.9305, sigma_h=0.006, support=SUPPORT, roles=("ref",))
    assert pl.destination_precondition(ok)["ok"]
    out = pl.destination_precondition(bad)
    assert not out["ok"] and out["values"]["sigma_top_limit_m"] == pytest.approx(0.5 * 0.0305 / 3.0)


# ------------------------------------------------------------------------------------------------ O5
def test_keepout_for_kinds():
    square = rect_polygon((0.0, 0.0), (1.0, 0.0), 0.025, 0.025)
    p = part("E", xyz=(0.0, 0.0, 0.95), top=0.95, sigma_h=0.002, support=SUPPORT, region=square, roles=("keepout",))
    k = pl.keepout_for(p, "clear_of", sigma_loc_m=0.005, sigma_h_m=0.002)
    assert k.kind == "clear_of" and k.z_low == pytest.approx(SUPPORT) and k.z_high == pytest.approx(0.97)
    assert np.max(np.abs(k.polygon_xy[:, 0])) == pytest.approx(0.045, abs=1e-9)    # 2.5 cm + 2 cm
    wide = pl.keepout_for(p, "hold_true", sigma_loc_m=0.01, sigma_h_m=0.01)
    assert np.max(np.abs(wide.polygon_xy[:, 0])) == pytest.approx(0.055, abs=1e-9)
    assert wide.thresholds[0].label == "ONLINE" and wide.z_high == pytest.approx(0.98)
    rotary = part("A", xyz=(0.1, 0.0, 0.955), top=0.955, move="hinged", circle=(np.array([0.1, 0.0]), 0.0476, 0.5))
    circle = pl.keepout_for(rotary, "articulated", sigma_loc_m=0.002, sigma_h_m=0.002)
    radii = np.linalg.norm(circle.polygon_xy - [0.1, 0.0], axis=1)
    assert radii.min() >= 0.0476 + 0.02 - 1e-9
    bar = part("A", xyz=(0.0, 0.0, 0.955), top=0.955, move="hinged",
               region=rect_polygon((0.0, 0.0), (1.0, 0.0), 0.05, 0.01))
    turned = pl.keepout_for(bar, "articulated", sigma_loc_m=0.0, sigma_h_m=0.0, theta_part_rad=math.pi / 2.0)
    assert np.ptp(turned.polygon_xy[:, 1]) > np.ptp(turned.polygon_xy[:, 0])        # rotated a quarter turn
    with pytest.raises(ValueError):
        pl.keepout_for(p, "upright", sigma_loc_m=0.0, sigma_h_m=0.0)
    assert pl.horizontal_sigma_m(p) == pytest.approx(0.002)


# ------------------------------------------------------------------------------------------------ hygiene
def test_every_result_carries_labelled_thresholds_and_is_json_safe():
    hf, a = lever_scene()
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    names = {t.name for t in plan.thresholds}
    assert {"P1_e_lat_mm", "P2_width_feasible_mm", "P2_width_infeasible_mm", "P3_pad_overlap_mm",
            "P4_known_fraction", "P5_sweep_margin_m", "P6_palm_margin_m", "P7_footprint_margin_m", "P8_band_m",
            "P9_bound_grip_z_m", "P10_retreat_m", "P11_hover_clearance_m", "P12_turn_clearance_m",
            "P16_column_inflate_m"} <= names
    for t in plan.thresholds:
        assert isinstance(t, Threshold) and t.label in th.LABEL_CLASSES and t.derivation.strip()
        assert th.threshold_id(t.name) is not None, t.name
    height = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    assert {"P6_target_grip_z_m", "P6_landing_margin_m"} <= {t.name for t in height.thresholds}
    hf2, c, d = thin_part_scene(0.006)
    infeasible = pl.plan_contact(c, hf2, inputs(), (), purpose="carry", next_held=d)
    assert infeasible.thresholds and all(isinstance(t, Threshold) for t in infeasible.thresholds)
    payload = pl.trace_payload(infeasible)
    assert len(payload["rejections"]) <= pl.TRACE_MAX_REJECTIONS and payload["infeasible"] == "palm_clearance"
    for value in (plan.to_dict(), height.to_dict(), infeasible.to_dict(), pl.trace_payload(plan), payload):
        json.dumps(value, allow_nan=False)
    assert plan.to_dict()["pid"] == "A" and "text" not in plan.to_dict()


def test_planner_never_reads_the_part_text():
    hf, a = lever_scene()
    renamed = PartRecord(**{**a.__dict__, "text": "something else entirely"})
    first = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    second = pl.plan_contact(renamed, hf, inputs(), (), purpose="turn")
    assert first.to_dict() == second.to_dict()


def test_planner_inputs_build_and_validation():
    start = ROBOT.pose(state8((0.0, 0.0, 1.19), -math.pi / 2.0, 79.9)).quaternion_xyzw
    built = pl.PlannerInputs.build(ROBOT, state8((0.0, 0.0, 1.19), -math.pi / 2.0, 79.9), g_open_mm=79.9,
                                   start_quaternion=start)
    assert built.sigma_align_m == pl.SIGMA_ALIGN_PRIOR_M and built.finger_q_mm == pytest.approx(39.95)
    assert built.yaw_budget_left[1] == pytest.approx(built.yaw_budget_left[-1])
    with pytest.raises(ValueError):
        pl.PlannerInputs(ROBOT, 79.9, 0.003, {1: 1.0}, built.pose, {}, None)
    with pytest.raises(ValueError):
        pl.PlannerInputs(ROBOT, -1.0, 0.003, {1: 1.0, -1: 1.0}, built.pose, {}, None)


def test_finger_columns_public_helper():
    hf, a = lever_scene()
    cols = pl.finger_columns((0.0, 0.004), -math.pi / 2.0, hf, ROBOT, e_lat_m=0.008, support_z=0.918,
                             finger_q_mm=39.95, exclude_cells=a.region_cells)
    assert [c.finger for c in cols] == [1, 2] and all(c.landing_z == pytest.approx(0.918) for c in cols)
    beyond = pl.finger_columns((0.2, 0.2), 0.0, hf, ROBOT, e_lat_m=0.0, support_z=0.95, finger_q_mm=40.0)
    assert all(c.landing_z == pytest.approx(SUPPORT) and not c.straddles_edge for c in beyond)   # a lower landing
    hidden = field()
    unknown(hidden, rect_polygon((0.0, 0.0), (1.0, 0.0), 0.1, 0.1))
    assert all(c.landing_z is None and c.known_fraction == 0.0 for c in pl.finger_columns(
        (0.0, 0.0), 0.0, hidden, ROBOT, e_lat_m=0.0, support_z=None, finger_q_mm=40.0))


# ================================================================ design review 1 (decision §2.4, §5.1 item 3)
def _cells_at(hf, centre, half=0.002):
    """The HeightField cells whose centres lie within +-half of ``centre`` (one 5 mm cell for the default)."""
    return hf.cells_in_polygon(rect_polygon(centre, (1.0, 0.0), half, half))


def _column_poly(plan_or_cols, finger):
    cols = plan_or_cols.columns if hasattr(plan_or_cols, "columns") else plan_or_cols
    return next(c.polygon_xy for c in cols if c.finger == finger)


def p7_refused_at_target(result):
    """Gap round (IF-HS): a P7 refusal at the full-overlap target g* either stands (footprint_blocked) or the candidate
    is admitted only at a stop raised by the P7 deficit, still on the stored field (no IF-PV parent reading)."""
    if isinstance(result, pl.Infeasible):
        return result.reason == "footprint_blocked"
    return bool(result.checks.get("hs_raise_mm", 0.0) > 0.0 and "p7_reading" not in result.checks)


def test_review1_constants_and_restatements():
    assert pl.P11_CAP_M == 0.10 and pl.PLANNER_VERSION.endswith("-7")
    # BPR-7: the registered-record reach multiplier is a named RULING constant (decision §2.4 P-E, §5.1 item 3)
    assert pl.P11_RECORD_SIGMA_K == 1.0 and pl.LABELS["P11_RECORD_SIGMA_K"][0] == "RULING"
    assert "P-F" in pl.LABELS["O5_SIGMA_K"][1] and "P-C" in pl.LABELS["V5_COHERENCE"][1]
    # review 2: the isotropy definition (>= 4 cells, aspect < 1.5) is the lead's ratified sign-off, so the aspect that
    # defines a region PCA axis is no longer pending
    assert "PENDING" not in pl.LABELS["AXIS_MIN_ASPECT"][1] and "ratified" in pl.LABELS["AXIS_MIN_ASPECT"][1]
    assert (pl.AXIS_MIN_ASPECT, pl.AXIS_MIN_REGION_CELLS) == (1.5, 4)
    scene = pytest.importorskip("gpt_field.agent_speedup_v2_scene")
    primitives = pytest.importorskip("gpt_field.agent_speedup_v2_primitives")
    # the scene's D10 aspect / cell numbers are retired as D10 deciders in review 2 (S-11), kept logged: equal while
    # they exist
    assert getattr(scene, "D10_MIN_ASPECT", pl.AXIS_MIN_ASPECT) == pl.AXIS_MIN_ASPECT
    assert getattr(scene, "D10_MIN_REGION_CELLS", pl.AXIS_MIN_REGION_CELLS) == pl.AXIS_MIN_REGION_CELLS
    assert pl.VIS_PROBE_LEG_M == primitives.D2_PROBE_M
    # review-2 fixer: the build's BPR-2 band-fill rule (and its restated S-6 constants) is withdrawn
    assert not hasattr(pl, "_band_fills") and not any(name.startswith("PB_") for name in pl.LABELS)
    assert pl.harmonic_contrast_min() == pytest.approx((1.5 - 1.0) / (1.5 + 1.0))
    status = th.v6_constant_status()
    assert status["P11_CAP_M"]["state"] == "labelled" and status["SIGMA_VC_M"]["state"] == "labelled"


# ------------------------------------------------------------------------------------- P-A RC-1a finger columns
def test_rc1a_fill_only_column_lands_on_the_support_and_widens_the_band():
    """Remedy 2 T1 test 1: a column holding only a D7b fill above the landing: touch_stop admitted with the support as
    its landing, upper_z = min(fill + t_f + band, cap); a touch at the fill height below the cap is expected, above
    the cap blocked_above_part."""
    hf, a = lever_scene()
    probe = pl.evaluate_candidate(a, hf, inputs(), "top_down", -math.pi / 2.0, (), purpose="turn")
    poly = _column_poly(probe, 1)
    stamp(hf, poly, 0.935, sigma=0.004, source="obstacle")
    plan = pl.evaluate_candidate(a, hf, inputs(), "top_down", -math.pi / 2.0, (), purpose="turn")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "touch_stop"
    col = next(c for c in plan.columns if c.finger == 1)
    assert col.landing_source == "support" and col.landing_z == pytest.approx(0.918)
    assert col.unverified_z == pytest.approx(0.935) and col.unverified_count > 0
    other = next(c for c in plan.columns if c.finger == 2)
    assert other.landing_source == "verified" and other.landing_z == pytest.approx(0.918)
    e = next(x for x in plan.expected if x.column == 1)
    cap = 0.955 + 0.0044 - 0.008
    assert e.grip_z == pytest.approx(0.918 + T_F) and e.unverified_grip_z == pytest.approx(0.935 + T_F)
    assert e.upper_z == pytest.approx(min(0.935 + T_F + e.band_m, cap))
    assert pl.classify_contact(plan, 0.935 + T_F, "contact_stall")["class"] == "expected"
    assert pl.classify_contact(plan, cap + 0.001, "contact_touch")["class"] == "blocked_above_part"
    assert plan.checks["unverified_cells"] >= 1 and plan.checks["admission"] == "marginal" and plan.marginal
    as_json(plan.to_dict())


def test_rc1a_a_measured_cell_at_the_same_height_still_refuses():
    """Remedy 2 T1 test 1 (negative): at 0.945 a D7b fill is admitted (the touch on it is above the P3 cap, so it is
    blocked_above_part, never a close), while a kappa cell at the same height refuses (P3 / P6 column_blocked)."""
    hf, a = lever_scene()
    probe = pl.evaluate_candidate(a, hf, inputs(), "top_down", -math.pi / 2.0, (), purpose="turn")
    poly = _column_poly(probe, 1)
    filled = field()
    filled.z[:], filled.source[:] = hf.z, hf.source
    stamp(filled, poly, 0.945, sigma=0.004, source="obstacle")
    plan = pl.evaluate_candidate(a, filled, inputs(), "top_down", -math.pi / 2.0, (), purpose="turn")
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "touch_stop"
    assert pl.classify_contact(plan, 0.945 + T_F, "contact_stall")["class"] == "blocked_above_part"
    stamp(hf, poly, 0.945, sigma=0.001, source="kappa")
    refused = pl.evaluate_candidate(a, hf, inputs(), "top_down", -math.pi / 2.0, (), purpose="turn")
    assert isinstance(refused, pl.CandidateRejection) and refused.reason == "column_blocked"
    assert refused.values["by_rule"]["touch_stop"]["check"] == "P3"
    assert refused.values["by_rule"]["height_stop"]["check"] == "P6"


def test_rc1a_height_stop_lists_unverified_column_cells():
    """RC-1a on height_stop: P6-landing is tested on the verified landing; a D7b fill above the landing limit is
    listed (values / checks unverified_cells) and makes the plan marginal; a kappa cell there refuses (P6)."""
    hf, c, d = thin_part_scene(0.002)
    stamp(hf, rect_polygon((0.045, 0.0), (1.0, 0.0), 0.002, 0.002), 1.03, sigma=0.002, source="obstacle")
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    assert plan.checks["column_unverified"] >= 1 and plan.checks["unverified_cells"] >= 1
    assert plan.checks["admission"] == "marginal" and plan.marginal
    assert all(col.landing_z == pytest.approx(SUPPORT) for col in plan.columns)
    hf2, c2, d2 = thin_part_scene(0.002)
    stamp(hf2, rect_polygon((0.045, 0.0), (1.0, 0.0), 0.002, 0.002), 1.03, sigma=0.002, source="kappa")
    refused = pl.plan_contact(c2, hf2, inputs(), (), purpose="carry", next_held=d2)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked"
    assert {r.values["by_rule"]["height_stop"]["check"] for r in refused.rejections} == {"P6"}


# ------------------------------------------------------------------------------------- P-A RC-1b closing sweeps
def test_rc1b_sweeps_fill_and_near_edge_cells_are_close_verified():
    """Remedy 2 T1 test 2: a D7b fill in the sweep is admitted (close_verified >= 1); a kappa cell at |along| <= w/2 +
    e_lat + cell/2 is admitted; a kappa cell beyond that band refuses (P5)."""
    for centre, source in (((0.020, -0.005), "obstacle"), ((0.015, -0.005), "kappa")):
        hf, c, d = thin_part_scene(0.002)
        stamp(hf, rect_polygon(centre, (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source=source)
        plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop", (centre, source)
        assert plan.checks["close_verified"] >= 1 and plan.checks["admission"] == "marginal"
        assert plan.checks["P5_sweep_ok"] is True
    # the near-edge band: w/2 + e_lat + cell/2 = 6.5 + 8 + 2.5 = 17 mm; a kappa cell at 25 mm refuses
    hf, c, d = thin_part_scene(0.002)
    stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked"
    assert {r.values["by_rule"]["height_stop"]["check"] for r in refused.rejections} == {"P5"}
    # a clean scene has nothing to verify (a measured support: F6-B makes a D7a-only landing unverified)
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    clean = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert clean.checks["unverified_cells"] == 0 and clean.checks["admission"] == "clean" and not clean.marginal


# ------------------------------------------------------------------------------------- P-A RC-1c / RC-1d (P7)
def test_rc1c_no_sigma_double_count_at_a_fill_argmax():
    """Remedy 2 T1 test 3: P7 with a D7b argmax adds no sigma (its stored height already holds 3 sigma_parent); with a
    kappa argmax the margin is unchanged (3 sigma of the cell)."""
    centre = (0.070, 0.020)                                   # in the hand footprint, outside the slot and columns
    out = {}
    for source in ("obstacle", "kappa"):
        hf, c, d = thin_part_scene(0.001)
        stamp(hf, rect_polygon(centre, (1.0, 0.0), 0.002, 0.002), 1.043, sigma=0.004, source=source)
        s = pl._Field(hf).stats(_cells_at(hf, centre))
        assert s["sigma"] == (pytest.approx(0.0) if source == "obstacle" else pytest.approx(0.004))
        out[source] = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(out["obstacle"], pl.ContactPlan)
    p7 = next(t for t in out["obstacle"].thresholds if t.name == "P7_footprint_margin_m")
    assert p7.value == pytest.approx(0.005)                   # max(5 mm, 3 sigma_h = 3 mm): no fill sigma
    assert p7_refused_at_target(out["kappa"])                # gap round: IF-HS may raise the stop over it
    p7k = next(t for t in out["kappa"].thresholds if t.name == "P7_footprint_margin_m")
    assert p7k.value == pytest.approx(0.012)                  # 3 x 4 mm at the kappa argmax (unchanged)


def test_rc1d_part_fill_cells_are_skipped_by_p7_only():
    """Remedy 2 T1 test 4: fills listed in part_fill_cells are skipped by P7 only; a column still sees them."""
    hf, c, d = thin_part_scene(0.001)
    footprint_fill = stamp(hf, rect_polygon((0.070, 0.020), (1.0, 0.0), 0.002, 0.002), 1.05, sigma=0.002,
                           source="obstacle")
    blocked = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert p7_refused_at_target(blocked)
    column_fill = stamp(hf, rect_polygon((0.045, 0.0), (1.0, 0.0), 0.002, 0.002), 1.03, sigma=0.002,
                        source="obstacle")
    fills = np.vstack([footprint_fill, column_fill])
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, part_fill_cells=fills)
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    seen = [col.unverified_z for col in plan.columns if col.unverified_z is not None]
    assert seen and max(seen) == pytest.approx(1.03)          # the column still sees the listed fill
    assert plan.checks["column_unverified"] >= 1
    one = pl.evaluate_candidate(c, hf, inputs(), "top_down", 0.0, (), purpose="carry", next_held=d,
                                part_fill_cells=fills)
    assert isinstance(one, pl.ContactPlan)


def test_unverified_admission_ranks_after_clean_candidates():
    """Remedy 1 §2.6 label: a candidate admitted only through unverified cells is marginal and ranked after clean
    candidates, even with a higher score."""
    hf, body = body_scene()
    for x in (0.045, 0.050, 0.055, -0.045, -0.050, -0.055):
        stamp(hf, rect_polygon((x, 0.0), (1.0, 0.0), 0.002, 0.002), 1.015, sigma=0.002, source="obstacle")
    probe = inputs(jaw=0.0, g_open=78.5, sigma_align=0.002)
    at_zero = pl.evaluate_candidate(body, hf, probe, "top_down", 0.0, (), purpose="carry")
    assert isinstance(at_zero, pl.ContactPlan) and at_zero.checks["admission"] == "marginal"
    best = pl.plan_contact(body, hf, probe, (), purpose="carry")
    assert isinstance(best, pl.ContactPlan) and best.checks["admission"] == "clean"
    assert abs(math.degrees(best.yaw_rad)) % 180.0 >= 15.0 - 1e-6           # the fills sit under the 0 deg columns
    assert best.score < at_zero.score + pl.SCORE_MARGINAL_PENALTY             # it ranks first although it scores lower


# ------------------------------------------------------------------------------------- P-B fixture landing
def test_pb_fixture_landing_below_the_band_is_expected_and_measured():
    hf, a = lever_scene()
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    assert plan.depth_rule == "touch_stop" and plan.move_prior == "hinged"
    lowest = min(plan.expected, key=lambda e: e.grip_z)
    z = 0.905
    assert plan.bound_grip_z < z < lowest.grip_z - lowest.band_m
    out = pl.classify_contact(plan, z, "contact_stall")
    assert out["class"] == "expected" and out["landing_measured"] is True
    assert out["surface_z"] == pytest.approx(z - T_F)
    assert pl.classify_contact(plan, z, "contact_touch")["landing_measured"] is True
    below_bound = pl.classify_contact(plan, plan.bound_grip_z - 0.002, "contact_stall")
    assert below_bound["class"] == "no_contact_where_expected" and not below_bound.get("landing_measured")
    assert pl.classify_contact(plan, plan.expected[0].grip_z, "contact_stall").get("landing_measured") is False
    # the same contact on a free part keeps its class
    hf, b = wall_scene()
    free = pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry")
    assert free.depth_rule == "touch_stop" and free.move_prior == "free"
    low = min(free.expected, key=lambda e: e.grip_z)
    zf = (free.bound_grip_z + low.grip_z - low.band_m) / 2.0
    out = pl.classify_contact(free, zf, "contact_stall")
    assert out["class"] == "no_contact_where_expected" and not out.get("landing_measured")
    as_json(out)


# ------------------------------------------------------------------------------------- P-C width-harmonic jaw
def _harmonic_part(hf, *, region_angle=math.pi / 2.0, widths=None, move="free", length=0.056, across=0.013):
    """A non-elongated record of the thin part: D10 not stored, widths measured along world y (the long axis)."""
    axis = np.array([math.cos(region_angle), math.sin(region_angle)])
    poly = rect_polygon((0.0, 0.016), axis, length / 2.0, across / 2.0)
    cells = hf.cells_in_polygon(poly)
    return part("C", xyz=(0.0, 0.0, 1.0345), top=1.0345, sigma_h=0.002, support=SUPPORT, move=move, region=poly,
                cells=cells, widths=widths_bar(across, length, math.pi / 2.0) if widths is None else widths)


def test_rc7_harmonic_jaw_when_the_region_axis_agrees():
    """Remedy 2 T1 test 7 under P-C: a non-elongated part whose width harmonic (contrast >= 0.2) and region PCA axis
    agree within YAW_GRID_RAD / 2: the first candidates are the harmonic's narrow jaw and its 180 deg twin."""
    hf, _c, d = thin_part_scene(0.002)
    c = _harmonic_part(hf)
    axis = pl.jaw_axis(c, hf)
    assert axis["source"] == "width_harmonic" and not axis["refine_axis"]
    assert axis["contrast"] >= pl.harmonic_contrast_min() and axis["agree_rad"] <= YAW_GRID_RAD / 2.0
    yaws = sorted(round(abs(math.degrees(pl._wrap(y)))) for y in pl.jaw_yaws(c, "top_down", inputs(), hf=hf))
    assert yaws == [0, 180]
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and np.allclose(np.abs(plan.jaw_world), (1.0, 0.0, 0.0), atol=1e-9)
    assert plan.checks["axis_source"] == "width_harmonic"
    # an elongated record is unchanged (D10 axis)
    hf2, c2, d2 = thin_part_scene(0.002)
    assert pl.jaw_axis(c2, hf2)["source"] == "d10"
    assert pl.plan_contact(c2, hf2, inputs(), (), purpose="carry", next_held=d2).checks["axis_source"] == "d10"


def test_rc7_disagreement_or_no_axis_asks_to_refine_the_axis():
    """P-C: the harmonic and the region axis 30 deg apart -> Infeasible(geometry_unmeasured, refine_axis=True); the
    same with a region too small for a PCA axis; a free part with a flat width function but an elongated region is
    unmeasured too (the plain 15 deg grid is not used for free grip parts)."""
    hf, _c, d = thin_part_scene(0.002)
    for record in (_harmonic_part(hf, region_angle=math.pi / 2.0 + math.radians(30.0)),
                   PartRecord(**{**_harmonic_part(hf).__dict__, "region_cells": _cells_at(hf, (0.0, 0.016)),
                                 "region_xy": None}),
                   _harmonic_part(hf, widths=widths_const(0.013, 0.001))):
        result = pl.plan_contact(record, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"
        assert result.refine_axis is True and pl.needs_axis_refinement(result)
        assert not pl.needs_height_refinement(result)
        assert pl.trace_payload(result)["refine_axis"] is True
        assert pl.jaw_yaws(record, "top_down", inputs(), hf=hf) == []
        as_json(result.to_dict())


def test_width_harmonic_needs_an_even_half_turn():
    """BPR-5: the discrete second harmonic is unbiased only on the full half-turn grid. A flat width function sampled
    over part of it (1 row, 2 rows at 0 / 15 deg, 6 rows over 0-75 deg) reads a large contrast (2.0, 1.93, 1.29)
    with the DFT formula; width_harmonic returns None there, and a free part's yaw is then unmeasured (refine_axis),
    a fixture keeps the grid. The full 12-row grid (also when offset from 0) keeps the harmonic."""
    hf, _c, d = thin_part_scene(0.002)
    full = widths_const(0.020, 0.001)
    flat = _harmonic_part(hf, widths=full)
    assert pl.width_coverage(full) and pl.width_harmonic(flat)[1] == pytest.approx(0.0, abs=1e-9)
    offset = tuple((psi + math.radians(3.0), m) for psi, m in full)
    assert pl.width_coverage(offset)
    shuffled = tuple(reversed(widths_bar(0.013, 0.056, math.pi / 2.0)))
    assert pl.width_coverage(shuffled)
    assert pl.width_harmonic(_harmonic_part(hf, widths=shuffled))[1] >= pl.harmonic_contrast_min()
    for rows in (full[:1], full[:2], full[:6], full[:11], full + full[:1], full[:6] + full[:6]):
        assert not pl.width_coverage(rows)
        record = _harmonic_part(hf, widths=rows)
        assert pl.width_harmonic(record) is None
        axis = pl.jaw_axis(record, hf)
        assert axis["source"] == "unmeasured" and axis["refine_axis"] is True and axis["width_coverage"] is False
        result = pl.plan_contact(record, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.Infeasible) and result.refine_axis is True
        fixture = _harmonic_part(hf, widths=rows, move="hinged")
        assert pl.jaw_axis(fixture, hf)["source"] == "grid"
    # the latent hazard: 6 flat rows over 0-75 deg read a spurious long axis at 37.5 deg; a region along it would
    # have "agreed" and admitted a harmonic jaw; the yaw is unmeasured instead
    along = _harmonic_part(hf, widths=full[:6], region_angle=math.radians(37.5))
    assert pl.jaw_axis(along, hf)["source"] == "unmeasured" and pl.jaw_yaws(along, "top_down", inputs(), hf=hf) == []
    # what the coverage rule guards against: the DFT contrast of a partial flat function
    for rows, contrast in ((full[:1], 2.0), (full[:2], 1.93), (full[:6], 1.29)):
        psi = np.array([p for p, _ in rows])
        w = np.array([m.value for _, m in rows])
        c2 = 2.0 * math.hypot(float(np.sum(w * np.cos(2 * psi))), float(np.sum(w * np.sin(2 * psi)))) / len(w)
        assert c2 / float(w.mean()) == pytest.approx(contrast, abs=0.01)


def test_rc7_isotropic_parts_and_fixtures_keep_the_grid():
    """A free part whose width function AND region are both isotropic has a measured axis-free shape: the grid stays
    (the round free body); a fixture without a width axis keeps the grid as before."""
    hf, body = body_scene()
    assert pl.jaw_axis(body, hf)["source"] == "isotropic"
    assert len(pl.jaw_yaws(body, "top_down", inputs(jaw=0.0, g_open=78.5), hf=hf)) == 15
    hf, f = cylinder_scene()
    assert pl.jaw_axis(f, hf)["source"] == "isotropic"
    hf, _c, _d = thin_part_scene(0.002)
    fixture = _harmonic_part(hf, widths=widths_const(0.013, 0.001), move="hinged")
    assert pl.jaw_axis(fixture, hf)["source"] == "grid"
    disagree = _harmonic_part(hf, region_angle=math.pi / 2.0 + math.radians(30.0), move="hinged")
    assert pl.jaw_axis(disagree, hf)["refine_axis"] is True
    # a third witness (S-4' / R11): when the scene's image tensor is oriented (coherence >= V5_COHERENCE) although the
    # widths and the region read isotropic (an under-grown region of a thin part), the axis is unmeasured
    hf, body = body_scene()
    check = {"image_axis": np.array([0.0, 1.0]), "region_axis": np.array([1.0, 0.0]), "agree_rad": math.pi / 2.0,
             "accepted": False, "source": "not_elongated"}
    oriented = PartRecord(**{**body.__dict__, "axis_check": dict(check, coherence=0.64)})
    flat = PartRecord(**{**body.__dict__, "axis_check": dict(check, coherence=0.30)})
    assert pl.jaw_axis(oriented, hf)["source"] == "unmeasured" and pl.jaw_axis(oriented, hf)["refine_axis"]
    refused = pl.plan_contact(oriented, hf, inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    assert isinstance(refused, pl.Infeasible) and refused.refine_axis
    assert pl.jaw_axis(flat, hf)["source"] == "isotropic"
    fixture = PartRecord(**{**oriented.__dict__, "move_prior": "hinged"})
    assert pl.jaw_axis(fixture, hf)["source"] == "unmeasured"


# ------------------------------------------------------------------------------------- P-D G13+
def test_g13_plus_flags_a_misaligned_close_on_free_parts_only():
    """Remedy 2 T1 test 8: misaligned when (gap - 1.0) - w_min > max(G13 floor, k sigma_w) + 16 mm sin(7.5 deg):
    flags at w_min + band + 2.09 mm + eps, not at w_min + 4 mm with a 3 mm band; fixtures never."""
    hf, c, d = thin_part_scene(0.002)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    yaw_term = 16.0 * math.sin(math.radians(7.5))
    assert plan.width["w_narrow_mm"] == pytest.approx(13.0)
    flag = pl.width_consistency(plan, 1.0 + 13.0 + 3.0 + yaw_term + 0.01)
    assert flag["misaligned"] is True and flag["misaligned_band_mm"] == pytest.approx(3.0 + yaw_term)
    assert flag["excess_mm"] == pytest.approx(3.0 + yaw_term + 0.01)
    ok = pl.width_consistency(plan, 1.0 + 13.0 + 4.0)
    assert ok["misaligned"] is False
    assert {t.name for t in flag["thresholds"]} >= {"G13_width_band_mm", "G13_misaligned_band_mm"}
    # BPR-7: decision §3 labels the G13+ band ROBOT + DD (one class per Threshold: DD, both named)
    mis = next(t for t in flag["thresholds"] if t.name == "G13_misaligned_band_mm")
    assert mis.label == "DD" and "ROBOT + DD" in mis.derivation and "PAD_LENGTH_R0_MM" in mis.derivation
    hf, a = lever_scene()
    lever = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    assert pl.width_consistency(lever, 60.0)["misaligned"] is False
    as_json(flag)


# ------------------------------------------------------------------------------------- P-E hover
def test_p11_cap_flags_a_first_stage_hover():
    hf = field()
    for sigma_h, clearance, first in ((0.09, 0.10, True), (0.015, 0.065, False), (0.002, 0.05, False)):
        seeded = PartRecord(pid="C", text="part C", move_prior="free", roles=frozenset({"grip"}),
                            xyz=np.array([0.0, 0.0, 0.93]), cov=np.eye(3) * 0.002 ** 2, sigma_h_m=sigma_h)
        out = pl.hover_for(seeded, hf, ROBOT, ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9)))
        assert out["clearance_m"] == pytest.approx(clearance) and out["first_stage"] is first
        assert out["first_stage_reasons"] == (["p11_cap"] if first else [])
        assert out["fingertip_z"] == pytest.approx(0.93 + clearance)
        p11 = next(t for t in out["thresholds"] if t.name == "P11_hover_clearance_m")
        assert p11.value == pytest.approx(clearance)


def test_hover_for_a_registered_record_uses_its_own_region():
    """P-E: a registered (not provisional) record's hover reference is its top and its region inflated by sigma_xy;
    an HF maximum beyond that (a tall neighbour within the seed's 3 sigma_xy) no longer raises it."""
    hf = field()
    region = rect_polygon((0.0, 0.0), (1.0, 0.0), 0.006, 0.02)
    cells = stamp(hf, region, 0.95)
    stamp(hf, rect_polygon((0.03, 0.0), (1.0, 0.0), 0.004, 0.004), 1.02)          # 2.4 cm beyond the region edge
    rec = part("C", xyz=(0.0, 0.0, 0.95), top=0.95, sigma_h=0.003, region=region, cells=cells, cov_sigma=0.01,
               widths=widths_bar(0.012, 0.04, math.pi / 2.0))
    pose = ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9))
    out = pl.hover_for(rec, hf, ROBOT, pose)
    assert out["registered"] is True and out["z_ref"] == pytest.approx(0.95)
    assert out["fingertip_z"] == pytest.approx(0.95 + 0.05)
    provisional = PartRecord(**{**rec.__dict__, "status": "seeded", "provisional": True})
    assert pl.hover_for(provisional, hf, ROBOT, pose)["z_ref"] == pytest.approx(1.02)
    near = field()
    near_cells = stamp(near, region, 0.95)
    stamp(near, rect_polygon((0.012, 0.0), (1.0, 0.0), 0.002, 0.002), 0.97)       # within sigma_xy of the region
    rec_near = PartRecord(**{**rec.__dict__, "region_cells": near_cells})
    assert pl.hover_for(rec_near, near, ROBOT, pose)["z_ref"] == pytest.approx(0.97)


def test_hover_visibility_keeps_the_part_above_the_fingertip_band():
    """P-E / remedy 1 P-2 (T1 test 12): the part's predicted wrist row stays above the fingertip band's top row by
    kappa D2 + the template radius; the hover moves along -R0 otherwise."""
    geometry = pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    hf = field()
    rec = part("C", xyz=(0.0, 0.0, 0.95), top=0.95, sigma_h=0.003, region=rect_polygon((0, 0), (1, 0), 0.006, 0.02),
               cells=stamp(hf, rect_polygon((0, 0), (1, 0), 0.006, 0.02), 0.95), widths=widths_const(0.012, 0.001))
    pose = ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9))
    f_px = 333.6
    plain = pl.hover_for(rec, hf, ROBOT, pose)
    target = np.asarray(plain["hover_grip_xyz"])

    def row(at):
        # BPR-6: the part's wrist row from the geometry module's own projection (upright frame), not the planner's
        # formula; the camera moves with the grip at the held orientation
        d_ = np.asarray(rec.xyz) - (at + (pose.camera - pose.grip))
        uv = geometry.project_direction(pose.R, d_, f_px)
        assert uv is not None
        return float(uv[1]), f_px / float(d_ @ pose.R[:, 2])
    v0, kappa0 = row(target)
    band_row = int(v0 + kappa0 * pl.VIS_PROBE_LEG_M + geometry.FT_PROBE_RADIUS_PX - 20)      # 20 px short
    out = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band_row)
    assert out["visibility_shift_m"] > 0.0
    moved = np.asarray(out["hover_grip_xyz"])
    r0 = pose.R[:, 0][:2] / np.linalg.norm(pose.R[:, 0][:2])
    assert float((moved[:2] - target[:2]) @ r0) < 0.0                       # along -R0
    v1, kappa1 = row(moved)
    assert v1 <= band_row - (kappa1 * pl.VIS_PROBE_LEG_M + geometry.FT_PROBE_RADIUS_PX) + 0.5
    clear = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band_row + 60)
    assert clear["visibility_shift_m"] == 0.0 and clear["hover_grip_xyz"] == pytest.approx(plain["hover_grip_xyz"])


# ------------------------------------------------------------------------------------- P-F lower-bound parts
def _lower_bound(pid, poly, hf, *, measured=0.95, upper=1.10):
    """A record whose top is only a lower bound (rulings R2): measured cells at ``measured``, top_upper_z ``upper``."""
    cells = stamp(hf, poly, measured)
    record = part(pid, xyz=(*np.mean(poly, axis=0), measured), top=measured, sigma_h=0.003, region=poly, cells=cells,
                  widths=widths_const(0.04, 0.001), roles=("carried",))
    return PartRecord(**{**record.__dict__, "top_bound": "lower", "top_upper_z": upper, "partial": True})


def test_hover_reference_ignores_a_lower_bound_neighbour_raise():
    """BPR-1 (P-E with P-F): a registered record next to a lower-bound neighbour keeps z_ref = its own top, even when
    the neighbour's raised footprint (its region inflated by O5_SIGMA_K sigma_xy) covers the record's cells; the
    raise still lifts the hover through P12 (the open hand clears the raised neighbour by the P12 margin), flagged
    first_stage "lower_bound". The same holds for a seeded record (the reference is the unraised field over its seed
    polygon), and a lower-bound part's own top_upper_z still sets its own reference."""
    hf = field()
    region = rect_polygon((0.0, 0.0), (1.0, 0.0), 0.006, 0.02)
    cells = stamp(hf, region, 0.95)
    rec = part("C", xyz=(0.0, 0.0, 0.95), top=0.95, sigma_h=0.003, region=region, cells=cells,
               widths=widths_bar(0.012, 0.04, math.pi / 2.0))
    nb = _lower_bound("E", rect_polygon((0.02, 0.0), (1.0, 0.0), 0.004, 0.004), hf, measured=0.95, upper=1.10)
    nb = PartRecord(**{**nb.__dict__, "cov": np.eye(3) * 0.005 ** 2})          # 3 sigma_xy = 15 mm: covers C
    raised_cells = pl._lower_bound_cells(nb, hf)
    assert len(pl._unique_cells([raised_cells, cells])) < len(raised_cells) + len(cells)   # the raise covers C
    pose = ROBOT.pose(state8((0.0, 0.0, 1.30), 0.0, 79.9))
    alone = pl.hover_for(rec, hf, ROBOT, pose)
    out = pl.hover_for(rec, hf, ROBOT, pose, parts=(nb,))
    assert out["registered"] is True
    assert out["z_ref"] == pytest.approx(0.95) and alone["z_ref"] == pytest.approx(0.95)
    assert out["clearance_m"] == pytest.approx(alone["clearance_m"])
    grip_z = out["hover_grip_xyz"][2]
    assert grip_z + ROBOT.p_min_m >= 1.10 + pl.P12_FLOOR_M - 1e-9                # P12: the palm clears the raise
    assert grip_z > alone["hover_grip_xyz"][2] and out["raised_m"] > 0.0
    assert out["first_stage"] is True and out["first_stage_reasons"] == ["lower_bound"]
    assert out["lower_bound_parts"] == ["E"]
    p12 = next(t for t in out["thresholds"] if t.name == "P12_turn_clearance_m")
    assert p12.value == pytest.approx(pl.P12_FLOOR_M)
    # a seeded record: the reference is the unraised field over its 3 sigma_xy seed polygon
    seeded = PartRecord(pid="C", text="part C", move_prior="free", roles=frozenset({"grip"}),
                        xyz=np.array([0.0, 0.0, 0.95]), cov=np.eye(3) * 0.002 ** 2, sigma_h_m=0.003)
    s_out = pl.hover_for(seeded, hf, ROBOT, pose, parts=(nb,))
    assert s_out["registered"] is False and s_out["z_ref"] == pytest.approx(0.95)
    assert s_out["hover_grip_xyz"][2] + ROBOT.p_min_m >= 1.10 + pl.P12_FLOOR_M - 1e-9
    # the lower-bound part itself: its own top_upper_z is its reference
    own = pl.hover_for(nb, hf, ROBOT, pose, parts=(rec,))
    assert own["z_ref"] == pytest.approx(1.10) and own["lower_bound"] is True
    as_json(out)


def test_top_upper_z_raises_free_path_clearance():
    hf = field(half=0.3)
    lower = _lower_bound("D", rect_polygon((0.0, 0.0), (1.0, 0.0), 0.02, 0.02), hf)
    pose = ROBOT.pose(state8((-0.15, 0.0, 1.0), -math.pi / 2.0, 79.9))
    plain = pl.free_path((-0.15, 0.0, 1.0), (0.15, 0.0, 1.0), hf, (), ROBOT, pose)
    raised = pl.free_path((-0.15, 0.0, 1.0), (0.15, 0.0, 1.0), hf, (), ROBOT, pose, parts=(lower,))
    assert plain["swept_max_z"] == pytest.approx(0.95) and raised["swept_max_z"] == pytest.approx(1.10)
    assert raised["transit_z"] >= 1.10 - ROBOT.p_min_m + pl.M1_FLOOR_M - 1e-9                # the palm clears it
    assert raised["transit_z"] > plain["transit_z"]
    assert raised["lower_bound_parts"] == ["D"]
    held = held_frame(pid="F", grip_pid="F", vacated=None, r_obj=0.02, offset_h=(0.0, 0.0, 0.0))
    carry = pl.transit_path(held, (-0.15, 0.0, 1.0), (0.15, 0.0), hf, (), ROBOT, pose, parts=(lower,))
    carry0 = pl.transit_path(held, (-0.15, 0.0, 1.0), (0.15, 0.0), hf, (), ROBOT, pose)
    assert carry["carry_z"] >= 1.10 + pl.M1_FLOOR_M + held.b_low.value - 1e-9 > carry0["carry_z"]


def test_top_upper_z_raises_p12_and_p7_and_refuses_contact_on_the_part():
    hf, c, d = thin_part_scene(0.002)
    lower = _lower_bound("E", rect_polygon((0.06, 0.06), (1.0, 0.0), 0.006, 0.006), hf, measured=0.95, upper=1.15)
    start = inputs(jaw=math.pi / 2.0)
    plain = pl.plan_contact(c, hf, start, (), purpose="carry", next_held=d)
    raised = pl.plan_contact(c, hf, start, (), purpose="carry", next_held=d, parts=(lower,))
    assert isinstance(plain, pl.ContactPlan) and isinstance(raised, pl.ContactPlan)
    assert raised.hover_grip_z > plain.hover_grip_z
    assert raised.hover_grip_z + ROBOT.p_min_m >= 1.15 + pl.P12_FLOOR_M - 1e-9
    # P7: a lower-bound part inside the hand footprint (outside the slot and the columns) blocks the footprint
    hf, c, d = thin_part_scene(0.002)
    beside = _lower_bound("E", rect_polygon((0.075, 0.02), (1.0, 0.0), 0.004, 0.004), hf, measured=0.95, upper=1.10)
    assert isinstance(pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d), pl.ContactPlan)
    blocked = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, parts=(beside,))
    assert isinstance(blocked, pl.Infeasible) and blocked.reason == "footprint_blocked"
    assert pl.lower_bound_blockers(blocked) == ["E"] and pl.trace_payload(blocked)["lower_bound_blockers"] == ["E"]
    # contact planning on the lower-bound part itself: refine the height first
    result = pl.plan_contact(beside, hf, inputs(), (), purpose="carry")
    assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"
    assert pl.needs_height_refinement(result) and not result.refine_axis
    one = pl.evaluate_candidate(beside, hf, inputs(), "top_down", 0.0, (), purpose="carry")
    assert isinstance(one, pl.CandidateRejection) and one.values["refine_height"] is True
    hover = pl.hover_for(beside, hf, ROBOT, ROBOT.pose(state8((0.0, 0.0, 1.25), 0.0, 79.9)))
    assert hover["z_ref"] == pytest.approx(1.10) and hover["lower_bound"] is True
    assert hover["first_stage"] is True and hover["first_stage_reasons"] == ["lower_bound"]


# ------------------------------------------------------------------------------------- S-10 (V5 reads the scene)
def test_v5_elongated_reads_the_scene_axis_check():
    hf, a = lever_scene()
    check = {"image_axis": np.array([1.0, 0.0]), "region_axis": np.array([1.0, 0.02]), "agree_rad": 0.02,
             "accepted": True, "source": "image"}
    scene_checked = PartRecord(**{**a.__dict__, "axis_check": check})
    out = pl.check_point_answer("elongated", (256, 256), scene_checked, hf, inputs(),
                                wrist_gray_upright=np.zeros((512, 512)),
                                structure_axis=lambda g, uv: (_ for _ in ()).throw(AssertionError("recomputed")))
    assert out["ok"] and out["values"]["source"] == "scene_axis_check"


# ------------------------------------------------------------------------------------- review 2 (decision §2.4)
def _checked(record, *, source, reason=None, accepted=False, collapsed=False, coherence=None):
    """``record`` with a review-2 (D10-OE, K-4) scene axis_check: image_axis = the outline axis, region_axis = the
    agreeing edge axis, agree_rad their angle; the documented extra keys method / region_collapsed / reason."""
    check = {"image_axis": np.array([0.0, 1.0]), "region_axis": np.array([0.02, 1.0]), "agree_rad": 0.02,
             "accepted": accepted, "source": source, "method": "outline_edges", "region_collapsed": collapsed,
             "reason": reason}
    if reason == "views_disagree":                          # S-12: an earlier accepted view, 9 deg away
        check["views"] = {"axes": [np.array([math.sin(math.radians(9.0)), math.cos(math.radians(9.0))])],
                          "disagree_rad": math.radians(9.0)}
    elif reason == "kept_earlier":                          # S-12: the remembered axis is the one carried
        check["views"] = {"axes": [np.array([0.0, 1.0])], "disagree_rad": None}
    if coherence is not None:
        check["coherence"] = coherence
    return PartRecord(**{**record.__dict__, "axis_check": check})


def test_ph_axis_inconsistent_asks_to_refine_even_with_an_agreeing_harmonic():
    """P-H (§6.1 item 3): RC-7 applies only to not_elongated. A record whose axis the scene refused as
    AXIS_INCONSISTENT ("no_edge_agreement", "views_disagree" of S-12) or whose region collapsed (S-13) gives
    Infeasible(geometry_unmeasured, refine_axis=True), although its width harmonic and region PCA axis agree (the
    same record without the refusal plans on the harmonic); fixtures too."""
    hf, _c, d = thin_part_scene(0.002)
    base = _harmonic_part(hf)
    assert pl.jaw_axis(base, hf)["source"] == "width_harmonic"
    refused = [_checked(base, source=AXIS_INCONSISTENT, reason="views_disagree"),
               _checked(base, source=AXIS_INCONSISTENT, reason="no_edge_agreement"),
               _checked(base, source=AXIS_INCONSISTENT),
               _checked(base, source="not_elongated", reason="outline_gate", collapsed=True),
               _checked(_harmonic_part(hf, move="hinged"), source=AXIS_INCONSISTENT, reason="views_disagree")]
    for record, why in zip(refused, ("views_disagree", "no_edge_agreement", AXIS_INCONSISTENT, "region_collapsed",
                                     "views_disagree")):
        axis = pl.jaw_axis(record, hf)
        assert axis["source"] == "unmeasured" and axis["refine_axis"] is True and axis["axis_refusal"] == why
        assert axis["contrast"] is None                         # the harmonic is not consulted (P-H)
        assert pl.jaw_yaws(record, "top_down", inputs(), hf=hf) == []
        result = pl.plan_contact(record, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"
        assert result.refine_axis is True and pl.needs_axis_refinement(result)
        assert not pl.needs_height_refinement(result)
        assert result.rejections[0].rule == "axis:D10" and result.rejections[0].values["axis_refusal"] == why
        payload = pl.trace_payload(result)
        assert payload["refine_axis"] is True and payload["width_unmeasured"] is False
        # BPRR2-6: the record carries the scene's decision (the planner compares nothing against it)
        assert {t.name for t in result.thresholds} == {"PH_scene_axis_refusal_carried"}
        carried = result.thresholds[0]
        assert carried.value == pytest.approx(pl.AXIS_AGREE_MAX_RAD) and "no planner comparison" in carried.derivation
        as_json(result.to_dict())
    # an elongated record (an accepted D10 axis, or S-12's kept_earlier axis) is unchanged
    hf2, c2, d2 = thin_part_scene(0.002)
    kept = _checked(c2, source="mask", reason="kept_earlier", accepted=True)
    assert pl.jaw_axis(kept, hf2)["source"] == "d10"
    assert isinstance(pl.plan_contact(kept, hf2, inputs(), (), purpose="carry", next_held=d2), pl.ContactPlan)


def test_ph_not_elongated_keeps_rc7_and_the_ratified_isotropy_grid():
    """P-H with the lead's sign-off 1: a not_elongated record takes the RC-7 harmonic jaw when the harmonic and the
    region PCA axis agree; a free part keeps the 15 deg grid only when the width harmonic is flat AND the region is
    isotropic (>= 4 cells, aspect < 1.5) AND the image coherence (when measured) is below 0.5; otherwise refine_axis."""
    hf, _c, d = thin_part_scene(0.002)
    rc7 = _checked(_harmonic_part(hf), source="not_elongated", reason="outline_gate")
    axis = pl.jaw_axis(rc7, hf)
    assert axis["source"] == "width_harmonic" and axis["axis_refusal"] is None and not axis["refine_axis"]
    plan = pl.plan_contact(rc7, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.checks["axis_source"] == "width_harmonic"
    # the isotropy grid: flat widths, an isotropic region of >= 4 cells, coherence below 0.5 (or not measured)
    hf, body = body_scene()
    region = pl.region_axis(body, hf)
    assert region["cells"] >= pl.AXIS_MIN_REGION_CELLS and region["ratio"] < pl.AXIS_MIN_ASPECT
    for coherence in (None, 0.30):
        rec = _checked(body, source="not_elongated", reason="outline_gate", coherence=coherence)
        assert pl.jaw_axis(rec, hf)["source"] == "isotropic"
        assert isinstance(pl.plan_contact(rec, hf, inputs(jaw=0.0, g_open=78.5), (), purpose="carry"),
                          pl.ContactPlan)
    # ... otherwise the yaw is unmeasured: an oriented image, an elongated region, a region below 4 cells
    oriented = _checked(body, source="not_elongated", reason="outline_gate", coherence=0.64)
    assert pl.jaw_axis(oriented, hf)["refine_axis"] is True
    hf, _c, d = thin_part_scene(0.002)
    flat_long = _checked(_harmonic_part(hf, widths=widths_const(0.013, 0.001)), source="not_elongated")
    small = PartRecord(**{**flat_long.__dict__, "region_cells": _cells_at(hf, (0.0, 0.016)), "region_xy": None})
    for rec in (flat_long, small):
        out = pl.jaw_axis(rec, hf)
        assert out["source"] == "unmeasured" and out["refine_axis"] is True and out["axis_refusal"] is None


def test_pi_width_unmeasured_is_geometry_unmeasured_at_p2():
    """P-I (§6.1 item 3, K-5): a registered record whose region collapsed carries width_by_yaw == () ("width
    unmeasured"): plan_contact returns Infeasible(geometry_unmeasured) at P2 (one "width:P2" rejection flagged
    width_unmeasured), before any candidate; evaluate_candidate refuses at P2 too. The refine_axis flag reports the
    axis state (an elongated record keeps its axis: False; a refused axis: True). BPRR2-7: ``width_state`` tells
    "unmeasurable" (``()``, K-5: the scene's collapsed region) from "not_measured" (``None``: no width function at
    all), and a free non-elongated part without widths has an unmeasured yaw (never the plain grid)."""
    hf, c, d = thin_part_scene(0.002)
    assert c.is_elongated()
    cases = ((PartRecord(**{**c.__dict__, "width_by_yaw": ()}), False, "unmeasurable"),
             (_checked(_harmonic_part(hf, widths=()), source=AXIS_INCONSISTENT, collapsed=True,
                       reason="no_edge_agreement"), True, "unmeasurable"),
             (_harmonic_part(hf, widths=(), move="hinged"), False, "unmeasurable"),
             (PartRecord(**{**c.__dict__, "width_by_yaw": None}), False, "not_measured"),
             (PartRecord(**{**_harmonic_part(hf).__dict__, "width_by_yaw": None}), True, "not_measured"),
             (PartRecord(**{**_harmonic_part(hf, move="hinged").__dict__, "width_by_yaw": None}), False,
              "not_measured"))
    for record, refine, state in cases:
        assert record.width_at(0.0) is None
        result = pl.plan_contact(record, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.Infeasible) and result.reason == "geometry_unmeasured"
        assert len(result.rejections) == 1 and result.rejections[0].rule == "width:P2"
        values = result.rejections[0].values
        assert values["width_unmeasured"] is True and values["width_state"] == state
        assert ("K-5" in values["note"]) is (state == "unmeasurable")
        assert result.refine_axis is refine and pl.width_unmeasured(result)
        assert pl.trace_payload(result)["width_unmeasured"] is True
        one = pl.evaluate_candidate(record, hf, inputs(), "top_down", 0.0, (), purpose="carry", next_held=d)
        assert isinstance(one, pl.CandidateRejection) and one.rule == "width:P2"
        assert one.reason == "geometry_unmeasured" and one.values["width_unmeasured"] is True
        assert one.values["width_state"] == state and pl.width_unmeasured(one)
        as_json(result.to_dict())
    # BPRR2-7: the public yaw enumeration agrees (a free part without widths: no yaws; a fixture: the grid)
    for widths in ((), None):
        free = PartRecord(**{**_harmonic_part(hf).__dict__, "width_by_yaw": widths})
        axis = pl.jaw_axis(free, hf)
        assert axis["source"] == "unmeasured" and axis["refine_axis"] is True and axis["axis_refusal"] is None
        assert pl.jaw_yaws(free, "top_down", inputs(), hf=hf) == [] and pl.rim_candidates(free, "top_down",
                                                                                         inputs(), hf=hf) == []
        fixture = PartRecord(**{**free.__dict__, "move_prior": "hinged"})
        assert pl.jaw_axis(fixture, hf)["source"] == "grid" and pl.jaw_yaws(fixture, "top_down", inputs(), hf=hf)
    # a measured width function plans as before
    assert isinstance(pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d), pl.ContactPlan)
    assert not pl.width_unmeasured(pl.plan_contact(_harmonic_part(hf, region_angle=2.0), hf, inputs(), (),
                                                   purpose="carry", next_held=d))


def test_ph_rim_candidates_precede_the_axis_refusal():
    """BPRR2-5 (the order is fixed here): a D11 rim record keeps its rim candidates when the scene refused its axis
    (AXIS_INCONSISTENT "no_edge_agreement", or an S-13 collapsed region whose earlier width set S-12 kept): rim yaws
    enumerate the circle and read no axis. The same refusals on a record without a rim ask to refine the axis."""
    hf, b = wall_scene()
    plain = pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry")
    assert isinstance(plain, pl.ContactPlan) and plain.rim_angle_rad is not None
    for refused, why in ((_checked(b, source=AXIS_INCONSISTENT, reason="no_edge_agreement"), "no_edge_agreement"),
                         (_checked(b, source=AXIS_INCONSISTENT, reason="views_disagree"), "views_disagree"),
                         (_checked(b, source="not_elongated", reason="outline_gate", collapsed=True),
                          "region_collapsed")):
        assert pl.axis_refusal(refused) == why
        axis = pl.jaw_axis(refused, hf)
        assert axis["source"] == "rim" and not axis["refine_axis"]
        plan = pl.plan_contact(refused, hf, inputs(jaw=0.0), (), purpose="carry")
        assert isinstance(plan, pl.ContactPlan) and plan.checks["axis_source"] == "rim"
        assert plan.rim_angle_rad == pytest.approx(plain.rim_angle_rad) and plan.yaw_rad == pytest.approx(plain.yaw_rad)
        # without the D11 circle the refusal holds (P-H)
        no_rim = PartRecord(**{**refused.__dict__, "circle": None})
        out = pl.plan_contact(no_rim, hf, inputs(jaw=0.0), (), purpose="carry")
        assert isinstance(out, pl.Infeasible) and out.refine_axis is True
        assert out.rejections[0].values["axis_refusal"] == why


def test_hover_first_stage_is_judged_at_the_shifted_hover():
    """BPRR2-4 (lead sign-off 3 with the P-2 visibility shift): first_stage "lower_bound" is decided at the hover's
    final xy. A lower-bound neighbour under the shifted hand only lifts the final hover: first_stage. A neighbour under
    the unshifted hand only leaves the final (shifted) hover unlifted: not first_stage."""
    pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    hf = field(source="plane", half=0.6)
    thin = rect_polygon((0.0, 0.016), (0.0, 1.0), 0.028, 0.0065)
    c = part("C", xyz=(0.0, 0.0, 1.0345), top=1.0345, sigma_h=0.002, region=thin, cells=stamp(hf, thin, 1.0345),
             widths=widths_bar(0.013, 0.056, math.pi / 2.0), elongation=(np.array([0.0, 1.0]), 0.9))
    pose = inputs().pose
    r0 = np.asarray(pose.R, float)[:2, 0] / np.linalg.norm(np.asarray(pose.R, float)[:2, 0])
    f_px, row = 128.0, 125.0
    alone = pl.hover_for(c, hf, ROBOT, pose, f_px=f_px, band_top_row=row)
    assert alone["visibility_shift_m"] > 0.05 and alone["first_stage"] is False
    unshifted = pl.hover_for(c, hf, ROBOT, pose)
    assert unshifted["visibility_shift_m"] == 0.0

    def neighbour(centre, half):
        poly = rect_polygon(centre, (1.0, 0.0), *half)
        e = part("E", xyz=(centre[0], centre[1], 0.95), top=0.95, region=poly, cells=hf.cells_in_polygon(poly),
                 widths=widths_const(0.04, 0.001), roles=("carried",))
        return PartRecord(**{**e.__dict__, "top_bound": "lower", "top_upper_z": 1.25, "partial": True})
    # under the shifted hand only: the unshifted hover does not meet it, the shifted one is lifted by it
    far = neighbour(np.asarray(c.xyz[:2]) - r0 * 0.32, (0.10, 0.05))
    assert pl.hover_for(c, hf, ROBOT, pose, parts=(far,))["first_stage"] is False
    lifted = pl.hover_for(c, hf, ROBOT, pose, parts=(far,), f_px=f_px, band_top_row=row)
    assert lifted["hover_grip_xyz"][2] > alone["hover_grip_xyz"][2] + 0.05
    assert lifted["first_stage"] is True and lifted["first_stage_reasons"] == ["lower_bound"]
    # under the unshifted hand only: it lifts the unshifted hover, the final shifted hover is not lifted (the
    # transient lift moves the hover further along -R0; its height is the part's again)
    near = neighbour(np.asarray(c.xyz[:2]) - r0 * 0.02, (0.01, 0.01))
    assert pl.hover_for(c, hf, ROBOT, pose, parts=(near,))["first_stage_reasons"] == ["lower_bound"]
    moved = pl.hover_for(c, hf, ROBOT, pose, parts=(near,), f_px=f_px, band_top_row=row)
    assert moved["visibility_shift_m"] > 0.05
    assert moved["hover_grip_xyz"][2] == pytest.approx(alone["hover_grip_xyz"][2])
    assert moved["first_stage"] is False and moved["first_stage_reasons"] == []


# ------------------------------------------------------------------------------------- P7 stays a vision check
_BAND_FILL_XY = (0.070, 0.020)            # in the hand footprint, outside the slot and the columns (thin part)


def test_p7_unlisted_same_height_fills_block_and_s6_listed_fills_are_excluded():
    """BPRR2-1 (review-1 P-A "P7 stays a vision check", S-6's deliberate negative): which D7b fills are the part's own
    halo is the scene's S-6 decision. A fill copying the part's height that S-6 did not list (parent at the part's
    top) blocks P7; the same cell listed by S-6 (part_fill_cells, RC-1d) is excluded and the plan stays clean.
    Gap round: the refusal holds at g* (IF-HS may raise the stop over the stored fill; a measured support keeps the
    listed plan clean under F6-B)."""
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), 1.0345 + 3 * 0.004, sigma=0.004,
          source="obstacle")                                       # parent 1.0345 = the part's top
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert p7_refused_at_target(refused)
    listed = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d,
                             part_fill_cells=_cells_at(hf, _BAND_FILL_XY))
    assert isinstance(listed, pl.ContactPlan) and listed.checks["admission"] == "clean"
    assert "p7_band_fills" not in listed.checks
    assert not any(t.name.startswith("P7_band") for t in listed.thresholds)


@pytest.mark.parametrize("parent_z, parent_sigma", [(1.0405, 0.004), (1.0430, 0.005), (1.0345, 0.004)])
def test_p7_fills_of_a_separate_object_at_the_parts_height_block(parent_z, parent_sigma):
    """BPRR2-1 (c), the T1 negative S-6 keeps: a separate raised object (not connected to the part) whose measured
    cells lie just outside the hand footprint and whose D7b fills (the occluded continuation, parent + 3 sigma) lie
    inside it, at heights within the part's top band. P7 refuses (footprint_blocked): the fills are not the part's
    halo, and a palm stop on a real surface there would read as an expected contact."""
    hf, c, d = thin_part_scene(0.002)
    base = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(base, pl.ContactPlan)
    grip = np.asarray(base.grip_point_xyz[:2])
    x_edge = float(ROBOT.hand_footprint_xy(grip, pl._unit_xy(base.yaw_rad))[:, 0].max())
    stamp(hf, rect_polygon((x_edge + 0.012, 0.03), (1.0, 0.0), 0.007, 0.01), parent_z, sigma=parent_sigma)
    fill_sigma = min(parent_sigma, 0.005)
    stamp(hf, rect_polygon((x_edge - 0.006, 0.03), (1.0, 0.0), 0.004, 0.01), parent_z + 3.0 * fill_sigma,
          sigma=fill_sigma, source="obstacle")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    # gap round: refused at g* on the stored fills; IF-HS may clear them, and only the IF-PV fallback (fills read at
    # parent + 1 sigma, marginal: the P14 ladder verifies the surface) may admit the candidate otherwise
    pv = isinstance(refused, pl.ContactPlan) and refused.checks.get("p7_reading") == "parent_sigma"
    assert p7_refused_at_target(refused) or (pv and refused.marginal and refused.admission == "marginal")
    for rejection in getattr(refused, "rejections", ()):
        assert rejection.reason in ("footprint_blocked", "yaw_budget"), rejection


def test_p7_measured_cells_and_keepouts_block():
    """P7 negatives kept from the review-2 build: a fill whose parent stands above the part's band, a measured cell at
    the same height and a keep-out over a band-height fill all block the footprint."""
    for z, sigma, source in ((1.0465, 0.0005, "obstacle"), (1.0465, 0.004, "kappa")):
        hf, c, d = thin_part_scene(0.002)
        stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), z, sigma=sigma, source=source)
        refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
        assert p7_refused_at_target(refused), source
    hf, c, d = thin_part_scene(0.002)
    stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), 1.0345 + 3 * 0.004, sigma=0.004,
          source="obstacle")
    keepout = KeepOut("E", "clear_of", rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.003, 0.003), SUPPORT, 1.0465, ())
    blocked = pl.plan_contact(c, hf, inputs(), (keepout,), purpose="carry", next_held=d)
    assert p7_refused_at_target(blocked)
    # a fill below the P7 limit needs nothing: the plan stays clean (a measured support, F6-B)
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), 1.0345 + 3 * 0.003, sigma=0.003,
          source="obstacle")
    low = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(low, pl.ContactPlan) and low.checks["admission"] == "clean"


# ------------------------------------------------------------------- live-first blockers: BP-1 pointing frame
def _framing_part(hf, *, cov_sigma=0.002):
    poly = rect_polygon((0, 0), (1, 0), 0.006, 0.02)
    return part("C", xyz=(0.0, 0.0, 0.95), top=0.95, sigma_h=0.003, region=poly, cells=stamp(hf, poly, 0.95),
                widths=widths_const(0.012, 0.001), cov_sigma=cov_sigma)


def test_bp1_restated_constants_equal_their_sources():
    """FRAMING_CROP_HALF_PX restates half the client's V6 fine crop; FRAMING_COARSE_SIGMA_PX the relation layer's
    coarse-first threshold 64 / sqrt(2 ln 2) (rulings item 9)."""
    from gpt_field import agent_speedup_v2_client as client
    assert pl.FRAMING_CROP_HALF_PX * 2 == client.FINE_CROP_PX
    assert pl.FRAMING_COARSE_SIGMA_PX == pytest.approx(64.0 / math.sqrt(2.0 * math.log(2.0)))
    try:
        from gpt_field import agent_speedup_v2_relations as relations
    except Exception:                                                  # noqa: BLE001 (BR's module, edited in parallel)
        relations = None
    if relations is not None and hasattr(relations, "COARSE_FIRST_SIGMA_PX"):
        assert pl.FRAMING_COARSE_SIGMA_PX == pytest.approx(relations.COARSE_FIRST_SIGMA_PX)


def test_bp1_pointing_frame_fine_crop_and_coarse_first_rules():
    """Grip-part lens BP-1: fine pointing needs the fine crop's bottom (row + 64 px) FT_PROBE_RADIUS_PX above the
    band top; coarse-first (sigma_uv > ~54.4 px, "auto") needs row + sigma_uv there; both keep P-2's own margin.
    Negative: a row that P-2 alone accepts (33-38 px above the band) is not framed for fine pointing."""
    geometry = pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    hf = field()
    rec = _framing_part(hf)
    pose = ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9))
    hover = np.array([0.0, 0.0, 1.10])
    f_px, radius = 333.6, float(geometry.FT_PROBE_RADIUS_PX)
    probe = pl.pointing_frame(rec, hover, pose, f_px=f_px, band_top_row=1000.0)
    assert probe["ok"] and probe["mode"] == "fine" and probe["sigma_uv_px"] < pl.FRAMING_COARSE_SIGMA_PX
    row, kappa = probe["row"], probe["kappa_px_per_m"]
    assert probe["crop_bottom_row"] == pytest.approx(row + 64.0)
    assert probe["margin_px"] == pytest.approx(max(64.0 + radius, kappa * pl.VIS_PROBE_LEG_M + radius))
    p2_only = row + kappa * pl.VIS_PROBE_LEG_M + radius + 1.0       # P-2 passes here, the fine crop does not
    assert pl._visibility_shift_m(rec.xyz, hover, pose, f_px, p2_only, radius) == 0.0
    tight = pl.pointing_frame(rec, hover, pose, f_px=f_px, band_top_row=p2_only)
    assert not tight["ok"] and tight["excess_px"] > 0.0
    edge = row + 64.0 + radius
    assert pl.pointing_frame(rec, hover, pose, f_px=f_px, band_top_row=edge)["ok"]
    assert not pl.pointing_frame(rec, hover, pose, f_px=f_px, band_top_row=edge - 1.0)["ok"]
    # coarse-first: a 6 cm lateral sigma gives sigma_uv > 54.4 px; the rule is row + sigma_uv + radius
    wide = _framing_part(field(), cov_sigma=0.06)
    coarse = pl.pointing_frame(wide, hover, pose, f_px=f_px, band_top_row=1000.0)
    assert coarse["mode"] == "coarse" and coarse["sigma_uv_px"] > pl.FRAMING_COARSE_SIGMA_PX
    assert coarse["crop_bottom_row"] == pytest.approx(coarse["row"] + coarse["sigma_uv_px"])
    fine = pl.pointing_frame(wide, hover, pose, f_px=f_px, band_top_row=1000.0, pointing="fine")
    assert fine["mode"] == "fine"
    with pytest.raises(ValueError):
        pl.pointing_frame(rec, hover, pose, f_px=f_px, band_top_row=400.0, pointing="nearest")


def test_bp1_hover_for_moves_the_hover_until_the_fine_crop_clears_the_band():
    """BP-1 in hover_for (``pointing``): the hover moves along -R0 until the fine crop's bottom lies
    FT_PROBE_RADIUS_PX above the band (the live crops reached 2-24 rows into it), P12 re-checked; without
    ``pointing`` hover_for keeps P-2 only (no behaviour change for surveys)."""
    geometry = pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    hf = field()
    rec = _framing_part(hf)
    pose = ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9))
    f_px = 333.6
    plain = pl.hover_for(rec, hf, ROBOT, pose)
    target = np.asarray(plain["hover_grip_xyz"])
    at = pl.pointing_frame(rec, target, pose, f_px=f_px, band_top_row=1000.0)
    band = at["row"] + at["kappa_px_per_m"] * pl.VIS_PROBE_LEG_M + geometry.FT_PROBE_RADIUS_PX + 2.0
    p2 = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band)
    assert p2["visibility_shift_m"] == 0.0 and p2["framing"] is None
    framed = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band, pointing="auto")
    frame = framed["framing"]
    assert frame["ok"] and frame["mode"] == "fine" and 0.0 < frame["shift_m"] <= pl.M2_SEGMENT_M
    assert frame["raised_m"] == 0.0 and framed["visibility_shift_m"] == pytest.approx(frame["shift_m"])
    moved = np.asarray(framed["hover_grip_xyz"])
    r0 = pose.R[:, 0][:2] / np.linalg.norm(pose.R[:, 0][:2])
    assert float((moved[:2] - target[:2]) @ r0) < 0.0                       # along -R0
    again = pl.pointing_frame(rec, moved, pose, f_px=f_px, band_top_row=band)
    assert again["ok"] and again["crop_bottom_row"] <= band - geometry.FT_PROBE_RADIUS_PX + 0.5
    assert any(t.name == "framing_margin_px" for t in framed["thresholds"])
    with pytest.raises(ValueError):
        pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band, pointing="nearest")


def test_bp1_shift_is_capped_at_one_m2_segment_then_the_hover_rises(monkeypatch):
    """BP-1: the -R0 shift stops at M2_SEGMENT_M; if the frame still fails the hover rises in cell steps (<=
    P11_CAP_M) while that helps (a rise moves the part's row toward the principal row, so it helps while the row lies
    below it) and the rise is no lower-bound lift (first_stage_reasons keeps no "lower_bound"). With a 5 mm segment
    (monkeypatched) the capped shift leaves the part below the principal row and the rise frames it; with the real
    7.5 cm segment a band that needs more shift is never framed (ok False, no rise, the shifted hover kept)."""
    pytest.importorskip("gpt_field.agent_speedup_v2_geometry")
    hf = field(half=0.6)
    rec = _framing_part(hf)
    pose = ROBOT.pose(state8((0.0, 0.0, 1.19), 0.0, 79.9))
    f_px = 333.6
    plain = pl.hover_for(rec, hf, ROBOT, pose)
    target = np.asarray(plain["hover_grip_xyz"])
    at = pl.pointing_frame(rec, target, pose, f_px=f_px, band_top_row=1000.0)
    kappa = at["kappa_px_per_m"]
    # the real segment: needs 1 cm more than M2_SEGMENT_M of shift
    band = at["row"] + at["margin_px"] - kappa * (pl.M2_SEGMENT_M + 0.01)
    out = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band, pointing="fine")
    frame = out["framing"]
    assert frame["shift_m"] == pytest.approx(pl.M2_SEGMENT_M) and frame["raised_m"] == 0.0 and not frame["ok"]
    assert out["hover_grip_xyz"][2] == pytest.approx(plain["hover_grip_xyz"][2])
    # a 5 mm segment: the part stays below the principal row after the shift; the rise frames it
    monkeypatch.setattr(pl, "M2_SEGMENT_M", 0.005)
    band = at["row"] + at["margin_px"] - kappa * 0.005 - 6.0
    risen = pl.hover_for(rec, hf, ROBOT, pose, f_px=f_px, band_top_row=band, pointing="fine")
    frame = risen["framing"]
    assert frame["shift_m"] == pytest.approx(0.005) and 0.0 < frame["raised_m"] <= pl.P11_CAP_M and frame["ok"]
    assert risen["hover_grip_xyz"][2] == pytest.approx(plain["hover_grip_xyz"][2] + frame["raised_m"])
    assert "lower_bound" not in risen["first_stage_reasons"]


# ------------------------------------------------------------------- live-first blockers: BP-2 blocking cells
def _column_cell(hf, column):
    centre = np.asarray(column.polygon_xy, float).mean(axis=0)
    return hf.index(centre)


def test_bp2_a_single_false_column_cell_blocks_and_asks_to_refine_cells_when_unconfirmed():
    """Grip-part lens BP-2: one false cell under a finger column (+110 mm, the rs2 false cell) refuses both depth
    rules (P3 column for touch_stop, P6 landing for height_stop); every rejection lists the cell (ij, xyz, z, sigma,
    source, unconfirmed); with the scene's ``hf.unconfirmed`` marking it the Infeasible asks ``refine_cells`` and
    ``blocking_cells`` returns it for BR-5. Negatives: unmarked, or with a second (confirmed) blocking cell, no
    refine_cells."""
    hf, c, d = thin_part_scene(0.002)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan)
    ij = _column_cell(hf, plan.columns[0])
    hf.z[ij], hf.source[ij] = 1.0345 + 0.110, hf_code("kappa")                 # a measured (parallax) false cell
    hf.unconfirmed = np.zeros(hf.shape, bool)
    plain = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plain, pl.Infeasible) and not plain.refine_cells and not pl.needs_cell_refinement(plain)
    for r in plain.rejections:
        cells = r.values["blocking"]["cells"]
        assert [list(ij)] == [cell["ij"] for cell in cells] and cells[0]["unconfirmed"] is False
        assert cells[0]["source"] == "kappa" and cells[0]["z"] == pytest.approx(1.1445)
        assert {f["check"] for f in r.values["by_rule"].values()} <= {"P3", "P6"}
    hf.unconfirmed[ij] = True
    marked = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(marked, pl.Infeasible) and marked.refine_cells and pl.needs_cell_refinement(marked)
    assert [cell["ij"] for cell in pl.blocking_cells(marked)] == [list(ij)]
    # the relation layer's keys (BR-5): the Infeasible's ``blocking_cells`` and each rejection's values
    assert marked.blocking_cells == pl.blocking_cells(marked)
    assert all(r.values["refine_cells"] and [c["ij"] for c in r.values["blocking_cells"]] == [list(ij)]
               for r in marked.rejections)
    assert all(r.values["blocking_cells"] == [] for r in plain.rejections) and plain.blocking_cells == []
    assert pl.trace_payload(marked)["refine_cells"] is True
    json.dumps(marked.to_dict(), allow_nan=False)
    # a second, confirmed blocking cell in the other column: some rule still fails on a confirmed cell only
    other = _column_cell(hf, plan.columns[1])
    hf.z[other], hf.source[other] = 1.0345 + 0.050, hf_code("kappa")
    mixed = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(mixed, pl.Infeasible) and not mixed.refine_cells
    assert any(not cell["unconfirmed"] for r in mixed.rejections for cell in r.values["blocking"]["cells"])


def test_bp2_footprint_blocking_lists_cells_and_an_overlay_is_never_unconfirmed():
    """BP-2 on P7: a measured cell over the footprint is listed (unconfirmed when the scene marks it, then
    refine_cells); a keep-out over the same cell is listed as "keepout" and never unconfirmed (no refine_cells)."""
    hf, c, d = thin_part_scene(0.002)
    cells = stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), 1.0465, sigma=0.004, source="kappa")
    hf.unconfirmed = np.zeros(hf.shape, bool)
    hf.unconfirmed[cells[:, 0], cells[:, 1]] = True
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "footprint_blocked" and refused.refine_cells
    listed = {tuple(cell["ij"]) for cell in pl.blocking_cells(refused)}
    assert listed == {tuple(c_) for c_ in cells.tolist()}
    keepout = KeepOut("E", "clear_of", rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.003, 0.003), SUPPORT, 1.0565, ())
    blocked = pl.plan_contact(c, hf, inputs(), (keepout,), purpose="carry", next_held=d)
    assert isinstance(blocked, pl.Infeasible) and not blocked.refine_cells
    sources = {cell["source"] for r in blocked.rejections for cell in r.values["blocking"]["cells"]}
    assert "keepout" in sources
    assert all(not cell["unconfirmed"] for r in blocked.rejections for cell in r.values["blocking"]["cells"]
               if cell["source"] == "keepout")


def test_bp2_every_infeasible_carries_a_snapshot_of_its_inputs():
    """BP-2: every Infeasible carries the inputs of its plan_contact call (keys as the gate's plan_inputs pickles;
    the HeightField deep-copied, so later scene writes do not alter it); re-planning on the snapshot reproduces the
    refusal; the snapshot is not compared nor serialised; a ContactPlan carries none."""
    hf, c, d = thin_part_scene(0.002)
    stamp(hf, rect_polygon(_BAND_FILL_XY, (1.0, 0.0), 0.002, 0.002), 1.0465, sigma=0.004, source="kappa")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, parts=(d,))
    snap = refused.inputs
    assert {"record", "hf", "inputs", "keepouts", "purpose", "next_held", "exclude", "depth_rules",
            "part_fill_cells", "parts"} <= set(snap)
    assert snap["record"] is c and snap["hf"] is not hf and snap["purpose"] == "carry" and snap["parts"] == (d,)
    hf.z[:] = SUPPORT                                                     # a later write to the live field
    assert float(np.nanmax(snap["hf"].z)) > 1.04
    again = pl.plan_contact(snap["record"], snap["hf"], snap["inputs"], snap["keepouts"], purpose=snap["purpose"],
                            next_held=snap["next_held"], exclude=snap["exclude"], depth_rules=snap["depth_rules"],
                            part_fill_cells=snap["part_fill_cells"], parts=snap["parts"])
    assert isinstance(again, pl.Infeasible) and again.reason == refused.reason and again == refused
    assert "inputs" not in refused.to_dict()
    hf2, c2, d2 = thin_part_scene(0.002)
    ok = pl.plan_contact(c2, hf2, inputs(), (), purpose="carry", next_held=d2)
    assert isinstance(ok, pl.ContactPlan) and not hasattr(ok, "inputs")


def test_bp2_closing_sweep_blocking_cells():
    """BP-2 on P5: the raised ring inside the closing sweep (test_closing_sweep_blocks_both_rules) is listed as the
    P5 rejection's blocking cells (above the sweep limit); refine_cells only when the scene marks every one of them
    unconfirmed."""
    hf, a = lever_scene()
    inner = {tuple(c) for c in hf.cells_in_polygon(disk((0.0, 0.0), 0.022))}
    region = {tuple(c) for c in a.region_cells}
    cells = np.array([c for c in hf.cells_in_polygon(disk((0.0, 0.0), 0.028))
                      if tuple(c) not in inner and tuple(c) not in region])
    hf.z[cells[:, 0], cells[:, 1]] = 0.930
    result = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    p5 = [r for r in result.rejections if r.values.get("by_rule", {}).get("touch_stop", {}).get("check") == "P5"]
    assert p5 and not result.refine_cells
    ring = {tuple(c) for c in cells.tolist()}
    for r in p5:
        block = r.values["by_rule"]["touch_stop"]["values"]["blocking"]
        assert block["count"] >= 1 and {tuple(c["ij"]) for c in block["cells"]} <= ring
        assert not any(c["unconfirmed"] for c in block["cells"])
        assert not r.values["refine_cells"]
    hf.unconfirmed = np.zeros(hf.shape, bool)
    hf.unconfirmed[cells[:, 0], cells[:, 1]] = True
    marked = pl.plan_contact(a, hf, inputs(), (), purpose="turn")
    flagged = [r for r in marked.rejections if r.values.get("by_rule", {}).get("touch_stop", {}).get("check") == "P5"]
    assert flagged and all(r.values["by_rule"]["touch_stop"]["values"]["blocking"]["all_unconfirmed"]
                           for r in flagged)
    assert marked.refine_cells and {tuple(c["ij"]) for c in pl.blocking_cells(marked)} <= ring


def test_bp3_the_planner_never_snaps_the_record_point_to_a_nearby_ridge():
    """BP-3: the planner grasps at the record's own point; a taller ridge 3 cm away (about 80-100 px at the hover,
    the live mispointings' distance to the right part) never pulls the grip point (the S-2 top-band seed lift is the
    scene's only snap; S-15 refuses a seed beyond q_now + 2 sigma_track)."""
    hf, c, d = thin_part_scene(0.002)
    ridge = rect_polygon((-0.12, 0.016), (0.0, 1.0), 0.02, 0.004)
    stamp(hf, ridge, 1.06)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan)
    assert np.allclose(plan.grip_point_xyz[:2], c.xyz[:2], atol=1e-9)


# ------------------------------------------------------------- grasp-refusal decision J (P-J, RC-1e, GC-1, PC-1, CJ-1)
def _offset_lever(offset=-0.008):
    """The lever scene with the pointed (grip) point ``offset`` across the ridge (purpose "turn" keeps it): the grip
    sits off the measured section centre, as the live grips (4-13.6 mm off the centre line)."""
    hf, a = lever_scene()
    a = dataclasses.replace(a, xyz=np.array([0.010, 0.004 + offset, 0.955]))
    return hf, a


def _halo_row(hf, y, z=0.954, sigma=0.002, x=(-0.010, 0.010)):
    """Measured cells one row beyond the ridge edge at the ridge height (the displaced own-edge samples)."""
    poly = rect_polygon(((x[0] + x[1]) / 2.0, y), (1.0, 0.0), (x[1] - x[0]) / 2.0, 0.002)
    return stamp(hf, poly, z, sigma=sigma)


def test_decision_j_constants_and_labels():
    """Decision §4: RC-1e's resolution is an alias of the field cell (no new number); GC-1's touch tolerance is a
    labelled DD constant; P-K and BP-2b are not adopted (no P7 ring exclusion, no inferred-cell refinement)."""
    from gpt_field.agent_speedup_v2_types import HF_CELL_M
    assert pl.P5_SIGMA_RESOLUTION_M == HF_CELL_M
    assert pl.GC_CONTACT_TOL_MM == 1.0 and pl.LABELS["GC_CONTACT_TOL_MM"][0] == "DD"
    assert "agent_speedup_v2_planner.GC_CONTACT_TOL_MM" in th.CONSTANT_LABELS
    assert pl.PLANNER_VERSION == "agent-speedup-v2-planner-7"
    hf, c, d = thin_part_scene(0.001)
    ctx = pl._Context(c, hf, inputs(), (), "carry", d, part_ring_cells=np.array([[1, 2], [3, 4]]))
    assert len(ctx.ring_cells) == 2 and len(ctx.own_cells) == len(c.region_cells) + 2
    assert not hasattr(ctx, "ring_p7_cells")


def test_pj_section_edge_close_verifies_a_cell_beyond_the_model_edge_of_an_off_centre_grip():
    """Remedy 1 P-J: grip 8 mm off the ridge centre; a ridge-height cell 4.5 mm beyond the ridge's far edge is 24 mm
    from the grip, beyond w/2 + e_lat + cell/2 (22.65 mm) but inside the measured section edge (region outer
    boundary 21.5 mm) + e_lat + cell/2 = 32 mm: close-verified, the candidate is admitted marginal; the same cell
    with the grip on the centre line was already inside the band (unchanged)."""
    hf, a = _offset_lever()
    _halo_row(hf, 0.020)
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    assert isinstance(plan, pl.ContactPlan) and plan.admission == "marginal"
    assert plan.checks["close_verified"] >= 1
    forced = pl.evaluate_candidate(a, hf, inputs(), "top_down", -math.pi / 2, (), purpose="turn",
                                   depth_rules=("height_stop",))
    assert isinstance(forced, pl.ContactPlan)


def test_pj_edge_is_capped_at_one_width_and_needs_own_cells_in_the_finger_strip():
    """Remedy 1 P-J: an overgrown region (own cells 40 mm past the grip) moves the section edge no farther than w; own
    cells outside the finger's sweep strip across the jaw do not move it; without own cells the edge is w/2."""
    hf, a = _offset_lever()
    ctx = pl._Context(a, hf, inputs(), (), "turn", None)
    w = a.width_at(-math.pi / 2).value
    edge, extent = pl._section_edge_m(ctx, a.xyz[:2], -math.pi / 2, 0.008, w / 2.0, "finger2", -1.0)
    assert extent == pytest.approx(0.015 + 0.0025 + 0.004, abs=1e-9) and edge == pytest.approx(extent)
    grown = np.vstack([a.region_cells, hf.cells_in_polygon(rect_polygon((0.0, 0.030), (1.0, 0.0), 0.004, 0.006))])
    wide = pl._Context(dataclasses.replace(a, region_cells=grown), hf, inputs(), (), "turn", None)
    edge2, extent2 = pl._section_edge_m(wide, a.xyz[:2], -math.pi / 2, 0.008, w / 2.0, "finger2", -1.0)
    assert extent2 > w and edge2 == pytest.approx(w)
    far = hf.cells_in_polygon(rect_polygon((0.045, 0.030), (1.0, 0.0), 0.002, 0.006))       # beyond the strip in x
    side = pl._Context(a, hf, inputs(), (), "turn", None, part_ring_cells=far)
    assert pl._section_edge_m(side, a.xyz[:2], -math.pi / 2, 0.008, w / 2.0, "finger2", -1.0)[1] == \
        pytest.approx(extent)
    bare = pl._Context(dataclasses.replace(a, region_cells=np.zeros((0, 2), int), region_xy=None), hf, inputs(), (),
                       "turn", None)
    assert pl._section_edge_m(bare, a.xyz[:2], -math.pi / 2, 0.008, w / 2.0, "finger2", -1.0) == (w / 2.0, None)


def test_pj_ring_cells_extend_the_section_edge():
    """Remedy 1 P-J with the S-17 ring: a region missing the ridge's far-edge row (under-segmented by one cell) puts
    a ridge-height cell 29 mm from the grip beyond its section band (16.5 + 8 + 2.5 = 27 mm): refused (P5); listing
    the far-edge row as ring cells moves the section edge one cell out (21.5 + 8 + 2.5 = 32 mm): admitted."""
    hf, a = _offset_lever()
    edge_row = [tuple(c) for c in a.region_cells if abs(hf.cell_xy(c)[1] - 0.015) < 1e-9]
    short = np.array([c for c in a.region_cells if tuple(c) not in set(edge_row)])
    b = dataclasses.replace(a, region_cells=short)
    _halo_row(hf, 0.025)
    refused = pl.plan_contact(b, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",))
    assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked"
    assert refused.inputs["part_ring_cells"] is None
    admitted = pl.plan_contact(b, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",),
                               part_ring_cells=np.array(edge_row))
    assert isinstance(admitted, pl.ContactPlan) and admitted.admission == "marginal"
    assert admitted.checks["part_ring_cells"] == len(edge_row)


def test_pj_a_real_obstacle_beside_the_part_beyond_the_section_band_still_refuses():
    """Negative (P-J / ring): the widest band P-J can give is the edge capped at w (+ e_lat + cell/2). Thin part (w 13
    mm): a kappa obstacle 25 mm from the grip is beyond 13 + 8 + 2.5 = 23.5 mm, so P5 refuses it with the S-17 ring
    listed and even when the scene had (wrongly) listed it as own cells. Off-centre lever: an obstacle 34-38 mm out
    (beyond 24.3 + 8 + 2.5 = 34.8 mm) refuses with the ring and as own cells; the same obstacle inside the band (28-32
    mm) is close-verified (marginal), as RC-1b always did."""
    hf, c, d = thin_part_scene(0.002)
    ring = stamp(hf, rect_polygon((0.010, 0.016), (0.0, 1.0), 0.028, 0.002), 1.034, sigma=0.002)   # one cell out
    obstacle = stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
    for listed in (ring, obstacle):
        refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, part_ring_cells=listed)
        assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked", len(listed)
        assert {r.values["by_rule"]["height_stop"]["check"] for r in refused.rejections} == {"P5"}

    def lever(y):
        hf, a = _offset_lever()
        ring = _halo_row(hf, 0.020)
        obstacle = stamp(hf, rect_polygon((0.0, y), (1.0, 0.0), 0.010, 0.002), 0.950, sigma=0.002)
        return hf, a, ring, obstacle
    hf, a, ring, obstacle = lever(0.032)
    for listed in (ring, np.vstack([ring, obstacle])):
        refused = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",),
                                  part_ring_cells=listed)
        assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked"
    hf, a, ring, obstacle = lever(0.026)
    inside = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",), part_ring_cells=ring)
    assert isinstance(inside, pl.ContactPlan) and inside.admission == "marginal"
    assert inside.checks["close_verified"] >= 1


def test_rc1e_an_imprecise_measured_sweep_cell_is_unverified():
    """Remedy 1 RC-1e: a kappa cell 25 mm from the grip in the closing sweep (beyond the near-edge band) refuses at
    sigma 2 mm (RC-1b test) but is close-verified (marginal) when its sigma exceeds the field resolution
    (P5_SIGMA_RESOLUTION_M = HF_CELL_M); P7 still sees such a cell with its sigma."""
    assert pl.P5_SIGMA_RESOLUTION_M == pytest.approx(0.005)
    for sigma, admitted in ((0.002, False), (0.005, False), (0.0051, True), (0.0176, True)):
        hf, c, d = thin_part_scene(0.002)
        stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=sigma, source="kappa")
        result = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
        assert isinstance(result, pl.ContactPlan) is admitted, sigma
        if admitted:
            assert result.marginal and result.checks["close_verified"] >= 1
    hf, c, d = thin_part_scene(0.001)
    stamp(hf, rect_polygon((0.070, 0.020), (1.0, 0.0), 0.002, 0.002), 1.050, sigma=0.0176, source="kappa")
    p7 = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(p7, pl.Infeasible) and p7.reason == "footprint_blocked"


def test_ring_cells_never_leave_p7_pk_not_adopted():
    """Negative (P-K not adopted): a measured cell in the hand footprint at the part's top (sigma 7 mm, so its own
    P7 margin blocks) refuses P7 whether or not it is listed as a ring cell; only part_fill_cells leave P7."""
    hf, c, d = thin_part_scene(0.001)
    cells = stamp(hf, rect_polygon((0.070, 0.020), (1.0, 0.0), 0.002, 0.002), 1.0340, sigma=0.007)
    blocked = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert p7_refused_at_target(blocked)
    listed = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d, part_ring_cells=cells)
    assert p7_refused_at_target(listed)
    # gap round: the ring listing changes nothing in P7 (the same refusal, or the same IF-HS raise over the cell)
    assert type(listed) is type(blocked)
    if isinstance(listed, pl.Infeasible):
        assert np.array_equal(listed.inputs["part_ring_cells"], cells)
    else:
        assert listed.checks["hs_raise_mm"] == pytest.approx(blocked.checks["hs_raise_mm"])


def test_inferred_blockers_are_diagnostics_only_bp2b_not_adopted():
    """Negative (BP-2b not adopted): a P7 refusal resting only on D7b fills lists them ``inferred`` (and the summary
    ``all_unverified``) as diagnostics, but sets no refine_cells and offers no BR-5 targets: refinement stays BS-2's
    rule (all-unconfirmed cells only); with a measured blocker beside them the summary is not all_unverified."""
    hf, c, d = thin_part_scene(0.001)
    # gap round: high enough that neither the IF-HS raise (P3 guard) nor the IF-PV parent reading admits it
    fill = stamp(hf, rect_polygon((0.070, 0.020), (1.0, 0.0), 0.002, 0.002), 1.07, sigma=0.002, source="obstacle")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "footprint_blocked"
    assert not refused.refine_cells and not pl.needs_cell_refinement(refused) and pl.blocking_cells(refused) == []
    blocking = [r.values["blocking"] for r in refused.rejections if r.values.get("blocking")]
    assert blocking and all(b["all_unverified"] and b["inferred"] >= 1 for b in blocking)
    rows = [row for b in blocking for row in b["cells"]]
    assert {tuple(row["ij"]) for row in rows} <= {tuple(f) for f in fill.tolist()}
    assert all(row["inferred"] and row["source"] == "obstacle" and not row["unconfirmed"] for row in rows)
    stamp(hf, rect_polygon((0.070, -0.020), (1.0, 0.0), 0.002, 0.002), 1.07, sigma=0.002, source="kappa")
    mixed = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(mixed, pl.Infeasible) and not mixed.refine_cells
    mixed_blocks = [r.values["blocking"] for r in mixed.rejections if r.values.get("blocking")]
    assert mixed_blocks and not any(b["all_unverified"] for b in mixed_blocks)


def test_gc1_close_gate_bounds_each_finger_by_its_p5_band_and_the_own_cells():
    """GC-1 gate: per finger, F = max(the P5 near band of that side (P-J edge + e_lat + cell/2), the own-cell extent
    in that finger's strip + the G13 band) and R = F + GC_CONTACT_TOL_MM; the gate records the evaluated grip point,
    the jaw, span_hi = w + G13 band, the tolerance and the R9/2 pad-face offset, and is JSON-safe. Off-centre lever
    with the S-17 ring: finger 2's band is the capped edge 24.3 + 8 + 2.5 = 34.8 mm (own extent 26.5 mm), finger 1's
    the model edge 12.15 + 8 + 2.5 = 22.65 mm (own extent 3.5 mm)."""
    from gpt_field import agent_speedup_v2_robot as robot_facts
    hf, a = _offset_lever()
    ring = _halo_row(hf, 0.020)
    plan = pl.plan_contact(a, hf, inputs(), (), purpose="turn", depth_rules=("height_stop",), part_ring_cells=ring)
    assert isinstance(plan, pl.ContactPlan)
    gate = plan.checks["gate"]
    json.dumps(gate, allow_nan=False)
    f1, f2 = gate["fingers"]["finger1"], gate["fingers"]["finger2"]
    assert f1["near_edge_m"] == pytest.approx(0.01215 + 0.008 + 0.0025, abs=1e-9)
    assert f1["own_extent_m"] == pytest.approx(0.0035, abs=1e-9) and f1["F_m"] == pytest.approx(f1["near_edge_m"])
    assert f2["near_edge_m"] == pytest.approx(0.0243 + 0.008 + 0.0025, abs=1e-9)
    assert f2["own_extent_m"] == pytest.approx(0.0265, abs=1e-9) and f2["F_m"] == pytest.approx(f2["near_edge_m"])
    band = max(pl.G13_FLOOR_MM, pl.G13_SIGMA_K * plan.width["sigma_w_mm"]) / 1000.0
    assert gate["band_g13_m"] == pytest.approx(band)
    for f in (f1, f2):
        assert f["F_m"] >= f["near_edge_m"] - 1e-12 and f["F_m"] >= f["own_extent_m"] + band - 1e-12
        assert f["R_m"] == pytest.approx(f["F_m"] + pl.GC_CONTACT_TOL_MM / 1000.0, abs=1e-12)
    assert gate["grip_xy"] == pytest.approx(list(plan.grip_point_xyz[:2]))
    assert gate["jaw_xy"] == pytest.approx(list(plan.jaw_world[:2]))
    assert gate["span_hi_mm"] == pytest.approx(plan.width["w_mm"] + band * 1000.0)
    assert gate["tol_mm"] == pl.GC_CONTACT_TOL_MM
    assert gate["pad_face_offset_mm"] == pytest.approx(robot_facts.PAD_SEPARATION_OFFSET_MM / 2.0)
    # the module function agrees with what the plan stored (pure)
    ctx = pl._Context(a, hf, inputs(), (), "turn", None, part_ring_cells=ring)
    again = pl.close_gate(ctx, plan.grip_point_xyz[:2], plan.yaw_rad, plan.width["e_lat_mm"] / 1000.0,
                          plan.width["w_mm"] / 2000.0, plan.width["sigma_w_mm"])
    assert as_json(again) == as_json({k: gate[k] for k in ("fingers", "band_g13_m")})
    # an overgrown own section (cells 40 mm past the grip on finger 2's side): the face moves out with the own cells
    grown = np.vstack([a.region_cells, hf.cells_in_polygon(rect_polygon((0.0, 0.030), (1.0, 0.0), 0.004, 0.006))])
    wide = pl._Context(dataclasses.replace(a, region_cells=grown), hf, inputs(), (), "turn", None)
    g2 = pl.close_gate(wide, plan.grip_point_xyz[:2], plan.yaw_rad, 0.008, plan.width["w_mm"] / 2000.0, 1.0)
    w2 = g2["fingers"]["finger2"]
    assert w2["own_extent_m"] > plan.width["w_mm"] / 1000.0
    assert w2["F_m"] == pytest.approx(w2["own_extent_m"] + band) and w2["F_m"] > w2["near_edge_m"]


def test_gc1_gate_never_admits_or_refuses_a_candidate(monkeypatch):
    """The gate is evidence for the close only: replacing close_gate changes no admission, rule, grip or score."""
    def scenes():
        hf, a = _offset_lever()
        yield (a, hf, dict(purpose="turn", depth_rules=("height_stop",), part_ring_cells=_halo_row(hf, 0.020)))
        hf, c, d = thin_part_scene(0.002)
        stamp(hf, rect_polygon((0.015, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
        yield (c, hf, dict(purpose="carry", next_held=d))
        hf, c, d = thin_part_scene(0.002)
        stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
        yield (c, hf, dict(purpose="carry", next_held=d))

    def summary(result):
        if isinstance(result, pl.ContactPlan):
            return ("plan", result.key(), result.admission, result.grip_point_xyz, result.score)
        return ("infeasible", result.reason, tuple(r.rule for r in result.rejections))
    real = [summary(pl.plan_contact(p, hf, inputs(), (), **kw)) for p, hf, kw in scenes()]
    assert [s[0] for s in real] == ["plan", "plan", "infeasible"]
    monkeypatch.setattr(pl, "close_gate", lambda *args, **kwargs: {"fingers": {}, "band_g13_m": 0.0})
    assert [summary(pl.plan_contact(p, hf, inputs(), (), **kw)) for p, hf, kw in scenes()] == real


def test_gc1_every_contact_plan_carries_a_json_safe_gate():
    """Every ContactPlan (wall rim, rotary lever, thin part, free body) carries checks["gate"] with
    both fingers' F <= R, and the plan's dict stays JSON-safe."""
    plans = []
    hf, b = wall_scene()
    plans.append(pl.plan_contact(b, hf, inputs(jaw=0.0), (), purpose="carry"))
    hf, a = lever_scene()
    plans.append(pl.plan_contact(a, hf, inputs(), (), purpose="turn"))
    hf, c, d = thin_part_scene(0.001)
    plans.append(pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d))
    hf, d = body_scene()
    plans.append(pl.plan_contact(d, hf, inputs(jaw=0.0, g_open=78.5, sigma_align=0.002), (), purpose="carry"))
    assert all(isinstance(p, pl.ContactPlan) for p in plans), [getattr(p, "reason", None) for p in plans]
    for plan in plans:
        gate = plan.checks["gate"]
        json.dumps(plan.to_dict(), allow_nan=False)
        assert set(gate) >= {"fingers", "grip_xy", "jaw_xy", "span_hi_mm", "tol_mm", "pad_face_offset_mm"}
        for finger in ("finger1", "finger2"):
            f = gate["fingers"][finger]
            assert set(f) == {"F_m", "R_m", "near_edge_m", "own_extent_m"}
            assert 0.0 < f["F_m"] < f["R_m"]


def test_gc1_gate_puts_a_surface_beside_the_part_beyond_the_finger_limit():
    """Negative (the GC-1 contract with the close): on a clean thin-part plan the part's own faces (6.5 mm out) lie
    inside each finger's expected face F (7.5 + 8 + 2.5 = 18 mm), and a surface beside the part 22.5 mm out (its
    near face) lies beyond R = 19 mm, so a finger meeting it during the close is an early contact (the guard aborts);
    the same surface measured in the field is refused by P5 before any close. (A measured support: F6-B.)"""
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.admission == "clean" and plan.yaw_rad == pytest.approx(0.0)
    gate = plan.checks["gate"]
    face = plan.width["w_mm"] / 2000.0
    obstacle_face = 0.025 - hf.cell_m / 2.0
    for finger in ("finger1", "finger2"):
        f = gate["fingers"][finger]
        assert f["F_m"] == pytest.approx(0.018, abs=1e-9) and f["R_m"] == pytest.approx(0.019, abs=1e-9)
        assert face <= f["F_m"] and obstacle_face > f["R_m"]
    assert plan.width["w_mm"] <= gate["span_hi_mm"] < plan.width["w_mm"] / 2.0 + obstacle_face * 1000.0
    stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "column_blocked"


def test_pc1_evaluate_candidate_at_the_gate_point_reproduces_the_plan_and_a_shifted_point_re_runs_the_checks():
    """PC-1 support: evaluate_candidate(grip_xy=the gate's grip point) reproduces the plan exactly; a realized hand
    20 mm across the jaw re-runs P16 / P5 there (z unchanged): an obstacle beside the part outside the planned finger
    strips but inside the shifted ones refuses (P5 column_blocked), and the clean field plans at the shifted point
    with its gate re-centred there. (A measured support: F6-B.)"""
    hf, c, d = thin_part_scene(0.002, plane_filled=False)
    stamp(hf, rect_polygon((0.025, -0.028), (1.0, 0.0), 0.006, 0.002), 1.04, sigma=0.002)
    plan = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(plan, pl.ContactPlan) and plan.admission == "clean"
    kw = dict(purpose="carry", next_held=d, depth_rules=(plan.depth_rule,))
    same = pl.evaluate_candidate(c, hf, inputs(), plan.family, plan.yaw_rad, (), grip_xy=plan.checks["gate"]["grip_xy"],
                                 **kw)
    assert as_json(same.to_dict()) == as_json(plan.to_dict())
    shifted_xy = np.asarray(plan.grip_point_xyz[:2]) + np.array([0.0, -0.020])
    moved = pl.evaluate_candidate(c, hf, inputs(), plan.family, plan.yaw_rad, (), grip_xy=shifted_xy, **kw)
    assert isinstance(moved, pl.CandidateRejection) and moved.reason == "column_blocked"
    assert moved.rule == f"{plan.depth_rule}:P5"
    assert moved.values["grip_point_xyz"][:2] == pytest.approx(list(shifted_xy))
    assert moved.values["grip_point_xyz"][2] == pytest.approx(plan.grip_point_xyz[2])
    hf2, c2, d2 = thin_part_scene(0.002)
    clean = pl.evaluate_candidate(c2, hf2, inputs(), plan.family, plan.yaw_rad, (), purpose="carry", next_held=d2,
                                  depth_rules=(plan.depth_rule,), grip_xy=shifted_xy)
    assert isinstance(clean, pl.ContactPlan)
    assert clean.grip_point_xyz == pytest.approx((*shifted_xy, plan.grip_point_xyz[2]))
    assert clean.checks["gate"]["grip_xy"] == pytest.approx(list(shifted_xy))


def test_cj1_representative_rejection_is_the_highest_progress_depth_rejection():
    """CJ-1: the depth-stage rejection with the highest by_rule progress (the first on a tie); orient / width / part /
    axis rejections are skipped; None for a refusal without a depth rejection and for a ContactPlan. On a real
    refusal it is one of the refusal's own depth rejections."""
    def rej(yaw, rule, progress=None):
        values = {} if progress is None else {"by_rule": {r: {"check": rule.split(":")[1], "progress": p}
                                                          for r, p in progress.items()}}
        return pl.CandidateRejection("top_down", yaw, rule, "column_blocked", values)
    rows = (rej(0.0, "orient:R12"), rej(0.5, "width:P2"), rej(1.0, "height_stop:P5", {"height_stop": 0.5}),
            rej(1.5, "touch_stop:P3", {"touch_stop": 0.6, "height_stop": 0.8}), rej(2.0, "part:P0"),
            rej(2.5, "axis:D10"), rej(3.0, "height_stop:P7", {"height_stop": 0.8}))
    best = pl.representative_rejection(pl.Infeasible("A", "column_blocked", rows, ()))
    assert best is rows[3]
    assert pl.representative_rejection(pl.Infeasible("A", "yaw_budget", rows[:2] + rows[4:6], ())) is None
    assert pl.representative_rejection(pl.Infeasible("A", "provisional", (), ())) is None
    hf, c, d = thin_part_scene(0.002)
    assert pl.representative_rejection(pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)) is None
    stamp(hf, rect_polygon((0.025, -0.005), (1.0, 0.0), 0.002, 0.002), 1.02, sigma=0.002, source="kappa")
    refused = pl.plan_contact(c, hf, inputs(), (), purpose="carry", next_held=d)
    rep = pl.representative_rejection(refused)
    assert rep in refused.rejections and not rep.rule.startswith(("orient:", "width:", "part:", "axis:"))
    depth = [r for r in refused.rejections if "by_rule" in r.values]
    top = max(max(v["progress"] for v in r.values["by_rule"].values()) for r in depth)
    assert max(v["progress"] for v in rep.values["by_rule"].values()) == top
