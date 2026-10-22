"""Gap round (judge decision §2.4, planner-7): focused tests of the planner rules IF-HS, IF-PV (fallback), IF-R3'
(M16 occupancy + centre-first + tight_fit), SAL-R2 (C-7'), F6-B (D7a-only landings unverified) and the D7b
restatement. CV-CI1 / CI2 / V5U are covered by test_agent_speedup_v2_cv5.py. Synthetic geometry only (the T1 builders
of test_agent_speedup_v2_planner); no simulator, no model, no truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_planner as pl
from gpt_field import agent_speedup_v2_scene as scene
from gpt_field.agent_speedup_v2_types import hf_code, rect_polygon

from test_agent_speedup_v2_planner import (ROBOT, SUPPORT, disk, field, held_frame, inputs, part, stamp, state8,
                                           thin_part_scene)

FOOTPRINT_XY = (0.070, 0.020)          # in the hand footprint of the thin part's plan, outside the slot and columns


def _plan(hf, c, d, keepouts=()):
    return pl.plan_contact(c, hf, inputs(), keepouts, purpose="carry", next_held=d)


def _g_star(c):
    offset = max(pl.P6_TARGET_FLOOR_MM / 1000.0, pl.P6_TARGET_SIGMA_K * float(c.sigma_h_m))
    return float(c.top_z.value) + float(ROBOT.pad_span_m[0]) - offset


# ------------------------------------------------------------------------------------------------ restatement
def test_d7b_fill_sigma_k_restates_the_scene_and_new_constants_are_labelled():
    assert pl.D7B_FILL_SIGMA_K == scene.D7_OBSTACLE_SIGMA_K
    assert pl.HS_P3_GUARD_SIGMA_K == 2.0 and pl.PV_PARENT_SIGMA_K == 1.0
    for name in ("HS_P3_GUARD_SIGMA_K", "PV_PARENT_SIGMA_K", "D7B_FILL_SIGMA_K", "CANDIDATE_YAW_TOL_RAD",
                 "V5U_CONE_LINES"):
        assert pl.LABELS[name][0] == "DD" and pl.LABELS[name][1]
    assert pl.PLANNER_VERSION == "agent-speedup-v2-planner-7"


# ------------------------------------------------------------------------------------------------ IF-HS
def test_hs_raises_the_stop_by_exactly_the_p7_deficit():
    """A structure level with the part's top in the hand footprint refuses P7 at g* (sigma_h 5 mm); the stop is
    raised by the P7 deficit, P7 then clears with zero margin left, and P3 holds with the top 2 sigma_h lower."""
    hf, c, d = thin_part_scene(0.005)
    stamp(hf, rect_polygon((0.014, 0.020), (1.0, 0.0), 0.006, 0.005), 1.034)
    plan = _plan(hf, c, d)
    assert isinstance(plan, pl.ContactPlan) and plan.depth_rule == "height_stop"
    raise_m = plan.checks["hs_raise_mm"] / 1000.0
    assert raise_m > 0.0
    assert plan.target_grip_z == pytest.approx(_g_star(c) + raise_m, abs=1e-9)
    assert plan.checks["P7_clearance_mm"] == pytest.approx(0.0, abs=1e-6)
    assert plan.checks["hs_guard_overlap_mm"] >= pl.P3_MIN_OVERLAP_MM
    assert {"HS_raise_m", "HS_P3_guard_top_z_m"} <= {t.name for t in plan.thresholds}
    # PC-1 / CJ-1 path: evaluate_candidate of the same candidate gets the same raise
    one = pl.evaluate_candidate(c, hf, inputs(), plan.family, plan.yaw_rad, (), purpose="carry", next_held=d,
                                depth_rules=("height_stop",))
    assert isinstance(one, pl.ContactPlan) and one.target_grip_z == pytest.approx(plan.target_grip_z)


def test_hs_refused_beyond_the_p3_guard_and_admitted_plans_unchanged():
    """Negative: a structure 5.5 mm above the part's top needs an 8.6 mm raise, beyond the guard (P3 with the top 2
    sigma_h lower) -> the P7 refusal stands with the retry's evidence. A plan whose P7 passes at g* carries no raise."""
    hf, c, d = thin_part_scene(0.005)
    stamp(hf, rect_polygon((0.014, 0.020), (1.0, 0.0), 0.006, 0.005), 1.040)
    refused = _plan(hf, c, d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "footprint_blocked"
    retries = [r.values["by_rule"]["height_stop"]["values"].get("hs_retry") for r in refused.rejections
               if r.values.get("by_rule", {}).get("height_stop", {}).get("check") == "P7"]
    assert retries and all(t and t["check"] == "P3" and t["guard_overlap_mm"] < pl.P3_MIN_OVERLAP_MM for t in retries)
    hf2, c2, d2 = thin_part_scene(0.002, plane_filled=False)
    clean = _plan(hf2, c2, d2)
    assert isinstance(clean, pl.ContactPlan) and "hs_raise_mm" not in clean.checks
    assert clean.target_grip_z == pytest.approx(_g_star(c2), abs=1e-9)


def test_hs_measured_obstacle_beyond_the_guard_stays_refused():
    """A measured cell well above the part's top (2.5 cm) in the hand footprint: no raise within the guard clears it."""
    hf, c, d = thin_part_scene(0.001, plane_filled=False)
    stamp(hf, rect_polygon(FOOTPRINT_XY, (1.0, 0.0), 0.002, 0.002), 1.06, sigma=0.002)
    refused = _plan(hf, c, d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "footprint_blocked"
    assert not pl.pv_fallback_applies(refused)                 # a measured blocker: no PV fallback


# ------------------------------------------------------------------------------------------------ IF-PV
def _fill_scene(parent, sigma=0.005, source="obstacle"):
    hf, c, d = thin_part_scene(0.001, plane_filled=False)
    stamp(hf, rect_polygon(FOOTPRINT_XY, (1.0, 0.0), 0.002, 0.002), parent + pl.D7B_FILL_SIGMA_K * sigma, sigma=sigma,
          source=source)
    return hf, c, d


def test_pv_admits_a_phantom_whose_parent_plus_one_sigma_clears_and_keeps_its_reading():
    """A D7b fill (stored parent + 3 sigma = 1.055) blocks P7 at g* and beyond the HS guard; read at its parent + 1
    sigma (1.045) it clears: the fallback plan is marginal and carries p7_reading; revalidate and the PC-1
    re-evaluation read P7 the same way (without the reading the same candidate is refused)."""
    hf, c, d = _fill_scene(1.040)
    plan = _plan(hf, c, d)
    assert isinstance(plan, pl.ContactPlan) and plan.checks["p7_reading"] == "parent_sigma"
    assert plan.admission == "marginal" and plan.marginal
    again = pl.revalidate(plan, c, hf, inputs(), (), purpose="carry", next_held=d)
    assert isinstance(again, pl.ContactPlan) and again.checks["p7_reading"] == "parent_sigma"
    kw = dict(purpose="carry", next_held=d, depth_rules=(plan.depth_rule,), grip_xy=plan.checks["gate"]["grip_xy"])
    pc1 = pl.evaluate_candidate(c, hf, inputs(), plan.family, plan.yaw_rad, (),
                                p7_reading=plan.checks.get("p7_reading"), **kw)
    assert isinstance(pc1, pl.ContactPlan) and pc1.target_grip_z == pytest.approx(plan.target_grip_z)
    stored = pl.evaluate_candidate(c, hf, inputs(), plan.family, plan.yaw_rad, (), **kw)
    assert isinstance(stored, pl.CandidateRejection) and stored.reason == "footprint_blocked"


def test_pv_refuses_a_fill_whose_parent_stands_above_the_limit_and_a_measured_blocker():
    """Negatives: a category-C fill (parent 1.050: parent + 1 sigma still above the limit even with HS) and a
    measured cell at the same stored height (no D7b blocker: the fallback never runs)."""
    hf, c, d = _fill_scene(1.050)
    refused = _plan(hf, c, d)
    assert isinstance(refused, pl.Infeasible) and refused.reason == "footprint_blocked"
    assert pl.pv_fallback_applies(refused)
    hf2, c2, d2 = _fill_scene(1.040, source="kappa")
    measured = _plan(hf2, c2, d2)
    assert isinstance(measured, pl.Infeasible) and not pl.pv_fallback_applies(measured)


def test_pv_never_runs_when_the_base_plan_is_admitted():
    """A fill below the P7 limit at g* (parent 1.030): the base plan is admitted, with the stored reading."""
    hf, c, d = _fill_scene(1.030)
    plan = _plan(hf, c, d)
    assert isinstance(plan, pl.ContactPlan) and "p7_reading" not in plan.checks and plan.admission == "clean"


# ------------------------------------------------------------------------------------------------ F6-B
def test_f6b_a_column_of_d7a_cells_only_is_an_unverified_landing():
    """The thin part's columns land on a D7a plane-filled support: marginal (P14 ladder, GC-1 active), the landing
    height unchanged and RC-1a's unverified count untouched."""
    hf, c, d = thin_part_scene(0.002)                       # the support is D7a plane-filled
    plan = _plan(hf, c, d)
    assert isinstance(plan, pl.ContactPlan) and plan.admission == "marginal" and plan.marginal
    assert plan.checks["plane_landing_columns"] == 2 and plan.checks["unverified_cells"] == 0
    assert all(col.landing_plane_only and col.landing_z == pytest.approx(SUPPORT) for col in plan.columns)
    measured = _plan(*thin_part_scene(0.002, plane_filled=False))
    assert measured.admission == "clean" and measured.checks["plane_landing_columns"] == 0
    assert [col.landing_z for col in measured.columns] == pytest.approx([col.landing_z for col in plan.columns])


def test_f6b_one_measured_cell_at_the_landing_keeps_the_label():
    """Negative: one measured cell at the landing height inside each finger column -> the landing is verified."""
    hf, c, d = thin_part_scene(0.002)
    base = _plan(hf, c, d)
    for col in base.columns:
        ij = hf.index(np.asarray(col.polygon_xy, float).mean(axis=0))      # the column's centre cell, measured
        hf.source[ij] = hf_code("kappa")
    plan = _plan(hf, c, d)
    assert isinstance(plan, pl.ContactPlan) and plan.checks["plane_landing_columns"] == 0
    assert plan.admission == "clean" and not any(col.landing_plane_only for col in plan.columns)


def test_f6b_d7a_column_under_a_d7b_fill_keeps_rc1a_count():
    """A D7b fill above the landing limit in a D7a column: marginal, and the RC-1a unverified count is the one the
    same fill gives over a measured support."""
    counts = {}
    for plane in (True, False):
        hf, c, d = thin_part_scene(0.002, plane_filled=plane)
        stamp(hf, rect_polygon((0.045, 0.0), (1.0, 0.0), 0.002, 0.002), 1.03, sigma=0.002, source="obstacle")
        plan = _plan(hf, c, d)
        assert isinstance(plan, pl.ContactPlan) and plan.marginal
        counts[plane] = (plan.checks["column_unverified"], plan.checks["unverified_cells"])
    assert counts[True] == counts[False] and counts[True][0] >= 1


# ------------------------------------------------------------------------------------------------ IF-R3' (M16)
def _destination(half=0.095, top=0.9305):
    hf = field()
    region = rect_polygon((0.0, 0.0), (1.0, 0.0), half, half)
    cells = stamp(hf, region, top)
    ref = part("B", xyz=(0.0, 0.0, top), top=top, sigma_h=0.002, move="fixed", region=region, cells=cells,
               roles=("ref",))
    return hf, ref


POSE = ROBOT.pose(state8((0.0, -0.1, 1.12), 0.0, 14.0))


def test_m16_fill_parent_reading_frees_the_region_and_targets_the_centre():
    """t2-like: D7b fills across the destination whose stored height (parent + 3 sigma) is above s_dest + band but
    whose parent + 1 sigma is not: they do not occupy, the target is the region centre and fits."""
    hf, ref = _destination()
    stamp(hf, rect_polygon((0.05, 0.05), (1.0, 0.0), 0.03, 0.03), 0.9305 + 3 * 0.005, sigma=0.005, source="obstacle")
    aim = pl.aim_target(ref, held_frame(r_obj=0.035, sigma_offset=0.003), hf, (), POSE)
    assert aim["ok"] and aim["fits"] and not aim["free_spot"] and aim["flags"] == []
    assert aim["occupied_by"] == {"verified": 0, "fill": 0} and aim["target_rule"] == "centre"
    assert aim["target_obj_xy"] == pytest.approx([0.0, 0.0], abs=0.003)


def test_m16_no_fitting_cell_takes_the_largest_clearance_and_flags_tight_fit():
    """An H1-sized held frame (r_obj 4.8 cm, sigma_offset 2.4 cm: need 7.2 cm) on a destination with a measured
    obstacle strip: no cell fits -> the largest-clearance free cell, flag tight_fit, the carry is not refused."""
    hf, ref = _destination(half=0.06)
    stamp(hf, rect_polygon((0.045, 0.0), (1.0, 0.0), 0.015, 0.06), 0.98)
    aim = pl.aim_target(ref, held_frame(r_obj=0.048, sigma_offset=0.024), hf, (), POSE)
    assert aim["ok"] and aim["reason"] is None and not aim["fits"]
    assert aim["flags"] == ["tight_fit"] and aim["target_rule"] == "largest_clearance"
    assert aim["target_obj_xy"][0] < 0.0 and aim["occupied_by"]["verified"] > 0


def test_m16_centre_first_beats_the_largest_disk():
    """A small occupied block near a corner: the frozen rule took the largest free disk (away from the block); the
    centre-first rule takes the fitting cell nearest the region centre."""
    hf, ref = _destination()
    stamp(hf, rect_polygon((0.08, 0.08), (1.0, 0.0), 0.006, 0.006), 0.99)
    aim = pl.aim_target(ref, held_frame(r_obj=0.035, sigma_offset=0.003), hf, (), POSE)
    assert aim["ok"] and aim["fits"] and aim["free_spot"] and aim["target_rule"] == "centre_first"
    assert aim["target_to_centre_m"] <= hf.cell_m


def test_m16_placed_object_with_a_fill_flank_keeps_its_footprint_out():
    """Negative: an object standing on the destination (measured top 5 cm up) with a D7b flank parented to its top
    (parent + 1 sigma above s_dest + band): the flank occupies, and the held footprint keeps out of it (gap >= 0)."""
    hf, ref = _destination()
    obj_top = 0.9805
    stamp(hf, disk((0.04, 0.0), 0.03), obj_top + 3 * 0.005, sigma=0.005, source="obstacle")      # the flank
    stamp(hf, disk((0.04, 0.0), 0.02), obj_top)                                                    # the object
    held = held_frame(r_obj=0.035, sigma_offset=0.003)
    aim = pl.aim_target(ref, held, hf, (), POSE)
    assert aim["ok"] and aim["fits"] and aim["occupied_by"]["fill"] > 0
    target = np.asarray(aim["target_obj_xy"])
    flank = hf.cells_in_polygon(disk((0.04, 0.0), 0.03))
    gap = np.min(np.linalg.norm(pl._cell_centres(hf, flank) - target, axis=1)) - hf.cell_m / 2.0 - held.r_obj_m
    assert gap >= -1e-9


# ------------------------------------------------------------------------------------------------ SAL-R2 (C-7')
def _dest_record(step, sigma_top, region=None, cells=None):
    return part("B", xyz=(0.0, 0.0, SUPPORT + step), top=SUPPORT + step, sigma_h=sigma_top, support=SUPPORT,
                move="fixed", region=region, cells=cells, roles=("ref",))


def test_c7p_burner_step_passes_laterally_on_a_12mm_hull_and_a_40mm_ring():
    small = pl.destination_precondition(_dest_record(0.0045, 0.0015, region=disk((0.0, 0.0), 0.0125)))
    assert small["ok"] and small["cue"] == "lateral"
    assert small["values"]["r_hull_m"] == pytest.approx(0.0125, abs=0.0005)
    hf = field()
    ring_cells = np.array([c for c in hf.cells_in_polygon(disk((0.0, 0.0), 0.040))
                           if np.linalg.norm(hf.cell_xy(tuple(c))) > 0.030])
    ring = pl.destination_precondition(_dest_record(0.0045, 0.0015, cells=ring_cells), hf)
    assert ring["ok"] and ring["cue"] == "lateral" and ring["values"]["r_hull_m"] > 0.035


def test_c7p_refusals_and_the_vertical_cue():
    """sigma_top 9.6 mm fails both clauses; a 24 mm step passes vertically without any region; a 6 mm hull fails the
    lateral clause."""
    noisy = pl.destination_precondition(_dest_record(0.0045, 0.0096, region=disk((0.0, 0.0), 0.04)))
    assert not noisy["ok"] and noisy["reason"] == "geometry_unmeasured" and noisy["cue"] is None
    tall = pl.destination_precondition(_dest_record(0.024, 0.0015))
    assert tall["ok"] and tall["cue"] == "vertical" and tall["values"]["r_hull_m"] is None
    tiny = pl.destination_precondition(_dest_record(0.0045, 0.0015, region=disk((0.0, 0.0), 0.006)))
    assert not tiny["ok"] and tiny["values"]["r_hull_m"] < pl.M4_FLOOR_M + 0.005
    assert math.isclose(tall["values"]["vertical_step_min_m"], 2.0 * pl.M8_FLOOR_M)


def test_c7p_unmeasured_support_is_refused_as_before():
    rec = part("B", xyz=(0.0, 0.0, 0.93), top=0.93, sigma_h=0.001, support=None, move="fixed", roles=("ref",))
    out = pl.destination_precondition(rec)
    assert not out["ok"] and out["reason"] == "geometry_unmeasured" and out["cue"] is None
