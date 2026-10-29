"""candidate_v5 (gap round, summary gaps 4 and 5): candidate identity (CI-1 tolerance, CI-2 re-validation of the
executing candidate, CI-3 re-Align from the grasp hover, CI-4 no second marginal descent after a refuting touch) and
the V5 grasp-point check without a candidate jaw (V5-U triage over every jaw line + the decisive check at the planned
jaw). Positive AND negative cases. No simulator, no model.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_planner as pl
from gpt_field import agent_speedup_v2_relations as R
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field.agent_speedup_v2_types import YAW_GRID_RAD

import test_agent_speedup_v2_planner as TP
import test_agent_speedup_v2_relations as TR


# ------------------------------------------------------------------------------------------------ CI-1
def test_ci1_candidate_identity_is_half_the_planner_yaw_grid():
    tol = pl.CANDIDATE_YAW_TOL_RAD
    assert tol == pytest.approx(YAW_GRID_RAD / 2.0)
    a = -0.78592                                                            # t8: -45.03 deg
    assert pl.same_candidate_angle(a, a + math.radians(-0.124))            # t8 post-Align drift: same candidate
    assert pl.same_candidate_angle(math.pi - 1e-3, -math.pi + 1e-3)       # across the wrap
    assert not pl.same_candidate_angle(a, a + YAW_GRID_RAD)               # the next grid candidate
    assert not pl.same_candidate_angle(a, a + tol + 1e-6)
    assert not pl.same_candidate_angle(a, a + math.pi)                    # the 180 deg twin
    assert pl.same_candidate(("top_down", a), ("top_down", a + 2e-3))
    assert not pl.same_candidate(("top_down", a), ("other_family", a))
    assert not pl.same_candidate(("top_down", a), ("top_down", a, None, 0.3))            # plain vs rim candidate
    assert not pl.same_candidate(("top_down", a, None, 0.3), ("top_down", a, None, 0.3 + math.pi))   # rim side
    assert pl.same_candidate(("top_down", a, None, 0.3), ("top_down", a + 1e-3, None, 0.301))


def test_ci1_never_merges_two_candidates_of_one_plan_call():
    """Every candidate set of one plan_contact call is >= YAW_GRID_RAD apart (grid) or pi apart (axis twins)."""
    hf, d = TP.body_scene()
    for jaw in (0.0, 0.0021, -1.3):
        yaws = pl.jaw_yaws(d, "top_down", TP.inputs(jaw=jaw, g_open=78.5))
        for i, y in enumerate(yaws):
            assert not any(pl.same_candidate_angle(y, z) for z in yaws[i + 1:])
    hf, a = TP.lever_scene()
    yaws = pl.jaw_yaws(a, "top_down", TP.inputs())
    assert len(yaws) == 2 and not pl.same_candidate_angle(*yaws)


def test_ci1_a_drifted_copy_of_an_excluded_candidate_stays_excluded():
    """t8 1215 / t2 1992-2231: the grid of a part without an axis is anchored at the current jaw; after Align the
    jaw drifted 0.12 deg and the excluded candidate came straight back (exact-float _same_angle)."""
    hf, d = TP.body_scene()
    first = pl.plan_contact(d, hf, TP.inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    assert isinstance(first, pl.ContactPlan)
    drifted = pl.plan_contact(d, hf, TP.inputs(jaw=0.0021, g_open=78.5), (), purpose="carry")
    assert isinstance(drifted, pl.ContactPlan) and drifted.candidate_key() != first.candidate_key()   # the bug
    assert pl.same_candidate(drifted.candidate_key(), first.candidate_key())
    again = pl.plan_contact(d, hf, TP.inputs(jaw=0.0021, g_open=78.5), (), purpose="carry",
                            exclude=[first.candidate_key()])
    assert isinstance(again, pl.ContactPlan)
    assert abs(pl._wrap(again.yaw_rad - first.yaw_rad)) >= YAW_GRID_RAD - 0.01        # the NEXT candidate
    # negative: an exclusion one grid step away never removes the candidate itself
    kept = pl.plan_contact(d, hf, TP.inputs(jaw=0.0, g_open=78.5), (), purpose="carry",
                           exclude=[("top_down", first.yaw_rad + YAW_GRID_RAD)])
    assert isinstance(kept, pl.ContactPlan) and kept.candidate_key() == first.candidate_key()
    # a rule-level exclusion (ContactPlan.key) at the drifted yaw removes only that rule
    skip_all, rules = pl._exclusion([first.key()], "top_down", first.yaw_rad + 0.002)
    assert not skip_all and rules == {first.depth_rule}


# ------------------------------------------------------------------------------------------------ CI-2
def test_ci2_revalidate_keeps_the_executing_candidate_after_a_jaw_drift():
    hf, d = TP.body_scene()
    first = pl.plan_contact(d, hf, TP.inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    later = TP.inputs(jaw=0.0021, g_open=78.5, sigma_align=0.0016)
    search = pl.plan_contact(d, hf, later, (), purpose="carry")
    assert search.candidate_key() != first.candidate_key()                  # frozen: "candidate_changed"
    kept = pl.revalidate(first, d, hf, later, (), purpose="carry")
    assert isinstance(kept, pl.ContactPlan)
    assert kept.yaw_rad == first.yaw_rad and kept.depth_rule == first.depth_rule
    assert kept.candidate_key() == first.candidate_key()


def test_ci2_revalidate_refuses_on_the_candidates_own_checks_never_candidate_changed():
    hf, d = TP.body_scene()
    first = pl.plan_contact(d, hf, TP.inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    # negative 1: excluded -> no candidate
    gone = pl.revalidate(first, d, hf, TP.inputs(jaw=0.0021, g_open=78.5), (), purpose="carry",
                         exclude=[first.candidate_key()])
    assert isinstance(gone, pl.Infeasible) and gone.reason == "no_feasible_candidate"
    # negative 2: a measured sigma that breaks the candidate's own P1 / P2 width check -> that check's reason
    wide = TP.inputs(jaw=0.0021, g_open=78.5, sigma_align=0.02)
    own = pl.evaluate_candidate(d, hf, wide, first.family, first.yaw_rad, (), purpose="carry")
    out = pl.revalidate(first, d, hf, wide, (), purpose="carry")
    if isinstance(own, pl.ContactPlan):
        assert isinstance(out, pl.ContactPlan)
    else:
        assert isinstance(out, pl.Infeasible) and out.reason == own.reason and out.reason != "candidate_changed"
    # negative 3: a record whose width / axis is no longer measured -> geometry_unmeasured
    bare = dataclasses.replace(d, width_by_yaw=())
    out = pl.revalidate(first, bare, hf, TP.inputs(jaw=0.0, g_open=78.5), (), purpose="carry")
    assert isinstance(out, pl.Infeasible) and out.reason == "geometry_unmeasured"


def test_ci2_grasp_descends_on_the_aligned_candidate(monkeypatch):
    first = TR.contact_plan()
    script = TR.GraspScript(monkeypatch, plans=[first], classes=["expected"])
    monkeypatch.setattr(R, "_align", TR.scripted({"ok": True, "reason": None, "sigma_align_m": 0.001,
                                                  "track": None, "jacobian": None, "offset": [0.0, -0.05]}))
    seen = []

    def revalidate(ctx, plan, record, **kw):
        seen.append((plan, kw.get("sigma_align_m"), list(kw.get("exclude") or ())))
        return plan
    monkeypatch.setattr(R, "_revalidate", revalidate)
    ctx = TR.make_ctx()
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "held_for_turn")
    assert len(seen) == 1 and seen[0][0] is first and seen[0][1] == 0.001
    assert [row for row in script.log if row[0] == "plan"] == [("plan", False)]      # no re-search
    assert "sigma_align_refused" not in outcome.evidence["attempts"][0]


def test_ci2_a_refused_revalidation_names_its_own_reason_and_returns_to_the_grasp_hover(monkeypatch):
    """Replaces test_grasp_names_the_refusal_after_align_and_puts_the_decisive_check_first (the sigma re-plan is now
    _revalidate, not _plan)."""
    first = TR.contact_plan()
    k6 = kernel.th_value("K6_yaw_budget_rad", 1.9, "ROBOT", "R12")
    p1 = kernel.th_value("P1_e_lat_mm", 9.1, "ONLINE", "P1")
    twin = pl.CandidateRejection("top_down", math.pi / 2, "orient:R12", "yaw_budget", {}, (k6,))
    script = TR.GraspScript(monkeypatch, plans=[first, pl.Infeasible("A", "yaw_budget", (twin,), (p1, k6))],
                            classes=[])
    monkeypatch.setattr(R, "_align", TR.scripted({"ok": True, "reason": None, "sigma_align_m": 0.001,
                                                  "track": None, "jacobian": None, "offset": [0.0, -0.05]}))
    monkeypatch.setattr(R, "_revalidate", lambda ctx, plan, record, **kw: pl.Infeasible("A", "column_blocked", (),
                                                                                          (p1,)))
    hovers = []

    def back(ctx, hover_z, *, quaternion=None, xyz=None):
        hovers.append(None if xyz is None else [float(v) for v in xyz])
        return {"reason": "travel_reached"}
        yield
    monkeypatch.setattr(R, "_return_to_hover", back)
    ctx = TR.make_ctx()
    start = [float(v) for v in ctx.view.position]
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "yaw_budget")
    assert outcome.evidence["attempts"][0]["sigma_align_refused"] == "column_blocked"
    assert script.excludes[1] == [first.candidate_key()]
    assert outcome.thresholds[0].name == "K6_yaw_budget_rad"
    assert "aligned candidate refused column blocked" in outcome.evidence_text
    assert "close" not in script.names()
    assert hovers == [start]                                                # CI-3: back to the grasp hover xy


# ------------------------------------------------------------------------------------------------ CI-3
def test_ci3_return_to_hover_lifts_then_moves_back_to_the_given_hover_position(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "_ensure_open", TR.scripted(None))

    def lift(ctx, travel, **kw):
        calls.append(("lift", round(float(travel), 6)))
        return "travel_reached"
        yield

    def return_to(ctx, target, quat, *, name):
        calls.append(("return_to", [round(float(v), 6) for v in target], name))
        return "goal_reached"
        yield
    monkeypatch.setattr(R, "_lift", lift)
    monkeypatch.setattr(R, "_return_to", return_to)
    ctx = TR.make_ctx()
    z0 = float(ctx.view.position[2])
    hover = [0.01, 0.02, z0 + 0.05]
    out = TR.drive(R._return_to_hover(ctx, z0 + 0.05, xyz=hover), ctx)
    assert out == "travel_reached"
    assert calls == [("lift", 0.05), ("return_to", [round(v, 6) for v in hover], "return_to_hover_xy")]
    calls.clear()                                                            # negative: no xyz -> lift only
    TR.drive(R._return_to_hover(ctx, z0 + 0.05), ctx)
    assert calls == [("lift", 0.05)]


# ------------------------------------------------------------------------------------------------ CI-4
def test_ci4_no_second_marginal_descent_after_a_refuting_touch(monkeypatch):
    ctx = TR.make_ctx()
    ctx.scene.hf = kernel.HeightField.empty(ctx.view.position[:2], half_extent_m=0.2)
    first = TR.gated_plan(ctx.view, marginal=True)
    second = TR.gated_plan(ctx.view, marginal=True, yaw=first.yaw_rad + YAW_GRID_RAD)
    abort = {"aborted": "early_finger_contact", "guard": TR.touch_guard(), "close_grip_z": 0.944}
    script = TR.GraspScript(monkeypatch, plans=[first, second], classes=["expected"] * 2,
                            closes=[abort, (True, False)])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    # SAL-F5 (kernel v8): early_finger_contact is a REASONS code of its own, the touch named in the text
    assert (outcome.status, outcome.reason) == ("failed", "early_finger_contact")
    assert outcome.evidence["next_grip"] and script.names().count("close") == 1
    assert "marginal candidates refused after a finger touched beyond its face limit" in outcome.evidence_text


def test_ci4_a_clean_candidate_still_runs_after_a_marginal_touch(monkeypatch):
    ctx = TR.make_ctx()
    ctx.scene.hf = kernel.HeightField.empty(ctx.view.position[:2], half_extent_m=0.2)
    first = TR.gated_plan(ctx.view, marginal=True)
    second = TR.gated_plan(ctx.view, yaw=first.yaw_rad + YAW_GRID_RAD)
    abort = {"aborted": "early_finger_contact", "guard": TR.touch_guard(), "close_grip_z": 0.944}
    script = TR.GraspScript(monkeypatch, plans=[first, second], classes=["expected"] * 2,
                            closes=[abort, (True, False)])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "held_for_turn")
    assert script.names().count("close") == 2


def test_ci4_a_touch_on_a_clean_candidate_does_not_refuse_marginal_ones(monkeypatch):
    ctx = TR.make_ctx()
    ctx.scene.hf = kernel.HeightField.empty(ctx.view.position[:2], half_extent_m=0.2)
    first = TR.gated_plan(ctx.view)
    second = TR.gated_plan(ctx.view, marginal=True, yaw=first.yaw_rad + YAW_GRID_RAD)
    abort = {"aborted": "early_finger_contact", "guard": TR.touch_guard(), "close_grip_z": 0.944}
    script = TR.GraspScript(monkeypatch, plans=[first, second], classes=["expected"] * 2,
                            closes=[abort, (True, False)])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "held_for_turn")
    assert script.names().count("close") == 2


# ------------------------------------------------------------------------------------------------ V5-U
def _unmeasured(record):
    return dataclasses.replace(record, width_by_yaw=(), elongation=None)


def test_v5u_a_correct_appendage_point_passes_the_any_line_triage_not_the_incidental_jaw():
    """t2 D (-51 deg), t8 B (-90 / -45 deg): the point on a thin appendage beside its body is rejected at a jaw
    whose finger column lands on the body; some jaw line clears both columns -> deferred to the planned jaw."""
    hf, c, d = TP.thin_part_scene(0.002)
    rec = _unmeasured(c)
    incidental = pl.check_point_answer("grasp_point", (256, 200), rec, hf, TP.inputs(), jaw_yaw=math.pi / 2)
    assert not incidental["ok"]                                             # the frozen fallback's verdict
    out = pl.check_point_any_jaw(rec, hf, TP.inputs(), uv=(256, 200))
    assert out["ok"] and out["deferred"] and out["reason"] is None
    v = out["values"]
    assert v["lines_total"] == int(round(math.pi / YAW_GRID_RAD)) == 12 and 0 < v["lines_clear"] < 12
    assert v["cone_lines"] >= pl.V5U_CONE_LINES == 3 == v["cone_needed"]
    assert v["jaw"] == "any_line" and abs(math.sin(math.radians(v["best_line_deg"]))) < 0.5   # across the part
    # every line's verdict is exactly V5 at that jaw (the triage invents no new rule)
    for deg, ok, _ in v["lines"]:
        assert ok == pl.check_point_answer("grasp_point", (256, 200), rec, hf, TP.inputs(),
                                           jaw_yaw=math.radians(deg))["ok"]


def _stub_scene():
    """A 30 mm thin stub (13 mm wide, top 1.0345) protruding along +x from a level body of radius 60 mm (top 1.0331):
    the handle-root / spout geometry of the recorded truth-off pointings (t8 A at the handle root, live smoke 003 D on
    the spout)."""
    hf = TP.field()
    TP.stamp(hf, TP.disk((0.0, 0.0), 0.06), 1.0331)
    stub = kernel.rect_polygon((0.075, 0.0), (1.0, 0.0), 0.015, 0.0065)
    cells = TP.stamp(hf, stub, 1.0345)
    return hf, stub, cells


@pytest.mark.parametrize("x, ok, cone", [(0.066, False, 1),        # 6 mm from the body: one jaw line clears
                                         (0.075, True, 3),         # mid stub: the best line and both neighbours
                                         (0.087, True, 5)])
def test_v5u_needs_a_jaw_cone_not_a_single_line(x, ok, cone):
    hf, stub, cells = _stub_scene()
    rec = _unmeasured(TP.part("D", xyz=(x, 0.0, 1.0345), top=1.0345, sigma_h=0.001, region=stub, cells=cells))
    out = pl.check_point_any_jaw(rec, hf, TP.inputs(), uv=(1, 1))
    assert out["values"]["cone_lines"] == cone and out["ok"] is ok and out["deferred"] is ok
    assert out["values"]["lines_clear"] >= 1                               # any-line triage would pass all three
    if not ok:
        text = R._check_text(out)
        assert text.startswith("pointed_not_grasp_point: the pointed point stands above both finger landings along 1 ")
        assert "needs 3" in text


@pytest.mark.parametrize("xyz, top", [((0.0, 0.0826, 1.0331), 1.0331),          # the level body top
                                       ((0.08, -0.05, TP.SUPPORT), TP.SUPPORT)])  # the bare support
def test_v5u_a_point_no_jaw_line_clears_is_still_rejected_and_never_deferred(xyz, top):
    hf, c, d = TP.thin_part_scene(0.002)
    wrong = TP.part("C", xyz=xyz, top=top, sigma_h=0.002)
    out = pl.check_point_any_jaw(wrong, hf, TP.inputs(), uv=(1, 1))
    assert not out["ok"] and not out["deferred"] and out["reason"] == "pointed_not_grasp_point"
    assert out["values"]["lines_clear"] == 0


def _non_elongated_round(monkeypatch):
    ctx = TR.reask_ctx(monkeypatch, registered_cells=TR.LOOP_CELLS)
    rec = dataclasses.replace(TR.cells_record("D", (0.03, -0.01, 1.04), TR.LOOP_CELLS), elongation=None,
                              axis_check=None)
    assert not rec.is_elongated()
    monkeypatch.setattr(R, "_probe_and_register", TR.scripted(lambda ctx, pid, seed, label: {
        "record": rec, "failure": None, "label": label, "probe": {}}))
    return ctx


def test_v5u_wiring_only_without_a_candidate_jaw(monkeypatch):
    seen = {"any": 0, "at": []}

    def any_jaw(record, hf, inputs, uv=None):
        seen["any"] += 1
        return {"ok": True, "reason": None, "values": {"jaw": "any_line"}, "thresholds": [], "deferred": True}

    def at_jaw(kind, uv, record, hf, inputs, **kw):
        seen["at"].append(kw.get("jaw_yaw"))
        return {"ok": True, "reason": None, "values": {}, "thresholds": []}
    # no candidate jaw (width / axis unmeasured): triage, deferral remembered
    ctx = _non_elongated_round(monkeypatch)
    monkeypatch.setattr(R, "_check_jaw", lambda ctx, record, **kw: (None, "geometry_unmeasured"))
    monkeypatch.setattr(R.planner, "check_point_any_jaw", any_jaw)
    monkeypatch.setattr(R.planner, "check_point_answer", at_jaw)
    out = TR.drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
                   points=lambda need: TR.answer(need, (262.0, 302.0)))
    assert out["record"] is not None and seen == {"any": 1, "at": []}
    assert ctx.memory["v5_deferred"]["D"]["uv"] == [262.0, 302.0]
    # no candidate jaw for another reason (CJ-1 found no representative rejection, lb smoke 2137 / 2225): triage too
    seen.update(any=0, at=[])
    ctx = _non_elongated_round(monkeypatch)
    monkeypatch.setattr(R, "_check_jaw", lambda ctx, record, **kw: (None, "footprint_blocked"))
    monkeypatch.setattr(R.planner, "check_point_any_jaw", any_jaw)
    monkeypatch.setattr(R.planner, "check_point_answer", at_jaw)
    out = TR.drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
                   points=lambda need: TR.answer(need, (262.0, 302.0)))
    assert out["record"] is not None and seen == {"any": 1, "at": []}
    # negative: a planned jaw exists -> V5 at that jaw, no triage, no deferral
    seen.update(any=0, at=[])
    ctx = _non_elongated_round(monkeypatch)
    monkeypatch.setattr(R, "_check_jaw", lambda ctx, record, **kw: (0.3, "planned"))
    monkeypatch.setattr(R.planner, "check_point_any_jaw", any_jaw)
    monkeypatch.setattr(R.planner, "check_point_answer", at_jaw)
    out = TR.drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
                   points=lambda need: TR.answer(need, (262.0, 302.0)))
    assert out["record"] is not None and seen["any"] == 0 and seen["at"] == [0.3]
    assert "D" not in (ctx.memory.get("v5_deferred") or {})


@pytest.mark.parametrize("ok", [False, True])
def test_v5u_the_deferred_point_is_judged_at_the_planned_jaw_before_any_motion(monkeypatch, ok):
    plan = TR.contact_plan(yaw=0.4)
    script = TR.GraspScript(monkeypatch, plans=[plan], classes=["expected"])
    seen = []

    def at_jaw(kind, uv, record, hf, inputs, **kw):
        seen.append((kind, list(uv), kw.get("jaw_yaw")))
        return {"ok": ok, "reason": None if ok else "pointed_not_grasp_point", "values": {}, "thresholds": []}
    monkeypatch.setattr(R.planner, "check_point_answer", at_jaw)
    monkeypatch.setattr(R, "_inputs", lambda ctx, **kw: None)
    ctx = TR.make_ctx()
    ctx.memory["v5_deferred"] = {"A": {"step": 0, "uv": [260.0, 300.0]}}
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert seen == [("grasp_point", [260.0, 300.0], plan.yaw_rad)]
    assert "A" not in ctx.memory["v5_deferred"]
    if ok:
        assert outcome.status == "succeeded" and "close" in script.names()
    else:
        assert (outcome.status, outcome.reason) == ("failed", "pointed_not_grasp_point")
        assert "orient" not in script.names() and "close" not in script.names()
        assert "fails the grasp-point check at the planned jaw" in outcome.evidence_text


def test_v5u_an_unmeasured_plan_after_a_deferred_point_says_width_axis_unmeasured(monkeypatch):
    TR.GraspScript(monkeypatch, plans=[TR.infeasible("A", "geometry_unmeasured", refine_axis=True)], classes=[])
    ctx = TR.make_ctx()
    ctx.memory["v5_deferred"] = {"A": {"step": 0, "uv": [260.0, 300.0]}}
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "geometry_unmeasured")
    assert "width/axis unmeasured" in outcome.evidence_text
