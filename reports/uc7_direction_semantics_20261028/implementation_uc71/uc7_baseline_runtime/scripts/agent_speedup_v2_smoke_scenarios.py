"""BR relation-smoke scenarios for ``scripts/agent_speedup_v2_relation_smoke.py`` (contract §C14.5; TEST TOOL).

Every scenario drives the real relation controllers (``agent_speedup_v2_relations``) through the policy's §C3.4 loop
on the real simulator with the ORACLE behind the offline scripted transport (zero model calls), and reports
diagnosis-only truth read live through ``sim.oracle(...)`` (never fed back to control). The harness imports this file
only for a scenario other than its own ``oracle_selfcheck``.

Scenarios (``SCENARIOS``):
- ``grasp_lever_t2``  calibrate -> plan -> register -> grasp(A, purpose turn) [--depth-rule] -> release;
- ``grasp_loop_t2``   ... grasp(C, obj D, purpose carry) -> hold 20 steps -> release;
- ``grasp_body_t2``   ... grasp(D, obj D, purpose carry) [--forced-yaw jaw_y | jaw_x];
- ``turn_t2``         ... turn(plan step 0) [--first-sense +1 | -1, contract sense about R2];
- ``endstop_t2``      ... grasp(A) -> TwistSegment(-1) -> TwistSegment(+1, after_reversal) -> HingeTwist(+1, end-stop
                      mode) -> release [C-41];
- ``place_t2``        ... grasp(C, obj D, carry) -> place_on(plan step 1);
- ``survey_t2``       calibrate -> plan -> register, then the empty-hand survey hovers of BR's ``survey_hover`` over
                      the lever A, the loop C and the destination B; every start and survey record's top height is
                      scored against the evaluator height at its point (diagnosis; the spec G1 protocol with 4 hover
                      offsets on episodes 1-5 is the scene owner's ``agent_speedup_v2_survey_probe.py``);
- ``chain_t2``        the full StepRunner exactly as the policy builds it (its default episode generator, with the
                      fulfiller's call counts); passed = raw LIBERO success AND physical_truth.v2_success (§C14.5, AC-2:
                      terminal truth plus the release-and-settle counterfactual).

Blocker fixes (2026-10-19, version 2): the grasp scenarios report the grip part's pointing path (``pointing_report``:
the BR-1 hover reference, the BP-1 / BR-2 framing, the BR-3 re-asks, the BR-4 escalation, BR-5's refinements) and
``turn_t2`` the effect watch's reference, the O7 remainders per phase with the scene's hand-mask rule and the
post-release confirmation (blocker-1 fix (c)).

usage: env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python scripts/agent_speedup_v2_relation_smoke.py \\
           --task libero_10_t2 --episode-index 0 --scenario turn_t2 --first-sense +1 --max-steps 700
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from gpt_field import agent_speedup_v2_geometry as geometry  # noqa: E402
from gpt_field import agent_speedup_v2_observers as observers  # noqa: E402
from gpt_field import agent_speedup_v2_primitives as primitives  # noqa: E402
from gpt_field import agent_speedup_v2_relations as relations  # noqa: E402
from gpt_field.agent_speedup_v2_plan import part_roles  # noqa: E402
from gpt_field.agent_speedup_v2_truth import physical_summary  # noqa: E402
from gpt_field.agent_speedup_v2_types import (FINGERTIP_EDGES, PlanAnswer, PlanNeed, PartRecord,  # noqa: E402
                                              StepContext, jsonable)
from gpt_field.contact_skills import HoldPrimitive  # noqa: E402

SCENARIOS_VERSION = "agent-speedup-v2-smoke-scenarios-2"
KNOB_JOINT = "flat_stove_1_button"
POT_BODY = "moka_pot_1_main"
Q_ON = 0.5                                  # TurnOn threshold of the task-2 knob (flat_stove.xml, diagnosis only)
HOLD_STEPS = 20
# ORACLE plan ids of task 2 and their true top heights (spec §7.1 [F], contract §C5 survey_t2; diagnosis only):
# the grip parts are measured by BR's pointing round at their grasp hover (_grip_hover: point -> ParallaxProbe ->
# register), the destination and the lower-bound object by BR's empty-hand survey_hover
SURVEY_GRIP = (("A", "turn", "A"), ("C", "carry", "D"))
SURVEY_HOVER = (("B", "destination"), ("D", "lower_bound_object"))
SURVEY_TRUTH_TOP_M = {"A": 0.955, "C": 1.0345, "B": 0.9305, "D": 1.05}
SURVEY_DH_MAX_M = 0.005                     # spec G1 p90 bound on |dh|, applied per record in one run (diagnosis)
SURVEY_SCORED = ("A", "C", "B")             # D's top is approximate (about 1.05 m at the image edge): reported only
FORCED_YAWS = {"jaw_y": math.pi / 2.0, "jaw_x": 0.0}


# ------------------------------------------------------------------------------------------------- truth
def knob_q(sim):
    try:
        return float(sim.oracle({"kind": "joints", "names": [KNOB_JOINT]})["q"][KNOB_JOINT])
    except Exception:  # noqa: BLE001 - diagnosis only
        return None


def body_pose(sim, name=POT_BODY):
    try:
        return sim.oracle({"kind": "bodies", "names": [name]})["poses"][name]
    except Exception:  # noqa: BLE001 - diagnosis only
        return None


def tilt_deg(quat_xyzw):
    x, y, _, _ = quat_xyzw
    zz = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def _relations_of(outcomes):
    return [(o.get("relation"), o.get("status"), o.get("reason")) for o in outcomes]


def _last(outcomes, relation):
    rows = [o for o in outcomes if o.get("relation") == relation]
    return rows[-1] if rows else None


def pointing_report(held):
    """Blocker-2 diagnosis of a grasp outcome: the hover reference (BR-1 coincidence), the planned and arrived
    framing (BP-1 / BR-2), the re-asks (BR-3), the escalation (BR-4), the failed-round memory and BR-5's cell
    refinements."""
    ev = held.get("evidence") or {}
    hover = ev.get("hover") or {}
    plan = ev.get("plan") or {}
    return {"reference": hover.get("reference"), "planned_framing": (hover.get("hover") or {}).get("framing"),
            "arrived_framing": hover.get("framing"), "reasks": hover.get("reasks"), "rejected": hover.get("rejected"),
            "escalation": hover.get("escalation"), "memory": ev.get("pointing"), "refine_cells": ev.get("refine_cells"),
            "last_check": hover.get("check"), "two_stage": hover.get("two_stage"), "br1": hover.get("br1"),
            "plan_refusal": {k: plan.get(k) for k in ("infeasible", "rejections", "refine_height", "refine_cells",
                                                      "blocking_cells", "lower_bound_blockers")} if plan else None}


# --------------------------------------------------------------------------------------------- episodes
def plan_and_register(ctx):
    """Calibrate+, one plan request (the ORACLE plan), merge_plan with that request's view and k, register_regions."""
    yield from relations.calibrate_plus(ctx)
    k = int(geometry.frame_alignment_turns(ctx.view.R, ctx.cal.external))
    answer = yield PlanNeed("plan", k, FINGERTIP_EDGES[k])
    if not isinstance(answer, PlanAnswer) or answer.plan is None:
        raise RuntimeError(f"the ORACLE plan did not parse: {getattr(answer, 'parse_error', None)}")
    plan = answer.plan
    roles = part_roles(plan)
    ctx.plan, ctx.roles = plan, roles
    ctx.scene.merge_plan(plan, roles, answer.view, answer.need.quarter_turns)
    yield from relations.register_regions(ctx, plan, roles)
    ctx.monitor.clear("relation")
    return plan, roles


def hold_forever(ctx):
    while True:
        gripper = "close" if ctx.holding is not None else "open"
        yield (lambda v, g=gripper: HoldPrimitive(v, steps=HOLD_STEPS, gripper=g, name="smoke_hold"))


def step_context(plan, i):
    return StepContext.from_step(i, plan.steps[i], attempt=1)


class Scenario:
    """Base: a policy scenario whose generator runs ``episode(ctx, args)`` then holds forever."""
    uses_policy = True
    name = "base"

    def overrides(self, args) -> dict:
        out = {}
        if getattr(args, "first_sense_value", None) is not None:
            out["first_sense"] = int(args.first_sense_value)
        if getattr(args, "depth_rule", "auto") in ("touch_stop", "height_stop"):
            out["depth_rule"] = args.depth_rule
        if getattr(args, "forced_yaw", "none") in FORCED_YAWS:
            out["forced_yaw_rad"] = FORCED_YAWS[args.forced_yaw]
        return out

    def factory(self, args):
        def make(ctx):
            def episode():
                self.state = {"q0": None}
                yield from self.episode(ctx, args)
                yield from hold_forever(ctx)
            return episode()
        return make

    def episode(self, ctx, args):
        raise NotImplementedError
        yield

    def report(self, sim, policy, outcomes, args) -> dict:
        return {"scenario": self.name, "outcomes": _relations_of(outcomes), "knob_q": knob_q(sim),
                "pot": body_pose(sim)}


class GraspLever(Scenario):
    name = "grasp_lever_t2"

    def episode(self, ctx, args):
        yield from plan_and_register(ctx)
        outcome = yield from relations.grasp(ctx, "A", obj_pid="A", purpose="turn")
        self.state["held"] = outcome.reason
        if outcome.status == "succeeded":
            yield (lambda v: HoldPrimitive(v, steps=HOLD_STEPS, gripper="close", name="smoke_hold_closed"))
            yield from relations.release(ctx, None)

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        held = _last(outcomes, "held")
        out["held"] = None if held is None else {k: held.get(k) for k in ("status", "reason", "evidence_text")}
        if held is not None:
            attempts = (held.get("evidence") or {}).get("attempts") or []
            out["attempts"] = [{k: a.get(k) for k in ("plan", "contact", "close", "width", "align")} for a in attempts]
            out["pointing"] = pointing_report(held)
        out["passed"] = bool(held and held.get("status") == "succeeded")
        return jsonable(out)


class GraspLoop(Scenario):
    name = "grasp_loop_t2"

    def episode(self, ctx, args):
        yield from plan_and_register(ctx)
        self.state["pot_before"] = None
        outcome = yield from relations.grasp(ctx, "C", obj_pid="D", purpose="carry")
        if outcome.status == "succeeded":
            yield (lambda v: HoldPrimitive(v, steps=HOLD_STEPS, gripper="close", name="smoke_hold_closed"))
            yield from relations.release(ctx, None)

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        held = _last(outcomes, "held")
        out["held"] = None if held is None else {k: held.get(k) for k in ("status", "reason", "evidence_text")}
        out["pointing"] = None if held is None else pointing_report(held)
        pose = body_pose(sim)
        out["pot_tilt_deg"] = None if pose is None else tilt_deg(pose["quat_xyzw"])
        out["passed"] = bool(held and held.get("status") == "succeeded" and held.get("reason") == "hold_pass")
        return jsonable(out)


class GraspBody(Scenario):
    name = "grasp_body_t2"

    def episode(self, ctx, args):
        yield from plan_and_register(ctx)
        yield from relations.grasp(ctx, "D", obj_pid="D", purpose="carry")

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        held = _last(outcomes, "held")
        out["held"] = None if held is None else {k: held.get(k) for k in ("status", "reason", "evidence_text")}
        out["forced_yaw"] = getattr(args, "forced_yaw", None)
        return jsonable(out)


class Turn(Scenario):
    name = "turn_t2"

    def episode(self, ctx, args):
        plan, _ = yield from plan_and_register(ctx)
        yield from relations.turn(ctx, step_context(plan, 0))

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        turned = _last(outcomes, "turned")
        out["turned"] = None if turned is None else {k: turned.get(k) for k in ("status", "reason", "evidence_text")}
        if turned is not None:
            ev = turned.get("evidence") or {}
            out["segments"] = ev.get("segments")
            out["hinge"] = ev.get("hinge")
            out["o7_free"] = ev.get("o7_free")
            out["post_check"] = ev.get("post_check")
            # blocker-1 fix (c): the watch's reference, the per-phase O7 remainders with the hand-mask rule, and the
            # post-release confirmation (verdict, visible share, share, lift)
            out["effect_reference"] = ev.get("effect_reference")
            out["o7_phases"] = [{"segment": s.get("segment"), "validity": s.get("validity")}
                                for s in ev.get("segments") or []] + [{"segment": "free", "validity": ev.get("o7_free")}]
            out["effect_confirm"] = ev.get("effect_confirm")
            out["flags"] = ev.get("flags")
        q = out.get("knob_q")
        out["passed"] = bool(turned and turned.get("status") == "succeeded" and q is not None and q >= Q_ON)
        return jsonable(out)


class EndStop(Scenario):
    """[C-41]: the end stop after a reversal, which turn itself never produces on task 2."""
    name = "endstop_t2"

    def episode(self, ctx, args):
        yield from plan_and_register(ctx)
        held = yield from relations.grasp(ctx, "A", obj_pid="A", purpose="turn")
        if held.status != "succeeded":
            return
        part = ctx.scene.parts.get("A")
        first = yield from relations._twist_segment(ctx, sense=-1, part=part, effect=None, exclude=[], label="free")
        self.state["first"] = first["reason"]
        second = yield from relations._twist_segment(ctx, sense=1, part=part, effect=None, exclude=[],
                                                      after_reversal=True, blocked_steps=first["steps"],
                                                      label="reversal")
        self.state["second"] = second["reason"]
        g6 = relations._g6(ctx)
        wedge_end = observers.WedgeObserver(mode="end_stop", robot=ctx.robot,
                                            sigma_g_mm_per_step=ctx.cal.sigma_g_mm_per_step, hold_floor_mm=g6)
        budget = relations._budget_left(ctx, 1)
        result, _ = yield (lambda v: primitives.HingeTwist(
            v, sense=1, wedge_end=wedge_end, effect=None, part_rotation=None, hold_floor_mm=float(g6.value),
            theta_target_rad=None, yaw_budget_left_rad=budget, centre_prior_xy=v.position[:2], sigma_c_m=0.0,
            wedge_capable=True, robot=ctx.robot, name="endstop_hinge"))
        self.state["hinge"] = result.get("reason")
        yield from relations._turn_release(ctx)

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        out.update(state=jsonable(getattr(self, "state", {})))
        out["passed"] = getattr(self, "state", {}).get("hinge") == "end_stop"
        return jsonable(out)


class Place(Scenario):
    name = "place_t2"

    def episode(self, ctx, args):
        plan, _ = yield from plan_and_register(ctx)
        step = step_context(plan, 1)
        yield from relations.place_on(ctx, step)

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        placed = _last(outcomes, "supported_on")
        out["supported_on"] = None if placed is None else {k: placed.get(k) for k in ("status", "reason",
                                                                                      "evidence_text")}
        if placed is not None:
            out["attempts"] = (placed.get("evidence") or {}).get("attempts")
        held = _last(outcomes, "held")
        out["pointing"] = None if held is None else pointing_report(held)
        pose = body_pose(sim)
        out["pot_tilt_deg"] = None if pose is None else tilt_deg(pose["quat_xyzw"])
        out["passed"] = bool(placed and placed.get("status") == "succeeded")
        return jsonable(out)


def truth_height(sim, xy):
    """Evaluator height at a world xy (robot excluded), or None (diagnosis only)."""
    try:
        return float(sim.oracle({"kind": "heights", "xy": [[float(xy[0]), float(xy[1])]]})["z"][0])
    except Exception:  # noqa: BLE001 - diagnosis only
        return None


def record_summary(record):
    if not isinstance(record, PartRecord) or record.xyz is None:
        return None
    return {"status": record.status, "provisional": record.provisional, "partial": record.partial,
            "xyz": [float(v) for v in record.xyz], "top_z": None if record.top_z is None else float(record.top_z.value),
            "sigma_h_m": record.sigma_h_m, "lower_bound_top": record.top_is_lower_bound(),
            "elongated": record.is_elongated()}


class Survey(Scenario):
    """The start survey, BR's pointing round at the grasp hovers of the lever and the loop, and BR's empty-hand
    survey hovers of the destination and the lower-bound object; every record's top against its part's true top
    (and against the evaluator height at the record's own point, which tells a wrong surface from a wrong height)."""
    name = "survey_t2"

    def episode(self, ctx, args):
        yield from plan_and_register(ctx)
        pids = [pid for pid, _, _ in SURVEY_GRIP] + [pid for pid, _ in SURVEY_HOVER]
        self.state["start"] = {pid: record_summary(ctx.scene.parts.get(pid)) for pid in pids}
        self.state["surveys"] = {}
        for pid, purpose, obj in SURVEY_GRIP:
            if pid not in ctx.scene.parts:
                continue
            yield from relations._ensure_open(ctx)
            hovered = yield from relations._grip_hover(ctx, pid, purpose=purpose, obj=obj)
            self.state["surveys"][pid] = {"status": "succeeded" if hovered.get("record") is not None else "failed",
                                          "reason": hovered.get("reason"), "path": "grip_hover",
                                          "check": hovered.get("check"),
                                          "record": record_summary(hovered.get("record"))}
        for pid, purpose in SURVEY_HOVER:
            if pid not in ctx.scene.parts or ctx.holding is not None:
                continue
            outcome = yield from relations.survey_hover(ctx, pid, purpose=purpose)
            self.state["surveys"][pid] = {"status": outcome.status, "reason": outcome.reason, "path": "survey_hover",
                                          "record": record_summary(ctx.scene.parts.get(pid))}

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        state = getattr(self, "state", {})
        scored = []
        for phase, rows in (("start", state.get("start") or {}),
                            ("survey", {k: (v or {}).get("record") for k, v in (state.get("surveys") or {}).items()})):
            for pid, rec in rows.items():
                if not rec or rec.get("top_z") is None:
                    scored.append({"phase": phase, "pid": pid, "top_z": None})
                    continue
                at_point = truth_height(sim, rec["xyz"][:2])
                truth = SURVEY_TRUTH_TOP_M.get(pid)
                scored.append({"phase": phase, "pid": pid, "top_z": rec["top_z"], "part_top_z": truth,
                               "dh_m": None if truth is None else rec["top_z"] - truth,
                               "surface_at_point_z": at_point,
                               "dh_point_m": None if at_point is None else rec["top_z"] - at_point,
                               "sigma_h_m": rec.get("sigma_h_m"), "lower_bound_top": rec.get("lower_bound_top")})
        out.update(state=jsonable(state), heights=scored)
        rows = {r["pid"]: r for r in scored if r["phase"] == "survey"}
        out["passed"] = all(pid in rows and rows[pid].get("dh_m") is not None
                            and abs(rows[pid]["dh_m"]) <= SURVEY_DH_MAX_M for pid in SURVEY_SCORED)
        return jsonable(out)


class Chain(Scenario):
    """The full StepRunner exactly as the policy builds it: no episode factory, so the policy's default
    ``StepRunner(ctx, relations=layer, call_counts=fulfiller.counts)`` runs (per-step call accounting included)."""
    name = "chain_t2"

    def factory(self, args):
        return None

    def report(self, sim, policy, outcomes, args):
        out = super().report(sim, policy, outcomes, args)
        status = getattr(sim, "status", {}) or {}
        raw = bool(status.get("libero_success", status.get("success")))
        out["libero_success"] = raw
        terminal = settled = None
        try:
            terminal = sim.truth()
            settled = sim.settle_truth() if terminal is not None else None
        except Exception as exc:  # noqa: BLE001 - diagnosis only
            out["truth_error"] = f"{type(exc).__name__}: {exc}"
        summary = physical_summary(terminal, settled, raw_libero_success=raw)
        out["physical_truth"] = {"summary": summary, "terminal_failing_checks": summary.get("terminal_failing_checks")}
        out["passed"] = bool(raw and summary.get("v2_success"))
        return jsonable(out)


SCENARIOS = {s.name: s for s in (GraspLever(), GraspLoop(), GraspBody(), Turn(), EndStop(), Place(), Survey(),
                                 Chain())}
