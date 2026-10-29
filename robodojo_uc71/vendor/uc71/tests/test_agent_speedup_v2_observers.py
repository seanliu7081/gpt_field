"""BO T1 tests for agent_speedup_v2_observers (contract §C9 T1 items 1-6, Appendix L 8 / 12).

Test files may read evaluator state (fake joints, masks, probe q) to score observers; the observers themselves see
only RGB and state8. SP / CP probe labels are WORLD-z labels (contract §C1.3): SP "+1" = contract sense -1.
"""

from __future__ import annotations

import base64
import glob
import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import agent_speedup_v2_fakes as F
from gpt_field import agent_speedup_v2_observers as O
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field.agent_speedup_v2_types import (HandPose, HeldFrame, KeepOut, Meas, ObsReport, RobotModel,
                                              WristCamera, rect_polygon)
from gpt_field.contact_skills import StepView, matrix_to_quaternion_xyzw, rotation_matrix
from gpt_field.observation import PolicyObservation, quaternion_to_axisangle
from gpt_field.wrist_servo import NCCTrack

ROOT = Path(__file__).resolve().parents[1]
LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
ROBOT = RobotModel()
T2_START = (-0.2014, 0.0187, 1.1893, 3.1397, 0.0067, -0.0861, 0.0387, -0.0387)
PROMPT = "observer test"


# ================================================================================================ helpers
def png_url(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_BLANK = png_url(np.zeros((8, 8, 3), np.uint8))


def make_view(state8, *, external=None, wrist=None):
    """StepView from state8 and UPRIGHT images (stored in the legacy, column-mirrored orientation)."""
    ext = _BLANK if external is None else png_url(np.ascontiguousarray(np.asarray(external)[:, ::-1]))
    wr = _BLANK if wrist is None else png_url(np.ascontiguousarray(np.asarray(wrist)[:, ::-1]))
    return StepView(PolicyObservation(ext, wr, tuple(float(v) for v in state8), PROMPT))


def rotz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_of(R):
    return np.asarray(quaternion_to_axisangle(matrix_to_quaternion_xyzw(np.asarray(R, float))))


def state_with(R, gap_mm, grip=T2_START[:3]):
    return (*grip, *rotvec_of(R), gap_mm / 2000.0, -gap_mm / 2000.0)


def trace_views(start_gap, trace, R0):
    """Views of an SP / CP closed-twist trace: world-z yaw increments (the probes' rotation) and finger gaps."""
    R = np.asarray(R0, float)
    views = [make_view(state_with(R, start_gap))]
    for row in trace:
        R = rotz(math.radians(row["dyaw_deg"])) @ R
        views.append(make_view(state_with(R, row["gap_mm"])))
    return views, R


def feed(observer, views, context=None):
    reports = []
    for before, after in zip(views[:-1], views[1:]):
        reports.append(observer.update(before, None, after, context))
    return reports


def strict_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def v1_hand_mask(before, after):
    """Stand-in for geometry.hand_motion_mask (v1 external_axes.hand_image_motion's mask), so the observer tests do
    not depend on the geometry module (contract §C2.2 Wave-1 parallelism)."""
    g0 = cv2.cvtColor(np.ascontiguousarray(before), cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(np.ascontiguousarray(after), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 21, 5, 7, 1.5, 0)
    changed = np.abs(g0.astype(int) - g1.astype(int)) > 12
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    mag = np.linalg.norm(flow, axis=2)
    sel = changed & (mag > 1.0)
    if int(sel.sum()) < 50:
        return None
    return sel & (mag >= float(np.quantile(mag[sel], 0.9)))


def no_hand(before, after):
    return None


def wedge(mode="sense", **kw):
    kw.setdefault("sigma_g_mm_per_step", 0.002)
    kw.setdefault("hold_floor_mm", 5.0)
    return O.WedgeObserver(mode=mode, robot=ROBOT, **kw)


SP = F.load_probe("sp_knob_sense_probe")
CP = F.load_probe("cp_knob_twist_probe")
R_T2 = rotation_matrix(np.asarray(T2_START[3:6]))


# =========================================================================== 1. WedgeObserver: SP replay, CP check
def test_sp_fresh_free_segment_decides_free_at_step_12():
    """SP "+1-1" first segment = contract -1 from a fresh close (the free sense): free_sense at k = 11."""
    order = SP["order_+1-1"]
    views, _ = trace_views(order["post_close_gap"], order["first"], R_T2)
    obs = wedge()
    obs.start(views[0])
    reports = feed(obs, views)
    assert [r.verdict for r in reports[:11]] == ["pending"] * 11
    final = reports[11]
    assert (final.verdict, final.reason) == ("met", "free_sense")
    ev = final.evidence
    assert ev["k"] == 11 and ev["decided_at"] == 11 and ev["k0"] == 1 and ev["class"] == "free"
    assert ev["early"] == pytest.approx(0.2605, abs=5e-4)
    assert ev["late"] == pytest.approx(0.0445, abs=5e-4)
    free = next(t for t in final.thresholds if t.name == "K1w_free_rate_mm")
    assert free.value == pytest.approx(0.25 * 0.2605, abs=1e-3)                 # max(0.032, 4 sigma_g, 0.065)
    print("SP fresh free: early %.4f late %.4f, %s" % (ev["early"], ev["late"], free.derivation))
    assert "K1W_FREE_LW*l*omega=0.03" in free.derivation                       # 0.1 l w ~ 0.032 (spec K1w)
    strict_json(final.to_dict())


def test_sp_fresh_blocked_segment_decides_blocked_at_step_12():
    """SP "-1+1" first = contract +1 (blocked): windows 0.42 / 0.445 >= max(0.5 l w ~ 0.16, 0.75 early = 0.227)."""
    order = SP["order_-1+1"]
    views, _ = trace_views(order["post_close_gap"], order["first"], R_T2)
    obs = wedge()
    obs.start(views[0])
    reports = feed(obs, views)
    assert all(r.verdict == "pending" for r in reports[:11])
    final = reports[11]
    assert (final.verdict, final.reason) == ("violated", "blocked_sense")
    window = final.evidence["window"]
    assert final.evidence["late"] == pytest.approx(0.4445, abs=5e-4)
    assert window["late_prev"] == pytest.approx(0.42075, abs=5e-4)
    blocked = next(t for t in final.thresholds if t.name == "K1w_blocked_rate_mm")
    assert blocked.value == pytest.approx(0.75 * 0.30275, abs=1e-3)               # 0.227 wins over 0.5 l w
    lw = final.evidence["lw"]
    assert 0.29 < lw < 0.34 and 0.5 * lw == pytest.approx(0.16, abs=0.01)         # spec: 0.5 l w ~ 0.16
    print("SP fresh blocked: late %.4f / %.4f, %s" % (final.evidence["late"], window["late_prev"], blocked.derivation))


def test_sp_reversal_unwedges_then_decides_free_at_k24():
    """SP "-1+1" second = contract -1 after a reversal of a 12-step blocked segment: unwedge 13 steps (<= 18),
    k0 = 13, free_sense at k = 24."""
    order = SP["order_-1+1"]
    v1, R = trace_views(order["post_close_gap"], order["first"], R_T2)
    views, _ = trace_views(order["first"][-1]["gap_mm"], order["second"], R)
    obs = wedge(after_reversal=True, blocked_steps=12)
    obs.start(views[0])
    reports = feed(obs, views)
    first_met = next(i for i, r in enumerate(reports) if r.verdict != "pending")
    final = reports[first_met]
    assert first_met == 24 and (final.verdict, final.reason) == ("met", "free_sense")
    assert final.evidence["k0"] == 13 and final.evidence["unwedge_steps"] == 13
    assert final.evidence["unwedge_limit"] == 18
    assert any(t.name == "K1w_unwedge_steps" and t.value == 18 for t in final.thresholds)
    assert not any(r.reason == "turn_slip" for r in reports)                    # the fall is the unwedge, not a slip


def test_sp_reversal_free_then_end_stop_within_5_steps_of_the_limit():
    """SP "+1-1" second = contract +1 after a reversal: k0 = 3, free_sense at k = 14; then K4w end_stop mode from
    k = 15: end_stop at k in {24, 25}, i.e. <= 5 steps after q < -0.004 at k = 22."""
    order = SP["order_+1-1"]
    _v, R = trace_views(order["post_close_gap"], order["first"], R_T2)
    second = order["second"]
    views, _ = trace_views(order["first"][-1]["gap_mm"], second, R)
    obs = wedge(after_reversal=True, blocked_steps=12)
    obs.start(views[0])
    reports = feed(obs, views[:16])
    decided = next(i for i, r in enumerate(reports) if r.verdict != "pending")
    assert decided == 14 and reports[14].reason == "free_sense" and reports[14].evidence["k0"] == 3
    stop = wedge(mode="end_stop")
    stop.start(views[15])                                                      # the view after step k = 14
    stop_reports = feed(stop, views[15:])
    met = next(i for i, r in enumerate(stop_reports) if r.verdict == "met")
    k_met = 15 + met
    limit_k = next(r["k"] for r in second if r["q"] < -0.004)
    assert limit_k == 22 and k_met in (24, 25) and k_met - limit_k <= 5
    assert stop_reports[met].reason == "end_stop"
    assert all(r.verdict == "pending" for r in stop_reports[:met])
    print("K4w end stop at k =", k_met, stop_reports[met].thresholds[-1].derivation)


def test_end_stop_fires_after_a_short_steady_phase():
    """K4w's median is the steady rate BEFORE the candidate windows: with only 4 steady steps before the limit (the
    fake's post-reversal case, BM finding) the stop still fires within 5 steps; a steady twist never fires."""
    R = R_T2
    yaw = math.radians(1.15)
    rates = [0.003, 0.002, 0.003, 0.002] + [0.18, 0.49, 0.44, 0.40, 0.45, 0.42]
    gaps = list(np.cumsum([26.0] + rates))
    views, Rk = [make_view(state_with(R, gaps[0]))], R
    for g in gaps[1:]:
        Rk = rotz(yaw) @ Rk
        views.append(make_view(state_with(Rk, g)))
    stop = wedge(mode="end_stop")
    stop.start(views[0])
    reps = feed(stop, views)
    met = next(i for i, r in enumerate(reps) if r.verdict == "met")
    assert 4 < met <= 4 + 5 and reps[met].reason == "end_stop"                   # the limit is at k = 4
    assert reps[met].evidence["window"]["median_rates"] >= 1
    steady = wedge(mode="end_stop")
    flat = [make_view(state_with(rotz(yaw * i) @ R, 26.0 + 0.003 * i)) for i in range(20)]
    steady.start(flat[0])
    assert all(r.verdict == "pending" for r in feed(steady, flat))


def _noisy(trace, rng, sigma):
    return [dict(row, gap_mm=row["gap_mm"] + float(rng.normal(0.0, sigma))) for row in trace]


@pytest.mark.parametrize("seed", [None] + list(range(10)))
def test_cp_traces_decide_the_same_senses_non_circular(seed):
    """CP (held out from every fit): probe_minus (world -1 = contract +1) from the fresh close is blocked; the
    reversal probe_plus continued by twist_cap0.1 (world +1 = contract -1, one continuous recording) unwedges and is
    free; also with +-0.02 mm Gaussian gap noise (10 seeds; sigma_g = the noise's rate std)."""
    logs = CP["logs"]
    noise = 0.0 if seed is None else 0.02
    rng = np.random.default_rng(seed)
    sigma_g = 0.002 if seed is None else noise * math.sqrt(2.0)
    minus = logs["probe_minus"] if seed is None else _noisy(logs["probe_minus"], rng, noise)
    views, R = trace_views(CP["post_close"]["gap_mm"], minus, R_T2)
    obs = wedge(sigma_g_mm_per_step=sigma_g)
    obs.start(views[0])
    reports = feed(obs, views)
    assert all(r.verdict == "pending" for r in reports)                         # 10 steps < the decision step
    now = obs.classify_now()
    assert now["class"] == "blocked", now
    plus_trace = logs["probe_plus"] + logs["twist_cap0.1"]
    assert abs(logs["probe_plus"][-1]["q"] - logs["twist_cap0.1"][0]["q"]) < 0.01      # continuous recording
    plus = plus_trace if seed is None else _noisy(plus_trace, rng, noise)
    views2, _ = trace_views(minus[-1]["gap_mm"], plus, R)
    rev = wedge(after_reversal=True, blocked_steps=len(minus), sigma_g_mm_per_step=sigma_g)
    rev.start(views2[0])
    reps = feed(rev, views2)
    decided = [r for r in reps if r.verdict != "pending"]
    assert decided and (decided[0].verdict, decided[0].reason) == ("met", "free_sense"), decided[:1]
    assert decided[0].evidence["k0"] in (10, 11, 12)
    assert not any(r.reason in ("blocked_sense", "turn_slip", "hold_lost") for r in reps)


def test_wedge_hold_lost_slip_unwedge_limit_and_undecided():
    R = R_T2
    yaw = math.radians(1.2)

    def views_of(gaps):
        out, Rk = [make_view(state_with(R, gaps[0]))], R
        for g in gaps[1:]:
            Rk = rotz(yaw) @ Rk
            out.append(make_view(state_with(Rk, g)))
        return out
    # hold floor
    obs = wedge(hold_floor_mm=20.0)
    vs = views_of([25.0, 25.1, 19.0])
    obs.start(vs[0])
    assert feed(obs, vs)[-1].reason == "hold_lost"
    # slip: two falls <= -0.5 l w (l w ~ 0.33 mm/step) after the first rate
    obs = wedge()
    vs = views_of([25.0, 25.3, 25.5, 25.2, 24.9])
    obs.start(vs[0])
    reps = feed(obs, vs)
    assert reps[2].verdict == "pending" and (reps[3].verdict, reps[3].reason) == ("violated", "turn_slip")
    # unwedge longer than ceil(1.5 x 4) = 6 steps after a 4-step blocked segment
    obs = wedge(after_reversal=True, blocked_steps=4)
    vs = views_of([28.0 - 0.3 * i for i in range(9)])
    obs.start(vs[0])
    reps = feed(obs, vs)
    assert (reps[6].verdict, reps[6].reason) == ("failed", "sense_undecided")
    assert reps[6].evidence["class"] == "unwedge_limit" and all(r.verdict == "pending" for r in reps[:6])
    # neither free nor blocked at steps 12 and 16: pending, class "undecided" at the extend step
    obs = wedge()
    rates = [0.1, 0.3, 0.3, 0.3, 0.3, 0.25, 0.22, 0.2, 0.18, 0.16, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    vs = views_of(list(np.cumsum([25.0] + rates)))
    obs.start(vs[0])
    reps = feed(obs, vs)
    assert reps[11].verdict == "pending" and reps[11].evidence["class"] == "pending"
    assert reps[15].verdict == "pending" and reps[15].evidence["class"] == "undecided"
    # gap noise with a still hand never reads as a slip (the 4 sigma_g floor)
    obs = wedge(sigma_g_mm_per_step=0.03)
    still = [make_view(state_with(R, 25.0 + d)) for d in (0.0, -0.05, -0.1, -0.06, -0.11)]
    obs.start(still[0])
    assert all(r.reason != "turn_slip" for r in feed(obs, still))


def test_wedge_stalled_hand_is_never_free():
    """BO-R3: a blocked twist whose hand stops yawing (wedge-locked / joint limit): the late rate falls with the
    hand, which the free test alone reads as free. The stall guard (late omega < 0.5 x early omega) gives neither
    class: pending at the decision step, 'undecided' at the extend step, window flag hand_stalled."""
    R = R_T2
    yaws = [1.1, 2.1, 1.8, 1.3, 0.9, 0.5, 0.3, 0.1] + [0.003] * 10                 # the real probe's stall profile
    rates = [0.6, 0.55, 0.37, 0.31, 0.12, 0.15, 0.02, 0.01] + [0.003] * 10
    views, Rk, gap = [make_view(state_with(R, 25.0))], R, 25.0
    for yaw, rate in zip(yaws, rates):
        Rk = rotz(math.radians(yaw)) @ Rk
        gap += rate
        views.append(make_view(state_with(Rk, gap)))
    obs = wedge(sigma_g_mm_per_step=0.003)
    obs.start(views[0])
    reps = feed(obs, views)
    assert not any(r.reason == "free_sense" for r in reps), [r.reason for r in reps]
    assert reps[11].verdict == "pending" and reps[11].evidence["window"]["hand_stalled"] is True
    assert reps[15].evidence["class"] == "undecided" and reps[15].evidence["window"]["hand_stalled"] is True
    stall = next(t for t in reps[15].thresholds if t.name == "K1w_late_omega_rad")
    assert stall.value == pytest.approx(0.5 * reps[15].evidence["window"]["omega_early"])


def test_wedge_hold_floor_threshold_label_is_carried():
    """BO-R8: the G6 floor passed as primitives.g6_hold_floor's Threshold keeps its winner label in the reports."""
    floor = th.Threshold("G6_hold_floor_mm", 5.0, "DD", "empty floor 4.5 mm + max(G6_FLOOR_MM=0.5, ...) [DD]")
    obs = O.WedgeObserver(mode="sense", robot=ROBOT, sigma_g_mm_per_step=0.002, hold_floor_mm=floor)
    views, _ = trace_views(SP["order_+1-1"]["post_close_gap"], SP["order_+1-1"]["first"], R_T2)
    obs.start(views[0])
    rep = feed(obs, views)[11]
    assert rep.reason == "free_sense" and rep.thresholds[0] is floor
    number = wedge(hold_floor_mm=5.0)
    number.start(views[0])
    assert feed(number, views[:2])[0].thresholds[0].label == "ONLINE"


def test_real_blocked_probe_with_a_stalled_hand_is_not_free():
    """BO-R3 on the real task-2 blocked recording (the M0 probe's increments; the hand stalls after k = 8 while the
    knob stays at its limit): never free_sense within the 16-step segment."""
    views, steps = _run_views(_latest_run("agent_speedup_v2_w1_bo_review_blocked_frames_*"), limit=16)
    obs = wedge(sigma_g_mm_per_step=0.003)
    obs.start(views[0])
    reps = feed(obs, views)
    assert not any(r.reason == "free_sense" for r in reps), [r.reason for r in reps]
    assert reps[11].evidence["window"].get("hand_stalled") is True


# ================================================================================ 2. PartRotationObserver (fakes)
_TWIST_CACHE = {}


def twist_frames(kind, sense, steps=16):
    """Rendered views of a scripted twist of the gripped rotary part (cached per scene and sense)."""
    key = (kind, sense, steps)
    if key not in _TWIST_CACHE:
        scene = {"lever": F.rotary_part_scene, "round": lambda: F.rotary_part_scene(lever=False),
                 "dark": F.dark_rotary_part_scene}[kind]()
        w = F.FakeTwoViewSim(scene, eef=scene.info["start_eef"])
        F.scripted_grasp_rotary(w)
        views, q = [StepView(w.observation())], [w.joint("knob")]
        for _ in range(steps):
            F.scripted_twist(w, sense, 1, close_first=False, render=True)
            views.append(StepView(w.observation()))
            q.append(w.joint("knob"))
        _TWIST_CACHE[key] = (views, q)
    return _TWIST_CACHE[key]


def rotation_reports(kind, sense, uv, radius=None, axis_uv=None):
    views, q = twist_frames(kind, sense)
    patch = {"uv": uv} if radius is None else {"uv": uv, "radius_px": radius}
    obs = O.PartRotationObserver(robot=ROBOT, patch=patch, axis_uv=axis_uv)
    obs.start(views[0])
    return feed(obs, views), q


@pytest.mark.parametrize("kind,uv", [("lever", (250.0, 300.0)), ("round", (180.0, 340.0)), ("round", (290.0, 360.0))])
def test_part_rotation_corotating_versus_blocked_on_the_fake(kind, uv):
    """Free sense (contract -1 on the task-2-like hinge): corotating at the decision steps 12 and 16; blocked
    (contract +1, the part stays at its limit while the hand yaws): part_not_corotating. Round part: patches on the
    disk texture (the boss between the finger pads is partly camera-rigid, see the next test)."""
    free, q_free = rotation_reports(kind, -1, uv)
    blocked, q_blocked = rotation_reports(kind, +1, uv)
    assert q_free[-1] > 0.15 and abs(q_blocked[-1] + 0.005) < 1e-3                # fake: free turns, blocked stays
    for k in (11, 15):
        assert (free[k].verdict, free[k].reason) == ("met", "corotating"), (k, free[k].evidence)
        assert (blocked[k].verdict, blocked[k].reason) == ("violated", "part_not_corotating"), k
    assert all(r.verdict == "pending" for r in free[:6])          # < K7_ONSET_DEG + 2 x 3 deg of hand yaw
    thr = free[11].thresholds[0]
    assert thr.name == "K7_rotation_deg" and thr.value >= O.K7_FLOOR_DEG
    hyp = blocked[15].evidence["hypothesis"]
    assert hyp["class"] == "part_not_corotating" and hyp["ncc_blocked"] - hyp["ncc_corotating"] >= hyp["need"] > 0
    assert {"K7_rotation_deg", "K7_rotation_drop", "K7_hypothesis_lead"} <= {t.name for t in blocked[15].thresholds}
    strict_json(blocked[15].to_dict())


def test_part_rotation_patch_beside_the_finger_pads_is_never_wrong():
    """The round fake's boss between the finger pads (partly camera-rigid pixels): the span hypothesis test is
    ambiguous on the blocked twist, so the observer stays pending there; never a wrong verdict in either sense."""
    for sense, wrong in ((-1, "part_not_corotating"), (+1, "corotating")):
        reports, _ = rotation_reports("round", sense, (250.0, 400.0))
        assert not any(r.reason == wrong for r in reports), sense


def test_part_rotation_unmeasurable_on_a_textureless_patch():
    views, _ = twist_frames("dark", +1)
    obs = O.PartRotationObserver(robot=ROBOT, patch={"uv": (250.0, 345.0)}, axis_uv=None)
    obs.start(views[0])
    rep = obs.update(views[0], None, views[1])
    assert (rep.verdict, rep.reason) == ("failed", "unmeasurable")
    assert rep.evidence["sense_by_wedge_only"] is True and rep.evidence["unmeasurable"] == "textureless_patch"
    for patch in (None, {"unmeasurable": "textureless"}):
        o = O.PartRotationObserver(robot=ROBOT, patch=patch, axis_uv=None)
        o.start(views[0])
        assert o.update(views[0], None, views[1]).verdict == "failed"


@pytest.mark.parametrize("sense", [+1, -1])
@pytest.mark.parametrize("uv", [(250.0, 345.0), (205.0, 345.0), (200.0, 360.0), (250.0, 300.0), (295.0, 345.0)])
def test_dark_rotary_part_measures_or_reports_unmeasurable_never_wrong(uv, sense):
    """Dark uniform rotary fake (features only on occluding edges): at the decision steps the observer either
    measures the right class or reports unmeasurable; never the wrong verdict."""
    reports, _ = rotation_reports("dark", sense, uv)
    right = "part_not_corotating" if sense > 0 else "corotating"
    wrong = "corotating" if sense > 0 else "part_not_corotating"
    for k in (11, 15):
        assert reports[k].reason != wrong, (uv, sense, k, reports[k].evidence)
        assert reports[k].reason in (right, "unmeasurable", None)


def test_part_rotation_structure_axis_mode_with_an_injected_axis():
    """Axis mode (textureless patch, a start axis given): the injected structure axis decides."""
    views, _ = twist_frames("dark", +1)
    state = {"angle": 0.0}

    def axis_fn(gray, uv):                       # stand-in: the part's image axis turns with the hand (blocked)
        state["angle"] += math.radians(1.15)
        return np.array([math.cos(state["angle"]), math.sin(state["angle"])]), 0.9, 5.0
    obs = O.PartRotationObserver(robot=ROBOT, patch={"uv": (250.0, 345.0)}, axis_uv=np.array([1.0, 0.0]),
                                 axis_fn=axis_fn)
    obs.start(views[0])
    reps = feed(obs, views)
    assert reps[11].evidence["mode"] == "axis" and reps[11].reason == "part_not_corotating"


# ------------------------------------------------- PartRotationObserver: degenerate patches, onset lag, pauses (BO-R1)
ROT_CENTRE = (256.0, 400.0)                       # the image rotation centre of the synthetic twists
_SYNTH = {}


def synthetic_texture():
    if "tex" not in _SYNTH:
        rng = np.random.default_rng(1)
        g = cv2.GaussianBlur(rng.integers(40, 200, (512, 512)).astype(np.uint8), (0, 0), 2.0)
        _SYNTH["tex"] = np.repeat(g[..., None], 3, axis=2)
    return _SYNTH["tex"]


def concentric_texture():
    """Isophotes concentric with the rotation centre (a smooth cap shaded about the knob axis): every rotation
    about the centre maps the image onto itself."""
    yy, xx = np.mgrid[0:512, 0:512].astype(float)
    g = np.clip(90 + 50 * np.sin(np.hypot(xx - ROT_CENTRE[0], yy - ROT_CENTRE[1]) / 7.0), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def synthetic_twist(apparent, steps=16, still=(), image=None, yaw_deg=1.15):
    """Views of a twist about the approach (yaw_deg per step, 0 on ``still`` steps) whose wrist image is the
    texture rotated about ROT_CENTRE by ``apparent(state)`` deg; ``state`` = {"theta", "phi", "d", "rel"} (hand yaw
    since the start, yaw since the last pause, this step's yaw, the previous apparent angle)."""
    tex = synthetic_texture() if image is None else image
    R = np.asarray(R_T2, float)
    views = [make_view(state_with(R, 20.0), wrist=tex)]
    state = {"theta": 0.0, "phi": 0.0, "d": 0.0, "rel": 0.0, "residual": 0.0}
    for k in range(steps):
        d = 0.0 if k in still else yaw_deg
        R = rotz(-math.radians(d)) @ R                    # the approach is world -z: +d about the approach
        state["d"] = d
        state["theta"] += d
        state["phi"] = 0.0 if d == 0.0 else state["phi"] + d
        state["rel"] = float(apparent(state))
        M = cv2.getRotationMatrix2D(ROT_CENTRE, state["rel"], 1.0)
        views.append(make_view(state_with(R, 20.0), wrist=cv2.warpAffine(tex, M, (512, 512), flags=cv2.INTER_LINEAR,
                                                                         borderMode=cv2.BORDER_REFLECT)))
    return views


def lagging_free_part(lag_max=5.7, scale=3.0, relax=0.5):
    """A free part that lags the hand by a compliance lag saturating at lag_max (real task-2 steady free twist: 3.0
    deg at 3.9 deg of yaw, 4.7 at 7.5, 5.6 at 11), relaxing by ``relax`` per still step and re-developing after."""
    def apparent(s):
        if s["d"] == 0.0:
            s["residual"] = s["rel"] * relax
            return s["residual"]
        return s["residual"] + (lag_max - s["residual"]) * (1.0 - math.exp(-s["phi"] / scale))
    return apparent


def rotation_series(views, uv=(336.0, 300.0)):
    obs = O.PartRotationObserver(robot=ROBOT, patch={"uv": uv}, axis_uv=None)
    obs.start(views[0])
    return feed(obs, views)


def test_part_rotation_rotation_degenerate_patch_is_unmeasurable():
    """K7_MIN_ROTATION_DROP: a textured patch whose rotation a translation imitates (isophotes concentric with the
    rotation axis, like the real knob cap) is 'rotation_degenerate' -> failed unmeasurable (sense_by_wedge_only),
    never 'corotating' on a part that does not turn."""
    blocked = synthetic_twist(lambda s: s["theta"], image=concentric_texture())
    reports = rotation_series(blocked)
    assert all((r.verdict, r.reason) == ("failed", "unmeasurable") for r in reports)
    ev = reports[0].evidence
    assert ev["unmeasurable"] == "rotation_degenerate" and ev["sense_by_wedge_only"] is True
    assert ev["rotation_drop"] < O.K7_MIN_ROTATION_DROP
    textured = rotation_series(synthetic_twist(lambda s: s["theta"]))
    assert textured[0].evidence["rotation_drop"] > 5 * O.K7_MIN_ROTATION_DROP
    assert textured[11].reason == "part_not_corotating"                        # the decision step (k = 11)


def test_part_rotation_free_part_lag_at_onset_is_not_a_blocked_verdict():
    """BO-R1: a free part's compliance lag (~5.7 deg, built over the first degrees of yaw) changes the apparent
    rotation by > 3 deg over the first 6 deg of every motion. The onset allowance keeps that span out of the test:
    no part_not_corotating at any step; corotating at the decision steps 12 and 16."""
    reports = rotation_series(synthetic_twist(lagging_free_part()))
    assert not any(r.reason == "part_not_corotating" for r in reports), [r.reason for r in reports]
    assert reports[11].reason == "corotating" and reports[15].reason == "corotating"
    assert reports[11].evidence["span_start_index"] >= reports[11].evidence["onset_index"] > 0
    assert reports[11].evidence["span_rotation_deg"] == pytest.approx(0.0, abs=O.K7_FLOOR_DEG)


def test_part_rotation_pause_restarts_the_onset_allowance():
    """A pause (hand still) lets the lag relax; it re-develops when the hand resumes. Pending while still
    (hand_still), the onset allowance restarts, and no part_not_corotating anywhere (the old trailing span flagged
    every resume)."""
    still = tuple(range(14, 20))
    reports = rotation_series(synthetic_twist(lagging_free_part(), steps=36, still=still))
    assert not any(r.reason == "part_not_corotating" for r in reports), [r.reason for r in reports]
    assert all(reports[k].verdict == "pending" and reports[k].evidence["hand_still"] for k in still)
    resumed = reports[20].evidence
    assert resumed["bout_start_index"] == 20 and resumed["onset_index"] is None
    assert reports[13].reason == "corotating" and reports[35].reason == "corotating"


def test_part_rotation_verdict_needs_two_agreeing_frames():
    """K7_CONFIRM_FRAMES: the first judged frame is pending (confirmed False); the verdict follows on the next."""
    reports = rotation_series(synthetic_twist(lambda s: s["theta"]))
    first = next(i for i, r in enumerate(reports) if r.evidence.get("raw_class"))
    assert reports[first].verdict == "pending" and reports[first].evidence["confirmed"] is False
    assert reports[first + 1].reason == "part_not_corotating" and reports[first + 1].evidence["confirmed"] is True


# --------------------------------------------- real task-2 wrist recordings (BO-R1; skipped unless the runs exist)
REQUIRE_RUN_DATA = os.environ.get("AGENT_SPEEDUP_V2_REQUIRE_RUN_DATA") == "1"
REVIEW_PATCHES = [(170.0, 330.0), (350.0, 330.0), (260.0, 280.0), (205.0, 300.0), (320.0, 300.0)]


def _latest_run(pattern):
    runs = sorted(glob.glob(str(ROOT / "runs" / pattern)), reverse=True)
    for run in runs:
        if (Path(run) / "steps.json").is_file():
            return Path(run)
    if REQUIRE_RUN_DATA:
        pytest.fail(f"run data {pattern} required (AGENT_SPEEDUP_V2_REQUIRE_RUN_DATA=1) but absent")
    pytest.skip(f"run data {pattern} is absent")


def _run_views(run, phases=("twist", "twist_hold"), limit=None):
    steps = json.loads((run / "steps.json").read_text())

    def url(path):
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def view(i):
        return StepView(PolicyObservation(url(run / "frames" / f"agentview_{i:03d}.png"),
                                          url(run / "frames" / f"wrist_{i:03d}.png"), tuple(steps[i]["state8"]),
                                          PROMPT))
    twist = [i for i, s in enumerate(steps) if s["phase"] in phases][:limit]
    return [view(twist[0] - 1)] + [view(i) for i in twist], [steps[i] for i in twist]


def _knob_grid(stride_px=30):
    """Patches whose 41 px core lies >= 90 % on the knob at the pre-twist pose (evaluator segmentation of the
    masks recording, used here only to place patches; the observer sees RGB and state8 only)."""
    run = _latest_run("agent_speedup_v2_w1_bofix_steady_blocked_masks_*")
    steps = json.loads((run / "steps.json").read_text())
    i0 = next(i for i, s in enumerate(steps) if s["phase"] == "twist") - 1
    knob = np.asarray(Image.open(run / "frames" / f"wrist_knob_{i0:03d}.png"))[:, ::-1] > 0
    r = O.K7_PATCH_RADIUS_PX
    return [(float(u), float(v)) for v in range(250, 400, stride_px) for u in range(140, 380, stride_px)
            if knob[v - r:v + r + 1, u - r:u + r + 1].mean() >= 0.9]


@pytest.mark.parametrize("pattern,limit", [("agent_speedup_v2_w1_bofix_steady_free_2*", None),
                                           ("agent_speedup_v2_w1_bofix_steady_pause_*", None),
                                           ("agent_speedup_v2_w1_bo_review_effect_frames_*", 160)])
def test_real_free_twists_never_read_part_not_corotating(pattern, limit):
    """Real task-2 FREE twists (steady K2 command at 1.15 deg/step; with a 10-step pause; the M0 probe's 8.6 deg
    increments): no part_not_corotating on the review's patches or on an on-knob grid; the steady twist reaches
    corotating (the knob's q is the truth: it turns)."""
    views, steps = _run_views(_latest_run(pattern), limit=limit)
    assert steps[-1]["knob_q"] - steps[0]["knob_q"] > 0.05
    patches = REVIEW_PATCHES + _knob_grid()
    verdicts = {}
    for uv in patches:
        reasons = [r.reason for r in rotation_series(views, uv)]
        assert "part_not_corotating" not in reasons, uv
        verdicts[uv] = reasons
    if "steady" in pattern:
        assert verdicts[(205.0, 300.0)][-1] == "corotating" and verdicts[(320.0, 300.0)][-1] == "corotating"
    for uv in ((170.0, 330.0), (350.0, 330.0)):                                    # the smooth cap
        assert verdicts[uv][0] == "unmeasurable"


@pytest.mark.parametrize("pattern,limit", [("agent_speedup_v2_w1_bofix_steady_blocked_2*", None),
                                           ("agent_speedup_v2_w1_bo_review_blocked_frames_*", None)])
def test_real_blocked_twists_never_read_corotating_on_the_review_patches(pattern, limit):
    """Real task-2 BLOCKED twists (knob at its -0.005 limit throughout): the review's patches never read corotating
    (the cap patches are rotation_degenerate, the lever-edge patches read part_not_corotating on the steady twist).
    The on-knob grid is reported with its measured residual bounded (a known limitation, not a target): patches at
    the lever's top end, at the cap silhouette and beside the finger pads can read corotating on late frames
    (34 of 529 judged frames on a 256-patch 10 px grid, 5 of 41 on this 30 px grid)."""
    views, steps = _run_views(_latest_run(pattern), limit=limit)
    assert max(abs(s["knob_q"] + 0.005) for s in steps[3:]) < 0.002
    for uv in REVIEW_PATCHES:
        reasons = [r.reason for r in rotation_series(views, uv)]
        assert "corotating" not in reasons, uv
        if "steady" in pattern and uv in ((205.0, 300.0), (320.0, 300.0)):
            assert reasons[-1] == "part_not_corotating", uv
    if "steady" in pattern:
        judged = wrong = 0
        for uv in _knob_grid():
            reasons = [r.reason for r in rotation_series(views, uv)]
            judged += sum(r in ("corotating", "part_not_corotating") for r in reasons)
            wrong += reasons.count("corotating")
        assert judged >= 20 and wrong <= 0.15 * judged, (wrong, judged)
        print("real blocked grid: %d wrong corotating frames of %d judged" % (wrong, judged))


# ===================================================================================== 3. EffectWatch (synthetic)
H = W = 96
REGION = np.zeros((H, W), bool)
REGION[20:80, 15:85] = True                       # 4200 px


def textured(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 120, (H, W, 1)).astype(np.uint8)
    return np.repeat(base, 3, axis=2)


def chroma_changed(img, mask, delta=(40, -20, -20)):
    out = img.astype(int).copy()
    out[mask] += np.array(delta)
    return np.clip(out, 0, 255).astype(np.uint8)


RING = np.zeros((H, W), bool)
RING[40:60, 30:70] = True                         # 800 px = 19 % of the region


def effect_views(on_from, n, base=None, ring=RING):
    base = textured() if base is None else base
    frames = [base] + [chroma_changed(base, ring) if k >= on_from else base for k in range(1, n + 1)]
    return [make_view(T2_START, external=f) for f in frames]


def watch(**kw):
    kw.setdefault("hand_mask_fn", no_hand)
    return O.EffectWatch(region=REGION, false_rate=kw.pop("false_rate", 0.0), sigma_px=0.3, **kw)


def test_effect_onset_after_three_frames_with_theta_e_at_the_first_frame():
    views = effect_views(on_from=4, n=8)
    ew = watch()
    ew.start(views[0])
    reps = [ew.update(views[k - 1], None, views[k], {"theta_part_rad": 0.1 * k}) for k in range(1, 9)]
    verdicts = [r.verdict for r in reps]
    assert verdicts == ["pending"] * 5 + ["met"] * 3                            # onset frames 4, 5, 6 -> met at 6
    met = reps[5]
    assert met.reason == "effect_onset" and met.evidence["onset_step"] == 4 and met.evidence["confirm_step"] == 6
    assert met.evidence["theta_e_rad"] == pytest.approx(0.4) and met.evidence["theta_confirm_rad"] == pytest.approx(0.6)
    assert met.evidence["fraction"] == pytest.approx(800 / 4200, abs=1e-3)
    names = {t.name for t in met.thresholds}
    assert {"O1_pixel_diff", "O1_fraction", "O8_min_rotation_rad"} <= names
    strict_json(met.to_dict())


def test_effect_o8_gating_and_unknown_part_angle():
    views = effect_views(on_from=1, n=8)
    ew = watch()
    ew.start(views[0])
    theta = [0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]                      # 5 deg = 0.0873 rad
    reps = [ew.update(views[k - 1], None, views[k], {"theta_part_rad": theta[k - 1]}) for k in range(1, 9)]
    assert [r.verdict for r in reps[:5]] == ["pending"] * 5                       # run of 5, but theta < 5 deg
    assert reps[5].verdict == "met" and reps[5].evidence["theta_confirm_rad"] == pytest.approx(0.09)
    assert reps[5].evidence["onset_step"] == 1 and reps[5].evidence["theta_e_rad"] == pytest.approx(0.02)
    ew2 = watch()
    ew2.start(views[0])
    reps2 = [ew2.update(views[k - 1], None, views[k], {"theta_part_rad": None}) for k in range(1, 9)]
    assert all(r.verdict == "pending" for r in reps2)                             # sense phase: never confirmed


def test_effect_luminance_only_change_never_counts():
    base = textured()
    bright = np.clip(base.astype(int) + 60 * REGION[..., None], 0, 255).astype(np.uint8)   # grey: L only
    views = [make_view(T2_START, external=f) for f in [base] + [bright] * 5]
    ew = watch()
    ew.start(views[0])
    reps = feed(ew, views, {"theta_part_rad": 1.0})
    assert all(r.verdict == "pending" and r.evidence["fraction"] == 0.0 for r in reps)


def test_effect_occluded_frames_are_skipped_without_breaking_the_run():
    views = effect_views(on_from=1, n=6)
    big = np.zeros((H, W), bool)
    big[10:90, 5:90] = True                                                       # covers the whole region

    def hand(before, after):
        hand.k += 1
        return big if hand.k == 2 else None
    hand.k = 0
    ew = watch(hand_mask_fn=hand)
    ew.start(views[0])
    reps = feed(ew, views, {"theta_part_rad": 1.0})
    assert reps[1].evidence["effect_occluded"] is True and reps[1].reason == "effect_occluded"
    assert reps[1].evidence["run"] == 1                                           # neither counted nor broken
    assert reps[2].verdict == "pending" and reps[3].verdict == "met"              # frames 1, 3, 4 form the run
    assert reps[3].evidence["onset_step"] == 1 and reps[3].evidence["occluded_frames"] == 1


def test_effect_o7_remaining_fraction_rule():
    region = np.zeros((200, 200), bool)
    region[50:150, 50:150] = True                                                  # 10000 px
    swept = np.zeros_like(region)
    swept[50:150, 50:52] = True                                                    # 200 px = 2 %
    swept[50:58, 52] = True                                                        # +8 -> 2.08 %
    swept = np.zeros_like(region)
    swept[50:150, 50:51] = True
    swept[50:110, 51] = True                                                       # 160 px = 1.6 %
    ok = O.EffectWatch.validate_region(region, exclude=(), swept=swept, false_rate=0.0)
    assert ok["valid"] and ok["remaining_fraction"] == pytest.approx(0.984) and ok["overlap_px"] == 160
    assert ok["valid_mask"].sum() == 9840 and ok["reason"] is None
    half = np.zeros_like(region)
    half[50:101, 50:150] = True                                                    # 5100 px -> 49 % remain
    bad = O.EffectWatch.validate_region(region, exclude=(half,), swept=None, false_rate=0.0)
    assert not bad["valid"] and bad["reason"] == "effect_region_invalid" and bad["remaining_fraction"] < 0.5
    small = np.zeros((200, 200), bool)
    small[0:20, 0:35] = True                                                       # 700 px
    cut = np.zeros_like(small)
    cut[0:20, 0:16] = True                                                         # 320 px -> 380 px remain (54 %)
    few = O.EffectWatch.validate_region(small, exclude=(), swept=cut, false_rate=0.0)
    assert not few["valid"] and few["remaining_px"] == 380 and few["remaining_fraction"] > 0.5
    assert not O.EffectWatch.validate_region(region, exclude=(), swept=None, false_rate=0.01)["valid"]
    assert O.EffectWatch.validate_region(region, exclude=(), swept=None, false_rate=0.0099)["valid"]
    assert {t.name for t in ok["thresholds"]} == {"O7_remaining_fraction", "O7_remaining_px", "O7_false_rate"}


def test_effect_valid_mask_phases_and_unwatched_segments():
    views = effect_views(on_from=1, n=6)
    ew = watch()
    ew.start(views[0])
    remainder = REGION & ~RING                                                     # the changing ring swept away
    ew.set_valid_mask(remainder, phase="sense")
    reps = feed(ew, views[:4], {"theta_part_rad": 1.0})
    assert all(r.evidence["fraction"] == 0.0 for r in reps) and reps[-1].evidence["phase"] == "sense"
    ew.set_valid_mask(None, phase="sense")
    assert ew.update(views[3], None, views[4], {"theta_part_rad": 1.0}).evidence["watched"] is False
    validity = O.EffectWatch.validate_region(REGION, exclude=(), swept=None, false_rate=0.0)
    ew.set_valid_mask(validity["valid_mask"], phase="free", validity=validity)
    reps = [ew.update(views[k - 1], None, views[k], {"theta_part_rad": 1.0}) for k in (4, 5, 6)]
    assert reps[-1].verdict == "met" and reps[-1].evidence["phase"] == "free"
    assert reps[-1].evidence["o7"] == {"phase": "free", "valid": True, "remaining_fraction": 1.0,
                                       "remaining_px": 4200, "reason": None}
    assert {"O7_remaining_fraction", "O7_remaining_px", "O7_false_rate"} <= {t.name for t in reps[-1].thresholds}
    strict_json(reps[-1].to_dict())
    with pytest.raises(ValueError):
        ew.set_valid_mask(REGION, phase="all")


def test_effect_persists_met_violated_pending():
    base = textured()
    on = chroma_changed(base, RING)
    hidden = {"mask": None}
    ew = watch(static_hand_mask_fn=lambda view: hidden["mask"])
    ew.start(make_view(T2_START, external=base))
    still_on = ew.persists(make_view(T2_START, external=on))
    assert still_on.verdict == "met" and still_on.evidence["stays"] is True
    lost = ew.persists(make_view(T2_START, external=base))
    assert (lost.verdict, lost.reason) == ("violated", "kept_relation_lost")
    hidden["mask"] = REGION.copy()
    assert ew.persists(make_view(T2_START, external=on)).verdict == "pending"


def test_motion_noise_meter_counts_only_moving_frames_and_ignores_luminance():
    base = textured()
    rng = np.random.default_rng(3)
    moving = {"on": True}

    def hand(before, after):
        if not moving["on"]:
            return None
        m = np.zeros((H, W), bool)
        m[0:10, 0:10] = True
        return m
    frames = [base]
    for k in range(6):
        f = base.astype(int).copy()
        f[REGION] += 50                                                            # reflection: luminance only
        pick = REGION & (rng.random((H, W)) < 0.005)
        f[pick] += np.array([60, -30, -30])                                        # rare chroma false changes
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    views = [make_view(T2_START, external=f) for f in frames]
    meter = O.MotionNoiseMeter(region=REGION, sigma_px=0.3, hand_mask_fn=hand)
    meter.start(views[0])
    feed(meter, views[:4])
    moving["on"] = False
    feed(meter, views[3:])                                                         # static frames: not counted
    res = meter.result()
    assert res["moving_frames"] == 3 and res["frames"] == 6
    assert 0.001 < res["false_rate"] < 0.01 and not res["no_motion"]
    lab = O.MotionNoiseMeter(region=REGION, sigma_px=0.3, statistic="lab", hand_mask_fn=lambda b, a: hand(b, a)
                             if True else None)
    moving["on"] = True
    lab.start(views[0])
    feed(lab, views)
    assert lab.result()["false_rate"] > 0.9                                        # luminance counts with |dLab|
    empty = O.MotionNoiseMeter(region=REGION, sigma_px=0.3, hand_mask_fn=no_hand)
    empty.start(views[0])
    feed(empty, views)
    assert empty.result()["false_rate"] == 0.0 and empty.result()["no_motion"]
    strict_json(res)


def test_select_camera_by_visible_region_share():
    pick = O.EffectWatch.select_camera({"external": {"region_px": 10000, "visible_px": 8500},
                                        "wrist": {"region_px": 50000, "visible_px": 34000}})
    assert pick["camera"] == "external" and pick["reason"] is None
    pick = O.EffectWatch.select_camera({"external": {"region_px": 10000, "visible_px": 3000},
                                        "wrist": {"region_px": 50000, "visible_px": 40000}}, wrist_warp=True)
    assert pick["camera"] == "wrist"
    tie = O.EffectWatch.select_camera({"wrist": {"region_px": 1000, "visible_px": 800},
                                       "external": {"region_px": 1000, "visible_px": 800}}, wrist_warp=True)
    assert tie["camera"] == "external"
    # BO effect spec 4 [C-27]: without a wrist reference warp the wrist camera is never chosen, even when it sees
    # more of the region (the as-run free phase: the wrist saw the ring only at the start of the free phase)
    kept = O.EffectWatch.select_camera({"external": {"region_px": 6000, "visible_px": 3600},
                                        "wrist": {"region_px": 20000, "visible_px": 19000}})
    assert kept["camera"] == "external" and kept["visible"]["wrist"]["eligible"] is False
    assert kept["visible"]["wrist"]["why"] == "no_reference_warp" and kept["wrist_warp"] is False
    lost = O.EffectWatch.select_camera({"external": {"region_px": 6000, "visible_px": 2000},
                                        "wrist": {"region_px": 20000, "visible_px": 19000}})
    assert lost["camera"] is None and lost["reason"] == "effect_region_invalid"
    none = O.EffectWatch.select_camera({"external": {"region_px": 600, "visible_px": 350}, "wrist": None})
    assert none["camera"] is None and none["reason"] == "effect_region_invalid"
    share = O.visible_share(REGION, hidden=RING)
    assert share["region_px"] == 4200 and share["visible_px"] == 3400
    with pytest.raises(ValueError):
        O.EffectWatch(region=REGION, false_rate=0.0, sigma_px=0.3, camera="wrist")


# ============================================================= 3. EffectWatch on the fake rings (Appendix L 12)
@pytest.mark.parametrize("ring,share", [(None, (0.12, 0.28)), (F.EFFECT_RING_PROBE_M, (0.059, 0.079))])
def test_effect_watch_on_both_fake_rings(ring, share):
    """The default ~20 % ring and the probe-matched ring (~6.9 %, the M0 probe's measured share): the motion
    false-change rate measured on the approach, the baseline before the twist, then a free twist through q_on:
    onset (first frame of the run) at the fake's own change step, zero detections before it."""
    scene = F.task2_like_scene(effect_ring_m=ring)
    w = F.FakeTwoViewSim(scene, eef=F.TASK2_START_STATE8[:3], start_tilt_deg=3.14)
    region = w.truth_mask("external", "burner_plate")                             # stands in for the pointed region
    view = StepView(w.observation())
    meter = O.MotionNoiseMeter(region=region, sigma_px=0.35, hand_mask_fn=v1_hand_mask)
    meter.start(view)
    hover = np.asarray(scene.info["hover_eef"]["knob"])
    for _ in range(60):
        err = hover - w.grip
        if np.linalg.norm(err) < 0.003:
            break
        w.step([*np.clip(err / 0.05 * 1.5, -1, 1), 1e-9, 0.0, 0.0, -1.0])
        after = StepView(w.observation())
        meter.update(view, None, after)
        view = after
    noise = meter.result()
    assert noise["moving_frames"] >= 5 and noise["false_rate"] < O.O7_FALSE_RATE_MAX
    assert O.EffectWatch.validate_region(region, exclude=(), swept=None, false_rate=noise["false_rate"])["valid"]
    F.scripted_grasp_rotary(w)
    view = StepView(w.observation())
    ew = O.EffectWatch(region=region, false_rate=noise["false_rate"], sigma_px=0.35, hand_mask_fn=v1_hand_mask)
    ew.start(view)
    theta, rows = 0.0, []
    for k in range(1, 70):
        F.scripted_twist(w, -1, 1, close_first=False, render=True)
        after = StepView(w.observation())
        theta += abs(ROBOT.yaw_about_approach(view.quaternion_xyzw, after.quaternion_xyzw, view.R[:, 2]))
        rep = ew.update(view, None, after, {"theta_part_rad": theta})
        rows.append((k, w.colour_region_changed("burner_plate"), rep))
        view = after
        if rep.verdict == "met" and k >= rep.evidence["confirm_step"] + 1:
            break
    first_changed = next(k for k, changed, _ in rows if changed)
    met = next(rep for _, _, rep in rows if rep.verdict == "met")
    assert met.evidence["onset_step"] == first_changed                            # first frame of the run [C-28]
    assert met.evidence["confirm_step"] == first_changed + 2
    assert not any(rep.verdict == "met" for k, _, rep in rows if k < first_changed + 2)
    assert all((rep.evidence["fraction"] or 0.0) < ew.fraction_th.value for k, _, rep in rows if k < first_changed)
    assert share[0] <= met.evidence["onset_share"] <= share[1]
    print("ring %s: onset share %.4f, motion false rate %.5f (%d frames)" % (ring, met.evidence["onset_share"],
                                                                            noise["false_rate"], noise["moving_frames"]))


# ================================================================ 3. EffectWatch on the M0 probe frames (real sim)
def probe_view(tag):
    return make_view(T2_START, external=F.effect_probe_frame(tag))


def test_effect_on_the_probe_key_frames():
    """M0 effect-probe frames (tests/data): against the pre-twist reference, the chroma share is 0 at the last frame
    before q = 0.5 and at the crossing frame (the scene changes one step later), and ~10 % at the first changed
    frame, while the luminance-inclusive statistic is already ~20 % before the onset (arm reflections)."""
    region = F.effect_probe_region_mask()
    ew = O.EffectWatch(region=region, false_rate=0.0, sigma_px=0.34, hand_mask_fn=v1_hand_mask)
    ew.start(probe_view("pretwist"))
    pre = ew.update(probe_view("pretwist"), None, probe_view("pre_cross"), {"theta_part_rad": 0.65})
    cross = ew.update(probe_view("pre_cross"), None, probe_view("cross"), {"theta_part_rad": 0.69})
    first = ew.update(probe_view("cross"), None, probe_view("cross_plus1"), {"theta_part_rad": 0.71})
    assert pre.evidence["fraction"] == 0.0 and cross.evidence["fraction"] == 0.0
    assert 0.07 <= first.evidence["fraction"] <= 0.13 and first.evidence["run"] == 1
    stays = RegionStaysCheck = O.RegionStaysCheck(region=region, baseline_rgb=F.effect_probe_frame("pretwist"),
                                                  projector_mask=lambda v: None, sigma_px=0.34, false_rate=0.0,
                                                  kind="effect")
    assert stays(probe_view("pre_cross")).verdict == "violated"                   # not yet changed
    assert stays(probe_view("after_retreat")).verdict == "met"                    # stays changed after release
    lab = O.change_magnitude(O._lab255(F.effect_probe_frame("pre_cross")), O._lab255(F.effect_probe_frame("pretwist")),
                             "lab")
    assert (lab[region] > 12).mean() > 0.15                                       # why chroma (Appendix L 8)
    records = F.load_probe("effect_probe_t2_records")
    index = records["frame_tags_to_record_index"]
    assert records["records"][index["cross_plus1"]]["knob_q"] == pytest.approx(0.51461)


def _probe_frames_run():
    for run in sorted(glob.glob(str(ROOT / "runs" / "agent_speedup_v2_w1_bo_effect_frames_*")), reverse=True):
        check = Path(run) / "replay_check.json"
        if check.is_file():
            data = json.loads(check.read_text())
            if data["records_match"]["max_knob_q_abs"] == 0.0 and all(
                    v["max_abs"] == 0 for v in data["tag_frames_vs_m0"].values()):
                return Path(run)
    return None


def test_effect_on_the_full_probe_sequence_detects_at_q_051_054_and_never_before_05():
    """Appendix L 8 validation on EVERY frame of the M0 effect probe (re-recorded bitwise-identical to M0's run by
    BO's dump script; skipped when that run directory is absent): only what a live policy has (the region mask, the
    pre-action reference, the motion false-change rate of the approach and descent, theta_part from state8); the
    knob q scores the result: first frame of the run at q 0.5146, confirmation at q 0.5429, nothing before q 0.5."""
    run = _probe_frames_run()
    if run is None:                    # BO-R7: a gate run sets AGENT_SPEEDUP_V2_REQUIRE_RUN_DATA=1, so this fails
        if REQUIRE_RUN_DATA:
            pytest.fail("Appendix L 8 validation needs runs/agent_speedup_v2_w1_bo_effect_frames_* "
                        "(AGENT_SPEEDUP_V2_REQUIRE_RUN_DATA=1)")
        pytest.skip("BO's full-frame probe re-recording (runs/agent_speedup_v2_w1_bo_effect_frames_*) is absent")
    steps = json.loads((run / "steps.json").read_text())

    def view(i):
        def url(path):
            return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return StepView(PolicyObservation(url(run / "frames" / f"agentview_{i:03d}.png"),
                                          url(run / "frames" / f"wrist_{i:03d}.png"), tuple(steps[i]["state8"]),
                                          PROMPT))
    region = F.effect_probe_region_mask()
    approach = [i for i, s in enumerate(steps) if s["phase"] in ("approach_above", "descend")]
    meter = O.MotionNoiseMeter(region=region, sigma_px=0.34, hand_mask_fn=v1_hand_mask)
    prev = view(approach[0] - 1)
    meter.start(prev)
    for i in approach:
        cur = view(i)
        meter.update(prev, None, cur)
        prev = cur
    noise = meter.result()
    assert noise["moving_frames"] >= 20 and noise["false_rate"] < O.O7_FALSE_RATE_MAX
    twist = [i for i, s in enumerate(steps) if s["phase"] == "twist"]
    ew = O.EffectWatch(region=region, false_rate=noise["false_rate"], sigma_px=0.34, hand_mask_fn=v1_hand_mask)
    prev = view(twist[0] - 1)
    ew.start(prev)
    theta, met = 0.0, None
    for i in twist:
        cur = view(i)
        theta += abs(ROBOT.yaw_about_approach(prev.quaternion_xyzw, cur.quaternion_xyzw, prev.R[:, 2]))
        rep = ew.update(prev, None, cur, {"theta_part_rad": theta, "step": i})
        if steps[i]["knob_q"] < 0.5:
            assert (rep.evidence["fraction"] or 0.0) < ew.fraction_th.value, i
        if rep.verdict == "met":
            met = rep
            break
        prev = cur
    assert met is not None
    q_first, q_confirm = steps[met.evidence["onset_step"]]["knob_q"], steps[met.evidence["confirm_step"]]["knob_q"]
    assert 0.5 <= q_first <= 0.52 and 0.53 <= q_confirm <= 0.545, (q_first, q_confirm)
    print("probe: onset q %.4f confirm q %.4f, motion false rate %.4f over %d frames, threshold %s"
          % (q_first, q_confirm, noise["false_rate"], noise["moving_frames"], ew.fraction_th.derivation))


# ======================================================================================= 3b. RegionStaysCheck
def test_region_stays_check_met_violated_pending_and_effect_inverted():
    base = textured()
    hidden = {"mask": None}
    check = O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: hidden["mask"],
                               sigma_px=0.3, false_rate=0.0)
    assert check(make_view(T2_START, external=base)).verdict == "met"
    moved = base.copy()
    moved[20:80, 15:50] = 200                                                      # the object left: luminance change
    lost = check(make_view(T2_START, external=moved))
    assert (lost.verdict, lost.reason) == ("violated", "kept_relation_lost") and lost.evidence["statistic"] == "lab"
    hidden["mask"] = REGION.copy()
    assert check(make_view(T2_START, external=moved)).verdict == "pending"
    effect = O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: None, sigma_px=0.3,
                                false_rate=0.0, kind="effect")
    assert effect(make_view(T2_START, external=chroma_changed(base, RING))).verdict == "met"
    gone = effect(make_view(T2_START, external=base))
    assert (gone.verdict, gone.reason) == ("violated", "kept_relation_lost") and gone.evidence["statistic"] == "chroma"
    strict_json(gone.to_dict())


# ========================================== 3c. effect visibility (blocker 1): reference, confirm(), stays, camera
def rows_mask(r0, r1, c0=15, c1=85):
    """A band of REGION rows (70 columns wide inside REGION: 70 px per row)."""
    mask = np.zeros((H, W), bool)
    mask[r0:r1, c0:c1] = True
    return mask


def ext_view(rgb):
    return make_view(T2_START, external=rgb)


def test_noise_meter_reference_is_its_start_frame_with_its_hidden_mask():
    """BO effect spec 1: the turn step's MotionNoiseMeter start frame (before the action, hand clear) is the
    reference: its RGB, its Lab and the projected hand mask at that frame, with the meter's max moving share."""
    base = textured()
    corner = np.zeros((H, W), bool)
    corner[0:10, 0:10] = True
    moving = {"on": True}

    def hand(before, after):
        return corner if moving["on"] else None
    meter = O.MotionNoiseMeter(region=REGION, sigma_px=0.3, hand_mask_fn=hand, static_hand_mask_fn=lambda v: corner)
    with pytest.raises(RuntimeError):
        meter.reference()
    meter.start(ext_view(base))
    ref = meter.reference()
    assert np.array_equal(ref["rgb"], base) and np.array_equal(ref["hidden"], corner)
    assert ref["source"] == "noise_meter_start" and ref["max_share"] is None and ref["moving_frames"] == 0
    assert np.allclose(ref["lab"], O._lab255(base))
    ref["rgb"][:] = 0                                                              # a copy: the meter keeps its own
    assert np.array_equal(meter.reference()["rgb"], base)
    feed(meter, [ext_view(base), ext_view(chroma_changed(base, RING))])           # one moving frame, ring changed
    assert meter.reference()["max_share"] == pytest.approx(800 / (4200 - 0), abs=1e-3)
    assert meter.reference()["moving_frames"] == 1


def test_effect_reference_is_the_meter_frame_unless_its_max_share_reaches_the_o1_threshold():
    """The watch judges confirm / persists / stays against the meter's start frame; a meter whose moving frames
    already changed by >= max(O1_FRACTION_FLOOR, O1_FALSE_RATE_K false_rate) falls back to the baseline (with the
    baseline's own hidden mask)."""
    clear, grasp = textured(1), textured(2)
    at_grasp = rows_mask(20, 50)                                                    # 2100 px of the region (50 %)
    small = rows_mask(70, 75)
    fn = (lambda v: at_grasp)
    ok = watch(reference={"rgb": clear, "hidden": small, "max_share": 0.0199}, static_hand_mask_fn=fn)
    assert ok.reference_source == "noise_meter_start" and np.array_equal(ok.reference_rgb, clear)  # before start
    ok.start(ext_view(grasp))
    assert ok.reference_source == "noise_meter_start" and np.array_equal(ok.hidden_ref, small)
    assert ok.reference_info["fallback"] is None and ok.reference_info["hidden_ref_px"] == int((small & REGION).sum())
    bad = watch(reference={"rgb": clear, "hidden": None, "max_share": 0.02}, static_hand_mask_fn=fn)
    assert bad.reference_source is None                                             # rejected; no baseline yet
    bad.start(ext_view(grasp))
    assert bad.reference_source == "baseline" and np.array_equal(bad.reference_rgb, grasp)
    assert np.array_equal(bad.hidden_ref, at_grasp)
    assert bad.reference_info["fallback"] == "reference_max_share"
    assert bad.reference_info["rejected_max_share"] == pytest.approx(0.02)
    scaled = watch(false_rate=0.003, reference={"rgb": clear, "max_share": 0.025})   # threshold max(0.02, 0.03)
    scaled.start(ext_view(grasp))
    assert scaled.reference_source == "noise_meter_start"
    plain = watch(static_hand_mask_fn=fn)                                           # no reference: the baseline
    plain.start(ext_view(grasp))
    assert plain.reference_source == "baseline" and plain.reference_info["fallback"] is None
    assert np.array_equal(plain.hidden_ref, at_grasp)
    pair = watch(reference=(clear, small))                                          # (rgb, hidden)
    assert pair.reference_source == "noise_meter_start" and np.array_equal(pair.hidden_ref, small)
    meter = O.MotionNoiseMeter(region=REGION, sigma_px=0.3, hand_mask_fn=no_hand)
    meter.start(ext_view(clear))
    assert np.array_equal(watch(reference=meter.reference()).reference_rgb, clear)
    with pytest.raises(ValueError):
        watch(reference=(np.zeros((10, 10, 3), np.uint8), None))
    with pytest.raises(ValueError):
        watch(reference=(clear, np.zeros((10, 10), bool)))
    with pytest.raises(ValueError):
        watch(reference={"rgb": clear, "max_share": float("nan")})
    strict_json(jsonable_info(ok.reference_info))


def jsonable_info(info):
    from gpt_field.agent_speedup_v2_types import jsonable
    return jsonable(info)


def test_effect_confirm_needs_three_passing_frames_and_reports_visibility():
    base = textured()
    on = chroma_changed(base, RING)
    ew = watch(reference=(base, None))
    now = rows_mask(20, 40)                                                          # 1400 px hidden, ring untouched
    reps = [ew.confirm(ext_view(on), now, {"step": 500 + k}) for k in range(3)]
    assert [(r.verdict, r.reason) for r in reps] == [("pending", "effect_confirming")] * 2 + [
        ("met", "effect_confirmed")]
    ev = reps[-1].evidence
    assert ev["check"] == "confirm" and ev["valid"] is True and ev["region_px"] == 4200
    assert ev["visible_px"] == 2800 and ev["visible_share"] == pytest.approx(2800 / 4200)
    assert ev["hidden_now_px"] == 1400 and ev["hidden_ref_px"] == 0 and ev["exclude_px"] == 0
    assert ev["share"] == pytest.approx(800 / 2800, abs=1e-3) and ev["run"] == 3 and ev["frames_needed"] == 3
    assert ev["first_step"] == 500 and reps[-1].step == 502 and ev["reference"]["source"] == "noise_meter_start"
    assert ev["flags"] == ["effect_confirmed"] and reps[0].evidence["flags"] == []
    assert reps[-1].progress == pytest.approx((800 / 2800) / 0.02, rel=1e-3)
    names = {t.name for t in reps[-1].thresholds}
    assert {"O1_pixel_diff", "O1_fraction", "O7_remaining_fraction", "O7_remaining_px"} <= names
    strict_json(reps[-1].to_dict())
    local = watch(reference=(base, None))
    assert [local.confirm(ext_view(on)).step for _ in range(2)] == [1, 2]            # no context: the local count


def test_effect_confirm_is_violated_when_the_visible_region_is_unchanged():
    base = textured()
    on = chroma_changed(base, RING)
    ew = watch(reference=(base, None))
    assert ew.confirm(ext_view(on)).reason == "effect_confirming"
    off = ew.confirm(ext_view(base))
    assert (off.verdict, off.reason) == ("violated", "effect_region_unchanged")
    assert off.evidence["run"] == 0 and off.evidence["share"] == 0.0 and off.evidence["flags"] == [
        "effect_region_unchanged"]
    assert [ew.confirm(ext_view(on)).verdict for _ in range(3)] == ["pending", "pending", "met"]   # a fresh run
    bright = np.clip(base.astype(int) + 60 * REGION[..., None], 0, 255).astype(np.uint8)        # luminance only
    lum = watch(reference=(base, None))
    assert lum.confirm(ext_view(bright)).verdict == "violated"
    below = np.zeros((H, W), bool)
    below[40:44, 30:50] = True                                                       # 80 px = 1.90 % < 2 %
    above = below.copy()
    above[44, 30:40] = True                                                          # 90 px = 2.14 %
    tiny = watch(reference=(base, None))
    assert tiny.confirm(ext_view(chroma_changed(base, below))).verdict == "violated"
    assert tiny.confirm(ext_view(chroma_changed(base, above))).reason == "effect_confirming"
    noisy = watch(false_rate=0.003, reference=(base, None))                          # threshold 3 %: 2.14 % fails
    assert noisy.confirm(ext_view(chroma_changed(base, above))).verdict == "violated"


def test_effect_confirm_is_unverifiable_when_reference_and_current_masks_together_hide_half():
    """visible = region - hidden_ref - hidden_now - exclude; valid iff >= O7_MIN_REMAINING_FRACTION of the region
    and >= O7_MIN_REMAINING_PX pixels. Neither mask alone hides half here; together they do."""
    base = textured()
    on = chroma_changed(base, RING)
    ref_hidden, now = rows_mask(20, 38), rows_mask(62, 80)                            # 1260 px each (30 %)
    assert watch(reference=(base, ref_hidden)).confirm(ext_view(on)).reason == "effect_confirming"
    assert watch(reference=(base, None)).confirm(ext_view(on), now).reason == "effect_confirming"
    both = watch(reference=(base, ref_hidden)).confirm(ext_view(on), now)
    assert (both.verdict, both.reason) == ("pending", "effect_unverifiable")
    assert both.evidence["visible_px"] == 1680 and both.evidence["valid"] is False and both.evidence["share"] is None
    assert both.evidence["flags"] == ["effect_unverifiable"] and both.progress is None
    excluded = watch(reference=(base, ref_hidden), exclude=[now]).confirm(ext_view(on))
    assert excluded.reason == "effect_unverifiable" and excluded.evidence["exclude_px"] == 1260
    later = watch(reference=(base, ref_hidden))
    later.set_exclude([now])
    assert later.confirm(ext_view(on)).reason == "effect_unverifiable"
    small = np.zeros((H, W), bool)
    small[0:20, 0:35] = True                                                         # 700 px
    cut = np.zeros((H, W), bool)
    cut[0:20, 0:16] = True                                                           # 380 px remain (54 %)
    few = O.EffectWatch(region=small, false_rate=0.0, sigma_px=0.3, reference=(base, None)).confirm(ext_view(on), cut)
    assert few.reason == "effect_unverifiable" and few.evidence["visible_share"] > 0.5
    run = watch(reference=(base, ref_hidden))                                         # skipped: no count, no break
    seq = [run.confirm(ext_view(on)), run.confirm(ext_view(on), now), run.confirm(ext_view(on)),
           run.confirm(ext_view(on))]
    assert [r.reason for r in seq] == ["effect_confirming", "effect_unverifiable", "effect_confirming",
                                       "effect_confirmed"]
    assert seq[1].evidence["run"] == 1 and seq[-1].evidence["unverifiable_frames"] == 1
    default = watch(reference=(base, None), static_hand_mask_fn=lambda v: REGION)   # hidden_now default: the
    assert default.confirm(ext_view(on)).reason == "effect_unverifiable"            # static hand mask at the view
    with pytest.raises(ValueError):
        watch(reference=(base, None)).confirm(ext_view(on), np.zeros((10, 10), bool))


def test_effect_confirm_ignores_pixels_hidden_in_the_reference():
    """A coloured occluder over the region in the reference that later leaves must not read as the effect."""
    base = textured()
    occluded = chroma_changed(base, RING)                                            # the reference, ring hidden
    naive = watch(reference=(occluded, None))
    assert [naive.confirm(ext_view(base)).verdict for _ in range(3)][-1] == "met"   # departure read as an effect
    honest = watch(reference=(occluded, RING))
    out = honest.confirm(ext_view(base))
    assert (out.verdict, out.evidence["share"]) == ("violated", 0.0)
    assert out.evidence["hidden_ref_px"] == 800 and out.evidence["visible_px"] == 3400


def test_effect_confirm_counts_a_view_once():
    base = textured()
    view = ext_view(chroma_changed(base, RING))
    ew = watch(reference=(base, None))
    reps = [ew.confirm(view) for _ in range(3)]
    assert [r.evidence["run"] for r in reps] == [1, 1, 1] and reps[-1].verdict == "pending"
    assert reps[-1].evidence["repeated_view"] is True


def test_effect_confirm_from_a_stored_reference_without_start():
    """BR's resume path: a fresh watch built from ``reference_record()`` confirms without start(); without any
    reference confirm raises; a rejected reference with no baseline yet is unverifiable until start()."""
    base = textured()
    on = chroma_changed(base, RING)
    first = watch(reference=(base, rows_mask(70, 75)))
    record = first.reference_record()
    assert record["source"] == "noise_meter_start" and record["max_share"] is None
    assert np.array_equal(record["hidden"], rows_mask(70, 75))
    again = watch(reference=record)
    assert [again.confirm(ext_view(on)).verdict for _ in range(3)] == ["pending", "pending", "met"]
    with pytest.raises(RuntimeError):
        watch().confirm(ext_view(on))
    with pytest.raises(RuntimeError):
        watch().reference_record()
    rejected = watch(reference={"rgb": base, "max_share": 0.5})
    out = rejected.confirm(ext_view(on))
    assert out.reason == "effect_unverifiable" and out.evidence["reference"]["source"] is None
    assert out.evidence["reference"]["fallback"] == "reference_max_share"
    rejected.start(ext_view(base))
    assert rejected.reference_source == "baseline"
    assert rejected.confirm(ext_view(on)).reason == "effect_confirming"


def test_effect_persists_judges_against_the_reference_not_the_grasp_baseline():
    """The as-run failure: the baseline at the grasp pose had the hand footprint over most of the region, so the
    stays / persists entry read 'occluded'. Against the meter's hand-clear reference it is judged."""
    clear = textured()
    on = chroma_changed(clear, RING)
    at_grasp = rows_mask(20, 62)                                                     # 2940 px (70 %)
    grasp = clear.copy()
    grasp[at_grasp] = 30                                                             # an achromatic hand
    static = {"mask": at_grasp}
    old = watch(static_hand_mask_fn=lambda v: static["mask"])
    old.start(ext_view(grasp))
    static["mask"] = None                                                            # released, hand away
    stale = old.persists(ext_view(on))
    assert (stale.verdict, stale.reason) == ("pending", "effect_unverifiable")
    assert stale.evidence["hidden_ref_px"] == 2940 and stale.evidence["stays"] is True
    static["mask"] = at_grasp
    ew = watch(reference=(clear, None), static_hand_mask_fn=lambda v: static["mask"])
    ew.start(ext_view(grasp))
    static["mask"] = None
    held = ew.persists(ext_view(on))
    assert (held.verdict, held.reason) == ("met", "effect_persists") and held.evidence["check"] == "persists"
    lost = ew.persists(ext_view(clear))
    assert (lost.verdict, lost.reason) == ("violated", "kept_relation_lost")
    static["mask"] = REGION.copy()
    assert ew.persists(ext_view(on)).reason == "effect_unverifiable"
    unwatched = watch(reference=(clear, None))                                       # the in-twist phase mask does
    unwatched.start(ext_view(grasp))                                                 # not gate persists
    unwatched.set_valid_mask(None, phase="free")
    assert unwatched.persists(ext_view(on)).verdict == "met"


def test_region_stays_effect_kind_uses_the_reference_hidden_mask_and_excludes():
    base = textured()
    on = chroma_changed(base, RING)
    ref_hidden, now = rows_mask(20, 38), rows_mask(62, 80)
    hidden = {"mask": None}
    check = O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: hidden["mask"],
                               sigma_px=0.3, false_rate=0.0, kind="effect", hidden_ref=ref_hidden)
    met = check(ext_view(on))
    assert met.verdict == "met" and met.evidence["visible_px"] == 4200 - 1260
    hidden["mask"] = now                                                             # neither alone hides half
    pend = check(ext_view(on))
    assert (pend.verdict, pend.reason) == ("pending", "effect_unverifiable")
    assert pend.evidence["effect_unverifiable"] is True and pend.evidence["visible_share"] == pytest.approx(0.4)
    assert {"O7_remaining_fraction", "O7_remaining_px"} <= {t.name for t in pend.thresholds}
    strict_json(pend.to_dict())
    hidden["mask"] = None
    excl = O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: None, sigma_px=0.3,
                              false_rate=0.0, kind="effect", hidden_ref=ref_hidden, exclude=[now])
    assert excl(ext_view(on)).reason == "effect_unverifiable"
    gone = check(ext_view(base))
    assert (gone.verdict, gone.reason) == ("violated", "kept_relation_lost")
    unchanged = O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: hidden["mask"],
                                   sigma_px=0.3, false_rate=0.0, hidden_ref=ref_hidden)
    assert unchanged(ext_view(base)).verdict == "met"
    hidden["mask"] = now                                                             # 60 % masked -> occluded
    occl = unchanged(ext_view(base))
    assert (occl.verdict, occl.reason) == ("pending", "occluded")
    with pytest.raises(ValueError):
        O.RegionStaysCheck(region=REGION, baseline_rgb=base, projector_mask=lambda v: None, sigma_px=0.3,
                           false_rate=0.0, hidden_ref=np.zeros((10, 10), bool))


def test_effect_stays_check_is_built_from_the_watch_reference():
    """BR's stays:turned entry: ``EffectWatch.stays_check()`` carries the watch's reference, hidden_ref, excludes,
    false rate and pixel noise, so it agrees with persists() on every frame."""
    base = textured()
    on = chroma_changed(base, RING)
    hidden = {"mask": None}
    ew = watch(false_rate=0.001, reference=(base, rows_mask(20, 30)), exclude=[rows_mask(70, 75)],
               static_hand_mask_fn=lambda v: hidden["mask"])
    stays = ew.stays_check()
    assert isinstance(stays, O.RegionStaysCheck) and stays.kind == "effect"
    assert stays.fraction_th.value == ew.fraction_th.value and stays.pixel_th.value == ew.pixel_th.value
    for mask in (None, rows_mask(30, 62)):
        hidden["mask"] = mask
        for frame in (on, base):
            assert stays(ext_view(frame)).verdict == ew.persists(ext_view(frame)).verdict
    other = ew.stays_check(projector_mask=lambda v: REGION)
    assert other(ext_view(on)).reason == "effect_unverifiable"
    with pytest.raises(RuntimeError):
        watch().stays_check()


def test_effect_confirm_on_the_probe_key_frames():
    """M0 probe frames (real sim): against the hand-clear reset frame, confirm() is violated at the last frame
    before q = 0.5 and at the crossing frame, and confirms on three frames from the first changed one on."""
    region = F.effect_probe_region_mask()
    ew = O.EffectWatch(region=region, false_rate=0.0, sigma_px=0.34, reference=(F.effect_probe_frame("reset"), None))
    assert ew.confirm(probe_view("pre_cross")).reason == "effect_region_unchanged"
    assert ew.confirm(probe_view("cross")).reason == "effect_region_unchanged"
    first = ew.confirm(probe_view("cross_plus1"))
    assert first.reason == "effect_confirming" and 0.07 <= first.evidence["share"] <= 0.13
    assert ew.confirm(probe_view("twist_end")).reason == "effect_confirming"
    done = ew.confirm(probe_view("after_retreat"))
    assert done.reason == "effect_confirmed" and done.evidence["visible_share"] == 1.0


def test_effect_confirm_on_the_full_probe_sequence_never_before_q_05():
    """Every frame of the M0 probe re-recording (skipped when absent): reference = the last static frame before
    the approach (the meter's start); no projected hand mask (none exists offline; the robot is achromatic).
    Never met at q < 0.5; met on the third changed frame and on every later frame up to the end."""
    run = _probe_frames_run()
    if run is None:
        if REQUIRE_RUN_DATA:
            pytest.fail("needs runs/agent_speedup_v2_w1_bo_effect_frames_* (AGENT_SPEEDUP_V2_REQUIRE_RUN_DATA=1)")
        pytest.skip("BO's full-frame probe re-recording (runs/agent_speedup_v2_w1_bo_effect_frames_*) is absent")
    steps = json.loads((run / "steps.json").read_text())

    def view(i):
        def url(path):
            return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return StepView(PolicyObservation(url(run / "frames" / f"agentview_{i:03d}.png"),
                                          url(run / "frames" / f"wrist_{i:03d}.png"), tuple(steps[i]["state8"]),
                                          PROMPT))
    approach = [i for i, s in enumerate(steps) if s["phase"] == "approach_above"]
    ew = O.EffectWatch(region=F.effect_probe_region_mask(), false_rate=0.0, sigma_px=0.34)
    ew.set_reference(np.asarray(view(approach[0] - 1).external_rgb_upright), None)
    verdicts = {}
    for i in range(approach[0], len(steps)):
        rep = ew.confirm(view(i), None, {"step": i})
        verdicts[i] = rep
        if steps[i]["knob_q"] < 0.5:
            assert rep.verdict == "violated", (i, rep.evidence["share"])
    changed = [i for i in verdicts if verdicts[i].verdict != "violated"]
    first = changed[0]
    assert steps[first]["knob_q"] >= 0.5 and verdicts[first + 2].reason == "effect_confirmed"
    assert verdicts[first + 2].evidence["first_step"] == first
    assert all(verdicts[i].verdict == "met" for i in range(first + 2, len(steps)))
    post = [i for i, s in enumerate(steps) if s["phase"] in ("release", "retreat", "static_post")]
    assert min(verdicts[i].evidence["share"] for i in post) > 0.05
    print("probe confirm: first changed frame %d (q %.4f), met from %d" % (first, steps[first]["knob_q"], first + 2))


# ============================================================ 4. CoMotion, Upright, KeepOut, Rest, SpringBack
class ExternalStandIn:
    """Duck-typed ExternalModel: J from the fake external camera's own Jacobian at a point (test side)."""

    def __init__(self, J):
        self.J = np.asarray(J, float)

    def predict_shift(self, dp):
        return self.J @ np.asarray(dp, float)


def test_comotion_met_when_the_object_moves_with_the_hand_and_violated_when_left_behind():
    scene = F.free_object_with_thin_part_scene()
    w = F.FakeTwoViewSim(scene, eef=(0.03, -0.10, 1.15))
    held = F.scripted_grasp_thin_part(w, lift_m=0.0)
    assert held is not None
    view = StepView(w.observation())
    gray = cv2.cvtColor(view.external_rgb_upright, cv2.COLOR_RGB2GRAY).astype(float)
    top = np.array([F.TASK2_MOKA_XY[0], F.TASK2_MOKA_XY[1], 1.02])
    uv_obj, _ = w.project("external", top)
    J = w.external.jacobian(top)
    background = np.array([0.15, -0.25, 0.9])                                     # a support point: never moves
    uv_bg, _ = w.project("external", background)
    good = O.CoMotionObserver(wrist_uv=None, wrist_view=view, ext_track=NCCTrack(gray, uv_obj, radius=12, search=24),
                              external=ExternalStandIn(J), robot=ROBOT, sigma_px=1.0)
    bad = O.CoMotionObserver(wrist_uv=None, wrist_view=view, ext_track=NCCTrack(gray, uv_bg, radius=12, search=24),
                             external=ExternalStandIn(J), robot=ROBOT, sigma_px=1.0)
    good.start(view)
    bad.start(view)
    verdicts_good, verdicts_bad = [], []
    for _ in range(12):
        w.step([0.0, 0.0, 0.5, 1e-9, 0.0, 0.0, 1.0])
        after = StepView(w.observation())
        verdicts_good.append(good.update(view, None, after))
        verdicts_bad.append(bad.update(view, None, after))
        view = after
    assert verdicts_good[0].verdict == "pending"                                  # |J dp| below 3 sigma at first
    assert verdicts_good[-1].verdict == "met", verdicts_good[-1].evidence
    assert (verdicts_bad[-1].verdict, verdicts_bad[-1].reason) == ("violated", "grip_not_on_obj")
    strict_json(verdicts_good[-1].to_dict())


def test_comotion_wrist_patch():
    """Wrist part: displaced beyond O3_WRIST_MAX_PX -> violated at once; a template miss (moved out of the search
    window) is pending for one frame and violated after O3_LOST_FRAMES consecutive misses (cause template_lost)."""
    img = textured(5)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    view = make_view(T2_START, wrist=img)
    obs = O.CoMotionObserver(wrist_uv=(256.0, 300.0), wrist_view=view, ext_track=None, external=None, robot=ROBOT)
    obs.start(view)
    assert obs.update(view, None, make_view(T2_START, wrist=img)).verdict == "met"
    displaced = obs.update(view, None, make_view(T2_START, wrist=np.roll(img, 15, axis=0)))
    assert (displaced.reason, displaced.evidence["cause"]) == ("grip_not_on_obj", "displaced")
    obs = O.CoMotionObserver(wrist_uv=(256.0, 300.0), wrist_view=view, ext_track=None, external=None, robot=ROBOT)
    obs.start(view)
    gone = make_view(T2_START, wrist=np.roll(img, 40, axis=0))                     # beyond the 24 px search
    first = obs.update(view, None, gone)
    assert first.verdict == "pending" and first.evidence["wrist"]["misses"] == 1
    second = obs.update(view, None, gone)
    assert (second.reason, second.evidence["cause"]) == ("grip_not_on_obj", "template_lost")


def _occluded(world, view, point):
    """``view`` with a grey box over the external pixel of world ``point`` (the arm passing in front of it)."""
    ext = np.ascontiguousarray(view.external_rgb_upright).copy()
    u, v = (int(c) for c in world.project("external", point)[0])
    ext[v - 30:v + 31, u - 30:u + 31] = 128
    return make_view(view.state, external=ext, wrist=view.wrist_rgb_upright)


def _carry_views(world, path, *, occlude_at=(), carried_point=None):
    """Views of a scripted carry along ``path`` (P-controller steps); the external image of the steps in
    ``occlude_at`` is occluded over ``carried_point`` (a point of the held object at the start) moved with the
    grip."""
    views = [StepView(world.observation())]
    grip0 = world.grip.copy()
    for target in path:
        for _ in range(40):
            err = np.asarray(target) - world.grip
            if np.linalg.norm(err) < 0.004:
                break
            world.step([*np.clip(err / 0.05 * 1.5, -1, 1), 1e-9, 0.0, 0.0, 1.0])
            view = StepView(world.observation())
            if len(views) in occlude_at:
                view = _occluded(world, view, np.asarray(carried_point) + (world.grip - grip0))
            views.append(view)
    return views


def test_comotion_long_perspective_carry_with_an_occluded_frame():
    """BO-R2: a >= 20 cm perspective carry (10 cm lift, 22 cm transit, 10 cm lowering) of the held object, J_ext
    from the start pose: the affine J dp prediction drifts beyond the old constant 3 sigma = 6 px, yet the
    scale-aware tolerance max(3 sigma, 0.5 |J dp|) keeps the carried object met; one occluded frame is pending
    (not a violation); a template left on the support is violated; two occluded frames in a row are violated."""
    scene = F.free_object_with_thin_part_scene()
    w = F.FakeTwoViewSim(scene, eef=(0.03, -0.10, 1.15))
    assert F.scripted_grasp_thin_part(w, lift_m=0.0) is not None
    v0 = StepView(w.observation())
    top = np.array([F.TASK2_MOKA_XY[0], F.TASK2_MOKA_XY[1], 1.02])
    g0 = v0.position
    path = [g0 + [0.0, 0.0, 0.10], g0 + [0.0, 0.22, 0.10], g0 + [0.0, 0.22, 0.0]]
    occluded = 30
    views = _carry_views(w, path, occlude_at=(occluded,), carried_point=top)
    gray0 = cv2.cvtColor(v0.external_rgb_upright, cv2.COLOR_RGB2GRAY).astype(float)
    J = w.external.jacobian(top)

    def observer(uv):
        obs = O.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=NCCTrack(gray0, uv, radius=12, search=24),
                                 external=ExternalStandIn(J), robot=ROBOT, sigma_px=1.0)
        obs.start(v0)
        return obs
    good = observer(w.project("external", top)[0])
    reps = [good.update(a, None, b) for a, b in zip(views[:-1], views[1:])]
    errors = [r.evidence["external"]["error_px"] for r in reps if r.evidence["external"]["error_px"] is not None]
    assert max(errors) > 6.0                                                      # the old tolerance would fire
    assert not any(r.verdict == "violated" for r in reps), [(i, r.evidence["external"]) for i, r in enumerate(reps)
                                                             if r.verdict == "violated"][:2]
    assert reps[occluded - 1].verdict == "pending" and reps[occluded - 1].evidence["external"]["misses"] == 1
    assert reps[-1].verdict == "met"
    tol = next(t for t in reps[-1].thresholds if t.name == "O3_external_px")
    assert tol.value == pytest.approx(0.5 * reps[-1].evidence["external"]["expected_norm_px"], rel=1e-6)
    left = observer(w.project("external", np.array([0.15, -0.25, 0.9]))[0])      # a support point: never moves
    lrep = [left.update(a, None, b) for a, b in zip(views[:-1], views[1:])]
    assert any((r.reason, r.evidence["cause"]) == ("grip_not_on_obj", "displaced") for r in lrep)
    twice = observer(w.project("external", top)[0])
    doubled = [_occluded(w, v, top + (v.position - g0)) for v in views[1:3]]
    r1 = twice.update(v0, None, doubled[0])
    r2 = twice.update(doubled[0], None, doubled[1])
    assert r1.verdict == "pending" and (r2.reason, r2.evidence["cause"]) == ("grip_not_on_obj", "template_lost")


# ---------------------------------------------- O-1: rotation of the carried part (review-1 decision §2.5, §5.1/5)
LIFT_ACTION_Z = 0.2                     # ~2.6 mm per fake step: 10 steps make the 2.5 cm VerifyHold lift
BODY_TOP = (F.TASK2_MOKA_XY[0], F.TASK2_MOKA_XY[1], 1.0331)     # the free object's top centre (fake truth)
SUPPORT_POINT = np.array([0.15, -0.25, 0.9])                     # a support point: never moves
_FAKE_BAND = {}


def fake_fingertip_band():
    """The fake robot's own wrist pixels as Calibrate+ measures them (geometry ``fingertip_band_mask`` of the open
    hold vs the end of a free-air close), computed once."""
    if "band" not in _FAKE_BAND:
        from gpt_field import agent_speedup_v2_geometry as geometry
        w = F.FakeTwoViewSim(F.free_object_with_thin_part_scene(), eef=(0.03, -0.10, 1.15))
        open_rgb = StepView(w.observation()).wrist_rgb_upright
        for _ in range(25):
            w.step([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])
        _FAKE_BAND["band"] = geometry.fingertip_band_mask(open_rgb, StepView(w.observation()).wrist_rgb_upright)
    return _FAKE_BAND["band"]


def _load_test_views(*, tilt_per_step_deg=0.0, steps=10):
    """A free object held by its thin part (scripted grasp, no lift), then a ~2.5 cm load-test lift in ``steps``
    steps; with ``tilt_per_step_deg`` the object turns in the pinch about the jaw by that much per step (the fake's
    ``tilt`` injection, truth side). Returns the world and the views (start view first)."""
    w = F.FakeTwoViewSim(F.free_object_with_thin_part_scene(), eef=(0.03, -0.10, 1.15))
    assert F.scripted_grasp_thin_part(w, lift_m=0.0) is not None
    views = [StepView(w.observation())]
    w.start_wrist_mask = w.truth_mask("wrist", "moka")         # the held object's wrist pixels (test side)
    for _ in range(steps):
        if tilt_per_step_deg:
            w.inject("tilt", deg=tilt_per_step_deg)
        w.step([0.0, 0.0, LIFT_ACTION_Z, 1e-9, 0.0, 0.0, 1.0])
        views.append(StepView(w.observation()))
    return w, views


def _load_observer(w, v0, *, wrist=True, external=True, ext_point=None, **kw):
    """CoMotionObserver as BR registers it for the load test: the object's external template and J at its top, and
    its rigid wrist patch given by the start world point (the top centre, seen by the wrist camera), with the
    calibrated fingertip band as the robot's own wrist pixels. ``ext_point`` moves the external template elsewhere
    (J stays the object's)."""
    top = np.array(BODY_TOP)
    gray = cv2.cvtColor(v0.external_rgb_upright, cv2.COLOR_RGB2GRAY).astype(float)
    point = top if ext_point is None else np.asarray(ext_point, float)
    kw.setdefault("wrist_patch", {"xyz": top} if wrist else None)
    obs = O.CoMotionObserver(wrist_uv=None, wrist_view=v0,
                             ext_track=NCCTrack(gray, w.project("external", point)[0], radius=12, search=24)
                             if external else None,
                             external=ExternalStandIn(w.external.jacobian(top)) if external else None, robot=ROBOT,
                             sigma_px=1.0, wrist_hand_mask=fake_fingertip_band() if wrist else None, **kw)
    obs.start(v0)
    return obs


def _support_point(view, uv=(70.0, 330.0)):
    """The bare support's world point seen at wrist pixel ``uv`` of ``view`` (it stays in the world)."""
    origin, direction = WristCamera.from_robot(512).ray(HandPose.from_view(view), uv)
    return origin + (F.TASK2_SUPPORT_Z - origin[2]) / direction[2] * direction


def test_o1_rotating_carried_object_is_grip_unstable_and_pure_translation_is_not():
    """§5.1 item 5: the held object turns 10 deg in the pinch during the 2.5 cm lift -> violated grip_unstable
    (cause part_rotated) from the wrist rigid patch, latched from its first confirmation (two frames over
    O3_ROTATION_MAX_DEG), with the O3_rotation_deg threshold in the report; the same lift with the object rigid in
    the hand (pure translation) -> never grip_unstable, met at the end."""
    w, views = _load_test_views(tilt_per_step_deg=1.0)
    assert w.held()["tilt_deg"] == pytest.approx(10.0, abs=0.5)
    assert 0.02 <= views[-1].position[2] - views[0].position[2] <= 0.03
    reps = feed(_load_observer(w, views[0]), views)
    last = reps[-1]
    assert (last.verdict, last.reason, last.evidence["cause"]) == ("violated", "grip_unstable", "part_rotated")
    rot = last.evidence["rotation"]
    assert rot["latched"] and rot["source"] == "wrist" and O.O3_ROTATION_MAX_DEG < rot["estimate_deg"] <= 11.0, rot
    patch = rot["wrist"]["patches"][0]
    assert rot["wrist"]["left_behind"] is False and patch["eligible"]
    # the height-robust sphere estimate: the smallest angle over the top +- one HeightField cell (here the camera is
    # 8 cm above the top, so the cell matters: ~9 deg nominal, ~5.4 deg robust), never above the nominal one
    assert patch["sphere_deg"] <= patch["sphere_nominal_deg"] == pytest.approx(10.0, abs=2.0)
    assert patch["sphere_height_spread_deg"] == pytest.approx(patch["sphere_nominal_deg"] - patch["sphere_deg"])
    threshold = next(t for t in last.thresholds if t.name == "O3_rotation_deg")
    assert threshold.value == O.O3_ROTATION_MAX_DEG and threshold.label == "DD"
    first = next(i for i, r in enumerate(reps) if r.reason == "grip_unstable")
    assert first >= 4 and all(r.reason == "grip_unstable" for r in reps[first:])
    # the object's external template lags J dp (the turn) before the wrist estimate confirms: external_lag first
    assert {r.evidence["cause"] for r in reps[first:]} <= {"external_lag", "part_rotated"}
    latched = next(r for r in reps if r.evidence["cause"] == "part_rotated")
    assert latched.evidence["rotation"]["latched_at"]["estimate_deg"] > O.O3_ROTATION_MAX_DEG
    strict_json(last.to_dict())
    w2, views2 = _load_test_views()
    reps2 = feed(_load_observer(w2, views2[0]), views2)
    assert not any(r.reason == "grip_unstable" for r in reps2)
    assert reps2[-1].verdict == "met", reps2[-1].evidence
    assert max(r.evidence["rotation"]["estimate_deg"] or 0.0 for r in reps2) < 0.5 * O.O3_ROTATION_MAX_DEG


def test_o1_external_rotation_is_evidence_only_and_rotation_verdict_false_never_latches():
    """The external template's in-plane rotation (minus the hand's own rotation about the ray) is measured and
    reported (it sees the 10 deg turn's component about the external ray, ~0.78 here) but never decides: without a
    wrist patch the rotation is ``rotation_unmeasured`` and the verdict is the O3 translation one. With the wrist
    patch, rotation_verdict=False keeps the over-threshold estimate as evidence only."""
    w, views = _load_test_views(tilt_per_step_deg=1.0)
    reps = feed(_load_observer(w, views[0], wrist=False), views)
    assert not any(r.reason == "grip_unstable" for r in reps)
    rot = reps[-1].evidence["rotation"]
    assert rot["rotation_unmeasured"] and rot["estimate_deg"] is None and rot["source"] is None
    assert rot["external"]["estimate_deg"] == pytest.approx(7.8, abs=2.0) and rot["external"]["hand_about_ray_deg"] < 0.5
    quiet = feed(_load_observer(w, views[0], rotation_verdict=False), views)
    assert not any(r.reason == "grip_unstable" for r in quiet)
    assert quiet[-1].evidence["rotation"]["over_threshold"] and not quiet[-1].evidence["rotation"]["verdict_enabled"]
    w2, views2 = _load_test_views()
    reps2 = feed(_load_observer(w2, views2[0], wrist=False), views2)
    assert reps2[-1].verdict == "met" and reps2[-1].evidence["rotation"]["external"]["estimate_deg"] < 2.0


def test_o1_external_lag_with_the_wrist_holding_is_never_met():
    """BO-R1-01: an external translation violation is never overruled into met. The object turns 6 deg in the pinch
    (the wrist estimate under-reads it below O3_ROTATION_MAX_DEG) while the external template lags J dp beyond its
    tolerance (here it sits on the support): the wrist says the part moves with the hand, so the lag is the part
    turning about the pinch -> violated grip_unstable, cause external_lag; with rotation_verdict=False (a carry-scope
    registration) the O3 rule -> grip_not_on_obj, cause displaced. The same rigid lift with the object's own
    template -> met."""
    w, views = _load_test_views(tilt_per_step_deg=0.6)
    assert w.held()["tilt_deg"] > O.O3_ROTATION_MAX_DEG
    reps = feed(_load_observer(w, views[0], ext_point=SUPPORT_POINT), views)
    last = reps[-1]
    assert last.evidence["rotation"]["estimate_deg"] < O.O3_ROTATION_MAX_DEG            # the wrist under-reads it
    assert last.evidence["wrist"]["left_behind"] is False
    ext = last.evidence["external"]
    assert ext["decided"] and ext["error_px"] > ext["tolerance_px"] and ext["lag_with_wrist_held"]
    assert (last.verdict, last.reason, last.evidence["cause"]) == ("violated", "grip_unstable", "external_lag")
    assert not any(r.verdict == "met" for r in reps if r.evidence["external"]["lag_with_wrist_held"])
    carry = feed(_load_observer(w, views[0], ext_point=SUPPORT_POINT, rotation_verdict=False), views)
    # gap round HS-H3: in the carry scope the measuring wrist model that reads the part held makes the displaced
    # external track unverified (pending), never met
    assert (carry[-1].verdict, carry[-1].evidence["cause"], carry[-1].evidence["external_cause"]) == \
        ("pending", "external_unverified", "displaced")
    w2, views2 = _load_test_views()
    rigid = feed(_load_observer(w2, views2[0]), views2)
    assert rigid[-1].verdict == "met" and not rigid[-1].evidence["external"]["lag_with_wrist_held"]


def test_o1_patch_left_behind_is_grip_not_on_obj_not_unstable():
    """The wrist patch on something that stays in the world while the hand lifts (the planned object is not in the
    hand): its motion matches the left-behind prediction (the start world point projected at the current pose), so
    no in-hand rotation is estimated -> pending while the prediction is below the decision distance, then violated
    grip_not_on_obj with cause left_behind; never grip_unstable although the sphere angle grows."""
    w, views = _load_test_views()
    obs = O.CoMotionObserver(wrist_uv=None, wrist_view=views[0], ext_track=None, external=None, robot=ROBOT,
                             wrist_patch={"xyz": _support_point(views[0])}, wrist_hand_mask=fake_fingertip_band())
    obs.start(views[0])
    reps = feed(obs, views)
    assert reps[0].verdict == "pending" and not any(r.reason == "grip_unstable" for r in reps)
    assert reps[0].evidence["rotation"]["rotation_pending"] or reps[0].evidence["rotation"]["estimate_deg"] is None
    last = reps[-1]
    assert (last.reason, last.evidence["cause"]) == ("grip_not_on_obj", "left_behind")
    assert last.evidence["rotation"]["left_behind"] and last.evidence["rotation"]["estimate_deg"] is None
    assert last.evidence["rotation"]["wrist"]["left_behind_error_px"] < 0.2 * \
        last.evidence["rotation"]["wrist"]["left_behind_shift_px"]


def test_o1_wrist_patches_selected_from_the_part_mask():
    """``wrist_patch={"mask", "z_m"}``: ``select_wrist_patches`` picks a far patch (the farthest textured window
    of the part, >= O3_PATCH_MIN_DIST_M from the pinch) and a texture patch outside its window, both inside the mask
    eroded by the window plus one projected HeightField cell, off the robot's own pixels and back-projected to z_m;
    the observer then flags the 10 deg turn (grip_unstable) and keeps the rigid lift met. The hand mask is
    required."""
    w, views = _load_test_views(tilt_per_step_deg=1.0)
    v0 = views[0]
    pose0 = HandPose.from_view(v0)
    band = fake_fingertip_band()
    patches = O.select_wrist_patches(v0.wrist_gray_upright, w.start_wrist_mask, pose0, BODY_TOP[2], hand_mask=band)
    assert [p["role"] for p in patches] == ["far", "texture"]
    far, texture = patches
    assert min(far["dist_m"], texture["dist_m"]) >= O.O3_PATCH_MIN_DIST_M
    assert far["std"] >= O.O3_PATCH_TEXTURE_FRACTION * max(far["std"], texture["std"])
    assert max(abs(far["uv"][0] - texture["uv"][0]), abs(far["uv"][1] - texture["uv"][1])) > 2 * O.K7_PATCH_RADIUS_PX
    margin = far["erode_px"]
    assert margin >= 1 and texture["erode_px"] == margin
    r = O.K7_PATCH_RADIUS_PX + margin
    for patch in patches:
        u, v = (int(round(c)) for c in patch["uv"])
        assert w.start_wrist_mask[v - r:v + r + 1, u - r:u + r + 1].all()           # window + one cell inside
        assert not band[v - r:v + r + 1, u - r:u + r + 1].any()                    # never on the robot
        assert patch["xyz"][2] == pytest.approx(BODY_TOP[2], abs=1e-9) and patch["std"] >= O.K7_MIN_STD_GREY
    assert O.select_wrist_patches(v0.wrist_gray_upright, np.zeros_like(w.start_wrist_mask), pose0, BODY_TOP[2],
                                  hand_mask=band) == []
    with pytest.raises(ValueError):
        O.select_wrist_patches(v0.wrist_gray_upright, w.start_wrist_mask, pose0, BODY_TOP[2], hand_mask=None)
    spec = {"mask": w.start_wrist_mask, "z_m": BODY_TOP[2]}
    reps = feed(_load_observer(w, v0, wrist_patch=spec), views)
    assert reps[-1].reason == "grip_unstable" and reps[-1].evidence["rotation"]["source"] == "wrist"
    assert len(reps[-1].evidence["rotation"]["wrist_patches"]) == 2
    assert reps[-1].evidence["rotation"]["estimate_deg"] == pytest.approx(10.0, abs=2.5)
    w2, views2 = _load_test_views()
    reps2 = feed(_load_observer(w2, views2[0], wrist_patch=spec), views2)
    assert reps2[-1].verdict == "met" and not any(r.reason == "grip_unstable" for r in reps2)
    with pytest.raises(ValueError):
        O.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=None, external=None, robot=ROBOT,
                           wrist_patch={"mask": w.start_wrist_mask}, wrist_hand_mask=band)       # no z_m
    with pytest.raises(ValueError):
        O.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=None, external=None, robot=ROBOT,
                           wrist_patch=spec)                                                     # no hand mask


def test_o1_hand_pixels_inside_the_mask_never_become_patches():
    """BO-R1-02: a mask that also contains the robot's own pixels (closed fingers, hand body) over a region that is
    left behind (bare support beside the object): the patches avoid the fingertip band and the rows below it, so
    the motion reads left behind -> grip_not_on_obj (never met). Without the hand's pixels removed (an empty hand
    mask, test side) the same mask yields patches on the hand, rigid with the camera, and an object left behind
    reads met: the failure the hand mask prevents. A mask of the hand alone gives no patch (pending, never met), and
    an explicit patch on the hand is refused (``patch_on_hand``)."""
    w, views = _load_test_views()
    v0 = views[0]
    band = fake_fingertip_band()
    support = np.zeros(band.shape, bool)
    support[150:380, 10:120] = True
    support &= ~w.start_wrist_mask
    hand = np.zeros(band.shape, bool)
    hand[395:500, 120:400] = True                      # closed fingers and hand body: rows the robot occupies
    assert not hand[:int(np.flatnonzero(band.any(axis=1)).min())].any() and (hand & band).sum() > 0.8 * hand.sum()

    def run(mask, hand_mask):
        obs = O.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=None, external=None, robot=ROBOT,
                                 wrist_patch={"mask": mask, "z_m": F.TASK2_SUPPORT_Z}, wrist_hand_mask=hand_mask)
        obs.start(v0)
        return obs, feed(obs, views)
    obs, reps = run(support | hand, band)
    assert all(p["uv"][1] + O.K7_PATCH_RADIUS_PX < 390 for p in obs.wrist_patches_used)
    assert (reps[-1].reason, reps[-1].evidence["cause"]) == ("grip_not_on_obj", "left_behind")
    assert not any(r.verdict == "met" for r in reps)
    blind, blind_reps = run(support | hand, np.zeros_like(band))
    assert all(hand[int(p["uv"][1]), int(p["uv"][0])] for p in blind.wrist_patches_used)
    assert blind_reps[-1].verdict == "met"                                   # what the hand mask prevents
    _obs, alone = run(hand, band)
    assert all(r.verdict == "pending" for r in alone)
    on_hand = O.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=None, external=None, robot=ROBOT,
                                 wrist_patch=[{"xyz": _support_point(v0)}, {"uv": (256.0, 450.0)}],
                                 wrist_hand_mask=band)
    on_hand.start(v0)
    reps = feed(on_hand, views)
    assert reps[-1].evidence["rotation"]["wrist"]["patches"][1] == {"unmeasurable": "patch_on_hand"}
    assert (reps[-1].reason, reps[-1].evidence["cause"]) == ("grip_not_on_obj", "left_behind")


def test_o1_mixed_patch_set_is_judged_by_the_patch_that_moves_with_the_hand():
    """A set with one patch held (the object's top) and one consistent with being left behind (bare support) is
    not left behind (``patches_disagree``): on real holds that turned 13-18 deg in the pinch a patch near the pinch
    matched the left-behind prediction while the far one read the turn (10 of 80 RS-7 holds at the first patch
    selection), so the rotation is judged on the held patch: met on a rigid lift, grip_unstable when the object
    turns 10 deg. (A held-looking patch on the robot itself is excluded upstream: see the hand-pixel test.)"""
    for tilt, expected in ((0.0, ("met", None)), (1.0, ("violated", "grip_unstable"))):
        w, views = _load_test_views(tilt_per_step_deg=tilt)
        obs = O.CoMotionObserver(wrist_uv=None, wrist_view=views[0], ext_track=None, external=None, robot=ROBOT,
                                 wrist_patch=[{"xyz": _support_point(views[0])}, {"xyz": np.array(BODY_TOP)}],
                                 wrist_hand_mask=fake_fingertip_band())
        obs.start(views[0])
        last = feed(obs, views)[-1]
        rows = last.evidence["rotation"]["wrist"]["patches"]
        assert [rows[0]["left_behind"], rows[1]["left_behind"]] == [True, False]
        assert last.evidence["rotation"]["wrist"]["patches_disagree"] and last.evidence["wrist"]["left_behind"] is False
        assert (last.verdict, last.reason) == expected and last.evidence["rotation"]["wrist"]["patch"] == 1


class _ScriptedWristSet:
    """Stand-in for the wrist patch set: returns scripted per-frame rows (``_WristRotationSet.measure`` format)."""

    def __init__(self, rows):
        self.rows, self.reason, self.drop = list(rows), None, [0.1, 0.1]

    def measure(self, gray, pose, decide_px):
        return copy_row(self.rows.pop(0))


def copy_row(row):
    return json.loads(json.dumps(row))


def _patch(left_behind, estimate):
    return {"uv": [100.0, 100.0], "left_behind": left_behind, "estimate_deg": estimate,
            "eligible": left_behind is False}


def _set_row(*patches):
    decided = [p["left_behind"] for p in patches if p["left_behind"] is not None]
    left = None if not decided else all(decided)
    eligible = [p["estimate_deg"] for p in patches if p["eligible"] and p["estimate_deg"] is not None]
    pending = [p["estimate_deg"] for p in patches if not p["eligible"] and p["left_behind"] is not True
               and p["estimate_deg"] is not None]
    return {"uv": [100.0, 100.0], "left_behind": left, "patches_disagree": len(set(decided)) > 1,
            "estimate_deg": max(eligible) if eligible and left is not True else None,
            "pending_estimate_deg": max(pending) if pending and left is not True else None, "patch": 0,
            "patches": list(patches)}


def test_o1_a_glitch_latch_never_overrides_a_sustained_left_behind_reading():
    """BO-R1-04 (scripted wrist rows on the fake's lift views): both patches read left behind, then one patch's
    tracker jumps for two frames (decided not left behind, estimate 9 deg: the latch fires), then both read left
    behind again -> after O3_ROTATION_CONFIRM_FRAMES left-behind frames the verdict is grip_not_on_obj (left_behind)
    with the latch kept as evidence (``latch_overridden``). An over-threshold estimate of a patch whose left-behind
    hypothesis is still undecided never counts toward the latch."""
    w, views = _load_test_views()
    lb = _set_row(_patch(True, 1.0), _patch(True, 1.0))
    glitch = _set_row(_patch(True, 1.0), _patch(False, 9.0))
    script = [lb, lb, glitch, glitch, lb, lb, lb, lb, lb, lb]
    obs = _load_observer(w, views[0], external=False, wrist_patch={"xyz": np.array(BODY_TOP)})
    obs.wrist_rot = _ScriptedWristSet(script)
    reps = feed(obs, views)
    assert [r.reason for r in reps[:2]] == ["grip_not_on_obj"] * 2
    assert reps[3].reason == "grip_unstable" and reps[3].evidence["rotation"]["latched"]
    assert reps[4].reason == "grip_unstable"                                        # one left-behind frame: latch
    assert (reps[5].reason, reps[5].evidence["cause"]) == ("grip_not_on_obj", "left_behind")
    last = reps[-1]
    assert (last.reason, last.evidence["cause"]) == ("grip_not_on_obj", "left_behind")
    assert last.evidence["rotation"]["latch_overridden"] == "left_behind_sustained"
    undecided = _set_row(_patch(None, 9.0), _patch(None, 9.0))
    obs = _load_observer(w, views[0], external=False, wrist_patch={"xyz": np.array(BODY_TOP)})
    obs.wrist_rot = _ScriptedWristSet([undecided] * 10)
    reps = feed(obs, views)
    assert all(r.verdict == "pending" and not r.evidence["rotation"]["latched"] for r in reps)
    assert reps[-1].evidence["rotation"]["rotation_pending"] and reps[-1].evidence["rotation"]["over_frames"] == 0


def test_o1_threshold_is_the_floor_alone_a_step_onset_is_flagged_at_its_second_frame():
    """BO-R1-08 (scripted wrist estimates 0, 0, 0, 7, 7 deg): the jitter of that series is 7/3 deg, so a
    max(5 deg, 3 sigma) threshold would equal the step on its second frame and never flag it; the threshold is
    O3_ROTATION_MAX_DEG alone (sigma_rotation reported), so the step is grip_unstable at its second frame."""
    w, views = _load_test_views(steps=5)
    held = [_set_row(_patch(False, value)) for value in (0.0, 0.0, 0.0, 7.0, 7.0)]
    obs = _load_observer(w, views[0], external=False, wrist_patch={"xyz": np.array(BODY_TOP)})
    obs.wrist_rot = _ScriptedWristSet(held)
    reps = feed(obs, views)
    assert [r.reason for r in reps] == [None, None, None, None, "grip_unstable"]
    rot = reps[-1].evidence["rotation"]
    assert rot["threshold_deg"] == O.O3_ROTATION_MAX_DEG and rot["sigma_rotation_deg"] == pytest.approx(7.0 / 3.0)


def test_o1_dilated_mask_and_a_wrong_height_never_over_read_a_rigid_hold():
    """BO-R1-05: a mask over-grown by 8 or 16 px (a projected region one or more cells too wide) keeps every
    template on the part after the one-cell erosion: the rigid lift stays met with a near-zero estimate, and the 10
    deg turn is still flagged. A patch whose height is 1 cm too high has a tracked ray that misses its sphere during
    the turn: that frame's sphere estimate is withheld (``ray_misses_sphere``), never an over-read."""
    band = fake_fingertip_band()
    w, views = _load_test_views()
    w_t, views_t = _load_test_views(tilt_per_step_deg=1.0)
    for size in (8, 16):
        kernel = np.ones((2 * size + 1, 2 * size + 1), np.uint8)
        spec = {"mask": cv2.dilate(w.start_wrist_mask.astype(np.uint8), kernel) > 0, "z_m": BODY_TOP[2]}
        obs = _load_observer(w, views[0], wrist_patch=spec)
        reps = feed(obs, views)
        assert reps[-1].verdict == "met" and reps[-1].evidence["rotation"]["estimate_deg"] < 0.5
        for patch in obs.wrist_patches_used:
            u, v = (int(round(c)) for c in patch["uv"])
            r = O.K7_PATCH_RADIUS_PX
            assert w.start_wrist_mask[v - r:v + r + 1, u - r:u + r + 1].all()
        assert feed(_load_observer(w_t, views_t[0], wrist_patch=spec), views_t)[-1].reason == "grip_unstable"
    pose0 = HandPose.from_view(views_t[0])
    origin, direction = WristCamera.from_robot(512).ray(pose0, (256.0, 140.0))
    high = origin + (BODY_TOP[2] + 0.01 - origin[2]) / direction[2] * direction
    obs = O.CoMotionObserver(wrist_uv=None, wrist_view=views_t[0], ext_track=None, external=None, robot=ROBOT,
                             wrist_patch={"uv": (256.0, 140.0), "xyz": high}, wrist_hand_mask=band)
    obs.start(views_t[0])
    rows = [r.evidence["rotation"]["wrist"]["patches"][0] for r in feed(obs, views_t)]
    missed = [row for row in rows if row["ray_meets_sphere"] is False]
    assert missed and all(row["sphere_deg"] is None and row["sphere_unmeasurable"] == "ray_misses_sphere"
                          for row in missed)
    assert all((row["estimate_deg"] or 0.0) < O.O3_ROTATION_MAX_DEG for row in missed)


# ------------------------------- O-1 at the measured yaw tail (review-2 decision §2.7 and §6.2 RS-7 extended, BO)
RS7_TILT_CLASSES_DEG = (8.0, 10.0, 13.4)       # the jaw errors of RS-7 extended (13.4 = largest accepted wrong axis)
RS7_LATERAL_MM = 4.0                            # with 4 mm of lateral error along the jaw


def _offset_load_test_views(total_tilt_deg, lateral_mm, *, steps=10):
    """``_load_test_views`` with the grasp moved ``lateral_mm`` along the jaw (R1: world +x after the scripted turn of
    ``scripted_grasp_thin_part``) and the held object turning ``total_tilt_deg`` in the pinch about the jaw, evenly
    over the ~2.5 cm load-test lift. Returns the world, the views (start view first) and the realized lateral error
    along the jaw (m, truth side)."""
    w = F.FakeTwoViewSim(F.free_object_with_thin_part_scene(), eef=(0.03, -0.10, 1.15))
    hover = np.asarray(w.scene.info["hover_eef"]["moka_handle"], float)
    grasp = np.asarray(w.scene.info["grasp_eef"]["moka_handle"], float)
    q = F._matrix_quat(F._rot_axis((0.0, 0.0, 1.0), math.pi / 2) @ w.R)
    F.scripted_goto(w, hover, gripper=-1.0, quat_xyzw=q, max_steps=300)
    jaw = np.asarray(w.R, float)[:, 1]
    jaw = np.r_[jaw[:2] / np.linalg.norm(jaw[:2]), 0.0]
    F.scripted_goto(w, grasp + 1e-3 * lateral_mm * jaw, gripper=-1.0, quat_xyzw=q)
    F.scripted_gripper(w, 1.0, 25)
    assert w.held() is not None and w.held()["group"] == "moka"
    lateral_m = float((np.asarray(w.grip, float) - grasp) @ jaw)
    views = [StepView(w.observation())]
    w.start_wrist_mask = w.truth_mask("wrist", "moka")
    for _ in range(steps):
        if total_tilt_deg:
            w.inject("tilt", deg=total_tilt_deg / steps)
        w.step([0.0, 0.0, LIFT_ACTION_Z, 1e-9, 0.0, 0.0, 1.0])
        views.append(StepView(w.observation()))
    return w, views, lateral_m


@pytest.mark.parametrize("tilt_deg", RS7_TILT_CLASSES_DEG)
@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_o1_flags_every_rs7_tilt_class_with_lateral_error(tilt_deg, sign):
    """Review-2 decision §2.7 / §6.2 (RS-7 extended, T1 mirror): the held object turns 8, 10 or 13.4 deg in the pinch
    during the 2.5 cm lift, grasped 4 mm off along the jaw (either side), with the patches selected from its visible
    wrist mask as BR registers the load test -> violated grip_unstable (cause part_rotated), latched by the end of the
    lift, from the wrist. The estimate is a lower bound of the turn (the height-robust sphere angle): never above it by
    more than 1 deg, and at least 0.8 of it (the fake reads 0.85-0.9)."""
    w, views, lateral_m = _offset_load_test_views(tilt_deg, sign * RS7_LATERAL_MM)
    assert 1000.0 * lateral_m == pytest.approx(sign * RS7_LATERAL_MM, abs=1.0)
    assert w.held()["tilt_deg"] == pytest.approx(tilt_deg, abs=0.5)
    reps = feed(_load_observer(w, views[0], wrist_patch={"mask": w.start_wrist_mask, "z_m": BODY_TOP[2]}), views)
    last = reps[-1]
    assert (last.verdict, last.reason, last.evidence["cause"]) == ("violated", "grip_unstable", "part_rotated")
    rot = last.evidence["rotation"]
    assert rot["latched"] and rot["source"] == "wrist" and rot["wrist"]["left_behind"] is False
    assert len(rot["wrist_patches"]) == 2
    assert 0.8 * tilt_deg <= rot["estimate_deg"] <= tilt_deg + 1.0, rot["estimate_deg"]
    first = next(i for i, r in enumerate(reps) if r.evidence["rotation"]["latched"])
    assert all(r.reason == "grip_unstable" for r in reps[first:])


@pytest.mark.parametrize("sign", (1.0, -1.0))
def test_o1_clean_hold_with_lateral_error_is_never_flagged(sign):
    """RS-7 extended, clean side: the same 4 mm lateral grasp with the object rigid in the hand (pure translation)
    -> never grip_unstable, met at the end, the estimate below half of O3_ROTATION_MAX_DEG."""
    w, views, lateral_m = _offset_load_test_views(0.0, sign * RS7_LATERAL_MM)
    assert 1000.0 * lateral_m == pytest.approx(sign * RS7_LATERAL_MM, abs=1.0) and w.held()["tilt_deg"] < 0.1
    reps = feed(_load_observer(w, views[0], wrist_patch={"mask": w.start_wrist_mask, "z_m": BODY_TOP[2]}), views)
    assert not any(r.reason == "grip_unstable" for r in reps)
    assert reps[-1].verdict == "met", reps[-1].evidence
    assert max(r.evidence["rotation"]["estimate_deg"] or 0.0 for r in reps) < 0.5 * O.O3_ROTATION_MAX_DEG


# ------------------------------------------------- the RS-7 tool's scorer (scripts/agent_speedup_v2_tilt_check.py)
def _tilt_check_script():
    import importlib.util
    import sys
    path = ROOT / "scripts" / "agent_speedup_v2_tilt_check.py"
    spec = importlib.util.spec_from_file_location("agent_speedup_v2_tilt_check_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rs7_attempt(flag, *, in_hand=7.0, tilt=12.0, error=False, err_jaw_mm=2.0):
    if error:
        return {"error": "Traceback: scripted failure"}
    return {"result": "hold_pass", "pre_close": {"err_jaw_mm": err_jaw_mm},
            "o1": {"verdict": "violated" if flag else "met", "flag": flag,
                   "reason": "grip_unstable" if flag else None, "cause": "part_rotated" if flag else None,
                   "estimate_deg": 7.0 if flag else 1.0, "source": "wrist",
                   "latched_at": {"step": 200} if flag else None, "external_series_deg": [1.0, None]},
            "truth_lift": {"in_hand_deg": in_hand if flag else 1.0, "tilt_deg": tilt if flag else 1.0},
            "truth_after_diag": {"tilt_deg": tilt if flag else 1.0}}


def _write_rs7(directory, episodes, overrides=()):
    """rs7_ep<k>.json reports: clean unflagged, every forced set flagged, then ``overrides`` (episode, variant,
    attempt) applied."""
    S = _tilt_check_script()
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob("rs7_ep*.json"):
        f.unlink()
    for episode in episodes:
        attempts = {S.CLEAN_SET: _rs7_attempt(False)}
        attempts.update({name: _rs7_attempt(True) for name in S.RS7_SETS})
        for ep, name, attempt in overrides:
            if ep == episode:
                attempts[name] = attempt
        (directory / f"rs7_ep{episode}.json").write_text(json.dumps({"episode": episode, "attempts": attempts}))
    return S.summarize(directory)


def test_tilt_check_variants_cover_the_rs7_extended_classes():
    """RS-7 extended (review-2 decision §6.2): forced sets at jaw errors 8, 10 and 13.4 deg, each with the four sign
    combinations of the jaw turn and the 4 mm lateral error, plus the clean reference; all in the default run."""
    S = _tilt_check_script()
    assert set(S.RS7_CLASSES) == {"8", "10", "13.4"}
    for name, sets in S.RS7_CLASSES.items():
        assert len(sets) == 4
        combos = {(math.copysign(1.0, S.VARIANTS[s][0]), math.copysign(1.0, S.VARIANTS[s][1])) for s in sets}
        assert combos == {(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)}
        assert {abs(S.VARIANTS[s][0]) for s in sets} == {float(name)}
        assert {abs(S.VARIANTS[s][1]) for s in sets} == {4.0}
    assert S.RS7_SETS == tuple(s for sets in S.RS7_CLASSES.values() for s in sets)
    assert S.VARIANTS[S.CLEAN_SET] == (0.0, 0.0)
    assert set(S.DEFAULT_VARIANTS) == {S.CLEAN_SET, *S.RS7_SETS}
    assert {"p10p4", "p10m4", "m10m4", "m10p4"} == set(S.RS7_CLASSES["10"])       # the review-1 sets keep their names


def test_tilt_check_scores_rs7_per_class_and_per_sign_set(tmp_path):
    """The RS-7 extended bound: every forced sign set of every class flagged in >= 4/5 of its holds (8/10 over ten
    episodes) and no clean hold flagged; one set at 3/5 fails its class and the gate (the class-level share, 18/20,
    is reported but does not rescue it); a flagged clean hold or an errored attempt fails; a run without a class
    (a review-1 directory) does not pass the extended gate."""
    eps = range(5)
    ok = _write_rs7(tmp_path / "ok", eps)
    assert ok["rs7_pass"] and ok["clean_pass"] and all(ok["rs7_pass_by_set"].values())
    assert all(c["pass"] and c["flags"] == 20 and c["n"] == 20 for c in ok["rs7_pass_by_class"].values())
    assert set(ok["rs7_pass_by_class"]) == {"8", "10", "13.4"} and not ok["classes_missing"]
    weak = _write_rs7(tmp_path / "weak", eps, [(3, "m8m4", _rs7_attempt(False)), (4, "m8m4", _rs7_attempt(False))])
    assert not weak["rs7_pass"] and not weak["rs7_pass_by_set"]["m8m4"] and weak["rs7_pass_by_set"]["p8p4"]
    eight = weak["rs7_pass_by_class"]["8"]
    assert not eight["pass"] and eight["class_share_pass"] and (eight["flags"], eight["n"]) == (18, 20)
    assert weak["rs7_pass_by_class"]["10"]["pass"] and weak["rs7_pass_by_class"]["13.4"]["pass"]
    one_miss = _write_rs7(tmp_path / "one", eps, [(2, "m13.4p4", _rs7_attempt(False))])
    assert one_miss["rs7_pass"] and one_miss["by_variant"]["m13.4p4"]["o1_flags"] == 4
    dirty = _write_rs7(tmp_path / "dirty", eps, [(0, "clean", _rs7_attempt(True))])
    assert not dirty["rs7_pass"] and not dirty["clean_pass"] and not any(dirty["rs7_pass_by_set"].values())
    errored = _write_rs7(tmp_path / "err", eps, [(1, "p10m4", _rs7_attempt(True, error=True))])
    assert not errored["rs7_pass"] and not errored["rs7_pass_by_set"]["p10m4"]
    ten = _write_rs7(tmp_path / "ten", range(10), [(k, "p8m4", _rs7_attempt(False)) for k in (0, 5)])
    assert ten["rs7_pass"] and ten["by_variant"]["p8m4"]["n"] == 10
    ten_bad = _write_rs7(tmp_path / "ten_bad", range(10), [(k, "p8m4", _rs7_attempt(False)) for k in (0, 5, 9)])
    assert not ten_bad["rs7_pass"] and not ten_bad["rs7_pass_by_set"]["p8m4"]
    S = _tilt_check_script()
    r1 = tmp_path / "r1"
    r1.mkdir()
    for episode in eps:
        attempts = {S.CLEAN_SET: _rs7_attempt(False), **{name: _rs7_attempt(True) for name in S.RS7_CLASSES["10"]}}
        (r1 / f"rs7_ep{episode}.json").write_text(json.dumps({"episode": episode, "attempts": attempts}))
    old = S.summarize(r1)
    assert not old["rs7_pass"] and set(old["classes_missing"]) == {"8", "13.4"}
    assert old["rs7_pass_by_class"]["10"]["pass"] and old["rs7_pass_by_class"]["8"] is None
    truth = ok["by_variant"]["p13.4p4"]
    assert (truth["in_hand_truth"], truth["agree_in_hand_5deg"]) == (5, 5)


def _rs7_not_held(result):
    """An attempt that never reached a held load test, as the tool writes it: an empty close (no O-1 report) or a
    VerifyHold failure before the first lift transition (an O-1 report without a verdict)."""
    empty = result == "empty_close"
    row = {"result": result, "pre_close": {"err_jaw_mm": 2.0},
           "close": {"object_present": not empty, "final_gap_mm": 0.0 if empty else 15.0}}
    if not empty:
        row["verify_hold"] = {"reason": result, "actions": 0}
        row["o1"] = {"verdict": None, "reason": None, "cause": None, "flag": False, "estimate_deg": None,
                     "source": None, "latched_at": None, "external_series_deg": []}
        row["truth_lift"] = {"in_hand_deg": None, "tilt_deg": None}
    return row


def test_tilt_check_counts_only_held_load_tests(tmp_path):
    """Review-2 BO-R2-1: only a HOLD (VerifyHold passed, O-1 reported) counts toward a set's n. Five clean attempts
    that closed empty are no clean holds: the clean set has n=0 and fails (it passed before the fix); one empty clean
    close among ten episodes leaves 9 holds and passes; a forced attempt that failed VerifyHold (even with an O-1
    flag) is reported in not_held and is neither a flagged nor an unflagged hold, so five dropped forced attempts
    fail their set and one dropped among five leaves 4 holds (< 5) and fails; truth agreement is over holds only."""
    eps = range(5)
    empty = _write_rs7(tmp_path / "empty", eps, [(k, "clean", _rs7_not_held("empty_close")) for k in eps])
    clean = empty["by_variant"]["clean"]
    assert not empty["clean_pass"] and not empty["rs7_pass"] and not any(empty["rs7_pass_by_set"].values())
    assert (clean["n"], clean["attempts"], clean["not_held"], clean["o1_flags"]) == (0, 5, 5, 0)
    assert clean["not_held_by_result"] == {"empty_close": 5} and clean["agree_in_hand_5deg"] == 0
    one = _write_rs7(tmp_path / "one_empty", range(10), [(3, "clean", _rs7_not_held("empty_close"))])
    assert one["clean_pass"] and one["rs7_pass"] and one["by_variant"]["clean"]["n"] == 9 and one["not_held"] == 1
    dropped = dict(_rs7_attempt(True), result="hold_lost")
    lost = _write_rs7(tmp_path / "lost", eps, [(k, "p8p4", dropped) for k in eps])
    v = lost["by_variant"]["p8p4"]
    assert (v["n"], v["not_held"], v["not_held_o1_flags"], v["o1_flags"]) == (0, 5, 5, 0)
    assert v["not_held_by_result"] == {"hold_lost": 5}
    assert lost["clean_pass"] and not lost["rs7_pass_by_set"]["p8p4"] and not lost["rs7_pass"]
    assert not lost["rs7_pass_by_class"]["8"]["pass"] and lost["rs7_pass_by_class"]["8"]["not_held"] == 5
    jam = _write_rs7(tmp_path / "jam", eps, [(2, "m10p4", _rs7_not_held("hold_jam"))])
    assert jam["by_variant"]["m10p4"]["n"] == 4 and not jam["rs7_pass_by_set"]["m10p4"] and not jam["rs7_pass"]


def test_tilt_check_realized_lateral_bins_are_diagnosis(tmp_path):
    """Review-2 BO-R2-2: the held forced holds of each class binned by jaw sign and REALIZED lateral error (>= 4 mm
    either side; smaller ones in ``under``), each bin scored as a sign set; never gating. A "+4 mm" set that realizes
    +2 mm lands in ``under``; +6 mm commanded holds realizing +4.5 mm fill the + bins; pooled runs count an
    (episode, variant) once."""
    S = _tilt_check_script()
    eps = range(5)
    over = [(k, name, _rs7_attempt(True, err_jaw_mm=-5.5 if name.endswith("m4") else 2.0))
            for k in eps for name in S.RS7_SETS]
    gate = _write_rs7(tmp_path / "gate", eps, over)
    real = gate["realized_lateral"]
    assert real["gating"] is False and gate["rs7_pass"]
    ten = real["classes"]["10"]
    assert ten["bins"]["p10m4"]["n"] == 5 and ten["bins"]["m10m4"]["pass"] and ten["under"]["n"] == 10
    assert ten["bins"]["p10p4"]["n"] == 0 and not ten["bins"]["p10p4"]["pass"] and not ten["pass"]
    lat6 = tmp_path / "lat6"
    lat6.mkdir()
    for k in eps:
        attempts = {S.CLEAN_SET: _rs7_attempt(False),
                    **{name: _rs7_attempt(True, err_jaw_mm=4.5) for name in S.REALIZED_P4_SETS}}
        (lat6 / f"rs7_ep{k}.json").write_text(json.dumps({"episode": k, "attempts": attempts}))
    pooled = S.realized_lateral([tmp_path / "gate", lat6, lat6])
    assert pooled["gating"] is False and pooled["duplicates_skipped"] == 5 * len(S.REALIZED_P4_SETS)
    for label, c in pooled["classes"].items():
        assert c["pass"], (label, c)
        assert c["bins"][f"p{label}p4"]["variants"] == [f"p{label}p6"] and c["bins"][f"p{label}p4"]["n"] == 5
    miss = S.realized_lateral([tmp_path / "gate"])
    assert not any(c["pass"] for c in miss["classes"].values())


def _rotated(img, angle_deg, centre):
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), float(angle_deg), 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def test_o1_wrist_in_plane_rotation_without_a_world_point():
    """Wrist patch by pixel only (``wrist_uv``, no world point): the v1 10 px translation rule plus the patch's
    in-plane rotation. By default (the contract signature, no ``wrist_patch``) the rotation is evidence only: the
    pre-review O3 verdicts. With rotation_verdict=True: a patch turning 1 deg per frame -> grip_unstable once two
    frames exceed 5 deg; a 3 px shift -> met; a single 10 deg frame between steady ones -> never confirmed; a
    textureless wrist with no external template -> rotation_unmeasured, never grip_unstable."""
    img = cv2.GaussianBlur(cv2.resize(textured(5), (512, 512), interpolation=cv2.INTER_NEAREST), (5, 5), 1.5)
    centre = (256.0, 300.0)
    view = make_view(T2_START, wrist=img)

    def observer(**kw):
        obs = O.CoMotionObserver(wrist_uv=centre, wrist_view=view, ext_track=None, external=None, robot=ROBOT, **kw)
        obs.start(view)
        return obs
    turning = [view] + [make_view(T2_START, wrist=_rotated(img, k, centre)) for k in range(1, 9)]
    default = feed(observer(), turning)
    assert not any(r.reason == "grip_unstable" for r in default)
    assert not default[-1].evidence["rotation"]["verdict_enabled"] and default[-1].evidence["rotation"]["over_threshold"]
    reps = feed(observer(rotation_verdict=True), turning)
    over = [r.evidence["rotation"]["estimate_deg"] for r in reps]
    assert over[-1] == pytest.approx(8.0, abs=1.0)
    first = next(i for i, r in enumerate(reps) if r.reason == "grip_unstable")
    assert over[first - 1] > O.O3_ROTATION_MAX_DEG and over[first - 2] <= O.O3_ROTATION_MAX_DEG
    assert reps[first].evidence["rotation"]["source"] == "wrist" and reps[first].evidence["wrist"]["mode"] == "v1_patch"
    shifted = feed(observer(rotation_verdict=True), [view, make_view(T2_START, wrist=np.roll(img, 3, axis=1))])
    assert shifted[-1].verdict == "met" and shifted[-1].evidence["rotation"]["estimate_deg"] < 1.0
    spike = feed(observer(rotation_verdict=True), [view, make_view(T2_START, wrist=_rotated(img, 3, centre)),
                                                   make_view(T2_START, wrist=_rotated(img, 7, centre)),
                                                   make_view(T2_START, wrist=_rotated(img, 3, centre)), view])
    assert not any(r.reason == "grip_unstable" for r in spike)
    assert spike[1].evidence["rotation"]["over_frames"] == 1 and spike[2].evidence["rotation"]["over_frames"] == 0
    flat = make_view(T2_START, wrist=np.full((512, 512, 3), 90, np.uint8))
    obs = O.CoMotionObserver(wrist_uv=centre, wrist_view=flat, ext_track=None, external=None, robot=ROBOT,
                             rotation_verdict=True)
    obs.start(flat)
    rep = obs.update(flat, None, flat)
    assert rep.evidence["rotation"]["rotation_unmeasured"] and rep.reason != "grip_unstable"
    assert rep.evidence["rotation"]["wrist"]["unmeasurable"] == "textureless_patch"
    with pytest.raises(ValueError):
        O.CoMotionObserver(wrist_uv=None, wrist_view=view, ext_track=None, external=None, robot=ROBOT,
                           wrist_patch={"radius_px": 20}, wrist_hand_mask=np.zeros((512, 512), bool))
    with pytest.raises(ValueError):
        O.CoMotionObserver(wrist_uv=centre, wrist_view=view, ext_track=None, external=None, robot=ROBOT,
                           rotation_verdict=1)


def held_frame(**kw):
    base = dict(pid="D", grip_pid="C", grasp_pose=T2_START, offset_h=np.array([0.08, 0.0, 0.0]), vacated_cells=None,
                sigma_offset_m=0.003, b_low=Meas(0.12, 0.003, "touch"), r_obj_m=0.033, extent_axes=None,
                tilt_deg=None, wrist_visible=False, silhouette=None, ext_template=None, gap_ref_mm=14.0,
                sigma_carry_mm=None, support_z_pick=Meas(0.9, 0.001, "plane"), g_close=14.0)
    base.update(kw)
    return HeldFrame(**base)


def test_upright_hand_orientation_commanded_moves_and_silhouette_tilt():
    q0 = matrix_to_quaternion_xyzw(R_T2)
    obs = O.UprightObserver(robot=ROBOT, grasp_quaternion=q0, held=held_frame())
    v0 = make_view(state_with(R_T2, 14.0))
    obs.start(v0)
    assert obs.update(v0, None, make_view(state_with(rotz(math.radians(5)) @ R_T2, 14.0))).verdict == "met"
    tilted = make_view(state_with(rotz(math.radians(20)) @ R_T2, 14.0))
    rep = obs.update(v0, None, tilted, {"primitive": "held_reach"})
    assert (rep.verdict, rep.reason) == ("violated", "object_rotated_in_hand") and rep.evidence["cause"] == \
        "hand_orientation"
    assert obs.update(v0, None, tilted, {"primitive": "orient_restore"}).verdict == "met"     # commanded
    sil = np.zeros((64, 64), bool)
    sil[20:44, 28:36] = True
    current = {"mask": sil}
    obs2 = O.UprightObserver(robot=ROBOT, grasp_quaternion=q0, held=held_frame(silhouette=sil),
                             silhouette_fn=lambda v: current["mask"])
    obs2.start(v0)
    assert obs2.update(v0, None, v0).verdict == "met"
    M = cv2.getRotationMatrix2D((32.0, 32.0), 35.0, 1.0)
    current["mask"] = cv2.warpAffine(sil.astype(np.uint8), M, (64, 64), flags=cv2.INTER_NEAREST) > 0
    rep2 = obs2.update(v0, None, v0)
    assert rep2.verdict == "violated" and rep2.evidence["tilt_change_deg"] > 20.0


def structure_axis_stand_in(gray, uv, half=12):
    """Stand-in for geometry.structure_axis (contract §C2.2: observer tests do not depend on BG): the unit axis
    along the dominant ridge, the coherence and the gradient rms of the (2 half + 1)^2 patch."""
    image = np.asarray(gray, float)
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = image[v - half:v + half + 1, u - half:u + half + 1]
    gy, gx = np.gradient(patch)
    tensor = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    values, vectors = np.linalg.eigh(tensor)
    coherence = float((values[1] - values[0]) / max(values[1] + values[0], 1e-9))
    across = vectors[:, 1]
    return np.array([-across[1], across[0]]), coherence, math.sqrt(max(values[1], 0.0) / patch.size)


def bar_image(angle_deg, centre=(256, 256)):
    """External image of an elongated held object (a bright bar on a textured support) tilted by angle_deg."""
    img = cv2.resize(textured(11), (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    box = cv2.boxPoints(((float(centre[0]), float(centre[1])), (14.0, 70.0), float(angle_deg))).astype(np.int32)
    cv2.fillConvexPoly(img, box, (220, 220, 220))
    return img


def test_upright_external_principal_axis_tilt():
    """BO-R5: O4 from the external principal axis (contract §C9): the held object's external template locates it
    and the structure axis there is compared with the start; 8 deg -> met, 32 deg -> violated in_hand_tilt
    (tilt_source external_axis). The caller's NCCTrack is not moved (a private copy is tracked)."""
    q0 = matrix_to_quaternion_xyzw(R_T2)
    start = make_view(state_with(R_T2, 14.0), external=bar_image(0.0))
    gray = cv2.cvtColor(start.external_rgb_upright, cv2.COLOR_RGB2GRAY).astype(float)
    track = NCCTrack(gray, (256.0, 256.0), radius=16, search=24)
    obs = O.UprightObserver(robot=ROBOT, grasp_quaternion=q0, held=held_frame(), ext_track=track,
                            axis_fn=structure_axis_stand_in)
    obs.start(start)
    small = obs.update(start, None, make_view(state_with(R_T2, 14.0), external=bar_image(8.0)), {"primitive": "held_reach"})
    assert small.verdict == "met" and small.evidence["tilt_source"] == "external_axis"
    assert small.evidence["tilt_change_deg"] == pytest.approx(8.0, abs=2.0)
    big = obs.update(start, None, make_view(state_with(R_T2, 14.0), external=bar_image(32.0)), {"primitive": "held_reach"})
    assert (big.verdict, big.reason, big.evidence["cause"]) == ("violated", "object_rotated_in_hand", "in_hand_tilt")
    assert big.evidence["tilt_change_deg"] > O.O4_TILT_MAX_DEG
    assert any(t.name == "O4_axis_coherence" for t in big.thresholds)
    assert np.allclose(track.uv, (256.0, 256.0))                                   # the caller's track is untouched
    strict_json(big.to_dict())


def test_upright_commanded_reorientation_flag_wins_over_the_primitive_name():
    """BO-R5: an explicit context flag decides whether a hand rotation is commanded; the primitive name prefix is
    only the fallback."""
    q0 = matrix_to_quaternion_xyzw(R_T2)
    obs = O.UprightObserver(robot=ROBOT, grasp_quaternion=q0, held=held_frame())
    v0 = make_view(state_with(R_T2, 14.0))
    obs.start(v0)
    turned = make_view(state_with(rotz(math.radians(20)) @ R_T2, 14.0))
    flagged = obs.update(v0, None, turned, {"primitive": "held_reach", "commanded_reorientation": True})
    assert flagged.verdict == "met" and flagged.evidence["commanded_by"] == "context"
    named = obs.update(v0, None, turned, {"primitive": "orient_restore", "commanded_reorientation": False})
    assert named.verdict == "violated" and named.evidence["cause"] == "hand_orientation"
    fallback = obs.update(v0, None, turned, {"primitive": "orient_restore"})
    assert fallback.verdict == "met" and fallback.evidence["commanded_by"] == "primitive_name"


def test_keepout_contact_inside_and_outside_the_volume():
    keep = KeepOut(pid="E", kind="clear_of", polygon_xy=rect_polygon((-0.20, 0.10), (1.0, 0.0), 0.05, 0.05),
                   z_low=0.90, z_high=1.00, thresholds=())
    obs = O.KeepOutObserver(keepouts=[keep], robot=ROBOT, held=None)
    above = make_view(state_with(R_T2, 20.0, grip=(-0.20, 0.10, 1.25)))
    inside = make_view(state_with(R_T2, 20.0, grip=(-0.20, 0.10, 0.96)))
    far = make_view(state_with(R_T2, 20.0, grip=(0.10, -0.20, 0.96)))
    obs.start(above)
    stall = {"primitive_feedback": {"type": "primitive_step", "reason": "contact_stall", "done": True}}
    assert obs.update(above, None, inside, {"primitive_feedback": {"ratio": 0.2}}).verdict == "met"   # no event
    rep = obs.update(above, None, inside, stall)
    assert (rep.verdict, rep.reason) == ("violated", "keepout_contact") and rep.evidence["hits"][0]["pid"] == "E"
    assert obs.update(above, None, far, stall).verdict == "met"
    assert obs.update(above, None, above, stall).verdict == "met"                 # above the volume
    with_held = O.KeepOutObserver(keepouts=[keep], robot=ROBOT, held=held_frame(offset_h=np.array([0.0, 0.0, 0.0])))
    with_held.start(above)
    hover = make_view(state_with(R_T2, 14.0, grip=(-0.20, 0.10, 1.05)))           # held object reaches 0.93
    assert with_held.update(above, None, hover, stall).evidence["hits"][0]["by"] in ("held", "hand")


def test_rest_observer_met_after_ten_quiet_frames_and_violated_on_motion():
    base = textured()
    obs = O.RestObserver(region=REGION, sigma_static_grey=0.3, hand_mask_fn=no_hand)
    views = [make_view(T2_START, external=base) for _ in range(12)]
    obs.start(views[0])
    reps = feed(obs, views)
    assert reps[8].verdict == "pending" and (reps[9].verdict, reps[9].reason) == ("met", "at_rest")
    moving = [base] + [np.roll(base, 2 * k, axis=1) for k in range(1, 11)]
    obs2 = O.RestObserver(region=REGION, sigma_static_grey=0.3, hand_mask_fn=no_hand)
    mviews = [make_view(T2_START, external=f) for f in moving]
    obs2.start(mviews[0])
    rep = feed(obs2, mviews)[-1]
    assert (rep.verdict, rep.reason) == ("violated", "object_moved_after_release")


def test_rest_observer_ignores_frames_while_the_hand_moves_and_masks_the_footprint():
    """BO-R6: while the hand retreats over the released object (grip moving > M11_HAND_STILL_M per step) the arm's
    pixels change the region; those frames are not measured (pending, hand_moving) and restart the quiet run; with
    the hand still, the projected hand footprint (static_hand_mask_fn) is masked too; then at_rest."""
    base = textured()
    rng = np.random.default_rng(4)
    arm = REGION.copy()
    arm[:, 50:] = False                                                           # the arm over part of the region
    frames, states = [base], [T2_START]
    for k in range(6):                     # retreat: 1 cm / step; the arm crosses the region, then leaves it
        f = base.copy()
        if k < 4:
            f[arm] = rng.integers(0, 255, (int(arm.sum()), 3))
        frames.append(f)
        states.append((*np.add(T2_START[:3], (0.0, 0.0, 0.01 * (k + 1))), *T2_START[3:]))
    fingers = np.zeros((H, W), bool)
    fingers[20:30, 15:85] = True                                                  # the still hand's footprint
    for k in range(O.M11_FRAMES):                                                 # still: only the fingers flicker
        f = base.copy()
        f[fingers] = 255 if k % 2 else 0
        frames.append(f)
        states.append(states[-1])
    views = [make_view(s, external=f) for s, f in zip(states, frames)]
    obs = O.RestObserver(region=REGION, sigma_static_grey=0.3, hand_mask_fn=no_hand,
                         static_hand_mask_fn=lambda view: fingers)
    obs.start(views[0])
    reps = feed(obs, views)
    assert all(r.verdict == "pending" and r.evidence["hand_moving"] for r in reps[:6])
    assert not any(r.verdict == "violated" for r in reps)
    assert (reps[-1].verdict, reps[-1].reason) == ("met", "at_rest") and reps[-1].evidence["measured_run"] == 10
    unmasked = O.RestObserver(region=REGION, sigma_static_grey=0.3, hand_mask_fn=no_hand)
    unmasked.start(views[6])
    assert feed(unmasked, views[6:])[-1].reason == "object_moved_after_release"      # the footprint would read
    moving = O.RestObserver(region=REGION, sigma_static_grey=0.3, hand_mask_fn=no_hand)
    moving.start(views[0])
    assert all(r.verdict == "pending" for r in feed(moving, views[:7]))              # never violated by the arm


def test_spring_back_met_when_static_and_violated_when_the_part_turns_back():
    img = cv2.resize(textured(9), (256, 256), interpolation=cv2.INTER_LINEAR)
    spec = {"uv": (128.0, 128.0), "camera": "external", "px_per_m": 600.0}
    obs = O.SpringBackObserver(patch_or_axis=spec, robot=ROBOT)
    views = [make_view(T2_START, external=img) for _ in range(11)]
    obs.start(views[0])
    reps = feed(obs, views)
    assert reps[9].verdict == "met" and reps[8].verdict == "pending" and reps[3].verdict == "pending"
    frames = [img]
    for k in range(1, 11):
        M = cv2.getRotationMatrix2D((128.0, 128.0), 1.0 * k, 1.0)
        frames.append(cv2.warpAffine(img, M, (256, 256), flags=cv2.INTER_LINEAR))
    obs2 = O.SpringBackObserver(patch_or_axis=spec, robot=ROBOT)
    rviews = [make_view(T2_START, external=f) for f in frames]
    obs2.start(rviews[0])
    rep = feed(obs2, rviews)[-1]
    assert (rep.verdict, rep.reason) == ("violated", "spring_back") and rep.evidence["change_deg"] > 3.0
    nothing = O.SpringBackObserver(patch_or_axis={"unmeasurable": "textureless"}, robot=ROBOT)
    nothing.start(views[0])
    assert nothing.update(views[0], None, views[1]).verdict == "failed"


def test_rest_and_spring_back_on_the_fake():
    """Rest (M11) on the free object's external region: still -> at_rest; nudged 4 mm per step -> moved. K9 on the
    rotary fake after release and a 3 cm retreat (wrist patch on the disk 35 mm off its centre): no spring ->
    met; a return spring -> spring_back."""
    scene = F.free_object_with_thin_part_scene()
    w = F.FakeTwoViewSim(scene, eef=(0.03, -0.10, 1.15))
    region = w.truth_mask("external", "moka")
    for move, expected in ((False, ("met", "at_rest")), (True, ("violated", "object_moved_after_release"))):
        w.reset()
        view = StepView(w.observation())
        obs = O.RestObserver(region=region, sigma_static_grey=0.35, hand_mask_fn=v1_hand_mask)
        obs.start(view)
        for _ in range(O.M11_FRAMES):
            if move:
                w.inject("move_group", group="moka", vector_m=(0.0, 0.004, 0.0))
            w.step([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
            after = StepView(w.observation())
            rep = obs.update(view, None, after)
            view = after
        assert (rep.verdict, rep.reason) == expected
    for spring, expected in ((0.0, "met"), (0.05, "violated")):
        scene = F.rotary_part_scene(spring=spring)
        w = F.FakeTwoViewSim(scene, eef=scene.info["start_eef"])
        F.scripted_grasp_rotary(w)
        F.scripted_twist(w, -1, 30)
        F.scripted_gripper(w, -1.0, 12)
        F.scripted_goto(w, w.grip + [0.0, 0.0, 0.03], gripper=-1.0)
        cx, cy = scene.info["truth"]["disk_centre_xy"]
        uv, _ = w.project("wrist", np.array([cx + 0.035, cy, 0.918]))
        view = StepView(w.observation())
        obs = O.SpringBackObserver(patch_or_axis={"uv": tuple(float(v) for v in uv), "camera": "wrist"}, robot=ROBOT)
        obs.start(view)
        for _ in range(O.K9_FRAMES):
            w.step([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, -1.0])
            after = StepView(w.observation())
            rep = obs.update(view, None, after)
            view = after
        assert rep.verdict == expected, rep.evidence
        if expected == "violated":
            assert rep.reason == "spring_back" and rep.evidence["change_deg"] > O.K9_FLOOR_DEG


def test_keepout_and_upright_on_the_fake():
    """KeepOut: an open-hand guarded descent (v1 AnchoredMove) that stalls on the keep-out part -> keepout_contact;
    the same descent onto bare support elsewhere -> met. Upright on the held free object: a carry keeps it met; a
    hand twist of 20 deg (not an Orient) or a 25 deg in-pinch tilt (silhouette) -> object_rotated_in_hand."""
    from gpt_field.contact_skills import AnchoredMove
    scene = F.task2_like_scene()
    handle = scene.body("frypan_handle")
    hx, hy = handle.centre[:2]
    keep = KeepOut(pid="E", kind="clear_of", polygon_xy=rect_polygon((hx, hy), (0.0, 1.0), 0.08, 0.025),
                   z_low=0.90, z_high=0.99, thresholds=())
    for start, expected in (((hx, hy, 1.05), "violated"), ((-0.20, -0.20, 1.00), "met")):
        w = F.FakeTwoViewSim(scene, eef=start, images=False)
        view = StepView(w.observation())
        obs = O.KeepOutObserver(keepouts=[keep], robot=ROBOT, held=None)
        obs.start(view)
        move = AnchoredMove(view, axis=(0.0, 0.0, -1.0), travel_m=0.15, gripper="open", stop_on_contact=True)
        rep = None
        while not move.done:
            action = move.next_action(view)
            w.step(action)
            after = StepView(w.observation())
            fb = move.observe_transition(view, action, after)
            rep = obs.update(view, action, after, {"primitive_feedback": fb, "primitive": move.name})
            view = after
            if rep.verdict == "violated":
                break
        assert move.reason in ("contact_stall", "contact_touch"), move.reason
        assert rep.verdict == expected, rep.evidence
    scene = F.free_object_with_thin_part_scene()
    w = F.FakeTwoViewSim(scene, eef=(0.03, -0.10, 1.15))
    assert F.scripted_grasp_thin_part(w) is not None
    view = StepView(w.observation())
    held = held_frame(silhouette=w.truth_mask("wrist", "moka"))
    obs = O.UprightObserver(robot=ROBOT, grasp_quaternion=view.quaternion_xyzw, held=held,
                            silhouette_fn=lambda v: w.truth_mask("wrist", "moka"))
    obs.start(view)
    for _ in range(4):
        w.step([0.3, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])
        after = StepView(w.observation())
        rep = obs.update(view, None, after, {"primitive": "held_reach"})
        view = after
    assert rep.verdict == "met", rep.evidence
    w.inject("tilt", deg=25.0)
    w.step([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, 1.0])
    after = StepView(w.observation())
    tilted = obs.update(view, None, after, {"primitive": "held_reach"})
    assert (tilted.verdict, tilted.reason) == ("violated", "object_rotated_in_hand"), tilted.evidence
    assert tilted.evidence["cause"] in ("rigid_hold", "in_hand_tilt")
    w2 = F.FakeTwoViewSim(scene, eef=(0.03, -0.10, 1.15), images=False)
    F.scripted_grasp_thin_part(w2)
    v0 = StepView(w2.observation())
    upright = O.UprightObserver(robot=ROBOT, grasp_quaternion=v0.quaternion_xyzw, held=held_frame())
    upright.start(v0)
    from gpt_field.contact_skills import quaternion_angle_rad
    for _ in range(40):                                                            # a hand twist, not an Orient
        F.scripted_twist(w2, +1, 1, close_first=False)
        if quaternion_angle_rad(w2.quat, v0.quaternion_xyzw) > math.radians(20.0):
            break
    turned = upright.update(v0, None, StepView(w2.observation()), {"primitive": "twist_segment"})
    assert (turned.verdict, turned.evidence["cause"]) == ("violated", "hand_orientation"), turned.evidence


def test_hold_floor_invariant():
    floor = th.Threshold("G6_hold_floor_mm", 12.0, "ONLINE", "test floor")
    inv = O.HoldFloorInvariant(floor=floor)
    v = make_view(state_with(R_T2, 14.0))
    inv.start(v)
    assert inv.update(v, None, v).verdict == "met"
    rep = inv.update(v, None, make_view(state_with(R_T2, 11.0)))
    assert (rep.verdict, rep.reason) == ("violated", "hold_lost") and rep.thresholds == (floor,)


# =========================================================================================== 5. InvariantMonitor
class Scripted:
    name = "scripted"

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.contexts = []
        self.started = 0

    def start(self, view):
        self.started += 1

    def update(self, before, action, after, context=None):
        self.contexts.append(dict(context or {}))
        verdict = self.verdicts.pop(0) if self.verdicts else "met"
        return ObsReport(self.name, verdict, None, "boom" if verdict == "violated" else None, {}, (), 0)


def test_invariant_monitor_preemption_peek_take_scopes_and_context():
    v = make_view(T2_START)
    mon = O.InvariantMonitor()
    a = Scripted(["met", "violated", "violated", "met", "violated"])
    b = Scripted(["violated", "violated"])
    c = Scripted(["met"] * 5)
    mon.add("hold_floor", a, view=v, preempt=True, scope="held")
    mon.add("comotion_load", b, view=v, preempt=False, scope="relation")
    mon.add("upright", c, view=v, preempt=True, scope="relation")
    assert a.started == b.started == c.started == 1
    ctx = {"primitive_feedback": {"x": 1}, "primitive": "held_reach", "relation": "supported_on", "held": "D"}
    mon.observe(v, None, v, ctx)
    assert mon.peek_preemption() is None                                          # b is preempt=False
    mon.observe(v, None, v, ctx)
    key, report = mon.peek_preemption()
    assert key == "hold_floor" and report.verdict == "violated"
    assert mon.peek_preemption()[0] == "hold_floor"                               # peeked, not consumed
    mon.observe(v, None, v, ctx)                                                  # continuing violation
    assert len(mon._queue) == 1
    assert mon.take_preemption()[0] == "hold_floor" and mon.peek_preemption() is None
    mon.observe(v, None, v, ctx)                                                  # met again
    mon.observe(v, None, v, ctx)                                                  # violated again -> queued
    assert mon.peek_preemption()[0] == "hold_floor"
    for obs in (a, b, c):
        assert all({"primitive_feedback", "primitive", "relation", "held", "step"} <= set(x) for x in obs.contexts)
    assert [x["step"] for x in a.contexts] == [1, 2, 3, 4, 5]
    snap = mon.snapshot()
    assert snap["comotion_load"]["verdict"] == "met" and snap["hold_floor"]["scope"] == "held"
    strict_json(snap)
    assert any(ch["key"] == "hold_floor" for ch in mon.last_changes())
    mon.clear("relation")
    assert mon.keys() == ["hold_floor"]
    mon.clear("held")                                                              # drops its queued report too
    assert mon.keys() == [] and mon.peek_preemption() is None
    mon.add("x", Scripted([]), view=v)
    with pytest.raises(ValueError):
        mon.add("x", Scripted([]), view=v)
    mon.clear("episode")
    assert mon.keys() == []


def test_invariant_monitor_rearms_a_delivered_key_while_the_violation_persists():
    """BO-R4: after delivery (take_preemption) a violation that persists queues a fresh report at the next observe
    (at most one queued per key), so it also preempts the recovery's next primitive; removing the invariant (the
    recovery's way to act while it is violated) stops it."""
    v = make_view(T2_START)
    mon = O.InvariantMonitor()
    hold = Scripted(["violated"] * 6)
    mon.add("hold_floor", hold, view=v, preempt=True, scope="held")
    mon.observe(v, None, v)
    mon.observe(v, None, v)
    assert len(mon._queue) == 1                                                   # one per key before delivery
    assert mon.take_preemption()[0] == "hold_floor" and mon.peek_preemption() is None
    mon.observe(v, None, v)                                                       # still violated -> re-queued
    key, report = mon.peek_preemption()
    assert key == "hold_floor" and report.verdict == "violated"
    assert [x["step"] for x in hold.contexts][-1] == 3                            # a fresh report, not the old one
    mon.take_preemption()
    mon.remove("hold_floor")
    mon.observe(v, None, v)
    assert mon.peek_preemption() is None and mon.preemptions == 2


# ================================================================================================ 6. StaysChecker
def test_stays_checker_order_replacement_and_results():
    v = make_view(T2_START)
    calls = []

    def check(name, verdict):
        def run(view):
            calls.append(name)
            return ObsReport(name, verdict, None, "kept_relation_lost" if verdict == "violated" else None, {}, (), 0)
        return run
    stays = O.StaysChecker()
    stays.add("stays:turned:A", check("a", "met"), step_i=0, kind="step")
    stays.add("hold_true:on:B", check("b", "pending"), step_i=None, kind="hold_true")
    stays.add("stays:supported_on:D:B", check("c", "violated"), step_i=1, kind="step")
    results = stays.check_all(v)
    assert [k for k, _ in results] == ["stays:turned:A", "hold_true:on:B", "stays:supported_on:D:B"]
    assert [r.verdict for _, r in results] == ["met", "pending", "violated"] and calls == ["a", "b", "c"]
    stays.add("stays:turned:A", check("a2", "violated"), step_i=0, kind="step")     # re-registered after a reopen
    assert stays.keys() == ["hold_true:on:B", "stays:supported_on:D:B", "stays:turned:A"]
    assert stays.check_all(v)[-1][1].verdict == "violated"
    assert stays.entry("stays:turned:A")["last"]["verdict"] == "violated"
    stays.remove("hold_true:on:B")
    assert len(stays.check_all(v)) == 2
    with pytest.raises(ValueError):
        stays.add("bad", check("x", "met"), step_i=None, kind="other")


# ======================================================================================= constants, lint, firewall
def test_restated_constants_equal_their_sources():
    try:
        from gpt_field import agent_speedup_v2_primitives as primitives
    except Exception as exc:                                            # pragma: no cover - parallel build
        pytest.skip(f"primitives module not importable: {exc}")
    assert O.UPRIGHT_REORIENT_RAD == primitives.M3_REORIENT_RAD
    from gpt_field import external_axes
    assert O.O1_DIFF_FLOOR == external_axes.DIFF_THRESHOLD
    from gpt_field import wrist_servo
    import inspect
    defaults = inspect.signature(wrist_servo.WristPatchMonitor.__init__).parameters
    assert O.O3_WRIST_MAX_PX == defaults["max_displacement_px"].default
    assert O.K7_MIN_PEAK == wrist_servo.NCCTrack.min_peak
    try:
        from gpt_field import agent_speedup_v2_scene as scene
    except Exception as exc:                                            # pragma: no cover - parallel build
        pytest.skip(f"scene module not importable: {exc}")
    assert O.O3_PATCH_MIN_DIST_M == scene.RIGID_PATCH_MIN_DIST_M


def test_lab_conversion_equals_geometry_to_lab255():
    try:
        from gpt_field import agent_speedup_v2_geometry as geometry
    except Exception as exc:                                            # pragma: no cover - parallel build
        pytest.skip(f"geometry module not importable: {exc}")
    rgb = F.effect_probe_frame("cross_plus1")
    np.testing.assert_array_equal(O._lab255(rgb), geometry.to_lab255(rgb))


def test_o3_model_fraction_equals_the_geometry_broyden_limit():
    try:
        from gpt_field import agent_speedup_v2_geometry as geometry
    except Exception as exc:                                            # pragma: no cover - parallel build
        pytest.skip(f"geometry module not importable: {exc}")
    assert O.O3_MODEL_FRACTION == geometry.BROYDEN_MAX_INNOVATION


def test_effect_onset_step_is_the_context_step_or_the_local_count():
    """BO-R8: onset_step / confirm_step are context["step"] when given, else the local update count (= onset_k)."""
    views = effect_views(on_from=2, n=6)
    local, logged = watch(), watch()
    local.start(views[0])
    logged.start(views[0])
    for k in range(1, 7):
        a = local.update(views[k - 1], None, views[k], {"theta_part_rad": 1.0})
        b = logged.update(views[k - 1], None, views[k], {"theta_part_rad": 1.0, "step": 100 + k})
    assert a.evidence["onset_step"] == a.evidence["onset_k"] == 2 and a.step == 6
    assert b.evidence["onset_step"] == 102 and b.evidence["onset_k"] == 2 and b.step == 106


def test_default_hand_mask_is_the_geometry_function():
    try:
        from gpt_field import agent_speedup_v2_geometry as geometry
    except Exception as exc:                                            # pragma: no cover - parallel build
        pytest.skip(f"geometry module not importable: {exc}")
    a, b = F.effect_probe_frame("pre_cross"), F.effect_probe_frame("cross")
    expected = geometry.hand_motion_mask(a, b)
    got = O._default_hand_mask(a, b)
    assert (expected is None and got is None) or np.array_equal(expected, got)


def test_observers_module_passes_the_t0_lint_and_literal_audit():
    if not Path(LIBERO, "libero/libero/bddl_files").is_dir():
        pytest.fail("LIBERO checkout required for the T0 lint vocabulary")
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    path = th.PACKAGE_DIR / "agent_speedup_v2_observers.py"
    assert "agent_speedup_v2_observers.py" in th.CONTROL_MODULES
    assert th.lint_module(path, vocab) == []
    assert th.literal_audit(path) == []
    labels = th.module_labels(path)
    assert labels and all(k.startswith("agent_speedup_v2_observers.") for k in labels)
    source = path.read_text()
    assert "start_poses" not in source and "measured_tasks" not in source


def test_contract_constant_values():
    """§C9 thresholds-owned table, verbatim."""
    expected = {"K1W_DECIDE_STEP": 12, "K1W_EXTEND_STEP": 16, "K1W_WINDOW": 4, "K1W_FIRST_RATE_INDEX": 1,
                "K1W_FREE_LW": 0.1, "K1W_FREE_SIGMA_K": 4.0, "K1W_FREE_EARLY": 0.25, "K1W_BLOCKED_LW": 0.5,
                "K1W_BLOCKED_EARLY": 0.75, "K1W_UNWEDGE_PER_BLOCKED_STEP": 1.5, "K4W_LW": 0.5, "K4W_MEDIAN_K": 5.0,
                "K7_FLOOR_DEG": 3.0, "K7_SIGMA_K": 3.0, "K7_SLIP_LW": 0.5, "K7_SLIP_STEPS": 2, "K9_FLOOR_DEG": 3.0,
                "K9_FLOOR_M": 0.003, "K9_SIGMA_K": 3.0, "K9_FRAMES": 10, "O1_DIFF_FLOOR": 12.0,
                "O1_PX_SIGMA_K": 4.0, "O1_FRACTION_FLOOR": 0.02, "O1_FALSE_RATE_K": 10.0, "O1_FRAMES": 3,
                "O1_HAND_DILATE_PX": 9, "O2_IOU_MIN": 0.8, "O2_SHIFT_FLOOR_PX": 4.0, "O2_SIGMA_K": 3.0,
                "O3_WRIST_MAX_PX": 10.0, "O3_SIGMA_K": 3.0, "O3_ROTATION_MAX_DEG": 5.0,
                "O3_ROTATION_CONFIRM_FRAMES": 2, "O4_TILT_MAX_DEG": 20.0, "O7_FALSE_RATE_MAX": 0.01,
                "O7_MIN_REMAINING_FRACTION": 0.5, "O7_MIN_REMAINING_PX": 400, "O8_MIN_ROTATION_RAD": math.radians(5),
                "M11_FLOOR_GREY": 3.0, "M11_SIGMA_K": 3.0, "M11_FRAMES": 10, "EFFECT_OCCLUDED_FRACTION": 0.5}
    for name, value in expected.items():
        assert getattr(O, name) == value, name
    assert O.OBSERVERS_VERSION == "agent-speedup-v2-observers-5"
