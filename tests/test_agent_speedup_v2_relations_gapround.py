"""Gap round (judge decision §2.5, relations owner): F6-A descent touch keep-out + B2 retry, X-1 disturbance calls,
MC-R2a hover-window resets, HS-H2 re-anchor template ownership, HS-H3 wrist tilt wiring, PR-1 post-release check,
SAL-R2 lateral_unverified and IF-R3' tight_fit. One positive and one negative case per rule. No simulator, no model.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_observers as observers
from gpt_field import agent_speedup_v2_planner as planner
from gpt_field import agent_speedup_v2_relations as R
from gpt_field import agent_speedup_v2_robot as robot_facts
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field.agent_speedup_v2_types import Keep, Meas

import test_agent_speedup_v2_relations as TR


class DisturbScene(TR.StubScene):
    """StubScene with the X-1 / MC-R2a surface (``disturb``, ``window_start``)."""

    def __init__(self, parts=()):
        super().__init__(parts)
        self.disturbed, self.windows = [], []

    def disturb(self, obj, step, cause):
        self.disturbed.append((obj, cause))

    def window_start(self, pid):
        self.windows.append(pid)


def column(finger, centre_xy, half=0.01):
    x, y = centre_xy
    poly = np.array([[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]])
    return planner.FingerColumn(finger, poly, 0.918, 0.002, 1.0, False)


def columned(plan):
    return dataclasses.replace(plan, columns=(column(1, (-0.2, 0.17)), column(2, (-0.2, 0.23))))


# ------------------------------------------------------------------------------------------------ F6-A
def test_f6a_a_descent_touch_stamps_both_columns_counts_a_retry_and_disturbs(monkeypatch):
    first = columned(TR.contact_plan(yaw=-math.pi / 2))
    second = TR.contact_plan(yaw=-math.pi / 2 + kernel.YAW_GRID_RAD)
    script = TR.GraspScript(monkeypatch, plans=[first, second], classes=["blocked_above_part", "expected"])
    monkeypatch.setattr(R, "_descend", TR.scripted({"event": "contact_touch", "contact_grip_z": 0.97}))
    ctx = TR.make_ctx([TR.part("A")])
    ctx.scene = DisturbScene([TR.part("A")])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "held_for_turn")
    assert outcome.local_retries == 1                                         # ruling L9: a counted retry
    stamped = script.keepouts[1]
    assert len(stamped) == 2 and all(k.kind == "clear_of" for k in stamped)
    assert [np.asarray(k.polygon_xy).tolist() for k in stamped] == [c.polygon_xy.tolist() for c in first.columns]
    surface = 0.97 - robot_facts.FINGERTIP_REACH_MM / 1000.0
    assert all(k.z_high == pytest.approx(surface) and k.z_low == pytest.approx(0.918) for k in stamped)
    assert ctx.scene.disturbed[0] == ("A", "descent_touch")
    assert ("contact", "descent_touch") in [(n, p.get("phase")) for n, p in ctx.events.rows]


def test_f6a_after_b2_touches_fails_blocked_above_part_and_never_closes(monkeypatch):
    plans = [columned(TR.contact_plan(yaw=-math.pi / 2 + i * kernel.YAW_GRID_RAD))
             for i in range(R.B2_LOCAL_RETRIES + 2)]
    script = TR.GraspScript(monkeypatch, plans=plans, classes=["blocked_above_part"] * len(plans))
    monkeypatch.setattr(R, "_descend", TR.scripted({"event": "contact_touch", "contact_grip_z": 0.97}))
    ctx = TR.make_ctx()
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "blocked_above_part")
    assert outcome.evidence["next_grip"] and "close" not in script.names()
    assert len(script.keepouts) == R.B2_LOCAL_RETRIES + 1                     # at most B2 + 1 touches per grasp
    assert len(script.keepouts[-1]) == 2 * R.B2_LOCAL_RETRIES


def test_f6a_a_descent_reaching_its_target_stamps_nothing_and_counts_nothing(monkeypatch):
    plan = columned(TR.contact_plan())
    script = TR.GraspScript(monkeypatch, plans=[plan], classes=["expected"])
    ctx = TR.make_ctx([TR.part("A")])
    ctx.scene = DisturbScene([TR.part("A")])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.local_retries) == ("succeeded", 0)
    assert script.keepouts == [[]]
    assert ctx.scene.disturbed == [("A", "close")]                            # only the hold-forming close


# ------------------------------------------------------------------------------------------------ X-1
def test_x1_the_hold_forming_close_disturbs_only_after_the_load_test(monkeypatch):
    order = []
    TR.GraspScript(monkeypatch, plans=[TR.contact_plan()], classes=["expected"])

    def load(ctx, *, grip, obj, attempt):
        order.append(("load", list(ctx.scene.disturbed)))
        return {"ok": True, "reason": "hold_pass", "held": None, "evidence": {}}
    monkeypatch.setattr(R, "_load_test", TR.scripted(load))
    monkeypatch.setattr(R, "_register_held_bus", lambda ctx, **kw: [])
    ctx = TR.make_ctx([TR.part("A")])
    ctx.scene = DisturbScene([TR.part("A")])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="carry"), ctx)
    assert outcome.status == "succeeded"
    assert order == [("load", [])]                                            # prior_at_close saw the old epoch
    assert ctx.scene.disturbed == [("A", "close")]


def test_x1_an_empty_close_disturbs_before_the_next_attempt_and_a_plain_scene_is_untouched(monkeypatch):
    script = TR.GraspScript(monkeypatch, plans=[TR.contact_plan()], classes=["expected", "expected"],
                            closes=[(False, False), (True, False)])
    ctx = TR.make_ctx([TR.part("A")])
    ctx.scene = DisturbScene([TR.part("A")])
    outcome = TR.drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert outcome.status == "succeeded" and script.names().count("close") == 2
    assert ctx.scene.disturbed == [("A", "close"), ("A", "close")]
    plain = TR.make_ctx([TR.part("A")])                                       # negative: no disturb on the scene
    R._disturb(plain, "A", "close")
    assert not hasattr(plain.scene, "disturbed")


# ------------------------------------------------------------------------------------------------ MC-R2a
def test_mc_r2a_hover_windows_call_window_start_and_fall_back_to_part_moved():
    ctx = TR.make_ctx([TR.part("A")])
    ctx.scene = DisturbScene([TR.part("A")])
    R._window_start(ctx, "A", "Z")
    assert ctx.scene.windows == ["A"] and ctx.scene.moved == []
    plain = TR.make_ctx([TR.part("A")])                                       # negative: the frozen S-12 reset
    R._window_start(plain, "A")
    assert plain.scene.moved == ["A"]


def test_mc_r2a_contact_paths_keep_part_moved_and_disturb():
    ctx = TR.make_ctx([TR.part("A"), TR.part("D", move="free")])
    ctx.scene = DisturbScene([TR.part("A"), TR.part("D", move="free")])
    R._moved_by_contact(ctx, "D", "A", "release")
    assert ctx.scene.moved == ["A", "D"] and ctx.scene.disturbed == [("D", "release")] and ctx.scene.windows == []


# ------------------------------------------------------------------------------------------------ HS-H2
def _reanchor_ctx(monkeypatch, hand):
    ctx = TR.make_ctx([TR.part("D", move="free", roles=("carried",))])
    ctx.holding = TR.held_frame()
    ctx.scene.projector = SimpleNamespace(project=lambda xyz: (np.array([200.0, 200.0]), 1.0))
    ctx.scene.hand_swept_mask = lambda camera, view, yaw, **kw: hand
    rng = np.random.default_rng(0)
    monkeypatch.setattr(R, "_external_gray", lambda view: rng.uniform(0, 255, (512, 512)))
    added = []
    monkeypatch.setattr(R, "_comotion", lambda ctx, held, obj, *, rotation_verdict: SimpleNamespace(held=held))
    monkeypatch.setattr(R, "_bus_add", lambda ctx, key, ob, **kw: added.append(key))
    return ctx, added


def test_hs_h2_a_reanchor_template_on_the_hand_is_refused(monkeypatch):
    hand = np.zeros((512, 512), bool)
    hand[195:205, 195:205] = True
    ctx, added = _reanchor_ctx(monkeypatch, hand)
    out = R._reanchor_comotion(ctx, "comotion:D")
    assert out["ok"] is False and out["why"] == "template_on_robot" and added == []


def test_hs_h2_a_reanchor_template_off_the_hand_is_kept(monkeypatch):
    hand = np.zeros((512, 512), bool)
    hand[400:450, 400:450] = True
    ctx, added = _reanchor_ctx(monkeypatch, hand)
    out = R._reanchor_comotion(ctx, "comotion:D")
    assert out["ok"] is True and added == ["comotion:D"]


# ------------------------------------------------------------------------------------------------ HS-H3
def _bus_ctx(monkeypatch, comotion):
    ctx = TR.make_ctx([TR.part("C", move="free"), TR.part("D", move="free", roles=("carried",))])
    ctx.grasp_quaternion = tuple(ctx.view.quaternion_xyzw)
    monkeypatch.setattr(R, "_comotion", lambda ctx, held, obj, *, rotation_verdict: comotion)
    monkeypatch.setattr(R, "_bus_add", lambda ctx, key, ob, **kw: ctx.memory.setdefault("bus", {}).update({key: ob}))
    return ctx


def test_hs_h3_a_carry_upright_reads_the_carry_comotion_wrist_estimate(monkeypatch):
    comotion = SimpleNamespace(wrist_model=True, started=False)
    ctx = _bus_ctx(monkeypatch, comotion)
    keys = R._register_held_bus(ctx, grip="C", obj="D", purpose="carry", keep=(Keep("upright", "D"),),
                                held=TR.held_frame())
    assert keys == ["hold_floor", "upright:D", "comotion:D"]                  # bus order unchanged
    upright = ctx.memory["bus"]["upright:D"]
    assert upright.tilt_fn(ctx.view) == 0.0                                   # before the observer starts
    comotion.__dict__.update(started=True, wrist_rot=SimpleNamespace(reason=None), wrist_misses=0,
                             rot_lost={"wrist": None}, left_behind_run=0, rot_series={"wrist": [1.0, 25.0]})
    assert upright.tilt_fn(ctx.view) == 25.0
    comotion.wrist_misses = observers.O3_LOST_FRAMES                          # the wrist lost: no estimate
    assert upright.tilt_fn(ctx.view) is None


def test_hs_h3_the_observers_wrist_tilt_accessor_is_passed_as_the_bound_method(monkeypatch):
    if not hasattr(observers.CoMotionObserver, "wrist_tilt_deg"):
        pytest.skip("observers owner's CoMotionObserver.wrist_tilt_deg not in this tree")

    class Co(observers.CoMotionObserver):
        def __init__(self):
            self.wrist_estimate_deg = 4.0
    comotion = Co()
    ctx = _bus_ctx(monkeypatch, comotion)
    R._register_held_bus(ctx, grip="C", obj="D", purpose="carry", keep=(Keep("upright", "D"),), held=TR.held_frame())
    upright = ctx.memory["bus"]["upright:D"]
    assert upright.tilt_fn.__self__ is comotion and upright.tilt_fn() == 4.0


def test_hs_h3_no_carry_or_no_wrist_model_leaves_the_external_evidence_deciding(monkeypatch):
    ctx = _bus_ctx(monkeypatch, SimpleNamespace(wrist_model=False, started=False))
    R._register_held_bus(ctx, grip="C", obj="D", purpose="hold", keep=(Keep("upright", "D"),), held=TR.held_frame())
    assert ctx.memory["bus"]["upright:D"].tilt_fn is None
    ctx = _bus_ctx(monkeypatch, SimpleNamespace(wrist_model=False, started=False))
    R._register_held_bus(ctx, grip="C", obj="D", purpose="carry", keep=(Keep("upright", "D"),), held=TR.held_frame())
    assert ctx.memory["bus"]["upright:D"].tilt_fn(ctx.view) is None


# ------------------------------------------------------------------------------------------------ PR-1
def _place(monkeypatch, *, capture=None, hand_now=None, survey=None, support=None, aim_fits=None, pre=None):
    script = TR.PlaceScript(monkeypatch, supports=["on_surface"])
    if support is not None:
        monkeypatch.setattr(R.planner, "support_classify", lambda *a, **k: dict(support))
    if aim_fits is not None:
        monkeypatch.setattr(R.planner, "aim_target", lambda *a, **k: {
            "ok": True, "target_obj_xy": [-0.046, 0.206], "target_grip_xy": [-0.046, 0.126], "tol_m": (0.005, 0.005),
            "m_free_m": 0.05, "thresholds": [], "fits": aim_fits})
    if pre is not None:
        monkeypatch.setattr(R, "_destination_check", lambda ctx, record: dict(pre))
    original = R._pre_release_capture
    if capture is not None:
        monkeypatch.setattr(R, "_pre_release_capture", lambda ctx, obj: capture(original(ctx, obj)))
    surveys = []
    if survey is not None:
        def run_survey(ctx, pid, *, need=None):
            surveys.append(pid)
            return survey(ctx, pid)
            yield
        monkeypatch.setattr(R, "_reacquire_at_rest", lambda ctx, pid, expected: run_survey(ctx, pid))
    ctx = TR.holding_ctx()
    if hand_now is not None:
        ctx.scene.hand_swept_mask = lambda camera, view, yaw, **kw: hand_now
    outcome = TR.drive(R.place_on(ctx, TR.place_step()), ctx)
    return outcome, script, surveys


def _changed(cap, where=None):
    rgb = np.asarray(cap["rgb"]).copy()
    mask = TR.effect_mask() if where is None else where
    rgb[mask] = 255 - rgb[mask]
    rgb[mask, 0] = np.where(rgb[mask, 0] > 127, 0, 255)
    return dict(cap, rgb=rgb)


def test_pr1_an_object_that_left_its_region_during_the_release_is_not_placed(monkeypatch):
    outcome, script, surveys = _place(monkeypatch, capture=_changed)
    assert (outcome.status, outcome.reason) == ("failed", "object_moved_after_release")
    assert outcome.evidence["post_release"]["verdict"] == "violated" and surveys == []
    assert "changed region percent" in outcome.evidence_text


def test_pr1_an_object_that_stays_with_the_hand_over_part_of_its_region_before_release_is_placed(monkeypatch):
    half = TR.effect_mask().copy()
    half[:, 390:] = False                                                    # the hand covered the left half

    def capture(cap):
        return dict(_changed(cap, where=half), hand=half)
    outcome, script, surveys = _place(monkeypatch, capture=capture)
    assert (outcome.status, outcome.reason) == ("succeeded", "placed")
    assert outcome.evidence["post_release"]["verdict"] == "met" and surveys == []


def _rest_record(ctx, pid, *, top):
    part = ctx.scene.parts[pid]
    shift = np.array([-0.046, 0.206]) - np.asarray(part.region_xy).mean(axis=0)
    return {"record": dataclasses.replace(part, top_z=Meas(top, 0.002, "kappa"),
                                          region_xy=np.asarray(part.region_xy) + shift), "reason": None}


def test_pr1_an_occluded_region_falls_back_on_a_registration_within_d8(monkeypatch):
    expected = 0.9305 + (0.955 - 0.918)                                       # S_dest + the pick-time height
    outcome, _, surveys = _place(monkeypatch, hand_now=np.ones((512, 512), bool),
                                 survey=lambda ctx, pid: _rest_record(ctx, pid, top=expected + 0.002))
    assert (outcome.status, outcome.reason) == ("succeeded", "placed") and surveys == ["D"]
    assert outcome.evidence["post_release"]["stage"] == "registration"


def test_pr1_a_fallback_top_20_mm_low_or_a_failed_registration_is_not_placed(monkeypatch):
    expected = 0.9305 + (0.955 - 0.918)
    outcome, _, _ = _place(monkeypatch, hand_now=np.ones((512, 512), bool),
                           survey=lambda ctx, pid: _rest_record(ctx, pid, top=expected - 0.020))
    assert (outcome.status, outcome.reason) == ("failed", "object_moved_after_release")
    assert outcome.evidence["post_release"]["dz_m"] == pytest.approx(-0.020)
    outcome, _, _ = _place(monkeypatch, hand_now=np.ones((512, 512), bool),
                           survey=lambda ctx, pid: {"record": None, "reason": "part_not_found"})
    assert (outcome.status, outcome.reason) == ("failed", "object_moved_after_release")
    assert outcome.evidence["post_release_unverified"] is True and "post release unverified" in outcome.evidence_text


# ------------------------------------------------------------------------------------------------ SAL-R2 / IF-R3'
_ON_SURFACE = {"class": "on_surface", "S_dest": 0.9305, "bottom_z": 0.95, "dz_m": 0.0, "lateral_tol_m": 0.02,
               "thresholds": []}


def test_sal_r2_a_lateral_only_destination_with_a_wide_lateral_tolerance_is_flagged(monkeypatch):
    pre = {"ok": True, "cue": "lateral", "values": {"r_hull_m": 0.012}, "thresholds": []}
    outcome, _, _ = _place(monkeypatch, support=_ON_SURFACE, pre=pre)
    assert (outcome.status, outcome.reason) == ("succeeded", "placed")
    assert outcome.evidence["lateral_unverified"] is True
    assert outcome.evidence["destination_precondition"]["cue"] == "lateral"


def test_sal_r2_a_vertical_cue_or_a_large_hull_is_not_flagged(monkeypatch):
    for pre in ({"ok": True, "cue": "vertical", "values": {"r_hull_m": 0.012}, "thresholds": []},
                {"ok": True, "cue": "lateral", "values": {"r_hull_m": 0.040}, "thresholds": []}):
        outcome, _, _ = _place(monkeypatch, support=_ON_SURFACE, pre=pre)
        assert outcome.reason == "placed" and "lateral_unverified" not in outcome.evidence


def test_if_r3_a_tight_fit_is_flagged_and_carried_never_refused(monkeypatch):
    outcome, script, _ = _place(monkeypatch, aim_fits=False)
    assert (outcome.status, outcome.reason) == ("succeeded", "placed") and "carry" in script.names()
    assert outcome.evidence["tight_fit"] is True and "tight_fit" in outcome.evidence["flags"]
    outcome, _, _ = _place(monkeypatch, aim_fits=True)                        # negative: fits -> no flag
    assert outcome.reason == "placed" and "tight_fit" not in outcome.evidence
