"""BM acceptance tests for agent_speedup_v2_primitives (contract §C8 T1 items 1-9, plus the module's lint).

The primitives get their verdicts from injected observers (BO's module, built in parallel). These tests drive them on
the Wave-0 fakes (tests/agent_speedup_v2_fakes.py: the SP-fitted hinge wedge model, the colour-change region, the
round part, the second-order plant) with test-side observer STAND-INS: a compact restatement of the contract §C9
K1w / K4w / K7 wedge rules on the measured gap and yaw, and truth-driven part-rotation and effect observers (the fake's
hinge angle and colour-region truth; tests may read truth). Scripted observers pin each decision-rule branch exactly.
When BO's ``agent_speedup_v2_observers`` exists, the same physical cases also run with its WedgeObserver (Wave-1
acceptance). Test files may import v1 freely (contract §C1.5 item 5)."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

import agent_speedup_v2_fakes as F
from gpt_field import agent_speedup_v1 as v1
from gpt_field.action import TRANSLATION_SCALE_M
from gpt_field import agent_speedup_v2_primitives as P
from gpt_field import agent_speedup_v2_robot as robot_facts
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field.agent_speedup_v2_types import CalibrationRecord, ObsReport, RobotModel, Term, th_max, twist_about
from gpt_field.contact_skills import (AnchoredMove, Calibrate, GripperCalibration, GripperSettlePrimitive,
                                      HoldPrimitive, Orient, StepView, VerifyHold, WaypointReach, quaternion_angle_rad)
from gpt_field.speed_first_gripper import GripperSettleConfig
from gpt_field.observation import PolicyObservation
from gpt_field.wrist_servo import Align

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
ROBOT = RobotModel()
PAD_MM = robot_facts.PAD_LENGTH_R0_MM


# ================================================================================================ test drivers
def run(world, primitive, max_steps=600, before_step=None, after_step=None):
    """Drive a primitive on a fake (v1 protocol); returns (outcome, actions)."""
    view = StepView(world.observation())
    actions = []
    while not primitive.done and len(actions) < max_steps:
        if before_step is not None:
            before_step(world, primitive, len(actions))
        action = primitive.next_action(view)
        actions.append(action)
        after = StepView(world.step(action))
        primitive.observe_transition(view, action, after)
        if after_step is not None:
            after_step(world, primitive, len(actions))
        view = after
    assert primitive.done, f"{primitive.name} did not finish in {max_steps} steps"
    return primitive.outcome(), actions


def json_safe(value):
    json.dumps(value, allow_nan=False)
    return True


class Recorder:
    def __init__(self):
        self.calls = []

    def on_transition(self, stage, before, after):
        assert isinstance(before, StepView) and isinstance(after, StepView)
        self.calls.append((stage, before.state.copy(), after.state.copy()))

    @property
    def stages(self):
        return [c[0] for c in self.calls]


def world(scene=None, **kw):
    scene = scene or F.task2_like_scene()
    kw.setdefault("eef", F.TASK2_START_STATE8[:3])
    kw.setdefault("start_tilt_deg", 3.14)
    kw.setdefault("images", False)
    return F.FakeTwoViewSim(scene, **kw)


def knob_world(*, lever=True, free_sense=-1, gap_noise_mm=F.GAP_NOISE_MM, dynamics="kinematic", **scene_kw):
    """A rotary part clamped by the scripted truth grasp (hover, touch the disk, P10 retreat, close) and held closed
    for 8 static steps (SP)."""
    scene = F.rotary_part_scene(lever=lever, free_sense=free_sense, **scene_kw)
    w = F.FakeTwoViewSim(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False,
                         gap_noise_mm=gap_noise_mm, dynamics=dynamics)
    F.scripted_grasp_rotary(w)
    F.scripted_gripper(w, 1.0, 8)
    assert w.held() is not None and w.held()["kind"] == "rotary"
    return w


# ================================================================================== observer stand-ins (tests only)
def _yaw_about_r2(before, after):
    return twist_about(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2])


class StandInWedge:
    """Contract §C9 WedgeObserver rules restated for the tests: K1w (mode "sense": rates r_k, window origin 1 for a
    fresh close or k0 = the first non-negative rate after the unwedge fall, decisions at the 12th and 16th step after
    the origin, class "undecided" at the extend step), K4w (mode "end_stop": the 4-step windows ending at two
    consecutive steps >= max(0.5 l w, 5 x the median |rate| BEFORE the windows)), K7 slip (r <= -max(0.5 l w,
    4 sigma_g) for 2 steps outside the unwedge) and the hold floor. Decisions latch. Two choices differ from a literal
    reading of §C9 and are reported to BO: the K4w median excludes the windows themselves (else a stop within ~4
    steps of the start never fires: the post-stop rates become the median), and K7 has a 4 sigma_g noise floor (else
    gap noise trips it when the hand barely yaws)."""
    name = "wedge"

    def __init__(self, *, mode, sigma_g=0.002, hold_floor_mm=0.0, after_reversal=False, blocked_steps=None):
        self.mode, self.sigma_g, self.hold_floor_mm = mode, sigma_g, hold_floor_mm
        self.after_reversal, self.blocked_steps = after_reversal, blocked_steps
        self.started = 0

    def start(self, view):
        self.started += 1
        self.rates, self.omega, self.k0 = [], [], (None if self.after_reversal else 1)
        self.fell, self.latched, self.slip_run, self.step = False, None, 0, 0

    def _report(self, verdict, reason, **evidence):
        evidence.update(mode=self.mode, k=len(self.rates) - 1, k0=self.k0, rates=list(self.rates))
        return ObsReport(self.name, verdict, None, reason, evidence, (), self.step)

    def update(self, before, action, after, context=None):
        self.step += 1
        rate = after.gap_mm - before.gap_mm
        self.rates.append(rate)
        self.omega.append(abs(_yaw_about_r2(before, after)))
        k = len(self.rates) - 1
        lw = PAD_MM * float(np.mean(self.omega))
        if self.latched is not None:
            return self.latched
        if after.gap_mm < self.hold_floor_mm:
            return self._report("violated", "hold_lost")
        if self.k0 is None:
            if rate < 0:
                self.fell = True
            elif self.fell or k > 0:
                self.k0 = k
            if self.k0 is None and k + 1 >= math.ceil(1.5 * (self.blocked_steps or 12)):
                self.latched = self._report("failed", "sense_undecided")
                return self.latched
            if self.k0 is None:
                return self._report("pending", None, phase="unwedge")
        self.slip_run = self.slip_run + 1 if rate <= -max(0.5 * lw, 4 * self.sigma_g) else 0
        if self.slip_run >= 2:
            return self._report("violated", "turn_slip", lw=lw)
        if self.mode == "sense":
            n = k + 1 if not self.after_reversal else k - self.k0 + 1
            if n in (12, 16):
                early = float(np.mean(self.rates[self.k0:self.k0 + 4]))

                def late(end):
                    return float(np.mean(self.rates[end - 3:end + 1]))
                if late(k) <= max(0.1 * lw, 4 * self.sigma_g, 0.25 * early):
                    self.latched = self._report("met", "free_sense", early=early, late=late(k), lw=lw)
                    return self.latched
                if min(late(k), late(k - 1)) >= max(0.5 * lw, 0.75 * early):
                    self.latched = self._report("violated", "blocked_sense", early=early, late=late(k), lw=lw)
                    return self.latched
                if n == 16:
                    return self._report("pending", None, **{"class": "undecided"})
            return self._report("pending", None)
        if k >= 4:
            windows = [float(np.mean(self.rates[j - 3:j + 1])) for j in (k - 1, k)]
            baseline = np.abs(self.rates[:k - 4])                    # the rates before both windows
            limit = max(0.5 * lw, 5.0 * float(np.median(baseline)) if baseline.size else 0.0)
            if min(windows) >= limit:
                self.latched = self._report("met", "end_stop", windows=windows, threshold=limit)
                return self.latched
        return self._report("pending", None)


class TruthPartRotation:
    """Part rotation relative to the hand from the fake's hinge truth, over a trailing 4-step window (a rate measure:
    the backlash lag after a fresh close, ~5 deg in the free sense, is not a stop): once the hand yawed >= 3 deg in
    the window, a relative change >= 3 deg -> violated part_not_corotating, else met corotating."""
    name = "part_rotation"

    def __init__(self, w, group="knob", *, unmeasurable=False, floor_deg=3.0, window=4):
        self.world, self.group, self.unmeasurable = w, group, unmeasurable
        self.floor, self.window = math.radians(floor_deg), window
        self.started = 0

    def start(self, view):
        self.started += 1
        self.q0, self.hand, self.trace, self.step = self.world.joint(self.group), 0.0, [(0.0, 0.0)], 0

    def update(self, before, action, after, context=None):
        self.step += 1
        axis = self.world.scene.groups[self.group]["hinge"].axis
        self.hand += twist_about(before.quaternion_xyzw, after.quaternion_xyzw, axis)
        self.trace.append((self.hand, self.world.joint(self.group) - self.q0))
        if self.unmeasurable:
            return ObsReport(self.name, "failed", None, "unmeasurable", {}, (), self.step)
        if len(self.trace) <= self.window:
            return ObsReport(self.name, "pending", None, None, {}, (), self.step)
        (h0, p0), (h1, p1) = self.trace[-1 - self.window], self.trace[-1]
        evidence = {"hand_window_rad": h1 - h0, "relative_window_rad": (h1 - h0) - (p1 - p0)}
        if abs(h1 - h0) < self.floor:
            return ObsReport(self.name, "pending", None, None, evidence, (), self.step)
        if abs(evidence["relative_window_rad"]) >= self.floor:
            return ObsReport(self.name, "violated", None, "part_not_corotating", evidence, (), self.step)
        return ObsReport(self.name, "met", None, "corotating", evidence, (), self.step)


class TruthEffect:
    """Effect onset from the fake's colour-region truth: 3 consecutive changed frames and theta_part >= 5 deg at the
    confirming frame; theta_e and onset_step at the FIRST frame of the run ([C-28]); latches."""
    name = "effect"

    def __init__(self, w, region="burner_plate"):
        self.world, self.region = w, region
        self.started = 0
        self.contexts = []

    def start(self, view):
        self.started += 1
        self.run, self.step, self.met = [], 0, None

    def update(self, before, action, after, context=None):
        self.step += 1
        self.contexts.append(dict(context or {}))
        if self.met is not None:
            return self.met
        theta = (context or {}).get("theta_part_rad")
        changed = bool(self.world.colour_region_changed(self.region))
        self.run = self.run + [(self.step, theta, self.world.steps)] if changed else []
        if len(self.run) >= 3 and theta is not None and theta >= math.radians(5.0):
            step0, theta0, world_step0 = self.run[-3]
            self.met = ObsReport(self.name, "met", None, "effect_onset",
                                 {"onset_step": step0, "theta_e_rad": 0.0 if theta0 is None else theta0,
                                  "world_step": world_step0}, (), self.step)
            return self.met
        return ObsReport(self.name, "pending", None, None, {"run": len(self.run)}, (), self.step)


class Scripted:
    """Scripted reports by step {step: (verdict, reason, evidence)}; default pending."""

    def __init__(self, name, script=None, default=("pending", None, {})):
        self.name, self.script, self.default = name, dict(script or {}), default
        self.started, self.contexts = 0, []

    def start(self, view):
        self.started += 1
        self.step = 0

    def update(self, before, action, after, context=None):
        self.step += 1
        self.contexts.append(context)
        verdict, reason, evidence = self.script.get(self.step, self.default)
        return ObsReport(self.name, verdict, None, reason, dict(evidence), (), self.step)


def segment(w, sense, *, wedge=None, rotation=None, effect=None, floor=1.5, budget=1.9, **kw):
    view = StepView(w.observation())
    return P.TwistSegment(view, sense=sense, wedge=wedge or StandInWedge(mode="sense"),
                          part_rotation=TruthPartRotation(w) if rotation is None else rotation,
                          hold_floor_mm=floor, yaw_budget_left_rad=budget, robot=ROBOT, effect=effect, **kw)


NO_ROTATION = object()


def hinge(w, sense, *, wedge=None, rotation=None, effect=None, floor=1.5, target=None, budget=1.9, prior=None,
          sigma_c=0.005, capable=True, **kw):
    view = StepView(w.observation())
    prior = view.position[:2].copy() if prior is None else prior
    rotation = TruthPartRotation(w) if rotation is None else (None if rotation is NO_ROTATION else rotation)
    return P.HingeTwist(view, sense=sense, wedge_end=wedge or StandInWedge(mode="end_stop"), effect=effect,
                        part_rotation=rotation, hold_floor_mm=floor,
                        theta_target_rad=target, yaw_budget_left_rad=budget, centre_prior_xy=prior,
                        sigma_c_m=sigma_c, wedge_capable=capable, robot=ROBOT, **kw)


def free_classified(w, sense=-1):
    """Run the free-sense TwistSegment so a HingeTwist starts after a free classification (spec §4.9 step 7)."""
    out, _ = run(w, segment(w, sense))
    assert out["reason"] == "free_sense", out["reason"]
    return out


# ================================================================================================ 1. CalibratePlus
def calibrate(w, recorder=None, before_step=None):
    cal = P.CalibratePlus(StepView(w.observation()), recorder=recorder, robot=ROBOT)
    out, actions = run(w, cal, max_steps=400, before_step=before_step)
    return cal, out, actions


def test_calibrate_plus_stages_two_closes_steps_and_recorder():
    w = world()
    rec = Recorder()
    cal, out, actions = calibrate(w, rec)
    assert out["status"] == "succeeded" and out["reason"] == "calibrated_online"
    ran = [s["stage"] for s in out["evidence"]["stages"]]
    assert ran == ["close_1", "yaw_plus", "yaw_minus", "open_1", "open_hold", "close_2", "open_2", "probe_1",
                   "probe_2", "probe_3", "vertical_up_1", "settle_up", "vertical_up_2", "settle_top",
                   "vertical_down_1", "settle_down", "vertical_down_2", "settle"]
    assert ran == [s for s in P.CalibratePlus.STAGES if s in ran]                      # the contract's order
    assert out["evidence"]["close_count"] == 2 and not out["evidence"]["third_close"]
    assert len(actions) == out["actions"] <= 220
    # the recorder sees every transition, labelled with its stage, in order
    assert len(rec.calls) == len(actions)
    assert set(rec.stages) <= set(P.CalibratePlus.STAGES)
    assert [s for i, s in enumerate(rec.stages) if i == 0 or rec.stages[i - 1] != s] == ran
    # the fingertip-band pair: the last open_hold frame and the end of close_2 share the pose
    last_hold = [c for c in rec.calls if c[0] == "open_hold"][-1][2]
    end_close2 = [c for c in rec.calls if c[0] == "close_2"][-1][2]
    np.testing.assert_allclose(last_hold[:6], end_close2[:6], atol=1e-3)
    assert last_hold[6] - last_hold[7] > 0.06 and end_close2[6] - end_close2[7] < 0.004  # open vs closed
    # the probe moves net to <= 2 mm, the orientation back at the start
    assert np.linalg.norm(out["evidence"]["final_offset_m"]) <= 0.002
    assert abs(out["evidence"]["final_yaw_rad"]) <= math.radians(0.5)
    assert json_safe(out) and cal.pointing_request is None


def test_calibrate_plus_record_and_gripper_quantities():
    w = world(gap_noise_mm=F.GAP_NOISE_MM)
    cal, out, _ = calibrate(w)
    record = cal.record()
    assert isinstance(record, CalibrationRecord) and json_safe(record.to_dict())
    assert len(record.close_curves_mm) == 2 and len(record.k15_mm) == 2
    assert record.gripper.source == "online" and len(record.gripper.empty_curve_mm) == P.CAL_CLOSE_ACTIONS
    assert [m["stage"] for m in record.probe_moves] == ["probe_1", "probe_2", "probe_3"]
    for i, (move, nominal) in enumerate(zip(record.probe_moves, P.CAL_PROBE_MOVES)):
        assert move["nominal_m"] == list(nominal) and len(move["start_state8"]) == 8
        # the commanded offset is the goal (relative to the calibration start) minus this stage's start
        goal = cal.start_xyz + np.sum(np.asarray(P.CAL_PROBE_MOVES[:i + 1]), axis=0)
        np.testing.assert_allclose(move["commanded_m"], goal - np.asarray(move["start_state8"][:3]), atol=1e-12)
        assert np.dot(move["world_displacement_m"], move["commanded_m"]) > 0.6 * np.dot(move["commanded_m"],
                                                                                           move["commanded_m"])
        assert [t["name"] for t in move["thresholds"]] == ["G9_open_finger_contact_mm"]
        assert move["thresholds"][0]["value"] == P.G9_FINGER_CONTACT_MM
    # decision §2.5 M-2: the +-4 cm vertical probe runs as two 2 cm halves per direction, a B4 settle between
    up_1, up_2, down_1, down_2 = record.vertical
    half = P.CAL_VERTICAL_PROBE_M / P.CAL_VERTICAL_HALVES
    assert [m["stage"] for m in record.vertical] == ["vertical_up_1", "vertical_up_2", "vertical_down_1",
                                                     "vertical_down_2"]
    for move in record.vertical:
        assert abs(move["world_displacement_m"][2]) >= half - 0.001 and move["reason"] == "travel_reached"
    # the up halves hold the up start xy (a pure vertical probe), the first half goes exactly 2 cm and the second
    # to 4 cm above the up start; the down halves return to the calibration start, the second to its height
    assert up_1["commanded_m"] == pytest.approx([0.0, 0.0, half])
    up_start = np.asarray(up_1["start_state8"][:3])
    for move in (up_1, up_2):
        np.testing.assert_allclose(move["lateral_target_xyz"], up_start)
        assert np.linalg.norm(np.subtract(move["end_state8"][:2], up_start[:2])) <= 0.001
    assert up_2["commanded_m"][2] == pytest.approx(up_start[2] + P.CAL_VERTICAL_PROBE_M - up_2["start_state8"][2])
    assert up_2["end_state8"][2] - up_start[2] >= P.CAL_VERTICAL_PROBE_M - 0.001
    for move in (down_1, down_2):
        np.testing.assert_allclose(move["lateral_target_xyz"], cal.start_xyz)
    assert down_1["end_state8"][2] - cal.start_xyz[2] == pytest.approx(half, abs=0.001)
    np.testing.assert_allclose(down_2["commanded_m"], cal.start_xyz - np.asarray(down_2["start_state8"][:3]),
                               atol=1e-9)
    assert abs(down_2["end_state8"][2] - cal.start_xyz[2]) <= 0.001
    q = P.gripper_quantities(record)
    # the closed empty hand in free air yaws rotation_response_rad(K2) per step on the fake (SP: 1.31 deg/step)
    fake_rate = F.rotation_response_rad(P.K2_TWIST_RAD)
    assert abs(q["free_yaw_rate_rad_per_step"] / fake_rate - 1.0) <= 0.10
    assert q["yaw_samples"] >= 4
    # sigma_g ~ the fake's gap-rate noise (sqrt 2 x the per-reading gap noise)
    noise = math.sqrt(2.0) * F.GAP_NOISE_MM
    assert 0.5 * noise <= q["sigma_g_mm_per_step"] <= 2.5 * noise
    assert q["sigma_close_mm"] >= P.G4_SIGMA_FLOOR_MM
    np.testing.assert_allclose(q["sigma_close_raw_mm"], abs(record.k15_mm[0] - record.k15_mm[1]) / math.sqrt(2))
    assert q["close_arbitration"]["used"] == [0, 1] and q["closes_agree"] and out["evidence"]["closes_agree"]
    names = [t.name for t in q["thresholds"]]
    assert names == ["G1_open_gap_mm", "G2_empty_k15_mm", "G3_touch_threshold_mm", "G4_sigma_close_mm",
                     "sigma_g_mm_per_step"]
    assert all(t.label in th.LABEL_CLASSES and t.derivation for t in q["thresholds"])
    assert q["thresholds"][0].value == record.gripper.open_gap_mm and q["thresholds"][0].label == "ONLINE"
    assert q["thresholds"][2].value == pytest.approx(record.gripper.touch_threshold_mm)


def _close_offsets(offsets):
    """before_step hook: inject ``offsets[stage]`` mm into the next new empty close while ``stage`` runs (the fake
    consumes a close_offset_mm injection at the start of the next close)."""
    def inject(w, cal, n):
        done = w.__dict__.setdefault("_bm_injected", set())
        if cal.stage in offsets and cal.stage not in done:
            w.inject("close_offset_mm", mm=offsets[cal.stage])
            done.add(cal.stage)
    return inject


def third_close_world():
    """The task-2-like fake with the opening capped at the start of the recorded close curve (77.4 mm). Without the
    cap, close_3 starts after the 20-step open_2 at ~78.2 mm, and the fake's empty close from above the recorded start
    lags behind the curve for good (it ends at 7-9 mm instead of ~1.2 mm; cross-module review finding BM-R2, owner
    BF), which would make close_3 a spurious outlier."""
    return world(open_limit_mm=F.CLOSE_CURVE_MM[0])


def test_calibrate_plus_third_close_arbitrates_the_outlier():
    """Contract §C8 T1 item 1 (a third close when the second is injected 0.8 mm off) and spec §4.5 item 2 (the third
    close decides between the first two): close_3 agrees with close_1, so the empty curve and sigma_close come from
    that pair and the off close_2 is excluded; G5 and G6 stay at the two-close scale."""
    w = third_close_world()
    cal, out, actions = calibrate(w, before_step=_close_offsets({"yaw_plus": 0.8}))
    ran = [s["stage"] for s in out["evidence"]["stages"]]
    assert ran[5:9] == ["close_2", "open_2", "close_3", "open_3"]
    assert out["evidence"]["third_close"] and out["evidence"]["close_count"] == 3
    assert out["reason"] == "calibrated_online" and len(actions) <= 260
    record = cal.record()
    k15 = record.k15_mm
    assert len(k15) == 3 and abs(k15[0] - k15[1]) > P.G5_FLOOR_MM
    assert abs(k15[2] - k15[0]) <= P.G5_FLOOR_MM                     # close_3 is a clean close (the fake behaves)
    arbitration = out["evidence"]["close_arbitration"]
    assert (arbitration["used"], arbitration["excluded"], arbitration["agree"]) == ([0, 2], [1], True)
    assert out["evidence"]["closes_agree"]
    # the empty curve is the mean of the agreeing pair: the outlier close is not in it
    pair = np.mean([record.close_curves_mm[0], record.close_curves_mm[2]], axis=0)
    np.testing.assert_allclose(record.gripper.empty_curve_mm, pair)
    q = P.gripper_quantities(record)
    assert q["close_arbitration"] == arbitration and q["closes_agree"]
    sigma_pair = abs(k15[0] - k15[2]) / math.sqrt(2.0)
    np.testing.assert_allclose(q["sigma_close_raw_mm"], sigma_pair)
    assert q["sigma_close_mm"] == max(sigma_pair, P.G4_SIGMA_FLOOR_MM)
    assert "closes 1+3 of 3" in q["thresholds"][3].derivation
    g5 = P.g5_present_threshold(record.gripper, q["sigma_close_mm"])
    g6 = P.g6_hold_floor(record.gripper, q["sigma_close_mm"])
    assert g5.value == pytest.approx(max(P.G5_FLOOR_MM, P.G5_SIGMA_K * q["sigma_close_mm"])) and g5.value <= 1.0
    assert g6.value == pytest.approx(record.gripper.empty_floor_mm + max(P.G6_FLOOR_MM, P.G6_SIGMA_K * sigma_pair))
    assert g6.value - record.gripper.empty_floor_mm <= 1.0
    assert json_safe(out)


def test_calibrate_plus_three_disagreeing_closes_use_the_closest_pair():
    """No pair of the three closes agrees within G5_FLOOR_MM: the closest pair sets the curve and sigma_close and
    the evidence says closes_agree False (the relation records calibration_degraded)."""
    w = third_close_world()
    cal, out, _ = calibrate(w, before_step=_close_offsets({"yaw_plus": 0.8, "open_2": 1.8}))
    k15 = cal.record().k15_mm
    assert out["evidence"]["close_count"] == 3 and min(abs(k15[0] - k15[1]), abs(k15[0] - k15[2]),
                                                        abs(k15[1] - k15[2])) > P.G5_FLOOR_MM
    arbitration = out["evidence"]["close_arbitration"]
    assert arbitration["used"] == [0, 1] and arbitration["excluded"] == [2] and not arbitration["agree"]
    assert out["evidence"]["closes_agree"] is False and out["reason"] == "calibrated_online"
    q = P.gripper_quantities(cal.record())
    assert not q["closes_agree"] and "no pair within G5_FLOOR_MM" in q["thresholds"][3].derivation
    np.testing.assert_allclose(q["sigma_close_raw_mm"], abs(k15[0] - k15[1]) / math.sqrt(2.0))


def test_close_arbitration_and_sigma_close_from_records():
    """The review cases: one absurd outlier, a clean third close equal to close 1, no agreeing pair."""
    assert P.close_arbitration((2.35, 3.15, 2.35)) == {"used": [0, 2], "excluded": [1], "delta_mm": 0.0,
                                                        "agree": True}
    outlier = P.close_arbitration((2.348, 3.114, 10.629))
    assert outlier["used"] == [0, 1] and outlier["excluded"] == [2] and not outlier["agree"]
    assert P.close_arbitration((2.35, 2.40))["used"] == [0, 1] and P.close_arbitration((2.35, 2.40))["agree"]
    assert P.close_arbitration((2.35,)) == {"used": [0], "excluded": [], "delta_mm": None, "agree": False}
    w = world()
    cal, _, _ = calibrate(w)
    import dataclasses
    clean = P.gripper_quantities(dataclasses.replace(cal.record(), k15_mm=(2.35, 3.15, 2.35)))
    assert clean["sigma_close_raw_mm"] == 0.0 and clean["sigma_close_mm"] == P.G4_SIGMA_FLOOR_MM
    g6 = P.g6_hold_floor(cal.record().gripper, clean["sigma_close_mm"])
    assert g6.value == pytest.approx(cal.record().gripper.empty_floor_mm + P.G6_FLOOR_MM)
    wild = P.gripper_quantities(dataclasses.replace(cal.record(), k15_mm=(2.348, 3.114, 10.629)))
    assert wild["sigma_close_raw_mm"] == pytest.approx(abs(2.348 - 3.114) / math.sqrt(2.0))
    assert P.g6_hold_floor(cal.record().gripper, wild["sigma_close_mm"]).value < cal.record().gripper.empty_floor_mm + 4


def test_calibrate_plus_default_without_closing_motion():
    w = world(gripper_stuck=True)
    cal, out, _ = calibrate(w)
    assert out["status"] == "succeeded" and out["reason"] == "calibrated_default"
    assert cal.record().gripper == GripperCalibration.default()
    assert json_safe(out)


VERTICAL_HALVES = ["vertical_up_1", "vertical_up_2", "vertical_down_1", "vertical_down_2"]


def test_calibrate_plus_vertical_halves_give_5_locatable_moves_and_7_external_samples():
    """Decision §2.5 M-2 / §5.1 item 4 (rulings item 5): the vertical probe split into two 2 cm halves per direction
    with a B4 settle between them. On the rendered task-2-like fake with BG's real SurveyRecorder, ExternalModel and
    HandLocator: 7 external samples (3 probe moves + 4 halves), 5 locatable moves (probe_1 + the 4 halves; probe_2 and
    probe_3 run across v_hat), every half starts from rest, and the purpose of the split holds: one location corrupted
    by 40 px among the 5 is singled out and dropped (with 3 it could not be, rulings item 5)."""
    from gpt_field import agent_speedup_v2_geometry as G

    w = F.FakeTwoViewSim(F.task2_like_scene(), eef=F.TASK2_START_STATE8[:3], start_tilt_deg=3.14)
    view = StepView(w.observation())
    recorder = G.SurveyRecorder(view, ROBOT, camera=None, external=True)
    stage_rec = Recorder()

    class Both:
        def on_transition(self, stage, before, after):
            recorder.on_transition(stage, before, after)
            stage_rec.on_transition(stage, before, after)
    cal = P.CalibratePlus(view, recorder=Both(), robot=ROBOT)
    out, _ = run(w, cal, max_steps=400)
    assert out["reason"] == "calibrated_online"
    # every vertical half after the first starts after a B4 settle (from rest)
    stages = stage_rec.stages
    starts = {s: i for i, s in reversed(list(enumerate(stages))) if s in VERTICAL_HALVES}
    for half in VERTICAL_HALVES[1:]:
        assert stages[starts[half] - 1].startswith("settle"), (half, stages[starts[half] - 1])
    result = recorder.result(None)
    samples = result.external_samples
    assert [s["stage"] for s in samples] == ["probe_1", "probe_2", "probe_3"] + VERTICAL_HALVES
    model = G.ExternalModel.fit(samples)
    assert model is not None

    def motion(sample):
        return model.J @ (np.asarray(sample["end_state8"][:3]) - np.asarray(sample["start_state8"][:3]))
    locatable = [s["stage"] for s in samples if G.HandLocator.motion_check(model, motion(s)) is None]
    assert locatable == ["probe_1"] + VERTICAL_HALVES
    for s in samples:
        if s["stage"] in VERTICAL_HALVES:                                   # each 2 cm half clears the along bound
            assert abs(float(motion(s) @ model.v_hat)) >= G.L1_MIN_ALONG_PX
    locator = G.HandLocator.fit(samples, model, ROBOT)
    assert locator is not None and [row["stage"] for row in locator.located] == locatable
    for bad in locatable:
        corrupted = [dict(s, mask_extreme_uv=(np.asarray(s["mask_extreme_uv"]) + [40.0, 0.0]).tolist())
                     if s["stage"] == bad else s for s in samples]
        refit = G.HandLocator.fit(corrupted, model, ROBOT)
        assert refit is not None and [row["stage"] for row in refit.located] == [s for s in locatable if s != bad]
    assert json_safe(out)


def test_g5_g6_thresholds():
    gripper = GripperCalibration.default()
    low = P.g5_present_threshold(gripper, 0.01)
    assert low.name == "G5_object_present_gap_mm" and low.value == P.G5_FLOOR_MM and low.label == "DD"
    high = P.g5_present_threshold(gripper, 0.3)
    assert high.value == pytest.approx(4 * 0.3) and high.label == "ONLINE" and "G5_FLOOR_MM" in high.derivation
    floor = P.g6_hold_floor(gripper, 0.01)
    assert floor.name == "G6_hold_floor_mm" and floor.value == pytest.approx(gripper.empty_floor_mm + 0.5)
    assert P.g6_hold_floor(gripper, 0.2).value == pytest.approx(gripper.empty_floor_mm + 1.2)


# ================================================================================================ 2. ParallaxProbe
@pytest.mark.parametrize("dynamics", ["kinematic", "second_order"])
def test_parallax_probe_legs_and_net(dynamics):
    w = world(F.planar_region_scene(), eef=(-0.20, 0.08, 1.05), start_tilt_deg=0.0, dynamics=dynamics)
    rec = Recorder()
    view = StepView(w.observation())
    probe = P.ParallaxProbe(view, recorder=rec, gripper="open")
    out, actions = run(w, probe)
    assert out["status"] == "succeeded", out
    legs = out["evidence"]["legs"]
    assert [leg["stage"] for leg in legs] == ["r1_out", "r1_back", "r0_out", "r0_back"]
    for leg in legs:                                               # the second-order arm coasts ~1.6 mm past a leg
        assert leg["reason"] == "travel_reached" and abs(leg["travel_m"] - P.D2_PROBE_M) <= 0.0025
    r1, r0 = view.R[:, 1], view.R[:, 0]
    out_r1 = np.subtract(legs[0]["end_xyz"], legs[0]["start_xyz"])
    out_r0 = np.subtract(legs[2]["end_xyz"], legs[2]["start_xyz"])
    assert out_r1 @ r1 >= 0.019 and abs(out_r1 @ r0) <= 0.002
    assert out_r0 @ r0 >= 0.019 and abs(out_r0 @ r1) <= 0.002
    assert out["evidence"]["net_displacement_norm_m"] <= 0.001
    assert rec.stages[0] == "settle" and set(rec.stages) == set(P.ParallaxProbe.STAGES)
    legs_in_order = [s for i, s in enumerate(rec.stages) if s != "settle" and (i == 0 or rec.stages[i - 1] != s)]
    assert legs_in_order == ["r1_out", "r1_back", "r0_out", "r0_back"]
    starts = [i for i, s in enumerate(rec.stages) if s != "settle" and rec.stages[i - 1] == "settle"]
    assert len(starts) == 4                                          # every leg starts after a settle (from rest)
    assert len(rec.calls) == len(actions) and json_safe(out)
    # the open-finger touch check is on by default (G3 floor), and every compared limit is recorded
    assert probe.touch_threshold_mm == P.G3_FLOOR_MM and out["evidence"]["touch_threshold_mm"] == P.G3_FLOOR_MM
    names = [t["name"] for t in out["evidence"]["thresholds"]]
    assert names == ["D2_probe_m", "G3_touch_threshold_mm"] + ["G12_stall_ratio"] * 4
    for leg in legs:
        (g12,) = leg["thresholds"]
        assert g12["name"] == "G12_stall_ratio" and g12["label"] in ("ONLINE", "DD")
        low, high = P._HoverProbeLeg.FREE_RATIO_BOUNDS                     # RC-8: the hover leg's own clip
        assert P.G12_STALL_FRACTION * low <= g12["value"] <= P.G12_STALL_FRACTION * high
        assert leg["stage"] in g12["derivation"] and "stall_min_steps=5" in g12["derivation"]


def test_parallax_probe_contact_on_an_obstacle():
    wall = F.FakeBody("wall", "aabb", (-0.2, -0.075, 0.9), (0.10, 0.02, 0.2), texture_std=5.0)
    w = world(F.FakeScene([wall]), eef=(-0.2, 0.0, 1.0), start_tilt_deg=0.0)
    out, _ = run(w, P.ParallaxProbe(StepView(w.observation()), recorder=None, gripper="open"))   # default G3 floor
    assert out["status"] == "failed" and out["reason"] == "probe_contact"
    assert out["evidence"]["legs"][-1]["reason"] in ("contact_touch", "contact_stall")
    g3 = [t for t in out["evidence"]["thresholds"] if t["name"] == "G3_touch_threshold_mm"]
    assert len(g3) == 1 and g3[0]["value"] == P.G3_FLOOR_MM and g3[0]["label"] == "DD"
    # a calibrated G3 passed in as the gripper_quantities Threshold is used and recorded as is
    calibrated = th_max("G3_touch_threshold_mm", Term("DD", P.G3_FLOOR_MM, "G3_FLOOR_MM"),
                        Term("ONLINE", 0.12, "G3_SIGMA_K*sigma_in"), unit="mm")
    w = world(F.FakeScene([wall]), eef=(-0.2, 0.0, 1.0), start_tilt_deg=0.0)
    probe = P.ParallaxProbe(StepView(w.observation()), recorder=None, gripper="open", touch_threshold_mm=calibrated)
    assert probe.touch_threshold_mm == 0.12 and probe.g3 is calibrated
    closed = P.ParallaxProbe(StepView(w.observation()), recorder=None, gripper="close")
    out, _ = run(w, closed)
    assert "G3_touch_threshold_mm" not in [t["name"] for t in out["evidence"]["thresholds"]]  # no touch when closed


# ================================================================================ 2b. RC-8 hover legs (decision M-1)
class RatioPlant:
    """Remedy 2 T1 test 9's fake plant: step k realizes ``ratios[k]`` x the commanded translation, so a leg's own
    realized / commanded ratio is exactly the recorded one; the orientation is held and the fingers stay put except
    for ``inward_mm`` {step: mm} on finger 1 (an open-finger touch)."""

    def __init__(self, ratios, *, inward_mm=None, start=(-0.2, 0.0, 1.0)):
        self.ratios = [float(r) for r in ratios]
        self.inward_mm = dict(inward_mm or {})
        self.k = 0
        self.state = np.array([*start, math.pi, 0.0, 0.0, 0.04, -0.04])

    @property
    def exhausted(self):
        return self.k >= len(self.ratios)

    def observation(self):
        blank = F._blank_url()
        return PolicyObservation(blank, blank, tuple(float(x) for x in self.state), "synthetic")

    def step(self, action):
        ratio = self.ratios[self.k]
        self.k += 1
        self.state[:3] += ratio * np.asarray(action[:3], dtype=float) * TRANSLATION_SCALE_M
        self.state[6] -= self.inward_mm.get(self.k, 0.0) / 1000.0
        return self.observation()


REPLAY_TRAVEL_M = 1.0      # longer than any recorded leg: only the end of the trace or the stall rule ends a replay


def replay(ratios, build, *, axis=(0.0, 1.0, 0.0), inward_mm=None):
    """Feed one recorded ratio sequence through a leg on the RatioPlant; stops at the leg's end or the trace's."""
    plant = RatioPlant(ratios, inward_mm=inward_mm)
    view = StepView(plant.observation())
    leg = build(view, axis)
    while not leg.done and not plant.exhausted:
        action = leg.next_action(view)
        after = StepView(plant.step(action))
        leg.observe_transition(view, action, after)
        view = after
    np.testing.assert_allclose(leg.ratios, list(ratios)[:len(leg.ratios)], rtol=1e-9, atol=1e-12)
    return leg


def hover_leg(view, axis):
    return P.hover_probe_leg(view, axis=axis, travel_m=REPLAY_TRAVEL_M, gripper="open")


def as_built_leg(view, axis):
    """The Wave-1 ParallaxProbe leg: v1's stall rule from step 3, the free ratio clipped to [0.18, 0.40]."""
    return P._ProbeLeg(view, axis=axis, travel_m=REPLAY_TRAVEL_M, gripper="open", command_m=P.P15_COMMAND_M,
                       stop_on_contact=True, touch_threshold_mm=P.G3_FLOOR_MM, lateral_cap_m=P.P15_LATERAL_CAP_M,
                       orientation_xyzw=view.quaternion_xyzw, lateral_target=view.position,
                       arm_stall_after_motion=True)


def stall_step(leg):
    return len(leg.ratios) if leg.done and leg.reason == "contact_stall" else None


def collapse_step(ratios):
    """remedy 2 stall_floor.py: the 1-based step whose ratio first falls below 0.1 after one at or above it."""
    return next((i + 1 for i, r in enumerate(ratios) if i >= 1 and r < 0.1 and ratios[i - 1] >= 0.1), None)


def test_hover_probe_leg_class_is_rc8():
    """Decision §2.5 M-1: _HoverProbeLeg = _ProbeLeg with STALL_MIN_STEPS = PROBE_STALL_MIN_STEPS and the free ratio
    clipped to (PROBE_FREE_RATIO_FLOOR, v1's upper bound); the v1 / Calibrate+ / GuardedLower / Touchdown legs keep
    v1's rule."""
    assert issubclass(P._HoverProbeLeg, P._ProbeLeg) and issubclass(P._ProbeLeg, AnchoredMove)
    assert P._HoverProbeLeg.STALL_MIN_STEPS == P.PROBE_STALL_MIN_STEPS == 5
    assert P._HoverProbeLeg.FREE_RATIO_BOUNDS == (P.PROBE_FREE_RATIO_FLOOR, AnchoredMove.FREE_RATIO_BOUNDS[1])
    assert P.PROBE_FREE_RATIO_FLOOR == 0.08
    # the floor sits between a leg starting in contact (<= 0.04 must stall: 0.04 < 0.6 x floor) and the slowest free
    # leg's own ratio (0.083, which must not)
    assert 0.04 < P.G12_STALL_FRACTION * P.PROBE_FREE_RATIO_FLOOR and P.PROBE_FREE_RATIO_FLOOR < 0.083
    for cls in (P._ProbeLeg, P.GuardedLower, P.Touchdown):
        assert cls.STALL_MIN_STEPS == AnchoredMove.STALL_MIN_STEPS
        assert cls.FREE_RATIO_BOUNDS == AnchoredMove.FREE_RATIO_BOUNDS == P.G12_RATIO_BOUNDS


def test_hover_probe_leg_replay_76_free_legs_and_4_contacts():
    """Decision §5.1 item 4 = remedy 2 T1 test 9: the 76 recorded robot-only free-air leg ratio sequences (44 complete
    no-stall legs of the rulings lens, 32 legs truncated at their as-built false stall in the grasp lens, among them
    the 4 slow episode-5 legs) and the 4 real open-hand contacts, fed through the hover leg on a plant that realizes
    the recorded ratios: 0 false stalls; every contact stalls at its collapse step or the next; a leg starting in
    contact (ratios <= 0.04) stalls at step 5. The as-built leg reproduces the defect on the same traces (31-33).

    Coverage (review finding BM-RV-3): a hover leg takes no stall decision before step PROBE_STALL_MIN_STEPS, so the
    28 truncated legs that hold only 3-4 ratios cannot stall by construction; the RC-8 rule is exercised on the 48
    legs that reach step 5 (the 44 complete legs and 4 truncated 5-ratio legs), every one of which measures its own
    ratio and is compared against it. The RS-3 real-sim legs below add complete legs near the stall limit."""
    assert len(STALL_REPLAY_FREE_LEGS) == 76 and len(STALL_REPLAY_CONTACTS) == 4
    assert sum(1 for leg in STALL_REPLAY_FREE_LEGS if leg[4]) == 32
    decided = []
    for leg in STALL_REPLAY_FREE_LEGS:
        replayed = replay(leg[5], hover_leg)
        assert stall_step(replayed) is None, leg[:4]
        assert replayed.free_ratio_measured == (len(leg[5]) >= P.PROBE_STALL_MIN_STEPS), leg[:4]
        if replayed.free_ratio_measured:
            decided.append(leg[:5])
    assert len(decided) == 48 and sum(1 for leg in decided if not leg[4]) == 44     # 44 complete + 4 truncated
    # control: the as-built leg on the same traces stalls on 3 complete rulings legs (margins -0.6 to -1.0 %, far from
    # float noise) and on 28-30 truncated ones: remedy 2's own replay arithmetic counts 30, but two truncated legs read
    # 0.156, recorded to 3 decimals, exactly v1's arming level 0.6 x 0.26, where the float comparison of the real
    # AnchoredMove can fall just short (today it does on both), so those two are float-sensitive by construction
    arming = P.G12_STALL_FRACTION * AnchoredMove.DEFAULT_FREE_RATIO
    boundary = [leg for leg in STALL_REPLAY_FREE_LEGS if leg[4] and any(abs(r - arming) < 1e-6 for r in leg[5])]
    assert len(boundary) == 2
    as_built = [leg[:5] for leg in STALL_REPLAY_FREE_LEGS if stall_step(replay(leg[5], as_built_leg)) is not None]
    assert sum(1 for leg in as_built if not leg[4]) == 3
    assert 30 - len(boundary) <= sum(1 for leg in as_built if leg[4]) <= 30
    delays = []
    for ep, pid, ratios in STALL_REPLAY_CONTACTS:
        collapse = collapse_step(ratios)
        k = stall_step(replay(ratios, hover_leg, axis=(0.0, 0.0, -1.0)))
        assert collapse is not None and k is not None, (ep, pid)
        delays.append((collapse, k))
        assert collapse <= k <= collapse + 1, (ep, pid, collapse, k)
    assert delays == [(20, 20), (20, 21), (20, 21), (21, 22)]           # remedy 2 §2.5, unchanged by RC-8
    for start_in_contact in ([0.04] * 8, [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.01, 0.02]):
        leg = replay(start_in_contact, hover_leg)
        assert stall_step(leg) == P.PROBE_STALL_MIN_STEPS and leg.free_ratio == P.PROBE_FREE_RATIO_FLOOR


def test_hover_probe_leg_replay_rs3_real_legs_near_the_limit():
    """Review finding BM-RV-3: complete real-sim RC-8 legs from RS-3 run 002 (the 6 closest to their stall limit and
    the 6 slowest of 500 legs) replay with 0 stalls, each deciding against its own measured ratio; the as-built leg
    false-stalls 5 of the slow legs at step 5. The stall-off reference descents onto the plate and the table stall
    at their collapse step (plate) or the next (table), as in the live run."""
    assert len(STALL_REPLAY_RS3_FREE_LEGS) == 12 and len(STALL_REPLAY_RS3_CONTACTS) == 2
    as_built = 0
    for ep, pid, stage, ratios in STALL_REPLAY_RS3_FREE_LEGS:
        leg = replay(ratios, hover_leg)
        assert stall_step(leg) is None and leg.free_ratio_measured, (ep, pid, stage)
        windows = [np.mean(ratios[k - 2:k]) for k in range(P.PROBE_STALL_MIN_STEPS, len(ratios) + 1)]
        assert min(windows) > 1.4 * P.G12_STALL_FRACTION * leg.free_ratio, (ep, pid, stage)    # >= 40 % margin
        as_built += stall_step(replay(ratios, as_built_leg)) is not None
    assert as_built == 5
    assert min(min(ratios[2:5]) for *_, ratios in STALL_REPLAY_RS3_FREE_LEGS) < P.G12_RATIO_BOUNDS[0]
    delays = {}
    for ep, site, ratios in STALL_REPLAY_RS3_CONTACTS:
        collapse, k = collapse_step(ratios), stall_step(replay(ratios, hover_leg, axis=(0.0, 0.0, -1.0)))
        assert collapse is not None and k is not None and collapse <= k <= collapse + 1, (ep, site, collapse, k)
        delays[site] = k - collapse
    assert delays == {"plate": 0, "table": 1}


def short_hover_leg(view, axis):
    """A 3 cm hover leg (the arrival descent's HOVER_LOCATE_M) with the touch stop out of reach: the stall rule alone,
    as for a closed hand, and a real budget (ceil(0.03 / 1.5 mm) + 10 = 30 steps)."""
    return P.hover_probe_leg(view, axis=axis, travel_m=0.03, gripper="open", touch_threshold_mm=1000.0)


def test_hover_probe_leg_early_contact_boundary_is_the_decisions_trade_off():
    """Review finding BM-RV-2 (the decision's rule kept, its boundary pinned): the own free ratio is the median of
    steps 3-5 even when they are in contact, clipped at PROBE_FREE_RATIO_FLOOR, so the stall limit of a leg in contact
    from the start is 0.6 x 0.08 = 0.048. A contact from the start that still yields 0.05-0.10 is never stalled and
    ends on its budget (a stop other than travel_reached, i.e. the caller's transit_contact); <= 0.048 stalls at step
    5; a contact at step 4 stalls at step 5, one step later than the as-built leg; a slow free start (0.083) followed by
    a 0.055 contact runs to the budget, where the as-built leg stalls at step 5."""
    limit = P.G12_STALL_FRACTION * P.PROBE_FREE_RATIO_FLOOR
    assert limit == pytest.approx(0.048)
    for ratio in (0.02, 0.04, 0.047):
        leg = replay([ratio] * 60, short_hover_leg)
        assert stall_step(leg) == P.PROBE_STALL_MIN_STEPS, ratio
    for ratio in (0.05, 0.06, 0.08, 0.10):
        leg = replay([ratio] * 60, short_hover_leg)
        assert leg.done and leg.reason != "contact_stall", ratio
        assert leg.free_ratio_measured and leg.free_ratio == pytest.approx(max(ratio, P.PROBE_FREE_RATIO_FLOOR))
        if ratio < 0.1:                                          # 30 steps x 10 mm x ratio < 3 cm: the budget ends it
            assert leg.max_steps == 30
            assert (leg.status, leg.reason, len(leg.ratios)) == ("failed", "budget", 30), ratio
    contact_at_4 = [0.11, 0.22, 0.24, 0.03] + [0.02] * 20
    assert stall_step(replay(contact_at_4, short_hover_leg)) == P.PROBE_STALL_MIN_STEPS
    assert stall_step(replay(contact_at_4, as_built_leg)) == 4
    contact_at_5 = [0.11, 0.22, 0.24, 0.24, 0.03] + [0.02] * 20
    assert stall_step(replay(contact_at_5, short_hover_leg)) == stall_step(replay(contact_at_5, as_built_leg)) == 5
    slow_then_contact = [0.083] * 4 + [0.055] * 40
    leg = replay(slow_then_contact, short_hover_leg)
    assert (leg.reason, len(leg.ratios)) == ("budget", 30)
    assert stall_step(replay(slow_then_contact, as_built_leg)) == 5
    # with the gripper open the calibrated G3 touch still stops such a leg at once
    touched = replay([0.05] * 60, hover_leg, inward_mm={6: 0.5})
    assert (touched.reason, len(touched.ratios)) == ("contact_touch", 6)


def test_hover_probe_leg_touch_stops_at_once_and_g12_records_its_own_rule():
    """G3 still stops an open hover leg at once (decision §2.5 M-1); the G12 record carries the leg's own clip and
    minimum step, so a slow free leg's measured 0.083 is recorded as 0.6 x 0.083, not v1's floor."""
    leg = replay([0.25] * 10, hover_leg, inward_mm={1: 0.5})
    assert leg.reason == "contact_touch" and len(leg.ratios) == 1 and leg.contact == "touch"
    fresh = hover_leg(StepView(RatioPlant([0.25]).observation()), (0.0, 1.0, 0.0))
    g12 = P.g12_stall_threshold(fresh, leg="r1_out")
    assert g12.label == "DD" and g12.value == pytest.approx(P.G12_STALL_FRACTION * AnchoredMove.DEFAULT_FREE_RATIO)
    assert "not decided" in g12.derivation and "PROBE_STALL_MIN_STEPS" in g12.derivation      # review BM-RV-8
    short = replay([0.25] * 4, hover_leg)                    # ended (trace over) before step 5: never compared
    assert not short.free_ratio_measured and "not decided" in P.g12_stall_threshold(short).derivation
    assert "not decided" not in P.g12_stall_threshold(replay([0.25] * 4, as_built_leg)).derivation
    slow = replay([0.05, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083], hover_leg)
    assert not slow.done and slow.free_ratio_measured and slow.free_ratio == pytest.approx(0.083)
    g12 = P.g12_stall_threshold(slow, leg="r0_back")
    assert g12.name == "G12_stall_ratio" and g12.label == "ONLINE"
    assert g12.value == pytest.approx(P.G12_STALL_FRACTION * 0.083)
    for text in ("ratio_low=0.08", "ratio_high=0.4", "stall_min_steps=5", "leg=r0_back"):
        assert text in g12.derivation, g12.derivation
    v1_leg = replay([0.05, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083], as_built_leg)
    assert stall_step(v1_leg) is not None                                    # v1's 0.18 floor stalls this leg
    g12 = P.g12_stall_threshold(v1_leg)
    assert g12.value == pytest.approx(P.G12_STALL_FRACTION * P.G12_RATIO_BOUNDS[0])
    assert "ratio_low=0.18" in g12.derivation and "stall_min_steps=3" in g12.derivation


def test_parallax_probe_and_arrival_descent_use_the_hover_leg():
    """ParallaxProbe builds every leg as a _HoverProbeLeg (settles stay B4 holds); Calibrate+'s vertical halves stay
    plain _ProbeLegs; ``hover_arrival_descent`` (offered to BR, decision §2.6) is a hover leg straight down holding
    the start xy and orientation, open, with the contact stops, and stops on a surface within 2 steps of the fake's
    first contact."""
    w = world(F.planar_region_scene(), eef=(-0.20, 0.08, 1.05), start_tilt_deg=0.0)
    kinds = {}
    probe = P.ParallaxProbe(StepView(w.observation()), recorder=None, gripper="open")
    run(w, probe, before_step=lambda _w, p, _n: kinds.setdefault(p.stage, set()).add(type(p.current)))
    assert kinds == {"settle": {HoldPrimitive}, **{leg: {P._HoverProbeLeg} for leg in P.PROBE_STAGES[1:]}}
    w = world()
    kinds = {}
    cal = P.CalibratePlus(StepView(w.observation()), robot=ROBOT)
    run(w, cal, max_steps=400, before_step=lambda _w, p, _n: kinds.setdefault(p.stage, set()).add(type(p.current)))
    assert all(kinds[s] == {P._ProbeLeg} for s in VERTICAL_HALVES)
    # free-air arrival descent: travel reached, no stall, pure vertical
    w = world(F.planar_region_scene(), eef=(-0.20, 0.08, 1.05), start_tilt_deg=0.0)
    view = StepView(w.observation())
    descent = P.hover_arrival_descent(view, travel_m=0.03)
    assert isinstance(descent, P._HoverProbeLeg) and descent.gripper == "open" and descent.stop_on_contact
    assert descent.arm_stall_after_motion and descent.touch_threshold_mm == P.G3_FLOOR_MM
    np.testing.assert_allclose(descent.axis, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(descent.anchor_xyz, view.position)
    np.testing.assert_allclose(descent.orientation_xyzw, view.quaternion_xyzw)
    out, _ = run(w, descent)
    assert out["reason"] == "travel_reached" and out["evidence"]["travel_m"] >= 0.03 - 0.001
    assert np.linalg.norm(np.subtract(out["evidence"]["end_xyz"][:2], view.position[:2])) <= 0.001
    # onto the region top: fingertips start 1.5 cm above it
    top = F.planar_region_scene().info["truth"]["top_z"]
    w = world(F.planar_region_scene(), eef=(-0.20, 0.08, top + ROBOT.t_f_m + 0.015), start_tilt_deg=0.0)
    since = len(w.contact_log)
    calibrated = th_max("G3_touch_threshold_mm", Term("DD", P.G3_FLOOR_MM, "G3_FLOOR_MM"),
                        Term("ONLINE", 0.12, "G3_SIGMA_K*sigma_in"), unit="mm")
    descent = P.hover_arrival_descent(StepView(w.observation()), travel_m=0.03, touch_threshold_mm=calibrated)
    assert descent.touch_threshold_mm == 0.12
    out, _ = run(w, descent)
    assert out["reason"] in ("contact_stall", "contact_touch"), out["reason"]
    contacts = [e["step"] for e in w.contact_log[since:] if e["kind"] in ("contact", "finger_landing")]
    assert contacts and w.steps - (contacts[0] + 1) <= 2
    assert json_safe(out)


# ================================================================================================ 3. TwistSegment
def _rotation_cap_checks(actions):
    for a in actions:
        rot = np.asarray(a[3:6]) * 0.5                               # rad (R10 scale)
        assert np.linalg.norm(rot) <= P.K2_TWIST_RAD + 1e-9
        assert np.any(np.asarray(a[3:6]) != 0.0)                     # never an exactly-zero rotation
        assert np.max(np.abs(np.asarray(a[:3]) * 0.05)) <= P.TWIST_HOLD_CAP_M + 1e-12
        assert a[6] == 1.0                                           # gripper closed


def test_twist_free_sense_at_step_12_and_blocked_at_step_12():
    w = knob_world()                                                 # task-2-like: free in contract -1
    out, actions = run(w, segment(w, -1))
    assert (out["status"], out["reason"], out["actions"]) == ("succeeded", "free_sense", 12)
    assert out["evidence"]["decided_by"] in ("wedge", "both") and w.joint("knob") > 0.1
    _rotation_cap_checks(actions)
    b = knob_world()
    out, actions = run(b, segment(b, +1))
    assert (out["status"], out["reason"], out["actions"]) == ("failed", "blocked_sense", 12)
    assert abs(b.joint("knob") - (-0.005)) < 1e-9
    _rotation_cap_checks(actions)
    ev = out["evidence"]
    assert ev["sense"] == 1 and ev["yaw_realized_rad"] > 0.15 and len(ev["gap_trace_mm"]) == 13
    assert ev["gap_trace_mm"][-1] - ev["gap_trace_mm"][0] > 3.0                       # wedged (SP +4.2 mm)
    assert json_safe(out)


def test_twist_reversal_after_blocked_is_free():
    w = knob_world()
    blocked, _ = run(w, segment(w, +1))
    wedge = StandInWedge(mode="sense", after_reversal=True, blocked_steps=blocked["actions"])
    out, _ = run(w, segment(w, -1, wedge=wedge))
    assert out["reason"] == "free_sense"
    k0 = out["evidence"]["wedge"]["evidence"]["k0"]
    assert 2 <= k0 <= math.ceil(1.5 * blocked["actions"]) and out["actions"] == k0 + 12


def test_twist_free_sense_plus_one_world():
    w = knob_world(free_sense=+1)
    out, _ = run(w, segment(w, +1))
    assert out["reason"] == "free_sense" and w.joint("knob") > 0.1


def test_twist_ambiguous_trace_decided_by_part_rotation_at_the_extend_step():
    undecided = {16: ("pending", None, {"class": "undecided"})}
    for script, expect in (({16: ("met", "corotating", {})}, ("succeeded", "free_sense")),
                           ({16: ("violated", "part_not_corotating", {})}, ("failed", "blocked_sense"))):
        w = knob_world()
        out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", undecided), rotation=Scripted("part_rotation", script)))
        assert (out["status"], out["reason"], out["actions"]) == (*expect, 16)
        assert out["evidence"]["decided_by"] == "part_rotation"
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", undecided),
                            rotation=Scripted("part_rotation", default=("failed", "unmeasurable", {}))))
    assert (out["reason"], out["actions"]) == ("sense_undecided", 16)
    assert out["evidence"]["sense_by_wedge_only"] and "sense_by_wedge_only" in out["evidence"]["flags"]
    # with the real fake hinge and an unmeasurable part the free case still decides by the wedge alone
    w = knob_world()
    out, _ = run(w, segment(w, -1, rotation=TruthPartRotation(w, unmeasurable=True)))
    assert out["reason"] == "free_sense" and out["evidence"]["decided_by"] == "wedge"


def test_twist_round_part_blocked_by_part_rotation():
    w = knob_world(lever=False)                                      # no wedge: a blocked twist slips the pads
    out, _ = run(w, segment(w, +1))
    assert out["evidence"]["wedge"]["reason"] == "free_sense"          # the gap reads free (no rise)
    assert (out["reason"], out["evidence"]["decided_by"]) == ("blocked_sense", "part_rotation")
    assert out["evidence"]["blocked_by_part_rotation"]
    f = knob_world(lever=False)
    out, _ = run(f, segment(f, -1))
    assert out["reason"] == "free_sense" and out["evidence"]["decided_by"] == "both"


def test_twist_hold_lost_below_the_g6_floor():
    w = knob_world()

    def lose(world_, primitive, n):
        if n == 3:
            world_.inject("hold_lost")
    floor = w.gap_mm - 0.5                       # the floor row wins when it fires on the same step as K7 slip
    out, actions = run(w, segment(w, -1, floor=floor), before_step=lose)
    assert out["reason"] == "hold_lost" and out["evidence"]["decided_by"] == "hold_floor"
    assert out["evidence"]["gap_trace_mm"][-1] < floor and len(actions) == 4
    # with a realistic floor (empty floor + 0.5 mm) the falling gap first trips the K7 slip pattern
    s = knob_world()
    out, _ = run(s, segment(s, -1, floor=1.6), before_step=lose)
    assert out["reason"] in ("turn_slip", "hold_lost")


def test_twist_decision_rows_with_scripted_observers():
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", {4: ("violated", "turn_slip", {})})))
    assert (out["reason"], out["actions"]) == ("turn_slip", 4)
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", {4: ("failed", "sense_undecided", {})})))
    assert out["reason"] == "sense_undecided" and out["evidence"]["decided_by"] == "wedge"
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", {12: ("violated", "blocked_sense", {})}),
                            rotation=Scripted("part_rotation", default=("violated", "part_not_corotating", {}))))
    assert (out["reason"], out["evidence"]["decided_by"]) == ("blocked_sense", "both")
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge"), budget=0.2))
    assert out["reason"] == "yaw_budget" and out["evidence"]["yaw_budget_left_rad"] < P.K2_TWIST_RAD
    assert out["evidence"]["decided_by"] == "budget"
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge")))
    assert (out["reason"], out["actions"]) == ("sense_undecided", P.TWIST_SEGMENT_MAX_STEPS)
    assert out["evidence"]["segment_cap"] and out["evidence"]["decided_by"] == "cap"
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=Scripted("wedge", {3: ("violated", "hold_lost", {})})))
    assert (out["reason"], out["actions"], out["evidence"]["decided_by"]) == ("hold_lost", 3, "wedge")
    w = knob_world()
    already = P.TwistSegment(StepView(w.observation()), sense=-1, wedge=Scripted("wedge"), part_rotation=None,
                             hold_floor_mm=1.0, yaw_budget_left_rad=0.05, robot=ROBOT)
    assert already.done and already.reason == "yaw_budget" and already.actions_executed == 0


def test_twist_observer_lifecycle_and_contexts():
    w = knob_world()
    wedge, rotation, effect = StandInWedge(mode="sense"), TruthPartRotation(w), TruthEffect(w)
    effect.start(None)                                                # the relation's pre-twist baseline [C-27]
    run(w, segment(w, -1, wedge=wedge, rotation=rotation, effect=effect))
    assert wedge.started == 1 and rotation.started == 1 and effect.started == 1      # effect never restarted
    assert len(effect.contexts) == 12 and all(c == {"theta_part_rad": None} for c in effect.contexts)
    with pytest.raises(ValueError):
        segment(w, 0)
    with pytest.raises(ValueError):
        segment(w, True)


# ================================================================================================ 4. HingeTwist
def test_hinge_effect_margin_with_theta_e_from_the_first_frame():
    w = knob_world()
    effect = TruthEffect(w)
    effect.start(None)
    run(w, segment(w, -1, effect=effect))
    ht = hinge(w, -1, effect=effect, prior=F.TASK2_KNOB_CENTRE_XY)
    out, _ = run(w, ht)
    ev = out["evidence"]
    assert (out["status"], out["reason"], ev["decided_by"]) == ("succeeded", "effect_margin_reached", "effect")
    theta_e = ev["theta_e_rad"]
    report = effect.met.evidence
    assert theta_e == pytest.approx(report["theta_e_rad"])
    truth_on = w._effect_on_since["burner_plate"]["on_step"]
    assert abs(report["world_step"] - truth_on) <= 1                                   # within one step of the onset
    margin = max(P.K5_FLOOR_RAD, P.K5_FRACTION * theta_e)
    history = ht.theta_history
    assert history[-1] >= theta_e + margin and history[-2] < theta_e + margin           # stops at the first step
    assert any(abs(h - theta_e) < 1e-12 for h in history)                              # theta_e = theta_part then
    assert 0.5 <= w.joint("knob") <= 0.75
    assert "K5_margin_rad" in [t["name"] for t in ev["thresholds"]] and json_safe(out)


def test_hinge_end_stop_after_a_reversal():
    """The endstop_t2 pattern [C-41]: free (-1), reversal free toward the lower limit (+1), HingeTwist(+1) into the
    limit -> end_stop within 5 steps of the joint reaching its lower limit."""
    w = knob_world()
    first, _ = run(w, segment(w, -1))
    assert first["reason"] == "free_sense"
    rev, _ = run(w, segment(w, +1, wedge=StandInWedge(mode="sense", after_reversal=True,
                                                     blocked_steps=first["actions"])))
    assert rev["reason"] == "free_sense"
    limit_at = []

    def watch(world_, primitive, n):
        if not limit_at and world_.joint("knob") <= -0.0049:
            limit_at.append(n)
    out, _ = run(w, hinge(w, +1), after_step=watch)
    assert out["reason"] == "end_stop" and out["evidence"]["decided_by"] in ("wedge", "both", "part_rotation")
    assert limit_at and 0 <= out["actions"] - limit_at[0] <= 5
    # the K4w path alone (no part-rotation observer) also stops within 5 steps
    w = knob_world()
    first, _ = run(w, segment(w, -1))
    run(w, segment(w, +1, wedge=StandInWedge(mode="sense", after_reversal=True, blocked_steps=first["actions"])))
    limit_at.clear()
    out, _ = run(w, hinge(w, +1, rotation=NO_ROTATION), after_step=watch)
    assert out["reason"] == "end_stop" and 0 <= out["actions"] - limit_at[0] <= 5


def test_hinge_round_part_stop_and_elongated_flat_stop():
    w = knob_world(lever=False, q_hi=0.35)
    free_classified(w)
    out, _ = run(w, hinge(w, -1, capable=False))
    assert (out["status"], out["reason"]) == ("succeeded", "end_stop")
    assert out["evidence"]["part_stop_without_wedge"] and out["evidence"]["decided_by"] == "part_rotation"
    assert out["evidence"]["gap_trend"] == "flat"
    e = knob_world(lever=True, wedge=False, q_hi=0.35)              # elongated part whose stop does not wedge
    free_classified(e)
    out, _ = run(e, hinge(e, -1, capable=True))
    assert (out["status"], out["reason"], out["evidence"]["decided_by"]) == ("failed", "turn_slip", "part_rotation")


def test_hinge_quarter_turn_and_yaw_budget():
    w = knob_world()
    free_classified(w)
    out, _ = run(w, hinge(w, -1, target=math.pi / 2))
    assert (out["status"], out["reason"]) == ("succeeded", "quarter_turn_reached")
    assert out["evidence"]["theta_part_rad"] >= math.pi / 2 and w.joint("knob") > 1.3
    b = knob_world()
    free_classified(b)
    out, _ = run(b, hinge(b, -1, budget=0.35))
    assert (out["status"], out["reason"]) == ("failed", "yaw_budget")
    assert out["evidence"]["yaw_budget_left_rad"] < P.K2_TWIST_RAD <= out["evidence"]["yaw_budget_left_rad"] + 0.03


def test_hinge_orbit_follow_engages_after_an_accepted_refit():
    w = knob_world(grip_offset_m=0.012)                               # the hinge 12 mm off the grip
    free_classified(w)
    true_hinge = np.asarray(w.scene.groups["knob"]["hinge"].centre_xy)
    ht = hinge(w, -1, target=0.6)                                     # prior = the grip point (orbit follow off)
    assert not ht.orbit_follow
    out, _ = run(w, ht)
    ev = out["evidence"]
    assert out["reason"] == "quarter_turn_reached" or out["reason"] == "half_turn_reached"
    accepted = [f for f in ev["centre_fits"] if f["accepted"]]
    assert accepted and accepted[0]["sigma_m"] < ev["thresholds"][1]["value"]
    assert ev["centre_source"] == "refit" and np.linalg.norm(np.asarray(ev["centre_xy"]) - true_hinge) <= 0.002
    assert ev["orbit_follow"]["engaged"] and ev["orbit_follow"]["steps"] > 0
    assert ev["tug"]["max_deviation_m"] < P.K11_FLOOR_M


def test_hinge_noisy_short_arc_keeps_the_prior():
    w = knob_world(grip_offset_m=0.012)
    free_classified(w)
    rng = np.random.default_rng(3)

    def noise(world_, primitive, n):
        world_.inject("drift", vector_m=tuple(rng.normal(0.0, 0.003, 2)) + (0.0,), steps=1)
    prior = StepView(w.observation()).position[:2].copy()
    out, _ = run(w, hinge(w, -1, target=0.11), before_step=noise)
    ev = out["evidence"]
    assert ev["centre_fits"] and not any(f["accepted"] for f in ev["centre_fits"])
    assert ev["centre_source"] == "prior" and np.allclose(ev["centre_xy"], prior)


def test_hinge_axis_tug_hold_lost_slip_and_stall():
    w = knob_world()
    free_classified(w)

    def tug(world_, primitive, n):
        if n == 6:
            world_.inject("tug", vector_m=(0.002, 0.0), steps=12)
    out, _ = run(w, hinge(w, -1), before_step=tug)
    assert (out["reason"], out["evidence"]["decided_by"]) == ("axis_tug", "tug")
    assert out["actions"] <= 6 + 12 and out["evidence"]["tug"]["steps"] >= P.K11_STEPS
    h = knob_world()
    free_classified(h)

    def lose(world_, primitive, n):
        if n == 4:
            world_.inject("hold_lost")
    out, _ = run(h, hinge(h, -1, floor=h.gap_mm - 0.5), before_step=lose)
    assert (out["reason"], out["actions"]) == ("hold_lost", 5)
    s = knob_world()
    free_classified(s)

    def slip(world_, primitive, n):
        if n == 5:
            world_.inject("slip", steps=3)
    out, _ = run(s, hinge(s, -1), before_step=slip)
    assert (out["reason"], out["evidence"]["decided_by"]) == ("turn_slip", "wedge")
    scene = F.rotary_part_scene()
    stuck = F.FakeTwoViewSim(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=False,
                             rotation_ratio=1e-4)
    F.scripted_grasp_rotary(stuck)
    out, _ = run(stuck, hinge(stuck, -1))
    assert (out["reason"], out["actions"]) == ("hand_yaw_stall", P.HAND_YAW_STALL_STEPS)


def test_hinge_rows_with_scripted_observers():
    w = knob_world()
    free_classified(w)
    out, _ = run(w, hinge(w, -1, wedge=Scripted("wedge", {3: ("met", "end_stop", {})})))
    assert (out["reason"], out["actions"]) == ("end_stop", 3)
    w = knob_world()
    free_classified(w)
    effect = Scripted("effect", default=("pending", None, {}))
    effect.script = {k: ("met", "effect_onset", {"onset_step": 2, "theta_e_rad": 0.0}) for k in range(2, 400)}
    effect.start(None)                                                # the relation starts EffectWatch [C-27]
    ht = hinge(w, -1, effect=effect)
    out, _ = run(w, ht)
    assert out["reason"] == "effect_margin_reached" and out["evidence"]["theta_e_rad"] == 0.0
    assert ht.theta_history[-1] >= P.K5_FLOOR_RAD > ht.theta_history[-2]
    assert all(c == {"theta_part_rad": pytest.approx(t)} for c, t in zip(effect.contexts, ht.theta_history[1:]))
    w = knob_world()
    free_classified(w)
    out, _ = run(w, hinge(w, -1, rotation=Scripted("part_rotation", {6: ("violated", "part_not_corotating", {})}),
                          capable=True))
    assert out["reason"] == "turn_slip" and out["evidence"]["gap_trend"] in ("flat", "fall")
    w = knob_world()
    resumed = hinge(w, -1, theta_start_rad=0.3, target=0.35)
    out, _ = run(w, resumed)
    assert out["reason"] == "quarter_turn_reached" and out["evidence"]["theta_start_rad"] == 0.3
    assert out["actions"] <= 5


# ================================================================================================ 5. HeldReach
def held_world(scene=None, **kw):
    w = world(scene or F.free_object_with_thin_part_scene(), eef=(0.03, -0.12, 1.15), start_tilt_deg=0.0, **kw)
    assert F.scripted_grasp_thin_part(w)["kind"] == "free"
    return w


def test_held_reach_sigma_carry_and_equality_with_waypoint_reach():
    a, b = held_world(), held_world()
    va, vb = StepView(a.observation()), StepView(b.observation())
    held = P.HeldReach(va, [0.02, 0.01, 0.04], hold_floor_mm=1.5, gap_ref_mm=va.gap_mm,
                       grasp_quaternion=va.quaternion_xyzw)
    plain = WaypointReach(vb, [0.02, 0.01, 0.04], gripper="close", name="held_reach",
                          orientation_xyzw=vb.quaternion_xyzw, hold_floor_mm=1.5, gap_ref_mm=vb.gap_mm)
    out_a, acts_a = run(a, held)
    out_b, acts_b = run(b, plain)
    assert out_a["reason"] == out_b["reason"] == "goal_reached"
    assert acts_a == acts_b
    ev = out_a["evidence"]
    assert ev["sigma_carry_mm"] is not None and ev["grasp_orientation_held"]
    names = [t["name"] for t in ev["G7_thresholds"]]
    assert names == ["G7_carry_total_rise_mm", "G7_carry_step_rise_mm"]
    assert ev["G7_thresholds"][0]["value"] == max(P.G7_TOTAL_FLOOR_MM, P.G7_TOTAL_SIGMA_K * ev["sigma_carry_mm"])
    assert held.GAP_STEP_RISE_MM == max(P.G7_STEP_FLOOR_MM, P.G7_STEP_SIGMA_K * ev["sigma_carry_mm"])
    assert {k: v for k, v in out_a["evidence"].items() if k in out_b["evidence"]} == out_b["evidence"]


def test_held_reach_contact_and_orientation_rule():
    post = F.FakeBody("post", "aabb", (0.10, 0.04, 0.900), (0.04, 0.04, 0.20), colour=(90, 90, 90))
    scene = F.free_object_with_thin_part_scene()
    w = held_world(F.FakeScene(scene.bodies + [post], info=scene.info))
    v = StepView(w.observation())
    out, _ = run(w, P.HeldReach(v, [0.07, 0.0, 0.0], hold_floor_mm=1.5, gap_ref_mm=v.gap_mm))
    assert out["status"] == "failed" and out["reason"] == "held_object_contact"
    # a grasp orientation farther than M3 from the current one is not restored mid-move
    far = F.FakeTwoViewSim(F.planar_region_scene(), eef=(-0.2, 0.0, 1.1), images=False)
    vf = StepView(far.observation())
    turned = F._matrix_quat(F._rot_axis((0, 0, 1), math.radians(20)) @ far.R)
    reach = P.HeldReach(vf, [0.0, 0.0, 0.02], hold_floor_mm=0.0, gap_ref_mm=vf.gap_mm, grasp_quaternion=turned)
    assert not reach.grasp_orientation_held and np.allclose(reach.orientation_xyzw, vf.quaternion_xyzw)
    near = F._matrix_quat(F._rot_axis((0, 0, 1), math.radians(10)) @ far.R)
    reach = P.HeldReach(vf, [0.0, 0.0, 0.02], hold_floor_mm=0.0, gap_ref_mm=vf.gap_mm, grasp_quaternion=near)
    assert reach.grasp_orientation_held
    with pytest.raises(ValueError):
        P.HeldReach(vf, [0.0, 0.0, 0.02], hold_floor_mm=0.0, gap_ref_mm=vf.gap_mm, gripper="open")


# ================================================================================================ 6. GuardedLower
def first_held_contact(w, since):
    events = [e for e in w.contact_log[since:] if e["kind"] in ("held_support", "creep_blocked")]
    return events[0]["step"] if events else None


@pytest.mark.parametrize("dynamics", ["second_order", "kinematic"])
def test_guarded_lower_support_stall_within_two_steps(dynamics):
    w = held_world(dynamics=dynamics)
    v = StepView(w.observation())
    since = len(w.contact_log)
    lower = P.GuardedLower(v, phase="slow", travel_m=0.06, aim_grip_xy=v.position[:2], drift_abort_m=0.005,
                           hold_floor_mm=1.5)
    out, actions = run(w, lower)
    assert (out["status"], out["reason"]) == ("succeeded", "contact_stall")
    contact = first_held_contact(w, since)
    assert contact is not None and w.steps - (contact + 1) <= 2
    assert out["evidence"]["support_cue"] == "contact_stall" and out["evidence"]["phase"] == "slow"
    names = [t["name"] for t in out["evidence"]["thresholds"]]
    assert names[:2] == ["G8_lower_step_rise_mm", "M5_drift_abort_m"] and "P15_command_m" in names
    g12 = out["evidence"]["thresholds"][-1]
    assert g12["name"] == "G12_stall_ratio" and g12["label"] == "ONLINE"            # free ratio measured by step 5
    assert g12["value"] == pytest.approx(P.G12_STALL_FRACTION * np.clip(lower.free_ratio, *P.G12_RATIO_BOUNDS))
    assert out["evidence"]["drift_at_start"] is False
    for a in actions:
        assert a[2] * 0.05 == pytest.approx(-P.P15_COMMAND_M) and a[6] == 1.0
    assert json_safe(out)


def test_guarded_lower_fast_handover_drift_and_unreachable():
    w = held_world(dynamics="second_order")
    v = StepView(w.observation())
    out, actions = run(w, P.GuardedLower(v, phase="fast", travel_m=0.06, aim_grip_xy=v.position[:2],
                                         drift_abort_m=0.005, hold_floor_mm=1.5))
    assert (out["status"], out["reason"]) == ("succeeded", "handover_contact_stall")
    assert actions[0][2] * 0.05 == pytest.approx(-P.LOWER_FAST_COMMAND_M)
    d = held_world(dynamics="second_order")
    v = StepView(d.observation())
    d.inject("drift", vector_m=(0.0015, 0.0, 0.0), steps=30)
    out, _ = run(d, P.GuardedLower(v, phase="slow", travel_m=0.06, aim_grip_xy=v.position[:2], drift_abort_m=0.005,
                                   hold_floor_mm=1.5))
    assert (out["status"], out["reason"]) == ("failed", "lateral_drift") and out["evidence"]["max_drift_m"] > 0.005
    assert out["evidence"]["drift_at_start"] is False and out["actions"] > 0
    # an aim error before any lowering: lateral_drift at construction, flagged drift_at_start (no lift needed)
    s0 = held_world(dynamics="second_order")
    v = StepView(s0.observation())
    off = P.GuardedLower(v, phase="slow", travel_m=0.06, aim_grip_xy=v.position[:2] + [0.006, 0.0],
                         drift_abort_m=0.005, hold_floor_mm=1.5)
    assert off.done and (off.reason, off.actions_executed) == ("lateral_drift", 0)
    assert off.outcome()["evidence"]["drift_at_start"] is True and json_safe(off.outcome())
    assert off.outcome()["evidence"]["thresholds"][-1]["label"] == "DD"              # free ratio not yet measured
    u = held_world(dynamics="second_order")
    v = StepView(u.observation())
    u.inject("lateral_block", steps=30)
    out, actions = run(u, P.GuardedLower(v, phase="slow", travel_m=0.06, aim_grip_xy=v.position[:2] + [0.004, 0.0],
                                         drift_abort_m=0.01, hold_floor_mm=1.5))
    assert (out["reason"], len(actions)) == ("lateral_unreachable", P.M5_SATURATED_STEPS)
    h = held_world()
    v = StepView(h.observation())
    already = P.GuardedLower(v, phase="slow", travel_m=0.05, aim_grip_xy=v.position[:2], drift_abort_m=0.005,
                             hold_floor_mm=v.gap_mm + 1.0)
    assert already.done and already.reason == "hold_lost"
    with pytest.raises(ValueError):
        P.GuardedLower(v, phase="medium", travel_m=0.05, aim_grip_xy=v.position[:2], drift_abort_m=0.005,
                       hold_floor_mm=1.0)


def _synthetic_view(state):
    blank = F._blank_url()
    return StepView(PolicyObservation(blank, blank, tuple(float(x) for x in state), "synthetic"))


def test_guarded_lower_g8_gap_rise_and_sigma_lower():
    """Synthetic state sequence (no simulator): a free lowering at the 0.26 ratio with gap noise, then one step whose
    gap rises past max(G8_FLOOR_MM, 4 sigma_lower) -> gap_rise (slow) / handover_gap_rise (fast)."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.1, 20)
    for phase, reason in (("slow", "gap_rise"), ("fast", "handover_gap_rise")):
        state = np.array([0.0, 0.0, 1.1, math.pi, 0.0, 0.0, 0.010, -0.010])
        view = _synthetic_view(state)
        lower = P.GuardedLower(view, phase=phase, travel_m=0.2, aim_grip_xy=(0.0, 0.0), drift_abort_m=0.005,
                               hold_floor_mm=5.0)
        k = 0
        while not lower.done:
            action = lower.next_action(view)
            nxt = state.copy()
            nxt[2] += 0.26 * action[2] * 0.05
            gap = 20.0 + noise[k % 20] * (k < 8) + (2.0 if k == 8 else 0.0)
            nxt[6], nxt[7] = gap / 2000.0, -gap / 2000.0
            after = _synthetic_view(nxt)
            lower.observe_transition(view, action, after)
            view, state, k = after, nxt, k + 1
        out = lower.outcome()
        assert out["reason"] == reason and out["actions"] == 9
        sigma = out["evidence"]["sigma_lower_mm"]
        assert sigma == pytest.approx(np.std(np.diff(np.r_[20.0, 20.0 + noise[:5]])))
        g8 = out["evidence"]["thresholds"][0]
        assert g8["name"] == "G8_lower_step_rise_mm" and g8["value"] == max(P.G8_FLOOR_MM, P.G8_SIGMA_K * sigma)


# ================================================================================================ 7. Touchdown, ReleaseVerify
def test_touchdown_and_release():
    w = held_world()
    v = StepView(w.observation())
    td = P.Touchdown(v, lift_m=0.025, hold_floor_mm=1.5, drift_abort_m=0.005)
    assert td.travel_m == pytest.approx(0.025 + P.TOUCHDOWN_EXTRA_M) and td.phase == "slow"
    out, _ = run(w, td)
    assert out["reason"] == "contact_stall" and td.touch_grip_z == pytest.approx(w.grip[2])
    assert out["evidence"]["touchdown"]["touch_grip_z"] == pytest.approx(w.grip[2])
    assert abs(w.grip[2] - 1.020) <= 0.004                                             # back at the pick height
    rec = Recorder()
    rv = P.ReleaseVerify(StepView(w.observation()), g_open_mm=78.0, recorder=rec)
    out, _ = run(w, rv)
    assert (out["status"], out["reason"]) == ("succeeded", "released") and w.held() is None
    assert out["evidence"]["gap_mm"] >= 78.0 - P.M12_OPEN_TOL_MM
    assert out["evidence"]["displacement_m"][2] >= P.M13_RETREAT_M - 0.001
    assert set(rec.stages) == {"release_open", "release_retreat"}
    assert [t["name"] for t in out["evidence"]["thresholds"]] == ["M12_released_gap_mm", "M13_retreat_m",
                                                                  "G3_touch_threshold_mm", "G12_stall_ratio"]
    assert out["evidence"]["thresholds"][2]["value"] == P.G3_FLOOR_MM == out["evidence"]["touch_threshold_mm"]
    assert "retreat" in out["evidence"]["thresholds"][3]["derivation"]
    assert out["evidence"]["settle_hand_yaw_rad"] is None                              # no settle stage
    assert json_safe(out)


def test_release_incomplete_and_touch_on_retreat():
    w = held_world()
    w.inject("open_limit_mm", mm=w.gap_mm)                                              # the gripper cannot open
    out, _ = run(w, P.ReleaseVerify(StepView(w.observation()), g_open_mm=78.0))
    assert (out["status"], out["reason"]) == ("failed", "release_incomplete") and out["evidence"]["retreat"] is None
    assert [t["name"] for t in out["evidence"]["thresholds"]] == ["M12_released_gap_mm", "M13_retreat_m"]
    shelf = F.FakeBody("shelf", "aabb", (-0.20, 0.08, 1.12), (0.30, 0.30, 0.02), colour=(120, 100, 80))
    o = world(F.FakeScene([shelf]), eef=(-0.20, 0.08, 1.005), start_tilt_deg=0.0)
    view = StepView(o.observation())
    out, _ = run(o, P.ReleaseVerify(view, g_open_mm=view.gap_mm))
    assert (out["status"], out["reason"]) == ("failed", "touch_on_retreat")
    assert out["evidence"]["retreat"]["reason"] in ("contact_stall", "contact_touch", "budget")
    assert [t["name"] for t in out["evidence"]["thresholds"]][2:] == ["G3_touch_threshold_mm", "G12_stall_ratio"]


def test_release_settle_first_on_a_rotary_part():
    w = knob_world()
    free_classified(w)
    run(w, hinge(w, -1, target=0.3))
    rec = Recorder()
    rv = P.ReleaseVerify(StepView(w.observation()), g_open_mm=78.0, recorder=rec, settle_first=True)
    out, _ = run(w, rv)
    assert out["reason"] == "released" and out["evidence"]["settle"]["reason"] == "hold_complete"
    assert rec.stages[0] == "release_settle" and set(rec.stages) == {"release_settle", "release_open", "release_retreat"}
    assert out["evidence"]["settle"]["evidence"]["gripper"] == "close" and w.held() is None
    # the hand yaw over the closed settle, signed about the start approach R2 (contract sense convention)
    start = _synthetic_view(rec.calls[0][1])
    settle_end = _synthetic_view([c for c in rec.calls if c[0] == "release_settle"][-1][2])
    yaw = twist_about(start.quaternion_xyzw, settle_end.quaternion_xyzw, start.R[:, 2])
    assert out["evidence"]["settle_hand_yaw_rad"] == pytest.approx(yaw, abs=1e-9)


# ================================================================================================ LiftMonitorAdapter
def test_lift_monitor_adapter_with_verify_hold():
    w = held_world()
    view = StepView(w.observation())
    rec = Recorder()
    monitor = P.LiftMonitorAdapter(rec, view)
    hold = VerifyHold(view, axis=(0, 0, 1), travel_m=0.02, calibration=GripperCalibration.default(),
                      hold_margin_mm=0.5, monitor=monitor)
    out, actions = run(w, hold)
    assert out["reason"] == "hold_pass" and not out["evidence"]["monitor_disagrees"]
    assert rec.stages == ["lift"] * len(actions) and monitor.frames == len(actions)
    np.testing.assert_array_equal(rec.calls[0][1], view.state)
    for (_, _, after), (_, before, _) in zip(rec.calls, rec.calls[1:]):
        np.testing.assert_array_equal(after, before)                                    # consecutive frame pairs


# ==================================================================== 7b. GC-1 guarded close (grasp-refusal decision J)
GC_XY = (-0.10, 0.10)
GC_GRIP_Z = 0.95
GC_BAR_W_MM = 24.3                           # the fake task-2 lever width (truth lever_width_m)
GC_SETTLE = GripperSettleConfig(max_pose_correction_m=0.03)        # relations' GRASP_SETTLE (v1 GRASP_SETTLE)


def gc_world(bodies=(), *, start_gap_mm=F.START_GAP_MM):
    """A top-down hand (R = diag(1, -1, -1): the jaw R1 is world -y, finger 1 (state8[6]) on -y) whose pad window
    (grip z - 4.4 .. + 11.6 mm) crosses the given boxes; the support 5 cm below the grip."""
    return F.FakeTwoViewSim(F.FakeScene(list(bodies)), eef=(*GC_XY, GC_GRIP_Z), start_tilt_deg=0.0, images=False,
                            start_gap_mm=start_gap_mm)


def gc_box(name, s_in_mm, s_out_mm, finger, *, role="fixed"):
    """A box occupying [s_in, s_out] mm outward from the grip on finger ``finger``'s side of the jaw (the closing
    finger meets its outer face s_out), through the pad window."""
    side = 1.0 if finger == 1 else -1.0
    mid_m = (s_in_mm + s_out_mm) / 2000.0
    return F.FakeBody(name, "aabb", (GC_XY[0], GC_XY[1] - side * mid_m, 0.9),
                      (0.03, (s_out_mm - s_in_mm) / 1000.0, 0.07), role=role)


def gc_bar():
    """The part: a GC_BAR_W_MM bar centred on the grip across the jaw (one rigid group: both pads clamp it)."""
    return F.FakeBody("bar", "aabb", (GC_XY[0], GC_XY[1], 0.9), (0.03, GC_BAR_W_MM / 1000.0, 0.07))


def gc_gate(r1_mm, r2_mm, span_hi_mm, *, tol_mm=1.0):
    """A planner checks['gate'] stand-in (planner.close_gate's keys) about the grip point along the hand's jaw."""
    return {"fingers": {"finger1": {"R_m": r1_mm / 1000.0}, "finger2": {"R_m": r2_mm / 1000.0}},
            "grip_xy": list(GC_XY), "jaw_xy": [0.0, -1.0], "span_hi_mm": span_hi_mm, "tol_mm": tol_mm,
            "pad_face_offset_mm": robot_facts.PAD_SEPARATION_OFFSET_MM / 2.0}


@pytest.fixture(scope="module")
def fake_cal():
    """Calibrate+ on the fake (as the fake chains run it): its empty close starts below the 80 mm joint limit (the
    first calibrated action ends at ~76.4 mm), so a later close from the opened hand starts above the calibration."""
    cal, out, _ = calibrate(world())
    assert out["reason"] == "calibrated_online"
    record = cal.record()
    return record.gripper, P.gripper_quantities(record)["sigma_close_mm"]


def guarded(w, cal, gate, *, active, config=GC_SETTLE):
    gripper, sigma = cal
    return P.GuardedClose(StepView(w.observation()), gripper=gripper, sigma_close_mm=sigma, gate=gate, active=active,
                          config=config)


def plain_close(w, config=GC_SETTLE):
    return run(w, GripperSettlePrimitive("close", StepView(w.observation()), config=config, name="grasp_close"))


def closing(action) -> bool:
    return float(action[6]) > 0.0


def test_guarded_close_free_motion_is_read_at_the_fingers_own_opening():
    """Judge amendment: free_mm(q) reads the calibrated close [g_open, curve...] at the gap 2 q, not at the action
    count (c[j] >= 2q > c[j + 1], next = c[j + 1] - t (c[j + 1] - c[j + 2])); above c[0] the first calibrated step,
    at or below the curve's end nothing."""
    view = _synthetic_view((0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.04, -0.04))
    gripper = GripperCalibration(80.0, (70.0, 50.0, 30.0, 20.0), 0.0, "online")
    gc = P.GuardedClose(view, gripper=gripper, sigma_close_mm=0.01, gate=None, active=True)
    assert not gc.active                                               # no gate: monitor-only, never aborts
    assert gc.free_mm(41.0) == pytest.approx(5.0) and gc.free_mm(40.0) == pytest.approx(5.0)   # >= c[0]
    assert gc.free_mm(35.0) == pytest.approx(10.0)                      # g = c[1]: t = 0, next = c[2]
    assert gc.free_mm(30.0) == pytest.approx(10.0)                      # t = 0.5: next = 50 - 0.5 (50 - 30) = 40
    assert gc.free_mm(12.5) == pytest.approx(2.5)                       # the last interval: next = c[-1]
    assert gc.free_mm(10.0) == 0.0 and gc.free_mm(5.0) == 0.0          # at / below the curve's end
    th_ = gc.touch_th
    assert (th_.name, th_.value, th_.label) == ("GC_touch_mm", P.GC_TOUCH_FLOOR_MM, "DD")
    loud = P.GuardedClose(view, gripper=gripper, sigma_close_mm=0.2, gate=None, active=False)
    assert loud.touch_th.value == pytest.approx(P.GC_TOUCH_SIGMA_K * 0.2) and loud.touch_th.label == "ONLINE"


@pytest.mark.parametrize("active", [True, False])
def test_guarded_close_untouched_is_the_v1_close_action_for_action(fake_cal, active):
    w = gc_world()
    out, actions = run(w, guarded(w, fake_cal, gc_gate(30.0, 30.0, 30.0), active=active))
    plain, plain_actions = plain_close(gc_world())
    assert actions == plain_actions and out["status"] == plain["status"] and out["reason"] == plain["reason"]
    guard = out["evidence"]["guard"]
    assert guard["touch"] == {"1": None, "2": None} and guard["verdict"] is None and not guard["would_abort"]
    assert out["evidence"]["close"]["evidence"]["gap_curve_mm"] == plain["evidence"]["gap_curve_mm"]
    assert json_safe(out) and guard["active"] is active


def test_guarded_close_on_the_part_within_its_limits_proceeds_unchanged(fake_cal):
    """A clean hold: both pads meet the part at its faces (s = w/2) inside R and the span w <= span_hi + tol."""
    w = gc_world([gc_bar()])
    out, actions = run(w, guarded(w, fake_cal, gc_gate(20.0, 20.0, GC_BAR_W_MM + 3.0), active=True))
    _, plain_actions = plain_close(gc_world([gc_bar()]))
    assert actions == plain_actions and out["status"] == "succeeded" and out["evidence"]["guard"]["verdict"] is None
    touch = out["evidence"]["guard"]["touch"]
    for finger in ("1", "2"):
        assert touch[finger]["s_mm"] == pytest.approx(GC_BAR_W_MM / 2.0, abs=0.1)
    assert [c["group"] for c in w.contact_log if c["kind"] == "clamp"] == ["bar"]


def test_guarded_close_obstacle_beyond_the_limit_aborts_and_opens_at_the_next_action(fake_cal):
    """An obstacle beside the grip on finger 1's side, meeting the finger at 25 mm against R1 = 15 mm: active ->
    failed early_finger_contact; the action after the touch step is the first open action; the hand opens back to
    the start opening (>= OPEN_SETTLE_MIN_ACTIONS open actions)."""
    w = gc_world([gc_box("obstacle", 15.0, 25.0, 1)])
    start_gap = w.gap_mm
    out, actions = run(w, guarded(w, fake_cal, gc_gate(15.0, 30.0, 40.0), active=True))
    assert (out["status"], out["reason"]) == ("failed", "early_finger_contact")
    verdict = out["evidence"]["guard"]["verdict"]
    assert verdict[0] == "early_finger_contact" and verdict[1]["finger"] == 1
    assert verdict[1]["s_mm"] == pytest.approx(25.0, abs=0.1) and verdict[1]["limit_mm"] == 15.0
    k = verdict[1]["k"]
    assert k >= P.GC_FIRST_JUDGED_K and all(closing(a) for a in actions[:k]) and not any(map(closing, actions[k:]))
    assert len(actions) - k >= P.OPEN_SETTLE_MIN_ACTIONS
    _, plain_actions = plain_close(gc_world([gc_box("obstacle", 15.0, 25.0, 1)]))
    assert actions[:k] == plain_actions[:k]                              # the close itself is the v1 close
    assert out["evidence"]["gap_mm"] >= start_gap - P.M12_OPEN_TOL_MM
    assert out["evidence"]["open"]["status"] == "succeeded" and json_safe(out)


def test_guarded_close_aborts_before_the_part_when_a_finger_meets_another_surface_first(fake_cal):
    """Safety rule: never close when a finger meets an unexpected surface before the part. The part (a bar at the
    grip) plus an obstacle that finger 2 meets at 26 mm (R2 = 16 mm): the close is abandoned before any pad reaches
    the part, so the part is never clamped."""
    w = gc_world([gc_bar(), gc_box("obstacle", 18.0, 26.0, 2)])
    out, actions = run(w, guarded(w, fake_cal, gc_gate(20.0, 16.0, GC_BAR_W_MM + 3.0), active=True))
    assert out["reason"] == "early_finger_contact" and out["evidence"]["guard"]["verdict"][1]["finger"] == 2
    assert out["evidence"]["guard"]["touch"]["1"] is None                # finger 1 never reached the part
    assert not [c for c in w.contact_log if c["kind"] == "clamp"]
    assert out["evidence"]["gap_mm"] > GC_BAR_W_MM + 10.0


def test_guarded_close_monitor_only_records_would_abort_with_identical_actions(fake_cal):
    w = gc_world([gc_box("obstacle", 15.0, 25.0, 1)])
    out, actions = run(w, guarded(w, fake_cal, gc_gate(15.0, 30.0, 40.0), active=False))
    _, plain_actions = plain_close(gc_world([gc_box("obstacle", 15.0, 25.0, 1)]))
    assert actions == plain_actions and out["status"] == "succeeded"
    guard = out["evidence"]["guard"]
    assert guard["would_abort"] and guard["verdict"][0] == "early_finger_contact" and not guard["active"]


def test_guarded_close_adjacent_obstacles_give_span_inconsistent(fake_cal):
    """Two separate surfaces, one per finger, each inside its R but 28 mm apart against span_hi + tol = 25 mm:
    span_inconsistent, then the open; the same gate on the part itself (24.3 mm, one group) proceeds."""
    bodies = [gc_box("left", 8.0, 14.0, 1), gc_box("right", 8.0, 14.0, 2)]
    w = gc_world(bodies)
    out, actions = run(w, guarded(w, fake_cal, gc_gate(20.0, 20.0, 24.0), active=True))
    assert (out["status"], out["reason"]) == ("failed", "span_inconsistent")
    verdict = out["evidence"]["guard"]["verdict"][1]
    assert verdict["span_mm"] == pytest.approx(28.0, abs=0.2) and verdict["span_hi_mm"] == 25.0
    assert not any(map(closing, actions[verdict["k"]:]))
    w = gc_world([gc_bar()])
    out, _ = run(w, guarded(w, fake_cal, gc_gate(20.0, 20.0, 24.0), active=True))
    assert out["status"] == "succeeded" and out["evidence"]["guard"]["verdict"] is None


def test_guarded_close_from_80_mm_against_a_curve_calibrated_from_78_5_mm_has_no_free_touch():
    """(judge) The 80 mm joint-limit opening against an empty close calibrated from 78.5 mm: no touch on free
    fingers (the free motion is read at each finger's own opening)."""
    cal_world = gc_world(start_gap_mm=78.5)
    calibration = GripperSettlePrimitive("close", StepView(cal_world.observation()),
                                         config=GripperSettleConfig(min_close_actions=P.CAL_CLOSE_ACTIONS,
                                                                    max_actions=P.CAL_CLOSE_ACTIONS))
    run(cal_world, calibration)
    gripper = GripperCalibration(78.5, tuple(calibration.gap_curve_mm), 0.002, "online")
    w = gc_world(start_gap_mm=80.0)
    out, _ = run(w, guarded(w, (gripper, 0.01), gc_gate(30.0, 30.0, 30.0), active=True))
    assert out["status"] == "succeeded" and out["evidence"]["guard"]["touch"] == {"1": None, "2": None}


def test_guarded_close_fake_chain_start_mismatch_touches_only_at_the_part(fake_cal):
    """(judge) The fake chains close from 79.95 mm while Calibrate+'s empty close ends its first action at ~76.4 mm:
    the step-indexed free model (remedy 2 as designed) falls short at action 3 by more than the touch threshold
    (the false early_finger_contact of every marginal close); read at the fingers' own openings, the first touch of
    each finger is the part's face."""
    gripper, sigma = fake_cal
    assert gripper.empty_curve_mm[0] == pytest.approx(76.42, abs=0.3)
    w = gc_world([gc_bar()], start_gap_mm=79.95)
    gc = guarded(w, fake_cal, gc_gate(20.0, 20.0, GC_BAR_W_MM + 3.0), active=True)
    out, _ = run(w, gc)
    assert out["status"] == "succeeded" and out["evidence"]["guard"]["verdict"] is None
    for finger in ("1", "2"):
        assert out["evidence"]["guard"]["touch"][finger]["s_mm"] == pytest.approx(GC_BAR_W_MM / 2.0, abs=0.1)
    c = [gripper.open_gap_mm, *gripper.empty_curve_mm]
    trace = out["evidence"]["guard"]["trace"]
    k = 3
    inward = trace[k - 2]["s_mm"][0] - trace[k - 1]["s_mm"][0]
    assert (c[k - 1] - c[k]) / 2.0 - inward >= gc.touch_th.value          # the step-indexed model would abort here


# ================================================================================================ 8. restated v1 constants
def test_restated_v1_constants_equal_their_sources():
    assert P.G12_STALL_FRACTION == AnchoredMove.STALL_FRACTION
    assert P.G12_RATIO_BOUNDS == AnchoredMove.FREE_RATIO_BOUNDS
    assert P.CAL_PROBE_MOVES == v1.PROBE_MOVES
    assert P.M13_RETREAT_M == v1.RETREAT_M
    assert P.LOWER_FAST_COMMAND_M == v1.PLACE_FAST_COMMAND_M
    assert P.OPEN_SETTLE_MIN_ACTIONS == v1.OPEN_SETTLE.min_open_actions
    assert P.M12_OPEN_TOL_MM == v1.OPEN_TOLERANCE_MM
    assert P.M3_REORIENT_RAD == v1.MAX_HELD_REORIENT_RAD
    assert P.G8_FLOOR_MM == v1.PLACE_GAP_RISE_MM
    defaults = inspect.signature(AnchoredMove.__init__).parameters
    assert (P.P15_COMMAND_M, P.P15_LATERAL_CAP_M) == (defaults["command_m"].default, defaults["lateral_cap_m"].default)
    assert P.G7_TOTAL_FLOOR_MM == WaypointReach.GAP_RISE_MM and P.G7_STEP_FLOOR_MM == WaypointReach.GAP_STEP_RISE_MM
    assert P.G9_FINGER_CONTACT_MM == WaypointReach.FINGER_CONTACT_MM
    assert P.CAL_MIN_CLOSING_MM == Calibrate.MIN_CLOSING_MOTION_MM
    assert GripperCalibration(77.0, (1.0,), 0.0, "online").touch_threshold_mm == P.G3_FLOOR_MM
    assert GripperCalibration(77.0, (1.0,), 0.1, "online").touch_threshold_mm == pytest.approx(P.G3_SIGMA_K * 0.1)
    # the Calibrate GripperSettleConfig values (contact_skills.py:510, :528, :532), read from v1's own run
    w = world(gap_noise_mm=0.0)
    cal = Calibrate(StepView(w.observation()))
    run(w, cal)
    assert cal.close.settle.config.min_close_actions == P.CAL_CLOSE_ACTIONS == cal.close.settle.config.max_actions
    assert cal.open.settle.config.min_open_actions == P.CAL_OPEN_MIN_ACTIONS
    assert cal.open.settle.config.max_actions == P.CAL_CLOSE_ACTIONS and cal.hold.steps == P.CAL_HOLD_STEPS
    assert P.M5_SATURATED_STEPS == Align.BLOCKED_STEPS and P.M5_SATURATED_REALIZED == Align.BLOCKED_MAX_REALIZED
    assert (P.HAND_YAW_STALL_STEPS, P.HAND_YAW_STALL_RAD) == (Orient.STALL_STEPS, Orient.STALL_MIN_DECREASE_RAD)
    assert P.CAL_YAW_MAX_STEPS == P.HAND_YAW_STALL_STEPS                       # the yaw-stage cap is the stall window
    assert math.radians(P.B4_QUIET_DEG) == Align.SETTLE_ROTATION_RAD == HoldPrimitive.QUIET_ROTATION_RAD
    assert P.B4_QUIET_MM / 1000.0 == Align.SETTLE_TRANSLATION_M == HoldPrimitive.QUIET_TRANSLATION_M
    assert (P.B4_QUIET_STEPS, P.B4_MAX_STEPS) == (Align.SETTLE_QUIET_STEPS, Align.SETTLE_MAX_STEPS)
    assert P.K2_TWIST_RAD < robot_facts.ROTATION_CAP_RAD_PER_STEP


# ================================================================================================ 9. B4 settle
def test_b4_settle_after_a_3cm_move_on_the_second_order_plant():
    w = world(F.planar_region_scene(), eef=(-0.20, 0.0, 1.10), start_tilt_deg=0.0, dynamics="second_order")
    move = AnchoredMove(StepView(w.observation()), axis=(1, 0, 0), travel_m=0.03, gripper="open",
                        command_m=0.03)
    out, _ = run(w, move)
    assert out["reason"] == "travel_reached"
    settle = P.b4_settle(StepView(w.observation()), gripper="open")
    assert settle.steps == P.B4_MAX_STEPS and settle.until_quiet and settle.quiet_steps == P.B4_QUIET_STEPS
    out, actions = run(w, settle)
    assert out["evidence"]["quiet"] and len(actions) <= P.B4_MAX_STEPS


# ================================================================================================ helpers, reserved
def test_fit_hinge_centre_recovers_the_centre_and_reports_sigma():
    rng = np.random.default_rng(1)
    centre, r0 = np.array([0.1, -0.2]), np.array([0.012, 0.004])
    angles = np.linspace(0.0, 0.5, 11)
    pts = np.array([centre + F._rot_axis((0, 0, 1), a)[:2, :2] @ r0 for a in angles])
    fit = P.fit_hinge_centre(pts, angles)
    np.testing.assert_allclose(fit["centre_xy"], centre, atol=1e-9)
    assert fit["sigma_m"] < 1e-9 and fit["points"] == 11
    noisy = P.fit_hinge_centre(pts + rng.normal(0, 0.0005, pts.shape), angles)
    short = P.fit_hinge_centre(pts[:4] + rng.normal(0, 0.0005, (4, 2)), angles[:4])
    assert short["sigma_m"] > noisy["sigma_m"] > 0.0
    assert P.fit_hinge_centre(pts[:2], angles[:2]) is None


def test_reserved_primitives_raise():
    view = _synthetic_view((0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.03, -0.03))
    for cls in P.RESERVED_PRIMITIVES:
        assert cls.__name__ in ("DOFProbe", "FollowAxis", "PushAlong", "RayServo", "RayDescent")
        with pytest.raises(NotImplementedError):
            cls(view)


def test_p15_thresholds_and_version():
    names = [(t.name, t.value, t.label) for t in P.p15_thresholds()]
    assert names == [("P15_command_m", P.P15_COMMAND_M, "DD"), ("P15_lateral_cap_m", P.P15_LATERAL_CAP_M, "DD")]
    assert P.PRIMITIVES_VERSION == "agent-speedup-v2-primitives-3"


# ================================================================================================ T0 lint and firewall
def test_module_passes_lint_literal_audit_and_firewall():
    path = Path(th.PACKAGE_DIR) / "agent_speedup_v2_primitives.py"
    vocabulary = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(path, vocabulary) == []
    assert th.lint_module(path, vocabulary, labels=th.CONSTANT_LABELS) == []
    assert th.literal_audit(path) == []
    assert "agent_speedup_v2_primitives.py" in th.CONTROL_MODULES
    assert not th.LABEL_MERGE_ERRORS.get("agent_speedup_v2_primitives.py")
    import ast
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):                                          # function-level imports included
        if isinstance(node, ast.ImportFrom):
            imported |= {node.module or ""} | {alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
    v2 = {name.split(".")[-1] for name in imported if "agent_speedup_v2" in name}
    assert v2 <= {"agent_speedup_v2_types", "agent_speedup_v2_thresholds", "agent_speedup_v2_robot"}, v2
    assert not any("agent_speedup_v1" in name for name in imported)
    v1_modules = {name.split(".")[-1] for name in imported if name and not name.startswith("agent_speedup")} - {
        "math", "numpy", "__future__", "annotations"}
    assert v1_modules <= {"contact_skills", "speed_first_gripper", "action"} | {
        n for n in v1_modules if not (Path(th.PACKAGE_DIR) / f"{n}.py").is_file()}, v1_modules
    source = path.read_text()
    assert "start_poses" not in source and "measured_tasks" not in source


def test_every_primitive_has_no_pointing_request_and_json_outcomes():
    w = knob_world()
    view = StepView(w.observation())
    built = [P.CalibratePlus(view, robot=ROBOT),
             P.ParallaxProbe(view, recorder=None, gripper="close"),
             segment(w, -1), hinge(w, -1),
             P.HeldReach(view, [0.0, 0.0, 0.01], hold_floor_mm=1.0, gap_ref_mm=view.gap_mm),
             P.GuardedLower(view, phase="slow", travel_m=0.01, aim_grip_xy=view.position[:2], drift_abort_m=0.005,
                            hold_floor_mm=1.0),
             P.Touchdown(view, lift_m=0.01, hold_floor_mm=1.0, drift_abort_m=0.005),
             P.ReleaseVerify(view, g_open_mm=78.0)]
    for primitive in built:
        assert primitive.pointing_request is None, primitive.name


# ================================================================================ Wave-1 acceptance: BO's observers
def _bo():
    try:
        from gpt_field import agent_speedup_v2_observers as observers
    except ImportError:
        pytest.skip("BO's agent_speedup_v2_observers is not built yet (the Wave-1 acceptance run executes this)")
    return observers


def _real_wedge(observers, mode, **kw):
    return observers.WedgeObserver(mode=mode, robot=ROBOT, sigma_g_mm_per_step=0.002, hold_floor_mm=1.5, **kw)


def test_real_wedge_free_and_blocked_at_step_12():
    observers = _bo()
    w = knob_world()
    out, _ = run(w, segment(w, -1, wedge=_real_wedge(observers, "sense"), rotation=Scripted("part_rotation")))
    assert (out["reason"], out["actions"]) == ("free_sense", 12)
    b = knob_world()
    blocked, _ = run(b, segment(b, +1, wedge=_real_wedge(observers, "sense"), rotation=Scripted("part_rotation")))
    assert (blocked["reason"], blocked["actions"]) == ("blocked_sense", 12)
    wedge = _real_wedge(observers, "sense", after_reversal=True, blocked_steps=blocked["actions"])
    out, _ = run(b, segment(b, -1, wedge=wedge, rotation=Scripted("part_rotation")))
    assert out["reason"] == "free_sense" and out["actions"] <= math.ceil(1.5 * 12) + 16


def test_real_wedge_end_stop_within_five_steps_of_the_limit():
    """Contract §C8 test 4 (end stop after a free classification, <= 5 steps after the limit) with BO's K4w. On the
    fake the reversal's free decision leaves only ~4 steady steps before the lower limit, so a K4w median that
    includes the post-stop rates would never fire (see StandInWedge); BO's K4w takes the median over the rates
    before the candidate windows (the earlier cross-module finding, since applied), and this check is enforced."""
    observers = _bo()
    w = knob_world()
    first, _ = run(w, segment(w, -1, wedge=_real_wedge(observers, "sense"), rotation=Scripted("part_rotation")))
    rev, _ = run(w, segment(w, +1, wedge=_real_wedge(observers, "sense", after_reversal=True,
                                                     blocked_steps=first["actions"]),
                            rotation=Scripted("part_rotation")))
    assert first["reason"] == rev["reason"] == "free_sense"
    limit_at = []

    def watch(world_, primitive, n):
        if not limit_at and world_.joint("knob") <= -0.0049:
            limit_at.append(n)
    out, _ = run(w, hinge(w, +1, wedge=_real_wedge(observers, "end_stop"), rotation=NO_ROTATION), after_step=watch)
    assert out["reason"] == "end_stop" and limit_at and 0 <= out["actions"] - limit_at[0] <= 5


@pytest.mark.parametrize("lever, sense, reason, decided_by", [
    (False, +1, "blocked_sense", "part_rotation"),          # round part: the wedge reads free, the part does not turn
    (False, -1, "free_sense", "both"),
    (True, +1, "blocked_sense", "both"),
    (True, -1, "free_sense", "both")])
def test_real_observers_decide_the_twist_sense_on_the_rendered_fake(lever, sense, reason, decided_by):
    """BO's WedgeObserver and PartRotationObserver (a rigid patch on the disk 35 mm from the grip, located by the
    fake's projection truth) inside TwistSegment on the rendered fake: the [C-29] rows end to end."""
    observers = _bo()
    scene = F.rotary_part_scene(lever=lever)
    w = F.FakeTwoViewSim(scene, eef=scene.info["start_eef"], start_tilt_deg=0.0, images=True)
    F.scripted_grasp_rotary(w)
    F.scripted_gripper(w, 1.0, 8)
    cx, cy = F.TASK2_KNOB_CENTRE_XY
    uv, _ = w.project("wrist", (cx + 0.035, cy, 0.918))
    rotation = observers.PartRotationObserver(robot=ROBOT, patch={"uv": tuple(float(v) for v in uv)}, axis_uv=None)
    out, _ = run(w, segment(w, sense, wedge=_real_wedge(observers, "sense"), rotation=rotation))
    assert (out["reason"], out["actions"], out["evidence"]["decided_by"]) == (reason, 12, decided_by)


# ======================================================================== RC-8 replay data (decision §5.1 item 4)
# Remedy 2 T1 test 9's inputs, exactly the sources of review1 contact/scripts/stall_floor.py (out/stall_floor_0.08.json:
# 76 free legs, 4 contacts): the rulings lens's stall-off ParallaxProbe legs and its open-hand descents onto the
# region top and the support beside the rotary part (runs/agent_speedup_v2_review1_rulings_stall_ep{1,2}), and the
# grasp lens's 32 legs truncated at their as-built false stall (review1 grasp_tolerance/probe_analysis.json, ratios
# recorded to 3 decimals). Ratios rounded to 6 decimals; the contacts keep their first 30 steps (collapse at 20-21).
# (source, episode, part id, stage, truncated at its as-built false stall, realized / commanded ratios)
STALL_REPLAY_FREE_LEGS = (
    ('rulings_nostall', 1, 'E', 'r1_out', False, (
        0.100968, 0.191839, 0.214016, 0.226943, 0.236726, 0.243914, 0.248898, 0.252106, 0.253982,
    )),
    ('rulings_nostall', 1, 'E', 'r1_back', False, (
        0.114392, 0.193738, 0.21634, 0.230283, 0.24047, 0.24773, 0.252413, 0.255157, 0.256538, 0.393386,
    )),
    ('rulings_nostall', 1, 'E', 'r0_out', False, (
        0.093692, 0.161198, 0.17125, 0.180222, 0.188026, 0.196148, 0.204627, 0.212922, 0.220522, 0.227229, 0.267794,
    )),
    ('rulings_nostall', 1, 'E', 'r0_back', False, (
        0.090736, 0.14674, 0.162929, 0.175062, 0.187346, 0.199646, 0.211457, 0.222367, 0.232094, 0.240474, 0.247445,
        0.253026,
    )),
    ('rulings_nostall', 1, 'B', 'r1_out', False, (
        0.098923, 0.184902, 0.205426, 0.218685, 0.228521, 0.236169, 0.241933, 0.246078, 0.248933, 0.462452,
    )),
    ('rulings_nostall', 1, 'B', 'r1_back', False, (
        0.107124, 0.186509, 0.2082, 0.221944, 0.23179, 0.239036, 0.244105, 0.247432, 0.249466, 0.250605,
    )),
    ('rulings_nostall', 1, 'B', 'r0_out', False, (
        0.099687, 0.169327, 0.17896, 0.186469, 0.192573, 0.198849, 0.205745, 0.212582, 0.219083, 0.22504, 0.32115,
    )),
    ('rulings_nostall', 1, 'B', 'r0_back', False, (
        0.092246, 0.15146, 0.167558, 0.179039, 0.190577, 0.20225, 0.213461, 0.223744, 0.232839, 0.240659, 0.247165,
        0.325757,
    )),
    ('rulings_nostall', 1, 'C', 'r1_out', False, (
        0.101695, 0.194455, 0.217528, 0.23012, 0.239331, 0.246047, 0.250712, 0.253636, 0.255241,
    )),
    ('rulings_nostall', 1, 'C', 'r1_back', False, (
        0.116049, 0.197118, 0.219911, 0.233422, 0.242953, 0.249483, 0.253657, 0.256084, 0.257349, 0.416565,
    )),
    ('rulings_nostall', 1, 'C', 'r0_out', False, (
        0.091866, 0.152707, 0.155903, 0.157466, 0.162329, 0.166532, 0.171518, 0.177374, 0.183664, 0.189974, 0.196018,
        0.20165,
    )),
    ('rulings_nostall', 1, 'C', 'r0_back', False, (
        0.072594, 0.117251, 0.126835, 0.134912, 0.142346, 0.151224, 0.161515, 0.172558, 0.183924, 0.195298, 0.206415,
        0.217048, 0.22702, 0.236277, 0.493784,
    )),
    ('rulings_nostall', 1, 'D', 'r1_out', False, (
        0.102025, 0.195555, 0.219138, 0.231863, 0.241126, 0.247853, 0.252466, 0.255405, 0.257097,
    )),
    ('rulings_nostall', 1, 'D', 'r1_back', False, (
        0.116691, 0.198407, 0.221363, 0.234862, 0.244338, 0.250806, 0.254916, 0.257272, 0.258406, 0.414902,
    )),
    ('rulings_nostall', 1, 'D', 'r0_out', False, (
        0.091431, 0.152962, 0.157036, 0.15879, 0.163538, 0.167532, 0.17221, 0.177726, 0.183737, 0.189867, 0.195842,
        0.201482,
    )),
    ('rulings_nostall', 1, 'D', 'r0_back', False, (
        0.072977, 0.119038, 0.12859, 0.136615, 0.143644, 0.15206, 0.161942, 0.172685, 0.183834, 0.195049, 0.206052,
        0.216605, 0.226514, 0.235703,
    )),
    ('rulings_nostall', 1, 'A', 'r1_out', False, (
        0.099184, 0.186904, 0.208914, 0.222447, 0.232127, 0.239542, 0.245011, 0.248854, 0.251437, 0.613146,
    )),
    ('rulings_nostall', 1, 'A', 'r1_back', False, (
        0.10634, 0.189137, 0.212219, 0.225701, 0.235176, 0.241997, 0.246671, 0.249689, 0.251512, 0.252524,
    )),
    ('rulings_nostall', 1, 'A', 'r0_out', False, (
        0.104067, 0.18544, 0.200308, 0.208876, 0.214453, 0.219695, 0.225287, 0.230688, 0.235741, 0.267047,
    )),
    ('rulings_nostall', 1, 'A', 'r0_back', False, (
        0.101664, 0.173994, 0.193455, 0.204443, 0.214011, 0.222978, 0.231219, 0.238567, 0.24487, 0.249969, 0.363123,
    )),
    ('rulings_nostall', 2, 'D', 'r1_out', False, (
        0.10166, 0.195072, 0.218735, 0.231486, 0.240795, 0.24759, 0.252279, 0.255287, 0.257035,
    )),
    ('rulings_nostall', 2, 'D', 'r1_back', False, (
        0.116484, 0.197891, 0.220813, 0.234371, 0.243937, 0.250507, 0.25472, 0.25717, 0.258385, 0.411484,
    )),
    ('rulings_nostall', 2, 'D', 'r0_out', False, (
        0.093648, 0.156616, 0.161038, 0.163037, 0.168041, 0.172318, 0.177241, 0.182965, 0.189129, 0.195393, 0.201485,
        0.240964,
    )),
    ('rulings_nostall', 2, 'D', 'r0_back', False, (
        0.07351, 0.121299, 0.131813, 0.140723, 0.148654, 0.157881, 0.168424, 0.179657, 0.191102, 0.202403, 0.21328,
        0.223507, 0.233011, 0.241833,
    )),
    ('rulings_nostall', 2, 'E', 'r1_out', False, (
        0.101816, 0.193169, 0.21516, 0.227786, 0.237377, 0.244571, 0.249662, 0.253021, 0.255096,
    )),
    ('rulings_nostall', 2, 'E', 'r1_back', False, (
        0.114589, 0.194041, 0.216722, 0.230713, 0.240944, 0.248164, 0.252888, 0.255724, 0.257203, 0.382321,
    )),
    ('rulings_nostall', 2, 'E', 'r0_out', False, (
        0.093892, 0.161435, 0.171411, 0.180428, 0.188242, 0.19619, 0.204477, 0.212668, 0.220302, 0.227071, 0.268248,
    )),
    ('rulings_nostall', 2, 'E', 'r0_back', False, (
        0.092084, 0.148565, 0.163873, 0.175586, 0.187872, 0.20018, 0.21198, 0.222865, 0.232554, 0.240889, 0.247812,
        0.253343,
    )),
    ('rulings_nostall', 2, 'E', 'r1_out', False, (
        0.101592, 0.192603, 0.214633, 0.227441, 0.237226, 0.244608, 0.249861, 0.253297, 0.255323,
    )),
    ('rulings_nostall', 2, 'E', 'r1_back', False, (
        0.114375, 0.193554, 0.216222, 0.230336, 0.240658, 0.24791, 0.2527, 0.255597, 0.257126, 0.375586,
    )),
    ('rulings_nostall', 2, 'E', 'r0_out', False, (
        0.094809, 0.164564, 0.175988, 0.185687, 0.193741, 0.201906, 0.210266, 0.218368, 0.225791, 0.232228, 0.36969,
    )),
    ('rulings_nostall', 2, 'E', 'r0_back', False, (
        0.088899, 0.148229, 0.16587, 0.17883, 0.191624, 0.204178, 0.215992, 0.226672, 0.235974, 0.243784, 0.250092,
        0.254968,
    )),
    ('rulings_nostall', 2, 'C', 'r1_out', False, (
        0.102951, 0.196211, 0.218774, 0.231248, 0.240391, 0.24702, 0.251487, 0.25417, 0.255621,
    )),
    ('rulings_nostall', 2, 'C', 'r1_back', False, (
        0.11565, 0.196176, 0.21882, 0.232352, 0.241985, 0.248723, 0.253112, 0.25564, 0.256878, 0.372944,
    )),
    ('rulings_nostall', 2, 'C', 'r0_out', False, (
        0.092013, 0.154144, 0.15835, 0.160818, 0.166199, 0.170906, 0.1764, 0.182605, 0.189118, 0.195598, 0.201774,
        0.228692,
    )),
    ('rulings_nostall', 2, 'C', 'r0_back', False, (
        0.072441, 0.118625, 0.129182, 0.138143, 0.146496, 0.15626, 0.167195, 0.178697, 0.190343, 0.201802, 0.212813,
        0.223159, 0.232775, 0.241645,
    )),
    ('rulings_nostall', 2, 'C', 'r1_out', False, (
        0.101497, 0.193892, 0.216619, 0.229098, 0.238246, 0.245019, 0.249645, 0.252437, 0.253908,
    )),
    ('rulings_nostall', 2, 'C', 'r1_back', False, (
        0.11549, 0.196399, 0.219008, 0.232445, 0.24192, 0.248411, 0.252576, 0.255127, 0.256412, 0.414802,
    )),
    ('rulings_nostall', 2, 'C', 'r0_out', False, (
        0.091502, 0.152597, 0.156213, 0.157971, 0.162974, 0.167431, 0.172605, 0.17852, 0.184813, 0.191111, 0.197143,
        0.202778,
    )),
    ('rulings_nostall', 2, 'C', 'r0_back', False, (
        0.073912, 0.119171, 0.129091, 0.137143, 0.144827, 0.153834, 0.164064, 0.175014, 0.18626, 0.197493, 0.208439,
        0.218888, 0.228679, 0.237789, 0.576334,
    )),
    ('rulings_nostall', 2, 'B', 'r1_out', False, (
        0.096206, 0.181224, 0.201703, 0.215299, 0.225562, 0.233575, 0.239725, 0.244263, 0.247501, 0.379352,
    )),
    ('rulings_nostall', 2, 'B', 'r1_back', False, (
        0.107171, 0.185227, 0.206789, 0.220454, 0.230409, 0.237811, 0.243043, 0.246527, 0.248702, 0.249964,
    )),
    ('rulings_nostall', 2, 'B', 'r0_out', False, (
        0.099921, 0.168927, 0.178015, 0.185189, 0.191216, 0.197338, 0.204145, 0.210937, 0.217419, 0.223386, 0.295792,
    )),
    ('rulings_nostall', 2, 'B', 'r0_back', False, (
        0.09079, 0.148892, 0.164806, 0.176352, 0.188017, 0.199928, 0.211335, 0.221768, 0.231077, 0.239145, 0.245912,
        0.284033,
    )),
    ('grasp_probe_stalled', 1, 'B', 'r0_back', True, (0.088, 0.144, 0.159,)),
    ('grasp_probe_stalled', 1, 'B', 'r0_back', True, (0.089, 0.145, 0.161,)),
    ('grasp_probe_stalled', 1, 'C', 'r0_out', True, (0.092, 0.154, 0.157,)),
    ('grasp_probe_stalled', 1, 'E', 'r0_back', True, (0.09, 0.146, 0.162,)),
    ('grasp_probe_stalled', 1, 'E', 'r0_back', True, (0.088, 0.147, 0.165,)),
    ('grasp_probe_stalled', 1, 'E', 'r0_back', True, (0.087, 0.144, 0.162,)),
    ('grasp_probe_stalled', 2, 'B', 'r0_back', True, (0.088, 0.145, 0.161,)),
    ('grasp_probe_stalled', 2, 'C', 'r0_out', True, (0.091, 0.152, 0.155, 0.156,)),
    ('grasp_probe_stalled', 2, 'C', 'r0_out', True, (0.092, 0.153, 0.157,)),
    ('grasp_probe_stalled', 2, 'D', 'r0_out', True, (0.091, 0.153, 0.157,)),
    ('grasp_probe_stalled', 2, 'D', 'r0_out', True, (0.091, 0.153, 0.158,)),
    ('grasp_probe_stalled', 2, 'E', 'r0_back', True, (0.09, 0.146, 0.162,)),
    ('grasp_probe_stalled', 2, 'E', 'r0_back', True, (0.088, 0.146, 0.164,)),
    ('grasp_probe_stalled', 2, 'E', 'r0_back', True, (0.086, 0.144, 0.161,)),
    ('grasp_probe_stalled', 3, 'B', 'r0_back', True, (0.088, 0.143, 0.159,)),
    ('grasp_probe_stalled', 3, 'B', 'r0_back', True, (0.088, 0.145, 0.161,)),
    ('grasp_probe_stalled', 3, 'D', 'r0_out', True, (0.09, 0.151, 0.155, 0.157,)),
    ('grasp_probe_stalled', 3, 'D', 'r0_out', True, (0.091, 0.153, 0.157,)),
    ('grasp_probe_stalled', 4, 'B', 'r0_back', True, (0.088, 0.143, 0.159,)),
    ('grasp_probe_stalled', 4, 'B', 'r0_back', True, (0.089, 0.145, 0.161,)),
    ('grasp_probe_stalled', 4, 'D', 'r0_back', True, (0.08, 0.133, 0.147, 0.157,)),
    ('grasp_probe_stalled', 5, 'B', 'r0_back', True, (0.088, 0.144, 0.159,)),
    ('grasp_probe_stalled', 5, 'B', 'r0_back', True, (0.089, 0.146, 0.161,)),
    ('grasp_probe_stalled', 5, 'C', 'r0_back', True, (0.059, 0.097, 0.1, 0.104, 0.109,)),
    ('grasp_probe_stalled', 5, 'C', 'r0_out', True, (0.092, 0.152, 0.155, 0.156,)),
    ('grasp_probe_stalled', 5, 'C', 'r0_back', True, (0.058, 0.097, 0.101, 0.103, 0.107,)),
    ('grasp_probe_stalled', 5, 'D', 'r0_back', True, (0.049, 0.088, 0.088, 0.083, 0.079,)),
    ('grasp_probe_stalled', 5, 'D', 'r0_out', True, (0.091, 0.153, 0.158,)),
    ('grasp_probe_stalled', 5, 'D', 'r0_back', True, (0.057, 0.098, 0.1, 0.099, 0.1,)),
    ('grasp_probe_stalled', 5, 'E', 'r0_back', True, (0.088, 0.142, 0.158,)),
    ('grasp_probe_stalled', 5, 'E', 'r0_back', True, (0.086, 0.143, 0.161,)),
    ('grasp_probe_stalled', 5, 'E', 'r0_back', True, (0.09, 0.146, 0.162,)),
)
# (episode, part id, ratios of an open-hand descent onto the surface, stall stop off; first 30 steps)
STALL_REPLAY_CONTACTS = (
    (1, 'B', (
        0.195502, 0.293632, 0.323793, 0.332605, 0.332903, 0.327804, 0.321192, 0.312933, 0.304441, 0.296393, 0.289114,
        0.282776, 0.277352, 0.271988, 0.265751, 0.259882, 0.255054, 0.251454, 0.24908, 0.041132, -0.015131, -0.002231,
        0.000644, 0.001075, 0.000643, 0.000361, 0.000279, 0.000222, 0.000164, 0.000107,
    )),
    (1, 'A', (
        0.123851, 0.233189, 0.257093, 0.26302, 0.264919, 0.265695, 0.266032, 0.266121, 0.26604, 0.265838, 0.265554,
        0.265224, 0.264877, 0.264536, 0.26422, 0.263938, 0.263699, 0.263503, 0.26335, 0.092375, -0.001127, 0.003371,
        0.002437, 0.001606, 0.001096, 0.000856, 0.000816, 0.000688, 0.000521, 0.000355,
    )),
    (2, 'B', (
        0.114487, 0.232169, 0.258089, 0.264418, 0.266599, 0.267789, 0.268608, 0.269175, 0.269517, 0.269649, 0.269597,
        0.269388, 0.269054, 0.268629, 0.268145, 0.267634, 0.267124, 0.266637, 0.266192, 0.088012, -0.014472, -0.002866,
        -0.000366, -4.1e-05, -7.3e-05, -6.3e-05, -3e-06, 4e-06, 4e-06, 4e-06,
    )),
    (2, 'A', (
        0.114733, 0.221698, 0.247906, 0.255523, 0.25934, 0.262196, 0.264471, 0.266193, 0.267373, 0.268131, 0.268498,
        0.268523, 0.268271, 0.267814, 0.267219, 0.266548, 0.265855, 0.265184, 0.264567, 0.264027, 0.069571, -0.000268,
        0.003095, 0.002184, 0.001446, 0.000999, 0.000827, 0.00079, 0.00066, 0.000496,
    )),
)

# RS-3 run 002 (runs/agent_speedup_v2_bm_r1_rs3_stall_002, scripts/agent_speedup_v2_stall_check.py; real task 2,
# episodes 1-5, 500 complete RC-8 legs of 100 grip-mode hovers, 0 stalls): the 6 legs with the smallest stall
# margin (arrival descents, margin 50-65 %) and the 6 with the lowest own free ratio (slow r0_back legs,
# 0.086-0.109, 5 of which the as-built leg stalls at step 5). Ratios rounded to 6 decimals.
# (episode, part id, stage, realized / commanded ratios)
STALL_REPLAY_RS3_FREE_LEGS = (
    (2, 'B', 'hover_arrival', (
        0.213568, 0.320195, 0.346683, 0.351743, 0.346837, 0.337214, 0.327443, 0.317234, 0.307123, 0.594032,
    )),
    (1, 'A', 'hover_arrival', (
        0.149407, 0.256027, 0.276975, 0.280076, 0.279806, 0.2783, 0.275692, 0.272905, 0.270169, 0.267632,
        0.265402, 0.442895,
    )),
    (4, 'A', 'hover_arrival', (
        0.147607, 0.254508, 0.275533, 0.278729, 0.278628, 0.277291, 0.274889, 0.272313, 0.269781, 0.267429,
        0.265359, 0.41711,
    )),
    (3, 'E', 'hover_arrival', (
        0.142866, 0.255983, 0.278891, 0.282451, 0.281865, 0.28093, 0.278647, 0.276095, 0.273515, 0.271031,
        0.268746, 0.506503,
    )),
    (4, 'D', 'hover_arrival', (
        0.123551, 0.234619, 0.256923, 0.261153, 0.261506, 0.260995, 0.260301, 0.259593, 0.258923, 0.258314,
        0.257779, 0.257329,
    )),
    (4, 'C', 'hover_arrival', (
        0.121738, 0.234131, 0.256764, 0.261041, 0.261416, 0.260932, 0.260264, 0.259578, 0.258926, 0.258332,
        0.257809, 0.257368, 0.832957,
    )),
    (5, 'D', 'r0_back', (
        0.050309, 0.088983, 0.088909, 0.08602, 0.084009, 0.086103, 0.089004, 0.092406, 0.097042, 0.102861,
        0.109769, 0.117669, 0.126456, 0.136032, 0.146293, 0.157126, 0.168403, 0.179984, 0.191716, 0.203434,
    )),
    (5, 'C', 'r0_back', (
        0.056455, 0.095436, 0.098494, 0.09981, 0.103855, 0.107741, 0.113331, 0.120427, 0.128725, 0.138033,
        0.148162, 0.158928, 0.170144, 0.181626, 0.193191, 0.204653, 0.215826, 0.23699,
    )),
    (5, 'C', 'r0_back', (
        0.058791, 0.097246, 0.100824, 0.103407, 0.107888, 0.112446, 0.11878, 0.126505, 0.135376, 0.145178,
        0.155702, 0.166752, 0.178142, 0.189676, 0.201157, 0.212386, 0.223184,
    )),
    (5, 'D', 'r0_back', (
        0.059786, 0.100489, 0.103633, 0.103934, 0.107021, 0.109991, 0.114622, 0.120813, 0.128258, 0.13675,
        0.146116, 0.156188, 0.166797, 0.177771, 0.188937, 0.200115, 0.211129,
    )),
    (5, 'C', 'r0_back', (
        0.058483, 0.095255, 0.099672, 0.105187, 0.110488, 0.117007, 0.125166, 0.134598, 0.145036, 0.156222,
        0.16791, 0.179864, 0.191858, 0.203682, 0.215133, 0.226018, 0.236165,
    )),
    (1, 'D', 'r0_back', (
        0.061006, 0.101784, 0.106132, 0.108851, 0.113263, 0.117738, 0.124013, 0.131752, 0.140632, 0.150405,
        0.160855, 0.171774, 0.182963, 0.194224, 0.205366, 0.216204, 0.230185,
    )),
)
# (episode, site, the first 30 ratios of the stall-off v1 reference descent from 5 cm onto B's plate and onto the
# footprint-clear table site; reference collapse at step 21 on both)
STALL_REPLAY_RS3_CONTACTS = (
    (1, 'plate', (
        0.114233, 0.232781, 0.25844, 0.264527, 0.266516, 0.267553, 0.268255, 0.26874, 0.269033, 0.269149,
        0.269106, 0.268927, 0.268639, 0.268271, 0.267851, 0.267405, 0.266958, 0.266531, 0.26614, 0.160192,
        -0.0188, -0.00512, -0.000641, -0.000102, -8.8e-05, -0.000101, -1.3e-05, 8e-06, 9e-06, 8e-06,
    )),
    (1, 'table', (
        0.108893, 0.215609, 0.237304, 0.243424, 0.246457, 0.248974, 0.251478, 0.25398, 0.256385, 0.258591,
        0.260524, 0.26214, 0.263422, 0.264379, 0.265035, 0.265426, 0.265596, 0.265591, 0.265456, 0.265233,
        0.086684, -0.013695, -0.000668, 0.000358, -0.000135, -0.00023, -0.000242, -0.000276, -0.000282,
        -0.00028,
    )),
)
