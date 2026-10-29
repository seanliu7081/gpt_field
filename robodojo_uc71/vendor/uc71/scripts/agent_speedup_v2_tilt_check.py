"""RS-7 (review-1 decision §5.2; extended by review-2 decision §2.7 / §6.2; owner BO): does O-1 see the carried part
ROTATE during the load-test lift?

A policy-free, zero-call TEST TOOL on real LIBERO task 2 (libero_10 task 2). Per episode it calibrates once (BM's
``CalibratePlus`` for the gripper quantities, then the start survey of ``agent_speedup_v2_survey_probe.calibrate``
for the pipeline's ``ExternalModel``), then for each variant resets the episode and drives one forced grasp of the
thin part C of the free object D with the real primitives: hover, ``Orient``, B4 settle, guarded ``AnchoredMove``
descent to the plan's g* (contact stop at the calibrated G3), the grasp close (``GripperSettlePrimitive``,
``classify_close`` with G5), and the load test exactly as contract §C10.2 item 10 orders it: the
``CoMotionObserver`` for D started on the load-test start view and updated on every transition of
``VerifyHold(+z, LIFT_TEST_M, G6 margin)`` (preempt=False semantics: its last report is the verdict the relation
reads after the lift). A diagnostic 6 cm lift and a 10-step hold follow, for the truth tilt only.

Variants (the plan's jaw turned by ``dyaw`` about world z and its grip point moved ``dj`` along the jaw):
``clean`` (0 deg, 0 mm: the reference plan) and the RS-7 classes (``RS7_CLASSES``), each the four sign combinations
of a jaw error with 4 mm along the jaw: the review-1 class of decision §1.3 (jaw +-10 deg: ``p10p4``, ``p10m4``,
``m10m4``, ``m10p4``) and the review-2 yaw-tail classes of decision 2 §6.2 (+-8 deg: ``p8p4`` ...; +-13.4 deg, the
largest accepted wrong axis measured live: ``p13.4p4``, ``p13.4m4``, ``m13.4m4``, ``m13.4p4``); smaller ones for
diagnosis (``p5p2``, ``m5m2``, ``p0p4``, ``p0m4``, ``p7p0``, ``m7p0``). ``dj`` is COMMANDED: the realized lateral
error at the close (``pre_close.err_jaw_mm``, against the reference plan's grip point) carries an episode-dependent
offset (the clean reference holds realize -0.4..-3.7 mm), so in the review-2 runs (episodes 0-9, fit and held-out)
the "+4 mm" sets realize -0.8..+3.6 mm (episode 9 -0.8..-0.6, episode 7 +0.0..+0.1) and the "-4 mm" sets
-4.7..-8.7 mm (fresh episodes 10-19: -0.7..+3.4 and -5.1..-7.5). The decision's "4 mm lateral" is therefore realized
on the -4 mm side only; commanded vs realized is for the lead to rule (review-2 BO-R2-2). The sign matters; the
summary scores every forced set, the diagnosis-only ``REALIZED_P4_SETS`` (``p8p6``, ``m8p6``, ``p10p6`` ...
``m13.4p6``: +6 mm commanded, realized +1.4..+5.5 mm, 28/60 at >= +4 mm) are reported in ``by_variant``, and
``realized_lateral`` (``--realized DIR ...`` pools runs) bins the held forced holds of each class by jaw sign and
realized lateral error (>= 4 mm either side) and scores each bin as a sign set; neither ever gates.

ORACLE use (none of it reaches an observer input except as the stand-ins named here): (1) plan cells: the reference
plan is BP's ``plan_contact`` (height_stop) on PartRecords built from the evaluator's collision cells of C and D (the
grasp-tolerance lens's reference, review-1 §1.3 "forced with the reference jaw"); (2) pointing / registration
stand-ins for the observer's inputs: the external template is placed at the external projection of the ORACLE
point of D (what ``HeldFrameEstimator.prior_at_close`` projects from D's registered footprint and top), and the
observer's wrist patches are selected by ``observers.select_wrist_patches`` on D's ORACLE visible wrist mask at the
load-test start, back-projected at D's truth-record top height (stand-ins for the held part's visible registered
region at this pose and its registered top); the robot's own wrist pixels are the fingertip band of the start
survey's free-air close (robot-only, the ``V2Calibration.fingertip_band`` BR passes); (3) scoring: the free
object's pose (tilt from vertical, in-hand rotation against the hand since the load test started) and the lateral
error at the close.

RS-7 bound (review-1 decision §5.2, extended by review-2 decision §6.2): O-1 (final ``CoMotionObserver`` report
after VerifyHold ``grip_unstable``) on >= 4/5 holds of each tilted class (jaw 8, 10 and 13.4 deg with 4 mm lateral)
and 0/5 ``clean``. The summary scores EVERY forced sign set (``rs7_pass_by_set``: >= 4/5 of its holds) and every
class (``rs7_pass_by_class``: all four of its sign sets pass; the class-level share is reported as
``class_share_pass``), and passes ``rs7_pass`` only when the clean set and every set of every class do (a directory
without a class, e.g. a review-1 run, lists it in ``classes_missing`` and does not pass). A HOLD is an attempt whose
VerifyHold passed (``hold_pass``) with an O-1 report on its load test: the population O-1 guards (review-1 §1: the
tilted holds the pipeline missed passed VerifyHold every time). Only holds count toward a set's ``n`` (>= 5 needed,
for the clean set too, so "0/5 clean" needs five real clean holds); an attempt that closed empty or failed VerifyHold
is reported in ``not_held`` (by result, with its O-1 flags) and never counts as a flagged or an unflagged hold
(review-2 BO-R2-1). The first review-1 report
scored p10p4 alone (the sign chosen after the pilot; at that baseline m10m4 flagged 3/5). The O-1 threshold
(O3_ROTATION_MAX_DEG, 5 deg) was fitted on the +-10 deg holds of episodes 0-4; episodes 5-9 are the held-out check,
and the 8 and 13.4 deg classes were not seen by the fit. Agreement is scored against three truths:
the in-hand rotation during the lift > 5 deg, the tilt after the VerifyHold lift > 10 deg, and the tilt after the
diagnostic lift > 10 deg. ``--settle-steps N`` adds N closed hold steps after VerifyHold whose transitions also
update the observer (``o1_settle``: what a BR that feeds its post-lift settle frames would read; the RS-7 verdict
stays the one after VerifyHold).

usage (project venv; one _LiberoWorker per process; the default variants are ``clean`` and every RS-7 class set):
  env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python scripts/agent_speedup_v2_tilt_check.py \\
      --episode 0 --out runs/agent_speedup_v2_bo_r2_rs7_001 [--variants clean,p8p4,p13.4m4] \\
      [--frames] [--settle-steps 10]
  .venv/bin/python scripts/agent_speedup_v2_tilt_check.py --summarize runs/agent_speedup_v2_bo_r2_rs7_001

The scorer (``summarize``) needs no simulator: BG's survey probe (driver, harness, start survey) is loaded on first
use by the episode driver only.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gpt_field import agent_speedup_v2_observers as observers  # noqa: E402
from gpt_field import agent_speedup_v2_planner as planner  # noqa: E402
from gpt_field import agent_speedup_v2_primitives as primitives  # noqa: E402
from gpt_field.agent_speedup_v2_types import (HeightField, Meas, PartRecord, RobotModel, YAW_GRID_RAD,  # noqa: E402
                                              hf_code, jsonable)
from gpt_field.contact_skills import (AnchoredMove, GripperSettlePrimitive, HoldPrimitive, Orient,  # noqa: E402
                                      VerifyHold, classify_close, quaternion_angle_rad)
from gpt_field.speed_first_gripper import GripperSettleConfig  # noqa: E402
from gpt_field.wrist_servo import NCCTrack  # noqa: E402

TILT_CHECK_VERSION = "agent-speedup-v2-tilt-check-4"
TASK = "libero_10_t2"
GRIP_PID, OBJ_PID = "C", "D"
# BR (Wave 2) constants restated (contract §C10.5; the v1 sources are cited there)
LIFT_TEST_M = 0.025                          # VerifyHold load-test lift (v1 agent_speedup_v1.py:52)
GRASP_SETTLE = GripperSettleConfig(max_pose_correction_m=0.03)    # v1 grasp close (:110)
ORIENT_TOLERANCE_RAD = math.radians(3.0)     # v1 Orient default
POST_ORIENT_HOLD_STEPS = 4                   # v1 (:49)
DIAG_LIFT_M = 0.06                           # diagnostic lift after the load test (truth tilt only, not BR)
DIAG_HOLD_STEPS = 10
TRUTH_HF_HALF_M = 0.25
TRUTH_HF_STEP_M = 0.005
TILTED_TRUTH_DEG = 10.0                      # wave-2 smoke grasp_loop_t2: truth tilt <= 10 deg after the lift
IN_HAND_TRUTH_DEG = 5.0                      # truth in-hand rotation that O-1 must flag (O3_ROTATION_MAX_DEG)
RS7_LATERAL_MM = 4.0                         # RS-7: 4 mm of lateral error along the jaw (review-1 §1.3)
RS7_JAW_ERRORS_DEG = {"8": 8.0, "10": 10.0, "13.4": 13.4}   # review-2 §6.2: 13.4 = the largest accepted wrong axis
CLEAN_SET = "clean"


def _sign_sets(label, deg, mm=RS7_LATERAL_MM):
    """The four sign combinations (jaw +-deg, +-mm along it) of one RS-7 class: {name: (dyaw_deg, dj_mm)}."""
    return {f"{a}{label}{b}{mm:g}": (sa * deg, sb * mm)
            for (a, sa), (b, sb) in ((("p", 1.0), ("p", 1.0)), (("p", 1.0), ("m", -1.0)), (("m", -1.0), ("m", -1.0)),
                                     (("m", -1.0), ("p", 1.0)))}


_CLASS_SETS = {label: _sign_sets(label, deg) for label, deg in RS7_JAW_ERRORS_DEG.items()}
RS7_CLASSES = {label: tuple(sets) for label, sets in _CLASS_SETS.items()}
RS7_SETS = tuple(name for sets in RS7_CLASSES.values() for name in sets)
# diagnosis only (never gating): the "+4 mm" sets realize -0.8..+3.6 mm (an episode-dependent offset of the reference
# plan, -0.4..-3.7 mm on the clean holds); +6 mm commanded realizes +1.4..+5.5 mm (28/60 at >= +4 mm), so the "+4 mm"
# side of every class is also seen at a realized ~4 mm
REALIZED_P4_SETS = {f"{a}{label}p6": (s * deg, 6.0) for label, deg in RS7_JAW_ERRORS_DEG.items()
                    for a, s in (("p", 1.0), ("m", -1.0))}
HELD_RESULT = "hold_pass"                    # a HOLD: VerifyHold passed (with an O-1 report on its load test)
REALIZED_LATERAL_MM = RS7_LATERAL_MM         # realized-lateral bins (diagnosis): |err_jaw_mm| >= 4 at the close
VARIANTS = {CLEAN_SET: (0.0, 0.0), **{n: v for sets in _CLASS_SETS.values() for n, v in sets.items()},
            "p5p2": (5.0, 2.0), "m5m2": (-5.0, -2.0), "p0p4": (0.0, 4.0), "p0m4": (0.0, -4.0), "p7p0": (7.0, 0.0),
            "m7p0": (-7.0, 0.0), **REALIZED_P4_SETS}
DEFAULT_VARIANTS = (CLEAN_SET,) + RS7_SETS
ROBOT = RobotModel()


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SP = {}


def survey_probe():
    """BG's survey probe script (driver, harness, start survey), loaded on first use (the scorer runs without it)."""
    if "module" not in _SP:
        _SP["module"] = _load("agent_speedup_v2_survey_probe", ROOT / "scripts" / "agent_speedup_v2_survey_probe.py")
    return _SP["module"]


# ------------------------------------------------------------------------------------------------ driver
def make_driver(sim, obs):
    """The survey probe's Driver with per-transition callbacks (observer updates and truth recording)."""
    if "driver" not in _SP:
        class Driver(survey_probe().Driver):
            def __init__(self, sim, obs):
                super().__init__(sim, obs)
                self.callbacks = []

            def step(self, action, stage=None, recorders=()):
                before = self.view
                after = super().step(action, stage, recorders)
                for callback in list(self.callbacks):
                    callback(stage, before, action, after)
                return after
        _SP["driver"] = Driver
    return _SP["driver"](sim, obs)


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2])))))


def angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


# ------------------------------------------------------------------------------------------------ truth (ORACLE)
def object_pose(sim, answers, pid):
    body = answers.parts[pid]["body"]
    pose = sim.oracle({"kind": "bodies", "names": [body]})["poses"][body]
    return np.asarray(pose["xyz"], dtype=float), quat_matrix(pose["quat_xyzw"])


def truth_hf(sim, centre_xy):
    """5 mm HeightField of the collision heights around centre (robot skipped) + per-cell owning geom."""
    hf = HeightField.empty(tuple(centre_xy), half_extent_m=TRUTH_HF_HALF_M)
    xs = np.arange(centre_xy[0] - TRUTH_HF_HALF_M + TRUTH_HF_STEP_M / 2, centre_xy[0] + TRUTH_HF_HALF_M, TRUTH_HF_STEP_M)
    ys = np.arange(centre_xy[1] - TRUTH_HF_HALF_M + TRUTH_HF_STEP_M / 2, centre_xy[1] + TRUTH_HF_HALF_M, TRUTH_HF_STEP_M)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    owners = {}
    for i in range(0, len(pts), 3000):
        reply = sim.oracle({"kind": "heights", "xy": pts[i:i + 3000]})
        for (x, y), z, hit in zip(pts[i:i + 3000], reply["z"], reply["hit"]):
            ij = hf.index((x, y))
            if ij is None or z is None:
                continue
            hf.z[ij], hf.sigma[ij], hf.known[ij], hf.source[ij], hf.stamp[ij] = z, 0.002, True, hf_code("kappa"), 0
            owners[ij] = hit["geom"] if hit else None
    return hf, owners


def widths_from_cells(xy, point_xy, sigma=0.001):
    rows = []
    for k in range(int(round(math.pi / YAW_GRID_RAD))):
        psi = k * YAW_GRID_RAD
        jaw = np.array([math.cos(psi), math.sin(psi)])
        normal = np.array([-jaw[1], jaw[0]])
        rel = xy - point_xy
        near = np.abs(rel @ normal) <= 0.0075
        span = (rel[near] @ jaw) if near.any() else np.array([0.0])
        rows.append((psi, Meas(float(span.max() - span.min() + TRUTH_HF_STEP_M), sigma, "kappa")))
    return tuple(rows)


def truth_records(sim, answers):
    """Truth-built PartRecords of C (its region geoms) and D (the rest of its owner) and the truth HeightField."""
    spec_c = answers.parts[GRIP_PID]
    point_c = np.asarray(answers.truth_points()[GRIP_PID], dtype=float)
    hf, owners = truth_hf(sim, point_c[:2] + np.array([0.0, 0.05]))
    region = set(spec_c["region"])
    owner_prefix = spec_c["owner"] + "_"
    cells_c = np.array([ij for ij, g in owners.items() if g in region], dtype=int)
    cells_d = np.array([ij for ij, g in owners.items() if g and g.startswith(owner_prefix) and g not in region],
                       dtype=int)
    records = {}
    for pid, cells, roles, point in ((GRIP_PID, cells_c, {"grip"}, point_c[:2]), (OBJ_PID, cells_d, {"carried"}, None)):
        xy = hf.origin_xy + cells[:, ::-1] * hf.cell_m
        z = hf.z[cells[:, 0], cells[:, 1]]
        top = float(np.percentile(z, 95))
        point = xy.mean(axis=0) if point is None else point
        c = xy - xy.mean(axis=0)
        _, v = np.linalg.eigh(c.T @ c)
        axis = v[:, 1] / np.linalg.norm(v[:, 1])
        records[pid] = PartRecord(pid=pid, text=f"part {pid}", move_prior="free", roles=frozenset(roles),
                                  xyz=np.array([point[0], point[1], top]), cov=np.eye(3) * 0.002 ** 2,
                                  top_z=Meas(top, 0.002, "kappa"), support_z=Meas(0.900, 0.001, "kappa"),
                                  region_cells=cells, width_by_yaw=widths_from_cells(xy, np.asarray(point)),
                                  elongation=(axis, 0.9) if pid == GRIP_PID else None, status="registered",
                                  registration_step=0, sigma_h_m=0.002)
    return records, hf


def reference_plan(view, records, hf, gripper):
    inputs = planner.PlannerInputs.build(ROBOT, view.state, g_open_mm=gripper.open_gap_mm,
                                         start_quaternion=view.quaternion_xyzw)
    plan = planner.plan_contact(records[GRIP_PID], hf, inputs, (), purpose="carry", next_held=records[OBJ_PID],
                                depth_rules=("height_stop",))
    if not isinstance(plan, planner.ContactPlan):
        raise RuntimeError(f"reference height_stop plan refused: {planner.trace_payload(plan)}")
    return plan


def forced_plan(plan, dyaw_deg, dj_mm):
    """The reference plan with its jaw turned by dyaw about world z and its grip point moved dj along the jaw."""
    jaw = np.asarray(plan.jaw_world, dtype=float)
    if dyaw_deg:
        c, s = math.cos(math.radians(dyaw_deg)), math.sin(math.radians(dyaw_deg))
        jaw = np.array([c * jaw[0] - s * jaw[1], s * jaw[0] + c * jaw[1], 0.0])
        q = ROBOT.quaternion_for(jaw, np.asarray(plan.approach_world, dtype=float))
        plan = dataclasses.replace(plan, jaw_world=tuple(jaw.tolist()), quaternion_xyzw=tuple(np.asarray(q).tolist()))
    unit = jaw[:2] / np.linalg.norm(jaw[:2])
    grip = np.asarray(plan.grip_point_xyz, dtype=float) + np.r_[dj_mm * 1e-3 * unit, 0.0]
    return dataclasses.replace(plan, grip_point_xyz=tuple(grip.tolist())), unit


# ------------------------------------------------------------------------------------------------ motions
def goto(d, target, quat, *, gripper="open"):
    d.goto(target, quat, gripper=gripper)


def orient(d, quat):
    angle = quaternion_angle_rad(d.view.quaternion_xyzw, quat)
    if angle <= ORIENT_TOLERANCE_RAD:
        return {"skipped": True, "angle_deg": math.degrees(angle)}
    out = d.run(Orient(d.view, target_xyzw=quat), "orient")
    d.run(HoldPrimitive(d.view, steps=POST_ORIENT_HOLD_STEPS, gripper="open"), "orient_settle")
    return {"skipped": False, "reason": out["reason"], "angle_deg": math.degrees(angle)}


def descend(d, plan, quat, g3_mm):
    d.run(primitives.b4_settle(d.view, gripper="open", orientation_xyzw=quat), "first_settle")
    travel = max(float(d.view.position[2]) - plan.target_grip_z, 0.001)
    out = d.run(AnchoredMove(d.view, axis=(0, 0, -1), travel_m=travel, gripper="open", stop_on_contact=True,
                             touch_threshold_mm=g3_mm, orientation_xyzw=quat, name="guarded_to_target"), "descent")
    return {"reason": out["reason"], "z": float(d.view.position[2]), "target_z": plan.target_grip_z,
            "contact_xyz": out["evidence"].get("contact_xyz")}


# ------------------------------------------------------------------------------------------------ observer inputs
def external_template(answers, view):
    """NCCTrack at the external projection of D's ORACLE point (stand-in for the projected registered footprint
    centre at its top, HeldFrameEstimator.prior_at_close)."""
    point = answers.truth_points()[OBJ_PID]
    uv = answers.project("external", [point])[0][0]
    gray = cv2.cvtColor(np.ascontiguousarray(view.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
    return NCCTrack(gray, (float(uv[0]), float(uv[1]))), [float(uv[0]), float(uv[1])]


# ------------------------------------------------------------------------------------------------ one attempt
def attempt(d, sim, answers, cal, ref, records, dyaw_deg, dj_mm, out_dir=None, settle_steps=0):
    plan, jaw = forced_plan(ref, dyaw_deg, dj_mm)
    up = np.array([-jaw[1], jaw[0]])
    quat = np.asarray(plan.quaternion_xyzw, dtype=float)
    row = {"dyaw_deg": dyaw_deg, "dj_mm": dj_mm, "plan": {"yaw_deg": math.degrees(plan.yaw_rad),
                                                          "target_grip_z": plan.target_grip_z,
                                                          "hover_grip_z": plan.hover_grip_z}}
    hover = np.array([plan.grip_point_xyz[0], plan.grip_point_xyz[1], plan.hover_grip_z])
    goto(d, hover, d.view.quaternion_xyzw.copy())
    row["orient"] = orient(d, quat)
    goto(d, hover, quat)
    row["descent"] = descend(d, plan, quat, cal["g3"].value)
    rel = d.view.position[:2] - np.asarray(ref.grip_point_xyz[:2], dtype=float)
    row["pre_close"] = {"err_jaw_mm": 1000 * float(rel @ jaw), "err_up_mm": 1000 * float(rel @ up)}
    settle = GripperSettlePrimitive("close", d.view, config=GRASP_SETTLE, name="grasp")
    d.run(settle, "close")
    klass = classify_close(settle, cal["record"].gripper, margin_mm=cal["g5"].value)
    row["close"] = {"object_present": klass["object_present"], "final_gap_mm": klass["final_gap_mm"]}
    if not klass["object_present"]:
        row["result"] = "empty_close"
        return row
    # ---- load test (contract §C10.2 item 10): CoMotion(D) started on this view, updated on every lift transition
    v0 = d.view
    mask = answers.visible_mask(OBJ_PID, "wrist")
    top_z = records[OBJ_PID].top_z.value
    track, ext_uv = external_template(answers, v0)
    obs = observers.CoMotionObserver(wrist_uv=None, wrist_view=v0, ext_track=track, external=cal["external"],
                                     robot=ROBOT, wrist_patch=None if mask is None else {"mask": mask, "z_m": top_z},
                                     wrist_hand_mask=cal["band"], rotation_verdict=True)
    obs.start(v0)
    row["observer_inputs"] = {"wrist_patches": obs.wrist_patches_used, "wrist_mask_px": None if mask is None else
                              int(mask.sum()), "z_m": top_z, "external_uv": ext_uv,
                              "external_J": None if cal["external"] is None else np.asarray(cal["external"].J).tolist()}
    p0, R0 = object_pose(sim, answers, OBJ_PID)
    Rh0 = np.asarray(v0.R, dtype=float)
    rel0 = Rh0.T @ R0
    reports, truth = [], []
    frames = [] if out_dir is None else [(v0.wrist_rgb_upright, v0.external_rgb_upright)]
    states = [[float(v) for v in v0.state]]

    def on_lift(stage, before, action, after):
        if stage not in ("lift", "lift_settle"):
            return
        report = obs.update(before, action, after, {"step": d.steps})
        reports.append(report)
        _p, R = object_pose(sim, answers, OBJ_PID)
        truth.append({"tilt_deg": tilt_deg(R), "in_hand_deg": angle_deg((np.asarray(after.R).T @ R) @ rel0.T)})
        states.append([float(v) for v in after.state])
        if out_dir is not None:
            frames.append((after.wrist_rgb_upright, after.external_rgb_upright))
    d.callbacks.append(on_lift)
    try:
        floor = cal["g6"].value
        out = d.run(VerifyHold(d.view, axis=(0, 0, 1), travel_m=LIFT_TEST_M, calibration=cal["record"].gripper,
                               hold_margin_mm=floor - cal["record"].gripper.empty_floor_mm), "lift")
    finally:
        d.callbacks.remove(on_lift)
    row["verify_hold"] = {"reason": out["reason"], "actions": out["actions"]}
    n_lift = len(reports)
    if settle_steps and out["reason"] == "hold_pass" and reports:
        d.callbacks.append(on_lift)
        try:
            d.run(HoldPrimitive(d.view, steps=settle_steps, gripper="close"), "lift_settle")
        finally:
            d.callbacks.remove(on_lift)
        final = reports[-1]
        row["o1_settle"] = {"steps": settle_steps, "verdict": final.verdict, "reason": final.reason,
                            "cause": final.evidence.get("cause"), "flag": final.reason == "grip_unstable",
                            "estimate_deg": final.evidence["rotation"].get("estimate_deg"),
                            "series_deg": [None if r.evidence["rotation"]["estimate_deg"] is None
                                           else round(r.evidence["rotation"]["estimate_deg"], 2)
                                           for r in reports[n_lift:]],
                            "truth_in_hand_deg": truth[-1]["in_hand_deg"], "truth_tilt_deg": truth[-1]["tilt_deg"]}
        row["last_settle_report"] = final.to_dict()
        row["settle_states8"] = states[n_lift + 1:]
    reports, truth, states = reports[:n_lift], truth[:n_lift], states[:n_lift + 1]
    settle_frames = [] if out_dir is None else frames[n_lift + 1:]
    if out_dir is not None:
        frames = frames[:n_lift + 1]
    last = reports[-1] if reports else None
    rot = None if last is None else last.evidence["rotation"]
    row["o1"] = {"verdict": None if last is None else last.verdict, "reason": None if last is None else last.reason,
                 "cause": None if last is None else last.evidence.get("cause"),
                 "flag": bool(last is not None and last.reason == "grip_unstable"),
                 "estimate_deg": None if rot is None else rot.get("estimate_deg"),
                 "source": None if rot is None else rot.get("source"),
                 "threshold_deg": None if rot is None else rot.get("threshold_deg"),
                 "latched_at": None if rot is None else rot.get("latched_at"),
                 "series_deg": [None if r.evidence["rotation"]["estimate_deg"] is None
                                else round(r.evidence["rotation"]["estimate_deg"], 2) for r in reports],
                 "external_series_deg": [None if (r.evidence["rotation"].get("external") or {}).get("estimate_deg")
                                         is None else round(r.evidence["rotation"]["external"]["estimate_deg"], 2)
                                         for r in reports],
                 "verdicts": [(r.verdict, r.reason) for r in reports]}
    row["last_report"] = None if last is None else last.to_dict()
    row["truth_lift"] = {"in_hand_deg": truth[-1]["in_hand_deg"] if truth else None,
                         "tilt_deg": truth[-1]["tilt_deg"] if truth else None,
                         "tilt_start_deg": tilt_deg(R0),
                         "in_hand_series_deg": [round(t["in_hand_deg"], 2) for t in truth]}
    if out["reason"] == "hold_pass":
        d.run(AnchoredMove(d.view, axis=(0, 0, 1), travel_m=DIAG_LIFT_M, gripper="close"), "diag_lift")
        d.run(HoldPrimitive(d.view, steps=DIAG_HOLD_STEPS, gripper="close"), "diag_hold")
        _p, R = object_pose(sim, answers, OBJ_PID)
        row["truth_after_diag"] = {"tilt_deg": tilt_deg(R),
                                   "in_hand_deg": angle_deg((np.asarray(d.view.R).T @ R) @ rel0.T),
                                   "gap_mm": d.view.gap_mm}
    row["result"] = out["reason"]
    row["lift_states8"] = states
    if out_dir is not None and frames:
        out_dir.mkdir(parents=True, exist_ok=True)
        if mask is not None:
            np.save(out_dir / "wrist_mask_obj.npy", mask)
        for k, (wr, ex) in enumerate(frames):
            cv2.imwrite(str(out_dir / f"w{k:03d}.png"), cv2.cvtColor(np.ascontiguousarray(wr), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"e{k:03d}.png"), cv2.cvtColor(np.ascontiguousarray(ex), cv2.COLOR_RGB2BGR))
        for k, (wr, ex) in enumerate(settle_frames):        # the settle transitions' after-views (s000 = first)
            cv2.imwrite(str(out_dir / f"sw{k:03d}.png"), cv2.cvtColor(np.ascontiguousarray(wr), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"se{k:03d}.png"), cv2.cvtColor(np.ascontiguousarray(ex), cv2.COLOR_RGB2BGR))
    return row


# ------------------------------------------------------------------------------------------------ episode
def calibrate(d):
    cal = primitives.CalibratePlus(d.view, recorder=None, robot=ROBOT)
    d.run(cal, "calibrate_plus")
    record = cal.record()
    q = primitives.gripper_quantities(record)
    survey = survey_probe().calibrate(d, ROBOT)
    return {"record": record, "q": q, "g3": q["thresholds"][2],
            "g5": primitives.g5_present_threshold(record.gripper, q["sigma_close_mm"]),
            "g6": primitives.g6_hold_floor(record.gripper, q["sigma_close_mm"]), "external": survey["external"],
            "band": survey["band"],
            "summary": {"fingertip_band_px": int(np.count_nonzero(survey["band"])),
                        "g_open_mm": record.gripper.open_gap_mm, "empty_floor_mm": record.gripper.empty_floor_mm,
                        "external_J": None if survey["external"] is None else np.asarray(survey["external"].J).tolist(),
                        "f_used": survey["camera"].f_used}}


def run_episode(args):
    harness = survey_probe().harness()
    oracle = survey_probe().oracle_module()
    sim, spec, _config, _monitor = harness.start_sim(TASK, oracle=True)
    report = {"tool": TILT_CHECK_VERSION, "observers": observers.OBSERVERS_VERSION, "episode": args.episode,
              "attempts": {}, "deviations": [
                  "reference plan = BP plan_contact (height_stop) on truth-built PartRecords of C and D (ORACLE plan "
                  "cells; the grasp-tolerance lens's reference)",
                  "observer inputs: external template at the external projection of D's ORACLE point; wrist patches "
                  "selected by observers.select_wrist_patches on D's ORACLE visible wrist mask, back-projected at D's "
                  "truth-record top (stand-ins for the scene's registration / HeldFrame projection, BR Wave 2); hand "
                  "mask = the start survey's fingertip band (robot-only)"], "settle_steps": args.settle_steps}
    path = args.out / f"rs7_ep{args.episode}.json"
    t0 = time.time()
    try:
        obs, episode = harness.reset_episode(sim, spec, args.episode)
        report["reset"] = episode
        d = make_driver(sim, obs)
        cal = calibrate(d)
        report["calibration"] = cal["summary"]
        np.save(args.out / f"fingertip_band_ep{args.episode}.npy", cal["band"])
        for name in args.variants.split(","):
            dyaw, dj = VARIANTS[name]
            obs, _ = harness.reset_episode(sim, spec, args.episode)
            d = make_driver(sim, obs)
            answers = oracle.OracleAnswers(sim, spec.key)
            ta = time.time()
            try:
                records, hf = truth_records(sim, answers)
                ref = reference_plan(d.view, records, hf, cal["record"].gripper)
                row = attempt(d, sim, answers, cal, ref, records, dyaw, dj,
                              out_dir=(args.out / f"frames_ep{args.episode}_{name}") if args.frames else None,
                              settle_steps=args.settle_steps)
            except Exception:  # noqa: BLE001 - a diagnosis tool keeps going
                row = {"error": traceback.format_exc()}
            row["wall_s"] = round(time.time() - ta, 1)
            row["steps"] = d.steps
            report["attempts"][name] = row
            o1 = row.get("o1") or {}
            print(name, row.get("result"), "O-1", o1.get("reason"), o1.get("estimate_deg"), o1.get("source"),
                  "truth in-hand", (row.get("truth_lift") or {}).get("in_hand_deg"),
                  "tilt after diag", (row.get("truth_after_diag") or {}).get("tilt_deg"),
                  row.get("error", "")[-300:], flush=True)
            path.write_text(json.dumps(jsonable(report), indent=1, allow_nan=True))
    finally:
        report["wall_s"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(jsonable(report), indent=1, allow_nan=True))
        sim.close()


def _held(row):
    """A HOLD the RS-7 bound counts: VerifyHold passed and O-1 reported on the load test (review-2 BO-R2-1: an empty
    close or a dropped hold is no evidence that O-1 leaves a clean hold alone or sees a tilted one)."""
    return "error" not in row and row.get("result") == HELD_RESULT and row.get("o1_verdict") is not None


def _flag_counts(items, key):
    """Per variant: ``n`` = holds (``_held``), the O-1 flags and the truth agreement over them; errored attempts in
    ``errors``; attempts that never reached a held load test (empty close, VerifyHold failure) in ``not_held``
    (``not_held_by_result``, with their O-1 flags in ``not_held_o1_flags``), never counted as holds."""
    ok = [r for r in items if "error" not in r]
    held = [r for r in ok if _held(r)]
    dropped = [r for r in ok if not _held(r)]
    by_result = {}
    for r in dropped:
        by_result[str(r.get("result"))] = by_result.get(str(r.get("result")), 0) + 1
    return {"n": len(held), "attempts": len(items), "errors": len(items) - len(ok), "not_held": len(dropped),
            "not_held_by_result": by_result, "not_held_o1_flags": sum(bool(r[key]) for r in dropped),
            "o1_flags": sum(bool(r[key]) for r in held),
            "in_hand_truth": sum((r["truth_in_hand_lift_deg"] or 0.0) > IN_HAND_TRUTH_DEG for r in held),
            "tilted_lift_truth": sum((r["truth_tilt_lift_deg"] or 0.0) > TILTED_TRUTH_DEG for r in held),
            "tilted_truth": sum((r["truth_tilt_after_diag_deg"] or 0.0) > TILTED_TRUTH_DEG for r in held),
            "agree_in_hand_5deg": sum(bool(r[key]) == ((r["truth_in_hand_lift_deg"] or 0.0) > IN_HAND_TRUTH_DEG)
                                      for r in held),
            "agree_lift_tilt_10deg": sum(bool(r[key]) == ((r["truth_tilt_lift_deg"] or 0.0) > TILTED_TRUTH_DEG)
                                         for r in held),
            "agree_with_truth_10deg": sum(bool(r[key]) == ((r["truth_tilt_after_diag_deg"] or 0.0) > TILTED_TRUTH_DEG)
                                          for r in held)}


def _set_pass(n, flags):
    """A sign set (or a realized-lateral bin): >= 5 holds, >= 4/5 of them flagged."""
    return bool(n >= 5 and 5 * flags >= 4 * n)


def _realized_bins(rows):
    """Diagnosis, never gating (review-2 BO-R2-2: the commanded "+4 mm" sets realize -0.8..+3.6 mm at the close, so
    the decision's "4 mm lateral" is realized on the -4 mm side only): per class, the held forced holds of every
    variant whose jaw error is the class's (the RS-7 sets and the +6 mm ``REALIZED_P4_SETS`` alike) binned by the jaw
    sign and the REALIZED lateral error at the close (``err_jaw_mm`` >= +4 -> ``p``, <= -4 -> ``m``; smaller in
    ``under``), each bin named and scored as a sign set (``_set_pass``); a class passes when all four bins do. Rows
    are the summary's; the same (episode, variant) seen twice (pooled runs) counts once (``duplicates_skipped``)."""
    seen, dup, out = set(), 0, {}
    pooled = []
    for r in rows:
        if not _held(r) or r["variant"] not in VARIANTS or r["variant"] == CLEAN_SET or r.get("err_jaw_mm") is None:
            continue
        key = (r["episode"], r["variant"])
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        pooled.append(r)
    for label, deg in RS7_JAW_ERRORS_DEG.items():
        bins = {f"{a}{label}{b}{REALIZED_LATERAL_MM:g}": [] for a in ("p", "m") for b in ("p", "m")}
        under, variants = [], set()
        for r in pooled:
            dyaw = VARIANTS[r["variant"]][0]
            if not math.isclose(abs(dyaw), deg):
                continue
            variants.add(r["variant"])
            e = float(r["err_jaw_mm"])
            side = "p" if e >= REALIZED_LATERAL_MM else "m" if e <= -REALIZED_LATERAL_MM else None
            if side is None:
                under.append(r)
            else:
                bins[f"{'p' if dyaw > 0 else 'm'}{label}{side}{REALIZED_LATERAL_MM:g}"].append(r)

        def score(items):
            flags = sum(bool(r["o1_flag"]) for r in items)
            errs = [float(r["err_jaw_mm"]) for r in items]
            return {"n": len(items), "flags": flags, "pass": _set_pass(len(items), flags),
                    "err_jaw_mm": [min(errs), max(errs)] if errs else None,
                    "variants": sorted({r["variant"] for r in items})}
        scored = {name: score(items) for name, items in bins.items()}
        out[label] = {"jaw_error_deg": deg, "realized_lateral_mm": REALIZED_LATERAL_MM, "variants": sorted(variants),
                      "bins": scored, "under": score(under), "pass": all(b["pass"] for b in scored.values())}
    return {"gating": False, "classes": out, "duplicates_skipped": dup}


def realized_lateral(directories):
    """``_realized_bins`` over the rows of several run directories pooled (e.g. the gating runs and the +6 mm
    diagnosis run of the same episodes); writes nothing."""
    rows = []
    for directory in directories:
        rows.extend(_rows(directory))
    report = _realized_bins(rows)
    report["directories"] = [str(d) for d in directories]
    return report


def _bound(verdict):
    """RS-7 per forced set: >= 4/5 of its holds flagged (8/10 over ten episodes), with no clean hold flagged (at
    least 5 holds each, ``_held``: attempts not held never count; no errors). Per class (review-2 §6.2): every one of
    its four sign sets passes (``class_share_pass``, >= 4/5 of the class's holds, is reported but never rescues a
    failing set); a class with no set in the run is None and listed in ``missing``. Returns (clean_ok, by_set,
    by_class, missing, pass)."""
    clean = verdict.get(CLEAN_SET, {})
    clean_ok = bool(clean.get("n", 0) >= 5 and not clean.get("errors") and clean.get("o1_flags", 1) == 0)
    by_set = {name: bool(clean_ok and not verdict[name]["errors"]
                         and _set_pass(verdict[name]["n"], verdict[name]["o1_flags"]))
              for name in RS7_SETS if name in verdict}
    by_class, missing = {}, []
    for label, sets in RS7_CLASSES.items():
        present = [name for name in sets if name in verdict]
        if not present:
            by_class[label] = None
            missing.append(label)
            continue
        n = sum(verdict[name]["n"] for name in present)
        flags = sum(verdict[name]["o1_flags"] for name in present)
        errors = sum(verdict[name]["errors"] for name in present)
        by_class[label] = {"jaw_error_deg": RS7_JAW_ERRORS_DEG[label], "lateral_mm": RS7_LATERAL_MM, "n": n,
                           "errors": errors, "flags": flags, "sets_missing": [s for s in sets if s not in verdict],
                           "not_held": sum(verdict[name]["not_held"] for name in present),
                           "class_share_pass": bool(clean_ok and not errors and n > 0 and 5 * flags >= 4 * n),
                           "in_hand_truth": sum(verdict[name]["in_hand_truth"] for name in present),
                           "agree_in_hand_5deg": sum(verdict[name]["agree_in_hand_5deg"] for name in present),
                           "pass": bool(len(present) == len(sets) and all(by_set[name] for name in sets))}
    passed = bool(clean_ok and not missing and all(c["pass"] for c in by_class.values()))
    return clean_ok, by_set, by_class, missing, passed


def _rows(directory):
    """One summary row per attempt of a run directory (``rs7_ep*.json``)."""
    rows = []
    for path in sorted(Path(directory).glob("rs7_ep*.json")):
        report = json.loads(path.read_text())
        for name, row in report["attempts"].items():
            if "error" in row:
                rows.append({"episode": report["episode"], "variant": name, "error": row["error"][-400:]})
                continue
            o1 = row.get("o1") or {}
            settle = row.get("o1_settle")
            rows.append({"episode": report["episode"], "variant": name, "result": row.get("result"),
                         "o1_verdict": o1.get("verdict"),
                         "o1_flag": o1.get("flag"), "o1_reason": o1.get("reason"), "o1_cause": o1.get("cause"),
                         "o1_estimate_deg": o1.get("estimate_deg"), "o1_source": o1.get("source"),
                         "o1_latched_step": (o1.get("latched_at") or {}).get("step"),
                         "o1_settle_flag": None if settle is None else settle.get("flag"),
                         "o1_settle_cause": None if settle is None else settle.get("cause"),
                         "o1_settle_estimate_deg": None if settle is None else settle.get("estimate_deg"),
                         "truth_in_hand_lift_deg": (row.get("truth_lift") or {}).get("in_hand_deg"),
                         "truth_tilt_lift_deg": (row.get("truth_lift") or {}).get("tilt_deg"),
                         "truth_tilt_after_diag_deg": (row.get("truth_after_diag") or {}).get("tilt_deg"),
                         "err_jaw_mm": (row.get("pre_close") or {}).get("err_jaw_mm"),
                         "max_external_deg": max([v for v in o1.get("external_series_deg") or [] if v is not None],
                                                 default=None)})
    return rows


def summarize(directory):
    rows = _rows(directory)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    verdict = {name: _flag_counts(items, "o1_flag") for name, items in by.items()}
    clean_ok, by_set, by_class, missing, passed = _bound(verdict)
    summary = {"tool": TILT_CHECK_VERSION,
               "bound": ("O-1 flags >= 4/5 of the holds (VerifyHold passed, O-1 reported; >= 5 each) of each forced "
                         "sign set of each tilted class (jaw 8, 10 and 13.4 deg, each with +-4 mm commanded along the "
                         "jaw) and 0/5 clean holds; attempts not held are in by_variant not_held, never counted"),
               "rs7_pass": passed, "rs7_pass_by_class": by_class, "classes_missing": missing,
               "rs7_pass_by_set": by_set, "clean_pass": clean_ok, "by_variant": verdict,
               "not_held": sum(v["not_held"] for v in verdict.values()), "realized_lateral": _realized_bins(rows),
               "episodes": sorted({r["episode"] for r in rows}), "rows": rows}
    settled = [r for r in rows if r.get("o1_settle_flag") is not None]
    if settled:
        settle_verdict = {name: _flag_counts([r for r in items if r.get("o1_settle_flag") is not None],
                                             "o1_settle_flag") for name, items in by.items()}
        settle_clean, settle_set, settle_class, settle_missing, settle_pass = _bound(settle_verdict)
        summary["settle"] = {"rs7_pass": settle_pass, "rs7_pass_by_class": settle_class,
                             "classes_missing": settle_missing, "rs7_pass_by_set": settle_set,
                             "clean_pass": settle_clean, "by_variant": settle_verdict}
    (Path(directory) / "rs7_summary.json").write_text(json.dumps(summary, indent=1, allow_nan=True))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episode", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help=f"comma list of {sorted(VARIANTS)}")
    parser.add_argument("--frames", action="store_true", help="save the load-test frames (PNG) per attempt")
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="closed hold steps after VerifyHold that also update the observer (o1_settle)")
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--realized", type=Path, nargs="+",
                        help="print the realized-lateral bins (diagnosis) over these run directories pooled")
    args = parser.parse_args(argv)
    if args.realized is not None:
        print(json.dumps(realized_lateral(args.realized), indent=1))
        return
    if args.summarize is not None:
        summary = summarize(args.summarize)
        print(json.dumps({k: summary[k] for k in ("rs7_pass", "rs7_pass_by_class", "classes_missing", "rs7_pass_by_set",
                                                  "clean_pass", "by_variant", "not_held", "settle")
                          if k in summary}, indent=1))
        return
    if args.episode is None or args.out is None:
        parser.error("--episode and --out are required (or --summarize DIR)")
    args.out.mkdir(parents=True, exist_ok=True)
    run_episode(args)


if __name__ == "__main__":
    main()
