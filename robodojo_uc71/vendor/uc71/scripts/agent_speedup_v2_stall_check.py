"""RS-3 (design review 1 decision §5.2, owner BM): the RC-8 hover-probe leg and its contact stops on real LIBERO task 2.

Diagnosis script: policy-free, zero model calls, one simulator worker per process. The ORACLE is used only for
placement (the scripted hover point standing in for pointing or a plan cell, and the choice of a footprint-clear
table site) and for scoring (truth clearance under the fingers, reference collapse steps). No control module imports
this file.

``run --episode N --out RUN_DIR`` does, per episode:

1. BM's ``CalibratePlus`` (decision §2.5 M-2: the vertical probe as 2 cm halves) with every transition kept, then
   replayed into BG's ``SurveyRecorder`` (external=True, fingertip-band exclude mask, started at probe_1 as BG's
   calibration does) to check the split: external samples, locatable moves, the along-v_hat motion of every move,
   ``HandLocator.fit``, a 40 px fault on each located move, f_hat.
2. 20 grip-mode hovers: parts B C D E A x the four +-1 cm offsets (target = ORACLE anchor + offset, fingertips the
   P11 floor 5 cm above it, i.e. ``hover_for`` on an empty HeightField). Goto 3 cm above the hover, settle,
   ``hover_arrival_descent`` (3 cm, G3 = the calibrated touch threshold), then ``ParallaxProbe`` (+-2 cm) as built.
   Every leg's ratio trace and stop reason are logged, with the truth clearance at any stop.
3. Contacts, each from a settled hover with an open hand straight down: three descents per site, (a1)
   ``hover_arrival_descent`` with the calibrated G3 (what BR will run), (a2) the same leg with the touch stop
   disabled (the stall rule alone), (b) a v1 ``AnchoredMove`` with the stall stop off (the reference collapse step).
   Sites:
   - ``plate``: B's region top (decision RS-3 "plate"), from 5 cm above it;
   - ``table``: a site whose open-finger footprints + 25 mm see only the lowest flat surface (the table), found from
     ORACLE heights among fixed offsets around A (decision RS-3 "table"; review finding BM-RV-1);
   - ``lever_top``: A + (0, -6 cm) from 5 cm above A's top, where a finger lands on A's top, not on the table below
     the grip (the site Wave-1 RS-3 run 001 reported as "support beside A"; reported only).
4. Near-start contacts (reported only; the RC-8 early-contact trade-off of the PROBE_FREE_RATIO_FLOOR derivation):
   plate and table with the fingertips 4 and 8 mm above the surface, (a1), (a2) and the v1-rule leg (G3 off).

``summarize RUN_DIR`` writes ``RUN_DIR/summary.json``. RS-3 passes when there are >= 100 hovers, 0 false
``contact_stall`` (a stall with more than FREE_CLEARANCE_MM of truth clearance under the fingers) and every plate and
table descent (a1 and a2) stops at the reference collapse step + <= 1.

usage (episodes 1-5 in parallel, one worker each):
    for ep in 1 2 3 4 5; do env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 \\
        /workspace/gpt_field/.venv/bin/python scripts/agent_speedup_v2_stall_check.py run --episode $ep \\
        --out runs/agent_speedup_v2_bm_r1_rs3_stall_002 & done; wait
    /workspace/gpt_field/.venv/bin/python scripts/agent_speedup_v2_stall_check.py summarize \\
        runs/agent_speedup_v2_bm_r1_rs3_stall_002
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field import agent_speedup_v2_geometry as G  # noqa: E402
from gpt_field import agent_speedup_v2_primitives as P  # noqa: E402
from gpt_field import agent_speedup_v2_robot as robot_facts  # noqa: E402
from gpt_field.agent_speedup_v2_types import RobotModel  # noqa: E402
from gpt_field.contact_skills import AnchoredMove, HoldingActionAdapter, HoldPrimitive, StepView  # noqa: E402

TASK_KEY = "libero_10_t2"
HOVER_LOCATE_M = 0.03            # decision §3 (BR): the arrival descent
HOVER_CLEARANCE_M = 0.05         # P11 floor: fingertips 5 cm above the part (hover_for on an empty HeightField)
HOVER_OFFSETS_M = ((0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01))
PARTS = ("B", "C", "D", "E", "A")
CONTACT_TRAVEL_M = 0.09
NEAR_START_M = (0.004, 0.008)
NEAR_TRAVEL_M = 0.03
NO_TOUCH_MM = 1000.0             # (a2): a touch threshold no finger reaches, so only the stall rule can stop
GOTO_TOL_M, NEAR_TOL_M, GOTO_MAX_STEPS, GOTO_GAIN, SETTLE_STEPS = 0.002, 0.0008, 250, 1.5, 14
# contact descents start within 0.5 mm of the same hover (a 2 mm start difference shifts the collapse by ~1 step)
CONTACT_TOL_M = 0.0005
FREE_CLEARANCE_MM = 10.0         # a stop with more truth clearance than this under the fingers is a false contact
TIGHT_MARGIN_M, WIDE_MARGIN_M, HEIGHT_STEP_M = 0.005, 0.025, 0.004
FLAT_TOL_M = 1e-4                # footprint-clear: the highest surface near the fingers equals the site height
TABLE_TOL_M = 0.001              # the table is the lowest surface among the candidate sites
TABLE_MAX_DROP_M = 0.20          # ... within 20 cm below A's anchor (a candidate off the table would see the floor)
# table-site candidates, offsets from A's anchor (ORACLE placement only); review finding BM-RV-1 found (-6, -12) cm
TABLE_CANDIDATES_M = ((0.0, -0.12), (0.06, -0.12), (-0.06, -0.12), (0.10, -0.10), (0.0, -0.16), (0.12, 0.0),
                      (-0.10, -0.10), (-0.06, -0.16))
LEVER_TOP_OFFSET_M = (0.0, -0.06)
FAULT_PX = 40.0
CONTACT_SITES_SCORED = ("plate", "table")
ROBOT = RobotModel()


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harness():
    return _load("agent_speedup_v2_relation_smoke", ROOT / "scripts" / "agent_speedup_v2_relation_smoke.py")


def quat_to_R(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Driver:
    def __init__(self, sim, obs):
        self.sim, self.view, self.steps = sim, StepView(obs), 0
        self.transitions = None          # list of (stage, before, after) while recording

    def step(self, action, stage=None):
        obs, _ = self.sim.step([float(v) for v in action])
        after = StepView(obs)
        if self.transitions is not None:
            self.transitions.append((stage, self.view, after))
        self.view = after
        self.steps += 1
        return after

    def run(self, primitive, stage=None, on_step=None):
        while not primitive.done:
            before = self.view
            current = getattr(primitive, "current", None)
            sub_stage = getattr(primitive, "stage", None)
            action = primitive.next_action(before)
            after = self.step(action, sub_stage or stage)
            primitive.observe_transition(before, action, after)
            if on_step is not None:
                on_step(primitive, current, sub_stage, before, after)
        return primitive.outcome()

    def goto(self, target, quat, tol=GOTO_TOL_M):
        adapter = HoldingActionAdapter()
        target = np.asarray(target, dtype=float)
        n = 0
        for _ in range(GOTO_MAX_STEPS):
            error = target - self.view.position
            if float(np.linalg.norm(error)) < tol:
                break
            delta = np.clip(error * GOTO_GAIN, -robot_facts.TRANSLATION_PER_UNIT_M, robot_facts.TRANSLATION_PER_UNIT_M)
            self.step(adapter.adapt(delta, quat, "open", self.view.state), "goto")
            n += 1
        self.run(HoldPrimitive(self.view, steps=SETTLE_STEPS, gripper="open", until_quiet=True,
                               orientation_xyzw=quat), "settle")
        return {"steps": n, "residual_mm": 1000.0 * float(np.linalg.norm(target - self.view.position))}


def heights(sim, xy_list):
    return sim.oracle({"kind": "heights", "xy": [[float(x), float(y)] for x, y in xy_list]})["z"]


def max_surface_near(sim, grip_xyz, R, g_open_mm, margin):
    """Highest collision surface (robot skipped) within ``margin`` of the open fingers' footprints at ``grip_xyz``."""
    jaw = np.asarray(R[:2, 1], dtype=float)
    jaw = jaw / np.linalg.norm(jaw)
    cols = np.vstack(ROBOT.finger_columns_xy(np.asarray(grip_xyz[:2], dtype=float), jaw, finger_q_mm=g_open_mm / 2.0))
    lo, hi = cols.min(axis=0) - margin, cols.max(axis=0) + margin
    grid = [(x, y) for y in np.arange(lo[1], hi[1] + 1e-9, HEIGHT_STEP_M)
            for x in np.arange(lo[0], hi[0] + 1e-9, HEIGHT_STEP_M)]
    zs = [v for v in heights(sim, grid) if v is not None]
    return max(zs) if zs else None


def clearance_mm(sim, grip_xyz, quat, g_open_mm):
    """(tight, wide) truth clearance between the fingertips and the highest surface within 5 / 25 mm of the open
    fingers' footprints."""
    tip = float(grip_xyz[2]) - ROBOT.t_f_m
    R = quat_to_R(quat)
    out = []
    for margin in (TIGHT_MARGIN_M, WIDE_MARGIN_M):
        z = max_surface_near(sim, grip_xyz, R, g_open_mm, margin)
        out.append(None if z is None else 1000.0 * (tip - z))
    return out


def collapse_step(ratios):
    """remedy 2 stall_floor.py: the 1-based step whose ratio first falls below 0.1 after one at or above it."""
    return next((i + 1 for i, r in enumerate(ratios) if i >= 1 and r < 0.1 and ratios[i - 1] >= 0.1), None)


def leg_row(leg, stage, sim, quat, g_open_mm):
    """A finished AnchoredMove leg: ratios, stop, its G12 record and, for any stop other than travel_reached, the
    truth clearance under the fingers at the stop."""
    row = {"stage": stage, "kind": type(leg).__name__, "reason": leg.reason, "steps": len(leg.ratios),
           "ratios": [round(float(r), 6) for r in leg.ratios], "free_ratio": float(leg.free_ratio),
           "free_ratio_measured": bool(leg.free_ratio_measured), "travel_m": float(leg.travel_measured_m),
           "requested_m": float(leg.travel_m), "contact": leg.contact, "stall_min_steps": leg.STALL_MIN_STEPS,
           "ratio_bounds": list(leg.FREE_RATIO_BOUNDS), "g12": P.g12_stall_threshold(leg).to_dict(),
           "start_xyz": [float(v) for v in leg.start_xyz], "end_xyz": [float(v) for v in leg.end_xyz]}
    if leg.reason != "travel_reached":
        tight, wide = clearance_mm(sim, leg.end_xyz, quat, g_open_mm)
        row["truth_clearance_mm"], row["truth_clearance_wide_mm"] = tight, wide
    return row


def calibrate_split_check(transitions, cal_record):
    """Replay Calibrate+'s probe-and-vertical transitions into BG's SurveyRecorder (as BG's calibration: started at
    the first probe move, fingertip-band exclude mask from the last open_hold frame and the end of close_2)."""
    open_rgb = [a for s, _, a in transitions if s == "open_hold"][-1].wrist_rgb_upright
    closed_rgb = [a for s, _, a in transitions if s == "close_2"][-1].wrist_rgb_upright
    band = G.fingertip_band_mask(open_rgb, closed_rgb)
    first = next(i for i, (s, _, _) in enumerate(transitions) if s == "probe_1")
    recorder = G.SurveyRecorder(transitions[first][1], ROBOT, camera=None, external=True, exclude_mask=band)
    t0 = time.monotonic()
    for stage, before, after in transitions[first:]:
        recorder.on_transition(stage, before, after)
    loom = recorder.result(None)
    samples = loom.external_samples
    f_hat = loom.loom.get("f_hat")
    out = {"external_samples": [s["stage"] for s in samples], "n_external": len(samples),
           "replay_s": round(time.monotonic() - t0, 2), "f_hat": None if f_hat is None else f_hat.to_dict(),
           "f_hat_rel_error_vs_R3": None if f_hat is None else f_hat.value / robot_facts.focal_length_px(512) - 1.0,
           "vertical_record": [{k: m.get(k) for k in ("stage", "reason", "world_displacement_m", "commanded_m")}
                               for m in cal_record.vertical]}
    model = G.ExternalModel.fit(samples)
    if model is None:
        out["external_model"] = None
        return out
    rows = []
    for s in samples:
        m = model.J @ (np.asarray(s["end_state8"][:3]) - np.asarray(s["start_state8"][:3]))
        rows.append({"stage": s["stage"], "motion_px": m.tolist(), "along_px": float(m @ model.v_hat),
                     "across_px": float(m @ model.n_hat), "check": G.HandLocator.motion_check(model, m),
                     "world_displacement_m": s["world_displacement_m"]})
    out["moves"] = rows
    out["locatable"] = [r["stage"] for r in rows if r["check"] is None]
    out["l1_min_along_px"] = float(G.L1_MIN_ALONG_PX)
    locator = G.HandLocator.fit(samples, model, ROBOT)
    out["locator"] = None if locator is None else {
        "located": [r["stage"] for r in locator.located], "sigma_consistency_px": locator.sigma_consistency_px,
        "sigma_loc_px": locator.sigma_loc_px, "residual_px": locator.residual_px}
    drops = {}
    if locator is not None:          # the purpose of the split (rulings item 5): one 40 px fault is dropped
        for bad in out["locatable"]:
            corrupted = [dict(s, mask_extreme_uv=(np.asarray(s["mask_extreme_uv"]) + [FAULT_PX, 0.0]).tolist())
                         if s["stage"] == bad and s.get("mask_extreme_uv") is not None else s for s in samples]
            refit = G.HandLocator.fit(corrupted, model, ROBOT)
            drops[bad] = None if refit is None else {"located": [r["stage"] for r in refit.located],
                                                     "dropped_bad": bad not in [r["stage"] for r in refit.located]}
    out["corrupt_40px"] = drops
    out["external_model"] = {"v_hat": model.v_hat.tolist(), "J": model.J.tolist()}
    return out


def find_table_site(sim, anchor, R, g_open_mm):
    """The first TABLE_CANDIDATES_M offset from ``anchor`` whose open-finger footprints + 25 mm see one flat
    height equal to the lowest height among the candidates on the table (ORACLE placement only)."""
    xys = [(anchor[0] + dx, anchor[1] + dy) for dx, dy in TABLE_CANDIDATES_M]
    zs = [None if z is None or z < anchor[2] - TABLE_MAX_DROP_M else z for z in heights(sim, xys)]
    lowest = min(z for z in zs if z is not None)
    tried = []
    for (dx, dy), xy, z in zip(TABLE_CANDIDATES_M, xys, zs):
        hi = max_surface_near(sim, (xy[0], xy[1], 1.0), R, g_open_mm, WIDE_MARGIN_M)
        ok = z is not None and hi is not None and abs(hi - z) < FLAT_TOL_M and abs(z - lowest) < TABLE_TOL_M
        tried.append({"offset_m": [dx, dy], "z": z, "max_near_z": hi, "clear": bool(ok)})
        if ok:
            return {"xy": [float(xy[0]), float(xy[1])], "z": float(z), "offset_m": [dx, dy], "tried": tried}
    return {"xy": None, "z": None, "tried": tried}


def descend(d, kind, travel, g3, quat):
    view = d.view
    if kind == "a1_g3":
        return P.hover_arrival_descent(view, travel_m=travel, touch_threshold_mm=g3)
    if kind == "a2_stall_only":
        return P.hover_arrival_descent(view, travel_m=travel, touch_threshold_mm=NO_TOUCH_MM)
    if kind == "b_reference":
        return AnchoredMove(view, axis=(0, 0, -1), travel_m=travel, gripper="open", stop_on_contact=False,
                            orientation_xyzw=quat, arm_stall_after_motion=True)
    if kind == "v1_rule":           # the Wave-1 leg: v1's step-3 stall rule, G3 off
        return P._ProbeLeg(view, axis=(0, 0, -1), travel_m=travel, gripper="open", command_m=P.P15_COMMAND_M,
                           stop_on_contact=True, touch_threshold_mm=NO_TOUCH_MM, lateral_cap_m=P.P15_LATERAL_CAP_M,
                           orientation_xyzw=quat, lateral_target=view.position, arm_stall_after_motion=True)
    raise ValueError(kind)


def contact_site(d, sim, name, xy, surface_z, hover_ref_z, g3, quat, g_open_mm):
    target = np.array([xy[0], xy[1], hover_ref_z + HOVER_CLEARANCE_M + ROBOT.t_f_m])
    site = {"site": name, "xy": list(map(float, xy)), "surface_z_under_grip": surface_z, "hover_ref_z": hover_ref_z,
            "target": target.tolist(), "runs": {}}
    for kind in ("b_reference", "a1_g3", "a2_stall_only"):
        d.goto(target, quat)
        site["goto_" + kind] = d.goto(target, quat, tol=CONTACT_TOL_M)     # same start for the three descents
        start_mm = 1000.0 * (float(d.view.position[2]) - ROBOT.t_f_m - hover_ref_z)
        leg = descend(d, kind, CONTACT_TRAVEL_M, g3, quat)
        d.run(leg, kind)
        row = leg_row(leg, kind, sim, quat, g_open_mm)
        row["start_tip_above_ref_mm"] = start_mm
        row["collapse_step_own"] = collapse_step(row["ratios"])
        row["tip_minus_surface_under_grip_mm"] = 1000.0 * (float(leg.end_xyz[2]) - ROBOT.t_f_m - surface_z)
        site["runs"][kind] = row
        print("contact", name, kind, row["reason"], row["steps"], row["collapse_step_own"], flush=True)
    d.goto(target, quat)
    ref = site["runs"]["b_reference"]["collapse_step_own"]
    for kind in ("a1_g3", "a2_stall_only"):
        run = site["runs"][kind]
        run["collapse_step_reference"] = ref
        run["delay_vs_reference"] = None if ref is None else run["steps"] - ref
        run["delay_vs_own"] = None if run["collapse_step_own"] is None else run["steps"] - run["collapse_step_own"]
    return site


def near_start(d, sim, name, xy, surface_z, g3, quat, g_open_mm):
    rows = []
    for h in NEAR_START_M:
        for kind in ("a1_g3", "a2_stall_only", "v1_rule"):
            d.goto([xy[0], xy[1], surface_z + HOVER_CLEARANCE_M + ROBOT.t_f_m], quat)
            goto = d.goto([xy[0], xy[1], surface_z + h + ROBOT.t_f_m], quat, tol=NEAR_TOL_M)
            start_mm = 1000.0 * (float(d.view.position[2]) - ROBOT.t_f_m - surface_z)
            leg = descend(d, kind, NEAR_TRAVEL_M, g3, quat)
            d.run(leg, kind)
            row = leg_row(leg, kind, sim, quat, g_open_mm)
            row.update(site=name, start_mm=start_mm, nominal_start_mm=1000.0 * h, goto=goto,
                       collapse_step_own=collapse_step(row["ratios"]),
                       end_tip_minus_surface_mm=1000.0 * (float(leg.end_xyz[2]) - ROBOT.t_f_m - surface_z))
            rows.append(row)
            print("near", name, round(start_mm, 2), kind, row["reason"], row["steps"],
                  [round(r, 3) for r in row["ratios"][:6]], flush=True)
    d.goto([xy[0], xy[1], surface_z + HOVER_CLEARANCE_M + ROBOT.t_f_m], quat)
    return rows


def run_episode(args):
    H = harness()
    oracle = H.oracle_module()
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    sim, spec, _cfg, _monitor = H.start_sim(TASK_KEY, oracle=True)
    report = {"tool": "scripts/agent_speedup_v2_stall_check.py (BM RS-3)", "episode_index": args.episode,
              "primitives_version": P.PRIMITIVES_VERSION, "hovers": [], "contacts": [], "near_start": []}
    try:
        obs, episode = H.reset_episode(sim, spec, args.episode)
        report["reset"] = episode
        answers = oracle.OracleAnswers(sim, spec.key)
        d = Driver(sim, obs)
        start_q = d.view.quaternion_xyzw.copy()
        # 1. Calibrate+ (M-2 split), every transition kept for the replay
        d.transitions = []
        cal = P.CalibratePlus(d.view, recorder=None, robot=ROBOT)
        cal_out = d.run(cal, "calibrate_plus")
        transitions, d.transitions = d.transitions, None
        record = cal.record()
        g3 = P.gripper_quantities(record)["thresholds"][2]
        g_open = record.gripper.open_gap_mm
        report["calibration"] = {"reason": cal_out["reason"], "steps": cal_out["actions"], "g_open_mm": g_open,
                                 "g3": g3.to_dict(), "stages": cal_out["evidence"]["stages"],
                                 "final_offset_m": cal_out["evidence"]["final_offset_m"],
                                 "split_check": calibrate_split_check(transitions, record)}
        del transitions
        print("calibration", cal_out["reason"], cal_out["actions"],
              report["calibration"]["split_check"].get("locatable"), flush=True)
        # 2. grip-mode hovers
        truth = answers.truth_points()
        for pid in args.parts:
            for offset in HOVER_OFFSETS_M:
                t = truth[pid]
                target = np.array([t[0] + offset[0], t[1] + offset[1], t[2] + HOVER_CLEARANCE_M + ROBOT.t_f_m])
                entry = {"pid": pid, "offset": list(offset), "anchor": list(map(float, t)), "target": target.tolist()}
                entry["goto"] = d.goto(target + np.array([0.0, 0.0, HOVER_LOCATE_M]), start_q)
                descent = P.hover_arrival_descent(d.view, travel_m=HOVER_LOCATE_M, touch_threshold_mm=g3)
                d.run(descent, "hover_arrival")
                entry["arrival"] = leg_row(descent, "hover_arrival", sim, start_q, g_open)
                entry["hover_clearance_mm"] = clearance_mm(sim, d.view.position, start_q, g_open)
                legs = []

                def on_step(_prim, current, stage, _before, _after):
                    if isinstance(current, AnchoredMove) and current.done:
                        legs.append(leg_row(current, stage, sim, start_q, g_open))
                probe = P.ParallaxProbe(d.view, recorder=None, gripper="open", amplitude_m=P.D2_PROBE_M,
                                        touch_threshold_mm=g3)
                out = d.run(probe, on_step=on_step)
                entry["probe"] = {"reason": out["reason"], "legs": legs,
                                  "net_displacement_norm_m": out["evidence"].get("net_displacement_norm_m")}
                report["hovers"].append(entry)
                stops = [(leg["stage"], leg["reason"], leg["steps"]) for leg in [entry["arrival"]] + legs
                         if leg["reason"] != "travel_reached"]
                print(pid, offset, entry["arrival"]["reason"], out["reason"], stops, flush=True)
        # 3. contacts
        R = quat_to_R(start_q)
        b = truth["B"]
        plate_z = heights(sim, [(b[0], b[1])])[0]
        report["contacts"].append(contact_site(d, sim, "plate", (b[0], b[1]), plate_z, plate_z, g3, start_q, g_open))
        a = truth["A"]
        table = find_table_site(sim, a, R, g_open)
        report["table_site"] = table
        if table["xy"] is None:
            raise RuntimeError("no footprint-clear table site among TABLE_CANDIDATES_M")
        report["contacts"].append(contact_site(d, sim, "table", table["xy"], table["z"], table["z"], g3, start_q,
                                               g_open))
        lever_xy = (a[0] + LEVER_TOP_OFFSET_M[0], a[1] + LEVER_TOP_OFFSET_M[1])
        lever_under = heights(sim, [lever_xy])[0]
        lever = contact_site(d, sim, "lever_top", lever_xy, lever_under, float(a[2]), g3, start_q, g_open)
        lever["note"] = ("a finger lands on A's top (anchor z), not on the surface under the grip; reported only, "
                         "not an RS-3 plate or table contact")
        report["contacts"].append(lever)
        # 4. near-start contacts (reported only)
        report["near_start"] += near_start(d, sim, "plate", (b[0], b[1]), plate_z, g3, start_q, g_open)
        report["near_start"] += near_start(d, sim, "table", table["xy"], table["z"], g3, start_q, g_open)
        report["steps"] = d.steps
    finally:
        report["wall_s"] = round(time.monotonic() - began, 1)
        sim.close()
    (args.out / f"episode_{args.episode}.json").write_text(json.dumps(H.jsonable(report), indent=1))
    print("done", args.episode, report["wall_s"], flush=True)


# ---------------------------------------------------------------------------------------------------- summary
def summarize(run_dir):
    run_dir = Path(run_dir)
    eps = [json.load(open(f)) for f in sorted(glob.glob(str(run_dir / "episode_*.json")))]
    hovers, legs, arrivals = 0, [], []
    false_stalls, true_stops, other_stops = [], [], []
    for e in eps:
        for h in e["hovers"]:
            hovers += 1
            arrivals.append(h["arrival"])
            legs.extend(h["probe"]["legs"])
            for r in [h["arrival"]] + h["probe"]["legs"]:
                if r["reason"] == "travel_reached":
                    continue
                clear = r.get("truth_clearance_mm")
                item = {"ep": e["episode_index"], "pid": h["pid"], "offset": h["offset"], "stage": r["stage"],
                        "reason": r["reason"], "steps": r["steps"], "clearance_mm": clear, "ratios": r["ratios"][:8]}
                if r["reason"] == "contact_stall" and (clear is None or clear > FREE_CLEARANCE_MM):
                    false_stalls.append(item)
                elif clear is not None and clear <= FREE_CLEARANCE_MM:
                    true_stops.append(item)
                else:
                    other_stops.append(item)
    all_legs = legs + arrivals
    free_ratios = [r["free_ratio"] for r in all_legs if r["free_ratio_measured"]]
    margins = []
    for r in all_legs:
        if r["reason"] == "travel_reached" and r["steps"] >= P.PROBE_STALL_MIN_STEPS:
            limit = P.G12_STALL_FRACTION * r["free_ratio"]
            windows = [(r["ratios"][k - 1] + r["ratios"][k - 2]) / 2.0
                       for k in range(P.PROBE_STALL_MIN_STEPS, r["steps"] + 1)]
            margins.append(min(windows) / limit - 1.0)
    contacts, lever_top = [], []
    for e in eps:
        for site in e["contacts"]:
            ref = site["runs"]["b_reference"]["collapse_step_own"]
            for kind in ("a1_g3", "a2_stall_only"):
                run = site["runs"][kind]
                caught = (run["reason"] in ("contact_stall", "contact_touch") and ref is not None
                          and run["steps"] <= ref + 1)
                row = {"ep": e["episode_index"], "site": site["site"], "run": kind, "reason": run["reason"],
                       "stop_step": run["steps"], "collapse_own": run["collapse_step_own"], "collapse_reference": ref,
                       "delay_vs_reference": run.get("delay_vs_reference"),
                       "truth_clearance_mm": run.get("truth_clearance_mm"),
                       "tip_minus_surface_under_grip_mm": run.get("tip_minus_surface_under_grip_mm"),
                       "start_mm": run.get("start_tip_above_ref_mm"),
                       "reference_start_mm": site["runs"]["b_reference"].get("start_tip_above_ref_mm"),
                       "caught_by_collapse_plus_1": caught}
                (contacts if site["site"] in CONTACT_SITES_SCORED else lever_top).append(row)

    def tally(site, kind):
        rows = [c for c in contacts if c["site"] == site and c["run"] == kind]
        return {"caught": f"{sum(c['caught_by_collapse_plus_1'] for c in rows)}/{len(rows)}",
                "start_mm_minus_reference": [round(c["start_mm"] - c["reference_start_mm"], 2) for c in rows
                                             if c["start_mm"] is not None and c["reference_start_mm"] is not None],
                "delays_vs_reference": sorted({c["delay_vs_reference"] for c in rows}, key=lambda v: (v is None, v)),
                "reasons": sorted({c["reason"] for c in rows}),
                "truth_clearance_mm": [round(c["truth_clearance_mm"], 2) for c in rows
                                       if c["truth_clearance_mm"] is not None]}
    split = []
    for e in eps:
        sc = e["calibration"]["split_check"]
        moves = {m["stage"]: m for m in sc.get("moves") or []}
        halves = [s for s in moves if s.startswith("vertical_")]
        split.append({"ep": e["episode_index"], "steps": e["calibration"]["steps"], "n_external": sc["n_external"],
                      "locatable": sc.get("locatable"), "located": (sc.get("locator") or {}).get("located"),
                      "sigma_consistency_px": (sc.get("locator") or {}).get("sigma_consistency_px"),
                      "corrupt_40px_dropped": sum(1 for v in (sc.get("corrupt_40px") or {}).values()
                                                  if v and v["dropped_bad"]),
                      "corrupt_40px_tried": len(sc.get("corrupt_40px") or {}),
                      "half_along_px": {s: round(moves[s]["along_px"], 2) for s in halves},
                      "min_half_abs_along_px": min((abs(moves[s]["along_px"]) for s in halves), default=None),
                      "f_hat_rel_error": sc.get("f_hat_rel_error_vs_R3"),
                      "vertical_reasons": [m["reason"] for m in sc.get("vertical_record") or []]})
    near = [{"ep": e["episode_index"], "site": r["site"], "start_mm": round(r["start_mm"], 2), "run": r["stage"],
             "reason": r["reason"], "steps": r["steps"], "collapse_own": r["collapse_step_own"],
             "ratios": [round(x, 3) for x in r["ratios"][:6]],
             "end_tip_minus_surface_mm": round(r["end_tip_minus_surface_mm"], 2)} for e in eps for r in e["near_start"]]
    out = {
        "run_dir": str(run_dir), "episodes": [e["episode_index"] for e in eps], "hovers": hovers,
        "probe_legs": len(legs), "arrival_descents": len(arrivals),
        "legs_reaching_step_5": sum(1 for r in all_legs if r["steps"] >= P.PROBE_STALL_MIN_STEPS),
        "false_contact_stalls": len(false_stalls), "false_contact_stall_detail": false_stalls,
        "true_stops_in_hovers": true_stops, "other_stops_in_hovers": other_stops,
        "probe_outcomes": {k: sum(1 for e in eps for h in e["hovers"] if h["probe"]["reason"] == k)
                           for k in ("probe_complete", "probe_contact")},
        "measured_free_ratio_min": min(free_ratios) if free_ratios else None,
        "free_leg_min_stall_margin": min(margins) if margins else None,
        "table_sites": [e.get("table_site", {}).get("offset_m") for e in eps],
        "rs3_contacts": {f"{site}/{kind}": tally(site, kind) for site in CONTACT_SITES_SCORED
                         for kind in ("a1_g3", "a2_stall_only")},
        "contacts": contacts, "lever_top_reported_only": lever_top, "near_start_reported_only": near,
        "calibrate_split": split,
        "wall_s": [e.get("wall_s") for e in eps], "steps": [e.get("steps") for e in eps],
    }
    out["RS3_pass"] = bool(hovers >= 100 and not false_stalls and contacts
                           and all(c["caught_by_collapse_plus_1"] for c in contacts)
                           and {c["site"] for c in contacts} == set(CONTACT_SITES_SCORED))
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1))
    brief = {k: v for k, v in out.items() if k not in ("contacts", "true_stops_in_hovers", "false_contact_stall_detail",
                                                        "lever_top_reported_only", "near_start_reported_only")}
    print(json.dumps(brief, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--episode", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--parts", nargs="+", default=list(PARTS))
    s = sub.add_parser("summarize")
    s.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        run_episode(args)
    else:
        summarize(args.run_dir)


if __name__ == "__main__":
    main()
