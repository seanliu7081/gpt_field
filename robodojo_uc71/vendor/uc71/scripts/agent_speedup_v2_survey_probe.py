"""Policy-free G1' survey probe on the real simulator (TEST TOOL; contract Appendix L decision 10, §C5 real-sim, BG;
design review 1 decision §2.3 G-6, §4, §5.2 RS-1 / RS-8).

A diagnosis and gate tool, zero model calls. It drives the robot with the pipeline's own primitives and modules (no
policy; BR, Wave 2, is restated where it is not built yet) and scores what the planner reads, the registered
PartRecord, against evaluator truth read through the ORACLE op (``scripts/agent_speedup_v2_oracle.py``).

PIPELINE PATH (rulings item 6, ruling 4d: the ORACLE-seeded survey is ended; ``--hover seeded`` stays a diagnosis):
1. Calibrate+ (``calibrate_plus``, BR's calibrate_plus restated): BM's ``CalibratePlus`` with a probe-tracker
   ``SurveyRecorder`` (grid only, external camera; decision §2.3 G-1): result(None) -> ``CameraModel.decide`` ->
   result(camera); ``ExternalModel``, ``HandLocator`` (G-5: n <= 3 absolute bound, 2 agreeing -> degraded),
   ``ExternalProjector``; fingertip band and false-change maps from the recorder's frames; ``V2Calibration``;
   BS ``SceneModel.start``.
2. The ORACLE plan answer (plan cells: the model's stand-in) -> ``merge_plan`` -> start registrations (grip-only
   parts provisional, the others allow_partial), as BR's register_regions.
3. Per part and hover offset (episodes 1-5 x 4 offsets of +-1 cm, from a fresh copy of the post-start scene so every
   measurement is independent): chain hover = BP ``hover_for`` on the MEASURED start record (+ the offset), BR's
   free_move restated (``planner.free_path`` waypoints, B4 settle); ORACLE fine pointing (the model's point_fine
   stand-in, 8 px quantization; a miss -> point_coarse -> point_fine); BM ``ParallaxProbe`` feeding a probe-tracker
   ``SurveyRecorder`` (exact seed, the dense lattice at the pointed pixel, the scene's relief targets);
   ``absorb_survey``; ``register(seed=fine)``. BR §2.6 restated: a ``first_stage`` hover (the P11 cap bound) is
   followed by hover_for(record) and a second hover when it moves > 1 cm in xy or > 2 cm in z; BR-1 (one more
   ParallaxProbe at the same yaw shifted by D2 along R1, surveys fused) on design review 2's triggers (below). The
   scored record is the one the scene holds after the hover (the last stage's that returned a PartRecord: a later
   stage that finds nothing or fails leaves the scene's registration in place; with S-12's view memory), and the
   same record is replayed through BP; every stage's record is kept as a diagnosis. The exact-seed lateral error
   uses the ORACLE truth pixel projected in the SEED pose.

ORACLE use, none of it entering a measurement: (1) the plan answer, (2) the fine / coarse pointing stand-in, (3)
scoring: the part's own visual and collision tops (``part_top_truth``), its collision cells on a heights grid (axis,
widths per jaw yaw, support under it), segment classes, the grip projection.

GATES (decision §4, per part class; A and C must pass G1' and G-yaw, B, D, E are reported):
- G1': (a) registered and not partial >= 18/20; (b) |top_z - visual top| median <= 3 mm, p90 <= 5 mm (collision
  error reported); (c) |dh| <= 2 sigma_h in >= 90 %, <= 3 sigma_h in >= 98 %, sigma_h <= the P6 limit in >= 90 %;
  (d) 0 violations of e = top_z - collision top in [-(14.4 - m_t), +(8 + m_t)] mm, m_t = max(2 mm, 1.5 sigma_h);
  (d') a hover with m_t >= 14.4 mm has an empty envelope ("sigma_above_envelope", listed, not a violation);
  (e) as amended by design review 2 (below: e1-e4; the review-1 share within max(3 mm, 15 %), review R7, is
  replaced and only reported); (f) f-hat within 5 % of 333.6.
- G-yaw (elongated grip parts): 0 accepted axes wrong by more than YAW_GRID_RAD / 2; >= 15/20 accepted (after the
  second hover); the grip error across the jaw with the exact seed p90 <= 4 mm; +-8 / +-16 px pointing reported.
- G1c offline part (admission, diagnosis here; RS-2 is BS + BP's): BP ``plan_contact`` on every final record of a grip
  part (``part_fill_cells`` passed when the planner takes it), plans per part, the plan's yaw error and top envelope.
Also reported: sigma_vc AT PART TOPS (ruling 4a: the median visual - collision top of each part), the raw-sample
statistics of the old G1 (on-part p95, z-scores, texture, edges; diagnosis), the HandLocator at the hovers and on the
Calibrate+ moves (the moves the locator refuses are listed, not scored), the Calibrate+ start-survey samples against
the decision's cited 110-121, the compute (RS-8: ms per tracked frame at the operating feature count, ``result()``
seconds per hover), and two diagnoses of the D10 axis for BS / BR / the lead (review R1): the S-4' outcome of the
image tensor on the raw and band-passed grey at several half sizes, and the tally if BR kept an accepted earlier-stage
axis.

DESIGN REVIEW 2 (decision §2.2 G-9, §2.6, §4):
- BR-1 restated with the amended triggers (§2.6): band n < TOP_BAND_SUPPORT_N; an axis refusal AXIS_INCONSISTENT
  (S-11 "no_edge_agreement", S-12 "views_disagree") ON A GRIP PART; ``region_collapsed`` (S-13) on a grip part; P-C
  ``refine_axis``. "not_elongated" is never a trigger. The adopted-but-untested ``cut_by_border`` trigger is not
  restated (the RegistrationFailure carries no cut direction; BR / BS, Wave 2).
- every hover starts from a fresh scene copy whose view memory is empty (S-12: ``scene.part_moved(pid)`` on the copy);
  chain -> second stage -> BR-1 is one memory window. The SCORED record is the record the scene holds after the
  hover (BR's semantics, §4.1), with S-12's view memory and S-13's collapse guard; runs recorded before S-12 / S-13
  existed (the judge's live runs) get both applied offline from their stages (``held_record``, mode "emulated").
- G1'(e) as amended (§4.2): e1 0 delivered widths from a collapsed region (grip parts); e2 the delivered width within
  the G13 band max(G13_FLOOR_MM, G13_SIGMA_K sigma_w) of the collision cross-section in >= 90 % of the registered
  hovers of a FREE grip part (move prior not in FIXTURE_PRIORS; the share over the delivered hovers, the judge's
  denominator, is reported beside it); e3 0 under-reads beyond the G13+ trip margin (band + PAD_LENGTH_R0_MM
  sin(YAW_GRID_RAD / 2)); e4 fixture grip parts reported (their consumers are gated by G1c); the share within
  max(3 mm, 15 %) and the ORACLE visible surface within 2 / 5 / 10 mm of the top reported for every part. The width
  truth is the collision cross-section at the held record's own grip point (§5 ruling; ``width_truth_at`` on the
  episode's ORACLE cell grid); runs recorded without the grid (the judge's, earlier BG runs) use the anchor strip.
- RS-4 as restated (§4.3) on the hover locator readings (V2v, geometry G-8): at the calibration yaw (+-5 deg) >= 9/10
  located and every located reading within 4 px of the ORACLE grip projection; at the other yaws every reading
  refused or within 2 sigma_loc. ``--hover-yaw-deg`` turns every transit / hover orientation about world z after
  Calibrate+ (the generality and RS-4 yaw sets).
- generality reports (§4.4): a run on fresh episodes (16-20) or a turned hover yaw is labelled "generality" (M7, not
  entry-gating); its gates are computed the same way. "gate" is only episodes 1-5 with every hover yaw known and
  within RS4_CALIBRATION_YAW_DEG; "other" (e.g. held-out 11-15) and "unknown" (a yaw neither the record nor its
  manifest gives) are labelled as such.

DESIGN REVIEW 2 CLOSE-OUT (lead rulings r2close; the scene side is BS's):
- ruling 3: after an S-15 ``seed_outside_region`` refusal at a GRIP part's hover stage (chain, second stage or BR-1)
  the survey restates BR's ONE re-point at the hover with the ORACLE stand-in (``repoint_after_seed_refusal``): fine
  pointing centred on the scene's predicted pixel, coarse first when its sigma exceeds COARSE_FIRST_SIGMA_PX (64 /
  sqrt(2 ln 2) px, rulings item 9), then one more registration from that seed on the same surveyed view (logged as
  the stage's ``repoint``); the stage scores the re-pointed seed and the record the scene holds afterwards;
- ruling 5: RS-4's calibration class bounds every located reading by max(RS4_ACCURACY_PX, RS4_SIGMA_K sigma_loc) (the
  literal 4 px reading reported); ruling 6: G1'(e2)'s share is over the hovers that delivered a width (the share over
  the registered hovers reported);
- ruling 8: every real-sim gate runs twice on the same code and both runs must pass: ``--pair RUN_A RUN_B
  --report-out PATH`` (``pair_report``, read only) gives each run's verdicts, ``both_pass``, the verdicts and hovers
  that differ and whether both ran the same sources; each aggregate reports ``r2close``: the S-15 refusals, admitted
  outside seeds, re-points and every gap within S15_NEAR_LIMIT_PX of its limit, and the ``edges_bimodal`` refusals
  of ruling 4 (a).

usage (project venv; the worker runs in --sim-python; one _LiberoWorker per process, so one episode per process):
  env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=2 .venv/bin/python scripts/agent_speedup_v2_survey_probe.py \\
      --episodes 1 --out runs/agent_speedup_v2_bg_r1_survey_001/ep1 [--parts A B C D E] [--offsets 4]
  .venv/bin/python scripts/agent_speedup_v2_survey_probe.py --reaggregate runs/agent_speedup_v2_bg_r1_survey_001
      (merges every episode_*.json below the directory into report.json)
"""

from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gpt_field import agent_speedup_v2_geometry as geometry  # noqa: E402
from gpt_field import agent_speedup_v2_robot as robot_facts  # noqa: E402
from gpt_field.agent_speedup_v2_types import (HandPose, HeightField, PartRecord, RobotModel, Seed,  # noqa: E402
                                              V2Calibration, YAW_GRID_RAD)
from gpt_field.contact_skills import (AnchoredMove, HoldPrimitive, HoldingActionAdapter, StepView,  # noqa: E402
                                      WaypointReach)

SURVEY_PROBE_VERSION = "agent-speedup-v2-survey-probe-7"
HARNESS_FILE = ROOT / "scripts" / "agent_speedup_v2_relation_smoke.py"
ORACLE_FILE = ROOT / "scripts" / "agent_speedup_v2_oracle.py"
PROBE_MOVES = ((0.040, 0.0, 0.0), (-0.040, 0.040, 0.0), (0.0, -0.040, 0.0))   # v1 PROBE_MOVES (legacy calibrate)
VERTICAL_PROBE_M = 0.04                     # R-1v (legacy calibrate)
HOVER_PROBE_M = 0.02                        # D2
HOVER_OFFSETS_M = ((0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01))
OPEN_HOLD_STEPS = 10
CLOSE_STEPS = 30
OPEN_STEPS = 20
SETTLE_STEPS = 14
GOTO_TOL_M = 0.002
GOTO_MAX_STEPS = 220
GOTO_GAIN = 1.5
EDGE_PX = 4.0                               # a feature within 4 px of the part mask border is an edge feature
TOP_PERCENTILE = 95                         # spec §4.5 registration step 4 (the raw-sample diagnosis)
HOVER_CLASS_MIN_SAMPLES = 3                 # raw-sample diagnosis: a hover measures a class from 3 on-part samples
HOVER_LOCATE_M = 0.03                       # the HandLocator's hover move: 3 cm vertical (along v_hat)
PART_TOP_RADIUS_M = 0.015                   # ORACLE part top: grid within 1.5 cm of the anchor (near the grasp)
PART_TOP_STEP_M = 0.0015                    # ORACLE part top grid spacing
PART_TOP_XY_TOL_MM = 3.0                    # a visual hit belongs to its grid column within 3 mm
TRUTH_GRID_HALF_M = 0.05                    # ORACLE part cells: heights grid +-5 cm around the anchor, 1 mm ...
TRUTH_GRID_STEP_M = 0.001
TRUTH_GRID_WIDE = {"B": (0.12, 0.002), "D": (0.12, 0.002)}   # ... +-12 cm at 2 mm for the plate and the body
SUPPORT_RING_M = (0.005, 0.020)             # ORACLE support under a part: non-part cells 5-20 mm outside it ...
SUPPORT_BELOW_TOP_M = 0.010                 # ... at least 1 cm below the part top (median)
FINE_QUANT_PX = 8.0                         # Seed quantization of a fine answer (16 px crop-point spacing, half)
COARSE_QUANT_PX = 32.0                      # Seed quantization of a coarse answer (64 px cell, half)
FINE_CROP_PX = 128                          # the fine pointing crop (spec §3.2), centred on the predicted pixel
# rulings item 9 / r2close lead ruling 3: point coarse first when the predicted pixel's sigma exceeds the crop's
# half width / sqrt(2 ln 2) (64 / 1.177 = 54.4 px)
COARSE_FIRST_SIGMA_PX = (FINE_CROP_PX / 2.0) / math.sqrt(2.0 * math.log(2.0))
S15_NEAR_LIMIT_PX = 0.25                    # r2close: an S-15 gap within 0.25 px of its limit is flagged (ruling 8)
TWO_STAGE_XY_M, TWO_STAGE_Z_M = 0.01, 0.02  # decision §2.6: take the second hover when it moves > 1 cm xy / 2 cm z
G1_MEDIAN_MM, G1_P90_MM, F_HAT_TOL, LOCATOR_TOL_PX = 3.0, 5.0, 0.05, 6.0
G1_MIN_MEASUREMENTS = 20                    # spec §8.2 M1: >= 20 hover-probe measurements per part class
G1P_REGISTERED_MIN = 18                     # decision §4 G1'(a): >= 18/20 registered and not partial
G1P_2SIGMA_SHARE, G1P_3SIGMA_SHARE, G1P_P6_SHARE = 0.90, 0.98, 0.90     # G1'(c)
ENVELOPE_LOW_MM, ENVELOPE_HIGH_MM = 14.4, 8.0   # G1'(d) remedy 1 §5 d: p_min - pad_top - m_t - P6_PALM_FLOOR = 14.4 -
ENVELOPE_MT_FLOOR_MM, ENVELOPE_MT_K = 2.0, 1.5  # m_t; pad_top + pad_bottom - P3 + m_t = 8 + m_t; m_t = max(2, 1.5 s)
WIDTH_TOL_MM, WIDTH_TOL_REL = 3.0, 0.15     # G1'(e) spec §8.2 M1
G1P_E2_SHARE = 0.90                         # G1'(e2) (decision 2 §4.2, lead ruling 6): >= 90 % of the delivered hovers
G1P_E_BAND_VISUAL_DEPTHS_MM = (2, 5, 10)    # G1'(e): the ORACLE visible surface within 2 / 5 / 10 mm of the top (reported)
GYAW_ACCEPTED_MIN = 15                      # G-yaw: >= 15/20 accepted axes
GYAW_LATERAL_P90_MM = 4.0                   # G-yaw: across-jaw grip error with the exact seed, p90
POINTING_ERRORS_PX = (8.0, 16.0)            # G-yaw: reported pointing errors
RS4_CALIBRATION_YAW_DEG = 5.0               # RS-4 (decision 2 §4.3): the calibration-yaw class is +-5 deg
RS4_LOCATED_SHARE = 0.9                     # ... located in >= 9/10 hovers
RS4_ACCURACY_PX = 4.0                       # ... every located reading within max(4 px, 2 sigma_loc) (lead ruling 5)
RS4_SIGMA_K = 2.0                           # other yaws: every reading refused or within 2 sigma_loc
GENERALITY_EPISODES = range(16, 21)         # decision 2 §4.4 / §6.2: the fresh-episode set (M7, not entry-gating)
GATE_EPISODES = range(1, 6)                 # decision 2 §6.2 RS-1: the gate set is episodes 1-5 at standard yaw
VISUAL_STRIP_HALF_M = 0.008                 # ORACLE visible-surface width: the pad strip half length (R4 / 2)
VISUAL_TOP_PERCENTILE = 99                  # ... the strip's visible top (99th percentile of its heights)
PART_CLASS = {"A": "rotary_ridge", "B": "region_plate", "C": "thin_loop", "D": "object_body", "E": "pan_handle"}
PART_MOVE = {"A": "hinged", "B": "fixed", "C": "free", "D": "free", "E": "free"}
ELONGATED = ("A", "C", "E")                 # scoring only: the parts whose truth region is elongated (G-yaw)
GATED = ("A", "C")                          # decision §5.2 RS-1: must pass G1' and G-yaw; B, D, E reported
DEVIATIONS = [
    "BR (Wave 2) is restated here: calibrate_plus, register_regions, free_move (planner.free_path + WaypointReach "
    "segments + B4 settle), the two-stage hover and the BR-1 second hover of decision §2.6, and register_regions' "
    "one point_coarse re-seed after a failed start registration; a part still unregistered is hovered from its "
    "seed (the coarse re-seed, else its plan seed) lifted by scene.estimate_seed_xyz",
    "while BP lacks decision §2.4 P-E (planner.P11_CAP_M absent) it is restated here: the P11 cap (first_stage), a "
    "registered record's own top as z_ref and the P-2 visibility rule, applied to the record at the hover and to "
    "the pointed pixel before the probe (the hand moves along -R0, the pixel carried by the move)",
    "ORACLE stand-ins for the model: the plan answer (plan cells) and the fine / coarse pointing answers"]


# ----------------------------------------------------------------------------------------------- plumbing
def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harness():
    return _load("agent_speedup_v2_relation_smoke", HARNESS_FILE)


def oracle_module():
    return harness().oracle_module()


def jsonable(value):
    return harness().jsonable(value)


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Driver:
    """Steps the real simulator and hands every transition to the active recorders (scripted, policy-free); with a
    ``scene`` every new view also goes to ``scene.on_view`` (the policy's per-act call)."""

    def __init__(self, sim, obs, scene=None):
        self.sim = sim
        self.view = StepView(obs)
        self.steps = 0
        self.scene = scene

    def step(self, action, stage=None, recorders=()):
        obs, _success = self.sim.step([float(v) for v in action])
        after = StepView(obs)
        for recorder in recorders:
            recorder.on_transition(stage, self.view, after)
        self.view = after
        self.steps += 1
        if self.scene is not None:
            self.scene.on_view(after, step=self.steps)
        return after

    def run(self, primitive, stage, recorders=()):
        while not primitive.done:
            action = primitive.next_action(self.view)
            before = self.view
            after = self.step(action, stage, recorders)
            primitive.observe_transition(before, action, after)
        return primitive.outcome()

    def goto(self, target, quat_xyzw, *, gripper="open"):
        """Proportional move of the grip to ``target`` holding the orientation (v1 HoldingActionAdapter: never an
        exactly-zero rotation); then a quiet settle."""
        adapter = HoldingActionAdapter()
        target = np.asarray(target, dtype=float)
        for _ in range(GOTO_MAX_STEPS):
            error = target - self.view.position
            if float(np.linalg.norm(error)) < GOTO_TOL_M:
                break
            delta = np.clip(error * GOTO_GAIN, -robot_facts.TRANSLATION_PER_UNIT_M, robot_facts.TRANSLATION_PER_UNIT_M)
            self.step(adapter.adapt(delta, quat_xyzw, gripper, self.view.state))
        self.run(HoldPrimitive(self.view, steps=SETTLE_STEPS, gripper=gripper, until_quiet=True,
                               orientation_xyzw=quat_xyzw), "settle")
        return float(np.linalg.norm(target - self.view.position))


# ----------------------------------------------------------------------------------------------- classification
def decode_mask(png_url):
    from PIL import Image
    raw = base64.b64decode(png_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw))) > 0


# ----------------------------------------------------------------------------------------------- classification
def classify_pixels(sim, answers, camera, uvs, masks=None):
    """Surface class of each pixel from the ORACLE (scoring only): first the parts' visible masks (the segment mask
    limited to the part's own collision boxes, ``OracleAnswers.visible_mask``: the rendered geoms are visual meshes,
    so collision-geom names alone cannot label them), then the segment owner / body at the pixel."""
    if not uvs:
        return []
    masks = part_masks(answers, camera) if masks is None else masks
    reply = sim.oracle({"kind": "segment", "camera": camera, "masks": [], "mask_png": False,
                        "uv": [list(map(float, uv)) for uv in uvs]})
    regions = {pid: set(spec.get("region") or []) for pid, spec in answers.parts.items()}
    out = []
    for uv, pixel in zip(uvs, reply["at"]):
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        label = None
        for pid in ("A", "C", "E", "B", "D"):
            mask = masks.get(pid)
            if mask is not None and 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                label = PART_CLASS[pid]
                break
        out.append(label or _surface_class(pixel, answers.parts, regions))
    return out


def part_masks(answers, camera):
    return {pid: answers.visible_mask(pid, camera) for pid in ("A", "B", "C", "D", "E")}


def _surface_class(pixel, parts, regions):
    if pixel is None or pixel.get("geom") is None:
        return "none"
    if pixel.get("robot"):
        return "robot"
    geom, body, owner = pixel.get("geom"), pixel.get("body"), pixel.get("owner")
    for pid in ("A", "C", "E"):
        if geom in regions[pid]:
            return PART_CLASS[pid]
    if body == parts["B"]["body"]:
        return PART_CLASS["B"]
    if body == parts["A"]["body"]:
        return "rotary_disk"
    if owner == parts["D"]["owner"]:
        return PART_CLASS["D"]
    if owner == parts["E"]["owner"]:
        return "pan_body"
    if owner == parts["A"]["owner"]:
        return "stove_base"
    if owner is None:
        return "support"
    return f"other:{owner}"


# ----------------------------------------------------------------------------------------------- scoring
def score_samples(sim, answers, result, view, camera_model, recorder=None):
    """Per valid feature: measured z, sigma, class, visual (wrist) and collision truth z at the measured xy. The
    visual truth's robot flag is kept (``visual_robot``: the rendered hit at that point is the robot, e.g. a finger
    between the camera and the surface; such truths are dropped from every error statistic, BG-R1) and its geom
    (``visual_geom``: None for the invisible grip-site cylinder in the ORACLE depth, sim.py:565; dropped too, G-6);
    ``edge_like`` (with the recorder): the feature's template is edge- or ridge-like (aperture weighted, D1);
    ``dense``: the feature is one of the probe tracker's dense-lattice seeds."""
    features = {f.fid: f for f in result.features}
    samples = list(result.samples)
    if not samples:
        return []
    classes = classify_pixels(sim, answers, "wrist", [s.uv for s in samples])
    truth = answers.surface_truth([s.xyz[:2] for s in samples], cameras=("wrist",))
    rows = []
    for sample, cls, tr in zip(samples, classes, truth):
        visual = (tr.get("visual") or {}).get("wrist") or {}
        feature = features[sample.feature_id]
        edge_like = None
        if recorder is not None and sample.feature_id in recorder.tracker.features:
            aperture = geometry.template_aperture(recorder.tracker.features[sample.feature_id]["template"])
            edge_like = not bool(np.allclose(aperture, np.eye(2)))
        rows.append({"fid": sample.feature_id, "key": feature.key, "uv": list(sample.uv), "xyz": list(sample.xyz),
                     "z": sample.xyz[2], "sigma_z": sample.sigma_z, "kappa": sample.kappa, "class": cls,
                     "visual_z": visual.get("z"), "visual_owner": visual.get("owner"),
                     "visual_geom": visual.get("geom"),
                     "visual_robot": bool(visual.get("robot")) or visual.get("owner") == "robot",
                     "visual_xy_offset_mm": visual.get("xy_offset_mm"), "edge_like": edge_like,
                     "dense": bool(feature.is_dense_lattice()),
                     "collision_z": tr.get("surface_z"),
                     "collision_owner": (tr.get("surface_hit") or {}).get("owner")})
    return rows


def truth_ok(row):
    """A sample's visual truth is usable: present, not the robot (rows recorded before the ``visual_robot`` field
    carry the robot as ``visual_owner``) and on a rendered geom (decision §2.3 G-6: a truth pixel whose geom is None,
    the invisible grip-site cylinder in the ORACLE depth, is dropped; rows recorded before ``visual_geom`` keep the
    old rule)."""
    if "visual_geom" in row and row["visual_geom"] is None:
        return False
    return row.get("visual_z") is not None and not row.get("visual_robot") and row.get("visual_owner") != "robot"


def part_top_truth(answers, pid):
    """The part's own VISUAL top from ORACLE geometry at the current (settled) view, independent of any measured
    xy (BG-R2): a grid (PART_TOP_STEP_M) within PART_TOP_RADIUS_M of the part's anchor (near the grasp point); a
    grid point counts when its highest collision surface is the part's own (its region geoms, else its body /
    owner) and the wrist camera's visual hit there is the part's owner, not the robot, within PART_TOP_XY_TOL_MM of
    the column. Returns {"visual_top_z" (max), "visual_p95_z", "collision_top_z", "points", "kept", "robot"}."""
    spec = answers.parts[pid]
    anchor = np.asarray(answers.truth_points()[pid][:2], dtype=float)
    steps = np.arange(-PART_TOP_RADIUS_M, PART_TOP_RADIUS_M + 1e-9, PART_TOP_STEP_M)
    grid = [anchor + (dx, dy) for dx in steps for dy in steps if math.hypot(dx, dy) <= PART_TOP_RADIUS_M + 1e-9]
    rows = answers.surface_truth([p.tolist() for p in grid], cameras=("wrist",))
    region = set(spec.get("region") or [])
    excluded = set()
    if spec.get("whole"):
        for other in answers.parts.values():
            if other is not spec and other.get("owner") == spec.get("owner"):
                excluded |= set(other.get("region") or [])
    visual, collision, robot = [], [], 0
    for row in rows:
        hit = row.get("surface_hit") or {}
        if region:
            mine = hit.get("geom") in region
        elif spec.get("whole"):
            mine = hit.get("owner") == spec.get("owner") and hit.get("geom") not in excluded
        else:
            mine = hit.get("body") == spec.get("body")
        if not mine or row.get("surface_z") is None:
            continue
        collision.append(float(row["surface_z"]))
        seen = (row.get("visual") or {}).get("wrist")
        if not seen or seen.get("z") is None:
            continue
        if seen.get("robot") or seen.get("owner") == "robot":
            robot += 1
            continue
        if seen.get("owner") != spec.get("owner") or float(seen.get("xy_offset_mm") or 0.0) > PART_TOP_XY_TOL_MM:
            continue
        visual.append(float(seen["z"]))
    return {"visual_top_z": max(visual) if visual else None,
            "visual_p95_z": float(np.percentile(visual, TOP_PERCENTILE)) if visual else None,
            "collision_top_z": max(collision) if collision else None, "points": len(grid), "on_part": len(collision),
            "kept": len(visual), "robot": robot, "anchor_xy": anchor.tolist()}


def _stats(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if values.size == 0:
        return {"n": 0}
    absolute = np.abs(values)
    return {"n": int(values.size), "median": float(np.median(values)), "median_abs": float(np.median(absolute)),
            "p90_abs": float(np.percentile(absolute, 90)), "max_abs": float(absolute.max()),
            "mean": float(values.mean())}


def class_table(rows):
    """Per surface class: errors (mm) against the visual and collision surfaces, D4 z-scores, visual - collision."""
    table = {}
    for cls in sorted({r["class"] for r in rows}):
        part = [r for r in rows if r["class"] == cls]
        visual = [1000.0 * (r["z"] - r["visual_z"]) for r in part if truth_ok(r)]
        collision = [1000.0 * (r["z"] - r["collision_z"]) for r in part if r["collision_z"] is not None]
        zscores = [(r["z"] - r["visual_z"]) / r["sigma_z"] for r in part if truth_ok(r) and r["sigma_z"] > 0]
        vc = [1000.0 * (r["visual_z"] - r["collision_z"]) for r in part if truth_ok(r) and r["collision_z"] is not None]
        table[cls] = {"n": len(part), "truth_robot": sum(not truth_ok(r) and r["visual_z"] is not None for r in part),
                      "z_measured_median": float(np.median([r["z"] for r in part])),
                      "z_visual_median": _median([r["visual_z"] for r in part if truth_ok(r)]),
                      "z_collision_median": _median([r["collision_z"] for r in part]),
                      "err_visual_mm": _stats(visual), "err_collision_mm": _stats(collision),
                      "zscore": {"n": len(zscores), "std": float(np.std(zscores)) if zscores else None,
                                 "robust_std": geometry._robust_sigma(zscores) if zscores else None},
                      "sigma_z_median_mm": float(1000.0 * np.median([r["sigma_z"] for r in part])),
                      "visual_minus_collision_mm": _stats(vc)}
    return table


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def texture_rates(gray, mask, result, classes_by_fid, cls, *, radius_px=None, min_std=None):
    """Share of grid patches on the part passing D15 (``radius_px`` / ``min_std``: the probe tracker's 11 px and
    band-passed 1 grey when given with the band-passed grey), and of the part's tracked features that passed D1."""
    if mask is None:
        return {"grid_patches": 0}
    r = geometry.FT_RADIUS_PX if radius_px is None else int(radius_px)
    threshold = geometry.D15_MIN_STD_GREY if min_std is None else float(min_std)
    rows = geometry.grid_positions(gray.shape[0], geometry.FT_GRID_SPACING_PX // 2, r)
    cols = geometry.grid_positions(gray.shape[1], geometry.FT_GRID_SPACING_PX // 2, r)
    on, passing = 0, 0
    for v in rows:
        for u in cols:
            if mask[v, u]:
                on += 1
                passing += float(gray[v - r:v + r + 1, u - r:u + r + 1].std()) >= threshold
    tracked = [f for f in result.features if classes_by_fid.get(f.fid) == cls]
    return {"grid_patches": on, "grid_pass_D15": passing, "grid_pass_rate": passing / on if on else None,
            "features_on_part": len(tracked), "features_valid_D1": sum(f.valid for f in tracked),
            "rejects": {k: sum(f.reject == k for f in tracked) for k in ("shift_too_small", "residual", "lost",
                                                                           "legs")}}


def edge_split(rows, mask, cls):
    """On-part sample errors (visual, mm) of edge features (within EDGE_PX of the mask border) and interior ones."""
    if mask is None:
        return {}
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    edge, interior = [], []
    for r in rows:
        if r["class"] != cls or not truth_ok(r):
            continue
        u, v = int(round(r["uv"][0])), int(round(r["uv"][1]))
        if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]):
            continue
        (edge if distance[v, u] <= EDGE_PX else interior).append(1000.0 * (r["z"] - r["visual_z"]))
    return {"edge_err_mm": _stats(edge), "interior_err_mm": _stats(interior), "edge_values_mm": edge,
            "interior_values_mm": interior}


# ----------------------------------------------------------------------------------------------- calibration
def _recorder_kwargs(**extra):
    """The keyword arguments the current ``geometry.SurveyRecorder`` takes (an older script that swaps its own
    recorder class in may not take the review-1 ones)."""
    try:
        params = inspect.signature(geometry.SurveyRecorder).parameters
    except (TypeError, ValueError):
        return {}
    return {key: value for key, value in extra.items() if key in params}


def calibrate(driver, robot):
    """LEGACY Calibrate+-like start (open hold, a free-air close and open for the fingertip band, then the v1 probe
    moves and the +-4 cm vertical probe) recorded by a probe-tracker ``SurveyRecorder`` with the grid only (decision
    §2.3 G-1: Calibrate+ uses probe-tracker mode). Kept for the review-1 lens scripts that call it after their own
    BM CalibratePlus; the pipeline path is ``calibrate_plus``."""
    start_q = driver.view.quaternion_xyzw.copy()
    driver.run(HoldPrimitive(driver.view, steps=OPEN_HOLD_STEPS, gripper="open"), "open_hold")
    open_rgb = driver.view.wrist_rgb_upright
    driver.run(HoldPrimitive(driver.view, steps=CLOSE_STEPS, gripper="close"), "close_2")
    closed_rgb = driver.view.wrist_rgb_upright
    driver.run(HoldPrimitive(driver.view, steps=OPEN_STEPS, gripper="open"), "open_2")
    band = geometry.fingertip_band_mask(open_rgb, closed_rgb)
    recorder = geometry.SurveyRecorder(driver.view, robot, camera=None, external=True, exclude_mask=band,
                                       start_step=driver.steps, **_recorder_kwargs(probe_tracker=True))
    start = driver.view.position.copy()
    outcomes = []
    for index, goal in enumerate(np.cumsum(PROBE_MOVES, axis=0)):
        outcomes.append(driver.run(WaypointReach(driver.view, start + goal - driver.view.position, gripper="open"),
                                   f"probe_{index + 1}", [recorder]))
    outcomes.append(driver.run(AnchoredMove(driver.view, axis=(0, 0, 1), travel_m=VERTICAL_PROBE_M, gripper="open"),
                               "vertical_up", [recorder]))
    travel = max(float(driver.view.position[2] - start[2]), 0.001)
    outcomes.append(driver.run(AnchoredMove(driver.view, axis=(0, 0, -1), travel_m=travel, gripper="open"),
                               "vertical_down", [recorder]))
    driver.run(HoldPrimitive(driver.view, steps=SETTLE_STEPS, gripper="open", until_quiet=True,
                             orientation_xyzw=start_q), "settle", [recorder])
    loom_result = recorder.result(None)
    camera = geometry.CameraModel.decide(loom_result.loom["f_hat"], r2f_granted=True)
    result = recorder.result(camera)
    external = geometry.ExternalModel.fit(result.external_samples)
    locator = geometry.HandLocator.fit(result.external_samples, external, robot) if external else None
    return {"recorder": recorder, "loom_result": loom_result, "result": result, "camera": camera,
            "external": external, "locator": locator, "band": band, "start_q": start_q, "outcomes": outcomes}


def _indexed_frames(frames, prefix):
    keyed = []
    for key, frame in frames.items():
        if key.startswith(prefix + "[") and key.endswith("]"):
            keyed.append((int(key[len(prefix) + 1:-1]), frame))
    return [frame for _, frame in sorted(keyed, key=lambda item: item[0])]


def calibrate_plus(driver, robot, *, grid_at_reference=True):
    """BR's calibrate_plus restated (contract §C10, BR is Wave 2): BM's ``CalibratePlus`` (gripper curves and the
    probe moves, the vertical probe in two 2 cm halves per direction, M-2) with a probe-tracker ``SurveyRecorder``
    (grid only, the external camera; decision §2.3 G-1), ``gripper_quantities`` (G3, G5, G6), result(None) ->
    ``CameraModel.decide`` (R-2f granted) -> result(camera); ``ExternalModel``; ``HandLocator`` (G-5; ``degraded``
    -> outcome calibration_degraded); ``ExternalProjector`` on the locator's anchor; fingertip band (open_hold vs the
    end of close_2), false-change maps and static grey noise from the recorder's frames; frame alignment; the
    ``V2Calibration`` the scene starts from. The recorder places its grid on the reference frame (the frame before
    probe_1), outside the fingertip band of its own open-hold / close_2 frames (``grid_at_reference``, the BG builder's
    addition, not in the decision: review R10 asks for an A/B before BR adopts it; ``--calib-grid construction``
    places the grid on the construction view instead, as decision §2.3 G-1 states it)."""
    from gpt_field import agent_speedup_v2_primitives as primitives
    start_q = driver.view.quaternion_xyzw.copy()
    recorder = geometry.SurveyRecorder(driver.view, robot, camera=None, external=True, start_step=driver.steps,
                                       **_recorder_kwargs(probe_tracker=True, grid_at_reference=grid_at_reference))
    primitive = primitives.CalibratePlus(driver.view, recorder=recorder, robot=robot)
    began = time.perf_counter()
    outcome = driver.run(primitive, "calibrate_plus")
    record = primitive.record()
    quantities = primitives.gripper_quantities(record)
    g5 = primitives.g5_present_threshold(record.gripper, quantities["sigma_close_mm"])
    g6 = primitives.g6_hold_floor(record.gripper, quantities["sigma_close_mm"])
    t_result = time.perf_counter()
    loom_result = recorder.result(None)
    camera = geometry.CameraModel.decide(loom_result.loom["f_hat"], r2f_granted=True)
    result = recorder.result(camera)
    result_s = time.perf_counter() - t_result
    external = geometry.ExternalModel.fit(result.external_samples)
    locator = geometry.HandLocator.fit(result.external_samples, external, robot) if external else None
    anchor = locator.anchor() if locator is not None else None
    projector = None if anchor is None else geometry.ExternalProjector(external, *anchor)
    frames = result.frames
    band = geometry.fingertip_band_mask(frames["open"], frames["closed"])
    statics = _indexed_frames(frames, "static")
    false_wrist, sigma_wrist = geometry.false_change_map(statics) if statics else (None, 0.0)
    ext_statics = _indexed_frames(frames, "ext_static")
    false_ext, sigma_ext = geometry.false_change_map(ext_statics) if ext_statics else (None, sigma_wrist)
    view = driver.view
    turns = geometry.frame_alignment_turns(view.R, external)
    thresholds = list(quantities["thresholds"]) + [g5, g6] + list(camera.thresholds)
    if external is not None:
        thresholds += list(external.thresholds)
    if locator is not None:
        thresholds += list(locator.thresholds)
    cal = V2Calibration(
        gripper=record.gripper, sigma_close_mm=quantities["sigma_close_mm"],
        sigma_g_mm_per_step=quantities["sigma_g_mm_per_step"],
        free_yaw_rate_rad_per_step=quantities["free_yaw_rate_rad_per_step"], camera=camera, external=external,
        locator=locator, projector=projector, fingertip_band=band,
        false_change={"wrist": false_wrist, "external": false_ext},
        sigma_px={"wrist": sigma_wrist, "external": sigma_ext}, sigma_static_grey=sigma_wrist, quarter_turns=turns,
        start_pose=tuple(float(v) for v in view.state), start_quaternion=tuple(float(v) for v in start_q),
        survey=result, mount={}, thresholds=thresholds)
    degraded = external is None or locator is None or bool(getattr(locator, "degraded", False))
    return {"recorder": recorder, "loom_result": loom_result, "result": result, "camera": camera,
            "external": external, "locator": locator, "projector": projector, "band": band, "start_q": start_q,
            "record": record, "quantities": quantities, "g3": quantities["thresholds"][2], "g5": g5, "g6": g6,
            "v2cal": cal, "outcome": outcome, "turns": turns,
            "outcome_reason": "calibration_degraded" if degraded else outcome.get("reason"),
            "compute": {"wall_s": round(time.perf_counter() - began, 2), "result_s": round(result_s, 3),
                        "features": len(recorder.tracker.features), "samples": len(result.samples),
                        "grid": jsonable(getattr(recorder, "grid_info", None)),
                        "grid_at_reference": bool(grid_at_reference)}}


def score_locator(sim, cal, robot):
    """HandLocator start tips against the ORACLE grip projection (settled move starts). Review finding R9: a move the
    locator itself refuses (``HandLocator.motion_check`` on J dp: across v_hat, or too short along it) is listed but
    not scored; ``within_6px`` reads the locatable moves only (``max_error_px_all`` keeps every row)."""
    rows = []
    external, locator = cal["external"], cal["locator"]
    if external is None:
        return {"available": False}
    states = [s["start_state8"] for s in cal["result"].external_samples]
    truth = sim.oracle({"kind": "project", "camera": "external", "xyz": [st[:3] for st in states]})["uv"]
    for sample, uv_truth in zip(cal["result"].external_samples, truth):
        tip = sample.get("mask_extreme_uv")
        refusal = None
        if sample.get("end_state8") is not None:
            motion = external.J @ (np.asarray(sample["end_state8"][:3], float) - np.asarray(sample["start_state8"][:3],
                                                                                              float))
            refusal = geometry.HandLocator.motion_check(external, motion)
        if tip is None or uv_truth is None:
            rows.append({"stage": sample["stage"], "located": None, "locatable": refusal is None,
                         "refusal": refusal})
            continue
        pose = HandPose.from_state8(sample["start_state8"])
        uv = np.asarray(tip) - geometry.fingertip_offset_px(external, robot, pose)
        rows.append({"stage": sample["stage"], "located": uv.tolist(), "truth": uv_truth,
                     "error_px": float(np.linalg.norm(uv - np.asarray(uv_truth))),
                     "shift_source": sample.get("shift_source"), "locatable": refusal is None, "refusal": refusal})
    errors = [r["error_px"] for r in rows if r.get("error_px") is not None and r["locatable"]]
    errors_all = [r["error_px"] for r in rows if r.get("error_px") is not None]
    return {"available": locator is not None, "sigma_loc_px": None if locator is None else locator.sigma_loc_px,
            "sigma_consistency_px": None if locator is None else getattr(locator, "sigma_consistency_px", None),
            "model_sigma_px": geometry.L1_MODEL_SIGMA_PX,
            "rows": rows, "locatable_moves": len(errors), "max_error_px": max(errors) if errors else None,
            "max_error_px_all": max(errors_all) if errors_all else None,
            "within_6px": bool(errors) and max(errors) <= LOCATOR_TOL_PX}


def external_truth(sim, cal):
    """The external Jacobian at the start grip by finite differences of the ORACLE projection (scoring only)."""
    grip = np.asarray(cal["result"].pose0[:3], dtype=float)
    step = 0.01
    points = [grip] + [grip + step * e for e in np.eye(3)]
    uv = np.asarray(sim.oracle({"kind": "project", "camera": "external", "xyz": [p.tolist() for p in points]})["uv"])
    J = np.stack([(uv[k + 1] - uv[0]) / step for k in range(3)], axis=1)
    _, _, vt = np.linalg.svd(J)
    ray = vt[-1] if vt[-1][2] < 0 else -vt[-1]
    return J, ray


# ----------------------------------------------------------------------------------------------- hovers
def hover_measurement(rows, cls, reference=None):
    """One hover's part measurement from its valid on-part samples (BG-R2): ``top_z`` (95th percentile, the spec
    §4.5 registration rule) and the median; ``measured`` only with at least HOVER_CLASS_MIN_SAMPLES on-part samples.
    ``reference`` {"visual_top_z", "collision_top_z", "source"} (the part's fixed truth top) gives the G1 errors
    ``err_top_visual_mm`` / ``err_top_collision_mm``. The earlier statistic (truth sampled at the measured xy,
    robot-owned truths dropped, BG-R1) is kept beside it as ``err_top_visual_at_xy_mm``. None without on-part
    samples."""
    on_part = [r for r in rows if r["class"] == cls]
    if not on_part:
        return None
    z = np.array([r["z"] for r in on_part])
    usable = [r for r in on_part if truth_ok(r)]
    visual = np.array([r["visual_z"] for r in usable])
    collisions = np.array([r["collision_z"] for r in on_part if r["collision_z"] is not None])
    top = float(np.percentile(z, TOP_PERCENTILE))
    out = {"n": len(on_part), "n_truth_usable": len(usable), "n_truth_robot": sum(
               r.get("visual_z") is not None and not truth_ok(r) for r in on_part),
           "measured": len(on_part) >= HOVER_CLASS_MIN_SAMPLES, "top_z": top, "z": float(np.median(z)),
           "sigma_z_median_mm": float(1000.0 * np.median([r["sigma_z"] for r in on_part])),
           "reference": reference, "err_top_visual_mm": None, "err_top_collision_mm": None}
    if reference is not None and reference.get("visual_top_z") is not None:
        out["err_top_visual_mm"] = 1000.0 * (top - float(reference["visual_top_z"]))
    if reference is not None and reference.get("collision_top_z") is not None:
        out["err_top_collision_mm"] = 1000.0 * (top - float(reference["collision_top_z"]))
    if visual.size:
        visual_top = float(np.percentile(visual, TOP_PERCENTILE))
        out.update(visual_top_at_xy_z=visual_top, err_top_visual_at_xy_mm=1000.0 * (top - visual_top),
                   visual_z_at_xy=float(np.median(visual)),
                   err_median_visual_at_xy_mm=1000.0 * (float(np.median(z)) - float(np.median(visual))))
    if collisions.size:
        out.update(collision_top_at_xy_z=float(np.percentile(collisions, TOP_PERCENTILE)),
                   err_median_collision_at_xy_mm=1000.0 * (float(np.median(z)) - float(np.median(collisions))))
    return out


def hover_target(answers, robot, pid, offset, view, hover_mode):
    """LEGACY / DIAGNOSIS (ruling 4d ended it for the gates; kept for ``--hover seeded`` and the review-1 lens
    scripts): BP's ``planner.hover_for`` on a seed PartRecord whose point is the ORACLE anchor + offset (sigma_h 0,
    empty HeightField); "camera" = the same hover with the grip moved by -WRIST_CAMERA_ALONG_R0_M R0."""
    from gpt_field import agent_speedup_v2_planner as planner
    truth = answers.truth_points()[pid]
    point = np.array([truth[0] + offset[0], truth[1] + offset[1], truth[2]])
    record = PartRecord(pid, PART_CLASS[pid], PART_MOVE[pid], frozenset({"grip"}), xyz=point, sigma_h_m=0.0)
    hover = planner.hover_for(record, HeightField.empty(point[:2]), robot, HandPose.from_state8(view.state))
    if hover.get("hover_grip_xyz") is None:
        raise RuntimeError(f"hover_for refused part {pid}: {hover.get('reason')}")
    target = np.asarray(hover["hover_grip_xyz"], dtype=float)
    if hover_mode == "camera":
        target = target - robot_facts.WRIST_CAMERA_ALONG_R0_M * view.R[:, 0]
    summary = {k: hover.get(k) for k in ("clearance_m", "fingertip_z", "z_ref", "raised_m", "reason")}
    summary["thresholds"] = [t.to_dict() for t in hover.get("thresholds") or [] if hasattr(t, "to_dict")]
    return target, summary


class _ViewLog:
    """A recorder that keeps every after-view (the HandLocator sweep)."""

    def __init__(self):
        self.views = []

    def on_transition(self, stage, before, after):
        self.views.append(after)


def locate_at_hover(driver, sim, cal, robot):
    """HandLocator at the hover (BG-R3): quiet settle, then a HOVER_LOCATE_M vertical move (along v_hat, the move
    the locator is validated on); located on (settled frame, end of the move) against the ORACLE projection of the
    settled grip. The 1-step pair (settled frame, first step) must be refused by the locator's preconditions. A
    diagnostic sweep gives the raw tip error (preconditions bypassed) against the along-v_hat image motion of every
    step of the move."""
    locator, external = cal["locator"], cal["external"]
    if locator is None or external is None:
        return None
    driver.run(HoldPrimitive(driver.view, steps=SETTLE_STEPS, gripper="open", until_quiet=True,
                             orientation_xyzw=cal["start_q"]), "settle")
    settled = driver.view
    pose0 = HandPose.from_state8(settled.state)
    log = _ViewLog()
    driver.run(AnchoredMove(settled, axis=(0, 0, 1), travel_m=HOVER_LOCATE_M, gripper="open"), "locate_up", [log])
    truth = sim.oracle({"kind": "project", "camera": "external", "xyz": [list(map(float, settled.state[:3]))]})["uv"][0]
    end = driver.view
    uv, sigma = locator.locate(settled.external_rgb_upright, end.external_rgb_upright, pose0,
                               after_pose=HandPose.from_state8(end.state))
    info = dict(locator.last_locate or {})
    one_uv, _ = locator.locate(settled.external_rgb_upright, log.views[0].external_rgb_upright, pose0,
                               after_pose=HandPose.from_state8(log.views[0].state)) if log.views else (None, None)
    one_info = dict(locator.last_locate or {}) if log.views else None
    band_px = float(np.linalg.norm(external.J @ np.array([0.0, 0.0, geometry.HL_HAND_BAND_M])))
    sweep = []
    for k, view in enumerate(log.views):
        motion = external.J @ (view.position - settled.position)
        tip = geometry.hand_tip_before(settled.external_rgb_upright, view.external_rgb_upright, motion,
                                       external.v_hat, band_px=band_px)
        located = None if tip is None else tip - geometry.fingertip_offset_px(external, robot, pose0)
        sweep.append({"step": k + 1, "along_px": float(motion @ external.v_hat),
                      "across_px": float(motion @ external.n_hat),
                      "error_px": None if located is None or truth is None else
                      float(np.linalg.norm(located - np.asarray(truth)))})
    yaw = None
    if cal.get("calibration_q") is not None:
        yaw = world_z_yaw_deg(cal["calibration_q"], settled.quaternion_xyzw)
    v0 = info.get("v0_uv")
    return {"located": None if uv is None else uv.tolist(), "truth": truth, "sigma_px": sigma,
            "error_px": None if uv is None or truth is None else float(np.linalg.norm(np.asarray(uv) - truth)),
            "v0_error_px": None if v0 is None or truth is None else float(np.linalg.norm(np.asarray(v0) - truth)),
            "yaw_from_calibration_deg": yaw,
            "locate": info, "one_step": {"located": None if one_uv is None else one_uv.tolist(), "locate": one_info},
            "sweep": sweep}


# ----------------------------------------------------------------------------------------------- truth geometry
def _owns(spec, hit, parts):
    """Whether a collision hit belongs to the part (scoring only): its region geoms, else the whole owner minus the
    other parts' regions of that owner, else its body."""
    if hit is None:
        return False
    region = set(spec.get("region") or [])
    if region:
        return hit.get("geom") in region
    if spec.get("whole"):
        excluded = set()
        for other in parts.values():
            if other is not spec and other.get("owner") == spec.get("owner"):
                excluded |= set(other.get("region") or [])
        return hit.get("owner") == spec.get("owner") and hit.get("geom") not in excluded
    return hit.get("body") == spec.get("body")


def _canonical_axis(axis):
    axis = np.asarray(axis, dtype=float)[:2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    return -axis if (axis[0] < 0.0 or (abs(axis[0]) < 1e-12 and axis[1] < 0.0)) else axis


def _axis_angle_deg(a, b):
    a, b = _canonical_axis(a), _canonical_axis(b)
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def turn_about_world_z(q_xyzw, angle_rad):
    """The orientation q (xyzw) turned by ``angle_rad`` about world +z (left-multiplied)."""
    half = 0.5 * float(angle_rad)
    x1, y1, z1, w1 = 0.0, 0.0, math.sin(half), math.cos(half)
    x2, y2, z2, w2 = (float(v) for v in q_xyzw)
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def _quaternion_matrix(q_xyzw):
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def world_z_yaw_deg(q_from_xyzw, q_to_xyzw):
    """The hand's turn about world +z (deg, in [-180, 180)) from one orientation to another: the jaw (R1) direction's
    world-xy angle change (a top-down hand; ``turn_about_world_z`` by t gives t)."""
    a, b = _quaternion_matrix(q_from_xyzw)[:2, 1], _quaternion_matrix(q_to_xyzw)[:2, 1]
    turn = math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0]))
    return (turn + 180.0) % 360.0 - 180.0


def strip_widths_mm(cells_xy, centre_xy, half_strip_m, step_m):
    """The scene's strip rule on ORACLE collision cells (scoring only): per jaw yaw of the YAW_GRID_RAD grid, the
    extent along the jaw of the cells within ``half_strip_m`` (half a pad length) of the jaw line through
    ``centre_xy``, + one grid step; {yaw_deg: mm} (a yaw whose strip holds no cell is absent)."""
    rel = np.asarray(cells_xy, dtype=float).reshape(-1, 2) - np.asarray(centre_xy, dtype=float)[:2]
    out = {}
    for k in range(int(round(math.pi / YAW_GRID_RAD))):
        yaw = k * YAW_GRID_RAD
        jaw = np.array([math.cos(yaw), math.sin(yaw)])
        normal = np.array([-jaw[1], jaw[0]])
        near = np.abs(rel @ normal) <= half_strip_m
        if near.any():
            along = rel[near] @ jaw
            out[int(round(math.degrees(yaw)))] = 1000.0 * (float(along.max() - along.min()) + step_m)
    return out


def _truth_grid_steps(half_m, step_m):
    return np.arange(-half_m, half_m + 1e-9, step_m)


def grid_cells_xy(grid):
    """The part's ORACLE collision cells (world xy, scoring only) from a ``cells_grid`` record (``part_geometry_truth``:
    the anchor, the grid half size and step, and the packed membership bits of the heights grid)."""
    steps = _truth_grid_steps(float(grid["half_m"]), float(grid["step_m"]))
    n = len(steps)
    bits = np.frombuffer(base64.b64decode(grid["bits_b64"]), dtype=np.uint8)
    mine = np.unpackbits(bits)[:n * n].reshape(n, n).astype(bool)
    anchor = np.asarray(grid["anchor_xy"], dtype=float)
    xy = np.stack(np.meshgrid(anchor[0] + steps, anchor[1] + steps), axis=-1)     # [row = dy, col = dx]
    return xy[mine]


def width_truth_at(geo, xyz):
    """G1'(e) width truth (decision 2 §5 ruling: the collision cross-section at the grip; for a non-prismatic part at
    the RECORD's grip point, not the ORACLE anchor): the narrowest strip width of the part's collision cells around
    ``xyz`` (``strip_widths_mm`` with the episode's own half strip and grid step). Returns (mm, at): at "grip_point";
    "anchor_fallback" when no part cell lies within the half strip (half a pad length, R4) of the grip point, so pads
    closing there would touch none of the part (an off-part record, e.g. a wrong-object registration, keeps the strict
    anchor truth and is still scored); "anchor" for a run recorded without the cell grid (older runs, the judge's)."""
    geo = geo or {}
    anchor = geo.get("w_min_mm")
    grid = geo.get("cells_grid")
    if not grid or xyz is None:
        return anchor, "anchor"
    cells = grid_cells_xy(grid)
    half_strip = float(grid["half_strip_m"])
    centre = np.asarray(xyz, dtype=float)[:2]
    if not len(cells) or float(np.min(np.linalg.norm(cells - centre, axis=1))) > half_strip:
        return anchor, "anchor_fallback"
    widths = strip_widths_mm(cells, centre, half_strip, float(grid["step_m"]))
    return min(widths.values()), "grip_point"


def part_geometry_truth(sim, answers, pid, robot):
    """ORACLE geometry of a part (scoring only), from a heights grid (TRUTH_GRID_STEP_M) within TRUTH_GRID_HALF_M
    of its anchor: the part's collision cells, the principal axis (elongated parts), the width per jaw yaw by the
    scene's strip rule (extent along the jaw of the cells within half a pad length of the anchor, + one grid step),
    the cross width (narrowest yaw) and the support under it (median of the non-part cells SUPPORT_RING_M outside
    the part and SUPPORT_BELOW_TOP_M below its top). ``cells_grid`` keeps the cell membership (packed bits) so the
    scorer takes the width truth at each record's own grip point (``width_truth_at``; decision 2 §5)."""
    spec = answers.parts[pid]
    anchor = np.asarray(answers.truth_points()[pid], dtype=float)
    half, step = TRUTH_GRID_WIDE.get(pid, (TRUTH_GRID_HALF_M, TRUTH_GRID_STEP_M))
    steps = _truth_grid_steps(half, step)
    n = len(steps)
    points = [(float(anchor[0] + dx), float(anchor[1] + dy)) for dy in steps for dx in steps]
    zs, hits = [], []
    for i in range(0, len(points), 3000):
        reply = sim.oracle({"kind": "heights", "xy": [list(p) for p in points[i:i + 3000]]})
        zs += reply["z"]
        hits += reply["hit"]
    z = np.array([np.nan if v is None else float(v) for v in zs]).reshape(n, n)
    mine = np.array([_owns(spec, hit, answers.parts) for hit in hits], dtype=bool).reshape(n, n)
    xy = np.array(points, dtype=float).reshape(n, n, 2)
    cells = xy[mine]
    out = {"anchor": anchor.tolist(), "cells": int(mine.sum()), "axis": None, "widths_mm": {}, "w_min_mm": None,
           "w_min_yaw_deg": None, "support_z": None, "collision_top_max": None, "collision_top_p95": None}
    if len(cells) < 3:
        return out
    cz = z[mine]
    out["collision_top_max"], out["collision_top_p95"] = float(np.nanmax(cz)), float(np.nanpercentile(cz, 95))
    if pid in ELONGATED:
        centred = cells - cells.mean(axis=0)
        values, vectors = np.linalg.eigh(centred.T @ centred)
        out["axis"] = _canonical_axis(vectors[:, 1]).tolist()
        out["aspect"] = float(math.sqrt(max(values[1], 0.0) / max(values[0], 1e-12)))
    half_strip = robot.pad_length_m / 2.0
    out["widths_mm"] = strip_widths_mm(cells, anchor[:2], half_strip, step)
    out["cells_grid"] = {"anchor_xy": [float(anchor[0]), float(anchor[1])], "half_m": float(half),
                         "step_m": float(step), "half_strip_m": float(half_strip),
                         "bits_b64": base64.b64encode(np.packbits(mine.ravel())).decode("ascii")}
    if out["widths_mm"]:
        yaw_deg = min(out["widths_mm"], key=out["widths_mm"].get)
        out["w_min_mm"], out["w_min_yaw_deg"] = out["widths_mm"][yaw_deg], yaw_deg
    distance = cv2.distanceTransform((~mine).astype(np.uint8), cv2.DIST_L2, 3) * step
    ring = (~mine) & (distance >= SUPPORT_RING_M[0]) & (distance <= SUPPORT_RING_M[1]) & np.isfinite(z) & \
        (z < out["collision_top_max"] - SUPPORT_BELOW_TOP_M)
    if ring.any():
        out["support_z"] = float(np.median(z[ring]))
        out["support_ring_cells"] = int(ring.sum())
    return out


# ----------------------------------------------------------------------------------------------- pipeline steps
def fine_point(answers, scene, driver, pid, turns):
    """The model's fine pointing, ORACLE stand-in: point_fine at the scene's predicted pixel (the 128 px crop
    centre); a 0 answer -> point_coarse on the current image -> point_fine at the coarse cell centre (spec §3.2).
    Returns (answered uv | None, info with the ORACLE truth pixel for scoring)."""
    pred = scene.predicted_uv(pid, driver.view)
    center = None if pred is None else [float(pred[0][0]), float(pred[0][1])]
    meta = {"pid": pid, "camera": "wrist", "quarter_turns": turns, "image_size": 512, "crop_px": FINE_CROP_PX}
    if center is not None:
        meta["center_uv"] = center
    answers.answer("point_fine", meta)
    rec = answers.records[-1]
    info = {"center_uv": center, "answer": rec["answer"], "answered_uv": rec.get("answered_uv"),
            "truth_uv": rec.get("truth_uv"), "truth_xyz": rec.get("truth_xyz"), "error_px": rec.get("error_px"),
            "rule": rec.get("answer_rule"), "calls": 1}
    uv = rec.get("answered_uv")
    if uv is None:
        answers.answer("point_coarse", {"pid": pid, "camera": "wrist", "quarter_turns": turns, "image_size": 512})
        crec = answers.records[-1]
        info["coarse"] = {k: crec.get(k) for k in ("answer", "answered_uv", "truth_uv", "error_px")}
        info["calls"] += 1
        centre = crec.get("answered_uv")
        if centre is None:
            return None, info
        meta["center_uv"] = list(map(float, centre))
        answers.answer("point_fine", meta)
        rec = answers.records[-1]
        info["calls"] += 1
        info["second_fine"] = {"answer": rec["answer"], "answered_uv": rec.get("answered_uv"),
                               "error_px": rec.get("error_px"), "rule": rec.get("answer_rule")}
        info["truth_uv"] = rec.get("truth_uv") or info["truth_uv"]
        uv = rec.get("answered_uv")
    return uv, info


def repoint_fine(answers, scene, driver, pid, turns):
    """Lead ruling 3 (r2close), BR's re-point restated with the ORACLE stand-in: the ONE re-pointing at the hover
    after an S-15 ``seed_outside_region`` refusal. Fine pointing centred on the scene's predicted pixel
    (``fine_point``: a 0 answer -> point_coarse -> point_fine, spec §3.2); coarse first (point_coarse, then
    point_fine at the coarse cell centre) when the prediction's sigma exceeds COARSE_FIRST_SIGMA_PX (rulings item 9)
    or there is no prediction. Returns (answered uv | None, info with the ORACLE truth pixel for scoring, the
    prediction's sigma and whether coarse went first)."""
    pred = scene.predicted_uv(pid, driver.view)
    sigma = None if pred is None else float(pred[1])
    if pred is not None and sigma <= COARSE_FIRST_SIGMA_PX:
        uv, info = fine_point(answers, scene, driver, pid, turns)
        info.update(predicted_sigma_px=sigma, coarse_first=False)
        return uv, info
    answers.answer("point_coarse", {"pid": pid, "camera": "wrist", "quarter_turns": turns, "image_size": 512})
    crec = answers.records[-1]
    info = {"center_uv": None, "answer": None, "answered_uv": None, "truth_uv": crec.get("truth_uv"),
            "truth_xyz": crec.get("truth_xyz"), "error_px": None, "rule": None, "calls": 1,
            "coarse": {k: crec.get(k) for k in ("answer", "answered_uv", "truth_uv", "error_px")},
            "predicted_sigma_px": sigma, "coarse_first": True}
    centre = crec.get("answered_uv")
    if centre is None:
        return None, info
    meta = {"pid": pid, "camera": "wrist", "quarter_turns": turns, "image_size": 512, "crop_px": FINE_CROP_PX,
            "center_uv": [float(v) for v in centre]}
    answers.answer("point_fine", meta)
    rec = answers.records[-1]
    info.update(center_uv=meta["center_uv"], answer=rec["answer"], answered_uv=rec.get("answered_uv"),
                truth_uv=rec.get("truth_uv") or info["truth_uv"], truth_xyz=rec.get("truth_xyz") or info["truth_xyz"],
                error_px=rec.get("error_px"), rule=rec.get("answer_rule"), calls=2)
    return rec.get("answered_uv"), info


def seed_outside_region(rec):
    """Whether a registration result is the S-15 refusal (part_not_found "seed_outside_region")."""
    return (not isinstance(rec, PartRecord) and getattr(rec, "reason", None) == "part_not_found"
            and (getattr(rec, "evidence", None) or {}).get("why") == "seed_outside_region")


def repoint_after_seed_refusal(driver, sim, answers, scene, pid, failure, turns):
    """Lead ruling 3 (r2close): after an S-15 refusal at a grip part's hover stage, re-point ONCE (``repoint_fine``)
    in the current view and register once more from that seed on the same surveyed view (no new probe); a second
    refusal ends the stage. Returns (registration result, the new Seed | None, the logged ``repoint`` dict: the
    refusal's gap / limit / excess, the pointing, the re-pointed seed's ORACLE truth pixel (scoring only) and the
    outcome)."""
    evidence = getattr(failure, "evidence", None) or {}
    uv, pointing = repoint_fine(answers, scene, driver, pid, turns)
    out = {"refusal": {k: evidence.get(k) for k in ("gap_px", "limit_px", "q_now_px", "sigma_k", "excess_px",
                                                    "seed_uv", "nearest_uv")},
           "pointing": pointing, "seed_uv": None if uv is None else [float(v) for v in uv], "result": None}
    if uv is None:
        out["result"] = "part_not_found_at_hover"
        return failure, None, out
    seed = Seed("wrist", tuple(float(v) for v in uv), driver.steps, "fine", FINE_QUANT_PX,
                tuple(float(v) for v in driver.view.state))
    out["truth_uv_seed"], out["truth_uv_source"] = seed_truth_uv(sim, pointing, driver.view)
    began = time.perf_counter()
    rec = scene.register(pid, driver.view, seed=seed, empty_hand=True)
    out["register_s"] = time.perf_counter() - began
    out["result"] = "registered" if isinstance(rec, PartRecord) else "registration_failed"
    if not isinstance(rec, PartRecord):
        out["failure"] = {"reason": getattr(rec, "reason", None),
                          "evidence": jsonable(getattr(rec, "evidence", None))}
    return rec, seed, out


class TimedRecorder:
    """Forwards on_transition to a SurveyRecorder and times it (RS-8: ms per tracked frame). The new view's wrist
    image is decoded first, outside the tracker's time (``decode_seconds``): the decode is the observation's cost,
    paid once per step by the first consumer that reads the grey image (the scene's ``on_view`` only caches the
    view, so in the pipeline that is the tracker too), not the tracker's compute that G-3 bounds. Design review 2
    ruling (decision §5): the decode is excluded from G-3's 40 ms bound, reported separately here and counted in
    AC-4's control_without_video_s."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.seconds = 0.0
        self.decode_seconds = 0.0
        self.frames = 0

    def on_transition(self, stage, before, after):
        began = time.perf_counter()
        _ = after.wrist_gray_upright
        decoded = time.perf_counter()
        self.recorder.on_transition(stage, before, after)
        self.seconds += time.perf_counter() - decoded
        self.decode_seconds += decoded - began
        self.frames += 1


def reach_hover(driver, target, quat, hf, robot, *, keepouts=(), segment_m=0.07):
    """BR free_move restated: ``planner.free_path`` waypoints over the HeightField, chained as v1 WaypointReach
    segments of at most ``segment_m`` per axis (the +-8 cm grid, M2), orientation held; then a B4 settle."""
    from gpt_field import agent_speedup_v2_planner as planner
    from gpt_field import agent_speedup_v2_primitives as primitives
    target = np.asarray(target, dtype=float)
    path = planner.free_path(driver.view.position.copy(), target, hf, keepouts, robot, HandPose.from_view(driver.view))
    reasons = []
    for waypoint in [np.asarray(w, dtype=float) for w in path["waypoints"]]:
        for _ in range(12):
            offset = waypoint - driver.view.position
            if float(np.max(np.abs(offset))) < 0.004:
                break
            count = max(1, int(math.ceil(float(np.max(np.abs(offset))) / segment_m)))
            out = driver.run(WaypointReach(driver.view, offset / count, gripper="open", orientation_xyzw=quat), "reach")
            reasons.append(out["reason"])
            if out["reason"] != "goal_reached":
                break
    driver.run(primitives.b4_settle(driver.view, gripper="open", orientation_xyzw=quat), "settle")
    return {"waypoints": [list(map(float, w)) for w in path["waypoints"]], "transit_z": path.get("transit_z"),
            "reasons": reasons, "residual_m": float(np.linalg.norm(target - driver.view.position)),
            "end": driver.view.position.tolist()}


P11_CAP_RESTATED_M = 0.10                   # decision §3 P11_CAP_M (BP owns it; used only while BP lacks P-E)


def _visibility_shift(record, target, pose, f_px, band_top_row):
    """Decision §2.4 P-E / remedy 1 P-2 restated: the part's predicted wrist row at the hover (the current
    orientation held) stays above the fingertip band's top row by at least kappa D2 + FT_PROBE_RADIUS_PX (the r0_out
    probe leg moves it down by kappa D2); else the hover moves along -R0 (horizontal) by the excess / kappa. Returns
    (target, shift_m)."""
    target = np.asarray(target, dtype=float).copy()
    if band_top_row is None or record.xyz is None:
        return target, 0.0
    R = np.asarray(pose.R, dtype=float)
    r0 = R[:, 0].copy()
    r0[2] = 0.0
    if float(np.linalg.norm(r0)) < 1e-6:
        return target, 0.0
    r0 /= float(np.linalg.norm(r0))
    camera_offset = np.asarray(pose.camera, dtype=float) - np.asarray(pose.grip, dtype=float)
    centre_v = geometry.principal_point()[1]
    total = 0.0
    for _ in range(2):
        d = np.asarray(record.xyz, dtype=float) - (target + camera_offset)
        depth = float(d @ R[:, 2])
        if not depth > 0.0:
            break
        v = centre_v - f_px * float(d @ R[:, 0]) / depth
        kappa = f_px / depth
        excess = v - (float(band_top_row) - (kappa * HOVER_PROBE_M + geometry.FT_PROBE_RADIUS_PX))
        if excess <= 0.0:
            break
        target[:2] -= (excess / kappa) * r0[:2]
        total += excess / kappa
    return target, total


def chain_hover(record, hf, robot, pose, offset, *, f_px=None, band_top_row=None):
    """The chain hover for a measured record: BP ``hover_for`` plus the survey offset in xy. Decision §2.4 P-E (P11
    cap -> ``first_stage``; a registered record's own top as z_ref; the P-2 visibility rule) is BP's; while
    ``planner.P11_CAP_M`` is absent it is restated here (declared in the report as ``restated_p_e``)."""
    from gpt_field import agent_speedup_v2_planner as planner
    hover = planner.hover_for(record, hf, robot, pose)
    info = {k: hover.get(k) for k in ("hover_grip_xyz", "clearance_m", "fingertip_z", "z_ref", "raised_m", "reason",
                                      "sigma_h_m", "first_stage")}
    if hover.get("hover_grip_xyz") is None:
        return None, info
    target = np.asarray(hover["hover_grip_xyz"], dtype=float).copy()
    first_stage = bool(hover.get("first_stage"))
    restate = getattr(planner, "P11_CAP_M", None) is None
    if restate:
        sigma_h = float(hover.get("sigma_h_m") or 0.0)
        z_ref = float(hover["z_ref"])
        registered = record.status == "registered" and not record.provisional and record.top_z is not None
        if registered:
            z_ref = float(record.top_z.value)
        wanted = planner.P11_SIGMA_K * sigma_h + planner.P11_OFFSET_M
        clearance = max(planner.P11_FLOOR_M, min(wanted, P11_CAP_RESTATED_M))
        grip_z = z_ref + clearance + robot.t_f_m
        if float(hover.get("raised_m") or 0.0) > 0.0:
            grip_z = max(grip_z, float(target[2]))
        first_stage = wanted > P11_CAP_RESTATED_M
        info["restated_p_e"] = {"z_ref": z_ref, "clearance_m": clearance, "grip_z": grip_z,
                                "hover_for_grip_z": float(target[2]), "registered_top": registered}
        target[2] = grip_z
    target[:2] += np.asarray(offset, dtype=float)
    if restate and f_px is not None:
        target, shift = _visibility_shift(record, target, pose, f_px, band_top_row)
        info["restated_p_e"]["visibility_shift_m"] = shift
    info.update(first_stage=first_stage, target=target.tolist())
    return target, info


def _backproject_at_z(pose, uv, z, f_px):
    direction = geometry.pixel_direction(pose, uv, f_px)
    if abs(float(direction[2])) < 1e-9:
        return None
    t = (float(z) - float(pose.camera[2])) / float(direction[2])
    return pose.camera + t * direction


def lateral_errors(rec, geo, seed_pose, truth_uv, f_px):
    """Across-jaw grip errors (mm, scoring; G-yaw): the jaw closes across the truth axis of an elongated part (else
    the radial error is given). ``pipeline``: the record point; ``exact``: the exact truth pixel back-projected at
    the measured top_z (the S-2 lift with an exact seed); ``pm8`` / ``pm16``: the largest |error| of that pixel moved
    by +-8 / +-16 px along the jaw's image direction."""
    anchor = np.asarray(geo["anchor"][:2], dtype=float)
    out = {"pipeline_mm": None, "exact_mm": None}
    axis = geo.get("axis")
    jaw = None if axis is None else np.array([-axis[1], axis[0]])

    def error(xy):
        rel = np.asarray(xy[:2], dtype=float) - anchor
        return 1000.0 * (float(rel @ jaw) if jaw is not None else float(np.linalg.norm(rel)))
    if rec is not None and getattr(rec, "xyz", None) is not None:
        out["pipeline_mm"] = error(rec.xyz)
    if rec is None or getattr(rec, "top_z", None) is None or truth_uv is None or seed_pose is None:
        return out
    z = float(rec.top_z.value)
    point = _backproject_at_z(seed_pose, truth_uv, z, f_px)
    if point is None:
        return out
    out["exact_mm"] = error(point)
    if jaw is not None:
        image_dir = np.array([float(jaw @ seed_pose.R[:2, 1]), -float(jaw @ seed_pose.R[:2, 0])])
        image_dir = image_dir / max(float(np.linalg.norm(image_dir)), 1e-12)
        for magnitude in POINTING_ERRORS_PX:
            values = []
            for sign in (1.0, -1.0):
                moved = _backproject_at_z(seed_pose, np.asarray(truth_uv, float) + sign * magnitude * image_dir, z,
                                          f_px)
                if moved is not None:
                    values.append(abs(error(moved)))
            out[f"pm{int(magnitude)}_mm"] = max(values) if values else None
    return out


def _p6_limit_m(rec, robot):
    from gpt_field import agent_speedup_v2_planner as planner
    return planner._p6_sigma_limit(robot.p_min_m, -robot.pad_span_m[0], planner._is_fixture(rec))


def score_record(rec, geo, top_truth, robot, *, seed_pose=None, truth_uv=None, f_px=None, event=None):
    """What G1' / G-yaw read of one registration (truth per stage; the aggregate re-scores the top against the
    episode's fixed reference)."""
    out = {"type": type(rec).__name__}
    if not isinstance(rec, PartRecord):
        out.update(reason=getattr(rec, "reason", None), evidence=jsonable(getattr(rec, "evidence", None)))
        return out
    sigma_h = rec.sigma_h_m
    out.update(status=rec.status, provisional=bool(rec.provisional), partial=bool(rec.partial),
               registered=rec.status == "registered" and not rec.provisional,
               sigma_h_mm=None if sigma_h is None else 1000.0 * sigma_h,
               xyz=None if rec.xyz is None else [float(v) for v in rec.xyz],
               top_z=None if rec.top_z is None else float(rec.top_z.value),
               support_z=None if rec.support_z is None else float(rec.support_z.value),
               support_sigma_mm=None if rec.support_z is None else 1000.0 * float(rec.support_z.sigma),
               top_bound=getattr(rec, "top_bound", None), top_upper_z=getattr(rec, "top_upper_z", None),
               top_band=jsonable(getattr(rec, "top_band", None)),
               axis_check=None if getattr(rec, "axis_check", None) is None else jsonable(
                   {k: v for k, v in rec.axis_check.items()}),
               elongated=rec.elongation is not None, move_prior=rec.move_prior)
    try:
        out["p6_limit_mm"] = 1000.0 * float(_p6_limit_m(rec, robot))
    except Exception:  # noqa: BLE001 - scoring helper only
        out["p6_limit_mm"] = None
    out["width_unmeasured"] = rec.width_by_yaw is not None and len(rec.width_by_yaw) == 0     # K-5 / S-13
    if rec.width_by_yaw:
        widths = {int(round(math.degrees(yaw))): 1000.0 * float(w.value) for yaw, w in rec.width_by_yaw}
        sigmas = {int(round(math.degrees(yaw))): 1000.0 * float(w.sigma) for yaw, w in rec.width_by_yaw}
        out["widths_mm"] = widths
        out["widths_sigma_mm"] = sigmas
        narrowest = min(widths, key=widths.get)
        out["w_min_mm"] = widths[narrowest]
        out["w_min_sigma_mm"] = sigmas[narrowest]
        truth, truth_at = width_truth_at(geo, out["xyz"])
        if truth is not None:
            truth = float(truth)
            out["w_min_err_mm"] = out["w_min_mm"] - truth
            out["w_truth_at"] = truth_at
            out["width_ok"] = abs(out["w_min_err_mm"]) <= max(WIDTH_TOL_MM, WIDTH_TOL_REL * truth)
    if rec.elongation is not None:
        out["axis"] = _canonical_axis(rec.elongation[0]).tolist()
        out["coherence"] = float(rec.elongation[1])
        if geo.get("axis") is not None:
            out["axis_err_deg"] = _axis_angle_deg(rec.elongation[0], geo["axis"])
    if out["top_z"] is not None and top_truth:
        if top_truth.get("visual_top_z") is not None:
            out["top_err_visual_mm"] = 1000.0 * (out["top_z"] - float(top_truth["visual_top_z"]))
        if top_truth.get("collision_top_z") is not None:
            out["top_err_collision_mm"] = 1000.0 * (out["top_z"] - float(top_truth["collision_top_z"]))
    if geo.get("support_z") is not None and out["support_z"] is not None:
        out["support_err_mm"] = 1000.0 * (out["support_z"] - float(geo["support_z"]))
    if seed_pose is not None and f_px is not None:
        out["lateral"] = lateral_errors(rec, geo, seed_pose, truth_uv, f_px)
    if event:
        out["event"] = {k: jsonable(event.get(k)) for k in (
            "cut_fraction", "aspect", "coherence", "image_coherence", "mask_coherence", "d10", "top_bound",
            "axis_source", "seed_band", "top_band", "axis_check", "support_check", "dsw", "flag", "axis_check_view",
            "d10_oe", "view_memory", "width_source", "collapse", "seed_check", "top_check") if k in event}
    return out


def _pointed_visibility(driver, scene, pid, uv, ctx, robot):
    """P-2 restated on the POINTED pixel (while BP lacks P-E; declared): when the answered pixel lies below the
    fingertip band's top row minus (kappa D2 + FT_PROBE_RADIUS_PX) at the part's measured height, the hand moves
    along -R0 by the excess / kappa before the probe and the pixel is carried by the move (the back-projected point
    re-projected; no new pointing). Returns (uv, info | None)."""
    from gpt_field import agent_speedup_v2_planner as planner
    if getattr(planner, "P11_CAP_M", None) is not None:
        return uv, None
    record = scene.parts.get(pid)
    band_row = geometry.fingertip_band_top_row(ctx["band"])
    if record is None or band_row is None:
        return uv, None
    z = float(record.top_z.value) if record.top_z is not None else (None if record.xyz is None else float(record.xyz[2]))
    if z is None:
        return uv, None
    f = ctx["camera"].f_used
    pose = HandPose.from_view(driver.view)
    point = _backproject_at_z(pose, uv, z, f)
    if point is None:
        return uv, None
    depth = float((point - pose.camera) @ pose.R[:, 2])
    if not depth > 0.0:
        return uv, None
    kappa = f / depth
    excess = float(uv[1]) - (float(band_row) - (kappa * HOVER_PROBE_M + geometry.FT_PROBE_RADIUS_PX))
    if excess <= 0.0:
        return uv, None
    r0 = np.asarray(pose.R[:, 0], dtype=float).copy()
    r0[2] = 0.0
    r0 /= max(float(np.linalg.norm(r0)), 1e-9)
    target = driver.view.position - (excess / kappa) * r0
    reach = reach_hover(driver, target, ctx["start_q"], scene.hf, robot)
    new_uv, _ = geometry.WristCamera(f).project(HandPose.from_view(driver.view), point)
    if new_uv is None:
        return uv, {"shift_m": excess / kappa, "reach": reach, "lost": True}
    return [float(v) for v in new_uv], {"shift_m": excess / kappa, "from_uv": list(map(float, uv)),
                                        "to_uv": [float(v) for v in new_uv], "reach_residual_m": reach["residual_m"]}


def seed_truth_uv(sim, pointing, view):
    """Review finding R4: the ORACLE truth pixel IN THE SEED POSE (scoring only): the pointed part's truth point
    projected into the wrist camera at the state the Seed is taken in (call it when the Seed is made, before the
    probe moves the hand; ``view`` is that state's view, unused by the ORACLE op, which reads the simulator's current
    state). The pointing-time ``truth_uv`` precedes
    ``_pointed_visibility``, which may move the hand; back-projecting it from the moved seed pose mixed two poses
    (E's 8.7 mm exact-seed p90 was that shift). Returns (uv | None, "seed_pose" | "pointing_time")."""
    xyz = pointing.get("truth_xyz")
    if xyz is None:
        return pointing.get("truth_uv"), "pointing_time"
    reply = sim.oracle({"kind": "project", "camera": "wrist", "xyz": [[float(v) for v in xyz]]})
    uv = (reply.get("uv") or [None])[0]
    return (None if uv is None else [float(v) for v in uv]), "seed_pose"


def _image_axis_world_xy(axis_img, pose):
    """The scene's ``_image_axis_world`` restated for scoring: an upright-image line direction as a world-xy axis."""
    a = np.asarray(axis_img, dtype=float)
    return _canonical_axis([float(a[0] * pose.R[0, 1] - a[1] * pose.R[0, 0]),
                            float(a[0] * pose.R[1, 1] - a[1] * pose.R[1, 0])])


def _registration_half_px(scene, pid):
    """The D10 tensor half size the registration used (from its D10_coherence threshold), or None."""
    import re
    for t in (getattr(scene, "_registration_thresholds", {}) or {}).get(pid) or []:
        if getattr(t, "name", None) == "D10_coherence":
            match = re.search(r"half_px=(\d+)", getattr(t, "derivation", "") or "")
            if match:
                return int(match.group(1))
    return None


def axis_variants(view, rec, scene, pid, geo, seed_pose, f_px):
    """RETIRED with S-4' (decision 2 §2.3 S-11; no longer called by ``measure_stage``, kept for the review-1 analyses).
    Review finding R1 (C's yaw, routed to BS + BG): DIAGNOSIS of the D10 image tensor at this registration. The
    grasp pixel (the record point projected in the seed pose) and the tensor half size the scene used; the BG
    ``structure_axis`` on the raw grey (the scene's call, reproduced), on the probe tracker's band-passed grey, and at
    the V5 half size and half the scene's; each axis against the truth axis (scoring) and through the scene's S-4'
    rule restated (the higher-coherence tensor of this variant and the region mask, candidate, region-cell agreement
    within AXIS_AGREE_MAX_RAD): would it be accepted, and wrong. No variant enters a measurement."""
    from gpt_field import agent_speedup_v2_scene as scene_mod
    from gpt_field.agent_speedup_v2_types import AXIS_AGREE_MAX_RAD
    check = getattr(rec, "axis_check", None) or {}
    event = (getattr(scene, "last_registration", None) or {}).get(pid) or {}
    if rec.xyz is None or geo.get("axis") is None:
        return None
    uv, _ = geometry.WristCamera(f_px).project(seed_pose, np.asarray(rec.xyz, dtype=float))
    half = _registration_half_px(scene, pid)
    if uv is None or half is None:
        return {"available": False, "half_px": half}
    gray = np.asarray(view.wrist_gray_upright)
    images = {"raw": gray, "bandpass": geometry.bandpass_grey(gray)}
    halves = {"half": half, "v5_half": int(geometry.V5_TENSOR_HALF_PX), "half_x0.5": max(4, int(round(0.5 * half)))}
    mask_coherence = event.get("mask_coherence")
    mask_axis = event.get("mask_axis")
    cell_axis = check.get("region_axis")
    cell_ratio = check.get("cell_ratio")
    cell_defined = cell_axis is not None and (cell_ratio is None or cell_ratio >= scene_mod.D10_MIN_ASPECT)
    aspect = check.get("aspect")
    out = {"available": True, "grasp_uv": [float(v) for v in uv], "half_px": half, "variants": {},
           "scene_image_axis_err_deg": None if event.get("image_axis") is None else
           _axis_angle_deg(event["image_axis"], geo["axis"]),
           "scene_accepted": bool(check.get("accepted")), "region_axis_err_deg": None if cell_axis is None else
           _axis_angle_deg(cell_axis, geo["axis"])}
    for image_name, image in images.items():
        for half_name, h in halves.items():
            axis_img, coherence, _ = geometry.structure_axis(image, tuple(uv), half_px=h)
            world = _image_axis_world_xy(axis_img, seed_pose)
            use_mask = mask_coherence is not None and mask_axis is not None and float(mask_coherence) > coherence
            deciding, deciding_coh = (np.asarray(mask_axis, float), float(mask_coherence)) if use_mask else \
                (world, float(coherence))
            candidate = (deciding_coh >= scene_mod.D10_MIN_COHERENCE and aspect is not None
                         and aspect >= scene_mod.D10_MIN_ASPECT and not rec.partial)
            agree = None if cell_axis is None else math.radians(_axis_angle_deg(deciding, cell_axis))
            accepted = bool(candidate and cell_defined and agree is not None and agree <= AXIS_AGREE_MAX_RAD)
            err = _axis_angle_deg(deciding, geo["axis"])
            out["variants"][f"{image_name}:{half_name}"] = {
                "half_px": h, "coherence": float(coherence), "axis_err_deg": _axis_angle_deg(world, geo["axis"]),
                "deciding": "mask" if use_mask else "image", "deciding_err_deg": err, "accepted": accepted,
                "wrong": accepted and err > math.degrees(YAW_GRID_RAD / 2.0)}
    return out


def _decode_png(url):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))


class _OracleCamera:
    """The ORACLE wrist camera rebuilt from a ``project`` reply (scoring only; the review-2 judge's ExtCam): its pose
    (cam_xyz, cam_xmat) and the focal length fitted to the reply's projections of known points."""

    def __init__(self, xyz, reply, size_px):
        self.p = np.asarray(reply["cam_xyz"], dtype=float)
        self.R = np.asarray(reply["cam_xmat"], dtype=float).reshape(3, 3)
        self.c = (float(size_px) - 1.0) / 2.0
        pairs = [(np.asarray(x, float), np.asarray(u, float)) for x, u in zip(xyz, reply["uv"]) if u is not None]
        local = (np.array([x for x, _ in pairs]) - self.p) @ self.R
        depth = -local[:, 2]
        a = np.r_[local[:, 0] / depth, -local[:, 1] / depth]
        b = np.r_[[u[0] - self.c for _, u in pairs], [u[1] - self.c for _, u in pairs]]
        self.f = float(a @ b / (a @ a))

    def points(self, us, vs, depth):
        local = np.column_stack([(us - self.c) * depth / self.f, -(vs - self.c) * depth / self.f, -depth])
        return self.p[None, :] + local @ self.R.T


def visual_surface_widths(mask, depth_m, camera, xyz, axis, depths_mm=G1P_E_BAND_VISUAL_DEPTHS_MM):
    """ORACLE visible-surface widths (decision 2 §4.2, reported; the judge's analyze_live rule): the part's visible
    wrist pixels back-projected with the rendered depth, inside the pad strip (+-VISUAL_STRIP_HALF_M along the truth
    axis) around the record point; their extent across the axis (mm) within D mm of the strip's visible top, for D
    in ``depths_mm`` ("D2" is the visual crest)."""
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return {}
    points = camera.points(us.astype(float), vs.astype(float), np.asarray(depth_m, float)[vs, us])
    a = _canonical_axis(axis)
    jaw = np.array([-a[1], a[0]])
    rel = points[:, :2] - np.asarray(xyz, dtype=float)[:2]
    strip = np.abs(rel @ a) <= VISUAL_STRIP_HALF_M
    if strip.sum() < 4:
        return {}
    top = np.percentile(points[strip, 2], VISUAL_TOP_PERCENTILE)
    out = {}
    for depth in depths_mm:
        sel = strip & (points[:, 2] >= top - depth / 1000.0)
        if sel.sum() > 3:
            out[f"D{depth}"] = 1000.0 * float(np.ptp(rel[sel] @ jaw))
    return out


def visual_surface_truth(sim, answers, pid, rec, geo):
    """ORACLE (scoring only) at the registration view: the visible-surface widths of ``visual_surface_widths`` across
    the truth axis at the record's point; None for a part without a truth axis."""
    if rec.xyz is None or geo.get("axis") is None:
        return None
    try:
        mask = answers.visible_mask(pid, "wrist")
        if mask is None:
            return {"available": False, "reason": "not_visible"}
        anchor, axis = np.asarray(geo["anchor"], float), np.asarray(geo["axis"], float)
        xyz = [[float(anchor[0] + t * axis[0]), float(anchor[1] + t * axis[1]), float(anchor[2])]
               for t in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        reply = sim.oracle({"kind": "project", "camera": "wrist", "xyz": xyz})
        seg = sim.oracle({"kind": "segment", "camera": "wrist", "masks": [{"name": "robot", "robot": True}],
                          "depth_png": True, "mask_png": False})
        depth = _decode_png(seg["depth_png"]).astype(float) * float(seg["depth_png_unit_m"])
        widths = visual_surface_widths(mask, depth, _OracleCamera(xyz, reply, mask.shape[1]), rec.xyz, axis)
        return {"available": True, **widths}
    except Exception as exc:  # noqa: BLE001 - a scoring capture only
        return {"available": False, "error": repr(exc)[:300]}


def measure_stage(driver, sim, answers, ctx, scene, pid, target, label):
    """One hover measurement: free_move to ``target`` (None: measure where the hand is), the ORACLE pointing
    stand-in, the ParallaxProbe with a probe-tracker SurveyRecorder (exact seed + dense lattice at the pointed pixel
    + the scene's relief targets), absorb_survey, register(seed). Returns (stage dict, PartRecord or failure)."""
    from gpt_field import agent_speedup_v2_primitives as primitives
    robot, camera = ctx["robot"], ctx["camera"]
    stage = {"label": label, "step_start": driver.steps}
    if target is not None:
        stage["reach"] = reach_hover(driver, target, ctx["start_q"], scene.hf, robot)
    stage["grip_xyz"] = driver.view.position.tolist()
    top_truth = part_top_truth(answers, pid)
    stage["part_top_truth"] = top_truth
    turns = geometry.frame_alignment_turns(driver.view.R, ctx["external"])
    uv, pointing = fine_point(answers, scene, driver, pid, turns)
    stage["pointing"] = pointing
    if uv is None:
        stage["result"] = "part_not_found_at_hover"
        return stage, None, None
    if ctx.get("pointed_visibility", True):
        uv, moved = _pointed_visibility(driver, scene, pid, uv, ctx, robot)
        if moved is not None:
            stage["pointed_visibility"] = moved
    seed = Seed("wrist", tuple(float(v) for v in uv), driver.steps, "fine", FINE_QUANT_PX,
                tuple(float(v) for v in driver.view.state))
    # R4: the truth pixel in the SEED pose, projected now (the probe below moves the hand and returns only roughly)
    truth_seed, truth_source = seed_truth_uv(sim, pointing, driver.view)
    stage["truth_uv_seed"], stage["truth_uv_source"] = truth_seed, truth_source
    targets = scene.survey_targets(pid, driver.view)
    recorder = geometry.SurveyRecorder(driver.view, robot, camera=camera, seeds=[(pid, tuple(uv))],
                                       relief=targets["relief"], exclude_mask=ctx["band"], external=False,
                                       start_step=driver.steps,
                                       **_recorder_kwargs(probe_tracker=True, dense_uv=tuple(uv)))
    timed = TimedRecorder(recorder)
    probe = primitives.ParallaxProbe(driver.view, recorder=timed, gripper="open", touch_threshold_mm=ctx["g3"],
                                     amplitude_m=HOVER_PROBE_M)
    outcome = driver.run(probe, "parallax")
    evidence = outcome.get("evidence") or {}
    stage["probe"] = {"status": outcome.get("status"), "reason": outcome.get("reason"),
                      "legs": [{k: leg.get(k) for k in ("stage", "reason", "status", "travel_m", "steps")}
                               for leg in evidence.get("legs") or []]}
    began = time.perf_counter()
    result = recorder.result(camera)
    result_s = time.perf_counter() - began
    changed = scene.absorb_survey(result, driver.view)
    began = time.perf_counter()
    rec = scene.register(pid, driver.view, seed=seed, empty_hand=True)
    register_s = time.perf_counter() - began
    if seed_outside_region(rec) and "grip" in ctx["roles"].get(pid, ()):
        # lead ruling 3 (r2close): BR re-points ONCE at the hover before giving up (ORACLE stand-in); the stage then
        # scores the re-pointed seed (its own truth pixel) and its registration
        rec, new_seed, stage["repoint"] = repoint_after_seed_refusal(driver, sim, answers, scene, pid, rec, turns)
        if new_seed is not None:
            stage["repoint"]["first_truth_uv_seed"] = truth_seed
            seed, truth_seed = new_seed, stage["repoint"]["truth_uv_seed"]
            stage["truth_uv_seed"], stage["truth_uv_source"] = truth_seed, stage["repoint"]["truth_uv_source"]
    stage["compute"] = {"features": len(recorder.tracker.features), "frames": timed.frames,
                        "track_ms_per_frame": 1000.0 * timed.seconds / max(timed.frames, 1),
                        "decode_ms_per_frame": 1000.0 * timed.decode_seconds / max(timed.frames, 1),
                        "adaptive_search": bool(recorder.adaptive_search),
                        "result_s": result_s, "register_s": register_s, "samples": len(result.samples),
                        "dense_features": recorder.summary().get("dense_features"), "cells_changed": changed}
    stage["survey"] = recorder.summary()
    seed_pose = HandPose.from_state8(seed.pose)
    stage["record"] = score_record(rec, ctx["geometry"][pid], top_truth, robot, seed_pose=seed_pose,
                                   truth_uv=truth_seed, f_px=camera.f_used,
                                   event=(scene.last_registration or {}).get(pid))
    if isinstance(rec, PartRecord):
        # the review-1 S-4' tensor variants (``axis_variants``) are not recorded since S-11 retired S-4' (decision 2
        # §2.3); the scene logs the D10-OE numbers in axis_check / its registration event
        stage["record"]["visual_truth"] = visual_surface_truth(sim, answers, pid, rec, ctx["geometry"][pid])
    stage["dsw"] = jsonable(getattr(scene, "last_dsw", None))
    stage["result"] = "registered" if isinstance(rec, PartRecord) else "registration_failed"
    stage["_survey"] = (result, recorder, driver.view)
    return stage, rec, seed


def _br1_trigger(rec, support_n, refine_axis=False, *, grip=True):
    """BR-1 restated with design review 2's amended triggers (decision 2 §2.6): the band has n < TOP_BAND_SUPPORT_N
    (any part); an axis refusal ``axis_check.source`` == AXIS_INCONSISTENT (S-11 "no_edge_agreement", S-12
    "views_disagree", a collapsed region whose outline passed) on a part with role grip (``grip``); ``region_collapsed``
    (S-13) on a grip part; P-C ``refine_axis`` (the planner's answer on this record; ``_refine_axis_requested`` asks it
    only for a grip part). "not_elongated" (a failed outline gate, an out-of-scope region) is never a trigger, and an
    axis the scene kept from an earlier view (S-12 "kept_earlier") is no refusal. None when the record carries
    neither a band nor an axis check."""
    from gpt_field.agent_speedup_v2_types import AXIS_INCONSISTENT
    if not isinstance(rec, PartRecord):
        return None
    band = getattr(rec, "top_band", None)
    check = getattr(rec, "axis_check", None)
    if band is None and check is None and not refine_axis:
        return None
    reasons = []
    if band is not None and int(band.get("n", 0)) < support_n:
        reasons.append("band_support")
    if grip and check is not None and check.get("source") == AXIS_INCONSISTENT:
        reasons.append("axis_inconsistent")
    if grip and check is not None and check.get("region_collapsed"):
        reasons.append("region_collapsed")
    if refine_axis:
        reasons.append("refine_axis")
    return reasons


def _refine_axis_requested(scene, rec, pid, ctx, driver):
    """The P-C hook of the BR-1 trigger (decision §2.6): BP ``plan_contact`` on the record of a grip part; True when
    it answers ``refine_axis`` (``Infeasible(..., refine_axis=True)``, P-C). False until BP lands P-C, or for a part
    that is not gripped."""
    from gpt_field import agent_speedup_v2_planner as planner
    if not isinstance(rec, PartRecord) or "grip" not in ctx["roles"].get(pid, ()):
        return False
    purpose, obj = _plan_purpose(ctx["plan"], pid)
    if purpose is None:
        return False
    try:
        inputs = planner.PlannerInputs.build(ctx["robot"], driver.view.state,
                                             g_open_mm=ctx["record"].gripper.open_gap_mm,
                                             start_quaternion=ctx["start_q"])
        kwargs = {}
        if "part_fill_cells" in inspect.signature(planner.plan_contact).parameters and \
                hasattr(scene, "part_fill_cells"):
            kwargs["part_fill_cells"] = scene.part_fill_cells(pid)
        next_held = scene.parts.get(obj) if (obj and obj != pid and purpose == "carry") else None
        result = planner.plan_contact(rec, scene.hf, inputs, (), purpose=purpose, next_held=next_held, **kwargs)
    except Exception:  # noqa: BLE001 - the hook is a trigger only
        return False
    return bool(getattr(result, "refine_axis", False))


def _scored_stage(stages):
    """Review finding R3: the stage whose record BR holds after the hover, the one scored and replayed: the last stage
    that returned a PartRecord (a later stage that found nothing or failed leaves the scene's earlier registration in
    place, ``SceneModel._failure`` keeps a registered record), else the last stage."""
    for index in range(len(stages) - 1, -1, -1):
        if (stages[index].get("record") or {}).get("type") == "PartRecord":
            return index
    return len(stages) - 1


def _plan_purpose(plan, pid):
    """(purpose, object part) of a grip part from the plan's steps (turn -> "turn"; a place / carry -> "carry";
    else "hold")."""
    for step in plan.steps:
        grips = list(getattr(step, "grip", ()) or ())
        if pid in grips:
            do = getattr(step, "do", None)
            obj = getattr(step, "obj", None)
            if do == "turn":
                return "turn", obj
            if do in ("place_on", "place_in", "push_to", "slide"):
                return "carry", obj
            return "hold", obj
    return None, None


def g1c_replay(scene, rec, pid, ctx, driver, geo, *, dump_path=None):
    """G1c offline part (decision §4, diagnosis here; BS + BP own RS-2): BP ``plan_contact`` on the final record of a
    grip part with the scene's HeightField (``part_fill_cells`` passed when the planner takes it): planned or the
    refusal, the plan's jaw yaw error against the truth jaw (across the truth axis), marginal / unverified cells."""
    from gpt_field import agent_speedup_v2_planner as planner
    purpose, obj = _plan_purpose(ctx["plan"], pid)
    if purpose is None or not isinstance(rec, PartRecord):
        return None
    inputs = planner.PlannerInputs.build(ctx["robot"], driver.view.state, g_open_mm=ctx["record"].gripper.open_gap_mm,
                                         start_quaternion=ctx["start_q"])
    next_held = scene.parts.get(obj) if (obj and obj != pid and purpose == "carry") else None
    kwargs = {}
    fills = None
    if "part_fill_cells" in inspect.signature(planner.plan_contact).parameters and hasattr(scene, "part_fill_cells"):
        fills = scene.part_fill_cells(pid)
        kwargs["part_fill_cells"] = fills
    try:
        result = planner.plan_contact(rec, scene.hf, inputs, (), purpose=purpose, next_held=next_held, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnosis only
        return {"purpose": purpose, "error": repr(exc)}
    out = {"purpose": purpose, "object": obj, "planned": isinstance(result, planner.ContactPlan),
           "part_fill_cells": None if fills is None else int(len(fills)), "trace": jsonable(planner.trace_payload(result))}
    if isinstance(result, planner.ContactPlan):
        out.update(depth_rule=result.depth_rule, marginal=bool(getattr(result, "marginal", False)),
                   unverified_cells=jsonable((getattr(result, "checks", None) or {}).get("unverified_cells")),
                   target_grip_z=result.target_grip_z, grip_point_xyz=list(map(float, result.grip_point_xyz)))
        if geo.get("axis") is not None:
            truth_jaw = np.array([-geo["axis"][1], geo["axis"][0]])
            out["jaw_err_deg"] = _axis_angle_deg(result.jaw_world[:2], truth_jaw)
    else:
        out["reason"] = getattr(result, "reason", None)
    if dump_path is not None:
        Path(dump_path).write_bytes(pickle.dumps({"hf": scene.hf, "record": rec, "inputs": inputs,
                                                  "next_held": next_held, "purpose": purpose,
                                                  "part_fill_cells": fills, "truth_geometry": geo}))
    return out


def copy_scene(scene):
    """An independent copy of a SceneModel for one hover measurement: deep-copied state; the modules it holds (the
    lazily imported geometry module) and the immutable calibration / robot / camera objects are shared."""
    import types
    memo = {}
    for name, value in vars(scene).items():
        if isinstance(value, types.ModuleType) or name in ("cal", "robot", "camera", "projector"):
            memo[id(value)] = value
    return copy.deepcopy(scene, memo)


def _seeded_start(scene, pid, record, ctx, driver):
    """BR restated for a part whose start registration failed (register_regions: part_not_found after the coarse
    re-seed): its hover is planned from its SEED (hover_for serves "a part that is only seeded"): the coarse wrist
    re-seed when there was an answer, else its plan seed (external), lifted by the scene's ``estimate_seed_xyz``;
    sigma_h from the seed covariance. The copy is used for the hover only (the scene keeps its record)."""
    import dataclasses
    seeds = list(ctx.get("start_seeds", {}).get(pid, [])) + list(reversed(list(record.seeds or [])))
    for seed in seeds:
        try:
            got = scene.estimate_seed_xyz(seed, driver.view)
        except Exception as exc:  # noqa: BLE001 - one seed's failure: try the next
            got, error = None, repr(exc)
        if got is None:
            continue
        point, cov = got
        sigma_h = float(math.sqrt(max(float(np.asarray(cov)[2, 2]), 0.0)))
        seeded = dataclasses.replace(record, xyz=np.asarray(point, dtype=float), cov=np.asarray(cov, dtype=float),
                                     sigma_h_m=sigma_h)
        return seeded, {"seed": {"camera": seed.camera, "source": seed.source, "uv": list(seed.uv)},
                        "xyz": [float(v) for v in point], "sigma_h_mm": 1000.0 * sigma_h}
    return record, {"seed": None, "seeds_tried": len(seeds)}


def pipeline_hover(driver, sim, answers, ctx, pid, offset, *, locate=True, br1=True, dump_dir=None, tag=""):
    """One G1' measurement of part ``pid`` at one hover offset on the pipeline path (module docstring step 3), from a
    fresh copy of the post-start scene."""
    from gpt_field import agent_speedup_v2_scene as scene_mod
    scene = copy_scene(ctx["snapshot"])
    driver.scene = scene
    # decision 2 §2.3 S-12: a fresh scene copy starts with an empty view memory (each hover is an independent
    # measurement; chain -> second stage -> BR-1 is one memory window)
    forget = getattr(scene, "part_moved", None)
    if callable(forget):
        forget(pid)
    scene.on_view(driver.view, step=driver.steps)
    robot = ctx["robot"]
    grip = "grip" in ctx["roles"].get(pid, ())
    row = {"pid": pid, "class": PART_CLASS[pid], "offset_m": list(offset), "stages": [], "flags": [], "grip": grip,
           "view_memory_reset": callable(forget)}
    start = scene.parts.get(pid)
    row["start_record"] = score_record(start, ctx["geometry"][pid], None, robot) if start is not None else None
    if start is not None and start.xyz is None:
        start, row["seeded_start"] = _seeded_start(scene, pid, start, ctx, driver)
    if start is None or start.xyz is None:
        row["result"] = "no_measured_start_record"
        return row
    band_row = geometry.fingertip_band_top_row(ctx["band"]) if ctx.get("visibility", True) else None
    target, hover = chain_hover(start, scene.hf, robot, HandPose.from_view(driver.view), offset,
                                f_px=ctx["camera"].f_used, band_top_row=band_row)
    row["hover"] = hover
    if target is None:
        row["result"] = "hover_refused"
        return row
    stage, rec, seed = measure_stage(driver, sim, answers, ctx, scene, pid, target, "chain")
    row["stages"].append(stage)
    if hover.get("first_stage") and isinstance(rec, PartRecord):
        second, info = chain_hover(rec, scene.hf, robot, HandPose.from_view(driver.view), offset,
                                   f_px=ctx["camera"].f_used, band_top_row=band_row)
        if second is not None:
            dxy = float(np.linalg.norm(second[:2] - driver.view.position[:2]))
            dz = float(driver.view.position[2] - second[2])
            row["two_stage"] = {"hover": info, "dxy_m": dxy, "dz_m": dz,
                                "taken": dxy > TWO_STAGE_XY_M or abs(dz) > TWO_STAGE_Z_M}
            if row["two_stage"]["taken"]:
                stage2, rec2, seed2 = measure_stage(driver, sim, answers, ctx, scene, pid, second, "second_stage")
                row["stages"].append(stage2)
                if isinstance(rec2, PartRecord):
                    rec, seed = rec2, seed2
    support_n = int(getattr(scene_mod, "TOP_BAND_SUPPORT_N", 10))
    refine = _refine_axis_requested(scene, rec, pid, ctx, driver)
    trigger = _br1_trigger(rec, support_n, refine_axis=refine, grip=grip)
    row["br1_trigger"] = trigger
    if br1 and trigger:
        shifted = driver.view.position + HOVER_PROBE_M * driver.view.R[:, 1]
        stage3, rec3, seed3 = measure_stage(driver, sim, answers, ctx, scene, pid, shifted, "br1")
        row["stages"].append(stage3)
        if isinstance(rec3, PartRecord):
            rec, seed = rec3, seed3
    scored = _scored_stage(row["stages"])
    final = row["stages"][scored]
    row["scored_stage"] = {"index": scored, "label": final.get("label"), "last": scored == len(row["stages"]) - 1}
    row["final"] = final.get("record")
    # decision 2 §4.1: the scored record is the one the scene HOLDS after the hover (S-12 view memory included); it is
    # the last registered stage's record (a failure keeps the earlier registration), checked here
    held = scene.parts.get(pid)
    memory = getattr(scene, "view_memory", None)
    row["held"] = {"is_scored_stage_record": held is rec if isinstance(rec, PartRecord) else None,
                   "view_memory": jsonable(memory(pid)) if callable(memory) else None}
    if isinstance(rec, PartRecord) and isinstance(held, PartRecord) and held is not rec and \
            held.status == "registered" and not held.provisional:
        rescored = score_record(held, ctx["geometry"][pid], final.get("part_top_truth"), robot,
                                seed_pose=None if seed is None or seed.pose is None else HandPose.from_state8(seed.pose),
                                truth_uv=final.get("truth_uv_seed"), f_px=ctx["camera"].f_used)
        recorded = {k: v for k, v in (final.get("record") or {}).items() if k not in ("event", "visual_truth")}
        if jsonable(rescored) == jsonable(recorded):
            row["held"]["is_scored_stage_record"] = "equal_content"      # a copy of the same registration
        else:
            for key in ("event", "visual_truth"):
                if key in (final.get("record") or {}):
                    rescored[key] = final["record"][key]
            row["final"] = rescored
            rec = held
    row["part_top_truth"] = final.get("part_top_truth")
    row["result"] = final.get("result")
    row["flags"] = [s["label"] for s in row["stages"]]
    survey = final.pop("_survey", None)
    for st in row["stages"]:
        st.pop("_survey", None)
    if survey is not None:
        result, recorder, view = survey
        rows = score_samples(sim, answers, result, view, ctx["camera"], recorder)
        mask = answers.visible_mask(pid, "wrist")
        classes0 = classify_pixels(sim, answers, "wrist", [f["uv0"].tolist() for f in recorder.tracker.features.values()])
        classes_by_fid = dict(zip(recorder.tracker.features, classes0))
        row["rows"] = rows
        row["classes"] = class_table(rows)
        row["texture"] = texture_rates(geometry.bandpass_grey(view.wrist_gray_upright), mask, result, classes_by_fid,
                                       PART_CLASS[pid], radius_px=geometry.FT_PROBE_RADIUS_PX,
                                       min_std=geometry.D15_PROBE_MIN_STD)
        row["edges"] = edge_split(rows, mask, PART_CLASS[pid])
        row["measurement"] = hover_measurement(rows, PART_CLASS[pid], {"source": "oracle_grid_this_hover",
                                                                        **(row["part_top_truth"] or {})})
    else:
        row["rows"] = []
    row["probe"] = final.get("probe")
    dump = None if dump_dir is None else Path(dump_dir) / f"plan_inputs_{pid}{tag}.pkl"
    row["g1c"] = g1c_replay(scene, rec, pid, ctx, driver, ctx["geometry"][pid], dump_path=dump) \
        if "grip" in ctx["roles"].get(pid, ()) else None
    row["locator"] = locate_at_hover(driver, sim, ctx, robot) if locate else None
    return row


# ----------------------------------------------------------------------------------------------- episode
def run_episode(args, episode_index, out):
    """One episode on the pipeline path (module docstring steps 1-3)."""
    from gpt_field import agent_speedup_v2_plan as plan_mod
    from gpt_field import agent_speedup_v2_scene as scene_mod
    h = harness()
    oracle = oracle_module()
    sim, spec, _config, monitor = h.start_sim(args.task, oracle=True, sim_python=args.sim_python,
                                              libero_root=args.libero_root)
    robot = RobotModel()
    began = time.monotonic()
    record = {"episode_index": episode_index, "hover_mode": args.hover, "hovers": []}
    try:
        obs, episode = h.reset_episode(sim, spec, episode_index)
        record["episode"] = episode
        answers = oracle.OracleAnswers(sim, spec.key)
        scene = scene_mod.SceneModel(robot)
        driver = Driver(sim, obs, scene=scene)
        scene.on_view(driver.view, step=0)
        cal = calibrate_plus(driver, robot, grid_at_reference=args.calib_grid == "reference")
        calibration_q = np.asarray(cal["start_q"], dtype=float).copy()
        if args.hover_yaw_deg:
            # decision 2 §4.3 / §4.4 (the generality and RS-4 yaw sets; the lens's record_chain rule): calibration
            # unchanged, then every transit / hover orientation is the start orientation turned about world z
            cal["start_q"] = turn_about_world_z(cal["start_q"], math.radians(args.hover_yaw_deg))
        record["hover_yaw_deg"] = float(args.hover_yaw_deg or 0.0)
        start_rows = score_samples(sim, answers, cal["result"], driver.view, cal["camera"], cal["recorder"])
        J_truth, ray_truth = external_truth(sim, cal)
        external = cal["external"]
        start_info = scene.start(cal["v2cal"], driver.view)
        turns = cal["turns"]
        plan = plan_mod.parse_plan(answers.answer("plan", {"quarter_turns": turns, "image_size": 512}))
        roles = plan_mod.part_roles(plan)
        record["roles"] = {pid: sorted(r) for pid, r in roles.items()}
        scene.merge_plan(plan, roles, driver.view, turns)
        starts, start_seeds = {}, {}
        for pid in ("B", "D", "E", "A", "C"):
            if pid not in roles:
                continue
            provisional = "grip" in roles[pid] and not (roles[pid] - {"grip"})
            reg = scene.register(pid, driver.view, empty_hand=True, provisional=provisional,
                                 allow_partial=not provisional)
            first = None
            if not isinstance(reg, PartRecord):
                # BR register_regions restated: a failed check -> ONE point_coarse (purpose region_seed; the ORACLE
                # stand-in) in the start view, then the registration again from that wrist seed
                first = {"type": type(reg).__name__, "reason": getattr(reg, "reason", None)}
                answers.answer("point_coarse", {"pid": pid, "camera": "wrist", "quarter_turns": turns,
                                                "image_size": 512})
                coarse = answers.records[-1].get("answered_uv")
                first["coarse_uv"] = coarse
                if coarse is not None:
                    seed = Seed("wrist", tuple(float(v) for v in coarse), driver.steps, "coarse", COARSE_QUANT_PX,
                                tuple(float(v) for v in driver.view.state))
                    start_seeds.setdefault(pid, []).append(seed)
                    reg = scene.register(pid, driver.view, seed=seed, empty_hand=True, provisional=provisional,
                                         allow_partial=not provisional)
            starts[pid] = {"first_attempt": first, "provisional_call": provisional, "type": type(reg).__name__,
                           "status": getattr(reg, "status", None), "reason": getattr(reg, "reason", None),
                           "sigma_h_mm": None if getattr(reg, "sigma_h_m", None) is None else 1000.0 * reg.sigma_h_m,
                           "top_bound": getattr(reg, "top_bound", None)}
        geometry_truth = {pid: part_geometry_truth(sim, answers, pid, robot) for pid in args.parts}
        anchors = answers.truth_points(detail=True)
        anchor_truth = answers.surface_truth([info["xyz"][:2] for info in anchors.values()],
                                             cameras=("wrist", "external"))
        anchor_rows = {pid: {"anchor_xyz": info["xyz"], "aabb_z": info["aabb_z"], "collision_z": tr.get("surface_z"),
                             "visual_wrist_z": ((tr.get("visual") or {}).get("wrist") or {}).get("z"),
                             "visual_external_z": ((tr.get("visual") or {}).get("external") or {}).get("z")}
                       for (pid, info), tr in zip(anchors.items(), anchor_truth)}
        loom = cal["loom_result"].loom
        locator = cal["locator"]
        record["calibration"] = {
            "steps": driver.steps, "outcome": cal["outcome"].get("reason"), "outcome_reason": cal["outcome_reason"],
            "camera": cal["camera"].to_dict(), "f_hat": None if loom["f_hat"] is None else loom["f_hat"].to_dict(),
            "f_hat_spread_px": loom["f_hat_spread"], "f_hat_features": len(loom["f_hat_by_fid"]),
            "f_hat_rel_error_vs_R3": None if loom["f_hat"] is None else
            loom["f_hat"].value / robot_facts.focal_length_px(512) - 1.0,
            "survey": cal["result"].summary(), "survey_compute": cal["compute"],
            "external": None if external is None else external.to_dict(),
            "external_truth": {"J": J_truth.tolist(), "ray": ray_truth.tolist(),
                               "v_hat_angle_deg": None if external is None else
                               geometry._angle_deg(external.v_hat, J_truth @ [0.0, 0.0, -1.0]),
                               "ray_angle_deg": None if external is None else
                               geometry._angle_deg(external.ray_dir, ray_truth),
                               "J_rel_error": None if external is None else
                               float(np.linalg.norm(external.J - J_truth) / np.linalg.norm(J_truth))},
            "quarter_turns": turns, "external_samples": cal["result"].external_samples,
            "locator": score_locator(sim, cal, robot),
            "locator_fit": None if locator is None else {"degraded": bool(getattr(locator, "degraded", False)),
                                                         "located": len(locator.located),
                                                         "dropped": getattr(locator, "dropped", None),
                                                         "sigma_loc_px": locator.sigma_loc_px},
            "fingertip_band_top_row": geometry.fingertip_band_top_row(cal["band"]),
            "start_survey_classes": class_table(start_rows), "anchors": anchor_rows,
            "gripper": {"g_open_mm": cal["record"].gripper.open_gap_mm, "g3_mm": cal["g3"].value,
                        "g5_mm": cal["g5"].value, "g6_mm": cal["g6"].value},
            "scene_start": start_info, "start_registration": starts, "geometry_truth": geometry_truth}
        record["calibration"]["region_B"] = region_iou(sim, answers, cal, start_rows, driver,
                                                       out / f"region_B_episode_{episode_index}.npz")
        ctx = {"robot": robot, "camera": cal["camera"], "external": external, "locator": locator,
               "start_q": cal["start_q"], "band": cal["band"], "g3": cal["g3"], "record": cal["record"],
               "plan": plan, "roles": roles, "geometry": geometry_truth, "snapshot": copy_scene(scene),
               "pointed_visibility": args.visibility, "visibility": args.visibility, "start_seeds": start_seeds,
               "calibration_q": calibration_q}
        dump_dir = out if args.dump else None
        for pid in args.parts:
            for k, offset in enumerate(HOVER_OFFSETS_M[:args.offsets]):
                began_hover = time.monotonic()
                try:
                    if args.hover == "seeded":
                        row = seeded_hover(driver, sim, answers, ctx, pid, offset, locate=args.locate)
                    else:
                        row = pipeline_hover(driver, sim, answers, ctx, pid, offset, locate=args.locate,
                                             br1=args.br1, dump_dir=dump_dir, tag=f"_ep{episode_index}_o{k}")
                except Exception:  # noqa: BLE001 - one hover's failure is recorded, the survey continues
                    row = {"pid": pid, "class": PART_CLASS[pid], "offset_m": list(offset), "result": "error",
                           "error": traceback.format_exc(), "stages": [], "rows": []}
                row["wall_s"] = round(time.monotonic() - began_hover, 1)
                record["hovers"].append(row)
                print(json.dumps({"episode": episode_index, "pid": pid, "offset": list(offset),
                                  "result": row.get("result"), "stages": row.get("flags"),
                                  "top_err_visual_mm": (row.get("final") or {}).get("top_err_visual_mm"),
                                  "sigma_h_mm": (row.get("final") or {}).get("sigma_h_mm"),
                                  "wall_s": row["wall_s"]}, default=str), flush=True)
        record["oracle_calls"] = answers.report() if hasattr(answers, "report") else None
        record["steps"] = driver.steps
    finally:
        record["wall_s"] = round(time.monotonic() - began, 1)
        record["monitor_options"] = monitor
        sim.close()
    path = out / f"episode_{episode_index}.json"
    path.write_text(json.dumps(jsonable(record), indent=1))
    return record


def seeded_hover(driver, sim, answers, ctx, pid, offset, *, locate=True):
    """DIAGNOSIS ONLY (ruling 4d): the ORACLE-seeded hover of the earlier survey (``hover_target``), then the same
    probe and registration as the pipeline's first stage."""
    scene = copy_scene(ctx["snapshot"])
    driver.scene = scene
    scene.on_view(driver.view, step=driver.steps)
    target, info = hover_target(answers, ctx["robot"], pid, offset, driver.view, "grip")
    stage, rec, _ = measure_stage(driver, sim, answers, ctx, scene, pid, target, "seeded")
    stage.pop("_survey", None)
    row = {"pid": pid, "class": PART_CLASS[pid], "offset_m": list(offset), "hover": {"seeded_oracle": True, **info},
           "stages": [stage], "final": stage.get("record"), "part_top_truth": stage.get("part_top_truth"),
           "result": stage.get("result"), "flags": ["seeded"], "rows": [], "probe": stage.get("probe")}
    row["locator"] = locate_at_hover(driver, sim, ctx, ctx["robot"]) if locate else None
    return row


def burner_polygon(sim, answers):
    """World polygon (k, 3) of the region part B's top face from its collision box corners (ORACLE; scoring only)."""
    spec = answers.parts["B"]
    item = {"name": "B", "geom": [spec["item"]["geom"]], "where": "aabb_centre"}
    corners = np.asarray(sim.oracle({"kind": "points", "corners": True, "items": [item]})["corners"]["B"][0], float)
    top = corners[np.argsort(corners[:, 2])[-4:]]
    hull = cv2.convexHull(top[:, :2].astype(np.float32)).reshape(-1, 2)
    return np.c_[hull, np.full(len(hull), float(top[:, 2].max()))]


def region_iou(sim, answers, cal, start_rows, driver, npz_path=None):
    """External mask of the region part B (diagnosis of the projector and of refine_region_mask), against the ORACLE
    visible mask of B. Two affine polygons through the projector anchored on the HandLocator at the start:
    "samples_hull" (the hull of B's truth-selected start-survey samples at their median z: smaller than the part,
    so a good refinement GROWS it) and "box_top" (B's collision box top face from ORACLE geometry: the projector's
    own error). Each is refined; the frames and masks are saved to ``npz_path``."""
    external, locator = cal["external"], cal["locator"]
    rows = [r for r in start_rows if r["class"] == PART_CLASS["B"]]
    if external is None or locator is None:
        return {"available": False, "samples": len(rows)}
    anchor = locator.anchor()
    if anchor is None:
        return {"available": False, "reason": "no_locator_anchor"}
    uv, grip, sigma = anchor
    view = driver.view
    rgb = np.asarray(view.external_rgb_upright)
    projector = geometry.ExternalProjector(external, uv, grip, sigma)
    truth = answers.visible_mask("B", "external")
    if truth is None:
        return {"available": False, "reason": "no_truth_mask"}

    def iou(mask):
        union = np.count_nonzero(mask | truth)
        return float(np.count_nonzero(mask & truth) / union) if union else None
    polygons = {"box_top": burner_polygon(sim, answers)}
    if len(rows) >= 3:
        xy = np.array([r["xyz"][:2] for r in rows])
        hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2)
        polygons["samples_hull"] = np.c_[hull, np.full(len(hull), float(np.median([r["z"] for r in rows])))]
    variants, saved = {}, {"external_rgb": rgb, "truth": truth}
    for name, polygon in polygons.items():
        affine = projector.mask(polygon, rgb.shape[:2])
        seed, sigma_px = projector.project(polygon.mean(axis=0))
        refined, info = geometry.refine_region_mask(rgb, affine, seed, sigma_px=sigma_px)
        variants[name] = {"vertices": len(polygon), "top_z": float(polygon[0, 2]), "affine_px": int(affine.sum()),
                          "truth_px": int(truth.sum()), "iou_affine": iou(affine), "iou_refined": iou(refined),
                          "refine": info, "projector_sigma_px": sigma_px, "seed_uv": seed.tolist()}
        saved[f"{name}_affine"], saved[f"{name}_refined"] = affine, refined
        saved[f"{name}_seed"], saved[f"{name}_sigma"] = seed, np.array(sigma_px)
    if npz_path is not None:
        np.savez_compressed(npz_path, **saved)
    hull = variants.get("samples_hull") or {}
    return {"available": True, "samples": len(rows), "variants": variants,
            "iou_affine": hull.get("iou_affine"), "iou_refined": hull.get("iou_refined"),
            "anchor_uv": uv.tolist(), "anchor_sigma_px": sigma,
            "anchor_truth_uv": sim.oracle({"kind": "project", "camera": "external",
                                           "xyz": [grip.tolist()]})["uv"][0]}


# ----------------------------------------------------------------------------------------------- aggregate
def episode_reference(record, pid):
    """The fixed G1 reference of one part in one episode (BG-R2): "oracle_grid" = the median over the part's hovers
    of ``part_top_truth`` (visual and collision tops near the anchor); runs recorded before it fall back to
    "fallback_sample_max" = the maximum non-robot on-part visual truth over all the part's hover samples of the
    episode (visual) and the anchor's collision surface (collision)."""
    cls = PART_CLASS[pid]
    hovers = [h for h in record["hovers"] if h["pid"] == pid]
    anchor = (record["calibration"].get("anchors") or {}).get(pid) or {}
    tops = [h.get("part_top_truth") or {} for h in hovers]
    visual = [t["visual_top_z"] for t in tops if t.get("visual_top_z") is not None]
    if visual:
        collision = [t["collision_top_z"] for t in tops if t.get("collision_top_z") is not None]
        return {"source": "oracle_grid", "visual_top_z": float(np.median(visual)),
                "collision_top_z": float(np.median(collision)) if collision else None,
                "per_hover_visual_top_z": visual, "spread_mm": 1000.0 * (max(visual) - min(visual)),
                "anchor_visual_wrist_z": anchor.get("visual_wrist_z")}
    samples = [row["visual_z"] for h in hovers for row in h.get("rows") or [] if row["class"] == cls and truth_ok(row)]
    return {"source": "fallback_sample_max", "visual_top_z": max(samples) if samples else None,
            "collision_top_z": anchor.get("collision_z"), "samples": len(samples),
            "anchor_visual_wrist_z": anchor.get("visual_wrist_z")}


def hover_label(record, hover):
    return f"ep{record['episode_index']}:{hover['offset_m'][0]:+.2f},{hover['offset_m'][1]:+.2f}"


def _model_sigma_for_unit_zscore(errors, sigmas):
    """The model-error term m (m) that brings the robust std of e / sqrt(sigma^2 + m^2) to 1 (0 when it already is
    at most 1): the D4 model term a part class would need (BG-R5)."""
    errors, sigmas = np.asarray(errors, dtype=float), np.asarray(sigmas, dtype=float)
    if errors.size < HOVER_CLASS_MIN_SAMPLES or geometry._robust_sigma(errors / sigmas) <= 1.0:
        return 0.0
    lo, hi = 0.0, max(float(np.max(np.abs(errors))), 1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if geometry._robust_sigma(errors / np.sqrt(sigmas ** 2 + mid ** 2)) > 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def _share(flags):
    flags = [bool(f) for f in flags if f is not None]
    return (sum(flags) / len(flags)) if flags else None


WIDTH_KEYS = ("widths_mm", "widths_sigma_mm", "w_min_mm", "w_min_sigma_mm", "w_min_err_mm", "width_ok")


def _check_of(record):
    return (record or {}).get("axis_check") or {}


def _live_view_memory(h):
    """Whether the hover was recorded with the scene's own S-12 view memory (its axis checks carry ``views``)."""
    return any("views" in _check_of(st.get("record")) for st in h.get("stages") or [])


def held_record(h):
    """(record dict, mode): the record the scene holds after the hover (decision 2 §4.1, BR's semantics). Mode "live":
    the scene ran S-12 / S-13 itself, the hover's ``final`` (the held record) as recorded; "final": a hover without
    registered stages (``final`` as recorded); "emulated": a run recorded before S-12 / S-13 (the review-2 judge's
    live runs): both are applied offline over the hover's registered stages in order (chain -> second stage -> BR-1,
    one memory window), exactly as decision 2 §2.3 states them:
    - axis (S-12): a stage's own accepted axis on a non-collapsed region that agrees within YAW_GRID_RAD / 2 with
      every remembered axis is accepted and remembered; one that disagrees (or meets a conflicted memory) is refused
      "views_disagree" and the memory is conflicted; a refusal (any reason, a collapsed region included) while the
      memory holds a non-conflicted axis carries the last remembered axis ("kept_earlier");
    - width (S-13): a collapsed region delivers the memory's last non-collapsed width set, else none (unmeasured);
      otherwise its own widths, remembered.
    The held record is the last registered stage's (point, top, lateral errors) with that axis and width set; its
    ``view_memory`` says how they were decided."""
    final = h.get("final") or {}
    stages = [st for st in h.get("stages") or [] if (st.get("record") or {}).get("type") == "PartRecord"]
    if not stages:
        return final, "final"
    if _live_view_memory(h):
        return final, "live"
    tol = math.degrees(YAW_GRID_RAD / 2.0)
    axes, conflicted, widths = [], False, None
    decision = width_source = None
    axis = None
    for st in stages:
        rec = st["record"]
        collapsed = bool(_check_of(rec).get("region_collapsed"))
        own = (rec.get("axis"), rec.get("axis_err_deg"), rec.get("coherence")) if rec.get("elongated") else None
        if own is not None and own[0] is not None and not collapsed:
            if conflicted or any(_axis_angle_deg(own[0], a[0]) > tol for a in axes):
                decision, conflicted, axis = "views_disagree", True, None
            else:
                decision, axis = "accepted", own
                axes.append(own)
        elif axes and not conflicted:
            decision, axis = "kept_earlier", axes[-1]
        else:
            decision, axis = "refused", None
        if collapsed:
            width_source = "kept_earlier" if widths is not None else "unmeasured"
        else:
            widths = {k: rec[k] for k in WIDTH_KEYS if k in rec} if rec.get("widths_mm") else None
            width_source = "region" if widths is not None else "unmeasured"
    held = {k: v for k, v in stages[-1]["record"].items() if k not in WIDTH_KEYS + ("axis", "axis_err_deg")}
    held["elongated"] = axis is not None
    if axis is not None:
        held["axis"], held["axis_err_deg"], held["coherence"] = axis
    current = None if width_source == "unmeasured" else widths
    if current:
        held.update(current)
    held["width_unmeasured"] = not current
    held["view_memory"] = {"mode": "emulated", "decision": decision, "width_source": width_source,
                           "remembered": len(axes), "conflicted": conflicted,
                           "collapsed": bool(_check_of(stages[-1]["record"]).get("region_collapsed"))}
    return held, "emulated"


def _final_ok(h):
    final = held_record(h)[0]
    return final.get("type") == "PartRecord" and bool(final.get("registered")) and not final.get("partial")


def _grip_parts(records):
    """Parts with role grip: the episode's recorded plan roles, else (older runs) the parts replayed through BP (G1c
    replays only grip parts)."""
    grips = set()
    for r in records:
        roles = r.get("roles")
        if roles:
            grips |= {pid for pid, rs in roles.items() if "grip" in rs}
        else:
            grips |= {h["pid"] for h in r.get("hovers") or [] if h.get("g1c")}
    return grips


def _width_truth(record, pid, final):
    """G1'(e) width truth of a held record: {"mm", "at", "anchor_mm", "grip_offset_mm"}. Decision 2 §5: the collision
    cross-section at the grip, for a non-prismatic part at the record's own grip point (``width_truth_at`` on the
    episode's ORACLE cell grid; "grip_point"); a run recorded without the grid is scored on the anchor strip ("anchor":
    the judge's runs and the earlier ones, the review-2 decision's own scorer), and so is a record whose grip strip
    holds no part cell ("anchor_fallback"). A record without a geometry truth falls back on its stage's own
    w_min_err_mm. ``anchor_mm`` (the anchor truth) and the grip point's distance from the anchor are reported."""
    geo = ((record.get("calibration") or {}).get("geometry_truth") or {}).get(pid) or {}
    anchor_mm = geo.get("w_min_mm")
    if anchor_mm is None and final.get("w_min_mm") is not None and final.get("w_min_err_mm") is not None:
        anchor_mm = final["w_min_mm"] - final["w_min_err_mm"]
    mm, at = width_truth_at(dict(geo, w_min_mm=anchor_mm), final.get("xyz"))
    offset = None
    if final.get("xyz") is not None and (geo.get("cells_grid") or {}).get("anchor_xy") is not None:
        offset = 1000.0 * float(np.hypot(*(np.asarray(final["xyz"][:2], float)
                                          - np.asarray(geo["cells_grid"]["anchor_xy"], float))))
    return {"mm": mm, "at": at, "anchor_mm": anchor_mm, "grip_offset_mm": offset}


def _width_source(record):
    """The width source of a record: the offline emulation's (``held_record`` view_memory), else the scene's own (live
    S-12 / S-13: the registration event)."""
    return (((record or {}).get("view_memory") or {}).get("width_source")
            or ((record or {}).get("event") or {}).get("width_source"))


def _collapsed_width_delivered(h, final, mode):
    """G1'(e1): whether the held record delivers a width set measured on a collapsed region (S-13 forbids it). As
    run (``mode`` "as_run": the hover's recorded ``final``, a diagnosis for runs recorded before S-13): a collapsed
    region with widths that S-12 did not keep from an earlier stage (width source "kept_earlier": the widths are an
    earlier non-collapsed stage's, legitimately delivered; review-2 implementation review R-3). Held: a collapsed
    region whose widths are no earlier non-collapsed stage's set (checked on the stages, not on the label; an
    emulated hold never does)."""
    collapsed = bool(_check_of(final).get("region_collapsed"))
    if not collapsed or not final.get("widths_mm") or (final.get("view_memory") or {}).get("mode") == "emulated":
        return False
    if mode == "as_run":
        return _width_source(final) != "kept_earlier"
    earlier = [(st.get("record") or {}).get("widths_mm") for st in h.get("stages") or []
               if (st.get("record") or {}).get("type") == "PartRecord"
               and not _check_of(st.get("record")).get("region_collapsed")]
    return final["widths_mm"] not in earlier


def width_band_mm(sigma_w_mm):
    """The planner's G13 band (decision 2 §4.2): max(G13_FLOOR_MM, G13_SIGMA_K sigma_w)."""
    from gpt_field import agent_speedup_v2_planner as planner
    return max(float(planner.G13_FLOOR_MM), float(planner.G13_SIGMA_K) * float(sigma_w_mm or 0.0))


def g13_trip_margin_mm(sigma_w_mm):
    """G13+'s false-trip margin: the band + PAD_LENGTH_R0_MM sin(YAW_GRID_RAD / 2) (the planner's G13+ excess)."""
    return width_band_mm(sigma_w_mm) + float(robot_facts.PAD_LENGTH_R0_MM) * math.sin(YAW_GRID_RAD / 2.0)


def g1_prime(records, pid):
    """Decision §4 G1' for one part class on the registered PartRecords the scene holds after each hover (decision 2
    §4.1: ``held_record``), with (e) as amended by decision 2 §4.2 (module docstring)."""
    from gpt_field import agent_speedup_v2_planner as planner
    rows = []
    f_hats = []
    grip = pid in _grip_parts(records)
    for r in records:
        f_err = (r.get("calibration") or {}).get("f_hat_rel_error_vs_R3")
        if f_err is not None:
            f_hats.append(f_err)
        reference = episode_reference(r, pid)
        for h in r["hovers"]:
            if h["pid"] != pid:
                continue
            final, mode = held_record(h)
            as_run = h.get("final") or {}
            entry = {"hover": hover_label(r, h), "result": h.get("result"), "stages": h.get("flags"),
                     "registered": _final_ok(h), "partial": final.get("partial"), "status": final.get("status"),
                     "sigma_h_mm": final.get("sigma_h_mm"), "top_z": final.get("top_z"),
                     "top_bound": final.get("top_bound"), "band_n": (final.get("top_band") or {}).get("n"),
                     "p6_limit_mm": final.get("p6_limit_mm"), "w_min_mm": final.get("w_min_mm"),
                     "w_min_sigma_mm": final.get("w_min_sigma_mm"), "w_min_err_mm": final.get("w_min_err_mm"),
                     "width_ok": final.get("width_ok"), "width_unmeasured": bool(final.get("width_unmeasured")),
                     "view_memory": mode, "width_source": _width_source(final),
                     "move_prior": final.get("move_prior") or PART_MOVE.get(pid),
                     "visual_truth": final.get("visual_truth")}
            truth_info = _width_truth(r, pid, final)
            truth = truth_info["mm"]
            entry.update(w_truth_mm=truth, w_truth_at=truth_info["at"], w_truth_anchor_mm=truth_info["anchor_mm"],
                         grip_offset_from_anchor_mm=truth_info["grip_offset_mm"])
            if entry["w_min_mm"] is not None and truth is not None:
                error = float(entry["w_min_mm"]) - float(truth)
                band = width_band_mm(entry["w_min_sigma_mm"])
                entry.update(width_err_mm=error, g13_band_mm=band, in_g13_band=abs(error) <= band + 1e-9,
                             g13plus_false_trip=-error > g13_trip_margin_mm(entry["w_min_sigma_mm"]) + 1e-9,
                             within_3mm_15pct=abs(error) <= max(WIDTH_TOL_MM, WIDTH_TOL_REL * float(truth)) + 1e-9)
            entry["collapsed_width_delivered"] = _collapsed_width_delivered(h, final, "held")
            entry["collapsed_width_delivered_as_run"] = _collapsed_width_delivered(h, as_run, "as_run")
            top, sigma = final.get("top_z"), final.get("sigma_h_mm")
            if top is not None and reference.get("visual_top_z") is not None:
                entry["err_visual_mm"] = 1000.0 * (top - float(reference["visual_top_z"]))
            if top is not None and reference.get("collision_top_z") is not None:
                entry["err_collision_mm"] = 1000.0 * (top - float(reference["collision_top_z"]))
            if sigma is not None and sigma > 0 and entry.get("err_visual_mm") is not None:
                entry["zscore"] = entry["err_visual_mm"] / sigma
            if sigma is not None and entry.get("err_collision_mm") is not None:
                m_t = max(ENVELOPE_MT_FLOOR_MM, ENVELOPE_MT_K * sigma)
                entry["envelope_mm"] = [-(ENVELOPE_LOW_MM - m_t), ENVELOPE_HIGH_MM + m_t]
                if m_t >= ENVELOPE_LOW_MM:
                    # review R9: the envelope is empty on the palm side once m_t >= 14.4 mm (sigma_h >= 9.6 mm); such
                    # a sigma_h is far above the P6 limit (G1'(c) counts it), not an envelope violation
                    entry["envelope_class"] = "sigma_above_envelope"
                    entry["envelope_ok"] = None
                else:
                    entry["envelope_class"] = "defined"
                    entry["envelope_ok"] = (entry["envelope_mm"][0] <= entry["err_collision_mm"]
                                            <= entry["envelope_mm"][1])
            if sigma is not None and entry["p6_limit_mm"] is not None:
                entry["p6_ok"] = sigma <= entry["p6_limit_mm"] + 1e-9
            rows.append(entry)
    good = [e for e in rows if e["registered"]]
    visual = [e["err_visual_mm"] for e in good if e.get("err_visual_mm") is not None]
    stats = _stats(visual)
    z = [abs(e["zscore"]) for e in good if e.get("zscore") is not None]
    free = bool(good) and all(e["move_prior"] not in planner.FIXTURE_PRIORS for e in good)
    delivered = [e for e in good if e.get("width_err_mm") is not None]
    width = {"grip": grip, "free": free, "registered": len(good), "delivered": len(delivered),
             "unmeasured": sum(e["width_unmeasured"] for e in good),
             "e1_collapsed_widths_delivered": [e["hover"] for e in rows if e["collapsed_width_delivered"]],
             "collapsed_widths_delivered_as_run": [e["hover"] for e in rows if e["collapsed_width_delivered_as_run"]],
             "in_g13_band": sum(bool(e.get("in_g13_band")) for e in delivered),
             "in_g13_band_share_of_registered": _ratio(sum(bool(e.get("in_g13_band")) for e in delivered), len(good)),
             "in_g13_band_share_of_delivered": _ratio(sum(bool(e.get("in_g13_band")) for e in delivered),
                                                      len(delivered)),
             "truth_at": {str(k): sum(e.get("w_truth_at") == k for e in delivered)
                          for k in sorted({str(e.get("w_truth_at")) for e in delivered})},
             "grip_offset_from_anchor_mm": _stats([e.get("grip_offset_from_anchor_mm") for e in delivered]),
             "truth_grip_minus_anchor_mm": _stats([e["w_truth_mm"] - e["w_truth_anchor_mm"] for e in delivered
                                                   if e.get("w_truth_anchor_mm") is not None]),
             "e3_g13plus_false_trips": [e["hover"] for e in delivered if e.get("g13plus_false_trip")],
             "within_3mm_15pct": sum(bool(e.get("within_3mm_15pct")) for e in delivered),
             "within_3mm_15pct_share_of_registered": _ratio(sum(bool(e.get("within_3mm_15pct")) for e in delivered),
                                                           len(good)),
             "width_err_mm": _stats([e["width_err_mm"] for e in delivered]),
             "sigma_w_mm": _stats([e.get("w_min_sigma_mm") for e in delivered]),
             "g13_band_mm": _stats([e.get("g13_band_mm") for e in delivered]),
             "width_sources": {str(k): sum(e.get("width_source") == k for e in good)
                               for k in sorted({str(e.get("width_source")) for e in good})},
             "visual_surface_mm": {f"D{d}": _stats([(e.get("visual_truth") or {}).get(f"D{d}") for e in good])
                                   for d in G1P_E_BAND_VISUAL_DEPTHS_MM},
             "width_minus_visual_mm": {f"D{d}": _stats([e["w_min_mm"] - e["visual_truth"][f"D{d}"] for e in good
                                                        if e.get("w_min_mm") is not None and
                                                        (e.get("visual_truth") or {}).get(f"D{d}") is not None])
                                       for d in G1P_E_BAND_VISUAL_DEPTHS_MM}}
    if not grip:
        width["e_class"] = "reported (not a grip part)"
    elif free:
        width["e_class"] = "e1-e3 (free grip part)"
    else:
        width["e_class"] = "e1 + e4 reported (fixture grip part: P2 / P5 / P16 gated by G1c)"
    out = {"class": PART_CLASS[pid], "hovers": len(rows), "registered": len(good), "per_hover": rows,
           "top_err_visual_mm": stats,
           "top_err_collision_mm": _stats([e["err_collision_mm"] for e in good if e.get("err_collision_mm") is not None]),
           "within_2sigma": _share([v <= 2.0 for v in z]), "within_3sigma": _share([v <= 3.0 for v in z]),
           "sigma_le_p6": _share([e.get("p6_ok") for e in good]),
           "sigma_h_mm": _stats([e["sigma_h_mm"] for e in good if e.get("sigma_h_mm") is not None]),
           "envelope_violations": [e["hover"] for e in good if e.get("envelope_ok") is False],
           "sigma_above_envelope": [e["hover"] for e in good if e.get("envelope_class") == "sigma_above_envelope"],
           "width_within_tol": _share([e.get("within_3mm_15pct") for e in delivered]),
           "w_min_err_mm": _stats([e["w_min_err_mm"] for e in good if e.get("w_min_err_mm") is not None]),
           "width": width, "view_memory_modes": {m: sum(e["view_memory"] == m for e in rows)
                                                 for m in ("live", "emulated", "final")},
           "f_hat_rel_error": _stats(f_hats), "lower_bound_tops": sum(e.get("top_bound") == "lower" for e in rows),
           "second_hovers": sum(bool(e["stages"]) and "br1" in e["stages"] for e in rows),
           "two_stage_hovers": sum(bool(e["stages"]) and "second_stage" in e["stages"] for e in rows)}
    e1 = not width["e1_collapsed_widths_delivered"]
    # lead ruling 6 (r2close, RATIFIED): e2's share is over the hovers that DELIVERED a width (the judge's
    # denominator); collapses are gated by e1 and registration by (a). The share over the registered hovers (the
    # literal decision 2 §4.2 text, review R-10) is reported beside it.
    width["e2_denominator"] = "delivered"
    width["e2_pass_registered"] = (width["in_g13_band_share_of_registered"] or 0.0) >= G1P_E2_SHARE - 1e-9
    width["e2_pass_delivered"] = (width["in_g13_band_share_of_delivered"] or 0.0) >= G1P_E2_SHARE - 1e-9
    if grip and free:
        e_ok = e1 and width["e2_pass_delivered"] and not width["e3_g13plus_false_trips"]
        width["e_width_if_registered_denominator"] = e1 and width["e2_pass_registered"] \
            and not width["e3_g13plus_false_trips"]
    elif grip:
        e_ok = e1
    else:
        e_ok = None
    checks = {
        "a_registered": len(good) >= G1P_REGISTERED_MIN and len(rows) >= G1_MIN_MEASUREMENTS,
        "b_height": bool(stats["n"]) and stats["median_abs"] <= G1_MEDIAN_MM and stats["p90_abs"] <= G1_P90_MM,
        "c_honesty": bool(z) and out["within_2sigma"] >= G1P_2SIGMA_SHARE and out["within_3sigma"] >= G1P_3SIGMA_SHARE
        and (out["sigma_le_p6"] or 0.0) >= G1P_P6_SHARE,
        "d_envelope": bool(good) and not out["envelope_violations"]
        and all(e.get("envelope_ok") is not None or e.get("envelope_class") == "sigma_above_envelope" for e in good),
        "e_width": e_ok,
        "f_camera": bool(f_hats) and max(abs(v) for v in f_hats) <= F_HAT_TOL}
    out["checks"] = checks
    out["pass"] = all(v is not False for v in checks.values()) and all(v is not None for k, v in checks.items()
                                                                        if k != "e_width")
    out["rules"] = {"a": f">= {G1P_REGISTERED_MIN}/{G1_MIN_MEASUREMENTS} registered and not partial",
                    "b": f"|top_z - visual top| median <= {G1_MEDIAN_MM} mm, p90 <= {G1_P90_MM} mm (episode reference)",
                    "c": "|dh| <= 2 sigma_h in >= 90 %, <= 3 sigma_h in >= 98 %, sigma_h <= P6 limit in >= 90 %",
                    "d": "0 violations of top_z - collision top in [-(14.4 - m_t), 8 + m_t] mm, m_t = max(2, 1.5 s); "
                         "a hover with m_t >= 14.4 mm (sigma_h >= 9.6 mm) is 'sigma_above_envelope' (an empty palm "
                         "side; G1'(c) and P6 refuse it), listed, not a violation",
                    "e": "decision 2 §4.2 on the held record (truth: the collision cross-section, narrowest jaw yaw, "
                         "at the record's own grip point (decision 2 §5; 'anchor' on runs recorded without the "
                         "ORACLE cell grid); gated share over the hovers that delivered a width (lead ruling 6), the "
                         "registered share reported): "
                         "e1 grip parts: 0 delivered widths from a collapsed region; e2 free grip parts: |w - truth| <= "
                         "max(G13_FLOOR_MM, G13_SIGMA_K sigma_w) in >= 90 % of the delivered hovers; e3 free grip "
                         "parts: 0 hovers with truth - w > that band + PAD_LENGTH_R0_MM sin(YAW_GRID_RAD / 2); e4 "
                         "fixture grip parts: reported (G1c gates their consumers); non-grip parts: reported (None); "
                         "the max(3 mm, 15 %) share and the visible surface within 2 / 5 / 10 mm reported",
                    "f": "f-hat within 5 % of 333.6 in every episode"}
    return out


def _scored_index(h):
    stages = h.get("stages") or []
    info = h.get("scored_stage") or {}
    return info.get("index", len(stages) - 1) if stages else None


def _axis_tally(pairs):
    """{accepted, wrong (> YAW_GRID_RAD / 2), axis_err_deg stats} of [(label, record dict)]."""
    accepted = [(label, f) for label, f in pairs if f.get("elongated")]
    wrong = [(label, round(f.get("axis_err_deg"), 2)) for label, f in accepted
             if f.get("axis_err_deg") is not None and f["axis_err_deg"] > math.degrees(YAW_GRID_RAD / 2.0)]
    return {"accepted": len(accepted), "wrong": wrong,
            "axis_err_deg": _stats([f.get("axis_err_deg") for _, f in accepted])}


def g_yaw(records, pid):
    """Decision §4 G-yaw for an elongated grip part on the record the scene HOLDS after the hover (decision 2 §4.1:
    S-12's view memory included; ``held_record``). Reported beside it: ``as_run`` (each hover's recorded final record:
    for a run recorded before S-12 the scene's record without view memory, the judge's "strict"), the view-memory
    decisions, and the older diagnoses (``diagnosis_keep_earlier_axis``; ``diagnosis_tensor_variants`` = the retired
    S-4' tensor variants when a run recorded them)."""
    hovers = [(hover_label(r, h), h) for r in records for h in r["hovers"] if h["pid"] == pid]
    finals = [(label, held_record(h)[0]) for label, h in hovers]
    accepted = [(label, f) for label, f in finals if f.get("elongated")]
    wrong = [(label, f.get("axis_err_deg")) for label, f in accepted
             if f.get("axis_err_deg") is not None and f["axis_err_deg"] > math.degrees(YAW_GRID_RAD / 2.0)]
    refused = [label for label, f in finals if not f.get("elongated") and f.get("type") == "PartRecord"]
    lateral = [f.get("lateral") or {} for _, f in finals if f.get("type") == "PartRecord"]
    exact = [abs(v["exact_mm"]) for v in lateral if v.get("exact_mm") is not None]
    keep = []
    for label, h in hovers:
        chosen = h.get("final") or {}
        if not chosen.get("elongated"):
            for stage in reversed((h.get("stages") or [])[:(_scored_index(h) or 0) + 1]):
                if (stage.get("record") or {}).get("elongated"):
                    chosen = stage["record"]
                    break
        keep.append((label, chosen))
    variants = {}
    for scope in ("scored", "all_stages"):
        tally = {}
        for label, h in hovers:
            stages = h.get("stages") or []
            if scope == "scored":
                index = _scored_index(h)
                stages = [] if index is None else [stages[index]]
            for stage in stages:
                for name, v in ((stage.get("axis_variants") or {}).get("variants") or {}).items():
                    t = tally.setdefault(name, {"n": 0, "accepted": 0, "wrong": [], "axis_err_deg": []})
                    t["n"] += 1
                    t["accepted"] += bool(v.get("accepted"))
                    t["axis_err_deg"].append(v.get("axis_err_deg"))
                    if v.get("wrong"):
                        t["wrong"].append((f"{label}:{stage.get('label')}", round(v.get("deciding_err_deg"), 1)))
        variants[scope] = {name: {"n": t["n"], "accepted": t["accepted"], "wrong": t["wrong"],
                                  "axis_err_deg": _stats(t["axis_err_deg"])} for name, t in sorted(tally.items())}
    decisions = {}
    for _, f in finals:
        key = str((f.get("view_memory") or {}).get("decision") or _check_of(f).get("reason")
                  or _check_of(f).get("source"))
        decisions[key] = decisions.get(key, 0) + 1
    stage_checks = [_check_of(st.get("record")) for _, h in hovers for st in h.get("stages") or []
                    if (st.get("record") or {}).get("type") == "PartRecord"]
    out = {"hovers": len(finals), "accepted": len(accepted), "refused": refused, "wrong_accepted": wrong,
           "axis_err_deg": _stats([f.get("axis_err_deg") for _, f in accepted]),
           "lateral_exact_mm": _stats(exact),
           "lateral_pipeline_mm": _stats([v.get("pipeline_mm") for v in lateral]),
           "lateral_pm8_mm": _stats([v.get("pm8_mm") for v in lateral]),
           "lateral_pm16_mm": _stats([v.get("pm16_mm") for v in lateral]),
           "view_memory_modes": {m: sum(held_record(h)[1] == m for _, h in hovers) for m in ("live", "emulated",
                                                                                              "final")},
           "decisions": decisions, "as_run": _axis_tally([(label, h.get("final") or {}) for label, h in hovers]),
           "stage_acceptance": {"registered_stages": len(stage_checks),
                                "accepted": sum(bool(c.get("accepted")) for c in stage_checks),
                                "reasons": {str(k): sum(str(c.get("reason")) == str(k) for c in stage_checks)
                                            for k in sorted({str(c.get("reason")) for c in stage_checks})}},
           "diagnosis_keep_earlier_axis": _axis_tally(keep), "diagnosis_tensor_variants": variants}
    checks = {"no_wrong_accepted": not wrong, "accepted": len(accepted) >= GYAW_ACCEPTED_MIN,
              "lateral_exact_p90": bool(exact) and float(np.percentile(exact, 90)) <= GYAW_LATERAL_P90_MM}
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


def rs4_calibration_bound_px(row):
    """Lead ruling 5: the calibration-class bound of one located reading, max(RS4_ACCURACY_PX, RS4_SIGMA_K sigma_loc)
    (RS4_ACCURACY_PX when the reading carries no sigma)."""
    sigma = row.get("sigma_px")
    return RS4_ACCURACY_PX if sigma is None else max(RS4_ACCURACY_PX, RS4_SIGMA_K * float(sigma))


def rs4_summary(records):
    """RS-4 as restated by decision 2 §4.3 on the hover locator readings (``locate_at_hover``: the settled hover
    view against the end of a HOVER_LOCATE_M vertical move, V2v): per yaw class (the hover's yaw from the
    calibration orientation; older runs: the episode's hover yaw), readings, located, errors against the ORACLE grip
    projection. Calibration yaw (|yaw| <= RS4_CALIBRATION_YAW_DEG): located in >= RS4_LOCATED_SHARE and every
    located reading within max(RS4_ACCURACY_PX, RS4_SIGMA_K sigma_loc) (lead ruling 5; the literal 4 px reading is
    reported); other yaws: 0 located readings beyond RS4_SIGMA_K sigma_loc."""
    classes = {}
    for r in records:
        for h in r["hovers"]:
            loc = h.get("locator")
            if not loc or "locate" not in loc:
                continue
            yaw = loc.get("yaw_from_calibration_deg")
            if yaw is None:
                yaw = record_hover_yaw_deg(r)
            if yaw is None:
                key = "unknown"
            else:
                key = "calibration" if abs(yaw) <= RS4_CALIBRATION_YAW_DEG else f"{int(round(yaw)):+d}"
            classes.setdefault(key, []).append({"hover": f"{h['pid']}:{hover_label(r, h)}", "yaw_deg": yaw,
                                                "error_px": loc.get("error_px"), "sigma_px": loc.get("sigma_px"),
                                                "v0_error_px": loc.get("v0_error_px"),
                                                "reason": (loc.get("locate") or {}).get("reason")})
    out = {}
    for key, rows in sorted(classes.items()):
        located = [row for row in rows if row["error_px"] is not None]
        errors = [row["error_px"] for row in located]
        beyond = [row["hover"] for row in located
                  if row["sigma_px"] is not None and row["error_px"] > RS4_SIGMA_K * float(row["sigma_px"])]
        entry = {"readings": len(rows), "located": len(located), "error_px": _stats(errors),
                 "within_accuracy": sum(e <= RS4_ACCURACY_PX for e in errors), "beyond_2sigma": beyond,
                 "refusals": {str(k): sum(row["reason"] == k for row in rows if row["error_px"] is None)
                              for k in sorted({row["reason"] for row in rows if row["error_px"] is None}, key=str)},
                 "v0_error_px": _stats([row["v0_error_px"] for row in rows])}
        if key == "calibration":
            enough = bool(rows) and len(located) >= RS4_LOCATED_SHARE * len(rows) - 1e-9
            # lead ruling 5 (r2close, RATIFIED): every located reading within max(RS4_ACCURACY_PX, RS4_SIGMA_K
            # sigma_loc) of the ORACLE grip projection (it resolves the held-out ep13 A 4.36 px reading at sigma 6 px)
            beyond_bound = [row["hover"] for row in located if row["error_px"] > rs4_calibration_bound_px(row) + 1e-9]
            entry["pass"] = enough and not beyond_bound
            entry["rule"] = (f">= {RS4_LOCATED_SHARE:.0%} located, every located reading within "
                             f"max({RS4_ACCURACY_PX} px, {RS4_SIGMA_K} sigma_loc) of the ORACLE grip projection "
                             "(lead ruling 5)")
            entry["beyond_bound"] = beyond_bound
            # the decision 2 §4.3 literal 4 px reading, reported (review R-5)
            entry["beyond_accuracy"] = [row["hover"] for row in located if row["error_px"] > RS4_ACCURACY_PX]
            entry["pass_literal_4px"] = enough and not entry["beyond_accuracy"]
        else:
            entry["pass"] = not beyond
            entry["rule"] = f"every reading refused or within {RS4_SIGMA_K} sigma_loc (0 accepted beyond)"
        out[key] = entry
    return out


def record_hover_yaw_deg(record):
    """An episode record's hover yaw (deg): its own ``hover_yaw_deg``, else its run manifest's (``reaggregate``
    attaches BG's ``hover_yaw_deg`` or the review-2 judge's ``yaw_deg`` as ``_manifest_hover_yaw_deg``), else None
    (unknown: never assumed standard)."""
    for key in ("hover_yaw_deg", "_manifest_hover_yaw_deg"):
        if record.get(key) is not None:
            return float(record[key])
    return None


def run_set(records):
    """Which set a run is (decision 2 §4.4 / §6.2; review-2 implementation review R-2): "generality" (fresh episodes
    16-20 or a turned hover yaw, |yaw| > RS4_CALIBRATION_YAW_DEG; M7, not entry-gating); "gate" only when every
    episode is in 1-5 and every record's hover yaw is known and within RS4_CALIBRATION_YAW_DEG (the RS-1 gate set);
    "unknown" when a record's yaw is unknown (neither the record nor its manifest says); "other" otherwise (e.g. the
    held-out episodes 11-15 at standard yaw)."""
    episodes = sorted({int(r.get("episode_index")) for r in records if r.get("episode_index") is not None})
    yaw_list = [record_hover_yaw_deg(r) for r in records]
    yaws = sorted({y for y in yaw_list if y is not None})
    unknown = not records or any(y is None for y in yaw_list)
    fresh = bool(episodes) and all(e in GENERALITY_EPISODES for e in episodes)
    turned = any(abs(y) > RS4_CALIBRATION_YAW_DEG for y in yaws)
    gate_episodes = bool(episodes) and all(e in GATE_EPISODES for e in episodes)
    if fresh or turned:
        role = "generality"
    elif unknown:
        role = "unknown"
    elif gate_episodes:
        role = "gate"
    else:
        role = "other"
    return {"episodes": episodes, "hover_yaw_deg": yaws, "role": role, "fresh_episodes": fresh,
            "turned_yaw": turned, "gate_episodes": gate_episodes, "yaw_unknown": unknown}


def g1c_offline(records, pid):
    rows = [h.get("g1c") for r in records for h in r["hovers"] if h["pid"] == pid and h.get("g1c")]
    planned = [g for g in rows if g.get("planned")]
    reasons = {}
    for g in rows:
        if not g.get("planned"):
            key = str(g.get("reason") or g.get("error"))
            reasons[key] = reasons.get(key, 0) + 1
    return {"replayed": len(rows), "planned": len(planned), "marginal": sum(bool(g.get("marginal")) for g in planned),
            "depth_rules": {rule: sum(g.get("depth_rule") == rule for g in planned)
                            for rule in sorted({g.get("depth_rule") for g in planned})},
            "jaw_err_deg": _stats([g.get("jaw_err_deg") for g in planned]),
            "jaw_err_over_7_5_deg": sum((g.get("jaw_err_deg") or 0.0) > 7.5 for g in planned),
            "refusals": reasons, "note": "diagnosis; G1c is RS-2 (BS + BP): live chain and truth envelope there"}


def sigma_vc_at_tops(records, parts):
    """Ruling 4a: sigma_vc at part TOPS = per part the median of (visual top - collision top) of the part's own top
    region (``part_top_truth`` at every final hover); the max |median| over the parts (and over the gated ones)."""
    per = {}
    for pid in parts:
        values = []
        for r in records:
            for h in r["hovers"]:
                top = h.get("part_top_truth") or {}
                if h["pid"] == pid and top.get("visual_top_z") is not None and top.get("collision_top_z") is not None:
                    values.append(1000.0 * (float(top["visual_top_z"]) - float(top["collision_top_z"])))
        per[pid] = _stats(values)
    medians = {pid: s.get("median") for pid, s in per.items() if s.get("n")}
    return {"per_part_visual_minus_collision_top_mm": per,
            "sigma_vc_at_tops_mm": max((abs(v) for v in medians.values()), default=None),
            "sigma_vc_at_tops_gated_mm": max((abs(v) for p, v in medians.items() if p in GATED), default=None),
            "rule": "ruling 4a: the at-top statistic (visual - collision top at the part's own top); SIGMA_VC_M = "
                    "4 mm (BP) unchanged"}


def compute_summary(records):
    """RS-8: ms per tracked frame (the SurveyRecorder's on_transition, image decoding included) and result() s. Review
    finding R6: the bound is gated on EVERY hover stage, at the probe tracker's operating feature count (nothing caps
    seeds + dense + relief + grid at 450); the stages at <= 450 and > 450 features are also reported apart."""
    stages = [s for r in records for h in r["hovers"] for s in h.get("stages") or [] if s.get("compute")]
    track = [s["compute"]["track_ms_per_frame"] for s in stages]
    over = [(s["compute"]["features"], round(s["compute"]["track_ms_per_frame"], 1)) for s in stages
            if s["compute"]["track_ms_per_frame"] > 40.0]
    calib = [(r.get("calibration") or {}).get("survey_compute") for r in records]
    low = [s["compute"]["track_ms_per_frame"] for s in stages if s["compute"]["features"] <= 450]
    high = [s["compute"]["track_ms_per_frame"] for s in stages if s["compute"]["features"] > 450]
    return {"stages": len(stages), "track_ms_per_frame": _stats(track),
            "decode_ms_per_frame": _stats([s["compute"].get("decode_ms_per_frame") for s in stages]),
            "track_plus_decode_ms_per_frame": _stats([s["compute"]["track_ms_per_frame"]
                                                      + (s["compute"].get("decode_ms_per_frame") or 0.0)
                                                      for s in stages]),
            "adaptive_search_stages": sum(bool(s["compute"].get("adaptive_search")) for s in stages),
            "track_ms_per_frame_at_le_450_features": _stats(low),
            "track_ms_per_frame_over_450_features": _stats(high),
            "result_s": _stats([s["compute"]["result_s"] for s in stages]),
            "register_s": _stats([s["compute"]["register_s"] for s in stages]),
            "features": _stats([s["compute"]["features"] for s in stages]),
            "stages_over_450_features": len(high),
            "over_40ms": over, "pass_40ms": bool(stages) and not over,
            "rule": "every stage <= 40 ms per tracked frame at its operating feature count (review R6)",
            "calibrate_plus": calib}


# review-2 implementation review R-7: two identical-code runs agree only to floating-point noise (1e-12 to 1e-4 relative
# in the start survey and the tracks: reduction-order noise, no timing enters a decision), which flipped 6 of 100
# hover outcomes on reported parts (episode 4, D / E) and none on A or C; the reported parts' numbers are therefore
# single-run numbers.
REPRODUCIBILITY_NOTE = {
    "gated_parts": list(GATED), "reported_parts_single_run": [p for p in PART_CLASS if p not in GATED],
    "note": "identical-code runs differ by floating-point noise that can flip a discrete decision; B / D / E counts "
            "are single-run numbers (two runs of the gate set: 94/100 hovers identical, all differences in episode 4 "
            "D / E; A and C identical)"}


def s15_summary(records):
    """Lead rulings 3 and 8 (r2close), per part: every S-15 seed check of the hover stages. ``refused``: the
    seed_outside_region refusals (gap, limit, gap - limit; ``after_repoint`` for the re-pointed seed's own refusal);
    ``admitted_outside``: registered stages whose seed lay outside its region within the limit (the scene's
    ``seed_check``; runs recorded before it carry none); ``repointed``: the stages that re-pointed and the outcomes;
    ``near_limit``: every gap within S15_NEAR_LIMIT_PX of its limit, refused or admitted (floating-point noise between
    the two required runs can flip it)."""
    per = {}
    for r in records:
        for h in r.get("hovers") or []:
            entry = per.setdefault(h["pid"], {"stages": 0, "seed_checks": 0, "refused": [], "admitted_outside": [],
                                              "repointed": [], "repoint_outcomes": {}, "near_limit": []})
            for st in h.get("stages") or []:
                entry["stages"] += 1
                label = f"{hover_label(r, h)}:{st.get('label')}"
                checks = []
                repoint = st.get("repoint")
                if repoint:
                    entry["repointed"].append(label)
                    key = str(repoint.get("result"))
                    entry["repoint_outcomes"][key] = entry["repoint_outcomes"].get(key, 0) + 1
                    checks.append(("refused", repoint.get("refusal") or {}, False))
                rec = st.get("record") or {}
                evidence = rec.get("evidence") or {}
                if rec.get("type") == "RegistrationFailure" and evidence.get("why") == "seed_outside_region":
                    checks.append(("refused", evidence, bool(repoint)))
                seed_check = (rec.get("event") or {}).get("seed_check")
                if rec.get("type") == "PartRecord" and seed_check:
                    entry["seed_checks"] += 1
                    if not seed_check.get("inside"):
                        checks.append(("admitted_outside", seed_check, bool(repoint)))
                for kind, check, after in checks:
                    gap, limit = check.get("gap_px"), check.get("limit_px")
                    if gap is None or limit is None:
                        continue
                    row = {"stage": label, "gap_px": float(gap), "limit_px": float(limit),
                           "excess_px": float(gap) - float(limit), "after_repoint": after}
                    entry[kind].append(row)
                    if abs(row["excess_px"]) <= S15_NEAR_LIMIT_PX + 1e-12:
                        entry["near_limit"].append(dict(row, kind=kind))
    for entry in per.values():
        entry["rule"] = (f"gaps within {S15_NEAR_LIMIT_PX} px of the S-15 limit are flagged (lead ruling 8); the limit "
                         "is q_now + S15_SEED_SIGMA_K TRACK_SIGMA_PX (lead ruling 3)")
    return per


def edges_bimodal_summary(records):
    """Lead ruling 4 (a) (r2close), per part: the registered stages whose D10-OE refusal came from a gated but bimodal
    agreeing edge ROI (the scene's ``d10_oe.edges_bimodal``) and the hovers whose held record carries it; runs
    recorded before the rule carry no flag (0)."""
    per = {}
    for r in records:
        for h in r.get("hovers") or []:
            entry = per.setdefault(h["pid"], {"registered_stages": 0, "refusals": [], "held": []})
            for st in h.get("stages") or []:
                rec = st.get("record") or {}
                if rec.get("type") != "PartRecord":
                    continue
                entry["registered_stages"] += 1
                if ((rec.get("event") or {}).get("d10_oe") or {}).get("edges_bimodal"):
                    entry["refusals"].append(f"{hover_label(r, h)}:{st.get('label')}")
            if _check_of(held_record(h)[0]).get("edges_bimodal"):
                entry["held"].append(hover_label(r, h))
    for entry in per.values():
        entry["count"] = len(entry["refusals"])
    return per


def aggregate(records, parts):
    g1p = {pid: g1_prime(records, pid) for pid in parts}
    rs4 = rs4_summary(records)
    gyaw = {pid: g_yaw(records, pid) for pid in parts if pid in ELONGATED}
    g1c = {pid: g1c_offline(records, pid) for pid in parts if any(h.get("g1c") for r in records for h in r["hovers"]
                                                                  if h["pid"] == pid)}
    gates = {pid: {"g1_prime": g1p[pid]["pass"], "g_yaw": gyaw[pid]["pass"] if pid in gyaw else None}
             for pid in parts}
    rs1 = all(g1p[pid]["pass"] and gyaw.get(pid, {}).get("pass", False) for pid in GATED if pid in parts)
    return {"rs1_pass_A_C": rs1 if all(pid in parts for pid in GATED) else None, "gates": gates,
            "set": run_set(records), "rs4": rs4, "reproducibility": REPRODUCIBILITY_NOTE,
            "g1_prime": g1p, "g_yaw": gyaw, "g1c_offline": g1c, "sigma_vc": sigma_vc_at_tops(records, parts),
            "r2close": {"s15": s15_summary(records), "edges_bimodal": edges_bimodal_summary(records),
                        "rules": "lead rulings 3 (S-15 2 sigma + one re-point), 4 (a) (edge second-peak gate), 5 "
                                 "(RS-4 max(4 px, 2 sigma_loc)), 6 (e2 over delivered hovers), 8 (every real-sim "
                                 "gate run twice: ``pair_report``)"},
            "compute": compute_summary(records), "calibrate_plus": calibrate_plus_summary(records),
            "raw_sample_diagnosis": raw_sample_diagnosis(records, parts)}


CALIB_SAMPLES_CITED = (110, 121)            # decision §2.3 / remedy 1 G-1: loop lens 111-121, E5 110-119 per episode


def calibrate_plus_summary(records):
    """Review finding R10: the Calibrate+ start-survey depth samples per episode against the 110-121 the decision
    cites (decision §2.3 G-1, remedy 1 E5), with the grid placement and f-hat of each episode."""
    rows = []
    for r in records:
        c = r.get("calibration") or {}
        compute = c.get("survey_compute") or {}
        rows.append({"episode": r.get("episode_index"), "samples": compute.get("samples"),
                     "features": compute.get("features"), "grid_at_reference": compute.get("grid_at_reference"),
                     "grid": compute.get("grid"), "f_hat_rel_error": c.get("f_hat_rel_error_vs_R3")})
    samples = [row["samples"] for row in rows if row["samples"] is not None]
    return {"episodes": rows, "samples": _stats(samples), "cited_range": list(CALIB_SAMPLES_CITED),
            "within_cited_range": sum(CALIB_SAMPLES_CITED[0] <= v <= CALIB_SAMPLES_CITED[1] for v in samples),
            "below_cited_range": sum(v < CALIB_SAMPLES_CITED[0] for v in samples)}


def raw_sample_diagnosis(records, parts):
    """The earlier G1 statistics on the RAW samples of every final hover survey (decision §4: kept as a diagnosis):
    on-part p95 tops against the episode reference, z-scores, texture, edges, the HandLocator, f-hat, region B."""
    g1, sigma_calibration = {}, {}
    for pid in parts:
        cls = PART_CLASS[pid]
        table, measured, references = [], [], {}
        for r in records:
            reference = episode_reference(r, pid)
            references[r["episode_index"]] = reference
            for h in r["hovers"]:
                if h["pid"] != pid:
                    continue
                m = hover_measurement(h.get("rows") or [], cls, reference)
                entry = {"hover": hover_label(r, h), "n": 0 if m is None else m["n"],
                         "n_truth_robot": 0 if m is None else m["n_truth_robot"],
                         "measured": bool(m and m["measured"]),
                         "err_top_visual_mm": None if m is None else m["err_top_visual_mm"]}
                table.append(entry)
                if m and m["measured"]:
                    measured.append(m)
        hovers = [h for r in records for h in r["hovers"] if h["pid"] == pid]
        visual = [m["err_top_visual_mm"] for m in measured if m["err_top_visual_mm"] is not None]
        collision = [m["err_top_collision_mm"] for m in measured if m["err_top_collision_mm"] is not None]
        at_xy = [m["err_top_visual_at_xy_mm"] for m in measured if m.get("err_top_visual_at_xy_mm") is not None]
        median_at_xy = [m["err_median_visual_at_xy_mm"] for m in measured
                        if m.get("err_median_visual_at_xy_mm") is not None]
        samples = [row for h in hovers for row in h.get("rows") or [] if row["class"] == cls and truth_ok(row)]
        sample_err = [1000.0 * (row["z"] - row["visual_z"]) for row in samples]
        zscores = [(row["z"] - row["visual_z"]) / row["sigma_z"] for row in samples if row["sigma_z"] > 0]
        vc = [1000.0 * (row["visual_z"] - row["collision_z"]) for row in samples if row["collision_z"] is not None]
        tex = [h.get("texture") or {} for h in hovers]
        visual_stats = _stats(visual)
        bounds_met = bool(visual_stats["n"]) and visual_stats["median_abs"] <= G1_MEDIAN_MM \
            and visual_stats["p90_abs"] <= G1_P90_MM
        robot_truths = sum(row.get("visual_z") is not None and not truth_ok(row)
                           for h in hovers for row in h.get("rows") or [] if row["class"] == cls)
        g1[pid] = {"class": cls, "hovers": len(hovers), "measurements": len(measured),
                   "unmeasured": [e["hover"] for e in table if not e["measured"]],
                   "per_hover": table,
                   "top_z_err_visual_mm": visual_stats, "top_z_err_collision_mm": _stats(collision),
                   "top_z_rule": "per hover with >= HOVER_CLASS_MIN_SAMPLES on-part samples: 95th percentile of the "
                                 "valid on-part samples (spec §4.5 registration step 4) against the episode's FIXED "
                                 "reference (episode_reference: the part's own visual top from ORACLE geometry; "
                                 "fallback for runs recorded before it: the max non-robot on-part visual truth)",
                   "references": references,
                   "top_z_err_visual_at_measured_xy_mm": _stats(at_xy),
                   "median_z_err_visual_at_measured_xy_mm": _stats(median_at_xy),
                   "sample_err_visual_mm": _stats(sample_err), "samples_truth_robot_dropped": robot_truths,
                   "zscore": {"n": len(zscores), "std": float(np.std(zscores)) if zscores else None,
                              "robust_std": geometry._robust_sigma(zscores) if zscores else None},
                   "visual_minus_collision_mm": _stats(vc),
                   "grid_pass_rate_D15": _ratio(sum(t.get("grid_pass_D15", 0) for t in tex),
                                                sum(t.get("grid_patches", 0) for t in tex)),
                   "features_valid_rate_D1": _ratio(sum(t.get("features_valid_D1", 0) for t in tex),
                                                    sum(t.get("features_on_part", 0) for t in tex)),
                   "edge_vs_interior": {
                       "edge_err_mm": _stats([e for h in hovers for e in _edge_values(h, "edge")]),
                       "interior_err_mm": _stats([e for h in hovers for e in _edge_values(h, "interior")])},
                   "g1_bounds_met": bounds_met,
                   "g1_status": ("not_run" if not hovers else
                                 "pass" if bounds_met and len(visual) >= G1_MIN_MEASUREMENTS else
                                 "insufficient_measurements" if bounds_met else "fail"),
                   "g1_pass": bounds_met and len(visual) >= G1_MIN_MEASUREMENTS,
                   "note": "raw samples (diagnosis); the gate is g1_prime on the registered PartRecord"}
    by_class, all_rows = {}, {}
    for r in records:
        for h in r["hovers"]:
            for row in h.get("rows") or []:
                all_rows.setdefault(row["class"], []).append(row)
            for cls in {row["class"] for row in h.get("rows") or []}:
                m = hover_measurement(h.get("rows") or [], cls)
                if m and m["measured"]:
                    by_class.setdefault(cls, []).append(m)
    classes = {cls: {"hover_measurements": len(ms),
                     "top_z_err_visual_at_measured_xy_mm": _stats([m.get("err_top_visual_at_xy_mm") for m in ms]),
                     "median_z_err_visual_at_measured_xy_mm": _stats([m.get("err_median_visual_at_xy_mm")
                                                                      for m in ms])}
               for cls, ms in sorted(by_class.items())}
    for cls, rows in sorted(all_rows.items()):
        usable = [row for row in rows if truth_ok(row) and row["sigma_z"] > 0]
        if len(usable) < HOVER_CLASS_MIN_SAMPLES:
            continue
        errors = [row["z"] - row["visual_z"] for row in usable]
        sigmas = [row["sigma_z"] for row in usable]
        edge = [row for row in usable if row.get("edge_like") is True]
        texture = [row for row in usable if row.get("edge_like") is False]
        sigma_calibration[cls] = {
            "n": len(usable), "robust_zscore_std": geometry._robust_sigma(np.divide(errors, sigmas)),
            "sigma_z_median_mm": 1000.0 * float(np.median(sigmas)),
            "robust_err_mm": 1000.0 * geometry._robust_sigma(errors),
            "model_sigma_for_unit_zscore_mm": 1000.0 * _model_sigma_for_unit_zscore(errors, sigmas),
            "robust_zscore_std_with_D4_model": geometry._robust_sigma(
                np.divide(errors, np.hypot(sigmas, geometry.D4_MODEL_SIGMA_M))),
            "edge_like": {"n": len(edge), "model_sigma_for_unit_zscore_mm": 1000.0 * _model_sigma_for_unit_zscore(
                [row["z"] - row["visual_z"] for row in edge], [row["sigma_z"] for row in edge])} if edge else None,
            "textured": {"n": len(texture), "model_sigma_for_unit_zscore_mm": 1000.0 * _model_sigma_for_unit_zscore(
                [row["z"] - row["visual_z"] for row in texture], [row["sigma_z"] for row in texture])}
            if texture else None}
    f_hats = [r["calibration"]["f_hat_rel_error_vs_R3"] for r in records if r["calibration"]["f_hat_rel_error_vs_R3"]
              is not None]
    locator = [r["calibration"]["locator"].get("max_error_px") for r in records]
    locator_used = [locator_used_moves(r["calibration"]) for r in records]
    hover_loc = hover_locator_summary(records)
    sigma_vc_parts = {pid: g1[pid]["visual_minus_collision_mm"].get("median_abs") for pid in parts}
    anchors = {pid: [r["calibration"]["anchors"][pid] for r in records if pid in (r["calibration"].get("anchors") or {})]
               for pid in parts}
    anchor_vc = {pid: _median([1000.0 * (a["visual_wrist_z"] - a["collision_z"]) for a in rows
                               if a["visual_wrist_z"] is not None and a["collision_z"] is not None])
                 for pid, rows in anchors.items()}
    reference_vc = {pid: _median([1000.0 * (ref["visual_top_z"] - ref["collision_top_z"])
                                  for ref in g1[pid]["references"].values()
                                  if ref.get("visual_top_z") is not None and ref.get("collision_top_z") is not None])
                    for pid in parts}
    candidates = [abs(v) for v in sigma_vc_parts.values() if v is not None]
    start = {}
    for r in records:
        for cls, entry in r["calibration"]["start_survey_classes"].items():
            start.setdefault(cls, []).append(entry)
    calibration_locator = [r["calibration"]["locator"] for r in records]
    probes = {}
    for r in records:
        for h in r["hovers"]:
            probe = h.get("probe")
            if probe is None:
                key = "not_recorded (pre-fix run: back-to-back v1 legs)"
            else:
                stopped = [leg["stage"] for leg in probe.get("legs") or []
                           if leg.get("reason") not in ("travel_reached", "nothing_to_do")]
                key = f"{probe.get('reason')}" + (f" ({', '.join(stopped)})" if stopped else "")
            probes.setdefault(h["pid"], {}).setdefault(key, 0)
            probes[h["pid"]][key] += 1
    return {"g1": g1, "classes_all_hovers": classes, "hover_probe_outcomes": probes, "sigma_calibration": {
                "depth_D4": sigma_calibration,
                "D4_MODEL_SIGMA_M": geometry.D4_MODEL_SIGMA_M,
                "locator_L1": {"sigma_loc_px": [c.get("sigma_loc_px") for c in calibration_locator],
                               "sigma_consistency_px": [c.get("sigma_consistency_px") for c in calibration_locator],
                               "calibration_abs_err_px": _stats([row["error_px"] for u in locator_used
                                                                 for row in u["rows"]]),
                               "calibration_abs_err_rms_px": _rms([row["error_px"] for u in locator_used
                                                                   for row in u["rows"]]),
                               "hover_abs_err_px": hover_loc.get("located_err_px"),
                               "hover_abs_err_rms_px": hover_loc.get("located_err_rms_px"),
                               "L1_MODEL_SIGMA_PX": geometry.L1_MODEL_SIGMA_PX}},
            "f_hat_rel_error": _stats(f_hats), "f_hat_within_5pct": bool(f_hats) and max(map(abs, f_hats)) <= F_HAT_TOL,
            "locator_calibration_max_px": locator, "locator_hover": hover_loc,
            "locator_used_moves": locator_used,
            "locator_within_6px_at_calibration": all(u["max_error_px"] is not None and
                                                     u["max_error_px"] <= LOCATOR_TOL_PX for u in locator_used),
            "locator_within_6px_at_hovers": hover_loc.get("within_6px"),
            "external": {"v_hat_angle_deg": [r["calibration"]["external_truth"]["v_hat_angle_deg"] for r in records],
                         "ray_angle_deg": [r["calibration"]["external_truth"]["ray_angle_deg"] for r in records],
                         "J_rel_error": [r["calibration"]["external_truth"]["J_rel_error"] for r in records],
                         "quarter_turns": [r["calibration"]["quarter_turns"] for r in records]},
            "region_B": [r["calibration"]["region_B"] for r in records],
            "start_survey": {cls: {"episodes": len(v), "z_measured_median": _median([e["z_measured_median"] for e in v]),
                                   "z_visual_median": _median([e["z_visual_median"] for e in v]),
                                   "err_visual_median_abs_mm": _median([e["err_visual_mm"].get("median_abs") for e in v]),
                                   "samples": sum(e["n"] for e in v)} for cls, v in start.items()},
            "sigma_vc": {"per_part_median_abs_visual_minus_collision_mm": sigma_vc_parts,
                         "at_anchor_visual_minus_collision_mm": anchor_vc,
                         "at_reference_top_visual_minus_collision_mm": reference_vc,
                         "sigma_vc_mm": max(candidates) if candidates else None,
                         "rule": "max over the surveyed robot-reachable parts of the median |visual - collision| at "
                                 "the measured on-part xy (robot-owned truths dropped; Appendix L decision 11); DD, "
                                 "re-derive at V5"}}


def _rms(values):
    values = [float(v) for v in values if v is not None]
    return math.sqrt(float(np.mean(np.square(values)))) if values else None


def hover_locator_summary(records):
    """HandLocator at the hovers. Runs from this version: located on the HOVER_LOCATE_M vertical move (errors
    against the ORACLE grip projection), the 1-step pair's refusals, and the sweep binned by the along-v_hat motion.
    Older runs located on the 1-step pair only (reported as such)."""
    hovers = [h for r in records for h in r["hovers"] if h.get("locator")]
    new = [h["locator"] for h in hovers if "locate" in h["locator"]]
    if not new:
        old = [h["locator"]["error_px"] for h in hovers if h["locator"].get("error_px") is not None]
        return {"pair": "one_step (pre-fix runs: the locator then had no preconditions)", "located_err_px": _stats(old),
                "located_err_rms_px": _rms(old), "within_6px": bool(old) and max(old) <= LOCATOR_TOL_PX}
    errors = [loc["error_px"] for loc in new if loc.get("error_px") is not None]
    by_pid = {}
    for h in hovers:
        if "locate" in h["locator"] and h["locator"].get("error_px") is not None:
            by_pid.setdefault(h["pid"], []).append(h["locator"]["error_px"])
    refusals = {}
    for loc in new:
        reason = ((loc.get("one_step") or {}).get("locate") or {}).get("reason")
        refusals[str(reason)] = refusals.get(str(reason), 0) + 1
    bins = {}
    for loc in new:
        for row in loc.get("sweep") or []:
            if row.get("error_px") is None:
                continue
            key = int(abs(row["along_px"]) // 4) * 4
            bins.setdefault(key, []).append(row["error_px"])
    return {"pair": f"settled hover frame vs the end of a {HOVER_LOCATE_M:.2f} m vertical move",
            "located_err_px": _stats(errors), "located_err_rms_px": _rms(errors),
            "located": len(errors), "refused": sum(loc.get("error_px") is None for loc in new),
            "refused_reasons": sorted({str((loc.get("locate") or {}).get("reason")) for loc in new
                                       if loc.get("error_px") is None}),
            "by_part_px": {pid: _stats(v) for pid, v in sorted(by_pid.items())},
            "one_step_pair_reasons": refusals,
            "sweep_err_by_along_px": {f"{k}-{k + 4}": _stats(v) for k, v in sorted(bins.items())},
            "within_6px": bool(errors) and max(errors) <= LOCATOR_TOL_PX}


def locator_used_moves(calibration):
    """The HandLocator's errors on the probe moves it locates on (its preconditions: image motion within
    L1_MAX_ACROSS_DEG of v_hat and at least L1_MIN_ALONG_PX along it); the other moves are reported in the rows but
    not used by the locator."""
    external = calibration.get("external") or {}
    if not external.get("J"):
        return {"max_error_px": None, "rows": []}
    J = np.asarray(external["J"], dtype=float)
    n_hat = np.asarray(external["n_hat"], dtype=float)
    v_hat = np.asarray(external["v_hat"], dtype=float)
    limit = math.sin(math.radians(geometry.L1_MAX_ACROSS_DEG))
    by_stage = {s["stage"]: s for s in calibration["external_samples"]}
    rows = []
    for row in calibration["locator"].get("rows", []):
        sample = by_stage.get(row["stage"])
        if sample is None or row.get("error_px") is None:
            continue
        m = J @ (np.asarray(sample["end_state8"][:3]) - np.asarray(sample["start_state8"][:3]))
        if float(np.linalg.norm(m)) > 0 and abs(float(m @ n_hat)) <= limit * float(np.linalg.norm(m)) \
                and abs(float(m @ v_hat)) >= geometry.L1_MIN_ALONG_PX:
            rows.append({"stage": row["stage"], "error_px": row["error_px"], "along_px": float(m @ v_hat)})
    errors = [r["error_px"] for r in rows]
    return {"max_error_px": max(errors) if errors else None, "rows": rows}


def _edge_values(hover, kind):
    return list((hover.get("edges") or {}).get(f"{kind}_values_mm") or [])


def _ratio(a, b):
    return a / b if b else None


# ----------------------------------------------------------------------------------------------- main
def build_parser():
    h = harness()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="libero_10_t2")
    parser.add_argument("--episodes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--parts", nargs="+", default=list(PART_CLASS), choices=list(PART_CLASS))
    parser.add_argument("--offsets", type=int, default=len(HOVER_OFFSETS_M), choices=range(0, 5))
    parser.add_argument("--hover", choices=("chain", "seeded"), default="chain",
                        help="chain (default, scored): BP hover_for on the measured start record (the pipeline "
                             "path); seeded: the ORACLE-seeded hover of the earlier survey (diagnosis only)")
    parser.add_argument("--no-locate", dest="locate", action="store_false",
                        help="skip the HandLocator diagnostic move at every hover")
    parser.add_argument("--no-br1", dest="br1", action="store_false", help="skip BR-1's second hover")
    parser.add_argument("--hover-yaw-deg", type=float, default=0.0,
                        help="turn every transit / hover orientation about world z by this many degrees after "
                             "Calibrate+ (decision 2 §4.3 RS-4 yaw classes, §4.4 generality +45 deg)")
    parser.add_argument("--no-visibility", dest="visibility", action="store_false",
                        help="diagnosis: no restated P-2 visibility rule (neither on the record nor on the pointing)")
    parser.add_argument("--dump", action="store_true", help="pickle every grip part's plan inputs (G1c replay)")
    parser.add_argument("--calib-grid", choices=("reference", "construction"), default="reference",
                        help="Calibrate+ grid placement: reference (default; the grid on the frame before probe_1, "
                             "outside the measured fingertip band) or construction (decision §2.3 G-1 as written; "
                             "review R10 A/B)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reaggregate", type=Path, default=None,
                        help="recompute report.json from every episode_*.json below a directory")
    parser.add_argument("--report-out", type=Path, default=None,
                        help="with --reaggregate: write the report here instead of <directory>/report.json (another "
                             "owner's run is never written to); with --pair: the pair report's path (required)")
    parser.add_argument("--pair", type=Path, nargs=2, default=None, metavar=("RUN_A", "RUN_B"),
                        help="lead ruling 8: compare two runs of the same gate on the same code (read only) and "
                             "report whether BOTH pass; needs --report-out")
    parser.add_argument("--allow-gate-episode", action="store_true",
                        help="permit episode 0 (the gate episode; out of G1' by default)")
    parser.add_argument("--sim-python", default=h.DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=h.DEFAULT_LIBERO_ROOT)
    return parser


def aggregation_provenance():
    """Which code computed a report (BG-R8): a re-aggregated report.json may differ from its run's manifest."""
    return {"script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": file_sha256(Path(__file__)),
            "survey_probe_version": SURVEY_PROBE_VERSION,
            "geometry_sha256": file_sha256(ROOT / "src/gpt_field/agent_speedup_v2_geometry.py"),
            "aggregated_utc": utc_stamp()}


def _unpack_bits(packed):
    shape = tuple(packed["shape"])
    return np.unpackbits(packed["bits"])[:shape[0] * shape[1]].reshape(shape).astype(bool)


def attach_stage_pickles(record, directory):
    """Runs recorded before the width sigma and the visible surface were scored (the review-2 judge's live runs,
    ``judge_live.py``, which pickled every stage as ``stage_<pid>_ep<e>_<hover>_<label>.pkl`` beside the episode
    file): fill each stage record's ``widths_sigma_mm`` / ``w_min_sigma_mm`` from the pickled record and its
    ``visual_truth`` from the pickled ORACLE capture (scoring only), then the hover's ``final`` from its scored stage.
    Returns the number of stages filled."""
    directory = Path(directory)
    filled = 0
    episode = record.get("episode_index")
    geometry_truth = (record.get("calibration") or {}).get("geometry_truth") or {}
    for h in record.get("hovers") or []:
        offset = h.get("offset_m") or [0.0, 0.0]
        tag = f"{offset[0]:+.2f}_{offset[1]:+.2f}"
        for stage in h.get("stages") or []:
            rec = stage.get("record") or {}
            path = directory / f"stage_{h['pid']}_ep{episode}_{tag}_{stage.get('label')}.pkl"
            if rec.get("type") != "PartRecord" or not path.is_file():
                continue
            with open(path, "rb") as fh:
                pickled = pickle.load(fh)
            widths = (pickled.get("record") or {}).get("widths") or []
            if widths and "w_min_sigma_mm" not in rec:
                sigmas = {int(round(math.degrees(y))): 1000.0 * float(sw) for y, _, sw in widths}
                narrowest = min(widths, key=lambda item: item[1])
                rec["widths_sigma_mm"] = sigmas
                rec["w_min_sigma_mm"] = 1000.0 * float(narrowest[2])
            truth, axis = pickled.get("truth") or {}, (geometry_truth.get(h["pid"]) or {}).get("axis")
            xyz = (pickled.get("record") or {}).get("xyz")
            if "visual_truth" not in rec and truth.get("mask_wrist") and truth.get("depth_png") and axis is not None \
                    and xyz is not None:
                mask = _unpack_bits(truth["mask_wrist"])
                depth = _decode_png(truth["depth_png"]).astype(float) * float(truth["depth_unit"])
                projection = truth["axis_proj_wrist"]
                camera = _OracleCamera(projection["xyz"], projection, mask.shape[1])
                rec["visual_truth"] = {"available": True, "source": "stage_pickle",
                                       **visual_surface_widths(mask, depth, camera, xyz, axis)}
            stage["record"] = rec
            filled += 1
        index = _scored_index(h)
        if index is not None and (h.get("stages") or [])[index].get("record") and \
                (h.get("final") or {}).get("type") == "PartRecord":
            scored = h["stages"][index]["record"]
            for key in ("widths_sigma_mm", "w_min_sigma_mm", "visual_truth"):
                if key in scored and key not in h["final"]:
                    h["final"][key] = scored[key]
    return filled


def load_run(directory):
    """(records, manifests, episode paths) of a run directory: every episode_*.json below it (one sub-directory per
    episode process), stage pickles attached (``attach_stage_pickles``) and a missing hover yaw taken from the
    episode's manifest (review R-2). Read only."""
    directory = Path(directory)
    paths = sorted(directory.rglob("episode_*.json"), key=lambda p: (p.parent.name, p.name))
    records = [json.loads(p.read_text()) for p in paths]
    for path, record in zip(paths, records):
        record["_stage_pickles_attached"] = attach_stage_pickles(record, path.parent)
        manifest_path = path.parent / "manifest.json"
        if record.get("hover_yaw_deg") is None and manifest_path.is_file():
            # review R-2: a record without its hover yaw (the judge's runs) takes its run manifest's
            manifest = json.loads(manifest_path.read_text())
            yaw = manifest.get("hover_yaw_deg", manifest.get("yaw_deg"))
            if yaw is not None:
                record["_manifest_hover_yaw_deg"] = float(yaw)
    manifests = [json.loads(p.read_text()) for p in sorted(directory.rglob("manifest.json"))]
    return records, manifests, paths


def gate_verdicts(agg):
    """The discrete verdicts of one aggregate that lead ruling 8 compares run against run: RS-1 (A and C), every
    part's G1' and G-yaw pass and their checks, every RS-4 yaw class and RS-8 (40 ms). None = not applicable."""
    out = {"rs1_pass_A_C": agg.get("rs1_pass_A_C")}
    for gate in ("g1_prime", "g_yaw"):
        for pid, g in sorted((agg.get(gate) or {}).items()):
            out[f"{gate}.{pid}"] = g.get("pass")
            for check, value in sorted((g.get("checks") or {}).items()):
                out[f"{gate}.{pid}.{check}"] = value
    for key, entry in sorted((agg.get("rs4") or {}).items()):
        out[f"rs4.{key}"] = entry.get("pass")
    out["rs8.pass_40ms"] = (agg.get("compute") or {}).get("pass_40ms")
    return out


def hover_signature(h):
    """The discrete outcome of one hover's held record (ruling 8's hover-level comparison): registered, axis accepted,
    axis wrong (> YAW_GRID_RAD / 2), width delivered, the stages run."""
    held = held_record(h)[0]
    elongated = bool(held.get("elongated"))
    return {"registered": _final_ok(h), "axis_accepted": elongated,
            "axis_wrong": elongated and (held.get("axis_err_deg") or 0.0) > math.degrees(YAW_GRID_RAD / 2.0),
            "width_delivered": bool(held.get("widths_mm")) and not held.get("width_unmeasured"),
            "stages": list(h.get("flags") or [])}


def _run_sources(manifests):
    sources = {}
    for manifest in manifests:
        for path, sha in (manifest.get("sources") or {}).items():
            sources.setdefault(path, set()).add(sha)
    return {path: sorted(shas) for path, shas in sources.items()}


def pair_report(dir_a, dir_b, parts):
    """Lead ruling 8 (r2close): every real-sim gate of the slice is run TWICE on the same code and BOTH runs must
    pass. Aggregates both runs (read only; nothing is written into either), reports each run's verdicts, ``both_pass``
    per verdict (True only when both runs pass it; None when it applies to neither), RS-1 on both, the verdicts the
    runs disagree on, the hovers whose held-record outcome differs, and whether both ran the same sources (the
    manifests' sha256) on the same set."""
    runs = []
    for directory in (dir_a, dir_b):
        records, manifests, paths = load_run(directory)
        agg = aggregate(records, parts)
        runs.append({"directory": str(directory), "episode_files": len(paths), "records": records,
                     "manifests": manifests, "aggregate": agg, "verdicts": gate_verdicts(agg)})
    va, vb = runs[0]["verdicts"], runs[1]["verdicts"]
    keys = sorted(set(va) | set(vb))
    both = {k: None if va.get(k) is None and vb.get(k) is None else (va.get(k) is True and vb.get(k) is True)
            for k in keys}

    def index(records):
        return {(int(r.get("episode_index")), h["pid"], tuple(round(float(x), 4) for x in h.get("offset_m") or ())): h
                for r in records for h in r.get("hovers") or []}
    ia, ib = index(runs[0]["records"]), index(runs[1]["records"])
    differences = []
    for key in sorted(set(ia) | set(ib), key=str):
        sa = hover_signature(ia[key]) if key in ia else None
        sb = hover_signature(ib[key]) if key in ib else None
        if sa != sb:
            differences.append({"episode": key[0], "pid": key[1], "offset_m": list(key[2]), "a": sa, "b": sb})
    sources = [_run_sources(run["manifests"]) for run in runs]
    source_differences = sorted(p for p in set(sources[0]) | set(sources[1])
                                if sources[0].get(p) != sources[1].get(p) or len(sources[0].get(p) or ()) > 1)
    sets = [run["aggregate"]["set"] for run in runs]
    counts = []
    for run in runs:
        agg = run["aggregate"]
        counts.append({pid: {"registered": (agg["g1_prime"].get(pid) or {}).get("registered"),
                             "accepted": (agg["g_yaw"].get(pid) or {}).get("accepted"),
                             "wrong": (agg["g_yaw"].get(pid) or {}).get("wrong_accepted")} for pid in parts})
    gated = {pid: bool(both.get(f"g1_prime.{pid}")) and bool(both.get(f"g_yaw.{pid}", True))
             for pid in GATED if pid in parts}
    return {"rule": "lead ruling 8: every real-sim gate is run twice on the same code; both runs must pass",
            "runs": [{"directory": run["directory"], "episode_files": run["episode_files"],
                      "set": run["aggregate"]["set"], "verdicts": run["verdicts"], "counts": c,
                      "r2close": run["aggregate"].get("r2close")} for run, c in zip(runs, counts)],
            "both_pass": both, "rs1_both_pass": both.get("rs1_pass_A_C") is True, "gated_parts_both_pass": gated,
            "verdicts_disagree": [k for k in keys if va.get(k) != vb.get(k)],
            "hover_differences": differences, "hovers_compared": len(set(ia) | set(ib)),
            "same_code": not source_differences and all(sources), "source_differences": source_differences,
            "same_set": all(sets[0].get(k) == sets[1].get(k) for k in ("episodes", "hover_yaw_deg", "role")),
            "aggregated_by": aggregation_provenance()}


def reaggregate(directory, parts, report_out=None):
    """Recompute report.json from every episode_*.json below ``directory`` (one sub-directory per episode process;
    no simulator); the report records the aggregating script's sha (``aggregated_by``) and the manifests found.
    Stage pickles beside an episode file (the review-2 judge's runs) fill the width sigma and the visible surface
    (``attach_stage_pickles``)."""
    directory = Path(directory)
    records, manifests, paths = load_run(directory)
    report = {"manifests": manifests, "episode_files": [str(p.relative_to(directory)) for p in paths],
              "aggregated_by": {**aggregation_provenance(), "reaggregated": True},
              "aggregate": aggregate(records, parts)}
    target = directory / "report.json" if report_out is None else Path(report_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jsonable(report), indent=1))
    report["written_to"] = str(target)
    return report


def summary_line(report):
    agg = report["aggregate"]
    return jsonable({
        "rs1_pass_A_C": agg["rs1_pass_A_C"], "gates": agg["gates"],
        "g1_prime": {pid: {"registered": e["registered"], "hovers": e["hovers"], "checks": e["checks"],
                           "top_err_visual_mm": {k: e["top_err_visual_mm"].get(k) for k in ("n", "median_abs",
                                                                                            "p90_abs")}}
                     for pid, e in agg["g1_prime"].items()},
        "g_yaw": {pid: {"accepted": e["accepted"], "wrong": e["wrong_accepted"], "checks": e["checks"],
                        "max_err_deg": e["axis_err_deg"].get("max_abs"), "as_run": e["as_run"],
                        "view_memory_modes": e["view_memory_modes"],
                        "lateral_exact_p90_mm": e["lateral_exact_mm"].get("p90_abs")} for pid, e in agg["g_yaw"].items()},
        "g1_prime_e": {pid: {k: e["width"].get(k) for k in (
            "e_class", "registered", "delivered", "in_g13_band", "in_g13_band_share_of_registered",
            "in_g13_band_share_of_delivered", "e2_pass_registered", "e2_pass_delivered", "within_3mm_15pct",
            "e1_collapsed_widths_delivered", "e3_g13plus_false_trips", "collapsed_widths_delivered_as_run",
            "truth_at")} for pid, e in agg["g1_prime"].items()},
        "set": agg.get("set"), "rs4": {k: {kk: v[kk] for kk in ("readings", "located", "pass")}
                                       for k, v in (agg.get("rs4") or {}).items()},
        "sigma_vc_at_tops_mm": agg["sigma_vc"]["sigma_vc_at_tops_mm"],
        "compute": {k: agg["compute"][k] for k in ("track_ms_per_frame", "result_s", "features", "pass_40ms")},
        "calibrate_plus_samples": (agg.get("calibrate_plus") or {}).get("samples")})


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pair is not None:
        if args.report_out is None:
            raise SystemExit("--pair needs --report-out (neither run directory is written to)")
        report = pair_report(args.pair[0], args.pair[1], args.parts)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(jsonable(report), indent=1))
        print(json.dumps(jsonable({k: report[k] for k in ("rs1_both_pass", "gated_parts_both_pass", "verdicts_disagree",
                                                         "same_code", "same_set")}
                                  | {"hover_differences": len(report["hover_differences"]),
                                     "counts": [run["counts"] for run in report["runs"]]}), indent=1, default=str))
        return 0
    if args.reaggregate is not None:
        report = reaggregate(args.reaggregate, args.parts, report_out=args.report_out)
        print(json.dumps(summary_line(report), indent=1, default=str))
        return 0
    if 0 in args.episodes and not args.allow_gate_episode:
        raise SystemExit("episode 0 is the gate episode and stays out of G1' (pass --allow-gate-episode)")
    out = args.out or ROOT / "runs" / f"agent_speedup_v2_bg_r1_survey_{utc_stamp()}"
    if out.exists():
        raise SystemExit(f"output {out} exists; choose a fresh directory")
    out.mkdir(parents=True)
    manifest = {"version": SURVEY_PROBE_VERSION, "geometry_version": geometry.GEOMETRY_VERSION, "task": args.task,
                "hover": args.hover, "locate": args.locate, "br1": args.br1, "dump": args.dump,
                "visibility": args.visibility, "calib_grid": args.calib_grid, "hover_yaw_deg": args.hover_yaw_deg,
                "episodes": args.episodes, "parts": args.parts, "offsets": args.offsets, "created_utc": utc_stamp(),
                "zero_model_calls": True, "scored": args.hover == "chain", "diagnosis_only": args.hover != "chain",
                "policy_free": True, "pipeline_path": args.hover == "chain",
                "deviations": DEVIATIONS + ([] if args.hover == "chain" else
                                            ["hover seeded: the ORACLE-seeded hover (ruling 4d: diagnosis only)"]),
                "environment": {k: os.environ.get(k) for k in ("MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES",
                                                               "PYTHONHASHSEED", "OMP_NUM_THREADS",
                                                               "OPENBLAS_NUM_THREADS")},
                "sources": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
                    Path(__file__).resolve(), ORACLE_FILE, HARNESS_FILE,
                    ROOT / "src/gpt_field/agent_speedup_v2_geometry.py",
                    ROOT / "src/gpt_field/agent_speedup_v2_scene.py",
                    ROOT / "src/gpt_field/agent_speedup_v2_primitives.py",
                    ROOT / "src/gpt_field/agent_speedup_v2_planner.py",
                    ROOT / "src/gpt_field/agent_speedup_v2_types.py")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    records = []
    for episode_index in args.episodes:
        records.append(run_episode(args, episode_index, out))
        print(json.dumps({"episode": episode_index, "steps": records[-1].get("steps"),
                          "wall_s": records[-1]["wall_s"]}), flush=True)
    report = {"manifest": manifest, "aggregated_by": {**aggregation_provenance(), "reaggregated": False},
              "aggregate": aggregate(records, args.parts)}
    (out / "report.json").write_text(json.dumps(jsonable(report), indent=1))
    print(json.dumps(summary_line(report), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
