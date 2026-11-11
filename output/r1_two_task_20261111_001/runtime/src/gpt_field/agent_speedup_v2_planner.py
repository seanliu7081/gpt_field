"""agent_SpeedUp_v2 planner (contract §C7): pure plans and verdicts from MEASURED part geometry.

Inputs are kernel records only: the PartRecord (top, support, width per jaw yaw, elongation, D11 circle, region),
the HeightField, the RobotModel (finger landing footprints at the measured G1 opening, hand footprint, palm
underside), keep-out volumes and the measured sigmas. A part's text is never read. The move prior decides only the
palm margin (P6: 0 on fixtures; max(5 mm, 2 sigma_h) on free and unknown parts, [C-22]) and never enables a palm
stop (P13, [C-47]: palm_stop needs a verified-fixed DOF, reserved).

Contents:
- GraspPlanner (spec §4.6): candidates = family x jaw yaw (top-down only in this slice; the frontal families and
  face_stop are reserved), the P16 finger columns at the measured opening inflated by e_lat (P1), the P2 three-way
  width, the depth rules touch_stop > height_stop with P3, P4, P5, P6, P7, P10, P12, P17 and the score;
- ContactClassifier (P8, P9), the clearance retreat (P10) and the width consistency at the close (G13);
- the V5 answer checks (spec §3.6 item 4), evaluated on the re-measured heights;
- O5 keep-out volumes;
- TransitPlanner: free-hand and held transit (M1, M2; [C-36] the held object's own pick-time cells excluded);
- AimSolver (M4, M16), LoweringPlan (M5-M8), SupportClassify (spec §4.9) and the destination sigma rule [C-7].

Conventions (§C1.3): world metres, z up. A candidate yaw is the FULL angle of the jaw R1's world-xy projection about
world +z, so the two 180 deg twins are distinct candidates (they differ in the side the wrist camera R0 looks to,
P17). A rim candidate (a seed in the outer half of its D11 circle on a thin wall) is (jaw yaw, rim angle): the rim
point centre + r (cos a, sin a) with a = yaw (finger 1 outside) or a = yaw + pi (finger 2 outside), so both finger
assignments at every rim point are offered; its key carries the rim angle (ContactPlan.key, candidate_key).
Top-down hand: R0 = the jaw turned +90 deg about world +z, and a world rotation by d about +z is a twist of about -d
about the approach R2 (contract sense).

Heights. Clearance checks (P3, P5, P6, P7, P11, P12, M1) use the part's measured sigma_h. The predicted contact
heights (the P8 bands and the P9 bound) use sigma_h + SIGMA_VC_M: the cameras see the visual surface while contacts
happen on the collision surface, and Appendix L 11 makes their largest measured difference a design-default height
error for contact predictions. The upper side of every P8 band is capped at the P3 close limit (top + pad bottom -
P3_MIN_OVERLAP_MM): a contact above it is blocked_above_part, because a close there no longer overlaps the part.

Own cells. The P16 columns see every surface under the fingers, the part's own cells included, except the measured
cross-section the pads close on (the part's cells within +-w/2 of the grip along the jaw): a protrusion of the part
under a finger is a landing. The P5 closing sweeps exclude the part's region (between a landing and the part, the
part's own cells are the edge the finger closes on, e.g. a curved wall), and P7 excludes it as the spec says.

Part band. The pads must overlap the part's band [part_low, top_z] (P3); part_low is the part's measured support
(the dominant plane around it), else the higher column landing: a PartRecord has no measured band-low, so P3 cannot
bind for an overhanging part (its band reads [support, top]). The palm clause of P6 also applies to touch_stop, at
the contact grip (the hand's lowest point on that descent; the retreat only matters for P3): a thin part far above
its support (an overhang seen from above cannot be told from a wall) is refused because the palm would meet it.

No state and no I/O. Every function returns the Threshold records it compared against (dict results carry them
under "thresholds"; jsonable_result converts a result for a trace event). The geometry module is imported lazily
(the V5 structure axis, the P-2 visibility rule's template radius) and can be injected.

Design review 1 (decision §2.4, THRESHOLDS_VERSION v6):
- P-A contact-verified admission (RC-1). D7b obstacle fills are unverified: a finger column lands on its verified
  maximum (measured cells, D7a plane fills, keep-outs), or on the part's support when it holds no verified cell
  (RC-1a); an unverified fill above the landing widens that column's expected band up to the P3 cap
  (``ExpectedContact.unverified_grip_z``). The closing sweeps skip D7b fills and the measured cells within |along
  jaw| <= w/2 + e_lat + cell/2 of the grip (RC-1b, "close_verified"). No sigma is counted twice at a D7b argmax
  (RC-1c), and the scene's ``part_fill_cells`` are excluded from P7 only (RC-1d). A candidate admitted only through
  unverified or close-verified cells is marginal (``ContactPlan.admission``), ranked after clean ones, and carries
  ``checks["unverified_cells"]`` (BR descends it with the P14 ladder). P7 stays a vision check.
- P-B: on a fixture-prior part a touch_stop contact below every band and above the P9 bound is expected with
  ``landing_measured`` (classify_contact).
- P-C: the jaw of a non-elongated part comes from its width harmonic only when the harmonic (contrast >= (A - 1) / (A
  + 1)) and the region PCA axis agree within YAW_GRID_RAD / 2; otherwise the yaw is unmeasured and plan_contact
  returns ``Infeasible("geometry_unmeasured", refine_axis=True)``. The plain grid stays for a fixture without a width
  axis and for a free part measured isotropic (flat width function AND isotropic region), in both cases only when
  the scene's image tensor coherence (when measured) is below V5_COHERENCE; never for an unmeasured free axis. A
  width function that does not cover the half-turn on the YAW_GRID_RAD grid gives no harmonic (width_harmonic), and
  a free part's yaw is then unmeasured (jaw_axis).
- P-D: G13+ ``misaligned`` in width_consistency (free parts).
- P-E: P11 capped at P11_CAP_M (``first_stage`` when the cap binds); a registered record's hover reference is its
  own top and region (inflated by P11_RECORD_SIGMA_K sigma_xy), always on the unraised field; the P-2 visibility
  rule (hover_for).
- P-F: a lower-bound part (``top_bound`` "lower") has its cells raised to ``top_upper_z`` in free_path, transit_path,
  P7, P12 and the hover's P12 (a neighbour's raise never enters the hover reference; the part's own top_upper_z
  does); contact planning on such a part asks to refine the height first.

Design review 2 (decision §2.4, THRESHOLDS_VERSION v7; no new §3 constant):
- P-H: the RC-7 width-harmonic jaw (P-C) applies only to a not_elongated record. A record whose axis the scene refused
  as AXIS_INCONSISTENT (D10-OE "no_edge_agreement", S-12 "views_disagree") or whose region collapsed (S-13) returns
  ``Infeasible("geometry_unmeasured", refine_axis=True)`` (axis_refusal, jaw_axis) without consulting the harmonic.
  A D11 rim record keeps its rim candidates (they enumerate the circle and read no axis; the rim test precedes the
  refusal, as before review 2).
- P-I: a record with ``width_by_yaw == ()`` (K-5, a collapsed region delivers no width) returns
  ``Infeasible("geometry_unmeasured")`` at P2 (one "width:P2" rejection flagged ``width_unmeasured``, before any
  candidate; width_unmeasured(); refine_axis reports the axis state); evaluate_candidate refuses at P2 the same way.
  ``width_state`` tells "unmeasurable" (``()``, K-5) from "not_measured" (``None``, a record without a width
  function).
- Lead sign-offs of the review-1 deviations (binding): (1) the isotropy exception (a free part keeps the 15 deg grid
  only when the width harmonic is flat AND the region is isotropic, >= AXIS_MIN_REGION_CELLS cells of aspect <
  AXIS_MIN_ASPECT, AND the scene's image coherence, when measured, is below V5_COHERENCE; otherwise refine_axis; a
  free part without a width function is unmeasured too, never the plain grid); (2) hover_for's P12 margin uses the
  field's own cell sigma, not sigma_h; (3) first_stage also on a P-F lift (reason "lower_bound"), judged at the
  hover's final xy (after the P-2 visibility shift); (4) the two adapted tests.
- No P7 change: P7 stays a vision check with RC-1d's part_fill_cells as its only fill exclusion (decision 2: "No
  other BP change"). The review-2 build's planner-side band-height fill rule (BPR-2) was withdrawn by the review-2
  fixer: it re-admitted the fills S-6 deliberately keeps (a separate raised object at the part's height), with no
  reach or connectivity test, and a palm stop on such a surface classified "expected". The loop's P7 refusals from
  fills S-6 misses are BS's S-6 (reach / connectivity) defect.

Live-first blockers (BSBP fixer, grip-part lens BP-1..BP-3; additive):
- BP-1: ``pointing_frame`` (zero calls) tells whether a hover frames a grip part for pointing: the fine crop's
  bottom (predicted row + FRAMING_CROP_HALF_PX) or, coarse-first (sigma_uv > FRAMING_COARSE_SIGMA_PX), row + sigma_uv
  lies FT_PROBE_RADIUS_PX above the fingertip band (P-2's margin kept). ``hover_for(..., pointing=...)`` applies it:
  a -R0 shift of at most M2_SEGMENT_M, then a rise in cell steps of at most P11_CAP_M (``framing`` in the result);
  ``pointing=None`` keeps P-2 only (no change for surveys).
- BP-2: a P3 (column), P5, P6 (landing) or P7 refusal lists its blocking cells (``values["blocking"]``: ij, xyz, z,
  sigma, source, unconfirmed from the scene's ``hf.unconfirmed``); ``refine_cells`` on the rejection and the
  Infeasible when a rule failed only on unconfirmed cells (``needs_cell_refinement``, ``blocking_cells`` = BR-5's
  survey targets); every Infeasible carries ``inputs`` (a snapshot with the HeightField deep-copied) to save.
- BP-3: no snapping: the grip point is the record's own point (the scene's S-2 top-band lift is the only snap).

Deviations from the decisions' wording still open for the lead's sign-off:
2. P-C on fixtures (BPR-3): the agreement rule (harmonic vs region PCA) and the coherence gate also apply to
   fixture-prior parts, which then refine the axis instead of falling back to the grid (decision 1: "the plain 15 deg
   grid is not used for FREE grip parts"). Counterfactual on the G1' records: a fixture grid fallback plans 1 of the
   fixture part's 5 refine_axis records (A_ep5_o0), outside the truth envelope (yaw error 15.1 deg); no in-envelope
   plan is lost.
3. P-C PCA definedness (BPR-4): the region axis also needs a cell aspect >= AXIS_MIN_ASPECT; decision 1 asks only >= 4
   cells. Review 2 supersedes BS's S-4' amendment it mirrored; it now follows from the ratified isotropy definition
   (a region of aspect < AXIS_MIN_ASPECT is isotropic: it has no axis to agree with). No RS-2 refusal came from the
   ratio alone.
4. P-F raise footprint: a lower-bound part's cells are its region inflated by O5_SIGMA_K sigma_xy (interpretation
   5); with BPR-1 that inflation reaches the hover only through P12.
(The review-2 build's deviations 5, BPR-2 band-height P7 fills, and 6, the P-H refusal before the D11 rim, were
withdrawn by the review-2 fixer; see above.)

Gap round (judge decision §2.4, planner-7; lead ratification L1, L8, L13-L15, L17):
- IF-HS (``_height_stop``): a height_stop refused at P7 at g* is re-run once at g* + d, d = the P7 deficit (ONLINE),
  every height_stop check at the raised stop; admitted only when P3 also holds with the part's top taken
  HS_P3_GUARD_SIGMA_K sigma_h lower. A plan whose P7 passes at g* is unchanged (no raise, no new key).
- IF-PV (``plan_contact``): only when the plan (with HS) is Infeasible and some P7 refusal lists D7b blockers, one
  more search reads every D7b fill in P7 at its parent + PV_PARENT_SIGMA_K sigma_parent (the stored fill holds the
  parent + D7B_FILL_SIGMA_K sigma_parent); such a plan is marginal and carries ``checks["p7_reading"] =
  "parent_sigma"``, and every re-evaluation of it (``revalidate``; ``evaluate_candidate(..., p7_reading=...)`` for
  PC-1) reads P7 the same way. P11, P12, the columns and the sweeps keep the full fills.
- IF-R3' (``aim_target``): a D7b fill occupies the destination only when its parent + PV_PARENT_SIGMA_K sigma_parent
  stands above s_dest + band (verified cells unchanged); with any occupied cell the target is the free cell nearest
  the region centre whose clearance meets M16's need, else the free cell of largest clearance (ties to the nearest
  centre); ``fits`` false never refuses: it is reported with the flag ``tight_fit``.
- SAL-R2 C-7' (``destination_precondition``): ok on the vertical cue (step >= 2 max(M8_FLOOR_M, M8_SIGMA_K
  sigma_top)) or the lateral cue (the region's hull inscribed radius around its centroid >= M4_FLOOR_M + HF_CELL_M
  and M8_SIGMA_K sigma_top <= M8_FLOOR_M); the result carries ``cue``.
- CV-CI1 / CI2 / V5U: ``same_candidate`` (CANDIDATE_YAW_TOL_RAD) in ``_exclusion``; ``revalidate`` (the executing
  candidate only); ``check_point_any_jaw`` (V5U_CONE_LINES).
- F6-B (``_columns``): a column whose verified landing comes only from D7a plane fills (no measured cell or keep-out
  within the P8 band of the landing in that column) is unverified for the admission label: the plan is marginal
  (P14 ladder, GC-1 active); the landing height is unchanged; ``checks["plane_landing_columns"]`` counts them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field as dataclass_field, replace as dataclass_replace
import math

import numpy as np

from . import agent_speedup_v2_robot as robot_facts
from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (AXIS_AGREE_MAX_RAD, AXIS_INCONSISTENT, HANDOVER_PREFIX, HF_CELL_M, HF_SOURCES,
                                     RIM_WALL_RADIUS_FRACTION, HandPose,
                                     HeightField, HeldFrame, KeepOut, PartRecord, RobotModel, Term, YAW_GRID_RAD,
                                     hf_code, jsonable, poly_contains, poly_inflate, rect_polygon, rotate_xy, th_max,
                                     th_value, thresholds_dicts)

PLANNER_VERSION = "agent-speedup-v2-planner-7"
FAMILIES = ("top_down", "frontal", "frontal_pitched_45")      # slice: top_down only (frontal reserved, §C0.3)
SLICE_FAMILIES = ("top_down",)
DEPTH_RULES = ("touch_stop", "face_stop", "height_stop", "palm_stop")
SLICE_DEPTH_RULES = ("touch_stop", "height_stop")                # face_stop (frontal) and palm_stop [C-47] reserved
PURPOSES = ("turn", "carry", "hold")
FIXTURE_PRIORS = ("fixed", "hinged", "sliding")                  # "free" and "unknown" count as free [C-22], [C-47]
WIDTH_CLASSES = ("feasible", "marginal")
EXPECTED_KINDS = ("fingertip_landing", "none_before_target", "palm_top", "face")
CONTACT_CLASSES = ("expected", "blocked_above_part", "no_contact_where_expected", "no_contact_within_bound",
                   "stopped_early")                              # stopped_early: additive, a non-contact stop short
                                                                  # of the plan's end (report the primitive's reason)
CONTACT_EVENTS = ("contact_stall", "contact_touch", "stall", "touch", "contact", "gap_rise", "gap_jump")
SUPPORT_CLASSES = ("on_surface", "perched", "blocked_above", "missed_surface", "no_support_within_bound",
                   "ambiguous")
SUPPORT_EVENTS = ("contact_stall", "contact_touch", "stall", "touch", "contact", "gap_rise")
POINT_CHECK_KINDS = ("grasp_point", "surface_point", "elongated")
INFEASIBLE_REASONS = ("part_too_wide", "column_blocked", "footprint_blocked", "turned_hand_blocked",
                      "no_pad_overlap", "palm_clearance", "yaw_budget", "frontal_needs_r5b", "geometry_unmeasured",
                      "no_feasible_candidate")
KEEPOUT_REQUEST_KINDS = ("clear_of", "articulated", "hold_true")
AXIS_SOURCES = ("d10", "rim", "width_harmonic", "isotropic", "grid", "unmeasured")   # P-C, jaw_axis
ADMISSIONS = ("clean", "marginal")             # P-A label: marginal = admitted only through unverified cells
FIXTURE_LANDING_EVENTS = ("contact_stall", "contact_touch", "stall", "touch")       # P-B: a fingertip stop
_OBSTACLE_CODE = hf_code("obstacle")           # D7b fills: unverified for contacts (RC-1)

# P1 lateral error budget, P2 width
P1_ALIGN_K = 2.0
P1_PAD_MARGIN_MM = 1.0
SIGMA_ALIGN_PRIOR_M = 0.003
P2_SIGMA_K = 2.0
P2_INFEASIBLE_MARGIN_MM = 2.0
# P3-P7 depth rules and clearances
P3_MIN_OVERLAP_MM = 8.0
P4_MIN_KNOWN_FRACTION = 0.7
P5_SIGMA_K = 3.0
P6_TARGET_FLOOR_MM = 2.0
P6_TARGET_SIGMA_K = 1.5
P6_PALM_FLOOR_MM = 5.0
P6_PALM_SIGMA_K = 2.0
P6_LANDING_FLOOR_MM = 3.0
P6_LANDING_SIGMA_K = 3.0
P7_FLOOR_MM = 5.0
P7_SIGMA_K = 3.0
# P8-P12 contact classification, bound, retreat, hover, turned hand
P8_FLOOR_M = 0.005
P8_SIGMA_K = 2.0
P9_FLOOR_M = 0.03
P9_SIGMA_K = 3.0
P10_MAX_RETREAT_MM = 8.0
P11_FLOOR_M = 0.05
P11_SIGMA_K = 3.0
P11_OFFSET_M = 0.02
P11_CAP_M = 0.10
P11_SEED_SIGMA_K = 3.0
P11_RECORD_SIGMA_K = 1.0
VIS_PROBE_LEG_M = 0.02
# P-C jaw axis of a non-elongated part (the scene's D10 numbers restated)
AXIS_MIN_ASPECT = 1.5
AXIS_MIN_REGION_CELLS = 4
P12_FLOOR_M = 0.01
P12_SIGMA_K = 3.0
P12_SWEEP_STEP_RAD = math.radians(5.0)
# P17 and the score (§4.6, DD: they only order feasible candidates)
P17_BONUS = 0.3
SCORE_TOUCH_BONUS = 0.5
SCORE_TURN_PENALTY_PER_QUARTER = 0.2
SCORE_MARGINAL_PENALTY = 1.0
# G13 width consistency
G13_FLOOR_MM = 3.0
G13_SIGMA_K = 3.0
# M motion and placement
M1_FLOOR_M = 0.02
M1_SIGMA_K = 3.0
M2_SEGMENT_M = 0.075
M4_FLOOR_M = 0.005
M4_FRACTION = 0.5
M5_FLOOR_M = 0.005
M5_FRACTION = 0.5
M6_ABOVE_M = 0.03
M6_SIGMA_K = 3.0
M7_FLOOR_M = 0.01
M7_TOL_K = 2.0
M8_FLOOR_M = 0.01
M8_SIGMA_K = 3.0
M16_FLOOR_M = 0.005
# O5 keep-outs
O5_FLOOR_M = 0.02
O5_SIGMA_K = 3.0
O5_TOP_FLOOR_M = 0.02
# V5 answer checks
V5_GRASP_SIGMA_K = 3.0
V5_AXIS_AGREE = 0.8
V5_COHERENCE = 0.5
V5_RMS = 2.0
V5_PLANE_FLOOR_M = 0.003
V5_PLANE_SIGMA_K = 3.0
# [C-7] destination sigma rule
DEST_SEPARATION_FRACTION = 0.5
DEST_SIGMA_K = 3.0
# other
SIGMA_VC_M = 0.004
RIM_SEED_RADIUS_FRACTION = 0.5
# RIM_WALL_RADIUS_FRACTION (the rim rule's thin wall) lives in the kernel (gap round: the scene's annulus model reads it)
CIRCLE_POLYGON_SIDES = 16
TRACE_MAX_REJECTIONS = 12
# live-first blockers (grip-part lens BP-1, BP-2)
FRAMING_CROP_HALF_PX = 64
FRAMING_COARSE_SIGMA_PX = FRAMING_CROP_HALF_PX / math.sqrt(2.0 * math.log(2.0))
BLOCKING_CELLS_MAX = 16
# --- grasp-refusal decision J (THRESHOLDS v8), remedy 1 RC-1e: a measured closing-sweep
# cell whose sigma exceeds the HeightField resolution (the RC-9a D7 parent cap) does not localize a surface in its
# cell: the sweep treats it as unverified (close-verified, marginal); P7 / P11 / P12 keep it with its sigma
P5_SIGMA_RESOLUTION_M = HF_CELL_M
# judge J, GC-1 close gate: the tolerance of a measured touch position (the guard's limit = expected face + this)
GC_CONTACT_TOL_MM = 1.0
# gap round (judge decision §2.4 / §6, THRESHOLDS v8): IF-HS P3 guard, IF-PV / IF-R3' fill reading at parent + k sigma
HS_P3_GUARD_SIGMA_K = 2.0
PV_PARENT_SIGMA_K = 1.0
# the scene's D7b fill rule (fill = parent z + 3 sigma_parent, stored with sigma_parent), restated: the planner may not
# import the scene; a test asserts the equality
D7B_FILL_SIGMA_K = 3.0
P7_READINGS = ("parent_sigma",)        # IF-PV: the only non-default P7 reading (None = the stored fills)
# the order of each depth rule's checks: a failed rule's progress = checks passed / checks (structural, no threshold)
TOUCH_CHECK_ORDER = ("P4", "P3", "P5", "P6_palm", "P7")
HEIGHT_CHECK_ORDER = ("P6_palm", "P4", "P6_landing", "P5", "P3", "P7")

LABELS = {
    "HS_P3_GUARD_SIGMA_K": ("DD", "IF-HS (gap round): a raised height stop g* + d is admitted only when P3 holds with "
                                  "the part's top 2 sigma_h lower (d <= 8 mm + offset - 2 sigma_h); 97 HS admissions "
                                  "scored against truth (inferred_fills, judge re-tally): k=0 -> 5, k=1 -> 2, k=2 -> 0 "
                                  "truth P3 failures (stored 80 -> 59 new admissions, p3t 5 -> 0)"),
    "PV_PARENT_SIGMA_K": ("DD", "IF-PV / IF-R3' (gap round): a D7b fill is read at its parent + 1 sigma_parent (the P7 "
                                "fallback reading and the M16 occupancy); 9,126 inferred P7 blockers with truth: k=1 "
                                "hides 0 of the 740 truly-above-limit cells, k=0 hides 54 (inferred_fills)"),
    "D7B_FILL_SIGMA_K": ("DD", "spec §4.5 D7b: a fill = parent z + 3 sigma_parent (parent sigma capped, RC-9a), stored "
                               "with that sigma; restates the scene's D7_OBSTACLE_SIGMA_K (the planner may not import "
                               "the scene); a test asserts the equality"),
    "V5U_CONE_LINES":("DD", "V5-U: 1 + 2 ceil(AXIS_AGREE_MAX_RAD / YAW_GRID_RAD) = 3 adjacent jaw lines (the best "
                             "line +- one YAW_GRID_RAD step), derived from the planner's own grid and P-C axis "
                             "agreement, not fitted; recorded jaw-less checks (gap round candidate_v5, pixel truth): "
                             "35 of 35 truth-on points clear >= 3 adjacent lines, 10 of 11 truth-off <= 2"),
    "CANDIDATE_YAW_TOL_RAD": ("DD", "CI-1 candidate identity: half the planner's own YAW_GRID_RAD candidate spacing "
                                    "(the P-C AXIS_AGREE_MAX_RAD derivation); every candidate set of one plan_contact "
                                    "call is >= YAW_GRID_RAD apart in yaw or pi apart; recorded grid drift after "
                                    "Orient / Align 0.01-0.15 deg (eval3post t2 / t8)"),
    "P1_ALIGN_K": ("DD", "spec §5.2 P1: e_lat = 2 sigma_align + sigma_w + 1.0 mm (the 2 of 2 sigma_align)"),
    "GC_CONTACT_TOL_MM": ("DD", "grasp-refusal decision GC-1 gate: tolerance of a measured touch position along the "
                                "jaw; FACT judge runs: the guard's touch span equals the measured pad separation "
                                "within 0.05 mm on 10 live closes; remedy 2: >= 12x the 0.08 mm max |touch - truth "
                                "contact| of 30 rigid touches"),
    "P1_PAD_MARGIN_MM": ("ROBOT", "spec §5.2 P1: the 1.0 mm term of e_lat, the R9 pad separation offset (finger gap - "
                                  "1.0 mm)"),
    "SIGMA_ALIGN_PRIOR_M": ("DD", "P1 prior Align residual before Align has run: v1 Align TOLERANCE_MM = 3.0 "
                                  "(wrist_servo.py:498), restated in metres; a test asserts the equality"),
    "P2_SIGMA_K": ("DD", "spec §5.2 P2: feasible w + 2 e_lat <= sep; infeasible w - 2 sigma_w >= sep - 2 mm (the 2)"),
    "P2_INFEASIBLE_MARGIN_MM": ("DD", "spec §5.2 P2: the 2 mm of 'infeasible if w - 2 sigma_w >= sep - 2 mm'"),
    "P3_MIN_OVERLAP_MM": ("ROBOT", "spec §5.2 P3: pad overlap >= 8 mm, half the 16 mm pad length (R4)"),
    "P4_MIN_KNOWN_FRACTION": ("DD", "spec §5.2 P4: each finger column has >= 70 % known cells (measured or D7 "
                                    "filled)"),
    "P5_SIGMA_K": ("DD", "spec §5.2 P5: no height above landing + 3 sigma_h between a landing and the part edge"),
    "P6_TARGET_FLOOR_MM": ("DD", "spec §5.2 P6: g* = top - 11.6 mm - max(2 mm, 1.5 sigma_h) (the 2 mm floor)"),
    "P6_TARGET_SIGMA_K": ("DD", "spec §5.2 P6: the 1.5 of max(2 mm, 1.5 sigma_h)"),
    "P6_PALM_FLOOR_MM": ("DD", "spec §5.2 P6: palm margin m_palm = max(5 mm, 2 sigma_h) on free parts (5 mm floor)"),
    "P6_PALM_SIGMA_K": ("DD", "spec §5.2 P6: the 2 of m_palm = max(5 mm, 2 sigma_h)"),
    "P6_LANDING_FLOOR_MM": ("DD", "spec §5.2 P6: column landings <= g* - 4.4 mm - max(3 mm, 3 sigma_h) (3 mm floor)"),
    "P6_LANDING_SIGMA_K": ("DD", "spec §5.2 P6: the 3 of max(3 mm, 3 sigma_h)"),
    "P7_FLOOR_MM": ("DD", "spec §5.2 P7: footprint max <= grip z + p_min - max(5 mm, 3 sigma_h) (5 mm floor)"),
    "P7_SIGMA_K": ("DD", "spec §5.2 P7: the 3 of max(5 mm, 3 sigma_h)"),
    "P8_FLOOR_M": ("DD", "spec §5.2 P8: contact band expected +- max(5 mm, 2 sigma_h) (5 mm floor)"),
    "P8_SIGMA_K": ("DD", "spec §5.2 P8: the 2 of max(5 mm, 2 sigma_h)"),
    "P9_FLOOR_M": ("DD", "spec §5.2 P9: approach bound expected contact - max(3 cm, 3 sigma_h); replaces v1's "
                         "0.300 m"),
    "P9_SIGMA_K": ("DD", "spec §5.2 P9: the 3 of max(3 cm, 3 sigma_h)"),
    "P10_MAX_RETREAT_MM": ("ROBOT", "spec §5.2 P10: the clearance retreat centring the pad band is clipped to 0-8 mm "
                                    "(half the 16 mm pad, R4)"),
    "P11_FLOOR_M": ("DD", "spec §5.2 P11: hover fingertips >= max(5 cm, 3 sigma_h + 2 cm) above the part top; v1 "
                          "hovers of 5-10 cm worked (spec v1 §17)"),
    "P11_SIGMA_K": ("DD", "spec §5.2 P11: the 3 of 3 sigma_h + 2 cm"),
    "P11_OFFSET_M": ("DD", "spec §5.2 P11: the 2 cm of 3 sigma_h + 2 cm"),
    "P11_CAP_M": ("DD", "design review 1 decision §2.4 P-E / §3 (remedy 1 P-1): P11 clearance = max(P11_FLOOR_M, "
                        "min(3 sigma_h + 2 cm, 10 cm)); v1 hovers of 5-10 cm worked (spec v1 §17), judge: band "
                        "support at the chain hovers 3/3/7 -> 6/27/16; a binding cap flags the hover first_stage"),
    "P11_SEED_SIGMA_K": ("DD", "hover_for: the seed uncertainty region is 3 sigma_xy around the seed point (the same "
                               "3 sigma as P11's height term)"),
    "P11_RECORD_SIGMA_K": ("RULING", "design review 1 decision §2.4 P-E and §5.1 item 3 (BP T1): a registered "
                                     "record's hover reference is its own top and its region inflated by sigma_xy "
                                     "(1 sigma_xy); an HF maximum beyond that no longer raises the hover"),
    "VIS_PROBE_LEG_M": ("DD", "design review 1 decision §2.4 P-E (remedy 1 P-2 visibility rule): the probe leg D2 "
                              "of kappa D2 + FT_PROBE_RADIUS_PX; restates spec §5.2 D2 (the primitives' D2_PROBE_M, "
                              "the planner does not import the primitives); a test asserts the equality"),
    "AXIS_MIN_ASPECT": ("DD", "design review 1 decision §2.4 P-C / §3: the width-harmonic contrast bound (A - 1) / "
                              "(A + 1) = 0.2; also the region-cell aspect below which a region is isotropic and "
                              "defines no PCA axis: the lead's ratified sign-off of BP deviation 1 (review 2 task: "
                              "isotropic = >= 4 cells and aspect < 1.5); A = the D10 aspect 1.5 (the scene's "
                              "D10_MIN_ASPECT, retired as a D10 decider by review 2 S-11 and kept logged); a test "
                              "asserts the equality while the scene keeps it"),
    "AXIS_MIN_REGION_CELLS": ("DD", "design review 1 decision §2.4 P-C / §3 D10_MIN_REGION_CELLS: a region PCA axis "
                                    "needs >= 4 HeightField cells, and the ratified isotropy grid an isotropic "
                                    "region of >= 4 cells (lead sign-off of BP deviation 1); restated from the "
                                    "scene (retired as a D10 decider by review 2 S-11, kept logged); a test asserts "
                                    "the equality while the scene keeps it"),
    "P12_FLOOR_M": ("DD", "spec §5.2 P12: the turned hand clears the height field by max(1 cm, 3 sigma_h)"),
    "P12_SIGMA_K": ("DD", "spec §5.2 P12: the 3 of max(1 cm, 3 sigma_h)"),
    "P12_SWEEP_STEP_RAD": ("DD", "P12 swept volume sampled every 5 deg of turn: a 104 mm hand corner (R7) moves "
                                 "9 mm between samples and the missed sliver is below 0.1 mm, under one 5 mm cell"),
    "P17_BONUS": ("DD", "spec §4.6 score: + 0.3 when P17 holds (the held object stays in the wrist view)"),
    "SCORE_TOUCH_BONUS": ("DD", "spec §4.6 score: + 0.5 if touch-terminated"),
    "SCORE_TURN_PENALTY_PER_QUARTER": ("DD", "spec §4.6 score: - 0.2 per 90 deg of turn"),
    "SCORE_MARGINAL_PENALTY": ("DD", "spec §4.6 score: - 1 if marginal"),
    "G13_FLOOR_MM": ("DD", "spec §5.2 G13: gap - 1.0 mm within w +- max(3 sigma_w, 3 mm) (3 mm floor)"),
    "G13_SIGMA_K": ("DD", "spec §5.2 G13: the 3 of max(3 sigma_w, 3 mm)"),
    "M1_FLOOR_M": ("DD", "spec §5.2 M1: lowest held point >= swept max height + max(2 cm, 3 sigma_h) (2 cm floor)"),
    "M1_SIGMA_K": ("DD", "spec §5.2 M1: the 3 of max(2 cm, 3 sigma_h)"),
    "M2_SEGMENT_M": ("DD", "spec §5.2 M2: held segments <= 7.5 cm, from the +-8 cm planning grid (contact_skills.py:"
                           "945)"),
    "M4_FLOOR_M": ("DD", "spec §5.2 M4: aim tolerance max(5 mm, 0.5 m_free) (5 mm floor)"),
    "M4_FRACTION": ("DD", "spec §5.2 M4: the 0.5 of 0.5 m_free"),
    "M5_FLOOR_M": ("DD", "spec §5.2 M5: lowering drift abort at max(5 mm, 0.5 m_free) (5 mm floor)"),
    "M5_FRACTION": ("DD", "spec §5.2 M5: the 0.5 of 0.5 m_free"),
    "M6_ABOVE_M": ("DD", "spec §5.2 M6: fast-to-slow switch 3 cm + 3 sigma_h above the expected support"),
    "M6_SIGMA_K": ("DD", "spec §5.2 M6: the 3 of 3 cm + 3 sigma_h"),
    "M7_FLOOR_M": ("DD", "spec §5.2 M7: lowering bound S_dest + b_low - max(1 cm, 2 tol_h) (1 cm floor)"),
    "M7_TOL_K": ("DD", "spec §5.2 M7: the 2 of 2 tol_h"),
    "M8_FLOOR_M": ("DD", "spec §5.2 M8: tol_h = max(1 cm, 3 sqrt(sigma_dest^2 + sigma_b^2 + sigma_touch^2))"),
    "M8_SIGMA_K": ("DD", "spec §5.2 M8: the 3 of tol_h; also the occupancy band of the destination (M16)"),
    "M16_FLOOR_M": ("DD", "spec §5.2 M16: free disk radius r_obj + max(5 mm, sigma_offset) (5 mm floor)"),
    "O5_FLOOR_M": ("DD", "spec §5.2 O5: keep-out region inflated by max(2 cm, 3 sigma_loc) [C-23] (2 cm floor)"),
    "O5_SIGMA_K": ("DD", "spec §5.2 O5: the 3 of 3 sigma_loc and of 3 sigma_h; also (decision §2.4 P-F, BP "
                         "interpretation 5, PENDING the lead's ruling) the inflation of a lower-bound part's raised "
                         "cells by 3 sigma_xy, as an O5 keep-out (_lower_bound_cells; used in P7, P12, M1 and the "
                         "hover's P12, never in the hover reference)"),
    "O5_TOP_FLOOR_M": ("DD", "spec §5.2 O5: vertically from the support to top + max(2 cm, 3 sigma_h)"),
    "V5_GRASP_SIGMA_K": ("DD", "spec §3.6 item 4: the pointed point stands >= 3 sigma_h above both finger-landing "
                               "columns (P16)"),
    "V5_AXIS_AGREE": ("DD", "v1 BAR_CHECK_MIN_AGREE 0.8 (wrist_servo.py:78), restated; a test asserts the equality"),
    "V5_COHERENCE": ("DD", "v1 BAR_CHECK_MIN_COHERENCE 0.5 (wrist_servo.py:79), restated; a test asserts the "
                           "equality; also the P-C isotropy gate (jaw_axis, BP deviation 1 as ratified by the lead "
                           "for review 2, decision 1 §2.4 P-C with §2.2 S-10 / R11): without a width axis the grid "
                           "needs the scene's logged image structure-tensor coherence below it (= the scene's "
                           "D10_MIN_COHERENCE, a diagnosis since review 2 S-11)"),
    "V5_RMS": ("DD", "v1 BAR_CHECK_MIN_RMS 2.0 grey/px (wrist_servo.py:80), restated; a test asserts the equality"),
    "V5_PLANE_FLOOR_M": ("DD", "spec §3.6 item 4 surface check: planar within D8 = max(3 mm, 3 sigma); restates the "
                               "scene's D8_FLOOR_M (the planner may not import the scene); a test asserts the "
                               "equality"),
    "V5_PLANE_SIGMA_K": ("DD", "the 3 of D8 max(3 mm, 3 sigma), restating the scene's D8_SIGMA_K with its sigma "
                               "combination hypot(sigma_point, sigma_cell); a test asserts the equality"),
    "DEST_SEPARATION_FRACTION": ("DD", "[C-7]: 3 sigma(top) <= 0.5 (top - support), else M8's tol_h can swallow the "
                                       "destination step"),
    "DEST_SIGMA_K": ("DD", "[C-7]: the 3 of 3 sigma(top)"),
    "SIGMA_VC_M": ("DD", "Appendix L 11: the max |visual - collision| surface offset over the reachable surveyed "
                         "parts (+4 mm, -2 mm, +3.5 mm, BH self-check W3), added to the height sigma of contact "
                         "predictions (P8 bands, P9 bound); re-derive at V5"),
    "RIM_SEED_RADIUS_FRACTION": ("DD", "a seed at >= 0.5 r from its D11 circle centre grasps the circle's wall (rim "
                                       "candidates around the circle); a seed nearer the centre grasps the part at "
                                       "the centre (a ridge over a disk)"),
    "CIRCLE_POLYGON_SIDES": ("DD", "a circle is replaced by its circumscribed 16-gon (radial excess 2 %, never inside "
                                   "the circle) for keep-outs and held footprints"),
    "TRACE_MAX_REJECTIONS": ("DD", "§C7 trace: the contact_plan event carries up to 12 rejections"),
    "FRAMING_CROP_HALF_PX": ("DD", "grip-part lens BP-1: half the V6 fine crop side (the client's FINE_CROP_PX = 128 "
                                   "px), restated (the planner may not import the client); a test asserts the "
                                   "equality. A fine-pointing hover keeps the crop's bottom (predicted row + 64 px) "
                                   "FT_PROBE_RADIUS_PX above the fingertip band (live crop bottoms 2-24 rows into the "
                                   "band; the ORACLE framed hovers 49 rows above it)"),
    "FRAMING_COARSE_SIGMA_PX": ("DD", "grip-part lens BP-1: the coarse-first threshold 64 / sqrt(2 ln 2) px (~54.4 "
                                      "px; rulings item 9, the relation layer's COARSE_FIRST_SIGMA_PX), restated; "
                                      "above it the framing rule is predicted row + sigma_uv <= band top - "
                                      "FT_PROBE_RADIUS_PX; a test asserts the equality"),
    "BLOCKING_CELLS_MAX": ("DD", "grip-part lens BP-2: a P3 / P5 / P6 / P7 refusal lists its blocking cells highest "
                                 "first, at most 16 in the trace (the count and the all-unconfirmed test cover all of "
                                 "them)"),
}


# ------------------------------------------------------------------------------------------------ small helpers
def _finite(value, name):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def _vec(value, size, name):
    out = np.array(value, dtype=float).reshape(-1)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return out


def _wrap(angle):
    """Angle in (-pi, pi]."""
    a = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def _same_angle(a, b):
    return abs(_wrap(float(a) - float(b))) < 1e-6


# CI-1 (candidate identity): two candidate yaws (or rim angles) name the same candidate when they lie within half the
# planner's own yaw grid. Every candidate set of one plan_contact call is >= YAW_GRID_RAD apart in yaw (grid / rim)
# or pi apart (axis twins, rim sides), so the bound never merges two candidates of one call; a grid anchored at a
# drifted jaw (jaw_yaws, an Orient / Align residual of 0.01-0.2 deg) maps back onto the same candidate.
CANDIDATE_YAW_TOL_RAD = YAW_GRID_RAD / 2.0
# V5-U (the grasp-point check without a candidate jaw): the point must clear both finger columns along a run of
# adjacent jaw lines at least this long: the best line and ceil(AXIS_AGREE_MAX_RAD / YAW_GRID_RAD) grid neighbours on
# each side (the jaw the re-measured record will plan lies within the P-C axis agreement of the line that clears)
V5U_CONE_LINES = 1 + 2 * math.ceil(AXIS_AGREE_MAX_RAD / YAW_GRID_RAD - 1e-9)


def same_candidate_angle(a, b) -> bool:
    """CI-1: |a - b| (wrapped) <= CANDIDATE_YAW_TOL_RAD."""
    return abs(_wrap(float(a) - float(b))) <= CANDIDATE_YAW_TOL_RAD + 1e-12


def same_candidate(a, b) -> bool:
    """CI-1: two candidate keys (ContactPlan.candidate_key / key, or exclusion items) name the same candidate: same
    family, yaw within CANDIDATE_YAW_TOL_RAD, and the same rim side (rim angle within the tolerance, or both None)."""
    a, b = tuple(a), tuple(b)
    if len(a) < 2 or len(b) < 2 or a[0] != b[0] or not same_candidate_angle(a[1], b[1]):
        return False
    ra = a[3] if len(a) >= 4 else None
    rb = b[3] if len(b) >= 4 else None
    if ra is None or rb is None:
        return ra is None and rb is None
    return same_candidate_angle(ra, rb)


def _unit_xy(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def _up_xy(angle):
    """World-xy R0 of a top-down hand whose jaw has the full angle ``angle`` (R0 = jaw turned +90 deg)."""
    return np.array([-math.sin(angle), math.cos(angle)])


def _jaw_angle(R):
    matrix = np.asarray(R, dtype=float)
    x, y = float(matrix[0, 1]), float(matrix[1, 1])
    if math.hypot(x, y) < 1e-6:
        raise ValueError("the jaw has no horizontal component")
    return math.atan2(y, x)


def _sigma_h(part):
    return 0.0 if part.sigma_h_m is None else float(part.sigma_h_m)


def _is_fixture(part):
    return part.move_prior in FIXTURE_PRIORS


def _sigma_or(value, default):
    return default if value is None or not math.isfinite(float(value)) else float(value)


def _circle_polygon(centre_xy, radius_m):
    """Circumscribed CIRCLE_POLYGON_SIDES-gon of the circle (CCW)."""
    centre = _vec(centre_xy, 2, "centre_xy")
    reach = float(radius_m) / math.cos(math.pi / CIRCLE_POLYGON_SIDES)
    angles = (np.arange(CIRCLE_POLYGON_SIDES) + 0.5) * (2.0 * math.pi / CIRCLE_POLYGON_SIDES)
    return centre + reach * np.column_stack([np.cos(angles), np.sin(angles)])


def _polygon_centroid(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return pts.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return pts.mean(axis=0)
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * area)


def _region_cells(part, hf):
    if part.region_cells is not None and len(part.region_cells):
        return np.asarray(part.region_cells, dtype=int).reshape(-1, 2)
    if part.region_xy is not None and len(part.region_xy) >= 3:
        return hf.cells_in_polygon(part.region_xy)
    return np.zeros((0, 2), dtype=int)


def _centre_xy(part, hf=None):
    """Footprint centre of a part: its region cells (with a HeightField), else its region polygon, else its point."""
    if hf is not None and part.region_cells is not None and len(part.region_cells):
        cells = np.asarray(part.region_cells, dtype=int).reshape(-1, 2)
        return hf.origin_xy + cells[:, ::-1].mean(axis=0) * hf.cell_m
    if part.region_xy is not None and len(part.region_xy) >= 3:
        return _polygon_centroid(part.region_xy)
    if part.xyz is not None:
        return np.asarray(part.xyz, dtype=float)[:2].copy()
    return None


def horizontal_sigma_m(part) -> float:
    """[C-23] the part's horizontal position sigma: sqrt(max eigenvalue of cov[:2, :2]); 0 without a covariance."""
    if part.cov is None:
        return 0.0
    cov = np.asarray(part.cov, dtype=float)[:2, :2]
    return float(math.sqrt(max(float(np.max(np.linalg.eigvalsh((cov + cov.T) / 2.0))), 0.0)))


def _winner(*terms):
    """(value, label, text) of the largest term (first wins a tie), as th_max picks it."""
    best = max(terms, key=lambda t: t.value)
    return best.value, best.label, best.text


def _unique_cells(chunks):
    chunks = [np.asarray(c, dtype=int).reshape(-1, 2) for c in chunks if len(c)]
    if not chunks:
        return np.zeros((0, 2), dtype=int)
    return np.unique(np.vstack(chunks), axis=0)


def _cell_centres(hf, cells):
    return hf.origin_xy + np.asarray(cells, dtype=float).reshape(-1, 2)[:, ::-1] * hf.cell_m


def _geometry():
    """The geometry module, imported lazily (the one import of it in this module)."""
    from . import agent_speedup_v2_geometry as geometry
    return geometry


def _axis_angle(a, b) -> float:
    """Angle between two world-xy axes mod pi, in [0, pi/2]."""
    a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
    norms = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if norms < 1e-12:
        return math.pi / 2.0
    return math.acos(min(1.0, abs(float(a @ b)) / norms))


def _lower_bound_cells(part, hf) -> np.ndarray:
    """P-F: the cells of a lower-bound part (``top_bound`` "lower"): its region polygon inflated by O5_SIGMA_K
    sigma_xy (its horizontal uncertainty, as an O5 keep-out; at least half a cell), its region cells, and around its
    point when it has no region (rulings item 8, the BS RS-6 emulation)."""
    reach = max(O5_SIGMA_K * horizontal_sigma_m(part), hf.cell_m / 2.0)
    chunks = []
    if part.region_xy is not None and len(part.region_xy) >= 3:
        chunks.append(hf.cells_in_polygon(poly_inflate(part.region_xy, reach)))
    if part.region_cells is not None and len(part.region_cells):
        chunks.append(np.asarray(part.region_cells, dtype=int).reshape(-1, 2))
    if not chunks and part.xyz is not None:
        chunks.append(hf.cells_in_polygon(rect_polygon(np.asarray(part.xyz, dtype=float)[:2], (1.0, 0.0), reach,
                                                       reach)))
    return _unique_cells(chunks)


def _lower_bound_parts(parts, *, skip=()) -> list:
    """The records of ``parts`` whose top is only a lower bound (rulings R2), minus the pids in ``skip``."""
    out, seen = [], set(p for p in skip if p is not None)
    for part in parts or ():
        if part is None or part.pid in seen or not part.top_is_lower_bound() or part.top_upper_z is None:
            continue
        seen.add(part.pid)
        out.append(part)
    return out


def jsonable_result(value):
    """A planner result as plain JSON for a trace event: Threshold records become their dicts, records with
    ``to_dict`` their dicts, numpy becomes Python (kernel jsonable)."""
    if isinstance(value, Threshold):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): jsonable_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_result(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and not isinstance(value, np.ndarray):
        return jsonable(to_dict())
    return jsonable(value)


class _Field:
    """HeightField queries with keep-out volumes as obstacles (their cells raised to z_high) and excluded cells.
    D7b obstacle fills are known but unverified for contacts (RC-1); lower-bound parts can be raised to their
    top_upper_z (P-F, ``raising``)."""

    def __init__(self, hf: HeightField, keepouts=(), *, exclude_pids=(), exclude_cells=None):
        self.hf = hf
        shape = hf.z.shape
        self.measured = hf.measured
        self.unverified = hf.known & (hf.source == _OBSTACLE_CODE)
        self.raised_pids = []
        self.raised_owner = {}                    # (iy, ix) -> pid of the lower-bound part whose raise tops the cell
        self.skip = np.zeros(shape, dtype=bool)
        if exclude_cells is not None and len(exclude_cells):
            cells = np.asarray(exclude_cells, dtype=int).reshape(-1, 2)
            inside = ((cells[:, 0] >= 0) & (cells[:, 0] < shape[0]) & (cells[:, 1] >= 0)
                      & (cells[:, 1] < shape[1]))
            cells = cells[inside]
            self.skip[cells[:, 0], cells[:, 1]] = True
        self.overlay = np.full(shape, -np.inf)
        self.keepout_pids = []
        excluded = set(p for p in exclude_pids if p is not None)
        for keepout in keepouts or ():
            if keepout.pid in excluded:
                continue
            cells = hf.cells_in_polygon(keepout.polygon_xy)
            if len(cells):
                iy, ix = cells[:, 0], cells[:, 1]
                self.overlay[iy, ix] = np.maximum(self.overlay[iy, ix], float(keepout.z_high))
                self.keepout_pids.append(keepout.pid)

    def excluding(self, cells) -> "_Field":
        """The same field (keep-outs shared) with ``cells`` also excluded."""
        out = object.__new__(_Field)
        out.__dict__.update(self.__dict__)
        out.skip = self.skip.copy()
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        if len(cells):
            shape = self.skip.shape
            inside = ((cells[:, 0] >= 0) & (cells[:, 0] < shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < shape[1]))
            cells = cells[inside]
            out.skip[cells[:, 0], cells[:, 1]] = True
        return out

    def reading_fills_at_parent(self, k) -> "_Field":
        """IF-PV: the same field (keep-outs, skips and raises shared) with every D7b fill read at its parent + k
        sigma_parent: the stored fill is parent + D7B_FILL_SIGMA_K sigma_parent with sigma_parent stored, so the read
        is z - (D7B_FILL_SIGMA_K - k) sigma. Measured cells, D7a fills and overlays are unchanged."""
        out = object.__new__(_Field)
        out.__dict__.update(self.__dict__)
        hf = copy.copy(self.hf)
        z = np.array(self.hf.z, dtype=float, copy=True)
        cells = self.unverified
        z[cells] = z[cells] - (D7B_FILL_SIGMA_K - float(k)) * np.nan_to_num(self.hf.sigma[cells], nan=0.0)
        hf.z = z
        out.hf = hf
        return out

    def raising(self, parts) -> "_Field":
        """P-F: the same field with every lower-bound part of ``parts`` raised to its top_upper_z over its cells
        (``_lower_bound_cells``), as an overlay (a keep-out-like obstacle, sigma 0); ``self`` when there is none."""
        lows = _lower_bound_parts(parts)
        if not lows:
            return self
        out = object.__new__(_Field)
        out.__dict__.update(self.__dict__)
        out.overlay = self.overlay.copy()
        out.raised_pids = list(self.raised_pids)
        out.raised_owner = dict(self.raised_owner)
        for part in lows:
            cells = _lower_bound_cells(part, self.hf)
            if len(cells):
                iy, ix = cells[:, 0], cells[:, 1]
                top = float(part.top_upper_z)
                for a, b in zip(iy.tolist(), ix.tolist()):
                    if not out.overlay[a, b] > top:
                        out.raised_owner[(a, b)] = part.pid
                out.overlay[iy, ix] = np.maximum(out.overlay[iy, ix], top)
                out.raised_pids.append(part.pid)
        return out

    def stats(self, cells, *, extra_skip=None) -> dict:
        """z_max (known cells and keep-outs), sigma there (0 on a keep-out and, RC-1c, on a D7b fill: its stored
        height already holds D7_OBSTACLE_SIGMA_K sigma_parent), known / measured fractions, z_min and the cell count,
        over ``cells`` minus the excluded ones (and minus ``extra_skip``, one bool per cell). RC-1a adds the verified
        maximum (measured cells, D7a plane fills, keep-outs: "z_max_verified", "sigma_verified", "ij_verified") and the
        unverified D7b fills ("z_max_unverified", "unverified_count", "unverified_z" = their heights, highest first);
        "z_max" keeps its meaning (every known cell), so P11, P12 and hover_for stay conservative. F6-B (gap round) adds
        "z_max_firm": the maximum over the measured cells and the overlays only (no D7a / D7b fill)."""
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        if len(cells):
            keep = ~self.skip[cells[:, 0], cells[:, 1]]
            if extra_skip is not None:
                keep &= ~np.asarray(extra_skip, dtype=bool)
            cells = cells[keep]
        empty = {"z_max": None, "sigma": None, "known_fraction": 0.0, "measured_fraction": 0.0, "z_min": None,
                 "count": 0, "keepout": False, "ij": None, "z_max_verified": None, "sigma_verified": None,
                 "ij_verified": None, "z_max_unverified": None, "unverified_count": 0, "unverified_z": np.zeros(0),
                 "raised_pid": None, "z_max_firm": None}
        if not len(cells):
            return empty
        iy, ix = cells[:, 0], cells[:, 1]
        known = self.hf.known[iy, ix]
        unverified = self.unverified[iy, ix]
        over = self.overlay[iy, ix]
        has_over = np.isfinite(over)
        z = np.where(known, self.hf.z[iy, ix], -np.inf)
        effective = np.maximum(z, over)
        known_any = known | has_over
        out = dict(empty, count=int(len(cells)), known_fraction=float(known_any.mean()),
                   measured_fraction=float(self.measured[iy, ix].mean()))
        if not known_any.any():
            return out
        k = int(np.argmax(np.where(known_any, effective, -np.inf)))
        from_overlay = bool(has_over[k] and over[k] >= z[k])
        sigma = 0.0 if (from_overlay or unverified[k]) else _sigma_or(self.hf.sigma[iy[k], ix[k]], 0.0)
        out.update(z_max=float(effective[k]), sigma=sigma, z_min=float(np.min(effective[known_any])),
                   keepout=from_overlay, ij=(int(iy[k]), int(ix[k])),
                   raised_pid=self.raised_owner.get((int(iy[k]), int(ix[k]))) if from_overlay else None)
        solid = known & ~unverified
        verified = solid | has_over
        if verified.any():
            z_solid = np.where(solid, z, -np.inf)
            z_verified = np.maximum(z_solid, over)
            kv = int(np.argmax(np.where(verified, z_verified, -np.inf)))
            over_v = bool(has_over[kv] and over[kv] >= z_solid[kv])
            out.update(z_max_verified=float(z_verified[kv]),
                       sigma_verified=0.0 if over_v else _sigma_or(self.hf.sigma[iy[kv], ix[kv]], 0.0),
                       ij_verified=(int(iy[kv]), int(ix[kv])))
            firm = (known & self.measured[iy, ix]) | has_over
            if firm.any():
                z_firm = np.maximum(np.where(known & self.measured[iy, ix], z, -np.inf), over)
                out["z_max_firm"] = float(np.max(np.where(firm, z_firm, -np.inf)))
        if unverified.any():
            heights = np.sort(z[unverified])[::-1]
            out.update(z_max_unverified=float(heights[0]), unverified_count=int(unverified.sum()),
                       unverified_z=heights)
        return out


# ------------------------------------------------------------------------------------------------ records
@dataclass(frozen=True, eq=False)
class PlannerInputs:
    """Measured quantities the GraspPlanner needs besides the part and the HeightField."""
    robot: RobotModel
    g_open_mm: float                 # G1; P16 columns use finger_q_mm = g_open_mm / 2 [C-50]
    sigma_align_m: float             # P1: measured Align residual std; prior SIGMA_ALIGN_PRIOR_M
    yaw_budget_left: dict            # {+1: rad, -1: rad} keyed by contract sense about R2 (R12, RobotModel)
    pose: HandPose
    mount: dict
    sigma_loc_px: float | None

    def __post_init__(self):
        if _finite(self.g_open_mm, "g_open_mm") <= 0.0:
            raise ValueError("g_open_mm must be > 0")
        if _finite(self.sigma_align_m, "sigma_align_m") < 0.0:
            raise ValueError("sigma_align_m must be >= 0")
        if set(self.yaw_budget_left) != {1, -1}:
            raise ValueError("yaw_budget_left must be keyed by the contract senses +1 and -1")
        for sense in (1, -1):
            _finite(self.yaw_budget_left[sense], "yaw_budget_left")
        if not isinstance(self.pose, HandPose):
            raise ValueError("pose must be a HandPose")

    @property
    def finger_q_mm(self) -> float:
        return float(self.g_open_mm) / 2.0

    @classmethod
    def build(cls, robot: RobotModel, state8, *, g_open_mm: float, start_quaternion, sigma_align_m=None,
              mount=None, sigma_loc_px=None) -> "PlannerInputs":
        """Inputs at the current state8: the R12 budget left in both senses from the episode's start orientation;
        sigma_align None = the prior SIGMA_ALIGN_PRIOR_M (before Align has run)."""
        pose = robot.pose(state8)
        budget = {sense: robot.yaw_budget_left(start_quaternion, pose.quaternion_xyzw, sense) for sense in (1, -1)}
        return cls(robot, float(g_open_mm), SIGMA_ALIGN_PRIOR_M if sigma_align_m is None else float(sigma_align_m),
                   budget, pose, dict(mount or {}), sigma_loc_px)


@dataclass(frozen=True, eq=False)
class FingerColumn:
    finger: int                      # 1 (on +jaw) or 2
    polygon_xy: np.ndarray           # (4,2) world xy, inflated by e_lat
    landing_z: float | None          # RC-1a: the highest VERIFIED surface (measured cells, D7a plane fills, keep-outs),
                                     # else the part's support when the column holds only D7b fills
    landing_sigma: float | None
    known_fraction: float            # P4: known = measured OR D7 filled (fills still count, unchanged)
    straddles_edge: bool             # the column holds surfaces farther apart than the P8 band (the higher lands)
    unverified_z: float | None = None        # additive (RC-1a): the highest D7b fill in the column
    unverified_count: int | None = None      # additive: the column's D7b fills
    landing_source: str | None = None        # additive: "verified" | "support" (None: no landing)
    unverified_heights: np.ndarray | None = None   # additive: the D7b fill heights, highest first (not serialized)
    landing_plane_only: bool = False         # additive (F6-B): the verified landing rests on D7a plane fills only (no
                                             # measured cell or keep-out within the P8 band of it): unverified

    def unverified_above(self, z) -> int:
        """The column's D7b fills above ``z`` (the cells only a contact can verify)."""
        if self.unverified_heights is None or not len(self.unverified_heights):
            return 0
        return int(np.count_nonzero(np.asarray(self.unverified_heights, dtype=float) > float(z) + 1e-9))

    def to_dict(self) -> dict:
        return jsonable({"finger": self.finger, "polygon_xy": self.polygon_xy, "landing_z": self.landing_z,
                         "landing_sigma": self.landing_sigma, "known_fraction": self.known_fraction,
                         "straddles_edge": self.straddles_edge, "unverified_z": self.unverified_z,
                         "unverified_count": self.unverified_count, "landing_source": self.landing_source,
                         "landing_plane_only": self.landing_plane_only})


@dataclass(frozen=True)
class ExpectedContact:
    kind: str                        # EXPECTED_KINDS
    grip_z: float                    # grip z at which the contact is expected
    band_m: float                    # P8 half band
    column: int | None
    close_limit_grip_z: float | None = None    # additive: the band's upper side is capped here (P3 close limit)
    unverified_grip_z: float | None = None     # additive (RC-1a): grip z of a touch on the column's highest D7b fill

    @property
    def upper_z(self) -> float:
        """The upper edge of the expected band: max(grip_z, unverified_grip_z) + band_m (RC-1a: a touch on a real
        fill below the cap is expected and P10 is recomputed from it), capped at the P3 close limit."""
        high = self.grip_z + self.band_m
        if self.unverified_grip_z is not None:
            high = max(high, float(self.unverified_grip_z) + self.band_m)
        return high if self.close_limit_grip_z is None else min(high, float(self.close_limit_grip_z))

    def to_dict(self) -> dict:
        return jsonable({"kind": self.kind, "grip_z": self.grip_z, "band_m": self.band_m, "column": self.column,
                         "close_limit_grip_z": self.close_limit_grip_z, "unverified_grip_z": self.unverified_grip_z,
                         "upper_z": self.upper_z})


@dataclass(frozen=True, eq=False)
class ContactPlan:
    pid: str
    family: str
    depth_rule: str
    purpose: str                     # "turn" | "carry" | "hold"
    jaw_world: tuple
    approach_world: tuple
    quaternion_xyzw: tuple
    yaw_turn_rad: float              # signed rotation about the CURRENT R2 the Orient must realize (contract sense)
    grip_point_xyz: tuple            # turn on a D11 part: circle centre projected on the axis; rim: the rim point;
                                     # otherwise the pointed (fine-seed) point
    hover_grip_z: float              # P11, raised until the hand swept by the Orient clears (P12 margin)
    target_grip_z: float | None      # height_stop g*
    bound_grip_z: float              # P9: lowest expected-contact grip z - max(P9_FLOOR_M, P9_SIGMA_K*sigma_c)
    expected: tuple                  # ExpectedContact, highest first
    clearance_retreat_m: float       # P10 at the expected landing (recomputed at contact)
    columns: tuple                   # (FingerColumn, FingerColumn)
    width: dict                      # {"w_mm","sigma_w_mm","sep_mm","e_lat_mm","class"}
    checks: dict                     # {"P3_overlap_mm","P5_sweep_ok","P6_palm_margin_mm","P7_clearance_mm",
                                     #  "P12_clearance_m","P17"} + review 1: "unverified_cells" (column fills above
                                     #  the landing / limit + close_verified sweep cells), "column_unverified",
                                     #  "close_verified", "admission", "axis_source", "part_fill_cells",
                                     #  "lower_bound_parts" + decision J: "part_ring_cells", "gate" (GC-1)
    align_mode: str                  # "across_bar" if part.is_elongated() else "both"
    score: float
    thresholds: tuple
    rim_angle_rad: float | None = None   # additive: a rim candidate's rim-point angle (yaw or yaw + pi)
    move_prior: str | None = None        # additive: the part's move prior (P-B fixture landing, P-D G13+ free parts)
    admission: str = "clean"             # additive (P-A label): "marginal" = admitted only through unverified or
                                         # close-verified cells; BR descends with the P14 ladder

    @property
    def yaw_rad(self) -> float:
        """The candidate yaw: the full world angle of the planned jaw (the key `exclude` takes)."""
        return math.atan2(float(self.jaw_world[1]), float(self.jaw_world[0]))

    @property
    def marginal(self) -> bool:
        """P2 marginal width, or (P-A) admitted only through unverified / close-verified cells: ranked after clean
        candidates."""
        return self.width.get("class") == "marginal" or self.admission == "marginal"

    @property
    def unverified_cells(self) -> int:
        return int(self.checks.get("unverified_cells") or 0)

    def key(self) -> tuple:
        """(family, yaw, depth_rule) (+ rim angle for a rim candidate): excludes this rule of this candidate in
        plan_contact(..., exclude=...)."""
        if self.rim_angle_rad is None:
            return (self.family, self.yaw_rad, self.depth_rule)
        return (self.family, self.yaw_rad, self.depth_rule, self.rim_angle_rad)

    def candidate_key(self) -> tuple:
        """Excludes every rule of this candidate: (family, yaw), or (family, yaw, None, rim angle) for a rim."""
        if self.rim_angle_rad is None:
            return (self.family, self.yaw_rad)
        return (self.family, self.yaw_rad, None, self.rim_angle_rad)

    def to_dict(self) -> dict:
        return jsonable({
            "pid": self.pid, "family": self.family, "depth_rule": self.depth_rule, "purpose": self.purpose,
            "yaw_deg": math.degrees(self.yaw_rad),
            "rim_angle_deg": None if self.rim_angle_rad is None else math.degrees(self.rim_angle_rad),
            "jaw_world": self.jaw_world, "approach_world": self.approach_world,
            "quaternion_xyzw": self.quaternion_xyzw, "yaw_turn_rad": self.yaw_turn_rad,
            "grip_point_xyz": self.grip_point_xyz, "hover_grip_z": self.hover_grip_z,
            "target_grip_z": self.target_grip_z, "bound_grip_z": self.bound_grip_z,
            "expected": [e.to_dict() for e in self.expected], "clearance_retreat_m": self.clearance_retreat_m,
            "columns": [c.to_dict() for c in self.columns], "width": self.width, "checks": self.checks,
            "align_mode": self.align_mode, "score": self.score, "move_prior": self.move_prior,
            "admission": self.admission, "marginal": self.marginal,
            "thresholds": thresholds_dicts(self.thresholds)})


@dataclass(frozen=True)
class CandidateRejection:
    family: str
    yaw_rad: float
    rule: str                        # "<depth rule or stage>:<check id>", e.g. "height_stop:P6", "width:P2"
    reason: str                      # INFEASIBLE_REASONS
    values: dict                     # JSON-safe evidence (by_rule holds every depth rule's failure)
    thresholds: tuple = ()           # the Threshold records it compared against (additive to §C7)
    rim_angle_rad: float | None = None   # additive: a rim candidate's rim-point angle

    def to_dict(self) -> dict:
        return jsonable({"family": self.family, "yaw_deg": math.degrees(self.yaw_rad),
                         "rim_angle_deg": None if self.rim_angle_rad is None else math.degrees(self.rim_angle_rad),
                         "rule": self.rule, "reason": self.reason, "values": self.values,
                         "thresholds": thresholds_dicts(self.thresholds)})


@dataclass(frozen=True)
class Infeasible:
    pid: str
    reason: str                      # the most frequent rejection reason (INFEASIBLE_REASONS)
    rejections: tuple
    thresholds: tuple
    refine_axis: bool = False        # additive (P-C): the jaw yaw is unmeasured; BR refines the axis (BR-1)
    refine_cells: bool = False       # additive (BP-2): a candidate failed only on unconfirmed cells (BS-2); BR-5
    # BP-2: a snapshot of plan_contact's inputs (the HeightField deep-copied at the refusal; the immutable records,
    # keep-outs and PlannerInputs by reference) so the caller can save them on every refusal; not compared, not
    # serialised by to_dict
    inputs: dict | None = dataclass_field(default=None, compare=False, repr=False)

    @property
    def blocking_cells(self) -> list[dict]:
        """BP-2: the module function ``blocking_cells(self)`` (BR-5's re-measurement targets)."""
        return _blocking_targets(self)

    def to_dict(self) -> dict:
        return jsonable({"pid": self.pid, "reason": self.reason, "refine_axis": self.refine_axis,
                         "refine_cells": self.refine_cells, "blocking_cells": _blocking_targets(self),
                         "rejections": [r.to_dict() for r in self.rejections],
                         "thresholds": thresholds_dicts(self.thresholds)})


def needs_cell_refinement(result) -> bool:
    """BP-2: True when a candidate of ``result`` was refused at P3 / P5 / P6 / P7 only by cells the scene marks
    unconfirmed (``hf.unconfirmed``, BS-2): BR re-measures those cells (one ParallaxProbe with their pixels as survey
    targets, BR-5) and re-plans once."""
    if isinstance(result, Infeasible):
        return bool(result.refine_cells) or any(bool(r.values.get("refine_cells")) for r in result.rejections)
    return isinstance(result, CandidateRejection) and bool(result.values.get("refine_cells"))


def _blocking_targets(result) -> list[dict]:
    rejections = result.rejections if isinstance(result, Infeasible) else (
        (result,) if isinstance(result, CandidateRejection) else ())
    return _merge_cells([r.values.get("blocking_cells") or () for r in rejections if r.values.get("refine_cells")])


def blocking_cells(result) -> list[dict]:
    """BP-2: the unique cells to re-measure ({"ij", "xyz", "z", "sigma", "source", "unconfirmed"}), highest first:
    the unconfirmed blocking cells of the rules that failed only on unconfirmed cells, over the rejections that ask
    ``refine_cells`` (each rejection also lists them as ``values["blocking_cells"]``; the Infeasible as its
    ``blocking_cells`` property): BR-5's survey targets."""
    return _blocking_targets(result)


def needs_height_refinement(result) -> bool:
    """Appendix L 3: True when a candidate was refused only because of the height sigma: the P6 palm clause (a free
    part's height_stop is feasible only while sigma_h <= 5.54 mm; ``sigma_h_limit_m`` gives the exact limit) or a
    P6 / P7 margin that the floor-only margin (sigma -> 0) would pass, or (P-F) the part's top is only a lower bound.
    The caller must first refine the height (touch or parallax probe, a lower hover) and re-plan; it must not plan
    the grasp on the unrefined height."""
    rejections = result.rejections if isinstance(result, Infeasible) else (
        (result,) if isinstance(result, CandidateRejection) else ())
    return any(bool(r.values.get("refine_height")) for r in rejections)


def lower_bound_blockers(result) -> list[str]:
    """P-F: the pids of the lower-bound parts whose top_upper_z raise decided a P7 refusal of ``result`` (an
    Infeasible or a CandidateRejection); BR refines them first (survey_hover, rulings item 8) and re-plans."""
    rejections = result.rejections if isinstance(result, Infeasible) else (
        (result,) if isinstance(result, CandidateRejection) else ())
    return sorted({pid for r in rejections for pid in (r.values.get("lower_bound_blockers") or ())})


def needs_axis_refinement(result) -> bool:
    """P-C: True when plan_contact refused because the jaw yaw of a non-elongated part is unmeasured (the width
    harmonic and the region axis disagree, or neither defines one): BR refines the axis (BR-1 second hover) and
    re-plans."""
    if isinstance(result, Infeasible):
        return bool(result.refine_axis) or any(bool(r.values.get("refine_axis")) for r in result.rejections)
    return isinstance(result, CandidateRejection) and bool(result.values.get("refine_axis"))


def width_unmeasured(result) -> bool:
    """P-I (design review 2 decision §2.4, K-5): True when the candidate or the whole plan was refused at P2 because the
    record carries no width function: ``width_by_yaw == ()`` (K-5, its region collapsed, S-13: BR re-surveys the
    part, BR-1 on region_collapsed, and re-plans) or None (never measured); the rejection's ``width_state`` tells
    them apart ("unmeasurable" / "not_measured"). Nothing is planned on an unmeasured width."""
    rejections = result.rejections if isinstance(result, Infeasible) else (
        (result,) if isinstance(result, CandidateRejection) else ())
    return any(bool(r.values.get("width_unmeasured")) for r in rejections)


def trace_payload(result) -> dict:
    """§C7 contact_plan trace: the chosen ContactPlan.to_dict() (or the Infeasible reason) and up to
    TRACE_MAX_REJECTIONS rejections {family, yaw_deg, rule, reason}."""
    if isinstance(result, ContactPlan):
        return {"plan": result.to_dict(), "infeasible": None, "rejections": []}
    rows = [dict({"family": r.family, "yaw_deg": math.degrees(r.yaw_rad), "rule": r.rule, "reason": r.reason},
                 **({} if r.rim_angle_rad is None else {"rim_angle_deg": math.degrees(r.rim_angle_rad)}))
            for r in result.rejections[:TRACE_MAX_REJECTIONS]]
    return jsonable({"plan": None, "infeasible": result.reason, "pid": result.pid, "rejections": rows,
                     "refine_height": needs_height_refinement(result),
                     "refine_axis": needs_axis_refinement(result),
                     "width_unmeasured": width_unmeasured(result),
                     "lower_bound_blockers": lower_bound_blockers(result),
                     "refine_cells": needs_cell_refinement(result), "blocking_cells": blocking_cells(result)})


# ------------------------------------------------------------------------------------------------ candidates
def families_for(part: PartRecord, inputs: PlannerInputs) -> tuple[str, ...]:
    """Slice: top_down only; the frontal families need a measured mount side and are reserved (§C0.3)."""
    return SLICE_FAMILIES


def _rim(part):
    """(centre (2,), radius) when the part's seed sits on the wall of its D11 circle, else None."""
    if part.circle is None or part.xyz is None or part.is_elongated():
        return None
    centre, radius = np.asarray(part.circle[0], dtype=float), float(part.circle[1])
    if not radius > 0.0:
        return None
    offset = np.asarray(part.xyz, dtype=float)[:2] - centre
    distance = float(np.linalg.norm(offset))
    if distance < RIM_SEED_RADIUS_FRACTION * radius:
        return None
    wall = part.width_at(math.atan2(offset[1], offset[0]))
    if wall is None or wall.value > RIM_WALL_RADIUS_FRACTION * radius:
        return None
    return centre, radius


def _grid_count():
    return max(1, int(round(2.0 * math.pi / YAW_GRID_RAD)))


def harmonic_contrast_min() -> float:
    """P-C / RC-7: the width-harmonic contrast bound (AXIS_MIN_ASPECT - 1) / (AXIS_MIN_ASPECT + 1) (0.2): D10's own
    aspect, no new number."""
    return (AXIS_MIN_ASPECT - 1.0) / (AXIS_MIN_ASPECT + 1.0)


def width_coverage(rows) -> bool:
    """BPR-5: True when the jaw yaws of ``rows`` (width_by_yaw) cover the half-turn evenly on the YAW_GRID_RAD grid:
    n = round(pi / YAW_GRID_RAD) yaws, distinct mod pi, consecutive ones (cyclically) one grid step apart. Only then
    is the discrete second harmonic below an unbiased contrast (a flat width sampled over part of the half-turn
    reads a large one: 1 row 2.0, 2 rows 1.93, 6 rows over 75 deg 1.29)."""
    n = int(round(math.pi / YAW_GRID_RAD))
    if not rows or len(rows) != n:
        return False
    psi = np.array([float(p) for p, _ in rows])
    if not np.isfinite(psi).all():
        return False
    phase = np.sort(np.mod(psi, math.pi))
    gaps = np.diff(np.append(phase, phase[0] + math.pi))
    return bool(np.all(np.abs(gaps - math.pi / n) <= 1e-6))


def width_harmonic(part: PartRecord):
    """RC-7: (long axis (2,) unit world xy, contrast c2 / c0) of the second harmonic of the measured width_by_yaw (the
    widest jaw direction is the long axis); None without a usable width function: no rows, a non-finite or
    non-positive width, or rows that do not cover the half-turn evenly on the yaw grid (width_coverage, BPR-5)."""
    rows = part.width_by_yaw
    if not rows or not width_coverage(rows):
        return None
    psi = np.array([float(p) for p, _ in rows])
    w = np.array([float(m.value) for _, m in rows])
    if not (np.isfinite(w).all() and np.isfinite(psi).all()) or not float(w.mean()) > 0.0:
        return None
    c, s = float(np.sum(w * np.cos(2.0 * psi))), float(np.sum(w * np.sin(2.0 * psi)))
    contrast = 2.0 * math.hypot(c, s) / len(w) / float(w.mean())
    phi = math.atan2(s, c) / 2.0
    return np.array([math.cos(phi), math.sin(phi)]), float(contrast)


def _second_moment(xy):
    """(major axis (2,), sqrt(lambda_major / lambda_minor)) of a point set's second moment."""
    centred = np.asarray(xy, dtype=float) - np.mean(xy, axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred)
    return vectors[:, 1], math.sqrt(max(float(values[1]), 0.0) / max(float(values[0]), 1e-12))


def region_axis(part: PartRecord, hf: HeightField | None = None) -> dict:
    """P-C: the second-moment (PCA) axis of the part's region cells ({"axis", "ratio", "cells"}), as the scene's S-4'
    computes it: the region cells' centres (``hf``), else the cells of a 5 mm grid inside the region polygon; the axis
    needs >= AXIS_MIN_REGION_CELLS cells ("axis" None below)."""
    if hf is not None and part.region_cells is not None and len(part.region_cells):
        xy = _cell_centres(hf, part.region_cells)
    elif part.region_xy is not None and len(part.region_xy) >= 3:
        poly = np.asarray(part.region_xy, dtype=float)
        centre = poly.mean(axis=0)
        grid = HeightField.empty(centre, half_extent_m=float(np.max(np.linalg.norm(poly - centre, axis=1))) + HF_CELL_M)
        xy = _cell_centres(grid, grid.cells_in_polygon(poly))
    else:
        xy = np.zeros((0, 2))
    if len(xy) < AXIS_MIN_REGION_CELLS:
        return {"axis": None, "ratio": None, "cells": int(len(xy))}
    axis, ratio = _second_moment(xy)
    return {"axis": axis, "ratio": ratio, "cells": int(len(xy))}


def axis_refusal(part: PartRecord) -> str | None:
    """P-H (design review 2 decision §2.4): the reason the scene refused this record's axis as inconsistent, else None.
    A refusal is an ``axis_check`` sourced AXIS_INCONSISTENT (D10-OE "no_edge_agreement", S-12 "views_disagree"; its
    ``reason``, else the source) or a region the scene flagged ``region_collapsed`` (S-13 refuses its axis: the region
    is not the part, so neither its PCA axis nor its widths measure the part's shape). An elongated record (an accepted
    axis, or S-12's remembered "kept_earlier" axis) is never refused here; "not_elongated" is not a refusal (RC-7)."""
    if part.is_elongated():
        return None
    check = part.axis_check if isinstance(part.axis_check, Mapping) else {}
    if check.get("region_collapsed") is True:
        return "region_collapsed"
    if check.get("source") == AXIS_INCONSISTENT:
        reason = check.get("reason")
        return reason if isinstance(reason, str) and reason else AXIS_INCONSISTENT
    return None


def jaw_axis(part: PartRecord, hf: HeightField | None = None) -> dict:
    """P-C (decision §2.4, RC-7 under the agreement rule): how the jaw yaw of the part is measured. Review 2 (P-H):
    RC-7 applies only to a not_elongated record; a record whose axis the scene refused as AXIS_INCONSISTENT (including
    S-12 "views_disagree") or whose region collapsed (S-13) is "unmeasured" with ``refine_axis``, the width harmonic
    not consulted ("axis_refusal" = axis_refusal(part)): an elongated region whose evidence disagrees must not fall
    back on a harmonic (RC-7 was 10.4 deg wrong on such a region in review 1). The D11 rim test comes first (as
    before review 2): rim candidates enumerate the circle and read no axis, so the scene's axis refusal does not
    withdraw them (the order is fixed by a test). Sources
    (AXIS_SOURCES): "d10" (the scene's elongation), "rim" (a D11 rim seed), "width_harmonic" (a non-elongated part
    whose width harmonic, contrast >= harmonic_contrast_min(), agrees with its region PCA axis within
    AXIS_AGREE_MAX_RAD = YAW_GRID_RAD / 2, the axis defined by >= AXIS_MIN_REGION_CELLS cells of aspect >=
    AXIS_MIN_ASPECT: a region below that aspect is isotropic by the ratified definition, BP deviation 3), "grid" (a
    fixture without a width axis: the YAW_GRID_RAD grid as before), "isotropic" (a free part whose width function is
    flat AND whose region is isotropic: the axis is measured absent, the grid stays) and "unmeasured"
    (``refine_axis``: the harmonic disagrees with or lacks a region axis, or a free part's flat width function meets
    an elongated or too small region, or a free part's width function does not cover the half-turn on the yaw grid
    (width_coverage, BPR-5), or a free part has no width function at all (P-I refuses it at P2 first); the plain grid
    is not used for free grip parts). Without a width axis the grid also
    needs the image to agree that there is none: when the scene measured the structure-tensor coherence
    (``axis_check["coherence"]``, S-4' / R11) and it reaches V5_COHERENCE (the v1 bar check, = the scene's
    D10_MIN_COHERENCE, logged as a diagnosis since review 2 S-11), the image sees an orientation the region and widths
    missed (an under-grown region of a thin part), so the axis is "unmeasured" (BP deviation 1, ratified by the lead
    for review 2: isotropic = flat harmonic AND >= 4 cells of aspect < 1.5 AND coherence < 0.5 when measured).
    Fixtures (BP deviation 2, for the lead's sign-off): the agreement rule and the coherence gate run BEFORE the
    fixture grid fallback, so a fixture-prior part whose harmonic disagrees with its region axis, or whose image is
    oriented without a width axis, also refines the axis instead of taking the grid (the decision withholds the grid
    only from free grip parts; on the G1' records the fixture fallback would have planned 1 of 5 such records,
    outside the truth envelope at 15.1 deg yaw error). Returns {"source", "axis", "contrast", "region_axis",
    "region_ratio", "region_cells", "agree_rad", "image_coherence", "width_rows", "width_coverage", "axis_refusal",
    "refine_axis", "thresholds"}."""
    out = {"source": None, "axis": None, "contrast": None, "region_axis": None, "region_ratio": None,
           "region_cells": None, "agree_rad": None, "image_coherence": None, "width_rows": 0, "width_coverage": None,
           "axis_refusal": None, "refine_axis": False, "thresholds": []}
    if part.is_elongated():
        out.update(source="d10", axis=np.asarray(part.elongation[0], dtype=float))
        return out
    if _rim(part) is not None:
        out["source"] = "rim"
        return out
    refusal = axis_refusal(part)
    if refusal is not None:
        check = part.axis_check if isinstance(part.axis_check, Mapping) else {}
        agree = check.get("agree_rad")
        out.update(source="unmeasured", refine_axis=True, axis_refusal=refusal,
                   agree_rad=None if agree is None else float(agree))
        # a record of the scene's decision, not a planner comparison: the value is the agreement bound the scene
        # applied (D10-OE outline vs edges, S-12 between views); the planner compares nothing against it
        out["thresholds"] = [th_value("PH_scene_axis_refusal_carried", AXIS_AGREE_MAX_RAD, "DD",
                                      "P-H, carried from the scene (no planner comparison): the scene refused the "
                                      "axis (AXIS_INCONSISTENT at its AXIS_AGREE_MAX_RAD agreement bound, or an S-13 "
                                      "collapsed region); RC-7 is not consulted, the yaw is unmeasured (refine_axis, "
                                      "BR-1)", reason=refusal, scene_source=str(check.get("source")),
                                      scene_agree_rad="none" if agree is None else float(agree))]
        return out
    harmonic = width_harmonic(part)
    region = region_axis(part, hf)
    cmin = harmonic_contrast_min()
    rows = part.width_by_yaw or ()
    out.update(region_axis=region["axis"], region_ratio=region["ratio"], region_cells=region["cells"],
               contrast=None if harmonic is None else harmonic[1], width_rows=len(rows),
               width_coverage=width_coverage(rows) if rows else None)
    out["thresholds"] = [
        th_value("D10_harmonic_contrast", cmin, "DD", "P-C: width-harmonic contrast c2/c0 >= (AXIS_MIN_ASPECT - 1) / "
                 "(AXIS_MIN_ASPECT + 1)", measured=out["contrast"], aspect=AXIS_MIN_ASPECT),
        th_value("D10_harmonic_agree_rad", AXIS_AGREE_MAX_RAD, "DD", "P-C: the harmonic axis within YAW_GRID_RAD / 2 "
                 "of the region PCA axis (>= AXIS_MIN_REGION_CELLS cells of aspect >= AXIS_MIN_ASPECT)",
                 cells=region["cells"], ratio=region["ratio"])]
    defined = region["axis"] is not None and region["ratio"] >= AXIS_MIN_ASPECT - 1e-9
    check = part.axis_check if isinstance(part.axis_check, Mapping) else {}
    coherence = check.get("coherence")
    coherence = None if coherence is None or not math.isfinite(float(coherence)) else float(coherence)
    out["image_coherence"] = coherence
    oriented = coherence is not None and coherence >= V5_COHERENCE - 1e-12
    out["thresholds"].append(th_value("D10_isotropy_coherence", V5_COHERENCE, "DD", "P-C: without a width axis the "
                                      "grid needs the image structure-tensor coherence below V5_COHERENCE (the "
                                      "scene's D10_MIN_COHERENCE)", measured=coherence))
    if harmonic is not None and harmonic[1] >= cmin - 1e-12:
        agree = _axis_angle(harmonic[0], region["axis"]) if defined else None
        out["agree_rad"] = agree
        if agree is not None and agree <= AXIS_AGREE_MAX_RAD + 1e-9:
            out.update(source="width_harmonic", axis=harmonic[0])
        else:
            out.update(source="unmeasured", refine_axis=True)
    elif oriented:
        out.update(source="unmeasured", refine_axis=True)
    elif harmonic is None and not _is_fixture(part):
        # BPR-5: a free part's width function off the grid; or none at all (P-C: no plain grid for a free part)
        out.update(source="unmeasured", refine_axis=True)
    elif harmonic is None or _is_fixture(part):
        out["source"] = "grid"
    elif region["axis"] is not None and region["ratio"] < AXIS_MIN_ASPECT - 1e-9:
        out["source"] = "isotropic"
    else:
        out.update(source="unmeasured", refine_axis=True)
    return out


def jaw_yaws(part: PartRecord, family: str, inputs: PlannerInputs, *, hf: HeightField | None = None,
             axis: dict | None = None) -> list[float]:
    """Candidate yaws (full jaw angles), smaller turns first. A measured axis (D10 elongation or, P-C, an agreeing
    width harmonic): perpendicular to it, both 180 deg twins. A rim (seed on the wall of its D11 circle): every
    YAW_GRID_RAD around the circle; each jaw yaw gives two rim candidates (rim_candidates). A fixture without a width
    axis or an isotropic free part: the YAW_GRID_RAD grid anchored at the current jaw, within the R12 budget. An
    unmeasured axis (P-C ``refine_axis``): none. ``axis`` = a jaw_axis result (computed from ``hf`` when None)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    angle0 = _jaw_angle(inputs.pose.R)
    axis = jaw_axis(part, hf) if axis is None else axis
    if axis["source"] in ("d10", "width_harmonic"):
        direction = np.asarray(axis["axis"], dtype=float)
        perpendicular = math.atan2(direction[1], direction[0]) + math.pi / 2.0
        yaws = [_wrap(perpendicular), _wrap(perpendicular + math.pi)]
    elif axis["source"] == "rim":
        yaws = [_wrap(angle0 + k * YAW_GRID_RAD) for k in range(_grid_count())]
    elif axis["source"] == "unmeasured":
        yaws = []
    else:
        half = _grid_count() // 2
        yaws = []
        for k in range(-half + 1, half + 1):
            turn = -k * YAW_GRID_RAD                                  # about R2 for a top-down hand
            sense = 1 if turn > 0.0 else -1
            if turn == 0.0 or abs(turn) <= float(inputs.yaw_budget_left[sense]) + 1e-9:
                yaws.append(_wrap(angle0 + k * YAW_GRID_RAD))
    return sorted(yaws, key=lambda y: abs(_wrap(y - angle0)))


def rim_candidates(part: PartRecord, family: str, inputs: PlannerInputs, *, hf: HeightField | None = None,
                   axis: dict | None = None) -> list[tuple[float, float | None]]:
    """(jaw yaw, rim angle) pairs in jaw_yaws order. A rim part offers both finger assignments at every rim point:
    rim angle = yaw (finger 1 outside) and yaw + pi (finger 2 outside; the same rim point as the 180 deg twin with
    finger 1 outside, reached with a turn smaller by a half turn). Any other part: (yaw, None)."""
    yaws = jaw_yaws(part, family, inputs, hf=hf, axis=axis)
    if _rim(part) is None:
        return [(yaw, None) for yaw in yaws]
    return [(yaw, _wrap(yaw + side)) for yaw in yaws for side in (0.0, math.pi)]


def _grip_point(part, yaw, purpose, rim, rim_angle=None):
    point = np.asarray(part.xyz, dtype=float)
    top = float(part.top_z.value) if part.top_z is not None else float(point[2])
    if rim is not None:
        centre, radius = rim
        xy = centre + radius * _unit_xy(yaw if rim_angle is None else rim_angle)
        return np.array([xy[0], xy[1], top])
    if purpose == "turn" and part.circle is not None and part.is_elongated():
        axis = np.asarray(part.elongation[0], dtype=float)
        centre = np.asarray(part.circle[0], dtype=float)
        along = float((centre - point[:2]) @ axis)
        xy = point[:2] + along * axis
        return np.array([xy[0], xy[1], point[2]])
    return point.copy()


def _orientation(robot, yaw):
    jaw = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    approach = np.array([0.0, 0.0, -1.0])
    return jaw, approach, np.asarray(robot.quaternion_for(jaw, approach), dtype=float)


def _yaw_turn(inputs, quaternion):
    """(signed turn about the current R2, sense, budget left in that sense). A turn of exactly a half turn may go
    either way: the sense with the larger budget is taken."""
    pose = inputs.pose
    turn = float(inputs.robot.yaw_about_approach(pose.quaternion_xyzw, quaternion, pose.approach))
    if abs(turn) >= math.pi - 1e-6:
        sense = max((1, -1), key=lambda s: float(inputs.yaw_budget_left[s]))
        turn = sense * abs(turn)
    sense = 1 if turn > 0.0 else -1
    return turn, sense, float(inputs.yaw_budget_left[sense])


class _Context:
    """Per-part planning state shared by the candidates of one plan_contact call (pure; rebuilt per call)."""

    def __init__(self, part, hf, inputs, keepouts, purpose, next_held, *, part_fill_cells=None, parts=(),
                 part_ring_cells=None, p7_reading=None):
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {PURPOSES}")
        if p7_reading is not None and p7_reading not in P7_READINGS:
            raise ValueError(f"p7_reading must be None or one of {P7_READINGS}")
        self.p7_reading = p7_reading
        self.part, self.hf, self.inputs, self.purpose, self.next_held = part, hf, inputs, purpose, next_held
        self.robot = inputs.robot
        self.region_cells = _region_cells(part, hf)
        skip_pids = {part.pid, None if next_held is None else next_held.pid}
        # field: the part region excluded (P5 sweeps); field_all: nothing excluded (the P16 columns, which skip only
        # the candidate's cross-section, section_cells). P7 (field_p7) also skips the scene's part_fill_cells (RC-1d)
        # and P11 / P12 (field_p12) see the field; both raise every other lower-bound part to its top_upper_z (P-F)
        self.field = _Field(hf, keepouts, exclude_pids=skip_pids, exclude_cells=self.region_cells)
        self.field_all = _Field(hf, keepouts, exclude_pids=skip_pids)
        self.part_fill_cells = (np.zeros((0, 2), dtype=int) if part_fill_cells is None
                                else np.asarray(part_fill_cells, dtype=int).reshape(-1, 2))
        lows = _lower_bound_parts([next_held, *(parts or ())], skip=(part.pid,))
        self.sigma_h = _sigma_h(part)
        self.point = np.asarray(part.xyz, dtype=float)
        self.top = float(part.top_z.value) if part.top_z is not None else float(self.point[2])
        # P-J (grasp-refusal decision): the scene's S-17 own-band ring (measured top-band cells one cell outside the
        # region); the part's measured section is the region plus the ring. P7 is unchanged (P-K not adopted)
        self.ring_cells = (np.zeros((0, 2), dtype=int) if part_ring_cells is None
                           else np.asarray(part_ring_cells, dtype=int).reshape(-1, 2))
        self.own_cells = _unique_cells([self.region_cells, self.ring_cells])
        self.field_p7 = self.field.excluding(self.part_fill_cells).raising(lows)
        if p7_reading == "parent_sigma":        # IF-PV: P7 only reads the D7b fills at parent + PV_PARENT_SIGMA_K sigma
            self.field_p7 = self.field_p7.reading_fills_at_parent(PV_PARENT_SIGMA_K)
        self.field_p12 = self.field_all.raising(lows)
        self.support = None if part.support_z is None else float(part.support_z.value)
        self.support_sigma = None if part.support_z is None else float(part.support_z.sigma)
        self.axis = jaw_axis(part, hf)
        self.fixture = _is_fixture(part)
        self.rim = _rim(part)
        self.angle0 = _jaw_angle(inputs.pose.R)
        self.finger_q = inputs.finger_q_mm
        self.t_f = self.robot.t_f_m
        self.p_min = self.robot.p_min_m
        self.pad_top = -float(self.robot.pad_span_m[0])        # 11.6 mm: pad top above the grip, top-down
        self.pad_bottom = float(self.robot.pad_span_m[1])      # 4.4 mm: pad bottom below the grip, top-down
        self.sep_mm = float(self.robot.pad_separation_mm(inputs.g_open_mm))
        self.held_centre = None if next_held is None else _centre_xy(next_held, hf)
        self.rim_width = None
        if self.rim is not None:
            offset = self.point[:2] - self.rim[0]
            self.rim_width = part.width_at(math.atan2(offset[1], offset[0]))
        self.p4 = th_value("P4_known_fraction", P4_MIN_KNOWN_FRACTION, "DD",
                           "P4_MIN_KNOWN_FRACTION: known = measured OR D7 filled")


# ------------------------------------------------------------------------------------------------ BP-2 blocking cells
def _blocking(field, cells, limit, *, verified=False, extra_skip=None) -> dict:
    """BP-2: the cells of ``cells`` (minus the field's excluded cells and ``extra_skip``) whose effective height (a
    known cell; D7b fills dropped when ``verified``, as the P16 landing; the max with a keep-out or P-F raise
    overlay) exceeds ``limit``: {"count", "unconfirmed" (cells the scene's ``hf.unconfirmed`` marks, BS-2; an
    overlay never is), "all_unconfirmed", "limit_z", "cells" (the highest BLOCKING_CELLS_MAX: "ij", "xyz", "z",
    "sigma", "source" (the HF source, "keepout" or "raised:<pid>"), "unconfirmed")}."""
    hf = field.hf
    out = {"count": 0, "unconfirmed": 0, "all_unconfirmed": False, "inferred": 0, "all_unverified": False,
           "limit_z": float(limit), "cells": []}
    cells = np.asarray(cells, dtype=int).reshape(-1, 2)
    if len(cells):
        keep = ~field.skip[cells[:, 0], cells[:, 1]]
        if extra_skip is not None:
            keep &= ~np.asarray(extra_skip, dtype=bool)
        cells = cells[keep]
    if not len(cells):
        return out
    iy, ix = cells[:, 0], cells[:, 1]
    known = hf.known[iy, ix] & np.isfinite(hf.z[iy, ix])
    if verified:
        known &= ~field.unverified[iy, ix]
    z = np.where(known, hf.z[iy, ix], -np.inf)
    over = field.overlay[iy, ix]
    effective = np.maximum(z, over)
    block = effective > float(limit) + 1e-9
    if not block.any():
        return out
    from_overlay = np.isfinite(over) & (over >= z)
    marks = getattr(hf, "unconfirmed", None)
    unconfirmed = (np.zeros(len(cells), dtype=bool) if marks is None
                   else np.asarray(marks, dtype=bool)[iy, ix] & ~from_overlay)
    # grasp-refusal decision (diagnostics only; BP-2b not adopted): a D7b fill is inferred, not measured; the rows and
    # the summary say so, refine_cells stays BS-2's rule (all-unconfirmed cells only)
    inferred = (hf.source[iy, ix] == _OBSTACLE_CODE) & ~from_overlay
    rows = []
    for k in np.flatnonzero(block)[np.argsort(-effective[block], kind="stable")][:BLOCKING_CELLS_MAX]:
        a, b = int(iy[k]), int(ix[k])
        xy = hf.cell_xy((a, b))
        if from_overlay[k]:
            pid = field.raised_owner.get((a, b))
            source, sigma = ("keepout" if pid is None else f"raised:{pid}"), 0.0
        else:
            source, sigma = HF_SOURCES[int(hf.source[a, b])], _sigma_or(hf.sigma[a, b], 0.0)
        rows.append({"ij": [a, b], "xyz": [float(xy[0]), float(xy[1]), float(effective[k])],
                     "z": float(effective[k]), "sigma": float(sigma), "source": source,
                     "unconfirmed": bool(unconfirmed[k]), "inferred": bool(inferred[k])})
    count, n_unconfirmed = int(block.sum()), int((block & unconfirmed).sum())
    n_unverified = int((block & (unconfirmed | inferred)).sum())
    out.update(count=count, unconfirmed=n_unconfirmed, all_unconfirmed=bool(count and n_unconfirmed == count),
               inferred=int((block & inferred).sum()), all_unverified=bool(count and n_unverified == count),
               cells=rows)
    return out


def _merge_cells(lists) -> list[dict]:
    """The unique cells (by ij) of several blocking cell lists, highest first."""
    rows, seen = [], set()
    for cells in lists:
        for cell in cells:
            if tuple(cell["ij"]) not in seen:
                seen.add(tuple(cell["ij"]))
                rows.append(cell)
    return sorted(rows, key=lambda c: -float(c["z"]))


def _merge_blocking(parts) -> dict | None:
    """BP-2: several checks' blocking sets as one (the union of the listed cells, highest first; counts summed;
    all_unconfirmed only when every non-empty set is all unconfirmed); None when none blocks."""
    parts = [b for b in parts if b and b.get("count")]
    if not parts:
        return None
    rows = _merge_cells([b["cells"] for b in parts])
    return {"count": sum(int(b["count"]) for b in parts), "unconfirmed": sum(int(b["unconfirmed"]) for b in parts),
            "all_unconfirmed": all(bool(b["all_unconfirmed"]) for b in parts),
            "inferred": sum(int(b.get("inferred", 0)) for b in parts),
            "all_unverified": all(bool(b.get("all_unverified")) for b in parts), "cells": rows[:BLOCKING_CELLS_MAX]}


# ------------------------------------------------------------------------------------------------ P16 columns
def _columns(field, grip_xy, yaw, robot, *, inflate_m, support_z, finger_q_mm, sigma_floor, support_sigma=None):
    """P16 columns with the RC-1a landing: the verified maximum, else the support for a column holding only D7b
    fills (landing_source "support"); the fills are reported (unverified_z / unverified_count / unverified_heights).
    F6-B (gap round): a verified landing with no measured cell or keep-out within its P8 band in the column rests on
    D7a plane fills only (``landing_plane_only``): unverified for the admission label, the height unchanged."""
    polys = robot.finger_columns_xy(grip_xy, _unit_xy(yaw), finger_q_mm=finger_q_mm, inflate_m=inflate_m)
    out = []
    for finger, poly in zip((1, 2), polys):
        s = field.stats(field.hf.cells_in_polygon(poly))
        if s["z_max_verified"] is not None:
            landing, sigma, source = s["z_max_verified"], s["sigma_verified"], "verified"
        elif s["unverified_count"] and support_z is not None:
            landing, sigma, source = float(support_z), _sigma_or(support_sigma, 0.0), "support"
        else:
            landing, sigma, source = None, None, None
        straddles = plane_only = False
        if landing is not None:
            band = max(P8_FLOOR_M, P8_SIGMA_K * max(_sigma_or(sigma, 0.0), sigma_floor))
            straddles = s["z_min"] is not None and (landing - s["z_min"]) > band
            if source == "verified":
                firm = s["z_max_firm"]
                plane_only = firm is None or firm < landing - band - 1e-9
        out.append(FingerColumn(finger, np.asarray(poly, dtype=float), landing, sigma, s["known_fraction"],
                                bool(straddles), unverified_z=s["z_max_unverified"],
                                unverified_count=int(s["unverified_count"]), landing_source=source,
                                unverified_heights=np.asarray(s["unverified_z"], dtype=float),
                                landing_plane_only=bool(plane_only)))
    return tuple(out)


def finger_columns(grip_xy, jaw_yaw: float, hf: HeightField, robot: RobotModel, *, e_lat_m: float,
                   support_z: float | None, finger_q_mm: float, exclude_cells=None,
                   keepouts=()) -> tuple[FingerColumn, FingerColumn]:
    """P16 (+P4): each finger's lowest-band footprint at the measured opening (finger_q_mm = g_open / 2, [C-50])
    inflated by e_lat, from the hover down; its landing is the highest VERIFIED surface in it (measured cells, D7a
    plane fills and keep-out tops; RC-1a: D7b obstacle fills are unverified, reported as unverified_z, and a column
    holding only D7b fills lands on ``support_z``; ``exclude_cells`` skipped: the planner passes the measured
    cross-section the pads close on, i.e. the part's cells within +-w/2 of the grip along the jaw, so the part's
    protrusions under a finger still land it, section_cells); a column holding surfaces farther apart
    than the P8 band straddles an edge (the higher lands). ``support_z`` (the part's support, contract signature) does
    not clip the column: a landing below the support (beyond the support's edge) is a real landing, and what lies
    between a landing and the part is the closing sweep's business (P5). ``jaw_yaw`` is the jaw's world angle (finger
    1 on +jaw)."""
    field = _Field(hf, keepouts, exclude_cells=exclude_cells)
    return _columns(field, _vec(grip_xy, 2, "grip_xy"), float(jaw_yaw), robot, inflate_m=float(e_lat_m),
                    support_z=support_z, finger_q_mm=float(finger_q_mm), sigma_floor=0.0)


def section_cells(region_cells, hf: HeightField, grip_xy, jaw_yaw: float, half_width_m: float) -> np.ndarray:
    """The measured cross-section the pads close on: the part's region cells whose centre lies within +-w/2 of the
    grip along the jaw (BP-1). Only these are skipped by the P16 columns: any other cell of the part under a finger
    (a protrusion beyond the cross-section) is a landing."""
    cells = np.asarray(region_cells, dtype=int).reshape(-1, 2)
    if not len(cells):
        return cells
    along = (_cell_centres(hf, cells) - _vec(grip_xy, 2, "grip_xy")) @ _unit_xy(float(jaw_yaw))
    return cells[np.abs(along) <= float(half_width_m) + 1e-9]


def _sweep_polygons(robot, grip_xy, yaw, *, finger_q_mm, inflate_m, half_width_m):
    """The closing sweeps: for each finger, the strip between the part edge (half_width_m from the grip along the
    jaw) and the column's inner edge, across the inflated column; None where the column reaches the part."""
    ranges = robot.finger_ranges_mm(finger_q_mm)
    jaw, up = _unit_xy(yaw), _up_xy(yaw)
    out = []
    for finger, side in (("finger1", 1.0), ("finger2", -1.0)):
        r1 = ranges[finger]["R1_mm"]
        r0 = ranges[finger]["R0_mm"]
        inner = (min(r1) if side > 0.0 else max(r1)) / 1000.0 - side * inflate_m
        start, end = side * half_width_m, inner
        if side * (end - start) <= 0.0:
            out.append(None)
            continue
        low, high = min(r0) / 1000.0 - inflate_m, max(r0) / 1000.0 + inflate_m
        centre = grip_xy + jaw * (start + end) / 2.0 + up * (low + high) / 2.0
        out.append(rect_polygon(centre, jaw, abs(end - start) / 2.0, (high - low) / 2.0))
    return tuple(out)


# ------------------------------------------------------------------------------------------------ P10 / P6 helpers
def clearance_retreat_m(plan: ContactPlan, contact_surface_z: float, part_top_z: float) -> tuple[float, Threshold]:
    """P10: the retreat after a fingertip contact on ``contact_surface_z`` that centres the pad band (R4: [g - 4.4,
    g + 11.6] mm top-down) on [contact surface, part top], clipped to 0..P10_MAX_RETREAT_MM."""
    return _retreat(_finite(contact_surface_z, "contact_surface_z"), _finite(part_top_z, "part_top_z"))


def _retreat(contact_surface_z, part_top_z):
    low, high = robot_facts.PAD_SPAN_ALONG_APPROACH_MM
    pad_centre_above_grip = -(low + high) / 2.0 / 1000.0           # 3.6 mm
    t_f = robot_facts.FINGERTIP_REACH_MM / 1000.0
    wanted = (contact_surface_z + part_top_z) / 2.0 - pad_centre_above_grip
    raw = wanted - (contact_surface_z + t_f)
    retreat = min(max(raw, 0.0), P10_MAX_RETREAT_MM / 1000.0)
    th = th_value("P10_retreat_m", retreat, "ROBOT",
                  "clip((surface + top) / 2 - pad centre - (surface + t_f), 0, P10_MAX_RETREAT_MM)",
                  surface_z=contact_surface_z, top_z=part_top_z, raw_m=raw)
    return retreat, th


def _palm_margin(part, sigma_h):
    if _is_fixture(part):
        return th_value("P6_palm_margin_m", 0.0, "DD", "P6: palm margin 0 on a fixture (move prior " +
                        part.move_prior + ")")
    return th_max("P6_palm_margin_m", Term("DD", P6_PALM_FLOOR_MM / 1000.0, "P6_PALM_FLOOR_MM"),
                  Term("ONLINE", P6_PALM_SIGMA_K * sigma_h, "P6_PALM_SIGMA_K*sigma_h"), unit="m")


def _p6_sigma_limit(p_min_m, pad_top_m, fixture):
    """Largest sigma_h for which the height_stop palm clause g* + p_min >= top + m_palm can hold (it is
    independent of the top): p_min - pad_top - max(floor, k sigma) - [max(floor, k sigma) for free parts] >= 0.
    None when even sigma_h = 0 fails. Exact for these piecewise-linear terms."""
    terms = [(P6_TARGET_FLOOR_MM / 1000.0, P6_TARGET_SIGMA_K)]
    if not fixture:
        terms.append((P6_PALM_FLOOR_MM / 1000.0, P6_PALM_SIGMA_K))
    available = p_min_m - pad_top_m

    def f(sigma):
        return available - sum(max(floor, k * sigma) for floor, k in terms)
    if f(0.0) < 0.0:
        return None
    points = sorted({0.0} | {floor / k for floor, k in terms})
    for left, right in zip(points, points[1:]):
        if f(right) < 0.0:
            return left + (right - left) * f(left) / (f(left) - f(right))
    slope = sum(k for _, k in terms)
    last = points[-1]
    return last + f(last) / slope


# ------------------------------------------------------------------------------------------------ evaluation
def _footprint(ctx, grip_xy, yaw, grip_z, sigma):
    """P7 on the capped field (a vision check): the hand footprint minus the finger slot, the part region and the
    scene's part_fill_cells (RC-1d), lower-bound parts raised (P-F); no sigma at a D7b argmax (RC-1c). Every other
    D7b fill stays a P7 blocker, including a fill copying the part's height that S-6 did not list (a separate raised
    object at the same height, or a band cell beyond the scene's reach / connectivity): which fills are the part's
    own halo is the scene's S-6 decision, not the planner's."""
    robot, hf = ctx.robot, ctx.hf
    poly = robot.hand_footprint_xy(grip_xy, _unit_xy(yaw))
    cells = hf.cells_in_polygon(poly)
    slot = robot.finger_slot_xy(grip_xy, _unit_xy(yaw), finger_q_mm=ctx.finger_q)
    in_slot = poly_contains(slot, _cell_centres(hf, cells)) if len(cells) else None
    s = ctx.field_p7.stats(cells, extra_skip=in_slot)
    sigma = max(sigma, _sigma_or(s["sigma"], 0.0))
    margin = th_max("P7_footprint_margin_m", Term("DD", P7_FLOOR_MM / 1000.0, "P7_FLOOR_MM"),
                    Term("ONLINE", P7_SIGMA_K * sigma, "P7_SIGMA_K*sigma_h"), unit="m")
    limit = grip_z + ctx.p_min - margin.value
    clearance = math.inf if s["z_max"] is None else limit - s["z_max"]
    ratio = math.inf if s["z_max"] is None else (grip_z + ctx.p_min - s["z_max"]) / margin.value
    ok = clearance >= -1e-9
    return {"ok": ok, "clearance_m": clearance, "ratio": ratio, "z_max": s["z_max"],
            "keepout": s["keepout"], "limit_z": limit, "raised_pid": s["raised_pid"],
            "blocking": None if ok else _blocking(ctx.field_p7, cells, limit, extra_skip=in_slot)}, margin


def _p7_sigma_limited(ctx, fp, grip_z_floor_only):
    """True when the P7 refusal is due to sigma alone: the footprint maximum would clear the floor-only margin at the
    grip the rule plans with sigma_h -> 0 (Appendix L 3 generalized: refine the height first, then re-plan)."""
    if fp["z_max"] is None:
        return False
    return bool(fp["z_max"] <= grip_z_floor_only + ctx.p_min - P7_FLOOR_MM / 1000.0 + 1e-9)


def _clear_z(field, robot, grip_xy, yaws, finger_q_mm, sigma_floor):
    """The swept hand (finger columns and hand footprint over ``yaws``): the lowest grip z that clears it by the P12
    margin, and the parts of that computation."""
    finger_cells, foot_cells = [], []
    for yaw in yaws:
        f1, f2 = robot.finger_columns_xy(grip_xy, _unit_xy(yaw), finger_q_mm=finger_q_mm)
        finger_cells += [field.hf.cells_in_polygon(f1), field.hf.cells_in_polygon(f2)]
        foot_cells.append(field.hf.cells_in_polygon(robot.hand_footprint_xy(grip_xy, _unit_xy(yaw))))
    fingers, foot = field.stats(_unique_cells(finger_cells)), field.stats(_unique_cells(foot_cells))
    sigma = max(sigma_floor, _sigma_or(fingers["sigma"], 0.0), _sigma_or(foot["sigma"], 0.0))
    margin = th_max("P12_turn_clearance_m", Term("DD", P12_FLOOR_M, "P12_FLOOR_M"),
                    Term("ONLINE", P12_SIGMA_K * sigma, "P12_SIGMA_K*sigma_h"), unit="m")
    needs = []
    if fingers["z_max"] is not None:
        needs.append(fingers["z_max"] + robot.t_f_m)
    if foot["z_max"] is not None:
        needs.append(foot["z_max"] - robot.p_min_m)
    required = (max(needs) + margin.value) if needs else -math.inf
    return required, margin, {"fingers_z_max": fingers["z_max"], "footprint_z_max": foot["z_max"],
                              "raised_pids": sorted({p for p in (fingers["raised_pid"], foot["raised_pid"]) if p})}


def _sweep_yaws(yaw_from, turn_world):
    if abs(turn_world) < 1e-9:
        return [yaw_from]
    count = int(math.ceil(abs(turn_world) / P12_SWEEP_STEP_RAD)) + 1
    return [yaw_from + turn_world * i / (count - 1) for i in range(count)]


def _p11(sigma_h):
    """P11 with the P-E cap: (Threshold max(P11_FLOOR_M, min(P11_SIGMA_K sigma_h + P11_OFFSET_M, P11_CAP_M)), first
    stage = the cap binds)."""
    wanted = P11_SIGMA_K * float(sigma_h) + P11_OFFSET_M
    value = max(P11_FLOOR_M, min(wanted, P11_CAP_M))
    if wanted <= P11_FLOOR_M:
        label, winner = "DD", "P11_FLOOR_M"
    elif wanted >= P11_CAP_M:
        label, winner = "DD", "P11_CAP_M"
    else:
        label, winner = "ONLINE", "P11_SIGMA_K*sigma_h + P11_OFFSET_M"
    th = th_value("P11_hover_clearance_m", value, label,
                  f"max(P11_FLOOR_M, min(P11_SIGMA_K*sigma_h + P11_OFFSET_M, P11_CAP_M)); winner {winner}",
                  sigma_h_m=float(sigma_h), wanted_m=wanted, cap_m=P11_CAP_M)
    return th, bool(wanted > P11_CAP_M + 1e-12)


def _hover(ctx, grip_xy):
    clearance, _ = _p11(ctx.sigma_h)
    hover = ctx.top + clearance.value + ctx.t_f
    static, margin, _ = _clear_z(ctx.field_p12, ctx.robot, grip_xy, [ctx.angle0], ctx.finger_q, ctx.sigma_h)
    return max(hover, static), clearance, margin


def _reject(family, yaw, stage, check, reason, values, thresholds, rim_angle=None):
    return CandidateRejection(family, float(yaw), f"{stage}:{check}", reason, jsonable(values), tuple(thresholds),
                              None if rim_angle is None else float(rim_angle))


def _width(ctx, yaw):
    """(w Meas, sigma_w mm, e_lat Threshold, class or None, thresholds, values) for the candidate yaw."""
    width = ctx.rim_width if ctx.rim is not None else ctx.part.width_at(yaw)
    if width is None:
        return None
    w_mm, sigma_w_mm = width.value * 1000.0, width.sigma * 1000.0
    sigma_align_mm = float(ctx.inputs.sigma_align_m) * 1000.0
    e_lat = P1_ALIGN_K * sigma_align_mm + sigma_w_mm + P1_PAD_MARGIN_MM
    th_e = th_value("P1_e_lat_mm", e_lat, "ONLINE", "P1_ALIGN_K*sigma_align + sigma_w + P1_PAD_MARGIN_MM",
                    sigma_align_mm=sigma_align_mm, sigma_w_mm=sigma_w_mm)
    feasible_max = ctx.sep_mm - P2_SIGMA_K * e_lat
    infeasible_min = ctx.sep_mm - P2_INFEASIBLE_MARGIN_MM + P2_SIGMA_K * sigma_w_mm
    th_f = th_value("P2_width_feasible_mm", feasible_max, "ONLINE", "sep - P2_SIGMA_K*e_lat (feasible when w <= it)",
                    sep_mm=ctx.sep_mm, e_lat_mm=e_lat)
    th_i = th_value("P2_width_infeasible_mm", infeasible_min, "ONLINE",
                    "sep - P2_INFEASIBLE_MARGIN_MM + P2_SIGMA_K*sigma_w (infeasible when w >= it)",
                    sep_mm=ctx.sep_mm, sigma_w_mm=sigma_w_mm)
    if w_mm >= infeasible_min - 1e-9:
        cls = None
    elif w_mm <= feasible_max + 1e-9:
        cls = "feasible"
    else:
        cls = "marginal"
    # G13+ (P-D) compares the close gap with the part's narrowest measured width over every jaw yaw
    rows = ctx.part.width_by_yaw or ()
    narrow = min([w_mm] + [float(m.value) * 1000.0 for _, m in rows if m is not None and math.isfinite(m.value)])
    return {"w_mm": w_mm, "sigma_w_mm": sigma_w_mm, "sep_mm": ctx.sep_mm, "e_lat_mm": e_lat, "class": cls,
            "w_narrow_mm": narrow, "thresholds": [th_e, th_f, th_i]}


def _sigma_contact(ctx, landing_sigma):
    return max(ctx.sigma_h, _sigma_or(landing_sigma, 0.0)) + SIGMA_VC_M


def _band(sigma_c):
    return th_max("P8_band_m", Term("DD", P8_FLOOR_M, "P8_FLOOR_M"),
                  Term("ONLINE", P8_SIGMA_K * sigma_c, "P8_SIGMA_K*(sigma_h + SIGMA_VC_M)"), unit="m")


def _bound(lowest_grip_z, sigma_c):
    value, label, text = _winner(Term("DD", P9_FLOOR_M, "P9_FLOOR_M"),
                                 Term("ONLINE", P9_SIGMA_K * sigma_c, "P9_SIGMA_K*(sigma_h + SIGMA_VC_M)"))
    return th_value("P9_bound_grip_z_m", lowest_grip_z - value, label,
                    f"lowest expected grip z - max(P9_FLOOR_M, P9_SIGMA_K*sigma_c); winner {text}",
                    lowest_grip_z=lowest_grip_z, sigma_c_m=sigma_c)


def _overlap_mm(pad_low, pad_high, part_low, part_top):
    return max(0.0, min(pad_high, part_top) - max(pad_low, part_low)) * 1000.0


def _progress(order, check):
    """Share of a depth rule's checks passed before ``check`` failed (structural: orders failures, no threshold)."""
    return order.index(check) / len(order)


def _close_limit(ctx):
    """P3 at the close (BP-4): the highest grip z at which the pads [g - pad bottom, g + pad top] still overlap the
    part's top by P3_MIN_OVERLAP_MM; the upper side of every P8 band is capped here."""
    limit = ctx.top + ctx.pad_bottom - P3_MIN_OVERLAP_MM / 1000.0
    return th_value("P3_close_limit_grip_z_m", limit, "ROBOT",
                    "top + pad bottom (R4) - P3_MIN_OVERLAP_MM: a contact above it is blocked_above_part",
                    top_z=ctx.top)


def _section_edge_m(ctx, grip_xy, yaw, inflate_m, half_width_m, finger, side):
    """P-J (remedy 1): (edge, extent) of the part on one finger's side, along the jaw from the grip. ``extent`` is the
    outer boundary (centre + cell/2) of the part's own cells (region + the scene's S-17 ring) whose centres lie inside
    that finger's sweep strip across the jaw (R0 range +- e_lat +- cell/2), None without such cells; ``edge`` is the
    farther of the model edge w/2 and that extent, never farther than w: the grip lies on the part, so its far edge
    is within one width of it. The grip is the pointed point, not the section centre, and Align does not re-centre
    the hand (FACT lb_smoke_001: grips 4-13.6 mm off the centre line)."""
    hf = ctx.hf
    half = float(half_width_m)
    cells = ctx.own_cells
    if not len(cells):
        return half, None
    r0 = ctx.robot.finger_ranges_mm(ctx.finger_q)[finger]["R0_mm"]
    low = min(r0) / 1000.0 - float(inflate_m) - hf.cell_m / 2.0
    high = max(r0) / 1000.0 + float(inflate_m) + hf.cell_m / 2.0
    rel = _cell_centres(hf, cells) - np.asarray(grip_xy, dtype=float)
    across = rel @ _up_xy(yaw)
    inside = (across >= low - 1e-9) & (across <= high + 1e-9)
    if not inside.any():
        return half, None
    extent = float(np.max(side * (rel[inside] @ _unit_xy(yaw)))) + hf.cell_m / 2.0
    return max(half, min(extent, 2.0 * half)), extent


def close_gate(ctx, grip_xy, yaw, inflate_m, half_width_m, sigma_w_mm):
    """GC-1 (judge J): per finger side, the farthest outward distance from the evaluated grip point along the jaw at
    which the part's own face is still expected: F = max(the P5 near band of that side (P-J section edge + e_lat +
    cell/2), the outer edge of the part's own cells in that finger's strip + the G13 band max(G13_FLOOR_MM,
    G13_SIGMA_K sigma_w)); the guard's limit R = F + GC_CONTACT_TOL_MM. Pure."""
    band = max(G13_FLOOR_MM, G13_SIGMA_K * float(sigma_w_mm)) / 1000.0
    fingers = {}
    for finger, side in (("finger1", 1.0), ("finger2", -1.0)):
        edge, extent = _section_edge_m(ctx, grip_xy, yaw, inflate_m, half_width_m, finger, side)
        near = edge + float(inflate_m) + ctx.hf.cell_m / 2.0
        face = near if extent is None else max(near, float(extent) + band)
        fingers[finger] = {"F_m": face, "R_m": face + GC_CONTACT_TOL_MM / 1000.0, "near_edge_m": near,
                           "own_extent_m": extent}
    return {"fingers": fingers, "band_g13_m": band}


def _sweeps_ok(ctx, grip_xy, yaw, inflate_m, half_width_m, limits):
    """Closing-sweep heights (part region excluded) against one limit per finger; (ok, values, close_verified).
    RC-1b: D7b obstacle fills and the measured cells whose centre lies within |along jaw| <= edge + e_lat + cell/2 of
    the grip (the part-edge uncertainty P1 already budgets) are verified by the close itself (G13 / G13+, co-motion)
    and skipped; those above the limit are counted as close_verified. Remedy 1: the edge is the P-J section edge of
    that finger's side (``_section_edge_m``, w/2 without own cells in the strip), and a measured cell whose sigma
    exceeds P5_SIGMA_RESOLUTION_M is unverified like a fill (RC-1e). Keep-out cells are never skipped; measured cells
    beyond the band still refuse."""
    polys = _sweep_polygons(ctx.robot, grip_xy, yaw, finger_q_mm=ctx.finger_q, inflate_m=inflate_m,
                            half_width_m=half_width_m)
    hf, field = ctx.hf, ctx.field
    jaw = _unit_xy(yaw)
    rows, ok, verified = [], True, 0
    for poly, limit, (finger, side) in zip(polys, limits, (("finger1", 1.0), ("finger2", -1.0))):
        if poly is None:
            rows.append({"z_max": None, "limit_z": limit, "close_verified": 0})
            continue
        edge, extent = _section_edge_m(ctx, grip_xy, yaw, inflate_m, half_width_m, finger, side)
        near_edge = edge + float(inflate_m) + hf.cell_m / 2.0
        cells = hf.cells_in_polygon(poly)
        skip, n_fill, n_near, n_imprecise = None, 0, 0, 0
        if len(cells):
            iy, ix = cells[:, 0], cells[:, 1]
            along = np.abs((_cell_centres(hf, cells) - np.asarray(grip_xy, dtype=float)) @ jaw)
            fills = field.unverified[iy, ix]
            imprecise = hf.measured[iy, ix] & (np.nan_to_num(hf.sigma[iy, ix], nan=0.0) > P5_SIGMA_RESOLUTION_M + 1e-12)
            near = hf.measured[iy, ix] & (along <= near_edge + 1e-9)
            skip = (fills | imprecise | near) & ~np.isfinite(field.overlay[iy, ix])
            with np.errstate(invalid="ignore"):
                above = skip & ~field.skip[iy, ix] & (np.nan_to_num(hf.z[iy, ix], nan=-np.inf) > limit + 1e-9)
            n_fill = int((above & fills).sum())
            n_imprecise = int((above & imprecise & ~fills & ~near).sum())
            n_near = int((above & ~fills).sum()) - n_imprecise
        s = field.stats(cells, extra_skip=skip)
        good = s["z_max"] is None or s["z_max"] <= limit + 1e-9
        ok &= good
        verified += n_fill + n_near + n_imprecise
        rows.append({"z_max": s["z_max"], "limit_z": limit, "ok": good,
                     "close_verified": n_fill + n_near + n_imprecise, "fills": n_fill, "near_edge": n_near,
                     "imprecise": n_imprecise, "near_edge_m": near_edge, "section_edge_m": edge,
                     "own_extent_m": extent,
                     "blocking": None if good else _blocking(field, cells, limit, extra_skip=skip)})
    return ok, rows, verified


def _touch_stop(ctx, c):
    order = TOUCH_CHECK_ORDER
    cols = c["columns"]
    for col in cols:
        if col.landing_z is None or col.known_fraction < P4_MIN_KNOWN_FRACTION - 1e-9:
            return {"fail": ("P4", "geometry_unmeasured",
                             {"landing_z": [k.landing_z for k in cols], "known": [k.known_fraction for k in cols]},
                             [ctx.p4], _progress(order, "P4"))}
    high, low = sorted(cols, key=lambda k: k.landing_z, reverse=True)
    g_contact = high.landing_z + ctx.t_f
    retreat, th10 = _retreat(high.landing_z, ctx.top)
    g_final = g_contact + retreat
    pad_low, pad_high = ctx.robot.pad_band_z(g_final)
    part_low = min(ctx.top, ctx.support if ctx.support is not None else high.landing_z)
    overlap = _overlap_mm(pad_low, pad_high, part_low, ctx.top)
    th3 = th_value("P3_pad_overlap_mm", P3_MIN_OVERLAP_MM, "ROBOT", "P3_MIN_OVERLAP_MM: half the 16 mm pad (R4)")
    base = [ctx.p4, th10, th3]
    values = {"contact_grip_z": g_contact, "retreat_m": retreat, "pads_z": [pad_low, pad_high],
              "part_band_z": [part_low, ctx.top], "overlap_mm": overlap}
    if overlap < P3_MIN_OVERLAP_MM - 1e-9:
        reason = "column_blocked" if pad_low > ctx.top - P3_MIN_OVERLAP_MM / 1000.0 else "no_pad_overlap"
        if reason == "column_blocked":
            # BP-2: a landing above top - P3 overlap + pad bottom - t_f leaves less than the P3 overlap whatever the
            # retreat (pad low = landing + t_f + retreat - pad bottom), so those column cells block
            land_max = ctx.top - P3_MIN_OVERLAP_MM / 1000.0 + ctx.pad_bottom - ctx.t_f
            values["blocking"] = _merge_blocking([_blocking(c["column_field"], c["column_field"].hf.cells_in_polygon(
                col.polygon_xy), land_max, verified=True) for col in cols])
        return {"fail": ("P3", reason, values, base, _progress(order, "P3"))}
    th5 = th_value("P5_sweep_margin_m", P5_SIGMA_K * ctx.sigma_h, "ONLINE", "P5_SIGMA_K*sigma_h",
                   sigma_h_m=ctx.sigma_h)
    limits = [col.landing_z + th5.value for col in cols]
    swept, rows, close_verified = _sweeps_ok(ctx, c["grip_xy"], c["yaw"], c["inflate_m"], c["half_width_m"], limits)
    values["sweeps"] = rows
    if not swept:
        values["blocking"] = _merge_blocking([row.get("blocking") for row in rows])
        return {"fail": ("P5", "column_blocked", values, base + [th5], _progress(order, "P5"))}
    # P6 palm clause at the CONTACT grip: the hand's lowest point on this descent (BP-2); the retreat is P3's business
    m_palm = _palm_margin(ctx.part, ctx.sigma_h)
    palm_margin = g_contact + ctx.p_min - ctx.top - m_palm.value
    values["palm_margin_m"] = palm_margin
    if palm_margin < -1e-9:
        # sigma-limited when the floor-only margin (sigma_h -> 0) would pass (Appendix L 3: refine the height first)
        floor_only = 0.0 if ctx.fixture else P6_PALM_FLOOR_MM / 1000.0
        values["refine_height"] = bool(g_contact + ctx.p_min - ctx.top - floor_only >= -1e-9)
        return {"fail": ("P6", "palm_clearance", values, base + [th5, m_palm], _progress(order, "P6_palm"))}
    fp, th7 = _footprint(ctx, c["grip_xy"], c["yaw"], g_contact, ctx.sigma_h)
    values["footprint"] = fp
    if not fp["ok"]:
        values["refine_height"] = _p7_sigma_limited(ctx, fp, g_contact)
        values["blocking"] = fp.get("blocking")
        return {"fail": ("P7", "footprint_blocked", values, base + [th5, m_palm, th7], _progress(order, "P7"))}
    cap = _close_limit(ctx)
    expected, ths, column_unverified = [], [], 0
    for col in sorted(cols, key=lambda k: k.landing_z, reverse=True):
        band = _band(_sigma_contact(ctx, col.landing_sigma))
        ths.append(band)
        # RC-1a: a D7b fill above the verified landing widens this column's band up to the P3 cap; the fills outside
        # the landing's own band are what only the contact can verify (the marginal label)
        unverified_grip = None
        if col.unverified_z is not None and col.unverified_z > col.landing_z + 1e-9:
            unverified_grip = col.unverified_z + ctx.t_f
        column_unverified += col.unverified_above(col.landing_z + band.value)
        expected.append(ExpectedContact("fingertip_landing", col.landing_z + ctx.t_f, band.value, col.finger,
                                        cap.value, unverified_grip_z=unverified_grip))
    bound = _bound(low.landing_z + ctx.t_f, _sigma_contact(ctx, low.landing_sigma))
    return {"rule": "touch_stop", "target": None, "bound": bound.value, "expected": tuple(expected),
            "retreat": retreat, "overlap_mm": overlap, "palm_margin_m": palm_margin, "footprint": fp,
            "sweep_ok": True, "column_unverified": column_unverified, "close_verified": close_verified,
            "unverified": {"columns": [k.unverified_z for k in cols]},
            "thresholds": base + [th5, m_palm, th7] + ths + [cap, bound]}


def _height_stop(ctx, c):
    """height_stop with IF-HS (gap round): the rule at the full-overlap target g*; a refusal at P7 is re-run once at
    g* + d, d = the P7 deficit (the footprint limit moves 1:1 with the grip z, every other check only relaxes or is
    re-checked: P3), and the raised plan is admitted only when P3 also holds with the part's top taken
    HS_P3_GUARD_SIGMA_K sigma_h lower; otherwise the P7 refusal stands with ``hs_retry`` evidence. A rule that passes
    at g* is returned unchanged."""
    out = _height_stop_at(ctx, c, 0.0)
    if "fail" not in out or out["fail"][0] != "P7":
        return out
    clearance = (out["fail"][2].get("footprint") or {}).get("clearance_m")
    if clearance is None or not math.isfinite(clearance) or clearance >= 0.0:
        return out
    raise_m = -float(clearance)
    again = _height_stop_at(ctx, c, raise_m)
    th_raise = th_value("HS_raise_m", raise_m, "ONLINE", "IF-HS: the P7 deficit at the full-overlap target g*")
    guard_mm = None
    if "fail" not in again:
        pad_low, pad_high = ctx.robot.pad_band_z(again["target"])
        part_low = min(ctx.top, ctx.support if ctx.support is not None else max(k.landing_z for k in c["columns"]))
        top_low = ctx.top - HS_P3_GUARD_SIGMA_K * ctx.sigma_h
        guard_mm = _overlap_mm(pad_low, pad_high, min(part_low, top_low), top_low)
        th_guard = th_value("HS_P3_guard_top_z_m", top_low, "ONLINE",
                            "IF-HS: P3 with the top HS_P3_GUARD_SIGMA_K*sigma_h lower (overlap >= P3_MIN_OVERLAP_MM)",
                            top_z=ctx.top, sigma_h_m=ctx.sigma_h, k=HS_P3_GUARD_SIGMA_K, overlap_mm=guard_mm)
        if guard_mm < P3_MIN_OVERLAP_MM - 1e-9:
            again = {"fail": ("P3", "no_pad_overlap", {"hs_guard_overlap_mm": guard_mm}, [th_raise, th_guard],
                              _progress(HEIGHT_CHECK_ORDER, "P3"))}
        else:
            again["hs"] = {"raise_m": raise_m, "guard_overlap_mm": guard_mm, "g_star_m": again["target"] - raise_m}
            again["thresholds"] = list(again["thresholds"]) + [th_raise, th_guard]
            return again
    out["fail"][2]["hs_retry"] = {"raise_m": raise_m, "check": again["fail"][0], "reason": again["fail"][1],
                                  "guard_overlap_mm": guard_mm}
    return out


def _height_stop_at(ctx, c, raise_m):
    """The height_stop checks with the stop at g* + ``raise_m`` (IF-HS; 0 = the full-overlap target g*)."""
    order = HEIGHT_CHECK_ORDER
    cols = c["columns"]
    offset, label, text = _winner(Term("DD", P6_TARGET_FLOOR_MM / 1000.0, "P6_TARGET_FLOOR_MM"),
                                  Term("ONLINE", P6_TARGET_SIGMA_K * ctx.sigma_h, "P6_TARGET_SIGMA_K*sigma_h"))
    g_star = ctx.top - ctx.pad_top - offset
    if raise_m:
        g_star = g_star + float(raise_m)
        text = f"{text}; raised by the IF-HS P7 deficit"
    th_target = th_value("P6_target_grip_z_m", g_star, label,
                         f"top - pad top (R4) - max(P6_TARGET_FLOOR_MM, P6_TARGET_SIGMA_K*sigma_h); winner {text}",
                         top_z=ctx.top, sigma_h_m=ctx.sigma_h)
    m_palm = _palm_margin(ctx.part, ctx.sigma_h)
    palm_margin = g_star + ctx.p_min - ctx.top - m_palm.value
    values = {"target_grip_z": g_star, "palm_margin_m": palm_margin}
    if palm_margin < -1e-9:
        limit = _p6_sigma_limit(ctx.p_min, ctx.pad_top, ctx.fixture)
        values.update(refine_height=limit is not None and ctx.sigma_h > limit, sigma_h_limit_m=limit,
                      sigma_h_m=ctx.sigma_h)
        return {"fail": ("P6", "palm_clearance", values, [th_target, m_palm], _progress(order, "P6_palm"))}
    for col in cols:
        if col.landing_z is None or col.known_fraction < P4_MIN_KNOWN_FRACTION - 1e-9:
            values.update(landing_z=[k.landing_z for k in cols], known=[k.known_fraction for k in cols])
            return {"fail": ("P4", "geometry_unmeasured", values, [th_target, m_palm, ctx.p4],
                             _progress(order, "P4"))}
    land_margin = th_max("P6_landing_margin_m", Term("DD", P6_LANDING_FLOOR_MM / 1000.0, "P6_LANDING_FLOOR_MM"),
                         Term("ONLINE", P6_LANDING_SIGMA_K * ctx.sigma_h, "P6_LANDING_SIGMA_K*sigma_h"), unit="m")
    landing_limit = g_star - ctx.pad_bottom - land_margin.value
    base = [th_target, m_palm, ctx.p4, land_margin]
    values.update(landing_z=[k.landing_z for k in cols], landing_limit_z=landing_limit)
    # RC-1a: P6-landing on the VERIFIED landings; D7b fills above the limit are listed (the guarded descent and P8
    # verify them: blocked_above_part never closes)
    column_unverified = sum(col.unverified_above(landing_limit) for col in cols)
    values["unverified"] = {"count": column_unverified, "max_z": [k.unverified_z for k in cols],
                            "limit_z": landing_limit}
    if any(col.landing_z > landing_limit + 1e-9 for col in cols):
        values["blocking"] = _merge_blocking([_blocking(c["column_field"], c["column_field"].hf.cells_in_polygon(
            col.polygon_xy), landing_limit, verified=True) for col in cols if col.landing_z > landing_limit + 1e-9])
        return {"fail": ("P6", "column_blocked", values, base, _progress(order, "P6_landing"))}
    swept, rows, close_verified = _sweeps_ok(ctx, c["grip_xy"], c["yaw"], c["inflate_m"], c["half_width_m"],
                                             [landing_limit, landing_limit])
    values["sweeps"] = rows
    if not swept:
        values["blocking"] = _merge_blocking([row.get("blocking") for row in rows])
        return {"fail": ("P5", "column_blocked", values, base, _progress(order, "P5"))}
    pad_low, pad_high = ctx.robot.pad_band_z(g_star)
    part_low = min(ctx.top, ctx.support if ctx.support is not None else max(k.landing_z for k in cols))
    overlap = _overlap_mm(pad_low, pad_high, part_low, ctx.top)
    th3 = th_value("P3_pad_overlap_mm", P3_MIN_OVERLAP_MM, "ROBOT", "P3_MIN_OVERLAP_MM: half the 16 mm pad (R4)")
    values.update(pads_z=[pad_low, pad_high], part_band_z=[part_low, ctx.top], overlap_mm=overlap)
    if overlap < P3_MIN_OVERLAP_MM - 1e-9:
        return {"fail": ("P3", "no_pad_overlap", values, base + [th3], _progress(order, "P3"))}
    fp, th7 = _footprint(ctx, c["grip_xy"], c["yaw"], g_star, ctx.sigma_h)
    values["footprint"] = fp
    if not fp["ok"]:
        values["refine_height"] = _p7_sigma_limited(ctx, fp, ctx.top - ctx.pad_top - P6_TARGET_FLOOR_MM / 1000.0)
        values["blocking"] = fp.get("blocking")
        return {"fail": ("P7", "footprint_blocked", values, base + [th3, th7], _progress(order, "P7"))}
    sigma_c = ctx.sigma_h + SIGMA_VC_M
    band = _band(sigma_c)
    bound = _bound(g_star, sigma_c)
    cap = _close_limit(ctx)                   # BP-4: a stop above it would close without the P3 overlap
    expected = (ExpectedContact("none_before_target", g_star, band.value, None, cap.value),)
    return {"rule": "height_stop", "target": g_star, "bound": bound.value, "expected": expected, "retreat": 0.0,
            "overlap_mm": overlap, "palm_margin_m": palm_margin, "footprint": fp, "sweep_ok": True,
            "column_unverified": column_unverified, "close_verified": close_verified,
            "unverified": values["unverified"], "thresholds": base + [th3, th7, band, cap, bound]}


def _evaluate(ctx, family, yaw, rules, rim_angle=None, grip_xy=None):
    """One candidate: a ContactPlan for the first depth rule (preference order) that passes, else a rejection.
    ``rim_angle`` names a rim candidate's rim point (None on a rim = the yaw itself, finger 1 outside)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if family not in SLICE_FAMILIES:
        raise NotImplementedError("frontal families are reserved in this build (contract §C0.3)")
    yaw = _wrap(yaw)
    rim_angle = None if ctx.rim is None else _wrap(yaw if rim_angle is None else rim_angle)
    part, robot = ctx.part, ctx.robot

    def reject(stage, check, reason, values, thresholds):
        return _reject(family, yaw, stage, check, reason, values, thresholds, rim_angle)
    grip = _grip_point(part, yaw, ctx.purpose, ctx.rim, rim_angle)
    if grip_xy is not None:                                     # PC-1 (judge J): the realized hand xy
        grip = np.r_[_vec(grip_xy, 2, "grip_xy"), grip[2]]
    grip_xy = grip[:2]
    jaw, approach, quaternion = _orientation(robot, yaw)
    turn, sense, budget = _yaw_turn(ctx.inputs, quaternion)
    values = {"grip_point_xyz": grip, "yaw_turn_rad": turn}
    if rim_angle is not None:
        values["rim_angle_rad"] = rim_angle
    if abs(turn) > budget + 1e-9:
        th = th_value("K6_yaw_budget_rad", budget, "ROBOT", "R12 budget left in the turn's sense", sense=sense)
        return reject("orient", "R12", "yaw_budget", values, [th])
    width = _width(ctx, yaw)
    if width is None:
        return reject("width", "P2", "geometry_unmeasured", values, [])
    ths = list(width.pop("thresholds"))
    values["width"] = width
    if width["class"] is None:
        return reject("width", "P2", "part_too_wide", values, ths)
    inflate_m = width["e_lat_mm"] / 1000.0
    half_width_m = width["w_mm"] / 1000.0 / 2.0
    th16 = th_value("P16_column_inflate_m", inflate_m, "ONLINE", "e_lat (P1) on every side of each column",
                    finger_q_mm=ctx.finger_q)
    ths.append(th16)
    # P16 (BP-1): only the measured cross-section the pads close on is skipped; the part's other cells land fingers
    section = section_cells(ctx.region_cells, ctx.hf, grip_xy, yaw, half_width_m)
    column_field = ctx.field_all.excluding(section)
    cols = _columns(column_field, grip_xy, yaw, robot, inflate_m=inflate_m,
                    support_z=ctx.support, finger_q_mm=ctx.finger_q, sigma_floor=ctx.sigma_h,
                    support_sigma=ctx.support_sigma)
    values["columns"] = [k.to_dict() for k in cols]
    values["section_cells"] = int(len(section))
    hover, th11, _ = _hover(ctx, grip_xy)
    turn_world = -turn                                          # top-down: about R2 ~ -z
    required, th12, sweep = _clear_z(ctx.field_p12, robot, grip_xy, _sweep_yaws(ctx.angle0, turn_world),
                                     ctx.finger_q, ctx.sigma_h)
    # P12: the Orient runs at the hover, so the hover is raised until the turned hand clears (BP-10); a top-down
    # candidate is never refused for a sweep a higher hover clears
    hover_raised = max(0.0, required - hover)
    hover = max(hover, required)
    p12_clearance = hover - required
    values.update(hover_grip_z=hover, p12=dict(sweep, required_grip_z=required, clearance_m=p12_clearance,
                                               hover_raised_m=hover_raised))
    ths += [th11, th12]
    if not p12_clearance >= -1e-9:
        return reject("orient", "P12", "turned_hand_blocked", values, ths)
    candidate = {"grip_xy": grip_xy, "yaw": yaw, "columns": cols, "inflate_m": inflate_m,
                 "half_width_m": half_width_m, "column_field": column_field}
    failures, result = {}, None
    for rule in rules:
        if rule == "touch_stop":
            outcome = _touch_stop(ctx, candidate)
        elif rule == "height_stop":
            outcome = _height_stop(ctx, candidate)
        elif rule == "palm_stop":
            outcome = {"fail": ("P13", "no_feasible_candidate",
                                {"note": "palm_stop needs a verified-fixed DOF; the DOF probe is reserved [C-47]"}, [],
                                0.0)}
        else:
            outcome = {"fail": ("family", "no_feasible_candidate", {"note": "face_stop needs a frontal family"}, [],
                                0.0)}
        if "fail" in outcome:
            failures[rule] = outcome["fail"]
            continue
        result = outcome
        break
    if result is None:
        if not failures:
            return reject("depth", "rules", "no_feasible_candidate", values, ths)
        # the representative failure: the rule that got furthest through its checks (the preferred rule on a tie)
        last = max(failures, key=lambda rule: failures[rule][4])
        check, reason = failures[last][0], failures[last][1]
        values["by_rule"] = {rule: {"check": f[0], "reason": f[1], "values": f[2], "progress": f[4]}
                             for rule, f in failures.items()}
        values["refine_height"] = any(bool(f[2].get("refine_height")) for f in failures.values())
        # P-F evidence: the lower-bound parts whose top_upper_z raise decided a P7 refusal (BR surveys them first)
        values["lower_bound_blockers"] = sorted({(f[2].get("footprint") or {}).get("raised_pid")
                                                 for f in failures.values()} - {None})
        # BP-2: the cells that blocked each rule (P3 column / P5 sweep / P6 landing / P7 footprint); refine_cells when
        # some rule failed only on cells the scene marks unconfirmed (BS-2): BR-5 re-measures them and re-plans
        blocks = [f[2].get("blocking") for f in failures.values() if f[2].get("blocking")]
        values["blocking"] = _merge_blocking(blocks)
        values["refine_cells"] = any(bool(b.get("all_unconfirmed")) for b in blocks)
        # the cells to re-measure (BR-5): the unconfirmed cells of the rules that failed only on unconfirmed cells
        values["blocking_cells"] = _merge_cells([b["cells"] for b in blocks if b.get("all_unconfirmed")])
        limit = next((f[2].get("sigma_h_limit_m") for f in failures.values() if "sigma_h_limit_m" in f[2]), None)
        values["sigma_h_limit_m"] = limit
        all_ths = ths + [t for f in failures.values() for t in f[3]]
        return reject(last, check, reason, values, all_ths)
    p17 = None
    if ctx.held_centre is not None:
        p17 = bool(float(_up_xy(yaw) @ (ctx.held_centre - grip_xy)) >= 0.0)
    ratios = [(width["sep_mm"] - width["w_mm"]) / (P2_SIGMA_K * width["e_lat_mm"]),
              result["overlap_mm"] / P3_MIN_OVERLAP_MM, result["footprint"]["ratio"]]
    score = min(ratios)
    if result["rule"] == "touch_stop":
        score += SCORE_TOUCH_BONUS
    score -= SCORE_TURN_PENALTY_PER_QUARTER * abs(turn) / (math.pi / 2.0)
    # P-A label (remedy 1 §2.6): a candidate admitted only through unverified column fills or close-verified sweep
    # cells is marginal, ranked after clean candidates; BR descends it with the P14 ladder
    unverified_cells = int(result["column_unverified"]) + int(result["close_verified"])
    # F6-B: a column landing on D7a plane fills only is unverified; IF-PV: a plan admitted with the parent-sigma P7
    # reading rests on inferred cells; both are marginal (P14 ladder, GC-1 active), unverified_cells unchanged (RC-1a)
    plane_columns = sum(1 for col in cols if col.landing_plane_only)
    unverified = unverified_cells > 0 or plane_columns > 0 or ctx.p7_reading is not None
    admission = "marginal" if unverified else "clean"
    if width["class"] == "marginal" or admission == "marginal":
        score -= SCORE_MARGINAL_PENALTY
    if p17:
        score += P17_BONUS
    checks = {"P3_overlap_mm": result["overlap_mm"], "P5_sweep_ok": result["sweep_ok"],
              "P6_palm_margin_mm": result["palm_margin_m"] * 1000.0,
              "P7_clearance_mm": result["footprint"]["clearance_m"] * 1000.0, "P12_clearance_m": p12_clearance,
              "P17": p17, "P12_hover_raised_m": hover_raised, "unverified_cells": unverified_cells,
              "column_unverified": int(result["column_unverified"]), "close_verified": int(result["close_verified"]),
              "unverified": result.get("unverified"), "admission": admission, "axis_source": ctx.axis["source"],
              "part_fill_cells": int(len(ctx.part_fill_cells)), "part_ring_cells": int(len(ctx.ring_cells)),
              "gate": dict(close_gate(ctx, grip_xy, yaw, inflate_m, half_width_m, width["sigma_w_mm"]),
                           grip_xy=[float(v) for v in grip_xy], jaw_xy=[float(v) for v in jaw[:2]],
                           span_hi_mm=float(width["w_mm"] + max(G13_FLOOR_MM, G13_SIGMA_K * width["sigma_w_mm"])),
                           tol_mm=GC_CONTACT_TOL_MM, pad_face_offset_mm=robot_facts.PAD_SEPARATION_OFFSET_MM / 2.0),
              "lower_bound_parts": sorted(set(ctx.field_p7.raised_pids) | set(ctx.field_p12.raised_pids)),
              "plane_landing_columns": int(plane_columns)}
    if ctx.p7_reading is not None:
        checks["p7_reading"] = ctx.p7_reading                    # IF-PV: every re-evaluation reads P7 the same way
    if result.get("hs"):
        checks["hs_raise_mm"] = float(result["hs"]["raise_m"]) * 1000.0     # IF-HS: the stop raised by the P7 deficit
        checks["hs_guard_overlap_mm"] = float(result["hs"]["guard_overlap_mm"])
    return ContactPlan(
        pid=part.pid, family=family, depth_rule=result["rule"], purpose=ctx.purpose,
        jaw_world=tuple(float(v) for v in jaw), approach_world=tuple(float(v) for v in approach),
        quaternion_xyzw=tuple(float(v) for v in quaternion), yaw_turn_rad=turn,
        grip_point_xyz=tuple(float(v) for v in grip), hover_grip_z=float(hover),
        target_grip_z=None if result["target"] is None else float(result["target"]),
        bound_grip_z=float(result["bound"]), expected=result["expected"],
        clearance_retreat_m=float(result["retreat"]), columns=cols,
        width={k: width[k] for k in ("w_mm", "sigma_w_mm", "sep_mm", "e_lat_mm", "class", "w_narrow_mm")},
        checks=jsonable(checks), align_mode="across_bar" if part.is_elongated() else "both",
        score=float(score), thresholds=tuple(ths + result["thresholds"]), rim_angle_rad=rim_angle,
        move_prior=part.move_prior, admission=admission)


def _rules(depth_rules):
    if depth_rules is None:
        return SLICE_DEPTH_RULES
    rules = tuple(depth_rules)
    unknown = [r for r in rules if r not in DEPTH_RULES]
    if unknown:
        raise ValueError(f"unknown depth rules {unknown}")
    return tuple(r for r in DEPTH_RULES if r in rules)


def _plannable(part):
    if part.provisional:
        return "provisional"
    if part.status != "registered":
        return f"status {part.status}"
    if part.xyz is None or part.top_z is None or part.sigma_h_m is None:
        return "no measured point, top or height sigma"
    return None


def _width_unmeasured_values(part):
    """P-I: the refusal values of a record without a width function, else None. ``width_state`` tells the two cases
    apart: "unmeasurable" = ``width_by_yaw == ()`` (K-5: the scene measured the region and delivers no width, S-13's
    collapsed region; BR re-surveys it, BR-1 on region_collapsed) and "not_measured" = ``width_by_yaw`` None (the
    record carries no width function at all; nothing says its region collapsed)."""
    rows = part.width_by_yaw
    if rows:
        return None
    check = part.axis_check if isinstance(part.axis_check, Mapping) else {}
    if rows is None:
        state, note = "not_measured", ("no width function in the record (width_by_yaw None: never measured); P2 "
                                       "cannot be evaluated: register the part's region and re-plan")
    else:
        state, note = "unmeasurable", ("the scene delivered no width (K-5 width_by_yaw == (): S-13 collapsed "
                                       "region); P2 cannot be evaluated: re-survey the part (BR-1) and re-plan")
    return {"width_unmeasured": True, "width_state": state, "width_rows": 0,
            "region_collapsed": check.get("region_collapsed"), "note": note}


def _lower_bound_values(part):
    """P-F: the refusal values of contact planning on a lower-bound part (refine the height first, rulings item 8)."""
    if not part.top_is_lower_bound():
        return None
    return {"refine_height": True, "top_bound": part.top_bound, "top_upper_z": part.top_upper_z,
            "note": "the top is only a lower bound (rulings R2): refine the height first (survey_hover / probe)"}


def evaluate_candidate(part: PartRecord, hf: HeightField, inputs: PlannerInputs, family: str, yaw: float, keepouts,
                       *, purpose: str, next_held: PartRecord | None = None,
                       depth_rules: Sequence[str] | None = None, rim_angle: float | None = None,
                       part_fill_cells=None, parts: Sequence[PartRecord] = (),
                       part_ring_cells=None, grip_xy=None, p7_reading: str | None = None
                       ) -> ContactPlan | CandidateRejection:
    """One candidate (family, yaw): P2 three-way width; the depth rules in preference order touch_stop > face_stop >
    height_stop > palm_stop (the slice offers touch_stop and height_stop; palm_stop never, P13 [C-47]) with P3, P4
    (known = measured OR filled), P5, P6, P7 (excluding the part region and the finger slot), P10, P12 and P17; the
    first rule that passes gives the plan. ``depth_rules`` restricts the rules (test overrides, "the next rule").
    ``rim_angle`` (rim parts only, additive): the rim point; None = the yaw itself (finger 1 outside).
    Review 1 (additive): ``part_fill_cells`` (n, 2) ij = the scene's S-6 part-band fills, excluded from P7 only
    (RC-1d); ``parts`` = other records whose lower-bound cells P7 and P12 raise to their top_upper_z (P-F; a
    lower-bound ``next_held`` is raised too). The yaw is the caller's: the P-C / P-H axis rule gates plan_contact's
    candidates, not an explicit yaw. A lower-bound part itself is refused with refine_height (P-F); a record without
    a width function at P2 (P-I, ``width_unmeasured``).
    Gap round (additive): ``p7_reading`` "parent_sigma" reads P7 as the IF-PV fallback admitted the plan (pass the
    plan's ``checks.get("p7_reading")`` when re-evaluating it, PC-1); the admission is then marginal."""
    reason = _plannable(part)
    if reason is not None:
        return _reject(family, yaw, "part", "registration", "geometry_unmeasured", {"note": reason}, [])
    lower = _lower_bound_values(part)
    if lower is not None:
        return _reject(family, yaw, "part", "top_bound", "geometry_unmeasured", lower, [])
    unmeasured = _width_unmeasured_values(part)
    if unmeasured is not None:
        return _reject(family, yaw, "width", "P2", "geometry_unmeasured", unmeasured, [])
    ctx = _Context(part, hf, inputs, keepouts, purpose, next_held, part_fill_cells=part_fill_cells, parts=parts,
                   part_ring_cells=part_ring_cells, p7_reading=p7_reading)
    return _evaluate(ctx, family, yaw, _rules(depth_rules), rim_angle, grip_xy=grip_xy)


def representative_rejection(result):
    """CJ-1: the depth-stage rejection of ``result`` (an Infeasible) with the highest by_rule progress, skipping
    orient / width / part / axis rules; None without one."""
    best, best_p = None, -1.0
    for r in getattr(result, "rejections", ()) or ():
        rule = str(r.rule or "")
        if rule.startswith(("orient:", "width:", "part:", "axis:")):
            continue
        by_rule = (r.values or {}).get("by_rule") or {}
        prog = max([float(v.get("progress", 0.0)) for v in by_rule.values()] or [0.0])
        if prog > best_p:
            best, best_p = r, prog
    return best


def revalidate(plan: ContactPlan, part: PartRecord, hf: HeightField, inputs: PlannerInputs, keepouts, *,
               exclude: Sequence[tuple] = (), depth_rules: Sequence[str] | None = None,
               **kwargs) -> ContactPlan | Infeasible:
    """CI-2 (the sigma re-validation after Align): the EXECUTING candidate alone (its family, yaw and rim angle; the
    depth rules ``exclude`` leaves it, in preference order) re-evaluated on ``inputs`` (the measured sigma_align and
    the realized pose), instead of a fresh search whose best key is compared with it. A ContactPlan when that
    candidate still passes every check; else Infeasible with the candidate's own rejection (never "the best candidate
    changed": a candidate the hand is aligned for is kept or refused on its own checks). When the search's best key
    equals the executing candidate's this returns the same plan as plan_contact (the same _evaluate call). IF-PV: a
    plan admitted with the parent-sigma P7 reading is re-validated with the same reading (``checks["p7_reading"]``)."""
    kwargs.setdefault("p7_reading", (plan.checks or {}).get("p7_reading"))
    axis = jaw_axis(part, hf)
    if axis["refine_axis"]:          # the same P-C / P-H gate plan_contact applies before any candidate
        rejection = _reject(plan.family, plan.yaw_rad, "axis", "D10", "geometry_unmeasured",
                            {k: v for k, v in axis.items() if k != "thresholds"}, axis["thresholds"])
        return Infeasible(part.pid, "geometry_unmeasured", (rejection,), tuple(axis["thresholds"]), refine_axis=True)
    skip_all, skip = _exclusion(exclude, plan.family, plan.yaw_rad, plan.rim_angle_rad)
    rules = tuple(r for r in _rules(depth_rules) if r not in skip)
    if skip_all or not rules:
        return Infeasible(part.pid, "no_feasible_candidate", (), ())
    result = evaluate_candidate(part, hf, inputs, plan.family, plan.yaw_rad, keepouts, depth_rules=rules,
                                rim_angle=plan.rim_angle_rad, **kwargs)
    if isinstance(result, ContactPlan):
        return result
    return Infeasible(part.pid, result.reason, (result,), tuple(result.thresholds))


def pv_fallback_applies(result) -> bool:
    """IF-PV trigger: ``result`` is an Infeasible with a depth rejection whose P7 refusal (any depth rule) lists at
    least one D7b blocker (BP-2 ``blocking["inferred"]``)."""
    if not isinstance(result, Infeasible):
        return False
    for rejection in result.rejections or ():
        values = rejection.values or {}
        rows = list((values.get("by_rule") or {}).values())
        if not rows and str(rejection.rule or "").endswith(":P7"):
            rows = [{"check": "P7", "values": values}]
        for row in rows:
            if row.get("check") != "P7":
                continue
            blocking = (row.get("values") or {}).get("blocking") or {}
            if int(blocking.get("inferred") or 0) > 0:
                return True
    return False


def _exclusion(exclude, family, yaw, rim_angle=None):
    """(all rules excluded, set of excluded rules) for the candidate. Items: (family, yaw) = every rule at that jaw
    yaw (both rim points of a rim jaw); (family, yaw, rule) = that rule there; (family, yaw, rule or None, rim angle)
    = that rule (None: every rule) of the rim candidate at that rim angle only (ContactPlan.key / candidate_key)."""
    rules = set()
    for item in exclude or ():
        item = tuple(item)
        if len(item) < 2 or item[0] != family or not same_candidate_angle(item[1], yaw):
            continue
        if len(item) >= 4 and (rim_angle is None or item[3] is None or not same_candidate_angle(item[3], rim_angle)):
            continue
        if len(item) == 2 or item[2] is None:
            return True, set(DEPTH_RULES)
        rules.add(item[2])
    return False, rules


def plan_contact(part: PartRecord, hf: HeightField, inputs: PlannerInputs, keepouts, *, purpose: str,
                 next_held: PartRecord | None = None, exclude: Sequence[tuple] = (),
                 depth_rules: Sequence[str] | None = None, part_fill_cells=None,
                 parts: Sequence[PartRecord] = (), part_ring_cells=None) -> ContactPlan | Infeasible:
    """§4.6: best = max(plans, key=(not marginal, score)) over families_for x jaw_yaws (x both rim points of a rim
    jaw, rim_candidates), smaller turns first on a tie. ``exclude`` holds the candidates already tried: (family,
    yaw) excludes every rule at that yaw, (family, yaw, depth_rule) only that rule (ContactPlan.key()); a rim
    candidate is named with its rim angle as a 4th item (ContactPlan.key / candidate_key, _exclusion). A
    provisional or unregistered part is refused (geometry_unmeasured); Infeasible.reason is the most frequent
    rejection reason. "marginal" covers the P2 width class and (P-A) a candidate admitted only through unverified /
    close-verified cells.
    Review 1 (additive): ``part_fill_cells`` = the scene's S-6 ``part_fill_cells(pid)`` (P7 only, RC-1d); ``parts`` =
    other PartRecords whose lower-bound cells P7 and P12 raise to top_upper_z (P-F; a lower-bound ``next_held`` too).
    A lower-bound part is refused (geometry_unmeasured with refine_height, P-F); a non-elongated part whose jaw axis
    is unmeasured (P-C, jaw_axis) gives ``Infeasible("geometry_unmeasured", refine_axis=True)``.
    Review 2: a record without a width function (``width_by_yaw == ()``, K-5) is refused at P2 before any candidate
    (P-I, one "width:P2" rejection flagged ``width_unmeasured``; refine_axis reports the axis state); a record whose
    axis the scene refused as AXIS_INCONSISTENT or whose region collapsed asks to refine the axis without consulting
    the width harmonic (P-H, axis_refusal).
    Live-first blockers (grip-part lens BP-2, additive): a refusal at P3 (column), P5, P6 (landing) or P7 lists its
    blocking cells (values "blocking": ij, xyz, z, sigma, source, unconfirmed); ``refine_cells`` (on the rejection and
    the Infeasible) when some rule failed only on cells the scene marks unconfirmed (``hf.unconfirmed``, BS-2;
    ``needs_cell_refinement``, ``blocking_cells``); every Infeasible carries ``inputs``, a snapshot of this call's
    inputs (keys as the gate's plan_inputs pickles: "record", "hf" (deep copy), "inputs", "keepouts", "purpose",
    "next_held", "exclude", "depth_rules", "part_fill_cells", "parts", "part_ring_cells") for the caller to save on
    every refusal.
    Grasp-refusal decision J (additive): ``part_ring_cells`` = the scene's S-17 ``part_ring_cells(pid)``; the part's
    own cells (region + ring) set each finger's P-J section edge of the RC-1b near band (``_section_edge_m``) and
    the GC-1 close gate (``close_gate``, ``checks["gate"]``, never an admission); P7 still excludes only
    ``part_fill_cells``. RC-1e: a measured sweep cell with sigma above P5_SIGMA_RESOLUTION_M is close-verified.
    Gap round IF-PV (the P7 fallback): only when this search (IF-HS included) is Infeasible and some P7 refusal lists
    D7b blockers (``pv_fallback_applies``), the search runs once more with P7 reading every D7b fill at its parent +
    PV_PARENT_SIGMA_K sigma_parent (``p7_reading="parent_sigma"``); a plan found so is marginal and carries
    ``checks["p7_reading"]``; otherwise the first search's Infeasible is returned."""
    kwargs = dict(purpose=purpose, next_held=next_held, exclude=exclude, depth_rules=depth_rules,
                  part_fill_cells=part_fill_cells, parts=parts, part_ring_cells=part_ring_cells)
    result = _plan_contact(part, hf, inputs, keepouts, **kwargs)
    if not isinstance(result, Infeasible):
        return result
    if pv_fallback_applies(result):
        fallback = _plan_contact(part, hf, inputs, keepouts, p7_reading="parent_sigma", **kwargs)
        if isinstance(fallback, ContactPlan):
            return fallback
    snapshot = {"record": part, "hf": copy.deepcopy(hf), "inputs": inputs, "keepouts": tuple(keepouts or ()),
                "purpose": purpose, "next_held": next_held, "exclude": tuple(exclude or ()),
                "depth_rules": None if depth_rules is None else tuple(depth_rules),
                "part_fill_cells": None if part_fill_cells is None else np.array(part_fill_cells, copy=True),
                "part_ring_cells": None if part_ring_cells is None else np.array(part_ring_cells, copy=True),
                "parts": tuple(parts or ()), "planner_version": PLANNER_VERSION}
    refine = bool(result.refine_cells) or any(bool(r.values.get("refine_cells")) for r in result.rejections)
    return dataclass_replace(result, refine_cells=refine, inputs=snapshot)


def _plan_contact(part: PartRecord, hf: HeightField, inputs: PlannerInputs, keepouts, *, purpose: str,
                  next_held: PartRecord | None = None, exclude: Sequence[tuple] = (),
                  depth_rules: Sequence[str] | None = None, part_fill_cells=None,
                  parts: Sequence[PartRecord] = (), part_ring_cells=None,
                  p7_reading: str | None = None) -> ContactPlan | Infeasible:
    """plan_contact without the BP-2 input snapshot and the IF-PV fallback (one search with ``p7_reading``)."""
    reason = _plannable(part)
    if reason is not None:
        return Infeasible(part.pid, "geometry_unmeasured",
                          (_reject("top_down", 0.0, "part", "registration", "geometry_unmeasured", {"note": reason},
                                   []),), ())
    lower = _lower_bound_values(part)
    if lower is not None:
        return Infeasible(part.pid, "geometry_unmeasured",
                          (_reject("top_down", 0.0, "part", "top_bound", "geometry_unmeasured", lower, []),), ())
    unmeasured = _width_unmeasured_values(part)
    if unmeasured is not None:
        # P-I: one P2 refusal before any candidate; refine_axis reports the axis state (P-H / P-C) for BR
        axis = jaw_axis(part, hf)
        unmeasured.update(axis_source=axis["source"], axis_refusal=axis["axis_refusal"])
        rejection = _reject("top_down", _jaw_angle(inputs.pose.R), "width", "P2", "geometry_unmeasured", unmeasured,
                            [])
        return Infeasible(part.pid, "geometry_unmeasured", (rejection,), (), refine_axis=bool(axis["refine_axis"]))
    ctx = _Context(part, hf, inputs, keepouts, purpose, next_held, part_fill_cells=part_fill_cells, parts=parts,
                   part_ring_cells=part_ring_cells, p7_reading=p7_reading)
    if ctx.axis["refine_axis"]:
        axis = {k: v for k, v in ctx.axis.items() if k != "thresholds"}
        rejection = _reject("top_down", ctx.angle0, "axis", "D10", "geometry_unmeasured",
                            dict(axis, refine_axis=True), ctx.axis["thresholds"])
        return Infeasible(part.pid, "geometry_unmeasured", (rejection,), tuple(ctx.axis["thresholds"]),
                          refine_axis=True)
    allowed = _rules(depth_rules)
    plans, rejections = [], []
    for family in families_for(part, inputs):
        for yaw, rim_angle in rim_candidates(part, family, inputs, axis=ctx.axis):
            skip_all, skip_rules = _exclusion(exclude, family, yaw, rim_angle)
            rules = tuple(r for r in allowed if r not in skip_rules)
            if skip_all or not rules:
                continue
            result = _evaluate(ctx, family, yaw, rules, rim_angle)
            (plans if isinstance(result, ContactPlan) else rejections).append(result)
    if plans:
        return max(plans, key=lambda p: (not p.marginal, p.score))
    if not rejections:
        return Infeasible(part.pid, "no_feasible_candidate", (), ())
    reason = Counter(r.reason for r in rejections).most_common(1)[0][0]
    seen, ths = set(), []
    for rejection in rejections:
        for th in rejection.thresholds:
            key = (th.name, th.value)
            if key not in seen:
                seen.add(key)
                ths.append(th)
    return Infeasible(part.pid, reason, tuple(rejections), tuple(ths))


# ------------------------------------------------------------------------------------------------ hover (P11)
def _visibility_shift_m(point, target, pose, f_px, band_top_row, template_px):
    """P-2 (remedy 1): the distance (m) the hover must move along -R0 so that the part's predicted wrist row (the
    current orientation held) lies at least kappa VIS_PROBE_LEG_M + template_px above the fingertip band's top row
    (the r0-out probe leg moves it down by kappa D2); 0 when it already does."""
    R = np.asarray(pose.R, dtype=float)
    d = np.asarray(point, dtype=float) - (np.asarray(target, dtype=float) + (np.asarray(pose.camera, dtype=float)
                                                                             - np.asarray(pose.grip, dtype=float)))
    depth = float(d @ R[:, 2])
    if not depth > 0.0:
        return 0.0
    kappa = float(f_px) / depth
    row = float(_geometry().principal_point()[1]) - float(f_px) * float(d @ R[:, 0]) / depth
    excess = row - (float(band_top_row) - (kappa * VIS_PROBE_LEG_M + float(template_px)))
    return max(0.0, excess / kappa)


POINTING_MODES = ("fine", "coarse", "auto")


def _predict_wrist(point, cov, hover_grip_xyz, pose, f_px):
    """(u, v, sigma_uv, depth) of ``point`` in the wrist camera with the grip at ``hover_grip_xyz`` and the current
    orientation held (geometry.project_direction for the pixel; sigma_uv through the projection Jacobian, the scene's
    predicted_uv rule); None behind the camera."""
    R = np.asarray(pose.R, dtype=float)
    camera = np.asarray(hover_grip_xyz, dtype=float) + (np.asarray(pose.camera, dtype=float)
                                                       - np.asarray(pose.grip, dtype=float))
    d = np.asarray(point, dtype=float) - camera
    depth = float(d @ R[:, 2])
    uv = _geometry().project_direction(R, d, float(f_px))
    if uv is None or not depth > 0.0:
        return None
    a, b, f = float(d @ R[:, 1]), float(d @ R[:, 0]), float(f_px)
    J = np.vstack([f * (R[:, 1] / depth - a * R[:, 2] / (depth * depth)),
                   -f * (R[:, 0] / depth - b * R[:, 2] / (depth * depth))])
    cov = np.zeros((3, 3)) if cov is None else np.asarray(cov, dtype=float)
    sigma = math.sqrt(max(float(np.linalg.eigvalsh(J @ cov @ J.T).max()), 0.0))
    return float(uv[0]), float(uv[1]), sigma, depth


def pointing_frame(part: PartRecord, hover_grip_xyz, pose: HandPose, *, f_px: float, band_top_row: float,
                   pointing: str = "auto") -> dict:
    """BP-1 (grip-part lens) [ONLINE]: does a hover (the grip at ``hover_grip_xyz``, the current orientation held)
    frame the part for pointing it? The part's predicted wrist row v and sigma_uv (``_predict_wrist``); the mode is
    ``pointing`` ("fine" / "coarse") or, "auto", coarse-first when sigma_uv > FRAMING_COARSE_SIGMA_PX (rulings item
    9). Fine: the fine crop's bottom v + FRAMING_CROP_HALF_PX lies FT_PROBE_RADIUS_PX above the fingertip band's top
    row; coarse: v + sigma_uv does; both keep P-2's own margin kappa VIS_PROBE_LEG_M + FT_PROBE_RADIUS_PX. Returns
    {"ok", "mode", "row", "col", "sigma_uv_px", "kappa_px_per_m", "margin_px", "framing_margin_px",
    "p2_margin_px", "band_top_row", "excess_px" (rows to move up; <= 0 when ok), "crop_bottom_row"}. BR applies it
    at the arrived hover before pointing (BR-2, zero calls)."""
    if pointing not in POINTING_MODES:
        raise ValueError(f"pointing must be one of {POINTING_MODES}")
    out = {"ok": False, "mode": None, "row": None, "col": None, "sigma_uv_px": None, "kappa_px_per_m": None,
           "margin_px": None, "framing_margin_px": None, "p2_margin_px": None, "band_top_row": float(band_top_row),
           "excess_px": None, "crop_bottom_row": None, "reason": None}
    if part.xyz is None:
        out["reason"] = "no_point"
        return out
    pred = _predict_wrist(part.xyz, part.cov, hover_grip_xyz, pose, f_px)
    if pred is None:
        out["reason"] = "behind_camera"
        return out
    u, v, sigma_uv, depth = pred
    mode = pointing if pointing != "auto" else ("coarse" if sigma_uv > FRAMING_COARSE_SIGMA_PX else "fine")
    radius = float(_geometry().FT_PROBE_RADIUS_PX)
    kappa = float(f_px) / depth
    p2 = kappa * VIS_PROBE_LEG_M + radius
    reach = FRAMING_CROP_HALF_PX if mode == "fine" else sigma_uv
    framing = reach + radius
    margin = max(p2, framing)
    excess = v - (float(band_top_row) - margin)
    out.update(ok=bool(excess <= 1e-6), mode=mode, row=v, col=u, sigma_uv_px=sigma_uv, kappa_px_per_m=kappa,
               margin_px=margin, framing_margin_px=framing, p2_margin_px=p2, excess_px=excess,
               crop_bottom_row=v + reach)
    return out


def hover_for(part: PartRecord, hf: HeightField, robot: RobotModel, pose: HandPose, *,
              parts: Sequence[PartRecord] = (), f_px: float | None = None,
              band_top_row: float | None = None, pointing: str | None = None) -> dict:
    """P11 + P12 for a part's hover (seeded, provisional or registered): fingertips >= max(P11_FLOOR_M,
    min(P11_SIGMA_K*sigma_h + P11_OFFSET_M, P11_CAP_M)) above the reference height, raised where the open hand at the
    CURRENT jaw yaw (kept) would not clear the HeightField by the P12 margin.
    - Reference (P-E), always on the UNRAISED HeightField: a seeded or provisional part: the highest known cell
      within its seed uncertainty (P11_SEED_SIGMA_K sigma_xy around the point, or its region inflated by it); a
      registered record: its own top and the cells of its region inflated by P11_RECORD_SIGMA_K sigma_xy (an HF
      maximum beyond that, a neighbour within the seed's 3 sigma_xy, no longer raises it).
    - P-E cap: when the cap binds the hover is ``first_stage`` (BR probes and registers there, then hovers again at
      hover_for(record) when it moves > 1 cm in xy or 2 cm in z).
    - P-F: this part's own top_upper_z joins the reference when this part is lower-bound; every lower-bound part
      (this one and those of ``parts``) is raised to its top_upper_z in the hover's P12 (the open hand clears it by
      the P12 margin). A neighbour's raise never enters the reference (BPR-1: P-E says a registered record's
      reference is its own top and region; the raised footprint, inflated by O5_SIGMA_K sigma_xy, can cover this
      part's cells). A hover that a raise lifts is also ``first_stage`` (its height comes from an unmeasured bound,
      not from the part; BR's two-stage rule takes the second hover only when it moves); ``first_stage_reasons``
      lists "p11_cap" / "lower_bound". The lift is judged at the returned hover's xy (after any P-2 visibility
      shift): the final grip z against the same hover on the unraised field.
    - P-2 visibility (remedy 1), with ``f_px`` (the wrist focal length) and ``band_top_row`` (the fingertip band's
      top row): the part's predicted row stays above the band top by kappa VIS_PROBE_LEG_M + FT_PROBE_RADIUS_PX; else
      the hover moves along -R0 ("visibility_shift_m"); P12 is re-checked there.
    - BP-1 framing (grip-part lens), with ``pointing`` ("fine" / "coarse" / "auto"; None = P-2 only) for a hover
      that precedes pointing a grip part: ``pointing_frame``'s rule (the fine crop, or v + sigma_uv for coarse-first,
      FT_PROBE_RADIUS_PX above the band) replaces P-2's margin when larger; the hover moves along -R0 by at most
      M2_SEGMENT_M, then, if the rule still fails, rises in HeightField-cell steps by at most P11_CAP_M (BR's
      stretch guard bounds the rise; BR-2 re-checks at the arrived hover). ``framing`` reports the final frame with
      "shift_m", "raised_m" and "ok"; the rise never counts as a lower-bound lift (first_stage)."""
    if pointing is not None and pointing not in POINTING_MODES:
        raise ValueError(f"pointing must be one of {POINTING_MODES}")
    if part.xyz is None:
        return {"hover_grip_xyz": None, "clearance_m": None, "thresholds": [], "reason": "geometry_unmeasured",
                "first_stage": False, "framing": None}
    point = np.asarray(part.xyz, dtype=float)
    sigma_xy = horizontal_sigma_m(part)
    registered = part.status == "registered" and not part.provisional and part.top_z is not None
    reach = max((P11_RECORD_SIGMA_K if registered else P11_SEED_SIGMA_K) * sigma_xy, hf.cell_m / 2.0)
    if part.region_xy is not None and len(part.region_xy) >= 3:
        poly = poly_inflate(part.region_xy, reach)
    else:
        poly = rect_polygon(point[:2], (1.0, 0.0), reach, reach)
    lows = _lower_bound_parts([part, *(parts or ())])
    plain = _Field(hf)
    field = plain.raising(lows)                    # P-F: the open hand's P12 only (BPR-1)
    cells = hf.cells_in_polygon(poly)
    if registered and part.region_cells is not None and len(part.region_cells):
        cells = _unique_cells([cells, part.region_cells])
    # P-E reference on the UNRAISED field over the part's own cells (its region inflated by sigma_xy, or its seed
    # polygon): a neighbour's P-F raise (its region inflated by O5_SIGMA_K sigma_xy, which can cover this part's
    # cells) is a clearance obstacle for the hand (P12 below), not this part's height; the part's own top_upper_z
    # enters only when this part itself is lower-bound
    s = plain.stats(cells)
    measured = [float(point[2])] + ([float(part.top_z.value)] if part.top_z is not None else [])
    if s["z_max"] is not None:
        measured.append(s["z_max"])
    lower = part.top_is_lower_bound() and part.top_upper_z is not None
    z_ref = max(measured + ([float(part.top_upper_z)] if lower else []))
    sigma_h = max(_sigma_h(part), _sigma_or(s["sigma"], 0.0))
    clearance, capped = _p11(sigma_h)
    base_z = z_ref + clearance.value + robot.t_f_m
    open_q = max(float(k) for k in robot_facts.finger_landing_footprints())
    jaw = _jaw_angle(pose.R)
    xy = point[:2].copy()
    # P12 at the hover: the open hand clears the known cells under it by max(P12_FLOOR_M, 3 sigma of those cells);
    # the part's own sigma_h is P11's (capped, P-E), otherwise a start-view sigma_h would re-raise a capped hover
    required, th12, hand = _clear_z(field, robot, xy, [jaw], open_q, 0.0)
    grip_z = max(base_z, required)
    shift, ths = 0.0, [clearance, th12]
    framing = None
    if f_px is not None and band_top_row is not None:
        geometry = _geometry()
        r0 = np.asarray(pose.R, dtype=float)[:2, 0]
        if float(np.linalg.norm(r0)) > 1e-6:
            r0 = r0 / float(np.linalg.norm(r0))
            cap = math.inf if pointing is None else M2_SEGMENT_M
            for _ in range(2):
                here = np.array([xy[0], xy[1], grip_z])
                if pointing is None:
                    step = _visibility_shift_m(point, here, pose, f_px, band_top_row, geometry.FT_PROBE_RADIUS_PX)
                else:
                    frame = pointing_frame(part, here, pose, f_px=f_px, band_top_row=band_top_row, pointing=pointing)
                    step = (max(0.0, frame["excess_px"]) / frame["kappa_px_per_m"]
                            if frame["kappa_px_per_m"] else 0.0)
                step = min(step, cap - shift)
                if step <= 1e-9:
                    break
                xy = xy - step * r0
                shift += step
                required, th12, hand = _clear_z(field, robot, xy, [jaw], open_q, 0.0)
                grip_z = max(base_z, required)
            ths = [clearance, th12, th_value("P11_visibility_shift_m", shift, "ONLINE",
                                             "P-2: hover moved along -R0 until the part's row lies kappa "
                                             "VIS_PROBE_LEG_M + FT_PROBE_RADIUS_PX above the fingertip band",
                                             band_top_row=float(band_top_row), f_px=float(f_px),
                                             template_px=float(geometry.FT_PROBE_RADIUS_PX))]
            if pointing is not None:
                framing = {"shift_m": shift, "shift_cap_m": M2_SEGMENT_M, "raised_m": 0.0, "raise_cap_m": P11_CAP_M}
    # first_stage (P-E cap; lead sign-off 3: a P-F lift), judged at the hover's FINAL xy, after the P-2 visibility
    # shift: the same hover without any lower-bound raise (the unraised field, the part's measured reference) is
    # "plain"; a lift above it comes from an unmeasured bound
    plain_z = max(max(measured) + clearance.value + robot.t_f_m,
                  _clear_z(plain, robot, xy, [jaw], open_q, 0.0)[0]) if field is not plain or lower else grip_z
    reasons = (["p11_cap"] if capped else []) + (["lower_bound"] if grip_z > plain_z + 1e-9 else [])
    if framing is not None:
        # BP-1: the shift (capped at M2_SEGMENT_M) did not frame the part: rise in cell steps up to P11_CAP_M
        frame = pointing_frame(part, np.array([xy[0], xy[1], grip_z]), pose, f_px=f_px, band_top_row=band_top_row,
                               pointing=pointing)
        if not frame["ok"]:
            rise = 0.0
            while rise < P11_CAP_M - 1e-9:
                rise = min(rise + hf.cell_m, P11_CAP_M)
                trial = pointing_frame(part, np.array([xy[0], xy[1], grip_z + rise]), pose, f_px=f_px,
                                       band_top_row=band_top_row, pointing=pointing)
                if trial["ok"]:
                    frame = trial
                    grip_z += rise
                    framing["raised_m"] = rise
                    break
        framing.update(frame)
        ths.append(th_value("framing_margin_px", float(frame["margin_px"] or 0.0), "ONLINE",
                            "BP-1: the part's predicted row + the fine crop half side (coarse-first: + sigma_uv) lies "
                            "FT_PROBE_RADIUS_PX above the fingertip band (and P-2's kappa VIS_PROBE_LEG_M + "
                            "FT_PROBE_RADIUS_PX); hover shifted along -R0 <= M2_SEGMENT_M, then raised <= P11_CAP_M",
                            mode=frame["mode"], row=frame["row"], sigma_uv_px=frame["sigma_uv_px"],
                            band_top_row=float(band_top_row), ok=bool(frame["ok"]), shift_m=shift,
                            raised_m=framing["raised_m"]))
    return {"hover_grip_xyz": [float(xy[0]), float(xy[1]), float(grip_z)], "clearance_m": clearance.value,
            "fingertip_z": grip_z - robot.t_f_m, "z_ref": z_ref, "sigma_h_m": sigma_h, "seed_reach_m": reach,
            "raised_m": max(0.0, grip_z - base_z), "hand": hand, "thresholds": ths, "reason": None,
            "first_stage": bool(reasons), "first_stage_reasons": reasons, "cap_m": P11_CAP_M,
            "registered": registered, "lower_bound": lower,
            "lower_bound_parts": list(field.raised_pids), "visibility_shift_m": shift, "framing": framing}


# ------------------------------------------------------------------------------------------------ P8 / P9, G13
def _base_event(event):
    """A primitive reason without the GuardedLower fast-phase HANDOVER_PREFIX (a handover carries its support cue)."""
    event = str(event)
    return event[len(HANDOVER_PREFIX):] if event.startswith(HANDOVER_PREFIX) else event


def classify_contact(plan: ContactPlan, contact_grip_z: float | None, event: str) -> dict:
    """P8: a contact event (CONTACT_EVENTS) at grip z inside an expected band -> "expected" (matched = its index in
    plan.expected, highest first); above every band -> "blocked_above_part" (never close); otherwise it descended
    past an expected contact -> "no_contact_where_expected". The upper side of every band is capped at the P3 close
    limit (ExpectedContact.upper_z, BP-4): a stop above it would close without the P3 pad overlap.
    No contact (any other event, e.g. a reached target or bound, a step budget):
    - height_stop: reached g* (final grip z None or at most the capped band above it) -> "expected";
    - touch_stop: reached the P9 bound (final grip z None or within the lowest band above the bound) ->
      "no_contact_within_bound"; below every band but above the bound -> "no_contact_where_expected";
    - otherwise the descent stopped short of the plan's end without a contact -> "stopped_early" (additive class:
      no contact classification applies; report the primitive's own reason).
    P-B (fixture landing): on a fixture-prior plan (``plan.move_prior`` in FIXTURE_PRIORS) a touch_stop fingertip
    stop (FIXTURE_LANDING_EVENTS) below every band but above the P9 bound is "expected" with ``landing_measured``
    True: BR stamps the touch (``surface_z`` = contact grip z - t_f) and recomputes P10 from it. Free parts keep their
    classes. Every contact result carries ``landing_measured`` and ``surface_z``."""
    base = _base_event(event)
    contact = base in CONTACT_EVENTS
    bands = [t for t in plan.thresholds if t.name in ("P8_band_m", "P9_bound_grip_z_m", "P3_close_limit_grip_z_m")]
    rows = [e.to_dict() for e in plan.expected]
    out = {"event": event, "contact_grip_z": None if contact_grip_z is None else float(contact_grip_z),
           "expected": rows, "bound_grip_z": plan.bound_grip_z, "thresholds": bands, "landing_measured": False}
    if not contact:
        z = None if contact_grip_z is None else _finite(contact_grip_z, "contact_grip_z")
        if plan.depth_rule == "height_stop":
            reached = z is None or z <= plan.expected[0].upper_z + 1e-9
            out.update({"class": "expected", "matched": 0} if reached else
                       {"class": "stopped_early", "matched": None})
            return out
        lowest = min(plan.expected, key=lambda e: e.grip_z)
        if z is None or z <= plan.bound_grip_z + lowest.band_m + 1e-9:
            out.update({"class": "no_contact_within_bound", "matched": None})
        elif z < lowest.grip_z - lowest.band_m - 1e-9:
            out.update({"class": "no_contact_where_expected", "matched": None})
        else:
            out.update({"class": "stopped_early", "matched": None})
        return out
    if contact_grip_z is None:
        raise ValueError("a contact event needs its grip z")
    z = _finite(contact_grip_z, "contact_grip_z")
    out["surface_z"] = z - robot_facts.FINGERTIP_REACH_MM / 1000.0
    for i, e in enumerate(plan.expected):
        if e.grip_z - e.band_m - 1e-9 <= z <= e.upper_z + 1e-9:
            out.update({"class": "expected", "matched": i})
            return out
    if z > max(e.upper_z for e in plan.expected):
        out.update({"class": "blocked_above_part", "matched": None})
        return out
    lowest = min(plan.expected, key=lambda e: e.grip_z)
    if (plan.depth_rule == "touch_stop" and plan.move_prior in FIXTURE_PRIORS and base in FIXTURE_LANDING_EVENTS
            and plan.bound_grip_z + 1e-9 < z < lowest.grip_z - lowest.band_m - 1e-9):
        out.update({"class": "expected", "matched": None, "landing_measured": True})
        return out
    out.update({"class": "no_contact_where_expected", "matched": None})
    return out


def width_consistency(plan: ContactPlan, gap_mm: float) -> dict:
    """G13: the pad separation (gap - 1.0 mm, R9) lies within [w_min - band, w_max + band] of the planned
    cross-section, band = max(G13_FLOOR_MM, G13_SIGMA_K*sigma_w). Outside = width_mismatch (evidence; a failure
    only together with no co-motion).
    G13+ (P-D, RC-6; free parts, checked right after the close and before the load test): ``misaligned`` when (gap -
    1.0) - w_narrow > band + PAD_LENGTH_R0_MM sin(YAW_GRID_RAD / 2), w_narrow = the part's narrowest measured width
    (``plan.width["w_narrow_mm"]``); BR opens, excludes this (family, yaw) and its twin, and takes the next
    candidate. A fixture plan is never misaligned (its gap is G13's business)."""
    w = float(plan.width["w_mm"])
    sigma_w = float(plan.width.get("sigma_w_mm", 0.0))
    w_min, w_max = float(plan.width.get("w_min_mm", w)), float(plan.width.get("w_max_mm", w))
    band = th_max("G13_width_band_mm", Term("DD", G13_FLOOR_MM, "G13_FLOOR_MM"),
                  Term("ONLINE", G13_SIGMA_K * sigma_w, "G13_SIGMA_K*sigma_w"), unit="mm")
    sep = float(robot_facts.pad_separation_mm(_finite(gap_mm, "gap_mm")))
    low, high = w_min - band.value, w_max + band.value
    narrow = float(plan.width.get("w_narrow_mm", w_min))
    yaw_term = float(robot_facts.PAD_LENGTH_R0_MM) * math.sin(YAW_GRID_RAD / 2.0)
    # decision §3 labels this band ROBOT + DD (the R4 pad length with the DD floor and yaw grid); a Threshold carries
    # one class, so it is DD with both named in the derivation (the scene's PB_REACH_M convention), and the band
    # term's own class (DD floor or ONLINE sigma_w) is kept in band_label
    misaligned_band = th_value("G13_misaligned_band_mm", band.value + yaw_term, "DD",
                               "G13+ (decision §3, ROBOT + DD): max(G13_FLOOR_MM, G13_SIGMA_K*sigma_w) [DD / "
                               "ONLINE] + PAD_LENGTH_R0_MM*sin(YAW_GRID_RAD/2) [ROBOT R4 pad length x DD half the "
                               "yaw grid]",
                               band_mm=band.value, band_label=band.label, yaw_term_mm=yaw_term, w_narrow_mm=narrow)
    fixture = plan.move_prior in FIXTURE_PRIORS
    excess = sep - narrow
    misaligned = bool(not fixture and excess > misaligned_band.value + 1e-9)
    return {"consistent": bool(low - 1e-9 <= sep <= high + 1e-9), "band_mm": band.value, "sep_mm": sep,
            "w_mm": w, "range_mm": [low, high], "misaligned": misaligned, "excess_mm": excess,
            "misaligned_band_mm": misaligned_band.value, "w_narrow_mm": narrow, "fixture": fixture,
            "thresholds": [band, misaligned_band]}


# ------------------------------------------------------------------------------------------------ V5
def _check_result(ok, reason, values, thresholds):
    return {"ok": bool(ok), "reason": None if ok else reason, "values": jsonable(values),
            "thresholds": list(thresholds)}


def check_point_answer(kind: str, uv, part: PartRecord, hf: HeightField, inputs: PlannerInputs, *,
                       jaw_yaw: float | None = None, wrist_gray_upright: np.ndarray | None = None,
                       held: HeldFrame | None = None, structure_axis=None) -> dict:
    """V5 (spec §3.6 item 4), on the RE-MEASURED heights (after the hover ParallaxProbe and registration):
    - "grasp_point": the pointed point stands >= V5_GRASP_SIGMA_K*sigma_h above BOTH finger-landing columns (P16)
      along the planned jaw (``jaw_yaw``; None = across the measured long axis, else the current jaw); never
      compared with every neighbour, so a level neighbour beside the point does not reject it;
    - "surface_point": planar within D8 (max(V5_PLANE_FLOOR_M, V5_PLANE_SIGMA_K*hypot(sigma_h, sigma_cell))) over
      ``held``'s footprint
      (r_obj around the point; without ``held`` over the part's region polygon);
    - "elongated": the structure axis of ``wrist_gray_upright`` at uv agrees with the measured long axis
      (agree >= V5_AXIS_AGREE, coherence >= V5_COHERENCE, gradient rms >= V5_RMS; v1 bar check rules).
      ``structure_axis`` injects geometry.structure_axis (None = the geometry module, imported here). Decision §2.2
      S-10: a record whose D10 axis the scene accepted with its S-4' ``axis_check`` is read from the scene, not
      recomputed (values "source" = "scene_axis_check").
    The grasp_point columns land as P16 does (RC-1a: verified maximum, else the part's support).
    Returns {"ok", "reason": None | "pointed_not_<kind>", "values", "thresholds"}."""
    if kind not in POINT_CHECK_KINDS:
        raise ValueError(f"kind must be one of {POINT_CHECK_KINDS}")
    values = {"kind": kind, "uv": None if uv is None else [float(v) for v in uv]}
    if part.xyz is None:
        values["checked"] = False
        reason = "pointed_not_grasp_point" if kind == "grasp_point" else (
            "pointed_not_surface" if kind == "surface_point" else "pointed_not_elongated")
        return _check_result(False, reason, values, [])
    point = np.asarray(part.xyz, dtype=float)
    sigma_h = _sigma_h(part)
    if kind == "grasp_point":
        if jaw_yaw is not None:
            yaw = float(jaw_yaw)
        elif part.is_elongated():
            axis = np.asarray(part.elongation[0], dtype=float)
            yaw = math.atan2(axis[1], axis[0]) + math.pi / 2.0
        else:
            yaw = _jaw_angle(inputs.pose.R)
        width = part.width_at(yaw)
        sigma_w_mm = 0.0 if width is None else width.sigma * 1000.0
        e_lat = P1_ALIGN_K * float(inputs.sigma_align_m) * 1000.0 + sigma_w_mm + P1_PAD_MARGIN_MM
        cols = _columns(_Field(hf), point[:2], yaw, inputs.robot, inflate_m=e_lat / 1000.0,
                        support_z=None if part.support_z is None else float(part.support_z.value),
                        finger_q_mm=inputs.finger_q_mm, sigma_floor=0.0)
        # contract §C7 / spec §3.6 item 4: >= V5_GRASP_SIGMA_K*sigma_h of the part (BP-3); the landing sigmas are
        # evidence only
        margin = th_value("V5_grasp_margin_m", V5_GRASP_SIGMA_K * sigma_h, "ONLINE", "V5_GRASP_SIGMA_K*sigma_h",
                          sigma_h_m=sigma_h)
        heights = [None if c.landing_z is None else float(point[2]) - c.landing_z for c in cols]
        ok = all(h is not None and h >= margin.value - 1e-9 for h in heights)
        values.update(point_z=float(point[2]), jaw_yaw_deg=math.degrees(yaw),
                      landing_z=[c.landing_z for c in cols], landing_sigma=[c.landing_sigma for c in cols],
                      above_m=heights, margin_m=margin.value)
        return _check_result(ok, "pointed_not_grasp_point", values, [margin])
    if kind == "surface_point":
        if held is not None:
            radius = float(held.r_obj_m)
            poly = rect_polygon(point[:2], (1.0, 0.0), radius, radius)
        elif part.region_xy is not None and len(part.region_xy) >= 3:
            poly, radius = np.asarray(part.region_xy, dtype=float), None
        else:
            values["checked"] = False
            return _check_result(True, None, values, [])
        s = _Field(hf).stats(hf.cells_in_polygon(poly))
        band = th_max("V5_plane_band_m", Term("DD", V5_PLANE_FLOOR_M, "V5_PLANE_FLOOR_M"),
                      Term("ONLINE", V5_PLANE_SIGMA_K * math.hypot(sigma_h, _sigma_or(s["sigma"], 0.0)),
                           "V5_PLANE_SIGMA_K*hypot(sigma_h, sigma_cell) (the scene's D8 combination)"), unit="m")
        if s["z_max"] is None:
            values.update(checked=False, known_fraction=s["known_fraction"])
            return _check_result(False, "pointed_not_surface", values, [band])
        spread = max(s["z_max"] - float(point[2]), float(point[2]) - s["z_min"])
        values.update(point_z=float(point[2]), z_range=[s["z_min"], s["z_max"]], spread_m=spread,
                      footprint_radius_m=radius)
        return _check_result(spread <= band.value + 1e-9, "pointed_not_surface", values, [band])
    # elongated
    agree_th = th_value("V5_axis_agree", V5_AXIS_AGREE, "DD", "V5_AXIS_AGREE (v1 bar check)")
    coherence_th = th_value("V5_coherence", V5_COHERENCE, "DD", "V5_COHERENCE (v1 bar check)")
    rms_th = th_value("V5_gradient_rms", V5_RMS, "DD", "V5_RMS grey per px (v1 bar check)")
    ths = [agree_th, coherence_th, rms_th]
    if not part.is_elongated():
        values["checked"] = False
        return _check_result(True, None, values, ths)
    check = part.axis_check
    if isinstance(check, Mapping) and check.get("accepted"):
        # S-10: V5 reads the scene's D10 + S-4' axis consensus instead of recomputing the image structure axis
        values.update(checked=True, source="scene_axis_check", agree_rad=check.get("agree_rad"),
                      tensor=check.get("source"))
        return _check_result(True, None, values, ths)
    if wrist_gray_upright is None or uv is None:
        values["checked"] = False
        return _check_result(True, None, values, ths)
    if structure_axis is None:
        structure_axis = _geometry().structure_axis
    image_axis, coherence, rms = structure_axis(wrist_gray_upright, uv)
    image_axis = np.asarray(image_axis, dtype=float).reshape(-1)[:2]
    axis = np.asarray(part.elongation[0], dtype=float)
    world = np.array([axis[0], axis[1], 0.0])
    R = inputs.pose.R
    expected_axis = np.array([float(world @ R[:, 1]), -float(world @ R[:, 0])])
    norms = float(np.linalg.norm(expected_axis)) * float(np.linalg.norm(image_axis))
    agree = 0.0 if norms < 1e-12 else abs(float(expected_axis @ image_axis)) / norms
    ok = agree >= V5_AXIS_AGREE and float(coherence) >= V5_COHERENCE and float(rms) >= V5_RMS
    values.update(checked=True, agree=agree, coherence=float(coherence), gradient_rms=float(rms))
    return _check_result(ok, "pointed_not_elongated", values, ths)


def check_point_any_jaw(part: PartRecord, hf: HeightField, inputs: PlannerInputs, uv=None) -> dict:
    """V5-U (the grasp-point check when no candidate jaw exists: the record's width or axis is unmeasured, so
    plan_contact offers no yaw): the V5 grasp_point rule on every jaw line of the yaw grid (round(pi / YAW_GRID_RAD)
    lines; a line and its 180 deg twin land the same two columns). "ok" (with ``deferred``) when a run of at least
    V5U_CONE_LINES adjacent lines clears both columns: the point is a grasp point over the jaw cone the re-measured
    record can plan, and the decisive check runs at the planned yaw (grasp); "pointed_not_grasp_point" otherwise: the
    point clears no jaw cone, a pixel-dependent failure worth a re-point. Never the incidental current jaw."""
    n = int(round(math.pi / YAW_GRID_RAD))
    lines, clear, best = [], 0, None
    margin = None
    thresholds = []
    for k in range(n):
        yaw = k * YAW_GRID_RAD
        r = check_point_answer("grasp_point", uv, part, hf, inputs, jaw_yaw=yaw)
        v = r.get("values") or {}
        above = v.get("above_m") or []
        low = min([a for a in above if a is not None], default=None)
        lines.append([math.degrees(yaw), bool(r["ok"]), low])
        margin = v.get("margin_m", margin)
        if r["ok"]:
            clear += 1
            if best is None or (low is not None and low > best[1]):
                best = (math.degrees(yaw), low)
        thresholds = r.get("thresholds") or thresholds
    # the longest run of adjacent clearing lines (circular: line n-1 neighbours line 0, its 180 deg twin)
    cone = n if clear == n else 0
    if clear < n:
        run = 0
        for ok in [row[1] for row in lines] * 2:
            run = run + 1 if ok else 0
            cone = max(cone, min(run, n))
    need = th_value("V5U_cone_lines", V5U_CONE_LINES, "DD", "V5U_CONE_LINES (best line +- the P-C axis agreement in "
                    "YAW_GRID_RAD steps)")
    values = {"kind": "grasp_point", "uv": None if uv is None else [float(x) for x in uv], "jaw": "any_line",
              "lines": lines, "lines_clear": clear, "lines_total": n, "cone_lines": cone,
              "cone_needed": int(V5U_CONE_LINES), "best_line_deg": None if best is None else best[0],
              "margin_m": margin, "point_z": None if part.xyz is None else float(part.xyz[2])}
    ok = cone >= V5U_CONE_LINES
    out = _check_result(ok, "pointed_not_grasp_point", values, [*thresholds, need])
    out["deferred"] = ok
    return out


# ------------------------------------------------------------------------------------------------ O5 keep-outs
def keepout_for(part: PartRecord, kind: str, *, sigma_loc_m: float, sigma_h_m: float,
                theta_part_rad: float | None = None) -> KeepOut:
    """O5: the part region inflated horizontally by max(O5_FLOOR_M, O5_SIGMA_K*sigma_loc_m) (sigma_loc_m = the
    part's horizontal position sigma, horizontal_sigma_m, [C-23]); vertically from its support to top +
    max(O5_TOP_FLOOR_M, O5_SIGMA_K*sigma_h_m). kind "articulated" (after a turn): the D11 circle when it passed,
    else the region polygon rotated by ``theta_part_rad`` (WORLD angle about +z; a top-down twist of contract sense
    s and size t is -s t) about the centre prior (circle centre, else the part point)."""
    if kind not in KEEPOUT_REQUEST_KINDS:
        raise ValueError(f"kind must be one of {KEEPOUT_REQUEST_KINDS}")
    sigma_loc, sigma_h = _finite(sigma_loc_m, "sigma_loc_m"), _finite(sigma_h_m, "sigma_h_m")
    inflate = th_max("O5_inflate_m", Term("DD", O5_FLOOR_M, "O5_FLOOR_M"),
                     Term("ONLINE", O5_SIGMA_K * sigma_loc, "O5_SIGMA_K*sigma_loc"), unit="m")
    top_margin = th_max("O5_top_m", Term("DD", O5_TOP_FLOOR_M, "O5_TOP_FLOOR_M"),
                        Term("ONLINE", O5_SIGMA_K * sigma_h, "O5_SIGMA_K*sigma_h"), unit="m")
    point = None if part.xyz is None else np.asarray(part.xyz, dtype=float)
    if part.top_z is not None:
        top = float(part.top_z.value)
    elif point is not None:
        top = float(point[2])
    else:
        raise ValueError("keepout_for needs a measured top or point")
    if kind == "articulated" and part.circle is not None:
        base = _circle_polygon(part.circle[0], float(part.circle[1]))
    else:
        if part.region_xy is not None and len(part.region_xy) >= 3:
            base = np.asarray(part.region_xy, dtype=float)
        elif point is not None:
            base = point[:2].reshape(1, 2)
        else:
            raise ValueError("keepout_for needs a region polygon or a point")
        if kind == "articulated" and theta_part_rad is not None:
            centre = point[:2] if point is not None else _polygon_centroid(base)
            base = rotate_xy(base, centre, _finite(theta_part_rad, "theta_part_rad"))
    polygon = poly_inflate(base, inflate.value)
    z_low = min(float(part.support_z.value), top) if part.support_z is not None else top
    return KeepOut(part.pid, kind, polygon, z_low, top + top_margin.value, (inflate, top_margin))


# ------------------------------------------------------------------------------------------------ transit (M1, M2)
def _split(points, max_len):
    """Waypoints with every consecutive distance <= max_len (straight pieces split evenly)."""
    out = [np.asarray(points[0], dtype=float)]
    for nxt in points[1:]:
        nxt = np.asarray(nxt, dtype=float)
        length = float(np.linalg.norm(nxt - out[-1]))
        if length < 1e-9:
            continue
        count = max(1, int(math.ceil(length / max_len - 1e-9)))
        start = out[-1]
        out += [start + (nxt - start) * (i / count) for i in range(1, count + 1)]
    return out


def _samples(a_xy, b_xy, step):
    a, b = np.asarray(a_xy, dtype=float), np.asarray(b_xy, dtype=float)
    count = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
    return [a + (b - a) * (i / count) for i in range(count + 1)]


def _path_result(waypoints, thresholds, **extra):
    segments = [float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:])]
    return dict(extra, waypoints=[[float(v) for v in p] for p in waypoints], segments=segments,
                thresholds=list(thresholds))


def free_path(start_grip, goal_grip, hf: HeightField, keepouts, robot: RobotModel, pose: HandPose, *,
              parts: Sequence[PartRecord] = ()) -> dict:
    """Empty-hand transit (M1 rule with the hand's own lowest points; M2 segments): up (or down) at the start to
    transit_z, over at transit_z, down to the goal. transit_z = max(goal z, the lowest grip z at which the finger
    columns (fingertips, R5) and the hand footprint (palm, R6) at the current jaw and opening clear every known cell
    and keep-out along the path by max(M1_FLOOR_M, M1_SIGMA_K*sigma)). P-F (additive ``parts``): a lower-bound part's
    cells are raised to its top_upper_z. {"waypoints", "transit_z", "segments", "required_z", "swept_max_z",
    "goal_clear", "lower_bound_parts", "thresholds"}."""
    start, goal = _vec(start_grip, 3, "start_grip"), _vec(goal_grip, 3, "goal_grip")
    field = _Field(hf, keepouts).raising(parts)
    yaw = _jaw_angle(pose.R)
    finger_q = max(float(pose.gap_mm) / 2.0, 0.0)
    finger_cells, foot_cells = [], []
    for xy in _samples(start[:2], goal[:2], hf.cell_m):
        f1, f2 = robot.finger_columns_xy(xy, _unit_xy(yaw), finger_q_mm=finger_q)
        finger_cells += [hf.cells_in_polygon(f1), hf.cells_in_polygon(f2)]
        foot_cells.append(hf.cells_in_polygon(robot.hand_footprint_xy(xy, _unit_xy(yaw))))
    fingers, foot = field.stats(_unique_cells(finger_cells)), field.stats(_unique_cells(foot_cells))
    needs = []
    if fingers["z_max"] is not None:
        needs.append((fingers["z_max"] + robot.t_f_m, fingers["sigma"]))
    if foot["z_max"] is not None:
        needs.append((foot["z_max"] - robot.p_min_m, foot["sigma"]))
    need, sigma = max(needs, key=lambda item: item[0]) if needs else (-math.inf, 0.0)
    margin = th_max("M1_transit_clearance_m", Term("DD", M1_FLOOR_M, "M1_FLOOR_M"),
                    Term("ONLINE", M1_SIGMA_K * _sigma_or(sigma, 0.0), "M1_SIGMA_K*sigma_h"), unit="m")
    required = need + margin.value
    transit_z = max(required, float(goal[2]))
    th2 = th_value("M2_segment_m", M2_SEGMENT_M, "DD", "M2_SEGMENT_M")
    legs = [start, np.array([start[0], start[1], transit_z]), np.array([goal[0], goal[1], transit_z]), goal]
    swept = max([z for z in (fingers["z_max"], foot["z_max"]) if z is not None], default=None)
    return _path_result(_split(legs, M2_SEGMENT_M), [margin, th2], transit_z=float(transit_z),
                        required_z=None if not math.isfinite(required) else float(required), swept_max_z=swept,
                        goal_clear=bool(float(goal[2]) >= required - 1e-9),
                        lower_bound_parts=list(field.raised_pids))


def transit_path(held: HeldFrame, start_grip, target_grip_xy, hf: HeightField, keepouts, robot: RobotModel,
                 pose: HandPose, *, parts: Sequence[PartRecord] = ()) -> dict:
    """§4.9 Carry: out from under overhangs first (a known cell above the grip inside the held footprint, or above
    the palm inside the hand footprint: moved horizontally away from them at the current height); then lift so that
    the lowest held point (grip - b_low) clears the swept maximum of the hand footprint and the held footprint
    (r_obj + sigma_offset around grip + R offset_h) plus max(M1_FLOOR_M, M1_SIGMA_K*sigma_h); then horizontal;
    segments <= M2_SEGMENT_M. The swept maximum EXCLUDES held.vacated_cells [C-36] and the held object's own
    keep-outs, so the object is never its own obstacle. P-F (additive ``parts``): every other lower-bound part's cells
    are raised to its top_upper_z. {"ok", "reason", "waypoints", "carry_z", "swept_max_z", "segments",
    "overhang_exit", "lower_bound_parts", "thresholds"}."""
    start, target = _vec(start_grip, 3, "start_grip"), _vec(target_grip_xy, 2, "target_grip_xy")
    field = _Field(hf, keepouts, exclude_pids={held.pid, held.grip_pid}, exclude_cells=held.vacated_cells).raising(
        _lower_bound_parts(parts, skip=(held.pid, held.grip_pid)))
    offset_xy = (pose.R @ _vec(held.offset_h, 3, "offset_h"))[:2]
    radius = float(held.r_obj_m) + float(held.sigma_offset_m)
    yaw = _jaw_angle(pose.R)
    b_low = float(held.b_low.value)

    def cells(grip_xy):
        obj = hf.cells_in_polygon(_circle_polygon(grip_xy + offset_xy, radius))
        hand = hf.cells_in_polygon(robot.hand_footprint_xy(grip_xy, _unit_xy(yaw)))
        return obj, hand

    def overhang(grip_xy):
        obj, hand = cells(grip_xy)
        found = []
        for group, limit in ((obj, float(start[2])), (hand, float(start[2]) + robot.p_min_m)):
            for iy, ix in np.asarray(group, dtype=int).reshape(-1, 2):
                if field.skip[iy, ix]:
                    continue
                z = max(float(hf.z[iy, ix]) if hf.known[iy, ix] else -math.inf, float(field.overlay[iy, ix]))
                if z > limit:
                    found.append((iy, ix))
        return found
    exit_info, exit_xy = None, start[:2].copy()
    above = overhang(exit_xy)
    if above:
        centroid = _cell_centres(hf, above).mean(axis=0)
        away = (start[:2] + offset_xy) - centroid
        if float(np.linalg.norm(away)) < 1e-9:
            away = target - start[:2]
        if float(np.linalg.norm(away)) < 1e-9:
            away = np.array([1.0, 0.0])
        away = away / float(np.linalg.norm(away))
        found = None
        for k in range(1, max(hf.z.shape) + 1):
            candidate = start[:2] + away * (k * hf.cell_m)
            if not overhang(candidate):
                found = candidate
                break
        if found is None:
            return {"ok": False, "reason": "lateral_unreachable", "waypoints": [[float(v) for v in start]],
                    "carry_z": None, "swept_max_z": None, "segments": [], "overhang_exit": None, "thresholds": []}
        exit_xy = found
        exit_info = {"cells": len(above), "direction": away, "distance_m": float(np.linalg.norm(found - start[:2]))}
    swept_cells = []
    for xy in _samples(exit_xy, target, hf.cell_m):
        obj, hand = cells(xy)
        swept_cells += [obj, hand]
    s = field.stats(_unique_cells(swept_cells))
    sigma = _sigma_or(s["sigma"], 0.0)
    margin = th_max("M1_transit_clearance_m", Term("DD", M1_FLOOR_M, "M1_FLOOR_M"),
                    Term("ONLINE", M1_SIGMA_K * sigma, "M1_SIGMA_K*sigma_h"), unit="m")
    required = -math.inf if s["z_max"] is None else s["z_max"] + margin.value + b_low
    carry_z = max(float(start[2]), required)
    th2 = th_value("M2_segment_m", M2_SEGMENT_M, "DD", "M2_SEGMENT_M")
    th_carry = th_value("M1_carry_grip_z_m", carry_z, margin.label,
                        "max(start z, swept max + M1 margin + b_low)", start_z=float(start[2]),
                        swept_max_z="none" if s["z_max"] is None else s["z_max"], b_low_m=b_low)
    points = [start]
    if exit_info is not None:
        points.append(np.array([exit_xy[0], exit_xy[1], start[2]]))
    points += [np.array([exit_xy[0], exit_xy[1], carry_z]), np.array([target[0], target[1], carry_z])]
    return _path_result(_split(points, M2_SEGMENT_M), [margin, th2, th_carry], ok=True, reason=None,
                        carry_z=float(carry_z), swept_max_z=s["z_max"], overhang_exit=jsonable(exit_info),
                        lowest_held_z=float(carry_z - b_low), lower_bound_parts=list(field.raised_pids))


# ------------------------------------------------------------------------------------------------ aim (M4, M16)
def aim_target(ref: PartRecord, held: HeldFrame, hf: HeightField, keepouts, pose: HandPose) -> dict:
    """M4 + M16 on the destination region (ref's region cells): a cell is occupied when its height (or a keep-out
    over it) exceeds the destination surface by max(M8_FLOOR_M, M8_SIGMA_K*sigma_dest); IF-R3' (gap round): a D7b
    fill is read at its parent + PV_PARENT_SIGMA_K sigma_parent for that test (verified cells: measured, D7a plane
    fills and keep-outs, unchanged). Unoccupied: the target is the region centre; occupied (centre-first): the free
    cell nearest the region centre whose free radius meets M16's need (r_obj + max(M16_FLOOR_M, sigma_offset)), else
    the free cell of largest free radius (ties to the nearest the centre). m_free = the free radius at the target -
    r_obj; tol = max(M4_FLOOR_M, M4_FRACTION*m_free) on both (free) axes; target grip = target - R offset_h.
    ``fits`` false never refuses: it is reported with the flag ``tight_fit`` (lead ruling L8); the only refusal is
    "no free cell" (support_above_destination). {"ok", "reason", "target_obj_xy", "target_grip_xy", "tol_m",
    "m_free_m", "r_free_m", "free_spot", "fits", "target_rule", "flags", "occupied_by", "thresholds"}."""
    region = _region_cells(ref, hf)
    if not len(region) or ref.top_z is None:
        return {"ok": False, "reason": "destination_not_registered", "target_obj_xy": None, "target_grip_xy": None,
                "tol_m": None, "m_free_m": None, "r_free_m": None, "free_spot": False, "fits": False,
                "thresholds": []}
    s_dest, sigma_dest = float(ref.top_z.value), float(ref.top_z.sigma)
    field = _Field(hf, keepouts, exclude_pids={held.pid, held.grip_pid, ref.pid}, exclude_cells=held.vacated_cells)
    band = th_max("M16_occupied_band_m", Term("DD", M8_FLOOR_M, "M8_FLOOR_M"),
                  Term("ONLINE", M8_SIGMA_K * sigma_dest, "M8_SIGMA_K*sigma_dest"), unit="m")
    r_obj, sigma_offset = float(held.r_obj_m), float(held.sigma_offset_m)
    need = th_max("M16_free_radius_m", Term("DD", r_obj + M16_FLOOR_M, "r_obj + M16_FLOOR_M"),
                  Term("ONLINE", r_obj + sigma_offset, "r_obj + sigma_offset"), unit="m")
    fill_read = th_value("M16_fill_reading_k", PV_PARENT_SIGMA_K, "DD",
                         "IF-R3': a D7b fill occupies when parent + PV_PARENT_SIGMA_K*sigma_parent > s_dest + band")

    def occupied(cells):
        cells = np.asarray(cells, dtype=int).reshape(-1, 2)
        iy, ix = cells[:, 0], cells[:, 1]
        z = np.where(hf.known[iy, ix] & ~field.skip[iy, ix], hf.z[iy, ix], -np.inf)
        fills = field.unverified[iy, ix]
        z = np.where(fills, z - (D7B_FILL_SIGMA_K - PV_PARENT_SIGMA_K) * np.nan_to_num(hf.sigma[iy, ix], nan=0.0), z)
        over = np.maximum(z, field.overlay[iy, ix]) > s_dest + band.value
        return over, over & fills & ~(field.overlay[iy, ix] > s_dest + band.value)
    ny, nx = hf.z.shape
    pad = int(math.ceil((need.value + hf.cell_m) / hf.cell_m)) + 1
    low = np.maximum(region.min(axis=0) - pad, 0)
    high = np.minimum(region.max(axis=0) + pad, [ny - 1, nx - 1])
    gy, gx = np.mgrid[low[0]:high[0] + 1, low[1]:high[1] + 1]
    window = np.column_stack([gy.ravel(), gx.ravel()])
    region_occupied, by_fill = occupied(region)
    occupied_by = {"verified": int((region_occupied & ~by_fill).sum()), "fill": int(by_fill.sum())}
    free_mask = np.zeros(hf.z.shape, dtype=bool)
    free_mask[region[~region_occupied, 0], region[~region_occupied, 1]] = True
    blocked = window[~free_mask[window[:, 0], window[:, 1]]]
    free = region[~region_occupied]
    if not len(free):
        return {"ok": False, "reason": "support_above_destination", "target_obj_xy": None, "target_grip_xy": None,
                "tol_m": None, "m_free_m": None, "r_free_m": 0.0, "free_spot": True, "fits": False,
                "occupied_by": occupied_by, "thresholds": [band, need, fill_read]}
    free_xy, blocked_xy = _cell_centres(hf, free), _cell_centres(hf, blocked)
    clearance = np.full(len(free_xy), math.inf)
    if len(blocked_xy):
        for start in range(0, len(free_xy), len(free_xy) // 8 + 1):        # chunks keep the distance matrix small
            chunk = free_xy[start:start + len(free_xy) // 8 + 1]
            nearest = np.sqrt(((chunk[:, None, :] - blocked_xy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            clearance[start:start + len(chunk)] = np.maximum(nearest - hf.cell_m / 2.0, 0.0)
    centre = _cell_centres(hf, region).mean(axis=0)
    any_occupied = bool(region_occupied.any())
    to_centre = np.sqrt(((free_xy - centre) ** 2).sum(axis=1))
    if not any_occupied:
        target = centre
        r_free = float(clearance[int(np.argmin(to_centre))])
        rule = "centre"
    else:
        fit = clearance >= need.value - 1e-9
        if fit.any():                                   # IF-R3' centre-first: the fitting cell nearest the centre
            k = int(np.flatnonzero(fit)[np.argmin(to_centre[fit])])
            rule = "centre_first"
        else:                                           # no fitting cell: the largest free radius, nearest the centre
            top = clearance >= float(np.max(clearance)) - 1e-9
            k = int(np.flatnonzero(top)[np.argmin(to_centre[top])])
            rule = "largest_clearance"
        target, r_free = free_xy[k], float(clearance[k])
    m_free = r_free - r_obj
    tol = th_max("M4_aim_tol_m", Term("DD", M4_FLOOR_M, "M4_FLOOR_M"),
                 Term("ONLINE", M4_FRACTION * max(m_free, 0.0), "M4_FRACTION*m_free"), unit="m")
    offset_xy = (pose.R @ _vec(held.offset_h, 3, "offset_h"))[:2]
    fits = bool(r_free >= need.value - 1e-9)
    return {"ok": True, "reason": None, "target_obj_xy": [float(v) for v in target],
            "target_grip_xy": [float(v) for v in np.asarray(target) - offset_xy], "tol_m": (tol.value, tol.value),
            "m_free_m": float(m_free), "r_free_m": r_free, "free_spot": any_occupied,
            "fits": fits, "occupied_fraction": float(region_occupied.mean()), "target_rule": rule,
            "flags": [] if fits else ["tight_fit"], "occupied_by": occupied_by,
            "target_to_centre_m": float(np.linalg.norm(np.asarray(target, dtype=float) - centre)),
            "thresholds": [band, need, fill_read, tol]}


# ------------------------------------------------------------------------------------------------ lowering (M5-M8)
def lowering_plan(ref: PartRecord, held: HeldFrame, aim: dict, *, sigma_touch_m: float) -> dict:
    """GuardedLower bounds: S_dest = the destination top; expected support grip = S_dest + b_low; fast until
    expected + M6_ABOVE_M + M6_SIGMA_K*sigma (M6); bound = expected - max(M7_FLOOR_M, M7_TOL_K*tol_h) (M7); tol_h =
    max(M8_FLOOR_M, M8_SIGMA_K*sqrt(sigma_dest^2 + sigma_b^2 + sigma_touch^2)) (M8); drift abort = max(M5_FLOOR_M,
    M5_FRACTION*m_free) (M5). sigma = sqrt(sigma_dest^2 + sigma_b^2)."""
    if ref.top_z is None:
        raise ValueError("lowering_plan needs the destination's measured top")
    s_dest, sigma_dest = float(ref.top_z.value), float(ref.top_z.sigma)
    b_low, sigma_b = float(held.b_low.value), float(held.b_low.sigma)
    sigma_touch = _finite(sigma_touch_m, "sigma_touch_m")
    tol = th_max("M8_tol_h_m", Term("DD", M8_FLOOR_M, "M8_FLOOR_M"),
                 Term("ONLINE", M8_SIGMA_K * math.sqrt(sigma_dest ** 2 + sigma_b ** 2 + sigma_touch ** 2),
                      "M8_SIGMA_K*sqrt(sigma_dest^2 + sigma_b^2 + sigma_touch^2)"), unit="m")
    expected = s_dest + b_low
    sigma = math.hypot(sigma_dest, sigma_b)
    fast = th_value("M6_fast_until_grip_z_m", expected + M6_ABOVE_M + M6_SIGMA_K * sigma, "ONLINE",
                    "expected support grip + M6_ABOVE_M + M6_SIGMA_K*sigma", expected_grip_z=expected, sigma_m=sigma)
    drop, label, text = _winner(Term("DD", M7_FLOOR_M, "M7_FLOOR_M"), Term(tol.label, M7_TOL_K * tol.value,
                                                                           "M7_TOL_K*tol_h"))
    bound = th_value("M7_bound_grip_z_m", expected - drop, label,
                     f"expected support grip - max(M7_FLOOR_M, M7_TOL_K*tol_h); winner {text}",
                     expected_grip_z=expected, tol_h_m=tol.value)
    m_free = aim.get("m_free_m") if isinstance(aim, Mapping) else None
    m_free = 0.0 if m_free is None else max(float(m_free), 0.0)
    drift = th_max("M5_drift_abort_m", Term("DD", M5_FLOOR_M, "M5_FLOOR_M"),
                   Term("ONLINE", M5_FRACTION * m_free, "M5_FRACTION*m_free"), unit="m")
    aim_tol = aim.get("tol_m") if isinstance(aim, Mapping) else None
    return {"S_dest": s_dest, "sigma_dest_m": sigma_dest, "b_low_m": b_low, "expected_support_grip_z": expected,
            "fast_until_grip_z": fast.value, "bound_grip_z": bound.value, "tol_h_m": tol.value,
            "drift_abort_m": drift.value, "aim_tol_m": None if aim_tol is None else [float(v) for v in aim_tol],
            "thresholds": [tol, fast, bound, drift]}


def support_classify(lowering: dict, g_event: float, held: HeldFrame, lateral_error_m: float, *, event: str) -> dict:
    """§4.9 SupportClassify at the support event (bottom = g_event - b_low): |bottom - S_dest| <= tol_h (M8) and the
    lateral error within the aim tolerance + sigma_offset -> on_surface (height right but laterally off ->
    ambiguous: release only if the lateral test passes, else repair); above by more than tol_h -> perched, or
    blocked_above when the event came above the fast-until height (before any support was expected); below ->
    missed_surface; no support cue (event not in SUPPORT_EVENTS, e.g. the M7 bound) -> no_support_within_bound. A
    GuardedLower fast-phase reason "handover_<cue>" (HANDOVER_PREFIX) is classified by its cue (BP-7): a perch or
    an obstacle met during the fast phase is a support event. place_in classes are reserved."""
    s_dest, tol, b_low = float(lowering["S_dest"]), float(lowering["tol_h_m"]), float(held.b_low.value)
    lateral = abs(_finite(lateral_error_m, "lateral_error_m"))
    aim_tol = lowering.get("aim_tol_m") or [M4_FLOOR_M, M4_FLOOR_M]
    lateral_tol = max(float(v) for v in aim_tol) + float(held.sigma_offset_m)
    tol_th = th_value("M8_tol_h_m", tol, "ONLINE", "from lowering_plan")
    lat_th = th_value("M4_aim_tol_m", lateral_tol, "ONLINE", "aim tolerance + sigma_offset (M4, M17)",
                      sigma_offset_m=float(held.sigma_offset_m))
    cue = _base_event(event)
    out = {"S_dest": s_dest, "tol_h_m": tol, "lateral_error_m": lateral, "lateral_tol_m": lateral_tol,
           "event": event, "support_cue": cue if cue in SUPPORT_EVENTS else None, "thresholds": [tol_th, lat_th]}
    if cue not in SUPPORT_EVENTS or g_event is None:
        out.update({"class": "no_support_within_bound", "bottom_z": None})
        return out
    bottom = _finite(g_event, "g_event") - b_low
    dz = bottom - s_dest
    out.update(bottom_z=bottom, dz_m=dz)
    if abs(dz) <= tol + 1e-9:
        out["class"] = "on_surface" if lateral <= lateral_tol + 1e-9 else "ambiguous"
    elif dz > 0.0:
        out["class"] = "blocked_above" if float(g_event) > float(lowering["fast_until_grip_z"]) else "perched"
    else:
        out["class"] = "missed_surface"
    return out


def _convex_hull(points) -> np.ndarray:
    """The convex hull (counter-clockwise, no repeated point) of 2-D points (Andrew's monotone chain)."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=float).reshape(-1, 2)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0.0:
            lower.pop()
        lower.append(pt)
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0.0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def region_hull_radius(part: PartRecord, hf: HeightField | None = None) -> float | None:
    """SAL-R2 (C-7'): the inscribed radius of the region's convex hull around the region's centroid (ONLINE): with a
    HeightField and region cells, the hull of the cells' corners about the mean cell centre; else the hull of the
    region polygon about its centroid; None without a region."""
    if hf is not None and part.region_cells is not None and len(part.region_cells):
        cells = np.asarray(part.region_cells, dtype=int).reshape(-1, 2)
        xy = _cell_centres(hf, cells)
        half = hf.cell_m / 2.0
        points = np.concatenate([xy + np.array([sx * half, sy * half]) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
        centre = xy.mean(axis=0)
    elif part.region_xy is not None and len(part.region_xy) >= 3:
        points = np.asarray(part.region_xy, dtype=float).reshape(-1, 2)
        centre = _polygon_centroid(points)
    else:
        return None
    hull = _convex_hull(points)
    if len(hull) < 3:
        return 0.0
    best = math.inf
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        edge = b - a
        t = float(np.clip(np.dot(centre - a, edge) / max(float(np.dot(edge, edge)), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(centre - (a + t * edge))))
    return best


def destination_precondition(ref: PartRecord, hf: HeightField | None = None) -> dict:
    """C-7' (gap round SAL-R2, lead ruling L1; replaces [C-7]): the destination is measurable before the grasp when
    the lowering can tell it apart from the surrounding support either vertically, step = top - support >=
    max(M8_FLOOR_M, M8_SIGMA_K*sigma_top) / DEST_SEPARATION_FRACTION (a landing on the support lies outside M8's
    tol_h with the [C-7] margin 2), or laterally, the region's hull inscribed radius around its centroid (r_hull,
    ``region_hull_radius``; ``hf`` gives the cell-corner hull, else the region polygon) >= M4_FLOOR_M + HF_CELL_M
    and M8_SIGMA_K*sigma_top <= M8_FLOOR_M (the top is measured to the lowering's floor tolerance). Otherwise an
    empty-hand survey_hover(ref) first. ``cue`` = "vertical" | "lateral" | None; ``values["c7_ok"]`` keeps the [C-7]
    verdict (DEST_SIGMA_K*sigma_top <= DEST_SEPARATION_FRACTION*step) as evidence."""
    if ref.top_z is None or ref.support_z is None:
        return {"ok": False, "reason": "geometry_unmeasured", "cue": None, "values": {}, "thresholds": []}
    step = float(ref.top_z.value) - float(ref.support_z.value)
    sigma_top = float(ref.top_z.sigma)
    limit = DEST_SEPARATION_FRACTION * step / DEST_SIGMA_K
    c7_ok = DEST_SIGMA_K * sigma_top <= DEST_SEPARATION_FRACTION * step + 1e-12
    tol_h = th_max("C7_tol_h_m", Term("DD", M8_FLOOR_M, "M8_FLOOR_M"),
                   Term("ONLINE", M8_SIGMA_K * sigma_top, "M8_SIGMA_K*sigma_top"), unit="m")
    step_min = th_value("C7_vertical_step_min_m", tol_h.value / DEST_SEPARATION_FRACTION, tol_h.label,
                        "C-7' vertical: max(M8_FLOOR_M, M8_SIGMA_K*sigma_top) / DEST_SEPARATION_FRACTION", step_m=step,
                        sigma_top_m=sigma_top)
    r_hull = region_hull_radius(ref, hf)
    r_min = th_value("C7_lateral_r_hull_min_m", M4_FLOOR_M + HF_CELL_M, "DD",
                     "C-7' lateral: M4_FLOOR_M + HF_CELL_M (the aim tolerance floor plus one field cell)",
                     r_hull_m=r_hull)
    vertical = step >= step_min.value - 1e-12
    lateral = (r_hull is not None and r_hull >= r_min.value - 1e-12
               and M8_SIGMA_K * sigma_top <= M8_FLOOR_M + 1e-12)
    ok = bool(vertical or lateral)
    cue = "vertical" if vertical else "lateral" if lateral else None
    return {"ok": ok, "reason": None if ok else "geometry_unmeasured", "cue": cue,
            "values": {"step_m": step, "sigma_top_m": sigma_top, "sigma_top_limit_m": limit, "c7_ok": bool(c7_ok),
                       "vertical_step_min_m": step_min.value, "r_hull_m": r_hull, "r_hull_min_m": r_min.value,
                       "tol_h_m": tol_h.value, "cue": cue},
            "thresholds": [tol_h, step_min, r_min]}
