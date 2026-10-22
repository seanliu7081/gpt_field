"""agent_SpeedUp_v2 observers: the relation-level verdicts of spec §4.10 for the task-2 slice (contract §C9).

Every observer reads only RGB and state8 (through ``StepView``), receives its dependencies at construction [C-4],
returns ``ObsReport`` records carrying the thresholds it compared against, never moves the robot and never calls
the model. Observers embedded in a primitive (wedge, part rotation, effect) end that primitive through its own
decision rules; the bus (``InvariantMonitor``) queues preemption reports of the invariants registered with
``preempt=True`` [C-5], [C-52].

Contents:
- ``WedgeObserver`` (K1w twist sense, K4w end stop, K7 slip rule, G6): the signed finger-gap rate while the closed
  hand twists a gripped rotary part. Rates r_k = gap after step k minus gap before it (k counted from 0); a fresh
  segment skips r_0 (the close / command transient) and is decided at its 12th step; after a reversal the window
  starts at the first non-negative rate after the unwedge fall (Appendix L 6).
- ``PartRotationObserver`` (K7 part observable): the apparent in-plane rotation of a rigid part patch in the wrist
  image (rigid with the hand) against its start appearance, by rotated-template search or by the structure axis.
- ``MotionNoiseMeter`` and ``EffectWatch`` (O1, O7, O8; Appendix L 8): the share of effect-region pixels whose
  CHROMA |d(a*, b*)| changed against the pre-action reference, outside the hand's motion mask, thresholded at
  max(floor, k x the false-change rate measured while the robot MOVED near the region (approach / align /
  descent); a luminance-only change (arm reflections on metal) never counts. Camera choice by visible region share
  (``EffectWatch.select_camera``); the static external camera compares per pixel, a moving camera needs a warped
  reference (``reference_fn``; reserved in the slice [C-27]).
- ``RegionStaysCheck`` [C-48] (stays of supported_on / turned / hold_true), ``CoMotionObserver`` (O3, and the
  review-1 O-1 rotation test of the carried part's rigid patch: ``grip_unstable``),
  ``UprightObserver`` (O4, M3, O2), ``KeepOutObserver`` (O5 contact), ``RestObserver`` (M11),
  ``SpringBackObserver`` (K9), ``HoldFloorInvariant`` (G6 on the bus).
- ``InvariantMonitor`` (the bus) and ``StaysChecker`` (spec §3.3 stays, contract §C3.7).

Geometry (``hand_motion_mask``, ``structure_axis``) is imported lazily inside the functions that need it, so this
module is testable before the geometry module lands; tests inject stand-ins through the ``*_fn`` keyword arguments.
Lab is OpenCV's 8-bit scaling [C-20] (L x 255 / 100, a + 128, b + 128), so |dLab| and |d(a, b)| compare with grey
levels; the conversion restates geometry ``to_lab255`` (equality is tested).

Twist sense (contract §C1.3, [C-25]): +1 is a positive rotation about the approach R2; observers use magnitudes of
the hand yaw about the current approach, so their verdicts do not depend on the sense label.

Step numbers: ``ObsReport.step`` (and EffectWatch's ``onset_step`` / ``confirm_step``) is ``context["step"]`` when
the caller passes an int (the InvariantMonitor passes its own transition count unless the caller gives one), else
the observer's own update count since its ``start`` (a primitive-local count; EffectWatch's ``onset_k`` /
``confirm_k`` are always local). Callers that log policy steps pass ``"step"`` in the context.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import math

import cv2
import numpy as np

from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (HF_CELL_M, HandPose, HeldFrame, KeepOut, ObsReport, RobotModel, Term,
                                     WristCamera, jsonable, poly_contains, th_max, th_value, thresholds_dicts)
from .contact_skills import StepView, quaternion_angle_rad
from .wrist_servo import NCCTrack, WristPatchMonitor

OBSERVERS_VERSION = "agent-speedup-v2-observers-3"

# ------------------------------------------------------------------------------------------------ K1w / K4w / K7
K1W_DECIDE_STEP = 12
K1W_EXTEND_STEP = 16
K1W_WINDOW = 4
K1W_FIRST_RATE_INDEX = 1
K1W_FREE_LW = 0.1
K1W_FREE_SIGMA_K = 4.0
K1W_FREE_EARLY = 0.25
K1W_BLOCKED_LW = 0.5
K1W_BLOCKED_EARLY = 0.75
K1W_UNWEDGE_PER_BLOCKED_STEP = 1.5
K1W_FALL_START_MAX_INDEX = 1
K1W_MIN_LATE_OMEGA_FRACTION = 0.5
K4W_LW = 0.5
K4W_MEDIAN_K = 5.0
K4W_CONSECUTIVE_WINDOWS = 2
K7_FLOOR_DEG = 3.0
K7_SIGMA_K = 3.0
K7_SLIP_LW = 0.5
K7_SLIP_STEPS = 2
K7_SLIP_SIGMA_K = 4.0
K7_SPAN_FACTOR = 2.0
K7_PATCH_RADIUS_PX = 20
K7_SEARCH_PX = 12
K7_ANGLE_STEP_DEG = 1.0
K7_ANGLE_SPAN_DEG = 4.0
K7_MIN_STD_GREY = 4.0
K7_MIN_PEAK = 0.6
K7_MIN_COHERENCE = 0.5
K7_LOST_FRAMES = 2
K7_MIN_JITTER_SAMPLES = 3
K7_MIN_ROTATION_DROP = 0.005
K7_ONSET_DEG = 4.5
K7_STILL_DEG_PER_STEP = 0.25
K7_HYPOTHESIS_MARGIN = 0.4
K7_HYPOTHESIS_SEARCH_PX = 24
K7_CONFIRM_FRAMES = 2
# ------------------------------------------------------------------------------------------------ K9
K9_FLOOR_DEG = 3.0
K9_FLOOR_M = 0.003
K9_SIGMA_K = 3.0
K9_FRAMES = 10
# ------------------------------------------------------------------------------------------------ O1 / O7 / O8
O1_DIFF_FLOOR = 12.0
O1_PX_SIGMA_K = 4.0
O1_FRACTION_FLOOR = 0.02
O1_FALSE_RATE_K = 10.0
O1_FRAMES = 3
O1_HAND_DILATE_PX = 9
O7_FALSE_RATE_MAX = 0.01
O7_MIN_REMAINING_FRACTION = 0.5
O7_MIN_REMAINING_PX = 400
O8_MIN_ROTATION_RAD = math.radians(5.0)
EFFECT_OCCLUDED_FRACTION = 0.5
# ------------------------------------------------------------------------------------------------ O2 / O3 / O4
O2_IOU_MIN = 0.8
O2_SHIFT_FLOOR_PX = 4.0
O2_SIGMA_K = 3.0
O3_WRIST_MAX_PX = 10.0
O3_SIGMA_K = 3.0
O3_EXTERNAL_FLOOR_PX = 2.0
O3_MODEL_FRACTION = 0.5
O3_LOST_FRAMES = 2
O3_ROTATION_MAX_DEG = 5.0
O3_ROTATION_CONFIRM_FRAMES = 2
O3_PATCH_MIN_DIST_M = 0.015
O3_PATCH_TEXTURE_FRACTION = 0.5
O4_TILT_MAX_DEG = 20.0
O4_MIN_COHERENCE = 0.5
O4_REACQUIRE_SPAN_DEG = 40.0
O4_REACQUIRE_STEP_DEG = 2.0
UPRIGHT_REORIENT_RAD = math.radians(15.0)
# ------------------------------------------------------------------------------------------------ M11
M11_FLOOR_GREY = 3.0
M11_SIGMA_K = 3.0
M11_FRAMES = 10
M11_HAND_STILL_M = 0.001
# ------------------------------------------------------------------------------------------------ O5 (contact side)
O5_SAMPLE_SPACING_M = 0.005
O5_HELD_RING_POINTS = 16
NOISE_REPORT_QUANTILE = 0.95
# ------------------------------------------------------------------------------------------------ Lab [C-20]
LAB_RGB_MAX = 255.0
LAB_L_SCALE = 2.55
LAB_AB_OFFSET = 128.0

LABELS = {
    "K1W_DECIDE_STEP": ("DD", "spec §5.2 K1w: a 12-step twist segment is decided at its 12th step (Appendix L 6); "
                              "after a reversal the count starts at the first non-negative rate"),
    "K1W_EXTEND_STEP": ("DD", "spec §5.2 K1w: an undecided segment is extended to its 16th step, then the part "
                              "rotation observer decides (contract §C8 [C-29])"),
    "K1W_WINDOW": ("DD", "spec §5.2 K1w: early and late windows are 4 consecutive rates (Appendix L 6)"),
    "K1W_FIRST_RATE_INDEX": ("DD", "spec §5.2 K1w / Appendix L 6: the early window of a fresh segment skips r_0, "
                                   "which carries the close and command transient (k0 = 1); [F, SP] r[1:5] gives "
                                   "the spec numbers 0.2605 / 0.065 / 0.227"),
    "K1W_FREE_LW": ("DD", "spec §5.2 K1w: free if late <= max(0.1 l w, 4 sigma_g, 0.25 early); [F, SP] 0.1 l w ~ "
                          "0.032 mm/step against a free late rate 0.045 and a blocked 0.445"),
    "K1W_FREE_SIGMA_K": ("DD", "spec §5.2 K1w: the 4 sigma_g gap-noise term of the free test"),
    "K1W_FREE_EARLY": ("DD", "spec §5.2 K1w: free if the late rate fell to <= 0.25 x the early rate"),
    "K1W_BLOCKED_LW": ("DD", "spec §5.2 K1w: blocked if late >= max(0.5 l w, 0.75 early) in the windows ending at "
                             "the decision step and the step before; [F, SP] 0.5 l w ~ 0.16 mm/step"),
    "K1W_BLOCKED_EARLY": ("DD", "spec §5.2 K1w: blocked if the late rate stayed >= 0.75 x the early rate"),
    "K1W_UNWEDGE_PER_BLOCKED_STEP": ("DD", "[C-21] the post-reversal unwedge fall must end within ceil(1.5 x the "
                                           "reversed segment's steps); [F, SP] 13 unwedge steps after a 12-step "
                                           "blocked segment, a fall of ~0.33 mm/step"),
    "K1W_FALL_START_MAX_INDEX": ("DD", "K1w after a reversal: the unwedge fall must begin by r_1 (one noisy "
                                       "non-negative reading at its start is tolerated); with no negative rate by "
                                       "then there is no fall and the window starts at r_0. [F, SP / CP] the falls "
                                       "start at r_0 (-0.092, -0.42 and -0.06 mm/step)"),
    "K1W_MIN_LATE_OMEGA_FRACTION": ("DD", "K1w stall guard: neither class unless the late window's mean hand yaw rate "
                                          "is >= 0.5 x the early window's (a stalled hand makes late -> 0 on a "
                                          "blocked part); spec K1w: the free late / early yaw ratio is 0.82-0.95, so "
                                          "the guard does not move the SP / CP verdicts; [F, real probe recording, "
                                          "review] a stalled blocked twist read 'free' at k = 11 without it"),
    "K4W_LW": ("DD", "spec §5.2 K4w: end stop when two consecutive 4-step windows reach max(0.5 l w, 5 x the median "
                     "rate since the free classification)"),
    "K4W_MEDIAN_K": ("DD", "spec §5.2 K4w: 5 x the median gap rate since the start; [F, SP] the steady rate was "
                           "<= 0.004 mm/step, then 0.18 -> 0.49 mm/step at the limit"),
    "K4W_CONSECUTIVE_WINDOWS": ("DD", "[C-30] K4w: the windows ending at two consecutive steps; [F, SP] fires "
                                      "2-3 steps after the limit, within G4's 5"),
    "K7_FLOOR_DEG": ("DD", "spec §5.2 K7: part rotation relative to the hand >= max(3 deg, 3 sigma_theta)"),
    "K7_SIGMA_K": ("DD", "spec §5.2 K7: the 3 sigma_theta term (measured jitter of the apparent rotation)"),
    "K7_SLIP_LW": ("DD", "spec §5.2 K7: a slip is a gap fall <= -0.5 l w for >= 2 steps outside an expected "
                         "unwedge"),
    "K7_SLIP_STEPS": ("DD", "spec §5.2 K7: the gap fall must last >= 2 steps"),
    "K7_SLIP_SIGMA_K": ("DD", "K7 slip noise floor: the fall must also exceed 4 sigma_g (as K1w's free test), so a "
                              "hand that yaws slowly (l w -> 0) does not read gap noise as a slip [C-8]"),
    "K7_SPAN_FACTOR": ("DD", "K7 part observable: corotation is judged over the trailing hand-yaw span of 2 x the K7 "
                             "threshold, so a blocked part (apparent rotation = the span) and a corotating one (~0) "
                             "lie on either side of the threshold; [F, SP] a free part lags the hand by a pad "
                             "compliance of ~5.6 deg that saturates (gap excess 1.7 mm at 14 mm/rad), so the "
                             "rotation since the start is not the test"),
    "K7_PATCH_RADIUS_PX": ("DD", "K7 rotated-template search: template radius 20 px (41 px square) at the part "
                                 "patch, twice the FeatureTracker radius, so 3 deg of rotation move its edge by "
                                 "~1 px; [F, robot-only real task-2 wrist frames, BO fix] steady free / paused / "
                                 "blocked twists over an 86-patch on-part grid: 0 wrong free verdicts at 16, 20 and "
                                 "24 px; wrong blocked frames 2 / 131 judged (16 px, but 31 patches and the T1 round "
                                 "fake's disk patch unmeasurable), 9 / 168 (20 px, every T1 fake case decided at the "
                                 "12th step), 22 / 213 (24 px)"),
    "K7_SEARCH_PX": ("DD", "K7 rotated-template search: +-12 px around the last match (the patch of a blocked part "
                           "orbits the grip by ~1 px per degree at 60 px from it)"),
    "K7_ANGLE_STEP_DEG": ("DD", "K7 rotated-template search: 1 deg candidate step around the last angle, refined by "
                                "a parabola"),
    "K7_ANGLE_SPAN_DEG": ("DD", "K7 rotated-template search: +-4 deg around the last angle; K2 twists 0.1 rad "
                                "commanded realize 1.1-1.3 deg per step"),
    "K7_MIN_STD_GREY": ("DD", "K7 patch texture: patch std >= 4 grey, the D15 / NCCTrack min_std, else the part is "
                              "unmeasurable by template"),
    "K7_MIN_PEAK": ("DD", "K7 rotated-template search: NCC peak >= 0.6 (v1 NCCTrack min_peak, V1)"),
    "K7_MIN_COHERENCE": ("DD", "K7 structure-axis mode: coherence >= 0.5 for a usable axis (the D10 coherence "
                               "floor)"),
    "K7_LOST_FRAMES": ("DD", "K7: two consecutive unmeasured frames make the part unmeasurable (V1 lost frames)"),
    "K7_MIN_JITTER_SAMPLES": ("DD", "K7: sigma_theta needs >= 3 second differences of the apparent rotation (taken "
                                    "after the bout's onset, so the lag build-up is not counted as jitter)"),
    "K7_MIN_ROTATION_DROP": ("DD", "K7 rotation observability at start: the start template rotated by +-K7_FLOOR_DEG "
                                   "and re-matched with translation search must lose >= 0.005 NCC, else the patch "
                                   "is 'rotation_degenerate' (unmeasurable); [F, real task-2 wrist frames, BO fix] "
                                   "patches on the rotary part's smooth cap drop 0.0011-0.0031 and read 'corotating' "
                                   "when it is blocked; the T1 fakes' patches drop 0.006-0.08, the real ridge-edge "
                                   "patches 0.017-0.075"),
    "K7_ONSET_DEG": ("DD", "K7 onset allowance: a corotation span may start only after 4.5 deg of hand yaw in the "
                           "current motion bout; [F, real steady free twist, BO fix] the free part lags the hand by "
                           "3.0 deg at 3.9 deg of yaw, 4.7 at 7.5, 5.6 at 11.0, 5.7 at 14.6 (SP: ~5.6 deg "
                           "saturating), so a 6 deg span from 4.5 deg changes by < 2 deg; 4.5 deg keeps the first "
                           "verdict at the 12th step of a 1.15 deg/step twist, where TwistSegment decides"),
    "K7_STILL_DEG_PER_STEP": ("DD", "K7 motion bout: a step with < 0.25 deg of hand yaw (a quarter of K2's realized "
                                    "1.1-1.3 deg/step) ends the bout; [F, real probe recording, review] the part "
                                    "caught up ~3 deg while the hand yawed < 0.3 deg/step and the lag re-developed "
                                    "at the next motion onset"),
    "K7_HYPOTHESIS_MARGIN": ("DD", "K7 span hypothesis test: the span-start patch matched in the current frame at "
                                   "0 rotation (corotating) and at the span's hand yaw (blocked); the winner must lead "
                                   "by >= 0.4 x the template's own contrast between the two; [F, BO fix] 0.4 keeps "
                                   "every T1 fake verdict at the 12th step and gives 0 wrong free verdicts on the real "
                                   "grid (0.5 loses the free round fake at the 12th step, 0.3 adds wrong blocked "
                                   "frames)"),
    "K7_HYPOTHESIS_SEARCH_PX": ("DD", "K7 span hypothesis test: +-24 px translation search, the orbit of a blocked "
                                      "patch over a 6 deg span at <= 200 px from the axis (21 px)"),
    "K7_CONFIRM_FRAMES": ("DD", "K7: a verdict needs the same class on 2 consecutive judged frames (V1 2-frame "
                                "rule); [F, real probe recordings, review] single-frame flips at motion onsets"),
    "K9_FLOOR_DEG": ("DD", "spec §5.2 K9: no spring-back if the part changes <= max(3 deg, 3 mm, 3 sigma) over 10 "
                           "frames after release"),
    "K9_FLOOR_M": ("DD", "spec §5.2 K9: the 3 mm position floor"),
    "K9_SIGMA_K": ("DD", "spec §5.2 K9: the 3 sigma term"),
    "K9_FRAMES": ("DD", "spec §5.2 K9: 10 frames after release"),
    "O1_DIFF_FLOOR": ("DD", "spec §5.2 O1: a pixel changed if |dLab| > max(12, 4 sigma_px); 12 = v1 DIFF_THRESHOLD "
                            "(external_axes.py:25); Appendix L 8 applies it to the chroma |d(a*, b*)|"),
    "O1_PX_SIGMA_K": ("DD", "spec §5.2 O1 / Appendix L 8: the 4 sigma pixel-noise term"),
    "O1_FRACTION_FLOOR": ("DD", "spec §5.2 O1 / Appendix L 8: effect when the changed share >= max(2 %, k x the "
                                "motion false-change rate)"),
    "O1_FALSE_RATE_K": ("DD", "spec §5.2 O1 / Appendix L 8: k = 10 x the false-change rate measured while the robot "
                              "moves near the region"),
    "O1_FRAMES": ("DD", "spec §5.2 O1: 3 consecutive frames"),
    "O1_HAND_DILATE_PX": ("DD", "O1 hand mask dilation: a 9 px square kernel, the v1 hand_image_motion dilation "
                                "(external_axes.py:53)"),
    "O7_FALSE_RATE_MAX": ("DD", "spec §5.2 O7: the region's false-change rate (Appendix L 8: the motion rate) must "
                                "be < 1 %"),
    "O7_MIN_REMAINING_FRACTION": ("DD", "[C-26] / Appendix L 2: >= 50 % of the region must remain outside the "
                                        "swept and part masks; no robot-only data yet; re-derive at V5"),
    "O7_MIN_REMAINING_PX": ("DD", "[C-26] / Appendix L 2: >= 400 px must remain (O1's 2 % of 400 px is 8 changed "
                                  "pixels, above the 1 % false rate); no robot-only data yet; re-derive at V5"),
    "O8_MIN_ROTATION_RAD": ("DD", "spec §5.2 O8: an onset counts only after >= 5 deg of part rotation"),
    "EFFECT_OCCLUDED_FRACTION": ("DD", "spec §4.10 effect_occluded: a frame with > 50 % of the region masked is "
                                       "skipped"),
    "O2_IOU_MIN": ("DD", "spec §5.2 O2: rigid hold silhouette IoU >= 0.8"),
    "O2_SHIFT_FLOOR_PX": ("DD", "spec §5.2 O2: silhouette centroid shift <= max(4 px, 3 sigma_track)"),
    "O2_SIGMA_K": ("DD", "spec §5.2 O2: the 3 sigma_track term"),
    "O3_WRIST_MAX_PX": ("DD", "spec §5.2 O3: the part template stays within 10 px in the wrist view (v1 "
                              "WristPatchMonitor default, wrist_servo.py:1248)"),
    "O3_SIGMA_K": ("DD", "spec §5.2 O3: the external template moves by J_ext dp +- 3 sigma"),
    "O3_EXTERNAL_FLOOR_PX": ("DD", "O3 external sigma floor 2 px: the NCC subpixel track and the affine J_ext "
                                   "prediction over a 2.5 cm load-test lift (~14 px) are not better than ~2 px; the "
                                   "measured locator / projector sigma wins when larger"),
    "O3_MODEL_FRACTION": ("DD", "O3 external tolerance also scales with the prediction: max(3 sigma, 0.5 |J dp|), "
                                "the v1 ImageJacobian.broyden max_innovation_fraction (geometry "
                                "BROYDEN_MAX_INNOVATION, restated; a test asserts equality); [F, SURVEY-pose fake "
                                "camera, BO fix] a 10 cm lift + 22.6 cm transit + 10 cm lowering deviates from J dp by "
                                "<= 0.08 |J dp| (J at the object) and <= 0.23 |J dp| (J from 15 cm away), up to 29 px "
                                "against the old constant 6 px; an object left behind deviates by |J dp|"),
    "O3_LOST_FRAMES": ("DD", "O3: a template miss counts as violated only after 2 consecutive misses (V1 2-frame "
                             "loss, as K7_LOST_FRAMES); one occluded frame is pending"),
    "O3_ROTATION_MAX_DEG": ("DD", "O-1 (review-1 decision §2.5): the carried part's in-hand rotation since the load "
                                  "test started (the largest eligible, height-robust wrist patch estimate) must stay "
                                  "<= 5 deg, else grip_unstable. FITTED on the RS-7 holds it is scored on (episodes "
                                  "0-4: runs/agent_speedup_v2_bo_r1_rs7_002, re-run after the r1 fix as "
                                  "runs/agent_speedup_v2_bo_r1_fix_rs7_001): estimate 0.08-2.81 deg on the 5 "
                                  "reference holds (truth 0.10-2.73), 5.51-10.75 on the 20 holds with the jaw +-10 deg "
                                  "and +-4 mm "
                                  "along it (truth 6.0-14.3); held-out episodes 5-9 "
                                  "(runs/agent_speedup_v2_bo_r1_fix_oos_001): 0.13-2.60 reference, 5.47-14.16 forced, "
                                  "and all 63 holds of both runs that tilted > 15 deg in a further 6 cm lift read >= "
                                  "5.42. Thin margin: those that turned only 5.4-6.0 deg in the 2.5 cm lift read "
                                  "5.42-5.6; re-derive on held-out data and at V5. Review-2 RS-7 extended (decision "
                                  "§6.2; the value unchanged; jaw 8, 10 and 13.4 deg, 13.4 = the largest accepted "
                                  "wrong axis, each with the four sign sets of +-4 mm along the jaw; the 8 and 13.4 "
                                  "deg classes unseen by the fit): runs/agent_speedup_v2_bo_r2_rs7_001 (episodes 0-4) "
                                  "120/120 forced holds and 0/5 reference flagged; runs/agent_speedup_v2_bo_r2_rs7_oos_"
                                  "001 (episodes 5-9) 119/120 forced (the 8 deg class 19/20) and 0/5 reference; the "
                                  "one miss read 3.93 deg on a 5.44 deg turn that tilted 5.6 deg after the further "
                                  "6 cm lift; every forced hold of both runs that tilted > 10 deg there was flagged "
                                  "(84/84: 38 fit + 46 held-out); the marginal set is jaw -8 deg / -4 mm (realized "
                                  "-4.7..-8.7 mm): 8/10 latched, 1/10 flagged only by the external lag (grip_unstable "
                                  "on the last 5 of its 11 lift frames; wrist 4.73 deg on a 6.83 deg turn), 1/10 the "
                                  "miss; estimate / true turn 0.67-1.15 (median 0.90). The lateral is COMMANDED: the "
                                  "+4 mm sets realize -0.8..+3.6 mm at the close and the -4 mm sets -4.7..-8.7 mm, so "
                                  "4 mm is realized on the - side only (commanded vs realized: a lead ruling); +6 mm "
                                  "commanded (a realized +1.4..+5.5 mm, runs/agent_speedup_v2_bo_r2_rs7_lat6_001): "
                                  "60/60 flagged, the smallest final estimate 5.01 deg; binned by realized lateral "
                                  "over episodes 0-9 (the tool's --realized), every hold at >= +4 mm flagged (28/28) "
                                  "and 59/60 at <= -4 mm. CLEAN MARGIN: reference final estimates 0.08-2.81 (fit) and "
                                  "0.13-2.60 (held-out), per-frame maxima 3.30 and 3.10; fresh episodes 10-19 "
                                  "(runs/agent_speedup_v2_bo_r2_review_rs7_fresh_001, 0/10 clean and 120/120 forced "
                                  "flagged, forced minimum 5.95) read 0.10-1.83 but one 4.29 deg, 0.71 under the "
                                  "threshold: a single-frame in-plane spike of the far patch on the last lift frame "
                                  "(its sphere angle 1.88, the other patch 1.65, truth flat at 2.21); only the 2-frame "
                                  "confirm (O3_ROTATION_CONFIRM_FRAMES) keeps it unflagged, so a 2-frame spike would "
                                  "false-flag a clean hold: watch the clean false-positive count end to end (BR / BX, "
                                  "the joint gate)"),
    "O3_ROTATION_CONFIRM_FRAMES": ("DD", "O-1: grip_unstable needs the rotation over its threshold on 2 consecutive "
                                         "frames (V1 2-frame rule, as K7_CONFIRM_FRAMES), then latches; a wrist "
                                         "left-behind reading sustained over the same 2 frames overrides the latch"),
    "O3_PATCH_MIN_DIST_M": ("DD", "O-1 wrist patches: >= 15 mm from the pinch point (the scene's rigid-patch rule, "
                                  "RIGID_PATCH_MIN_DIST_M, spec §4.5 step 8, restated: observers may not import the "
                                  "scene); nearer, a rotation moves the patch by < 1 px per degree"),
    "O3_PATCH_TEXTURE_FRACTION": ("DD", "O-1 far patch: the farthest candidate among those with >= 0.5 x the best "
                                        "candidate's window std; [F, BO r1 RS-7 frames] the farthest candidate "
                                        "overall sat on the bright top of the carried part (std 5-7 grey against "
                                        "102 at its edges) and over-read the in-hand rotation by up to 4 deg"),
    "O4_TILT_MAX_DEG": ("DD", "spec §5.2 O4: in-hand tilt change <= 20 deg ([F] the D003 object hung 15 deg and "
                              "was placeable)"),
    "O4_MIN_COHERENCE": ("DD", "O4 external principal axis: the structure-axis coherence must be >= 0.5 at the "
                               "start and now (the D10 coherence floor, as K7_MIN_COHERENCE), else the tilt is not "
                               "measured from the external camera"),
    "O4_REACQUIRE_SPAN_DEG": ("DD", "O4: when the held object's external template is lost (a tilt beyond ~20 deg "
                                    "defeats the translation-only match: [F, BO fix] an elongated object loses it "
                                    "at 24 deg), the start template is searched rotated over +-40 deg (2 x "
                                    "O4_TILT_MAX_DEG), so a tilt past the threshold is measured, not hidden"),
    "O4_REACQUIRE_STEP_DEG": ("DD", "O4 re-acquisition: 2 deg steps (a tenth of O4_TILT_MAX_DEG)"),
    "UPRIGHT_REORIENT_RAD": ("DD", "spec §5.2 M3 / O4: the hand within 15 deg of the grasp orientation; restated "
                                   "from the primitives' M3_REORIENT_RAD because observers may not import "
                                   "primitives (§C2.2); a unit test asserts equality"),
    "M11_FLOOR_GREY": ("DD", "spec §5.2 M11: at rest when the frame difference over the object region <= max(3 "
                             "grey, 3 sigma_static) for 10 frames"),
    "M11_SIGMA_K": ("DD", "spec §5.2 M11: the 3 sigma_static term"),
    "M11_FRAMES": ("DD", "spec §5.2 M11: 10 frames"),
    "M11_HAND_STILL_M": ("DD", "M11 precondition: a frame counts only when the grip moved <= 1 mm in the step; [F, "
                               "real probe recording, review] release 0.005-0.085 mm/step, static after the retreat "
                               "0.04-0.59 mm/step, retreat 0.8-12.7 mm/step (its frames read 11.7-18.4 grey over a "
                               "static part, against the 3 grey threshold)"),
    "O5_SAMPLE_SPACING_M": ("DD", "O5 contact test: the hand footprint is sampled every 5 mm (one HeightField cell) "
                                  "for the keep-out intersection"),
    "O5_HELD_RING_POINTS": ("DD", "O5 contact test: the held footprint circle (radius r_obj) is sampled by 16 "
                                  "points"),
    "LAB_RGB_MAX": ("DD", "[C-20] 8-bit RGB full scale (geometry RGB_MAX, restated)"),
    "LAB_L_SCALE": ("DD", "[C-20] OpenCV 8-bit Lab scaling: L x 255 / 100 (geometry LAB_L_SCALE, restated)"),
    "LAB_AB_OFFSET": ("DD", "[C-20] OpenCV 8-bit Lab scaling: a + 128, b + 128 (geometry LAB_AB_OFFSET, restated)"),
    "NOISE_REPORT_QUANTILE": ("DD", "MotionNoiseMeter report only: the 95th percentile of the moving-frame changed "
                                    "shares is logged beside the mean (the rate O1 / O7 use); never compared"),
}

VERDICT_MET, VERDICT_VIOLATED, VERDICT_PENDING, VERDICT_FAILED = "met", "violated", "pending", "failed"
WEDGE_MODES = ("sense", "end_stop")
EFFECT_PHASES = ("all", "sense", "free")
STATISTICS = ("chroma", "lab")
STAYS_KINDS = ("unchanged", "effect")
MONITOR_SCOPES = ("relation", "held", "episode")
STAYS_ENTRY_KINDS = ("step", "hold_true")
CONTACT_EVENTS = ("contact_stall", "contact_touch", "gap_rise", "gap_jump", "held_object_contact", "finger_contact",
                  "stall", "touch")


# ======================================================================================================== helpers
def _finite(value, name) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def _nonnegative(value, name) -> float:
    out = _finite(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return out


def _view(view, name="view") -> StepView:
    if not isinstance(view, StepView):
        raise TypeError(f"{name} must be a StepView")
    return view


def _mask(mask, name, shape=None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D mask")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    return array.astype(bool)


def _report(name, verdict, reason, evidence, thresholds, step, progress=None) -> ObsReport:
    return ObsReport(name=name, verdict=verdict, progress=None if progress is None else float(progress),
                     reason=reason, evidence=jsonable(evidence), thresholds=tuple(thresholds), step=int(step))


def _step(context, count) -> int:
    value = (context or {}).get("step") if isinstance(context, Mapping) else None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return int(count)


def _lab255(rgb) -> np.ndarray:
    """[C-20] CIE Lab in OpenCV 8-bit scaling (L x 255 / 100, a + 128, b + 128) as float32 from float RGB: the
    geometry module's ``to_lab255`` restated (same algorithm and constants; a unit test asserts equality), so the
    effect statistic does not depend on the geometry module at run time."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("expected an HxWx3 RGB image")
    image = np.ascontiguousarray(array[..., :3].astype(np.float32) / np.float32(LAB_RGB_MAX))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    out = np.empty_like(lab)
    out[..., 0] = lab[..., 0] * np.float32(LAB_L_SCALE)
    out[..., 1:] = lab[..., 1:] + np.float32(LAB_AB_OFFSET)
    return out


def _gray(rgb) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(rgb))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def change_magnitude(lab_now: np.ndarray, lab_ref: np.ndarray, statistic: str = "chroma") -> np.ndarray:
    """Per-pixel change of Lab255 images: ``chroma`` = |d(a*, b*)| (Appendix L 8: a luminance-only change, such as
    an arm reflection on metal, does not count); ``lab`` = |dLab| (spec O1 as written)."""
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}")
    d = np.asarray(lab_now, dtype=np.float32) - np.asarray(lab_ref, dtype=np.float32)
    if statistic == "chroma":
        return np.hypot(d[..., 1], d[..., 2])
    return np.sqrt(np.sum(d * d, axis=-1))


def o1_pixel_threshold(sigma_px: float) -> Threshold:
    """O1 per-pixel test: changed if the change > max(O1_DIFF_FLOOR, O1_PX_SIGMA_K sigma_px)."""
    sigma = _nonnegative(sigma_px, "sigma_px")
    return th_max("O1_pixel_diff", Term("DD", O1_DIFF_FLOOR, "O1_DIFF_FLOOR"),
                  Term("ONLINE", O1_PX_SIGMA_K * sigma, "O1_PX_SIGMA_K*sigma_px"), unit="Lab255")


def o1_fraction_threshold(false_rate: float) -> Threshold:
    """O1 share test (Appendix L 8): changed share >= max(O1_FRACTION_FLOOR, O1_FALSE_RATE_K x the motion
    false-change rate)."""
    rate = _nonnegative(false_rate, "false_rate")
    return th_max("O1_fraction", Term("DD", O1_FRACTION_FLOOR, "O1_FRACTION_FLOOR"),
                  Term("ONLINE", O1_FALSE_RATE_K * rate, "O1_FALSE_RATE_K*motion_false_rate"), unit="share")


def _dilate(mask, size) -> np.ndarray:
    if mask is None:
        return None
    kernel = np.ones((int(size), int(size)), np.uint8)
    return cv2.dilate(np.asarray(mask, bool).astype(np.uint8), kernel) > 0


def _default_hand_mask(before_rgb, after_rgb):
    from . import agent_speedup_v2_geometry as geometry
    return geometry.hand_motion_mask(before_rgb, after_rgb)


def _default_axis(gray, uv):
    from . import agent_speedup_v2_geometry as geometry
    return geometry.structure_axis(gray, uv)


def _hand_mask(fn, before_rgb, after_rgb, shape) -> np.ndarray:
    """The dilated hand motion mask between two frames (empty when the hand did not move visibly)."""
    raw = (fn or _default_hand_mask)(before_rgb, after_rgb)
    if raw is None:
        return np.zeros(shape, bool)
    return _dilate(_mask(raw, "hand motion mask", shape), O1_HAND_DILATE_PX)


def _static_mask(fn, view, shape) -> np.ndarray:
    if fn is None:
        return np.zeros(shape, bool)
    raw = fn(view)
    return np.zeros(shape, bool) if raw is None else _mask(raw, "hand footprint mask", shape)


def _camera_rgb(view: StepView, camera: str) -> np.ndarray:
    if camera == "external":
        return view.external_rgb_upright
    if camera == "wrist":
        return view.wrist_rgb_upright
    raise ValueError("camera must be 'external' or 'wrist'")


def _window_mean(values, end, size):
    """Mean of values[end - size + 1 : end + 1] (None when the window is incomplete)."""
    lo = end - size + 1
    if lo < 0 or end >= len(values):
        return None
    return float(np.mean(values[lo:end + 1]))


def _axis_angle_deg(axis, ref) -> float:
    """Signed angle (deg) from image axis ``ref`` to ``axis``, both undirected (wrapped to (-90, 90] deg)."""
    a = math.atan2(float(axis[1]), float(axis[0]))
    b = math.atan2(float(ref[1]), float(ref[0]))
    d = math.remainder(a - b, math.pi)              # in [-pi/2, pi/2]
    if d <= -math.pi / 2.0:
        d += math.pi
    return math.degrees(d)


def _jitter_sigma(series) -> float:
    """Measurement jitter of a smooth series: RMS of its second differences / sqrt(sum of the stencil squares)."""
    values = np.asarray(series, dtype=float)
    stencil = np.array([1.0, -2.0, 1.0])
    if values.size < len(stencil) + K7_MIN_JITTER_SAMPLES - 1:
        return 0.0
    second = np.convolve(values, stencil, mode="valid")
    return float(math.sqrt(float(np.mean(second * second)) / float(np.sum(stencil * stencil))))


# ================================================================================================ WedgeObserver
class WedgeObserver:
    """K1w (mode ``sense``), K4w (mode ``end_stop``), the K7 slip rule and the G6 hold floor (contract §C9).

    Indexing (Appendix L 6): r_k (k = 0, 1, ...) = gap after step k - gap before step k (mm/step, signed); w_k =
    |hand yaw about the current approach| (rad/step); l w = pad length (R4, mm) x the mean w over a window.

    ``sense``: a fresh segment's early window is r[1:5] and its decision step is the 12th (k = 11), extended to the
    16th. After a reversal (``after_reversal``) the unwedge fall comes first: k0 = the first k with r_k >= 0 after
    it, the early window is r[k0:k0+4] and the decision is at k - k0 + 1 = 12 (then 16); the fall must end within
    ceil(1.5 x ``blocked_steps``) steps [C-21] (None: the longest segment, the extend step), else failed
    ``sense_undecided``. At a decision step: free if late <= max(0.1 l w, 4 sigma_g, 0.25 early) -> met
    ``free_sense``; blocked if late >= max(0.5 l w, 0.75 early) in the windows ending at the decision step and the
    step before -> violated ``blocked_sense``; else pending (class ``pending``, then ``undecided`` at the extend
    step). ``end_stop``: met ``end_stop`` when the 4-step windows ending at two consecutive steps both reach
    max(0.5 l w, 5 x the median rate since the start, taken over the rates before those windows) [C-30]. Both
    modes: gap < the hold floor -> violated ``hold_lost``; a fall <= -max(0.5 l w, 4 sigma_g) for 2 steps outside
    the unwedge -> violated ``turn_slip``. A decision is sticky; ``hold_lost`` is checked first on every frame.

    Stall guard: neither class is given while the late window's mean hand yaw rate is < K1W_MIN_LATE_OMEGA_FRACTION
    x the early window's (a hand that stops yawing makes the late rate fall on a blocked part too); the window
    record carries ``hand_stalled`` and the extend step reports class ``undecided``. ``hold_floor_mm`` is the G6
    Threshold (primitives.g6_hold_floor, its winner label is carried into the reports) or a number (reported as an
    ONLINE value)."""

    name = "wedge"

    def __init__(self, *, mode: str, robot: RobotModel, sigma_g_mm_per_step: float,
                 hold_floor_mm: "float | Threshold", after_reversal: bool = False, blocked_steps: int | None = None):
        if mode not in WEDGE_MODES:
            raise ValueError(f"mode must be one of {WEDGE_MODES}")
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        self.mode = mode
        self.robot = robot
        self.sigma_g = _nonnegative(sigma_g_mm_per_step, "sigma_g_mm_per_step")
        self.hold_floor_th = hold_floor_mm if isinstance(hold_floor_mm, Threshold) else None
        self.hold_floor_mm = _finite(hold_floor_mm.value if self.hold_floor_th is not None else hold_floor_mm,
                                     "hold_floor_mm")
        self.after_reversal = bool(after_reversal)
        if blocked_steps is not None and (isinstance(blocked_steps, bool) or int(blocked_steps) != blocked_steps
                                          or blocked_steps < 0):
            raise ValueError("blocked_steps must be a nonnegative integer or None")
        self.blocked_steps = None if blocked_steps is None else int(blocked_steps)
        span = K1W_EXTEND_STEP if self.blocked_steps is None else self.blocked_steps
        self.unwedge_limit = int(math.ceil(K1W_UNWEDGE_PER_BLOCKED_STEP * span))
        self.pad_length_mm = robot.pad_length_m * 1000.0
        self.started = False

    def start(self, view) -> None:
        view = _view(view)
        self.gap0 = view.gap_mm
        self.last_gap = view.gap_mm
        self.rates: list[float] = []
        self.omega: list[float] = []
        self.k0 = None if self.after_reversal else K1W_FIRST_RATE_INDEX
        self.count_origin = None if self.after_reversal else 0
        self.fall_seen = False
        self.slip_run = 0
        self.decision = None             # (verdict, reason, class, k)
        self.cls = "pending"
        self.last_early = self.last_late = self.last_lw = None
        self.thresholds: list[Threshold] = [self.hold_floor_th or th_value(
            "G6_hold_floor_mm", self.hold_floor_mm, "ONLINE",
            "hold floor passed in as a number (pass primitives.g6_hold_floor's Threshold to carry its winner label)")]
        self.decided_at = None
        self.started = True

    # ----------------------------------------------------------------------------------------------- numbers
    def _lw(self, end, size) -> float:
        lo = max(0, end - size + 1)
        window = self.omega[lo:end + 1]
        return self.pad_length_mm * float(np.mean(window)) if window else 0.0

    def _unwedge(self, k) -> None:
        """After a reversal: locate k0 (the first non-negative rate after the initial fall)."""
        r = self.rates[k]
        if r < 0.0:
            self.fall_seen = True
            return
        if self.fall_seen:
            self.k0, self.count_origin = k, k
        elif k >= K1W_FALL_START_MAX_INDEX:
            self.k0, self.count_origin = 0, 0         # no fall by r_1: nothing to unwedge

    def _slip_threshold(self, k) -> Threshold:
        lw = self._lw(k, K1W_WINDOW)
        return th_max("K7_slip_rate_mm", Term("ONLINE", K7_SLIP_LW * lw, "K7_SLIP_LW*l*omega"),
                      Term("ONLINE", K7_SLIP_SIGMA_K * self.sigma_g, "K7_SLIP_SIGMA_K*sigma_g"), unit="mm/step")

    def _classify(self, k):
        """The K1w rules on the windows ending at step k (no state change): (class, thresholds, record) with class
        "free" | "blocked" | "pending" (neither) | "unknown" (windows incomplete)."""
        if self.k0 is None:
            return "unknown", [], {}
        early = _window_mean(self.rates, self.k0 + K1W_WINDOW - 1, K1W_WINDOW)
        late = _window_mean(self.rates, k, K1W_WINDOW)
        late_prev = _window_mean(self.rates, k - 1, K1W_WINDOW)
        if early is None or late is None or late_prev is None or k - 1 - K1W_WINDOW + 1 < self.k0:
            return "unknown", [], {}
        lw, lw_prev = self._lw(k, K1W_WINDOW), self._lw(k - 1, K1W_WINDOW)
        free = th_max("K1w_free_rate_mm", Term("ONLINE", K1W_FREE_LW * lw, "K1W_FREE_LW*l*omega"),
                      Term("ONLINE", K1W_FREE_SIGMA_K * self.sigma_g, "K1W_FREE_SIGMA_K*sigma_g"),
                      Term("ONLINE", K1W_FREE_EARLY * early, "K1W_FREE_EARLY*early"), unit="mm/step")
        blocked = th_max("K1w_blocked_rate_mm", Term("ONLINE", K1W_BLOCKED_LW * lw, "K1W_BLOCKED_LW*l*omega"),
                         Term("ONLINE", K1W_BLOCKED_EARLY * early, "K1W_BLOCKED_EARLY*early"), unit="mm/step")
        blocked_prev = th_max("K1w_blocked_rate_mm", Term("ONLINE", K1W_BLOCKED_LW * lw_prev,
                                                          "K1W_BLOCKED_LW*l*omega(prev window)"),
                              Term("ONLINE", K1W_BLOCKED_EARLY * early, "K1W_BLOCKED_EARLY*early"), unit="mm/step")
        omega_early = float(np.mean(self.omega[self.k0:self.k0 + K1W_WINDOW]))
        omega_late = float(np.mean(self.omega[k - K1W_WINDOW + 1:k + 1]))
        stall = th_value("K1w_late_omega_rad", K1W_MIN_LATE_OMEGA_FRACTION * omega_early, "ONLINE",
                         "K1W_MIN_LATE_OMEGA_FRACTION x the early window's mean hand yaw rate (rad/step)",
                         omega_early=omega_early)
        stalled = omega_early <= 0.0 or omega_late < stall.value
        record = {"k": k, "early": early, "late": late, "late_prev": late_prev, "lw": lw, "lw_prev": lw_prev,
                  "free_threshold": free.value, "blocked_threshold": blocked.value,
                  "blocked_threshold_prev": blocked_prev.value, "omega_early": omega_early,
                  "omega_late": omega_late, "hand_stalled": stalled}
        if stalled:
            cls = "hand_stalled"
        elif late <= free.value:
            cls = "free"
        elif late >= blocked.value and late_prev >= blocked_prev.value:
            cls = "blocked"
        else:
            cls = "pending"
        return cls, [free, blocked, stall], record

    def classify_now(self) -> dict:
        """Diagnosis: the K1w class the rules give on the windows ending at the latest step, whatever the decision
        step (a probe trace shorter than the decision step; the verdict itself is never changed)."""
        if not self.started or not self.rates:
            return {"class": "unknown"}
        cls, ths, record = self._classify(len(self.rates) - 1)
        return jsonable({"class": cls, **record, "thresholds": thresholds_dicts(ths)})

    def _sense_decision(self, k):
        n = k - self.count_origin + 1
        if n not in (K1W_DECIDE_STEP, K1W_EXTEND_STEP):
            return None
        cls, ths, record = self._classify(k)
        if cls == "unknown":
            return None
        self.last_early, self.last_late, self.last_lw = record["early"], record["late"], record["lw"]
        self.window_record = {**record, "decision_index": n}
        self.thresholds = [self.thresholds[0], *ths]
        if cls == "free":
            return VERDICT_MET, "free_sense", "free"
        if cls == "blocked":
            return VERDICT_VIOLATED, "blocked_sense", "blocked"
        if n == K1W_EXTEND_STEP:                     # neither, or a stalled hand (window["hand_stalled"])
            return VERDICT_PENDING, None, "undecided"
        return None

    def _end_stop_decision(self, k):
        first = k - K4W_CONSECUTIVE_WINDOWS + 1 - K1W_WINDOW + 1          # first rate of the candidate windows
        if first < 0:
            return None
        # the steady rate since the start: the rates BEFORE the candidate windows (a median that included them would
        # rise with the stop itself when few steady steps precede it); none yet -> no median term
        before = self.rates[:first]
        median = float(np.median(before)) if before else 0.0
        ok, last = [], None
        for end in range(k - K4W_CONSECUTIVE_WINDOWS + 1, k + 1):
            mean = _window_mean(self.rates, end, K1W_WINDOW)
            lw = self._lw(end, K1W_WINDOW)
            thr = th_max("K4w_stop_rate_mm", Term("ONLINE", K4W_LW * lw, "K4W_LW*l*omega"),
                         Term("ONLINE", K4W_MEDIAN_K * median, "K4W_MEDIAN_K*median_rate"), unit="mm/step")
            ok.append(mean is not None and mean >= thr.value)
            last = (mean, lw, thr)
        mean, lw, thr = last
        self.last_late, self.last_lw = mean, lw
        self.window_record = {"median_rate": median, "median_rates": len(before), "stop_threshold": thr.value}
        self.thresholds = [self.thresholds[0], thr]
        if all(ok):
            return VERDICT_MET, "end_stop", "end_stop"
        return None

    # ---------------------------------------------------------------------------------------------- update
    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("WedgeObserver.start(view) must be called first")
        before, after = _view(before, "before"), _view(after, "after")
        rate = after.gap_mm - self.last_gap
        self.last_gap = after.gap_mm
        w = abs(self.robot.yaw_about_approach(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2]))
        self.rates.append(float(rate))
        self.omega.append(float(w))
        k = len(self.rates) - 1
        step = _step(context, len(self.rates))
        if after.gap_mm < self.hold_floor_mm:
            return self._out(VERDICT_VIOLATED, "hold_lost", step, k)
        if self.decision is not None:
            verdict, reason, _cls, _k = self.decision
            return self._out(verdict, reason, step, k)
        in_unwedge = self.mode == "sense" and self.k0 is None
        if in_unwedge:
            self._unwedge(k)
            if self.k0 is None and k + 1 > self.unwedge_limit:
                self.decision = (VERDICT_FAILED, "sense_undecided", "unwedge_limit", k)
                self.cls, self.decided_at = "unwedge_limit", k
                self.thresholds = [self.thresholds[0], self._unwedge_threshold()]
                return self._out(VERDICT_FAILED, "sense_undecided", step, k)
        slip = self._slip_threshold(k)
        outside = not in_unwedge and (self.mode == "end_stop" or k >= (self.k0 or 0))
        self.slip_run = self.slip_run + 1 if (outside and rate <= -slip.value and slip.value > 0.0) else 0
        if self.slip_run >= K7_SLIP_STEPS:
            self.thresholds = [self.thresholds[0], slip]
            return self._out(VERDICT_VIOLATED, "turn_slip", step, k)
        if self.mode == "sense":
            found = None if self.k0 is None else self._sense_decision(k)
        else:
            found = self._end_stop_decision(k)
        if found is not None:
            verdict, reason, cls = found
            self.cls = cls
            if cls != "undecided":
                self.decision = (verdict, reason, cls, k)
                self.decided_at = k
            if self.after_reversal and self.mode == "sense":
                self.thresholds.append(self._unwedge_threshold())
            return self._out(verdict, reason, step, k)
        return self._out(VERDICT_PENDING, None, step, k)

    def _unwedge_threshold(self) -> Threshold:
        return th_value("K1w_unwedge_steps", float(self.unwedge_limit), "DD",
                        "ceil(K1W_UNWEDGE_PER_BLOCKED_STEP x blocked steps)", blocked_steps=self.blocked_steps
                        if self.blocked_steps is not None else K1W_EXTEND_STEP)

    def _progress(self, k):
        if self.mode == "sense":
            if self.count_origin is None:
                return None
            return float(min(1.0, (k - self.count_origin + 1) / K1W_DECIDE_STEP))
        if self.last_late is None or not getattr(self, "window_record", None):
            return None
        thr = self.window_record.get("stop_threshold")
        return None if not thr else float(self.last_late / thr)

    def _out(self, verdict, reason, step, k) -> ObsReport:
        evidence = {"mode": self.mode, "k": k, "rates": [round(r, 4) for r in self.rates],
                    "early": self.last_early, "late": self.last_late, "lw": self.last_lw,
                    "omega": float(np.mean(self.omega[-K1W_WINDOW:])) if self.omega else None, "class": self.cls,
                    "decided_at": self.decided_at, "k0": self.k0, "after_reversal": self.after_reversal,
                    "unwedge_steps": self.k0 if self.after_reversal else None, "unwedge_limit": self.unwedge_limit
                    if self.after_reversal else None, "gap_mm": self.last_gap, "gap0_mm": self.gap0,
                    "window": dict(getattr(self, "window_record", {}) or {})}
        return _report(self.name, verdict, reason, evidence, self.thresholds, step, self._progress(k))


# ======================================================================================== rotated-template tracking
class _RotationTracker:
    """Apparent in-plane rotation (deg) and position (px) of a gray-image patch against its start appearance:
    rotated-template NCC search around the last angle and position (TM_CCOEFF_NORMED), parabola-refined."""

    def __init__(self, gray0, uv, *, radius_px):
        gray = np.asarray(gray0, dtype=np.float32)
        self.r = int(radius_px)
        self.big = int(math.ceil(self.r * math.sqrt(2.0))) + 1
        h, w = gray.shape
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        self.reason = None
        if not (self.big <= u < w - self.big and self.big <= v < h - self.big):
            self.reason = "patch_out_of_view"
            return
        self.source = gray[v - self.big:v + self.big + 1, u - self.big:u + self.big + 1].copy()
        core = self.source[self.big - self.r:self.big + self.r + 1, self.big - self.r:self.big + self.r + 1]
        self.std0 = float(core.std())
        if self.std0 < K7_MIN_STD_GREY:
            self.reason = "textureless_patch"
            return
        self.angle = 0.0
        self.uv = np.array([float(u), float(v)])
        self.peak = None

    def _template(self, angle_deg):
        return _rotated_core(self.source, self.big, self.r, angle_deg)

    def rotation_drop(self, gray0) -> float:
        """K7 rotation observability: 1 - the best NCC of the start template rotated by +-K7_FLOOR_DEG, re-matched in
        the start image with +-K7_SEARCH_PX translation (a rotation that a translation can imitate does not count)."""
        peaks = [_best_ncc(gray0, self._template(sign * K7_FLOOR_DEG), self.uv, K7_SEARCH_PX) for sign in (-1.0, 1.0)]
        peaks = [p for p in peaks if p is not None]
        return 1.0 - max(peaks) if peaks else 0.0

    def measure(self, gray):
        """(angle_deg, uv, peak); angle / uv None when no candidate reaches K7_MIN_PEAK."""
        image = np.asarray(gray, dtype=np.float32)
        h, w = image.shape
        s = K7_SEARCH_PX
        u, v = int(round(self.uv[0])), int(round(self.uv[1]))
        x0, x1 = max(0, u - s - self.r), min(w, u + s + self.r + 1)
        y0, y1 = max(0, v - s - self.r), min(h, v + s + self.r + 1)
        crop = image[y0:y1, x0:x1]
        if crop.shape[0] < 2 * self.r + 1 or crop.shape[1] < 2 * self.r + 1:
            return None, None, None
        n = int(round(K7_ANGLE_SPAN_DEG / K7_ANGLE_STEP_DEG))
        angles = self.angle + K7_ANGLE_STEP_DEG * np.arange(-n, n + 1)
        peaks, locs = [], []
        for angle in angles:
            score = cv2.matchTemplate(crop, self._template(angle), cv2.TM_CCOEFF_NORMED)
            _lo, peak, _lo_loc, loc = cv2.minMaxLoc(score)
            peaks.append(float(peak))
            locs.append((loc, score))
        best = int(np.argmax(peaks))
        peak = peaks[best]
        self.peak = peak
        if peak < K7_MIN_PEAK:
            return None, None, peak
        angle = float(angles[best])
        if 0 < best < len(angles) - 1:
            a, b, c = peaks[best - 1], peaks[best], peaks[best + 1]
            curvature = a - 2.0 * b + c
            if curvature < -1e-9:
                angle += K7_ANGLE_STEP_DEG * float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))
        (lx, ly), score = locs[best]
        sub = np.zeros(2)
        if 0 < lx < score.shape[1] - 1:
            a, b, c = score[ly, lx - 1], score[ly, lx], score[ly, lx + 1]
            curvature = a - 2.0 * b + c
            sub[0] = float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5)) if curvature < -1e-9 else 0.0
        if 0 < ly < score.shape[0] - 1:
            a, b, c = score[ly - 1, lx], score[ly, lx], score[ly + 1, lx]
            curvature = a - 2.0 * b + c
            sub[1] = float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5)) if curvature < -1e-9 else 0.0
        self.angle = angle
        self.uv = np.array([x0 + lx + self.r + sub[0], y0 + ly + self.r + sub[1]])
        return angle, self.uv.copy(), peak


def _rotated_core(source, big, r, angle_deg):
    """The (2r+1)^2 core of ``source`` (a (2 big + 1)^2 patch) rotated by ``angle_deg`` about its centre (OpenCV
    sense)."""
    c = float(big)
    M = cv2.getRotationMatrix2D((c, c), float(angle_deg), 1.0)
    rotated = cv2.warpAffine(source, M, source.shape[::-1], flags=cv2.INTER_LINEAR)
    return rotated[big - r:big + r + 1, big - r:big + r + 1]


def _source_at(gray, uv, big):
    """The (2 big + 1)^2 float32 patch of ``gray`` centred at ``uv`` (None when it leaves the image)."""
    image = np.asarray(gray)
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = image.shape[:2]
    if not (big <= u < w - big and big <= v < h - big):
        return None
    return image[v - big:v + big + 1, u - big:u + big + 1].astype(np.float32).copy()


def _best_ncc(gray, template, uv, search):
    """The best TM_CCOEFF_NORMED score of ``template`` in ``gray`` within +-``search`` px of ``uv`` (None when the
    window is smaller than the template)."""
    image = np.asarray(gray, dtype=np.float32)
    r = (template.shape[0] - 1) // 2
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = image.shape[:2]
    x0, x1 = max(0, u - search - r), min(w, u + search + r + 1)
    y0, y1 = max(0, v - search - r), min(h, v + search + r + 1)
    crop = image[y0:y1, x0:x1]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return None
    return float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())


class _PatchRotation:
    """Shared measurement core of PartRotationObserver and SpringBackObserver: the apparent rotation (deg) of a
    part patch in one camera, by rotated template (textured patch) or by the structure axis (``axis_uv`` given).
    A textured template whose rotation a translation can imitate (rotation drop < K7_MIN_ROTATION_DROP, e.g. a
    smooth cap shaded about the rotation axis) is ``rotation_degenerate``: the axis mode is used when an axis is
    given, else the patch is unmeasurable."""

    def __init__(self, gray0, patch, axis_uv, axis_fn):
        self.mode = None
        self.reason = None
        self.drop = None
        self.axis_fn = axis_fn
        if not isinstance(patch, Mapping) or "unmeasurable" in patch or patch.get("uv") is None:
            self.reason = str(patch.get("unmeasurable")) if isinstance(patch, Mapping) and \
                patch.get("unmeasurable") else "no_patch"
            return
        uv = np.asarray(patch["uv"], dtype=float).reshape(-1)[:2]
        if uv.shape != (2,) or not np.isfinite(uv).all():
            self.reason = "no_patch"
            return
        self.uv0 = uv
        radius = patch.get("radius_px")
        tracker = _RotationTracker(gray0, uv, radius_px=K7_PATCH_RADIUS_PX if radius is None else int(radius))
        if tracker.reason is None:
            self.drop = tracker.rotation_drop(np.asarray(gray0, dtype=np.float32))
            if self.drop < K7_MIN_ROTATION_DROP:
                tracker.reason = "rotation_degenerate"
        if tracker.reason is None:
            self.mode, self.tracker = "template", tracker
            return
        if axis_uv is not None:
            axis = np.asarray(axis_uv, dtype=float).reshape(-1)[:2]
            norm = float(np.linalg.norm(axis))
            if axis.shape == (2,) and norm > 1e-9:
                self.mode, self.axis0 = "axis", axis / norm
                return
        self.reason = tracker.reason

    def measure(self, gray):
        """(angle_deg | None, uv | None, quality)"""
        if self.mode == "template":
            angle, uv, peak = self.tracker.measure(gray)
            return angle, uv, peak
        axis, coherence, _rms = (self.axis_fn or _default_axis)(np.asarray(gray, dtype=np.float32), self.uv0)
        if axis is None or coherence is None or float(coherence) < K7_MIN_COHERENCE:
            return None, None, None if coherence is None else float(coherence)
        return _axis_angle_deg(np.asarray(axis, dtype=float), self.axis0), self.uv0.copy(), float(coherence)


# ======================================================================================= PartRotationObserver
class PartRotationObserver:
    """K7 part observable (spec §4.9 Turn step 5): the apparent rotation of the gripped part in the wrist image
    (which is rigid with the hand) against its start appearance; ``patch`` = {"uv", optional "radius_px"} on the
    part (a ``PartRecord.rigid_patch``; {"unmeasurable": reason} or None = no patch); ``axis_uv`` = the part's unit
    image axis at the start (structure-axis mode, used when the patch is textureless or rotation-degenerate).

    Motion bouts: the observer's start, and every step whose hand yaw is < K7_STILL_DEG_PER_STEP (the hand paused;
    a free part's lag behind the hand relaxes and re-develops at the next onset), begin a bout; while the hand is
    still the observer is pending (``hand_still``). Within a bout, a corotation span may start only after
    K7_ONSET_DEG of hand yaw (the free part's compliance lag builds up over the first degrees of every motion).

    Judgement (template mode): over the trailing hand-yaw span of K7_SPAN_FACTOR x thr (thr = max(K7_FLOOR_DEG,
    K7_SIGMA_K sigma_theta), sigma_theta from the bout's post-onset samples) the tracked apparent rotation changed by
    < thr -> class ``corotating``, else ``part_not_corotating``; the class counts only when the span hypothesis test
    agrees: the span-start patch matched in the current frame at 0 rotation (corotating) and at the span's hand yaw
    (blocked; a world-fixed part turns in the camera with the hand's yaw) must differ by >= K7_HYPOTHESIS_MARGIN x
    the template's own contrast between the two angles (measured in the span-start frame), else the frame is
    ambiguous (pending). A verdict needs the same class on K7_CONFIRM_FRAMES consecutive judged frames: met
    ``corotating`` | violated ``part_not_corotating``. Structure-axis mode (no template) applies the bout, onset
    and persistence rules to the axis angle. Failed ``unmeasurable`` (flag ``sense_by_wedge_only``): no patch,
    textureless without an axis, ``rotation_degenerate`` (a rotation by K7_FLOOR_DEG that a translation imitates,
    e.g. a smooth cap shaded about the rotation axis), out of view, or lost for K7_LOST_FRAMES frames. Not latched:
    a part that stops (its end stop) turns violated.

    Measured on real task-2 wrist frames (BO fix; steady K2 twists, free / paused / blocked, 256 on-part patches): 0
    wrong part_not_corotating on the free twists; on the blocked twist 34 of 529 judged frames read corotating, all
    on patches beside the finger pads, on the elongated ridge's top end or at the cap silhouette (mixed camera-rigid or
    rotation-invariant content). A rigid patch should be chosen on the part's own texture away from those."""

    name = "part_rotation"

    def __init__(self, *, robot: RobotModel, patch: dict | None, axis_uv: np.ndarray | None,
                 axis_fn: Callable | None = None):
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        self.robot = robot
        self.patch = patch
        self.axis_uv = axis_uv
        self.axis_fn = axis_fn
        self.started = False

    def start(self, view) -> None:
        view = _view(view)
        gray = view.wrist_gray_upright
        self.core = _PatchRotation(gray, self.patch, self.axis_uv, self.axis_fn)
        self.hand_deg = [0.0]
        self.rot_deg = [0.0]
        self.theta_hand = 0.0
        self.lost = 0
        self.count = 0
        self.bout_start = 0
        self.raw: list[str | None] = []
        self.samples: dict[int, dict] = {}
        self.unmeasurable = self.core.reason if self.core.mode is None else None
        if self.core.mode == "template":
            self._keep_sample(0, gray, self.core.uv0)
        self.started = True

    # ------------------------------------------------------------------------------------------- samples
    def _keep_sample(self, index, gray, uv) -> None:
        """Span-start material for the hypothesis test: the rotation source patch and the surrounding search crop."""
        tracker = self.core.tracker
        image = np.asarray(gray, dtype=np.float32)
        source = _source_at(image, uv, tracker.big)
        reach = K7_HYPOTHESIS_SEARCH_PX + tracker.r
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        h, w = image.shape[:2]
        x0, y0 = max(0, u - reach), max(0, v - reach)
        crop = image[y0:min(h, v + reach + 1), x0:min(w, u + reach + 1)].copy()
        self.samples[index] = {"uv": np.array([float(uv[0]), float(uv[1])]), "source": source, "crop": crop,
                               "crop_uv": np.array([float(uv[0]) - x0, float(uv[1]) - y0])}
        for old in [k for k in self.samples if k < self.bout_start]:
            del self.samples[old]

    def _onset_index(self):
        """The first sample of the current bout with >= K7_ONSET_DEG of hand yaw since the bout start."""
        h0 = self.hand_deg[self.bout_start]
        for index in range(self.bout_start, len(self.hand_deg)):
            if abs(self.hand_deg[index] - h0) >= K7_ONSET_DEG:
                return index
        return None

    def threshold(self) -> Threshold:
        """thr = max(K7_FLOOR_DEG, K7_SIGMA_K sigma_theta), sigma_theta over the current bout's post-onset samples."""
        onset = self._onset_index()
        sigma = 0.0 if onset is None else _jitter_sigma(self.rot_deg[onset:])
        return th_max("K7_rotation_deg", Term("DD", K7_FLOOR_DEG, "K7_FLOOR_DEG"),
                      Term("ONLINE", K7_SIGMA_K * sigma, "K7_SIGMA_K*sigma_theta"), unit="deg")

    def _hypothesis(self, j, gray, span_deg) -> dict:
        """The span hypothesis test (class docstring): corotating vs blocked for the span starting at sample j."""
        sample = self.samples.get(j)
        tracker = self.core.tracker
        if sample is None or sample["source"] is None:
            return {"class": None, "reason": "span_start_out_of_view"}
        still = _rotated_core(sample["source"], tracker.big, tracker.r, 0.0)
        turned = _rotated_core(sample["source"], tracker.big, tracker.r, span_deg)
        search = K7_HYPOTHESIS_SEARCH_PX
        co = _best_ncc(gray, still, sample["uv"], search)
        bl = _best_ncc(gray, turned, sample["uv"], search)
        own = _best_ncc(sample["crop"], turned, sample["crop_uv"], search)
        if co is None or bl is None or own is None:
            return {"class": None, "reason": "search_window_out_of_view"}
        contrast = 1.0 - own
        need = K7_HYPOTHESIS_MARGIN * contrast
        lead = co - bl
        cls = None
        if contrast >= K7_MIN_ROTATION_DROP and lead >= need:
            cls = "corotating"
        elif contrast >= K7_MIN_ROTATION_DROP and -lead >= need:
            cls = "part_not_corotating"
        return {"class": cls, "reason": None if cls else "ambiguous", "ncc_corotating": co, "ncc_blocked": bl,
                "contrast": contrast, "lead": lead, "need": need, "span_deg": span_deg}

    # ---------------------------------------------------------------------------------------------- update
    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("PartRotationObserver.start(view) must be called first")
        before, after = _view(before, "before"), _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        dyaw = self.robot.yaw_about_approach(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2])
        self.theta_hand += dyaw
        if self.unmeasurable is not None:
            return self._out(VERDICT_FAILED, "unmeasurable", step, None)
        gray = after.wrist_gray_upright
        angle, uv, quality = self.core.measure(gray)
        if angle is None:
            self.lost += 1
            self.raw.append(None)
            if self.lost >= K7_LOST_FRAMES:
                self.unmeasurable = "lost"
                return self._out(VERDICT_FAILED, "unmeasurable", step, None)
            return self._out(VERDICT_PENDING, None, step, None, quality=quality)
        self.lost = 0
        self.hand_deg.append(math.degrees(self.theta_hand))
        self.rot_deg.append(float(angle))
        n = len(self.hand_deg) - 1
        still = abs(math.degrees(dyaw)) < K7_STILL_DEG_PER_STEP
        if still:
            self.bout_start = n
        if self.core.mode == "template":
            self._keep_sample(n, gray, uv)
        bout = {"bout_start_index": self.bout_start, "hand_still": still, "onset_index": None}
        if still:
            self.raw.append(None)
            return self._out(VERDICT_PENDING, None, step, None, quality=quality, uv=uv, extra=bout)
        onset = self._onset_index()
        bout["onset_index"] = onset
        thr = self.threshold()
        span_needed = K7_SPAN_FACTOR * thr.value
        hand_now = self.hand_deg[n]
        j = None
        if onset is not None:
            for index in range(n - 1, onset - 1, -1):
                if abs(hand_now - self.hand_deg[index]) >= span_needed:
                    j = index
                    break
        if j is None:
            self.raw.append(None)
            return self._out(VERDICT_PENDING, None, step, thr, quality=quality, uv=uv, extra=bout)
        change = self.rot_deg[n] - self.rot_deg[j]
        span = hand_now - self.hand_deg[j]
        cls = "corotating" if abs(change) < thr.value else "part_not_corotating"
        record = {**bout, "span_hand_deg": span, "span_rotation_deg": change, "span_start_index": j,
                  "raw_class": cls, "hypothesis": None}
        if self.core.mode == "template":
            hypothesis = self._hypothesis(j, gray, span)
            record["hypothesis"] = hypothesis
            if hypothesis["class"] != cls:
                record["raw_class"] = None
                self.raw.append(None)
                return self._out(VERDICT_PENDING, None, step, thr, quality=quality, uv=uv, extra=record)
        self.raw.append(cls)
        confirmed = len(self.raw) >= K7_CONFIRM_FRAMES and all(c == cls for c in self.raw[-K7_CONFIRM_FRAMES:])
        record["confirmed"] = confirmed
        if not confirmed:
            return self._out(VERDICT_PENDING, None, step, thr, quality=quality, uv=uv, extra=record)
        if cls == "corotating":
            return self._out(VERDICT_MET, "corotating", step, thr, quality=quality, uv=uv, extra=record)
        return self._out(VERDICT_VIOLATED, "part_not_corotating", step, thr, quality=quality, uv=uv, extra=record)

    def _out(self, verdict, reason, step, thr, *, quality=None, uv=None, extra=None) -> ObsReport:
        onset = self._onset_index() if self.started and self.hand_deg else None
        evidence = {"mode": self.core.mode, "hand_yaw_deg": math.degrees(self.theta_hand),
                    "rotation_deg": self.rot_deg[-1] if self.rot_deg else None,
                    "rotation_series_deg": [round(v, 3) for v in self.rot_deg],
                    "hand_series_deg": [round(v, 3) for v in self.hand_deg], "quality": quality,
                    "uv": None if uv is None else [float(uv[0]), float(uv[1])], "unmeasurable": self.unmeasurable,
                    "rotation_drop": self.core.drop,
                    "sigma_theta_deg": 0.0 if onset is None else _jitter_sigma(self.rot_deg[onset:])}
        if self.unmeasurable is not None:
            evidence["sense_by_wedge_only"] = True
        evidence.update(extra or {})
        progress = None
        if thr is not None and extra and extra.get("span_rotation_deg") is not None:
            progress = abs(extra["span_rotation_deg"]) / thr.value
        ths = [th_value("K7_rotation_drop", K7_MIN_ROTATION_DROP, "DD", "K7_MIN_ROTATION_DROP (start template)")]
        if thr is not None:
            ths.insert(0, thr)
        hypothesis = (extra or {}).get("hypothesis")
        if hypothesis and hypothesis.get("need") is not None:
            ths.append(th_value("K7_hypothesis_lead", hypothesis["need"], "ONLINE",
                                "K7_HYPOTHESIS_MARGIN x the span-start template's measured contrast",
                                contrast=hypothesis["contrast"]))
        return _report(self.name, verdict, reason, evidence, ths, step, progress)


# ======================================================================================= effect: noise and watch
def visible_share(region, hidden=None) -> dict:
    """{"region_px", "visible_px", "share"} of a region mask in one camera image (``hidden`` = hand / occluder)."""
    reg = _mask(region, "region")
    visible = reg if hidden is None else reg & ~_mask(hidden, "hidden", reg.shape)
    n = int(reg.sum())
    return {"region_px": n, "visible_px": int(visible.sum()), "share": float(visible.sum()) / n if n else 0.0}


class MotionNoiseMeter:
    """The false-change rate of a region while the robot MOVES near it (Appendix L 8): every transition with a
    visible hand motion compares the region (outside the dilated hand motion mask, and outside the projected hand
    footprint when ``static_hand_mask_fn`` is given) against the frame at ``start``; the rate is the mean changed
    share over those moving frames. Register it (preempt=False) during the approach / align / descent before the
    action; ``result()`` gives the ``false_rate`` EffectWatch, ``validate_region`` and RegionStaysCheck take."""

    name = "motion_noise"

    def __init__(self, *, region: np.ndarray, sigma_px: float, camera: str = "external", statistic: str = "chroma",
                 hand_mask_fn: Callable | None = None, static_hand_mask_fn: Callable | None = None):
        self.region = _mask(region, "region")
        if not self.region.any():
            raise ValueError("region must contain at least one pixel")
        if statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}")
        self.camera = camera
        self.statistic = statistic
        self.pixel_th = o1_pixel_threshold(sigma_px)
        self.hand_mask_fn = hand_mask_fn
        self.static_hand_mask_fn = static_hand_mask_fn
        self.started = False

    def start(self, view) -> None:
        view = _view(view)
        self.ref_lab = _lab255(_camera_rgb(view, self.camera))
        if self.ref_lab.shape[:2] != self.region.shape:
            raise ValueError("region mask and camera image differ in shape")
        self.ref_hidden = _static_mask(self.static_hand_mask_fn, view, self.region.shape)
        self.shares: list[float] = []
        self.count = 0
        self.started = True

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("MotionNoiseMeter.start(view) must be called first")
        before, after = _view(before, "before"), _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        rgb0, rgb1 = _camera_rgb(before, self.camera), _camera_rgb(after, self.camera)
        raw = (self.hand_mask_fn or _default_hand_mask)(rgb0, rgb1)
        if raw is not None:
            hand = _dilate(_mask(raw, "hand motion mask", self.region.shape), O1_HAND_DILATE_PX)
            hand |= _static_mask(self.static_hand_mask_fn, after, self.region.shape) | self.ref_hidden
            ok = self.region & ~hand
            if ok.any():
                change = change_magnitude(_lab255(rgb1), self.ref_lab, self.statistic)
                self.shares.append(float((change[ok] > self.pixel_th.value).mean()))
        return _report(self.name, VERDICT_PENDING, None, self._summary(), (self.pixel_th,), step,
                       float(len(self.shares)))

    def _summary(self) -> dict:
        shares = np.asarray(self.shares, dtype=float)
        return {"camera": self.camera, "statistic": self.statistic, "moving_frames": int(shares.size),
                "frames": self.count, "false_rate": float(shares.mean()) if shares.size else None,
                "max_share": float(shares.max()) if shares.size else None,
                "p95_share": float(np.quantile(shares, NOISE_REPORT_QUANTILE)) if shares.size else None}

    def result(self) -> dict:
        """{"false_rate" (mean share over the moving frames; 0.0 with none, flagged), "moving_frames", "frames",
        "max_share", "p95_share", "camera", "statistic", "no_motion", "thresholds"}."""
        summary = self._summary()
        summary["no_motion"] = summary["moving_frames"] == 0
        if summary["false_rate"] is None:
            summary["false_rate"] = 0.0
        summary["thresholds"] = thresholds_dicts([self.pixel_th])
        return summary


class EffectWatch:
    """O1 / O7 / O8 (spec §4.9 Turn step 6; Appendix L 8): the effect region's CHROMA change against the
    pre-action baseline (``start``, before the first TwistSegment [C-27]).

    ``region`` = the refined region mask of the pointed effect part in ``camera`` (never chosen by name);
    ``false_rate`` = the motion false-change rate (``MotionNoiseMeter``); ``sigma_px`` = the static pixel noise.
    Per frame: hand mask = ``hand_mask_fn`` (default geometry ``hand_motion_mask``) of (before, after), dilated by
    O1_HAND_DILATE_PX (plus ``static_hand_mask_fn``, the projected hand footprint, when given); a pixel changed if
    |d(a*, b*)| > max(O1_DIFF_FLOOR, O1_PX_SIGMA_K sigma_px); the changed share of the valid, unmasked pixels >=
    max(O1_FRACTION_FLOOR, O1_FALSE_RATE_K false_rate) for O1_FRAMES consecutive frames, with theta_part >= O8 at
    the confirming frame -> met ``effect_onset``, ``onset_step`` / ``theta_e_rad`` taken at the FIRST frame of the
    run [C-28] (latched). A frame with > EFFECT_OCCLUDED_FRACTION of the valid mask hand-masked is skipped (neither
    counts nor breaks the run; flag ``effect_occluded``). ``context`` {"theta_part_rad": float | None}: None (or
    negative) counts as 0 (the sense phase, before the free classification). ``camera`` "wrist" (reserved [C-27])
    needs ``reference_fn(view) -> (region_mask_now, baseline_lab_now)``, the baseline warped into the current
    frame."""

    name = "effect"

    def __init__(self, *, region: np.ndarray, false_rate: float, sigma_px: float, camera: str = "external",
                 hand_mask_fn: Callable | None = None, static_hand_mask_fn: Callable | None = None,
                 reference_fn: Callable | None = None):
        self.region = _mask(region, "region")
        if not self.region.any():
            raise ValueError("region must contain at least one pixel")
        if camera not in ("external", "wrist"):
            raise ValueError("camera must be 'external' or 'wrist'")
        if camera == "wrist" and reference_fn is None:
            raise ValueError("a wrist EffectWatch needs reference_fn (a warped baseline; reserved [C-27])")
        self.camera = camera
        self.false_rate = _nonnegative(false_rate, "false_rate")
        self.sigma_px = _nonnegative(sigma_px, "sigma_px")
        self.pixel_th = o1_pixel_threshold(self.sigma_px)
        self.fraction_th = o1_fraction_threshold(self.false_rate)
        self.o8_th = th_value("O8_min_rotation_rad", O8_MIN_ROTATION_RAD, "DD", "O8_MIN_ROTATION_RAD")
        self.hand_mask_fn = hand_mask_fn
        self.static_hand_mask_fn = static_hand_mask_fn
        self.reference_fn = reference_fn
        self.valid_mask = self.region.copy()
        self.phase = "all"
        self.validity = None
        self.started = False

    # ------------------------------------------------------------------------------------------------ O7
    @staticmethod
    def validate_region(region: np.ndarray, *, exclude: Sequence[np.ndarray], swept: np.ndarray | None,
                        false_rate: float) -> dict:
        """O7 [C-26], Appendix L 2 / 8: remainder = region - the exclude masks (obj / grip regions) - ``swept`` (the
        dilated hand-swept mask); valid iff the remainder >= O7_MIN_REMAINING_FRACTION of the region AND >=
        O7_MIN_REMAINING_PX pixels AND the (motion) false rate < O7_FALSE_RATE_MAX."""
        reg = _mask(region, "region")
        remainder = reg.copy()
        for index, mask in enumerate(exclude or ()):
            remainder &= ~_mask(mask, f"exclude[{index}]", reg.shape)
        if swept is not None:
            remainder &= ~_mask(swept, "swept", reg.shape)
        rate = _nonnegative(false_rate, "false_rate")
        region_px, remaining_px = int(reg.sum()), int(remainder.sum())
        fraction = remaining_px / region_px if region_px else 0.0
        ths = [th_value("O7_remaining_fraction", O7_MIN_REMAINING_FRACTION, "DD", "O7_MIN_REMAINING_FRACTION"),
               th_value("O7_remaining_px", float(O7_MIN_REMAINING_PX), "DD", "O7_MIN_REMAINING_PX"),
               th_value("O7_false_rate", O7_FALSE_RATE_MAX, "DD", "O7_FALSE_RATE_MAX (motion false-change rate)")]
        valid = (fraction >= O7_MIN_REMAINING_FRACTION and remaining_px >= O7_MIN_REMAINING_PX
                 and rate < O7_FALSE_RATE_MAX)
        return {"valid": bool(valid), "reason": None if valid else "effect_region_invalid", "region_px": region_px,
                "remaining_px": remaining_px, "remaining_fraction": float(fraction),
                "overlap_px": region_px - remaining_px, "false_rate": rate, "thresholds": ths,
                "valid_mask": remainder}

    @staticmethod
    def select_camera(visible: Mapping[str, Mapping | None]) -> dict:
        """Appendix L 8 camera choice: the camera that keeps more of the region visible, among those keeping >=
        O7_MIN_REMAINING_FRACTION of it and >= O7_MIN_REMAINING_PX pixels (ties: the external camera, which does
        not yaw with the hand). ``visible[camera]`` = ``visible_share(...)`` or None (not projected)."""
        rows, eligible = {}, []
        for camera, row in visible.items():
            if row is None:
                rows[camera] = None
                continue
            n, seen = int(row["region_px"]), int(row["visible_px"])
            share = seen / n if n else 0.0
            rows[camera] = {"region_px": n, "visible_px": seen, "share": share}
            if share >= O7_MIN_REMAINING_FRACTION and seen >= O7_MIN_REMAINING_PX:
                eligible.append(camera)
        chosen = max(eligible, key=lambda c: (rows[c]["share"], c == "external")) if eligible else None
        return {"camera": chosen, "visible": rows, "reason": None if chosen else "effect_region_invalid",
                "thresholds": [th_value("O7_remaining_fraction", O7_MIN_REMAINING_FRACTION, "DD",
                                        "O7_MIN_REMAINING_FRACTION (visible share)"),
                               th_value("O7_remaining_px", float(O7_MIN_REMAINING_PX), "DD", "O7_MIN_REMAINING_PX")]}

    def set_valid_mask(self, valid_mask: np.ndarray | None, *, phase: str, validity: Mapping | None = None) -> None:
        """Per sense segment (``sense``) or after the free classification (``free``) [C-26]: O1 runs on this mask
        only; None = not watched in this phase (frames are skipped and the run restarts). ``validity`` (the
        ``validate_region`` result) is carried into the reports: its O7 thresholds and {phase, remaining_fraction,
        remaining_px, valid} for the ``effect`` trace event."""
        if phase not in EFFECT_PHASES[1:]:
            raise ValueError("phase must be 'sense' or 'free'")
        self.valid_mask = None if valid_mask is None else _mask(valid_mask, "valid_mask", self.region.shape) \
            & self.region
        self.phase = phase
        if validity is None:
            self.validity = None
        else:
            self.validity = {"phase": phase, "valid": bool(validity.get("valid")),
                             "remaining_fraction": validity.get("remaining_fraction"),
                             "remaining_px": validity.get("remaining_px"), "reason": validity.get("reason"),
                             "thresholds": tuple(t for t in validity.get("thresholds", ()) if isinstance(t, Threshold))}

    # --------------------------------------------------------------------------------------------- watch
    def start(self, view) -> None:
        view = _view(view)
        rgb = _camera_rgb(view, self.camera)
        self.baseline_rgb = np.asarray(rgb).copy()
        self.baseline_lab = _lab255(rgb)
        if self.camera == "external" and self.baseline_lab.shape[:2] != self.region.shape:
            raise ValueError("region mask and camera image differ in shape")
        self.count = 0
        self.run: list[dict] = []
        self.onset = None
        self.history: list[dict] = []
        self.occluded_frames = 0
        self.started = True

    def _frame(self, view, hand) -> dict:
        """Share of the valid, unmasked region pixels whose chroma changed against the baseline."""
        if self.camera == "external":
            valid, base = self.valid_mask, self.baseline_lab
        else:
            region_now, base = self.reference_fn(view)
            valid = _mask(region_now, "reference region", hand.shape)
        lab = _lab255(_camera_rgb(view, self.camera))
        valid_px = int(valid.sum())
        masked_px = int((valid & hand).sum())
        out = {"valid_px": valid_px, "masked_px": masked_px,
               "masked_fraction": masked_px / valid_px if valid_px else None, "share": None, "occluded": False}
        if valid_px == 0:
            return out
        if masked_px > EFFECT_OCCLUDED_FRACTION * valid_px:
            out["occluded"] = True
            return out
        ok = valid & ~hand
        change = change_magnitude(lab, base, "chroma")
        out["share"] = float((change[ok] > self.pixel_th.value).mean())
        return out

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("EffectWatch.start(view) must be called first")
        before, after = _view(before, "before"), _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        theta = (context or {}).get("theta_part_rad") if isinstance(context, Mapping) else None
        theta_known = theta is not None
        theta_val = max(0.0, _finite(theta, "theta_part_rad")) if theta_known else 0.0
        if self.camera == "external" and self.valid_mask is None:
            self.run = []
            return self._out(VERDICT_MET if self.onset else VERDICT_PENDING, "effect_onset" if self.onset else None,
                             step, {"watched": False, "share": None})
        rgb0, rgb1 = _camera_rgb(before, self.camera), _camera_rgb(after, self.camera)
        shape = rgb1.shape[:2]
        hand = _hand_mask(self.hand_mask_fn, rgb0, rgb1, shape) | _static_mask(self.static_hand_mask_fn, after, shape)
        frame = self._frame(after, hand)
        frame.update({"k": self.count, "step": step, "theta_part_rad": theta_val, "theta_known": theta_known})
        self.history.append({key: frame[key] for key in ("k", "share", "occluded", "theta_part_rad")})
        if frame["occluded"]:
            self.occluded_frames += 1
        elif frame["share"] is None or frame["share"] < self.fraction_th.value:
            self.run = []
        else:
            self.run.append(frame)
            if self.onset is None and len(self.run) >= O1_FRAMES and theta_val >= O8_MIN_ROTATION_RAD:
                first = self.run[0]
                self.onset = {"onset_step": first["step"], "onset_k": first["k"],
                              "theta_e_rad": first["theta_part_rad"], "theta_known_at_onset": first["theta_known"],
                              "onset_share": first["share"], "confirm_step": step, "confirm_k": self.count,
                              "theta_confirm_rad": theta_val, "phase": self.phase}
        if self.onset is not None:
            return self._out(VERDICT_MET, "effect_onset", step, frame)
        return self._out(VERDICT_PENDING, "effect_occluded" if frame["occluded"] else None, step, frame)

    def persists(self, view) -> ObsReport:
        """Stays (after release): met while the region is still changed against the pre-action baseline; violated
        ``kept_relation_lost`` when it is not; pending when > EFFECT_OCCLUDED_FRACTION of it is masked by the
        projected hand footprint (``static_hand_mask_fn``) or the region is not watched."""
        if not self.started:
            raise RuntimeError("EffectWatch.start(view) must be called first")
        view = _view(view)
        if self.camera == "external" and self.valid_mask is None:
            return self._out(VERDICT_PENDING, None, self.count, {"watched": False, "share": None}, stays=True)
        shape = _camera_rgb(view, self.camera).shape[:2]
        frame = self._frame(view, _static_mask(self.static_hand_mask_fn, view, shape))
        if frame["occluded"] or frame["share"] is None:
            return self._out(VERDICT_PENDING, "effect_occluded" if frame["occluded"] else None, self.count, frame,
                             stays=True)
        if frame["share"] >= self.fraction_th.value:
            return self._out(VERDICT_MET, "effect_persists", self.count, frame, stays=True)
        return self._out(VERDICT_VIOLATED, "kept_relation_lost", self.count, frame, stays=True)

    def _out(self, verdict, reason, step, frame, stays=False) -> ObsReport:
        evidence = {"camera": self.camera, "statistic": "chroma", "phase": self.phase, "fraction": frame.get("share"),
                    "threshold": self.fraction_th.value, "pixel_threshold": self.pixel_th.value,
                    "false_rate": self.false_rate, "run": len(self.run), "occluded": bool(frame.get("occluded")),
                    "effect_occluded": bool(frame.get("occluded")), "masked_fraction": frame.get("masked_fraction"),
                    "valid_px": frame.get("valid_px"), "watched": frame.get("watched", True),
                    "occluded_frames": self.occluded_frames, "stays": stays,
                    "onset_step": None, "theta_e_rad": None}
        if self.onset is not None:
            evidence.update(self.onset)
        evidence["o7"] = None if self.validity is None else {k: v for k, v in self.validity.items()
                                                              if k != "thresholds"}
        progress = None if frame.get("share") is None else frame["share"] / self.fraction_th.value
        ths = (self.pixel_th, self.fraction_th, self.o8_th)
        if self.validity is not None:
            ths = ths + tuple(self.validity.get("thresholds", ()))
        return _report(self.name, verdict, reason, evidence, ths, step, progress)


# ============================================================================================ RegionStaysCheck
class RegionStaysCheck:
    """[C-48] external-camera stays check of a region against a baseline (supported_on, turned / effect,
    hold_true). Hand mask = ``projector_mask(view)`` (the projected hand footprint at the CURRENT pose; no motion is
    needed). > EFFECT_OCCLUDED_FRACTION of the region masked -> pending; else the O1 per-pixel test and share
    threshold (max(O1_FRACTION_FLOOR, O1_FALSE_RATE_K false_rate)) against the baseline: ``kind`` "unchanged":
    changed -> violated ``kept_relation_lost``, else met; ``kind`` "effect" (inverted): met while the region is
    STILL changed against its pre-action baseline, violated when it is not. ``statistic``: None = chroma for the
    effect kind (Appendix L 8) and |dLab| for the unchanged kind (a displaced object changes mostly luminance)."""

    name = "region_stays"

    def __init__(self, *, region: np.ndarray, baseline_rgb: np.ndarray, projector_mask: Callable[[StepView], np.ndarray],
                 sigma_px: float, false_rate: float, kind: str = "unchanged", statistic: str | None = None):
        self.region = _mask(region, "region")
        if not self.region.any():
            raise ValueError("region must contain at least one pixel")
        if kind not in STAYS_KINDS:
            raise ValueError(f"kind must be one of {STAYS_KINDS}")
        self.kind = kind
        self.statistic = statistic or ("chroma" if kind == "effect" else "lab")
        if self.statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}")
        self.baseline_lab = _lab255(baseline_rgb)
        if self.baseline_lab.shape[:2] != self.region.shape:
            raise ValueError("region mask and baseline image differ in shape")
        if not callable(projector_mask):
            raise ValueError("projector_mask must be callable")
        self.projector_mask = projector_mask
        self.pixel_th = o1_pixel_threshold(sigma_px)
        self.fraction_th = o1_fraction_threshold(false_rate)
        self.checks = 0

    def __call__(self, view: StepView) -> ObsReport:
        view = _view(view)
        self.checks += 1
        rgb = view.external_rgb_upright
        hidden = _static_mask(self.projector_mask, view, self.region.shape)
        region_px = int(self.region.sum())
        masked = int((self.region & hidden).sum())
        evidence = {"kind": self.kind, "statistic": self.statistic, "region_px": region_px,
                    "masked_fraction": masked / region_px, "fraction": None, "threshold": self.fraction_th.value,
                    "pixel_threshold": self.pixel_th.value}
        ths = (self.pixel_th, self.fraction_th)
        if masked > EFFECT_OCCLUDED_FRACTION * region_px:
            evidence["effect_occluded"] = True
            return _report(self.name, VERDICT_PENDING, "occluded", evidence, ths, self.checks)
        ok = self.region & ~hidden
        change = change_magnitude(_lab255(rgb), self.baseline_lab, self.statistic)
        share = float((change[ok] > self.pixel_th.value).mean())
        evidence["fraction"] = share
        changed = share >= self.fraction_th.value
        holds = changed if self.kind == "effect" else not changed
        progress = share / self.fraction_th.value
        if holds:
            return _report(self.name, VERDICT_MET, None, evidence, ths, self.checks, progress)
        return _report(self.name, VERDICT_VIOLATED, "kept_relation_lost", evidence, ths, self.checks, progress)


# ============================================================================================ CoMotionObserver
def _rotation_angle_deg(R) -> float:
    """The angle (deg) of a rotation matrix."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(np.asarray(R, dtype=float))) - 1.0) / 2.0))))


def _rotvec_of(R) -> np.ndarray:
    """The rotation vector (rad) of a rotation matrix below pi (axis from its skew part)."""
    R = np.asarray(R, dtype=float)
    angle = math.radians(_rotation_angle_deg(R))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = float(np.linalg.norm(w))
    if norm < 1e-12 or angle < 1e-12:
        return np.zeros(3)
    return w / norm * angle


def _external_ray(external) -> np.ndarray | None:
    """The external camera's unit viewing ray (``ray_dir``, else the null vector of the affine J), or None."""
    ray = getattr(external, "ray_dir", None)
    if ray is None:
        J = getattr(external, "J", None)
        if J is None:
            return None
        J = np.asarray(J, dtype=float).reshape(2, 3)
        if not np.isfinite(J).all():
            return None
        ray = np.linalg.svd(J)[2][-1]
    ray = np.asarray(ray, dtype=float).reshape(3)
    norm = float(np.linalg.norm(ray))
    return ray / norm if norm > 1e-12 and np.isfinite(ray).all() else None


def _predict_shift(external, dp) -> np.ndarray | None:
    if external is None:
        return None
    if hasattr(external, "predict_shift"):
        return np.asarray(external.predict_shift(dp), dtype=float).reshape(2)
    J = getattr(external, "J", None)
    return None if J is None else np.asarray(J, dtype=float).reshape(2, 3) @ np.asarray(dp, dtype=float)


def _robot_pixels(hand_mask, shape) -> np.ndarray:
    """The robot's own pixels in the upright wrist image: ``hand_mask`` (the calibrated fingertip band,
    ``V2Calibration.fingertip_band``: where the fingers can be at any gap) and every row from the band's topmost row
    down to the image bottom (the hand body enters from the image bottom, below the fingers, R0 up; [F, BO r1 fix]
    the real band spans rows 385-511 of the full width, the fake's misses the palm's bottom-left corner)."""
    robot = _mask(hand_mask, "hand_mask", shape).copy()
    rows = np.flatnonzero(robot.any(axis=1))
    if len(rows):
        robot[int(rows.min()):, :] = True
    return robot


def _window_on_robot(robot, uv, r) -> bool:
    """True when the (2r+1) window at ``uv`` contains a robot pixel (a template there would ride with the hand)."""
    if robot is None:
        return False
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = robot.shape
    return bool(robot[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)].any())


class _WristPatchRotation:
    """O-1 wrist measurement: the carried part's rigid patch tracked in the hand-mounted camera from the load-test
    start view (rotated-template search, K7 tracker). The camera is rigid with the hand, so a held part's patch stays
    put; every motion of it is in-hand motion (which is why a patch on the robot's own pixels is refused: it would
    read "held, not rotating" whatever the part does). Estimates of the in-hand rotation magnitude:
    - in-plane: the patch's apparent rotation (only when its template observes rotation, K7_MIN_ROTATION_DROP); a
      lower bound of the rotation (its component about the optical axis);
    - sphere (``xyz`` given: the patch's world point at the start): the pinch fixes the grip point, so the part can
      only turn about it; the start hand-frame point p is moved onto the tracked pixel's ray at |p| from the grip,
      and the angle between p and that point is the rotation. The point's height is known only to about one
      HeightField cell (HF_CELL_M; the G1' (b) p90 bound of a registered top is 5 mm), so the estimate is the
      SMALLEST angle over the start points at z and z +- HF_CELL_M along the start ray; [F, BO r1 fix, RS-7 frames]
      near the pinch (~28 mm) a 5 mm height change moves the angle by up to 2-9 deg on turning holds, at the far
      patch (~100 mm) by <= 0.8 deg. A ray that misses the sphere at the nominal height (the closest-approach point
      over-read 26-35 deg with the height 10 mm low) is ``ray_misses_sphere``: no sphere estimate that frame.
    The sphere estimate is a lower bound only when the tracked pixel is right and the height is within that cell; a
    template that drifts onto background (a mask over-grown past the silhouette) is not bounded, which is why
    ``select_wrist_patches`` erodes the mask by one projected cell. ``deg_per_px`` (the angle change for a 1 px
    pixel error) is reported, never compared: [F] 0.14-0.94 deg/px on exact inputs and 0.16-0.28 on the ray-miss
    over-reads, so it does not separate them.
    With ``xyz`` the hypothesis "left behind" (the part stays in the world while the hand moves) is also tested: its
    prediction is the start world point projected at the current pose; a tracked shift within O3_MODEL_FRACTION of
    that predicted shift (once the prediction exceeds the decision distance) is consistent with being left behind,
    and then the sphere angle does not measure a rotation. Only a patch whose hypothesis is decided against "left
    behind" (or a patch without a world point, which has no such hypothesis) gives an ``eligible`` estimate: the
    rotation test never counts a frame before that decision."""

    def __init__(self, gray0, patch: Mapping, pose0: HandPose, camera: WristCamera, robot=None):
        self.reason = None
        self.camera = camera
        self.p = self.xyz0 = None
        self.p_band = []
        self.offset = np.zeros(2)
        self.drop = None
        self.inplane = False
        uv = patch.get("uv")
        xyz = patch.get("xyz")
        projected = None
        if xyz is not None:
            xyz = np.asarray(xyz, dtype=float).reshape(-1)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError("wrist_patch['xyz'] must be a finite world point")
            projected, _depth = camera.project(pose0, xyz)
            if projected is None:
                self.reason = "patch_behind_camera"
                return
            self.xyz0 = xyz
            self.p = pose0.R.T @ (xyz - pose0.grip)
            if float(np.linalg.norm(self.p)) < 1e-9:
                self.p = None
            else:
                origin, d = camera.ray(pose0, projected)
                if abs(float(d[2])) > 1e-9:
                    for dz in (-HF_CELL_M, HF_CELL_M):
                        point = origin + (float(xyz[2]) + dz - float(origin[2])) / float(d[2]) * d
                        self.p_band.append(pose0.R.T @ (point - pose0.grip))
            if uv is None:
                uv = projected
        if uv is None:
            self.reason = "no_patch"
            return
        uv = np.asarray(uv, dtype=float).reshape(-1)[:2]
        if uv.shape != (2,) or not np.isfinite(uv).all():
            self.reason = "no_patch"
            return
        radius = K7_PATCH_RADIUS_PX if patch.get("radius_px") is None else int(patch["radius_px"])
        if _window_on_robot(robot, uv, radius):
            self.reason = "patch_on_hand"
            return
        self.tracker = _RotationTracker(gray0, uv, radius_px=radius)
        if self.tracker.reason is not None:
            self.reason = self.tracker.reason
            return
        self.uv0 = self.tracker.uv.copy()
        if projected is not None:
            self.projected0 = np.asarray(projected, dtype=float)
            self.offset = self.projected0 - self.uv0      # the patch point against its (integer) template centre
        self.drop = self.tracker.rotation_drop(np.asarray(gray0, dtype=np.float32))
        self.inplane = self.drop >= K7_MIN_ROTATION_DROP

    def _sphere_deg(self, pose: HandPose, uv, p) -> tuple[float, bool]:
        """(angle between p and the point of the pixel's ray at |p| from the grip, ray meets that sphere)."""
        c = pose.R.T @ (pose.camera - pose.grip)
        _origin, d = self.camera.ray(pose, uv)
        d = pose.R.T @ d
        r = float(np.linalg.norm(p))
        b = float(c @ d)
        disc = b * b - (float(c @ c) - r * r)
        if disc < 0.0:
            t, meets = -b, False                                  # the ray's closest approach to the grip
        else:
            roots = (-b - math.sqrt(disc), -b + math.sqrt(disc))
            t = min(roots, key=lambda value: float(np.linalg.norm(c + value * d - p)))
            meets = True
        q = c + t * d
        cosine = float(p @ q) / max(r * float(np.linalg.norm(q)), 1e-12)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), meets

    def measure(self, gray, pose: HandPose, decide_px: float) -> dict:
        angle, uv, peak = self.tracker.measure(gray)
        row = {"uv": None, "peak": None if peak is None else float(peak), "displacement_px": None,
               "inplane_deg": None, "sphere_deg": None, "ray_meets_sphere": None, "sphere_height_spread_deg": None,
               "deg_per_px": None, "left_behind_shift_px": None, "left_behind_error_px": None, "left_behind": None,
               "estimate_deg": None, "eligible": False}
        if uv is None:
            return row
        shift = np.asarray(uv, dtype=float) - self.uv0
        row.update(uv=[float(uv[0]), float(uv[1])], displacement_px=float(np.linalg.norm(shift)))
        estimates = []
        if self.inplane:
            row["inplane_deg"] = float(angle)
            estimates.append(abs(float(angle)))
        if self.p is not None:
            target = np.asarray(uv, dtype=float) + self.offset
            sphere, meets = self._sphere_deg(pose, target, self.p)
            row["ray_meets_sphere"] = meets
            behind, _depth = self.camera.project(pose, self.xyz0)
            if behind is not None:
                predicted = np.asarray(behind, dtype=float) - self.projected0
                lb_shift = float(np.linalg.norm(predicted))
                lb_error = float(np.linalg.norm(shift - predicted))
                row.update(left_behind_shift_px=lb_shift, left_behind_error_px=lb_error)
                if lb_shift > decide_px:
                    row["left_behind"] = bool(lb_error <= O3_MODEL_FRACTION * lb_shift)
            if meets:
                band = [sphere] + [a for a, ok in (self._sphere_deg(pose, target, q) for q in self.p_band) if ok]
                per_px = max(abs(self._sphere_deg(pose, target + step, self.p)[0] - sphere) for step in np.eye(2))
                row.update(sphere_deg=min(band), sphere_nominal_deg=sphere,
                           sphere_height_spread_deg=max(band) - min(band), deg_per_px=per_px)
                if row["left_behind"] is not True:
                    estimates.append(row["sphere_deg"])
            else:
                row["sphere_unmeasurable"] = "ray_misses_sphere"
        row["estimate_deg"] = max(estimates) if estimates else None
        row["eligible"] = self.xyz0 is None or row["left_behind"] is False
        return row


def select_wrist_patches(gray, mask, pose: HandPose, z_m: float, camera: WristCamera | None = None, *,
                         hand_mask) -> list[dict]:
    """O-1 rigid patches of a carried part in the upright wrist image of the load-test start view.

    ``mask`` = the part's VISIBLE pixels there (occlusion-aware: its registered region projected at this pose minus
    whatever the scene knows to be in front of it, not a plain region projection), ``z_m`` = the world height its
    pixels are back-projected to (its registered top), ``hand_mask`` (required) = the robot's own wrist pixels (the
    calibrated fingertip band; ``_robot_pixels`` adds the hand body below it). The robot's pixels are removed from the
    mask first ([F, BO r1 review] closed-finger and hand-body pixels inside the mask became both patches, rigid with
    the camera, and an object left behind read met in 11/15 replays); the rest is eroded by the patch window
    (K7_PATCH_RADIUS_PX) plus one HeightField cell (HF_CELL_M) projected at the median depth of the mask pixels, so a
    mask over-grown by up to one cell still keeps every template on the part ([F, BO r1 review] an 8 px dilation
    moved the far patch onto the silhouette edge and a rigid hold read 6.9 deg). Candidates: pixels of the eroded
    mask inside the image, with texture (window std >= K7_MIN_STD_GREY) and >= O3_PATCH_MIN_DIST_M from the pinch
    point. Returns up to two: the farthest candidate (the largest moment arm: a rotation moves it most) (among the
    candidates with >= O3_PATCH_TEXTURE_FRACTION of the best texture) and the most textured candidate outside that
    one's window (the scene's rigid-patch rule: the best tracked); each {"uv", "xyz", "std", "dist_m", "role",
    "erode_px"}. A rotation about the line from the pinch point to one patch does not move that patch; two patches at
    different places are not both blind."""
    image = np.asarray(gray, dtype=np.float32)
    part = _mask(mask, "mask", image.shape)
    if hand_mask is None:
        raise ValueError("hand_mask (the robot's own wrist pixels, the calibrated fingertip band) is required")
    visible = part & ~_robot_pixels(hand_mask, image.shape)
    camera = camera or WristCamera.from_robot(int(image.shape[1]))
    c = camera.principal_point
    R = np.asarray(pose.R, dtype=float)

    def rays(us, vs):
        """Unnormalised pixel rays (unit R2 component) and their depth t along R2 to the plane z = z_m."""
        d = (R[:, 2][None, :] + ((us - c[0]) / camera.f_px)[:, None] * R[:, 1][None, :]
             - ((vs - c[1]) / camera.f_px)[:, None] * R[:, 0][None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (float(z_m) - float(pose.camera[2])) / d[:, 2]
        return d, t
    vs, us = np.nonzero(visible)
    if not len(us):
        return []
    _d, t_all = rays(us.astype(float), vs.astype(float))
    t_all = t_all[np.isfinite(t_all) & (t_all > 0.0)]
    if not len(t_all):
        return []
    margin = int(math.ceil(camera.f_px * HF_CELL_M / float(np.median(t_all))))
    r = K7_PATCH_RADIUS_PX
    big = int(math.ceil(r * math.sqrt(2.0))) + 1
    size = 2 * (r + margin) + 1
    inner = cv2.erode(visible.astype(np.uint8), np.ones((size, size), np.uint8), borderType=cv2.BORDER_CONSTANT,
                      borderValue=0) > 0
    inner[:big, :], inner[-big:, :], inner[:, :big], inner[:, -big:] = False, False, False, False
    vs, us = np.nonzero(inner)
    if not len(us):
        return []
    k = (2 * r + 1, 2 * r + 1)
    mean = cv2.boxFilter(image, -1, k)
    std = np.sqrt(np.maximum(cv2.boxFilter(image * image, -1, k) - mean * mean, 0.0))[vs, us]
    d, t = rays(us.astype(float), vs.astype(float))
    world = np.asarray(pose.camera, dtype=float)[None, :] + t[:, None] * d
    dist = np.linalg.norm(world - np.asarray(pose.grip, dtype=float)[None, :], axis=1)
    ok = np.isfinite(t) & (t > 0.0) & (std >= K7_MIN_STD_GREY) & (dist >= O3_PATCH_MIN_DIST_M)
    index = np.flatnonzero(ok)
    if not len(index):
        return []
    chosen = []
    strong = index[std[index] >= O3_PATCH_TEXTURE_FRACTION * float(std[index].max())]
    far = int(strong[int(np.argmax(dist[strong]))])
    chosen.append((far, "far"))
    apart = index[(np.abs(us[index] - us[far]) > 2 * r) | (np.abs(vs[index] - vs[far]) > 2 * r)]
    if len(apart):
        chosen.append((int(apart[int(np.argmax(std[apart]))]), "texture"))
    return [{"uv": [float(us[i]), float(vs[i])], "xyz": world[i].tolist(), "std": float(std[i]),
             "dist_m": float(dist[i]), "role": role, "erode_px": margin} for i, role in chosen]


class _WristRotationSet:
    """O-1 wrist measurement over one or more rigid patches (``_WristPatchRotation`` each). The motion is left behind
    only when every patch whose prediction is decidable is consistent with it: [F, BO r1 fix, RS-7 frames] with the
    first selection (one patch ~28 mm from the pinch), on 10 of 80 real holds that turned 13-18 deg in the pinch the
    near patch's motion matched the left-behind prediction while the far patch read the turn, so a mixed set is a part
    moving with the hand (``patches_disagree`` is reported; with the hand rows excluded no set of the 110 re-run
    holds was mixed);
    the robot's own pixels, which would read "held" for a part left behind, are refused as patches
    (``select_wrist_patches``, ``patch_on_hand``). The estimate is the largest ELIGIBLE patch estimate (its patch is
    decided not left behind, or has no world point); an estimate of an undecided patch is reported as
    ``pending_estimate_deg`` only."""

    def __init__(self, gray0, patches, pose0: HandPose, camera: WristCamera, robot=None):
        self.cores = [_WristPatchRotation(gray0, patch, pose0, camera, robot) for patch in patches]
        usable = [core for core in self.cores if core.reason is None]
        self.reason = None if usable else (self.cores[0].reason if self.cores else "no_patch")
        self.drop = [core.drop for core in self.cores]

    def measure(self, gray, pose: HandPose, decide_px: float) -> dict:
        rows = [core.measure(gray, pose, decide_px) if core.reason is None else {"unmeasurable": core.reason}
                for core in self.cores]
        measured = [(i, row) for i, row in enumerate(rows) if row.get("uv") is not None]
        out = {"uv": None, "left_behind": None, "patches_disagree": False, "estimate_deg": None,
               "pending_estimate_deg": None, "patch": None, "patches": rows}
        if not measured:
            return out
        decided = [row["left_behind"] for _, row in measured if row.get("left_behind") is not None]
        out["left_behind"] = None if not decided else bool(all(decided))
        out["patches_disagree"] = len(set(decided)) > 1
        valid = [(row["estimate_deg"], i, row["eligible"]) for i, row in measured
                 if row.get("estimate_deg") is not None and row.get("left_behind") is not True]
        if out["left_behind"] is not True:
            eligible = [(e, i) for e, i, ok in valid if ok]
            pending = [e for e, _i, ok in valid if not ok]
            if eligible:
                out["estimate_deg"], out["patch"] = max(eligible)
            if pending:
                out["pending_estimate_deg"] = max(pending)
        first = measured[0][1]
        out.update(uv=first["uv"], peak=first.get("peak"), displacement_px=first.get("displacement_px"),
                   sphere_deg=first.get("sphere_deg"), inplane_deg=first.get("inplane_deg"),
                   left_behind_shift_px=first.get("left_behind_shift_px"),
                   left_behind_error_px=first.get("left_behind_error_px"))
        return out


class _ExternalPatchRotation:
    """O-1 external measurement: the apparent in-plane rotation of the carried part's external template (rotated-
    template search, K7 tracker, its search centre moved by the predicted J dp of each step) minus the hand's own
    rotation about the camera ray since the start (the whole hand rotation when the ray is unknown): the in-hand
    rotation's component about the ray, a lower bound of its magnitude."""

    def __init__(self, gray0, uv, external, pose0: HandPose):
        self.reason = None
        self.external = external
        self.ray = _external_ray(external)
        self.R0 = np.asarray(pose0.R, dtype=float)
        self.grip_prev = np.asarray(pose0.grip, dtype=float)
        self.drop = None
        self.tracker = _RotationTracker(gray0, uv, radius_px=K7_PATCH_RADIUS_PX)
        if self.tracker.reason is not None:
            self.reason = self.tracker.reason
            return
        self.drop = self.tracker.rotation_drop(np.asarray(gray0, dtype=np.float32))
        if self.drop < K7_MIN_ROTATION_DROP:
            self.reason = "rotation_degenerate"

    def measure(self, gray, pose: HandPose) -> dict:
        grip = np.asarray(pose.grip, dtype=float)
        step = _predict_shift(self.external, grip - self.grip_prev)
        self.grip_prev = grip
        if step is not None and np.isfinite(step).all():
            self.tracker.uv = self.tracker.uv + step
        angle, uv, peak = self.tracker.measure(gray)
        w = _rotvec_of(np.asarray(pose.R, dtype=float) @ self.R0.T)
        hand_deg = math.degrees(abs(float(w @ self.ray)) if self.ray is not None else float(np.linalg.norm(w)))
        row = {"uv": None if uv is None else [float(uv[0]), float(uv[1])], "peak": None if peak is None else
               float(peak), "apparent_deg": None if angle is None else float(angle), "hand_about_ray_deg": hand_deg,
               "estimate_deg": None}
        if angle is not None:
            row["estimate_deg"] = max(0.0, abs(float(angle)) - hand_deg)
        return row


class CoMotionObserver:
    """O3 co-motion with the O-1 rotation test (review-1 decision §2.5).

    Translation (O3): the wrist part patch stays within O3_WRIST_MAX_PX (v1 ``WristPatchMonitor`` at ``wrist_uv``)
    AND the external template (``ext_track``, an NCCTrack on the external image) moves by the predicted J_ext dp
    (dp = grip displacement since ``start``) within max(O3_SIGMA_K sigma, O3_MODEL_FRACTION |J dp|): sigma =
    max(``sigma_px``, O3_EXTERNAL_FLOOR_PX) covers the track and locator noise, the |J dp| term the affine model's
    own error over a long carry (J is local; a carried object deviates from J dp by <= ~0.25 |J dp|, an object
    left behind by |J dp|). Pending until |J dp| exceeds O3_SIGMA_K sigma [C-46] (an object left behind cannot be
    told from a carried one before that); a template that is not found (occlusion, specular peak drop) is pending
    for that frame and counts as violated only after O3_LOST_FRAMES consecutive misses (V1 2-frame loss); violated
    ``grip_not_on_obj`` (the relation names it ``lost_in_transport`` during the carry). A part without a wrist patch
    or an external template is judged by the other one; with neither the observer stays pending.

    Rotation (O-1): the carried part's rigid patch must not ROTATE in the hand either. ``wrist_patch`` is one of:
    {"mask" (the part's VISIBLE pixels in the start view's upright wrist image: occlusion-aware, not a plain
    projection of its registered region), "z_m" (their world height, its registered top)}: the patches are chosen by
    ``select_wrist_patches``; one or more {"uv" (pixel in the start view), "xyz" (its world point at the start),
    optional "radius_px"} with at least one of uv / xyz (uv defaults to the projection of xyz). With a
    ``wrist_patch`` the robot's own wrist pixels ``wrist_hand_mask`` (the calibrated fingertip band,
    ``V2Calibration.fingertip_band``) are required: no patch may lie on them (``patch_on_hand``). Without
    ``wrist_patch`` the wrist rotation uses ``wrist_uv`` (in-plane estimate only). With world points (the wrist
    model) the wrist translation test is the three-hypothesis test of ``_WristPatchRotation`` instead of the v1 10 px
    rule (that rule's pixel limit depends on the patch's distance from the grip): left behind -> violated
    ``grip_not_on_obj`` (cause ``left_behind``); a measured rotation over threshold -> ``grip_unstable``; otherwise
    held (pending until the left-behind prediction exceeds the decision distance O3_SIGMA_K sigma).

    The external template is never overruled into ``met``: when the wrist model has decided that the part moves
    with the hand but the external template lags J dp beyond its tolerance, the part is turning about the pinch (its
    far template lags; [F, BO r1 review] a tilted hold whose wrist estimate crossed 5 deg on the last frame only had
    an 11.0 px lag against 9.0 px) -> violated ``grip_unstable`` (cause ``external_lag``) while the rotation verdict
    is on, else the O3 rule (``grip_not_on_obj``, cause ``displaced``).

    The in-hand rotation estimate is the wrist one (the largest eligible patch estimate; none while the wrist motion
    is consistent with the part being left behind: it is not in the hand). The external template's in-plane rotation
    (``_ExternalPatchRotation``) is measured and reported but never decides: [F, BO r1 RS-7] it read up to 5.5 deg
    on reference holds whose true in-hand rotation was <= 2.7 deg (the moving hand and the background inside the
    template) and 2.1-13.0 deg on forced ones, so it does not separate them; a part the wrist cannot measure is
    reported ``rotation_unmeasured``. grip_unstable when the estimate exceeds O3_ROTATION_MAX_DEG on
    O3_ROTATION_CONFIRM_FRAMES consecutive frames (the estimate's jitter sigma_rotation is reported, never compared:
    measured over the series it is meant to judge, it rises with the rotation onset itself); the verdict then
    latches and takes precedence over ``grip_not_on_obj`` (a part that turns about the pinch was gripped but is not
    held stably: open and re-grasp, not repoint), except when the wrist set has read "left behind" on the last
    O3_ROTATION_CONFIRM_FRAMES frames (a tracking glitch of one patch must not turn a sustained left-behind reading
    into a re-grasp).

    ``rotation_verdict``: None (default) = on exactly when ``wrist_patch`` is given (the load-test registration
    form); the contract signature (``wrist_uv`` only) keeps the pre-review O3 verdicts and reports the in-plane
    estimate as evidence. Pass False on EVERY carry-scope registration (where O4 judges the in-hand tilt: a held part
    pressed onto a support slides in the pads, which the sphere estimate reads as a rotation), True to judge the
    in-plane estimate of a ``wrist_uv``-only registration."""

    name = "comotion"

    def __init__(self, *, wrist_uv: tuple | None, wrist_view: StepView, ext_track: "NCCTrack | None",
                 external: "ExternalModel | None", robot: RobotModel, sigma_px: float | None = None,
                 wrist_patch: Mapping | None = None, wrist_camera: WristCamera | None = None,
                 wrist_hand_mask: np.ndarray | None = None, rotation_verdict: bool | None = None):
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        if rotation_verdict is not None and not isinstance(rotation_verdict, bool):
            raise ValueError("rotation_verdict must be None or a bool")
        _view(wrist_view, "wrist_view")
        if wrist_camera is not None and not isinstance(wrist_camera, WristCamera):
            raise ValueError("wrist_camera must be a WristCamera")
        self.robot = robot
        self.wrist_patch = self._patch_spec(wrist_patch)
        if self.wrist_patch is not None and wrist_hand_mask is None:
            raise ValueError("a wrist_patch needs wrist_hand_mask (the robot's own wrist pixels, the calibrated "
                             "fingertip band)")
        self.wrist_hand_mask = None if wrist_hand_mask is None else _mask(wrist_hand_mask, "wrist_hand_mask")
        self.rotation_verdict = (self.wrist_patch is not None) if rotation_verdict is None else rotation_verdict
        self.wrist_model = self.wrist_patch is not None and (
            "mask" in self.wrist_patch or any(p.get("xyz") is not None for p in self.wrist_patch["patches"]))
        self.wrist_camera = wrist_camera
        self.wrist_uv = wrist_uv
        self.monitor = None if wrist_uv is None or self.wrist_model else \
            WristPatchMonitor(wrist_view, uv=wrist_uv, max_displacement_px=O3_WRIST_MAX_PX)
        self.ext_track = ext_track
        self.external = external
        sigma = O3_EXTERNAL_FLOOR_PX if sigma_px is None else _nonnegative(sigma_px, "sigma_px")
        self.sigma_th = th_max("O3_external_sigma_px", Term("DD", O3_EXTERNAL_FLOOR_PX, "O3_EXTERNAL_FLOOR_PX"),
                               Term("ONLINE", sigma, "sigma_px (locator / projector)"), unit="px")
        self.decide_th = th_value("O3_external_decide_px", O3_SIGMA_K * self.sigma_th.value, self.sigma_th.label,
                                  "O3_SIGMA_K x sigma (|J dp| needed before the external test decides)",
                                  sigma=self.sigma_th.value)
        self.wrist_th = th_value("O3_wrist_px", O3_WRIST_MAX_PX, "DD", "O3_WRIST_MAX_PX")
        self.rotation_th = th_value("O3_rotation_deg", O3_ROTATION_MAX_DEG, "DD", "O3_ROTATION_MAX_DEG")
        self.started = False

    @staticmethod
    def _patch_spec(wrist_patch):
        """None | {"patches": [patch, ...]} | {"mask", "z_m"} from the ``wrist_patch`` argument."""
        if wrist_patch is None:
            return None
        if isinstance(wrist_patch, Mapping) and "mask" in wrist_patch:
            z = wrist_patch.get("z_m")
            if z is None or not math.isfinite(float(z)):
                raise ValueError("a wrist_patch mask needs the finite height 'z_m' of its pixels")
            return {"mask": _mask(wrist_patch["mask"], "wrist_patch['mask']"), "z_m": float(z)}
        patches = [wrist_patch] if isinstance(wrist_patch, Mapping) else list(wrist_patch)
        if not patches or any(not isinstance(p, Mapping) or (p.get("uv") is None and p.get("xyz") is None)
                              for p in patches):
            raise ValueError("wrist_patch must be {'mask', 'z_m'}, or one or more mappings with 'uv' and / or 'xyz'")
        return {"patches": [dict(p) for p in patches]}

    def start(self, view) -> None:
        view = _view(view)
        self.grip0 = view.position.copy()
        self.uv0 = None if self.ext_track is None else np.asarray(self.ext_track.uv, dtype=float).copy()
        self.count = 0
        self.ext_misses = 0
        self.wrist_misses = 0
        pose0 = HandPose.from_view(view)
        gray_w = view.wrist_gray_upright
        shape = np.asarray(gray_w).shape[:2]
        camera = self.wrist_camera or WristCamera.from_robot(int(shape[1]))
        robot = None if self.wrist_hand_mask is None else _robot_pixels(self.wrist_hand_mask, shape)
        spec = self.wrist_patch
        if spec is None:
            patches = None if self.wrist_uv is None else [{"uv": self.wrist_uv}]
        elif "mask" in spec:
            patches = select_wrist_patches(gray_w, spec["mask"], pose0, spec["z_m"], camera,
                                           hand_mask=self.wrist_hand_mask)
        else:
            patches = spec["patches"]
        self.wrist_patches_used = None if patches is None else jsonable(patches)
        self.wrist_rot = None if patches is None else _WristRotationSet(gray_w, patches, pose0, camera, robot)
        self.ext_rot = None
        if self.ext_track is not None:
            self.ext_rot = _ExternalPatchRotation(_gray(view.external_rgb_upright), self.uv0, self.external, pose0)
        self.rot_series = {"wrist": [], "external": []}
        self.rot_misses = {"wrist": 0, "external": 0}
        self.rot_lost = {"wrist": None, "external": None}
        self.rot_over = 0
        self.rot_latched = None
        self.left_behind_run = 0
        self.started = True

    def _expected(self, dp):
        return _predict_shift(self.external, dp)

    def _external_threshold(self, expected_norm) -> Threshold:
        return th_max("O3_external_px", Term(self.sigma_th.label, O3_SIGMA_K * self.sigma_th.value, "O3_SIGMA_K*sigma"),
                      Term("ONLINE", O3_MODEL_FRACTION * expected_norm, "O3_MODEL_FRACTION*|J dp|"), unit="px")

    # ------------------------------------------------------------------------------------------- O-1 rotation
    def _camera_rotation(self, camera, core, measure):
        """One camera's O-1 row (None when the camera has no usable rotation patch)."""
        if core is None:
            return None
        if core.reason is not None or self.rot_lost[camera] is not None:
            return {"unmeasurable": core.reason or self.rot_lost[camera], "rotation_drop": core.drop}
        row = measure()
        if row["uv"] is None:
            self.rot_misses[camera] += 1
            if self.rot_misses[camera] >= O3_LOST_FRAMES:
                self.rot_lost[camera] = "lost"
            row["misses"] = self.rot_misses[camera]
            return row
        self.rot_misses[camera] = 0
        if row["estimate_deg"] is not None:
            self.rot_series[camera].append(float(row["estimate_deg"]))
        row.update(rotation_drop=core.drop, misses=0)
        return row

    def _rotation(self, after, pose) -> tuple[dict, Threshold | None]:
        rows = {"wrist": self._camera_rotation(
                    "wrist", self.wrist_rot,
                    lambda: self.wrist_rot.measure(after.wrist_gray_upright, pose, self.decide_th.value)),
                "external": self._camera_rotation(
                    "external", self.ext_rot,
                    lambda: self.ext_rot.measure(_gray(after.external_rgb_upright), pose))}
        wrist = rows["wrist"]
        modelled = self.wrist_model and wrist is not None and wrist.get("uv") is not None
        left_behind = modelled and wrist.get("left_behind") is True
        pending = wrist is not None and wrist.get("estimate_deg") is None and \
            wrist.get("pending_estimate_deg") is not None
        # the verdict reads the hand-mounted camera only; the external in-plane estimate is evidence (class docstring)
        if left_behind or wrist is None or wrist.get("estimate_deg") is None:
            candidates = []                    # left behind: the part is not in the hand, nothing to rotate in it
        else:
            candidates = [(wrist["estimate_deg"], "wrist")]
        evidence = {**rows, "estimate_deg": None, "source": None, "threshold_deg": None, "over_frames": self.rot_over,
                    "latched": self.rot_latched is not None,
                    "rotation_unmeasured": not candidates and not left_behind and not pending,
                    "rotation_pending": pending, "left_behind": left_behind,
                    "verdict_enabled": self.rotation_verdict, "wrist_patches": self.wrist_patches_used}
        if not candidates:
            self.rot_over = 0
            evidence["over_frames"] = 0
            return evidence, None
        estimate, source = max(candidates)
        threshold = self.rotation_th
        self.rot_over = self.rot_over + 1 if estimate > threshold.value and self.rotation_verdict else 0
        evidence.update(estimate_deg=estimate, source=source, threshold_deg=threshold.value, over_frames=self.rot_over,
                        sigma_rotation_deg=_jitter_sigma(self.rot_series[source]),
                        over_threshold=bool(estimate > threshold.value))
        return evidence, threshold

    # ---------------------------------------------------------------------------------------------- update
    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("CoMotionObserver.start(view) must be called first")
        after = _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        pose = HandPose.from_view(after)
        evidence = {"wrist": None, "external": None, "rotation": None, "cause": None}
        wrist_ok, ext_ok = None, None
        ths = [self.wrist_th, self.sigma_th, self.decide_th]
        rotation, rot_th = self._rotation(after, pose)
        evidence["rotation"] = rotation
        if rot_th is not None:
            ths.append(rot_th)
        wrist_measured = False
        if self.wrist_model and self.wrist_rot is not None and self.wrist_rot.reason is None:
            row = rotation["wrist"]
            wrist_measured = True
            if row is None or row.get("uv") is None:                      # template miss: V1 2-frame loss
                self.wrist_misses += 1
                wrist_ok = False if self.wrist_misses >= O3_LOST_FRAMES else None
            else:
                self.wrist_misses = 0
                wrist_ok = None if row["left_behind"] is None else not row["left_behind"]
            evidence["wrist"] = {"mode": "model", "misses": self.wrist_misses,
                                 "left_behind": None if row is None else row.get("left_behind"),
                                 "patches_disagree": None if row is None else row.get("patches_disagree")}
        elif self.monitor is not None and self.monitor.available:
            wrist_measured = True
            wrist = self.monitor.observe(after)
            if wrist["displacement_px"] is None:                       # template miss: V1 2-frame loss
                self.wrist_misses += 1
                wrist_ok = False if self.wrist_misses >= O3_LOST_FRAMES else None
            else:
                self.wrist_misses = 0
                wrist_ok = bool(wrist["ok"])
            evidence["wrist"] = {**wrist, "mode": "v1_patch", "misses": self.wrist_misses}
        left_now = self.wrist_model and wrist_ok is False and self.wrist_misses < O3_LOST_FRAMES
        self.left_behind_run = self.left_behind_run + 1 if left_now else 0
        external_lag = False
        if self.ext_track is not None and self.external is not None and self.uv0 is not None:
            dp = after.position - self.grip0
            expected = self._expected(dp)
            norm = float(np.linalg.norm(expected))
            ext_th = self._external_threshold(norm)
            ths.append(ext_th)
            uv, peak = self.ext_track.match(_gray(after.external_rgb_upright))
            row = {"expected_px": expected.tolist(), "expected_norm_px": norm,
                   "peak": None if peak is None else float(peak), "measured_px": None, "error_px": None,
                   "tolerance_px": ext_th.value, "decided": norm > self.decide_th.value, "misses": 0}
            if uv is None:
                self.ext_misses += 1
            else:
                self.ext_misses = 0
                measured = np.asarray(uv, dtype=float) - self.uv0
                row["measured_px"] = measured.tolist()
                row["error_px"] = float(np.linalg.norm(measured - expected))
            row["misses"] = self.ext_misses
            if self.ext_misses >= O3_LOST_FRAMES:
                ext_ok = False
            elif row["decided"] and row["error_px"] is not None:
                ext_ok = row["error_px"] <= ext_th.value
                # the wrist model says the part moves with the hand while its external template lags J dp: the part
                # turns about the pinch (class docstring); never overruled into met
                external_lag = bool(not ext_ok and self.wrist_model and wrist_ok is True)
            row["lag_with_wrist_held"] = external_lag
            evidence["external"] = row
        if self.rot_latched is None and self.rot_over >= O3_ROTATION_CONFIRM_FRAMES:
            self.rot_latched = {"step": step, "estimate_deg": rotation["estimate_deg"], "source": rotation["source"],
                                "threshold_deg": rotation["threshold_deg"]}
        if self.rot_latched is not None:
            rotation["latched"] = True
            rotation["latched_at"] = dict(self.rot_latched)
            if self.left_behind_run < O3_ROTATION_CONFIRM_FRAMES:
                evidence["cause"] = "part_rotated"
                return _report(self.name, VERDICT_VIOLATED, "grip_unstable", evidence, ths, step)
            rotation["latch_overridden"] = "left_behind_sustained"
        if external_lag and self.rotation_verdict:
            evidence["cause"] = "external_lag"
            return _report(self.name, VERDICT_VIOLATED, "grip_unstable", evidence, ths, step)
        if wrist_ok is False or ext_ok is False:
            lost = (wrist_ok is False and self.wrist_misses >= O3_LOST_FRAMES) or \
                (ext_ok is False and self.ext_misses >= O3_LOST_FRAMES)
            left = self.wrist_model and wrist_ok is False and not lost
            evidence["cause"] = "template_lost" if lost else ("left_behind" if left else "displaced")
            return _report(self.name, VERDICT_VIOLATED, "grip_not_on_obj", evidence, ths, step)
        if ext_ok is None and (self.ext_track is not None and self.external is not None):
            return _report(self.name, VERDICT_PENDING, None, evidence, ths, step)
        if wrist_ok is None and ext_ok is None:
            return _report(self.name, VERDICT_PENDING, None, evidence, ths, step)
        if wrist_ok is None and wrist_measured:
            return _report(self.name, VERDICT_PENDING, None, evidence, ths, step)
        return _report(self.name, VERDICT_MET, None, evidence, ths, step)


# ============================================================================================== UprightObserver
def _silhouette_moments(mask):
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
    values, vectors = np.linalg.eigh(cov)
    return np.array([cx, cy]), vectors[:, int(np.argmax(values))]


class UprightObserver:
    """O4 upright (+ O2 rigid hold): violated ``object_rotated_in_hand`` when the hand is more than M3
    (UPRIGHT_REORIENT_RAD) from the grasp orientation outside a commanded reorientation, or when the in-hand tilt
    changed by more than O4_TILT_MAX_DEG.

    Commanded reorientation: ``context["commanded_reorientation"]`` (bool) when the caller gives it; otherwise the
    active primitive's name starting with "orient" (the Orient primitives; a fallback for callers that do not set
    the flag).

    In-hand tilt, first available of: ``tilt_fn(view)`` (deg, e.g. a held-frame estimator); the EXTERNAL principal
    axis (contract §C9, spec §4.10): ``ext_track`` (the held object's external NCCTrack; a private copy is tracked)
    locates the object in the external image and the structure axis (``axis_fn``, default geometry
    ``structure_axis``) at that point gives its image axis, compared with the axis at ``start`` (coherence >=
    O4_MIN_COHERENCE at both); the wrist silhouette principal axis (``silhouette_fn(view)`` with
    ``held.silhouette``), which also gives O2 (IoU >= O2_IOU_MIN, centroid shift <= max(O2_SHIFT_FLOOR_PX,
    O2_SIGMA_K sigma_track)), reported with the same reason. The external axis change includes the hand's own
    rotation about the camera ray, which the M3 test bounds separately. Without any measurement only the hand
    orientation is checked (evidence ``tilt_unmeasured``)."""

    name = "upright"

    def __init__(self, *, robot: RobotModel, grasp_quaternion, held: HeldFrame, ext_track=None,
                 reorient_rad: float | None = None, tilt_fn: Callable | None = None,
                 silhouette_fn: Callable | None = None, sigma_track_px: float | None = None,
                 axis_fn: Callable | None = None):
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        if not isinstance(held, HeldFrame):
            raise ValueError("held must be a HeldFrame")
        q = np.asarray(grasp_quaternion, dtype=float).reshape(-1)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
            raise ValueError("grasp_quaternion must be a finite quaternion (x, y, z, w)")
        if ext_track is not None and not callable(getattr(ext_track, "match", None)):
            raise ValueError("ext_track must be an NCCTrack (or None)")
        self.robot = robot
        self.grasp_q = q / np.linalg.norm(q)
        self.held = held
        self.ext_track = None if ext_track is None else copy.deepcopy(ext_track)   # never moves the caller's track
        self.axis_fn = axis_fn
        tol = UPRIGHT_REORIENT_RAD if reorient_rad is None else _finite(reorient_rad, "reorient_rad")
        self.m3_th = th_value("M3_reorient_rad", tol, "DD", "UPRIGHT_REORIENT_RAD (spec M3)")
        self.o4_th = th_value("O4_tilt_deg", O4_TILT_MAX_DEG, "DD", "O4_TILT_MAX_DEG")
        self.o2_iou_th = th_value("O2_iou", O2_IOU_MIN, "DD", "O2_IOU_MIN")
        sigma = 0.0 if sigma_track_px is None else _nonnegative(sigma_track_px, "sigma_track_px")
        self.o2_shift_th = th_max("O2_shift_px", Term("DD", O2_SHIFT_FLOOR_PX, "O2_SHIFT_FLOOR_PX"),
                                  Term("ONLINE", O2_SIGMA_K * sigma, "O2_SIGMA_K*sigma_track"), unit="px")
        self.coherence_th = th_value("O4_axis_coherence", O4_MIN_COHERENCE, "DD", "O4_MIN_COHERENCE")
        self.tilt_fn = tilt_fn
        self.silhouette_fn = silhouette_fn
        self.ref_moments = None if held.silhouette is None else _silhouette_moments(held.silhouette)
        self.started = False

    def _external_axis(self, gray):
        """(axis, coherence, uv) of the held object in the external image (axis None when not usable; uv None when
        the translation-only template match failed)."""
        uv, _peak = self.ext_track.match(gray)
        if uv is None:
            return None, None, None
        self.ext_last_uv = np.asarray(uv, dtype=float).copy()
        axis, coherence, _rms = (self.axis_fn or _default_axis)(gray, uv)
        if axis is None or coherence is None or float(coherence) < O4_MIN_COHERENCE:
            return None, None if coherence is None else float(coherence), [float(uv[0]), float(uv[1])]
        return np.asarray(axis, dtype=float), float(coherence), [float(uv[0]), float(uv[1])]

    def _reacquire_tilt(self, gray):
        """Template lost (a large in-hand tilt defeats the translation-only match): the start template rotated over
        +-O4_REACQUIRE_SPAN_DEG in O4_REACQUIRE_STEP_DEG steps, searched around the last location; (|best angle|,
        peak) when the best peak reaches the track's min_peak, else (None, peak)."""
        if self.ext_source is None:
            return None, None
        n = int(round(O4_REACQUIRE_SPAN_DEG / O4_REACQUIRE_STEP_DEG))
        best_angle, best_peak = None, None
        for angle in O4_REACQUIRE_STEP_DEG * np.arange(-n, n + 1):
            peak = _best_ncc(gray, _rotated_core(self.ext_source, self.ext_big, self.ext_r, float(angle)),
                             self.ext_last_uv, int(self.ext_track.search))
            if peak is not None and (best_peak is None or peak > best_peak):
                best_angle, best_peak = float(angle), peak
        if best_peak is None or best_peak < float(self.ext_track.min_peak):
            return None, best_peak
        return abs(best_angle), best_peak

    def start(self, view) -> None:
        view = _view(view)
        self.tilt0 = None if self.tilt_fn is None else self.tilt_fn(view)
        self.ext_axis0 = self.ext_coherence0 = self.ext_source = None
        if self.ext_track is not None:
            gray = _gray(view.external_rgb_upright)
            self.ext_last_uv = np.asarray(self.ext_track.uv, dtype=float).copy()
            self.ext_r = int(self.ext_track.radius)
            self.ext_big = int(math.ceil(self.ext_r * math.sqrt(2.0))) + 1
            self.ext_source = _source_at(gray, self.ext_last_uv, self.ext_big)
            self.ext_axis0, self.ext_coherence0, _uv = self._external_axis(gray)
        self.count = 0
        self.started = True

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("UprightObserver.start(view) must be called first")
        after = _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        ctx = context if isinstance(context, Mapping) else {}
        flag = ctx.get("commanded_reorientation")
        if isinstance(flag, bool):
            commanded, commanded_by = flag, "context"
        else:
            commanded, commanded_by = str(ctx.get("primitive") or "").startswith("orient"), "primitive_name"
        angle = quaternion_angle_rad(after.quaternion_xyzw, self.grasp_q)
        evidence = {"hand_angle_rad": angle, "commanded_reorientation": commanded, "commanded_by": commanded_by,
                    "tilt_change_deg": None, "tilt_source": None, "iou": None, "centroid_shift_px": None,
                    "tilt_unmeasured": True, "external_axis": None, "cause": None}
        ths = [self.m3_th, self.o4_th]
        violated = None
        if angle > self.m3_th.value and not commanded:
            violated = "hand_orientation"
        if self.tilt_fn is not None:
            tilt = self.tilt_fn(after)
            if tilt is not None and self.tilt0 is not None:
                evidence.update(tilt_change_deg=abs(float(tilt) - float(self.tilt0)), tilt_source="tilt_fn",
                                tilt_unmeasured=False)
        if self.ext_track is not None:
            ths.append(self.coherence_th)
            gray = _gray(after.external_rgb_upright)
            axis, coherence, uv = self._external_axis(gray)
            row = {"uv": uv, "coherence": coherence, "coherence0": self.ext_coherence0,
                   "usable": axis is not None and self.ext_axis0 is not None, "template_lost": uv is None,
                   "reacquired_tilt_deg": None, "reacquire_peak": None}
            if row["usable"] and evidence["tilt_change_deg"] is None:
                evidence.update(tilt_change_deg=abs(_axis_angle_deg(axis, self.ext_axis0)),
                                tilt_source="external_axis", tilt_unmeasured=False)
            elif uv is None and evidence["tilt_change_deg"] is None:
                tilt, peak = self._reacquire_tilt(gray)
                row.update(reacquired_tilt_deg=tilt, reacquire_peak=peak)
                if tilt is not None:
                    evidence.update(tilt_change_deg=tilt, tilt_source="external_template_rotation",
                                    tilt_unmeasured=False)
            evidence["external_axis"] = row
        if self.silhouette_fn is not None and self.held.silhouette is not None and self.ref_moments is not None:
            now = self.silhouette_fn(after)
            if now is not None:
                ref = np.asarray(self.held.silhouette, bool)
                cur = _mask(now, "silhouette", ref.shape)
                union = int((ref | cur).sum())
                iou = float((ref & cur).sum()) / union if union else 0.0
                moments = _silhouette_moments(cur)
                evidence["iou"] = iou
                ths += [self.o2_iou_th, self.o2_shift_th]
                if moments is not None:
                    shift = float(np.linalg.norm(moments[0] - self.ref_moments[0]))
                    evidence["centroid_shift_px"] = shift
                    if evidence["tilt_change_deg"] is None:
                        evidence.update(tilt_change_deg=abs(_axis_angle_deg(moments[1], self.ref_moments[1])),
                                        tilt_source="silhouette_axis", tilt_unmeasured=False)
                    if violated is None and (iou < O2_IOU_MIN or shift > self.o2_shift_th.value):
                        violated = "rigid_hold"
        if violated is None and evidence["tilt_change_deg"] is not None and \
                evidence["tilt_change_deg"] > O4_TILT_MAX_DEG:
            violated = "in_hand_tilt"
        if violated is not None:
            evidence["cause"] = violated
            return _report(self.name, VERDICT_VIOLATED, "object_rotated_in_hand", evidence, ths, step,
                           angle / self.m3_th.value)
        return _report(self.name, VERDICT_MET, None, evidence, ths, step, angle / self.m3_th.value)


# ============================================================================================== KeepOutObserver
def _feedback_event(feedback) -> str | None:
    """A contact event in a primitive's per-step feedback: its finishing reason or an explicit ``event`` key."""
    if not isinstance(feedback, Mapping):
        return None
    for key in ("event", "reason"):
        value = feedback.get(key)
        if isinstance(value, str) and value in CONTACT_EVENTS:
            return value
    return None


class KeepOutObserver:
    """O5 contact side: violated ``keepout_contact`` when a stall / touch / gap-rise event (in
    ``context["primitive_feedback"]``) occurs while the hand footprint (R7 rectangle at the after pose, from the
    fingertips to the hand top) or the held footprint (radius r_obj about the held reference point, from its lowest
    point to the grip) intersects a keep-out volume (polygon x [z_low, z_high]); met otherwise."""

    name = "keepout"

    def __init__(self, *, keepouts: Sequence[KeepOut], robot: RobotModel, held: HeldFrame | None):
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        self.keepouts = tuple(keepouts)
        if not all(isinstance(k, KeepOut) for k in self.keepouts):
            raise ValueError("keepouts must be KeepOut records")
        if held is not None and not isinstance(held, HeldFrame):
            raise ValueError("held must be a HeldFrame or None")
        self.robot = robot
        self.held = held
        self.started = False

    def start(self, view) -> None:
        _view(view)
        self.count = 0
        self.started = True

    def _hand_samples(self, view):
        pose = self.robot.pose(view.state)
        jaw = pose.jaw[:2]
        if float(np.linalg.norm(jaw)) < 1e-6:
            jaw = np.array([0.0, 1.0])
        poly = self.robot.hand_footprint_xy(pose.grip[:2], jaw)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        xs = np.arange(lo[0], hi[0] + O5_SAMPLE_SPACING_M, O5_SAMPLE_SPACING_M)
        ys = np.arange(lo[1], hi[1] + O5_SAMPLE_SPACING_M, O5_SAMPLE_SPACING_M)
        grid = np.array([[x, y] for x in xs for y in ys])
        pts = np.vstack([poly, grid[poly_contains(poly, grid)]]) if grid.size else poly
        z_low = float(pose.grip[2]) - self.robot.t_f_m
        z_high = float(pose.hand[2])
        return poly, pts, (z_low, z_high), pose

    def _held_samples(self, pose):
        centre = self.held.object_ref_point(pose)
        angles = np.linspace(0.0, 2.0 * math.pi, O5_HELD_RING_POINTS, endpoint=False)
        ring = centre[:2] + self.held.r_obj_m * np.c_[np.cos(angles), np.sin(angles)]
        pts = np.vstack([ring, centre[:2][None, :]])
        return ring, pts, (self.held.lowest_point_z(pose), float(pose.grip[2]))

    @staticmethod
    def _intersects(keepout, poly, pts, z_range) -> bool:
        if z_range[1] < keepout.z_low or z_range[0] > keepout.z_high:
            return False
        if poly_contains(keepout.polygon_xy, pts).any():
            return True
        return bool(poly_contains(poly, np.asarray(keepout.polygon_xy, float)).any())

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("KeepOutObserver.start(view) must be called first")
        after = _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        feedback = (context or {}).get("primitive_feedback") if isinstance(context, Mapping) else None
        event = _feedback_event(feedback)
        hits = []
        if event is not None and self.keepouts:
            poly, pts, z_range, pose = self._hand_samples(after)
            volumes = [("hand", poly, pts, z_range)]
            if self.held is not None:
                volumes.append(("held", *self._held_samples(pose)))
            for keepout in self.keepouts:
                for who, p, s, z in volumes:
                    if self._intersects(keepout, p, s, z):
                        hits.append({"pid": keepout.pid, "kind": keepout.kind, "by": who})
        ths = tuple(t for keepout in self.keepouts for t in keepout.thresholds)
        evidence = {"event": event, "hits": hits, "keepouts": [k.pid for k in self.keepouts]}
        if hits:
            return _report(self.name, VERDICT_VIOLATED, "keepout_contact", evidence, ths, step)
        return _report(self.name, VERDICT_MET, None, evidence, ths, step)


# =============================================================================================== RestObserver
class RestObserver:
    """M11 at rest (contract §C10.4 order: after ReleaseVerify, with the hand still). Per frame: the mean frame
    difference (grey) over the object region outside the hand (the dilated motion mask of ``hand_mask_fn``, plus the
    projected hand footprint ``static_hand_mask_fn(view)`` when given, as EffectWatch / MotionNoiseMeter). A frame
    in which the grip moved by more than M11_HAND_STILL_M (or whose region is fully masked) is not measured: while
    the hand moves its other pixels over the region would read as object motion, so the quiet run restarts
    (pending, ``hand_moving``). Over the last M11_FRAMES consecutive measured frames: all <= max(M11_FLOOR_GREY,
    M11_SIGMA_K sigma_static) -> met ``at_rest``; any above -> violated ``object_moved_after_release``; pending
    until M11_FRAMES measured frames were seen."""

    name = "rest"

    def __init__(self, *, region: np.ndarray, sigma_static_grey: float, camera: str = "external",
                 hand_mask_fn: Callable | None = None, static_hand_mask_fn: Callable | None = None):
        self.region = _mask(region, "region")
        if not self.region.any():
            raise ValueError("region must contain at least one pixel")
        if camera not in ("external", "wrist"):
            raise ValueError("camera must be 'external' or 'wrist'")
        self.camera = camera
        self.hand_mask_fn = hand_mask_fn
        self.static_hand_mask_fn = static_hand_mask_fn
        sigma = _nonnegative(sigma_static_grey, "sigma_static_grey")
        self.th = th_max("M11_rest_diff_grey", Term("DD", M11_FLOOR_GREY, "M11_FLOOR_GREY"),
                         Term("ONLINE", M11_SIGMA_K * sigma, "M11_SIGMA_K*sigma_static"), unit="grey")
        self.still_th = th_value("M11_hand_still_m", M11_HAND_STILL_M, "DD", "M11_HAND_STILL_M (grip motion per step)")
        self.started = False

    def start(self, view) -> None:
        _view(view)
        self.diffs: list[float | None] = []
        self.run: list[float] = []
        self.count = 0
        self.started = True

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("RestObserver.start(view) must be called first")
        before, after = _view(before, "before"), _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        moved_m = float(np.linalg.norm(after.position - before.position))
        hand_moving = moved_m > M11_HAND_STILL_M
        diff = None
        if not hand_moving:
            rgb0, rgb1 = _camera_rgb(before, self.camera), _camera_rgb(after, self.camera)
            hand = _hand_mask(self.hand_mask_fn, rgb0, rgb1, self.region.shape)
            hand |= _static_mask(self.static_hand_mask_fn, after, self.region.shape)
            ok = self.region & ~hand
            diff = float(np.abs(_gray(rgb1) - _gray(rgb0))[ok].mean()) if ok.any() else None
        self.diffs.append(diff)
        if diff is None:
            self.run = []
        else:
            self.run.append(diff)
        recent = self.run[-M11_FRAMES:]
        evidence = {"diff_grey": diff, "recent": list(recent), "threshold": self.th.value, "frames": self.count,
                    "measured_run": len(self.run), "hand_moving": hand_moving, "grip_moved_m": moved_m}
        ths = (self.th, self.still_th)
        if len(recent) < M11_FRAMES:
            return _report(self.name, VERDICT_PENDING, None, evidence, ths, step, len(recent) / M11_FRAMES)
        if max(recent) > self.th.value:
            return _report(self.name, VERDICT_VIOLATED, "object_moved_after_release", evidence, ths, step)
        return _report(self.name, VERDICT_MET, "at_rest", evidence, ths, step, 1.0)


# ========================================================================================= SpringBackObserver
class SpringBackObserver:
    """K9 spring-back after release (start it after the retreat, with the hand still): over K9_FRAMES frames the
    part's apparent rotation <= max(K9_FLOOR_DEG, K9_SIGMA_K sigma_deg) and its position change <= max(K9_FLOOR_M,
    K9_SIGMA_K sigma_m) -> met; else violated ``spring_back``. ``patch_or_axis`` = {"uv", optional "camera"
    ("external" default | "wrist"), "radius_px", "axis_uv" (structure-axis mode), "px_per_m" (image scale at the
    part, for the position test; without it only the rotation is tested)} or {"unmeasurable": reason}."""

    name = "spring_back"

    def __init__(self, *, patch_or_axis: dict, robot: RobotModel, axis_fn: Callable | None = None):
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        if not isinstance(patch_or_axis, Mapping):
            raise ValueError("patch_or_axis must be a dict")
        self.spec = dict(patch_or_axis)
        self.camera = self.spec.get("camera", "external")
        if self.camera not in ("external", "wrist"):
            raise ValueError("camera must be 'external' or 'wrist'")
        px_per_m = self.spec.get("px_per_m")
        self.px_per_m = None if px_per_m is None else _finite(px_per_m, "px_per_m")
        self.robot = robot
        self.axis_fn = axis_fn
        self.started = False

    def _gray(self, view):
        return view.wrist_gray_upright if self.camera == "wrist" else _gray(view.external_rgb_upright)

    def start(self, view) -> None:
        view = _view(view)
        self.core = _PatchRotation(self._gray(view), self.spec, self.spec.get("axis_uv"), self.axis_fn)
        self.rot = [0.0]
        self.pos_px = [0.0]
        self.count = 0
        self.lost = 0
        self.started = True

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("SpringBackObserver.start(view) must be called first")
        after = _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        if self.core.mode is None:
            return _report(self.name, VERDICT_FAILED, "unmeasurable", {"unmeasurable": self.core.reason}, (), step)
        angle, uv, quality = self.core.measure(self._gray(after))
        if angle is None:
            self.lost += 1
            if self.lost >= K7_LOST_FRAMES:
                return _report(self.name, VERDICT_FAILED, "unmeasurable", {"unmeasurable": "lost"}, (), step)
        else:
            self.lost = 0
            self.rot.append(float(angle))
            self.pos_px.append(float(np.linalg.norm(np.asarray(uv) - self.core.uv0)))
        sigma_deg = _jitter_sigma(self.rot)
        rot_th = th_max("K9_change_deg", Term("DD", K9_FLOOR_DEG, "K9_FLOOR_DEG"),
                        Term("ONLINE", K9_SIGMA_K * sigma_deg, "K9_SIGMA_K*sigma_deg"), unit="deg")
        ths = [rot_th]
        change_deg = max(abs(v) for v in self.rot)
        change_m = None
        pos_ok = True
        if self.px_per_m:
            sigma_m = _jitter_sigma(self.pos_px) / self.px_per_m
            pos_th = th_max("K9_change_m", Term("DD", K9_FLOOR_M, "K9_FLOOR_M"),
                            Term("ONLINE", K9_SIGMA_K * sigma_m, "K9_SIGMA_K*sigma_m"), unit="m")
            ths.append(pos_th)
            change_m = max(self.pos_px) / self.px_per_m
            pos_ok = change_m <= pos_th.value
        evidence = {"mode": self.core.mode, "camera": self.camera, "change_deg": change_deg, "change_m": change_m,
                    "frames": self.count, "quality": quality, "position_unmeasured": self.px_per_m is None}
        if change_deg > rot_th.value or not pos_ok:
            return _report(self.name, VERDICT_VIOLATED, "spring_back", evidence, ths, step)
        if self.count < K9_FRAMES:
            return _report(self.name, VERDICT_PENDING, None, evidence, ths, step, self.count / K9_FRAMES)
        return _report(self.name, VERDICT_MET, None, evidence, ths, step, 1.0)


# ========================================================================================= HoldFloorInvariant
class HoldFloorInvariant:
    """G6 while holding (bus): violated ``hold_lost`` when the finger gap < ``floor.value``
    (primitives.g6_hold_floor), met otherwise."""

    name = "hold_floor"

    def __init__(self, *, floor: Threshold):
        if not isinstance(floor, Threshold):
            raise ValueError("floor must be a Threshold (primitives.g6_hold_floor)")
        self.floor = floor
        self.started = False

    def start(self, view) -> None:
        _view(view)
        self.count = 0
        self.started = True

    def update(self, before, action, after, context=None) -> ObsReport:
        if not self.started:
            raise RuntimeError("HoldFloorInvariant.start(view) must be called first")
        after = _view(after, "after")
        self.count += 1
        step = _step(context, self.count)
        evidence = {"gap_mm": after.gap_mm, "floor_mm": self.floor.value}
        if after.gap_mm < self.floor.value:
            return _report(self.name, VERDICT_VIOLATED, "hold_lost", evidence, (self.floor,), step)
        return _report(self.name, VERDICT_MET, None, evidence, (self.floor,), step)


# ============================================================================================ InvariantMonitor
class InvariantMonitor:
    """The relation-level invariant bus (§C3.3, §C3.5). ``observe`` runs after every primitive transition and
    passes the context {"primitive_feedback", "primitive", "relation", "held"} (plus "step", the bus transition
    count unless the caller gives one) to every registered observer. An observer registered with ``preempt=True``
    whose verdict turns ``violated`` queues ONE preemption report (at most one queued per key); the queue is peeked,
    never consumed, until ``take_preemption`` [C-52]. Delivery re-arms the key: a violation that persists queues a
    fresh report at the next ``observe``, so a still-violated invariant also preempts the recovery's next primitive.
    A recovery that must act while an invariant is violated (e.g. release after ``hold_lost``) removes it, or clears
    its scope, before yielding that primitive. ``preempt=False`` observers only report (``snapshot``). Scopes:
    "relation" (cleared on every relation end), "held" (cleared on release), "episode"; ``clear("episode")`` removes
    everything. Removing an observer drops its undelivered reports."""

    def __init__(self):
        self._entries: dict[str, dict] = {}
        self._queue: list[tuple[str, ObsReport]] = []
        self.transitions = 0
        self.preemptions = 0
        self._changes: list[dict] = []

    def add(self, key: str, observer, *, view: StepView, preempt: bool = True, scope: str = "relation") -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty string")
        if key in self._entries:
            raise ValueError(f"observer {key!r} is already registered (one owner each, §C10.1)")
        if scope not in MONITOR_SCOPES:
            raise ValueError(f"scope must be one of {MONITOR_SCOPES}")
        if not callable(getattr(observer, "update", None)) or not callable(getattr(observer, "start", None)):
            raise ValueError("observer must implement start(view) and update(before, action, after, context)")
        observer.start(_view(view))
        self._entries[key] = {"observer": observer, "preempt": bool(preempt), "scope": scope, "last": None,
                              "rearmed": False}

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)
        self._queue = [(k, r) for k, r in self._queue if k != key]

    def clear(self, scope: str) -> None:
        if scope not in MONITOR_SCOPES:
            raise ValueError(f"scope must be one of {MONITOR_SCOPES}")
        for key in [k for k, e in self._entries.items() if scope == "episode" or e["scope"] == scope]:
            self.remove(key)

    def keys(self, scope: str | None = None) -> list[str]:
        return [k for k, e in self._entries.items() if scope is None or e["scope"] == scope]

    def observer(self, key: str):
        return self._entries[key]["observer"]

    def observe(self, before, action, after, context=None) -> list[ObsReport]:
        self.transitions += 1
        ctx = dict(context or {})
        ctx.setdefault("step", self.transitions)
        reports, self._changes = [], []
        for key, entry in list(self._entries.items()):
            report = entry["observer"].update(before, action, after, ctx)
            previous = entry["last"]
            entry["last"] = report
            reports.append(report)
            changed = previous is None or (previous.verdict, previous.reason) != (report.verdict, report.reason)
            if changed:
                self._changes.append({"key": key, "name": report.name, "verdict": report.verdict,
                                      "reason": report.reason, "progress": report.progress,
                                      "thresholds": thresholds_dicts(report.thresholds)})
            if entry["preempt"] and report.verdict == VERDICT_VIOLATED and (changed or entry["rearmed"]) \
                    and not any(k == key for k, _ in self._queue):
                self._queue.append((key, report))
                entry["rearmed"] = False
        return reports

    def last_changes(self) -> list[dict]:
        """``observer`` trace events of the last ``observe``: verdict changes (key, name, verdict, reason, progress,
        thresholds)."""
        return [dict(change) for change in self._changes]

    def peek_preemption(self) -> tuple[str, ObsReport] | None:
        return self._queue[0] if self._queue else None

    def take_preemption(self) -> tuple[str, ObsReport] | None:
        """Removes the delivered report and re-arms its key (see the class docstring)."""
        if not self._queue:
            return None
        self.preemptions += 1
        key, report = self._queue.pop(0)
        if key in self._entries:
            self._entries[key]["rearmed"] = True
        return key, report

    def snapshot(self) -> dict:
        out = {}
        for key, entry in self._entries.items():
            last = entry["last"]
            out[key] = {"verdict": None if last is None else last.verdict,
                        "reason": None if last is None else last.reason,
                        "progress": None if last is None else last.progress, "preempt": entry["preempt"],
                        "scope": entry["scope"]}
        return jsonable(out)


# =============================================================================================== StaysChecker
class StaysChecker:
    """Spec §3.3 stays / hold_true (§C3.7): entries run in registration order at every relation boundary; a
    re-registered key replaces its entry (moved to the end). ``check_all`` returns [(key, ObsReport)] and keeps the
    last result per key."""

    def __init__(self):
        self._entries: dict[str, dict] = {}

    def add(self, key: str, check: Callable[[StepView], ObsReport], *, step_i: int | None, kind: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a nonempty string")
        if not callable(check):
            raise ValueError("check must be callable")
        if kind not in STAYS_ENTRY_KINDS:
            raise ValueError(f"kind must be one of {STAYS_ENTRY_KINDS}")
        if step_i is not None and (isinstance(step_i, bool) or int(step_i) != step_i or step_i < 0):
            raise ValueError("step_i must be a nonnegative integer or None")
        self._entries.pop(key, None)
        self._entries[key] = {"check": check, "step_i": None if step_i is None else int(step_i), "kind": kind,
                              "last": None}

    def remove(self, key: str) -> None:
        self._entries.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._entries)

    def entry(self, key: str) -> dict:
        e = self._entries[key]
        return {"key": key, "step_i": e["step_i"], "kind": e["kind"],
                "last": None if e["last"] is None else e["last"].to_dict()}

    def check_all(self, view: StepView) -> list[tuple[str, ObsReport]]:
        view = _view(view)
        results = []
        for key, entry in list(self._entries.items()):
            report = entry["check"](view)
            if not isinstance(report, ObsReport):
                raise TypeError(f"stays check {key!r} must return an ObsReport")
            entry["last"] = report
            results.append((key, report))
        return results
