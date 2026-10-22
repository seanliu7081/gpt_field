"""agent_SpeedUp_v2 vision measurement (contract §C5; spec §4.5, §3.6 items 2 and 6, D1-D4, D14, D15, V1, V3, L1).

Turns wrist and external images plus the robot's own measured motion (state8) into metric measurements. No
HeightField policy and no registration (those are the scene's); nothing here moves the robot.

- ``FeatureTracker``: cv2 TM_CCOEFF_NORMED template tracks with the v1 NCCTrack quadratic subpixel peak (V7), search
  windows centred on a predicted shift.
- ``SurveyRecorder`` (a ProbeRecorder): tracks wrist features on every step of the probe stages with the shift
  predicted from the MEASURED hand motion and the running per-feature kappa (no depth prior of any kind), keeps the
  open / closed / static frames and the external hand motion of every probe move, and produces a ``SurveyResult``:
  per-feature kappa (D1), f-free looming depth and f-hat (D3, [C-35]) and, given a camera model, depth samples (D4).
- ``fit_feature_kappa`` / ``depth_from_kappa`` / ``loom_depth``: the per-feature pinhole models. Both model the
  measured camera motion exactly (in-plane d_a, d_b along R1, R0 and axial d_z along R2), so the 3.14 deg start tilt
  (2.19 mm of in-plane motion in a 4 cm vertical probe) is not mistaken for looming.
- ``CameraModel`` (R3 under R-2f, cross-checked by f-hat), ``ExternalModel`` (2x3 J_ext, the image direction v_hat of
  world down, n_hat, the viewing ray), ``HandLocator`` (absolute external grip pixel, L1) and ``ExternalProjector``
  (affine projection anchored on a located grip, re-anchorable [C-51]); ``refine_region_mask`` (Lab region growing).
- Image helpers: ``hand_motion_mask`` (the v1 hand-motion algorithm, returning the mask), ``frame_alignment_turns``
  (§3.6 item 2), ``structure_axis`` (the v1 structure tensor, returning the orientation), ``reacquire`` (V3),
  ``to_lab255`` [C-20], ``fingertip_band_mask``, ``false_change_map``.

Measured refinements beyond the contract text (each from the fake world or the task-2 survey probe):
- templates are resampled at the predicted looming scale during axial moves (fixed templates biased looming depth
  by 1-2.5 percent toward the patch's contrast centroid);
- edge- and ridge-like templates are aperture-weighted in the kappa fit (``template_aperture``): a match slides along
  an edge, ring or ridge;
- D4's sigma_kappa is the larger of the residual standard error (with a lag-1 autocorrelation inflation) and the
  probe legs' kappa scatter (out-and-back legs share a position-repeating tracking bias the residuals cannot see);
- J_ext is fitted on the HAND's own image motion (``hand_band_shift``), not on v1's fastest-decile flow (arm links
  that move ~20 percent faster in the image than the grip); the HandLocator's fingertip is the extreme of the
  undilated, motion-consistent frame-difference outline (``motion_outline``, ``outline_tip_before``) with a
  finger-footprint offset model (``fingertip_offset_px``), not the extreme of the dilated fastest-decile mask;
- the HandLocator refuses frame pairs outside its validated conditions (``HandLocator.motion_check``: hand image
  motion within L1_MAX_ACROSS_DEG of v_hat and at least L1_MIN_ALONG_PX along it; a 1-step pair of a probe leg gave
  100-150 px errors at task-2 hovers), and its sigma carries an absolute model term (L1_MODEL_SIGMA_PX) next to the
  displacement-consistency term, which cancels any offset common to all locations;
- ``refine_region_mask`` caps its margin, uses a multi-modal Lab colour model of an eroded / seed-centred core, and
  returns the affine mask when the refined area leaves [REGION_MIN_AREA_RATIO, REGION_MAX_AREA_RATIO] x the affine
  area (a large projector sigma flooded 38 percent of the image);
- D4_MODEL_SIGMA_M is a diagnosis only (design review 1 ruling 4b): it is never added to DepthSample.sigma_z; the
  model error of a part top is applied once, at part level, by the scene (TOP_MODEL_M).

Design review 1 (decision §2.3, measurement remedy R):
- probe-tracker mode of ``SurveyRecorder`` (G-1, G-2): seeds placed EXACTLY (no relocation to the most textured
  patch), a dense lattice every FT_DENSE_STEP_PX within FT_DENSE_WINDOW_PX of the pointed pixel (keys "dense:<k>"),
  11 px templates (FT_PROBE_RADIUS_PX), the band-passed grey (``bandpass_grey``) with D15_PROBE_MIN_STD and an NCC
  floor of V1_PROBE_MIN_PEAK. Hover ParallaxProbes pass ``probe_tracker=True, dense_uv=<fine point>``; Calibrate+
  passes ``probe_tracker=True`` (grid only);
- compute (G-3): always (exact, the same tracks to 1e-9 px): the per-feature matchTemplate calls of a frame shared
  between the calling thread (FT_MATCH_MAIN_SHARE) and a FT_MATCH_WORKERS-thread pool (``match_scores``), the
  scaled-template warp inputs computed once per feature, a per-state HandPose cache, a finite-only ``nan_to_num``, the
  vectorised sub-pixel peak (``subpixel_peaks``), shift prediction, derotation and kappa design; opt-in
  (``adaptive_search=True``; review finding R2, design review 2 decision §5: signed off as off by default and NOT
  adopted, turning it on needs an RS-8 with off-part parity): a feature with a running kappa searched
  +-FT_TRACKED_SEARCH_PX around its prediction;
- L1 with few locatable moves (G-5): with n <= L1_MIN_SAMPLES located moves the consistency bound is the absolute
  sqrt 2 x L1_SIGMA_K x L1_FLOOR_PX (12.7 px; the self-scaled bound accepted a 40 px fault in 15/15 cases), and
  two agreeing locations still fit, flagged ``degraded`` (outcome reason calibration_degraded).

Design review 2 (decision §2.2):
- G-7 orientation kernels for the scene's D10-OE (S-11): ``orientation_peak`` (doubled-angle 1 deg histogram,
  ORIENT_SMOOTH_DEG circular smoothing, the peak refined within YAW_GRID_RAD / 2; share and second peak),
  ``outline_orientation`` (the largest outline back-projected at the top, OUTLINE_TANGENT_PTS chords, two_sided) and
  ``edge_orientation`` (Sobel of the D10_EDGE_BLUR_PX-blurred luma, the top D10_EDGE_TOP_FRACTION in an ROI minus an
  exclusion), with ``backproject_at_z`` / ``image_directions_at_z``; the lens's numbers are reproduced on the judge's
  466 live stages;
- G-8 HandLocator V2v: the across coordinate of ``locate`` is the validated finger pair (``finger_pair_across``,
  ``finger_band_depth_px``, ``finger_extremes_px``), refusing instead of reading the hand body.

Frames (contract §C1.3): state8 = [grip xyz, right_hand rotvec, q1, q2]; R columns R0 (image up), R1 (jaw, image
right), R2 (approach = optical axis); camera C = grip - 0.097 R2 + 0.05 R0. Upright wrist pixels: u right, v down,
centred principal point; centred coordinates u' = u - c_u, v' = c_v - v; kappa = f / z px per metre of in-plane
translation (J = -kappa P, P = diag(1, -1)).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os

import cv2
import numpy as np

from . import agent_speedup_v2_robot as robot
from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (DENSE_FEATURE_KEY_PREFIX, YAW_GRID_RAD, DepthSample, FeatureRecord, HandPose, Meas,
                                     RobotModel, SurveyResult, Term, WRIST_IMAGE_PX, WristCamera, jsonable, th_max,
                                     th_value)
from .contact_skills import StepView
from .external_axes import fit_external_axes
from .mark_prompting import gravity_quarter_turns
from .wrist_servo import NCCTrack

GEOMETRY_VERSION = "agent-speedup-v2-geometry-4"

# ---------------------------------------------------------------------------------------------- thresholds (§C5)
D1_MIN_SHIFT_PX = 6.0
D1_RESIDUAL_FRACTION = 0.1
D1_RESIDUAL_FLOOR_PX = 0.6
D1_LEG_AGREEMENT = 0.35
D1_MIN_SAMPLES = 4
D1_LEG_SPAN_DEG = 30.0
D1_MIN_APERTURE = 0.01
D1_ANISOTROPY_RATIO = 0.1
D1_LEG_MIN_INFORMATION = 0.25
D3_AGREE_FLOOR = 0.05
D3_SIGMA_K = 3.0
D3_MIN_FEATURES = 3
D4_PIXEL_NOISE_PX = 1.5
D4_MAX_AUTOCORRELATION = 0.9
D14_MAX_RESIDUAL_DEG = 30.0
D14_MIN_MOTION_M = 0.02
D14_MIN_IMAGE_PX = 3.0
D14_MAX_CONDITION = 50.0
D14_MIN_SAMPLES = 4
D15_MIN_STD_GREY = 4.0
V1_MIN_PEAK = 0.6
V1_LOST_FRAMES = 2
V3_HALF_PX = 48
V3_MIN_PEAK = 0.6
V3_SIGMA_K = 3.0
V3_SCALES = (1.08, 1.16, 1.25, 0.93, 0.86)
L1_FLOOR_PX = 3.0
L1_SIGMA_K = 3.0
L1_TIP_BAND_PX = 8.0
L1_MIN_SAMPLES = 3
L1_MAX_ACROSS_DEG = 30.0
L1_MIN_ALONG_PX = 8.0
L1_MOTION_AGREEMENT = 0.5
L1_MODEL_SIGMA_PX = 5.5
D4_MODEL_SIGMA_M = 0.009
RAYLEIGH_MEDIAN_TO_RMS = 1.0 / math.sqrt(math.log(2.0))
EXTERNAL_IMAGE_PX = 512
HL_HAND_BAND_M = 0.08
HL_MIN_BLOB_PX = 10
FT_RADIUS_PX = 10
FT_SEARCH_PX = 32
FT_GRID_SPACING_PX = 36
FT_MAX_FEATURES = 150
FT_MIN_SCALE = 0.7
FT_SCALE_TOL = 0.01
RELIEF_OFFSET_M = 0.015
RELIEF_SEARCH_PX = 8
HM_DIFF_GREY = 12
HM_TOP_FRACTION = 0.10
HM_MIN_PIXELS = 50
HM_MIN_FLOW_PX = 1.0
HM_FARNEBACK = (0.5, 5, 21, 5, 7, 1.5)
HM_DILATE_PX = 9
V5_TENSOR_HALF_PX = 12
BROYDEN_MIN_STEP_M = 0.0005
BROYDEN_MAX_INNOVATION = 0.5
BROYDEN_INNOVATION_FLOOR_PX = 3.0
PROJECTOR_AFFINE_FRACTION = 0.2
REGION_LAB_FLOOR = 12.0
REGION_LAB_BIN = 6.0
REGION_MODE_FRACTION = 0.03
REGION_MODE_RELATIVE = 0.1
REGION_CORE_RADIUS_PX = 12
REGION_MAX_MARGIN_PX = 48
REGION_MAX_AREA_RATIO = 4.0
REGION_MIN_AREA_RATIO = 0.6
REGION_CLOSE_PX = 7
FINGERTIP_BAND_MARGIN_PX = 4
FINGERTIP_BAND_OPEN_PX = 5
RUNNING_KAPPA_MIN_BASELINE_M = 0.002
FT_PROBE_RADIUS_PX = 5
FT_DENSE_STEP_PX = 4
FT_DENSE_WINDOW_PX = 40
FT_BANDPASS_SIGMAS = (0.8, 3.0)
FT_BANDPASS_GAIN = 4.0
FT_BANDPASS_MID_GREY = 128.0
D15_PROBE_MIN_STD = 1.0
V1_PROBE_MIN_PEAK = 0.9
FT_TRACKED_SEARCH_PX = 12
FT_MATCH_WORKERS = 8
FT_MATCH_MAIN_SHARE = 0.3
ORIENT_SMOOTH_DEG = 3.0
OUTLINE_TANGENT_PTS = 4
D10_EDGE_TOP_FRACTION = 0.25
D10_EDGE_BLUR_PX = 1.0
OUTLINE_SIDE_MIN_PTS = 3
EDGE_MIN_ROI_PX = 20
EDGE_MIN_VOTES = 10
MAD_TO_SIGMA = 1.4826
GREY_WEIGHTS = (0.299, 0.587, 0.114)
RGB_MAX = 255.0
LAB_L_SCALE = 2.55
LAB_AB_OFFSET = 128.0
SUBPIXEL_PINV = np.linalg.pinv(np.array([[1.0, x, y, x * x, x * y, y * y] for y in (-1, 0, 1) for x in (-1, 0, 1)]))

LABELS = {
    "D1_MIN_SHIFT_PX": ("DD", "D1 kappa probe validity: the largest feature shift must reach 6 px (v1 Align "
                              "ACROSS_PROBE_MIN_SHIFT_PX, wrist_servo.py:517)"),
    "D1_RESIDUAL_FRACTION": ("DD", "D1: fit residual below 0.1 x the rms shift (v1 Align ACROSS_PROBE_MAX_RESIDUAL, "
                                   "wrist_servo.py:518)"),
    "D1_RESIDUAL_FLOOR_PX": ("DD", "D1: residual floor 0.6 px (v1 Align across-bar fit, wrist_servo.py:1101)"),
    "D1_LEG_AGREEMENT": ("DD", "D1: per-leg kappa estimates agree within 35 percent (v1 Align "
                               "ACROSS_PROBE_LEG_AGREEMENT, wrist_servo.py:519)"),
    "D1_MIN_SAMPLES": ("DD", "D1: at least 4 tracked probe frames per feature fit (v1 Align "
                             "ACROSS_PROBE_MIN_SAMPLES)"),
    "D1_LEG_SPAN_DEG": ("DD", "D1 / D4: probe legs within 30 deg of one axis (out and back) count as one direction "
                              "(v1 external_axes MIN_SAMPLE_ANGLE_DEG: directions 30 deg apart span the plane)"),
    "D1_MIN_APERTURE": ("DD", "D1 aperture weighting: a template's weakest direction keeps at least 1 percent of its "
                              "strongest eigenvalue (a 0.1 amplitude weight; a perfect edge would otherwise be singular)"),
    "D1_ANISOTROPY_RATIO": ("DD", "D1 aperture weighting applies only to edge- or ridge-like templates whose weak "
                                  "eigenvalue is below 0.1 of the strong one (object edges, ridges, rings: 0.001-0.05 on the task-2 "
                                  "start view; textures 0.1-0.5); 2-D textures are fitted unweighted"),
    "D1_LEG_MIN_INFORMATION": ("DD", "D1 leg agreement: a probe leg whose motion lies along a template's weak direction "
                                     "(less than a quarter of the information of an unweighted leg) cannot measure "
                                     "kappa and is left out of the agreement (task-2 hover: legs along the ridge or "
                                     "the region part's rings read 0.1-0.2 of the true kappa)"),
    "D3_AGREE_FLOOR": ("DD", "D3 camera model: f-hat agrees with R3 within max(5 percent, 3 sigma) (spec §4.5)"),
    "D3_SIGMA_K": ("DD", "D3: the 3 sigma term of the f-hat agreement band (spec §4.5 CameraModel rule)"),
    "D3_MIN_FEATURES": ("DD", "D3: f-hat needs looming depth on at least 3 independent features (a median of fewer "
                              "is one feature's error)"),
    "D4_PIXEL_NOISE_PX": ("DD", "D4 pixel noise 1.5 px (v1 wrist_servo PIXEL_NOISE_PX, real-sim fit residual); "
                                "used for two-frame looming depth, where no residual exists"),
    "D4_MAX_AUTOCORRELATION": ("DD", "D4 sigma_kappa: the lag-1 autocorrelation of consecutive-frame residuals is "
                                     "clipped at 0.9 (variance inflation at most 19x) so a smooth residual trend "
                                     "cannot claim an unbounded sigma"),
    "D14_MAX_RESIDUAL_DEG": ("DD", "D14 J_ext residual: angle between J dp and the measured hand image shift <= 30 "
                                   "deg (v1 external_axes MAX_RESIDUAL_DEG, external_axes.py:31)"),
    "D14_MIN_MOTION_M": ("DD", "D14: a usable probe move covers >= 2 cm (v1 external_axes MIN_SAMPLE_MOTION_M)"),
    "D14_MIN_IMAGE_PX": ("DD", "D14: a usable probe move shifts the hand image >= 3 px (v1 external_axes "
                               "MIN_IMAGE_MOTION_PX)"),
    "D14_MAX_CONDITION": ("DD", "D14: probe displacements must span 3-D with condition <= 50 (v1 wrist_servo "
                                "MAX_CONDITION)"),
    "D14_MIN_SAMPLES": ("DD", "D14: a corrupted probe move is dropped only when at least 4 moves remain (3 are the "
                              "minimum for a 2x3 fit; one more checks it)"),
    "D15_MIN_STD_GREY": ("DD", "D15 feature texture test: patch std >= 4 grey on a 21 px patch (v1 NCCTrack "
                               "min_std, wrist_servo.py:134)"),
    "V1_MIN_PEAK": ("DD", "V1: an NCC match below peak 0.6 is a miss (v1 NCCTrack min_peak, wrist_servo.py:129)"),
    "V1_LOST_FRAMES": ("DD", "V1: a feature is lost after 2 consecutive misses (v1 Align LOST_FRAMES)"),
    "V3_HALF_PX": ("DD", "V3 re-acquisition search +-48 px around the prediction (spec §5.2 V3)"),
    "V3_MIN_PEAK": ("DD", "V3: re-acquired match peak >= 0.6 (spec §5.2 V3, v1 NCCTrack min_peak)"),
    "V3_SIGMA_K": ("DD", "V3: the re-acquired point lies within 3 sigma of the prediction (spec §5.2 V3)"),
    "V3_SCALES": ("DD", "V3: extra template scales tried around the kappa ratio (v1 NCCTrack REACQUIRE_SCALES)"),
    "L1_FLOOR_PX": ("DD", "L1 hand locator consistency floor 3 px (spec §5.2 L1)"),
    "L1_SIGMA_K": ("DD", "L1: displacement residuals within 3 sigma; also the mask dilation of refined region masks "
                         "in units of the projector sigma (contract [C-51])"),
    "L1_TIP_BAND_PX": ("DD", "L1: the fingertip pixel is the across-centroid of the outline pixels within 8 px of the "
                             "extreme along v_hat, so both fingertips count (task-2 agentview, BG survey probe: with "
                             "3 px one finger was often alone in the band, 17 px across error; 6-10 px kept both)"),
    "L1_MAX_ACROSS_DEG": ("DD", "L1: a probe move locates the hand only when its image motion is within 30 deg of "
                                "v_hat (moves across v_hat smear the hand band's centroid by up to half the motion: "
                                "8-16 px across error on the task-2 agentview y probes)"),
    "L1_MIN_SAMPLES": ("DD", "L1: at least 3 located probe moves (two displacements) to validate the locator; with at "
                             "most this many the consistency bound is the absolute sqrt 2 x L1_SIGMA_K x L1_FLOOR_PX = "
                             "12.7 px, and 2 agreeing locations fit flagged degraded (design review 1 decision §2.3 "
                             "G-5, rulings item 5: the self-scaled bound accepted a 40 px fault in 15/15 cases, the "
                             "absolute bound dropped it in 15/15 and kept clean data in 5/5)"),
    "L1_MIN_ALONG_PX": ("DD", "L1 precondition of fit and locate: the hand's image motion along v_hat must reach "
                              "the tip band L1_TIP_BAND_PX (8 px), so the before and after fingertips are separate "
                              "extremes (BG fix survey probe, 200 hovers x every step of a 3 cm vertical move: tip "
                              "error median 35 px / max 143 px below 4 px of motion, max 122 px at 4-8 px, median "
                              "4.5-5.3 px / max 12 px from 8 px on; the 1-step pair of a probe leg moves 0.7-1.1 px; "
                              "the 4 cm x probe of Calibrate+ moves 11 px)"),
    "L1_MOTION_AGREEMENT": ("DD", "L1 locate: the measured hand motion replaces J dp only when its length agrees "
                                  "within 50 percent (and its direction within D14's 30 deg)"),
    "L1_MODEL_SIGMA_PX": ("DD", "L1 absolute model error of a located grip (radial rms against the evaluator grip "
                                "projection, which the displacement-consistency sigma cannot see: it cancels a common "
                                "offset): 5.4-5.5 px over 200 task-2 hover locations on a 3 cm vertical move (median "
                                "4.8-4.9 px, max 10 px; parts C and E 7-10 px), 3.3 px on the Calibrate+ moves; BG fix "
                                "survey probe runs bgfix_*_001; re-derive at V5"),
    "D4_MODEL_SIGMA_M": ("DD", "diagnosis only (design review 1 decision §2.3 G-4, ruling 4b; value unchanged, never "
                               "applied): the model error of a part top is the scene's TOP_MODEL_M (part level, 3 mm, "
                               "5 mm below TOP_BAND_SUPPORT_N band members). The number is the per-sample term that "
                               "brought the robust z-score std of the grip parts' samples to 1 in the BG fix survey "
                               "probe grip_settled_001 (A 0, C 9.05 mm, E 4.3 mm, body 1.5 mm, B 0); per sample it would "
                               "widen the D8 bands to about 27 mm and merge the planes [C-7] keeps apart"),
    "FT_PROBE_RADIUS_PX": ("DD", "probe tracker template radius 5 px = 11 px templates (design review 1 decision §3, "
                                 "§2.3 G-1): 7-11 px templates attach within 0.2 mm on the task-2 thin grip part; 21 px "
                                 "templates lag about 3 px on its tube sides (+10 to +12 mm)"),
    "FT_DENSE_STEP_PX": ("DD", "probe tracker dense lattice spacing 4 px (decision §3, §2.3 G-1): a 6 px lattice "
                               "breaks the task-2 thin grip part at yaw -90 deg (p90 23.8 mm)"),
    "FT_DENSE_WINDOW_PX": ("DD", "probe tracker dense lattice radius 40 px around the pointed pixel (decision §3, = the "
                                 "scene's TOP_WINDOW_PX): about 1.2-2 cm at hover depth, the band-top window"),
    "FT_BANDPASS_SIGMAS": ("DD", "probe tracker band-pass grey: difference of Gaussians of sigma 0.8 px and 3.0 px "
                                 "(decision §3, §2.3 G-2; remedy 1 prefilter comparison on the task-2 thin grip part): "
                                 "removes view-fixed shading and keeps the template-scale texture"),
    "FT_BANDPASS_GAIN": ("DD", "probe tracker band-pass gain 4 (decision §3 FT_BANDPASS_SIGMAS: '(0.8, 3.0) px, gain "
                               "4'): the difference of Gaussians of a few-grey texture is scaled into the grey range"),
    "FT_BANDPASS_MID_GREY": ("DD", "the band-passed grey's zero level, 8-bit mid grey 128 (decision §2.3 G-2: 128 + 4 "
                                   "(G(0.8) - G(3.0)))"),
    "D15_PROBE_MIN_STD": ("DD", "probe tracker D15: band-passed 11 px patch std >= 1.0 grey (decision §3, §2.3 G-1; "
                                "remedy 1 prefilter comparison); re-derive ONLINE (k x band-passed sigma_static) at V5"),
    "V1_PROBE_MIN_PEAK": ("DD", "probe tracker V1: an NCC match below peak 0.9 is a miss (decision §3, §2.3 G-1): with "
                                "dense seeds the task-2 thin grip part's band-top p90 error 12.4 -> 3.2 mm (a template "
                                "straddling a depth step deforms under parallax and is dropped)"),
    "FT_TRACKED_SEARCH_PX": ("DD", "OPT-IN adaptive search of the probe tracker (decision §3, §2.3 G-3; off by "
                                   "default since review finding R2: it cannot meet the |duv| <= 0.1 px acceptance "
                                   "and the exact threaded path meets the 40 ms bound): a feature with a running "
                                   "kappa is searched +-12 px around its kappa-predicted shift (per-step shifts <= 11 "
                                   "px are predicted by the running kappa; remedy 1 E1 on the probe tracker: the same "
                                   "accuracy over 100 hovers at half the time), the others +-FT_SEARCH_PX; design "
                                   "review 2 decision §3 / §5: opt-in and NOT adopted (turning it on needs an RS-8 "
                                   "with off-part parity)"),
    "FT_MATCH_WORKERS": ("DD", "compute only (decision §2.3 G-3 exact speed-up, review findings R2 / R6): the "
                               "matchTemplate calls of one tracked frame not run by the calling thread go as 8 "
                               "contiguous shares to an 8-thread pool (cv2 releases the GIL; every score map is the "
                               "same call on the same arrays, so the tracks equal the serial loop's exactly). "
                               "Recorded task-2 hovers, 468 features: serial 43 ms per frame; 2-16 threads alone "
                               "22-27 ms warm but 48-57 ms after 30-80 ms of idle (the simulator step); 8 with "
                               "FT_MATCH_MAIN_SHARE 25 / 25 / 28 ms at 0 / 30 / 80 ms idle"),
    "FT_MATCH_MAIN_SHARE": ("DD", "compute only (G-3, with FT_MATCH_WORKERS): the calling thread matches the first 30 "
                                  "percent of a frame's features itself, so pool threads woken on cores that idled "
                                  "during the simulator step are not the critical path (shares 0.0-0.4 tried: 0.3 "
                                  "gave 25-28 ms per frame at 0-80 ms idle, 0.0 gave 48 ms after idle, 0.4 34 ms)"),
    "ORIENT_SMOOTH_DEG": ("DD", "G-7 orientation_peak (design review 2 decision §2.2, §3; review-2 axis lens): the "
                                "doubled-angle orientation histogram of 1 deg bins is circularly smoothed by a Gaussian "
                                "of sigma 3 deg before its peak is taken; the peak is refined by the weighted mean "
                                "within YAW_GRID_RAD / 2. Fitted on the two thin grip parts of task 2: re-derive on the "
                                "cavity-edge and sliding-fixture regressions before M7"),
    "OUTLINE_TANGENT_PTS": ("DD", "G-7 outline_orientation (decision §2.2, §3; axis lens mask_outline): the tangent at "
                                  "an outline point is the chord between the outline points 4 steps before and after "
                                  "it (about +-4 px), back-projected at the measured top and length-weighted; a contour "
                                  "of fewer than 3 x 4 points has no orientation"),
    "D10_EDGE_TOP_FRACTION": ("DD", "G-7 edge_orientation (decision §2.2, §3; axis lens edge_hist): only the strongest "
                                    "quarter of the gradient magnitudes inside the ROI minus the exclusion vote"),
    "D10_EDGE_BLUR_PX": ("DD", "G-7 edge_orientation (decision §2.2, §3; axis lens edge_hist): the luma is blurred by a "
                               "Gaussian of sigma 1 px before the 3 x 3 Sobel gradient"),
    "OUTLINE_SIDE_MIN_PTS": ("DD", "G-7 outline_orientation two_sided (design review 2 decision §2.2, axis lens "
                                   "mask_outline): at least 3 outline points whose tangent lies in the peak window on "
                                   "EACH side of the axis line. A behavioural minimum, not a shape test: a half-disk "
                                   "is two-sided (its arc runs parallel to the flat edge near its apex) and an "
                                   "aspect-1.25 rectangle passes the outline gate (T1 pins both; review-2 "
                                   "implementation review R-9)"),
    "EDGE_MIN_ROI_PX": ("DD", "G-7 edge_orientation (axis lens edge_hist minimum): an ROI of fewer than 20 pixels "
                              "after the exclusion has no edge orientation"),
    "EDGE_MIN_VOTES": ("DD", "G-7 edge_orientation (axis lens edge_hist minimum): fewer than 10 voting edge pixels "
                             "(the strongest D10_EDGE_TOP_FRACTION of the ROI) give no edge orientation"),
    "RAYLEIGH_MEDIAN_TO_RMS": ("DD", "rms / median of the norm of a 2-D isotropic Gaussian error (Rayleigh): "
                                     "1 / sqrt(ln 2)"),
    "EXTERNAL_IMAGE_PX": ("ROBOT", "upright external (agentview) image size: M0 renders both cameras at one "
                                   "image_size, 512 px"),
    "HL_HAND_BAND_M": ("DD", "the hand's lowest 8 cm along world down (fingertip reach 9.3 mm R5 + palm 31 mm R6 + "
                             "4 cm of the hand body) is tracked for the hand's own image motion (6 cm was sensitive "
                             "to a 2 deg v_hat error on the task-2 agentview, 9 cm reached the arm links)"),
    "HL_MIN_BLOB_PX": ("DD", "hand locator: silhouette pieces of at least 10 px are kept (a fingertip seen from the "
                             "external camera can be a separate piece of a few tens of px; single-pixel noise is "
                             "not)"),
    "FT_RADIUS_PX": ("DD", "FeatureTracker template radius 10 px = 21 px patches (v1 NCCTrack radius)"),
    "FT_SEARCH_PX": ("DD", "FeatureTracker search +-32 px around the predicted position (v1 NCCTrack search)"),
    "FT_GRID_SPACING_PX": ("DD", "spec §4.5 auto-selected features on a 36 px grid"),
    "FT_MAX_FEATURES": ("DD", "spec §4.5: up to 150 auto-selected textured patches"),
    "FT_MIN_SCALE": ("DD", "FeatureTracker: templates are resampled down to 0.7 x (a 4 cm rise from 12 cm depth "
                           "shrinks the image to 0.75 x, the nearest surface D1 allows)"),
    "FT_SCALE_TOL": ("DD", "FeatureTracker: a predicted image scale change below 1 percent keeps the capture "
                           "template (0.01 x a 10 px half patch is 0.1 px, the V7 tolerance)"),
    "RELIEF_OFFSET_M": ("DD", "spec §4.5 relief probes on the part's principal axes at +-(w/2 + 15 mm)"),
    "RELIEF_SEARCH_PX": ("DD", "spec §4.5: the highest-contrast patch within +-8 px of a relief point"),
    "HM_DIFF_GREY": ("DD", "hand motion mask: changed pixels differ by > 12 grey (v1 external_axes DIFF_THRESHOLD; the "
                           "static scene differs by < 3); also the false-change and fingertip-band difference"),
    "HM_TOP_FRACTION": ("DD", "hand motion mask: the fastest decile of changed pixels is the hand (v1 TOP_FRACTION)"),
    "HM_MIN_PIXELS": ("DD", "hand motion mask: at least 50 moving changed pixels (v1 MIN_CHANGED_PIXELS)"),
    "HM_MIN_FLOW_PX": ("DD", "hand motion mask: moving pixels flow > 1 px (v1 MIN_FLOW_PX)"),
    "HM_FARNEBACK": ("DD", "Farneback parameters of v1 hand_image_motion (external_axes.py:51): pyramid scale, "
                           "levels, window, iterations, poly_n, poly_sigma"),
    "HM_DILATE_PX": ("DD", "hand motion mask: the changed mask is dilated by a 9 px square (external_axes.py:53)"),
    "V5_TENSOR_HALF_PX": ("DD", "structure tensor half size 12 px = 25 px patch (v1 BAR_CHECK_HALF_PX, "
                                "wrist_servo.py:77)"),
    "BROYDEN_MIN_STEP_M": ("DD", "J_ext Broyden secants only on steps >= 0.5 mm (v1 ImageJacobian.broyden "
                                 "min_step_m)"),
    "BROYDEN_MAX_INNOVATION": ("DD", "J_ext Broyden: an innovation above half the predicted motion is a tracking "
                                     "jump (v1 ImageJacobian.broyden max_innovation_fraction)"),
    "BROYDEN_INNOVATION_FLOOR_PX": ("DD", "J_ext Broyden innovation floor 3 px (v1 ImageJacobian.broyden)"),
    "PROJECTOR_AFFINE_FRACTION": ("DD", "affine external projection error is about 20 percent of the predicted shift "
                                        "15-30 cm from the anchor (contract [C-51] probe); re-anchor beyond that"),
    "REGION_LAB_FLOOR": ("DD", "region refinement: a pixel joins the region when its Lab colour (8-bit scaling "
                               "[C-20]) lies within 12 of a colour of the region's core (12 = the O1 / v1 change "
                               "floor)"),
    "REGION_LAB_BIN": ("DD", "region refinement: the core's colour model is a Lab histogram of 6-unit bins (half the "
                             "REGION_LAB_FLOOR radius, so the floor spans two bins)"),
    "REGION_MODE_FRACTION": ("DD", "region refinement: a colour belongs to the core's model when at least 3 percent of "
                                   "the core lies within REGION_LAB_FLOOR of it: a ringed or two-tone region keeps "
                                   "every tone (BG fix, task-2 agentview region B: refined IoU 0.65-0.73 against 0.46-"
                                   "0.50 for a single median colour and 0.54 affine), while a 2-3 percent sliver of "
                                   "background in the core (an offset just beyond 3 sigma) does not join (1 percent "
                                   "flooded a synthetic textured background)"),
    "REGION_MODE_RELATIVE": ("DD", "region refinement: a secondary colour also needs a tenth of the dominant colour's "
                                   "mass (BG fix: a seed on the shaded side of the fake slab's top face, an affine "
                                   "offset of 20 px, put the side into the model at 0.03 alone: IoU 0.77 against the "
                                   "top face, 0.81 with this; task-2 region B 0.65-0.70 against 0.48-0.57 affine)"),
    "REGION_CORE_RADIUS_PX": ("DD", "region refinement: when the eroded affine mask is too small for a colour model, "
                                    "the core is the seed-centred disk of 12 px radius (about 450 px, a 21 px "
                                    "template's area)"),
    "REGION_MAX_MARGIN_PX": ("DD", "region refinement: the affine mask grows (and the core erodes) by at most 48 px: a "
                                   "projector sigma of 50-57 px (the affine distance term far from the anchor, BG "
                                   "review: region B 0.37 m away) otherwise dilated it by 165 px and the region "
                                   "flooded 38 percent of the image; 48 px covers the measured affine errors (18-41 "
                                   "px for parts 27-29 cm from the start anchor, BG review locator probe)"),
    "REGION_MAX_AREA_RATIO": ("DD", "region refinement: a refined area above 4 x the affine area is a flood and the "
                                    "affine mask is returned (refinement corrects the position, not the size; the "
                                    "BG review flood was 45 x)"),
    "REGION_MIN_AREA_RATIO": ("DD", "region refinement: a refined area below 0.6 x the affine area is a partial capture "
                                    "(one tone of a two-tone region when a large sigma leaves only a seed core: 0.54 "
                                    "on the BG fix synthetic) and the affine mask is returned"),
    "REGION_CLOSE_PX": ("DD", "region refinement: grooves and specular bands inside a region are closed by a 7 px "
                              "disk (smaller than any region a part is registered from)"),
    "FINGERTIP_BAND_MARGIN_PX": ("DD", "fingertip band dilated by 4 px against finger-edge antialiasing"),
    "FINGERTIP_BAND_OPEN_PX": ("DD", "fingertip band: changed lines thinner than 5 px (object contours flickering under "
                                     "the 0.05 mm hand drift of a held close, task-2 BG survey probe) are opened away; "
                                     "the moving fingers are blobs tens of px wide"),
    "RUNNING_KAPPA_MIN_BASELINE_M": ("DD", "the search-window prediction uses a feature's running kappa only after 2 mm "
                                           "of measured in-plane motion (about 2 px at kappa 900 px/m; shorter "
                                           "baselines are tracking noise); before that the median running kappa"),
    "MAD_TO_SIGMA": ("DD", "normal-consistency factor of the median absolute deviation (robust sigma)"),
    "GREY_WEIGHTS": ("DD", "luma weights of the v1 grey images (contact_skills.StepView, external_axes._gray)"),
    "RGB_MAX": ("DD", "8-bit RGB full scale"),
    "LAB_L_SCALE": ("DD", "[C-20] OpenCV 8-bit Lab scaling: L x 255 / 100"),
    "LAB_AB_OFFSET": ("DD", "[C-20] OpenCV 8-bit Lab scaling: a + 128, b + 128"),
    "SUBPIXEL_PINV": ("DD", "pseudo-inverse of the 3x3 quadratic peak design (v1 NCCTrack._QUADRATIC_PINV)"),
}

# Stage classes of the probe primitives (§C8): gripper stages are never tracked (the fingers cross the image);
# vertical stages carry the looming signal; every other tracked stage with motion is an in-plane probe leg.
GRIPPER_STAGE_PREFIXES = ("close_", "open_", "yaw_", "release_")
VERTICAL_STAGE_PREFIXES = ("vertical_", "lift")
MOTION_STAGE_PREFIXES = ("probe_", "vertical_", "r1_", "r0_", "lift")
STATIC_STAGE = "open_hold"
CLOSED_STAGE = "close_2"
KAPPA_SOURCE = "kappa"
LOOM_SOURCE = "loom"
R3_SOURCE = "R3"
F_HAT_SOURCE = "f_hat"
# fit_feature_kappa's reject when a feature has fewer than the required probe frames; the kernel's FeatureRecord
# reasons (FEATURE_REJECTS, B0) have no such code, so SurveyRecorder.result records it as "lost" (when the feature
# was lost) or "shift_too_small" (the probe produced too little evidence)
TOO_FEW_SAMPLES = "too_few_samples"
# HandLocator.flags when fitted on two agreeing locations (G-5): the kernel's "calibrated" outcome reason
DEGRADED_FLAG = "calibration_degraded"


# ================================================================================================ small helpers
def _luma(rgb) -> np.ndarray:
    """Float grey of an RGB image (StepView.wrist_gray_upright weights); a 2-D input is returned as float."""
    image = np.asarray(rgb, dtype=float)
    if image.ndim == 2:
        return image
    return image[..., :3] @ np.asarray(GREY_WEIGHTS, dtype=float)


def _luma_u8(rgb) -> np.ndarray:
    """v1 external_axes._gray: uint8 grey by truncation."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB image")
    weights = GREY_WEIGHTS
    return (weights[0] * image[..., 0] + weights[1] * image[..., 1] + weights[2] * image[..., 2]).astype(np.uint8)


def _gray32(gray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim == 3:
        image = _luma(image).astype(np.float32)
    if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError("gray must be a finite 2-D image")
    return image


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if not norm > 1e-12:
        raise ValueError("zero vector")
    return v / norm


def _robust_sigma(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return MAD_TO_SIGMA * float(np.median(np.abs(values - np.median(values))))


def _weighted_median(values, weights) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cumulative = np.cumsum(w)
    return float(v[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])


def _stage_kind(stage: str) -> str:
    """'gripper' | 'vertical' | 'motion' | 'static' for a probe-primitive stage label."""
    if stage == STATIC_STAGE or stage.startswith(GRIPPER_STAGE_PREFIXES):
        return "gripper"
    if stage.startswith(VERTICAL_STAGE_PREFIXES):
        return "vertical"
    if stage.startswith(MOTION_STAGE_PREFIXES):
        return "motion"
    return "static"


def principal_point(size_px: int | None = None) -> np.ndarray:
    size = WRIST_IMAGE_PX if size_px is None else int(size_px)
    return np.array([(size - 1) / 2.0, (size - 1) / 2.0])


def centred(uv, size_px: int | None = None) -> np.ndarray:
    """Centred coordinates (u', v') = (u - c_u, c_v - v) of an upright pixel."""
    c = principal_point(size_px)
    uv = np.asarray(uv, dtype=float)
    return np.array([uv[0] - c[0], c[1] - uv[1]])


def uncentred(centred_uv, size_px: int | None = None) -> np.ndarray:
    c = principal_point(size_px)
    return np.array([c[0] + centred_uv[0], c[1] - centred_uv[1]])


def pixel_direction(pose: HandPose, uv, f_px: float, size_px: int | None = None) -> np.ndarray:
    """World direction of a wrist pixel's ray, scaled so its component along R2 is 1 (spec §4.5 point formula)."""
    a, b = centred(uv, size_px) / float(f_px)
    return pose.R[:, 2] + a * pose.R[:, 1] + b * pose.R[:, 0]


def project_direction(R, direction, f_px: float, size_px: int | None = None) -> np.ndarray | None:
    """Upright pixel of a world direction seen with orientation R (None when it points behind the camera)."""
    d = np.asarray(direction, dtype=float)
    z = float(d @ R[:, 2])
    if not z > 1e-9:
        return None
    return uncentred((float(f_px) * float(d @ R[:, 1]) / z, float(f_px) * float(d @ R[:, 0]) / z), size_px)


def derotate_pixel(uv, R_from, R_to, f_px: float, size_px: int | None = None) -> np.ndarray | None:
    """The pixel of the same viewing ray if the camera had orientation ``R_to`` (same centre): removes the
    rotation-induced image motion, which is depth-free; the error from an f misestimate scales with the rotation."""
    a, b = centred(uv, size_px) / float(f_px)
    R_from = np.asarray(R_from, dtype=float)
    return project_direction(R_to, R_from[:, 2] + a * R_from[:, 1] + b * R_from[:, 0], f_px, size_px)


def image_direction_world(pose: HandPose, axis_uv) -> np.ndarray:
    """World direction (unit, 3-D) of an upright image direction (du right, dv down) on a plane facing the camera:
    du * R1 - dv * R0; callers project it onto the horizontal for a top-down hand."""
    du, dv = (float(v) for v in axis_uv)
    return _unit(du * pose.R[:, 1] - dv * pose.R[:, 0])


def bandpass_grey(gray) -> np.ndarray:
    """The probe tracker's band-passed grey (decision §2.3 G-2): FT_BANDPASS_MID_GREY + FT_BANDPASS_GAIN x
    (G(sigma_fine) - G(sigma_coarse)) with FT_BANDPASS_SIGMAS, float32 (an RGB input is converted to luma first).
    The difference of Gaussians removes the low-frequency shading that moves with the view (a dark glossy part's
    sheen) and keeps the texture at the template scale; templates and the D15 test use the filtered patch."""
    image = _gray32(gray)
    fine, coarse = FT_BANDPASS_SIGMAS
    return (FT_BANDPASS_MID_GREY + FT_BANDPASS_GAIN * (cv2.GaussianBlur(image, (0, 0), fine)
                                                       - cv2.GaussianBlur(image, (0, 0), coarse))).astype(np.float32)


def dense_lattice(uv, *, exclude_mask=None, size_px: int | None = None) -> list[tuple[str, np.ndarray]]:
    """The probe tracker's dense seeds around a pointed pixel (decision §2.3 G-1): a lattice every FT_DENSE_STEP_PX
    within FT_DENSE_WINDOW_PX (a disk), keys "dense:<k>" (the kernel's DENSE_FEATURE_KEY_PREFIX, k counting the kept
    points), skipping points whose 11 px template would leave the image and pixels inside ``exclude_mask`` (the
    fingertip band). The points are placed exactly (no texture relocation)."""
    size = WRIST_IMAGE_PX if size_px is None else int(size_px)
    excluded = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
    r, window, step = FT_PROBE_RADIUS_PX, FT_DENSE_WINDOW_PX, FT_DENSE_STEP_PX
    u0, v0 = float(uv[0]), float(uv[1])
    out = []
    for dv in range(-window, window + 1, step):
        for du in range(-window, window + 1, step):
            if du * du + dv * dv > window * window:
                continue
            u, v = u0 + du, v0 + dv
            if not (r <= u <= size - 1 - r and r <= v <= size - 1 - r):
                continue
            if excluded is not None and excluded[int(round(v)), int(round(u))]:
                continue
            out.append((f"{DENSE_FEATURE_KEY_PREFIX}{len(out)}", np.array([u, v])))
    return out


def best_textured_patch(gray, uv, *, search_px: int = RELIEF_SEARCH_PX, radius_px: int = FT_RADIUS_PX,
                        min_std: float = D15_MIN_STD_GREY) -> tuple[np.ndarray | None, float]:
    """The integer patch centre within +-search_px of uv with the highest patch std (spec §4.5 relief probes; v1
    Align._textured_default idea), or (None, best std) when no patch passes ``min_std``."""
    image = _gray32(gray)
    h, w = image.shape
    r = int(radius_px)
    u0, v0 = int(round(float(uv[0]))), int(round(float(uv[1])))
    best, best_std = None, -1.0
    for v in range(max(r, v0 - search_px), min(h - 1 - r, v0 + search_px) + 1):
        for u in range(max(r, u0 - search_px), min(w - 1 - r, u0 + search_px) + 1):
            std = float(image[v - r:v + r + 1, u - r:u + r + 1].std())
            if std > best_std:
                best, best_std = np.array([float(u), float(v)]), std
    if best is None or best_std < min_std:
        return None, max(best_std, 0.0)
    return best, best_std


# ================================================================================================ FeatureTracker
def subpixel_peak(score: np.ndarray, row: int, col: int) -> np.ndarray:
    """v1 NCCTrack._subpixel restated: 2-D quadratic fit on the 3x3 peak neighbourhood; 1-D parabolas at window
    borders; offsets clipped to +-0.5 px. Returns (du, dv)."""
    rows, cols = score.shape
    if 0 < row < rows - 1 and 0 < col < cols - 1:
        c = SUBPIXEL_PINV @ np.asarray(score[row - 1:row + 2, col - 1:col + 2], dtype=float).ravel()
        hessian = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
        if hessian[0, 0] < -1e-9 and np.linalg.det(hessian) > 1e-12:
            offset = np.linalg.solve(hessian, -c[1:3])
            if np.all(np.abs(offset) <= 1.0):
                return np.clip(offset, -0.5, 0.5)
    sub = np.zeros(2)
    if 0 < col < cols - 1:
        sub[0] = _parabola(score[row, col - 1], score[row, col], score[row, col + 1])
    if 0 < row < rows - 1:
        sub[1] = _parabola(score[row - 1, col], score[row, col], score[row + 1, col])
    return sub


def _parabola(minus, peak, plus) -> float:
    minus, peak, plus = float(minus), float(peak), float(plus)
    curvature = minus - 2 * peak + plus
    return float(np.clip(0.5 * (minus - plus) / curvature, -0.5, 0.5)) if curvature < -1e-9 else 0.0


def _peak_neighbourhood(score: np.ndarray, row: int, col: int) -> tuple[np.ndarray, bool, bool]:
    """(3x3 neighbourhood of a score peak, NaN outside the window; column interior; row interior)."""
    rows, cols = score.shape
    out = np.full((3, 3), np.nan)
    r0, r1, c0, c1 = max(row - 1, 0), min(row + 2, rows), max(col - 1, 0), min(col + 2, cols)
    out[r0 - row + 1:r1 - row + 1, c0 - col + 1:c1 - col + 1] = score[r0:r1, c0:c1]
    return out, bool(0 < col < cols - 1), bool(0 < row < rows - 1)


def subpixel_peaks(neighbourhoods, col_interior, row_interior) -> np.ndarray:
    """``subpixel_peak`` vectorised over many peaks (decision §2.3 G-3, same math): neighbourhoods (n, 3, 3) of score
    values around each peak (``_peak_neighbourhood``), and whether each peak's column / row is inside its window.
    Interior peaks get the 2-D quadratic fit when its Hessian is negative definite and the offset within +-1 px
    (clipped to +-0.5); the others the 1-D parabolas on the interior axes. Returns (n, 2) offsets (du, dv)."""
    n3 = np.asarray(neighbourhoods, dtype=float).reshape(-1, 3, 3)
    count = n3.shape[0]
    out = np.zeros((count, 2))
    if count == 0:
        return out
    col_in = np.asarray(col_interior, dtype=bool).reshape(count)
    row_in = np.asarray(row_interior, dtype=bool).reshape(count)
    done = np.zeros(count, dtype=bool)
    both = np.flatnonzero(col_in & row_in)
    if both.size:
        c = n3[both].reshape(-1, 9) @ SUBPIXEL_PINV.T
        h00, h01, h11 = 2.0 * c[:, 3], c[:, 4], 2.0 * c[:, 5]
        det = h00 * h11 - h01 * h01
        usable = (h00 < -1e-9) & (det > 1e-12)
        safe = np.where(usable, det, 1.0)
        du = (-h11 * c[:, 1] + h01 * c[:, 2]) / safe
        dv = (h01 * c[:, 1] - h00 * c[:, 2]) / safe
        accept = usable & (np.abs(du) <= 1.0) & (np.abs(dv) <= 1.0)
        out[both[accept]] = np.clip(np.c_[du[accept], dv[accept]], -0.5, 0.5)
        done[both[accept]] = True

    def parabola(minus, peak, plus):
        curvature = minus - 2 * peak + plus
        ok = curvature < -1e-9
        return np.where(ok, np.clip(0.5 * (minus - plus) / np.where(ok, curvature, -1.0), -0.5, 0.5), 0.0)
    rest = ~done
    cols = np.flatnonzero(rest & col_in)
    if cols.size:
        out[cols, 0] = parabola(n3[cols, 1, 0], n3[cols, 1, 1], n3[cols, 1, 2])
    rows_ = np.flatnonzero(rest & row_in)
    if rows_.size:
        out[rows_, 1] = parabola(n3[rows_, 0, 1], n3[rows_, 1, 1], n3[rows_, 2, 1])
    return out


def _match_window(gray: np.ndarray, template: np.ndarray, centre: np.ndarray, search_px: int):
    """The search window of ``_match_peak``: (lo_u, lo_v, hi_u, hi_v, r, crop view) of a +-search_px search around
    round(centre) clipped so the template stays inside the image, or None when nothing is left."""
    h, w = gray.shape
    r = (template.shape[0] - 1) // 2
    x, y = int(round(float(centre[0]))), int(round(float(centre[1])))
    lo0, lo1 = max(r, x - search_px), max(r, y - search_px)
    hi0, hi1 = min(w - 1 - r, x + search_px), min(h - 1 - r, y + search_px)
    if hi0 < lo0 or hi1 < lo1:
        return None
    return lo0, lo1, hi0, hi1, r, gray[lo1 - r:hi1 + r + 1, lo0 - r:hi0 + r + 1]


def _match_scores(jobs) -> list:
    """TM_CCOEFF_NORMED score maps of [(crop, template)] (one thread's share of a frame's matches)."""
    return [cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED) for crop, template in jobs]


_MATCH_POOL: dict = {}


def _match_pool():
    """The process's FT_MATCH_WORKERS-thread pool for the per-feature matchTemplate calls (created on first use; a
    forked child, whose copy has no threads, makes its own)."""
    pid = os.getpid()
    pool = _MATCH_POOL.get(pid)
    if pool is None:
        _MATCH_POOL.clear()
        pool = _MATCH_POOL[pid] = ThreadPoolExecutor(max_workers=FT_MATCH_WORKERS, thread_name_prefix="v2_match")
    return pool


def match_scores(jobs) -> list:
    """``_match_scores`` of a whole frame's [(crop, template)], in order (decision §2.3 G-3, exact). With more jobs than
    FT_MATCH_WORKERS the calling thread runs the first FT_MATCH_MAIN_SHARE of them itself while the rest is split into
    FT_MATCH_WORKERS contiguous shares on ``_match_pool`` (cv2 releases the GIL inside matchTemplate). Every score map
    is the same call on the same arrays, so the result is identical to the serial loop. The calling thread's own share
    keeps a pool that idled between frames (the simulator step) off the critical path: on recorded task-2 hovers at
    468 features with 30 / 80 ms of idle before each frame, 4 workers alone took 48 / 48 ms per frame (serial 43 ms),
    8 workers with the 0.3 own share 25 / 28 ms."""
    jobs = list(jobs)
    count = len(jobs)
    workers = int(FT_MATCH_WORKERS)
    if workers <= 1 or count <= workers:
        return _match_scores(jobs)
    own = int(round(FT_MATCH_MAIN_SHARE * count))
    rest = jobs[own:]
    shares = [rest[k * len(rest) // workers:(k + 1) * len(rest) // workers] for k in range(workers)]
    pool = _match_pool()
    futures = [pool.submit(_match_scores, share) for share in shares if share]
    out = _match_scores(jobs[:own])
    for future in futures:
        out.extend(future.result())
    return out


def _window_peak(gray: np.ndarray, window, score: np.ndarray):
    """The peak of one window's score map: (window corner + peak (col, row) as uv | None, peak, peak neighbourhood |
    None, column interior, row interior); a peak on a window edge that is not the image edge is a miss.
    ``nan_to_num`` runs only on a non-finite score (G-3, exact)."""
    h, w = gray.shape
    lo0, lo1, hi0, hi1, r, _ = window
    if not np.isfinite(score).all():
        score = np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0)
    rows, cols = score.shape
    row, col = divmod(int(np.argmax(score)), cols)
    peak = float(min(max(float(score[row, col]), -1.0), 1.0))
    on_edge = ((col == 0 and lo0 > r) or (col == cols - 1 and hi0 < w - 1 - r)
               or (row == 0 and lo1 > r) or (row == rows - 1 and hi1 < h - 1 - r))
    if on_edge:
        return None, peak, None, False, False
    neighbourhood, col_in, row_in = _peak_neighbourhood(score, row, col)
    return np.array([lo0 + col, lo1 + row], dtype=float), peak, neighbourhood, col_in, row_in


def _match_peak(gray: np.ndarray, template: np.ndarray, centre: np.ndarray, search_px: int):
    """The integer part of ``_match``: (window corner + peak (col, row) as uv | None, peak, peak neighbourhood | None,
    column interior, row interior) of one template (``_match_window`` then ``_window_peak``)."""
    window = _match_window(gray, template, centre, search_px)
    if window is None:
        return None, -1.0, None, False, False
    return _window_peak(gray, window, cv2.matchTemplate(window[5], template, cv2.TM_CCOEFF_NORMED))


def _match(gray: np.ndarray, template: np.ndarray, centre: np.ndarray, search_px: int):
    """(template-centre uv | None, peak) of a TM_CCOEFF_NORMED search +-search_px around round(centre); a peak on
    the window edge that is not the image edge is a miss (v1 NCCTrack._search rule)."""
    corner, peak, neighbourhood, col_in, row_in = _match_peak(gray, template, centre, search_px)
    if corner is None:
        return None, peak
    return corner + subpixel_peaks(neighbourhood[None], [col_in], [row_in])[0], peak


class FeatureTracker:
    """§4.5 item 2, §3.8: many template tracks on one camera. Each template is the (2r+1)^2 patch centred on the
    nearest pixel to the requested point (moved inward at the image border), the fractional / border offset kept so
    ``uv`` always reports the requested point (v1 NCCTrack semantics); templates are fixed. A match below
    ``min_peak`` is a miss; ``V1_LOST_FRAMES`` consecutive misses lose the feature."""

    def __init__(self, gray0: np.ndarray, uvs, *, radius_px: int = FT_RADIUS_PX, search_px: int = FT_SEARCH_PX,
                 min_std: float = D15_MIN_STD_GREY, min_peak: float = V1_MIN_PEAK, keys=None):
        gray = _gray32(gray0)
        self.radius = int(radius_px)
        self.search = int(search_px)
        self.min_std = float(min_std)
        self.min_peak = float(min_peak)
        self.features: dict[int, dict] = {}
        self.rejected: list[dict] = []
        self.frames = 0
        keys = list(keys) if keys is not None else [None] * len(list(uvs))
        for index, uv in enumerate(uvs):
            self.add(gray, uv, key=keys[index] if index < len(keys) else None)

    def add(self, gray, uv, *, key=None) -> int | None:
        """Add one feature; returns its fid, or None (recorded in ``rejected``) when the patch fails D15 or lies
        outside the image."""
        image = _gray32(gray)
        h, w = image.shape
        r = self.radius
        uv = np.asarray(uv, dtype=float)
        if not (0.0 <= uv[0] <= w - 1 and 0.0 <= uv[1] <= h - 1) or w < 2 * r + 1 or h < 2 * r + 1:
            self.rejected.append({"key": key, "uv": jsonable(uv), "reason": "out_of_view"})
            return None
        centre = np.clip(np.rint(uv), [r, r], [w - 1 - r, h - 1 - r]).astype(int)
        patch = image[centre[1] - r:centre[1] + r + 1, centre[0] - r:centre[0] + r + 1]
        std = float(patch.std())
        if std < self.min_std:
            self.rejected.append({"key": key, "uv": jsonable(uv), "reason": "textureless", "std": std})
            return None
        fid = len(self.features) + len(self.rejected)
        while fid in self.features:
            fid += 1
        source_r = int(math.ceil(r / FT_MIN_SCALE)) + 2
        source = np.full((2 * source_r + 1, 2 * source_r + 1), np.nan, dtype=np.float32)
        x, y = int(centre[0]), int(centre[1])
        y0, y1, x0, x1 = max(y - source_r, 0), min(y + source_r + 1, h), max(x - source_r, 0), min(x + source_r + 1, w)
        source[y0 - y + source_r:y1 - y + source_r, x0 - x + source_r:x1 - x + source_r] = image[y0:y1, x0:x1]
        self.features[fid] = {"key": key, "uv0": uv.copy(), "uv": uv.copy(), "centre": centre.astype(float),
                              "offset": uv - centre, "template": np.ascontiguousarray(patch, dtype=np.float32),
                              "source": source, "source_xy": uv - (centre - source_r), "std": std, "peak": None,
                              "misses": 0, "lost": False, "history": [(self.frames, uv.copy())],
                              # scaled_template's two warp inputs, computed once (G-3 exact: the same arrays)
                              "source_filled": np.nan_to_num(source, nan=0.0),
                              "source_valid": np.isfinite(source).astype(np.float32)}
        return fid

    def scaled_template(self, fid, scale: float) -> np.ndarray | None:
        """The feature's template resampled at ``scale`` (new / capture image scale) and centred EXACTLY on the
        feature point (sub-pixel), from its stored source patch; None when the source does not cover it."""
        feature = self.features[fid]
        r = self.radius
        s = float(scale)
        if not s >= FT_MIN_SCALE:
            return None
        sx, sy = feature["source_xy"]
        matrix = np.array([[1.0 / s, 0.0, sx - r / s], [0.0, 1.0 / s, sy - r / s]])
        size = 2 * r + 1
        flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        source = feature["source"]
        filled = feature.get("source_filled")
        filled = np.nan_to_num(source, nan=0.0) if filled is None else filled
        valid_in = feature.get("source_valid")
        valid_in = np.isfinite(source).astype(np.float32) if valid_in is None else valid_in
        warped = cv2.warpAffine(filled, matrix, (size, size), flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        valid = cv2.warpAffine(valid_in, matrix, (size, size), flags=flags,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        if float(valid.min()) < 1.0 - 1e-6:
            return None
        return np.ascontiguousarray(warped, dtype=np.float32)

    @property
    def active(self) -> list[int]:
        return [fid for fid, feature in self.features.items() if not feature["lost"]]

    def _shift_for(self, fid, order, predicted_shift_px):
        if predicted_shift_px is None:
            return np.zeros(2)
        if isinstance(predicted_shift_px, Mapping):
            value = predicted_shift_px.get(fid)
            return np.zeros(2) if value is None else np.asarray(value, dtype=float)
        shift = np.asarray(predicted_shift_px, dtype=float)
        if shift.shape == (2,):
            return shift
        return shift[order.index(fid)]

    def track(self, gray: np.ndarray, predicted_shift_px: np.ndarray | dict | None = None,
              predicted_scale: Mapping | None = None, search_px_by_fid: Mapping | None = None) -> dict[int, tuple]:
        """Match every active feature in ``gray`` around its last position plus the predicted shift (a global (2,)
        shift, an (n, 2) array in ``active`` order, or {fid: (2,)}); returns fid -> (uv | None, peak).
        ``predicted_scale`` {fid: s} (image scale now / at capture, from a predicted depth change along the optical
        axis): the template is resampled at s and centred exactly on the feature point, so looming does not bias
        the match toward the patch's contrast centroid. ``search_px_by_fid`` {fid: px} overrides the search half
        width per feature (G-3 adaptive search; missing fids use ``search``). The frame's template matches run in
        three exact steps: every feature's search window, all score maps (``match_scores``: FT_MATCH_WORKERS threads),
        every peak; the sub-pixel peaks of all matches are fitted in one vectorised step (``subpixel_peaks``)."""
        image = _gray32(gray)
        self.frames += 1
        order = self.active
        out, hits, jobs = {}, [], []
        for fid in order:
            feature = self.features[fid]
            shift = self._shift_for(fid, order, predicted_shift_px)
            search = self.search if search_px_by_fid is None else int(search_px_by_fid.get(fid, self.search))
            scale = None if predicted_scale is None else predicted_scale.get(fid)
            template = None if scale is None else self.scaled_template(fid, scale)
            if template is not None:
                jobs.append((fid, template, _match_window(image, template, feature["uv"] + shift, search), None))
            else:
                jobs.append((fid, feature["template"], _match_window(image, feature["template"],
                                                                     feature["centre"] + shift, search),
                             feature["offset"]))
        scores = iter(match_scores([(window[5], template) for _, template, window, _ in jobs if window is not None]))
        for fid, _, window, offset in jobs:
            feature = self.features[fid]
            if window is None:
                corner, peak, neighbourhood, col_in, row_in = None, -1.0, None, False, False
            else:
                corner, peak, neighbourhood, col_in, row_in = _window_peak(image, window, next(scores))
            feature["peak"] = peak
            if corner is None or peak < self.min_peak:
                feature["misses"] += 1
                if feature["misses"] >= V1_LOST_FRAMES:
                    feature["lost"] = True
                out[fid] = (None, peak)
                continue
            hits.append((fid, corner, offset, peak, neighbourhood, col_in, row_in))
        if hits:
            subs = subpixel_peaks([h[4] for h in hits], [h[5] for h in hits], [h[6] for h in hits])
            for (fid, corner, offset, peak, _, _, _), sub in zip(hits, subs):
                feature = self.features[fid]
                uv = corner + sub if offset is None else corner + sub + offset
                feature["misses"] = 0
                feature["uv"] = uv
                feature["centre"] = uv - feature["offset"]
                feature["history"].append((self.frames, uv.copy()))
                out[fid] = (uv.copy(), peak)
        return {fid: out[fid] for fid in order}

    @staticmethod
    def auto_grid(gray, *, spacing_px: int = FT_GRID_SPACING_PX, max_features: int = FT_MAX_FEATURES,
                  mask=None, radius_px: int = FT_RADIUS_PX, min_std: float = D15_MIN_STD_GREY) -> list[tuple]:
        """Centres of a centred grid of (2r+1)^2 patches passing D15 (std >= min_std), highest std first, at most
        ``max_features``, skipping centres inside ``mask`` (bool, True = excluded)."""
        image = _gray32(gray)
        h, w = image.shape
        rows = grid_positions(h, spacing_px, radius_px)
        cols = grid_positions(w, spacing_px, radius_px)
        excluded = None if mask is None else np.asarray(mask, dtype=bool)
        r = int(radius_px)
        found = []
        for v in rows:
            for u in cols:
                if excluded is not None and excluded[v, u]:
                    continue
                std = float(image[v - r:v + r + 1, u - r:u + r + 1].std())
                if std >= min_std:
                    found.append((std, float(u), float(v)))
        found.sort(key=lambda item: -item[0])
        return [(u, v) for _, u, v in found[:int(max_features)]]

    def to_dict(self) -> dict:
        lost = sum(bool(f["lost"]) for f in self.features.values())
        return {"features": len(self.features), "lost": lost, "rejected": len(self.rejected), "frames": self.frames,
                "radius_px": self.radius, "search_px": self.search, "min_std": self.min_std,
                "min_peak": self.min_peak}


def grid_positions(size: int, spacing_px: int, radius_px: int) -> list[int]:
    """Integer centres of a grid with ``spacing_px`` spacing, centred in [r, size - 1 - r]."""
    r = int(radius_px)
    span = size - 1 - 2 * r
    if span < 0:
        return []
    count = span // int(spacing_px) + 1
    start = r + (span - (count - 1) * int(spacing_px)) // 2
    return [start + k * int(spacing_px) for k in range(count)]


# ================================================================================================ depth models
def _sample_parts(sample):
    ds = np.asarray(sample[0], dtype=float)
    dp = np.asarray(sample[1], dtype=float)
    dz = float(sample[2])
    uv_end = None if len(sample) < 4 or sample[3] is None else np.asarray(sample[3], dtype=float)
    return ds, dp, dz, uv_end


def fit_feature_kappa(samples: Sequence[tuple], *, f_px: float | None = None, intercept: bool = True,
                      size_px: int | None = None, min_samples: int | None = None,
                      aperture: np.ndarray | None = None) -> tuple[float | None, float | None, float, str | None]:
    """(kappa, sigma_kappa, residual_px, reject) of one feature from (ds px, dp_inplane (d_a, d_b) m along (R1, R0),
    dp_axial d_z m along R2[, uv_end upright px]) samples, all relative to one reference frame (rotation removed).

    Pinhole model, exact for a translating camera: u1' - u0' = u1' d_z / z0 - kappa d_a and v1' - v0' =
    v1' d_z / z0 - kappa d_b with z0 = f / kappa, i.e. ds = kappa * (-d_a + u1' d_z / f, d_b - v1' d_z / f) (+ an
    intercept b that absorbs the reference frame's own tracking error). The axial term is the looming of the
    measured axial camera motion (about 2.2 mm per 4 cm probe at the 3.14 deg start tilt); it is linear in kappa,
    so no iteration is needed. f enters only through that small term (f_px None = R3). The term needs the end pixel:
    a sample with axial motion (d_z != 0) and no uv_end raises ValueError (ignoring the axial term would bias kappa,
    contract §C5); a 3-tuple with d_z = 0 is exact. sigma_kappa is the least-squares standard error from the
    residual, inflated for the lag-1 autocorrelation of consecutive-frame residuals (D4); D1 rules: at least
    ``min_samples`` frames (None = D1_MIN_SAMPLES; fewer: reject "too_few_samples", the caller maps it to the kernel's
    FeatureRecord reasons), shift >= 6 px and residual <= max(0.6 px, 0.1 rms shift). ``aperture`` (a 2x2 matrix, the
    square root of the template's normalized structure tensor, ``template_aperture``) weights every image
    displacement by what the template can measure: along an edge or a ridge the match slides (the aperture
    problem), so that component counts little in the fit, in the shift test and in the residual."""
    f = robot.focal_length_px(WRIST_IMAGE_PX if size_px is None else size_px) if f_px is None else float(f_px)
    weight = np.eye(2) if aperture is None else np.asarray(aperture, dtype=float).reshape(2, 2)
    count = len(samples)
    ds_all, g_all = np.zeros((count, 2)), np.zeros((count, 2))
    c = principal_point(size_px)
    for k, sample in enumerate(samples):
        ds, dp, dz, uv_end = _sample_parts(sample)
        if uv_end is not None:
            u1, v1 = uv_end[0] - c[0], c[1] - uv_end[1]
        elif dz != 0.0:
            raise ValueError("a kappa sample with axial camera motion (dp_axial != 0) needs its end pixel uv_end: "
                             "the axial looming term is modelled, not ignored")
        else:
            u1 = v1 = 0.0
        ds_all[k] = ds
        g_all[k] = (-dp[0] + u1 * dz / f, dp[1] - v1 * dz / f)
    if count < (D1_MIN_SAMPLES if min_samples is None else int(min_samples)):
        return None, None, 0.0, TOO_FEW_SAMPLES
    # rows per sample: weight @ [[g0, 1, 0], [g1, 0, 1]] (vectorised, G-3)
    blocks = np.zeros((count, 2, 3))
    blocks[:, :, 0] = g_all
    blocks[:, 0, 1] = blocks[:, 1, 2] = 1.0
    A = np.einsum("ij,njk->nik", weight, blocks).reshape(-1, 3)
    weighted = ds_all @ weight.T
    b = weighted.reshape(-1)
    shifts = np.linalg.norm(weighted, axis=1).tolist()
    if not intercept:
        A = A[:, :1]
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    kappa = float(params[0])
    residuals = b - A @ params
    dof = max(len(b) - A.shape[1], 1)
    sigma_px = math.sqrt(float(residuals @ residuals) / dof)
    residual_px = math.sqrt(float(np.mean(residuals ** 2)))
    try:
        covariance = np.linalg.inv(A.T @ A)
        sigma_kappa = (sigma_px * math.sqrt(max(float(covariance[0, 0]), 0.0))
                       * math.sqrt(_correlation_inflation(residuals)))
    except np.linalg.LinAlgError:
        sigma_kappa = None
    if max(shifts) < D1_MIN_SHIFT_PX:
        return kappa, sigma_kappa, residual_px, "shift_too_small"
    rms_shift = math.sqrt(float(np.mean(np.square(shifts))))
    if not (math.isfinite(kappa) and kappa > 0.0) or sigma_kappa is None:
        return None, None, residual_px, "residual"
    if residual_px > max(D1_RESIDUAL_FLOOR_PX, D1_RESIDUAL_FRACTION * rms_shift):
        return kappa, sigma_kappa, residual_px, "residual"
    return kappa, sigma_kappa, residual_px, None


def _correlation_inflation(residuals) -> float:
    """Variance inflation (1 + rho) / (1 - rho) for the lag-1 autocorrelation rho >= 0 of consecutive-frame
    residuals (per image axis; rows alternate u, v): template tracking errors of neighbouring frames are correlated
    (subpixel bias varies smoothly with the fractional shift), so n frames carry fewer than n independent errors."""
    r = np.asarray(residuals, dtype=float)
    rhos = []
    for axis in (r[0::2], r[1::2]):
        if axis.size < D1_MIN_SAMPLES:
            continue
        energy = float(axis @ axis)
        if energy > 1e-12:
            rhos.append(float(axis[:-1] @ axis[1:]) / energy)
    if not rhos:
        return 1.0
    rho = float(np.clip(np.mean(rhos), 0.0, D4_MAX_AUTOCORRELATION))
    return (1.0 + rho) / (1.0 - rho)


def template_aperture(patch) -> np.ndarray:
    """Square root of a template's structure tensor normalized to its largest eigenvalue (eigenvalue ratios floored at
    D1_MIN_APERTURE): the per-direction weight of a template match's displacement (1 across an edge, sqrt of the
    eigenvalue ratio along it); the identity for 2-D textures (weak / strong >= D1_ANISOTROPY_RATIO)."""
    image = np.asarray(patch, dtype=float)
    gy, gx = np.gradient(image)
    tensor = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    values, vectors = np.linalg.eigh(tensor)
    top = float(values[-1])
    if not top > 1e-12:
        return np.eye(2)
    ratios = np.clip(values / top, D1_MIN_APERTURE, 1.0)
    if float(ratios[0]) >= D1_ANISOTROPY_RATIO:
        return np.eye(2)              # a 2-D texture measures both directions: unweighted (D4 sigma stays calibrated)
    return vectors @ np.diag(np.sqrt(ratios)) @ vectors.T


def depth_from_kappa(kappa, sigma_kappa, f_px, sigma_f) -> tuple[float, float]:
    """D4: Z = f / kappa along the optical axis; sigma_Z / Z = sigma_kappa / kappa (+) sigma_f / f."""
    kappa, f = float(kappa), float(f_px)
    if not (kappa > 0.0 and f > 0.0):
        raise ValueError("kappa and f must be > 0")
    z = f / kappa
    rel = math.hypot(float(sigma_kappa or 0.0) / kappa, float(sigma_f or 0.0) / f)
    return z, z * rel


def loom_depth(u0, u1, d_cam_m, kappa_px_per_m, *, sigma_px: float | None = None,
               sigma_kappa: float | None = None, size_px: int | None = None) -> tuple[float | None, float | None]:
    """D3, one feature, two frames, f-free [C-35]: depth z0 along R2 at the first frame from the image motion of a
    camera displaced by d_cam_m = (d_a, d_b, d_z) (measured, in the first frame's (R1, R0, R2) basis; rotation
    removed). Per image axis z0 = u1' d_z / (u1' - u0' + kappa d_a) (and v', d_b); the two axes are fused by inverse
    variance from the pixel noise (sigma_px None = D4_PIXEL_NOISE_PX) and sigma_kappa. (None, None) when neither
    axis carries a looming signal."""
    noise = D4_PIXEL_NOISE_PX if sigma_px is None else float(sigma_px)
    s_kappa = 0.0 if sigma_kappa is None else float(sigma_kappa)
    a0, b0 = centred(u0, size_px)
    a1, b1 = centred(u1, size_px)
    d_a, d_b, d_z = (float(v) for v in d_cam_m)
    estimates = []
    for c0, c1, d_in in ((a0, a1, d_a), (b0, b1, d_b)):
        numerator = c1 * d_z
        denominator = c1 - c0 + float(kappa_px_per_m) * d_in
        if abs(denominator) < 1e-9:
            continue
        z = numerator / denominator
        dz_du1 = (d_z - z) / denominator
        dz_du0 = z / denominator
        dz_dk = -z * d_in / denominator
        variance = (dz_du1 ** 2 + dz_du0 ** 2) * noise ** 2 + (dz_dk * s_kappa) ** 2
        if variance > 0.0 and z > 0.0:
            estimates.append((z, variance))
    if not estimates:
        return None, None
    weights = [1.0 / var for _, var in estimates]
    z = sum(w * value for w, (value, _) in zip(weights, estimates)) / sum(weights)
    return float(z), float(math.sqrt(1.0 / sum(weights)))


def loom_fit(frames, kappa, *, sigma_kappa: float | None = None, size_px: int | None = None) -> tuple:
    """Multi-frame f-free looming depth of one feature: frames = [(uv upright px (rotation removed), d_cam
    (d_a, d_b, d_z) m relative to the reference frame)], the reference frame included with d = 0. Unknowns
    (z0, p_u = z0 u0', p_v = z0 v0'), per frame and axis: z0 (u_k' + kappa d_a) - p_u = u_k' d_z (the reference
    pixel is estimated, not trusted). Returns (z0 | None, sigma_z0 | None, residual_px)."""
    rows, targets = [], []
    for uv, d in frames:
        a, b = centred(uv, size_px)
        d_a, d_b, d_z = (float(v) for v in d)
        rows.append([a + float(kappa) * d_a, -1.0, 0.0])
        targets.append(a * d_z)
        rows.append([b + float(kappa) * d_b, 0.0, -1.0])
        targets.append(b * d_z)
    if len(rows) < 2 * D1_MIN_SAMPLES:
        return None, None, 0.0
    A, y = np.array(rows), np.array(targets)
    try:
        params, *_ = np.linalg.lstsq(A, y, rcond=None)
        covariance = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return None, None, 0.0
    z0 = float(params[0])
    if not (math.isfinite(z0) and z0 > 0.0):
        return None, None, 0.0
    residuals = y - A @ params
    dof = max(len(y) - A.shape[1], 1)
    sigma = math.sqrt(float(residuals @ residuals) / dof)
    sigma_z = sigma * math.sqrt(max(float(covariance[0, 0]), 0.0))
    if sigma_kappa:
        # kappa enters through the in-plane terms kappa d_a, kappa d_b: refit with kappa + sigma_kappa
        shifted = A.copy()
        shifted[:, 0] += float(sigma_kappa) * np.array([d[k % 2] for _, d in frames for k in range(2)])
        moved, *_ = np.linalg.lstsq(shifted, y, rcond=None)
        sigma_z = math.hypot(sigma_z, float(moved[0]) - z0)
    return z0, float(sigma_z), float(sigma / z0)


# ================================================================================================ CameraModel
def r3_focal_px(size_px: int | None = None) -> float:
    """R3 under R-2f: f = 333.6 px at 512 px."""
    return robot.focal_length_px(WRIST_IMAGE_PX if size_px is None else size_px)


def r3_sigma_px(size_px: int | None = None) -> float:
    """The R3 value's own rounding against the robot asset's camera (fovy 75 deg -> 333.626 px at 512)."""
    asset_f = float(robot.ROBOT_FACTS["asset"]["wrist_camera"]["f_px_at_512"])
    size = WRIST_IMAGE_PX if size_px is None else size_px
    return abs(asset_f * size / robot.FOCAL_REFERENCE_IMAGE_PX - r3_focal_px(size))


@dataclass
class CameraModel:
    """§4.5 CameraModel rule (R3, D3): f_used = R3 when R-2f is granted and |f_hat / R3 - 1| <= max(D3_AGREE_FLOOR,
    3 sigma_f_hat / R3); otherwise f_hat with the larger sigma (disagreement logged)."""
    f_used: float
    sigma_f: float
    source: str                          # "R3" | "f_hat"
    f_hat: Meas | None
    disagreement: bool
    thresholds: list[Threshold] = field(default_factory=list)

    @classmethod
    def decide(cls, f_hat: Meas | None, *, r2f_granted: bool) -> "CameraModel":
        r3, r3_sigma = r3_focal_px(), r3_sigma_px()
        if f_hat is None:
            if not r2f_granted:
                raise ValueError("no focal length: R-2f not granted and no looming f-hat measured")
            return cls(r3, r3_sigma, R3_SOURCE, None, False, [th_value(
                "D3_f_agreement", D3_AGREE_FLOOR, "DD", "no f-hat measured: R3 used unchecked (R-2f)",
                r3_px=r3)])
        rel = abs(f_hat.value / r3 - 1.0)
        threshold = th_max("D3_f_agreement", Term("DD", D3_AGREE_FLOOR, "D3_AGREE_FLOOR"),
                           Term("ONLINE", D3_SIGMA_K * f_hat.sigma / r3, "3*sigma_f_hat/R3"), unit="(relative)")
        agree = rel <= threshold.value
        if r2f_granted and agree:
            return cls(r3, r3_sigma, R3_SOURCE, f_hat, False, [threshold])
        return cls(float(f_hat.value), max(float(f_hat.sigma), r3_sigma), F_HAT_SOURCE, f_hat, not agree,
                   [threshold])

    def wrist(self) -> WristCamera:
        return WristCamera(self.f_used)

    def to_dict(self) -> dict:
        return jsonable({"f_used": self.f_used, "sigma_f": self.sigma_f, "source": self.source,
                         "f_hat": None if self.f_hat is None else self.f_hat.to_dict(),
                         "disagreement": self.disagreement, "thresholds": [t.to_dict() for t in self.thresholds]})


# ================================================================================================ image helpers
def _hand_flow(before_rgb, after_rgb):
    """v1 hand_image_motion (external_axes.py:42-62) restated: (pick mask, median flow, pixels, changed) or None."""
    before, after = _luma_u8(before_rgb), _luma_u8(after_rgb)
    if before.shape != after.shape:
        raise ValueError("frames must have the same size")
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = HM_FARNEBACK
    flow = cv2.calcOpticalFlowFarneback(before, after, None, pyr_scale, int(levels), int(winsize), int(iterations),
                                        int(poly_n), poly_sigma, 0)
    changed = np.abs(before.astype(int) - after.astype(int)) > HM_DIFF_GREY
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((HM_DILATE_PX, HM_DILATE_PX), np.uint8)) > 0
    magnitude = np.linalg.norm(flow, axis=2)
    selected = changed & (magnitude > HM_MIN_FLOW_PX)
    if int(selected.sum()) < HM_MIN_PIXELS:
        return None
    threshold = float(np.quantile(magnitude[selected], 1.0 - HM_TOP_FRACTION))
    pick = selected & (magnitude >= threshold)
    vector = np.median(flow[pick], axis=0)
    return pick, vector, int(pick.sum()), int(selected.sum())


def hand_motion_mask(before_rgb, after_rgb) -> np.ndarray | None:
    """The v1 hand-motion mask (DIFF 12, dilate 9, flow > 1 px, fastest decile): the moving hand's pixels in the
    BEFORE frame (Farneback flow is defined on the first frame), or None (fewer than 50 pixels)."""
    result = _hand_flow(before_rgb, after_rgb)
    return None if result is None else result[0]


def hand_motion(before_rgb, after_rgb) -> dict | None:
    """hand_motion_mask plus v1 hand_image_motion's fields: {"mask", "image_shift_px", "pixels", "changed_pixels"}."""
    result = _hand_flow(before_rgb, after_rgb)
    if result is None:
        return None
    pick, vector, pixels, changed = result
    return {"mask": pick, "image_shift_px": [float(vector[0]), float(vector[1])], "pixels": pixels,
            "changed_pixels": changed}


def mask_extreme(mask, v_hat, *, band_px: float = L1_TIP_BAND_PX) -> np.ndarray | None:
    """The extreme point of a mask along +v_hat: the coordinate along v_hat of the farthest pixel, and the mean
    across-coordinate of the pixels within ``band_px`` of it (both fingertips of a top-down hand, not one)."""
    if mask is None:
        return None
    vs, us = np.nonzero(np.asarray(mask, dtype=bool))
    if us.size == 0:
        return None
    v_hat = _unit(np.asarray(v_hat, dtype=float)[:2])
    n_hat = np.array([v_hat[1], -v_hat[0]])
    pts = np.c_[us.astype(float), vs.astype(float)]
    along = pts @ v_hat
    top = float(along.max())
    band = along >= top - float(band_px)
    across = float(np.mean(pts[band] @ n_hat))
    return top * v_hat + across * n_hat


def to_lab255(rgb) -> np.ndarray:
    """[C-20] CIE Lab in OpenCV 8-bit scaling (L * 255 / 100, a + 128, b + 128) as float32, from float RGB (no
    8-bit quantization), so |dLab| is comparable to grey levels."""
    image = np.asarray(rgb, dtype=np.float32)[..., :3] / np.float32(RGB_MAX)
    lab = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2LAB)
    out = np.empty_like(lab)
    out[..., 0] = lab[..., 0] * np.float32(LAB_L_SCALE)
    out[..., 1:] = lab[..., 1:] + np.float32(LAB_AB_OFFSET)
    return out


def structure_axis(gray, uv, half_px: int = V5_TENSOR_HALF_PX) -> tuple[np.ndarray, float, float]:
    """The v1 bar_point_check structure tensor (wrist_servo.py:83-114) re-implemented to return the orientation:
    (unit image axis (du, dv) ALONG the ridge = across the dominant gradient, coherence, gradient rms) of the
    (2 half + 1)^2 patch at uv (clipped at the image border). The axis is a line direction: du >= 0."""
    image = np.asarray(_luma(gray), dtype=float)
    h, w = image.shape
    r = int(half_px)
    u0 = int(np.clip(round(float(uv[0])), r, w - 1 - r))
    v0 = int(np.clip(round(float(uv[1])), r, h - 1 - r))
    patch = image[v0 - r:v0 + r + 1, u0 - r:u0 + r + 1]
    gy, gx = np.gradient(patch)
    tensor = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
    values, vectors = np.linalg.eigh(tensor)
    coherence = float((values[1] - values[0]) / max(values[1] + values[0], 1e-9))
    across = vectors[:, 1]
    along = np.array([-across[1], across[0]])
    if along[0] < 0.0 or (abs(along[0]) < 1e-12 and along[1] < 0.0):
        along = -along
    rms = math.sqrt(max(float(values[1]), 0.0) / patch.size)
    return along / max(float(np.linalg.norm(along)), 1e-12), coherence, rms


# ================================================================================ G-7 orientation kernels (D10-OE)
# Design review 2 decision §2.2 G-7: pure vision, moved from the review-2 axis lens (orientation_peak, mask_outline,
# edge_hist; ROI forms region and hull only, the scene builds them) and vectorised. The scene's S-11 decides with
# them (gates, ROIs, the exclusion and the agreement are the scene's); nothing here reads a truth.
def _canonical_xy(axis) -> np.ndarray:
    """A world-xy line direction as a unit vector with x > 0 (x = 0: y > 0)."""
    a = np.asarray(axis, dtype=float)[:2]
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    return -a if (a[0] < -1e-9 or (abs(float(a[0])) <= 1e-9 and a[1] < 0.0)) else a


def _orientation_window_deg() -> float:
    """The orientation peak window: half the yaw grid (decision §3: YAW_GRID_RAD / 2, no new number), in deg rounded
    to 1e-9 so the radian round trip (7.499999...) does not move a histogram bin exactly 2 windows away."""
    return round(math.degrees(YAW_GRID_RAD / 2.0), 9)


def _line_angle_deg(dirs_xy) -> np.ndarray:
    """Line orientation in [0, 180) deg of (N, 2) directions (NaN rows stay NaN)."""
    d = np.asarray(dirs_xy, dtype=float).reshape(-1, 2)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180


def _off_axis_deg(angles_deg, axis_deg) -> np.ndarray:
    """Signed line-orientation difference in [-90, 90) deg."""
    return (np.asarray(angles_deg, dtype=float) - float(axis_deg) + 90) % 180 - 90


def backproject_at_z(pose: HandPose, uv, z: float, f_px: float, size_px: int | None = None) -> np.ndarray:
    """Upright wrist pixels (N, 2) -> world points (N, 3) on the horizontal plane at height z (exact ray-plane
    intersection, ``pixel_direction`` vectorised); NaN rows for rays parallel to the plane."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    c = principal_point(size_px)
    a = (uv[:, 0] - c[0]) / float(f_px)
    b = (c[1] - uv[:, 1]) / float(f_px)
    R = np.asarray(pose.R, dtype=float)
    d = R[:, 2][None, :] + a[:, None] * R[:, 1][None, :] + b[:, None] * R[:, 0][None, :]
    out = np.full((len(uv), 3), np.nan)
    ok = np.abs(d[:, 2]) > 1e-9
    t = (float(z) - float(pose.camera[2])) / d[ok, 2]
    out[ok] = np.asarray(pose.camera, dtype=float)[None, :] + t[:, None] * d[ok]
    return out


def image_directions_at_z(pose: HandPose, uv, directions_uv, z: float, f_px: float,
                          size_px: int | None = None) -> np.ndarray:
    """World-xy directions (N, 2, not normalised) of upright-image directions (du, dv) at pixels uv, carried onto the
    plane at height z by the exact differential of ``backproject_at_z`` (the homography's local Jacobian):
    dX/du = (t / f)(R1 - d R1_z / d_z), dX/dv = -(t / f)(R0 - d R0_z / d_z) with d the pixel's ray and t its
    parameter; NaN rows where the ray is parallel to the plane."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    e = np.asarray(directions_uv, dtype=float).reshape(-1, 2)
    c = principal_point(size_px)
    f = float(f_px)
    a = (uv[:, 0] - c[0]) / f
    b = (c[1] - uv[:, 1]) / f
    R = np.asarray(pose.R, dtype=float)
    d = R[:, 2][None, :] + a[:, None] * R[:, 1][None, :] + b[:, None] * R[:, 0][None, :]
    out = np.full((len(uv), 2), np.nan)
    ok = np.abs(d[:, 2]) > 1e-9
    t = (float(z) - float(pose.camera[2])) / d[ok, 2]
    dk = d[ok]
    du = (t / f)[:, None] * (R[:, 1][None, :] - dk * (R[2, 1] / dk[:, 2])[:, None])
    dv = -(t / f)[:, None] * (R[:, 0][None, :] - dk * (R[2, 0] / dk[:, 2])[:, None])
    out[ok] = (e[ok, 0][:, None] * du + e[ok, 1][:, None] * dv)[:, :2]
    return out


def orientation_peak(dirs_xy, weights) -> tuple[np.ndarray | None, float, float | None]:
    """G-7: the dominant LINE orientation of weighted directions (decision §2.2; axis lens ``orientation_peak``): a
    doubled-angle histogram of 1 deg bins over [0, 180), circularly smoothed by a Gaussian of ORIENT_SMOOTH_DEG; the
    peak bin refined by the weighted mean of the directions within +-YAW_GRID_RAD / 2 of it. Returns (unit world-xy
    axis (x > 0), share = the weight within that window / all weight, second = the smoothed histogram's highest
    value at least 2 windows from the peak / its peak value); (None, 0.0, None) with fewer than 3 finite directions
    or no weight. Non-finite rows are dropped."""
    d = np.asarray(dirs_xy, dtype=float).reshape(-1, 2)
    w = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.isfinite(d).all(axis=1) & np.isfinite(w)
    d, w = d[ok], w[ok]
    if len(d) < 3 or not float(w.sum()) > 0.0:
        return None, 0.0, None
    window = _orientation_window_deg()
    bins = 180                                   # 1 deg bins over the line orientations [0, 180)
    angle = _line_angle_deg(d)
    hist, _ = np.histogram(angle, bins=bins, range=(0.0, float(bins)), weights=w)
    offset = np.arange(bins)
    distance = np.minimum(offset, bins - offset).astype(float)
    kernel = np.exp(-0.5 * (distance / ORIENT_SMOOTH_DEG) ** 2)
    smooth = np.real(np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel)))
    peak = int(np.argmax(smooth))
    off = _off_axis_deg(angle, peak)
    near = np.abs(off) <= window
    share = float(w[near].sum() / w.sum())
    mean = float(peak) + (float(np.average(off[near], weights=w[near])) if near.any() and w[near].sum() > 0 else 0.0)
    far = np.abs(_off_axis_deg(offset, peak)) > 2.0 * window
    second = float(smooth[far].max() / max(float(smooth[peak]), 1e-12)) if far.any() else 0.0
    rad = math.radians(mean)
    return _canonical_xy([math.cos(rad), math.sin(rad)]), share, second


def outline_orientation(mask, pose: HandPose, z: float, f_px: float) -> dict | None:
    """G-7 outline orientation of a region (decision §2.2; axis lens ``mask_outline``): the largest external outline
    of ``mask`` (upright wrist pixels, every outline pixel) back-projected at height z (``backproject_at_z``); the
    tangent at each outline point = the chord between the points OUTLINE_TANGENT_PTS before and after it, weighted by
    half its length (length-weighted); ``orientation_peak`` of the tangents. ``two_sided``: at least
    OUTLINE_SIDE_MIN_PTS (3) outline points whose tangent lies within the peak window on EACH side of the axis line
    through the outline's mean point (a straight edge on one side only, a one-sided silhouette, is not a strip).
    Returns {"axis" (unit world xy), "share", "second", "two_sided", "n" (outline points), "sides" (points per side),
    "width_m" (the median across distance of the two sides' points, a diagnosis)} or None (no outline, fewer than
    3 x OUTLINE_TANGENT_PTS outline points, or no peak)."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    k = int(OUTLINE_TANGENT_PTS)
    if len(outline) < 3 * k:
        return None
    world = backproject_at_z(pose, outline, z, f_px, size_px=m.shape[1])[:, :2]
    chords = np.roll(world, -k, axis=0) - np.roll(world, k, axis=0)
    length = np.linalg.norm(chords, axis=1)
    dirs = chords / np.maximum(length, 1e-12)[:, None]
    axis, share, second = orientation_peak(dirs, length / 2.0)
    if axis is None:
        return None
    normal = np.array([-axis[1], axis[0]])
    near = np.abs(_off_axis_deg(_line_angle_deg(dirs), math.degrees(math.atan2(axis[1], axis[0])) % 180)) \
        <= _orientation_window_deg()
    with np.errstate(invalid="ignore"):
        across = (world - np.nanmean(world, axis=0)) @ normal
        positive, negative = near & (across > 0.0), near & (across < 0.0)
    sides = (int(positive.sum()), int(negative.sum()))
    two_sided = min(sides) >= OUTLINE_SIDE_MIN_PTS       # decision §2.2: at least 3 outline points on each side
    width = float(np.median(across[positive]) - np.median(across[negative])) if two_sided else None
    return {"axis": axis, "share": share, "second": second, "two_sided": bool(two_sided), "n": int(len(outline)),
            "sides": list(sides), "width_m": width}


def edge_orientation(grey, roi_mask, exclude_mask, pose: HandPose, z: float, f_px: float) -> dict | None:
    """G-7 silhouette-edge orientation (decision §2.2; axis lens ``edge_hist``): Sobel gradients of the luma blurred
    by D10_EDGE_BLUR_PX; inside ``roi_mask`` minus ``exclude_mask`` (the scene passes the robot mask and the
    fingertip band) the strongest D10_EDGE_TOP_FRACTION of the gradient magnitudes vote; each edge pixel's direction
    (across its gradient) is carried onto the plane at height z (``image_directions_at_z``) and weighted by its
    magnitude; ``orientation_peak`` of those. An excluded pixel neither votes nor sets the magnitude quantile.
    Returns {"axis", "share", "second", "n" (voting pixels), "roi_px"} or None (an ROI of fewer than EDGE_MIN_ROI_PX
    pixels or fewer than EDGE_MIN_VOTES voting edges after the exclusion, the lens's minima, or no peak)."""
    image = np.asarray(_luma(grey), dtype=np.float32)
    roi = np.asarray(roi_mask, dtype=bool)
    if roi.shape != image.shape:
        raise ValueError("roi_mask must have the image's shape")
    if exclude_mask is not None:
        excluded = np.asarray(exclude_mask, dtype=bool)
        if excluded.shape != image.shape:
            raise ValueError("exclude_mask must have the image's shape")
        roi = roi & ~excluded
    if int(roi.sum()) < EDGE_MIN_ROI_PX:
        return None
    blurred = cv2.GaussianBlur(image, (0, 0), D10_EDGE_BLUR_PX)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    threshold = float(np.percentile(magnitude[roi], 100 * (1.0 - D10_EDGE_TOP_FRACTION)))
    vs, us = np.nonzero(roi & (magnitude >= max(threshold, 1e-6)))
    if len(us) < EDGE_MIN_VOTES:
        return None
    gxs, gys, mags = gx[vs, us].astype(float), gy[vs, us].astype(float), magnitude[vs, us].astype(float)
    edge = np.column_stack([-gys, gxs]) / np.maximum(mags, 1e-12)[:, None]
    dirs = image_directions_at_z(pose, np.column_stack([us, vs]).astype(float), edge, z, f_px,
                                 size_px=image.shape[1])
    axis, share, second = orientation_peak(dirs, mags)
    if axis is None:
        return None
    return {"axis": axis, "share": share, "second": second, "n": int(len(us)), "roi_px": int(roi.sum())}


def frame_alignment_turns(R: np.ndarray, external: "ExternalModel | None") -> int:
    """§3.6 item 2: counter-clockwise quarter turns k for the shown wrist image. Frontal hand: v1
    gravity_quarter_turns(R[2,1], R[2,0]). Top-down hand: the k whose turned image-right direction
    [R1, -R0, -R1, +R0][k] best matches the external image-right world direction (from J_ext's horizontal part);
    0 without an external model."""
    R = np.asarray(R, dtype=float)
    if robot.is_frontal(R[:, 2]):
        return int(gravity_quarter_turns(float(R[2, 1]), float(R[2, 0])))
    if external is None:
        return 0
    right = external.image_right_world_xy()
    if right is None:
        return 0
    candidates = [R[:2, 1], -R[:2, 0], -R[:2, 1], R[:2, 0]]
    scores = [float(c @ right) for c in candidates]
    return int(np.argmax(scores))


def _template_from_source(track: NCCTrack, angle_rad: float, scale: float):
    """The track's template rotated counter-clockwise on screen by ``angle_rad`` and scaled by ``scale`` (sampled
    from the track's larger source patch), shrunk to the largest radius the source covers; None if too small."""
    source = np.asarray(track._source, dtype=np.float32)
    source_r = (source.shape[0] - 1) // 2
    spread = abs(math.cos(angle_rad)) + abs(math.sin(angle_rad))
    radius = min(track.radius, int(math.floor((source_r - 1) * scale / spread)))
    if radius < (track.radius + 1) // 2:
        return None
    size = 2 * radius + 1
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    # output pixel p (centred, image u right / v down) samples the source at rot(-angle) p / scale; a CCW screen
    # rotation by +angle maps (du, dv) to (cos du + sin dv, -sin du + cos dv) with v down
    inverse = np.array([[cos, -sin], [sin, cos]]) / scale
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = inverse
    matrix[:, 2] = np.array([source_r, source_r]) - inverse @ np.array([radius, radius])
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(np.nan_to_num(source, nan=0.0), matrix, (size, size), flags=flags,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid = cv2.warpAffine(np.isfinite(source).astype(np.float32), matrix, (size, size), flags=flags,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if float(valid.min()) < 1.0 - 1e-6 or float(warped.std()) < track.min_std:
        return None
    return np.ascontiguousarray(warped, dtype=np.float32), inverse


def reacquire(track: NCCTrack, gray, predicted_uv, *, yaw_change_rad: float | None = None,
              kappa_ratio: float | None = None, sigma_px: float | None = None) -> tuple[np.ndarray | None, float]:
    """V3 re-acquisition of a v1 NCCTrack after a hand yaw / depth change: the template is rotated by the hand yaw
    about the approach since its capture (contract sense; the image content turns counter-clockwise on screen by
    +yaw) and scaled by the kappa ratio (new / old), plus v1's REACQUIRE_SCALES around it; searched +-V3_HALF_PX
    around the predicted point; accepted at peak >= V3_MIN_PEAK within V3_SIGMA_K * max(sigma_px, D4 pixel noise)
    of the prediction. None arguments mean no rotation, unit scale and zero prediction sigma. The track is not
    modified (the caller re-templates). Returns (uv | None, best peak)."""
    image = _gray32(gray)
    angle = 0.0 if yaw_change_rad is None else float(yaw_change_rad)
    base = 1.0 if kappa_ratio is None else float(kappa_ratio)
    if not base > 0.0:
        raise ValueError("kappa_ratio must be > 0")
    gate = V3_SIGMA_K * max(0.0 if sigma_px is None else float(sigma_px), D4_PIXEL_NOISE_PX)
    predicted = np.asarray(predicted_uv, dtype=float)
    offset = np.asarray(track.uv, dtype=float) - np.asarray(track._center, dtype=float)
    best_uv, best_accepted, best_peak = None, -1.0, -1.0
    for factor in (1.0,) + tuple(V3_SCALES):
        built = _template_from_source(track, angle, base * factor)
        if built is None:
            continue
        template, inverse = built
        moved_offset = np.linalg.inv(inverse) @ offset
        centre, peak = _match(image, template, predicted - moved_offset, V3_HALF_PX)
        best_peak = max(best_peak, peak)
        if centre is None:
            continue
        uv = centre + moved_offset
        if peak >= V3_MIN_PEAK and float(np.linalg.norm(uv - predicted)) <= gate and peak > best_accepted:
            best_uv, best_accepted = uv, peak
    return best_uv, (best_accepted if best_uv is not None else best_peak)


def fingertip_band_mask(open_rgb, closed_rgb) -> np.ndarray:
    """§4.5 item 6: the wrist-image region the fingers can occupy, from the free-air close difference (open-hold
    frame vs the end of the second close, same pose). Changed pixels (> HM_DIFF_GREY grey) in components of at
    least HM_MIN_PIXELS; between the leftmost and rightmost changed columns, every pixel from the (interpolated)
    topmost changed row down to the image bottom (the fingers enter from the bottom edge and sweep across it while
    closing); dilated by FINGERTIP_BAND_MARGIN_PX."""
    open_grey, closed_grey = _luma(open_rgb), _luma(closed_rgb)
    if open_grey.shape != closed_grey.shape:
        raise ValueError("frames must have the same size")
    changed = (np.abs(open_grey - closed_grey) > HM_DIFF_GREY).astype(np.uint8)
    # a sub-mm hand drift while the fingers close flickers every object contour (thin lines); the moving fingers
    # are solid blobs: an opening removes the lines
    opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FINGERTIP_BAND_OPEN_PX, FINGERTIP_BAND_OPEN_PX))
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HM_DILATE_PX, HM_DILATE_PX))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    keep = np.zeros(changed.shape, dtype=bool)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= HM_MIN_PIXELS:
            keep |= labels == label
    band = np.zeros(changed.shape, dtype=bool)
    columns = np.nonzero(keep.any(axis=0))[0]
    if columns.size == 0:
        return band
    tops = np.array([int(np.argmax(keep[:, c])) for c in columns], dtype=float)
    span = np.arange(int(columns.min()), int(columns.max()) + 1)
    top_rows = np.interp(span, columns, tops)
    for column, top in zip(span, top_rows):
        band[int(math.floor(top)):, column] = True
    margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FINGERTIP_BAND_MARGIN_PX + 1,
                                                           2 * FINGERTIP_BAND_MARGIN_PX + 1))
    return cv2.dilate(band.astype(np.uint8), margin) > 0


def fingertip_band_top_row(band) -> int | None:
    """Topmost row of a fingertip band mask (the v1 FINGERTIP_BAND_ROW_FRACTION replacement), or None."""
    rows = np.nonzero(np.asarray(band, dtype=bool).any(axis=1))[0]
    return None if rows.size == 0 else int(rows.min())


def false_change_map(static_frames: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
    """§4.5 item 5 from frames of a static robot and scene: (per-pixel share of consecutive frame pairs whose grey
    difference exceeds HM_DIFF_GREY, float32; sigma_px = the static grey noise, rms difference / sqrt 2)."""
    frames = [_luma(frame) for frame in static_frames]
    if not frames:
        raise ValueError("no frames")
    if len(frames) < 2:
        return np.zeros(frames[0].shape, dtype=np.float32), 0.0
    diffs = [np.abs(b - a) for a, b in zip(frames[:-1], frames[1:])]
    stack = np.stack(diffs)
    probability = np.mean(stack > HM_DIFF_GREY, axis=0).astype(np.float32)
    sigma = math.sqrt(float(np.mean(stack ** 2)) / 2.0)
    return probability, sigma


# ================================================================================================ ExternalModel
class ExternalModel:
    """§4.5 ExternalModel (R-1x + R-1v): J (2x3 px per metre of hand xyz) from the robot's own probe moves;
    v_hat = unit(J (0, 0, -1)), the image direction of world down (the fingertips of a top-down hand are the extreme
    of its image along +v_hat); n_hat = v_hat turned so that it points image-right for a downward v_hat ((v_y, -v_x));
    ray_dir = the unit null vector of J with d_z < 0 (the external camera's viewing ray at the anchor)."""

    def __init__(self, J, *, residual_deg=None, axes_summary=None, thresholds=None, rejected=None, samples=None):
        self.J = np.asarray(J, dtype=float).reshape(2, 3)
        if not np.isfinite(self.J).all():
            raise ValueError("J must be finite")
        self.residual_deg = [float(v) for v in (residual_deg or [])]
        self.axes_summary = dict(axes_summary or {})
        self.thresholds = list(thresholds or [])
        self.rejected = list(rejected or [])
        self.samples = samples
        self.broyden_updates = 0
        self.broyden_rejected = 0
        self._derive()

    def _derive(self):
        self.v_hat = _unit(self.J @ np.array([0.0, 0.0, -1.0]))
        self.n_hat = np.array([self.v_hat[1], -self.v_hat[0]])
        _, _, vt = np.linalg.svd(self.J)
        ray = vt[-1]
        self.ray_dir = ray if ray[2] < 0.0 else -ray

    @classmethod
    def fit(cls, external_samples: list[dict]) -> "ExternalModel | None":
        """Least-squares J over the probe moves (S = J W); D14: every move's angle between J w and its measured hand
        shift <= D14_MAX_RESIDUAL_DEG. A failing move is dropped (the worst first) while D14_MIN_SAMPLES remain;
        otherwise None. Unusable moves (no flow, < 2 cm, < 3 px) are skipped."""
        usable, skipped = [], []
        for sample in external_samples:
            shift = sample.get("image_shift_px")
            world = np.asarray(sample.get("world_displacement_m"), dtype=float)
            if shift is None or world.shape != (3,):
                skipped.append({"stage": sample.get("stage"), "reason": "no_flow"})
            elif float(np.linalg.norm(world)) < D14_MIN_MOTION_M:
                skipped.append({"stage": sample.get("stage"), "reason": "too_little_motion"})
            elif float(np.linalg.norm(shift)) < D14_MIN_IMAGE_PX:
                skipped.append({"stage": sample.get("stage"), "reason": "no_image_motion"})
            else:
                usable.append((sample.get("stage"), world, np.asarray(shift, dtype=float)))
        rejected = list(skipped)
        while True:
            if len(usable) < 3:
                return None
            W = np.stack([w for _, w, _ in usable], axis=1)
            S = np.stack([s for _, _, s in usable], axis=1)
            singular = np.linalg.svd(W, compute_uv=False)
            if singular[-1] < 1e-9 or singular[0] / singular[-1] > D14_MAX_CONDITION:
                return None
            J = S @ np.linalg.pinv(W)
            residuals = [_angle_deg(J @ w, s) for _, w, s in usable]
            worst = int(np.argmax(residuals))
            if residuals[worst] <= D14_MAX_RESIDUAL_DEG:
                break
            if len(usable) - 1 < D14_MIN_SAMPLES:
                return None
            # a corrupted move bends the joint fit, so the largest joint residual can sit on a good move: drop the
            # move whose removal leaves the most consistent set (every remaining move predicted by the others,
            # leave-one-out, within D14); when none does (e.g. 3 horizontal moves, where any 2 fit exactly and
            # the culprit is not identifiable) the fit is rejected, as v1 does
            best, best_score = None, math.inf
            for index in range(len(usable)):
                check = _leave_one_out_deg([item for k, item in enumerate(usable) if k != index])
                if any(value is None for value in check):
                    continue
                if max(check) < best_score:
                    best, best_score = index, max(check)
            if best is None or best_score > D14_MAX_RESIDUAL_DEG:
                return None
            loo = _leave_one_out_deg(usable)
            worst = best
            stage, _, _ = usable.pop(worst)
            rejected.append({"stage": stage, "reason": "residual", "residual_deg": residuals[worst],
                             "leave_one_out_deg": loo[worst]})
        horizontal = [{"world_displacement_m": list(w), "image_shift_px": list(s)} for _, w, s in usable
                      if abs(float(w[2])) < float(np.linalg.norm(w[:2]))]
        summary = fit_external_axes(horizontal)
        threshold = th_value("D14_residual_deg", D14_MAX_RESIDUAL_DEG, "DD",
                             "D14: max angle between J dp and the measured hand image shift (v1 MAX_RESIDUAL_DEG)",
                             max_residual_deg=max(residuals), moves=len(usable))
        return cls(J, residual_deg=residuals, axes_summary=summary, thresholds=[threshold], rejected=rejected,
                   samples=[stage for stage, _, _ in usable])

    def broyden(self, dp_m, ds_px) -> bool:
        """Rank-one secant J += (ds - J dp) dp^T / |dp|^2 on a clean non-rotating step (the caller checks the
        rotation); rejected when |dp| < BROYDEN_MIN_STEP_M or the innovation exceeds max(3 px, 0.5 |J dp|)."""
        dp, ds = np.asarray(dp_m, dtype=float), np.asarray(ds_px, dtype=float)
        norm2 = float(dp @ dp)
        if math.sqrt(norm2) < BROYDEN_MIN_STEP_M:
            return False
        predicted = self.J @ dp
        innovation = ds - predicted
        limit = max(BROYDEN_INNOVATION_FLOOR_PX, BROYDEN_MAX_INNOVATION * float(np.linalg.norm(predicted)))
        if not np.isfinite(innovation).all() or float(np.linalg.norm(innovation)) > limit:
            self.broyden_rejected += 1
            return False
        self.J = self.J + np.outer(innovation, dp) / norm2
        self._derive()
        self.broyden_updates += 1
        return True

    def predict_shift(self, dp_m) -> np.ndarray:
        return self.J @ np.asarray(dp_m, dtype=float)

    def n_row(self, s_a_px, s_b_px) -> float:
        """n_hat . (s_a - s_b): parallax-free in the affine model for a hand-attached feature against a static one."""
        return float(self.n_hat @ (np.asarray(s_a_px, dtype=float) - np.asarray(s_b_px, dtype=float)))

    def image_right_world_xy(self) -> np.ndarray | None:
        """World horizontal direction (unit xy) whose motion moves the image right (J's 2x2 horizontal part)."""
        block = self.J[:, :2]
        if abs(float(np.linalg.det(block))) < 1e-9:
            return None
        return _unit(np.linalg.solve(block, np.array([1.0, 0.0])))

    def to_dict(self) -> dict:
        return jsonable({"J": self.J, "residual_deg": self.residual_deg, "v_hat": self.v_hat, "n_hat": self.n_hat,
                         "ray_dir": self.ray_dir, "samples": self.samples, "rejected": self.rejected,
                         "broyden_updates": self.broyden_updates, "axes_available": self.axes_summary.get("available"),
                         "thresholds": [t.to_dict() for t in self.thresholds]})


def _leave_one_out_deg(moves) -> list[float | None]:
    """Per (stage, w, s) move: the D14 angle of its shift against the J fitted on the other moves (None when the
    others do not span 3-D within D14_MAX_CONDITION)."""
    out = []
    for index, (_, w, s) in enumerate(moves):
        others = [item for k, item in enumerate(moves) if k != index]
        if len(others) < 3:
            out.append(None)
            continue
        W = np.stack([item[1] for item in others], axis=1)
        values = np.linalg.svd(W, compute_uv=False)
        if values[-1] < 1e-9 or values[0] / values[-1] > D14_MAX_CONDITION:
            out.append(None)
            continue
        J = np.stack([item[2] for item in others], axis=1) @ np.linalg.pinv(W)
        out.append(_angle_deg(J @ w, s))
    return out


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return math.degrees(math.pi)
    return math.degrees(math.acos(float(np.clip(a @ b / denominator, -1.0, 1.0))))


# ================================================================================================ HandLocator
def _shifted(grey, shift_px):
    """grey(p + shift) by bilinear resampling (border replicated)."""
    matrix = np.array([[1.0, 0.0, float(shift_px[0])], [0.0, 1.0, float(shift_px[1])]])
    return cv2.warpAffine(np.asarray(grey, dtype=np.float32), matrix, (grey.shape[1], grey.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def motion_outline(before_rgb, after_rgb, motion_px=None, *, window=None) -> np.ndarray | None:
    """The moving hand's outline in one external frame pair: pixels whose grey changed by more than HM_DIFF_GREY
    (NOT dilated, so the silhouette's extreme is not pushed outward) and, given the hand's image motion m, that
    are motion-consistent (before(p) = after(p + m) or after(p) = before(p - m) within HM_DIFF_GREY: covered or
    uncovered by the moving hand; in-place changes such as arm reflections on metal fail it). Inside ``window``
    every piece of at least HL_MIN_BLOB_PX is kept (fingertips are separate pieces when the finger bodies above
    them barely change); without a window, the largest piece after a HM_DILATE_PX closing. None when fewer than
    HM_MIN_PIXELS remain. It is the union of the hand silhouettes of BOTH frames: its extreme along m belongs to the
    later frame, against m to the earlier one."""
    before, after = _luma(before_rgb).astype(np.float32), _luma(after_rgb).astype(np.float32)
    if before.shape != after.shape:
        raise ValueError("frames must have the same size")
    changed = np.abs(before - after) > HM_DIFF_GREY
    if motion_px is not None:
        m = np.asarray(motion_px, dtype=float)
        covered = np.abs(after - _shifted(before, -m)) <= HM_DIFF_GREY
        uncovered = np.abs(before - _shifted(after, m)) <= HM_DIFF_GREY
        changed &= covered | uncovered
    if window is not None:
        changed &= np.asarray(window, dtype=bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
        keep = np.zeros(changed.shape, dtype=bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= HL_MIN_BLOB_PX:
                keep |= labels == label
        return keep if int(keep.sum()) >= HM_MIN_PIXELS else None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HM_DILATE_PX, HM_DILATE_PX))
    closed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < HM_MIN_PIXELS:
        return None
    return labels == best


def finger_extremes_px(model: ExternalModel, robot_model: RobotModel, pose: HandPose) -> dict:
    """Per finger, the image offset (top along v_hat, across along n_hat; px from the grip pixel) of its fingertip
    extreme by the rule of ``mask_extreme``: the bottom face of its landing footprint (R5 reach t_f beyond the grip
    along R2, the asset footprint ranges at the current per-finger opening, robot facts R5 / R8 [C-50]) mapped by J,
    the corners within L1_TIP_BAND_PX of the lowest one along v_hat averaged across. {"finger1": (top, across),
    "finger2": (top, across)}."""
    q_mm = max(pose.gap_mm, 0.0) / 2.0
    ranges = robot_model.finger_ranges_mm(q_mm)
    v_hat, n_hat = model.v_hat, model.n_hat
    fingers = {}
    for finger in ("finger1", "finger2"):
        (r1_lo, r1_hi), (r0_lo, r0_hi) = ranges[finger]["R1_mm"], ranges[finger]["R0_mm"]
        corners = []
        for r0 in (r0_lo, r0_hi):
            for r1 in (r1_lo, r1_hi):
                offset = pose.R @ np.array([r0 / 1000.0, r1 / 1000.0, robot_model.t_f_m])
                corners.append(model.J @ offset)
        corners = np.array(corners)
        along = corners @ v_hat
        top = float(along.max())
        band = along >= top - L1_TIP_BAND_PX
        fingers[finger] = (top, float(np.mean(corners[band] @ n_hat)))
    return fingers


def fingertip_offset_px(model: ExternalModel, robot_model: RobotModel, pose: HandPose) -> np.ndarray:
    """Image offset of the fingertip extreme (the rule of ``mask_extreme``) from the grip pixel, from the robot's
    own geometry: the mean of the two fingers' ``finger_extremes_px``. The silhouette's lowest point along v_hat is a
    footprint corner (the one toward the camera), not the grip axis."""
    fingers = list(finger_extremes_px(model, robot_model, pose).values())
    # the two fingers are symmetric about the grip axis: their mean extreme point (choosing one finger would flip
    # with the sign of a small cross term of J)
    top = float(np.mean([item[0] for item in fingers]))
    across = float(np.mean([item[1] for item in fingers]))
    return top * model.v_hat + across * model.n_hat


def finger_band_depth_px(model: ExternalModel, robot_model: RobotModel, pose: HandPose) -> float:
    """G-8 V2v finger band depth D_f (px along v_hat; ROBOT + ONLINE): the image length of the fingers below the palm
    at this pose, from the fingertip bottom faces (R5: t_f beyond the grip along R2) up to the LOWEST image point of
    the palm underside (R6: p_min behind the grip, over the R7 hand footprint rectangle in R0 x R1), mapped by J
    relative to the grip; at least L1_TIP_BAND_PX. Every robot number (t_f, p_min and the R7 footprint) is read from
    ``robot_model``: the footprint corners come from its ``hand_footprint_xy`` in the hand's own (R1, R0) frame (grip
    at the origin, jaw along +x), so a non-default RobotModel stays self-consistent (review-2 implementation review
    R-8)."""
    R = np.asarray(pose.R, dtype=float)
    v_hat = model.v_hat
    tip = float((model.J @ (R @ np.array([0.0, 0.0, robot_model.t_f_m]))) @ v_hat)
    corners = np.asarray(robot_model.hand_footprint_xy((0.0, 0.0), (1.0, 0.0)), dtype=float)   # rows (R1, R0) m
    palm = [float((model.J @ (R @ np.array([r0, r1, -robot_model.p_min_m]))) @ v_hat) for r1, r0 in corners]
    return max(tip - max(palm), L1_TIP_BAND_PX)


def finger_pair_across(outline, motion_px, v_hat, tip_before, depth_px: float,
                       separation_px: float) -> tuple[float | None, dict]:
    """G-8 V2v (design review 2 decision §2.2; the locator lens's validated finger-pair rule ``locate_pair``): the
    across coordinate (along n_hat, BEFORE frame) of the grip from the two fingers of a moving outline
    (``motion_outline``). (1) Finger band: the outline pixels within ``depth_px`` (D_f, ``finger_band_depth_px``) of
    the before-frame tip along v_hat (when the hand moved down the image by more than the tip band, the extreme
    belongs to the after frame and the outline is shifted back by the motion first, as ``outline_tip_before`` does).
    (2) Split at the widest gap of the across coordinates: exactly two pieces of at least HL_MIN_BLOB_PX pixels each,
    else "finger_pair_too_few" (fewer than 2 x HL_MIN_BLOB_PX band pixels) or "finger_pair_one_piece" (also when the
    widest gap is at most sqrt 2 px: a connected band, BG refinement of the lens rule). (3) Validate
    against the robot: |measured - predicted separation| <= PROJECTOR_AFFINE_FRACTION x predicted + L1_FLOOR_PX
    (``separation_px`` = the footprint model's two-finger separation across), else "finger_pair_inconsistent". (4)
    Across = the mean of the two piece centroids (pixel counts do not weight one finger), minus half the pair's
    across motion (the union holds both frames, as in V0). Returns (across | None, info with "reason")."""
    m = np.asarray(motion_px, dtype=float)
    v = _unit(np.asarray(v_hat, dtype=float)[:2])
    n = np.array([v[1], -v[0]])
    vs, us = np.nonzero(outline)
    points = np.c_[us, vs].astype(float)
    if float(m @ v) > L1_TIP_BAND_PX:
        points = points - m
    band = points[points @ v >= float(np.asarray(tip_before, dtype=float) @ v) - float(depth_px)]
    across = np.sort(band @ n)
    info = {"depth_px": float(depth_px), "band_px": int(across.size), "separation_pred_px": float(separation_px)}
    if across.size < 2 * HL_MIN_BLOB_PX:
        return None, {**info, "reason": "finger_pair_too_few"}
    gaps = np.diff(across)
    k = int(np.argmax(gaps))
    first, second = across[:k + 1], across[k + 1:]
    info.update(gap_px=float(gaps[k]), pieces_px=[int(first.size), int(second.size)])
    # a widest gap of at most one 8-neighbour pixel diagonal (sqrt 2 px) splits no piece: a connected band
    # projects without one, so it is one piece however the sort happens to split it
    if float(gaps[k]) <= math.sqrt(2.0) or min(first.size, second.size) < HL_MIN_BLOB_PX:
        return None, {**info, "reason": "finger_pair_one_piece"}
    measured = float(second.mean() - first.mean())
    info["separation_px"] = measured
    if abs(measured - float(separation_px)) > PROJECTOR_AFFINE_FRACTION * float(separation_px) + L1_FLOOR_PX:
        return None, {**info, "reason": "finger_pair_inconsistent"}
    return 0.5 * float(first.mean() + second.mean()) - 0.5 * float(m @ n), {**info, "reason": None}


def outline_tip_before(outline, motion_px, v_hat) -> np.ndarray | None:
    """The BEFORE frame's fingertip from the union outline of a frame pair (``motion_outline``) and the hand's image
    motion m: the outline's extreme along v_hat belongs to the after frame when m moves along +v_hat by more than
    the tip band (then m is subtracted), to the before frame when it moves against it; when |m . v_hat| is within
    the band, both frames' tips are in the band and its across-centroid is midway (half of m's across part is
    subtracted)."""
    tip = mask_extreme(outline, v_hat)
    if tip is None:
        return None
    m = np.asarray(motion_px, dtype=float)
    v_hat = _unit(np.asarray(v_hat, dtype=float)[:2])
    n_hat = np.array([v_hat[1], -v_hat[0]])
    along = float(m @ v_hat)
    if along > L1_TIP_BAND_PX:
        return tip - m
    if along < -L1_TIP_BAND_PX:
        return tip
    return tip - 0.5 * float(m @ n_hat) * n_hat


def hand_tip_before(before_rgb, after_rgb, motion_px, v_hat, *, band_px: float) -> np.ndarray | None:
    """The BEFORE frame's fingertip pixel of a moving hand from the motion-consistent union outline of the pair
    (``motion_outline``, the largest piece: the moving hand, not in-place changes such as reflections): along v_hat
    its extreme (``outline_tip_before``: the silhouette's lowest point, sharp); across v_hat the centroid of its
    lowest ``band_px`` (fingers and the lower hand, symmetric about the grip axis along the jaw), which is steadier
    than the few tip pixels; the union holds both frames, so half of m's across part is subtracted. A hand moving
    ACROSS v_hat still smears that band: ``HandLocator`` locates on moves within L1_MAX_ACROSS_DEG of v_hat."""
    m = np.asarray(motion_px, dtype=float)
    outline = motion_outline(before_rgb, after_rgb, m)
    tip = outline_tip_before(outline, m, v_hat)
    if tip is None:
        return None
    v_hat = _unit(np.asarray(v_hat, dtype=float)[:2])
    n_hat = np.array([v_hat[1], -v_hat[0]])
    vs, us = np.nonzero(outline)
    points = np.c_[us, vs].astype(float)
    along = points @ v_hat
    keep = along >= along.max() - float(band_px)
    across = float(np.mean(points[keep] @ n_hat)) - 0.5 * float(m @ n_hat)
    return float(tip @ v_hat) * v_hat + across * n_hat


def hand_band_shift(before_rgb, after_rgb, v_hat, *, band_px: float) -> tuple[np.ndarray | None, dict]:
    """Image motion of the HAND (not of the arm links, which the v1 fastest decile picks: in the task-2 agentview
    they move ~20 percent faster in the image than the grip) between two external frames: the moving pixels (v1
    changed and flowing mask) within ``band_px`` of their extreme along v_hat (the fingers and the hand's lowest
    part) are matched as a masked template (SQDIFF, sub-pixel peak). The template comes from the frame in which
    that band is the hand itself: the before frame when the hand moves up the image, else the after frame (the
    reversed match), since a moving mask's leading band is flow bleed over the background. Returns (shift | None,
    info)."""
    before, after = _luma(before_rgb).astype(np.float32), _luma(after_rgb).astype(np.float32)
    flow_info = hand_motion(before_rgb, after_rgb)
    if flow_info is None:
        return None, {"reason": "no_hand_motion"}
    flow = np.asarray(flow_info["image_shift_px"], dtype=float)
    reverse = float(flow @ np.asarray(v_hat, dtype=float)[:2]) > 0.0
    source, target = (after, before) if reverse else (before, after)
    selected = _moving_pixels(after_rgb, before_rgb) if reverse else _moving_pixels(before_rgb, after_rgb)
    if selected is None:
        return None, {"reason": "no_hand_motion"}
    vs, us = np.nonzero(selected)
    along = np.c_[us, vs].astype(float) @ _unit(np.asarray(v_hat, dtype=float)[:2])
    keep = along >= along.max() - float(band_px)
    mask = np.zeros(selected.shape, dtype=bool)
    mask[vs[keep], us[keep]] = True
    if int(mask.sum()) < HM_MIN_PIXELS:
        return None, {"reason": "band_too_small"}
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    search = int(math.ceil(float(np.linalg.norm(flow)))) + FT_SEARCH_PX
    h, w = source.shape
    R0, R1, C0, C1 = max(0, r0 - search), min(h, r1 + search), max(0, c0 - search), min(w, c1 + search)
    template = np.ascontiguousarray(source[r0:r1, c0:c1])
    weights = np.ascontiguousarray(mask[r0:r1, c0:c1].astype(np.float32))
    score = cv2.matchTemplate(np.ascontiguousarray(target[R0:R1, C0:C1]), template, cv2.TM_SQDIFF, mask=weights)
    score = np.nan_to_num(score, nan=np.inf)
    row, col = np.unravel_index(int(np.argmin(score)), score.shape)
    finite = score[np.isfinite(score)]
    scale = float(finite.max()) if finite.size and float(finite.max()) > 0.0 else 1.0
    sub = subpixel_peak(-np.where(np.isfinite(score), score, scale) / scale, row, col)
    shift = np.array([C0 + col + sub[0] - c0, R0 + row + sub[1] - r0])
    if reverse:
        shift = -shift
    return shift, {"reverse": reverse, "band_pixels": int(mask.sum()), "flow_px": flow.tolist()}


def _moving_pixels(before_rgb, after_rgb) -> np.ndarray | None:
    """v1 hand_image_motion's moving set (changed by > HM_DIFF_GREY, dilated, flow > HM_MIN_FLOW_PX), before frame."""
    before, after = _luma_u8(before_rgb), _luma_u8(after_rgb)
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma = HM_FARNEBACK
    flow = cv2.calcOpticalFlowFarneback(before, after, None, pyr_scale, int(levels), int(winsize), int(iterations),
                                        int(poly_n), poly_sigma, 0)
    changed = np.abs(before.astype(int) - after.astype(int)) > HM_DIFF_GREY
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((HM_DILATE_PX, HM_DILATE_PX), np.uint8)) > 0
    selected = changed & (np.linalg.norm(flow, axis=2) > HM_MIN_FLOW_PX)
    return selected if int(selected.sum()) >= HM_MIN_PIXELS else None


class HandLocator:
    """§4.5 HandLocator (L1): the grip's ABSOLUTE external pixel from a frame pair of a moving hand. Along v_hat: the
    extreme of the moving outline (``motion_outline``: the undilated frame difference, which is the union of both
    frames' silhouettes, so the extreme is assigned to the frame it belongs to from the hand's image motion J dp);
    across (design review 2 decision §2.2 G-8, V2v): the mean of the two finger pieces of the outline's finger band
    (``finger_pair_across``), validated against the footprint model's finger separation, else the reading is refused
    ("finger_pair_too_few", "finger_pair_one_piece", "finger_pair_inconsistent"); u_grip = u_tip -
    ``fingertip_offset_px`` (the footprint corner model; for a fingertip on the grip axis it is u_tip + J (-t_f R2)).
    ``fit`` (the calibration moves) is unchanged (V0's extreme-band tips).

    Preconditions (``motion_check``, enforced by both ``fit`` and ``locate``): the hand's image motion between the two
    frames lies within L1_MAX_ACROSS_DEG of v_hat (a move across v_hat smears the hand band's centroid) and its
    component along v_hat exceeds L1_MIN_ALONG_PX (the before and after fingertips are separate extremes); a 1-step
    pair of a probe leg (about 1 px) or a horizontal leg across v_hat is refused, a vertical move of 2-3 cm is not.

    Validated on the probe moves: located displacements against J dp. ``sigma_consistency_px`` = one location's
    radial rms from the displacement residuals (rms / sqrt 2); it cancels any offset common to all locations, so
    ``sigma_loc_px`` = sigma_consistency (+) L1_MODEL_SIGMA_PX (the absolute model error measured against the
    evaluator projection, re-derived at V5)."""

    def __init__(self, model: ExternalModel, robot_model: RobotModel, *, sigma_loc_px: float, residual_px,
                 thresholds=None, located=None, sigma_consistency_px: float | None = None, degraded: bool = False,
                 dropped=None):
        self.model = model
        self.robot = robot_model
        self.sigma_loc_px = float(sigma_loc_px)
        self.sigma_consistency_px = float(sigma_loc_px if sigma_consistency_px is None else sigma_consistency_px)
        self.residual_px = [float(v) for v in residual_px]
        self.thresholds = list(thresholds or [])
        self.located = list(located or [])
        self.degraded = bool(degraded)
        self.dropped = list(dropped or [])
        self.last_locate: dict | None = None

    @property
    def flags(self) -> tuple[str, ...]:
        """("calibration_degraded",) when the locator was fitted on two agreeing locations only (G-5), else ()."""
        return (DEGRADED_FLAG,) if self.degraded else ()

    @staticmethod
    def motion_check(model: ExternalModel, motion_px) -> str | None:
        """None when a hand image motion meets the L1 preconditions, else the reason ("no_motion", "across_v_hat",
        "motion_too_small")."""
        m = np.asarray(motion_px, dtype=float)
        norm = float(np.linalg.norm(m))
        if not norm > 1e-9:
            return "no_motion"
        if abs(float(m @ model.n_hat)) > math.sin(math.radians(L1_MAX_ACROSS_DEG)) * norm:
            return "across_v_hat"
        if abs(float(m @ model.v_hat)) < L1_MIN_ALONG_PX:
            return "motion_too_small"
        return None

    @classmethod
    def fit(cls, external_samples, model: ExternalModel, robot: RobotModel) -> "HandLocator | None":
        """From the probe moves' start-frame fingertips (``mask_extreme_uv`` with ``start_state8`` and
        ``end_state8``; moves failing ``motion_check`` on J dp are skipped): every pair of located grips must move as J
        predicts within sqrt 2 x the per-location bound (``consistency_threshold``: max(L1_FLOOR_PX, L1_SIGMA_K robust
        sigma) with more than L1_MIN_SAMPLES locations, the absolute L1_SIGMA_K x L1_FLOOR_PX with at most that many,
        decision §2.3 G-5); the location most involved in failing pairs is dropped while L1_MIN_SAMPLES remain (a
        tie, e.g. one failing pair of three, drops the tied end whose removal leaves the smaller largest residual
        among the remaining pairs; an exact tie gives None). Two agreeing locations (left after a drop, or the only
        two) still fit, flagged ``degraded`` (outcome reason calibration_degraded, not None); two that disagree, or
        fewer than two, give None."""
        rows = []
        for sample in external_samples:
            tip = sample.get("mask_extreme_uv")
            state, end = sample.get("start_state8"), sample.get("end_state8")
            if tip is None or state is None or end is None:
                continue
            m = model.J @ (np.asarray(end[:3], dtype=float) - np.asarray(state[:3], dtype=float))
            if cls.motion_check(model, m) is not None:
                continue                      # across v_hat (smeared band centroid) or too short a move
            pose = HandPose.from_state8(state)
            rows.append({"stage": sample.get("stage"), "grip": pose.grip.copy(),
                         "uv": np.asarray(tip, dtype=float) - fingertip_offset_px(model, robot, pose)})
        dropped = []
        while len(rows) >= 2:
            pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
            residuals = [float(np.linalg.norm((rows[j]["uv"] - rows[i]["uv"])
                                              - model.J @ (rows[j]["grip"] - rows[i]["grip"]))) for i, j in pairs]
            threshold = cls.consistency_threshold(residuals, len(rows))
            limit = math.sqrt(2.0) * threshold.value
            failing = [k for k, r in enumerate(residuals) if r > limit]
            if not failing:
                consistency = math.sqrt(float(np.mean(np.square(residuals))) / 2.0)
                return cls(model, robot, sigma_loc_px=math.hypot(consistency, L1_MODEL_SIGMA_PX),
                           sigma_consistency_px=consistency, residual_px=residuals, thresholds=[threshold],
                           located=[{"stage": r["stage"], "uv": r["uv"].tolist(), "grip": r["grip"].tolist()}
                                    for r in rows], degraded=len(rows) < L1_MIN_SAMPLES, dropped=dropped)
            if len(rows) < L1_MIN_SAMPLES:
                return None                   # two locations that disagree: the culprit is not identifiable
            counts = np.zeros(len(rows), dtype=int)
            for k in failing:
                i, j = pairs[k]
                counts[i] += 1
                counts[j] += 1
            tied = [int(k) for k in np.flatnonzero(counts == counts.max())]
            choice = tied[0]
            remaining = None
            if len(tied) > 1:
                # the failing pairs do not single out one location (one failing pair of three: both ends equally
                # suspect; list order would drop the first-listed end, a clean move in 5/15 real cases at a 12 px
                # fault): drop the end whose removal leaves the most consistent rest (the smallest largest pair
                # residual among the remaining locations); an exact tie stays unidentifiable -> None
                rest = {k: max((residuals[n] for n, (i, j) in enumerate(pairs) if k not in (i, j)), default=0.0)
                        for k in tied}
                best = min(rest.values())
                choice_set = [k for k in tied if rest[k] <= best]
                if len(choice_set) > 1:
                    return None
                choice, remaining = choice_set[0], best
            worst = rows.pop(choice)
            dropped.append({"stage": worst["stage"], "uv": worst["uv"].tolist(), "limit_px": limit,
                            "failing_pairs": int(counts.max()), "tied": len(tied),
                            "rest_max_residual_px": remaining})
        return None

    @staticmethod
    def consistency_threshold(residuals, locations: int) -> Threshold:
        """The per-location L1 bound for ``locations`` located moves with these pair residuals (px): more than
        L1_MIN_SAMPLES: max(L1_FLOOR_PX, L1_SIGMA_K x the robust sigma RAYLEIGH_MEDIAN_TO_RMS x median / sqrt 2); at
        most L1_MIN_SAMPLES: the absolute L1_SIGMA_K x L1_FLOOR_PX (G-5: with 3 pairs a single fault is in 2 of
        them and sets the median, so the self-scaled bound accepted a 40 px fault in 15/15 real cases)."""
        if int(locations) > L1_MIN_SAMPLES:
            robust = RAYLEIGH_MEDIAN_TO_RMS * float(np.median(residuals)) / math.sqrt(2.0)
            return th_max("L1_locator_px", Term("DD", L1_FLOOR_PX, "L1_FLOOR_PX"),
                          Term("ONLINE", L1_SIGMA_K * robust, "3*sigma_loc_robust"), unit="px per location")
        return th_value("L1_locator_px", L1_SIGMA_K * L1_FLOOR_PX, "DD",
                        "L1_SIGMA_K x L1_FLOOR_PX: the absolute bound with at most L1_MIN_SAMPLES located moves "
                        "(decision §2.3 G-5; pairs within sqrt 2 x this = 12.7 px)", locations=int(locations))

    def locate(self, before_rgb, after_rgb, pose: HandPose, *,
               after_pose: HandPose | None = None) -> tuple[np.ndarray | None, float]:
        """(u_grip of the hand in the BEFORE frame, sigma_loc_px); (None, sigma) when the pair fails the L1
        preconditions (``motion_check``), no moving outline is found or V2v refuses the finger pair (G-8); the reason
        is kept in ``last_locate`` (with V2v's numbers under "v2v" and V0's reading under "v0_uv", a diagnosis).
        ``pose`` is the before frame's hand pose; ``after_pose`` (the after frame's) gives the hand's image motion
        J dp (the calibration J, valid near the probe moves), else the measured hand motion (hand_band_shift;
        HL_HAND_BAND_M sets only that band now, no longer the across coordinate)."""
        band_px = float(np.linalg.norm(self.model.J @ np.array([0.0, 0.0, HL_HAND_BAND_M])))
        motion, _ = hand_band_shift(before_rgb, after_rgb, self.model.v_hat, band_px=band_px)
        source = "measured"
        if after_pose is not None:
            predicted = self.model.J @ (after_pose.grip - pose.grip)
            # the measured hand motion is used when it agrees with J dp (D14's angle and L1_MOTION_AGREEMENT length)
            if motion is None or _angle_deg(motion, predicted) > D14_MAX_RESIDUAL_DEG or \
                    abs(float(np.linalg.norm(motion)) / max(float(np.linalg.norm(predicted)), 1e-9) - 1.0) > \
                    L1_MOTION_AGREEMENT:
                motion, source = predicted, "J_dp"
        if motion is None:
            self.last_locate = {"reason": "no_hand_motion", "motion_px": None, "source": source}
            return None, self.sigma_loc_px
        motion = np.asarray(motion, dtype=float)
        info = {"motion_px": motion.tolist(), "along_px": float(motion @ self.model.v_hat),
                "across_px": float(motion @ self.model.n_hat), "source": source}
        reason = self.motion_check(self.model, motion)
        if reason is not None:
            self.last_locate = {"reason": reason, **info}
            return None, self.sigma_loc_px
        v_hat, n_hat = self.model.v_hat, self.model.n_hat
        outline = motion_outline(before_rgb, after_rgb, motion)
        tip = outline_tip_before(outline, motion, v_hat)
        if tip is None:
            self.last_locate = {"reason": "no_outline", **info}
            return None, self.sigma_loc_px
        # design review 2 decision §2.2 G-8: the across coordinate is V2v's finger pair (V0's centroid of the lowest
        # HL_HAND_BAND_M of the outline, kept below as a diagnosis, read the hand body: 5.7-9.9 px at RS-4)
        fingers = finger_extremes_px(self.model, self.robot, pose)
        separation = abs(fingers["finger1"][1] - fingers["finger2"][1])
        depth = finger_band_depth_px(self.model, self.robot, pose)
        across, pair = finger_pair_across(outline, motion, v_hat, tip, depth, separation)
        offset = fingertip_offset_px(self.model, self.robot, pose)
        vs, us = np.nonzero(outline)
        points = np.c_[us, vs].astype(float)
        lowest = points @ v_hat >= float((points @ v_hat).max()) - band_px
        v0_across = float(np.mean(points[lowest] @ n_hat)) - 0.5 * float(motion @ n_hat)
        info.update(v2v=pair, v0_uv=(float(tip @ v_hat) * v_hat + v0_across * n_hat - offset).tolist())
        if across is None:
            self.last_locate = {"reason": pair["reason"], **info}
            return None, self.sigma_loc_px
        self.last_locate = {"reason": None, **info}
        return float(tip @ v_hat) * v_hat + across * n_hat - offset, self.sigma_loc_px

    def anchor(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(anchor_uv, anchor_grip, sigma_px) for an ExternalProjector from the validated calibration locations: each
        located grip carried to the last one's position by J and averaged (sigma = sigma_consistency / sqrt n (+)
        L1_MODEL_SIGMA_PX: the model error is common to all locations and does not average out)."""
        rows = [row for row in self.located if row.get("grip") is not None]
        if not rows:
            return None
        grip = np.asarray(rows[-1]["grip"], dtype=float)
        carried = [np.asarray(row["uv"], dtype=float) + self.model.J @ (grip - np.asarray(row["grip"], dtype=float))
                   for row in rows]
        sigma = math.hypot(self.sigma_consistency_px / math.sqrt(len(rows)), L1_MODEL_SIGMA_PX)
        return np.mean(carried, axis=0), grip, sigma

    def to_dict(self) -> dict:
        return jsonable({"sigma_loc_px": self.sigma_loc_px, "sigma_consistency_px": self.sigma_consistency_px,
                         "model_sigma_px": L1_MODEL_SIGMA_PX, "residual_px": self.residual_px,
                         "located": self.located, "degraded": self.degraded, "flags": list(self.flags),
                         "dropped": self.dropped, "thresholds": [t.to_dict() for t in self.thresholds]})


# ================================================================================================ ExternalProjector
class ExternalProjector:
    """Affine external projection u(X) = anchor_uv + J (X - anchor_grip), anchored on a located grip pixel; sigma
    = anchor sigma (+) PROJECTOR_AFFINE_FRACTION x the predicted shift (the affine model's perspective error grows
    with the distance from the anchor; re-anchor near a far region [C-51])."""

    def __init__(self, model: ExternalModel, anchor_uv, anchor_grip, sigma_px: float):
        self.model = model
        self.anchor_uv = np.asarray(anchor_uv, dtype=float).reshape(2)
        self.anchor_grip = np.asarray(anchor_grip, dtype=float).reshape(3)
        self.sigma_px = float(sigma_px)

    def project(self, xyz) -> tuple[np.ndarray, float]:
        shift = self.model.J @ (np.asarray(xyz, dtype=float).reshape(3) - self.anchor_grip)
        sigma = math.hypot(self.sigma_px, PROJECTOR_AFFINE_FRACTION * float(np.linalg.norm(shift)))
        return self.anchor_uv + shift, sigma

    def project_polygon(self, poly_xyz) -> tuple[np.ndarray, float]:
        points = np.asarray(poly_xyz, dtype=float).reshape(-1, 3)
        projected = [self.project(p) for p in points]
        return np.array([uv for uv, _ in projected]), max(sigma for _, sigma in projected)

    def mask(self, poly_xyz, shape: tuple | None = None, erode_px: float | None = None) -> np.ndarray:
        """Filled projected polygon (bool, upright external image); shape None = EXTERNAL_IMAGE_PX square (pass the
        image's own shape when it is at hand); erode_px None = none."""
        h, w = (EXTERNAL_IMAGE_PX, EXTERNAL_IMAGE_PX) if shape is None else (int(shape[0]), int(shape[1]))
        uv, _ = self.project_polygon(poly_xyz)
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(out, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], 1)
        if erode_px:
            size = 2 * int(math.ceil(float(erode_px))) + 1
            out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        return out > 0

    def reanchored(self, anchor_uv, anchor_grip, sigma_px) -> "ExternalProjector":
        return ExternalProjector(self.model, anchor_uv, anchor_grip, sigma_px)

    def distance_m(self, xyz) -> float:
        return float(np.linalg.norm(np.asarray(xyz, dtype=float).reshape(3) - self.anchor_grip))

    def to_dict(self) -> dict:
        return jsonable({"anchor_uv": self.anchor_uv, "anchor_grip": self.anchor_grip, "sigma_px": self.sigma_px})


def _region_colour_model(lab, core) -> np.ndarray:
    """Boolean Lab histogram (REGION_LAB_BIN bins per axis) of the colours accepted for a region: a bin is accepted
    when at least REGION_MODE_FRACTION of the core's pixels lie within REGION_LAB_FLOOR of it (the core's histogram
    summed over a ball of that radius in bin units) and that mass is at least REGION_MODE_RELATIVE of the dominant
    colour's. Multi-modal: every substantial tone of a ringed or two-tone region, not a sliver of a neighbour."""
    bins = int(math.ceil((RGB_MAX + 1.0) / REGION_LAB_BIN))
    index = np.clip((lab[core] / REGION_LAB_BIN).astype(int), 0, bins - 1)
    flat = (index[:, 0] * bins + index[:, 1]) * bins + index[:, 2]
    counts = np.bincount(flat, minlength=bins ** 3).reshape(bins, bins, bins).astype(float)
    radius = int(math.ceil(REGION_LAB_FLOOR / REGION_LAB_BIN))
    padded = np.pad(counts, radius)
    near = np.zeros_like(counts)
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            for c in range(-radius, radius + 1):
                if a * a + b * b + c * c <= radius * radius:
                    near += padded[radius + a:radius + a + bins, radius + b:radius + b + bins,
                                   radius + c:radius + c + bins]
    return (near >= REGION_MODE_FRACTION * float(core.sum())) & (near >= REGION_MODE_RELATIVE * float(near.max()))


def refine_region_mask(external_rgb, affine_mask, seed_uv, *, sigma_px: float) -> tuple[np.ndarray, dict]:
    """[C-51] Lab region growing around a projected region. Steps:

    1. margin = L1_SIGMA_K x sigma_px, capped at REGION_MAX_MARGIN_PX (a projector sigma of tens of px, dominated by
       the affine distance term, must not open the whole image); the region may grow into the affine mask dilated
       by the margin.
    2. Core (the colour model's sample): the affine mask eroded by the margin; when fewer than HM_MIN_PIXELS
       remain, the seed-centred disks of REGION_CORE_RADIUS_PX inside the affine mask, else the disks alone
       (``seed_uv``: one (2,) pixel or several, e.g. the projected centroid and the plan cell centre). Never the
       whole affine mask, which mixes in the background around an offset region.
    3. Colour model (``_region_colour_model``): the Lab colours (8-bit scaling, [C-20]) within REGION_LAB_FLOOR of at
       least REGION_MODE_FRACTION of the core (and REGION_MODE_RELATIVE of its dominant colour); candidates are the
       allowed pixels of an accepted colour. Grooves are closed (REGION_CLOSE_PX), the components holding a seed
       (else the one overlapping the core most) kept and holes filled.
    4. Refinement corrects the position, not the size: a region larger than REGION_MAX_AREA_RATIO x the affine
       area is a flood, one smaller than REGION_MIN_AREA_RATIO x it a partial capture; the affine mask is then
       returned with ``info["reason"]`` "area_cap" / "area_floor" (also "no_core" / "no_region" when nothing could
       be grown).

    Returns (mask, {"iou_vs_affine", "pixels", "core_pixels", "core", "margin_px", "reason", ...})."""
    affine = np.asarray(affine_mask, dtype=bool)
    lab = to_lab255(external_rgb)
    h, w = affine.shape
    wanted = int(math.ceil(L1_SIGMA_K * float(sigma_px)))
    margin = int(min(max(1, wanted), REGION_MAX_MARGIN_PX))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    allowed = cv2.dilate(affine.astype(np.uint8), disk) > 0
    seeds = np.asarray(seed_uv, dtype=float).reshape(-1, 2)
    yy, xx = np.mgrid[0:h, 0:w]
    seed_disks = np.zeros((h, w), dtype=bool)
    for u, v in seeds:
        seed_disks |= (xx - float(u)) ** 2 + (yy - float(v)) ** 2 <= float(REGION_CORE_RADIUS_PX) ** 2
    core, core_kind = cv2.erode(affine.astype(np.uint8), disk) > 0, "eroded_affine"
    if int(core.sum()) < HM_MIN_PIXELS:
        core, core_kind = seed_disks & affine, "seed_disk_in_affine"
    if int(core.sum()) < HM_MIN_PIXELS:
        core, core_kind = seed_disks, "seed_disk"
    info = {"iou_vs_affine": None, "pixels": 0, "core_pixels": int(core.sum()), "core": core_kind,
            "margin_px": margin, "margin_wanted_px": wanted, "affine_pixels": int(affine.sum()), "seeds": len(seeds),
            "reason": None}
    if not core.any():
        info.update(reason="no_core", pixels=int(affine.sum()), iou_vs_affine=1.0 if affine.any() else None)
        return affine.copy(), info
    accepted = _region_colour_model(lab, core)
    bins = accepted.shape[0]
    index = np.clip((lab / REGION_LAB_BIN).astype(int), 0, bins - 1)
    candidate = (allowed & accepted[index[..., 0], index[..., 1], index[..., 2]]).astype(np.uint8)
    close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REGION_CLOSE_PX, REGION_CLOSE_PX))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close) & allowed.astype(np.uint8)
    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    chosen = set()
    for u, v in seeds:
        iu, iv = int(round(u)), int(round(v))
        if 0 <= iu < w and 0 <= iv < h and labels[iv, iu] > 0:
            chosen.add(int(labels[iv, iu]))
    if not chosen:
        overlap = [(int(np.count_nonzero((labels == k) & core)), k) for k in range(1, count)]
        overlap = [item for item in overlap if item[0] > 0]
        if overlap:
            chosen.add(max(overlap)[1])
    info["colours"] = int(accepted.sum())
    if not chosen:
        info.update(reason="no_region", pixels=int(affine.sum()), iou_vs_affine=1.0 if affine.any() else None)
        return affine.copy(), info
    region = np.isin(labels, sorted(chosen))
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = region
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    region = region | (padded[1:-1, 1:-1] == 0)
    grown = int(region.sum())
    info["grown_pixels"] = grown
    if affine.any() and not REGION_MIN_AREA_RATIO * int(affine.sum()) <= grown <= REGION_MAX_AREA_RATIO * int(
            affine.sum()):
        info.update(reason="area_cap" if grown > int(affine.sum()) else "area_floor", pixels=int(affine.sum()),
                    iou_vs_affine=1.0)
        return affine.copy(), info
    union = np.count_nonzero(region | affine)
    info.update(iou_vs_affine=(np.count_nonzero(region & affine) / union) if union else None, pixels=grown)
    return region, info


# ================================================================================================ SurveyRecorder
class SurveyRecorder:
    """ProbeRecorder for one survey (the Calibrate+ start survey, a hover ParallaxProbe, a lift). Features are
    created on the construction view: the seeds [(key, uv)] (the exact pixel when it passes D15, else the most
    textured patch within RELIEF_SEARCH_PX), the relief points (always the most textured patch within
    RELIEF_SEARCH_PX) and, with ``auto_grid``, up to FT_MAX_FEATURES grid patches outside ``exclude_mask``.

    ``on_transition(stage, before, after)``: gripper stages (close / open / yaw / open_hold / release) are never
    tracked; the open_hold frames ("open", "static[k]", and "ext_static[k]" with ``external``) and the last close_2
    frame ("closed") are kept. Every other stage is tracked on every step with the shift predicted from the measured
    hand motion and the running per-feature kappa; the reference frame (``pose0``) is the frame before the first
    motion stage. With ``external`` the hand image motion of every motion-stage run (one probe move) is measured
    between its first and last external frames.

    ``probe_tracker`` (decision §2.3 G-1, the measurement remedy R): every grey the recorder tracks is band-passed
    (``bandpass_grey``); templates are FT_PROBE_RADIUS_PX (11 px) with the D15 test D15_PROBE_MIN_STD on the filtered
    patch and the NCC floor V1_PROBE_MIN_PEAK; seeds are placed EXACTLY (a seed failing D15 is listed in
    ``unplaced``, never moved to a textured neighbour, which on a thin dark part is its outline or the surface
    behind it); ``dense_uv`` adds ``dense_lattice`` around that pixel (keys "dense:<k>"); relief points take the most
    textured 11 px patch within RELIEF_SEARCH_PX and the grid the filtered D15 test. Without it the v1-like tracker
    runs (21 px templates on the raw grey, seeds relocated when textureless). ``adaptive_search`` (G-3, OPT-IN; None
    = off in both modes): a feature with a running kappa is searched +-FT_TRACKED_SEARCH_PX around its prediction,
    the others +-FT_SEARCH_PX. Off, every feature is searched +-FT_SEARCH_PX and the tracks equal the unoptimised path
    exactly; with the shared-thread matching (``match_scores``) that path meets decision §2.3 G-3's acceptance as
    written (<= 40 ms per tracked frame, also at the operating 451-472 features, and |duv| 0 px against the prototype;
    RS-8 replays the 40 recorded task-2 hovers, RS-1 measures it live). The adaptive search changes the tracks (1-6
    more > 5 percent depth errors on the fake kappa planes in the v1-like mode; on the 40 hovers 2428 differing D1
    decisions and slightly more off-part false samples, A 165 vs 156, C 36 vs 33), so it stays opt-in until the lead
    restates G-3 for it (review finding R2). The other speed-ups are exact and always on. ``grid_at_reference``
    (Calibrate+, BR's calibrate_plus): the auto grid is placed on the reference frame, outside the fingertip band of
    the recorder's own open-hold / close_2 frames (``_place_grid``), instead of on the construction view."""

    def __init__(self, view: StepView, robot: RobotModel, *, camera: "CameraModel | None",
                 seeds: Sequence[tuple[str, tuple]] = (), relief: Sequence[tuple[str, tuple]] = (),
                 auto_grid: bool = True, exclude_mask: np.ndarray | None = None, external: bool = False,
                 start_step: int | None = None, probe_tracker: bool = False, dense_uv=None,
                 adaptive_search: bool | None = None, grid_at_reference: bool = False):
        self.robot = robot
        self.camera = camera
        self.external = bool(external)
        self.start_step = 0 if start_step is None else int(start_step)
        self.probe_tracker = bool(probe_tracker)
        self.adaptive_search = False if adaptive_search is None else bool(adaptive_search)
        if dense_uv is not None and not self.probe_tracker:
            raise ValueError("dense_uv needs probe_tracker=True (the dense lattice is the probe tracker's)")
        self.dense_uv = None if dense_uv is None else tuple(float(v) for v in dense_uv)
        self.view0 = view
        self._poses: dict[tuple, HandPose] = {}
        self.unplaced = []
        self._grid_pending = bool(auto_grid and grid_at_reference)
        self._grid_exclude = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
        self.grid_info: dict = {"at_reference": bool(grid_at_reference), "placed": not self._grid_pending}
        auto_grid = bool(auto_grid and not grid_at_reference)
        gray = self._grey(view)
        if self.probe_tracker:
            self.tracker = self._probe_tracker(gray, seeds, relief, auto_grid, exclude_mask)
        else:
            uvs, keys = [], []
            for key, uv in seeds:
                uv = np.asarray(uv, dtype=float)
                placed = uv
                if not _textured(gray, uv):
                    placed, _ = best_textured_patch(gray, uv)
                if placed is None:
                    self.unplaced.append({"key": key, "uv": jsonable(uv), "reason": "textureless"})
                    continue
                uvs.append(placed)
                keys.append(key)
            for key, uv in relief:
                placed, _ = best_textured_patch(gray, uv)
                if placed is None:
                    self.unplaced.append({"key": key, "uv": jsonable(np.asarray(uv, dtype=float)),
                                          "reason": "textureless"})
                    continue
                uvs.append(placed)
                keys.append(key)
            if auto_grid:
                for uv in FeatureTracker.auto_grid(gray, mask=exclude_mask):
                    uvs.append(np.asarray(uv, dtype=float))
                    keys.append(None)
            self.tracker = FeatureTracker(gray, uvs, keys=keys)
        self.frames: dict = {}
        if self.external:
            self.frames["ext_start"] = view.external_rgb_upright
        self.stages: dict[str, int] = {}
        self.transitions = 0
        self._last = view
        self._ref: StepView | None = None
        self._ref_index: int | None = None
        self._samples: dict[int, list[tuple]] = {}
        self._runs: list[dict] = []
        self._static = 0
        self._kappa_sums: dict[int, np.ndarray] = {}
        self._external_cache = None

    def _probe_tracker(self, gray, seeds, relief, auto_grid, exclude_mask) -> FeatureTracker:
        """The probe tracker's features on the band-passed construction grey: exact seeds, the dense lattice, the
        relief points and the grid (decision §2.3 G-1)."""
        tracker = FeatureTracker(gray, [], radius_px=FT_PROBE_RADIUS_PX, min_std=D15_PROBE_MIN_STD,
                                 min_peak=V1_PROBE_MIN_PEAK)
        for key, uv in seeds:
            if tracker.add(gray, uv, key=key) is None:
                self.unplaced.append({"key": key, "uv": jsonable(np.asarray(uv, dtype=float)),
                                      "reason": tracker.rejected[-1]["reason"]})
        if self.dense_uv is not None:
            for key, uv in dense_lattice(self.dense_uv, exclude_mask=exclude_mask, size_px=gray.shape[1]):
                tracker.add(gray, uv, key=key)
        for key, uv in relief:
            placed, _ = best_textured_patch(gray, uv, radius_px=FT_PROBE_RADIUS_PX, min_std=D15_PROBE_MIN_STD)
            if placed is None or tracker.add(gray, placed, key=key) is None:
                self.unplaced.append({"key": key, "uv": jsonable(np.asarray(uv, dtype=float)),
                                      "reason": "textureless"})
        if auto_grid:
            # the grid's D15 test on the probe tracker's own 11 px patch (G-1 (a)), so every selected point passes
            # add() and no grid slot of the FT_MAX_FEATURES budget is lost; the grid positions are those of the 21 px
            # grid (grid_positions centres the same 36 px lattice for both radii on a 512 px image)
            for uv in FeatureTracker.auto_grid(gray, mask=exclude_mask, radius_px=FT_PROBE_RADIUS_PX,
                                               min_std=D15_PROBE_MIN_STD):
                tracker.add(gray, np.asarray(uv, dtype=float), key=None)
        return tracker

    def _place_grid(self, view: StepView) -> None:
        """``grid_at_reference``: the auto grid placed on the REFERENCE frame (the frame before the first motion
        stage), outside ``exclude_mask`` and the fingertip band measured from this recorder's own kept open-hold and
        close_2 frames (Calibrate+: the band is unknown when the recorder is built, and finger patches would otherwise
        take grid slots of the FT_MAX_FEATURES budget); the grid features are not tracked across the gripper stages
        before it."""
        gray = self._grey(view)
        mask = self._grid_exclude
        band = None
        if "open" in self.frames and "closed" in self.frames:
            band = fingertip_band_mask(self.frames["open"], self.frames["closed"])
            mask = band if mask is None else (mask | band)
        min_std = D15_PROBE_MIN_STD if self.probe_tracker else D15_MIN_STD_GREY
        added = 0
        for uv in FeatureTracker.auto_grid(gray, mask=mask, radius_px=self.tracker.radius, min_std=min_std):
            added += self.tracker.add(gray, np.asarray(uv, dtype=float), key=None) is not None
        self._grid_pending = False
        self.grid_info = {"at_reference": True, "placed": True, "features": int(added),
                          "band_px": None if band is None else int(np.count_nonzero(band)),
                          "fingertip_band_top_row": None if band is None else fingertip_band_top_row(band)}

    def _grey(self, view: StepView) -> np.ndarray:
        """The grey the tracker sees: the band-passed wrist grey in probe-tracker mode, else the wrist grey."""
        gray = view.wrist_gray_upright
        return bandpass_grey(gray) if self.probe_tracker else gray

    def _pose(self, state) -> HandPose:
        """HandPose of a state8, cached per state within this recorder (G-3; HandPose is read-only here)."""
        key = tuple(float(v) for v in state)
        pose = self._poses.get(key)
        if pose is None:
            pose = self._poses[key] = HandPose.from_state8(state)
        return pose

    # ------------------------------------------------------------------------------------------- recording
    def on_transition(self, stage: str, before: StepView, after: StepView) -> None:
        self.transitions += 1
        self.stages[stage] = self.stages.get(stage, 0) + 1
        kind = _stage_kind(stage)
        if stage == STATIC_STAGE:
            self.frames["open"] = after.wrist_rgb_upright
            self.frames[f"static[{self._static}]"] = after.wrist_rgb_upright
            if self.external:
                self.frames[f"ext_static[{self._static}]"] = after.external_rgb_upright
            self._static += 1
        elif stage == CLOSED_STAGE:
            self.frames["closed"] = after.wrist_rgb_upright
        run = self._runs[-1] if self._runs else None
        if run is None or run["stage"] != stage or run["closed"]:
            if run is not None:
                run["closed"] = True
            run = {"stage": stage, "kind": kind, "before": before, "after": after, "steps": 0, "closed": False,
                   "external": None}
            self._runs.append(run)
        run["after"] = after
        run["steps"] += 1
        if kind == "gripper":
            return
        if self._ref is None and kind in ("motion", "vertical"):
            if not _same_frame(self._last, before):
                self._track(self._last, before, sample=False)
            if self._grid_pending:
                self._place_grid(before)
            self._ref = before
            self._ref_index = self.transitions - 1
            for fid in self.tracker.active:
                self._samples[fid] = [(stage, self.tracker.features[fid]["uv"].copy(), tuple(before.state), "ref")]
        elif not _same_frame(self._last, before):
            self._track(self._last, before, sample=self._ref is not None, stage=stage)
        self._track(before, after, sample=self._ref is not None, stage=stage)

    def _f_ref(self) -> float:
        return self.camera.f_used if self.camera is not None else r3_focal_px()

    def _predicted_shifts(self, frm: StepView, to: StepView, running: dict | None = None) -> tuple[dict, dict]:
        """({fid: predicted image shift}, {fid: predicted image scale}) from the measured pose change. The feature's
        point is placed at depth f / kappa along its reference ray (running per-feature kappa, else the median of the
        running kappas); without any kappa yet the ray is projected at infinity (rotation only, no scale). The
        scale is the depth at template capture over the depth now (looming); only changes beyond FT_SCALE_TOL are
        passed on. No depth prior is used: kappa comes from this survey's own measured motion. ``running``: the
        active features' running kappas when the caller has them."""
        f = self._f_ref()
        camera = WristCamera(f)
        pose_a, pose_b = self._pose(frm.state), self._pose(to.state)
        capture = self._pose(self.view0.state)
        ref = self._pose(self._ref.state) if self._ref is not None else None
        if running is None:
            running = {fid: self._running_kappa(fid) for fid in self.tracker.active}
        known = [k for k in running.values() if k is not None]
        fallback = float(np.median(known)) if known else None
        shifts, scales = {}, {}
        fids = list(self.tracker.active)
        if not fids:
            return shifts, scales
        # vectorised over the features (G-3; the same per-feature formulas): the ray of each feature from its
        # reference pixel (or its current pixel before the reference exists), its point at depth f / kappa
        features = self.tracker.features
        uvs = np.array([features[fid]["uv"] for fid in fids], dtype=float)
        kappas = np.array([running[fid] if running[fid] is not None else (np.nan if fallback is None else fallback)
                           for fid in fids], dtype=float)
        with_kappa = np.isfinite(kappas) & (kappas > 0.0)
        from_ref = np.array([ref is not None and fid in self._samples for fid in fids], dtype=bool)
        ray_uv = np.array([self._samples[fid][0][1] if inside else features[fid]["uv"]
                           for fid, inside in zip(fids, from_ref)], dtype=float)
        frame_R = np.where(from_ref[:, None, None], (ref or pose_a).R[None], pose_a.R[None])
        origin = np.where(from_ref[:, None], (ref or pose_a).camera[None], pose_a.camera[None])
        c = principal_point(WRIST_IMAGE_PX)
        ab = np.c_[ray_uv[:, 0] - c[0], c[1] - ray_uv[:, 1]] / f
        rays = frame_R[:, :, 2] + ab[:, :1] * frame_R[:, :, 1] + ab[:, 1:] * frame_R[:, :, 0]
        depth = np.where(with_kappa, f / np.where(with_kappa, kappas, 1.0), 0.0)
        points = origin + depth[:, None] * rays
        rel = points - pose_b.camera
        a_b, b_b, z_b = rel @ pose_b.R[:, 1], rel @ pose_b.R[:, 0], rel @ pose_b.R[:, 2]
        depth_capture = (points - capture.camera) @ capture.R[:, 2]
        # no kappa yet: the ray from the current pixel at infinity (rotation only)
        cur = np.c_[uvs[:, 0] - c[0], c[1] - uvs[:, 1]] / f
        far = pose_a.R[:, 2][None] + cur[:, :1] * pose_a.R[:, 1][None] + cur[:, 1:] * pose_a.R[:, 0][None]
        a_f, b_f, z_f = far @ pose_b.R[:, 1], far @ pose_b.R[:, 0], far @ pose_b.R[:, 2]
        cb = camera.principal_point
        for k, fid in enumerate(fids):
            if with_kappa[k]:
                if not z_b[k] > 0.0:
                    continue
                predicted = np.array([cb[0] + f * a_b[k] / z_b[k], cb[1] - f * b_b[k] / z_b[k]])
                if depth_capture[k] > 0.0:
                    scale = float(depth_capture[k] / z_b[k])
                    if abs(scale - 1.0) >= FT_SCALE_TOL:
                        scales[fid] = scale
            else:
                if not z_f[k] > 1e-9:
                    continue
                predicted = uncentred((f * a_f[k] / z_f[k], f * b_f[k] / z_f[k]))
            shifts[fid] = predicted - uvs[k]
        return shifts, scales

    def _running_kappa(self, fid) -> float | None:
        sums = self._kappa_sums.get(fid)
        if sums is None or sums[1] < RUNNING_KAPPA_MIN_BASELINE_M ** 2:
            return None
        kappa = float(sums[0] / sums[1])
        return kappa if kappa > 0.0 else None

    def _track(self, frm: StepView, to: StepView, *, sample: bool, stage: str | None = None) -> None:
        running = {fid: self._running_kappa(fid) for fid in self.tracker.active}
        shifts, scales = self._predicted_shifts(frm, to, running)
        search = None
        if self.adaptive_search:
            search = {fid: FT_TRACKED_SEARCH_PX if kappa is not None else FT_SEARCH_PX
                      for fid, kappa in running.items()}
        found = self.tracker.track(self._grey(to), shifts, scales, search_px_by_fid=search)
        self._last = to
        if not sample or self._ref is None:
            return
        kind = _stage_kind(stage) if stage else "static"
        ref = self._pose(self._ref.state)
        pose = self._pose(to.state)
        d = pose.camera - ref.camera
        g = np.array([-float(d @ ref.R[:, 1]), float(d @ ref.R[:, 0])])
        for fid, (uv, _) in found.items():
            if uv is None or fid not in self._samples:
                continue
            self._samples[fid].append((stage, uv.copy(), tuple(to.state), kind))
            if kind != "vertical":
                ds = uv - self._samples[fid][0][1]
                sums = self._kappa_sums.setdefault(fid, np.zeros(2))
                sums += np.array([float(ds @ g), float(g @ g)])

    # ------------------------------------------------------------------------------------------- result
    def _external_samples(self) -> list[dict]:
        """One sample per motion-stage run (probe move), from its first and last external frames: the measured grip
        displacement; ``flow_shift_px`` = the v1 hand flow (fastest decile: arm links that move faster in the image
        than the grip); ``image_shift_px`` = the HAND's own image motion (``hand_band_shift``, source "hand_band";
        the flow when it fails); ``mask_extreme_uv`` = the fingertip in the move's first frame
        (``outline_tip_before`` of the motion-consistent union outline). v_hat comes from a provisional J fitted on
        the flows, then twice from the hand shifts."""
        if not self.external:
            return []
        runs = [run for run in self._runs if run["kind"] in ("motion", "vertical")]
        key = tuple((run["stage"], run["steps"]) for run in runs)
        if self._external_cache is not None and self._external_cache[0] == key:
            return [dict(sample) for sample in self._external_cache[1]]
        samples = self._measure_external(runs)
        self._external_cache = (key, samples)
        return [dict(sample) for sample in samples]

    def _measure_external(self, runs) -> list[dict]:
        samples = []
        for run in runs:
            first, last = run["before"].external_rgb_upright, run["after"].external_rgb_upright
            if run["external"] is None or run["external"]["steps"] != run["steps"]:
                run["external"] = {"steps": run["steps"], "motion": hand_motion(first, last)}
            motion = run["external"]["motion"]
            flow = None if motion is None else motion["image_shift_px"]
            samples.append({"stage": run["stage"],
                            "world_displacement_m": (run["after"].position - run["before"].position).tolist(),
                            "image_shift_px": flow, "flow_shift_px": flow, "mask_extreme_uv": None,
                            "pixels": None if motion is None else motion["pixels"],
                            "start_state8": list(map(float, run["before"].state)),
                            "end_state8": list(map(float, run["after"].state)), "shift_source": "flow"})
        model = ExternalModel.fit(samples)
        for _ in range(2):
            if model is None:
                return samples
            band_px = float(np.linalg.norm(model.J @ np.array([0.0, 0.0, HL_HAND_BAND_M])))
            for run, sample in zip(runs, samples):
                shift, _ = hand_band_shift(run["before"].external_rgb_upright, run["after"].external_rgb_upright,
                                           model.v_hat, band_px=band_px)
                if shift is not None:
                    sample["image_shift_px"], sample["shift_source"] = shift.tolist(), "hand_band"
                else:
                    sample["image_shift_px"], sample["shift_source"] = sample["flow_shift_px"], "flow"
            refit = ExternalModel.fit(samples)
            if refit is None:
                break
            model = refit
        band_px = float(np.linalg.norm(model.J @ np.array([0.0, 0.0, HL_HAND_BAND_M])))
        for run, sample in zip(runs, samples):
            m = model.J @ (run["after"].position - run["before"].position)
            tip = hand_tip_before(run["before"].external_rgb_upright, run["after"].external_rgb_upright, m,
                                  model.v_hat, band_px=band_px)
            sample["mask_extreme_uv"] = None if tip is None else tip.tolist()
        return samples

    def _feature_series(self, fid, f):
        """(kappa samples [(ds, dp, dz, uv)], legs {stage index: [...]}, loom frames [(uv, d)]) for one feature,
        rotation removed to the reference orientation."""
        series = self._samples.get(fid) or []
        if not series:
            return [], [], []
        ref = self._pose(self._ref.state)
        uv0 = np.asarray(series[0][1], dtype=float)
        kappa_samples, loom_frames, legs = [], [(uv0, (0.0, 0.0, 0.0))], []
        current = None
        last_sample = (np.zeros(2), np.zeros(2), 0.0, uv0)
        rest = series[1:]
        if not rest:
            return kappa_samples, legs, loom_frames
        # derotation and camera displacement of every sample at once (G-3; derotate_pixel's formula)
        poses = [self._pose(state) for _, _, state, _ in rest]
        Rs = np.stack([pose.R for pose in poses])
        cams = np.stack([pose.camera for pose in poses])
        uvs = np.array([uv for _, uv, _, _ in rest], dtype=float)
        c = principal_point(WRIST_IMAGE_PX)
        ab = np.c_[uvs[:, 0] - c[0], c[1] - uvs[:, 1]] / float(f)
        rays = Rs[:, :, 2] + ab[:, :1] * Rs[:, :, 1] + ab[:, 1:] * Rs[:, :, 0]
        ray_z = rays @ ref.R[:, 2]
        ray_a, ray_b = rays @ ref.R[:, 1], rays @ ref.R[:, 0]
        moved = cams - ref.camera
        d_cams = np.c_[moved @ ref.R[:, 1], moved @ ref.R[:, 0], moved @ ref.R[:, 2]]
        for k, (stage, _, _, kind) in enumerate(rest):
            if not ray_z[k] > 1e-9:
                continue
            uv_d = uncentred((float(f) * ray_a[k] / ray_z[k], float(f) * ray_b[k] / ray_z[k]))
            d_cam = d_cams[k]
            if kind == "vertical":
                loom_frames.append((uv_d, tuple(d_cam)))
            else:
                sample = (uv_d - uv0, d_cam[:2], d_cam[2], uv_d)
                kappa_samples.append(sample)
                if kind == "motion":
                    if current is None or current["stage"] != stage:
                        # a leg starts at the previous frame (the end of the previous leg, or the reference)
                        current = {"stage": stage, "samples": [last_sample] if last_sample is not None else []}
                        legs.append(current)
                    current["samples"].append(sample)
                last_sample = sample
        return kappa_samples, legs, loom_frames

    def result(self, camera: "CameraModel | None") -> SurveyResult:
        """Per-feature kappa (D1) and f-free looming depth (D3) always; DepthSamples (D4, source "kappa") only with a
        camera model, with Z = camera.f_used / kappa. Callable repeatedly (Calibrate+: result(None) ->
        CameraModel.decide(f_hat) -> result(camera))."""
        f = camera.f_used if camera is not None else (self.camera.f_used if self.camera is not None
                                                      else r3_focal_px())
        sigma_f = camera.sigma_f if camera is not None else None
        features, samples = [], []
        z_by_fid, kappa_by_fid, f_by_fid, sigma_f_by_fid = {}, {}, {}, {}
        ref_pose = self._pose(self._ref.state) if self._ref is not None else None
        wrist = WristCamera(f)
        for fid, feature in self.tracker.features.items():
            if self._ref is None or fid not in self._samples:
                features.append(FeatureRecord(fid, feature["key"], tuple(map(float, feature["uv0"])), None, 0, None,
                                              None, None, None, None, False, "lost" if feature["lost"]
                                              else "shift_too_small"))
                continue
            uv0 = np.asarray(self._samples[fid][0][1], dtype=float)
            kappa_samples, legs, loom_frames = self._feature_series(fid, f)
            aperture = template_aperture(feature["template"])
            kappa, sigma_kappa, residual, reject = fit_feature_kappa(kappa_samples, f_px=f, aperture=aperture)
            if reject is None:
                leg_kappas, leg_dirs = [], []
                for leg in legs:
                    leg_k = _leg_kappa(leg["samples"], f, aperture)
                    if leg_k is not None and leg_k > 0.0:
                        leg_kappas.append(leg_k)
                        leg_dirs.append(np.asarray(leg["samples"][-1][1]) - np.asarray(leg["samples"][0][1]))
                if len(leg_kappas) >= 2 and max(leg_kappas) / min(leg_kappas) - 1.0 > D1_LEG_AGREEMENT:
                    reject = "legs"
                else:
                    sigma_kappa = max(sigma_kappa, _leg_sigma(leg_kappas, leg_dirs))
            if reject is not None and feature["lost"] and len(kappa_samples) < D1_MIN_SAMPLES:
                reject = "lost"
            elif reject == TOO_FEW_SAMPLES:
                reject = "shift_too_small"             # kernel FEATURE_REJECTS: too little probe evidence
            valid = reject is None
            z = sigma_z = xyz = None
            if valid:
                kappa_by_fid[fid] = kappa
                if len(loom_frames) > D1_MIN_SAMPLES:
                    z0, sz0, _ = loom_fit(loom_frames, kappa, sigma_kappa=sigma_kappa)
                    if z0 is not None:
                        z_by_fid[fid] = (z0, sz0)
                        f_i = kappa * z0
                        f_by_fid[fid] = f_i
                        sigma_f_by_fid[fid] = f_i * math.hypot(sigma_kappa / kappa, sz0 / z0)
                if camera is not None:
                    z, sigma_z = depth_from_kappa(kappa, sigma_kappa, f, sigma_f)
                    point = wrist.backproject(ref_pose, uv0, z)
                    xyz = tuple(float(v) for v in point)
                    direction = pixel_direction(ref_pose, uv0, f)
                    sigma_world_z = abs(float(direction[2])) * sigma_z
                    lateral = float(np.linalg.norm(direction[:2])) * sigma_z
                    sigma_xy = math.hypot(lateral, residual * z / f)
                    if not np.allclose(aperture, np.eye(2)):
                        # an edge-like template moves with its dominant (usually the occluding, nearer) edge, so
                        # its depth belongs somewhere within the template: the xy is uncertain by its half size
                        sigma_xy = math.hypot(sigma_xy, self.tracker.radius * z / f)
                    samples.append(DepthSample(xyz, sigma_world_z, sigma_xy, KAPPA_SOURCE, tuple(map(float, uv0)),
                                               fid, self.start_step + (self._ref_index or 0), kappa))
                elif fid in z_by_fid:
                    z, sigma_z = z_by_fid[fid]
            uv_end = feature["uv"] if not feature["lost"] else (feature["history"][-1][1] if feature["history"]
                                                                 else None)
            features.append(FeatureRecord(fid, feature["key"], tuple(map(float, uv0)),
                                          None if uv_end is None else tuple(map(float, uv_end)),
                                          len(kappa_samples) + len(loom_frames) - 1,
                                          kappa, sigma_kappa, z, sigma_z, xyz, valid, reject))
        f_hat = None
        if len(f_by_fid) >= D3_MIN_FEATURES:
            values = np.array(list(f_by_fid.values()))
            sigmas = np.array([max(sigma_f_by_fid[k], 1e-9) for k in f_by_fid])
            centre = _weighted_median(values, 1.0 / sigmas ** 2)
            # standard error of a median: sqrt(pi / 2) x sigma / sqrt(n), sigma = the larger of the measured
            # per-feature spread and the typical per-feature fit sigma
            typical = max(_robust_sigma(values), float(np.median(sigmas)))
            f_hat = Meas(centre, math.sqrt(math.pi / 2.0) * typical / math.sqrt(len(values)), LOOM_SOURCE)
        loom = {"z_by_fid": {fid: z for fid, (z, _) in z_by_fid.items()},
                "sigma_z_by_fid": {fid: s for fid, (_, s) in z_by_fid.items()},
                "kappa_by_fid": kappa_by_fid, "f_hat_by_fid": f_by_fid, "f_hat": f_hat,
                "f_hat_spread": _robust_sigma(list(f_by_fid.values())) if f_by_fid else None}
        pose0 = tuple(map(float, (self._ref or self.view0).state))
        pose_end = tuple(map(float, self._last.state))
        return SurveyResult(pose0, pose_end, dict(self.stages), features, samples, loom, self._external_samples(),
                            self.frames)

    def thresholds(self) -> list[Threshold]:
        if self.probe_tracker:
            texture = th_value("D15_min_std_grey", D15_PROBE_MIN_STD, "DD",
                               "D15 band-passed patch std (probe tracker, D15_PROBE_MIN_STD, decision §2.3 G-1)",
                               radius_px=FT_PROBE_RADIUS_PX)
            peak = th_value("V1_min_peak", V1_PROBE_MIN_PEAK, "DD",
                            "V1 NCC miss below this peak (probe tracker, V1_PROBE_MIN_PEAK)",
                            lost_frames=V1_LOST_FRAMES)
        else:
            texture = th_value("D15_min_std_grey", D15_MIN_STD_GREY, "DD", "D15 patch std (v1 NCCTrack min_std)")
            peak = th_value("V1_min_peak", V1_MIN_PEAK, "DD", "V1 NCC miss below this peak",
                            lost_frames=V1_LOST_FRAMES)
        return [th_value("D1_min_shift_px", D1_MIN_SHIFT_PX, "DD", "D1 largest feature shift (v1 Align)"),
                th_value("D1_max_residual_px", D1_RESIDUAL_FLOOR_PX, "DD",
                         "D1 residual <= max(D1_RESIDUAL_FLOOR_PX, D1_RESIDUAL_FRACTION * rms shift)",
                         fraction=D1_RESIDUAL_FRACTION, leg_agreement=D1_LEG_AGREEMENT),
                texture, peak]

    def summary(self) -> dict:
        keys = [feature["key"] for feature in self.tracker.features.values()]
        dense = sum(isinstance(key, str) and key.startswith(DENSE_FEATURE_KEY_PREFIX) for key in keys)
        return jsonable({"transitions": self.transitions, "stages": self.stages, "tracker": self.tracker.to_dict(),
                         "unplaced": self.unplaced, "reference_step": self._ref_index,
                         "probe_tracker": self.probe_tracker, "dense_uv": self.dense_uv, "dense_features": dense,
                         "adaptive_search": self.adaptive_search, "grid": self.grid_info})


def _leg_sigma(leg_kappas, leg_dirs) -> float:
    """D4, second estimate of a feature's kappa uncertainty from its probe legs: legs along one axis (parallel or
    antiparallel within D1_LEG_SPAN_DEG: out and back revisit the same image positions, so they share the
    position-repeating tracking bias the residuals cannot see) are averaged into one direction estimate; the
    standard error over directions (two or more), else over the legs; 0 for fewer than two."""
    groups: list[list] = []
    limit = math.cos(math.radians(D1_LEG_SPAN_DEG))
    for kappa, direction in zip(leg_kappas, leg_dirs):
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 1e-12 else direction
        for group in groups:
            if abs(float(group[0] @ unit)) >= limit:
                group[1].append(kappa)
                break
        else:
            groups.append([unit, [kappa]])
    values = [float(np.mean(group[1])) for group in groups] if len(groups) >= 2 else list(leg_kappas)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _leg_kappa(samples, f, aperture=None) -> float | None:
    """kappa (at the reference depth) of one probe leg: the leg's samples are relative to the REFERENCE frame
    (exact axial model), with the leg's own intercept (its start frame), aperture-weighted; None when the leg spans
    fewer than D1_MIN_SHIFT_PX of measurable shift, has fewer than two frames, or moves along the template's weak
    direction (less than D1_LEG_MIN_INFORMATION of the information of an unweighted leg)."""
    if len(samples) < 2:
        return None
    weight = np.eye(2) if aperture is None else np.asarray(aperture, dtype=float)
    shifts = np.array([weight @ np.asarray(s[0], dtype=float) for s in samples])
    if float(np.max(np.linalg.norm(shifts - shifts[0], axis=1))) < D1_MIN_SHIFT_PX:
        return None
    moves = np.array([np.asarray(s[1], dtype=float) for s in samples])
    moves = moves - moves[0]
    raw = float(np.sum(moves ** 2))
    if raw <= 1e-12 or float(np.sum((moves @ weight.T) ** 2)) < D1_LEG_MIN_INFORMATION * raw:
        return None
    kappa, _, _, _ = fit_feature_kappa(samples, f_px=f, min_samples=2, aperture=aperture)
    return kappa


def _same_frame(a: StepView, b: StepView) -> bool:
    """Whether two views are the same observation (the policy builds a new StepView of the same observation for
    the next act, so identity of the views is not enough)."""
    if a is b:
        return True
    obs_a, obs_b = getattr(a, "observation", None), getattr(b, "observation", None)
    if obs_a is not None and obs_a is obs_b:
        return True
    return bool(np.array_equal(a.state, b.state) and obs_a is not None and obs_b is not None
                and getattr(obs_a, "wrist_image", None) == getattr(obs_b, "wrist_image", None))


def _textured(gray, uv) -> bool:
    image = _gray32(gray)
    h, w = image.shape
    r = FT_RADIUS_PX
    centre = np.clip(np.rint(np.asarray(uv, dtype=float)), [r, r], [w - 1 - r, h - 1 - r]).astype(int)
    return float(image[centre[1] - r:centre[1] + r + 1, centre[0] - r:centre[0] + r + 1].std()) >= D15_MIN_STD_GREY


def module_summary() -> dict:
    return {"geometry": GEOMETRY_VERSION, "r3_focal_px": r3_focal_px(), "r3_sigma_px": r3_sigma_px()}
