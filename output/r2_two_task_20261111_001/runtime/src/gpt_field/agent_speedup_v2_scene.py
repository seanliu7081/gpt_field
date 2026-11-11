"""agent_SpeedUp_v2 scene model (contract §C6, [C-2]): HeightField fusion, PartRecord registration, region masks,
keep-outs and the in-hand pose estimate. It plans nothing and moves nothing.

- Fusion (spec §4.5): depth samples stamp the kernel HeightField by source authority (touch > kappa = loom > dense >
  the D7 fills); unknown cells are filled by D7a (support-plane colour, no D9 edge, within 2 cm of the plane's known
  cells, applied transitively over plane-filled cells) or D7b (obstacle at the highest known neighbour within 2 cm
  plus 3 sigma). Touches stamp D5 surfaces. Cells of a lifted object are vacated [C-36].
- Registration (spec §4.5 steps 1-8, §3.6 items 1 and 7): a seed is lifted to 3-D; the measured heights near it are
  grouped into height bands (D8); every wrist pixel is labelled with a band by marker watershed on the image (D9
  intensity edges) whose markers are the measured samples (a sample outvoted in its own D9 segment is not one), the
  plane fills and, when the scene has cached views from a probe move, a two-view height sweep (``_parallax``); for a
  pointed (fine) seed a second pass removes from its band the pixels some other height explains better. The part is
  its band's largest D9-edge-free facet at the seed plus the band's thin or enclosed facets across internal edges;
  its support is the band that surrounds it; top_z, widths per jaw yaw in the pad strip, the elongation (D10), a
  circle (D11), a rigid patch and the external projection follow. Partial (border-cut) and provisional records carry
  sigmas that cover the region-choice ambiguity and, for a cut, the measured surfaces its hidden part may reach;
  their top is a lower bound (``PartRecord.top_bound`` "lower" with ``top_upper_z``, rulings R2).
- Design review 1 (decision §2.2, THRESHOLDS_VERSION v6): a wrist POINT seed (fine / reacquire / lookat) takes its
  height from the highest coherent band of the sigma-gated samples within TOP_WINDOW_PX of the seed (``_top_band``;
  S-1..S-3): the seed is lifted onto the band top and top_z is the band's 95th percentile with the honest part-level
  sigma_h = hypot(statistical, TOP_MODEL_M or TOP_MODEL_THIN_M); the model term enters sigma_h only, never a D8 band
  or the region growth. D10 needs the S-4' axis consensus (the deciding tensor's axis within YAW_GRID_RAD / 2 of the
  region cells' PCA axis, >= D10_MIN_REGION_CELLS cells whose own second-moment aspect is >= D10_MIN_ASPECT, so an
  isotropic cell set is no arbiter; ``PartRecord.axis_check``, refusal ``reason``). After a point-seeded
  registration the dense-lattice hover samples off the part are withdrawn from the HeightField (S-5 DSW; they stay
  registration markers); ``part_fill_cells`` gives the D7b fills that are the part's own halo (S-6, for BP's P7);
  the support must lie below the lowest top-band member (S-7); D7b parents' sigma is capped (RC-9a);
  ``absorb_survey(refine_pid=...)`` stamps only the target's samples (RC-9b); ``reanchor_from`` takes a HandLocator
  reading on an explicit settled pair and the cache scan pairs non-consecutive views (rulings item 4).
- Design review 2 (decision 2 §2.3, THRESHOLDS_VERSION v7): S-11 D10-OE replaces S-4' for a point-seeded whole
  (not side-cut) registration: the outline gate (G-7 ``outline_orientation``: share >= D10_OUTLINE_SHARE_MIN, second
  peak <= D10_OUTLINE_SECOND_MAX, two-sided) and a gated edge axis (G-7 ``edge_orientation`` in the region dilated
  D10_EDGE_REGION_DILATE_PX or the top-band hull dilated D10_EDGE_HULL_DILATE_PX, the robot mask excluded, share >=
  D10_EDGE_SHARE_MIN) within AXIS_AGREE_MAX_RAD; elongation = (outline axis, outline share); refusals NOT_ELONGATED
  "outline_gate" / AXIS_INCONSISTENT "no_edge_agreement" (``_d10_oe``); the S-4' tensor, aspect and cells are logged
  only (an area seed or a side cut gets no D10, NOT_ELONGATED with reason "area_seed" / "partial_side"). S-12: a per
  part view memory (``_views``; ``part_moved(pid)`` forgets it, ``view_memory(pid)`` reads it) of the accepted axes
  and the last non-collapsed width set: a fresh acceptance disagreeing with a remembered axis is refused
  "views_disagree" and conflicts the memory; a refusal while the memory holds an axis carries it ("kept_earlier",
  ``_axis_decision``). S-13: a region whose top-band hull (dilated COLLAPSE_HULL_DILATE_PX) holds more than
  REGION_COLLAPSE_RATIO x its pixels is collapsed: axis refused, width_by_yaw () unless the memory holds widths
  (``_collapse_check``). S-14: a whole registration reading below an existing lower-bound top keeps the flag
  (``_keep_lower_bound``, top_inconsistent): the earlier record is kept whole, the view's reading only logged.
  S-15: a point seed farther than q_now + S15_SEED_SIGMA_K x TRACK_SIGMA_PX from its region
  is refused (part_not_found "seed_outside_region"; lead ruling 3); the S-15 diagnosis adds one general rule
  (extended by lead ruling 7, below). S-16: the external hand-swept mask's margin is max((SWEPT_DILATE_PX - 1) / 2,
  ceil(2 sigma_proj)). S-17: the point seed's fallback band carries the ``_region_band`` keys; a side cut is a lower
  bound only for a measured (not a plane) part band (BSR-4); the support's S-7 fallback is the next admissible ring
  band (measured or plane) measured from the ring's own samples (BSR-6).
- Design review 2 close-out (lead rulings r2close 1-8): ruling 1 ratifies the ring-pixel majority as the primary
  support rule and the ring-own inverse-variance fallback (``_support``); ruling 2 ratifies S-14 keeping the WHOLE
  earlier record (``_keep_lower_bound``); ruling 3 widens the S-15 radius to q_now + S15_SEED_SIGMA_K x
  TRACK_SIGMA_PX (beyond it BR re-points once at the hover; the evidence logs q_now, the factor and gap - limit, and
  every registered point seed logs its ``seed_check``); ruling 4 (a): an S-11 edge ROI counts as agreeing evidence
  only when its own histogram passes the outline's second-peak gate (second <= D10_OUTLINE_SECOND_MAX) as well as
  its share gate, so an outline agreeing only with bimodal edges is refused AXIS_INCONSISTENT "no_edge_agreement"
  with ``edges_bimodal`` (``_d10_oe``); ruling 7: a point-seeded registration ON the measured base support plane
  (planes[0], the D8 same-surface test) registered the support, not the part, and is refused, part_not_found
  "band_on_support": its top band (``_band_on_support``: every point-seeded part on a non-provisional call, grip
  parts only on a provisional call) or, on a non-provisional call, its measured top (``_top_on_support``, an
  unbanded seed included); area-seeded records are never refused by it.
- Wave 2 entry gate (lead decisions F1-F3, BS r3): F1 every width sigma carries the part-level model term,
  sigma_w = hypot(pixel sigma_w, W_MODEL_M) (the width values are untouched); F2 S-5 also withdraws the part's own
  dense-lattice samples INSIDE the region dilated by one cell that read above the record's top + max(D8_FLOOR_M,
  D8_SIGMA_K sigma_h) (never for a lower-bound top, nor when most of the region's own lattice reads above it, BS r3
  review; ``last_dsw`` "withdrawn_near"); F3 S-6 grows through unmeasured
  cells (unknown or D7b fills, never D7a plane fills) that touch measured raised cells, so a part's own body joined
  through an unmeasured cell or two is connected, while observed support or a run of >= 3 unmeasured cells still
  separates a raised neighbour (``part_fill_log`` "bridged_cells").
- Live-first blockers (BSBP fixer; effect lens fix (c) and grip-part lens BS-1 / BS-2): S-16' replaces S-16 for the
  external hand-swept mask (``hand_swept_mask``): [ROBOT] the palm prism (R7 rectangle from the R6 palm underside up
  to the hand) and two finger columns (R1 +-q .. +-(q + the asset's finger_and_pad_R1 outer edge), R0 +- 6.14 mm,
  from the R5 fingertip reach up to the palm underside), each hulled per consecutive footprint pair; projected with
  ``hand_projector`` (the kept HandLocator reading nearest the grip in 3-D, never the [C-51] region projector); within
  HAND_MASK_LOCAL_RADIUS_M of that anchor every hull vertex takes its own margin max(4 px, ceil(2 max(sigma_grip,
  sigma_vertex))) (BSBP amendment of the spec's single max(4 px, ceil(2 sigma_grip)): the spec rule failed the RS-9
  gate, 8 of 10 calibration-yaw hovers, min 96.5 %, in both runs; the amendment passed 20 of 20 in both), the
  far-vertex sigma as one kernel beyond; logged in ``mask_log["hand_swept"]``. BS-1: an area-seeded registration (plan cell, coarse, an external lift; provisional
  included) whose region shares cell IoU >= SHARED_REGION_IOU_MIN with another part's current region is refused
  part_not_found "region_of_part" (the grip part of a plan grip / obj pair, else the later registration; the refused
  part keeps its record); the object registered after its grip part keeps the region and the grip part returns to
  its record before its area-seeded registration (``_yield_region``); ``region_overlap(a, b)`` gives the cell IoU
  (BR-1 / BR-3). BS-2: ``hf.unconfirmed`` marks single measured cells above every known 8-neighbour by more than the
  D8 tolerance (>= UNCONFIRMED_MIN_NEIGHBOURS known) until agreeing samples of >= UNCONFIRMED_MIN_GROUPS measurement
  groups confirm them (``_refresh_unconfirmed`` after every HeightField write; never deleted). BS-3 (the object's
  lower-bound raise) is traced, not changed: S-14 keeps the whole earlier lower-bound record (lead ruling 2), so a
  whole survey view reading the object's main surface below its lower-bound top never clears top_upper_z.
- Scene-side photometric sweeps (ratified for the slice, decision §6 R7; the cost kernel moves to geometry at the
  next BG / BS pass): (1) ``_support_plane_by_parallax`` at ``start`` finds the support plane's height and colour
  from the Calibrate+ probe views when its texture is too weak for kappa features; that plane seeds the D7a plane
  fills, so it IS a depth source for the support plane (not listed in spec §4.5; its fills are counted by plane
  source in ``fill_unknown`` and ``summary``); (2) ``_parallax`` labels pixels for registration only and never stamps
  the HeightField.
- Region masks per camera (O1 / O7), the hand-swept mask (O7), the keep-out store and the in-hand pose estimate
  (spec §4.7, M14, M17) are the other services.

Notes for Wave 2 (BR, BX, BO, BP; BS review R3, R5, R12; design review 1):
- ``register(provisional=True)`` accepts a border-cut grip-only part on its own (any cut below
  PARTIAL_MAX_CUT_FRACTION, then partial=True); no ``allow_partial`` is needed for provisional calls. A part with
  no wrist seed (plan w = 0) is registered from its external seed (``estimate_seed_xyz`` lifts it; ``predicted_uv``
  falls back to it too).
- A record's top is a lower bound when ``PartRecord.top_is_lower_bound()`` (a side cut by the image border or the
  robot, the [C-38] partial view); ``top_upper_z`` = max(start fingertip height, highest measured surface).
  ``needs_survey_hover(pid)`` says when BR runs ``survey_hover`` before relying on the part (rulings item 8); a
  whole re-registration clears the flag. An END cut (an elongated part running under the fingers) is not partial
  and its top is not a lower bound.
- ``predicted_uv``'s sigma is the coarse-first input of rulings item 9: point coarse first when it exceeds
  64 / sqrt(2 ln 2) px (BR owns the rule).
- After a place, call ``release_held(view)`` with the release view: the object's record follows it (``moved``).
- ``region_mask(..., "external")`` may re-anchor the projector [C-51]: from then on ``scene.projector`` differs from
  ``cal.projector``; use ``scene.projector``. A refined mask failing the sanity bounds is replaced by the affine one,
  flagged "refine_rejected" in ``mask_log``.
- ``hand_swept_mask``'s ``yaw_range_rad`` is relative to the current hand yaw (0 = now).
- Design review 2 (BR): call ``scene.part_moved(pid)`` after any contact with the part or its object and after a
  grasp, release or carry (``release_held``, ``drop`` and ``vacate_held`` call it for the parts they rewrite); a hover
  sequence chain -> second stage -> BR-1 is one memory window. BR-1 triggers on ``axis_check["source"]`` ==
  AXIS_INCONSISTENT or ``axis_check["region_collapsed"]`` on a grip part, never on NOT_ELONGATED. A record with
  ``width_by_yaw == ()`` has an unmeasured width (K-5, S-13). ``mask_log["hand_swept"]`` logs the S-16 margin.
- Lead rulings r2close (BR): a part_not_found "seed_outside_region" (S-15) means re-point ONCE at the hover (fine
  pointing centred on ``predicted_uv``, coarse first when its sigma exceeds 64 / sqrt(2 ln 2) px) and register again
  before giving up (ruling 3); a part_not_found "band_on_support" means the pointed window or region measured the
  base support (ruling 7). An ``axis_check["edges_bimodal"]`` refusal is AXIS_INCONSISTENT "no_edge_agreement"
  (ruling 4 (a)), so it triggers BR-1 on a grip part like any other.
- ``HeldFrame.extent_axes`` is packed (3, 2): rows major axis, minor axis (unit world xy), extents (m);
  ``HeldFrame.tilt_deg`` stays None (the external principal-axis tilt cue of spec §4.7 item 5 is not implemented;
  M14 reads the wrist template shift, the silhouette IoU, the external co-motion and the gap change).

Geometry (vision) is imported lazily inside functions as ``from . import agent_speedup_v2_geometry as geometry``
(contract §C2.2); tests inject a stand-in with the same functions through ``SceneModel(robot, vision=...)``.
Frames and units follow the kernel (contract §C1.3); every threshold the scene compares against is a runtime
Threshold record listed in the registration / fill / held-frame evidence.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import copy
import dataclasses
import math
import time
import warnings

import cv2
import numpy as np

from . import agent_speedup_v2_robot as robot_facts
from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (AXIS_AGREE_MAX_RAD, AXIS_INCONSISTENT, AXIS_KEPT_EARLIER, DENSE_FEATURE_KEY_PREFIX,
                                     HF_CELL_M, HF_FILLS, HF_SOURCES, NOT_ELONGATED, RIM_WALL_RADIUS_FRACTION,
                                     YAW_GRID_RAD, DepthSample,
                                     HandPose, HeightField, HeldFrame, KeepOut, Meas, PartRecord, RegistrationFailure,
                                     RobotModel, Seed, StepView, Term, WristCamera, jsonable, poly_contains,
                                     object_words, poly_inflate, rect_polygon, same_object, th_max, th_value,
                                     thresholds_dicts)
from .mark_prompting import coarse_cell_center, unrotate_pixel
from .wrist_servo import NCCTrack

# scene-7: gap round (judge decision §2.2): X-1 disturbance epochs (``disturb`` / ``epoch_of`` / ``window_start``),
# SAL-R1 stale-fill consistency, SAL-R4 BS-2 neighbours without D7b fills, HS-H1 whole-body held frame, HS-H2
# external-template ownership, MC-R1 seed-band fallback, MC-R2a kept whole record, MC-R2b annulus model, MC-R3
# part-object consistency (PO3, PO1s, PO5, PO6).
SCENE_VERSION = "agent-speedup-v2-scene-7"

# --- D7 unknown cells (spec §4.5 HeightField, §5.2 D7)
D7_PLANE_RADIUS_M = 0.02
D7_COLOUR_SIGMA_K = 3.0
D7_PLANE_SIGMA_K = 3.0
D7_OBSTACLE_RADIUS_M = 0.02
D7_OBSTACLE_SIGMA_K = 3.0
# --- D8 same plane, D9 region edge
D8_FLOOR_M = 0.003
D8_SIGMA_K = 3.0
D9_GRADIENT_RATIO = 4.0
D9_DEPTH_SIGMA_K = 3.0
D9_MEDIAN_WINDOW_PX = 21
# --- D10 elongated, D11 circle
D10_MIN_COHERENCE = 0.5
D10_MIN_ASPECT = 1.5
D10_PATCH_WIDTH_FRACTION = 0.6
D10_PATCH_MIN_HALF_PX = 12
D11_RMS_FLOOR_PX = 1.5
D11_RMS_FRACTION = 0.03
D11_MIN_POINTS = 12
D11_OUTLIER_K = 3.0
# --- M14 Touchdown trigger, M17 offset sigma fallback (spec §4.7)
M14_GAP_FLOOR_MM = 1.0
M14_GAP_SIGMA_K = 6.0
M14_IOU_MIN = 0.8
M14_SHIFT_FLOOR_PX = 4.0
M14_SHIFT_SIGMA_K = 3.0
M17_FRACTION = 0.5
# --- registration (spec §4.5 steps 4 and 8)
TOP_PERCENTILE = 95
RIGID_PATCH_MIN_DIST_M = 0.015
# --- scene design defaults (descriptive prefixes; derivations in LABELS)
PLANE_MIN_CELLS = 10
PLANE_MAX_COUNT = 4
BAND_MIN_SAMPLES = 3
BAND_MAX_COUNT = 8
BAND_EM_ITERATIONS = 50
MARKER_RADIUS_PX = 2
EDGE_HALF_WIDTH_PX = 1
REGION_MIN_PX = 25
REJECT_MIN_PX = 3
REGION_CHOICE_SIGMA_K = 3.0
FACET_BRIDGE_PX = 7
FACET_MERGE_RATIO = 1.0
FACET_ENCLOSED_FRACTION = 0.75
PARTIAL_CUT_FRACTION = 0.15
PARTIAL_MAX_CUT_FRACTION = 0.5
SUPPORT_RING_M = 0.015
PARALLAX_WINDOW_PX = 7
PARALLAX_MIN_BASELINE_M = 0.01
PARALLAX_MAX_BASELINE_M = 0.05
PARALLAX_MAX_AXIAL_M = 0.01
PARALLAX_MAX_FRAMES = 4
PARALLAX_DROP_WORST_MIN = 3
REGION_PCA_MAX_POINTS = 4096
PARALLAX_MARGIN_K = 3.0
PARALLAX_SWEEP_STEP_M = 0.005
PARALLAX_CROP_HALF_PX = 128
PARALLAX_SWEEP_MARGIN_M = 0.02
KEYFRAME_SPACING_M = 0.01
KEYFRAME_MAX_COUNT = 12
KEYFRAME_MAX_AGE_STEPS = 400
RECENT_VIEW_COUNT = 32
TRACK_SIGMA_PX = 1.5
NCC_RADIUS_PX = 10
GREY_QUANTIZATION_SIGMA = 1.0 / math.sqrt(12)
INNER_FOV_HALF_RAD = math.radians(30.0)
SWEEP_STEP_RAD = math.radians(5.0)
SWEPT_DILATE_PX = 9
REANCHOR_RADIUS_M = 0.15
REANCHOR_MAX_TRIES = 4
REFINE_MAX_AREA_RATIO = 2.0
REFINE_MIN_IOU = 1.0 / 3.0
RELOCALIZE_MAX_DRAG_M = 0.04
PRE_CLOSE_GAP_MARGIN_MM = 5.0
SAME_POSE_M = 0.002
SAME_POSE_RAD = math.radians(0.5)
STATIC_DIFF_FLOOR_GREY = 3.0
STATIC_DIFF_K = 4.0
STATIC_WINDOW_PX = 5
MAD_TO_SIGMA = 1.4826
SEED_REGION_RADIUS_PX = 64
SUPPORT_SWEEP_M = 0.5
SUPPORT_SWEEP_STEP_M = 0.005
SUPPORT_FINE_STEP_M = 0.0005
SUPPORT_DOWNSAMPLE = 4
SUPPORT_MIN_CONTRAST = 0.2
RELOCALIZE_PATCHES = 8
# --- design review 1 (decision §2.2 and §3, THRESHOLDS_VERSION v6)
TOP_WINDOW_PX = 40
TOP_GATE_M = 0.002
TOP_BAND_K = BAND_MIN_SAMPLES                  # decision §3: 3 (= BAND_MIN_SAMPLES)
TOP_BAND_SIGMA_K = 2.0
TOP_BAND_SUPPORT_N = 10
TOP_MODEL_M = 0.003
TOP_MODEL_THIN_M = 0.005
D10_MIN_REGION_CELLS = 4
PB_REACH_M = 0.05
D7_PARENT_SIGMA_CAP_M = HF_CELL_M              # decision §3 / RC-9a: the field's resolution (5 mm)
REFINE_SIGMA_K = REGION_CHOICE_SIGMA_K         # RC-9b: the target's region dilated by 3 sigma_xy (see LABELS note)

# --- design review 2 (decision 2 §2.3 / §3, THRESHOLDS_VERSION v7): D10-OE (S-11) and the collapse guard (S-13)
D10_OUTLINE_SHARE_MIN = 0.20
D10_OUTLINE_SECOND_MAX = 0.8
D10_EDGE_SHARE_MIN = 0.20
D10_EDGE_REGION_DILATE_PX = 6
D10_EDGE_HULL_DILATE_PX = 8
REGION_COLLAPSE_RATIO = 2.0
COLLAPSE_HULL_DILATE_PX = 2
# --- design review 2 close-out (lead rulings, r2close): the S-15 seed radius (ruling 3)
S15_SEED_SIGMA_K = 2.0
# --- Wave 2 entry gate fixes (lead decisions F1-F3, BS r3): the part-level width model term (F1)
W_MODEL_M = 0.0015
# --- live-first blockers (effect lens S-16', grip-part lens BS-1 / BS-2)
HAND_MASK_LOCAL_RADIUS_M = 0.10
SWEEP_DISC_SIDES = 16
SHARED_REGION_IOU_MIN = 0.5
UNCONFIRMED_MIN_NEIGHBOURS = 3
UNCONFIRMED_MIN_GROUPS = 2
# --- grasp-refusal decision J (THRESHOLDS v8), remedy 1 S-17: own-band ring
OWN_RING_CELLS = 1
# --- gap round (judge decision §2.2 / §6, THRESHOLDS v8): MC-R2b annulus arc, MC-R3 PO5 sub-part region bound
ANNULUS_MIN_ARC_DEG = 90.0
SUBPART_REGION_MAX_RATIO = 1.0

LABELS = {
    "D7_PLANE_RADIUS_M": ("DD", "spec §4.5 D7a: a cell within 2 cm of a fitted support plane's known cells may be "
                                "plane-filled (spec §5.2 D7)"),
    "D7_COLOUR_SIGMA_K": ("DD", "spec §4.5 D7a colour test |dLab| <= 3 sigma_plane of the plane's measured colour "
                                "(Lab in OpenCV 8-bit scaling [C-20])"),
    "D7_PLANE_SIGMA_K": ("DD", "spec §4.5 D7a: plane-filled cells sit at the plane height + 3 sigma_plane"),
    "D7_OBSTACLE_RADIUS_M": ("DD", "spec §4.5 D7b: obstacle fill from the known neighbours within 2 cm"),
    "D7_OBSTACLE_SIGMA_K": ("DD", "spec §4.5 D7b: obstacle at the highest known neighbour + 3 sigma (of that "
                                  "neighbour)"),
    "D8_FLOOR_M": ("DD", "spec §5.2 D8 same plane |dh| <= max(3 mm, 3 sigma_h): the 3 mm floor"),
    "D8_SIGMA_K": ("DD", "spec §5.2 D8 same plane: 3 sigma_h"),
    "D9_GRADIENT_RATIO": ("DD", "spec §5.2 D9 region edge: |grad I| > 4 x the local median, here the larger of the "
                                "local median over D9_MEDIAN_WINDOW_PX and the image median (a flat window's median "
                                "is ~0 and would make noise an edge)"),
    "D9_DEPTH_SIGMA_K": ("DD", "spec §5.2 D9 region edge: a depth step > 3 sigma_h separates two height bands"),
    "D9_MEDIAN_WINDOW_PX": ("DD", "D9 local median window: the 21 px feature patch of spec §5.2 D15 (v1 NCCTrack "
                                  "radius 10), so the median is taken over one texture patch"),
    "D10_MIN_COHERENCE": ("DD", "spec §5.2 D10 structure-tensor coherence >= 0.5 (v1 BAR_CHECK_MIN_COHERENCE, "
                                "wrist_servo.py:79)"),
    "D10_MIN_ASPECT": ("DD", "spec §5.2 D10 elongated when the region aspect w_long / w_short >= 1.5"),
    "D10_PATCH_WIDTH_FRACTION": ("DD", "D10 tensor patch half-size = 0.6 x the measured cross width in px, so the "
                                       "patch spans both long edges and 0.1 w beyond each (the [C-40] track rule)"),
    "D10_PATCH_MIN_HALF_PX":("DD", "D10 tensor patch half-size floor = v1 BAR_CHECK_HALF_PX (wrist_servo.py:77)"),
    "D11_RMS_FLOOR_PX": ("DD", "spec §5.2 D11 circle accepted when rms <= max(1.5 px, 3 % r): the 1.5 px floor (v1 "
                               "PIXEL_NOISE_PX)"),
    "D11_RMS_FRACTION": ("DD", "spec §5.2 D11 circle accepted when rms <= 3 % of the radius"),
    "D11_MIN_POINTS": ("DD", "D11 circle fit needs >= 12 outline points (4 x the 3 circle parameters)"),
    "D11_OUTLIER_K": ("DD", "D11 robust refit: outline points beyond 3 robust sigma (1.4826 MAD) of the radial "
                            "residual (an occluding part, the robot or a label spur) are dropped, twice"),
    "M14_GAP_FLOOR_MM": ("DD", "spec §5.2 M14 Touchdown trigger: lift gap change > max(1 mm, 6 sigma_close)"),
    "M14_GAP_SIGMA_K": ("DD", "spec §5.2 M14: 6 sigma_close (G4)"),
    "M14_IOU_MIN": ("DD", "spec §5.2 M14: lift silhouette IoU against the pre-lift mask < 0.8 triggers Touchdown"),
    "M14_SHIFT_FLOOR_PX": ("DD", "spec §5.2 M14: template shift > max(4 px, 3 sigma_track)"),
    "M14_SHIFT_SIGMA_K": ("DD", "spec §5.2 M14: 3 sigma_track"),
    "M17_FRACTION": ("DD", "spec §5.2 M17: IoU < 0.8 or not wrist-visible -> sigma_offset = max(measured, "
                           "0.5 r_obj)"),
    "TOP_PERCENTILE": ("DD", "spec §4.5 registration step 4: top_z = the 95th percentile of the region heights "
                             "(inverse-variance weighted; taken as the median of the samples within the D8 band of "
                             "that percentile, sigma = max(their median sigma, their robust scatter))"),
    "RIGID_PATCH_MIN_DIST_M": ("DD", "spec §4.5 registration step 8: the rigid patch lies >= 15 mm from the grip "
                                     "point"),
    "PLANE_MIN_CELLS": ("DD", "a fitted support plane needs >= 10 measured cells (relative error of a colour std "
                              "from n cells ~ 1/sqrt(2(n-1)) = 24 %)"),
    "PLANE_MAX_COUNT": ("DD", "at most 4 support planes are fitted (the support and up to 3 raised planar "
                              "regions in one start view)"),
    "BAND_MIN_SAMPLES": ("DD", "a height band needs >= 3 samples (fewer is not a surface but a stray sample)"),
    "BAND_MAX_COUNT": ("DD", "at most 8 height bands in one view (support, up to 4 planes, parts on them)"),
    "BAND_EM_ITERATIONS": ("DD", "height-mode mixture fit: at most 50 EM iterations (converges in < 20 on the "
                                 "fake and recorded views)"),
    "MARKER_RADIUS_PX": ("DD", "watershed marker disk of a height sample: 2 px, inside the half-cell footprint of a 5 "
                               "mm HeightField cell at the start view (~4.5 px per cell)"),
    "EDGE_HALF_WIDTH_PX": ("DD", "a region takes its D9 outline pixels within 1 px: half the 2-3 px band a 3 x 3 Sobel "
                                 "edge covers, so the outline sits at the edge's middle"),
    "REGION_MIN_PX": ("DD", "a registered region needs >= 25 px (5 x 5 px, about one HeightField cell at the start "
                            "view)"),
    "REJECT_MIN_PX": ("DD", "a parallax rejection of the pointed band counts from 3 connected pixels (an isolated "
                            "pixel or pair is window noise; a crease line of window labels is kept)"),
    "REGION_CHOICE_SIGMA_K": ("DD", "an area seed's (plan cell, coarse) region choice: the farthest other candidate "
                                    "region in the seed square lies within 3 sigma of the chosen one (lateral and "
                                    "height), the D8 3 sigma"),
    "FACET_BRIDGE_PX": ("DD", "a same-band facet joins the face across an intensity edge line up to 7 px wide (one "
                              "PARALLAX_WINDOW_PX window: the double D9 line of a bevel at the hover)"),
    "FACET_MERGE_RATIO": ("DD", "a same-band facet joins the pointed face only when it is no larger than the face grown "
                                "so far (bevels, insets and markings border a face and are smaller; a larger same-height "
                                "surface beyond an intensity edge is the support or the body the part is attached to)"),
    "FACET_ENCLOSED_FRACTION": ("DD", "a same-band facet is a marking or an inset of the face when >= 3/4 of its outline "
                                      "lies within FACET_BRIDGE_PX of the face (an attached body or a neighbour meets "
                                      "the face along one side only)"),
    "PARTIAL_CUT_FRACTION": ("DD", "[C-38] a region is partial when >= 15 % of its outline lies on the image border "
                                   "or the robot's own pixels (one short side of a 3:1 part is 12.5 %)"),
    "PARTIAL_MAX_CUT_FRACTION": ("DD", "[C-38] more than half the outline cut: less than half the part seen, not "
                                       "registered"),
    "SUPPORT_RING_M": ("DD", "spec §4.5 step 3 annulus around the region: 15 mm, the relief-probe offset of spec "
                             "§4.5 item 2"),
    "PARALLAX_WINDOW_PX": ("DD", "two-view band test: 7 x 7 px photometric window (half the 21 px D15 patch)"),
    "PARALLAX_MIN_BASELINE_M": ("DD", "a cached view is a parallax partner when the hand moved >= 1 cm in the image "
                                      "plane (half the D2 2 cm hover probe)"),
    "PARALLAX_MAX_BASELINE_M": ("DD", "a parallax partner lies within 5 cm of this view in the image plane: the 4 cm "
                                      "Calibrate+ probe moves (v1 PROBE_MOVES) plus 1 cm of realized-motion slack; a "
                                      "farther cached view (a transit) differs in occlusion and shading"),
    "PARALLAX_MAX_AXIAL_M": ("DD", "a parallax partner lies within 1 cm of this view along the optical axis (probe moves "
                                   "are in-plane: 2.2 mm axial over a 4 cm move at the 3.14 deg start tilt; half the D2 "
                                   "probe)"),
    "PARALLAX_MAX_FRAMES": ("DD", "at most 4 parallax partners (the D2 probe's 4 legs)"),
    "PARALLAX_DROP_WORST_MIN": ("DD", "with >= 3 valid partners the worst is dropped (a surface beside a taller one is "
                                      "occluded in the partners that move toward it; with 2 the mean of both is kept)"),
    "REGION_PCA_MAX_POINTS": ("DD", "region principal axes from at most 4096 back-projected pixels (a uniform stride "
                                    "over the region bounds the cost; 4096 = 64 x 64, a region of a fine-crop's size)"),
    "PARALLAX_MARGIN_K": ("DD", "a pixel's band is confident when the second-best band's photometric cost exceeds the "
                                "best by 3 x the pixel-difference noise"),
    "PARALLAX_SWEEP_STEP_M": ("DD", "two-view band test sweep step 5 mm (one HeightField cell; the band heights "
                                    "themselves are always swept)"),
    "PARALLAX_CROP_HALF_PX": ("DD", "a point seed's two-view band test covers +-128 px around it (two V6 fine crops): "
                                    "the pointed part at its hover"),
    "PARALLAX_SWEEP_MARGIN_M": ("DD", "two-view band test sweeps 2 cm (the D2 hover probe amplitude) beyond the "
                                      "lowest and highest band, so the extreme bands have rival heights on both sides"),
    "OTHER_HEIGHT": ("DD", "parallax label code of a confident height at none of the measured bands (band labels are "
                           ">= 0, unconfident pixels -1)"),
    "KEYFRAME_SPACING_M": ("DD", "cached views are kept when >= 1 cm from every kept view (half the D2 probe)"),
    "KEYFRAME_MAX_COUNT": ("DD", "at most 12 cached views at distinct poses (Calibrate+ probe corners + one hover "
                                 "probe)"),
    "KEYFRAME_MAX_AGE_STEPS": ("DD", "cached views older than 400 steps are dropped (2 x the ~200-step Calibrate+)"),
    "RECENT_VIEW_COUNT": ("DD", "the last 32 views are kept (longer than a 30-action grasp close, v1 GRASP_SETTLE)"),
    "NCC_RADIUS_PX": ("DD", "v1 NCCTrack default template radius 10 px (21 px patch, wrist_servo.py:117), restated "
                            "[C-15]; equality tested"),
    "TRACK_SIGMA_PX": ("DD", "image localization noise 1.5 px = v1 PIXEL_NOISE_PX (wrist_servo.py:33)"),
    "GREY_QUANTIZATION_SIGMA": ("DD", "8-bit rounding noise 1/sqrt(12) grey: the floor of the static pixel noise"),
    "INNER_FOV_HALF_RAD": ("DD", "spec §3.6 item 5 / §4.7 item 2: the inner 30 deg of the 37.5 deg wrist half-FOV "
                                 "(R3)"),
    "SWEEP_STEP_RAD": ("DD", "hand-swept mask: footprints every 5 deg, consecutive hulls joined (corner chord 9.5 mm "
                             "at the 108.7 mm R7 footprint corner radius, inside the joined hull)"),
    "SWEPT_DILATE_PX": ("DD", "hand-swept mask dilation default = O1_HAND_DILATE_PX: a 9 px square kernel (4 px per "
                              "side), the v1 hand_image_motion dilation (external_axes.py:53), restated"),
    "REANCHOR_RADIUS_M": ("DD", "[C-51] the affine external projector is validated within 15 cm of its anchor "
                                "(contract §C5 test 7); farther regions re-anchor"),
    "REANCHOR_MAX_TRIES": ("DD", "[C-51] at most 4 HandLocator attempts per region mask on cached moving-hand frame "
                                 "pairs (newest first; each a frame difference)"),
    "REFINE_MAX_AREA_RATIO": ("DD", "[C-51] a refined external mask larger than 2 x the affine mask is a flood "
                                    "(refinement corrects the position and adds the visible sides), the affine stands"),
    "REFINE_MIN_IOU": ("DD", "[C-51] a refined external mask sharing IoU < 1/3 with the affine mask lies elsewhere (an "
                             "offset of half the region along one axis gives 1/3; the [C-51] affine error, ~20 % of the "
                             "shift, ~10 px at 0.37 m, is well inside), the affine stands"),
    "RELOCALIZE_MAX_DRAG_M": ("DD", "spec §4.7 item 3: close drag search up to 4 cm ([F] D003 drag 2.6 cm + 50 %)"),
    "PRE_CLOSE_GAP_MARGIN_MM": ("DD", "a cached view is pre-close when its finger gap exceeds the closed gap by >= 5 "
                                      "mm (10 x the G5 floor)"),
    "SAME_POSE_M": ("DD", "two views share a pose within 2 mm (twice the 1 mm probe net displacement, contract §C8 "
                          "test 2)"),
    "SAME_POSE_RAD": ("DD", "two views share a pose within 0.5 deg (the hand-yaw stall resolution, contract §C8)"),
    "STATIC_DIFF_FLOOR_GREY": ("DD", "zero-flow pixels differ by < 3 grey (v1 external_axes.py:25: the static scene "
                                     "differs by < 3 between frames)"),
    "STATIC_DIFF_K": ("DD", "zero-flow pixels differ by <= 4 sigma of the static pixel noise (O1 pixel rule)"),
    "STATIC_WINDOW_PX": ("DD", "zero-flow test on 5 x 5 px means (one texture grain of the 21 px D15 patch)"),
    "SEED_REGION_RADIUS_PX": ("DD", "a textureless seed takes the heights of its D9 segment within 64 px: half the "
                                    "128 px fine crop (V6) the pointing answer refers to"),
    "SUPPORT_SWEEP_M": ("DD", "support-plane parallax sweep: 0.5 m below the fingertips; R14 puts the start grip "
                              "0.2655 m above the R13 mount base, which stands on its support, so 0.5 m reaches "
                              "supports down to ~0.23 m below the mount base"),
    "SUPPORT_SWEEP_STEP_M": ("DD", "coarse support sweep step: one 5 mm HeightField cell"),
    "SUPPORT_FINE_STEP_M": ("DD", "fine support sweep step 0.5 mm (1/6 of the D8 3 mm floor), +-1 coarse step"),
    "SUPPORT_DOWNSAMPLE": ("DD", "coarse support sweep on 4x downsampled images (128 px at 512)"),
    "SUPPORT_MIN_CONTRAST": ("DD", "a parallax support plane is accepted when its photometric cost is >= 20 % below "
                                   "the sweep median (a support with texture to measure)"),
    "RELOCALIZE_PATCHES": ("DD", "close-drag re-localization tracks up to 8 separated patches and takes the shift "
                                 "most of them agree on (a drag can carry part of the object out of view or under "
                                 "the fingers; 8 leaves a majority when half the object leaves the view)"),
    "MAD_TO_SIGMA": ("DD", "robust sigma = 1.4826 x the median absolute deviation (1 / Phi^-1(3/4) for a normal "
                           "distribution)"),
    # design review 1 (decision §2.2 / §3, v6); the aliases TOP_BAND_K (= BAND_MIN_SAMPLES), D7_PARENT_SIGMA_CAP_M
    # (= HF_CELL_M) and REFINE_SIGMA_K (= REGION_CHOICE_SIGMA_K) carry no literal of their own. REFINE_SIGMA_K is
    # RC-9b's "region dilated by 3 sigma_xy" (remedy 2 §RC-9 (b)): a lateral containment multiple, the same 3 sigma
    # (the D8 3 sigma) by which REGION_CHOICE_SIGMA_K contains an area seed's candidate regions laterally; it shares
    # the value, not the region-choice meaning (BS review BSR-9 b)
    "TOP_WINDOW_PX": ("DD", "decision §2.2 S-2 / §3 (= geometry FT_DENSE_WINDOW_PX): a point seed's top band takes the "
                            "samples projecting within 40 px of the seed in the seed's pose (~1.2-2 cm at hover depth, "
                            "the probe tracker's dense lattice window; loop lens and remedy 1 E2)"),
    "TOP_GATE_M": ("DD", "decision §2.2 S-2 / §3: only samples with sigma_z <= 2 mm enter the top band (loop lens "
                         "sigma_z gate)"),
    "TOP_BAND_SIGMA_K": ("DD", "decision §2.2 S-1 / §3: band half-width b = max(D8_FLOOR_M, 2 x the median sample "
                               "sigma); the top band's lower edge in S-6 is top - max(D8_FLOOR_M, 2 sigma_h) (loop lens "
                               "ablation: floor 3 mm, k 2)"),
    "TOP_BAND_SUPPORT_N": ("DD", "decision §2.2 S-3 / §3: a top band of >= 10 members takes TOP_MODEL_M, fewer "
                                 "TOP_MODEL_THIN_M (loop lens: n < 10 -> p90 4.5 mm, max 13.3 mm); also the band size "
                                 "of the region_too_small hull fallback (S-4')"),
    "TOP_MODEL_M": ("DD", "decision §2.2 S-3 / §3, ruling 4b: the part-level model term of the top's sigma_h when the "
                          "band has >= TOP_BAND_SUPPORT_N members, 3 mm (remedy 1 E2: thin part 2 sigma coverage 0.979, "
                          "3 sigma 1.000, P6 pass 1.00, other parts 1.000; judge live sigma_h 3.0 mm); it enters "
                          "sigma_h only, never a D8 band or the region growth; re-derive at V5"),
    "TOP_MODEL_THIN_M": ("DD", "decision §2.2 S-3 / §3: the model term when the band has fewer than "
                               "TOP_BAND_SUPPORT_N members, 5 mm (remedy 1 E2: 3 sigma coverage 1.000 and below the "
                               "5.54 mm P6 limit; judge live sigma_h 5.0-5.1 mm)"),
    "D10_MIN_REGION_CELLS": ("DD", "decision §2.2 S-4' / §3: an elongation needs >= 4 HeightField cells in the region "
                                   "(S-4' rejected 5/5 bad axes and kept 14/14 good ones on the judge's 19 live "
                                   "records). The decision's derivation 'second moment defined with 2 x 2 cells' is "
                                   "corrected (BS review BSR-1): a 2 x 2 block's second moment is isotropic and "
                                   "defines no axis, so the cells' PCA axis also needs the cells' own second-moment "
                                   "aspect >= D10_MIN_ASPECT (no new number; pending the lead's sign-off)"),
    "PB_REACH_M": ("DD", "decision §2.2 S-6 / §3 (ROBOT + DD): part-band fill parents lie within 5 cm of the record "
                         "point: the R7 hand half-footprint along R0 (31.6 mm) + D7_OBSTACLE_RADIUS_M (20 mm), rounded "
                         "to 5 cm"),
    # design review 2 (decision 2 §2.3 / §3, v7). Every value was fitted on one task's two grip parts (the lens's
    # rows and the judge's live sets): re-derive on the cavity-edge and sliding-fixture regressions before M7
    "D10_OUTLINE_SHARE_MIN": ("DD", "decision 2 §2.3 S-11 / §3: the outline gate needs >= 20 % of the outline length "
                                    "within YAW_GRID_RAD / 2 of its orientation peak (axis lens sensitivity: 0 wrong "
                                    "axes at >= 0.20, one 16.4 deg wrong at 0.15)"),
    "D10_OUTLINE_SECOND_MAX": ("DD", "decision 2 §2.3 S-11 / §3: the outline's second peak (>= 2 windows away) is at "
                                     "most 0.8 of its main peak (axis lens outline gates; judge live2: the 8.3 deg "
                                     "wrong axis had 0.83). r2close lead ruling 4 (a): the same bound applies to each "
                                     "edge ROI's own histogram, an edge ROI counts as agreeing evidence only when its "
                                     "second peak is <= 0.8 (offline on every stored set: C acceptances backed only by "
                                     "bimodal edges 3 wrong of 31, by a unimodal agreeing ROI 6 of 265; the 89 deg "
                                     "near-isotropic acceptance had region-edge second 0.95)"),
    "D10_EDGE_SHARE_MIN": ("DD", "decision 2 §2.3 S-11 / §3: an edge ROI's orientation histogram is gated at share >= "
                                 "0.20 (axis lens)"),
    "D10_EDGE_REGION_DILATE_PX": ("DD", "decision 2 §2.3 S-11 / §3: edge ROI (a) = the region dilated 6 px (axis "
                                        "lens ROI form region: the silhouette edges just outside the outline)"),
    "D10_EDGE_HULL_DILATE_PX": ("DD", "decision 2 §2.3 S-11 / §3: edge ROI (b) = the top-band members' hull dilated "
                                      "8 px (axis lens ROI form hull)"),
    "REGION_COLLAPSE_RATIO": ("DD", "decision 2 §2.3 S-13 / §3: a region collapsed when its top-band hull holds more "
                                    "than 2 x its pixels (axis lens: the collapses of grip part A were bands of 75-94 "
                                    "members on 0-1 cells; judge live: 0 collapsed widths delivered)"),
    "COLLAPSE_HULL_DILATE_PX": ("DD", "decision 2 §2.3 S-13 / §3: the collapse guard's band hull is dilated 2 px (axis "
                                      "lens region_guarded)"),
    "S15_SEED_SIGMA_K": ("DD", "r2close lead ruling 3: the S-15 refusal radius is q_now + 2 TRACK_SIGMA_PX; measured "
                               "seed gaps 9.8-15.1 px refused at the 1 sigma radius on gate hovers (beyond the radius "
                               "BR re-points once at the hover). Offline (r2close analysis, q_now ~ 7.96 px, limit ~ "
                               "10.96 px): it admits 1 of the 4 logged C gate refusals (9.85 px); 11.0-15.1 px stay "
                               "refused, so the re-point carries the recovery"),
    # Wave 2 entry gate (lead decision F1, BS r3)
    "W_MODEL_M": ("DD", "lead decision F1 (Wave 2 entry gate): the part-level model term of every width sigma, "
                        "sigma_w = hypot(pixel sigma_w, W_MODEL_M), on the pattern of TOP_MODEL_M (147 unique C "
                        "hovers: width RMS 2.42 mm vs median pixel sigma_w 1.48 mm; offline on 21 current-code survey "
                        "sets 1.5 mm gives C 0 false G13+ trips and e2 >= 0.947 in every set of >= 10 hovers; the "
                        "hinged fixture part A, reported only, keeps 15 trips over those sets, 2.0 mm would clear them). "
                        "It enters sigma_w only, never a "
                        "width value, the narrowest-width choice, the S-13 collapse test or the region growth; "
                        "re-derive at V5"),
    # live-first blockers (effect lens fix (c) S-16'; grip-part lens BS-1 / BS-2)
    "HAND_MASK_LOCAL_RADIUS_M": ("DD", "effect lens S-16': within 0.10 m (3-D) of the hand projector's anchor the "
                                       "external hand mask's margin is local (max(4 px, ceil(2 sigma_grip)) in the "
                                       "spec; per hull vertex max(4 px, ceil(2 max(sigma_grip, sigma_vertex))) after "
                                       "the BSBP RS-9 amendment), the far-vertex sigma beyond it (truth replays of live "
                                       "smokes 2 and 3: the undilated palm + finger hull plus 2 sigma_grip covered >= "
                                       "98 % of the truth gripper silhouette in 100 % of 989 frames up to 0.12 m from "
                                       "the anchor, 0.00-0.82 of the frames beyond)"),
    "SWEEP_DISC_SIDES": ("DD", "S-16' local rule (BSBP amendment): a hull vertex's own 2 sigma margin is drawn as the "
                               "circumscribed 16-gon of that radius (never inside the disc; 2 % radial excess)"),
    "SHARED_REGION_IOU_MIN": ("DD", "grip-part lens BS-1: an area-seeded registration whose region shares HeightField-"
                                    "cell IoU >= 0.5 with another part's current region registered that part's "
                                    "region (live: IoU 1.0 in 3 of 3 coincidences; 0 coincidences in 67 ORACLE-run "
                                    "registrations)"),
    "UNCONFIRMED_MIN_NEIGHBOURS": ("DD", "grip-part lens BS-2: a single measured cell above every known 8-neighbour by "
                                         "more than the D8 tolerance is unconfirmed only with >= 3 known neighbours (a "
                                         "border or a sparse cell is no isolated spike)"),
    "UNCONFIRMED_MIN_GROUPS": ("DD", "grip-part lens BS-2: an unconfirmed cell is confirmed by a LATER sample within D8, "
                                     "i.e. agreeing samples of >= 2 measurement groups (absorbed surveys; a sample of "
                                     "no survey is a group of its own step): one probe's dense lattice is one group"),
    "OWN_RING_CELLS": ("DD", "remedy 1 S-17: the registered region's one-cell ring, as S-5 DSW and lead decision F2 "
                             "keep it (within one HF_CELL_M of the region the registration's own lateral sigma, 2.3 mm "
                             "median and 3.8 mm max on the stored records, leaves the part's outline), restricted to "
                             "measured cells in the part's S-6 top band"),
    "ANNULUS_MIN_ARC_DEG": ("DD", "gap round MC-R2b: an annulus model is accepted only on an arc of >= 90 deg of its "
                                  "circle (quarter-circle fit conditioning; accepted arcs on the recorded set "
                                  "108-262 deg, measure_consistency)"),
    "SUBPART_REGION_MAX_RATIO": ("RULING", "gap round MC-R3 PO5 (lead ruling L5): a grip sub-part's region holds at "
                                           "most 1.0 x the cells of its object's start-view region (definitional: a "
                                           "sub-part is not larger than its object; recorded correct <= 0.96, wrong "
                                           ">= 1.73)"),
}

_FILL_CODES = tuple(HF_SOURCES.index(name) for name in HF_FILLS)
_PLANE_CODE = HF_SOURCES.index("plane")
_OBSTACLE_CODE = HF_SOURCES.index("obstacle")
_TOUCH_CODE = HF_SOURCES.index("touch")
_POINT_SEEDS = ("fine", "reacquire", "lookat")
_SEED_PRIORITY = {name: index for index, name in enumerate(("plan_cell", "coarse", "reacquire", "fine", "lookat"))}
_CAMERAS = ("wrist", "external")
OTHER_HEIGHT = -2                          # parallax label: a confident height at none of the measured bands
_NCC_RADIUS = NCC_RADIUS_PX


# ------------------------------------------------------------------------------------------------- vision access
def _geometry_module():
    """The lazily imported geometry module (contract §C2.2)."""
    from . import agent_speedup_v2_geometry as geometry
    return geometry


# --------------------------------------------------------------------------------------------- small math helpers
def _project(camera: WristCamera, pose: HandPose, xyz):
    """Vectorized pinhole projection (contract §C1.3): (uv (n,2) with NaN behind the camera, depth z (n,))."""
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    d = pts - pose.camera
    a, b, z = d @ pose.R[:, 1], d @ pose.R[:, 0], d @ pose.R[:, 2]
    c = camera.principal_point
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([c[0] + camera.f_px * a / z, c[1] - camera.f_px * b / z])
    uv[~(z > 0.0)] = np.nan
    return uv, z


def _rays(camera: WristCamera, pose: HandPose, uv):
    """Ray directions (n,3) with a unit component along the optical axis (spec §4.5 world point formula)."""
    pix = np.asarray(uv, dtype=float).reshape(-1, 2)
    c = camera.principal_point
    return (pose.R[:, 2][None, :] + ((pix[:, 0] - c[0]) / camera.f_px)[:, None] * pose.R[:, 1][None, :]
            - ((pix[:, 1] - c[1]) / camera.f_px)[:, None] * pose.R[:, 0][None, :])


def _backproject_z(camera: WristCamera, pose: HandPose, uv, z_plane):
    """World points (n,3) where the pixels' rays meet the horizontal plane z = z_plane (NaN when they do not)."""
    d = _rays(camera, pose, uv)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(z_plane) - pose.camera[2]) / d[:, 2]
    out = pose.camera[None, :] + t[:, None] * d
    out[~(t > 0.0)] = np.nan
    return out


def _inside(uv, shape):
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    h, w = shape[:2]
    ok = np.isfinite(uv).all(axis=1)
    with np.errstate(invalid="ignore"):
        ok &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= h - 1)
    return ok


def _pix(uv):
    """Nearest integer pixel (columns u, v) of in-image uv."""
    return np.rint(np.asarray(uv, dtype=float).reshape(-1, 2)).astype(int)


def _is_dense_feature(feature) -> bool:
    """A survey feature of the probe tracker's dense lattice (key "dense:<k>", decision §2.3 G-1)."""
    key = getattr(feature, "key", None)
    return isinstance(key, str) and key.startswith(DENSE_FEATURE_KEY_PREFIX)


def _dilate_cells(mask, cells: int):
    """A bool HeightField mask grown by ``cells`` 8-connected cells (3 x 3 dilations)."""
    grown = np.asarray(mask, dtype=np.uint8)
    for _ in range(int(cells)):
        grown = cv2.dilate(grown, np.ones((3, 3), np.uint8))
    return grown > 0


def _cell_overlap(a, b, shape) -> dict:
    """HeightField-cell overlap of two (n, 2) ij sets: {"iou", "inter", "cells", "other_cells", "inside_fraction"}."""
    nx = int(shape[1])
    la = np.unique(np.asarray(a, dtype=int).reshape(-1, 2) @ np.array([nx, 1]))
    lb = np.unique(np.asarray(b, dtype=int).reshape(-1, 2) @ np.array([nx, 1]))
    inter = int(len(np.intersect1d(la, lb, assume_unique=True)))
    union = int(len(la) + len(lb) - inter)
    return {"iou": inter / union if union else 0.0, "inter": inter, "cells": int(len(la)), "other_cells": int(len(lb)),
            "inside_fraction": inter / len(la) if len(la) else 0.0}


def _hand_footprint_mm() -> tuple[tuple[float, float], tuple[float, float]]:
    """[ROBOT] R7: the asset's conservative hand rectangle (R1 range, R0 range) in mm (RobotModel's own source)."""
    asset = robot_facts.ROBOT_FACTS["asset"]
    r1, r0 = asset["hand_footprint_R1_mm"], asset["hand_footprint_R0_mm"]
    return (float(r1[0]), float(r1[1])), (float(r0[0]), float(r0[1]))


def _finger_column_extent_mm() -> tuple[float, float]:
    """[ROBOT] S-16' finger columns: (the finger_and_pad_R1 outer edge beyond the per-finger key q, the largest |R0|
    of a finger's lowest-band footprint), mm, over every asset key (26.4 mm and 6.14 mm in robot_facts_v2.json)."""
    outer, r0 = [], []
    for key, fingers in robot_facts.finger_landing_footprints().items():
        q = float(key)
        for finger in fingers.values():
            outer.append(max(abs(float(v)) for v in finger["finger_and_pad_R1_mm"]) - q)
            r0.append(max(abs(float(v)) for v in finger["R0_mm"]))
    return max(outer), max(r0)


def _box_xy(grip_xy, jaw_unit_xy, r1_range_mm, r0_range_mm) -> np.ndarray:
    """(4, 2) world-xy corners of a hand-frame box (R1 along the jaw, R0 = the jaw turned +90 deg about world +z, the
    kernel's top-down convention of RobotModel.hand_footprint_xy), ranges in mm."""
    jaw = np.asarray(jaw_unit_xy, dtype=float)
    jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-12)
    up = np.array([-jaw[1], jaw[0]])
    (a1, b1), (a0, b0) = r1_range_mm, r0_range_mm
    corners = np.array([[a1, a0], [b1, a0], [b1, b0], [a1, b0]], dtype=float) / 1000.0
    return np.asarray(grip_xy, dtype=float)[None, :2] + np.outer(corners[:, 0], jaw) + np.outer(corners[:, 1], up)


def _same_pose(a: HandPose, b: HandPose) -> bool:
    if float(np.linalg.norm(a.grip - b.grip)) > SAME_POSE_M:
        return False
    cos = (float(np.trace(a.R.T @ b.R)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cos))) <= SAME_POSE_RAD


def _robust_sigma(values) -> float:
    """1.4826 x the median absolute deviation (a normal sigma estimate robust to outliers)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    mad = float(np.median(np.abs(values - np.median(values))))
    return MAD_TO_SIGMA * mad


def _disk_offsets(radius_cells: int):
    r = int(radius_cells)
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]


def _shift(array, dy, dx, fill):
    """array shifted so that out[iy, ix] = array[iy + dy, ix + dx] (``fill`` outside)."""
    out = np.full_like(array, fill)
    ny, nx = array.shape[:2]
    ys, yd = (slice(dy, ny), slice(0, ny - dy)) if dy >= 0 else (slice(0, ny + dy), slice(-dy, ny))
    xs, xd = (slice(dx, nx), slice(0, nx - dx)) if dx >= 0 else (slice(0, nx + dx), slice(-dx, nx))
    out[yd, xd] = array[ys, xs]
    return out


def _polygon_area_centroid(poly):
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0, (p.mean(axis=0) if len(p) else np.zeros(2))
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = float(cross.sum()) / 2.0
    if abs(area) < 1e-12:
        return 0.0, p.mean(axis=0)
    cx = float(((x + xn) * cross).sum()) / (6 * area)
    cy = float(((y + yn) * cross).sum()) / (6 * area)
    return abs(area), np.array([cx, cy])


def _principal_axes(points):
    """(axes (2,2) rows = unit major / minor axis, extents (2,) = projection ranges) of world-xy points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T) if len(pts) > 2 else np.eye(2)
    values, vectors = np.linalg.eigh(cov)
    major, minor = vectors[:, 1], vectors[:, 0]
    extents = np.array([np.ptp((pts - centre) @ major), np.ptp((pts - centre) @ minor)])
    return np.vstack([major, minor]), extents


def _second_moment_axis(points):
    """(canonical unit major axis, sqrt(lambda_major / lambda_minor) or None when the minor moment vanishes) of the
    second moment of world-xy points (S-4' region cells): the ratio is the points' own aspect, 1 for an isotropic set
    (whose eigenvector is arbitrary), None (unbounded) for collinear points."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return None, None
    centred = pts - pts.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred / len(pts))
    lo, hi = float(values[0]), float(values[1])
    if hi <= 0.0:
        return None, None
    scale = float(np.abs(centred).max()) ** 2
    ratio = None if lo <= 1e-12 * scale else math.sqrt(hi / lo)
    return _canonical_axis(vectors[:, 1]), ratio


def _canonical_axis(axis_xy):
    a = np.asarray(axis_xy, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    if a[0] < -1e-9 or (abs(a[0]) <= 1e-9 and a[1] < 0.0):
        a = -a
    return a


def _clip_halfplane(poly, normal, offset):
    """Sutherland-Hodgman: the part of ``poly`` with normal . p <= offset."""
    out = []
    pts = list(np.asarray(poly, dtype=float).reshape(-1, 2))
    if not pts:
        return np.zeros((0, 2))
    prev = pts[-1]
    prev_in = float(normal @ prev) <= offset
    for cur in pts:
        cur_in = float(normal @ cur) <= offset
        if cur_in != prev_in:
            denom = float(normal @ (cur - prev))
            t = (offset - float(normal @ prev)) / denom if abs(denom) > 1e-12 else 0.0
            out.append(prev + t * (cur - prev))
        if cur_in:
            out.append(cur)
        prev, prev_in = cur, cur_in
    return np.array(out).reshape(-1, 2)


def _strip_extent(poly, centre_xy, jaw_unit, half_strip):
    """Extent along ``jaw_unit`` of the polygon inside the strip |(p - centre) . n| <= half_strip, n = jaw turned +90
    deg (the pad length along R0 of a top-down hand); None when the strip misses the polygon."""
    n = np.array([-jaw_unit[1], jaw_unit[0]])
    c = float(n @ centre_xy)
    clipped = _clip_halfplane(poly, n, c + half_strip)
    clipped = _clip_halfplane(clipped, -n, -(c - half_strip))
    if len(clipped) < 2:
        return None
    proj = clipped @ jaw_unit
    return float(proj.max() - proj.min())


def _fit_circle(points):
    """Algebraic (Kasa) circle fit; (centre (2,), radius, rms residual) or None."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < D11_MIN_POINTS:
        return None
    A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))])
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    centre = sol[:2]
    r2 = float(sol[2] + centre @ centre)
    if r2 <= 0.0:
        return None
    radius = math.sqrt(r2)
    residual = np.linalg.norm(p - centre, axis=1) - radius
    return centre, radius, float(np.sqrt(np.mean(residual ** 2)))


def _mask_coherence(mask, uv, half_px):
    """Structure tensor of the region mask (smoothed by a MARKER_RADIUS_PX Gaussian, so label jaggies of a few px do
    not count) in a (2 half + 1)^2 patch at uv: (unit image axis along the region's sides (du, dv), coherence in
    [0, 1]); a region whose sides run straight and parallel through the patch has coherence ~1, a round one ~0."""
    h, w = mask.shape
    u0, v0 = int(round(float(uv[0]))), int(round(float(uv[1])))
    r = int(half_px)
    y0, y1, x0, x1 = max(v0 - r, 0), min(v0 + r + 1, h), max(u0 - r, 0), min(u0 + r + 1, w)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None, 0.0
    patch = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), MARKER_RADIUS_PX)[y0:y1, x0:x1]
    gy, gx = np.gradient(patch)
    tensor = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]], dtype=float)
    values, vectors = np.linalg.eigh(tensor)
    total = float(values.sum())
    if total <= 1e-12:
        return None, 0.0
    coherence = float((values[1] - values[0]) / total)
    across = vectors[:, 1]                                   # dominant gradient: across the sides
    return np.array([-across[1], across[0]]), coherence


def _axis_angle(a, b) -> float:
    """Angle (rad, in [0, pi/2]) between two line directions."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    c = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return math.acos(min(1.0, c))


def _orientation(result) -> dict | None:
    """A G-7 orientation kernel's answer (geometry ``outline_orientation`` / ``edge_orientation``: a mapping with
    "axis", "share", "second" and, for the outline, "two_sided"; or a tuple (axis, share, second[, two_sided])) as
    {"axis" (canonical unit world-xy), "share", "second" (None when not given), "two_sided" (None when not given)};
    None when it found no orientation."""
    if result is None:
        return None
    if isinstance(result, Mapping):
        axis, share = result.get("axis"), result.get("share")
        second, two_sided = result.get("second"), result.get("two_sided")
    elif hasattr(result, "axis") and hasattr(result, "share"):
        axis, share = result.axis, result.share
        second, two_sided = getattr(result, "second", None), getattr(result, "two_sided", None)
    else:
        items = tuple(result)
        if len(items) < 3:
            return None
        axis, share, second = items[:3]
        two_sided = items[3] if len(items) > 3 else None
    if axis is None or share is None:
        return None
    a = np.asarray(axis, dtype=float).reshape(-1)[:2]
    if len(a) < 2 or not np.isfinite(a).all() or float(np.linalg.norm(a)) <= 1e-12 or not math.isfinite(float(share)):
        return None
    return {"axis": _canonical_axis(a), "share": float(share),
            "second": None if second is None or not math.isfinite(float(second)) else float(second),
            "two_sided": None if two_sided is None else bool(two_sided)}


def _image_axis_world(axis_img, pose) -> np.ndarray:
    """A line direction (du, dv) of the upright wrist image as a unit world-xy axis (u along R1, v along -R0)."""
    a = np.asarray(axis_img, dtype=float)
    return _canonical_axis([float(a[0] * pose.R[0, 1] - a[1] * pose.R[0, 0]),
                            float(a[0] * pose.R[1, 1] - a[1] * pose.R[1, 0])])


def _poly_distance(poly, point) -> float:
    """Distance (m) from a world-xy point to a polygon: 0 inside, else to the nearest edge."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    p = np.asarray(point, dtype=float).reshape(-1)[:2]
    if not len(poly):
        return float("inf")
    if len(poly) >= 3 and bool(np.asarray(poly_contains(poly, p[None, :])).reshape(-1)[0]):
        return 0.0
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((p[None, :] - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p[None, :], axis=1)))


def _iou(a, b) -> float:
    union = int(np.count_nonzero(a | b))
    return 0.0 if union == 0 else float(np.count_nonzero(a & b)) / union


def _centroid_px(mask):
    vs, us = np.nonzero(mask)
    if not len(us):
        return None
    return np.array([float(us.mean()), float(vs.mean())])


def _disk_kernel(radius_px: float):
    r = max(int(math.ceil(float(radius_px))), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_boundaries(labels):
    """Assign watershed boundary pixels (-1) to a 4-neighbour's label."""
    out = labels.copy()
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        hole = out == -1
        if not hole.any():
            break
        cand = _shift(out, dy, dx, -1)
        take = hole & (cand >= 1)
        out[take] = cand[take]
    return out


# ------------------------------------------------------------------------------------------------- scene model
class SceneModel:
    """Spec §4.5 / §4.7 scene state (contract §C6). One instance per policy episode.

    ``vision`` (optional, keyword): an object with the geometry functions the scene uses (``to_lab255``,
    ``structure_axis``, ``refine_region_mask``, ``reacquire``) and ``RELIEF_OFFSET_M``; None = the geometry module
    (imported lazily). ``step`` is the policy step of the last ``on_view`` (counted per call; the policy may pass it)."""

    def __init__(self, robot: RobotModel, *, vision=None):
        self.robot = robot
        self._vision = vision
        self.hf: HeightField | None = None
        self.parts: dict[str, PartRecord] = {}
        self.retired: dict[str, PartRecord] = {}
        self.keepouts: dict[str, KeepOut] = {}
        self.cal = None
        self.camera: WristCamera | None = None
        self.projector = None
        self.held: HeldFrame | None = None
        self.planes: list[dict] = []
        self.baselines: dict = {}
        self.step = 0
        self.view: StepView | None = None
        self.mask_log: dict = {}
        self.last_registration: dict = {}
        self.last_external_lift: dict | None = None
        self.moved: dict = {}
        self._locations: list = []
        self._registration_thresholds: dict = {}
        self._extents: dict = {}
        self._held_local: dict = {}
        self._count = 0
        self._recent: deque = deque(maxlen=RECENT_VIEW_COUNT)
        self._keyframes: list = []
        self._surveys: list = []
        self._survey_info: list = []              # per absorbed survey: {"pose0", "dense_centre_uv", "dense"}
        # every sample ever absorbed (registration markers); "dense": from a probe's dense lattice (S-5); "marker":
        # a registration marker only, not stamped (withdrawn by DSW or outside a refine_pid target, RC-9b);
        # "alive": False once its cell was vacated with a lifted object [C-36]; "survey": index into _survey_info
        self._samples = {"xyz": np.zeros((0, 3)), "sigma": np.zeros(0), "source": [], "step": np.zeros(0, int),
                         "dense": np.zeros(0, bool), "marker": np.zeros(0, bool), "alive": np.zeros(0, bool),
                         "withdrawn": np.zeros(0, bool), "survey": np.zeros(0, int)}
        self._plane_member = None
        self._fill_parent = None
        self._image_cache: dict = {}
        self._band_cache = None                   # (seed, band) of the register call in progress (S-2 / S-3)
        self.last_dsw: dict = {}                  # pid -> the S-5 withdrawal log of its last point-seeded registration
        self.part_fill_log: dict = {}             # pid -> the S-6 part-band fill attribution log
        self.part_ring_log: dict = {}             # pid -> the S-17 own-band ring log (remedy 1)
        self.last_reanchor: dict | None = None    # the last reanchor_from result (rulings item 4)
        # decision 2 S-12: pid -> the view memory since the last part_moved(pid): {"axes": [(unit axis, outline
        # share, step)], "conflicted", "widths" (the last non-collapsed width set or None), "width_step"}
        self._views: dict = {}
        # grip-part lens BS-1: (P, Q) for every plan step whose grip lists P with obj Q != P (merge_plan); pid -> "area" /
        # "point" / "external", the seed kind of the part's last successful registration; pid -> (record, extents)
        # before its last area-seeded registration (what a grip part that yields its object's region returns to)
        self._grip_pairs: set = set()
        self._record_kind: dict = {}
        self._area_prior: dict = {}
        self.region_conflicts: list = []          # every BS-1 refusal / yield (JSON-safe)
        # gap round X-1 (judge decision §2.2): one disturbance epoch per part, raised for every part of an object by
        # ``disturb`` (a contact-class event naming any part of it); pid -> int (0 until the first disturbance) and
        # the disturbance log. HS-H1: pid -> the largest-area whole registration of the current epoch ("body").
        # MC-R2a: pid -> the last non-collapsed point registration ({"record", "epoch", "step"}). MC-R3: pid -> the
        # first record with a 3-D point (the start-view record) and pid -> the epoch of the committed record
        self.epoch: dict = {}
        self.disturbances: list = []
        self._body: dict = {}
        self._whole: dict = {}
        self._start_record: dict = {}
        self._record_epoch: dict = {}
        self._fill_parent_z = None                # SAL-R1: the parent's height each D7b fill was derived from
        self.last_fill_consistency: dict = {}     # SAL-R1: the last stale-fill pass {"vacated", "parents"}
        self.last_seed_band_fallback: dict | None = None  # MC-R1: the last fallback attempt (JSON-safe log)

    # ------------------------------------------------------------------------------------------- collaborators
    @property
    def vision(self):
        if self._vision is None:
            self._vision = _geometry_module()
        return self._vision

    def _require(self):
        if self.hf is None or self.camera is None:
            raise RuntimeError("SceneModel.start(cal, view) must run first")
        return self.hf, self.camera

    # -------------------------------------------------------------------------------------------- start / views
    def start(self, cal, view: StepView) -> dict:
        """HeightField around the start grip; absorb the Calibrate+ survey; fit planes; fill unknown cells (D7);
        the "start" baseline. Returns the hf_summary block of the calibration event."""
        self.cal = cal
        camera = cal.camera.wrist() if hasattr(cal.camera, "wrist") else cal.camera
        if not isinstance(camera, WristCamera):
            raise ValueError("cal.camera must be a CameraModel (wrist()) or a WristCamera")
        self.camera = camera
        self.projector = getattr(cal, "projector", None)
        anchor_uv = getattr(self.projector, "anchor_uv", None)
        anchor_grip = getattr(self.projector, "anchor_grip", None)
        if anchor_uv is not None and anchor_grip is not None:
            # the calibration anchor is a kept reading too: a region near it re-anchors back after a reanchor_from
            self._locations.append({"step": -1, "uv": np.asarray(anchor_uv, dtype=float),
                                    "grip": np.asarray(anchor_grip, dtype=float),
                                    "sigma": float(getattr(self.projector, "sigma_px", 0.0) or 0.0),
                                    "source": "calibration"})
        self.hf = HeightField.empty(np.asarray(view.position[:2], dtype=float))
        self.hf.unconfirmed = np.zeros(self.hf.shape, dtype=bool)      # BS-2 export (``_refresh_unconfirmed``)
        self._plane_member = np.full(self.hf.shape, -1, dtype=np.int16)
        self._fill_parent = np.full(self.hf.shape + (2,), -1, dtype=np.int32)
        self._fill_parent_z = np.full(self.hf.shape, np.nan)
        if self.view is not view:                 # the policy's on_view of this act already counted it
            self._remember(view)
        changed = 0
        survey = getattr(cal, "survey", None)
        if survey is not None:
            changed = self.absorb_survey(survey, view)
        self.planes = self._fit_planes(view)
        support = self._support_plane_by_parallax(view)
        if support is not None:
            self._add_support_plane(support)
        fill = self.fill_unknown(view)
        self.capture_baseline("start", view)
        return jsonable({"hf_summary": self.hf.summary(), "cells_changed": changed, "fill": fill,
                         "planes": [self._plane_summary(p) for p in self.planes]})

    def on_view(self, view: StepView, step: int | None = None) -> dict:
        """Cache the view (recent ring and pose keyframes); returns parts_uv {pid: {"wrist", "external"}} [C-18]."""
        self._remember(view, step)
        out = {}
        if self.camera is None:
            return out
        pose = HandPose.from_view(view)
        for pid, part in self.parts.items():
            xyz = self._current_xyz(pid, pose)
            if xyz is None:
                continue
            uv, _ = _project(self.camera, pose, xyz)
            wrist = uv[0] if _inside(uv, (self.camera.size_px, self.camera.size_px))[0] else None
            external = None
            if self.projector is not None:
                try:
                    euv, _ = self.projector.project(np.asarray(xyz, dtype=float))
                    external = [float(euv[0]), float(euv[1])]
                except (ValueError, TypeError, AttributeError):
                    external = None
            out[pid] = {"wrist": None if wrist is None else [float(wrist[0]), float(wrist[1])], "external": external}
        return out

    def _remember(self, view, step=None):
        if step is not None:
            self.step = int(step)
        else:
            self.step = self._count
        self._count = self.step + 1
        self.view = view
        self._recent.append((self.step, view))
        pose = HandPose.from_view(view)
        self._keyframes = [(s, v) for s, v in self._keyframes if self.step - s <= KEYFRAME_MAX_AGE_STEPS]
        far = all(float(np.linalg.norm(HandPose.from_view(v).grip - pose.grip)) >= KEYFRAME_SPACING_M
                  for _, v in self._keyframes)
        if far:
            self._keyframes.append((self.step, view))
            if len(self._keyframes) > KEYFRAME_MAX_COUNT:
                self._keyframes = self._keyframes[-KEYFRAME_MAX_COUNT:]

    def invalidate_views(self) -> None:
        """Forget cached views (the scene changed: an object was lifted or a part turned)."""
        keep = self._recent[-1] if self._recent else None
        self._recent.clear()
        self._keyframes = []
        if keep is not None:
            self._recent.append(keep)

    def capture_baseline(self, key: str, view: StepView) -> dict:
        """baselines[key] = {"wrist", "external" (upright rgb copies), "pose" (state8)}."""
        entry = {"wrist": np.array(view.wrist_rgb_upright), "external": np.array(view.external_rgb_upright),
                 "pose": tuple(float(v) for v in view.state)}
        self.baselines[key] = entry
        return entry

    # ------------------------------------------------------------------------------------------ image caches
    def _images(self, view):
        key = id(view)
        cached = self._image_cache.get(key)
        if cached is not None and cached["view"] is view:
            return cached
        rgb = np.ascontiguousarray(view.wrist_rgb_upright)
        lab = np.asarray(self.vision.to_lab255(rgb), dtype=np.float32)
        grad = np.zeros(lab.shape[:2], dtype=np.float32)
        for channel in range(lab.shape[2]):
            gx = cv2.Sobel(lab[..., channel], cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(lab[..., channel], cv2.CV_32F, 0, 1, ksize=3)
            grad = np.maximum(grad, np.hypot(gx, gy))
        grad8 = np.clip(grad, 0, 255).astype(np.uint8)
        local = cv2.medianBlur(grad8, D9_MEDIAN_WINDOW_PX).astype(np.float32)
        floor = float(np.median(grad))
        edge = grad > D9_GRADIENT_RATIO * np.maximum(local, floor)
        _, components = cv2.connectedComponents((~edge).astype(np.uint8), connectivity=4)
        cached = {"view": view, "rgb": rgb, "lab": lab, "grad": grad, "edge": edge, "components": components,
                  "gray": np.asarray(view.wrist_gray_upright, dtype=np.float32), "robot": self._robot_mask(rgb.shape)}
        if len(self._image_cache) >= PARALLAX_MAX_FRAMES + 2:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[key] = cached
        return cached

    def _robot_mask(self, shape):
        """The robot's own wrist pixels: the measured fingertip band and every row below its lowest row."""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        band = getattr(self.cal, "fingertip_band", None) if self.cal is not None else None
        if band is None:
            return mask
        band = np.asarray(band, dtype=bool)
        if band.shape != (h, w):
            return mask
        mask |= band
        rows = np.flatnonzero(band.any(axis=1))
        if len(rows):
            mask[int(rows.max()) + 1:, :] = True
        return mask

    def _pixel_sigma(self) -> float:
        sigma = 0.0
        if self.cal is not None:
            px = getattr(self.cal, "sigma_px", None) or {}
            sigma = float(px.get("wrist", 0.0) or 0.0)
            sigma = max(sigma, float(getattr(self.cal, "sigma_static_grey", 0.0) or 0.0))
        return max(sigma, GREY_QUANTIZATION_SIGMA)

    # ------------------------------------------------------------------------------------------- fusion
    def absorb_survey(self, survey, view: StepView, *, exclude_mask=None, refine_pid: str | None = None) -> int:
        """Stamp the survey's depth samples (skipping samples whose pixel lies in the robot mask or ``exclude_mask``,
        e.g. the held-object mask) and keep them as registration markers; returns the cells changed. Samples of the
        survey's dense lattice (feature key "dense:<k>", decision §2.3 G-1) are flagged for the S-5 withdrawal.
        ``refine_pid`` (RC-9b, decision §2.2 S-8): a refinement survey of that part stamps only the samples inside the
        part's region (its 3-D point without one) dilated by REFINE_SIGMA_K sigma_xy; the others stay registration
        markers, so a worse refinement view never rewrites another part's cells."""
        hf, _ = self._require()
        if any(s is survey for s in self._surveys):
            return 0
        self._surveys.append(survey)
        samples = list(getattr(survey, "samples", ()) or ())
        robot_mask = self._robot_mask((self.camera.size_px, self.camera.size_px))
        excluded = None if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
        keep = []
        for sample in samples:
            if sample.uv is not None:
                u, v = int(round(sample.uv[0])), int(round(sample.uv[1]))
                for mask in (robot_mask, excluded):
                    if mask is not None and 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                        break
                else:
                    keep.append(sample)
                continue
            keep.append(sample)
        features = list(getattr(survey, "features", ()) or ())
        dense_features = [f for f in features if _is_dense_feature(f)]
        dense_fids = {f.fid for f in dense_features}
        dense = np.array([s.feature_id is not None and s.feature_id in dense_fids for s in keep], dtype=bool)
        stamp = np.ones(len(keep), dtype=bool)
        refine = None
        if refine_pid is not None:
            area = self._refine_area(refine_pid)
            xy = np.array([s.xyz[:2] for s in keep], dtype=float).reshape(-1, 2)
            stamp = poly_contains(area, xy) if area is not None and len(xy) else np.zeros(len(keep), dtype=bool)
            refine = {"pid": refine_pid, "area": area is not None, "stamped": int(stamp.sum()),
                      "markers_only": int((~stamp).sum())}
        pose0 = getattr(survey, "pose0", None)
        pose0 = tuple(float(v) for v in pose0) if pose0 is not None and len(pose0) == 8 else None
        centre = None
        if dense_features:
            centre = np.median(np.array([np.asarray(f.uv0, dtype=float)[:2] for f in dense_features]), axis=0)
        self._survey_info.append({"pose0": pose0, "dense_centre_uv": centre, "dense": int(dense.sum()),
                                  "refine": refine})
        stamped = [s for s, k in zip(keep, stamp) if k]
        changed = hf.stamp_samples(stamped, self.step)
        self._add_samples(keep, dense=dense, marker=~stamp, survey=len(self._survey_info) - 1)
        for ij in {hf.index(s.xyz[:2]) for s in stamped} - {None}:
            self._plane_member[ij] = -1 if hf.source[ij] != _PLANE_CODE else self._plane_member[ij]
            self._fill_parent[ij] = -1
        self._refresh_unconfirmed()
        return int(changed)

    def _refine_area(self, pid):
        """RC-9b: the world-xy polygon a refinement survey of ``pid`` may stamp: the part's region (else its point)
        dilated by REFINE_SIGMA_K x its horizontal sigma (at least half a cell around a bare point); None without a
        3-D point."""
        rec = self.parts.get(pid)
        if rec is None or rec.xyz is None:
            return None
        cov = np.zeros((3, 3)) if rec.cov is None else np.asarray(rec.cov, dtype=float)
        sigma_xy = math.sqrt(max(float(np.linalg.eigvalsh(cov[:2, :2]).max()), 0.0))
        margin = REFINE_SIGMA_K * sigma_xy
        if rec.region_xy is not None and len(rec.region_xy) >= 3:
            return poly_inflate(np.asarray(rec.region_xy, dtype=float), margin)
        half = max(margin, self.hf.cell_m / 2.0)
        return rect_polygon(np.asarray(rec.xyz, dtype=float)[:2], (1.0, 0.0), half, half)

    def _add_samples(self, samples, *, dense=None, marker=None, survey=None):
        if not samples:
            return
        n = len(samples)
        s = self._samples
        s["xyz"] = np.vstack([s["xyz"], np.array([d.xyz for d in samples], dtype=float)])
        s["sigma"] = np.concatenate([s["sigma"], np.array([d.sigma_z for d in samples], dtype=float)])
        s["source"] = list(s["source"]) + [d.source for d in samples]
        s["step"] = np.concatenate([s["step"], np.array([d.step for d in samples], int)])
        s["dense"] = np.concatenate([s["dense"], np.zeros(n, bool) if dense is None else np.asarray(dense, bool)])
        s["marker"] = np.concatenate([s["marker"], np.zeros(n, bool) if marker is None else np.asarray(marker, bool)])
        s["alive"] = np.concatenate([s["alive"], np.ones(n, bool)])
        s["withdrawn"] = np.concatenate([s["withdrawn"], np.zeros(n, bool)])
        s["survey"] = np.concatenate([s["survey"], np.full(n, -1 if survey is None else int(survey), int)])

    def _sample_cells(self, xyz=None):
        """(iy, ix, inside) HeightField indices of the samples (all of them by default)."""
        hf = self.hf
        xyz = self._samples["xyz"] if xyz is None else np.asarray(xyz, dtype=float).reshape(-1, 3)
        ix = np.floor((xyz[:, 0] - hf.origin_xy[0]) / hf.cell_m + 0.5).astype(int)
        iy = np.floor((xyz[:, 1] - hf.origin_xy[1]) / hf.cell_m + 0.5).astype(int)
        ny, nx = hf.shape
        return iy, ix, (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def _live_mask(self) -> np.ndarray:
        """bool per sample: alive and agreeing with its HeightField cell (D8), so stale samples of moved or vacated
        objects are ignored; a marker-only sample (withdrawn by S-5 or outside a refine_pid target) is live where its
        cell is not measured or agrees with it."""
        hf = self.hf
        xyz, sigma = self._samples["xyz"], self._samples["sigma"]
        ok = np.zeros(len(xyz), dtype=bool)
        if not len(xyz):
            return ok
        iy, ix, inside = self._sample_cells()
        ii, jj = iy[inside], ix[inside]
        cell_z, cell_s = hf.z[ii, jj], hf.sigma[ii, jj]
        measured = hf.measured[ii, jj]
        tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(sigma[inside], np.nan_to_num(cell_s)))
        with np.errstate(invalid="ignore"):
            agree = measured & (np.abs(xyz[inside, 2] - cell_z) <= tol)
        ok[inside] = self._samples["alive"][inside] & (agree | (self._samples["marker"][inside] & ~measured))
        return ok

    def _live_samples(self):
        """(xyz, sigma, sources) of the live samples (``_live_mask``)."""
        xyz, sigma = self._samples["xyz"], self._samples["sigma"]
        if not len(xyz):
            return xyz, sigma, []
        ok = self._live_mask()
        sources = [s for s, k in zip(self._samples["source"], ok) if k]
        return xyz[ok], sigma[ok], sources

    def _drop_stale_fills(self) -> int:
        """SAL-R1 (gap round, judge decision §2.2): a D7b obstacle fill lives only while its parent cell
        (``_fill_parent``) holds the value it was derived from (``_fill_parent_z``). A parent that was vacated,
        re-stamped (another height) or withdrawn (S-5 DSW, F2, RC-9b refine, [C-36] held vacate, HS-H1 vacate) leaves
        its fills stale: every such fill is vacated FIELD-WIDE (not only in the current view); ``fill_unknown``
        re-derives it from the remaining measured / D7a cells at the next view where it is visible. Parents are
        measured or D7a cells only (a fill never parents a fill). No number (exact equality of the stored height).
        Runs before every BS-2 refresh, i.e. after every HeightField write of the scene; returns the fills
        vacated."""
        hf = self.hf
        if hf is None or self._fill_parent is None:
            return 0
        if self._fill_parent_z is None or self._fill_parent_z.shape != hf.shape:
            self._fill_parent_z = np.full(hf.shape, np.nan)
        py, px = self._fill_parent[..., 0], self._fill_parent[..., 1]
        fills = (hf.source == _OBSTACLE_CODE) & (py >= 0) & (px >= 0)
        if not fills.any():
            return 0
        ppy, ppx = py[fills], px[fills]
        parent_ok = hf.measured[ppy, ppx] | (hf.source[ppy, ppx] == _PLANE_CODE)
        z_now, z_then = hf.z[ppy, ppx], self._fill_parent_z[fills]
        with np.errstate(invalid="ignore"):
            same = np.isfinite(z_now) & np.isfinite(z_then) & (np.abs(z_now - z_then) <= 1e-9)
        stale_idx = np.argwhere(fills)[~(parent_ok & same)]
        if not len(stale_idx):
            return 0
        hf.vacate(stale_idx, self.step)
        parents = {(int(a), int(b)) for a, b in np.stack([ppy, ppx], axis=-1)[~(parent_ok & same)]}
        self._fill_parent[stale_idx[:, 0], stale_idx[:, 1]] = -1
        self._fill_parent_z[stale_idx[:, 0], stale_idx[:, 1]] = np.nan
        self._plane_member[stale_idx[:, 0], stale_idx[:, 1]] = -1
        self.last_fill_consistency = {"step": int(self.step), "vacated": int(len(stale_idx)),
                                      "parents": int(len(parents))}
        return int(len(stale_idx))

    def _refresh_unconfirmed(self) -> int:
        """BS-2 (grip-part lens), exported as ``hf.unconfirmed`` (bool per cell, planner BP-2): a MEASURED cell (dense /
        kappa / loom; a touch is never unconfirmed) that stands above EVERY measured or D7a plane 8-neighbour by more
        than the D8 tolerance max(D8_FLOOR_M, D8_SIGMA_K hypot(sigma_c, sigma_n)), with at least
        UNCONFIRMED_MIN_NEIGHBOURS such neighbours, is unconfirmed until a later sample within D8 of its height
        confirms it: agreeing live samples from >= UNCONFIRMED_MIN_GROUPS measurement groups (absorbed surveys; a
        sample of no survey is a group of its own step), so one probe's dense lattice never confirms itself. Such
        cells are never deleted (about half of them are real one-cell structures). SAL-R4 (gap round): D7b obstacle
        fills are no neighbours (a spike's own fills surround it at its height + 3 sigma and would hide it). Runs
        the SAL-R1 stale-fill pass first; recomputed after every HeightField write of the scene; returns the number
        of unconfirmed cells."""
        hf = self.hf
        if hf is None:
            return 0
        self._drop_stale_fills()
        z = hf.z
        sig = np.nan_to_num(hf.sigma, nan=0.0)
        known = (hf.measured | (hf.source == _PLANE_CODE)) & np.isfinite(z)
        candidate = hf.measured & known & (hf.source != _TOUCH_CODE)
        above = candidate.copy()
        count = np.zeros(hf.shape, dtype=int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                kn = _shift(known, dy, dx, False)
                zn = _shift(np.where(known, z, np.nan), dy, dx, np.nan)
                sn = _shift(sig, dy, dx, 0.0)
                tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(sig, sn))
                with np.errstate(invalid="ignore"):
                    higher = (z - zn) > tol
                count += kn
                above &= ~kn | higher
        spike = above & (count >= UNCONFIRMED_MIN_NEIGHBOURS)
        unconfirmed = spike & ~self._confirmed(spike)
        hf.unconfirmed = unconfirmed
        return int(unconfirmed.sum())

    def _confirmed(self, cells_mask) -> np.ndarray:
        """BS-2: the cells of ``cells_mask`` whose current height is backed by agreeing (D8) live, not withdrawn
        samples of >= UNCONFIRMED_MIN_GROUPS measurement groups."""
        hf = self.hf
        out = np.zeros(hf.shape, dtype=bool)
        if not cells_mask.any() or not len(self._samples["xyz"]):
            return out
        iy, ix, inside = self._sample_cells()
        usable = inside & self._samples["alive"] & ~self._samples["withdrawn"]
        lookup = np.zeros(len(usable), dtype=bool)
        lookup[usable] = cells_mask[iy[usable], ix[usable]]
        idx = np.flatnonzero(lookup)
        if not len(idx):
            return out
        zs, ss = self._samples["xyz"][idx, 2], self._samples["sigma"][idx]
        cy, cx = iy[idx], ix[idx]
        cz, cs = hf.z[cy, cx], np.nan_to_num(hf.sigma[cy, cx], nan=0.0)
        agree = np.abs(zs - cz) <= np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(ss, cs))
        groups = {}
        surveys, steps = self._samples["survey"][idx], self._samples["step"][idx]
        for k in np.flatnonzero(agree):
            group = ("survey", int(surveys[k])) if int(surveys[k]) >= 0 else ("step", int(steps[k]))
            groups.setdefault((int(cy[k]), int(cx[k])), set()).add(group)
        for (a, b), found in groups.items():
            if len(found) >= UNCONFIRMED_MIN_GROUPS:
                out[a, b] = True
        return out

    def unconfirmed_cells(self) -> np.ndarray:
        """BS-2: the (n, 2) ij of the cells ``hf.unconfirmed`` marks (single high cells awaiting confirmation)."""
        if self.hf is None:
            return np.zeros((0, 2), dtype=int)
        return np.argwhere(np.asarray(getattr(self.hf, "unconfirmed", np.zeros(self.hf.shape, bool)), dtype=bool))

    def stamp_touch(self, footprint_xy, surface_z: float, sigma: float, step: int) -> int:
        """D5: every cell of the touching footprint at the touched surface z (source touch); returns cells changed."""
        hf, _ = self._require()
        poly = np.asarray(footprint_xy, dtype=float).reshape(-1, 2)
        cells = hf.cells_in_polygon(poly)
        if not len(cells):
            ij = hf.index(poly.mean(axis=0))
            cells = np.zeros((0, 2), int) if ij is None else np.array([ij])
        changed = 0
        samples = []
        for iy, ix in cells:
            changed += bool(hf.stamp_ij((int(iy), int(ix)), surface_z, sigma, "touch", int(step)))
            self._plane_member[iy, ix] = -1
            self._fill_parent[iy, ix] = -1
            xy = hf.cell_xy((iy, ix))
            samples.append(DepthSample((float(xy[0]), float(xy[1]), float(surface_z)), float(sigma), hf.cell_m / 2.0,
                                       "touch", None, None, int(step), None))
        self._add_samples(samples)
        self._refresh_unconfirmed()
        return int(changed)

    def touch_surface(self, grip_z: float, sigma_touch: float) -> tuple[Meas, Threshold]:
        """D5 fingertip landing: surface z = grip z - t_f (R5) with sigma_touch (one stall window of realized
        motion); returns (Meas, the D5_fingertip_reach_m threshold)."""
        reach = self.robot.t_f_m
        threshold = th_value("D5_fingertip_reach_m", reach, "ROBOT", "R5 fingertip reach t_f beyond the grip",
                             t_f_m=reach)
        return Meas(float(grip_z) - reach, float(sigma_touch), "touch"), threshold

    # ------------------------------------------------------------------------------------------------ planes
    def _fit_planes(self, view) -> list[dict]:
        """Support planes from the measured cells (height modes with >= PLANE_MIN_CELLS cells) and their colour in
        the view (Lab of each member cell's pixel)."""
        hf = self.hf
        measured = hf.measured
        if int(measured.sum()) < PLANE_MIN_CELLS:
            return []
        z, s = hf.z[measured], hf.sigma[measured]
        cells = np.argwhere(measured)
        bands = _height_bands(z, s, np.ones(len(z)))
        images = self._images(view)
        pose = HandPose.from_view(view)
        planes = []
        for band in sorted(bands, key=lambda b: -b["n"]):
            members = cells[band["members"]]
            if len(members) < PLANE_MIN_CELLS or len(planes) >= PLANE_MAX_COUNT:
                continue
            xy = hf.origin_xy + members[:, ::-1] * hf.cell_m
            uv, _ = _project(self.camera, pose, np.column_stack([xy, hf.z[members[:, 0], members[:, 1]]]))
            ok = _inside(uv, images["lab"].shape)
            px = _pix(uv[ok])
            ok_px = ~images["robot"][px[:, 1], px[:, 0]] if len(px) else np.zeros(0, bool)
            px = px[ok_px]
            if len(px) < PLANE_MIN_CELLS:
                continue
            lab = images["lab"][px[:, 1], px[:, 0]]
            centre = np.median(lab, axis=0)
            spread = np.array([_robust_sigma(lab[:, k]) for k in range(lab.shape[1])])
            spread = np.maximum(spread, self._pixel_sigma())
            zs = hf.z[members[:, 0], members[:, 1]]
            sigma_z = max(_robust_sigma(zs), float(np.median(hf.sigma[members[:, 0], members[:, 1]])) / math.sqrt(
                len(zs)))
            index = len(planes)
            self._plane_member[members[:, 0], members[:, 1]] = index
            planes.append({"z": Meas(float(np.median(zs)), sigma_z, "plane"), "lab_mean": centre, "lab_sigma": spread,
                           "cells": int(len(members)), "seeds": np.zeros((0, 2), dtype=int), "source": "measured"})
        return planes

    @staticmethod
    def _plane_summary(plane):
        return {"z": plane["z"].to_dict(), "lab_mean": plane["lab_mean"], "lab_sigma": plane["lab_sigma"],
                "cells": plane["cells"], "seeds": int(len(plane.get("seeds", ()))),
                "source": plane.get("source", "measured")}

    def _support_plane_by_parallax(self, view):
        """The support plane when its texture is too weak for kappa features (spec §4.5: most support cells start
        unknown): the height whose plane homography best maps the cached probe views onto this view over the whole
        image (the median pixel cost; the support is most of the image), swept from the fingertips down
        SUPPORT_SWEEP_M (coarse on downsampled images, then fine). The pixels whose own best coarse height is that
        one and whose colour agrees with theirs seed it and give its colour; sigma = the robust spread of their own
        best heights in the fine sweep. None without probe views or when the cost minimum is shallow."""
        pose = HandPose.from_view(view)
        partners = self._parallax_partners(view, pose)
        if not partners:
            return None
        images = self._images(view)
        top = float(pose.grip[2]) - self.robot.t_f_m
        coarse = np.arange(top, top - SUPPORT_SWEEP_M, -SUPPORT_SWEEP_STEP_M)
        cost_c = self._sweep_cost(view, pose, partners, coarse, SUPPORT_DOWNSAMPLE, images)
        if cost_c is None:
            return None
        curve = np.nanmedian(cost_c.reshape(len(coarse), -1), axis=1)
        if not np.isfinite(curve).any():
            return None
        k = int(np.nanargmin(curve))
        median = float(np.nanmedian(curve))
        if not median > 0.0 or (median - float(curve[k])) / median < SUPPORT_MIN_CONTRAST:
            return None
        # support pixels: their own coarse minimum is the plane's (+-1 step; a weak texture has no per-pixel margin,
        # but textureless pixels scatter their minima over the whole sweep, so few land there by chance), then
        # colour-consistent with the bulk of those pixels (D7_COLOUR_SIGMA_K robust sigma, once)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            filled_c = np.where(np.isfinite(cost_c), cost_c, np.inf)
            own_k = np.argmin(filled_c, axis=0)
            low = (np.abs(own_k - k) <= 1) & np.isfinite(filled_c.min(axis=0))
        h, w = images["gray"].shape
        seeds = cv2.resize(low.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        seeds &= ~images["robot"]
        if int(seeds.sum()) < PLANE_MIN_CELLS:
            return None
        lab_all = images["lab"]
        centre = np.median(lab_all[seeds], axis=0)
        spread = np.array([_robust_sigma(lab_all[seeds][:, c]) for c in range(lab_all.shape[2])])
        spread = np.maximum(spread, self._pixel_sigma())
        seeds &= np.linalg.norm(lab_all - centre[None, None, :], axis=-1) <= D7_COLOUR_SIGMA_K * float(
            np.linalg.norm(spread))
        if int(seeds.sum()) < PLANE_MIN_CELLS:
            return None
        fine = np.arange(coarse[k] + SUPPORT_SWEEP_STEP_M, coarse[k] - SUPPORT_SWEEP_STEP_M, -SUPPORT_FINE_STEP_M)
        cost_f = self._sweep_cost(view, pose, partners, fine, 1, images)
        if cost_f is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            curve_f = np.nanmedian(np.where(seeds[None, :, :], cost_f, np.nan).reshape(len(fine), -1), axis=1)
            best_z = float(fine[int(np.nanargmin(curve_f))])
            per_best = np.nanargmin(np.where(np.isfinite(cost_f), cost_f, np.inf), axis=0)
        heights = fine[per_best]
        sigma = max(_robust_sigma(heights[seeds]), SUPPORT_FINE_STEP_M)
        lab = images["lab"][seeds]
        vs, us = np.nonzero(seeds)
        world = _backproject_z(self.camera, pose, np.column_stack([us, vs]).astype(float), best_z)
        cells = {self.hf.index(p[:2]) for p in world if np.isfinite(p).all()} - {None}
        return {"z": Meas(best_z, sigma, "plane"), "lab_mean": np.median(lab, axis=0),
                "lab_sigma": np.maximum(np.array([_robust_sigma(lab[:, c]) for c in range(lab.shape[1])]),
                                        self._pixel_sigma()),
                "cells": int(len(cells)), "seeds": np.array(sorted(cells), dtype=int).reshape(-1, 2),
                "source": "parallax", "contrast": (median - float(curve[k])) / median}

    def _sweep_cost(self, view, pose, partners, heights, scale, images, crop=None, per_pixel=False):
        """(len(heights), H', W') per-pixel photometric cost of horizontal planes at ``heights``: the window mean
        |grey difference| per partner, combined over the partners as the mean without the worst (with >= 3; as
        ``_parallax``). A partner counts at a pixel only when it is valid there at EVERY height, so all heights of a
        pixel are compared on the same partners (a height whose warp leaves one partner's image or meets the robot
        would otherwise be scored on fewer, different partners). ``scale`` downsamples the whole image; ``crop`` =
        (x0, y0, x1, y1) full-resolution pixels restricts a scale-1 sweep to that window."""
        g0 = images["gray"]
        h, w = g0.shape
        x0, y0, x1, y1 = (0, 0, w, h) if crop is None else (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
        if crop is not None:
            scale = 1
        hs, ws = (y1 - y0) // scale, (x1 - x0) // scale
        S = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        T = np.array([[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]])
        own_full = (~images["robot"]).astype(np.float32)
        if scale > 1:
            g0s = cv2.resize(g0, (ws, hs), interpolation=cv2.INTER_AREA)
            own = cv2.resize(own_full, (ws, hs), interpolation=cv2.INTER_AREA)
        else:
            g0s, own = g0[y0:y1, x0:x1], own_full[y0:y1, x0:x1]
        own = (own > 1.0 - 1e-6).astype(np.float32)
        corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])
        kernel = (PARALLAX_WINDOW_PX, PARALLAX_WINDOW_PX)
        per = np.full((len(partners), len(heights), hs, ws), np.nan, dtype=np.float32)
        raw = np.full((len(partners), len(heights), hs, ws), np.nan, dtype=np.float32) if per_pixel else None
        full = float(PARALLAX_WINDOW_PX * PARALLAX_WINDOW_PX)
        for i, other in enumerate(partners):
            op = HandPose.from_view(other)
            gk = np.asarray(other.wrist_gray_upright, dtype=np.float32)
            free_k = 1.0 - self._robot_mask(gk.shape).astype(np.float32)
            if scale > 1:
                gk = cv2.resize(gk, (ws, hs), interpolation=cv2.INTER_AREA)
                free_k = cv2.resize(free_k, (ws, hs), interpolation=cv2.INTER_AREA)
            for j, z in enumerate(heights):
                pts = _backproject_z(self.camera, pose, corners, z)
                if not np.isfinite(pts).all():
                    continue
                dst, _ = _project(self.camera, op, pts)
                if not np.isfinite(dst).all():
                    continue
                H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst.astype(np.float32))
                M = S @ H @ np.linalg.inv(S) if scale > 1 else H @ T
                flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                warped = cv2.warpPerspective(gk, M, (ws, hs), flags=flags, borderMode=cv2.BORDER_CONSTANT)
                valid = cv2.warpPerspective(free_k, M, (ws, hs), flags=flags, borderMode=cv2.BORDER_CONSTANT)
                valid = (valid > 1.0 - 1e-6).astype(np.float32) * own
                diff = np.abs(g0s - warped)
                cost = cv2.boxFilter(diff * valid, -1, kernel, normalize=False)
                count = cv2.boxFilter(valid, -1, kernel, normalize=False)
                with np.errstate(invalid="ignore", divide="ignore"):
                    per[i, j] = np.where(count >= full / 2.0, cost / np.maximum(count, 1e-6), np.nan)
                if raw is not None:
                    raw[i, j] = np.where(valid > 0.0, diff, np.nan)
            bad = ~np.isfinite(per[i]).all(axis=0)
            per[i][:, bad] = np.nan
            if raw is not None:
                raw[i][:, bad | ~np.isfinite(raw[i]).all(axis=0)] = np.nan
        finite = np.isfinite(per)
        if not finite.any():
            return (None, None) if per_pixel else None
        total = np.where(finite, per, 0.0).sum(axis=0)
        used = finite.sum(axis=0)
        if len(partners) >= PARALLAX_DROP_WORST_MIN:
            worst = np.where(finite, per, -np.inf).max(axis=0)
            drop = used >= PARALLAX_DROP_WORST_MIN
            total = np.where(drop, total - np.where(np.isfinite(worst), worst, 0.0), total)
            used = np.where(drop, used - 1, used)
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = np.where(used > 0, total / np.maximum(used, 1), np.nan)
        if not per_pixel:
            return combined
        return combined, _nan_median_axis0(raw)

    def _add_support_plane(self, support):
        """The parallax support plane joins the planes (merged into a measured plane within the D8 tolerance) as
        planes[0], the base support."""
        z = support["z"]
        for index, plane in enumerate(self.planes):
            tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(plane["z"].sigma, z.sigma))
            if abs(plane["z"].value - z.value) <= tol:
                plane["seeds"] = np.vstack([plane.get("seeds", np.zeros((0, 2), int)), support["seeds"]])
                if index:
                    self.planes.insert(0, self.planes.pop(index))
                    self._renumber_planes(index)
                return
        self.planes.insert(0, support)
        self._renumber_planes(None)
        if len(self.planes) > PLANE_MAX_COUNT:
            self.planes = self.planes[:PLANE_MAX_COUNT]
            self._plane_member[self._plane_member >= PLANE_MAX_COUNT] = -1

    def _renumber_planes(self, moved):
        """Plane indices after an insertion at 0 (``moved`` = the old index of a plane moved to 0, or None)."""
        member = self._plane_member
        old = member.copy()
        if moved is None:
            member[old >= 0] = old[old >= 0] + 1
        else:
            member[(old >= 0) & (old < moved)] = old[(old >= 0) & (old < moved)] + 1
            member[old == moved] = 0

    def fill_unknown(self, view: StepView, *, around_xy=None, radius_m=None) -> dict:
        """D7 (spec §4.5): unknown and filled cells visible in the view (within radius_m of around_xy when given)
        are recomputed: D7a plane fill (colour within 3 sigma of a fitted plane, no D9 edge, within 2 cm of the
        plane's known cells, applied transitively), else D7b obstacle at max(known neighbour + 3 sigma) within 2 cm;
        fills that no longer qualify are vacated. Returns counts by kind and the D7 thresholds."""
        hf, camera = self._require()
        pose = HandPose.from_view(view)
        images = self._images(view)
        ny, nx = hf.shape
        gy, gx = np.mgrid[0:ny, 0:nx]
        cell_xy = np.stack([hf.origin_xy[0] + gx * hf.cell_m, hf.origin_xy[1] + gy * hf.cell_m], axis=-1)
        area = ~hf.measured
        if around_xy is not None and radius_m is not None:
            area &= np.linalg.norm(cell_xy - np.asarray(around_xy, dtype=float)[None, None, :], axis=-1) <= radius_m
        old_fills = np.isin(hf.source, _FILL_CODES) & area
        new_source = np.zeros(hf.shape, dtype=np.int8)
        new_z = np.full(hf.shape, np.nan)
        new_s = np.full(hf.shape, np.nan)
        new_member = np.full(hf.shape, -1, dtype=np.int16)
        r_plane = int(round(D7_PLANE_RADIUS_M / hf.cell_m))
        thresholds = []
        counts = {"plane": 0, "obstacle": 0, "vacated": 0, "plane_by_source": {}}
        for index, plane in enumerate(self.planes):
            z_p, s_p = plane["z"].value, plane["z"].sigma
            band = D7_COLOUR_SIGMA_K * float(np.linalg.norm(plane["lab_sigma"]))
            # D9 "no edge between them": pixel connectivity through edge-free pixels of the plane's colour
            pixel_ok = (np.linalg.norm(images["lab"] - plane["lab_mean"][None, None, :], axis=-1) <= band)
            pixel_ok &= ~images["edge"] & ~images["robot"]
            _, plane_components = cv2.connectedComponents(pixel_ok.astype(np.uint8), connectivity=4)
            comp = self._cell_components(images, pose, cell_xy, z_p, plane_components)
            passes = area & (comp > 0) & (new_source == 0)
            seed_mask = np.zeros(hf.shape, dtype=bool)
            seeds = np.asarray(plane.get("seeds", np.zeros((0, 2), int)), dtype=int).reshape(-1, 2)
            if len(seeds):
                seed_mask[seeds[:, 0], seeds[:, 1]] = True
            seed_mask &= ~hf.measured
            known = ((self._plane_member == index) & hf.measured) | (new_member == index) | seed_mask
            # a segment seeds the plane only when most of the measured cells seen inside it are the plane's (a noisy
            # sample of a lower surface beside the plane height does not make that surface the plane)
            own = self._cell_components(images, pose, cell_xy, np.where(hf.measured, hf.z, z_p), plane_components)
            seen = np.bincount(own[hf.measured & (own > 0)], minlength=int(plane_components.max()) + 1)
            mine = np.bincount(own[hf.measured & (own > 0) & (self._plane_member == index)],
                               minlength=len(seen))
            seeded = np.bincount(comp[seed_mask & (comp > 0)], minlength=len(seen))
            seen, mine = seen + seeded, mine + seeded
            eligible = np.flatnonzero(2 * mine > seen)
            known &= np.isin(comp, eligible)
            known_comp = np.where(known, comp, 0)
            filled = np.zeros(hf.shape, dtype=bool)
            offsets = _disk_offsets(r_plane)
            for _ in range(max(ny, nx)):
                reach = np.zeros(hf.shape, dtype=bool)
                for dy, dx in offsets:
                    neighbour = _shift(known_comp, dy, dx, 0)
                    reach |= (neighbour > 0) & (neighbour == comp)
                grow = reach & passes & ~filled & ~known
                if not grow.any():
                    break
                filled |= grow
                known_comp = np.where(grow, comp, known_comp)
                known |= grow
            filled |= seed_mask & area & (new_source == 0) & np.isin(comp, eligible)
            fill_z = z_p + D7_PLANE_SIGMA_K * s_p
            new_source[filled], new_z[filled], new_s[filled], new_member[filled] = _PLANE_CODE, fill_z, s_p, index
            counts["plane"] += int(filled.sum())
            source = plane.get("source", "measured")
            counts["plane_by_source"][source] = counts["plane_by_source"].get(source, 0) + int(filled.sum())
            thresholds.append(th_value("D7_colour_band", band, "ONLINE", "D7_COLOUR_SIGMA_K * |sigma_Lab of the plane|",
                                       plane=index, k=D7_COLOUR_SIGMA_K, plane_source=source))
            thresholds.append(th_value("D7_fill_height_m", fill_z, "ONLINE", "plane z + D7_PLANE_SIGMA_K * sigma_plane",
                                       plane=index, z=z_p, sigma=s_p, plane_source=source))
        # D7b obstacle fill from measured and plane-filled neighbours (never from other obstacle fills); the parent's
        # sigma is capped at D7_PARENT_SIGMA_CAP_M (RC-9a, decision §2.2 S-8: a parent with a 20 mm sigma would raise
        # a 6 cm phantom wall), so a fill = parent z + D7_OBSTACLE_SIGMA_K min(sigma_parent, cap), stored with that sigma
        base_known = hf.measured | (new_source == _PLANE_CODE)
        raw_sig = np.nan_to_num(np.where(hf.measured, hf.sigma, new_s))
        sig = np.minimum(raw_sig, D7_PARENT_SIGMA_CAP_M)
        upper = np.where(hf.measured, hf.z + D7_OBSTACLE_SIGMA_K * sig, -np.inf)
        upper = np.where(new_source == _PLANE_CODE, new_z + D7_OBSTACLE_SIGMA_K * sig, upper)
        best = np.full(hf.shape, -np.inf)
        best_s = np.full(hf.shape, np.nan)
        parent = np.full(hf.shape + (2,), -1, dtype=np.int32)
        for dy, dx in _disk_offsets(int(round(D7_OBSTACLE_RADIUS_M / hf.cell_m))):
            cand = _shift(upper, dy, dx, -np.inf)
            cand_s = _shift(sig, dy, dx, np.nan)
            better = cand > best
            best = np.where(better, cand, best)
            best_s = np.where(better, cand_s, best_s)
            parent[better] = np.stack([gy[better] + dy, gx[better] + dx], axis=-1)
        obstacle = area & (new_source == 0) & ~base_known & np.isfinite(best)
        obstacle &= self._cells_visible(images, pose, cell_xy, np.where(np.isfinite(best), best, np.nan))
        new_source[obstacle], new_z[obstacle], new_s[obstacle] = _OBSTACLE_CODE, best[obstacle], best_s[obstacle]
        counts["obstacle"] = int(obstacle.sum())
        filled_z = best[obstacle]
        thresholds.append(th_value("D7_obstacle_sigma_k", D7_OBSTACLE_SIGMA_K, "DD",
                                   "obstacle fill = max(known neighbour z + D7_OBSTACLE_SIGMA_K * sigma) within "
                                   "D7_OBSTACLE_RADIUS_M (the value is the sigma multiplier; the fill heights are in "
                                   "the inputs)", radius_m=D7_OBSTACLE_RADIUS_M, cells=int(obstacle.sum()),
                                   fill_z_min=float(filled_z.min()) if len(filled_z) else None,
                                   fill_z_max=float(filled_z.max()) if len(filled_z) else None))
        parents = parent[obstacle]
        capped = int((raw_sig[parents[:, 0], parents[:, 1]] > D7_PARENT_SIGMA_CAP_M).sum()) if len(parents) else 0
        counts["parent_sigma_capped"] = capped
        thresholds.append(th_value("D7_parent_sigma_cap_m", D7_PARENT_SIGMA_CAP_M, "DD",
                                   "obstacle fill parent sigma = min(sigma_parent, D7_PARENT_SIGMA_CAP_M = HF_CELL_M) "
                                   "(RC-9a, decision §2.2 S-8)", capped_fills=capped))
        # stamp: fills replace fills; a fill never replaces a measurement (HeightField precedence [C-33])
        for code, name in ((_PLANE_CODE, "plane"), (_OBSTACLE_CODE, "obstacle")):
            for iy, ix in np.argwhere(new_source == code):
                hf.stamp_ij((int(iy), int(ix)), float(new_z[iy, ix]), float(np.nan_to_num(new_s[iy, ix])), name,
                            self.step)
        stamped = new_source != 0
        self._plane_member[stamped] = new_member[stamped]
        self._fill_parent[stamped] = -1
        self._fill_parent[obstacle] = parent[obstacle]
        if self._fill_parent_z is None or self._fill_parent_z.shape != hf.shape:
            self._fill_parent_z = np.full(hf.shape, np.nan)
        self._fill_parent_z[stamped] = np.nan
        if obstacle.any():
            # SAL-R1: the height each fill was derived from (its parent's value after this pass's stamps)
            pp = parent[obstacle]
            self._fill_parent_z[obstacle] = hf.z[pp[:, 0], pp[:, 1]]
        # stale fills (visible in this view, no longer produced) are vacated: "fill_unknown recomputes"
        visible = self._cells_visible(images, pose, cell_xy, np.where(old_fills, hf.z, np.nan))
        stale = old_fills & ~stamped & visible
        counts["vacated"] = int(hf.vacate(np.argwhere(stale), self.step)) if stale.any() else 0
        self._plane_member[stale] = -1
        self._fill_parent[stale] = -1
        self._fill_parent_z[stale] = np.nan
        counts["unconfirmed"] = self._refresh_unconfirmed()
        return jsonable({**counts, "thresholds": thresholds_dicts(thresholds)})

    def _cell_pixels(self, images, pose, cell_xy, z):
        """Pixel (u, v) per cell centre at height z (scalar or per cell), and whether it is a usable pixel (inside
        the image, not the robot)."""
        ny, nx = cell_xy.shape[:2]
        zz = np.broadcast_to(np.asarray(z, dtype=float), (ny, nx))
        pts = np.column_stack([cell_xy.reshape(-1, 2), zz.reshape(-1)])
        uv, _ = _project(self.camera, pose, pts)
        ok = _inside(uv, images["lab"].shape)
        px = np.zeros((len(uv), 2), dtype=int)
        px[ok] = _pix(uv[ok])
        ok[ok] &= ~images["robot"][px[ok, 1], px[ok, 0]]
        return px.reshape(ny, nx, 2), ok.reshape(ny, nx)

    def _cell_components(self, images, pose, cell_xy, z, components):
        px, ok = self._cell_pixels(images, pose, cell_xy, z)
        comp = np.zeros(ok.shape, dtype=np.int32)
        comp[ok] = components[px[ok][:, 1], px[ok][:, 0]]
        return comp

    def _cells_visible(self, images, pose, cell_xy, z):
        z = np.asarray(z, dtype=float)
        finite = np.isfinite(z)
        _, ok = self._cell_pixels(images, pose, cell_xy, np.where(finite, z, 0.0))
        return ok & finite

    # -------------------------------------------------------------------------------------------- plan seeds
    def merge_plan(self, plan, roles: dict, view: StepView, quarter_turns: int) -> dict[str, PartRecord]:
        """§3.6 items 1 and 7: a part whose (id, text) is unchanged keeps its record (roles refreshed; an unregistered
        record gets the new plan cells); any other part gets a new record seeded from its w cell (cell centre of the
        k-turned wrist image -> unrotate -> upright uv, quantization 32 px) and x cell (external upright). ``view`` and
        ``quarter_turns`` are those of the plan / replan request whose images the cells refer to."""
        wrist_w = int(np.asarray(view.wrist_rgb_upright).shape[1])
        ext_w = int(np.asarray(view.external_rgb_upright).shape[1])
        quantization = coarse_cell_center(1, wrist_w)[0]          # half a coarse cell (64 px at 512 -> 32 px)
        ext_quantization = coarse_cell_center(1, ext_w)[0]
        pose = tuple(float(v) for v in view.state)
        out = {}
        new_parts = {}
        for part in plan.parts:
            seeds = []
            if part.w_cell:
                turned = coarse_cell_center(part.w_cell, wrist_w)
                uv = unrotate_pixel(turned, int(quarter_turns), wrist_w)
                seeds.append(Seed("wrist", uv, self.step, "plan_cell", quantization, pose))
            if part.x_cell:
                uv = coarse_cell_center(part.x_cell, ext_w)
                seeds.append(Seed("external", uv, self.step, "plan_cell", ext_quantization, pose))
            part_roles = frozenset(roles.get(part.pid, frozenset()))
            old = self.parts.get(part.pid)
            if old is not None and old.text == part.text:
                keep = old.status == "registered" or old.provisional
                old_seeds = old.seeds if keep else [s for s in old.seeds if s.source != "plan_cell"] + seeds
                record = dataclasses.replace(old, roles=part_roles, move_prior=part.move, seeds=list(old_seeds))
            else:
                if old is not None:
                    self.retired[part.pid] = old
                record = PartRecord(part.pid, part.text, part.move, part_roles, seeds=seeds)
                self._extents.pop(part.pid, None)
            new_parts[part.pid] = record
            out[part.pid] = record
        for pid in set(self.parts) - set(new_parts):
            self.retired[pid] = self.parts[pid]
        self.parts = new_parts
        # BS-1: which part grips which object in this plan (a grip part P of a step whose obj is Q != P)
        self._grip_pairs = {(str(grip), str(step.obj)) for step in getattr(plan, "steps", ()) or ()
                            for grip in (getattr(step, "grip", ()) or ()) if str(grip) != str(step.obj)}
        return out

    def _best_seed(self, record: PartRecord, camera="wrist") -> Seed | None:
        seeds = [s for s in record.seeds if s.camera == camera]
        if not seeds:
            return None
        return max(seeds, key=lambda s: (_SEED_PRIORITY.get(s.source, -1), s.step))

    def _seed_band(self, seed) -> dict | None:
        """S-2 / S-3 (decision §2.2): the top band of a wrist POINT seed: the live samples with sigma_z <= TOP_GATE_M
        that project within TOP_WINDOW_PX of ``seed.uv`` in the seed's pose, through ``_top_band``. Returns None for
        another seed, else {"rule" ("band" only when a coherent band exists), "n_window", "n_gated", the
        ``_top_band`` info} and, for a band, "top", "sigma_h" = hypot(stat_m, model_m), "members" (sample indices),
        "lowest_z" / "lowest_sigma" (the lowest member, S-7). Cached for the register call in progress."""
        if seed is None or seed.camera != "wrist" or seed.source not in _POINT_SEEDS or self.camera is None:
            return None
        if self._band_cache is not None and self._band_cache[0] is seed:
            return self._band_cache[1]
        out = {"rule": None, "n_window": 0, "n_gated": 0}
        idx = np.flatnonzero(self._live_mask())
        if not len(idx):
            return out
        xyz, sigma = self._samples["xyz"][idx], self._samples["sigma"][idx]
        pose = HandPose.from_state8(seed.pose)
        uv, depth = _project(self.camera, pose, xyz)
        with np.errstate(invalid="ignore"):
            near = (np.isfinite(uv).all(axis=1) & (depth > 0.0)
                    & (np.linalg.norm(uv - np.asarray(seed.uv, dtype=float)[None, :], axis=1) <= TOP_WINDOW_PX))
        gated = near & (sigma <= TOP_GATE_M)
        out.update(n_window=int(near.sum()), n_gated=int(gated.sum()))
        if int(gated.sum()) < TOP_BAND_K:
            return out
        top, members, info = _top_band(xyz[gated, 2], sigma[gated])
        out.update(info)
        if info["rule"] != "band":
            return out
        chosen = idx[np.flatnonzero(gated)[members]]
        lowest = int(chosen[int(np.argmin(self._samples["xyz"][chosen, 2]))])
        out.update(top=float(top), sigma_h=math.hypot(info["stat_m"], info["model_m"]), members=chosen,
                   lowest_z=float(self._samples["xyz"][lowest, 2]), lowest_sigma=float(self._samples["sigma"][lowest]))
        return out

    def estimate_seed_xyz(self, seed: Seed, view: StepView) -> tuple[np.ndarray, np.ndarray] | None:
        """3-D of a seed. A wrist POINT seed with a top band (``_seed_band``, decision §2.2 S-2) is lifted onto the
        band top, its depth sigma the honest sigma_h of S-3 (the segmentation's seed band is then the top layer, not
        a flank or a lower segment). Otherwise: the median depth of the survey samples within the seed's
        quantization (in the image of the seed's pose); else, for a seed taken at the view's pose, the top band of
        the samples in its D9 segment (a textureless surface measured only along its outline, ``_segment_seed_xyz``);
        else the first HeightField surface along the wrist ray. External seeds intersect the affine external ray with
        the HeightField. Returns (xyz, cov 3x3) or None."""
        hf, camera = self._require()
        if seed.camera == "external":
            return self._external_seed_xyz(seed)
        pose = HandPose.from_state8(seed.pose)
        q = float(seed.quantization_px)
        band = self._seed_band(seed)
        if band is not None and band.get("rule") == "band":
            point = _backproject_z(camera, pose, np.asarray(seed.uv, dtype=float), band["top"])[0]
            if np.isfinite(point).all():
                depth = float((point - pose.camera) @ pose.R[:, 2])
                return point, self._seed_cov(pose, seed.uv, depth, q, band["sigma_h"])
        xyz, sigma, _ = self._live_samples()
        if len(xyz):
            uv, z = _project(camera, pose, xyz)
            ok = _inside(uv, (camera.size_px, camera.size_px))
            near = ok & (np.linalg.norm(uv - np.asarray(seed.uv)[None, :], axis=1) <= q)
            if near.any():
                depth = float(np.median(z[near]))
                point = camera.backproject(pose, seed.uv, depth)
                sigma_depth = max(_robust_sigma(z[near]), float(np.median(sigma[near])))
                return point, self._seed_cov(pose, seed.uv, depth, q, sigma_depth)
            lifted = self._segment_seed_xyz(seed, view, pose, xyz, sigma, uv, ok)
            if lifted is not None:
                return lifted
        origin, direction = camera.ray(pose, seed.uv)
        step = hf.cell_m / 2.0
        reach = int(math.ceil(2.0 * _hf_extent(hf) / step))
        for k in range(1, reach):
            p = origin + direction * (k * step)
            z, s, known = hf.sample(p[:2])
            if known and p[2] <= z:
                depth = float((p - pose.camera) @ pose.R[:, 2])
                sigma_depth = float(s) / max(abs(float(direction[2])), 1e-6)
                return p, self._seed_cov(pose, seed.uv, depth, q, sigma_depth)
        return None

    def _segment_seed_xyz(self, seed, view, pose, xyz, sigma, uv, ok):
        """A seed on a textureless surface: the samples inside the seed's D9 segment (the edge-free image region at the
        seed, its boundary pixels included, within SEED_REGION_RADIUS_PX) give the height: the TOP_PERCENTILE band of
        their heights (the pointed surface is the top of what it points at); the seed pixel is lifted onto that
        height. Only for a seed taken at the view's pose."""
        if not _same_pose(pose, HandPose.from_view(view)):
            return None
        images = self._images(view)
        h, w = images["edge"].shape
        u0, v0 = int(round(seed.uv[0])), int(round(seed.uv[1]))
        if not (0 <= v0 < h and 0 <= u0 < w):
            return None
        label = images["components"][v0, u0]
        if label <= 0:
            return None
        region = images["components"] == label
        region = cv2.dilate(region.astype(np.uint8), _disk_kernel(MARKER_RADIUS_PX)) > 0
        disk = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(disk, (u0, v0), SEED_REGION_RADIUS_PX, 1, -1)
        region &= disk > 0
        px = np.zeros((len(uv), 2), dtype=int)
        px[ok] = _pix(uv[ok])
        inside = ok.copy()
        inside[ok] = region[px[ok, 1], px[ok, 0]]
        if not inside.any():
            return None
        z_top, sigma_depth, _ = _top_cluster(xyz[inside, 2], sigma[inside])
        point = _backproject_z(self.camera, pose, np.asarray(seed.uv, dtype=float), z_top)[0]
        if not np.isfinite(point).all():
            return None
        depth = float((point - pose.camera) @ pose.R[:, 2])
        return point, self._seed_cov(pose, seed.uv, depth, float(seed.quantization_px), sigma_depth)

    def _seed_cov(self, pose, uv, depth, q_px, sigma_depth):
        sigma_lat = (q_px / math.sqrt(3)) * depth / self.camera.f_px
        d = _rays(self.camera, pose, uv)[0]
        d = d / np.linalg.norm(d)
        cov = sigma_lat ** 2 * (np.outer(pose.R[:, 0], pose.R[:, 0]) + np.outer(pose.R[:, 1], pose.R[:, 1]))
        return cov + sigma_depth ** 2 * np.outer(d, d)

    def _external_seed_xyz(self, seed):
        """An external seed lifted to 3-D (contract §C6 estimate_seed_xyz). The affine external ray through the seed
        pixel (ExternalProjector: J and its viewing ray, the null vector of J) is marched down from the ceiling
        (``_lift_ceiling``) to the first known HeightField surface. The external camera sees the first surface along
        the ray, so the part is that surface, or it lies in UNKNOWN cells the ray crossed before it (a part the wrist
        never saw, e.g. a plan part with w = 0): the lift is then the middle of the ray span from the first unknown
        cell to the hit (to the support plane without a hit), with that span as a uniform sigma along the ray (length /
        sqrt(12)). Across the ray: the seed's quantization (q / sqrt(3)) and the projector's own sigma at the point
        (anchor sigma plus the affine distance term), both through the pseudo-inverse of J. Returns (xyz, cov 3x3) or
        None (no projector or support plane, or the ray meets nothing below the ceiling)."""
        projector = self.projector
        model = getattr(projector, "model", None)
        J = getattr(model, "J", None)
        if projector is None or J is None or self.hf is None:
            return None
        J = np.asarray(J, dtype=float)
        anchor_uv = np.asarray(getattr(projector, "anchor_uv"), dtype=float)
        anchor = np.asarray(getattr(projector, "anchor_grip"), dtype=float)
        hf = self.hf
        ray_dir = getattr(model, "ray_dir", None)
        ray = np.asarray(ray_dir if ray_dir is not None else np.cross(J[0], J[1]), dtype=float)
        ray = ray / np.linalg.norm(ray)
        if ray[2] > 0.0:
            ray = -ray
        if abs(float(ray[2])) < 1e-6:
            return None
        pinv = np.linalg.pinv(J)
        x0 = anchor + pinv @ (np.asarray(seed.uv, dtype=float) - anchor_uv)
        per_height = ray / float(ray[2])                       # ray displacement per metre of height
        ceiling = self._lift_ceiling()
        floor = float(self.planes[0]["z"].value) if self.planes else None
        step = hf.cell_m / 2.0
        start = x0 + ray * ((ceiling - float(x0[2])) / float(ray[2]))
        first_unknown, hit, hit_sigma = None, None, 0.0
        for k in range(int(math.ceil(2.0 * 2.0 * _hf_extent(hf) / step))):
            p = start + ray * (k * step)
            if floor is not None and p[2] < floor:
                break
            z, s, known = hf.sample(p[:2])
            if known and p[2] <= z:
                hit, hit_sigma = p + ray * ((float(z) - float(p[2])) / float(ray[2])), float(s)
                break
            if first_unknown is None and not known and hf.index(p[:2]) is not None:
                first_unknown = p
        if hit is None and floor is not None and first_unknown is not None:
            hit, hit_sigma = x0 + ray * ((floor - float(x0[2])) / float(ray[2])), float(self.planes[0]["z"].sigma)
        if hit is None:
            return None
        if first_unknown is None:
            point, sigma_z, kind = hit, hit_sigma, "surface"
        else:
            top_z, low_z = float(first_unknown[2]), float(hit[2])
            point = x0 + ray * (((top_z + low_z) / 2.0 - float(x0[2])) / float(ray[2]))
            sigma_z, kind = max((top_z - low_z) / math.sqrt(12), hit_sigma), "unknown_span"
        try:
            _, sigma_proj = projector.project(point)
        except (ValueError, TypeError, AttributeError):
            sigma_proj = 0.0
        sigma_px2 = (float(seed.quantization_px) / math.sqrt(3)) ** 2 + float(sigma_proj) ** 2
        cov = sigma_px2 * (pinv @ pinv.T) + sigma_z ** 2 * np.outer(per_height, per_height)
        self.last_external_lift = jsonable({"kind": kind, "xyz": point, "ceiling": ceiling, "floor": floor,
                                            "sigma_z": sigma_z, "sigma_px": math.sqrt(sigma_px2)})
        return point, cov

    def _lift_ceiling(self) -> float:
        """The highest surface an external lift considers: the start fingertips (grip z - t_f at cal.start_pose; the
        robot starts above the scene) or the highest measured surface when higher."""
        ceiling = -np.inf
        start = getattr(self.cal, "start_pose", None) if self.cal is not None else None
        if start:
            ceiling = float(start[2]) - self.robot.t_f_m
        hf = self.hf
        if hf is not None and hf.measured.any():
            ceiling = max(ceiling, float(np.nanmax(np.where(hf.measured, hf.z, -np.inf))))
        if not np.isfinite(ceiling) and self.view is not None:
            ceiling = float(HandPose.from_view(self.view).grip[2]) - self.robot.t_f_m
        return float(ceiling)

    # ---------------------------------------------------------------------------------------- band labelling
    def _height_samples(self, view, pose, images):
        """Height samples of the view: live survey / touch samples and plane-filled cells (at the plane height),
        projected (uv, z, sigma, kind) with kind 0 measured, 1 plane fill."""
        hf = self.hf
        xyz, sigma, sources = self._live_samples()
        parts = []
        if len(xyz):
            uv, _ = _project(self.camera, pose, xyz)
            parts.append((uv, xyz[:, 2], sigma, np.zeros(len(xyz), int)))
        fills = np.argwhere(hf.source == _PLANE_CODE)
        if len(fills):
            member = self._plane_member[fills[:, 0], fills[:, 1]]
            ok = (member >= 0) & (member < len(self.planes))
            fills, member = fills[ok], member[ok]
            if len(fills):
                z = np.array([self.planes[m]["z"].value for m in member])
                s = np.array([self.planes[m]["z"].sigma for m in member])
                pts = np.column_stack([hf.origin_xy + fills[:, ::-1] * hf.cell_m, z])
                uv, _ = _project(self.camera, pose, pts)
                parts.append((uv, z, s, np.ones(len(fills), int)))
        if not parts:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0, int)
        uv = np.vstack([p[0] for p in parts])
        z = np.concatenate([p[1] for p in parts])
        s = np.concatenate([p[2] for p in parts])
        kind = np.concatenate([p[3] for p in parts])
        ok = _inside(uv, images["lab"].shape)
        px = _pix(uv[ok])
        ok_idx = np.flatnonzero(ok)
        ok[ok_idx[images["robot"][px[:, 1], px[:, 0]]]] = False
        return uv[ok], z[ok], s[ok], kind[ok]

    def _segment(self, view, *, seed_uv=None, seed_z=None, seed_sigma=None):
        """Band labelling of the wrist view (module docstring): returns {"labels" (H, W) int band index or -1 robot /
        unknown, "bands": [{"z","sigma","n","measured"}], "seed_band", "base_band", "parallax", "thresholds"}."""
        pose = HandPose.from_view(view)
        images = self._images(view)
        h, w = images["lab"].shape[:2]
        uv, z, s, kind = self._height_samples(view, pose, images)
        if seed_z is not None:
            uv = np.vstack([uv, np.asarray(seed_uv, dtype=float).reshape(1, 2)])
            z = np.concatenate([z, [float(seed_z)]])
            s = np.concatenate([s, [float(seed_sigma or 0.0)]])
            kind = np.concatenate([kind, [0]])
        weights = np.ones(len(z))
        bands = _height_bands(z, s, weights, kind) if len(z) else []
        seed_band = None
        if seed_z is not None:
            seed_band = _band_of(bands, float(seed_z), float(seed_sigma or 0.0))
            if seed_band is None:
                bands.append({"z": float(seed_z), "sigma": float(seed_sigma or 0.0), "n": 1,
                              "members": np.array([len(z) - 1]), "measured": 1})
                bands.sort(key=lambda b: b["z"])
                seed_band = _band_of(bands, float(seed_z), float(seed_sigma or 0.0))
        n_bands = len(bands)
        markers = np.zeros((h, w), dtype=np.int32)
        conflict = np.zeros((h, w), dtype=bool)
        thresholds = [self._d8_threshold(s), self._d9_depth_threshold(s),
                      th_value("D9_gradient_ratio", D9_GRADIENT_RATIO, "DD",
                               "region edge where |grad Lab| > D9_GRADIENT_RATIO x max(local median over "
                               "D9_MEDIAN_WINDOW_PX, image median)", window_px=D9_MEDIAN_WINDOW_PX)]
        par = (self._parallax(view, pose, images, bands, focus=seed_band,
                              centre_uv=seed_uv if seed_band is not None else None) if n_bands >= 2 else None)
        rejected = None
        if par is not None:
            # a window straddling a strong (D9) edge measures the occluding side: a band there is not placed (the
            # intensity watershed places that boundary); the parallax markers serve the weak edges between bands
            near_edge = cv2.dilate(images["edge"].astype(np.uint8), _disk_kernel(PARALLAX_WINDOW_PX // 2)) > 0
            confident = (par["labels"] >= 0) & ~near_edge
            if seed_band is not None and par.get("rejected") is not None:
                # for the pointed band (second pass below): a pixel some other height explains better, by the
                # confidence margin, is not the pointed top, at edges too (the occluding contour of a lower surface,
                # a side face or a descending curve seen in perspective)
                rejected = par["rejected"]
            # surfaces are contiguous: a band labels a pixel only inside a D9 segment that holds one of its samples
            # (two close bands, e.g. 12 mm apart, differ by ~2 px of parallax and noise can swap them far away)
            comps = images["components"]
            allowed = np.zeros((n_bands, h, w), dtype=bool)
            for b, band in enumerate(bands):
                members = np.asarray(band["members"], dtype=int)
                members = members[members < len(uv)]
                px = _pix(uv[members]) if len(members) else np.zeros((0, 2), int)
                segs = np.unique(comps[px[:, 1], px[:, 0]]) if len(px) else np.zeros(0, int)
                allowed[b] = np.isin(comps, segs[segs > 0]) | (comps == 0)
            label = np.clip(par["labels"], 0, None)
            confident &= np.take_along_axis(allowed, label[None, :, :], axis=0)[0]
            markers[confident] = par["labels"][confident] + 1
            thresholds.append(par["threshold"])
        sample_marks = np.zeros((h, w), dtype=np.int32)
        disk = np.argwhere(_disk_kernel(MARKER_RADIUS_PX) > 0) - MARKER_RADIUS_PX          # (k, 2) dv, du
        dissent = _dissenting_samples(bands, uv, kind, images["components"])
        for b, band in enumerate(bands):
            members = np.asarray(band["members"], dtype=int)
            members = members[(members < len(uv)) & ~dissent[np.clip(members, 0, len(dissent) - 1)]]
            layer = np.zeros((h, w), dtype=bool)
            if len(members):
                centres = _pix(uv[members])
                vv = (centres[:, 1][:, None] + disk[None, :, 0]).ravel()
                uu = (centres[:, 0][:, None] + disk[None, :, 1]).ravel()
                ok = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
                layer[vv[ok], uu[ok]] = True
            on = layer
            conflict |= on & (sample_marks > 0) & (sample_marks != b + 1)
            sample_marks[on] = b + 1
        sample_marks[conflict] = 0
        free = markers == 0                       # confident parallax windows outrank single noisy samples
        markers[free & (sample_marks > 0)] = sample_marks[free & (sample_marks > 0)]
        markers[free & conflict] = 0
        if seed_band is not None and seed_uv is not None:
            cv2.circle(markers, (int(round(seed_uv[0])), int(round(seed_uv[1]))), MARKER_RADIUS_PX, seed_band + 1, -1)
        markers[images["robot"]] = 0              # the robot's pixels take no label of their own (masked below):
        # a robot marker would flood the textureless support next to the fingers
        if not (markers > 0).any() or n_bands == 0:
            return {"labels": np.full((h, w), -1, dtype=np.int32), "bands": bands, "seed_band": seed_band,
                    "base_band": None, "parallax": par is not None, "thresholds": thresholds,
                    "robot": images["robot"], "edge": images["edge"]}
        rgb = np.ascontiguousarray(images["rgb"])
        ws = _fill_boundaries(cv2.watershed(rgb, markers.copy()))
        labels = np.where((ws >= 1) & (ws <= n_bands), ws - 1, -1).astype(np.int32)
        labels[images["robot"]] = -1
        # a plane band needs plane evidence: a pixel flooded into it that fails the plane's D7a colour test and holds
        # none of the band's measured samples is unknown (an unmeasured object on the support is not the support);
        # the pointed seed's own band is exempt (a part may share a plane's height with another colour)
        for b, band in enumerate(bands):
            plane = self._plane_of(band) if band.get("plane") and b != seed_band else None
            if plane is None:
                continue
            limit = D7_COLOUR_SIGMA_K * float(np.linalg.norm(plane["lab_sigma"]))
            colour_ok = np.linalg.norm(images["lab"] - plane["lab_mean"][None, None, :], axis=-1) <= limit
            labels[(labels == b) & ~colour_ok & (sample_marks != b + 1)] = -1
        if rejected is not None and rejected.any() and seed_band is not None:
            # second pass for the pointed band only: the rejected pixels (connected groups of >= REJECT_MIN_PX; a
            # window-based label is already smoothed, so a thin line along a crease is kept) are a "not this band" marker beside the first pass's own evidence markers (samples, confident
            # windows, the seed), so the band's outline settles on the strongest ridge between its evidence and the
            # blobs, or half-way across a textureless stretch between them (a descending curve or side face whose only
            # evidence is its outline); the other bands keep the first labelling; the second pass only takes pixels
            blobs = _drop_specks(rejected, REJECT_MIN_PX)
            second = markers.copy()
            second[blobs & (second != seed_band + 1)] = n_bands + 1
            second[images["robot"]] = 0
            ws2 = _fill_boundaries(cv2.watershed(rgb, second))
            mine = (ws2 == seed_band + 1) & ~images["robot"]
            labels[(labels == seed_band) & ~mine] = -1
        return {"labels": labels, "bands": bands, "seed_band": seed_band, "base_band": self._base_band(bands),
                "parallax": par is not None, "thresholds": thresholds, "robot": images["robot"], "edge": images["edge"],
                "parallax_info": None if par is None else par["info"]}

    def _plane_of(self, band):
        """The fitted plane a plane band stands for (the nearest within the D8 tolerance), or None."""
        best, best_d = None, None
        for plane in self.planes:
            d = abs(plane["z"].value - float(band["z"]))
            if d <= max(D8_FLOOR_M, D8_SIGMA_K * plane["z"].sigma) and (best_d is None or d < best_d):
                best, best_d = plane, d
        return best

    @staticmethod
    def _per_sample_threshold(name, k, sigma):
        """A D8 / D9 tolerance as compared: per sample max(D8_FLOOR_M, k * sigma_i); the record's value is the median
        realized tolerance, its inputs the floor, k and the realized range."""
        sigma = np.asarray(sigma, dtype=float)
        tol = np.maximum(D8_FLOOR_M, k * sigma) if len(sigma) else np.array([D8_FLOOR_M])
        label = "ONLINE" if float(np.median(tol)) > D8_FLOOR_M else "DD"
        return th_value(name, float(np.median(tol)), label, "per sample max(D8_FLOOR_M, k * sigma_i), median over the "
                        "samples", floor_m=D8_FLOOR_M, k=k, tol_min_m=float(tol.min()), tol_max_m=float(tol.max()),
                        samples=int(len(sigma)))

    def _d9_depth_threshold(self, sigma):
        return self._per_sample_threshold("D9_depth_step_m", D9_DEPTH_SIGMA_K, sigma)

    def _d8_threshold(self, sigma):
        return self._per_sample_threshold("D8_same_plane_m", D8_SIGMA_K, sigma)

    def _base_band(self, bands):
        if not bands:
            return None
        if self.planes:
            base = self.planes[0]["z"]
            index = _band_of(bands, base.value, base.sigma)
            if index is not None:
                return index
        return int(np.argmax([b["n"] for b in bands]))

    def _parallax_partners(self, view, pose):
        """Cached views at least PARALLAX_MIN_BASELINE_M away in the image plane (largest baselines first, one per
        direction quadrant), at most PARALLAX_MAX_FRAMES."""
        seen, out = set(), []
        cands = []
        for _, other in list(self._keyframes) + list(self._recent):
            if other is view or any(other is c[1] for c in cands):
                continue
            op = HandPose.from_view(other)
            d = op.grip - pose.grip
            inplane = np.array([float(d @ pose.R[:, 1]), float(d @ pose.R[:, 0])])
            base = float(np.linalg.norm(inplane))
            axial = abs(float(d @ pose.R[:, 2]))
            if PARALLAX_MIN_BASELINE_M <= base <= PARALLAX_MAX_BASELINE_M and axial <= PARALLAX_MAX_AXIAL_M:
                cands.append((base, other, inplane))
        for base, other, inplane in sorted(cands, key=lambda c: -c[0]):
            quadrant = int(((math.atan2(inplane[1], inplane[0]) + math.pi / 4) % (2.0 * math.pi)) // (math.pi / 2.0))
            if quadrant in seen:
                continue
            seen.add(quadrant)
            out.append(other)
            if len(out) >= PARALLAX_MAX_FRAMES:
                break
        return out

    def _parallax(self, view, pose, images, bands, focus=None, centre_uv=None):
        """Two-view photometric consistency of the height bands (the D9 depth-step cue between bands): the plane
        homographies of a height sweep (every band height and a PARALLAX_SWEEP_STEP_M grid from PARALLAX_SWEEP_MARGIN_M
        below the lowest band to that much above the highest) map the partner views onto this view; per partner the
        cost is the mean |grey difference| over a PARALLAX_WINDOW_PX window, combined over the partners as the mean
        without the worst (with >= 3 partners: a surface next to a taller one is occluded in the partners that move
        toward it). A pixel is labelled band b when its best sweep height lies within b's D8 tolerance and every height
        beyond that tolerance costs at least PARALLAX_MARGIN_K x the local noise more (confident; the local noise is
        the larger of the static pixel noise and the window residual at the best height); a confident best height
        within no band's tolerance is labelled OTHER_HEIGHT (a surface at none of the measured heights, e.g. a side
        face or a part's descending curve seen in perspective); else -1. For the ``focus`` band (a point seed's)
        ``rejected`` marks the pixels some other height explains better by that margin (window costs, and per pixel
        against the pixel's residual at its best height). The sweep labels pixels for the band watershed and never
        stamps the HeightField; ``centre_uv`` (a point seed) restricts it to the PARALLAX_CROP_HALF_PX window."""
        partners = self._parallax_partners(view, pose)
        if not partners or not bands:
            return None
        h, w = images["gray"].shape
        zs = np.array([float(b["z"]) for b in bands])
        grid = np.arange(zs.max() + PARALLAX_SWEEP_MARGIN_M, zs.min() - PARALLAX_SWEEP_MARGIN_M,
                         -PARALLAX_SWEEP_STEP_M)
        heights = np.unique(np.r_[grid, zs])
        crop = None
        if centre_uv is not None:
            cu, cv = int(round(float(centre_uv[0]))), int(round(float(centre_uv[1])))
            crop = (max(cu - PARALLAX_CROP_HALF_PX, 0), max(cv - PARALLAX_CROP_HALF_PX, 0),
                    min(cu + PARALLAX_CROP_HALF_PX + 1, w), min(cv + PARALLAX_CROP_HALF_PX + 1, h))
        pixel_cost = None
        if focus is not None:
            cost, pixel_cost = self._sweep_cost(view, pose, partners, heights, 1, images, crop=crop, per_pixel=True)
        else:
            cost = self._sweep_cost(view, pose, partners, heights, 1, images, crop=crop)
        x0, y0, x1, y1 = (0, 0, w, h) if crop is None else crop
        noise = math.sqrt(2.0) * self._pixel_sigma()
        margin = PARALLAX_MARGIN_K * noise
        threshold = th_value("D9_parallax_margin_grey", margin, "ONLINE",
                             "two-view band test: every sweep height beyond the band's D8 tolerance costs >= "
                             "PARALLAX_MARGIN_K * max(sqrt(2) * sigma_px, the window residual at the best height) more "
                             "than the best (the value is the static-noise floor)", sigma_px=self._pixel_sigma(),
                             k=PARALLAX_MARGIN_K, sweep_step_m=PARALLAX_SWEEP_STEP_M,
                             sweep_margin_m=PARALLAX_SWEEP_MARGIN_M)
        labels = np.full((h, w), -1, dtype=np.int32)
        if cost is None:
            return {"labels": labels, "threshold": threshold, "margin": margin, "best_z": None,
                    "info": {"partners": len(partners), "confident_px": 0, "other_px": 0, "margin_grey": margin,
                             "heights": int(len(heights)), "focus": focus}}
        filled = np.where(np.isfinite(cost), cost, np.inf)
        best_i = np.argmin(filled, axis=0)
        best = np.take_along_axis(filled, best_i[None], axis=0)[0]
        best_z = heights[best_i]
        tol = np.array([max(D8_FLOOR_M, D8_SIGMA_K * float(b["sigma"])) for b in bands])
        dist = np.abs(best_z[None, :, :] - zs[:, None, None])                  # (n_bands, h, w)
        near = dist <= tol[:, None, None]
        label = np.where(near.any(axis=0), np.argmin(np.where(near, dist, np.inf), axis=0), OTHER_HEIGHT)
        # confidence: the cheapest height outside the tolerance of the label's band (of the best height itself for
        # OTHER_HEIGHT) costs at least the margin more than the best
        centre = np.where(label >= 0, zs[np.clip(label, 0, None)], best_z)
        reach = np.where(label >= 0, tol[np.clip(label, 0, None)], D8_FLOOR_M)
        outside = np.abs(heights[:, None, None] - centre[None, :, :]) > reach[None, :, :]
        rival = np.min(np.where(outside, filled, np.inf), axis=0)
        # the window residual at the best height is the local view-to-view photometric noise (specular change,
        # interpolation), usually above the static pixel noise on real renders: the rival must exceed the best by
        # PARALLAX_MARGIN_K x the larger of the two
        with np.errstate(invalid="ignore"):
            local = PARALLAX_MARGIN_K * np.maximum(noise, np.where(np.isfinite(best), best, 0.0))
            confident = np.isfinite(best) & np.isfinite(rival) & ((rival - best) >= local)
        labels_c = np.full(best.shape, -1, dtype=np.int32)
        labels_c[confident] = label[confident]
        labels[y0:y1, x0:x1] = labels_c
        rejected = None
        if focus is not None and 0 <= int(focus) < len(bands):
            # the pointed band is rejected where its own heights (its D8 tolerance) cost the margin more than the best
            # height: another height explains the pixel better (a side face, a descending curve, a lower surface)
            own = np.abs(heights - zs[int(focus)]) <= tol[int(focus)]
            own_cost = np.min(np.where(own[:, None, None], filled, np.inf), axis=0)
            with np.errstate(invalid="ignore"):
                rejected = np.isfinite(best) & np.isfinite(own_cost) & ((own_cost - best) >= local)
            # and per pixel (no window mixing across the part's own outline): the pointed band's own per-pixel
            # |difference| exceeds the pixel's |difference| at its best window height by the margin (a textured side
            # face; a textureless pixel differs alike at every height and is never rejected this way)
            if pixel_cost is not None:
                filled_px = np.where(np.isfinite(pixel_cost), pixel_cost, np.inf)
                own_px = np.min(np.where(own[:, None, None], filled_px, np.inf), axis=0)
                best_px = np.take_along_axis(filled_px, best_i[None], axis=0)[0]
                with np.errstate(invalid="ignore"):
                    rejected |= np.isfinite(own_px) & np.isfinite(best_px) & ((own_px - best_px) >= local)
            whole = np.zeros((h, w), dtype=bool)
            whole[y0:y1, x0:x1] = rejected
            rejected = whole
        best_whole = np.full((h, w), np.nan)
        best_whole[y0:y1, x0:x1] = np.where(confident, best_z, np.nan)
        return {"labels": labels, "threshold": threshold, "margin": margin, "rejected": rejected,
                "best_z": best_whole,
                "info": {"partners": len(partners), "confident_px": int((labels >= 0).sum()),
                         "other_px": int((labels == OTHER_HEIGHT).sum()), "margin_grey": margin,
                         "heights": int(len(heights)), "focus": focus}}

    # ---------------------------------------------------------------------------------------------- register
    def register(self, pid: str, view: StepView, *, seed: Seed | None = None, survey=None, empty_hand: bool,
                 provisional: bool = False, allow_partial: bool = False) -> PartRecord | RegistrationFailure:
        """§4.5 registration 1-8 (module docstring). Without a wrist seed the part's external seed is lifted and
        carried into the view (plan w = 0). provisional [C-37]: the centroid (xyz, cov) of the region that best
        overlaps the seed, status stays "seeded"; any cut below PARTIAL_MAX_CUT_FRACTION is accepted (partial=True
        for a side cut), and the sigmas cover the seed's quantization, the region choice and a cut. allow_partial
        [C-38]: a region side-cut by the image border or the robot is accepted with partial=True, the lateral and
        width sigmas inflated by 1 / (1 - cut) (not the top's, rulings R2), the region-choice and hidden-height terms,
        no elongation or circle, and its top a lower bound (``top_bound`` "lower", ``top_upper_z``); an end cut of an
        elongated region (the fingers over a grip part at its hover) is not partial.

        Design review 1 (decision §2.2): a wrist point seed takes its top band (``_seed_band``) for the seed lift
        (S-2), top_z and sigma_h (S-3), the support check (S-7) and, when the region is too small, the band members'
        hull (S-4' fallback); after a point-seeded registration that returns a PartRecord the dense-lattice samples
        off the part are withdrawn (S-5, ``last_dsw``; never re-registered on the cleaned field)."""
        hf, camera = self._require()
        record = self.parts.get(pid)
        if record is None:
            return self._failure(pid, "part_not_found", {"why": "unknown_part"}, ())
        if survey is not None:
            self.absorb_survey(survey, view)
        if seed is not None and not any(s is seed for s in record.seeds):
            record = dataclasses.replace(record, seeds=list(record.seeds) + [seed])
            self.parts[pid] = record
        seed = seed if seed is not None else self._best_seed(record)
        external_lift = None
        if seed is None:
            # no wrist seed (a plan part with w = 0): the external seed lifted onto the HeightField (or the support)
            # and carried into this view as an area seed of its own projected uncertainty
            seed, external_lift = self._wrist_seed_from_external(record, view)
            if seed is None:
                if provisional and external_lift is not None:
                    return self._register_external_provisional(record, view, external_lift)
                why = "no_seed" if external_lift is None else "seed_out_of_view"
                return self._failure(pid, "part_not_found", {"why": why, "external_lift": None if external_lift is None
                                                             else external_lift[0]}, ())
        point_band = self._seed_band(seed) if external_lift is None else None
        self._band_cache = (seed, point_band)
        before = self.parts.get(pid)
        before_extents = copy.deepcopy(self._extents.get(pid))
        try:
            result = self._register_seed(record, view, seed, external_lift, point_band, empty_hand=empty_hand,
                                         provisional=provisional, allow_partial=allow_partial)
        finally:
            self._band_cache = None
        fallback = False
        if (isinstance(result, RegistrationFailure) and not provisional and point_band is not None
                and point_band.get("rule") == "band"
                and (result.evidence or {}).get("why") in ("seed_outside_region", "region_too_small")):
            # MC-R1 (gap round): the seed-surface band fallback, same view, zero calls
            alternative = self._seed_band_fallback(self.parts.get(pid, record), view, seed, point_band, result,
                                                   empty_hand=empty_hand, allow_partial=allow_partial)
            if alternative is not None:
                result, fallback = alternative, True
        if isinstance(result, PartRecord) and result.xyz is not None and pid not in self._start_record:
            self._start_record[pid] = result          # MC-R3: the part's start-view record (its first 3-D record)
        if isinstance(result, PartRecord) and result is not before:
            # BS-1 bookkeeping: the seed kind of this registration; an area-seeded one keeps the record it replaced
            kind = "external" if external_lift is not None else ("point" if seed.source in _POINT_SEEDS else "area")
            self._record_kind[pid] = kind
            if kind == "point":
                self._area_prior.pop(pid, None)
            else:
                self._area_prior[pid] = (before, before_extents)
        if point_band is not None and isinstance(result, PartRecord) and not result.provisional and not fallback:
            floor = ((float(point_band["lowest_z"]), float(point_band["lowest_sigma"]))
                     if point_band.get("rule") == "band" else None)
            dsw = self._withdraw_dense(pid, result, view, floor=floor)
            self.last_dsw[pid] = dsw
            if isinstance(self.last_registration.get(pid), dict):
                self.last_registration[pid]["dsw"] = dsw
        return result

    def _register_seed(self, record, view, seed, external_lift, point_band, *, empty_hand, provisional,
                       allow_partial):
        """The registration steps after the seed is chosen (``register``)."""
        camera = self.camera
        pid = record.pid
        pose = HandPose.from_view(view)
        point_seed = seed.source in _POINT_SEEDS
        seed_pose = HandPose.from_state8(seed.pose)
        banded = point_band is not None and point_band.get("rule") == "band"
        lifted = self.estimate_seed_xyz(seed, view) if external_lift is None else external_lift
        if lifted is None and not point_seed and self.planes:
            # an area seed (plan cell / coarse) only needs a depth to be carried to another pose: its cell lies on
            # or above the support, so the support plane is used
            base = self.planes[0]["z"]
            xyz_base = _backproject_z(camera, seed_pose, np.asarray(seed.uv, dtype=float), base.value)[0]
            if np.isfinite(xyz_base).all():
                depth = float((xyz_base - seed_pose.camera) @ seed_pose.R[:, 2])
                lifted = xyz_base, self._seed_cov(seed_pose, seed.uv, depth, float(seed.quantization_px), base.sigma)
        if lifted is None:
            reason = "geometry_unmeasured" if point_seed else "part_not_found"
            return self._failure(pid, reason, {"why": "seed_not_lifted", "seed": seed.to_dict()}, ())
        xyz0, cov0 = lifted
        if _same_pose(seed_pose, pose):
            seed_uv = np.asarray(seed.uv, dtype=float)
        else:
            seed_uv = _project(camera, pose, xyz0)[0][0]
        if not _inside(seed_uv, (camera.size_px, camera.size_px))[0]:
            return self._failure(pid, "part_not_found", {"why": "seed_out_of_view"}, ())
        depth_seed = float((xyz0 - seed_pose.camera) @ seed_pose.R[:, 2])
        depth_now = float((xyz0 - pose.camera) @ pose.R[:, 2])
        q_now = float(seed.quantization_px) * (depth_seed / max(depth_now, 1e-6))
        on_support, support_th = (self._band_on_support(record, point_band, provisional=provisional) if banded
                                  else (None, None))
        if on_support is not None:
            # decision 2 §2.3 S-15 diagnosis (the band lift of a thin grip part at the image border), extended by lead
            # ruling 7 (r2close) to every point-seeded part: the top band of the pointed window is the base support
            # itself, so the window measured the support, not the part's top
            return self._failure(pid, "part_not_found", {"why": "band_on_support", "source": "top_band",
                                                         **on_support, "seed_band": _band_log(point_band)},
                                 [support_th])
        if not provisional:
            self.fill_unknown(view)
        # the segmentation's seed sigma: a band-lifted seed takes one member's height scale (spread_m), never the
        # part-level model term (decision §2.2 binding detail: TOP_MODEL_M enters sigma_h only, not a D8 band or the
        # region growth); otherwise the lift's own depth sigma
        sigma0 = float(point_band["spread_m"]) if banded else math.sqrt(max(float(cov0[2, 2]), 0.0))
        seg = self._segment(view, seed_uv=seed_uv if point_seed else None,
                            seed_z=float(xyz0[2]) if point_seed else None, seed_sigma=sigma0 if point_seed else None)
        chosen = self._choose_component(seg, seed_uv, q_now, point_seed, self._images(view)["edge"])
        thresholds = list(seg["thresholds"])
        if chosen is None:
            if provisional and external_lift is not None:
                return self._register_external_provisional(record, view, external_lift)
            return self._failure(pid, "part_not_found", {"why": "no_region_at_seed", "seed": seed.to_dict(),
                                                         "bands": _bands_summary(seg["bands"])}, thresholds)
        band_index, mask = chosen
        mask = _clean_mask(mask, seed_uv)
        fragmented = False
        if int(mask.sum()) < REGION_MIN_PX and band_index is not None:
            # D9 edges inside one band (surface markings: rings, grooves) cut it into pieces below the minimum
            # region at the seed: the band's own connected region at the seed stands in for the edge-free one
            whole = seg["labels"] == band_index
            if whole.any():
                count, comps = cv2.connectedComponents(whole.astype(np.uint8), connectivity=4)
                u, v = _nearest_mask_pixel(whole, seed_uv)
                mask = _clean_mask(comps == comps[int(v), int(u)], seed_uv)
                fragmented = True
        hull_region = None
        if int(mask.sum()) < REGION_MIN_PX and banded and int(point_band["n"]) >= TOP_BAND_SUPPORT_N:
            # decision §2.2 S-4' (remedy 1's region_too_small fallback): a point seed whose top band holds >=
            # TOP_BAND_SUPPORT_N members takes the members' convex hull, dilated by the probe template radius
            hull, hull_region = self._band_hull_mask(point_band, pose)
            if hull is not None and int(hull.sum()) >= REGION_MIN_PX:
                mask = hull
        if int(mask.sum()) < REGION_MIN_PX:
            return self._failure(pid, "geometry_unmeasured", {"why": "region_too_small", "pixels": int(mask.sum()),
                                                              "band_hull": hull_region}, thresholds)
        seg = dict(seg, edge_fragmented=fragmented, band_hull=hull_region)
        images = self._images(view)
        cut_all, cut, cut_kind = self._cut_analysis(mask, images["robot"])
        # [C-38] an end cut of an elongated region (its width across the axis whole, only its length hidden: the
        # fingers over a grip part at its hover) is no partial view; a side cut is. A provisional centroid [C-37]
        # only centres a hover, so any cut below PARTIAL_MAX_CUT_FRACTION is accepted for it (then partial)
        refuse = cut >= PARTIAL_MAX_CUT_FRACTION or (cut >= PARTIAL_CUT_FRACTION and not allow_partial
                                                     and not provisional)
        if refuse:
            return self._failure(pid, "geometry_unmeasured", {"why": "cut_by_border", "cut_fraction": cut,
                                                              "cut_all": cut_all, "cut_kind": cut_kind,
                                                              "flag": "partial_registration"}, thresholds)
        partial = cut >= PARTIAL_CUT_FRACTION
        inflate = 1.0 / (1.0 - cut)
        ambiguity = self._seed_ambiguity(seg, seed_uv, q_now, pose, mask) if not point_seed else None
        if band_index is None:
            # an edge-bounded region with no band of its own (an unmeasured object): its height from the samples
            # inside or on its outline, else the lifted point seed
            band = self._region_band(mask, pose)
            if band is None and point_seed:
                # decision 2 S-17: the lifted point seed as a band of its own, with the keys of ``_region_band``
                # (``_bands_summary`` reads "n"; no sample is a member, it is no plane)
                band = {"z": float(xyz0[2]), "sigma": sigma0, "measured": 1, "members": np.zeros(0, int), "n": 0,
                        "plane": False}
            if band is None:
                return self._failure(pid, "geometry_unmeasured", {"why": "region_without_depth",
                                                                  "pixels": int(mask.sum())}, thresholds)
            seg = dict(seg, bands=list(seg["bands"]) + [band])
            band_index = len(seg["bands"]) - 1
        band = seg["bands"][band_index]
        top = self._top_height(mask, view, pose, band, point_band=point_band)
        if point_seed and not provisional:
            # lead ruling 7 (r2close): the measured top of a point-seeded registration on the base support plane (a
            # point seed without a top band included, e.g. the review-2 object-body fine seed with one gated sample
            # whose region is the support; coarse / plan-cell seeds are area seeds, never refused here) registered
            # the support, not the part
            on_top, top_th = self._top_on_support(top, point_band if banded else None)
            if on_top is not None:
                return self._failure(pid, "part_not_found", {"why": "band_on_support", "source": "measured_top",
                                                             **on_top, "seed_band": _band_log(point_band)},
                                     list(thresholds) + [top_th])
        conflict = None
        if not point_seed:
            # grip-part lens BS-1 (primary fix): an area-seeded registration (plan cell, coarse, an external lift carried
            # into the view; provisional ones included) whose region is another part's current region (HeightField
            # cell IoU >= SHARED_REGION_IOU_MIN) registered that part, not this one
            conflict = self._shared_region(pid, mask, pose, float(top.value))
            if conflict is not None:
                thresholds = list(thresholds) + [conflict.pop("threshold")]
                if conflict["decision"] == "refuse":
                    self.region_conflicts.append(jsonable({"step": self.step, "pid": pid, **conflict}))
                    return self._failure(pid, "part_not_found", {"why": "region_of_part", **conflict}, thresholds,
                                         keep_record=True)
        if banded:
            thresholds.append(self._top_band_threshold(point_band))
        kappa = camera.f_px / max(float(pose.camera[2] - top.value), 1e-6)
        # rulings R2: a side-cut (partial) region's top is a lower bound; its upper value is the ceiling of the
        # external lift (max(start fingertip height, highest measured surface)); an end cut is not partial.
        # Decision 2 §2.3 S-17 (BSR-4): only when the part band is not a plane band (a planar surface cut by the
        # border shows its top: its band IS the plane's height, measured where it is seen)
        lower = bool(partial) and not band.get("plane")
        top_upper = max(self._lift_ceiling(), float(top.value)) if lower else None
        cut_info = {"cut_fraction": cut, "cut_all": cut_all, "cut_kind": cut_kind,
                    "top_bound": "lower" if lower else None, "top_upper_z": top_upper, "hidden_sigma_h": 0.0,
                    "part_band_plane": bool(band.get("plane"))}
        if cut_all >= PARTIAL_CUT_FRACTION:
            # a D8 comparison of samples with the top: the band top enters with one member's height scale, never the
            # part-level model term (decision §2.2 binding detail)
            top_d8 = Meas(top.value, float(point_band["spread_m"]), top.source) if banded else top
            cut_info["hidden_sigma_h"] = self._hidden_sigma_h(mask, pose, top_d8, kappa)
        if provisional:
            result = self._register_provisional(record, view, pose, mask, top, kappa, inflate, cut_info, partial,
                                                q_now if not point_seed else float(seed.quantization_px), ambiguity,
                                                thresholds, external=external_lift is not None, point_band=point_band)
        else:
            result = self._register_full(record, view, pose, seed, seed_uv, point_seed, mask, seg, band_index, top,
                                         kappa, inflate, cut_info, partial, empty_hand, thresholds, ambiguity,
                                         point_band=point_band, q_now=q_now)
        if conflict is not None and isinstance(result, PartRecord):
            self._yield_region(pid, conflict)
        return result

    # ------------------------------------------------------------------------------------- BS-1 shared regions
    def _part_cells(self, pid) -> np.ndarray | None:
        """The (n, 2) ij of a part's CURRENT region (BS-1): its region_cells, else (a provisional record) the cells
        of its registered outline polygon; None for a part with no 3-D point, not registered nor provisional, or
        held (a held part's region moves with the hand)."""
        rec = self.parts.get(pid)
        if rec is None or rec.xyz is None or not (rec.status == "registered" or rec.provisional):
            return None
        if self.held is not None and pid in (self.held.pid, self.held.grip_pid):
            return None
        if rec.region_cells is not None and len(rec.region_cells):
            return np.asarray(rec.region_cells, dtype=int).reshape(-1, 2)
        poly = (self._extents.get(pid) or {}).get("polygon")
        if poly is not None and len(poly) >= 3:
            cells = self.hf.cells_in_polygon(poly)
            return cells if len(cells) else None
        return None

    def region_overlap(self, pid_a: str, pid_b: str) -> dict | None:
        """BS-1 / BR-1 / BR-3 helper: the HeightField-cell overlap of two parts' current regions ({"iou", "inter",
        "cells", "other_cells", "inside_fraction" (share of a's cells inside b's)}), None when either has none."""
        if self.hf is None:
            return None
        a, b = self._part_cells(pid_a), self._part_cells(pid_b)
        if a is None or b is None:
            return None
        return _cell_overlap(a, b, self.hf.shape)

    def _shared_region(self, pid, mask, pose, z) -> dict | None:
        """BS-1: the other parts whose current region shares cell IoU >= SHARED_REGION_IOU_MIN with this area-seeded
        registration's region (its outline polygon at its top, the cells _register_full would store). None without
        one. "refuse" when this part grips one of them (P in a plan step's grip, Q its obj: P is refused) or is not
        gripped by it (the earlier registration keeps the region); "yield" when every such part grips THIS part (the
        object keeps its region; each grip part returns to its record before its last area-seeded registration,
        ``_yield_region``). Carries a "threshold" (popped by the caller)."""
        poly = self._mask_polygon(mask, pose, z)
        cells = None if poly is None else self.hf.cells_in_polygon(poly)
        if cells is None or not len(cells):
            return None
        rows = []
        for other in self.parts:
            if other == pid:
                continue
            other_cells = self._part_cells(other)
            if other_cells is None:
                continue
            stats = _cell_overlap(cells, other_cells, self.hf.shape)
            if stats["iou"] >= SHARED_REGION_IOU_MIN:
                rows.append({"other": other, **stats,
                             "this_grips_other": (pid, other) in self._grip_pairs,
                             "other_grips_this": (other, pid) in self._grip_pairs})
        if not rows:
            return None
        rows.sort(key=lambda r: -r["iou"])
        refused = [r for r in rows if r["this_grips_other"] or not r["other_grips_this"]]
        head = refused[0] if refused else rows[0]
        threshold = th_value("shared_region_iou", SHARED_REGION_IOU_MIN, "DD",
                             "BS-1 (grip-part lens): an area-seeded registration whose region shares HeightField-cell "
                             "IoU >= SHARED_REGION_IOU_MIN with another part's current region is refused (the grip "
                             "part of a plan step's grip / obj pair, else the later registration)",
                             measured=float(head["iou"]), other=head["other"], cells=int(head["cells"]),
                             other_cells=int(head["other_cells"]))
        if refused:
            return {"decision": "refuse", "other": head["other"], "iou": float(head["iou"]),
                    "inter": int(head["inter"]), "cells": int(head["cells"]), "other_cells": int(head["other_cells"]),
                    "rule": "grip_part_refused" if head["this_grips_other"] else "earlier_keeps",
                    "overlaps": rows, "threshold": threshold}
        return {"decision": "yield", "yield": [r["other"] for r in rows], "overlaps": rows, "threshold": threshold}

    def _yield_region(self, pid, conflict) -> None:
        """BS-1, the object registered after its grip part took its region: each grip part whose current record came
        from an area seed returns to the record it had before that registration (its earlier, independent estimate;
        BR-1 hovers from it); a point-seeded grip record is left to BR (BR-3's IoU check) and only logged."""
        entries = []
        for other in conflict.get("yield", ()):
            entry = {"step": self.step, "pid": pid, "decision": "yield", "yielded": other, "reverted": False,
                     "overlap": next((r for r in conflict["overlaps"] if r["other"] == other), None)}
            prior = self._area_prior.get(other)
            if self._record_kind.get(other) in ("area", "external") and prior is not None and prior[0] is not None:
                record, extents = prior
                self.parts[other] = record
                if extents is None:
                    self._extents.pop(other, None)
                else:
                    self._extents[other] = extents
                self._area_prior.pop(other, None)
                self._record_kind.pop(other, None)
                entry.update(reverted=True, reverted_to={"status": record.status, "provisional": record.provisional,
                                                         "registration_step": record.registration_step})
            entries.append(jsonable(entry))
        self.region_conflicts.extend(entries)
        if isinstance(self.last_registration.get(pid), dict):
            self.last_registration[pid]["region_of_part"] = entries

    def _band_hull_mask(self, point_band, pose):
        """(mask | None, info) of the S-4' region_too_small fallback: the top-band members projected into the view,
        their convex hull dilated by the probe tracker's template radius (geometry FT_PROBE_RADIUS_PX, the wrist
        template radius NCC_RADIUS_PX for a tracker without it), the robot's pixels removed."""
        size = (self.camera.size_px, self.camera.size_px)
        pts = self._samples["xyz"][np.asarray(point_band["members"], dtype=int)]
        uv, _ = _project(self.camera, pose, pts)
        uv = uv[np.isfinite(uv).all(axis=1)]
        radius = float(getattr(self.vision, "FT_PROBE_RADIUS_PX", NCC_RADIUS_PX))
        info = {"members": int(len(pts)), "radius_px": radius}
        if len(uv) < 3:
            return None, dict(info, pixels=0)
        mask = np.zeros(size, dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
        mask = (cv2.dilate(mask, _disk_kernel(radius)) > 0) & ~self._robot_mask(size)
        return mask, dict(info, pixels=int(mask.sum()))

    def _band_on_support(self, record, point_band, *, provisional=False):
        """(info, Threshold) when a point seed's top band lies on the base support (planes[0]) by the D8 same-surface
        test, max(D8_FLOOR_M, D8_SIGMA_K x hypot(the support's sigma, one member's height scale)), else (None, None).
        A part registered at its support's height is a mis-registration: such a band measured the support around a
        part whose own pixels hold no sample (a dark thin part at the image border, decision 2 §2.3 S-15 diagnosis;
        the 155.7 mm low object-body record of lead ruling 7). Scope (lead ruling 7, r2close): every point-seeded
        part on a non-provisional call; on a provisional call grip parts only (the S-15 diagnosis rule, unchanged).
        Only a point seed has a top band, so an area-seeded record is never refused here. No new number."""
        if not self.planes or point_band is None or point_band.get("rule") != "band":
            return None, None
        if provisional and "grip" not in record.roles:
            return None, None
        base = self.planes[0]["z"]
        spread = float(point_band.get("spread_m") or 0.0)
        tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(base.sigma), spread))
        dz = float(point_band["top"]) - float(base.value)
        if abs(dz) > tol:
            return None, None
        threshold = th_value("top_band_on_support_m", tol, "DD",
                             "lead ruling 7: a point seed's top band within max(D8_FLOOR_M, D8_SIGMA_K "
                             "hypot(sigma_support, band spread)) of the base support plane measured the support, not "
                             "the part (D8 same surface)", measured=dz, support_z=float(base.value),
                             band_top_z=float(point_band["top"]), members=int(point_band.get("n", 0)),
                             provisional=bool(provisional))
        return {"band_top_z": float(point_band["top"]), "support_z": float(base.value), "dz_m": dz, "tol_m": tol}, \
            threshold

    def _top_on_support(self, top, point_band):
        """Lead ruling 7 (r2close), the measured top: (info, Threshold) when a point-seeded registration's measured top
        lies on the base support (planes[0]) by the D8 same-surface test max(D8_FLOOR_M, D8_SIGMA_K x
        hypot(sigma_support, sigma_top)), else (None, None). sigma_top is the top band's member height scale
        (``spread_m``) for a banded seed, else the measured top's own sigma (``_top_height``: its samples' or the part
        band's), never the TOP_MODEL_M part-level model term. It covers the seeds without a top band (the review-2
        object-body record: a fine seed with one gated sample, its region the base support). No new number."""
        if not self.planes:
            return None, None
        base = self.planes[0]["z"]
        sigma_top = float(point_band["spread_m"]) if point_band is not None else float(top.sigma)
        tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(base.sigma), sigma_top))
        dz = float(top.value) - float(base.value)
        if abs(dz) > tol:
            return None, None
        threshold = th_value("top_on_support_m", tol, "DD", "lead ruling 7: a point-seeded registration's measured top "
                             "within max(D8_FLOOR_M, D8_SIGMA_K hypot(sigma_support, sigma_top)) of the base support "
                             "plane registered the support, not the part (D8 same surface)", measured=dz,
                             support_z=float(base.value), top_z=float(top.value), sigma_top_m=sigma_top)
        return {"top_z": float(top.value), "support_z": float(base.value), "dz_m": dz, "tol_m": tol,
                "sigma_top_m": sigma_top}, threshold

    def _members_hull(self, point_band, pose, dilate_px):
        """The top-band members (``_seed_band``) projected into the view at ``pose``: their convex hull dilated by a
        ``dilate_px`` disk (decision 2: the D10-OE hull ROI, S-11, and the collapse guard's band hull, S-13; the axis
        lens's band hull, no robot removal), or None with fewer than 3 projected members."""
        if point_band is None or point_band.get("rule") != "band" or "members" not in point_band:
            return None
        size = (self.camera.size_px, self.camera.size_px)
        pts = self._samples["xyz"][np.asarray(point_band["members"], dtype=int)]
        uv, _ = _project(self.camera, pose, pts)
        uv = uv[np.isfinite(uv).all(axis=1)]
        if len(uv) < 3:
            return None
        uv = np.clip(uv, -float(max(size)), 2.0 * float(max(size)))           # bounded raster coordinates
        hull = np.zeros(size, dtype=np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
        if dilate_px:
            hull = cv2.dilate(hull, _disk_kernel(dilate_px))
        return hull > 0

    def _collapse_check(self, mask, point_band, pose) -> dict | None:
        """S-13 (decision 2 §2.3, the axis lens's collapse guard): the region collapsed onto a facet smaller than its
        measured top when the top-band hull (``_members_hull``, dilated COLLAPSE_HULL_DILATE_PX) holds more than
        REGION_COLLAPSE_RATIO x the region's pixels. None without a top band of >= 3 projected members (not
        evaluated); else {"collapsed", "hull_px", "region_px", "ratio", "threshold"}."""
        hull = self._members_hull(point_band, pose, COLLAPSE_HULL_DILATE_PX)
        if hull is None:
            return None
        hull_px, region_px = int(hull.sum()), int(np.count_nonzero(mask))
        ratio = float(hull_px) / max(region_px, 1)
        threshold = th_value("region_collapse_ratio", REGION_COLLAPSE_RATIO, "DD",
                             "S-13: the region collapsed when its top-band hull (dilated COLLAPSE_HULL_DILATE_PX) holds "
                             "more than REGION_COLLAPSE_RATIO x its pixels", measured=ratio, hull_px=hull_px,
                             region_px=region_px, dilate_px=COLLAPSE_HULL_DILATE_PX)
        return {"collapsed": bool(hull_px > REGION_COLLAPSE_RATIO * region_px), "hull_px": hull_px,
                "region_px": region_px, "ratio": ratio, "threshold": threshold}

    @staticmethod
    def _top_band_threshold(point_band):
        return th_value("D4_top_model_m", float(point_band["model_m"]), "DD",
                        "point-seed top band (decision §2.2 S-1..S-3): samples sigma_z <= TOP_GATE_M within "
                        "TOP_WINDOW_PX; highest band of >= TOP_BAND_K within max(D8_FLOOR_M, TOP_BAND_SIGMA_K median "
                        "sigma); top = its TOP_PERCENTILE; sigma_h = hypot(stat, TOP_MODEL_M if n >= "
                        "TOP_BAND_SUPPORT_N else TOP_MODEL_THIN_M)", n=int(point_band["n"]),
                        stat_m=float(point_band["stat_m"]), n_window=int(point_band["n_window"]),
                        n_gated=int(point_band["n_gated"]), gate_m=TOP_GATE_M, window_px=TOP_WINDOW_PX,
                        support_n=TOP_BAND_SUPPORT_N)

    def _choose_component(self, seg, seed_uv, q_px, point_seed, edge):
        """The part's region. A point seed on a non-base band: the largest D9-edge-free facet of its band within the
        seed's quantization square, joined by the band's smaller facets across its internal edges
        (``_merge_facets``); a point seed on unknown pixels: its own edge-free region. An area seed (or a point seed
        that landed on the base support): the band (or unknown) region overlapping the seed's quantization square
        most, taken as that band's largest facet in the square plus its smaller facets. Edge pixels of the band that
        border the face are added back."""
        labels = seg["labels"]
        h, w = labels.shape
        base = seg["base_band"]
        u, v = int(round(float(seed_uv[0]))), int(round(float(seed_uv[1])))
        unknown = (labels == -1) & ~seg.get("robot", np.zeros_like(edge))
        q = max(int(math.ceil(q_px)), 1)
        y0, y1, x0, x1 = max(v - q, 0), min(v + q + 1, h), max(u - q, 0), min(u + q + 1, w)

        def face_of(b, comps, inside):
            window = comps[y0:y1, x0:x1]
            ids = np.unique(window[window > 0])
            if not len(ids):
                return None
            areas = np.bincount(comps.ravel())
            face = comps == int(max(ids, key=lambda c: areas[c]))
            return self._with_edges(_merge_facets(face, comps, inside), inside)

        if 0 <= v < h and 0 <= u < w:
            b = int(labels[v, u])
            b = None if b < 0 else b
            if b != base and (b is not None or unknown[v, u]):
                inside = unknown if b is None else labels == b
                _, comps = self._band_regions(inside, edge)
                if b is not None:
                    face = face_of(b, comps, inside)
                    if face is not None:
                        return b, face
                elif (comps > 0).any():
                    at = _nearest_mask_pixel(comps > 0, (u, v))
                    return b, self._with_edges(comps == comps[int(at[1]), int(at[0])], inside)
        best = None
        for b in [b for b in range(len(seg["bands"])) if b != base] + [None]:
            inside = unknown if b is None else labels == b
            _, comps = self._band_regions(inside, edge)
            window = comps[y0:y1, x0:x1]
            ids, overlaps = np.unique(window[window > 0], return_counts=True)
            for cid, overlap in zip(ids, overlaps):
                if best is None or overlap > best[0]:
                    best = (int(overlap), b, comps, inside, int(cid))
        if best is None or best[0] < REGION_MIN_PX:
            return None
        _, b, comps, inside, cid = best
        if b is None:
            return b, self._with_edges(comps == cid, inside)
        return b, face_of(b, comps, inside)

    @staticmethod
    def _band_regions(inside, edge):
        return cv2.connectedComponents((inside & ~edge).astype(np.uint8), connectivity=4)

    @staticmethod
    def _with_edges(region, band_mask):
        """A region plus the band's D9 edge pixels that border it within EDGE_HALF_WIDTH_PX (the inner half of its
        own outline: the true boundary lies in the middle of the 2-3 px Sobel edge band)."""
        grown = cv2.dilate(region.astype(np.uint8), _disk_kernel(EDGE_HALF_WIDTH_PX)) > 0
        return region | (grown & band_mask)

    @staticmethod
    def _cut_fraction(mask, robot):
        return SceneModel._cut_analysis(mask, robot)[0]

    @staticmethod
    def _cut_analysis(mask, robot) -> tuple[float, float, str]:
        """[C-38] (fraction of the outline on the image border or the robot's own pixels, the fraction that counts as
        a partial view, kind "none" | "end" | "side"). An END cut: the region is elongated in the image (principal
        extent >= D10_MIN_ASPECT x the cross extent) and every cut outline point lies within half a cross extent of an
        end of its long axis, so the width across the axis is whole and only the length is hidden (an elongated part
        running under the fingers): it does not count. Any other cut is a SIDE cut (a border sliver can make a round
        part look elongated) and counts in full."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0, 0.0, "none"
        pts = np.vstack([c.reshape(-1, 2) for c in contours])
        h, w = mask.shape
        border = (pts[:, 0] <= 0) | (pts[:, 0] >= w - 1) | (pts[:, 1] <= 0) | (pts[:, 1] >= h - 1)
        near_robot = cv2.dilate(robot.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        border |= near_robot[pts[:, 1], pts[:, 0]]
        fraction = float(border.mean())
        if not border.any():
            return fraction, 0.0, "none"
        vs, us = np.nonzero(mask)
        xy = np.column_stack([us, vs]).astype(float)
        centre = xy.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((xy - centre).T)) if len(xy) > 2 else (None, None)
        if vectors is None:
            return fraction, fraction, "side"
        major, minor = vectors[:, 1], vectors[:, 0]
        t, n = (xy - centre) @ major, (xy - centre) @ minor
        length, width = float(np.ptp(t)), max(float(np.ptp(n)), 1.0)
        tc = (pts[border].astype(float) - centre) @ major
        at_end = (tc >= t.max() - width / 2.0) | (tc <= t.min() + width / 2.0)
        if length >= D10_MIN_ASPECT * width and at_end.all():
            return fraction, 0.0, "end"
        return fraction, fraction, "side"

    def _seed_ambiguity(self, seg, seed_uv, q_px, pose, mask) -> dict:
        """The other candidate regions of an area seed (contract §C6 provisional / partial sigma): every region of a
        non-base band or of unknown pixels overlapping the seed's quantization square by >= REGION_MIN_PX, with its
        centroid (px) and height (its band, or the samples inside / on its outline). The chosen region is one of
        them; a wrong choice is the provisional record's dominant error."""
        labels, bands = seg["labels"], seg["bands"]
        h, w = labels.shape
        u, v = int(round(float(seed_uv[0]))), int(round(float(seed_uv[1])))
        q = max(int(math.ceil(q_px)), 1)
        y0, y1, x0, x1 = max(v - q, 0), min(v + q + 1, h), max(u - q, 0), min(u + q + 1, w)
        edge = seg["edge"]
        unknown = (labels == -1) & ~seg.get("robot", np.zeros_like(edge))
        out = []
        for b in [b for b in range(len(bands)) if b != seg["base_band"]] + [None]:
            inside = unknown if b is None else labels == b
            _, comps = self._band_regions(inside, edge)
            window = comps[y0:y1, x0:x1]
            ids, overlaps = np.unique(window[window > 0], return_counts=True)
            for cid, overlap in zip(ids, overlaps):
                if overlap < REGION_MIN_PX:
                    continue
                region = comps == cid
                z = float(bands[b]["z"]) if b is not None else None
                if z is None:
                    band = self._region_band(region, pose)
                    z = None if band is None else float(band["z"])
                out.append({"overlap": int(overlap), "centroid_px": _centroid_px(region), "z": z,
                            "chosen": bool(np.count_nonzero(region & mask) >= 0.5 * np.count_nonzero(region))})
        return {"candidates": out, "q_px": float(q_px)}

    def _mask_samples(self, mask, pose):
        xyz, sigma, sources = self._live_samples()
        if not len(xyz):
            return xyz, sigma, sources
        uv, _ = _project(self.camera, pose, xyz)
        ok = _inside(uv, mask.shape)
        px = _pix(uv[ok])
        idx = np.flatnonzero(ok)[mask[px[:, 1], px[:, 0]]]
        return xyz[idx], sigma[idx], [sources[i] for i in idx]

    def _top_height(self, mask, view, pose, band, *, point_band=None) -> Meas:
        """top_z. A wrist point seed with a top band (decision §2.2 S-3, replacing spec §4.5 step 4 for point-seeded
        parts): the band's TOP_PERCENTILE height with the honest sigma_h = hypot(statistical, part-level model term),
        no 1 / (1 - cut) factor (rulings R2). Otherwise (area seeds, or no band): the TOP_PERCENTILE height of the
        measured samples inside the region, taken as the median of the samples within the D8 band of that percentile
        (spec §4.5 step 4); the band height when nothing is measured."""
        if point_band is not None and point_band.get("rule") == "band":
            members = np.asarray(point_band["members"], dtype=int)
            used = [self._samples["source"][i] for i in members]
            source = max(set(used), key=used.count) if used else "kappa"
            if source not in ("kappa", "loom", "touch", "dense"):
                source = "kappa"
            return Meas(float(point_band["top"]), float(point_band["sigma_h"]), source)
        xyz, sigma, sources = self._mask_samples(mask, pose)
        tol_band = max(D8_FLOOR_M, D8_SIGMA_K * float(band["sigma"]))
        if len(xyz):
            in_band = np.abs(xyz[:, 2] - band["z"]) <= tol_band
            if in_band.any():
                xyz, sigma = xyz[in_band], sigma[in_band]
                sources = [s for s, k in zip(sources, in_band) if k]
                z95 = _weighted_percentile(xyz[:, 2], sigma, TOP_PERCENTILE)
                tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * sigma)
                top = np.abs(xyz[:, 2] - z95) <= tol
                values = xyz[top, 2]
                source = max(set(sources), key=sources.count) if sources else "kappa"
                if source not in ("kappa", "loom", "touch", "dense"):
                    source = "kappa"
                # the samples' own sigma, or their measured scatter when they disagree beyond it (a claimed sigma
                # smaller than the spread of the top samples is not the top's uncertainty)
                return Meas(float(np.median(values)), max(float(np.median(sigma[top])), _robust_sigma(values)), source)
        return Meas(float(band["z"]), float(band["sigma"]), "plane" if not band.get("measured") else "kappa")

    def _support(self, mask, seg, part_band, kappa, pose, *, floor=None
                 ) -> tuple[int | None, Meas | None, np.ndarray | None, dict]:
        """The support band: the majority band below the part in the SUPPORT_RING_M ring around the region; its
        height from the measured samples inside the support region around the part (the band height without).

        S-7 (decision §2.2), with ``floor`` = (z, sigma) of the lowest top-band member of a point seed: the support
        (plane or measured mode) must lie below that member by at least max(D8_FLOOR_M, D8_SIGMA_K x hypot(its sigma,
        the member's)) (two heights are distinct surfaces by the D8 test); the test applies to the band level and to
        the support's measured height (its region median, with its sigma). A ring majority that does not pass is
        rejected (logged). Decision 2 S-17 (BSR-6): the fallback is the next admissible ring band below the floor
        by ring pixels, a measured band or a plane (``fallback`` "ring_below"), its height taken from the ring's own
        samples of that band (``_ring_height``: inverse-variance weighted, >= BAND_MIN_SAMPLES; else its region's);
        with no such ring band the highest plane band below the floor ("plane_below"); with none the support stays
        unmeasured (None, "unmeasured") and the planner takes its P4 / geometry_unmeasured path. The primary choice
        stays the ring-pixel majority (BS deviation 5, ratified by lead ruling 1, r2close, with the ring-own
        inverse-variance fallback; the nearest-admissible-surface rule is not adopted): replayed on the live-captured
        inputs of 109 registered A / C / E hovers (BS fixer, runs/agent_speedup_v2_bs_r2_fix_support_before_*), a
        nearest-surface primary (the highest admissible D8 mode of the ring samples, or the highest admissible
        ring-evidenced band)
        picks the part's own flank or its attached body (gate episodes within 3 mm: A 7/20, C 0/19, E 1/18 against
        15/20, 18/19, 16/18), and the ring-sample median fails C and E (9/19, 3/18). Returns (band index, Meas,
        region, check log)."""
        labels = seg["labels"]
        bands = seg["bands"]
        ring_px = max(int(math.ceil(kappa * SUPPORT_RING_M)), MARKER_RADIUS_PX + 1)
        outer = cv2.dilate(mask.astype(np.uint8), _disk_kernel(ring_px)) > 0
        inner = cv2.dilate(mask.astype(np.uint8), _disk_kernel(MARKER_RADIUS_PX)) > 0
        ring = outer & ~inner
        tol = max(D8_FLOOR_M, D9_DEPTH_SIGMA_K * float(bands[part_band]["sigma"]))     # a D9 depth step below
        check = {"floor_z": None if floor is None else float(floor[0]), "rejected": [], "fallback": None}

        def below_floor(band):
            if floor is None:
                return True
            step = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(band["sigma"]), float(floor[1])))
            return float(band["z"]) <= float(floor[0]) - step

        def measure(b, ring_own=False):
            band = bands[b]
            _, comps = cv2.connectedComponents((labels == b).astype(np.uint8), connectivity=4)
            touching = np.unique(comps[ring & (labels == b)])
            touching = touching[touching > 0]
            region = np.isin(comps, touching) if len(touching) else None
            meas = None
            if ring_own:
                # decision 2 S-17 (BSR-6): a fallback surface is measured from the ring's own samples of that band
                # (the surface next to the part), with at least BAND_MIN_SAMPLES of them, inverse-variance weighted
                # (``_ring_height``: imprecise samples agreeing within their wide tolerance do not widen it)
                meas = self._ring_height(ring & (labels == b), pose, band)
            if meas is None and region is not None:
                meas = self._region_height(region & ~inner, pose, band)
            if meas is None:
                meas = Meas(float(band["z"]), float(band["sigma"]), "kappa" if band.get("measured") else "plane")
            return meas, region

        ring_px_of = {b: int(np.count_nonzero(ring & (labels == b))) for b in range(len(bands))}
        lower = [b for b, band in enumerate(bands) if band["z"] < bands[part_band]["z"] - tol and ring_px_of[b] > 0]
        best = max(lower, key=lambda b: ring_px_of[b]) if lower else None
        # S-7 applies to the band level AND to the support's measured height (the median of the samples in the
        # support region, which can sit up to ~3 sigma above the band level; BS review BSR-9 a). Decision 2 S-17
        # (BSR-6): a rejected ring majority falls back to the next admissible ring band below the floor by ring
        # pixels, measured or plane (a measured disk under a part beats the table plane farther out), measured from
        # the ring's own samples; only then the highest plane band below the floor
        rejected, fallback_ring = set(), False
        while best is not None:
            entry = {"z": float(bands[best]["z"]), "sigma": float(bands[best]["sigma"]),
                     "ring_px": ring_px_of[best], "plane": bool(bands[best].get("plane"))}
            if below_floor(bands[best]):
                meas, region = measure(best, ring_own=fallback_ring)
                if below_floor({"z": meas.value, "sigma": meas.sigma}):
                    check["support_z"] = float(meas.value)
                    return best, meas, region, check
                entry.update(measured_z=float(meas.value), measured_sigma=float(meas.sigma))
            check["rejected"].append(entry)
            rejected.add(best)
            ring_bands = [b for b in lower if b not in rejected and below_floor(bands[b])]
            planes = [b for b, band in enumerate(bands) if band.get("plane") and b not in rejected
                      and below_floor(band)]
            if ring_bands:
                best, fallback_ring = max(ring_bands, key=lambda b: ring_px_of[b]), True
                check["fallback"] = "ring_below"
            else:
                best, fallback_ring = (max(planes, key=lambda b: float(bands[b]["z"])) if planes else None), False
                check["fallback"] = "plane_below" if best is not None else "unmeasured"
        return None, None, None, check

    def _region_band(self, mask, pose) -> dict | None:
        """A pseudo height band of a region without one: the TOP_PERCENTILE band of the samples inside the region
        or on its outline (within a feature patch radius: an occluding-edge feature's patch centre may lie just
        outside the region it measures); None without samples."""
        grown = cv2.dilate(mask.astype(np.uint8), _disk_kernel(_NCC_RADIUS)) > 0
        xyz, sigma, _ = self._mask_samples(grown, pose)
        if not len(xyz):
            return None
        z_top, sigma_top, members = _top_cluster(xyz[:, 2], sigma)
        return {"z": z_top, "sigma": sigma_top, "measured": int(len(members)), "members": np.zeros(0, int),
                "n": int(len(members)), "plane": False}

    def _region_height(self, region, pose, band) -> Meas | None:
        """Median height of the measured samples inside an image region that lie within the D8 band of the region's
        band (iterated once about their own median); None without such samples."""
        xyz, sigma, sources = self._mask_samples(region, pose)
        if not len(xyz):
            return None
        tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(sigma, float(band["sigma"])))
        keep = np.abs(xyz[:, 2] - float(band["z"])) <= tol
        if not keep.any():
            return None
        centre = float(np.median(xyz[keep, 2]))
        keep = np.abs(xyz[:, 2] - centre) <= np.maximum(D8_FLOOR_M, D8_SIGMA_K * sigma)
        if not keep.any():
            return None
        used = [src for src, k in zip(sources, keep) if k]
        source = max(set(used), key=used.count)
        return Meas(float(np.median(xyz[keep, 2])), float(np.median(sigma[keep])), source)

    def _ring_height(self, region, pose, band) -> Meas | None:
        """Decision 2 S-17 (BSR-6), the fallback surface measured from the ring's own samples of a band (BS fix review
        BSR6-ring-own-sigma): the samples within the band's D8 tolerance (hypot of their sigma and the band's), the
        centre their inverse-variance weighted median, then the samples within their own D8 tolerance of it (at least
        BAND_MIN_SAMPLES): value = their weighted median, sigma = max(the weighted median of their sigmas, their
        weighted robust scatter). The weights are ``_weighted_percentile``'s (1 / max(sigma_i, D8_FLOOR_M /
        D8_SIGMA_K)^2): the precise samples measure the surface, and imprecise ones that agree with it only within
        their own wide tolerance neither move it nor widen its sigma (the plain median of the claimed sigmas did, and
        a correctly chosen disk then failed S-7 on its sigma alone). None with fewer than BAND_MIN_SAMPLES."""
        xyz, sigma, sources = self._mask_samples(region, pose)
        if len(xyz) < BAND_MIN_SAMPLES:
            return None
        z = xyz[:, 2]
        tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(sigma, float(band["sigma"])))
        keep = np.abs(z - float(band["z"])) <= tol
        if int(keep.sum()) < BAND_MIN_SAMPLES:
            return None
        centre = _weighted_percentile(z[keep], sigma[keep], 50)
        keep = np.abs(z - centre) <= np.maximum(D8_FLOOR_M, D8_SIGMA_K * sigma)
        if int(keep.sum()) < BAND_MIN_SAMPLES:
            return None
        value = _weighted_percentile(z[keep], sigma[keep], 50)
        spread = MAD_TO_SIGMA * _weighted_percentile(np.abs(z[keep] - value), sigma[keep], 50)
        used = [src for src, k in zip(sources, keep) if k]
        return Meas(float(value), max(float(_weighted_percentile(sigma[keep], sigma[keep], 50)), float(spread)),
                    max(set(used), key=used.count))

    def _ambiguity_sigmas(self, ambiguity, centre_px, top_z) -> tuple[float, float, dict]:
        """(lateral px, height m, info) of the region choice of an area seed: the farthest other candidate's centroid
        and height divided by REGION_CHOICE_SIGMA_K (a wrong choice lands within that many sigma); zeros for a point
        seed or a single candidate."""
        if not ambiguity:
            return 0.0, 0.0, {"candidates": 0}
        others = [c for c in ambiguity["candidates"] if not c["chosen"]]
        lat = max((float(np.linalg.norm(np.asarray(c["centroid_px"]) - np.asarray(centre_px))) for c in others
                   if c["centroid_px"] is not None), default=0.0)
        dz = max((abs(float(c["z"]) - float(top_z)) for c in others if c["z"] is not None), default=0.0)
        return lat / REGION_CHOICE_SIGMA_K, dz / REGION_CHOICE_SIGMA_K, {
            "candidates": len(ambiguity["candidates"]), "farthest_centroid_px": lat, "farthest_dz_m": dz}

    def _hidden_sigma_h(self, mask, pose, top, kappa) -> float:
        """[C-38] a cut region's top is a lower bound: the measured surfaces higher than it (beyond its D8
        tolerance) inside it or within its SUPPORT_RING_M annulus may be its hidden part, so sigma_h reaches the
        highest of them at REGION_CHOICE_SIGMA_K sigma (0 when none is higher)."""
        ring_px = max(int(math.ceil(kappa * SUPPORT_RING_M)), MARKER_RADIUS_PX + 1)
        near = cv2.dilate(mask.astype(np.uint8), _disk_kernel(ring_px)) > 0
        xyz, sigma, _ = self._mask_samples(near, pose)
        if not len(xyz):
            return 0.0
        tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * np.hypot(sigma, float(top.sigma)))
        higher = xyz[:, 2] - float(top.value) > tol
        if not higher.any():
            return 0.0
        return float(np.max(xyz[higher, 2]) - float(top.value)) / REGION_CHOICE_SIGMA_K

    def _cut_extent_m(self, poly, cut) -> float:
        """[C-38] the lateral sigma of a side-cut region's centroid: the part continues beyond the cut by about
        cut / (1 - cut) of its seen extent, which moves the centroid by up to half of that."""
        if poly is None or len(poly) < 3 or cut <= 0.0:
            return 0.0
        _, extents = _principal_axes(poly)
        return float(extents[0]) * cut / (1.0 - cut) / 2.0

    def _register_provisional(self, record, view, pose, mask, top, kappa, inflate, cut_info, partial, q_px,
                              ambiguity, thresholds, *, external=False, point_band=None):
        """[C-37] the region centroid only (hover centring). sigma_lat covers the seed's quantization (q / sqrt(3)),
        the choice among the regions in the seed cell and, for a side cut, the hidden extent (the lateral terms take
        the 1 / (1 - cut) factor); sigma_h covers the heights of the other candidate regions (the reviewer's
        wrong-region risk) and the hidden height, without the cut factor (rulings R2: a cut top is a lower bound,
        ``top_bound`` "lower" with ``top_upper_z``, not a wider sigma)."""
        centre = _centroid_px(mask)
        xyz = _backproject_z(self.camera, pose, centre, top.value)[0]
        poly = self._mask_polygon(mask, pose, top.value)
        choice_px, choice_h, info = self._ambiguity_sigmas(ambiguity, centre, top.value)
        cut = float(cut_info["cut_fraction"])
        sigma_lat = math.hypot(math.hypot(TRACK_SIGMA_PX, float(q_px) / math.sqrt(3)), choice_px) / kappa * inflate
        sigma_lat = math.hypot(sigma_lat, self._cut_extent_m(poly, cut) if partial else 0.0)
        sigma_h = max(top.sigma, choice_h, float(cut_info.get("hidden_sigma_h") or 0.0))
        cov = np.diag([sigma_lat ** 2, sigma_lat ** 2, sigma_h ** 2])
        if poly is not None:
            axes, extents = _principal_axes(poly)
            self._extents[record.pid] = {"axes": axes, "extents": extents, "polygon": poly}
        new = dataclasses.replace(record, xyz=xyz, cov=cov, sigma_h_m=sigma_h, provisional=True, status="seeded",
                                  reason=None, view_pose=tuple(float(v) for v in view.state),
                                  registration_step=self.step, partial=bool(partial), registered_empty_hand=False,
                                  top_bound=cut_info.get("top_bound"), top_upper_z=cut_info.get("top_upper_z"),
                                  top_band=None, axis_check=None)
        self.parts[record.pid] = new
        thresholds = list(thresholds) + [self._choice_threshold(q_px, choice_px, choice_h, kappa)]
        payload = self._payload(new, "provisional", thresholds, extra={
            **cut_info, "flag": "provisional_registration", "ambiguity": info, "seed_from_external": bool(external),
            "top_sigma_raw": float(top.sigma), "inflate": inflate, "choice_sigma_h": float(choice_h),
            "seed_band": _band_log(point_band)})
        self.last_registration[record.pid] = payload
        self._registration_thresholds[record.pid] = list(thresholds)
        return new

    @staticmethod
    def _choice_threshold(q_px, choice_px, choice_h, kappa):
        return th_value("region_choice_sigma", REGION_CHOICE_SIGMA_K, "DD",
                        "area-seed sigma: lateral sqrt(sigma_track^2 + (q/sqrt3)^2 + (farthest candidate centroid / "
                        "REGION_CHOICE_SIGMA_K)^2) / kappa; height >= farthest candidate height difference / "
                        "REGION_CHOICE_SIGMA_K", q_px=float(q_px), choice_px=float(choice_px),
                        choice_h_m=float(choice_h), kappa_px_per_m=float(kappa))

    def _wrist_seed_from_external(self, record, view):
        """(wrist area Seed at this view | None, external lift (xyz, cov) | None) for a part whose only seed is
        external (plan w = 0): the lift projected into the wrist view, its quantization sqrt(3) x the projected sigma;
        no seed when it projects outside the view or onto the robot's own pixels, or when that quantization is coarser
        than a plan cell's (half a coarse cell)."""
        ext = self._best_seed(record, "external")
        lifted = None if ext is None else self._external_seed_xyz(ext)
        if lifted is None:
            return None, None
        xyz, cov = lifted
        pose = HandPose.from_view(view)
        uv, _ = _project(self.camera, pose, xyz)
        size = (self.camera.size_px, self.camera.size_px)
        if not _inside(uv, size)[0] or self._robot_mask(size)[int(round(uv[0, 1])), int(round(uv[0, 0]))]:
            return None, lifted
        J = self._wrist_jacobian(pose, xyz)
        sigma_px = math.sqrt(max(float(np.linalg.eigvalsh(J @ cov @ J.T).max()), 0.0))
        quantization = max(math.sqrt(3) * sigma_px, TRACK_SIGMA_PX)
        wrist_w = int(np.asarray(view.wrist_rgb_upright).shape[1])
        if quantization > coarse_cell_center(1, wrist_w)[0]:
            # coarser than a plan cell: a wrist region chosen in that square would be a guess; the external lift
            # itself is the provisional point
            return None, lifted
        seed = Seed("wrist", (float(uv[0, 0]), float(uv[0, 1])), self.step, "plan_cell", quantization,
                    tuple(float(v) for v in view.state))
        return seed, lifted

    def _register_external_provisional(self, record, view, lifted):
        """[C-37] a grip-only part with only an external seed outside the wrist view: the external lift itself is the
        provisional point (hover centring; cov from the external quantization and the height range)."""
        xyz, cov = lifted
        sigma_h = math.sqrt(max(float(cov[2, 2]), 0.0))
        new = dataclasses.replace(record, xyz=np.asarray(xyz, dtype=float), cov=np.asarray(cov, dtype=float),
                                  sigma_h_m=sigma_h, provisional=True, status="seeded", reason=None,
                                  view_pose=tuple(float(v) for v in view.state), registration_step=self.step,
                                  partial=False, registered_empty_hand=False, top_bound=None, top_upper_z=None,
                                  top_band=None, axis_check=None)
        self.parts[record.pid] = new
        payload = self._payload(new, "provisional", [], extra={"flag": "provisional_registration",
                                                               "seed_from_external": True,
                                                               "external_only": True})
        self.last_registration[record.pid] = payload
        self._registration_thresholds[record.pid] = []
        return new

    def _mask_polygon(self, mask, pose, z):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(contour, TRACK_SIGMA_PX, True).reshape(-1, 2).astype(float)
        if len(approx) < 3:
            return None
        world = _backproject_z(self.camera, pose, approx, z)
        if not np.isfinite(world).all():
            return None
        return world[:, :2]

    def _register_full(self, record, view, pose, seed, seed_uv, point_seed, mask, seg, band_index, top, kappa,
                       inflate, cut_info, partial, empty_hand, thresholds, ambiguity=None, *, point_band=None,
                       q_now=None):
        images = self._images(view)
        banded = point_band is not None and point_band.get("rule") == "band"
        camera = self.camera
        poly = self._mask_polygon(mask, pose, top.value)
        if poly is None:
            return self._failure(record.pid, "geometry_unmeasured", {"why": "no_outline"}, thresholds)
        seed_check = None
        if point_seed:
            grasp_px = np.asarray(seed_uv, dtype=float)
            u0, v0 = int(round(grasp_px[0])), int(round(grasp_px[1]))
            inside = 0 <= v0 < mask.shape[0] and 0 <= u0 < mask.shape[1] and bool(mask[v0, u0])
            q_px = float(seed.quantization_px) if q_now is None else float(q_now)
            radius = q_px + S15_SEED_SIGMA_K * TRACK_SIGMA_PX
            seed_check = {"inside": bool(inside), "gap_px": 0.0, "limit_px": radius, "q_now_px": q_px,
                          "sigma_k": S15_SEED_SIGMA_K, "sigma_px": TRACK_SIGMA_PX, "excess_px": -radius}
            if not inside:
                # decision 2 §2.3 S-15: a pointed pixel farther from the chosen region than its quantization (now)
                # plus the localization noise did not point at that region; the nearest region pixel is no grasp
                # point (it was snapped to before). Lead ruling 3 (r2close): the noise term is S15_SEED_SIGMA_K x
                # TRACK_SIGMA_PX (2 sigma); beyond it BR re-points once at the hover before giving up
                nearest = _nearest_mask_pixel(mask, grasp_px)
                gap = float(np.linalg.norm(nearest - grasp_px))
                limit = th_value("seed_region_px", radius, "DD",
                                 "S-15 (lead ruling 3): the point seed lies within q_now + S15_SEED_SIGMA_K x "
                                 "TRACK_SIGMA_PX of its chosen region", measured=gap, q_now_px=q_px,
                                 sigma_k=S15_SEED_SIGMA_K, sigma_px=TRACK_SIGMA_PX, excess_px=gap - radius)
                seed_check.update(gap_px=gap, excess_px=gap - radius, nearest_uv=[float(v) for v in nearest])
                if gap > limit.value:
                    return self._failure(record.pid, "part_not_found", {
                        "why": "seed_outside_region", "gap_px": gap, "limit_px": limit.value, "q_now_px": q_px,
                        "sigma_k": S15_SEED_SIGMA_K, "excess_px": gap - limit.value, "seed_uv": grasp_px,
                        "nearest_uv": nearest, "region_px": int(np.count_nonzero(mask))},
                        list(thresholds) + [limit])
                thresholds.append(limit)
                grasp_px = nearest
        else:
            grasp_px = _centroid_px(mask)
            if not mask[int(round(grasp_px[1])), int(round(grasp_px[0]))]:
                grasp_px = _nearest_mask_pixel(mask, grasp_px)
        xyz = _backproject_z(camera, pose, grasp_px, top.value)[0]
        q_px = float(seed.quantization_px) if point_seed else 0.0
        cut = float(cut_info["cut_fraction"])
        partial_side = bool(partial)
        top_sigma_raw = float(top.sigma)
        # an area seed's region choice (the other candidate regions in its cell) and, for a side cut, the hidden
        # extent widen the sigmas (the lateral ones by 1 / (1 - cut) too); a side-cut region's top is a lower bound
        # (top_bound "lower" with top_upper_z), so its sigma_h takes no cut factor (rulings R2, decision §2.2 S-3)
        choice_px, choice_h, choice_info = self._ambiguity_sigmas(ambiguity, grasp_px, top.value)
        sigma_lat = math.hypot(math.hypot(TRACK_SIGMA_PX, q_px / math.sqrt(3)), choice_px) / kappa * inflate
        sigma_lat = math.hypot(sigma_lat, self._cut_extent_m(poly, cut) if partial_side else 0.0)
        sigma_h = max(top.sigma, choice_h, float(cut_info.get("hidden_sigma_h") or 0.0))
        if ambiguity:
            thresholds.append(self._choice_threshold(0.0, choice_px, choice_h, kappa))
        cov = np.diag([sigma_lat ** 2, sigma_lat ** 2, sigma_h ** 2])
        top = Meas(top.value, sigma_h, top.source)
        floor = (point_band["lowest_z"], point_band["lowest_sigma"]) if banded else None
        support_band, support, support_region, support_check = self._support(mask, seg, band_index, kappa, pose,
                                                                             floor=floor)
        if support is not None:
            support = Meas(support.value, support.sigma * inflate, support.source)
        # widths per jaw yaw in the pad strip (spec §4.5 step 5). Lead decision F1: each width's sigma is the pixel
        # term (two outline localizations, the top's height scale, the cut factor) combined with the part-level model
        # term W_MODEL_M, sigma_w = hypot(pixel sigma_w, W_MODEL_M); the width values are untouched
        half_strip = self.robot.pad_length_m / 2.0
        depth = max(float(pose.camera[2] - top.value), 1e-6)
        widths, width_px = [], {}
        sigma_px_w = []
        for k in range(int(round(math.pi / YAW_GRID_RAD))):
            yaw = k * YAW_GRID_RAD
            jaw = np.array([math.cos(yaw), math.sin(yaw)])
            extent = _strip_extent(poly, xyz[:2], jaw, half_strip)
            if extent is None:
                proj = poly @ jaw
                extent = float(proj.max() - proj.min())
            sigma_pixel = math.hypot(2.0 * TRACK_SIGMA_PX / kappa, extent * top.sigma / depth) * inflate
            sigma_px_w.append(sigma_pixel)
            widths.append((yaw, Meas(extent, math.hypot(sigma_pixel, W_MODEL_M), "kappa")))
            width_px[k] = extent * kappa
        thresholds.append(th_value("width_model_m", W_MODEL_M, "DD", "lead decision F1: sigma_w = hypot(pixel sigma_w, "
                                   "W_MODEL_M) for every width of the record (the values are untouched)",
                                   pixel_sigma_w_min_m=min(sigma_px_w), pixel_sigma_w_max_m=max(sigma_px_w)))
        # D10 diagnosis (decision 2 §2.3 S-11: retired as deciders, kept logged in axis_check and the thresholds):
        # the structure tensor at the grasp point, patch half-size 0.6 x the measured cross width (the [C-40] track
        # rule), the image's (v1 bar_point_check, geometry.structure_axis) and the region mask's (the higher
        # coherence is the "tensor" axis), the region aspect and the region cells' second-moment axis (S-4')
        w_min = min(m.value for _, m in widths)
        half = max(D10_PATCH_MIN_HALF_PX, int(math.ceil(D10_PATCH_WIDTH_FRACTION * w_min * kappa)))
        mask_axis_img, mask_coherence = _mask_coherence(mask, grasp_px, half)
        axes, extents = _principal_axes(poly if len(poly) >= 3 else np.vstack([poly, poly]))
        region_pts = self._region_points(mask, pose, top.value)
        if region_pts is not None and len(region_pts) >= 3:
            axes, extents = _principal_axes(region_pts)
        aspect = float(extents[0] / max(extents[1], 1e-9))
        image_axis_img, image_coherence = None, None
        try:
            image_axis_img, image_coherence, _ = self.vision.structure_axis(images["gray"], tuple(grasp_px),
                                                                            half_px=half)
            image_axis_img, image_coherence = np.asarray(image_axis_img, dtype=float), float(image_coherence)
        except (ValueError, TypeError, IndexError):
            image_axis_img, image_coherence = None, None
        if image_coherence is not None and (mask_axis_img is None or image_coherence >= mask_coherence):
            axis_img, coherence, tensor_source = image_axis_img, image_coherence, "image"
        else:
            axis_img, coherence, tensor_source = mask_axis_img, float(mask_coherence), "region_mask"
        thresholds.append(th_value("D10_coherence", D10_MIN_COHERENCE, "DD",
                                   "diagnosis (decision 2 S-11, no D10 decider): structure-tensor coherence at the "
                                   "grasp point (the higher of the image and the region-mask tensors) against "
                                   "D10_MIN_COHERENCE", measured=coherence, half_px=half, tensor=tensor_source,
                                   image_coherence=image_coherence, mask_coherence=float(mask_coherence)))
        thresholds.append(th_value("D10_aspect", D10_MIN_ASPECT, "DD", "diagnosis (decision 2 S-11, no D10 decider): "
                                   "region aspect against D10_MIN_ASPECT", measured=aspect))
        axis_world = None if axis_img is None else _image_axis_world(axis_img, pose)
        mask_axis_world = None if mask_axis_img is None else _image_axis_world(mask_axis_img, pose)
        image_axis_world = None if image_axis_img is None else _image_axis_world(image_axis_img, pose)
        region_axis = _canonical_axis(axes[0])
        region_cells = self.hf.cells_in_polygon(poly)
        # MC-R3 (gap round): the part-object consistency gate (PO3, PO1s, PO5) on point and area seeds; the refused
        # part keeps its earlier record
        po_refusal, po_thresholds = self._part_object_check(record.pid, xyz, region_cells, top)
        thresholds.extend(po_thresholds)
        if po_refusal is not None:
            return self._failure(record.pid, "part_not_found", po_refusal, thresholds, keep_record=True)
        cell_axis, cell_ratio = None, None
        if len(region_cells) >= D10_MIN_REGION_CELLS:
            cell_xy = self.hf.origin_xy + np.asarray(region_cells, dtype=float)[:, ::-1] * self.hf.cell_m
            cell_axis, cell_ratio = _second_moment_axis(cell_xy)
        diagnosis = {"cells": int(len(region_cells)), "cell_ratio": cell_ratio, "cell_axis": cell_axis,
                     "tensor": tensor_source, "coherence": float(coherence), "aspect": aspect,
                     "pixel_axis": region_axis, "tensor_axis": axis_world,
                     "tensor_cell_agree_rad": None if axis_world is None or cell_axis is None else
                     _axis_angle(axis_world, cell_axis)}
        # S-13: the collapse guard (a point seed with a top band); S-11: D10-OE on a point-seeded whole region
        # (provisional records never reach here); S-12: the part's view memory decides the record's axis and width
        collapse = self._collapse_check(mask, point_band, pose) if point_seed else None
        collapsed = bool(collapse is not None and collapse["collapsed"])
        if collapse is not None:
            thresholds.append(collapse.pop("threshold"))
        # MC-R2b (gap round): an accepted annulus on the part band's component at the seed replaces the S-13 hull
        # test for this record only; MC-R2a: the collapse stage, and a collapsed view that keeps the whole record
        band_comp = self._band_component(seg, band_index, seed_uv) if point_seed else None
        annulus = None
        if point_seed and not partial_side and band_comp is not None:
            annulus = self._annulus(band_comp, pose, top, kappa)
            if annulus is not None:
                thresholds.extend(annulus.pop("thresholds"))
        annulus_ok = bool(annulus is not None and annulus["accepted"])
        if annulus_ok and collapse is not None:
            collapse["replaced_by"] = "annulus" if collapsed else None
            collapsed = False
        if collapsed and band_comp is not None:
            collapse["stage"] = ("edge" if collapse["hull_px"] <= REGION_COLLAPSE_RATIO * int(band_comp.sum())
                                 else "band")
        kept_whole, kept_log = (self._kept_whole(record.pid, xyz, top, sigma_lat) if collapsed and point_seed
                                else (None, None))
        scope = "partial_side" if partial_side else (None if point_seed else "area_seed")
        oe = None if scope is not None else self._d10_oe(mask, images, pose, top.value, point_band,
                                                         edges=not collapsed)
        if oe is not None:
            thresholds.extend(oe.pop("thresholds"))
        decided = self._axis_decision(record.pid, oe, scope=scope, collapsed=collapsed,
                                      collapse_evaluated=collapse is not None, widths=tuple(widths))
        thresholds.extend(decided["thresholds"])
        remembered_annulus = (self._views.get(record.pid) or {}).get("annulus")
        if annulus_ok:
            decided = self._annulus_widths(decided, annulus, xyz, kappa)
        if decided["memory"] is not None:
            decided["memory"]["annulus"] = (
                {"centre": np.asarray(annulus["centre"], dtype=float), "radius_m": annulus["radius_m"],
                 "wall_m": annulus["wall_m"], "rms_px": annulus["rms_px"], "step": int(self.step)}
                if annulus_ok else remembered_annulus)
        elongation = decided["elongation"]
        axis_check = {**diagnosis, **decided["check"]}
        d10 = bool(axis_check["accepted"])
        axis_source = axis_check["source"]
        # circle (D11): the bounded support region around the part, else the part's own outline
        circle, circle_info, d11 = (None, [{"skipped": "partial"}], None) if partial_side else self._circle(
            mask, support_region, support, pose, images, top.value)
        if d11 is not None:
            thresholds.append(d11)
        annulus_source = None
        if annulus_ok:
            circle = (np.asarray(annulus["centre"], dtype=float), float(annulus["radius_m"]), float(annulus["rms_px"]))
            annulus_source = "annulus"
        elif circle is None and remembered_annulus is not None and not partial_side:
            circle = (np.asarray(remembered_annulus["centre"], dtype=float), float(remembered_annulus["radius_m"]),
                      float(remembered_annulus["rms_px"]))
            annulus_source = AXIS_KEPT_EARLIER
        rigid = self._rigid_patch(mask, images, pose, xyz, top.value, view)
        track = None
        try:
            track = NCCTrack(images["gray"], tuple(grasp_px))
        except ValueError:
            track = None
        external = self._external_projection(xyz, poly, top.value, view)
        known = float(self.hf.known[region_cells[:, 0], region_cells[:, 1]].mean()) if len(region_cells) else 0.0
        new = dataclasses.replace(
            record, xyz=xyz, cov=cov, track=track, track_pose=tuple(float(v) for v in view.state), top_z=top,
            support_z=support, region_xy=poly, region_cells=region_cells, width_by_yaw=decided["widths"],
            elongation=elongation, circle=circle, rigid_patch=rigid, external=external,
            registered_empty_hand=bool(empty_hand), view_pose=tuple(float(v) for v in view.state),
            registration_step=self.step, provisional=False, partial=bool(partial), status="registered", reason=None,
            sigma_h_m=sigma_h, top_bound=cut_info.get("top_bound"), top_upper_z=cut_info.get("top_upper_z"),
            top_band=_top_band_field(point_band) if banded else None, axis_check=axis_check)
        old = self.parts.get(record.pid)
        top_check = po6 = None
        if kept_whole is not None:
            new = kept_whole                  # MC-R2a: a collapsed view does not replace the last whole registration
        else:
            new, top_check = self._keep_lower_bound(old, new)
            if top_check is not None:
                thresholds.append(top_check.pop("threshold"))
            if not (top_check is not None and top_check["kept"]):
                new, po6 = self._po6_check(record.pid, old, new)
                if po6 is not None:
                    thresholds.append(po6.pop("threshold"))
            if old is not None and old.status == "registered" and not old.partial and new.partial:
                new = old                         # a cut view never replaces a whole registration
        self.parts[record.pid] = new
        if new is not old and decided["memory"] is not None and kept_whole is None:
            self._views[record.pid] = decided["memory"]           # S-12: committed with the record it decided
        kept_view = (kept_whole is not None or (top_check is not None and top_check["kept"])
                     or (po6 is not None and po6["kept"]))
        if not kept_view:
            # S-14 / PO6 / MC-R2a: an inconsistent view's outline is not the kept record's (its extents stay the
            # earlier view's)
            self._extents[record.pid] = {"axes": axes, "extents": extents, "polygon": poly}
        if new is not old and kept_whole is None:
            # X-1 bookkeeping: the epoch of the committed record (PO6), HS-H1's body, MC-R2a's whole record
            self._record_epoch[record.pid] = self.epoch_of(record.pid)
            self._note_body(record.pid, new, collapsed)
            if point_seed and not collapsed and not new.partial and new.status == "registered":
                self._whole[record.pid] = {"record": new, "epoch": self.epoch_of(record.pid), "step": int(self.step)}
        flag = "partial_registration" if new.partial else None
        if (top_check is not None and top_check["kept"]) or (po6 is not None and po6["kept"]):
            flag = "top_inconsistent"
        if kept_whole is not None:
            flag = "kept_whole"
        payload = self._payload(new, seed.source, thresholds, extra={
            **cut_info, "top_sigma_raw": top_sigma_raw, "inflate": inflate, "choice_sigma_h": float(choice_h),
            "known_fraction": known, "aspect": aspect,
            "coherence": coherence,
            "ambiguity": choice_info,
            "tensor": tensor_source, "mask_coherence": mask_coherence, "mask_axis": mask_axis_world,
            "image_coherence": image_coherence, "image_axis": image_axis_world, "region_axis": region_axis,
            "axis_source": axis_source,
            "axis_candidate": None if axis_world is None else axis_world,
            "d10": bool(d10), "d10_oe": oe, "collapse": collapse,
            "width_source": "kept_whole" if kept_whole is not None else decided["width_source"],
            "kept_whole_check": kept_log, "annulus": annulus, "annulus_source": annulus_source, "po6": po6,
            "view_memory": decided["log"], "circle_fit": circle_info, "parallax": seg.get("parallax_info"),
            "bands": _bands_summary(seg["bands"]), "part_band": band_index, "support_band": support_band,
            "edge_fragmented": bool(seg.get("edge_fragmented")), "band_hull": seg.get("band_hull"),
            "seed_band": _band_log(point_band), "top_band": new.top_band, "axis_check": new.axis_check,
            "axis_check_view": axis_check, "support_check": support_check, "kept_whole": new is old, "top_check": top_check,
            "seed_check": seed_check, "flag": flag})
        self.last_registration[record.pid] = payload
        self._registration_thresholds[record.pid] = list(thresholds)
        return new

    # ------------------------------------------------------------------- gap round: MC-R1 / R2a / R2b / R3
    def _nearest_band(self, seed, s1) -> dict | None:
        """MC-R1: from the same window, gate and grouping as S-1 (``_seed_band``), the coherent band that holds the
        gated sample nearest the seed pixel (members within b of that sample, re-centred once on their median, >=
        TOP_BAND_K), in ``_seed_band``'s format; None without one."""
        idx = np.flatnonzero(self._live_mask())
        if not len(idx):
            return None
        xyz, sigma = self._samples["xyz"][idx], self._samples["sigma"][idx]
        pose = HandPose.from_state8(seed.pose)
        uv, depth = _project(self.camera, pose, xyz)
        seed_uv = np.asarray(seed.uv, dtype=float)
        with np.errstate(invalid="ignore"):
            dist = np.linalg.norm(uv - seed_uv[None, :], axis=1)
            near = np.isfinite(uv).all(axis=1) & (depth > 0.0) & (dist <= TOP_WINDOW_PX)
        gated = np.flatnonzero(near & (sigma <= TOP_GATE_M))
        if len(gated) < TOP_BAND_K:
            return None
        z, sg = xyz[gated, 2], sigma[gated]
        b = max(D8_FLOOR_M, TOP_BAND_SIGMA_K * float(np.median(sg)))
        i0 = int(np.argmin(dist[gated]))
        members = np.flatnonzero(np.abs(z - z[i0]) <= b)
        if len(members) < TOP_BAND_K:
            return None
        members = np.flatnonzero(np.abs(z - float(np.median(z[members]))) <= b)
        if len(members) < TOP_BAND_K:
            return None
        zb, sb = z[members], sg[members]
        n = int(len(members))
        scatter = _robust_sigma(zb)
        spread = max(float(np.median(sb)), scatter)
        chosen = idx[gated[members]]
        lowest = int(chosen[int(np.argmin(self._samples["xyz"][chosen, 2]))])
        stat, model = spread / math.sqrt(n), _top_model_m(n)
        return {"rule": "band", "n_window": int(near.sum()), "n_gated": int(len(gated)), "n": n, "stat_m": stat,
                "model_m": model, "scatter_m": scatter, "spread_m": spread, "b_m": b,
                "top": float(np.percentile(zb, TOP_PERCENTILE)), "sigma_h": math.hypot(stat, model), "members": chosen,
                "lowest_z": float(self._samples["xyz"][lowest, 2]),
                "lowest_sigma": float(self._samples["sigma"][lowest]), "nearest_px": float(dist[gated][i0])}

    def _seed_band_fallback(self, record, view, seed, s1, failure, *, empty_hand, allow_partial):
        """MC-R1 (gap round, judge decision §2.2): a banded point seed that failed ``seed_outside_region`` or
        ``region_too_small`` (never a collapse) is registered again in the same view (zero calls) on the nearest
        coherent band to the seed pixel (``_nearest_band``) when it differs from S-1 by more than b. Accepted only
        with G1 (not collapsed, widths measured), G2 (narrowest width <= the open pad separation, ROBOT / ONLINE), G3
        (< SHARED_REGION_IOU_MIN of its cells inside a sibling's region), G5b (< M17_FRACTION of the S-1 members
        inside the nearest band's hull: else S-1 is a raised feature on the pointed surface and stands), G6 (>=
        TOP_BAND_SUPPORT_N members), G7 (S-1 is not the part's own previous top within D8) and PO5 (in
        ``_register_full``). The fallback record skips S-5 DSW and is flagged ``seed_band_fallback`` with both bands.
        Returns the PartRecord or None (the scene restored, the attempt logged in ``last_seed_band_fallback``)."""
        pid = record.pid
        why = (failure.evidence or {}).get("why")
        log = {"step": int(self.step), "pid": pid, "trigger": why, "s1": _band_log(s1), "nearest": None,
               "accepted": False, "refused": None}
        self.last_seed_band_fallback = log
        near = self._nearest_band(seed, s1)
        if near is None:
            log["refused"] = "no_nearest_band"
            return None
        log["nearest"] = _band_log(near)
        if abs(float(near["top"]) - float(s1["top"])) <= float(s1["b_m"]):
            log["refused"] = "same_band"
            return None
        if int(near["n"]) < TOP_BAND_SUPPORT_N:
            log["refused"] = "G6_support"
            return None
        prev = self.parts.get(pid)
        if prev is not None and prev.status == "registered" and prev.top_z is not None:
            tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(s1["sigma_h"]), float(prev.top_z.sigma)))
            if abs(float(s1["top"]) - float(prev.top_z.value)) <= tol:
                log["refused"] = "G7_own_top"
                return None
        pose = HandPose.from_view(view)
        hull = self._members_hull(near, pose, 0)
        if hull is not None:
            uv, _ = _project(self.camera, pose, self._samples["xyz"][np.asarray(s1["members"], dtype=int)])
            ok = np.isfinite(uv).all(axis=1)
            px = np.rint(uv[ok]).astype(int)
            h, w = hull.shape
            inb = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
            inside = int(hull[px[inb, 1], px[inb, 0]].sum())
            share = inside / max(int(len(uv)), 1)
            log["s1_inside_nearest_hull"] = share
            if share >= M17_FRACTION:
                log["refused"] = "G5b_enclosed"
                return None
        snapshot = {}
        for name in ("parts", "_views", "_extents", "_body", "_whole", "_record_epoch", "last_registration",
                     "_registration_thresholds"):
            value = getattr(self, name).get(pid)
            snapshot[name] = copy.copy(value) if isinstance(value, (dict, list)) else value

        def restore():
            for name, value in snapshot.items():
                store = getattr(self, name)
                if value is None:
                    store.pop(pid, None)
                else:
                    store[pid] = value
        self._band_cache = (seed, near)
        try:
            result = self._register_seed(record, view, seed, None, near, empty_hand=empty_hand, provisional=False,
                                         allow_partial=allow_partial)
        finally:
            self._band_cache = None
        if not isinstance(result, PartRecord) or result.provisional:
            log["refused"] = "fallback_failed:" + str(getattr(result, "reason", None))
            log["fallback_evidence"] = getattr(result, "evidence", None)
            restore()
            return None
        payload = self.last_registration.get(pid) if isinstance(self.last_registration.get(pid), dict) else {}
        collapse = payload.get("collapse") or {}
        guards = {"G1": bool(not collapse.get("collapsed") and result.width_by_yaw)}
        open_gap = getattr(getattr(self.cal, "gripper", None), "open_gap_mm", None)
        narrowest = min((m.value for _, m in result.width_by_yaw), default=None) if result.width_by_yaw else None
        separation = None if open_gap is None else self.robot.pad_separation_mm(float(open_gap)) / 1000.0
        guards["G2"] = bool(narrowest is not None and separation is not None and narrowest <= separation)
        cells = self._record_cells(result)
        share_sibling = 0.0
        if cells is not None and len(cells):
            mine = {tuple(c) for c in cells.tolist()}
            for other in self.object_parts(pid) - {pid}:
                other_cells = self._part_cells(other)
                if other_cells is not None and len(other_cells):
                    share_sibling = max(share_sibling, len(mine & {tuple(c) for c in np.asarray(other_cells, int)
                                                                   .tolist()}) / len(mine))
        guards["G3"] = share_sibling < SHARED_REGION_IOU_MIN
        log.update(guards=guards, narrowest_m=narrowest, separation_m=separation, sibling_share=share_sibling)
        if not all(guards.values()):
            log["refused"] = "guard:" + ",".join(k for k, v in guards.items() if not v)
            restore()
            return None
        log["accepted"] = True
        if isinstance(self.last_registration.get(pid), dict):
            self.last_registration[pid]["flag"] = "seed_band_fallback"
            self.last_registration[pid]["seed_band_fallback"] = jsonable(log)
        return result

    def _record_cells(self, record) -> np.ndarray | None:
        """The (n, 2) ij of a record's region: its region_cells, else the cells of its outline polygon."""
        if record is None:
            return None
        if record.region_cells is not None and len(record.region_cells):
            return np.asarray(record.region_cells, dtype=int).reshape(-1, 2)
        if record.region_xy is not None and len(record.region_xy) >= 3:
            return self.hf.cells_in_polygon(record.region_xy)
        return None

    def _part_object_check(self, pid, xyz, region_cells, top) -> tuple[dict | None, list]:
        """MC-R3 (gap round, judge decision §2.2; lead ruling L5) at registration, point and area seeds:

        - PO3 ``region_of_other_object``: the part's cells lie >= SHARED_REGION_IOU_MIN inside the current region of a
          part of ANOTHER object (not in ``object_parts``) at the same surface (|dtop| <= max(D8_FLOOR_M, D8_SIGMA_K
          hypot sigma)); an object resting on another one's top stands higher and passes;
        - PO1s ``far_from_object``: a grip sub-part's point (BS-1 grip pair (pid, Q), Q != pid) lies farther than
          ext_start(Q) + PB_REACH_M from Q's start-view region (ext_start = its circumradius about the vertex mean);
        - PO5 ``larger_than_object``: a grip sub-part's region holds more than SUBPART_REGION_MAX_RATIO x the cells of
          Q's start-view region.
        Returns (refusal evidence {"why", ...} or None, thresholds)."""
        thresholds = []
        cells = set() if region_cells is None else {tuple(c) for c in np.asarray(region_cells, int).reshape(-1, 2)
                                                     .tolist()}
        own = self.object_parts(pid)
        if cells and top is not None:
            for other, rec in self.parts.items():
                if other in own or rec.top_z is None or not (rec.status == "registered" or rec.provisional):
                    continue
                other_cells = self._part_cells(other)
                if other_cells is None or not len(other_cells):
                    continue
                inside = len(cells & {tuple(c) for c in np.asarray(other_cells, int).tolist()}) / len(cells)
                if inside < SHARED_REGION_IOU_MIN:
                    continue
                tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(top.sigma), float(rec.top_z.sigma)))
                dz = float(top.value) - float(rec.top_z.value)
                thresholds.append(th_value("PO3_inside_other_object", SHARED_REGION_IOU_MIN, "DD",
                                           "MC-R3 PO3: a part's cells inside another object's part region at the "
                                           "same surface (D8) registered that object", measured=inside, other=other,
                                           dz_m=dz, tol_m=tol))
                if abs(dz) <= tol:
                    return {"why": "region_of_other_object", "other": other, "inside": inside, "dz_m": dz,
                            "tol_m": tol}, thresholds
        for a, obj in sorted(self._grip_pairs):
            if a != pid or obj == pid:
                continue
            start = self._start_record.get(obj)
            if start is None:
                continue
            if xyz is not None and start.region_xy is not None and len(start.region_xy) >= 3:
                ext = np.asarray(start.region_xy, dtype=float)
                r_ext = float(np.max(np.linalg.norm(ext - ext.mean(axis=0)[None, :], axis=1)))
                distance = _poly_distance(ext, np.asarray(xyz, dtype=float)[:2])
                limit = r_ext + PB_REACH_M
                thresholds.append(th_value("PO1s_object_reach_m", limit, "DD", "MC-R3 PO1s: a grip sub-part's point "
                                           "lies within ext_start(obj) + PB_REACH_M of its object's start-view region",
                                           measured=distance, obj=obj, ext_start_m=r_ext, reach_m=PB_REACH_M))
                if distance > limit:
                    return {"why": "far_from_object", "obj": obj, "distance_m": distance, "limit_m": limit,
                            "ext_start_m": r_ext}, thresholds
            start_cells = self._record_cells(start)
            if cells and start_cells is not None and len(start_cells):
                ratio = len(cells) / len(start_cells)
                thresholds.append(th_value("PO5_subpart_region_ratio", SUBPART_REGION_MAX_RATIO, "RULING",
                                           "MC-R3 PO5 (lead ruling L5): a grip sub-part's region holds at most "
                                           "SUBPART_REGION_MAX_RATIO x its object's start-view cells", measured=ratio,
                                           obj=obj, cells=len(cells), object_cells=int(len(start_cells))))
                if ratio > SUBPART_REGION_MAX_RATIO:
                    return {"why": "larger_than_object", "obj": obj, "ratio": ratio, "cells": len(cells),
                            "object_cells": int(len(start_cells))}, thresholds
        return None, thresholds

    def _po6_check(self, pid, old, new) -> tuple[PartRecord, dict | None]:
        """MC-R3 PO6 (the upward twin of S-14): a grip sub-part's re-registration in the same epoch (X-1: the part was
        not disturbed since its record) whose top lies ABOVE the earlier whole record's top by more than max(D8_FLOOR_M,
        D8_SIGMA_K hypot sigma) keeps the earlier record whole (lead ruling 2) and logs top_inconsistent. An
        object-level part (no grip pair) or a disturbed part decides alone. Returns (record, check | None)."""
        sub_part = any(a == pid and b != pid for a, b in self._grip_pairs)
        if (not sub_part or old is None or old.status != "registered" or old.partial or old.provisional
                or old.top_z is None or new.top_z is None or new.partial
                or self._record_epoch.get(pid) != self.epoch_of(pid)):
            return new, None
        tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(old.top_z.sigma), float(new.top_z.sigma)))
        jump = float(new.top_z.value) - float(old.top_z.value)
        kept = jump > tol
        check = {"kept": bool(kept), "earlier_top_z": float(old.top_z.value), "measured_top_z": float(new.top_z.value),
                 "jump_m": jump, "tol_m": tol, "epoch": self.epoch_of(pid),
                 "threshold": th_value("PO6_top_jump_m", tol, "DD", "MC-R3 PO6: an undisturbed sub-part's "
                                       "re-registration may not read its top above the earlier record by more than "
                                       "max(D8_FLOOR_M, D8_SIGMA_K x hypot sigma)", measured=jump, kept=bool(kept))}
        return (old if kept else new), check

    def _band_component(self, seg, band_index, seed_uv):
        """MC-R2a / R2b: the part band's connected label component at the seed (4-connected; the scene's fragmented
        region set), or None when the band has no labelled pixels."""
        labels = seg.get("labels") if isinstance(seg, Mapping) else None
        if labels is None or band_index is None:
            return None
        whole = np.asarray(labels) == band_index
        if not whole.any():
            return None
        _, comps = cv2.connectedComponents(whole.astype(np.uint8), connectivity=4)
        u, v = _nearest_mask_pixel(whole, np.asarray(seed_uv, dtype=float))
        return comps == comps[int(v), int(u)]

    def _annulus(self, component, pose, top, kappa) -> dict | None:
        """MC-R2b (gap round, judge decision §2.2): the annulus model of the band component (geometry
        ``annulus_fit`` on its area points at the top height, D11_OUTLIER_K / D11_MIN_POINTS). Accepted when the arc
        >= ANNULUS_MIN_ARC_DEG, the wall t <= RIM_WALL_RADIUS_FRACTION r, the centre pixel lies outside the component
        and the radial rms <= t. None without the kernel or the points; else the fit, the checks and thresholds."""
        fit_fn = getattr(self.vision, "annulus_fit", None)
        pts = self._region_points(component, pose, float(top.value))
        if fit_fn is None or pts is None or len(pts) < D11_MIN_POINTS:
            return None
        fit = fit_fn(pts, outlier_k=D11_OUTLIER_K, min_points=D11_MIN_POINTS)
        if fit is None:
            return {"accepted": False, "why": "no_fit", "thresholds": []}
        centre, radius, wall = np.asarray(fit["centre"], dtype=float), float(fit["radius_m"]), float(fit["wall_m"])
        uv, _ = _project(self.camera, pose, np.r_[centre, float(top.value)][None, :])
        u, v = (int(round(float(uv[0][0]))), int(round(float(uv[0][1])))) if np.isfinite(uv).all() else (-1, -1)
        h, w = component.shape
        centre_inside = bool(0 <= v < h and 0 <= u < w and component[v, u])
        checks = {"arc": float(fit["arc_deg"]) >= ANNULUS_MIN_ARC_DEG,
                  "wall": wall <= RIM_WALL_RADIUS_FRACTION * radius,
                  "centre_outside": not centre_inside, "rms": float(fit["rms_m"]) <= wall}
        accepted = all(checks.values())
        thresholds = [th_value("annulus_arc_deg", ANNULUS_MIN_ARC_DEG, "DD", "MC-R2b: the annulus arc spans >= "
                               "ANNULUS_MIN_ARC_DEG", measured=float(fit["arc_deg"])),
                      th_value("annulus_wall_fraction", RIM_WALL_RADIUS_FRACTION, "DD", "MC-R2b: the annulus wall t <= "
                               "RIM_WALL_RADIUS_FRACTION x r", measured=wall / max(radius, 1e-9))]
        return {"accepted": bool(accepted), "checks": checks, "centre": centre, "radius_m": radius, "wall_m": wall,
                "rms_m": float(fit["rms_m"]), "rms_px": float(fit["rms_m"]) * float(kappa),
                "arc_deg": float(fit["arc_deg"]), "n": int(fit["n"]), "thresholds": thresholds}

    @staticmethod
    def _annulus_widths(decided, annulus, xyz, kappa) -> dict:
        """MC-R2b: the radial-yaw width of an accepted annulus is its wall t (the jaw closes across the wall at the
        grip point; the other yaws keep the region's widths)."""
        widths = list(decided["widths"])
        offset = np.asarray(xyz, dtype=float)[:2] - np.asarray(annulus["centre"], dtype=float)
        if not widths or not float(np.linalg.norm(offset)) > 0.0:
            return decided
        radial = math.atan2(offset[1], offset[0]) % math.pi
        k = min(range(len(widths)), key=lambda i: min(abs(float(widths[i][0]) % math.pi - radial),
                                                      math.pi - abs(float(widths[i][0]) % math.pi - radial)))
        sigma = math.hypot(2.0 * TRACK_SIGMA_PX / max(float(kappa), 1e-9), W_MODEL_M)
        widths[k] = (widths[k][0], Meas(float(annulus["wall_m"]), sigma, "kappa"))
        return dict(decided, widths=tuple(widths), width_source="annulus")

    def _kept_whole(self, pid, xyz, top, sigma_lat) -> tuple[PartRecord | None, dict | None]:
        """MC-R2a (gap round, judge decision §2.2; lead ruling L6): a collapsed point registration keeps the last
        non-collapsed point registration of the part when the epoch is unchanged (X-1), the new grip point lies in
        its region within HF_CELL_M + REGION_CHOICE_SIGMA_K sigma_lat and the top is the same surface (|dtop| <=
        max(D8_FLOOR_M, D8_SIGMA_K hypot sigma)). Returns (the kept record or None, log)."""
        whole = self._whole.get(pid)
        if whole is None:
            return None, {"kept": False, "why": "no_whole_record"}
        if whole["epoch"] != self.epoch_of(pid):
            return None, {"kept": False, "why": "disturbed"}
        rec = whole["record"]
        if rec.region_xy is None or len(rec.region_xy) < 3 or rec.top_z is None:
            return None, {"kept": False, "why": "no_region"}
        limit = HF_CELL_M + REGION_CHOICE_SIGMA_K * float(sigma_lat)
        distance = _poly_distance(rec.region_xy, np.asarray(xyz, dtype=float)[:2])
        tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(top.sigma), float(rec.top_z.sigma)))
        dtop = abs(float(top.value) - float(rec.top_z.value))
        kept = distance <= limit and dtop <= tol
        log = {"kept": bool(kept), "why": None if kept else ("grip_outside" if distance > limit else "top_differs"),
               "whole_step": whole["step"], "distance_m": distance, "limit_m": limit, "dtop_m": dtop, "tol_m": tol}
        return (rec if kept else None), log

    def _d10_oe(self, mask, images, pose, z, point_band, *, edges: bool = True) -> dict:
        """S-11 (decision 2 §2.3), D10-OE: the axis lens's outline + edges consensus on G-7's vision kernels.

        Outline gate: ``vision.outline_orientation(mask, pose, z, f)`` (the largest outline back-projected at the
        measured top z) passes when its share >= D10_OUTLINE_SHARE_MIN, its second peak <= D10_OUTLINE_SECOND_MAX
        and it is two-sided. Edge evidence (``edges``; skipped for a collapsed region, S-13, whose axis is refused):
        ``vision.edge_orientation(grey, roi, exclude, pose, z, f)`` in (a) the region dilated
        D10_EDGE_REGION_DILATE_PX and (b) the top-band hull dilated D10_EDGE_HULL_DILATE_PX, the exclusion the
        robot mask (the fingertip band and the rows below it); each ROI gated at share >= D10_EDGE_SHARE_MIN and, lead
        ruling 4 (a) (r2close), at its own second peak <= D10_OUTLINE_SECOND_MAX: an ROI counts only when it passes
        both. The outline axis is accepted within AXIS_AGREE_MAX_RAD of a counting edge axis (the nearer one is
        reported); when only a gated but bimodal ROI agrees the refusal is AXIS_INCONSISTENT "no_edge_agreement" with
        ``edges_bimodal`` True (the 89 deg near-isotropic chain acceptance of review 2: region-ROI second 0.95).
        Returns {"accepted", "reason" ("outline_gate" | "no_edge_agreement" | None), "axis" / "share" (the outline's,
        None without one), "edge_axis", "edge_roi", "agree_rad" (of the counting ROIs), "outline", "outline_ok",
        "edges" (per ROI: axis, share, second, gated (share), unimodal (second), counts, agree_rad), "edges_bimodal",
        "kernels" (False when the vision lacks them: then the outline gate fails), "ms", "thresholds"}. Without the
        kernels nothing is accepted."""
        f = float(self.camera.f_px)
        outline_fn = getattr(self.vision, "outline_orientation", None)
        edge_fn = getattr(self.vision, "edge_orientation", None)
        out = {"accepted": False, "reason": "outline_gate", "axis": None, "share": None, "edge_axis": None,
               "edge_roi": None, "agree_rad": None, "outline": None, "outline_ok": False, "edges": {},
               "edges_bimodal": False, "kernels": outline_fn is not None and edge_fn is not None, "ms": None,
               "thresholds": []}
        if not out["kernels"]:
            return out
        began = time.perf_counter()
        try:
            outline = _orientation(outline_fn(mask, pose, float(z), f))
        except (ValueError, TypeError, IndexError, cv2.error):
            outline = None
        out["outline"] = outline
        second = None if outline is None else (outline["second"] or 0.0)
        ok = bool(outline is not None and outline["share"] >= D10_OUTLINE_SHARE_MIN
                  and second <= D10_OUTLINE_SECOND_MAX and outline["two_sided"])
        out["outline_ok"] = ok
        if outline is not None:
            out.update(axis=outline["axis"], share=float(min(max(outline["share"], 0.0), 1.0)))
        out["thresholds"] += [
            th_value("D10_outline_share", D10_OUTLINE_SHARE_MIN, "DD", "S-11 outline gate: the outline length share "
                     "within YAW_GRID_RAD / 2 of its orientation peak >= D10_OUTLINE_SHARE_MIN, two-sided",
                     measured=None if outline is None else outline["share"],
                     two_sided=None if outline is None else outline["two_sided"]),
            th_value("D10_outline_second", D10_OUTLINE_SECOND_MAX, "DD", "S-11 outline gate: second peak (>= 2 "
                     "windows away) / main peak <= D10_OUTLINE_SECOND_MAX", measured=second)]
        if ok and edges:
            rois = [("region", cv2.dilate(mask.astype(np.uint8), _disk_kernel(D10_EDGE_REGION_DILATE_PX)) > 0)]
            hull = self._members_hull(point_band, pose, D10_EDGE_HULL_DILATE_PX)
            if hull is not None:
                rois.append(("hull", hull))
            best = None
            for name, roi in rois:
                try:
                    edge = _orientation(edge_fn(images["gray"], roi, images["robot"], pose, float(z), f))
                except (ValueError, TypeError, IndexError, cv2.error):
                    edge = None
                entry = {"axis": None, "share": None, "second": None, "gated": False, "unimodal": False,
                         "counts": False, "agree_rad": None}
                if edge is not None:
                    # lead ruling 4 (a): an ROI counts as edge evidence only when it passes its share gate AND its
                    # own histogram passes the outline's second-peak gate (a bimodal edge histogram agreeing with a
                    # near-isotropic outline is no independent evidence of the axis)
                    gated = bool(edge["share"] >= D10_EDGE_SHARE_MIN)
                    unimodal = bool((edge["second"] or 0.0) <= D10_OUTLINE_SECOND_MAX)
                    entry.update(axis=edge["axis"], share=edge["share"], second=edge["second"], gated=gated,
                                 unimodal=unimodal, counts=gated and unimodal,
                                 agree_rad=_axis_angle(outline["axis"], edge["axis"]))
                out["edges"][name] = entry
                if entry["counts"] and (best is None or entry["agree_rad"] < best[0]):
                    best = (entry["agree_rad"], name, edge["axis"])
            if best is not None:
                out.update(agree_rad=float(best[0]), edge_roi=best[1], edge_axis=best[2])
            out["accepted"] = bool(best is not None and best[0] <= AXIS_AGREE_MAX_RAD)
            # a refusal that only a gated but bimodal ROI would have accepted (ruling 4 (a)'s refusals)
            out["edges_bimodal"] = bool(not out["accepted"] and any(
                e["gated"] and not e["unimodal"] and e["agree_rad"] is not None
                and e["agree_rad"] <= AXIS_AGREE_MAX_RAD for e in out["edges"].values()))
            out["thresholds"] += [
                th_value("D10_edge_share", D10_EDGE_SHARE_MIN, "DD", "S-11 edge evidence: an ROI's edge orientation "
                         "share >= D10_EDGE_SHARE_MIN (ROIs: the region dilated D10_EDGE_REGION_DILATE_PX, the "
                         "top-band hull dilated D10_EDGE_HULL_DILATE_PX; the robot mask excluded)",
                         **{f"{name}_share": out["edges"][name]["share"] for name, _ in rois}),
                th_value("D10_edge_second", D10_OUTLINE_SECOND_MAX, "DD", "S-11 edge evidence (lead ruling 4 (a)): an "
                         "ROI's own second peak (>= 2 windows away) / main peak <= D10_OUTLINE_SECOND_MAX",
                         **{f"{name}_second": out["edges"][name]["second"] for name, _ in rois},
                         edges_bimodal=out["edges_bimodal"]),
                th_value("D10_axis_agree_rad", AXIS_AGREE_MAX_RAD, "DD", "S-11 D10-OE: the outline axis within "
                         "AXIS_AGREE_MAX_RAD (YAW_GRID_RAD / 2) of an edge axis that passes its share and second-peak "
                         "gates", measured=out["agree_rad"], roi=out["edge_roi"], accepted=out["accepted"])]
        out["reason"] = None if out["accepted"] else (
            "outline_gate" if not ok else ("no_edge_agreement" if edges else None))
        out["ms"] = 1000.0 * (time.perf_counter() - began)
        return out

    def _axis_decision(self, pid, oe, *, scope, collapsed, collapse_evaluated, widths) -> dict:
        """S-11 + S-12 + S-13 (decision 2 §2.3): the record's axis and width from this view's D10-OE result ``oe``
        (None out of scope: ``scope`` "partial_side" or "area_seed") and the part's view memory (``_views``, since
        the last ``part_moved``).

        - a fresh acceptance (D10-OE accepted, region not collapsed) that agrees within AXIS_AGREE_MAX_RAD with every
          remembered axis is accepted (source "outline_edges") and remembered; one that disagrees with a remembered
          axis, or meets a conflicted memory, is refused AXIS_INCONSISTENT "views_disagree" and the memory is
          marked conflicted (later registrations stay refused until ``part_moved``);
        - a refusal (any reason, collapse included; out-of-scope area seeds too, not a side cut) while the memory
          holds a non-conflicted accepted axis carries the last remembered axis (accepted, source and reason
          "kept_earlier", ``refused_as`` logs this view's own verdict); while the memory is conflicted any refusal is
          AXIS_INCONSISTENT "views_disagree" (``refused_as`` its own verdict);
        - a refusal labels: outline gate failed -> NOT_ELONGATED "outline_gate"; no agreeing edge ->
          AXIS_INCONSISTENT "no_edge_agreement"; a collapsed region whose outline passed -> AXIS_INCONSISTENT
          (reason None); out of scope -> NOT_ELONGATED with the scope as reason (no D10-OE method);
        - width (S-13): a collapsed region delivers the memory's last non-collapsed width set, else () (K-5: width
          unmeasured); otherwise its own widths, remembered when in scope.
        Returns {"check" (the kernel axis_check keys + method, region_collapsed, views, reason, refused_as),
        "elongation", "widths", "width_source", "memory" (the updated entry to commit, None out of scope), "log",
        "thresholds"}."""
        mem = self._views.get(pid)
        axes = [] if mem is None else list(mem["axes"])
        conflicted = bool(mem is not None and mem["conflicted"])
        own = None if oe is None else oe.get("axis")
        disagree = max((_axis_angle(own, a) for a, _, _ in axes), default=None) if own is not None else None
        check = {"image_axis": own, "region_axis": None if oe is None else oe.get("edge_axis"),
                 "agree_rad": None if oe is None else oe.get("agree_rad"), "accepted": False, "source": None,
                 "method": None if oe is None else "outline_edges",
                 "region_collapsed": bool(collapsed) if collapse_evaluated else None, "reason": None,
                 "refused_as": None, "edges_bimodal": None if oe is None else bool(oe.get("edges_bimodal"))}
        elongation = None
        new_axes, new_conflicted = list(axes), conflicted
        if oe is None:
            check.update(source=NOT_ELONGATED, reason=scope)
        elif oe["accepted"] and not collapsed:
            if conflicted or (disagree is not None and disagree > AXIS_AGREE_MAX_RAD):
                check.update(source=AXIS_INCONSISTENT, reason="views_disagree")
                new_conflicted = True
            else:
                check.update(accepted=True, source="outline_edges")
                elongation = (np.asarray(own, dtype=float), float(oe["share"]))
                new_axes.append((np.asarray(own, dtype=float), float(oe["share"]), int(self.step)))
        else:
            reason = oe["reason"]
            check.update(source=NOT_ELONGATED if reason == "outline_gate" else AXIS_INCONSISTENT, reason=reason)
        if not check["accepted"] and axes and not new_conflicted and scope != "partial_side":
            check["refused_as"] = {"source": check["source"], "reason": check["reason"]}
            check.update(accepted=True, source=AXIS_KEPT_EARLIER, reason=AXIS_KEPT_EARLIER)
            last_axis, last_share, _ = axes[-1]
            elongation = (np.asarray(last_axis, dtype=float), float(last_share))
        elif (new_conflicted and new_axes and not check["accepted"] and check["reason"] != "views_disagree"
              and scope != "partial_side"):
            # a conflicted memory: the part's views accepted elongated axes that disagree, so any later refusal is
            # the views' disagreement (AXIS_INCONSISTENT, P-H refine_axis), never NOT_ELONGATED (no RC-7 harmonic)
            check["refused_as"] = {"source": check["source"], "reason": check["reason"]}
            check.update(source=AXIS_INCONSISTENT, reason="views_disagree")
        views = None
        if new_axes or mem is not None:
            views = {"axes": [a for a, _, _ in new_axes], "disagree_rad": disagree, "conflicted": new_conflicted,
                     "steps": [s for _, _, s in new_axes]}
        check["views"] = views
        remembered = None if mem is None else mem.get("widths")
        if collapsed:
            out_widths = remembered if remembered is not None else ()
            width_source = "kept_earlier" if remembered is not None else "unmeasured"
            keep_widths, width_step = remembered, None if mem is None else mem.get("width_step")
        else:
            out_widths, width_source = tuple(widths), "region"
            keep_widths, width_step = (tuple(widths), int(self.step)) if scope is None else (
                remembered, None if mem is None else mem.get("width_step"))
        memory = None
        if scope is None:
            memory = {"axes": new_axes, "conflicted": new_conflicted, "widths": keep_widths, "width_step": width_step}
        thresholds = []
        if axes or oe is not None:
            thresholds.append(th_value("S12_view_agree_rad", AXIS_AGREE_MAX_RAD, "DD",
                                       "S-12 view memory: an accepted axis agrees within AXIS_AGREE_MAX_RAD with every "
                                       "axis accepted for the part since its last part_moved", measured=disagree,
                                       remembered=len(axes), conflicted=bool(new_conflicted),
                                       decided=str(check["source"])))
        log = {"remembered": len(axes), "conflicted_before": conflicted, "conflicted": new_conflicted,
               "disagree_rad": disagree, "width_source": width_source, "scope": scope or "outline_edges"}
        return {"check": check, "elongation": elongation, "widths": out_widths, "width_source": width_source,
                "memory": memory, "log": log, "thresholds": thresholds}

    def _keep_lower_bound(self, old, new):
        """S-14 (decision 2 §2.3, BSR-4 related): a whole registration whose top lies below an existing lower-bound
        top of the same part (``top_bound`` "lower") by more than max(D8_FLOOR_M, D8_SIGMA_K x its sigma_h) does not
        clear the flag: it is logged top_inconsistent and the lower bound and top_upper_z are kept. The view read a
        surface below one the part is known to reach, so none of its geometry is the part's: its xyz and region are
        back-projected at the low top (displaced along the rays), and its widths, axis and support come from that
        region. The earlier record is therefore kept whole (its lower-bound top, top_upper_z, xyz, region, widths,
        axis, support and cov), and this view's reading is only logged in the check (``measured_*``); lead ruling 2
        (r2close) ratifies keeping the WHOLE earlier record. Returns (record, check | None); the record is ``old``
        itself when kept."""
        if (old is None or not old.top_is_lower_bound() or old.top_z is None or new.partial
                or new.top_is_lower_bound() or new.top_z is None):
            return new, None
        tol = max(D8_FLOOR_M, D8_SIGMA_K * float(new.sigma_h_m or 0.0))
        below = float(old.top_z.value) - float(new.top_z.value)
        kept = below > tol
        check = {"kept": bool(kept), "lower_bound_z": float(old.top_z.value), "measured_top_z": float(new.top_z.value),
                 "below_m": below, "tol_m": tol, "top_upper_z": float(old.top_upper_z),
                 "threshold": th_value("S14_lower_bound_m", tol, "DD", "S-14: a whole registration clears a lower-bound "
                                       "top only when it does not read below it by more than max(D8_FLOOR_M, "
                                       "D8_SIGMA_K x sigma_h)", measured=below, sigma_h_m=float(new.sigma_h_m or 0.0),
                                       kept=bool(kept))}
        if not kept:
            return new, check
        check.update(kept_record="earlier", measured_xyz=None if new.xyz is None else [float(v) for v in new.xyz],
                     measured_sigma_h_m=float(new.sigma_h_m or 0.0),
                     measured_support_z=None if new.support_z is None else float(new.support_z.value),
                     measured_elongated=bool(new.elongation is not None),
                     measured_region_cells=0 if new.region_cells is None else int(len(new.region_cells)))
        return old, check

    def part_moved(self, pid: str) -> None:
        """S-12 (decision 2 §2.3): the part (or its object) was touched, grasped, released or carried: its view memory
        (the accepted axes and the last non-collapsed width set) is forgotten, so the next registration decides
        alone. BR calls it after any contact with the part or its object; ``release_held`` and ``drop`` call it for
        the parts they rewrite. It forgets only the view memory (the record stays), so it is also the window reset:
        a conflicted memory refuses every later registration AXIS_INCONSISTENT "views_disagree" (P-H refine_axis,
        BR-1 on a grip part), whatever the view's own verdict, until it is called. Decision 2 §2.6 makes chain ->
        second stage -> BR-1 one memory window; the BR hand-off (for the lead to ratify): call it at the start of each
        hover window of a part (the survey probe does, per hover) so that a conflict found in one window cannot pin
        refine_axis for the rest of the episode."""
        self._views.pop(pid, None)

    # -------------------------------------------------------------------- gap round X-1: disturbance epochs
    def object_parts(self, pid: str) -> set:
        """X-1: the parts of ``pid``'s object: ``pid``, its grip pairs (BS-1 ``_grip_pairs``, either side) and every
        part whose object head (``object_words`` of its text, both known) names the same object (``same_object``).
        Never a part of another object; a part whose text names no object is its own object."""
        out = {pid}
        for a, b in self._grip_pairs:
            if a == pid:
                out.add(b)
            elif b == pid:
                out.add(a)
        record = self.parts.get(pid)
        words = None if record is None else object_words(record.text)
        if words is not None:
            for other, rec in self.parts.items():
                other_words = object_words(rec.text)
                if other_words is not None and same_object(words, other_words):
                    out.add(other)
        return out

    def epoch_of(self, pid: str) -> int:
        """X-1: the part's disturbance epoch (0 until the first ``disturb`` naming any part of its object)."""
        return int(self.epoch.get(pid, 0))

    def disturb(self, obj: str, step: int | None = None, cause: str | None = None) -> int:
        """X-1 (judge decision §2.2, merge of HS-H1, MC-R2a, MC-PO6, S-12): a contact-class event (descent contact,
        GC-1 touch or abort, close, release, put-down, carry start, transit contact) naming any part of ``obj``
        raises the disturbance epoch of every part of that object (``object_parts``); what was measured before
        (HS-H1's body record, MC-R2a's whole record, PO6's unmoved record) no longer describes it. The view memory is
        forgotten by ``part_moved``, which contact paths call as before. No number. Returns the new epoch of
        ``obj``."""
        parts = self.object_parts(obj)
        for pid in parts:
            self.epoch[pid] = self.epoch_of(pid) + 1
            self._body.pop(pid, None)
            self._whole.pop(pid, None)
        self.disturbances.append({"step": int(self.step if step is None else step), "obj": obj,
                                  "cause": cause, "parts": sorted(parts)})
        return self.epoch_of(obj)

    def window_start(self, pid: str) -> None:
        """MC-R2a (judge decision §2.2, amends review-2 S-12, lead ruling L6): the start of a hover window resets
        only the part's axis memory and its conflict flag; the remembered widths, the whole record (MC-R2a), the
        body record (HS-H1), the remembered annulus and the epoch survive until a disturbance."""
        mem = self._views.get(pid)
        if mem is None:
            return
        self._views[pid] = dict(mem, axes=[], conflicted=False)

    def body_record(self, pid: str) -> dict | None:
        """HS-H1: the largest-area whole registration of ``pid`` made in its current epoch (non-partial,
        non-provisional, not S-13 collapsed): {"poly", "cells", "top", "area_m2", "step", "epoch"}, else None."""
        body = self._body.get(pid)
        if body is None or body["epoch"] != self.epoch_of(pid):
            return None
        return body

    def _note_body(self, pid: str, record: PartRecord, collapsed: bool) -> None:
        """HS-H1 bookkeeping after a committed whole registration: keep it when it is the largest of the epoch."""
        if (record is None or record.partial or record.provisional or record.status != "registered" or collapsed
                or record.region_xy is None or len(record.region_xy) < 3):
            return
        area, _ = _polygon_area_centroid(np.asarray(record.region_xy, dtype=float))
        body = self.body_record(pid)
        if body is not None and body["area_m2"] >= area:
            return
        self._body[pid] = {"poly": np.asarray(record.region_xy, dtype=float).copy(),
                           "cells": None if record.region_cells is None else
                           np.asarray(record.region_cells, dtype=int).reshape(-1, 2).copy(),
                           "top": record.top_z, "area_m2": float(area), "step": int(self.step),
                           "epoch": self.epoch_of(pid)}

    def view_memory(self, pid: str) -> dict | None:
        """S-12 diagnosis: a JSON-safe copy of the part's view memory ({"axes", "shares", "steps", "conflicted",
        "widths_mm", "width_step"}) or None."""
        mem = self._views.get(pid)
        if mem is None:
            return None
        return jsonable({"axes": [a for a, _, _ in mem["axes"]], "shares": [s for _, s, _ in mem["axes"]],
                         "steps": [t for _, _, t in mem["axes"]], "conflicted": mem["conflicted"],
                         "widths_mm": None if mem["widths"] is None else
                         [[math.degrees(y), w.value * 1000.0] for y, w in mem["widths"]],
                         "width_step": mem["width_step"]})

    def _region_points(self, mask, pose, z):
        vs, us = np.nonzero(mask)
        if not len(us):
            return None
        stride = max(1, int(len(us) // REGION_PCA_MAX_POINTS))
        pts = _backproject_z(self.camera, pose, np.column_stack([us[::stride], vs[::stride]]).astype(float), z)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts[:, :2]

    def _circle(self, mask, support_region, support, pose, images, top_z):
        """D11: a circle fitted to the outline of the bounded support region that encloses the part (the round base
        of a rotary part), else to the part's own outline; (circle, info, threshold)."""
        candidates = []
        if support_region is not None and support is not None:
            cut = self._cut_fraction(support_region, images["robot"])
            if cut < PARTIAL_CUT_FRACTION:
                candidates.append(("support", support_region | mask, support.value, mask))
        candidates.append(("part", mask, top_z, None))
        info = []
        for name, region, z, exclude in candidates:
            kernel = _disk_kernel(MARKER_RADIUS_PX)
            region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
            h, w = region.shape
            keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
            near_robot = cv2.dilate(images["robot"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            keep &= ~near_robot[pts[:, 1], pts[:, 0]]
            if exclude is not None:
                near_part = cv2.dilate(exclude.astype(np.uint8), _disk_kernel(MARKER_RADIUS_PX + 1)) > 0
                keep &= ~near_part[pts[:, 1], pts[:, 0]]
            pts = pts[keep].astype(float)
            world = _backproject_z(self.camera, pose, pts, z)[:, :2] if len(pts) else np.zeros((0, 2))
            fit = _fit_circle(world)
            if fit is None:
                info.append({"candidate": name, "points": int(len(world)), "accepted": False})
                continue
            centre, radius, rms = fit
            for _ in range(2):                    # robust refit: drop points beyond 3 robust sigma, twice
                resid = np.linalg.norm(world - centre, axis=1) - radius
                inliers = np.abs(resid) <= D11_OUTLIER_K * max(_robust_sigma(resid), 1e-9)
                if inliers.sum() < D11_MIN_POINTS or inliers.all():
                    break
                refit = _fit_circle(world[inliers])
                if refit is None:
                    break
                centre, radius, rms = refit
                world = world[inliers]
            depth = max(float(pose.camera[2] - z), 1e-6)
            kappa = self.camera.f_px / depth
            rms_px, radius_px = rms * kappa, radius * kappa
            threshold = th_max("D11_rms_px", Term("DD", D11_RMS_FLOOR_PX, "D11_RMS_FLOOR_PX"),
                               Term("ONLINE", D11_RMS_FRACTION * radius_px, "D11_RMS_FRACTION*r_px"), unit="px")
            ok = rms_px <= threshold.value
            encloses = bool(np.linalg.norm(centre - _polygon_area_centroid(
                self._mask_polygon(mask, pose, top_z) if name == "support" else world)[1]) < radius)
            accepted = bool(ok and (name == "part" or encloses))
            info.append({"candidate": name, "points": int(len(world)), "centre": centre, "radius_m": radius,
                         "rms_px": rms_px, "accepted": accepted})
            if accepted:
                return (np.asarray(centre, dtype=float), float(radius), float(rms_px)), info, threshold
        return None, info, None

    def _rigid_patch(self, mask, images, pose, xyz, z, view):
        """Spec §4.5 step 8: the most textured patch of the region >= RIGID_PATCH_MIN_DIST_M from the grip point."""
        gray = images["gray"]
        r = _NCC_RADIUS
        inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        vs, us = np.nonzero(inner)
        if not len(us):
            return {"unmeasurable": "too_small"}
        pts = _backproject_z(self.camera, pose, np.column_stack([us, vs]).astype(float), z)
        dist = np.linalg.norm(pts[:, :2] - np.asarray(xyz[:2])[None, :], axis=1)
        far = np.isfinite(dist) & (dist >= RIGID_PATCH_MIN_DIST_M)
        if not far.any():
            return {"unmeasurable": "too_small"}
        k = (2 * r + 1, 2 * r + 1)
        mean = cv2.boxFilter(gray, -1, k)
        sq = cv2.boxFilter(gray * gray, -1, k)
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        cand = np.flatnonzero(far)
        i = int(cand[int(np.argmax(std[vs[cand], us[cand]]))])     # the most textured patch (NCCTrack min_std decides)
        uv = (float(us[i]), float(vs[i]))
        try:
            track = NCCTrack(gray, uv)
        except ValueError:
            return {"unmeasurable": "textureless"}
        return {"uv": [uv[0], uv[1]], "track": track, "pose": tuple(float(v) for v in view.state),
                "dist_m": float(dist[i])}

    def _external_projection(self, xyz, poly, z, view):
        projector = self.projector
        if projector is None:
            return None
        try:
            uv, sigma = projector.project(np.asarray(xyz, dtype=float))
            poly3 = np.column_stack([poly, np.full(len(poly), float(z))])
            puv, _ = projector.project_polygon(poly3)
        except (ValueError, TypeError, AttributeError):
            return None
        template = None
        try:
            gray = cv2.cvtColor(np.ascontiguousarray(view.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
            template = NCCTrack(gray, (float(uv[0]), float(uv[1])))
        except (ValueError, TypeError):
            template = None
        return {"uv": [float(uv[0]), float(uv[1])], "sigma_px": float(sigma),
                "polygon_uv": np.asarray(puv, dtype=float).tolist(), "template": template}

    def _failure(self, pid, reason, evidence, thresholds, *, keep_record: bool = False) -> RegistrationFailure:
        """A RegistrationFailure; a seeded record becomes "failed" unless ``keep_record`` (BS-1: the refused part
        keeps its previous record unchanged)."""
        failure = RegistrationFailure(pid, reason, jsonable(evidence), tuple(thresholds))
        record = self.parts.get(pid)
        if (not keep_record and record is not None and record.status != "registered"
                and not record.provisional):
            self.parts[pid] = dataclasses.replace(record, status="failed", reason=reason)
        self.last_registration[pid] = jsonable({"pid": pid, "source": None, "status": "failed", "reason": reason,
                                                "evidence": evidence, "thresholds": thresholds_dicts(thresholds)})
        self._registration_thresholds[pid] = list(thresholds)
        return failure

    def _payload(self, record: PartRecord, source, thresholds, *, extra=None) -> dict:
        sigma_xyz = None if record.cov is None else np.sqrt(np.clip(np.diag(record.cov), 0.0, None))
        payload = {"pid": record.pid, "source": source, "status": record.status, "reason": record.reason,
                   "xyz": record.xyz, "sigma_xyz": sigma_xyz,
                   "top_z": None if record.top_z is None else record.top_z.to_dict(),
                   "support_z": None if record.support_z is None else record.support_z.to_dict(),
                   "width_mm": None if record.width_by_yaw is None else
                   [[math.degrees(yaw), w.value * 1000.0, w.sigma * 1000.0] for yaw, w in record.width_by_yaw],
                   "elongation": None if record.elongation is None else {"axis": record.elongation[0],
                                                                         "coherence": record.elongation[1]},
                   "circle": None if record.circle is None else {"centre": record.circle[0], "radius_m":
                                                                 record.circle[1], "rms_px": record.circle[2]},
                   "rigid_patch": None if record.rigid_patch is None else {k: v for k, v in record.rigid_patch.items()
                                                                           if k != "track"},
                   "registered_empty_hand": record.registered_empty_hand, "provisional": record.provisional,
                   "partial": record.partial, "sigma_h_m": record.sigma_h_m, "step": self.step,
                   "thresholds": thresholds_dicts(thresholds)}
        payload.update(extra or {})
        return jsonable(payload)

    def registration_thresholds(self, pid: str) -> list[Threshold]:
        """The Threshold records of the last register call of ``pid`` (for the registered outcome, AC-6)."""
        return list(self._registration_thresholds.get(pid, ()))

    def needs_survey_hover(self, pid: str) -> bool:
        """Rulings item 8 / decision §2.6: True when BR must run ``survey_hover(pid)`` before relying on the part
        (e.g. before ``grasp(C, obj=pid)``): its registration is partial or its top only a lower bound. A whole
        re-registration from a view that sees the part clears both."""
        part = self.parts.get(pid)
        return part is not None and (bool(part.partial) or part.top_is_lower_bound())

    def partial_parts_in_view(self, view: StepView) -> list[str]:
        """[C-38] partial parts whose whole registered outline now projects inside the view (re-register them)."""
        if self.camera is None:
            return []
        pose = HandPose.from_view(view)
        out = []
        for pid, part in self.parts.items():
            if not part.partial or part.region_xy is None or part.top_z is None:
                continue
            ext = self._extents.get(pid, {})
            poly = np.asarray(ext.get("polygon", part.region_xy), dtype=float)
            grown = poly_inflate(poly, part.sigma_h_m or 0.0)
            uv, _ = _project(self.camera, pose, np.column_stack([grown, np.full(len(grown), part.top_z.value)]))
            images_robot = self._robot_mask((self.camera.size_px, self.camera.size_px))
            ok = _inside(uv, images_robot.shape)
            if ok.all() and not images_robot[_pix(uv)[:, 1], _pix(uv)[:, 0]].any():
                out.append(pid)
        return out

    # ------------------------------------------------------------------------------------------- predictions
    def _current_xyz(self, pid, pose: HandPose):
        if pid in self._held_local:
            return pose.grip + pose.R @ self._held_local[pid]
        part = self.parts.get(pid)
        return None if part is None or part.xyz is None else np.asarray(part.xyz, dtype=float)

    def predicted_uv(self, pid: str, view: StepView, camera: str = "wrist") -> tuple[np.ndarray, float] | None:
        """Re-acquisition prediction (§3.6 item 6): the part's 3-D point projected into ``camera`` at the view's pose
        with its sigma in px (the 3-D covariance through the projection Jacobian; external adds the projector's
        affine term). A part with only seeds is lifted with estimate_seed_xyz first (its wrist seed, else its external
        seed)."""
        if self.camera is None or camera not in _CAMERAS:
            return None
        part = self.parts.get(pid)
        if part is None:
            return None
        pose = HandPose.from_view(view)
        xyz = self._current_xyz(pid, pose)
        cov = part.cov
        if xyz is None:
            seed = self._best_seed(part)
            lifted = None if seed is None else self.estimate_seed_xyz(seed, view)
            if lifted is None:                    # no wrist seed (plan w = 0) or none liftable: the external seed
                ext = self._best_seed(part, "external")
                lifted = None if ext is None else self._external_seed_xyz(ext)
            if lifted is None:
                return None
            xyz, cov = lifted
        cov = np.zeros((3, 3)) if cov is None else np.asarray(cov, dtype=float)
        if camera == "wrist":
            uv, z = _project(self.camera, pose, xyz)
            if not np.isfinite(uv).all():
                return None
            J = self._wrist_jacobian(pose, xyz)
            sigma = math.sqrt(max(float(np.linalg.eigvalsh(J @ cov @ J.T).max()), 0.0))
            return uv[0], sigma
        if self.projector is None:
            return None
        try:
            uv, sigma_proj = self.projector.project(np.asarray(xyz, dtype=float))
        except (ValueError, TypeError, AttributeError):
            return None
        J = getattr(getattr(self.projector, "model", None), "J", None)
        extra = 0.0
        if J is not None:
            J = np.asarray(J, dtype=float)
            extra = max(float(np.linalg.eigvalsh(J @ cov @ J.T).max()), 0.0)
        return np.asarray(uv, dtype=float), math.sqrt(float(sigma_proj) ** 2 + extra)

    def _wrist_jacobian(self, pose, xyz):
        d = np.asarray(xyz, dtype=float) - pose.camera
        a, b, z = float(d @ pose.R[:, 1]), float(d @ pose.R[:, 0]), float(d @ pose.R[:, 2])
        f = self.camera.f_px
        du = f * (pose.R[:, 1] / z - a * pose.R[:, 2] / (z * z))
        dv = -f * (pose.R[:, 0] / z - b * pose.R[:, 2] / (z * z))
        return np.vstack([du, dv])

    def reacquire_part(self, pid: str, view: StepView) -> dict:
        """Zero-call re-acquisition (§3.6 item 6, V3): predict the pixel, rotate / scale the part's template by the
        measured yaw change and kappa ratio, search around the prediction (geometry.reacquire); on success a
        "reacquire" seed is added. Returns {"ok", "uv", "peak", "predicted_uv", "sigma_px"}."""
        part = self.parts.get(pid)
        pred = self.predicted_uv(pid, view)
        if part is None or pred is None or part.track is None or part.track_pose is None:
            return {"ok": False, "uv": None, "peak": None, "predicted_uv": None if pred is None else pred[0],
                    "sigma_px": None if pred is None else pred[1]}
        uv_pred, sigma = pred
        pose, then = HandPose.from_view(view), HandPose.from_state8(part.track_pose)
        yaw = self.robot.yaw_about_approach(then.quaternion_xyzw, pose.quaternion_xyzw, then.approach)
        z_then = float((np.asarray(part.xyz) - then.camera) @ then.R[:, 2])
        z_now = float((np.asarray(part.xyz) - pose.camera) @ pose.R[:, 2])
        uv, peak = self.vision.reacquire(part.track, np.asarray(view.wrist_gray_upright, dtype=float), uv_pred,
                                         yaw_change_rad=yaw, kappa_ratio=z_then / max(z_now, 1e-6), sigma_px=sigma)
        ok = uv is not None
        if ok:
            seed = Seed("wrist", (float(uv[0]), float(uv[1])), self.step, "reacquire", max(TRACK_SIGMA_PX, 1.0),
                        tuple(float(v) for v in view.state))
            self.parts[pid] = dataclasses.replace(part, seeds=list(part.seeds) + [seed])
        return jsonable({"ok": ok, "uv": uv, "peak": peak, "predicted_uv": uv_pred, "sigma_px": sigma})

    def survey_targets(self, pid: str, view: StepView) -> dict:
        """Seeds and relief probes for a SurveyRecorder at the current pose (spec §4.5 item 2): the part's predicted
        pixel and the four points at +-(w/2 + RELIEF_OFFSET_M) on its principal axes at its support height."""
        part = self.parts.get(pid)
        pred = self.predicted_uv(pid, view)
        out = {"seeds": [], "relief": []}
        if part is None or pred is None:
            return out
        size = (self.camera.size_px, self.camera.size_px)
        robot_mask = self._robot_mask(size)
        uv = pred[0]
        if _inside(uv, size)[0] and not robot_mask[int(round(uv[1])), int(round(uv[0]))]:
            out["seeds"].append((pid, (float(uv[0]), float(uv[1]))))
        pose = HandPose.from_view(view)
        xyz = self._current_xyz(pid, pose)
        if xyz is None:
            return out
        ext = self._extents.get(pid)
        if part.elongation is not None:
            major = np.asarray(part.elongation[0], dtype=float)
            axes = np.vstack([major, [-major[1], major[0]]])
        elif ext is not None:
            axes = np.asarray(ext["axes"], dtype=float)
        else:
            axes = np.vstack([pose.R[:2, 1], pose.R[:2, 0]])
            axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        extents = np.zeros(2)
        if ext is not None:
            poly = np.asarray(ext["polygon"], dtype=float)
            extents = np.array([np.ptp(poly @ axes[0]), np.ptp(poly @ axes[1])])
        z = part.support_z.value if part.support_z is not None else float(xyz[2])
        offset = float(self.vision.RELIEF_OFFSET_M)
        k = 0
        for a in range(2):
            for sign in (1.0, -1.0):
                p = np.r_[np.asarray(xyz[:2]) + sign * (extents[a] / 2.0 + offset) * axes[a], z]
                ruv, _ = _project(self.camera, pose, p)
                if _inside(ruv, size)[0] and not robot_mask[int(round(ruv[0, 1])), int(round(ruv[0, 0]))]:
                    out["relief"].append((f"relief:{pid}:{k}", (float(ruv[0, 0]), float(ruv[0, 1]))))
                k += 1
        return out

    # ---------------------------------------------------------------------------------------------- masks
    def region_mask(self, pid: str, camera: str, view: StepView, *, erode_px: float | None = None):
        """wrist: the region polygon at top_z projected through the wrist camera; external [C-51]: the projector
        (re-anchored by a HandLocator reading when the region lies beyond REANCHOR_RADIUS_M of its anchor and the
        hand is nearer), the affine mask refined by geometry.refine_region_mask at the projected centroid; the
        affine-vs-refined IoU is logged in ``mask_log``. None when the part has no region."""
        part = self.parts.get(pid)
        if part is None or part.region_xy is None or part.top_z is None or self.camera is None:
            return None
        pose = HandPose.from_view(view)
        poly = np.asarray(part.region_xy, dtype=float)
        z = part.top_z.value
        if pid in self._held_local and self.held is not None:
            poly3 = self._held_polygon(pid, pose)
        else:
            poly3 = np.column_stack([poly, np.full(len(poly), z)])
        if camera == "wrist":
            size = (self.camera.size_px, self.camera.size_px)
            uv, _ = _project(self.camera, pose, poly3)
            if not np.isfinite(uv).all():
                return None
            mask = np.zeros(size, dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(uv).astype(np.int32)], 1)
            mask = mask > 0
        elif camera == "external":
            if self.projector is None:
                return None
            self._maybe_reanchor(poly3, pose)
            shape = np.asarray(view.external_rgb_upright).shape[:2]
            affine = np.asarray(self.projector.mask(poly3, shape), dtype=bool)
            centre_uv, sigma = self.projector.project(poly3.mean(axis=0))
            # refinement seeds: the projected centroid and the plan x cell centre when it lies in the affine mask
            seeds = [np.asarray(centre_uv, dtype=float)]
            cell = self._best_seed(part, "external")
            if cell is not None and cell.source == "plan_cell":
                cu, cv = int(round(cell.uv[0])), int(round(cell.uv[1]))
                if 0 <= cv < shape[0] and 0 <= cu < shape[1] and affine[cv, cu]:
                    seeds.append(np.asarray(cell.uv, dtype=float))
            try:
                mask, info = self.vision.refine_region_mask(np.asarray(view.external_rgb_upright), affine,
                                                            np.vstack(seeds) if len(seeds) > 1 else tuple(seeds[0]),
                                                            sigma_px=sigma)
                mask = np.asarray(mask, dtype=bool)
            except (ValueError, TypeError):
                mask, info = affine, {"iou_vs_affine": 1.0, "pixels": int(affine.sum()), "refine": "failed"}
            # sanity: refinement corrects the position (and adds visible sides), it does not grow the region many-fold
            # or move it onto another one; otherwise the affine mask stands, flagged
            sanity = None
            iou_ar = _iou(affine, mask)
            if affine.any() and (mask.sum() > REFINE_MAX_AREA_RATIO * affine.sum() or iou_ar < REFINE_MIN_IOU):
                sanity = {"refine_rejected": True, "refined_px": int(mask.sum()), "iou_affine_refined": iou_ar,
                          "thresholds": thresholds_dicts([
                              th_value("refine_max_area_ratio", REFINE_MAX_AREA_RATIO, "DD",
                                       "refined area <= REFINE_MAX_AREA_RATIO x the affine area",
                                       measured=float(mask.sum()) / float(affine.sum())),
                              th_value("refine_min_iou", REFINE_MIN_IOU, "DD",
                                       "IoU(refined, affine) >= REFINE_MIN_IOU", measured=iou_ar)])}
                mask = affine
            self.mask_log.setdefault(pid, {})["external"] = jsonable({
                "step": self.step, "affine_px": int(affine.sum()), "refined_px": int(mask.sum()),
                "iou_affine_refined": _iou(affine, mask), "refine": info, "sigma_px": sigma, "seeds": len(seeds),
                "sanity": sanity, "flag": None if sanity is None else "refine_rejected"})
        else:
            raise ValueError("camera must be 'wrist' or 'external'")
        if erode_px:
            mask = cv2.erode(mask.astype(np.uint8), _disk_kernel(erode_px)) > 0
        return mask

    def _held_polygon(self, pid, pose):
        local = self._held_local.get(f"{pid}:polygon")
        if local is None:
            part = self.parts[pid]
            return np.column_stack([part.region_xy, np.full(len(part.region_xy), part.top_z.value)])
        return pose.grip[None, :] + local @ pose.R.T

    def _view_step(self, view) -> int:
        """The policy step a cached view was shown at (identity in the recent ring or keyframes), else the current."""
        for s, v in list(self._recent) + list(self._keyframes):
            if v is view:
                return int(s)
        return int(self.step)

    def _locate_order(self, before, after):
        """(located frame, other frame) of a HandLocator pair (``_locate``): the frame the hand moves UP the external
        image from (the image direction of world down, the external model's v_hat, else world z)."""
        locator = getattr(self.cal, "locator", None) if self.cal is not None else None
        model = getattr(locator, "model", None) or getattr(self.projector, "model", None)
        J, v_hat = getattr(model, "J", None), getattr(model, "v_hat", None)
        dp = HandPose.from_view(after).grip - HandPose.from_view(before).grip
        if J is not None and v_hat is not None:
            down = float(np.asarray(v_hat, dtype=float)[:2] @ (np.asarray(J, dtype=float) @ dp)) > 0.0
        else:
            down = float(dp[2]) < 0.0
        return (after, before) if down else (before, after)    # moving down the image: the later, lower frame

    def _locate(self, before, after):
        """(uv, sigma, reason, located view) of one HandLocator reading on a frame pair. The pair is ordered so that
        the hand moves UP the external image from the located frame to the other (the image direction of world down,
        the external model's v_hat, else world z): the silhouette's lowest point then belongs to the located frame
        itself (the locator's sharp case, the validated settled 3 cm move), not to the other frame minus the
        predicted motion. The located grip is that frame's."""
        locator = getattr(self.cal, "locator", None) if self.cal is not None else None
        if locator is None:
            return None, None, "no_locator", None
        first, second = self._locate_order(before, after)
        pose_1, pose_2 = HandPose.from_view(first), HandPose.from_view(second)
        frames = (np.asarray(first.external_rgb_upright), np.asarray(second.external_rgb_upright))
        try:                                      # the located grip is the FIRST frame's (HandLocator.locate)
            uv, sigma = locator.locate(*frames, pose_1, after_pose=pose_2)
        except TypeError:
            uv, sigma = locator.locate(*frames, pose_1)
        except (ValueError, AttributeError):
            uv, sigma = None, None
        reason = None if uv is not None else (getattr(locator, "last_locate", None) or {}).get("reason", "no_reading")
        return uv, sigma, reason, first

    def reanchor_from(self, before: StepView, after: StepView) -> dict:
        """Rulings item 4 (decision §2.2 S-9, §6 4c): a HandLocator reading on an explicit frame pair, BR's settled
        view at the top of the hover-arrival descent (``before``) and the view at the hover after it (``after``). A
        reading is kept (any located (uv, grip) pair is a valid anchor while the external camera is static) and the
        affine projector is re-anchored on it when its current anchor lies beyond REANCHOR_RADIUS_M of the located
        grip [C-51] (the hand works there next); ``region_mask`` re-anchors near a far region from the kept readings.
        The reading is of the frame the hand moved UP the external image from (``_locate``: for a descent, the
        hover view ``after``), so its grip is where the hand is. Returns {"ok", "status" ("reanchored" | "located" |
        "refused" | "no_locator" | "no_hand_motion"), "uv", "sigma_px", "grip", "step", "located" ("before" |
        "after"), "reason", "anchor_dist_m"}; the result is also ``last_reanchor``."""
        pose_b, pose_a = HandPose.from_view(before), HandPose.from_view(after)
        out = {"ok": False, "status": None, "uv": None, "sigma_px": None, "grip": None, "step": None,
               "located": None, "reason": None, "anchor_dist_m": None}
        locator = getattr(self.cal, "locator", None) if self.cal is not None else None
        if locator is None:
            out.update(status="no_locator", reason="no_locator")
        elif _same_pose(pose_b, pose_a):
            out.update(status="no_hand_motion", reason="no_hand_motion")
        else:
            uv, sigma, reason, located = self._locate(before, after)
            out.update(reason=reason)
            if uv is None:
                out["status"] = "refused"
            else:
                step = self._view_step(located)
                grip = np.asarray(HandPose.from_view(located).grip, dtype=float)
                out.update(step=step, grip=grip, located="before" if located is before else "after")
                reading = {"step": step, "uv": np.asarray(uv, dtype=float), "grip": grip, "sigma": float(sigma),
                           "source": "reanchor_from"}
                self._locations = [r for r in self._locations
                                   if r["step"] != step or r.get("source") == "calibration"]
                self._locations.append(reading)
                out.update(ok=True, status="located", uv=reading["uv"], sigma_px=reading["sigma"])
                anchor = getattr(self.projector, "anchor_grip", None)
                if anchor is not None:
                    far = float(np.linalg.norm(np.asarray(anchor, dtype=float)[:2] - grip[:2]))
                    out["anchor_dist_m"] = far
                    if far > REANCHOR_RADIUS_M:
                        self.projector = self.projector.reanchored(reading["uv"], reading["grip"], reading["sigma"])
                        out["status"] = "reanchored"
        self.last_reanchor = jsonable(out)
        self.mask_log.setdefault("projector", {})["reanchor_from"] = self.last_reanchor
        return self.last_reanchor

    def _reanchor_pairs(self, centre, far):
        """Candidate cached frame pairs for a HandLocator reading nearer ``centre`` than the anchor (``far``): any two
        views of the recent ring (rulings item 4: not only consecutive ones, whose 1-step motion of ~1 px the locator
        refuses), the newer one first; for each newer view the older views whose hand moved, settled ones (shown
        twice at one pose, e.g. at the top of a hover-arrival descent) first, then the larger hand motion."""
        recent = list(self._recent)
        poses = [HandPose.from_view(v) for _, v in recent]
        settled = [k > 0 and recent[k][0] - recent[k - 1][0] == 1 and _same_pose(poses[k - 1], poses[k])
                   for k in range(len(recent))]
        done = {r["step"] for r in self._locations}             # views already located (either frame of a pair)
        for a in range(len(recent) - 1, 0, -1):
            if recent[a][0] in done:
                continue
            befores = []
            for b in range(a):
                s_b = recent[b][0]
                if s_b >= recent[a][0] or s_b in done or _same_pose(poses[b], poses[a]):
                    continue
                located = self._locate_order(recent[b][1], recent[a][1])[0]
                grip = poses[b].grip if located is recent[b][1] else poses[a].grip
                if float(np.linalg.norm(centre[:2] - grip[:2])) >= far:
                    continue                          # the located frame's reading is no nearer than the anchor
                motion = float(np.linalg.norm(poses[a].grip - poses[b].grip))
                befores.append((not settled[b], -motion, b))
            if befores:
                b = min(befores)[2]
                yield recent[b], recent[a]

    def _maybe_reanchor(self, poly3, pose):
        """[C-51] re-anchor the affine external projector on a HandLocator reading nearer the region, when the
        region lies beyond REANCHOR_RADIUS_M of the current anchor. The kept readings (the calibration anchor, every
        ``reanchor_from`` reading and earlier scans) are used first; without one near the region, readings are
        taken from cached frame pairs (``_reanchor_pairs``: non-consecutive pairs allowed, a settled earlier view
        preferred; a static hand, e.g. at the grasp pose after the close, gives no reading), at most
        REANCHOR_MAX_TRIES attempts per call, and every successful reading is kept. The kept reading nearest the
        region is used when it is nearer than the current anchor."""
        projector = self.projector
        anchor = getattr(projector, "anchor_grip", None)
        locator = getattr(self.cal, "locator", None) if self.cal is not None else None
        if anchor is None or locator is None:
            return
        centre = np.asarray(poly3, dtype=float).mean(axis=0)
        far = float(np.linalg.norm(centre[:2] - np.asarray(anchor, dtype=float)[:2]))
        log = self.mask_log.setdefault("projector", {})
        if far <= REANCHOR_RADIUS_M:
            return
        dist = lambda r: float(np.linalg.norm(centre[:2] - np.asarray(r["grip"], dtype=float)[:2]))
        best = min(self._locations, key=dist) if self._locations else None
        tries, reasons = 0, []
        if best is None or dist(best) > REANCHOR_RADIUS_M:
            for (s_b, before), (_, after) in self._reanchor_pairs(centre, far):
                if tries >= REANCHOR_MAX_TRIES:
                    break
                tries += 1
                uv, sigma, reason, located = self._locate(before, after)
                if uv is None:
                    reasons.append(reason)
                    continue
                self._locations.append({"step": int(self._view_step(located)), "uv": np.asarray(uv, dtype=float),
                                        "grip": np.asarray(HandPose.from_view(located).grip, dtype=float),
                                        "sigma": float(sigma), "source": "cache_scan"})
                break
            best = min(self._locations, key=dist) if self._locations else None
        if best is None or dist(best) >= far:
            log["reanchor"] = jsonable({"step": self.step, "status": "locator_unavailable", "anchor_dist_m": far,
                                        "tries": tries, "reasons": reasons})
            return
        self.projector = projector.reanchored(best["uv"], best["grip"], best["sigma"])
        log["reanchor"] = jsonable({"step": self.step, "status": "reanchored", "anchor_dist_m": far,
                                    "new_anchor_dist_m": dist(best), "uv": best["uv"], "sigma_px": best["sigma"],
                                    "reading_step": best["step"], "reading_source": best.get("source"),
                                    "tries": tries})

    def hand_projector(self, view: StepView):
        """S-16' item 3 (effect lens fix (c)) [ONLINE]: the external projector for HAND masks: the scene's projector
        re-anchored on the kept HandLocator reading (``_locations``: the calibration anchor, every ``reanchor_from``
        reading and the cache-scan readings) nearest the current grip in 3-D, or the projector itself when its own
        anchor is nearer; never the region-mask projector as such, which [C-51] may have re-anchored near a far
        region. Returns (projector | None, info {"source", "step", "uv", "grip", "sigma_px", "grip_anchor_dist_m"});
        a projector without an anchor (a duck type with ``project`` only) is used as it is (distance None)."""
        base = self.projector
        info = {"source": None, "step": None, "uv": None, "grip": None, "sigma_px": None, "grip_anchor_dist_m": None}
        if base is None:
            return None, info
        grip = np.asarray(HandPose.from_view(view).grip, dtype=float)
        projector = base
        own = getattr(base, "anchor_grip", None)
        own_dist = None if own is None else float(np.linalg.norm(np.asarray(own, dtype=float) - grip))
        best = None
        if self._locations and callable(getattr(base, "reanchored", None)):
            best = min(self._locations, key=lambda r: float(np.linalg.norm(np.asarray(r["grip"], dtype=float) - grip)))
            best_dist = float(np.linalg.norm(np.asarray(best["grip"], dtype=float) - grip))
            if own_dist is not None and own_dist <= best_dist:
                best = None
        if best is not None:
            projector = base.reanchored(best["uv"], best["grip"], best["sigma"])
            info.update(source=best.get("source"), step=best.get("step"))
        else:
            info.update(source="projector")
        anchor = getattr(projector, "anchor_grip", None)
        if anchor is not None:
            info.update(uv=getattr(projector, "anchor_uv", None), grip=np.asarray(anchor, dtype=float),
                        sigma_px=getattr(projector, "sigma_px", None),
                        grip_anchor_dist_m=float(np.linalg.norm(np.asarray(anchor, dtype=float) - grip)))
        return projector, info

    def _hand_components(self, pose: HandPose) -> list[tuple]:
        """S-16' item 1 [ROBOT] (effect lens fix (c)): the hand as three boxes (name, R1 range mm, R0 range mm, z_low,
        z_high), never hulled together: the palm prism = the R7 rectangle (asset hand_footprint_R1_mm / _R0_mm) from
        the R6 palm underside (grip + p_min) up to the hand's z; two finger columns from the R5 fingertip reach (grip
        - t_f) up to the palm underside, R1 from +-q to +-(q + the asset's finger_and_pad_R1 outer edge beyond the
        key, 26.4 mm at every key), R0 +- the largest finger R0 extent (6.14 mm), q = the measured gap / 2."""
        grip_z = float(pose.grip[2])
        palm_z = grip_z + self.robot.p_min_m
        q = max(float(pose.gap_mm), 0.0) / 2.0
        outer, r0 = _finger_column_extent_mm()
        r1_palm, r0_palm = _hand_footprint_mm()
        return [("palm", r1_palm, r0_palm, palm_z, max(float(pose.hand[2]), palm_z)),
                ("finger1", (q, q + outer), (-r0, r0), grip_z - self.robot.t_f_m, palm_z),
                ("finger2", (-q - outer, -q), (-r0, r0), grip_z - self.robot.t_f_m, palm_z)]

    def hand_swept_mask(self, camera: str, view: StepView, yaw_range_rad: tuple[float, float], *,
                        dilate_px: float | None = None):
        """O7 input: the hand at the current grip swept over ``yaw_range_rad`` (signed, about the current R2,
        contract senses) and projected into the camera; consecutive footprints every SWEEP_STEP_RAD are joined by
        their convex hull; dilated with a dilate_px square kernel (None = SWEPT_DILATE_PX = O1_HAND_DILATE_PX).
        ``yaw_range_rad`` is RELATIVE to the current hand yaw (0 = now): [C-26]'s "[current yaw, current yaw + s N
        omega]" is (0, s N omega).

        Wrist camera: the R7 hand footprint from the grip z band (fingertips, R5) up to the palm underside (R6), the
        part of the hand below the camera (unchanged).

        External camera, S-16' (effect lens fix (c); replaces decision 2 S-16): [ROBOT] the palm prism and the two
        finger columns of ``_hand_components``, each hulled separately per consecutive footprint pair (never across
        components); projected with ``hand_projector`` (the kept HandLocator reading nearest the grip); [ONLINE] while
        |grip - anchor| (3-D) < HAND_MASK_LOCAL_RADIUS_M ("local") every hull vertex is grown by its own disc of radius
        max((side - 1) / 2, ceil(2 max(sigma_grip, sigma_vertex))) (sigma_grip = the hand projector's sigma at the
        current grip, sigma_vertex its sigma at the vertex; a circumscribed SWEEP_DISC_SIDES-gon, joined by the pair's
        hull, no square kernel): the BSBP amendment of the spec's single max(4 px, ceil(2 sigma_grip)) kernel, which
        the RS-9 gate refused (the affine scale error across the 20 cm hand). Beyond the radius ("far") the kernel's
        margin per side is max((side - 1) / 2, ceil(2 sigma_far)), sigma_far at the hull's farthest vertex (S-16's
        rule on this hull). ``mask_log["hand_swept"]`` logs the model, both sigmas, the distance, the rule, the
        anchor, "margin_px" (the largest margin), "margin_min_px" (local: the smallest) and "sigma_proj_px" (local:
        sigma_grip; far: sigma_far)."""
        if self.camera is None:
            return None
        pose = HandPose.from_view(view)
        lo, hi = sorted((float(yaw_range_rad[0]), float(yaw_range_rad[1])))
        count = max(1, int(math.ceil((hi - lo) / SWEEP_STEP_RAD)))
        yaws = np.linspace(lo, hi, count + 1)
        anchor_info = None
        if camera == "wrist":
            shape = (self.camera.size_px, self.camera.size_px)
            r1, r0 = _hand_footprint_mm()
            components = [("hand", r1, r0, float(pose.grip[2]) - self.robot.t_f_m,
                           float(pose.grip[2]) + self.robot.p_min_m)]   # the wrist camera sees below its palm only
            projector = None
        elif camera == "external":
            projector, anchor_info = self.hand_projector(view)
            if projector is None:
                return None
            shape = np.asarray(view.external_rgb_upright).shape[:2]
            components = self._hand_components(pose)
        else:
            raise ValueError("camera must be 'wrist' or 'external'")
        mask = np.zeros(shape, dtype=np.uint8)
        jaw0 = pose.R[:2, 1] / max(float(np.linalg.norm(pose.R[:2, 1])), 1e-9)
        down = float(pose.approach[2]) < 0.0
        jaws = []
        for yaw in yaws:
            angle = -yaw if down else yaw            # a +yaw about R2 ~ world -z turns the world-xy jaw by -yaw
            c, s = math.cos(angle), math.sin(angle)
            jaws.append(np.array([c * jaw0[0] - s * jaw0[1], s * jaw0[0] + c * jaw0[1]]))
        hulls = []                                # (uv (n, 2), sigma per vertex (n,) or None) per footprint pair
        for _, r1_mm, r0_mm, z_low, z_high in components:
            prisms = []
            for jaw in jaws:
                poly = _box_xy(pose.grip[:2], jaw, r1_mm, r0_mm)
                prisms.append(np.vstack([np.column_stack([poly, np.full(len(poly), z_low)]),
                                         np.column_stack([poly, np.full(len(poly), z_high)])]))
            pairs = list(zip(prisms[:-1], prisms[1:])) or [(prisms[0], prisms[0])]
            for a, b in pairs:                    # one component's consecutive footprints: never across components
                pts = np.vstack([a, b])
                if projector is None:
                    uv, _ = _project(self.camera, pose, pts)
                    uv, sig = uv[np.isfinite(uv).all(axis=1)], None
                else:
                    projected = [projector.project(p) for p in pts]
                    uv = np.array([np.asarray(q[0], dtype=float) for q in projected])
                    sig = np.array([float(q[1]) if np.isfinite(float(q[1])) else 0.0 for q in projected])
                if len(uv) >= 3:
                    hulls.append((np.clip(uv, -float(max(shape)), 2.0 * float(max(shape))), sig))
        sigma_far = max([0.0] + [float(sig.max()) for _, sig in hulls if sig is not None and len(sig)])
        dilate = SWEPT_DILATE_PX if dilate_px is None else float(dilate_px)
        side = int(round(dilate))
        base_margin = max((side - 1) // 2, 0)
        margin = base_margin
        rule = None
        if projector is not None:
            # S-16' item 2 (ONLINE): sigma_grip = hypot(sigma_anchor, PROJECTOR_AFFINE_FRACTION |J (grip - anchor)|),
            # the hand projector's sigma at the grip; within the local radius every hull vertex takes its own margin
            # max(base, ceil(2 max(sigma_grip, sigma_vertex))) (BSBP amendment, RS-9: the spec's single sigma_grip
            # margin left 1.1-3.5 % of the hand silhouette uncovered at 5 of 10 calibration-yaw hovers, the affine
            # scale error across the 20 cm hand); S-16's far-vertex sigma as one kernel beyond it
            sigma_grip = float(projector.project(np.asarray(pose.grip, dtype=float))[1])
            if not math.isfinite(sigma_grip):
                sigma_grip = 0.0
            dist = anchor_info.get("grip_anchor_dist_m")
            rule = "local" if dist is not None and dist < HAND_MASK_LOCAL_RADIUS_M else "far"
            if rule == "far":
                margin = max(base_margin, int(math.ceil(2.0 * sigma_far)))
            else:
                margin = max(base_margin, int(math.ceil(2.0 * max([sigma_grip] + [float(sig.max()) for _, sig in hulls
                                                                                 if sig is not None and len(sig)]))))
        if rule == "local":
            # per-vertex discs (a circumscribed SWEEP_DISC_SIDES-gon of radius r_v) joined by the pair's convex hull:
            # the Minkowski sum of the hull with each vertex's own margin
            ring = np.array([[math.cos(t), math.sin(t)] for t in
                             np.linspace(0.0, 2.0 * math.pi, SWEEP_DISC_SIDES, endpoint=False)])
            ring = ring / math.cos(math.pi / SWEEP_DISC_SIDES)
            margin_min = None
            for uv, sig in hulls:
                radius = np.maximum(float(base_margin), np.ceil(2.0 * np.maximum(sigma_grip, sig)))
                margin_min = float(radius.min()) if margin_min is None else min(margin_min, float(radius.min()))
                grown = (uv[:, None, :] + radius[:, None, None] * ring[None, :, :]).reshape(-1, 2)
                cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(grown).astype(np.int32)), 1)
            side = 1
        else:
            margin_min = None
            for uv, _ in hulls:
                cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 1)
            if margin > base_margin:
                side = 2 * margin + 1
        if projector is not None:
            q = max(float(pose.gap_mm), 0.0) / 2.0
            outer, r0 = _finger_column_extent_mm()
            self.mask_log["hand_swept"] = jsonable({
                "step": self.step, "camera": camera, "model": "S-16' palm prism + finger columns",
                "components": [c[0] for c in components], "finger_q_mm": q, "finger_outer_r1_mm": outer,
                "finger_r0_mm": r0, "palm_z": [components[0][3], components[0][4]],
                "finger_z": [components[1][3], components[1][4]], "yaw_range_rad": [lo, hi],
                "sigma_grip_px": sigma_grip, "sigma_far_px": sigma_far,
                "sigma_proj_px": sigma_grip if rule == "local" else sigma_far, "grip_anchor_dist_m": dist,
                "local_radius_m": HAND_MASK_LOCAL_RADIUS_M, "rule": rule, "per_vertex": rule == "local",
                "margin_px": margin, "margin_min_px": margin_min, "base_margin_px": base_margin, "side_px": side,
                "anchor": anchor_info})
        if side > 1:                              # a dilate_px square kernel, as O1 / v1 hand_image_motion dilate
            mask = cv2.dilate(mask, np.ones((side, side), np.uint8))
        return mask > 0

    # ------------------------------------------------------------------------------------------ bookkeeping
    def vacate_held(self, held: HeldFrame, step: int) -> int:
        """[C-36] after the lift: the held object's pick-time cells and the D7b fills derived from them are vacated;
        their samples are dropped and the cached views forgotten (the scene changed)."""
        hf, _ = self._require()
        cells = np.zeros((0, 2), int) if held.vacated_cells is None else np.asarray(held.vacated_cells, int)
        cells = cells.reshape(-1, 2)
        for pid in (held.pid, held.grip_pid):
            self.part_moved(pid)                  # S-12: the lifted object's parts move with the hand
        self.disturb(held.pid, step, "carry_start")       # X-1: the lifted object's records describe its old pose
        if not len(cells):
            return 0
        cleared = hf.vacate(cells, int(step))
        self._plane_member[cells[:, 0], cells[:, 1]] = -1
        self._fill_parent[cells[:, 0], cells[:, 1]] = -1
        if len(self._samples["xyz"]):
            # the samples of the vacated cells (registration markers included) left with the object
            lookup = np.zeros(hf.shape, dtype=bool)
            lookup[cells[:, 0], cells[:, 1]] = True
            iy, ix, inside = self._sample_cells()
            gone = np.zeros(len(inside), dtype=bool)
            gone[inside] = lookup[iy[inside], ix[inside]]
            self._samples["alive"] = self._samples["alive"] & ~gone
        self.invalidate_views()
        self._refresh_unconfirmed()
        return int(cleared)

    def plane_fills_by_source(self) -> dict:
        """D7a plane-filled cells counted by the source of the plane they came from ("measured": kappa cells,
        "parallax": the support sweep at start, BS review R7)."""
        if self.hf is None or self._plane_member is None:
            return {}
        out = {}
        fills = (self.hf.source == _PLANE_CODE) & (self._plane_member >= 0)
        for index, plane in enumerate(self.planes):
            n = int((fills & (self._plane_member == index)).sum())
            if n:
                source = plane.get("source", "measured")
                out[source] = out.get(source, 0) + n
        return out

    def derived_fills(self, cells) -> np.ndarray:
        """The obstacle-filled cells whose D7b parent lies in ``cells`` (n,2) ij."""
        cells = np.asarray(cells, int).reshape(-1, 2)
        if self._fill_parent is None or not len(cells):
            return np.zeros((0, 2), int)
        parent = self._fill_parent
        obstacle = self.hf.source == _OBSTACLE_CODE
        lookup = np.zeros(self.hf.shape, dtype=bool)
        lookup[cells[:, 0], cells[:, 1]] = True
        py, px = parent[..., 0], parent[..., 1]
        valid = obstacle & (py >= 0) & (px >= 0)
        hit = np.zeros(self.hf.shape, dtype=bool)
        hit[valid] = lookup[py[valid], px[valid]]
        return np.argwhere(hit)

    def _withdraw_dense(self, pid, record: PartRecord, view: StepView, *, floor=None) -> dict:
        """S-5 DSW (decision §2.2), after a point-seeded registration that returned ``record``: the samples of the
        probe's dense lattice (feature key "dense:<k>") of the surveys whose lattice was seeded at this part (the
        part's point projects within TOP_WINDOW_PX of the lattice's median start pixel in the survey's start pose)
        that lie above support_z + D8_FLOOR_M and outside the registered region dilated by one cell are withdrawn
        from the HeightField: each affected cell is vacated and re-stamped from its remaining samples in their
        original order, then ``fill_unknown`` runs once, so the vacated cells become D7 fills (unverified for
        contacts, RC-1). The withdrawn samples stay registration markers; the part is not re-registered on the
        cleaned field. Grid, relief and start-survey samples are never withdrawn (they measure the neighbours). The
        dense lattice exists to measure the pointed part, so its samples off the part are no obstacle evidence (an
        occlusion-layer template places them along their ray beside the part).

        Lead decision F2 (Wave 2 entry gate, extends S-5): a dense sample of the same lattice INSIDE the registered
        region dilated by one cell (the ring S-5 keeps: within one HF_CELL_M, 5 mm, of the region's cells; the
        registration's own lateral sigma is 2.3 mm median, 3.8 mm max on the stored A / C records, so a sample of the
        part's outline displaced by it stays inside) that reads above the record's own top by more than the D8
        same-surface tolerance, z > top + max(D8_FLOOR_M, D8_SIGMA_K sigma_h), is withdrawn the same way: the part's
        own lattice reads no surface of the part above its measured top (offline on the stored surveys: 146 of the
        170 A and 10 of the 46 C dense samples reading > 15 mm above the visual surface and above that tolerance;
        the other 24 / 36 are S-5's; 10 of the 32 990 others touched, all 10-14 mm above a visual surface
        below the top). A record whose top is only a lower bound (``top_is_lower_bound``: the part may reach
        higher) is never withdrawn by F2 ("near_skipped" "top_lower_bound"). Nor is a record whose top its own region
        contradicts (BS r3 review): when MOST of the lattice samples inside the region cells read above that reference
        (2 n_above > n_region, a majority, no tuned number) the top is in doubt (a lower band taken for the top), so F2
        is skipped ("near_skipped" "top_contradicted"; counts in "near_region") and S-5 alone runs. Without this a top
        read 15 mm low withdrew 49 of the part's own correct lattice samples and turned 10 of its 13 measured region
        cells into fills (most of them support-plane fills); on the 424 stored A / C records the largest share of
        region samples above the reference is 0.026 (4 / 153), and live (BS r3 review re-runs, every part, 1644
        registrations incl. the RS-2 chain) 0.042, so the check never applied there.

        Without a measured support (S-7 left it unmeasured on purpose) the S-5 height reference is the S-7 floor of
        ``floor`` = (z, sigma) of the lowest top-band member: z - max(D8_FLOOR_M, D8_SIGMA_K sigma), the highest
        height an admissible support may have, so the withdrawn set is a subset of what any admissible support would
        withdraw (BS review BSR-7; the decision is silent here, pending the lead's OK); without either S-5 is skipped
        ("off_skipped" "support_unmeasured"; with F2 skipped too the whole step is, "skipped" "support_unmeasured").
        Returns {"dense", "withdrawn" (S-5 + F2), "withdrawn_off" (S-5), "withdrawn_near" (F2), "affected",
        "vacated", "rebuilt", "surveys", "skipped", "off_skipped", "near_skipped", "fill", "reference" ("support" |
        "s7_floor"), "reference_z", "near_reference_z"}, plus "near_region" {"samples", "above"} when F2's premise
        check ran."""
        hf = self.hf
        log = {"dense": 0, "withdrawn": 0, "withdrawn_off": 0, "withdrawn_near": 0, "affected": 0, "vacated": 0,
               "rebuilt": 0, "surveys": 0, "skipped": None, "off_skipped": None, "near_skipped": None, "fill": None,
               "reference": None, "reference_z": None, "near_reference_z": None}
        s = self._samples
        if record.region_cells is None or not len(record.region_cells) or record.xyz is None:
            return dict(log, skipped="no_region")
        if record.support_z is not None:
            log.update(reference="support", reference_z=float(record.support_z.value))
        elif floor is not None:
            log.update(reference="s7_floor",
                       reference_z=float(floor[0]) - max(D8_FLOOR_M, D8_SIGMA_K * float(floor[1])))
        else:
            log["off_skipped"] = "support_unmeasured"
        # F2: the record's own top (a measured top, not a lower bound) plus the D8 same-surface tolerance
        near_tol, sigma_h = None, None
        if record.top_z is None:
            log["near_skipped"] = "no_top"
        elif record.top_is_lower_bound():
            log["near_skipped"] = "top_lower_bound"
        else:
            sigma_h = float(record.sigma_h_m if record.sigma_h_m is not None else record.top_z.sigma)
            near_tol = max(D8_FLOOR_M, D8_SIGMA_K * sigma_h)
            log["near_reference_z"] = float(record.top_z.value) + near_tol
        if log["reference_z"] is None and log["near_reference_z"] is None:
            return dict(log, skipped="support_unmeasured")
        if not s["dense"].any():
            return dict(log, skipped="no_dense_samples")
        point = np.asarray(record.xyz, dtype=float)
        owned = []
        for k, info in enumerate(self._survey_info):
            if not info["dense"] or info["pose0"] is None or info["dense_centre_uv"] is None:
                continue
            uv, _ = _project(self.camera, HandPose.from_state8(info["pose0"]), point)
            if np.isfinite(uv).all() and float(np.linalg.norm(uv[0] - info["dense_centre_uv"])) <= TOP_WINDOW_PX:
                owned.append(k)
        log["surveys"] = len(owned)
        if not owned:
            return dict(log, skipped="no_lattice_at_part")
        candidates = s["dense"] & s["alive"] & ~s["withdrawn"] & np.isin(s["survey"], owned)
        log["dense"] = int(candidates.sum())
        region = np.zeros(hf.shape, dtype=bool)
        cells = np.asarray(record.region_cells, dtype=int).reshape(-1, 2)
        region[cells[:, 0], cells[:, 1]] = True
        grown = _dilate_cells(region, 1)
        iy, ix, inside = self._sample_cells()
        off = np.zeros(len(inside), dtype=bool)
        off[inside] = ~grown[iy[inside], ix[inside]]
        near = np.zeros(len(inside), dtype=bool)
        near[inside] = grown[iy[inside], ix[inside]]
        z = s["xyz"][:, 2]
        withdraw_off = np.zeros(len(inside), dtype=bool)
        if log["reference_z"] is not None:
            withdraw_off = candidates & off & (z > float(log["reference_z"]) + D8_FLOOR_M)
        withdraw_near = np.zeros(len(inside), dtype=bool)
        if log["near_reference_z"] is not None:
            # F2 premise check (BS r3 review): the region's own lattice must not contradict the top it is cleaned
            # against. When most of the lattice samples inside the region cells read above top + tolerance, the record's
            # top is what is in doubt (a lower band taken for the top), not those samples: F2 is skipped and S-5 alone
            # runs, so a mis-registration never erases the part's own measured surface from the planning field
            on_region = np.zeros(len(inside), dtype=bool)
            on_region[inside] = region[iy[inside], ix[inside]]
            n_region = int((candidates & on_region).sum())
            n_above = int((candidates & on_region & (z > float(log["near_reference_z"]))).sum())
            log["near_region"] = {"samples": n_region, "above": n_above}
            if 2 * n_above > n_region:
                log["near_skipped"] = "top_contradicted"
            else:
                withdraw_near = candidates & near & (z > float(log["near_reference_z"]))
        withdraw = withdraw_off | withdraw_near
        log.update(withdrawn=int(withdraw.sum()), withdrawn_off=int(withdraw_off.sum()),
                   withdrawn_near=int(withdraw_near.sum()))
        if not withdraw.any():
            return log
        s["withdrawn"] = s["withdrawn"] | withdraw
        s["marker"] = s["marker"] | withdraw
        keep = s["alive"] & ~s["marker"] & inside
        nx = hf.shape[1]
        cell_id = np.where(inside, iy * nx + ix, -1)
        affected = sorted(set(cell_id[withdraw].tolist()))
        for c in affected:
            ij = (int(c // nx), int(c % nx))
            hf.vacate([ij], self.step)
            rest = np.flatnonzero(keep & (cell_id == c))
            for k in rest:
                hf.stamp_ij(ij, float(s["xyz"][k, 2]), float(s["sigma"][k]), s["source"][k], self.step)
            log["rebuilt" if len(rest) else "vacated"] += 1
            self._plane_member[ij] = -1
            self._fill_parent[ij] = -1
        log["affected"] = len(affected)
        fill = self.fill_unknown(view)
        log["fill"] = {k: fill.get(k) for k in ("plane", "obstacle", "vacated")}
        thresholds = self._registration_thresholds.setdefault(pid, [])
        if log["reference_z"] is not None:
            thresholds.append(th_value(
                "D8_dense_withdrawal_m", D8_FLOOR_M, "DD", "S-5 DSW: dense-lattice samples above support_z + "
                "D8_FLOOR_M outside the region dilated by one cell are withdrawn (without a support: the S-7 floor)",
                support_z=None if record.support_z is None else float(record.support_z.value),
                reference=log["reference"], reference_z=float(log["reference_z"]), dilate_cells=1,
                withdrawn=log["withdrawn_off"]))
        if near_tol is not None:
            thresholds.append(th_value(
                "D8_dense_near_part_m", near_tol, "DD", "lead decision F2 (S-5 extended): dense-lattice samples inside "
                "the region dilated by one cell above the record's top + max(D8_FLOOR_M, D8_SIGMA_K sigma_h) are "
                "withdrawn", top_z=float(record.top_z.value), sigma_h_m=sigma_h,
                reference_z=float(log["near_reference_z"]), dilate_cells=1, withdrawn=log["withdrawn_near"],
                near_skipped=log["near_skipped"], near_region=log.get("near_region")))
        return log

    def part_fill_cells(self, pid: str) -> np.ndarray:
        """S-6 PB (decision §2.2): the (n, 2) ij of the D7b obstacle fills that are the part's own halo, for BR to pass
        to BP's P7 (RC-1d) only. A fill qualifies when its D7b parent (``_fill_parent``, as ``derived_fills``) is a
        MEASURED cell that lies in the part's top band [top - max(D8_FLOOR_M, TOP_BAND_SIGMA_K sigma_h) -
        D8_FLOOR_M, top + D8_SIGMA_K sigma_h + D8_FLOOR_M], is 8-connected to the registered region through measured
        cells raised above the support (z >= support_z + max(D8_FLOOR_M, D8_SIGMA_K sigma_h); without a measured
        support, through band cells) and lies within PB_REACH_M of the record point. D7b copies a neighbour's height
        up to 2 cm around the part, so such fills are the part's height spread, not an obstacle; a separate raised
        object at the same height, not connected through raised cells, keeps its fills. The log is in
        ``part_fill_log[pid]``.

        Lead decision F3 (Wave 2 entry gate, the S-6 connectivity gap, BP finding BPRR2-1): the growth also crosses an
        UNMEASURED cell (unknown or a D7b fill; never a D7a plane fill, which is the support seen) that touches
        (8-connected) a measured raised cell within the reach. One field cell beside measured raised ground is no
        evidence that two surfaces are apart: an occluded or textureless join of a part to its own body leaves such
        cells. Two surfaces count as separate only across observed support (a measured cell at the support level or
        a plane fill) or across an unmeasured cell that no raised measurement touches (a run of >= 3 unmeasured
        cells); the separate object keeps its fills. The bridge cells are never parents (a parent is measured).
        FACT (truth scoring, stored records): the thin grip part's remaining P7 blockers were fills of its own
        object's measured cells at the part's height 46-50 mm away, joined through 1-2 unmeasured cells; under the
        palm at g* the truth stands at most 0.5 mm above the part's top (palm underside +14.9 mm, P7 limit +5.9 mm),
        the fills at +7 to +16 mm stand over the open support, 133 mm below. The log adds ``bridged_cells``."""
        empty = np.zeros((0, 2), dtype=int)
        rec = self.parts.get(pid)
        hf = self.hf
        log = {"fills": 0, "parents": 0, "grown_cells": 0, "bridged_cells": 0, "skipped": None}
        if (hf is None or rec is None or rec.region_cells is None or not len(rec.region_cells) or rec.top_z is None
                or rec.xyz is None or self._fill_parent is None):
            self.part_fill_log[pid] = dict(log, skipped="not_registered")
            return empty
        top, sigma_h = float(rec.top_z.value), float(rec.sigma_h_m or 0.0)
        lo = top - max(D8_FLOOR_M, TOP_BAND_SIGMA_K * sigma_h) - D8_FLOOR_M
        hi = top + D8_SIGMA_K * sigma_h + D8_FLOOR_M
        measured = hf.measured
        with np.errstate(invalid="ignore"):
            band = measured & (hf.z >= lo) & (hf.z <= hi)
            if rec.support_z is not None:
                raised = measured & (hf.z >= float(rec.support_z.value) + max(D8_FLOOR_M, D8_SIGMA_K * sigma_h))
            else:
                raised = band
        ny, nx = hf.shape
        gy, gx = np.mgrid[0:ny, 0:nx]
        centres = np.stack([hf.origin_xy[0] + gx * hf.cell_m, hf.origin_xy[1] + gy * hf.cell_m], axis=-1)
        near = np.linalg.norm(centres - np.asarray(rec.xyz, dtype=float)[None, None, :2], axis=-1) <= PB_REACH_M
        grown = np.zeros(hf.shape, dtype=bool)
        cells = np.asarray(rec.region_cells, dtype=int).reshape(-1, 2)
        grown[cells[:, 0], cells[:, 1]] = True
        allowed = raised & near
        # F3: an unmeasured cell (unknown or a D7b fill, never a D7a plane fill) touching measured raised ground
        bridge = ~measured & (hf.source != _PLANE_CODE) & near & _dilate_cells(allowed, 1)
        crossable = allowed | bridge
        for _ in range(ny * nx):
            step = _dilate_cells(grown, 1) & crossable & ~grown
            if not step.any():
                break
            grown |= step
        parents = grown & band & near
        parent = self._fill_parent
        py, px = parent[..., 0], parent[..., 1]
        valid = (hf.source == _OBSTACLE_CODE) & (py >= 0) & (px >= 0)
        hit = np.zeros(hf.shape, dtype=bool)
        hit[valid] = parents[py[valid], px[valid]]
        out = np.argwhere(hit)
        self.part_fill_log[pid] = jsonable(dict(log, fills=int(len(out)), parents=int(parents.sum()),
                                                grown_cells=int(grown.sum()), bridged_cells=int((grown & bridge).sum()),
                                                band_m=[lo, hi], reach_m=PB_REACH_M,
                                                support_z=None if rec.support_z is None else rec.support_z.value))
        return out

    def part_ring_cells(self, pid: str) -> np.ndarray:
        """S-17 (grasp-refusal remedy 1): the (n, 2) ij of the MEASURED cells (dense / kappa / loom: never a touch, a
        D7 fill or a region cell) within OWN_RING_CELLS (8-connected) of the registered region whose height lies in
        the part's top band, S-6's [top - max(D8_FLOOR_M, TOP_BAND_SIGMA_K sigma_h) - D8_FLOOR_M, top + D8_SIGMA_K
        sigma_h + D8_FLOOR_M]. The mask outline and the 5 mm raster leave the part's own top cells one cell outside its
        region (FACT, lb_smoke_001 truth replay: the grip part's far-edge row read 0.950-0.954 m against a 0.955 m top
        surface inside the cells, outside the region at 512 / 754 / 1074). BR passes them to BP: the P-J section edge
        of the closing sweeps and the GC-1 close gate (P7 never sees them: P-K is not adopted). Empty without a
        registered region or top; the log is in ``part_ring_log[pid]``. No new number beyond the S-5 ring."""
        empty = np.zeros((0, 2), dtype=int)
        rec = self.parts.get(pid)
        hf = self.hf
        if hf is None or rec is None or rec.region_cells is None or not len(rec.region_cells) or rec.top_z is None:
            self.part_ring_log[pid] = {"cells": 0, "skipped": "not_registered"}
            return empty
        top, sigma_h = float(rec.top_z.value), float(rec.sigma_h_m or 0.0)
        lo = top - max(D8_FLOOR_M, TOP_BAND_SIGMA_K * sigma_h) - D8_FLOOR_M
        hi = top + D8_SIGMA_K * sigma_h + D8_FLOOR_M
        region = np.zeros(hf.shape, dtype=bool)
        cells = np.asarray(rec.region_cells, dtype=int).reshape(-1, 2)
        region[cells[:, 0], cells[:, 1]] = True
        with np.errstate(invalid="ignore"):
            band = hf.measured & (hf.source != _TOUCH_CODE) & (hf.z >= lo) & (hf.z <= hi)
        out = np.argwhere(_dilate_cells(region, OWN_RING_CELLS) & band & ~region)
        self.part_ring_log[pid] = jsonable({"cells": int(len(out)), "band_m": [lo, hi], "ring_cells": OWN_RING_CELLS,
                                            "skipped": None})
        return out

    def drop(self, pid: str) -> None:
        """Repoint: status "dropped"; text kept; seeds cleared except the plan cells; measured fields cleared."""
        part = self.parts.get(pid)
        if part is None:
            return
        self.parts[pid] = PartRecord(part.pid, part.text, part.move_prior, part.roles,
                                     seeds=[s for s in part.seeds if s.source == "plan_cell"], dof=part.dof,
                                     status="dropped", reason=None)
        self._extents.pop(pid, None)
        self._area_prior.pop(pid, None)           # BS-1: a dropped part has no area-seeded record to return to
        self._record_kind.pop(pid, None)
        self._whole.pop(pid, None)                # MC-R2a / HS-H1: a repointed part's records are not kept
        self._body.pop(pid, None)
        self.part_moved(pid)                      # S-12: a repointed part starts a new view memory

    def release_held(self, view: StepView | None = None) -> dict:
        """The hand released its object. With the release ``view``, every hand-attached part is written back into its
        record at the release pose first (so parts_uv, predicted_uv and region_mask follow the placed object): xyz =
        grip + R . local; the held object's region_xy and top_z from its hand-attached top outline, support_z = the
        new top minus its measured height (pick-time top - support), cov from sigma_offset (horizontal) and b_low's
        sigma (vertical); registered_empty_hand False, so a step that needs an empty-hand registration re-surveys it
        (the record is "moved by hand", also listed in ``moved``); templates, rigid patch and external projection
        are cleared (taken at the pick pose). Then no held frame and no hand-attached transforms; cached views are
        forgotten (the scene changed). Returns {pid: "moved"} for the parts written."""
        written = {}
        held, local = self.held, self._held_local
        if view is not None and held is not None and local:
            pose = HandPose.from_view(view)
            for pid in [k for k in local if ":" not in k]:
                part = self.parts.get(pid)
                if part is None:
                    continue
                xyz = pose.grip + pose.R @ np.asarray(local[pid], dtype=float)
                # the placed object's top comes from its hand-attached outline: no lower bound (rulings R2, cleared
                # together) and no top band (it described the pick-time top_z)
                changes = {"xyz": xyz, "track": None, "track_pose": None, "rigid_patch": None, "external": None,
                           "registered_empty_hand": False, "view_pose": tuple(float(v) for v in view.state),
                           "registration_step": self.step, "top_bound": None, "top_upper_z": None, "top_band": None}
                sigma_v = math.hypot(float(part.sigma_h_m or 0.0), float(held.b_low.sigma))
                sigma_lat = float(held.sigma_offset_m)
                poly_local = local.get(f"{pid}:polygon")
                if poly_local is not None:
                    pts = pose.grip[None, :] + np.asarray(poly_local, dtype=float) @ pose.R.T
                    top_value = float(np.median(pts[:, 2]))
                    changes.update({"region_xy": pts[:, :2], "region_cells": self.hf.cells_in_polygon(pts[:, :2]),
                                    "top_z": Meas(top_value, sigma_v, part.top_z.source if part.top_z else "kappa")})
                    if part.top_z is not None and part.support_z is not None:
                        height = float(part.top_z.value) - float(part.support_z.value)
                        changes["support_z"] = Meas(top_value - height, math.hypot(sigma_v, part.support_z.sigma),
                                                    part.support_z.source)
                    self._extents.pop(pid, None)
                changes["cov"] = np.diag([sigma_lat ** 2, sigma_lat ** 2, sigma_v ** 2])
                changes["sigma_h_m"] = sigma_v
                new = dataclasses.replace(part, **changes)
                self.parts[pid] = new
                self.moved[pid] = {"step": self.step, "pose": tuple(float(v) for v in view.state)}
                self.last_registration[pid] = self._payload(new, "release", [], extra={"flag": "moved_by_hand"})
                written[pid] = "moved"
            self.invalidate_views()
        for pid in set(written) | ({held.pid, held.grip_pid} if held is not None else set()):
            self.part_moved(pid)                  # S-12: a carried and released part starts a new view memory
        if held is not None:
            self.disturb(held.pid, None, "release")       # X-1: the released object starts a new epoch
        self.held = None
        self._held_local = {}
        return written

    def add_keepout(self, keepout: KeepOut) -> None:
        """Keep-out store, keyed "<kind>:<pid>" (a newer keep-out of the same part and kind replaces the older)."""
        self.keepouts[f"{keepout.kind}:{keepout.pid}"] = keepout

    def summary(self) -> dict:
        return jsonable({
            "scene_version": SCENE_VERSION, "step": self.step,
            "hf": None if self.hf is None else self.hf.summary(),
            "plane_fills_by_source": self.plane_fills_by_source(),
            "planes": [self._plane_summary(p) for p in self.planes],
            "parts": {pid: {"status": p.status, "provisional": p.provisional, "partial": p.partial,
                            "xyz": p.xyz, "top_z": None if p.top_z is None else p.top_z.value,
                            "support_z": None if p.support_z is None else p.support_z.value,
                            "elongated": p.is_elongated(), "registration_step": p.registration_step}
                      for pid, p in self.parts.items()},
            "keepouts": {key: k.to_dict() for key, k in self.keepouts.items()},
            "held": None if self.held is None else self.held.to_dict(),
            "baselines": sorted(self.baselines), "mask_log": self.mask_log,
            "cached_views": {"recent": len(self._recent), "keyframes": len(self._keyframes)},
            "samples": int(len(self._samples["xyz"])),
            "unconfirmed_cells": 0 if self.hf is None else int(np.count_nonzero(getattr(self.hf, "unconfirmed",
                                                                                         False))),
            "region_conflicts": self.region_conflicts})


# ------------------------------------------------------------------------------------------------ band helpers
def _hf_extent(hf: HeightField) -> float:
    """Side length of the HeightField (m)."""
    ny, nx = hf.shape
    return float(max(ny, nx) * hf.cell_m)


def _height_bands(z, sigma, weights, kind=None) -> list[dict]:
    """Height bands (D8) of samples; kind 0 = measured, 1 = a plane fill at its plane's height.

    The measured heights are explained by surface heights with each sample's own sigma (``_height_modes``: a 1-D
    mixture, BIC); each measured sample joins its most probable surface when within max(D8_FLOOR_M, D8_SIGMA_K
    sigma_i) of it; surfaces with fewer than BAND_MIN_SAMPLES measured members are dropped (their samples re-join).
    A plane height within max(D8_FLOOR_M, D8_SIGMA_K sigma / sqrt(n)) of a measured surface is that surface
    (band z = the plane z); fills join their plane's band. Sorted by z."""
    z = np.asarray(z, dtype=float)
    if not len(z):
        return []
    s = np.maximum(np.asarray(sigma, dtype=float), D8_FLOOR_M / D8_SIGMA_K)
    w = np.asarray(weights, dtype=float)
    kind = np.zeros(len(z), int) if kind is None else np.asarray(kind, int)
    measured = np.flatnonzero(kind == 0)
    tol = np.maximum(D8_FLOOR_M, D8_SIGMA_K * s)
    means, mix = (_height_modes(z[measured], s[measured], w[measured]) if len(measured)
                  else (np.zeros(0), np.zeros(0)))
    means, mix = list(means), list(mix)
    assigned = np.full(len(z), -1, dtype=int)
    for _ in range(len(means) + 1):
        assigned[:] = -1
        if not means:
            break
        mu, pi = np.asarray(means), np.asarray(mix)
        zm, sm = z[measured], s[measured]
        score = np.log(np.maximum(pi, 1e-12))[None, :] - 0.5 * ((zm[:, None] - mu[None, :]) / sm[:, None]) ** 2
        best = np.argmax(score, axis=1)
        close = np.abs(zm - mu[best]) <= tol[measured]
        assigned[measured[close]] = best[close]
        support = np.bincount(assigned[measured][assigned[measured] >= 0], minlength=len(means))
        weak = int(np.argmin(support))
        if support[weak] >= BAND_MIN_SAMPLES:
            break
        means.pop(weak)
        mix.pop(weak)
    bands = []
    for index, mean in enumerate(means):
        members = np.flatnonzero(assigned == index)
        if len(members):
            bands.append({"z": float(mean), "sigma": float(np.median(s[members])), "members": members,
                          "measured": int(len(members)), "plane": False})
    for value in np.unique(z[kind == 1]):
        fills = np.flatnonzero((kind == 1) & (z == value))
        merged = None
        for band in bands:
            limit = max(D8_FLOOR_M, D8_SIGMA_K * band["sigma"] / math.sqrt(max(band["measured"], 1)))
            if not band["plane"] and abs(band["z"] - float(value)) <= limit:
                merged = band
                break
        if merged is None:
            merged = {"z": float(value), "sigma": float(np.median(s[fills])), "members": np.zeros(0, int),
                      "measured": 0, "plane": True}
            bands.append(merged)
        merged["z"], merged["plane"] = float(value), True
        merged["members"] = np.r_[merged["members"], fills].astype(int)
    for band in bands:
        band["n"] = int(len(band["members"]))
    return sorted(bands, key=lambda b: b["z"])


def _height_modes(z, sigma, weights):
    """Surface heights of noisy samples (means, mixing weights): a 1-D Gaussian mixture with each sample's own
    measurement sigma (surfaces have no width of their own), K = 1 .. BAND_MAX_COUNT components fitted by EM
    (BAND_EM_ITERATIONS), K chosen by BIC. Unlike a kernel density, a small band beside a large one is kept when its
    samples sit several sigma away (a 7-sample band 3.7 sigma below a 43-sample one)."""
    z = np.asarray(z, dtype=float)
    s = np.asarray(sigma, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(z)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    inv = 1.0 / (s * s)
    best_bic, best = None, None
    for k in range(1, min(BAND_MAX_COUNT, n) + 1):
        means = np.quantile(z, (np.arange(k) + 0.5) / k)
        mix = np.full(k, 1.0 / k)
        for _ in range(BAND_EM_ITERATIONS):
            log_p = np.log(np.maximum(mix, 1e-12))[None, :] - 0.5 * (z[:, None] - means[None, :]) ** 2 * inv[:, None]
            top = log_p.max(axis=1, keepdims=True)
            resp = np.exp(log_p - top)
            resp /= resp.sum(axis=1, keepdims=True)
            resp *= w[:, None]
            mass = resp.sum(axis=0)
            mix = mass / max(float(mass.sum()), 1e-12)
            new = (resp * (z * inv)[:, None]).sum(axis=0) / np.maximum((resp * inv[:, None]).sum(axis=0), 1e-12)
            new = np.where(mass > 1e-9, new, means)
            if float(np.max(np.abs(new - means))) < 1e-9:
                means = new
                break
            means = new
        log_p = np.log(np.maximum(mix, 1e-12))[None, :] - 0.5 * (z[:, None] - means[None, :]) ** 2 * inv[:, None]
        top = log_p.max(axis=1)
        log_l = float(np.sum(w * (top + np.log(np.exp(log_p - top[:, None]).sum(axis=1)))))
        bic = -2.0 * log_l + (2 * k - 1) * math.log(max(float(w.sum()), 2.0))
        if best_bic is None or bic < best_bic:
            keep = mix > 1e-9
            best_bic, best = bic, (means[keep], mix[keep])
    order = np.argsort(best[0])
    return best[0][order], best[1][order]


def _weighted_percentile(z, sigma, q) -> float:
    """The q-th percentile of heights weighted by their inverse variance: the lowest height whose cumulative weight
    reaches q % (a sample with a 10x larger sigma counts 1 % as much; spec §4.5 step 4's percentile over samples of
    unequal precision)."""
    z = np.asarray(z, dtype=float)
    w = 1.0 / np.maximum(np.asarray(sigma, dtype=float), D8_FLOOR_M / D8_SIGMA_K) ** 2
    order = np.argsort(z)
    cum = np.cumsum(w[order])
    k = int(np.searchsorted(cum, float(q) / 100 * float(cum[-1]), side="left"))
    return float(z[order][min(k, len(z) - 1)])


def _top_cluster(z, sigma) -> tuple[float, float, np.ndarray]:
    """The top surface of a set of height samples (spec §4.5 registration step 4 on a sample set): the median of the
    samples within the D8 band of their inverse-variance weighted TOP_PERCENTILE height (a stray, imprecise sample
    above the surface carries little weight); sigma = max(their median sigma, their robust scatter). Returns
    (z, sigma, member indices)."""
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z95 = _weighted_percentile(z, sigma, TOP_PERCENTILE)
    keep = np.flatnonzero(np.abs(z - z95) <= np.maximum(D8_FLOOR_M, D8_SIGMA_K * sigma))
    if not len(keep):
        keep = np.arange(len(z))
    return float(np.median(z[keep])), max(float(np.median(sigma[keep])), _robust_sigma(z[keep])), keep


def _top_model_m(n) -> float:
    """S-3 part-level model term of a band top: TOP_MODEL_M with >= TOP_BAND_SUPPORT_N members, else
    TOP_MODEL_THIN_M (decision §3, ruling 4b)."""
    return TOP_MODEL_M if int(n) >= TOP_BAND_SUPPORT_N else TOP_MODEL_THIN_M


def _top_band(z, sigma) -> tuple[float | None, np.ndarray, dict]:
    """S-1 (decision §2.2): the highest coherent band of height samples. Scanning the samples from the highest down,
    the first z_i with at least TOP_BAND_K samples within b = max(D8_FLOOR_M, TOP_BAND_SIGMA_K x the median sample
    sigma), re-centred once on its members' median (still >= TOP_BAND_K); top = the members' TOP_PERCENTILE height.
    A stray sample above the surface has no TOP_BAND_K partners and is passed over. Returns (top, member indices,
    info) with info {"rule": "band", "n", "stat_m" = max(median sigma, robust scatter) / sqrt(n), "model_m"
    (``_top_model_m``), "scatter_m", "spread_m" = max(median sigma, robust scatter) (one member's height scale),
    "b_m"}. Without such a band it falls back to ``_top_cluster`` (info "rule" "top_cluster", "stat_m" its sigma)
    and, with no samples, returns (None, [], {"rule": None, ...})."""
    z = np.asarray(z, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    info = {"rule": None, "n": 0, "stat_m": None, "model_m": None, "scatter_m": None, "spread_m": None, "b_m": None}
    if not len(z):
        return None, np.zeros(0, dtype=int), info
    b = max(D8_FLOOR_M, TOP_BAND_SIGMA_K * float(np.median(sigma)))
    info["b_m"] = b
    members = None
    if len(z) >= TOP_BAND_K:
        for i in np.argsort(-z, kind="stable"):
            near = np.flatnonzero(np.abs(z - z[i]) <= b)
            if len(near) < TOP_BAND_K:
                continue
            centre = float(np.median(z[near]))
            near = np.flatnonzero(np.abs(z - centre) <= b)
            if len(near) >= TOP_BAND_K:
                members = near
                break
    if members is None:
        top, s, keep = _top_cluster(z, sigma)
        info.update(rule="top_cluster", n=int(len(keep)), stat_m=float(s), model_m=_top_model_m(len(keep)),
                    spread_m=float(s))
        return float(top), np.asarray(keep, dtype=int), info
    zb, sb = z[members], sigma[members]
    n = int(len(members))
    scatter = _robust_sigma(zb)
    spread = max(float(np.median(sb)), scatter)
    info.update(rule="band", n=n, stat_m=spread / math.sqrt(n), model_m=_top_model_m(n), scatter_m=scatter,
                spread_m=spread)
    return float(np.percentile(zb, TOP_PERCENTILE)), members, info


def _nan_median_axis0(stack) -> np.ndarray:
    """The median over axis 0 ignoring NaN (NaN where every entry is NaN), for a short axis 0 (the partners)."""
    ordered = np.sort(stack, axis=0)                                          # NaN sort last
    n = np.isfinite(stack).sum(axis=0)
    lo = np.clip((n - 1) // 2, 0, None)[None]
    hi = np.clip(n // 2, 0, None)[None]
    med = 0.5 * (np.take_along_axis(ordered, lo, axis=0)[0] + np.take_along_axis(ordered, hi, axis=0)[0])
    return np.where(n > 0, med, np.nan)


def _drop_specks(mask, min_px) -> np.ndarray:
    """The mask without its 8-connected groups smaller than min_px pixels."""
    count, comps, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_px)
    return keep[comps]


def _dissenting_samples(bands, uv, kind, components) -> np.ndarray:
    """Measured samples outvoted inside their own D9 segment (bool per sample): in a segment holding >=
    BAND_MIN_SAMPLES measured samples, a sample whose band holds less than half of them is not a marker. An
    occluding-edge feature measures the front surface while its patch centre lies on the surface behind (a raised
    part's edge seen from a pixel of its lower base), and it would flood that segment with the front's band."""
    n = len(uv)
    out = np.zeros(max(n, 1), dtype=bool)
    if n == 0:
        return out
    band_of = np.full(n, -1, dtype=int)
    for b, band in enumerate(bands):
        members = np.asarray(band["members"], dtype=int)
        band_of[members[members < n]] = b
    h, w = components.shape
    px = _pix(uv)
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    seg = np.zeros(n, dtype=int)
    seg[inside] = components[px[inside, 1], px[inside, 0]]
    measured = (np.asarray(kind)[:n] == 0) & (band_of >= 0) & (seg > 0)
    for s_id in np.unique(seg[measured]):
        idx = np.flatnonzero(measured & (seg == s_id))
        if len(idx) < BAND_MIN_SAMPLES:
            continue
        counts = np.bincount(band_of[idx])
        out[idx[counts[band_of[idx]] * 2 < len(idx)]] = True
    return out


def _band_of(bands, z, sigma) -> int | None:
    """Index of the nearest band within the D8 tolerance of a height (None when no band is that close)."""
    best, best_d = None, None
    for index, band in enumerate(bands):
        tol = max(D8_FLOOR_M, D8_SIGMA_K * math.hypot(float(sigma), float(band["sigma"])))
        d = abs(float(z) - float(band["z"]))
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = index, d
    return best


def _bands_summary(bands) -> list[dict]:
    """The registration event's band list (decision 2 S-17: a band without "n" counts its members)."""
    return [{"z": b["z"], "sigma": b["sigma"], "n": int(b.get("n", len(b.get("members", ())))),
             "measured": b.get("measured", 0), "plane": bool(b.get("plane"))} for b in bands]


def _band_log(point_band) -> dict | None:
    """The registration event's view of a point seed's top band (``SceneModel._seed_band``) without its indices."""
    if point_band is None:
        return None
    return {k: v for k, v in point_band.items() if k != "members"}


def _top_band_field(point_band) -> dict:
    """PartRecord.top_band (kernel K-1): {"n", "stat_m", "model_m"} plus the S-1 extras."""
    keys = ("n", "stat_m", "model_m", "scatter_m", "spread_m", "b_m", "n_window", "n_gated", "rule")
    return {k: point_band[k] for k in keys if k in point_band}


def _merge_facets(face, comps, band_mask):
    """A labelled band's face across its own internal intensity edges (bevels, markings, an inset: the band
    watershed already placed the band's outline on the edges between bands, so a D9 edge inside one band is not a
    region edge). An edge-free component of the same band within FACET_BRIDGE_PX of the face (through the band's own
    edge pixels) joins when it is no larger than the face grown so far (FACET_MERGE_RATIO) and is a bevel or a
    marking: THIN (its mean half-width, area / outline length, is at most FACET_BRIDGE_PX) or ENCLOSED (at least
    FACET_ENCLOSED_FRACTION of its outline lies within FACET_BRIDGE_PX of the face). A wide same-height surface beside
    the face (the body a part is attached to, a neighbour at the same height) stays out; repeated until nothing
    joins."""
    areas = np.bincount(comps.ravel())
    face = face.copy()
    area = int(face.sum())
    merged = set(np.unique(comps[face & (comps > 0)]).tolist())
    kernel = np.ones((3, 3), np.uint8)
    band = band_mask.astype(np.uint8)
    for _ in range(int(areas.size)):
        reach = face.astype(np.uint8)
        for _ in range(FACET_BRIDGE_PX):
            reach = cv2.dilate(reach, kernel) & band
        near = cv2.dilate(face.astype(np.uint8), _disk_kernel(FACET_BRIDGE_PX)) > 0
        ids = [int(c) for c in np.unique(comps[(reach > 0) & (comps > 0)]) if int(c) not in merged]
        added = False
        for c in sorted(ids, key=lambda c: areas[c]):
            if areas[c] > FACET_MERGE_RATIO * area:
                continue
            facet = comps == c
            contours, _ = cv2.findContours(facet.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = np.vstack([k.reshape(-1, 2) for k in contours]) if contours else np.zeros((0, 2), int)
            if not len(outline):
                continue
            thin = float(areas[c]) / float(len(outline)) <= FACET_BRIDGE_PX
            enclosed = float(near[outline[:, 1], outline[:, 0]].mean()) >= FACET_ENCLOSED_FRACTION
            if thin or enclosed:
                face |= facet
                area += int(areas[c])
                merged.add(c)
                added = True
        if not added:
            break
    return face


def _clean_mask(mask, seed_uv):
    """Open and close a region mask with the MARKER_RADIUS_PX disk (label jaggies narrower than a marker go) and keep
    the component at (or nearest to) the seed."""
    kernel = _disk_kernel(MARKER_RADIUS_PX)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel) > 0
    cleaned &= cv2.dilate(mask.astype(np.uint8), kernel) > 0
    if not cleaned.any():
        return mask
    count, comps = cv2.connectedComponents(cleaned.astype(np.uint8), connectivity=4)
    if count <= 2:
        return cleaned
    u, v = _nearest_mask_pixel(cleaned, seed_uv)
    return comps == comps[int(v), int(u)]


def _nearest_mask_pixel(mask, uv):
    vs, us = np.nonzero(mask)
    d = (us - float(uv[0])) ** 2 + (vs - float(uv[1])) ** 2
    i = int(np.argmin(d))
    return np.array([float(us[i]), float(vs[i])])


# ------------------------------------------------------------------------------------------- in-hand pose (§4.7)
class _LiftRecorder:
    """ProbeRecorder of the load-test lift (spec §4.7 items 4 and 5): keeps the first and the last frames."""

    def __init__(self, held: HeldFrame, view: StepView):
        self.held = held
        self.first = view
        self.last = view
        self.transitions = 0
        self.stages: dict[str, int] = {}

    def on_transition(self, stage: str, before: StepView, after: StepView) -> None:
        if self.transitions == 0 and before is not None:
            self.first = before
        self.last = after
        self.transitions += 1
        self.stages[stage] = self.stages.get(stage, 0) + 1


class HeldFrameEstimator:
    """Spec §4.7 in-hand pose: the prior at the close, the re-localization before the lift, the lift silhouette and
    external co-motion (M14 Touchdown trigger, M17 offset sigma) and the Touchdown update of b_low."""

    def __init__(self, scene: SceneModel, robot: RobotModel):
        self.scene = scene
        self.robot = robot
        self.last_relocalize: dict | None = None
        self.last_prior: dict | None = None       # HS-H1 / HS-H2 evidence of the last prior_at_close (JSON-safe)

    # ------------------------------------------------------------------------------------------- footprint
    def _footprint(self, obj_pid):
        part = self.scene.parts[obj_pid]
        poly = None if part.region_xy is None else np.asarray(part.region_xy, dtype=float)
        if poly is not None and len(poly) >= 3:
            _, centre = _polygon_area_centroid(poly)
            r_obj = float(np.max(np.linalg.norm(poly - centre[None, :], axis=1)))
            axes, extents = _principal_axes(poly)
            return centre, r_obj, np.vstack([axes, extents]), poly
        if part.xyz is None:
            raise ValueError(f"part {obj_pid} has no registered footprint")
        centre = np.asarray(part.xyz, dtype=float)[:2]
        cov = np.zeros((3, 3)) if part.cov is None else np.asarray(part.cov, dtype=float)
        r_obj = math.sqrt(max(float(np.linalg.eigvalsh(cov[:2, :2]).max()), 0.0))
        return centre, r_obj, None, None

    def _top_z(self, obj_pid):
        part = self.scene.parts[obj_pid]
        if part.top_z is not None:
            return part.top_z.value
        return float(np.asarray(part.xyz, dtype=float)[2])

    def _wrist_visible(self, pose: HandPose, offset_h, centre_xy, top_z) -> bool:
        """P17 geometry: the footprint centre lies on the camera side (offset along R0 >= 0) and, at the object's top
        height, within INNER_FOV_HALF_RAD of the wrist optical axis."""
        if float(offset_h[0]) < 0.0:
            return False
        d = np.r_[np.asarray(centre_xy, dtype=float), top_z] - pose.camera
        along = float(d @ pose.R[:, 2])
        if along <= 0.0:
            return False
        lateral = float(np.linalg.norm(d - along * pose.R[:, 2]))
        return math.atan2(lateral, along) <= INNER_FOV_HALF_RAD

    def _m17(self, measured_m, r_obj):
        return th_max("M17_sigma_offset_m", Term("ONLINE", measured_m, "measured sigma_offset"),
                      Term("DD", M17_FRACTION * r_obj, "M17_FRACTION*r_obj"), unit="m")

    def _body_footprint(self, obj_pid: str, grip_pid: str, pose: HandPose) -> dict:
        """HS-H1 (gap round, judge decision §2.2; lead ruling L11): the held frame from the object's whole body. The
        object's body record (``SceneModel.body_record``: the largest-area whole, non-provisional, non-collapsed
        registration of its current X-1 epoch) replaces the current record's region when it is larger and: an
        appendage grip (grip != obj): it lies within one HF_CELL_M of the grip part's region; a whole grip: it
        contains the grip xy; it overlaps no other object's registered region beyond OWN_RING_CELLS. It is written
        back to the scene record (region, cells and, without a lower-bound top, its top), so ``_attach``, the wrist
        visible mask, the template point and ``_vacated_cells`` use the body. Returns the log."""
        scene = self.scene
        hf = scene.hf
        part = scene.parts.get(obj_pid)
        log = {"footprint": "current", "why": None}
        body = scene.body_record(obj_pid)
        if part is None or body is None or hf is None:
            log["why"] = "no_body_record"
            return log
        current = (_polygon_area_centroid(part.region_xy)[0]
                   if part.region_xy is not None and len(part.region_xy) >= 3 else 0.0)
        log.update(body_step=body["step"], body_area_m2=body["area_m2"], current_area_m2=float(current))
        if not body["area_m2"] > current * (1.0 + 1e-6):
            log["why"] = "current_is_largest"
            return log
        cells = body["cells"] if body["cells"] is not None and len(body["cells"]) else hf.cells_in_polygon(body["poly"])
        mask = np.zeros(hf.shape, dtype=bool)
        if len(cells):
            mask[cells[:, 0], cells[:, 1]] = True
        if grip_pid != obj_pid:
            grip = scene.parts.get(grip_pid)
            near = None
            if grip is not None and grip.region_xy is not None and len(grip.region_xy) >= 3:
                near = hf.cells_in_polygon(poly_inflate(grip.region_xy, hf.cell_m))
            elif grip is not None and grip.xyz is not None:
                ij = hf.index(np.asarray(grip.xyz, dtype=float)[:2])
                near = None if ij is None else np.array([ij], dtype=int)
            adjacent = near is not None and len(near) and bool(mask[near[:, 0], near[:, 1]].any())
            if not adjacent:
                log["why"] = "not_adjacent_to_grip"
                return log
        elif not bool(np.asarray(poly_contains(body["poly"], np.asarray(pose.grip[:2], dtype=float)[None, :]))
                      .reshape(-1)[0]):
            log["why"] = "grip_outside_body"
            return log
        core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=OWN_RING_CELLS) > 0
        own = scene.object_parts(obj_pid)
        for other in scene.parts:
            if other in own:
                continue
            other_cells = scene._part_cells(other)
            if other_cells is not None and len(other_cells) and bool(core[other_cells[:, 0], other_cells[:, 1]].any()):
                log.update(why="overlaps_other_object", other=other)
                return log
        changes = {"region_xy": np.asarray(body["poly"], dtype=float).copy(), "region_cells": np.asarray(cells).copy()}
        if part.top_bound is None and body["top"] is not None:
            changes["top_z"] = body["top"]
        try:
            scene.parts[obj_pid] = dataclasses.replace(part, **changes)
        except ValueError as exc:                 # a record invariant (kept the current record)
            log.update(why="record_invariant", error=str(exc))
            return log
        log["footprint"] = "body"
        return log

    def _owned_template(self, template, view) -> tuple:
        """HS-H2 (gap round, scene part): the held object's external template is object evidence only when its
        (2r+1)^2 NCC window holds no pixel of the hand mask ``hand_swept_mask('external', view, (0, 0))`` (S-16');
        else None with flag ``external_template_on_robot``. Without a hand mask the template is kept (logged)."""
        if template is None:
            return None, {"template": None}
        uv = np.asarray(template.uv, dtype=float)
        info = {"template_uv": [float(uv[0]), float(uv[1])], "flag": None}
        try:
            hand = self.scene.hand_swept_mask("external", view, (0.0, 0.0))
        except (ValueError, TypeError, AttributeError, RuntimeError, IndexError):
            hand = None
        if hand is None:
            info["hand_mask"] = "unavailable"
            return template, info
        hand = np.asarray(hand, dtype=bool)
        r = int(template.radius)
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        window = hand[max(0, v - r):max(0, v + r + 1), max(0, u - r):max(0, u + r + 1)]
        info["hand_px"] = int(window.sum())
        if window.any():
            info["flag"] = "external_template_on_robot"
            return None, info
        return template, info

    def prior_at_close(self, obj_pid: str, grip_pid: str, view: StepView, *, support_z_pick: Meas,
                       gap_mm: float) -> HeldFrame:
        """§4.7 item 1: offset_h0 = registered footprint centre (at the pick support height: the object's bottom
        centre) - grip in hand coordinates (R0, R1, R2); b_low0 = g_close - support_z(pick); wrist_visible by P17.
        Gap round: HS-H1 (``_body_footprint``: the whole-body record; an appendage grip on it takes sigma_offset >=
        M17_FRACTION r_obj), HS-H2 (``_owned_template``) and X-1 (the close disturbs the object after the frame is
        read: ``disturb(obj, cause="close")``). The evidence is ``last_prior``."""
        scene = self.scene
        pose = HandPose.from_view(view)
        body_log = self._body_footprint(obj_pid, grip_pid, pose)
        centre_xy, r_obj, extent_axes, poly = self._footprint(obj_pid)
        top_z = self._top_z(obj_pid)
        centre = np.r_[centre_xy, support_z_pick.value]
        offset_h = pose.R.T @ (centre - pose.grip)
        visible = self._wrist_visible(pose, offset_h, centre_xy, top_z)
        part = scene.parts[obj_pid]
        cov = np.zeros((3, 3)) if part.cov is None else np.asarray(part.cov, dtype=float)
        measured = math.sqrt(max(float(np.linalg.eigvalsh(cov[:2, :2]).max()), 0.0))
        sigma_offset = measured if visible else self._m17(measured, r_obj).value
        if body_log["footprint"] == "body" and grip_pid != obj_pid and sigma_offset < M17_FRACTION * r_obj:
            # HS-H1: an appendage grip on the body footprint: the offset is not measured at the grip
            sigma_offset = M17_FRACTION * r_obj
            body_log["sigma_offset_floor_m"] = float(sigma_offset)
        b_low = Meas(float(pose.grip[2]) - support_z_pick.value, support_z_pick.sigma, support_z_pick.source)
        template = None
        projector = scene.projector
        if projector is not None:
            try:
                uv, _ = projector.project(np.r_[centre_xy, top_z])
                gray = cv2.cvtColor(np.ascontiguousarray(view.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
                template = NCCTrack(gray, (float(uv[0]), float(uv[1])))
            except (ValueError, TypeError, AttributeError):
                template = None
        template, template_log = self._owned_template(template, view)
        held = HeldFrame(pid=obj_pid, grip_pid=grip_pid, grasp_pose=tuple(float(v) for v in view.state),
                         offset_h=offset_h, vacated_cells=None, sigma_offset_m=float(sigma_offset), b_low=b_low,
                         r_obj_m=float(r_obj), extent_axes=extent_axes, tilt_deg=None, wrist_visible=bool(visible),
                         silhouette=None, ext_template=template, gap_ref_mm=float(gap_mm), sigma_carry_mm=None,
                         support_z_pick=support_z_pick, g_close=float(pose.grip[2]), touchdown=None)
        self._attach(held, pose, poly, top_z)
        scene.held = held
        self.last_prior = jsonable({"obj": obj_pid, "grip": grip_pid, "body": body_log, "template": template_log,
                                    "r_obj_m": float(r_obj), "sigma_offset_m": float(sigma_offset),
                                    "offset_h": offset_h})
        scene.disturb(obj_pid, None, "close")     # X-1: read before the close's disturbance
        return held

    def _attach(self, held, pose, poly, top_z):
        """Hand-frame copies of the held parts' points and of the object's top outline (they move with the hand)."""
        scene = self.scene
        local = {}
        for pid in {held.pid, held.grip_pid}:
            part = scene.parts.get(pid)
            if part is not None and part.xyz is not None:
                local[pid] = pose.R.T @ (np.asarray(part.xyz, dtype=float) - pose.grip)
        if poly is not None:
            pts = np.column_stack([poly, np.full(len(poly), top_z)])
            local[f"{held.pid}:polygon"] = (pts - pose.grip[None, :]) @ pose.R
        grip = scene.parts.get(held.grip_pid)
        if held.grip_pid != held.pid and grip is not None and grip.region_xy is not None and grip.top_z is not None:
            gpts = np.column_stack([grip.region_xy, np.full(len(grip.region_xy), grip.top_z.value)])
            local[f"{held.grip_pid}:polygon"] = (gpts - pose.grip[None, :]) @ pose.R
        scene._held_local = local

    def _shift_attached(self, held, pose, drag_world):
        """Re-localization moved the object by ``drag_world`` (world, hand static): shift its hand-frame copies."""
        local = self.scene._held_local
        delta = pose.R.T @ drag_world
        if held.pid in local:
            local[held.pid] = local[held.pid] + delta
        key = f"{held.pid}:polygon"
        if key in local:
            local[key] = local[key] + delta[None, :]

    def _held_mask(self, held, view):
        """The object's top outline (hand frame) projected into the wrist view, minus the robot pixels."""
        scene = self.scene
        pose = HandPose.from_view(view)
        local = scene._held_local.get(f"{held.pid}:polygon")
        size = (scene.camera.size_px, scene.camera.size_px)
        if local is None:
            return np.zeros(size, dtype=bool)
        pts = pose.grip[None, :] + local @ pose.R.T
        uv, _ = _project(scene.camera, pose, pts)
        if not np.isfinite(uv).all():
            return np.zeros(size, dtype=bool)
        mask = np.zeros(size, dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(uv).astype(np.int32)], 1)
        return (mask > 0) & ~scene._robot_mask(size)

    def _pre_close_view(self, view):
        pose = HandPose.from_view(view)
        for _, other in reversed(list(self.scene._recent)):
            if other is view:
                continue
            if other.gap_mm >= view.gap_mm + PRE_CLOSE_GAP_MARGIN_MM and _same_pose(HandPose.from_view(other), pose):
                return other
        return None

    def relocalize(self, held: HeldFrame, view: StepView) -> HeldFrame:
        """§4.7 item 3 (hand static): the object's displacement during the close, from the most textured patch of
        its projected top outline tracked from the last pre-close view at this pose (wrist; the external view when
        not wrist-visible); offset_h moves by that drag. Without a pre-close view the prior is kept."""
        scene = self.scene
        pose = HandPose.from_view(view)
        pre = self._pre_close_view(view)
        evidence = {"method": None, "drag_m": None, "shift_px": None, "peak": None, "status": "no_pre_close_view"}
        drag = None
        if pre is not None:
            drag, evidence = (self._wrist_drag(held, pre, view) if held.wrist_visible
                              else self._external_drag(held, pre, view))
        if drag is None:
            self.last_relocalize = jsonable(evidence)
            return held
        new_offset = np.asarray(held.offset_h, dtype=float) + pose.R.T @ drag
        new = dataclasses.replace(held, offset_h=new_offset)
        self._shift_attached(held, pose, drag)
        scene.held = new
        self.last_relocalize = jsonable(evidence)
        return new

    def _wrist_drag(self, held, pre, view):
        """The close drag in the wrist view: the consensus shift of the object's texture patches from the last
        pre-close view to the current one (``_patch_shift``), converted to a horizontal world drag at the object's
        top depth."""
        pose = HandPose.from_view(view)
        top_z = self._top_z(held.pid)
        kappa = self.scene.camera.f_px / max(float(pose.camera[2] - top_z), 1e-6)
        shift, evidence = self._patch_shift(held, pre, view, kappa)
        evidence = dict(evidence, method="wrist", drag_m=None)
        if shift is None:
            return None, evidence
        du, dv = float(shift[0]), float(shift[1])
        drag = (du / kappa) * pose.R[:, 1] - (dv / kappa) * pose.R[:, 0]
        drag = drag - (drag @ np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])
        evidence.update({"drag_m": drag})
        return drag, evidence

    def _patch_shift(self, held, pre, view, kappa):
        """(consensus shift (du, dv) px | None, evidence) of up to RELOCALIZE_PATCHES well-textured, mutually separated
        patches of the object's projected top in ``pre`` (no strong edge inside: texture, not outline) tracked into
        ``view`` (search RELOCALIZE_MAX_DRAG_M at kappa): the shift most of them agree on (within M14_SHIFT_SIGMA_K x
        TRACK_SIGMA_PX), at least two when two were found."""
        scene = self.scene
        mask = self._held_mask(held, pre)
        r = _NCC_RADIUS
        inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        inner[:r, :], inner[-r:, :], inner[:, :r], inner[:, -r:] = False, False, False, False
        inner &= ~(cv2.dilate(scene._images(pre)["edge"].astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1),
                                                                                     np.uint8)) > 0)
        evidence = {"shift_px": None, "peak": None, "status": "no_patch", "patches": 0, "found": 0}
        vs, us = np.nonzero(inner)
        if not len(us):
            return None, evidence
        gray_pre = np.asarray(pre.wrist_gray_upright, dtype=float)
        gray_now = np.asarray(view.wrist_gray_upright, dtype=float)
        k = (2 * r + 1, 2 * r + 1)
        g = gray_pre.astype(np.float32)
        std = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, k) - cv2.boxFilter(g, -1, k) ** 2, 0.0))
        order = np.argsort(-std[vs, us])
        search = int(math.ceil(kappa * RELOCALIZE_MAX_DRAG_M))
        chosen, shifts, peaks = [], [], []
        for i in order:
            uv = np.array([float(us[i]), float(vs[i])])
            if any(np.max(np.abs(uv - c)) <= 2 * r + 1 for c in chosen):
                continue
            chosen.append(uv)
            try:
                track = NCCTrack(gray_pre, tuple(uv), search=search)
            except ValueError:
                break                                          # sorted by texture: the rest is textureless too
            match, peak = track.match(gray_now)
            peaks.append(peak)
            if match is not None:
                shifts.append(np.asarray(match, dtype=float) - uv)
            if len(chosen) >= RELOCALIZE_PATCHES:
                break
        evidence.update({"patches": len(chosen), "found": len(shifts), "peak": max(peaks) if peaks else None})
        if not shifts:
            evidence["status"] = "lost" if chosen else "textureless"
            return None, evidence
        # consensus: the shift most patches agree on (a smooth texture can give a false NCC peak elsewhere, and a
        # patch whose true position left the view always does)
        shifts = np.asarray(shifts)
        near = np.linalg.norm(shifts[:, None, :] - shifts[None, :, :], axis=2) <= M14_SHIFT_SIGMA_K * TRACK_SIGMA_PX
        best = int(np.argmax(near.sum(axis=1)))
        agree = near[best]
        evidence["agreeing"] = int(agree.sum())
        if len(shifts) >= 2 and agree.sum() < 2:
            evidence["status"] = "inconsistent"
            return None, evidence
        shift = np.median(shifts[agree], axis=0)
        evidence.update({"shift_px": [float(shift[0]), float(shift[1])], "status": "ok"})
        return shift, evidence

    def _external_drag(self, held, pre, view):
        scene = self.scene
        evidence = {"method": "external", "drag_m": None, "shift_px": None, "peak": None, "status": "no_projector"}
        projector = scene.projector
        J = getattr(getattr(projector, "model", None), "J", None)
        if projector is None or J is None:
            return None, evidence
        centre_xy, _, _, _ = self._footprint(held.pid)
        uv, _ = projector.project(np.r_[centre_xy, self._top_z(held.pid)])
        J = np.asarray(J, dtype=float)
        search = int(math.ceil(float(np.linalg.norm(J[:, :2], ord=2)) * RELOCALIZE_MAX_DRAG_M))
        gray_pre = cv2.cvtColor(np.ascontiguousarray(pre.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
        gray_now = cv2.cvtColor(np.ascontiguousarray(view.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
        try:
            track = NCCTrack(gray_pre, (float(uv[0]), float(uv[1])), search=search)
        except ValueError:
            evidence["status"] = "textureless"
            return None, evidence
        match, peak = track.match(gray_now)
        evidence["peak"] = peak
        if match is None:
            evidence["status"] = "lost"
            return None, evidence
        shift = np.asarray(match, dtype=float) - np.asarray(uv, dtype=float)
        dxy = np.linalg.lstsq(J[:, :2], shift, rcond=None)[0]
        drag = np.r_[dxy, 0.0]
        evidence.update({"drag_m": drag, "shift_px": shift, "status": "ok"})
        return drag, evidence

    def lift_recorder(self, held: HeldFrame, view: StepView):
        """§4.7 items 4 / 5: the recorder fed by LiftMonitorAdapter during VerifyHold's lift (stage "lift")."""
        return _LiftRecorder(held, view)

    def after_lift(self, held: HeldFrame, recorder, *, gap_before_mm: float,
                   gap_after_mm: float) -> tuple[HeldFrame, dict]:
        """The lift silhouette (zero-flow pixels while the scene looms) against the pre-lift mask, the external
        template co-motion against J dp, the gap change: the M14 Touchdown trigger (or not wrist-visible); the M17
        offset sigma; extent_axes; held.vacated_cells = the object's (and grip part's) pick-time cells and the D7b
        fills derived from them [C-36]."""
        scene = self.scene
        first, last = recorder.first, recorder.last
        pose0, pose1 = HandPose.from_view(first), HandPose.from_view(last)
        pre = self._held_mask(held, first)
        size = pre.shape
        sigma_px = scene._pixel_sigma()
        g0 = np.asarray(first.wrist_gray_upright, dtype=np.float32)
        g1 = np.asarray(last.wrist_gray_upright, dtype=np.float32)
        diff = cv2.blur(np.abs(g1 - g0), (STATIC_WINDOW_PX, STATIC_WINDOW_PX))
        static = diff <= max(STATIC_DIFF_FLOOR_GREY, STATIC_DIFF_K * sigma_px)
        reach = int(math.ceil(M14_SHIFT_FLOOR_PX + M14_SHIFT_SIGMA_K * TRACK_SIGMA_PX))
        around = cv2.dilate(pre.astype(np.uint8), _disk_kernel(reach)) > 0
        silhouette = static & around & ~scene._robot_mask(size)
        iou = _iou(silhouette, pre) if pre.any() else 0.0
        c_pre, c_sil = _centroid_px(pre), _centroid_px(silhouette)
        centroid_shift = None if c_pre is None or c_sil is None else float(np.linalg.norm(c_sil - c_pre))
        ext_measured = ext_expected = template_shift = None
        projector = scene.projector
        J = getattr(getattr(projector, "model", None), "J", None)
        if held.ext_template is not None and J is not None:
            track = copy.deepcopy(held.ext_template)
            e0 = cv2.cvtColor(np.ascontiguousarray(first.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
            e1 = cv2.cvtColor(np.ascontiguousarray(last.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)
            m0, _ = track.match(e0)
            m1, _ = track.match(e1) if m0 is not None else (None, None)
            if m0 is not None and m1 is not None:
                measured = np.asarray(m1, dtype=float) - np.asarray(m0, dtype=float)
                expected = np.asarray(J, dtype=float) @ (pose1.grip - pose0.grip)
                ext_measured, ext_expected = float(np.linalg.norm(measured)), float(np.linalg.norm(expected))
                template_shift = float(np.linalg.norm(measured - expected))
        sigma_close = float(getattr(scene.cal, "sigma_close_mm", 0.0) or 0.0) if scene.cal is not None else 0.0
        th_gap = th_max("M14_gap_change_mm", Term("DD", M14_GAP_FLOOR_MM, "M14_GAP_FLOOR_MM"),
                        Term("ONLINE", M14_GAP_SIGMA_K * sigma_close, "M14_GAP_SIGMA_K*sigma_close"), unit="mm")
        th_iou = th_value("M14_iou", M14_IOU_MIN, "DD", "lift silhouette IoU against the pre-lift mask >= M14_IOU_MIN",
                          measured=iou)
        th_shift = th_max("M14_shift_px", Term("DD", M14_SHIFT_FLOOR_PX, "M14_SHIFT_FLOOR_PX"),
                          Term("DD", M14_SHIFT_SIGMA_K * TRACK_SIGMA_PX,
                               "M14_SHIFT_SIGMA_K*TRACK_SIGMA_PX (sigma_track = the v1 PIXEL_NOISE_PX design default)"),
                          unit="px")
        gap_change = abs(float(gap_after_mm) - float(gap_before_mm))
        reasons = []
        if gap_change > th_gap.value:
            reasons.append("gap_change")
        if iou < th_iou.value:
            reasons.append("silhouette_iou")
        # the wrist template shift (spec §4.7 item 6): the object's own texture patches tracked from the first to the
        # last lift frame (a rigidly held object does not move in the hand-mounted camera); the silhouette centroid
        # shift is logged only (it also moves with any registration error of the pre-lift outline)
        wrist_shift = wrist_track = None
        if held.wrist_visible:
            kappa0 = scene.camera.f_px / max(float(pose0.camera[2] - self._top_z(held.pid)), 1e-6)
            shift_px, wrist_track = self._patch_shift(held, first, last, kappa0)
            wrist_shift = None if shift_px is None else float(np.linalg.norm(shift_px))
        if (wrist_shift is not None and wrist_shift > th_shift.value) or (
                template_shift is not None and template_shift > th_shift.value):
            reasons.append("template_shift")
        if not held.wrist_visible:
            reasons.append("not_wrist_visible")
        top_z = self._top_z(held.pid)
        kappa = scene.camera.f_px / max(float(pose0.camera[2] - top_z), 1e-6)
        measured = math.hypot(float(held.sigma_offset_m), TRACK_SIGMA_PX / kappa)
        if centroid_shift is not None:
            measured = math.hypot(measured, centroid_shift / kappa)
        th_m17 = self._m17(measured, held.r_obj_m)
        fallback = iou < M14_IOU_MIN or not held.wrist_visible
        sigma_offset = th_m17.value if fallback else measured
        cells = self._vacated_cells(held)
        new = dataclasses.replace(held, silhouette=silhouette, sigma_offset_m=float(sigma_offset), vacated_cells=cells)
        scene.held = new
        thresholds = [th_gap, th_iou, th_shift] + ([th_m17] if fallback else [])
        evidence = jsonable({
            "iou": iou, "centroid_shift_px": centroid_shift, "wrist_template_shift_px": wrist_shift,
            "wrist_track": wrist_track, "gap_change_mm": gap_change,
            "ext_comotion_px": ext_measured, "ext_expected_px": ext_expected, "template_shift_px": template_shift,
            "touchdown": {"needed": bool(reasons), "reasons": reasons}, "sigma_offset_m": sigma_offset,
            "silhouette_px": int(silhouette.sum()), "pre_lift_px": int(pre.sum()), "lift_frames": recorder.transitions,
            "vacated_cells": int(len(cells)), "thresholds": thresholds_dicts(thresholds)})
        return new, evidence

    def _vacated_cells(self, held):
        scene = self.scene
        hf = scene.hf
        cells = []
        for pid in (held.pid, held.grip_pid):
            part = scene.parts.get(pid)
            if part is None:
                continue
            if part.region_cells is not None and len(part.region_cells):
                cells.append(np.asarray(part.region_cells, dtype=int).reshape(-1, 2))
            if part.region_xy is not None and len(part.region_xy) >= 3:
                cells.append(hf.cells_in_polygon(poly_inflate(part.region_xy, hf.cell_m)))
        if not cells:
            return np.zeros((0, 2), dtype=int)
        own = np.unique(np.vstack(cells), axis=0)
        derived = scene.derived_fills(own)
        return np.unique(np.vstack([own, derived]), axis=0) if len(derived) else own

    def after_touchdown(self, held: HeldFrame, g_touch: float, sigma_touch: float, *, pivot: bool) -> HeldFrame:
        """§4.7 item 6: b_low = g_touch - support_z(pick) by touch; a pivot (silhouette change during Touchdown) is
        flagged touchdown_pivot and inflates sigma_b to the M17 bound (half the object radius)."""
        b = float(g_touch) - held.support_z_pick.value
        sigma_b = math.hypot(float(sigma_touch), held.support_z_pick.sigma)
        flags = []
        threshold = None
        if pivot:
            # spec §4.7 item 6: a pivot during Touchdown inflates sigma_b; the bound taken is M17's half object radius
            # (the pivot can tip the lowest point by up to that much)
            threshold = th_max("M14_pivot_sigma_b_m", Term("ONLINE", sigma_b, "touch sigma_b"),
                               Term("DD", M17_FRACTION * held.r_obj_m, "M17_FRACTION*r_obj (spec §4.7 item 6 pivot)"),
                               unit="m")
            sigma_b = threshold.value
            flags.append("touchdown_pivot")
        touchdown = jsonable({"g_touch": float(g_touch), "sigma_touch": float(sigma_touch), "pivot": bool(pivot),
                              "b_low_before": held.b_low.to_dict(), "b_low_after": {"value": b, "sigma": sigma_b},
                              "flags": flags, "thresholds": thresholds_dicts([threshold] if threshold else [])})
        new = dataclasses.replace(held, b_low=Meas(b, sigma_b, "touch"), touchdown=touchdown)
        self.scene.held = new
        return new
