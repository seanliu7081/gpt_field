"""agent_SpeedUp_v2 relation controllers (contract §C10; spec §4.8, §4.9, §4.11 local recovery; decisions 1 and 2 §2.6;
lead rulings r2close 3).

Every controller is a Python generator (contract §C3.2): it yields primitive FACTORIES (``factory(view) ->
Primitive``; the policy runs the primitive to done and sends back ``(outcome, primitive)``) and, at phase boundaries
only, ``PointNeed`` records (the policy's fulfiller makes the one model call and sends back a ``PointAnswer``). No
model call happens inside a primitive, no primitive asks for pointing and every ``Align`` is built with
``allow_pointing=False``. A controller returns one ``RelationOutcome`` and records it, and only it, through
``ctx.record`` [C-39]; an outcome received from a nested controller (a grasp inside a turn, a destination survey
inside a place) is never recorded twice.

Slice (task-2 minimal set, user decision 2026-10-19 "live first"):
- ``calibrate_plus``: CalibratePlus with a probe-tracker SurveyRecorder -> V2Calibration (camera model, external
  model, hand locator, projector, fingertip band, false-change maps, frame alignment, mount) -> ``scene.start``;
  outcome "calibrated" (``ctx.cal`` is set);
- ``register_regions``: start-view registration of every part with a role (region parts empty-hand and possibly
  partial; grip-only parts provisional), one coarse re-seed after a failure, and (when the hand is empty) an
  empty-hand survey hover of a REGION part that still fails; one "registered" outcome per part;
- ``survey_hover``: a hover survey of one part (P11 hover, the arrival descent, ParallaxProbe, register) with the
  lower-hover retry;
- ``grasp`` (purpose turn / carry / hold): hover by ``planner.hover_for`` (two-stage, decision 1 §2.6), the arrival
  descent with ``scene.reanchor_from``, pointing (coarse first when the predicted sigma exceeds the fine crop's
  reach, rulings item 9), ParallaxProbe, registration, one re-point after ``seed_outside_region`` (lead ruling 3),
  the answer check on the RE-MEASURED heights, the BR-1 second hover (decision 2 §2.6 triggers), the contact plan,
  Orient, Align (``across_bar`` for an elongated part; along-jaw residual within V2), the descent (P14 ladder for a
  touch_stop or marginal plan, one guarded move for a clean height_stop), the contact classification (never close
  above the band), the clearance retreat, the close (G5, G13, G13+), the load test by purpose (VerifyHold with
  co-motion and the O-1 rotation verdict for carry / hold; the first twist for a turn), the in-hand frame
  (relocalize, after_lift, Touchdown when M14 fires, vacate_held) and the held-scope bus invariants;
- ``turn``: budget pre-check, grasp, the effect watch (baseline before the first segment, motion false-change rate
  measured on the approach, O7 validity per phase), TwistSegment(s) with the WedgeObserver (reversal on a blocked
  sense), HingeTwist (orbit follow, end stop, effect margin, quarter / half turn), ReleaseVerify(settle_first),
  spring-back and persistence checks, the stays entry and the articulated keep-out;
- ``place_on``: the destination precondition (empty-hand survey), grasp, aim (M4 / M16), the held transit (C-36:
  the object's own cells vacated), the external AimCheck row, GuardedLower fast and slow, SupportClassify (never
  open when perched), release gated on the support height, rest and follow checks, the stays entry;
- ``release`` (the implicit or explicit release) and ``hold``;
- ``slide``, ``place_in`` and ``push_to``: ``relation_not_implemented`` (reserved, §C0.3).

Model-facing evidence text is built only by ``evidence_text`` (§C10.6): magnitudes in cm and degrees, parts by pid,
threshold names with values, never world axes, signed vectors or plan texts.

Review fixes (BR review, 2026-10-19): turn resume with the kept free sense and the effect pre-check (BRR-1, §C3.8);
the motionless-refusal guard on the step controllers (BRR-2); the put-back before opening after a failed load test
(BRR-3); AimCheck's correction kept by the lowering and the HeldFrame (BRR-4); the destination precondition before
the hand leaves (BRR-5); local recovery while holding, §C10.2 step 11 (BRR-6); the co-motion re-anchor after a lost
template with the hold intact (found by the first unpatched carry, BRR-7); unverified co-motion / rotation flagged
(BRR-8); the turned part's keep-out and stale record on every exit after a twist (BRR-9); no invented pick support
(BRR-10); the top-down stretch guard kept as a disclosed deviation (BRR-11, needs a lead ruling); labelled re-asks,
px units, the ambiguous-support flag and the last-refresh V2 check (BRR-15/16).

Live-first blocker fixes (BR fixer, 2026-10-19; version 2):
- blocker 1, effect visibility (fix (c), both (a) and (b)): the effect watch takes the turn step's MotionNoiseMeter
  start frame as its reference (``meter.reference()``; the watch falls back to the grasp-pose baseline itself) and
  the obj / grip regions as excludes; the O7 phases keep their structure on the scene's S-16' hand mask (its rule is
  logged per phase); after the release every effect criterion is confirmed (``_effect_confirm``: ``confirm`` on
  static frames, lifting the open hand in 1 cm guarded steps up to POST_CONFIRM_LIFT_MAX_M while the region is
  hidden): no onset + met -> turned_by_effect (flag effect_confirmed_after_release, theta_e null), violated ->
  effect_not_observed (effect_region_unchanged), still pending -> effect_not_observed (effect_unverifiable); an
  onset whose confirmation is violated fails as before. The effect region is taken once, before the grasp, and the
  watch is kept per turned part (``memory["effect_ref"]``): the stays entry and the BX skip_step check use its
  reference, the resume pre-check confirms on a fresh watch of it;
- blocker 2, grip-part pointing: BR-1 a grip part whose record coincides with its object's (same point or region IoU
  >= SHARED_REGION_IOU) hovers from its earliest independent estimate; BR-2 the hover before pointing a grip part of
  another part is planned with BP-1 (``hover_for(pointing="auto")``) and framed again at the arrived hover
  (``planner.pointing_frame``; a guarded -R0 shift, then a rise when the capped shift is not enough); BR-3 a re-ask
  never resends the same crop (coarse on the full view with "the pointed point lies on part <obj> itself" when the
  rejected point's region overlaps the object's at IoU >= REASK_OBJECT_IOU, else a fine crop re-centred on the
  rejected pixel; ``marker_uv`` only when ``ctx.pointing_marker``); BR-4 after ESCALATE_FAILED_ROUNDS failed rounds
  (or a repeated answer id) one object-framing survey with coarse pointing per plan step, then repair, with the
  measured facts in ``evidence["pointing"]`` and the evidence text; BR-5 one ParallaxProbe of the blocking cells and
  one re-plan per grasp on the planner's ``refine_cells``.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import dataclasses
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from . import agent_speedup_v2_geometry as geometry
from . import agent_speedup_v2_observers as observers
from . import agent_speedup_v2_planner as planner
from . import agent_speedup_v2_primitives as primitives
from . import agent_speedup_v2_robot as robot_facts
from .agent_speedup_v2_primitives import CalibratePlus
from .agent_speedup_v2_scene import D8_FLOOR_M, D8_SIGMA_K, TOP_BAND_SUPPORT_N
from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (AXIS_INCONSISTENT, EVIDENCE_TEXT_MAX_CHARS, HANDOVER_PREFIX, PREEMPTED_PREFIX,
                                     PREVIOUS_REJECTED_MAX_CHARS, REASONS, WRIST_IMAGE_PX, HandPose, KeepOut, Meas,
                                     PartRecord, PointAnswer, PointNeed, RelationOutcome, Seed, V2Calibration,
                                     WristCamera, jsonable, poly_inflate, rect_polygon, th_value, thresholds_dicts)
from .contact_skills import (AnchoredMove, GripperSettlePrimitive, HoldPrimitive, Orient, VerifyHold, WaypointReach,
                             classify_close, quaternion_angle_rad)
from .speed_first_gripper import GripperSettleConfig
from .wrist_servo import Align, NCCTrack

RELATIONS_VERSION = "agent-speedup-v2-relations-3"

LABELS = {
    "B2_LOCAL_RETRIES": ("DD", "spec §4.11 (B2): two local retries per relation (v1 local_retries = 2, "
                               "agent_speedup_v1.py:233)"),
    "TOUCH_KEEPOUT_DEPTH_M": ("DD", "grasp-refusal decision J, GC-1 touch keep-out: used only without a measured "
                                    "support (the record's support, else the base plane): the keep-out reaches 5 cm "
                                    "below the pad bottom (remedy 2 design); a deeper keep-out only refuses more"),
    "P14_LADDER_SEGMENT_M": ("DD", "P14: v1 LADDER_SEGMENT_M 20 mm descent segments (agent_speedup_v1.py:45, "
                                   "restated)"),
    "P14_SETTLE_STEPS": ("DD", "P14: v1 LADDER_SETTLE_STEPS, a 3-step settle between ladder segments "
                               "(agent_speedup_v1.py:46, restated)"),
    "P14_FIRST_SETTLE_MAX_STEPS": ("DD", "P14: v1 FIRST_SETTLE_MAX_STEPS quiet settle before the first segment "
                                         "(agent_speedup_v1.py:47, restated)"),
    "V4_REPOINTS": ("DD", "spec §4.8 registered: at most 2 re-points per part per view (V4)"),
    "POST_ORIENT_HOLD_STEPS": ("DD", "v1 POST_ORIENT_HOLD_STEPS: the residual rotation after Orient settles "
                                     "(agent_speedup_v1.py:49, restated)"),
    "ORIENT_TOLERANCE_RAD": ("DD", "v1 Orient default tolerance 3 deg (contact_skills.Orient, restated): a smaller "
                                   "planned turn is not commanded"),
    "ORIENT_BEFORE_TRANSIT_RAD": ("DD", "decision 1 §2.6: Orient before the transit to the hover when the planned "
                                        "yaw differs from the hover yaw by more than 45 deg"),
    "LIFT_TEST_M": ("DD", "v1 LIFT_TEST_M load-test lift 25 mm (agent_speedup_v1.py:52, restated)"),
    "EMPTY_CLOSE_SHIFT_FRACTION": ("DD", "spec §4.8 held recovery: re-align toward the measured centre line by half "
                                         "the measured width"),
    "PERCH_LIFT_M": ("DD", "spec §4.9 SupportClassify perched: lift 1.5 cm, re-measure, re-aim"),
    "DRIFT_LIFT_M": ("DD", "spec §4.9 GuardedLower drift invariant: lift 1 cm and re-aim"),
    "GRASP_SETTLE_MAX_POSE_CORRECTION_M": ("DD", "v1 GRASP_SETTLE max_pose_correction_m 3 cm "
                                                 "(agent_speedup_v1.py:110, restated)"),
    "OPEN_TOLERANCE_MM": ("DD", "v1 OPEN_TOLERANCE_MM: the gripper counts as open within 2 mm of the calibrated "
                                "open gap (agent_speedup_v1.py:105, restated)"),
    "OPEN_SETTLE_MIN_ACTIONS": ("DD", "v1 OPEN_SETTLE min_open_actions 15 (agent_speedup_v1.py:111, restated)"),
    "HELD_CONTACT_BACKOFF_M": ("DD", "v1 HELD_CONTACT_BACKOFF_M 1 cm back-off after a held-object contact "
                                     "(agent_speedup_v1.py:64, restated)"),
    "MIN_MOVE_OFFSET_M": ("DD", "v1 MIN_MOVE_OFFSET_M: WaypointReach treats offsets up to 5 mm as reached "
                                "(agent_speedup_v1.py:39, restated)"),
    "REACH_LIMIT_MEMORY": ("DD", "v1 REACH_LIMIT_MEMORY recorded reach limits (agent_speedup_v1.py:73, restated)"),
    "REACH_LIMIT_RADIUS_M": ("DD", "v1 REACH_LIMIT_RADIUS_M: a recorded limit applies within 3 cm "
                                   "(agent_speedup_v1.py:74, restated)"),
    "REACH_LIMIT_PUSH_M": ("DD", "v1 REACH_LIMIT_PUSH_M: a move pushing more than 1 cm further toward a recorded "
                                 "limit is refused (agent_speedup_v1.py:75, restated)"),
    "REACH_LIMIT_MAX_LIFT_M": ("DD", "v1 REACH_LIMIT_MAX_LIFT_M: limits met while lifting more than 1 cm are not "
                                     "remembered (agent_speedup_v1.py:76, restated)"),
    "ELONGATED_TRACK_RADIUS_FRACTION": ("DD", "[C-40]: a template of diameter 1.2 w spans an elongated part and "
                                              "reaches 0.1 w beyond each long edge"),
    "HOVER_LOCATE_M": ("DD", "decision 1 §2.6 / rulings item 4: the settled 3 cm vertical arrival descent at every "
                             "hover (median 4.8 px of external hand motion)"),
    "PROBE_RETRIES": ("DD", "a hover ParallaxProbe that ends early is repeated once from its start pose ([F, BR "
                            "smoke, real task 2] a free-air probe at 1.33 m ended its last leg with an RC-8 stall at "
                            "19.8 of 20 mm); a contact that repeats is reported"),
    "TRANSIT_CONTACT_LIFT_M": ("DD", "decision 1 §2.6: a contact on the arrival descent lifts 1 cm"),
    "TWO_STAGE_XY_M": ("DD", "decision 1 §2.6: the second hover is taken when it moves more than 1 cm in xy"),
    "TWO_STAGE_Z_M": ("DD", "decision 1 §2.6: the second hover is taken when it moves more than 2 cm in z"),
    "FINE_CROP_HALF_PX": ("DD", "V6: half of the 128 px fine crop (client FINE_CROP_PX)"),
    "COARSE_FIRST_SIGMA_PX": ("DD", "rulings item 9: coarse pointing first when the predicted sigma exceeds "
                                    "64 / sqrt(2 ln 2) px (about 54 px)"),
    "REACH_TRIES_PER_WAYPOINT": ("DD", "free and held moves: at most 12 WaypointReach legs per planned waypoint "
                                       "(each leg a fraction of the remaining offset)"),
    "QUARTER_TURN_RAD": ("DD", "spec §3.3 until quarter_turn: pi / 2"),
    "EVIDENCE_TEXT_ITEMS": ("DD", "evidence text: at most 3 magnitudes and 2 thresholds (the 200-char budget)"),
    "CM_PER_M": ("DD", "unit conversion for the model-facing evidence text (spec §3.7: centimetres)"),
    "MM_PER_CM": ("DD", "unit conversion for the model-facing evidence text (millimetre gaps in centimetres)"),
    "AIMCHECK_SIGMA_K": ("DD", "spec §4.9 AimCheck: a disagreement above 3 sigma is corrected along n-hat"),
    "FOLLOW_SIGMA_K": ("DD", "spec §4.9 release follow check: the object's external point moved by more than "
                             "3 sigma with the retreat"),
    "WORLD_UP": ("DD", "world +z unit vector (§C1.3 world frame, z up)"),
    "WORLD_DOWN": ("DD", "world -z unit vector (§C1.3 world frame, z up)"),
    "FIRST_STEP": ("DD", "policy step index of the episode's first act (counted from 0)"),
    "HOVER_LIFT_SIGMA_RATIO": ("DD", "[C-37] hover centre: the provisional start record unless its horizontal "
                                     "sigma (the region-choice ambiguity included) is more than twice that of the "
                                     "lift of its own plan cell (quantization and depth only); then the cell's lift"),
    "AIMCHECK_MIN_PEAK": ("DD", "AimCheck: the held object's external template counts as found at NCC peak >= 0.8 "
                                "(above v1 NCCTrack's 0.6 match floor; a correction needs a confident match)"),
    "POINT_CHECK_REASKS": ("DD", "contract §C10 grasp_point_round: ONE re-ask of a failed answer check (carrying the "
                                 "measured reason) per pointing round"),
    "MOTIONLESS_HOLD_STEPS": ("DD", "BRR-2: a relation failure repeated without any motion from an unchanged hand "
                                    "state holds one stays-recheck cycle (20 steps, the policy's HOLD_RECHECK_STEPS, "
                                    "1 s at 20 Hz) before it is returned, doubled per further repeat"),
    "MOTIONLESS_HOLD_MAX_STEPS": ("DD", "BRR-2: the repeat hold doubles up to 16 hold cycles (320 steps), so an "
                                        "unchanged refusal can be repaired about ten times in a 2400-step episode "
                                        "instead of twice per step"),
    "COINCIDE_XYZ_M": ("DD", "blocker-2 spec BR-1: two records whose points lie within 1 mm are the same measurement (the "
                             "three live coincidences, a grip part registered with its object's region, had identical "
                             "points; distinct parts of one object lie centimetres apart)"),
    "SHARED_REGION_IOU": ("DD", "blocker-2 spec BR-1 / BS-1: a grip part's record coincides with its object's at "
                                "region-cell IoU >= 0.5 (overlap 1.0 in 3/3 live coincidences, 0 coincident pairs in "
                                "67 ORACLE-run registrations)"),
    "REASK_OBJECT_IOU": ("DD", "blocker-2 spec BR-3: a rejected pointing whose registered region overlaps the object's "
                               "region at IoU >= 0.2 lies on the object itself (correct grip records max 0.13 over "
                               "321 records, wrong ones min 0.24 over 5): the re-ask goes coarse on the full view"),
    "ESCALATE_FAILED_ROUNDS": ("DD", "blocker-2 spec BR-4: after 2 failed pointing rounds on the same grip part at the "
                                     "same object (or the same answer id twice) one object-framing survey runs"),
    "ESCALATIONS_PER_STEP": ("DD", "blocker-2 spec BR-4: at most one object-framing survey per plan step, then the "
                                   "failure goes to repair"),
    "REFINE_CELLS_PER_GRASP": ("DD", "blocker-2 spec BR-5: at most one ParallaxProbe of the blocking cells and one "
                                     "re-plan per grasp on refine_cells"),
    "POST_CONFIRM_LIFT_STEP_M": ("DD", "blocker-1 spec BR-2: the post-release confirmation lifts in 1 cm guarded steps "
                                       "(stop on contact) while the effect region is hidden"),
    "POST_CONFIRM_LIFT_MAX_M": ("DD", "blocker-1 spec BR-2: at most 8 cm of post-release lift (the S-16 masks needed "
                                      "+3.0 / +4.4 cm on smokes 3 / 2; 8 cm left 0.84-0.86 of the region visible)"),
    "PERCENT": ("DD", "unit conversion for the model-facing evidence text (a share as a percentage, PR-1)"),
    "CONFIRM_FRAMES_MAX": ("DD", "blocker-1 spec BR-2: the static frames judged per lift height, twice O1's 3-frame "
                                 "run (a run broken once by one noisy frame can still complete)"),
}

B2_LOCAL_RETRIES = 2
P14_LADDER_SEGMENT_M = 0.020
P14_SETTLE_STEPS = 3
P14_FIRST_SETTLE_MAX_STEPS = 14
V4_REPOINTS = 2
POST_ORIENT_HOLD_STEPS = 4
ORIENT_TOLERANCE_RAD = math.radians(3.0)
ORIENT_BEFORE_TRANSIT_RAD = math.radians(45.0)
LIFT_TEST_M = 0.025
EMPTY_CLOSE_SHIFT_FRACTION = 0.5
PERCH_LIFT_M = 0.015
DRIFT_LIFT_M = 0.01
GRASP_SETTLE_MAX_POSE_CORRECTION_M = 0.03
OPEN_TOLERANCE_MM = 2.0
OPEN_SETTLE_MIN_ACTIONS = 15
HELD_CONTACT_BACKOFF_M = 0.010
MIN_MOVE_OFFSET_M = 0.005
REACH_LIMIT_MEMORY = 4
REACH_LIMIT_RADIUS_M = 0.030
REACH_LIMIT_PUSH_M = 0.010
REACH_LIMIT_MAX_LIFT_M = 0.010
ELONGATED_TRACK_RADIUS_FRACTION = 0.6
HOVER_LOCATE_M = 0.03
TRANSIT_CONTACT_LIFT_M = 0.01
PROBE_RETRIES = 1
TWO_STAGE_XY_M = 0.01
TWO_STAGE_Z_M = 0.02
FINE_CROP_HALF_PX = 64.0
COARSE_FIRST_SIGMA_PX = FINE_CROP_HALF_PX / math.sqrt(2.0 * math.log(2.0))
REACH_TRIES_PER_WAYPOINT = 12
QUARTER_TURN_RAD = math.pi / 2.0
HALF_TURN_RAD = math.pi
EVIDENCE_TEXT_ITEMS = 3
CM_PER_M = 100.0
PERCENT = 100.0
MM_PER_CM = 10.0
AIMCHECK_SIGMA_K = 3.0
AIMCHECK_MAX_CORRECTION_M = planner.M2_SEGMENT_M
FOLLOW_SIGMA_K = 3.0
AIMCHECK_MIN_PEAK = 0.8
HOVER_LIFT_SIGMA_RATIO = 2.0
POINT_CHECK_REASKS = 1
MOTIONLESS_HOLD_STEPS = 20
MOTIONLESS_HOLD_MAX_STEPS = 320
COINCIDE_XYZ_M = 0.001
SHARED_REGION_IOU = 0.5
REASK_OBJECT_IOU = 0.2
ESCALATE_FAILED_ROUNDS = 2
ESCALATIONS_PER_STEP = 1
REFINE_CELLS_PER_GRASP = 1
TOUCH_KEEPOUT_DEPTH_M = 0.05
POST_CONFIRM_LIFT_STEP_M = 0.01
POST_CONFIRM_LIFT_MAX_M = 0.08
CONFIRM_FRAMES_MAX = 2 * observers.O1_FRAMES

FIRST_STEP = 0
GRASP_SETTLE = GripperSettleConfig(max_pose_correction_m=GRASP_SETTLE_MAX_POSE_CORRECTION_M)
OPEN_SETTLE = GripperSettleConfig(min_open_actions=OPEN_SETTLE_MIN_ACTIONS)
WORLD_UP = (0.0, 0.0, 1.0)
WORLD_DOWN = (0.0, 0.0, -1.0)
REGION_ROLES = ("carried", "ref", "effect", "keepout", "hold_true")
CONTACT_REASONS = ("contact_stall", "contact_touch")
TURN_CRITERIA = {"effect_margin_reached": "effect", "end_stop": "stop", "quarter_turn_reached": "quarter_turn",
                 "half_turn_reached": "half_turn"}
TURN_SUCCESS = {"effect": "turned_by_effect", "stop": "turned_to_stop", "quarter_turn": "turned_quarter",
                "half_turn": "turned_half"}
SLIP_REASONS = ("turn_slip", "hold_lost")
SUPPORT_RETRY_CLASSES = ("perched", "blocked_above", "missed_surface")
# the kernel registry has no code for an "ambiguous" support (height within tolerance, lateral outside): it fails as
# lateral_drift and carries the flag support_ambiguous in its evidence and evidence text (BRR-16; a REASONS code of
# its own is the kernel owner's)
SUPPORT_FAILURE = {"perched": "support_above_destination", "blocked_above": "support_above_destination",
                   "missed_surface": "missed_surface", "no_support_within_bound": "no_support_within_bound",
                   "ambiguous": "lateral_drift"}
SUPPORT_FLAGS = {"ambiguous": "support_ambiguous"}
HELD_LOST_REASONS = ("hold_lost", "grip_not_on_obj")
UNVERIFIED_HOLD_FLAGS = ("comotion_unverified", "rotation_unmeasured")
SEED_REFUSAL = "seed_outside_region"
CUT_REFUSAL = "cut_by_border"
NO_STEP_PIDS = {"obj": None, "grip": [], "ref": None}
# v1 WaypointReach endings that end a move at once (the others are retried within REACH_TRIES_PER_WAYPOINT)
MOVE_ABORT_REASONS = ("reach_limit", "uncontrolled_motion", "field_boundary_reached", "left_planning_grid",
                      "hold_lost", "held_object_contact")
# blocker-2 spec BR-3: the measured re-ask texts (<= PREVIOUS_REJECTED_MAX_CHARS); the ringed variant is the one the live
# pointing check moved onto the part (1/1 with the marker of BC-2), the plain one is used until the client draws it
REASK_ON_OBJECT_TEXT = "the pointed point lies on part {obj} itself, not on a separate part"
REASK_ON_OBJECT_RINGED_TEXT = "the ringed point lies on the top surface of part {obj} itself, not on a separate part"
EFFECT_CONFIRM_FLAGS = {"met": "effect_confirmed_after_release", "violated": "effect_region_unchanged",
                        "pending": "effect_unverifiable"}


# ================================================================================================= context
@dataclass
class RelationContext:
    """The relation layer's shared state (contract §C10). The policy builds it with the keyword arguments robot,
    scene, monitor, stays, events, held_estimator and overrides, sets ``view`` and ``step`` before every advance and
    drains ``pending_outcomes`` after it. ``memory`` holds the controllers' episode memory: the gripper thresholds
    of Calibrate+ ("gripper"), the start-view seed lifts ("seed_lift"), the reach limits met ("reach_limits"), the
    last plans per part ("plans") and the gripped fixture part of a turn ("gripped"). ``survey_recorder`` is a
    test-only injection point (a callable with geometry.SurveyRecorder's signature); production leaves it None.
    Blocker-2 spec: ``memory["start_records"]`` keeps every part's start-view record (BR-1's independent hover
    reference), ``memory["pointing"]`` the failed pointing rounds per (plan step, grip part, object) (BR-4, the
    evidence BX-1 / BX-2 read) and ``memory["step_scope"]`` the plan step a guarded controller runs; blocker-1 spec:
    ``memory["effect_ref"]`` the turned part's effect watch with its pre-action reference (stays and resume)."""
    robot: object
    scene: object
    monitor: object
    stays: object
    events: object
    view: object = None
    step: int = FIRST_STEP
    cal: V2Calibration | None = None
    held_estimator: object = None
    holding: object = None
    grasp_quaternion: tuple | None = None
    hold_gap_ref_mm: float | None = None
    plan: object = None
    roles: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    overrides: Mapping = field(default_factory=dict)
    active_relation: str | None = None
    pending_outcomes: list = field(default_factory=list)
    consecutive_rejections: int | None = None
    memory: dict = field(default_factory=dict)
    survey_recorder: object = None      # test-only stand-in for geometry.SurveyRecorder (None = the geometry class)
    pointing_marker: bool = True        # the fulfiller's client draws PointNeed.marker_uv (blocker-2 spec BC-2, the
    # client's marker); False (a client that cannot draw one): a re-ask carries the measured text alone

    def record(self, outcome: RelationOutcome) -> RelationOutcome:
        if not isinstance(outcome, RelationOutcome):
            raise TypeError("ctx.record takes a RelationOutcome")
        self.pending_outcomes.append(outcome)
        bucket = self.counts.setdefault("recorded", {})
        bucket[outcome.relation] = bucket.get(outcome.relation, 0) + 1
        return outcome

    def note_rejection(self, kind: str) -> None:
        self.consecutive_rejections = int(self.consecutive_rejections or 0) + 1
        rejected = self.counts.setdefault("rejections", {})
        rejected[kind] = rejected.get(kind, 0) + 1

    def note_accepted(self, kind: str) -> None:
        self.consecutive_rejections = 0


# ================================================================================================= helpers
def _count(ctx, key: str) -> int:
    ctx.counts[key] = int(ctx.counts.get(key, 0)) + 1
    return ctx.counts[key]


def _emit(ctx, event: str, /, **payload) -> None:
    events = getattr(ctx, "events", None)
    if events is not None:
        events.emit(event, **payload)


def _run(factory):
    """Yield one primitive factory; returns (outcome dict, primitive or None)."""
    reply = yield factory
    if not (isinstance(reply, tuple) and len(reply) == 2 and isinstance(reply[0], Mapping)):
        raise TypeError("a primitive factory is answered with (outcome, primitive)")
    return dict(reply[0]), reply[1]


def _preempted(result) -> bool:
    return str((result or {}).get("reason") or "").startswith(PREEMPTED_PREFIX)


def _reason_code(text, default: str) -> str:
    """A REASONS code for an outcome (a preemption keeps its observer's reason when that is a code)."""
    text = str(text or "")
    if text.startswith(PREEMPTED_PREFIX):
        text = text[len(PREEMPTED_PREFIX):]
    return text if text in REASONS else default


def _thresholds(items) -> list[Threshold]:
    """Threshold records from records or their dicts (primitive and planner evidence); malformed ones dropped."""
    out, seen = [], set()
    for item in items or ():
        try:
            if isinstance(item, Threshold):
                th = item
            elif isinstance(item, Mapping):
                th = Threshold(str(item["name"]), float(item["value"]), str(item["label"]),
                               str(item.get("derivation") or item["name"]))
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(th.value):
            continue
        key = (th.name, th.value)
        if key not in seen:
            seen.add(key)
            out.append(th)
    return out


def _pids(obj=None, grip=None, ref=None) -> dict:
    return {"obj": obj, "grip": [p for p in (grip or ()) if p is not None], "ref": ref}


def _outcome(ctx, relation, status, reason, *, pids, evidence, thresholds=(), start=None, retries=None,
             resume=None, record=True) -> RelationOutcome:
    steps = 0 if start is None else max(0, int(ctx.step) - int(start))
    outcome = RelationOutcome(relation, status, reason, dict(pids), jsonable(evidence or {}), _thresholds(thresholds),
                              "", int(retries or 0), steps, resume=resume)
    outcome.evidence_text = evidence_text(outcome)
    if record:
        ctx.record(outcome)
    return outcome


def _view(ctx):
    return ctx.view


def _pose(ctx) -> HandPose:
    return HandPose.from_view(ctx.view)


def _gripper(ctx) -> dict:
    return ctx.memory.get("gripper") or {}


def _g3(ctx):
    return _gripper(ctx).get("g3")


def _g5_mm(ctx) -> float:
    g5 = _gripper(ctx).get("g5")
    return primitives.G5_FLOOR_MM if g5 is None else float(g5.value)


def _g6(ctx) -> Threshold:
    g6 = _gripper(ctx).get("g6")
    if g6 is None:
        g6 = primitives.g6_hold_floor(ctx.cal.gripper, float(ctx.cal.sigma_close_mm))
    return g6


def _g_open_mm(ctx) -> float:
    return float(ctx.cal.gripper.open_gap_mm)


def _f_px(ctx) -> float:
    camera = getattr(ctx.cal, "camera", None)
    return float(getattr(camera, "f_used", robot_facts.FOCAL_PX_AT_512))


def _band_top_row(ctx):
    band = getattr(ctx.cal, "fingertip_band", None)
    return None if band is None else geometry.fingertip_band_top_row(band)


def _turns(ctx) -> int:
    return int(geometry.frame_alignment_turns(ctx.view.R, getattr(ctx.cal, "external", None)))


def _start_quaternion(ctx):
    return tuple(ctx.cal.start_quaternion)


def _budget_left(ctx, sense: int) -> float:
    return float(ctx.robot.yaw_budget_left(_start_quaternion(ctx), ctx.view.quaternion_xyzw, int(sense)))


def _text_of(ctx, pid: str) -> str:
    plan = getattr(ctx, "plan", None)
    if plan is not None:
        try:
            return plan.part(pid).text
        except KeyError:
            pass
    part = ctx.scene.parts.get(pid)
    return "" if part is None else part.text


def _keepouts(ctx, *, skip=()) -> list:
    return [k for k in ctx.scene.keepouts.values() if k.pid not in set(skip)]


def _other_parts(ctx, *, skip=()) -> list:
    return [p for p in ctx.scene.parts.values() if p.pid not in set(skip)]


def _part_moved(ctx, *pids) -> None:
    for pid in pids:
        if pid is not None and pid in ctx.scene.parts:
            ctx.scene.part_moved(pid)


def _window_start(ctx, *pids) -> None:
    """MC-R2a (gap round §2.5, ruling L6): a hover window of the part starts. The scene's ``window_start`` resets only
    the view-memory axes and their conflict flag; the whole record and its widths survive until a disturbance (X-1).
    A scene without ``window_start`` keeps the S-12 rule (``part_moved``: the whole view memory is forgotten)."""
    for pid in pids:
        if pid is None or pid not in ctx.scene.parts:
            continue
        start = getattr(ctx.scene, "window_start", None)
        if callable(start):
            start(pid)
        else:
            ctx.scene.part_moved(pid)


def _disturb(ctx, obj, cause) -> None:
    """X-1 (gap round §2.2): a contact-class event naming a part of ``obj`` (descent touch, GC-1 touch, close,
    release, put-down, carry start, transit contact) starts a new disturbance epoch of the object
    (``SceneModel.disturb(obj, step, cause)``); H1's body record, MC-R2a's kept whole registration and PO6's unmoved
    sub-part rule read only records of the current epoch. No-op on a scene without ``disturb``."""
    disturb = getattr(ctx.scene, "disturb", None)
    if obj is not None and callable(disturb) and obj in ctx.scene.parts:
        disturb(obj, int(ctx.step), str(cause))


def _moved_by_contact(ctx, obj, grip, cause) -> None:
    """A contact path (MC-R2a: contact paths keep ``part_moved`` and add the X-1 disturbance): the parts' view memory
    is forgotten and the object's epoch advances."""
    _part_moved(ctx, grip, obj)
    _disturb(ctx, obj, cause)


def _registration_event(ctx, pid, phase) -> None:
    payload = dict(ctx.scene.last_registration.get(pid) or {"pid": pid})
    payload["phase"] = phase
    _emit(ctx, "registration", **payload)


def _recorder(ctx, view, **kwargs):
    """A SurveyRecorder for ``view`` (geometry's class; a test may inject a stand-in via ctx.survey_recorder)."""
    factory = getattr(ctx, "survey_recorder", None) or geometry.SurveyRecorder
    return factory(view, ctx.robot, **kwargs)


def _registered(part) -> bool:
    return isinstance(part, PartRecord) and part.status == "registered" and not part.provisional


def _holds(ctx, grip_pid) -> bool:
    held = getattr(ctx, "holding", None)
    return held is not None and getattr(held, "grip_pid", None) == grip_pid


# --------------------------------------------------------------------------------------------- evidence text
def _fmt_cm(value_m) -> str:
    return f"{float(value_m) * CM_PER_M:.1f} cm"


def evidence_text(outcome: RelationOutcome) -> str:
    """§C10.6 / §C1.7: <= 200 characters from the reason's registry meaning, the parts by pid, up to
    EVIDENCE_TEXT_ITEMS measured magnitudes (``evidence["text_items"]``: [label, value, unit] with unit "m" ->
    centimetres, "rad" -> degrees, "mm" -> centimetres, "px" -> image pixels, "count" -> a plain count), the short
    flags of ``evidence["text_flags"]`` (module-written words, e.g. "comotion unverified") and the first thresholds
    by name and value. Never world axes, signed vectors, plan texts or vocabulary words (labels are written by this
    module)."""
    pids = outcome.pids or {}
    obj, ref = pids.get("obj"), pids.get("ref")
    grips = [g for g in (pids.get("grip") or []) if g and g != obj]
    who = f"part {obj}" if obj else "no part"
    if grips:
        who += " via part " + "/".join(grips)
    if ref:
        who += f" onto part {ref}"
    text = f"{outcome.relation} {outcome.status} for {who}: {outcome.reason}"
    items = []
    for item in (outcome.evidence or {}).get("text_items") or []:
        try:
            label, value, unit = str(item[0]), float(item[1]), str(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if not math.isfinite(value):
            continue
        if unit == "rad":
            items.append(f"{label} {math.degrees(value):.0f} deg")
        elif unit == "mm":
            items.append(f"{label} {value / MM_PER_CM:.1f} cm")
        elif unit == "count":
            items.append(f"{label} {value:.0f}")
        elif unit == "px":
            items.append(f"{label} {value:.0f} px")
        else:
            items.append(f"{label} {_fmt_cm(value)}")
        if len(items) >= EVIDENCE_TEXT_ITEMS:
            break
    if items:
        text += "; " + ", ".join(items)
    flags = [str(f).replace("_", " ") for f in (outcome.evidence or {}).get("text_flags") or [] if f]
    if flags:
        text += "; " + ", ".join(flags)
    names = []
    for th in outcome.thresholds[:EVIDENCE_TEXT_ITEMS - 1]:
        unit = th.name.rsplit("_", 1)[-1]
        if unit == "m":
            names.append(f"{th.name} {_fmt_cm(th.value)}")
        elif unit == "rad":
            names.append(f"{th.name} {math.degrees(th.value):.0f} deg")
        else:
            names.append(f"{th.name} {th.value:.3g}")
    if names:
        text += "; " + ", ".join(names)
    return text[:EVIDENCE_TEXT_MAX_CHARS]


# ================================================================================================ Calibrate+
def _indexed_frames(frames: Mapping, prefix: str) -> list:
    keyed = []
    for key, frame in (frames or {}).items():
        if isinstance(key, str) and key.startswith(prefix + "[") and key.endswith("]"):
            try:
                keyed.append((int(key[len(prefix) + 1:-1]), frame))
            except ValueError:
                continue
    return [frame for _, frame in sorted(keyed, key=lambda item: item[0])]


def calibrate_plus(ctx):
    """Spec §4.5 Calibrate+ orchestration (contract §C10; v1 ``_calibrate`` pattern re-written): CalibratePlus with a
    probe-tracker SurveyRecorder (grid only, placed on the reference frame, the external camera recorded) ->
    gripper_quantities (G1-G4, G5, G6) -> result(None) -> CameraModel.decide (R-2f) -> result(camera) ->
    ExternalModel -> HandLocator -> ExternalProjector (the locator's anchor) -> fingertip band (open_hold vs the end
    of close_2) -> false-change maps and static noise -> frame alignment k -> mount decision -> V2Calibration ->
    scene.start. Sets ``ctx.cal``; emits "calibration"; records "calibrated" (calibrated_online /
    calibrated_default, or calibration_degraded when the external model or the locator is missing, the locator is
    degraded or the closes disagree). Returns the V2Calibration."""
    start = int(ctx.step)
    first_view = ctx.view
    start_quaternion = tuple(float(v) for v in first_view.quaternion_xyzw)
    holder = {}

    def factory(view):
        recorder = _recorder(ctx, view, camera=None, external=True, start_step=int(ctx.step), probe_tracker=True,
                             grid_at_reference=True)
        holder["recorder"] = recorder
        return CalibratePlus(view, recorder=recorder, robot=ctx.robot)
    result, primitive = yield from _run(factory)
    recorder = holder["recorder"]
    if not isinstance(primitive, CalibratePlus) or not primitive.done:
        raise RuntimeError("Calibrate+ did not run to its end (a preemption before any bus invariant is a defect)")
    record = primitive.record()
    quantities = primitives.gripper_quantities(record)
    g5 = primitives.g5_present_threshold(record.gripper, quantities["sigma_close_mm"])
    g6 = primitives.g6_hold_floor(record.gripper, quantities["sigma_close_mm"])
    g3 = quantities["thresholds"][2]
    loom = recorder.result(None)
    f_hat = (loom.loom or {}).get("f_hat") if loom.loom is not None else None
    camera = geometry.CameraModel.decide(f_hat, r2f_granted=bool(robot_facts.RULINGS.get(
        "R-2f_focal_length_as_robot_fact")))
    survey = recorder.result(camera)
    external = geometry.ExternalModel.fit(survey.external_samples)
    locator = geometry.HandLocator.fit(survey.external_samples, external, ctx.robot) if external is not None else None
    anchor = locator.anchor() if locator is not None else None
    projector = None if anchor is None else geometry.ExternalProjector(external, *anchor)
    frames = survey.frames or {}
    view = ctx.view
    if frames.get("open") is not None and frames.get("closed") is not None:
        band = geometry.fingertip_band_mask(frames["open"], frames["closed"])
    else:
        band = np.zeros(np.asarray(view.wrist_gray_upright).shape[:2], dtype=bool)
    statics = _indexed_frames(frames, "static")
    ext_statics = _indexed_frames(frames, "ext_static")
    false_wrist, sigma_wrist = geometry.false_change_map(statics) if statics else (None, 0.0)
    false_ext, sigma_ext = geometry.false_change_map(ext_statics) if ext_statics else (None, sigma_wrist)
    turns = geometry.frame_alignment_turns(view.R, external)
    mount = robot_facts.mount_decision(view.state).to_dict()
    thresholds = list(quantities["thresholds"]) + [g5, g6] + list(camera.thresholds)
    if external is not None:
        thresholds += list(external.thresholds)
    if locator is not None:
        thresholds += list(getattr(locator, "thresholds", []) or [])
    cal = V2Calibration(
        gripper=record.gripper, sigma_close_mm=float(quantities["sigma_close_mm"]),
        sigma_g_mm_per_step=float(quantities["sigma_g_mm_per_step"]),
        free_yaw_rate_rad_per_step=float(quantities["free_yaw_rate_rad_per_step"]), camera=camera, external=external,
        locator=locator, projector=projector, fingertip_band=band,
        false_change={"wrist": false_wrist, "external": false_ext},
        sigma_px={"wrist": float(sigma_wrist), "external": float(sigma_ext)}, sigma_static_grey=float(sigma_wrist),
        quarter_turns=int(turns), start_pose=tuple(float(v) for v in view.state),
        start_quaternion=start_quaternion, survey=survey, mount=mount, thresholds=thresholds)
    ctx.cal = cal
    ctx.memory["gripper"] = {"g3": g3, "g5": g5, "g6": g6, "quantities": jsonable(
        {k: v for k, v in quantities.items() if k != "thresholds"})}
    hf_summary = ctx.scene.start(cal, view)
    degraded = (external is None or locator is None or bool(getattr(locator, "degraded", False))
                or not quantities.get("closes_agree", True))
    reason = "calibration_degraded" if degraded else str(result.get("reason") or "calibrated_online")
    if reason not in ("calibrated_online", "calibrated_default", "calibration_degraded"):
        reason = "calibration_degraded"
    summary = cal.summary()
    summary["hf_summary"] = hf_summary
    _emit(ctx, "calibration", **summary)
    evidence = {"primitive": {"reason": result.get("reason"), "status": result.get("status"),
                              "actions": result.get("actions")},
                "closes_agree": quantities.get("closes_agree"), "external": external is not None,
                "locator": locator is not None, "projector": projector is not None,
                "locator_degraded": bool(getattr(locator, "degraded", False)), "quarter_turns": int(turns),
                "camera": camera.to_dict(), "survey": survey.summary(),
                "text_items": [["open gap", float(record.gripper.open_gap_mm), "mm"]]}
    _outcome(ctx, "calibrated", "succeeded", reason, pids=dict(NO_STEP_PIDS), evidence=evidence,
             thresholds=thresholds, start=start)
    return cal


# ============================================================================================ registration
def _seed_lift(ctx, pid, view) -> None:
    """The start-view lift of a part's wrist plan seed (scene.estimate_seed_xyz): a hover reference when the
    start registration is missing or less certain than the seed itself (kept in ctx.memory["seed_lift"])."""
    part = ctx.scene.parts.get(pid)
    if part is None:
        return
    for seed in part.seeds:
        if seed.camera != "wrist":
            continue
        try:
            lifted = ctx.scene.estimate_seed_xyz(seed, view)
        except (ValueError, TypeError, KeyError, IndexError):
            lifted = None
        if lifted is not None:
            xyz, cov = lifted
            ctx.memory.setdefault("seed_lift", {})[pid] = (np.asarray(xyz, dtype=float).copy(),
                                                           np.asarray(cov, dtype=float).copy())
            return


def _registration_outcome(ctx, pid, result, *, start, phase, retries=None, extra=None) -> RelationOutcome:
    _registration_event(ctx, pid, phase)
    thresholds = ctx.scene.registration_thresholds(pid)
    evidence = {"phase": phase, **(extra or {})}
    if isinstance(result, PartRecord):
        flags = []
        if result.partial:
            flags.append("partial_registration")
        if result.provisional:
            flags.append("provisional_registration")
        if result.top_is_lower_bound():
            flags.append("lower_bound_top")
        evidence.update(record=result.to_dict(), flags=flags, **{flag: True for flag in flags})
        if result.top_z is not None:
            evidence["text_items"] = [["top height sigma", float(result.sigma_h_m or 0.0), "m"]]
        return _outcome(ctx, "registered", "succeeded", "registered", pids=_pids(pid), evidence=evidence,
                        thresholds=thresholds, start=start, retries=retries)
    reason = getattr(result, "reason", None) or "part_not_found"
    evidence.update(failure=getattr(result, "evidence", None))
    return _outcome(ctx, "registered", "failed", _reason_code(reason, "part_not_found"), pids=_pids(pid),
                    evidence=evidence, thresholds=list(thresholds) + list(getattr(result, "thresholds", ()) or ()),
                    start=start, retries=retries)


def register_regions(ctx, plan, roles):
    """§C3.6 step 3: every part with a region role (carried, ref, effect, keepout, hold_true) registers with an empty
    hand from the current (start) view, partial allowed [C-38]; every grip-only part registers provisionally [C-37].
    A failure gets one point_coarse (wrist, region_seed) carrying the measured reason and one more registration from
    the same view. Only then (the view no longer needed) does a REGION part that still fails get one empty-hand survey
    hover when the hand is empty (spec §4.8 registered: hover -> ParallaxProbe -> PartRecord). One "registered"
    outcome per part, in plan order. Parts with no role (re-registration of a subset) are skipped. Returns the
    outcomes."""
    order = [p.pid for p in plan.parts]
    region = [pid for pid in order if set(roles.get(pid, ())) & set(REGION_ROLES)]
    grip_only = [pid for pid in order if roles.get(pid) and pid not in region]
    results, retries, starts = {}, {}, {}
    for pid in region + grip_only:
        if pid not in ctx.scene.parts:
            continue
        starts[pid] = int(ctx.step)
        provisional = pid in grip_only
        _seed_lift(ctx, pid, ctx.view)
        result = ctx.scene.register(pid, ctx.view, empty_hand=ctx.holding is None, provisional=provisional,
                                    allow_partial=not provisional)
        retries[pid] = 0
        if not isinstance(result, PartRecord):
            retries[pid] = 1
            _registration_event(ctx, pid, "start_view")
            reason = str(getattr(result, "reason", "part_not_found"))
            seed = yield from _coarse_seed(ctx, pid, purpose="region_seed", previous=reason)
            if seed is not None:
                result = ctx.scene.register(pid, ctx.view, seed=seed, empty_hand=ctx.holding is None,
                                            provisional=provisional, allow_partial=not provisional)
        results[pid] = result
    for pid in region:
        if pid in results and not isinstance(results[pid], PartRecord) and ctx.holding is None:
            _registration_event(ctx, pid, "start_view_reseed")
            surveyed = yield from _survey(ctx, pid)
            results[pid] = surveyed.get("record") or results[pid]
            retries[pid] = 2
    out = []
    kept = ctx.memory.setdefault("start_records", {})
    for pid in region + grip_only:
        if pid in results:
            if isinstance(results[pid], PartRecord) and results[pid].xyz is not None:
                kept.setdefault(pid, results[pid])     # blocker-2 spec BR-1: the earliest record of the part
            out.append(_registration_outcome(ctx, pid, results[pid], start=starts[pid], phase="start_view",
                                             retries=retries[pid]))
    return out


def _coarse_seed(ctx, pid, *, purpose, previous):
    """One point_coarse on the current wrist image -> a coarse Seed or None."""
    need = PointNeed("point_coarse", pid, _text_of(ctx, pid), "wrist", None, _turns(ctx), purpose,
                     previous_rejected=str(previous)[:PREVIOUS_REJECTED_MAX_CHARS] if previous else None)
    answer = yield need
    _count(ctx, "pointing_calls")
    reply = getattr(answer, "reply", None) if isinstance(answer, PointAnswer) else None
    uv = None if reply is None else reply.uv_upright
    _emit(ctx, "pointing", kind="point_coarse", pid=pid, camera="wrist", purpose=purpose,
          answer_id=None if reply is None else reply.answer_id, uv_upright=uv, round=0,
          check={"ok": uv is not None, "reason": None if uv is not None else "not_visible"}, source="relations")
    if uv is None:
        return None
    ctx.note_accepted("point_coarse")
    return Seed("wrist", tuple(float(v) for v in uv), int(ctx.step), "coarse",
                max(float(reply.quantization_px), 1.0), tuple(float(v) for v in ctx.view.state))


# ================================================================================================== moves
def _reach_blocked(ctx, offset) -> dict | None:
    """v1 reach-limit memory (agent_speedup_v1.py:546-556, :783-786, restated): a horizontal push of more than
    REACH_LIMIT_PUSH_M toward a limit recorded within REACH_LIMIT_RADIUS_M of the hand is refused."""
    position = np.asarray(ctx.view.position, dtype=float)
    push_xy = np.asarray(offset, dtype=float)[:2]
    for limit in reversed(ctx.memory.get("reach_limits", [])):
        xyz = np.asarray(limit["xyz"], dtype=float)
        near = (float(np.linalg.norm(position[:2] - xyz[:2])) <= REACH_LIMIT_RADIUS_M
                and abs(float(position[2] - xyz[2])) <= REACH_LIMIT_RADIUS_M)
        push = float(push_xy @ np.asarray(limit["direction_xy"], dtype=float))
        if near and push > REACH_LIMIT_PUSH_M:
            return {"limit": limit, "push_m": push}
    return None


def _note_reach_limit(ctx, result) -> None:
    evidence = result.get("evidence") or {}
    direction = evidence.get("blocked_direction_xy")
    end = evidence.get("end_xyz")
    commanded = np.asarray(evidence.get("commanded_offset_m") or evidence.get("offset_m") or (0.0, 0.0, 0.0),
                           dtype=float).reshape(-1)[:3]
    if end is None or commanded.shape != (3,) or float(commanded[2]) > REACH_LIMIT_MAX_LIFT_M:
        return
    horizontal = commanded[:2]
    norm = float(np.linalg.norm(horizontal))
    if direction is None:
        if norm < REACH_LIMIT_PUSH_M:
            return
        direction = horizontal / norm
    direction = np.asarray(direction, dtype=float).reshape(-1)[:2]
    if float(np.linalg.norm(direction)) < 1e-9:
        return
    direction = direction / float(np.linalg.norm(direction))
    limits = ctx.memory.setdefault("reach_limits", [])
    limits.append({"step": int(ctx.step), "xyz": [float(v) for v in np.asarray(end, dtype=float)[:3]],
                   "direction_xy": [float(v) for v in direction]})
    del limits[:-REACH_LIMIT_MEMORY]


def _leg_waypoints(ctx, waypoint):
    """The WaypointReach legs toward one waypoint: offsets of at most M2_SEGMENT_M (the +-8 cm field grid)."""
    offset = np.asarray(waypoint, dtype=float) - np.asarray(ctx.view.position, dtype=float)
    span = float(np.max(np.abs(offset)))
    count = max(1, int(math.ceil(span / planner.M2_SEGMENT_M - 1e-9)))
    return offset / count


def free_move(ctx, goal_grip_xyz, *, gripper: str, orientation_xyzw=None, skip_parts=(), path=None):
    """Empty-hand transit (planner.free_path: up, over at transit_z, down; M1, M2) as chained v1 WaypointReach legs
    with the orientation held (default: the current one), the v1 reach-limit memory and MIN_MOVE_OFFSET_M arrival,
    then a B4 settle. Returns {"ok", "reason", "path", "legs"}."""
    goal = np.asarray(goal_grip_xyz, dtype=float).reshape(3)
    quat = np.asarray(ctx.view.quaternion_xyzw if orientation_xyzw is None else orientation_xyzw, dtype=float)
    if path is None:
        path = planner.free_path(np.asarray(ctx.view.position, dtype=float), goal, ctx.scene.hf,
                                 _keepouts(ctx, skip=skip_parts), ctx.robot, _pose(ctx),
                                 parts=_other_parts(ctx, skip=skip_parts))
    legs = []
    info = {"ok": True, "reason": "goal_reached", "path": planner.jsonable_result(
        {k: path.get(k) for k in ("waypoints", "transit_z", "required_z", "swept_max_z", "lower_bound_parts")}),
        "legs": legs}
    for waypoint in [np.asarray(w, dtype=float) for w in path.get("waypoints") or []]:
        for _ in range(REACH_TRIES_PER_WAYPOINT):
            remaining = waypoint - np.asarray(ctx.view.position, dtype=float)
            if float(np.linalg.norm(remaining)) <= MIN_MOVE_OFFSET_M:
                break
            offset = _leg_waypoints(ctx, waypoint)
            blocked = _reach_blocked(ctx, offset)
            if blocked is not None:
                info.update(ok=False, reason="reach_limit", blocked=jsonable(blocked))
                return info
            result, _ = yield from _run(lambda v, o=offset: WaypointReach(v, o, gripper=gripper, orientation_xyzw=quat,
                                                                          name="free_move"))
            reason = str(result.get("reason") or "")
            legs.append({"reason": reason, "steps": result.get("actions")})
            if reason == "reach_limit":
                _note_reach_limit(ctx, result)
            if _preempted(result) or reason in MOVE_ABORT_REASONS:
                info.update(ok=False, reason=_reason_code(reason, "reach_limit" if reason == "reach_limit"
                                                          else "pose_drift"), primitive_reason=reason)
                return info
        if float(np.linalg.norm(waypoint - np.asarray(ctx.view.position, dtype=float))) > MIN_MOVE_OFFSET_M:
            info.update(ok=False, reason="pose_drift", primitive_reason=legs[-1]["reason"] if legs else None)
            return info
    yield from _run(lambda v: primitives.b4_settle(v, gripper=gripper, orientation_xyzw=quat, name="move_settle"))
    info["residual_m"] = float(np.linalg.norm(goal - np.asarray(ctx.view.position, dtype=float)))
    _emit(ctx, "transit", kind="free", **{k: info["path"].get(k) for k in ("transit_z", "swept_max_z")},
          waypoints=len(path.get("waypoints") or []), legs=len(legs), residual_m=info["residual_m"])
    return info


def _contact_backoff(ctx, offset, *, name):
    """v1 :903-940 back-off after a held-object contact: HELD_CONTACT_BACKOFF_M against a rising move, else straight
    up (gripper closed). Returns the back-off move's reason."""
    offset = np.asarray(offset, dtype=float).reshape(3)
    rising = float(offset[2]) > 0.0 and float(np.linalg.norm(offset)) > 1e-9
    if rising:
        vector = -offset / float(np.linalg.norm(offset)) * HELD_CONTACT_BACKOFF_M
    else:
        vector = np.array([0.0, 0.0, HELD_CONTACT_BACKOFF_M])
    travel = float(np.linalg.norm(vector))
    back, _ = yield from _run(lambda v, a=tuple(vector / travel), t=travel: AnchoredMove(
        v, axis=a, travel_m=t, gripper="close", name=name))
    return back.get("reason")


def _preemption_info(result) -> dict:
    """The bus invariant behind a preempted primitive result: its key and report reason / evidence (§C3.3)."""
    evidence = (result or {}).get("evidence") or {}
    report = evidence.get("report") or {}
    return {"invariant": evidence.get("invariant"), "cause": (report.get("evidence") or {}).get("cause"),
            "tilt_change_deg": (report.get("evidence") or {}).get("tilt_change_deg")}


def _template_lost(ctx, preemption) -> bool:
    """A held-scope co-motion preemption whose cause is a LOST external template (no measured displacement) while
    the finger gap still holds above G6 (``_reanchor_comotion``)."""
    pre = preemption or {}
    return (str(pre.get("invariant") or "").startswith("comotion:") and pre.get("cause") == "template_lost"
            and ctx.holding is not None and float(ctx.view.gap_mm) >= float(_g6(ctx).value))


def _reanchor_comotion(ctx, key):
    """BRR-7 (found by the first unpatched carry, FakeTwoViewSim place_on via the thin part): the carry-scope
    CoMotionObserver preempted grip_not_on_obj with cause ``template_lost`` 18 cm into the carry while the object
    was held (truth) and the gap stayed at 12.9 mm: its pick-time external NCC template fell to peak 0.57-0.59
    (below the 0.6 match floor) as the object's appearance changed, and every repair retry was preempted again
    after 1-3 steps. A left-behind object keeps its template matched in place (cause ``displaced``), so a lost
    template with the hold intact is no evidence of a loss: the invariant is re-anchored ONCE per move with a fresh
    template of the object's predicted top point in the current external view (the observer restarts from this
    pose: preempt=True, scope held). The HeldFrame's own template is kept (AimCheck must never match a template cut
    at the predicted point). Returns {"ok", "uv"|"why"}."""
    held = ctx.holding
    projector = getattr(ctx.scene, "projector", None)
    part = ctx.scene.parts.get(getattr(held, "pid", None))
    if held is None or projector is None or part is None or part.top_z is None:
        return {"ok": False, "why": "no_projector_or_top"}
    pose = _pose(ctx)
    top = float(part.top_z.value) + float(pose.grip[2]) - float(held.g_close)
    uv, _ = projector.project(np.r_[held.object_ref_point(pose)[:2], top])
    try:
        track = NCCTrack(_external_gray(ctx.view), (float(uv[0]), float(uv[1])))
    except ValueError as exc:
        return {"ok": False, "why": str(exc)}
    if _window_on_robot(ctx, uv, int(track.radius)):
        # HS-H2 (gap round §2.5): a template window holding hand pixels rides with the hand, not the object
        return {"ok": False, "why": "template_on_robot", "uv": [float(v) for v in uv]}
    old = ctx.monitor.observer(key) if key in ctx.monitor.keys() else None
    ctx.monitor.remove(key)
    fresh = _comotion(ctx, dataclasses.replace(held, ext_template=track), held.pid, rotation_verdict=False)
    if key in (ctx.memory.get("carry_comotion") or {}):
        ctx.memory["carry_comotion"][key] = fresh            # HS-H3: the Upright tilt_fn follows the fresh observer
    _retarget_tilt_fn(ctx, old, fresh)
    _bus_add(ctx, key, fresh, preempt=True, scope="held")
    return {"ok": True, "uv": [float(v) for v in uv]}


def _window_on_robot(ctx, uv, radius) -> bool:
    """HS-H2: the (2r+1)^2 NCC window at ``uv`` contains a pixel of the scene's S-16' hand mask
    (``hand_swept_mask('external', view, (0, 0))``) in the current external view. False when no mask is available."""
    try:
        hand = ctx.scene.hand_swept_mask("external", ctx.view, (0.0, 0.0))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
    if hand is None:
        return False
    hand = np.asarray(hand, dtype=bool)
    u, v, r = int(round(float(uv[0]))), int(round(float(uv[1]))), int(radius)
    h, w = hand.shape[:2]
    return bool(hand[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)].any())


def held_move(ctx, waypoints, *, kind="carry"):
    """Held transit (spec §4.9 Carry, M2): chained HeldReach legs (v1 held checks: hold_lost, held_object_contact,
    reach_limit; the grasp orientation held within M3). A held-object contact backs off by HELD_CONTACT_BACKOFF_M
    (v1 :903-940 rule: against a rising move, else straight up) and fails held_object_contact; a lost hold fails
    lost_in_transport; a keep-out preemption (O5) backs off the same way and fails keepout_contact (contract §C10.2
    step 11); another bus preemption fails with its reason and carries the invariant key (``preemption``: the caller
    restores an upright violation). Returns {"ok", "reason", "legs"}."""
    legs = []
    info = {"ok": True, "reason": "goal_reached", "legs": legs}
    hold_floor = float(_g6(ctx).value)
    gap_ref = ctx.hold_gap_ref_mm
    reanchored = False
    for waypoint in [np.asarray(w, dtype=float) for w in waypoints]:
        for _ in range(REACH_TRIES_PER_WAYPOINT):
            remaining = waypoint - np.asarray(ctx.view.position, dtype=float)
            if float(np.linalg.norm(remaining)) <= MIN_MOVE_OFFSET_M:
                break
            offset = _leg_waypoints(ctx, waypoint)
            blocked = _reach_blocked(ctx, offset)
            if blocked is not None:
                info.update(ok=False, reason="reach_limit", blocked=jsonable(blocked))
                return info
            result, _ = yield from _run(lambda v, o=offset: primitives.HeldReach(
                v, o, hold_floor_mm=hold_floor, gap_ref_mm=gap_ref, grasp_quaternion=ctx.grasp_quaternion,
                name=f"held_{kind}"))
            legs.append({"reason": result.get("reason"), "steps": result.get("actions")})
            reason = str(result.get("reason") or "")
            if reason == "reach_limit":
                _note_reach_limit(ctx, result)
            if _preempted(result) and not reanchored and _template_lost(ctx, _preemption_info(result)):
                reanchored = True
                info["comotion_reanchored"] = _reanchor_comotion(ctx, _preemption_info(result)["invariant"])
                if info["comotion_reanchored"]["ok"]:
                    continue
            if _preempted(result):
                code = _reason_code(reason, "lost_in_transport")
                code = {"grip_not_on_obj": "lost_in_transport", "hold_lost": "lost_in_transport",
                        "grip_unstable": "object_rotated_in_hand"}.get(code, code)
                info.update(ok=False, reason=code, primitive_reason=reason, preemption=_preemption_info(result))
                if code == "keepout_contact":
                    info["backoff"] = yield from _contact_backoff(ctx, offset, name="keepout_backoff")
                return info
            if reason == "hold_lost" or ctx.view.gap_mm < hold_floor:
                info.update(ok=False, reason="lost_in_transport", primitive_reason=reason)
                return info
            if reason == "held_object_contact":
                _disturb(ctx, getattr(ctx.holding, "pid", None), "transit_contact")       # X-1
                back = yield from _contact_backoff(ctx, offset, name="contact_backoff")
                info.update(ok=False, reason="held_object_contact", primitive_reason=reason, backoff=back)
                return info
            if result.get("status") != "succeeded":
                info.update(ok=False, reason=_reason_code(reason, "lateral_unreachable"), primitive_reason=reason)
                return info
    return info


def _ensure_open(ctx):
    """v1 ``_ensure_open``: open the gripper unless it is already within OPEN_TOLERANCE_MM of the open gap."""
    if ctx.view.gap_mm < _g_open_mm(ctx) - OPEN_TOLERANCE_MM:
        yield from _run(lambda v: GripperSettlePrimitive("open", v, config=OPEN_SETTLE, name="open"))


def _lift(ctx, travel_m, *, gripper, name, orientation_xyzw=None):
    """A guarded straight lift along world +z (stop on contact)."""
    if travel_m <= MIN_MOVE_OFFSET_M / 2.0:
        return {"reason": "travel_reached", "status": "succeeded"}
    quat = None if orientation_xyzw is None else np.asarray(orientation_xyzw, dtype=float)
    result, _ = yield from _run(lambda v: AnchoredMove(v, axis=WORLD_UP, travel_m=float(travel_m), gripper=gripper,
                                                       orientation_xyzw=quat, stop_on_contact=True, name=name))
    return result


def _arrive(ctx, hover_xyz, *, orientation_xyzw=None, skip_parts=()):
    """Decision 1 §2.6 hover arrival: free_move to the hover + HOVER_LOCATE_M (settled), then the guarded
    HOVER_LOCATE_M descent onto the hover (``hover_arrival_descent``: RC-8 stall, G3 touch), then
    ``scene.reanchor_from(settled, arrived)`` (the HandLocator reading of the projector). A stop other than the
    travel is transit_contact: lift TRANSIT_CONTACT_LIFT_M. Returns {"ok", "reason", "move", "descent",
    "reanchor"}."""
    target = np.asarray(hover_xyz, dtype=float).reshape(3)
    above = target + np.array([0.0, 0.0, HOVER_LOCATE_M])
    move = yield from free_move(ctx, above, gripper="open", orientation_xyzw=orientation_xyzw, skip_parts=skip_parts)
    info = {"ok": move["ok"], "reason": move["reason"], "move": move, "hover_xyz": target}
    if not move["ok"]:
        return info
    settled = ctx.view
    quat = np.asarray(settled.quaternion_xyzw, dtype=float)
    travel = max(float(settled.position[2] - target[2]), MIN_MOVE_OFFSET_M)
    result, _ = yield from _run(lambda v: primitives.hover_arrival_descent(
        v, travel_m=travel, touch_threshold_mm=_g3(ctx), orientation_xyzw=quat, name="hover_arrival"))
    info["descent"] = {"reason": result.get("reason"), "steps": result.get("actions")}
    if result.get("reason") != "travel_reached":
        lift = yield from _lift(ctx, TRANSIT_CONTACT_LIFT_M, gripper="open", name="transit_contact_lift")
        info.update(ok=False, reason="transit_contact", lift=lift.get("reason"))
        return info
    try:
        info["reanchor"] = jsonable(ctx.scene.reanchor_from(settled, ctx.view))
    except (ValueError, TypeError, AttributeError) as exc:
        info["reanchor"] = {"ok": False, "status": "error", "reason": type(exc).__name__}
    return info


# =============================================================================================== pointing
def _inputs(ctx, *, sigma_align_m=None) -> "planner.PlannerInputs":
    locator = getattr(ctx.cal, "locator", None)
    return planner.PlannerInputs.build(ctx.robot, ctx.view.state, g_open_mm=_g_open_mm(ctx),
                                       start_quaternion=_start_quaternion(ctx), sigma_align_m=sigma_align_m,
                                       mount=getattr(ctx.cal, "mount", None),
                                       sigma_loc_px=getattr(locator, "sigma_loc_px", None))


def _in_image(uv, size) -> bool:
    return uv is not None and 0.0 <= float(uv[0]) <= size - 1.0 and 0.0 <= float(uv[1]) <= size - 1.0


def _ask_point(ctx, kind, pid, *, camera, purpose, center_uv, previous, round_index, marker_uv=None):
    """One PointNeed -> (PointReply or None). The pointing event carries no check (the caller emits it).
    ``marker_uv`` (blocker-2 spec BR-3: the rejected point, upright px of ``camera``) is sent only when the context's
    client draws markers (``ctx.pointing_marker``, BC-2) and the point lies in the image."""
    turns = _turns(ctx) if camera == "wrist" else 0
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1]) if camera == "wrist" else \
        float(np.asarray(ctx.view.external_rgb_upright).shape[1])
    centre = None
    if center_uv is not None:
        centre = tuple(float(np.clip(v, 0.0, size - 1.0)) for v in center_uv)
    marker = None
    if marker_uv is not None and bool(getattr(ctx, "pointing_marker", False)) and _in_image(marker_uv, size):
        marker = tuple(float(v) for v in marker_uv)
    need = PointNeed(kind, pid, _text_of(ctx, pid), camera, centre if kind == "point_fine" else None, turns, purpose,
                     previous_rejected=None if not previous else str(previous)[:PREVIOUS_REJECTED_MAX_CHARS],
                     marker_uv=marker)
    answer = yield need
    _count(ctx, "pointing_calls")
    reply = getattr(answer, "reply", None) if isinstance(answer, PointAnswer) else None
    _emit(ctx, "pointing", kind=kind, pid=pid, camera=camera, purpose=purpose,
          answer_id=None if reply is None else reply.answer_id,
          uv_upright=None if reply is None else reply.uv_upright,
          quantization_px=None if reply is None else reply.quantization_px, round=int(round_index or 0),
          schema_error=bool(getattr(answer, "schema_error", False)), check=None, source="relations",
          center_uv=None if centre is None or kind != "point_fine" else list(centre),
          marker_uv=None if marker is None else list(marker))
    return reply


def _seed_of(ctx, reply, source) -> Seed:
    return Seed("wrist", tuple(float(v) for v in reply.uv_upright), int(ctx.step), source,
                max(float(reply.quantization_px), 1.0), tuple(float(v) for v in ctx.view.state))


def _part_height(ctx, pid) -> float | None:
    part = ctx.scene.parts.get(pid)
    if part is not None and part.top_z is not None:
        return float(part.top_z.value)
    if part is not None and part.xyz is not None:
        return float(part.xyz[2])
    lift = ctx.memory.get("seed_lift", {}).get(pid)
    return None if lift is None else float(lift[0][2])


def _pointed_visibility(ctx, pid, seed, *, z=None):
    """P-2 on the POINTED pixel (decision 1 §2.4 P-E visibility, applied to the answer; zero calls): when the pointed
    pixel lies below the fingertip band's top row minus (kappa VIS_PROBE_LEG_M + FT_PROBE_RADIUS_PX) at the part's
    height estimate (``z``, else ``_part_height``; the r0_out probe leg moves it down by kappa D2), the hand moves
    along -R0 (horizontal) by the excess / kappa and the pointed point is carried into the new view (back-projected
    at that height, re-projected; no new pointing). Returns (seed, info or None)."""
    row = _band_top_row(ctx)
    z = _part_height(ctx, pid) if z is None else float(z)
    if row is None or z is None:
        return seed, None
    pose = _pose(ctx)
    camera = WristCamera(_f_px(ctx))
    origin, direction = camera.ray(pose, seed.uv)
    if abs(float(direction[2])) < 1e-9:
        return seed, None
    point = origin + (z - float(origin[2])) / float(direction[2]) * direction
    depth = float((point - pose.camera) @ pose.R[:, 2])
    if not depth > 0.0:
        return seed, None
    kappa = camera.kappa_at(depth)
    excess = float(seed.uv[1]) - (float(row) - (kappa * planner.VIS_PROBE_LEG_M + geometry.FT_PROBE_RADIUS_PX))
    if excess <= 0.0:
        return seed, None
    r0 = np.asarray(pose.R[:, 0], dtype=float).copy()
    r0[2] = 0.0
    if float(np.linalg.norm(r0)) < 1e-9:
        return seed, None
    r0 = r0 / float(np.linalg.norm(r0))
    shift = min(excess / kappa, planner.M2_SEGMENT_M)
    moved = yield from _shift_hover(ctx, -r0[:2], shift)
    uv, _ = camera.project(_pose(ctx), point)
    info = {"shift_m": shift, "from_uv": list(seed.uv), "move": moved.get("reason")}
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
    if uv is None or not _in_image(uv, size):
        info["lost"] = True
        return seed, info
    info["to_uv"] = [float(v) for v in uv]
    carried = Seed("wrist", (float(uv[0]), float(uv[1])), int(ctx.step), seed.source, seed.quantization_px,
                   tuple(float(v) for v in ctx.view.state))
    return carried, info


def point_part(ctx, pid, *, camera, purpose, center_uv, sigma_uv_px, previous_rejected, round_index=None,
               marker_uv=None):
    """The pointing path only (contract §C10): point_coarse first when the part may lie outside the fine crop (the
    prediction is missing or outside the image, or its sigma exceeds COARSE_FIRST_SIGMA_PX, rulings item 9);
    else point_fine at the prediction; a fine 0 -> point_coarse on the current image -> point_fine at the coarse
    cell centre (spec §3.2; a second fine 0 keeps the coarse seed). ``marker_uv`` rings a rejected point on every
    ask of the path (blocker-2 spec BR-3, sent only when ``ctx.pointing_marker``). Returns {"seed", "reason", "calls",
    "path", "answer": (stage, answer id) of the reply the seed came from, "centre": the first fine crop centre}."""
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
    coarse_first = (center_uv is None or not _in_image(center_uv, size)
                    or (sigma_uv_px is not None and float(sigma_uv_px) > COARSE_FIRST_SIGMA_PX))
    calls, path = 0, []
    first_centre = None if coarse_first else [float(v) for v in center_uv]
    if not coarse_first:
        reply = yield from _ask_point(ctx, "point_fine", pid, camera=camera, purpose=purpose, center_uv=center_uv,
                                      previous=previous_rejected, round_index=round_index, marker_uv=marker_uv)
        calls += 1
        path.append("fine")
        if reply is not None and reply.uv_upright is not None:
            return {"seed": _seed_of(ctx, reply, "fine"), "reason": None, "calls": calls, "path": path,
                    "answer": ("fine", reply.answer_id), "centre": first_centre}
        ctx.note_accepted("point_fine")              # a 0 answer that leads to the coarse stage (§C3.4)
    reply = yield from _ask_point(ctx, "point_coarse", pid, camera=camera, purpose=purpose, center_uv=None,
                                  previous=previous_rejected, round_index=round_index, marker_uv=marker_uv)
    calls += 1
    path.append("coarse")
    if reply is None or reply.uv_upright is None:
        return {"seed": None, "reason": "part_not_found", "calls": calls, "path": path, "answer": None,
                "centre": first_centre}
    coarse = _seed_of(ctx, reply, "coarse")
    coarse_answer = ("coarse", reply.answer_id)
    reply = yield from _ask_point(ctx, "point_fine", pid, camera=camera, purpose=purpose, center_uv=coarse.uv,
                                  previous=previous_rejected, round_index=round_index, marker_uv=marker_uv)
    calls += 1
    path.append("fine")
    if first_centre is None:
        first_centre = [float(v) for v in coarse.uv]
    if reply is not None and reply.uv_upright is not None:
        return {"seed": _seed_of(ctx, reply, "fine"), "reason": None, "calls": calls, "path": path,
                "answer": ("fine", reply.answer_id), "centre": first_centre}
    return {"seed": coarse, "reason": None, "calls": calls, "path": path, "answer": coarse_answer,
            "centre": first_centre}


def _check_event(ctx, pid, seed, check, round_index, kind) -> None:
    _emit(ctx, "pointing", kind="check", pid=pid, camera="wrist", purpose=kind, answer_id=None,
          uv_upright=list(seed.uv), quantization_px=seed.quantization_px, round=int(round_index),
          check={"ok": bool(check.get("ok")), "reason": check.get("reason"), "values": check.get("values")},
          source="relations")


def _probe_summary(result) -> dict:
    """The ParallaxProbe outcome as relation evidence: reason, status, steps and per leg its stage, reason, travel
    and finger contact (why a probe ended early is otherwise lost)."""
    evidence = result.get("evidence") or {}
    legs = [{k: leg.get(k) for k in ("stage", "reason", "travel_m", "steps", "contact", "end_xyz")}
            for leg in evidence.get("legs") or ()]
    return {"reason": result.get("reason"), "status": result.get("status"), "steps": result.get("actions"),
            "legs": legs, "net_displacement_m": evidence.get("net_displacement_m")}


def _return_to(ctx, xyz, quaternion_xyzw, *, name):
    """Back to a hover pose (``xyz``, held orientation) by one WaypointReach and a B4 settle; nothing when already
    within MIN_MOVE_OFFSET_M. Returns the reach reason."""
    offset = np.asarray(xyz, dtype=float).reshape(3) - np.asarray(ctx.view.position, dtype=float)
    if float(np.linalg.norm(offset)) < MIN_MOVE_OFFSET_M:
        return "goal_reached"
    quat = np.asarray(quaternion_xyzw, dtype=float)
    result, _ = yield from _run(lambda v: WaypointReach(v, offset, gripper="open", orientation_xyzw=quat, name=name))
    if _preempted(result):
        return result.get("reason")
    yield from _run(lambda v: primitives.b4_settle(v, gripper="open", orientation_xyzw=quat, name=name + "_settle"))
    return result.get("reason")


def _run_probe(ctx, make_recorder):
    """ParallaxProbe (D2, open hand, G3 finger touch) recording into ``make_recorder(view)`` (a fresh recorder per
    attempt). A probe that ends early (``probe_contact``: an RC-8 stall or a finger touch on a leg) returns to the
    probe's start pose and probes again, PROBE_RETRIES times, so the seeds stay valid for that pose; a contact that
    repeats is reported. Returns (outcome, recorder of the last attempt, [probe summary per attempt])."""
    attempts = []
    start_xyz = np.asarray(ctx.view.position, dtype=float).copy()
    start_quat = np.asarray(ctx.view.quaternion_xyzw, dtype=float).copy()
    while True:
        holder = {}

        def factory(v):
            holder["recorder"] = make_recorder(v)
            return primitives.ParallaxProbe(v, recorder=holder["recorder"], gripper="open",
                                            touch_threshold_mm=_g3(ctx))
        result, _ = yield from _run(factory)
        attempts.append(_probe_summary(result))
        if _preempted(result) or result.get("reason") != "probe_contact" or len(attempts) > PROBE_RETRIES:
            return result, holder.get("recorder"), attempts
        back = yield from _return_to(ctx, start_xyz, start_quat, name="probe_return")
        attempts[-1]["return"] = back
        if back != "goal_reached":
            return result, holder.get("recorder"), attempts


def _probe_and_register(ctx, pid, seed, *, label):
    """One hover measurement (spec §4.9 Grasp step 3): ParallaxProbe (D2) with a probe-tracker SurveyRecorder (the
    exact seed, the dense lattice at it, the scene's relief targets; the fingertip band excluded) ->
    scene.absorb_survey -> scene.register(seed) -> [C-38] the partial parts the same survey now sees whole are
    re-registered. Returns {"record", "failure", "probe", "label"}."""
    view = ctx.view
    targets = ctx.scene.survey_targets(pid, view)
    band = getattr(ctx.cal, "fingertip_band", None)
    seed_uv = tuple(float(v) for v in seed.uv)
    result, recorder, attempts = yield from _run_probe(ctx, lambda v: _recorder(
        ctx, v, camera=ctx.cal.camera, seeds=[(pid, seed_uv)], relief=targets.get("relief") or (), exclude_mask=band,
        external=False, start_step=int(ctx.step), probe_tracker=True, dense_uv=seed_uv))
    out = {"label": label, "probe": attempts[-1], "probe_attempts": len(attempts), "record": None, "failure": None}
    if len(attempts) > 1:
        out["probe_retried"] = attempts[:-1]
    if _preempted(result) or result.get("reason") != "probe_complete":
        out["failure"] = {"reason": _reason_code(result.get("reason"), "probe_contact"), "why": result.get("reason")}
        return out
    survey = recorder.result(ctx.cal.camera)
    out["cells_changed"] = int(ctx.scene.absorb_survey(survey, ctx.view))
    rec = ctx.scene.register(pid, ctx.view, seed=seed, empty_hand=ctx.holding is None)
    _registration_event(ctx, pid, label)
    if isinstance(rec, PartRecord):
        out["record"] = rec
    else:
        out["failure"] = {"reason": rec.reason, "why": (rec.evidence or {}).get("why"), "evidence": rec.evidence}
    if ctx.holding is None:
        for other in ctx.scene.partial_parts_in_view(ctx.view):
            if other == pid:
                continue
            again = ctx.scene.register(other, ctx.view, empty_hand=True, allow_partial=True)
            _registration_event(ctx, other, f"{label}_partial_reregistration")
            out.setdefault("reregistered", {})[other] = isinstance(again, PartRecord) and not again.partial
    return out


# ========================================================================== blocker-2 spec (BR-1 .. BR-5, 2026-10-19)
def _region_iou(a, b) -> float | None:
    """IoU of two records' HeightField region cells; None when either record has no region cells."""
    cells = []
    for record in (a, b):
        ij = getattr(record, "region_cells", None)
        if ij is None or not len(ij):
            return None
        cells.append({(int(i), int(j)) for i, j in np.asarray(ij, dtype=int).reshape(-1, 2)})
    union = len(cells[0] | cells[1])
    return len(cells[0] & cells[1]) / union if union else None


def _coincides(a, b) -> dict | None:
    """Blocker-2 spec BR-1: record ``a`` coincides with record ``b`` (both with a 3-D point) when their points lie within
    COINCIDE_XYZ_M or their regions overlap at IoU >= SHARED_REGION_IOU (a grip part registered with its object's
    region: live smokes 002 / 003, overlap 1.0). Returns {"distance_m", "iou"} when they coincide, else None."""
    if not (isinstance(a, PartRecord) and isinstance(b, PartRecord)) or a.xyz is None or b.xyz is None:
        return None
    distance = float(np.linalg.norm(np.asarray(a.xyz, dtype=float) - np.asarray(b.xyz, dtype=float)))
    iou = _region_iou(a, b)
    if distance <= COINCIDE_XYZ_M or (iou is not None and iou >= SHARED_REGION_IOU):
        return {"distance_m": distance, "iou": iou}
    return None


def _seed_lift_record(ctx, pid, part):
    """The start-view lift of the part's own wrist plan seed (``_seed_lift``) as a seeded record (no region, width or
    axis: only the lift's point and covariance), or None."""
    lift = ctx.memory.get("seed_lift", {}).get(pid)
    if lift is None or part is None:
        return None
    xyz, cov = lift
    cov = np.asarray(cov, dtype=float)
    return PartRecord(pid, part.text, part.move_prior, part.roles, seeds=list(part.seeds),
                      xyz=np.asarray(xyz, dtype=float).copy(), cov=cov.copy(),
                      sigma_h_m=math.sqrt(max(float(cov[2, 2]), 0.0)), status="seeded",
                      registration_step=part.registration_step if part.registration_step is not None
                      else int(ctx.step))


def _independent_reference(ctx, pid, obj):
    """Blocker-2 spec BR-1: the earliest estimate of grip part ``pid`` that is independent of its object ``obj``: the
    start-view record (``memory["start_records"]``, the same text), else the start-view seed lift, among those that
    coincide neither with the object's current record nor with the object's start record; the [C-37] sigma rule
    picks the lift over the start record when the start record's horizontal sigma exceeds HOVER_LIFT_SIGMA_RATIO x
    the lift's. Returns (record, source) or (None, None)."""
    part = ctx.scene.parts.get(pid)
    starts = ctx.memory.get("start_records") or {}
    others = [r for r in (ctx.scene.parts.get(obj), starts.get(obj)) if isinstance(r, PartRecord)]

    def independent(record):
        return record is not None and record.xyz is not None and all(_coincides(record, o) is None for o in others)
    first = starts.get(pid)
    if first is not None and part is not None and first.text != part.text:
        first = None                            # a replan gave the id to another part
    first = first if independent(first) else None
    lifted = _seed_lift_record(ctx, pid, part)
    lifted = lifted if independent(lifted) else None
    if first is not None and (lifted is None or _sigma_xy(first) <= HOVER_LIFT_SIGMA_RATIO * _sigma_xy(lifted)):
        return first, "independent:start_view"
    if lifted is not None:
        return lifted, "independent:seed_lift"
    return None, None


def _predict_uv(ctx, record):
    """The record's 3-D point projected into the current wrist view: (uv, sigma_px, depth_m) with the covariance
    carried through the projection Jacobian (the scene's ``predicted_uv`` rule on this record), or None."""
    if record is None or record.xyz is None:
        return None
    pose = _pose(ctx)
    camera = WristCamera(_f_px(ctx))
    xyz = np.asarray(record.xyz, dtype=float)
    uv, depth = camera.project(pose, xyz)
    if uv is None or not np.isfinite(uv).all():
        return None
    d = xyz - pose.camera
    a, b, z = float(d @ pose.R[:, 1]), float(d @ pose.R[:, 0]), float(d @ pose.R[:, 2])
    f = camera.f_px
    J = np.vstack([f * (pose.R[:, 1] / z - a * pose.R[:, 2] / (z * z)),
                   -f * (pose.R[:, 0] / z - b * pose.R[:, 2] / (z * z))])
    cov = np.zeros((3, 3)) if record.cov is None else np.asarray(record.cov, dtype=float)
    sigma = math.sqrt(max(float(np.linalg.eigvalsh(J @ cov @ J.T).max()), 0.0))
    return np.asarray(uv, dtype=float), sigma, float(depth)


def _pointing_prediction(ctx, pid, reference=None):
    """(uv, sigma_px) of the pointing prediction: the independent BR-1 reference when given (the scene's record of
    the part is then its object's), else the scene's re-acquisition prediction; (None, None) when there is none."""
    if reference is not None:
        pred = _predict_uv(ctx, reference)
        return (None, None) if pred is None else (pred[0], pred[1])
    pred = ctx.scene.predicted_uv(pid, ctx.view)
    return (None, None) if pred is None else (pred[0], pred[1])


def _record_top(record) -> float | None:
    if record is None:
        return None
    if record.top_z is not None:
        return float(record.top_z.value)
    return None if record.xyz is None else float(record.xyz[2])


def _carry_uv(ctx, seed, z):
    """The pointed pixel of ``seed`` (taken at the seed's pose) carried into the current wrist view at height ``z``
    (back-projected onto that plane, re-projected; zero calls). None when it does not project."""
    if seed is None or z is None:
        return None
    camera = WristCamera(_f_px(ctx))
    then = HandPose.from_state8(seed.pose) if seed.pose else _pose(ctx)
    origin, direction = camera.ray(then, seed.uv)
    if abs(float(direction[2])) < 1e-9:
        return None
    point = origin + (float(z) - float(origin[2])) / float(direction[2]) * direction
    uv, depth = camera.project(_pose(ctx), point)
    return None if uv is None or not depth > 0.0 else (float(uv[0]), float(uv[1]))


def _band_limit_row(ctx) -> float | None:
    """The lowest image row a framed target may reach: the fingertip band's top row minus FT_PROBE_RADIUS_PX (the
    margin of planner BP-1). None without a band."""
    row = _band_top_row(ctx)
    return None if row is None else float(row) - float(geometry.FT_PROBE_RADIUS_PX)


def _horizontal(axis) -> np.ndarray | None:
    v = np.asarray(axis, dtype=float).reshape(-1)[:2].copy()
    norm = float(np.linalg.norm(v))
    return None if norm < 1e-9 else v / norm


def _shift_image(ctx, du_px, dv_px, kappa, *, name):
    """A zero-call hover shift that moves the wrist image content by about (du, dv) px (image u is R1, image v is
    -R0, §C1.3: the content moves against the hand), at most M2_SEGMENT_M: one guarded AnchoredMove along the
    horizontal shift (stop on contact; a WaypointReach would treat a shift below MIN_MOVE_OFFSET_M as reached), then
    a B4 settle. Returns {"ok", "reason", "shift_m", "needed_m"}."""
    R = np.asarray(ctx.view.R, dtype=float)
    r1, r0 = _horizontal(R[:, 1]), _horizontal(R[:, 0])
    if r1 is None or r0 is None or not kappa > 0.0:
        return {"ok": False, "reason": "no_direction", "shift_m": 0.0, "needed_m": None}
    move = (-float(du_px) * r1 + float(dv_px) * r0) / float(kappa)
    needed = float(np.linalg.norm(move))
    distance = min(needed, planner.M2_SEGMENT_M)
    if distance < MIN_MOVE_OFFSET_M / 2.0:
        return {"ok": True, "reason": "no_shift", "shift_m": 0.0, "needed_m": needed}
    axis = np.r_[move / needed, 0.0]
    quat = np.asarray(ctx.view.quaternion_xyzw, dtype=float)
    result, _ = yield from _run(lambda v: AnchoredMove(v, axis=axis, travel_m=distance, gripper="open",
                                                       stop_on_contact=True, touch_threshold_mm=_g3_mm(ctx),
                                                       orientation_xyzw=quat, name=name))
    yield from _run(lambda v: primitives.b4_settle(v, gripper="open", orientation_xyzw=quat, name=name + "_settle"))
    return {"ok": result.get("status") == "succeeded", "reason": result.get("reason"), "shift_m": distance,
            "needed_m": needed}


def _g3_mm(ctx):
    g3 = _g3(ctx)
    return None if g3 is None else float(g3.value)


def _framing_raise(ctx, record, delta_m, *, name):
    """A zero-call framing raise of the hover by ``delta_m`` (at most planner.P11_CAP_M, BP-1's rise bound) within the
    R-5 stretch cap (``_stretch_cap`` at the raised hover). Returns {"raised_m", "reason", "stretch"}."""
    position = np.asarray(ctx.view.position, dtype=float)
    wanted = float(position[2]) + min(float(delta_m), planner.P11_CAP_M)
    capped = _stretch_cap(ctx, record, {"hover_grip_xyz": [float(position[0]), float(position[1]), wanted],
                                        "clearance_m": 0.0})
    target = capped.get("hover_grip_xyz")
    top = float(target[2]) if target is not None else float(position[2])
    delta = top - float(position[2])
    if delta < MIN_MOVE_OFFSET_M / 2.0:
        return {"raised_m": 0.0, "reason": "stretch_cap", "stretch": capped.get("stretch")}
    result = yield from _lift(ctx, delta, gripper="open", name=name,
                              orientation_xyzw=np.asarray(ctx.view.quaternion_xyzw, dtype=float))
    return {"raised_m": float(ctx.view.position[2]) - float(position[2]), "reason": result.get("reason"),
            "stretch": capped.get("stretch")}


def _frames_grip_part(pid, obj) -> bool:
    """Blocker-2 spec BR-2 / BP-1 scope: a hover before pointing a GRIP part of another part (the step's grip P with
    obj Q != P, the lens's pairs); a part gripped as itself (a fixture turned by its own body) keeps P-2 only."""
    return obj is not None and obj != pid


def _frame_summary(frame) -> dict:
    return {k: frame.get(k) for k in ("ok", "mode", "row", "sigma_uv_px", "excess_px", "margin_px",
                                      "crop_bottom_row", "band_top_row", "reason")}


def _frame_for_pointing(ctx, pid, record):
    """Blocker-2 spec BR-2, planner BP-1 (``planner.pointing_frame``) applied at the ARRIVED hover before the first
    pointing call (zero calls): the prediction of ``record`` (the hover reference: the BR-1 independent estimate when
    the part's record is its object's) must leave the fine crop (or, coarse-first, v + sigma_uv) FT_PROBE_RADIUS_PX
    above the fingertip band; else the hover shifts along -R0 by the excess / kappa (at most M2_SEGMENT_M, a guarded
    move) and, when the capped shift is not enough, rises in HeightField-cell steps (at most P11_CAP_M, within the
    stretch cap) to the first height that frames it. [F, live smokes 002 / 003] the crop bottoms reached 2-24 rows
    into the band and the graspable bar sat under the fingertips. Returns {"applied", "ok", "before", "after",
    "shift", "raise"}."""
    row = _band_top_row(ctx)
    if row is None or record is None or record.xyz is None:
        return {"applied": False, "ok": False, "reason": "no_band" if row is None else "no_point"}

    def frame(rise=None):
        here = np.asarray(ctx.view.position, dtype=float) + np.array([0.0, 0.0, float(rise or 0.0)])
        return planner.pointing_frame(record, here, _pose(ctx), f_px=_f_px(ctx), band_top_row=float(row))
    first = frame()
    info = {"applied": False, "ok": bool(first.get("ok")), "before": _frame_summary(first)}
    if first.get("ok") or first.get("excess_px") is None or not first.get("kappa_px_per_m"):
        return info
    info["applied"] = True
    shifted = yield from _shift_image(ctx, 0.0, -float(first["excess_px"]), float(first["kappa_px_per_m"]),
                                      name="frame_shift")
    info["shift"] = {k: shifted.get(k) for k in ("reason", "shift_m", "needed_m")}
    now = frame()
    capped = shifted.get("needed_m") is not None and float(shifted["needed_m"]) > planner.M2_SEGMENT_M
    if not now.get("ok") and capped:            # the capped shift alone cannot frame it: the first framing rise
        step = float(getattr(ctx.scene.hf, "cell_m", 0.0) or 0.0) or MIN_MOVE_OFFSET_M
        rise = 0.0
        while rise < planner.P11_CAP_M - 1e-9:
            rise = min(rise + step, planner.P11_CAP_M)
            if frame(rise).get("ok"):
                info["raise"] = yield from _framing_raise(ctx, record, rise, name="frame_raise")
                now = frame()
                break
        else:
            info["raise"] = {"raised_m": 0.0, "reason": "no_framing_height"}
    info.update(ok=bool(now.get("ok")), after=_frame_summary(now))
    return info


def _frame_object(ctx, record):
    """Blocker-2 spec BR-4 framing (zero calls): the object's region polygon inflated by P11_RECORD_SIGMA_K sigma_xy, at
    its top height, must lie inside the wrist image and above the fingertip band (FT_PROBE_RADIUS_PX margin): a raise
    within the stretch cap when its extent does not fit, then a shift that brings it inside. Returns the info."""
    xyz = np.asarray(record.xyz, dtype=float)
    reach = max(planner.P11_RECORD_SIGMA_K * _sigma_xy(record), float(getattr(ctx.scene.hf, "cell_m", 0.0) or 0.0))
    if record.region_xy is not None and len(record.region_xy) >= 3:
        poly = poly_inflate(np.asarray(record.region_xy, dtype=float), reach)
    else:
        poly = rect_polygon(xyz[:2], (1.0, 0.0), reach, reach)
    z = _record_top(record)
    points = np.column_stack([np.asarray(poly, dtype=float), np.full(len(poly), z)])
    camera = WristCamera(_f_px(ctx))
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
    bottom = _band_limit_row(ctx)
    bottom = size - 1.0 if bottom is None else bottom

    def project():
        pose = _pose(ctx)
        rows = [camera.project(pose, p) for p in points]
        uv = np.array([r[0] for r in rows if r[0] is not None and r[1] > 0.0], dtype=float).reshape(-1, 2)
        depth = float(np.mean([r[1] for r in rows if r[0] is not None and r[1] > 0.0] or [0.0]))
        return uv, depth

    def fits(uv):
        return bool(len(uv) == len(points) and uv[:, 0].min() >= 0.0 and uv[:, 0].max() <= size - 1.0
                    and uv[:, 1].min() >= 0.0 and uv[:, 1].max() <= bottom)
    uv, depth = project()
    info = {"points": int(len(points)), "bottom_row": bottom, "fits_before": fits(uv)}
    if not len(uv) or not depth > 0.0:
        info["reason"] = "not_projected"
        return info
    extent = np.ptp(uv, axis=0)
    space = np.array([size - 1.0, bottom])
    scale = float(min(space[0] / max(extent[0], 1e-9), space[1] / max(extent[1], 1e-9)))
    if scale < 1.0:
        info["raise"] = yield from _framing_raise(ctx, record, depth * (1.0 / scale - 1.0), name="frame_object_raise")
        uv, depth = project()
    if len(uv) and depth > 0.0 and not fits(uv):
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        du = -lo[0] if lo[0] < 0.0 else (size - 1.0 - hi[0] if hi[0] > size - 1.0 else 0.0)
        dv = bottom - hi[1] if hi[1] > bottom else (-lo[1] if lo[1] < 0.0 else 0.0)
        kappa = camera.kappa_at(depth)
        shifted = yield from _shift_image(ctx, du, dv, kappa, name="frame_object_shift")
        info["shift"] = {k: shifted.get(k) for k in ("reason", "shift_m", "needed_m")}
        uv, depth = project()
    info["fits"] = fits(uv)
    return info


def _pointing_entry(ctx, pid, obj) -> dict:
    """The failed-pointing memory of grip part ``pid`` at object ``obj`` within the current plan step (blocker-2 spec
    BR-4; the measured facts BX-1 / BX-2 report): failed rounds, their answers, the rejected pixels, the largest
    region overlap with the object and the escalations run."""
    scope = ctx.memory.get("step_scope")
    key = "|".join(str(v) for v in (scope, pid, obj, _text_of(ctx, pid)))
    return ctx.memory.setdefault("pointing", {}).setdefault(key, {
        "pid": pid, "obj": obj, "step": scope, "failed_rounds": 0, "answers": [], "rejected": [], "iou_obj_max": None,
        "escalations": 0, "last_text": None})


def _note_failed_round(ctx, pid, obj, *, reason, answer=None, uv=None, iou=None, text=None) -> dict:
    entry = _pointing_entry(ctx, pid, obj)
    entry["failed_rounds"] += 1
    if answer is not None:
        entry["answers"].append([str(answer[0]), answer[1]])
    if uv is not None:
        entry["rejected"].append({"uv": [float(v) for v in uv], "reason": reason, "iou_obj": iou,
                                  "step": int(ctx.step)})
    if iou is not None:
        entry["iou_obj_max"] = iou if entry["iou_obj_max"] is None else max(entry["iou_obj_max"], iou)
    if text:
        entry["last_text"] = text
    return entry


def _should_escalate(entry) -> bool:
    """BR-4 trigger: ESCALATE_FAILED_ROUNDS failed rounds, or the same answer id twice, with no escalation yet."""
    if entry["escalations"] >= ESCALATIONS_PER_STEP:
        return False
    answers = [tuple(a) for a in entry["answers"] if a[1] is not None]
    return entry["failed_rounds"] >= ESCALATE_FAILED_ROUNDS or len(answers) != len(set(answers))


def _pointing_evidence(ctx, pid, obj) -> dict:
    entry = _pointing_entry(ctx, pid, obj)
    return jsonable({k: entry[k] for k in ("failed_rounds", "escalations", "iou_obj_max", "rejected", "answers")})


def _reask(ctx, seed, record, check, *, obj, pid, iou, last_centre, moved):
    """Blocker-2 spec BR-3, a re-ask must change something: when the rejected point's registered region overlaps the
    object's region at IoU >= REASK_OBJECT_IOU, a coarse pointing on the full view with the measured text "the pointed
    point lies on part <obj> itself"; otherwise a fine crop re-centred on the pointed pixel (carried into the current
    view) with the check's measured text, unless that crop would be the one just sent (centre within the answer's
    quantization and no hand motion): then the full view too. The rejected point is ringed (``marker_uv``) when the
    client draws markers. Returns {"centre", "sigma", "text", "marker", "mode"}."""
    carried = _carry_uv(ctx, seed, _record_top(record))
    marker = carried if bool(getattr(ctx, "pointing_marker", False)) else None
    on_object = obj is not None and obj != pid and iou is not None and iou >= REASK_OBJECT_IOU
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
    same_crop = (carried is None or not _in_image(carried, size)
                 or (moved is None and last_centre is not None
                     and float(np.linalg.norm(np.asarray(carried) - np.asarray(last_centre)))
                     <= max(float(seed.quantization_px), 1.0)))
    if on_object or same_crop:
        template = REASK_ON_OBJECT_RINGED_TEXT if marker is not None else REASK_ON_OBJECT_TEXT
        text = template.format(obj=obj) if on_object else _check_text(check)
        return {"centre": None, "sigma": None, "text": text[:PREVIOUS_REJECTED_MAX_CHARS], "marker": marker,
                "mode": "coarse_on_object" if on_object else "coarse_full_view"}
    return {"centre": carried, "sigma": float(seed.quantization_px), "text": _check_text(check), "marker": marker,
            "mode": "fine_recentred"}


def _escalate(ctx, pid, obj, *, purpose, held):
    """Blocker-2 spec BR-4, escalate without a call first (at most ESCALATIONS_PER_STEP per plan step): an empty-hand
    hover from the OBJECT's record (``_hover_target``, its own clearance), the arrival descent, ``_frame_object``
    (its inflated region inside the image and above the band; zero calls), then ONE pointing round of the grip part
    with coarse pointing first on that full view, carrying the last measured rejection, and no re-ask (a failure goes
    to repair). Returns (info, round result or None)."""
    entry = _pointing_entry(ctx, pid, obj)
    entry["escalations"] += 1
    info = {"ran": True, "failed_rounds": entry["failed_rounds"], "reason": None}
    record = ctx.scene.parts.get(obj)
    if record is None or record.xyz is None:
        info["reason"] = "part_not_found"
        return info, None
    hover = _hover_target(ctx, record, skip=(pid, obj))
    info["hover"] = planner.jsonable_result({k: hover.get(k) for k in ("hover_grip_xyz", "clearance_m", "reason",
                                                                       "stretch")})
    target = hover.get("hover_grip_xyz")
    if target is None:
        info["reason"] = _reason_code(hover.get("reason"), "geometry_unmeasured")
        return info, None
    arrival = yield from _arrive(ctx, target, skip_parts=(pid, obj))
    info["arrival"] = {k: arrival.get(k) for k in ("ok", "reason", "descent")}
    if not arrival["ok"]:
        info["reason"] = _reason_code(arrival["reason"], "transit_contact")
        return info, None
    _window_start(ctx, pid)
    info["framing"] = yield from _frame_object(ctx, record)
    round_ = yield from grasp_point_round(ctx, pid, purpose=purpose, kind="grasp_point", held=held,
                                          previous=entry.get("last_text"), obj=obj, coarse_first=True, reasks=0)
    info["reason"] = round_.get("reason")
    return info, round_


def _cell_refinement(plan) -> list:
    """Blocker-2 spec BR-5: the cells to re-measure when the plan's P3 / P5 / P6 / P7 refusal rests only on cells the
    scene marks unconfirmed (planner BP-2 ``needs_cell_refinement`` / ``blocking_cells``, scene BS-2); [] otherwise."""
    if not isinstance(plan, planner.Infeasible) or not planner.needs_cell_refinement(plan):
        return []
    out = []
    for cell in planner.blocking_cells(plan) or ():
        try:
            out.append(dict(cell, ij=[int(cell["ij"][0]), int(cell["ij"][1])]))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def _refine_cells(ctx, pid, cells):
    """Blocker-2 spec BR-5: one ParallaxProbe at the current hover with the blocking cells' pixels (each cell's point,
    else its centre at its stored height, projected into the wrist view, off the fingertip band) as survey targets
    (placed exactly, and as relief points on the most textured patch near them), then ``scene.absorb_survey``.
    Returns {"ok", "reason", "targets", "cells_changed", "probe"}."""
    hf = ctx.scene.hf
    pose = _pose(ctx)
    camera = WristCamera(_f_px(ctx))
    size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
    band = getattr(ctx.cal, "fingertip_band", None)
    band = None if band is None else np.asarray(band, dtype=bool)
    targets = []
    for k, cell in enumerate(cells):
        iy, ix = (int(v) for v in cell["ij"])
        xyz = cell.get("xyz")
        if xyz is None:
            z = cell.get("z")
            if z is None:
                z = float(hf.z[iy, ix]) if hf is not None else math.nan
            xyz = np.r_[hf.cell_xy((iy, ix)), float(z)] if hf is not None else None
        xyz = None if xyz is None else np.asarray(xyz, dtype=float).reshape(-1)[:3]
        if xyz is None or len(xyz) != 3 or not np.isfinite(xyz).all():
            continue
        uv, depth = camera.project(pose, xyz)
        if uv is None or not depth > 0.0 or not _in_image(uv, size):
            continue
        if band is not None and band[int(round(float(uv[1]))), int(round(float(uv[0])))]:
            continue
        targets.append((f"refine:{k}", (float(uv[0]), float(uv[1]))))
    info = {"ok": False, "reason": None, "cells": len(cells), "targets": len(targets)}
    if not targets:
        info["reason"] = "cells_out_of_view"
        return info
    relief = [(f"relief:{key}", uv) for key, uv in targets]
    result, recorder, attempts = yield from _run_probe(ctx, lambda v: _recorder(
        ctx, v, camera=ctx.cal.camera, seeds=targets, relief=relief, exclude_mask=band, external=False,
        start_step=int(ctx.step), probe_tracker=True))
    info["probe"] = attempts[-1]
    if _preempted(result) or result.get("reason") != "probe_complete":
        info["reason"] = _reason_code(result.get("reason"), "probe_contact")
        return info
    info["cells_changed"] = int(ctx.scene.absorb_survey(recorder.result(ctx.cal.camera), ctx.view))
    info["ok"] = True
    _window_start(ctx, pid)
    return info


def _check_jaw(ctx, record, *, kind, purpose, obj):
    """The planned jaw for the V5 grasp-point check (planner.check_point_answer "along the planned jaw"): for a
    record without an accepted long axis, the jaw of the contact plan the grasp would execute on this record
    (``_plan``); None (the planner's default: across the long axis, else the current jaw) for an elongated record,
    another kind, or no candidate. [F, BR smoke, real task 2] a grip part beside its object, checked at the
    current jaw, put one finger column on the object (landing 1.9 cm above the point) and rejected both pointings.
    CJ-1 (grasp-refusal decision J): without a feasible plan, the jaw of planner.representative_rejection (the depth
    rejection that got furthest, i.e. the candidate the grasp would try), labelled "would_plan"."""
    if kind != "grasp_point" or record.is_elongated():
        return None, None
    plan = _plan(ctx, record, purpose=purpose, obj=obj)
    if not isinstance(plan, planner.ContactPlan):
        rep = planner.representative_rejection(plan)          # CJ-1: the jaw the grasp would try first
        if rep is not None:
            return float(rep.yaw_rad), "would_plan"
        return None, getattr(plan, "reason", None)
    jaw = np.asarray(plan.jaw_world, dtype=float)
    return math.atan2(float(jaw[1]), float(jaw[0])), "planned"


def grasp_point_round(ctx, pid, *, purpose, kind="grasp_point", held=None, previous=None, obj=None, reference=None,
                      coarse_first=False, reasks=POINT_CHECK_REASKS):
    """Contract §C10 (X: F-M8) at the current hover: point_part -> ParallaxProbe -> absorb -> register(seed) ->
    planner.check_point_answer(kind) on the RE-MEASURED heights. A refusal ``seed_outside_region`` (S-15) gets ONE
    re-point at this hover (fine at the predicted pixel, coarse first above COARSE_FIRST_SIGMA_PX; lead ruling 3); a
    failed answer check gets ctx.note_rejection and at most ``reasks`` re-asks (<= V4_REPOINTS pointings per part per
    view), each of which changes the request (blocker-2 spec BR-3, ``_reask``: coarse on the full view when the rejected
    point lies on the object, else a fine crop re-centred on the rejected pixel; never the same crop again).
    ``reference`` (blocker-2 spec BR-1): the independent record the prediction comes from when the scene's record of the
    part is its object's; ``coarse_first`` (BR-4's escalation round): coarse pointing on the full view first. Every
    pointing that ends without an accepted record is noted per (plan step, part, object) (``_note_failed_round``).
    Returns {"record", "reason", "rounds", "calls", "seed", "stages", "rejected", "reasks"}."""
    out = {"record": None, "reason": None, "rounds": 0, "calls": 0, "seed": None, "stages": [], "rejected": [],
           "reasks": []}
    centre0, sigma0 = _pointing_prediction(ctx, pid, reference)
    centre, sigma = (None, None) if coarse_first else (centre0, sigma0)
    z_ref = _record_top(reference)
    other = ctx.scene.parts.get(obj) if obj is not None and obj != pid else None
    marker = None
    repointed = False
    asks = 0
    while True:
        pointed = yield from point_part(ctx, pid, camera="wrist", purpose=purpose_kind(kind), center_uv=centre,
                                        sigma_uv_px=sigma, previous_rejected=previous, round_index=out["rounds"],
                                        marker_uv=marker)
        out["calls"] += pointed["calls"]
        seed = pointed["seed"]
        if seed is None:
            out["reason"] = pointed["reason"] or "part_not_found"
            _note_failed_round(ctx, pid, obj, reason=out["reason"], answer=pointed.get("answer"))
            return out
        seed, moved = yield from _pointed_visibility(ctx, pid, seed, z=z_ref)
        if moved is not None:
            out.setdefault("visibility", []).append(moved)
        out["seed"] = seed
        measured = yield from _probe_and_register(ctx, pid, seed, label=f"hover_round_{out['rounds']}")
        out["stages"].append({k: v for k, v in measured.items() if k != "record"})
        failure = measured["failure"]
        if failure is not None:
            if "evidence" in failure:          # the pointed seed did not register (not a probe that ended early)
                _note_failed_round(ctx, pid, obj, reason=str(failure.get("why") or failure.get("reason")),
                                   answer=pointed.get("answer"), uv=seed.uv)
            if failure.get("why") == SEED_REFUSAL and not repointed:
                repointed = True
                out["rounds"] += 1
                pred = _pointing_prediction(ctx, pid, reference)
                centre, sigma = (centre0, sigma0) if pred[0] is None else pred
                if coarse_first:
                    centre, sigma = None, None
                previous = "the pointed pixel lies outside the measured region of the part"
                continue
            out["reason"] = _reason_code(failure.get("reason"), "part_not_found")
            out["failure"] = failure
            return out
        record = measured["record"]
        jaw_yaw, jaw_source = _check_jaw(ctx, record, kind=kind, purpose=purpose, obj=obj)
        if kind == "grasp_point" and jaw_yaw is None and not record.is_elongated():
            # V5-U: no candidate jaw exists (width / axis unmeasured, or no plan and no representative rejection):
            # triage over every jaw line, never the incidental current jaw; the decisive check runs at the planned
            # jaw once the re-measured record plans (grasp)
            check = planner.check_point_any_jaw(record, ctx.scene.hf, _inputs(ctx), uv=seed.uv)
            if check.get("deferred"):
                ctx.memory.setdefault("v5_deferred", {})[pid] = {"step": int(ctx.step), "uv": list(seed.uv)}
        else:
            check = planner.check_point_answer(kind, seed.uv, record, ctx.scene.hf, _inputs(ctx), jaw_yaw=jaw_yaw,
                                               wrist_gray_upright=np.asarray(ctx.view.wrist_gray_upright),
                                               held=held)
            if check.get("ok") and kind == "grasp_point":
                (ctx.memory.get("v5_deferred") or {}).pop(pid, None)
        check = dict(check, jaw_source=jaw_source)
        _check_event(ctx, pid, seed, check, out["rounds"], kind)
        out["check"] = jsonable({k: check.get(k) for k in ("ok", "reason", "values", "jaw_source")})
        out["thresholds"] = check.get("thresholds")
        if check.get("ok"):
            ctx.note_accepted("point_fine")
            out["record"] = record
            return out
        ctx.note_rejection("point_check")
        iou = _region_iou(record, other) if other is not None else None
        reask = _reask(ctx, seed, record, check, obj=obj, pid=pid, iou=iou, last_centre=pointed.get("centre"),
                       moved=moved)
        out["rejected"].append({"uv": [float(v) for v in seed.uv], "iou_obj": iou, "answer": pointed.get("answer"),
                                "reason": check.get("reason")})
        _note_failed_round(ctx, pid, obj, reason=str(check.get("reason") or "pointed_not_grasp_point"),
                           answer=pointed.get("answer"), uv=seed.uv, iou=iou, text=reask["text"])
        asks += 1
        out["rounds"] += 1
        if asks > reasks or out["rounds"] > V4_REPOINTS:
            out["reason"] = _reason_code(check.get("reason"), "pointed_not_grasp_point")
            return out
        out["reasks"].append({k: reask[k] for k in ("mode", "text")} | {
            "centre": None if reask["centre"] is None else list(reask["centre"]),
            "marker": None if reask["marker"] is None else list(reask["marker"])})
        centre, sigma, previous, marker = reask["centre"], reask["sigma"], reask["text"], reask["marker"]


def purpose_kind(kind: str) -> str:
    """PointNeed purpose of an answer-check kind."""
    return {"grasp_point": "grasp_point", "surface_point": "surface_point"}.get(kind, "region_seed")


def _check_text(check) -> str:
    """The measured reason of a failed answer check (previous_rejected, <= 120 chars; no axes, no names)."""
    values = check.get("values") or {}
    reason = str(check.get("reason") or "pointed_not_grasp_point")
    above = values.get("above_m")
    if values.get("jaw") == "any_line" and values.get("cone_lines") is not None:
        # V5-U: no candidate jaw; the measured fact is the jaw cone over which the point clears both finger landings
        text = (f"{reason}: the pointed point stands above both finger landings along {int(values['cone_lines'])} "
                f"adjacent jaw lines, needs {int(values.get('cone_needed') or 0)}")
        return text[:PREVIOUS_REJECTED_MAX_CHARS]
    if isinstance(above, list) and above and all(a is not None for a in above):
        lowest = min(float(a) for a in above)
        margin = values.get("margin_m")
        text = f"{reason}: the pointed point stands {lowest * CM_PER_M:.1f} cm above a finger landing"
        if margin is not None:
            text += f", needs {float(margin) * CM_PER_M:.1f} cm"
        return text[:PREVIOUS_REJECTED_MAX_CHARS]
    return reason[:PREVIOUS_REJECTED_MAX_CHARS]


# ================================================================================================= hovers
def _sigma_xy(part) -> float:
    return float(planner.horizontal_sigma_m(part)) if part is not None and part.xyz is not None else math.inf


def _hover_reference(ctx, pid, obj=None):
    """The record a hover over ``pid`` is planned from (hover_for serves a seeded, provisional or registered part):
    the registered record; else the provisional start record [C-37] (the scene's hover centre), unless its region
    choice was so ambiguous that its horizontal sigma exceeds HOVER_LIFT_SIGMA_RATIO x that of the start-view lift
    of its own wrist plan seed (``_seed_lift``: the cell's quantization and depth only), or it has no 3-D point;
    then that lift. Blocker-2 spec BR-1: when ``obj`` (the step's object, another part) is given and the part's record
    coincides with the object's (``_coincides``: the same point or region IoU >= SHARED_REGION_IOU), the hover comes
    from the part's earliest independent estimate (``_independent_reference``); [F, live smokes 002 / 003] a grip
    part registered with its object's region put every hover over the object's centre, with the graspable part under
    the fingertip band. Returns (record or None, source)."""
    part = ctx.scene.parts.get(pid)
    if part is None:
        return None, None
    other = ctx.scene.parts.get(obj) if obj is not None and obj != pid else None
    shared = _coincides(part, other) if other is not None else None
    if shared is not None:
        ctx.memory.setdefault("shared_region", {})[pid] = dict(shared, obj=obj, step=int(ctx.step))
        independent, source = _independent_reference(ctx, pid, obj)
        if independent is not None:
            return independent, source
        ctx.memory["shared_region"][pid]["independent"] = None
    if _registered(part) and part.xyz is not None:
        return part, "registered"
    lifted = None
    lift = ctx.memory.get("seed_lift", {}).get(pid)
    if lift is not None:
        xyz, cov = lift
        sigma_h = math.sqrt(max(float(np.asarray(cov)[2, 2]), 0.0))
        lifted = dataclasses.replace(part, xyz=np.asarray(xyz, dtype=float), cov=np.asarray(cov, dtype=float),
                                     sigma_h_m=sigma_h, registration_step=part.registration_step
                                     if part.registration_step is not None else int(ctx.step))
    if part.xyz is not None and (lifted is None or _sigma_xy(part) <= HOVER_LIFT_SIGMA_RATIO * _sigma_xy(lifted)):
        return part, "provisional" if part.provisional else part.status
    if lifted is not None:
        return lifted, "seed_lift"
    return None, None


def _hover_target(ctx, record, *, skip=(), pointing=None):
    """planner.hover_for (P11 / P12 / P-E / P-F / P-2; BP-1 framing with ``pointing`` for a hover before pointing a
    grip part, blocker-2 spec) at the current pose, then the R-5 stretch guard (``_stretch_cap``)."""
    kwargs = {} if pointing is None else {"pointing": pointing}
    hover = planner.hover_for(record, ctx.scene.hf, ctx.robot, _pose(ctx), parts=_other_parts(ctx, skip=skip),
                              f_px=_f_px(ctx), band_top_row=_band_top_row(ctx), **kwargs)
    return _stretch_cap(ctx, record, hover)


def _stretch_cap(ctx, record, hover) -> dict:
    """The R-5 stretch guard on a hover (spec §4.9 P "Reach", §5.3; v1 ``wrist_to_shoulder_m``): the wrist point
    (grip - R13 0.204 m along the current approach R2) stays within the mount's ``frontal_stretch_limit_m`` of the
    shoulder at the arrival point (hover + HOVER_LOCATE_M) and over the probe's D2 legs at the hover (their
    horizontal reach added to the wrist's). A hover above that height is lowered to it while its fingertips still
    clear the part's own measured top by the hover's P11 clearance; otherwise it is refused (``reach_limit``).
    Applied to top-down hands as well (a disclosed deviation from spec §5.3, which limits the guard to frontal hands;
    it reuses the mount's frontal limit and needs a lead ruling with a top-down limit of its own, BRR-11): [F, BR
    smoke, real task 2] a top-down first-stage hover over a lower-bound part at a wrist-shoulder distance of 0.786 m
    let the approach tilt from 3.7 to 9.7 deg, its last probe leg stalled and the return reach made no progress,
    while the survey hovers at 0.73 m probed cleanly; [F, BR-fix chain run 4, lowering-only variant] a top-down hover
    lowered to its clearance floor beyond the limit (wrist 0.762 m + the 2 cm probe leg) was never reached: three
    re-grasps failed pose_drift after 75 steps each. The refusal makes no motion; a repeated one is held by the
    BRR-2 guard (``_motionless_guard``). ``frontal`` is logged. Without a mount (no shoulder) the hover is
    unchanged."""
    target = hover.get("hover_grip_xyz")
    mount = getattr(ctx.cal, "mount", None) or {}
    shoulder, limit = mount.get("shoulder_xyz"), mount.get("frontal_stretch_limit_m")
    if target is None or shoulder is None or limit is None:
        return hover
    limit = float(limit)
    centre = np.asarray(shoulder, dtype=float)
    wrist_offset = -float(robot_facts.WRIST_FROM_GRIP_M) * np.asarray(ctx.view.R, dtype=float)[:, 2]
    horizontal = float(np.linalg.norm(np.asarray(target[:2], dtype=float) + wrist_offset[:2] - centre[:2]))

    def top_grip_z(reach_m):
        if reach_m >= limit:
            return None
        return float(centre[2] + math.sqrt(limit * limit - reach_m * reach_m) - wrist_offset[2])
    probe = top_grip_z(horizontal + primitives.D2_PROBE_M)
    arrival = top_grip_z(horizontal)
    planned = float(target[2])
    info = {"limit_m": limit, "planned_z": planned, "wrist_horizontal_m": horizontal,
            "cap_z": None if probe is None or arrival is None else min(probe, arrival - HOVER_LOCATE_M)}
    out = dict(hover, stretch=info)
    if info["cap_z"] is not None and planned <= info["cap_z"]:
        info["lowered_m"] = 0.0
        return out
    tops = [float(record.xyz[2])] + ([float(record.top_z.value)] if record.top_z is not None else [])
    floor = max(tops) + float(hover.get("clearance_m") or 0.0) + float(ctx.robot.t_f_m)
    info["floor_z"] = floor
    frontal = bool(robot_facts.is_frontal(np.asarray(ctx.view.R, dtype=float)[:, 2]))
    info["frontal"] = frontal
    if info["cap_z"] is None or info["cap_z"] < floor:
        out.update(hover_grip_xyz=None, reason="reach_limit")
        return out
    info["lowered_m"] = planned - info["cap_z"]
    out["hover_grip_xyz"] = [float(target[0]), float(target[1]), info["cap_z"]]
    out["fingertip_z"] = info["cap_z"] - float(ctx.robot.t_f_m)
    return out


def _shift_hover(ctx, direction_xy, distance_m, *, name="br1_shift"):
    """A short lateral move of the hover (BR-1; the blocker-2 spec framing shifts): one WaypointReach at the current
    height, then a B4 settle."""
    d = np.asarray(direction_xy, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return {"ok": False, "reason": "no_direction"}
    offset = np.r_[d / norm * float(distance_m), 0.0]
    quat = np.asarray(ctx.view.quaternion_xyzw, dtype=float)
    result, _ = yield from _run(lambda v: WaypointReach(v, offset, gripper="open", orientation_xyzw=quat, name=name))
    yield from _run(lambda v: primitives.b4_settle(v, gripper="open", orientation_xyzw=quat, name=name + "_settle"))
    return {"ok": result.get("status") == "succeeded", "reason": result.get("reason")}


def _restage_seed(ctx, pid, *, purpose, kind):
    """A seed for a later stage at a new hover of the same part: zero-call re-acquisition (V3) first, else the
    pointing path at the prediction. Returns (seed or None, calls)."""
    got = ctx.scene.reacquire_part(pid, ctx.view)
    part = ctx.scene.parts.get(pid)
    if got.get("ok") and part is not None and part.seeds and part.seeds[-1].source == "reacquire":
        return part.seeds[-1], 0
    pred = ctx.scene.predicted_uv(pid, ctx.view)
    pointed = yield from point_part(ctx, pid, camera="wrist", purpose=purpose_kind(kind),
                                    center_uv=None if pred is None else pred[0],
                                    sigma_uv_px=None if pred is None else pred[1], previous_rejected=None)
    return pointed["seed"], pointed["calls"]


def _br1_triggers(ctx, record, pid, *, grip: bool, plan_result=None) -> list[str]:
    """Decision 2 §2.6 BR-1 triggers: the top band has n < TOP_BAND_SUPPORT_N (any part); an axis refusal
    AXIS_INCONSISTENT or a collapsed region (S-13) on a grip part; P-C ``refine_axis`` from the planner.
    ``not_elongated`` is never a trigger; an axis kept from an earlier view is no refusal."""
    if not isinstance(record, PartRecord):
        return []
    reasons = []
    band = record.top_band
    if band is not None and int(band.get("n", 0)) < TOP_BAND_SUPPORT_N:
        reasons.append("band_support")
    check = record.axis_check or {}
    if grip and check.get("source") == AXIS_INCONSISTENT:
        reasons.append("axis_inconsistent")
    if grip and check.get("region_collapsed"):
        reasons.append("region_collapsed")
    if plan_result is not None and planner.needs_axis_refinement(plan_result):
        reasons.append("refine_axis")
    return reasons


def _toward_part_xy(ctx, pid) -> np.ndarray | None:
    """World horizontal direction that moves the part's predicted wrist pixel toward the image centre (the away
    direction of a border cut): image u is R1, image v is -R0 (§C1.3)."""
    pred = ctx.scene.predicted_uv(pid, ctx.view)
    if pred is None:
        return None
    centre = geometry.principal_point()
    du, dv = (np.asarray(pred[0], dtype=float) - centre)
    R = np.asarray(ctx.view.R, dtype=float)
    direction = du * R[:2, 1] - dv * R[:2, 0]
    return direction if float(np.linalg.norm(direction)) > 1e-9 else None


# ================================================================================================= surveys
def _survey(ctx, pid, *, need=None):
    """Empty-hand survey of one part (spec §4.8 registered): hover by planner.hover_for (P11 / P12), the arrival
    descent, ParallaxProbe with the scene's survey targets, absorb, register(empty_hand, partial allowed). When
    ``need(record)`` (the next controller's sigma need, e.g. [C-7]) still fails, the hover is re-planned once with P11
    on the re-measured sigma_h and, if that hover is lower, the probe repeats there (spec §4.8 "lower the hover";
    P11's 5 cm floor bounds it). Returns {"record", "reason", "stages"}."""
    out = {"record": None, "reason": None, "stages": []}
    if ctx.holding is not None:
        out["reason"] = "precondition_unmet"
        return out
    reference, _ = _hover_reference(ctx, pid)
    if reference is None or reference.xyz is None:
        out["reason"] = "part_not_found"
        return out
    lowered = False
    current_z = None
    while True:
        hover = _hover_target(ctx, reference, skip=(pid,))
        target = hover.get("hover_grip_xyz")
        if target is None:
            out["reason"] = _reason_code(hover.get("reason"), "geometry_unmeasured")
            out["stretch"] = hover.get("stretch")
            return out
        if current_z is not None and float(target[2]) >= current_z - MIN_MOVE_OFFSET_M:
            out["reason"] = "geometry_unmeasured"
            out["lower_hover"] = {"available": False, "planned_z": float(target[2]), "current_z": current_z}
            return out
        arrival = yield from _arrive(ctx, target, skip_parts=(pid,))
        stage = {"hover": planner.jsonable_result({k: hover.get(k) for k in (
            "hover_grip_xyz", "clearance_m", "z_ref", "first_stage", "first_stage_reasons", "stretch", "flags")}),
            "arrival": {k: arrival.get(k) for k in ("ok", "reason", "descent", "reanchor")}}
        out["stages"].append(stage)
        if not arrival["ok"]:
            out["reason"] = _reason_code(arrival["reason"], "probe_contact")
            return out
        current_z = float(ctx.view.position[2])
        _window_start(ctx, pid)
        view = ctx.view
        targets = ctx.scene.survey_targets(pid, view)
        seeds = list(targets.get("seeds") or [])
        relief = list(targets.get("relief") or [])
        companions = _companions(ctx, pid, view)
        for other in companions:
            seeds += list(ctx.scene.survey_targets(other, view).get("seeds") or [])
        band = getattr(ctx.cal, "fingertip_band", None)
        dense = tuple(float(v) for v in seeds[0][1]) if seeds else None
        result, recorder, attempts = yield from _run_probe(ctx, lambda v: _recorder(
            ctx, v, camera=ctx.cal.camera, seeds=seeds, relief=relief, exclude_mask=band, external=False,
            start_step=int(ctx.step), probe_tracker=True, dense_uv=dense))
        stage["probe"] = attempts[-1]
        stage["probe_attempts"] = len(attempts)
        if _preempted(result) or result.get("reason") != "probe_complete":
            out["reason"] = _reason_code(result.get("reason"), "probe_contact")
            return out
        survey = recorder.result(ctx.cal.camera)
        ctx.scene.absorb_survey(survey, ctx.view)
        record = ctx.scene.register(pid, ctx.view, empty_hand=True, allow_partial=True)
        _registration_event(ctx, pid, "survey_hover")
        stage["companions"] = {}
        for other in companions:
            measured = ctx.scene.register(other, ctx.view, empty_hand=True, allow_partial=True)
            _registration_event(ctx, other, "survey_hover:companion")
            stage["companions"][other] = (measured.status if isinstance(measured, PartRecord)
                                          else "failed:" + str(measured.reason))
        if not isinstance(record, PartRecord):
            out["reason"] = _reason_code(record.reason, "part_not_found")
            out["failure"] = record.evidence
            return out
        out["record"] = record
        if need is None or need(record):
            out["reason"] = None
            return out
        if lowered:
            out["reason"] = "geometry_unmeasured"
            return out
        lowered = True
        reference = record


def _companions(ctx, pid, view) -> list:
    """The other parts an empty-hand survey of ``pid`` also measures (spec §4.5 item 2, [C-38]'s rule that a survey
    re-registers what it now sees): parts not registered, or registered with only a lower-bound top (P-F), not held,
    whose predicted pixel lies in this wrist view. Their seeds join the survey's and each is registered after it, so
    a lower-bound bound from the start view (which raises every hover over the part) is replaced by a measurement."""
    out = []
    for other, part in ctx.scene.parts.items():
        if other == pid or other == ctx.holding or not isinstance(part, PartRecord):
            continue
        if _registered(part) and not part.top_is_lower_bound():
            continue
        if ctx.scene.survey_targets(other, view).get("seeds"):
            out.append(other)
    return out


def survey_hover(ctx, pid: str, *, purpose: str, need=None):
    """Contract §C10 survey_hover: an empty-hand hover survey of ``pid`` (``_survey``) recorded as one "registered"
    outcome. ``purpose`` is logged ("destination", "lower_bound_object", "region_seed")."""
    start = int(ctx.step)
    out = yield from _survey(ctx, pid, need=need)
    record = out["record"]
    if record is not None and out["reason"] is None:
        return _registration_outcome(ctx, pid, record, start=start, phase=f"survey_hover:{purpose}",
                                     extra={"stages": out["stages"]})
    _registration_event(ctx, pid, f"survey_hover:{purpose}")
    return _outcome(ctx, "registered", "failed", _reason_code(out["reason"], "geometry_unmeasured"), pids=_pids(pid),
                    evidence={"phase": f"survey_hover:{purpose}", "stages": out["stages"],
                              "failure": out.get("failure"), "lower_hover": out.get("lower_hover")},
                    thresholds=ctx.scene.registration_thresholds(pid), start=start)


# ================================================================================================== grasp
def _grip_hover(ctx, pid, *, purpose, obj, held=None):
    """Grasp steps 1-3 at a grip part's hover (contract §C10.2; decision 1 §2.6 two-stage hover; decision 2 §2.6
    BR-1): hover_for on the hover reference -> arrival (reanchor) -> pointing round (probe, register, answer
    check) -> the second hover of a first_stage hover when it moves > TWO_STAGE_XY_M / TWO_STAGE_Z_M -> BR-1 (a
    D2 shift along R1, or toward the part after a border cut, a re-acquired seed, probe and register) when a
    trigger fires. The hover window is one S-12 memory window. Blocker-2 spec: the hover reference is the part's
    independent estimate when its record coincides with the object's (BR-1, ``_hover_reference``); the arrived hover
    is framed for pointing before the first call (BR-2, ``_frame_for_pointing``); a round that fails after
    ESCALATE_FAILED_ROUNDS failed rounds (or a repeated answer id) at this object escalates once per plan step to an
    object-framing survey with coarse pointing (BR-4, ``_escalate``), whose record then takes the second-hover rule.
    Returns {"record", "reason", "calls", "hover", "stages", "two_stage", "br1", "framing", "escalation",
    "pointing"}."""
    out = {"record": None, "reason": None, "calls": 0, "stages": [], "two_stage": None, "br1": None}
    reference, source = _hover_reference(ctx, pid, obj)
    independent = reference if str(source or "").startswith("independent") else None
    out["reference"] = {"source": source, "sigma_xy_m": None if reference is None else _sigma_xy(reference),
                        "xyz": None if reference is None or reference.xyz is None else
                        [float(v) for v in reference.xyz],
                        "shared_region": (ctx.memory.get("shared_region") or {}).get(pid)
                        if obj is not None and obj != pid else None}
    if reference is None or reference.xyz is None:
        out["reason"] = "part_not_found"
        return out
    skip = tuple(p for p in (pid, obj) if p is not None)
    remembered = (ctx.memory.get("plans") or {}).get(pid)
    if remembered is not None and quaternion_angle_rad(ctx.view.quaternion_xyzw, remembered["quaternion"]) \
            > ORIENT_BEFORE_TRANSIT_RAD:
        # decision 1 §2.6: Orient BEFORE the transit when the part's last planned yaw differs from the hover yaw by
        # more than ORIENT_BEFORE_TRANSIT_RAD (a re-grasp); a large Orient at the hover can end at the reach limit
        oriented = yield from _orient(ctx, remembered["quaternion"])
        out["orient_before_transit"] = jsonable(oriented)
    hover = _hover_target(ctx, reference, skip=skip, pointing="auto" if _frames_grip_part(pid, obj) else None)
    out["hover"] = planner.jsonable_result({k: hover.get(k) for k in (
        "hover_grip_xyz", "clearance_m", "z_ref", "sigma_h_m", "first_stage", "first_stage_reasons",
        "visibility_shift_m", "reason", "stretch", "flags", "framing")})
    target = hover.get("hover_grip_xyz")
    if target is None:
        out["reason"] = _reason_code(hover.get("reason"), "geometry_unmeasured")
        return out
    arrival = yield from _arrive(ctx, target, skip_parts=skip)
    out["arrival"] = {k: arrival.get(k) for k in ("ok", "reason", "descent", "reanchor")}
    if not arrival["ok"]:
        out["reason"] = _reason_code(arrival["reason"], "transit_contact")
        return out
    _window_start(ctx, pid)
    if _frames_grip_part(pid, obj):
        out["framing"] = jsonable((yield from _frame_for_pointing(ctx, pid, reference)))
    hover_xyz = np.asarray(ctx.view.position, dtype=float).copy()
    round_ = yield from grasp_point_round(ctx, pid, purpose=purpose, kind="grasp_point", held=held, obj=obj,
                                          reference=independent)
    out["calls"] += round_["calls"]
    out["stages"] += round_["stages"]
    out["check"] = round_.get("check")
    out["rejected"] = round_.get("rejected")
    out["reasks"] = round_.get("reasks")
    record = round_["record"]
    cut = record is None and (round_.get("failure") or {}).get("why") == CUT_REFUSAL
    first_stage = bool(hover.get("first_stage"))
    if record is None and not cut and obj is not None and obj != pid and ctx.holding is None \
            and _should_escalate(_pointing_entry(ctx, pid, obj)):
        escalation, again = yield from _escalate(ctx, pid, obj, purpose=purpose, held=held)
        out["escalation"] = jsonable(escalation)
        if again is not None:
            out["calls"] += again["calls"]
            out["stages"] += again["stages"]
            out["check"] = again.get("check") or out["check"]
            out["rejected"] = list(out["rejected"] or []) + list(again.get("rejected") or [])
            round_ = again if again["record"] is not None else dict(again, reason=again.get("reason")
                                                                    or round_.get("reason"))
            record = again["record"]
            cut = record is None and (again.get("failure") or {}).get("why") == CUT_REFUSAL
            first_stage = True                  # the hover was the object's: the part's own hover comes next
            hover_xyz = np.asarray(ctx.view.position, dtype=float).copy()
    out["pointing"] = _pointing_evidence(ctx, pid, obj)
    if record is None and not cut:
        out["reason"] = round_["reason"] or "part_not_found"
        out["thresholds"] = round_.get("thresholds")
        return out
    if record is not None and first_stage:
        second = _hover_target(ctx, record, skip=skip)
        target2 = second.get("hover_grip_xyz")
        if target2 is not None:
            dxy = float(np.linalg.norm(np.asarray(target2[:2]) - hover_xyz[:2]))
            dz = float(hover_xyz[2] - float(target2[2]))
            taken = dxy > TWO_STAGE_XY_M or abs(dz) > TWO_STAGE_Z_M
            out["two_stage"] = {"dxy_m": dxy, "dz_m": dz, "taken": taken}
            if taken:
                arrival2 = yield from _arrive(ctx, target2, skip_parts=skip)
                out["two_stage"]["arrival"] = arrival2.get("reason")
                if arrival2["ok"]:
                    hover_xyz = np.asarray(ctx.view.position, dtype=float).copy()
                    seed, calls = yield from _restage_seed(ctx, pid, purpose=purpose, kind="grasp_point")
                    out["calls"] += calls
                    if seed is not None:
                        measured = yield from _probe_and_register(ctx, pid, seed, label="second_stage")
                        out["stages"].append({k: v for k, v in measured.items() if k != "record"})
                        if measured["record"] is not None:
                            record = measured["record"]
    triggers = ["cut_by_border"] if cut else []
    if record is not None:
        plan_result = _plan(ctx, record, purpose=purpose, obj=obj)
        triggers += _br1_triggers(ctx, record, pid, grip=True, plan_result=plan_result)
    if triggers:
        direction = _toward_part_xy(ctx, pid) if cut else np.asarray(ctx.view.R, dtype=float)[:2, 1]
        out["br1"] = {"triggers": triggers}
        if direction is not None:
            shifted = yield from _shift_hover(ctx, direction, primitives.D2_PROBE_M)
            out["br1"]["shift"] = shifted.get("reason")
            seed, calls = yield from _restage_seed(ctx, pid, purpose=purpose, kind="grasp_point")
            out["calls"] += calls
            if seed is not None:
                measured = yield from _probe_and_register(ctx, pid, seed, label="br1")
                out["stages"].append({k: v for k, v in measured.items() if k != "record"})
                if measured["record"] is not None:
                    record = measured["record"]
                out["br1"]["registered"] = measured["record"] is not None
    if record is None:
        out["reason"] = "geometry_unmeasured" if cut else (round_["reason"] or "part_not_found")
        return out
    out["record"] = record
    out["hover_xyz"] = [float(v) for v in ctx.view.position]
    return out


def _depth_rules(ctx):
    rule = ctx.overrides.get("depth_rule") if ctx.overrides else None
    return (rule,) if rule in planner.SLICE_DEPTH_RULES else None


def _plan_inputs(ctx, record, *, purpose, obj, sigma_align_m=None, extra_keepouts=()):
    """The planner inputs of a grasp plan (``_plan``) and of its PC-1 re-evaluation (``_realized_envelope``):
    (PlannerInputs, keep-outs, keyword arguments). Contract §C7: the scene's keep-outs other than the part's and its
    object's own, plus ``extra_keepouts`` (the grasp-scoped GC-1 touch keep-outs); the S-6 part fills for P7; the
    S-17 own-band ring cells for the P-J section edge (None when the scene has none); the other records for P-F;
    next_held = the object for a carry or hold through another part."""
    pid = record.pid
    next_held = None
    if purpose in ("carry", "hold") and obj is not None and obj != pid:
        next_held = ctx.scene.parts.get(obj)
    skip = tuple(p for p in (pid, obj) if p is not None)
    keepouts = _keepouts(ctx, skip=skip) + list(extra_keepouts or ())
    try:
        fills = ctx.scene.part_fill_cells(pid)
    except (KeyError, ValueError, TypeError):
        fills = None
    try:
        ring = ctx.scene.part_ring_cells(pid)
    except (KeyError, ValueError, TypeError, AttributeError):
        ring = None
    kwargs = dict(purpose=purpose, next_held=next_held, part_fill_cells=fills, parts=_other_parts(ctx, skip=(pid,)),
                  part_ring_cells=ring)
    return _inputs(ctx, sigma_align_m=sigma_align_m), keepouts, kwargs


def _revalidate(ctx, plan, record, *, purpose, obj, exclude=(), sigma_align_m=None, extra_keepouts=()):
    """CI-2: planner.revalidate of the executing candidate with ``_plan_inputs`` (the measured sigma_align, the
    realized pose, the grasp's touch keep-outs) and the test override's depth rules."""
    inputs, keepouts, kwargs = _plan_inputs(ctx, record, purpose=purpose, obj=obj, sigma_align_m=sigma_align_m,
                                            extra_keepouts=extra_keepouts)
    return planner.revalidate(plan, record, ctx.scene.hf, inputs, keepouts, exclude=tuple(exclude),
                              depth_rules=_depth_rules(ctx), **kwargs)


def _plan(ctx, record, *, purpose, obj, exclude=(), sigma_align_m=None, extra_keepouts=()):
    """planner.plan_contact on the record with ``_plan_inputs`` (the grasp passes its touch keep-outs as
    ``extra_keepouts``); the test-only ``forced_yaw_rad`` override evaluates that one candidate."""
    pid = record.pid
    inputs, keepouts, kwargs = _plan_inputs(ctx, record, purpose=purpose, obj=obj, sigma_align_m=sigma_align_m,
                                            extra_keepouts=extra_keepouts)
    kwargs["depth_rules"] = _depth_rules(ctx)
    forced = ctx.overrides.get("forced_yaw_rad") if ctx.overrides else None
    if forced is not None:
        result = planner.evaluate_candidate(record, ctx.scene.hf, inputs, planner.SLICE_FAMILIES[0], float(forced),
                                            keepouts, **kwargs)
        if isinstance(result, planner.ContactPlan):
            return result
        return planner.Infeasible(pid, result.reason, (result,), tuple(result.thresholds))
    return planner.plan_contact(record, ctx.scene.hf, inputs, keepouts, exclude=tuple(exclude), **kwargs)


def _orient(ctx, quaternion):
    """Grasp step 5: v1 Orient when the planned orientation differs by more than ORIENT_TOLERANCE_RAD, then
    POST_ORIENT_HOLD_STEPS of hold."""
    angle = quaternion_angle_rad(ctx.view.quaternion_xyzw, quaternion)
    if angle <= ORIENT_TOLERANCE_RAD:
        return {"ok": True, "skipped": True, "angle_rad": angle}
    quat = np.asarray(quaternion, dtype=float)
    result, _ = yield from _run(lambda v: Orient(v, target_xyzw=quat, name="orient"))
    yield from _run(lambda v: HoldPrimitive(v, steps=POST_ORIENT_HOLD_STEPS, gripper="open", name="orient_settle"))
    reason = str(result.get("reason") or "")
    ok = result.get("status") == "succeeded"
    if reason == "reach_limit":
        _note_reach_limit(ctx, result)
    code = {"budget": "orient_budget", "rotation_stall": "orient_stall", "pose_drift": "pose_drift"}.get(
        reason, _reason_code(reason, "orient_stall"))
    return {"ok": ok, "skipped": False, "angle_rad": angle, "reason": None if ok else code, "primitive": reason}


def _mark(ctx, plan, record):
    """Align's mark (contract §C10.2 step 6): an NCCTrack at the planned grip point in the current wrist view (radius
    max(FT_RADIUS_PX, ceil(ELONGATED_TRACK_RADIUS_FRACTION w_px)) on an elongated part, [C-40]) with the robot's grip
    axis offset; a textureless point falls back on the part's rigid patch (offset by the patch's offset from the
    grip point in (R1, R0)). Returns (track, offset, info) or (None, None, info)."""
    pose = _pose(ctx)
    camera = WristCamera(_f_px(ctx))
    gray = np.asarray(ctx.view.wrist_gray_upright)
    grip_point = np.asarray(plan.grip_point_xyz, dtype=float)
    uv, depth = camera.project(pose, grip_point)
    info = {"uv": None if uv is None else [float(v) for v in uv], "depth_m": float(depth)}
    radius = int(geometry.FT_RADIUS_PX)
    if record.is_elongated() and uv is not None and depth > 0.0:
        width = record.width_at(plan.yaw_rad)
        if width is not None:
            w_px = float(width.value) * camera.kappa_at(depth)
            radius = max(radius, int(math.ceil(ELONGATED_TRACK_RADIUS_FRACTION * w_px)))
    info["radius_px"] = radius
    base = np.asarray(ctx.robot.grip_axis_offset_m, dtype=float)
    if uv is not None:
        try:
            return NCCTrack(gray, uv, radius=radius), base, info
        except ValueError as exc:
            info["grip_point"] = str(exc)
    patch = record.rigid_patch or {}
    if patch.get("uv") is not None and patch.get("pose") is not None and record.top_z is not None:
        then = HandPose.from_state8(patch["pose"])
        origin, direction = camera.ray(then, patch["uv"])
        if abs(float(direction[2])) > 1e-9:
            xyz = origin + (float(record.top_z.value) - float(origin[2])) / float(direction[2]) * direction
            puv, _ = camera.project(pose, xyz)
            if puv is not None:
                try:
                    track = NCCTrack(gray, puv)
                    d = xyz - grip_point
                    offset = base + np.array([float(d @ pose.R[:, 1]), float(d @ pose.R[:, 0])])
                    info["rigid_patch"] = True
                    return track, offset, info
                except ValueError as exc:
                    info["rigid_patch"] = str(exc)
    return None, None, info


def _lateral_target(view, along_xyz):
    """across_bar: the lateral anchor that holds the along-axis (R0) coordinate of the planned grip point."""
    if along_xyz is None:
        return None
    along = np.asarray(view.R, dtype=float)[:, 0]
    position = np.asarray(view.position, dtype=float)
    return position + float((np.asarray(along_xyz, dtype=float) - position) @ along) * along


def _align(ctx, plan, record):
    """Grasp step 6: v1 Align (allow_pointing=False, axes = the plan's align mode; across_bar holds the oriented
    quaternion and the planned along point; the setpoint row limit = the fingertip band's top row). Succeeds only
    with the last residual within the effective tolerance (V2). Returns {"ok", "reason", "align", "track",
    "jacobian", "offset", "sigma_align_m", "residual_px"}."""
    track, offset, info = _mark(ctx, plan, record)
    out = {"ok": False, "reason": None, "mark": info, "align": None, "track": None, "jacobian": None,
           "offset": None, "sigma_align_m": None, "residual_px": None}
    if track is None:
        out["reason"] = "mark_lost"
        return out
    across = plan.align_mode == "across_bar"
    quat = np.asarray(plan.quaternion_xyzw, dtype=float)
    along = np.asarray(plan.grip_point_xyz, dtype=float) if across else None
    row = _band_top_row(ctx)
    result, align = yield from _run(lambda v: Align(
        v, offset_m=offset, gripper="open", track=track, allow_pointing=False, axes=plan.align_mode,
        hold_xyzw=quat if across else None, along_point_xyz=along,
        setpoint_row_limit=None if row is None else float(row), name="align"))
    evidence = result.get("evidence") or {}
    out.update(align={"reason": result.get("reason"), "residual_px": evidence.get("residual_px"),
                      "residual_mm_est": evidence.get("residual_mm_est"),
                      "tolerance_px": evidence.get("tolerance_px_effective"), "steps": result.get("actions"),
                      "along_error_m": evidence.get("along_error_m")}, offset=[float(v) for v in offset])
    if _preempted(result) or result.get("status") != "succeeded":
        out["reason"] = _reason_code(result.get("reason"), "align_no_convergence")
        return out
    residual = evidence.get("residual_px")
    tolerance = evidence.get("tolerance_px_effective")
    out["residual_px"] = residual
    if result.get("reason") == "aligned" and residual is not None and tolerance is not None \
            and float(residual) > float(tolerance) + 1e-9:
        out["reason"] = "align_no_convergence"
        return out
    mm = evidence.get("residual_mm_est")
    out["sigma_align_m"] = None if mm is None else abs(float(mm)) / 1000.0
    out.update(ok=True, track=align.track if align is not None else track,
               jacobian=align.jacobian if align is not None else None)
    out["threshold"] = th_value("V2_align_tol_px", float(tolerance) if tolerance is not None else
                                float(evidence.get("tolerance_px") or 0.0), "DD",
                                "v1 Align effective tolerance (recorded)", residual_px=float(residual or 0.0))
    return out


def ladder_descent(ctx, align_info, plan, *, quaternion):
    """v1 ``_descend_ladder`` (agent_speedup_v1.py:1144-1198) re-written: a quiet first settle
    (P14_FIRST_SETTLE_MAX_STEPS), then P14_LADDER_SEGMENT_M guarded segments (stop on contact, G3 touch, the
    orientation held, across_bar: the along point held and the stall armed after motion) with a P14_SETTLE_STEPS
    settle and an Align refresh (allow_pointing=False) between segments until the setpoint cutoff; bounded by the
    plan's P9 bound (touch_stop) or g* (height_stop). Returns {"event", "contact_grip_z", "segments", "refreshes"}."""
    across = plan.align_mode == "across_bar"
    along = np.asarray(plan.grip_point_xyz, dtype=float) if across else None
    end_z = float(plan.target_grip_z) if plan.depth_rule == "height_stop" and plan.target_grip_z is not None \
        else float(plan.bound_grip_z)
    quat = np.asarray(quaternion, dtype=float)
    g3 = _g3(ctx)
    touch = None if g3 is None else float(g3.value)
    row = _band_top_row(ctx)
    yield from _run(lambda v: HoldPrimitive(v, steps=P14_FIRST_SETTLE_MAX_STEPS, gripper="open", until_quiet=True,
                                            orientation_xyzw=quat, name="ladder_first_settle"))
    track = align_info.get("track")
    jacobian = align_info.get("jacobian")
    offset = align_info.get("offset")
    cutoff = track is None
    segments, refreshes = [], []
    event, contact_z = None, None
    while float(ctx.view.position[2]) > end_z + MIN_MOVE_OFFSET_M / 2.0:
        remaining = float(ctx.view.position[2]) - end_z
        travel = remaining if cutoff else min(P14_LADDER_SEGMENT_M, remaining)
        result, _ = yield from _run(lambda v, t=travel: AnchoredMove(
            v, axis=WORLD_DOWN, travel_m=t, gripper="open", stop_on_contact=True, touch_threshold_mm=touch,
            orientation_xyzw=quat, lateral_target=_lateral_target(v, along), arm_stall_after_motion=across,
            name="guarded_approach"))
        reason = str(result.get("reason") or "")
        evidence = result.get("evidence") or {}
        segments.append({"reason": reason, "z": float(ctx.view.position[2]), "steps": result.get("actions")})
        if reason in CONTACT_REASONS:
            event = reason
            contact = evidence.get("contact_xyz") or evidence.get("end_xyz")
            contact_z = float(contact[2]) if contact is not None else float(ctx.view.position[2])
            break
        if _preempted(result) or result.get("status") != "succeeded":
            event = reason
            # the true stop height (BI review F1): with None, classify_contact reads a height_stop plan as having
            # reached g* ("expected") and the grasp closed wherever the hand stopped, possibly above the band; the
            # clean height_stop path (_descend) already reports its stop height
            contact_z = float(ctx.view.position[2])
            break
        if cutoff:
            continue
        yield from _run(lambda v: HoldPrimitive(v, steps=P14_SETTLE_STEPS, gripper="open", orientation_xyzw=quat,
                                                name="ladder_settle"))
        refresh, refreshed = yield from _run(lambda v, t=track, j=jacobian: Align(
            v, offset_m=offset, gripper="open", track=t, jacobian=j, reprobe=across, allow_pointing=False,
            setpoint_row_limit=None if row is None else float(row), name="align_refresh", axes=plan.align_mode,
            hold_xyzw=quat if across else None, along_point_xyz=along))
        refresh_ev = refresh.get("evidence") or {}
        refreshes.append({"reason": refresh.get("reason"), "residual_px": refresh_ev.get("residual_px"),
                          "tolerance_px": refresh_ev.get("tolerance_px_effective"), "steps": refresh.get("actions")})
        if refresh.get("status") != "succeeded" or refresh.get("reason") == "setpoint_cutoff":
            cutoff = True
        elif refreshed is not None:
            track, jacobian = refreshed.track, refreshed.jacobian
    if event is None:
        event = "target_reached" if plan.depth_rule == "height_stop" else "bound_reached"
        contact_z = float(ctx.view.position[2])
    return {"event": event, "contact_grip_z": contact_z, "segments": segments, "refreshes": refreshes,
            "end_z": end_z, "last_refresh": _last_servo_refresh(refreshes)}


def _last_servo_refresh(refreshes) -> dict | None:
    """Decision 1 §2.6 ("the last refresh must report an along-jaw residual <= V2"): the last ladder refresh that
    servoed (a setpoint_cutoff refresh ends before servoing: the mark is past the fingertip band's row, so the refresh
    before it is the last one) and measured a residual, with ``within_v2`` = residual <= its effective tolerance.
    None when no refresh measured one."""
    for row in reversed(list(refreshes or ())):
        if row.get("reason") == "setpoint_cutoff":
            continue
        residual, tolerance = row.get("residual_px"), row.get("tolerance_px")
        if residual is None or tolerance is None:
            continue
        return dict(row, within_v2=bool(float(residual) <= float(tolerance) + 1e-9))
    return None


def _descend(ctx, plan, align_info, *, quaternion):
    """Grasp step 7: the P14 ladder for a touch_stop or marginal plan (decision 1 §2.6), one guarded move to g* for
    a clean height_stop."""
    if plan.depth_rule == "touch_stop" or plan.marginal:
        return (yield from ladder_descent(ctx, align_info, plan, quaternion=quaternion))
    across = plan.align_mode == "across_bar"
    along = np.asarray(plan.grip_point_xyz, dtype=float) if across else None
    quat = np.asarray(quaternion, dtype=float)
    g3 = _g3(ctx)
    yield from _run(lambda v: HoldPrimitive(v, steps=P14_FIRST_SETTLE_MAX_STEPS, gripper="open", until_quiet=True,
                                            orientation_xyzw=quat, name="descent_settle"))
    travel = float(ctx.view.position[2]) - float(plan.target_grip_z)
    if travel <= MIN_MOVE_OFFSET_M / 2.0:
        return {"event": "target_reached", "contact_grip_z": float(ctx.view.position[2]), "segments": []}
    result, _ = yield from _run(lambda v: AnchoredMove(
        v, axis=WORLD_DOWN, travel_m=travel, gripper="open", stop_on_contact=True,
        touch_threshold_mm=None if g3 is None else float(g3.value), orientation_xyzw=quat,
        lateral_target=_lateral_target(v, along), arm_stall_after_motion=across, name="guarded_to_target"))
    reason = str(result.get("reason") or "")
    evidence = result.get("evidence") or {}
    if reason in CONTACT_REASONS:
        contact = evidence.get("contact_xyz") or evidence.get("end_xyz")
        z = float(contact[2]) if contact is not None else float(ctx.view.position[2])
        return {"event": reason, "contact_grip_z": z, "segments": [{"reason": reason, "steps": result.get("actions")}]}
    event = "target_reached" if reason == "travel_reached" else reason
    return {"event": event, "contact_grip_z": float(ctx.view.position[2]),
            "segments": [{"reason": reason, "steps": result.get("actions")}]}


def close_and_classify(ctx, plan, part, *, purpose=None, obj=None, extra_keepouts=(), sigma_align_m=None):
    """v1 ``_grasp_attempt_close`` re-written (grasp-refusal decision J): PC-1 first (``_realized_envelope``: a hand
    farther than e_lat from the point the plan was checked at is re-evaluated there; no close unless that gives a
    ContactPlan), then the close as primitives.GuardedClose (GC-1; its actions are GripperSettlePrimitive("close",
    GRASP_SETTLE)'s; active, i.e. able to abandon the close and open, on a marginal plan, monitor-only on a clean one)
    -> classify_close with the G5 margin -> planner.width_consistency (G13, G13+). Returns {"present", "gap_mm",
    "close", "width", "plan" (the plan the close used), "envelope", "guard"}; no close: {"aborted":
    "pose_out_of_envelope", "envelope"}; a GC-1 abort: {"aborted": early_finger_contact | span_inconsistent, "guard",
    "close_grip_z" (the grip height the close ran at)}. Lead rulings 3 and 4 (2026-10-20)."""
    envelope = _realized_envelope(ctx, plan, part, purpose=plan.purpose if purpose is None else purpose, obj=obj,
                                  extra_keepouts=extra_keepouts, sigma_align_m=sigma_align_m)
    env = jsonable({k: v for k, v in envelope.items() if k != "plan"})
    _emit(ctx, "contact", pid=plan.pid, phase="close_envelope", envelope=env)
    if not envelope["ok"]:
        return {"present": False, "aborted": "pose_out_of_envelope", "envelope": env, "guard": None, "width": None,
                "drag_m": 0.0, "gap_mm": float(ctx.view.gap_mm), "close": {"reason": "pose_out_of_envelope"},
                "plan": plan}
    plan = envelope["plan"]
    before = np.asarray(ctx.view.position, dtype=float).copy()
    gate = (plan.checks or {}).get("gate")
    result, guarded = yield from _run(lambda v: primitives.GuardedClose(
        v, gripper=ctx.cal.gripper, sigma_close_mm=float(ctx.cal.sigma_close_mm), gate=gate,
        active=bool(plan.marginal), config=GRASP_SETTLE, name="grasp_close"))
    drag = float(np.linalg.norm(np.asarray(ctx.view.position, dtype=float)[:2] - before[:2]))
    guard = ((result.get("evidence") or {}).get("guard") if isinstance(result, Mapping) else None)
    if guard is None and isinstance(guarded, primitives.GuardedClose):
        guard = guarded.guard_evidence()       # a preempted close's synthetic outcome carries the bus report only
    _emit(ctx, "contact", pid=plan.pid, phase="close_guard",
          guard=jsonable({k: v for k, v in (guard or {}).items() if k != "trace"}))
    aborted = (result.get("reason") if result.get("status") == "failed"
               and result.get("reason") in primitives.GC_REASONS else None)
    if aborted is None and isinstance(guarded, primitives.GuardedClose) and guarded.aborting:
        aborted = guarded.verdict[0]           # preempted while abandoning the close: it stays abandoned (no classify)
    if aborted is not None:
        return {"present": False, "aborted": aborted, "guard": guard, "width": None, "drag_m": drag,
                "gap_mm": float(ctx.view.gap_mm), "close": {"reason": aborted}, "envelope": env,
                "plan": plan, "close_grip_z": float(before[2]), "preempted": _preempted(result)}
    settle = guarded.settle if isinstance(guarded, primitives.GuardedClose) else guarded
    if not isinstance(settle, GripperSettlePrimitive) or not settle.gap_curve_mm:
        return {"present": False, "gap_mm": float(ctx.view.gap_mm), "close": {"reason": result.get("reason")},
                "width": None, "preempted": _preempted(result), "guard": guard, "envelope": env, "plan": plan}
    close = classify_close(settle, ctx.cal.gripper, margin_mm=_g5_mm(ctx))
    width = planner.width_consistency(plan, float(close["final_gap_mm"]))
    return {"present": bool(close["object_present"]), "gap_mm": float(close["final_gap_mm"]), "close": close,
            "width": width, "preempted": _preempted(result), "drag_m": drag, "guard": guard, "envelope": env,
            "plan": plan}


def _jaw_unit_xy(plan) -> np.ndarray:
    jaw = np.asarray(plan.jaw_world[:2], dtype=float)
    return jaw / max(float(np.linalg.norm(jaw)), 1e-9)


def _gate_origin(plan) -> np.ndarray:
    """The grip xy the plan's checks were evaluated at (checks["gate"]["grip_xy"]; the grip point without a gate)."""
    gate = (plan.checks or {}).get("gate") or {}
    return np.asarray(gate.get("grip_xy") or plan.grip_point_xyz[:2], dtype=float)


def _realized_envelope(ctx, plan, record, *, purpose, obj, extra_keepouts=(), sigma_align_m=None):
    """PC-1 (grasp-refusal decision J, lead ruling 4): the plan's P16 columns and P5 sweeps were inflated by e_lat
    (P1) about checks["gate"]["grip_xy"]; a hand farther than e_lat from it along or across the jaw is outside what
    was checked: the same (family, yaw, depth rule) is re-evaluated at the realized hand xy (z unchanged) with
    ``_plan_inputs``, and the close proceeds only on a ContactPlan (its gate replaces the plan's; the plan's grip
    point is kept). Returns {"ok", "du_m", "dv_m", "e_lat_m", "re_evaluated", "plan"} (+ "reason", "rejection")."""
    jaw = _jaw_unit_xy(plan)
    d = np.asarray(ctx.view.position[:2], dtype=float) - _gate_origin(plan)
    du, dv = float(d @ jaw), float(d @ np.array([-jaw[1], jaw[0]]))
    e_lat = float(plan.width["e_lat_mm"]) / 1000.0
    out = {"ok": True, "du_m": du, "dv_m": dv, "e_lat_m": e_lat, "re_evaluated": False, "plan": plan}
    if max(abs(du), abs(dv)) <= e_lat + 1e-9:
        return out
    inputs, keepouts, kwargs = _plan_inputs(ctx, record, purpose=purpose, obj=obj, sigma_align_m=sigma_align_m,
                                            extra_keepouts=extra_keepouts)
    again = planner.evaluate_candidate(record, ctx.scene.hf, inputs, plan.family, plan.yaw_rad, keepouts,
                                       depth_rules=(plan.depth_rule,), rim_angle=plan.rim_angle_rad,
                                       grip_xy=np.asarray(ctx.view.position[:2], dtype=float),
                                       p7_reading=(plan.checks or {}).get("p7_reading"), **kwargs)
    ok = isinstance(again, planner.ContactPlan)
    out.update(re_evaluated=True, ok=ok, reason=None if ok else "pose_out_of_envelope",
               rejection=None if ok else jsonable({"rule": again.rule, "reason": again.reason}),
               plan=dataclasses.replace(again, grip_point_xyz=plan.grip_point_xyz) if ok else plan)
    return out


def _touch_keepout(ctx, plan, guard, record, *, grip_z):
    """GC-1 early_finger_contact (grasp-refusal decision J): a grasp-scoped clear_of keep-out over the cell where a
    closing finger met a surface beyond its face limit: one cell along the jaw ending at the touch (the guard's s,
    outward from the gate's grip point on that finger's side), across the finger's R0 pad extent + e_lat; from the
    record's measured support (else the base plane, else TOUCH_KEEPOUT_DEPTH_M below) up to the pad bottom (R4) at
    ``grip_z``, the height the close ran at (the hand is back at the hover when the keep-out is built)."""
    verdict = (guard or {}).get("verdict") or [None, {}]
    info = verdict[1] or {}
    finger = int(info["finger"])
    side = 1.0 if finger == 1 else -1.0                 # finger 1 (state8[6]) on +jaw
    s_m = float(info["s_mm"]) / 1000.0
    cell = float(ctx.scene.hf.cell_m)
    jaw = _jaw_unit_xy(plan)
    r0 = ctx.robot.finger_ranges_mm(_inputs(ctx).finger_q_mm)[f"finger{finger}"]["R0_mm"]
    half_across = max(abs(float(v)) for v in r0) / 1000.0 + float(plan.width["e_lat_mm"]) / 1000.0
    poly = rect_polygon(_gate_origin(plan) + jaw * side * (s_m - cell / 2.0), jaw, cell / 2.0, half_across)
    z_high = float(grip_z) - float(ctx.robot.pad_span_m[1])
    base = _base_plane(ctx)
    z_low = (float(record.support_z.value) if record.support_z is not None
             else float(base.value) if base is not None else z_high - TOUCH_KEEPOUT_DEPTH_M)
    return KeepOut(pid=f"touch:{plan.pid}:{int(ctx.step)}", kind="clear_of", polygon_xy=poly,
                   z_low=min(z_low, z_high), z_high=z_high, thresholds=())


def _descent_touch_keepouts(ctx, plan, klass, record, *, contact_grip_z) -> list:
    """F6-A (gap round §2.5 / §4.3, ruling L9): a descent stopped above the expected band (``blocked_above_part``)
    measured a surface there: one grasp-scoped clear_of keep-out per finger column of the executed plan (both: a
    wrist-level stop does not say which finger touched), polygon = the plan's FingerColumn.polygon_xy (already
    e_lat-inflated), z_high = the classified surface (contact grip z - FINGERTIP_REACH_MM, ROBOT), z_low as
    ``_touch_keepout`` (the record's support, else the base plane, else TOUCH_KEEPOUT_DEPTH_M below). Keep-outs are
    verified landings (RC-1a), so a later candidate whose column overlaps the area is refused at P6. [] without a
    stop height or without columns."""
    surface = klass.get("surface_z")
    if surface is None and contact_grip_z is not None:
        surface = float(contact_grip_z) - robot_facts.FINGERTIP_REACH_MM / 1000.0
    if surface is None or not math.isfinite(float(surface)):
        return []
    z_high = float(surface)
    base = _base_plane(ctx)
    z_low = (float(record.support_z.value) if record.support_z is not None
             else float(base.value) if base is not None else z_high - TOUCH_KEEPOUT_DEPTH_M)
    out = []
    for column in plan.columns or ():
        poly = getattr(column, "polygon_xy", None)
        if poly is None or len(poly) < 3:
            continue
        out.append(KeepOut(pid=f"touch:{plan.pid}:{int(ctx.step)}:f{int(column.finger)}", kind="clear_of",
                           polygon_xy=np.asarray(poly, dtype=float).copy(), z_low=min(z_low, z_high), z_high=z_high,
                           thresholds=()))
    return out


def _envelope_text(envelope) -> list:
    """Evidence text items of a PC-1 refusal: the hand's offset along and across the jaw against e_lat (m)."""
    env = envelope or {}
    if env.get("e_lat_m") is None:
        return []
    return [["hand offset across jaw", abs(float(env["dv_m"])), "m"],
            ["hand offset along jaw", abs(float(env["du_m"])), "m"], ["checked offset", float(env["e_lat_m"]), "m"]]


def _span_text(guard) -> list:
    """Evidence text items of a GC-1 span_inconsistent abort: the touched span against span_hi + tol (mm)."""
    verdict = (guard or {}).get("verdict") or [None, {}]
    info = verdict[1] or {}
    if info.get("span_mm") is None:
        return []
    return [["finger span", float(info["span_mm"]), "mm"], ["span limit", float(info["span_hi_mm"]), "mm"]]


def _touch_text(attempts) -> list:
    """Text flags of the GC-1 early_finger_contact aborts of a grasp: the finger, its touch and its limit (cm)."""
    out = []
    for attempt in attempts:
        verdict = ((attempt.get("guard") or {}).get("verdict") or [None, {}])
        if verdict[0] == "early_finger_contact":
            info = verdict[1] or {}
            out.append(f"finger {int(info['finger'])} touch {float(info['s_mm']) / MM_PER_CM:.1f} cm beyond "
                       f"{float(info['limit_mm']) / MM_PER_CM:.1f} cm")
    return out


def _return_to_hover(ctx, hover_z, *, quaternion=None, xyz=None):
    """Open, then straight up to the hover height (v1 ``_return_to_hover``). CI-3: with ``xyz`` (the grasp's hover
    position, from which its first Orient / Align started) the hand then moves back to that position at the hover
    height (``_return_to``, the current orientation held; nothing within MIN_MOVE_OFFSET_M), so the next candidate's
    Align starts where hover_for's P-2 visibility shift put the mark, never at an aligned pose whose mark sits at the
    setpoint next to the fingertip band."""
    yield from _ensure_open(ctx)
    travel = float(hover_z) - float(ctx.view.position[2])
    lifted = yield from _lift(ctx, travel, gripper="open", name="return_to_hover", orientation_xyzw=quaternion)
    if xyz is None:
        return lifted
    target = np.asarray(xyz, dtype=float).reshape(3)
    quat = ctx.view.quaternion_xyzw if quaternion is None else quaternion
    reached = yield from _return_to(ctx, target, quat, name="return_to_hover_xy")
    return lifted if reached == "goal_reached" else reached


def _base_plane(ctx) -> Meas | None:
    """The scene's measured base support plane: by the scene's own rule planes[0] (the parallax support plane merged
    into, or inserted as, planes[0]; ruling 7 uses the same plane), read only when it is a measured plane record."""
    planes = getattr(ctx.scene, "planes", None) or []
    base = planes[0].get("z") if planes and isinstance(planes[0], Mapping) else None
    return base if isinstance(base, Meas) and math.isfinite(float(base.value)) else None


def _support_pick(ctx, obj, grip) -> Meas | None:
    """The pick support height of a carried object (HeldFrame b_low = g_close - support): the object's (else its grip
    part's) own measured support; else the scene's base support plane when it lies below the object's measured top
    and below the hand (a support above either is not this object's). None when no measured support qualifies: the
    carry is then refused (geometry_unmeasured) before the close, never planned from an invented height (BRR-10)."""
    for pid in (obj, grip):
        part = ctx.scene.parts.get(pid)
        if part is not None and part.support_z is not None:
            return part.support_z
    base = _base_plane(ctx)
    part = ctx.scene.parts.get(obj)
    top = None
    if part is not None:
        top = float(part.top_z.value) if part.top_z is not None else (None if part.xyz is None else
                                                                      float(part.xyz[2]))
    if base is None or top is None or not (float(base.value) < top and float(base.value) <
                                           float(ctx.view.position[2])):
        return None
    return base


def _visible_mask(ctx, pid):
    try:
        mask = ctx.scene.region_mask(pid, "wrist", ctx.view)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    band = getattr(ctx.cal, "fingertip_band", None)
    if band is not None and np.shape(band) == mask.shape:
        mask = mask & ~np.asarray(band, dtype=bool)
    return mask if mask.any() else None


def _comotion(ctx, held, obj, *, rotation_verdict):
    """CoMotionObserver (O3 + O-1) on the held object: the external template of the HeldFrame and the object's
    visible wrist pixels at its top height (the fingertip band as the robot's own wrist pixels)."""
    part = ctx.scene.parts.get(obj)
    top = None if part is None else (part.top_z.value if part.top_z is not None else
                                     (None if part.xyz is None else float(part.xyz[2])))
    mask = _visible_mask(ctx, obj)
    band = getattr(ctx.cal, "fingertip_band", None)
    patch = None if (mask is None or top is None or band is None) else {"mask": mask, "z_m": float(top)}
    locator = getattr(ctx.cal, "locator", None)
    sigma = getattr(locator, "sigma_loc_px", None)
    return observers.CoMotionObserver(
        wrist_uv=None, wrist_view=ctx.view, ext_track=held.ext_template, external=getattr(ctx.cal, "external", None),
        robot=ctx.robot, sigma_px=None if sigma is None else float(sigma), wrist_patch=patch,
        wrist_camera=WristCamera(_f_px(ctx)), wrist_hand_mask=None if patch is None else band,
        rotation_verdict=bool(rotation_verdict) if patch is not None else False)


def _bus_add(ctx, key, observer, *, preempt, scope):
    ctx.monitor.remove(key)
    ctx.monitor.add(key, observer, view=ctx.view, preempt=preempt, scope=scope)


def _wrist_tilt_fn(ctx, key):
    """HS-H3 (gap round §2.5 / §2.6): the UprightObserver's ``tilt_fn`` during a carry: the in-hand rotation the
    carry CoMotionObserver's wrist model measures (deg, since that observer's start, i.e. the grasp; read from the
    observer registered under ``key`` in ``ctx.memory["carry_comotion"]``, so a BRR-7 re-anchor's fresh observer is
    followed). Used only when the observer has no ``wrist_tilt_deg`` accessor (``_carry_tilt_fn`` passes that bound
    method, which the UprightObserver recognises as relative): its last measured wrist estimate while the wrist model
    measures (patches set, not lost, fewer than O3_LOST_FRAMES misses, not read left behind). 0.0 before the
    observer starts (its reference is the grasp pose itself, so the Upright start reference is zero); None when there
    is no wrist model or it does not measure (the external evidence then decides as before)."""
    def fn(view):
        observer = (ctx.memory.get("carry_comotion") or {}).get(key)
        if observer is None:
            return None
        if not getattr(observer, "started", False):
            return 0.0 if getattr(observer, "wrist_model", False) else None
        rot = getattr(observer, "wrist_rot", None)
        if not getattr(observer, "wrist_model", False) or rot is None or getattr(rot, "reason", None) is not None:
            return None
        if int(getattr(observer, "wrist_misses", 0)) >= observers.O3_LOST_FRAMES or \
                (getattr(observer, "rot_lost", None) or {}).get("wrist") is not None or \
                int(getattr(observer, "left_behind_run", 0)) > 0:
            return None
        series = (getattr(observer, "rot_series", None) or {}).get("wrist") or []
        return float(series[-1]) if series else 0.0
    return fn


def _carry_tilt_fn(ctx, comotion, key):
    """HS-H3: the carry Upright's tilt_fn: the CoMotionObserver's own ``wrist_tilt_deg`` (bound method; the observers
    module reads it as a tilt relative to the grasp), else the attribute fallback ``_wrist_tilt_fn``."""
    accessor = getattr(comotion, "wrist_tilt_deg", None)
    return accessor if callable(accessor) else _wrist_tilt_fn(ctx, key)


def _retarget_tilt_fn(ctx, old, fresh) -> None:
    """HS-H3 after a BRR-7 re-anchor: an Upright on the held bus whose tilt_fn is the replaced observer's
    ``wrist_tilt_deg`` reads the fresh observer's (its estimate restarts at the re-anchor pose)."""
    accessor = getattr(fresh, "wrist_tilt_deg", None)
    if old is None or not callable(accessor):
        return
    for key in ctx.monitor.keys("held"):
        upright = ctx.monitor.observer(key)
        upright = getattr(upright, "inner", upright)
        if getattr(getattr(upright, "tilt_fn", None), "__self__", None) is old:
            upright.tilt_fn = accessor


def _register_held_bus(ctx, *, grip, obj, purpose, keep, held):
    """§C10.1: the held-scope invariants, registered once by grasp: HoldFloorInvariant (G6), UprightObserver for
    each keep.upright, KeepOutObserver over every keep-out other than the held parts' own, and for a carry the
    CoMotionObserver (preempt, rotation verdict off: O4 judges the tilt). HS-H3: during a carry the UprightObserver's
    ``tilt_fn`` is the carry CoMotionObserver's wrist estimate (``_wrist_tilt_fn``; the observer is built first, the
    bus order is unchanged). Returns the keys."""
    keys = []
    _bus_add(ctx, "hold_floor", observers.HoldFloorInvariant(floor=_g6(ctx)), preempt=True, scope="held")
    keys.append("hold_floor")
    comotion_key = f"comotion:{obj}"
    comotion = None
    if purpose == "carry" and held is not None:
        comotion = _comotion(ctx, held, obj, rotation_verdict=False)
        ctx.memory.setdefault("carry_comotion", {})[comotion_key] = comotion
    if held is not None:
        for item in keep or ():
            if getattr(item, "c", None) == "upright":
                key = f"upright:{item.p}"
                tilt_fn = _carry_tilt_fn(ctx, comotion, comotion_key) if comotion is not None else None
                _bus_add(ctx, key, observers.UprightObserver(robot=ctx.robot, grasp_quaternion=ctx.grasp_quaternion,
                                                             held=held, ext_track=held.ext_template,
                                                             tilt_fn=tilt_fn),
                         preempt=True, scope="held")
                keys.append(key)
    keepouts = _keepouts(ctx, skip=(grip, obj))
    if keepouts:
        _bus_add(ctx, "keepout", observers.KeepOutObserver(keepouts=keepouts, robot=ctx.robot, held=held),
                 preempt=True, scope="held")
        keys.append("keepout")
    if comotion is not None:
        _bus_add(ctx, comotion_key, comotion, preempt=True, scope="held")
        keys.append(comotion_key)
    return keys


def _load_test(ctx, *, grip, obj, attempt):
    """Grasp step 10 for carry / hold: HeldFrame prior at the close -> relocalize (hand static) -> CoMotion on the
    bus with preempt=False [C-46] -> VerifyHold lift LIFT_TEST_M (G6 hold margin, the LiftMonitorAdapter feeding
    the lift silhouette) -> the co-motion verdict -> after_lift (M14 Touchdown trigger, M17) -> Touchdown when it
    fires -> vacate_held [C-36]. A co-motion verdict that stays pending or missing, or an O-1 rotation verdict that
    is switched off for want of the wrist patch, passes the test but is flagged (``comotion_unverified`` /
    ``rotation_unmeasured``, evidence and evidence text; BRR-8). No measured pick support -> failed
    geometry_unmeasured before the lift (BRR-10). Returns {"ok", "reason", "held", "evidence", "flags",
    "lift_start_z"}."""
    est = ctx.held_estimator
    support = _support_pick(ctx, obj, grip)
    lift_start_z = float(ctx.view.position[2])
    if support is None:
        return {"ok": False, "reason": "geometry_unmeasured", "held": None, "flags": ["support_unmeasured"],
                "lift_start_z": lift_start_z, "evidence": {"support": None, "attempt": attempt}}
    try:
        held = est.prior_at_close(obj, grip, ctx.view, support_z_pick=support, gap_mm=float(ctx.view.gap_mm))
    except ValueError as exc:                   # no registered footprint of the object: refused before the lift
        return {"ok": False, "reason": "geometry_unmeasured", "held": None, "flags": ["footprint_unmeasured"],
                "lift_start_z": lift_start_z, "evidence": {"footprint": str(exc), "attempt": attempt}}
    held = est.relocalize(held, ctx.view)
    key = f"load_comotion:{obj}"
    flags = []
    rotation_on = None
    try:
        comotion = _comotion(ctx, held, obj, rotation_verdict=True)
        rotation_on = bool(getattr(comotion, "rotation_verdict", False))
        _bus_add(ctx, key, comotion, preempt=False, scope="relation")
    except ValueError as exc:
        key = None
        reg_error = str(exc)
    else:
        reg_error = None
    if not rotation_on:
        flags.append("rotation_unmeasured")
    recorder = est.lift_recorder(held, ctx.view)
    gap_before = float(ctx.view.gap_mm)
    g6 = _g6(ctx)
    margin = float(g6.value) - float(ctx.cal.gripper.empty_floor_mm)
    result, _ = yield from _run(lambda v: VerifyHold(v, axis=WORLD_UP, travel_m=LIFT_TEST_M,
                                                     calibration=ctx.cal.gripper, hold_margin_mm=margin,
                                                     monitor=primitives.LiftMonitorAdapter(recorder, v),
                                                     name="load_test"))
    snap = ctx.monitor.snapshot().get(key) if key is not None else None
    if key is not None:
        ctx.monitor.remove(key)
    evidence = {"verify_hold": {"reason": result.get("reason"), "min_gap_mm": (result.get("evidence") or {}).get(
        "min_gap_mm")}, "comotion": snap, "comotion_error": reg_error, "attempt": attempt,
        "lift_m": float(ctx.view.position[2]) - lift_start_z}
    reason = str(result.get("reason") or "")

    def failed(code):
        return {"ok": False, "reason": code, "held": held, "evidence": evidence, "flags": flags,
                "lift_start_z": lift_start_z}
    if _preempted(result):
        return failed(_reason_code(reason, "hold_lost"))
    if reason == "hold_jam":
        return failed("hold_jam")
    if reason != "hold_pass":
        return failed("hold_lost")
    verdict = (snap or {}).get("verdict")
    cause = (snap or {}).get("reason")
    if verdict == "violated" and cause in ("grip_not_on_obj", "grip_unstable"):
        return failed(cause)
    evidence["comotion_decided"] = verdict == "met"
    if verdict != "met":
        flags.append("comotion_unverified")             # pending or missing after the lift (BRR-8): logged, not met
        evidence["comotion_verdict"] = verdict
    held, after = est.after_lift(held, recorder, gap_before_mm=gap_before, gap_after_mm=float(ctx.view.gap_mm))
    evidence["after_lift"] = after
    if (after.get("touchdown") or {}).get("needed"):
        touch, prim = yield from _run(lambda v: primitives.Touchdown(
            v, lift_m=LIFT_TEST_M, hold_floor_mm=float(g6.value), drift_abort_m=planner.M5_FLOOR_M))
        evidence["touchdown"] = {"reason": touch.get("reason")}
        g_touch = getattr(prim, "touch_grip_z", None) if prim is not None else None
        if g_touch is not None:
            sigma_touch = _sigma_touch()
            held = est.after_touchdown(held, float(g_touch), sigma_touch, pivot=False)
            lift = yield from _lift(ctx, LIFT_TEST_M, gripper="close", name="touchdown_lift")
            evidence["touchdown"]["lift"] = lift.get("reason")
    ctx.scene.vacate_held(held, int(ctx.step))
    held = ctx.scene.held if ctx.scene.held is not None else held
    evidence["flags"] = list(flags)
    return {"ok": True, "reason": "hold_pass", "held": held, "evidence": evidence, "flags": flags,
            "lift_start_z": lift_start_z}


def _put_back(ctx, lift_start_z):
    """BRR-3: a failed load test (grip_not_on_obj, grip_unstable, hold_jam, hold_lost) may leave the object hanging
    up to LIFT_TEST_M above its pick support: before the gripper opens it is lowered back by a slow Touchdown (aimed
    at the current grip xy, travel the realized lift + TOUCHDOWN_EXTRA_M, ending at the support cue). The Touchdown's
    hold floor is the calibrated empty floor: only a hand closed on nothing skips the lowering. Nothing is lowered
    when the hand did not rise. Returns {"lowered", "lift_m", "reason", "touch_grip_z"}."""
    lifted = float(ctx.view.position[2]) - float(lift_start_z)
    if lifted <= MIN_MOVE_OFFSET_M / 2.0:
        return {"lowered": False, "lift_m": lifted, "reason": None, "touch_grip_z": None}
    floor = float(ctx.cal.gripper.empty_floor_mm)
    result, prim = yield from _run(lambda v: primitives.Touchdown(v, lift_m=lifted, hold_floor_mm=floor,
                                                                  drift_abort_m=planner.M5_FLOOR_M, name="put_back"))
    return {"lowered": True, "lift_m": lifted, "reason": result.get("reason"),
            "touch_grip_z": getattr(prim, "touch_grip_z", None) if prim is not None else None}


def _sigma_touch() -> float:
    """The support-touch height uncertainty: one stall window of the free realized P15 command (v1 AnchoredMove)."""
    return float(primitives.P15_COMMAND_M * AnchoredMove.DEFAULT_FREE_RATIO * AnchoredMove.STALL_WINDOW)


def _release_grip(ctx, *, settle_first=False):
    """Open and retreat (ReleaseVerify) after a failed close or load test; the held bus scope is cleared first."""
    ctx.monitor.clear("held")
    result, _ = yield from _run(lambda v: primitives.ReleaseVerify(v, g_open_mm=_g_open_mm(ctx),
                                                                   touch_threshold_mm=_g3(ctx),
                                                                   settle_first=settle_first, name="release"))
    if ctx.scene.held is not None:
        ctx.scene.release_held(ctx.view)
    ctx.holding = None
    return result


def _stamp_landing(ctx, plan, contact_grip_z) -> dict:
    """P-B fixture landing (``landing_measured``): stamp the touched surface under both finger columns (D5)."""
    surface, threshold = ctx.scene.touch_surface(float(contact_grip_z), _sigma_touch())
    jaw = np.asarray(plan.jaw_world[:2], dtype=float)
    jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-9)
    cells = 0
    for poly in ctx.robot.finger_columns_xy(np.asarray(ctx.view.position[:2], dtype=float), jaw,
                                            finger_q_mm=float(ctx.view.gap_mm) / 2.0):
        cells += int(ctx.scene.stamp_touch(poly, surface.value, surface.sigma, int(ctx.step)))
    return {"surface_z": surface.value, "sigma": surface.sigma, "cells": cells, "threshold": threshold}


def _twin(plan):
    yaw = plan.yaw_rad + math.pi
    return (plan.family, math.atan2(math.sin(yaw), math.cos(yaw)))


def _shift_point(plan, record):
    """Empty-close recovery (spec §4.8): the grip point moved along the jaw toward the region's centre line by
    EMPTY_CLOSE_SHIFT_FRACTION of the measured width."""
    width = record.width_at(plan.yaw_rad)
    w = float(width.value) if width is not None else float(plan.width.get("w_mm", 0.0)) / 1000.0
    jaw = np.asarray(plan.jaw_world, dtype=float)
    jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-9)
    point = np.asarray(plan.grip_point_xyz, dtype=float)
    centre = None
    if record.region_xy is not None and len(record.region_xy) >= 3:
        centre = np.asarray(record.region_xy, dtype=float).mean(axis=0)
    side = 1.0
    if centre is not None:
        side = 1.0 if float((centre - point[:2]) @ jaw[:2]) >= 0.0 else -1.0
    moved = point + side * EMPTY_CLOSE_SHIFT_FRACTION * w * np.r_[jaw[:2], 0.0]
    return dataclasses.replace(plan, grip_point_xyz=tuple(float(v) for v in moved))


def _plan_summary(plan) -> dict:
    return jsonable({"family": plan.family, "depth_rule": plan.depth_rule, "yaw_deg": math.degrees(plan.yaw_rad),
                     "yaw_turn_rad": plan.yaw_turn_rad, "grip_point_xyz": plan.grip_point_xyz,
                     "target_grip_z": plan.target_grip_z, "bound_grip_z": plan.bound_grip_z,
                     "align_mode": plan.align_mode, "marginal": plan.marginal, "admission": plan.admission,
                     "width": plan.width})


def grasp(ctx, grip_pid: str, *, obj_pid: str, purpose: str, resume: dict | None = None, keep=()):
    """Spec §4.9 Grasp, contract §C10.2 (purpose "turn" | "carry" | "hold"; relation "held"). Precondition: the hand
    is empty or already holds this grip part (the StepRunner inserts the implicit release; otherwise
    precondition_unmet). While the object of a carry is partial or lower-bound its survey hover runs first
    (decision 1 §2.6). Local recovery (<= B2 retries): blocked_above_part -> never close, back to the hover, a
    grasp-scoped touch keep-out over both executed finger columns and a counted retry (F6-A, ruling L9; after B2
    retries failed blocked_above_part, next grip); no_contact_where_expected -> re-measure once and the next rule;
    empty_close -> re-align toward the centre line; G13+ misaligned or O-1 grip_unstable -> open, the next candidate
    (and its twin); grip_not_on_obj -> release and one re-point; at the close (grasp-refusal decision J,
    ``close_and_classify``): pose_out_of_envelope (PC-1, no close action) -> the candidate excluded, a retry;
    early_finger_contact (GC-1, the close abandoned and opened) -> the candidate and its twin excluded and a touch
    keep-out for the rest of this grasp (not a retry); span_inconsistent (GC-1) -> the candidate and its twin
    excluded, a retry. A failed load test lowers the object back onto its pick support before the release (BRR-3);
    a carry or hold without a measured pick support is refused at the hover (BRR-10). A failure that sends the
    caller to the next grip part carries ``evidence["next_grip"]``. ``resume`` is accepted for the contract signature
    (§C10) and unused: a grasp restarts from its hover; the turn keeps the twist's own resume (BRR-1). Records one
    "held" outcome."""
    start = int(ctx.step)
    pids = _pids(obj_pid, [grip_pid])
    evidence = {"purpose": purpose, "attempts": [], "next_grip": False}
    thresholds: list = []
    retries = 0
    contact = [None]        # X-1: the cause of this attempt's contact, flushed unless the attempt ends in a hold

    def flush():
        if contact[0] is not None:
            _disturb(ctx, obj_pid, contact[0])
            contact[0] = None

    def fail(reason, *, next_grip=False, **extra):
        flush()
        evidence.update(jsonable(extra))
        evidence["next_grip"] = bool(next_grip)
        refused = [a["sigma_align_refused"] for a in evidence["attempts"]
                   if isinstance(a.get("sigma_align_refused"), str)]
        if refused:      # a candidate the re-plan after Align refused names its own reason (integration, live smoke)
            evidence["text_flags"] = [*(evidence.get("text_flags") or []),
                                      *[f"aligned candidate refused {r}" for r in dict.fromkeys(refused)]]
        touches = _touch_text(evidence["attempts"])
        if touches:      # GC-1: the closes abandoned on an unexpected surface name the finger, its touch and limit
            evidence["text_flags"] = [*(evidence.get("text_flags") or []), *touches]
        return _outcome(ctx, "held", "failed", _reason_code(reason, "no_feasible_candidate"), pids=pids,
                        evidence=evidence, thresholds=thresholds, start=start, retries=retries)

    if ctx.holding is not None and not _holds(ctx, grip_pid):
        return fail("precondition_unmet", holding=getattr(ctx.holding, "pid", None))
    if purpose in ("carry", "hold") and obj_pid != grip_pid and obj_pid in ctx.scene.parts \
            and ctx.scene.needs_survey_hover(obj_pid):
        surveyed = yield from survey_hover(ctx, obj_pid, purpose="lower_bound_object",
                                           need=lambda record: not ctx.scene.needs_survey_hover(obj_pid))
        evidence["object_survey"] = {"status": surveyed.status, "reason": surveyed.reason}
    yield from _ensure_open(ctx)
    hovered = yield from _grip_hover(ctx, grip_pid, purpose=purpose, obj=obj_pid)
    evidence["hover"] = {k: hovered.get(k) for k in ("hover", "arrival", "two_stage", "br1", "calls", "check",
                                                      "hover_xyz", "reference", "orient_before_transit", "stages",
                                                      "framing", "escalation", "rejected", "reasks")}
    evidence["pointing"] = hovered.get("pointing")
    thresholds += _thresholds(hovered.get("thresholds"))
    record = hovered["record"]
    if record is None:
        return fail(hovered["reason"] or "part_not_found", next_grip=True,
                    **_pointing_failure_text(ctx, grip_pid, obj_pid, hovered))
    if purpose in ("carry", "hold") and _support_pick(ctx, obj_pid, grip_pid) is None:
        # BRR-10: no measured pick support -> no HeldFrame b_low; refused at the hover, before any contact
        return fail("geometry_unmeasured", flags=["support_unmeasured"], support_unmeasured=True,
                    text_flags=["support unmeasured"])
    hover_z = float(ctx.view.position[2])
    hover_xyz = np.asarray(ctx.view.position, dtype=float).copy()   # CI-3: every next candidate re-Aligns from here
    excluded: list = []
    touched: list = []                      # GC-1: grasp-scoped touch keep-outs (early_finger_contact)
    pending = None
    remeasured = repointed = False
    marginal_touched = False                # CI-4: a GC-1 touch refuted a marginal admission on this record
    refined = 0
    while True:
        flush()             # X-1: the previous attempt's contact did not end in a hold
        plan = pending if pending is not None else _plan(ctx, record, purpose=purpose, obj=obj_pid, exclude=excluded,
                                                         extra_keepouts=touched)
        pending = None
        _emit(ctx, "contact_plan", **dict(planner.trace_payload(plan), pid=grip_pid, obj=obj_pid, purpose=purpose))
        cells = _cell_refinement(plan)
        if cells and refined < REFINE_CELLS_PER_GRASP:
            # blocker-2 spec BR-5: every blocking cell is unconfirmed (BS-2): re-measure them once, then re-plan once
            refined += 1
            refinement = yield from _refine_cells(ctx, grip_pid, cells)
            evidence.setdefault("refine_cells", []).append(jsonable(dict(refinement, reason_before=plan.reason)))
            if refinement.get("ok"):
                continue
        if not isinstance(plan, planner.ContactPlan):
            # the decisive rejections' thresholds first (the evidence text names the first few): the reason's own
            # check, not the width checks every candidate passed on the way (integration, live smoke 2026-10-19)
            decisive = [t for r in plan.rejections if r.reason == plan.reason for t in r.thresholds
                        if isinstance(t, Threshold)]
            checks = {str(r.rule).rsplit(":", 1)[-1] + "_" for r in plan.rejections if r.reason == plan.reason}
            decisive.sort(key=lambda t: not any(str(t.name).startswith(c) for c in checks))   # stable
            ordered, seen = [], set()
            for t in [*decisive, *plan.thresholds]:
                key = (getattr(t, "name", id(t)), repr(getattr(t, "value", None)))
                if key not in seen:
                    seen.add(key)
                    ordered.append(t)
            thresholds[:0] = ordered
            unmeasured_flags = (["width/axis unmeasured"] if plan.reason == "geometry_unmeasured" and
                                (ctx.memory.get("v5_deferred") or {}).get(grip_pid) is not None else [])
            return fail(plan.reason, next_grip=True, plan=planner.trace_payload(plan),
                        refine_height=planner.needs_height_refinement(plan),
                        refine_axis=planner.needs_axis_refinement(plan),
                        **({"text_flags": unmeasured_flags} if unmeasured_flags else {}))
        if marginal_touched and plan.marginal:
            return fail("early_finger_contact", next_grip=True, plan=planner.trace_payload(plan),
                        text_flags=["marginal candidates refused after a finger touched beyond its face limit"])
        thresholds += [t for t in plan.thresholds if isinstance(t, Threshold)]
        attempt = {"plan": _plan_summary(plan)}
        evidence["attempts"].append(attempt)
        deferred = (ctx.memory.get("v5_deferred") or {}).get(grip_pid)
        if deferred is not None:
            # V5-U decisive check: the pointed point at the jaw this plan executes, on the record it executes
            v5 = planner.check_point_answer("grasp_point", deferred.get("uv"), record, ctx.scene.hf, _inputs(ctx),
                                            jaw_yaw=plan.yaw_rad)
            attempt["v5_at_plan"] = jsonable({k: v5.get(k) for k in ("ok", "reason", "values")})
            _emit(ctx, "pointing", kind="check", pid=grip_pid, camera="wrist", purpose="grasp_point",
                  answer_id=None, uv_upright=deferred.get("uv"), round=-1,
                  check={"ok": bool(v5.get("ok")), "reason": v5.get("reason"), "values": v5.get("values")},
                  source="relations:v5_at_plan")
            if not v5.get("ok"):
                ctx.memory["v5_deferred"].pop(grip_pid, None)
                return fail("pointed_not_grasp_point", next_grip=True,
                            text_flags=["pointed point fails the grasp-point check at the planned jaw"])
            ctx.memory["v5_deferred"].pop(grip_pid, None)
        ctx.memory.setdefault("plans", {})[grip_pid] = {"quaternion": list(plan.quaternion_xyzw),
                                                       "yaw_rad": plan.yaw_rad}
        oriented = yield from _orient(ctx, plan.quaternion_xyzw)
        attempt["orient"] = jsonable(oriented)
        if not oriented["ok"]:
            excluded.append(plan.candidate_key())
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            if len(evidence["attempts"]) > B2_LOCAL_RETRIES + len(excluded):
                return fail(oriented["reason"] or "orient_stall")
            continue
        aligned = yield from _align(ctx, plan, record)
        attempt["align"] = {k: aligned.get(k) for k in ("ok", "reason", "align", "mark", "sigma_align_m")}
        if not aligned["ok"]:
            yield from _return_to_hover(ctx, hover_z)
            return fail(aligned["reason"] or "align_no_convergence")
        if aligned.get("threshold") is not None:
            thresholds.append(aligned["threshold"])
        if aligned["sigma_align_m"] is not None:
            # CI-2: the candidate the hand is aligned for, re-evaluated alone at the measured sigma (no re-search)
            again = _revalidate(ctx, plan, record, purpose=purpose, obj=obj_pid, exclude=excluded,
                                sigma_align_m=aligned["sigma_align_m"], extra_keepouts=touched)
            if not isinstance(again, planner.ContactPlan):
                attempt["sigma_align_refused"] = again.reason
                excluded.append(plan.candidate_key())
                yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
                continue
            again = dataclasses.replace(again, grip_point_xyz=plan.grip_point_xyz)
            plan = again
        descent = yield from _descend(ctx, plan, aligned, quaternion=plan.quaternion_xyzw)
        klass = planner.classify_contact(plan, descent["contact_grip_z"], descent["event"])
        attempt["descent"] = {k: descent.get(k) for k in ("event", "contact_grip_z", "end_z", "refreshes",
                                                           "last_refresh")}
        attempt["contact"] = planner.jsonable_result({k: klass.get(k) for k in (
            "class", "matched", "contact_grip_z", "event", "landing_measured", "surface_z", "bound_grip_z")})
        thresholds += _thresholds(klass.get("thresholds"))
        _emit(ctx, "contact", **dict(attempt["contact"], pid=grip_pid, obj=obj_pid))
        kind = klass["class"]
        if descent["event"] in CONTACT_REASONS:
            contact[0] = "descent_contact"
        if kind == "blocked_above_part":
            # F6-A (gap round §2.5, ruling L9): the touch above the band is a measurement (a grasp-scoped clear_of
            # keep-out over both executed finger columns at the touched height: every later candidate overlapping
            # it is refused at P6 before any motion), a counted B2 retry and a disturbance of the object (X-1)
            excluded.append(plan.candidate_key())
            stamped = _descent_touch_keepouts(ctx, plan, klass, record, contact_grip_z=descent["contact_grip_z"])
            touched += stamped
            retries += 1
            contact[0] = None
            _disturb(ctx, obj_pid, "descent_touch")
            attempt["touch_keepouts"] = [k.to_dict() for k in stamped]
            attempt["recovery"] = "descent_touch"
            _emit(ctx, "contact", pid=grip_pid, obj=obj_pid, phase="descent_touch",
                  contact_grip_z=descent["contact_grip_z"], keepouts=len(stamped), retries=retries,
                  surface_z=None if not stamped else stamped[0].z_high)
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            if retries > B2_LOCAL_RETRIES:
                return fail("blocked_above_part", next_grip=True,
                            text_items=[["descent touches", retries, "count"]],
                            text_flags=["descent stopped above the part"])
            continue
        if kind == "no_contact_where_expected":
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            if remeasured:
                return fail("no_contact_where_expected", next_grip=True)
            remeasured = True
            flush()         # X-1: the re-measure below belongs to the epoch after this contact
            seed, _ = yield from _restage_seed(ctx, grip_pid, purpose=purpose, kind="grasp_point")
            if seed is not None:
                measured = yield from _probe_and_register(ctx, grip_pid, seed, label="contact_remeasure")
                if measured["record"] is not None:
                    record = measured["record"]
                    marginal_touched = False
            excluded.append(plan.key())
            continue
        if kind != "expected":
            yield from _return_to_hover(ctx, hover_z)
            reason = "no_contact_within_bound" if kind == "no_contact_within_bound" else \
                _reason_code(descent["event"], "pose_drift")
            return fail(reason)
        last_refresh = descent.get("last_refresh")
        if last_refresh is not None and not last_refresh["within_v2"]:
            # decision 1 §2.6: the last ladder refresh must report an along-jaw residual <= V2 before the close
            yield from _return_to_hover(ctx, hover_z)
            return fail("align_no_convergence", last_refresh=last_refresh,
                        text_items=[["last align residual", float(last_refresh["residual_px"]), "px"],
                                    ["align tolerance", float(last_refresh["tolerance_px"]), "px"]])
        if plan.depth_rule == "touch_stop" and descent["event"] in CONTACT_REASONS:
            surface = klass.get("surface_z")
            if klass.get("landing_measured"):
                stamped = _stamp_landing(ctx, plan, descent["contact_grip_z"])
                attempt["landing_stamp"] = jsonable({k: v for k, v in stamped.items() if k != "threshold"})
                thresholds.append(stamped["threshold"])
                surface = stamped["surface_z"]
            top = float(record.top_z.value) if record.top_z is not None else float(record.xyz[2])
            if surface is not None:
                retreat, th = planner.clearance_retreat_m(plan, float(surface), top)
                thresholds.append(th)
                attempt["retreat_m"] = float(retreat)
                yield from _lift(ctx, retreat, gripper="open", name="clearance_retreat",
                                 orientation_xyzw=plan.quaternion_xyzw)
        closed = yield from close_and_classify(ctx, plan, record, purpose=purpose, obj=obj_pid,
                                               extra_keepouts=touched, sigma_align_m=aligned.get("sigma_align_m"))
        if closed.get("aborted") in ("early_finger_contact", "span_inconsistent"):
            contact[0] = "gc1_touch"
        elif not closed.get("aborted"):
            contact[0] = "close"
        plan = closed.get("plan") or plan
        attempt["close"] = jsonable({k: closed.get(k) for k in ("present", "gap_mm", "close")})
        attempt["envelope"] = closed.get("envelope")
        if closed.get("aborted"):
            attempt["guard"] = jsonable(closed.get("guard"))
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            if closed["aborted"] == "pose_out_of_envelope":
                # PC-1: the hand is outside the pose the plan was checked for, and the plan does not hold there
                retries += 1
                excluded.append(plan.candidate_key())
                attempt["recovery"] = "pose_out_of_envelope"
                if retries > B2_LOCAL_RETRIES:
                    return fail("pose_out_of_envelope", text_items=_envelope_text(closed.get("envelope")))
                continue
            # GC-1: the close was abandoned and the gripper opened before any lift
            excluded += [plan.candidate_key(), _twin(plan)]            # the twin shares the jaw line
            if closed["aborted"] == "early_finger_contact":
                # an unexpected surface beside the part: the candidate, its twin and that surface are refused for
                # the rest of this grasp; not a retry (the excluded candidates bound the loop)
                keepout = _touch_keepout(ctx, plan, closed.get("guard"), record, grip_z=closed["close_grip_z"])
                touched.append(keepout)
                attempt["touch_keepout"] = keepout.to_dict()
                attempt["recovery"] = "early_finger_contact"
                if plan.marginal:
                    # CI-4: the touch contradicts the unverified / close-verified section this marginal admission
                    # rested on; the record's other marginal candidates rest on the same unmeasured section
                    marginal_touched = True
                continue
            retries += 1
            attempt["recovery"] = "span_inconsistent"
            if retries > B2_LOCAL_RETRIES:
                return fail("no_feasible_candidate", next_grip=True, flags=["span_inconsistent"],
                            text_flags=["span_inconsistent"], text_items=_span_text(closed.get("guard")))
            continue
        width = closed.get("width") or {}
        thresholds += _thresholds(width.get("thresholds"))
        attempt["width"] = planner.jsonable_result({k: v for k, v in width.items() if k != "thresholds"})
        _emit(ctx, "contact", pid=grip_pid, obj=obj_pid, phase="close", gap_mm=closed.get("gap_mm"),
              present=closed.get("present"), width=attempt["width"])
        if not closed["present"]:
            retries += 1
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            if retries > B2_LOCAL_RETRIES:
                return fail("empty_close")
            pending = _shift_point(plan, record)
            attempt["recovery"] = "empty_close_shift"
            continue
        if width.get("misaligned"):
            retries += 1
            yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
            excluded += [plan.candidate_key(), _twin(plan)]
            attempt["recovery"] = "misaligned"
            if retries > B2_LOCAL_RETRIES:
                return fail("no_feasible_candidate", next_grip=True, flags=["misaligned"])
            continue
        flags = [] if width.get("consistent", True) else ["width_mismatch"]
        if purpose == "turn":
            held = None
            reason = "held_for_turn"
            load = {"evidence": {"load_test": "first_twist_segment"}}
        else:
            load = yield from _load_test(ctx, grip=grip_pid, obj=obj_pid, attempt=len(evidence["attempts"]))
            attempt["load"] = load["evidence"]
            flags += list(load.get("flags") or [])
            if not load["ok"]:
                # BRR-3: lower the object back onto its pick support before opening; then the record is stale
                attempt["put_back"] = yield from _put_back(ctx, load.get("lift_start_z", ctx.view.position[2]))
                yield from _release_grip(ctx)
                contact[0] = None
                _moved_by_contact(ctx, obj_pid, grip_pid, "release")
                yield from _return_to_hover(ctx, hover_z, xyz=hover_xyz)
                if load["reason"] == "grip_not_on_obj":
                    if repointed:
                        return fail("grip_not_on_obj", next_grip=True, flags=flags)
                    repointed = True
                    round_ = yield from grasp_point_round(ctx, grip_pid, purpose=purpose, kind="grasp_point",
                                                          previous="the gripped part did not move with the object",
                                                          obj=obj_pid)
                    if round_["record"] is None:
                        return fail(round_["reason"] or "grip_not_on_obj", next_grip=True)
                    record = round_["record"]
                    excluded = []
                    marginal_touched = False
                    continue
                if load["reason"] == "grip_unstable":
                    retries += 1
                    excluded += [plan.candidate_key(), _twin(plan)]
                    if retries > B2_LOCAL_RETRIES:
                        return fail("grip_unstable", flags=flags)
                    continue
                return fail(load["reason"], flags=flags)
            held = load["held"]
            reason = "hold_pass"
        ctx.grasp_quaternion = tuple(float(v) for v in ctx.view.quaternion_xyzw)
        ctx.hold_gap_ref_mm = float(ctx.view.gap_mm)
        ctx.holding = held
        if purpose == "turn":
            ctx.memory["gripped"] = {"pid": grip_pid, "obj": obj_pid, "plan": _plan_summary(plan),
                                     "gap_mm": float(ctx.view.gap_mm), "close_drag_m": _close_drag(closed)}
        # X-1: the hold-forming descent contact and close disturb the object only now, after the load test's
        # prior_at_close read the current epoch's whole-body record (HS-H1)
        contact[0] = None
        _moved_by_contact(ctx, obj_pid, grip_pid, "close")
        evidence["bus"] = _register_held_bus(ctx, grip=grip_pid, obj=obj_pid, purpose=purpose, keep=keep, held=held)
        if held is not None:
            _emit(ctx, "held_frame", **dict(held.to_dict(), after_lift=load["evidence"].get("after_lift")))
        evidence.update(flags=flags, **{flag: True for flag in flags}, plan=_plan_summary(plan),
                        gap_mm=float(ctx.view.gap_mm),
                        text_items=[["finger gap", float(ctx.view.gap_mm), "mm"]],
                        text_flags=[f for f in flags if f in UNVERIFIED_HOLD_FLAGS])
        return _outcome(ctx, "held", "succeeded", reason, pids=pids, evidence=evidence, thresholds=thresholds,
                        start=start, retries=retries)


def _pointing_failure_text(ctx, grip, obj, hovered) -> dict:
    """Blocker-2 spec (the measured facts BX-1 / BX-2 read): a grasp that failed in its pointing carries
    ``evidence["pointing"]`` (failed rounds, escalations, rejected pixels, the largest region overlap with the
    object) and, for the evidence text, the failed-round count and the flags "pointed point on part <obj> itself"
    (overlap >= REASK_OBJECT_IOU) and "region shared with part <obj>" (BR-1 coincidence)."""
    pointing = hovered.get("pointing") or {}
    out = {"pointing": pointing}
    rounds = int(pointing.get("failed_rounds") or 0)
    if rounds:
        out["text_items"] = [["failed pointing rounds", rounds, "count"]]
    flags = []
    iou = pointing.get("iou_obj_max")
    if obj is not None and obj != grip and iou is not None and float(iou) >= REASK_OBJECT_IOU:
        flags.append(f"pointed point on part {obj} itself")
    if ((hovered.get("reference") or {}).get("shared_region")) is not None:
        flags.append(f"region shared with part {obj}")
    if (hovered.get("escalation") or {}).get("ran"):
        flags.append("object survey ran")
    if flags:
        out["text_flags"] = flags
    return out


def _close_drag(closed) -> float:
    """The horizontal grip displacement during the close (metres; the K11 sigma_c term of spec §4.9 Turn step 4)."""
    try:
        return abs(float((closed or {}).get("drag_m") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def hold(ctx, step):
    """Relation held for a ``hold`` step: grasp(grip[0], obj, purpose "hold"), then grip[1] when the first part sends
    the controller on (contract §C3.6). The grasp's own outcome is returned (it is recorded once, by grasp)."""
    start = int(ctx.step)
    outcome = yield from _grasp_with_fallback(ctx, step, purpose="hold")
    if outcome is None:                         # already holding a grip part of this step
        outcome = _outcome(ctx, "held", "succeeded", "hold_pass", pids=_pids(step.obj, list(step.grip)),
                           evidence={"already_held": True}, start=start)
    return outcome


def _grasp_with_fallback(ctx, step, *, purpose):
    grips = list(step.grip) or [step.obj]
    outcome = None
    for grip in grips:
        if _holds(ctx, grip):
            return None
        outcome = yield from grasp(ctx, grip, obj_pid=step.obj, purpose=purpose, keep=step.keep)
        if outcome.status == "succeeded" or not (outcome.evidence or {}).get("next_grip"):
            return outcome
    return outcome


def _grasp_failure(held) -> tuple[list, dict]:
    """(thresholds, evidence extras) of a step controller that fails because its grasp failed (integration, live
    smoke 2026-10-19): the controller's outcome carries the GRASP's thresholds and text (flag "grasp failed"), not
    its own pre-checks (the turn's K6 budget, place_on's destination check), which passed and would misattribute the
    cause in the model-facing evidence text (§C10.6)."""
    grasp_evidence = held.evidence or {}
    flags = ["grasp failed", *[f for f in grasp_evidence.get("text_flags") or [] if f]]
    extra = {"failed_in": "grasp", "text_flags": flags}
    if grasp_evidence.get("text_items"):
        extra["text_items"] = list(grasp_evidence["text_items"])
    return list(held.thresholds or []), extra


# =================================================================================================== turn
def _hand_mask_fn(ctx):
    scene = ctx.scene

    def fn(view):
        return scene.hand_swept_mask("external", view, (0.0, 0.0))
    return fn


def _external_region(ctx, pid):
    try:
        mask = ctx.scene.region_mask(pid, "external", ctx.view)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    return mask if mask.any() else None


def _sigma_px(ctx, camera) -> float:
    return float((getattr(ctx.cal, "sigma_px", None) or {}).get(camera, 0.0) or 0.0)


def _patch_now(ctx, part):
    """The part's rigid patch (or its grip point) carried into the current wrist view at its top height; None when
    it leaves the image or lies on the fingertip band. Returns the PartRotationObserver ``patch`` and ``axis_uv``."""
    if part is None or part.xyz is None:
        return None, None
    pose = _pose(ctx)
    camera = WristCamera(_f_px(ctx))
    top = float(part.top_z.value) if part.top_z is not None else float(part.xyz[2])
    xyz = None
    patch = part.rigid_patch or {}
    if patch.get("uv") is not None and patch.get("pose") is not None:
        then = HandPose.from_state8(patch["pose"])
        origin, direction = camera.ray(then, patch["uv"])
        if abs(float(direction[2])) > 1e-9:
            xyz = origin + (top - float(origin[2])) / float(direction[2]) * direction
    out = None
    if xyz is not None:
        uv, depth = camera.project(pose, xyz)
        size = float(np.asarray(ctx.view.wrist_gray_upright).shape[1])
        band = getattr(ctx.cal, "fingertip_band", None)
        if uv is not None and depth > 0.0 and _in_image(uv, size):
            iu, iv = int(round(float(uv[0]))), int(round(float(uv[1])))
            on_band = band is not None and bool(np.asarray(band, dtype=bool)[iv, iu])
            if not on_band:
                out = {"uv": (float(uv[0]), float(uv[1]))}
    axis_uv = None
    if part.elongation is not None:
        axis = np.asarray(part.elongation[0], dtype=float)
        world = np.array([axis[0], axis[1], 0.0])
        R = np.asarray(pose.R, dtype=float)
        image = np.array([float(world @ R[:, 1]), -float(world @ R[:, 0])])
        if float(np.linalg.norm(image)) > 1e-9:
            axis_uv = image / float(np.linalg.norm(image))
    return out, axis_uv


def _wedge_event(ctx, segment, result, mode) -> None:
    evidence = result.get("evidence") or {}
    wedge = (evidence.get("wedge") or {}).get("evidence") or {}
    _emit(ctx, "wedge", mode=mode, sense=evidence.get("sense"), k=evidence.get("steps"),
          rates=evidence.get("rates_mm_per_step"), early=wedge.get("early"), late=wedge.get("late"),
          lw=wedge.get("lw"), omega=wedge.get("omega"), cls=wedge.get("class"), decided_at=wedge.get("decided_at"),
          k0=wedge.get("k0"), reason=result.get("reason"), decided_by=evidence.get("decided_by"),
          segment=segment, thresholds=(evidence.get("wedge") or {}).get("thresholds"))


def _sense_validity(ctx, effect, *, sense, steps, exclude, phase):
    """O7 [C-26] for one phase: the region minus the obj / grip regions and the hand-swept mask over (0, sense x
    steps x omega_free) (relative to the current hand yaw); valid -> O1 runs on the remainder only, else the effect
    is not watched in this phase."""
    if effect is None:
        return None
    swept = ctx.scene.hand_swept_mask("external", ctx.view, (0.0, float(sense) * float(steps)
                                                             * float(ctx.cal.free_yaw_rate_rad_per_step)))
    validity = observers.EffectWatch.validate_region(effect.region, exclude=exclude, swept=swept,
                                                     false_rate=effect.false_rate)
    effect.set_valid_mask(validity["valid_mask"] if validity["valid"] else None, phase=phase, validity=validity)
    out = {k: v for k, v in validity.items() if k != "valid_mask"}
    if "thresholds" in out:                     # BRR-15: records, not Threshold reprs, in events and evidence
        out["thresholds"] = thresholds_dicts(_thresholds(out["thresholds"]))
    # blocker-1 spec BR-1: the phase's hand mask as the scene built it (its model, margins and the rule applied)
    out["hand_mask"] = jsonable((getattr(ctx.scene, "mask_log", None) or {}).get("hand_swept"))
    return out


def _twist_segment(ctx, *, sense, part, effect, exclude, after_reversal=False, blocked_steps=None, label):
    """One TwistSegment with a fresh sense WedgeObserver and PartRotationObserver; the effect watch's sense-phase
    validity is set first (the most the segment can turn: K1W_EXTEND_STEP, plus the unwedge limit after a
    reversal)."""
    span = observers.K1W_EXTEND_STEP
    if after_reversal:
        span += int(math.ceil(observers.K1W_UNWEDGE_PER_BLOCKED_STEP
                              * (observers.K1W_EXTEND_STEP if blocked_steps is None else blocked_steps)))
    validity = _sense_validity(ctx, effect, sense=sense, steps=span, exclude=exclude, phase="sense")
    patch, axis_uv = _patch_now(ctx, part)
    g6 = _g6(ctx)
    wedge = observers.WedgeObserver(mode="sense", robot=ctx.robot, sigma_g_mm_per_step=ctx.cal.sigma_g_mm_per_step,
                                    hold_floor_mm=g6, after_reversal=after_reversal, blocked_steps=blocked_steps)
    rotation = observers.PartRotationObserver(robot=ctx.robot, patch=patch, axis_uv=axis_uv)
    budget = _budget_left(ctx, sense)
    result, segment = yield from _run(lambda v: primitives.TwistSegment(
        v, sense=sense, wedge=wedge, part_rotation=rotation, hold_floor_mm=float(g6.value),
        yaw_budget_left_rad=budget, robot=ctx.robot, effect=effect, name=f"twist_{label}"))
    _wedge_event(ctx, label, result, "sense")
    return {"result": result, "reason": str(result.get("reason") or ""), "sense": sense, "validity": validity,
            "steps": int(result.get("actions") or 0), "segment": label}


def _turn_release(ctx):
    """Turn step 6: release in place (ReleaseVerify with the closed B4 settle first: the part comes to rest)."""
    ctx.monitor.clear("held")
    result, _ = yield from _run(lambda v: primitives.ReleaseVerify(v, g_open_mm=_g_open_mm(ctx),
                                                                   touch_threshold_mm=_g3(ctx), settle_first=True,
                                                                   name="turn_release"))
    ctx.holding = None
    ctx.memory.pop("gripped", None)
    return result


def _regrasp_for_turn(ctx, step, grip):
    """Turn slip / tug local recovery: release, then grasp the same part again (purpose turn)."""
    yield from _turn_release(ctx)
    _moved_by_contact(ctx, step.obj, grip, "release")
    outcome = yield from grasp(ctx, grip, obj_pid=step.obj, purpose="turn", keep=step.keep)
    return outcome


def _turn_resume(ctx, step) -> dict | None:
    """BRR-1: what an earlier twist of ``step.obj`` measured, or None: the failed outcome's ``resume`` handed back by
    a repair retry (§C3.8, StepContext.resume), else this episode's memory of the part's last twist
    (``ctx.memory["turned"]``: a replan or a reopened step). Only a record with a measured free sense counts."""
    text = _text_of(ctx, step.obj)
    for record in (getattr(step, "resume", None), (ctx.memory.get("turned") or {}).get(step.obj)):
        if isinstance(record, Mapping) and record.get("free_sense") in (1, -1) \
                and record.get("pid", step.obj) == step.obj and record.get("text", text) == text:
            return dict(record)                 # (a replan that gave the id to another part does not inherit it)
    return None


def _remember_turn(ctx, obj, *, free, theta, net) -> dict:
    """BRR-1: the resume record of a twist (also kept per part in ``ctx.memory["turned"]``): the free sense, the
    part angle in HingeTwist's measure (``theta_part_rad``, the next HingeTwist's ``theta_start``), the net hand yaw
    turned in the free sense since the first twist of the part (sense segments included), the hand orientation and
    the yaw budget left in the free sense."""
    record = jsonable({"pid": obj, "text": _text_of(ctx, obj), "free_sense": int(free), "theta_part_rad": float(theta),
                       "theta_net_rad": float(net),
                       "hand_quaternion_xyzw": [float(v) for v in ctx.view.quaternion_xyzw],
                       "budget_left_rad": _budget_left(ctx, int(free)), "step": int(ctx.step)})
    ctx.memory.setdefault("turned", {})[obj] = dict(record)
    return record


def _mark_turned(ctx, obj, grip, *, theta_world, centre_xy=None) -> dict:
    """BRR-9: a twist moved the part. Its articulated keep-out is added (or replaced) at ``theta_world``, the WORLD
    angle turned since its record was measured; the records of the turned part and its grip part are then stale (the
    region, the long axis, the widths and the rigid patch turned with it), so they are dropped (scene.drop: the next
    grasp re-measures the part at its hover), keeping a rotation-invariant hover centre as the part's seed lift: the
    hinge centre HingeTwist measured (``centre_xy``), else the D11 circle centre, else the record's point, at its
    measured top, with the record's own covariance. Returns the evidence."""
    _articulated_keepout(ctx, obj, theta_world=theta_world)
    out = {"keepout": f"articulated:{obj}" in getattr(ctx.scene, "keepouts", {}), "dropped": [],
           "theta_world_rad": float(theta_world)}
    drop = getattr(ctx.scene, "drop", None)
    for pid in dict.fromkeys(p for p in (obj, grip) if p is not None):
        record = ctx.scene.parts.get(pid)
        if record is None or record.xyz is None or not callable(drop):
            continue
        if centre_xy is not None and pid == obj:
            centre = np.asarray(centre_xy, dtype=float).reshape(-1)[:2]
        elif record.circle is not None:
            centre = np.asarray(record.circle[0], dtype=float).reshape(-1)[:2]
        else:
            centre = np.asarray(record.xyz, dtype=float)[:2]
        top = float(record.top_z.value) if record.top_z is not None else float(record.xyz[2])
        if record.cov is not None and np.isfinite(centre).all():
            ctx.memory.setdefault("seed_lift", {})[pid] = (np.r_[centre, top],
                                                           np.asarray(record.cov, dtype=float).copy())
        drop(pid)
        out["dropped"].append(pid)
    return out


def _stored_effect(ctx, step) -> dict | None:
    """Blocker-1 spec BR-3 / BR-4: the effect watch an earlier turn of ``step.obj`` kept (``memory["effect_ref"]``:
    its region taken once before the grasp, its pre-action reference and excludes), when it watched this step's
    effect part and the part's text is unchanged; else None."""
    effect = [u.p for u in step.until if u.o == "effect" and u.p is not None]
    stored = (ctx.memory.get("effect_ref") or {}).get(step.obj)
    if not effect or not isinstance(stored, Mapping) or stored.get("pid") != effect[0] \
            or stored.get("text") != _text_of(ctx, step.obj):
        return None
    return stored


def _remember_effect(ctx, obj, effect_pid, watch) -> None:
    ctx.memory.setdefault("effect_ref", {})[obj] = {"pid": effect_pid, "text": _text_of(ctx, obj), "watch": watch,
                                                    "step": int(ctx.step)}


def _fresh_watch(ctx, stored):
    """A new EffectWatch on the stored region, noise, excludes and resolved reference (``reference_record``): it
    confirms without ``start`` and carries no run of an earlier judgement."""
    old = stored["watch"]
    return observers.EffectWatch(region=old.region, false_rate=old.false_rate, sigma_px=old.sigma_px,
                                 camera="external", static_hand_mask_fn=_hand_mask_fn(ctx),
                                 reference=old.reference_record(), exclude=old.exclude)


def _effect_check_now(ctx, step):
    """BRR-1: the turn's effect, judged now. Blocker-1 spec BR-4: with a stored effect watch (``_stored_effect``) a
    fresh watch on its reference runs ``_effect_confirm`` here without a lift (the static frames of O1); otherwise
    (no earlier watch) the start-baseline stays check of ``stays_check_for`` on the current view. Returns ((key,
    check) or None, ObsReport or None)."""
    stored = _stored_effect(ctx, step)
    if stored is not None:
        watch = _fresh_watch(ctx, stored)
        confirm = yield from _effect_confirm(ctx, watch, lift_max_m=0.0)
        return (f"stays:turned:{step.obj}", watch.stays_check(projector_mask=_hand_mask_fn(ctx))), confirm["report"]
    made = stays_check_for(ctx, step)
    if made is None:
        return None, None
    return made, made[1](ctx.view)


def _effect_confirm(ctx, watch, *, lift_max_m=POST_CONFIRM_LIFT_MAX_M):
    """Blocker-1 spec BR-2, the post-release confirmation: ``watch.confirm`` (visible = region - reference hidden -
    hand now - excludes, O7 shares; O1 changed share against the pre-action reference) on static frames (one-step
    holds, up to CONFIRM_FRAMES_MAX per height). Met on O1's run of O1_FRAMES; violated after O1_FRAMES consecutive
    frames below the share; an unverifiable frame (the region hidden) lifts the open hand along world +z by
    POST_CONFIRM_LIFT_STEP_M (guarded AnchoredMove, stop on contact) up to ``lift_max_m``; still pending there (or
    after a contact) is pending. Returns {"verdict", "reason", "lift_m", "frames", "levels", "visible_share",
    "share", "flags", "report"}."""
    start_z = float(ctx.view.position[2])
    quat = np.asarray(ctx.view.quaternion_xyzw, dtype=float)
    gripper = "close" if (ctx.holding is not None or ctx.memory.get("gripped")) else "open"   # never opens a hold
    out = {"verdict": "pending", "reason": "effect_unverifiable", "frames": 0, "levels": [], "report": None}
    stopped = False
    lifts_left = int(math.ceil(float(lift_max_m) / POST_CONFIRM_LIFT_STEP_M - 1e-9)) if lift_max_m > 0.0 else 0
    while True:
        level = {"lift_m": float(ctx.view.position[2]) - start_z, "reasons": []}
        out["levels"].append(level)
        below = 0
        for _ in range(CONFIRM_FRAMES_MAX):
            yield from _run(lambda v: HoldPrimitive(v, steps=1, gripper=gripper, name="effect_confirm_hold"))
            report = watch.confirm(ctx.view, None, {"step": int(ctx.step)})
            out["report"] = report
            out["frames"] += 1
            level["reasons"].append(report.reason)
            below = below + 1 if report.verdict == "violated" else 0
            if report.verdict == "met":
                out.update(verdict="met", reason=report.reason)
                break
            if below >= observers.O1_FRAMES:
                out.update(verdict="violated", reason=report.reason)
                break
            if report.reason == "effect_unverifiable":
                break                           # the hand is still: the next frame hides as much
        if out["verdict"] != "pending" or stopped:
            break
        lifted = float(ctx.view.position[2]) - start_z
        if lifts_left <= 0 or gripper != "open" or lifted >= float(lift_max_m) - MIN_MOVE_OFFSET_M / 2.0:
            break
        lifts_left -= 1
        moved = yield from _lift(ctx, min(POST_CONFIRM_LIFT_STEP_M, float(lift_max_m) - lifted), gripper="open",
                                 name="effect_confirm_lift", orientation_xyzw=quat)
        level["lift"] = moved.get("reason")
        if moved.get("reason") in CONTACT_REASONS or _preempted(moved) or moved.get("status") != "succeeded":
            out["lift_stopped"] = moved.get("reason")
            stopped = True                      # judged once more where the lift stopped, then no further lift
    ev = (out["report"].evidence if out["report"] is not None else {}) or {}
    out.update(lift_m=float(ctx.view.position[2]) - start_z, visible_share=ev.get("visible_share"),
               share=ev.get("share"), flags=[EFFECT_CONFIRM_FLAGS[out["verdict"]]],
               reference=ev.get("reference"))
    return out


def _segment_net(segment, free) -> float:
    """The part angle a sense segment turned, in the free sense (a blocked segment's backlash counts against it)."""
    theta = ((segment.get("result") or {}).get("evidence") or {}).get("theta_sense_rad")
    try:
        return float(theta) * int(segment["sense"]) * int(free) if theta is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def turn(ctx, step):
    """Spec §4.9 Turn, contract §C10.3; relation "turned". Budget pre-check [C-41]; the effect region's motion
    false-change rate is measured on the approach (MotionNoiseMeter on the bus, Appendix L 8); grasp (purpose turn);
    EffectWatch baseline before the first segment (external camera, [C-27]); the first TwistSegment in sense
    ``overrides["first_sense"]`` (default +1 about the approach, the spec's arbitrary first sense; on task 2 it is
    the blocked sense, [C-25]) -> free | blocked (reverse once, after_reversal with the blocked steps) | undecided
    (the other sense once) | slip (re-grasp, <= B2); the free-sense budget check; O7 free-phase validity;
    HingeTwist (orbit follow, centre refit [C-49], end stop, effect margin, quarter / half turn); the stop matched
    against ``step.until``; release in place (settle first); spring-back (K9) and effect persistence; the stays
    entry ``stays:turned:<pid>`` and the articulated keep-out. Records one "turned" outcome.

    Resume (BRR-1, contract §C3.8 retry-with-resume): every exit after a twist that measured the free sense carries
    ``resume`` (free sense, part angle, net angle, hand orientation, budget left; also kept per part in
    ``ctx.memory["turned"]``). A turn that starts with such a record (``_turn_resume``) first judges the effect
    criterion against the start baseline (``stays_check_for``): already met -> succeeded ``turned_by_effect`` with
    flag ``effect_held_before_twist`` and no motion at all (the stays entry is registered); otherwise it twists in
    the KEPT free sense only (never the first-sense default, never a reversal: the part is never turned back past
    where it started), continuing the part angle; blocked in the kept sense is the stop in that sense (``stop``
    criterion -> turned_to_stop; else effect_not_observed / not_rotatable with flag ``kept_sense_blocked``).
    A preemption other than a slip (a keep-out contact, an upright violation) releases in place and fails with its
    own reason instead of re-grasping (BRR-6). Every exit after a twist adds the articulated keep-out and drops the
    turned part's stale record (BRR-9, ``_mark_turned``)."""
    start = int(ctx.step)
    obj = step.obj
    grips = list(step.grip) or [obj]
    kinds = {u.o for u in step.until}
    effect_pids = [u.p for u in step.until if u.o == "effect" and u.p is not None]
    theta_target = HALF_TURN_RAD if "half_turn" in kinds else (QUARTER_TURN_RAD if "quarter_turn" in kinds
                                                                 else None)
    pids = _pids(obj, grips)
    evidence = {"until": sorted(kinds), "segments": []}
    thresholds = []
    retries = 0
    resume = _turn_resume(ctx, step)
    kept = None if resume is None else int(resume["free_sense"])
    theta0 = 0.0 if resume is None else float(resume.get("theta_part_rad") or 0.0)
    net0 = 0.0 if resume is None else float(resume.get("theta_net_rad") or theta0)
    # free: the measured free sense; theta: the part angle (HingeTwist measure, cumulative over attempts); net: the
    # net free-sense angle of this attempt; since_record: the angle turned since the grip part was last registered
    track = {"free": kept, "theta": theta0, "net": 0.0, "since_record": 0.0, "moved": False, "centre": None,
             "grip": grips[0]}
    if resume is not None:
        evidence["resume"] = resume

    def done(status, reason, **extra):
        evidence.update(jsonable(extra))
        out_resume = resume if status != "succeeded" else None     # nothing moved now: the earlier record holds
        if track["moved"] and track["free"] is not None:
            out_resume = _remember_turn(ctx, obj, free=track["free"], theta=track["theta"],
                                        net=net0 + track["net"])
            evidence["resume_out"] = out_resume
            evidence["turned_marks"] = _mark_turned(ctx, obj, track["grip"],
                                                    theta_world=-track["free"] * track["since_record"],
                                                    centre_xy=track["centre"])
        return _outcome(ctx, "turned", status, reason, pids=pids, evidence=evidence, thresholds=thresholds,
                        start=start, retries=retries, resume=out_resume if status != "succeeded" else None)

    budget_th = th_value("K6_yaw_budget_rad", max(_budget_left(ctx, 1), _budget_left(ctx, -1)), "ROBOT",
                         "R12 yaw budget left about the approach, the larger of both senses")
    thresholds.append(budget_th)
    if theta_target is not None and theta_target - theta0 > budget_th.value:
        return done("failed", "yaw_budget", text_items=[["target angle", theta_target - theta0, "rad"],
                                                        ["budget left", budget_th.value, "rad"]])
    effect_pid = effect_pids[0] if effect_pids else None
    if resume is not None and effect_pid is not None:
        made, report = yield from _effect_check_now(ctx, step)
        evidence["effect_before_twist"] = None if report is None else report.to_dict()
        if report is not None and report.verdict == "met":
            key, check = made
            ctx.stays.add(key, check, step_i=step.i, kind="step")
            thresholds += list(report.thresholds)
            return done("succeeded", "turned_by_effect", theta_part_rad=theta0, criterion="effect",
                        flags=["effect_held_before_twist"], effect_held_before_twist=True,
                        text_items=[["part angle", net0, "rad"]], text_flags=["effect held before twist"])
    noise_key = None
    region0 = None
    # the effect region is taken ONCE, before the grasp: region_mask may re-anchor the projector near a far region
    # [C-51], which after the grasp would move it away from the hand and inflate every hand mask of the twist (S-16
    # margin ~ the projector sigma at the hand); the external camera is static, so the mask stays valid
    if effect_pid is not None:
        region0 = _external_region(ctx, effect_pid)
        if region0 is not None:
            noise_key = f"noise:{effect_pid}"
            _bus_add(ctx, noise_key, observers.MotionNoiseMeter(region=region0, sigma_px=_sigma_px(ctx, "external"),
                                                                camera="external",
                                                                static_hand_mask_fn=_hand_mask_fn(ctx)),
                     preempt=False, scope="relation")
        else:
            evidence["effect_region"] = "unavailable"
    held = yield from _grasp_with_fallback(ctx, step, purpose="turn")
    if held is None:
        held = _outcome(ctx, "held", "succeeded", "held_for_turn", pids=pids, evidence={"already_held": True},
                        start=start)
    evidence["grasp"] = {"status": held.status, "reason": held.reason}
    if held.status != "succeeded":
        if noise_key is not None:
            ctx.monitor.remove(noise_key)
        grasp_thresholds, extra = _grasp_failure(held)
        thresholds[:] = grasp_thresholds
        return done("failed", held.reason, next_grip=False, **extra)
    grip = (held.pids.get("grip") or grips)[0]
    track["grip"] = grip
    part = ctx.scene.parts.get(grip)
    effect = None
    if noise_key is not None:
        meter = ctx.monitor.observer(noise_key)
        noise = meter.result()
        ctx.monitor.remove(noise_key)
        evidence["noise"] = noise
        # blocker-1 spec (BO 1 / BR-3): the post-release judgements compare against the meter's start frame (before
        # the approach, the hand clear) with its own hidden mask; the watch falls back to this baseline when the
        # meter's moving frames already changed by the O1 share
        effect = observers.EffectWatch(region=region0, false_rate=float(noise["false_rate"]),
                                       sigma_px=_sigma_px(ctx, "external"), camera="external",
                                       static_hand_mask_fn=_hand_mask_fn(ctx), reference=meter.reference())
        effect.start(ctx.view)
    # the obj / grip regions (near the hand, so no far re-anchoring) after the grasp registered them at the hover; the
    # effect part's own region is region0 (blocker-1 spec BR-3: never re-taken after the grasp)
    exclude = [m for m in (_external_region(ctx, obj), _external_region(ctx, grip) if grip != obj else None)
               if m is not None]
    if effect is not None:
        effect.set_exclude(exclude)
        evidence["effect_reference"] = jsonable(effect.reference_info)
        _remember_effect(ctx, obj, effect_pid, effect)
    if kept is not None:
        sense = kept                            # BRR-1: the measured free sense, never the first-sense default
    else:
        sense = int(ctx.overrides.get("first_sense", 1)) if ctx.overrides else 1
        sense = 1 if sense >= 0 else -1
    free = None
    first = yield from _twist_segment(ctx, sense=sense, part=part, effect=effect, exclude=exclude, label="first")
    evidence["segments"].append(_segment_summary(first))
    current = first
    reversed_once = False
    attempt_segments = [first]
    while free is None:
        reason = current["reason"]
        if reason == "free_sense":
            free = current["sense"]
            break
        if reason in SLIP_REASONS or _preempted(current["result"]):
            code = _reason_code(reason, "hold_lost")
            if code not in SLIP_REASONS:        # BRR-6: a keep-out / upright preemption is no slip
                yield from _turn_release(ctx)
                return done("failed", code, preemption=_preemption_info(current["result"]))
            retries += 1
            if retries > B2_LOCAL_RETRIES:
                yield from _turn_release(ctx)
                return done("failed", "turn_slip" if code == "turn_slip" else "hold_lost")
            again = yield from _regrasp_for_turn(ctx, step, grip)
            if again.status != "succeeded":
                return done("failed", again.reason)
            part = ctx.scene.parts.get(grip)
            current = yield from _twist_segment(ctx, sense=current["sense"], part=part, effect=effect,
                                                exclude=exclude, label=f"regrasp_{retries}")
            evidence["segments"].append(_segment_summary(current))
            attempt_segments = [current]
            continue
        if reason == "yaw_budget":
            yield from _turn_release(ctx)
            return done("failed", "effect_not_observed" if kept is not None and "effect" in kinds else "yaw_budget",
                        text_items=[["part angle", net0, "rad"]] if kept is not None else [])
        if reason in ("blocked_sense", "sense_undecided") and kept is not None:
            # BRR-1: no reversal on a resumed turn (it would turn the part back past where it started)
            yield from _turn_release(ctx)
            text = [["part angle", net0, "rad"]]
            if reason == "sense_undecided":
                return done("failed", "sense_undecided", flags=["kept_sense_undecided"], text_items=text)
            if "stop" in kinds:
                _turn_stays(ctx, step, obj=obj, effect=None, effect_pid=effect_pid)
                return done("succeeded", "turned_to_stop", theta_part_rad=theta0, stop="blocked_sense",
                            criterion="stop", flags=["kept_sense_blocked"], text_items=text,
                            text_flags=["stop in kept sense"])
            return done("failed", "effect_not_observed" if "effect" in kinds else "not_rotatable",
                        flags=["kept_sense_blocked"], text_items=text, text_flags=["blocked in kept sense"])
        if reason in ("blocked_sense", "sense_undecided"):
            if reversed_once:
                yield from _turn_release(ctx)
                return done("failed", "not_rotatable" if reason == "blocked_sense" else "sense_undecided")
            reversed_once = True
            blocked_steps = current["steps"] if reason == "blocked_sense" else None
            current = yield from _twist_segment(ctx, sense=-current["sense"], part=part, effect=effect,
                                                exclude=exclude, after_reversal=True, blocked_steps=blocked_steps,
                                                label="reversal")
            evidence["segments"].append(_segment_summary(current))
            attempt_segments.append(current)
            continue
        yield from _turn_release(ctx)
        return done("failed", _reason_code(reason, "sense_undecided"))
    evidence["free_sense"] = free
    seg_net = sum(_segment_net(s, free) for s in attempt_segments)
    track.update(free=free, net=seg_net, since_record=seg_net, moved=True)
    left = _budget_left(ctx, free)
    thresholds.append(th_value("K6_yaw_budget_left_rad", left, "ROBOT", "R12 yaw budget left in the free sense"))
    if theta_target is not None and theta_target - theta0 > left:
        yield from _turn_release(ctx)
        return done("failed", "yaw_budget", text_items=[["target angle", theta_target - theta0, "rad"],
                                                        ["budget left", left, "rad"]])
    validity = None
    if effect is not None:
        validity = _sense_validity(ctx, effect, sense=free, steps=left / max(
            float(ctx.cal.free_yaw_rate_rad_per_step), 1e-9), exclude=exclude, phase="free")
        evidence["o7_free"] = validity
        if not validity["valid"]:
            evidence["effect_region_invalid"] = True
            evidence.setdefault("flags", []).append("effect_region_invalid")
    watch = effect if (effect is not None and validity is not None and validity["valid"]) else None
    theta = theta0
    tugged = False
    hinge = None
    while True:
        before = theta
        result, hinge = yield from _hinge(ctx, sense=free, part=part, effect=watch, theta_target=theta_target,
                                          theta_start=theta)
        hinge_ev = result.get("evidence") or {}
        evidence.setdefault("hinge", []).append({k: hinge_ev.get(k) for k in (
            "theta_part_rad", "theta_e_rad", "onset_step", "centre_xy", "centre_source", "orbit_follow", "tug",
            "decided_by", "flags", "steps", "yaw_budget_left_rad")} | {"reason": result.get("reason")})
        _wedge_event(ctx, "hinge", result, "end_stop")
        thresholds += _thresholds(hinge_ev.get("thresholds"))
        theta = float(hinge_ev.get("theta_part_rad") or theta)
        track.update(theta=theta, net=track["net"] + (theta - before),
                     since_record=track["since_record"] + (theta - before))
        if hinge_ev.get("centre_source") == "refit" and hinge_ev.get("centre_xy") is not None:
            track["centre"] = hinge_ev.get("centre_xy")
        stop = str(result.get("reason") or "")
        if stop in SLIP_REASONS or _preempted(result):
            code = _reason_code(stop, "hold_lost")
            if code not in SLIP_REASONS:        # BRR-6: a keep-out / upright preemption is no slip
                yield from _turn_release(ctx)
                return done("failed", code, theta_part_rad=theta, preemption=_preemption_info(result),
                            text_items=[["part angle", net0 + track["net"], "rad"]])
            retries += 1
            if retries > B2_LOCAL_RETRIES:
                yield from _turn_release(ctx)
                return done("failed", "turn_slip", theta_part_rad=theta)
            again = yield from _regrasp_after_twist(ctx, step, grip, track)
            if again.status != "succeeded":
                return done("failed", again.reason, theta_part_rad=theta)
            part = ctx.scene.parts.get(grip)
            continue
        if stop == "axis_tug" and not tugged:
            tugged = True
            again = yield from _regrasp_after_twist(ctx, step, grip, track)
            if again.status != "succeeded":
                return done("failed", again.reason, theta_part_rad=theta)
            part = ctx.scene.parts.get(grip)
            continue
        break
    _effect_event(ctx, watch, hinge_ev, validity)
    criterion = TURN_CRITERIA.get(stop)
    wanted = criterion is not None and criterion in kinds
    text = [["part angle", net0 + track["net"], "rad"]]
    # blocker-1 spec BR-2: every effect criterion is confirmed after the release (with the post-release lift), also
    # when the twist stopped for another reason (no onset in the twist: the region may have been hidden)
    confirm_after = effect is not None and "effect" in kinds and (criterion == "effect" or not wanted)
    if not wanted and not confirm_after:
        yield from _turn_release(ctx)
        return done("failed", _unwanted_stop_reason(stop, kinds), theta_part_rad=theta, stop=stop, text_items=text)
    released = yield from _turn_release(ctx)
    evidence["release"] = {"reason": released.get("reason"),
                           "settle_hand_yaw_rad": (released.get("evidence") or {}).get("settle_hand_yaw_rad")}
    post = yield from _turn_post_check(ctx, obj=obj, effect=effect if confirm_after else None)
    evidence["post_check"] = {k: v for k, v in post.items() if k != "confirm"}
    if post.get("spring_back") == "violated":
        return done("failed", "spring_back", theta_part_rad=theta, text_items=text)
    if confirm_after:
        confirm = post.get("confirm") or {}
        verdict = confirm.get("verdict", post.get("persists"))
        evidence["effect_confirm"] = _confirm_summary(confirm)
        flags = [EFFECT_CONFIRM_FLAGS[verdict]] if verdict in EFFECT_CONFIRM_FLAGS else []
        if confirm.get("report") is not None:
            thresholds += list(confirm["report"].thresholds)
        lift = [["post-release lift", float(confirm.get("lift_m") or 0.0), "m"]] if confirm.get("lift_m") else []
        if criterion == "effect":                # the onset was latched in the twist
            if verdict == "violated":
                return done("failed", "effect_not_observed", theta_part_rad=theta, persists=False, flags=flags,
                            text_items=text + lift, text_flags=flags)
            flags = [] if verdict == "met" else ["effect_confirm_pending"]
        elif verdict == "met":                   # no onset in the twist; seen after the release
            _turn_stays(ctx, step, obj=obj, effect=effect, effect_pid=effect_pid)
            return done("succeeded", "turned_by_effect", theta_part_rad=theta, theta_e_rad=None, stop=stop,
                        criterion="effect", flags=flags, **{flag: True for flag in flags}, text_items=text + lift,
                        text_flags=flags)
        else:
            return done("failed", _unwanted_stop_reason(stop, kinds), theta_part_rad=theta, stop=stop, flags=flags,
                        **{flag: True for flag in flags}, text_items=text + lift, text_flags=flags)
        evidence["flags"] = list(dict.fromkeys([*(evidence.get("flags") or []), *flags]))
    _turn_stays(ctx, step, obj=obj, effect=effect if criterion == "effect" else None, effect_pid=effect_pid)
    return done("succeeded", TURN_SUCCESS[criterion], theta_part_rad=theta, stop=stop, criterion=criterion,
                text_items=text)


def _unwanted_stop_reason(stop, kinds) -> str:
    """The failure code of a twist stop that matches no criterion of the step (and no effect seen after it)."""
    if stop == "axis_tug":
        return "axis_tug"
    if stop == "yaw_budget" and "effect" not in kinds:
        return "yaw_budget"
    if stop == "hand_yaw_stall":
        return "hand_yaw_stall"
    return "effect_not_observed"


def _confirm_summary(confirm) -> dict:
    """The post-release confirmation as evidence (blocker-1 spec BR-2: verdict, visible share, share, lift)."""
    return jsonable({k: confirm.get(k) for k in ("verdict", "reason", "lift_m", "frames", "visible_share", "share",
                                                 "flags", "levels", "lift_stopped", "reference")})


def _regrasp_after_twist(ctx, step, grip, track):
    """A slip / tug re-grasp after the part turned: the part's stale record is dropped first (``_mark_turned``, the
    keep-out at the angle turned since its record) and the angle since the record restarts at the new grasp."""
    yield from _turn_release(ctx)
    if track["free"] is not None:
        _mark_turned(ctx, step.obj, grip, theta_world=-track["free"] * track["since_record"],
                     centre_xy=track["centre"])
    track["since_record"] = 0.0
    _moved_by_contact(ctx, step.obj, grip, "release")
    outcome = yield from grasp(ctx, grip, obj_pid=step.obj, purpose="turn", keep=step.keep)
    return outcome


def _segment_summary(segment) -> dict:
    result = segment["result"]
    ev = result.get("evidence") or {}
    return jsonable({"segment": segment["segment"], "sense": segment["sense"], "reason": segment["reason"],
                     "steps": segment["steps"], "decided_by": ev.get("decided_by"), "flags": ev.get("flags"),
                     "theta_sense_rad": ev.get("theta_sense_rad"), "validity": segment.get("validity"),
                     "gap_trace_mm": ev.get("gap_trace_mm")})


def _hinge(ctx, *, sense, part, effect, theta_target, theta_start):
    """HingeTwist in the free sense (spec §4.9 Turn step 4): the centre prior = the part's D11 circle centre, else
    the grip point; sigma_c = max(the circle fit in metres, the close drag) (K11_SIGMA_C_FLOOR_M inside)."""
    g6 = _g6(ctx)
    centre = np.asarray(ctx.view.position, dtype=float)[:2].copy()
    sigma_c = float((ctx.memory.get("gripped") or {}).get("close_drag_m") or 0.0)
    if part is not None and part.circle is not None:
        centre = np.asarray(part.circle[0], dtype=float).reshape(-1)[:2]
        top = float(part.top_z.value) if part.top_z is not None else float(part.xyz[2])
        depth = max(float(_pose(ctx).camera[2]) - top, 1e-6)
        sigma_c = max(sigma_c, float(part.circle[2]) * depth / _f_px(ctx))
    patch, axis_uv = _patch_now(ctx, part)
    wedge_end = observers.WedgeObserver(mode="end_stop", robot=ctx.robot,
                                        sigma_g_mm_per_step=ctx.cal.sigma_g_mm_per_step, hold_floor_mm=g6)
    rotation = observers.PartRotationObserver(robot=ctx.robot, patch=patch, axis_uv=axis_uv)
    budget = _budget_left(ctx, sense)
    wedge_capable = bool(part.is_elongated()) if part is not None else False
    return (yield from _run(lambda v: primitives.HingeTwist(
        v, sense=sense, wedge_end=wedge_end, effect=effect, part_rotation=rotation, hold_floor_mm=float(g6.value),
        theta_target_rad=theta_target, yaw_budget_left_rad=budget, centre_prior_xy=centre, sigma_c_m=sigma_c,
        wedge_capable=wedge_capable, robot=ctx.robot, theta_start_rad=theta_start or None, name="hinge_twist")))


def _effect_event(ctx, watch, hinge_ev, validity) -> None:
    report = (hinge_ev or {}).get("effect") or {}
    ev = report.get("evidence") or {}
    _emit(ctx, "effect", camera="external", watched=watch is not None, verdict=report.get("verdict"),
          onset_step=ev.get("onset_step"), theta_e_rad=hinge_ev.get("theta_e_rad"), fraction=ev.get("fraction"),
          threshold=(report.get("thresholds") or [None])[0], validity=validity)


def _turn_post_check(ctx, *, obj, effect):
    """Turn step 7: spring-back over K9_FRAMES held frames (SpringBackObserver on the bus, the part's external
    point) and, for an effect criterion, the post-release confirmation (blocker-1 spec BR-2, ``_effect_confirm``:
    ``confirm`` against the pre-action reference, lifting while the region is hidden); "persists" carries its
    verdict."""
    out = {}
    pred = ctx.scene.predicted_uv(obj, ctx.view, "external")
    spec = {"unmeasurable": "no_external_point"} if pred is None else {
        "uv": (float(pred[0][0]), float(pred[0][1])), "camera": "external"}
    key = f"spring_back:{obj}"
    try:
        _bus_add(ctx, key, observers.SpringBackObserver(patch_or_axis=spec, robot=ctx.robot), preempt=False,
                 scope="relation")
    except ValueError as exc:
        key = None
        out["spring_back_error"] = str(exc)
    yield from _run(lambda v: HoldPrimitive(v, steps=observers.K9_FRAMES, gripper="open", name="spring_back_hold"))
    if key is not None:
        snap = ctx.monitor.snapshot().get(key) or {}
        out["spring_back"] = snap.get("verdict")
        out["spring_back_reason"] = snap.get("reason")
        ctx.monitor.remove(key)
    if effect is not None and out.get("spring_back") != "violated":
        confirm = yield from _effect_confirm(ctx, effect)
        out["confirm"] = confirm
        out["persists"] = confirm["verdict"]
        out["persists_reason"] = confirm["reason"]
    return out


def _turn_stays(ctx, step, *, obj, effect, effect_pid) -> None:
    """Turn step 8 [C-48]: ``stays:turned:<obj>``, the effect variant of RegionStaysCheck (the effect region still
    changed against its pre-twist baseline) or, without an effect criterion, the part's external region unchanged
    against the post-release view."""
    if effect is not None:
        # blocker-1 spec BR-3: region0 with the watch's pre-action reference, its hidden mask and the excludes
        check = effect.stays_check(projector_mask=_hand_mask_fn(ctx))
    else:
        region = _external_region(ctx, obj)
        if region is None:
            return
        check = observers.RegionStaysCheck(region=region, baseline_rgb=np.asarray(ctx.view.external_rgb_upright),
                                           projector_mask=_hand_mask_fn(ctx), sigma_px=_sigma_px(ctx, "external"),
                                           false_rate=0.0, kind="unchanged")
    ctx.stays.add(f"stays:turned:{obj}", check, step_i=step.i, kind="step")


def _articulated_keepout(ctx, pid, *, theta_world) -> None:
    part = ctx.scene.parts.get(pid)
    if part is None or part.xyz is None:
        return
    try:
        keepout = planner.keepout_for(part, "articulated", sigma_loc_m=planner.horizontal_sigma_m(part),
                                      sigma_h_m=float(part.sigma_h_m or 0.0), theta_part_rad=float(theta_world))
    except ValueError:
        return
    ctx.scene.add_keepout(keepout)


# =============================================================================================== place_on
def _destination_check(ctx, record) -> dict:
    """[C-7] planner.destination_precondition; a destination registered without its own support height (its ring
    measured nothing below it) is checked against the scene's measured base support plane (planes[0]) instead."""
    pre = dict(planner.destination_precondition(record))
    if pre.get("ok") or record.support_z is not None or record.top_z is None:
        return pre
    base = _base_plane(ctx)
    if base is None or base.value >= float(record.top_z.value):
        return pre
    out = dict(planner.destination_precondition(dataclasses.replace(record, support_z=base)))
    out["support_source"] = "base_plane"
    return out


def _destination_ok(ctx):
    def need(record):
        return bool(_destination_check(ctx, record).get("ok"))
    return need


def _aim_check(ctx, held, ref, aim):
    """Spec §4.9 AimCheck, the external n-hat row (parallax-free for the vertical offset between the held object and
    the destination): n_hat . (template - projected target) against n_hat . J (object - target, horizontal). A
    disagreement above AIMCHECK_SIGMA_K sigma (re-measured on a second frame after a settle) is corrected along the
    world direction whose image motion is n_hat (one HeldReach leg, <= AIMCHECK_MAX_CORRECTION_M). The wrist row is
    not run in the slice (evidence aimcheck_unavailable; the external row decides alone, contract §C10.4 step 5)."""
    out = {"wrist": "aimcheck_unavailable"}
    external = getattr(ctx.cal, "external", None)
    projector = ctx.scene.projector
    template = held.ext_template
    if external is None or projector is None or template is None or ref.top_z is None:
        out["external"] = "aimcheck_unavailable"
        return out

    def row():
        track = copy.deepcopy(template)
        uv, peak = track.match(_external_gray(ctx.view))
        if uv is None or peak is None or float(peak) < AIMCHECK_MIN_PEAK:
            return None
        pose = _pose(ctx)
        obj_xy = held.object_ref_point(pose)[:2]
        target_xy = np.asarray(aim["target_obj_xy"], dtype=float)
        target_uv, sigma = projector.project(np.r_[target_xy, float(ref.top_z.value)])
        J = np.asarray(external.J, dtype=float)
        n_hat = np.asarray(external.n_hat, dtype=float)
        measured = float(n_hat @ (np.asarray(uv, dtype=float) - np.asarray(target_uv, dtype=float)))
        expected = float(n_hat @ (J[:, :2] @ (obj_xy - target_xy)))
        return {"measured_px": measured, "expected_px": expected, "disagreement_px": measured - expected,
                "sigma_px": float(math.hypot(float(sigma), float(getattr(projector, "sigma_px", 0.0)))),
                "peak": float(peak)}
    first = row()
    out["external_rows"] = [first]
    if first is None:
        out["external"] = "aimcheck_unavailable"
        return out
    limit = AIMCHECK_SIGMA_K * first["sigma_px"]
    out["threshold"] = th_value("AIMCHECK_n_row_px", limit, "ONLINE", "AIMCHECK_SIGMA_K x projector sigma",
                                sigma_px=first["sigma_px"])
    if abs(first["disagreement_px"]) <= limit:
        out["external"] = "agrees"
        return out
    quat = np.asarray(ctx.view.quaternion_xyzw, dtype=float)
    yield from _run(lambda v: primitives.b4_settle(v, gripper="close", orientation_xyzw=quat, name="aimcheck_settle"))
    second = row()
    out["external_rows"].append(second)
    if second is None or abs(second["disagreement_px"]) <= limit:
        out["external"] = "transient"
        return out
    J2 = np.asarray(external.J, dtype=float)[:, :2]
    direction = J2.T @ np.asarray(external.n_hat, dtype=float)
    norm2 = float(direction @ direction)
    if norm2 < 1e-12:
        out["external"] = "aimcheck_unavailable"
        return out
    d = -second["disagreement_px"] * direction / norm2
    if float(np.linalg.norm(d)) > AIMCHECK_MAX_CORRECTION_M:
        out["external"] = "correction_too_large"
        out["correction_m"] = float(np.linalg.norm(d))
        return out
    moved = yield from held_move(ctx, [np.asarray(ctx.view.position, dtype=float) + np.r_[d, 0.0]], kind="aim")
    out.update(external="corrected", correction_m=float(np.linalg.norm(d)), correction_xy=[float(v) for v in d],
               move=moved.get("reason"), move_ok=bool(moved.get("ok")))
    if not moved.get("ok"):
        out["move_failure"] = {k: moved.get(k) for k in ("reason", "primitive_reason", "preemption", "backoff")}
    return out


def _apply_aim_correction(ctx, held, aim, correction_xy):
    """BRR-4: a confirmed AimCheck disagreement is a MEASURED offset error of the held object, e = -d (d = the hand
    correction that puts the object on the target): the HeldFrame offset moves by R^T e (with the scene's
    hand-attached copies, as ``relocalize`` moves them for a drag), so a re-aim (planner.aim_target) keeps the
    correction, and the lowering holds the corrected grip target (target_grip_xy + d) instead of pulling the hand back
    to the uncorrected one. Returns (held, aim)."""
    d = np.asarray(correction_xy, dtype=float).reshape(-1)[:2]
    pose = _pose(ctx)
    e = np.r_[-d, 0.0]
    new = dataclasses.replace(held, offset_h=np.asarray(held.offset_h, dtype=float) + pose.R.T @ e)
    shift = getattr(ctx.held_estimator, "_shift_attached", None)
    if callable(shift):
        shift(held, pose, e)
    if getattr(ctx.scene, "held", None) is not None and getattr(ctx.scene.held, "pid", None) == held.pid:
        ctx.scene.held = new
    ctx.holding = new
    corrected = dict(aim, target_grip_xy=[float(v) for v in np.asarray(aim["target_grip_xy"], dtype=float) + d],
                     aim_correction_xy=[float(v) for v in d])
    return new, corrected


def _external_gray(view) -> np.ndarray:
    """The upright external image as float grey (the scene's NCCTrack templates use this conversion)."""
    return cv2.cvtColor(np.ascontiguousarray(view.external_rgb_upright), cv2.COLOR_RGB2GRAY).astype(float)


def _lower(ctx, lowering, aim):
    """Spec §4.9 GuardedLower: fast to the M6 height (a support cue there hands over: a support event), then slow to
    the M7 bound; the drift invariant (M5) and lateral_unreachable inside. Returns {"reason", "event", "g_event",
    "lateral_error_m", "phases"}."""
    aim_xy = np.asarray(aim["target_grip_xy"], dtype=float)
    drift = float(lowering["drift_abort_m"])
    floor = float(_g6(ctx).value)
    phases = []
    out = {"reason": None, "event": None, "g_event": None, "lateral_error_m": None, "phases": phases}
    fast = float(ctx.view.position[2]) - float(lowering["fast_until_grip_z"])
    if fast > MIN_MOVE_OFFSET_M / 2.0:
        result, _ = yield from _run(lambda v: primitives.GuardedLower(v, phase="fast", travel_m=fast, aim_grip_xy=aim_xy,
                                                                      drift_abort_m=drift, hold_floor_mm=floor))
        ev = result.get("evidence") or {}
        phases.append({"phase": "fast", "reason": result.get("reason"), "steps": result.get("actions")})
        out.update(lateral_error_m=ev.get("drift_m"), g_event=ev.get("event_grip_z"))
        if _preempted(result) or result.get("status") != "succeeded":
            out["reason"] = _reason_code(result.get("reason"), "lateral_drift")
            out["drift_at_start"] = ev.get("drift_at_start")
            yield from _lower_preempted(ctx, result, out)
            return out
        if str(result.get("reason") or "").startswith(HANDOVER_PREFIX):
            out.update(reason="support_event", event=result.get("reason"))
            return out
    slow = float(ctx.view.position[2]) - float(lowering["bound_grip_z"])
    if slow <= MIN_MOVE_OFFSET_M / 2.0:
        out.update(reason="support_event", event="bound_reached", g_event=float(ctx.view.position[2]))
        return out
    result, _ = yield from _run(lambda v: primitives.GuardedLower(v, phase="slow", travel_m=slow, aim_grip_xy=aim_xy,
                                                                  drift_abort_m=drift, hold_floor_mm=floor))
    ev = result.get("evidence") or {}
    phases.append({"phase": "slow", "reason": result.get("reason"), "steps": result.get("actions")})
    out.update(lateral_error_m=ev.get("drift_m"), g_event=ev.get("event_grip_z"))
    if _preempted(result) or result.get("status") != "succeeded":
        out["reason"] = _reason_code(result.get("reason"), "lateral_drift")
        yield from _lower_preempted(ctx, result, out)
        return out
    out.update(reason="support_event", event=result.get("reason"))
    return out


def _lower_preempted(ctx, result, out):
    """A bus preemption of GuardedLower: its invariant goes into ``out["preemption"]``; a keep-out contact (O5) backs
    off straight up by HELD_CONTACT_BACKOFF_M (v1 rule for a descending move; contract §C10.2 step 11)."""
    if not _preempted(result):
        return
    out["preemption"] = _preemption_info(result)
    if out["reason"] == "keepout_contact":
        out["backoff"] = yield from _contact_backoff(ctx, WORLD_DOWN, name="keepout_backoff")


def _external_point(ctx, held):
    """The held object's external template match (uv, peak) in the current external view, or None."""
    template = getattr(held, "ext_template", None) if held is not None else None
    if template is None:
        return None
    track = copy.deepcopy(template)
    uv, peak = track.match(_external_gray(ctx.view))
    return None if uv is None else (np.asarray(uv, dtype=float), float(peak or 0.0))


def _rest_check(ctx, obj):
    """Spec §4.9 rest (M11): RestObserver over the object's external region while the hand is still, M11_FRAMES."""
    region = _external_region(ctx, obj)
    if region is None:
        return {"verdict": None, "reason": "region_unavailable"}
    key = f"rest:{obj}"
    _bus_add(ctx, key, observers.RestObserver(region=region, sigma_static_grey=float(ctx.cal.sigma_static_grey),
                                              camera="external", static_hand_mask_fn=_hand_mask_fn(ctx)),
             preempt=False, scope="relation")
    yield from _run(lambda v: HoldPrimitive(v, steps=observers.M11_FRAMES, gripper="open", name="rest_hold"))
    snap = ctx.monitor.snapshot().get(key) or {}
    ctx.monitor.remove(key)
    return {"verdict": snap.get("verdict"), "reason": snap.get("reason")}


class _Resumed:
    """An observer put back on the bus after a recovery WITHOUT a restart (InvariantMonitor.add calls ``start``,
    which would re-take its start references: an UprightObserver would forget the grasp-time tilt)."""

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "resumed")

    def start(self, view):
        return None

    def update(self, before, action, after, context=None):
        return self.inner.update(before, action, after, context)


def _empty_hand_cleanup(ctx, obj, grip):
    """A lost hold with the finger gap below G6 (the object is no longer in the hand): open the hand (the held scope
    cleared, ``ctx.holding`` None; the scene writes the hand-attached records back at the release pose, the best
    estimate of where a dropped object lies) and forget the parts' view memory, so a repair retry grasps again,
    re-measuring at its hover, instead of carrying an empty hand (each retry of which a still-violated hold
    invariant would preempt at its first action). The records are kept, not dropped: [F, real task 2, BR-fix chain
    run 4] a dropped grip record fell back on its start-view seed lift, 6 cm off the part on the support, and every
    re-grasp hover failed pose_drift. Nothing happens while the gap still holds. Returns the evidence fields."""
    if ctx.holding is None or float(ctx.view.gap_mm) >= float(_g6(ctx).value):
        return {}
    released = yield from _release_grip(ctx)
    _moved_by_contact(ctx, obj, grip, "release")
    return {"hold_lost_released": released.get("reason"), "flags": ["hold_lost_released"]}


def _restore_upright(ctx, key):
    """Contract §C10.2 step 11, the upright row: the violated UprightObserver ``key`` leaves the bus (a still-violated
    invariant would preempt the restore), the hand turns back to the grasp orientation (Orient, gripper closed,
    POST_ORIENT_HOLD_STEPS settle) and the SAME observer re-judges the hold at the restored pose. Met -> it goes back
    on the bus unrestarted ({"ok": True}); still violated (an in-hand tilt beyond O4, a rigid-hold loss) or no
    observer -> {"ok": False} with the cause."""
    out = {"ok": False, "key": key}
    if key is None or key not in ctx.monitor.keys() or ctx.grasp_quaternion is None:
        out["why"] = "no_upright_observer"
        return out
    observer = ctx.monitor.observer(key)
    ctx.monitor.remove(key)
    quat = np.asarray(ctx.grasp_quaternion, dtype=float)
    result, _ = yield from _run(lambda v: Orient(v, target_xyzw=quat, gripper="close", name="orient_upright_restore"))
    yield from _run(lambda v: HoldPrimitive(v, steps=POST_ORIENT_HOLD_STEPS, gripper="close",
                                            name="upright_restore_settle"))
    out["orient"] = result.get("reason")
    report = observer.update(ctx.view, None, ctx.view, {"commanded_reorientation": False,
                                                        "primitive": "upright_recheck", "step": int(ctx.step)})
    out["recheck"] = {"verdict": report.verdict, "reason": report.reason,
                      "cause": (report.evidence or {}).get("cause"),
                      "tilt_change_deg": (report.evidence or {}).get("tilt_change_deg")}
    if report.verdict == "violated":
        out["observer"] = observer              # the caller may put it back (a put-down that found no support)
        return out
    ctx.monitor.add(key, _Resumed(observer), view=ctx.view, preempt=True, scope="held")
    out["ok"] = True
    return out


def _lowest_support(ctx, held) -> float:
    """The lowest measured support the held object can be set down on: its pick support, or the scene's base plane
    when that is lower. A put-down lowers toward it and stops at the FIRST support cue on the way (a higher surface
    under the object ends it earlier), so the bound never ends the lowering above a support it could reach."""
    lowest = float(held.support_z_pick.value)
    base = _base_plane(ctx)
    return min(lowest, float(base.value)) if base is not None else lowest


def _put_down(ctx, held):
    """Contract §C10.2 step 11 (an in-hand tilt beyond O4 that the orientation restore did not clear): the object is
    set down on its PICK support (spec §4.7 item 6, Touchdown "back onto the pick support"; the place it stood is
    known to bear it) BEFORE the hand opens: a held transit back over the pick grip xy (planner.transit_path, the
    carry's own rules) when the hand has left it, then a slow Touchdown straight down, its travel the object's
    lowest point above the lowest support it can reach (``_lowest_support``) + TOUCHDOWN_EXTRA_M, ending at the first
    support cue. The Touchdown's hold floor is the calibrated empty floor. ``touched`` False (no support cue: the
    bound, a drift, a failed return, a preemption, or a keep-out contact queued at the touch) means the object is
    not resting on a free support: the caller must NOT open. [F, real task 2, chain_t2 ep0 BR-fix runs] a put-down
    bounded by the highest height-field cell under the footprint (1.04 m) ended bound_reached 3 cm lower, the hand
    opened and the object fell over; a put-down where the tilt was found (2 cm into the carry) touched down on the
    neighbouring keep-out. Returns {"reason", "travel_m", "support_z", "touched", "returned"}."""
    out = {"reason": None, "travel_m": None, "support_z": None, "touched": False, "returned": None}
    pick = np.asarray(held.grasp_pose, dtype=float).reshape(-1)[:3]
    position = np.asarray(ctx.view.position, dtype=float)
    if float(np.linalg.norm(position[:2] - pick[:2])) > MIN_MOVE_OFFSET_M:
        path = planner.transit_path(held, position, pick[:2], ctx.scene.hf, _keepouts(ctx, skip=(held.pid,
                                    held.grip_pid)), ctx.robot, _pose(ctx), parts=_other_parts(
                                    ctx, skip=(held.pid, held.grip_pid)))
        if not path.get("ok"):
            out["returned"] = {"ok": False, "reason": path.get("reason")}
            return out
        moved = yield from held_move(ctx, path["waypoints"][1:], kind="put_down")
        out["returned"] = {k: moved.get(k) for k in ("ok", "reason")}
        if not moved.get("ok"):
            return out
    pose = _pose(ctx)
    support = _lowest_support(ctx, held)
    travel = max(float(held.lowest_point_z(pose)) - support, MIN_MOVE_OFFSET_M)
    floor = float(ctx.cal.gripper.empty_floor_mm)
    result, prim = yield from _run(lambda v: primitives.Touchdown(v, lift_m=travel, hold_floor_mm=floor,
                                                               drift_abort_m=planner.M5_FLOOR_M, name="put_down"))
    queued = ctx.monitor.peek_preemption()
    on_keepout = queued is not None and str(queued[0]).startswith("keepout")
    touched = prim is not None and getattr(prim, "touch_grip_z", None) is not None and not on_keepout
    out.update(reason=result.get("reason"), travel_m=travel, support_z=support, touched=bool(touched),
               keepout_contact=bool(on_keepout))
    return out


def _lateral_unverified(pre, support) -> dict | None:
    """SAL-R2 use: the destination precondition passed on C-7''s lateral cue only (``pre["cue"] == "lateral"``) and
    SupportClassify's lateral tolerance (aim tol + sigma_offset) exceeds the destination's hull inscribed radius
    (``pre["values"]["r_hull_m"]``): {"lateral_tol_m", "r_hull_m"}; else None."""
    if (pre or {}).get("cue") != "lateral":
        return None
    r_hull = ((pre or {}).get("values") or {}).get("r_hull_m")
    tol = (support or {}).get("lateral_tol_m")
    if r_hull is None or tol is None or not float(tol) > float(r_hull):
        return None
    return {"lateral_tol_m": float(tol), "r_hull_m": float(r_hull)}


def _pre_release_capture(ctx, obj) -> dict:
    """PR-1 step 1 (gap round §5): right after SupportClassify on_surface and before the release: the external
    frame, the object's external region at the hold pose (``_external_region``; the whole body under HS-H1) and the
    S-16' hand mask in that frame."""
    try:
        hand = ctx.scene.hand_swept_mask("external", ctx.view, (0.0, 0.0))
    except (ValueError, TypeError, KeyError, AttributeError):
        hand = None
    return {"rgb": np.asarray(ctx.view.external_rgb_upright).copy(), "region": _external_region(ctx, obj),
            "hand": None if hand is None else np.asarray(hand, dtype=bool).copy(), "step": int(ctx.step)}


def _reacquire_at_rest(ctx, obj, expected):
    """PR-1 step 3: one registration of the object at its expected rest record (``release_held``'s write-back) with
    zero model calls: the empty-hand hover over it (``_hover_target`` / ``_arrive``, a hover window), then a
    ParallaxProbe and ``register`` seeded at the record's predicted wrist pixel (source "reacquire"; the scene's
    own seed is the pick-time one). Returns {"record", "reason"}."""
    if expected is None or expected.xyz is None:
        return {"record": None, "reason": "part_not_found"}
    hover = _hover_target(ctx, expected, skip=(obj,))
    target = hover.get("hover_grip_xyz")
    if target is not None:
        arrival = yield from _arrive(ctx, target, skip_parts=(obj,))
        if not arrival["ok"]:
            return {"record": None, "reason": _reason_code(arrival["reason"], "transit_contact")}
        _window_start(ctx, obj)
    predicted = _predict_uv(ctx, expected)
    if predicted is None or not _in_image(predicted[0], WRIST_IMAGE_PX):
        return {"record": None, "reason": "part_not_found"}
    seed = Seed("wrist", (float(predicted[0][0]), float(predicted[0][1])), int(ctx.step), "reacquire",
                max(float(predicted[1]), 1.0), tuple(float(v) for v in ctx.view.state))
    measured = yield from _probe_and_register(ctx, obj, seed, label="post_release")
    if measured["record"] is None:
        return {"record": None, "reason": (measured.get("failure") or {}).get("reason") or "part_not_found"}
    return {"record": measured["record"], "reason": None}


def _post_release_check(ctx, obj, ref, pre_release, support):
    """PR-1 (gap round §5, ruling L10): ``placed`` needs positive evidence that the object is still where it was
    released. Step 2: observers.RegionStaysCheck of the pre-release region against the PRE-release frame (hidden_ref
    = the pre-release hand mask, the current hand mask excluded), evaluated once on the current view: violated ->
    not placed; met -> placed. Step 3 (pending: occluded, or no region): one empty-hand survey registration of the
    object at its written-back rest record (``_survey``, zero model calls): placed when the measured top lies within
    max(D8_FLOOR_M, D8_SIGMA_K hypot(sigma_top, sigma_expected)) of S_dest + h_obj (h_obj = the written-back
    record's top - support: its pick-time height) and its centre within the destination region dilated by
    SupportClassify's lateral tolerance (aim tol + sigma_offset); a failed registration is not placed with flag
    ``post_release_unverified``. Returns {"ok", "evidence", "text_items", "flag"}."""
    out = {"ok": False, "evidence": {"stage": None}, "text_items": [], "flag": None}
    region = (pre_release or {}).get("region")
    if region is not None:
        check = observers.RegionStaysCheck(region=region, baseline_rgb=pre_release["rgb"],
                                           projector_mask=_hand_mask_fn(ctx), hidden_ref=pre_release.get("hand"),
                                           sigma_px=_sigma_px(ctx, "external"), false_rate=0.0, kind="unchanged")
        report = check(ctx.view)
        ev = report.evidence or {}
        out["evidence"].update(stage="region", verdict=report.verdict, reason=report.reason,
                               fraction=ev.get("fraction"), threshold=ev.get("threshold"),
                               visible_share=ev.get("visible_share"))
        if report.verdict == "violated":
            out["text_items"] = [["changed region percent", PERCENT * float(ev.get("fraction") or 0.0), "count"],
                                 ["change limit percent", PERCENT * float(ev.get("threshold") or 0.0), "count"]]
            return out
        if report.verdict == "met":
            out["ok"] = True
            return out
    expected = ctx.scene.parts.get(obj)
    out["evidence"]["stage"] = "registration"
    surveyed = yield from _reacquire_at_rest(ctx, obj, expected)
    record = surveyed.get("record")
    out["evidence"]["survey_reason"] = surveyed.get("reason")
    if record is None or record.top_z is None or expected is None or expected.top_z is None:
        out["flag"] = "post_release_unverified"
        return out
    s_dest = (support or {}).get("S_dest")
    if s_dest is not None and expected.support_z is not None:
        expected_top = float(s_dest) + float(expected.top_z.value) - float(expected.support_z.value)
    else:
        expected_top = float(expected.top_z.value)
    sigma = math.hypot(float(record.top_z.sigma), float(expected.top_z.sigma))
    tol = max(D8_FLOOR_M, D8_SIGMA_K * sigma)
    dz = float(record.top_z.value) - expected_top
    centre = (np.asarray(record.region_xy, dtype=float).mean(axis=0) if record.region_xy is not None
              and len(record.region_xy) >= 3 else np.asarray(record.xyz[:2], dtype=float))
    dest = ctx.scene.parts.get(ref)
    lateral_tol = float((support or {}).get("lateral_tol_m") or 0.0)
    inside = None
    if dest is not None and dest.region_xy is not None and len(dest.region_xy) >= 3:
        contour = np.asarray(dest.region_xy, dtype=np.float32).reshape(-1, 1, 2)
        signed = float(cv2.pointPolygonTest(contour, (float(centre[0]), float(centre[1])), True))
        inside = bool(signed >= -lateral_tol - 1e-9)
    out["evidence"].update(top_m=float(record.top_z.value), expected_top_m=expected_top, dz_m=dz, tol_m=tol,
                           centre_in_destination=inside, lateral_tol_m=lateral_tol)
    out["ok"] = bool(abs(dz) <= tol + 1e-9 and inside is not False)
    if not out["ok"]:
        out["text_items"] = [["top offset", abs(dz), "m"], ["top tolerance", tol, "m"]]
    return out


def place_on(ctx, step):
    """Spec §4.8 supported_on, §4.9 Carry ... Release; contract §C10.4; relation "supported_on". Sole owner of the
    destination precondition: ``ref`` registered with an empty hand (else an empty-hand survey_hover, or failed
    destination_not_registered while holding) and [C-7] (else the survey with the lower-hover retry). Grasp of
    grip[0] (then grip[1]) for a carry unless a grip part is held; aim (M4 / M16); the held transit (the object's
    own cells excluded [C-36]); AimCheck (external row); GuardedLower fast / slow (drift -> lift DRIFT_LIFT_M and
    re-aim, <= B2); SupportClassify: on_surface -> release; perched / blocked_above -> NEVER open, lift PERCH_LIFT_M,
    relocalize, re-aim (<= B2), then support_above_destination; missed_surface -> lift to clearance, re-aim (<= B2);
    no_support_within_bound -> lift to clearance, failed [C-32]. Release (the held bus scope cleared), rest (M11),
    the follow check, the stays entry ``stays:supported_on:<obj>:<ref>`` [C-48]. Records one outcome.

    Review fixes: entered while holding, a failed [C-7] destination precondition fails geometry_unmeasured BEFORE
    the hand leaves (BRR-5: the destination survey needs an empty hand); a confirmed AimCheck correction is kept by
    the lowering and the HeldFrame (BRR-4, ``_apply_aim_correction``); local recovery while holding (contract
    §C10.2 step 11, BRR-6): an upright violation restores the grasp orientation and retries (<= B2), an in-hand tilt
    the restore does not clear puts the object down, releases and re-grasps ONCE, then object_rotated_in_hand; a
    keep-out contact backs off HELD_CONTACT_BACKOFF_M and fails keepout_contact."""
    start = int(ctx.step)
    obj, ref = step.obj, step.ref
    grips = list(step.grip) or [obj]
    pids = _pids(obj, grips, ref)
    evidence = {"attempts": []}
    thresholds = []
    retries = 0
    regrasped = False
    flags_out: list = []     # evidence flags of the placement (IF-R3' tight_fit, SAL-R2 lateral_unverified, PR-1)

    def done(status, reason, **extra):
        if flags_out:
            extra = dict(extra)
            extra["flags"] = [*flags_out, *[f for f in extra.get("flags") or [] if f not in flags_out]]
            extra.update({flag: True for flag in flags_out})
        evidence.update(jsonable(extra))
        return _outcome(ctx, "supported_on", status, reason, pids=pids, evidence=evidence, thresholds=thresholds,
                        start=start, retries=retries)

    if ref is None or ref not in ctx.scene.parts:
        return done("failed", "destination_not_registered")
    destination = ctx.scene.parts[ref]
    if not (_registered(destination) and destination.registered_empty_hand):
        if ctx.holding is not None:
            return done("failed", "destination_not_registered", holding=getattr(ctx.holding, "pid", None))
        surveyed = yield from survey_hover(ctx, ref, purpose="destination", need=_destination_ok(ctx))
        evidence["destination_survey"] = {"status": surveyed.status, "reason": surveyed.reason}
        if surveyed.status != "succeeded":
            return done("failed", "destination_not_registered")
    destination = ctx.scene.parts[ref]
    pre = _destination_check(ctx, destination)
    thresholds += list(pre.get("thresholds") or [])
    evidence["destination_precondition"] = {k: pre.get(k) for k in ("ok", "reason", "values", "support_source",
                                                                     "cue")}
    if not pre.get("ok") and ctx.holding is not None:
        # BRR-5: the precondition must hold before the hand leaves; its survey needs an empty hand
        return done("failed", "geometry_unmeasured", destination_precondition=pre.get("values"),
                    holding=getattr(ctx.holding, "pid", None), text_flags=["destination survey needs an empty hand"])
    if not pre.get("ok"):
        surveyed = yield from survey_hover(ctx, ref, purpose="destination", need=_destination_ok(ctx))
        evidence["destination_survey"] = {"status": surveyed.status, "reason": surveyed.reason}
        destination = ctx.scene.parts[ref]
        pre = _destination_check(ctx, destination)
        if not pre.get("ok"):
            return done("failed", "geometry_unmeasured", destination_precondition=pre.get("values"))
        evidence["destination_precondition"] = {k: pre.get(k) for k in ("ok", "reason", "values", "support_source",
                                                                         "cue")}
    if not any(_holds(ctx, g) for g in grips):
        if ctx.holding is not None:
            return done("failed", "precondition_unmet", holding=getattr(ctx.holding, "pid", None))
        held_out = yield from _grasp_with_fallback(ctx, step, purpose="carry")
        evidence["grasp"] = {"status": held_out.status, "reason": held_out.reason}
        if held_out.status != "succeeded":
            grasp_thresholds, extra = _grasp_failure(held_out)
            thresholds[:] = grasp_thresholds
            return done("failed", held_out.reason, **extra)
    held = ctx.holding
    grip = getattr(held, "grip_pid", grips[0])
    skip = (obj, grip)

    def recover(reason, failure):
        """Contract §C10.2 step 11 while holding: "retry" or the failure reason (keep-out back-offs happen in
        held_move / _lower)."""
        nonlocal held, grip, retries, regrasped
        if reason != "object_rotated_in_hand":
            return reason
        record = {"reason": reason, "preemption": (failure or {}).get("preemption")}
        evidence["attempts"][-1].setdefault("recovery", []).append(record)
        if retries >= B2_LOCAL_RETRIES:
            return reason
        restored = yield from _restore_upright(ctx, ((failure or {}).get("preemption") or {}).get("invariant"))
        record["restore"] = {k: v for k, v in restored.items() if k != "observer"}
        if restored["ok"]:
            retries += 1
            return "retry"
        if regrasped:
            return reason
        record["put_down"] = yield from _put_down(ctx, held)
        if not record["put_down"]["touched"]:
            # never open in the air: the object stays held (its upright invariant back on the bus, unrestarted)
            key = ((failure or {}).get("preemption") or {}).get("invariant")
            if restored.get("observer") is not None and key and key not in ctx.monitor.keys():
                ctx.monitor.add(key, _Resumed(restored["observer"]), view=ctx.view, preempt=True, scope="held")
            return reason
        released = yield from _release_grip(ctx)
        record["release"] = released.get("reason")
        _moved_by_contact(ctx, obj, grip, "put_down")
        regrasped = True
        retries += 1
        again = yield from _grasp_with_fallback(ctx, step, purpose="carry")
        record["regrasp"] = None if again is None else {"status": again.status, "reason": again.reason}
        if again is None or again.status != "succeeded" or ctx.holding is None:
            return reason if again is None else _reason_code(again.reason, reason)
        held = ctx.holding
        grip = getattr(held, "grip_pid", grip)
        return "retry"

    while True:
        skip = (obj, grip)
        destination = ctx.scene.parts[ref]
        aim = planner.aim_target(destination, held, ctx.scene.hf, _keepouts(ctx, skip=skip), _pose(ctx))
        thresholds += list(aim.get("thresholds") or [])
        attempt = {"aim": planner.jsonable_result({k: aim.get(k) for k in (
            "ok", "reason", "target_obj_xy", "target_grip_xy", "tol_m", "m_free_m", "fits", "free_spot")})}
        evidence["attempts"].append(attempt)
        if not aim.get("ok"):
            return done("failed", _reason_code(aim.get("reason"), "lateral_unreachable"))
        if aim.get("fits") is False:
            # IF-R3' (ruling L8): the aim's free radius is short of M16's need: evidence, never a refusal (the
            # guarded lowering, SupportClassify and PR-1 verify the placement)
            attempt["flags"] = [*attempt.get("flags", []), "tight_fit"]
            if "tight_fit" not in flags_out:
                flags_out.append("tight_fit")
        path = planner.transit_path(held, np.asarray(ctx.view.position, dtype=float), aim["target_grip_xy"],
                                    ctx.scene.hf, _keepouts(ctx, skip=skip), ctx.robot, _pose(ctx),
                                    parts=_other_parts(ctx, skip=skip))
        thresholds += list(path.get("thresholds") or [])
        _emit(ctx, "transit", kind="held", waypoints=len(path.get("waypoints") or []), carry_z=path.get("carry_z"),
              swept_max_z=path.get("swept_max_z"), lowest_held_z=path.get("lowest_held_z"),
              overhang_exit=path.get("overhang_exit"), ok=path.get("ok"))
        if not path.get("ok"):
            return done("failed", _reason_code(path.get("reason"), "lateral_unreachable"))
        carry_z = float(path.get("carry_z") or ctx.view.position[2])
        _disturb(ctx, obj, "carry_start")                                                 # X-1
        moved = yield from held_move(ctx, path["waypoints"][1:], kind="carry")
        attempt["carry"] = {k: moved.get(k) for k in ("ok", "reason", "preemption", "backoff", "comotion_reanchored")}
        if not moved["ok"]:
            action = yield from recover(moved["reason"], moved)
            if action == "retry":
                continue
            cleanup = (yield from _empty_hand_cleanup(ctx, obj, grip)) if action == "lost_in_transport" else {}
            return done("failed", action, **cleanup)
        check = yield from _aim_check(ctx, held, destination, aim)
        if isinstance(check.get("threshold"), Threshold):
            thresholds.append(check.pop("threshold"))
        attempt["aimcheck"] = jsonable(check)
        if check.get("external") == "corrected" and check.get("correction_xy") is not None:
            held, aim = _apply_aim_correction(ctx, held, aim, check["correction_xy"])
            attempt["aim_corrected_grip_xy"] = list(aim["target_grip_xy"])
        if check.get("move_failure") and check["move_failure"].get("reason") in (
                "object_rotated_in_hand", "keepout_contact", "lost_in_transport"):
            action = yield from recover(check["move_failure"]["reason"], check["move_failure"])
            if action == "retry":
                continue
            return done("failed", action)
        lowering = planner.lowering_plan(destination, held, aim, sigma_touch_m=_sigma_touch())
        thresholds += list(lowering.get("thresholds") or [])
        attempt["lowering"] = planner.jsonable_result({k: v for k, v in lowering.items() if k != "thresholds"})
        lowered = yield from _lower(ctx, lowering, aim)
        attempt["lower"] = jsonable(lowered)
        if lowered["reason"] != "support_event":
            reason = lowered["reason"]
            if reason in HELD_LOST_REASONS and not attempt.get("comotion_reanchored") \
                    and _template_lost(ctx, lowered.get("preemption")):
                attempt["comotion_reanchored"] = _reanchor_comotion(ctx, lowered["preemption"]["invariant"])
                if attempt["comotion_reanchored"]["ok"] and retries < B2_LOCAL_RETRIES:
                    retries += 1
                    continue
            if reason in HELD_LOST_REASONS:
                cleanup = yield from _empty_hand_cleanup(ctx, obj, grip)
                return done("failed", "lost_in_transport", **cleanup)
            if reason == "lateral_drift" and retries < B2_LOCAL_RETRIES:
                retries += 1
                if not lowered.get("drift_at_start"):
                    yield from _lift(ctx, DRIFT_LIFT_M, gripper="close", name="drift_lift")
                continue
            if reason in ("object_rotated_in_hand", "keepout_contact"):
                action = yield from recover(reason, lowered)
                if action == "retry":
                    continue
                return done("failed", action)
            return done("failed", _reason_code(reason, "lateral_drift"))
        support = planner.support_classify(lowering, lowered.get("g_event"), held,
                                           float(lowered.get("lateral_error_m") or 0.0), event=lowered["event"])
        thresholds += list(support.get("thresholds") or [])
        attempt["support"] = planner.jsonable_result({k: v for k, v in support.items() if k != "thresholds"})
        _emit(ctx, "support", **dict(attempt["support"], pid=obj, ref=ref))
        klass = support["class"]
        if klass == "on_surface":
            lateral = _lateral_unverified(pre, support)
            if lateral is not None:
                # SAL-R2 (gap round §3.4): C-7' passed on its lateral clause only and the lateral tolerance exceeds the
                # destination's hull radius: a landing beside it cannot be told apart laterally (diagnostic only)
                attempt["lateral_unverified"] = lateral
                if "lateral_unverified" not in flags_out:
                    flags_out.append("lateral_unverified")
            break
        if klass in ("perched", "blocked_above") and retries < B2_LOCAL_RETRIES:
            retries += 1
            yield from _lift(ctx, PERCH_LIFT_M, gripper="close", name="perch_lift")
            held = ctx.held_estimator.relocalize(held, ctx.view)
            ctx.holding = held
            continue
        if klass == "missed_surface" and retries < B2_LOCAL_RETRIES:
            retries += 1
            yield from _lift(ctx, max(carry_z - float(ctx.view.position[2]), PERCH_LIFT_M), gripper="close",
                             name="clearance_lift")
            continue
        if klass == "no_support_within_bound":
            yield from _lift(ctx, max(carry_z - float(ctx.view.position[2]), PERCH_LIFT_M), gripper="close",
                             name="clearance_lift")
        bottom = support.get("bottom_z")
        text = [] if bottom is None else [["bottom above surface", float(bottom) - float(support["S_dest"]), "m"]]
        flag = SUPPORT_FLAGS.get(klass)
        return done("failed", SUPPORT_FAILURE.get(klass, "missed_surface"), text_items=text,
                    **({"flags": [flag], flag: True, "text_flags": [flag]} if flag else {}))
    before = _external_point(ctx, held)
    pre_release = _pre_release_capture(ctx, obj)                        # PR-1 step 1, at the hold pose
    released = yield from _release_grip(ctx)
    _moved_by_contact(ctx, obj, grip, "put_down")
    evidence["release"] = {"reason": released.get("reason")}
    if released.get("reason") != "released":
        return done("failed", _reason_code(released.get("reason"), "release_incomplete"))
    after = _external_point(ctx, held)
    if before is not None and after is not None:
        moved_px = float(np.linalg.norm(after[0] - before[0]))
        limit = FOLLOW_SIGMA_K * max(_sigma_px(ctx, "external"), observers.O3_EXTERNAL_FLOOR_PX)
        evidence["follow"] = {"moved_px": moved_px, "limit_px": limit}
        thresholds.append(th_value("FOLLOW_external_px", limit, "DD", "FOLLOW_SIGMA_K x max(sigma_px, "
                                   "O3_EXTERNAL_FLOOR_PX)"))
        if moved_px > limit:
            return done("failed", "object_follows_hand", text_items=[["object image motion", moved_px, "px"]])
    rest = yield from _rest_check(ctx, obj)
    evidence["rest"] = rest
    if rest.get("verdict") == "violated":
        return done("failed", "object_moved_after_release")
    verified = yield from _post_release_check(ctx, obj, ref, pre_release, support)
    evidence["post_release"] = verified["evidence"]
    if not verified["ok"]:
        if verified.get("flag") and verified["flag"] not in flags_out:
            flags_out.append(verified["flag"])
        return done("failed", "object_moved_after_release", text_items=verified.get("text_items") or [],
                    **({"text_flags": [verified["flag"]]} if verified.get("flag") else {}))
    region = _external_region(ctx, obj)
    if region is not None:
        check = observers.RegionStaysCheck(region=region, baseline_rgb=np.asarray(ctx.view.external_rgb_upright),
                                           projector_mask=_hand_mask_fn(ctx), sigma_px=_sigma_px(ctx, "external"),
                                           false_rate=0.0, kind="unchanged")
        ctx.stays.add(f"stays:supported_on:{obj}:{ref}", check, step_i=step.i, kind="step")
    bottom = (evidence["attempts"][-1].get("support") or {}).get("dz_m")
    return done("succeeded", "placed", text_items=[] if bottom is None else [["bottom offset", bottom, "m"]])


# ================================================================================================ release
def release(ctx, step=None):
    """The implicit (``step`` None) or explicit release: ReleaseVerify (open settle, retreat along -R2) with the held
    bus scope cleared first; the scene's held frame is written back (``release_held``); ``ctx.holding`` cleared.
    Records one "released" outcome.

    F4 (cross-module review, 2026-10-19; release only on measured support): a CARRIED object (``ctx.holding`` a
    HeldFrame; a turned fixture is not carried) is first set down on its pick support (``_put_down``: back over
    the pick grip xy, slow Touchdown to the first support cue) and the hand opens only when the put-down touched; no
    support cue -> failed ``no_support_within_bound`` (flag ``release_without_support``), the object stays held.
    [F] the BX F7 rerun of a place_on that failed no_support_within_bound releases implicitly while the hand holds the
    object at clearance height; before this the hand opened there and dropped it."""
    start = int(ctx.step)
    held = ctx.holding
    gripped = ctx.memory.get("gripped") or {}
    obj = getattr(held, "pid", None) or gripped.get("obj") or (None if step is None else step.obj)
    grip = getattr(held, "grip_pid", None) or gripped.get("pid")
    pids = _pids(obj, [grip] if grip else [], None)
    if held is None and not gripped and ctx.view.gap_mm >= _g_open_mm(ctx) - OPEN_TOLERANCE_MM:
        return _outcome(ctx, "released", "succeeded", "released", pids=pids, evidence={"nothing_held": True},
                        start=start)
    put = None
    if held is not None and getattr(held, "support_z_pick", None) is not None:
        put = yield from _put_down(ctx, held)
        if not put.get("touched"):
            # never open in the air: the carried object stays held (its held bus scope untouched)
            return _outcome(ctx, "released", "failed", "no_support_within_bound", pids=pids,
                            evidence={"put_down": put, "flags": ["release_without_support"],
                                      "release_without_support": True,
                                      "text_flags": ["no support under the held object, hand kept closed"]},
                            start=start)
    result = yield from _release_grip(ctx)
    ctx.memory.pop("gripped", None)
    _moved_by_contact(ctx, obj, grip, "release")
    reason = str(result.get("reason") or "")
    status = "succeeded" if reason == "released" else "failed"
    return _outcome(ctx, "released", status, reason if reason in REASONS else "release_incomplete", pids=pids,
                    evidence={"release": {"reason": reason, "gap_mm": float(ctx.view.gap_mm)}, "put_down": put,
                              "text_items": [["finger gap", float(ctx.view.gap_mm), "mm"]]},
                    thresholds=_thresholds(((result.get("evidence") or {}).get("thresholds"))), start=start)


def not_implemented(ctx, step):
    """slide, place_in and push_to are reserved in the slice (§C0.3): failed relation_not_implemented."""
    outcome = _outcome(ctx, step.do, "failed", "relation_not_implemented", pids=_pids(step.obj, list(step.grip),
                                                                                       step.ref),
                       evidence={"do": step.do, "reserved": True}, start=int(ctx.step))
    return outcome
    yield                                       # a generator like every controller


def stays_check_for(ctx, step):
    """BX hook (skip_step of a failed step, spec §3.7): the relation's stays check now, or None. A turn with an
    effect criterion: its effect region changed against the pre-action reference of the turn's stored effect watch
    (blocker-1 spec BR-3: region0, the reference's hidden mask, the excludes; ``EffectWatch.stays_check``), else
    (no turn of the part watched it yet) against the start baseline (the effect variant of RegionStaysCheck); other
    relations have no check here."""
    if step.do != "turn":
        return None
    stored = _stored_effect(ctx, step)
    if stored is not None:
        return f"stays:turned:{step.obj}", stored["watch"].stays_check(projector_mask=_hand_mask_fn(ctx))
    effect = [u.p for u in step.until if u.o == "effect" and u.p is not None]
    baseline = ((ctx.scene.baselines.get("start") or {}).get("external") if hasattr(ctx.scene, "baselines")
                else None)
    if not effect or baseline is None:
        return None
    region = _external_region(ctx, effect[0])
    if region is None:
        return None
    check = observers.RegionStaysCheck(region=region, baseline_rgb=np.asarray(baseline),
                                       projector_mask=_hand_mask_fn(ctx), sigma_px=_sigma_px(ctx, "external"),
                                       false_rate=0.0, kind="effect")
    return f"stays:turned:{step.obj}", check


# ======================================================================================= motionless refusals
def _same_hand_state(ctx, record) -> bool:
    """The hand is where a remembered motionless refusal left it: position within MIN_MOVE_OFFSET_M, orientation
    within ORIENT_TOLERANCE_RAD, the same held part."""
    position = np.asarray(ctx.view.position, dtype=float)
    return (float(np.linalg.norm(position - np.asarray(record["position"], dtype=float))) <= MIN_MOVE_OFFSET_M
            and quaternion_angle_rad(ctx.view.quaternion_xyzw, record["quaternion"]) <= ORIENT_TOLERANCE_RAD
            and getattr(ctx.holding, "pid", None) == record["holding"])


def _motionless_guard(ctx, outcome, *, start, start_pose=None):
    """BRR-2: a failure that made no motion (``steps`` 0: a deterministic refusal, e.g. a reach-limit, a
    precondition or a geometry refusal; or no NET motion: the hand back within MIN_MOVE_OFFSET_M and
    ORIENT_TOLERANCE_RAD of where the controller started, e.g. a still-violated invariant that preempts the first
    action of every retry) is flagged ``motionless`` in its evidence. It is remembered per (relation,
    parts, reason) with the hand state; the same refusal again from an unchanged hand state (a repair retry that
    changed nothing) is flagged ``motionless_repeat`` with its count in the evidence text, and the controller holds
    MOTIONLESS_HOLD_STEPS (doubled per further repeat, at most MOTIONLESS_HOLD_MAX_STEPS) before it returns, so
    an unchanged refusal cannot cost two repair calls per control step. The outcome is still the failure (the
    repair decides); the hold changes nothing in the scene. Returns the hold length (0 when none)."""
    if not isinstance(outcome, RelationOutcome) or outcome.status != "failed":
        return 0
    if int(ctx.step) != int(start) and not (start_pose is not None and _same_hand_state(ctx, start_pose)):
        return 0
    key = "|".join(str(v) for v in (outcome.relation, outcome.pids.get("obj"), ",".join(outcome.pids.get("grip")
                                                                                      or ()),
                                     outcome.pids.get("ref"), outcome.reason))
    memory = ctx.memory.setdefault("motionless", {})
    record = memory.get(key)
    repeat = record is not None and _same_hand_state(ctx, record)
    count = int(record["count"]) + 1 if repeat else 1
    memory[key] = {"count": count, "position": [float(v) for v in ctx.view.position],
                   "quaternion": [float(v) for v in ctx.view.quaternion_xyzw],
                   "holding": getattr(ctx.holding, "pid", None), "step": int(ctx.step)}
    outcome.evidence["motionless"] = True
    hold = 0
    if count > 1:
        hold = int(min(MOTIONLESS_HOLD_STEPS * 2 ** (count - 2), MOTIONLESS_HOLD_MAX_STEPS))
        outcome.evidence.update(motionless_repeat=count - 1, motionless_hold_steps=hold)
        items = list(outcome.evidence.get("text_items") or [])
        outcome.evidence["text_items"] = [["same refusal again, repeats", count - 1, "count"]] + items
        outcome.thresholds = list(outcome.thresholds) + [th_value(
            "MOTIONLESS_hold_steps", hold, "DD", "MOTIONLESS_HOLD_STEPS doubled per repeat, at most "
            "MOTIONLESS_HOLD_MAX_STEPS", repeat=count - 1)]
    outcome.evidence_text = evidence_text(outcome)
    if hold:
        gripper = "close" if (ctx.holding is not None or ctx.memory.get("gripped")) else "open"
        yield from _run(lambda v: HoldPrimitive(v, steps=hold, gripper=gripper, name="motionless_repeat_hold"))
    return hold


def _guarded(controller):
    """A step controller with the BRR-2 motionless-refusal guard (``_motionless_guard``) on its outcome."""
    def run(ctx, step):
        start = int(ctx.step)
        ctx.memory["step_scope"] = int(step.i)      # blocker-2 spec BR-4: escalations and failed rounds per plan step
        start_pose = {"position": [float(v) for v in ctx.view.position],
                      "quaternion": [float(v) for v in ctx.view.quaternion_xyzw],
                      "holding": getattr(ctx.holding, "pid", None)}
        outcome = yield from controller(ctx, step)
        yield from _motionless_guard(ctx, outcome, start=start, start_pose=start_pose)
        return outcome
    run.__name__ = getattr(controller, "__name__", "controller")
    run.__doc__ = controller.__doc__
    run.__wrapped__ = controller
    return run


CONTROLLERS = {"hold": _guarded(hold), "turn": _guarded(turn), "place_on": _guarded(place_on), "release": release,
               "slide": not_implemented, "place_in": not_implemented, "push_to": not_implemented}
