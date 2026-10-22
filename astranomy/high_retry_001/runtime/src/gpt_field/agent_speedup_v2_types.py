"""agent_SpeedUp_v2 shared kernel: one vocabulary of data and math for every v2 module (contract §C4, [C-1]).

Contents: the registries (pids, relation kinds, sources, verdicts, outcome reasons, trace events), the plan data
records, the robot and wrist-camera pose math (``HandPose``, ``WristCamera``, ``RobotModel``; every number comes
from M0's ``agent_speedup_v2_robot``), ``HeightField`` storage and queries with the source precedence of spec §4.5
([C-33]), world-xy polygon helpers, the measurement / scene / need / answer / step / outcome records, the observer
and recorder protocols, and the threshold helpers (``th_max``, ``th_value``). No vision, no planning policy, no I/O.

Frames and units (contract §C1.3): LIBERO world, metres, z up; radians; time in 20 Hz control steps. state8 =
[grip xyz, right_hand world rotvec, q1, q2]; the columns of R = rotation_matrix(state8[3:6]) are R0 (wrist image
up), R1 (jaw, wrist image right) and R2 (approach, the camera optical axis). Hand origin = grip - 0.097 R2 (fact
R1); wrist camera C = hand + 0.05 R0 (fact R2). Wrist pixels are upright, u right, v down, principal point at the
image centre. Twist sense +1 is a positive rotation about the approach R2 [C-25]. HeightField indices are
ij = (iy, ix). Fields without a unit suffix are SI; a suffix (_mm, _px, _deg) overrides.

The kernel imports no other v2 module than M0's robot and thresholds modules (not even for type checking):
geometry classes appear only in string annotations. ``jsonable`` is a restated copy of the v1 helper [C-34].

Construction-time invariants downstream modules rely on: a ``PartRecord`` with a 3-D point carries ``sigma_h_m``
(registered and provisional records also ``registration_step``), and its ``elongation`` is stored only when D10
passes (its ``axis_check`` accepted the axis: D10-OE or the remembered axis of S-12; earlier S-4'); a lower-bound
top (``top_bound``) carries its ``top_upper_z``; a ``top_band`` keeps ``sigma_h_m`` >= hypot(stat_m, model_m); an
accepted ``axis_check`` with a view memory agrees with every remembered axis; a ``RelationOutcome`` names a
relation of ``OUTCOME_RELATIONS`` and a succeeded one carries a success reason of that relation
(``SUCCESS_REASONS``). ``poly_inflate`` moves every side out
by exactly the margin at any yaw; ``poly_iou`` refines its raster below one HeightField cell for small polygons.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import agent_speedup_v2_robot as robot
from .action import _quaternion_product, _rotvec_quaternion
from .agent_speedup_v2_thresholds import LABEL_CLASSES, Threshold
from .contact_skills import (GripperCalibration, Primitive, StepView, axes_to_quaternion_xyzw, finger_gap_mm,
                             matrix_to_quaternion_xyzw, quaternion_angle_rad, rotation_matrix)
from .wrist_servo import P_MATRIX, principal_point

# types-2: design review 1 decision §2.1 K-1 (PartRecord top_bound, top_upper_z, top_band, axis_check), the O-1 and
# hover-arrival reasons, the S-4' / G13+ evidence flags and the G-1 dense-lattice feature key prefix.
# types-3: design review 2 decision §2.1 K-4 (axis_check D10-OE semantics and its documented extra keys method,
# region_collapsed, views, reason; axis_angle_rad), K-5 (width_by_yaw = () on a registered record: width unmeasured).
TYPES_VERSION = "agent-speedup-v2-types-3"

# ------------------------------------------------------------------------------------------------ registries
PIDS = ("A", "B", "C", "D", "E", "F", "G", "H")
DO_KINDS = ("hold", "turn", "slide", "place_on", "place_in", "push_to", "release")
SLICE_DO_KINDS = ("hold", "turn", "place_on", "release")          # slide / place_in / push_to are reserved (§C0.3)
MOVE_PRIORS = ("free", "fixed", "hinged", "sliding", "unknown")
SENSES = ("any", "open", "close")
KEEP_KINDS = ("upright", "clear_of")
UNTIL_KINDS = ("effect", "stop", "fits", "quarter_turn", "half_turn")
HOLD_TRUE_RELS = ("on", "in", "open", "closed", "effect")
REPAIR_ACTIONS = ("retry", "repoint", "use_part", "insert_step", "skip_step", "swap_next", "replan", "hold_still")
ROLES = ("grip", "carried", "ref", "effect", "keepout", "hold_true")
# grip: in a step's grip list (or the default grip = obj of turn / slide); carried: obj of place_on / place_in /
# push_to / hold; ref: a step's ref; effect: an until.effect part; keepout: a keep.clear_of part; hold_true: named
# by a hold_true claim.
PLAN_SOURCES = ("plan", "replan")
MEAS_SOURCES = ("kappa", "loom", "touch", "dense", "plane", "prior")
DEPTH_SOURCES = ("kappa", "loom", "touch", "dense")                # DepthSample.source
HF_SOURCES = ("none", "dense", "plane", "obstacle", "loom", "kappa", "touch")      # int8 codes 0..6 (index)
HF_FILLS = ("plane", "obstacle")                                   # D7a / D7b: inferred, not measured
HF_RANK = {"none": 0, "plane": 1, "obstacle": 1, "dense": 2, "loom": 3, "kappa": 3, "touch": 4}
HF_DISAGREE_SIGMA_K = 3.0
CAMERAS = ("wrist", "external")
SEED_SOURCES = ("plan_cell", "fine", "coarse", "reacquire", "lookat")   # lookat reserved (§C0.3)
POINT_STAGES = ("coarse", "fine")                                  # v1 pointing stages (PointReply.stage)
PLAN_KINDS = ("plan", "replan")
POINT_KINDS = ("point_fine", "point_coarse")
REPAIR_KINDS = ("repair", "review")
NEED_KINDS = PLAN_KINDS + POINT_KINDS + REPAIR_KINDS               # = the client's request kinds (§C12)
POINT_PURPOSES = ("grasp_point", "region_seed", "surface_point")
FINGERTIP_EDGES = ("bottom", "right", "top", "left")               # mark_prompting.FINGERTIP_EDGE values
QUARTER_TURN_COUNT = 4
FEATURE_REJECTS = ("shift_too_small", "residual", "lost", "legs")   # D1 reasons of a FeatureRecord
# FeatureRecord.key prefixes (decision §2.3 G-1, §2.2 S-5): "relief:<pid>:<k>" relief seeds; "dense:<k>" the probe
# tracker's dense lattice around the pointed seed, the only samples DSW may withdraw (grid samples have key None).
RELIEF_FEATURE_KEY_PREFIX = "relief:"
DENSE_FEATURE_KEY_PREFIX = "dense:"
PART_STATUSES = ("seeded", "registered", "failed", "dropped")
# PartRecord fields of design review 1 (decision §2.1 K-1):
TOP_BOUNDS = ("lower",)                                            # top_bound: rulings R2, a hidden or cut top
TOP_BAND_KEYS = ("n", "stat_m", "model_m")                         # top_band: S-1..S-3 band statistics of top_z
AXIS_CHECK_KEYS = ("image_axis", "region_axis", "agree_rad", "accepted", "source")   # axis_check: required keys
AXIS_INCONSISTENT = "axis_inconsistent"                            # axis_check source of an axis refusal (BR-1, P-H)
# Design review 2 (decision §2.1 K-4): D10-OE, the outline + edges consensus (S-11) with the per-part view memory
# (S-12) and the collapse guard (S-13). The required keys keep their names with D10-OE semantics (image_axis = the
# outline axis, region_axis = the agreeing edge axis, agree_rad = their angle); the documented extra keys:
AXIS_CHECK_EXTRA_KEYS = ("method", "region_collapsed", "views", "reason")
AXIS_CHECK_METHODS = ("outline_edges",)                            # method: the D10-OE consensus (S-11)
NOT_ELONGATED = "not_elongated"                                    # axis_check source of a failed outline gate
AXIS_KEPT_EARLIER = "kept_earlier"                                 # S-12: the remembered axis is carried
# reason -> the axis_check source a refusal with that reason carries: the outline gate failed (not elongated: P-C's
# RC-7 harmonic may run, P-H), the outline gate passed but no gated edge axis agrees, or an accepted axis disagrees
# with an accepted axis of the part's view memory (S-12). Both of the latter are AXIS_INCONSISTENT (refine_axis).
AXIS_REFUSAL_SOURCES = {"outline_gate": NOT_ELONGATED, "no_edge_agreement": AXIS_INCONSISTENT,
                        "views_disagree": AXIS_INCONSISTENT}
AXIS_CHECK_REASONS = tuple(AXIS_REFUSAL_SOURCES) + (AXIS_KEPT_EARLIER,)
AXIS_VIEWS_KEYS = ("axes", "disagree_rad")                         # axis_check views: S-12 view memory
KEEPOUT_KINDS = ("clear_of", "articulated", "hold_true")
STEP_STATUSES = ("pending", "active", "done", "failed", "stays", "skipped")       # skipped [C-31]
RELATIONS = ("calibrated", "registered", "held", "turned", "supported_on", "released")
OUTCOME_STATUSES = ("succeeded", "failed", "rejected")             # [C-13] runner vocabulary
VERDICTS = ("pending", "met", "violated", "failed")                # ObsReport (§4.3)
PREEMPTED_PREFIX = "preempted:"
HANDOVER_PREFIX = "handover_"
STATE8_FIELDS = ("grip_x", "grip_y", "grip_z", "rotvec_x", "rotvec_y", "rotvec_z", "q1", "q2")
EVIDENCE_TEXT_MAX_CHARS = 200
PREVIOUS_REJECTED_MAX_CHARS = 120
YAW_GRID_RAD = math.radians(15.0)
AXIS_AGREE_MAX_RAD = YAW_GRID_RAD / 2.0
HF_CELL_M = 0.005
HF_HALF_EXTENT_M = 0.5
WRIST_IMAGE_PX = 512
POLY_MITRE_MAX_RAD = math.pi / 2.0
POLY_IOU_MIN_CELLS = 40

LABELS = {
    "HF_RANK": ("DD", "spec §4.5 depth authority: touch 4 > kappa = loom 3 > dense 2 > the D7a plane and D7b "
                      "obstacle fills 1 > none 0; fills rank below every measurement [C-33]"),
    "HF_DISAGREE_SIGMA_K": ("DD", "§C4.4 equal-rank measurements that disagree by more than 3 combined sigma are a "
                                  "scene change: the newer replaces instead of fusing"),
    "QUARTER_TURN_COUNT": ("DD", "frame alignment: k counter-clockwise quarter turns, k in 0..3 (mark_prompting "
                                 "rotate_quarter; a full turn is four quarter turns)"),
    "EVIDENCE_TEXT_MAX_CHARS": ("DD", "spec §3.7 evidence rule: executor-written evidence text is at most 200 "
                                      "characters"),
    "PREVIOUS_REJECTED_MAX_CHARS": ("DD", "§C4.7 PointNeed.previous_rejected: the measured reason sent with a "
                                          "re-point is at most 120 characters"),
    "YAW_GRID_RAD": ("DD", "spec §4.6 jaw-yaw candidates every 15 deg"),
    "AXIS_AGREE_MAX_RAD": ("DD", "design review 1 decision §3 (D10 / RC-7 agreement, no new number): half the "
                                 "15 deg yaw grid, 7.5 deg. S-4' accepts a D10 axis, and P-C a width-harmonic jaw, only "
                                 "when the image or mask tensor axis and the region-cell PCA axis agree within it; "
                                 "§1.3 axis consensus test: kept all 14 axes with error <= 7.3 deg and refused all 5 "
                                 "with error 10.5-76.5 deg. Design review 2 decision §3 (no new number): the D10-OE "
                                 "outline / edge agreement (S-11) and the S-12 view agreement; judge live: with S-12 "
                                 "the gate set accepted 20/20 and 17/20 grip-part axes with 0 beyond 7.5 deg"),
    "HF_CELL_M": ("DD", "spec §4.3 HeightField 5 mm world cells"),
    "HF_HALF_EXTENT_M": ("DD", "covers the R13 full stretch 0.804 m minus the R14 start offset 0.454 m = 0.35 m "
                               "around the start grip, plus 0.15 m"),
    "WRIST_IMAGE_PX": ("ROBOT", "upright wrist image size (M0 image size; R3 stated at 512 px)"),
    "POLY_MITRE_MAX_RAD": ("DD", "poly_inflate corners: each convex corner is closed by tangent pieces of at most 90 "
                                 "deg, so every side moves out by exactly the margin whatever the world yaw, and no "
                                 "point lies farther than sqrt 2 x the margin from the polygon (a rectangle corner is "
                                 "one piece: its offset is exact)"),
    "POLY_IOU_MIN_CELLS": ("DD", "poly_iou resolves at least 40 raster cells across the larger side of the union "
                                 "bounding box (IoU error of order 1/40 for compact polygons, so parts smaller than a "
                                 "5 mm HeightField cell are still compared); larger polygons keep the 5 mm cell"),
}

REASONS = {
    # calibrated
    "calibrated_online": "Calibrate+ measured the gripper, the camera model and the external model online",
    "calibrated_default": "Calibrate+ fell back to the recorded default gripper calibration",
    "calibration_degraded": "the external model or the hand locator is unavailable; logged, the episode continues",
    # registered
    "registered": "the part was registered from its seed and survey",
    "part_not_found": "the part was not found at its predicted pixel after the allowed re-points",
    "pointed_not_grasp_point": "the pointed pixel does not stand above both finger-landing columns",
    "pointed_not_surface": "the pointed surface is not planar over the held footprint",
    "pointed_not_elongated": "the structure axis at the pointed pixel disagrees with the measured long axis",
    "geometry_unmeasured": "the part geometry needed for a contact plan is not measured",
    "destination_not_registered": "the destination region is not registered with an empty hand",
    "probe_contact": "a survey probe move touched something",
    "mark_lost": "the tracked mark was lost",
    "transit_contact": "a contact occurred on the guarded vertical arrival descent at a hover (decision §2.6)",
    # held
    "hold_pass": "the grasp passed the load test with co-motion",
    "held_for_turn": "the grip part is held for a turn",
    "empty_close": "the close ended at the empty curve: nothing between the pads",
    "blocked_above_part": "a contact occurred above the expected band of the part",
    "column_blocked": "a finger column is blocked above the landing",
    "footprint_blocked": "the hand footprint clearance (P7) fails",
    "turned_hand_blocked": "the turned hand volume does not clear the height field (P12)",
    "part_too_wide": "the part cross-section exceeds the pad separation (P2)",
    "no_pad_overlap": "the pads would overlap the part by less than the P3 minimum",
    "palm_clearance": "the palm margin (P6) fails",
    "no_feasible_candidate": "no candidate contact plan is feasible",
    "no_contact_where_expected": "no contact occurred in the expected band",
    "no_contact_within_bound": "no contact occurred before the approach bound (P9)",
    "hold_lost": "the hold was lost (gap at the hold floor or no co-motion)",
    "hold_jam": "the hold test jammed",
    "grip_unstable": "the gripped part or its body tilted during the load-test lift (O-1): the grip is unstable",
    "grip_not_on_obj": "the gripped part does not move with the planned object",
    "precondition_unmet": "a precondition of the relation is not met",
    "orient_budget": "the reorientation exceeds the yaw budget",
    "orient_stall": "the reorientation stalled",
    "pose_drift": "the hand drifted from the planned pose",
    "align_no_convergence": "Align did not converge",
    "align_blocked": "Align was blocked by a contact",
    "reach_limit": "the arm reached its stretch limit",
    "yaw_budget": "the hand-yaw budget about the approach (R12) is exhausted",
    "frontal_needs_r5b": "frontal hands need ruling R-5b outside the verified mount (reserved)",
    # turned
    "turned_by_effect": "the turn ended after the effect onset plus the K5 margin",
    "turned_to_stop": "the turn reached the mechanism end stop",
    "turned_quarter": "the part turned a quarter turn",
    "turned_half": "the part turned a half turn",
    "not_rotatable": "both senses are blocked: the part does not rotate",
    "sense_undecided": "the wedge observer could not decide the sense",
    "turn_slip": "the pads slipped on the part during the turn",
    "axis_tug": "the realized motion deviated from the commanded orbit (K11)",
    "effect_not_observed": "the effect criterion was not observed within the turn",
    "hand_yaw_stall": "the hand yaw stalled",
    "spring_back": "the part sprang back after release (K9)",
    # supported_on
    "placed": "the object rests on the destination surface, released and at rest",
    "support_above_destination": "the support met lies above the destination surface",
    "missed_surface": "the support met lies below the destination surface",
    "no_support_within_bound": "no support was met before the lowering bound (M7)",
    "lateral_drift": "the held object drifted laterally beyond the M5 abort",
    "lateral_unreachable": "the aim target is laterally unreachable",
    "held_object_contact": "the held object touched something during the carry",
    "lost_in_transport": "the held object was lost during the carry",
    "object_moved_after_release": "the object moved after release",
    "object_follows_hand": "the object follows the hand on the retreat",
    # released
    "released": "the hand is open and clear of the object",
    "release_incomplete": "the gripper did not open to the released gap (M12)",
    "touch_on_retreat": "a touch occurred during the retreat",
    # keeps and stays
    "object_rotated_in_hand": "the held object rotated in the hand beyond O4",
    "keepout_contact": "the hand or the held object entered a keep-out volume (O5)",
    "kept_relation_lost": "a completed relation or a hold_true claim no longer holds",
    # plan and step
    "invalid_plan": "the plan failed validation twice",
    "relation_not_implemented": "the relation is reserved in this build",
    "repair_invalid": "the repair answer failed validation twice",
}
RELATION_REASONS = {
    "calibrated": ("calibrated_online", "calibrated_default", "calibration_degraded"),
    "registered": ("registered", "part_not_found", "pointed_not_grasp_point", "pointed_not_surface",
                   "pointed_not_elongated", "geometry_unmeasured", "destination_not_registered", "probe_contact",
                   "mark_lost", "transit_contact"),
    "held": ("hold_pass", "held_for_turn", "empty_close", "blocked_above_part", "column_blocked", "footprint_blocked",
             "turned_hand_blocked", "part_too_wide", "no_pad_overlap", "palm_clearance", "no_feasible_candidate",
             "no_contact_where_expected", "no_contact_within_bound", "hold_lost", "hold_jam", "grip_not_on_obj",
             "precondition_unmet", "orient_budget", "orient_stall", "pose_drift", "align_no_convergence",
             "mark_lost", "align_blocked", "reach_limit", "yaw_budget", "frontal_needs_r5b", "grip_unstable",
             "transit_contact"),
    "turned": ("turned_by_effect", "turned_to_stop", "turned_quarter", "turned_half", "not_rotatable",
               "sense_undecided", "turn_slip", "axis_tug", "effect_not_observed", "yaw_budget", "hand_yaw_stall",
               "spring_back"),
    "supported_on": ("placed", "support_above_destination", "missed_surface", "no_support_within_bound",
                     "lateral_drift", "lateral_unreachable", "held_object_contact", "lost_in_transport",
                     "object_moved_after_release", "object_follows_hand", "destination_not_registered"),
    "released": ("released", "release_incomplete", "touch_on_retreat", "object_follows_hand"),
    "keeps": ("object_rotated_in_hand", "keepout_contact", "kept_relation_lost"),
    "plan": ("invalid_plan", "relation_not_implemented", "repair_invalid"),
}
# The reasons a SUCCEEDED outcome of each relation carries (§C17). A succeeded outcome must use one of its own
# relation's; a failed or rejected outcome may cross relations (a turn or place_on fails with a nested grasp or
# registration reason, a grasp with a planner Infeasible reason such as geometry_unmeasured) but never uses a success
# reason. calibration_degraded is logged and the episode continues, so it is accepted with either status.
SUCCESS_REASONS = {
    "calibrated": ("calibrated_online", "calibrated_default", "calibration_degraded"),
    "registered": ("registered",),
    "held": ("hold_pass", "held_for_turn"),
    "turned": ("turned_by_effect", "turned_to_stop", "turned_quarter", "turned_half"),
    "supported_on": ("placed",),
    "released": ("released",),
}
EITHER_STATUS_REASONS = ("calibration_degraded",)
RESERVED_DO_KINDS = tuple(kind for kind in DO_KINDS if kind not in SLICE_DO_KINDS)      # slide, place_in, push_to
# RelationOutcome.relation values (§C4.7): the six relations; a reserved relation by its do kind (bare, or written
# "<do> (reserved)"); and the two §C17 rows that are no relation of their own: "plan" for the StepRunner's
# plan-level outcomes (invalid_plan, repair_invalid) and "keeps" for a keeps / stays loss not tied to one relation.
OUTCOME_RELATIONS = (RELATIONS + RESERVED_DO_KINDS + tuple(f"{kind} (reserved)" for kind in RESERVED_DO_KINDS)
                     + ("keeps", "plan"))
EVIDENCE_FLAGS = ("width_mismatch", "effect_region_invalid", "effect_occluded", "aimcheck_unavailable",
                  "camera_model_disagreement", "sense_by_wedge_only", "blocked_by_part_rotation",
                  "part_stop_without_wedge", "touchdown_pivot", "model_claim_hold_still", "wrong_mechanism",
                  "partial_registration", "provisional_registration",
                  # design review 1: S-4' refused the D10 axis (axis_check source); G13+ flagged a misaligned close;
                  # the record's top is only a lower bound (rulings R2, top_bound "lower")
                  AXIS_INCONSISTENT, "misaligned", "lower_bound_top",
                  # design review 2: the S-13 collapse guard fired (axis_check region_collapsed; a BR-1 trigger)
                  "region_collapsed")
PRIMITIVE_REASONS = ("free_sense", "blocked_sense", "effect_margin_reached", "end_stop", "quarter_turn_reached",
                     "half_turn_reached", "bound_reached", "probe_contact", "released", "release_incomplete",
                     "touch_on_retreat")                           # v2 additions; v1 reasons pass through unchanged
EVENTS = ("relation_plan", "plan_validation", "step_start", "step_end", "observer", "wedge", "lookat", "pointing",
          "repair", "review", "model_claim", "threshold_set", "calibration", "registration", "contact_plan",
          "contact", "held_frame", "effect", "support", "transit", "invariant", "stays_check")
NOMINAL_EVENTS = ("relation_plan", "plan_validation", "step_start", "step_end", "threshold_set", "calibration",
                  "registration", "pointing", "contact_plan", "contact", "wedge", "effect", "held_frame", "transit",
                  "observer", "stays_check")


# ------------------------------------------------------------------------------------------- JSON and checks
def jsonable(value: Any) -> Any:
    """Plain JSON-safe copy: numpy to Python, non-finite floats to None (restated agent_speedup_v1.jsonable)."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _choice(value, options, name):
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {tuple(options)}, got {value!r}")
    return value


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative(value, name):
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _vector(value, size, name):
    try:
        result = np.array(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _vector_tuple(value, size, name):
    return tuple(float(v) for v in _vector(value, size, name))


def _optional_uv(value, name):
    return None if value is None else _vector_tuple(value, 2, name)


def _quarter_turns(value, name="quarter_turns"):
    if _count(value, name) >= QUARTER_TURN_COUNT:
        raise ValueError(f"{name} must be in 0..{QUARTER_TURN_COUNT - 1}")
    return int(value)


def _optional_pid(value, name):
    return None if value is None else _choice(value, PIDS, name)


def _readonly(array):
    array = np.array(array, dtype=float)
    array.setflags(write=False)
    return array


def _summarize(obj):
    """JSON-safe summary of a collaborator object (geometry models, NCC tracks): to_dict / summary / type name."""
    if obj is None:
        return None
    for method in ("to_dict", "summary"):
        call = getattr(obj, method, None)
        if callable(call):
            return jsonable(call())
    return {"type": type(obj).__name__}


# ------------------------------------------------------------------------------------------------ small records
@dataclass(frozen=True)
class Meas:
    """A measured scalar (spec §4.3): finite value, sigma >= 0 (same unit), source in MEAS_SOURCES."""
    value: float
    sigma: float
    source: str

    def __post_init__(self):
        object.__setattr__(self, "value", _finite(self.value, "Meas.value"))
        object.__setattr__(self, "sigma", _nonnegative(self.sigma, "Meas.sigma"))
        _choice(self.source, MEAS_SOURCES, "Meas.source")

    def to_dict(self) -> dict:
        return {"value": self.value, "sigma": self.sigma, "source": self.source}


@dataclass(frozen=True)
class Seed:
    """One PartRecord seed: an upright pixel of ``camera`` at policy step ``step``, taken at pose ``pose``."""
    camera: str                      # "wrist" | "external"
    uv: tuple[float, float]          # upright pixels of that camera
    step: int                        # policy step index
    source: str                      # SEED_SOURCES
    quantization_px: float           # coarse 32.0, fine 8.0 (cell 64 / crop-point spacing 16, half)
    pose: tuple[float, ...]          # state8 when the image was taken

    def __post_init__(self):
        _choice(self.camera, CAMERAS, "Seed.camera")
        object.__setattr__(self, "uv", _vector_tuple(self.uv, 2, "Seed.uv"))
        _count(self.step, "Seed.step")
        _choice(self.source, SEED_SOURCES, "Seed.source")
        quantization = _finite(self.quantization_px, "Seed.quantization_px")
        if quantization <= 0.0:
            raise ValueError("Seed.quantization_px must be > 0")
        object.__setattr__(self, "quantization_px", quantization)
        object.__setattr__(self, "pose", _vector_tuple(self.pose, len(STATE8_FIELDS), "Seed.pose"))

    def to_dict(self) -> dict:
        return jsonable({"camera": self.camera, "uv": self.uv, "step": self.step, "source": self.source,
                         "quantization_px": self.quantization_px, "pose": self.pose})


# ---------------------------------------------------------------------------------------------------- plan data
@dataclass(frozen=True)
class PlanPart:
    pid: str
    text: str
    move: str                        # MOVE_PRIORS (a prior only, §3.3)
    w_cell: int                      # wrist grid cell 1..64, 0 = not visible
    x_cell: int                      # external grid cell 1..64, 0 = not visible

    def __post_init__(self):
        _choice(self.pid, PIDS, "PlanPart.pid")
        if not isinstance(self.text, str):
            raise ValueError("PlanPart.text must be a string")
        _choice(self.move, MOVE_PRIORS, "PlanPart.move")
        _count(self.w_cell, "PlanPart.w_cell")
        _count(self.x_cell, "PlanPart.x_cell")


@dataclass(frozen=True)
class Keep:
    c: str                           # KEEP_KINDS
    p: str

    def __post_init__(self):
        _choice(self.c, KEEP_KINDS, "Keep.c")
        _choice(self.p, PIDS, "Keep.p")


@dataclass(frozen=True)
class Until:
    o: str                           # UNTIL_KINDS
    p: str | None
    txt: str | None

    def __post_init__(self):
        _choice(self.o, UNTIL_KINDS, "Until.o")
        _optional_pid(self.p, "Until.p")
        if self.txt is not None and not isinstance(self.txt, str):
            raise ValueError("Until.txt must be a string or None")


@dataclass(frozen=True)
class HoldTrue:
    rel: str                         # HOLD_TRUE_RELS
    a: str
    b: str | None

    def __post_init__(self):
        _choice(self.rel, HOLD_TRUE_RELS, "HoldTrue.rel")
        _choice(self.a, PIDS, "HoldTrue.a")
        _optional_pid(self.b, "HoldTrue.b")


@dataclass(frozen=True)
class PlanStep:
    do: str
    obj: str
    grip: tuple[str, ...]            # preference order; () = (obj,)
    ref: str | None
    sense: str
    keep: tuple[Keep, ...]
    until: tuple[Until, ...]

    def __post_init__(self):
        _choice(self.do, DO_KINDS, "PlanStep.do")
        _choice(self.obj, PIDS, "PlanStep.obj")
        object.__setattr__(self, "grip", tuple(_choice(p, PIDS, "PlanStep.grip") for p in self.grip))
        _optional_pid(self.ref, "PlanStep.ref")
        _choice(self.sense, SENSES, "PlanStep.sense")
        object.__setattr__(self, "keep", tuple(self.keep))
        object.__setattr__(self, "until", tuple(self.until))
        if not all(isinstance(k, Keep) for k in self.keep) or not all(isinstance(u, Until) for u in self.until):
            raise ValueError("PlanStep.keep / until must hold Keep / Until records")

    @property
    def grip_parts(self) -> tuple[str, ...]:
        """The grip list with the default applied (grip == () means (obj,))."""
        return self.grip or (self.obj,)

    def to_json(self) -> dict:
        return {"do": self.do, "obj": self.obj, "grip": list(self.grip), "ref": self.ref, "sense": self.sense,
                "keep": [{"c": k.c, "p": k.p} for k in self.keep],
                "until": [{"o": u.o, "p": u.p, "txt": u.txt} for u in self.until]}


@dataclass(frozen=True)
class Plan:
    scene: str
    parts: tuple[PlanPart, ...]
    hold_true: tuple[HoldTrue, ...]
    steps: tuple[PlanStep, ...]
    source: str                      # "plan" | "replan"

    def __post_init__(self):
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "hold_true", tuple(self.hold_true))
        object.__setattr__(self, "steps", tuple(self.steps))
        _choice(self.source, PLAN_SOURCES, "Plan.source")

    def part(self, pid) -> PlanPart:
        for part in self.parts:
            if part.pid == pid:
                return part
        raise KeyError(pid)

    def to_json(self) -> dict:
        """Wire JSON in the §3.3 insertion order (scene, parts, hold_true, steps); ``source`` is not on the wire."""
        return {"scene": self.scene,
                "parts": [{"id": p.pid, "text": p.text, "move": p.move, "w": p.w_cell, "x": p.x_cell}
                          for p in self.parts],
                "hold_true": [{"rel": h.rel, "a": h.a, "b": h.b} for h in self.hold_true],
                "steps": [s.to_json() for s in self.steps]}


@dataclass(frozen=True)
class RepairAnswer:
    why: str
    action: str                      # REPAIR_ACTIONS
    part: str | None
    step: PlanStep | None

    def __post_init__(self):
        if not isinstance(self.why, str):
            raise ValueError("RepairAnswer.why must be a string")
        _choice(self.action, REPAIR_ACTIONS, "RepairAnswer.action")
        _optional_pid(self.part, "RepairAnswer.part")
        if self.step is not None and not isinstance(self.step, PlanStep):
            raise ValueError("RepairAnswer.step must be a PlanStep or None")

    def to_json(self) -> dict:
        return {"why": self.why, "action": self.action, "part": self.part,
                "step": None if self.step is None else self.step.to_json()}


# ------------------------------------------------------------------------------------------ robot and camera
def quaternion_approach_axis(q_xyzw) -> np.ndarray:
    """Third column (R2, the approach) of the rotation of a unit quaternion (x, y, z, w)."""
    q = _vector(q_xyzw, 4, "quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)])


def twist_about(q_before_xyzw, q_after_xyzw, axis) -> float:
    """Signed rotation (rad) about ``axis`` of the world-frame rotation taking q_before to q_after (swing-twist
    decomposition of q_after * q_before^-1; positive by the right-hand rule about ``axis``)."""
    before = _vector(q_before_xyzw, 4, "q_before")
    after = _vector(q_after_xyzw, 4, "q_after")
    before, after = before / np.linalg.norm(before), after / np.linalg.norm(after)
    unit = _vector(axis, 3, "axis")
    norm = float(np.linalg.norm(unit))
    if norm < 1e-9:
        raise ValueError("axis must have nonzero norm")
    unit = unit / norm
    relative = _quaternion_product(after, np.r_[-before[:3], before[3]])
    if relative[3] < 0.0:
        relative = -relative
    return float(2.0 * math.atan2(float(relative[:3] @ unit), float(relative[3])))


@dataclass(frozen=True, eq=False)
class HandPose:
    """Hand pose from one state8 (contract §C1.3)."""
    grip: np.ndarray                 # (3,) world grip site
    R: np.ndarray                    # (3,3) columns R0 (image up), R1 (jaw), R2 (approach)
    hand: np.ndarray                 # grip - 0.097 R2 (fact R1)
    camera: np.ndarray               # hand + 0.05 R0 (fact R2)
    gap_mm: float
    quaternion_xyzw: np.ndarray

    @property
    def up(self) -> np.ndarray:
        return self.R[:, 0]

    @property
    def jaw(self) -> np.ndarray:
        return self.R[:, 1]

    @property
    def approach(self) -> np.ndarray:
        return self.R[:, 2]

    @classmethod
    def from_state8(cls, state8) -> "HandPose":
        state = _vector(state8, len(STATE8_FIELDS), "state8")
        R = rotation_matrix(state[3:6])
        grip = state[:3].copy()
        hand = grip - robot.GRIP_SITE_ALONG_R2_M * R[:, 2]
        camera = hand + robot.WRIST_CAMERA_ALONG_R0_M * R[:, 0]
        return cls(_readonly(grip), _readonly(R), _readonly(hand), _readonly(camera), finger_gap_mm(state),
                   _readonly(_rotvec_quaternion(state[3:6].copy())))

    @classmethod
    def from_view(cls, view: StepView) -> "HandPose":
        return cls.from_state8(view.state)

    def to_dict(self) -> dict:
        return jsonable({"grip": self.grip, "R": self.R, "hand": self.hand, "camera": self.camera,
                         "gap_mm": self.gap_mm, "quaternion_xyzw": self.quaternion_xyzw})


@dataclass(frozen=True)
class WristCamera:
    """Pinhole wrist camera under R-2 / R-2f (§C1.3); f is chosen by geometry.CameraModel."""
    f_px: float
    size_px: int = WRIST_IMAGE_PX

    def __post_init__(self):
        f = _finite(self.f_px, "WristCamera.f_px")
        if f <= 0.0:
            raise ValueError("WristCamera.f_px must be > 0")
        object.__setattr__(self, "f_px", f)
        if _count(self.size_px, "WristCamera.size_px") < 1:
            raise ValueError("WristCamera.size_px must be >= 1")

    @classmethod
    def from_robot(cls, size_px: int | None = None) -> "WristCamera":
        """The R3 camera (ruling R-2f): f = focal_length_px(size), 333.6 px at 512 px."""
        size = WRIST_IMAGE_PX if size_px is None else int(size_px)
        return cls(robot.focal_length_px(size), size)

    @property
    def principal_point(self) -> np.ndarray:
        return principal_point(self.size_px, self.size_px)

    def project(self, pose: HandPose, xyz) -> tuple[np.ndarray | None, float]:
        """(upright uv or None when z <= 0, depth z along the optical axis R2)."""
        d = _vector(xyz, 3, "xyz") - pose.camera
        a, b, z = float(d @ pose.R[:, 1]), float(d @ pose.R[:, 0]), float(d @ pose.R[:, 2])
        if not z > 0.0:
            return None, z
        c = self.principal_point
        return np.array([c[0] + self.f_px * a / z, c[1] - self.f_px * b / z]), z

    def _direction(self, pose: HandPose, uv) -> np.ndarray:
        u, v = _vector(uv, 2, "uv")
        c = self.principal_point
        return (pose.R[:, 2] + ((u - c[0]) / self.f_px) * pose.R[:, 1]
                - ((v - c[1]) / self.f_px) * pose.R[:, 0])

    def ray(self, pose: HandPose, uv) -> tuple[np.ndarray, np.ndarray]:
        """(origin C, unit world direction) of the pixel's viewing ray."""
        direction = self._direction(pose, uv)
        return pose.camera.copy(), direction / np.linalg.norm(direction)

    def backproject(self, pose: HandPose, uv, depth_z: float) -> np.ndarray:
        """Spec §4.5 world point: C + Z (R2 + ((u - c_u)/f) R1 - ((v - c_v)/f) R0), Z along the optical axis."""
        return pose.camera + _finite(depth_z, "depth_z") * self._direction(pose, uv)

    def kappa_at(self, depth_z: float) -> float:
        """f / z: px per metre of in-plane hand translation at depth z."""
        z = _finite(depth_z, "depth_z")
        if z <= 0.0:
            raise ValueError("depth_z must be > 0")
        return self.f_px / z

    def to_dict(self) -> dict:
        return {"f_px": self.f_px, "size_px": self.size_px}


def _rect_corners(r1_range, r0_range):
    """CCW corners in (R1, R0) coordinates of an axis rectangle."""
    (a1, b1), (a0, b0) = r1_range, r0_range
    return np.array([[a1, a0], [b1, a0], [b1, b0], [a1, b0]], dtype=float)


class RobotModel:
    """Robot facts as a model object (one instance per policy; every number comes from M0 agent_speedup_v2_robot).

    Top-down hands (R2 ~ world -z) are assumed by the footprint helpers: the world-xy direction of R0 is the jaw
    direction turned +90 deg about world +z (R0 = R1 x R2)."""

    def __init__(self):
        self.t_f_m = robot.FINGERTIP_REACH_MM / 1000.0                       # R5
        self.p_min_m = robot.PALM_UNDERSIDE_MM / 1000.0                      # R6
        self.pad_span_m = tuple(v / 1000.0 for v in robot.PAD_SPAN_ALONG_APPROACH_MM)   # R4 along the approach
        self.pad_length_m = robot.PAD_LENGTH_R0_MM / 1000.0                  # R4 (pad length along R0)
        self.grip_axis_offset_m = (0.0, -robot.WRIST_CAMERA_ALONG_R0_M)      # (R1, R0) offset of the grip axis
        asset = robot.ROBOT_FACTS["asset"]
        self._hand_r1_mm = tuple(float(v) for v in asset["hand_footprint_R1_mm"])
        self._hand_r0_mm = tuple(float(v) for v in asset["hand_footprint_R0_mm"])
        footprints = robot.finger_landing_footprints()
        keys = sorted(footprints, key=float)
        self._footprint_q_mm = np.array([float(k) for k in keys])
        self._footprints = {
            finger: {axis: np.array([[float(v) for v in footprints[k][finger][axis]] for k in keys])
                     for axis in ("R1_mm", "R0_mm")}
            for finger in ("finger1", "finger2")}

    # --- poses
    def pose(self, state8) -> HandPose:
        return HandPose.from_state8(state8)

    def pad_band_z(self, grip_z) -> tuple[float, float]:
        """R4, top-down: pad world z interval [g - 4.4 mm, g + 11.6 mm]."""
        return robot.pad_world_z_band_top_down(_finite(grip_z, "grip_z"))

    def pad_separation_mm(self, gap_mm) -> float:
        """R9: pad separation from the measured finger gap."""
        return robot.pad_separation_mm(_finite(gap_mm, "gap_mm"))

    # --- footprints (world xy, metres)
    @staticmethod
    def _frame_xy(jaw_unit_xy):
        jaw = _vector(jaw_unit_xy, 2, "jaw_unit_xy")
        norm = float(np.linalg.norm(jaw))
        if norm < 1e-9:
            raise ValueError("jaw_unit_xy must have nonzero norm")
        jaw = jaw / norm
        return jaw, np.array([-jaw[1], jaw[0]])

    def _to_world(self, grip_xy, jaw_unit_xy, corners_mm):
        jaw, up = self._frame_xy(jaw_unit_xy)
        origin = _vector(grip_xy, 2, "grip_xy")
        metres = np.asarray(corners_mm, float) / 1000.0
        return origin + np.outer(metres[:, 0], jaw) + np.outer(metres[:, 1], up)

    def finger_ranges_mm(self, finger_q_mm: float) -> dict:
        """Per-finger (R1, R0) ranges in mm at the per-finger opening q (mm), linear between the asset keys and
        clamped to the key range [C-50]."""
        q = _finite(finger_q_mm, "finger_q_mm")
        out = {}
        for finger, axes in self._footprints.items():
            out[finger] = {axis: tuple(float(np.interp(q, self._footprint_q_mm, axes[axis][:, j]))
                                       for j in range(axes[axis].shape[1]))
                           for axis in axes}
        return out

    def finger_columns_xy(self, grip_xy, jaw_unit_xy, *, finger_q_mm: float,
                          inflate_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """P16 [C-50]: each finger's lowest-band footprint at the measured opening (finger_q_mm = g_open / 2, G1) as a
        (4,2) CCW world-xy polygon, finger1 on +jaw, inflated by ``inflate_m`` on every side (None = 0)."""
        inflate_mm = 0.0 if inflate_m is None else _nonnegative(inflate_m, "inflate_m") * 1000.0
        ranges = self.finger_ranges_mm(finger_q_mm)
        polygons = []
        for finger in ("finger1", "finger2"):
            r1, r0 = ranges[finger]["R1_mm"], ranges[finger]["R0_mm"]
            corners = _rect_corners((r1[0] - inflate_mm, r1[1] + inflate_mm), (r0[0] - inflate_mm, r0[1] + inflate_mm))
            polygons.append(self._to_world(grip_xy, jaw_unit_xy, corners))
        return polygons[0], polygons[1]

    def finger_slot_xy(self, grip_xy, jaw_unit_xy, *, finger_q_mm: float) -> np.ndarray:
        """P7 exclusion: the slot between the two finger columns (inner edges along R1) over the pad length along R0
        (+-PAD_LENGTH_R0_MM / 2)."""
        ranges = self.finger_ranges_mm(finger_q_mm)
        inner2, inner1 = ranges["finger2"]["R1_mm"][1], ranges["finger1"]["R1_mm"][0]
        half = robot.PAD_LENGTH_R0_MM / 2.0
        return self._to_world(grip_xy, jaw_unit_xy, _rect_corners((inner2, inner1), (-half, half)))

    def hand_footprint_xy(self, grip_xy, jaw_unit_xy) -> np.ndarray:
        """R7 conservative hand rectangle (asset hand_footprint_R1_mm / _R0_mm), world xy."""
        return self._to_world(grip_xy, jaw_unit_xy, _rect_corners(self._hand_r1_mm, self._hand_r0_mm))

    def palm_underside_z(self, grip_z, xi_mm: float | None = None) -> float | None:
        """R6 profile, top-down: grip_z + p(xi) / 1000 at jaw coordinate xi (None = 0 mm); None outside the hand."""
        p_mm = robot.palm_underside_at(0.0 if xi_mm is None else _finite(xi_mm, "xi_mm"))
        return None if p_mm is None else _finite(grip_z, "grip_z") + p_mm / 1000.0

    # --- yaw (contract sense = rotation about the approach, §C1.3)
    def jaw_yaw(self, R) -> float:
        """Angle of R1's world-xy projection about world +z, mod pi."""
        matrix = np.asarray(R, dtype=float)
        x, y = float(matrix[0, 1]), float(matrix[1, 1])
        if math.hypot(x, y) < 1e-6:
            raise ValueError("the jaw has no horizontal component")
        return math.atan2(y, x) % math.pi

    def yaw_about_approach(self, q_before_xyzw, q_after_xyzw, approach) -> float:
        """Signed rad about ``approach`` of the world rotation from q_before to q_after (a +d R2 world rotvec gives
        +d)."""
        return twist_about(q_before_xyzw, q_after_xyzw, approach)

    def net_yaw(self, start_q, q_now) -> float:
        """Net hand yaw about the START approach axis since the start orientation (contract sense sign)."""
        return twist_about(start_q, q_now, quaternion_approach_axis(start_q))

    def yaw_budget_left(self, start_q, q_now, sense: int) -> float:
        """R12: yaw_budget_rad() - sense * net_yaw, for sense +1 / -1."""
        if sense not in (1, -1) or isinstance(sense, bool):
            raise ValueError("sense must be +1 or -1")
        return robot.yaw_budget_rad() - sense * self.net_yaw(start_q, q_now)

    def quaternion_for(self, jaw_world, approach_world) -> np.ndarray:
        return axes_to_quaternion_xyzw(jaw_world, approach_world)

    def thresholds(self) -> list[Threshold]:
        return robot.robot_thresholds()

    def to_dict(self) -> dict:
        return jsonable({"t_f_m": self.t_f_m, "p_min_m": self.p_min_m, "pad_span_m": self.pad_span_m,
                         "pad_length_m": self.pad_length_m, "grip_axis_offset_m": self.grip_axis_offset_m,
                         "yaw_budget_rad": robot.yaw_budget_rad(), "robot_version": robot.ROBOT_VERSION})


# ------------------------------------------------------------------------------------------ polygon helpers
def _points(pts_xy) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=float)
    return pts.reshape(-1, 2)


def rect_polygon(centre_xy, axis_unit_xy, half_along_m: float, half_across_m: float) -> np.ndarray:
    """(4,2) CCW rectangle centred on centre_xy, ``half_along_m`` along the axis and ``half_across_m`` across."""
    centre = _vector(centre_xy, 2, "centre_xy")
    axis = _vector(axis_unit_xy, 2, "axis_unit_xy")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("axis_unit_xy must have nonzero norm")
    a = axis / norm
    b = np.array([-a[1], a[0]])
    ha, hc = _nonnegative(half_along_m, "half_along_m"), _nonnegative(half_across_m, "half_across_m")
    return np.array([centre - ha * a - hc * b, centre + ha * a - hc * b, centre + ha * a + hc * b,
                     centre - ha * a + hc * b])


def convex_hull(pts_xy) -> np.ndarray:
    """CCW convex hull (Andrew's monotone chain), collinear points dropped."""
    pts = sorted({(float(x), float(y)) for x, y in _points(pts_xy)})
    if len(pts) <= 2:
        return np.array(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def poly_inflate(poly_xy, margin_m: float) -> np.ndarray:
    """Convex offset of the polygon's convex hull by ``margin_m`` that does not depend on the world yaw: every side
    moves out along its own normal by exactly the margin, and each corner is closed by tangent pieces to the
    margin circle of at most POLY_MITRE_MAX_RAD each. The result (CCW) contains the hull's Minkowski sum with a disk
    of radius margin and lies within sqrt(2) x margin of the hull; a rectangle grows by exactly the margin on every
    side, a segment becomes the rectangle around it, and a single point a world-axis square of half-side margin
    (a point has no orientation)."""
    margin = _nonnegative(margin_m, "margin_m")
    hull = convex_hull(poly_xy)
    if len(hull) > 1:                                  # drop numerically coincident hull vertices (noise normals)
        keep = [0] + [i for i in range(1, len(hull)) if np.linalg.norm(hull[i] - hull[i - 1]) > 1e-9]
        hull = hull[keep]
        if len(hull) > 1 and np.linalg.norm(hull[-1] - hull[0]) <= 1e-9:
            hull = hull[:-1]
    if margin == 0.0 or len(hull) == 0:
        return hull
    if len(hull) == 1:
        return hull[0] + np.array([[-margin, -margin], [margin, -margin], [margin, margin], [-margin, margin]])
    out = []
    count = len(hull)
    for i in range(count):
        incoming, outgoing = hull[i] - hull[i - 1], hull[(i + 1) % count] - hull[i]
        start = math.atan2(-incoming[0], incoming[1])            # outward normal of the incoming side (CCW hull)
        turn = (math.atan2(-outgoing[0], outgoing[1]) - start) % (2.0 * math.pi)
        pieces = max(1, math.ceil(turn / POLY_MITRE_MAX_RAD - 1e-9))
        step = turn / pieces
        reach = margin / math.cos(step / 2.0)
        for k in range(pieces):
            angle = start + step * (k + 0.5)
            out.append(hull[i] + reach * np.array([math.cos(angle), math.sin(angle)]))
    return np.array(out)


def poly_contains(poly_xy, pts_xy) -> np.ndarray:
    """Even-odd point-in-polygon test, one bool per point (any simple polygon)."""
    poly, pts = _points(poly_xy), _points(pts_xy)
    inside = np.zeros(len(pts), dtype=bool)
    if len(poly) < 3:
        return inside
    x, y = pts[:, 0], pts[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_cross)
        j = i
    return inside


def _raster(bounds_low, bounds_high, cell_m):
    nx = max(int(math.ceil((bounds_high[0] - bounds_low[0]) / cell_m)), 1)
    ny = max(int(math.ceil((bounds_high[1] - bounds_low[1]) / cell_m)), 1)
    xs = bounds_low[0] + (np.arange(nx) + 0.5) * cell_m
    ys = bounds_low[1] + (np.arange(ny) + 0.5) * cell_m
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def poly_iou(a_xy, b_xy) -> float:
    """Intersection over union, rasterized over the union bounding box on cells of min(HF_CELL_M, larger side /
    POLY_IOU_MIN_CELLS): HeightField cells for polygons of at least 20 cm, finer for smaller ones, so identical
    polygons below one HeightField cell still give 1.0 (error of order 1 / POLY_IOU_MIN_CELLS for compact
    polygons). 0.0 when a polygon has fewer than 3 vertices or the raster resolves no cell of either (zero area, or
    a sliver thinner than one raster cell)."""
    a, b = _points(a_xy), _points(b_xy)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    both = np.vstack([a, b])
    low, high = both.min(axis=0), both.max(axis=0)
    extent = float(np.max(high - low))
    if extent <= 0.0:
        return 0.0
    centres = _raster(low, high, min(HF_CELL_M, extent / POLY_IOU_MIN_CELLS))
    in_a, in_b = poly_contains(a, centres), poly_contains(b, centres)
    union = int(np.count_nonzero(in_a | in_b))
    return 0.0 if union == 0 else float(np.count_nonzero(in_a & in_b)) / union


def rotate_xy(poly_xy, centre_xy, angle_rad: float) -> np.ndarray:
    """Rotate points counter-clockwise (about world +z) by angle_rad about centre_xy."""
    pts = _points(poly_xy)
    centre = _vector(centre_xy, 2, "centre_xy")
    c, s = math.cos(_finite(angle_rad, "angle_rad")), math.sin(angle_rad)
    rotation = np.array([[c, -s], [s, c]])
    return (pts - centre) @ rotation.T + centre


# ------------------------------------------------------------------------------------------------ HeightField
def hf_code(source: str) -> int:
    return HF_SOURCES.index(_choice(source, HF_SOURCES, "HeightField source"))


_FILL_CODES = tuple(HF_SOURCES.index(name) for name in HF_FILLS)


@dataclass(eq=False)
class HeightField:
    """World grid of 5 mm cells (spec §4.3); indices ij = (iy, ix). known = measured OR filled (D7a / D7b); a cell is
    measured when its source is neither "none" nor a fill. Unknown cells hold z = sigma = NaN (internal only)."""
    origin_xy: np.ndarray            # world xy of cell (0, 0) centre
    cell_m: float
    z: np.ndarray                    # (ny, nx) float
    sigma: np.ndarray                # (ny, nx) float
    known: np.ndarray                # (ny, nx) bool
    source: np.ndarray               # (ny, nx) int8 HF_SOURCES code
    stamp: np.ndarray                # (ny, nx) int32 policy step of the last write (-1 never)

    @classmethod
    def empty(cls, centre_xy, *, half_extent_m=HF_HALF_EXTENT_M, cell_m=HF_CELL_M) -> "HeightField":
        cell = _finite(cell_m, "cell_m")
        if cell <= 0.0:
            raise ValueError("cell_m must be > 0")
        half_cells = int(round(_nonnegative(half_extent_m, "half_extent_m") / cell))
        n = 2 * half_cells + 1
        centre = _vector(centre_xy, 2, "centre_xy")
        shape = (n, n)
        return cls(origin_xy=centre - half_cells * cell, cell_m=cell, z=np.full(shape, np.nan),
                   sigma=np.full(shape, np.nan), known=np.zeros(shape, dtype=bool),
                   source=np.zeros(shape, dtype=np.int8), stamp=np.full(shape, -1, dtype=np.int32))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.z.shape)

    @property
    def measured(self) -> np.ndarray:
        """bool (ny, nx): measured cells (source not none and not a fill)."""
        return (self.source != 0) & ~np.isin(self.source, _FILL_CODES)

    def _inside(self, iy, ix) -> bool:
        ny, nx = self.z.shape
        return 0 <= iy < ny and 0 <= ix < nx

    def index(self, xy) -> tuple[int, int] | None:
        try:
            x, y = _vector(xy, 2, "xy")
        except ValueError:
            return None
        ix = int(math.floor((x - self.origin_xy[0]) / self.cell_m + 0.5))
        iy = int(math.floor((y - self.origin_xy[1]) / self.cell_m + 0.5))
        return (iy, ix) if self._inside(iy, ix) else None

    def cell_xy(self, ij) -> np.ndarray:
        iy, ix = int(ij[0]), int(ij[1])
        return np.array([self.origin_xy[0] + ix * self.cell_m, self.origin_xy[1] + iy * self.cell_m])

    def _write(self, iy, ix, z, sigma, code, step):
        self.z[iy, ix], self.sigma[iy, ix] = z, sigma
        self.source[iy, ix], self.stamp[iy, ix], self.known[iy, ix] = code, step, True

    def stamp_ij(self, ij, z, sigma, source: str, step: int) -> bool:
        """Precedence by HF_RANK [C-33]: a higher rank replaces; a lower rank is ignored; equal-rank fills: the newer
        replaces; equal-rank measurements fuse by inverse variance unless |dz| > HF_DISAGREE_SIGMA_K * sqrt(s1^2 +
        s2^2), where the newer replaces (scene change). Returns True if the cell was written."""
        code = hf_code(source)
        if code == 0:
            raise ValueError("cannot stamp source 'none' (use vacate)")
        z, sigma, step = _finite(z, "z"), _nonnegative(sigma, "sigma"), _count(step, "step")
        iy, ix = int(ij[0]), int(ij[1])
        if not self._inside(iy, ix):
            return False
        old = HF_SOURCES[int(self.source[iy, ix])]
        new_rank, old_rank = HF_RANK[source], HF_RANK[old]
        if old == "none" or new_rank > old_rank or (new_rank == old_rank and source in HF_FILLS):
            self._write(iy, ix, z, sigma, code, step)
            return True
        if new_rank < old_rank:
            return False
        z0, s0 = float(self.z[iy, ix]), float(self.sigma[iy, ix])
        if abs(z - z0) > HF_DISAGREE_SIGMA_K * math.sqrt(s0 * s0 + sigma * sigma):
            self._write(iy, ix, z, sigma, code, step)
            return True
        v0, v1 = s0 * s0, sigma * sigma
        if v1 == 0.0:
            fused_z, fused_s = z, 0.0
        elif v0 == 0.0:
            fused_z, fused_s = z0, 0.0
        else:
            w0, w1 = 1.0 / v0, 1.0 / v1
            fused_z, fused_s = (w0 * z0 + w1 * z) / (w0 + w1), math.sqrt(1.0 / (w0 + w1))
        self._write(iy, ix, fused_z, fused_s, code, step)
        return True

    def stamp_xy(self, xy, z, sigma, source: str, step: int) -> bool:
        ij = self.index(xy)
        return False if ij is None else self.stamp_ij(ij, z, sigma, source, step)

    def stamp_samples(self, samples: Iterable["DepthSample"], step: int) -> int:
        changed = 0
        for sample in samples:
            changed += bool(self.stamp_xy(sample.xyz[:2], sample.xyz[2], sample.sigma_z, sample.source, step))
        return changed

    def vacate(self, ij_cells, step: int) -> int:
        """[C-36] cells a lifted object left: source "none", known False; returns the number of cells cleared."""
        step = _count(step, "step")
        cleared = 0
        for iy, ix in np.asarray(ij_cells, dtype=int).reshape(-1, 2):
            if not self._inside(iy, ix):
                continue
            cleared += int(self.source[iy, ix] != 0)
            self.z[iy, ix] = self.sigma[iy, ix] = np.nan
            self.source[iy, ix], self.known[iy, ix], self.stamp[iy, ix] = 0, False, step
        return cleared

    def cells_in_polygon(self, poly_xy) -> np.ndarray:
        """(n, 2) ij of the cells whose centres lie inside the polygon."""
        poly = _points(poly_xy)
        if len(poly) < 3:
            return np.zeros((0, 2), dtype=int)
        ny, nx = self.z.shape
        low = np.floor((poly.min(axis=0) - self.origin_xy) / self.cell_m).astype(int)
        high = np.ceil((poly.max(axis=0) - self.origin_xy) / self.cell_m).astype(int)
        ix0, ix1 = max(int(low[0]), 0), min(int(high[0]), nx - 1)
        iy0, iy1 = max(int(low[1]), 0), min(int(high[1]), ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            return np.zeros((0, 2), dtype=int)
        gy, gx = np.mgrid[iy0:iy1 + 1, ix0:ix1 + 1]
        ij = np.column_stack([gy.ravel(), gx.ravel()])
        centres = self.origin_xy + ij[:, ::-1] * self.cell_m
        return ij[poly_contains(poly, centres)]

    def max_in_polygon(self, poly_xy, *, exclude=None) -> tuple[float | None, float | None, float, float]:
        """(z_max over known cells, sigma at that cell, known fraction, measured fraction) of the cells inside the
        polygon, skipping the (n, 2) ij cells of ``exclude``. Empty polygon -> (None, None, 0.0, 0.0)."""
        cells = self.cells_in_polygon(poly_xy)
        if exclude is not None and len(cells):
            skip = {(int(a), int(b)) for a, b in np.asarray(exclude, dtype=int).reshape(-1, 2)}
            cells = np.array([c for c in cells if (int(c[0]), int(c[1])) not in skip], dtype=int).reshape(-1, 2)
        if not len(cells):
            return None, None, 0.0, 0.0
        iy, ix = cells[:, 0], cells[:, 1]
        known = self.known[iy, ix]
        measured = self.measured[iy, ix]
        known_fraction, measured_fraction = float(known.mean()), float(measured.mean())
        if not known.any():
            return None, None, known_fraction, measured_fraction
        heights = np.where(known, self.z[iy, ix], -np.inf)
        best = int(np.argmax(heights))
        return float(self.z[iy[best], ix[best]]), float(self.sigma[iy[best], ix[best]]), known_fraction, \
            measured_fraction

    def sample(self, xy) -> tuple[float | None, float | None, bool]:
        ij = self.index(xy)
        if ij is None or not self.known[ij]:
            return None, None, False
        return float(self.z[ij]), float(self.sigma[ij]), True

    def summary(self) -> dict:
        counts = {name: int(np.count_nonzero(self.source == code)) for code, name in enumerate(HF_SOURCES)}
        known = self.z[self.known]
        ny, nx = self.z.shape
        far = self.cell_xy((ny - 1, nx - 1))
        return jsonable({"shape": [ny, nx], "cell_m": self.cell_m, "origin_xy": self.origin_xy,
                         "extent_xy": [list(self.origin_xy), list(far)], "counts": counts,
                         "known": int(np.count_nonzero(self.known)), "measured": int(np.count_nonzero(self.measured)),
                         "z_range": [float(known.min()), float(known.max())] if known.size else None})


# ------------------------------------------------------------------------------------------ measurement records
@dataclass(frozen=True)
class DepthSample:
    """One world depth sample (geometry -> scene)."""
    xyz: tuple[float, float, float]
    sigma_z: float
    sigma_xy: float
    source: str                      # DEPTH_SOURCES
    uv: tuple[float, float] | None
    feature_id: int | None
    step: int
    kappa: float | None

    def __post_init__(self):
        object.__setattr__(self, "xyz", _vector_tuple(self.xyz, 3, "DepthSample.xyz"))
        object.__setattr__(self, "sigma_z", _nonnegative(self.sigma_z, "DepthSample.sigma_z"))
        object.__setattr__(self, "sigma_xy", _nonnegative(self.sigma_xy, "DepthSample.sigma_xy"))
        _choice(self.source, DEPTH_SOURCES, "DepthSample.source")
        object.__setattr__(self, "uv", _optional_uv(self.uv, "DepthSample.uv"))
        _count(self.step, "DepthSample.step")

    def to_dict(self) -> dict:
        return jsonable({"xyz": self.xyz, "sigma_z": self.sigma_z, "sigma_xy": self.sigma_xy, "source": self.source,
                         "uv": self.uv, "feature_id": self.feature_id, "step": self.step, "kappa": self.kappa})


@dataclass(frozen=True)
class FeatureRecord:
    """One tracked wrist feature of a survey; key = pid, "relief:<pid>:<k>", "dense:<k>" (the probe tracker's dense
    lattice around the pointed seed, decision §2.3 G-1) or None (auto grid)."""
    fid: int
    key: str | None
    uv0: tuple
    uv_end: tuple | None
    samples: int
    kappa: float | None
    sigma_kappa: float | None
    z: float | None
    sigma_z: float | None
    xyz: tuple | None
    valid: bool
    reject: str | None               # FEATURE_REJECTS or None

    def __post_init__(self):
        if self.reject is not None:
            _choice(self.reject, FEATURE_REJECTS, "FeatureRecord.reject")

    def is_dense_lattice(self) -> bool:
        """True for a dense-lattice feature (key "dense:<k>"): the samples S-5 (DSW) may withdraw off the part."""
        return isinstance(self.key, str) and self.key.startswith(DENSE_FEATURE_KEY_PREFIX)

    def to_dict(self) -> dict:
        return jsonable(self.__dict__)


@dataclass(eq=False)
class SurveyResult:
    """geometry.SurveyRecorder.result(camera); ``frames`` are uint8 upright images kept for later use and never
    serialized: "open" (open hold), "closed" (end of close 2, same pose), "static[k]", "ext_start"."""
    pose0: tuple
    pose_end: tuple
    stages: dict[str, int]
    features: list[FeatureRecord]
    samples: list[DepthSample]       # use camera.f_used (empty when the camera is None)
    loom: dict | None                # {"z_by_fid", "kappa_by_fid", "f_hat": Meas | None} [C-35]
    external_samples: list[dict]     # {"stage","world_displacement_m","image_shift_px","mask_extreme_uv","pixels"}
    frames: dict

    def summary(self) -> dict:
        rejects: dict[str, int] = {}
        for feature in self.features:
            if feature.reject:
                rejects[feature.reject] = rejects.get(feature.reject, 0) + 1
        z_values = [s.xyz[2] for s in self.samples]
        kappas = [f.kappa for f in self.features if f.valid and f.kappa is not None]
        f_hat = (self.loom or {}).get("f_hat")
        return jsonable({"pose0": self.pose0, "pose_end": self.pose_end, "stages": self.stages,
                         "features": len(self.features), "valid_features": sum(bool(f.valid) for f in self.features),
                         "samples": len(self.samples), "rejects_by_reason": rejects,
                         "median_sample_z": float(np.median(z_values)) if z_values else None,
                         "median_kappa": float(np.median(kappas)) if kappas else None,
                         "f_hat": f_hat.to_dict() if isinstance(f_hat, Meas) else f_hat,
                         "external_samples": len(self.external_samples), "frames": sorted(self.frames)})


@dataclass(frozen=True)
class CalibrationRecord:
    """primitives.CalibratePlus.record()."""
    gripper: GripperCalibration      # built as in v1 Calibrate._conclude (contact_skills.py:538-553)
    close_curves_mm: tuple[tuple[float, ...], ...]
    k15_mm: tuple[float, ...]
    yaw_probe: dict                  # {"commanded_rad", "realized_rad_per_step": [...], "gap_rate_mm_per_step": [...]}
    open_hold: dict                  # {"gaps_mm": [...], "inward_steps_mm": [...]}
    probe_moves: tuple[dict, ...]    # {"stage", "commanded_m", "reason", "world_displacement_m",
                                     #  "start_state8", "end_state8"}
    vertical: tuple[dict, ...]
    stage_outcomes: tuple[dict, ...]
    steps: int

    def to_dict(self) -> dict:
        return jsonable({"gripper": self.gripper.to_dict(), "close_curves_mm": self.close_curves_mm,
                         "k15_mm": self.k15_mm, "yaw_probe": self.yaw_probe, "open_hold": self.open_hold,
                         "probe_moves": self.probe_moves, "vertical": self.vertical,
                         "stage_outcomes": self.stage_outcomes, "steps": self.steps})


@dataclass(eq=False)
class V2Calibration:
    """Built by relations.calibrate_plus; immutable after the build."""
    gripper: GripperCalibration
    sigma_close_mm: float            # G4
    sigma_g_mm_per_step: float       # §4.5 item 1, K1w
    free_yaw_rate_rad_per_step: float
    camera: "CameraModel"
    external: "ExternalModel | None"
    locator: "HandLocator | None"
    projector: "ExternalProjector | None"
    fingertip_band: np.ndarray       # bool mask, upright wrist
    false_change: dict               # camera -> float array (per-pixel false-change probability), §4.5 item 5
    sigma_px: dict                   # camera -> float (static pixel noise, grey)
    sigma_static_grey: float
    quarter_turns: int               # k at the plan-time pose
    start_pose: tuple                # state8 at the end of Calibrate+ (the plan-time view)
    start_quaternion: tuple          # orientation of the episode's FIRST observation (R12 yaw reference)
    survey: SurveyResult
    mount: dict                      # _robot.mount_decision(...).to_dict()
    thresholds: list[Threshold]      # G1-G6, D3, D14, L1, camera-model decision

    def summary(self) -> dict:
        band = np.asarray(self.fingertip_band, dtype=bool) if self.fingertip_band is not None else None
        false_change = {camera: {"mean": float(np.mean(rate)), "max": float(np.max(rate))}
                        for camera, rate in (self.false_change or {}).items()
                        if rate is not None and np.size(rate)}
        return jsonable({"gripper": self.gripper.to_dict(), "sigma_close_mm": self.sigma_close_mm,
                         "sigma_g_mm_per_step": self.sigma_g_mm_per_step,
                         "free_yaw_rate_rad_per_step": self.free_yaw_rate_rad_per_step,
                         "camera": _summarize(self.camera), "external": _summarize(self.external),
                         "locator": _summarize(self.locator), "projector": _summarize(self.projector),
                         "fingertip_band_px": None if band is None else int(np.count_nonzero(band)),
                         "false_change": false_change, "sigma_px": self.sigma_px,
                         "sigma_static_grey": self.sigma_static_grey, "quarter_turns": self.quarter_turns,
                         "start_pose": self.start_pose, "start_quaternion": self.start_quaternion,
                         "survey": self.survey.summary() if self.survey is not None else None, "mount": self.mount,
                         "thresholds": thresholds_dicts(self.thresholds)})


# ------------------------------------------------------------------------------------------------ scene records
def _required_keys(value, keys, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dict with the keys {tuple(keys)}")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{name} misses the keys {missing}")
    return dict(value)


def _top_band_dict(value):
    """PartRecord.top_band (S-1..S-3): {"n": member count, "stat_m": statistical sigma, "model_m": part-level model
    term}, both sigmas finite >= 0 in metres; extra keys kept; None passes."""
    if value is None:
        return None
    out = _required_keys(value, TOP_BAND_KEYS, "PartRecord.top_band")
    out["n"] = _count(value["n"], "PartRecord.top_band n")
    out["stat_m"] = _nonnegative(value["stat_m"], "PartRecord.top_band stat_m")
    out["model_m"] = _nonnegative(value["model_m"], "PartRecord.top_band model_m")
    return out


def _unit_axis(value, name):
    if value is None:
        return None
    axis = _vector(value, 2, name)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(f"{name} must be a nonzero world-xy axis")
    return axis / norm


def axis_angle_rad(a_xy, b_xy) -> float:
    """Angle between two undirected world-xy axes (nonzero 2-vectors, any length or sign), mod pi: in [0, pi/2].
    The D10 / S-12 agreement measure (compared against AXIS_AGREE_MAX_RAD)."""
    a = _unit_axis(a_xy, "axis_angle_rad a_xy")
    b = _unit_axis(b_xy, "axis_angle_rad b_xy")
    return float(math.acos(min(1.0, abs(float(a @ b)))))


def _axis_angle_field(value, name):
    if value is None:
        return None
    angle = _nonnegative(value, name)
    if angle > math.pi / 2.0 + 1e-9:
        raise ValueError(f"{name} is an axis angle mod pi: it lies in [0, pi/2]")
    return angle


def _axis_views_dict(value):
    """axis_check ``views`` (S-12): {"axes": the accepted world-xy axes of the part's view memory (nonzero, stored as
    unit vectors; may be empty), "disagree_rad": the largest angle between this registration's outline axis and a
    remembered axis (None when nothing was compared)}; extra keys kept (e.g. the memory's conflicted mark)."""
    out = _required_keys(value, AXIS_VIEWS_KEYS, "PartRecord.axis_check views")
    axes = value["axes"]
    if isinstance(axes, (str, bytes, Mapping)) or not isinstance(axes, (Iterable, np.ndarray)):
        raise ValueError("PartRecord.axis_check views axes must be a sequence of world-xy axes")
    out["axes"] = tuple(_unit_axis(_vector(axis, 2, "PartRecord.axis_check views axis"),
                                   "PartRecord.axis_check views axis") for axis in axes)
    out["disagree_rad"] = _axis_angle_field(value["disagree_rad"], "PartRecord.axis_check views disagree_rad")
    return out


def _axis_check_dict(value):
    """PartRecord.axis_check: both axes None or nonzero world-xy vectors (stored as unit vectors), agree_rad None or an
    axis angle in [0, pi/2], accepted a bool, source a nonempty string. Extra keys kept; None passes.

    An accepted check is never sourced AXIS_INCONSISTENT or NOT_ELONGATED; unless it carries the remembered axis
    (reason AXIS_KEPT_EARLIER) it has both axes and agree_rad <= AXIS_AGREE_MAX_RAD (D10-OE: the outline axis and an
    agreeing gated edge axis; S-4': the tensor and region-cell axes), no collapsed region, and its outline axis agrees
    within AXIS_AGREE_MAX_RAD with every axis of its ``views``.

    The documented extra keys (design review 2 decision §2.1 K-4), validated when present:

    - ``method``: None (a pre-D10-OE check) or one of AXIS_CHECK_METHODS ("outline_edges");
    - ``region_collapsed``: None (not evaluated) or a bool, S-13 (the band hull exceeds REGION_COLLAPSE_RATIO x the
      region): the record's own axis is refused (the remembered one may be carried, reason AXIS_KEPT_EARLIER);
    - ``views``: None or {"axes", "disagree_rad"} (``_axis_views_dict``), the S-12 view memory;
    - ``reason``: None or a nonempty string; one of AXIS_CHECK_REASONS when ``method`` is set. A refusal reason
      ("outline_gate", "no_edge_agreement", "views_disagree") marks a refused axis sourced as AXIS_REFUSAL_SOURCES
      says; "kept_earlier" marks an accepted check that carries the remembered axis. "views_disagree" and
      "kept_earlier" need a view memory with at least one axis."""
    if value is None:
        return None
    out = _required_keys(value, AXIS_CHECK_KEYS, "PartRecord.axis_check")
    out["image_axis"] = _unit_axis(value["image_axis"], "PartRecord.axis_check image_axis")
    out["region_axis"] = _unit_axis(value["region_axis"], "PartRecord.axis_check region_axis")
    agree = _axis_angle_field(value["agree_rad"], "PartRecord.axis_check agree_rad")
    out["agree_rad"] = agree
    if not isinstance(value["accepted"], (bool, np.bool_)):
        raise ValueError("PartRecord.axis_check accepted must be a bool")
    out["accepted"] = accepted = bool(value["accepted"])
    source = value["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("PartRecord.axis_check source must be a nonempty string")
    method = out.get("method")
    if method is not None:
        _choice(method, AXIS_CHECK_METHODS, "PartRecord.axis_check method")
    collapsed = out.get("region_collapsed")
    if collapsed is not None:
        if not isinstance(collapsed, (bool, np.bool_)):
            raise ValueError("PartRecord.axis_check region_collapsed must be None or a bool")
        out["region_collapsed"] = collapsed = bool(collapsed)
    if out.get("views") is not None:
        out["views"] = _axis_views_dict(out["views"])
    views = out.get("views")
    reason = out.get("reason")
    if reason is not None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("PartRecord.axis_check reason must be None or a nonempty string")
        if method is not None:
            _choice(reason, AXIS_CHECK_REASONS, "PartRecord.axis_check reason")
    kept = reason == AXIS_KEPT_EARLIER
    if reason in AXIS_REFUSAL_SOURCES:
        if accepted:
            raise ValueError(f"PartRecord.axis_check reason {reason!r} marks a refused axis")
        if source != AXIS_REFUSAL_SOURCES[reason]:
            raise ValueError(f"a PartRecord.axis_check refused with reason {reason!r} is sourced "
                             f"{AXIS_REFUSAL_SOURCES[reason]!r}, got {source!r}")
    if reason in ("views_disagree", AXIS_KEPT_EARLIER) and (views is None or not views["axes"]):
        raise ValueError(f"PartRecord.axis_check reason {reason!r} needs views with a remembered axis (S-12)")
    if kept and not accepted:
        raise ValueError("PartRecord.axis_check reason 'kept_earlier' carries the remembered axis: accepted is True")
    if accepted:
        if source in (AXIS_INCONSISTENT, NOT_ELONGATED):
            raise ValueError(f"PartRecord.axis_check source {source!r} marks a refused axis")
        if not kept:
            if out["image_axis"] is None or out["region_axis"] is None or agree is None:
                raise ValueError("an accepted PartRecord.axis_check needs both axes and agree_rad")
            if agree > AXIS_AGREE_MAX_RAD + 1e-9:
                raise ValueError("an accepted PartRecord.axis_check needs agree_rad <= AXIS_AGREE_MAX_RAD")
            if collapsed:
                raise ValueError("a collapsed region's axis is refused (S-13); only a remembered axis is carried")
            if views is not None and any(axis_angle_rad(out["image_axis"], axis) > AXIS_AGREE_MAX_RAD + 1e-9
                                         for axis in views["axes"]):
                raise ValueError("an accepted PartRecord.axis_check disagrees with its view memory: S-12 refuses it "
                                 "(views_disagree)")
    return out


@dataclass(eq=False)
class PartRecord:
    """Spec §4.3 part record; measured, never derived from names. Fields after ``roles`` default to the
    "seeded, nothing measured" state. Invariants checked at construction (so the scene builds registered and
    provisional records through the constructor or ``dataclasses.replace``, never by mutating a seeded record):

    - a record with a 3-D point (``xyz``) carries ``sigma_h_m``, a finite height sigma >= 0 in metres, provisional
      records included: the planner's formulas (P6, P11, keep-outs) read it. Only a record without ``xyz`` (seeded,
      nothing measured) may hold None;
    - a "registered" or provisional record has ``xyz`` and a ``registration_step`` (policy step >= 0);
    - ``elongation`` is stored only when D10 passes (design review 2 S-11, D10-OE: the outline gate AND an agreeing
      gated edge axis, or S-12's remembered axis; before it, design review 1 S-4': structure-tensor coherence AND
      region aspect AND the axis consensus; all owned by the scene). A failing (axis, coherence) goes in the
      registration event and evidence only, never in this field: ``is_elongated()`` (Align across_bar, wedge
      capability) trusts its presence. The axis is a unit world-xy vector and the coherence (D10-OE: the outline
      share) lies in [0, 1];
    - ``width_by_yaw``: None = not measured (a seeded record); ``()`` = measured as unmeasurable. Design review 2 K-5:
      a registered, non-provisional record carries ``()`` when its region collapsed (S-13) and the view memory holds
      no earlier width set: ``width_at`` returns None for every yaw and the planner answers geometry_unmeasured (P2,
      P-I).

    Design review 1 fields (decision §2.1 K-1; None = not measured / not applicable):

    - ``top_bound`` ("lower" | None) and ``top_upper_z`` (m): rulings R2. A hidden or cut top is only a lower bound;
      ``top_upper_z`` = max(start fingertip height, highest measured surface) is the height every clearance consumer
      (free_path, transit_path, P7, P12, hover_for) raises the part's cells to, and contact planning on such a part
      refines the height first (P-F). The two are set and cleared together, and ``top_upper_z`` >= ``top_z.value``;
    - ``top_band`` ({"n", "stat_m", "model_m"}): S-1..S-3, the band statistics behind ``top_z`` (member count, the
      statistical sigma and the part-level model term, metres). It needs ``top_z``, and ``sigma_h_m`` >=
      hypot(stat_m, model_m): the model term enters the planner's height sigma (and only that, the scene's rule);
    - ``axis_check`` ({"image_axis", "region_axis", "agree_rad", "accepted", "source"}): the D10 axis evidence
      (world xy axes stored as unit vectors; None when not measured), their angle mod pi in [0, pi/2], whether the D10
      axis was accepted and the source (``AXIS_INCONSISTENT`` or ``NOT_ELONGATED`` for a refusal). Design review 2
      (K-4, D10-OE, ``method`` "outline_edges"): image_axis = the outline axis, region_axis = the agreeing (or
      best) gated edge axis, agree_rad = their angle; the documented extra keys ``method``, ``region_collapsed``
      (S-13), ``views`` (S-12: the accepted axes remembered for the part and this view's disagreement, radians) and
      ``reason`` ("outline_gate" / "no_edge_agreement" / "views_disagree" refuse, "kept_earlier" carries the
      remembered axis) are validated by ``_axis_check_dict``. Design review 1 (S-4'): the image or mask tensor axis
      and the region-cell PCA axis. An accepted check has both axes and agree_rad <= ``AXIS_AGREE_MAX_RAD`` (unless
      it carries the remembered axis); a refused check leaves ``elongation`` None; an accepted check with ``views``
      has an elongation within ``AXIS_AGREE_MAX_RAD`` of every remembered axis. Other extra keys are kept as
      given."""
    pid: str
    text: str
    move_prior: str
    roles: frozenset[str]
    seeds: list[Seed] = field(default_factory=list)
    xyz: np.ndarray | None = None              # world point of the grasp / seed point (m)
    cov: np.ndarray | None = None              # 3x3 m^2
    track: "NCCTrack | None" = None            # v1 wrist_servo.NCCTrack at the grasp point
    track_pose: tuple | None = None            # state8 at template capture
    top_z: Meas | None = None
    support_z: Meas | None = None
    region_xy: np.ndarray | None = None        # (k,2) world polygon of the connected region (D8 / D9)
    region_cells: np.ndarray | None = None     # (n,2) HeightField ij
    width_by_yaw: tuple[tuple[float, Meas], ...] | None = None   # jaw yaw (rad, world, mod pi) -> width (m);
    # () on a registered record = width unmeasured (S-13 collapsed region, K-5)
    elongation: tuple[np.ndarray, float] | None = None           # (unit world-horizontal axis (2,), coherence);
    # set ONLY when D10 passes (D10-OE / S-12); a failing result goes in the registration event / evidence
    circle: tuple[np.ndarray, float, float] | None = None        # (centre xy (2,), radius m, rms px)   D11
    cavity: dict | None = None                 # reserved
    mount_side: np.ndarray | None = None       # reserved
    dof: dict | None = None                    # written by turn
    rigid_patch: dict | None = None            # {"uv","track","pose","dist_m"} or {"unmeasurable": reason}
    external: dict | None = None               # {"uv","sigma_px","polygon_uv","template"} (projected)
    registered_empty_hand: bool = False
    view_pose: tuple = ()
    registration_step: int | None = None       # policy step of the register call; required once registered
    provisional: bool = False                  # [C-37] start-view centroid of a grip-only part (hover centring only)
    partial: bool = False                      # [C-38] region cut by the image border; sigma inflated
    status: str = "seeded"                     # PART_STATUSES
    reason: str | None = None
    sigma_h_m: float | None = None             # representative height sigma for planning; required with xyz
    top_bound: str | None = None               # TOP_BOUNDS: "lower" when the top is hidden or cut (rulings R2)
    top_upper_z: float | None = None           # upper bound of a lower-bound top (m); set with top_bound
    top_band: dict | None = None               # {"n", "stat_m", "model_m"} band statistics of top_z (S-1..S-3)
    axis_check: dict | None = None             # AXIS_CHECK_KEYS (+ AXIS_CHECK_EXTRA_KEYS) D10 axis evidence

    def __post_init__(self):
        _choice(self.pid, PIDS, "PartRecord.pid")
        _choice(self.move_prior, MOVE_PRIORS, "PartRecord.move_prior")
        self.roles = frozenset(self.roles)
        unknown = self.roles - set(ROLES)
        if unknown:
            raise ValueError(f"PartRecord.roles has unknown roles {sorted(unknown)}")
        _choice(self.status, PART_STATUSES, "PartRecord.status")
        if self.sigma_h_m is not None:
            self.sigma_h_m = _nonnegative(self.sigma_h_m, "PartRecord.sigma_h_m")
        if self.registration_step is not None:
            self.registration_step = _count(self.registration_step, "PartRecord.registration_step")
        if self.xyz is not None and self.sigma_h_m is None:
            raise ValueError("PartRecord with a 3-D point (xyz) needs sigma_h_m (the planner reads it)")
        if (self.status == "registered" or self.provisional) and (self.xyz is None or self.registration_step is None):
            raise ValueError("a registered or provisional PartRecord needs xyz and registration_step")
        if self.elongation is not None:
            if not isinstance(self.elongation, (tuple, list)) or len(self.elongation) != 2:
                raise ValueError("PartRecord.elongation must be (unit axis (2,), coherence)")
            axis = _vector(self.elongation[0], 2, "PartRecord.elongation axis")
            coherence = _finite(self.elongation[1], "PartRecord.elongation coherence")
            if abs(float(np.linalg.norm(axis)) - 1.0) > 1e-6 or not 0.0 <= coherence <= 1.0:
                raise ValueError("PartRecord.elongation needs a unit world-xy axis and a coherence in [0, 1]")
            self.elongation = (axis, coherence)
        self._check_top_bound()
        self.top_band = _top_band_dict(self.top_band)
        if self.top_band is not None:
            if self.top_z is None:
                raise ValueError("PartRecord.top_band describes top_z: a record with a top band needs top_z")
            floor = math.hypot(self.top_band["stat_m"], self.top_band["model_m"])
            if self.sigma_h_m is not None and self.sigma_h_m < floor - 1e-9:
                raise ValueError("PartRecord.sigma_h_m must hold the band's statistical and model terms: "
                                 f"sigma_h_m >= hypot(stat_m, model_m) = {floor:.6f} m")
        self.axis_check = _axis_check_dict(self.axis_check)
        if self.axis_check is not None and not self.axis_check["accepted"] and self.elongation is not None:
            raise ValueError("PartRecord.elongation must be None when axis_check refused the axis (D10)")
        views = None if self.axis_check is None else self.axis_check.get("views")
        if self.elongation is not None and views is not None and any(
                axis_angle_rad(self.elongation[0], axis) > AXIS_AGREE_MAX_RAD + 1e-9 for axis in views["axes"]):
            raise ValueError("PartRecord.elongation must agree within AXIS_AGREE_MAX_RAD with every remembered axis "
                             "of axis_check views (S-12)")

    def _check_top_bound(self):
        if self.top_bound is not None:
            _choice(self.top_bound, TOP_BOUNDS, "PartRecord.top_bound")
        if self.top_upper_z is not None:
            self.top_upper_z = _finite(self.top_upper_z, "PartRecord.top_upper_z")
        if (self.top_bound is None) != (self.top_upper_z is None):
            raise ValueError("PartRecord.top_bound 'lower' and top_upper_z are set and cleared together (rulings R2)")
        if self.top_upper_z is not None and isinstance(self.top_z, Meas) and self.top_upper_z < self.top_z.value:
            raise ValueError("PartRecord.top_upper_z must be >= top_z.value (the measured lower bound)")

    def top_is_lower_bound(self) -> bool:
        """True when the top is only a lower bound (rulings R2): clearance consumers raise the part's cells to
        ``top_upper_z`` and contact planning refines the height first (P-F)."""
        return self.top_bound == "lower"

    def width_at(self, yaw_rad) -> Meas | None:
        """Width at the nearest grid jaw yaw (circular distance mod pi); None when no width was measured."""
        if not self.width_by_yaw:
            return None
        yaw = _finite(yaw_rad, "yaw_rad") % math.pi

        def gap(item):
            d = abs((float(item[0]) % math.pi) - yaw)
            return min(d, math.pi - d)
        return min(self.width_by_yaw, key=gap)[1]

    def is_elongated(self) -> bool:
        """True when an elongation is stored. The kernel cannot evaluate D10 (its thresholds and the aspect belong to
        the scene), so the field's presence IS the D10 verdict: the scene stores an elongation only when D10
        passes (see the class docstring)."""
        if self.elongation is None:
            return False
        axis, coherence = self.elongation
        return bool(np.isfinite(np.asarray(axis, dtype=float)).all() and math.isfinite(float(coherence)))

    def to_dict(self) -> dict:
        def meas(value):
            return None if value is None else value.to_dict()
        return jsonable({
            "pid": self.pid, "text": self.text, "move_prior": self.move_prior, "roles": sorted(self.roles),
            "seeds": [s.to_dict() for s in self.seeds], "xyz": self.xyz, "cov": self.cov,
            "track": _summarize(self.track), "track_pose": self.track_pose, "top_z": meas(self.top_z),
            "support_z": meas(self.support_z), "region_xy": self.region_xy,
            "region_cells": None if self.region_cells is None else int(len(self.region_cells)),
            "width_by_yaw": None if self.width_by_yaw is None else [[yaw, w.to_dict()] for yaw, w in self.width_by_yaw],
            "elongation": None if self.elongation is None else {"axis": self.elongation[0],
                                                                "coherence": self.elongation[1]},
            "circle": None if self.circle is None else {"centre": self.circle[0], "radius_m": self.circle[1],
                                                        "rms_px": self.circle[2]},
            "cavity": self.cavity, "mount_side": self.mount_side, "dof": self.dof,
            "rigid_patch": None if self.rigid_patch is None else {k: (_summarize(v) if k == "track" else v)
                                                                  for k, v in self.rigid_patch.items()},
            "external": None if self.external is None else {k: v for k, v in self.external.items() if k != "template"},
            "registered_empty_hand": self.registered_empty_hand, "view_pose": self.view_pose,
            "registration_step": self.registration_step, "provisional": self.provisional, "partial": self.partial,
            "status": self.status, "reason": self.reason, "sigma_h_m": self.sigma_h_m, "top_bound": self.top_bound,
            "top_upper_z": self.top_upper_z, "top_band": self.top_band, "axis_check": self.axis_check})


@dataclass(frozen=True)
class RegistrationFailure:
    pid: str
    reason: str                      # REASONS key
    evidence: dict
    thresholds: tuple[Threshold, ...]

    def __post_init__(self):
        _choice(self.reason, REASONS, "RegistrationFailure.reason")

    def to_dict(self) -> dict:
        return jsonable({"pid": self.pid, "reason": self.reason, "evidence": self.evidence,
                         "thresholds": thresholds_dicts(self.thresholds)})


@dataclass(frozen=True, eq=False)
class KeepOut:
    """O5 volume (built by planner.keepout_for)."""
    pid: str
    kind: str                        # KEEPOUT_KINDS
    polygon_xy: np.ndarray
    z_low: float
    z_high: float
    thresholds: tuple[Threshold, ...]

    def __post_init__(self):
        _choice(self.kind, KEEPOUT_KINDS, "KeepOut.kind")
        object.__setattr__(self, "z_low", _finite(self.z_low, "KeepOut.z_low"))
        object.__setattr__(self, "z_high", _finite(self.z_high, "KeepOut.z_high"))
        if self.z_high < self.z_low:
            raise ValueError("KeepOut.z_high must be >= z_low")

    def to_dict(self) -> dict:
        return jsonable({"pid": self.pid, "kind": self.kind, "polygon_xy": self.polygon_xy, "z_low": self.z_low,
                         "z_high": self.z_high, "thresholds": thresholds_dicts(self.thresholds)})


@dataclass(eq=False)
class HeldFrame:
    """Spec §4.3 / §4.7 in-hand pose of the held object."""
    pid: str
    grip_pid: str
    grasp_pose: tuple                # state8 at hold_pass
    offset_h: np.ndarray             # footprint centre in the hand frame (R0, R1, R2 coordinates, m)
    vacated_cells: np.ndarray | None  # [C-36] HF ij cells the object (and its D7b fills) occupied at pick time
    sigma_offset_m: float
    b_low: Meas                      # grip z minus the object's lowest point (m)
    r_obj_m: float
    extent_axes: np.ndarray | None   # (2,2) principal axes (world xy) and (2,) extents
    tilt_deg: Meas | None
    wrist_visible: bool
    silhouette: np.ndarray | None    # bool mask, upright wrist
    ext_template: "NCCTrack | None"
    gap_ref_mm: float
    sigma_carry_mm: float | None
    support_z_pick: Meas
    g_close: float
    touchdown: dict | None = None

    def object_ref_point(self, pose: HandPose) -> np.ndarray:
        """World footprint centre: grip + R @ offset_h."""
        return pose.grip + pose.R @ _vector(self.offset_h, 3, "offset_h")

    def lowest_point_z(self, pose: HandPose) -> float:
        return float(pose.grip[2]) - self.b_low.value

    def to_dict(self) -> dict:
        silhouette = None if self.silhouette is None else int(np.count_nonzero(self.silhouette))
        return jsonable({"pid": self.pid, "grip_pid": self.grip_pid, "grasp_pose": self.grasp_pose,
                         "offset_h": self.offset_h,
                         "vacated_cells": None if self.vacated_cells is None else int(len(self.vacated_cells)),
                         "sigma_offset_m": self.sigma_offset_m, "b_low": self.b_low.to_dict(),
                         "r_obj_m": self.r_obj_m, "extent_axes": self.extent_axes,
                         "tilt_deg": None if self.tilt_deg is None else self.tilt_deg.to_dict(),
                         "wrist_visible": self.wrist_visible, "silhouette_px": silhouette,
                         "ext_template": _summarize(self.ext_template), "gap_ref_mm": self.gap_ref_mm,
                         "sigma_carry_mm": self.sigma_carry_mm, "support_z_pick": self.support_z_pick.to_dict(),
                         "g_close": self.g_close, "touchdown": self.touchdown})


# ------------------------------------------------------------------------------- needs, answers, steps, outcomes
def _last_outcome(value):
    """[C-45] repair payload: exactly {"reason", "evidence_text", "thresholds": [[name, value], ...]} or None."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"reason", "evidence_text", "thresholds"}:
        raise ValueError("last_outcome must have exactly the keys reason, evidence_text, thresholds")
    _choice(value["reason"], REASONS, "last_outcome.reason")
    if not isinstance(value["evidence_text"], str) or len(value["evidence_text"]) > EVIDENCE_TEXT_MAX_CHARS:
        raise ValueError(f"last_outcome.evidence_text must be a string of <= {EVIDENCE_TEXT_MAX_CHARS} chars")
    for pair in value["thresholds"]:
        if len(pair) != 2 or not isinstance(pair[0], str):
            raise ValueError("last_outcome.thresholds must be [name, value] pairs")
        _finite(pair[1], "last_outcome threshold value")
    return value


@dataclass(frozen=True)
class PlanNeed:
    kind: str                        # "plan" | "replan"
    quarter_turns: int
    fingertip_edge: str              # FINGERTIP_EDGES (of the turned wrist image)
    status_block: dict | None = None
    reask_error: str | None = None
    holding: str | None = None

    def __post_init__(self):
        _choice(self.kind, PLAN_KINDS, "PlanNeed.kind")
        _quarter_turns(self.quarter_turns, "PlanNeed.quarter_turns")
        _choice(self.fingertip_edge, FINGERTIP_EDGES, "PlanNeed.fingertip_edge")
        if self.status_block is not None and not isinstance(self.status_block, Mapping):
            raise ValueError("PlanNeed.status_block must be a dict or None")
        if self.reask_error is not None and not isinstance(self.reask_error, str):
            raise ValueError("PlanNeed.reask_error must be a string or None")
        _optional_pid(self.holding, "PlanNeed.holding")


@dataclass(frozen=True)
class PointNeed:
    kind: str                        # "point_fine" | "point_coarse"
    pid: str
    text: str
    camera: str
    center_uv: tuple[float, float] | None    # fine: crop centre (upright px of ``camera``)
    quarter_turns: int               # wrist: alignment for the shown image; external: 0
    purpose: str                     # POINT_PURPOSES
    previous_rejected: str | None = None     # measured reason, <= 120 chars
    marker_uv: tuple | None = None   # reserved (LookAt predicted hit)

    def __post_init__(self):
        _choice(self.kind, POINT_KINDS, "PointNeed.kind")
        _choice(self.pid, PIDS, "PointNeed.pid")
        if not isinstance(self.text, str):
            raise ValueError("PointNeed.text must be a string")
        _choice(self.camera, CAMERAS, "PointNeed.camera")
        object.__setattr__(self, "center_uv", _optional_uv(self.center_uv, "PointNeed.center_uv"))
        if self.kind == "point_fine" and self.center_uv is None:
            raise ValueError("a point_fine need requires center_uv")
        _quarter_turns(self.quarter_turns, "PointNeed.quarter_turns")
        if self.camera == "external" and self.quarter_turns:
            raise ValueError("external pointing is never turned (quarter_turns 0)")
        _choice(self.purpose, POINT_PURPOSES, "PointNeed.purpose")
        if self.previous_rejected is not None and (not isinstance(self.previous_rejected, str)
                                                   or len(self.previous_rejected) > PREVIOUS_REJECTED_MAX_CHARS):
            raise ValueError(f"previous_rejected must be a string of <= {PREVIOUS_REJECTED_MAX_CHARS} chars")
        object.__setattr__(self, "marker_uv", _optional_uv(self.marker_uv, "PointNeed.marker_uv"))


@dataclass(frozen=True)
class RepairNeed:
    kind: str                        # "repair" | "review"
    quarter_turns: int
    status_block: dict
    holding: str | None = None
    last_outcome: dict | None = None  # exactly {"reason", "evidence_text", "thresholds": [[name, value]]} [C-45]

    def __post_init__(self):
        _choice(self.kind, REPAIR_KINDS, "RepairNeed.kind")
        _quarter_turns(self.quarter_turns, "RepairNeed.quarter_turns")
        if not isinstance(self.status_block, Mapping):
            raise ValueError("RepairNeed.status_block must be a dict")
        _optional_pid(self.holding, "RepairNeed.holding")
        _last_outcome(self.last_outcome)


@dataclass(frozen=True)
class CallRecord:
    kind: str                        # NEED_KINDS
    latency_s: float
    request_key: str | None
    request_log: str | None
    ok: bool
    error: str | None = None

    def __post_init__(self):
        _choice(self.kind, NEED_KINDS, "CallRecord.kind")
        object.__setattr__(self, "latency_s", _nonnegative(self.latency_s, "CallRecord.latency_s"))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "latency_s": self.latency_s, "request_key": self.request_key,
                "request_log": self.request_log, "ok": bool(self.ok), "error": self.error}


@dataclass(frozen=True)
class PointReply:
    """client.request_point return; uv mapped back to upright pixels of ``camera``."""
    stage: str                       # POINT_STAGES
    camera: str
    answer_id: int | None            # None = 0 (not visible) or schema null
    uv_upright: tuple[float, float] | None
    quantization_px: float
    origin: tuple[int, int] | None
    crop_px: int | None
    upsample: int | None
    quarter_turns: int

    def __post_init__(self):
        _choice(self.stage, POINT_STAGES, "PointReply.stage")
        _choice(self.camera, CAMERAS, "PointReply.camera")
        if self.answer_id is not None:
            _count(self.answer_id, "PointReply.answer_id")
        object.__setattr__(self, "uv_upright", _optional_uv(self.uv_upright, "PointReply.uv_upright"))
        object.__setattr__(self, "quantization_px", _nonnegative(self.quantization_px, "PointReply.quantization_px"))
        _quarter_turns(self.quarter_turns, "PointReply.quarter_turns")

    def to_dict(self) -> dict:
        return jsonable(self.__dict__)


@dataclass(frozen=True)
class PointAnswer:
    need: PointNeed
    reply: PointReply | None
    schema_error: bool
    call: CallRecord


@dataclass(frozen=True)
class PlanAnswer:
    need: PlanNeed
    raw: dict | None
    plan: Plan | None
    parse_error: str | None
    schema_error: bool
    call: CallRecord
    view: StepView                   # the view whose images were sent (merge_plan maps its cells)


@dataclass(frozen=True)
class RepairReply:
    need: RepairNeed
    raw: dict | None
    answer: RepairAnswer | None
    schema_error: bool
    call: CallRecord


@dataclass(frozen=True)
class StepContext:
    """What a controller receives for one plan step."""
    i: int
    do: str
    obj: str
    grip: tuple[str, ...]
    ref: str | None
    sense: str
    keep: tuple[Keep, ...]
    until: tuple[Until, ...]
    attempt: int
    resume: dict | None = None

    def __post_init__(self):
        _count(self.i, "StepContext.i")
        _choice(self.do, DO_KINDS, "StepContext.do")
        _choice(self.obj, PIDS, "StepContext.obj")
        object.__setattr__(self, "grip", tuple(_choice(p, PIDS, "StepContext.grip") for p in self.grip))
        _optional_pid(self.ref, "StepContext.ref")
        _choice(self.sense, SENSES, "StepContext.sense")
        object.__setattr__(self, "keep", tuple(self.keep))
        object.__setattr__(self, "until", tuple(self.until))
        _count(self.attempt, "StepContext.attempt")

    @classmethod
    def from_step(cls, i: int, step: PlanStep, *, attempt: int, resume: dict | None = None,
                  grip: tuple[str, ...] | None = None) -> "StepContext":
        """Context for plan step i (grip defaults to the step's grip list with () -> (obj,))."""
        return cls(i, step.do, step.obj, tuple(grip) if grip is not None else step.grip_parts, step.ref, step.sense,
                   step.keep, step.until, attempt, resume)


@dataclass
class StepState:
    """Spec §4.3 step state."""
    i: int
    step: PlanStep
    status: str                      # STEP_STATUSES
    reason: str | None
    evidence: dict
    calls: dict
    steps: int
    attempts: int
    outcomes: list[int] = field(default_factory=list)

    def __post_init__(self):
        _choice(self.status, STEP_STATUSES, "StepState.status")

    @classmethod
    def pending(cls, i: int, step: PlanStep) -> "StepState":
        return cls(i, step, "pending", None, {}, {}, 0, 0, [])

    def compact(self) -> dict:
        return jsonable({"i": self.i, "do": self.step.do, "obj": self.step.obj, "grip": list(self.step.grip),
                         "ref": self.step.ref, "status": self.status, "reason": self.reason, "calls": self.calls,
                         "steps": self.steps, "attempts": self.attempts, "outcomes": self.outcomes})


def thresholds_dicts(ths) -> list[dict]:
    """Threshold records (or their dicts) as JSON-safe dicts."""
    out = []
    for item in ths or ():
        out.append(jsonable(item.to_dict() if isinstance(item, Threshold) else dict(item)))
    return out


@dataclass
class RelationOutcome:
    """A relation's outcome (§C4.7). Checked at construction: ``relation`` in OUTCOME_RELATIONS; ``reason`` a
    REASONS key; a succeeded outcome carries a success reason of its own relation (SUCCESS_REASONS) and a failed or
    rejected one no success reason (EITHER_STATUS_REASONS excepted). Failure reasons may cross relations."""
    relation: str                    # OUTCOME_RELATIONS: a relation, a reserved do kind, "plan" or "keeps"
    status: str                      # OUTCOME_STATUSES
    reason: str                      # REASONS key
    pids: dict                       # {"obj": .., "grip": .., "ref": ..}
    evidence: dict                   # JSON-safe; metres / rad; gaps in mm
    thresholds: list[Threshold]
    evidence_text: str               # <= 200 chars (relations.evidence_text)
    local_retries: int
    steps: int
    calls: dict = field(default_factory=dict)
    resume: dict | None = None
    model_claims: list = field(default_factory=list)

    def __post_init__(self):
        _choice(self.relation, OUTCOME_RELATIONS, "RelationOutcome.relation")
        _choice(self.status, OUTCOME_STATUSES, "RelationOutcome.status")
        _choice(self.reason, REASONS, "RelationOutcome.reason")
        if self.status == "succeeded":
            _choice(self.reason, SUCCESS_REASONS.get(self.relation, ()),
                    f"RelationOutcome.reason of a succeeded {self.relation!r} outcome")
        elif self.reason not in EITHER_STATUS_REASONS and any(self.reason in codes
                                                              for codes in SUCCESS_REASONS.values()):
            raise ValueError(f"a {self.status} outcome cannot carry the success reason {self.reason!r}")
        if not isinstance(self.evidence_text, str) or len(self.evidence_text) > EVIDENCE_TEXT_MAX_CHARS:
            raise ValueError(f"evidence_text must be a string of <= {EVIDENCE_TEXT_MAX_CHARS} chars")
        if not all(isinstance(t, Threshold) for t in self.thresholds):
            raise ValueError("RelationOutcome.thresholds must hold Threshold records")
        _count(self.local_retries, "RelationOutcome.local_retries")
        _count(self.steps, "RelationOutcome.steps")

    def repair_payload(self) -> dict:
        """[C-45] the repair / review ``last_outcome``: exactly reason, evidence_text and [[name, value]] pairs."""
        return {"reason": self.reason, "evidence_text": self.evidence_text,
                "thresholds": [[t.name, t.value] for t in self.thresholds]}

    def record(self, n: int, policy_step: int, plan_step: int | None) -> dict:
        """Runner-facing record (§C18.5); ``skill`` = ``relation`` keeps M0's stuck alert and console working."""
        return jsonable({"n": n, "step": policy_step, "skill": self.relation, "relation": self.relation,
                         "status": self.status, "reason": self.reason, "evidence": self.evidence,
                         "thresholds": thresholds_dicts(self.thresholds), "evidence_text": self.evidence_text,
                         "pids": self.pids, "plan_step": plan_step, "local_retries": self.local_retries,
                         "steps": self.steps, "calls": self.calls, "model_claims": self.model_claims})


@dataclass(frozen=True)
class ObsReport:
    """Spec §4.3 observer report plus the observer name and the policy step."""
    name: str
    verdict: str                     # VERDICTS
    progress: float | None
    reason: str | None
    evidence: dict
    thresholds: tuple[Threshold, ...]
    step: int

    def __post_init__(self):
        _choice(self.verdict, VERDICTS, "ObsReport.verdict")
        object.__setattr__(self, "thresholds", tuple(self.thresholds))
        if self.progress is not None:
            object.__setattr__(self, "progress", _finite(self.progress, "ObsReport.progress"))

    def to_dict(self) -> dict:
        return jsonable({"name": self.name, "verdict": self.verdict, "progress": self.progress,
                         "reason": self.reason, "evidence": self.evidence,
                         "thresholds": thresholds_dicts(self.thresholds), "step": self.step})


# ------------------------------------------------------------------------------------------------ protocols
@runtime_checkable
class Observer(Protocol):
    """[C-4] dependencies injected at construction; no ctx argument; per-step extras arrive in ``context``."""
    name: str

    def start(self, view: StepView) -> None: ...

    def update(self, before: StepView, action, after: StepView,
               context: Mapping | None = None) -> ObsReport: ...


@runtime_checkable
class ProbeRecorder(Protocol):
    """Primitives push their probe frames; geometry / scene implement it."""

    def on_transition(self, stage: str, before: StepView, after: StepView) -> None: ...


@runtime_checkable
class TraceSink(Protocol):
    def emit(self, event: str, **payload) -> None: ...      # event in EVENTS; payload JSON-safe


PrimitiveFactory = Callable[[StepView], Primitive]
Yieldable = PrimitiveFactory | PointNeed | PlanNeed | RepairNeed
Reply = tuple | PointAnswer | PlanAnswer | RepairReply


# --------------------------------------------------------------------------------------------- threshold helpers
@dataclass(frozen=True)
class Term:
    """One term of a runtime threshold, e.g. Term("DD", 0.5, "G5_FLOOR_MM")."""
    label: str
    value: float
    text: str

    def __post_init__(self):
        _choice(self.label, LABEL_CLASSES, "Term.label")
        object.__setattr__(self, "value", _finite(self.value, "Term.value"))
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Term.text must be a nonempty string")


def _fmt(value) -> str:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    return f"{float(value):.4g}"


def _th_extreme(kind, name, terms, unit):
    if not terms:
        raise ValueError(f"th_{kind} needs at least one term")
    if not all(isinstance(t, Term) for t in terms):
        raise ValueError("terms must be Term records")
    pick = max if kind == "max" else min
    winner = pick(terms, key=lambda t: t.value)          # first term wins a tie
    listing = ", ".join(f"{t.text}={_fmt(t.value)}" for t in terms)
    derivation = f"{kind}({listing}) = {_fmt(winner.value)} {unit}; winner {winner.text} [{winner.label}]"
    return Threshold(name, winner.value, winner.label, derivation)


def th_max(name: str, *terms: Term, unit: str) -> Threshold:
    """value = max of the term values; label = the winning term's label; the derivation lists every term, e.g.
    "max(G5_FLOOR_MM=0.5, 4*sigma_close=0.12) = 0.5 mm; winner G5_FLOOR_MM [DD]"."""
    return _th_extreme("max", name, terms, unit)


def th_min(name: str, *terms: Term, unit: str) -> Threshold:
    """As th_max with the minimum (clips and upper bounds)."""
    return _th_extreme("min", name, terms, unit)


def th_value(name: str, value: float, label: str, formula: str, **inputs) -> Threshold:
    """A computed threshold: derivation = "<formula> with k=v, ..." (values .4g)."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a nonempty string")
    derivation = formula if not inputs else f"{formula} with " + ", ".join(f"{k}={_fmt(v)}" for k, v in inputs.items())
    return Threshold(name, value, label, derivation)


def module_versions() -> dict:
    return {"types": TYPES_VERSION, "robot": robot.ROBOT_VERSION}

