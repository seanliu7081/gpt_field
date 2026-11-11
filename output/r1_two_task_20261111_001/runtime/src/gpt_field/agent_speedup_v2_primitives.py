"""agent_SpeedUp_v2 primitives: closed-loop contact primitives at 20 Hz on state8 (contract §C8; spec §4.5, §4.9).

Every class here follows v1's ``contact_skills.Primitive`` protocol unchanged (contract §C3.1): ``next_action(view)``
then ``observe_transition(before, action, after)``; a primitive may be done at construction; it ends through
``_finish(status, reason, evidence)`` with JSON-safe evidence. Primitives never call the model (``pointing_request``
is always None) and never import geometry, scene, planner or observers: vision arrives as an injected
``ProbeRecorder`` and verdicts as injected ``Observer`` objects (kernel protocols). Every action is built by v1's
``HoldingActionAdapter`` (never an exactly-zero rotation).

Contents:
- ``CalibratePlus`` (spec §4.5 Calibrate+): v1 Calibrate's empty close / open / open hold re-written as one composite
  with the closed-hand yaw probe held inside the empty close, a second (and, on disagreement, a third) empty close,
  the three v1 external-axes probe moves, the vertical +-4 cm probe (R-1v) as two 2 cm halves per direction with a
  settle between them (design review 1 decision §2.5 M-2) and a B4 settle; ``gripper_quantities``,
  ``g5_present_threshold`` and ``g6_hold_floor`` turn its record into the G1-G6 thresholds.
- ``ParallaxProbe`` (D2): settle, then +-a out and back along R1, then R0, with every frame pushed to the recorder;
  its legs are ``_HoverProbeLeg`` (decision §2.5 M-1, RC-8: stall decided from step 5 on the leg's own free ratio,
  floor PROBE_FREE_RATIO_FLOOR), built by ``hover_probe_leg``; ``hover_arrival_descent`` is the same leg straight
  down for the relations' hover arrival (decision §2.6).
- ``TwistSegment`` (spec §4.9 Turn steps 2-3, K2) and ``HingeTwist`` (steps 4-8, K3, K5, K11) with the decision rules
  of [C-29]: the injected wedge, part-rotation and effect observers decide; the primitives only command the twist,
  hold the position softly ([C-10]), refit the hinge centre ([C-49]), follow the orbit and count the hand yaw.
- ``HeldReach`` (G7, M3), ``GuardedLower`` / ``Touchdown`` (G8, M5, P15), ``ReleaseVerify`` (M12, M13) and
  ``LiftMonitorAdapter`` (VerifyHold's monitor slot).
- ``GuardedClose`` (grasp-refusal decision J, GC-1): the grasp close, action for action v1's
  ``GripperSettlePrimitive("close")``, watching each finger's world pad face against the planner's close gate; an
  active guard abandons the close and opens on a finger touch beyond its limit or an inconsistent span.
- Reserved (contract §C0.3): ``DOFProbe``, ``FollowAxis``, ``PushAlong``, ``RayServo``, ``RayDescent``.

Observer lifecycle: ``TwistSegment`` and ``HingeTwist`` call ``start(view)`` at construction on the observers they
own for their own duration (``wedge`` / ``wedge_end`` and ``part_rotation``); they never call ``start`` on ``effect``,
whose pre-twist baseline the relation takes once before the first segment ([C-27]). Embedded observers never
preempt; they end their primitive through its own decision rules.

Twist sense (contract §C1.3, [C-25]): +1 is a positive rotation about the approach R2 (world -z for a top-down hand).
The hinge-centre refit and the orbit follow assume a top-down hand (the slice); angles about world z come from the
measured orientations.
"""

from __future__ import annotations

import math

import numpy as np

from . import agent_speedup_v2_robot as robot_facts
from .action import _quaternion_product, _rotvec_quaternion
from .agent_speedup_v2_thresholds import Threshold
from .agent_speedup_v2_types import (HANDOVER_PREFIX, CalibrationRecord, RobotModel, Term, jsonable,
                                     quaternion_approach_axis, th_max, th_value, thresholds_dicts, twist_about)
from .contact_skills import (AnchoredMove, GripperCalibration, GripperSettlePrimitive, HoldingActionAdapter,
                             HoldPrimitive, Primitive, StepView, WaypointReach, quaternion_angle_rad)
from .speed_first_gripper import GripperSettleConfig

PRIMITIVES_VERSION = "agent-speedup-v2-primitives-3"

# ---------------------------------------------------------------------------------------- G1-G4: Calibrate+
CAL_CLOSE_ACTIONS = 30
CAL_OPEN_MIN_ACTIONS = 15
CAL_HOLD_STEPS = 10
CAL_MIN_CLOSING_MM = 20.0
G4_SIGMA_FLOOR_MM = 0.01
G5_CLOSE_K = 15
CAL_YAW_PROBE_RAD = math.radians(5.0)
CAL_YAW_TRANSIENT_STEPS = 2
CAL_YAW_MAX_STEPS = 12
SIGMA_G_FLOOR_MM = 0.001
MAD_TO_SIGMA = 1.4826
CAL_VERTICAL_PROBE_M = 0.04
CAL_VERTICAL_HALVES = 2
CAL_PROBE_MOVES = ((0.040, 0.0, 0.0), (-0.040, 0.040, 0.0), (0.0, -0.040, 0.0))
G3_FLOOR_MM = 0.10
G3_SIGMA_K = 3.0
# ---------------------------------------------------------------------------------------- G5-G9, G12
G5_FLOOR_MM = 0.5
G5_SIGMA_K = 4.0
G6_FLOOR_MM = 0.5
G6_SIGMA_K = 6.0
G7_TOTAL_FLOOR_MM = 1.0
G7_TOTAL_SIGMA_K = 6.0
G7_STEP_FLOOR_MM = 0.4
G7_STEP_SIGMA_K = 4.0
G7_SIGMA_STEPS = 5
G8_FLOOR_MM = 0.3
G8_SIGMA_K = 4.0
G8_SIGMA_STEPS = 5
G9_FINGER_CONTACT_MM = 1.0
G12_STALL_FRACTION = 0.6
G12_RATIO_BOUNDS = (0.18, 0.40)
# ---------------------------------------------------------------------------------------- D2, RC-8 hover legs
D2_PROBE_M = 0.02
PROBE_STALL_MIN_STEPS = 5
PROBE_FREE_RATIO_FLOOR = 0.08
# v1 AnchoredMove's upper free-ratio clip (contact_skills.py:586), read from v1, not restated: the RC-8 hover leg
# keeps it (decision §2.5 M-1 writes AnchoredMove.FREE_RATIO_BOUNDS[1])
_, _V1_FREE_RATIO_HIGH = AnchoredMove.FREE_RATIO_BOUNDS
# ---------------------------------------------------------------------------------------- K2, K3, K5, K11, twist
K2_TWIST_RAD = 0.1
K3_FLOOR_M = 0.003
K5_FLOOR_RAD = 0.1
K5_FRACTION = 0.25
K11_FLOOR_M = 0.003
K11_SIGMA_K = 3.0
K11_STEPS = 4
K11_SIGMA_C_FLOOR_M = 0.005
TWIST_REFIT_EVERY = 5
TWIST_HOLD_CAP_M = 0.010
TWIST_SEGMENT_MAX_STEPS = 48
PART_STOP_WINDOW_STEPS = 4
PART_STOP_FLAT_LW = 0.1
HAND_YAW_STALL_RAD = math.radians(0.5)
HAND_YAW_STALL_STEPS = 12
# ---------------------------------------------------------------------------------------- P15, M3, M5, M12, M13
P15_COMMAND_M = 0.010
P15_LATERAL_CAP_M = 0.003
M3_REORIENT_RAD = math.radians(15.0)
M5_SATURATED_STEPS = 4
M5_SATURATED_REALIZED = 0.05
LOWER_FAST_COMMAND_M = 0.020
TOUCHDOWN_EXTRA_M = 0.01
M12_OPEN_TOL_MM = 2.0
M13_RETREAT_M = 0.03
OPEN_SETTLE_MIN_ACTIONS = 15
# ---------------------------------------------------------------------------------------- B4
B4_QUIET_DEG = 0.03
B4_QUIET_MM = 0.3
B4_QUIET_STEPS = 2
B4_MAX_STEPS = 14
# ---------------------------------------------------------------------------------------- GC-1 guarded close
GC_TOUCH_FLOOR_MM = 0.5
GC_TOUCH_SIGMA_K = 4.0
GC_FIRST_JUDGED_K = 2
GC_FREE_MIN_MM = 1.0
GC_REASONS = ("early_finger_contact", "span_inconsistent")

LABELS = {
    "GC_TOUCH_FLOOR_MM": ("DD", "grasp-refusal decision GC-1: a finger touches when its world inward motion falls "
                                "short of the calibrated free close at its own opening by this; remedy 2: free fingers "
                                "fall short by <= 0.19 mm over 186 free steps (15 recorded closes); judge: rigid "
                                "touches 1.36-4.27 mm on 10 live closes, a pushed free body 0.54 mm"),
    "GC_TOUCH_SIGMA_K": ("DD", "GC-1: x G4 sigma_close (as G5)"),
    "GC_FIRST_JUDGED_K": ("DD", "GC-1: the first close action is not judged (it starts from rest at any opening)"),
    "GC_FREE_MIN_MM": ("DD", "GC-1: an action whose calibrated free motion is < 1 mm per finger (the curve tail) is "
                             "not judged"),
    "CAL_CLOSE_ACTIONS": ("DD", "v1 Calibrate empty close GripperSettleConfig(min_close_actions=30, max_actions=30), "
                                "contact_skills.py:510 (restated; robot-only)"),
    "CAL_OPEN_MIN_ACTIONS": ("DD", "v1 Calibrate open GripperSettleConfig(min_open_actions=15, max_actions=30), "
                                   "contact_skills.py:528 (restated)"),
    "CAL_HOLD_STEPS": ("DD", "v1 Calibrate open hold of 10 steps, contact_skills.py:532 (G1 open gap, G3 inward noise)"),
    "CAL_MIN_CLOSING_MM": ("DD", "v1 Calibrate online rule: the empty close moves the fingers >= 20 mm, "
                                 "contact_skills.py:504 (restated)"),
    "G4_SIGMA_FLOOR_MM": ("DD", "[C-8] numerical floor of sigma_close: robot-only closes can agree to ~0; every "
                                "comparison that uses it has a larger DD floor (G5, G6 0.5 mm), so it never decides"),
    "G5_CLOSE_K": ("DD", "v1 classify_close compares gap(k = 15) with the empty curve at k = 15 and v1 Calibrate needs "
                         ">= 15 close samples for an online curve, contact_skills.py:541, :569 (restated)"),
    "CAL_YAW_PROBE_RAD": ("DD", "spec §4.5 Calibrate+ item 1: a +-5 deg closed-hand yaw probe at the K2 cap while the "
                                "empty close is held ([F, SP] 1.31 deg/step, gap moving <= 0.003 mm/step)"),
    "CAL_YAW_TRANSIENT_STEPS": ("DD", "[F, SP / CP] the hand-yaw onset realizes about 0.48 then 0.91 of the steady "
                                      "rate; the first two steps of each yaw stage are left out of the free yaw rate"),
    "CAL_YAW_MAX_STEPS": ("DD", "yaw stage cap = HAND_YAW_STALL_STEPS, the 12-step hand-yaw stall window: 5 deg takes "
                                "about 5 steps at the measured onset (0.48, 0.91 x) and the 1.31 deg/step free rate "
                                "([F, SP]), so a stage still short of 5 deg after one full stall window ends "
                                "(recorded, failed budget) instead of running on"),
    "SIGMA_G_FLOOR_MM": ("DD", "[C-8] numerical floor of sigma_g in mm/step; K1w compares 4 sigma_g with larger terms "
                               "(0.1 l w ~ 0.032 mm/step), so it never decides"),
    "MAD_TO_SIGMA": ("DD", "Gaussian consistency factor of the median absolute deviation, 1 / Phi^-1(0.75) = 1.4826: "
                           "sigma_g is a robust std of the step-to-step rate change / sqrt 2 (exact for uncorrelated "
                           "rate noise; uncorrelated reading noise gives sqrt 3, so it reads 1.22 x high, conservative) "
                           "because [F, robot-only, real LIBERO free-air empty close] the close still creeps at -0.05 "
                           "mm/step during the first yaw steps, then stays flat"),
    "CAL_VERTICAL_PROBE_M": ("DD", "spec §4.5 Calibrate+ item 4 / §5.2 D3: the vertical +-4 cm probe (looming, the "
                                   "third J_ext column, hand-locator validation; ruling R-1v granted)"),
    "CAL_VERTICAL_HALVES": ("DD", "design review 1 decision §2.5 M-2 (rulings item 5, §6 item 4e): each 4 cm vertical "
                                  "leg runs as two 2 cm halves with a B4 settle between them, so task 2 has 5 "
                                  "locatable calibration moves (probe_1 + 4 halves) instead of 3 and 7 external "
                                  "samples instead of 5 ([F, rulings] with 3 moves the self-scaled L1 bound accepted a "
                                  "40 px fault 15/15, with 5 it is dropped; [F, BG fix runs] a 4 cm vertical move "
                                  "is 26-28 px along v_hat at the start pose, so a 2 cm half clears L1_MIN_ALONG_PX). "
                                  "Margin traded for count: [F, real task 2, episodes 0-9, RS-3 run 002 and the BM "
                                  "review] "
                                  "each half moves 11.6-14.0 px along v_hat against L1_MIN_ALONG_PX = 8 px, so a "
                                  "start pose or camera at <= ~0.6 x task 2's J scale loses all four halves at once "
                                  "(1 locatable move, where the unsplit 4 cm moves keep 3); derived on task 2 only, "
                                  "re-checked at V5 on the other tasks' start poses"),
    "CAL_PROBE_MOVES": ("DD", "v1 PROBE_MOVES, agent_speedup_v1.py:72: the three 4 cm external-axes probe moves, net "
                              "zero (restated)"),
    "G3_FLOOR_MM": ("DD", "v1 GripperCalibration.touch_threshold_mm = max(0.10, 3 sigma_in), contact_skills.py:227 "
                          "(restated floor)"),
    "G3_SIGMA_K": ("DD", "v1 GripperCalibration.touch_threshold_mm = max(0.10, 3 sigma_in), contact_skills.py:227 "
                         "(restated factor)"),
    "G5_FLOOR_MM": ("DD", "G5 floor: v1 close margin, from the empty-close spread 0.35 mm (robot-only; spec v1 §6.5; "
                          "contact_skills.py:556)"),
    "G5_SIGMA_K": ("DD", "spec §5.2 G5: object present when gap(15) > empty(15) + max(0.5 mm, 4 sigma_close)"),
    "G6_FLOOR_MM": ("DD", "spec §5.2 G6 floor: 0.5 mm above the empty floor (replaces v1's TUNED-initial 1.0 mm, "
                          "agent_speedup_v1.py:106; [F, D003] the held object cleared v1's floor by only 0.87 mm)"),
    "G6_SIGMA_K": ("DD", "spec §5.2 G6: hold floor = empty floor + max(0.5 mm, 6 sigma_close)"),
    "G7_TOTAL_FLOOR_MM": ("DD", "provisional G7 total floor = v1 WaypointReach GAP_RISE_MM, contact_skills.py:886 (1.8 x "
                                "the recorded 0.55 mm clean-carry deviation); M0 re-derives it from robot-only traces"),
    "G7_TOTAL_SIGMA_K": ("DD", "spec §5.2 G7: a total rise >= max(1.0 mm, 6 sigma_carry) is held contact"),
    "G7_STEP_FLOOR_MM": ("DD", "provisional G7 per-step floor = v1 WaypointReach GAP_STEP_RISE_MM, contact_skills.py:887 "
                               "(2.7 x the 0.149 mm/step free-carry noise)"),
    "G7_STEP_SIGMA_K": ("DD", "spec §5.2 G7: a per-step rise >= max(0.4 mm, 4 sigma_carry) is held contact"),
    "G7_SIGMA_STEPS": ("DD", "spec §5.2 G7: sigma_carry from the held move's first 5 free steps"),
    "G8_FLOOR_MM": ("DD", "provisional G8 floor = v1 PLACE_GAP_RISE_MM, agent_speedup_v1.py:62 (3.3 x the recorded "
                          "0.09 mm lowering drift); re-derived in M0 like G7"),
    "G8_SIGMA_K": ("DD", "spec §5.2 G8: a per-step rise >= max(0.3 mm, 4 sigma_lower) is a support cue"),
    "G8_SIGMA_STEPS": ("DD", "spec §5.2 G8: sigma_lower from the lowering's first 5 free steps"),
    "G9_FINGER_CONTACT_MM": ("DD", "v1 WaypointReach FINGER_CONTACT_MM, contact_skills.py:890: an open-finger gap change "
                                   ">= 1.0 mm is a contact (robot-only: settled open fingers drift <= 0.55 mm)"),
    "G12_STALL_FRACTION": ("DD", "v1 AnchoredMove STALL_FRACTION, contact_skills.py:590: stall when the 2-step "
                                 "realized / commanded ratio < 0.6 x the move's free ratio (restated, record only)"),
    "G12_RATIO_BOUNDS": ("DD", "v1 AnchoredMove FREE_RATIO_BOUNDS, contact_skills.py:586: the free ratio is clipped to "
                               "0.18-0.40 (restated, record only)"),
    "D2_PROBE_M": ("DD", "spec §5.2 D2: hover probe amplitude +-2 cm (disparity kappa x 0.02 >= 40 px for Z <= 0.17 m; "
                         "v1's 2-6 mm probes gave 10-20 % kappa error)"),
    "PROBE_STALL_MIN_STEPS": ("DD", "design review 1 decision §2.5 M-1 / §3 (remedy 2 RC-8; DD now, ONLINE at V5): a "
                                    "hover-probe leg takes no stall decision before step 5, where v1 AnchoredMove "
                                    "measures the leg's own free ratio median(r[2:5]) (contact_skills.py:676-678); "
                                    "[F, robot-only replay of 76 free-air legs] the v1 step-3 decision against the "
                                    "0.26 default gave 33 false stalls, deciding from step 5 on the leg's own ratio 4 "
                                    "(0 with PROBE_FREE_RATIO_FLOOR); 4 real open-hand contacts still stall within one "
                                    "step of their ratio collapse"),
    "PROBE_FREE_RATIO_FLOOR": ("DD", "design review 1 decision §2.5 M-1 / §3 (remedy 2 RC-8), robot-only: the lower "
                                     "clip of a hover-probe leg's own free ratio, above 0.067 (a leg that starts in "
                                     "contact reads ratios <= 0.04 and must stall at step 5: 0.04 < 0.6 x 0.067) and "
                                     "below 0.083 (the lowest own free ratio of the 48 robot-only free-air legs that "
                                     "reached step 5, task-2 episode 5); floors 0.06-0.10 replay identically "
                                     "(0 false stalls of 76 legs, judge live 0 of 304 legs). Trade-off: the own "
                                     "ratio is the median of steps 3-5 even when those steps are already in "
                                     "contact, so a leg that starts on, or by step 4 meets, a yielding contact whose "
                                     "ratios stay above 0.6 x the floor (0.048; e.g. 0.05-0.10) is never stalled and "
                                     "ends on its budget, and a contact at step 4 stalls at step 5, not 4 ([F, T1 "
                                     "replay] starts at 0.05 / 0.06 and a 0.083 free start then 0.055 run to the "
                                     "budget; [F, RS-3 run 002, real task 2, 5 episodes, stall rule alone] rigid "
                                     "surfaces 4-8 mm below the start read <= 0.046 in contact and stall at their "
                                     "collapse + 1 to + 2: region top 4 mm step 5 vs 3, flat support 4 mm 5 vs 4, "
                                     "flat support 8 mm 6 vs 5; with G3 an open hand stops at collapse + 0 to + 1; a "
                                     "closed hand has no such backstop)"),
    "K2_TWIST_RAD": ("DD", "spec §5.2 K2: twist increment <= 0.1 rad per step, below the 0.15 rad ROBOT cap (R10); "
                           "[F, CP] a 0.5 rad cap lost the grip part"),
    "K3_FLOOR_M": ("DD", "spec §5.2 K3: orbit follow when the fitted hinge offset > max(3 mm, the Align tolerance)"),
    "K5_FLOOR_RAD": ("DD", "spec §5.2 K5: after the effect onset turn on to theta_e + max(0.1 rad, 0.25 theta_e)"),
    "K5_FRACTION": ("DD", "spec §5.2 K5: after the effect onset turn on to theta_e + max(0.1 rad, 0.25 theta_e)"),
    "K11_FLOOR_M": ("DD", "spec §5.2 K11: axis tug when the realized-versus-commanded lateral deviation > max(3 mm, "
                          "2 sin(theta/2) sigma_c + 3 sigma_pos) for 4 steps"),
    "K11_SIGMA_K": ("DD", "spec §5.2 K11: the 3 sigma_pos term of the axis-tug threshold"),
    "K11_STEPS": ("DD", "spec §5.2 K11: the axis-tug deviation must persist for 4 steps"),
    "K11_SIGMA_C_FLOOR_M": ("DD", "spec §4.9 Turn step 4: sigma_c = max(fit sigma, 5 mm, the measured close drag)"),
    "TWIST_REFIT_EVERY": ("DD", "spec §4.9 Turn step 4: the hinge centre is refit from the realized grip path every 5 "
                                "steps once the sense is free"),
    "TWIST_HOLD_CAP_M": ("DD", "[C-10] twist soft hold: the segment-start anchor correction clipped to 10 mm per axis "
                               "per step, the clip SP and CP used ([F, CP] the hand orbited by compliance without tug "
                               "and turned the part 1.2 rad)"),
    "TWIST_SEGMENT_MAX_STEPS": ("DD", "TwistSegment safety cap: the wedge observer decides by the K1w extend step 16 "
                                      "after its window origin, which a reversal delays by at most the unwedge limit "
                                      "ceil(1.5 x 16) = 24 steps [C-21]; 48 = 16 + 24 + 8 spare steps"),
    "PART_STOP_WINDOW_STEPS": ("DD", "[C-29] the HingeTwist part-stop split (rows 4-5) reads the mean gap rate over the "
                                     "last 4 steps, the K1w / K4w window length (spec §4.9 Turn step 3)"),
    "PART_STOP_FLAT_LW": ("DD", "[C-29] a mean gap rate below -0.1 l w over that window is a gap fall (the pads unwedge: "
                                "slip), within +-0.1 l w the gap is flat; 0.1 l w is the K1w free-sense rate bound "
                                "(spec §5.2 K1w; [F, SP] 0.1 l w ~ 0.032 mm/step)"),
    "HAND_YAW_STALL_RAD": ("DD", "spec §4.9 Turn step 6: hand_yaw_stall when the hand yaw progresses < 0.5 deg over 12 "
                                 "steps = v1 Orient STALL_MIN_DECREASE_RAD, contact_skills.py:783"),
    "HAND_YAW_STALL_STEPS": ("DD", "spec §4.9 Turn step 6: the 12-step hand-yaw stall window = v1 Orient STALL_STEPS, "
                                   "contact_skills.py:782"),
    "P15_COMMAND_M": ("DD", "spec §5.2 P15: guarded command 10 mm per step = v1 AnchoredMove command_m default, "
                            "contact_skills.py:603 (restated; this module is the sole owner)"),
    "P15_LATERAL_CAP_M": ("DD", "spec §5.2 P15: lateral cap 3 mm per step = v1 AnchoredMove lateral_cap_m default, "
                                "contact_skills.py:606 (restated; this module is the sole owner)"),
    "M3_REORIENT_RAD": ("DD", "spec §5.2 M3: a held reorientation of at most 15 deg = v1 MAX_HELD_REORIENT_RAD, "
                              "agent_speedup_v1.py:65"),
    "M5_SATURATED_STEPS": ("DD", "spec §5.2 M5: lateral_unreachable after 4 saturated correction steps = v1 Align "
                                 "BLOCKED_STEPS, wrist_servo.py:527"),
    "M5_SATURATED_REALIZED": ("DD", "a saturated correction step that realizes <= 5 % of the commanded lateral "
                                    "correction toward the aim counts = v1 Align BLOCKED_MAX_REALIZED, "
                                    "wrist_servo.py:526"),
    "LOWER_FAST_COMMAND_M": ("DD", "v1 PLACE_FAST_COMMAND_M, agent_speedup_v1.py:61: the fast lowering command "
                                   "(restated)"),
    "TOUCHDOWN_EXTRA_M": ("DD", "spec §4.7 item 6: Touchdown lowers at most the lift + 1 cm onto the pick support"),
    "M12_OPEN_TOL_MM": ("DD", "spec §5.2 M12: released when gap >= g_open - 2 mm = v1 OPEN_TOLERANCE_MM, "
                              "agent_speedup_v1.py:105"),
    "M13_RETREAT_M": ("DD", "spec §5.2 M13: retreat 3 cm along -R2 = v1 RETREAT_M, agent_speedup_v1.py:54"),
    "OPEN_SETTLE_MIN_ACTIONS": ("DD", "v1 OPEN_SETTLE = GripperSettleConfig(min_open_actions=15), "
                                      "agent_speedup_v1.py:111 (restated)"),
    "B4_QUIET_DEG": ("DD", "spec §5.2 B4: quiet when the hand rotates <= 0.03 deg/step = v1 Align SETTLE_ROTATION_RAD, "
                           "wrist_servo.py:504 (and HoldPrimitive QUIET_ROTATION_RAD)"),
    "B4_QUIET_MM": ("DD", "spec §5.2 B4: quiet when the hand moves <= 0.3 mm/step = v1 Align SETTLE_TRANSLATION_M, "
                          "wrist_servo.py:505 (and HoldPrimitive QUIET_TRANSLATION_M)"),
    "B4_QUIET_STEPS": ("DD", "spec §5.2 B4: 2 quiet steps = v1 Align SETTLE_QUIET_STEPS, wrist_servo.py:506"),
    "B4_MAX_STEPS": ("DD", "spec §5.2 B4: a settle ends within 14 steps = v1 Align SETTLE_MAX_STEPS, wrist_servo.py:507"),
    "WORLD_UP": ("ROBOT", "LIBERO world frame, z up (contract §C1.3): the vertical probe axis and the axis of the "
                          "hinge-centre refit for a top-down hand"),
}

CAL_STAGES = ("close_1", "yaw_plus", "yaw_minus", "open_1", "open_hold", "close_2", "open_2", "close_3", "open_3",
              "probe_1", "probe_2", "probe_3", "vertical_up_1", "settle_up", "vertical_up_2", "settle_top",
              "vertical_down_1", "settle_down", "vertical_down_2", "settle")
# the B4 settles between the vertical halves (decision §2.5 M-2); their labels carry no "vertical_" prefix, so the
# survey recorder treats them as static frames, never as an external-axes move of their own
CAL_VERTICAL_SETTLES = ("settle_up", "settle_top", "settle_down")
PROBE_STAGES = ("settle", "r1_out", "r1_back", "r0_out", "r0_back")
LOWER_PHASES = ("fast", "slow")
SUPPORT_CUES = ("contact_stall", "contact_touch", "gap_rise")
# evidence "decided_by": contract §C8 lists wedge | part_rotation | both for the TwistSegment sense rows; the rows
# where no observer decided the sense say which rule ended the primitive instead
TWIST_DECIDED_BY = ("wedge", "part_rotation", "both", "hold_floor", "budget", "cap")
HINGE_DECIDED_BY = ("wedge", "part_rotation", "both", "hold_floor", "tug", "effect", "rotation", "budget", "stall")
WORLD_UP = (0.0, 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ helpers
def _sense(value) -> int:
    if isinstance(value, (bool, np.bool_)) or value not in (1, -1):
        raise ValueError("sense must be +1 or -1 (rotation about the approach R2)")
    return int(value)


def _finite(value, name) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive(value, name) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _capped(vector, cap) -> np.ndarray:
    """Norm-capped copy (v1 contact_skills._capped, restated)."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if cap <= 0.0:
        return np.zeros(3)
    return vector * (cap / norm) if norm > cap else vector


def _report(report) -> dict | None:
    if report is None:
        return None
    to_dict = getattr(report, "to_dict", None)
    return jsonable(to_dict()) if callable(to_dict) else jsonable(report)


def _verdict(report) -> str | None:
    return None if report is None else getattr(report, "verdict", None)


def _reason(report) -> str | None:
    return None if report is None else getattr(report, "reason", None)


def _evidence(report) -> dict:
    value = None if report is None else getattr(report, "evidence", None)
    return value if isinstance(value, dict) else {}


def _rotation_state(report) -> str:
    """Part-rotation observer state from its verdict (§C9): corotating | part_not_corotating | unmeasurable |
    pending."""
    return {"met": "corotating", "violated": "part_not_corotating", "failed": "unmeasurable"}.get(_verdict(report),
                                                                                                 "pending")


def _rot2(angle) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _twist_target(view: StepView, sense: int, angle: float) -> np.ndarray:
    """The current orientation rotated by sense * angle about the current approach R2 (world-frame rotvec
    sense * angle * R2, contract §C1.3)."""
    return _quaternion_product(_rotvec_quaternion(sense * angle * view.R[:, 2]), view.quaternion_xyzw)


def _twist_action(adapter, view: StepView, sense: int, target_xyz) -> tuple[tuple, np.ndarray]:
    """One twist step: K2 about R2 in ``sense``, position correction toward ``target_xyz`` clipped per axis to
    TWIST_HOLD_CAP_M [C-10], gripper close. Returns (action, correction)."""
    correction = np.clip(np.asarray(target_xyz, dtype=float) - view.position, -TWIST_HOLD_CAP_M, TWIST_HOLD_CAP_M)
    return adapter.adapt(correction, _twist_target(view, sense, K2_TWIST_RAD), "close", view.state), correction


def _gap_rate(before: StepView, after: StepView) -> float:
    return float(after.gap_mm - before.gap_mm)


def _k15(curve) -> float | None:
    curve = list(curve)
    return float(curve[min(G5_CLOSE_K, len(curve)) - 1]) if curve else None


def fit_hinge_centre(points_xy, angles_rad) -> dict | None:
    """Least-squares hinge centre of a rigid planar rotation (spec §4.9 Turn step 4): p_k = c + Rz(phi_k) r0 with
    unknown centre c and initial radius vector r0 (linear in both), phi_k the measured rotation about world z since
    the first point. Returns {"centre_xy", "sigma_m" (sqrt of the largest eigenvalue of the centre covariance),
    "rms_m", "points", "arc_rad"} or None with fewer than 3 points or a singular arc."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    phi = np.asarray(angles_rad, dtype=float).reshape(-1)
    if len(pts) < 3 or len(phi) != len(pts):
        return None
    rows = []
    for angle in phi:
        rows.append(np.hstack([np.eye(2), _rot2(angle)]))
    A = np.vstack(rows)
    b = pts.reshape(-1)
    normal = A.T @ A
    if np.linalg.cond(normal) > 1.0 / 1e-12:
        return None
    x = np.linalg.solve(normal, A.T @ b)
    residual = b - A @ x
    dof = len(b) - len(x)
    s2 = float(residual @ residual) / dof if dof > 0 else 0.0
    cov = s2 * np.linalg.inv(normal)[:2, :2]
    sigma = math.sqrt(max(float(np.max(np.linalg.eigvalsh(cov))), 0.0))
    return {"centre_xy": [float(x[0]), float(x[1])], "sigma_m": sigma,
            "rms_m": math.sqrt(float(residual @ residual) / len(pts)), "points": int(len(pts)),
            "arc_rad": float(np.max(phi) - np.min(phi))}


def p15_thresholds() -> list[Threshold]:
    """The P15 guarded-move pair this module owns (relations import them, contract §C2.2)."""
    return [th_value("P15_command_m", P15_COMMAND_M, "DD", "P15_COMMAND_M guarded command per step (v1 AnchoredMove)"),
            th_value("P15_lateral_cap_m", P15_LATERAL_CAP_M, "DD",
                     "P15_LATERAL_CAP_M lateral cap per step (v1 AnchoredMove)")]


def g9_finger_contact_threshold() -> Threshold:
    """G9 (record only): the open-finger contact limit v1 WaypointReach compares an open move's gap change with
    (FINGER_CONTACT_MM; free moves and the Calibrate+ probe moves record it)."""
    return th_value("G9_open_finger_contact_mm", G9_FINGER_CONTACT_MM, "DD",
                    "G9_FINGER_CONTACT_MM open-finger gap change (v1 WaypointReach FINGER_CONTACT_MM)")


def g12_stall_threshold(move: AnchoredMove, **inputs) -> Threshold:
    """G12 (record only): the stall limit a contact-stopping AnchoredMove compares its 2-step realized / commanded
    ratio with, G12_STALL_FRACTION x the move's free ratio (median of its steps 3-5 once measured, else v1's default
    ratio), clipped to the move's OWN bounds and decided from the move's own minimum step: v1's G12_RATIO_BOUNDS and
    step 3 for v1 legs, (PROBE_FREE_RATIO_FLOOR, v1's upper bound) and PROBE_STALL_MIN_STEPS for a hover-probe leg
    (RC-8, ``_HoverProbeLeg``). A hover-probe leg that ended before PROBE_STALL_MIN_STEPS never compared a ratio:
    its record shows v1's default ratio and says "not decided". ``inputs`` (e.g. ``leg="r1_out"``) go into the
    derivation."""
    low, high = move.FREE_RATIO_BOUNDS
    ratio = float(np.clip(move.free_ratio, low, high))
    measured = bool(move.free_ratio_measured)
    if measured:
        source = "measured free ratio"
    elif isinstance(move, _HoverProbeLeg):
        source = "v1 default free ratio, not decided (the leg ended before PROBE_STALL_MIN_STEPS)"
    else:
        source = "v1 default free ratio (not yet measured)"
    bounds = ("(PROBE_FREE_RATIO_FLOOR, v1 upper bound), from step PROBE_STALL_MIN_STEPS (RC-8 hover leg)"
              if isinstance(move, _HoverProbeLeg) else "G12_RATIO_BOUNDS (v1)")
    return th_value("G12_stall_ratio", G12_STALL_FRACTION * ratio, "ONLINE" if measured else "DD",
                    f"G12_STALL_FRACTION x {source}, clipped to {bounds}", G12_STALL_FRACTION=G12_STALL_FRACTION,
                    free_ratio=ratio, ratio_low=low, ratio_high=high, stall_min_steps=move.STALL_MIN_STEPS, **inputs)


def touch_threshold(touch_threshold_mm=None) -> Threshold:
    """G3 as an open-finger touch limit: a Threshold passed in (the calibrated ``gripper_quantities`` G3) is kept;
    a number is the caller's calibrated G3; None means the G3 floor (G3_FLOOR_MM)."""
    if isinstance(touch_threshold_mm, Threshold):
        _positive(touch_threshold_mm.value, "touch_threshold_mm")
        return touch_threshold_mm
    if touch_threshold_mm is None:
        return th_value("G3_touch_threshold_mm", G3_FLOOR_MM, "DD", "G3_FLOOR_MM (no calibrated G3 passed in)")
    value = _positive(touch_threshold_mm, "touch_threshold_mm")
    return th_value("G3_touch_threshold_mm", value, "DD" if value == G3_FLOOR_MM else "ONLINE",
                    "calibrated G3 passed in: max(G3_FLOOR_MM, G3_SIGMA_K sigma_in)")


def b4_settle(view: StepView, *, gripper: str, orientation_xyzw=None, name: str = "settle") -> HoldPrimitive:
    """B4 quiet settle: v1 HoldPrimitive(until_quiet) capped at B4_MAX_STEPS with B4_QUIET_STEPS quiet steps
    (HoldPrimitive's own quiet test is B4: <= B4_QUIET_DEG and <= B4_QUIET_MM per step)."""
    return HoldPrimitive(view, steps=B4_MAX_STEPS, gripper=gripper, name=name, until_quiet=True,
                         quiet_steps=B4_QUIET_STEPS, orientation_xyzw=orientation_xyzw)


# -------------------------------------------------------------------------------------- gripper thresholds
def g5_present_threshold(gripper: GripperCalibration, sigma_close_mm: float) -> Threshold:
    """G5: the object-present MARGIN max(G5_FLOOR_MM, G5_SIGMA_K sigma_close) over the empty curve at k = 15
    (pass ``value`` as v1 classify_close's ``margin_mm``)."""
    sigma = max(_finite(sigma_close_mm, "sigma_close_mm"), 0.0)
    base = th_max("G5_object_present_gap_mm", Term("DD", G5_FLOOR_MM, "G5_FLOOR_MM"),
                  Term("ONLINE", G5_SIGMA_K * sigma, "G5_SIGMA_K*sigma_close"), unit="mm")
    empty = gripper.empty_gap_at(G5_CLOSE_K)
    return Threshold(base.name, base.value, base.label,
                     f"{base.derivation}; object present when gap(15) > empty(15) {empty:.4g} mm + this margin "
                     f"({gripper.source} empty curve)")


def g6_hold_floor(gripper: GripperCalibration, sigma_close_mm: float) -> Threshold:
    """G6: the hold floor = empty floor + max(G6_FLOOR_MM, G6_SIGMA_K sigma_close) (a finger gap below it is a lost
    hold; VerifyHold's ``hold_margin_mm`` is ``value - gripper.empty_floor_mm``)."""
    sigma = max(_finite(sigma_close_mm, "sigma_close_mm"), 0.0)
    margin = th_max("G6_hold_margin_mm", Term("DD", G6_FLOOR_MM, "G6_FLOOR_MM"),
                    Term("ONLINE", G6_SIGMA_K * sigma, "G6_SIGMA_K*sigma_close"), unit="mm")
    floor = gripper.empty_floor_mm
    return Threshold("G6_hold_floor_mm", floor + margin.value, margin.label,
                     f"empty floor {floor:.4g} mm ({gripper.source}) + {margin.derivation}")


def close_arbitration(k15_mm) -> dict:
    """Which empty closes set the empty curve and sigma_close (spec §4.5 item 2, §5.2 G4). Two closes agree when
    their gap(15) values differ by <= G5_FLOOR_MM; the third close, run only when the first two disagree, decides
    between them: the pair with the smallest |delta| is used and the other close is excluded, so one outlier close
    never sets the curve or sigma_close. Returns {"used": [i, j], "excluded": [...], "delta_mm", "agree"} (indices
    into ``k15_mm``); ``agree`` is False when no pair agrees within G5_FLOOR_MM (the closest pair is still used and
    CalibratePlus reports ``closes_agree`` False for the relation to record calibration_degraded)."""
    k15 = [_finite(v, "k15_mm") for v in k15_mm]
    if len(k15) < 2:
        return {"used": list(range(len(k15))), "excluded": [], "delta_mm": None, "agree": False}
    delta, i, j = min((abs(k15[a] - k15[b]), a, b) for a in range(len(k15)) for b in range(a + 1, len(k15)))
    return {"used": [i, j], "excluded": [k for k in range(len(k15)) if k not in (i, j)], "delta_mm": float(delta),
            "agree": bool(delta <= G5_FLOOR_MM)}


def gripper_quantities(record: CalibrationRecord) -> dict:
    """G1-G4 and the yaw-probe quantities of a CalibratePlus record (spec §4.5 items 1-2): sigma_close (G4,
    |delta gap(15)| / sqrt 2 of the agreeing pair of empty closes chosen by ``close_arbitration``, floored [C-8]),
    sigma_g (robot-only gap-rate noise while the closed empty hand yaws: MAD_TO_SIGMA x the median absolute deviation
    of the step-to-step change of the rate, / sqrt 2, so the creep of a still-settling close is not counted as noise;
    exact for uncorrelated rate noise, 1.22 x high for uncorrelated reading noise; floored [C-8]), the free yaw rate
    (median |realized yaw| per step outside the onset steps) and their thresholds."""
    gripper = record.gripper
    k15 = [float(v) for v in record.k15_mm]
    arbitration = close_arbitration(k15)
    used = [k15[i] for i in arbitration["used"]]
    sigma_close_raw = float(np.std(used, ddof=1)) if len(used) >= 2 else 0.0
    pair = "+".join(str(i + 1) for i in arbitration["used"]) or "none"
    yaw = record.yaw_probe or {}
    realized = np.abs(np.asarray(yaw.get("realized_rad_per_step", []), dtype=float))
    transient = np.asarray(yaw.get("transient", [False] * len(realized)), dtype=bool)
    steady = realized[~transient] if transient.shape == realized.shape else realized
    if steady.size:
        omega = float(np.median(steady))
    else:
        omega = float(np.median(realized)) if realized.size else 0.0
    rates = np.asarray(yaw.get("gap_rate_mm_per_step", []), dtype=float)
    if rates.size >= 3:
        change = np.diff(rates)
        sigma_g_raw = MAD_TO_SIGMA * float(np.median(np.abs(change - np.median(change)))) / math.sqrt(2.0)
    else:
        sigma_g_raw = float(np.std(rates)) if rates.size else 0.0
    online = gripper.source == "online"
    g1 = th_value("G1_open_gap_mm", gripper.open_gap_mm, "ONLINE" if online else "DD",
                  "mean open-hold gap (G1)" if online else "v1 recorded default open gap (no closing motion)")
    g2 = th_value("G2_empty_k15_mm", gripper.empty_gap_at(G5_CLOSE_K), "ONLINE" if online else "DD",
                  "empty-close curve at k = 15 (G2)", closes_used=pair, closes_run=len(k15))
    g3 = th_max("G3_touch_threshold_mm", Term("DD", G3_FLOOR_MM, "G3_FLOOR_MM"),
                Term("ONLINE", G3_SIGMA_K * gripper.inward_noise_mm, "G3_SIGMA_K*sigma_in"), unit="mm")
    g4 = th_max("G4_sigma_close_mm",
                Term("ONLINE", sigma_close_raw, f"|delta gap(15)|/sqrt2 of closes {pair} of {len(k15)}"
                     + ("" if arbitration["agree"] else " (no pair within G5_FLOOR_MM)")),
                Term("DD", G4_SIGMA_FLOOR_MM, "G4_SIGMA_FLOOR_MM"), unit="mm")
    sigma_g = th_max("sigma_g_mm_per_step", Term("ONLINE", sigma_g_raw, "robust std of the yaw-probe gap rates"),
                     Term("DD", SIGMA_G_FLOOR_MM, "SIGMA_G_FLOOR_MM"), unit="mm/step")
    return {"sigma_close_mm": g4.value, "sigma_close_raw_mm": sigma_close_raw, "sigma_g_mm_per_step": sigma_g.value,
            "sigma_g_raw_mm_per_step": sigma_g_raw, "free_yaw_rate_rad_per_step": omega, "k15_mm": k15,
            "close_arbitration": arbitration, "closes_agree": arbitration["agree"],
            "yaw_samples": int(steady.size), "thresholds": [g1, g2, g3, g4, sigma_g]}


# ------------------------------------------------------------------------------------------------ Calibrate+
class _ClosedYaw(Primitive):
    """A Calibrate+ yaw stage: the closed empty hand yaws ``sense`` x K2 about R2 per step, position anchored (per-axis
    clip TWIST_HOLD_CAP_M), until its net yaw about the reference approach reaches ``stop_net_rad`` (sense-wise) or
    CAL_YAW_MAX_STEPS steps; records the realized yaw and the gap rate of every step."""

    def __init__(self, view: StepView, *, sense: int, stop_net_rad: float, reference_xyzw, robot: RobotModel,
                 name: str):
        super().__init__(name)
        self.sense = _sense(sense)
        self.stop_net_rad = _finite(stop_net_rad, "stop_net_rad")
        self.reference_xyzw = np.asarray(reference_xyzw, dtype=float).copy()
        self.reference_axis = quaternion_approach_axis(self.reference_xyzw)
        self.robot = robot
        self.anchor = view.position.copy()
        self.adapter = HoldingActionAdapter(max_rotation_rad=K2_TWIST_RAD)
        self.realized_rad: list[float] = []
        self.gap_rates_mm: list[float] = []
        self.net_rad = self._net(view)
        if self._reached():
            self._finish("succeeded", "yaw_reached", self._evidence_now())

    def _net(self, view) -> float:
        return twist_about(self.reference_xyzw, view.quaternion_xyzw, self.reference_axis)

    def _reached(self) -> bool:
        return self.sense * (self.net_rad - self.stop_net_rad) >= 0.0

    def _next(self, view):
        action, _ = _twist_action(self.adapter, view, self.sense, self.anchor)
        return action

    def _observe(self, before, action, after):
        realized = self.robot.yaw_about_approach(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2])
        self.realized_rad.append(realized)
        self.gap_rates_mm.append(_gap_rate(before, after))
        self.net_rad = self._net(after)
        if self._reached():
            self._finish("succeeded", "yaw_reached", self._evidence_now())
        elif self.actions_executed >= CAL_YAW_MAX_STEPS:
            self._finish("failed", "budget", self._evidence_now())
        return {"realized_rad": realized, "net_rad": self.net_rad, "gap_mm": after.gap_mm}

    def _evidence_now(self) -> dict:
        return jsonable({"sense": self.sense, "stop_net_rad": self.stop_net_rad, "net_rad": self.net_rad,
                         "realized_rad_per_step": self.realized_rad, "gap_rate_mm_per_step": self.gap_rates_mm,
                         "steps": self.actions_executed})


class _ProbeLeg(AnchoredMove):
    """An AnchoredMove leg whose last command is shortened to the remaining travel at the move's free ratio, so an
    out-and-back pair returns to its start instead of overshooting by up to one realized step."""

    def _next(self, view):
        remaining = self.travel_m - float((view.position - self.start_xyz) @ self.axis)
        command = min(self.command_m, max(remaining, 0.0) / self.free_ratio)
        lateral = _capped(-self.lateral_offset(view.position), self.lateral_cap_m)
        return self.adapter.adapt(command * self.axis + lateral, self.orientation_xyzw, self.gripper, view.state)


class _HoverProbeLeg(_ProbeLeg):
    """RC-8 hover-probe leg (design review 1 decision §2.5 M-1, remedy 2 RC-8): v1 AnchoredMove's stall arithmetic
    reused as is (contact_skills.py:672-679, :694-697), with two class attributes changed. No stall decision before
    step PROBE_STALL_MIN_STEPS, the step at which the leg's own free ratio exists (``clip(median(r[2:5]), floor,
    high)``); the clip floor lowered from v1's 0.18 to the robot-only PROBE_FREE_RATIO_FLOOR, so a slow but free leg
    (own ratio 0.083) is compared with its own ratio, while a leg that starts in contact (ratios <= 0.04) still stalls
    at step 5. The open-finger touch (G3) is checked every step and still stops a leg at once. Used by every
    ParallaxProbe leg and by ``hover_arrival_descent``; Calibrate+'s legs, GuardedLower and Touchdown keep v1's
    rule.

    Trade-off kept by the decision (PROBE_FREE_RATIO_FLOOR derivation): the own ratio is measured on steps 3-5 even
    when they are already in contact, so a contact that begins by step 4 and still yields ratios above
    G12_STALL_FRACTION x PROBE_FREE_RATIO_FLOOR (0.048) is never stalled and ends on ``budget``, and a contact at
    step 4 stalls at step 5. An open hand still has G3; a closed hand has no such backstop."""

    STALL_MIN_STEPS = PROBE_STALL_MIN_STEPS
    FREE_RATIO_BOUNDS = (PROBE_FREE_RATIO_FLOOR, _V1_FREE_RATIO_HIGH)


def hover_probe_leg(view: StepView, *, axis, travel_m: float, gripper: str, touch_threshold_mm=None,
                    orientation_xyzw=None, lateral_target=None, name: str = "hover_probe_leg") -> _HoverProbeLeg:
    """A guarded hover leg with the ParallaxProbe contact stops (decision §2.5 M-1): a ``_HoverProbeLeg`` along the
    world ``axis`` for ``travel_m`` at the P15 command and lateral cap, stop on contact (the RC-8 stall rule, armed
    once the leg moves, v1 ``arm_stall_after_motion``; with the gripper open also the G3 finger touch at
    ``touch_threshold_mm``, a number or the ``gripper_quantities`` Threshold, default the G3 floor), the
    orientation held (default: the view's) and a lateral hold on ``lateral_target`` (default: the view's position).
    ``g12_stall_threshold(leg)`` records its stall limit."""
    if gripper not in ("open", "close"):
        raise ValueError("gripper must be 'open' or 'close'")
    g3 = touch_threshold(touch_threshold_mm)
    return _HoverProbeLeg(view, axis=axis, travel_m=_positive(travel_m, "travel_m"), gripper=gripper,
                          command_m=P15_COMMAND_M, stop_on_contact=True, touch_threshold_mm=g3.value,
                          lateral_cap_m=P15_LATERAL_CAP_M,
                          orientation_xyzw=view.quaternion_xyzw if orientation_xyzw is None else orientation_xyzw,
                          lateral_target=view.position if lateral_target is None else lateral_target,
                          arm_stall_after_motion=True, name=name)


def hover_arrival_descent(view: StepView, *, travel_m: float, touch_threshold_mm=None, orientation_xyzw=None,
                          name: str = "hover_arrival") -> _HoverProbeLeg:
    """The hover-arrival descent offered to the relations (decision §2.6 "Hover arrival", rulings item 4, RC-5): from a
    settled hover, an open-hand ``hover_probe_leg`` straight down along world -z for ``travel_m`` (the caller's
    HOVER_LOCATE_M), holding the start xy and orientation, stopping on a G3 touch or an RC-8 stall. A stop other than
    ``travel_reached`` (``contact_touch``, ``contact_stall`` and also ``budget``, which is how a contact that yields
    above the stall limit from the start ends, see ``_HoverProbeLeg``) is the caller's ``transit_contact``; the
    settled start frame and the end frame are the caller's ``reanchor_from`` pair.

    Not a sufficient safety check on a yielding free part that the open fingers straddle: neither G3 nor the stall
    sees a part that slides or tips between the fingers, so it can be pushed and tilted before either stops the leg,
    or the leg reaches its travel ([F, BM review, real task 2, 5 episodes] descents started 5-10 mm above a light
    free elongated part pushed it 12.6-21.8 mm and tilted it 8.8-14.8 deg before G3 or the stall, or with no stop
    at all; v1's rule behaves the same; rigid and heavier parts moved <= 1.5 mm). Transit contact on a free part
    therefore needs an observer or visual check as well (relations and observers)."""
    return hover_probe_leg(view, axis=-np.asarray(WORLD_UP, dtype=float), travel_m=travel_m, gripper="open",
                           touch_threshold_mm=touch_threshold_mm, orientation_xyzw=orientation_xyzw, name=name)


class CalibratePlus(Primitive):
    """Spec §4.5 Calibrate+ (episode start, zero calls, ~190 steps), contract §C8.

    Stages (``STAGES``; close_3 / open_3 only when the two empty closes disagree at k = 15 by more than G5_FLOOR_MM):
    close_1 (v1 empty close, 30 actions) -> yaw_plus / yaw_minus (closed empty hand, +5 deg about R2 and back at K2
    per step, anchored: the free yaw rate and sigma_g) -> open_1 -> open_hold (v1 open hold, 10 steps: g_open, inward
    noise, static frames) -> close_2 -> open_2 [-> close_3 -> open_3] -> probe_1..3 (v1 PROBE_MOVES through
    WaypointReach, open; goals relative to the calibration start so errors do not accumulate; each record carries the
    offset commanded from its stage start and the nominal v1 move) -> the vertical probe +-4 cm along world z as
    CAL_VERTICAL_HALVES = 2 halves per direction with a B4 settle between consecutive halves (decision §2.5 M-2,
    rulings item 5): vertical_up_1 -> settle_up -> vertical_up_2 -> settle_top -> vertical_down_1 -> settle_down ->
    vertical_down_2 (v1-rule AnchoredMove legs, the last command trimmed to the remaining travel; the up halves hold
    the up start xy, so they are purely vertical, and the down halves return to the calibration start height and xy;
    each half is its own external-axes move and starts from rest: 5 locatable moves and 7 external samples on task 2
    instead of 3 and 5) -> settle (B4 quiet hold at the start orientation). ``recorder.on_transition(stage, before,
    after)`` runs on every transition; the
    fingertip-band pair is the last open_hold frame and the end of close_2 (same pose). The empty curve is the mean
    of the closes ``close_arbitration`` picks (both closes when they agree; with a third close the agreeing pair, the
    outlier excluded; with no agreeing pair the closest pair and evidence ``closes_agree`` False). The outcome follows
    v1's rule on those closes: succeeded ``calibrated_online`` when they moved the fingers >= CAL_MIN_CLOSING_MM over
    >= 15 actions, else ``calibrated_default`` with the recorded default curve. A probe or vertical stage that ends
    early is recorded and the calibration continues."""

    STAGES = CAL_STAGES

    def __init__(self, view: StepView, *, recorder=None, robot: RobotModel, name: str = "calibrate_plus"):
        super().__init__(name)
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        self.robot = robot
        self.recorder = recorder
        self.start_state = view.state.copy()
        self.start_xyz = view.position.copy()
        self.start_xyzw = view.quaternion_xyzw.copy()
        self.start_gap_mm = view.gap_mm
        self.closes: list[GripperSettlePrimitive] = []
        self.opens: list[GripperSettlePrimitive] = []
        self.hold: HoldPrimitive | None = None
        self.yaw_stages: list[_ClosedYaw] = []
        self.probe_moves: list[dict] = []
        self.vertical: list[dict] = []
        self.stage_outcomes: list[dict] = []
        self.stages_run: list[str] = []
        self.third_close = False
        self._vertical_start: np.ndarray | None = None
        self.calibration: GripperCalibration | None = None
        self._queue = [s for s in CAL_STAGES if s not in ("close_3", "open_3")]
        self.stage: str | None = None
        self.current: Primitive | None = None
        self._stage_view: StepView | None = None
        self._advance(view)

    # ------------------------------------------------------------------ stages
    def _build(self, stage: str, view: StepView) -> Primitive | None:
        tag = f"{self.name}_{stage}"
        if stage.startswith("close_"):
            return GripperSettlePrimitive("close", view, config=GripperSettleConfig(
                min_close_actions=CAL_CLOSE_ACTIONS, max_actions=CAL_CLOSE_ACTIONS), name=tag)
        if stage.startswith("open_") and stage != "open_hold":
            return GripperSettlePrimitive("open", view, config=GripperSettleConfig(
                min_open_actions=CAL_OPEN_MIN_ACTIONS, max_actions=CAL_CLOSE_ACTIONS), name=tag)
        if stage == "yaw_plus":
            return _ClosedYaw(view, sense=1, stop_net_rad=CAL_YAW_PROBE_RAD, reference_xyzw=self.start_xyzw,
                              robot=self.robot, name=tag)
        if stage == "yaw_minus":
            return _ClosedYaw(view, sense=-1, stop_net_rad=0.0, reference_xyzw=self.start_xyzw, robot=self.robot,
                              name=tag)
        if stage == "open_hold":
            return HoldPrimitive(view, steps=CAL_HOLD_STEPS, gripper="open", name=tag, orientation_xyzw=self.start_xyzw)
        if stage.startswith("probe_"):
            index = int(stage.split("_")[1]) - 1
            goal = self.start_xyz + np.sum(np.asarray(CAL_PROBE_MOVES[:index + 1], dtype=float), axis=0)
            return WaypointReach(view, goal - view.position, gripper="open", name=tag,
                                 orientation_xyzw=self.start_xyzw)
        if stage.startswith("vertical_"):
            return self._vertical_half(stage, view, tag)
        if stage == "settle" or stage in CAL_VERTICAL_SETTLES:
            return b4_settle(view, gripper="open", orientation_xyzw=self.start_xyzw, name=tag)
        raise ValueError(f"unknown Calibrate+ stage {stage!r}")

    def _vertical_half(self, stage: str, view: StepView, tag: str) -> Primitive | None:
        """Decision §2.5 M-2: half i of CAL_VERTICAL_HALVES of the vertical probe, a v1-rule _ProbeLeg (no contact
        stop, the last command trimmed to the remaining travel). Goals are relative, so arrival errors do not add up:
        up half i ends CAL_VERTICAL_PROBE_M x i / CAL_VERTICAL_HALVES above the up start and holds the up start xy (a
        purely vertical probe); down half i ends that share of the probe below the top, so the last one returns to
        the calibration start height, holding the calibration start xy. None (skipped) when nothing is left."""
        _, direction, index = stage.split("_")
        share = int(index) / CAL_VERTICAL_HALVES
        if direction == "up":
            if self._vertical_start is None:
                self._vertical_start = view.position.copy()
            axis, hold = np.asarray(WORLD_UP, dtype=float), self._vertical_start
            travel = float(self._vertical_start[2] + CAL_VERTICAL_PROBE_M * share - view.position[2])
        else:
            axis, hold = -np.asarray(WORLD_UP, dtype=float), self.start_xyz
            travel = float(view.position[2] - (self.start_xyz[2] + CAL_VERTICAL_PROBE_M * (1.0 - share)))
        if travel <= AnchoredMove.TRAVEL_TOLERANCE_M:
            return None
        return _ProbeLeg(view, axis=axis, travel_m=travel, gripper="open", command_m=P15_COMMAND_M,
                         lateral_cap_m=P15_LATERAL_CAP_M, orientation_xyzw=self.start_xyzw, lateral_target=hold,
                         name=tag)

    def _advance(self, view: StepView) -> None:
        while self._queue:
            stage = self._queue.pop(0)
            primitive = self._build(stage, view)
            if primitive is None:
                self.stage_outcomes.append({"stage": stage, "primitive": None, "status": "skipped",
                                            "reason": "nothing_to_do", "actions": 0})
                continue
            self.stage, self.current, self._stage_view = stage, primitive, view
            self.stages_run.append(stage)
            self._register(stage, primitive)
            if primitive.done:
                self._stage_done(stage, primitive, view)
                continue
            return
        self.stage, self.current = None, None
        self._conclude(view)

    def _register(self, stage: str, primitive: Primitive) -> None:
        if stage.startswith("close_"):
            self.closes.append(primitive)
        elif stage.startswith("open_") and stage != "open_hold":
            self.opens.append(primitive)
        elif stage == "open_hold":
            self.hold = primitive
        elif stage.startswith("yaw_"):
            self.yaw_stages.append(primitive)

    def _stage_done(self, stage: str, primitive: Primitive, after: StepView) -> None:
        outcome = primitive.outcome()
        self.stage_outcomes.append({"stage": stage, "primitive": outcome["primitive"], "status": outcome["status"],
                                    "reason": outcome["reason"], "actions": outcome["actions"]})
        start = self._stage_view
        if stage.startswith("probe_") or stage.startswith("vertical_"):
            entry = {"stage": stage, "reason": outcome["reason"], "status": outcome["status"],
                     "world_displacement_m": (after.position - start.position).tolist(),
                     "start_state8": start.state.tolist(), "end_state8": after.state.tolist()}
            if stage.startswith("probe_"):
                # the goal is relative to the calibration start, so the offset commanded from this stage's start
                # differs from the nominal v1 move by the previous stages' arrival errors
                entry["commanded_m"] = np.asarray(primitive.offset_m, dtype=float).tolist()
                entry["nominal_m"] = list(CAL_PROBE_MOVES[int(stage.split("_")[1]) - 1])
                entry["thresholds"] = thresholds_dicts([g9_finger_contact_threshold()])
                self.probe_moves.append(entry)
            else:
                # travel along the leg axis plus the lateral return to the hold line (zero for vertical_up, which
                # holds its own start; vertical_down holds the calibration start xy and carries the return)
                commanded = float(primitive.travel_m) * primitive.axis - primitive.lateral_offset(primitive.start_xyz)
                entry["commanded_m"] = np.asarray(commanded, dtype=float).tolist()
                entry["lateral_target_xyz"] = np.asarray(primitive.anchor_xyz, dtype=float).tolist()
                self.vertical.append(entry)
        if stage == "close_2" and len(self.closes) >= 2:
            first, second = _k15(self.closes[0].gap_curve_mm), _k15(self.closes[1].gap_curve_mm)
            if first is not None and second is not None and abs(first - second) > G5_FLOOR_MM:
                self.third_close = True
                at = self._queue.index("open_2") + 1 if "open_2" in self._queue else 0
                self._queue[at:at] = ["close_3", "open_3"]

    def _next(self, view):
        return self.current.next_action(view)

    def _observe(self, before, action, after):
        stage, current = self.stage, self.current
        feedback = current.observe_transition(before, action, after)
        if self.recorder is not None:
            self.recorder.on_transition(stage, before, after)
        if current.done:
            self._stage_done(stage, current, after)
            self._advance(after)
        return {"stage": stage, "gap_mm": after.gap_mm, "stage_done": bool(feedback.get("done"))}

    # ------------------------------------------------------------------ conclusion
    def _scored_closes(self) -> list[GripperSettlePrimitive]:
        """The empty closes with a gap(15) value, in run order (the order of ``record().k15_mm``)."""
        return [c for c in self.closes if _k15(c.gap_curve_mm) is not None]

    @staticmethod
    def _online(closes) -> bool:
        return bool(closes) and all(
            len(c.gap_curve_mm) >= G5_CLOSE_K and c.start_gap_mm - min(c.gap_curve_mm) >= CAL_MIN_CLOSING_MM
            for c in closes)

    def _conclude(self, view: StepView) -> None:
        curves = [list(c.gap_curve_mm) for c in self.closes]
        scored = self._scored_closes()
        arbitration = close_arbitration([_k15(c.gap_curve_mm) for c in scored])
        used = [scored[i] for i in arbitration["used"]]
        online = self._online(used) and self.hold is not None and bool(self.hold.gaps_mm)
        if online:
            used_curves = [list(c.gap_curve_mm) for c in used]
            n = min(len(c) for c in used_curves)
            mean_curve = np.mean(np.asarray([c[:n] for c in used_curves], dtype=float), axis=0)
            self.calibration = GripperCalibration(float(np.mean(self.hold.gaps_mm)), tuple(float(v) for v in mean_curve),
                                                  self.hold.inward_noise_mm, "online")
        else:
            self.calibration = GripperCalibration.default()
        first = curves[0] if curves else []
        closing = self.closes[0].start_gap_mm - min(first) if first else 0.0
        self._finish("succeeded", "calibrated_online" if online else "calibrated_default", jsonable({
            "calibration": self.calibration.to_dict(), "online": online, "closing_motion_mm": closing,
            "close_count": len(self.closes), "third_close": self.third_close,
            "close_arbitration": arbitration, "closes_agree": arbitration["agree"],
            "k15_mm": [_k15(c) for c in curves], "start_gap_mm": self.start_gap_mm,
            "stages": self.stage_outcomes, "yaw_probe": self._yaw_probe(), "probe_moves": self.probe_moves,
            "vertical": self.vertical, "steps": self.actions_executed,
            "final_offset_m": (view.position - self.start_xyz).tolist(),
            "final_yaw_rad": twist_about(self.start_xyzw, view.quaternion_xyzw, quaternion_approach_axis(self.start_xyzw))}))

    def _yaw_probe(self) -> dict:
        realized, rates, transient, stages = [], [], [], []
        for stage in self.yaw_stages:
            for i, (r, g) in enumerate(zip(stage.realized_rad, stage.gap_rates_mm)):
                realized.append(r)
                rates.append(g)
                transient.append(i < CAL_YAW_TRANSIENT_STEPS)
                stages.append(stage.name)
        return {"commanded_rad": K2_TWIST_RAD, "probe_rad": CAL_YAW_PROBE_RAD, "realized_rad_per_step": realized,
                "gap_rate_mm_per_step": rates, "transient": transient, "stage": stages}

    def record(self) -> CalibrationRecord:
        """The CalibrationRecord (kernel §C4.5) of the finished calibration."""
        if not self.done:
            raise RuntimeError("CalibratePlus has not finished")
        hold = self.hold
        return CalibrationRecord(
            gripper=self.calibration,
            close_curves_mm=tuple(tuple(float(v) for v in c.gap_curve_mm) for c in self.closes),
            k15_mm=tuple(v for v in (_k15(c.gap_curve_mm) for c in self.closes) if v is not None),
            yaw_probe=jsonable(self._yaw_probe()),
            open_hold=jsonable({"gaps_mm": [] if hold is None else list(hold.gaps_mm),
                                "inward_steps_mm": [] if hold is None else list(hold.inward_steps_mm)}),
            probe_moves=tuple(jsonable(m) for m in self.probe_moves),
            vertical=tuple(jsonable(m) for m in self.vertical),
            stage_outcomes=tuple(jsonable(s) for s in self.stage_outcomes),
            steps=self.actions_executed)


# ----------------------------------------------------------------------------------------------- ParallaxProbe
class ParallaxProbe(Primitive):
    """D2 hover probe (spec §4.5 item 2, §4.9 Grasp step 3): +-``amplitude_m`` out and back along R1, then along R0
    (axes frozen at the end of the first settle), each leg a ``_HoverProbeLeg`` (``hover_probe_leg``) with a lateral
    hold on the probe start, stop on contact (stall armed once the leg moves, v1 ``arm_stall_after_motion``, and
    decided by the RC-8 rule: from step PROBE_STALL_MIN_STEPS on the leg's own free ratio clipped at
    PROBE_FREE_RATIO_FLOOR; [F, review 1 grasp lens] v1's step-3 rule ended 32 free-air legs of 100 grip-mode
    hovers with a false contact_stall), the orientation held, and every leg
    preceded by a B4 settle (stage ``settle``) so it starts from rest: [F, robot-only, real LIBERO free-air hover
    probe] a back leg started straight after its out leg realized 0.004 then 0.12-0.14 of the command while the arm
    still coasted out, which v1's stall rule reads as a contact. With the gripper open each leg also stops on a finger
    touch at ``touch_threshold_mm`` (the calibrated G3, a number or the ``gripper_quantities`` Threshold; default
    the G3 floor, as ReleaseVerify), so an open finger brushing a part ends the leg (spec D2). Every transition goes
    to ``recorder.on_transition(stage, ...)``. A leg that ends other than at its travel (a stall, a finger touch or no
    progress) fails the probe with ``probe_contact``. Evidence: {"net_displacement_m", "legs" (each with its G12
    stall record), "steps", "axes", "amplitude_m", "thresholds" (D2, G3 when open, G12 per leg)}."""

    STAGES = PROBE_STAGES

    def __init__(self, view: StepView, *, recorder, gripper: str, amplitude_m: float = D2_PROBE_M,
                 name: str = "parallax_probe", touch_threshold_mm=None):
        super().__init__(name)
        if gripper not in ("open", "close"):
            raise ValueError("gripper must be 'open' or 'close'")
        self.recorder = recorder
        self.gripper = gripper
        self.amplitude_m = _positive(amplitude_m, "amplitude_m")
        self.g3 = touch_threshold(touch_threshold_mm)
        self.touch_threshold_mm = self.g3.value
        self.g12: list[Threshold] = []
        self.start_xyz = view.position.copy()
        self.start_xyzw = view.quaternion_xyzw.copy()
        self.axes: dict | None = None
        self.legs: list[dict] = []
        self._queue = [stage for leg in PROBE_STAGES[1:] for stage in ("settle", leg)]
        self.stage: str | None = None
        self.current: Primitive | None = None
        self._stage_view: StepView | None = None
        self._advance(view)

    def _build(self, stage: str, view: StepView) -> Primitive | None:
        tag = f"{self.name}_{stage}"
        if stage == "settle":
            return b4_settle(view, gripper=self.gripper, orientation_xyzw=self.start_xyzw, name=tag)
        if self.axes is None:
            self.axes = {"r1": view.R[:, 1].copy(), "r0": view.R[:, 0].copy()}
            self.start_xyz = view.position.copy()
        axis = self.axes[stage[:2]]
        if stage.endswith("_out"):
            travel, direction = self.amplitude_m, axis
        else:
            travel, direction = float((view.position - self.start_xyz) @ axis), -axis
            if travel <= AnchoredMove.TRAVEL_TOLERANCE_M:
                return None
        return hover_probe_leg(view, axis=direction, travel_m=travel, gripper=self.gripper,
                               touch_threshold_mm=self.touch_threshold_mm, orientation_xyzw=self.start_xyzw,
                               lateral_target=self.start_xyz, name=tag)

    def _advance(self, view: StepView) -> None:
        while self._queue and not self.done:
            stage = self._queue.pop(0)
            primitive = self._build(stage, view)
            if primitive is None:
                self.legs.append({"stage": stage, "reason": "nothing_to_do", "travel_m": 0.0, "steps": 0})
                continue
            self.stage, self.current, self._stage_view = stage, primitive, view
            if primitive.done:
                self._stage_done(stage, primitive, view)
                continue
            return
        if not self.done:
            self._finish("succeeded", "probe_complete", self._evidence_now(view))

    def _stage_done(self, stage: str, primitive: Primitive, after: StepView) -> None:
        outcome = primitive.outcome()
        evidence = outcome.get("evidence") or {}
        if stage != "settle":
            g12 = g12_stall_threshold(primitive, leg=stage)
            self.g12.append(g12)
            self.legs.append({"stage": stage, "reason": outcome["reason"], "status": outcome["status"],
                              "travel_m": evidence.get("travel_m"), "steps": outcome["actions"],
                              "contact": evidence.get("contact"), "start_xyz": self._stage_view.position.tolist(),
                              "end_xyz": after.position.tolist(), "thresholds": thresholds_dicts([g12])})
            if outcome["reason"] != "travel_reached":
                self._finish("failed", "probe_contact", self._evidence_now(after))

    def _next(self, view):
        return self.current.next_action(view)

    def _observe(self, before, action, after):
        stage, current = self.stage, self.current
        current.observe_transition(before, action, after)
        if self.recorder is not None:
            self.recorder.on_transition(stage, before, after)
        if current.done:
            self._stage_done(stage, current, after)
            self._advance(after)
        return {"stage": stage, "gap_mm": after.gap_mm}

    def _evidence_now(self, view: StepView) -> dict:
        axes = None if self.axes is None else {k: v.tolist() for k, v in self.axes.items()}
        return jsonable({"net_displacement_m": (view.position - self.start_xyz).tolist(),
                         "net_displacement_norm_m": float(np.linalg.norm(view.position - self.start_xyz)),
                         "legs": self.legs, "steps": self.actions_executed, "axes": axes,
                         "amplitude_m": self.amplitude_m, "gripper": self.gripper,
                         "touch_threshold_mm": self.touch_threshold_mm if self.gripper == "open" else None,
                         "thresholds": thresholds_dicts([th_value("D2_probe_m", self.amplitude_m, "DD",
                                                                  "D2_PROBE_M hover probe amplitude"),
                                                         *([self.g3] if self.gripper == "open" else []),
                                                         *self.g12])})


# ------------------------------------------------------------------------------------------------ twists
class TwistSegment(Primitive):
    """Spec §4.9 Turn steps 2-3 (K2; K1w decided by the injected wedge observer), contract §C8 [C-29].

    Each step targets the current orientation rotated by ``sense`` x K2_TWIST_RAD about R2 (HoldingActionAdapter
    capped at K2), holds the segment-start position softly (correction clipped to TWIST_HOLD_CAP_M per axis, [C-10])
    and keeps the gripper closed. After every step the wedge, part-rotation and (optional, [C-27]) effect observers
    are updated; the decision rules, evaluated every step, first match wins:
    1. hold floor (gap < ``hold_floor_mm``) or wedge violated ``hold_lost`` -> failed ``hold_lost``;
    2. wedge violated ``turn_slip`` -> failed ``turn_slip``;
    3. the yaw budget left for this sense below one K2 command -> failed ``yaw_budget``;
    4. wedge violated ``blocked_sense`` -> failed ``blocked_sense``; wedge met ``free_sense`` -> succeeded
       ``free_sense`` unless the part does not co-rotate (round part: failed ``blocked_sense``, flag
       ``blocked_by_part_rotation``); wedge pending with class ``undecided`` (the extend step) -> the part-rotation
       observer decides (corotating -> ``free_sense``, part_not_corotating -> ``blocked_sense``, otherwise failed
       ``sense_undecided`` with flag ``sense_by_wedge_only``); wedge failed -> failed ``sense_undecided``;
    5. safety cap TWIST_SEGMENT_MAX_STEPS without a wedge decision -> failed ``sense_undecided``.
    Evidence ``decided_by`` (TWIST_DECIDED_BY): ``wedge`` / ``part_rotation`` / ``both`` for the observer rows (contract
    §C8); ``hold_floor`` (the segment's own gap check, row 1), ``budget`` (row 3) and ``cap`` (row 5) name the rule
    when no observer decided."""

    def __init__(self, view: StepView, *, sense: int, wedge, part_rotation, hold_floor_mm: float,
                 yaw_budget_left_rad: float, robot: RobotModel, effect=None, name: str = "twist_segment"):
        super().__init__(name)
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        if wedge is None:
            raise ValueError("TwistSegment needs a wedge observer")
        self.sense = _sense(sense)
        self.wedge, self.part_rotation, self.effect = wedge, part_rotation, effect
        self.robot = robot
        self.hold_floor_mm = _finite(hold_floor_mm, "hold_floor_mm")
        self.budget_start_rad = _finite(yaw_budget_left_rad, "yaw_budget_left_rad")
        self.anchor = view.position.copy()
        self.start_xyzw = view.quaternion_xyzw.copy()
        self.adapter = HoldingActionAdapter(max_rotation_rad=K2_TWIST_RAD)
        self.yaw_rad = 0.0
        self.gaps_mm = [view.gap_mm]
        self.rates_mm: list[float] = []
        self.yaw_steps_rad: list[float] = []
        self.max_correction_m = 0.0
        self.wedge_report = self.rotation_report = self.effect_report = None
        self.decided_by: str | None = None
        self.flags: list[str] = []
        self.thresholds = [th_value("K2_twist_rad", K2_TWIST_RAD, "DD", "K2_TWIST_RAD twist command per step, below "
                                    "the R10 ROBOT cap"),
                           th_value("K6_yaw_budget_left_rad", self.budget_start_rad, "ROBOT",
                                    "R12 yaw budget left about the approach for this sense at the segment start")]
        wedge.start(view)
        if part_rotation is not None:
            part_rotation.start(view)
        if view.gap_mm < self.hold_floor_mm:
            self._end("failed", "hold_lost", "hold_floor")
        elif self.budget_left_rad < K2_TWIST_RAD:
            self._end("failed", "yaw_budget", "budget")

    @property
    def budget_left_rad(self) -> float:
        return self.budget_start_rad - self.sense * self.yaw_rad

    def _next(self, view):
        action, correction = _twist_action(self.adapter, view, self.sense, self.anchor)
        self.max_correction_m = max(self.max_correction_m, float(np.max(np.abs(correction))))
        return action

    def _observe(self, before, action, after):
        dyaw = self.robot.yaw_about_approach(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2])
        self.yaw_rad += dyaw
        self.yaw_steps_rad.append(dyaw)
        self.rates_mm.append(_gap_rate(before, after))
        self.gaps_mm.append(after.gap_mm)
        context = {"sense": self.sense, "theta_hand_rad": self.sense * self.yaw_rad, "primitive": self.name}
        self.wedge_report = self.wedge.update(before, action, after, context)
        if self.part_rotation is not None:
            self.rotation_report = self.part_rotation.update(before, action, after, context)
        if self.effect is not None:
            self.effect_report = self.effect.update(before, action, after, {"theta_part_rad": None})
        decision = self._decide(after)
        if decision is not None:
            self._end(*decision)
        return {"yaw_step_rad": dyaw, "gap_mm": after.gap_mm, "rate_mm": self.rates_mm[-1],
                "wedge": _verdict(self.wedge_report), "part_rotation": _rotation_state(self.rotation_report)}

    def _decide(self, after):
        verdict, reason = _verdict(self.wedge_report), _reason(self.wedge_report)
        rotation = _rotation_state(self.rotation_report)
        if verdict == "violated" and reason == "hold_lost":
            return "failed", "hold_lost", "wedge"
        if after.gap_mm < self.hold_floor_mm:
            return "failed", "hold_lost", "hold_floor"
        if verdict == "violated" and reason == "turn_slip":
            return "failed", "turn_slip", "wedge"
        if self.budget_left_rad < K2_TWIST_RAD:
            return "failed", "yaw_budget", "budget"
        if verdict == "violated" and reason == "blocked_sense":
            return "failed", "blocked_sense", "both" if rotation == "part_not_corotating" else "wedge"
        if verdict == "met":
            if rotation == "part_not_corotating":
                self.flags.append("blocked_by_part_rotation")
                return "failed", "blocked_sense", "part_rotation"
            return "succeeded", "free_sense", "both" if rotation == "corotating" else "wedge"
        if verdict == "failed":
            return "failed", "sense_undecided", "wedge"
        undecided = _evidence(self.wedge_report).get("class") == "undecided" or reason == "sense_undecided"
        if verdict == "pending" and undecided:
            if rotation == "corotating":
                return "succeeded", "free_sense", "part_rotation"
            if rotation == "part_not_corotating":
                return "failed", "blocked_sense", "part_rotation"
            self.flags.append("sense_by_wedge_only")
            return "failed", "sense_undecided", "wedge"
        if verdict == "violated":
            return "failed", "sense_undecided", "wedge"
        if self.actions_executed >= TWIST_SEGMENT_MAX_STEPS:
            self.flags.append("segment_cap")
            return "failed", "sense_undecided", "cap"
        return None

    def _end(self, status, reason, decided_by):
        self.decided_by = decided_by
        self._finish(status, reason, self._evidence_now())

    def _evidence_now(self) -> dict:
        evidence = {"sense": self.sense, "steps": self.actions_executed, "yaw_realized_rad": self.yaw_rad,
                    "theta_sense_rad": self.sense * self.yaw_rad, "yaw_steps_rad": self.yaw_steps_rad,
                    "gap_trace_mm": self.gaps_mm, "rates_mm_per_step": self.rates_mm,
                    "wedge": _report(self.wedge_report), "part_rotation": _report(self.rotation_report),
                    "effect": _report(self.effect_report), "decided_by": self.decided_by, "flags": self.flags,
                    "yaw_budget_left_rad": self.budget_left_rad, "max_position_correction_m": self.max_correction_m,
                    "anchor_xyz": self.anchor.tolist(), "thresholds": thresholds_dicts(self.thresholds)}
        evidence.update({flag: True for flag in self.flags})
        return jsonable(evidence)


class HingeTwist(Primitive):
    """Spec §4.9 Turn steps 4-8 after the free classification, contract §C8 [C-29].

    theta_part = the hand yaw about R2 accumulated in this primitive times ``sense`` (plus ``theta_start_rad`` for a
    resumed turn). Each step: K2 about R2 in ``sense``, position soft-held at the start grip or, once the hinge centre
    estimate lies more than K3 from it, commanded on the orbit p = c + Rz(phi)(p0 - c) (phi = the measured rotation
    about world z since the start), clipped per axis to TWIST_HOLD_CAP_M. The centre starts at ``centre_prior_xy``; it
    is refit from the realized grip path every TWIST_REFIT_EVERY steps and a refit replaces it only when its sigma is
    below sigma_c [C-49]. ``effect.update`` receives {"theta_part_rad": theta_part}; the onset angle theta_e is the
    one the effect report carries (first frame of the O1 run, [C-28]). Decision rules (every step, first match wins):
    1. gap < ``hold_floor_mm`` or wedge_end violated hold_lost -> failed ``hold_lost``;
    2. K11 tug for K11_STEPS steps -> failed ``axis_tug``;
    3. wedge_end met (K4w end stop) -> succeeded ``end_stop``;
    4. wedge_end violated turn_slip, or part_not_corotating with a gap fall -> failed ``turn_slip``;
    5. part_not_corotating with a flat gap: ``wedge_capable`` -> failed ``turn_slip``; otherwise succeeded ``end_stop``
       with flag ``part_stop_without_wedge``; part_not_corotating with a rising gap before K4w confirms -> succeeded
       ``end_stop`` (spec §5.2 K4w: "or a part stop from K7"; the gap trend is the mean rate over the last
       PART_STOP_WINDOW_STEPS steps against +-PART_STOP_FLAT_LW l w);
    6. effect onset at theta_e and theta_part >= theta_e + max(K5_FLOOR_RAD, K5_FRACTION theta_e) -> succeeded
       ``effect_margin_reached``;
    7. ``theta_target_rad`` reached -> succeeded ``quarter_turn_reached`` / ``half_turn_reached``;
    8. the yaw budget left below one K2 command -> failed ``yaw_budget``;
    9. theta_part progress < HAND_YAW_STALL_RAD over HAND_YAW_STALL_STEPS -> failed ``hand_yaw_stall``.
    Evidence ``decided_by`` (HINGE_DECIDED_BY) names the deciding observer (``wedge``, ``part_rotation``, ``both``,
    ``effect``) or rule (``hold_floor``, ``tug``, ``rotation`` for the target, ``budget``, ``stall``). theta_part is
    the hand-based angle at the stop: the part can coast on inside the pads afterwards (see ReleaseVerify
    ``settle_first``), so the final part angle is larger and is read from a part observable after the settle."""

    def __init__(self, view: StepView, *, sense: int, wedge_end, effect, part_rotation, hold_floor_mm: float,
                 theta_target_rad: float | None, yaw_budget_left_rad: float, centre_prior_xy, sigma_c_m: float,
                 wedge_capable: bool, robot: RobotModel, name: str = "hinge_twist",
                 theta_start_rad: float | None = None, align_tol_m: float | None = None):
        super().__init__(name)
        if not isinstance(robot, RobotModel):
            raise ValueError("robot must be a RobotModel")
        if wedge_end is None:
            raise ValueError("HingeTwist needs an end-stop wedge observer")
        self.sense = _sense(sense)
        self.wedge, self.effect, self.part_rotation = wedge_end, effect, part_rotation
        self.robot = robot
        self.hold_floor_mm = _finite(hold_floor_mm, "hold_floor_mm")
        self.theta_target_rad = None if theta_target_rad is None else _positive(theta_target_rad, "theta_target_rad")
        self.budget_start_rad = _finite(yaw_budget_left_rad, "yaw_budget_left_rad")
        self.wedge_capable = bool(wedge_capable)
        self.theta_start_rad = 0.0 if theta_start_rad is None else _finite(theta_start_rad, "theta_start_rad")
        prior = np.asarray(centre_prior_xy, dtype=float).reshape(-1)[:2]
        if prior.shape != (2,) or not np.isfinite(prior).all():
            raise ValueError("centre_prior_xy must be a finite world xy")
        self.p0 = view.position.copy()
        self.q0 = view.quaternion_xyzw.copy()
        self.centre = prior.copy()
        self.centre_source = "prior"
        self.sigma_c = th_max("K11_sigma_c_m", Term("ONLINE", _finite(sigma_c_m, "sigma_c_m"), "sigma_c (fit, drag)"),
                              Term("DD", K11_SIGMA_C_FLOOR_M, "K11_SIGMA_C_FLOOR_M"), unit="m")
        terms = [Term("DD", K3_FLOOR_M, "K3_FLOOR_M")]
        if align_tol_m is not None:
            terms.append(Term("ONLINE", _finite(align_tol_m, "align_tol_m"), "Align tolerance"))
        self.k3 = th_max("K3_orbit_offset_m", *terms, unit="m")
        self.adapter = HoldingActionAdapter(max_rotation_rad=K2_TWIST_RAD)
        self.yaw_used_rad = 0.0
        self.theta_history = [self.theta_part_rad]
        self.path_xy = [self.p0[:2].copy()]
        self.phi_z = [0.0]
        self.gaps_mm = [view.gap_mm]
        self.rates_mm: list[float] = []
        self.omega_rad: list[float] = []
        self.centre_fits: list[dict] = []
        self.sigma_pos_m = 0.0
        self.tug_steps = 0
        self.tug_trace: list[dict] = []
        self.orbit_follow = self._orbit_far()
        self.orbit_steps = 0
        self.max_correction_m = 0.0
        self.theta_e_rad: float | None = None
        self.onset_step: int | None = None
        self.k5: Threshold | None = None
        self.wedge_report = self.rotation_report = self.effect_report = None
        self.decided_by: str | None = None
        self.flags: list[str] = []
        self.gap_trend: str | None = None
        wedge_end.start(view)
        if part_rotation is not None:
            part_rotation.start(view)
        if view.gap_mm < self.hold_floor_mm:
            self._end("failed", "hold_lost", "hold_floor")
        elif self.budget_left_rad < K2_TWIST_RAD:
            self._end("failed", "yaw_budget", "budget")

    # ------------------------------------------------------------------ state
    @property
    def theta_part_rad(self) -> float:
        return self.theta_start_rad + self.sense * self.yaw_used_rad

    @property
    def budget_left_rad(self) -> float:
        return self.budget_start_rad - self.sense * self.yaw_used_rad

    def _orbit_far(self) -> bool:
        return float(np.linalg.norm(self.centre - self.p0[:2])) > self.k3.value

    def _expected_xy(self, phi: float) -> np.ndarray:
        return self.centre + _rot2(phi) @ (self.p0[:2] - self.centre)

    # ------------------------------------------------------------------ control
    def _next(self, view):
        if self.orbit_follow:
            phi = twist_about(self.q0, view.quaternion_xyzw, WORLD_UP)
            target = np.r_[self._expected_xy(phi), self.p0[2]]
        else:
            target = self.p0
        action, correction = _twist_action(self.adapter, view, self.sense, target)
        self.max_correction_m = max(self.max_correction_m, float(np.max(np.abs(correction))))
        return action

    def _observe(self, before, action, after):
        dyaw = self.robot.yaw_about_approach(before.quaternion_xyzw, after.quaternion_xyzw, before.R[:, 2])
        self.yaw_used_rad += dyaw
        self.omega_rad.append(abs(dyaw))
        self.theta_history.append(self.theta_part_rad)
        phi = twist_about(self.q0, after.quaternion_xyzw, WORLD_UP)
        self.path_xy.append(after.position[:2].copy())
        self.phi_z.append(phi)
        self.rates_mm.append(_gap_rate(before, after))
        self.gaps_mm.append(after.gap_mm)
        if self.orbit_follow:
            self.orbit_steps += 1
        context = {"sense": self.sense, "theta_hand_rad": self.sense * self.yaw_used_rad,
                   "theta_part_rad": self.theta_part_rad, "primitive": self.name}
        self.wedge_report = self.wedge.update(before, action, after, context)
        if self.part_rotation is not None:
            self.rotation_report = self.part_rotation.update(before, action, after, context)
        if self.effect is not None:
            self.effect_report = self.effect.update(before, action, after, {"theta_part_rad": self.theta_part_rad})
            self._latch_onset()
        if self.actions_executed % TWIST_REFIT_EVERY == 0:
            self._refit()
        deviation, limit = self._tug(after, phi)
        decision = self._decide(after)
        if decision is not None:
            self._end(*decision)
        return {"theta_part_rad": self.theta_part_rad, "gap_mm": after.gap_mm, "rate_mm": self.rates_mm[-1],
                "orbit_follow": self.orbit_follow, "tug_deviation_m": deviation, "tug_limit_m": limit,
                "wedge": _verdict(self.wedge_report), "part_rotation": _rotation_state(self.rotation_report),
                "effect": _verdict(self.effect_report)}

    def _latch_onset(self) -> None:
        if self.theta_e_rad is not None or _verdict(self.effect_report) != "met":
            return
        evidence = _evidence(self.effect_report)
        theta = evidence.get("theta_e_rad")
        self.theta_e_rad = max(_finite(theta, "theta_e_rad") if theta is not None else self.theta_part_rad, 0.0)
        step = evidence.get("onset_step")
        self.onset_step = int(step) if isinstance(step, (int, np.integer)) else self.actions_executed
        self.k5 = th_max("K5_margin_rad", Term("DD", K5_FLOOR_RAD, "K5_FLOOR_RAD"),
                         Term("DD", K5_FRACTION * self.theta_e_rad, "K5_FRACTION*theta_e"), unit="rad")

    def _refit(self) -> None:
        fit = fit_hinge_centre(self.path_xy, self.phi_z)
        if fit is None:
            return
        accepted = fit["sigma_m"] < self.sigma_c.value
        fit.update(step=self.actions_executed, accepted=accepted, sigma_c_m=self.sigma_c.value)
        self.centre_fits.append(fit)
        if accepted:
            self.centre = np.asarray(fit["centre_xy"], dtype=float)
            self.centre_source = "refit"
            self.sigma_pos_m = float(fit["rms_m"])
            self.orbit_follow = self._orbit_far()

    def _tug(self, after: StepView, phi: float) -> tuple[float, float]:
        deviation = float(np.linalg.norm(after.position[:2] - self._expected_xy(phi)))
        limit = max(K11_FLOOR_M, 2.0 * abs(math.sin(phi / 2.0)) * self.sigma_c.value + K11_SIGMA_K * self.sigma_pos_m)
        self.tug_steps = self.tug_steps + 1 if deviation > limit else 0
        self.tug_trace.append({"step": self.actions_executed, "deviation_m": deviation, "limit_m": limit})
        return deviation, limit

    def _gap_trend(self) -> str:
        window = self.rates_mm[-PART_STOP_WINDOW_STEPS:]
        omega = self.omega_rad[-PART_STOP_WINDOW_STEPS:]
        lw = self.robot.pad_length_m * 1000.0 * float(np.mean(omega)) if omega else 0.0
        mean = float(np.mean(window)) if window else 0.0
        if mean < -PART_STOP_FLAT_LW * lw:
            return "fall"
        return "rise" if mean > PART_STOP_FLAT_LW * lw else "flat"

    def _decide(self, after):
        verdict, reason = _verdict(self.wedge_report), _reason(self.wedge_report)
        rotation = _rotation_state(self.rotation_report)
        if verdict == "violated" and reason == "hold_lost":
            return "failed", "hold_lost", "wedge"
        if after.gap_mm < self.hold_floor_mm:
            return "failed", "hold_lost", "hold_floor"
        if self.tug_steps >= K11_STEPS:
            return "failed", "axis_tug", "tug"
        if verdict == "met":
            return "succeeded", "end_stop", "both" if rotation == "part_not_corotating" else "wedge"
        if verdict == "violated" and reason == "turn_slip":
            return "failed", "turn_slip", "wedge"
        if rotation == "part_not_corotating":
            self.gap_trend = self._gap_trend()
            if self.gap_trend == "fall":
                return "failed", "turn_slip", "part_rotation"
            if self.gap_trend == "flat":
                if self.wedge_capable:
                    return "failed", "turn_slip", "part_rotation"
                self.flags.append("part_stop_without_wedge")
                return "succeeded", "end_stop", "part_rotation"
            return "succeeded", "end_stop", "part_rotation"         # rise: spec §5.2 K4w "or a part stop from K7"
        if self.k5 is not None and self.theta_part_rad >= self.theta_e_rad + self.k5.value:
            return "succeeded", "effect_margin_reached", "effect"
        if self.theta_target_rad is not None and self.theta_part_rad >= self.theta_target_rad:
            quarter = abs(self.theta_target_rad - math.pi / 2) <= abs(self.theta_target_rad - math.pi)
            return "succeeded", "quarter_turn_reached" if quarter else "half_turn_reached", "rotation"
        if self.budget_left_rad < K2_TWIST_RAD:
            return "failed", "yaw_budget", "budget"
        if len(self.theta_history) > HAND_YAW_STALL_STEPS:
            progress = self.theta_history[-1] - self.theta_history[-1 - HAND_YAW_STALL_STEPS]
            if progress < HAND_YAW_STALL_RAD:
                return "failed", "hand_yaw_stall", "stall"
        return None

    def _end(self, status, reason, decided_by):
        self.decided_by = decided_by
        self._finish(status, reason, self._evidence_now())

    def _evidence_now(self) -> dict:
        thresholds = [self.k3, self.sigma_c,
                      th_value("K2_twist_rad", K2_TWIST_RAD, "DD", "K2_TWIST_RAD twist command per step"),
                      th_value("K6_yaw_budget_left_rad", self.budget_start_rad, "ROBOT",
                               "R12 yaw budget left about the approach for this sense at the start")]
        if self.k5 is not None:
            thresholds.append(self.k5)
        evidence = {"sense": self.sense, "steps": self.actions_executed, "theta_part_rad": self.theta_part_rad,
                    "theta_start_rad": self.theta_start_rad, "theta_e_rad": self.theta_e_rad,
                    "onset_step": self.onset_step, "theta_target_rad": self.theta_target_rad,
                    "yaw_used_rad": self.yaw_used_rad, "yaw_budget_left_rad": self.budget_left_rad,
                    "centre_xy": self.centre.tolist(), "centre_source": self.centre_source,
                    "centre_fits": self.centre_fits, "orbit_follow": {"engaged": self.orbit_follow,
                                                                      "steps": self.orbit_steps},
                    "tug": {"steps": self.tug_steps, "max_deviation_m": max((t["deviation_m"] for t in self.tug_trace),
                                                                           default=0.0),
                            "trace": self.tug_trace[-K11_STEPS:], "sigma_pos_m": self.sigma_pos_m},
                    "gap_trace_mm": self.gaps_mm, "rates_mm_per_step": self.rates_mm, "gap_trend": self.gap_trend,
                    "wedge_capable": self.wedge_capable, "effect": _report(self.effect_report),
                    "wedge": _report(self.wedge_report), "part_rotation": _report(self.rotation_report),
                    "decided_by": self.decided_by, "flags": self.flags,
                    "max_position_correction_m": self.max_correction_m, "thresholds": thresholds_dicts(thresholds)}
        evidence.update({flag: True for flag in self.flags})
        return jsonable(evidence)


# ------------------------------------------------------------------------------------------------ held moves
class HeldReach(WaypointReach):
    """G7 / M3 held segment (v1 ``_held_reach`` pattern, agent_speedup_v1.py:857-865): v1 WaypointReach with the
    gripper closed, the hold floor and the grasp-gap reference, holding the grasp orientation when it is within
    M3_REORIENT_RAD of the current one (else the current one). The held-contact limits (instance ``GAP_RISE_MM`` /
    ``GAP_STEP_RISE_MM``) start at the G7 floors and, after G7_SIGMA_STEPS free steps, rise to
    max(floor, k sigma_carry) with sigma_carry the std of those steps' gap changes. Otherwise identical to
    WaypointReach; the evidence adds {"sigma_carry_mm", "G7_thresholds", "grasp_orientation_held"}."""

    def __init__(self, view: StepView, offset_m, *, hold_floor_mm: float, gap_ref_mm: float, grasp_quaternion=None,
                 name: str = "held_reach", **reach_kwargs):
        for key in ("gripper", "orientation_xyzw"):
            if key in reach_kwargs:
                raise ValueError(f"HeldReach sets {key} itself")
        self.GAP_RISE_MM = G7_TOTAL_FLOOR_MM
        self.GAP_STEP_RISE_MM = G7_STEP_FLOOR_MM
        self.sigma_carry_mm: float | None = None
        self._carry_steps_mm: list[float] = []
        self.g7_thresholds = [
            th_value("G7_carry_total_rise_mm", G7_TOTAL_FLOOR_MM, "DD", "G7_TOTAL_FLOOR_MM until sigma_carry is measured"),
            th_value("G7_carry_step_rise_mm", G7_STEP_FLOOR_MM, "DD", "G7_STEP_FLOOR_MM until sigma_carry is measured")]
        orientation = None
        if grasp_quaternion is not None and quaternion_angle_rad(view.quaternion_xyzw, grasp_quaternion) <= M3_REORIENT_RAD:
            orientation = np.asarray(grasp_quaternion, dtype=float)
        self.grasp_orientation_held = orientation is not None
        super().__init__(view, offset_m, gripper="close", name=name, orientation_xyzw=orientation,
                         hold_floor_mm=hold_floor_mm, gap_ref_mm=gap_ref_mm, **reach_kwargs)

    def _observe(self, before, action, after):
        self._carry_steps_mm.append(_gap_rate(before, after))
        feedback = super()._observe(before, action, after)
        if self.sigma_carry_mm is None and not self.done and len(self._carry_steps_mm) >= G7_SIGMA_STEPS:
            sigma = float(np.std(self._carry_steps_mm[:G7_SIGMA_STEPS]))
            self.sigma_carry_mm = sigma
            total = th_max("G7_carry_total_rise_mm", Term("DD", G7_TOTAL_FLOOR_MM, "G7_TOTAL_FLOOR_MM"),
                           Term("ONLINE", G7_TOTAL_SIGMA_K * sigma, "G7_TOTAL_SIGMA_K*sigma_carry"), unit="mm")
            step = th_max("G7_carry_step_rise_mm", Term("DD", G7_STEP_FLOOR_MM, "G7_STEP_FLOOR_MM"),
                          Term("ONLINE", G7_STEP_SIGMA_K * sigma, "G7_STEP_SIGMA_K*sigma_carry"), unit="mm")
            self.GAP_RISE_MM, self.GAP_STEP_RISE_MM = total.value, step.value
            self.g7_thresholds = [total, step]
        return feedback

    def _evidence_now(self, reason: str) -> dict:
        evidence = super()._evidence_now(reason)
        evidence.update(jsonable({"sigma_carry_mm": getattr(self, "sigma_carry_mm", None),
                                  "G7_thresholds": thresholds_dicts(getattr(self, "g7_thresholds", ())),
                                  "grasp_orientation_held": getattr(self, "grasp_orientation_held", False)}))
        return evidence


class GuardedLower(AnchoredMove):
    """Spec §4.9 GuardedLower (G8, M5, P15), an AnchoredMove straight down with the gripper closed and the lateral
    hold on the aim (``aim_grip_xy``) at the P15 lateral cap.

    - ``phase="fast"``: command LOWER_FAST_COMMAND_M; a support cue (stall, touch, G8 gap rise) only hands over:
      succeeded ``handover_<cue>``; the travel -> succeeded ``travel_reached``.
    - ``phase="slow"``: command P15_COMMAND_M; stall / touch -> succeeded ``contact_stall`` / ``contact_touch``; a
      per-step gap rise >= max(G8_FLOOR_MM, G8_SIGMA_K sigma_lower) (sigma_lower from the first ``sigma_window``
      steps) -> succeeded ``gap_rise``; the travel -> succeeded ``bound_reached``.
    - both: |grip xy - aim| > ``drift_abort_m`` -> failed ``lateral_drift``; M5_SATURATED_STEPS consecutive steps
      with the lateral correction at its cap and <= M5_SATURATED_REALIZED of it realized -> failed
      ``lateral_unreachable``; gap < ``hold_floor_mm`` -> failed ``hold_lost``; no progress within the budget ->
      failed ``budget``.
    - a start already farther than ``drift_abort_m`` from the aim fails ``lateral_drift`` at construction (0 actions)
      with evidence ``drift_at_start`` True: an aim error before any lowering, which the relation re-aims without
      the lift of a drift recovery.
    The evidence records G8, M5, P15 and G12 (the stall limit of the contact stop)."""

    def __init__(self, view: StepView, *, phase: str, travel_m: float, aim_grip_xy, drift_abort_m: float,
                 hold_floor_mm: float, sigma_window: int = G8_SIGMA_STEPS, name: str | None = None):
        if phase not in LOWER_PHASES:
            raise ValueError(f"phase must be one of {LOWER_PHASES}")
        aim = np.asarray(aim_grip_xy, dtype=float).reshape(-1)[:2]
        if aim.shape != (2,) or not np.isfinite(aim).all():
            raise ValueError("aim_grip_xy must be a finite world xy")
        if isinstance(sigma_window, (bool, np.bool_)) or not isinstance(sigma_window, (int, np.integer)) \
                or sigma_window < 1:
            raise ValueError("sigma_window must be a positive integer")
        self.phase = phase
        self.aim_xy = aim
        self.drift_abort_m = _positive(drift_abort_m, "drift_abort_m")
        self.hold_floor_mm = _finite(hold_floor_mm, "hold_floor_mm")
        self.sigma_window = int(sigma_window)
        self._lower_steps_mm: list[float] = []
        self.sigma_lower_mm: float | None = None
        self.g8 = th_value("G8_lower_step_rise_mm", G8_FLOOR_MM, "DD", "G8_FLOOR_MM until sigma_lower is measured")
        self.saturated_steps = 0
        self.max_drift_m = 0.0
        self.drift_m = 0.0
        command = LOWER_FAST_COMMAND_M if phase == "fast" else P15_COMMAND_M
        super().__init__(view, axis=-np.asarray(WORLD_UP), travel_m=travel_m, gripper="close", command_m=command,
                         stop_on_contact=True, lateral_cap_m=P15_LATERAL_CAP_M,
                         lateral_target=(aim[0], aim[1], float(view.position[2])), name=name or f"lower_{phase}")
        self.drift_m = self.max_drift_m = float(np.linalg.norm(view.position[:2] - aim))
        self.drift_at_start = self.drift_m > self.drift_abort_m
        if view.gap_mm < self.hold_floor_mm:
            self._finish("failed", "hold_lost", self._evidence_now("hold_lost"))
        elif self.drift_m > self.drift_abort_m:
            self._finish("failed", "lateral_drift", self._evidence_now("lateral_drift"))

    def _pre_event(self, before, after):
        step = _gap_rate(before, after)
        self._lower_steps_mm.append(step)
        if self.sigma_lower_mm is None and len(self._lower_steps_mm) >= self.sigma_window:
            self.sigma_lower_mm = float(np.std(self._lower_steps_mm[:self.sigma_window]))
            self.g8 = th_max("G8_lower_step_rise_mm", Term("DD", G8_FLOOR_MM, "G8_FLOOR_MM"),
                             Term("ONLINE", G8_SIGMA_K * self.sigma_lower_mm, "G8_SIGMA_K*sigma_lower"), unit="mm")
        self.drift_m = float(np.linalg.norm(after.position[:2] - self.aim_xy))
        self.max_drift_m = max(self.max_drift_m, self.drift_m)
        offset_before = float(np.linalg.norm(self.lateral_offset(before.position)))
        offset_after = float(np.linalg.norm(self.lateral_offset(after.position)))
        saturated = offset_before > self.lateral_cap_m
        unrealized = offset_before - offset_after <= M5_SATURATED_REALIZED * self.lateral_cap_m
        self.saturated_steps = self.saturated_steps + 1 if saturated and unrealized else 0
        if after.gap_mm < self.hold_floor_mm:
            return "hold_lost"
        if self.drift_m > self.drift_abort_m:
            return "lateral_drift"
        if self.saturated_steps >= M5_SATURATED_STEPS:
            return "lateral_unreachable"
        if step >= self.g8.value:
            return "gap_rise"
        return None

    def _resolve(self, event):
        if event in ("hold_lost", "lateral_drift", "lateral_unreachable", "budget"):
            return "failed", event
        if self.phase == "fast":
            if event == "travel_reached":
                return "succeeded", "travel_reached"
            return "succeeded", HANDOVER_PREFIX + event
        if event == "travel_reached":
            return "succeeded", "bound_reached"
        if event in SUPPORT_CUES:
            return "succeeded", event
        return "failed", event

    def _evidence_now(self, event: str | None = None) -> dict:
        self._last_event = event
        return super()._evidence_now(event)

    def _extra_evidence(self):
        thresholds = [self.g8,
                      th_value("M5_drift_abort_m", self.drift_abort_m, "ONLINE", "M5 drift abort passed in (planner)"),
                      th_value("M5_saturated_steps", M5_SATURATED_STEPS, "DD", "M5_SATURATED_STEPS"),
                      *p15_thresholds(), g12_stall_threshold(self)]
        event = getattr(self, "_last_event", None)
        return jsonable({"phase": self.phase, "aim_grip_xy": self.aim_xy, "drift_m": self.drift_m,
                         "max_drift_m": self.max_drift_m, "drift_abort_m": self.drift_abort_m,
                         "drift_at_start": bool(getattr(self, "drift_at_start", False)),
                         "sigma_lower_mm": self.sigma_lower_mm, "saturated_steps": self.saturated_steps,
                         "hold_floor_mm": self.hold_floor_mm, "event_grip_z": float(self.end_xyz[2]),
                         "support_cue": event if event in SUPPORT_CUES else None,
                         "thresholds": thresholds_dicts(thresholds)})


class Touchdown(GuardedLower):
    """Spec §4.7 item 6: a slow GuardedLower back onto the pick support, travel <= ``lift_m`` + TOUCHDOWN_EXTRA_M,
    aimed at the current grip xy unless ``aim_grip_xy`` is given. ``touch_grip_z`` is the grip z at the support event
    (b_low = g_touch - support_z(pick) is the caller's)."""

    def __init__(self, view: StepView, *, lift_m: float, hold_floor_mm: float, drift_abort_m: float, aim_grip_xy=None,
                 sigma_window: int = G8_SIGMA_STEPS, name: str = "touchdown"):
        self.lift_m = _positive(lift_m, "lift_m")
        aim = view.position[:2].copy() if aim_grip_xy is None else aim_grip_xy
        super().__init__(view, phase="slow", travel_m=self.lift_m + TOUCHDOWN_EXTRA_M, aim_grip_xy=aim,
                         drift_abort_m=drift_abort_m, hold_floor_mm=hold_floor_mm, sigma_window=sigma_window,
                         name=name)

    @property
    def touch_grip_z(self) -> float | None:
        if self.status == "succeeded" and self.reason in SUPPORT_CUES:
            return float(self.end_xyz[2])
        return None

    def _extra_evidence(self):
        evidence = super()._extra_evidence()
        cue = evidence.get("support_cue")
        evidence["touchdown"] = jsonable({"lift_m": self.lift_m, "travel_bound_m": self.travel_m,
                                          "touch_grip_z": float(self.end_xyz[2]) if cue is not None else None,
                                          "extra_m": TOUCHDOWN_EXTRA_M})
        return evidence


class ReleaseVerify(Primitive):
    """Release and retreat (v1 ``_release_and_back_off`` pattern, agent_speedup_v1.py:986-997; M12, M13): an open
    settle (OPEN_SETTLE_MIN_ACTIONS), then, if the gap reached g_open - M12_OPEN_TOL_MM, a retreat of M13_RETREAT_M
    along -R2 with the gripper open (stop on contact; finger touch at ``touch_threshold_mm``: the calibrated G3, a
    number or the ``gripper_quantities`` Threshold, default the G3 floor). Succeeded ``released``; failed
    ``release_incomplete`` (the fingers did not open: no retreat, the object would be dragged) or
    ``touch_on_retreat`` (a stall, a finger touch or no progress on the retreat). The evidence records M12, M13 and,
    once the retreat ran, G3 and the retreat's G12 stall limit.

    ``settle_first=True`` adds a closed B4 settle before the open (stage ``settle``): the hand and the gripped part
    come to rest first. [F, real LIBERO rotary part, Wave-1 primitive check] released straight after a twist the part
    coasted on from q 0.64 to 1.00 rad (its own low damping) while the hand stood still; after a 9-step closed settle
    it stayed within 0.0003 rad through the release, so a spring-back check after a turn needs it. During that
    closed settle the part still coasts about 0.11 rad inside the pads while the hand yaws only about 0.01 rad (same
    check, three resets), so the hand-based theta_part of the twist understates the final part angle and the hand
    yaw cannot correct it: a final part angle needs a part observable read after the settle. The evidence carries
    ``settle_hand_yaw_rad``, the signed hand yaw about the start approach R2 over the settle (contract sense
    convention), as the record of what the hand did. Every transition goes to
    ``recorder.on_transition("release_settle" | "release_open" | "release_retreat", ...)`` when a recorder is given."""

    def __init__(self, view: StepView, *, g_open_mm: float, recorder=None, name: str = "release",
                 touch_threshold_mm=None, settle_first: bool = False):
        super().__init__(name)
        self.g_open_mm = _positive(g_open_mm, "g_open_mm")
        self.recorder = recorder
        self.g3 = touch_threshold(touch_threshold_mm)
        self.touch_threshold_mm = self.g3.value
        self.g12: Threshold | None = None
        self.settle_hand_yaw_rad: float | None = None
        self.settle_first = bool(settle_first)
        self.m12 = th_value("M12_released_gap_mm", self.g_open_mm - M12_OPEN_TOL_MM, "ONLINE",
                            "g_open - M12_OPEN_TOL_MM", g_open_mm=self.g_open_mm, M12_OPEN_TOL_MM=M12_OPEN_TOL_MM)
        self.m13 = th_value("M13_retreat_m", M13_RETREAT_M, "DD", "M13_RETREAT_M retreat along -R2")
        self.settle_outcome: dict | None = None
        self.open_outcome: dict | None = None
        self.retreat_outcome: dict | None = None
        self.start_xyz = view.position.copy()
        self.start_xyzw = view.quaternion_xyzw.copy()
        if self.settle_first:
            self.stage, self.current = "settle", b4_settle(view, gripper="close", name=f"{name}_settle")
        else:
            self.stage, self.current = "open", self._open(view)

    def _open(self, view: StepView) -> Primitive:
        return GripperSettlePrimitive("open", view, config=GripperSettleConfig(min_open_actions=OPEN_SETTLE_MIN_ACTIONS),
                                      name=f"{self.name}_open")

    def _next(self, view):
        return self.current.next_action(view)

    def _observe(self, before, action, after):
        stage, current = self.stage, self.current
        current.observe_transition(before, action, after)
        if self.recorder is not None:
            self.recorder.on_transition(f"release_{stage}", before, after)
        if current.done:
            if stage == "settle":
                self.settle_outcome = current.outcome()
                self.settle_hand_yaw_rad = twist_about(self.start_xyzw, after.quaternion_xyzw,
                                                       quaternion_approach_axis(self.start_xyzw))
                self.stage, self.current = "open", self._open(after)
            elif stage == "open":
                self.open_outcome = current.outcome()
                if after.gap_mm >= self.m12.value:
                    self.stage = "retreat"
                    self.current = AnchoredMove(after, axis=-after.R[:, 2], travel_m=M13_RETREAT_M, gripper="open",
                                                command_m=P15_COMMAND_M, lateral_cap_m=P15_LATERAL_CAP_M,
                                                stop_on_contact=True, touch_threshold_mm=self.touch_threshold_mm,
                                                name=f"{self.name}_retreat")
                else:
                    self._finish("failed", "release_incomplete", self._evidence_now(after))
            else:
                self.retreat_outcome = current.outcome()
                self.g12 = g12_stall_threshold(current, leg="retreat")
                if current.reason == "travel_reached":
                    self._finish("succeeded", "released", self._evidence_now(after))
                else:
                    self._finish("failed", "touch_on_retreat", self._evidence_now(after))
        return {"stage": stage, "gap_mm": after.gap_mm}

    def _evidence_now(self, view: StepView) -> dict:
        retreat = [] if self.g12 is None else [self.g3, self.g12]
        return jsonable({"settle": self.settle_outcome, "open": self.open_outcome, "retreat": self.retreat_outcome,
                         "gap_mm": view.gap_mm, "g_open_mm": self.g_open_mm, "settle_first": self.settle_first,
                         "settle_hand_yaw_rad": self.settle_hand_yaw_rad, "touch_threshold_mm": self.touch_threshold_mm,
                         "displacement_m": (view.position - self.start_xyz).tolist(), "steps": self.actions_executed,
                         "thresholds": thresholds_dicts([self.m12, self.m13, *retreat])})


def _gate_limit_mm(finger) -> float | None:
    limit = (finger or {}).get("R_m")
    return None if limit is None else 1000.0 * float(limit)


class GuardedClose(Primitive):
    """GC-1 guarded close (grasp-refusal decision J, remedy 2 with the judge's free-motion amendment).

    Every action is ``GripperSettlePrimitive("close", view, config=config)``'s, so an uninterrupted close IS the v1
    close (``self.settle`` is that inner close; classify_close reads it). After each transition, per finger i, the
    world pad-face position s_i along the gate's jaw, outward from the gate's evaluated grip point:
    s_1 = (xy - grip_xy) . jaw + q_1 - R9/2, s_2 = -(xy - grip_xy) . jaw + |q_2| - R9/2 (state8[6] is finger 1, on
    +jaw). The world inward motion s_i(before) - s_i(after) is compared with the calibrated free close of that finger
    at its OWN opening before the action (``free_mm``: the empty curve [g_open, empty_curve...] read at 2 q_i, so a
    close that starts from another opening than the calibration does not shift the prediction).

    Touch: at action k >= GC_FIRST_JUDGED_K, where the free motion is >= GC_FREE_MIN_MM, a shortfall >=
    max(GC_TOUCH_FLOOR_MM, GC_TOUCH_SIGMA_K sigma_close) (the first one per finger is kept). Verdicts: a touch beyond
    the gate's R_i is ``early_finger_contact``; both fingers touched with s_1 + s_2 > span_hi + tol is
    ``span_inconsistent``. ``active`` (a marginal plan with a gate): the next action starts
    ``GripperSettlePrimitive("open", min_open_actions=OPEN_SETTLE_MIN_ACTIONS)`` and the primitive then fails with
    the verdict's reason. Monitor-only (a clean plan, or no gate): the same evidence plus ``would_abort``; it never
    aborts. ``gate`` = the planner's ``checks["gate"]`` ({"fingers": {"finger1": {"R_m", ...}, "finger2": ...},
    "grip_xy", "jaw_xy", "span_hi_mm", "tol_mm", "pad_face_offset_mm"})."""

    def __init__(self, view: StepView, *, gripper: GripperCalibration, sigma_close_mm: float, gate, active: bool,
                 config: GripperSettleConfig | None = None, name: str = "grasp_close"):
        super().__init__(name)
        self.settle = GripperSettlePrimitive("close", view, config=config, name=name)
        self.opening: GripperSettlePrimitive | None = None
        g = dict(gate or {})
        self.active = bool(active) and bool(g)
        # the calibrated free close [g_open, empty_curve...] (G1, G2)
        self.curve_mm = np.array([float(gripper.open_gap_mm), *(float(v) for v in gripper.empty_curve_mm)],
                                 dtype=float)
        self.touch_th = th_max("GC_touch_mm", Term("DD", GC_TOUCH_FLOOR_MM, "GC_TOUCH_FLOOR_MM"),
                               Term("ONLINE", GC_TOUCH_SIGMA_K * _finite(sigma_close_mm, "sigma_close_mm"),
                                    "GC_TOUCH_SIGMA_K*sigma_close"), unit="mm")
        self.origin = np.asarray(g.get("grip_xy", view.position[:2]), dtype=float)
        jaw = np.asarray(g.get("jaw_xy", view.R[:2, 1]), dtype=float)
        self.jaw = jaw / max(float(np.linalg.norm(jaw)), 1e-9)
        self.offset_mm = float(g.get("pad_face_offset_mm", robot_facts.PAD_SEPARATION_OFFSET_MM / 2.0))
        fingers = g.get("fingers") or {}
        self.limit_mm = {i: _gate_limit_mm(fingers.get(f"finger{i}")) for i in (1, 2)}
        self.span_hi_mm = (None if g.get("span_hi_mm") is None
                           else float(g["span_hi_mm"]) + float(g.get("tol_mm") or 0.0))
        self.k = 0
        self.prev_faces = self._faces(view)
        self.prev_q = self._q(view)
        self.touch = {1: None, 2: None}
        self.trace: list[dict] = []
        self.verdict: tuple | None = None

    @staticmethod
    def _q(view: StepView) -> tuple[float, float]:
        return (float(view.state[6]) * 1000.0, -float(view.state[7]) * 1000.0)

    def _faces(self, view: StepView) -> tuple[float, float]:
        c = float((np.asarray(view.position[:2], dtype=float) - self.origin) @ self.jaw) * 1000.0
        q1, q2 = self._q(view)
        return (c + q1 - self.offset_mm, -c + q2 - self.offset_mm)

    def free_mm(self, q_mm: float) -> float:
        """The free inward motion of one finger in the next close action, from its opening q (mm): with the total
        gap g = 2 q and the calibrated close c = [g_open, empty_curve...], find c[j] >= g > c[j + 1]; the next gap is
        c[j + 1] - t (c[j + 1] - c[j + 2]) with t = (c[j] - g) / (c[j] - c[j + 1]); free = (g - next) / 2. Above c[0]:
        the first calibrated step; at or below the curve's end: 0."""
        c, g = self.curve_mm, 2.0 * float(q_mm)
        if g >= c[0]:
            return max(c[0] - c[1], 0.0) / 2.0 if len(c) > 1 else 0.0
        j = int(np.searchsorted(-c, -g, side="right")) - 1          # c[j] >= g > c[j + 1]
        if j + 1 >= len(c):
            return 0.0
        t = (c[j] - g) / max(c[j] - c[j + 1], 1e-9)
        nxt = c[j + 1] - t * ((c[j + 1] - c[j + 2]) if j + 2 < len(c) else 0.0)
        return max(g - nxt, 0.0) / 2.0

    def _next(self, view):
        return (self.opening or self.settle).next_action(view)

    def _check(self) -> tuple | None:
        for i in (1, 2):
            touch, limit = self.touch[i], self.limit_mm[i]
            if touch is not None and limit is not None and touch["s_mm"] > limit + 1e-9:
                return ("early_finger_contact", {"finger": i, "s_mm": touch["s_mm"], "limit_mm": limit,
                                                 "k": touch["k"]})
        if self.touch[1] is not None and self.touch[2] is not None and self.span_hi_mm is not None:
            span = self.touch[1]["s_mm"] + self.touch[2]["s_mm"]
            if span > self.span_hi_mm + 1e-9:
                return ("span_inconsistent", {"span_mm": span, "span_hi_mm": self.span_hi_mm,
                                              "k": max(self.touch[1]["k"], self.touch[2]["k"])})
        return None

    def _observe(self, before, action, after):
        if self.opening is not None:
            self.opening.observe_transition(before, action, after)
            if self.opening.done:
                self._finish("failed", self.verdict[0], self._evidence_now(after))
            return {"stage": "abort_open", "gap_mm": after.gap_mm}
        self.settle.observe_transition(before, action, after)
        self.k += 1
        faces = self._faces(after)
        frees = []
        for i in (1, 2):
            free = self.free_mm(self.prev_q[i - 1]) if self.k >= GC_FIRST_JUDGED_K else 0.0
            inward = self.prev_faces[i - 1] - faces[i - 1]
            frees.append(free)
            if (self.touch[i] is None and self.k >= GC_FIRST_JUDGED_K and free >= GC_FREE_MIN_MM
                    and free - inward >= self.touch_th.value):
                self.touch[i] = {"k": self.k, "s_mm": faces[i - 1], "deficit_mm": free - inward}
        self.trace.append({"k": self.k, "s_mm": list(faces), "free_mm": frees})
        self.prev_faces, self.prev_q = faces, self._q(after)
        if self.verdict is None:
            self.verdict = self._check()
            if self.verdict is not None and self.active:
                self.opening = GripperSettlePrimitive("open", after, config=GripperSettleConfig(
                    min_open_actions=OPEN_SETTLE_MIN_ACTIONS), name=f"{self.name}_abort_open")
                return {"stage": "abort", "reason": self.verdict[0], "gap_mm": after.gap_mm}
        if self.settle.done:
            self._finish(self.settle.status, self.settle.reason, self._evidence_now(after))
        return {"stage": "close", "gap_mm": after.gap_mm}

    @property
    def aborting(self) -> bool:
        """True once an active guard has abandoned the close (its abort open has started), whether or not that open
        finished: a preemption delivered during the open still leaves the close abandoned."""
        return self.opening is not None

    def guard_evidence(self) -> dict:
        """The guard's evidence so far (the outcome's ``evidence["guard"]``; also for a preempted close, whose
        synthetic outcome carries the bus report instead)."""
        return jsonable({"active": self.active, "touch": self.touch, "limit_mm": self.limit_mm,
                         "span_hi_mm": self.span_hi_mm, "verdict": self.verdict,
                         "would_abort": self.verdict is not None and not self.active,
                         "trace": self.trace, "thresholds": thresholds_dicts([self.touch_th])})

    def _evidence_now(self, view: StepView) -> dict:
        return jsonable({"guard": self.guard_evidence(),
                         "close": self.settle.outcome() if self.settle.done else None,
                         "open": self.opening.outcome() if self.opening is not None and self.opening.done else None,
                         "gap_mm": view.gap_mm})


class LiftMonitorAdapter:
    """Duck-types v1 WristPatchMonitor for ``VerifyHold(monitor=...)``: each ``observe(view)`` forwards (previous,
    view) to ``recorder.on_transition("lift", ...)`` (the lift silhouette, spec §4.7 item 4) and reports
    {"available": True, "ok": True}, so VerifyHold's status never depends on it (v1 semantics)."""

    def __init__(self, recorder, view: StepView):
        if not isinstance(view, StepView):
            raise TypeError("LiftMonitorAdapter needs the StepView the lift starts from")
        self.recorder = recorder
        self.previous = view
        self.frames = 0

    def observe(self, view) -> dict:
        if self.recorder is not None:
            self.recorder.on_transition("lift", self.previous, view)
        self.previous = view
        self.frames += 1
        return {"available": True, "ok": True, "frames": self.frames}


# --------------------------------------------------------------------------------------------- reserved (§C0.3)
class _ReservedPrimitive(Primitive):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is reserved in the slice (contract §C0.3)")


class DOFProbe(_ReservedPrimitive):
    """Reserved: the closed-grip DOF / mobility probe (spec §4.9 Grasp step 11, K10)."""


class FollowAxis(_ReservedPrimitive):
    """Reserved: the guarded move along a measured free axis (spec §4.9 Slide step 3, K8)."""


class PushAlong(_ReservedPrimitive):
    """Reserved: the push primitive of PushTo (spec §4.9 PushTo, O6)."""


class RayServo(_ReservedPrimitive):
    """Reserved: the external-ray servo of LookAt (spec §4.9 LookAt step 2, L2)."""


class RayDescent(_ReservedPrimitive):
    """Reserved: the guarded descent along the external ray of LookAt (spec §4.9 LookAt step 4, L3)."""


RESERVED_PRIMITIVES = (DOFProbe, FollowAxis, PushAlong, RayServo, RayDescent)
