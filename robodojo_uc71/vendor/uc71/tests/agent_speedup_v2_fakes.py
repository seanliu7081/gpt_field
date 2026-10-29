"""Fakes for the agent_SpeedUp_v2 tests (no simulator, no model). Extends tests/agent_speedup_fakes.py by import."""

from __future__ import annotations

import copy

from agent_speedup_fakes import FakeWristSim

from gpt_field.agent_speedup_v2_contacts import ALLOWED, FORBIDDEN, IGNORED, MODE


def generic_record(detected=False, first_by_class=None, forbidden_points=0):
    counts = {f"points_{name}": 0 for name in (*FORBIDDEN, *ALLOWED, *IGNORED)}
    counts.update(checks=1, physics_substeps=0, forbidden_contact_points=forbidden_points)
    return {"enabled": True, "mode": MODE, "detected": detected, "stop_on_collision": False,
            "collision_policy": "record_only", "success_qualification": "raw_libero_goal",
            "first_event": (next(iter((first_by_class or {}).values())) if detected else None),
            "first_by_class": dict(first_by_class or {}), "counts": counts}


def v1_record(detected=False):
    """The chained v1 task_manipulation_v1 monitor record (record-only)."""
    return {"enabled": True, "mode": "task_manipulation_v1", "detected": detected, "stop_on_collision": False,
            "collision_policy": "record_only", "success_qualification": "raw_libero_goal", "first_event": None,
            "counts": {"checks": 1, "physics_substeps": 0, "forbidden_contact_points": 0}}


def v2_status(steps=0, success=False, task_id=2, record=None, v1=None):
    status = {"task_id": task_id, "episode_index": 0, "steps": steps, "success": success,
              "terminated": bool(success), "libero_success": success,
              "collision": record or generic_record(), "collision_free_success": success}
    if v1 is not None:
        status["collision_v1_record_only"] = v1
    return status


def truth_result(strict, raw, why_not=(), state="terminal"):
    atom = {"atom": ["on", "obj_1", "target_1"], "kind": "on_object", "raw": raw, "strict": strict,
            "why_not": list(why_not), "checks": {}, "libero": {}, "numbers": {}}
    return {"diagnosis_only": True, "affects_completion": False, "raw_conjunction": raw, "check_success": raw,
            "strict_conjunction": strict, "atoms": [atom], "state": state}


class FakeV2Sim:
    """Runner-facing fake of V2SimProcess: the v2 status contract plus the evaluator-only truth ops.

    ``chained``: the status carries the chained v1 record (bowl_plate / drawer_bowl). ``interrupt_at``: that
    step raises KeyboardInterrupt with its reply left queued and ``_in_flight`` True, as V2SimProcess does when
    an interrupt lands inside a request. ``stale_reply``: a reply already queued, so every truth op reads the
    previous op's reply (a misaligned JSONL pipe)."""

    instances = []

    def __init__(self, config=None, *, prompt, task_id=2, success_at=None, collision_at=None, tasks=None,
                 terminal_truth=None, settled_truth=None, truth_error=None, wrist=None, chained=False,
                 interrupt_at=None, stale_reply=None):
        self.config, self.prompt, self.task_id = config, prompt, task_id
        self.success_at, self.collision_at = success_at, collision_at
        self.chained, self.interrupt_at = chained, interrupt_at
        self._in_flight = False
        self._queue = [] if stale_reply is None else [copy.deepcopy(stale_reply)]
        self.wrist = wrist or FakeWristSim(prompt=prompt)
        self._tasks = tasks if tasks is not None else [{"task_id": task_id, "prompt": prompt}]
        self.status = v2_status(task_id=task_id, v1=v1_record() if chained else None)
        self.metadata = {"fake": True, "collision_check_mode": MODE}
        self.actions, self.ops, self.closed = [], [], False
        self.terminal_truth = terminal_truth or truth_result(False, False, ["supported_by_target"])
        self.settled_truth = settled_truth or truth_result(False, False, ["supported_by_target"],
                                                           state="after_release_settle")
        self.truth_error = truth_error
        FakeV2Sim.instances.append(self)

    def start(self):
        self.ops.append("start")
        return self

    def list_tasks(self):
        return copy.deepcopy(self._tasks)

    def reset(self, task_id, episode_index):
        self.ops.append("reset")
        self.status = v2_status(task_id=task_id, v1=v1_record() if self.chained else None)
        return self.wrist.observation()

    def step(self, action):
        if self.closed:
            raise RuntimeError("closed")
        n = len(self.actions) + 1
        if self.interrupt_at is not None and n == self.interrupt_at:
            self._in_flight = True       # the worker executes the step; its reply stays in the pipe
            self._queue.append({"observation": {"observation/state": [0.0] * 8}, "status": dict(self.status)})
            raise KeyboardInterrupt()
        self.actions.append(list(action))
        obs = self.wrist.step(action)
        success = self.success_at is not None and n >= self.success_at
        record = generic_record()
        if self.collision_at is not None and n >= self.collision_at:
            first = {"class": "robot_scene", "action_index": self.collision_at - 1}
            record = generic_record(True, {"robot_scene": first, "finger_object": {"action_index": 0}}, 3)
        self.status = v2_status(n, success, self.task_id, record, v1_record() if self.chained else None)
        return obs, success

    def _reply(self, value):
        """A queued (stale) reply is read first; this op's reply then waits in the queue."""
        if self._queue:
            self._queue.append(value)
            return self._queue.pop(0)
        return value

    def truth(self):
        self.ops.append("truth")
        if self.truth_error:
            raise self.truth_error
        return self._reply(copy.deepcopy(self.terminal_truth))

    def settle_truth(self, protocol=None):
        self.ops.append("settle_truth")
        return self._reply(copy.deepcopy(self.settled_truth))

    def answer_key(self):
        self.ops.append("answer_key")
        return self._reply({"diagnosis_only": True, "goal_atoms": [["on", "obj_1", "target_1"]]})

    def close(self):
        self.ops.append("close")
        self.closed = True


# =====================================================================================================================
# FAKES-APPEND (builder BF; agent_speedup_v2 contract §C15). Everything below this line is the T1 world (spec §8.3):
# FakeScene / FakeBody / FakeHinge / FakeEffect, FakeExternalCamera, FakeTwoViewSim (two cameras, the robot's own
# geometry, contact, grasp, held objects, a hinge with the SP-fitted wedge model, colour-change regions), the
# FakeOracleAnswers scripted answerer, the runner-facing FakeV2WorldSim and the scene builders. M0's section above is
# unchanged.
#
# Conventions (the real stack's): state8 = [grip xyz, right_hand world rotvec, q1, q2]; R = rotation_matrix(state8[3:6])
# with columns R0 (wrist image up), R1 (jaw, wrist image right), R2 (approach); hand origin = grip - 0.097 R2 (fact R1);
# wrist camera = hand + 0.05 R0 (fact R2), pinhole u = c + f a/z, v = c - f b/z (identical to FakeWristSim.project);
# actions are 7 floats: OSC world deltas scaled by 0.05 m / 0.5 rad, gripper +1 close / -1 open / 0 hold; images are
# delivered in the LEGACY orientation (upright[:, ::-1]) as PNG data URLs.
#
# EVALUATOR TRUTH: everything a FakeTwoViewSim knows about bodies, joints, contacts and the goal is truth. It is reachable
# only through the explicit truth accessors (heights, body_pose, joint, held, project, contact_log, truth_mask,
# colour_region_changed, goal_satisfied, physical_truth, truth_record) and the scene's ``info`` dict; policy code under
# test only ever sees observation()/step() output.
#
# Numbers marked [SP] were fitted to the spec-revision knob probe ONLY (tests/data/agent_speedup_v2/sp_knob_sense_probe.
# json); the CP knob traces (cp_knob_twist_probe.json) are held out for the non-circular check in
# tests/test_agent_speedup_v2_fakes.py. Numbers marked [ROBOT] come from robot_facts_v2.json (via M0 _robot) or from
# recorded robot-only traces (gripper curves, free-air yaw); [CP-air] = CP's free-air yaw (no knob in the hand).
# =====================================================================================================================

import json as _json
import math as _math
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path

import cv2 as _cv2
import numpy as _np

from agent_speedup_fakes import png_url as _png_url

from gpt_field import agent_speedup_v2_robot as _robot
from gpt_field.action import _quaternion_product as _qmul, _rotvec_quaternion as _rvq
from gpt_field.mark_prompting import rotate_pixel as _rotate_pixel
from gpt_field.observation import PolicyObservation as _PolicyObservation, quaternion_to_axisangle as _q2rv
from gpt_field.robot_tool_geometry import _rotation_matrix as _rotmat

FAKES_V2_VERSION = "agent-speedup-v2-fakes-bf-1"
DATA_DIR = _Path(__file__).resolve().parent / "data" / "agent_speedup_v2"
IMAGE_PX = 512
CENTRE_PX = (IMAGE_PX - 1) / 2.0
TASK2_PROMPT = "turn on the stove and put the moka pot on it"

# --- [ROBOT] robot facts (robot_facts_v2.json through M0 _robot) ---------------------------------------------------
_ASSET = _robot.ROBOT_FACTS["asset"]
GRIP_ALONG_R2_M = _robot.GRIP_SITE_ALONG_R2_M                                   # 0.097 (R1)
CAMERA_ALONG_R0_M = _robot.WRIST_CAMERA_ALONG_R0_M                              # 0.05 (R2)
FINGERTIP_REACH_M = _ASSET["fingertip_reach_beyond_grip_mm"] / 1000.0           # 0.00925 (R5, asset)
PAD_SPAN_R2_M = tuple(v / 1000.0 for v in _robot.PAD_SPAN_ALONG_APPROACH_MM)    # (-0.0116, +0.0044) about the grip (R4)
PAD_HALF_R0_M = _robot.PAD_LENGTH_R0_MM / 2000.0                                # 0.008 (pad 16 mm along R0)
PAD_FACE_INSET_M = _ASSET["pad_separation_offset_mm"]["at_limit_q"] / 2000.0    # pad inner face at q - 0.5065 mm (R9)
FINGER_Q_MAX_M = _ASSET["finger_joint_ranges_m"]["gripper0_finger_joint1"][1]   # 0.04 per finger
_FOOT = _ASSET["finger_landing_footprints_by_finger_q_mm"]["0"]                 # footprints at q = 0 (R1 range = q + this)
FINGER1_R0_MM = tuple(_FOOT["finger1"]["R0_mm"])                                 # (-5.48, 6.14)
FINGER1_R1_MM = tuple(_FOOT["finger1"]["R1_mm"])                                 # (-0.1, 14.29) relative to q
FINGER1_BODY_R1_MM = tuple(_FOOT["finger1"]["finger_and_pad_R1_mm"])             # (-0.51, 26.4) relative to q
PALM_PROFILE_MM = tuple(tuple(row) for row in _robot.palm_profile())            # [(xi along R1, p behind the grip)]
HAND_R0_MM = tuple(_ASSET["hand_footprint_R0_mm"])                               # (-31.64, 31.62)
HAND_R1_MM = tuple(_ASSET["hand_footprint_R1_mm"])                               # (-103.99, 100.43)
WRIST_FROM_GRIP_M = _robot.WRIST_FROM_GRIP_M                                    # 0.204 (R13)
SHOULDER_XYZ = tuple(_np.add(_robot.VERIFIED_MOUNT_BASE_XYZ, (0.0, 0.0, _robot.SHOULDER_ABOVE_BASE_M)))
TASK2_START_STATE8 = (-0.2014, 0.0187, 1.1893, 3.1397, 0.0067, -0.0861, 0.0387, -0.0387)   # SURVEY t2_dump.json

# --- [ROBOT] recorded gripper curves (runs/agent_speedup_v1_bowl_zero_shot_001 trace, v1 Calibrate, real LIBERO) ------
# close from the reset opening 77.4 mm (k = 1..30), then open from 1.186 mm (k = 1..30, then the reach steps)
CLOSE_CURVE_MM = (77.4, 77.174, 73.677, 67.359, 59.427, 50.588, 41.24, 31.607, 21.813, 13.391, 8.386, 5.573, 3.998,
                  3.117, 2.624, 2.349, 2.195, 2.097, 2.007, 1.921, 1.839, 1.76, 1.684, 1.612, 1.543, 1.477, 1.414,
                  1.353, 1.295, 1.239, 1.186)
OPEN_CURVE_MM = (1.186, 1.772, 5.718, 12.301, 20.383, 29.305, 38.7, 48.359, 58.168, 66.598, 71.608, 74.423, 76.0,
                 76.882, 77.375, 77.651, 77.805, 77.903, 77.993, 78.079, 78.161, 78.24, 78.315, 78.388, 78.457, 78.523,
                 78.586, 78.647, 78.705, 78.761, 78.814, 78.865, 78.913, 78.96, 79.004, 79.047, 79.088, 79.127, 79.164,
                 79.2, 79.235, 79.267, 79.299, 79.329, 79.358, 79.385, 79.411, 79.437, 79.461, 79.484, 79.506, 79.527)
CLOSE_TAIL_FLOOR_MM = 0.95       # empty-close tail beyond the record: g <- floor + (g - floor) * 0.82 (SP: 1.04 at k = 33)
CLOSE_TAIL_RATE = 0.82
OPEN_TAIL_LIMIT_MM = 80.0         # 2 x the finger joint limit; tail g <- limit - (limit - g) * 0.956 (recorded creep)
OPEN_TAIL_RATE = 0.956
PAD_CONTACT_GAP_MM = 2.0 * _ASSET["pad_contact_finger_q_mm"]                  # 1.012: the pads touch each other
START_GAP_MM = 77.4               # LIBERO reset opening (q 0.0387)
CLAMP_CREEP_MM = 0.0013           # [SP] closed on the lever: 25.391 -> 25.382 over 8 static steps
CLAMP_CREEP_TAU = 30.0

# --- [ROBOT, CP-air] wrist rotation response -------------------------------------------------------------------------
# realized |rotation| per step for a commanded c rad: ROT_GAIN c / sqrt(1 + (c / ROT_SAT_RAD)^2); fitted to the closed
# empty hand (SP: 0.1 rad -> 1.313 deg/step) and CP's free-air yaw (0.15 -> 1.866, 0.5 -> 3.705 deg/step); a first-order
# angular-velocity lag (rho 0.17, averaged over the step) reproduces the 0.48 / 0.91 / 0.975 onset of SP and CP.
ROT_GAIN = 0.2399
ROT_SAT_RAD = 0.321
ROT_LAG = 0.17
# --- [SP] hinge / wedge model (fit_hinge2: SP traces only; see the test file's non-circular CP check) --------------
# Known deviations of the fitted model (reviews BF-R9, BF-R11; tests on the fake must not assume SP's exact k0):
# - blocked sense: the rates of the first 3 steps after the close are 0.29-0.33 mm/step (SP 0.341-0.348) and then rise
#   monotonically to ~0.46 instead of staying flat with noise (SP 0.34-0.51); the K1w blocked verdict still holds.
# - reversal: the first-order hand-yaw lag (ROT_LAG) keeps the gap rising at the reversal step itself (+0.07 against
#   CP's -0.06 mm/step), and the unwedge fall lasts one step longer than CP's (10 steps against 9; SP 14 against 13),
#   so after a 12-step blocked segment the K1w free decision lands at k = 16 on the fake against SP's k = 14 (the
#   lower limit is reached at k = 22 on both).
WEDGE_A_MM_PER_RAD = 14.0589      # gap excess = a |phi| + b phi^2, phi = hand yaw minus part yaw since the close
WEDGE_B_MM_PER_RAD2 = 20.7690
HINGE_A_SAME = 0.0031660          # rad/step^2 pad-friction drive when the slip pushes the wedged side (ramped by phi_r)
HINGE_A_OPP = 0.000646            # rad/step^2 drive while unwedging (slip against the wedged side)
HINGE_BETA = 0.067987             # 1/step viscous damping of the part (MuJoCo damping 1.0, [F])
HINGE_PHI_R = 0.039245            # rad: the drive ramps in with |phi| up to this
HINGE_SUBSTEPS = 20
YAW_LOAD_K_TAU = 0.1292           # [SP] hand-yaw rate factor 1 - k_tau tau/A_same - k_phi max(0, |phi| - phi0)
YAW_LOAD_K_PHI = 1.2934
YAW_LOAD_PHI0 = 0.1981
ORBIT_STIFFNESS = 0.8             # [CP] the lever drags the grip onto its orbit (CP: 13 mm off the hinge, no tug at 1.2 rad)
# --- translation -------------------------------------------------------------------------------------------------------
TRANSLATION_RATIO = 0.26          # [ROBOT] free realized/commanded ratio (v1 AnchoredMove.DEFAULT_FREE_RATIO; CP 0.268)
SECOND_ORDER_W = 11.6             # v1 DynWristSim spring-damper (recorded 512-px traces): 0.115/0.209/0.234 -> 0.256
SECOND_ORDER_Z = 1.0
SECOND_ORDER_STEADY = 0.256
SECOND_ORDER_SUBSTEPS = 25
# --- contact ------------------------------------------------------------------------------------------------------------
FINGER_DEFLECT_GAIN = 0.6         # inward deflection of a landing finger: clip(gain x blocked free motion, 0.7, 2.0) mm
FINGER_DEFLECT_MM = (0.7, 2.0)
FINGER_DEFLECT_CAP_MM = 3.0
HELD_PUSH_RATIO = 0.04            # [CP lower] after the stall the hand still creeps 0.03-0.045 of the command
HELD_GAP_RISE_MM = 0.08           # [CP lower] 0.05-0.13 mm/step while the held object is pressed onto a support
HELD_TILT_DEG_PER_STEP = 0.5      # [CP lower] the pot pivots in the pinch while pressed (not modelled: pose kept)
HELD_OBSTACLE_GAP_RISE_MM = 0.45  # a held object pushed sideways into an obstacle twists in the pinch (v1 live 001:
                                  # contact episodes opened the pinch +1.3..+1.7 mm; WaypointReach GAP_STEP_RISE 0.4)
CONTACT_TOL_M = 0.001             # LIBERO-like contact: lowest point within 1 mm of the support top
SUPPORT_CONTACT_EPS_M = 1e-7      # the support plane is a half-space collider: a point is in it below support_z - eps
ROBOT_SAMPLE_MM = 4.0             # robot collision samples (palm, hand top, finger walls) at most 4 mm apart
SUPPORT_OWNER = -2                # contact owner id of the support plane in the contact sweeps
RELEASE_CLEAR_MM = 0.25           # a hold ends once the pads have opened this far past the clamped cross-section
TIP_MAX_DEG = 25.0                # palm pressing a free overhang tips the object about its base edge (P13 negative),
                                  # by the angle that lets the pressed point yield the commanded push, up to this tilt
SETTLE_UPRIGHT_MAX_DEG = 25.0
# --- rendering -----------------------------------------------------------------------------------------------------------
NEAR_M = 1e-4
SIDE_GAIN = 0.75
TEXTURE_PX = 1024
PIXEL_NOISE_GREY = 0.35
GAP_NOISE_MM = 0.002              # [SP] static closed gap 0.001-0.002 mm/step; closed empty yaw <= 0.003 mm/step
EXTERNAL_DYNAMIC_STEPS = 5


# ----------------------------------------------------------------------------------------------- small math helpers
def _unit(v):
    v = _np.asarray(v, float)
    n = float(_np.linalg.norm(v))
    return v / n if n > 1e-15 else v * 0.0


def _rot_axis(axis, angle):
    """Rotation matrix about a unit axis (right-hand rule)."""
    return _rotmat(_np.asarray(_unit(axis), float) * float(angle))


def _rotvec_of(R):
    q = _matrix_quat(R)
    return _np.asarray(_q2rv(q), float)


def _matrix_quat(R):
    m = _np.asarray(R, float)
    tr = float(_np.trace(m))
    if tr > 0:
        s = 2.0 * _math.sqrt(1.0 + tr)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    else:
        i = int(_np.argmax(_np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * _math.sqrt(max(1e-15, 1.0 + m[i, i] - m[j, j] - m[k, k]))
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    q = _np.asarray(q, float)
    q /= _np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def _quat_matrix(q):
    return _rotmat(_np.asarray(_q2rv(_np.asarray(q, float) / _np.linalg.norm(q)), float))


def _yaw_of(R):
    return _math.atan2(R[1, 0], R[0, 0])


def _tilt_deg(R):
    return _math.degrees(_math.acos(float(_np.clip(R[2, 2], -1.0, 1.0))))


def _wrap(angle):
    return (angle + _math.pi) % (2 * _math.pi) - _math.pi


def rotation_response_rad(command_rad):
    """[ROBOT, CP-air] steady realized rotation per step for a commanded rotation magnitude (rad)."""
    c = abs(float(command_rad))
    return ROT_GAIN * c / _math.sqrt(1.0 + (c / ROT_SAT_RAD) ** 2)


def wedge_gap_excess_mm(phi_rad):
    """[SP] finger-gap excess over the clamped width when the hand is yawed phi relative to an elongated part."""
    p = abs(float(phi_rad))
    return WEDGE_A_MM_PER_RAD * p + WEDGE_B_MM_PER_RAD2 * p * p


def _curve_step(curve, gap_mm, phase, *, closing):
    """Next gap on a recorded robot curve from an arbitrary current gap.

    Gap-based inverse lookup (+1 step), limited by the curve's own per-step change at ``phase`` (steps since the
    command started), so a close or open that starts away from the recorded start accelerates as the robot does."""
    c = _np.asarray(curve, float)
    idx = _np.arange(len(c))
    g = float(gap_mm)
    k = int(max(0, phase))
    if closing:
        if g <= c[-1]:
            looked = CLOSE_TAIL_FLOOR_MM + (g - CLOSE_TAIL_FLOOR_MM) * CLOSE_TAIL_RATE
        elif g >= c[0]:
            looked = c[1] - (g - c[0])
        else:
            looked = float(_np.interp(float(_np.interp(-g, -c, idx)) + 1.0, idx, c))
        limit = g - (c[k] - c[k + 1]) if k + 1 < len(c) else -_np.inf
        return max(looked, limit, min(g, PAD_CONTACT_GAP_MM))
    if g >= c[-1]:
        looked = OPEN_TAIL_LIMIT_MM - (OPEN_TAIL_LIMIT_MM - g) * OPEN_TAIL_RATE
    elif g <= c[0]:
        looked = c[1]
    else:
        looked = float(_np.interp(float(_np.interp(g, c, idx)) + 1.0, idx, c))
    limit = g + (c[k + 1] - c[k]) if k + 1 < len(c) else _np.inf
    return min(looked, limit, OPEN_TAIL_LIMIT_MM)


# ------------------------------------------------------------------------------------------------------- textures
class _TextureBank:
    """Zero-mean unit-std PERIODIC noise textures (TEXTURE_PX^2, built by circular Gaussian filtering in the Fourier
    domain, so wrap-around has no seam) with pre-blurred mip levels; sampled by cv2.remap.

    ``iso``: isotropic (sigma 1.5 texels); ``wood``: streaks (sigma 6 x 1.5 texels along texture u) mixed 0.8 : 0.6 with
    an isotropic field (so a shift along the streaks is still measurable). A body sets the texel
    size (texture_scale_m) and the std; level L >= 1 is the base blurred by 0.8 x 2^(L-1) texels more (minification, so a
    far view loses contrast as a renderer's mipmaps do). Banks are cached per seed (pure function of the seed)."""

    LEVELS = 4
    _CACHE = {}

    def __new__(cls, seed):
        key = int(seed)
        if key not in cls._CACHE:
            bank = super().__new__(cls)
            bank._build(key)
            cls._CACHE[key] = bank
        return cls._CACHE[key]

    def _build(self, seed):
        rng = _np.random.default_rng(seed + 7919)
        fy = _np.fft.fftfreq(TEXTURE_PX)[:, None]
        fx = _np.fft.fftfreq(TEXTURE_PX)[None, :]
        self.levels = {}
        spectra = [_np.fft.fft2(rng.standard_normal((TEXTURE_PX, TEXTURE_PX))) for _ in range(2)]

        def field(spectrum, su, sv, extra):
            gu, gv = _math.hypot(su, extra), _math.hypot(sv, extra)
            transfer = _np.exp(-2.0 * _math.pi ** 2 * ((fx * gu) ** 2 + (fy * gv) ** 2))
            return _np.real(_np.fft.ifft2(spectrum * transfer))

        for kind in ("iso", "wood"):
            levels = []
            for level in range(self.LEVELS):
                extra = 0.0 if level == 0 else 0.8 * 2 ** (level - 1)
                if kind == "iso":
                    tex = field(spectra[0], 1.5, 1.5, extra)
                else:                                   # streaks along u with a 2-D component (no aperture problem)
                    a, b = field(spectra[1], 6.0, 1.5, extra), field(spectra[0], 1.5, 1.5, extra)
                    tex = 0.8 * a / max(float(a.std()), 1e-12) + 0.6 * b / max(float(b.std()), 1e-12) if level == 0 \
                        else 0.8 * a / base_std[0] + 0.6 * b / base_std[1]
                if level == 0:
                    if kind == "wood":
                        base_std = (max(float(a.std()), 1e-12), max(float(b.std()), 1e-12))
                    mean, std = float(tex.mean()), max(float(tex.std()), 1e-9)
                levels.append(((tex - mean) / std).astype(_np.float32))
            self.levels[kind] = levels

    def sample(self, kind, u, v, texels_per_px):
        """Bilinear sample at texel coordinates (u, v) (float arrays); level from the pixel footprint in texels."""
        level = 0
        if texels_per_px > 1.25:
            level = int(min(self.LEVELS - 1, 1 + int(_math.floor(_math.log2(texels_per_px / 1.25)))))
        tex = self.levels[kind][level]
        shape = _np.shape(u)
        flat_u = _np.mod(_np.ravel(u), TEXTURE_PX).astype(_np.float32)
        flat_v = _np.mod(_np.ravel(v), TEXTURE_PX).astype(_np.float32)
        n = flat_u.size
        if n == 0:
            return _np.zeros(shape, _np.float32)
        width = 1024 if n > 1024 else n
        rows = -(-n // width)
        pad = rows * width - n
        mx = _np.pad(flat_u, (0, pad)).reshape(rows, width)       # cv2.remap needs both map sides < SHRT_MAX
        my = _np.pad(flat_v, (0, pad)).reshape(rows, width)
        out = _cv2.remap(tex, mx, my, _cv2.INTER_LINEAR, borderMode=_cv2.BORDER_WRAP)
        return out.ravel()[:n].reshape(shape)


# ------------------------------------------------------------------------------------------- primitive intersection
def _slab(o, d, lo, hi):
    """Entry/exit parameters of lines o + t d (rows) through the slab lo <= x <= hi along one axis."""
    with _np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin, tmax = _np.minimum(t1, t2), _np.maximum(t1, t2)
    zero = d == 0
    if zero.any():
        inside = (o >= lo) & (o <= hi)
        tmin = _np.where(zero, _np.where(inside, -_np.inf, _np.inf), tmin)
        tmax = _np.where(zero, _np.where(inside, _np.inf, -_np.inf), tmax)
    return tmin, tmax


def _box_hits(o, d, hx, hy, z0, z1):
    """Lines o + t d (o (3,) or (N,3); d (N,3)) against the box [-hx,hx]x[-hy,hy]x[z0,z1]: (t_in, t_out, face).
    Faces: 0 -x, 1 +x, 2 -y, 3 +y, 4 bottom, 5 top."""
    o = _np.asarray(o)
    ox, oy, oz = (o[0], o[1], o[2]) if o.ndim == 1 else (o[:, 0], o[:, 1], o[:, 2])
    ax, bx = _slab(ox, d[:, 0], -hx, hx)
    ay, by = _slab(oy, d[:, 1], -hy, hy)
    az, bz = _slab(oz, d[:, 2], z0, z1)
    t_in = _np.maximum(_np.maximum(ax, ay), az)
    t_out = _np.minimum(_np.minimum(bx, by), bz)
    face = _np.where(t_in == az, 4 + (d[:, 2] <= 0), _np.where(t_in == ay, 2 + (d[:, 1] <= 0), 0 + (d[:, 0] <= 0)))
    return t_in, t_out, face.astype(_np.int8)


def _cyl_hits(o, d, r, z0, z1):
    """Lines against the vertical cylinder x^2 + y^2 <= r^2, z0 <= z <= z1: (t_in, t_out, face) with 0 side, 4/5 caps."""
    o = _np.asarray(o)
    ox, oy, oz = (o[0], o[1], o[2]) if o.ndim == 1 else (o[:, 0], o[:, 1], o[:, 2])
    dx, dy = d[:, 0], d[:, 1]
    A = dx * dx + dy * dy
    B = 2.0 * (ox * dx + oy * dy)
    C = ox * ox + oy * oy - r * r
    disc = B * B - 4.0 * A * C
    vertical = A < 1e-18
    with _np.errstate(divide="ignore", invalid="ignore"):
        sq = _np.sqrt(_np.maximum(disc, 0.0))
        inv = 0.5 / A
        c1 = (-B - sq) * inv
        c2 = (-B + sq) * inv
    if vertical.any():
        c1 = _np.where(vertical, _np.where(C <= 0, -_np.inf, _np.inf), c1)
        c2 = _np.where(vertical, _np.where(C <= 0, _np.inf, -_np.inf), c2)
    miss = (disc < 0) & ~vertical
    if miss.any():
        c1 = _np.where(miss, _np.inf, c1)
        c2 = _np.where(miss, -_np.inf, c2)
    zmin, zmax = _slab(oz, d[:, 2], z0, z1)
    t_in = _np.maximum(c1, zmin)
    t_out = _np.minimum(c2, zmax)
    face = _np.where(c1 >= zmin, 0, _np.where(d[:, 2] < 0, 5, 4))
    return t_in, t_out, face.astype(_np.int8)


class _Piece:
    """One convex primitive of a body: ``box`` (half sizes hx, hy, z range) or ``cyl`` (radius, z range) in its own local
    frame; ``R0``/``t0`` place it in the body's INITIAL world pose; the body group's rigid transform moves it."""

    __slots__ = ("kind", "hx", "hy", "r", "z0", "z1", "R0", "t0", "body", "index", "tex_offset", "top_only_effect")

    def __init__(self, kind, *, hx=0.0, hy=0.0, r=0.0, z0, z1, R0=None, t0=(0.0, 0.0, 0.0), body=None, index=0):
        self.kind = kind
        self.hx, self.hy, self.r = float(hx), float(hy), float(r)
        self.z0, self.z1 = float(z0), float(z1)
        self.R0 = _np.eye(3) if R0 is None else _np.asarray(R0, float)
        self.t0 = _np.asarray(t0, float)
        self.body = body
        self.index = index
        self.tex_offset = (0.0, 0.0)

    def hits(self, o_local, d_local):
        if self.kind == "box":
            return _box_hits(o_local, d_local, self.hx, self.hy, self.z0, self.z1)
        return _cyl_hits(o_local, d_local, self.r, self.z0, self.z1)

    def contains_local(self, p, margin=0.0):
        z_ok = (p[:, 2] >= self.z0 - margin) & (p[:, 2] <= self.z1 + margin)
        if self.kind == "box":
            return z_ok & (_np.abs(p[:, 0]) <= self.hx + margin) & (_np.abs(p[:, 1]) <= self.hy + margin)
        return z_ok & (p[:, 0] ** 2 + p[:, 1] ** 2 <= (self.r + margin) ** 2)

    def corners_local(self):
        hx, hy = (self.hx, self.hy) if self.kind == "box" else (self.r, self.r)
        return _np.array([[sx * hx, sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (self.z0, self.z1)])

    def face_uv(self, p, face):
        """Texture coordinates (metres) of local hit points per face (continuous across a body's faces)."""
        u = _np.empty(len(p))
        v = _np.empty(len(p))
        caps = face >= 4
        u[caps], v[caps] = p[caps, 0], p[caps, 1]
        if self.kind == "box":
            fx = (face == 0) | (face == 1)
            fy = (face == 2) | (face == 3)
            u[fx], v[fx] = p[fx, 1] + 3.0 * self.hy, p[fx, 2] + 0.37
            u[fy], v[fy] = p[fy, 0] + 2.0 * self.hx, p[fy, 2] + 0.61
        else:
            side = face == 0
            u[side], v[side] = self.r * _np.arctan2(p[side, 1], p[side, 0]) + 7.0 * self.r, p[side, 2] + 0.53
        return u, v

    def sample_points_local(self, spacing):
        """Surface sample points (local frame) for collision tests: the bottom, the top and the side walls."""
        pts = []
        zs = _np.linspace(self.z0, self.z1, max(2, int(_math.ceil((self.z1 - self.z0) / spacing)) + 1))
        if self.kind == "box":
            nx = max(2, int(_math.ceil(2 * self.hx / spacing)) + 1)
            ny = max(2, int(_math.ceil(2 * self.hy / spacing)) + 1)
            xs, ys = _np.linspace(-self.hx, self.hx, nx), _np.linspace(-self.hy, self.hy, ny)
            gx, gy = _np.meshgrid(xs, ys)
            for z in (self.z0, self.z1):
                pts.append(_np.c_[gx.ravel(), gy.ravel(), _np.full(gx.size, z)])
            ring = _np.r_[_np.c_[xs, _np.full(nx, -self.hy)], _np.c_[xs, _np.full(nx, self.hy)],
                          _np.c_[_np.full(ny, -self.hx), ys], _np.c_[_np.full(ny, self.hx), ys]]
        else:
            n = max(8, int(_math.ceil(2 * _math.pi * self.r / spacing)))
            ang = _np.linspace(0, 2 * _math.pi, n, endpoint=False)
            ring = _np.c_[self.r * _np.cos(ang), self.r * _np.sin(ang)]
            disc = [ring * f for f in _np.linspace(0.0, 1.0, max(2, int(_math.ceil(self.r / spacing)) + 1))]
            disc = _np.unique(_np.round(_np.vstack(disc), 6), axis=0)
            for z in (self.z0, self.z1):
                pts.append(_np.c_[disc, _np.full(len(disc), z)])
        for z in zs:
            pts.append(_np.c_[ring, _np.full(len(ring), z)])
        return _np.vstack(pts)


# --------------------------------------------------------------------------------------------------- scene records
@_dataclass
class FakeHinge:
    """Rotary mechanism of a ``rotary`` body group (all evaluator truth).

    ``free_sense`` is the CONTRACT sense (rotation about the approach R2 of a top-down hand, §C1.3) that is free at the
    start: the part starts at its lower limit and q increases under a world rotation about ``axis`` = (0, 0, -free_sense)
    (task 2: free_sense -1, axis +z, as SP: world +z took q 0 -> 0.134 in 12 steps). Damping and the wedge gap-rate model
    are the [SP] fit; there is no return spring unless ``spring`` > 0 (a per-step relaxation toward q_rest after release).
    ``wedge`` False is a round part: no gap change with the relative yaw, the pads slip when it is blocked."""

    centre_xy: tuple
    free_sense: int = -1
    q: float = 0.0
    q_lo: float = -0.005
    q_hi: float = 2.1
    q_on: float = 0.5
    wedge: bool = True
    spring: float = 0.0
    q_rest: float = 0.0
    a_same: float = HINGE_A_SAME
    a_opp: float = HINGE_A_OPP
    beta: float = HINGE_BETA
    phi_r: float = HINGE_PHI_R
    wedge_a: float = WEDGE_A_MM_PER_RAD
    wedge_b: float = WEDGE_B_MM_PER_RAD2

    def __post_init__(self):
        if self.free_sense not in (-1, 1):
            raise ValueError("free_sense must be -1 or +1 (contract sense)")
        self.q0 = float(self.q)
        self.qd = 0.0
        self.last_torque = 0.0

    @property
    def axis(self):
        return _np.array([0.0, 0.0, -float(self.free_sense)])

    def gap_excess_mm(self, phi):
        if not self.wedge:
            return 0.0
        p = abs(float(phi))
        return self.wedge_a * p + self.wedge_b * p * p

    def advance(self, dpsi, phi, *, slipping=False):
        """One control step: the hand yaws ``dpsi`` (rad, about ``axis``); returns the new relative angle phi.
        Pad friction drives the part (A_same / A_opp, ramped by phi_r), the part has viscous damping and hard limits;
        it sticks to the hand once it catches up while friction carries the damping."""
        w = float(dpsi)
        q, qd = self.q, self.qd
        dt = 1.0 / HINGE_SUBSTEPS
        tau_sum = 0.0
        for _ in range(HINGE_SUBSTEPS):
            s = w - qd
            if slipping:
                A = 0.0
            elif phi * s >= 0.0:
                A = self.a_same * min(1.0, abs(phi) / self.phi_r)
            else:
                A = self.a_opp
            if abs(s) < 1e-12 and self.beta * abs(qd) <= A:
                qd_new, tau = w, self.beta * abs(qd)
            else:
                tau = A
                qd_new = qd + (A * _math.copysign(1.0, s) - self.beta * qd) * dt
                if (w - qd) * (w - qd_new) < 0:
                    qd_new = w
            q_new = q + qd_new * dt
            if q_new <= self.q_lo and qd_new < 0:
                q_new, qd_new = self.q_lo, 0.0
            if q_new >= self.q_hi and qd_new > 0:
                q_new, qd_new = self.q_hi, 0.0
            phi += w * dt - (q_new - q)
            q, qd = q_new, qd_new
            tau_sum += tau * dt
        self.q, self.qd = q, qd
        self.last_torque = tau_sum
        return phi

    def relax(self):
        """Released: the damped part stops; with a spring it relaxes toward q_rest."""
        self.qd = 0.0
        if self.spring > 0:
            self.q = float(self.q - self.spring * (self.q - self.q_rest))


@_dataclass
class FakeEffect:
    """Colour-change region on the top face of a body: a ring (r_in..r_out about centre_xy, initial world) whose CHROMA
    changes (rgb added; luminance about kept) once the driver hinge's q >= q_on, after ``lag_steps`` (LIBERO toggles the
    burner site one step after TurnOn holds; M0 effect probe). ~20 % of a 19 x 19 cm region by default."""

    driver: str
    centre_xy: tuple
    r_in: float = 0.025
    r_out: float = 0.055
    chroma_rgb: tuple = (95.0, -32.0, -36.0)
    lag_steps: int = 1
    never: bool = False


_BODY_KINDS = ("plane", "aabb", "cylinder", "disk", "ridge", "loop")


@_dataclass
class FakeBody:
    """One rigid body (evaluator truth). ``centre`` = (x, y, z_bottom) of its footprint centre in the initial world;
    ``size``: plane/aabb/ridge (sx, sy, sz); cylinder/disk (radius, height); loop: explicit ``parts``.
    Bodies sharing ``group`` move rigidly together (a rotary part's disk and ridge; a free object's body, thin part and
    spout). ``role`` (fixed | free | rotary) and ``hinge`` are read from the group's first body that sets them.
    ``parts`` (loop / compound): [("box", (x, y, z_bottom), (sx, sy, sz), yaw) | ("cyl", (x, y, z_bottom), (r, h))]."""

    name: str
    kind: str
    centre: tuple
    size: tuple = ()
    yaw: float = 0.0
    colour: tuple = (150.0, 150.0, 150.0)
    texture_std: float = 8.0
    texture_scale_m: float = 0.0015
    texture: str = "iso"
    role: str = "fixed"
    group: str | None = None
    hinge: FakeHinge | None = None
    mass_kg: float = 0.1
    tilt_deg: float = 0.0
    effect: FakeEffect | None = None
    reflect_grey: float = 0.0
    side_gain: float = SIDE_GAIN
    pattern: str | None = None
    parts: tuple = ()

    def __post_init__(self):
        if self.kind not in _BODY_KINDS:
            raise ValueError(f"unknown body kind {self.kind!r}")
        if self.role not in ("fixed", "free", "rotary"):
            raise ValueError(f"unknown role {self.role!r}")
        self.group = self.group or self.name
        self.colour = tuple(float(c) for c in self.colour)
        self.pieces = self._build_pieces()

    def _build_pieces(self):
        pieces = []
        specs = list(self.parts) if self.parts else []
        if not specs:
            x, y, zb = self.centre
            if self.kind in ("plane", "aabb", "ridge"):
                specs = [("box", (x, y, zb), tuple(self.size), self.yaw)]
            elif self.kind in ("cylinder", "disk"):
                specs = [("cyl", (x, y, zb), tuple(self.size))]
            else:
                raise ValueError("a loop body needs explicit parts")
        for index, spec in enumerate(specs):
            kind = spec[0]
            cx, cy, zb = spec[1]
            if kind == "box":
                sx, sy, sz = spec[2]
                yaw = float(spec[3]) if len(spec) > 3 else self.yaw
                pieces.append(_Piece("box", hx=sx / 2, hy=sy / 2, z0=0.0, z1=sz, R0=_rot_axis((0, 0, 1), yaw),
                                     t0=(cx, cy, zb), body=self, index=index))
            elif kind == "cyl":
                r, h = spec[2]
                pieces.append(_Piece("cyl", r=r, z0=0.0, z1=h, t0=(cx, cy, zb), body=self, index=index))
            else:
                raise ValueError(f"unknown piece kind {kind!r}")
        return pieces

    @property
    def top_z(self):
        return max(p.t0[2] + p.z1 for p in self.pieces)

    @property
    def bottom_z(self):
        return min(p.t0[2] + p.z0 for p in self.pieces)


class FakeScene:
    """Bodies + mechanisms + the support plane (all evaluator truth).

    ``support_texture``: "wood" (streaks) or "iso"; ``support_texture_std`` ~3 grey reproduces the start view where only
    ~54/143 grid patches pass D15 (std >= 4). ``support_patches``: [{"polygon": [[x, y], ...], "colour": rgb,
    "std": grey}] flat colour patches on the support (no height). ``goal``: [("turnon", rotary_group) |
    ("on", free_group, destination_body)]; ``info``: builder facts for tests (names, truth points, oracle plan)."""

    def __init__(self, bodies, *, support_z=0.900, texture_seed=0, support_colour=(196.0, 176.0, 148.0),
                 support_texture_std=3.0, support_texture="wood", support_texture_scale_m=0.0015, support_patches=(),
                 goal=(), info=None):
        self.bodies = list(bodies)
        names = [b.name for b in self.bodies]
        if len(set(names)) != len(names):
            raise ValueError("body names must be unique")
        self.support_z = float(support_z)
        self.texture_seed = int(texture_seed)
        self.support_colour = tuple(float(c) for c in support_colour)
        self.support_texture_std = float(support_texture_std)
        self.support_texture = support_texture
        self.support_texture_scale_m = float(support_texture_scale_m)
        self.support_patches = [dict(p) for p in support_patches]
        self.goal = [tuple(g) for g in goal]
        self.info = dict(info or {})
        self.groups = {}
        for body in self.bodies:
            g = self.groups.setdefault(body.group, {"bodies": [], "role": None, "hinge": None})
            g["bodies"].append(body)
            if g["role"] is None or body.role != "fixed":
                g["role"] = body.role if g["role"] in (None, "fixed") else g["role"]
            if body.hinge is not None:
                g["hinge"] = body.hinge
        for name, g in self.groups.items():
            if g["role"] == "rotary" and g["hinge"] is None:
                raise ValueError(f"rotary group {name!r} needs a hinge")
        rng = _np.random.default_rng(self.texture_seed)
        for body in self.bodies:
            for piece in body.pieces:
                piece.tex_offset = tuple(float(v) for v in rng.uniform(0, TEXTURE_PX, 2))

    def body(self, name):
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(name)


def _copy_scene(scene):
    """A copy of ``scene`` whose MUTABLE parts (bodies, pieces, hinges, effects, groups) are its own, so the hinge
    state and the effect injections of one FakeTwoViewSim never reach another sim built on the same scene (review
    BF-R5). Geometry, texture offsets and the ``info`` dict are carried over unchanged."""
    import copy as _copy
    twin = _copy.copy(scene)
    body_map, bodies = {}, []
    for body in scene.bodies:
        nb = _copy.copy(body)
        if body.hinge is not None:
            nb.hinge = _copy.deepcopy(body.hinge)
        if body.effect is not None:
            nb.effect = _copy.deepcopy(body.effect)
        nb.pieces = []
        for piece in body.pieces:
            np_ = _copy.copy(piece)
            np_.body = nb
            nb.pieces.append(np_)
        body_map[body.name] = nb
        bodies.append(nb)
    twin.bodies = bodies
    twin.groups = {}
    for name, g in scene.groups.items():
        members = [body_map[b.name] for b in g["bodies"]]
        hinge = next((b.hinge for b in members if b.hinge is not None), None)
        twin.groups[name] = {"bodies": members, "role": g["role"], "hinge": hinge}
    return twin


# ------------------------------------------------------------------------------------------------------------ cameras
class FakeExternalCamera:
    """Static pinhole external camera (truth only): default = the task-2 agentview (SURVEY t2_dump.json: position
    (0.6586, 0, 1.6104), looking toward -x pitched 38.9 deg down, fovy 45 deg -> f 618 px at 512; image right = +y)."""

    def __init__(self, *, position=(0.6586, 0.0, 1.6104), pitch_deg=38.9, yaw_deg=180.0, fovy_deg=45.0, f_px=None,
                 size_px=IMAGE_PX):
        p, y = _math.radians(pitch_deg), _math.radians(yaw_deg)
        self.position = _np.asarray(position, float)
        self.forward = _np.array([_math.cos(p) * _math.cos(y), _math.cos(p) * _math.sin(y), -_math.sin(p)])
        self.right = _unit(_np.cross(self.forward, [0.0, 0.0, 1.0]))
        self.down = _unit(_np.cross(self.forward, self.right))
        self.size_px = int(size_px)
        self.f_px = float(f_px) if f_px is not None else (self.size_px / 2.0) / _math.tan(_math.radians(fovy_deg) / 2.0)
        self.pitch_deg = float(pitch_deg)

    @property
    def axes(self):
        """Columns: world directions of image right (u), image down (v) and the optical axis."""
        return _np.column_stack([self.right, self.down, self.forward])

    def project(self, xyz):
        X = _np.atleast_2d(_np.asarray(xyz, float)) - self.position
        z = X @ self.forward
        c = (self.size_px - 1) / 2.0
        with _np.errstate(divide="ignore", invalid="ignore"):
            uv = _np.c_[c + self.f_px * (X @ self.right) / z, c + self.f_px * (X @ self.down) / z]
        return uv, z

    def jacobian(self, xyz):
        """Truth 2x3 image Jacobian (px per metre of world motion) at a world point."""
        X = _np.asarray(xyz, float) - self.position
        z = float(X @ self.forward)
        a, b = float(X @ self.right), float(X @ self.down)
        f = self.f_px
        return _np.vstack([f * (self.right * z - a * self.forward) / z ** 2, f * (self.down * z - b * self.forward) / z ** 2])

    def ray(self, uv):
        c = (self.size_px - 1) / 2.0
        d = self.axes @ _np.array([(uv[0] - c) / self.f_px, (uv[1] - c) / self.f_px, 1.0])
        return self.position.copy(), _unit(d)


class _WristCam:
    """The R-2 wrist pinhole for a hand pose (truth), identical to FakeWristSim.project with this f."""

    def __init__(self, grip, R, f_px, size_px=IMAGE_PX):
        self.R = R
        hand = grip - GRIP_ALONG_R2_M * R[:, 2]
        self.position = hand + CAMERA_ALONG_R0_M * R[:, 0]
        self.f_px = float(f_px)
        self.size_px = int(size_px)

    @property
    def axes(self):
        # image right = R1, image down = -R0, optical axis = R2
        return _np.column_stack([self.R[:, 1], -self.R[:, 0], self.R[:, 2]])

    def project(self, xyz):
        X = _np.atleast_2d(_np.asarray(xyz, float)) - self.position
        a, b, z = X @ self.R[:, 1], X @ self.R[:, 0], X @ self.R[:, 2]
        c = (self.size_px - 1) / 2.0
        with _np.errstate(divide="ignore", invalid="ignore"):
            uv = _np.c_[c + self.f_px * a / z, c - self.f_px * b / z]
        return uv, z


# --------------------------------------------------------------------------------------------- robot assembly (truth)
def _hand_boxes(q1, q2, *, external):
    """Robot pieces as boxes in the HAND frame about the grip: (name, centre (c0, c1, c2) along (R0, R1, R2), half sizes,
    rgb, texture std). Fingers: the [ROBOT] lowest-band footprint (R0 -5.48..6.14 mm, R1 q-0.1..q+14.29 mm) from the
    tip (grip + 9.25 mm) to the pad top (grip - 11.6 mm), then the wider finger body (R1 q-0.51..q+26.4) up to the palm
    (grip - 31 mm, R6). The palm block is narrowed to +-15 mm in R0 for the wrist view (the real image shows the hand's
    lower hull as a band from v ~430); the upper hand (+-31.6 mm, the asset footprint), the forearm and the wrist link
    are rendered in the external view only (they are behind the wrist camera)."""
    t = FINGERTIP_REACH_M
    pad_top = PAD_SPAN_R2_M[0]
    palm = -_robot.PALM_UNDERSIDE_MM / 1000.0
    boxes = []
    for finger, q in ((1, q1), (2, -q2)):
        sgn = 1.0 if finger == 1 else -1.0
        r0 = FINGER1_R0_MM if finger == 1 else (-FINGER1_R0_MM[1], -FINGER1_R0_MM[0])
        lo, hi = q * 1000 + FINGER1_R1_MM[0], q * 1000 + FINGER1_R1_MM[1]
        c1 = sgn * (lo + hi) / 2000.0
        boxes.append((f"finger{finger}_tip", ((r0[0] + r0[1]) / 2000.0, c1, (t + pad_top) / 2),
                      ((r0[1] - r0[0]) / 2000.0, (hi - lo) / 2000.0, (t - pad_top) / 2), (214.0, 214.0, 210.0), 9.0))
        blo, bhi = q * 1000 + FINGER1_BODY_R1_MM[0], q * 1000 + FINGER1_BODY_R1_MM[1]
        boxes.append((f"finger{finger}_body", ((r0[0] + r0[1]) / 2000.0, sgn * (blo + bhi) / 2000.0, (pad_top + palm) / 2),
                      ((r0[1] - r0[0]) / 2000.0, (bhi - blo) / 2000.0, (pad_top - palm) / 2), (58.0, 58.0, 62.0), 6.0))
    boxes.append(("palm", (0.0, (HAND_R1_MM[0] + HAND_R1_MM[1]) / 2000.0, (palm - 0.075) / 2),
                  (0.015, (HAND_R1_MM[1] - HAND_R1_MM[0]) / 2000.0, (palm + 0.075) / 2), (40.0, 40.0, 44.0), 4.0))
    if external:
        boxes.append(("hand", (0.0, (HAND_R1_MM[0] + HAND_R1_MM[1]) / 2000.0, (-0.075 - GRIP_ALONG_R2_M) / 2),
                      ((HAND_R0_MM[1] - HAND_R0_MM[0]) / 2000.0, (HAND_R1_MM[1] - HAND_R1_MM[0]) / 2000.0,
                       (GRIP_ALONG_R2_M - 0.075) / 2), (226.0, 226.0, 222.0), 14.0))
        boxes.append(("forearm", (0.0, 0.0, -(GRIP_ALONG_R2_M + WRIST_FROM_GRIP_M) / 2),
                      (0.04, 0.04, (WRIST_FROM_GRIP_M - GRIP_ALONG_R2_M) / 2), (232.0, 232.0, 230.0), 12.0))
    return boxes


def _grid_mm(lo, hi, spacing_mm=ROBOT_SAMPLE_MM):
    """Evenly spaced samples from lo to hi (mm) at most ``spacing_mm`` apart (both ends included)."""
    return _np.linspace(lo, hi, max(2, int(_math.ceil((hi - lo) / spacing_mm - 1e-9)) + 1))


def _palm_points_hand():
    """[ROBOT] palm underside sample points (hand frame about the grip): a grid at most ROBOT_SAMPLE_MM (4 mm) apart in
    R1 (xi) and R0 over the asset hand footprint, at p(xi) behind the grip (R6 profile), so any object wider than 4 mm
    under the palm meets a sample (review BF-R4)."""
    xs = _grid_mm(*HAND_R1_MM)
    ps = _np.interp(xs, [r[0] for r in PALM_PROFILE_MM], [r[1] for r in PALM_PROFILE_MM])
    rows = _grid_mm(*HAND_R0_MM)
    pts = [(r0 / 1000.0, x / 1000.0, -p / 1000.0) for r0 in rows for x, p in zip(xs, ps)]
    return _np.asarray(pts, float)


def _hand_top_points():
    """Upper-hand sample points (hand frame about the grip): the top face of the hand box at the hand origin
    (R2 = -97 mm), the asset footprint at most 4 mm apart in R0 and R1; they meet overhangs when the hand retreats or
    lifts."""
    r0 = _grid_mm(*HAND_R0_MM) / 1000.0
    r1 = _grid_mm(*HAND_R1_MM) / 1000.0
    a, b = _np.meshgrid(r0, r1)
    return _np.c_[a.ravel(), b.ravel(), _np.full(a.size, -GRIP_ALONG_R2_M)]


def _ring(a, b):
    """Perimeter samples of the rectangle spanned by the sample vectors a (R0) and b (R1)."""
    return _np.r_[_np.c_[a, _np.full(len(a), b[0])], _np.c_[a, _np.full(len(a), b[-1])],
                  _np.c_[_np.full(len(b), a[0]), b], _np.c_[_np.full(len(b), a[-1]), b]]


def _finger_points_hand(finger, q):
    """Finger sample points (hand frame), at most 4 mm apart: the lowest-band footprint grid at the tip (R1 q-0.1 ..
    q+14.29 mm), its side walls up to the palm, and the wider finger body (R1 q-0.51 .. q+26.4 mm, [ROBOT]) from the
    pad top up to the palm: its side walls and the underside of its outer ledge (the side kind)."""
    sgn = 1.0 if finger == 1 else -1.0
    r0 = FINGER1_R0_MM if finger == 1 else (-FINGER1_R0_MM[1], -FINGER1_R0_MM[0])
    qm = abs(q) * 1000
    a = _grid_mm(*r0) / 1000.0
    b = sgn * _grid_mm(qm + FINGER1_R1_MM[0], qm + FINGER1_R1_MM[1]) / 1000.0
    ga, gb = _np.meshgrid(a, b)
    bottom = _np.c_[ga.ravel(), gb.ravel(), _np.full(ga.size, FINGERTIP_REACH_M)]
    palm = -_robot.PALM_UNDERSIDE_MM
    zs = _grid_mm(palm, FINGERTIP_REACH_M * 1000 - 1.0) / 1000.0
    ring = _ring(a, b)
    sides = [_np.c_[ring, _np.full(len(ring), z)] for z in zs]
    pad_top = PAD_SPAN_R2_M[0] * 1000
    body_b = sgn * _grid_mm(qm + FINGER1_BODY_R1_MM[0], qm + FINGER1_BODY_R1_MM[1]) / 1000.0
    body_ring = _ring(a, body_b)
    sides += [_np.c_[body_ring, _np.full(len(body_ring), z)] for z in _grid_mm(palm, pad_top) / 1000.0]
    ledge = sgn * _grid_mm(qm + FINGER1_R1_MM[1], qm + FINGER1_BODY_R1_MM[1]) / 1000.0
    la, lb = _np.meshgrid(a, ledge)
    sides.append(_np.c_[la.ravel(), lb.ravel(), _np.full(la.size, pad_top / 1000.0)])
    return bottom, _np.vstack(sides)


# --------------------------------------------------------------------------------------------------------- the world
class FakeTwoViewSim:
    """The T1 world (spec §8.3; contract §C15): kinematic or second-order arm, wrist + external cameras, the robot's own
    finger/palm geometry, contacts, grasps, held objects, hinges with the [SP] wedge model and colour-change regions.
    Contacts: the robot's sample points (<= 4 mm apart) and a held object's points are swept against every body piece
    AND the support plane (a half-space below ``scene.support_z``), so fingers, palm and held objects stop on bare
    support exactly as ``heights()`` reports it. The scene's mutable parts (hinges, effects) are copied per sim.

    Compatible with ``FakeV2Sim(wrist=FakeTwoViewSim(...))``: ``observation()`` / ``step(action)`` return a
    ``PolicyObservation``; ``advance(action)`` steps the physics without rendering (state8 only). Extra keyword options
    (beyond the contract signature): ``gap_noise_mm`` (std of each finger-gap reading), ``pixel_noise_grey`` (sparse
    +-1 grey temporal noise, sigma ~ this), ``gripper_stuck`` (no closing motion: v1 Calibrate's default-curve rule),
    ``open_limit_mm``, ``start_gap_mm``, ``images`` (False: blank observation frames for physics-only tests).
    State: ``state`` (state8), ``grip``, ``quat``, ``R``, ``gap_mm``, ``steps``, ``actions``, ``contact_log``.
    Test-only injections: ``inject(kind, **params)``; ``reset()`` restores the constructed state; ``clone()``."""

    def __init__(self, scene, *, eef=(-0.20, 0.019, 1.19), rotvec=(_math.pi, 0.0, 0.0), prompt=None,
                 dynamics="kinematic", translation_ratio=TRANSLATION_RATIO, rotation_ratio=None,
                 f_wrist=_robot.FOCAL_PX_AT_512,
                 start_tilt_deg=0.0, external=None, seed=0, close_drag_m=(0.0, 0.0), held_tilt_deg=0.0,
                 gap_noise_mm=GAP_NOISE_MM, pixel_noise_grey=PIXEL_NOISE_GREY, gripper_stuck=False,
                 open_limit_mm=None, start_gap_mm=START_GAP_MM, images=True):
        if dynamics not in ("kinematic", "second_order"):
            raise ValueError("dynamics must be 'kinematic' or 'second_order'")
        self.scene = _copy_scene(scene)       # sim-owned hinges / effects (two sims on one scene stay independent)
        self.prompt = prompt or scene.info.get("prompt") or TASK2_PROMPT
        self.dynamics = dynamics
        self.translation_ratio = float(translation_ratio)
        self.rotation_ratio = None if rotation_ratio is None else float(rotation_ratio)
        self.f_wrist = float(f_wrist)
        self.external = external or FakeExternalCamera()
        self.seed = int(seed)
        self.close_drag_m = _np.asarray(close_drag_m, float)
        self.held_tilt_deg = float(held_tilt_deg)
        self.gap_noise_mm = float(gap_noise_mm)
        self.pixel_noise_grey = float(pixel_noise_grey)
        self.gripper_stuck = bool(gripper_stuck)
        self.open_limit_mm = None if open_limit_mm is None else float(open_limit_mm)
        self.images = bool(images)          # False: observations carry one cached blank frame (physics-only tests)
        self.textures = _TextureBank(scene.texture_seed)
        R = _rotmat(_np.asarray(rotvec, float))
        if start_tilt_deg:
            R = _rot_axis((0.0, 1.0, 0.0), _math.radians(start_tilt_deg)) @ R
        self._init = {"grip": _np.asarray(eef, float).copy(), "q": _matrix_quat(R), "gap": float(start_gap_mm),
                      "open_limit_mm": self.open_limit_mm,
                      "effect_never": {b.name: bool(b.effect.never) for b in self.scene.bodies if b.effect is not None}}
        self._noise = {}
        rng = _np.random.default_rng(self.seed + 104729)
        for cam in ("wrist", "external"):
            # sparse +-1 grey temporal noise (the real renders' static frames differ by sigma ~0.3-0.8 dLab, M0 effect
            # probe); a share p of the pixels flips by +-1, so sigma = sqrt(p) grey and PNG frames stay compact
            share = min(1.0, self.pixel_noise_grey ** 2)
            self._noise[cam] = [((rng.random((IMAGE_PX, IMAGE_PX, 1)) < share)
                                 * rng.choice(_np.array([-1, 1], _np.int16), (IMAGE_PX, IMAGE_PX, 1))).astype(_np.int16)
                                for _ in range(5)] if self.pixel_noise_grey > 0 else None
        self._pixel_grid = {}
        self._static_ext = None
        self._palm_pts = _palm_points_hand()
        self._top_pts = _hand_top_points()
        self.reset()

    # ------------------------------------------------------------------------------------------------------- state
    def reset(self):
        """Back to the constructed initial state (bodies, joints, arm, gripper, injections); returns the observation."""
        self.grip = self._init["grip"].copy()
        self.quat = self._init["q"].copy()
        g = self._init["gap"] / 2000.0
        self.q1, self.q2 = g, -g
        self.vel = _np.zeros(3)
        self.omega = _np.zeros(3)
        self.steps = 0
        self.actions = []
        self.contact_log = []
        self.events = []
        self._held = None
        self._grip_cmd = 0.0
        self._phase = 0
        self._deflect = [0.0, 0.0]
        self._finger_contact = [None, None]
        self._clamp_age = 0
        self._injections = []
        self.open_limit_mm = self._init["open_limit_mm"]                 # injections restored (review BF-R5)
        for name, never in self._init["effect_never"].items():
            self.scene.body(name).effect.never = never
        self._rng = _np.random.default_rng(self.seed)
        self._group = {}
        for name, g in self.scene.groups.items():
            self._group[name] = {"R": _np.eye(3), "t": _np.zeros(3), "version": 0, "moved_step": -10 ** 6}
            if g["hinge"] is not None:
                h = g["hinge"]
                h.q, h.qd, h.last_torque = h.q0, 0.0, 0.0
            for body in g["bodies"]:
                if body.tilt_deg:
                    self._tip_group(name, _np.array([1.0, 0.0]), _math.radians(body.tilt_deg), record=False)
        self._close_offset_mm = 0.0
        self._lost_group = None
        self._q_noise = _np.zeros(2)
        self._effect_on_since = {}
        self._effect_state = {}
        self._turnon_seen = {}
        self._obs_cache = None
        self._render_cache = {}
        self._static_ext = None
        self._update_effects()
        self.state = self._state8()
        return self.observation()

    @property
    def R(self):
        return _quat_matrix(self.quat)

    @property
    def gap_mm(self):
        return float((self.q1 - self.q2) * 1000.0)

    def wrist_camera(self):
        return _WristCam(self.grip, self.R, self.f_wrist)

    # ------------------------------------------------------------------------------------------------- injections
    def inject(self, kind, **params):
        """Test-only perturbations (truth side). Kinds:
        ``close_offset_mm`` (mm): the next clamped/empty close ends ``mm`` off; ``slip`` (steps): the pads lose the
        rotary part (friction off, the wedge relaxes: a K7 gap fall) for n steps; ``hold_lost``: the held object/part is
        dropped and the fingers close on the empty curve; ``tug`` (vector_m, steps): the grip is displaced off the hinge
        orbit by vector_m per step; ``drift`` (vector_m, steps): an uncommanded hand drift per step; ``lateral_block``
        (steps): lateral (xy) hand motion is not realized; ``tilt`` (deg): the held object tilts in the pinch about the
        jaw; ``open_limit_mm`` (mm): the gripper cannot open beyond mm (at the clamp width a hold is never released:
        the release_incomplete world); ``effect_never`` (name): that region never changes; ``set_joint`` (group, q);
        ``move_group`` (group, vector_m). ``reset()`` undoes every injection."""
        item = {"kind": kind, **params, "at_step": self.steps}
        self._obs_cache = None
        if kind == "tilt":
            if self._held is not None and self._held["kind"] == "free":
                self._tilt_held(_math.radians(float(params.get("deg", 15.0))))
        elif kind == "hold_lost":
            lost = self._held["group"] if self._held is not None else None
            self._release(reason="hold_lost")
            self._lost_group = lost                     # the pads slipped off it: they close past it until opened
            self._grip_cmd, self._phase = 1.0, 30
        elif kind == "open_limit_mm":
            self.open_limit_mm = float(params["mm"])
        elif kind == "effect_never":
            for body in self.scene.bodies:
                if body.effect is not None and (params.get("name") in (None, body.name)):
                    body.effect.never = True
        elif kind == "set_joint":
            g = self.scene.groups[params["group"]]
            g["hinge"].q = float(params["q"])
            self._place_rotary(params["group"])
        elif kind == "move_group":
            st = self._group[params["group"]]
            st["t"] = st["t"] + _np.asarray(params["vector_m"], float)
            self._touch_group(params["group"])
        else:
            if kind not in ("close_offset_mm", "slip", "tug", "drift", "lateral_block"):
                raise ValueError(f"unknown injection {kind!r}")
            item["left"] = int(params.get("steps", 1))
            self._injections.append(item)
        self.events.append(item)
        return item

    def _active(self, kind):
        return [i for i in self._injections if i["kind"] == kind and i.get("left", 0) > 0]

    def _tick_injections(self):
        for item in self._injections:
            if item["kind"] != "close_offset_mm" and item.get("left", 0) > 0:
                item["left"] -= 1

    # ----------------------------------------------------------------------------------------------- group poses
    def _touch_group(self, name):
        st = self._group[name]
        st["version"] += 1
        st["moved_step"] = self.steps

    def _place_rotary(self, name):
        h = self.scene.groups[name]["hinge"]
        c = _np.array([h.centre_xy[0], h.centre_xy[1], 0.0])
        Rg = _rot_axis(h.axis, h.q - h.q0)
        st = self._group[name]
        st["R"], st["t"] = Rg, c - Rg @ c
        self._touch_group(name)

    def piece_pose(self, piece):
        st = self._group[piece.body.group]
        return st["R"] @ piece.R0, st["R"] @ piece.t0 + st["t"]

    def _world_point(self, group, x0):
        st = self._group[group]
        return st["R"] @ _np.asarray(x0, float) + st["t"]

    def body_pose(self, name):
        """Truth: the body's footprint-centre bottom point, rotation, yaw and tilt in the current world."""
        body = self.scene.body(name)
        st = self._group[body.group]
        R = st["R"] @ _rot_axis((0, 0, 1), body.yaw)
        xyz = st["R"] @ _np.asarray(body.centre, float) + st["t"]
        return {"xyz": xyz, "R": R, "yaw_rad": _yaw_of(R), "tilt_deg": _tilt_deg(R), "quat_xyzw": _matrix_quat(R),
                "top_z": max(float((st["R"] @ (p.t0 + p.R0 @ [0.0, 0.0, p.z1]) + st["t"])[2]) for p in body.pieces)}

    def joint(self, name):
        """Truth: the hinge q of a rotary group (or of the group of a named body)."""
        group = name if name in self.scene.groups else self.scene.body(name).group
        h = self.scene.groups[group]["hinge"]
        if h is None:
            raise KeyError(f"{name!r} has no hinge")
        return float(h.q)

    def held(self):
        """Truth: what the gripper holds (None, or {group, kind, since_step, gap_clamp_mm, tilt_deg})."""
        if self._held is None:
            return None
        h = self._held
        out = {"group": h["group"], "kind": h["kind"], "since_step": h["since"], "gap_clamp_mm": h["gap_clamp_mm"],
               "lifted": bool(h.get("lifted", False))}
        if h["kind"] == "free":
            out["tilt_deg"] = _tilt_deg(self._group[h["group"]]["R"])
        return out

    # ------------------------------------------------------------------------------------------------- the pieces
    def _scene_pieces(self, *, exclude_groups=()):
        out = []
        for body in self.scene.bodies:
            if body.group in exclude_groups:
                continue
            for piece in body.pieces:
                R, t = self.piece_pose(piece)
                out.append((piece, R, t))
        return out

    def heights(self, xy, *, exclude_groups=()):
        """Truth: the highest collision surface per world xy (support plane included; the robot is never included)."""
        P = _np.atleast_2d(_np.asarray(xy, float))
        z = _np.full(len(P), self.scene.support_z)
        origin = _np.c_[P, _np.full(len(P), 5.0)]
        down = _np.tile([0.0, 0.0, -1.0], (len(P), 1))
        for piece, R, t in self._scene_pieces(exclude_groups=exclude_groups):
            o = (origin - t) @ R
            d = down @ R
            t_in, t_out, _ = piece.hits(o, d)
            hit = (t_in <= t_out) & (t_out > 0)
            z = _np.where(hit, _np.maximum(z, 5.0 - _np.maximum(t_in, 0.0)), z)
        return z

    # ------------------------------------------------------------------------------------------------- rendering
    def _rays(self, cam):
        key = (cam.size_px, round(cam.f_px, 6))
        if key not in self._pixel_grid:
            c = (cam.size_px - 1) / 2.0
            u, v = _np.meshgrid(_np.arange(cam.size_px, dtype=_np.float32), _np.arange(cam.size_px, dtype=_np.float32))
            self._pixel_grid[key] = (((u - c) / cam.f_px).astype(_np.float32), ((v - c) / cam.f_px).astype(_np.float32))
        return self._pixel_grid[key]

    def _robot_pieces(self, external):
        R = self.R
        out = []
        boxes = _hand_boxes(self.q1, self.q2, external=external)
        for index, (name, c, half, rgb, std) in enumerate(boxes):
            piece = _Piece("box", hx=half[0], hy=half[1], z0=-half[2], z1=half[2], body=None, index=1000 + index)
            piece.tex_offset = (97.0 * index, 31.0 * index)
            out.append((piece, R, self.grip + R @ _np.asarray(c, float), name, rgb, std))
        if external:
            wrist = self.grip - WRIST_FROM_GRIP_M * R[:, 2]
            shoulder = _np.asarray(SHOULDER_XYZ, float)
            axis = shoulder - wrist
            length = float(_np.linalg.norm(axis))
            if length > 0.05:
                z = axis / length
                x = _unit(_np.cross(z, [0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else _np.cross(z, [1.0, 0.0, 0.0]))
                y = _np.cross(z, x)
                piece = _Piece("box", hx=0.045, hy=0.045, z0=0.0, z1=length, index=1100)
                piece.tex_offset = (401.0, 77.0)
                out.append((piece, _np.column_stack([x, y, z]), wrist, "link", (222.0, 222.0, 220.0), 10.0))
        return out

    def _bbox(self, cam, piece, R, t):
        corners = (piece.corners_local() @ R.T) + t
        uv, z = cam.project(corners)
        if _np.any(~_np.isfinite(z)) or _np.any(z <= 0.02):
            if _np.all(z <= NEAR_M):
                return None
            return 0, cam.size_px, 0, cam.size_px
        u0, v0 = _np.floor(uv.min(axis=0)) - 1
        u1, v1 = _np.ceil(uv.max(axis=0)) + 2
        u0, v0 = int(max(0, u0)), int(max(0, v0))
        u1, v1 = int(min(cam.size_px, u1)), int(min(cam.size_px, v1))
        if u0 >= u1 or v0 >= v1:
            return None
        return v0, v1, u0, u1

    def _shade_piece(self, cam, piece, R, t, box, rgb_buf, depth_buf, id_buf, ident, colour, std, *, body=None,
                     camera_name="wrist"):
        v0, v1, u0, u1 = box
        xs, ys = self._rays(cam)
        xs, ys = xs[v0:v1, u0:u1].ravel(), ys[v0:v1, u0:u1].ravel()
        M = (R.T @ cam.axes).astype(_np.float32)                         # local direction = M @ (x, y, 1)
        d = _np.empty((xs.size, 3), _np.float32)
        for i in range(3):
            d[:, i] = M[i, 0] * xs + M[i, 1] * ys + M[i, 2]
        o = (R.T @ (cam.position - t)).astype(_np.float32)
        t_in, t_out, face = piece.hits(o, d)
        cur = depth_buf[v0:v1, u0:u1]
        win = (t_in <= t_out) & (t_in > NEAR_M) & (t_in < cur.ravel())
        if not win.any():
            return
        tt = t_in[win]
        p = o + d[win] * tt[:, None]
        f = face[win]
        uu, vv = piece.face_uv(p, f)
        scale = body.texture_scale_m if body is not None else 0.002
        kind = body.texture if body is not None else "iso"
        texels_per_px = float(tt[:: max(1, tt.size // 64)].mean()) / cam.f_px / scale
        tex = self.textures.sample(kind, uu / scale + piece.tex_offset[0], vv / scale + piece.tex_offset[1],
                                   texels_per_px)
        side = body.side_gain if body is not None else SIDE_GAIN
        gain = _np.where(f >= 4, _np.float32(1.0), _np.float32(side))
        rgb = _np.asarray(colour, _np.float32)[None, :] * gain[:, None] + (_np.float32(std) * tex)[:, None]
        if body is not None:
            if body.pattern == "rings":
                rr = _np.hypot(p[:, 0], p[:, 1])
                ring = (f >= 4) & (rr < 0.085) & (_np.mod(rr, 0.018) < 0.006)
                rgb[ring] -= 22.0
            if body.effect is not None and self._effect_state.get(body.name):
                e = body.effect
                local_c = piece.R0.T @ (_np.asarray([e.centre_xy[0], e.centre_xy[1], 0.0]) - piece.t0)
                rr = _np.hypot(p[:, 0] - local_c[0], p[:, 1] - local_c[1])
                ring = (f == 5) & (rr >= e.r_in) & (rr <= e.r_out)
                rgb[ring] += _np.asarray(e.chroma_rgb, _np.float32)[None, :]
            if camera_name == "external" and body.reflect_grey > 0:
                rgb += self._reflection(body, piece, p, f)[:, None].astype(_np.float32)
        win2 = win.reshape(v1 - v0, u1 - u0)
        rgb_buf[v0:v1, u0:u1][win2] = rgb
        cur[win2] = tt
        id_buf[v0:v1, u0:u1][win2] = ident

    def _reflection(self, body, piece, p, face):
        """Arm-reflection-like LUMINANCE pattern on a metal top face (external view): bands whose phase follows the hand
        pose and whose amplitude falls with the hand's distance; static when the hand is static (the M0 effect probe:
        16-30 % of the burner plate changes |dLab| > 12 while the arm moves near it, the chroma stays)."""
        R = self.R
        tips = self.grip + FINGERTIP_REACH_M * R[:, 2]
        centre = self._world_point(body.group, piece.t0 + piece.R0 @ [0.0, 0.0, piece.z1])
        dist = float(_np.linalg.norm(tips - centre))
        amp = body.reflect_grey * _math.exp(-max(0.0, dist - 0.08) / 0.22)
        yaw = _yaw_of(R)
        phase = 2 * _math.pi * (tips[0] + 0.6 * tips[1]) / 0.035 + 2.0 * yaw + 9.0 * tips[2]
        wave = _np.sin(2 * _math.pi * (p[:, 0] * 0.8 + p[:, 1] * 0.6) / 0.03 + phase)
        return _np.where(face >= 4, amp * _np.clip(wave, 0.0, 1.0) ** 3, 0.0)

    def _render_support(self, cam, rgb_buf, depth_buf, id_buf):
        xs, ys = self._rays(cam)
        A = cam.axes.astype(_np.float32)
        C = cam.position.astype(_np.float32)
        dz = A[2, 0] * xs + A[2, 1] * ys + A[2, 2]
        with _np.errstate(divide="ignore", invalid="ignore"):
            t = (_np.float32(self.scene.support_z) - C[2]) / dz
        valid = (dz < 0) & (t > NEAR_M)
        t = _np.where(valid, t, _np.float32(_np.inf))
        tv = _np.where(valid, t, 0)
        hx = C[0] + tv * (A[0, 0] * xs + A[0, 1] * ys + A[0, 2])
        hy = C[1] + tv * (A[1, 0] * xs + A[1, 1] * ys + A[1, 2])
        scale = self.scene.support_texture_scale_m
        sub = t[::32, ::32]
        sub = sub[_np.isfinite(sub)]
        med = float(_np.median(sub)) if sub.size else 1.0
        tex = self.textures.sample(self.scene.support_texture, hx / scale + 13.0, hy / scale + 29.0,
                                   med / cam.f_px / scale)
        base = _np.asarray(self.scene.support_colour, _np.float32)
        rgb_buf[...] = base[None, None, :] + (_np.float32(self.scene.support_texture_std) * tex)[..., None]
        for patch in self.scene.support_patches:
            poly = _np.asarray(patch["polygon"], float)
            inside = _points_in_convex(_np.c_[hx.ravel(), hy.ravel()], poly).reshape(hx.shape) & valid
            if inside.any():
                ptex = self.textures.sample("iso", hx[inside] / 0.002 + 211.0, hy[inside] / 0.002 + 5.0, 1.0)
                rgb_buf[inside] = (_np.asarray(patch["colour"], _np.float32)[None, :]
                                   + (_np.float32(patch.get("std", 1.0)) * ptex)[:, None])
        rgb_buf[~valid] = (120.0, 120.0, 124.0)
        depth_buf[...] = t
        id_buf[...] = _np.where(valid, 0, -1)

    def _render(self, camera):
        """Render one 512 upright RGB view with its depth (ray parameter along the optical axis) and body-id buffers
        (id 0 support, 1 + scene body index, 1000 + robot piece, -1 none). Bodies and the robot are exact ray-cast
        boxes/cylinders (the height map's own primitives, depth-tested per pixel within each piece's screen bbox) with
        world-anchored, mip-filtered textures, so features move rigidly and edges are exact to subpixel."""
        key = (camera, self.steps, self._version_key())
        if key in self._render_cache:
            return self._render_cache[key]
        external = camera == "external"
        cam = self.external if external else self.wrist_camera()
        n = cam.size_px
        if external:
            rgb, depth, ident, dynamic = self._external_static()
            rgb, depth, ident = rgb.copy(), depth.copy(), ident.copy()
        else:
            rgb = _np.empty((n, n, 3), _np.float32)
            depth = _np.empty((n, n), _np.float32)
            ident = _np.empty((n, n), _np.int16)
            self._render_support(cam, rgb, depth, ident)
            dynamic = None
        index_of = {b.name: i for i, b in enumerate(self.scene.bodies)}
        for piece, R, t in self._scene_pieces():
            body = piece.body
            if dynamic is not None and body.name not in dynamic:
                continue
            box = self._bbox(cam, piece, R, t)
            if box is not None:
                self._shade_piece(cam, piece, R, t, box, rgb, depth, ident, 1 + index_of[body.name], body.colour,
                                  body.texture_std, body=body, camera_name=camera)
        for piece, R, t, name, colour, std in self._robot_pieces(external):
            box = self._bbox(cam, piece, R, t)
            if box is not None:
                self._shade_piece(cam, piece, R, t, box, rgb, depth, ident, piece.index, colour, std, camera_name=camera)
        noise = self._noise.get(camera)
        if noise is not None:
            rgb = rgb + noise[self.steps % len(noise)]
        out = (_np.clip(_np.round(rgb), 0, 255).astype(_np.uint8), depth, ident)
        self._render_cache = {key: out, **{k: v for k, v in self._render_cache.items() if k[1] == self.steps}}
        return out

    def _version_key(self):
        return tuple((name, st["version"]) for name, st in sorted(self._group.items())) + tuple(
            sorted((k, bool(v)) for k, v in self._effect_state.items())) + (round(self.q1, 7), round(self.q2, 7))

    def _external_static(self):
        """Cached external layer: the support and every body whose group has not moved for EXTERNAL_DYNAMIC_STEPS and
        has no time-varying appearance; the rest is rendered per frame on top (depth-tested)."""
        dynamic = set()
        held_group = self._held["group"] if self._held is not None else None
        for body in self.scene.bodies:
            st = self._group[body.group]
            if (body.group == held_group or self.steps - st["moved_step"] <= EXTERNAL_DYNAMIC_STEPS
                    or body.effect is not None or body.reflect_grey > 0):
                dynamic.add(body.name)
        static_key = tuple((b.name, self._group[b.group]["version"]) for b in self.scene.bodies if b.name not in dynamic)
        if self._static_ext is None or self._static_ext[0] != static_key:
            cam = self.external
            n = cam.size_px
            rgb = _np.empty((n, n, 3), _np.float32)
            depth = _np.empty((n, n), _np.float32)
            ident = _np.empty((n, n), _np.int16)
            self._render_support(cam, rgb, depth, ident)
            index_of = {b.name: i for i, b in enumerate(self.scene.bodies)}
            for piece, R, t in self._scene_pieces():
                if piece.body.name in dynamic:
                    continue
                box = self._bbox(cam, piece, R, t)
                if box is not None:
                    self._shade_piece(cam, piece, R, t, box, rgb, depth, ident, 1 + index_of[piece.body.name],
                                      piece.body.colour, piece.body.texture_std, body=piece.body, camera_name="external")
            self._static_ext = (static_key, rgb, depth, ident)
        _, rgb, depth, ident = self._static_ext
        return rgb, depth, ident, dynamic

    def wrist_upright(self):
        return self._render("wrist")[0]

    def external_upright(self):
        return self._render("external")[0]

    def observation(self):
        if self._obs_cache is not None and self._obs_cache[0] == self.steps:
            return self._obs_cache[1]
        if not self.images:
            blank = _blank_url()
            obs = _PolicyObservation(blank, blank, tuple(float(v) for v in self._state8()), self.prompt)
            self._obs_cache = (self.steps, obs)
            return obs
        wrist = _np.ascontiguousarray(self.wrist_upright()[:, ::-1])
        ext = _np.ascontiguousarray(self.external_upright()[:, ::-1])
        obs = _PolicyObservation(_png_data_url(ext), _png_data_url(wrist), tuple(float(v) for v in self._state8()),
                                 self.prompt)
        self._obs_cache = (self.steps, obs)
        return obs

    # ---------------------------------------------------------------------------------------------- truth accessors
    def project(self, camera, xyz):
        """Truth: upright pixel(s) and depth(s) of world point(s) in ``camera`` ("wrist" | "external")."""
        cam = self.external if camera == "external" else self.wrist_camera()
        uv, z = cam.project(xyz)
        if _np.ndim(xyz) == 1:
            return (uv[0] if z[0] > NEAR_M else None), float(z[0])
        return uv, z

    def truth_mask(self, camera, name):
        """Truth: pixels where ``name`` (a body, a group, "robot" or "support") is the visible surface."""
        _, _, ident = self._render(camera)
        if name == "robot":
            return ident >= 1000
        if name == "support":
            return ident == 0
        names = [b.name for b in self.scene.bodies]
        if name in names:
            return ident == 1 + names.index(name)
        if name in self.scene.groups:
            ids = [1 + i for i, b in enumerate(self.scene.bodies) if b.group == name]
            return _np.isin(ident, ids)
        raise KeyError(name)

    def depth_map(self, camera):
        """Truth: per-pixel depth along the optical axis (inf where nothing is hit)."""
        return self._render(camera)[1]

    def colour_region_changed(self, name):
        """Truth: the effect region of body ``name`` renders changed (on) at the current step."""
        return bool(self._effect_state.get(name, False))

    def effect_onset(self, name):
        """Truth: {"q_cross_step", "on_step"} for the effect region of ``name`` (None before it happens)."""
        return self._effect_on_since.get(name)

    # ------------------------------------------------------------------------------------------------------ dynamics
    def step(self, action):
        """One 20 Hz control step; returns the PolicyObservation after it."""
        self.advance(action)
        return self.observation()

    def advance(self, action):
        """One control step WITHOUT rendering (physics only; tests that read only state8 / truth)."""
        a = _np.asarray(action, float)
        if a.shape != (7,) or not _np.isfinite(a).all() or _np.any(_np.abs(a) > 1 + 1e-9):
            raise ValueError("invalid fake action")
        self.actions.append([float(v) for v in a])
        p0, q0 = self.grip.copy(), self.quat.copy()
        rot = self._rotation_step(a)
        cmd = TRANSLATION_PER_UNIT * a[:3]
        dp = self._translation_step(a)
        for item in self._active("drift"):
            dp = dp + _np.asarray(item["vector_m"], float)
        if self._active("lateral_block"):
            dp[:2] = 0.0
        if self._held is not None and self._held["kind"] == "rotary":
            dp, rot = self._rotary_motion(dp, rot)
        elif self._held is not None and self._held["kind"] == "fixed":
            dp = dp * 0.02
        self._move_with_contacts(p0, q0, dp, rot, cmd)
        if self._held is not None and self._held["kind"] == "free":
            self._follow_held()
        self._gripper_step(float(a[6]))
        for name, g in self.scene.groups.items():
            h = g["hinge"]
            if h is not None and not (self._held is not None and self._held["group"] == name):
                before = h.q
                h.relax()
                if h.q != before:
                    self._place_rotary(name)
        self.steps += 1
        self._sample_noise()
        self._update_effects()
        self._tick_injections()
        self.state = self._state8()
        return self.state

    def _sample_noise(self):
        sigma = self.gap_noise_mm / _math.sqrt(2.0) / 1000.0
        self._q_noise = self._rng.normal(0.0, sigma, 2) if sigma > 0 else _np.zeros(2)

    def _state8(self):
        rv = _q2rv(self.quat)
        n = getattr(self, "_q_noise", _np.zeros(2))
        q1 = float(_np.clip(self.q1 + n[0], 0.0, FINGER_Q_MAX_M))
        q2 = float(_np.clip(self.q2 + n[1], -FINGER_Q_MAX_M, 0.0))
        return _np.array([*self.grip, *rv, q1, q2], float)

    def _rotation_step(self, a):
        w_cmd = ROTATION_PER_UNIT * a[3:6]
        c = float(_np.linalg.norm(w_cmd))
        if c < 1e-12:
            target = _np.zeros(3)
        else:
            rate = self.rotation_ratio * c if self.rotation_ratio is not None else rotation_response_rad(c)
            target = w_cmd / c * rate * self._yaw_load_factor()
        omega_new = target + (self.omega - target) * ROT_LAG
        rot = 0.5 * (self.omega + omega_new)
        self.omega = omega_new
        return rot

    def _yaw_load_factor(self):
        if self._held is None or self._held["kind"] != "rotary":
            return 1.0
        h = self.scene.groups[self._held["group"]]["hinge"]
        f = 1.0 - YAW_LOAD_K_TAU * h.last_torque / max(h.a_same, 1e-12) \
            - YAW_LOAD_K_PHI * max(0.0, abs(self._held["phi"]) - YAW_LOAD_PHI0)
        return float(_np.clip(f, 0.5, 1.0))

    def _translation_step(self, a):
        cmd = TRANSLATION_PER_UNIT * a[:3]
        if self.dynamics == "kinematic":
            return self.translation_ratio * cmd
        goal = cmd * (self.translation_ratio / SECOND_ORDER_STEADY)
        pos, vel = _np.zeros(3), self.vel.copy()
        dt = 0.05 / SECOND_ORDER_SUBSTEPS
        w, z = SECOND_ORDER_W, SECOND_ORDER_Z
        for _ in range(SECOND_ORDER_SUBSTEPS):
            vel = vel + (w * w * (goal - pos) - 2.0 * z * w * vel) * dt
            pos = pos + vel * dt
        self.vel = vel
        return pos

    # ---------------------------------------------------------------------------------------------- rotary hold
    def _rotary_motion(self, dp, rot):
        st = self._held
        h = self.scene.groups[st["group"]]["hinge"]
        dpsi = float(rot @ h.axis)
        slipping = bool(self._active("slip"))
        phi = h.advance(dpsi, st["phi"], slipping=slipping)
        if slipping:
            phi *= 0.3                                       # the pads slide round the part: the wedge relaxes (K7)
            st["slipped"] = True
        elif st.get("slipped"):
            st["q_close"], st["p_close"], st["slipped"] = h.q, self.grip.copy(), False
        st["phi"] = phi
        self._place_rotary(st["group"])
        c = _np.asarray(h.centre_xy, float)
        rel = _np.r_[st["p_close"][:2] - c, 0.0]
        orbit = c + (_rot_axis(h.axis, h.q - st["q_close"]) @ rel)[:2]
        for item in self._active("tug"):
            st["tug"] = st["tug"] + _np.asarray(item["vector_m"], float)[:2]
        free_xy = self.grip[:2] + dp[:2]
        xy = orbit + (free_xy - orbit) * (1.0 - ORBIT_STIFFNESS) + st["tug"]
        dp = _np.r_[xy - self.grip[:2], dp[2] * 0.02]
        excess = h.gap_excess_mm(phi)
        self._set_clamped_gap(st["gap_clamp_mm"] + excess)
        self.contact_log.append({"step": self.steps, "kind": "twist", "group": st["group"], "q": h.q, "phi": phi,
                                 "gap_mm": self.gap_mm})
        return dp, rot

    def _set_clamped_gap(self, gap_mm):
        st = self._held
        centre = st["clamp_centre_m"]
        half = gap_mm / 2000.0
        self.q1, self.q2 = centre + half, centre - half

    # ------------------------------------------------------------------------------------------------- contacts
    def _assembly_points(self):
        """Robot + held-object sample points in the HAND frame about the grip, with their kinds and finger ids."""
        pts, kinds = [], []
        for finger, q in ((1, self.q1), (2, self.q2)):
            bottom, sides = _finger_points_hand(finger, q)
            pts += [bottom, sides]
            kinds += [_np.full(len(bottom), finger), _np.full(len(sides), 10 + finger)]
        pts.append(self._palm_pts)
        kinds.append(_np.full(len(self._palm_pts), 20))
        pts.append(self._top_pts)
        kinds.append(_np.full(len(self._top_pts), 21))
        if self._held is not None and self._held["kind"] == "free":
            world = self._held_points_world()
            R = self.R
            pts.append((world - self.grip) @ R)
            kinds.append(_np.full(len(world), 30))
        return _np.vstack(pts), _np.concatenate(kinds)

    def _held_points_world(self):
        st = self._held
        if "points0" not in st:
            pts = []
            for body in self.scene.groups[st["group"]]["bodies"]:
                for piece in body.pieces:
                    local = piece.sample_points_local(0.006)
                    pts.append(local @ piece.R0.T + piece.t0)
            st["points0"] = _np.vstack(pts)
        g = self._group[st["group"]]
        return st["points0"] @ g["R"].T + g["t"]

    def _move_with_contacts(self, p0, q0, dp, rot, cmd):
        self._pressed_fingers = set()
        q1 = _qmul(_rvq(rot), q0)
        if float(_np.linalg.norm(dp)) < 1e-12 and float(_np.linalg.norm(rot)) < 1e-12:
            self.grip, self.quat = p0 + dp, q1
            self._relax_deflection(set())
            return
        local, kinds = self._assembly_points()
        exclude = () if self._held is None else (self._held["group"],)
        hit = self._first_contact(p0, q0, dp, rot, local, exclude)
        if hit is None:
            self.grip, self.quat = p0 + dp, q1
            self._relax_deflection(set())
            return
        lo, hi, blocked, owner_hi = hit["lo"], hit["hi"], hit["blocked"], hit["owner"]
        pose, free_at = hit["pose"], hit["free_at"]
        blocked_kinds = set(int(k) for k in kinds[blocked])
        owners = {int(o) for o in owner_hi[blocked]}
        support_hit = SUPPORT_OWNER in owners
        pieces = [hit["cands"][o][0] for o in owners if o >= 0]
        p_c, R_c = pose(lo)
        q_c = _qmul(_rvq(lo * rot), q0)
        rest = (1.0 - lo) * dp
        # a palm pressing DOWN on a free object's OVERHANG (the contact beyond the object's base footprint) tips the
        # object about its base edge; the hand then goes on only as far as the tipped object lets it (review BF-R2).
        # A press inside the footprint, a tilt at TIP_MAX_DEG or any other contact blocks like a fixed body.
        groups = {p.body.group for p in pieces}
        if blocked_kinds == {20} and not support_hit and len(groups) == 1 and dp[2] < -1e-9 and rest[2] < 0:
            group = next(iter(groups))
            if (self.scene.groups[group]["role"] == "free"
                    and not (self._held is not None and self._held["group"] == group)):
                p_h, R_h = pose(hi)
                contact_xy = (local[blocked] @ R_h.T + p_h)[:, :2].mean(axis=0)
                tip = self._palm_tip(group, contact_xy, -float(rest[2]))
                if tip is not None:
                    after = self._first_contact(p_c, q_c, rest, (1.0 - lo) * rot, local, exclude)
                    if after is None:
                        self.grip, self.quat = p0 + dp, q1
                    else:
                        self.grip = p_c + after["lo"] * rest
                        self.quat = _qmul(_rvq((lo + after["lo"] * (1.0 - lo)) * rot), q0)
                        if self.dynamics == "second_order":
                            self.vel = self.vel * 0.0
                    if self._held is not None and self._held["kind"] == "free":
                        self._follow_held()
                    self.contact_log.append({"step": self.steps, "kind": "palm_tip", "group": group,
                                             "push_mm": -float(rest[2]) * 1000.0, **tip,
                                             "grip_z": float(self.grip[2])})
                    self._relax_deflection(set())
                    return
        downward = dp[2] < -1e-9 and blocked_kinds <= {1, 2, 20, 30}
        if downward and rest[2] < 0:
            slide = _np.r_[rest[:2], 0.0]
            if float(_np.linalg.norm(slide)) > 1e-9 and free_at(p_c + slide, R_c):
                p_c = p_c + slide
        if not downward and _np.any(_np.abs(rest[:2]) > 0) and abs(rest[2]) > 1e-9 and not self._active("lateral_block"):
            vert = _np.r_[0.0, 0.0, rest[2]]
            if free_at(p_c + vert, R_c):
                p_c = p_c + vert
        self.grip, self.quat = p_c, q_c
        if self._held is not None and self._held["kind"] == "free":
            self._follow_held()                                     # the object came along up to the contact
        if self.dynamics == "second_order":
            self.vel = self.vel * 0.0
        landing = {k for k in blocked_kinds if k in (1, 2)}
        names = sorted({p.body.name for p in pieces} | ({"support"} if support_hit else set()))
        event = {"step": self.steps, "kind": "contact", "point_kinds": sorted(blocked_kinds), "bodies": names,
                 "fraction": lo, "grip_z": float(p_c[2])}
        if landing and dp[2] < 0:
            event["kind"] = "finger_landing"
            event["fingers"] = sorted(landing)
            surface = float(p_c[2] - FINGERTIP_REACH_M)
            event["surface_z"] = surface
            for finger in landing:
                push_mm = abs(rest[2]) * 1000.0
                self._deflect_finger(finger, float(_np.clip(FINGER_DEFLECT_GAIN * push_mm, *FINGER_DEFLECT_MM)))
            self._pressed_fingers = set(landing)
        elif 30 in blocked_kinds:
            event["kind"] = "held_support" if dp[2] < 0 else "held_obstacle"
            if dp[2] < 0 and cmd[2] < 0:
                self._held_press(cmd)
            elif self._held is not None:
                self._open_pinch(HELD_OBSTACLE_GAP_RISE_MM)
        elif 20 in blocked_kinds:
            event["kind"] = "palm"
        elif 21 in blocked_kinds:
            event["kind"] = "hand_top"
        else:
            event["kind"] = "finger_side"
            for finger in (1, 2):
                if 10 + finger in blocked_kinds and self._held is None:
                    self._deflect_finger(finger, float(_np.clip(FINGER_DEFLECT_GAIN * float(_np.linalg.norm(rest)) * 1000,
                                                                *FINGER_DEFLECT_MM)))
        self._relax_deflection(landing | {k - 10 for k in blocked_kinds if k in (11, 12)})
        self.contact_log.append(event)

    def _first_contact(self, p0, q0, dp, rot, local, exclude):
        """First contact of the sample points ``local`` (hand frame about the grip) along the motion p0 + s dp,
        rvq(s rot) q0, s in [0, 1], against the scene pieces (the ``exclude`` groups left out) and the SUPPORT PLANE
        (a half-space collider below support_z, review BF-R1): None when the whole motion is free, else a record with
        the last free fraction ``lo`` (9 bisections), ``hi``, the ``blocked`` point mask, the per-point ``owner``
        (candidate index, SUPPORT_OWNER, -1), ``cands``, ``pose(s)`` and ``free_at(p, R)``. Points already inside at
        s = 0 never block (a motion may always leave a penetration it started in)."""
        cands, swept_min_z = self._candidates(p0, q0, dp, rot, local, exclude)
        support_z = self.scene.support_z - SUPPORT_CONTACT_EPS_M
        use_support = swept_min_z < support_z
        if not cands and not use_support:
            return None

        def hits_at(p, R):
            world = local @ R.T + p
            hit = world[:, 2] < support_z if use_support else _np.zeros(len(local), bool)
            owner = _np.where(hit, SUPPORT_OWNER, -1)
            for k, (piece, Rp, tp) in enumerate(cands):
                mask = piece.contains_local((world - tp) @ Rp)
                owner[mask & ~hit] = k
                hit |= mask
            return hit, owner

        def pose(s):
            return p0 + s * dp, _quat_matrix(_qmul(_rvq(s * rot), q0))

        start_hit, _ = hits_at(*pose(0.0))
        h1, _ = hits_at(*pose(1.0))
        if not _np.any(h1 & ~start_hit):
            return None
        lo, hi = 0.0, 1.0
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            h, _ = hits_at(*pose(mid))
            if _np.any(h & ~start_hit):
                hi = mid
            else:
                lo = mid
        h_hi, owner_hi = hits_at(*pose(hi))

        def free_at(p, R):
            h, _ = hits_at(p, R)
            return not _np.any(h & ~start_hit)

        return {"lo": lo, "hi": hi, "blocked": h_hi & ~start_hit, "owner": owner_hi, "cands": cands, "pose": pose,
                "free_at": free_at}

    def _candidates(self, p0, q0, dp, rot, local, exclude):
        """Scene pieces whose world AABB meets the swept AABB of the sample points (the held group excluded: the
        pads clamp it and its own points move with the hand), and that AABB's lowest z (3 mm margin included)."""
        R0m = _quat_matrix(q0)
        R1m = _quat_matrix(_qmul(_rvq(rot), q0))
        w0 = local @ R0m.T + p0
        w1 = local @ R1m.T + p0 + dp
        lo = _np.minimum(w0.min(axis=0), w1.min(axis=0)) - 0.003
        hi = _np.maximum(w0.max(axis=0), w1.max(axis=0)) + 0.003
        out = []
        for piece, Rp, tp in self._scene_pieces(exclude_groups=exclude):
            corners = piece.corners_local() @ Rp.T + tp
            if _np.all(corners.max(axis=0) >= lo) and _np.all(corners.min(axis=0) <= hi):
                out.append((piece, Rp, tp))
        return out, float(lo[2])                  # with the 3 mm margin (a rotating step's arc dips below its ends)

    def _deflect_finger(self, finger, mm):
        i = finger - 1
        room = max(0.0, FINGER_DEFLECT_CAP_MM - self._deflect[i])
        mm = min(mm, room)
        if mm <= 0:
            return
        self._deflect[i] += mm
        if finger == 1:
            self.q1 = max(0.0, self.q1 - mm / 1000.0)
        else:
            self.q2 = min(0.0, self.q2 + mm / 1000.0)

    def _relax_deflection(self, keep):
        for i in (0, 1):
            if i + 1 in keep or self._deflect[i] <= 0:
                continue
            back = self._deflect[i] * 0.7
            self._deflect[i] -= back
            if i == 0:
                self.q1 = min(FINGER_Q_MAX_M, self.q1 + back / 1000.0)
            else:
                self.q2 = max(-FINGER_Q_MAX_M, self.q2 - back / 1000.0)

    def _held_press(self, cmd):
        """The held object rests on a support while the hand is still commanded down: the hand creeps (CP lower: ratio
        0.03-0.045 after the stall), the object stays and slides up in the pinch, the gap rises. The creep is itself
        contact-checked (the robot's own points against the scene and the support, the held group excluded: the pads
        slide along it), and the hold ends once the pad window no longer meets the held object between the pad faces
        (the pads slid off it, review BF-R13): the fingers then close past it, as after ``hold_lost``."""
        st = self._held
        if st is None or st["kind"] != "free":
            return
        g = self._group[st["group"]]
        creep = _np.array([0.0, 0.0, HELD_PUSH_RATIO * float(cmd[2])])
        local, kinds = self._assembly_points()
        robot = kinds < 30
        hit = self._first_contact(self.grip, self.quat, creep, _np.zeros(3), local[robot], (st["group"],))
        self.grip = self.grip + (1.0 if hit is None else hit["lo"]) * creep
        R = self.R
        st["t_rel"] = R.T @ (g["t"] - self.grip)
        st["R_rel"] = R.T @ g["R"]
        if hit is not None:
            self.contact_log.append({"step": self.steps, "kind": "creep_blocked", "group": st["group"],
                                     "fraction": hit["lo"], "grip_z": float(self.grip[2])})
        if not self._pads_meet(st["group"]):
            group = st["group"]
            self._release(reason="pads_off_part")
            self._lost_group = group
            self._grip_cmd, self._phase = 1.0, 30
            return
        self._open_pinch(HELD_GAP_RISE_MM)
        st["pressed"] = st.get("pressed", 0) + 1

    def _pads_meet(self, group):
        """Truth: some pad-window line meets ``group`` between the two pad faces."""
        f1, f2 = self.q1 - PAD_FACE_INSET_M, self.q2 + PAD_FACE_INSET_M
        return any(piece.body.group == group and s_in < f1 and s_out > f2 for s_in, s_out, piece in self._pad_intervals())

    def _open_pinch(self, mm):
        st = self._held
        if st is None:
            return
        st["gap_clamp_mm"] += mm
        self.q1 += mm / 2000.0
        self.q2 -= mm / 2000.0

    # --------------------------------------------------------------------------------------------- free objects
    def _follow_held(self):
        st = self._held
        g = self._group[st["group"]]
        R = self.R
        g["R"] = R @ st["R_rel"]
        g["t"] = self.grip + R @ st["t_rel"]
        self._touch_group(st["group"])
        lifted = self.grip[2] - st["z_attach"] > 0.005
        st["lifted"] = bool(st.get("lifted") or lifted)
        if lifted and self.held_tilt_deg and not st.get("tilt_applied"):
            st["tilt_applied"] = True
            self._tilt_held(_math.radians(self.held_tilt_deg))

    def _tilt_held(self, angle):
        st = self._held
        Rt = _rot_axis((0.0, 1.0, 0.0), angle)            # about the jaw R1 (hand frame) through the grip
        st["R_rel"] = Rt @ st["R_rel"]
        st["t_rel"] = Rt @ st["t_rel"]
        self._follow_held()

    def _base_point(self, group):
        body = self.scene.groups[group]["bodies"][0]
        return self._world_point(group, body.centre)

    def _footprint_edge(self, group, direction_xy):
        """Distance (m) from the group's base point to the edge of its base footprint along ``direction_xy``: the
        first body's lowest piece (a cylinder's radius; a box's rectangle, yaw and group rotation included)."""
        body = self.scene.groups[group]["bodies"][0]
        piece = min(body.pieces, key=lambda p: p.t0[2] + p.z0)
        if piece.kind == "cyl":
            return piece.r
        R = self._group[group]["R"] @ piece.R0
        u = R.T @ _np.r_[_unit(direction_xy), 0.0]
        ux, uy = abs(float(u[0])), abs(float(u[1]))
        return min(piece.hx / ux if ux > 1e-12 else _np.inf, piece.hy / uy if uy > 1e-12 else _np.inf)

    def _palm_tip(self, group, contact_xy, push_m):
        """P13 negative: the palm presses ``push_m`` down at ``contact_xy`` on a free group. Beyond the base footprint
        (an overhang, lever L > 0 past the edge) the group tips about that edge by atan(push / L), up to a total tilt
        of TIP_MAX_DEG; returns {"lever_m", "angle_deg", "tilt_deg"}, or None when the press is inside the footprint
        or the tilt is used up (then the object blocks like a fixed body)."""
        base = self._base_point(group)
        off = _np.asarray(contact_xy, float) - base[:2]
        d = float(_np.linalg.norm(off))
        if d < 1e-9:
            return None
        lever = d - self._footprint_edge(group, off)
        room = _math.radians(TIP_MAX_DEG) - _math.radians(_tilt_deg(self._group[group]["R"]))
        if lever <= 1e-4 or room <= 1e-6:
            return None
        angle = min(room, _math.atan2(max(0.0, float(push_m)), lever))
        if angle <= 0:
            return None
        self._tip_group(group, off, angle)
        return {"lever_m": lever, "angle_deg": _math.degrees(angle), "tilt_deg": _tilt_deg(self._group[group]["R"])}

    def _tip_group(self, group, direction_xy, angle, record=True):
        """Rotate a free group about the base edge on ``direction_xy`` so that side goes down by ``angle``."""
        base = self._base_point(group)
        u = _np.r_[_unit(direction_xy), 0.0]
        pivot = base + u * self._footprint_edge(group, direction_xy)
        axis = _np.cross([0.0, 0.0, 1.0], u)
        Rt = _rot_axis(axis, angle)
        st = self._group[group]
        st["R"] = Rt @ st["R"]
        st["t"] = Rt @ (st["t"] - pivot) + pivot
        if record:
            self._touch_group(group)

    def _settle_group(self, group):
        """Released free object: upright again if tilted < SETTLE_UPRIGHT_MAX_DEG, then it drops (or lifts out of any
        penetration) onto the highest surface under its bottom points."""
        st = self._group[group]
        base = self._base_point(group)
        if _tilt_deg(st["R"]) < SETTLE_UPRIGHT_MAX_DEG:
            R_new = _rot_axis((0, 0, 1), _yaw_of(st["R"]))
            body = self.scene.groups[group]["bodies"][0]
            st["R"] = R_new
            st["t"] = base - R_new @ _np.asarray(body.centre, float)
        pts = []
        for body in self.scene.groups[group]["bodies"]:
            for piece in body.pieces:
                local = piece.sample_points_local(0.006)
                pts.append(local @ piece.R0.T + piece.t0)
        world = _np.vstack(pts) @ st["R"].T + st["t"]
        support = self.heights(world[:, :2], exclude_groups=(group,))
        dz = float(_np.max(support - world[:, 2]))
        st["t"] = st["t"] + _np.array([0.0, 0.0, dz])
        self._touch_group(group)

    def _release(self, reason="opened"):
        st = self._held
        if st is None:
            return
        self._held = None
        if st["kind"] == "free":
            self._settle_group(st["group"])
        elif st["kind"] == "rotary":
            self.scene.groups[st["group"]]["hinge"].qd = 0.0
        self.contact_log.append({"step": self.steps, "kind": "release", "group": st["group"], "reason": reason})

    # -------------------------------------------------------------------------------------------------- gripper
    def _gripper_step(self, command):
        sign = 1 if command > 0 else -1 if command < 0 else 0
        if sign != 0:
            same = sign == (1 if self._grip_cmd > 0 else -1 if self._grip_cmd < 0 else 0)
            self._phase = self._phase + 1 if same else 0
            if not same and sign > 0:
                offsets = [i for i in self._injections if i["kind"] == "close_offset_mm" and i.get("left", 0) > 0]
                self._close_offset_mm = float(offsets[0]["mm"]) if offsets else 0.0
                if offsets:
                    offsets[0]["left"] = 0
            self._grip_cmd = command
        if sign < 0:
            self._lost_group = None
        if self._held is not None and sign >= 0:
            self._clamp_age += 1
            st = self._held
            if st["kind"] != "rotary":
                creep = CLAMP_CREEP_MM * _math.exp(-self._clamp_age / CLAMP_CREEP_TAU)
                st["gap_clamp_mm"] -= creep
                self.q1 -= creep / 2000.0
                self.q2 += creep / 2000.0
            return
        if sign == 0 or (sign > 0 and self.gripper_stuck):
            return
        if sign > 0:
            self._close_step()
            return
        limit = OPEN_TAIL_LIMIT_MM if self.open_limit_mm is None else self.open_limit_mm
        for finger in (1, 2):
            if finger in getattr(self, "_pressed_fingers", ()):
                continue                                      # a finger pressed on a surface does not creep open
            defl = self._deflect[finger - 1] / 1000.0
            q = (self.q1 if finger == 1 else -self.q2) + defl          # actuator position
            nxt = min(_curve_step(OPEN_CURVE_MM, 2000.0 * q, self._phase, closing=False), limit) / 2000.0
            nxt = max(nxt, q) if 2000.0 * q < limit else min(q, limit / 2000.0)
            if finger == 1:
                self.q1 = nxt - defl
            else:
                self.q2 = -(nxt - defl)
        # a hold ends only once the pads have opened past the clamped cross-section (review BF-R3): a gripper that
        # cannot open (``open_limit_mm`` at the clamp width) keeps pinching, so the object stays attached and moves with
        # the hand (the release_incomplete world)
        if self._held is not None and self.gap_mm > self._held_contact_gap_mm() + RELEASE_CLEAR_MM:
            self._release("opened")

    def _held_contact_gap_mm(self):
        """The finger gap at which the pads touch the held part: the clamped width (+ the wedge excess when twisted)."""
        st = self._held
        gap = st["gap_clamp_mm"]
        if st["kind"] == "rotary":
            gap += self.scene.groups[st["group"]]["hinge"].gap_excess_mm(st["phi"])
        return gap

    def _pad_intervals(self):
        """Material intervals along the jaw line in the pad window: list of (s_in, s_out, piece) in metres along R1
        about the grip (the [ROBOT] pad span along R2 and the 16 mm pad length along R0)."""
        R = self.R
        out = []
        offsets = [(a / 1000.0, c / 1000.0) for a in (-7.0, -3.5, 0.0, 3.5, 7.0) for c in (-11.0, -7.0, -3.0, 1.0, 4.0)]
        origins = _np.asarray([self.grip + R[:, 0] * a + R[:, 2] * c for a, c in offsets])
        d = _np.tile(R[:, 1], (len(origins), 1))
        skip = () if self._lost_group is None else (self._lost_group,)
        for piece, Rp, tp in self._scene_pieces(exclude_groups=skip):
            corners = piece.corners_local() @ Rp.T + tp
            if _np.any(_np.linalg.norm(corners.mean(axis=0)[:2] - self.grip[:2]) > 0.5):
                continue
            o = (origins - tp) @ Rp
            dl = d @ Rp
            t_in, t_out, _ = piece.hits(o, dl)
            ok = t_in <= t_out
            for i in _np.flatnonzero(ok):
                out.append((float(t_in[i]), float(t_out[i]), piece))
        return out

    def _close_step(self):
        inset = PAD_FACE_INSET_M
        self._deflect = [0.0, 0.0]                                       # closing fingers leave the landing contact
        nxt = []
        for finger in (1, 2):
            q = self.q1 if finger == 1 else -self.q2
            g_now = 2000.0 * q - self._close_offset_mm
            g_next = _curve_step(CLOSE_CURVE_MM, g_now, self._phase, closing=True) + self._close_offset_mm
            nxt.append(min(q, max(g_next / 2000.0, PAD_CONTACT_GAP_MM / 2000.0)))
        f1, f2 = self.q1 - inset, self.q2 + inset
        f1n, f2n = nxt[0] - inset, -nxt[1] + inset
        stop1 = stop2 = None
        for s_in, s_out, piece in self._pad_intervals():
            if s_in < f1 + 1e-9 and s_out > f1n - 1e-9:
                cand = min(s_out, f1)
                if stop1 is None or cand > stop1[0]:
                    stop1 = (cand, piece)
            if s_out > f2 - 1e-9 and s_in < f2n + 1e-9:
                cand = max(s_in, f2)
                if stop2 is None or cand < stop2[0]:
                    stop2 = (cand, piece)
        g1 = stop1[1].body.group if stop1 else None
        g2 = stop2[1].body.group if stop2 else None
        role = lambda g: self.scene.groups[g]["role"]
        if stop1 and stop2 and g1 == g2:
            self.q1, self.q2 = stop1[0] + inset, stop2[0] - inset
            self._clamp(g1)
            return
        if stop1 and not stop2 and role(g1) == "free":
            self._push_group(g1, -(f1 - f1n - (f1 - stop1[0])) * self.R[:, 1])
            self.q1, self.q2 = nxt[0], -nxt[1]
            return
        if stop2 and not stop1 and role(g2) == "free":
            self._push_group(g2, (f2n - f2 - (stop2[0] - f2)) * self.R[:, 1])
            self.q1, self.q2 = nxt[0], -nxt[1]
            return
        self.q1 = stop1[0] + inset if stop1 else nxt[0]
        self.q2 = stop2[0] - inset if stop2 else -nxt[1]

    def _push_group(self, group, vector):
        st = self._group[group]
        st["t"] = st["t"] + _np.asarray(vector, float)
        self._touch_group(group)

    def _clamp(self, group):
        role = self.scene.groups[group]["role"]
        gap = self.gap_mm
        kind = {"free": "free", "rotary": "rotary"}.get(role, "fixed")
        st = {"group": group, "kind": kind, "since": self.steps, "gap_clamp_mm": gap, "z_attach": float(self.grip[2]),
              "clamp_centre_m": (self.q1 + self.q2) / 2.0}
        self._clamp_age = 0
        if kind == "free":
            if _np.any(self.close_drag_m != 0):
                self._push_group(group, _np.r_[self.close_drag_m[:2], 0.0])
            g = self._group[group]
            R = self.R
            st["R_rel"] = R.T @ g["R"]
            st["t_rel"] = R.T @ (g["t"] - self.grip)
        elif kind == "rotary":
            h = self.scene.groups[group]["hinge"]
            st.update(q_close=h.q, p_close=self.grip.copy(), phi=0.0, tug=_np.zeros(2), slipped=False)
        self._held = st
        self.contact_log.append({"step": self.steps, "kind": "clamp", "group": group, "gap_mm": gap, "role": kind})

    # -------------------------------------------------------------------------------------------------- effects
    def _update_effects(self):
        for body in self.scene.bodies:
            e = body.effect
            if e is None:
                continue
            h = self.scene.groups[e.driver]["hinge"]
            on_now = h.q >= h.q_on
            rec = self._effect_on_since.get(body.name)
            if on_now and rec is None:
                rec = {"q_cross_step": self.steps, "on_step": self.steps + int(e.lag_steps), "q_at_cross": float(h.q)}
                self._effect_on_since[body.name] = rec
            elif not on_now and rec is not None:
                self._effect_on_since.pop(body.name)
                rec = None
            self._effect_state[body.name] = bool(rec is not None and not e.never and self.steps >= rec["on_step"])

    # ----------------------------------------------------------------------------------------------- goal / truth
    def goal_atoms(self):
        return [list(atom) for atom in self.scene.goal]

    def _bottom_points(self, group, spacing=0.004):
        pts = []
        for body in self.scene.groups[group]["bodies"]:
            for piece in body.pieces:
                local = piece.sample_points_local(spacing)
                pts.append(local[local[:, 2] <= piece.z0 + 1e-9] @ piece.R0.T + piece.t0)
        g = self._group[group]
        return _np.vstack(pts) @ g["R"].T + g["t"]

    def _on(self, group, dest):
        """LIBERO-like On(group, dest) (raw: centre over the destination region and touching it) plus strict checks."""
        body = self.scene.body(dest)
        info = self.scene.info.get("regions", {}).get(dest, {})
        pose = self.body_pose(dest)
        half = info.get("half_xy")
        if half is None:
            p = body.pieces[0]
            half = (p.hx, p.hy) if p.kind == "box" else (p.r, p.r)
        centre = _np.asarray(info.get("centre_xy", pose["xyz"][:2]), float)
        bottom = self._bottom_points(group)
        top = pose["top_z"]
        dest_h = self.heights(bottom[:, :2], exclude_groups=(group,))
        other_h = self.heights(bottom[:, :2], exclude_groups=(group, body.group))
        on_dest = _np.abs(dest_h - top) < 1e-6
        touching = bottom[:, 2] <= dest_h + CONTACT_TOL_M
        base = self._base_point(group)
        rel = base[:2] - centre
        Rd = pose["R"]
        local = _np.array([rel @ Rd[:2, 0], rel @ Rd[:2, 1]])
        in_region = bool(abs(local[0]) <= half[0] and abs(local[1]) <= half[1])
        contact = bool(_np.any(touching & on_dest))
        supported_other = bool(_np.any(touching & ~on_dest & (bottom[:, 2] <= other_h + CONTACT_TOL_M)))
        held = self._held is not None and self._held["group"] == group
        moved = self.steps - self._group[group]["moved_step"] <= 1
        tilt0 = max(b.tilt_deg for b in self.scene.groups[group]["bodies"])
        checks = {"not_held": not held, "at_rest": not moved, "supported_by_target": contact,
                  "supported_only_by_target": contact and not supported_other, "com_over_region": in_region,
                  "orientation_kept": abs(_tilt_deg(self._group[group]["R"]) - tilt0) <= 15.0}
        return {"raw": bool(in_region and contact), "checks": checks,
                "numbers": {"lowest_z": float(bottom[:, 2].min()), "dest_top_z": float(top),
                            "centre_offset_m": [float(v) for v in local]}}

    def _turnon(self, group):
        h = self.scene.groups[group]["hinge"]
        held = self._held is not None and self._held["group"] == group
        checks = {"libero_threshold": bool(h.q >= h.q_on), "robot_not_touching_part": not held,
                  "joint_at_rest": abs(h.qd) < 1e-4}
        return {"raw": bool(h.q >= h.q_on), "checks": checks, "numbers": {"q": float(h.q), "q_on": float(h.q_on)}}

    def explain(self, atom):
        if atom[0] == "turnon":
            r, kind = self._turnon(atom[1]), "articulation"
        elif atom[0] == "on":
            r, kind = self._on(atom[1], atom[2]), "on_fixture_region"
        else:
            raise ValueError(f"unsupported fake atom {atom!r}")
        return {"atom": list(atom), "kind": kind, "raw": r["raw"], "strict": bool(all(r["checks"].values())),
                "why_not": [k for k, v in r["checks"].items() if not v], "checks": dict(r["checks"]), "libero": {},
                "numbers": r["numbers"]}

    def goal_satisfied(self):
        """Truth: the world's goal predicate (every atom raw-true), as LIBERO's check_success."""
        return bool(self.scene.goal) and all(self.explain(a)["raw"] for a in self.goal_atoms())

    def truth_record(self, state="terminal"):
        """Truth record shaped like M0's ``truth_result`` / evaluate_goal (diagnosis only)."""
        atoms = [self.explain(a) for a in self.goal_atoms()]
        return {"truth_version": FAKES_V2_VERSION, "diagnosis_only": True, "affects_completion": False,
                "raw_conjunction": all(a["raw"] for a in atoms), "check_success": self.goal_satisfied(),
                "strict_conjunction": all(a["strict"] for a in atoms), "atoms": atoms, "state": state}

    def physical_truth(self):
        """Truth summary: per-atom strict verdicts, hinge angles, the held object and the goal predicate."""
        record = self.truth_record()
        joints = {name: float(g["hinge"].q) for name, g in self.scene.groups.items() if g["hinge"] is not None}
        return {"goal_satisfied": record["check_success"], "strict": record["strict_conjunction"],
                "atoms": record["atoms"], "joints": joints, "held": self.held(), "steps": self.steps}

    def clone(self):
        """Independent copy of the dynamic world (textures and caches shared); used by the settle counterfactual."""
        import copy as _copy
        twin = _copy.copy(self)
        twin.scene = _copy_scene(self.scene)
        twin._init = _copy.deepcopy(self._init)
        twin._group = {k: {kk: (vv.copy() if isinstance(vv, _np.ndarray) else vv) for kk, vv in v.items()}
                       for k, v in self._group.items()}
        twin._held = None if self._held is None else _copy.deepcopy(self._held)
        twin._injections = _copy.deepcopy(self._injections)
        twin._effect_on_since = _copy.deepcopy(self._effect_on_since)
        twin._effect_state = dict(self._effect_state)
        twin.contact_log = list(self.contact_log)
        twin.actions = list(self.actions)
        twin.events = list(self.events)
        twin._deflect = list(self._deflect)
        twin._rng = _np.random.default_rng(self.seed + 17 + self.steps)
        twin._render_cache = {}
        twin._obs_cache = None
        twin._static_ext = None
        twin.grip, twin.quat, twin.vel, twin.omega = (self.grip.copy(), self.quat.copy(), self.vel.copy(),
                                                       self.omega.copy())
        return twin


TRANSLATION_PER_UNIT = 0.05      # [ROBOT] R10: 0.05 m per normalized action unit
ROTATION_PER_UNIT = 0.5          # [ROBOT] R10: 0.5 rad per normalized action unit


_BLANK_URL = None


def _blank_url():
    global _BLANK_URL
    if _BLANK_URL is None:
        _BLANK_URL = _png_data_url(_np.full((IMAGE_PX, IMAGE_PX, 3), 128, _np.uint8))
    return _BLANK_URL


def _png_data_url(rgb):
    """Lossless RGB PNG data URL (cv2 deflate level 1: ~2x faster than PIL for 512 px frames; PIL reads it as RGB)."""
    import base64
    ok, buf = _cv2.imencode(".png", _np.ascontiguousarray(_np.asarray(rgb, _np.uint8)[..., ::-1]),
                            [_cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _points_in_convex(points, polygon):
    """Points (n,2) inside a convex polygon (k,2), either orientation."""
    P = _np.asarray(points, float)
    poly = _np.asarray(polygon, float)
    a, b = poly, _np.roll(poly, -1, axis=0)
    cross = ((b[:, 0] - a[:, 0])[None, :] * (P[:, 1:2] - a[None, :, 1])
             - (b[:, 1] - a[:, 1])[None, :] * (P[:, 0:1] - a[None, :, 0]))
    return _np.all(cross >= 0, axis=1) | _np.all(cross <= 0, axis=1)


# ----------------------------------------------------------------------------------------------- oracle answers
# The answer rules mirror BH's ORACLE stub (scripts/agent_speedup_v2_oracle.py; review BF-R8), so a T1 chain on the
# fake never passes on answers the real stub would refuse: pixel centres at integers, image [-0.5, W - 0.5), crop
# [origin - 0.5, origin + crop - 0.5), the 8 x 8 coarse cells and fine points (v1 layout, or the v2 client's
# ``fine_point_pixel_at`` once it exists), the plan's border rule (nearest visible part pixel within
# ORACLE_PLAN_BORDER_REACH_PX, any distance for a ``whole`` part, and only when that cell shows at least
# ORACLE_PLAN_MIN_VISIBLE_FRACTION of the part), and the perturbation of pointing answers in the SHOWN (turned) frame,
# never of plan cells. tests/test_agent_speedup_v2_fakes.py checks these numbers and helpers against BH's.
ORACLE_GRID = 8                          # = BH GRID
ORACLE_FINE_CROP_PX = 128                # = BH FINE_CROP_PX (the client's V6 crop)
ORACLE_PLAN_MIN_VISIBLE_FRACTION = 0.01  # = BH PLAN_MIN_VISIBLE_FRACTION
ORACLE_PLAN_BORDER_REACH_PX = 64         # = BH PLAN_BORDER_REACH_PX


def oracle_in_image(uv, width=IMAGE_PX):
    return uv is not None and -0.5 <= float(uv[0]) < width - 0.5 and -0.5 <= float(uv[1]) < width - 0.5


def oracle_coarse_cell_of(uv, width=IMAGE_PX):
    """The 8 x 8 cell (1..64) containing ``uv`` (0 outside the image)."""
    if not oracle_in_image(uv, width):
        return 0
    cell = width / ORACLE_GRID
    col = min(max(int(_math.floor(float(uv[0]) / cell)), 0), ORACLE_GRID - 1)
    row = min(max(int(_math.floor(float(uv[1]) / cell)), 0), ORACLE_GRID - 1)
    return row * ORACLE_GRID + col + 1


def oracle_crop_origin(center_uv, crop_px=ORACLE_FINE_CROP_PX, width=IMAGE_PX):
    limit = width - crop_px
    return tuple(int(min(max(round(float(c) - crop_px / 2.0), 0), limit)) for c in center_uv)


def oracle_in_crop(uv, origin, crop_px):
    return all(origin[i] - 0.5 <= float(uv[i]) < origin[i] + crop_px - 0.5 for i in (0, 1))


def _v1_fine_point_pixel_at(origin, point_id, *, crop_px=ORACLE_FINE_CROP_PX):
    row, col = divmod(int(point_id) - 1, ORACLE_GRID)
    spacing = crop_px / ORACLE_GRID
    return (float(origin[0] + (col + 0.5) * spacing), float(origin[1] + (row + 0.5) * spacing))


def oracle_fine_point_mapping():
    """The client's ``fine_point_pixel_at`` when the v2 client exists (Wave 1), else the v1 layout (as BH)."""
    try:
        from gpt_field.agent_speedup_v2_client import fine_point_pixel_at
    except ImportError:
        return _v1_fine_point_pixel_at, "v1_layout_restated"
    return fine_point_pixel_at, "agent_speedup_v2_client.fine_point_pixel_at"


def oracle_nearest_fine_point(uv, origin, crop_px, mapping):
    return min(range(1, ORACLE_GRID * ORACLE_GRID + 1),
               key=lambda pid: _math.dist(mapping(origin, pid, crop_px=crop_px), (float(uv[0]), float(uv[1]))))


class FakeOracleAnswers:
    """ScriptedAnswers (contract §C12 duck type ``answer(kind, meta) -> dict``) from FakeTwoViewSim TRUTH, with BH's
    ORACLE answer rules (see the block comment above).

    ``plan``: a wire-JSON plan template (§3.3) whose parts' ``w``/``x`` cells are filled at answer time: the cell of
    the projected truth point in the k-turned wrist grid / the upright external grid; a truth point outside the image
    takes the cell of the part's nearest VISIBLE pixel (truth mask) when that pixel is within
    ORACLE_PLAN_BORDER_REACH_PX (any distance for a ``whole`` part, whose mask is its whole group) and the cell shows
    >= ORACLE_PLAN_MIN_VISIBLE_FRACTION of the part, else 0; default ``scene.info["oracle_plan"]``. ``points``:
    {pid: (body_name, initial-world xyz)} (moves with the body's group) or {pid: world xyz}; default
    ``scene.info["oracle_points"]``; ``whole``: pids that stand for a whole object (default
    ``scene.info["oracle_whole"]``). point_fine answers the crop point nearest the shown truth and 0 when it lies
    OUTSIDE the crop (as a correct model would, [C-37]); point_coarse answers the cell (0 outside the image); repair ->
    retry; review -> hold_still. ``perturb_px`` shifts every POINTING answer's shown truth by that many px (a fixed
    seeded direction per pid, in the shown frame; BH draws a fresh direction per answer); ``wrong_item`` =
    {pid: other_pid} answers another part's point, or a pid string (BH's form: the next part, cyclic). Meta keys read:
    kind, camera, pid, center_uv (upright), origin (shown frame), crop_px, quarter_turns, image_size. Every answer is
    logged with its rule."""

    FINE_GRID = ORACLE_GRID
    COARSE_GRID = ORACLE_GRID

    def __init__(self, world, *, plan=None, points=None, perturb_px=0.0, wrong_item=None, seed=0, repairs=None,
                 whole=None):
        self.world = world
        self.plan = plan if plan is not None else world.scene.info.get("oracle_plan")
        self.points = dict(points if points is not None else world.scene.info.get("oracle_points", {}))
        self.whole = set(whole if whole is not None else world.scene.info.get("oracle_whole", ()))
        self.perturb_px = float(perturb_px)
        if isinstance(wrong_item, str):
            order = list(self.points)
            if wrong_item not in order:
                raise ValueError(f"wrong_item {wrong_item!r} is not an oracle point")
            wrong_item = {wrong_item: order[(order.index(wrong_item) + 1) % len(order)]}
        self.wrong_item = dict(wrong_item or {})
        self.repairs = list(repairs or [])
        rng = _np.random.default_rng(seed)
        self._dirs = {pid: _unit(rng.normal(size=2)) for pid in "ABCDEFGHN"}
        self.mapping, self.mapping_name = oracle_fine_point_mapping()
        self.log = []

    def _source(self, pid):
        return self.wrong_item.get(pid, pid)

    def truth_xyz(self, pid):
        spec = self.points[self._source(pid)]
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
            body = self.world.scene.body(spec[0])
            return self.world._world_point(body.group, spec[1])
        return _np.asarray(spec, float)

    def truth_uv(self, pid, camera):
        """Upright (unperturbed) truth pixel of a part in ``camera``, or None when behind the camera."""
        uv, z = self.world.project(camera, self.truth_xyz(pid))
        if uv is None or z <= 0:
            return None
        return _np.asarray(uv, float)

    def shown_uv(self, pid, camera, quarter_turns=0, size=IMAGE_PX):
        """The pointing answer's truth: the turned truth pixel plus the perturbation (shown frame)."""
        uv = self.truth_uv(pid, camera)
        if uv is None:
            return None
        if quarter_turns % 4:
            uv = _np.asarray(_rotate_pixel(uv, quarter_turns % 4, size), float)
        return uv + self.perturb_px * self._dirs.get(pid, _np.zeros(2))

    def _mask(self, pid, camera):
        spec = self.points.get(self._source(pid))
        if not (isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str)):
            return None
        name = self.world.scene.body(spec[0]).group if self._source(pid) in self.whole else spec[0]
        mask = self.world.truth_mask(camera, name)
        return mask if mask.any() else None

    def plan_cell(self, pid, camera, quarter_turns=0, size=IMAGE_PX):
        """(cell, rule) of a part in ``camera`` (BH ``_plan_cell``)."""
        k = quarter_turns % 4
        uv = self.truth_uv(pid, camera)
        if uv is None:
            return 0, "behind_camera"
        if oracle_in_image(uv, size):
            return oracle_coarse_cell_of(_rotate_pixel(uv, k, size) if k else uv, size), "truth_in_image"
        mask = self._mask(pid, camera)
        if mask is None:
            return 0, "not_visible"
        vs, us = _np.nonzero(mask)
        distance = _np.hypot(us - float(uv[0]), vs - float(uv[1]))
        i = int(_np.argmin(distance))
        reach = _math.inf if self._source(pid) in self.whole else ORACLE_PLAN_BORDER_REACH_PX
        if float(distance[i]) > reach:
            return 0, "truth_outside_image"
        nearest = (float(us[i]), float(vs[i]))
        candidate = oracle_coarse_cell_of(_rotate_pixel(nearest, k, size) if k else nearest, size)
        row, col = divmod(candidate - 1, ORACLE_GRID)
        cell = size // ORACLE_GRID
        shown = _np.rot90(mask, k)[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
        if shown.sum() >= ORACLE_PLAN_MIN_VISIBLE_FRACTION * cell * cell:
            return candidate, "border_cut_visible"
        return 0, "not_visible"

    def cells(self, pid, quarter_turns):
        return (self.plan_cell(pid, "wrist", quarter_turns)[0], self.plan_cell(pid, "external", 0)[0])

    def plan_answer(self, quarter_turns, rules=None):
        if self.plan is None:
            raise ValueError("FakeOracleAnswers needs a plan template for plan requests")
        plan = _json.loads(_json.dumps(self.plan))
        for part in plan["parts"]:
            if part["id"] in self.points or part["id"] in self.wrong_item:
                (part["w"], w_rule), (part["x"], x_rule) = (self.plan_cell(part["id"], "wrist", quarter_turns),
                                                            self.plan_cell(part["id"], "external", 0))
                if rules is not None:
                    rules[part["id"]] = {"w_rule": w_rule, "x_rule": x_rule}
        return plan

    def fine_answer(self, pid, camera, *, quarter_turns=0, origin=None, center_uv=None, crop_px=ORACLE_FINE_CROP_PX,
                    size=IMAGE_PX):
        """(point_id, shown uv): the crop point nearest the shown truth, 0 when it lies outside the image or crop."""
        k = quarter_turns % 4
        shown = self.shown_uv(pid, camera, k, size)
        if origin is None:
            c = (size / 2.0, size / 2.0) if center_uv is None else _np.asarray(center_uv, float)
            origin = oracle_crop_origin(_rotate_pixel(c, k, size) if k else c, crop_px, size)
        origin = (int(origin[0]), int(origin[1]))
        if shown is None or not (oracle_in_image(shown, size) and oracle_in_crop(shown, origin, crop_px)):
            return 0, shown
        return oracle_nearest_fine_point(shown, origin, crop_px, self.mapping), shown

    def answer(self, kind, meta):
        meta = dict(meta or {})
        kind = meta.get("kind", kind)
        size = int(meta.get("image_size", IMAGE_PX) or IMAGE_PX)
        k = int(meta.get("quarter_turns", 0) or 0) % 4
        record = {"kind": kind, "pid": meta.get("pid"), "camera": meta.get("camera"), "step": self.world.steps}
        if kind in ("plan", "replan"):
            rules = {}
            out = self.plan_answer(k, rules)
            record["cells"] = {p["id"]: [p["w"], p["x"]] for p in out["parts"]}
            record["rules"] = rules
        elif kind == "point_fine":
            pid, camera = meta["pid"], meta.get("camera", "wrist")
            answer_id, uv = self.fine_answer(pid, camera, quarter_turns=k, origin=meta.get("origin"),
                                             center_uv=meta.get("center_uv"),
                                             crop_px=int(meta.get("crop_px", ORACLE_FINE_CROP_PX) or ORACLE_FINE_CROP_PX),
                                             size=size)
            out = {"point_id": answer_id}
            record.update(answer=answer_id, rule="nearest_fine_point" if answer_id else "truth_outside_crop",
                          truth_shown_uv=None if uv is None else [float(v) for v in uv])
        elif kind == "point_coarse":
            pid, camera = meta["pid"], meta.get("camera", "wrist")
            uv = self.shown_uv(pid, camera, k, size)
            out = {"cell_id": oracle_coarse_cell_of(uv, size) if uv is not None else 0}
            record.update(answer=out["cell_id"], rule="truth_cell" if out["cell_id"] else "truth_outside_image",
                          truth_shown_uv=None if uv is None else [float(v) for v in uv])
        elif kind == "repair":
            out = dict(self.repairs.pop(0)) if self.repairs else {"why": "oracle", "action": "retry", "part": None,
                                                                   "step": None}
        elif kind == "review":
            out = {"why": "oracle", "action": "hold_still", "part": None, "step": None}
        else:
            raise ValueError(f"unknown request kind {kind!r}")
        record["answer_json"] = out
        self.log.append(record)
        return out

    def report(self):
        return {"answers": list(self.log), "perturb_px": self.perturb_px, "wrong_item": dict(self.wrong_item),
                "whole": sorted(self.whole), "fine_mapping": self.mapping_name}


# ------------------------------------------------------------------------------------------- runner-facing world sim
class FakeV2WorldSim(FakeV2Sim):
    """FakeV2Sim's runner status contract driven by a FakeTwoViewSim world: success = the world's goal predicate (every
    atom raw-true, like LIBERO's check_success; it may fire before the lowering stall), ``truth()`` / ``settle_truth()``
    return truth_result-shaped records from the world's own geometry (the settle counterfactual runs on a clone:
    open 10, lift 8, settle 20 steps as M0's RELEASE_SETTLE). No ``success_at`` script.
    ``FakeV2WorldSim(FakeTwoViewSim(task2_like_scene()))``; a runner ``sim_factory`` passes the config second."""

    RELEASE_SETTLE = {"open_steps": 10, "lift_steps": 8, "settle_steps": 20, "lift_action_z": 0.5}

    def __init__(self, world, config=None, *, prompt=None, task_id=2, tasks=None, chained=False, collision_at=None,
                 truth_error=None, interrupt_at=None, stale_reply=None):
        super().__init__(config, prompt=prompt or world.prompt, task_id=task_id, tasks=tasks, wrist=world,
                         chained=chained, collision_at=collision_at, truth_error=truth_error,
                         interrupt_at=interrupt_at, stale_reply=stale_reply)
        self.world = world
        self.success_seen = False

    def reset(self, task_id, episode_index):
        self.ops.append("reset")
        self.world.reset()
        self.success_seen = False
        self.status = v2_status(task_id=task_id, v1=v1_record() if self.chained else None)
        return self.world.observation()

    def step(self, action):
        if self.closed:
            raise RuntimeError("closed")
        n = len(self.actions) + 1
        if self.interrupt_at is not None and n == self.interrupt_at:
            self._in_flight = True
            self._queue.append({"observation": {"observation/state": [0.0] * 8}, "status": dict(self.status)})
            raise KeyboardInterrupt()
        self.actions.append(list(action))
        obs = self.world.step(action)
        success = self.success_seen or self.world.goal_satisfied()
        self.success_seen = success
        record = generic_record()
        if self.collision_at is not None and n >= self.collision_at:
            first = {"class": "robot_scene", "action_index": self.collision_at - 1}
            record = generic_record(True, {"robot_scene": first}, 1)
        self.status = v2_status(n, success, self.task_id, record, v1_record() if self.chained else None)
        return obs, success

    def truth(self):
        self.ops.append("truth")
        if self.truth_error:
            raise self.truth_error
        return self._reply(self.world.truth_record("terminal"))

    def settle_truth(self, protocol=None):
        self.ops.append("settle_truth")
        p = {**self.RELEASE_SETTLE, **(protocol or {})}
        twin = self.world.clone()
        opened = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        actions = ([opened] * p["open_steps"] + [[0.0, 0.0, float(p["lift_action_z"]), 0.0, 0.0, 0.0, -1.0]]
                   * p["lift_steps"] + [opened] * p["settle_steps"])
        for action in actions:
            twin.step(action)
        record = twin.truth_record("after_release_settle")
        record["release_settle"] = {"actions": len(actions), "protocol": p}
        return self._reply(record)

    def answer_key(self):
        self.ops.append("answer_key")
        return self._reply({"diagnosis_only": True, "goal_atoms": self.world.goal_atoms()})


# ------------------------------------------------------------------------------------------------- scene builders
TASK2_KNOB_CENTRE_XY = (-0.2008, 0.2074)       # SURVEY t2_dump.json / spec §7.1 [F]
TASK2_HINGE_XY = (-0.1964, 0.2061)
TASK2_BURNER_XY = (-0.0464, 0.2061)
TASK2_MOKA_XY = (0.030, -0.007)
TASK2_SUPPORT_Z = 0.900
EFFECT_RING_PROBE_M = (0.0455, 0.055)          # a ring whose rendered share of the plate (external view, chroma > 12)
                                               # is ~6.9 %, the M0 effect probe's post-onset red-shift share 0.0688
                                               # (effect_probe_t2_records.json); the default ring is the contract's 20 %
TASK2_ORACLE_PLAN = {
    "scene": "stove off; knob with lever below the stove; moka pot at the top edge; frying pan right",
    "parts": [{"id": "A", "text": "stove knob: lever on top of the knob", "move": "hinged", "w": 0, "x": 0},
              {"id": "B", "text": "stove: ring burner plate", "move": "fixed", "w": 0, "x": 0},
              {"id": "C", "text": "moka pot: black handle loop", "move": "free", "w": 0, "x": 0},
              {"id": "D", "text": "moka pot: whole pot", "move": "free", "w": 0, "x": 0},
              {"id": "E", "text": "frying pan: long handle", "move": "free", "w": 0, "x": 0}],
    "hold_true": [],
    "steps": [{"do": "turn", "obj": "A", "grip": ["A"], "ref": None, "sense": "any", "keep": [],
               "until": [{"o": "effect", "p": "B", "txt": "burner rings turn red"}, {"o": "stop", "p": None, "txt": None}]},
              {"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
               "keep": [{"c": "upright", "p": "D"}, {"c": "clear_of", "p": "E"}], "until": []}]}
TASK2_ORACLE_PLAN["scene"] = TASK2_ORACLE_PLAN["scene"][:80]


def _knob_bodies(centre_xy, hinge, *, lever=True, dark=False, group="knob", lever_length=0.085):
    """A rotary part: a disk (r 47.6 mm, top 18 mm above the support) with an elongated ridge (24.3 mm wide, top
    55 mm above the support) or, for ``lever=False``, a round knob (r 12.5 mm) on top. ``dark``: uniform dark surfaces
    (patch std ~1-3 grey) so features exist only on the occluding edges."""
    x, y = centre_xy
    disk_std, part_std = (1.2, 1.0) if dark else (7.0, 6.0)
    disk = FakeBody("knob_disk", "disk", (x, y, TASK2_SUPPORT_Z), (0.0476, 0.018), colour=(36.0, 36.0, 38.0),
                    texture_std=disk_std, role="rotary", group=group, hinge=hinge, texture_scale_m=0.0012)
    if lever:
        top = FakeBody("knob_lever", "ridge", (x, y, TASK2_SUPPORT_Z + 0.018), (lever_length, 0.0243, 0.037),
                       colour=(28.0, 28.0, 30.0), texture_std=part_std, role="rotary", group=group,
                       texture_scale_m=0.0012)
    else:
        top = FakeBody("knob_round", "cylinder", (x, y, TASK2_SUPPORT_Z + 0.018), (0.0125, 0.037),
                       colour=(28.0, 28.0, 30.0), texture_std=part_std, role="rotary", group=group,
                       texture_scale_m=0.0012)
    return [disk, top]


def _burner_body(centre_xy, driver, *, name="burner_plate", effect=True, effect_ring_m=None):
    """A 19 x 19 cm fixed planar region, top 30.5 mm above the support, dark metal with ring grooves, arm-reflection
    luminance and (``effect``) a chroma-changing ring switched by ``driver``: r 2.5-5.5 cm (~20 % of the region, the
    contract §C15 figure) unless ``effect_ring_m`` = (r_in, r_out); EFFECT_RING_PROBE_M (~6.9 % rendered) matches the
    M0 effect probe's measured red-shift share (review BF-R10, a contract-vs-probe discrepancy reported to the lead)."""
    ring = {} if effect_ring_m is None else {"r_in": float(effect_ring_m[0]), "r_out": float(effect_ring_m[1])}
    eff = FakeEffect(driver=driver, centre_xy=tuple(centre_xy), **ring) if effect else None
    return FakeBody(name, "aabb", (centre_xy[0], centre_xy[1], TASK2_SUPPORT_Z), (0.19, 0.19, 0.0305),
                    colour=(66.0, 66.0, 70.0), texture_std=5.0, pattern="rings", reflect_grey=34.0, effect=eff,
                    texture_scale_m=0.0015)


def _moka_bodies(centre_xy, *, group="moka"):
    """A free object: a 66 mm cylinder (top 133.1 mm above the support), a spout, and a thin handle loop toward -y:
    13 mm thick (x), ~60 mm long, outer segment z 0.967-1.0345 (spec §7.1; t2_dump.json g13/g14)."""
    x, y = centre_xy
    body = FakeBody("moka_body", "cylinder", (x, y, TASK2_SUPPORT_Z), (0.033, 0.1331), colour=(190.0, 190.0, 194.0),
                    texture_std=10.0, role="free", group=group, mass_kg=0.068)
    edge = y - 0.033
    loop = FakeBody("moka_handle", "loop", (x, edge - 0.030, 0.956), parts=(
        ("box", (x, edge - 0.0495, 0.967), (0.013, 0.020, 0.0675), 0.0),
        ("box", (x, edge - 0.01975, 1.0245), (0.013, 0.0395, 0.010), 0.0),
        ("box", (x, edge - 0.01975, 0.956), (0.013, 0.0395, 0.011), 0.0)),
        colour=(24.0, 24.0, 26.0), texture_std=3.0, role="free", group=group)
    spout = FakeBody("moka_spout", "aabb", (x, y + 0.0405, 1.005), (0.012, 0.015, 0.027), colour=(182.0, 182.0, 186.0),
                     texture_std=8.0, role="free", group=group)
    return [body, loop, spout]


def _frypan_bodies():
    pan = FakeBody("frypan_pan", "cylinder", (-0.0742, -0.2723, TASK2_SUPPORT_Z), (0.10, 0.040),
                   colour=(40.0, 40.0, 42.0), texture_std=4.0, role="free", group="frypan")
    handle = FakeBody("frypan_handle", "aabb", (-0.07425, -0.0797, 0.9286), (0.0253, 0.1334, 0.0292),
                      colour=(30.0, 30.0, 32.0), texture_std=3.0, role="free", group="frypan")
    return [pan, handle]


def _local_top(name, xy, z):
    return (name, (float(xy[0]), float(xy[1]), float(z)))


def task2_like_scene(*, texture_seed=0, free_sense=-1, effect=True, effect_ring_m=None):
    """libero_10 task 2 geometry (spec §7.1 [F]): knob (hinge 4.6 mm off the disk centre, range [-0.005, 2.1], q_on 0.5,
    free in contract sense -1), burner plate B, moka pot D with its loop C, frying pan E (keep-out); support 0.900 with a
    low-texture (std 3) wood so only part of the start view passes D15. Goal: TurnOn(knob) AND On(moka, burner)."""
    hinge = FakeHinge(centre_xy=TASK2_HINGE_XY, free_sense=free_sense, q=0.0, q_lo=-0.005, q_hi=2.1, q_on=0.5)
    bodies = (_knob_bodies(TASK2_KNOB_CENTRE_XY, hinge)
              + [_burner_body(TASK2_BURNER_XY, "knob", effect=effect, effect_ring_m=effect_ring_m)]
              + _moka_bodies(TASK2_MOKA_XY) + _frypan_bodies())
    mx, my = TASK2_MOKA_XY
    info = {"prompt": TASK2_PROMPT, "rotary": "knob", "rotary_part": "knob_lever", "effect_region": "burner_plate",
            "free_object": "moka", "thin_part": "moka_handle", "destination": "burner_plate", "keepout": "frypan",
            "start_eef": TASK2_START_STATE8[:3], "start_tilt_deg": 3.14,
            "hover_eef": {"knob": (TASK2_KNOB_CENTRE_XY[0], TASK2_KNOB_CENTRE_XY[1], 1.03),
                          "moka_handle": (mx, my - 0.0795, 1.11)},
            "regions": {"burner_plate": {"centre_xy": TASK2_BURNER_XY, "half_xy": (0.075, 0.075)}},
            "oracle_points": {"A": _local_top("knob_lever", TASK2_KNOB_CENTRE_XY, 0.955),
                              "B": _local_top("burner_plate", TASK2_BURNER_XY, 0.9305),
                              "C": _local_top("moka_handle", (mx, my - 0.0795), 1.0345),
                              "D": _local_top("moka_body", TASK2_MOKA_XY, 1.0331),
                              "E": _local_top("frypan_handle", (-0.07425, -0.0797), 0.9578)},
            "oracle_plan": TASK2_ORACLE_PLAN, "oracle_whole": ["D"],
            "truth": {"lever_width_m": 0.0243, "lever_top_z": 0.955, "disk_top_z": 0.918, "disk_radius_m": 0.0476,
                      "burner_top_z": 0.9305, "loop_width_m": 0.013, "loop_top_z": 1.0345, "moka_top_z": 1.0331,
                      "hinge_xy": TASK2_HINGE_XY, "disk_centre_xy": TASK2_KNOB_CENTRE_XY}}
    return FakeScene(bodies, support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed, support_texture_std=3.0,
                     goal=[("turnon", "knob"), ("on", "moka", "burner_plate")], info=info)


def rotary_part_scene(*, lever=True, free_sense=-1, dark=False, grip_offset_m=0.0046, wedge=None, spring=0.0,
                      effect=True, texture_seed=0, support_texture_std=8.0, q_lo=-0.005, q_hi=2.1, q_on=0.5,
                      q0=0.0, hinge_kw=None, effect_ring_m=None):
    """A rotary part on a textured support with a colour-change region it switches (task-2 layout without the free
    objects). ``lever`` False = round part (no wedge: blocked twists slip the pads, the part image does not rotate);
    ``free_sense`` = the contract sense free at the start; ``grip_offset_m`` = hinge offset from the disk centre along
    +x (12 mm for the orbit-follow test); ``wedge`` overrides the gap model (False: an elongated part whose stop does not
    wedge); ``spring`` > 0 springs back after release (K9)."""
    cx, cy = TASK2_KNOB_CENTRE_XY
    hinge = FakeHinge(centre_xy=(cx + grip_offset_m, cy), free_sense=free_sense, q=q0, q_lo=q_lo, q_hi=q_hi,
                      q_on=q_on, wedge=lever if wedge is None else bool(wedge), spring=spring, **dict(hinge_kw or {}))
    bodies = _knob_bodies((cx, cy), hinge, lever=lever, dark=dark)
    if effect:
        bodies.append(_burner_body(TASK2_BURNER_XY, "knob", effect_ring_m=effect_ring_m))
    part = "knob_lever" if lever else "knob_round"
    info = {"prompt": TASK2_PROMPT, "rotary": "knob", "rotary_part": part, "effect_region": "burner_plate" if effect else None,
            "hover_eef": {"knob": (cx, cy, 1.03)}, "grasp_eef": {"knob": (cx, cy, 0.933)},
            "start_eef": (cx, cy - 0.19, 1.19),
            "regions": {"burner_plate": {"centre_xy": TASK2_BURNER_XY, "half_xy": (0.075, 0.075)}},
            "oracle_points": {"A": _local_top(part, (cx, cy), 0.955), "B": _local_top("burner_plate", TASK2_BURNER_XY, 0.9305)}
            if effect else {"A": _local_top(part, (cx, cy), 0.955)},
            "truth": {"lever_width_m": 0.0243 if lever else 0.025, "part_top_z": 0.955, "disk_top_z": 0.918,
                      "disk_radius_m": 0.0476, "hinge_xy": hinge.centre_xy, "disk_centre_xy": (cx, cy),
                      "elongation_axis_xy": (1.0, 0.0) if lever else None}}
    return FakeScene(bodies, support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed, support_texture_std=support_texture_std,
                     goal=[("turnon", "knob")], info=info)


def dark_rotary_part_scene(**kwargs):
    """The dark variant: a uniform dark ridge on a uniform dark disk (patch std ~1-3 grey) on a lighter support."""
    return rotary_part_scene(dark=True, **kwargs)


def _neighbour_centre(mx, my):
    """The neighbour handle's footprint centre for a free object at (mx, my): 4.5 cm off in x, 16 cm off in y (its
    near end at my - 0.110, 1.75 cm past the loop's far end at my - 0.0925)."""
    return (mx - 0.045, my - 0.160)


def free_object_with_thin_part_scene(*, texture_seed=0, support_texture_std=8.0, destination=True, neighbour=False):
    """A free object (66 mm cylinder) with a thin part (13 mm thick, ~60 mm long loop) and, optionally, the destination
    region (19 x 19 cm, top 30.5 mm above the support) for place_on; ``neighbour`` adds a second free object's thin
    handle (20 mm wide, along y) 4.5 cm beside the loop in x and beyond its end in y, with the oracle point N at its
    centre, so a top-down grasp across it at N (jaw along x) keeps the whole hand footprint (R0 +-31.6 mm along y)
    clear of the free object: N's palm band ends >= 3 cm short of the loop (review BF-R7), and the wrong grasp clamps
    the neighbour with no co-motion of the object (grip_not_on_obj, contract §C10.7 item 7)."""
    bodies = _moka_bodies(TASK2_MOKA_XY)
    if destination:
        bodies.append(_burner_body(TASK2_BURNER_XY, None, effect=False))
    if neighbour:
        mx, my = TASK2_MOKA_XY
        bodies.append(FakeBody("other_handle", "aabb", _neighbour_centre(mx, my) + (0.9286,), (0.020, 0.10, 0.0292),
                               colour=(30.0, 30.0, 32.0), texture_std=3.0, role="free", group="other"))
    mx, my = TASK2_MOKA_XY
    goal = [("on", "moka", "burner_plate")] if destination else []
    info = {"prompt": "put the pot on the plate region", "free_object": "moka", "thin_part": "moka_handle",
            "destination": "burner_plate" if destination else None,
            "hover_eef": {"moka_handle": (mx, my - 0.0795, 1.11)}, "grasp_eef": {"moka_handle": (mx, my - 0.0795, 1.020)},
            "regions": {"burner_plate": {"centre_xy": TASK2_BURNER_XY, "half_xy": (0.075, 0.075)}},
            "oracle_points": {"C": _local_top("moka_handle", (mx, my - 0.0795), 1.0345),
                              "D": _local_top("moka_body", TASK2_MOKA_XY, 1.0331),
                              **({"B": _local_top("burner_plate", TASK2_BURNER_XY, 0.9305)} if destination else {})},
            "oracle_whole": ["D"],
            "truth": {"loop_width_m": 0.013, "loop_length_m": 0.0595, "loop_top_z": 1.0345, "body_radius_m": 0.033,
                      "body_top_z": 1.0331, "loop_axis_xy": (0.0, 1.0), "destination_top_z": 0.9305}}
    if neighbour:
        info["oracle_points"]["N"] = _local_top("other_handle", _neighbour_centre(mx, my), 0.9578)
        info["hover_eef"]["other_handle"] = (*_neighbour_centre(mx, my), 1.10)
        info["grasp_eef"]["other_handle"] = (*_neighbour_centre(mx, my), 0.945)
    return FakeScene(bodies, support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed,
                     support_texture_std=support_texture_std, goal=goal, info=info)


def planar_region_scene(*, texture_seed=0, centre_xy=(-0.20, 0.08), support_texture_std=8.0):
    """A fixed 19 x 19 cm planar region, top 30.5 mm above the support, centred near the start view."""
    region = FakeBody("region", "aabb", (centre_xy[0], centre_xy[1], TASK2_SUPPORT_Z), (0.19, 0.19, 0.0305),
                      colour=(92.0, 92.0, 98.0), texture_std=7.0, texture_scale_m=0.0015)
    info = {"region": "region", "truth": {"top_z": TASK2_SUPPORT_Z + 0.0305, "support_z": TASK2_SUPPORT_Z,
                                          "polygon_xy": [[centre_xy[0] + sx * 0.095, centre_xy[1] + sy * 0.095]
                                                         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]},
            "oracle_points": {"B": _local_top("region", centre_xy, TASK2_SUPPORT_Z + 0.0305)}}
    return FakeScene([region], support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed,
                     support_texture_std=support_texture_std, info=info)


def textureless_dark_aabb_scene(*, texture_seed=0):
    """D7a negative: a wood-like textured plane (std 8) beside a uniform patch of the plane's own colour (std 1), and a
    textureless dark aabb (5 x 5 x 3 cm, std 0.5) standing on the light plane."""
    colour = (196.0, 176.0, 148.0)
    patch = {"polygon": [[-0.27, -0.03], [-0.21, -0.03], [-0.21, 0.03], [-0.27, 0.03]], "colour": colour, "std": 1.0}
    box = FakeBody("dark_box", "aabb", (-0.15, 0.04, TASK2_SUPPORT_Z), (0.05, 0.05, 0.03), colour=(30.0, 30.0, 32.0),
                   texture_std=0.5)
    info = {"uniform_patch_polygon": patch["polygon"], "dark_box": "dark_box",
            "truth": {"box_top_z": TASK2_SUPPORT_Z + 0.03, "support_z": TASK2_SUPPORT_Z}}
    return FakeScene([box], support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed, support_colour=colour,
                     support_texture_std=8.0, support_patches=[patch], info=info)


def raised_obstacle_on_destination_scene(*, texture_seed=0, height_m=0.04):
    """place_on negative: the free object, the destination region and a fixed obstacle standing on the region whose top
    is ``height_m`` above the region's top (perched / blocked_above)."""
    scene = free_object_with_thin_part_scene(texture_seed=texture_seed)
    bx, by = TASK2_BURNER_XY
    obstacle = FakeBody("obstacle", "aabb", (bx, by, TASK2_SUPPORT_Z + 0.0305), (0.06, 0.06, height_m),
                        colour=(120.0, 60.0, 40.0), texture_std=6.0)
    info = dict(scene.info)
    info["obstacle"] = "obstacle"
    info["truth"] = {**info["truth"], "obstacle_top_z": TASK2_SUPPORT_Z + 0.0305 + height_m}
    return FakeScene(scene.bodies + [obstacle], support_z=TASK2_SUPPORT_Z, texture_seed=texture_seed,
                     support_texture_std=8.0, goal=scene.goal, info=info)


# ------------------------------------------------------------------------------------------------------- test data
def load_probe(name):
    """The verbatim probe copies under tests/data/agent_speedup_v2/ (``sp_knob_sense_probe``, ``cp_knob_twist_probe``,
    ``cp_pot_lower_probe``) and the effect-probe subset (``effect_probe_t2_records``). Their ``sign`` / order labels are
    WORLD-z labels (contract §C1.3): SP/CP "+1" = contract sense -1 for a top-down hand."""
    return _json.loads((DATA_DIR / f"{name}.json").read_text())


EFFECT_PROBE_TAGS = ("reset", "pretwist", "pre_cross", "cross", "cross_plus1", "twist_end", "after_retreat")


def effect_probe_frame(tag, camera="agentview", *, upright=True):
    """An M0 effect-probe frame (RGB uint8) from tests/data/agent_speedup_v2/effect_probe_t2/. The files are verbatim
    POLICY-orientation frames (columns mirrored); ``upright`` (default) mirrors them back, as StepView does. Record
    indices of the tags: load_probe("effect_probe_t2_records")["frame_tags_to_record_index"]."""
    from PIL import Image
    rgb = _np.asarray(Image.open(DATA_DIR / "effect_probe_t2" / f"{tag}_{camera}.png").convert("RGB"))
    return _np.ascontiguousarray(rgb[:, ::-1]) if upright else rgb


def effect_probe_region_mask(camera="agentview"):
    """The effect probe's burner-plate top region (UPRIGHT agentview frame; derived from the t2_dump.json geometry with
    the probe's own plane_hits rule: 10286 px against the probe's 10285, edge-identical to its overlay)."""
    from PIL import Image
    return _np.asarray(Image.open(DATA_DIR / "effect_probe_t2" / f"region_{camera}.png").convert("L")) > 127


def survey_frame(name):
    """SURVEY/tasks/frames start views copied to tests/data/agent_speedup_v2/survey_frames (upright RGB uint8):
    ``t2_wrist`` (the V7 / texture-report image) and ``t2_agentview``."""
    from PIL import Image
    return _np.asarray(Image.open(DATA_DIR / "survey_frames" / f"{name}.png").convert("RGB"))


# ------------------------------------------------------------------------------- scripted truth-driven helpers
def _adapter(cap):
    from gpt_field.contact_skills import HoldingActionAdapter
    return HoldingActionAdapter(max_rotation_rad=cap)


def scripted_goto(world, target_xyz, *, gripper=-1.0, quat_xyzw=None, tol_m=0.0015, tol_rad=0.02, max_steps=200,
                  gain=1.5, render=False):
    """Truth-driven P move of the grip to ``target_xyz`` (tests only), holding or turning to ``quat_xyzw``; the
    probes' goto recipe (knob_sense_probe.py). Returns the number of steps."""
    target = _np.asarray(target_xyz, float)
    goal_q = world.quat.copy() if quat_xyzw is None else _np.asarray(quat_xyzw, float)
    for k in range(max_steps):
        err = target - world.grip
        R_err = _quat_matrix(goal_q) @ world.R.T
        rv = _rotvec_of(R_err)
        if float(_np.linalg.norm(err)) < tol_m and float(_np.linalg.norm(rv)) < tol_rad:
            return k
        rot = _np.clip(rv / ROTATION_PER_UNIT * gain, -1, 1)
        if not _np.any(rot):
            rot = _np.array([1e-9, 0.0, 0.0])
        action = [*_np.clip(err / TRANSLATION_PER_UNIT * gain, -1, 1), *rot, gripper]
        (world.step if render else world.advance)(action)
    return max_steps


def scripted_gripper(world, command, steps, *, render=False):
    for _ in range(int(steps)):
        (world.step if render else world.advance)([0.0, 0.0, 0.0, 1e-9, 0.0, 0.0, float(command)])


def scripted_twist(world, sense, steps, *, hold_xyz=None, cap_rad=0.1, world_z=False, close_first=True, log=None,
                   render=False):
    """The SP/CP twist recipe (tests only): each step targets the current orientation rotated by 2 cap about R2
    (``sense`` = contract sense) or, with ``world_z``, about world z (``sense`` = the SP/CP world-z label), through v1's
    HoldingActionAdapter capped at ``cap_rad``; position held at ``hold_xyz`` with a +-10 mm per-axis clip; gripper
    "close" (``close_first`` False only skips the explicit first close command; the adapter repeats close). Returns
    per-step logs {k, dyaw_deg (world z), q, gap_mm} (q = first rotary joint)."""
    ad = _adapter(cap_rad)
    hold = world.grip.copy() if hold_xyz is None else _np.asarray(hold_xyz, float)
    ad.adapt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], "close", world.state)      # adapter gripper state: close
    rotary = next((name for name, g in world.scene.groups.items() if g["hinge"] is not None), None)
    log = [] if log is None else log
    for k in range(int(steps)):
        s = world.state
        R = _rotmat(s[3:6])
        axis = _np.array([0.0, 0.0, 1.0]) if world_z else R[:, 2]
        target = _matrix_quat(_rot_axis(axis, sense * cap_rad * 2.0) @ R)
        a = ad.adapt(_np.clip(hold - s[:3], -0.01, 0.01), target, ("close" if close_first else "hold") if k == 0
                     else "hold", s)
        y0 = _yaw_of(R)
        (world.step if render else world.advance)(list(a))
        s2 = world.state
        log.append({"k": k, "dyaw_deg": round(_math.degrees(_wrap(_yaw_of(_rotmat(s2[3:6])) - y0)), 3),
                    "q": None if rotary is None else round(world.joint(rotary), 5),
                    "gap_mm": float((s2[6] - s2[7]) * 1000.0)})
    return log


def scripted_grasp_rotary(world, *, retreat_m=0.0056, close_steps=20, render=False):
    """Hover above the rotary part (scene.info hover_eef), touch down on the disk with the open hand, retreat
    (the spec's P10 5.6 mm), close: leaves the part clamped (world.held()["kind"] == "rotary")."""
    hover = world.scene.info["hover_eef"]["knob"]
    scripted_goto(world, hover, gripper=-1.0, render=render)
    for _ in range(40):
        z = world.grip[2]
        (world.step if render else world.advance)([0.0, 0.0, -0.2, 1e-9, 0.0, 0.0, -1.0])
        if abs(world.grip[2] - z) < 1e-5:
            break
    scripted_goto(world, world.grip + [0.0, 0.0, retreat_m], gripper=-1.0, tol_m=0.0003, render=render)
    scripted_gripper(world, 1.0, close_steps, render=render)
    return world.held()


def scripted_grasp_thin_part(world, *, grip_z=1.020, close_steps=25, lift_m=0.025, render=False):
    """Turn the jaw across the thin part (R1 -> +x, R0 -> +y: the camera on the object's side), descend to the
    height-stop grip z, close and lift: leaves the free object held."""
    hover = _np.asarray(world.scene.info["hover_eef"]["moka_handle"], float)
    q = _matrix_quat(_rot_axis((0.0, 0.0, 1.0), _math.pi / 2) @ world.R)
    scripted_goto(world, hover, gripper=-1.0, quat_xyzw=q, max_steps=300, render=render)
    scripted_goto(world, [hover[0], hover[1], grip_z], gripper=-1.0, quat_xyzw=q, render=render)
    scripted_gripper(world, 1.0, close_steps, render=render)
    if lift_m:
        scripted_goto(world, world.grip + [0.0, 0.0, lift_m], gripper=1.0, quat_xyzw=q, tol_m=0.001, render=render)
    return world.held()
