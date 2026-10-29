"""agent_SpeedUp_v1 contact skills on the kinematic FakeWristSim; no simulator or model I/O."""

import json
import math

import numpy as np
import pytest

from agent_speedup_fakes import FakeWristSim, png_url
from gpt_field.action import ActionAdapter, _rotvec_quaternion
from gpt_field.contact_skills import (
    CONTACT_SKILLS_VERSION, DEFAULT_EMPTY_CURVE_MM, OPEN_GAP_MM_DEFAULT, AnchoredMove, Calibrate,
    GripperCalibration, GripperSettlePrimitive, HoldingActionAdapter, HoldPrimitive, Orient, Primitive,
    StepView, VerifyHold, WaypointReach, axes_to_quaternion_xyzw, classify_close, finger_gap_mm,
    matrix_to_quaternion_xyzw, quaternion_angle_rad, rotation_matrix)
from gpt_field.observation import PolicyObservation
from gpt_field.speed_first_gripper import GripperSettleConfig

TINY = png_url(np.zeros((8, 8, 3), np.uint8))
FRONTAL = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)


class StateSim(FakeWristSim):
    """FakeWristSim dynamics with a tiny placeholder image: contact skills read state8 only."""

    def observation(self):
        return PolicyObservation(TINY, TINY, tuple(float(v) for v in self.state), self.prompt)


def view_of(sim):
    return StepView(sim.observation())


def shifted(view, **changes):
    state = list(view.state)
    for index, value in changes.items():
        state[int(index[1:])] = value
    return StepView(PolicyObservation(TINY, TINY, tuple(state), "task"))


def run(primitive, sim, view=None, limit=400):
    view = view if view is not None else view_of(sim)
    feedback = []
    while not primitive.done:
        assert len(feedback) < limit
        action = primitive.next_action(view)
        assert len(action) == 7 and all(type(v) is float and abs(v) <= 1 for v in action)
        assert any(action[3:6]), "exactly-zero rotation would keep a stale OSC goal"
        after = StepView(sim.step(action))
        feedback.append(primitive.observe_transition(view, action, after))
        view = after
    json.dumps(feedback, allow_nan=False)
    return view, feedback


def outcome_of(primitive):
    outcome = primitive.outcome()
    json.dumps(outcome, allow_nan=False)
    assert set(outcome) == {"primitive", "status", "reason", "evidence", "actions"}
    assert outcome["actions"] == primitive.actions_executed
    return outcome


# ---------------------------------------------------------------- constants
def test_default_curve_and_calibration_record():
    assert CONTACT_SKILLS_VERSION == "agent-speedup-v1-contact-skills-2"
    assert len(DEFAULT_EMPTY_CURVE_MM) == 30
    for k, gap in ((5, 50.7), (10, 8.4), (15, 2.5), (20, 1.88), (25, 1.5), (30, 1.2)):
        assert DEFAULT_EMPTY_CURVE_MM[k - 1] == pytest.approx(gap)
    assert all(a >= b for a, b in zip(DEFAULT_EMPTY_CURVE_MM, DEFAULT_EMPTY_CURVE_MM[1:]))
    assert DEFAULT_EMPTY_CURVE_MM[0] < OPEN_GAP_MM_DEFAULT == 77.4
    calibration = GripperCalibration.default()
    assert calibration.source == "default_recorded" and calibration.inward_noise_mm == 0.02
    assert calibration.touch_threshold_mm == 0.10
    assert GripperCalibration(77.0, (5.0, 2.0), 0.05, "online").touch_threshold_mm == pytest.approx(0.15)
    assert calibration.empty_gap_at(0) == DEFAULT_EMPTY_CURVE_MM[0]
    assert calibration.empty_gap_at(15) == pytest.approx(2.5)
    assert calibration.empty_gap_at(99) == calibration.empty_floor_mm == pytest.approx(1.2)
    record = calibration.to_dict()
    json.dumps(record, allow_nan=False)
    assert GripperCalibration.from_dict(record) == calibration
    with pytest.raises(ValueError):
        GripperCalibration(77.0, (), 0.0, "online")
    with pytest.raises(ValueError):
        GripperCalibration(77.0, (1.0,), -0.1, "online")
    with pytest.raises(ValueError):
        GripperCalibration(77.0, (1.0,), 0.0, "oracle")
    with pytest.raises(ValueError):
        calibration.empty_gap_at(True)


# ------------------------------------------------------------ geometry
def test_start_and_frontal_tool_axes_and_axes_quaternion():
    R = rotation_matrix((math.pi, 0.0, 0.0))
    np.testing.assert_allclose(R[:, 1], [0, -1, 0], atol=1e-12)    # jaw along world -y
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)    # approach down
    np.testing.assert_allclose(R[:, 0], np.cross(R[:, 1], R[:, 2]), atol=1e-12)
    recorded = rotation_matrix((-3.139837268246076, 0.0028468604393764676, 0.08770975201502634))
    assert recorded[:, 1] @ [0, -1, 0] > 0.99 and recorded[:, 2] @ [0, 0, -1] > 0.99
    frontal = axes_to_quaternion_xyzw((0, 0, 1), (0, -1, 0))
    np.testing.assert_allclose(frontal, FRONTAL, atol=1e-12)
    view = StepView(PolicyObservation(TINY, TINY, (0, 0, 1, math.pi / 2, 0, 0, .02, -.02), "task"))
    np.testing.assert_allclose(view.R[:, 1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(view.R[:, 2], [0, -1, 0], atol=1e-12)
    # The jaw is Gram-Schmidt orthogonalized against the exact approach axis.
    np.testing.assert_allclose(axes_to_quaternion_xyzw((0, 0.3, 2), (0, -2, 0)), FRONTAL, atol=1e-12)
    start = _rotvec_quaternion(np.array([math.pi, 0, 0]))
    assert quaternion_angle_rad(start, FRONTAL) == pytest.approx(math.pi / 2)
    assert quaternion_angle_rad(-start, FRONTAL) == pytest.approx(math.pi / 2)
    assert quaternion_angle_rad(FRONTAL, -np.asarray(FRONTAL)) == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        axes_to_quaternion_xyzw((0, 0, 1), (0, 0, -3))
    with pytest.raises(ValueError):
        matrix_to_quaternion_xyzw(np.eye(2))


def test_quaternion_round_trips():
    rng = np.random.default_rng(3)
    rotvecs = [rng.normal(size=3) * rng.uniform(0, math.pi) for _ in range(300)]
    rotvecs += [np.array([math.pi, 0, 0]), np.array([0, math.pi, 0]), np.array([0, 0, math.pi]), np.zeros(3)]
    for rotvec in rotvecs:
        expected = _rotvec_quaternion(rotvec)
        R = rotation_matrix(rotvec)
        for q in (matrix_to_quaternion_xyzw(R), axes_to_quaternion_xyzw(R[:, 1], R[:, 2])):
            assert q[3] >= 0 and np.linalg.norm(q) == pytest.approx(1)
            assert min(np.abs(q - expected).max(), np.abs(q + expected).max()) < 1e-9
            assert quaternion_angle_rad(q, expected) < 1e-7
    assert finger_gap_mm((0, 0, 0, 0, 0, 0, 0.0387, -0.0387)) == pytest.approx(77.4)


def test_holding_adapter_never_emits_exactly_zero_rotation():
    state = (0.1, 0.2, 0.9, math.pi, 0, 0, 0.0387, -0.0387)
    q = _rotvec_quaternion(np.array(state[3:6]))
    assert ActionAdapter().adapt((0, 0, 0), q, "close", state)[3:6] == (0.0, 0.0, 0.0)
    action = HoldingActionAdapter().adapt((0, 0, 0), q, "close", state)
    assert action == (0.0, 0.0, 0.0, 2e-9, 0.0, 0.0, 1.0)
    turning = HoldingActionAdapter().adapt((0.01, 0, 0), FRONTAL, "open", state)
    assert turning == ActionAdapter().adapt((0.01, 0, 0), FRONTAL, "open", state)
    assert np.linalg.norm(np.asarray(turning[3:6]) * 0.5) == pytest.approx(0.15)


def test_step_view_is_an_immutable_cached_upright_snapshot():
    sim = FakeWristSim()
    view = StepView(sim.observation())
    assert view.state.shape == (8,) and view.state.dtype == np.float64
    assert view.gap_mm == pytest.approx(77.4)
    np.testing.assert_allclose(view.position, sim.state[:3])
    np.testing.assert_allclose(view.R, sim.R)
    np.testing.assert_allclose(view.quaternion_xyzw, _rotvec_quaternion(sim.state[3:6]))
    np.testing.assert_array_equal(view.wrist_rgb_upright, sim.wrist_upright())   # legacy -> upright
    assert view.wrist_rgb_upright is view.wrist_rgb_upright
    assert view.wrist_gray_upright.shape == (512, 512) and view.wrist_gray_upright.dtype == np.float64
    assert view.external_rgb_upright.shape == (512, 512, 3)
    with pytest.raises(ValueError):
        view.state[0] = 1.0
    with pytest.raises(ValueError):
        view.wrist_rgb_upright[0, 0, 0] = 1
    with pytest.raises(AttributeError):
        view.state = np.zeros(8)
    sim.state[0] += 0.1
    assert view.position[0] == pytest.approx(0.10)


# ------------------------------------------------------------ anchored move
def test_anchored_move_reaches_travel_on_a_frozen_axis_with_lateral_anchor():
    sim = StateSim()
    view = view_of(sim)
    move = AnchoredMove(view, axis=(0, 0, -2), travel_m=0.02, gripper="open",
                        lateral_target=(0.105, 0.02, 1.0), name="lower")
    end, feedback = run(move, sim, view)
    outcome = outcome_of(move)
    assert outcome["status"] == "succeeded" and outcome["reason"] == "travel_reached"
    evidence = outcome["evidence"]
    assert evidence["axis"] == [0.0, 0.0, -1.0]
    assert 0.0195 <= evidence["travel_m"] < 0.0225
    assert evidence["contact"] is None and evidence["finger_deflected"] is None
    assert abs(end.position[0] - 0.105) < 0.0015
    assert evidence["free_ratio"] == pytest.approx(0.26) and evidence["free_ratio_measured"]
    assert all(step["ratio"] == pytest.approx(0.26) for step in feedback[:2])
    assert quaternion_angle_rad(end.quaternion_xyzw, view.quaternion_xyzw) < 1e-6
    assert all(action[6] == -1.0 for action in sim.actions)
    assert outcome["actions"] == len(sim.actions) == math.ceil(0.0195 / 0.0026)


def test_anchored_move_stalls_on_a_supported_floor():
    sim = StateSim(floor_z=0.985)
    move = AnchoredMove(view_of(sim), axis=(0, 0, -1), travel_m=0.1, gripper="open", stop_on_contact=True)
    run(move, sim)
    outcome = outcome_of(move)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "contact_stall")
    evidence = outcome["evidence"]
    assert evidence["contact"] == "stall" and evidence["contact_xyz"][2] == pytest.approx(0.985)
    assert evidence["travel_m"] == pytest.approx(0.015)
    assert evidence["free_ratio_measured"] and evidence["free_ratio"] == pytest.approx(0.26)
    assert (np.mean(evidence["ratio_window"]) < AnchoredMove.STALL_FRACTION * 0.26
            and evidence["ratio_window"][-1] == 0 and len(evidence["ratio_window"]) == AnchoredMove.STALL_WINDOW)
    # Without contact detection the same blocked move only exhausts its budget.
    sim = StateSim(floor_z=0.985)
    blind = AnchoredMove(view_of(sim), axis=(0, 0, -1), travel_m=0.1, gripper="open", max_steps=12)
    run(blind, sim)
    assert (blind.status, blind.reason, blind.actions_executed) == ("failed", "budget", 12)


@pytest.mark.parametrize("finger,name", [(1, "finger_1"), (2, "finger_2")])
def test_anchored_move_open_finger_touch(finger, name):
    sim = StateSim(deflect=lambda s: (finger, 0.5) if s.state[2] < 0.992 else None)
    threshold = GripperCalibration.default().touch_threshold_mm
    move = AnchoredMove(view_of(sim), axis=(0, 0, -1), travel_m=0.05, gripper="open",
                        stop_on_contact=True, touch_threshold_mm=threshold)
    _, feedback = run(move, sim)
    outcome = outcome_of(move)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "contact_touch")
    assert outcome["evidence"]["contact"] == "touch" and outcome["evidence"]["finger_deflected"] == name
    assert feedback[-1]["inward_mm"][finger - 1] == pytest.approx(0.5)
    assert outcome["evidence"]["contact_xyz"][2] < 0.992
    # Touch needs an explicit threshold (and an open gripper); otherwise travel completes.
    sim = StateSim(deflect=lambda s: (finger, 0.5) if s.state[2] < 0.992 else None)
    blind = AnchoredMove(view_of(sim), axis=(0, 0, -1), travel_m=0.02, gripper="open", stop_on_contact=True)
    run(blind, sim)
    assert blind.reason == "travel_reached"


def test_anchored_move_gap_jump():
    sim = StateSim(deflect=lambda s: (2, 0.3) if s.steps >= 3 else None)
    move = AnchoredMove(view_of(sim), axis=(0, -1, 0), travel_m=0.05, gripper="open", stop_on_gap_jump=True)
    _, feedback = run(move, sim)
    outcome = outcome_of(move)
    assert (outcome["status"], outcome["reason"], outcome["actions"]) == ("succeeded", "gap_jump", 4)
    assert feedback[-1]["gap_step_mm"] == pytest.approx(-0.3)
    assert outcome["evidence"]["contact"] is None


def test_anchored_move_rejects_commands_beyond_the_osc_limit():
    view = view_of(StateSim())
    with pytest.raises(ValueError):
        AnchoredMove(view, axis=(0, 0, 0), travel_m=0.02, gripper="open")
    with pytest.raises(ValueError):
        AnchoredMove(view, axis=(0, 0, 1), travel_m=0.02, gripper="hold")
    with pytest.raises(ValueError):
        AnchoredMove(view, axis=(0, 0, 1), travel_m=0.02, gripper="open", command_m=0.049)


# ---------------------------------------------------------- gripper skills
def test_calibrate_learns_an_online_empty_curve_from_the_fake_gripper():
    sim = StateSim()
    start = view_of(sim)
    calibrate = Calibrate(start)
    end, _ = run(calibrate, sim, start)
    outcome = outcome_of(calibrate)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "calibrated_online")
    calibration = calibrate.calibration
    assert calibration.source == "online" and len(calibration.empty_curve_mm) == 30
    assert calibration.empty_curve_mm[0] == pytest.approx(69.4)
    assert calibration.empty_floor_mm == pytest.approx(1.2)
    assert calibration.open_gap_mm == pytest.approx(77.4)
    assert calibration.inward_noise_mm < 1e-9 and calibration.touch_threshold_mm == 0.10
    assert outcome["evidence"]["calibration"] == calibration.to_dict()
    assert outcome["evidence"]["close_settle_k"] == 30 and outcome["evidence"]["open_settle_k"] == 15
    assert outcome["actions"] == 30 + 15 + 10 == len(sim.actions)
    assert [a[6] for a in sim.actions] == [1.0] * 30 + [-1.0] * 25
    assert np.linalg.norm(end.position - start.position) < 1e-9


def test_calibrate_falls_back_to_recorded_curve_without_closing_motion():
    sim = StateSim()
    sim.state[6:8] = (0.0006, -0.0006)          # already closed: no measurable closing motion
    calibrate = Calibrate(view_of(sim))
    run(calibrate, sim)
    outcome = outcome_of(calibrate)
    assert outcome["reason"] == "calibrated_default"
    assert calibrate.calibration == GripperCalibration.default()
    assert outcome["evidence"]["close_stop_reason"] == "action_budget_exhausted"


def test_gripper_settle_and_classify_close_empty_versus_object():
    calibration = GripperCalibration.default()
    results = {}
    for label, width in (("empty", None), ("object", 0.02)):
        sim = StateSim(object_width_m=width)
        settle = GripperSettlePrimitive("close", view_of(sim))
        run(settle, sim)
        outcome = outcome_of(settle)
        assert (outcome["status"], outcome["reason"]) == ("succeeded", "measured_fingers_stable")
        assert settle.settle_k == 15 == len(settle.gap_curve_mm)
        assert outcome["evidence"]["gap_curve_mm"] == settle.gap_curve_mm
        results[label] = classify_close(settle, calibration)
        json.dumps(results[label], allow_nan=False)
    assert results["empty"]["object_present"] is False
    assert results["empty"]["gap_k15_mm"] == pytest.approx(1.2)
    assert results["object"]["object_present"] is True
    assert results["object"]["gap_k15_mm"] == pytest.approx(20.0)
    assert results["object"]["empty_k15_mm"] == pytest.approx(2.5) and results["object"]["settle_k"] == 15
    sim = StateSim()
    opening = GripperSettlePrimitive("open", view_of(sim))
    with pytest.raises(ValueError):
        classify_close(opening, calibration)


def test_gripper_settle_keeps_local_settle_exact_state_contract():
    sim = StateSim()
    view = view_of(sim)
    settle = GripperSettlePrimitive("close", view)
    action = settle.next_action(view)
    assert action[3] == 2e-9 and action[6] == 1.0
    after = StepView(sim.step(action))
    settle.observe_transition(view, action, after)
    with pytest.raises(ValueError, match="preceding measured feedback"):
        settle.next_action(shifted(after, s0=after.state[0] + 1e-6))
    assert settle.next_action(after) is not None


def test_hold_primitive_records_inward_finger_steps():
    sim = StateSim(deflect=lambda s: (1, 0.05) if s.steps == 2 else None)
    hold = HoldPrimitive(view_of(sim), steps=5, gripper="open")
    run(hold, sim)
    outcome = outcome_of(hold)
    assert outcome["reason"] == "hold_complete" and len(hold.inward_steps_mm) == 10
    assert hold.inward_steps_mm[4] == pytest.approx(0.05) and hold.inward_noise_mm > 0


# --------------------------------------------------------------- VerifyHold
def grasped(**kwargs):
    sim = StateSim(object_width_m=0.02, **kwargs)
    settle = GripperSettlePrimitive("close", view_of(sim))
    view, _ = run(settle, sim)
    return sim, view


class Monitor:
    def __init__(self):
        self.calls = 0

    def observe(self, view):
        self.calls += 1
        return {"available": True, "ok": self.calls < 3, "peak": np.float64(0.9), "displacement_px": 2.0}


def test_verify_hold_pass_with_secondary_monitor():
    sim, view = grasped()
    monitor = Monitor()
    verify = VerifyHold(view, axis=(0, 0, 1), travel_m=0.025, calibration=GripperCalibration.default(),
                        monitor=monitor)
    run(verify, sim, view)
    outcome = outcome_of(verify)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "hold_pass")
    evidence = outcome["evidence"]
    assert evidence["min_gap_mm"] == pytest.approx(20.0) and evidence["hold_floor_mm"] == pytest.approx(2.2)
    assert evidence["monitor_disagrees"] is True and evidence["monitor"]["peak"] == 0.9
    assert monitor.calls == outcome["actions"]
    assert all(action[6] == 1.0 for action in sim.actions)


def test_verify_hold_detects_lost_grasp_and_jam():
    sim, view = grasped(slip_after_lift_m=0.008)
    verify = VerifyHold(view, axis=(0, 0, 1), travel_m=0.025, calibration=GripperCalibration.default())
    run(verify, sim, view)
    outcome = outcome_of(verify)
    assert (outcome["status"], outcome["reason"]) == ("failed", "hold_lost")
    assert outcome["evidence"]["travel_m"] < 0.025 and outcome["evidence"]["gap_end_mm"] < 2.2
    sim, view = grasped(plane=((0, 0, -1), -1.004))
    verify = VerifyHold(view, axis=(0, 0, 1), travel_m=0.025, calibration=GripperCalibration.default())
    run(verify, sim, view)
    outcome = outcome_of(verify)
    assert (outcome["status"], outcome["reason"]) == ("failed", "hold_jam")
    assert outcome["evidence"]["contact"] == "stall"
    assert outcome["evidence"]["contact_xyz"][2] == pytest.approx(1.004)


def test_verify_hold_after_empty_close_fails_without_moving():
    sim = StateSim()
    settle = GripperSettlePrimitive("close", view_of(sim))
    view, _ = run(settle, sim)
    verify = VerifyHold(view, axis=(0, 0, 1), travel_m=0.025, calibration=GripperCalibration.default())
    assert verify.done and verify.outcome()["reason"] == "hold_lost" and verify.outcome()["actions"] == 0
    with pytest.raises(RuntimeError):
        verify.next_action(view)


# ------------------------------------------------------------------- Orient
def test_orient_converges_to_the_frontal_pose_holding_position():
    sim = StateSim()
    start = view_of(sim)
    orient = Orient(start, target_xyzw=FRONTAL)
    assert orient.gripper == "open"
    end, _ = run(orient, sim, start)
    outcome = outcome_of(orient)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "orientation_reached")
    assert outcome["evidence"]["initial_angle_deg"] == pytest.approx(90.0)
    assert outcome["evidence"]["final_angle_deg"] <= 3.0 and outcome["actions"] <= 12
    np.testing.assert_allclose(end.R[:, 1], [0, 0, 1], atol=0.06)
    np.testing.assert_allclose(end.R[:, 2], [0, -1, 0], atol=0.06)
    assert np.linalg.norm(end.position - start.position) < 1e-9
    assert Orient(end, target_xyzw=FRONTAL).done and Orient(end, target_xyzw=FRONTAL).actions_executed == 0


@pytest.mark.parametrize("ratio,reason", [(0.06, "budget"), (0.004, "rotation_stall")])
def test_orient_budget_and_rotation_stall(ratio, reason):
    sim = StateSim(rotation_ratio=ratio)
    orient = Orient(view_of(sim), target_xyzw=FRONTAL)
    run(orient, sim)
    outcome = outcome_of(orient)
    assert (outcome["status"], outcome["reason"]) == ("failed", reason)
    budget = outcome["evidence"]["budget_steps"]
    assert budget in (65, 66)
    assert outcome["actions"] == (budget if reason == "budget" else 12)


# ------------------------------------------------------------ WaypointReach
def test_waypoint_reach_goal():
    sim = StateSim()
    start = view_of(sim)
    reach = WaypointReach(start, [0.04, -0.03, -0.05], gripper="open")
    end, _ = run(reach, sim, start)
    outcome = outcome_of(reach)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "goal_reached")
    evidence = outcome["evidence"]
    assert evidence["final_error_m"] <= 0.005 and outcome["actions"] <= 30
    np.testing.assert_allclose(evidence["measured_displacement_m"], [0.04, -0.03, -0.05], atol=0.005)
    assert evidence["commanded_offset_m"] == [0.04, -0.03, -0.05]
    assert quaternion_angle_rad(end.quaternion_xyzw, start.quaternion_xyzw) < 1e-6
    assert all(action[6] == -1.0 for action in sim.actions)


def test_waypoint_reach_stops_at_construction_or_without_progress():
    view = view_of(StateSim())
    boundary = WaypointReach(view, [0.09, 0, 0], gripper="close")
    assert boundary.done and boundary.actions_executed == 0
    outcome = outcome_of(boundary)
    assert (outcome["status"], outcome["reason"]) == ("failed", "field_boundary_reached")
    assert outcome["evidence"]["plan_reason"] == "goal_outside_known_grid"
    with pytest.raises(RuntimeError):
        boundary.next_action(view)
    near = WaypointReach(view, [0.003, 0, 0], gripper="open")
    assert near.done and near.outcome()["reason"] == "goal_reached"
    low = view_of(StateSim(eef=(0.1, 0.0, 0.02)))
    invalid = WaypointReach(low, [0, 0, -0.05], gripper="open")
    assert invalid.done and invalid.outcome()["reason"] == "no_valid_plan"
    sim = StateSim(floor_z=0.99)
    blocked = WaypointReach(view_of(sim), [0, 0, -0.05], gripper="open")
    run(blocked, sim)
    assert (blocked.status, blocked.reason) == ("failed", "no_progress")


# --------------------------------------------------------------- invariants
def test_pending_action_invariants():
    sim = StateSim()
    view = view_of(sim)
    move = AnchoredMove(view, axis=(0, 0, -1), travel_m=0.01, gripper="open")
    assert isinstance(move, Primitive) and move.pointing_request is None
    with pytest.raises(RuntimeError, match="no pending"):
        move.observe_transition(view, (0.0,) * 7, view)
    with pytest.raises(RuntimeError, match="not finished"):
        move.outcome()
    action = move.next_action(view)
    with pytest.raises(RuntimeError, match="observe_transition"):
        move.next_action(view)
    after = StepView(sim.step(action))
    wrong = list(action)
    wrong[0] += 1e-6
    with pytest.raises(ValueError, match="exact pending action"):
        move.observe_transition(view, wrong, after)
    with pytest.raises(ValueError, match="exact pending action"):
        move.observe_transition(view, action[:6], after)
    with pytest.raises(ValueError, match="before-state"):
        move.observe_transition(after, action, after)
    with pytest.raises(TypeError):
        move.observe_transition(view, action, after.observation)
    assert move.pending and move.actions_executed == 0
    feedback = move.observe_transition(view, list(action), after)
    assert feedback["step"] == 1 and feedback["primitive"] == "anchored_move"
    run(move, sim, after)
    assert move.done
    with pytest.raises(RuntimeError, match="finished"):
        move.next_action(after)


# ---------------------------------------------- live run 001 stop events (contact-skills-2)
def test_waypoint_reach_stops_at_the_workspace_edge_with_reach_limit():
    sim = StateSim(eef=(0.10, 0.0, 1.0), tilt_beyond_x=0.11)
    reach = WaypointReach(view_of(sim), [0.06, 0.0, 0.0], gripper="open")
    run(reach, sim)
    outcome = outcome_of(reach)
    assert (outcome["status"], outcome["reason"]) == ("failed", "reach_limit")
    evidence = outcome["evidence"]
    assert outcome["actions"] <= 8 and evidence["final_angle_deg"] >= math.degrees(0.05)
    assert evidence["blocked_direction_xy"] == pytest.approx([1.0, 0.0, 0.0], abs=0.05)
    assert evidence["progress_fraction"] < 0.5
    free = StateSim(eef=(0.10, 0.0, 1.0))
    inward = WaypointReach(view_of(free), [-0.06, 0.0, 0.0], gripper="open")
    run(inward, free)
    assert outcome_of(inward)["reason"] == "goal_reached"


def test_waypoint_reach_held_mode_stops_on_hold_loss_and_on_held_object_contact():
    sim, view = grasped(slip_after_lift_m=0.012)
    reach = WaypointReach(view, [0.0, 0.0, 0.05], gripper="close", hold_floor_mm=2.2, gap_ref_mm=20.0)
    run(reach, sim, view)
    outcome = outcome_of(reach)
    assert (outcome["status"], outcome["reason"]) == ("failed", "hold_lost")
    assert outcome["evidence"]["gap_end_mm"] < 2.2 and outcome["evidence"]["measured_displacement_m"][2] < 0.045
    sim, view = grasped()
    start = sim.steps
    sim.deflect = lambda s: (2, -1.5) if s.steps >= start + 3 else None
    reach = WaypointReach(view, [0.05, 0.0, 0.0], gripper="close", hold_floor_mm=2.2, gap_ref_mm=20.0)
    run(reach, sim, view)
    outcome = outcome_of(reach)
    assert (outcome["status"], outcome["reason"]) == ("failed", "held_object_contact")
    assert outcome["evidence"]["max_gap_rise_mm"] == pytest.approx(1.5, abs=0.01)
    assert outcome["evidence"]["gap_baseline_mm"] == pytest.approx(20.0)
    # a move that starts already wedged uses its own start gap as the baseline
    sim, view = grasped()
    wedged = shifted(view, q6=view.state[6] + 0.004)
    reach = WaypointReach(wedged, [0.03, 0.0, 0.0], gripper="close", hold_floor_mm=2.2, gap_ref_mm=20.0)
    assert reach.gap_baseline_mm == pytest.approx(24.0)


def test_waypoint_reach_budget_endings_name_their_mechanism():
    sim = StateSim(translation_ratio=0.01)
    reach = WaypointReach(view_of(sim), [0.05, 0.0, 0.0], gripper="open")
    run(reach, sim)
    outcome = outcome_of(reach)
    assert (outcome["status"], outcome["reason"], outcome["actions"]) == ("failed", "blocked", 30)
    assert outcome["evidence"]["progress_fraction"] < 0.3
    assert outcome["evidence"]["blocked_direction_xy"] == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    drift = StateSim()
    reach = WaypointReach(view_of(drift), [0.02, 0.0, 0.0], gripper="open", reach_limit=False)
    view = view_of(drift)
    while not reach.done:
        action = reach.next_action(view)
        after = StepView(drift.step(action))
        if reach.actions_executed >= 1:   # keep pitching the hand after arrival
            drift.state[3:6] = (np.array(drift.state[3:6]) + np.array([0.0, 0.08, 0.0])).tolist()
            after = StepView(drift.observation())
        reach.observe_transition(view, action, after)
        view = after
    outcome = outcome_of(reach)
    assert outcome["reason"] == "orientation_drift" and outcome["evidence"]["final_error_m"] <= 0.015


def test_waypoint_reach_uncontrolled_motion_and_left_planning_grid():
    sim = StateSim()
    reach = WaypointReach(view_of(sim), [0.05, 0.0, 0.0], gripper="open")
    view = view_of(sim)
    action = reach.next_action(view)
    sim.step(action)
    sim.state[2] -= 0.03     # uncommanded 3 cm drop in one step
    after = StepView(sim.observation())
    reach.observe_transition(view, action, after)
    assert reach.outcome()["reason"] == "uncontrolled_motion"
    sim = StateSim()
    reach = WaypointReach(view_of(sim), [0.05, 0.0, 0.0], gripper="open")
    view = view_of(sim)
    action = reach.next_action(view)
    sim.step(action)
    sim.state[1] += 0.0195    # overshoot out of the +-0.08 m planning grid (under the 20 mm jump test)
    sim.state[0] += 0.075
    after = StepView(sim.observation())
    reach.observe_transition(view, action, after)
    assert reach.done and reach.outcome()["reason"] in ("left_planning_grid", "uncontrolled_motion")


def test_orient_fails_fast_on_pose_drift_and_accepts_a_loose_budget_ending():
    class Drifting(StateSim):
        def step(self, action):
            obs = super().step(action)
            self.state[0] -= 0.004    # the hand is dragged back while rotating (workspace edge)
            return self.observation()

    sim = Drifting()
    orient = Orient(view_of(sim), target_xyzw=FRONTAL, max_pose_drift_m=0.02)
    run(orient, sim)
    outcome = outcome_of(orient)
    assert (outcome["status"], outcome["reason"]) == ("failed", "pose_drift")
    assert outcome["evidence"]["pose_error_m"] > 0.02 and outcome["actions"] < 12
    sim = StateSim(rotation_ratio=0.06)
    loose = Orient(view_of(sim), target_xyzw=FRONTAL, budget_tolerance_rad=math.radians(90))
    run(loose, sim)
    assert loose.outcome()["reason"] == "orientation_reached_loose"


def test_anchored_move_gap_rise_marks_a_held_support_contact():
    sim, view = grasped()
    start = sim.steps
    sim.deflect = lambda s: (2, -0.4) if s.steps >= start + 6 else None
    move = AnchoredMove(view, axis=(0, 0, -1), travel_m=0.05, gripper="close", stop_on_contact=True,
                        gap_rise_mm=0.3)
    run(move, sim, view)
    outcome = outcome_of(move)
    assert (outcome["status"], outcome["reason"]) == ("succeeded", "gap_rise")
    assert outcome["evidence"]["gap_end_mm"] - outcome["evidence"]["gap_start_mm"] >= 0.3


# ---------------------------------------------- drawer live run 001 (contact-skills-3)
def test_waypoint_reach_open_finger_contact_and_quiet_hold_and_stall_arming():
    sim = StateSim()
    start = sim.steps
    sim.deflect = lambda s: (1, 1.2) if s.steps >= start + 3 else None    # a finger pushed 1.2 mm inward
    reach = WaypointReach(view_of(sim), [0.05, 0.0, 0.0], gripper="open")
    run(reach, sim)
    outcome = outcome_of(reach)
    assert outcome["reason"] == "finger_contact" and outcome["evidence"]["finger_gap_change_mm"] < -1.0
    quiet = HoldPrimitive(view_of(StateSim()), steps=14, gripper="open", until_quiet=True)
    run(quiet, StateSim())
    assert quiet.outcome()["actions"] == 2 and quiet.outcome()["evidence"]["quiet"] is True
    # a move started while the arm drifts back realizes ~0 at first: the stall arms only after 5 steps
    drifting = StateSim(translation_ratio=0.02)
    move = AnchoredMove(view_of(drifting), axis=(0, -1, 0), travel_m=0.05, gripper="open", stop_on_contact=True,
                        arm_stall_after_motion=True)
    run(move, drifting)
    assert move.outcome()["reason"] == "contact_stall" and move.outcome()["actions"] >= 5
    default = StateSim(translation_ratio=0.02)
    unarmed = AnchoredMove(view_of(default), axis=(0, -1, 0), travel_m=0.05, gripper="open", stop_on_contact=True)
    run(unarmed, default)
    assert unarmed.outcome()["actions"] == AnchoredMove.STALL_MIN_STEPS     # default unchanged (rim, lift, pull)
