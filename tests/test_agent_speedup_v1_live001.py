"""Regression tests for the agent_SpeedUp_v1 live run 001 fixes (policy-2 / contact-skills-2 / prompt-2).

Scenarios mirror the verified failure analysis of runs/agent_speedup_v1_bowl_zero_shot_001: external-view
direction map, transport that never noticed the drop, the place lowering bound, the workspace edge, the
restore that rotated in place at the edge, and the evidence the model never saw. ORACLE fakes only.
"""

import json
import math

import numpy as np
import pytest

from agent_speedup_fakes import FakeWristSim
from gpt_field.agent_speedup_v1 import AgentSpeedUpV1Policy, tilt_from_straight_down_deg
from gpt_field.agent_speedup_v1_client import AgentSpeedUpV1Client
from gpt_field.contact_skills import axes_to_quaternion_xyzw
from test_agent_speedup_v1_policy import OracleTransport, decision, outcomes, run

RIM = (0.10, 0.0, 0.952)


def make(sim, decisions, *, external_axes="off", local_retries=1):
    transport = OracleTransport(sim, decisions)
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key="bowl_plate", transport=transport)
    return AgentSpeedUpV1Policy(client, local_retries=local_retries, external_axes=external_axes), transport


def summaries(transport):
    """state_summary of every decision request, in order."""
    result = []
    for payload in transport.calls:
        if "skill" in payload["text"]["format"]["schema"]["properties"]:
            result.append(json.loads(payload["input"][0]["content"][0]["text"])["state_summary"])
    return result


def rim_sim(**kwargs):
    return FakeWristSim(eef=(0.10, 0.018, 1.03), marks=(RIM,), floor_z=0.916, object_width_m=0.005,
                        grasp_predicate=lambda s: s.grip_axis_lateral_error(RIM) < 0.012 and s.state[2] < 0.94,
                        **kwargs)


GRASP = decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")


def done_with(skill, count=1):
    return lambda policy, _: len(outcomes(policy, skill)) >= count


def test_probe_moves_publish_a_measured_external_map_and_ledger_shifts():
    sim = FakeWristSim(external_px_per_m=800.0)
    policy, transport = make(sim, [decision("reach", waypoint_offset_m=[0.0, 0.03, 0.0]), decision("retreat")],
                             external_axes="measured")
    run(policy, sim, 400, stop=done_with("retreat"))
    fit = policy._external_axes_fit
    assert fit["available"] and len(fit["probes"]) == 3
    first, second = summaries(transport)[:2]
    axes = first["external_image_axes_world"]
    assert axes["image_right"] == pytest.approx([0.0, 1.0, 0.0], abs=0.05)
    assert axes["image_up"] == pytest.approx([-1.0, 0.0, 0.0], abs=0.05)
    line = second["ledger"][-1]
    shift = line["external_image_shift_px_right_down"]      # +y moved the hand image-right (24 px nominal)
    assert line["skill"] == "reach" and shift[0] > 15 and abs(shift[1]) < 4
    np.testing.assert_allclose(sim.state[:2], [0.10, 0.05], atol=0.012)   # probes are net zero
    off, transport_off = make(FakeWristSim(), [decision("retreat")])
    run(off, off.client.transport.sim, 120, stop=done_with("retreat"))
    assert summaries(transport_off)[0]["external_image_axes_world"] is None


def test_transport_stops_the_step_the_hold_is_lost_and_remembers_the_drop():
    sim = rim_sim(slip_after_lift_m=0.04)
    policy, transport = make(sim, [GRASP, decision("transport", waypoint_offset_m=[0.0, 0.0, 0.05]),
                                   decision("retreat")])
    run(policy, sim, 900, stop=done_with("transport"))
    moved = outcomes(policy, "transport")[0]
    assert (moved["status"], moved["reason"]) == ("failed", "lost_in_transport")
    assert moved["evidence"]["steps"] < 20, "must stop as soon as the gap collapses, not at the budget"
    assert not policy.holding
    history = policy._target_history
    assert history["held_target"] == "black_bowl" and history["lost_at_step"] is not None
    run(policy, sim, 5)
    assert summaries(transport)[-1]["target_history"]["lost_at_step"] == history["lost_at_step"]


def test_transport_held_object_contact_lifts_one_centimetre_and_keeps_the_grasp():
    sim = rim_sim()
    policy, transport = make(sim, [GRASP, decision("transport", waypoint_offset_m=[0.03, 0.0, -0.03])])
    run(policy, sim, 700, stop=done_with("grasp_with"))
    assert policy.holding
    pick_z = policy._pick_floor_z
    sim.deflect = lambda s: (2, -1.5) if s.state[2] < pick_z + 0.02 else None   # pressed onto a support
    run(policy, sim, 200, stop=done_with("transport"))
    moved = outcomes(policy, "transport")[0]
    assert (moved["status"], moved["reason"]) == ("failed", "held_object_contact")
    assert moved["evidence"]["backoff"]["evidence"]["travel_m"] == pytest.approx(0.010, abs=0.002)
    assert "lifted 1 cm" in moved["detail"] and policy.holding


def test_held_object_contact_while_rising_backs_off_against_the_move_not_up():
    """Drawer 002 executor lens F4: under an overhang, lifting 1 cm after the contact repeated it until the drop."""
    sim = rim_sim()
    policy, transport = make(sim, [GRASP, decision("transport", waypoint_offset_m=[0.0, 0.0, 0.06])])
    run(policy, sim, 700, stop=done_with("grasp_with"))
    assert policy.holding
    ceiling = sim.state[2] + 0.03
    sim.deflect = lambda s: (2, -1.5) if s.state[2] > ceiling else None   # pressed up against an overhang
    run(policy, sim, 200, stop=done_with("transport"))
    moved = outcomes(policy, "transport")[0]
    assert (moved["status"], moved["reason"]) == ("failed", "held_object_contact")
    assert "moved back 1 cm against the move" in moved["detail"] and policy.holding
    assert sim.state[2] < ceiling


def test_held_contact_while_rising_after_a_blocked_grasp_backs_out_from_under_the_overhang():
    sim = rim_sim()
    policy, transport = make(sim, [GRASP, decision("transport", waypoint_offset_m=[0.0, 0.0, 0.06])])
    run(policy, sim, 700, stop=done_with("grasp_with"))
    policy._held_away_world = [0.0, 1.0, 0.0]          # as recorded by a hold_pass after blocked_above_rim
    start = sim.state[:3].copy()
    ceiling = start[2] + 0.03
    sim.deflect = lambda s: (2, -1.5) if s.state[2] > ceiling else None
    run(policy, sim, 200, stop=done_with("transport"))
    moved = outcomes(policy, "transport")[0]
    assert moved["reason"] == "held_object_contact" and "out from under the overhang" in moved["detail"]
    assert sim.state[1] - start[1] > 0.006 and policy.holding


def test_place_release_hover_never_descends_and_lowering_is_relative_to_the_pick_height():
    sim = rim_sim()
    lift = decision("transport", waypoint_offset_m=[0.0, 0.0, 0.075])
    policy, transport = make(sim, [GRASP, lift, lift, decision("place_release", waypoint_offset_m=[0.02, 0.0, -0.03])])
    run(policy, sim, 1400, stop=done_with("place_release"))
    placed = outcomes(policy, "place_release")[0]
    assert (placed["status"], placed["reason"]) == ("succeeded", "released"), placed.get("detail")
    evidence = placed["evidence"]
    assert evidence["reach"]["evidence"]["measured_displacement_m"][2] > -0.002     # dz clipped to >= 0
    lowered = (evidence["lower_fast"]["evidence"]["travel_m"] + evidence["lower"]["evidence"]["travel_m"])
    assert lowered > 0.15, "the old fixed 0.15 m bound would have stopped above the support"
    assert evidence["lower"]["reason"] in ("contact_stall", "gap_rise")
    assert evidence["lower_bound_m"] > 0.15 and not policy.holding
    assert policy._target_history["released_at_step"] is not None
    held = [s for s in summaries(transport) if s["holding_object"]]
    assert held and all(s["held_height_above_pick_m"] is not None for s in held)


def test_workspace_edge_is_reported_recorded_and_guards_further_pushes():
    sim = FakeWristSim(eef=(0.05, 0.02, 1.0), tilt_beyond_x=0.09)
    policy, transport = make(sim, [decision("reach", waypoint_offset_m=[0.07, 0.0, 0.0]),
                                   decision("reach", waypoint_offset_m=[0.05, 0.02, 0.0]),
                                   decision("reach", waypoint_offset_m=[-0.05, 0.0, 0.0])])
    run(policy, sim, 400, stop=done_with("reach", 3))
    edge, guarded, back = outcomes(policy, "reach")
    assert (edge["status"], edge["reason"]) == ("failed", "reach_limit")
    assert edge["evidence"]["steps"] <= 8 and "workspace edge" in edge["detail"]
    assert (guarded["status"], guarded["reason"]) == ("rejected", "beyond_recorded_reach_limit")
    assert back["status"] == "succeeded"
    second = summaries(transport)[1]
    assert second["reach_limits"][0]["blocked_direction_world"] == pytest.approx([1.0, 0.0, 0.0], abs=0.05)
    assert second["ledger"][-1]["reason"] == "reach_limit" and "detail" in second["ledger"][-1]


def test_restore_moves_back_toward_the_robot_and_up_before_turning_top_down():
    from gpt_field.contact_skills import matrix_to_quaternion_xyzw
    from gpt_field.observation import quaternion_to_axisangle
    pitch = math.radians(-20.0)     # hand pitched outward (approach gains +x), as at the workspace edge
    top_down = np.diag([1.0, -1.0, -1.0])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rotvec = quaternion_to_axisangle(matrix_to_quaternion_xyzw(Ry @ top_down))
    sim = FakeWristSim(eef=(0.10, 0.02, 1.0), rotvec=tuple(rotvec))
    assert tilt_from_straight_down_deg(sim.R) == pytest.approx(20.0, abs=0.5) and sim.R[0, 2] > 0.3
    policy, transport = make(sim, [GRASP])
    policy._start_quaternion = axes_to_quaternion_xyzw((0., -1., 0.), (0., 0., -1.))
    start_x = sim.state[0]
    run(policy, sim, 300, stop=done_with("grasp_with"))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["reason"] == "orientation_restored", grasp
    restore = grasp["evidence"]["orient_start"]
    moved = restore["pre_move"]["measured_displacement_m"]
    assert moved[0] < -0.04 and moved[2] > 0.02, "inward and up before rotating"
    assert sim.state[0] < start_x - 0.04 and tilt_from_straight_down_deg(sim.R) < 5.0
    assert restore["evidence"]["max_pose_drift_m"] == pytest.approx(0.02)


def test_state_summary_rounds_and_reports_tilt_counts_and_loop_warning():
    sim = FakeWristSim()
    bad = decision("transport", waypoint_offset_m=[0.02, 0.0, 0.0])      # rejected: nothing is held
    policy, transport = make(sim, [bad] * 6 + [decision("retreat")])
    run(policy, sim, 200, stop=done_with("retreat"))
    states = summaries(transport)
    assert states[0]["approach_tilt_from_straight_down_deg"] == pytest.approx(0.0, abs=0.5)
    assert states[0]["counts"]["failed_in_a_row"] == 0 and states[0]["loop_warning"] is None
    assert states[5]["counts"]["failed_in_a_row"] == 5 and "repeating similar" in states[5]["loop_warning"]

    def decimals(value):
        if isinstance(value, float):
            text = repr(value)
            return len(text.split(".")[1]) if "." in text and "e" not in text else 0
        if isinstance(value, dict):
            return max([decimals(v) for v in value.values()] or [0])
        if isinstance(value, list):
            return max([decimals(v) for v in value] or [0])
        return 0

    assert all(decimals(state) <= 4 for state in states)


def test_reach_limit_memory_keeps_horizontal_pushes_only():
    """Real smoke 012: a move back and 3 cm up lost orientation near the edge; remembering its horizontal
    remainder (-x, toward the robot) would have blocked the way back."""
    policy, _ = make(FakeWristSim(), [])
    policy._note_reach_limit({"blocked_direction_xy": [-1.0, 0.0, 0.0], "end_xyz": [0.10, 0.0, 1.146],
                              "commanded_offset_m": [-0.075, 0.0, 0.03], "final_angle_deg": 6.2})
    assert policy._reach_limits == []
    policy._note_reach_limit({"blocked_direction_xy": [1.0, 0.0, 0.0], "end_xyz": [0.133, -0.011, 1.117],
                              "commanded_offset_m": [0.075, 0.0, -0.01], "final_angle_deg": 3.0})
    assert len(policy._reach_limits) == 1 and policy._reach_limits[0]["blocked_direction_world"][0] == 1.0
