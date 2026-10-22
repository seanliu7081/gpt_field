"""Speed features (prompt-4): multi-waypoint reach/transport and chained handle skills. ORACLE fakes only."""

import math

import numpy as np
import pytest

from agent_speedup_fakes import FakeWristSim
from gpt_field.agent_speedup_v1 import AgentSpeedUpV1Policy, tilt_from_straight_down_deg
from gpt_field.agent_speedup_v1_client import AgentSpeedUpV1Client
from test_agent_speedup_v1_drawer001 import BarOracle, bar_sim
from test_agent_speedup_v1_policy import decision, outcomes, run


def make(sim, decisions, task_key="bowl_plate", oracle=None):
    transport = (oracle or BarOracle)(sim, decisions)
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key=task_key, transport=transport)
    return AgentSpeedUpV1Policy(client, local_retries=1, external_axes="off"), transport


def test_reach_with_more_waypoints_runs_a_whole_detour_in_one_decision():
    sim = FakeWristSim(eef=(0.10, 0.02, 1.0))
    policy, transport = make(sim, [decision("reach", waypoint_offset_m=[0.0, 0.0, 0.04],
                                            more_waypoints_m=[[0.05, 0.0, 0.0], [0.0, 0.0, -0.04]])])
    start = sim.state[:3].copy()
    run(policy, sim, 300, stop=lambda p, d: bool(outcomes(p, "reach")))
    reach = outcomes(policy, "reach")[0]
    assert reach["status"] == "succeeded" and len(reach["evidence"]["segments"]) == 3
    np.testing.assert_allclose(sim.state[:3] - start, [0.05, 0.0, 0.0], atol=0.008)
    np.testing.assert_allclose(reach["evidence"]["measured_displacement_m"], [0.05, 0.0, 0.0], atol=0.008)
    assert transport.kinds.count("decision") == 2   # the detour, plus the next decision asked in the same act


def test_path_stops_at_the_first_failing_waypoint():
    sim = FakeWristSim(eef=(0.10, 0.02, 1.0), floor_z=0.98)
    policy, _ = make(sim, [decision("reach", waypoint_offset_m=[0.03, 0.0, 0.0],
                                    more_waypoints_m=[[0.0, 0.0, -0.06], [0.03, 0.0, 0.0]])])
    run(policy, sim, 300, stop=lambda p, d: bool(outcomes(p, "reach")))
    reach = outcomes(policy, "reach")[0]
    assert reach["status"] == "failed" and reach["detail"].startswith("waypoint 2 of 3 ended")
    assert [s["segment"] for s in reach["evidence"]["segments"]] == [1, 2]


def test_invalid_combinations_are_rejected():
    sim = FakeWristSim()
    policy, _ = make(sim, [decision("retreat", more_waypoints_m=[[0.0, 0.0, 0.02]]),
                           decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left",
                                    max_travel_m=0.05),
                           decision("reach", waypoint_offset_m=[0.0, 0.0, 0.02], more_waypoints_m=[[0.0, 0.0, 0.001]])])
    run(policy, sim, 200, stop=lambda p, d: len([o for o in p.outcomes if o["status"] == "rejected"]) >= 3)
    rejected = [o for o in policy.outcomes if o["status"] == "rejected"]
    assert [o["reason"] for o in rejected[:3]] == ["invalid_parameters"] * 3


def test_grasp_then_pull_in_one_decision_and_release_turns_top_down():
    sim = bar_sim((0.012, -0.06, 1.112))
    policy, transport = make(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y",
                 max_travel_m=0.08),
        decision("place_release", waypoint_offset_m=[0.0, 0.0, 0.0])], task_key="drawer_bowl")
    policy._start_quaternion = None
    run(policy, sim, 1200, stop=lambda p, d: bool(outcomes(p, "place_release")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["status"] == "succeeded" and grasp["reason"].startswith("hold_pass_then_pull")
    assert grasp["evidence"]["pulled_total_m"] == pytest.approx(0.08, abs=0.01)
    release = outcomes(policy, "place_release")[0]
    assert release["reason"] == "released_handle" and "turned the hand top-down" in release["detail"]
    assert tilt_from_straight_down_deg(sim.R) < 5.0 and not policy.holding
    assert transport.kinds.count("decision") == 3   # grasp+pull, release+turn, plus the next decision
