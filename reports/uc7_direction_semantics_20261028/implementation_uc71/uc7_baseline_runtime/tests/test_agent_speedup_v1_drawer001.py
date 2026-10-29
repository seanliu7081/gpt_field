"""Regression tests for the agent_SpeedUp_v1 drawer live run 001 fixes (uniform handle bar).

runs/agent_speedup_v1_drawer_zero_shot_001: twelve handle grasps failed in Align before contact because a
pointed point on a uniform bar is localized only across the bar. ORACLE fakes only.
"""

import math
import types

import numpy as np
import pytest

from agent_speedup_fakes import DRAWER_PROMPT, FakeWristSim
from gpt_field.agent_speedup_v1 import AgentSpeedUpV1Policy
from gpt_field.agent_speedup_v1_client import AgentSpeedUpV1Client
from test_agent_speedup_v1_policy import OracleTransport, decision, outcomes, run

BAR_START, BAR_END, BAR_CENTER = (-0.06, -0.14, 1.10), (0.06, -0.14, 1.10), np.array([0.0, -0.14, 1.10])


class BarOracle(OracleTransport):
    """Points at the bar centre (ORACLE projection); the fake renders no disc mark."""

    def __init__(self, sim, decisions):
        super().__init__(types.SimpleNamespace(marks=[BAR_CENTER], project=sim.project), decisions)
        self._real = sim

    def _uv(self):
        self.sim.R = self._real.R
        return super()._uv()


def make(sim, decisions):
    transport = BarOracle(sim, decisions)
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key="drawer_bowl", transport=transport)
    return AgentSpeedUpV1Policy(client, local_retries=1, external_axes="off"), transport


def bar_sim(eef):
    on_bar = lambda s: (abs(s.state[2] - BAR_CENTER[2]) < 0.006 and abs(s.state[0]) < 0.05 and s.state[1] < -0.14)
    return FakeWristSim(eef=eef, marks=(), bars=((BAR_START, BAR_END, 0.003),), plane=((0.0, 1.0, 0.0), -0.162),
                        object_width_m=0.0173, prompt=DRAWER_PROMPT, grasp_predicate=on_bar)


@pytest.mark.parametrize("eef", [(0.012, -0.06, 1.112), (0.035, -0.06, 1.085)])
def test_frontal_bar_pinch_grasps_a_uniform_handle_from_an_off_centre_hover(eef):
    sim = bar_sim(eef)
    policy, transport = make(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y"),
        decision("pull", max_travel_m=0.08)])
    run(policy, sim, 900, stop=lambda p, d: bool(outcomes(p, "pull")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert (grasp["status"], grasp["reason"]) == ("succeeded", "hold_pass"), grasp
    align = grasp["evidence"]["details"][0]["align"]
    assert align["reason"] == "aligned" and align["along_record"]["source"] == "pointed"
    assert transport.kinds[:3] == ["decision", "coarse", "fine"]
    pull = outcomes(policy, "pull")[0]
    assert pull["status"] == "succeeded" and pull["evidence"]["pulled_total_m"] == pytest.approx(0.08, abs=0.01)


def test_frontal_pointing_images_are_turned_world_up_and_answers_map_back():
    sim = bar_sim((0.035, -0.06, 1.085))
    policy, transport = make(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y")])
    rows = run(policy, sim, 700, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    pointing = [r for d, _ in rows for r in d.diagnostics["pointing"]]
    assert pointing and all(r["quarter_turns"] in (1, 3) for r in pointing)   # jaw vertical: world up = image right/left
    import json
    contexts = [json.loads(call["input"][0]["content"][0]["text"])["context"] for call in transport.calls
                if "cell_id" in call["text"]["format"]["schema"]["properties"]
                or "point_id" in call["text"]["format"]["schema"]["properties"]]
    assert contexts and all("world up is at the top" in c["image_orientation"] for c in contexts)
    # the oracle answered in the turned image; the grasp can only succeed if the point was mapped back correctly
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["reason"] == "hold_pass", grasp


class WrongFirstOracle(BarOracle):
    """First ``wrong`` fine answers land 40 px across the bar (off it), like the live wine-rack / side-wall picks."""

    def __init__(self, sim, decisions, wrong):
        super().__init__(sim, decisions)
        self.wrong, self.fine_calls = wrong, 0

    def __call__(self, endpoint, payload, headers, timeout_s):
        if "point_id" in payload["text"]["format"]["schema"]["properties"]:
            self.fine_calls += 1
        return super().__call__(endpoint, payload, headers, timeout_s)

    def _uv(self):
        uv = super()._uv()
        return uv + np.array([0.0, 40.0]) if 0 < self.fine_calls <= self.wrong else uv   # turned image: across = rows


@pytest.mark.parametrize("wrong,reason", [(1, "hold_pass"), (2, "pointed_not_a_bar")])
def test_bar_answers_failing_the_orientation_check_are_re_asked_once(wrong, reason):
    sim = bar_sim((0.012, -0.06, 1.112))
    transport = WrongFirstOracle(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y")], wrong)
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key="drawer_bowl", transport=transport)
    policy = AgentSpeedUpV1Policy(client, local_retries=0, external_axes="off")
    rows = run(policy, sim, 700, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["reason"] == reason, grasp
    fine = [r for d, _ in rows for r in d.diagnostics["pointing"] if r["stage"] == "fine"]
    assert [r["bar_check"]["ok"] for r in fine] == ([False, True] if wrong == 1 else [False, False])
    assert [r["pointing_round"] for r in fine] == [1, 2]
    import json
    stages = [("coarse" if "cell_id" in c["text"]["format"]["schema"]["properties"] else "fine",
               json.loads(c["input"][0]["content"][0]["text"])["context"].get("previous_answer_rejected"))
              for c in transport.calls if "skill" not in c["text"]["format"]["schema"]["properties"]]
    # the rejected answer restarts from the COARSE stage, with the feedback in context
    assert [stage for stage, _ in stages] == ["coarse", "fine", "coarse", "fine"]
    assert stages[0][1] is None and "not on a bar" in stages[2][1]


class NotVisibleOracle(BarOracle):
    def __call__(self, endpoint, payload, headers, timeout_s):
        properties = payload["text"]["format"]["schema"]["properties"]
        if "cell_id" in properties:
            self.calls.append(payload)
            self.kinds.append("coarse")
            from test_field_history import envelope
            return envelope({"cell_id": 0}, id="nv", model="gpt-6-astra", reasoning={"effort": "medium"})
        return super().__call__(endpoint, payload, headers, timeout_s)


def test_per_target_loop_warning_fires_after_three_failed_grasps_despite_successful_reaches():
    """Drawer live 001: 12 handle grasps failed while successful reaches kept resetting failed_in_a_row."""
    sim = bar_sim((0.012, -0.06, 1.112))
    grasp = decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y")
    nudge = decision("reach", waypoint_offset_m=[0.0, 0.0, 0.006])
    transport = NotVisibleOracle(sim, [grasp, nudge, grasp, nudge, grasp, nudge])
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key="drawer_bowl", transport=transport)
    policy = AgentSpeedUpV1Policy(client, local_retries=0, external_axes="off")
    run(policy, sim, 900, stop=lambda p, d: len(outcomes(p, "reach")) >= 3)
    reasons = [o["reason"] for o in outcomes(policy, "grasp_with")]
    assert reasons == ["mark_not_visible"] * 3
    assert "hidden behind them" in outcomes(policy, "grasp_with")[0]["detail"]
    import json
    states = [json.loads(c["input"][0]["content"][0]["text"])["state_summary"] for c in transport.calls
              if "skill" in c["text"]["format"]["schema"]["properties"]]
    assert states[-1]["loop_warning"] and "top_drawer_handle failed 3 times" in states[-1]["loop_warning"]
    assert states[-1]["world_up_in_wrist_image"] in ("right", "left")


def test_frontal_stretch_guard_refuses_far_hovers_and_outward_goals():
    """Drawer live 001: from step 436 the arm was pinned at full stretch (wrist 0.788-0.804 m from the shoulder)."""
    from gpt_field.agent_speedup_v1 import FRONTAL_STRETCH_LIMIT_M, wrist_to_shoulder_m
    from gpt_field.contact_skills import axes_to_quaternion_xyzw
    from gpt_field.observation import quaternion_to_axisangle
    frontal = tuple(quaternion_to_axisangle(np.asarray(axes_to_quaternion_xyzw((0., 0., 1.), (0., -1., 0.)))))
    far = (0.097, -0.016, 1.10)                                  # the live step-420 hover
    assert wrist_to_shoulder_m(far, (0, -1, 0)) > FRONTAL_STRETCH_LIMIT_M
    sim = bar_sim(far)
    policy, transport = make(sim, [decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch",
                                            approach="world_-y")])
    run(policy, sim, 200, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["reason"] == "hover_beyond_frontal_reach" and "robot base" in grasp["detail"]
    assert "coarse" not in transport.kinds, "refused before any turn, pointing or motion"
    near = FakeWristSim(eef=(0.03, -0.05, 1.09), rotvec=frontal)
    policy, transport = make(near, [decision("reach", waypoint_offset_m=[0.075, 0.0, 0.0]),
                                    decision("reach", waypoint_offset_m=[-0.05, 0.0, 0.0])])
    run(policy, near, 300, stop=lambda p, d: len(outcomes(p, "reach")) >= 2)
    outward, inward = outcomes(policy, "reach")
    assert (outward["status"], outward["reason"]) == ("rejected", "beyond_frontal_reach")
    assert inward["status"] == "succeeded"
    import json
    state = json.loads(transport.calls[0]["input"][0]["content"][0]["text"])["state_summary"]
    assert state["frontal_arm_reach"]["limit_m"] == FRONTAL_STRETCH_LIMIT_M
    assert state["robot_base_direction_world"][0] < -0.9


class ShelfSim(FakeWristSim):
    """Something above the container stops the hand near the standard rim point (drawer live 002: the palm landed
    on the opened drawer's front panel 14 cm above the rim); 2.5 cm farther out the descent is free."""

    def __init__(self, *args, rim, shelf_z, **kwargs):
        super().__init__(*args, **kwargs)
        self.rim, self.shelf_z = np.asarray(rim, float), shelf_z

    def step(self, action):
        obs = super().step(action)
        if self.grip_axis_lateral_error(self.rim) < 0.015 and self.state[2] < self.shelf_z:
            self.state[2] = self.shelf_z
            obs = self.observation()
        return obs


def test_rim_contact_before_the_ladder_cutoff_is_blocked_and_retried_outward():
    rim = (0.10, 0.0, 0.952)
    sim = ShelfSim(eef=(0.10, 0.018, 1.10), marks=(rim,), floor_z=0.916, object_width_m=0.005, rim=rim, shelf_z=1.07,
                   grasp_predicate=lambda s: s.grip_axis_lateral_error(rim) < 0.04 and s.state[2] < 0.94)
    transport = OracleTransport(sim, [decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")])
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key="bowl_plate", transport=transport)
    policy = AgentSpeedUpV1Policy(client, local_retries=2, external_axes="off")
    run(policy, sim, 1500, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["evidence"]["attempt_reasons"][0] == "blocked_above_rim", grasp["evidence"]["attempt_reasons"]
    assert grasp["reason"] == "hold_pass"
    first, second = grasp["evidence"]["details"][:2]
    assert first.get("close") is None, "a blocked descent must not close the gripper"
    assert second["shift_m"] == pytest.approx(-0.025)       # outward, never toward the container centre
    away = -np.asarray(sim.R[:, 1], float)                  # wrist_left: the container centre is at +jaw
    away[2] = 0.0
    np.testing.assert_allclose(grasp["evidence"]["away_from_blocked_side_world"], away / np.linalg.norm(away),
                               atol=0.02)
    assert "before lifting more than about 5 cm" in grasp["detail"]


def test_orientation_restored_is_not_counted_as_a_failed_grasp():
    """Drawer live 002: orientation_restored counted as a failure, so the loop warning fired after 2 real failures."""
    sim = bar_sim((0.012, -0.06, 1.112))
    policy, _ = make(sim, [])
    params = {"target": "black_bowl", "strategy": "rim_pinch", "side": "wrist_left"}
    for reason in ("orientation_restored", "empty_close", "orientation_restored", "empty_close"):
        policy._finish({"skill": "grasp_with", "params": params, "status": "failed", "reason": reason})
    assert policy._grasp_failures_by_target["black_bowl"] == 2 and policy._failed_in_a_row == 2
    assert policy._loop_warning() is None
    policy._finish({"skill": "grasp_with", "params": params, "status": "failed", "reason": "blocked_above_rim"})
    assert policy._grasp_failures_by_target["black_bowl"] == 3 and "failed 3 times" in policy._loop_warning()
