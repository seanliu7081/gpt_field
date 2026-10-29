"""End-to-end policy tests for agent_SpeedUp_v1 on the ORACLE fake wrist simulator.

The oracle transport answers pointing questions from the fake's true mark
projection (test-only geometry); decisions are scripted. Everything runs
offline through the real client, primitives, servo and mark prompting.
"""

import copy
import json
import math

import numpy as np
import pytest

from agent_speedup_fakes import DRAWER_PROMPT, FakeWristSim
from gpt_field.agent_speedup_v1 import AgentSpeedUpV1Policy
from gpt_field.contact_skills import axes_to_quaternion_xyzw
from gpt_field.agent_speedup_v1_client import AgentSpeedUpV1Client
from gpt_field.mark_prompting import (coarse_cell_size, fine_crop_origin, fine_crop_size, gravity_quarter_turns,
                                     rotate_pixel)
from test_field_history import envelope

NULL_PARAMS = {"target": None, "strategy": None, "side": None, "approach": None,
               "waypoint_offset_m": None, "more_waypoints_m": None, "max_travel_m": None}


def decision(skill, **params):
    return {"situation": f"scripted {skill}", "skill": skill, "params": {**NULL_PARAMS, **params}}


class OracleTransport:
    """Scripted decisions; pointing answers computed from the fake's true mark projection."""

    def __init__(self, sim, decisions, *, mark_index=0, coarse_override=None):
        self.sim, self.decisions = sim, list(decisions)
        self.mark_index, self.coarse_override = mark_index, coarse_override
        self.calls, self.kinds = [], []
        self._cell = None

    def _uv(self):
        """ORACLE mark pixel in the image the pointing model sees (turned so world up is at the top)."""
        uv, _ = self.sim.project(self.sim.marks[self.mark_index])
        R = self.sim.R
        return np.asarray(rotate_pixel(uv, gravity_quarter_turns(R[2, 1], R[2, 0]), 512))

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(copy.deepcopy(payload))
        properties = payload["text"]["format"]["schema"]["properties"]
        if "cell_id" in properties:
            self.kinds.append("coarse")
            uv = self._uv()
            cell = coarse_cell_size(512)
            self._cell = (int(np.clip(uv[1] // cell, 0, 7)) * 8 + int(np.clip(uv[0] // cell, 0, 7)) + 1
                          if self.coarse_override is None else self.coarse_override)
            data = {"cell_id": self._cell}
        elif "point_id" in properties:
            self.kinds.append("fine")
            uv = self._uv()
            context = json.loads(payload["input"][0]["content"][0]["text"]).get("context") or {}
            margin = 0.5 if context.get("strategy") == "frontal_bar_pinch" else 0.25   # handles use the wide crop
            origin = np.asarray(fine_crop_origin(self._cell, 512, margin), float)
            spacing = fine_crop_size(512, margin) / 8
            local = (uv - origin) / spacing
            col, row = (int(np.clip(local[0], 0, 7)), int(np.clip(local[1], 0, 7)))
            data = {"point_id": row * 8 + col + 1}
        else:
            self.kinds.append("decision")
            data = self.decisions.pop(0) if self.decisions else decision("retreat")
        return envelope(data, id=f"oracle-{len(self.calls)}", model="gpt-6-astra", reasoning={"effort": "medium"})


def make(sim, decisions, *, task_key="bowl_plate", local_retries=1, **transport_kwargs):
    transport = OracleTransport(sim, decisions, **transport_kwargs)
    client = AgentSpeedUpV1Client(token_provider=lambda: "offline-agent", task_key=task_key, transport=transport)
    return AgentSpeedUpV1Policy(client, local_retries=local_retries), transport


def run(policy, sim, steps, *, stop=None):
    obs = sim.observation()
    rows = []
    for _ in range(steps):
        decision_obj = policy.act(obs)
        after = sim.step(decision_obj.action)
        feedback = policy.observe_transition(obs, decision_obj, after)
        json.dumps(decision_obj.diagnostics, allow_nan=False)
        json.dumps(feedback, allow_nan=False)
        rows.append((decision_obj, feedback))
        obs = after
        if stop is not None and stop(policy, decision_obj):
            break
    return rows


def outcomes(policy, skill=None):
    return [o for o in policy.outcomes if skill is None or o["skill"] == skill]


def test_calibration_runs_before_any_model_call_and_first_decision_is_stateless():
    sim = FakeWristSim()
    policy, transport = make(sim, [decision("reach", waypoint_offset_m=[0.0, 0.0, -0.02])])
    rows = run(policy, sim, 140, stop=lambda p, d: len(outcomes(p, "reach")) == 1)
    first_call = next(i for i, (d, _) in enumerate(rows) if d.diagnostics["refreshed"])
    assert first_call >= 40, "calibration must run locally before the first decision"
    assert all(not d.diagnostics["api_calls_in_act"] for d, _ in rows[:first_call])
    assert policy.calibration is not None and policy.calibration.source == "online"
    assert policy.calibration.empty_floor_mm < 3.0 < policy.calibration.open_gap_mm
    payload = transport.calls[0]
    assert len(payload["input"]) == 1 and payload["input"][0]["role"] == "user"
    images = [part for part in payload["input"][0]["content"] if part["type"] == "input_image"]
    assert len(images) == 2 and all(part["detail"] == "high" for part in images)
    reach = outcomes(policy, "reach")[0]
    assert reach["status"] == "succeeded"
    assert reach["evidence"]["measured_displacement_m"][2] == pytest.approx(-0.02, abs=0.006)
    assert all(d.diagnostics["controller"] == "agent_skill" for d, _ in rows)


def test_rim_pinch_aligns_descends_to_floor_and_verifies_hold():
    rim = (0.10, 0.0, 0.952)
    sim = FakeWristSim(eef=(0.10, 0.018, 1.03), marks=(rim,), floor_z=0.916, object_width_m=0.005,
                       grasp_predicate=lambda s: s.grip_axis_lateral_error(rim) < 0.012 and s.state[2] < 0.94)
    policy, transport = make(sim, [decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")])
    run(policy, sim, 600, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["status"] == "succeeded", grasp
    assert grasp["reason"] == "hold_pass"
    assert transport.kinds[:3] == ["decision", "coarse", "fine"]
    assert policy.holding
    detail = grasp["evidence"]["details"][0]
    assert detail["contact"]["reason"] in ("contact_stall", "contact_touch")
    assert detail["close"]["object_present"] is True
    assert sim.attached


def test_rim_pinch_empty_close_retries_locally_without_choice_calls():
    rim = (0.10, 0.0, 0.952)
    sim = FakeWristSim(eef=(0.10, 0.018, 1.03), marks=(rim,), floor_z=0.916, object_width_m=0.005,
                       grasp_predicate=lambda s: False)
    policy, transport = make(sim, [decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")],
                             local_retries=1)
    run(policy, sim, 900, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["status"] == "failed"
    assert grasp["evidence"]["attempt_reasons"] == ["empty_close", "empty_close"]
    # The retry re-used the hover template: no pointing or choice call between the first decision and the
    # outcome; the only later call is the next decision issued once the macro has ended.
    assert transport.kinds[:3] == ["decision", "coarse", "fine"]
    assert transport.kinds[3:] in ([], ["decision"]), transport.kinds
    assert policy.agent_stats()["counts"]["local_retries_used"] == 1
    assert not policy.holding


def test_frontal_bar_pinch_orients_aligns_stalls_on_face_and_pulls():
    bar = (0.0, -0.14, 1.10)
    sim = FakeWristSim(eef=(0.012, -0.06, 1.112), marks=(bar,), render_marks=False,
                       bars=(((-0.06, -0.14, 1.10), (0.06, -0.14, 1.10), 0.003),), plane=((0.0, 1.0, 0.0), -0.162),
                       object_width_m=0.0173, prompt=DRAWER_PROMPT,
                       grasp_predicate=lambda s: s.grip_axis_lateral_error(bar) < 0.01 and s.state[1] < -0.14)
    policy, transport = make(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y"),
        decision("pull", max_travel_m=0.08)], task_key="drawer_bowl")
    run(policy, sim, 700, stop=lambda p, d: bool(outcomes(p, "pull")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["status"] == "succeeded", grasp
    R = sim.R
    assert R[:, 2] @ np.array([0.0, -1.0, 0.0]) > math.cos(math.radians(5))
    assert abs(R[:, 1] @ np.array([0.0, 0.0, 1.0])) > math.cos(math.radians(5))
    pull = outcomes(policy, "pull")[0]
    assert pull["status"] == "succeeded", pull
    assert pull["evidence"]["pulled_total_m"] == pytest.approx(0.08, abs=0.01)


def test_invalid_decision_is_rejected_with_an_outcome_and_a_new_decision_in_the_same_act():
    sim = FakeWristSim()
    policy, transport = make(sim, [decision("transport", waypoint_offset_m=[0.0, 0.0, 0.01]),
                                   decision("reach", waypoint_offset_m=None),
                                   decision("retreat")])
    rows = run(policy, sim, 200, stop=lambda p, d: bool(outcomes(p, "retreat")))
    rejected = [o for o in policy.outcomes if o["status"] == "rejected"]
    assert [o["reason"] for o in rejected] == ["precondition_unmet", "invalid_parameters"]
    assert policy.agent_stats()["counts"]["rejected_decisions"] == 2
    acts_with_two = [d for d, _ in rows if d.diagnostics["api_calls_in_act"].count("decision") == 2]
    assert acts_with_two, "a rejected decision is followed by another decision in the same act"
    assert outcomes(policy, "retreat")[0]["status"] == "succeeded"
    assert any(line["status"] == "rejected" for line in policy.ledger)


def test_pending_and_observation_invariants():
    sim = FakeWristSim()
    policy, _ = make(sim, [])
    obs = sim.observation()
    first = policy.act(obs)
    with pytest.raises(RuntimeError):
        policy.act(obs)
    after = sim.step(first.action)
    with pytest.raises(ValueError):
        policy.observe_transition(after, first, after)
    policy.observe_transition(obs, first, after)
    with pytest.raises(RuntimeError):
        policy.observe_transition(obs, first, after)
    with pytest.raises(ValueError):
        policy.act(sim.step([0, 0, 0, 0, 0, 0, -1]))
    stats = policy.agent_stats()
    json.dumps(stats, allow_nan=False)
    assert stats["rulings_assumed"]["R-1_online_image_jacobian"] is True


def test_never_emits_exact_zero_rotation():
    sim = FakeWristSim()
    policy, _ = make(sim, [decision("reach", waypoint_offset_m=[0.02, 0.0, 0.0]), decision("retreat")])
    rows = run(policy, sim, 160)
    assert all(any(d.action[3:6]) for d, _ in rows)


REAL_START_ROTVEC = (-3.1398, 0.0028, 0.0877)   # recorded LIBERO start: R[:,1]·z is numerical noise (-0.0006)


def _bar_sim(**kwargs):
    bar = (0.0, -0.14, 1.10)
    return FakeWristSim(eef=(0.012, -0.06, 1.112), marks=(bar,), render_marks=False,
                       bars=(((-0.06, -0.14, 1.10), (0.06, -0.14, 1.10), 0.003),), plane=((0.0, 1.0, 0.0), -0.162),
                        object_width_m=0.0173, prompt=DRAWER_PROMPT,
                        grasp_predicate=lambda s: s.grip_axis_lateral_error(bar) < 0.01 and s.state[1] < -0.14,
                        **kwargs), bar


def test_frontal_jaw_sign_takes_the_smaller_rotation_from_the_noisy_real_start():
    sim, _ = _bar_sim(rotvec=REAL_START_ROTVEC)
    assert sim.R[:, 1] @ np.array([0.0, 0.0, 1.0]) < 0, "fixture must reproduce the negative noise"
    policy, _ = make(sim, [decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch",
                                    approach="world_-y")], task_key="drawer_bowl")
    rows = run(policy, sim, 700, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    orient_steps = sum(1 for d, _ in rows if d.diagnostics["primitive"] == "orient")
    assert orient_steps < 20, f"a ~90 deg turn takes ~11 steps at the cap here, not {orient_steps}"
    assert sim.R[:, 1] @ np.array([0.0, 0.0, 1.0]) > math.cos(math.radians(5))


def test_ledger_carries_params_and_handle_release_opens_in_place():
    sim, _ = _bar_sim()
    policy, transport = make(sim, [
        decision("grasp_with", target="top_drawer_handle", strategy="frontal_bar_pinch", approach="world_-y"),
        decision("pull", max_travel_m=0.05),
        decision("place_release", waypoint_offset_m=[0.0, 0.0, 0.0])], task_key="drawer_bowl")
    rows = run(policy, sim, 900, stop=lambda p, d: bool(outcomes(p, "place_release")))
    release = outcomes(policy, "place_release")[0]
    assert release["status"] == "succeeded" and release["reason"] == "released_handle"
    assert not any(d.diagnostics["primitive"] == "lower" for d, _ in rows)
    assert not policy.holding and sim.gap_mm > 70
    lines = {line["skill"]: line for line in policy.ledger}
    assert lines["pull"]["params"] == {"max_travel_m": 0.05}
    assert lines["grasp_with"]["params"]["strategy"] == "frontal_bar_pinch"
    assert policy._compact(outcomes(policy, "grasp_with")[0])["evidence"]["last_attempt"]["gap_k15_mm"] > 10


def test_rim_pinch_after_a_turned_hand_restores_orientation_and_asks_again():
    rim = (0.10, 0.0, 0.952)
    # The hand is frontal (as after a handle) while the episode started top-down.
    sim = FakeWristSim(eef=(0.10, 0.018, 1.03), marks=(rim,), floor_z=0.916, rotvec=(math.pi / 2, 0.0, 0.0))
    policy, transport = make(sim, [decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")])
    policy._start_quaternion = axes_to_quaternion_xyzw((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    run(policy, sim, 400, stop=lambda p, d: bool(outcomes(p, "grasp_with")))
    grasp = outcomes(policy, "grasp_with")[0]
    assert grasp["reason"] == "orientation_restored" and "choose side" in grasp["detail"]
    assert transport.kinds.count("coarse") == 0, "no pointing in the stale (turned) view"
