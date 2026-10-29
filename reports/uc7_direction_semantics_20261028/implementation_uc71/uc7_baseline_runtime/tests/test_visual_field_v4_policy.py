"""Behavioral integration checks: meaningful local actions, missing wrist, recovery."""
import copy

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.visual_field_policy_v4 import VisualFieldPolicyV4
from gpt_field.visual_monitor_v4 import RoleAwareVisualMonitor
from gpt_field.visual_field_policy import images


Q = (0., -.5, 0., -2., 0., 1.5, 0.)
ORIGIN = np.array([.2, 0., 1.])
PATCHES = [np.random.default_rng(seed).integers(0, 256, (11, 11, 3), dtype=np.uint8) for seed in range(3)]


def observation(offset=(0., 0., 0.), *, static_feature=False, missing=False, payload=False, slip=False):
    pos = np.asarray(offset, float)
    im = np.full((224, 224, 3), 40, dtype=np.uint8)
    move = np.zeros(2) if static_feature else np.rint(pos[:2]*1000).astype(int)
    for i, point in enumerate(([50+move[0], 90+move[1]], [160, 140], [30+move[0], 70+move[1]])):
        if missing and i == 0:
            continue
        if i == 2 and not payload:
            continue
        if slip and i == 0:
            point = [50, 90]
        x, y = map(int, point)
        if 5 <= x < 219 and 5 <= y < 219:
            im[y-5:y+6, x-5:x+6] = PATCHES[i]
    # Deliberately uninformative/cropped wrist: it must not gate the primary field.
    wrist = np.full_like(im, 70)
    return PolicyObservation(image_to_data_url(im, rotate=False), image_to_data_url(wrist, rotate=False),
        tuple(ORIGIN+pos)+(0., 0., 0., .04, .04), "move the object to its target", Q, (0.,)*7)


def prediction(*, field=True, payload=False, phase="approach", actions=None):
    return {"controller": "visual_field" if field else "direct", "phase": phase,
        "target_id": "destination", "hypothesis_id": "same_subgoal", "reason": "fixture hypothesis",
        "visual_references": {"external": {
            "moving_id": "object" if payload else "tool", "moving_kind": "payload" if payload else "tool",
            "moving_point_uv": [50/223, 90/223], "target_id": "destination",
            "target_anchor_uv": [160/223, 140/223], "target_point_uv": [110/223, 90/223],
            "tool_point_uv": [30/223, 70/223] if payload else None}, "wrist": None},
        "field_intent": {"goal_offset_m": [.06, 0., 0.], "orientation_target_xyzw": [0., 0., 0., 1.],
            "gripper_command": "hold", "goal_tolerance_m": .005, "max_actions": 30, "max_travel_m": .12,
            "command_limit_m": .01, "tool_radius_m": .015, "uncertainty_m": .005,
            "obstacles": [], "unknown_regions": []} if field else None,
        "direct_actions": None if field else (actions or [[.1, 0., 0., 0., 0., 0., 0.]]*5)}


class Client:
    last_request_key = "offline-fixture"

    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.blocks = []

    def reset_history(self):
        pass

    def predict(self, obs):
        self.calls += 1
        return copy.deepcopy(self.value)

    def record_block(self, block):
        self.blocks.append(copy.deepcopy(block))


def test_one_request_drives_twenty_real_feedback_actions_with_missing_wrist():
    client = Client(prediction())
    policy = VisualFieldPolicyV4(client)
    offset = np.zeros(3)
    before = observation(offset)
    for _ in range(20):
        decision = policy.act(before)
        assert decision.diagnostics["selection"]["accepted"], decision.diagnostics["selection"]
        assert decision.action[3:] == (0., 0., 0., 0.)
        offset += np.asarray(decision.action[:3])*.05*.25  # Actuator lag.
        after = observation(offset)
        policy.observe_transition(before, decision, after)
        before = after
    policy.finish()
    assert client.calls == 1
    assert offset[0] > .03
    assert policy.field_effective_actions == 20
    assert policy.field_goal_improvement_m > .03
    assert policy.monitor.response_ready
    assert not policy.monitor.summary()["wrist_target_reliable"]
    assert client.blocks[0]["effective_field_actions"] == 20


def test_direct_contact_executes_exact_five_supplied_actions_unchanged():
    supplied = [[.01*i, -.03, .0, .0, .0, .0, 1.] for i in range(5)]
    client = Client(prediction(field=False, phase="grasp", actions=supplied))
    policy = VisualFieldPolicyV4(client)
    obs = observation()
    for action in supplied:
        d = policy.act(obs)
        assert list(d.action) == action
        policy.observe_transition(obs, d, obs)
    assert client.calls == 1 and len(policy.attempts) == 1
    assert policy.field_nonzero_actions == 0
    policy.act(obs)
    assert client.calls == 2


def test_static_wrong_control_feature_cannot_validate_and_endlessly_reuse_field():
    policy = VisualFieldPolicyV4(Client(prediction()))
    offset = np.zeros(3)
    before = observation(offset, static_feature=True)
    for _ in range(8):
        d = policy.act(before)
        offset += np.asarray(d.action[:3])*.05*.5
        after = observation(offset, static_feature=True)
        policy.observe_transition(before, d, after)
        before = after
        if policy.refresh_reason:
            break
    assert policy.refresh_reason == "controlled_feature_unresponsive"
    assert not policy.monitor.response_ready
    assert policy.client.calls == 1
    assert .006 <= offset[0] < .015


def test_lost_external_track_refreshes_without_removing_geometry():
    pred = prediction()
    pred["field_intent"]["unknown_regions"] = [{"id": "far_unknown", "lower_offset_m": [-.1, .07, -.1],
        "upper_offset_m": [.1, .1, .1], "uncertainty_m": 0.}]
    policy = VisualFieldPolicyV4(Client(pred))
    before = observation()
    d = policy.act(before)
    after = observation((.001, 0., 0.), missing=True)
    policy.observe_transition(before, d, after)
    assert policy.refresh_reason == "external_reference_unavailable"
    assert len(policy.local.summary()["inflated_regions"]) == 1


def test_payload_transport_needs_observed_co_motion_not_just_model_phase():
    policy = VisualFieldPolicyV4(Client(prediction(payload=True, phase="transport")))
    d = policy.act(observation(payload=True))
    assert not d.diagnostics["selection"]["accepted"]
    assert d.action == (0.,)*7
    assert d.diagnostics["selection"]["status"] == "payload_co_motion_required_use_direct_test"


def test_monitor_observes_payload_and_tool_without_wrist_tool_then_invalidates_on_open():
    monitor = RoleAwareVisualMonitor()
    p = prediction(payload=True)
    obs = observation(payload=True)
    monitor.configure(images(obs), p["visual_references"], obs.state)
    for n in range(1, 4):
        obs = observation((.001*n, 0., 0.), payload=True)
        monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert monitor.attachment_evidence
    monitor.observe(images(obs), obs.state, [0., 0., 0., 0., 0., 0., -1.])
    assert not monitor.attachment_evidence


def test_pure_static_image_does_not_gain_attachment_or_visual_correction():
    monitor = RoleAwareVisualMonitor()
    p = prediction(payload=True)
    for i in range(7):
        obs = observation((.001*i, 0., 0.), payload=True, static_feature=True)
        if i == 0:
            monitor.configure(images(obs), p["visual_references"], obs.state)
        else:
            monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert not monitor.attachment_evidence
    assert monitor.planar_correction(ORIGIN+[.02, 0., 0.]) is None


def test_attachment_survives_destination_change_but_not_payload_change():
    monitor = RoleAwareVisualMonitor()
    p = prediction(payload=True)
    obs = observation(payload=True)
    monitor.configure(images(obs), p["visual_references"], obs.state)
    for n in range(1, 4):
        obs = observation((.001*n, 0., 0.), payload=True)
        monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert monitor.attachment_evidence
    refs = copy.deepcopy(p["visual_references"])
    refs["external"].update(target_id="new_destination", moving_point_uv=[53/223, 90/223],
        tool_point_uv=[33/223, 70/223])
    monitor.configure(images(obs), refs, obs.state)
    assert monitor.attachment_evidence and monitor.attachment_samples == 3
    assert monitor.response.summary()["sample_count"] == 0  # New reference-relative mapping.
    refs["external"]["moving_id"] = "different_payload"
    monitor.configure(images(obs), refs, obs.state)
    assert not monitor.attachment_evidence


def test_opposing_tool_and_payload_image_motion_is_not_attachment():
    monitor = RoleAwareVisualMonitor()
    p = prediction(payload=True)
    obs = observation(payload=True)
    monitor.configure(images(obs), p["visual_references"], obs.state)
    for n in range(1, 4):
        obs = observation((.001*n, 0., 0.), payload=True)
        frames = images(obs)
        frames["external"] = frames["external"].copy()
        frames["external"][65:76, 25+n:36+n] = 40
        frames["external"][65:76, 25-n:36-n] = PATCHES[2]
        monitor.observe(frames, obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert not monitor.attachment_evidence


def test_validated_planar_response_can_propose_bounded_visual_goal_correction():
    monitor = RoleAwareVisualMonitor()
    p = prediction()
    p["visual_references"]["external"]["target_point_uv"] = [64/223, 96/223]
    obs = observation()
    monitor.configure(images(obs), p["visual_references"], obs.state)
    xy = np.zeros(2)
    for n in range(12):
        xy[n % 2] += .001
        obs = observation((*xy, 0.))
        monitor.observe(images(obs), obs.state, [.1, .1, 0., 0., 0., 0., 0.])
    assert monitor.response_ready and monitor.response.rank == 2
    goal = ORIGIN+[.02, .01, 0.]
    correction = monitor.planar_correction(goal)
    assert correction is not None
    assert 0 < np.linalg.norm(correction-goal) <= .00300001
    assert correction[2] == goal[2]
    # A genuine height-domain change prevents use of this planar correction.
    obs = observation((*xy, .006))
    monitor.observe(images(obs), obs.state, [0., 0., .1, 0., 0., 0., 0.])
    assert monitor.planar_correction(goal) is None


def test_action_must_be_acknowledged_and_hidden_state_is_not_an_input():
    policy = VisualFieldPolicyV4(Client(prediction()))
    obs = observation()
    d = policy.act(obs)
    with pytest.raises(RuntimeError):
        policy.act(obs)
    with pytest.raises(ValueError):
        policy.observe_transition(observation((.001, 0., 0.)), d, obs)
    policy.observe_transition(obs, d, obs)
