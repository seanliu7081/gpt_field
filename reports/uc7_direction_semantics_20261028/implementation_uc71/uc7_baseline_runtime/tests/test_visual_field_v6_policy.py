"""V6 feedback/action integration with real local planners and RGB trackers.

Synthetic image motion is a test plant, never a hidden policy input. No live
model, simulator poses, evaluator flags or predeclared attachment is supplied.
"""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.visual_field_policy_v6 import VisualFieldPolicyV6
from test_visual_field_v4_policy import Q, ORIGIN
from test_visual_field_v6_protocol import client as subscription_client, prediction as wire_prediction
from test_visual_field_protocol import observation as wire_observation


TEXTURES = [np.random.default_rng(501+i).integers(0, 256, (15, 15, 3), dtype=np.uint8) for i in range(3)]
MOVING, TARGET, TOOL = np.array([60, 75]), np.array([165, 150]), np.array([35, 50])


def projection(position):
    x, y, z = position
    return np.rint([1000*x, 1000*(.5*y-z)]).astype(int)


def observation(position=(0., 0., 0.), *, payload=False, fingers=(.04, -.04), missing=False, slip=False):
    shift = projection(position)
    im = np.full((224, 224, 3), 40, dtype=np.uint8)
    moving_shift = np.zeros(2, dtype=int) if slip else shift
    points = [MOVING+moving_shift, TARGET, TOOL+shift]
    for index, point in enumerate(points):
        if (index == 0 and missing) or (index == 2 and not payload):
            continue
        x, y = point
        if 8 <= x < 216 and 8 <= y < 216:
            im[y-7:y+8, x-7:x+8] = TEXTURES[index]
    wrist = np.full_like(im, 70)
    return PolicyObservation(image_to_data_url(im, rotate=False), image_to_data_url(wrist, rotate=False),
        (*tuple(ORIGIN+np.asarray(position)), 0., 0., 0., *fingers),
        "move the object to its target", Q, (0.,)*7)


def prediction(controller="visual_field", *, phase="approach", payload=False, retain=False,
               goal=(.06, 0., 0.), motion_axes="xyz", actions=None, gripper="close"):
    value = wire_prediction(controller=controller, phase=phase, retain=retain)
    value["target_id"] = "destination"
    ext = value["visual_references"]["external"]
    ext.update(moving_id="object" if payload else "tool", moving_kind="payload" if payload else "tool",
               target_id="destination", goal_offset_uv=((MOVING+projection(goal)-TARGET)/223).tolist())
    if ext["seed"] is not None:
        ext["seed"].update(moving_point_uv=(MOVING/223).tolist(), moving_roi_uv=None,
            target_anchor_uv=(TARGET/223).tolist(), target_roi_uv=None,
            tool_point_uv=(TOOL/223).tolist() if payload else None, tool_roi_uv=None)
    if controller == "visual_field":
        intent = value["field_intent"]
        intent.pop("orientation_command")
        intent.update(orientation_target_xyzw=[0., 0., 0., 1.], motion_axes=motion_axes,
                      goal_offset_m=list(goal), max_actions=80, command_limit_m=.012)
    elif controller == "direct":
        value["direct_actions"] = copy.deepcopy(actions or [[.08, 0., 0., 0., 0., 0., 0.]]*5)
    else:
        value["gripper_intent"] = {"command": gripper, "max_actions": 25}
    return value


class Client:
    last_request_key = "offline-policy-fixture"

    def __init__(self, value):
        self.value, self.calls, self.blocks = value, 0, []

    def reset_history(self):
        pass

    def predict(self, obs):
        self.calls += 1
        return copy.deepcopy(self.value)

    def record_block(self, block):
        self.blocks.append(copy.deepcopy(block))


def test_one_explicit_gripper_intent_executes_until_measured_settling_without_more_requests():
    value = prediction("gripper", phase="grasp")
    value["visual_references"] = {"external": None, "wrist": None}
    client = Client(value)
    policy = VisualFieldPolicyV6(client)
    obs = observation()
    decisions = []
    for index in range(20):
        decision = policy.act(obs)
        decisions.append(decision)
        assert decision.action == (0., 0., 0., 0., 0., 0., 1.)
        assert decision.diagnostics["selection"]["accepted"]
        # Nine changing samples and then three settled measured transitions.
        aperture = max(.01, .04-.003*(index+1))
        after = replace(obs, state=(*obs.state[:6], aperture, -aperture))
        row = policy.observe_transition(obs, decision, after)
        obs = after
        if policy.refresh_reason:
            break
    assert 5 < len(decisions) <= 25 and client.calls == 1
    assert policy.refresh_reason == "gripper_settled"
    assert policy.gripper_actions == len(decisions)
    assert row["gripper_execution"]["consecutive_stable_finger_changes"] == 3
    assert len(policy.attempts) == 1
    assert not policy.monitor.attachment_evidence
    assert not row["gripper_execution"]["grasp_certified"]
    policy.finish()
    assert client.blocks[0]["controller"] == "gripper"
    assert client.blocks[0]["steps"] == len(decisions)


def test_direct_five_actions_are_unchanged_and_do_not_auto_start_gripper_settling():
    actions = [[.01*i, -.02, .01, .0, .01*i, .0, 1.] for i in range(5)]
    value = prediction("direct", phase="grasp", actions=actions)
    value["visual_references"] = {"external": None, "wrist": None}
    client = Client(value)
    policy = VisualFieldPolicyV6(client)
    obs = observation()
    for action in actions:
        decision = policy.act(obs)
        assert list(decision.action) == action
        policy.observe_transition(obs, decision, obs)
    assert client.calls == 1 and policy.gripper_actions == 0
    assert policy.gripper.intent is None
    policy.act(obs)
    assert client.calls == 2


def test_vertical_field_release_reuses_motion_and_never_auto_opens_or_translates_laterally():
    value = prediction(phase="release", goal=(0., 0., -.045), motion_axes="z")
    client = Client(value)
    policy = VisualFieldPolicyV6(client)
    offset = np.zeros(3)
    obs = observation(offset)
    for _ in range(35):
        decision = policy.act(obs)
        assert decision.diagnostics["controller"] == "visual_field"
        if not decision.diagnostics["selection"]["accepted"]:
            assert decision.diagnostics["selection"]["status"] == "measured_local_goal_arrival"
            assert decision.action == (0.,)*7
            row = policy.observe_transition(obs, decision, obs)
            break
        assert np.linalg.norm(decision.action[:2]) < 1e-8
        assert decision.action[2] < 0 and decision.action[3:] == (0.,)*4
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset)
        row = policy.observe_transition(obs, decision, after)
        obs = after
        if policy.refresh_reason:
            break
    assert client.calls == 1 and policy.field_effective_actions > 5
    assert offset[2] < -.015 and np.linalg.norm(offset[:2]) < 1e-8
    assert policy.attempts == [] and policy.gripper_actions == 0
    assert not policy._allows_planar_correction()
    assert row["last_field_handoff"]["motion_axes"] == "z"


def run_observed_test_lift(policy, *, slip=False):
    obs = observation(payload=True, fingers=(.01, -.01))
    for index in range(1, 6):
        decision = policy.act(obs)
        after = observation((.001*index, 0., 0.), payload=True, fingers=(.01, -.01), slip=slip)
        policy.observe_transition(obs, decision, after)
        obs = after
    return obs


def test_observed_direct_test_lift_then_retained_payload_field_uses_existing_evidence():
    direct = prediction("direct", phase="grasp", payload=True)
    direct["target_id"] = "object"  # External destination is an independent stationary anchor.
    client = Client(direct)
    policy = VisualFieldPolicyV6(client)
    obs = run_observed_test_lift(policy)
    assert policy.monitor.attachment_evidence
    moving = policy.monitor.tracks["moving"]
    tool = policy.monitor.tracks["tool"]
    samples = policy.monitor.attachment_samples
    client.value = prediction(phase="transport", payload=True, retain=True)
    decision = policy.act(obs)
    assert decision.diagnostics["selection"]["accepted"], decision.diagnostics["selection"]
    assert policy.monitor.tracks["moving"] is moving and policy.monitor.tracks["tool"] is tool
    assert policy.monitor.attachment_evidence and policy.monitor.attachment_samples == samples
    assert policy.monitor.reference_update_status == "external_retained"
    assert client.calls == 2 and decision.action[6] == 0.


def test_payload_vertical_release_reuses_observed_attachment_and_preserves_horizontal_pose():
    direct = prediction("direct", phase="grasp", payload=True)
    direct["target_id"] = "object"
    client = Client(direct)
    policy = VisualFieldPolicyV6(client)
    obs = run_observed_test_lift(policy)
    release = prediction(phase="release", payload=True, retain=True,
                         goal=(0., 0., -.04), motion_axes="z")
    release["visual_references"]["external"]["goal_offset_uv"] = (
        (MOVING+projection((.005, 0., -.04))-TARGET)/223).tolist()
    client.value = release
    offset = np.array([.005, 0., 0.])
    count = 0
    for _ in range(25):
        decision = policy.act(obs)
        selected = decision.diagnostics["selection"]
        if not selected["accepted"]:
            assert selected["status"] == "measured_local_goal_arrival"
            policy.observe_transition(obs, decision, obs)
            break
        assert policy.monitor.attachment_evidence
        assert decision.action[3:] == (0.,)*4
        assert np.linalg.norm(decision.action[:2]) < 1e-8
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset, payload=True, fingers=(.01, -.01))
        policy.observe_transition(obs, decision, after)
        obs, count = after, count+1
        if policy.refresh_reason:
            break
    assert count > 5 and client.calls == 2
    assert offset[:2] == pytest.approx([.005, 0.], abs=1e-8)
    assert offset[2] < -.02
    assert policy.gripper_actions == 0


@pytest.mark.parametrize("failure", ["no_motion", "slip", "wrong_identity", "lost_track"])
def test_payload_transport_cannot_invent_attachment_or_revive_invalid_retained_tracks(failure):
    direct = prediction("direct", phase="grasp", payload=True)
    direct["target_id"] = "object"
    client = Client(direct)
    policy = VisualFieldPolicyV6(client)
    if failure == "no_motion":
        obs = observation(payload=True)
        for _ in range(5):
            decision = policy.act(obs)
            policy.observe_transition(obs, decision, obs)
    else:
        obs = run_observed_test_lift(policy, slip=failure == "slip")
    if failure in ("wrong_identity", "lost_track"):
        assert policy.monitor.attachment_evidence
    value = prediction(phase="transport", payload=True, retain=True)
    if failure == "wrong_identity": value["visual_references"]["external"]["moving_id"] = "another_object"
    if failure == "lost_track": policy.monitor.tracks["moving"]._fail("deliberate_observed_occlusion")
    client.value = value
    decision = policy.act(obs)
    assert not decision.diagnostics["selection"]["accepted"]
    assert decision.action == (0.,)*7
    assert not policy.monitor.attachment_evidence
    assert client.calls == 2


def test_gripper_pose_drift_refreshes_without_changing_the_requested_target_or_claiming_grasp():
    client = Client(prediction("gripper", phase="grasp"))
    policy = VisualFieldPolicyV6(client)
    before = observation()
    decision = policy.act(before)
    after = observation((.006, 0., 0.))
    row = policy.observe_transition(before, decision, after)
    assert policy.refresh_reason == "gripper_pose_drift"
    assert row["gripper_execution"]["measured_pose_drift_m"] == pytest.approx(.006)
    assert client.calls == 1 and not policy.monitor.attachment_evidence


def test_real_subscription_repair_counts_both_requests_and_entire_predict_latency(monkeypatch):
    import gpt_field.visual_field_policy_v6 as policy_module
    invalid, valid = wire_prediction(), wire_prediction()
    invalid["visual_references"]["external"]["update"] = "retain"  # Still includes seed data.
    client, transport = subscription_client(invalid, valid)
    clock = [10.]
    def timed_transport(*args, **kwargs):
        clock[0] += 1.25
        return transport(*args, **kwargs)
    client.transport = timed_transport
    monkeypatch.setattr(policy_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    policy = VisualFieldPolicyV6(client)
    decision = policy.act(wire_observation())
    assert len(transport.calls) == 2
    assert policy.model_requests == 2 and policy.intent_refreshes == 1
    assert decision.diagnostics["model_pipeline_s"] == pytest.approx(2.5)
    assert decision.diagnostics["feedback"]["efficiency"]["model_requests"] == 2
    assert client.history_stats()["semantic_repair_requests"] == 1
