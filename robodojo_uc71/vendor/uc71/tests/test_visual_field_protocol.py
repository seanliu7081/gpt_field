"""Independent offline client/policy integration checks; no robot or live model."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.demonstration import load_demonstration
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import image_to_data_url, PolicyObservation
from gpt_field.visual_field_policy import VisualFieldPolicy
from gpt_field.visual_field_subscription import VisualFieldSubscriptionClient, shared_guidance
from test_demonstration import make_bundle
from test_field_history import FakeTransport, envelope


def observation(index=0, *, task="bowl_plate", xyz=(.1, .2, 1.)):
    pixels = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    image = image_to_data_url(pixels, size=32, rotate=False)
    return PolicyObservation(image, image, (*xyz, 0., 0., 0., .04, -.04),
        resolve_distance_task(task).prompt, (0., -.3, 0., -2., 0., 2., .8), (0.,)*7)


def prediction(*, field=False, phase="approach", actions=None):
    return {"controller": "visual_field" if field else "direct", "phase": phase,
        "target_id": "task_target", "hypothesis_id": "approach_1", "reason": "Visible target alignment.",
        "visual_references": {"external": None, "wrist": None},
        "field_intent": {"goal_offset_m": [.03, 0., 0.], "orientation_target_xyzw": [0, 0, 0, 1],
                         "gripper_command": "hold"} if field else None,
        "direct_actions": None if field else (actions if actions is not None else [[0., 0., 0., 0., 0., 0., 1.]]*5)}


def client(*responses, controller="hybrid", **kwargs):
    transport = FakeTransport(*(envelope(p, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"vf_{i}")
                                for i, p in enumerate(responses)))
    instance = VisualFieldSubscriptionClient(controller=controller, token_provider=lambda: "offline-oauth-fixture",
                                            transport=transport, **kwargs)
    return instance, transport


def input_texts(payload):
    return [json.loads(part["text"]) for msg in payload["input"] for part in msg.get("content", [])
            if part.get("type") == "input_text" and part.get("text", "").startswith("{")]


def test_mock_requests_keep_medium_subscription_allowed_q_and_image_history():
    instance, transport = client(*(prediction() for _ in range(4)))
    for i in range(4):
        instance.predict(observation(i).to_dict())
    for i, payload in enumerate(transport.calls):
        assert payload["model"] == "gpt-6-astra"
        assert payload["reasoning"] == {"effort": "medium"}
        assert payload["store"] is False
        assert sum(p.get("type") == "input_image" for m in payload["input"] for p in m.get("content", [])) == min(i+1, 2)*2
        record = [v for v in input_texts(payload) if v.get("type") == "observation"][-1]
        assert set(record) == {"type", "observation_index", "prompt", "observation/state",
                               "observation/joint_position", "observation/joint_velocity", "observable_context"}
        assert len(record["observation/state"]) == 8
        assert len(record["observation/joint_position"]) == len(record["observation/joint_velocity"]) == 7
        assert record["observable_context"] == {}
    budget = instance.budget_snapshot()
    assert budget["billing"] == "chatgpt_subscription" and budget["api_key_calls"] == 0
    assert budget["api_calls"] == 4


@pytest.mark.parametrize("extra", ["object_poses", "depth", "contacts", "success", "camera_calibration"])
def test_hidden_observation_fields_rejected_before_transport(extra):
    instance, transport = client(prediction())
    obs = observation().to_dict(); obs[extra] = "hidden-fixture"
    with pytest.raises(ValueError):
        instance.predict(obs)
    assert not transport.calls


def test_missing_joints_and_wrong_task_rejected_without_request():
    instance, transport = client(prediction())
    for obs in [replace(observation(), joint_position=None, joint_velocity=None), observation(task="drawer_bowl")]:
        with pytest.raises(ValueError):
            instance.predict(obs.to_dict())
    assert not transport.calls


@pytest.mark.parametrize("tamper", ["completion", "wrong_branch", "six_actions", "nan", "bool", "wrong_quaternion"])
def test_strict_response_contract(tamper):
    value = prediction(field=tamper == "wrong_quaternion")
    if tamper == "completion": value["task_complete"] = True
    elif tamper == "wrong_branch": value["field_intent"] = prediction(field=True)["field_intent"]
    elif tamper == "six_actions": value["direct_actions"].append([0.]*7)
    elif tamper == "nan": value["direct_actions"][0][0] = float("nan")
    elif tamper == "bool": value["direct_actions"][0][0] = True
    elif tamper == "wrong_quaternion": value["field_intent"]["orientation_target_xyzw"] = [0, 0, 0, 0]
    instance, _ = client()
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)


def test_direct_arm_rejects_field_and_both_share_guidance():
    hybrid, _ = client(); direct, _ = client(controller="direct")
    with pytest.raises(SchemaError):
        direct.validate_response(prediction(field=True), None)
    common = shared_guidance(resolve_distance_task("bowl_plate"), False)
    for instance in (hybrid, direct):
        assert common in instance.build_payload(observation().to_dict())["instructions"]
    with pytest.raises(ValueError):
        VisualFieldSubscriptionClient(AstraConfig(model="other-model"), token_provider=lambda: "fixture")


@pytest.mark.parametrize("reported", [{"model": "other-model"}, {"reasoning": {"effort": "high"}}])
def test_mismatched_provider_model_or_effort_rejected_no_retry(reported):
    instance, transport = client()
    transport.responses.append(envelope(prediction(), model="gpt-6-astra", reasoning={"effort": "medium"}))
    transport.responses[0].update(reported)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(transport.calls) == 1
    assert instance.history_stats()["observations"] == 0


def test_static_demo_once_outside_live_history_and_no_autoplay(tmp_path):
    path = tmp_path / "demo.json"; path.write_text(json.dumps(make_bundle(n=4)))
    demo = load_demonstration(path, "drawer_bowl")
    instance, transport = client(prediction(), prediction(), task_key="drawer_bowl", demonstration=demo)
    obs = observation(task="drawer_bowl")
    for _ in range(2): instance.predict(obs.to_dict())
    for payload in transport.calls:
        assert payload["input"][0] == demo.message()
        assert payload["input"].count(demo.message()) == 1
        assert "Never replay its coordinates/actions" in payload["instructions"]
    assert demo.message() not in instance.history_snapshot()


def test_real_policy_direct_executes_five_raw_actions_and_records_only_block_summary():
    actions = [[.1*i, -.1*i, .01, .02, -.01, 0., -1. if i < 2 else 1.] for i in range(5)]
    instance, transport = client(prediction(actions=actions), prediction(), controller="direct")
    policy = VisualFieldPolicy(instance)
    obs = observation()
    for i, action in enumerate(actions):
        decision = policy.act(obs)
        assert decision.action == tuple(action)
        assert decision.diagnostics["selection"] is None
        with pytest.raises(RuntimeError): policy.act(obs)
        after = replace(obs, state=(obs.state[0]+.001, *obs.state[1:]))
        policy.observe_transition(obs, decision, after); obs = after
        assert len(transport.calls) == 1
    next_decision = policy.act(obs)
    assert len(transport.calls) == 2
    records = input_texts(transport.calls[-1])
    blocks = [r for r in records if r.get("type") == "visual_execution_block"]
    assert len(blocks) == 1 and blocks[0]["steps"] == 5
    assert (blocks[0]["first_step"], blocks[0]["last_step"]) == (0, 4)
    assert blocks[0]["task_success_certified"] is False
    assert not any(r.get("type") == "observable_transition" for r in records)
    assert len(policy.attempts) == 1
    assert next_decision.diagnostics["refreshed"]


class NoAttachmentLocal:
    """Isolate the policy gate while treating local selection as successful."""
    def configure(self, *args, **kwargs):
        return {"tracks_reliable": True, "geometry_reliable": True, "co_motion_evidence": False}
    def observe(self, *args, **kwargs):
        return self.configure()
    def select_action(self, proposed, **kwargs):
        return {"action": list(proposed), "status": "visual_distance_descent", "need_refresh": False}


def test_repeated_transport_cannot_bypass_missing_attachment_evidence():
    instance, transport = client(prediction(phase="grasp"), prediction(field=True, phase="transport"),
                                  prediction(field=True, phase="transport"))
    policy = VisualFieldPolicy(instance); policy.local = NoAttachmentLocal()
    obs = observation()
    for _ in range(5):
        decision = policy.act(obs); policy.observe_transition(obs, decision, obs)
    for _ in range(2):
        decision = policy.act(obs)
        assert decision.action[:3] == (0., 0., 0.), "Repeated transport intent bypassed an unverified attachment"
        policy.observe_transition(obs, decision, obs)
    assert len(transport.calls) == 3


class ConstantCostLocal(NoAttachmentLocal):
    def configure(self, *args, **kwargs):
        return {**super().configure(), "views": {v: {"distance_cost_px": 40.} for v in ("external", "wrist")}}


def test_measured_oscillation_without_visual_descent_does_not_extend_field():
    instance, transport = client(prediction(field=True), prediction(field=True))
    policy = VisualFieldPolicy(instance); policy.local = ConstantCostLocal()
    obs = observation()
    for i in range(30):
        decision = policy.act(obs)
        after = replace(obs, state=(.101 if i % 2 == 0 else .1, *obs.state[1:]))
        policy.observe_transition(obs, decision, after); obs = after
    assert len(transport.calls) == 1
    next_decision = policy.act(obs)
    assert next_decision.diagnostics["refreshed"]
    assert len(transport.calls) == 2 and policy.budget == 30
    records = input_texts(transport.calls[-1])
    block = [v for v in records if v.get("type") == "visual_execution_block"][-1]
    assert block["measured_path_m"] > .02
    assert block["refresh_reason"] == "field_reuse_boundary"


def test_missing_references_hold_and_request_fresh_semantics():
    instance, transport = client(prediction(field=True), prediction())
    policy = VisualFieldPolicy(instance); obs = observation()
    decision = policy.act(obs)
    assert decision.action[:6] == (0.,)*6
    assert decision.diagnostics["selection"]["status"] == "visual_reference_unavailable"
    assert policy.refresh_reason
    policy.observe_transition(obs, decision, obs)
    assert policy.act(obs).diagnostics["refreshed"]
    assert len(transport.calls) == 2


@pytest.mark.parametrize("change_target_identity", [False, True])
def test_real_policy_learns_from_direct_feedback_and_retains_only_compatible_identity(change_target_identity):
    from test_visual_field import images as synthetic_images, seeds as synthetic_seeds, state as synthetic_state

    commands = [[.1, 0, 0], [0, .1, 0], [-.1, 0, 0], [0, -.1, 0], [.1, 0, 0],
                [0, .1, 0], [-.1, 0, 0], [0, -.1, 0], [.1, 0, 0], [0, .1, 0]]
    values = [prediction(actions=[[*a, 0., 0., 0., 1.] for a in commands[:5]]),
              prediction(actions=[[*a, 0., 0., 0., 1.] for a in commands[5:]]), prediction(field=True)]
    for value in values:
        value["visual_references"] = synthetic_seeds()
    if change_target_identity:
        for ref in values[-1]["visual_references"].values(): ref["target_id"] = "new_destination"
    instance, transport = client(*values)
    policy = VisualFieldPolicy(instance)
    q = np.array([0., 0., 0., -2., 0., 2., .8])

    def make_obs():
        frames = synthetic_images(q)
        return PolicyObservation(*(image_to_data_url(frames[v], size=96, rotate=False) for v in ("external", "wrist")),
            tuple(synthetic_state(q)), resolve_distance_task("bowl_plate").prompt, tuple(q), (0.,)*7)

    obs = make_obs()
    for command in commands:
        decision = policy.act(obs)
        assert list(decision.action[:3]) == command
        q[:3] += np.asarray(command)*.1
        after = make_obs(); policy.observe_transition(obs, decision, after); obs = after
    learned = policy.local.summary()
    assert learned["joint_to_image"]["reliable"] and learned["action_to_joint"]["reliable"]
    decision = policy.act(obs)
    configured = decision.diagnostics["feedback"]
    assert configured["joint_to_image"]["reliable"] is (not change_target_identity)
    assert configured["action_to_joint"]["reliable"] is (not change_target_identity)
    if not change_target_identity:
        assert decision.diagnostics["selection"]["status"] == "visual_distance_descent"
    assert len(transport.calls) == 3


class AttachmentLocal(NoAttachmentLocal):
    def __init__(self, transition):
        self.transition = transition
        self.configure_count = 0
        self.transition_observed = False

    def summary(self, *, evidence=False):
        view = {"moving_kind": "payload", "moving_id": "bowl_A", "target_id": "grasp_point",
                "reliable": True, "tool_track_reliable": True}
        if self.configure_count >= 2:
            view["target_id"] = "plate_goal"
            if self.transition == "lost_payload": view["reliable"] = False
            elif self.transition == "lost_tool": view["tool_track_reliable"] = False
            elif self.transition == "changed_payload": view["moving_id"] = "bowl_B"
            elif self.transition == "tool_instead": view["moving_kind"] = "tool"
            elif self.transition == "lose_after_observe" and self.transition_observed:
                view["tool_track_reliable"] = False
        return {"co_motion_evidence": evidence, "object_tool_relative_change_px": 0. if evidence else None,
                "tracks_reliable": view["reliable"], "geometry_reliable": True,
                "views": {v: copy.deepcopy(view) for v in ("external", "wrist")}}

    def configure(self, *args, **kwargs):
        self.configure_count += 1
        return self.summary()

    def observe(self, *args, **kwargs):
        self.transition_observed = self.configure_count >= 2
        return self.summary(evidence=self.configure_count == 1)


@pytest.mark.parametrize("transition", ["lost_payload", "lost_tool", "changed_payload", "tool_instead", "new_destination"])
def test_attachment_latch_is_bound_to_visible_payload_but_allows_new_destination(transition):
    instance, _ = client(prediction(phase="grasp"), prediction(field=True, phase="transport"))
    policy = VisualFieldPolicy(instance); policy.local = AttachmentLocal(transition)
    obs = observation()
    for _ in range(5):
        decision = policy.act(obs); policy.observe_transition(obs, decision, obs)
    assert policy.attachment_evidence
    decision = policy.act(obs)
    if transition == "new_destination":
        assert policy.attachment_evidence
        assert any(abs(x) > 0 for x in decision.action[:3])
    else:
        assert not policy.attachment_evidence
        assert policy.attachment_identity is None
        assert decision.action[:6] == (0.,)*6


def test_attachment_latch_clears_when_tool_track_is_lost_after_transport_action():
    instance, _ = client(prediction(phase="grasp"), prediction(field=True, phase="transport"))
    policy = VisualFieldPolicy(instance); policy.local = AttachmentLocal("lose_after_observe")
    obs = observation()
    for _ in range(5):
        decision = policy.act(obs); policy.observe_transition(obs, decision, obs)
    decision = policy.act(obs)
    assert any(abs(x) > 0 for x in decision.action[:3])
    policy.observe_transition(obs, decision, obs)
    assert not policy.attachment_evidence and policy.attachment_identity is None
    assert policy.act(obs).action[:6] == (0.,)*6
