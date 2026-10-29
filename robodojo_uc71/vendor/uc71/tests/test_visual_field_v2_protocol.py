"""V2 protocol/policy regression; synthetic RGB and fake subscription only."""
import copy
from dataclasses import replace
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_policy_v2 import VisualFieldPolicyV2
from gpt_field.visual_field_subscription import VisualFieldSubscriptionClient, output_schema as schema_v1
from gpt_field.visual_field_subscription_v2 import (
    VisualFieldSubscriptionClientV2, VERSION, output_schema, shared_guidance,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, prediction, input_texts


def client(*responses, **kwargs):
    transport = FakeTransport(*(envelope(value, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v2_{i}")
                                for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV2(token_provider=lambda: "offline-oauth-fixture",
                                          transport=transport, **kwargs), transport


def seed():
    return {"moving_id": "payload_spot", "target_id": "target_rim_spot", "moving_kind": "payload",
            "tool_point_uv": [.25, .4], "moving_point_uv": [.3, .4],
            "target_anchor_uv": [.7, .6], "target_point_uv": [.6, .5],
            "moving_radius_uv": .01, "blocked_boxes_uv": [], "unknown_boxes_uv": []}


def assert_unique_required(value):
    if isinstance(value, dict):
        if "required" in value:
            assert len(value["required"]) == len(set(value["required"]))
        for child in value.values():
            assert_unique_required(child)
    elif isinstance(value, list):
        for child in value:
            assert_unique_required(child)


def test_v2_anchor_schema_is_unique_required_and_v1_unchanged():
    assert_unique_required(output_schema())
    value = prediction()
    value["visual_references"] = {v: seed() for v in ("external", "wrist")}
    instance, _ = client()
    assert instance.validate_response(value, None) == value
    for view in ("external", "wrist"):
        broken = copy.deepcopy(value)
        del broken["visual_references"][view]["target_anchor_uv"]
        with pytest.raises(SchemaError):
            instance.validate_response(broken, None)
    assert "target_anchor_uv" not in json.dumps(schema_v1())
    legacy = VisualFieldSubscriptionClient(token_provider=lambda: "offline-fixture")
    with pytest.raises(SchemaError):
        legacy.validate_response(value, None)


@pytest.mark.parametrize("kwargs", [{"controller": "direct"}, {"demonstration": object()}])
def test_v2_rejects_new_direct_arm_and_demonstrations(kwargs):
    with pytest.raises(ValueError, match="Hybrid only"):
        client(**kwargs)


@pytest.mark.parametrize("extra", ["task_complete", "libero_success", "depth", "object_poses"])
def test_output_and_input_privacy_reject_unrequested_fields(extra):
    instance, transport = client(prediction())
    value = prediction(); value[extra] = True
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)
    obs = observation().to_dict(); obs[extra] = True
    with pytest.raises(ValueError):
        instance.predict(obs)
    instance.observable_context = {"visual_feedback": {extra: True}}
    with pytest.raises(ValueError):
        instance.predict(observation().to_dict())
    assert not transport.calls


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_prompt_medium_oauth_anchor_probe_recovery_and_no_success_flag(task):
    instance, transport = client(prediction(), task_key=task)
    instance.predict(observation(task=task).to_dict())
    payload = transport.calls[0]
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["model"] == "gpt-6-astra"
    assert payload["text"]["format"]["name"] == "visual_field_rgb_joints_v2"
    assert instance.prompt_version == VERSION + "-hybrid"
    text = payload["instructions"]
    assert shared_guidance(resolve_distance_task(task)) in text
    assert "real, currently visible, textured landmark" in text
    assert "virtual location" in text and "no automatic upward recovery" in text
    assert "stable orientation and gripper" in text and "nonzero information-gathering" in text
    assert "never output a task-completion flag" in text
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert sum(p["type"] == "input_image" for m in payload["input"] for p in m.get("content", [])) == 2


class Local:
    def __init__(self, *, reliable=True, payload=False, attachment=False):
        self.reliable, self.payload, self.attachment = reliable, payload, attachment
        self.tool_uv = [.2, .4]
        self.observations = 0

    def summary(self):
        view = {"status": "observed" if self.reliable else "track_lost", "reliable": self.reliable,
                "geometry_reliable": self.reliable, "moving_id": "payload_A", "target_id": "goal_A",
                "moving_kind": "payload" if self.payload else "tool",
                "moving_track_reliable": self.reliable, "target_anchor_track_reliable": self.reliable,
                "tool_track_reliable": self.reliable, "moving_point_uv": [.3, .4],
                "target_point_uv": [.3, .4], "target_anchor_uv": [.4, .5],
                "tool_point_uv": self.tool_uv, "image_scale_px": [100., 100.],
                "goal_error_px": 0., "distance_cost_px": 10.}
        return {"tracks_reliable": self.reliable, "geometry_reliable": self.reliable,
                "co_motion_evidence": self.attachment, "object_tool_relative_change_px": 0.,
                "views": {v: copy.deepcopy(view) for v in ("external", "wrist")},
                "recovery": {"missing_views": [] if self.reliable else ["external", "wrist"],
                             "suggested_action": "request same-identity image seeds and bounded Direct probe"}}

    def configure(self, *args, **kwargs):
        return self.summary()

    def observe(self, *args, **kwargs):
        self.observations += 1
        return self.summary()

    def select_action(self, proposed, **kwargs):
        return {"action": list(proposed), "status": "visual_distance_descent", "need_refresh": False}


def step(policy, obs, after=None):
    decision = policy.act(obs)
    after = obs if after is None else after
    row = policy.observe_transition(obs, decision, after)
    return decision, row


def zero_prediction(hypothesis="same_attempt"):
    value = prediction(phase="verify", actions=[[0.]*7 for _ in range(5)])
    value["hypothesis_id"] = hypothesis
    value["reason"] = "I think finished; hold."
    return value


def test_repeated_zero_blocks_request_changed_measurement_even_when_hypothesis_renamed():
    instance, transport = client(*(zero_prediction(str(i)) for i in range(5)))
    policy = VisualFieldPolicyV2(instance); policy.local = Local()
    obs = observation()
    for _ in range(10):
        decision, row = step(policy, obs)
        assert decision.action == (0.,)*7
        assert row["unexecuted_zero_actions"] == 0
    decision, row = step(policy, obs)
    context = input_texts(transport.calls[-1])[-1]["observable_context"]
    assert context["task_stagnation"]["active"]
    assert context["task_stagnation"]["zero_progress_blocks"] == 2
    assert "nonzero" in context["task_stagnation"]["next_action_requirement"]
    assert row["unexecuted_zero_actions"] == 4
    assert policy.refresh_reason == "persistent_zero_chunk_requires_informative_observation"
    step(policy, obs)  # A repeated zero response still runs: no implicit episode stop.
    assert policy.step_index == 12
    assert len(transport.calls) == 4
    assert not any(event.get("episode_stop") for event in policy.events)
    assert policy.placement_evidence["state"] == "uncertain"


def test_real_measured_progress_clears_stagnation_without_certifying_task():
    moving = prediction(actions=[[.1, 0., 0., 0., 0., 0., 1.]]*5)
    instance, _ = client(zero_prediction(), zero_prediction(), moving)
    policy = VisualFieldPolicyV2(instance); policy.local = Local()
    obs = observation()
    for _ in range(10):
        step(policy, obs)
    after = replace(obs, state=(obs.state[0]+.004, *obs.state[1:]))
    decision, _ = step(policy, obs, after)
    assert decision.action == tuple(moving["direct_actions"][0])
    assert not policy.task_stagnation["active"]
    assert policy.task_stagnation["evidence"]["task_progress_certified"] is False
    assert policy.placement_evidence["task_success_certified"] is False


def test_missing_tracking_requests_concrete_recovery_without_modifying_moving_direct5():
    actions = [[.02, 0., 0., 0., 0., 0., 1.]]*5
    instance, transport = client(*(prediction(actions=actions) for _ in range(4)))
    policy = VisualFieldPolicyV2(instance); policy.local = Local(reliable=False)
    obs = observation()
    for i in range(11):
        decision, row = step(policy, obs)
        assert decision.action == tuple(actions[i % 5])
        assert row["unexecuted_zero_actions"] == 0
    context = input_texts(transport.calls[-1])[-1]["observable_context"]
    assert context["task_stagnation"]["active"]
    assert context["visual_feedback"]["recovery"]["missing_views"] == ["external", "wrist"]
    assert len(transport.calls) == 3


@pytest.mark.parametrize("actions", [
    [[0., 0., 0., 0., 0., 0., 1.]]*5,
    [[0.]*7, [.01, 0., 0., 0., 0., 0., 0.], [0.]*7, [0.]*7, [0.]*7],
])
def test_only_entirely_zero_chunk_can_be_shortened(actions):
    instance, _ = client(prediction(actions=actions))
    policy = VisualFieldPolicyV2(instance); policy.local = Local()
    policy._activate("fixture_stall", {"measured": True})
    for action in actions:
        decision, row = step(policy, observation())
        assert decision.action == tuple(action)
        assert row["unexecuted_zero_actions"] == 0
    assert policy.step_index == 5


@pytest.mark.parametrize("phase", ["transport", "align"])
@pytest.mark.parametrize("payload", [False, True])
def test_field_payload_motion_requires_co_motion_and_payload_identity(phase, payload):
    value = prediction(field=True, phase=phase)
    value["field_intent"]["gripper_command"] = "close"
    instance, transport = client(value, value)
    policy = VisualFieldPolicyV2(instance); policy.local = Local(payload=payload)
    for _ in range(2):
        decision, _ = step(policy, observation())
        assert decision.action[:6] == (0.,)*6
        assert decision.diagnostics["attachment_gate"]
    assert len(transport.calls) == 2
    assert "test_motion" in policy.refresh_reason
    assert policy.diagnostic_counters["attachment_gate_actions"] == 2


def test_observed_attachment_survives_destination_refresh_and_allows_field_payload_alignment():
    value = prediction(field=True, phase="align")
    value["field_intent"]["gripper_command"] = "close"
    instance, _ = client(prediction(phase="grasp"), value)
    policy = VisualFieldPolicyV2(instance); policy.local = Local(payload=True, attachment=True)
    obs = observation()
    for _ in range(5):
        step(policy, obs)
    assert policy.attachment_evidence
    decision = policy.act(obs)
    assert any(abs(x) > 0 for x in decision.action[:3])
    assert decision.diagnostics["attachment_gate"] is None


def test_placement_requires_post_open_tracked_separation_and_time_not_labels_or_holds():
    value = prediction(phase="release", actions=[[0., 0., 0., 0., 0., 0., -1.]]*5)
    instance, _ = client(value, value)
    policy = VisualFieldPolicyV2(instance); local = Local(payload=True); policy.local = local
    obs = observation()
    step(policy, obs)
    assert policy.placement_evidence["state"] == "uncertain"
    local.tool_uv = [.16, .4]  # Actual tracked tool separation; payload/target stay aligned.
    for _ in range(3):
        step(policy, obs)
    assert policy.placement_evidence["state"] == "observed"
    assert policy.placement_evidence["stable_observations"] == 3
    assert not policy.placement_evidence["support_3d_verified"]
    assert not policy.placement_evidence["task_success_certified"]
    local.reliable = False
    step(policy, obs)
    assert policy.placement_evidence["state"] == "uncertain"


def test_bounded_history_preserves_recent_complete_turns_and_aggregates_older_feedback(tmp_path):
    instance, transport = client(*(prediction() for _ in range(12)), log_dir=tmp_path)
    for i in range(12):
        instance.predict(observation(i).to_dict())
        instance.record_block({"type": "visual_execution_block", "chunk_index": i, "steps": 5,
            "last_step": 5*i+4, "phase": "grasp", "target_id": "bowl", "hypothesis_id": "h1",
            "controller": "direct", "after_state": list(observation().state), "refresh_reason": "direct_chunk_complete",
            "measured_path_m": .01, "commanded_path_m": .025,
            "task_stagnation": {"active": False}, "placement_evidence": {"state": "uncertain"},
            "task_success_certified": False})
    latest = transport.calls[-1]
    records = input_texts(latest)
    observations = [r for r in records if r["type"] == "observation"]
    assert [r["observation_index"] for r in observations] == [10, 11]
    memories = [r for r in records if r["type"] == "visual_history_summary"]
    assert len(memories) == 1
    memory = memories[0]
    assert memory["observations_summarized"] == memory["execution_blocks_summarized"] == 10
    assert memory["executed_actions_summarized"] == 50
    assert memory["measured_path_m"] == pytest.approx(.1)
    assert len([r for r in records if r["type"] == "visual_execution_block"]) == 1
    assert sum(p.get("type") == "input_image" for m in latest["input"] for p in m.get("content", [])) == 4
    assert len(list(tmp_path.glob("*.request.json"))) == 12
    assert len(list(tmp_path.glob("*.result.json"))) == 12
    assert instance.history_stats()["observations"] == 12
    assert instance.history_stats()["raw_logs"] == "retained_unmodified"


def test_real_controller_missing_anchor_cannot_bootstrap_blindly():
    instance, transport = client(prediction(field=True), prediction())
    policy = VisualFieldPolicyV2(instance)
    decision, _ = step(policy, observation())
    assert decision.action[:6] == (0.,)*6
    assert decision.diagnostics["selection"]["status"] == "visual_reference_unavailable"
    next_decision = policy.act(observation())
    assert next_decision.diagnostics["refreshed"]
    assert len(transport.calls) == 2


def test_regrasp_invalidates_release_baseline_before_a_new_release_cycle():
    from gpt_field.policy import Decision
    instance, _ = client()
    policy = VisualFieldPolicyV2(instance)
    local = Local(payload=True)
    policy.feedback = local.summary()
    obs = observation()
    opened = Decision((0., 0., 0., 0., 0., 0., -1.), {})
    baseline = local.summary()
    policy._placement_observe(baseline, obs, opened, obs)
    local.tool_uv = [.16, .4]
    policy.feedback = local.summary()
    for _ in range(3):
        policy._placement_observe(policy.feedback, obs, opened, obs)
    assert policy.placement_evidence["state"] == "observed"
    policy._placement_observe(policy.feedback, obs, Decision((0., 0., 0., 0., 0., 0., 1.), {}), obs)
    assert policy._release_reference is None and policy._placement_samples == 0
    policy._placement_observe(policy.feedback, obs, opened, obs)
    assert policy.placement_evidence["state"] == "uncertain"
    assert not policy.placement_evidence["observed_gripper_separation"]


@pytest.mark.parametrize("reseed", [False, True])
def test_attachment_epoch_invalidates_same_named_payload_reseed_but_not_new_goal(reseed):
    class EpochLocal(Local):
        def __init__(self):
            super().__init__(payload=True, attachment=True)
            self.configured = 0

        def configure(self, *args, **kwargs):
            self.configured += 1
            return self.summary()

        def summary(self):
            value = super().summary()
            value["attachment_epoch"] = 2 if self.configured > 1 and reseed else 1
            if self.configured > 1:
                value["co_motion_evidence"] = False
                for view in value["views"].values():
                    view["target_id"] = "new_destination_anchor"
            return value

    value = prediction(field=True, phase="transport")
    value["field_intent"]["gripper_command"] = "close"
    instance, _ = client(prediction(phase="grasp"), value)
    policy = VisualFieldPolicyV2(instance); policy.local = EpochLocal()
    obs = observation()
    for _ in range(5):
        step(policy, obs)
    assert policy.attachment_evidence
    decision = policy.act(obs)
    assert policy.attachment_evidence is (not reseed)
    assert (decision.action[:6] == (0.,)*6) is reseed
