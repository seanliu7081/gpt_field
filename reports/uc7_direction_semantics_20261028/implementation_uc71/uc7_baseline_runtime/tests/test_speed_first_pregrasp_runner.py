"""Offline policy contract tests; no model service or robot simulator is used."""
import copy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.speed_first_pregrasp import (PREGRASP_VERSION, pregrasp_instructions,
    pregrasp_reference_schema)
from gpt_field.speed_first_v3 import (ChatGPTPlanSpeedFirstV3Client, SpeedFirstV3Policy,
    FINE_GRID, speed_first_v3_instructions, speed_first_v3_schema)
from test_field_history import FakeTransport, envelope, observation
from test_speed_first_v2 import prediction, operation, tick, records


def reference(identifier="contact-1", target="plate"):
    view = dict(tool_contact_uv=[.25, .4], target_contact_uv=[.55, .4],
        tool_roi_uv=[.15, .3, .35, .5], target_roi_uv=[.45, .3, .65, .5],
        desired_offset_uv=[-.02, 0.], tool_reference_mode="tracked_material")
    return dict(reference_id=identifier, target_id=target,
                external=copy.deepcopy(view), wrist=copy.deepcopy(view))


def enabled(raw, ref=None):
    raw = copy.deepcopy(raw)
    raw["control_context"]["pregrasp_reference"] = ref
    return raw


def alignment(*, direct=False, goal=(.03, 0., 0.), ref=None):
    raw = prediction(phase="align", direct=direct, goal=goal)
    if direct:
        raw["direct"]["actions"] = [[.01 * (i + 1), 0., .01, 0., 0., 0., -1.] for i in range(5)]
    else:
        raw["field"]["gripper_command"] = "open"
    return enabled(raw, reference() if ref is None else ref)


def controller(*responses, active=True):
    transport = FakeTransport(*(envelope(raw, id=f"pregrasp-{i}", model="gpt-6-astra",
        reasoning={"effort": "medium"}) for i, raw in enumerate(responses)))
    client = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=active,
        token_provider=lambda: "offline-only", transport=transport)
    return SpeedFirstV3Policy(client), transport


def pregrasp_records(messages):
    return [row for row in records(messages) if row.get("type") == "pregrasp_visual_feedback"]


def test_disabled_payload_exact_and_enabled_schema_is_only_opt_in():
    base = ChatGPTPlanSpeedFirstV3Client(token_provider=lambda: pytest.fail("credentials requested"))
    disabled = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=False,
        token_provider=lambda: pytest.fail("credentials requested"))
    active = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=True,
        token_provider=lambda: pytest.fail("credentials requested"))
    original = base.build_payload(observation().to_dict())
    assert disabled.build_payload(observation().to_dict()) == original
    changed = active.build_payload(observation().to_dict())
    assert active.prompt_version == base.prompt_version + "+" + PREGRASP_VERSION
    assert changed["instructions"] == original["instructions"] + "\n\n" + pregrasp_instructions()
    changed["instructions"] = original["instructions"]
    context = changed["text"]["format"]["schema"]["properties"]["control_context"]
    assert context["properties"].pop("pregrasp_reference") == pregrasp_reference_schema()
    context["required"].remove("pregrasp_reference")
    assert changed == original


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_flag_is_boolean(value):
    for factory in (ChatGPTPlanSpeedFirstV3Client, speed_first_v3_schema, speed_first_v3_instructions):
        with pytest.raises(ValueError, match="pregrasp_feedback"):
            factory(pregrasp_feedback=value)


def test_validation_is_atomic_and_raw_reference_is_retained():
    valid = alignment(direct=True)
    malformed = copy.deepcopy(valid)
    malformed["control_context"]["pregrasp_reference"]["external"]["tool_contact_uv"] = [9., 9.]
    bad_action = copy.deepcopy(valid)
    bad_action["direct"]["actions"][0][0] = 10.
    fake = FakeTransport(envelope(valid), envelope(malformed), envelope(bad_action), APIError("offline failure"))
    client = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=True,
        token_provider=lambda: "offline-only", transport=fake)
    result = client.predict(observation().to_dict())
    assert "control_context" not in result
    assert client.last_control_context == {"phase": "align", "visual_reference": None}
    assert client.last_pregrasp_reference == reference()
    assert client.last_compact_prediction == valid
    history = client.history_snapshot()
    count = client._observation_count
    for invalid in (malformed, bad_action):
        with pytest.raises(SchemaError):
            client.validate_response(invalid, FINE_GRID)
        assert client.last_pregrasp_reference == reference()
        assert client.last_compact_prediction == valid
    for error in (SchemaError, SchemaError, APIError):
        with pytest.raises(error):
            client.predict(observation().to_dict())
        assert client.history_snapshot() == history
        assert client._observation_count == count
        assert client.last_pregrasp_reference is None
        assert client.last_compact_prediction is None
    assert any(row.get("control_context", {}).get("pregrasp_reference") == reference()
               for row in records(history))


@pytest.mark.parametrize("mutate", [
    lambda r: r["control_context"].update(phase="grasp"),
    lambda r: r["control_context"].update(visual_reference={"not": "a valid simultaneous reference"}),
    lambda r: r["control_context"]["pregrasp_reference"].update(target_id="black_bowl"),
    lambda r: r["field"].update(gripper_command="close"),
    lambda r: r["field"].update(gripper_command="hold"),
])
def test_nonnull_reference_requires_align_same_target_and_explicit_open_field(mutate):
    raw = alignment()
    mutate(raw)
    client = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=True, token_provider=lambda: "offline-only")
    with pytest.raises(SchemaError):
        client.validate_response(raw, FINE_GRID)
    assert client.history_snapshot() == []


def test_reference_never_changes_a_close_operation_or_closed_direct_into_alignment():
    client = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=True, token_provider=lambda: "offline-only")
    close = enabled(operation(), reference())
    close["control_context"]["phase"] = "align"
    with pytest.raises(SchemaError, match="Opening/closing"):
        client.validate_response(close, FINE_GRID)
    raw = alignment(direct=True)
    raw["direct"]["actions"][0][6] = 1.
    with pytest.raises(SchemaError, match="retain open"):
        client.validate_response(raw, FINE_GRID)


def test_enabled_null_preserves_field_direct_and_local_gripper_actions():
    raw = [prediction(), prediction(direct=True, phase="verify"), operation()]
    policies = [controller(*raw, active=False), controller(*(enabled(r) for r in raw))]
    trajectories = []
    for policy, transport in policies:
        obs = observation()
        trajectory = []
        for i in range(21):
            obs, decision, _ = tick(policy, obs, fingers=(.015, -.015) if i >= 6 else None)
            trajectory.append(decision.action)
        assert len(transport.calls) == 3
        assert policy.gripper_operation.actions_executed == 15
        assert policy._completion is None
        assert pregrasp_records(policy.client.history_snapshot()) == []
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1]
    assert policies[1][0].pregrasp_stats()["applied_goals"] == 0
    for call in policies[1][1].calls:
        assert len(pregrasp_records(call["input"])) == 1
    assert all(not pregrasp_records(call["input"]) for call in policies[0][1].calls)


class FeedbackFixture:
    """Controlled geometry estimates isolate policy integration from tracker tests."""
    maximum_orientation_change_rad = .035

    def __init__(self, correction=(0., 0., .004)):
        self.correction = np.asarray(correction)
        self.valid = True
        self.error = 12.
        self.reference_id = None
        self.observed = self.invalidated = 0
        self.raise_on_update = False

    def diagnostics(self):
        return dict(status="tracked" if self.valid else "tracking_lost", reference_id=self.reference_id,
            tracking_valid=self.valid, response_validated=self.valid, proposal_available=self.valid,
            pixel_error_norm=self.error if self.valid else None, observed_rank=1, depth_support=1.,
            task_success_certified=False, grasp_certified=False)

    def configure(self, obs, ref, *, gripper_open, target_id):
        assert ref is None or gripper_open
        self.reference_id = None if ref is None else ref["reference_id"]
        if ref is None:
            return self.invalidate("pregrasp_reference_not_requested")
        return self.diagnostics()

    def observe(self, obs):
        self.observed += 1
        self.error = max(1., self.error - 1.)
        return self.diagnostics()

    def invalidate(self, reason):
        self.valid = False
        self.invalidated += 1
        return {**self.diagnostics(), "status": reason}

    def corrected_goal(self, obs, nominal):
        if self.raise_on_update:
            raise ValueError("offline tracker failure")
        return (np.asarray(nominal) + self.correction if self.valid else None), self.diagnostics()


def test_measured_reference_updates_xyz_field_without_altering_five_direct_actions():
    direct, field = alignment(direct=True), alignment()
    policy, transport = controller(direct, field)
    fixture = policy.pregrasp = FeedbackFixture()
    obs = observation()
    for expected in direct["direct"]["actions"]:
        obs, decision, _ = tick(policy, obs)
        assert decision.action == tuple(expected)
        assert policy.pregrasp_stats()["applied_goals"] == 0
    obs, first, _ = tick(policy, obs, fraction=.05)
    nominal = policy._nominal_goal.copy()
    assert len(transport.calls) == 2
    assert fixture.observed == 6 and fixture.invalidated == 0
    assert policy.pregrasp_stats()["applied_goals"] == 1
    np.testing.assert_allclose(policy.field.center + np.asarray(
        policy._field_cache.effective_compact_field["goal_offset_m"]), nominal + [0., 0., .004])
    assert policy.client.last_compact_prediction == field
    decision = policy.act(obs)
    assert decision.action[2] > 0.
    assert decision.action[3:6] == (0., 0., 0.)
    assert decision.action[6] == -1.
    assert decision.diagnostics["execution_kind"] == "field_tracking"
    assert policy.pregrasp_stats()["references_seeded"] == 1
    assert not policy.pregrasp_stats()["last_feedback"]["task_success_certified"]


@pytest.mark.parametrize("failure", ["tracking", "exception", "bad_goal", "over_six_mm", "rotation"])
def test_invalid_feedback_restores_nominal_field_without_hold_or_extra_request(failure):
    policy, transport = controller(alignment())
    fixture = policy.pregrasp = FeedbackFixture()
    obs, _, _ = tick(policy, observation(), fraction=.05)
    nominal = policy._nominal_goal.copy()
    assert policy.pregrasp_stats()["applied_goals"] == 1
    if failure == "tracking":
        fixture.valid = False
        obs, _, _ = tick(policy, obs, fraction=.05)
    else:
        if failure == "exception":
            fixture.raise_on_update = True
        elif failure == "bad_goal":
            fixture.correction = np.asarray([0., 0., .04])
        elif failure == "over_six_mm":
            fixture.correction = np.asarray([0., 0., .0061])
        else:
            policy._field_cache.effective_compact_field["orientation_target_xyzw"] = [0., 0., math.sin(.1), math.cos(.1)]
        policy._last_pregrasp_update_step = -100
        policy._maybe_update_goal(obs)
    np.testing.assert_allclose(policy.field.center + np.asarray(
        policy._field_cache.effective_compact_field["goal_offset_m"]), nominal)
    assert policy.pregrasp_stats()["nominal_goal_restorations"] == 1
    assert not policy._pregrasp_alignment_active()
    assert len(transport.calls) == 1
    decision = policy.act(obs)
    assert np.linalg.norm(decision.action[:3]) > 0.
    assert decision.action[6] == -1.
    assert len(transport.calls) == 1
    if failure == "exception":
        assert policy.pregrasp_stats()["errors"] == 1


def test_last_measured_epoch_survives_close_is_bounded_and_never_committed_as_history():
    raw = alignment(direct=True)
    policy, transport = controller(raw, enabled(operation()))
    fixture = policy.pregrasp = FeedbackFixture()
    obs = observation()
    for _ in range(5):
        obs, _, _ = tick(policy, obs)
    measured = copy.deepcopy(policy.pregrasp_local_feedback["recent_epoch_summary"])
    assert measured["image_error_reduction_px"] == 5.
    assert measured["last_measured_step"] == 5 and not measured["stale"]
    obs, decision, _ = tick(policy, obs, fingers=(.015, -.015))
    assert decision.action[6] == 1.
    summary = policy.pregrasp_stats()["last_feedback"]["recent_epoch_summary"]
    assert summary["stale"] and summary["last_measured_step"] == 5
    assert summary["latest_pixel_error_norm"] == measured["latest_pixel_error_norm"]
    assert summary["invalidation_reason"] == "pregrasp_reference_not_requested"
    assert not summary["grasp_certified"]
    assert len(json.dumps(summary)) < 1000
    payload = policy.client.build_payload(obs.to_dict())
    current = pregrasp_records(payload["input"])
    assert len(current) == 1 and current[0]["recent_epoch_summary"] == summary
    assert pregrasp_records(policy.client.history_snapshot()) == []
    assert all(len(pregrasp_records(call["input"])) == 1 for call in transport.calls)
    policy.reset()
    assert policy.pregrasp_stats()["applied_goals"] == 0
    assert "recent_epoch_summary" not in policy.pregrasp_stats()["last_feedback"]
    assert policy.client.latest_pregrasp_feedback is None


def test_feedback_must_match_pending_action_before_tracker_mutates():
    policy, _ = controller(alignment(direct=True))
    fixture = policy.pregrasp = FeedbackFixture()
    obs = observation()
    decision = policy.act(obs)
    with pytest.raises(ValueError, match="exact pending"):
        policy.observe_transition(replace(obs, state=(.2, *obs.state[1:])), decision, obs)
    assert fixture.observed == 0
    assert policy.pregrasp_stats()["observed_transitions"] == 0


def test_local_correction_keeps_thirty_action_limit_then_refreshes_model():
    policy, transport = controller(alignment(), enabled(prediction(direct=True)))
    policy.pregrasp = FeedbackFixture()
    obs = observation()
    for i in range(30):
        obs, decision, _ = tick(policy, obs, fraction=.005)
        assert len(transport.calls) == 1
        assert decision.diagnostics["execution_kind"] == "field_tracking"
        assert policy._chunk_length == 30
    assert policy.pregrasp_stats()["applied_goals"] > 0
    assert policy.pending_field_event["reason"] == "field_budget_exhausted"
    obs, decision, _ = tick(policy, obs)
    assert len(transport.calls) == 2
    assert decision.diagnostics["execution_kind"] == "direct_five_actions"
    assert policy._chunk_length == 5


def test_synthetic_rgb_and_measured_xyz_drive_real_field_without_additional_requests():
    from test_speed_first_pregrasp import (sample, reference as rgb_reference, TRAJECTORY, BASE)
    path = [*TRAJECTORY, TRAJECTORY[-1]]
    responses = []
    for start in (0, 5):
        _, points = sample(path[start])
        ref = rgb_reference(points)
        ref["target_id"] = "plate"
        row = alignment(direct=True, ref=ref)
        row["direct"]["actions"] = [[*((np.asarray(b) - a) / .05), 0., 0., 0., -1.]
            for a, b in zip(path[start:start+5], path[start+1:start+6])]
        responses.append(row)
    _, points = sample(path[-1])
    ref = rgb_reference(points)
    ref["target_id"] = "plate"
    responses.append(alignment(goal=(.002, .001, .001), ref=ref))
    policy, transport = controller(*responses)
    obs, _ = sample(path[0])
    obs = replace(obs, prompt=policy.client.task_prompt)
    for index, delta in enumerate(path[1:]):
        decision = policy.act(obs)
        assert decision.action == tuple(responses[index // 5]["direct"]["actions"][index % 5])
        after, _ = sample(delta)
        after = replace(after, prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    previous_error = policy.pregrasp_local_feedback["pixel_error_norm"]
    assert policy.pregrasp_local_feedback["response_validated"]
    assert policy.pregrasp_local_feedback["observed_rank"] == 3
    assert len(transport.calls) == 2
    for _ in range(3):
        decision = policy.act(obs)
        delta = np.asarray(obs.state[:3]) - BASE[:3] + .01 * np.asarray(decision.action[:3])
        after, _ = sample(delta)
        after = replace(after, prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert len(transport.calls) == 3
    assert policy.pregrasp_stats()["references_seeded"] == 1
    assert policy.pregrasp_stats()["applied_goals"] > 0
    assert decision.action[2] > 0.
    assert policy.pregrasp_local_feedback["pixel_error_norm"] < previous_error
    assert policy.client.last_compact_prediction == responses[-1]
    assert not policy.pregrasp_local_feedback["grasp_certified"]


@pytest.mark.parametrize("view,mode", [("wrist", "fixed_wrist"), ("wrist", "tracked_material"),
                                      ("external", "tracked_material")])
def test_independent_view_reference_is_validated_atomically_and_preserved(view, mode):
    ref = reference()
    ref["external" if view == "wrist" else "wrist"] = None
    ref[view]["tool_reference_mode"] = mode
    raw = alignment(direct=True, ref=ref)
    client = ChatGPTPlanSpeedFirstV3Client(pregrasp_feedback=True, token_provider=lambda: "offline-only")
    accepted = client.validate_response(raw, FINE_GRID)
    assert accepted["direct"] == raw["direct"]
    assert client.last_pregrasp_reference == ref
    assert client.last_compact_prediction == raw
    malformed = copy.deepcopy(raw)
    malformed["control_context"]["pregrasp_reference"][view] = None
    with pytest.raises(SchemaError):
        client.validate_response(malformed, FINE_GRID)
    assert client.last_pregrasp_reference == ref
    assert client.last_compact_prediction == raw
    assert client.history_snapshot() == []


def test_current_feedback_cannot_report_lost_view_error_as_alignment_progress():
    policy, transport = controller(alignment(direct=True), alignment(direct=True))
    fixture = policy.pregrasp = FeedbackFixture()
    obs = observation()
    obs, _, _ = tick(policy, obs)
    both = {**fixture.diagnostics(), "active_views": ["external", "wrist"], "pixel_error_norm": 15.}
    diagnostic = policy._remember_pregrasp_measurement(both)
    assert diagnostic["recent_epoch_summary"]["image_error_reduction_px"] == 0.
    prior_basis_changes = diagnostic["recent_epoch_summary"]["measurement_basis_changes"]
    policy._pregrasp_epoch_summary["applied_goal_count"] = 2
    wrist = {**fixture.diagnostics(), "active_views": ["wrist"], "pixel_error_norm": 3.,
             "views": {"external": {"status": "occluded", "tracking_valid": False},
                       "wrist": {"status": "tracked", "tracking_valid": True}}}
    diagnostic = policy._remember_pregrasp_measurement(wrist)
    summary = diagnostic["recent_epoch_summary"]
    assert summary["image_error_reduction_px"] == 0.
    assert summary["measurement_basis_changes"] == prior_basis_changes + 1
    assert summary["applied_goal_count"] == 2
    assert summary["measurement_views"] == ["wrist"]
    policy.client.latest_pregrasp_feedback = diagnostic
    current = pregrasp_records(policy.client.build_payload(obs.to_dict())["input"])
    assert len(current) == 1 and current[0]["views"] == wrist["views"]
    assert len(json.dumps(current)) < 2500
    assert pregrasp_records(policy.client.history_snapshot()) == []
    assert len(transport.calls) == 1


def test_wrist_only_rgb_updates_field_and_keeps_fit_across_same_id_refresh():
    from test_speed_first_pregrasp import (sample, reference as rgb_reference, TRAJECTORY, BASE)
    path = [*TRAJECTORY, TRAJECTORY[-1]]

    def wrist_ref(delta):
        _, points = sample(delta)
        ref = rgb_reference(points)
        ref["target_id"] = "plate"
        ref["external"] = None
        ref["wrist"]["tool_reference_mode"] = "fixed_wrist"
        return ref

    responses = []
    for start in (0, 5):
        row = alignment(direct=True, ref=wrist_ref(path[start]))
        row["direct"]["actions"] = [[*((np.asarray(b) - a) / .05), 0., 0., 0., -1.]
            for a, b in zip(path[start:start+5], path[start+1:start+6])]
        responses.append(row)
    responses.append(alignment(goal=(.002, .001, .001), ref=wrist_ref(path[-1])))
    policy, transport = controller(*responses)
    obs, _ = sample(path[0], blank="external_tool")
    obs = replace(obs, prompt=policy.client.task_prompt)
    for index, delta in enumerate(path[1:]):
        decision = policy.act(obs)
        assert decision.action == tuple(responses[index // 5]["direct"]["actions"][index % 5])
        after, _ = sample(delta, blank="external_tool")
        after = replace(after, prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    prior = copy.deepcopy(policy.pregrasp_local_feedback)
    assert prior["response_validated"], prior
    assert prior["active_views"] == ["wrist"]
    assert prior["observed_rank"] == 2
    assert len(transport.calls) == 2
    for _ in range(3):
        decision = policy.act(obs)
        assert decision.diagnostics["execution_kind"] == "field_tracking"
        goal = policy.field.center + np.asarray(policy._field_cache.effective_compact_field["goal_offset_m"])
        correction = goal - policy._nominal_goal
        assert np.linalg.norm(correction) <= .006 + 1e-12
        # The wrist fixture has no information along this null direction.
        assert abs(correction @ np.array([1., -1., 0.])) < 1e-5
        delta = np.asarray(obs.state[:3]) - BASE[:3] + .01 * np.asarray(decision.action[:3])
        after, _ = sample(delta, blank="external_tool")
        after = replace(after, prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    current = policy.pregrasp_local_feedback
    assert len(transport.calls) == 3
    assert policy.pregrasp_stats()["references_seeded"] == 1
    assert policy.pregrasp_stats()["applied_goals"] > 0
    assert current["pixel_error_norm"] < prior["pixel_error_norm"]
    assert current["sample_count"] >= prior["sample_count"]
    assert current["held_out_samples"] >= prior["held_out_samples"]
    assert not current["grasp_certified"]
    assert policy.local_visual_total_s > 0 and policy.local_update_total_s > 0
    assert 0 < policy.pregrasp_stats()["local_seconds"] <= policy.local_visual_total_s + policy.local_update_total_s
    assert policy._chunk_length == 30
    assert policy.client.last_compact_prediction == responses[-1]
