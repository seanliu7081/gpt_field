"""Offline typed operation, local execution and ablation tests; no simulator/API."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.fine_distance_planner import FineCompactField, FineDistancePlanner
from gpt_field.planner import PlannerConfig
from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client, FINE_GRID
from gpt_field.speed_first_v4 import (ChatGPTPlanSpeedFirstV4Client, SpeedFirstV4Policy,
    WaypointAblationPlanner, speed_first_v4_schema)
from test_field_history import FakeTransport, envelope, observation
from test_speed_first_v2 import prediction, operation as gripper_request, tick, records
from test_speed_first_pregrasp_runner import reference, FeedbackFixture


def intent(kind="move", *, direct=False, goal=(.06, 0., 0.), budget=None,
           continuation=False, verify=None, ref=None, identifier=None):
    close = kind in ("grasp_close", "release")
    raw = (gripper_request("open" if kind == "release" else "close") if close else
           prediction(goal=goal, direct=direct))
    del raw["control_context"]
    target = raw["target_region"]
    if ref is not None:
        ref = copy.deepcopy(ref)
        ref["target_id"] = target
    mode = "pregrasp" if kind == "pregrasp" else "held_placement" if kind == "place_align" else "none"
    raw.update(operation=dict(id=identifier or "op-" + kind, kind=kind, target_id=target,
        intent="Execute the visible selected relation", expected_result="Inspect actual measured response",
        local_action_budget=budget if budget is not None else 5 if direct else 30,
        allow_local_continuation=continuation, prerequisites=[], verify_motion=verify),
        visual_intent=dict(mode=mode, pregrasp_reference=ref, placement_reference=None,
            tracking_reference=None, unavailable_reasons=dict(external=None if ref else "No distinct material patch",
                                                             wrist=None if ref else "Occluded target patch")),
        interaction_review=dict(previous_attempt=None, next_change="Inspect current views", prerequisites=[]))
    return raw


def controller(*responses, mode="field"):
    fake = FakeTransport(*(envelope(r, id=f"v4-{i}", model="gpt-6-astra",
                                   reasoning={"effort": "medium"})
                           for i, r in enumerate(responses)))
    client = ChatGPTPlanSpeedFirstV4Client(planner_mode=mode,
        token_provider=lambda: "offline-v4", transport=fake)
    return SpeedFirstV4Policy(client, planner_mode=mode), fake


def test_v4_is_new_explicit_schema_default_v3_unchanged():
    kwargs = dict(token_provider=lambda: pytest.fail("No credentials expected"))
    base = ChatGPTPlanSpeedFirstV3Client(**kwargs)
    original = copy.deepcopy(base.build_payload(observation().to_dict()))
    revised = ChatGPTPlanSpeedFirstV4Client(**kwargs).build_payload(observation().to_dict())
    assert "control_context" not in revised["text"]["format"]["schema"]["properties"]
    assert {"operation", "visual_intent"} <= revised["text"]["format"]["schema"]["properties"].keys()
    assert revised["reasoning"] == {"effort": "medium"}
    assert base.build_payload(observation().to_dict()) == original
    assert "phase=align" not in revised["instructions"]


def test_atomic_validation_rejects_unsafe_or_inconsistent_operation():
    client = ChatGPTPlanSpeedFirstV4Client(token_provider=lambda: "offline")
    valid = intent()
    client.validate_response(valid, FINE_GRID)
    mutations = [lambda r: r["operation"].update(target_id="plate"),
        lambda r: r["operation"].update(local_action_budget=60),
        lambda r: r.update(gripper_operation={"command": "close"}),
        lambda r: r["field"]["orientation_target_xyzw"].append(0.),
        lambda r: r["operation"].update(kind="grasp_close"),
        lambda r: r["visual_intent"].update(pregrasp_reference=reference())]
    for mutation in mutations:
        bad = copy.deepcopy(valid)
        mutation(bad)
        with pytest.raises(SchemaError):
            client.validate_response(bad, FINE_GRID)
        assert client.last_operation == valid["operation"]
        assert client.last_compact_prediction == valid
    assert not client.history_snapshot()


@pytest.mark.parametrize("kind,direct", [("pregrasp", False), ("pregrasp", True),
                                       ("place_align", False), ("place_align", True)])
def test_operation_gripper_semantics_apply_even_with_all_visual_references_null(kind, direct):
    raw = intent(kind, direct=direct)
    if kind == "pregrasp":
        if direct:
            raw["direct"]["actions"][0][6] = 1.
        else:
            raw["field"]["gripper_command"] = "close"
    # place_align fixtures default open, hence are contradictory without edits.
    client = ChatGPTPlanSpeedFirstV4Client(token_provider=lambda: "offline")
    with pytest.raises(SchemaError, match="gripper|commands"):
        client.validate_response(raw, FINE_GRID)
    assert client.last_operation is None and not client.history_snapshot()


def test_pregrasp_maps_semantic_operation_and_direct_actions_remain_exact():
    direct = intent("pregrasp", direct=True, ref=reference())
    field = intent("pregrasp", ref=reference())
    policy, fake = controller(direct, field)
    policy.pregrasp = FeedbackFixture(correction=(0., .004, 0.))
    obs = observation()
    for action in direct["direct"]["actions"]:
        obs, d, _ = tick(policy, obs)
        assert d.action == tuple(action)
        assert d.diagnostics["execution_kind"] == "direct_five_actions"
        assert policy.operation["kind"] == "pregrasp"
    obs, d, _ = tick(policy, obs, fraction=.05)
    assert len(fake.calls) == 2
    assert policy.control_context["phase"] == "align"  # internal adapter only
    assert policy.pregrasp_stats()["applied_goals"] > 0
    np.testing.assert_allclose(policy.field.center + np.asarray(
        policy._field_cache.effective_compact_field["goal_offset_m"]), policy._nominal_goal + [0., .004, 0.])
    assert "control_context" not in policy.client.last_compact_prediction


def test_current_ledger_not_recursive_and_failed_request_commits_nothing():
    p, fake = controller(intent("direct", direct=True), intent("direct", direct=True))
    obs = observation()
    for _ in range(6):
        obs, _, _ = tick(p, obs)
    for payload in fake.calls:
        current = [r for r in records(payload["input"]) if r.get("type") == "current_operation_feedback"]
        assert len(current) == 1
    assert not [r for r in records(p.client.history_snapshot()) if r.get("type") == "current_operation_feedback"]
    history = p.client.history_snapshot()
    ledger = p.client.operation_ledger.presentation(p.client._observation_count, current_state=obs.state)
    fake.responses.append(APIError("offline failure"))
    with pytest.raises(APIError):
        p.client.predict(obs.to_dict())
    assert p.client.history_snapshot() == history
    assert p.client.operation_ledger.presentation(p.client._observation_count, current_state=obs.state) == ledger


def test_finite_field_continuation_requires_actual_progress_and_does_not_requery_at30():
    p, fake = controller(intent(goal=(.075, 0., 0.), budget=60, continuation=True), intent("direct", direct=True))
    obs = observation()
    for i in range(60):
        # Deliberately slow measured motion to test exact30->60 handoff.
        after_xyz = np.asarray(obs.state[:3]) + [.0006, 0., 0.]
        obs, d, _ = tick(p, obs, actual_position=after_xyz)
        assert len(fake.calls) == 1
        assert d.diagnostics["execution_kind"] == "field_tracking"
    assert p.operation_stats()["continuations_granted"] == 1
    assert p.operation_stats()["continuation_actions"] == 30
    assert p.pending_field_event["reason"] == "field_budget_exhausted"
    obs, _, _ = tick(p, obs)
    assert len(fake.calls) == 2


def test_motion_without_goal_progress_does_not_extend_or_hide_stall():
    p, fake = controller(intent(goal=(.075, 0., 0.), budget=60, continuation=True), intent("direct", direct=True))
    obs = observation()
    for _ in range(30):
        obs, _, _ = tick(p, obs, actual_position=np.asarray(obs.state[:3]) + [0., .0006, 0.])
        assert len(fake.calls) == 1
    assert p.operation_stats()["continuations_denied"] == 1
    assert p.pending_field_event is not None
    tick(p, obs)
    assert len(fake.calls) == 2
    stalled, calls = controller(intent(budget=60, continuation=True), intent("direct", direct=True))
    obs = observation()
    for _ in range(5):
        obs, _, _ = tick(stalled, obs, fraction=0.)
    assert stalled.pending_field_event["reason"] == "no_progress"
    tick(stalled, obs)
    assert len(calls.calls) == 2


def test_explicit_close_then_verify_uses_fresh_measured_pose_and_no_extra_model_call():
    verify = dict(offset_xyz_m=[0., 0., .02], keep_orientation=True)
    raw = intent("grasp_close", budget=60, continuation=True, verify=verify)
    p, fake = controller(raw, intent("direct", direct=True))
    obs = observation()
    for i in range(15):
        # Pose drift is observed, not mistaken for the initial nominal position.
        obs, d, _ = tick(p, obs, fingers=(.015, -.015),
            actual_position=[.102, .2, 1.001], rotvec=[0., 0., .03])
        assert d.diagnostics["execution_kind"] == "local_gripper_settle"
        assert d.action[6] == 1.
    fresh = obs
    obs, d, _ = tick(p, obs)
    assert len(fake.calls) == 1
    assert d.diagnostics["execution_kind"] == "local_close_verification"
    assert d.diagnostics["refreshed"] is False
    assert d.diagnostics["chunk_step"] == 15
    assert d.action[6] == 1. and d.action[3:6] == (0., 0., 0.)
    np.testing.assert_allclose(p._nominal_goal, np.asarray(fresh.state[:3]) + verify["offset_xyz_m"])
    assert p.operation_stats()["close_verify_started"] == 1
    assert not p.operation_stats()["current"]["grasp_certified"]
    assert p.pending_field_event["reason"] == "goal_reached"
    tick(p, obs)
    assert len(fake.calls) == 2


def test_unsettled_close_never_executes_requested_verification():
    verify = dict(offset_xyz_m=[.02, 0., 0.], keep_orientation=True)
    p, fake = controller(intent("grasp_close", budget=60, continuation=True, verify=verify),
                         intent("direct", direct=True))
    obs = observation()
    for _ in range(30):
        obs, d, _ = tick(p, obs)
        assert d.diagnostics["execution_kind"] == "local_gripper_settle"
    assert p.operation_stats()["close_verify_skipped_unsettled"] == 1
    assert p.operation_stats()["verification_actions"] == 0
    tick(p, obs)
    assert len(fake.calls) == 2


def test_verification_has_own_finite_cap_inside_total60_and_failure_to_move_returns_model():
    verify = dict(offset_xyz_m=[.03, 0., 0.], keep_orientation=True)
    p, fake = controller(intent("grasp_close", budget=60, continuation=True, verify=verify),
                         intent("direct", direct=True))
    obs = observation()
    for _ in range(15):
        obs, _, _ = tick(p, obs, fingers=(.015, -.015))
    for _ in range(5):
        obs, d, _ = tick(p, obs, fraction=0.)
        assert d.diagnostics["execution_kind"] == "local_close_verification"
    assert len(fake.calls) == 1
    assert p.pending_field_event["reason"] == "no_progress"
    tick(p, obs)
    assert len(fake.calls) == 2


def test_bad_transition_is_atomic_for_operation_ledger_visual_and_budgets():
    p, _ = controller(intent())
    obs = observation()
    decision = p.act(obs)
    before_stats = p.operation_stats()
    before_ledger = p.client.operation_ledger.presentation(1, current_state=obs.state)
    with pytest.raises(ValueError, match="exact pending"):
        p.observe_transition(replace(obs, state=(.2, *obs.state[1:])), decision, obs)
    assert p.operation_stats() == before_stats
    assert p.client.operation_ledger.presentation(1, current_state=obs.state) == before_ledger


def test_waypoint_ablation_is_different_on_obstacle_not_only_name():
    compact = intent(goal=(.012, 0., 0.))["field"]
    del compact["goal_reference"]
    compact["blocked_boxes"] = [dict(lower_offset_m=[.004, -.002, -.002],
                                    upper_offset_m=[.008, .002, .002])]
    field = FineCompactField(FINE_GRID, (.1, .2, 1.), compact)
    config = PlannerConfig(horizon=10)
    args = dict(goal_xyz=(.112, .2, 1.), compact_prediction=compact, goal_tolerance_m=.001)
    field_plan = FineDistancePlanner(config).plan_analytic(field, (.1, .2, 1.), **args)
    line_plan = WaypointAblationPlanner(config).plan_analytic(field, (.1, .2, 1.), **args)
    assert field_plan.status == "move"
    assert line_plan.status == "no_valid_candidates"
    assert line_plan.reason == "straight_path_blocked_no_detour_search"
    assert line_plan.search_metadata["uses_field_cost"] is False
    selected = np.asarray(field_plan.candidates[0].positions)
    assert np.max(np.linalg.norm(selected[:, 1:] - [.2, 1.], axis=1)) > .002


def test_waypoint_mode_preserves_actual_local_execution_and_clear_case_endpoint():
    runs = []
    for mode in ("field", "waypoint"):
        p, fake = controller(intent(goal=(.06, 0., 0.)), mode=mode)
        obs = observation()
        for _ in range(5):
            obs, d, _ = tick(p, obs)
            if p.pending_field_event:
                break
        runs.append(obs.state)
        assert len(fake.calls) == 1
        assert p.pending_field_event["reason"] == "goal_reached"
        assert d.diagnostics["planner_mode"] == mode
    np.testing.assert_allclose(runs[0], runs[1])


def test_changed_operation_identity_is_rejected_before_model_history_commit():
    first = intent("direct", direct=True, identifier="same-id")
    second = intent("move", identifier="same-id")
    p, fake = controller(first, second)
    obs = observation()
    for _ in range(5):
        obs, _, _ = tick(p, obs)
    history = p.client.history_snapshot()
    ledger = p.client.operation_ledger.presentation(1, current_state=obs.state)
    with pytest.raises(SchemaError, match="cannot change kind"):
        p.act(obs)
    assert len(fake.calls) == 2
    assert p.client._observation_count == 1
    assert p.client.history_snapshot() == history
    assert p.client.operation_ledger.presentation(1, current_state=obs.state) == ledger
    assert p._pending is None


def test_actual_image_evidence_reaches_current_operation_feedback_without_changing_direct():
    from test_operation_visual import observation as textured_observation, reference as roi_reference
    raw = intent("verify_grasp", direct=True)
    raw["direct"]["actions"] = [[.06, 0., 0., 0., 0., 0., 1.] for _ in range(5)]
    raw["visual_intent"]["tracking_reference"] = roi_reference()
    raw["visual_intent"]["unavailable_reasons"] = dict(external=None, wrist=None)
    p, fake = controller(raw)
    obs = replace(textured_observation(), prompt=p.client.task_prompt)
    for i in range(1, 6):
        d = p.act(obs)
        assert d.action == tuple(raw["direct"]["actions"][i-1])
        next_obs = replace(textured_observation(
            {("external", role): (i*2, 0) for role in ("tool", "object")}, state_dx=.003*i),
            prompt=p.client.task_prompt)
        p.observe_transition(obs, d, next_obs)
        obs = next_obs
    assert len(fake.calls) == 1
    feedback = p.current_operation_feedback()
    assert feedback["visual_evidence"]["verification_evidence"] == "supporting"
    assert not feedback["grasp_certified"] and not feedback["task_success_certified"]
    assert p.operation_stats()["visual_seconds"] > 0.
    assert p.local_visual_total_s >= p.operation_stats()["visual_seconds"]
    current = records(p.client.build_payload(obs.to_dict())["input"])[-1]
    assert current["type"] == "current_operation_feedback"
    assert current["execution"]["visual_evidence"]["verification_evidence"] == "supporting"


def test_operation_tracking_exception_cannot_insert_hold_or_an_extra_request():
    p, fake = controller(intent("direct", direct=True))
    def broken(*args):
        raise RuntimeError("Image backend fixture failed")
    p.outcome.configure = broken
    p.outcome.observe = broken
    obs = observation()
    expected = intent("direct", direct=True)["direct"]["actions"]
    for action in expected:
        obs, d, _ = tick(p, obs)
        assert d.action == tuple(action)
    assert len(fake.calls) == 1
    assert p.operation_stats()["visual_errors"] == 6
    assert p.outcome_feedback["verification_evidence"] == "unknown"


@pytest.mark.parametrize("bad", [(.007, 0., 0.), (float("nan"), 0., 0.)])
def test_pregrasp_invalid_goal_restores_nominal_and_falls_through_without_model_call(bad):
    p, fake = controller(intent("pregrasp", ref=reference()))
    fixture = p.pregrasp = FeedbackFixture(correction=(0., .004, 0.))
    obs, _, _ = tick(p, observation(), fraction=.05)
    nominal = p._nominal_goal.copy()
    assert p.pregrasp_stats()["applied_goals"] == 1
    fixture.correction = np.asarray(bad)
    p._last_pregrasp_update_step = -100
    obs, d, _ = tick(p, obs, fraction=.05)
    np.testing.assert_allclose(p.field.center + np.asarray(
        p._field_cache.effective_compact_field["goal_offset_m"]), nominal)
    assert len(fake.calls) == 1
    assert d.diagnostics["execution_kind"] == "field_tracking"
    assert np.linalg.norm(d.action[:3]) > 0.


def test_verification_30action_budget_expires_even_when_slowly_making_progress():
    p, fake = controller(intent("grasp_close", budget=60, continuation=True,
        verify=dict(offset_xyz_m=[.03, 0., 0.], keep_orientation=True)), intent("direct", direct=True))
    obs = observation()
    for _ in range(15):
        obs, _, _ = tick(p, obs, fingers=(.015, -.015))
    for _ in range(30):
        obs, d, _ = tick(p, obs, actual_position=np.asarray(obs.state[:3]) + [.0006, 0., 0.])
        assert d.diagnostics["execution_kind"] == "local_close_verification"
        assert len(fake.calls) == 1
    assert p.operation_stats()["verification_actions"] == 30
    assert p.pending_field_event["reason"] == "field_budget_exhausted"
    tick(p, obs)
    assert len(fake.calls) == 2


def test_confirmed_overload_retries_same_payload_without_history_or_operation_duplication(monkeypatch):
    error = APIError("offline provider overload")
    error.error_code, error.http_status = "server_is_overloaded", 503
    monkeypatch.setattr("gpt_field.speed_first_hybrid.time.sleep", lambda _: None)
    fake = FakeTransport(error, envelope(intent("direct", direct=True), model="gpt-6-astra",
                                        reasoning={"effort": "medium"}))
    client = ChatGPTPlanSpeedFirstV4Client(token_provider=lambda: "offline", transport=fake)
    policy = SpeedFirstV4Policy(client)
    tick(policy, observation())
    assert len(fake.calls) == 2 and fake.calls[0] == fake.calls[1]
    assert policy.operation_stats()["operations"] == 1
    assert client._observation_count == 1


@pytest.mark.parametrize("failure", ["too_large", "nonfinite", "lost", "exception"])
def test_held_placement_correction_has_same_total6mm_limit_and_nominal_fallback(failure):
    from types import SimpleNamespace
    from test_speed_first_visual import reference as placement_reference
    class HeldFeedback:
        tracker = SimpleNamespace(maximum_orientation_change_rad=.035, image_tolerance_px=2.)
        correction = np.array([0., .004, 0.])
        valid = True
        fail = False
        def diagnostics(self):
            return dict(calibration_valid=self.valid, proposal_available=self.valid,
                        pixel_error_norm=12., task_success_certified=False)
        def configure(self, *args, **kwargs):
            return self.diagnostics()
        def observe(self, observation):
            return self.diagnostics()
        def invalidate(self, reason):
            self.valid = False
            return self.diagnostics()
        def corrected_goal(self, observation, nominal):
            if self.fail:
                raise ValueError("Offline vision error")
            return (nominal + self.correction if self.valid else None), self.diagnostics()
    raw = intent("place_align")
    raw["field"]["gripper_command"] = "close"
    raw["visual_intent"]["placement_reference"] = placement_reference()
    raw["visual_intent"]["unavailable_reasons"]["external"] = None
    p, fake = controller(raw)
    p.visual = HeldFeedback()
    obs, _, _ = tick(p, observation(), fraction=.05)
    assert p.local_goal_updates == 1
    nominal = p._nominal_goal.copy()
    if failure == "too_large":
        p.visual.correction = np.array([0., .007, 0.])
    elif failure == "nonfinite":
        p.visual.correction = np.array([float("nan"), 0., 0.])
    elif failure == "lost":
        p.visual.valid = False
    else:
        p.visual.fail = True
    p._last_visual_update_step = -100
    obs, d, _ = tick(p, obs, fraction=.05)
    np.testing.assert_allclose(p.field.center + np.asarray(
        p._field_cache.effective_compact_field["goal_offset_m"]), nominal)
    assert len(fake.calls) == 1
    assert np.linalg.norm(d.action[:3]) > 0.
