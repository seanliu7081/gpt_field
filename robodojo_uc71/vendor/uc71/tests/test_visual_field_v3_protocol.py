"""V3 intent, typed evidence and local acquisition contracts without live services."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.astra import SchemaError
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_policy_v3 import VisualFieldPolicyV3, ProgressLedger
from gpt_field.visual_field_subscription_v3 import VisualFieldSubscriptionClientV3, output_schema, VERSION
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import prediction as old_prediction, observation, input_texts
from test_visual_field_v2_protocol import assert_unique_required


def prediction(*, field=False, actions=None, grant=None):
    value = old_prediction(field=field, actions=actions)
    value["observation_intent"] = grant
    return value


def reference():
    return {"moving_id": "tool", "target_id": "target", "moving_kind": "tool", "tool_point_uv": None,
            "moving_point_uv": [.3, .3], "target_anchor_uv": [.8, .8], "target_point_uv": [.7, .7],
            "moving_footprint_uv": {"lower_uv": [.28, .28], "upper_uv": [.32, .32]},
            "blocked_boxes_uv": [], "unknown_boxes_uv": []}


def grant(**updates):
    result = {"blocker_codes": ["response_unsupported"], "directions": [[.1, 0., 0.]],
              "max_steps": 3, "max_displacement_m": .02,
              "corridors_uv": {v: {"lower_uv": [.1, .1], "upper_uv": [.9, .9]} for v in ("external", "wrist")}}
    return {**result, **updates}


def client(*values, **kwargs):
    transport = FakeTransport(*(envelope(v, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v3_{i}")
                                for i, v in enumerate(values)))
    return VisualFieldSubscriptionClientV3(token_provider=lambda: "offline-fixture", transport=transport, **kwargs), transport


def blocker(code="start_unknown_conflict", view="wrist"):
    return {"code": code, "category": "semantic", "view": view, "feature_id": "moving", "region_id": "unknown_A",
            "evidence": {"overlap": True}, "remedy": "Correct the contradictory mask; calibration cannot repair it"}


def feedback(*, blockers=(), rank=0, validations=0, error=20., reliable=True, epoch=1):
    model = {"rank": rank, "validation_count": validations, "residual": 1., "sample_count": validations+2,
             "status": "ready" if validations >= 2 else "awaiting_samples", "reliable": validations >= 2}
    view = {"moving_id": "tool", "target_id": "target", "moving_kind": "tool", "reliable": reliable,
            "moving_track_reliable": reliable, "tool_track_reliable": False, "geometry_reliable": True,
            "landmark_epoch": epoch, "model_epoch": epoch, "goal_epoch": 1,
            "goal_descriptor": {"moving_id": "tool", "target_id": "target", "target_offset_px": [0, 0]},
            "goal_error_px": error, "distance_cost_px": error,
            "eef_to_image": copy.deepcopy(model), "eef_to_target_anchor": copy.deepcopy(model)}
    return {"views": {v: copy.deepcopy(view) for v in ("external", "wrist")},
            "tracks_reliable": reliable, "geometry_reliable": True, "blockers": list(copy.deepcopy(blockers)),
            "feasibility": {"semantic_repair_required": bool(blockers)}, "co_motion_evidence": False,
            "attachment_epoch": 1, "task_success_certified": False}


def test_new_schema_roles_footprint_grant_and_old_protocol_separation():
    assert_unique_required(output_schema())
    value = prediction(field=True, grant=grant())
    value["visual_references"] = {v: reference() for v in ("external", "wrist")}
    instance, _ = client()
    assert instance.validate_response(value, None) == value
    legacy = copy.deepcopy(value)
    for ref in legacy["visual_references"].values():
        del ref["moving_footprint_uv"]
        ref["moving_radius_uv"] = .1
    with pytest.raises(SchemaError):
        instance.validate_response(legacy, None)
    from gpt_field.visual_field_subscription_v2 import output_schema as v2_schema
    assert "observation_intent" not in v2_schema()["properties"]
    assert "moving_footprint_uv" not in json.dumps(v2_schema())


@pytest.mark.parametrize("change", ["direct_grant", "zero_direction", "large_norm", "fractional_steps", "zero_radius", "null_corridors"])
def test_invalid_local_grants_rejected_without_actions(change):
    value = prediction(field=True, grant=grant())
    if change == "direct_grant": value = prediction(grant=grant())
    if change == "zero_direction": value["observation_intent"]["directions"] = [[0., 0., 0.]]
    if change == "large_norm": value["observation_intent"]["directions"] = [[.1, .1, 0.]]
    if change == "fractional_steps": value["observation_intent"]["max_steps"] = 2.5
    if change == "zero_radius": value["observation_intent"]["max_displacement_m"] = 0.
    if change == "null_corridors": value["observation_intent"]["corridors_uv"] = {"external": None, "wrist": None}
    instance, transport = client()
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)
    assert not transport.calls


def test_unclipped_footprint_is_reported_as_coverage_blocker_not_silently_clipped():
    from gpt_field.visual_field_v3 import preflight_reference
    value = prediction()
    ref = reference(); ref["moving_footprint_uv"]["lower_uv"] = [-.1, .2]
    value["visual_references"]["external"] = ref
    instance, _ = client()
    assert instance.validate_response(value, None)["visual_references"]["external"]["moving_footprint_uv"]["lower_uv"][0] == -.1
    result = preflight_reference(ref, (128, 128))
    assert any(b["code"] == "footprint_out_of_view" for b in result["blockers"])


@pytest.mark.parametrize("case", ["same_landmark", "blank_landmark", "duplicate_region", "blank_region"])
def test_invalid_identity_fails_in_protocol_before_local_configuration(case):
    value = prediction(); ref = reference()
    if case == "same_landmark": ref["target_id"] = ref["moving_id"]
    if case == "blank_landmark": ref["moving_id"] = " "
    if case in ("duplicate_region", "blank_region"):
        ref["blocked_boxes_uv"] = [{"object_id": "obstacle" if case == "duplicate_region" else " ",
            "lower_uv": [.4, .4], "upper_uv": [.5, .5], "role": "obstacle"}]
        if case == "duplicate_region":
            ref["unknown_boxes_uv"] = [{"object_id": "obstacle", "lower_uv": [.6, .6],
                "upper_uv": [.7, .7], "source": "occluded"}]
    value["visual_references"]["external"] = ref
    instance, _ = client()
    with pytest.raises(SchemaError):
        instance.validate_response(value, None)


def test_typed_blocker_does_not_clear_for_unrelated_motion_or_other_view_information():
    ledger = ProgressLedger()
    first = feedback(blockers=[blocker()])
    ledger.update(first, semantic=True)
    # State movement is intentionally absent from this evidence interface.
    unchanged = copy.deepcopy(first)
    assert ledger.update(unchanged) == []
    improved_external = copy.deepcopy(first)
    improved_external["views"]["external"]["eef_to_image"].update(rank=1, validation_count=1)
    assert ledger.update(improved_external)
    assert len(ledger.active) == 1
    assert next(iter(ledger.active.values()))["view"] == "wrist"
    lost = feedback(reliable=False)
    ledger.update(lost)
    assert ledger.active, "Disappearing reference cannot clear unknown space"


def test_goal_redefinition_or_landmark_reseed_does_not_invent_distance_progress():
    ledger = ProgressLedger(); ledger.update(feedback(error=30.), semantic=True)
    ledger.update(feedback(error=2., epoch=2))
    assert ledger.task_distance_events == 0
    value = feedback(error=1., epoch=2); value["views"]["external"]["goal_epoch"] = 2
    value["views"]["wrist"]["goal_epoch"] = 2
    ledger.update(value)
    assert ledger.task_distance_events == 0
    value = feedback(error=.5, epoch=2)
    ledger.update(value, semantic=True)
    assert ledger.task_distance_events == 0


def test_only_new_predictive_information_or_same_goal_progress_counts():
    ledger = ProgressLedger(); ledger.update(feedback(rank=1, validations=2), semantic=True)
    ledger.update(feedback(rank=1, validations=40))
    assert ledger.information_events == 0, "Routine validation counting is not perpetual information gain"
    ledger.update(feedback(rank=2, validations=41))
    assert ledger.information_events > 0
    ledger.update(feedback(rank=2, validations=42, error=17.))
    assert ledger.task_distance_events == 2


def test_same_named_landmark_reseed_cannot_claim_measured_blocker_resolution():
    ledger = ProgressLedger(); ledger.update(feedback(blockers=[blocker()]), semantic=True)
    ledger.update(feedback(epoch=2))
    assert ledger.information_events == 0
    assert ledger.superseded_counts["start_unknown_conflict"] == 1
    assert not ledger.resolved_counts


class Local:
    def __init__(self, outputs=(), *, info=None):
        self.outputs = list(outputs)
        self.info = feedback() if info is None else info
        self.grants = []
        self.observations = 0

    def configure(self, *args, **kwargs):
        return copy.deepcopy(self.info)

    def observe(self, *args, **kwargs):
        self.observations += 1
        return copy.deepcopy(self.info)

    def select_action(self, proposed, *, q=None, observation_intent=None):
        self.grants.append(copy.deepcopy(observation_intent))
        kind = self.outputs.pop(0) if self.outputs else "visual_distance_descent"
        accepted = kind in ("visual_distance_descent", "local_information_probe")
        action = [.1, 0., 0., 0., 0., 0., 0.] if accepted else [0.]*7
        return {"status": kind, "accepted": accepted, "action": action, "need_refresh": not accepted,
                "diagnostics": {"predicted_eef_delta_m": [.004, 0., 0.], "prediction_residual_margin_m": .0001}}


def step(policy, obs, dx=0.):
    decision = policy.act(obs)
    after = replace(obs, state=(obs.state[0]+dx, *obs.state[1:]))
    row = policy.observe_transition(obs, decision, after)
    return decision, after, row


def test_supported_local_acquisition_resumes_same_intent_without_model_refresh():
    value = prediction(field=True, grant=grant())
    instance, transport = client(value)
    policy = VisualFieldPolicyV3(instance)
    policy.local = Local(["local_information_probe", "local_information_probe", "visual_distance_descent"])
    obs = observation()
    for index in range(3):
        decision, obs, _ = step(policy, obs, .002)
        assert decision.diagnostics["refreshed"] is (index == 0)
    assert len(transport.calls) == 1
    assert policy._grant_steps == 2 and policy.diagnostic_counters["local_information_probe_actions"] == 2
    assert policy.local_mode == "persistent_field"
    assert policy.progress.probe_count == 2
    assert not policy.task_stagnation["evidence"]["information_events"]


@pytest.mark.parametrize("case", ["count", "radius", "direction", "structural"])
def test_probe_cannot_exceed_grant_or_bypass_semantic_conflict(case):
    grant_value = grant(max_steps=1 if case == "count" else 3,
                        max_displacement_m=.005 if case == "radius" else .02)
    if case == "direction": grant_value["directions"] = [[0., .1, 0.]]
    instance, _ = client(prediction(field=True, grant=grant_value))
    policy = VisualFieldPolicyV3(instance)
    policy.local = Local(["local_information_probe"]*3,
                         info=feedback(blockers=[blocker()]) if case == "structural" else None)
    obs = observation()
    if case in ("count", "radius"):
        _, obs, _ = step(policy, obs, .003)
    decision = policy.act(obs)
    assert not decision.diagnostics["selection"]["accepted"]
    assert decision.action == (0.,)*7
    assert policy.refresh_reason == "local_observation_grant_boundary"


def test_probe_remaining_radius_includes_prediction_uncertainty():
    instance, _ = client(prediction(field=True, grant=grant(max_displacement_m=.00705)))
    policy = VisualFieldPolicyV3(instance); policy.local = Local(["local_information_probe"]*2)
    obs = observation(); _, obs, _ = step(policy, obs, .003)
    # Current radius .003 + prediction .004 fits, but the .0001 residual margin does not.
    decision = policy.act(obs)
    assert decision.action == (0.,)*7
    assert decision.diagnostics["selection"]["status"] == "observation_grant_boundary"
    assert policy._grant_steps == 1


def test_unexpected_measured_probe_overshoot_requests_new_observation_without_episode_stop():
    instance, _ = client(prediction(field=True, grant=grant(max_displacement_m=.005)))
    policy = VisualFieldPolicyV3(instance); policy.local = Local(["local_information_probe"])
    _, obs, _ = step(policy, observation(), .006)
    assert policy.refresh_reason == "observation_grant_measured_displacement_exceeded"
    assert policy._available_grant(obs) is None
    assert policy.events[-1]["episode_stop"] is False


def test_rejected_field_close_intent_executes_no_gripper_change():
    value = prediction(field=True)
    value["field_intent"]["gripper_command"] = "close"
    instance, _ = client(value)
    policy = VisualFieldPolicyV3(instance); policy.local = Local(["semantic_correction_required"])
    decision, _, _ = step(policy, observation())
    assert decision.action == (0.,)*7
    assert policy.last_gripper == -1. and policy.adapter._gripper == -1.
    assert not policy.attempts


def test_direct_five_actions_unchanged_but_infeasibility_and_motion_stall_reach_next_context():
    actions = [[.1, 0., 0., 0., 0., 0., 1.]]*5
    instance, transport = client(*(prediction(actions=actions) for _ in range(4)))
    policy = VisualFieldPolicyV3(instance); policy.local = Local(info=feedback(blockers=[blocker()]))
    obs = observation()
    for i in range(11):
        decision, obs, _ = step(policy, obs, .004)
        assert decision.action == tuple(actions[i % 5])
    context = input_texts(transport.calls[-1])[-1]["observable_context"]
    assert context["task_stagnation"]["active"]
    assert context["task_stagnation"]["blocks_without_information"] == 2
    assert context["task_stagnation"]["reason"] == "semantic_reference_repair_required"
    assert context["progress_ledger"]["active_blockers"][0]["code"] == "start_unknown_conflict"
    assert policy.step_index == 11


def test_persistent_probe_ledger_survives_many_compacted_requests_without_privileged_input(tmp_path):
    instance, transport = client(*(prediction() for _ in range(6)), log_dir=tmp_path)
    policy = VisualFieldPolicyV3(instance); policy.local = Local(info=feedback(blockers=[blocker()]))
    obs = observation()
    for _ in range(26):
        _, obs, _ = step(policy, obs, .002)
    latest = transport.calls[-1]
    assert latest["reasoning"] == {"effort": "medium"}
    assert latest["text"]["format"]["name"] == "visual_field_rgb_joints_v3"
    records = input_texts(latest)
    memory = next(r for r in records if r["type"] == "visual_history_summary")
    assert memory["progress_ledger"]["active_blockers"]
    assert len([r for r in records if r["type"] == "observation"]) == 2
    assert len(list(tmp_path.glob("*.request.json"))) == 6
    instance.observable_context = {"progress_ledger": {"libero_success": False}}
    with pytest.raises(ValueError):
        instance.build_payload(obs.to_dict())
    assert instance.prompt_version == VERSION+"-hybrid"


def test_real_rgb_core_executes_multiple_local_steps_under_one_persistent_field_intent():
    from test_visual_field_v2 import images as frames, seeds as old_seeds, state
    from gpt_field.visual_field_v3 import VisualFieldControllerV3
    q = np.array([0., 0., 0., -2., 0., 2., .8])
    eef = np.zeros(2)

    def refs(at):
        result = old_seeds(target=(94, 38))
        moving = (np.array([42., 38.])+np.array(at)*400)/127
        for ref in result.values():
            ref.pop("moving_radius_uv")
            ref["moving_point_uv"] = moving.tolist()
            ref["moving_footprint_uv"] = {"lower_uv": (moving-.02).tolist(), "upper_uv": (moving+.02).tolist()}
        return result

    direct = prediction(actions=[[.1, 0., 0., 0., 0., 0., 0.]]*5)
    direct["visual_references"] = refs(eef)
    field = prediction(field=True)
    field["field_intent"]["goal_offset_m"] = [.04, 0., 0.]
    field["visual_references"] = refs([.025, 0.])
    instance, transport = client(direct, field)
    policy = VisualFieldPolicyV3(instance)
    policy.local = VisualFieldControllerV3(grid_size=16, joint_lower=[-3.]*7, joint_upper=[3.]*7)

    def obs():
        rgb = frames(eef)
        return PolicyObservation(*(image_to_data_url(rgb[v], size=128, rotate=False) for v in ("external", "wrist")),
                                 tuple(state(eef)), resolve_distance_task("bowl_plate").prompt, tuple(q), (0.,)*7)

    current = obs()
    for i in range(8):
        decision = policy.act(current)
        if i >= 5:
            assert decision.diagnostics["selection"]["accepted"]
            assert decision.diagnostics["selection"]["status"] == "visual_distance_descent"
        delta = np.array(decision.action[:2])*.05
        eef += delta
        q[:2] += np.array(decision.action[:2])*.1
        after = obs(); policy.observe_transition(current, decision, after); current = after
    assert len(transport.calls) == 2
    assert policy.progress.task_distance_events > 0
    assert policy.step_index == 8
