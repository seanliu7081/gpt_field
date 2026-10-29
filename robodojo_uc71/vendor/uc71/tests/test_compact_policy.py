"""Compact predictions preserve execution/history and distinguish fixed routing."""

import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.compact_field import COMPACT_FIELD_VERSION, expand_compact_field
from gpt_field.hybrid_policy import HybridFieldDirectPolicy, ModelTaskComplete
from gpt_field.hybrid_subscription import HYBRID_GRID
from test_compact_subscription import client, compact_field, direct, hybrid
from test_field_history import FakeTransport, envelope, observation
from test_hybrid_policy import execute


@pytest.mark.parametrize("mode", ["field", "direct", "hybrid"])
def test_compact_diagnostics_preserve_actual_model_output_and_routing_provenance(mode):
    raw = direct() if mode == "direct" else compact_field() if mode == "field" else hybrid()
    source = raw["field"] if mode == "hybrid" else raw if mode == "field" else None
    if source is not None:
        source["motion_mode"] = "hold"
    instance = client(FakeTransport(envelope(raw)), controller_mode=mode)
    policy = HybridFieldDirectPolicy(instance)
    decision = policy.act(observation())
    details = decision.diagnostics
    assert details["compact_prediction"] == raw
    assert details["field_representation"] == COMPACT_FIELD_VERSION
    assert details["routing_is_model_prediction"] is (mode == "hybrid")
    if mode != "hybrid":
        assert "routing was not predicted" in details["routing"]["routing_reason"]
        assert "controller" not in details["compact_prediction"]
    if source is None:
        assert decision.action == tuple(raw["actions"][0])
        assert "plan" not in details and "field_prediction" not in details
        assert details["field_expansion_latency_s"] == 0
    else:
        dense = expand_compact_field(source, HYBRID_GRID)
        assert details["field_prediction"] == details["hybrid_prediction"]["field"] == dense
        assert len(dense["values"]) == 35937
        assert details["field_center"] == list(observation().state[:3])
        assert details["plan_refreshed"] and details["plan_execution_steps"] == 10
        assert details["field_expansion_latency_s"] >= 0
        assert decision.action[:3] == (0., 0., 0.)
    # Diagnostics are evidence copies, not references into the client's accepted
    # raw response or its conversation history.
    details["compact_prediction"]["fixture_tamper"] = True
    assert instance.last_compact_prediction == raw
    assert "fixture_tamper" not in json.dumps(instance.history_snapshot())
    assert '"values"' not in json.dumps(instance.history_snapshot())


def test_compact_handoffs_keep_30_and_5_action_cadence_and_raw_history():
    raw = [hybrid(), hybrid(True), hybrid()]
    for index in (0, 2):
        raw[index]["field"]["motion_mode"] = "hold"
    raw[2]["field"]["stage"] = "returned_field"
    responses = [envelope(item, id=f"compact-{index}") for index, item in enumerate(raw)]
    transport = FakeTransport(*responses)
    instance = client(transport)
    policy = HybridFieldDirectPolicy(instance)
    obs = observation()
    decisions = []
    centers = []
    for step in range(36):
        centers.append(list(obs.state[:3]))
        obs, decision, feedback = execute(policy, obs, step)
        decisions.append(decision)
        details = decision.diagnostics
        assert details["routing_is_model_prediction"] is True
        assert details["field_representation"] == COMPACT_FIELD_VERSION
        assert ("compact_prediction" in details) == details["refreshed"]
        assert ("field_expansion_latency_s" in details) == details["refreshed"]
        assert feedback["step"] == step
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["refreshed"]] == [0, 30, 35]
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics.get("plan_refreshed")] == [0, 10, 20, 35]
    assert [d.diagnostics["step"] for d in decisions if "plan_latency_s" in d.diagnostics] == [0, 10, 20, 35]
    assert decisions[35].diagnostics["field_center"] == centers[35] != centers[0]
    assert decisions[35].diagnostics["plan_start_step"] == 35
    assert decisions[35].diagnostics["chunk_index"] == 2
    assert decisions[35].diagnostics["stage"] == "returned_field"
    assert len(transport.calls) == 3
    assert instance.history_stats()["feedback_records"] == 36
    assert instance.history_stats()["observations"] == 3
    for index in (1, 2):
        assert responses[index - 1]["output"][-1] in transport.calls[index]["input"]
        assert '"values"' not in json.dumps(transport.calls[index]["input"])
    assert decisions[30].diagnostics["field_expansion_latency_s"] == 0


def test_invalid_compact_field_cannot_advance_policy_or_commit_history():
    raw = copy.deepcopy(hybrid())
    raw["field"]["goal_offset_m"][0] = .501
    instance = client(FakeTransport(envelope(raw)))
    policy = HybridFieldDirectPolicy(instance)
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert policy.step_index == 0 and policy.controller is None
    assert policy._pending is None and policy.field is None
    assert instance.history_snapshot() == []


@pytest.mark.parametrize("contact", [False, True])
def test_explicit_completion_stops_before_planning_and_latches_without_another_call(contact, monkeypatch):
    raw = hybrid(contact)
    raw.update(task_complete=True, completion_reason="Bowl released on the plate; gripper is clear.")
    raw["target_region"] = "plate"
    if contact:
        raw["direct"]["actions"] = [[0, 0, 0, 0, 0, -0., -1]] * 5
    else:
        raw["field"].update(motion_mode="hold", gripper_command="open")
    fake = FakeTransport(envelope(raw, reasoning={"effort": "medium"}))
    instance = client(fake, reasoning_effort="medium", stop_on_model_complete=True)
    policy = HybridFieldDirectPolicy(instance)

    def no_action(*args, **kwargs):
        pytest.fail("Completion must not reach the local planner or action adapter")

    monkeypatch.setattr(policy._field_policy, "act", no_action)
    with pytest.raises(ModelTaskComplete) as stopped:
        policy.act(observation())
    assert stopped.value.details["task_complete"] is True
    assert stopped.value.details["compact_prediction"] == raw
    assert stopped.value.details["requested_reasoning_effort"] == "medium"
    assert "task_complete" not in stopped.value.prediction
    assert "plan" not in stopped.value.details
    assert policy.step_index == 0 and policy._pending is None
    assert policy.controller is None and policy.field is None
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["feedback_records"] == 0
    assert len(fake.calls) == 1
    with pytest.raises(ModelTaskComplete):
        policy.act(observation())
    assert len(fake.calls) == 1
    policy.reset()
    assert policy._completion is None and instance.last_task_complete is False


def test_temporary_hold_is_not_completion_and_final_decision_preserves_prior_feedback():
    temporary = hybrid()
    temporary["field"]["motion_mode"] = "hold"
    temporary.update(task_complete=False, completion_reason="")
    done = copy.deepcopy(temporary)
    done.update(task_complete=True, completion_reason="The bowl is released and the gripper is clear.")
    done["target_region"] = "plate"
    fake = FakeTransport(envelope(temporary), envelope(done))
    instance = client(fake, reasoning_effort="medium", stop_on_model_complete=True)
    policy = HybridFieldDirectPolicy(instance)
    obs = observation()
    for step in range(30):
        obs, _, _ = execute(policy, obs, step)
    with pytest.raises(ModelTaskComplete) as stopped:
        policy.act(obs)
    assert stopped.value.details["step"] == 30
    assert instance.history_stats()["feedback_records"] == policy.step_index == 30
    assert instance.history_stats()["observations"] == len(fake.calls) == 2
    assert policy._pending is None
