"""Mocked paired contact routing: real local execution, no live model/simulator."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.contact_diagnostic_policy import (
    ChatGPTPlanContactDiagnosticClient, ContactDiagnosticPolicy,
    contact_diagnostic_schema, _memory_transcript,
)
from gpt_field.speed_first_v4 import ChatGPTPlanSpeedFirstV4Client, FINE_GRID
from test_field_history import FakeTransport, envelope, observation
from test_speed_first_v2 import records, tick
from test_speed_first_v4 import intent


def wire(raw=None, scope="contact_alignment", reason="Correct visible hand-to-target alignment"):
    return {"decision": {"scope": scope, "scope_reason": reason,
                         "response": intent("pregrasp") if raw is None else copy.deepcopy(raw)}}


def controller(*values, arm="current", **kwargs):
    fake = FakeTransport(*(envelope(value, id=f"contact-{index}", model="gpt-6-astra",
                                   reasoning={"effort": "medium"})
                           for index, value in enumerate(values)))
    client = ChatGPTPlanContactDiagnosticClient(contact_arm=arm,
        token_provider=lambda: "offline-contact", transport=fake, **kwargs)
    return ContactDiagnosticPolicy(client), fake


def test_defaults_remain_exact_and_diagnostic_arms_share_inputs_and_nonrouting_instructions():
    base = ChatGPTPlanSpeedFirstV4Client(token_provider=lambda: pytest.fail("No token needed"))
    original = copy.deepcopy(base.build_payload(observation().to_dict()))
    payloads = []
    for arm in ("current", "direct5"):
        policy, fake = controller(arm=arm)
        payloads.append(policy.client.build_payload(observation().to_dict()))
        assert fake.calls == []
        assert policy.client.reasoning_effort == "medium"
        assert payloads[-1]["reasoning"] == {"effort": "medium"}
        assert "SHORT CONTACT DIAGNOSTIC ONLY" in payloads[-1]["instructions"]
        assert "controller converts" not in payloads[-1]["instructions"]
    assert payloads[0]["input"] == payloads[1]["input"]
    assert payloads[0]["instructions"].split("CONTACT ROUTING ARM")[0] == payloads[1]["instructions"].split("CONTACT ROUTING ARM")[0]
    assert base.build_payload(observation().to_dict()) == original
    assert "decision" not in original["text"]["format"]["schema"]["properties"]


def test_schema_enforces_direct5_contact_but_keeps_free_space_field():
    schemas = [contact_diagnostic_schema(contact_arm=arm) for arm in ("current", "direct5")]
    branches = [{b["properties"]["scope"]["enum"][0]: b for b in s["properties"]["decision"]["anyOf"]}
                for s in schemas]
    for scope in ("free_space_reposition", "gripper_operation", "engagement_check"):
        assert branches[0][scope] == branches[1][scope]
    forced = branches[1]["contact_alignment"]["properties"]["response"]["properties"]
    assert forced["controller"] == {"type": "string", "enum": ["contact_direct"]}
    assert forced["field"] == {"type": "null"}
    assert forced["direct"] == {"$ref": "#/$defs/direct_actions"}
    assert schemas[1]["$defs"]["direct_actions"]["properties"]["actions"]["minItems"] == 5
    assert forced["operation"]["properties"]["local_action_budget"] == {"type": "integer", "enum": [5]}
    assert schemas[0]["$defs"] == schemas[1]["$defs"]
    clients = [controller(arm=arm)[0].client for arm in ("current", "direct5")]
    def accepts(client, value):
        try:
            client.validate_response(value, FINE_GRID)
            return True
        except SchemaError:
            return False
    field = wire()
    assert accepts(clients[0], field)
    assert not accepts(clients[1], field)
    ordinary = wire(intent("pregrasp", direct=True))
    assert all(accepts(client, ordinary) for client in clients)
    free = wire(intent("move"), "free_space_reposition")
    assert all(accepts(client, free) for client in clients)
    # Renaming the operation cannot bypass its declared contact scope.
    for kind in ("move", "recover"):
        renamed = wire(intent(kind))
        assert accepts(clients[0], renamed)
        assert not accepts(clients[1], renamed)
    # A declared pregrasp cannot be relabeled as free space in either arm.
    wrong_scope = wire(intent("pregrasp"), "free_space_reposition")
    assert all(not accepts(client, wrong_scope) for client in clients)


def test_runtime_rejects_route_before_history_commit_without_conversion_or_requery():
    p, fake = controller(wire(), arm="direct5")
    with pytest.raises(SchemaError, match="ordinary Direct"):
        p.act(observation())
    assert len(fake.calls) == 1
    assert p.client.history_snapshot() == []
    assert p.client.last_contact_diagnostic is None
    assert p.step_index == 0 and p.contact_diagnostic_stats()["accepted_responses"] == 0
    assert p.client.budget_snapshot()["api_calls"] == 1


def test_direct_contact_actions_execute_exactly_five_and_free_space_retains_field():
    raw = intent("pregrasp", direct=True)
    raw["direct"]["actions"] = [[.012*i, -.011, .007, .008, -.006, .002, -1.] for i in range(1, 6)]
    p, fake = controller(wire(raw), wire(intent("move"), "free_space_reposition"), arm="direct5")
    obs = observation()
    assert p.needs_model_request()
    for index, expected in enumerate(raw["direct"]["actions"]):
        obs, decision, _ = tick(p, obs)
        assert decision.action == tuple(expected)
        assert decision.diagnostics["execution_kind"] == "direct_five_actions"
        assert decision.diagnostics["contact_diagnostic"]["scope"] == "contact_alignment"
        assert len(fake.calls) == 1
        assert p.needs_model_request() is (index == 4)
    obs, decision, _ = tick(p, obs)
    assert len(fake.calls) == 2 and decision.diagnostics["execution_kind"] == "field_tracking"
    stats = p.contact_diagnostic_stats()
    assert stats["ordinary_direct_actions"] == 5 and stats["field_actions"] == 1
    assert stats["contact_alignment_direct_responses"] == 1
    assert stats["free_space_responses_after_contact_started"] == 1
    assert stats["closure_operations_started"] == stats["completed_close_attempts"] == 0


def test_current_contact_field_remains_field_and_renaming_is_visible():
    p, fake = controller(wire(intent("move", goal=(.06, 0., 0.))))
    obs, decision, _ = tick(p, observation(), fraction=.1)
    assert decision.diagnostics["execution_kind"] == "field_tracking"
    stats = p.contact_diagnostic_stats()
    assert stats["contact_alignment_field_responses"] == 1
    assert stats["contact_alignment_with_move_or_recover_kind"] == 1
    assert not stats["scope_independently_verified"]
    assert len(fake.calls) == 1 and p.client.last_contact_wire["decision"]["scope"] == "contact_alignment"


@pytest.mark.parametrize("arm", ["current", "direct5"])
def test_close_check_finishes_without_model_request_or_canned_check_and_counts_only_completed_feedback(arm):
    check = {"offset_xyz_m": [.0, -.02, .0], "keep_orientation": True}
    raw = intent("grasp_close", budget=60, continuation=True, verify=check)
    p, fake = controller(wire(raw, "gripper_operation"), arm=arm)
    obs = observation()
    decision = p.act(obs)
    assert p.contact_diagnostic_stats()["closure_requests"] == 1
    assert p.contact_diagnostic_stats()["closure_operations_started"] == 0
    with pytest.raises(RuntimeError, match="complete feedback"):
        p.needs_model_request()
    bad_before = replace(obs, state=(.2, *obs.state[1:]))
    before_stats = p.contact_diagnostic_stats()
    with pytest.raises(ValueError, match="exact pending"):
        p.observe_transition(bad_before, decision, obs)
    assert p.contact_diagnostic_stats() == before_stats
    after = replace(obs, state=(*obs.state[:6], .015, -.015))
    p.observe_transition(obs, decision, after)
    obs = after
    for _ in range(14):
        obs, decision, _ = tick(p, obs, fingers=(.015, -.015))
        assert not p.needs_model_request()
    stats = p.contact_diagnostic_stats()
    assert stats["closure_operations_started"] == stats["closure_operations_completed"] == 1
    assert stats["completed_close_attempts"] == 0
    before_check = np.asarray(obs.state[:3])
    obs, decision, _ = tick(p, obs)
    assert decision.diagnostics["execution_kind"] == "local_close_verification"
    assert decision.action[1] < 0 and decision.action[0] == decision.action[2] == 0
    np.testing.assert_allclose(p._nominal_goal, before_check + check["offset_xyz_m"])
    stats = p.contact_diagnostic_stats()
    assert stats["completed_close_attempts"] == stats["completed_close_check_attempts"] == 1
    assert stats["verification_actions"] == 1
    assert p.needs_model_request() and len(fake.calls) == 1


def test_close_without_check_counts_at_settling_and_unsettled_check_counts_as_failed_attempt():
    p, _ = controller(wire(intent("grasp_close"), "gripper_operation"))
    obs = observation()
    for _ in range(15):
        obs, _, _ = tick(p, obs, fingers=(.015, -.015))
    stats = p.contact_diagnostic_stats()
    assert stats["completed_close_attempts"] == 1
    assert stats["completed_close_check_attempts"] == 0 and p.needs_model_request()
    check = {"offset_xyz_m": [0., .02, 0.], "keep_orientation": True}
    p, fake = controller(wire(intent("grasp_close", budget=60, continuation=True, verify=check), "gripper_operation"))
    obs = observation()
    for _ in range(30):
        obs, _, _ = tick(p, obs)
    stats = p.contact_diagnostic_stats()
    assert stats["completed_close_attempts"] == 1 and stats["verification_actions"] == 0
    assert stats["responses"][0]["close_check_end_reason"] == "unsettled_close"
    assert p.needs_model_request() and len(fake.calls) == 1


def test_accepted_wire_and_legacy_attempt_memory_both_preserved_without_recursive_wrapper():
    close = wire(intent("grasp_close"), "gripper_operation")
    release = wire(intent("release"), "gripper_operation")
    release["decision"]["response"]["target_region"] = "black_bowl"
    release["decision"]["response"]["operation"]["target_id"] = "black_bowl"
    p, fake = controller(close, release)
    obs = observation()
    for _ in range(15):
        obs, _, _ = tick(p, obs, fingers=(.015, -.015))
    accepted_before = copy.deepcopy(p.client.history_snapshot())
    obs, _, _ = tick(p, obs)
    output = [v for v in records(p.client.history_snapshot()) if "decision" in v]
    assert output == [close, release]
    memory_rows = [v for v in records(fake.calls[1]["input"]) if v.get("type") == "attempt_memory"]
    assert len(memory_rows) == 1
    memory = memory_rows[0]
    assert memory["attempts"][-1]["close_operation_requests"] == 1
    assert memory["attempts"][-1]["close_operation_actions"] == 15
    assert _memory_transcript(accepted_before) != accepted_before
    assert [v for v in records(accepted_before) if "decision" in v] == [close]
    assert not [v for v in records(p.client.history_snapshot()) if v.get("type") == "attempt_memory"]


def test_wrapper_preserves_field_block_compaction_old_digests_and_exact_recent_wire():
    values = [wire(intent("move", goal=(.01, 0., 0.), identifier=f"move-{i}"),
                   "free_space_reposition", reason=f"Reposition with clearance {i}") for i in range(6)]
    p, fake = controller(*values)
    obs = observation()
    for _ in values:
        obs, _, _ = tick(p, obs)
        assert p.needs_model_request()
    presented = records(fake.calls[-1]["input"])
    digest = next(row for row in presented if row.get("type") == "older_history_digest")
    assert all(row["controller"] == "coarse_field" for row in digest["records"])
    assert all(row["historical_field_intent"]["goal_offset_m"] == [.01, 0., 0.]
               for row in digest["records"])
    assert sum(row.get("type") == "execution_block_summary" for row in presented) == 3
    raw_recent = [part["text"] for item in fake.calls[-1]["input"]
                  for part in item.get("content", []) if part.get("type") == "output_text"]
    assert raw_recent == [json.dumps(value) for value in values[2:5]]
    assert [v for v in records(p.client.history_snapshot()) if "decision" in v] == values
    assert "_contact_wire_slot" not in json.dumps(fake.calls)
    assert "_contact_wire_slot" not in json.dumps(p.client.history_snapshot())


@pytest.mark.parametrize("mutation", [
    lambda v: v["decision"].update(scope="free_space_reposition"),
    lambda v: v["decision"].update(scope=[]),
    lambda v: v["decision"]["response"].update(target_region="plate"),
    lambda v: v["decision"]["response"]["operation"].update(target_id="plate"),
    lambda v: v["decision"]["response"]["field"].update(gripper_command="close"),
    lambda v: v.update(object_pose=[0., 0., 0.]),
])
def test_bad_wrapper_scope_target_or_contact_semantics_is_atomic(mutation):
    p, _ = controller()
    good = wire()
    p.client.validate_response(good, FINE_GRID)
    before = copy.deepcopy(p.client.last_contact_wire)
    bad = copy.deepcopy(good)
    mutation(bad)
    with pytest.raises(SchemaError):
        p.client.validate_response(bad, FINE_GRID)
    assert p.client.last_contact_wire == before
    assert p.client.last_compact_prediction == good["decision"]["response"]
    assert not p.client.history_snapshot()


def test_invalid_source_sensors_demonstration_or_effort_never_reaches_transport():
    p, fake = controller()
    bad = observation().to_dict()
    bad["object_pose"] = [0., 0., 0.]
    with pytest.raises(ValueError):
        p.client.build_payload(bad)
    assert fake.calls == []
    for kwargs in ({"demonstration": object()}, {"reasoning_effort": "high"},
                   {"diagnostic_target": "drawer_handle", "task_key": "bowl_plate"}):
        with pytest.raises((ValueError, TypeError)):
            ChatGPTPlanContactDiagnosticClient(token_provider=lambda: pytest.fail("No credentials"), **kwargs)


def test_standalone_check_is_bounded_but_never_generated_by_local_policy():
    p, _ = controller()
    too_far = wire(intent("verify_grasp", goal=(.031, 0., 0.)), "engagement_check")
    with pytest.raises(SchemaError, match="3 cm"):
        p.client.validate_response(too_far, FINE_GRID)
    direct = intent("verify_grasp", direct=True)
    direct["direct"]["actions"] = [[.2, 0., 0., 0., 0., 0., 1.]] * 5
    with pytest.raises(SchemaError, match="3 cm"):
        p.client.validate_response(wire(direct, "engagement_check"), FINE_GRID)
    assert p.client.last_contact_wire is None


def test_error_has_no_fallback_request_and_retains_accepted_history():
    raw = wire(intent("pregrasp", direct=True))
    p, fake = controller(raw, arm="direct5")
    obs = observation()
    for _ in range(5):
        obs, _, _ = tick(p, obs)
    accepted = p.client.history_snapshot()
    stats = p.contact_diagnostic_stats()
    fake.responses.append(APIError("offline unavailable"))
    with pytest.raises(APIError):
        p.act(obs)
    assert len(fake.calls) == 2 and p.client.history_snapshot() == accepted
    assert p.contact_diagnostic_stats() == stats
