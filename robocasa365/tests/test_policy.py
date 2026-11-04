"""Pure offline mobile adapter contracts: no simulator, provider or model calls."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rc365.policy import (
    MobilePrimitive, RoboCasa365Client, RoboCasa365Policy,
    mobile_schema, rc365_schema, validate_decision, validate_mobile,
)
from gpt_field.astra import SchemaError
from gpt_field.contact_skills import StepView
from gpt_field.agent_speedup_v2_user_check_agent_client import agent_schema
from gpt_field.agent_speedup_v2_user_check_client import PlanLock


def command(steps=3, **changes):
    result = {"base": [.1, -.1, .05], "torso": .04, "steps": steps}
    result.update(changes)
    return result


def answer(kind="mobile", **changes):
    result = {"belief": "The route is visibly clear but attachment is uncertain.",
              "plan": "Reposition briefly and inspect current images.",
              "why": "Improve reach from measured robot feedback.",
              "type": kind, "motion": None, "macro": None, "execution": None,
              "obstacle_map": None, "mobile": command() if kind == "mobile" else None}
    result.update(changes)
    return result


def view(x=0., torso=.1):
    # Deliberate forbidden unrelated attributes expose accidental whole-observation copies.
    obs = SimpleNamespace(
        state=[x, 0., .8, np.pi, 0., 0., .02, -.02],
        mobile_state={"source": "robot_proprioception", "base_position_m": [x, 0., 0.],
                      "base_quaternion_wxyz": [1., 0., 0., 0.], "torso_qpos": [torso]},
        object_positions={"FORBIDDEN_OBJECT_TRUTH": [8., 9., 10.]}, success=True,
    )
    return StepView(obs)


def make_policy(response, gripper="close"):
    plan = {"scene": "One object.", "parts": [
        {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1}],
        "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                    "sense": "any", "keep": [], "until": []}]}
    lock = PlanLock(plan, task_prompt="Hold the object.")
    client = RoboCasa365Client(task_prompt="Hold the object.", approved_plan=plan, lock=lock)
    calls = []
    def fake_request(current, context):
        calls.append(deepcopy(context))
        return validate_decision(deepcopy(response))
    client.request_agent = fake_request
    policy = RoboCasa365Policy(client, lock=lock)
    policy._client_checked = True
    policy.ctx.memory["user_check_gripper_command"] = gripper
    return policy, calls


def test_mobile_schema_is_strict_and_does_not_modify_baseline():
    original = agent_schema()
    schema = rc365_schema()
    assert agent_schema() == original
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(original["required"]) | {"mobile"}
    assert schema["properties"]["type"]["enum"][-1] == "mobile"
    assert mobile_schema()["additionalProperties"] is False
    assert set(mobile_schema()["required"]) == {"base", "torso", "steps"}


@pytest.mark.parametrize("bad", [
    None, [], {}, {"base": [0, 0, 0], "torso": 0},
    command(extra=True), command(base=[0, 0]), command(base=[0, 0, 0, 0]),
    command(base="0,0,0"), command(base=[True, 0, 0]),
    command(base=[np.bool_(False), 0, 0]), command(base=[".1", 0, 0]),
    command(base=[float("nan"), 0, 0]), command(base=[0, float("inf"), 0]),
    command(base=[0, 0, -.20001]), command(torso=.20001), command(torso=True),
    command(torso=float("nan")), command(steps=0), command(steps=6),
    command(steps=1.0), command(steps=True), command(steps=np.int64(2)),
])
def test_invalid_mobile_commands_rejected_before_primitive(bad):
    with pytest.raises(ValueError):
        validate_mobile(bad)
    with pytest.raises(SchemaError):
        validate_decision(answer(mobile=bad))


@pytest.mark.parametrize("steps", range(1, 6))
def test_exact_chunk_length_keeps_gripper_and_requires_matching_ack(steps):
    before = view()
    primitive = MobilePrimitive(before, command(steps=steps), "close")
    for index in range(steps):
        action = primitive.next_action(before)
        assert len(action) == 7 and action[-1] == 1.
        assert action[:3] == (0., 0., 0.)
        # Mobile overrides the arm-only OSC refresh: native mobile must get exact zero arm delta.
        assert action[3:6] == (0., 0., 0.)
        with pytest.raises(RuntimeError, match="observe_transition"):
            primitive.next_action(before)
        wrong = list(action); wrong[-1] = -1.
        with pytest.raises(ValueError, match="exact pending action"):
            primitive.observe_transition(before, wrong, before)
        after = view(x=(index + 1) * .001, torso=.1 + index * .001)
        feedback = primitive.observe_transition(before, action, after)
        assert primitive.done is (index + 1 == steps)
        assert feedback["after_mobile_state"] == after.observation.mobile_state
        before = after
    assert primitive.outcome()["actions"] == steps
    assert primitive.reason == "bounded_mobile_chunk_complete"
    with pytest.raises(RuntimeError, match="finished"):
        primitive.next_action(before)


@pytest.mark.parametrize("changes", [
    {"motion": {"x": 1}}, {"macro": {"step_index": 0}},
    {"execution": {"segments": []}}, {"unexpected": 1},
    {"belief": ""}, {"why": True}, {"type": "teleport"},
])
def test_mobile_rejects_combined_actions_and_invalid_envelope(changes):
    with pytest.raises(SchemaError):
        validate_decision(answer(**changes))


def test_nonmobile_requires_null_and_all_responses_require_mobile_field():
    for kind in ("observe", "finish"):
        assert validate_decision(answer(kind))["mobile"] is None
        with pytest.raises(SchemaError):
            validate_decision(answer(kind, mobile=command()))
    missing = answer(); del missing["mobile"]
    with pytest.raises(SchemaError):
        validate_decision(missing)


def test_validation_does_not_mutate_caller_command():
    source = answer(mobile=command(base=[-.2, 0, .2], torso=-.2, steps=1))
    snapshot = deepcopy(source)
    parsed = validate_decision(source)
    assert source == snapshot
    parsed["mobile"]["base"][0] = 0.
    assert source == snapshot


def test_feedback_is_robot_measurement_without_claiming_progress_or_task_success():
    before, after = view(), view(.004, .105)
    primitive = MobilePrimitive(before, command(steps=1), "open")
    before_snapshot = deepcopy(before.observation.mobile_state)
    action = primitive.next_action(before)
    assert action[-1] == -1.
    feedback = primitive.observe_transition(before, action, after)
    assert feedback["before_mobile_state"] == before_snapshot
    assert feedback["after_mobile_state"] == after.observation.mobile_state
    assert feedback["semantic_result"].startswith("unknown")
    assert "FORBIDDEN_OBJECT_TRUTH" not in json.dumps(feedback)
    assert "task_success" not in feedback
    after.observation.mobile_state["base_position_m"][0] = 999.
    assert primitive.outcome()["evidence"]["after_mobile_state"]["base_position_m"][0] == .004


@pytest.mark.parametrize("gripper,value", [("open", -1.), ("close", 1.)])
def test_policy_emits_arm7_plus_mobile_diagnostics_and_reobserves_after_chunk(gripper, value):
    policy, calls = make_policy(answer(mobile=command(steps=3)), gripper)
    before = view().observation
    for step in range(3):
        decision = policy.act(before)
        assert len(decision.action) == 7 and decision.action[-1] == value
        assert decision.diagnostics["mobile_command"] == {"base": [.1, -.1, .05], "torso": .04}
        assert decision.diagnostics["execution_kind"] == "mobile_native:bounded_chunk"
        after = view((step + 1) * .001).observation
        result = policy.observe_transition(before, decision, after)
        assert result["mobile_state"] == after.mobile_state
        assert result["task_success"] == "not_available_to_policy"
        assert policy.ctx.memory["user_check_gripper_command"] == gripper
        before = after
    assert len(calls) == 1
    assert policy._need_agent is True and policy._primitive is None
    assert policy.agent_stats()["counts"]["mobile_actions"] == 3
    assert policy.agent_stats()["counts"]["direct_actions"] == 0
    assert policy._feedback["action"] == "mobile"
    assert policy._feedback["status"] == "chunk_complete"
    assert policy._feedback["executed_steps"] == 3
    assert policy._feedback["semantic_result"].startswith("unknown")
    assert "FORBIDDEN_OBJECT_TRUTH" not in json.dumps(calls)
    assert calls[0]["measured_state"]["mobile"] == view().observation.mobile_state
    # A fourth physical command needs another fake executive decision.
    policy.act(before)
    assert len(calls) == 2
