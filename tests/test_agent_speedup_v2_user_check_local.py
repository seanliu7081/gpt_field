"""Focused real-client/native-policy wiring checks with scripted model replies."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field import agent_speedup_v2 as base
from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_user_check_client import ReviewedV2Client, ReviewedV2Policy
from gpt_field.agent_speedup_v2_user_check_motion import parse_reviewed_repair
from gpt_field.contact_skills import StepView
from gpt_field.observation import PolicyObservation

import agent_speedup_v2_fakes as fakes
from test_agent_speedup_v2 import FlagObserver, outcome
from test_agent_speedup_v2_user_check_runtime import runner_fixture


def move(*, gripper="keep"):
    return {"why": "Lift in world +z and inspect the new view.", "action": "local_motion", "part": None,
            "step": {"frame": "world", "position_mode": "relative", "xyz_m": [0., 0., .03],
                     "quaternion_wxyz": None, "gripper": gripper, "steps": 1}}


def test_scripted_high_client_native_policy_motion_acknowledgement_fresh_view_and_retry(monkeypatch):
    runner, lock = runner_fixture()
    ctx = runner.ctx
    ctx.memory = {"user_check_gripper_command": "open"}
    failed = runner.states[0]
    failed.status, failed.attempts = "failed", 1
    failure = outcome("supported_on", "failed", "precondition_unmet", {"obj": "A", "grip": ["B"], "ref": "C"})
    failure.evidence["motionless_repeat"] = 2
    retry = {"why": "The measured lift changed the starting pose. Retry.", "action": "retry", "part": None,
             "step": None}
    transport = ScriptedV2Transport(FixedAnswers(repairs=[move(), retry]), effort="high")
    client = ReviewedV2Client(task_prompt=lock.task_prompt, approved_plan=lock.plan(), lock=lock, transport=transport)
    policy = ReviewedV2Policy(client, lock=lock)
    # Start at a real recovery boundary; skip unrelated calibration/perception
    # setup while retaining the production policy's act/ack/dispatch machinery.
    policy.ctx, policy._runner, policy._events = ctx, runner, ctx.events
    policy.fulfiller.ctx, policy.fulfiller.events = ctx, ctx.events
    runner.call_counts = policy.fulfiller.counts
    policy._gen = runner.ask("repair", failed, failure)
    monkeypatch.setattr(policy, "_parts_uv", lambda view: {})
    before = ctx.view.observation
    decision = policy.act(before)
    assert decision.diagnostics["primitive"] == "local_motion"
    assert decision.action[2] > 0 and decision.action[-1] == -1.
    assert len(transport.calls) == 1
    after_state = list(before.state)
    after_state[2] += .01
    image = fakes._png_data_url(np.full((512, 512, 3), 80, dtype=np.uint8))
    after = PolicyObservation(image, image, tuple(after_state), lock.task_prompt)
    feedback = policy.observe_transition(before, decision, after)
    assert feedback["primitive"] == "local_motion"
    policy.act(after)
    assert policy.final_value[0].action == "retry"
    assert len(transport.calls) == 2
    assert all(call["reasoning"]["effort"] == "high" for call in transport.calls)
    fresh = json.loads(transport.calls[1]["input"][0]["content"][0]["text"])
    assert fresh["local_control"]["context"]["xyz_m"][2] == pytest.approx(after_state[2])
    assert fresh["local_control"]["last_feedback"]["executed_steps"] == 1
    assert fresh["local_control"]["last_feedback"]["task_effect"] == "unverified"
    images = [[item["image_url"] for item in call["input"][0]["content"] if item["type"] == "input_image"]
              for call in transport.calls]
    assert images[0] != images[1]
    assert lock.plan() == runner.plan.to_json()
    assert failed.status == "failed"  # selecting retry has not executed a macro


def test_local_close_is_unknown_until_measured_open_and_never_fakes_holding():
    runner, lock = runner_fixture()
    ctx = runner.ctx
    ctx.memory = {"user_check_gripper_command": "open"}
    ctx.cal = SimpleNamespace(gripper=SimpleNamespace(open_gap_mm=40.))
    answer = parse_reviewed_repair(move(gripper="close"))
    primitive = SimpleNamespace(gripper_command="close")
    runner._reconcile_local_contact(answer.step, {"actions": 1}, primitive, runner.states[0])
    assert ctx.holding is None and runner._unknown_local_grasp()
    retry = parse_reviewed_repair({"why": "Retry.", "action": "retry", "part": None, "step": None})
    reply = SimpleNamespace(schema_error=False, answer=retry)
    issues = runner.repair_issues(reply, 0, review=False, failed=runner.states[0])
    assert any(isinstance(issue, dict) and issue["code"] == "skill_precondition" for issue in issues)
    # Current fixture is measured fully open; an acknowledged open can now
    # establish empty resources, with old movable geometry invalidated.
    opening = deepcopy(answer.step)
    opening["gripper"] = "open"
    runner._reconcile_local_contact(opening, {"actions": 1}, SimpleNamespace(gripper_command="open"),
                                    runner.states[0])
    assert not runner._unknown_local_grasp() and ctx.holding is None
    assert ctx.memory["user_check_grasp_state"]["source"] == "measured_full_open"
    assert {"A", "B"} <= runner._awaiting


def test_rejected_open_and_preempted_segment_preserve_existing_observers():
    runner, _ = runner_fixture(holding=True)
    ctx = runner.ctx
    ctx.memory = {"user_check_gripper_command": "close"}
    ctx.monitor.add("held:test", FlagObserver(), view=ctx.view, scope="held")
    ctx.monitor.add("keep:test", FlagObserver(), view=ctx.view, scope="relation")
    rejected = move(gripper="open")
    rejected["step"]["xyz_m"] = [0., 0., .2]
    gen = runner.execute_local_motion(parse_reviewed_repair(rejected), runner.states[0], kind="repair")
    primitive = next(gen)(ctx.view)
    assert primitive.done and primitive.actions_executed == 0
    with pytest.raises(StopIteration):
        gen.send((primitive.outcome(), primitive))
    assert ctx.holding.pid == "A"
    assert set(ctx.monitor.keys()) == {"held:test", "keep:test"}
    gen = runner.execute_local_motion(parse_reviewed_repair(move()), runner.states[0], kind="repair")
    next(gen)
    preempted = {"status": "failed", "reason": base.PREEMPTED_PREFIX + "flag_violated", "actions": 0,
                 "evidence": {"invariant": "keep:test", "report": {}}}
    with pytest.raises(StopIteration):
        gen.send((preempted, None))
    assert set(ctx.monitor.keys()) == {"held:test", "keep:test"}
