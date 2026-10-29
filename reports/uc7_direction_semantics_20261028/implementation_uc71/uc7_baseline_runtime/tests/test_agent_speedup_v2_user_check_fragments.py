"""Focused controller/dispatch contracts; no simulator or model calls."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.contact_skills import StepView
from gpt_field.agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client, validate_agent_decision
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_client import PlanLock


def view(x=0., gap=.08):
    return StepView(SimpleNamespace(state=[x, 0., .8, np.pi, 0., 0., gap/2, -gap/2]))


def segment(skill="move_relative", xyz=None, **changes):
    result = dict(skill=skill, xyz_m=[.04, 0., 0.] if xyz is None else xyz,
                  quaternion_wxyz=None, gripper="keep", profile="transit",
                  max_steps=40, observation_after=False)
    if skill in ("set_gripper", "wait"):
        result["xyz_m"] = None
    result.update(changes)
    return result


def advance(primitive, before, after):
    action = primitive.next_action(before)
    primitive.observe_transition(before, action, after)
    return action


def test_current_measured_state_drives_next_target_and_preserves_closed_gripper():
    p = FragmentSequencePrimitive(view(), {"segments": [segment(), segment()]}, keep_gripper="close")
    near = view(.038)
    assert advance(p, view(), near)[-1] == 1.
    advance(p, near, near)
    assert not p.done
    # The second relative move starts at actual .038, not the first target .040.
    assert p.progress()["target_xyz_m"][0] == pytest.approx(.078)
    assert p.progress()["completed_segment_indices"] == [0]


def test_observation_boundary_never_autoplays_remaining_release():
    p = FragmentSequencePrimitive(view(), {"segments": [segment(observation_after=True),
        segment("set_gripper", gripper="open")]}, keep_gripper="close")
    end = view(.04)
    advance(p, view(), end)
    advance(p, end, end)
    assert p.done and p.reason == "observation_required"
    assert p.gripper_command == "close"
    assert p.outcome()["evidence"]["pending_segment_indices"] == [1]


def test_stall_returns_control_without_executing_following_close():
    start = view()
    p = FragmentSequencePrimitive(start, {"segments": [segment(),
        segment("set_gripper", gripper="close")]}, keep_gripper="open")
    for _ in range(20):
        if p.done:
            break
        assert advance(p, start, start)[-1] == -1.
    assert p.done and p.reason == "segment_stall"
    assert p.progress()["completed_segment_indices"] == []
    assert p.gripper_command == "open"


def test_close_waits_for_stability_without_claiming_grasp():
    start = view(gap=.04)
    p = FragmentSequencePrimitive(start, {"segments": [segment("set_gripper", gripper="close")]})
    for _ in range(11):
        advance(p, start, start)
        assert not p.done
    advance(p, start, start)
    assert p.done
    assert p.outcome()["evidence"]["grasp_success"] == "unknown"


def test_invalid_later_command_rejected_before_any_sequence_action():
    with pytest.raises(ValueError):
        FragmentSequencePrimitive(view(), {"segments": [segment(), segment(gripper="open")]})
    p = FragmentSequencePrimitive(view(), {"segments": [segment("move_to_pose", [4., 0., .8])]})
    assert p.done and p.actions_executed == 0 and p.reason == "target_envelope_rejected"


def test_policy_dispatches_framework_fragment_and_records_current_state():
    plan = {"scene": "One object.", "parts": [
        {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1}],
        "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                    "sense": "any", "keep": [], "until": []}]}
    lock = PlanLock(plan, task_prompt="Hold the object.")
    client = AgentFirstV2Client(task_prompt="Hold the object.", approved_plan=plan, lock=lock)
    decision = validate_agent_decision({"belief": "Unverified grasp.", "plan": "Move and inspect.",
        "why": "Measured approach.", "type": "execute", "motion": None, "macro": None,
        "execution": {"segments": [segment()]}})
    client.request_agent = lambda *_: decision
    policy = AgentFirstV2Policy(client, lock=lock)
    policy._client_checked = True
    before = view().observation
    action = policy.act(before)
    assert action.diagnostics["macro"] == "framework_fragment"
    policy.observe_transition(before, action, view(.01).observation)
    assert policy.agent_stats()["counts"]["fragment_actions"] == 1
    assert policy._context(view(.01))["execution"]["shared_state"]["holding_status"] == "unverified"
