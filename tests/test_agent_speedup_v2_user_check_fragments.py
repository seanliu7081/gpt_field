"""Focused controller/dispatch contracts; no simulator or model calls."""
from copy import deepcopy
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.contact_skills import StepView
from gpt_field.action import TRANSLATION_SCALE_M
from gpt_field.astra import SchemaError
from gpt_field import agent_speedup_v2_user_check_fragments as fragments
from gpt_field.agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client, validate_agent_decision
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_client import PlanLock
from gpt_field.agent_speedup_v2_user_check_routing import TransitObstacleMap


def view(x=0., gap=.08, *, xyz=None):
    position = [x, 0., .8] if xyz is None else list(xyz)
    return StepView(SimpleNamespace(state=[*position, np.pi, 0., 0., gap/2, -gap/2]))


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


def test_null_and_omitted_route_use_default_planner_clear_path_and_complete():
    legacy = FragmentSequencePrimitive(view(), {"segments": [segment()]})
    explicit = FragmentSequencePrimitive(view(), {"segments": [segment(route=None)]})
    for primitive in (legacy, explicit):
        routing = primitive.progress()["segments"][0]["routing"]
        assert routing["planner"]["methods"] == ["direct_segment"]
        assert routing["planner"]["calls"] == 0
        assert routing["path_xyz_m"][-1] == pytest.approx([.04, 0., .8])
    before = view()
    for x in (.01, .025, .04, .04):
        after = view(x)
        assert advance(legacy, before, after) == advance(explicit, before, after)
        assert legacy.done == explicit.done
        before = after
    assert legacy.reason == explicit.reason == "fragment_complete"


@pytest.mark.parametrize("configured,expected_transit,expected_scale", [
    (None, .02, .05), ("0.10", .10, .10),
])
def test_configured_fragment_adapter_preserves_transit_target_and_contact_limit(
        configured, expected_transit, expected_scale):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    if configured is None:
        environment.pop("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M", None)
    else:
        environment["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = configured
    result = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import json
        from types import SimpleNamespace
        import numpy as np
        from gpt_field.action import TRANSLATION_SCALE_M
        from gpt_field.contact_skills import StepView
        from gpt_field.agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive

        def view(position):
            return StepView(SimpleNamespace(state=[*position, np.pi, 0., 0., .04, -.04]))

        def segment(skill, xyz, profile):
            return dict(skill=skill, xyz_m=xyz, quaternion_wxyz=None, gripper="keep",
                        profile=profile, max_steps=80, observation_after=False, route=None)

        before = view([0., 0., .8])
        transit = FragmentSequencePrimitive(before, {"segments": [
            segment("move_relative", [.3, .4, 0.], "transit")]})
        assert not transit.done
        norms = []
        first_normalized = None
        for _ in range(80):
            if transit.done:
                break
            action = transit.next_action(before)
            delta = np.array(action[:3]) * TRANSLATION_SCALE_M
            if first_normalized is None:
                first_normalized = list(action[:3])
            norms.append(float(np.linalg.norm(delta)))
            after = view(before.position + delta)
            transit.observe_transition(before, action, after)
            before = after
        assert transit.done and transit.status == "succeeded", transit.progress()
        assert np.linalg.norm(before.position - [.3, .4, .8]) <= .004

        contact = FragmentSequencePrimitive(view([0., 0., .8]), {"segments": [
            segment("approach", [.03, .04, 0.], "contact")]})
        contact_delta = np.array(contact.next_action(view([0., 0., .8]))[:3]) * TRANSLATION_SCALE_M
        assert "routing" not in contact.progress()["segments"][0]
        outside = FragmentSequencePrimitive(view([0., 0., .8]), {"segments": [
            segment("move_to_pose", [.5001, 0., .8], "transit")]})
        assert outside.done and outside.reason == "target_envelope_rejected"
        assert outside.actions_executed == 0
        print(json.dumps({"scale_m": TRANSLATION_SCALE_M,
                          "first_normalized": first_normalized,
                          "first_transit_m": norms[0], "max_transit_m": max(norms),
                          "final_position": before.position.tolist(),
                          "contact_m": float(np.linalg.norm(contact_delta))}))
    """)], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    measured = json.loads(result.stdout)
    assert measured["scale_m"] == expected_scale
    assert measured["first_transit_m"] == pytest.approx(expected_transit)
    assert measured["max_transit_m"] == pytest.approx(expected_transit)
    assert measured["first_normalized"] == pytest.approx(
        np.array([.6, .8, 0.]) * expected_transit / expected_scale)
    assert measured["final_position"] == pytest.approx([.3, .4, .8], abs=.004)
    assert measured["contact_m"] == pytest.approx(.005)


def test_route_progress_can_move_away_without_completing_or_stalling_model_goal(monkeypatch):
    class Detour:
        """First leg of a route around an obstruction, with a temporary waypoint."""
        def __init__(self, *args, **kwargs):
            pass

        def command(self, position):
            return (-.002, 0., 0.), {}

        def progress(self, position):
            return .2 + float(position[0])

        def diagnostics(self):
            return {"path_xyz_m": [[0., 0., .8], [-.01, 0., .8], [.04, 0., .8]],
                    "path_length_m": .2, "planner": "test_detour"}

    monkeypatch.setattr(fragments, "TransitRoute", Detour)
    p = FragmentSequencePrimitive(view(), {"segments": [
        segment(route={"avoid_boxes": [{"lower_xyz_m": [.2, .2, .78],
                                         "upper_xyz_m": [.24, .24, .82]}], "clearance_m": .01}),
        segment("set_gripper", gripper="close")]})
    before = view()
    for n in range(1, 21):
        after = view(-.002 * n)
        assert advance(p, before, after)[-1] == -1.
        assert not p.done
        assert p.progress()["target_xyz_m"] == pytest.approx([.04, 0., .8])
        assert p.progress()["completed_segment_indices"] == []
        before = after
    # The temporary route waypoint was passed, and final-goal distance grew,
    # but along-route distance decreased: neither arrival nor a stall occurred.
    assert p.progress()["position_error_m"] == pytest.approx(.08)


@pytest.mark.parametrize("shared", [False, True])
def test_unavailable_route_returns_control_without_following_close(shared):
    route = {"avoid_boxes": [{"lower_xyz_m": [.03, -.02, .78],
                              "upper_xyz_m": [.05, .02, .82]}], "clearance_m": .005}
    scene = TransitObstacleMap()
    scene.update(route, step=3)
    p = FragmentSequencePrimitive(view(), {"segments": [segment(route=None if shared else route),
        segment("set_gripper", gripper="close")]}, obstacle_map=scene.snapshot() if shared else None)
    assert p.done and p.status == "failed"
    assert p.actions_executed == 0
    assert p.gripper_command == "open"
    assert p.progress()["completed_segment_indices"] == []
    assert p.progress()["pending_segment_indices"] == [1]


def test_stage_route_override_does_not_change_shared_map_for_next_grant():
    scene = TransitObstacleMap()
    scene.update({"avoid_boxes": [{"lower_xyz_m": [.03, -.02, .78],
                                  "upper_xyz_m": [.05, .02, .82]}], "clearance_m": .005}, step=3)
    snapshot = scene.snapshot()
    override = {"avoid_boxes": [{"lower_xyz_m": [.2, .2, .9],
                                 "upper_xyz_m": [.3, .3, 1.]}], "clearance_m": .01}
    temporary = FragmentSequencePrimitive(view(), {"segments": [segment(route=override)]},
                                          obstacle_map=snapshot)
    assert not temporary.done
    assert temporary.progress()["segments"][0]["routing"]["planner"]["methods"] == ["direct_segment"]
    assert scene.snapshot() == snapshot
    reused = FragmentSequencePrimitive(view(), {"segments": [segment(route=None)]},
                                       obstacle_map=scene.snapshot())
    assert reused.done and reused.reason == "route_unavailable"
    assert reused.actions_executed == 0


def test_contact_gripper_and_wait_ignore_shared_obstacles(monkeypatch):
    def unexpected_route(*args, **kwargs):
        pytest.fail("contact, gripper and wait stages must not enter the transit planner")

    monkeypatch.setattr(fragments, "TransitRoute", unexpected_route)
    scene = TransitObstacleMap()
    scene.update({"avoid_boxes": [{"lower_xyz_m": [-.1, -.1, .7],
                                  "upper_xyz_m": [.1, .1, .9]}], "clearance_m": .01}, step=0)
    primitive = FragmentSequencePrimitive(view(), {"segments": [
        segment("approach", profile="contact", route=None),
        segment("set_gripper", gripper="close", route=None),
        segment("wait", max_steps=2, route=None),
    ]}, obstacle_map=scene.snapshot())
    before = view()
    for _ in range(60):
        if primitive.done:
            break
        action = primitive.next_action(before)
        after = view(xyz=before.position + np.asarray(action[:3]) * TRANSLATION_SCALE_M)
        primitive.observe_transition(before, action, after)
        before = after
    assert primitive.done and primitive.status == "succeeded"
    assert primitive.progress()["completed_segment_indices"] == [0, 1, 2]
    assert primitive.gripper_command == "close"
    assert primitive.progress()["grasp_success"] == "unknown"


def test_policy_reuses_obstacles_for_real_detours_across_grants_then_resets():
    plan = {"scene": "One object.", "parts": [
        {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1}],
        "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                    "sense": "any", "keep": [], "until": []}]}
    lock = PlanLock(plan, task_prompt="Hold the object.")
    client = AgentFirstV2Client(task_prompt="Hold the object.", approved_plan=plan, lock=lock)
    box = {"lower_xyz_m": [.15, -.035, .755], "upper_xyz_m": [.25, .035, .845]}
    shared = {"avoid_boxes": [box], "clearance_m": .015}
    responses = iter(validate_agent_decision({
        "belief": "Estimated obstacle between the targets.", "plan": "Move around it.",
        "why": "Preserve the obstacle estimate across observations.", "type": "execute",
        "motion": None, "macro": None, "obstacle_map": obstacle_map,
        "execution": {"segments": [segment("move_to_pose", target, max_steps=120, route=None)]},
    }) for target, obstacle_map in [([.4, 0., .8], shared), ([0., 0., .8], None)])
    contexts = []

    def request_agent(_view, context):
        contexts.append(deepcopy(context))
        return next(responses)

    client.request_agent = request_agent
    policy = AgentFirstV2Policy(client, lock=lock)
    policy._client_checked = True
    before = view()
    points, routing_by_grant = [before.position.copy()], {}
    for _ in range(240):
        decision = policy.act(before.observation)
        grant = policy.agent_stats()["counts"]["fragment_grants"]
        routing_by_grant.setdefault(grant, deepcopy(policy._primitive.progress()["segments"][0]["routing"]))
        after = view(xyz=before.position + np.asarray(decision.action[:3]) * TRANSLATION_SCALE_M)
        policy.observe_transition(before.observation, decision, after.observation)
        points.append(after.position.copy())
        before = after
        if grant == 2 and policy._need_agent:
            break
    assert len(contexts) == 2
    assert policy._last_fragment_result["reason"] == "fragment_complete"
    assert before.position == pytest.approx([0., 0., .8], abs=.004)
    assert contexts[1]["execution"]["obstacle_map"]["avoid_boxes"] == [box]
    assert contexts[1]["execution"]["obstacle_map"]["source"] == "model_estimate"
    for grant in (1, 2):
        assert routing_by_grant[grant]["planner"]["calls"] > 0
    low = np.asarray(box["lower_xyz_m"]) - shared["clearance_m"]
    high = np.asarray(box["upper_xyz_m"]) + shared["clearance_m"]
    for start, end in zip(points[:-1], points[1:]):
        samples = np.linspace(start, end, int(np.ceil(np.linalg.norm(end - start) / .002)) + 1)
        assert not np.any(np.all((samples >= low) & (samples <= high), axis=1))
    policy.reset()
    reset_map = policy._context(view())["execution"]["obstacle_map"]
    assert reset_map["avoid_boxes"] == []
    assert reset_map["source"] == "uninitialized"
    assert contexts[1]["execution"]["obstacle_map"]["avoid_boxes"] == [box]


def test_blocked_final_measurement_cannot_complete_route_and_start_close():
    route = {"avoid_boxes": [{"lower_xyz_m": [.043, -.02, .78],
                              "upper_xyz_m": [.06, .02, .82]}], "clearance_m": 0.}
    p = FragmentSequencePrimitive(view(), {"segments": [segment(route=route),
        segment("set_gripper", gripper="close")]})
    goal = view(.04)
    advance(p, view(), goal)
    assert not p.done
    # Still within the 4 mm arrival tolerance, but now inside the supplied box.
    advance(p, goal, view(.0435))
    assert p.done and p.reason == "route_unavailable"
    assert p.gripper_command == "open"
    assert p.progress()["completed_segment_indices"] == []


def test_executive_wire_schema_accepts_transit_route_but_rejects_contact_route():
    plan = {"scene": "One object.", "parts": [
        {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1}],
        "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                    "sense": "any", "keep": [], "until": []}]}
    client = AgentFirstV2Client(task_prompt="Hold the object.", approved_plan=plan,
                               lock=PlanLock(plan, task_prompt="Hold the object."))
    wire_segment = client._schemas["agent"]["properties"]["execution"]["anyOf"][0]["properties"]["segments"]["items"]
    assert "route" in wire_segment["required"]
    assert {"type": "null"} in wire_segment["properties"]["route"]["anyOf"]
    route = {"avoid_boxes": [{"lower_xyz_m": [.01, -.01, .79],
                              "upper_xyz_m": [.03, .01, .81]}], "clearance_m": .005}
    decision = {"belief": "Estimated obstacle.", "plan": "Transit around it.",
                "why": "Reach the selected pose.", "type": "execute", "motion": None, "macro": None,
                "execution": {"segments": [segment(route=route)]}}
    client._request_kind = "agent"
    assert client.validate_response(decision) == validate_agent_decision(decision)
    assert client.validate_response(decision)["execution"]["segments"][0]["route"] == route
    for skill in ("approach", "move_relative"):
        decision["execution"] = {"segments": [segment(skill, profile="contact", route=route)]}
        with pytest.raises(SchemaError, match="route"):
            client.validate_response(decision)
