"""Focused queue/actual-action checks with public observations; no simulator."""
from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.action import TRANSLATION_SCALE_M
from gpt_field.astra import SchemaError
from gpt_field.contact_skills import StepView
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame
from gpt_field.foundation_vision_harness_V2 import executor


def obs(xyz=(0., 0., .2), gap=.04):
    return SimpleNamespace(state=np.r_[xyz, [0., 0., 0., gap, -gap]],
        joint_position=np.zeros(7), joint_velocity=np.zeros(7), geometry={}, prompt="Move the object.")


def frame(view, geometry, frame_id, captured_at_s):
    K = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    return VisionFrame(frame_id, captured_at_s, tuple(CameraFrame(name,
        np.zeros((32, 32, 3), np.uint8), K, np.eye(4), captured_at_s) for name in ("external", "wrist")))


class Service:
    def __init__(self):
        self.snapshot = None
        self.token = 0
        self.stats, self.errors, self.submissions = {}, [], []
    def submit(self, view, targets, *, force=False, frame_id=None):
        self.token += 1
        self.submissions.append((frame_id, force))
        points = np.array([[x, y, .2+z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
        body = ObjectGeometry("body", "object", points, np.zeros_like(points), "observed", {},
                              {"observed_frame_id": frame_id, "sigma_m": [.002]*3})
        self.snapshot = SceneSnapshot(frame_id, monotonic(), monotonic(), {"body": body},
            np.empty((0, 3)), None, frame(view, {}, frame_id, monotonic()),
            evidence={"revision": self.token, "request_token": self.token})
        return self.token
    def latest(self): return self.snapshot
    def wait(self, token, timeout_s): return self.snapshot
    def invalidate(self, ids, reason): pass
    def close(self): pass


def candidate(delta=(.006, 0., 0.), skill="move_relative", gripper="keep"):
    return {"skill": skill, "xyz_m": None if skill == "set_gripper" else list(delta),
        "quaternion_wxyz": None, "gripper": gripper, "profile": "contact", "max_steps": 30}


def segment(index, candidates=None, checkpoint="none"):
    return {"id": str(index), "object": "body", "anchor": None, "contact_mode": "grasp",
        "contact_parts": ["finger", "pad"], "keep_upright": False, "checkpoint": checkpoint,
        "candidates": candidates or [candidate()]}


class Client:
    def __init__(self, segments, final="finish"):
        self.segments, self.final, self.contexts = segments, final, []
    def request_harness(self, view, context):
        self.contexts.append(deepcopy(context))
        if len(self.contexts) == 1:
            return {"status": "plan", "why": "Follow the queue.", "targets": [],
                    "segments": deepcopy(self.segments), "completion_conditions": []}
        return {"status": self.final, "why": "Observed terminal state.", "targets": [],
                "segments": [], "completion_conditions": []}


class Filter:
    def __init__(self, behavior=None):
        self.proposals, self.actual, self.stats = [], [], {}
        self.behavior = behavior
    def filter(self, action, view, scene, segment, held=None):
        self.proposals.append((tuple(action), deepcopy(segment)))
        return self.behavior(action, segment, len(self.proposals)) if self.behavior else {
            "status": "execute", "action": action, "evidence": {}}
    def observe_transition(self, before, actual, after):
        self.actual.append(tuple(actual))


def policy(monkeypatch, segments, filter=None, final="finish"):
    monkeypatch.setattr(executor, "frame_from_step_view", frame)
    service, client = Service(), Client(segments, final)
    motion = filter or Filter()
    harness = executor.FoundationVisionHarnessPolicy(client, scene_service=service, command_filter=motion)
    return harness, service, client, motion


def step(harness, before, decision):
    after = obs(before.state[:3]+np.asarray(decision.action[:3])*TRANSLATION_SCALE_M,
                gap=.012 if decision.action[6] > 0 else .04)
    harness.observe_transition(before, decision, after)
    return after


def test_seven_segments_continue_locally_and_preserve_relative_entry(monkeypatch):
    harness, service, client, motion = policy(monkeypatch, [segment(i) for i in range(7)])
    current = obs()
    for _ in range(80):
        decision = harness.act(current)
        if harness.terminal:
            break
        current = step(harness, current, decision)
    assert harness.terminal["kind"] == "finish"
    assert len(client.contexts) == 2
    assert len(harness.history) == 7
    assert all(h["status"] == "succeeded" for h in harness.history)
    np.testing.assert_allclose(current.state[:3], [.042, 0., .2], atol=.002)
    assert sum(force for _, force in service.submissions) == 1  # source binding only
    assert motion.actual and all(len(action) == 7 for action in motion.actual)


def test_modified_action_is_acknowledged_and_rejected_proposal_is_not_counted(monkeypatch):
    def behavior(action, segment, n):
        if n == 1:
            return {"status": "reobserve", "reason": "identity_unresolved"}
        modified = list(action)
        modified[0] *= .5
        return {"status": "scaled", "action": modified, "evidence": {"scale": .5}}
    harness, service, client, motion = policy(monkeypatch, [segment(0)], Filter(behavior))
    before = obs()
    before = step(harness, before, harness.act(before))
    decision = harness.act(before)
    assert len(motion.proposals) == 2
    assert harness.primitive.actions_executed == 0
    assert decision.action[0] == motion.proposals[-1][0][0]*.5
    after = step(harness, before, decision)
    assert harness.primitive.actions_executed == 1
    assert motion.actual[-1] == decision.action
    assert len(client.contexts) == 1 and harness.terminal is None
    assert after.state[0] > 0


def test_initial_candidate_failure_refreshes_then_uses_alternative(monkeypatch):
    def behavior(action, segment, n):
        if segment["_candidate"]["xyz_m"][0] > 0:
            return {"status": "candidate_unavailable", "reason": "external_obstacle"}
        return {"status": "execute", "action": action, "evidence": {}}
    harness, service, client, motion = policy(monkeypatch,
        [segment(0, [candidate((.01, 0., 0.)), candidate((-.01, 0., 0.))])], Filter(behavior))
    current = obs()
    current = step(harness, current, harness.act(current))
    decision = harness.act(current)
    assert decision.action[0] < 0
    assert harness.candidate_index == 1 and harness.terminal is None
    assert len(client.contexts) == 1 and len(motion.proposals) == 3
    assert sum(force for _, force in service.submissions) == 2  # binding + one blocker refresh


def test_missing_hold_evidence_hands_back_without_auto_carry_or_false_failure(monkeypatch):
    commands = [segment(0, [candidate(skill="set_gripper", gripper="close")]),
                segment(1, [candidate((0., 0., .025))], "held"),
                segment(2, [candidate((.05, 0., 0.))])]
    harness, service, client, motion = policy(monkeypatch, commands, final="blocked")
    current = obs()
    for _ in range(80):
        decision = harness.act(current)
        if harness.terminal:
            break
        current = step(harness, current, decision)
    assert harness.terminal["kind"] == "blocked"
    assert len(client.contexts) == 2
    assert client.contexts[-1]["feedback"]["reason"] == "held_evidence_unresolved"
    assert client.contexts[-1]["held"]["held"]["confirmed"] is False
    assert not any(s["id"] == "2" for _, s in motion.proposals)
    assert harness.grasp_attempts == 1


def test_payload_reobservation_updates_missing_shape_without_confirming_hold(monkeypatch):
    harness, service, client, motion = policy(monkeypatch,
        [segment(0, [candidate(skill="set_gripper", gripper="close")])])
    current = obs()
    current = step(harness, current, harness.act(current))
    current = step(harness, current, harness.act(current))
    assert harness.payload.held is not None
    harness.payload.held.update(extent_unknown=True, points_in_gripper=np.empty((0, 3)))
    assert harness._recover({"status": "reobserve", "reason": "payload_extent_unknown"},
                            harness.last_view)
    assert not harness.payload.held["extent_unknown"]
    assert len(harness.payload.held["points_in_gripper"]) == 8
    assert not harness.payload.held["confirmed"]
    assert harness.gripper_command == "close" and len(client.contexts) == 1


def test_protocol_failure_is_timed_and_repaired_before_any_fragment_executes(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(executor, "monotonic", lambda: clock[0])
    harness, service, client, motion = policy(monkeypatch, [segment(0)])
    attempts = []
    def request(view, context):
        attempts.append(deepcopy(context))
        clock[0] += 7.25
        client.last_harness_timing = {"first_visible_output_s": 5., "transport_elapsed_s": 7.}
        if len(attempts) == 1:
            raise SchemaError("checkpoint identity needs semantic repair")
        return {"status": "plan", "why": "Object reference repaired.", "targets": [],
                "segments": [segment(0)], "completion_conditions": []}
    client.request_harness = request
    current = obs()
    decision = harness.act(current)
    assert harness.terminal is None and harness.need_model
    assert not motion.proposals and len(attempts) == 1
    assert harness.model_seconds == 7.25
    assert decision.diagnostics["api_calls"][0]["ok"] is False
    current = step(harness, current, decision)
    current = step(harness, current, harness.act(current))
    assert attempts[1]["feedback"]["reason"] == "protocol_repair_required"
    decision = harness.act(current)
    assert decision.diagnostics["primitive"] == "measured_v2_fragment"
    assert len(attempts) == 2 and len(motion.proposals) == 1
    assert harness.agent_stats()["model_requests"][0]["request_timing"]["first_visible_output_s"] == 5.


def test_repeated_unrepaired_protocol_ends_cleanly_without_dispatching_candidates(monkeypatch):
    harness, service, client, motion = policy(monkeypatch, [segment(0)])
    def request(view, context):
        raise SchemaError("ambiguous checkpoint object")
    client.request_harness = request
    current = obs()
    current = step(harness, current, harness.act(current))
    harness.act(current)
    assert harness.terminal["kind"] == "blocked"
    assert harness.terminal["reason"] == "protocol_repair_exhausted_without_changed_state"
    assert harness.model_decisions == 2 and not motion.proposals
    assert all(not record["ok"] for record in harness.model_requests)


@pytest.mark.parametrize("expected,passed", [("body", True), ("different_object", False)])
def test_release_checkpoint_matches_actual_release_identity_on_free_retreat(monkeypatch, expected, passed):
    harness, service, client, motion = policy(monkeypatch, [])
    harness.need_model = False
    harness.checkpoint_pending = {**segment(0, checkpoint="released"),
                                 "object": None, "checkpoint_object": expected, "contact_mode": "free"}
    harness.payload.released = {"object_id": "body"}
    harness.payload.report = {"state": "released_stationary", "object_id": "body"}
    assert harness._checkpoint(StepView(obs())) is passed
    if not passed:
        assert harness.need_model
        assert harness.feedback["reason"] == "checkpoint_object_unresolved"


def test_pixel_grounding_reaches_real_measured_fragment_without_metric_guess(monkeypatch):
    from test_fvh_v2_grounding import scene, inputs
    harness, service, client, motion = policy(monkeypatch, [])
    proposal, command = inputs(xyz=None)
    harness.source_scene, harness.scene = scene(), scene(1)
    harness.active = {**segment(0, [proposal]), **command, "candidates": [proposal]}
    view = StepView(obs((.025, .005, 1.03)))
    harness.entry_state = view.state.copy()
    harness.need_model = False
    assert harness._activate(view)
    np.testing.assert_allclose(harness.primitive._target_xyz, [.005, .005, 1.])
    np.testing.assert_allclose(harness.primitive._target_wxyz, [1., 0., 0., 0.])
    assert harness.grounding_events[-1]["status"] == "grounded"
    assert harness.grounding_events[-1]["evidence"]["position_source"] == "material_point_projection"
