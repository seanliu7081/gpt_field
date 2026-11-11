"""Focused fake-client checks; no model, perception backend or simulator calls."""
from copy import deepcopy

import numpy as np
import pytest

from gpt_field.contact_skills import StepView
from gpt_field.foundation_vision_harness_V2 import executor
from test_fvh_v2_executor import Client, Filter, Service, candidate, frame, obs, segment, step


@pytest.fixture(autouse=True)
def public_frames(monkeypatch):
    monkeypatch.setattr(executor, "frame_from_step_view", frame)


class DelayedService(Service):
    def __init__(self, *, delay_initial=False):
        super().__init__()
        self.delay_initial = delay_initial
        self.pending, self.waits = {}, []

    def submit(self, view, targets, *, force=False, frame_id=None):
        ready = self.snapshot
        token = super().submit(view, targets, force=force, frame_id=frame_id)
        if force or self.delay_initial:
            self.pending[token] = self.snapshot
            self.snapshot = ready
        return token

    def wait(self, token, timeout_s):
        self.waits.append(token)
        self.snapshot = self.pending.pop(token, self.snapshot)
        return self.snapshot


def free(index, delta=(.006, 0., 0.)):
    return {**segment(index, [candidate(delta)]), "object": None,
            "contact_mode": "free", "contact_parts": []}


def harness(segments, *, service=None, motion=None, client=None):
    service = service or DelayedService()
    client = client or Client(segments)
    return executor.FoundationVisionHarnessPolicy(client, scene_service=service,
        command_filter=motion or Filter()), service, client


def test_initial_model_overlaps_perception_but_execution_waits_for_one_scene():
    policy, service, client = harness([free(0)], service=DelayedService(delay_initial=True))
    current = obs()
    current = step(policy, current, policy.act(current))
    assert len(client.contexts) == 1 and not service.waits
    assert client.contexts[0]["scene"]["status"] == "pending"
    decision = policy.act(current)
    assert decision.diagnostics["primitive"] == "measured_v2_fragment"
    assert len(service.waits) == 1
    assert not any(force for _, force in service.submissions)


def test_mixed_queue_executes_free_segment_before_source_wait_and_keeps_exact_source():
    contact = {**segment(1), "anchor": "body"}
    policy, service, _ = harness([free(0), contact])
    current = obs()
    current = step(policy, current, policy.act(current))
    source_token = policy.binding_token
    submissions = len(service.submissions)
    decision = policy.act(current)
    assert decision.diagnostics["primitive"] == "measured_v2_fragment"
    assert policy.active["id"] == "0"
    assert not service.waits and policy.binding_token == source_token
    assert len(service.submissions) == submissions  # pending source was not coalesced
    for _ in range(30):
        current = step(policy, current, decision)
        assert not service.waits and policy.binding_token == source_token
        assert len(service.submissions) == submissions
        if policy.active is None:
            break
        decision = policy.act(current)
        assert policy.active["id"] == "0"
        assert not service.waits and len(service.submissions) == submissions
    assert policy.active is None
    assert policy.history[-1]["status"] == "succeeded"
    decision = policy.act(current)
    assert policy.active["id"] == "1"
    assert service.waits == [source_token]
    assert policy.source_scene.frame_id == 0
    assert sum(force for _, force in service.submissions) == 1  # no forced second frame
    source = policy.source_scene
    current = step(policy, current, decision)
    policy._refresh(StepView(current))
    assert policy.source_scene is source and policy.source_scene.frame_id == 0
    assert policy.scene.frame_id != policy.source_scene.frame_id


def test_repeated_blocker_refreshes_once_and_changed_proposal_cannot_repeat_model_loop():
    class ChangingClient(Client):
        def request_harness(self, view, context):
            self.contexts.append(deepcopy(context))
            return {"status": "plan", "why": "Try another direction.", "targets": [],
                    "segments": [free(str(len(self.contexts)), (.01 * len(self.contexts), 0., 0.))],
                    "completion_conditions": []}

    def reject(action, segment, number):
        # Two samples on opposite sides of a rounding boundary are one blocker.
        x = .0099 if number < 3 else .0101
        return {"status": "candidate_unavailable", "reason": "local_candidate_blocked",
                "evidence": {"rejected_options": [{"detail": {"examples": [
                    {"part": "link2", "point_object_id": "", "point_world_m": [x, .1, .5],
                     "reason": "observed_surface_intersection"}]}}]}}

    policy, service, client = harness([], client=ChangingClient([]), motion=Filter(reject))
    current = obs()
    for _ in range(6):
        decision = policy.act(current)
        if policy.terminal:
            break
        current = step(policy, current, decision)
    assert len(client.contexts) == 2
    assert policy.terminal["reason"] == "recovery_exhausted_without_changed_evidence"
    assert len(service.waits) == 1  # one purposeful refresh, no model-response barriers
    assert policy.grasp_attempts == 0


def test_blocker_identity_tracks_named_surface_motion_and_robot_joint_posture():
    registry = []
    def keys(point, observation):
        evidence = {"examples": [{"part": "link2", "point_object_id": "obstacle",
                                  "point_world_m": point, "reason": "observed_surface_intersection"}]}
        return executor._failure_keys("local_candidate_blocked", evidence,
                                      StepView(observation), free(0), registry)

    initial = keys([.0099, .1, .5], obs())
    assert keys([.0101, .1, .5], obs()) == initial
    assert keys([.06, .1, .5], obs()) != initial
    changed = obs()
    changed.joint_position = np.array([.15, 0., 0., 0., 0., 0., 0.])
    assert keys([.0099, .1, .5], changed) != initial
