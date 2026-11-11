"""Small synthetic lifecycle checks; no model, robot or simulator is started."""
from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np

from gpt_field.action import TRANSLATION_SCALE_M
from gpt_field.foundation_harness.types import KeyPose, ObjectGeometry, SceneSnapshot
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame
from gpt_field.foundation_vision_harness_V1 import executor
from gpt_field.foundation_vision_harness_V1.geometry import Grounder


class PublicObservation(SimpleNamespace):
    # Accidental dependence on evaluator channels must fail this public-input test.
    @property
    def success(self):
        raise AssertionError("evaluator success entered policy")

    @property
    def contacts(self):
        raise AssertionError("evaluator contacts entered policy")

    @property
    def object_poses(self):
        raise AssertionError("evaluator object poses entered policy")


def observation(state=None):
    return PublicObservation(state=np.array([0., 0., .1, 0., 0., 0., .04, -.04]
                                             if state is None else state),
        joint_position=np.zeros(7), joint_velocity=np.zeros(7), geometry={},
        prompt="Move the object onto the support.")


def frame(view, geometry, frame_id, captured_at_s):
    intrinsics = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    cameras = []
    for name, x in (("external", 0.), ("wrist", .1)):
        extrinsics = np.eye(4)
        extrinsics[0, 3] = -x
        cameras.append(CameraFrame(name, np.zeros((32, 32, 3), np.uint8),
                                   intrinsics, extrinsics, captured_at_s))
    return VisionFrame(frame_id, captured_at_s, tuple(cameras))


def corners(center):
    return np.asarray(center)+np.array([[x, y, z] for x in (-.01, .01)
                                                   for y in (-.01, .01)
                                                   for z in (-.01, .01)])


class SceneFixture:
    """Explicit synthetic perception, independently moved by the test fixture."""
    def __init__(self, *, missing_body=False):
        self.center = np.array([0., 0., .1])
        self.missing_body = missing_body
        self.submissions, self.waits, self.errors = [], [], []
        self.stats = {}
        self.token = 0
        self.snapshot = None
        self.fail_next = False
        self.invalidations = []

    def submit(self, view, targets, *, force=False, frame_id=None):
        self.token += 1
        self.submissions.append((self.token, frame_id, deepcopy(targets), force))
        if self.fail_next:
            self.errors.append({"token": self.token, "frame_id": frame_id,
                                "type": "RuntimeError", "message": "depth unavailable"})
            return self.token
        objects = {}
        for name, role, center, parent in (("body", "object", self.center, None),
                ("part", "region", self.center, "body"),
                ("support", "support", [.1, 0., .09], None)):
            points = np.empty((0, 3)) if name == "body" and self.missing_body else corners(center)
            objects[name] = ObjectGeometry(name, name, points, np.zeros_like(points),
                "supported" if len(points) else "missing", {},
                {"role": role, "parent_id": parent, "sigma_m": [.002]*3,
                 "observed_frame_id": frame_id, "observed_at_s": monotonic()})
            if role == "support":
                objects[name].evidence["support_plane"] = {
                    "normal": [0., 0., 1.], "point": [.1, 0., .09], "sigma_m": .002}
        points = np.concatenate([o.points for o in objects.values()])
        self.snapshot = SceneSnapshot(frame_id, monotonic(), monotonic(), objects, points,
            None, frame(view, None, frame_id, monotonic()),
            evidence={"revision": self.token, "request_token": self.token,
                      "source_frame_id": frame_id, "perception_status": "ok"})
        return self.token

    def latest(self):
        return self.snapshot

    def wait(self, token, timeout_s):
        self.waits.append(token)
        return None if self.fail_next else self.snapshot

    def invalidate(self, ids, reason):
        self.invalidations.append((ids, reason))

    def close(self):
        pass


def response(kind="pick_place"):
    return {"status": "plan", "why": "Use the selected material and support regions.", "targets": [
        {"id": "body", "query": "object", "role": "object", "parent_id": None,
         "frame_id": 0, "region_pixels": {"external": [1, 1, 10, 10], "wrist": None}},
        {"id": "part", "query": "surface", "role": "region", "parent_id": "body",
         "frame_id": 0, "region_pixels": {"external": [2, 2, 8, 8], "wrist": None}}],
        "operations": [{"kind": kind, "object": "body", "grasp_region": "part",
                        "target": "support", "relation": "on"}],
        "completion_conditions": ["object released on support"]}


class ClientFixture:
    def __init__(self, first=None):
        self.first = first or response()
        self.contexts = []

    def request_harness(self, view, context):
        self.contexts.append(deepcopy(context))
        return deepcopy(self.first) if len(self.contexts) == 1 else {
            "status": "finish", "why": "The fixture observes release on the support.",
            "targets": [], "operations": [], "completion_conditions": []}


class GrounderFixture:
    def __init__(self):
        self.calls = []

    def check_release_relation(self, operation, scene, object_points):
        return Grounder().check_release_relation(operation, scene, object_points)

    def ground(self, operation, scene, state8, q7, *, phase, held=None):
        self.calls.append((phase, scene.frame_id, np.asarray(state8).copy(), deepcopy(held)))
        # Real lift grounding enforces the observed whole-body payload contract.
        if phase == "lift":
            return Grounder().ground(operation, scene, state8, q7, phase=phase, held=held)
        pose = np.eye(4)
        destinations = {"approach": [0., 0., .12], "align": [0., 0., .1],
                        "transport": [.1, 0., .25], "preplace": [.1, 0., .15],
                        "place": [.1, 0., .1], "retreat": [.1, 0., .2]}
        pose[:3, 3] = destinations.get(phase, state8[:3])
        gripper = "open" if phase in {"approach", "align", "release"} else "keep"
        return KeyPose(phase, pose, gripper)


class PlannerFixture:
    def __init__(self, *, blocked=False):
        self.blocked = blocked
        self.calls = []

    def plan(self, keypose, scene, state8, q7, *, held=None):
        return SimpleNamespace(keypose=keypose, waypoints=[])

    def command(self, plan, scene, state8, q7, qd7=None, *, held=None):
        self.calls.append((plan.keypose.name, scene.frame_id, deepcopy(held)))
        if self.blocked:
            return {"blocked": True, "evidence": {"reason": "predicted_scene_collision"}}
        delta = plan.keypose.pose[:3, 3]-state8[:3]
        arrived = np.linalg.norm(delta) < 1e-8
        delta *= min(1., .025/max(1e-9, np.linalg.norm(delta)))
        return {"arrived": arrived, "delta_xyz": delta,
                "quaternion_xyzw": [0., 0., 0., 1.], "gripper": plan.keypose.gripper,
                "evidence": {"source": "synthetic_local_motion"}}


def harness(monkeypatch, *, missing_body=False, blocked=False):
    monkeypatch.setattr(executor, "frame_from_step_view", frame)
    service, client = SceneFixture(missing_body=missing_body), ClientFixture()
    grounder, planner = GrounderFixture(), PlannerFixture(blocked=blocked)
    policy = executor.FoundationVisionHarnessPolicy(client, scene_service=service,
        grounder=grounder, planner=planner,
        config=executor.HarnessConfig(close_min_steps=2, open_min_steps=2, settle_window=2))
    return policy, service, client, grounder, planner


def transition(policy, service, before, decision, *, disturbance=None):
    state = before.state.copy()
    state[:3] += np.asarray(decision.action[:3])*TRANSLATION_SCALE_M
    if disturbance is not None:
        state[:3] += np.asarray(disturbance)
    closing = decision.action[6] > 0
    state[6:8] = [.012, -.012] if closing else [.04, -.04]
    if closing:
        service.center = state[:3].copy()
    after = observation(state)
    policy.observe_transition(before, decision, after)
    return after


def test_same_frame_binding_then_fresh_observation_precedes_motion(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch)
    before = observation()
    decision = policy.act(before)
    assert policy.terminal is None
    assert len(client.contexts) == 1 and not planner.calls and not grounder.calls
    assert np.linalg.norm(decision.action[:3]) == 0
    bound = [s for s in service.submissions if s[2]]
    assert bound and bound[0][1] == 0 and bound[0][2][1]["frame_id"] == 0
    # A different measured observation after the model wait, frozen at ingestion.
    after = transition(policy, service, before, decision, disturbance=[.001, 0., 0.])
    policy.act(after)
    assert grounder.calls[0][1] == 1
    assert grounder.calls[0][2][0] == .001
    assert len(client.contexts) == 1
    assert "point_sigma_m" not in str(client.contexts[0])
    assert not {"success", "contacts", "object_poses"}.intersection(client.contexts[0])


def test_pick_place_executes_close_confirmed_carry_and_release_without_per_phase_model(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch)
    before, commands, reports = observation(), [], []
    for _ in range(100):
        decision = policy.act(before)
        if policy.terminal is not None:
            break
        commands.append((policy.phase, decision.action[6]))
        before = transition(policy, service, before, decision)
        reports.append(deepcopy(policy.payload.report))
    assert policy.terminal is not None and policy.terminal["kind"] == "finish"
    assert len(client.contexts) == 2  # one complete plan, then semantic completion review
    assert policy.grasp_attempts == 1
    assert sum(phase == "close" and cmd > 0 for phase, cmd in commands) >= 2
    assert sum(phase == "release" and cmd < 0 for phase, cmd in commands) >= 2
    assert any(r["state"] == "held_confirmed" for r in reports)
    carry = [held for phase, _, held in planner.calls if phase == "transport"]
    assert carry and all(held["confirmed"] for held in carry)
    assert policy.payload.held is None and policy.payload.released is not None
    assert policy.operation_history[-1]["status"] == "operation_completed"
    assert policy.operation_history[-1]["interaction"]["state"] == "release_observed"


def test_unchanged_local_collision_blocks_after_one_refresh_without_reasking(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch, blocked=True)
    before = observation()
    before = transition(policy, service, before, policy.act(before))
    policy.act(before)
    assert policy.terminal["kind"] == "blocked"
    assert policy.terminal["reason"] == "predicted_scene_collision"
    assert len(planner.calls) == 2 and len(client.contexts) == 1
    policy.act(before)
    assert len(client.contexts) == 1 and len(planner.calls) == 2


def test_missing_whole_body_extent_blocks_lift_before_any_carry(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch, missing_body=True)
    before = observation()
    for _ in range(25):
        decision = policy.act(before)
        if policy.terminal is not None:
            break
        before = transition(policy, service, before, decision)
    assert policy.terminal["kind"] == "blocked"
    assert policy.terminal["reason"] == "payload_extent_unknown"
    assert policy.grasp_attempts == 1 and len(client.contexts) == 1
    assert not any(phase in {"lift", "transport"} for phase, _, _ in planner.calls)


def test_new_perception_failure_cannot_dispatch_from_previous_scene(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch)
    before = observation()
    before = transition(policy, service, before, policy.act(before))
    service.fail_next = True
    policy.act(before)
    assert policy.terminal["kind"] == "blocked"
    assert not planner.calls and not grounder.calls
    assert len(client.contexts) == 1


def test_retained_unobserved_geometry_cannot_confirm_a_hold(monkeypatch):
    policy, service, client, grounder, planner = harness(monkeypatch)
    before = observation()
    service.submit(executor.StepView(before), [], force=True, frame_id=0)
    policy.payload.attach_candidate(service.latest(), "body", executor.StepView(before))
    state = before.state.copy()
    state[2] += .03
    state[6:8] = [.012, -.012]
    after = observation(state)
    service.center = state[:3]
    service.submit(executor.StepView(after), [], force=True, frame_id=1)
    policy.scene = service.latest()
    obj = policy.scene.objects["body"]
    obj.status = "unobserved"
    obj.evidence["observed_frame_id"] = 0
    assert not policy._confirm_hold_from_scene(executor.StepView(after))
    assert not policy.payload.held["confirmed"]
