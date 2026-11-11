"""One focused integration check: no model weights, network, or simulator.

All images, scene surfaces and transport replies below are synthetic software
fixtures. They are never task-policy inputs or evidence of manipulation success.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import json

import numpy as np
import pytest

from gpt_field.contact_skills import StepView, matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.client import FoundationHarnessClient, validate_contract
from gpt_field.foundation_harness.executor import FoundationHarnessPolicy
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, PlanningFailure
from gpt_field.foundation_harness.perception import HarnessPerception
from gpt_field.foundation_harness.types import SceneSnapshot, KeyPose, OperationPlan
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, Detection, VisionFrame
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.observation import image_to_data_url
from gpt_field.robodojo_direct_sim import PandaKinematics, rotation_vector
from test_foundation_vision_bridge import metadata
from test_direct_history_video import MemoryWriter


def target():
    return {"id": "object", "query": "box", "selections": {"external": None, "wrist": None},
            "region_pixels": {"external": None, "wrist": None}}


def operation(kind="move"):
    return {"kind": kind, "object": None, "target": None, "grasp_region": None,
            "contact_mode": "free_object", "keep_upright": False, "placement_relation": None,
            "goal_pose": None, "displacement": [.01, 0., 0.] if kind == "move" else None,
            "approach_direction": None, "keyposes": []}


def answer():
    return {"status": "plan", "why": "Move toward a selected visible region.", "targets": [target()],
            "operations": [operation()], "completion_conditions": ["inspect the new view"]}


def observation():
    state = (.2, -.1, 1., .2, -.3, .1, .04, -.04)
    url = image_to_data_url(np.zeros((512, 512, 3), np.uint8), size=512, rotate=False)
    return CalibratedObservation(url, url, state, "Move the box.", (0.,)*7, (0.,)*7,
                                 geometry=metadata(list(state), 512))


class Transport:
    offline_mock = True

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        return {"id": f"synthetic-{len(self.calls)}", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True,
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(self.reply or answer())}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "output_tokens_details": {"reasoning_tokens": 0}}}


def test_real_client_protocol_preserves_scene_and_semantics():
    transport = Transport()
    client = FoundationHarnessClient(task_prompt="Move the box.", transport=transport)
    result = client.request_harness(StepView(observation()), {"step": 0,
        "task_prompt": "Move the box.", "robot_joints": [0.]*7,
        "scene": {"frame_id": 1, "objects": [], "candidates": []}})
    assert result == answer() and len(transport.calls) == 1
    payload = transport.calls[0]
    assert payload["reasoning"]["effort"] == "high"
    assert payload["text"]["format"]["schema"]["properties"]["operations"]
    assert len([part for part in payload["input"][-1]["content"] if part["type"] == "input_image"]) == 2
    assert client.history_stats()["messages"] >= 2
    invalid = answer()
    invalid["operations"][0]["grasp_region"] = "invented_id"
    with pytest.raises(Exception, match="known target"):
        validate_contract(invalid)


def test_perception_partial_geometry_and_same_frame_selection_age(monkeypatch, tmp_path):
    from gpt_field.foundation_harness import perception
    clock = [100.]
    monkeypatch.setattr(perception, "perf_counter", lambda: clock[0])
    def bridge(view, geometry, frame_id, captured_at_s):
        cameras = []
        for name, x in (("external", 0.), ("wrist", .1)):
            extrinsic = np.eye(4); extrinsic[0, 3] = -x
            k = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
            cameras.append(CameraFrame(name, np.zeros((32, 32, 3), np.uint8), k, extrinsic, captured_at_s))
        return VisionFrame(frame_id, captured_at_s, tuple(cameras))
    monkeypatch.setattr(perception, "frame_from_step_view", bridge)
    def segment(frame, targets):
        mask = np.zeros((32, 32), bool); mask[5:27, 5:27] = True
        return [Detection("external", item.key, mask, .8) for item in targets]
    def depth(frame):
        return {cam.name: DepthEstimate(cam.name, np.full((32, 32), 2.), cam.intrinsics,
                                        cam.world_to_camera) for cam in frame.cameras}
    engine = SimpleNamespace(config=EngineConfig(max_age_s=5.), segmenter=SimpleNamespace(segment=segment),
                             depth_estimator=SimpleNamespace(estimate=depth))
    service = HarnessPerception(engine, tmp_path)
    view = SimpleNamespace(observation=object())
    first = service.update(view, {}, [target()])
    assert first.objects["object"].status == "partial"
    assert len(first.objects["object"].points) > 0
    assert len(first.scene_point_object_ids) == len(first.scene_points)
    selected = target(); selected["selections"]["external"] = first.candidates[0]["id"]
    clock[0] = 130.
    second = service.update(view, {}, [selected])
    assert second.captured_at_s == first.captured_at_s == 100.
    assert second.evidence["stale_at_generation"] is True


@pytest.fixture
def real_robot():
    path = Path("/venv/fm_dno/lib/python3.10/site-packages/robosuite/models/assets/robots/panda/robot.xml")
    if not path.is_file():
        pytest.skip("static Panda assets are not installed")
    return PandaKinematics.from_xml(path)


def test_static_robot_full_path_and_unknown_space(real_robot):
    q = np.array([0., -.5, 0., -2., 0., 1.5, .8])
    pose, _ = real_robot.forward(q)
    state = np.r_[pose[:3, 3], rotation_vector(pose[:3, :3]), .04, -.04]
    geometry = SimpleNamespace(query=lambda points, **kwargs: [{"state": "estimated_free"} for _ in points])
    now = perf_counter()
    scene = SceneSnapshot(1, now, now, {}, np.empty((0, 3)), geometry, None)
    planner = CandidatePlanner(real_robot, MotionConfig(max_planning_s=20.))
    op = operation()
    op['displacement'] = None
    xyzw = matrix_to_quaternion_xyzw(pose[:3, :3])
    wxyz = [xyzw[3], *xyzw[:3]]
    op['keyposes'] = [
        {'name': 'move', 'pose': {'position': (pose[:3, 3]+[.01, 0., 0.]).tolist(), 'quaternion_wxyz': wxyz},
         'gripper': 'keep', 'profile': 'transit', 'observe_after': False},
        {'name': 'move', 'pose': {'position': [100., 100., 100.], 'quaternion_wxyz': wxyz},
         'gripper': 'keep', 'profile': 'transit', 'observe_after': True}]
    op['_keypose_index'] = 0
    plan = planner.plan(op, scene, state, q)
    assert len(plan.keyposes) == len(plan.joint_paths) == 1
    assert len(plan.joint_paths[0]) >= 2
    assert plan.evidence["continuous_collision_certified"] is False
    assert plan.keyposes[0].metadata['pose_source'] == 'astra'
    assert len(planner.robot.shapes(planner.bound, q)) >= 10
    geometry.query = lambda points, **kwargs: [{"state": "unknown"} for _ in points]
    verdict = planner.check_motion(q, plan.joint_paths[0][-1], scene, state)
    assert verdict["safe"] and verdict["unknown_samples"] > 0
    assert verdict["unknown_space_allowed"] is True


def test_policy_first_request_fresh_boundary_and_local_execution(tmp_path):
    reply = answer()
    op = reply["operations"][0]
    op["displacement"] = None
    op["keyposes"] = [
        {"name": "pregrasp", "pose": {"position": [.21, -.1, 1.], "quaternion_wxyz": [1., 0., 0., 0.]},
         "gripper": "keep", "profile": "transit", "observe_after": True},
        {"name": "grasp", "pose": {"position": [.22, -.1, 1.], "quaternion_wxyz": [1., 0., 0., 0.]},
         "gripper": "keep", "profile": "contact", "observe_after": False}]
    transport = Transport(reply)
    client = FoundationHarnessClient(task_prompt="Move the box.", transport=transport)
    policy = FoundationHarnessPolicy(client, vision_engine=SimpleNamespace(config=EngineConfig(max_age_s=5.)),
                                     output_dir=tmp_path)
    def snapshot(view, geometry, targets, force=False):
        now = perf_counter()
        return SceneSnapshot(1, now, now, {}, np.empty((0, 3)), None, None)
    policy.perception = SimpleNamespace(update=snapshot)
    view = StepView(observation())
    start = grip_pose(view.state)
    def forward(q):
        pose = start.copy(); pose[:3, 3] += np.asarray(q)[:3]
        return pose, np.zeros((6, 7))
    bound = SimpleNamespace(forward=forward)
    planned_indices = []
    def plan(op, scene, state, joints, held=None):
        planned_indices.append(op["_keypose_index"])
        goal = start.copy(); goal[0, 3] += .01
        return OperationPlan(op, [KeyPose("pregrasp", goal, metadata={"observe_after": True,
            "astra_keypose_index": op["_keypose_index"], "pose_source": "astra"})],
            [[[0.]*7, [.01, 0., 0., 0., 0., 0., 0.]]], .01)
    policy.planner = SimpleNamespace(plan=plan, bound=bound, check_motion=lambda *a, **k: {"safe": True})
    before = observation()
    first = policy.act(before)
    assert first.diagnostics["primitive"] == "foundation:observation"
    assert len(transport.calls) == 1 and np.linalg.norm(first.action[:3]) == 0
    after = observation()
    policy.observe_transition(before, first, after)
    second = policy.act(after)
    assert second.action[0] > 0 and len(transport.calls) == 1
    assert policy.holding is None
    assert policy.agent_stats()["model_decisions"] == 1
    assert policy.planning_snapshot()["keyposes"][0]["name"] == "pregrasp"
    # Reaching this selected observation boundary discards the unexecuted suffix.
    # No second pose may be planned from the old image, even though it was batched.
    latest = observation()
    policy.observe_transition(after, second, latest)
    policy.stage._finish("succeeded", "measured_pose_reached", {})
    third = policy.act(latest)
    assert third.diagnostics["primitive"] == "foundation:observation"
    assert planned_indices == [0] and policy.need_model
    assert not policy.operations and policy.operation_history[-1]["status"] == "segment_completed"
    assert policy.operation_history[-1]["evidence"]["remaining_keyposes"] == 1
    # The detailed planning reason must reach feedback, not only RuntimeError text.
    policy.observe_transition(latest, third, observation())
    policy.current_operation = deepcopy(op)
    def rejected(*args, **kwargs):
        raise PlanningFailure("observed_obstacle_intersection", {"part": "finger", "point_object_id": "object"})
    policy.planner.plan = rejected
    assert not policy._plan(policy._view(policy._expected_observation))
    assert policy.feedback["detail"]["evidence"]["part"] == "finger"


def test_video_renders_planned_geometry_without_a_rollout(tmp_path, monkeypatch):
    from gpt_field.foundation_harness.video import FoundationHarnessVideoRecorder
    from gpt_field.policy import Decision
    monkeypatch.setattr(FoundationHarnessVideoRecorder, "writer_class", MemoryWriter)
    snapshot = {"step": 0, "stage": "move", "scene_points": [[.1, .2, .3]],
        "keyposes": [{"name": "goal", "position": [.2, .3, .4], "rotation": np.eye(3).tolist()}],
        "selected_path": [[.1, .1, .3], [.2, .3, .4]], "measured_path": [], "status": "executing"}
    video = FoundationHarnessVideoRecorder(tmp_path / "memory.mp4", planning_provider=lambda: snapshot)
    obs = observation()
    video.observe(obs, step=0)
    video.decision(obs, Decision((0., 0., 0., 0., 0., 0., -1.), {"step": 0, "primitive": "foundation:move"}))
    assert video.writer.last_frame.size == (1600, 960)
    video.close(status="interface_check")
    rows = [json.loads(row) for row in video.events_path.read_text().splitlines()]
    assert any(row.get("foundation_harness_planning", {}).get("keyposes") for row in rows)
