"""Synthetic-only Astranomy mobile protocol checks: zero simulator/model calls.

No saved successful episode, task object coordinates or demonstration is used.
The camera fixture exposes only named robot/camera measurements.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.action import ROTATION_SCALE_RAD, TRANSLATION_SCALE_M, _rotvec_quaternion
from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.astranomy_agentic_motion import (
    ShortTargetConfig, ShortTargetExecutor, TargetLimitError, TargetValidationError,
    measured_pose, rejected_feedback,
)
from gpt_field.observation import decode_image, image_to_data_url
from gpt_field.robocasa_user_check_geometry import (
    DEFAULT_EXTERNAL_CAMERA, GRIP_SITE, HAND_BODY, HAND_TO_GRIP_M,
    WRIST_CAMERA, WRIST_POSITION_HAND_M, WRIST_ROTATION_HAND_FROM_CAMERA,
    extract_robocasa_geometry,
)
from rc365.sim import MOBILE_COMMAND_FRAME, MOBILE_SCHEMA, MobileObservation
from rc365.astranomy import (
    MOBILE_TOOL_NAME, TOOL_NAME, MobileExecutor, RoboCasaAstranomyClient,
    model_request, tool_definitions, validate_command, validate_mobile_feedback,
)

TOKEN = "offline-robocasa-astranomy-token"
STATE = [1.2, -.3, 1.1, 0., 0., 0., .04, -.04]
TASK = "Pick the cans from the counter and place them in their designated side in the cabinet."
PIXELS = np.zeros((512, 512, 3), dtype=np.uint8)
PIXELS[20:40, 30:50] = [255, 0, 0]
PIXELS[400:420, 450:470] = [0, 255, 0]
# Raw transport mirrors horizontally; the model must receive upright images once.
IMAGE = image_to_data_url(PIXELS[:, ::-1], size=512, rotate=False)


class CameraModel:
    cam_fovy = np.array([60., 75.])
    cam_bodyid = np.array([1, 0])
    cam_mode = np.array([0, 0])
    cam_pos = np.array([[0., 0., 0.], WRIST_POSITION_HAND_M])
    cam_quat = np.array([[1., 0., 0., 0.], [0., 2 ** -.5, 2 ** -.5, 0.]])

    def camera_name2id(self, name):
        return {DEFAULT_EXTERNAL_CAMERA: 0, WRIST_CAMERA: 1}[name]

    def body_name2id(self, name):
        assert name == HAND_BODY
        return 0

    def site_name2id(self, name):
        assert name == GRIP_SITE
        return 0

    def __getattr__(self, name):
        raise AssertionError(f"Unapproved simulator model read: {name}")


class CameraData:
    def __init__(self, state, camera_shift=0.):
        grip = np.asarray(state[:3]); hand = grip - HAND_TO_GRIP_M
        self.body_xmat = np.array([np.eye(3).ravel()])
        self.body_xpos = np.array([hand]); self.site_xpos = np.array([grip])
        self.cam_xpos = np.array([[camera_shift, 0., 2.], hand + WRIST_POSITION_HAND_M])
        self.cam_xmat = np.array([np.eye(3).ravel(), WRIST_ROTATION_HAND_FROM_CAMERA.ravel()])

    def __getattr__(self, name):
        raise AssertionError(f"Unapproved simulator data read: {name}")


def mobile_state(step=0, x=0.):
    return {"schema_version": MOBILE_SCHEMA, "command_frame": MOBILE_COMMAND_FRAME,
            "base_joint_position": [x, 0., 0.], "base_joint_velocity": [0., 0., 0.],
            "torso_joint_position": [.1], "torso_joint_velocity": [0.],
            "base_position_world_m": [x, 0., 0.], "base_rotation_world": np.eye(3).tolist(),
            "control_mode": "arm", "step": step}


def observation(step=0, *, x=0., state=None):
    state = list(STATE if state is None else state)
    env = SimpleNamespace(sim=SimpleNamespace(model=CameraModel(), data=CameraData(state, x)))
    geometry = extract_robocasa_geometry(env, image_size=512, observation_state=state)
    return MobileObservation(image=IMAGE, wrist_image=IMAGE, state=tuple(state), prompt=TASK,
                             joint_position=(0.,) * 7, joint_velocity=(0.,) * 7,
                             geometry=geometry, mobile_state=mobile_state(step, x))


def arm_target(delta=(0., 0., 0.), rotation=(0., 0., 0.), gripper=0):
    q = _rotvec_quaternion(np.asarray(rotation))
    return [*(np.asarray(STATE[:3]) + delta), q[3], *q[:3], gripper]


def mobile_command(steps=3, **changes):
    return {"base": [.1, -.1, .05], "torso": .1, "steps": steps, **changes}


def native_response(name=TOOL_NAME, arguments=None, ident=1, **changes):
    if arguments is None:
        arguments = {"target": arm_target(), "steps": 3}
    return {"id": f"resp_{ident}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "high"}, "usage": {"input_tokens": 100, "output_tokens": 25},
            "output": [{"type": "reasoning", "id": f"rs_{ident}", "summary": [],
                        "encrypted_content": f"opaque-{ident}"},
                       {"type": "message", "role": "assistant", "id": f"msg_{ident}",
                        "content": [{"type": "output_text", "text": "Move briefly and inspect."}]},
                       {"type": "function_call", "id": f"fc_{ident}", "call_id": f"call_{ident}",
                        "status": "completed", "namespace": "robot", "name": name,
                        "arguments": json.dumps(arguments)}], **changes}


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies); self.requests = []

    def __call__(self, endpoint, payload, headers, timeout):
        self.requests.append(deepcopy(payload))
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        assert self.replies, "Unexpected additional inference request"
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def client(replies=(), **kwargs):
    transport = FakeTransport(replies)
    c = RoboCasaAstranomyClient(AstraConfig(), token_provider=lambda: TOKEN,
                                transport=transport, **kwargs)
    return c, transport


def execute_arm_feedback(args):
    executor = ShortTargetExecutor(STATE, args["target"], args["steps"])
    for _ in range(args["steps"]):
        action = executor.next_action(STATE)
        executor.observe_transition(STATE, action, STATE)
    return executor.feedback()


def execute_mobile_feedback(command, start=0, gripper=-1.):
    obs = observation(start)
    executor = MobileExecutor(obs, command, gripper=gripper)
    for step in range(command["steps"]):
        action = executor.next_action(obs)
        after = observation(start + step + 1, x=.001 * (step + 1))
        executor.observe_transition(obs, action, after)
        obs = after
    return executor.feedback(), executor


def test_two_strict_native_tools_and_preview_never_authenticates():
    def no_auth():
        raise AssertionError("Preview must not authenticate")
    c = RoboCasaAstranomyClient(AstraConfig(), token_provider=no_auth)
    payload = c.build_payload(model_request(observation(), 0))
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False
    assert payload["store"] is False and payload["stream"] is True
    assert len(payload["tools"]) == 1 and payload["tools"][0]["type"] == "namespace"
    assert payload["tools"][0]["name"] == "robot"
    tools = payload["tools"][0]["tools"]
    assert tools == tool_definitions()
    assert {t["name"] for t in tools} == {TOOL_NAME, MOBILE_TOOL_NAME}
    for tool in tools:
        assert tool["strict"] is True
        params = tool["parameters"]
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])
        assert params["properties"]["steps"]["minimum"] == 1
        assert params["properties"]["steps"]["maximum"] == 5
    assert c.budget_snapshot()["subscription_inference_calls"] == 0
    assert c.history_snapshot() == []
    assert "robocasa" in payload["instructions"].lower()
    assert "gripper0_right_grip_site" in payload["instructions"]
    assert "robot0_right_hand" in payload["instructions"]


def test_model_request_uses_upright_rgb_and_measured_robocasa_geometry_only():
    obs = observation(step=2, x=.01)
    request = model_request(obs, 2)
    assert set(request) == {"external_image", "wrist_image", "robot_state", "joint_positions",
                           "geometry", "mobile_state", "step"}
    assert request["robot_state"] == measured_pose(STATE)
    assert request["geometry"]["gripper"]["position_reference"] == GRIP_SITE
    assert request["geometry"]["cameras"]["external"]["position_world_m"][0] == .01
    assert request["mobile_state"] == obs.mobile_state
    for key in ("external_image", "wrist_image"):
        np.testing.assert_array_equal(np.asarray(decode_image(request[key])), PIXELS)
    c, _ = client()
    payload = c.build_payload(request)
    text = json.loads(payload["input"][-1]["content"][0]["text"])
    assert text["task"] == TASK
    assert text["geometry"] == request["geometry"]
    assert text["mobile_state"] == obs.mobile_state
    assert not {"reward", "success", "task_success", "object_pose", "contacts"} & set(text)


@pytest.mark.parametrize("path", [(), ("geometry",), ("geometry", "cameras", "external"),
                                  ("geometry", "gripper"), ("mobile_state",)])
def test_model_payload_rejects_evaluator_truth_at_every_allowed_boundary(path):
    request = model_request(observation(), 0); node = request
    for key in path:
        node = node[key]
    node["object_pose"] = [91., 92., 93.]
    c, _ = client()
    with pytest.raises(ValueError):
        c.build_payload(request)
    assert c.history_snapshot() == []


@pytest.mark.parametrize("command", [
    {"base": [0., 0.], "torso": 0., "steps": 2},
    mobile_command(base=[True, 0., 0.]), mobile_command(base=[0., float("nan"), 0.]),
    mobile_command(base=[0., 0., .200001]), mobile_command(torso=-.200001),
    mobile_command(torso=float("inf")), mobile_command(steps=True),
    mobile_command(steps=0), mobile_command(steps=6), mobile_command(steps=2.),
    mobile_command(gripper=1), mobile_command(target=arm_target()),
])
def test_mobile_tool_rejects_invalid_or_combined_arguments(command):
    with pytest.raises(SchemaError):
        validate_command({"name": MOBILE_TOOL_NAME, "arguments": command})


@pytest.mark.parametrize("count", [0, 2])
def test_zero_or_parallel_native_tool_calls_are_rejected_without_history(count):
    response = native_response()
    call = response["output"][-1]
    response["output"] = response["output"][:-1] + [deepcopy(call) for _ in range(count)]
    if count == 2:
        response["output"][-1].update(name=MOBILE_TOOL_NAME, call_id="second-call",
                                      arguments=json.dumps(mobile_command()))
    c, transport = client([response])
    with pytest.raises(SchemaError):
        c.predict(model_request(observation(), 0))
    assert len(transport.requests) == 1 and c.history_snapshot() == []


@pytest.mark.parametrize("name", ["finish", "execute", "shell", "robot.execute_mobile"])
def test_only_two_native_tool_names_are_allowed(name):
    c, _ = client([native_response(name=name)])
    with pytest.raises(SchemaError):
        c.predict(model_request(observation(), 0))
    assert c.history_snapshot() == []


def test_full_visual_reasoning_tool_history_survives_more_than_six_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-paid-key-fixture")
    replies = [native_response(name=MOBILE_TOOL_NAME if i % 2 else TOOL_NAME,
                               arguments=mobile_command() if i % 2 else {"target": arm_target(), "steps": 3},
                               ident=i + 1) for i in range(8)]
    c, transport = client(replies, log_dir=tmp_path)
    for i in range(8):
        obs = observation(i * 3)
        command = c.predict(model_request(obs, i * 3))
        with pytest.raises(ValueError):
            c.build_payload(model_request(obs, i * 3))
        if command["name"] == MOBILE_TOOL_NAME:
            result, _ = execute_mobile_feedback(command["arguments"], i * 3)
        else:
            result = execute_arm_feedback(command["arguments"])
        c.record_result(result)
    history = c.history_snapshot()
    assert sum(x.get("role") == "user" for x in history) == 8
    assert sum(x.get("type") == "function_call" for x in history) == 8
    outputs = [x for x in history if x.get("type") == "function_call_output"]
    assert [x["call_id"] for x in outputs] == [f"call_{i}" for i in range(1, 9)]
    assert [x["encrypted_content"] for x in history if x.get("type") == "reasoning"] == [f"opaque-{i}" for i in range(1, 9)]
    assert sum(z.get("type") == "input_image" for x in history for z in x.get("content", [])) == 16
    assert "omitted" not in json.dumps(history)
    assert c.history_stats()["image_history_limit"] is None
    assert c.history_stats()["feedback_records"] == 8
    assert c.budget_snapshot()["api_key_calls"] == 0
    assert len(transport.requests) == 8
    preview = c.build_payload(model_request(observation(24), 24))
    preview["input"][0]["content"].clear()
    assert c.history_snapshot() == history
    for path in tmp_path.glob("*.json"):
        assert TOKEN not in path.read_text() and "forbidden-paid-key-fixture" not in path.read_text()


@pytest.mark.parametrize("steps", range(1, 6))
@pytest.mark.parametrize("gripper", [-1., 1.])
def test_mobile_exact_zero_arm_preserved_gripper_and_exact_selected_duration(steps, gripper):
    obs = observation()
    executor = MobileExecutor(obs, mobile_command(steps=steps), gripper=gripper)
    for i in range(steps):
        action = executor.next_action(obs)
        assert tuple(action) == (0., 0., 0., 0., 0., 0., gripper)
        with pytest.raises(RuntimeError):
            executor.next_action(obs)
        wrong = list(action); wrong[0] = 1e-9
        with pytest.raises((ValueError, RuntimeError)):
            executor.observe_transition(obs, wrong, observation(i + 1))
        after = observation(i + 1, x=.001 * (i + 1))
        executor.observe_transition(obs, action, after)
        assert executor.executed_steps == i + 1
        assert executor.done is (i + 1 == steps)
        obs = after
    feedback = validate_mobile_feedback(executor.feedback())
    assert feedback["status"] == "segment_complete"
    assert feedback["requested_steps"] == feedback["executed_steps"] == steps
    assert feedback["simulation_time_s"] == pytest.approx(steps / 20)
    assert feedback["gripper_command"] == gripper
    assert feedback["measured_mobile_state"] == obs.mobile_state
    assert not {"success", "task_success", "object_pose", "contacts", "grasp_success"} & set(feedback)
    with pytest.raises(RuntimeError):
        executor.next_action(obs)


def test_mobile_feedback_rejects_evaluator_leak_and_wrong_acknowledgement():
    result, _ = execute_mobile_feedback(mobile_command())
    for key in ("success", "reward", "object_pose", "contacts", "task_progress"):
        with pytest.raises(ValueError):
            validate_mobile_feedback({**result, key: True})
    injected = deepcopy(result); injected["measured_mobile_state"]["object_pose"] = [1., 2., 3.]
    with pytest.raises(ValueError):
        validate_mobile_feedback(injected)
    c, _ = client([native_response(name=MOBILE_TOOL_NAME, arguments=mobile_command(steps=2))])
    c.predict(model_request(observation(), 0))
    with pytest.raises(ValueError):
        c.record_result(result)
    assert c.history_stats()["pending_tool_result"] is True
    assert c.history_stats()["feedback_records"] == 0


def test_mobile_interruption_reports_only_completed_steps():
    obs = observation()
    executor = MobileExecutor(obs, mobile_command(steps=5), gripper=1.)
    action = executor.next_action(obs); after = observation(1, x=.002)
    executor.observe_transition(obs, action, after)
    executor.interrupt("operator stop")
    result = validate_mobile_feedback(executor.feedback())
    assert result["status"] == "interrupted" and result["executed_steps"] == 1
    assert result["requested_steps"] == 5 and result["simulation_time_s"] == .05
    assert result["measured_mobile_state"] == after.mobile_state


@pytest.mark.parametrize("steps", range(1, 6))
def test_original_arm_short_target_caps_and_fixed_count_are_preserved(steps):
    target = arm_target(delta=(.03, .04, 0.), rotation=(.21, .14, .07), gripper=1)
    executor = ShortTargetExecutor(STATE, target, steps)
    for index in range(steps):
        action = executor.next_action(STATE)
        assert np.linalg.norm(action[:3]) * TRANSLATION_SCALE_M == pytest.approx(.02)
        assert np.linalg.norm(action[3:6]) * ROTATION_SCALE_RAD == pytest.approx(.1)
        assert action[-1] == 1.
        executor.observe_transition(STATE, action, STATE)
        assert executor.done is (index + 1 == steps)
    assert executor.feedback()["position_error_m"] == pytest.approx(.05)
    assert executor.feedback()["status"] == "segment_complete"
    assert executor.config == ShortTargetConfig(.05, .35, .02, .1)


@pytest.mark.parametrize("target", [arm_target(delta=(.050001, 0., 0.)),
                                   arm_target(delta=(.04, .04, 0.)),
                                   arm_target(rotation=(.350001, 0., 0.))])
def test_far_arm_targets_are_zero_action_rejections_not_clipped(target):
    with pytest.raises(TargetLimitError) as error:
        ShortTargetExecutor(STATE, target, 3)
    feedback = rejected_feedback(STATE, target, 3, str(error.value))
    assert feedback["executed_steps"] == 0 and feedback["simulation_time_s"] == 0
    assert feedback["status"] == "rejected"
    assert feedback["measured_pose"] == measured_pose(STATE)


@pytest.mark.parametrize("steps", [0, 6, True, 1.0])
def test_arm_duration_cannot_escape_one_to_five_steps(steps):
    with pytest.raises(TargetValidationError):
        ShortTargetExecutor(STATE, arm_target(), steps)


def test_mobile_executor_rejects_stale_robot_or_mobile_sensor_ack():
    before = observation()
    executor = MobileExecutor(before, mobile_command(steps=2), gripper=-1.)
    stale_step = replace(before, mobile_state=mobile_state(step=1))
    stale_pose = replace(before, state=tuple([STATE[0] + .001, *STATE[1:]]))
    for stale in (stale_step, stale_pose):
        with pytest.raises((ValueError, RuntimeError)):
            executor.next_action(stale)
    action = executor.next_action(before)
    for stale in (stale_step, stale_pose):
        with pytest.raises((ValueError, RuntimeError)):
            executor.observe_transition(stale, action, observation(1))
    assert executor.executed_steps == 0
    executor.observe_transition(before, action, observation(1))
    assert executor.executed_steps == 1


@pytest.mark.parametrize("changes", [
    {"status": "success"}, {"status": "task_complete"}, {"executed_steps": 4},
    {"executed_steps": True}, {"requested_steps": 0}, {"control_hz": 15},
    {"simulation_time_s": .2}, {"gripper_command": 0},
])
def test_mobile_feedback_describes_only_consistent_robot_segments(changes):
    result, _ = execute_mobile_feedback(mobile_command(steps=3))
    with pytest.raises(ValueError):
        validate_mobile_feedback({**result, **changes})


def test_legacy_libero_geometry_cannot_enter_robocasa_client():
    from gpt_field import astranomy_agentic_geometry as legacy
    request = model_request(observation(), 0)
    old = deepcopy(request["geometry"])
    old["schema_version"] = legacy.GEOMETRY_VERSION
    old["gripper"] = deepcopy(legacy._GRIPPER)
    old["cameras"]["external"].update(mujoco_name="agentview", pose_source="fixed_world_camera")
    legacy.validate_geometry_metadata(old)  # Valid legacy data, wrong robot contract here.
    request["geometry"] = old
    c, _ = client()
    with pytest.raises(ValueError):
        c.build_payload(request)


def test_mobile_feedback_same_steps_wrong_command_is_not_committed():
    c, _ = client([native_response(name=MOBILE_TOOL_NAME, arguments=mobile_command())])
    c.predict(model_request(observation(), 0))
    result, _ = execute_mobile_feedback(mobile_command(base=[.1, 0., 0.]))
    with pytest.raises(ValueError):
        c.record_result(result)
    assert c.history_stats()["pending_tool_result"] is True
    assert c.history_stats()["feedback_records"] == 0


@pytest.mark.parametrize("change", ["position", "orientation", "gripper"])
def test_arm_feedback_same_steps_wrong_target_is_not_committed(change):
    c, _ = client([native_response()])
    command = c.predict(model_request(observation(), 0))
    args = deepcopy(command["arguments"])
    if change == "position":
        args["target"][0] += .001
    elif change == "orientation":
        args["target"][3:7] = [math.cos(.01), 0., 0., math.sin(.01)]
    else:
        args["target"][7] = 1
    result = execute_arm_feedback(args)
    with pytest.raises(ValueError):
        c.record_result(result)
    assert c.history_stats()["pending_tool_result"] is True
    assert c.history_stats()["feedback_records"] == 0
    c.record_result(execute_arm_feedback(command["arguments"]))
    assert c.history_stats()["feedback_records"] == 1


def test_arm_normalized_quaternion_feedback_matches_the_native_call():
    args = {"target": arm_target(), "steps": 3}
    args["target"][3:7] = [2., 0., 0., 0.]
    c, _ = client([native_response(arguments=args)])
    command = c.predict(model_request(observation(), 0))
    result = execute_arm_feedback(command["arguments"])
    assert result["requested_target"][3:7] == [1., 0., 0., 0.]
    c.record_result(result)
    assert c.history_stats()["feedback_records"] == 1


def test_reported_medium_response_is_rejected_for_fixed_high_client():
    c, transport = client([native_response(reasoning={"effort": "medium"})])
    with pytest.raises(APIError):
        c.predict(model_request(observation(), 0))
    assert len(transport.requests) == 1
    assert transport.requests[0]["reasoning"] == {"effort": "high"}
    assert c.history_snapshot() == []


def test_transport_failure_has_no_automatic_retry_or_api_key_fallback():
    c, transport = client([APIError("offline infrastructure failure")])
    with pytest.raises(APIError):
        c.predict(model_request(observation(), 0))
    assert len(transport.requests) == 1
    assert c.budget_snapshot()["api_key_calls"] == 0
    assert c.history_snapshot() == []
