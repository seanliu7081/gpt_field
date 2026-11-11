"""Astranomy short-target baseline adapted only for RoboCasa robot interfaces.

The existing EEF target, 1--5-step controller and complete accepted history are
retained. A separate bounded native mobile tool preserves the user's base/torso
capability. No UC7.1 fragment, initial plan, obstacle map or skill is exposed.
Importing this module never creates a simulator or sends an inference request.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Mapping

import numpy as np
from PIL import Image

from gpt_field.astra import AstraConfig, SchemaError, _canonical
from gpt_field.astranomy_agentic_client import (
    AstranomyAgenticClient, HISTORY_PROTOCOL, TOOL_NAMESPACE, TOOL_NAME,
    OBSERVATION_KEYS as BASE_OBSERVATION_KEYS, _finite, _pose,
    tool_definitions as arm_tool_definitions, validate_command as validate_arm_command,
    validate_feedback as validate_arm_feedback,
)
from gpt_field.astranomy_agentic_motion import measured_pose, _target_values
from gpt_field.hybrid_subscription import ChatGPTPlanHybridClient
from gpt_field.observation import decode_image, image_to_data_url
from gpt_field.robocasa_user_check_geometry import (
    GEOMETRY_VERSION, validate_geometry_metadata as _shared_geometry_validator,
)
from .sim import validate_mobile_command, validate_mobile_state

CLIENT_VERSION = "astranomy-agentic-robocasa-mobile-client-1"
PROMPT_VERSION = "astranomy-agentic-robocasa-mobile-prompt-1"
TASK_KEY = "robocasa365_RestockPantry"
TASK_PROMPT = "Pick the cans from the counter and place them in their designated side in the cabinet."
MOBILE_TOOL_NAME = "execute_mobile"
OBSERVATION_KEYS = BASE_OBSERVATION_KEYS | {"mobile_state"}
MOBILE_FEEDBACK_KEYS = frozenset({"status", "requested_command", "measured_mobile_state",
    "requested_steps", "executed_steps", "simulation_time_s", "control_hz", "gripper_command", "detail"})


def validate_geometry_metadata(value) -> dict:
    """This isolated client accepts only the explicit RoboCasa robot schema."""
    if not isinstance(value, Mapping) or value.get("schema_version") != GEOMETRY_VERSION:
        raise ValueError("Expected the RoboCasa camera and robot geometry schema")
    return _shared_geometry_validator(value)


def instructions(geometry: Mapping) -> str:
    """Preserve the original generic arm prompt with explicit robot-only changes."""
    validate_geometry_metadata(geometry)
    return f"""Interface {PROMPT_VERSION}. You control one simulated PandaOmron: a Franka Panda with a PandaGripper, a mobile base and torso.
Task: {TASK_PROMPT}

Use the current external and wrist RGB images, measured robot state, camera calibration, and this episode's action feedback. Retain the full chronological text and visual history. Images, robot state and camera poses in one observation are synchronized; interpret an older image with its own calibration. There are no demonstrations or prior task trajectories. Object poses, depth, segmentation, contacts, rewards and subtask-progress labels are not supplied.

Before moving, formulate a brief sequence of subgoals from the task and images. Keep track of the current subgoal, evaluate measured outcomes, and revise your plan when necessary. You may express a short plan or update in text. Then call exactly one robot tool: execute_eef_target or execute_mobile. Optional shell, code, web, image-editing, delegation and other tools are unavailable. There is no done or give_up tool. Only the environment evaluator determines success; an issued command, grasp-looking image, or reached pose does not establish completion.

For execute_eef_target, supply only target and steps. The complete target is [x,y,z,qw,qx,qy,qz,gripper]. XYZ is ABSOLUTE in the fixed simulation world frame, in metres. World +z is upward. Positive world directions do not rotate with the wrist. The position locates gripper0_right_grip_site between the fingers. The unit wxyz quaternion gives robot0_right_hand orientation in world coordinates; it is not the site's rotated visualization frame. Supply all eight entries every time. Gripper is 0=open or 1=closed. The robot_state has this same eight-entry pose convention and a measured continuous closure value in its last entry; it is feedback, not proof of attachment. The joint_positions field contains seven measured arm joint angles in radians.

Choose an integer steps from 1 through 5. The target must be no more than 0.05 m translation and 0.35 rad rotation from the latest measured pose. One unchanged target is held for the selected steps at the native 20 Hz (0.05 s per step). It is not a waypoint sequence or an arrival-waiting macro. Each step recomputes world-frame pose error from measured state and limits the requested increment to 0.02 m translation and 0.1 rad rotation before the native OSC controller. Actual displacement may be smaller. The requested gripper command is applied on every step concurrently with arm motion. Simulation pauses during model inference. Shorten segments near contact and judge arrival, finger motion and attachment from subsequent observations and measured tracking error. During this arm tool, the base receives zero velocity and the torso holds its latest position.

For execute_mobile, supply base=[forward,sideways,yaw], torso and steps. Each base component and torso is a normalized native controller input in [-0.2,0.2]; steps is an integer from 1 through 5. Base translation is relative to the current base heading. The measured mobile_state describes the native frame and reports base/torso robot sensors. Positive yaw is counterclockwise. After the native frame transform, actuator command scales are 1 m/s, 1 m/s and 1.5 rad/s for the three base axes. Positive torso raises it; the native torso delta scale is 0.05 m, so input 0.2 requests 0.01 m per step. These are command scales, not measured motion guarantees. The arm follows the base in native mobile mode with zero arm delta; the previous finger command is kept. This tool cannot grasp, open or close the gripper. Do not combine arm and mobile tools in one response. Keep the whole hand and carried object clear, choose short chunks from visible free space, and inspect fresh images and sensors afterward. There is no autonomous navigation or whole-body collision planner. The mobile tool result describes executed steps and measured robot state, not goal achievement.

Each observation's geometry contains only camera calibration and fixed robot dimensions. Camera position_world_m and rotation_world_from_camera map OpenGL camera coordinates to world: p_world = p_camera_origin + R * p_camera. Camera +X points image-right, +Y image-up, and -Z forward. The supplied RGB is upright and uncropped: pixel u grows right and v grows down. For camera point [X,Y,Z] with Z<0, u=cx+fx*X/(-Z) and v=cy-fy*Y/(-Z), using focal_length_px and principal_point_px. A pixel specifies a ray; depth must be inferred from visual evidence. The wrist pose_source is observed_robot_pose_fixed_mount: its pose is derived from this observation's measured robot pose and fixed wrist-camera mounting transform, synchronizing it with the same RGB rather than a later simulator state. The external camera is mounted on the robot and its world pose can change with base/torso motion; use each observation's own synchronized calibration. Do not use image axes directly as world actions or assume the image center is the grasp point.

The Panda hand's local +Z is the finger approach direction and local Y is the jaw separation axis; both rotate with the quaternion. The grip site is 0.097 m along hand-local +Z from the hand origin. Action XYZ ALREADY locates the grip site: do not add another 0.097 m offset. For symmetric fingers, pad centers have midpoint [0,0,-0.0036] m relative to the grip site in hand axes. Each pad measures [0.016,0.008,0.016] m along those axes. Nominal open inner gap is 0.079 m. These fixed asset dimensions explain the tool, not scene clearance or current contact. Assess the complete hand and fingers in both views when positioning; visible object centers need not be suitable contact points.
"""


def tool_definitions() -> list[dict]:
    fields = {"base": {"type": "array", "items": {"type": "number", "minimum": -.2, "maximum": .2},
                       "minItems": 3, "maxItems": 3},
              "torso": {"type": "number", "minimum": -.2, "maximum": .2},
              "steps": {"type": "integer", "minimum": 1, "maximum": 5}}
    return arm_tool_definitions() + [{"type": "function", "name": MOBILE_TOOL_NAME, "strict": True,
        "description": "Repeat one bounded native base/torso command for 1 through 5 steps, retaining the previous gripper command and zero arm delta. No arrival or navigation guarantee.",
        "parameters": {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}]


def _mobile_arguments(arguments) -> dict:
    if not isinstance(arguments, dict) or set(arguments) != {"base", "torso", "steps"}:
        raise ValueError("execute_mobile requires exactly base, torso and steps")
    if type(arguments["steps"]) is not int or not 1 <= arguments["steps"] <= 5:
        raise ValueError("steps must be an integer from 1 through 5")
    return {**validate_mobile_command({k: arguments[k] for k in ("base", "torso")}), "steps": arguments["steps"]}


def validate_command(value) -> dict:
    if isinstance(value, dict) and value.get("name") == TOOL_NAME:
        return validate_arm_command(value)
    try:
        if not isinstance(value, dict) or set(value) != {"name", "arguments"} or value["name"] != MOBILE_TOOL_NAME:
            raise ValueError("Only execute_eef_target or execute_mobile is allowed")
        return {"name": MOBILE_TOOL_NAME, "arguments": _mobile_arguments(value["arguments"])}
    except (ValueError, TypeError) as exc:
        raise SchemaError(str(exc)) from None


def _native_gripper(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (-1, 1):
        raise ValueError("mobile requires the previous native gripper command -1=open or +1=close")
    return float(value)


def validate_mobile_feedback(value) -> dict:
    if not isinstance(value, Mapping) or set(value) != MOBILE_FEEDBACK_KEYS:
        raise ValueError("Mobile feedback contains missing or unexpected fields")
    clean = dict(value)
    if value["status"] not in ("segment_complete", "interrupted"):
        raise ValueError("Mobile feedback status describes only executed command duration")
    clean["requested_command"] = _mobile_arguments(value["requested_command"])
    requested, executed = value["requested_steps"], value["executed_steps"]
    if (type(requested) is not int or requested != clean["requested_command"]["steps"]
            or type(executed) is not int or not 0 <= executed <= requested):
        raise ValueError("Mobile feedback step counts are inconsistent")
    if value["status"] == "segment_complete" and executed != requested:
        raise ValueError("A complete mobile segment must execute the selected step count")
    if type(value["control_hz"]) is not int or value["control_hz"] != 20:
        raise ValueError("Mobile feedback requires native20Hz")
    duration = _finite(value["simulation_time_s"], "simulation_time_s", minimum=0)
    if not math.isclose(duration, executed / 20, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Mobile simulation time differs from step count")
    clean["simulation_time_s"] = duration
    clean["gripper_command"] = _native_gripper(value["gripper_command"])
    clean["measured_mobile_state"] = validate_mobile_state(value["measured_mobile_state"])
    if not isinstance(value["detail"], str) or not value["detail"] or len(value["detail"]) > 2000:
        raise ValueError("Mobile feedback detail must be a short motion description")
    return clean


def model_request(obs, step: int) -> dict:
    if obs.prompt != TASK_PROMPT:
        raise ValueError("Expected the official RestockPantry task instruction")
    if type(step) is not int or step < 0 or obs.mobile_state["step"] != step:
        raise ValueError("Observation step must match the synchronized robot sensor timestamp")
    geometry = validate_geometry_metadata(obs.geometry)
    mobile = validate_mobile_state(obs.mobile_state)
    def upright(url):
        image = decode_image(url).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image_to_data_url(np.asarray(image), size=512, rotate=False)
    return {"external_image": upright(obs.image), "wrist_image": upright(obs.wrist_image),
            "robot_state": measured_pose(obs.state), "joint_positions": list(obs.joint_position),
            "geometry": geometry, "step": step, "mobile_state": mobile}


class RoboCasaAstranomyClient(AstranomyAgenticClient):
    """Reuse subscription transport/history; replace only task and robot interfaces."""
    history_protocol = HISTORY_PROTOCOL

    def __init__(self, config=None, *, token_provider, log_dir=None, transport=None, clock=time.monotonic):
        config = config or AstraConfig()
        if config.model != "gpt-6-astra":
            raise ValueError("RoboCasa Astranomy requires gpt-6-astra/high")
        if config.max_api_calls is not None or config.max_wall_time_s is not None:
            raise ValueError("Full tasks have no call or wall-time cap")
        self.overload_retries = 0
        self._overload_sleep = time.sleep
        self._overload_retry_attempts = 0
        self._overload_retry_wait_s = 0.0
        self.task_key = TASK_KEY
        # Avoid the LIBERO-only Astranomy task registry completely. Retain its
        # inherited prediction/accepted-output history and subscription machinery.
        ChatGPTPlanHybridClient.__init__(self, config, token_provider=token_provider,
            transport=transport, log_dir=log_dir, clock=clock, image_history=1,
            task_prompt=TASK_PROMPT, reasoning_effort="high")
        self.image_history = None

    @property
    def prompt_version(self):
        return PROMPT_VERSION

    def reset_history(self):
        super().reset_history()
        self._pending_tool_name = None

    def build_payload(self, request, grid=None, mode="astranomy_agentic"):
        if mode != "astranomy_agentic":
            raise ValueError("Only the Astranomy native-tool loop is available")
        if self._pending_call_id is not None:
            raise ValueError("Record execution feedback before requesting another action")
        if not isinstance(request, Mapping) or set(request) != OBSERVATION_KEYS:
            raise ValueError("Observation contains missing or unexpected fields")
        state = _pose(request["robot_state"], "robot_state")
        if type(request["step"]) is not int or request["step"] < 0:
            raise ValueError("step must be a nonnegative integer")
        joints = request["joint_positions"]
        if not isinstance(joints, (list, tuple)) or len(joints) != 7:
            raise ValueError("joint_positions must be seven measured arm joint angles")
        geometry = validate_geometry_metadata(request["geometry"])
        mobile = validate_mobile_state(request["mobile_state"])
        if mobile["step"] != request["step"]:
            raise ValueError("Mobile robot state timestamp differs from the image observation")
        observation = {"type": "observation", "step": request["step"], "task": self.task_prompt,
            "robot_state": state, "joint_positions": [_finite(v, "joint_positions") for v in joints],
            "geometry": geometry, "mobile_state": mobile}
        content = [{"type": "input_text", "text": _canonical(observation)}]
        for field, label in (("external_image", "External RGB camera"), ("wrist_image", "Wrist RGB camera")):
            decoded = decode_image(request[field])
            if decoded.size != (geometry["image"]["width"], geometry["image"]["height"]):
                raise ValueError("Camera image dimensions differ from geometry metadata")
            content.extend([{"type": "input_text", "text": label},
                            {"type": "input_image", "detail": "high", "image_url": request[field]}])
        return {"model": self.config.model, "store": False, "stream": True,
            "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"],
            "instructions": instructions(geometry),
            "input": copy.deepcopy(self._history + [{"role": "user", "content": content}]),
            "tools": [{"type": "namespace", "name": TOOL_NAMESPACE,
                       "description": "Execute one short arm or native mobile command.", "tools": tool_definitions()}],
            "tool_choice": "required", "parallel_tool_calls": False}

    def request_key(self, payload, grid=None, mode="astranomy_agentic"):
        return hashlib.sha256(_canonical({"payload": payload, "client_version": CLIENT_VERSION,
                                         "history_protocol": HISTORY_PROTOCOL}).encode()).hexdigest()

    def validate_response(self, data, grid=None):
        return validate_command(data)

    def predict(self, request, grid=None, mode="astranomy_agentic"):
        with self._history_lock:
            result = super().predict(request, grid, mode)
            self._pending_tool_name = result["name"]
            return result

    def record_result(self, feedback):
        with self._history_lock:
            if self._pending_tool_name == TOOL_NAME:
                checked = validate_arm_feedback(feedback)
                expected = _target_values(self._pending_command["target"], self._pending_command["steps"])
                actual = _target_values(checked["requested_target"], checked["requested_steps"])
                if not np.allclose(actual, expected, rtol=0, atol=1e-12):
                    raise ValueError("Arm feedback requested_target differs from the pending command")
                super().record_result(feedback)
                self._pending_tool_name = None
                return
            if self._pending_tool_name != MOBILE_TOOL_NAME or self._pending_call_id is None:
                raise ValueError("No pending mobile call accepts this feedback")
            checked = validate_mobile_feedback(feedback)
            if checked["requested_command"] != _mobile_arguments(self._pending_command):
                raise ValueError("Mobile feedback command differs from the pending tool call")
            self._history.append({"type": "function_call_output", "call_id": self._pending_call_id,
                                  "output": _canonical(checked)})
            self._feedback_count += 1
            self._pending_call_id = self._pending_command = self._pending_tool_name = None


class MobileExecutor:
    """One 1--5-step native command hold, retaining the previous finger command."""

    def __init__(self, obs, command, gripper):
        self.command = _mobile_arguments(command)
        self.gripper = _native_gripper(gripper)
        self.requested_steps = self.command["steps"]
        self.executed_steps = 0
        self.expected_state = np.array(obs.state, dtype=float, copy=True)
        self.expected_mobile = validate_mobile_state(obs.mobile_state)
        self.pending = None
        self.done = False
        self.status = "running"
        self.detail = "fixed native mobile command pending; duration does not establish navigation or task success"
        self.mobile_command = {k: copy.deepcopy(self.command[k]) for k in ("base", "torso")}

    def next_action(self, obs):
        if self.done:
            raise RuntimeError("mobile segment is already finished")
        if self.pending is not None:
            raise RuntimeError("observe_transition must acknowledge the previous action")
        state = np.asarray(obs.state, dtype=float)
        mobile = validate_mobile_state(obs.mobile_state)
        if not np.array_equal(state, self.expected_state) or mobile != self.expected_mobile:
            raise ValueError("Mobile action observation differs from the latest acknowledged robot state")
        action = (0., 0., 0., 0., 0., 0., self.gripper)
        self.pending = (state.copy(), copy.deepcopy(mobile), action)
        return action

    def observe_transition(self, before_obs, action, after_obs):
        if self.pending is None:
            raise RuntimeError("No issued mobile action awaits acknowledgement")
        state, mobile, issued = self.pending
        before_mobile = validate_mobile_state(before_obs.mobile_state)
        after_mobile = validate_mobile_state(after_obs.mobile_state)
        if (not np.array_equal(np.asarray(before_obs.state), state) or before_mobile != mobile
                or tuple(action) != issued or after_mobile["step"] != mobile["step"] + 1):
            raise ValueError("Mobile acknowledgement differs from the issued action or robot timestamp")
        self.expected_state = np.array(after_obs.state, dtype=float, copy=True)
        self.expected_mobile = after_mobile
        self.pending = None
        self.executed_steps += 1
        if self.executed_steps == self.requested_steps:
            self.done = True
            self.status = "segment_complete"
            self.detail = "selected native mobile steps executed with gripper command kept; inspect fresh RGB and robot motion; no arrival or task-success claim"
        return self.feedback()

    def feedback(self, obs=None):
        mobile = self.expected_mobile if obs is None else validate_mobile_state(obs.mobile_state)
        if mobile != self.expected_mobile:
            raise ValueError("Feedback must use the latest acknowledged mobile observation")
        return {"status": self.status, "requested_command": copy.deepcopy(self.command),
                "measured_mobile_state": copy.deepcopy(mobile), "requested_steps": self.requested_steps,
                "executed_steps": self.executed_steps, "simulation_time_s": self.executed_steps / 20,
                "control_hz": 20, "gripper_command": self.gripper, "detail": self.detail}

    def interrupt(self, detail="Host interrupted the native mobile segment"):
        self.pending = None
        self.done = True
        self.status = "interrupted"
        self.detail = str(detail)
