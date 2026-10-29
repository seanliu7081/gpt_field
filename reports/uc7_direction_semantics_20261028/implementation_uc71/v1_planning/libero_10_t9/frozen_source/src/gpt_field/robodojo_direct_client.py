"""RoboDojo-style absolute-pose tool protocol over ChatGPT subscription OAuth.

This is a new baseline: existing Direct5 clients and recorded runs are untouched.
The model proposes one move_eef or give_up call; the host executes it and returns
robot-only target-arrival feedback. Native Responses function calls are supported
by the existing subscription SSE transport, so no JSON-envelope/API-key fallback
is needed. This module never steps a simulator or interprets encrypted reasoning.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any, Mapping

from .astra import APIError, AstraConfig, SchemaError, _canonical, _strict_loads
from .direct_history import _trim_images
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient
from .observation import decode_image


CLIENT_VERSION = "robodojo-direct-client-v2"
PROMPT_VERSION = "robodojo-direct-panda-zero-shot-v1"
HISTORY_PROTOCOL = "responses_native_pose_tools_full_text_latest_two_image_turns_v1"
TOOL_NAMESPACE = "robot"
POSE_KEYS = ("x", "y", "z", "pitch_deg", "roll_deg", "yaw_deg", "gripper")
MOVE_KEYS = (*POSE_KEYS, "note")
FEEDBACK_KEYS = frozenset({"status", "requested_target", "measured_state", "position_error_m",
                           "orientation_error_deg", "executed_steps", "detail"})
OPTIONAL_FEEDBACK_KEYS = frozenset({"gripper_applied", "gripper_error"})


def _finite(value: Any, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _pose(value: Any, name: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(POSE_KEYS):
        raise ValueError(f"{name} must contain exactly the seven pose/gripper fields")
    return {key: _finite(value[key], f"{name}.{key}",
                         minimum=0 if key == "gripper" else None,
                         maximum=1 if key == "gripper" else None) for key in POSE_KEYS}


def tool_definitions() -> list[dict]:
    """Strict nullable properties express partial absolute pose updates."""
    props = {}
    for key in POSE_KEYS:
        numeric = {"type": "number"}
        if key == "gripper":
            numeric.update(minimum=0, maximum=1)
        props[key] = {"anyOf": [numeric, {"type": "null"}]}
    props["note"] = {"anyOf": [{"type": "string", "maxLength": 1000}, {"type": "null"}]}
    return [
        {"type": "function", "name": "move_eef", "strict": True,
         "description": "Move the single Panda grasp point to a partial absolute world pose. "
                        "Use null for unchanged fields. Apply a supplied gripper command after arm arrival.",
         "parameters": {"type": "object", "properties": props, "required": list(props),
                        "additionalProperties": False}},
        {"type": "function", "name": "give_up", "strict": True,
         "description": "End this unsuccessful attempt if you cannot continue. This never declares success.",
         "parameters": {"type": "object", "properties": {
             "reason": {"type": "string", "minLength": 1, "maxLength": 1000}},
             "required": ["reason"], "additionalProperties": False}},
    ]


def instructions(task_key: str = "bowl_plate") -> str:
    task = resolve_distance_task(task_key)
    recipe = (
        "Locate the requested bowl and plate; grasp and lift the bowl; carry it over the plate; "
        "lower and release it onto the plate; withdraw and inspect the result."
        if task.key == "bowl_plate" else
        "Locate the top-drawer handle; grasp it and pull the drawer open; release the handle; "
        "grasp and lift the bowl; carry it to the open drawer; lower and release it inside; "
        "withdraw and inspect the result."
    )
    return (
        f"Interface {PROMPT_VERSION}. You control one simulated Franka Panda arm. Task: {task.prompt}. "
        "Choose exactly one move_eef or give_up tool call per turn. Use the external and wrist RGB "
        "views, measured robot state, and previous actions and execution results to decide. "
        "The latest two image turns and every prior textual observation, accepted response and "
        "execution result remain available. Any encrypted reasoning is replayed without inspection. "
        "The robot_state contains the world position of the grasp point between the fingers in "
        "metres, measured orientation in degrees, and measured gripper opening (0=closed, 1=open). "
        "Optional joint_positions are seven measured arm angles in radians, provided only as "
        "read-only proprioception. No object ground-truth poses, depth, camera calibration, contact "
        "labels, task-progress measurements, demonstrations or successful action trajectories are supplied. "
        "All x/y/z commands are ABSOLUTE in the fixed simulation world frame; +z is upward. "
        "Angles follow the RoboDojo naming convention: pitch_deg rotates about fixed world +x, "
        "roll_deg about +y, yaw_deg about +z. The orientation matrix is "
        "Rz(yaw) Ry(roll) Rx(pitch) Rx(pi); all three angles zero point the gripper straight down. "
        "Commands refer to the grasp point, not the wrist flange. Omitted or null arm fields retain "
        "the CURRENT MEASURED pose; an omitted or null gripper field retains the prior gripper command. "
        "A gripper value of 0 closes and 1 opens; intermediate openings are allowed. note is an "
        "optional short observable explanation, not a required reasoning trace. Supply at least one "
        "non-null movement or gripper field. "
        "The local adapter uses Cartesian target motion, robot kinematics and the existing OSC "
        "controller at 20 Hz. It is not cuRobo and performs no automatic object grasping or task "
        "planning. A tool has at most 8.5 simulated seconds for arm motion and, if requested, "
        "1.5 seconds for the gripper; a path too long for this bound is rejected rather than sped up. "
        "Simulation pauses during inference. It returns after measured target arrival "
        "or a local motion timeout/rejection, with fresh images and robot-only arrival errors. "
        "A combined arm/gripper call first moves with the previous gripper command, then applies "
        "the new opening only after arm arrival. A failed arm move does not apply a new gripper "
        "command. Inspect the returned status and measured pose before choosing the next target; "
        "requesting a pose does not prove arrival, and closing does not prove a grasp. "
        f"Task-stage recipe (instruction only, not a demonstration or measured stage): {recipe} "
        "LIBERO alone evaluates task success; no model-controlled success action exists. "
        "give_up ends the attempt as unsuccessful, never as completed. There is no episode action, "
        "call or wall-time cap; local move bounds do not end the task."
    )


def validate_command(value: Any) -> dict:
    try:
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("Command must contain exactly name and arguments")
        name, args = value["name"], value["arguments"]
        if not isinstance(args, dict):
            raise ValueError("Function arguments must be an object")
        if name == "give_up":
            if set(args) != {"reason"} or not isinstance(args["reason"], str) \
                    or not args["reason"].strip() or len(args["reason"]) > 1000:
                raise ValueError("give_up requires one nonempty reason of at most 1000 characters")
            return copy.deepcopy(value)
        if name != "move_eef" or set(args) - set(MOVE_KEYS):
            raise ValueError("Only move_eef and give_up with their declared arguments are allowed")
        checked = {}
        for key, item in args.items():
            if item is None:
                continue
            if key == "note":
                if not isinstance(item, str) or len(item) > 1000:
                    raise ValueError("note must be a string of at most 1000 characters")
                checked[key] = item
            else:
                checked[key] = _finite(item, key, minimum=0 if key == "gripper" else None,
                                       maximum=1 if key == "gripper" else None)
        if not set(checked).intersection(POSE_KEYS):
            raise ValueError("move_eef needs at least one non-null pose or gripper field")
        return {"name": name, "arguments": checked}
    except ValueError as exc:
        raise SchemaError(str(exc)) from None


def validate_feedback(value: Any) -> dict:
    if not isinstance(value, Mapping) or not FEEDBACK_KEYS <= set(value) \
            or set(value) - FEEDBACK_KEYS - OPTIONAL_FEEDBACK_KEYS:
        raise ValueError("Execution result has missing or unexpected fields")
    if value["status"] not in ("reached", "incomplete", "rejected", "user_stop"):
        raise ValueError("Execution status must describe robot motion only")
    if type(value["executed_steps"]) is not int or value["executed_steps"] < 0:
        raise ValueError("executed_steps must be a nonnegative integer")
    if not isinstance(value["detail"], str) or len(value["detail"]) > 2000:
        raise ValueError("detail must be a short robot-motion description")
    result = dict(value)
    for key in ("requested_target", "measured_state"):
        result[key] = _pose(value[key], key)
    for key in ("position_error_m", "orientation_error_deg", "gripper_error"):
        if key in value:
            result[key] = _finite(value[key], key, minimum=0)
    if "gripper_applied" in value and type(value["gripper_applied"]) is not bool:
        raise ValueError("gripper_applied must be boolean")
    return result


class RoboDojoDirectClient(ChatGPTPlanHybridClient):
    """One native robot tool call per request, shared audited OAuth transport."""

    history_protocol = HISTORY_PROTOCOL

    def __init__(self, config=None, *, token_provider, task_key="bowl_plate", transport=None,
                 log_dir=None, clock=time.monotonic, image_history=2, reasoning_effort="medium"):
        config = config or AstraConfig()
        if config.model != "gpt-6-astra" or reasoning_effort != "medium":
            raise ValueError("RoboDojo Direct requires gpt-6-astra with medium reasoning")
        if config.max_api_calls is not None or config.max_wall_time_s is not None:
            raise ValueError("RoboDojo Direct full tasks have no episode call or wall-time cap")
        if type(image_history) is not int or image_history != 2:
            raise ValueError("RoboDojo Direct retains exactly the latest two image turns")
        self.task_spec = resolve_distance_task(task_key)
        self.task_key = self.task_spec.key
        super().__init__(config, token_provider=token_provider, transport=transport, log_dir=log_dir,
                         clock=clock, image_history=image_history, task_prompt=self.task_spec.prompt,
                         reasoning_effort=reasoning_effort)

    @property
    def prompt_version(self):
        return PROMPT_VERSION

    def reset_history(self):
        super().reset_history()
        self._pending_call_id = None
        self._gave_up = False

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_PROTOCOL, pending_tool_result=self._pending_call_id is not None,
                      text_history="all_observations_native_outputs_and_robot_tool_results")
        return result

    def build_payload(self, request, grid=None, mode="robodojo_direct"):
        if mode != "robodojo_direct":
            raise ValueError("RoboDojo Direct supports only robodojo_direct mode")
        if self._gave_up:
            raise ValueError("give_up has ended this attempt; start a new episode explicitly")
        if self._pending_call_id is not None:
            raise ValueError("Record execution feedback before requesting another action")
        required = {"external_image", "wrist_image", "robot_state", "step"}
        if not isinstance(request, Mapping) or not required <= set(request) \
                or set(request) - required - {"joint_positions"}:
            raise ValueError("Request has missing or unexpected observation fields")
        state = _pose(request["robot_state"], "robot_state")
        if type(request["step"]) is not int or request["step"] < 0:
            raise ValueError("step must be a nonnegative integer")
        observation = {"type": "observation", "step": request["step"],
                       "task": self.task_prompt, "robot_state": state}
        if "joint_positions" in request:
            joints = request["joint_positions"]
            if not isinstance(joints, (list, tuple)) or len(joints) != 7:
                raise ValueError("joint_positions must be seven measured arm joint angles")
            observation["joint_positions"] = [_finite(v, "joint_positions") for v in joints]
        content = [{"type": "input_text", "text": _canonical(observation)}]
        for field, label in (("external_image", "External RGB camera"),
                             ("wrist_image", "Wrist RGB camera")):
            decode_image(request[field])
            content.extend([{"type": "input_text", "text": label},
                            {"type": "input_image", "detail": "high", "image_url": request[field]}])
        return {"model": self.config.model, "store": False, "stream": True,
                "reasoning": {"effort": "medium"}, "include": ["reasoning.encrypted_content"],
                "instructions": instructions(self.task_key),
                "input": _trim_images(self._history + [{"role": "user", "content": content}], 2),
                # Sign-in-with-ChatGPT documents function/custom tools inside
                # namespaces (or additional_tools), unlike flat API-key tools.
                "tools": [{"type": "namespace", "name": TOOL_NAMESPACE,
                           "description": "Single-arm robot motion and unsuccessful termination.",
                           "tools": tool_definitions()}],
                "tool_choice": "required", "parallel_tool_calls": False}

    def request_key(self, payload, grid=None, mode="robodojo_direct"):
        return hashlib.sha256(_canonical({"payload": payload, "client_version": CLIENT_VERSION,
                                         "history_protocol": HISTORY_PROTOCOL}).encode()).hexdigest()

    @staticmethod
    def _extract(response):
        if response.get("error") or response.get("status") != "completed":
            raise APIError("Incomplete/failed Responses output is never executed")
        output = response.get("output")
        if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
            raise SchemaError("Response must contain a complete output-item list")
        calls = []
        for item in output:
            if item.get("status", "completed") != "completed":
                raise SchemaError("Incomplete output item is never executed")
            if item.get("type") == "function_call":
                calls.append(item)
            elif item.get("type") == "message":
                content = item.get("content")
                if not isinstance(content, list) or any(not isinstance(p, dict) for p in content):
                    raise SchemaError("Malformed assistant message")
                if any(p.get("type") == "refusal" for p in content):
                    raise APIError("Astra refused this request; no retry or fallback")
            elif item.get("type") != "reasoning":
                raise SchemaError("Unexpected model tool/output type")
        if len(calls) != 1:
            raise SchemaError("Expected exactly one move_eef or give_up function call")
        call = calls[0]
        # Responses preserves the bare function name and supplies namespace in
        # a separate optional field. Never execute a same-named foreign tool.
        if "namespace" in call and call["namespace"] != TOOL_NAMESPACE:
            raise SchemaError("Function call namespace must be robot when supplied")
        if not isinstance(call.get("call_id"), str) or not call["call_id"] \
                or not isinstance(call.get("arguments"), str):
            raise SchemaError("Function call requires a call_id and JSON arguments")
        try:
            args = _strict_loads(call["arguments"])
        except (ValueError, TypeError):
            raise SchemaError("Function arguments are not strict JSON") from None
        return {"name": call.get("name"), "arguments": args}

    def validate_response(self, data, grid=None):
        return validate_command(data)

    def predict(self, request, grid=None, mode="robodojo_direct"):
        with self._history_lock:
            result = super().predict(request, grid, mode)
            call = next(item for item in self.last_response["output"] if item.get("type") == "function_call")
            self._pending_call_id = call["call_id"]
            self._gave_up = result["name"] == "give_up"
            return result

    def record_result(self, feedback):
        """Pair one observable execution result with the accepted native call."""
        with self._history_lock:
            if self._pending_call_id is None or self._gave_up:
                raise ValueError("No outstanding move_eef call accepts an execution result")
            checked = validate_feedback(feedback)
            self._history.append({"type": "function_call_output", "call_id": self._pending_call_id,
                                  "output": _canonical(checked)})
            self._feedback_count += 1
            self._pending_call_id = None
