"""Astranomy-style short-target Agentic loop over ChatGPT subscription OAuth.

This client retains the complete accepted visual/text/tool transcript. It reuses
the audited subscription SSE transport, not the API-key client or Codex CLI.
Only the robot target tool is exposed; motion and LIBERO success remain external.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from typing import Any, Mapping

from .astra import APIError, AstraConfig, SchemaError, _canonical, _strict_loads
from .astranomy_agentic_geometry import validate_geometry_metadata
from .astranomy_agentic_prompt import PROMPT_VERSION, instructions
from .distance_tasks import resolve_distance_task
from .hybrid_subscription import ChatGPTPlanHybridClient
from .observation import decode_image


CLIENT_VERSION = "astranomy-agentic-subscription-client-v1"
HISTORY_PROTOCOL = "responses_native_short_target_full_visual_and_text_history_v1"
TOOL_NAMESPACE = "robot"
TOOL_NAME = "execute_eef_target"
OBSERVATION_KEYS = frozenset({"external_image", "wrist_image", "robot_state",
                              "joint_positions", "geometry", "step"})
FEEDBACK_KEYS = frozenset({"status", "requested_target", "measured_pose", "position_error_m",
                           "orientation_error_rad", "requested_steps", "executed_steps",
                           "simulation_time_s", "control_hz", "gripper_command", "detail"})


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


def _pose(value: Any, name: str, *, command=False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 8:
        raise ValueError(f"{name} must be the complete eight-value absolute pose")
    pose = [_finite(v, name) for v in value]
    if not any(pose[3:7]):
        raise ValueError(f"{name} quaternion must have nonzero norm")
    if not 0 <= pose[7] <= 1 or (command and pose[7] not in (0, 1)):
        raise ValueError(f"{name} gripper must be " + ("binary 0=open,1=closed" if command else "in [0,1]"))
    return pose


def tool_definitions() -> list[dict]:
    return [{"type": "function", "name": TOOL_NAME, "strict": True,
             "description": "Repeat one complete absolute robot target for 1 through 5 control steps. "
                            "target=[x,y,z,qw,qx,qy,qz,gripper]; gripper is binary 0=open,1=closed. "
                            "The selected steps do not guarantee target arrival.",
             "parameters": {"type": "object", "properties": {
                 "target": {"type": "array", "items": {"type": "number"},
                            "minItems": 8, "maxItems": 8},
                 "steps": {"type": "integer", "minimum": 1, "maximum": 5}},
                 "required": ["target", "steps"], "additionalProperties": False}}]


def validate_command(value: Any) -> dict:
    """Validate representation only; the motion tool checks measured-pose limits."""
    try:
        if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
            raise ValueError("Command must contain exactly name and arguments")
        if value["name"] != TOOL_NAME:
            raise ValueError("Only execute_eef_target is allowed")
        args = value["arguments"]
        if not isinstance(args, dict) or set(args) != {"target", "steps"}:
            raise ValueError("execute_eef_target requires exactly target and steps")
        if type(args["steps"]) is not int or not 1 <= args["steps"] <= 5:
            raise ValueError("steps must be an integer from 1 through 5")
        return {"name": TOOL_NAME, "arguments": {
            "target": _pose(args["target"], "target", command=True), "steps": args["steps"]}}
    except ValueError as exc:
        raise SchemaError(str(exc)) from None


def validate_feedback(value: Any) -> dict:
    if not isinstance(value, Mapping) or set(value) != FEEDBACK_KEYS:
        raise ValueError("Execution result has missing or unexpected fields")
    if value["status"] not in ("segment_complete", "interrupted", "rejected"):
        raise ValueError("Execution status must describe the robot segment only")
    requested, executed = value["requested_steps"], value["executed_steps"]
    if type(requested) is not int or not 1 <= requested <= 5:
        raise ValueError("requested_steps must be an integer from 1 through 5")
    if type(executed) is not int or not 0 <= executed <= requested:
        raise ValueError("executed_steps must be an integer between zero and requested_steps")
    if value["status"] == "rejected" and executed != 0:
        raise ValueError("Rejected targets must execute zero steps")
    if value["status"] == "segment_complete" and executed != requested:
        raise ValueError("A complete segment must execute its selected steps")
    if type(value["control_hz"]) is not int or value["control_hz"] != 20:
        raise ValueError("control_hz must be the native 20 Hz")
    if type(value["gripper_command"]) is not int or value["gripper_command"] not in (0, 1):
        raise ValueError("gripper_command must be binary 0=open,1=closed")
    if not isinstance(value["detail"], str) or len(value["detail"]) > 2000:
        raise ValueError("detail must be a short robot-motion description")
    result = dict(value)
    result["requested_target"] = _pose(value["requested_target"], "requested_target", command=True)
    result["measured_pose"] = _pose(value["measured_pose"], "measured_pose")
    for key in ("position_error_m", "orientation_error_rad", "simulation_time_s"):
        result[key] = _finite(value[key], key, minimum=0)
    if not math.isclose(result["simulation_time_s"], executed / 20, rel_tol=0, abs_tol=1e-9):
        raise ValueError("simulation_time_s must match executed steps at 20 Hz")
    if result["requested_target"][7] != result["gripper_command"]:
        raise ValueError("gripper_command must match requested_target")
    return result


class AstranomyAgenticClient(ChatGPTPlanHybridClient):
    """One native short-target tool call with all accepted observation history."""

    history_protocol = HISTORY_PROTOCOL

    def __init__(self, config=None, *, token_provider, task_key="bowl_plate", transport=None,
                 log_dir=None, clock=time.monotonic, image_history=None, reasoning_effort="medium"):
        config = config or AstraConfig()
        if config.model != "gpt-6-astra" or reasoning_effort != "medium":
            raise ValueError("Astranomy Agentic requires gpt-6-astra with medium reasoning")
        if config.max_api_calls is not None or config.max_wall_time_s is not None:
            raise ValueError("Astranomy Agentic full tasks have no episode call or wall-time cap")
        if image_history is not None:
            raise ValueError("Astranomy Agentic retains every image turn; no image-history limit")
        self.task_spec = resolve_distance_task(task_key)
        self.task_key = self.task_spec.key
        # The shared constructor requires a positive legacy image-history value.
        # This subclass never invokes its image trimming path and reports no limit.
        super().__init__(config, token_provider=token_provider, transport=transport, log_dir=log_dir,
                         clock=clock, image_history=1, task_prompt=self.task_spec.prompt,
                         reasoning_effort=reasoning_effort)
        self.image_history = None

    @property
    def prompt_version(self):
        return PROMPT_VERSION

    def reset_history(self):
        super().reset_history()
        self._pending_call_id = None
        self._pending_command = None

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_PROTOCOL, pending_tool_result=self._pending_call_id is not None,
                      image_history_limit=None,
                      text_history="all_observations_native_outputs_and_robot_tool_results")
        return result

    def build_payload(self, request, grid=None, mode="astranomy_agentic"):
        if mode != "astranomy_agentic":
            raise ValueError("Astranomy Agentic supports only astranomy_agentic mode")
        if self._pending_call_id is not None:
            raise ValueError("Record execution feedback before requesting another action")
        if not isinstance(request, Mapping) or set(request) != OBSERVATION_KEYS:
            raise ValueError("Request has missing or unexpected observation fields")
        state = _pose(request["robot_state"], "robot_state")
        if type(request["step"]) is not int or request["step"] < 0:
            raise ValueError("step must be a nonnegative integer")
        joints = request["joint_positions"]
        if not isinstance(joints, (list, tuple)) or len(joints) != 7:
            raise ValueError("joint_positions must be seven measured arm joint angles")
        geometry = validate_geometry_metadata(request["geometry"])
        observation = {"type": "observation", "step": request["step"], "task": self.task_prompt,
                       "robot_state": state, "joint_positions": [_finite(v, "joint_positions") for v in joints],
                       "geometry": geometry}
        content = [{"type": "input_text", "text": _canonical(observation)}]
        for field, label in (("external_image", "External RGB camera"), ("wrist_image", "Wrist RGB camera")):
            decoded = decode_image(request[field])
            if decoded.size != (geometry["image"]["width"], geometry["image"]["height"]):
                raise ValueError("Camera image dimensions differ from geometry metadata")
            content.extend([{"type": "input_text", "text": label},
                            {"type": "input_image", "detail": "high", "image_url": request[field]}])
        return {"model": self.config.model, "store": False, "stream": True,
                "reasoning": {"effort": "medium"}, "include": ["reasoning.encrypted_content"],
                "instructions": instructions(self.task_key, geometry),
                "input": copy.deepcopy(self._history + [{"role": "user", "content": content}]),
                "tools": [{"type": "namespace", "name": TOOL_NAMESPACE,
                           "description": "Execute one short absolute robot target.", "tools": tool_definitions()}],
                "tool_choice": "required", "parallel_tool_calls": False}

    def request_key(self, payload, grid=None, mode="astranomy_agentic"):
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
            raise SchemaError("Expected exactly one execute_eef_target function call")
        call = calls[0]
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

    def predict(self, request, grid=None, mode="astranomy_agentic"):
        with self._history_lock:
            result = super().predict(request, grid, mode)
            call = next(item for item in self.last_response["output"] if item.get("type") == "function_call")
            self._pending_call_id = call["call_id"]
            self._pending_command = copy.deepcopy(result["arguments"])
            return result

    def record_result(self, feedback):
        """Pair robot-only segment feedback with exactly one accepted native call."""
        with self._history_lock:
            if self._pending_call_id is None:
                raise ValueError("No outstanding execute_eef_target call accepts an execution result")
            checked = validate_feedback(feedback)
            if checked["requested_steps"] != self._pending_command["steps"]:
                raise ValueError("Feedback requested_steps differs from the pending command")
            self._history.append({"type": "function_call_output", "call_id": self._pending_call_id,
                                  "output": _canonical(checked)})
            self._feedback_count += 1
            self._pending_call_id = None
            self._pending_command = None
