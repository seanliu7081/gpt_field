"""Direct OSC actions with explicit, observable conversation history.

This is a separate policy from the frozen field experiment. It reuses the
audited Astra transport, accounting, and five-by-seven action validator. History
contains observations, complete ordered output items from validated Responses,
and measured robot-state feedback. Encrypted reasoning is retained as opaque
API data, never inspected or rendered. Every textual record is retained. Only
old image pairs are omitted. Context exhaustion is a terminal API error, never
silent truncation.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from .action import ROTATION_SCALE_RAD, TRANSLATION_SCALE_M
from .astra import AstraClient, _canonical, _numbers, validate_prediction
from .observation import PolicyObservation
from .policy import Decision
from .schema import GridSpec


TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
PROMPT_VERSION = "astra-direct-history-v1"
HISTORY_PROTOCOL = "responses_output_transcript_v1"
CHUNK_SIZE = 5
PROGRESS_WINDOW = 25


def _allowed(observation: Mapping[str, Any]) -> PolicyObservation:
    result = PolicyObservation.from_dict(observation)
    if result.prompt != TASK:
        raise ValueError("Task text does not match the requested bowl task")
    return result


def _trim_images(messages: list[dict], keep: int) -> list[dict]:
    """Copy and omit old images, preserving every original textual record."""
    result = copy.deepcopy(messages)
    image_messages = [i for i, message in enumerate(result)
                      if message.get("role") == "user" and isinstance(message.get("content"), list)
                      and any(part.get("type") == "input_image" for part in message["content"])]
    for index in image_messages[:-keep]:
        result[index]["content"] = [
            {"type": "input_text", "text": "[Older camera image omitted by fixed image-retention rule.]"}
            if part.get("type") == "input_image" else part
            for part in result[index]["content"]
        ]
    return result


class DirectHistoryAstraClient(AstraClient):
    """One manually managed transcript; no server-side conversation storage.

    The inherited ``predict`` performs all API work. Overrides customize only
    payload/provenance, transcript commit, and interrupted-request accounting.
    Failed requests and schema retries never commit a conversation turn.
    """

    def __init__(self, config=None, *, image_history: int = 2, **kwargs):
        if type(image_history) is not int or image_history < 1:
            raise ValueError("image_history must be a positive integer")
        super().__init__(config, **kwargs)
        self.image_history = image_history
        self._history_lock = threading.RLock()
        self._history_request_settled = True
        self.reset_history()

    def reset_history(self) -> None:
        """Start a new episode without changing the lifetime API ledger."""
        with self._history_lock:
            self._history: list[dict] = []
            self._observation_count = 0
            self._feedback_count = 0

    def history_snapshot(self) -> list[dict]:
        with self._history_lock:
            return copy.deepcopy(self._history)

    def history_stats(self) -> dict:
        with self._history_lock:
            return {
                "protocol": HISTORY_PROTOCOL,
                "observations": self._observation_count,
                "feedback_records": self._feedback_count,
                "messages": len(self._history),
                "retained_image_pairs": sum(
                    message.get("role") == "user" and isinstance(message.get("content"), list)
                    and any(part.get("type") == "input_image" for part in message["content"])
                    for message in self._history),
                "image_history_limit": self.image_history,
                "text_history": "all_observations_validated_responses_output_and_state_feedback",
                "reasoning_replay": "opaque_encrypted_output_items",
            }

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "direct") -> dict:
        if mode != "direct":
            raise ValueError("DirectHistoryAstraClient supports only direct mode")
        grid = grid or GridSpec()
        if grid.frame != "world":
            raise ValueError("The direct-history baseline requires the verified world frame")
        allowed = _allowed(observation)
        with self._history_lock:
            payload = super().build_payload(allowed.to_dict(), grid, mode)
            observation_text = {
                "type": "observation", "observation_index": self._observation_count,
                "prompt": allowed.prompt, "observation/state": list(allowed.state),
            }
            payload["input"][0]["content"][0]["text"] = _canonical(observation_text)
            payload["input"] = _trim_images(self._history + payload["input"], self.image_history)
            payload["include"] = ["reasoning.encrypted_content"]
            payload["instructions"] += (f" Only the most recent {self.image_history} observation "
                                        "image pairs are included; earlier textual observations and "
                                        "execution feedback remain in this conversation.")
            payload["text"]["format"]["name"] = "astra_direct_history_v1"
            return payload

    @staticmethod
    def _instructions(grid: GridSpec, mode: str) -> str:
        return (
            f"Interface {PROMPT_VERSION}. Control the robot to complete the task using the two "
            "RGB camera views, robot state, and prior observation/action/execution-feedback "
            "history below. Prior valid model responses are also retained in this conversation. "
            "Older images may be omitted under the stated retention rule; all textual records remain. "
            "No object poses, depth, contact, rewards, benchmark success, or future images are supplied. "
            "State has EIGHT values: [end-effector position xyz in meters, orientation axis-angle xyz "
            "in radians, two gripper joint positions]. These are NOT seven robot joint angles. "
            "All positions and rotation increments use the world frame. Return exactly FIVE "
            "consecutive LIBERO OSC actions, each with SEVEN normalized numbers in [-1,1]: "
            "[translation dx,dy,dz, rotation axis-angle dx,dy,dz, gripper]. Translation 1 means "
            "a 0.05-meter controller delta target per axis; rotation 1 means a 0.5-radian "
            "world-frame delta target. Gripper +1 closes, -1 opens; 0 commands zero incremental "
            "gripper movement, it does not automatically close or certify a grasp. Zero translation "
            "and rotation request no change from the current measured pose. Each action's delta "
            "target is relative to the current measured pose, not the previous requested target. "
            "Actions execute at 20 Hz: five actions "
            "span 0.25 seconds of simulation, then new images and measured results are supplied. "
            "Controller targets are NOT guaranteed realized motion. Compare commands with measured "
            "movement and reconsider repeated ineffective actions, alignment, and approach direction. "
            "Manage grasping and task progression yourself; judge grasp and placement from visual "
            "and robot-state evidence. Do not assume that closing the gripper proves a grasp. "
            "Feedback progress statistics describe robot motion only, not object motion or task "
            "success, and the controller does not automatically recover, change phase, or close "
            "the gripper for you. Return only JSON matching the actions schema."
        )

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
                    "prompt_version": PROMPT_VERSION, "endpoint": self.config.endpoint,
                    "history_protocol": HISTORY_PROTOCOL, "image_history": self.image_history}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _write_log(self, path: Path | None, data: Any) -> None:
        # The base client's top-level prompt version is a module constant. Replace
        # it only in this subclass's records; frozen field logs stay untouched.
        if isinstance(data, dict) and "payload" in data and "request_ordinal" in data:
            data = {**data, "prompt_version": PROMPT_VERSION,
                    "history_protocol": HISTORY_PROTOCOL, "image_history": self.image_history,
                    "history_stats_before_request": self.history_stats()}
            self._history_request_settled = False

        def redact(value):
            if isinstance(value, str) and self._api_key:
                return value.replace(self._api_key, "[REDACTED]")
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [redact(item) for item in value]
            return value

        super()._write_log(path, redact(data))

    def _record_usage(self, response, reservation) -> None:
        self._history_request_settled = True
        super()._record_usage(response, reservation)

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "direct") -> dict:
        grid = grid or GridSpec()
        with self._history_lock:
            initial_calls = self._api_calls
            try:
                # Build twice deliberately: preview is side-effect free; inherited
                # retry logic reuses one identical payload and validates strictly.
                payload = self.build_payload(observation, grid, mode)
                result = super().predict(observation, grid, mode)
            except BaseException as exc:
                if not isinstance(exc, Exception) and self._api_calls > initial_calls:
                    # The base client catches Exception, not process interrupts.
                    # Its full reservation was already booked before transport.
                    if not self._history_request_settled:
                        self._unknown_usage_calls += 1
                    if self.last_log_path is not None:
                        result_path = Path(self.last_log_path.replace(".request.json", ".result.json"))
                        if not result_path.exists():
                            self._write_log(result_path, {
                                "status": "error", "error_type": type(exc).__name__,
                                "error": "Request interrupted; unsettled usage retains full reservation",
                                "budget": self.budget_snapshot(),
                            })
                raise
            # Preserve ordering, item IDs, assistant phase, and opaque encrypted
            # reasoning exactly as returned. Invalid responses never reach here.
            self._history = payload["input"] + copy.deepcopy(self.last_response["output"])
            self._observation_count += 1
            return copy.deepcopy(result)

    def record_transition(self, feedback: Mapping[str, Any]) -> None:
        """Append only our exact observable-feedback schema, never evaluator data."""
        expected = {
            "type", "step", "chunk_index", "chunk_step", "before_state", "after_state",
            "normalized_action", "desired_translation_delta_m", "desired_world_rotation_delta_rad",
            "measured_translation_delta_m", "measured_translation_norm_m",
            "commanded_translation_norm_m", "translation_tracking_error_m",
            "gripper_joint_delta", "gripper_command", "rolling_progress",
        }
        progress_expected = {
            "window_size", "translation_command_steps", "low_motion_translation_steps",
            "commanded_translation_path_m", "measured_translation_path_m",
            "measured_net_translation_m", "gripper_close_steps",
        }
        if set(feedback) != expected or feedback.get("type") != "execution_feedback":
            raise ValueError("Feedback contains missing or non-allowlisted fields")
        progress = feedback["rolling_progress"]
        if not isinstance(progress, dict) or set(progress) != progress_expected:
            raise ValueError("Progress feedback contains non-allowlisted fields")
        for key in ("step", "chunk_index", "chunk_step"):
            if type(feedback[key]) is not int or feedback[key] < 0:
                raise ValueError(f"Feedback {key} must be a nonnegative integer")
        if feedback["chunk_step"] >= CHUNK_SIZE:
            raise ValueError("Feedback chunk_step exceeds five-action chunk")
        for key, length in (("before_state", 8), ("after_state", 8), ("normalized_action", 7),
                            ("desired_translation_delta_m", 3), ("desired_world_rotation_delta_rad", 3),
                            ("measured_translation_delta_m", 3), ("gripper_joint_delta", 2)):
            _numbers(feedback[key], length, key)
        scalar_keys = ("measured_translation_norm_m", "commanded_translation_norm_m",
                       "translation_tracking_error_m", "gripper_command")
        for value in [*(feedback[key] for key in scalar_keys), *progress.values()]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Feedback statistics must contain only finite numbers")
        with self._history_lock:
            self._history.append({"role": "user", "content": [{
                "type": "input_text", "text": _canonical(dict(feedback)),
            }]})
            self._feedback_count += 1


class DirectHistoryPolicy:
    """Execute all five direct actions, feeding each observed transition back.

    No local field planner, action clipping, grasp state machine, forced closure,
    or stall recovery is installed. Stagnation statistics inform the next model
    request without changing its chosen commands. Evaluator success is never an
    input to this object.
    """

    def __init__(self, client: DirectHistoryAstraClient):
        self.client = client
        self.grid = GridSpec()
        self.method = "direct_history"
        self.update_steps = CHUNK_SIZE
        self.reset()

    def reset(self) -> None:
        self.client.reset_history()
        self.step_index = 0
        self.prediction = None
        self.chunk: list[list[float]] = []
        self.latest_feedback = None
        self._pending: tuple[PolicyObservation, Decision] | None = None
        self._expected_observation: PolicyObservation | None = None
        self._progress: list[dict] = []

    def act(self, observation: PolicyObservation) -> Decision:
        allowed = _allowed(observation.to_dict())
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next action")
        if self._expected_observation is not None and allowed != self._expected_observation:
            raise ValueError("Next observation must equal the preceding transition's after observation")
        refreshed = not self.chunk
        chunk_index, chunk_step = divmod(self.step_index, CHUNK_SIZE)
        details = {"step": self.step_index, "method": self.method, "refreshed": refreshed,
                   "chunk_index": chunk_index, "chunk_step": chunk_step, "status": "direct_action"}
        if refreshed:
            started = time.monotonic()
            raw = self.client.predict(allowed.to_dict(), self.grid, mode="direct")
            validate_prediction(raw, self.grid, "direct")
            self.prediction = copy.deepcopy(raw)
            self.chunk = [[float(value) for value in action] for action in raw["actions"]]
            details.update({"prediction": copy.deepcopy(raw),
                            "api_latency_s": time.monotonic() - started,
                            "api_request_key": self.client.last_request_key,
                            "api_request_log": self.client.last_log_path})
        details["history_stats"] = self.client.history_stats()
        details["feedback"] = copy.deepcopy(self.latest_feedback)
        decision = Decision(tuple(self.chunk.pop(0)), details)
        self._pending = (allowed, decision)
        self.step_index += 1
        return decision

    def observe_transition(self, before: PolicyObservation, decision: Decision,
                           after: PolicyObservation) -> dict:
        before = _allowed(before.to_dict())
        after = _allowed(after.to_dict())
        if self._pending is None:
            raise RuntimeError("No action is awaiting transition feedback")
        pending_before, pending_decision = self._pending
        if before != pending_before or decision is not pending_decision:
            raise ValueError("Feedback must correspond to the exact pending observation and decision")
        action = _numbers(decision.action, 7, "executed action")
        if any(abs(value) > 1 for value in action):
            raise ValueError("Executed action must be normalized to [-1,1]")
        desired = [TRANSLATION_SCALE_M * value for value in action[:3]]
        measured = [after.state[i] - before.state[i] for i in range(3)]
        norm = lambda values: math.sqrt(sum(value * value for value in values))
        row = {"before": before.state[:3], "after": after.state[:3],
               "commanded": norm(desired), "measured": norm(measured), "close": action[6] > 0}
        progress = (self._progress + [row])[-PROGRESS_WINDOW:]
        summary = {
            "window_size": len(progress),
            "translation_command_steps": sum(item["commanded"] >= 0.001 for item in progress),
            "low_motion_translation_steps": sum(item["commanded"] >= 0.001
                                                  and item["measured"] <= 0.0001 for item in progress),
            "commanded_translation_path_m": sum(item["commanded"] for item in progress),
            "measured_translation_path_m": sum(item["measured"] for item in progress),
            "measured_net_translation_m": norm([progress[-1]["after"][i] - progress[0]["before"][i]
                                                   for i in range(3)]),
            "gripper_close_steps": sum(item["close"] for item in progress),
        }
        step = self.step_index - 1
        feedback = {
            "type": "execution_feedback", "step": step,
            "chunk_index": step // CHUNK_SIZE, "chunk_step": step % CHUNK_SIZE,
            "before_state": list(before.state), "after_state": list(after.state),
            "normalized_action": action, "desired_translation_delta_m": desired,
            "desired_world_rotation_delta_rad": [ROTATION_SCALE_RAD * value for value in action[3:6]],
            "measured_translation_delta_m": measured,
            "measured_translation_norm_m": norm(measured), "commanded_translation_norm_m": norm(desired),
            "translation_tracking_error_m": norm([measured[i] - desired[i] for i in range(3)]),
            "gripper_joint_delta": [after.state[i] - before.state[i] for i in (6, 7)],
            "gripper_command": action[6], "rolling_progress": summary,
        }
        self.client.record_transition(feedback)
        self._progress = progress
        self.latest_feedback = copy.deepcopy(feedback)
        self._expected_observation = after
        self._pending = None
        return copy.deepcopy(feedback)
