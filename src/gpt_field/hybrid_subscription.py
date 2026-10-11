"""One subscription request/history for visual routing and robot control.

Each accepted response selects either the frozen coarse field interface or the
frozen five-action direct interface. Routing is a model judgment from allowable
observations and measured execution feedback, never an oracle or a numerical
threshold on an uncalibrated field. Execution and chunk boundaries live in the
policy; this module never steps the simulator.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from typing import Any, Mapping
from uuid import uuid4

from .astra import (
    APIError, AstraClient, BudgetExceeded, SchemaError, _canonical, output_schema,
    validate_prediction,
)
from .chatgpt_plan import (
    ChatGPTPlanFieldClient, TRANSPORT_VERSION, UNSUPPORTED_FIELDS,
    _access_token, _failure_metadata,
)
from .direct_history import HISTORY_PROTOCOL, _allowed, _trim_images
from .schema import GridSpec


HYBRID_PROMPT_VERSION = "astra-hybrid-field-direct-history-v1"
PROMPT_VERSION = HYBRID_PROMPT_VERSION
HYBRID_KEYS = frozenset(("controller", "target_region", "near_target", "routing_reason", "field", "direct"))


def hybrid_output_schema(grid: GridSpec | None = None) -> dict[str, Any]:
    grid = grid or GridSpec()
    props = {
        "controller": {"type": "string", "enum": ["coarse_field", "contact_direct"]},
        "target_region": {"type": "string", "enum": ["black_bowl", "plate"]},
        "near_target": {"type": "boolean"},
        "routing_reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "field": {"anyOf": [output_schema(grid, "field"), {"type": "null"}]},
        "direct": {"anyOf": [output_schema(grid, "direct"), {"type": "null"}]},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def validate_hybrid_prediction(data: Any, grid: GridSpec | None = None) -> dict[str, Any]:
    grid = grid or GridSpec()
    if grid != GridSpec():
        raise ValueError("Hybrid control requires the original 9x9x9 world grid")
    if not isinstance(data, dict) or set(data) != HYBRID_KEYS:
        raise SchemaError("Hybrid output has missing or unexpected fields")
    if data["controller"] not in ("coarse_field", "contact_direct"):
        raise SchemaError("Hybrid controller must be coarse_field or contact_direct")
    if data["target_region"] not in ("black_bowl", "plate"):
        raise SchemaError("Hybrid target_region must be black_bowl or plate")
    if type(data["near_target"]) is not bool:
        raise SchemaError("Hybrid near_target must be a boolean")
    reason = data["routing_reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise SchemaError("Hybrid routing_reason must be a short nonempty visual explanation")
    contact = data["controller"] == "contact_direct"
    if data["near_target"] != contact:
        raise SchemaError("Hybrid near_target must be true exactly when contact_direct is selected")
    active, inactive = ("direct", "field") if contact else ("field", "direct")
    if data[inactive] is not None:
        raise SchemaError("Hybrid inactive controller branch must be null")
    validate_prediction(data[active], grid, active)
    return data


class ChatGPTPlanHybridClient(ChatGPTPlanFieldClient):
    """Single OAuth transport, model, usage ledger and full accepted transcript."""

    @staticmethod
    def _instructions(grid: GridSpec, mode: str) -> str:
        original = AstraClient._instructions(grid, "field")
        field_contract = "State is " + original.split("State is ", 1)[1]
        return (
            f"Interface {HYBRID_PROMPT_VERSION}. Complete the bowl-on-plate task using only the "
            "two current RGB camera images, robot state, task text, and retained observation/model-output/"
            "execution-feedback history. No object poses, depth, contact sensors, rewards, benchmark "
            "success, demonstrations, or future images are provided. Return one strict JSON wrapper "
            "with controller, target_region, near_target, routing_reason, field, and direct. "
            "Choose exactly one controller for the next FIVE actions (0.25 seconds at 20 Hz), then "
            "reconsider from new images and measured feedback. Routing is reversible at every five-action "
            "boundary; there is no automatic phase machine, forced grasp, or automatic recovery. "
            "Use coarse_field for free-space approach or transport. Use contact_direct near the active "
            "interaction region when fine alignment, grasping, immediate lift clearance, placement, or "
            "release needs direct coordinated actions. target_region names that active interaction region: "
            "black_bowl for bowl approach/grasp/lift clearance or plate for placement approach/alignment/release. "
            "Holding the bowl alone does not mean contact_direct forever: free-space transport toward the "
            "plate should use coarse_field until near the placement interaction. Infer near-target status "
            "visually with observed motion, not from a numeric potential value, a field minimum, a presumed "
            "metric distance, or a parsed stage label. Set near_target=true exactly for contact_direct, "
            "and false exactly for coarse_field. routing_reason is a brief observable visual justification "
            "(at most 500 characters), not a step-by-step reasoning trace. "
            "For coarse_field set field to a valid original field object and direct=null. The local planner "
            "replans after each measured action using that field, executing its first move. It uses 1 cm "
            "translation primitives and horizon three. The orientation adapter caps each rotation command "
            "at 0.15 radians; field gripper hold repeats the last executed nonzero open/close intent "
            "across BOTH controllers, initially open. Direct gripper 0 does not change that remembered "
            "open/close intent. "
            "For contact_direct set field=null and direct to an object with only actions: exactly FIVE "
            "consecutive actions, each SEVEN finite normalized numbers in [-1,1], ordered "
            "[translation dx,dy,dz, world rotation axis-angle dx,dy,dz, gripper]. These actions are executed "
            "as provided with no field planner, action correction, or forced closure. Translation 1 means "
            "a 0.05-meter controller delta target per axis; rotation 1 means a 0.5-radian world-frame delta "
            "target. Gripper +1 closes, -1 opens, and 0 commands zero incremental gripper motion. Zero "
            "translation/rotation requests no change from the measured pose. Each action is relative to "
            "its actual measured pose, not a previously requested target. Commands do not guarantee "
            "realized motion or successful grasp. Both controllers use the same eight-value robot state "
            "and world frame. The following original field contract applies ONLY inside field when "
            "coarse_field is selected; its instruction to match the schema refers to that nested object: "
            + field_contract +
            " In the shared transcript each historical field stays centered at the first three state "
            "values of its own preceding observation, never the current observation. Historical direct "
            "actions remain the previously selected five-action sequence. Feedback gives actual normalized "
            "actions, measured before/after robot states, tracking error, gripper joint changes, and robot "
            "motion statistics. chunk_index/chunk_step identify five-action routing intervals for either "
            "controller. Compare intended and measured movement; use visual evidence to assess object "
            "movement. Gripper closure or robot movement alone does not prove grasp or task success. "
            "Return only the complete hybrid JSON wrapper; the unused controller branch must be null."
        )

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict[str, Any]:
        if mode != "hybrid":
            raise ValueError("ChatGPTPlanHybridClient supports only hybrid mode")
        grid = grid or GridSpec()
        if grid != GridSpec():
            raise ValueError("Hybrid control requires the original 9x9x9 world grid")
        allowed = _allowed(observation)
        with self._history_lock:
            payload = AstraClient.build_payload(self, allowed.to_dict(), grid, "field")
            for name in UNSUPPORTED_FIELDS:
                payload.pop(name, None)
            payload["store"], payload["stream"] = False, True
            payload["input"][0]["content"][0]["text"] = _canonical({
                "type": "observation", "observation_index": self._observation_count,
                "prompt": allowed.prompt, "observation/state": list(allowed.state),
            })
            payload["input"] = _trim_images(self._history + payload["input"], self.image_history)
            payload["include"] = ["reasoning.encrypted_content"]
            payload["instructions"] += (
                f" Retain only the most recent {self.image_history} observation image pairs; all earlier "
                "textual observations, accepted complete hybrid output items (including opaque reasoning), "
                "and execution feedback remain in the shared conversation."
            )
            payload["text"]["format"] = {"type": "json_schema", "name": "astra_hybrid_history_v1",
                "strict": True, "schema": hybrid_output_schema(grid)}
            return payload

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
            "prompt_version": HYBRID_PROMPT_VERSION, "endpoint": self.config.endpoint,
            "history_protocol": HISTORY_PROTOCOL, "image_history": self.image_history}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _write_log(self, path, data: Any) -> None:
        def redact(value):
            if isinstance(value, str):
                for token in self._used_access_tokens:
                    value = value.replace(token, "[REDACTED]")
                return value
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [redact(item) for item in value]
            return value
        if isinstance(data, dict):
            data = {**data, "billing": "chatgpt_subscription", "transport_version": TRANSPORT_VERSION}
            if "payload" in data and "request_ordinal" in data:
                data.update(prompt_version=HYBRID_PROMPT_VERSION, history_protocol=HISTORY_PROTOCOL,
                    image_history=self.image_history, history_stats_before_request=self.history_stats())
                self._history_request_settled = False
        AstraClient._write_log(self, path, redact(data))

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "hybrid") -> dict[str, Any]:
        """One attempt using shared subscription machinery and hybrid validation."""
        grid = grid or GridSpec()
        with self._history_lock, self._lock:
            self.preflight()
            payload = self.build_payload(observation, grid, mode)
            _, timeout = self._reserve()
            token = _access_token(self.token_provider)
            self._used_access_tokens.add(token)
            self.last_request_key = self.request_key(payload, grid, mode)
            self.last_response = None
            ident = f"{self._api_calls + 1:06d}-{uuid4().hex}"
            path = self.log_dir / ident if self.log_dir is not None else None
            request_path = path.with_suffix(".request.json") if path is not None else None
            self._write_log(request_path, {
                "request_key": self.last_request_key, "attempt": 1, "grid": asdict(grid),
                "mode": mode, "payload": payload, "configuration": asdict(self.config),
                "request_ordinal": self._api_calls + 1,
                "budget_elapsed_s_before": max(0.0, self.clock() - self._started), "timeout_s": timeout,
            })
            self.last_log_path = str(request_path) if request_path is not None else None
            self._api_calls += 1
            settled = False
            try:
                response = self.transport(self.config.endpoint, payload, {
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }, timeout)
                if not isinstance(response, Mapping):
                    raise APIError("Subscription transport did not return a completed response")
                self.last_response = dict(response)
                self._write_log(path.with_suffix(".response.json") if path is not None else None, response)
                self._record_usage(response)
                settled = True
                if "model" in response and response["model"] != self.config.model:
                    raise APIError("Subscription response model differs from requested model; output rejected")
                if (self.config.max_wall_time_s is not None
                        and self.clock() - self._started >= self.config.max_wall_time_s):
                    raise BudgetExceeded("max_wall_time_s reached while waiting for subscription response")
                result = validate_hybrid_prediction(self._extract(response), grid)
                self._write_log(path.with_suffix(".result.json") if path is not None else None,
                                {"status": "valid", "budget": self.budget_snapshot()})
                self._history = payload["input"] + copy.deepcopy(response["output"])
                self._observation_count += 1
                return copy.deepcopy(result)
            except BaseException as exc:
                if not settled:
                    self._unknown_usage_calls += 1
                self._history_request_settled = True
                self._write_log(path.with_suffix(".result.json") if path is not None else None,
                    {"status": "error", "error_type": type(exc).__name__,
                     "error": "Subscription request failed; no retry or API-key fallback",
                     **_failure_metadata(exc), "budget": self.budget_snapshot()})
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, (APIError, SchemaError, BudgetExceeded)):
                    message = str(exc)
                    for used in self._used_access_tokens:
                        message = message.replace(used, "[REDACTED]")
                    sanitized = type(exc)(message)
                    for key, value in _failure_metadata(exc).items():
                        setattr(sanitized, key, value)
                    raise sanitized from None
                raise APIError(f"Subscription request failed ({type(exc).__name__}); no retry") from None
