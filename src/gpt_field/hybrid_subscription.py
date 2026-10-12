"""One subscription request/history for visual routing and robot control.

Each accepted response selects either a coarse field or a direct-action chunk.
Routing is a model judgment from allowable
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
    APIError, AstraClient, BudgetExceeded, SchemaError, _canonical, _numbers, output_schema,
    validate_prediction,
)
from .chatgpt_plan import (
    ChatGPTPlanFieldClient, TRANSPORT_VERSION, UNSUPPORTED_FIELDS,
    _access_token, _failure_metadata,
)
from .direct_history import HISTORY_PROTOCOL, TASK, _allowed, _trim_images
from .schema import GridSpec


HYBRID_GRID = GridSpec(size=33, spacing_m=0.005)
HYBRID_CHUNK_SIZE = 5
HYBRID_FIELD_UPDATE_STEPS = 30
HYBRID_PLANNER_HORIZON = 10
HYBRID_FIELD_EXECUTION_STEPS = 10
HYBRID_REASONING_EFFORT = "high"
LEGACY_HYBRID_PROMPT_VERSION = "astra-hybrid-field-direct-history-v1"
HYBRID_SYMMETRIC_PROMPT_VERSION = "astra-hybrid-field-direct-history-v2"
HYBRID_ASYMMETRIC_PROMPT_VERSION = "astra-hybrid-field-direct-history-v3"
HYBRID_RECEDING_PROMPT_VERSION = "astra-hybrid-field-direct-history-v4"
HYBRID_PROMPT_VERSION = "astra-hybrid-field-direct-history-v5"
PROMPT_VERSION = HYBRID_PROMPT_VERSION
HYBRID_KEYS = frozenset(("controller", "target_region", "near_target", "routing_reason", "field", "direct"))
HYBRID_TARGET_REGIONS = ("black_bowl", "plate")


def _target_regions(value) -> tuple[str, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("target_regions must contain unique nonempty strings")
    return tuple(value)


def _validate_configuration(grid: GridSpec, chunk_size: int) -> None:
    if grid not in (HYBRID_GRID, GridSpec()):
        raise ValueError("Hybrid control requires the 33x33x33 grid or explicit legacy 9x9x9 grid")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def _field_interval(grid: GridSpec, chunk_size: int, field_update_steps: int | None) -> int:
    _validate_configuration(grid, chunk_size)
    if field_update_steps is None:
        return chunk_size if grid == GridSpec() else HYBRID_FIELD_UPDATE_STEPS
    if type(field_update_steps) is not int or field_update_steps < 1:
        raise ValueError("field_update_steps must be a positive integer")
    return field_update_steps


def _planner_horizon(grid: GridSpec, chunk_size: int, field_steps: int,
                     planner_horizon: int | None) -> int:
    if planner_horizon is None:
        return 3 if grid == GridSpec() or chunk_size == field_steps else HYBRID_PLANNER_HORIZON
    if type(planner_horizon) is not int or not 1 <= planner_horizon <= 10:
        raise ValueError("planner_horizon must be an integer between 1 and 10")
    return planner_horizon


def _field_execution_interval(grid: GridSpec, chunk_size: int, field_steps: int,
                              horizon: int, field_execution_steps: int | None) -> int:
    if field_execution_steps is None:
        return horizon if grid == HYBRID_GRID and chunk_size != field_steps and horizon != 3 else 1
    if type(field_execution_steps) is not int or not 1 <= field_execution_steps <= horizon:
        raise ValueError("field_execution_steps must be an integer between 1 and planner_horizon")
    return field_execution_steps


def hybrid_prompt_version(grid: GridSpec, chunk_size: int,
                          field_update_steps: int | None = None,
                          planner_horizon: int | None = None,
                          field_execution_steps: int | None = None) -> str:
    field_steps = _field_interval(grid, chunk_size, field_update_steps)
    horizon = _planner_horizon(grid, chunk_size, field_steps, planner_horizon)
    execution_steps = _field_execution_interval(grid, chunk_size, field_steps, horizon, field_execution_steps)
    if execution_steps != 1:
        return HYBRID_PROMPT_VERSION
    if horizon != 3:
        return HYBRID_RECEDING_PROMPT_VERSION
    if grid == GridSpec() and chunk_size == field_steps == 5:
        return LEGACY_HYBRID_PROMPT_VERSION
    if chunk_size == field_steps:
        return HYBRID_SYMMETRIC_PROMPT_VERSION
    return HYBRID_ASYMMETRIC_PROMPT_VERSION


def hybrid_output_schema(grid: GridSpec | None = None,
                         chunk_size: int = HYBRID_CHUNK_SIZE, *,
                         target_regions=HYBRID_TARGET_REGIONS) -> dict[str, Any]:
    grid = grid or HYBRID_GRID
    _validate_configuration(grid, chunk_size)
    regions = _target_regions(target_regions)
    direct_schema = output_schema(grid, "direct")
    direct_schema["properties"]["actions"].update(minItems=chunk_size, maxItems=chunk_size)
    props = {
        "controller": {"type": "string", "enum": ["coarse_field", "contact_direct"]},
        "target_region": {"type": "string", "enum": list(regions)},
        "near_target": {"type": "boolean"},
        "routing_reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "field": {"anyOf": [output_schema(grid, "field"), {"type": "null"}]},
        "direct": {"anyOf": [direct_schema, {"type": "null"}]},
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def validate_hybrid_prediction(data: Any, grid: GridSpec | None = None,
                               chunk_size: int = HYBRID_CHUNK_SIZE, *,
                               target_regions=HYBRID_TARGET_REGIONS) -> dict[str, Any]:
    grid = grid or HYBRID_GRID
    _validate_configuration(grid, chunk_size)
    regions = _target_regions(target_regions)
    if not isinstance(data, dict) or set(data) != HYBRID_KEYS:
        raise SchemaError("Hybrid output has missing or unexpected fields")
    if data["controller"] not in ("coarse_field", "contact_direct"):
        raise SchemaError("Hybrid controller must be coarse_field or contact_direct")
    if data["target_region"] not in regions:
        raise SchemaError("Hybrid target_region must be " + " or ".join(regions))
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
    if contact:
        direct = data["direct"]
        if not isinstance(direct, dict) or set(direct) != {"actions"}:
            raise SchemaError("direct output has missing or unexpected fields")
        try:
            actions = direct["actions"]
            if not isinstance(actions, (list, tuple)) or len(actions) != chunk_size:
                raise ValueError(f"direct output must contain exactly {chunk_size} actions")
            for action in actions:
                if any(abs(value) > 1 for value in _numbers(action, 7, "action")):
                    raise ValueError("direct actions must be in [-1, 1]")
        except (ValueError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc
    else:
        validate_prediction(data[active], grid, active)
    return data


class ChatGPTPlanHybridClient(ChatGPTPlanFieldClient):
    """Single OAuth transport, model, usage ledger and full accepted transcript."""

    def __init__(self, config=None, *, grid: GridSpec | None = None,
                 chunk_size: int = HYBRID_CHUNK_SIZE,
                 field_update_steps: int | None = None,
                 planner_horizon: int | None = None,
                 field_execution_steps: int | None = None,
                 reasoning_effort: str | None = HYBRID_REASONING_EFFORT, **kwargs):
        self.grid = grid or HYBRID_GRID
        _validate_configuration(self.grid, chunk_size)
        self.chunk_size = chunk_size
        self.field_update_steps = _field_interval(self.grid, chunk_size, field_update_steps)
        self.planner_horizon = _planner_horizon(
            self.grid, self.chunk_size, self.field_update_steps, planner_horizon)
        self.field_execution_steps = _field_execution_interval(
            self.grid, self.chunk_size, self.field_update_steps, self.planner_horizon, field_execution_steps)
        if reasoning_effort is not None and reasoning_effort not in (
                "minimal", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("reasoning_effort must be a supported effort string or None")
        self.reasoning_effort = reasoning_effort
        super().__init__(config, **kwargs)

    @property
    def prompt_version(self) -> str:
        return hybrid_prompt_version(
            self.grid, self.chunk_size, self.field_update_steps, self.planner_horizon, self.field_execution_steps)

    @staticmethod
    def _legacy_instructions(grid: GridSpec, mode: str) -> str:
        original = AstraClient._instructions(grid, "field")
        field_contract = "State is " + original.split("State is ", 1)[1]
        return (
            f"Interface {LEGACY_HYBRID_PROMPT_VERSION}. Complete the bowl-on-plate task using only the "
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

    @staticmethod
    def _instructions(grid: GridSpec, mode: str,
                      chunk_size: int = HYBRID_CHUNK_SIZE,
                      field_update_steps: int | None = None,
                      planner_horizon: int | None = None,
                      field_execution_steps: int | None = None) -> str:
        field_steps = _field_interval(grid, chunk_size, field_update_steps)
        horizon = _planner_horizon(grid, chunk_size, field_steps, planner_horizon)
        execution_steps = _field_execution_interval(grid, chunk_size, field_steps, horizon, field_execution_steps)
        version = hybrid_prompt_version(grid, chunk_size, field_steps, horizon, execution_steps)
        instructions = ChatGPTPlanHybridClient._legacy_instructions(grid, mode)
        if version == LEGACY_HYBRID_PROMPT_VERSION:
            return instructions
        instructions = (instructions
            .replace(LEGACY_HYBRID_PROMPT_VERSION, version)
            .replace("FIVE", str(chunk_size))
            .replace("0.25 seconds", f"{chunk_size / 20:g} seconds")
            .replace("five-action", f"{chunk_size}-action")
            .replace("valid original field object", "valid field object")
            .replace("following original field contract", "following field contract"))
        if field_steps != chunk_size:
            instructions = (instructions
                .replace(
                    f"Choose exactly one controller for the next {chunk_size} actions "
                    f"({chunk_size / 20:g} seconds at 20 Hz), then "
                    "reconsider from new images and measured feedback. Routing is reversible at every "
                    f"{chunk_size}-action boundary;",
                    "Choose exactly one controller for the next routing interval. "
                    f"A coarse_field interval lasts {field_steps} executed actions "
                    f"({field_steps / 20:g} seconds at 20 Hz); a contact_direct interval lasts "
                    f"{chunk_size} executed actions ({chunk_size / 20:g} seconds at 20 Hz). "
                    "Reconsider from new images and measured feedback after that interval. "
                    "Routing is reversible at each controller-specific interval boundary;")
                .replace(
                    "For coarse_field set field to a valid field object and direct=null. The local planner ",
                    "For coarse_field set field to a valid field object and direct=null. "
                    f"Reuse this fixed-center field for {field_steps} executed actions, then request a "
                    "new field or choose direct control. The local planner ")
                .replace(
                    f"chunk_index/chunk_step identify {chunk_size}-action routing intervals for either controller.",
                    "chunk_index identifies consecutive routing intervals in chronological order across "
                    "BOTH controllers; chunk_step counts from zero within the selected controller's interval "
                    f"({field_steps} actions for coarse_field and {chunk_size} for contact_direct). "
                    "Interval lengths vary when routing changes."))
        if horizon != 3:
            horizon_contract = (
                f"translation primitives and horizon {horizon}: each candidate contains {horizon} "
                "one-centimeter moves, and only its first move is executed before replanning from "
                "the measured pose. "
            )
            if horizon > 6:
                horizon_contract += (
                    "Exact lattice dynamic programming merges equivalent partial paths while "
                    "preserving the terminal-potential objective and tie-breaking rules. "
                )
            horizon_contract += (
                f"All candidate positions and segments must remain inside the fixed +/-{grid.half_width_m:g}-meter "
                "cube around its generation center; lookahead does not expand field coverage. "
                f"Each candidate has {horizon} cm total path length. "
            )
            if horizon * .01 > grid.half_width_m:
                horizon_contract += (
                    f"From the field center it cannot travel straight {horizon} cm and must turn "
                    "to remain inside the cube. "
                )
            instructions = instructions.replace(
                "translation primitives and horizon three. ", horizon_contract)
        if execution_steps != 1:
            execution_contract = (
                f"plans a complete {horizon}-step translation sequence from the actual measured pose, "
                f"then executes the next {execution_steps} successive fixed world-frame translation "
                "primitives from that selected sequence in order. "
                "Each queued translation is the planned primitive delta, not a correction toward a "
                "previously planned waypoint; it does not repeat the first primitive for every action. "
                "Remaining translations are not regenerated or corrected before the execution window ends. "
                f"It replans from the new actual measured pose after {execution_steps} executed actions "
                "using the same fixed-center field. "
            )
            if field_steps == 30 and execution_steps == 10:
                execution_contract += (
                    "There are three local plans per complete 30-action field interval, at "
                    "interval-relative steps 0, 10, and 20. "
                )
            execution_contract += (
                "A field refresh or controller switch discards any unexecuted translation suffix. "
                "If no complete valid plan exists, translation stays zero for that execution window "
                "until its next scheduled planning or field-refresh boundary. "
                "The orientation adapter still uses the latest measured robot state on every action, "
                "applies the field's gripper intent, and records execution feedback after every action. "
            )
            instructions = (instructions
                .replace(
                    "replans after each measured action using that field, executing its first move. ",
                    execution_contract)
                .replace(
                    "one-centimeter moves, and only its first move is executed before replanning from "
                    "the measured pose. ",
                    f"one-centimeter moves; the selected plan supplies the next {execution_steps} "
                    "translation primitives for execution. "))
        return instructions

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "hybrid") -> dict[str, Any]:
        if mode != "hybrid":
            raise ValueError("ChatGPTPlanHybridClient supports only hybrid mode")
        grid = grid or self.grid
        _validate_configuration(grid, self.chunk_size)
        if grid != self.grid:
            raise ValueError("Requested grid must match the configured hybrid grid")
        allowed = _allowed(observation, getattr(self, "task_prompt", TASK))
        with self._history_lock:
            payload = AstraClient.build_payload(self, allowed.to_dict(), grid, "field")
            payload["instructions"] = self._instructions(
                grid, mode, self.chunk_size, self.field_update_steps, self.planner_horizon, self.field_execution_steps)
            for name in UNSUPPORTED_FIELDS:
                payload.pop(name, None)
            payload["store"], payload["stream"] = False, True
            if self.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.reasoning_effort}
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
            version = self.prompt_version.rsplit("-", 1)[-1]
            payload["text"]["format"] = {"type": "json_schema", "name": f"astra_hybrid_history_{version}",
                "strict": True, "schema": hybrid_output_schema(grid, self.chunk_size,
                    target_regions=getattr(self, "target_regions", HYBRID_TARGET_REGIONS))}
            return payload

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
            "prompt_version": hybrid_prompt_version(
                grid, self.chunk_size, self.field_update_steps, self.planner_horizon, self.field_execution_steps), "endpoint": self.config.endpoint,
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
                data.update(prompt_version=self.prompt_version, history_protocol=getattr(self, "history_protocol", HISTORY_PROTOCOL),
                    image_history=self.image_history, history_stats_before_request=self.history_stats())
                self._history_request_settled = False
        AstraClient._write_log(self, path, redact(data))

    def validate_response(self, data: Any, grid: GridSpec) -> dict[str, Any]:
        """Validate a raw model response; subclasses may normalize its representation."""
        return validate_hybrid_prediction(data, grid, self.chunk_size,
            target_regions=getattr(self, "target_regions", HYBRID_TARGET_REGIONS))

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "hybrid") -> dict[str, Any]:
        """One attempt using shared subscription machinery and hybrid validation."""
        grid = grid or self.grid
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
                reported_reasoning = response.get("reasoning")
                reported_effort = (reported_reasoning.get("effort")
                                   if isinstance(reported_reasoning, Mapping) else None)
                if (self.reasoning_effort is not None and reported_effort is not None
                        and reported_effort != self.reasoning_effort):
                    error = APIError("Subscription response reasoning effort differs from requested effort; output rejected")
                    error.error_code = "response_reasoning_effort_mismatch"
                    raise error
                if (self.config.max_wall_time_s is not None
                        and self.clock() - self._started >= self.config.max_wall_time_s):
                    raise BudgetExceeded("max_wall_time_s reached while waiting for subscription response")
                result = self.validate_response(self._extract(response), grid)
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
