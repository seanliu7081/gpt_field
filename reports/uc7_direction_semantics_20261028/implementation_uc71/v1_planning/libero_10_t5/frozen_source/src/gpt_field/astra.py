"""Auditable, stateless Responses API client for the field experiments.

Only ``predict`` sends a request. Logs never serve as a response cache. Budget
reservations use Astra's full documented input context and long-context premium,
so a small cost cap can reject even a small request. Prices are explicit
experiment configuration; requests explicitly use the standard service tier.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .schema import FieldPrediction, GridSpec

ENDPOINT = "https://api.openai.com/v1/responses"
ASTRA_CONTEXT_TOKENS = 1_050_000
# Verified against https://developers.openai.com/api/docs/models/gpt-6-astra:
# beyond 272K input tokens, premiums apply to the FULL request, not its excess.
ASTRA_LONG_CONTEXT_THRESHOLD = 272_000
ASTRA_LONG_INPUT_MULTIPLIER = 2.0
ASTRA_LONG_OUTPUT_MULTIPLIER = 1.5
ACCOUNTING_VERSION = 2
PROMPT_VERSION = "astra-potential-v1"
OBSERVATION_KEYS = frozenset((
    "observation/image", "observation/wrist_image", "observation/state", "prompt",
))


class APIError(RuntimeError):
    """Transport, refusal, or incomplete-response failure; never retried."""

    def __init__(self, message: str, *, response_body: str | None = None):
        super().__init__(message)
        self.response_body = response_body


class SchemaError(ValueError):
    """Malformed structured model output (at most one new sample allowed)."""


class MissingCredentials(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class AstraConfig:
    model: str = "gpt-6-astra"
    endpoint: str = ENDPOINT
    max_output_tokens: int = 8192
    timeout_s: float = 180.0
    max_api_calls: int | None = None
    max_cost_usd: float | None = None
    max_wall_time_s: float | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cached_input_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None
    max_input_tokens: int = ASTRA_CONTEXT_TOKENS

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be explicitly specified")
        if self.endpoint != ENDPOINT:
            raise ValueError("Only the official OpenAI Responses endpoint is supported")
        for name in ("max_output_tokens", "max_input_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model == "gpt-6-astra" and self.max_output_tokens > 128_000:
            raise ValueError("max_output_tokens exceeds Astra's documented output limit")
        if self.max_api_calls is not None and (
            type(self.max_api_calls) is not int or self.max_api_calls < 0
        ):
            raise ValueError("max_api_calls must be a nonnegative integer")
        for name in ("timeout_s", "max_cost_usd", "max_wall_time_s",
                     "input_price_per_million", "output_price_per_million",
                     "cached_input_price_per_million", "cache_write_price_per_million"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.timeout_s == 0:
            raise ValueError("timeout_s must be positive")


def _strict_loads(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _canonical(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def urllib_transport(endpoint: str, payload: dict[str, Any], headers: dict[str, str],
                     timeout_s: float) -> Mapping[str, Any]:
    """One HTTP attempt; request headers, including the key, are never logged."""
    request = Request(endpoint, data=_canonical(payload).encode("utf-8"),
                      headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise APIError(f"OpenAI HTTP {exc.code}; raw error retained in request log",
                       response_body=body) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise APIError(f"OpenAI transport failure ({type(exc).__name__}); no automatic retry") from exc
    try:
        parsed = _strict_loads(body)
    except (ValueError, TypeError) as exc:
        raise APIError("OpenAI response envelope is not valid JSON", response_body=body) from exc
    if not isinstance(parsed, dict):
        raise APIError("OpenAI response envelope must be an object", response_body=body)
    return parsed


def _array(items: dict[str, Any], length: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": items}
    if length is not None:
        schema.update(minItems=length, maxItems=length)
    return schema


def output_schema(grid: GridSpec, mode: str = "field") -> dict[str, Any]:
    """JSON Schema subset supported by Responses Structured Outputs."""
    if mode == "direct":
        props = {"actions": _array(_array({"type": "number", "minimum": -1,
                                           "maximum": 1}, 7), 5)}
    elif mode in ("field", "waypoint"):
        props = {
            "stage": {"type": "string"},
            "orientation_target_xyzw": _array({"type": "number"}, 4),
            "gripper_command": {"type": "string", "enum": ["open", "close", "hold"]},
            "motion_mode": {"type": "string", "enum": ["move", "hold"]},
            "blocked_indices": _array({"type": "integer", "minimum": 0, "maximum": grid.count - 1}),
            "unknown_indices": _array({"type": "integer", "minimum": 0, "maximum": grid.count - 1}),
        }
        if mode == "field":
            props["values"] = _array({"type": "number"}, grid.count)
        else:
            props["waypoint_xyz"] = _array({"type": "number"}, 3)
    else:
        raise ValueError("mode must be field, waypoint, or direct")
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _numbers(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{label} must contain exactly {length} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError(f"{label} must contain finite numbers, never booleans")
    return [float(x) for x in values]


def validate_prediction(data: Any, grid: GridSpec, mode: str = "field") -> dict[str, Any]:
    """Validate locally as well as requesting server-side strict output."""
    schema = output_schema(grid, mode)
    if not isinstance(data, dict) or set(data) != set(schema["required"]):
        raise SchemaError(f"{mode} output has missing or unexpected fields")
    try:
        for name, specification in schema["properties"].items():
            if specification["type"] == "array" and not isinstance(data[name], (list, tuple)):
                raise ValueError(f"{name} must be an array")
        if mode == "direct":
            actions = data["actions"]
            if not isinstance(actions, (list, tuple)) or len(actions) != 5:
                raise ValueError("direct output must contain exactly five actions")
            for action in actions:
                if any(abs(x) > 1 for x in _numbers(action, 7, "action")):
                    raise ValueError("direct actions must be in [-1, 1]")
        elif mode == "field":
            FieldPrediction.from_dict(data, grid)
        else:
            _numbers(data["waypoint_xyz"], 3, "waypoint_xyz")
            # Reuse identical auxiliary-head and mask validation, without changing
            # or returning the synthetic values used only for this validation.
            auxiliary = {key: value for key, value in data.items() if key != "waypoint_xyz"}
            auxiliary["values"] = [0.0] * grid.count
            FieldPrediction.from_dict(auxiliary, grid)
    except (ValueError, TypeError, KeyError) as exc:
        raise SchemaError(str(exc)) from exc
    return data


class AstraClient:
    """Serial client; all retries count toward the same lifetime budget.

    ``transport(endpoint, payload, headers, timeout_s)`` returns the complete
    Responses JSON envelope. Supply an explicit dummy key for fake transports.
    No network or credential check occurs during construction or build_payload.
    """

    def __init__(self, config: AstraConfig | None = None, *, api_key: str | None = None,
                 transport: Callable[..., Mapping[str, Any]] | None = None,
                 log_dir: str | Path | None = None, clock: Callable[[], float] = time.monotonic):
        self.config = config or AstraConfig()
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.transport = transport or urllib_transport
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._api_calls = 0
        self._input_tokens = self._output_tokens = self._cached_tokens = self._reasoning_tokens = 0
        self._cost_upper = 0.0
        self._unknown_usage_calls = 0
        self._budget_invalid = False
        self.last_request_key: str | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_log_path: str | None = None

    def _input_rate(self) -> float | None:
        c = self.config
        if c.input_price_per_million is None or c.cache_write_price_per_million is None:
            return None
        return max(c.input_price_per_million, c.cache_write_price_per_million,
                   c.cached_input_price_per_million or 0)

    def _reservation(self) -> float | None:
        return self._token_cost_upper(self.config.max_input_tokens, self.config.max_output_tokens)

    def _token_cost_upper(self, input_tokens: int, output_tokens: int) -> float | None:
        rate = self._input_rate()
        if rate is None or self.config.output_price_per_million is None:
            return None
        long_context = self.config.model == "gpt-6-astra" and input_tokens > ASTRA_LONG_CONTEXT_THRESHOLD
        input_multiplier = ASTRA_LONG_INPUT_MULTIPLIER if long_context else 1.0
        output_multiplier = ASTRA_LONG_OUTPUT_MULTIPLIER if long_context else 1.0
        return (input_tokens * rate * input_multiplier
                + output_tokens * self.config.output_price_per_million * output_multiplier) / 1_000_000

    def preflight(self) -> None:
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise MissingCredentials("OPENAI_API_KEY is missing; set it explicitly before live Astra calls")
        if self.config.max_cost_usd is not None:
            if self._reservation() is None:
                raise ValueError("max_cost_usd requires explicit input_price_per_million, "
                                 "output_price_per_million, and cache_write_price_per_million")
            if self.config.model != "gpt-6-astra":
                raise ValueError("A hard cost cap currently requires gpt-6-astra's verified context bound")
            if self.config.max_input_tokens < ASTRA_CONTEXT_TOKENS:
                raise ValueError("Hard cost reservation requires max_input_tokens >= 1050000; "
                                 "this client does not assume an unverified image token estimate")

    def budget_snapshot(self) -> dict[str, Any]:
        return {
            "snapshot_version": ACCOUNTING_VERSION,
            "configuration": asdict(self.config),
            "api_calls": self._api_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_input_tokens": self._cached_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "cost_usd_upper_bound": self._cost_upper if self._reservation() is not None else None,
            "reservation_per_call_usd": self._reservation(),
            "unknown_usage_calls": self._unknown_usage_calls,
            "elapsed_s": max(0.0, self.clock() - self._started),
            "budget_bound_violated": self._budget_invalid,
            "max_api_calls": self.config.max_api_calls,
            "max_cost_usd": self.config.max_cost_usd,
            "max_wall_time_s": self.config.max_wall_time_s,
            "cost_accounting": "standard service tier; maximum input-side configured rate; "
                               "Astra inputs >272000 apply 2x input/cache and 1.5x output to the full request; "
                               "unresolved calls retain full long-context reservation",
            "wall_time_accounting": "checked before/after requests; socket timeout bounded by remaining time",
        }

    def load_budget_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore one run's ledger before any requests from this client.

        Configuration, including prices and caps, must match exactly. Follow this
        with ``reconcile_request_logs`` to account for a crash before persistence.
        Downtime between processes is excluded from accumulated run wall time.
        """
        with self._lock:
            if self._api_calls:
                raise ValueError("Budget restoration requires a client with no attempted calls")
            if snapshot.get("snapshot_version") != ACCOUNTING_VERSION or snapshot.get("configuration") != asdict(self.config):
                raise ValueError("Budget snapshot version or Astra configuration does not match")
            counters = {
                "api_calls": "_api_calls", "input_tokens": "_input_tokens",
                "output_tokens": "_output_tokens", "cached_input_tokens": "_cached_tokens",
                "reasoning_tokens": "_reasoning_tokens", "unknown_usage_calls": "_unknown_usage_calls",
            }
            for key in counters:
                value = snapshot.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid budget snapshot counter: {key}")
            elapsed = snapshot.get("elapsed_s")
            cost = snapshot.get("cost_usd_upper_bound")
            for name, value in (("elapsed_s", elapsed), ("cost_usd_upper_bound", cost)):
                if name == "cost_usd_upper_bound" and value is None and self._reservation() is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid budget snapshot value: {name}")
            if type(snapshot.get("budget_bound_violated")) is not bool:
                raise ValueError("Invalid budget snapshot violation flag")
            if snapshot["unknown_usage_calls"] > snapshot["api_calls"]:
                raise ValueError("Unknown-usage calls exceed total attempted calls")
            for key, attr in counters.items():
                setattr(self, attr, snapshot[key])
            self._cost_upper = float(cost) if cost is not None else 0.0
            self._budget_invalid = snapshot["budget_bound_violated"]
            self._started = self.clock() - float(elapsed)

    def reconcile_request_logs(self) -> int:
        """Conservatively charge attempts newer than the restored snapshot.

        A request logged just before a crash might have reached the server even
        without a response file. Retain its full token reservation; do not infer
        that a missing response means the request was free. This intentionally
        overcounts requests logged but never dispatched. Logs are never replayed.
        Returns the number of newly charged attempts.
        """
        with self._lock:
            if self.log_dir is None:
                raise ValueError("Crash-safe reconciliation requires this run's API log_dir")
            records = []
            for path in self.log_dir.glob("*.request.json"):
                try:
                    record = _strict_loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    raise ValueError(f"Cannot reconcile request log {path.name}") from exc
                if (not isinstance(record, dict) or record.get("configuration") != asdict(self.config)
                        or record.get("accounting_version") != ACCOUNTING_VERSION):
                    raise ValueError(f"Request log configuration mismatch: {path.name}")
                ordinal = record.get("request_ordinal")
                if type(ordinal) is not int or ordinal < 1:
                    raise ValueError(f"Invalid request ordinal in {path.name}")
                for key in ("budget_elapsed_s_before", "timeout_s"):
                    value = record.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid timing in {path.name}")
                records.append(record)
            records.sort(key=lambda record: record["request_ordinal"])
            if [record["request_ordinal"] for record in records] != list(range(1, len(records) + 1)):
                raise ValueError("Request log sequence has missing or duplicate attempts")
            if len(records) < self._api_calls:
                raise ValueError("Request logs are incomplete relative to the saved budget snapshot")
            extra = records[self._api_calls:]
            if extra:
                reservation = self._reservation()
                self._api_calls += len(extra)
                self._unknown_usage_calls += len(extra)
                if reservation is not None:
                    self._cost_upper += len(extra) * reservation
                elapsed = max(self.clock() - self._started,
                              max(record["budget_elapsed_s_before"] + record["timeout_s"] for record in extra))
                self._started = self.clock() - elapsed
            return len(extra)

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec,
                      mode: str = "field") -> dict[str, Any]:
        if set(observation) != OBSERVATION_KEYS:
            raise ValueError("Policy observation must have exactly the four allowed keys")
        state = _numbers(observation["observation/state"], 8, "observation/state")
        if not isinstance(observation["prompt"], str) or not observation["prompt"].strip():
            raise ValueError("prompt must be a nonempty string")
        for name in ("observation/image", "observation/wrist_image"):
            value = observation[name]
            if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
                raise ValueError(f"{name} must be an inline PNG data URL")
            try:
                decoded = base64.b64decode(value.split(",", 1)[1], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} has invalid base64") from exc
            if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"{name} is not PNG data")
        instructions = self._instructions(grid, mode)
        user_text = _canonical({"prompt": observation["prompt"], "observation/state": state})
        return {
            "model": self.config.model,
            # Avoid Project-level auto routing to a premium service tier.
            "service_tier": "default",
            "store": False,
            "max_output_tokens": self.config.max_output_tokens,
            "instructions": instructions,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_text", "text": "External camera:"},
                {"type": "input_image", "image_url": observation["observation/image"], "detail": "high"},
                {"type": "input_text", "text": "Wrist camera:"},
                {"type": "input_image", "image_url": observation["observation/wrist_image"], "detail": "high"},
            ]}],
            "text": {"format": {"type": "json_schema", "name": f"astra_{mode}_v1",
                                  "strict": True, "schema": output_schema(grid, mode)}},
        }

    @staticmethod
    def _instructions(grid: GridSpec, mode: str) -> str:
        base = (
            f"Interface {PROMPT_VERSION}. Infer the next local manipulation decision from only "
            "the current two RGB images, robot state, and task text. No previous conversation, "
            "demonstrations, object state, depth, contact signals, or future images are provided. "
            "State is [end-effector position xyz in meters, orientation axis-angle xyz in radians, "
            "two gripper joint positions]; these are NOT seven robot joint angles. "
            f"All position and orientation coordinates use the {grid.frame} frame. "
        )
        if mode == "direct":
            return base + (
                "Return exactly five consecutive LIBERO OSC control actions, each containing seven "
                "normalized values in [-1,1]: translation dx,dy,dz, rotation axis-angle dx,dy,dz, "
                "and gripper. Positive gripper closes and negative opens. Translation action 1 "
                "corresponds to a 0.05-meter controller delta target; rotation action 1 corresponds "
                "to a 0.5-radian controller delta target. These are controller targets, not guaranteed "
                "realized motion. Express rotational increments in the configured world frame. "
                "Zero translation/rotation holds that target; zero gripper stops commanded joint "
                "increment. Return JSON matching the schema."
            )
        base += (
            f"The grid has {grid.size} points on each axis with spacing {grid.spacing_m} meters. "
            "Its center is the first three state values at this observation; it remains fixed. "
            f"At indices (i,j,k) in [0,{grid.size - 1}], point xyz is center + "
            f"{grid.spacing_m} * (i-{(grid.size - 1) / 2},j-{(grid.size - 1) / 2},k-{(grid.size - 1) / 2}). "
            f"Flat array index is i*{grid.size}*{grid.size}+j*{grid.size}+k. "
            "Infer the current local stage and a unit xyzw target quaternion. Choose gripper open, "
            "close, or hold. Choose motion_mode=hold when translation should wait (for example "
            "during a gripper change); otherwise move. Interpret hypothetical positions with the "
            "scene and inferred stage held fixed, and the selected orientation and gripper intent. "
            "Mark predicted blocked and unknown points with separate, disjoint index lists; do not "
            "claim whole-arm collision certification from the images. Unknown means insufficient "
            "visual evidence, not free space. The stage target can lie beyond this local grid. "
        )
        if mode == "waypoint":
            return base + "Return one waypoint_xyz in meters in the same frame, with the auxiliary fields and masks."
        return base + (
            f"Return exactly {grid.count} finite unitless potential values, one per grid point. "
            "Lower values indicate more desirable progress toward the inferred stage target. "
            "This is a local spatial preference potential, NOT metric distance or calibrated "
            "cost-to-go; there is no Eikonal, minimum-zero, nonnegativity, or meter claim. "
            "Use a consistent smooth local scale and express obstacle-aware route preferences. "
            "Values at blocked/unknown indices are ignored by the planner but must be finite. "
            "Return only JSON matching the schema."
        )

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
                    "prompt_version": PROMPT_VERSION, "endpoint": self.config.endpoint}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _write_log(self, path: Path | None, data: Any) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")

    def _reserve(self) -> tuple[float | None, float]:
        c = self.config
        if self._budget_invalid:
            raise BudgetExceeded("Previous API usage exceeded the configured reservation envelope")
        if c.max_api_calls is not None and self._api_calls >= c.max_api_calls:
            raise BudgetExceeded("max_api_calls reached (schema retries also count)")
        remaining = c.max_wall_time_s - (self.clock() - self._started) if c.max_wall_time_s is not None else None
        if remaining is not None and remaining <= 0:
            raise BudgetExceeded("max_wall_time_s reached")
        reservation = self._reservation()
        if c.max_cost_usd is not None and self._cost_upper + reservation > c.max_cost_usd + 1e-12:
            raise BudgetExceeded("Insufficient max_cost_usd for conservative full-context request "
                                 "reservation including long-context pricing premiums")
        return reservation, min(c.timeout_s, remaining) if remaining is not None else c.timeout_s

    def _record_usage(self, response: Mapping[str, Any], reservation: float | None) -> None:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self._unknown_usage_calls += 1
            return
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        if type(inp) is not int or type(out) is not int or inp < 0 or out < 0:
            self._unknown_usage_calls += 1
            return
        self._input_tokens += inp
        self._output_tokens += out  # output_tokens already includes reasoning tokens.
        for section, field, attr in (("input_tokens_details", "cached_tokens", "_cached_tokens"),
                                     ("output_tokens_details", "reasoning_tokens", "_reasoning_tokens")):
            details = usage.get(section)
            value = details.get(field) if isinstance(details, dict) else None
            if type(value) is int and value >= 0:
                setattr(self, attr, getattr(self, attr) + value)
        if reservation is not None:
            observed_upper = self._token_cost_upper(inp, out)
            self._cost_upper += observed_upper - reservation
        if inp > self.config.max_input_tokens or out > self.config.max_output_tokens:
            self._budget_invalid = True
            raise BudgetExceeded("API reported usage exceeding the conservative token reservation envelope")

    @staticmethod
    def _extract(response: Mapping[str, Any]) -> Any:
        if response.get("error"):
            raise APIError("Responses API returned an error; inspect raw response log")
        if response.get("status") != "completed":
            raise APIError(f"Responses API status={response.get('status')!r}; incomplete output is never executed")
        texts: list[str] = []
        output = response.get("output")
        if not isinstance(output, list):
            raise SchemaError("Response contains no output list")
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                raise SchemaError("Response message content is malformed")
            for part in content:
                if not isinstance(part, dict):
                    raise SchemaError("Response content part is malformed")
                if part.get("type") == "refusal":
                    raise APIError("Astra refused this request; no automatic retry or fallback")
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise SchemaError("Structured output text is not a string")
                    texts.append(part["text"])
        if len(texts) != 1:
            raise SchemaError("Expected exactly one structured output text item")
        try:
            return _strict_loads(texts[0])
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"Invalid structured JSON: {exc}") from exc

    def predict(self, observation: Mapping[str, Any], grid: GridSpec,
                mode: str = "field") -> dict[str, Any]:
        with self._lock:
            self.preflight()
            payload = self.build_payload(observation, grid, mode)
            self.last_request_key = self.request_key(payload, grid, mode)
            for attempt in range(2):
                reservation, timeout = self._reserve()
                ident = f"{self._api_calls + 1:06d}-{uuid4().hex}"
                path = self.log_dir / ident if self.log_dir is not None else None
                request_path = path.with_suffix(".request.json") if path is not None else None
                self._write_log(request_path, {"request_key": self.last_request_key,
                                "attempt": attempt + 1, "prompt_version": PROMPT_VERSION,
                                "grid": asdict(grid), "mode": mode, "payload": payload,
                                "reservation_usd": reservation, "configuration": asdict(self.config),
                                "accounting_version": ACCOUNTING_VERSION,
                                "request_ordinal": self._api_calls + 1,
                                "budget_elapsed_s_before": max(0.0, self.clock() - self._started),
                                "timeout_s": timeout})
                self.last_log_path = str(request_path) if request_path is not None else None
                self._api_calls += 1
                if reservation is not None:
                    self._cost_upper += reservation
                settled = False
                try:
                    response = self.transport(self.config.endpoint, payload,
                                              {"Authorization": f"Bearer {self._api_key}",
                                               "Content-Type": "application/json"}, timeout)
                    if not isinstance(response, Mapping):
                        raise APIError("Transport did not return a Responses object")
                    self.last_response = dict(response)
                    self._write_log(path.with_suffix(".response.json") if path is not None else None, response)
                    settled = True
                    self._record_usage(response, reservation)
                    if (self.config.max_wall_time_s is not None
                            and self.clock() - self._started >= self.config.max_wall_time_s):
                        raise BudgetExceeded("max_wall_time_s reached while waiting for the response")
                    result = validate_prediction(self._extract(response), grid, mode)
                    self._write_log(path.with_suffix(".result.json") if path is not None else None,
                                    {"status": "valid", "budget": self.budget_snapshot()})
                    return result
                except Exception as exc:
                    if not settled:
                        self._unknown_usage_calls += 1
                    self._write_log(path.with_suffix(".result.json") if path is not None else None,
                                    {"status": "error", "error_type": type(exc).__name__,
                                     "error": str(exc), "raw_error_body": getattr(exc, "response_body", None),
                                     "budget": self.budget_snapshot()})
                    if isinstance(exc, SchemaError) and attempt == 0:
                        continue
                    if isinstance(exc, (SchemaError, APIError, BudgetExceeded)):
                        raise
                    raise APIError(f"Astra request failed ({type(exc).__name__}); no automatic retry") from exc
            raise AssertionError("unreachable")
