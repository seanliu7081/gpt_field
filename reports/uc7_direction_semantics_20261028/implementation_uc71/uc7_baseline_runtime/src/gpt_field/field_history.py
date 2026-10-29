"""The original field planner with local conversation and execution feedback.

Only the model's information changes. Field interpretation, refresh cadence,
local planning, and action adaptation remain in the frozen ``FieldPolicy``.
Transcript transport and observable feedback reuse the audited direct-history
implementation, including opaque Responses output replay and cost accounting.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .astra import AstraClient, _canonical
from .direct_history import (
    CHUNK_SIZE, HISTORY_PROTOCOL, TASK, DirectHistoryAstraClient,
    DirectHistoryPolicy, _allowed, _trim_images,
)
from .observation import PolicyObservation
from .policy import Decision, FieldPolicy
from .schema import GridSpec


PROMPT_VERSION = "astra-field-history-v1"


class FieldHistoryAstraClient(DirectHistoryAstraClient):
    """Replay all accepted field outputs and feedback, retaining two image pairs.

Failed requests never commit transcript entries. The inherited implementation
preserves complete ordered Responses output items, IDs, assistant phase, and
opaque encrypted reasoning without inspecting their contents.
"""

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "field") -> dict:
        if mode != "field":
            raise ValueError("FieldHistoryAstraClient supports only field mode")
        grid = grid or GridSpec()
        if grid != GridSpec():
            raise ValueError("The field-history experiment requires the original 9x9x9 world grid")
        allowed = _allowed(observation)
        with self._history_lock:
            # Skip the direct-only payload override while retaining the base
            # client's image validation and the exact original field schema.
            payload = AstraClient.build_payload(self, allowed.to_dict(), grid, mode)
            payload["input"][0]["content"][0]["text"] = _canonical({
                "type": "observation", "observation_index": self._observation_count,
                "prompt": allowed.prompt, "observation/state": list(allowed.state),
            })
            payload["input"] = _trim_images(self._history + payload["input"], self.image_history)
            payload["include"] = ["reasoning.encrypted_content"]
            payload["instructions"] += (
                f" Only the most recent {self.image_history} observation image pairs are included; "
                "earlier textual observations, complete valid field outputs, and execution feedback "
                "remain in this conversation."
            )
            payload["text"]["format"]["name"] = "astra_field_history_v1"
            return payload

    @staticmethod
    def _instructions(grid: GridSpec, mode: str) -> str:
        # Reuse the original field description verbatim, replacing only its
        # stateless introduction and adding the history/feedback interface.
        original = AstraClient._instructions(grid, "field")
        field_description = "State is " + original.split("State is ", 1)[1]
        return (
            f"Interface {PROMPT_VERSION}. Infer the next local manipulation decision from the "
            "two RGB images, robot state, task text, and prior observation/field/action/execution-"
            "feedback history below. Prior valid model responses are retained in this conversation. "
            "No demonstrations, object state, depth, contact signals, rewards, benchmark success, "
            "or future images are provided. "
            + field_description +
            " A new field is requested every five executed actions. The same local planner replans "
            "from the measured end-effector position after each action and executes its first move. "
            "Every historical field is centered at the first three state values of its own preceding "
            "observation, not the current observation; its origin remains fixed until replaced. "
            "Execution-feedback records contain actual normalized actions and measured before/after "
            "robot states. Each normalized_action has seven values: translation dx,dy,dz, world "
            "rotation axis-angle dx,dy,dz, and gripper. Translation 1 means a 0.05-meter controller "
            "delta target; rotation 1 means a 0.5-radian world-frame delta target. Gripper +1 closes "
            "and -1 opens. Your field gripper intent hold retains the previous open/close command, "
            "initially open. Actions execute at 20 Hz; five actions span 0.25 seconds of simulation. "
            "The feedback chunk_index and chunk_step identify five-action field-refresh "
            "intervals, not a model-generated direct-action chunk. Desired controller targets are "
            "not guaranteed realized motion. Compare commands with measured movement and reconsider "
            "repeated ineffective actions, alignment, and approach direction when generating the next "
            "field, orientation, and gripper intent. Manage grasping and task progression yourself "
            "from visual and robot-state evidence; closing the gripper does not prove a grasp. "
            "Progress statistics describe robot motion only, not object motion or task success. "
            "Feedback does not automatically change phase, alter the planner, recover, or close the "
            "gripper for you."
        )

    def request_key(self, payload: Mapping[str, Any], grid: GridSpec, mode: str) -> str:
        material = {"payload": payload, "grid": asdict(grid), "mode": mode,
                    "prompt_version": PROMPT_VERSION, "endpoint": self.config.endpoint,
                    "history_protocol": HISTORY_PROTOCOL, "image_history": self.image_history}
        return hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()

    def _write_log(self, path: Path | None, data: Any) -> None:
        # The inherited override stamps the direct baseline's module constant;
        # keep the same bookkeeping/redaction with this variant's provenance.
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

        AstraClient._write_log(self, path, redact(data))

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "field") -> dict:
        return super().predict(observation, grid, mode)


class FieldHistoryPolicy(FieldPolicy):
    """Unchanged field control with mandatory feedback between actual actions.

    Defaults are intentionally fixed to the original field experiment: 9 cubed
    grid, 2 cm spacing, five-action refresh, 1 cm primitives, horizon three, and
    the original quaternion/gripper adapter. Stage labels remain diagnostic.
    """

    def __init__(self, client: FieldHistoryAstraClient):
        super().__init__(client, method="field", update_steps=CHUNK_SIZE)
        self.method = "field_history"

    def reset(self) -> None:
        super().reset()
        self.client.reset_history()
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
        original = super().act(allowed)
        step = original.diagnostics["step"]
        details = {**original.diagnostics, "chunk_index": step // CHUNK_SIZE,
                   "chunk_step": step % CHUNK_SIZE,
                   "history_stats": self.client.history_stats(),
                   "feedback": copy.deepcopy(self.latest_feedback)}
        decision = Decision(original.action, details)
        self._pending = (allowed, decision)
        return decision

    # The exact baseline feedback schema, thresholds, rolling window, and
    # pending-action identity checks remain shared rather than reimplemented.
    observe_transition = DirectHistoryPolicy.observe_transition
