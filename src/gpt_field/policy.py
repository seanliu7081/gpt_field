"""Policy orchestration. This module never imports the simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Protocol

import numpy as np

from .action import ActionAdapter
from .baselines import transform_field, waypoint_field
from .field import GridField
from .observation import PolicyObservation, decode_image
from .planner import LocalPlanner, PlannerConfig
from .schema import FieldPrediction, GridSpec


class Predictor(Protocol):
    def predict(self, observation: dict, grid: GridSpec, mode: str = "field") -> dict: ...


@dataclass(frozen=True)
class Decision:
    action: tuple[float, ...]
    diagnostics: dict


class FieldPolicy:
    """Use one fresh prediction every update_steps, replan from measured xyz every step."""

    METHODS = ("field", "field_one", "affine", "shuffled", "minimum", "waypoint", "direct")

    def __init__(self, client: Predictor, *, method: str = "field", grid: GridSpec | None = None,
                 planner_config: PlannerConfig | None = None, update_steps: int = 5,
                 seed: int = 7):
        if method not in self.METHODS:
            raise ValueError(f"unknown method {method!r}")
        if type(update_steps) is not int or update_steps < 1:
            raise ValueError("update_steps must be a positive integer")
        if method == "direct" and update_steps != 5:
            raise ValueError("direct baseline returns five-action chunks; update_steps must be 5")
        self.client, self.method = client, method
        self.grid, self.update_steps, self.seed = grid or GridSpec(), update_steps, seed
        config = planner_config or PlannerConfig()
        self.planner = LocalPlanner(replace(config, horizon=1) if method == "field_one" else config)
        self.adapter = ActionAdapter()
        self.reset()

    def reset(self):
        self.field = None
        self.prediction = None
        self.chunk = []
        self.step_index = 0
        self.last_stage = None
        self.adapter.reset()

    def act(self, observation: PolicyObservation) -> Decision:
        # Revalidate even direct caller-created objects at this boundary.
        allowed = PolicyObservation.from_dict(observation.to_dict())
        refresh = self.step_index % self.update_steps == 0
        details = {"step": self.step_index, "method": self.method, "refreshed": refresh}
        if refresh:
            mode = self.method if self.method in ("waypoint", "direct") else "field"
            api_started = time.monotonic()
            raw = self.client.predict(allowed.to_dict(), self.grid, mode=mode)
            details["api_latency_s"] = time.monotonic() - api_started
            details["api_request_key"] = getattr(self.client, "last_request_key", None)
            details["api_request_log"] = getattr(self.client, "last_log_path", None)
            self.prediction = raw
            details["prediction"] = raw
            if self.method == "direct":
                actions = np.asarray(raw["actions"], dtype=float)
                if actions.shape != (5, 7) or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
                    raise ValueError("direct baseline must provide five finite normalized 7D actions")
                self.chunk = actions.tolist()
            else:
                center = allowed.state[:3]
                if self.method == "waypoint":
                    self.field = waypoint_field(self.grid, center, raw, raw["waypoint_xyz"])
                else:
                    prediction = FieldPrediction.from_dict(raw, self.grid)
                    self.field = GridField(self.grid, center, prediction)
                    transform = self.method if self.method in ("affine", "shuffled", "minimum") else "field"
                    self.field = transform_field(self.field, transform, seed=self.seed + self.step_index)
                stage = raw["stage"]
                details["stage_changed"] = self.last_stage is not None and stage != self.last_stage
                self.last_stage = stage
                details["field_center"] = list(center)
                details["field_prediction"] = self.field.prediction.to_dict()
        if self.method == "direct":
            action = tuple(self.chunk.pop(0))
            details["status"] = "direct_action"
        else:
            plan_started = time.monotonic()
            result = self.planner.plan(self.field, allowed.state[:3])
            details["plan_latency_s"] = time.monotonic() - plan_started
            details["plan"] = result.to_dict()
            details["status"] = result.status
            details["stage"] = self.prediction["stage"]
            details["field_center"] = list(self.field.center)
            action = tuple(self.adapter.adapt(result.delta_xyz,
                self.prediction["orientation_target_xyzw"],
                self.prediction["gripper_command"], allowed.state))
        self.step_index += 1
        return Decision(action, details)


class OpenPIPolicy:
    """Optional official openpi websocket reference. Server/checkpoint managed externally."""

    def __init__(self, host: str = "localhost", port: int = 8000, update_steps: int = 5):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.client = WebsocketClientPolicy(host, port)
        if type(update_steps) is not int or update_steps < 1:
            raise ValueError("update_steps must be positive")
        self.update_steps = update_steps
        self.reset()

    def reset(self):
        self.chunk = []
        self.step_index = 0

    def act(self, observation: PolicyObservation) -> Decision:
        observation = PolicyObservation.from_dict(observation.to_dict())
        refreshed = not self.chunk
        if refreshed:
            data = observation.to_dict()
            data["observation/image"] = np.asarray(decode_image(observation.image))
            data["observation/wrist_image"] = np.asarray(decode_image(observation.wrist_image))
            data["observation/state"] = np.asarray(observation.state)
            values = np.asarray(self.client.infer(data)["actions"], dtype=float)
            if values.ndim != 2 or values.shape[1] != 7 or len(values) < self.update_steps:
                raise ValueError("openpi server returned an invalid action chunk")
            chunk = values[:self.update_steps]
            if not np.isfinite(chunk).all() or (np.abs(chunk) > 1).any():
                raise ValueError("openpi actions must be finite and normalized")
            self.chunk = chunk.tolist()
        decision = Decision(tuple(self.chunk.pop(0)), {"method": "pi05", "refreshed": refreshed,
                                                     "step": self.step_index, "status": "reference_action"})
        self.step_index += 1
        return decision
