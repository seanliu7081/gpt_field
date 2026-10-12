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
from .tracking import PathTracker


class Predictor(Protocol):
    def predict(self, observation: dict, grid: GridSpec, mode: str = "field") -> dict: ...


@dataclass(frozen=True)
class Decision:
    action: tuple[float, ...]
    diagnostics: dict


class FieldPolicy:
    """Refresh the field every update_steps; execute a configured prefix per plan."""

    METHODS = ("field", "field_one", "affine", "shuffled", "minimum", "waypoint", "direct")

    def __init__(self, client: Predictor, *, method: str = "field", grid: GridSpec | None = None,
                 planner_config: PlannerConfig | None = None, update_steps: int = 5,
                 seed: int = 7, execution_steps: int = 1, execution_mode: str = "fixed",
                 tracking_lookahead_m: float = 0.03, tracking_max_command_m: float = 0.05,
                 field_improvements: bool = False):
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
        if type(execution_steps) is not int or not 1 <= execution_steps <= self.planner.config.horizon:
            raise ValueError("execution_steps must be an integer between 1 and the planner horizon")
        if method == "direct" and execution_steps != 1:
            raise ValueError("direct baseline requires execution_steps=1")
        if execution_mode not in ("fixed", "tracking"):
            raise ValueError("execution_mode must be 'fixed' or 'tracking'")
        if method == "direct" and execution_mode != "fixed":
            raise ValueError("direct baseline requires fixed execution_mode")
        self.execution_steps = execution_steps
        self.execution_mode = execution_mode
        if type(field_improvements) is not bool:
            raise ValueError("field_improvements must be a boolean")
        if field_improvements and (method != "field" or execution_mode != "tracking"):
            raise ValueError("Field improvements require field tracking")
        self.field_improvements = field_improvements
        self.tracker = PathTracker(lookahead_m=tracking_lookahead_m, max_command_m=tracking_max_command_m)
        if field_improvements:
            self.tracker.endpoint_tolerance_m = 0.005
        self.adapter = ActionAdapter()
        self.reset()

    def clear_pending_plan(self):
        """Discard an unexecuted field-plan suffix without changing gripper intent."""
        self._plan_result = None
        self._plan_deltas = []
        self._plan_start_step = None
        self._plan_execution_steps = 0
        self.tracker.reset()

    def reset(self):
        self.field = None
        self.prediction = None
        self.chunk = []
        self.step_index = 0
        self.last_stage = None
        self.clear_pending_plan()
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
                self.clear_pending_plan()
        if self.method == "direct":
            action = tuple(self.chunk.pop(0))
            details["status"] = "direct_action"
        else:
            plan_refreshed = (self.execution_steps == 1 or not self._plan_deltas
                              or (self.field_improvements
                                  and self.tracker.endpoint_reached(allowed.state[:3])))
            if plan_refreshed:
                plan_started = time.monotonic()
                compact = getattr(self.client, "effective_compact_field", None)
                if self.field_improvements:
                    if compact is None:
                        raise RuntimeError("Improved planning requires the effective compact prediction")
                    goal = np.asarray(self.field.center) + np.asarray(compact["goal_offset_m"])
                    result = self.planner.plan_analytic(
                        self.field, allowed.state[:3], goal_xyz=goal,
                        compact_prediction=compact, goal_tolerance_m=0.005)
                else:
                    result = self.planner.plan(self.field, allowed.state[:3])
                details["plan_latency_s"] = time.monotonic() - plan_started
                if self.execution_mode == "tracking":
                    self.tracker.reset(None if result.selected_index is None else
                                       result.candidates[result.selected_index].positions)
                if self.execution_steps > 1:
                    self._plan_result = result
                    self._plan_start_step = self.step_index
                    self._plan_execution_steps = min(
                        self.execution_steps, self.update_steps - self.step_index % self.update_steps)
                    if result.selected_index is None:
                        self._plan_deltas = [(0., 0., 0.)] * self._plan_execution_steps
                    else:
                        selected = result.candidates[result.selected_index]
                        self._plan_deltas = []
                        for index, primitive in enumerate(selected.primitives[:self._plan_execution_steps]):
                            # Fixed world-frame primitives, independent of later
                            # measured tracking error or the recorded waypoints.
                            if self.field_improvements:
                                delta = np.subtract(selected.positions[index + 1], selected.positions[index]).tolist()
                            else:
                                delta = [0., 0., 0.]
                                delta["xyz".index(primitive[1])] = self.planner.config.step_m * (
                                    1 if primitive[0] == "+" else -1)
                            self._plan_deltas.append(tuple(delta))
                        # Geometric waypoints and control ticks are independent:
                        # a short path still receives feedback tracking until it
                        # arrives or the ten-action local planning interval ends.
                        if self.field_improvements:
                            self._plan_deltas += [(0., 0., 0.)] * (
                                self._plan_execution_steps - len(self._plan_deltas))
            else:
                result = self._plan_result
            if self.execution_steps > 1:
                delta_xyz = self._plan_deltas.pop(0)
                details.update(plan_refreshed=plan_refreshed, plan_start_step=self._plan_start_step,
                               plan_step=self.step_index - self._plan_start_step,
                               plan_execution_steps=self._plan_execution_steps,
                               planned_delta_xyz=list(delta_xyz))
            else:
                delta_xyz = result.delta_xyz
            if self.execution_mode == "tracking":
                details["execution_mode"] = "tracking"
                details["planned_delta_xyz"] = list(delta_xyz)
                delta_xyz, tracking = self.tracker.command(
                    allowed.state[:3], self.field, self.planner.config.workspace_min,
                    self.planner.config.workspace_max)
                details["commanded_delta_xyz"] = list(delta_xyz)
                details["tracking"] = tracking
            details["plan"] = result.to_dict()
            details["status"] = result.status
            details["stage"] = self.prediction["stage"]
            details["field_center"] = list(self.field.center)
            action = tuple(self.adapter.adapt(delta_xyz,
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
