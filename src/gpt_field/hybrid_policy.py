"""Astra selects field planning or direct control at five-action boundaries.

Only the routing and orchestration are new. Coarse actions pass through the
original FieldPolicy; direct actions are executed exactly as validated. Both
branches share one transcript, step counter and observable-feedback window.
"""

from __future__ import annotations

import copy
import time

from .direct_history import CHUNK_SIZE, DirectHistoryPolicy, _allowed
from .hybrid_subscription import validate_hybrid_prediction
from .observation import PolicyObservation
from .policy import Decision, FieldPolicy
from .schema import GridSpec


class _FieldCache:
    """Supply one already validated field to the frozen planner without I/O."""

    def __init__(self, client):
        self.client = client
        self.clear()

    def clear(self):
        self.observation = self.prediction = None
        self.consumed = False

    def prepare(self, observation, prediction):
        self.observation = observation
        self.prediction = copy.deepcopy(prediction)
        self.consumed = False

    @property
    def last_request_key(self):
        return self.client.last_request_key

    @property
    def last_log_path(self):
        return self.client.last_log_path

    def predict(self, observation, grid, mode="field"):
        if mode != "field" or grid != GridSpec():
            raise ValueError("Cached coarse predictions use the original field grid")
        if self.consumed or self.prediction is None:
            raise RuntimeError("A fresh hybrid field is required before planning")
        if PolicyObservation.from_dict(observation) != self.observation:
            raise ValueError("Field must be centered at its actual generation observation")
        self.consumed = True
        return copy.deepcopy(self.prediction)


class HybridFieldDirectPolicy:
    """Reversible mode selection, committed for five actually executed actions.

Routing is Astra's visual judgment, not an oracle proximity/contact predicate.
No forced closure, grasp controller, automatic recovery, or reward input exists.
"""

    def __init__(self, client):
        self.client = client
        self.grid = GridSpec()
        self.method = "hybrid_field_direct"
        self.update_steps = CHUNK_SIZE
        self._field_cache = _FieldCache(client)
        self._field_policy = FieldPolicy(self._field_cache, method="field",
                                         update_steps=CHUNK_SIZE)
        self.reset()

    @property
    def planner(self):
        return self._field_policy.planner

    @property
    def adapter(self):
        return self._field_policy.adapter

    @property
    def field(self):
        return self._field_policy.field if self.controller == "coarse_field" else None

    def reset(self):
        self.client.reset_history()
        self._field_cache.clear()
        self._field_policy.reset()
        self.step_index = 0
        self.controller = None
        self.prediction = None
        self.routing = None
        self.chunk = []
        self.latest_feedback = None
        self._pending = None
        self._expected_observation = None
        self._progress = []

    def act(self, observation: PolicyObservation) -> Decision:
        allowed = _allowed(observation.to_dict())
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next action")
        if self._expected_observation is not None and allowed != self._expected_observation:
            raise ValueError("Next observation must equal the preceding transition's after observation")
        step = self.step_index
        refreshed = step % CHUNK_SIZE == 0
        previous_controller = self.controller
        api_latency = None
        if refreshed:
            started = time.monotonic()
            envelope = validate_hybrid_prediction(
                self.client.predict(allowed.to_dict(), self.grid, mode="hybrid"), self.grid)
            api_latency = time.monotonic() - started
            self.prediction = copy.deepcopy(envelope)
            self.controller = envelope["controller"]
            self.routing = {key: copy.deepcopy(envelope[key]) for key in
                            ("controller", "target_region", "near_target", "routing_reason")}
            if self.controller == "coarse_field":
                self._field_cache.prepare(allowed, envelope["field"])
                self.chunk = []
            else:
                self.chunk = copy.deepcopy(envelope["direct"]["actions"])

        if self.controller == "coarse_field":
            # Global boundaries force a newly centered field after returning
            # from direct mode. The original planner and adapter do all math.
            self._field_policy.step_index = step
            original = self._field_policy.act(allowed)
            action, details = original.action, dict(original.diagnostics)
        elif self.controller == "contact_direct":
            if not self.chunk:
                raise RuntimeError("Direct chunk exhausted before its next refresh boundary")
            action = tuple(float(value) for value in self.chunk.pop(0))
            details = {"status": "direct_action", "stage": "contact_direct:" + self.routing["target_region"]}
            if refreshed:
                details["prediction"] = copy.deepcopy(self.prediction["direct"])
        else:
            raise RuntimeError("No validated controller has been selected")

        details.update({"step": step, "method": self.method, "refreshed": refreshed,
                        "controller": self.controller, "routing": copy.deepcopy(self.routing),
                        "routing_decision_step": step - step % CHUNK_SIZE,
                        "routing_is_model_prediction": True,
                        "previous_controller": previous_controller,
                        "mode_changed": previous_controller is not None and previous_controller != self.controller,
                        "chunk_index": step // CHUNK_SIZE, "chunk_step": step % CHUNK_SIZE,
                        "field_refreshed": refreshed and self.controller == "coarse_field",
                        "direct_chunk_refreshed": refreshed and self.controller == "contact_direct",
                        "history_stats": self.client.history_stats(),
                        "feedback": copy.deepcopy(self.latest_feedback)})
        if refreshed:
            details.update({"hybrid_prediction": copy.deepcopy(self.prediction),
                            "api_latency_s": api_latency,
                            "api_request_key": self.client.last_request_key,
                            "api_request_log": self.client.last_log_path})
        decision = Decision(tuple(action), details)
        self._pending = (allowed, decision)
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        feedback = DirectHistoryPolicy.observe_transition(self, before, decision, after)
        # A direct zero is executed as zero, while the last nonzero open/close
        # intent remains available to a later field's "hold" command.
        grip = decision.action[6]
        if grip != 0:
            self.adapter._gripper = 1.0 if grip > 0 else -1.0
        return feedback
