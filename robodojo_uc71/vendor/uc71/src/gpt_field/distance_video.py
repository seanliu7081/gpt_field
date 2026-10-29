"""Render local Dijkstra cost maps separately from the model's raw guidance."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import ImageDraw

from .field import GridField
from .hybrid_video import HybridVideoRecorder
from .schema import FieldPrediction
from .video import BG, GREEN, MUTED, PANEL, TEXT, WIDTH, YELLOW, _font


class DistanceHybridVideoRecorder(HybridVideoRecorder):
    """Drop-in hybrid recorder for DistanceHybridPolicy decision diagnostics.

    ``field_prediction`` remains the original guidance in recorded events.
    ``distance_field_display`` is a separate, finite display representation of
    the local solve, with unreachable points masked; it never feeds the policy.
    Local revisions replace the displayed map without inventing a GPT refresh.
    Missing solves never borrow a preceding field's map.
    """

    def __init__(self, *args, **kwargs):
        self._distance_display = None
        self._distance_step = None
        self._distance_goal = None
        self._distance_cost_unit = None
        super().__init__(*args, **kwargs)

    def decision(self, observation, decision, *, decision_latency_s=None):
        details = decision.diagnostics
        search = details.get("plan", {}).get("search") or {}
        display = details.get("distance_field_display")
        controller = details["controller"]
        plan_status = details.get("plan", {}).get("status")
        if controller != "coarse_field":
            self._clear_distance_display()
        elif display is not None:
            if search.get("method") != "dijkstra_distance_field" or plan_status != "move":
                raise ValueError("Distance display requires a current Dijkstra move plan")
            self._distance_display = GridField(
                self.grid, details["field_center"], FieldPrediction.from_dict(display, self.grid))
            self._distance_step = int(details["step"])
            self._distance_goal = deepcopy(search.get("goal_xyz"))
            self._distance_cost_unit = search.get("cost_unit", "m")
        elif details.get("refreshed") or details.get("local_field_updated") or details.get("plan_refreshed"):
            # A cache-hit replan can use this exact previous display, but a new
            # goal/center, hold, invalid start, or absent solve cannot do so.
            same_cached_map = (
                self._distance_display is not None and search.get("cache_hit") is True
                and search.get("method") == "dijkstra_distance_field" and plan_status == "move"
                and list(self._distance_display.center) == list(details.get("field_center", ()))
                and self._distance_goal == search.get("goal_xyz"))
            if not same_cached_map:
                self._clear_distance_display()
        # The parent deep-copies diagnostics and logs raw guidance unchanged.
        # Its fresh-field assignment is corrected immediately before each emit.
        super().decision(observation, decision, decision_latency_s=decision_latency_s)

    def _clear_distance_display(self):
        self._distance_display = None
        self._distance_step = None
        self._distance_goal = None
        self._distance_cost_unit = None

    def _emit(self, event, repeats, **extra):
        if self.controller != "coarse_field" or event == "model_completion":
            self._clear_distance_display()
        self.field = self._distance_display
        self.field_step = self._distance_step
        metadata = {
            "field_display_source": ("local_dijkstra_cost_to_go" if self.field is not None else "unavailable"),
            "field_display_cost_unit": self._distance_cost_unit,
            "local_field_revision": self.details.get("local_field_revision", 0),
            "local_goal_updates": self.details.get("local_goal_updates", 0),
            "field_reuse_extensions": self.details.get("field_reuse_extensions", 0),
            "local_visual_feedback": deepcopy(self.details.get("local_visual_feedback")),
            "local_metadata_scope": "pre_action_decision",
            "task_success_certified": False,
        }
        if event in ("field_refresh", "local_replan") and self.details.get("distance_field_display") is not None:
            metadata["distance_field_display"] = deepcopy(self.details["distance_field_display"])
        super()._emit(event, repeats, **{**metadata, **extra})

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 50), fill=BG)
        title = ("SCRIPTED SMOKE | LOCAL DISTANCE FIELD + DIRECT | NO LIVE MODEL CALLS"
                 if self.offline_smoke else "ASTRA / LIBERO | LOCAL DISTANCE FIELD + DIRECT CONTROL")
        draw.text((24, 17), title, font=_font(25, True), fill=YELLOW if self.offline_smoke else TEXT)
        return canvas

    def _field_panels(self, canvas, draw):
        super()._field_panels(canvas, draw)
        draw.rectangle((425, 138, 1576, 169), fill=BG)
        draw.text((425, 142), "LOCAL DIJKSTRA COST-TO-GO + SELECTED PATH", font=_font(20, True), fill=TEXT)
        if self.field is None:
            draw.rectangle((444, 237, 1560, 333), fill=PANEL)
            draw.text((450, 245), "No current local distance map", font=_font(25), fill=MUTED)
            draw.text((450, 290), "No valid solve is available for this event. Raw GPT guidance is retained in the logs.",
                      font=_font(17), fill=MUTED)
            return
        draw.rectangle((425, 561, 1580, 590), fill=BG)
        line = (f"Local solve from step {self.field_step} | model field #{self.field_count} | "
                f"local revision {self.details.get('local_field_revision', 0)} | "
                f"reuse extensions {self.details.get('field_reuse_extensions', 0)}")
        draw.text((425, 565), line, font=_font(16, True), fill=TEXT)
        draw.rectangle((425, 615, 1580, 644), fill=BG)
        unit = "weighted meters" if self._distance_cost_unit == "weighted_m" else "meters (grid graph)"
        values = np.asarray(self.field.prediction.values)
        valid = np.ones(len(values), dtype=bool)
        valid[list(self.field.prediction.blocked_indices + self.field.prediction.unknown_indices)] = False
        samples = values[valid]
        span = f"{samples.min():.4g} to {samples.max():.4g}" if len(samples) else "no reachable vertices"
        draw.text((425, 619), f"Cost-to-go in {unit}: {span}. Teal=lower; orange=higher; purple=blocked; gray=unknown/unreachable.",
                  font=_font(14), fill=MUTED)

    def _event_strip(self, draw):
        super()._event_strip(draw)
        labels = ("1 OBSERVE", "2 GPT GUIDANCE", "3 DIJKSTRA PLAN", "4 EXECUTE / FEEDBACK")
        active = {"observation": 0, "routing_decision": 1, "field_refresh": 1,
                  "local_replan": 2, "field_action": 3, "executed": 3}.get(self.event)
        for index, label in enumerate(labels):
            x = 425 + index * 291
            draw.rounded_rectangle((x, 654, x + 277, 695), radius=8,
                                   fill=(32, 91, 90) if index == active else PANEL,
                                   outline=GREEN if index == active else (44, 56, 74), width=2)
            draw.text((x + 11, 666), label, font=_font(14, True), fill=TEXT if index == active else MUTED)
