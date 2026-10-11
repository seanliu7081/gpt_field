"""One truthful video timeline for model-routed field and direct control."""

from __future__ import annotations

from copy import deepcopy
import json
import time

from PIL import Image, ImageDraw

from .direct_history_video import DirectHistoryVideoRecorder
from .field import GridField
from .field_history_video import FieldHistoryVideoRecorder
from .schema import FieldPrediction
from .video import BG, HEIGHT, MUTED, TEXT, WIDTH, YELLOW, _font, _short


class HybridVideoRecorder(FieldHistoryVideoRecorder):
    """Field heatmaps for coarse control, action tables for contact control.

Routing is a model prediction, never a simulator contact assertion. Switching
to direct control clears the rendered field and local plan completely. This
recorder does not infer success or contacts, and never makes a model request.
"""

    def __init__(self, path, *, grid=None, episode_id="hybrid", fps=20, offline_smoke=False):
        self.controller = None
        self.routing = None
        self.previous_controller = None
        self.chunk = []
        self.chunk_step = 0
        self.hybrid_refresh_count = 0
        self.field_count = 0
        self.direct_chunk_count = 0
        super().__init__(path, grid=grid, episode_id=episode_id, fps=fps, offline_smoke=offline_smoke)

    def decision(self, observation, decision, *, decision_latency_s=None):
        self.observation = observation
        self.details = deepcopy(decision.diagnostics)
        self.step = int(self.details["step"])
        self.action = tuple(decision.action)
        self.history_stats = deepcopy(self.details.get("history_stats"))
        self.history_stats_scope = "decision" if self.history_stats is not None else "not_recorded"
        self.feedback = deepcopy(self.details.get("feedback"))
        self.chunk_step = int(self.details.get("chunk_step", self.step % 5))
        self.decision_latency_s = decision_latency_s
        self._previous_xyz = tuple(observation.state[:3])
        self._measured_delta = None
        self.previous_controller, self.controller = self.controller, self.details["controller"]
        if self.controller not in ("coarse_field", "contact_direct"):
            raise ValueError("Unknown hybrid controller in video decision")
        self.routing = deepcopy(self.details["routing"])
        fresh = bool(self.details.get("refreshed"))
        if self.controller == "contact_direct":
            self.field = None
            self.field_step = None
            self.details.pop("plan", None)
            self.details.pop("field_prediction", None)
            self.details.pop("field_center", None)
        elif fresh:
            self.field = GridField(self.grid, self.details["field_center"],
                                  FieldPrediction.from_dict(self.details["field_prediction"], self.grid))
            self.field_step = self.step
            self.chunk = []
        if fresh:
            self.hybrid_refresh_count += 1
            self.last_api_latency_s = self.details.get("api_latency_s")
            self.last_api_request_key = self.details.get("api_request_key")
            self.last_api_request_log = self.details.get("api_request_log")
            if self.controller == "coarse_field":
                self.field_count += 1
                self.refresh_count = self.field_count
            else:
                self.direct_chunk_count += 1
                self.refresh_count = self.direct_chunk_count
                self.chunk = deepcopy(self.details["prediction"]["actions"])
            complete_details, complete_action = self.details, self.action
            self.details = {key: value for key, value in complete_details.items() if key != "plan"}
            self.action = ()
            try:
                self._emit("routing_decision", max(1, round(self.writer.fps * .3)))
                if self.previous_controller is not None and self.previous_controller != self.controller:
                    self._emit("controller_switch", max(1, round(self.writer.fps * .4)))
                self._emit("field_refresh" if self.controller == "coarse_field" else "action_chunk",
                           max(1, round(self.writer.fps * .6)))
            finally:
                self.details, self.action = complete_details, complete_action
        self._emit("local_replan" if self.controller == "coarse_field" else "direct_action",
                   max(2, round(self.writer.fps * .15)) if fresh and self.controller == "coarse_field" else 2)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None, history_stats=None):
        if history_stats is not None:
            self.details["history_stats"] = deepcopy(history_stats)
        super().executed(observation, step=step, success=success, sim_status=sim_status,
                         feedback=feedback, history_stats=history_stats)

    def _emit(self, event, repeats, **extra):
        if self.closed:
            raise RuntimeError("Cannot record a closed hybrid episode")
        self.event = event
        first = self.frame_count
        self.writer.write(self.render_frame(), repeats)
        plan = self.details.get("plan", {}) if self.controller == "coarse_field" else {}
        record = {"event": event, "step": self.step, "frame_start": first, "frame_count": repeats,
            "video_time_s": first / self.writer.fps, "wall_elapsed_s": time.monotonic() - self._started,
            "policy": "hybrid_field_direct", "offline_mock": self.offline_mock, "offline_smoke": self.offline_smoke,
            "controller": self.controller, "previous_controller": self.previous_controller,
            "routing": deepcopy(self.routing), "routing_source": "model_prediction",
            "hybrid_refresh_count": self.hybrid_refresh_count, "field_refresh_count": self.field_count,
            "direct_chunk_count": self.direct_chunk_count, "chunk_step": self.chunk_step,
            "field_step": self.field_step, "field_center": list(self.field.center) if self.field is not None else None,
            "status": self.status, "success": self.success,
            "decision_latency_s": self.decision_latency_s, "api_latency_s": self.last_api_latency_s,
            "api_request_key": self.last_api_request_key, "api_request_log": self.last_api_request_log,
            "plan_latency_s": self.details.get("plan_latency_s") if plan else None,
            "selected_index": plan.get("selected_index"), "candidate_count": plan.get("candidate_count"),
            "valid_candidate_count": plan.get("valid_count"),
            "state": list(self.observation.state) if self.observation else None, "action": list(self.action),
            "history_stats": deepcopy(self.history_stats), "history_stats_scope": self.history_stats_scope,
            "feedback": deepcopy(self.feedback), "feedback_step": self.feedback.get("step") if self.feedback else None,
            "measured_delta_xyz_m": list(self._measured_delta) if self._measured_delta is not None else None, **extra}
        if event == "routing_decision":
            record["hybrid_prediction"] = deepcopy(self.details.get("hybrid_prediction"))
        if event == "field_refresh":
            record["field_prediction"] = deepcopy(self.details["field_prediction"])
        if event == "action_chunk":
            record["actions"] = deepcopy(self.chunk)
        self._events.write(json.dumps(record, allow_nan=False) + "\n")
        self._events.flush()

    def render_frame(self):
        if self.controller != "contact_direct":
            canvas = super().render_frame()
            draw = ImageDraw.Draw(canvas)
        else:
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
            draw = ImageDraw.Draw(canvas)
            if self.observation is not None:
                draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
                self._cameras(canvas, draw)
                DirectHistoryVideoRecorder._chunk_panel(self, draw)
                DirectHistoryVideoRecorder._trace_panel(self, draw, (425, 443, 974, 651), (0, 1))
                DirectHistoryVideoRecorder._trace_panel(self, draw, (994, 443, 1576, 651), (0, 2))
                DirectHistoryVideoRecorder._event_strip_direct(self, draw)
                DirectHistoryVideoRecorder._feedback_panel(self, draw)
                self._footer(draw)
        draw.rectangle((0, 0, WIDTH, 87), fill=BG)
        title = ("SCRIPTED SMOKE / LIBERO | HYBRID CONTROL | NO LIVE MODEL CALLS" if self.offline_smoke else
                 "ASTRA / LIBERO | HYBRID FIELD + DIRECT ACTIONS + HISTORY")
        draw.text((24, 17), title, font=_font(25, True), fill=YELLOW if self.offline_smoke else TEXT)
        routing = self.routing or {}
        mode = (self.controller or "awaiting model routing").upper().replace("_", " ")
        line = (f"CHATGPT SUBSCRIPTION | {mode} | target: {routing.get('target_region', '-')} | "
                f"near target: {routing.get('near_target', '-')} (model prediction, not contact measurement)")
        draw.text((24, 58), _short(line, 155), font=_font(17), fill=MUTED)
        return canvas
