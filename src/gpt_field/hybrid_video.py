"""One truthful video timeline for model-routed field and direct control."""

from __future__ import annotations

from copy import deepcopy
import json
import time

from PIL import Image, ImageDraw

from .direct_history_video import DirectHistoryVideoRecorder
from .field import GridField
from .field_history_video import FieldHistoryVideoRecorder
from .hybrid_subscription import HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID
from .schema import FieldPrediction, GridSpec
from .video import BG, HEIGHT, MUTED, PANEL, TEXT, WIDTH, YELLOW, _font, _short, _vector


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
        self.update_steps = 5 if grid == GridSpec() else HYBRID_FIELD_UPDATE_STEPS
        self.hybrid_refresh_count = 0
        self.field_count = 0
        self.direct_chunk_count = 0
        super().__init__(path, grid=grid if grid is not None else HYBRID_GRID,
                         episode_id=episode_id, fps=fps, offline_smoke=offline_smoke)

    def decision(self, observation, decision, *, decision_latency_s=None):
        self.observation = observation
        self.details = deepcopy(decision.diagnostics)
        self.step = int(self.details["step"])
        self.action = tuple(decision.action)
        self.history_stats = deepcopy(self.details.get("history_stats"))
        self.history_stats_scope = "decision" if self.history_stats is not None else "not_recorded"
        self.feedback = deepcopy(self.details.get("feedback"))
        self.decision_latency_s = decision_latency_s
        self._previous_xyz = tuple(observation.state[:3])
        self._measured_delta = None
        self.previous_controller, self.controller = self.controller, self.details["controller"]
        if self.controller not in ("coarse_field", "contact_direct"):
            raise ValueError("Unknown hybrid controller in video decision")
        # Historical recordings may lack explicit per-controller intervals.
        # Their direct response length and original grid preserve the old cadence.
        if self.controller == "contact_direct":
            actions = self.details.get("prediction", {}).get("actions", self.chunk)
            default_steps = len(actions) or HYBRID_CHUNK_SIZE
        else:
            default_steps = 5 if self.grid == GridSpec() else HYBRID_FIELD_UPDATE_STEPS
        self.update_steps = int(self.details.get("update_steps", default_steps))
        routing_start = self.details.get("routing_decision_step", self.step - self.step % self.update_steps)
        self.chunk_step = int(self.details.get("chunk_step", self.step - routing_start))
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
            plan_keys = {"plan", "plan_refreshed", "plan_start_step", "plan_step",
                         "plan_execution_steps", "planned_delta_xyz", "commanded_delta_xyz", "tracking"}
            self.details = {key: value for key, value in complete_details.items() if key not in plan_keys}
            self.action = ()
            try:
                self._emit("routing_decision", max(1, round(self.writer.fps * .3)))
                if self.previous_controller is not None and self.previous_controller != self.controller:
                    self._emit("controller_switch", max(1, round(self.writer.fps * .4)))
                self._emit("field_refresh" if self.controller == "coarse_field" else "action_chunk",
                           max(1, round(self.writer.fps * .6)))
            finally:
                self.details, self.action = complete_details, complete_action
        if self.controller == "coarse_field":
            event = "local_replan" if self.details.get("plan_refreshed", True) else "field_action"
            pause_for_plan = self.details.get("plan_refreshed", fresh)
        else:
            event, pause_for_plan = "direct_action", False
        self._emit(event, max(2, round(self.writer.fps * .15)) if pause_for_plan else 2)

    def model_completion(self, observation, *, step, completion_reason,
                         decision_latency_s=None, diagnostics):
        """Record a terminal model judgment without inventing a robot action."""
        self.observation = observation
        self.step = step
        self.details = deepcopy(diagnostics)
        self.action = ()
        self.previous_controller = self.controller
        self.controller = diagnostics["controller"]
        self.routing = deepcopy(diagnostics["routing"])
        self.field = None
        self.field_step = None
        self.chunk = []
        self.chunk_step = 0
        self.update_steps = 5 if self.controller == "contact_direct" else HYBRID_FIELD_UPDATE_STEPS
        self._measured_delta = None
        self.history_stats = deepcopy(diagnostics["history_stats"])
        self.history_stats_scope = "decision"
        self.feedback = deepcopy(diagnostics.get("feedback"))
        self.decision_latency_s = decision_latency_s
        self.last_api_latency_s = diagnostics["api_latency_s"]
        self.last_api_request_key = diagnostics["api_request_key"]
        self.last_api_request_log = diagnostics["api_request_log"]
        self.hybrid_refresh_count += 1
        if self.controller == "coarse_field":
            self.field_count += 1
            self.refresh_count = self.field_count
        else:
            self.direct_chunk_count += 1
            self.refresh_count = self.direct_chunk_count
        self.status = "model_declared_complete"
        self._emit("model_completion", max(1, round(self.writer.fps)),
                   task_complete=True, completion_reason=completion_reason,
                   compact_prediction=deepcopy(diagnostics["compact_prediction"]))

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
            "routing": deepcopy(self.routing), "routing_source": (
                "model_prediction" if self.details.get("routing_is_model_prediction", True) else "forced_direct"),
            "hybrid_refresh_count": self.hybrid_refresh_count, "field_refresh_count": self.field_count,
            "direct_chunk_count": self.direct_chunk_count, "chunk_step": self.chunk_step,
            "update_steps": self.update_steps, "grid": self.grid.to_dict(),
            "field_step": self.field_step, "field_center": list(self.field.center) if self.field is not None else None,
            "status": self.status, "success": self.success,
            "decision_latency_s": self.decision_latency_s, "api_latency_s": self.last_api_latency_s,
            "api_request_key": self.last_api_request_key, "api_request_log": self.last_api_request_log,
            "plan_latency_s": self.details.get("plan_latency_s") if plan else None,
            "selected_index": plan.get("selected_index"), "candidate_count": plan.get("candidate_count"),
            "valid_candidate_count": plan.get("valid_count"),
            "planner_search": deepcopy(plan.get("search")),
            "state": list(self.observation.state) if self.observation else None, "action": list(self.action),
            "history_stats": deepcopy(self.history_stats), "history_stats_scope": self.history_stats_scope,
            "feedback": deepcopy(self.feedback), "feedback_step": self.feedback.get("step") if self.feedback else None,
            "measured_delta_xyz_m": list(self._measured_delta) if self._measured_delta is not None else None, **extra}
        if plan and "plan_refreshed" in self.details:
            record.update({key: deepcopy(self.details.get(key)) for key in (
                "plan_refreshed", "plan_start_step", "plan_step", "plan_execution_steps", "planned_delta_xyz")})
            for key in ("execution_mode", "commanded_delta_xyz", "tracking"):
                if key in self.details:
                    record[key] = deepcopy(self.details[key])
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
                self._chunk_panel(draw)
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

    def _chunk_panel(self, draw):
        """Page through a full chunk while keeping the issued action visible."""
        draw.rounded_rectangle((425, 139, 1576, 427), radius=12, fill=PANEL)
        label = "SCRIPTED RESPONSE" if self.offline_smoke else "ASTRA ACTION CHUNK"
        count = len(self.chunk)
        page_size = 5
        start = (min(self.chunk_step, count - 1) // page_size) * page_size if count else 0
        stop = min(start + page_size, count)
        heading = (f"{label}  #{self.direct_chunk_count}  |  {count} actions, "
                   f"{count / 20:g} s simulated time  |  showing {start + 1}-{stop}")
        draw.text((446, 153), heading, font=_font(20, True), fill=TEXT)
        draw.text((446, 186), "Normalized controls: dx, dy, dz, rx, ry, rz, gripper (+close / -open / 0 no increment)",
                  font=_font(16), fill=MUTED)
        if not self.chunk:
            draw.text((446, 244), "Awaiting first action chunk", font=_font(22), fill=MUTED)
        for index in range(start, stop):
            action = self.chunk[index]
            y = 221 + (index - start) * 32
            active = index == self.chunk_step and bool(self.action)
            if active:
                draw.rounded_rectangle((440, y - 3, 1557, y + 26), radius=5, fill=(34, 73, 74))
            grip = "CLOSE" if action[6] > 0 else "OPEN" if action[6] < 0 else "ZERO"
            draw.text((450, y), f"{index + 1}   {_vector(action, 3)}     {grip}",
                      font=_font(18, active), fill=YELLOW if active else TEXT)
        history = self.details.get("history_stats", {})
        history_line = (f"History: {history.get('observations', 0)} model observations | "
                        f"{history.get('feedback_records', 0)} executed-step reports | "
                        f"{history.get('retained_image_pairs', 0)} image pairs | "
                        f"{history.get('messages', 0)} records")
        draw.text((446, 393), history_line, font=_font(14), fill=MUTED)

    def _footer(self, draw):
        if self.status == "model_declared_complete":
            text = "ASTRA COMPLETION JUDGMENT | No action issued from terminal response. LIBERO success is recorded separately."
            draw.text((24, 932), text, font=_font(13), fill=MUTED)
            return
        if self.controller is None:
            text = "Awaiting model routing. "
        else:
            text = (f"Controller block: {self.update_steps} actions, "
                    f"{self.update_steps / 20:g} s simulated time. ")
        text += "Simulator 20 Hz. API waits omitted. Feedback uses robot observations; LIBERO evaluates success separately."
        draw.text((24, 932), text, font=_font(13), fill=MUTED)
