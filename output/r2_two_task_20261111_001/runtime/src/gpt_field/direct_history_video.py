"""Annotated direct-action playback using recorded observations and commands."""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image, ImageDraw

from .observation import PolicyObservation
from .policy import Decision
from .video import (
    BG, CYAN, GREEN, HEIGHT, MAGENTA, MUTED, PANEL, RED, TEXT, WIDTH,
    YELLOW, EpisodeVideoRecorder, _font, _short, _vector,
)


class DirectHistoryVideoRecorder(EpisodeVideoRecorder):
    """Show direct commands and measured feedback without inventing field events.

    ``offline_smoke`` labels every frame as a scripted integration diagnostic.
    It never represents mock responses as Astra-generated robot behavior.
    """

    def __init__(self, path, *, episode_id="direct_history", fps=20,
                 offline_smoke=False):
        self.offline_smoke = bool(offline_smoke)
        self.chunk = []
        self.chunk_step = 0
        self.feedback = None
        super().__init__(path, episode_id=episode_id, fps=fps)

    def decision(self, observation: PolicyObservation, decision: Decision, *,
                 decision_latency_s=None):
        self.observation = observation
        self.details = dict(decision.diagnostics)
        self.action = tuple(decision.action)
        self.step = int(self.details.get("step", self.step))
        self.chunk_step = int(self.details.get("chunk_step", self.step % 5))
        self.decision_latency_s = decision_latency_s
        self._previous_xyz = tuple(observation.state[:3])
        self._measured_delta = None
        self.feedback = None
        if self.details.get("refreshed"):
            self.refresh_count += 1
            self.chunk = self.details.get("prediction", {}).get("actions", [])
            self.last_api_latency_s = self.details.get("api_latency_s")
            self.last_api_request_key = self.details.get("api_request_key")
            self.last_api_request_log = self.details.get("api_request_log")
            issued = self.action
            self.action = ()
            try:
                self._emit("action_chunk", max(1, round(self.writer.fps * 0.6)))
            finally:
                self.action = issued
        self._emit("direct_action", 2)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None):
        self.feedback = feedback
        super().executed(observation, step=step, success=success, sim_status=sim_status)

    def _emit(self, event, repeats, **extra):
        if self.closed:
            raise RuntimeError("cannot record a closed episode")
        self.event = event
        first = self.frame_count
        self.writer.write(self.render_frame(), repeats)
        record = {
            "event": event, "step": self.step, "frame_start": first,
            "frame_count": repeats, "video_time_s": first / self.writer.fps,
            "wall_elapsed_s": time.monotonic() - self._started,
            "policy": "direct_history", "offline_smoke": self.offline_smoke,
            "chunk_count": self.refresh_count, "chunk_step": self.chunk_step,
            "status": self.status, "success": self.success,
            "decision_latency_s": self.decision_latency_s,
            "api_latency_s": self.last_api_latency_s,
            "api_request_key": self.last_api_request_key,
            "api_request_log": self.last_api_request_log,
            "state": list(self.observation.state) if self.observation else None,
            "action": list(self.action),
            "measured_delta_xyz_m": list(self._measured_delta) if self._measured_delta is not None else None,
            "history_stats": self.details.get("history_stats"),
            "feedback": self.feedback, **extra,
        }
        if event == "action_chunk":
            record["actions"] = self.chunk
        self._events.write(json.dumps(record, allow_nan=False) + "\n")
        self._events.flush()

    def render_frame(self):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        title = ("SCRIPTED SMOKE / LIBERO  |  NO LIVE ASTRA CALLS" if self.offline_smoke
                 else "ASTRA / LIBERO  |  DIRECT ACTIONS + HISTORY")
        draw.text((24, 17), title, font=_font(26, True), fill=YELLOW if self.offline_smoke else TEXT)
        draw.text((24, 57), _short(self.episode_id, 110), font=_font(18), fill=MUTED)
        if self.observation is None:
            return canvas
        draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
        self._cameras(canvas, draw)
        self._chunk_panel(draw)
        self._trace_panel(draw, (425, 443, 974, 651), (0, 1))
        self._trace_panel(draw, (994, 443, 1576, 651), (0, 2))
        self._event_strip_direct(draw)
        self._feedback_panel(draw)
        footer = ("Integration diagnostic only; scripted actions are not model performance. " if self.offline_smoke else "")
        footer += "Annotated playback; 20 Hz simulator. API waits omitted. Traces are measured positions."
        draw.text((24, 932), footer, font=_font(14), fill=MUTED)
        return canvas

    def _chunk_panel(self, draw):
        draw.rounded_rectangle((425, 139, 1576, 427), radius=12, fill=PANEL)
        label = "SCRIPTED RESPONSE" if self.offline_smoke else "ASTRA ACTION CHUNK"
        draw.text((446, 153), f"{label}  #{self.refresh_count}  |  5 actions, 0.25 s simulated time",
                  font=_font(20, True), fill=TEXT)
        draw.text((446, 186), "Normalized controls: dx, dy, dz, rx, ry, rz, gripper (+close / -open / 0 no increment)",
                  font=_font(16), fill=MUTED)
        if not self.chunk:
            draw.text((446, 244), "Awaiting first action chunk", font=_font(22), fill=MUTED)
        for index, action in enumerate(self.chunk):
            y = 221 + index * 32
            active = index == self.chunk_step and self.event != "action_chunk"
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
        draw.text((446, 393), history_line,
                  font=_font(14), fill=MUTED)

    def _trace_panel(self, draw, box, axes):
        left, top, right, bottom = box
        draw.rounded_rectangle(box, radius=12, fill=PANEL)
        names = "XYZ"
        draw.text((left + 16, top + 10), f"MEASURED {names[axes[0]]}{names[axes[1]]} PATH (m)",
                  font=_font(16, True), fill=MAGENTA)
        points = np.asarray(self.trace, dtype=float)
        if len(points) == 0:
            return
        values = points[:, axes]
        low, high = values.min(axis=0), values.max(axis=0)
        center = (low + high) / 2
        span = np.maximum(high - low, 0.04) * 1.15
        low, high = center - span / 2, center + span / 2
        positions = (values - low) / (high - low)
        pixels = [(left + 55 + float(p[0]) * (right - left - 90),
                   bottom - 35 - float(p[1]) * (bottom - top - 82)) for p in positions]
        if len(pixels) > 1:
            draw.line(pixels, fill=MAGENTA, width=3)
        x, y = pixels[-1]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=TEXT)
        draw.text((left + 16, bottom - 24),
                  f"{names[axes[0]]}: {low[0]:+.3f} .. {high[0]:+.3f}    {names[axes[1]]}: {low[1]:+.3f} .. {high[1]:+.3f}",
                  font=_font(13), fill=MUTED)

    def _event_strip_direct(self, draw):
        labels = ("1 OBSERVE", "2 GENERATE ACTIONS", "3 ISSUE ACTION", "4 EXECUTE / FEEDBACK")
        active = {"observation": 0, "action_chunk": 1, "direct_action": 2, "executed": 3}.get(self.event)
        for index, label in enumerate(labels):
            x = 425 + index * 291
            draw.rounded_rectangle((x, 668, x + 277, 711), radius=8,
                                   fill=(32, 91, 90) if index == active else PANEL,
                                   outline=GREEN if index == active else (44, 56, 74), width=2)
            draw.text((x + 10, 682), label, font=_font(14, True), fill=TEXT if index == active else MUTED)

    def _feedback_panel(self, draw):
        draw.rounded_rectangle((425, 729, 1576, 916), radius=12, fill=PANEL)
        label = self.status if self.event == "terminal" else self.event
        draw.text((442, 741), f"STEP {self.step:04d}  |  {label.upper().replace('_', ' ')}",
                  font=_font(20, True), fill=GREEN if self.success else CYAN)
        draw.text((442, 777), "EEF world m: " + _vector(self.observation.state[:3], 4), font=_font(16), fill=TEXT)
        draw.text((1025, 777), "Finger joints: " + _vector(self.observation.state[6:], 4), font=_font(16), fill=TEXT)
        requested = np.asarray(self.action[:3]) * 0.05 if self.action else None
        draw.text((442, 808), "Requested delta m: " + (_vector(requested, 4) if requested is not None else "not issued"),
                  font=_font(16), fill=YELLOW)
        draw.text((1025, 808), "Measured delta m: " + (_vector(self._measured_delta, 4) if self._measured_delta is not None else "awaiting execution"),
                  font=_font(15), fill=MAGENTA)
        draw.text((442, 839), "Feedback uses robot observations only. Task success is checked separately by LIBERO.",
                  font=_font(16), fill=MUTED)
        latency = f"Last request: {self.last_api_latency_s:.2f} s" if self.last_api_latency_s is not None else "No request completed"
        if self._error:
            latency = "Error: " + _short(self._error, 125)
        elif self.event == "terminal":
            latency = f"Evaluator status: {self.status} | {latency}"
        draw.text((442, 879), latency, font=_font(16), fill=RED if self._error else MUTED)
