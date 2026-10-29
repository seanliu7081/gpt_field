"""Annotated agent_SpeedUp_v1 playback: macro skills, primitives, outcomes and pointing.

The agent has no model action chunk, so this recorder never reads
``details["prediction"]``. It renders only recorded observations, the issued
action and the policy's own diagnostics (skill decision, primitive, last macro
outcome, pointing results, API latency). Camera panels show the LEGACY policy
images exactly as the simulator delivered them (left-right mirrored); the model
receives upright copies built in the client. Mark overlays are converted from
upright executor pixels to legacy pixels (u_legacy = W - 1 - u). Evaluator
success is displayed but never fed back to the policy.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw

from . import video as _video
from .direct_history_video import DirectHistoryVideoRecorder
from .observation import PolicyObservation, decode_image
from .policy import Decision
from .video import BG, CYAN, GREEN, HEIGHT, MAGENTA, MUTED, PANEL, RED, TEXT, WIDTH, YELLOW, _font, _short, _vector

AGENT_VIDEO_VERSION = "agent-speedup-v1-video-1"
_UV_KEYS = ("uv", "uv_upright", "pixel_uv", "mark_uv")
_VERBOSE_FEEDBACK = {"type", "step", "before_state", "after_state", "normalized_action", "rolling_progress",
                     "measured_translation_delta_m", "gripper_joint_delta"}


def json_safe(value):
    """Plain JSON value: numpy scalars/arrays converted, non-finite floats -> None."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return str(value)


def _uv(record):
    if not isinstance(record, dict):
        return None
    for key in _UV_KEYS:
        value = record.get(key)
        if (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)):
            return float(value[0]), float(value[1])
    return None


def _format_param(value):
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
        return _vector(value, 3)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pointing_text(result):
    if not isinstance(result, dict):
        return _short(result, 40)
    stage = result.get("stage", "?")
    parts = [str(stage)]
    for key, name in (("cell_id", "cell"), ("point_id", "point")):
        if key in result:
            parts.append(f"{name} {result[key] if result[key] is not None else 'null'}")
    uv = _uv(result)
    if uv is not None:
        parts.append(f"uv ({uv[0]:.0f}, {uv[1]:.0f})")
    if result.get("status") is not None:
        parts.append(str(result["status"]))
    return " ".join(parts)


class _SessionMP4Stream(_video.MP4Stream):
    """MP4Stream whose ffmpeg runs in its own session: a terminal Ctrl+C (SIGINT to the foreground process
    group) no longer kills the encoder mid-episode, so the runner can still finalize the video
    (first live run 2026-10-16 lost its video and was misclassified this way). If the runner dies, ffmpeg
    reads EOF on its stdin and still finalizes the partial file.

    ``__init__`` is MP4Stream.__init__ (gpt_field/video.py) with ``start_new_session=True`` on the Popen;
    write/close are inherited unchanged. No global (subprocess.Popen, video.MP4Stream) is patched."""

    def __init__(self, path, *, fps=20):
        if type(fps) is not int or fps < 1:
            raise ValueError("video fps must be a positive integer")
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise RuntimeError("ffmpeg is required for annotated MP4 recording")
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("episode video path must end in .mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(".partial.mp4")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(f"refusing to overwrite video: {self.path}")
        self.fps, self.frame_count, self.closed = fps, 0, False
        self._errors = tempfile.TemporaryFile(mode="w+b")
        self._process = subprocess.Popen([
            encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{WIDTH}x{HEIGHT}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-threads", "1", "-movflags", "+faststart", str(self.partial),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._errors, start_new_session=True)


class AgentSpeedUpV1VideoRecorder(DirectHistoryVideoRecorder):
    """One frame sequence per executed step: skill decision or primitive action, then execution."""

    writer_class = _SessionMP4Stream

    def __init__(self, path, *, episode_id="agent_speedup_v1", fps=20, offline_smoke=False, grid=None):
        self.controller = None
        self.routing = None
        self.macro = None
        self.primitive = None
        self.execution_kind = None
        self.macro_step = None
        self.last_outcome = None
        self.pointing = []
        self.pointing_count = 0
        self.api_calls_in_act = []
        self.history_stats = None
        # DirectHistoryVideoRecorder.__init__ state.
        self.offline_smoke = bool(offline_smoke)
        self.chunk = []
        self.chunk_step = 0
        self.feedback = None
        # EpisodeVideoRecorder.__init__ state, in the same order, except that self.writer is the session
        # encoder (the base class builds video.MP4Stream by module-global name, so it cannot be called here
        # without patching that global). tests/test_agent_speedup_v1_runner.py checks attribute parity.
        self.path = Path(path)
        self.grid = None
        self.episode_id = episode_id
        self.events_path = self.path.with_suffix(".events.jsonl")
        if self.events_path.exists():
            raise FileExistsError(f"refusing to overwrite event index: {self.events_path}")
        self.writer = self.writer_class(self.path, fps=fps)
        self._events = self.events_path.open("x")
        self._started = time.monotonic()
        self.observation = None
        self.field = None
        self.details = {}
        self.action = ()
        self.trace = []
        self.refresh_count = 0
        self.field_step = None
        self.step = 0
        self.event = "observation"
        self.status = "running"
        self.success = False
        self.closed = False
        self.decision_latency_s = None
        self.last_api_latency_s = None
        self.last_api_request_key = None
        self.last_api_request_log = None
        self._previous_xyz = None
        self._measured_delta = None
        self._error = None

    def decision(self, observation: PolicyObservation, decision: Decision, *, decision_latency_s=None):
        details = json_safe(deepcopy(dict(decision.diagnostics)))
        self.observation = observation
        self.details = details
        self.action = tuple(float(value) for value in decision.action)
        self.step = int(details.get("step", self.step))
        self.chunk, self.chunk_step = [], 0
        self.decision_latency_s = decision_latency_s
        self._previous_xyz = tuple(observation.state[:3])
        self._measured_delta = None
        self.feedback = None
        self.history_stats = details.get("history_stats")
        self.controller = details.get("controller")
        self.macro, self.primitive = details.get("macro"), details.get("primitive")
        self.execution_kind, self.macro_step = details.get("execution_kind"), details.get("macro_step")
        self.api_calls_in_act = list(details.get("api_calls_in_act") or [])
        if details.get("outcome") is not None:
            self.last_outcome = details["outcome"]
        pointing = list(details.get("pointing") or [])
        if pointing:
            self.pointing = pointing
            self.pointing_count += len(pointing)
        if self.api_calls_in_act or details.get("api_latency_s") is not None:
            self.last_api_latency_s = details.get("api_latency_s")
            self.last_api_request_key = details.get("api_request_key")
            self.last_api_request_log = details.get("api_request_log")
        if details.get("refreshed"):
            self.refresh_count += 1
            self.routing = details.get("routing")
            self._emit("skill_decision", max(1, round(self.writer.fps * 0.6)))
        else:
            self._emit("agent_action", 4 if pointing else 2)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None, history_stats=None):
        if history_stats is not None:
            self.history_stats = json_safe(deepcopy(history_stats))
        super().executed(observation, step=step, success=success,
                         sim_status=json_safe(dict(sim_status or {})), feedback=json_safe(deepcopy(feedback)))

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
            "policy": "agent_speedup_v1", "video_version": AGENT_VIDEO_VERSION,
            "offline_smoke": self.offline_smoke, "controller": self.controller,
            "skill_decision_count": self.refresh_count, "pointing_count": self.pointing_count,
            "macro": self.macro, "primitive": self.primitive, "execution_kind": self.execution_kind,
            "macro_step": self.macro_step, "status": self.status, "success": self.success,
            "decision_latency_s": self.decision_latency_s,
            "api_latency_s": self.last_api_latency_s,
            "api_request_key": self.last_api_request_key,
            "api_request_log": self.last_api_request_log,
            "api_calls_in_act": self.api_calls_in_act,
            "state": list(self.observation.state) if self.observation else None,
            "action": list(self.action),
            "measured_delta_xyz_m": list(self._measured_delta) if self._measured_delta is not None else None,
            "history_stats": self.history_stats, "feedback": self.feedback, **extra,
        }
        if event == "skill_decision":
            record["routing"] = self.routing
        if event in ("skill_decision", "agent_action"):
            record["outcome"] = self.details.get("outcome")
            record["pointing"] = self.details.get("pointing") or []
        self._events.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
        self._events.flush()

    # rendering ------------------------------------------------------------
    def render_frame(self):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        title = ("SCRIPTED SMOKE / LIBERO  |  AGENT SPEEDUP V1  |  NO LIVE ASTRA CALLS" if self.offline_smoke
                 else "ASTRA / LIBERO  |  AGENT SPEEDUP V1")
        draw.text((24, 17), title, font=_font(26, True), fill=YELLOW if self.offline_smoke else TEXT)
        draw.text((24, 57), _short(self.episode_id, 110), font=_font(18), fill=MUTED)
        if self.observation is None:
            draw.text((24, 125), "No observation recorded", font=_font(25), fill=MUTED)
            return canvas
        draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
        self._cameras(canvas, draw)
        self._chunk_panel(draw)
        self._trace_panel(draw, (425, 443, 974, 651), (0, 1))
        self._trace_panel(draw, (994, 443, 1576, 651), (0, 2))
        self._event_strip_direct(draw)
        self._feedback_panel(draw)
        footer = "Integration diagnostic only; scripted decisions are not model performance. " if self.offline_smoke else ""
        footer += ("Annotated playback; 20 Hz simulator; API waits omitted. Cameras: legacy (mirrored) policy images; "
                   "the model receives upright copies.")
        draw.text((24, 932), footer, font=_font(14), fill=MUTED)
        return canvas

    def _mark_pixels(self):
        marks = [(uv, YELLOW) for uv in map(_uv, self.pointing) if uv is not None]
        evidence = (self.last_outcome or {}).get("evidence") if isinstance(self.last_outcome, dict) else None
        for source in (self.details, evidence if isinstance(evidence, dict) else {}):
            for key, color in (("mark_uv", CYAN), ("setpoint_uv", GREEN)):
                uv = _uv({"uv": source.get(key)})
                if uv is not None:
                    marks.append((uv, color))
        return marks

    def _cameras(self, canvas, draw):
        for y, label, url, wrist in ((139, "AGENT VIEW", self.observation.image, False),
                                     (535, "WRIST VIEW", self.observation.wrist_image, True)):
            draw.rounded_rectangle((24, y, 401, y + 381), radius=12, fill=PANEL)
            image = decode_image(url)
            width, height = image.size
            draw.text((40, y + 6), label, font=_font(16, True), fill=TEXT)
            draw.text((40, y + 26), f"policy input (legacy orientation)  |  RGB {width} x {height}",
                      font=_font(12), fill=MUTED)
            ratio = min(352 / width, 336 / height)
            shown = image.resize((round(width * ratio), round(height * ratio)),
                                 Image.Resampling.LANCZOS if ratio < 1 else Image.Resampling.NEAREST)
            x0, y0 = 36 + (352 - shown.width) // 2, y + 42 + (336 - shown.height) // 2
            canvas.paste(shown, (x0, y0))
            if not wrist:
                continue
            for (u, v), color in self._mark_pixels():
                if not (0 <= u < width and 0 <= v < height):
                    continue
                px, py = x0 + (width - 1 - u + .5) * ratio, y0 + (v + .5) * ratio
                draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=color, width=2)
                draw.line((px - 11, py, px + 11, py), fill=color, width=1)
                draw.line((px, py - 11, px, py + 11), fill=color, width=1)

    def _chunk_panel(self, draw):
        draw.rounded_rectangle((425, 139, 1576, 427), radius=12, fill=PANEL)
        label = "SCRIPTED SKILL DECISION" if self.offline_smoke else "ASTRA SKILL DECISION"
        draw.text((446, 151), f"{label}  #{self.refresh_count}  |  macro: {self.macro or 'none'}",
                  font=_font(20, True), fill=TEXT)
        routing = self.routing if isinstance(self.routing, dict) else None
        if routing is None:
            draw.text((446, 186), "Awaiting first skill decision (Calibrate runs locally without model calls)",
                      font=_font(17), fill=MUTED)
        else:
            params = routing.get("params") if isinstance(routing.get("params"), dict) else {}
            text = ", ".join(f"{key}={_format_param(value)}" for key, value in params.items() if value is not None)
            draw.text((446, 184), _short(f"Skill: {routing.get('skill')}   |   {text or 'no parameters'}", 118),
                      font=_font(17, True), fill=YELLOW)
            lines = textwrap.wrap("Situation: " + str(routing.get("situation") or ""), 138)[:2]
            for index, line in enumerate(lines):
                draw.text((446, 212 + 22 * index), line, font=_font(15), fill=TEXT)
        executor = f"Executor: primitive {self.primitive or '-'}  |  {self.execution_kind or '-'}"
        if self.macro_step is not None:
            executor += f"  |  macro step {self.macro_step}"
        draw.text((446, 262), _short(executor, 118), font=_font(16), fill=CYAN)
        outcome = self.last_outcome if isinstance(self.last_outcome, dict) else None
        if outcome is None:
            draw.text((446, 290), "Last outcome: none yet", font=_font(16), fill=MUTED)
        else:
            status = str(outcome.get("status", "?"))
            name = outcome.get("skill") or outcome.get("macro") or outcome.get("primitive") or "?"
            color = GREEN if status == "succeeded" else RED if status == "failed" else TEXT
            draw.text((446, 290), _short(f"Last outcome: {name} {status.upper()} ({outcome.get('reason', '-')})", 118),
                      font=_font(16, True), fill=color)
        pointing = " -> ".join(_pointing_text(result) for result in self.pointing) or "none yet"
        draw.text((446, 318), _short(f"Pointing (upright px): {pointing}   |   total {self.pointing_count}", 130),
                  font=_font(15), fill=TEXT)
        latency = f"{self.last_api_latency_s:.2f} s" if isinstance(self.last_api_latency_s, (int, float)) else "-"
        calls = ", ".join(map(str, self.api_calls_in_act)) or "none"
        history = self.history_stats if isinstance(self.history_stats, dict) else {}
        kinds = history.get("calls_by_kind") if isinstance(history.get("calls_by_kind"), dict) else {}
        totals = ", ".join(f"{str(key).replace('pointing_', '')} {value}" for key, value in kinds.items()) or "-"
        draw.text((446, 346), _short(f"API calls this act: {calls}  |  last latency {latency}  |  totals: {totals}", 130),
                  font=_font(15), fill=TEXT)
        draw.text((446, 393), _short(f"Requests: {history.get('protocol', 'stateless')} | "
                                     f"{history.get('accepted_requests', 0)} accepted | "
                                     f"{history.get('messages', 0)} retained messages | "
                                     f"{history.get('retained_image_pairs', 0)} retained image pairs", 140),
                  font=_font(14), fill=MUTED)

    def _event_strip_direct(self, draw):
        labels = ("1 OBSERVE", "2 SKILL DECISION", "3 PRIMITIVE ACTION", "4 EXECUTE / FEEDBACK")
        active = {"observation": 0, "skill_decision": 1, "agent_action": 2, "executed": 3}.get(self.event)
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
        state = self.observation.state
        draw.text((442, 777), "EEF world m: " + _vector(state[:3], 4), font=_font(16), fill=TEXT)
        draw.text((1025, 777), f"Finger gap: {(state[6] - state[7]) * 1000:.2f} mm", font=_font(16), fill=TEXT)
        if self.action:
            grip = "close" if self.action[6] > 0 else "open" if self.action[6] < 0 else "hold"
            requested = "Requested delta m: " + _vector(np.asarray(self.action[:3]) * 0.05, 4) + f"  |  gripper {grip}"
        else:
            requested = "Requested delta m: not issued"
        draw.text((442, 808), requested, font=_font(16), fill=YELLOW)
        draw.text((1025, 808), "Measured delta m: " + (_vector(self._measured_delta, 4) if self._measured_delta is not None
                                                        else "awaiting execution"), font=_font(15), fill=MAGENTA)
        if isinstance(self.feedback, dict) and self.feedback:
            shown = {key: value for key, value in self.feedback.items() if key not in _VERBOSE_FEEDBACK}
            text = "Executor feedback: " + json.dumps(shown, separators=(",", ":"), allow_nan=False)
        else:
            text = "Feedback uses robot observations only. Task success is checked separately by LIBERO."
        draw.text((442, 839), _short(text, 140), font=_font(15), fill=MUTED)
        latency = (f"Last request: {self.last_api_latency_s:.2f} s" if isinstance(self.last_api_latency_s, (int, float))
                   else "No request completed")
        if self._error:
            latency = "Error: " + _short(self._error, 125)
        elif self.event == "terminal":
            latency = f"Evaluator status: {self.status} | {latency}"
        draw.text((442, 879), latency, font=_font(16), fill=RED if self._error else MUTED)
