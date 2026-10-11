"""Stream truthful, annotated observation/field/plan/execution MP4s.

Playback is an event summary, not wall-clock simulation or API waiting footage.
Only supplied policy observations, recorded predictions and planner candidates are
rendered. Evaluator success is displayed but never fed back to the policy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .field import GridField
from .observation import PolicyObservation, decode_image
from .policy import Decision
from .schema import FieldPrediction, GridSpec


WIDTH, HEIGHT = 1600, 960
BG = (12, 18, 30)
PANEL = (22, 31, 47)
TEXT = (232, 239, 250)
MUTED = (154, 173, 197)
CYAN = (59, 201, 236)
YELLOW = (255, 220, 84)
MAGENTA = (239, 123, 216)
GREEN = (102, 227, 161)
RED = (250, 129, 125)


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _vector(values, precision=3):
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in values) + "]"


def _short(text: Any, limit: int):
    value = str(text)
    return value if len(value) <= limit else value[:limit - 3] + "..."


def project_world(points, lower, upper, axes, side: int) -> np.ndarray:
    """Project world coordinates onto an image plane, without changing units.

    Positive vertical world coordinates point upward. Heatmap grid vertices are
    drawn at the same pixel coordinates as paths, including the domain edges.
    """
    points = np.asarray(points, dtype=float)
    low, high = np.asarray(lower), np.asarray(upper)
    horizontal = (points[..., axes[0]] - low[axes[0]]) / (high[axes[0]] - low[axes[0]])
    vertical = (points[..., axes[1]] - low[axes[1]]) / (high[axes[1]] - low[axes[1]])
    return np.stack((horizontal * (side - 1), (1 - vertical) * (side - 1)), axis=-1)


class MP4Stream:
    """Bounded-memory rawvideo encoder; finalized files replace partial files."""

    def __init__(self, path: str | Path, *, fps: int = 20):
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
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._errors)

    def write(self, frame: Image.Image, count: int = 1):
        if self.closed:
            raise RuntimeError("cannot write a closed video")
        if frame.mode != "RGB" or frame.size != (WIDTH, HEIGHT):
            raise ValueError("video frames must be RGB at the recorder canvas size")
        if type(count) is not int or count < 1:
            raise ValueError("frame count must be positive")
        data = frame.tobytes()
        try:
            for _ in range(count):
                self._process.stdin.write(data)
                self.frame_count += 1
        except (BrokenPipeError, OSError) as exc:
            self._process.wait(timeout=30)
            self._errors.seek(0)
            message = self._errors.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg stopped while recording: {message}") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self._process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
                raise RuntimeError("ffmpeg did not finish the episode video")
            self._errors.seek(0)
            message = self._errors.read().decode(errors="replace")
            if code != 0:
                raise RuntimeError(f"ffmpeg failed to finalize episode video: {message}")
            if self.frame_count:
                self.partial.replace(self.path)
            elif self.partial.exists():
                self.partial.unlink()
        finally:
            self._errors.close()


class EpisodeVideoRecorder:
    """Receive actual episode events and write annotated video plus event index.

    Call ``observe`` after reset, ``decision`` after policy.act, ``executed``
    after sim.step, and ``close`` with the evaluator's terminal status. Latency
    passed as decision_latency_s includes planning; only diagnostics explicitly
    named api_latency_s is labeled API latency.
    """

    def __init__(self, path: str | Path, *, grid: GridSpec | None = None,
                 episode_id: str = "episode", fps: int = 20):
        self.path = Path(path)
        self.grid = grid
        self.episode_id = episode_id
        self.events_path = self.path.with_suffix(".events.jsonl")
        if self.events_path.exists():
            raise FileExistsError(f"refusing to overwrite event index: {self.events_path}")
        self.writer = MP4Stream(self.path, fps=fps)
        self._events = self.events_path.open("x")
        self._started = time.monotonic()
        self.observation: PolicyObservation | None = None
        self.field: GridField | None = None
        self.details: dict = {}
        self.action: tuple = ()
        self.trace: list[tuple] = []
        self.refresh_count = 0
        self.field_step: int | None = None
        self.step = 0
        self.event = "observation"
        self.status = "running"
        self.success = False
        self.closed = False
        self.decision_latency_s: float | None = None
        self.last_api_latency_s: float | None = None
        self.last_api_request_key: str | None = None
        self.last_api_request_log: str | None = None
        self._previous_xyz: tuple | None = None
        self._measured_delta: tuple | None = None
        self._error: str | None = None

    @property
    def frame_count(self):
        return self.writer.frame_count

    def observe(self, observation: PolicyObservation, *, step: int = 0):
        self.observation, self.step = observation, step
        self.trace.append(tuple(observation.state[:3]))
        self._emit("observation", max(1, round(self.writer.fps * 0.6)))

    def decision(self, observation: PolicyObservation, decision: Decision, *,
                 decision_latency_s: float | None = None):
        self.observation = observation
        self.details = dict(decision.diagnostics)
        self.action = tuple(decision.action)
        self.step = int(self.details.get("step", self.step))
        self.decision_latency_s = decision_latency_s
        self._previous_xyz = tuple(observation.state[:3])
        self._measured_delta = None
        fresh = bool(self.details.get("refreshed"))
        if fresh:
            self.refresh_count += 1
            self.field_step = self.step
            raw = self.details.get("field_prediction")
            if raw is not None and self.grid is not None:
                self.field = GridField(self.grid, self.details["field_center"],
                                       FieldPrediction.from_dict(raw, self.grid))
            else:
                # Do not carry a stale field across a response with no field.
                self.field = None
            self.last_api_latency_s = self.details.get("api_latency_s")
            self.last_api_request_key = self.details.get("api_request_key")
            self.last_api_request_log = self.details.get("api_request_log")
        if fresh:
            # The field arrives before search. Show that real intermediate
            # product separately, with no selected route or issued action.
            complete_details, complete_action = self.details, self.action
            self.details = {key: value for key, value in complete_details.items() if key != "plan"}
            self.details["status"] = "field_received"
            self.action = ()
            try:
                self._emit("field_refresh", max(1, round(self.writer.fps * 0.6)))
            finally:
                self.details, self.action = complete_details, complete_action
        self._emit("local_replan", max(2, round(self.writer.fps * 0.15)) if fresh else 2)

    def executed(self, observation: PolicyObservation, *, step: int,
                 success: bool, sim_status: dict | None = None):
        self.observation, self.step, self.success = observation, step, bool(success)
        self.trace.append(tuple(observation.state[:3]))
        if self._previous_xyz is not None:
            self._measured_delta = tuple(np.asarray(observation.state[:3]) - self._previous_xyz)
        self._emit("executed", 2, sim_status=sim_status or {})

    def close(self, *, status: str, error: str | None = None):
        if self.closed:
            return
        self.status, self._error = status, error
        self.success = status == "success"
        try:
            if self.observation is not None:
                self._emit("terminal", max(1, round(self.writer.fps * 3.0)))
        finally:
            self.closed = True
            self._events.close()
            self.writer.close()

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
            "field_refresh_count": self.refresh_count, "field_step": self.field_step,
            "status": self.status, "success": self.success,
            "decision_latency_s": self.decision_latency_s,
            "api_latency_s": self.last_api_latency_s,
            "api_request_key": self.last_api_request_key,
            "api_request_log": self.last_api_request_log,
            "plan_latency_s": self.details.get("plan_latency_s"),
            "stage": self.details.get("stage"),
            "selected_index": self.details.get("plan", {}).get("selected_index"),
            "state": list(self.observation.state) if self.observation else None,
            "action": list(self.action), **extra,
        }
        self._events.write(json.dumps(record, allow_nan=False) + "\n")
        self._events.flush()

    def render_frame(self) -> Image.Image:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 17), "ASTRA / LIBERO  ·  CLOSED-LOOP FIELD CONTROL", font=_font(27, True), fill=TEXT)
        draw.text((24, 57), _short(self.episode_id, 110), font=_font(18), fill=MUTED)
        if self.observation is None:
            draw.text((24, 125), "No observation recorded", font=_font(25), fill=MUTED)
            return canvas
        draw.text((24, 94), _short(self.observation.prompt, 135), font=_font(21), fill=TEXT)
        self._cameras(canvas, draw)
        self._field_panels(canvas, draw)
        self._event_strip(draw)
        self._details_panel(draw)
        draw.text((24, 932), "Slowed annotated playback; simulator 20 Hz. API waits omitted (measured latency shown). Each execution event is one real step.",
                  font=_font(15), fill=MUTED)
        return canvas

    def _cameras(self, canvas, draw):
        for y, label, url in ((139, "AGENT VIEW", self.observation.image),
                              (535, "WRIST VIEW", self.observation.wrist_image)):
            draw.rounded_rectangle((24, y, 401, y + 381), radius=12, fill=PANEL)
            image = decode_image(url)
            draw.text((40, y + 12), label, font=_font(17, True), fill=TEXT)
            draw.text((252, y + 13), f"RGB {image.width} x {image.height}", font=_font(14), fill=MUTED)
            image.thumbnail((352, 336), Image.Resampling.LANCZOS)
            # Upscale only for presentation, never substitute a new image.
            ratio = min(352 / image.width, 336 / image.height)
            image = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.NEAREST)
            canvas.paste(image, (36 + (352 - image.width) // 2, y + 42 + (336 - image.height) // 2))

    def _field_panels(self, canvas, draw):
        stage = self.details.get("stage")
        if stage is None and self.field is not None:
            stage = self.field.prediction.stage
        draw.text((425, 142), "RECORDED FIELD + ACTUAL PLANNER SEARCH", font=_font(20, True), fill=TEXT)
        draw.text((425, 174), "Heatmaps: nearest grid planes. All path overlays: world-coordinate projections.",
                  font=_font(16), fill=MUTED)
        if self.field is None:
            draw.rounded_rectangle((425, 206, 1576, 600), radius=12, fill=PANEL)
            draw.text((450, 245), "No model field recorded for this event", font=_font(26), fill=MUTED)
            draw.text((450, 290), "Field panels appear only after a real field prediction is returned.", font=_font(19), fill=MUTED)
            return
        field = self.field
        values = np.asarray(field.prediction.values).reshape((field.grid.size,) * 3)
        valid = np.ones(field.grid.count, dtype=bool)
        valid[list(field.prediction.blocked_indices + field.prediction.unknown_indices)] = False
        samples = values.ravel()[valid]
        minimum, maximum = (float(samples.min()), float(samples.max())) if samples.size else (0.0, 0.0)
        for panel_x, axes, fixed in ((425, (0, 1), 2), (816, (0, 2), 1), (1207, (1, 2), 0)):
            self._slice(canvas, draw, panel_x, axes, fixed, values, minimum, maximum)
        draw.text((425, 565), f"Stage: {_short(stage, 40)}   |   field #{self.refresh_count} from step {self.field_step}   |   age {self.step - self.field_step} steps",
                  font=_font(17, True), fill=GREEN if self.event == "field_refresh" else TEXT)
        draw.text((425, 592), f"Fixed field center (world m): {_vector(field.center)}   |   grid {field.grid.size} cubed, {field.grid.spacing_m:g} m spacing",
                  font=_font(16), fill=MUTED)
        draw.text((425, 619), f"Potential: teal {minimum:.4g} (lower) to orange {maximum:.4g}; arbitrary units. Predicted masks: purple=blocked, gray=unknown.",
                  font=_font(16), fill=MUTED)

    def _slice(self, canvas, draw, panel_x, axes, fixed, values, minimum, maximum):
        field = self.field
        names = "xyz"
        side = 288
        plot_x, plot_y = panel_x + 54, 245
        draw.rounded_rectangle((panel_x, 207, panel_x + 369, 553), radius=10, fill=PANEL)
        coordinate = self.observation.state[fixed]
        plane = int(np.clip(round((coordinate - field.lower[fixed]) / field.grid.spacing_m), 0, field.grid.size - 1))
        fixed_world = field.lower[fixed] + plane * field.grid.spacing_m
        draw.text((panel_x + 12, 216), f"{names[axes[0]].upper()}{names[axes[1]].upper()} at {names[fixed]}={fixed_world:+.3f} m",
                  font=_font(16, True), fill=TEXT)
        plot = Image.new("RGB", (side, side), PANEL)
        pd = ImageDraw.Draw(plot)
        size = field.grid.size
        blocked, unknown = set(field.prediction.blocked_indices), set(field.prediction.unknown_indices)
        for a in range(size):
            for b in range(size):
                index = [0, 0, 0]
                index[axes[0]], index[axes[1]], index[fixed] = a, b, plane
                flat = index[0] * size * size + index[1] * size + index[2]
                value = values[tuple(index)]
                normalized = 0.5 if maximum == minimum else float(np.clip((value - minimum) / (maximum - minimum), 0, 1))
                low, high = np.array((28, 135, 137)), np.array((227, 124, 63))
                color = tuple((low * (1 - normalized) + high * normalized).astype(int))
                if flat in blocked:
                    color = (83, 46, 77)
                elif flat in unknown:
                    color = (78, 87, 104)
                px, py = a * (side - 1) / (size - 1), (size - 1 - b) * (side - 1) / (size - 1)
                half = (side - 1) / (size - 1) / 2
                rectangle = (round(px - half), round(py - half), round(px + half), round(py + half))
                pd.rectangle(rectangle, fill=color, outline=(43, 59, 71))
                if flat in blocked or flat in unknown:
                    radius = max(2, int(half * 0.45))
                    pd.line((px - radius, py - radius, px + radius, py + radius), fill=MUTED, width=1)
                    pd.line((px + radius, py - radius, px - radius, py + radius), fill=MUTED, width=1)
        plan = self.details.get("plan", {})
        overlay = Image.new("RGBA", (side, side))
        od = ImageDraw.Draw(overlay)
        candidates = plan.get("candidates", [])
        for candidate in candidates:
            if not candidate.get("valid"):
                continue
            pixels = project_world(candidate["positions"], field.lower, field.upper, axes, side)
            od.line([tuple(point) for point in pixels], fill=(*CYAN, 85), width=1)
        plot = Image.alpha_composite(plot.convert("RGBA"), overlay).convert("RGB")
        pd = ImageDraw.Draw(plot)
        if len(self.trace) > 1:
            pixels = project_world(self.trace, field.lower, field.upper, axes, side)
            pd.line([tuple(point) for point in pixels], fill=MAGENTA, width=2)
        selected = plan.get("selected_index")
        if selected is not None and 0 <= selected < len(candidates):
            points = candidates[selected]["positions"]
            pixels = project_world(points, field.lower, field.upper, axes, side)
            pd.line([tuple(point) for point in pixels], fill=(22, 26, 32), width=8)
            pd.line([tuple(point) for point in pixels], fill=YELLOW, width=4)
            for px, py in pixels[1:]:
                pd.ellipse((px - 3, py - 3, px + 3, py + 3), fill=YELLOW)
            if len(pixels) > 1:
                px, py = pixels[1]
                pd.rectangle((px - 5, py - 5, px + 5, py + 5), outline=TEXT, width=2)
        px, py = project_world([self.observation.state[:3]], field.lower, field.upper, axes, side)[0]
        pd.ellipse((px - 6, py - 6, px + 6, py + 6), fill=TEXT, outline=BG, width=2)
        canvas.paste(plot, (plot_x, plot_y))
        for fraction in (0.0, 0.5, 1.0):
            xvalue = field.lower[axes[0]] + fraction * (field.upper[axes[0]] - field.lower[axes[0]])
            yvalue = field.lower[axes[1]] + fraction * (field.upper[axes[1]] - field.lower[axes[1]])
            tx = plot_x + fraction * (side - 1)
            ty = plot_y + (1 - fraction) * (side - 1)
            draw.text((tx - 22, plot_y + side + 3), f"{xvalue:+.2f}", font=_font(11), fill=MUTED)
            draw.text((panel_x + 4, ty - 6), f"{yvalue:+.2f}", font=_font(11), fill=MUTED)
        draw.text((plot_x + side // 2 + 34, plot_y + side + 3), names[axes[0]], font=_font(12, True), fill=TEXT)
        draw.text((panel_x + 14, plot_y + 16), names[axes[1]], font=_font(13, True), fill=TEXT)

    def _event_strip(self, draw):
        labels = ("1  OBSERVE", "2  ASTRA REFRESH", "3  LOCAL REPLAN", "4  EXECUTE / OBSERVE")
        active = {"observation": 0, "field_refresh": 1, "local_replan": 2, "executed": 3}.get(self.event)
        for index, label in enumerate(labels):
            x = 425 + index * 291
            draw.rounded_rectangle((x, 654, x + 277, 695), radius=8,
                                   fill=(32, 91, 90) if index == active else PANEL,
                                   outline=GREEN if index == active else (44, 56, 74), width=2)
            draw.text((x + 11, 666), label, font=_font(15, True), fill=TEXT if index == active else MUTED)
        for x, color, label in ((425, CYAN, "valid candidates"), (638, YELLOW, "selected horizon"),
                                (862, MAGENTA, "measured trace"), (1077, TEXT, "current EEF / square: first move")):
            draw.line((x, 719, x + 22, 719), fill=color, width=3)
            draw.text((x + 30, 709), label, font=_font(14), fill=MUTED)

    def _details_panel(self, draw):
        draw.rounded_rectangle((425, 742, 1576, 916), radius=12, fill=PANEL)
        event_label = self.event.upper().replace("_", " ")
        if self.event == "terminal":
            event_label = "TERMINAL: " + self.status.upper().replace("_", " ")
        color = GREEN if self.status == "success" else RED if self.event == "terminal" else CYAN
        draw.text((442, 754), f"STEP {self.step:04d}  |  {event_label}", font=_font(20, True), fill=color)
        plan = self.details.get("plan", {})
        draw.text((442, 787), f"EEF world m: {_vector(self.observation.state[:3])}", font=_font(16), fill=TEXT)
        draw.text((442, 815), f"Action normalized 7D: {_vector(self.action, 2) if self.action else 'not issued'}", font=_font(15), fill=TEXT)
        if self._measured_delta is not None:
            line = "Executed displacement m: " + _vector(self._measured_delta, 4)
        else:
            line = "First planned delta m: " + (_vector(plan["delta_xyz"], 4) if "delta_xyz" in plan else "not available")
        draw.text((442, 843), line, font=_font(15), fill=MAGENTA if self._measured_delta is not None else YELLOW)
        prediction = self.field.prediction if self.field else None
        draw.text((1030, 786), f"Search: {plan.get('valid_count', 0)} / {plan.get('candidate_count', 0)} valid; selected {plan.get('selected_index', '-')}", font=_font(15), fill=TEXT)
        draw.text((1030, 815), f"Planner: {_short(self.details.get('status', 'not yet run'), 48)}", font=_font(15), fill=TEXT)
        auxiliary = f"Gripper: {prediction.gripper_command}; motion: {prediction.motion_mode}" if prediction else "No auxiliary field commands recorded"
        draw.text((1030, 843), auxiliary, font=_font(15), fill=TEXT)
        latency = []
        if self.last_api_latency_s is not None:
            latency.append(f"last API {self.last_api_latency_s:.2f}s")
        if self.decision_latency_s is not None:
            latency.append(f"this policy decision {self.decision_latency_s:.3f}s")
        bottom = " | ".join(latency) or "No request latency recorded yet"
        if self._error:
            bottom = "Error: " + _short(self._error, 125)
        elif self.event == "terminal":
            bottom = f"Evaluator outcome: {self.status}. " + bottom
        draw.text((442, 879), bottom, font=_font(16), fill=RED if self._error else MUTED)
