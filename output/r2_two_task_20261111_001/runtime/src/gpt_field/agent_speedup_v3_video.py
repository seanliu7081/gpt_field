"""Compact v3 playback of the existing public RGB; no additional simulator render."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

from .observation import decode_image
from .video import MP4Stream

VIDEO_VERSION = "agent-speedup-v3-compact-video-6"
CANVAS = (640, 384)


class CompactMP4Stream(MP4Stream):
    """Use the existing bounded close/drain/error handling with a small canvas."""
    def __init__(self, path, *, fps=20):
        if type(fps) is not int or fps < 1:
            raise ValueError("video fps must be positive")
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise RuntimeError("ffmpeg is required for v3 playback")
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("video path must end in .mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_suffix(".partial.mp4")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self.fps, self.frame_count, self.closed = fps, 0, False
        self._errors = tempfile.TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen([
                encoder, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{CANVAS[0]}x{CANVAS[1]}",
                "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-threads", "1",
                "-movflags", "+faststart", str(self.partial)], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self._errors, start_new_session=True)
        except BaseException:
            self._errors.close()
            raise

    def write(self, frame, count=1):
        if self.closed:
            raise RuntimeError("cannot write a closed video")
        if frame.mode != "RGB" or frame.size != CANVAS or type(count) is not int or count < 1:
            raise ValueError("invalid compact video frame")
        data = frame.tobytes()
        try:
            for _ in range(count):
                self._process.stdin.write(data)
                self.frame_count += 1
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("ffmpeg stopped during compact video recording") from exc


class AgentSpeedUpV3VideoRecorder:
    writer_class = CompactMP4Stream

    def __init__(self, path, *, episode_id="agent_speedup_v3", fps=20, offline_smoke=False,
                 grid=None, clock=time.monotonic):
        self.path, self.episode_id = Path(path), episode_id
        self.events_path = self.path.with_suffix(".events.jsonl")
        if self.events_path.exists():
            raise FileExistsError(f"refusing to overwrite {self.events_path}")
        self.writer = self.writer_class(path, fps=fps)
        try:
            self._events = self.events_path.open("x")
        except BaseException:
            self.writer.close()
            raise
        self._clock, self._started = clock, clock()
        self.closed = False
        self.observation = None
        self.step, self.model_calls = 0, 0
        self.status, self.phase = "running", "initial"
        self.action, self.intent = [], None
        self.decision_latency_s = None
        self._images = {}
        self._font = ImageFont.load_default()
        self.timings = {"render_s": 0.0, "encode_write_s": 0.0, "event_write_s": 0.0,
                        "close_drain_s": 0.0, "rendered_frames": 0, "decision_events": 0,
                        "executed_events": 0}

    @property
    def frame_count(self):
        return self.writer.frame_count

    def _event(self, event, **values):
        started = self._clock()
        row = {"event": event, "step": self.step, "frame_count": self.frame_count,
               "wall_elapsed_s": self._clock()-self._started, "video_version": VIDEO_VERSION,
               "full_action_feedback_contact_audit": "trace.jsonl", **values}
        self._events.write(json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n")
        self._events.flush()
        self.timings["event_write_s"] += self._clock()-started

    def _panel(self, camera, url):
        old = self._images.get(camera)
        if old is not None and old[0] is url:
            return old[1]
        # Decode the already returned RGB, never ask the simulator for a frame.
        image = decode_image(url).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image.thumbnail((312, 264), Image.Resampling.BILINEAR)
        self._images[camera] = (url, image)
        return image

    def render_frame(self):
        canvas = Image.new("RGB", CANVAS, (16, 21, 29))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"AGENT SPEEDUP V3 | step {self.step} | {self.status} | calls {self.model_calls}",
                  font=self._font, fill="white")
        draw.text((8, 25), f"{self.phase} | intent {self.intent or '-'}"[:100], font=self._font, fill="white")
        if self.observation is not None:
            for x, camera, url in ((4, "external", self.observation.image),
                                   (324, "wrist", self.observation.wrist_image)):
                image = self._panel(camera, url)
                canvas.paste(image, (x+(312-image.width)//2, 56+(264-image.height)//2))
                draw.text((x+4, 43), camera, font=self._font, fill="white")
            draw.text((8, 326), str(self.observation.prompt)[:100], font=self._font, fill="white")
        draw.text((8, 343), "action " + " ".join(f"{x:.3f}" for x in self.action),
                  font=self._font, fill="white")
        draw.text((8, 363), "20 Hz action playback; model waits omitted; full evidence in trace.jsonl",
                  font=self._font, fill=(170, 185, 200))
        return canvas

    def _frame(self):
        started = self._clock()
        frame = self.render_frame()
        self.timings["render_s"] += self._clock()-started
        started = self._clock()
        self.writer.write(frame, 1)
        self.timings["encode_write_s"] += self._clock()-started
        self.timings["rendered_frames"] += 1

    def observe(self, observation, *, step=0):
        self.observation, self.step = observation, step
        self._frame()
        self._event("observation")

    def decision(self, observation, decision, *, decision_latency_s=None):
        details = decision.diagnostics
        self.observation = observation
        self.step = int(details.get("step", self.step))
        self.action = [float(value) for value in decision.action]
        self.phase = str(details.get("primitive") or details.get("macro") or "action")
        params = (details.get("routing") or {}).get("params") or {}
        self.intent = (params.get("motion_context") or {}).get("intent")
        self.model_calls += len(details.get("api_calls") or [])
        self.decision_latency_s = decision_latency_s
        self.timings["decision_events"] += 1
        # Every decision is indexed, while the following executed frame supplies
        # its RGB. The complete diagnostics are already in the unmodified trace.
        self._event("decision", action=self.action, phase=self.phase, intent=self.intent,
                    decision_latency_s=decision_latency_s,
                    model_request_key=details.get("api_request_key"),
                    model_calls=details.get("api_calls") or [], frame_rendered=False)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None, history_stats=None):
        self.observation, self.step = observation, step
        self.status = "success" if success else "running"
        self.timings["executed_events"] += 1
        self._frame()
        self._event("executed", success=bool(success), action=self.action)

    def close(self, *, status, error=None):
        if self.closed:
            return
        started = self._clock()
        self.status = status
        try:
            if self.observation is not None:
                self._frame()
            self._event("terminal", status=status, error=error)
        finally:
            self.closed = True
            self._events.close()
            try:
                self.writer.close()  # Synchronous drain, including process wait.
            finally:
                self.timings["close_drain_s"] = self._clock()-started
                self.path.with_suffix(".timing.json").write_text(json.dumps({
                    "version": VIDEO_VERSION, **self.timings, "frame_count": self.frame_count,
                    "canvas": list(CANVAS), "encoding": "synchronous; drained before returning",
                    "close_drain_contains_terminal_render_and_write": True,
                    "no_additional_simulator_render": True}, indent=2)+"\n")
