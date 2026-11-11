"""Bounded asynchronous recording; encoding never gates robot dispatch.

Every queued item owns its RGB/geometry snapshot. The renderer never reads the
live policy. When it falls behind, intermediate sampled frames are dropped and
the newest scene is retained; the event index records real capture timestamps.
"""
from __future__ import annotations

from copy import deepcopy
import queue
import threading
import time

from PIL import ImageDraw

from ..foundation_harness.video import FoundationHarnessVideoRecorder
from ..agent_speedup_v1_video import BG, TEXT, MUTED, CYAN, _font


class _V1VideoRecorder(FoundationHarnessVideoRecorder):
    """Reuse camera/map drawing with accurate V1 ownership and timing labels."""

    def _chunk_panel(self, draw):
        super()._chunk_panel(draw)
        from ..agent_speedup_v1_video import PANEL
        draw.rectangle((440, 150, 1555, 183), fill=PANEL)
        draw.text((444, 155), "GROUNDED OPERATIONS / LOCAL GEOMETRIC KEYPOSES",
                  font=_font(19, True), fill=CYAN)

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 10, 1580, 49), fill=BG)
        draw.text((24, 17), "FOUNDATION VISION HARNESS V1 | GROUNDED OPERATIONS",
                  font=_font(25, True), fill=TEXT)
        draw.rectangle((20, 940, 1580, 961), fill=BG)
        draw.text((24, 943), "Sampled asynchronous playback; frames may be dropped. Capture timestamps are in the event index; online wall time includes recording overhead.",
                  font=_font(12), fill=MUTED)
        return canvas

    def _emit(self, event, repeats, **extra):
        # One output frame per captured event. The former recorder intentionally
        # repeated frames for narrative pacing, which is inappropriate here.
        super()._emit(event, 1, **{**extra, "capture": self.planning_debug.get("capture")})


class AsyncVideoRecorder:
    """Small latest-frame queue; close/join happens outside online timing.

    ``renderer_factory`` is an offline test seam. It must accept a path and a
    planning_provider and implement observe/decision/executed/close.
    """

    def __init__(self, path, *, episode_id="foundation_vision_harness_V1",
                 queue_size=3, renderer_factory=None, clock=time.monotonic):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        self.path, self.episode_id = path, episode_id
        self._factory = renderer_factory or _V1VideoRecorder
        self._clock = clock
        self._queue = queue.Queue(maxsize=queue_size)
        self._finish = threading.Event()
        self._capture = {}
        self._status, self._terminal_error = "running", None
        self._renderer = None
        self._submitted = self._rendered = self._dropped = 0
        self._worker_error = None
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="fvh-v1-video", daemon=True)
        self._worker.start()

    def submit(self, observation, *, step, planning, decision=None,
               decision_latency_s=None, event="executed"):
        """Capture now without blocking on image drawing or the video encoder."""
        if self._closed:
            raise RuntimeError("video recorder is closed")
        if self._worker_error is not None:
            return False
        capture = deepcopy(planning or {})
        capture["capture"] = {"step": int(step), "monotonic_s": self._clock(),
                              "event": str(event), "dropped_before": self._dropped}
        item = (deepcopy(observation), int(step), capture,
                deepcopy(decision), decision_latency_s, event)
        self._submitted += 1
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._dropped += 1
            except queue.Empty:  # Worker consumed it between put/get.
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                return False
        return True

    def _work(self):
        renderer = None
        try:
            renderer = self._factory(self.path, episode_id=self.episode_id, fps=10,
                                     planning_provider=lambda: self._capture)
            self._renderer = renderer
            while not self._finish.is_set() or not self._queue.empty():
                try:
                    observation, step, capture, decision, latency, event = self._queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    self._capture = capture
                    if decision is not None:
                        renderer.decision(observation, decision, decision_latency_s=latency)
                    elif event == "initial":
                        # Legacy observe does not refresh its planning provider.
                        renderer._refresh()
                        renderer.observe(observation, step=step)
                    else:
                        # A raw evaluator success never labels autonomous video
                        # completion. The terminal status is supplied at close.
                        renderer.executed(observation, step=step, success=False)
                    self._rendered += 1
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._worker_error = f"{type(error).__name__}: {error}"
        finally:
            if renderer is not None:
                try:
                    renderer.close(status=self._status, error=self._terminal_error or self._worker_error)
                except BaseException as error:
                    self._worker_error = f"{type(error).__name__}: {error}"

    def close(self, *, status, error=None):
        if not self._closed:
            self._closed = True
            self._status, self._terminal_error = status, error
            self._finish.set()
            self._worker.join()
        return self.summary()

    def summary(self):
        return {"path": str(self.path), "asynchronous": True,
                "sampled_events": self._submitted, "rendered_events": self._rendered,
                "dropped_events": self._dropped, "queue_capacity": self._queue.maxsize,
                "worker_alive": self._worker.is_alive(), "error": self._worker_error,
                "playback_is_wall_clock": False}
