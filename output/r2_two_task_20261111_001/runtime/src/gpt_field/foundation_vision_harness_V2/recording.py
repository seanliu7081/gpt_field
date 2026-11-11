"""V2 labels over the existing bounded asynchronous recording implementation."""
from PIL import ImageDraw

from ..foundation_vision_harness_V1.recording import (
    AsyncVideoRecorder as _AsyncRecorder, _V1VideoRecorder,
)
from ..agent_speedup_v1_video import BG, PANEL, TEXT, CYAN, _font


class _V2VideoRecorder(_V1VideoRecorder):
    def _chunk_panel(self, draw):
        super()._chunk_panel(draw)
        draw.rectangle((440, 150, 1555, 183), fill=PANEL)
        draw.text((444, 155), "ASTRA CANDIDATES / MEASURED V2 EXECUTION",
                  font=_font(19, True), fill=CYAN)

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 10, 1580, 49), fill=BG)
        draw.text((24, 17), "FOUNDATION VISION HARNESS V2 | COMMITTED ACTION QUEUE",
                  font=_font(25, True), fill=TEXT)
        return canvas

    def _emit(self, event, repeats, **extra):
        return super()._emit(event, repeats,
                            **{**extra, "harness_version": "foundation_vision_harness_V2",
                               "policy": "foundation_vision_harness_V2",
                               "video_version": "foundation_vision_harness_V2"})


class AsyncVideoRecorder(_AsyncRecorder):
    def __init__(self, path, *, episode_id="foundation_vision_harness_V2", **kwargs):
        kwargs.setdefault("renderer_factory", _V2VideoRecorder)
        super().__init__(path, episode_id=episode_id, **kwargs)
        self._worker.name = "fvh-v2-video"
