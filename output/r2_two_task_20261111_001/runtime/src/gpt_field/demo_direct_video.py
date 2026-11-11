"""Use the shared recorder with an explicit pure Direct title."""
from PIL import ImageDraw

from .hybrid_video import HybridVideoRecorder
from .video import BG, TEXT, WIDTH, _font


class DemoDirectVideoRecorder(HybridVideoRecorder):
    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 50), fill=BG)
        draw.text((24, 17), "ASTRA / LIBERO | ONE DEMONSTRATION + DIRECT CONTROL",
                  font=_font(25, True), fill=TEXT)
        return canvas


class ZeroShotDirectVideoRecorder(HybridVideoRecorder):
    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 50), fill=BG)
        draw.text((24, 17), "ASTRA / LIBERO | ZERO-SHOT DIRECT / NO OBSTACLE COLLISION",
                  font=_font(25, True), fill=TEXT)
        return canvas


class HistoricalZeroShotDirectVideoRecorder(HybridVideoRecorder):
    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 50), fill=BG)
        draw.text((24, 17), "ASTRA / LIBERO | ZERO-SHOT DIRECT CONTROL",
                  font=_font(25, True), fill=TEXT)
        return canvas
