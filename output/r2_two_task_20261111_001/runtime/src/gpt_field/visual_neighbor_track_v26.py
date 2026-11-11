"""Opt-in neighbor matching with an exact descriptor stationary candidate.

An asymmetric discrete NCC peak can acquire a parabolic subpixel offset even
when the previous descriptor samples are identical in the next image. Repeating
that numerical offset can distort a larger rectangle transported by a small
feature bundle. This module preserves every inherited matching and identity
check before considering the exact same-coordinate descriptor as a candidate.

Descriptor equality is neither whole-object stationarity nor semantic identity,
and supplies no independent robot-motion or collision-clearance evidence.
"""
from __future__ import annotations

import numpy as np

from .visual_monitor_v5 import LocalFeatureBundle, _sample


class ExactDescriptorNeighborTrackV26(LocalFeatureBundle):
    """Keep exact unchanged grayscale samples at their existing coordinates."""

    def _match(self, source, destination, point, *, search=7):
        matched = super()._match(source, destination, point, search=search)
        if matched is None:
            # No rescue of low texture, ambiguous peaks, occlusion or bounds.
            return None
        point = np.asarray(point, float)
        # One pixel is the sum of nearest-integer rounding (<=.5) and the
        # inherited bounded parabola (<=.5), not a new search/capture radius.
        if np.max(np.abs(matched[0]-point)) > 1.:
            return matched
        previous = _sample(source, point+self.offsets)
        current = _sample(destination, point+self.offsets)
        if previous is not None and current is not None and np.array_equal(previous, current):
            return point.copy(), 1.
        return matched

    def summary(self):
        result = super().summary()
        result["exact_descriptor_matching"] = {
            "legacy_success_required": True,
            "sample_equality": "exact_bilinear_grayscale_descriptor",
            "maximum_legacy_offset_per_axis_px": 1.,
            "whole_object_stationarity_certified": False,
            "automatic_reseeding": False,
        }
        return result
