"""Seed-preserving part correspondence with exact stationary descriptor candidates.

Asymmetric parabolic NCC interpolation need not map an unchanged raster to zero
motion. The already vetted V26 same-coordinate descriptor candidate is applied
here to persistent part tracks, after ordinary texture/uniqueness/search checks.
This is numerical image registration, not semantic robot-part identification.
"""
from __future__ import annotations

import numpy as np

from .visual_monitor_v5 import _sample
from .visual_reference_track_v21 import PartFeatureTrackV21


class ExactDescriptorPartTrackV32(PartFeatureTrackV21):
    """Retain all V21 identity/appearance constraints and avoid exact-raster drift."""

    def _match(self, source, destination, point, *, search=7):
        matched = super()._match(source, destination, point, search=search)
        if matched is None:
            return None
        point = np.asarray(point, float)
        # Nearest integer rounding plus the bounded parabolic refinement is at
        # most one pixel per axis. A more distant match is not reinterpreted.
        if np.max(np.abs(matched[0]-point)) > 1.:
            return matched
        previous = _sample(source, point+self.offsets)
        current = _sample(destination, point+self.offsets)
        if previous is not None and current is not None and np.array_equal(previous, current):
            return point.copy(), 1.
        return matched

    def summary(self):
        result = super().summary()
        result['exact_descriptor_matching'] = {
            'version': 'exact-descriptor-part-matching-v32',
            'legacy_success_required': True,
            'sample_equality': 'exact_bilinear_grayscale_descriptor',
            'maximum_legacy_offset_per_axis_px': 1.,
            'bounded_seed_verified_appearance_unchanged': True,
            'independent_robot_motion_or_semantic_evidence': False,
            'whole_object_stationarity_certified': False,
            'physical_budgets_modified': False,
        }
        return result
