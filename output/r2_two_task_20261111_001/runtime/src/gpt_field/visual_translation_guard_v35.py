"""Keep an unchanged hand outline in its original tracking coordinates.

Refreshing a declaration at the already observed pose must not convert the
transported quadrilateral to an AABB and then back to a different quadrilateral.
The hook runs immediately after V11 configuration, before the inherited V12/V26
scalar-window and V27 boundary snapshots. No admission rule or ledger changes.
"""
from __future__ import annotations

import copy

import numpy as np

from .visual_translation_guard_v11 import ProjectedTranslationGuardV11, _bounds, _gray
from .visual_translation_guard_v34 import ProjectedTranslationGuardV34


class _RetainHandSourceAfterV11(ProjectedTranslationGuardV11):
    """An MRO hook below V12, so all higher observers see the retained source."""

    def configure(self, images, motion_context, state8, *, target_id, monitor=None):
        track = self._tool(monitor)
        old_track = self.tool_track
        old_source = None if self.tool_source is None else self.tool_source.copy()
        prior_extent = getattr(self, '_v35_source_outline_before_configure', None)
        old_box = (None if self.tool_box is None else self.tool_box.copy()) if prior_extent is None else prior_extent[0]
        old_scale = (None if self.scale is None else self.scale.copy()) if prior_extent is None else prior_extent[1]
        current = np.asarray(state8, float)
        previous_context = self.context if isinstance(self.context, dict) else {}
        incoming = motion_context if isinstance(motion_context, dict) else {}
        eligible = False
        same_declared_extent = False
        checks = {}
        try:
            pixels = _gray(images['external'])
            declared = np.asarray(incoming.get('tool_external_roi_uv'), float)
            # A checked offer restates the *current transported* AABB. That is
            # not a new shape declaration; inversion would still change the
            # original quadrilateral. A genuinely different extent is excluded.
            same_declared_extent = bool(
                incoming.get('tool_external_roi_uv') == previous_context.get('tool_external_roi_uv')
                or (old_box is not None and old_scale is not None and declared.shape == (4,)
                    and np.isfinite(declared).all()
                    and np.allclose(declared * np.tile(old_scale, 2), old_box,
                                    atol=1e-7, rtol=0.)))
            checks = {
                'valid_context_and_domain': self.context_invalid is None and self.domain_invalid is None,
                'same_target': target_id == self.target_id,
                'same_reliable_track': track is not None and track is old_track and track.reliable,
                'source_available': old_source is not None and old_source.shape == (4, 2)
                    and np.isfinite(old_source).all(),
                'box_and_scale_available': old_box is not None and old_scale is not None,
                'same_guard_state': np.array_equal(current, self.state),
                'same_monitor_state': np.array_equal(current, getattr(monitor, 'state', None)),
                'same_observed_pixels': np.array_equal(pixels, getattr(track, 'previous_image', None)),
                'same_declared_extent': same_declared_extent,
            }
            eligible = all(checks.values())
            if eligible:
                eligible = np.allclose(_bounds(old_source @ track.matrix + track.translation),
                                       old_box, atol=1e-7, rtol=0.)
                checks['same_transported_source_bounds'] = eligible
        except (AttributeError, KeyError, TypeError, ValueError):
            # Malformed or unbound input retains all inherited validation.
            eligible = False
        result = super().configure(images, motion_context, state8,
                                   target_id=target_id, monitor=monitor)
        retained = bool(
            eligible and self.context_invalid is None and self.domain_invalid is None
            and self.tool_track is old_track and old_track.reliable
            and np.array_equal(self.scale, old_scale)
            and self.tool_box is not None
            and np.allclose(self.tool_box, old_box, atol=1e-7, rtol=0.)
        )
        if retained:
            self.tool_source = old_source
        self.source_continuity = {
            'retained': retained,
            'eligibility': {key: bool(value) for key, value in checks.items()},
            'reason': ('same_observation_track_and_declared_hand_outline'
                       if retained else 'inherited_configuration_or_revision'),
            'repair_changes_current_outline_extent': False,
            'repair_changes_admission_or_warning': False,
            'physical_debt_replenished': False,
            'clearance_certified': False,
        }
        return result


class ProjectedTranslationGuardV35(ProjectedTranslationGuardV34, _RetainHandSourceAfterV11):
    """Opt-in source-coordinate continuity; all V34 role checks are inherited."""

    version = 'same-observation-hand-source-continuity-v35'

    def configure(self, *args, **kwargs):
        # V12 intentionally clears tool_box before forwarding a supported fresh
        # ROI to V11. Capture its pre-refresh extent before that validation, but
        # restore source coordinates only inside the post-V11 hook above.
        previous = getattr(self, '_v35_source_outline_before_configure', None)
        self._v35_source_outline_before_configure = (
            None if self.tool_box is None else self.tool_box.copy(),
            None if self.scale is None else self.scale.copy(),
        )
        try:
            return super().configure(*args, **kwargs)
        finally:
            if previous is None:
                del self._v35_source_outline_before_configure
            else:
                self._v35_source_outline_before_configure = previous

    def summary(self):
        result = super().summary()
        result['hand_source_continuity'] = copy.deepcopy(
            getattr(self, 'source_continuity', None))
        return result
