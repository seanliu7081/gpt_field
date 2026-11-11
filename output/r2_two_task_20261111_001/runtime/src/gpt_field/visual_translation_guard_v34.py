"""Bind prospective Field roles for previews without changing live permission."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import numpy as np

from .action import _rotvec_quaternion
from .visual_translation_guard_v33 import ProjectedTranslationGuardV33


class ProjectedTranslationGuardV34(ProjectedTranslationGuardV33):
    @contextmanager
    def preview_field_role(self, state8, monitor):
        context = self.reference_role_context
        refs = ((getattr(monitor, 'refs', {}) or {}).get('external') or {})
        state = np.asarray(state8, float)
        eligible = (
            isinstance(context, dict)
            and context.get('reference_role') == refs.get('reference_role') == 'rigid_transit_proxy'
            and context.get('moving_kind') == refs.get('moving_kind') == 'tool'
            and context.get('controller') in ('direct', 'visual_field')
            and context.get('phase') in ('approach', 'align', 'recover')
            and context.get('target_id') == refs.get('target_id') == self.target_id
            and context.get('target_contact') == 'none'
            and context.get('contact_alignment') is None
            and context.get('intended_contact_id') is None
            and context.get('translation_mode') in ('transit', 'separate')
            and state.shape == (8,) and np.isfinite(state).all()
            and np.array_equal(state, self.state)
            and np.array_equal(state, getattr(monitor, 'state', None))
        )
        if not eligible:
            yield False
            return
        # This is a proposed noncontact replacement, not a repair of an invalid
        # reference or a live controller change. The original proxy support,
        # whole-hand geometry, pose-domain and physical-debt checks still run.
        prospective = copy.deepcopy(context)
        prospective.update(controller='visual_field', phase='recover',
            translation_mode='separate', gripper_command='hold',
            orientation_target_xyzw=_rotvec_quaternion(state[3:6]).tolist())
        self.reference_role_context = prospective
        try:
            yield True
        finally:
            self.reference_role_context = context

    def preview_assess(self, delta_xyz, *, prospective_translation_mode=None, **kwargs):
        if (prospective_translation_mode != 'separate'
                or kwargs.get('orientation_hold') is not True
                or kwargs.get('rotation_rad', 0.) != 0.):
            return super().preview_assess(delta_xyz,
                prospective_translation_mode=prospective_translation_mode, **kwargs)
        with self.preview_field_role(self.state, kwargs.get('monitor')):
            return super().preview_assess(delta_xyz,
                prospective_translation_mode=prospective_translation_mode, **kwargs)
