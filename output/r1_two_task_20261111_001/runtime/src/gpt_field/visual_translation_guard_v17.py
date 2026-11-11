"""Preserve actual proposed translations in accepted and rejected diagnostics."""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v16 import ProjectedTranslationGuardV16


class ProjectedTranslationGuardV17(ProjectedTranslationGuardV16):
    def assess(self, delta_xyz, **kwargs):
        result = super().assess(delta_xyz, **kwargs)
        # This is the proposal checked here, never the zero hold executed after
        # a refusal. Pure preview uses a deep copy and retains its own proposal.
        result = copy.deepcopy(result)
        result.setdefault('diagnostics', {})['checked_proposal'] = {
            'delta_xyz_m': np.asarray(delta_xyz, float).tolist(),
            'nominal_clear': bool(kwargs.get('nominal_clear', False)),
            'nominal_away': bool(kwargs.get('nominal_away', False)),
            'rotation_rad': float(kwargs.get('rotation_rad', 0.)),
            'translation_mode': (self.context or {}).get('translation_mode'),
            'executed': False,
            'meaning': 'Actual candidate at this admission call; observe alone records execution.',
        }
        return result
