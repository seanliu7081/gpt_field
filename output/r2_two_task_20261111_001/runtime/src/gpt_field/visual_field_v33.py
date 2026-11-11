"""Preview a reviewed adverse-motion abort without granting live permission."""
from __future__ import annotations

from .visual_field_v31 import MetricFieldExecutorV31


class MetricFieldExecutorV33(MetricFieldExecutorV31):
    def preview_recovery(self, state8, joint_position, goal_offset_m, **kwargs):
        guard = self.translation_guard
        if guard is None:
            return super().preview_recovery(state8, joint_position, goal_offset_m, **kwargs)
        with guard.preview_reverse_abort_context(state8, goal_offset_m):
            return super().preview_recovery(state8, joint_position, goal_offset_m, **kwargs)

