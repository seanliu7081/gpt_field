"""Preview the proposed Field controller after a reference-only Direct handoff."""
from __future__ import annotations

from .visual_field_v33 import MetricFieldExecutorV33


class MetricFieldExecutorV34(MetricFieldExecutorV33):
    def preview_recovery(self, state8, joint_position, goal_offset_m, **kwargs):
        guard = self.translation_guard
        if (not callable(getattr(guard, 'preview_field_role', None))
                or (self.intent or {}).get('gripper_command') != 'hold'):
            return super().preview_recovery(state8, joint_position, goal_offset_m, **kwargs)

        # The inherited executor constructs this same recovery intent with the
        # measured orientation held. Bind its controller semantics transactionally
        # before that preview; all geometry, reference and measured-debt gates
        # still inspect the exact prospective command. Live selection has no
        # access to this temporary context and must bind a real model response.
        context = guard.reference_role_context
        with guard.preview_field_role(state8, self.visual_monitor) as rebound:
            result = super().preview_recovery(state8, joint_position, goal_offset_m, **kwargs)
        if not rebound:
            return result
        result.setdefault('diagnostics', {})['prospective_reference_role_binding'] = {
            'source_controller': context['controller'],
            'proposed_controller': 'visual_field',
            'role': 'rigid_transit_proxy',
            'explicit_model_selection_required': True,
            'live_role_context_restored': True,
            'geometry_and_physical_debt_unchanged': True,
            'clearance_certified': False,
        }
        return result
