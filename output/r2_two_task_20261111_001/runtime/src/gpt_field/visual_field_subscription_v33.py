"""V33 recovery guidance; V32 wire, validation and presentation are unchanged."""
from __future__ import annotations

from .visual_field_subscription_v32 import (
    VisualFieldSubscriptionClientV32,
    output_schema,
    geometry_semantics,
    shared_guidance as guidance_v32,
    HISTORY_VERSION,
    FEEDBACK_VERSION,
)

VERSION = 'visual-field-rgb-joints-v33-observed-adverse-checkpoint-recovery-v1'
RECOVERY_GUIDANCE = (
    'CHECKPOINT RECOVERY. After observed adverse motion, explicitly select a current accepted '
    'reverse_observed offer when appropriate. This finite reverse may abandon an incomplete '
    'observation checkpoint for the old heading; it does not erase the old evidence or refund '
    'measured travel, per-heading/shared debt or intent limits. Use the offered goal and limits '
    'with phase=recover and an explicit subgoal_update=replace; retain or reseed alone grants '
    'nothing. Read translation_guard.precheckpoint_reverse_abort for the interrupted checkpoint. '
    'Every actual segment still passes current '
    'geometry, reference, contact and adverse-motion checks. Rejected or stale previews are not '
    'admitted choices, and copying an offer is not execution permission or a clearance certificate. '
    'No executable offer means report the current blockage honestly; no extra confirmation '
    'motion, new perception input or automatic recovery is authorized.'
)


def shared_guidance(task, demonstration=False):
    return guidance_v32(task, demonstration) + '\n\n' + RECOVERY_GUIDANCE


class VisualFieldSubscriptionClientV33(VisualFieldSubscriptionClientV32):
    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format']['name'] = 'visual_field_rgb_joints_v33'
        return payload
