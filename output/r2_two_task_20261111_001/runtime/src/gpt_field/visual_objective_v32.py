"""Separate a rigid motion witness from an anatomical image objective.

An upper rigid feature can support measured translation. Its pixel offset is
not the location of the fingers or a pinch target. Metric Field planning and
the separately tracked hand outline remain available for that role.
"""
from __future__ import annotations

import copy

from .visual_goal_field_v31 import VisualGoalFieldV31
from .visual_task_progress_v21 import VisualTaskProgressV21
from .visual_field_proposals_v31 import FieldProposalBankV31, observation_binding


def uses_rigid_proxy(prediction):
    refs = ((prediction or {}).get('visual_references') or {}).get('external') or {}
    return refs.get('reference_role') == 'rigid_transit_proxy'


class VisualGoalFieldV32(VisualGoalFieldV31):
    version = 'anatomical-objective-separated-from-motion-proxy-v32'

    def _reset_estimator(self, reason):
        super()._reset_estimator(reason)
        self._v32_goal_replay_report = None

    @staticmethod
    def _capture(monitor, prediction):
        if uses_rigid_proxy(prediction):
            return None, 'rigid_proxy_has_no_anatomical_image_goal'
        return VisualGoalFieldV31._capture(monitor, prediction)

    def summary(self):
        result = super().summary()
        replay = getattr(self, '_v32_goal_replay_report', None)
        result['retrospective_training'] = (copy.deepcopy(replay)
            if replay and replay.get('reference_epoch') == self.epoch else None)
        result['held_out_records_are_prospective_only'] = True
        return result


class VisualTaskProgressV32(VisualTaskProgressV21):
    @staticmethod
    def _snapshot(monitor, prediction):
        if uses_rigid_proxy(prediction):
            return None
        return VisualTaskProgressV21._snapshot(monitor, prediction)


class FieldProposalBankV32(FieldProposalBankV31):
    def build(self, policy, obs, visual_report):
        if not uses_rigid_proxy(policy.prediction):
            return super().build(policy, obs, visual_report)
        self.binding = observation_binding(obs)
        self.offers, self.rejected = [], []
        self.visual_report = copy.deepcopy(visual_report)
        result = self.summary()
        result['reason'] = 'proxy_motion_support_is_not_anatomical_goal_support'
        result['ordinary_metric_field_and_recovery_choices_available'] = True
        return result
