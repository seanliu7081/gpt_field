"""Use the actual recovery planner and tracker when presenting Field options."""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_policy_v17 import VisualFieldPolicyV17
from .visual_field_policy_v21 import VisualFieldPolicyV21
from .visual_field_v22 import MetricFieldExecutorV22
from .visual_field_subscription_v22 import geometry_semantics


class VisualFieldPolicyV22(VisualFieldPolicyV21):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV22()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self._recovery_joint_position = None

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        result['recovery_execution_contract'] = {
            'version': 'same-planner-tracker-recovery-preview-v22',
            'explicit_model_choice_required': True,
            'details': 'Short observed-motion reversals use a checked continuous Field segment. Offers preview the actual planner/tracker first action; every executed candidate is freshly checked.',
            'clearance_certified': False, 'physical_budgets_reset': False}
        return result

    def _refresh(self, obs):
        self._recovery_joint_position = obs.joint_position
        return super()._refresh(obs)

    def _recovery_scene_assess(self, executor, state8, delta_xyz):
        # Pure prospective assessment: do not overwrite the live policy's
        # scene_materialization or use the previous Direct branch as Field.
        material = self._materialize(state8)
        material['applied_metric_ids'] = [r['id'] for r in executor.region_geometry]
        material['applied_image_ids'] = [r['id'] for r in executor.translation_guard.neighbors]
        prediction = copy.deepcopy(self.prediction)
        prediction.update(controller='visual_field', phase='recover',
                          field_intent=copy.deepcopy(executor.intent),
                          motion_context=copy.deepcopy(executor.translation_guard.context))
        return self.scene_guard.assess(state8, delta_xyz, materialization=material,
            field=executor, monitor=self.monitor, prediction=prediction,
            contact_alignment=self.contact_alignment)

    def _screen_recovery_options(self, state8):
        # Refresh may screen the same snapshot several times. Replace prior
        # reverse offers; retain ordinary axis candidates and their checks.
        for key in ('candidates', 'rejected_candidates'):
            self.recovery_options[key] = [c for c in self.recovery_options.get(key, [])
                if c.get('source') != 'reverse_observed_worsening_motion']
        VisualFieldPolicyV17._screen_recovery_options(self, state8)
        if self.local.intent is None or self._recovery_joint_position is None:
            return
        for offer in self.translation_guard.recovery_directions():
            offset = np.asarray(offer['direction_xyz'], float)*.008
            preview = self.local.preview_recovery(state8, self._recovery_joint_position,
                offset, goal_tolerance_m=.003, command_limit_m=.002, max_travel_m=.008,
                scene_assessor=self._recovery_scene_assess)
            if not preview['accepted']:
                self.recovery_options['rejected_candidates'].append({
                    **copy.deepcopy(offer), 'reason': preview['status'],
                    'actual_executor_preview': preview})
                continue
            delta = np.asarray(preview['action'][:3], float)*.05
            self.recovery_options['candidates'].append({**copy.deepcopy(offer),
                'checked_nominal_segment_m': .008,
                'recommended_goal_offset_m': offset.tolist(),
                'recommended_goal_tolerance_m': .003,
                'recommended_command_limit_m': .002,
                'recommended_local_max_travel_m': .008,
                'preview_first_command_delta_m': delta.tolist(),
                'preview_reason': preview['status'],
                'actual_executor_preview': preview,
                'motion_axes': 'xyz', 'required_translation_mode': 'separate',
                'requires_explicit_subgoal_update': True,
                'local_limit_does_not_override_persistent_guard_allowance': True,
                'image_separation_observed': False, 'actual_segment_must_be_rechecked': True})
        if self.recovery_options['candidates']:
            self.recovery_options['reason'] = 'actual_planner_tracker_checked_recovery_candidates'

    def act(self, obs):
        self._recovery_joint_position = obs.joint_position
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v22'
        return decision

    def observe_transition(self, before, decision, after):
        self._recovery_joint_position = after.joint_position
        return super().observe_transition(before, decision, after)
