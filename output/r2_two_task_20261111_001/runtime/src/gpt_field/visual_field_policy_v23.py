"""Explicit, observed uncertainty egress without replacing task Field planning."""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v22 import VisualFieldPolicyV22
from .visual_field_v23 import MetricFieldExecutorV23
from .visual_translation_guard_v23 import ProjectedTranslationGuardV23
from .visual_scene_guard_v23 import SceneCoverageGuardV23
from .visual_field_subscription_v23 import geometry_semantics


class VisualFieldPolicyV23(VisualFieldPolicyV22):
    def reset(self):
        super().reset()
        self.translation_guard = ProjectedTranslationGuardV23()
        self.scene_guard = SceneCoverageGuardV23()
        self.local = MetricFieldExecutorV23()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess

    def _bind_egress_context(self):
        prediction = self.prediction or {}
        context = prediction.get('motion_context') or {}
        self.local.egress_context = {
            'controller': prediction.get('controller'), 'phase': prediction.get('phase'),
            'purpose': context.get('purpose'),
            'translation_mode': context.get('translation_mode')}

    def _egress_requested(self):
        p = self.prediction or {}
        c = p.get('motion_context') or {}
        return (p.get('controller') == 'visual_field' and p.get('phase') == 'recover'
            and c.get('purpose') == 'reference_repair' and c.get('translation_mode') == 'separate')

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        guard = getattr(self, 'translation_guard', None)
        evidence = getattr(guard, 'egress_evidence', None)
        result['uncertainty_egress'] = None if evidence is None else evidence.summary()
        return result

    def _recovery_scene_assess(self, executor, state8, delta_xyz):
        material = self._materialize(state8)
        material['applied_metric_ids'] = [r['id'] for r in executor.region_geometry]
        material['applied_image_ids'] = [r['id'] for r in executor.translation_guard.neighbors]
        prediction = copy.deepcopy(self.prediction)
        context = copy.deepcopy(executor.translation_guard.context)
        if (getattr(executor, 'egress_context', None) or {}).get('purpose') == 'reference_repair':
            context.update(purpose='reference_repair', translation_mode='separate')
        prediction.update(controller='visual_field', phase='recover',
            field_intent=copy.deepcopy(executor.intent), motion_context=context)
        return self.scene_guard.assess(state8, delta_xyz, materialization=material,
            field=executor, monitor=self.monitor, prediction=prediction,
            contact_alignment=self.contact_alignment)

    def _screen_recovery_options(self, state8):
        self._bind_egress_context()
        for key in ('candidates', 'rejected_candidates'):
            self.recovery_options[key] = [c for c in self.recovery_options.get(key, [])
                if c.get('source') != 'initial_uncertainty_egress']
        super()._screen_recovery_options(state8)
        if self.local.intent is None or self._recovery_joint_position is None:
            return
        for candidate in self.local.egress_candidates(state8):
            offset = np.asarray(candidate['goal_offset_m'], float)
            preview = self.local.preview_egress(state8, self._recovery_joint_position,
                offset, scene_assessor=self._recovery_scene_assess)
            if not preview.get('accepted'):
                self.recovery_options['rejected_candidates'].append({
                    'source': 'initial_uncertainty_egress', 'goal_offset_m': offset.tolist(),
                    'reason': preview['status']})
                continue
            delta = np.asarray(preview['action'][:3], float)*.05
            self.recovery_options['candidates'].append({
                'source': 'initial_uncertainty_egress',
                'direction_xyz': (offset/np.linalg.norm(offset)).tolist(),
                'recommended_goal_offset_m': offset.tolist(),
                'recommended_goal_tolerance_m': .003,
                'recommended_command_limit_m': .003,
                'recommended_local_max_travel_m': .04,
                'recommended_max_actions': 120,
                'checked_nominal_segment_m': float(np.linalg.norm(offset)),
                'preview_first_command_delta_m': delta.tolist(),
                'preview_reason': preview['status'],
                'required_phase': 'recover', 'required_purpose': 'reference_repair',
                'required_translation_mode': 'separate', 'motion_axes': 'xyz',
                'requires_explicit_subgoal_update': True,
                'initial_probe_measured_limit_m': .004,
                'continued_motion_requires_fresh_image_evidence': True,
                'all_scene_hypotheses_retained': True, 'clearance_certified': False,
                'physical_debt_replenished': False,
                'actual_segment_must_be_rechecked': True})
        if any(c.get('source') == 'initial_uncertainty_egress'
               for c in self.recovery_options['candidates']):
            self.recovery_options['reason'] = 'actual_executor_checked_initial_uncertainty_egress'

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        self._bind_egress_context()
        return latency

    def _field_control_limits(self, obs):
        self._bind_egress_context()
        result = super()._field_control_limits(obs)
        if self._egress_requested():
            # A V20 weak-reference deferral still needs candidate admission.
            # For egress that admission is its explicit geometry/evidence
            # contract, not the remote-transit's disjoint-halo requirement.
            # Every other monitor refusal remains in `result` unchanged.
            self.local.proprioceptive_transit_request = None
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v23'
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self.translation_guard.observe_egress(before.state, decision.action, after.state)
        if self.refresh_reason:
            self._screen_recovery_options(after.state)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
