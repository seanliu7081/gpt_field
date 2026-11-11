"""Explicit recovery selection and feedback-aware Field tracking.

Recovery family belongs to the persistent model subgoal. A reference refresh or
temporary Direct action cannot implicitly switch the planner or renew travel.
"""
from __future__ import annotations

import copy

from .visual_field_policy_v24 import VisualFieldPolicyV24
from .visual_field_v13 import ContactSubgoalLedger
from .visual_field_v25 import MetricFieldExecutorV25, RECOVERY_ROUTES
from .robot_translation_response_v25 import MeasuredTranslationResponseV25
from .visual_translation_guard_v25 import ProjectedTranslationGuardV25


class RecoverySubgoalLedgerV25(ContactSubgoalLedger):
    def apply(self, state8, prediction):
        route = prediction.get('recovery_route')
        request = prediction.get('subgoal_update') or {}
        field = prediction.get('controller') == 'visual_field'
        if route not in RECOVERY_ROUTES or (not field and route != 'normal'):
            return self._result(False, 'subgoal_recovery_route_invalid', request=request)
        if (field and request.get('action') == 'retain' and self.active is not None
                and route != self.active.get('recovery_route')):
            return self._result(False, 'recovery_route_change_requires_explicit_replace', request=request)
        result = super().apply(state8, prediction)
        if result['accepted'] and field:
            self.active['recovery_route'] = route
            result['feedback'] = self.summary()
        return result


class VisualFieldPolicyV25(VisualFieldPolicyV24):
    def reset(self):
        super().reset()
        self.subgoals = RecoverySubgoalLedgerV25()
        self.robot_response = MeasuredTranslationResponseV25()
        self.translation_guard = ProjectedTranslationGuardV25()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV25()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.route_binding_result = None

    def _bind_egress_context(self):
        super()._bind_egress_context()
        if self.field_configured and self.prediction:
            # Called after both the executor and current translation guard are
            # configured, before prospective screening and actual selection.
            route = self.prediction.get('recovery_route')
            bound = self.local.recovery_route_binding
            if bound is not None and bound['recovery_route'] == route:
                # A refusal hold updates the observed state without refreshing
                # the stopped executor. Preserve its binding; actual select
                # still checks the exact expected state and fixed goal.
                return
            self.route_binding_result = self.local.bind_recovery_route(route, self.translation_guard.state)
            if not self.route_binding_result['accepted']:
                self.subgoal_rejection = self.route_binding_result['reason']

    def _egress_requested(self):
        return (self.prediction or {}).get('recovery_route') == 'initial_uncertainty_egress'

    def _screen_recovery_options(self, state8):
        super()._screen_recovery_options(state8)
        for candidate in self.recovery_options.get('candidates', []):
            candidate['required_recovery_route'] = {
                'initial_uncertainty_egress': 'initial_uncertainty_egress',
                'reverse_observed_worsening_motion': 'reverse_observed',
            }.get(candidate.get('source'), 'normal')

    def _observable_feedback(self):
        from .visual_field_subscription_v25 import geometry_semantics
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        result['recovery_route_binding'] = copy.deepcopy(
            getattr(self.local, 'recovery_route_binding', None))
        result['recovery_execution_contract'].update(
            version='explicit-route-and-current-command-checks-v25',
            route_change_requires_explicit_subgoal_replace=True,
            recent_response_can_change_command_permission_without_new_budget=True)
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v25'
        decision.diagnostics['recovery_route_binding'] = copy.deepcopy(self.local.recovery_route_binding)
        return decision
