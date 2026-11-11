"""Reference repair from paid observations and finite model-conditioned routes.

Robot-reference evidence and geometric risk are separate claims. The model
chooses the semantic reference and fixed Field goal; local execution retains
the scene hypotheses and interrupts on contrary or unavailable observations.
"""
from __future__ import annotations

import copy
import time

from .visual_field_policy_v23 import VisualFieldPolicyV23
from .visual_field_v24 import MetricFieldExecutorV24
from .visual_translation_guard_v24 import ProjectedTranslationGuardV24
from .visual_reference_candidates_v24 import CausalReferenceCandidatesV24
from .visual_field_subscription_v24 import geometry_semantics


class VisualFieldPolicyV24(VisualFieldPolicyV23):
    def reset(self):
        super().reset()
        self.reference_candidates = CausalReferenceCandidatesV24()
        self.client.reference_candidate_provider = self.reference_candidates.image
        self.client.reference_candidate_resolver = self._resolve_reference_candidate
        self.translation_guard = ProjectedTranslationGuardV24()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local = MetricFieldExecutorV24()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.route_reference_selection = None

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result['geometry_semantics'] = geometry_semantics()
        result['route_reference_selection'] = copy.deepcopy(
            getattr(self, 'route_reference_selection', None))
        bank = getattr(self, 'reference_candidates', None)
        state = getattr(getattr(self, 'monitor', None), 'state', None)
        if bank is not None and state is not None and hasattr(bank, 'raw_support_evidence'):
            result['controlled_reference_witness'] = bank.raw_support_evidence(state, self.monitor)
        return result

    def _screen_recovery_options(self, state8):
        super()._screen_recovery_options(state8)
        for candidate in self.recovery_options.get('candidates', []):
            if candidate.get('source') == 'initial_uncertainty_egress':
                candidate.update(
                    continued_motion_requires_fresh_image_evidence=True,
                    continuation_contract='finite_model_conditioned_route_with_causal_reference_and_current_contradiction_checks',
                    positive_image_gap_is_required=False,
                    fixed_contextual_measured_limit_m=.040,
                    reference_repair_confirmation_limit_m=.004,
                    reference_repair_confirmation_can_renew=False,
                    clearance_certified=False)

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        started = time.monotonic()
        raw = self.prediction.get('raw_wire_prediction') or {}
        identity = self.prediction.get('reference_candidate_id', raw.get('reference_candidate_id'))
        resolved = self._resolved_reference_candidates.get(identity) if identity is not None else None
        self.route_reference_selection = (
            None if resolved is None else self.translation_guard.bind_reference_selection(resolved, obs.state))
        # Preview admission sees the same current explicit reference selection
        # as the subsequent actual action. This never commits physical travel.
        if resolved is not None:
            self._screen_recovery_options(obs.state)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v24'
        selected = decision.diagnostics.get('selection') or {}
        if not selected.get('accepted', True):
            started = time.monotonic()
            self._review_paid_reference(obs, selected.get('status'))
            self.feedback = self._observable_feedback()
            decision.diagnostics['feedback'] = copy.deepcopy(self.feedback)
            elapsed = time.monotonic()-started
            decision.diagnostics['local_planning_s'] += elapsed
            self.local_total_s += elapsed
        return decision

    def _review_paid_reference(self, observation, reason):
        report = self.reference_candidates.handoff(observation, self.monitor, reason)
        identity = report.get('review_id')
        if report.get('review_required') and identity is not None:
            if identity not in self._requested_reference_reviews:
                self._requested_reference_reviews.add(identity)
                if not self.refresh_reason:
                    self.refresh_reason = 'causal_controlled_reference_review'
                self.events.append({'step': self.step_index-1,
                    'reason': 'paid_observation_reference_review', 'review_id': identity,
                    'physical_allowance_renewed': False})

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self._review_paid_reference(after, self.refresh_reason)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
