"""Stable robot-image measurements with distinct motion and hand roles."""
from __future__ import annotations

import copy
import time

from .visual_field_policy_v31 import VisualFieldPolicyV31
from .visual_monitor_v32 import RoleAwareVisualMonitorV32
from .visual_reference_candidates_v32 import CausalReferenceCandidatesV32
from .visual_translation_guard_v32 import ProjectedTranslationGuardV32
from .visual_objective_v32 import (
    VisualGoalFieldV32, VisualTaskProgressV32, FieldProposalBankV32, uses_rigid_proxy,
)


class VisualFieldPolicyV32(VisualFieldPolicyV31):
    def reset(self):
        super().reset()
        self.monitor = RoleAwareVisualMonitorV32()
        self.reference_candidates = CausalReferenceCandidatesV32()
        self.client.reference_candidate_provider = self.reference_candidates.image
        self.client.reference_candidate_resolver = self._resolve_reference_candidate
        self.translation_guard = ProjectedTranslationGuardV32()
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.translation_guard.bind_reference_witness(self.reference_candidates)
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.visual_goal_field = VisualGoalFieldV32()
        self.task_progress = VisualTaskProgressV32()
        self.field_proposals = FieldProposalBankV32()
        self.client.field_proposal_resolver = self.field_proposals.resolve
        self.visual_goal_replay = {'accepted': False, 'reason': 'no_selected_anatomical_candidate'}

    def _bind_egress_context(self):
        guard = getattr(self, 'translation_guard', None)
        if hasattr(guard, 'bind_reference_role_context'):
            guard.bind_reference_role_context(self.prediction)
        return super()._bind_egress_context()

    def _allows_planar_correction(self):
        return not uses_rigid_proxy(self.prediction) and super()._allows_planar_correction()

    def _refresh(self, obs):
        latency = super()._refresh(obs)
        started = time.monotonic()
        identity = (self.prediction or {}).get('reference_candidate_id')
        candidate = self._resolved_reference_candidates.get(identity) if identity is not None else None
        if candidate is not None and not uses_rigid_proxy(self.prediction):
            from .visual_goal_replay_v32 import warm_start_visual_goal
            self.visual_goal_replay = warm_start_visual_goal(
                self.visual_goal_field, self.monitor, candidate, obs, self.reference_replay)
        else:
            self.visual_goal_replay = {'accepted': False,
                'reason': 'proxy_has_no_anatomical_goal' if uses_rigid_proxy(self.prediction)
                          else 'no_selected_anatomical_candidate'}
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def _observable_feedback(self):
        result = super()._observable_feedback()
        from .visual_field_subscription_v32 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        result['visual_goal_replay'] = copy.deepcopy(getattr(self, 'visual_goal_replay', None))
        result['rigid_proxy_use'] = {
            'active': uses_rigid_proxy(self.prediction),
            'anatomical_goal_or_contact_evidence': False,
            'whole_hand_outline_tracked_independently': True,
            'proxy_transform_transports_hand_outline': False,
            'metric_field_planning_retained': True,
        }
        health = result.get('visual', {}).get('reference_health')
        if isinstance(health, dict) and health.get('candidate_review_required'):
            health['recommended_action'] = (
                'Inspect current raw views and numbered candidates. Choose a visible '
                'tool-anatomy feature, or explicitly declare rigid_transit_proxy for '
                'fixed-orientation noncontact translation with an independent actual-hand '
                'point and outline. A proxy is not a pinch location. Selection retains '
                'all current scene constraints and measured travel accounting.')
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v32'
        return decision
