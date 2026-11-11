"""Measured visual-goal Field proposals and explicit scene-belief review.

Existing fixed-goal planning/tracking and all actual action checks remain in
force. New proposals are selected explicitly by the model; paid observations
change an image response estimate, never scene geometry or semantic identity.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v30 import VisualFieldPolicyV30
from .visual_field_proposals_v31 import FieldProposalBankV31
from .visual_goal_field_v31 import VisualGoalFieldV31
from .visual_scene_inventory_v31 import VisualSceneInventoryV31
from .visual_field_v31 import MetricFieldExecutorV31


class VisualFieldPolicyV31(VisualFieldPolicyV30):
    def reset(self):
        super().reset()
        self.inventory = VisualSceneInventoryV31()
        self.inventory.revision_context_provider = lambda: (self.prediction or {}).get('scene_revisions', [])
        self.client.scene_revision_validator = self.inventory.validate_revision_update
        self.translation_guard.bind_egress_runtime(self.inventory, self.monitor)
        self.local = MetricFieldExecutorV31()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.visual_goal_field = VisualGoalFieldV31()
        self.field_proposals = FieldProposalBankV31()
        self.client.field_proposal_resolver = self.field_proposals.resolve
        self.scene_review = {'required': False, 'reasons': [], 'clearance_certified': False}

    def _observable_feedback(self):
        result = super()._observable_feedback()
        from .visual_field_subscription_v31 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        field = getattr(self, 'visual_goal_field', None)
        result['visual_goal_field'] = None if field is None else field.summary()
        bank = getattr(self, 'field_proposals', None)
        result['field_proposals'] = None if bank is None else bank.summary()
        result['scene_review'] = copy.deepcopy(getattr(self, 'scene_review', None))
        return result

    def _allows_planar_correction(self):
        if (self.prediction or {}).get('field_proposal_id') is not None:
            return False
        return super()._allows_planar_correction()

    def _review_scene(self, obs, report):
        reasons = []
        progress = self.task_progress.summary().get('active') or {}
        if progress.get('review_required'):
            reasons.append({'reason': 'metric_goal_and_observed_image_progress_disagree',
                'evidence': {key: progress.get(key) for key in ('target_id', 'world_goal_xyz',
                    'metric_improvement_m', 'image_error_change_px', 'measured_path_m')}})
        # A geometric target centre is not the declared feature objective. Only
        # report directional disagreement for model review, never correct 3D
        # geometry from an underdetermined image response.
        target = (self.prediction or {}).get('target_id')
        entries = self.inventory.summary(obs.state)['entries']
        item = next((row for row in entries if row['id'] == target), None)
        if item and item.get('current_geometry'):
            delta = np.asarray(item['current_geometry']['center_offset_m'], float)
            for candidate in report.get('candidates', []):
                move = np.asarray(candidate['goal_offset_m'], float)
                if np.dot(move, delta) < -.1*np.linalg.norm(move)*np.linalg.norm(delta):
                    reasons.append({'reason': 'image_improving_heading_opposes_metric_target_centre',
                        'target_id': target, 'current_target_center_offset_m': delta.tolist(),
                        'image_improving_goal_offset_m': move.tolist(),
                        'interpretation': 'Review target estimate, selected anatomy and desired visual offset; not a measured object pose.'})
                    break
        rejected = self.field_proposals.rejected
        if report.get('candidates') and not self.field_proposals.offers:
            reasons.append({'reason': 'visual_goal_candidates_have_no_current_admissible_scene_route',
                            'candidate_rejections': copy.deepcopy(rejected)})
        if self.local.stop_reason in ('invalid_measured_start', 'invalid_goal_hypothesis'):
            reasons.append({'reason': self.local.stop_reason,
                'interpretation': 'Inspect current RGB and estimate provenance. Model geometry is revisable with explicit evidence, not certified occupancy.'})
        self.scene_review = {'required': bool(reasons), 'reasons': reasons,
            'current_observation_index': self.inventory.observation_index,
            'object_estimate_provenance': [{key: copy.deepcopy(row[key]) for key in
                ('id', 'geometry_revision_count', 'estimated_at_observation',
                 'metric_age_observations') if key in row} for row in entries],
            'instruction': 'Use current RGB, robot state and measured response to review the identified estimates. If correcting an estimate, declare its geometry and explicit scene_revisions evidence. Null retains the old estimate. A revision is an unverified hypothesis and still undergoes all action checks.',
            'automatic_scene_change': False, 'clearance_certified': False}

    def _refresh(self, obs):
        started = time.monotonic()
        axes = (self.local.intent or {}).get('motion_axes', 'xyz')
        report = self.visual_goal_field.propose(obs.state, motion_axes=axes,
            metric_goal_xyz=self.goal, max_goal_distance_m=.030, max_candidates=6,
            joint_position=obs.joint_position)
        self.field_proposals.build(self, obs, report)
        self._review_scene(obs, report)
        preparation_s = time.monotonic()-started
        latency = super()._refresh(obs)
        started = time.monotonic()
        self.visual_goal_field.configure(obs.state, self.monitor, self.prediction,
                                         joint_position=obs.joint_position)
        self.feedback = self._observable_feedback()
        elapsed = preparation_s + time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self.visual_goal_field.observe(before.state, decision.action, after.state,
            monitor=self.monitor, prediction=self.prediction, joint_position=after.joint_position,
            acknowledged=(decision.diagnostics.get('controller') == 'visual_field'
                          and decision.diagnostics.get('selection', {}).get('accepted') is True))
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v31'
        if decision.diagnostics.get('refreshed'):
            decision.diagnostics['field_proposal_resolution'] = copy.deepcopy(
                (self.prediction or {}).get('field_proposal_resolution'))
            decision.diagnostics['scene_review'] = copy.deepcopy(self.scene_review)
        return decision
