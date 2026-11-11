"""Explicitly reviewed adverse-motion abort without renewing physical credit.

Only the inherited pre-checkpoint direction-change refusal is intercepted.
Geometry, finite reverse route, observed worsening, response and boundary
checks remain inherited. A preview proposes a model choice; only an exact
acknowledged nonzero command records that the abort actually happened.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import numpy as np

from .visual_translation_guard_v32 import ProjectedTranslationGuardV32


class ProjectedTranslationGuardV33(ProjectedTranslationGuardV32):
    version = 'reviewed-precheckpoint-adverse-motion-abort-v33'

    def __init__(self):
        self.reverse_abort_context = None
        self._reverse_abort_review = None
        self._reverse_abort_preview_context = None
        self._reverse_abort_scope = None
        self._reverse_abort_used = None
        self._reverse_abort_pending = None
        self.reverse_abort_events = []
        self.last_reverse_abort = None
        super().__init__()

    def configure(self, *args, **kwargs):
        self.reverse_abort_context = None
        self._reverse_abort_pending = None
        return super().configure(*args, **kwargs)

    def bind_reverse_abort_context(self, prediction, model_requests, *, fixed_goal_xyz=None):
        """Bind a current explicit model replacement to the executor's fixed goal.

        The raw relative goal is deliberately NOT added to each new state.
        Rebinding refreshes observation provenance, never route origin or debt.
        """
        p = prediction or {}
        motion = p.get('motion_context') or {}
        update = p.get('subgoal_update') or {}
        intent = p.get('field_intent') or {}
        refs = (p.get('visual_references') or {}).get('external') or {}
        monitor = getattr(self, 'egress_monitor', None)
        try:
            goal = np.asarray(fixed_goal_xyz, float)
            good = (self.state is not None and goal.shape == (3,) and np.isfinite(goal).all()
                and p.get('controller') == 'visual_field' and p.get('phase') == 'recover'
                and p.get('recovery_route') == 'reverse_observed'
                and p.get('target_id') == self.target_id and p.get('target_contact') == 'none'
                and p.get('contact_alignment') is None and refs.get('moving_kind') == 'tool'
                and intent.get('gripper_command') == 'hold'
                and motion.get('translation_mode') == 'separate'
                and motion.get('purpose') == 'reference_repair'
                and motion.get('intended_contact_id') is None
                and update.get('action') == 'replace'
                and isinstance(update.get('subgoal_id'), str) and bool(update['subgoal_id'])
                and isinstance(update.get('reason'), str) and bool(update['reason'].strip())
                and isinstance(model_requests, int) and not isinstance(model_requests, bool)
                and model_requests > 0 and monitor is not None
                and getattr(monitor, 'intent_id', None) == model_requests)
        except (ValueError, TypeError):
            good = False
        previous = self._reverse_abort_review
        if good and previous is not None:
            if model_requests < previous['model_requests']:
                good = False
            elif model_requests == previous['model_requests']:
                good = (previous['intent_id'] == monitor.intent_id
                    and previous['target_id'] == self.target_id
                    and previous['subgoal_id'] == update['subgoal_id']
                    and np.array_equal(previous['goal_xyz_m'], goal))
        if good and (previous is None or model_requests != previous['model_requests']):
            self._reverse_abort_review = {'model_requests': model_requests,
                'intent_id': monitor.intent_id, 'target_id': self.target_id,
                'subgoal_id': update['subgoal_id'], 'goal_xyz_m': goal.tolist(),
                'review_observation_epoch': self.sweep_observation_epoch,
                'reviewed_histories': copy.deepcopy({identity: self.worsening_motion.get(identity)
                    for identity in sorted(self.worsening)})}
        self.reverse_abort_context = None if not good else {
            'kind': 'actual_model_review', 'model_requests': model_requests,
            'review_observation_epoch': self._reverse_abort_review['review_observation_epoch'],
            'reviewed_histories': copy.deepcopy(self._reverse_abort_review['reviewed_histories']),
            'intent_id': monitor.intent_id, 'subgoal_id': update['subgoal_id'],
            'reason': update['reason'], 'target_id': self.target_id,
            'goal_xyz_m': goal.tolist(), 'state8': self.state.tolist(),
            'observation_epoch': self.sweep_observation_epoch}
        return {'bound': bool(good), 'physical_debt_modified': False,
                'clearance_certified': False}

    @contextmanager
    def preview_reverse_abort_context(self, state8, goal_offset_m):
        """Pure prospective replacement, usable only inside executor preview.

        Preserve live track identity and restore all provisional state even if
        the prospective executor raises. No preview can authorize a later act.
        """
        original = self.__dict__
        working = original.copy()
        working['_command_exhausted_experiments'] = set(self._command_exhausted_experiments)
        state, offset = np.asarray(state8, float), np.asarray(goal_offset_m, float)
        good = (state.shape == (8,) and offset.shape == (3,)
                and np.isfinite(np.r_[state, offset]).all()
                and np.array_equal(state, self.state)
                and 0 < np.linalg.norm(offset) <= .008+1e-10)
        working['_reverse_abort_preview_context'] = None if not good else {
            'kind': 'prospective_model_choice', 'target_id': self.target_id,
            'review_observation_epoch': self.sweep_observation_epoch,
            'reviewed_histories': copy.deepcopy({identity: self.worsening_motion.get(identity)
                for identity in sorted(self.worsening)}),
            'state8': state.tolist(), 'goal_xyz_m': (state[:3]+offset).tolist(),
            'observation_epoch': self.sweep_observation_epoch}
        try:
            self.__dict__ = working
            yield
        finally:
            self.__dict__ = original

    def _current_abort_scope(self, command, kwargs):
        route = self._finite_reverse_contract(kwargs.get('nominal_regions') or [], command,
                                               float(kwargs.get('rotation_rad', 0.)))
        context = self._reverse_abort_preview_context or self.reverse_abort_context
        if route is None or context is None:
            return None
        monitor = kwargs.get('monitor')
        try:
            good = (context['observation_epoch'] == self.sweep_observation_epoch
                and np.array_equal(context['state8'], self.state)
                and np.array_equal(context['goal_xyz_m'], route['goal_xyz_m'])
                and context['target_id'] == self.target_id
                and monitor is not None and np.array_equal(monitor.state, self.state)
                and (context['kind'] == 'prospective_model_choice'
                     or monitor.intent_id == context['intent_id'] == context['model_requests']))
        except (KeyError, ValueError, TypeError, AttributeError):
            good = False
        return {'route': route, 'context': copy.deepcopy(context)} if good else None

    def _experiment_for(self, direction, *, nominal_away):
        proposal, error = super()._experiment_for(direction, nominal_away=nominal_away)
        if error != 'translation_recovery_direction_change_requires_review':
            return proposal, error
        scope = self._reverse_abort_scope
        index = self._active_experiment()
        if scope is None or index is None or not self.worsening:
            return proposal, error
        row = self.experiments[index]
        histories = [self.worsening_motion.get(identity) for identity in sorted(self.worsening)]
        try:
            # These are paid physical directions, not visual-reference IDs.
            good = (row['measured_travel_m'] > 0 and row.get('acknowledged_actions', 0) > 0
                and bool(histories)
                and all(self.worsening_motion.get(identity) == scope['context']['reviewed_histories'].get(identity)
                        for identity in self.worsening)
                and all(h is not None
                    and h['measured_net_translation_m'] >= .001
                    and h['observation_epoch'] <= scope['context']['review_observation_epoch']
                    and h['geometry_sha256'] == scope['route']['geometry_sha256']
                    and self._angle(np.asarray(row['direction']), np.asarray(h['measured_direction_xyz']))
                        <= self.reverse_tolerance_rad+1e-9
                    and self._angle(direction, -np.asarray(h['measured_direction_xyz']))
                        <= self.reverse_tolerance_rad+1e-9 for h in histories))
        except (KeyError, TypeError, ValueError):
            good = False
        if not good:
            return proposal, error
        self._reverse_abort_used = {
            'source_experiment_index': index,
            'source_measured_travel_m': float(row['measured_travel_m']),
            'source_direction_xyz': copy.deepcopy(row['direction']),
            'worsening_ids': sorted(self.worsening),
            'context': copy.deepcopy(scope['context']),
            'route': copy.deepcopy(scope['route']),
            'overridden_reason': error, 'checkpoint_m': float(
                self._finite_reverse_original_checkpoint or self.information_checkpoint_m),
            'requires_explicit_model_choice': scope['context']['kind'] == 'prospective_model_choice'}
        return {'index': None, 'direction': direction.tolist(),
                'support_generation': self._support_generation}, None

    def assess(self, delta_xyz, **kwargs):
        command = np.asarray(delta_xyz, float)
        previous = self._reverse_abort_scope
        self._reverse_abort_pending = None
        self._reverse_abort_used = None
        self._reverse_abort_scope = self._current_abort_scope(command, kwargs)
        try:
            result = super().assess(delta_xyz, **kwargs)
        finally:
            self._reverse_abort_scope = previous
        used = self._reverse_abort_used
        self.last_reverse_abort = None if used is None else {
            **copy.deepcopy(used), 'accepted': bool(result['accepted']), 'reason': result['reason'],
            'physical_allowance_replenished': False, 'observed_worsening_erased': False,
            'clearance_certified': False}
        if used is not None and result['accepted']:
            # A smaller admitted cap requires caller reselection; only that
            # exact capped proposal is allowed to acquire acknowledgement.
            self._reverse_abort_pending = {**copy.deepcopy(used),
                'state8': self.state.copy(), 'delta_xyz_m': command.copy(),
                'cap_m': float(result['cap_m']), 'rotation_rad': float(kwargs.get('rotation_rad', 0.)),
                'observation_epoch': self.sweep_observation_epoch}
        result.setdefault('diagnostics', {})['precheckpoint_reverse_abort'] = self._abort_summary()
        return result

    def observe(self, images, state8, action, *, monitor=None):
        pending = self._reverse_abort_pending
        command = np.asarray(action, float)[:3]*.05
        moving = np.linalg.norm(command) > 1e-8
        current = self.reverse_abort_context
        matched = bool(pending is not None and moving
            and pending['context']['kind'] == 'actual_model_review'
            and pending['context'] == current
            and pending['observation_epoch'] == self.sweep_observation_epoch
            and np.array_equal(pending['state8'], self.state)
            and np.allclose(pending['delta_xyz_m'], command, atol=1e-10, rtol=0.)
            and np.linalg.norm(command) <= pending['cap_m']+1e-9
            and abs(float(action[6])) <= 1e-12
            and np.linalg.norm(np.asarray(action)[3:6])*.5 <= pending['rotation_rad']+1e-10
            and monitor is not None and monitor.intent_id == current['intent_id'])
        mismatch = pending is not None and moving and not matched
        if mismatch:
            # Prevent a stale/provisional admission from creating a direction.
            # Charge its real travel to the last uncertain physical row through
            # the existing observer, and never call it an acknowledged abort.
            self._experiment_pending = self._motion_classification_pending = self._retreat_pending = None
            self._charge_uncertain_motion = True
            self.domain_invalid = 'translation_reverse_abort_acknowledgement_mismatch'
        before = None if self.state is None else self.state.copy()
        super().observe(images, state8, action, monitor=monitor)
        if matched:
            self.reverse_abort_events.append({
                'source_experiment_index': pending['source_experiment_index'],
                'reverse_experiment_index': self._last_charged_experiment,
                'model_requests': pending['context']['model_requests'],
                'subgoal_id': pending['context']['subgoal_id'],
                'route_id': pending['route']['route_id'],
                'geometry_sha256': pending['route']['geometry_sha256'],
                'goal_xyz_m': pending['route']['goal_xyz_m'],
                'acknowledged_delta_xyz_m': command.tolist(),
                'measured_travel_m': float(np.linalg.norm(np.asarray(state8)[:3]-before[:3])),
                'shared_measured_travel_m_after': self.observation_travel_m,
                'prior_row_retained': True, 'physical_allowance_replenished': False})
        self._reverse_abort_pending = None
        if mismatch:
            self.domain_invalid = 'translation_reverse_abort_acknowledgement_mismatch'
        return self.summary()

    def _abort_summary(self):
        return {'last_admission': copy.deepcopy(self.last_reverse_abort),
            'acknowledged_aborts': copy.deepcopy(self.reverse_abort_events),
            'pending_exact_acknowledgement': self._reverse_abort_pending is not None,
            'physical_allowance_replenished': False,
            'current_full_geometry_and_visual_checks_unchanged': True,
            'clearance_certified': False}

    def summary(self):
        result = super().summary()
        result['precheckpoint_reverse_abort'] = self._abort_summary()
        result['recovery_experiments']['new_direction_requires'] += (
            '; V33 also permits explicit current model replacement by a checked finite reverse '
            "of the active paid heading's acknowledged adverse motion before that checkpoint; "
            'all prior debt and other checks remain')
        return result
