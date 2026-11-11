"""Separate finite evidence acquisition from empirically supported intent execution.

The model's obstacle geometry is still a hypothesis. This opt-in lifecycle
changes only V27's final, fully validated transit budget branch. It does not
clear warnings, forgive physical exploration debt, or reuse old validation on
new headings. Every command requires fresh geometry and image/robot evidence.
"""
from __future__ import annotations

import copy
import json
import numpy as np

from .visual_translation_guard_v28 import ProjectedTranslationGuardV28


class ProjectedTranslationGuardV29(ProjectedTranslationGuardV28):
    version = 'validated-normal-intent-lifecycle-v29'
    supported_command_limit_m = .003
    _validated_budget_reasons = frozenset((
        'projected_ambiguous_extent_bounded_model_separating_transit',
        'translation_projected_conflict_measured_allowance_exhausted',
        'translation_recovery_direction_exhausted',
        'translation_recovery_direction_not_distinct',
        'translation_recovery_experiment_limit',
        'translation_recovery_direction_change_requires_review',
    ))

    def __init__(self):
        super().__init__()
        self._validated_pending = None
        self._validated_last_executed_key = None
        self._validated_rows = {}
        self.last_validated_intent_admission = None
        self.last_validated_intent_observation = None
        self.supported_execution_travel_m = 0.
        self.supported_execution_actions = 0
        self.supported_residual_hold_travel_m = 0.
        self.unmatched_execution_travel_m = 0.
        self.unmatched_residual_hold_travel_m = 0.
        self._unmatched_motion_active = False

    def configure(self, *args, **kwargs):
        # Refresh/reseed invalidates a provisional selection, never the last
        # executed class or any physical ledger (including residual holds).
        self._validated_pending = None
        return super().configure(*args, **kwargs)

    @staticmethod
    def _context(contract, monitor):
        origin = np.asarray(contract['fixed_origin_xyz_m'], float)
        goal = np.asarray(contract['fixed_goal_xyz_m'], float)
        intent = getattr(monitor, 'intent_id', None)
        if (origin.shape != (3,) or goal.shape != (3,) or intent is None
                or not np.isfinite(np.r_[origin, goal]).all()):
            raise ValueError('fixed intent unavailable')
        key = json.dumps([intent, contract['target_id']], sort_keys=True)
        return key, {'model_intent_key': intent, 'target_id': contract['target_id'],
            'fixed_origin_xyz_m': origin.tolist(), 'fixed_goal_xyz_m': goal.tolist(),
            'geometry_sha256': contract['geometry_sha256'],
            'scene_ids': copy.deepcopy(contract['scene_ids']), 'route_family': 'normal'}

    @staticmethod
    def _same_context(original, current):
        # Compare to the immutable first acknowledgement, never to the latest
        # numerically reconstructed goal; repeated tiny changes cannot creep.
        for key in original:
            if key in ('fixed_origin_xyz_m', 'fixed_goal_xyz_m'):
                if np.max(np.abs(np.asarray(original[key])-current[key])) > 2e-12:
                    return False
            elif original[key] != current.get(key):
                return False
        return True

    def _fully_validated_branch(self, base, command, kwargs):
        """Do not intercept any earlier contract, boundary or reference veto."""
        info = self.last_projected_warning_transit or {}
        if (base.get('reason') not in self._validated_budget_reasons
                or info.get('reason') != base.get('reason')
                or not info.get('geometry_current') or not self.worsening
                or set(info.get('per_object_response', {})) != set(self.worsening)
                or any(not row.get('validated') for row in info['per_object_response'].values())):
            return None
        # V28 can use the same physical-budget refusal strings during COLD
        # acquisition. Requiring V27's complete fresh per-object evidence is
        # what prevents those earlier acquisition failures from being promoted.
        contract = self._transit_contract(kwargs.get('nominal_regions') or [], command,
                                         float(kwargs.get('rotation_rad', 0.)))
        if contract is None or self.context_invalid or self.domain_invalid:
            return None
        return contract

    def _finish_supported(self, base, info, reason, *, accepted=False, cap=0.):
        self.last_validated_intent_admission = {**copy.deepcopy(info),
            'accepted': accepted, 'reason': reason, 'command_cap_m': float(cap)}
        self._recovery_reason = reason
        self._v12_last_result = {'accepted': accepted, 'cap_m': float(cap),
            'need_refresh': not accepted, 'reason': reason}
        result = {**self._v12_last_result, 'diagnostics': self.summary()}
        result['diagnostics']['checked_proposal'] = copy.deepcopy(
            base.get('diagnostics', {}).get('checked_proposal'))
        return result

    def assess(self, delta_xyz, **kwargs):
        self._validated_pending = None
        self.last_validated_intent_admission = None
        events = self.events
        base = super().assess(delta_xyz, **kwargs)
        command = np.asarray(delta_xyz, float)
        size = float(np.linalg.norm(command))
        contract = self._fully_validated_branch(base, command, kwargs)
        if contract is None:
            return base
        monitor = kwargs.get('monitor')
        try:
            key, context = self._context(contract, monitor)
        except (KeyError, TypeError, ValueError):
            return base
        # Only provisional dictionaries change during assessment/preview.
        self._motion_classification_pending = self._experiment_pending = None
        self._retreat_pending = self._heading_pending = None
        row = self._validated_rows.get(key)
        remaining = float(contract['remaining_measured_travel_m'])
        actions_remaining = int(contract['remaining_actions'])
        info = {'mode': 'conditional_empirically_validated_intent_execution',
            'base_reason': base['reason'], 'context': context,
            'warning_ids': sorted(self.worsening),
            'per_object_response': copy.deepcopy(self.last_projected_warning_transit['per_object_response']),
            'acquisition_shared_travel_m_unchanged': float(self.observation_travel_m),
            'acquisition_debt_replenished': False, 'clearance_certified': False}
        if row is not None:
            if not self._same_context(row['context'], context):
                return self._finish_supported(base, info, 'translation_validated_intent_binding_changed')
            # Current executor debt is authoritative for ALL actions. This
            # second bound makes replaying an older remaining-intent number
            # unable to renew this guard's already acknowledged execution.
            remaining = min(remaining, max(0., row['initial_remaining_travel_m']-row['measured_travel_m']))
            actions_remaining = min(actions_remaining,
                row['initial_remaining_actions']-row['acknowledged_actions'])
        direction = command/size
        for index, experiment in enumerate(self.experiments):
            adverse = self.heading_acquisition.rows.get(index, {}).get('adverse_response')
            if adverse and self._angle(direction, experiment['direction']) <= self.compatible_direction_rad+1e-9:
                return self._finish_supported(base, info, 'translation_validated_intent_prior_heading_contradiction')
        info.update(remaining_measured_travel_m=remaining, remaining_actions=actions_remaining)
        if actions_remaining <= 0 or remaining <= 0.:
            return self._finish_supported(base, info, 'translation_validated_intent_budget_exhausted')
        requested = min(size, self.supported_command_limit_m)
        conversion = kwargs['translation_response'].command_cap_for_measured_limit(
            direction, remaining, requested, self.state)
        cap = min(requested, float(conversion['command_cap_m']))
        info['conversion'] = copy.deepcopy(conversion)
        if not np.isfinite(cap) or cap < self.minimum_command_m-1e-9:
            return self._finish_supported(base, info, 'translation_validated_intent_budget_exhausted')
        self.events = events
        if size > cap+1e-12:
            # This is only a request to regenerate and recheck a shorter actual
            # candidate. It carries NO executable classification or ledger row.
            if size-cap <= 1e-10:
                cap = max(0., cap-2e-10)
            return self._finish_supported(base, info,
                'translation_validated_intent_candidate_requires_cap_reselection', accepted=True, cap=cap)
        self._motion_classification_pending = {'uncertain': False,
            'direction': direction.tolist(), 'command_cap_m': cap}
        self._validated_pending = {'key': key, 'context': copy.deepcopy(context),
            'state8': self.state.tolist(), 'observation_epoch': self.sweep_observation_epoch,
            'command_delta_m': command.tolist(), 'command_cap_m': cap,
            'rotation_margin_rad': max(.04, abs(float(kwargs.get('rotation_rad', 0.)))),
            'tool_track_identity': id(self.tool_track),
            'remaining_measured_travel_m': remaining, 'remaining_actions': actions_remaining}
        return self._finish_supported(base, info,
            'projected_warning_empirically_validated_fixed_intent_execution', accepted=True, cap=cap)

    def _supported_acknowledgement_extra_valid(self, pending):
        return True

    def observe(self, images, state8, action, *, monitor=None):
        before = None if self.state is None else self.state.copy()
        pending = self._validated_pending
        self._validated_pending = None
        action = np.asarray(action, float)
        command = action[:3]*.05
        nonzero = float(np.linalg.norm(command)) > 1e-8
        matched = bool(pending is not None and nonzero and before is not None
            and np.array_equal(pending['state8'], before)
            and pending['observation_epoch'] == self.sweep_observation_epoch
            and pending['tool_track_identity'] == id(self.tool_track)
            and np.allclose(pending['command_delta_m'], command, atol=1e-12, rtol=0.)
            and abs(float(action[6])) <= 1e-12
            and .5*float(np.linalg.norm(action[3:6])) <= pending['rotation_margin_rad']+1e-12
            and self._supported_acknowledgement_extra_valid(pending))
        mismatch = pending is not None and nonzero and not matched
        if mismatch:
            # Never allow the inherited looser direction/cap acknowledgement to
            # turn an unmatched command into debt-free supported execution.
            self._motion_classification_pending = self._experiment_pending = self._heading_pending = None
        previous_key = self._validated_last_executed_key
        previous_unmatched = self._unmatched_motion_active
        super().observe(images, state8, action, monitor=monitor)
        travel = 0. if before is None else float(np.linalg.norm(np.asarray(state8)[:3]-before[:3]))
        result = {'matched': matched, 'is_hold': not nonzero, 'measured_path_m': travel,
            'acquisition_debt_replenished': False, 'clearance_certified': False}
        if nonzero:
            self._validated_last_executed_key = pending['key'] if matched else None
            self._unmatched_motion_active = bool(mismatch and not self._charge_uncertain_motion)
        if ((mismatch and not self._charge_uncertain_motion)
                or (not nonzero and previous_unmatched)):
            # A contract violation is never supported-execution credit. Keep
            # its actual path separately when the previous executed class was
            # debt-free; the inherited observer already charged any previous
            # uncertain class to its original physical ledger.
            self.unmatched_execution_travel_m += travel
            if not nonzero:
                self.unmatched_residual_hold_travel_m += travel
            result['unmatched_execution_path_recorded'] = True
        key = pending['key'] if matched else (previous_key if not nonzero else None)
        if key is not None:
            if matched and key not in self._validated_rows:
                self._validated_rows[key] = {'context': copy.deepcopy(pending['context']),
                    'initial_remaining_travel_m': pending['remaining_measured_travel_m'],
                    'initial_remaining_actions': pending['remaining_actions'],
                    'measured_travel_m': 0., 'acknowledged_actions': 0,
                    'residual_hold_travel_m': 0., 'residual_hold_transitions': 0}
            row = self._validated_rows[key]
            row['measured_travel_m'] += travel
            row['acknowledged_actions'] += int(matched)
            self.supported_execution_travel_m += travel
            self.supported_execution_actions += int(matched)
            if not nonzero:
                row['residual_hold_travel_m'] += travel
                row['residual_hold_transitions'] += 1
                self.supported_residual_hold_travel_m += travel
            result['context_key'] = key
        if mismatch:
            self.domain_invalid = 'translation_validated_intent_acknowledgement_mismatch'
            result['reason'] = self.domain_invalid
        self.last_validated_intent_observation = result
        return self.summary()

    def summary(self):
        result = super().summary()
        if not hasattr(self, '_validated_rows'):
            return result
        result['validated_intent_execution'] = {
            'last_admission': copy.deepcopy(self.last_validated_intent_admission),
            'last_observation': copy.deepcopy(self.last_validated_intent_observation),
            'command_limit_m': self.supported_command_limit_m,
            'acknowledged_actions': self.supported_execution_actions,
            'measured_travel_m': self.supported_execution_travel_m,
            'residual_hold_travel_m': self.supported_residual_hold_travel_m,
            'unmatched_execution_travel_m': self.unmatched_execution_travel_m,
            'unmatched_residual_hold_travel_m': self.unmatched_residual_hold_travel_m,
            'intents': copy.deepcopy(list(self._validated_rows.values())),
            'pending_exact_acknowledgement': self._validated_pending is not None,
            'last_executed_supported': self._validated_last_executed_key is not None,
            'remaining_budget_authority': 'current fixed model intent action and measured-travel limits',
            'acquisition_fallback': 'remaining original heading/shared allowance; never renewed',
            'acquisition_debt_replenished': False, 'clearance_certified': False}
        return result
