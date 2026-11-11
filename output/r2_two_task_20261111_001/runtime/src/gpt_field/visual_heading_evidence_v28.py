"""Finite, causal acquisition of a previously unobserved transit heading.

This helper proposes command sizes and records acknowledgements. The caller
owns geometry, visual admission, image/robot response estimators and ALL travel
budgets. In particular, this module neither trains an estimator nor copies old
held-out evidence into a new heading. A proposed action never spends or grants
physical travel. Recorded path below is an audit mirror, not a new allowance.
"""
from __future__ import annotations

import copy
import math
import numbers

import numpy as np


class HeadingEvidenceAcquisitionV28:
    version = 'finite-causal-normal-heading-acquisition-v28'
    cold_command_m = .001
    maximum_command_m = .003
    minimum_command_m = .001
    heading_budget_m = .008
    shared_budget_m = .020
    goal_serialization_tolerance_m = 2e-12
    command_scale_m = .05
    _prefix = 'translation_heading_evidence_'

    def __init__(self):
        self.rows = {}
        self.last_experiment_index = None
        self.last_observation = None
        self.acknowledged_actions = 0
        self.residual_hold_transitions = 0

    @staticmethod
    def _vector(value, size):
        raw = np.asarray(value, dtype=object)
        if raw.shape != (size,) or any(isinstance(x, (bool, np.bool_)) or
                not isinstance(x, numbers.Real) for x in raw):
            raise ValueError('expected finite real vector')
        result = np.asarray(value, float)
        if not np.isfinite(result).all():
            raise ValueError('expected finite real vector')
        return result

    @staticmethod
    def _amount(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError('expected nonnegative finite amount')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('expected nonnegative finite amount')
        return value

    @staticmethod
    def _key(value):
        """Stable, owned key; identities are provided by the current caller."""
        if isinstance(value, dict):
            return tuple((str(k), HeadingEvidenceAcquisitionV28._key(v))
                         for k, v in sorted(value.items(), key=lambda pair: str(pair[0])))
        if isinstance(value, (list, tuple)):
            return tuple(HeadingEvidenceAcquisitionV28._key(v) for v in value)
        if isinstance(value, np.ndarray):
            return HeadingEvidenceAcquisitionV28._key(value.tolist())
        if value is None or isinstance(value, (str, numbers.Real, bool)):
            return value
        raise ValueError('context key must contain serializable scalar identities')

    def _result(self, suffix, **kwargs):
        return {'accepted': False, 'reason': self._prefix+suffix,
            'command_cap_m': 0., 'cap_m': 0., 'pending': None,
            'clearance_certified': False, 'evidence_created': False,
            'physical_budget_allocated_m': 0., **kwargs}

    def requested_cap(self, state8, command_delta_m, *, heading_used_m,
                      shared_used_m, intent_remaining_m, intent_remaining_actions,
                      robot_response):
        """Pure command construction; this does NOT authorize the candidate.

        Invert the empirical *measured* allowance before testing geometry. Four
        bounded 1 mm cold commands can establish the robot response without
        consuming the entire 8 mm heading budget at unit gain. Subsequent scale
        growth is exclusively the existing estimator's <=1.5x observed range.
        """
        try:
            state = self._vector(state8, 8)
            command = self._vector(command_delta_m, 3)
            norm = float(np.linalg.norm(command))
            used = self._amount(heading_used_m)
            shared = self._amount(shared_used_m)
            intent = self._amount(intent_remaining_m)
            actions = self._amount(intent_remaining_actions)
            if not actions.is_integer():
                raise ValueError('remaining actions must be integral')
        except (TypeError, ValueError):
            return self._result('current_binding_invalid', candidate_only=True)
        if norm <= 1e-12:
            return self._result('no_useful_command', candidate_only=True)
        remaining = max(0., min(self.heading_budget_m-used,
            self.shared_budget_m-shared, intent))
        if actions <= 0 or remaining <= 0:
            return self._result('budget_exhausted', candidate_only=True,
                remaining_measured_m=remaining)
        requested = min(norm, self.maximum_command_m)
        try:
            conversion = robot_response.command_cap_for_measured_limit(
                command/norm, remaining, requested, state)
            prediction = conversion.get('prediction') or {}
            if prediction.get('responsive') is False:
                return self._result('observed_robot_nonresponsive', candidate_only=True,
                    conversion=copy.deepcopy(conversion), remaining_measured_m=remaining)
            cap = min(requested, self._amount(conversion['command_cap_m']))
            if conversion.get('supported') is not True:
                cap = min(cap, self.cold_command_m)
            if cap < self.minimum_command_m-1e-12:
                return self._result('no_useful_command', candidate_only=True,
                    conversion=copy.deepcopy(conversion), remaining_measured_m=remaining)
            # Query exactly the returned command, never an unsupported discarded
            # 3 mm tail. Predict is pure and adds neither samples nor validation.
            prediction = robot_response.predict(command/norm*cap, state)
            if prediction.get('responsive') is False:
                return self._result('observed_robot_nonresponsive', candidate_only=True,
                    conversion=copy.deepcopy(conversion), prediction=copy.deepcopy(prediction),
                    remaining_measured_m=remaining)
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._result('robot_response_unavailable', candidate_only=True)
        return self._result('candidate_supported_scale' if conversion.get('supported')
            else 'candidate_cold_scale', usable=True, candidate_only=True,
            command_cap_m=float(cap), cap_m=float(cap), remaining_measured_m=remaining,
            command_delta_m=(command/norm*cap).tolist(), conversion=copy.deepcopy(conversion),
            prediction=copy.deepcopy(prediction), budget_source='caller_existing_physical_ledgers')

    def qualification(self, image_evidence, warning_ids, robot_prediction):
        """Inspect evidence, never synthesize/transfer held-out validation."""
        rows = image_evidence if isinstance(image_evidence, dict) else {}
        contradicted = [identity for identity in warning_ids
                        if isinstance(rows.get(identity), dict) and
                        rows[identity].get('contradicted') is True]
        missing = []
        for identity in warning_ids:
            row = rows.get(identity)
            if not isinstance(row, dict) or not (row.get('validated') is True and
                    row.get('held_out_count', 0) >= 2 and
                    row.get('current_supported_heading') is True and
                    row.get('recent_measured_sample') is True):
                missing.append(identity)
        robot_valid = robot_prediction.get('supported') is True and robot_prediction.get('responsive') is True
        return {'ready': bool(robot_valid and not missing and not contradicted),
            'contradicted_ids': contradicted, 'awaiting_object_ids': missing,
            'robot_response_ready': bool(robot_valid),
            'historical_validation_credit': False, 'clearance_certified': False}

    def _binding(self, state, command, binding):
        try:
            goal = self._vector(binding['fixed_goal_xyz_m'], 3)
            origin = self._vector(binding['fixed_origin_xyz_m'], 3)
            ids = binding['warning_ids']
            if (not isinstance(ids, (list, tuple)) or not ids or
                    any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids))):
                return None
            valid = (binding['eligible'] is True and binding['route_family'] == 'normal'
                and binding['empty_tool'] is True and binding['target_contact'] == 'none'
                and binding['contact_or_visual_goal_dependency'] is False
                and binding['full_scene_sweep_accepted'] is True
                and binding['hard_segment_clear'] is True
                and binding['outside_target_contact_band'] is True
                and np.array_equal(self._vector(binding['state8'], 8), state)
                and np.array_equal(self._vector(binding['start_xyz_m'], 3), state[:3])
                and np.allclose(self._vector(binding['end_xyz_m'], 3), state[:3]+command,
                    atol=1e-10, rtol=0.)
                and isinstance(binding['observation_epoch'], numbers.Integral)
                and not isinstance(binding['observation_epoch'], bool)
                and isinstance(binding['geometry_sha256'], str) and bool(binding['geometry_sha256'])
                and binding.get('model_intent_key') is not None
                and binding.get('representation_key') is not None)
            if not valid:
                return None
            return {'model_intent_key': self._key(binding['model_intent_key']),
                'representation_key': self._key(binding['representation_key']),
                'geometry_sha256': binding['geometry_sha256'], 'fixed_goal_xyz_m': goal.tolist(),
                'fixed_origin_xyz_m': origin.tolist(), 'warning_ids': list(ids),
                'observation_epoch': int(binding['observation_epoch'])}
        except (TypeError, ValueError, KeyError):
            return None

    def assess(self, state8, command_delta_m, *, binding, experiment,
               heading_used_m, shared_used_m, intent_remaining_m,
               intent_remaining_actions, robot_response, image_evidence):
        """Pure admission adjunct AFTER the caller's complete safety gates.

        ``experiment`` is an existing physical row index or None for the row
        which the caller may create on acknowledgement. Context/representation
        revision changes only evidence segmentation; the caller's remaining
        physical allowance and this row's adverse robot history remain intact.
        """
        try:
            state = self._vector(state8, 8)
            command = self._vector(command_delta_m, 3)
            if experiment is not None and (isinstance(experiment, bool) or
                    not isinstance(experiment, numbers.Integral) or experiment < 0):
                raise ValueError('invalid physical experiment index')
        except (TypeError, ValueError):
            return self._result('current_binding_invalid')
        context = self._binding(state, command, binding)
        if context is None:
            return self._result('current_binding_invalid')
        row = self.rows.get(experiment)
        if row is not None:
            if row['adverse_response']:
                return self._result('observed_robot_nonresponsive', experiment_index=int(experiment),
                    adverse_response=copy.deepcopy(row['adverse_response']))
            old = row['context']
            if context['model_intent_key'] == old['model_intent_key']:
                if (context['geometry_sha256'] != old['geometry_sha256'] or
                        not np.array_equal(context['fixed_origin_xyz_m'], old['fixed_origin_xyz_m']) or
                        np.max(abs(np.asarray(context['fixed_goal_xyz_m'])-
                                   old['fixed_goal_xyz_m'])) > self.goal_serialization_tolerance_m):
                    return self._result('geometry_or_domain_changed')
                # Compare every retained serialization to the original value,
                # never update a rounded goal into a creeping comparison base.
                context['fixed_goal_xyz_m'] = copy.deepcopy(old['fixed_goal_xyz_m'])
        cap = self.requested_cap(state, command, heading_used_m=heading_used_m,
            shared_used_m=shared_used_m, intent_remaining_m=intent_remaining_m,
            intent_remaining_actions=intent_remaining_actions, robot_response=robot_response)
        if not cap.get('usable'):
            return cap
        qualification = self.qualification(image_evidence, context['warning_ids'], cap['prediction'])
        if qualification['contradicted_ids']:
            return self._result('evidence_contradicted', qualification=qualification)
        norm = float(np.linalg.norm(command))
        if norm > cap['command_cap_m']+1e-12:
            # Do not acknowledge a cap of a larger, differently proven segment.
            return self._result('candidate_requires_cap_reselection', usable=True,
                command_cap_m=cap['command_cap_m'], cap_m=cap['command_cap_m'],
                qualification=qualification, conversion=cap['conversion'])
        mode = 'validated_transit_ready' if qualification['ready'] else (
            'admitted_supported_command' if cap['conversion'].get('supported') else 'admitted_cold_command')
        pending = {'version': self.version, 'before_state8': state.tolist(),
            'command_delta_m': command.tolist(), 'experiment_index': experiment,
            'context': copy.deepcopy(context), 'mode': mode,
            'qualification': copy.deepcopy(qualification), 'command_cap_m': cap['command_cap_m']}
        return self._result(mode, accepted=True, command_cap_m=cap['command_cap_m'], cap_m=cap['command_cap_m'],
            pending=pending, qualification=qualification, prediction=cap['prediction'],
            conversion=cap['conversion'], remaining_measured_m=cap['remaining_measured_m'],
            evidence_segment_changed=bool(row is not None and (
                context['model_intent_key'] != row['context']['model_intent_key'] or
                context['representation_key'] != row['context']['representation_key'])))

    def observe(self, before8, action7, after8, *, pending=None, experiment_index=None):
        """Mirror exactly one actual acknowledgement AFTER parent accounting."""
        before = self._vector(before8, 8)
        after = self._vector(after8, 8)
        action = self._vector(action7, 7)
        if np.max(abs(action)) > 1.+1e-12:
            raise ValueError('normalized action outside [-1,1]')
        command = action[:3]*self.command_scale_m
        measured = after[:3]-before[:3]
        command_norm = float(np.linalg.norm(command))
        travel = float(np.linalg.norm(measured))
        result = {'matched': False, 'is_hold': command_norm <= 1e-8,
            'measured_path_m': travel, 'evidence_created': False,
            'physical_budget_allocated_m': 0., 'clearance_certified': False}
        if result['is_hold']:
            # A downstream veto can replace a provisional new-heading command
            # with a hold. It cannot create that row or overwrite attribution.
            row = self.rows.get(self.last_experiment_index)
            if row is not None:
                row['measured_path_m'] += travel
                row['residual_hold_path_m'] += travel
                row['residual_hold_transitions'] += 1
                self.residual_hold_transitions += 1
                result['experiment_index'] = self.last_experiment_index
            self.last_observation = copy.deepcopy(result)
            return result
        try:
            valid = (isinstance(pending, dict) and pending['version'] == self.version
                and isinstance(experiment_index, numbers.Integral) and
                not isinstance(experiment_index, bool) and experiment_index >= 0
                and pending['experiment_index'] in (None, experiment_index)
                and np.array_equal(pending['before_state8'], before)
                and np.allclose(pending['command_delta_m'], command, atol=1e-12, rtol=0.)
                and command_norm <= pending['command_cap_m']+1e-12)
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            self.last_experiment_index = None
            result['reason'] = self._prefix+('acknowledgement_mismatch' if pending is not None
                                          else 'unrelated_motion')
            self.last_observation = copy.deepcopy(result)
            return result
        index = int(experiment_index)
        context = copy.deepcopy(pending['context'])
        row = self.rows.get(index)
        if row is None:
            row = {'experiment_index': index, 'context': context, 'segments': 1,
                'acknowledged_actions': 0, 'measured_path_m': 0., 'command_path_m': 0.,
                'residual_hold_path_m': 0., 'residual_hold_transitions': 0,
                'recent_robot_observations': [], 'adverse_response': None}
            self.rows[index] = row
        elif (context['model_intent_key'] != row['context']['model_intent_key'] or
                context['representation_key'] != row['context']['representation_key']):
            row['segments'] += 1
            row['context'] = context
        row['acknowledged_actions'] += 1
        row['measured_path_m'] += travel
        row['command_path_m'] += command_norm
        direction = command/command_norm
        longitudinal = float(measured@direction)
        row['recent_robot_observations'].append({'direction': direction.tolist(),
            'command_norm_m': command_norm, 'measured_norm_m': travel,
            'longitudinal_m': longitudinal, 'gain': longitudinal/command_norm})
        row['recent_robot_observations'] = row['recent_robot_observations'][-4:]
        recent = row['recent_robot_observations']
        # Tiny first residuals are inconclusive. Four coherent acknowledged
        # samples detect sustained blockage/opposition before further scaling;
        # keep this evidence across later model/ref/representation revisions.
        coherent = len(recent) == 4 and all(float(np.dot(x['direction'], direction))
                    >= math.cos(math.radians(15.))-1e-12 for x in recent)
        if coherent and max(x['gain'] for x in recent) <= .02:
            row['adverse_response'] = {'reason': 'observed_opposed_or_nonresponsive_heading',
                'sample_count': 4, 'maximum_longitudinal_gain': max(x['gain'] for x in recent),
                'preserved_across_representation_or_intent_changes': True}
        row['last_qualification'] = copy.deepcopy(pending['qualification'])
        self.last_experiment_index = index
        self.acknowledged_actions += 1
        result.update(matched=True, experiment_index=index, context_segments=row['segments'],
            mode=pending['mode'], adverse_response=copy.deepcopy(row['adverse_response']))
        self.last_observation = copy.deepcopy(result)
        return result

    def summary(self):
        return {'version': self.version, 'cold_command_m': self.cold_command_m,
            'maximum_command_m': self.maximum_command_m, 'minimum_command_m': self.minimum_command_m,
            'physical_budget_authority': 'parent_guard_existing_heading_shared_and_intent_ledgers',
            'physical_budget_allocated_m': 0., 'evidence_training_performed': False,
            'historical_validation_credit': False, 'clearance_certified': False,
            'acknowledged_actions': self.acknowledged_actions,
            'residual_hold_transitions': self.residual_hold_transitions,
            'last_experiment_index': self.last_experiment_index,
            'experiments': copy.deepcopy(list(self.rows.values())),
            'last_observation': copy.deepcopy(self.last_observation)}
