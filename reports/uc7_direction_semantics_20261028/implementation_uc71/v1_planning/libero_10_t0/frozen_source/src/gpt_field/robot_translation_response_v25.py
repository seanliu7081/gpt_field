"""Ordered evidence for a recently recovering measured translation response.

The historical empirical envelope is unchanged. A short, explicitly labeled
recent-response regime can replace only the pooled nonresponse decision, after
actual contiguous feedback shows two positive, nondeclining responses whose
combined own/query projected travel exceeds the summed measurement floors. It does
not identify a physical cause, certify clearance, amplify a requested command,
issue settling actions, or create additional travel allowance.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from .robot_translation_response import MeasuredTranslationResponse


class MeasuredTranslationResponseV25(MeasuredTranslationResponse):
    version = 'ordered-recent-robot-translation-response-v25'
    recent_regime_samples = 2
    positive_gain_threshold = .02  # Same threshold as the inherited estimator.

    def reset(self):
        super().reset()
        self._ordered_recent = deque(maxlen=self.recent_regime_samples)

    def observe(self, before_state8, action7, after_state8, *, before_joints=None, after_joints=None):
        report = super().observe(before_state8, action7, after_state8,
                                 before_joints=before_joints, after_joints=after_joints)
        if report['domain_reset_reason'] is not None:
            self._ordered_recent.clear()
        self._ordered_recent.append({
            'sequence': self.transitions,
            'sample_added': report['sample_added'],
            'is_hold': report['is_hold'],
            'command': np.asarray(report['command_delta_m'], float),
            'measured': np.asarray(report['measured_delta_m'], float),
            'command_norm': report['command_norm_m'],
        })
        return report

    def _recent_regime(self, direction, state, joints):
        result = {'eligible': False, 'reason': 'recent_response_not_available',
                  'minimum_contiguous_samples': self.recent_regime_samples,
                  'positive_gain_threshold': self.positive_gain_threshold,
                  'per_transition_measurement_floor_m': self.measurement_floor_m,
                  'minimum_aggregate_projected_displacement_m': self.recent_regime_samples*self.measurement_floor_m,
                  'samples': [], 'clearance_certified': False,
                  'physical_cause_identified': False, 'allowance_added_m': 0.}
        if (state is None or self.last_state is None
                or not np.allclose(state, self.last_state, atol=1e-8, rtol=0.)
                or (joints is None) != (self.last_joints is None)
                or (joints is not None and not np.allclose(joints, self.last_joints, atol=1e-8, rtol=0.))):
            result['reason'] = 'recent_response_requires_current_acknowledged_endpoint'
            return result
        if len(self._ordered_recent) != self.recent_regime_samples:
            return result
        previous = None
        gains = []
        projected_motion = []
        for row in self._ordered_recent:
            if not row['sample_added'] or row['is_hold']:
                result['reason'] = 'recent_response_interrupted_by_hold_or_invalid_sample'
                return result
            if previous is not None and row['sequence'] != previous+1:
                result['reason'] = 'recent_response_not_contiguous'
                return result
            previous = row['sequence']
            cosine = float(row['command'] @ direction)/row['command_norm']
            if cosine < math.cos(self.direction_cone_rad)-1e-12:
                result['reason'] = 'recent_response_direction_changed'
                return result
            gain = float(row['command'] @ row['measured'])/row['command_norm']**2
            own_progress = gain*row['command_norm']
            query_progress = float(row['measured'] @ direction)
            result['samples'].append({'sequence': row['sequence'], 'gain': gain,
                                      'command_norm_m': row['command_norm'],
                                      'longitudinal_motion_m': own_progress,
                                      'query_direction_motion_m': query_progress})
            if (gain <= self.positive_gain_threshold
                    or min(own_progress, query_progress) <= 0.):
                result['reason'] = 'recent_response_not_distinguishably_positive'
                return result
            gains.append(gain)
            projected_motion.append(min(own_progress, query_progress))
        if gains[-1] < gains[-2]-1e-12:
            result['reason'] = 'recent_response_is_declining'
            return result
        result['aggregate_projected_motion_m'] = float(sum(projected_motion))
        if sum(projected_motion) <= self.recent_regime_samples*self.measurement_floor_m:
            result['reason'] = 'recent_response_aggregate_not_distinguishable'
            return result
        result.update(eligible=True, reason='two_fresh_positive_nondeclining_responses',
                      recent_mean_gain=float(np.median(gains)),
                      interpretation='Current ordered feedback is consistent with a responsive regime; it is not a guarantee of continued response.')
        return result

    def _historical_profile(self, samples, direction):
        """Exact inherited envelope formula; no old opposed/tail sample dropped."""
        gains = np.asarray([float(s['command'] @ s['measured'])/s['command_norm']**2 for s in samples])
        norm_gains = np.asarray([s['measured_norm']/s['command_norm'] for s in samples])
        gain_pad = self.gain_margin+.1*float(np.ptp(gains))
        residual = max(float(np.linalg.norm(s['measured']-g*s['command']))/s['command_norm']
                       for s, g in zip(samples, gains))
        angle = max(math.acos(np.clip(float(s['command'] @ direction)/s['command_norm'], -1., 1.))
                    for s in samples)
        return {'observed_max_command_m': max(s['command_norm'] for s in samples),
                'noise_m': max(self.measurement_floor_m, self.hold_drift_m),
                'norm_gain_upper': float(max(norm_gains))+self.gain_margin+.1*float(np.ptp(norm_gains)),
                'longitudinal_low_gain': float(min(gains))-gain_pad,
                'longitudinal_high_gain': float(max(gains))+gain_pad,
                'lateral_gain_upper': residual+self.gain_margin+float(max(abs(gains)))*math.sin(angle)}

    def predict(self, command_delta_m, state8=None, *, joint_position=None):
        result = super().predict(command_delta_m, state8, joint_position=joint_position)
        if result['reason'] != 'robot_response_unresponsive_or_opposed':
            return result
        command = np.asarray(command_delta_m, float)
        norm = float(np.linalg.norm(command))
        direction = command/norm
        state, joints = self._context(state8, joint_position)
        regime = self._recent_regime(direction, state, joints)
        result['recent_response_regime'] = regime
        if not regime['eligible']:
            return result
        samples = self._matching(direction)
        profile = self._historical_profile(samples, direction)
        low, high, lateral, upper, probe = self._envelope_at(norm, profile)
        scales = [s['command_norm'] for s in samples]
        result.update(supported=True, responsive=True,
            reason='robot_response_recent_responsive_regime',
            inherited_responsiveness_decision='robot_response_unresponsive_or_opposed',
            responsiveness_basis='fresh_contiguous_ordered_feedback_not_pooled_median',
            mean_delta_m=(regime['recent_mean_gain']*command).tolist(),
            mean_regime_longitudinal_gain=regime['recent_mean_gain'],
            max_measured_norm_m=upper, longitudinal_interval_m=[low, high], lateral_radius_m=lateral,
            measured_component_lower_m=(np.minimum(direction*low, direction*high)-lateral).tolist(),
            measured_component_upper_m=(np.maximum(direction*low, direction*high)+lateral).tolist(),
            measurement_and_hold_drift_margin_m=profile['noise_m'],
            interpolated_below_observed_scale=norm<min(scales)-1e-12,
            bounded_scale_probe=norm>max(scales)+1e-12, scale_probe_margin_m=probe,
            empirical_envelope_parameters=profile,
            envelope_geometry='unchanged full-history longitudinal/lateral and radial envelopes',
            envelope_sample_count=len(samples), prior_samples_discarded_for_regime=0)
        return result

    def summary(self):
        result = super().summary()
        result['recent_response_regime'] = {
            'maximum_history_transitions': self.recent_regime_samples,
            'retained_transition_sequences': [r['sequence'] for r in self._ordered_recent],
            'holds_or_invalid_samples_break_evidence': True,
            'all_original_empirical_envelope_samples_retained': True,
            'queries_create_evidence': False, 'physical_allowance_added_m': 0.}
        return result
