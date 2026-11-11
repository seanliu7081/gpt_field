"""Local command-to-measured-translation response from robot feedback only.

A normalized OSC XYZ command times 0.05 is a *commanded* increment; it is not
one measured robot step. This helper keeps those units distinct. It estimates
one scalar longitudinal response for nearby command directions, with an
empirical range and lateral residual envelope. It does not infer contacts,
obstacles, camera calibration, payload attachment or task completion.

The interval is an empirical uncertainty envelope, not a guaranteed robot
reachability bound or calibrated probability. Limited downscaling assumes a
locally linear response anchored at zero command; holds are never fit as
successful motion samples. Scale probes are bounded to 1.5 times the largest observed command and carry
additional uncertainty; they are explicitly distinguished from observed scale.
"""
from __future__ import annotations

from collections import deque
import math
from numbers import Real

import numpy as np


class MeasuredTranslationResponse:
    version = 'robot-command-measured-response-v1'
    command_scale_m = .05
    minimum_samples = 4
    maximum_samples = 24
    direction_cone_rad = math.radians(15.)
    minimum_scale_fraction = .5
    maximum_scale_fraction = 1.5
    hold_window_transitions = 8
    orientation_domain_rad = .015
    joint_domain_rad = .12  # maximum absolute joint change from the local anchor
    finger_domain_m = .002
    position_domain_m = .04
    measurement_floor_m = .00005
    gain_margin = .02
    cold_command_cap_m = .002

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.maximum_samples)
        self.anchor_state = self.anchor_joints = None
        self.last_state = self.last_joints = None
        self.transitions = self.holds = self.nonzero_samples = 0
        self.domain_resets = 0
        self.hold_drift_m = 0.
        self.all_time_hold_drift_m = 0.
        self._recent_holds = deque(maxlen=self.hold_window_transitions)
        self.last_reset_reason = None
        self.last_observation = None

    @staticmethod
    def _vector(value, size, name):
        raw = np.asarray(value, dtype=object)
        if raw.shape != (size,) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in raw):
            raise ValueError(f'{name} must contain {size} finite real numbers')
        result = np.asarray(raw, float)
        if not np.isfinite(result).all():
            raise ValueError(f'{name} must contain {size} finite real numbers')
        return result

    @staticmethod
    def _scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f'{name} must be finite and nonnegative')
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
        return number

    @staticmethod
    def _rotation_distance(a, b):
        def quaternion(vector):
            angle = float(np.linalg.norm(vector))
            factor = math.sin(angle/2)/angle if angle else .5
            return np.r_[vector*factor, math.cos(angle/2)]
        qa, qb = quaternion(a), quaternion(b)
        # Sign-invariant and stable near equal orientations and the pi branch.
        return float(4*math.asin(min(1., min(np.linalg.norm(qa-qb),np.linalg.norm(qa+qb))/2)))

    def _domain_reason(self, state, joints):
        if self.anchor_state is None:
            return 'robot_response_uninitialized'
        if self._rotation_distance(state[3:6],self.anchor_state[3:6]) > self.orientation_domain_rad+1e-12:
            return 'robot_response_orientation_domain_changed'
        if np.max(abs(state[6:]-self.anchor_state[6:])) > self.finger_domain_m+1e-12:
            return 'robot_response_finger_domain_changed'
        if np.linalg.norm(state[:3]-self.anchor_state[:3]) > self.position_domain_m+1e-12:
            return 'robot_response_position_domain_changed'
        if (joints is None) != (self.anchor_joints is None):
            return 'robot_response_joint_feedback_domain_changed'
        if joints is not None and np.max(abs(joints-self.anchor_joints)) > self.joint_domain_rad+1e-12:
            return 'robot_response_joint_domain_changed'
        return None

    def _new_domain(self, state, joints, reason):
        if self.anchor_state is not None:
            self.domain_resets += 1
        self.samples.clear()
        self._recent_holds.clear()
        self.hold_drift_m = 0.
        self.anchor_state = state.copy()
        self.anchor_joints = None if joints is None else joints.copy()
        self.last_reset_reason = reason

    def observe(self, before_state8, action7, after_state8, *, before_joints=None, after_joints=None):
        """Account one acknowledged transition; never issue or alter an action.

        Optional joints must be supplied as a pair. A changed robot domain drops
        old samples. A transition that itself leaves the domain is not reused
        for the next one; its actual movement remains reported in diagnostics.
        """
        before = self._vector(before_state8,8,'before_state8')
        after = self._vector(after_state8,8,'after_state8')
        action = self._vector(action7,7,'action7')
        if np.any(abs(action)>1.+1e-12):
            raise ValueError('action7 must be normalized within [-1,1]')
        if (before_joints is None) != (after_joints is None):
            raise ValueError('before_joints and after_joints must both be supplied or both omitted')
        qb = None if before_joints is None else self._vector(before_joints,7,'before_joints')
        qa = None if after_joints is None else self._vector(after_joints,7,'after_joints')
        reset_reason = self._domain_reason(before,qb)
        if self.last_state is not None and not np.allclose(before,self.last_state,atol=1e-8,rtol=0.):
            reset_reason = 'robot_response_observation_discontinuity'
        if self.last_joints is not None and qb is not None and not np.allclose(qb,self.last_joints,atol=1e-8,rtol=0.):
            reset_reason = 'robot_response_joint_observation_discontinuity'
        if reset_reason:
            self._new_domain(before,qb,reset_reason)
        command = action[:3]*self.command_scale_m
        measured = after[:3]-before[:3]
        command_norm = float(np.linalg.norm(command))
        measured_norm = float(np.linalg.norm(measured))
        after_reason = self._domain_reason(after,qa)
        if after_reason:
            self._new_domain(after,qa,after_reason)
            reset_reason = after_reason
        is_hold = command_norm <= 1e-8
        sample_added = not is_hold and after_reason is None
        if sample_added:
            self.samples.append({'command':command.copy(),'measured':measured.copy(),
                                 'command_norm':command_norm,'measured_norm':measured_norm})
            self.nonzero_samples += 1
        elif is_hold:
            self.holds += 1
            self.all_time_hold_drift_m = max(self.all_time_hold_drift_m, measured_norm)
        self._recent_holds.append(measured_norm if is_hold else 0.)
        self.hold_drift_m = max(self._recent_holds, default=0.)
        self.transitions += 1
        self.last_state = after.copy()
        self.last_joints = None if qa is None else qa.copy()
        self.last_observation = {'sample_added':sample_added,'is_hold':is_hold,
            'command_delta_m':command.tolist(),'measured_delta_m':measured.tolist(),
            'command_norm_m':command_norm,'measured_norm_m':measured_norm,
            'domain_reset_reason':reset_reason,'clearance_certified':False}
        return dict(self.last_observation)

    def _context(self, state8, joint_position):
        if state8 is None:
            state = self.last_state
        else:
            state = self._vector(state8,8,'state8')
        if joint_position is not None:
            joints = self._vector(joint_position,7,'joint_position')
        elif state is not None and self.last_state is not None and np.allclose(state,self.last_state,atol=1e-8,rtol=0.):
            joints = self.last_joints
        else:
            joints = None
        return state,joints

    def _matching(self, direction):
        cosine = math.cos(self.direction_cone_rad)
        return [s for s in self.samples if float(s['command']@direction)/s['command_norm'] >= cosine-1e-12]

    @staticmethod
    def _envelope_at(norm, profile):
        probe=max(0.,norm-profile['observed_max_command_m'])*profile['norm_gain_upper']
        low=profile['longitudinal_low_gain']*norm-profile['noise_m']-probe
        high=profile['longitudinal_high_gain']*norm+profile['noise_m']+probe
        lateral=profile['lateral_gain_upper']*norm+profile['noise_m']+probe
        radial=profile['norm_gain_upper']*norm+profile['noise_m']+probe
        upper=min(math.hypot(max(abs(low),abs(high)),lateral),radial)
        return low,high,lateral,upper,probe

    def predict(self, command_delta_m, state8=None, *, joint_position=None):
        """Predict measured XYZ from commanded metres, not normalized actions.

        Support is local to robot configuration, direction and observed command
        scale. Half the smallest observed command permits labeled anchored
        downscaling. At most 1.5 times the largest observed command permits a
        labeled scale probe with added uncertainty. Further expansion requires
        an actual acknowledged observation. Prediction never adds evidence.
        """
        command = self._vector(command_delta_m,3,'command_delta_m')
        if np.any(abs(command)>self.command_scale_m+1e-12):
            raise ValueError('command_delta_m exceeds normalized OSC axis limits')
        norm = float(np.linalg.norm(command))
        state,joints = self._context(state8,joint_position)
        result = {'supported':False,'responsive':None,'mean_delta_m':None,
            'max_measured_norm_m':None,'longitudinal_interval_m':None,'lateral_radius_m':None,
            'measured_component_lower_m':None,'measured_component_upper_m':None,
            'command_scale_range_m':None,'observed_command_scale_range_m':None,
            'command_norm_m':norm,'sample_count':0,
            'reason':'robot_response_uninitialized','uncertainty_kind':'empirical_range_plus_residual_margin',
            'guaranteed_bound':False,'clearance_certified':False,'task_success_certified':False,
            'scale_assumption':'local scalar response; downscale to 0.5*observed_min or explicit uncertain probe up to 1.5*observed_max'}
        if norm <= 1e-8:
            result.update(reason='zero_command_not_response_evidence',mean_delta_m=[0.,0.,0.])
            return result
        reason = 'robot_response_uninitialized' if state is None else self._domain_reason(state,joints)
        if reason:
            result['reason']=reason
            return result
        direction=command/norm
        samples=self._matching(direction)
        result['sample_count']=len(samples)
        if len(samples)<self.minimum_samples:
            result['reason']='robot_response_direction_not_observed_enough'
            return result
        scales=np.asarray([s['command_norm']for s in samples])
        lower_scale,upper_scale=float(min(scales)*self.minimum_scale_fraction),float(max(scales)*self.maximum_scale_fraction)
        result['command_scale_range_m']=[lower_scale,upper_scale]
        result['observed_command_scale_range_m']=[float(min(scales)),float(max(scales))]
        if norm<lower_scale-1e-12 or norm>upper_scale+1e-12:
            result['reason']='robot_response_command_scale_unsupported'
            return result
        gains=np.asarray([float(s['command']@s['measured'])/s['command_norm']**2 for s in samples])
        median_gain=float(np.median(gains))
        recent=samples[-3:]
        recent_gain=float(np.median([float(s['command']@s['measured'])/s['command_norm']**2 for s in recent]))
        responsive=bool(median_gain>.02 and recent_gain>.02)
        result['responsive']=responsive
        if not responsive:
            result.update(reason='robot_response_unresponsive_or_opposed',median_longitudinal_gain=median_gain,
                          recent_longitudinal_gain=recent_gain)
            return result
        gain_pad=self.gain_margin+.1*float(np.ptp(gains))
        noise=max(self.measurement_floor_m,self.hold_drift_m)
        residual=max(float(np.linalg.norm(s['measured']-g*s['command']))/s['command_norm']
                     for s,g in zip(samples,gains))
        largest_angle=max(math.acos(np.clip(float(s['command']@direction)/s['command_norm'],-1.,1.)) for s in samples)
        mean=median_gain*command
        # A radial envelope estimated directly from observed motion norms does
        # not assume worst longitudinal and lateral residuals coincide. The
        # intersection remains an empirical envelope, never a guaranteed bound.
        norm_gains=np.asarray([s['measured_norm']/s['command_norm']for s in samples])
        profile={'observed_max_command_m':float(max(scales)),
                 'noise_m':noise,
                 'norm_gain_upper':float(max(norm_gains))+self.gain_margin+.1*float(np.ptp(norm_gains)),
                 'longitudinal_low_gain':float(min(gains))-gain_pad,
                 'longitudinal_high_gain':float(max(gains))+gain_pad,
                 'lateral_gain_upper':residual+self.gain_margin+float(max(abs(gains)))*math.sin(largest_angle)}
        along_low,along_high,lateral,upper,scale_probe_margin=self._envelope_at(norm,profile)
        low_components=np.minimum(direction*along_low,direction*along_high)-lateral
        high_components=np.maximum(direction*along_low,direction*along_high)+lateral
        result.update(supported=True,reason='robot_response_supported',mean_delta_m=mean.tolist(),
            max_measured_norm_m=upper,longitudinal_interval_m=[along_low,along_high],lateral_radius_m=lateral,
            measured_component_lower_m=low_components.tolist(),measured_component_upper_m=high_components.tolist(),
            median_longitudinal_gain=median_gain,recent_longitudinal_gain=recent_gain,
            measurement_and_hold_drift_margin_m=noise,
            interpolated_below_observed_scale=norm<float(min(scales))-1e-12,
            bounded_scale_probe=norm>float(max(scales))+1e-12,
            scale_probe_margin_m=scale_probe_margin,
            empirical_envelope_parameters=profile,
            envelope_geometry='intersection of empirical longitudinal/lateral and radial envelopes')
        return result

    def command_cap_for_measured_limit(self, direction, measured_limit_m, requested_command_m,
                                       state8=None, *, joint_position=None):
        """Invert only supported empirical response; never enlarge the request.

        An unsupported cold result supplies a bounded command under an explicit
        unit-gain assumption. It is NOT returned as an empirical measured-motion
        bound. In particular, a nonresponsive measured robot receives cap zero,
        rather than gain inversion that would amplify a stalled command.
        """
        direction=self._vector(direction,3,'direction')
        length=float(np.linalg.norm(direction))
        if length<=1e-12:raise ValueError('direction must be nonzero')
        direction=direction/length
        limit=self._scalar(measured_limit_m,'measured_limit_m')
        requested=self._scalar(requested_command_m,'requested_command_m')
        if requested*float(max(abs(direction)))>self.command_scale_m+1e-12:
            raise ValueError('requested_command_m exceeds normalized OSC axis limits')
        result={'supported':False,'command_cap_m':0.,'prediction':None,
                'measured_limit_m':limit,'requested_command_m':requested,
                'reason':'robot_response_no_translation_allowance',
                'clearance_certified':False,'guaranteed_bound':False,'cold_fallback':False}
        if limit==0 or requested==0:return result
        state,joints=self._context(state8,joint_position)
        reason='robot_response_uninitialized' if state is None else self._domain_reason(state,joints)
        samples=[] if reason else self._matching(direction)
        if len(samples)>=self.minimum_samples:
            smallest=min(s['command_norm']for s in samples)*self.minimum_scale_fraction
            largest=min(requested,max(s['command_norm']for s in samples)*self.maximum_scale_fraction)
            if largest>=smallest-1e-12:
                prediction=self.predict(direction*largest,state,joint_position=joints)
                result['prediction']=prediction
                if prediction['responsive'] is False:
                    result['reason']=prediction['reason']
                    return result
                if prediction['supported']:
                    profile=prediction['empirical_envelope_parameters']
                    if self._envelope_at(smallest,profile)[3]>limit:
                        result['reason']='measured_limit_below_empirical_command_support'
                        return result
                    left,right=smallest,largest
                    if prediction['max_measured_norm_m']>limit:
                        for _ in range(50):
                            midpoint=(left+right)/2
                            if self._envelope_at(midpoint,profile)[3]<=limit:left=midpoint
                            else:right=midpoint
                        largest=left
                        prediction=self.predict(direction*largest,state,joint_position=joints)
                    result.update(supported=True,command_cap_m=largest,prediction=prediction,
                                  reason='empirical_measured_limit_inverted_without_amplification')
                    return result
        result.update(command_cap_m=min(requested,limit,self.cold_command_cap_m),
            reason=reason or 'robot_response_not_calibrated_at_requested_direction_or_scale',
            cold_fallback=True,cold_assumption='bounded command under unit-gain assumption; actual displacement is not certified')
        return result

    def summary(self):
        return {'version':self.version,'samples':len(self.samples),'transitions':self.transitions,
                'nonzero_samples':self.nonzero_samples,'holds_excluded_from_fit':self.holds,
                'domain_resets':self.domain_resets,'last_reset_reason':self.last_reset_reason,
                'hold_drift_margin_m':self.hold_drift_m,'all_time_hold_drift_m':self.all_time_hold_drift_m,
                'hold_window_transitions':self.hold_window_transitions,'minimum_samples':self.minimum_samples,
                'direction_cone_rad':self.direction_cone_rad,'minimum_scale_fraction':self.minimum_scale_fraction,
                'maximum_scale_fraction':self.maximum_scale_fraction,
                'state_available':self.last_state is not None,'joint_feedback_available':self.last_joints is not None,
                'domains':{'orientation_rad':self.orientation_domain_rad,'joint_max_rad':self.joint_domain_rad,
                           'finger_m':self.finger_domain_m,'position_m':self.position_domain_m},
                'environment_inputs':[],'clearance_certified':False,'guaranteed_bound':False,
                'task_success_certified':False,'last_observation':self.last_observation}
