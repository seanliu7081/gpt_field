"""Causal wrist-bearing goals, separate from hand anatomy and scene geometry.

The declared desired image coordinate is an objective, not an observed anchor.
V(dx) = .5 ||p_target - p_desired + dx J||^2 has squared-pixel units. Only
acknowledged measured robot motion trains J; predictions are recorded before
their observations enter training. This module neither executes nor authorizes
motion and never interprets image centering as range, grasp, contact or success.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import deque

import numpy as np

from .visual_field import SupportedLinearModel
from .visual_monitor_v4 import _orientation_distance


class WristBearingGoalFieldV35:
    version = 'causal-observed-subspace-wrist-bearing-goals-v35'
    aggregate_distance_m = .003
    maximum_aggregate_frames = 64
    minimum_goal_distance_m = .006
    maximum_goal_distance_m = .030
    orientation_domain_rad = .035
    finger_domain_m = .002
    joint_domain_rad = .35
    # An untrained observation may update the current pixels/state, but cannot
    # carry an old local response to an arbitrary new robot position. This is
    # separate from the proposed path length (which is checked by the caller).
    untrained_position_domain_m = 2 * aggregate_distance_m
    heading_orthogonal_fraction = .15

    def __init__(self):
        self.epoch = 0
        self.status = 'wrist_bearing_unconfigured'
        self.state = self.current = self._aggregate = None
        self._key = self._track = self._monitor = self._prediction = None
        self._origin = self._origin_joints = self._current_joints = None
        self._aggregate_path = 0.
        self._aggregate_frames = 0
        self.observed_transitions = self.training_updates = 0
        self.measured_training_path_m = 0.
        self.model = self._new_model()
        self.held_out = deque(maxlen=16)
        self.sample_positions = deque(maxlen=24)

    @staticmethod
    def _new_model():
        return SupportedLinearModel(residual_limit=2.5, min_samples=3,
            min_rank=1, max_samples=24, minimum_excitation=.0015)

    @staticmethod
    def _vector(value, length, name):
        result = np.asarray(value, float)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(name + ' must have finite measured components')
        return result.copy()

    @staticmethod
    def _joints(value):
        if value is None:
            return None
        result = np.asarray(value, float)
        if result.ndim != 1 or not 1 <= len(result) <= 32 or not np.isfinite(result).all():
            raise ValueError('joint_position must be a finite measured vector')
        return result.copy()

    @classmethod
    def _capture(cls, state, monitor, prediction):
        refs = (prediction or {}).get('visual_references') or {}
        wrist, target = refs.get('wrist'), (prediction or {}).get('target_id')
        if not wrist or not target or wrist.get('target_id') != target:
            return None, 'wrist_bearing_explicit_target_unavailable'
        live = (getattr(monitor, 'refs', None) or {}).get('wrist')
        track = getattr(monitor, 'tracks', {}).get('wrist_target')
        if (not live or live.get('target_id') != target or track is None
                # Production LocalFeatureBundle/PartFeatureTrack stores this
                # as ident; only its serialized summary calls it identity.
                or not track.reliable or getattr(track, 'ident', None) != target):
            return None, 'wrist_bearing_current_reference_unavailable'
        external = refs.get('external') or {}
        if external.get('moving_kind') == 'payload' and external.get('moving_id') == target:
            return None, 'wrist_bearing_target_is_declared_moving_payload'
        try:
            measured = cls._vector(getattr(monitor, 'state', None), 8, 'monitor_state')
            if not np.allclose(measured, state, atol=1e-9, rtol=0.):
                return None, 'wrist_bearing_monitor_state_stale'
            desired = cls._vector(wrist.get('desired_target_uv'), 2, 'desired_target_uv')
            live_desired = cls._vector(live.get('desired_target_uv'), 2, 'live_desired_target_uv')
            point = cls._vector(track.position, 2, 'target_position')
            scale = cls._vector(track.scale, 2, 'image_scale')
            if (not np.array_equal(desired, live_desired) or np.any(desired < 0)
                    or np.any(desired > 1) or np.any(scale <= 0)
                    or np.any(point < 0) or np.any(point > scale)):
                return None, 'wrist_bearing_image_domain_unavailable'
            residual, inliers = float(track.last_residual_px), int(track.last_inliers)
            if not np.isfinite(residual) or residual < 0 or inliers < 3:
                return None, 'wrist_bearing_measurement_noise_unavailable'
            noise = max(.08, residual / np.sqrt(inliers))
        except (ValueError, TypeError, AttributeError):
            return None, 'wrist_bearing_image_domain_unavailable'
        return {'key': (id(track), target, tuple(desired), tuple(scale)),
            'track': track, 'target_id': target, 'desired_target_uv': desired,
            'target_px': point, 'scale': scale,
            'error_px': point - desired * scale, 'noise_px': float(noise)}, None

    def _reset_estimator(self, reason):
        self.status = reason
        self.model = self._new_model()
        self.held_out.clear()
        self.sample_positions.clear()
        self._aggregate = None
        self._aggregate_frames = 0
        self._aggregate_path = 0.
        self.training_updates = 0
        self.measured_training_path_m = 0.

    def _domain_reason(self, state, joints):
        if self._origin is None:
            return 'wrist_bearing_unconfigured'
        if self._origin_joints is None and joints is not None:
            return 'wrist_bearing_joint_feedback_domain_added'
        if _orientation_distance(state[3:6], self._origin[3:6]) > self.orientation_domain_rad:
            return 'wrist_bearing_orientation_domain_changed'
        if np.max(abs(state[6:] - self._origin[6:])) > self.finger_domain_m:
            return 'wrist_bearing_finger_domain_changed'
        if self._origin_joints is not None:
            if joints is None or joints.shape != self._origin_joints.shape:
                return 'wrist_bearing_joint_feedback_unavailable'
            if np.max(abs(joints - self._origin_joints)) > self.joint_domain_rad:
                return 'wrist_bearing_joint_domain_changed'
        return None

    def _set_current(self, state, snap, joints):
        self.state = state.copy()
        self._current_joints = None if joints is None else joints.copy()
        self.current = {k: copy.deepcopy(v) for k, v in snap.items() if k not in ('key', 'track')}
        self.current['state8'] = state.copy()

    def _new_epoch(self, state, snap, joints, reason):
        self._reset_estimator(reason)
        self.epoch += 1
        self._key, self._track = snap['key'], snap['track']
        self._origin = state.copy()
        self._origin_joints = None if joints is None else joints.copy()
        self._set_current(state, snap, joints)
        self._aggregate = copy.deepcopy(self.current)

    def _unavailable(self, state, joints, reason):
        self._reset_estimator(reason)
        self._key = self._track = self.current = None
        self.state = state.copy()
        self._current_joints = None if joints is None else joints.copy()

    def configure(self, state8, monitor, prediction, *, joint_position=None):
        state, joints = self._vector(state8, 8, 'state8'), self._joints(joint_position)
        self._monitor, self._prediction = monitor, prediction
        snap, reason = self._capture(state, monitor, prediction)
        if snap is None:
            self._unavailable(state, joints, reason)
            return self.summary()
        continuous = self.state is None or np.allclose(state, self.state, atol=1e-9, rtol=0.)
        domain = self._domain_reason(state, joints) if self._origin is not None else None
        if snap['key'] != self._key or not continuous or domain:
            reason = domain or ('wrist_bearing_reference_or_desired_bearing_changed'
                if continuous else 'wrist_bearing_unacknowledged_state_gap')
            self._new_epoch(state, snap, joints, reason)
        elif self.current is not None and np.linalg.norm(snap['target_px'] - self.current['target_px']) > 1e-8:
            self._new_epoch(state, snap, joints, 'wrist_bearing_unacknowledged_pixel_revision')
        else:
            self._set_current(state, snap, joints)
        return self.summary()

    def observe(self, before8, action7, after8, *, monitor, prediction,
                joint_position=None, acknowledged=True):
        before, after = (self._vector(x, 8, 'state8') for x in (before8, after8))
        action, joints = self._vector(action7, 7, 'action7'), self._joints(joint_position)
        self._monitor, self._prediction = monitor, prediction
        self.observed_transitions += 1
        snap, reason = self._capture(after, monitor, prediction)
        if snap is None:
            self._unavailable(after, joints, reason)
            return self.summary()
        domain = self._domain_reason(after, joints)
        if (self.state is None or not np.allclose(before, self.state, atol=1e-9, rtol=0.)
                or snap['key'] != self._key or domain):
            reason = domain or ('wrist_bearing_reference_or_desired_bearing_changed'
                if snap['key'] != self._key else 'wrist_bearing_unacknowledged_state_gap')
            self._new_epoch(after, snap, joints, reason)
            return self.summary()
        old = self.current
        self._set_current(after, snap, joints)
        flow = snap['target_px'] - old['target_px']
        noise = max(snap['noise_px'], old['noise_px'])
        # A large image discontinuity needs new evidence; it is not labeled as
        # physical target motion. A stationary-robot target change cannot train J.
        distance = float(np.linalg.norm(after[:3] - before[:3]))
        rotation = _orientation_distance(after[3:6], before[3:6])
        unexplained = (distance < 1e-6 and rotation < 1e-5
            and np.linalg.norm(flow) > max(.5, 3 * noise))
        if unexplained or np.linalg.norm(flow) > max(6., 6 * noise):
            self._new_epoch(after, snap, joints, 'wrist_bearing_image_change_requires_new_evidence')
            return self.summary()
        if (acknowledged is not True or np.linalg.norm(action[:3]) <= 1e-10
                or action[6] != 0 or np.linalg.norm(action[3:6]) * .5 > .005):
            if self._response_position_gap(after) > self.untrained_position_domain_m:
                self._new_epoch(after, snap, joints, 'wrist_bearing_untrained_position_domain_changed')
                return self.summary()
            self._aggregate = copy.deepcopy(self.current)
            self._aggregate_path = 0.
            self._aggregate_frames = 0
            self.status = 'wrist_bearing_observed_without_training'
            return self.summary()
        if self._aggregate is None:
            self._aggregate = copy.deepcopy(old)
        self._aggregate_frames += 1
        self._aggregate_path += distance
        delta = after[:3] - self._aggregate['state8'][:3]
        norm = float(np.linalg.norm(delta))
        if norm >= self.aggregate_distance_m:
            if norm >= .65 * self._aggregate_path:
                observed = self.current['error_px'] - self._aggregate['error_px']
                predicted = self.model.predict(delta)
                if predicted is not None:
                    self.held_out.append({'delta_m': delta.copy(),
                        'prediction_px': predicted.copy(), 'flow_px': observed.copy(),
                        'error2': float(np.sum((observed - predicted) ** 2)),
                        'zero2': float(observed @ observed),
                        'noise_px': max(self.current['noise_px'], self._aggregate['noise_px']),
                        'training_samples_before_prediction': len(self.model.samples),
                        'observed_transition': self.observed_transitions})
                self.model.add(delta, observed)
                self.sample_positions.append((self._aggregate['state8'][:3].copy(), after[:3].copy()))
                self.training_updates += 1
                self.measured_training_path_m += self._aggregate_path
            self._aggregate = copy.deepcopy(self.current)
            self._aggregate_path = 0.
            self._aggregate_frames = 0
        elif self._aggregate_frames >= self.maximum_aggregate_frames:
            self._aggregate = copy.deepcopy(self.current)
            self._aggregate_path = 0.
            self._aggregate_frames = 0
        self.status = ('wrist_bearing_response_fitted' if self.model.matrix is not None
                       else 'wrist_bearing_accumulating_measured_motion')
        return self.summary()

    def _response_position_gap(self, state):
        if not self.sample_positions:
            return 0.
        # The most recent measured training endpoint, not an arbitrary point
        # from old history. Repeated subthreshold holds cannot move this anchor.
        return float(np.linalg.norm(state[:3] - self.sample_positions[-1][1]))

    def _heading_evidence(self, direction):
        rows = []
        for row in list(self.held_out)[-8:]:
            delta = row['delta_m']
            projection = float(delta @ direction)
            if (projection > 0 and np.linalg.norm(delta - projection * direction)
                    <= self.heading_orthogonal_fraction * np.linalg.norm(delta) + 1e-10):
                rows.append(row)
        error, zero = sum(r['error2'] for r in rows), sum(r['zero2'] for r in rows)
        return {'validated': bool(len(rows) >= 2 and zero > .5 and error < .75 * zero),
            'count': len(rows), 'error2': error, 'zero2': zero,
            'last_observed_transition': max((r['observed_transition'] for r in rows), default=None)}

    def _span(self, direction):
        if not self.sample_positions:
            return 0.
        points = np.asarray([p for pair in self.sample_positions for p in pair])
        return float(np.ptp(points @ direction))

    def _axes_basis(self, axes):
        allowed = {'xyz': (0, 1, 2), 'xy': (0, 1), 'z': (2,), 'x': (0,), 'y': (1,)}
        if axes not in allowed:
            raise ValueError('motion_axes must be xyz, xy, z, x or y')
        if self.model.basis is None:
            return np.zeros((3, 0))
        projected = self.model.basis.copy()
        projected[:, [i for i in range(3) if i not in allowed[axes]]] = 0.
        _, singular, vt = np.linalg.svd(projected, full_matrices=False)
        return vt[singular > 1e-7].T

    def propose(self, state8, *, motion_axes='xyz', metric_goal_xyz=None,
                max_goal_distance_m=.030, max_candidates=6, joint_position=None):
        state, joints = self._vector(state8, 8, 'state8'), self._joints(joint_position)
        limit = float(max_goal_distance_m)
        if not np.isfinite(limit) or limit < 0 or type(max_candidates) is not int or not 1 <= max_candidates <= 24:
            raise ValueError('proposal limits must be finite and bounded')
        result = {'version': self.version, 'objective': 'declared_wrist_target_bearing',
            'accepted': False, 'reason': self.status, 'candidates': [],
            'metric_image_disagreement': None, 'current_snapshot': self._snapshot_summary(),
            'supported_rank': self.model.rank,
            'supported_basis_xyz': None if self.model.basis is None else self.model.basis.tolist(),
            'geometry_checked': False, 'execution_authorized': False,
            'physical_budget_changed': False, 'task_success_certified': False,
            'target_pose_estimated': False, 'anatomical_alignment_certified': False}
        if self.state is None or not np.allclose(state, self.state, atol=1e-9, rtol=0.):
            result['reason'] = 'wrist_bearing_proposal_state_stale'
            return result
        if self._current_joints is not None and joints is None:
            result['reason'] = 'wrist_bearing_proposal_current_joint_feedback_required'
            return result
        if joints is not None and (self._current_joints is None
                or joints.shape != self._current_joints.shape
                or not np.allclose(joints, self._current_joints, atol=1e-9, rtol=0.)):
            result['reason'] = 'wrist_bearing_proposal_joint_feedback_stale'
            return result
        if self.current is None:
            # Keep the actual capture failure visible, e.g. mismatched target
            # identity, rather than disguising it as merely insufficient fit.
            result['reason'] = self.status
            return result
        if self.model.matrix is None:
            result['reason'] = 'wrist_bearing_response_unavailable'
            return result
        if self._response_position_gap(state) > self.untrained_position_domain_m:
            result['reason'] = 'wrist_bearing_proposal_outside_response_position_domain'
            return result
        current, reason = self._capture(state, self._monitor, self._prediction)
        if (current is None or current['key'] != self._key
                or np.linalg.norm(current['target_px'] - self.current['target_px']) > 1e-8):
            result['reason'] = reason or 'wrist_bearing_proposal_reference_stale'
            return result
        if self._domain_reason(state, self._current_joints):
            result['reason'] = 'wrist_bearing_proposal_domain_changed'
            return result
        basis = self._axes_basis(motion_axes)
        if not basis.shape[1]:
            result['reason'] = 'wrist_bearing_axes_unobserved'
            return result
        error, matrix = self.current['error_px'], self.model.matrix
        projector = self.model.basis.T @ self.model.basis
        if metric_goal_xyz is not None:
            metric = self._vector(metric_goal_xyz, 3, 'metric_goal_xyz') - state[:3]
            supported = projector @ metric
            slope = float(error @ (supported @ matrix))
            result['metric_image_disagreement'] = {
                'metric_goal_xyz_m': (state[:3] + metric).tolist(),
                'observed_subspace_component_m': supported.tolist(),
                'unobserved_component_m': (metric - supported).tolist(),
                'local_potential_directional_derivative_px2': slope,
                'opposes_visual_descent_in_observed_subspace': bool(slope > 0),
                'not_a_target_pose_or_geometry_rejection': True}
        jacobian = basis.T @ matrix
        regularizer = max(1e-8, float(np.linalg.norm(jacobian)) ** 2 * 1e-5)
        solved = -np.linalg.solve(jacobian @ jacobian.T + regularizer * np.eye(basis.shape[1]),
                                 jacobian @ error)
        directions = [basis @ solved]
        directions += [basis @ (basis.T @ row['delta_m']) for row in self.held_out]
        unique = []
        for delta in directions:
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-10:
                continue
            unit = delta / norm
            if (np.linalg.norm(unit - projector @ unit) > self.heading_orthogonal_fraction
                    or any(float(unit @ previous) > .995 for previous in unique)):
                continue
            unique.append(unit)
        rejection, candidates = 'wrist_bearing_no_validated_descent_heading', []
        for direction in unique:
            evidence = self._heading_evidence(direction)
            if not evidence['validated']:
                continue
            gain = direction @ matrix
            slope = float(error @ gain)
            if slope >= 0 or np.linalg.norm(gain) <= 20.:
                continue
            span = self._span(direction)
            trust = min(limit, self.maximum_goal_distance_m, span)
            upper = min(trust, max(0., -slope / float(gain @ gain)))
            if upper < self.minimum_goal_distance_m:
                rejection = 'wrist_bearing_no_useful_trusted_goal'
                continue
            lengths = sorted({min(upper, length) for length in (.006, .010, .020, .030, upper)}, reverse=True)
            for length in lengths:
                if length < self.minimum_goal_distance_m - 1e-12:
                    continue
                delta = direction * length
                predicted = error + delta @ matrix
                reduction = float(np.linalg.norm(error) - np.linalg.norm(predicted))
                noise = max(.25, self.current['noise_px'], float(self.model.residual or 0.))
                point = predicted + self.current['desired_target_uv'] * self.current['scale']
                if (reduction <= max(.5, 2 * noise) or np.any(point < 2 * noise)
                        or np.any(point > self.current['scale'] - 2 * noise)):
                    continue
                binding = {'epoch': self.epoch, 'state8': state.tolist(),
                    'joint_position': None if self._current_joints is None else self._current_joints.tolist(),
                    'target_id': self.current['target_id'],
                    'desired_target_uv': self.current['desired_target_uv'].tolist(),
                    'target_px': self.current['target_px'].tolist(),
                    'error_px': error.tolist(), 'observed_transitions': self.observed_transitions,
                    'goal_offset_m': delta.tolist()}
                token = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
                candidates.append({'proposal_id': 'wrist-bearing-goal-' + token,
                    'objective': 'declared_wrist_target_bearing',
                    'goal_offset_m': delta.tolist(), 'goal_xyz_m': (state[:3] + delta).tolist(),
                    'motion_axes': motion_axes, 'predicted_error_vector_px': predicted.tolist(),
                    'predicted_error_px': float(np.linalg.norm(predicted)),
                    'predicted_error_reduction_px': reduction, 'predicted_target_px': point.tolist(),
                    'current_potential_px2': .5 * float(error @ error),
                    'predicted_potential_px2': .5 * float(predicted @ predicted),
                    'measured_directional_span_m': span, 'trust_limit_m': trust,
                    'proposal_distance_m': length, 'prequential_heading_evidence': evidence,
                    'prediction_noise_proxy_px': noise, 'binding': binding,
                    'full_route_geometry_and_current_admission_required': True,
                    'explicit_model_choice_required': True,
                    'predicted_goal_requires_live_validation': True,
                    'semantic_objective_certified': False, 'clearance_certified': False,
                    'target_pose_estimated': False, 'anatomical_alignment_certified': False})
        candidates.sort(key=lambda row: (-row['predicted_error_reduction_px'], row['proposal_id']))
        result['candidates'] = candidates[:max_candidates]
        result['accepted'] = bool(result['candidates'])
        result['reason'] = 'wrist_bearing_ranked_fixed_goal_proposals' if candidates else rejection
        return result

    def _snapshot_summary(self):
        if self.current is None:
            return None
        current = self.current
        return {'epoch': self.epoch, 'state8': current['state8'].tolist(),
            'joint_position': None if self._current_joints is None else self._current_joints.tolist(),
            'target_id': current['target_id'], 'target_px': current['target_px'].tolist(),
            'desired_target_uv': current['desired_target_uv'].tolist(),
            'error_vector_px': current['error_px'].tolist(),
            'error_px': float(np.linalg.norm(current['error_px'])), 'noise_px': current['noise_px'],
            'desired_coordinate_is_observed_anchor': False}

    def summary(self):
        return {'version': self.version, 'status': self.status, 'epoch': self.epoch,
            'current_snapshot': self._snapshot_summary(), 'response': self.model.summary(),
            'response_matrix_px_per_m': None if self.model.matrix is None else self.model.matrix.tolist(),
            'supported_basis_xyz': None if self.model.basis is None else self.model.basis.tolist(),
            'observed_transitions': self.observed_transitions, 'training_updates': self.training_updates,
            'measured_training_path_m': self.measured_training_path_m,
            'untrained_position_domain_m': self.untrained_position_domain_m,
            'distance_from_latest_training_endpoint_m': (
                None if self.state is None or not self.sample_positions
                else self._response_position_gap(self.state)),
            'held_out_predictions': len(self.held_out),
            'held_out_error2': sum(row['error2'] for row in self.held_out),
            'held_out_zero_motion_error2': sum(row['zero2'] for row in self.held_out),
            'held_out_records': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in row.items()} for row in self.held_out],
            'held_out_records_are_prospective_only': True,
            'potential_definition': '.5 * squared norm of wrist target minus declared desired bearing plus locally predicted flow',
            'potential_unit': 'pixel_squared', 'maximum_goal_distance_m': self.maximum_goal_distance_m,
            'minimum_goal_distance_m': self.minimum_goal_distance_m,
            'physical_budget_owned': False, 'execution_authorized': False,
            'clearance_certified': False, 'semantic_objective_certified': False,
            'task_success_certified': False, 'target_pose_estimated': False,
            'anatomical_alignment_certified': False,
            'limitations': ['The target identity and desired bearing are model declarations, not verified semantics.',
                'Pixel bearing is not depth, pinch alignment, contact, attachment or task completion.',
                'A motion proxy supplies no anatomical contact point and does not define this objective.',
                'Only measured supported subspaces and prospectively validated headings produce proposals.',
                'Untrained motion cannot carry fitted response beyond 6mm from its latest training endpoint.',
                'All proposals still require explicit model choice and current full geometry/action admission.']}
