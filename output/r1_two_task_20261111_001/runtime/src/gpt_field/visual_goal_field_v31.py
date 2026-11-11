"""Observed-subspace image objective proposals, without execution permission.

The field is V(dx) = .5 ||e + dx J||^2 (pixel squared), for the CURRENT
declared relative tool/target image objective.  It is neither a scene SDF nor
an estimate of the target's 3-D pose.  Only acknowledged measured translation
trains J. Every validation prediction precedes adding its own observation.
Rank, heading validation, measured span and robot/reference domains bound
proposals. No action, persistent goal, geometry, or physical budget is changed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import deque

import numpy as np

from .visual_field import SupportedLinearModel
from .visual_monitor_v4 import _orientation_distance


class VisualGoalFieldV31:
    version = 'observed-subspace-visual-goal-field-v31'
    aggregate_distance_m = .003
    maximum_aggregate_frames = 64
    minimum_goal_distance_m = .006
    maximum_goal_distance_m = .030
    orientation_domain_rad = .035
    finger_domain_m = .002
    joint_domain_rad = .35
    heading_orthogonal_fraction = .15

    def __init__(self):
        self.epoch = 0
        self.status = 'visual_objective_unconfigured'
        self._key = self._track_refs = self._origin = self._origin_joints = None
        self._current_joints = None
        self.state = self.current = self._aggregate = None
        self._aggregate_path = 0.
        self._aggregate_frames = 0
        self.observed_transitions = 0
        self.training_updates = 0
        self.measured_training_path_m = 0.
        self.model = self._new_model()
        self.held_out = deque(maxlen=16)
        self.sample_positions = deque(maxlen=24)

    @staticmethod
    def _new_model():
        # Sub-millimetre cross-axis drift is not independent 3-D excitation.
        return SupportedLinearModel(residual_limit=2.5, min_samples=3,
            min_rank=1, max_samples=24, minimum_excitation=.0015)

    @staticmethod
    def _vector(value, length, name):
        result = np.asarray(value, float)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(name+' must have finite measured components')
        return result.copy()

    @classmethod
    def _joints(cls, value):
        if value is None:
            return None
        result = np.asarray(value, float)
        if result.ndim != 1 or not 1 <= len(result) <= 32 or not np.isfinite(result).all():
            raise ValueError('joint_position must be a finite measured vector')
        return result.copy()

    @staticmethod
    def _capture(monitor, prediction):
        ext = ((prediction or {}).get('visual_references') or {}).get('external') or {}
        target = (prediction or {}).get('target_id')
        if ext.get('moving_kind') != 'tool' or not target or ext.get('target_id') != target:
            return None, 'visual_objective_role_unavailable'
        if not getattr(monitor, '_external_reliable', lambda: False)():
            return None, 'visual_objective_reference_unavailable'
        tracks = getattr(monitor, 'tracks', {})
        moving, anchor = tracks.get('moving'), tracks.get('target')
        if (moving is None or anchor is None or not moving.reliable or not anchor.reliable
                or moving is anchor):
            return None, 'visual_objective_reference_unavailable'
        try:
            offset = VisualGoalFieldV31._vector(ext.get('goal_offset_uv'), 2, 'goal_offset_uv')
            scale = VisualGoalFieldV31._vector(moving.scale, 2, 'image_scale')
            anchor_scale = VisualGoalFieldV31._vector(anchor.scale, 2, 'anchor_scale')
            tool = VisualGoalFieldV31._vector(moving.position, 2, 'tool_position')
            fixed = VisualGoalFieldV31._vector(anchor.position, 2, 'anchor_position')
            if (not np.array_equal(scale, anchor_scale) or np.any(scale <= 0)
                    or np.any(tool < 0) or np.any(tool > scale)
                    or np.any(fixed < 0) or np.any(fixed > scale)):
                return None, 'visual_objective_image_domain_unavailable'
            noise = float(np.sqrt(sum(max(.08, float(t.last_residual_px)
                / np.sqrt(max(3, int(t.last_inliers))))**2 for t in (moving, anchor))))
            if not np.isfinite(noise):
                return None, 'visual_objective_noise_unavailable'
        except (ValueError, TypeError, AttributeError):
            return None, 'visual_objective_image_domain_unavailable'
        return {'key': (id(moving), id(anchor), target, tuple(offset), tuple(scale)),
            'track_refs': (moving, anchor), 'target_id': target,
            'moving_id': ext.get('moving_id'), 'goal_offset_uv': offset.copy(),
            'error_px': tool-fixed-offset*scale, 'noise_px': noise,
            'anchor_px': fixed, 'moving_px': tool, 'scale': scale}, None

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

    def _domain_reason(self, state, joints=None):
        if self._origin is None:
            return 'visual_objective_unconfigured'
        if self._origin_joints is None and joints is not None:
            return 'visual_objective_joint_feedback_domain_added'
        if _orientation_distance(state[3:6], self._origin[3:6]) > self.orientation_domain_rad:
            return 'visual_objective_orientation_domain_changed'
        if np.max(abs(state[6:]-self._origin[6:])) > self.finger_domain_m:
            return 'visual_objective_finger_domain_changed'
        if self._origin_joints is not None:
            if joints is None or joints.shape != self._origin_joints.shape:
                return 'visual_objective_joint_feedback_unavailable'
            if np.max(abs(joints-self._origin_joints)) > self.joint_domain_rad:
                return 'visual_objective_joint_domain_changed'
        return None

    def _set_current(self, state, snap, joints=None):
        self.state = state.copy()
        self._current_joints = None if joints is None else joints.copy()
        self.current = {k: copy.deepcopy(v) for k, v in snap.items()
                        if k not in ('key', 'track_refs')}
        self.current['state8'] = state.copy()

    def _new_epoch(self, state, snap, joints, reason):
        self._reset_estimator(reason)
        self.epoch += 1
        self._key, self._track_refs = snap['key'], snap['track_refs']
        self._origin, self._origin_joints = state.copy(), joints
        self._set_current(state, snap, joints)
        self._aggregate = copy.deepcopy(self.current)

    def configure(self, state8, monitor, prediction, *, joint_position=None):
        state = self._vector(state8, 8, 'state8')
        joints = self._joints(joint_position)
        snap, reason = self._capture(monitor, prediction)
        if snap is None:
            self._reset_estimator(reason)
            self._key = self._track_refs = self.current = None
            self.state = state
            self._current_joints = joints
            return self.summary()
        continuous = self.state is None or np.allclose(state, self.state, atol=1e-9, rtol=0.)
        domain = self._domain_reason(state, joints) if self._origin is not None else None
        if snap['key'] != self._key or not continuous or domain:
            reason = domain or ('visual_objective_reference_or_offset_changed' if continuous
                                else 'visual_objective_unacknowledged_state_gap')
            self._new_epoch(state, snap, joints, reason)
        else:
            # An unacknowledged pixel revision also must not train cross-epoch
            # response, even if a caller reuses the same track objects.
            if (self.current is not None and
                    (np.linalg.norm(snap['error_px']-self.current['error_px']) > 1e-8
                     or np.linalg.norm(snap['anchor_px']-self.current['anchor_px']) > 1e-8)):
                self._new_epoch(state, snap, joints, 'visual_objective_unacknowledged_pixel_revision')
            else:
                self._set_current(state, snap, joints)
        return self.summary()

    def observe(self, before8, action7, after8, *, monitor, prediction,
                joint_position=None, acknowledged=True):
        before, after = (self._vector(x, 8, 'state8') for x in (before8, after8))
        action = self._vector(action7, 7, 'action7')
        joints = self._joints(joint_position)
        snap, reason = self._capture(monitor, prediction)
        self.observed_transitions += 1
        if snap is None:
            self._reset_estimator(reason)
            self.current = None
            self.state = after.copy()
            self._current_joints = joints
            return self.summary()
        domain = self._domain_reason(after, joints)
        if (self.state is None or not np.allclose(before, self.state, atol=1e-9, rtol=0.)
                or snap['key'] != self._key or domain):
            reason = domain or ('visual_objective_reference_or_offset_changed'
                if snap['key'] != self._key else 'visual_objective_unacknowledged_state_gap')
            self._new_epoch(after, snap, joints, reason)
            return self.summary()
        old = self.current
        self._set_current(after, snap, joints)
        if old is None or np.linalg.norm(snap['anchor_px']-old['anchor_px']) > 6.:
            self._new_epoch(after, snap, joints, 'visual_objective_anchor_motion_unusable')
            return self.summary()
        distance = float(np.linalg.norm(after[:3]-before[:3]))
        # No hold, rejected action, or finger operation produces training credit.
        # A hold's residual movement updates current error but never bridges an
        # acquisition aggregate with another command.
        if (not acknowledged or np.linalg.norm(action[:3]) <= 1e-10
                or action[6] != 0 or np.linalg.norm(action[3:6])*.5 > .005):
            self._aggregate = copy.deepcopy(self.current)
            self._aggregate_frames = 0
            self._aggregate_path = 0.
            self.status = 'visual_objective_observed_without_training'
            return self.summary()
        if self._aggregate is None:
            self._aggregate = copy.deepcopy(old)
        self._aggregate_frames += 1
        self._aggregate_path += distance
        delta = after[:3]-self._aggregate['state8'][:3]
        norm = float(np.linalg.norm(delta))
        if norm >= self.aggregate_distance_m:
            if norm >= .65*self._aggregate_path:
                flow = self.current['error_px']-self._aggregate['error_px']
                prediction_before_training = self.model.predict(delta)
                if prediction_before_training is not None:
                    self.held_out.append({'delta_m': delta.copy(),
                        'prediction_px': prediction_before_training.copy(), 'flow_px': flow.copy(),
                        'error2': float(np.sum((flow-prediction_before_training)**2)),
                        'zero2': float(flow@flow),
                        'noise_px': max(self.current['noise_px'], self._aggregate['noise_px']),
                        'training_samples_before_prediction': len(self.model.samples),
                        'observed_transition': self.observed_transitions})
                self.model.add(delta, flow)
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
        self.status = ('visual_objective_response_fitted' if self.model.matrix is not None
                       else 'visual_objective_accumulating_measured_motion')
        return self.summary()

    def _heading_evidence(self, direction):
        rows = []
        for row in list(self.held_out)[-8:]:
            delta = row['delta_m']; projection = float(delta@direction)
            if projection > 0 and np.linalg.norm(delta-projection*direction) <= \
                    self.heading_orthogonal_fraction*np.linalg.norm(delta)+1e-10:
                rows.append(row)
        error = sum(r['error2'] for r in rows)
        zero = sum(r['zero2'] for r in rows)
        # A fit is not validation. Require at least two later predictions with
        # actual observable signal and improvement over the zero-flow baseline.
        valid = bool(len(rows) >= 2 and zero > .5 and error < .75*zero)
        return {'validated': valid, 'count': len(rows), 'error2': error, 'zero2': zero,
            'last_observed_transition': max((r['observed_transition'] for r in rows), default=None)}

    def _span(self, direction):
        if not self.sample_positions:
            return 0.
        points = np.asarray([p for pair in self.sample_positions for p in pair])
        return float(np.ptp(points@direction))

    def _axes_basis(self, axes):
        allowed = {'xyz': (0,1,2), 'xy': (0,1), 'z': (2,), 'x': (0,), 'y': (1,)}
        if axes not in allowed:
            raise ValueError('motion_axes must be xyz, xy, z, x or y')
        if self.model.basis is None:
            return np.zeros((3, 0))
        # Exact intersection can lose a useful almost-planar measured heading
        # from tiny robot drift. Project onto allowed axes, then require the
        # result still be supported by the original measured subspace.
        projected = self.model.basis.copy()
        projected[:, [i for i in range(3) if i not in allowed[axes]]] = 0.
        _, singular, vt = np.linalg.svd(projected, full_matrices=False)
        basis = vt[singular > 1e-7].T
        return basis

    def propose(self, state8, *, motion_axes='xyz', metric_goal_xyz=None,
                max_goal_distance_m=.03, max_candidates=6, joint_position=None):
        state = self._vector(state8, 8, 'state8')
        query_joints = self._joints(joint_position)
        limit = float(max_goal_distance_m)
        if not np.isfinite(limit) or limit < 0 or type(max_candidates) is not int or not 1 <= max_candidates <= 24:
            raise ValueError('proposal limits must be finite and bounded')
        result = {'version': self.version, 'accepted': False, 'reason': self.status,
            'candidates': [], 'metric_image_disagreement': None,
            'current_snapshot': self._snapshot_summary(), 'supported_rank': self.model.rank,
            'supported_basis_xyz': None if self.model.basis is None else self.model.basis.tolist(),
            'geometry_checked': False, 'execution_authorized': False,
            'physical_budget_changed': False, 'task_success_certified': False,
            'target_pose_estimated': False}
        if self.state is None or not np.allclose(state, self.state, atol=1e-9, rtol=0.):
            result['reason'] = 'visual_objective_proposal_state_stale'
            return result
        if query_joints is not None and (self._current_joints is None
                or query_joints.shape != self._current_joints.shape
                or not np.allclose(query_joints, self._current_joints, atol=1e-9, rtol=0.)):
            result['reason'] = 'visual_objective_proposal_joint_feedback_stale'
            return result
        if self.current is None or self.model.matrix is None:
            result['reason'] = 'visual_objective_response_unavailable'
            return result
        if (self._track_refs is None or any(not t.reliable for t in self._track_refs)
                or np.linalg.norm(np.asarray(self._track_refs[0].position)-self.current['moving_px']) > 1e-8
                or np.linalg.norm(np.asarray(self._track_refs[1].position)-self.current['anchor_px']) > 1e-8):
            result['reason'] = 'visual_objective_proposal_reference_stale'
            return result
        if self._domain_reason(state, self._current_joints):
            result['reason'] = 'visual_objective_proposal_domain_changed'
            return result
        basis = self._axes_basis(motion_axes)
        if not basis.shape[1]:
            result['reason'] = 'visual_objective_axes_unobserved'
            return result
        error = self.current['error_px']; matrix = self.model.matrix
        projector = self.model.basis.T@self.model.basis
        if metric_goal_xyz is not None:
            metric = self._vector(metric_goal_xyz, 3, 'metric_goal_xyz')-state[:3]
            supported = projector@metric
            value = float(error@(supported@matrix))
            result['metric_image_disagreement'] = {
                'metric_goal_xyz_m': (state[:3]+metric).tolist(),
                'observed_subspace_component_m': supported.tolist(),
                'unobserved_component_m': (metric-supported).tolist(),
                'local_potential_directional_derivative_px2': value,
                'opposes_visual_descent_in_observed_subspace': bool(value > 0),
                'not_a_target_pose_or_geometry_rejection': True}
        jacobian = basis.T@matrix
        # Damped least squares in the measured allowed subspace; no component
        # in an unseen direction is invented to make a 3-D target estimate.
        regularizer = max(1e-8, float(np.linalg.norm(jacobian))**2*1e-5)
        solved = -np.linalg.solve(jacobian@jacobian.T+regularizer*np.eye(basis.shape[1]), jacobian@error)
        raw = basis@solved
        directions = [raw]
        directions += [basis@(basis.T@row['delta_m']) for row in self.held_out]
        unique = []
        for d in directions:
            norm = float(np.linalg.norm(d))
            if norm <= 1e-10:
                continue
            u = d/norm
            if (np.linalg.norm(u-projector@u) > self.heading_orthogonal_fraction
                    or any(float(u@prev) > .995 for prev in unique)):
                continue
            unique.append(u)
        candidates = []
        rejection = 'visual_objective_no_validated_descent_heading'
        for direction in unique:
            evidence = self._heading_evidence(direction)
            if not evidence['validated']:
                continue
            gain = direction@matrix
            slope = float(error@gain)
            if slope >= 0 or np.linalg.norm(gain) <= 20.:
                continue
            span = self._span(direction)
            trust = min(limit, self.maximum_goal_distance_m, span)
            optimum = max(0., -slope/float(gain@gain))
            upper = min(trust, optimum)
            if upper < self.minimum_goal_distance_m:
                rejection = 'visual_objective_no_useful_trusted_goal'
                continue
            lengths = sorted({min(upper, v) for v in (.006, .010, .020, .030, upper)}, reverse=True)
            for length in lengths:
                if length < self.minimum_goal_distance_m-1e-12:
                    continue
                delta = direction*length
                predicted = error+delta@matrix
                reduction = float(np.linalg.norm(error)-np.linalg.norm(predicted))
                noise = max(.25, self.current['noise_px'], float(self.model.residual or 0.))
                if reduction <= max(.5, 2*noise):
                    continue
                binding = {'epoch': self.epoch, 'state8': state.tolist(),
                    'joint_position': None if self._current_joints is None else self._current_joints.tolist(),
                    'error_px': error.tolist(), 'observed_transitions': self.observed_transitions,
                    'goal_offset_m': delta.tolist()}
                token = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
                candidates.append({'proposal_id': 'visual-goal-'+token,
                    'goal_offset_m': delta.tolist(), 'goal_xyz_m': (state[:3]+delta).tolist(),
                    'motion_axes': motion_axes, 'predicted_error_vector_px': predicted.tolist(),
                    'predicted_error_px': float(np.linalg.norm(predicted)),
                    'predicted_error_reduction_px': reduction,
                    'current_potential_px2': .5*float(error@error),
                    'predicted_potential_px2': .5*float(predicted@predicted),
                    'measured_directional_span_m': span, 'trust_limit_m': trust,
                    'proposal_distance_m': length, 'prequential_heading_evidence': evidence,
                    'prediction_noise_proxy_px': noise, 'binding': binding,
                    'full_route_geometry_and_current_admission_required': True,
                    'predicted_goal_requires_live_validation': True,
                    'semantic_objective_certified': False, 'clearance_certified': False})
        candidates.sort(key=lambda c: (-c['predicted_error_reduction_px'], c['proposal_id']))
        result['candidates'] = candidates[:max_candidates]
        result['accepted'] = bool(result['candidates'])
        result['reason'] = 'visual_objective_ranked_fixed_goal_proposals' if candidates else rejection
        return result

    def _snapshot_summary(self):
        if self.current is None:
            return None
        c = self.current
        return {'epoch': self.epoch, 'state8': c['state8'].tolist(), 'target_id': c['target_id'],
            'joint_position': None if self._current_joints is None else self._current_joints.tolist(),
            'moving_id': c['moving_id'], 'goal_offset_uv': c['goal_offset_uv'].tolist(),
            'error_vector_px': c['error_px'].tolist(), 'error_px': float(np.linalg.norm(c['error_px'])),
            'noise_px': c['noise_px'], 'moving_px': c['moving_px'].tolist(),
            'anchor_px': c['anchor_px'].tolist()}

    def summary(self):
        return {'version': self.version, 'status': self.status, 'epoch': self.epoch,
            'current_snapshot': self._snapshot_summary(), 'response': self.model.summary(),
            'response_matrix_px_per_m': None if self.model.matrix is None else self.model.matrix.tolist(),
            'supported_basis_xyz': None if self.model.basis is None else self.model.basis.tolist(),
            'observed_transitions': self.observed_transitions, 'training_updates': self.training_updates,
            'measured_training_path_m': self.measured_training_path_m,
            'held_out_predictions': len(self.held_out),
            'held_out_error2': sum(r['error2'] for r in self.held_out),
            'held_out_zero_motion_error2': sum(r['zero2'] for r in self.held_out),
            'held_out_records': [{k:(v.tolist() if isinstance(v, np.ndarray) else v)
                for k,v in row.items()} for row in self.held_out],
            'potential_definition': '.5 * squared norm of current relative pixel residual plus locally predicted flow',
            'maximum_goal_distance_m': self.maximum_goal_distance_m,
            'minimum_goal_distance_m': self.minimum_goal_distance_m,
            'clearance_certified': False, 'semantic_objective_certified': False,
            'task_success_certified': False, 'target_pose_estimated': False,
            'physical_budget_owned': False,
            'limitations': ['The declared anatomical feature/offset can be wrong despite predictive image motion.',
                'Only observed motion subspace and prospectively validated headings are proposed.',
                'The image potential cannot revise or certify scene geometry or contact.',
                'Goal extrapolation is bounded by measured span and still requires current Field admission.']}
