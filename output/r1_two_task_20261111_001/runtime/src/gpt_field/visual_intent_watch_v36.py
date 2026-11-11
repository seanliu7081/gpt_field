"""Causal intent review before a tracked wrist target loses useful visibility.

This observer consumes existing RGB-track and measured-response summaries. It
does not choose motion, fit a response, invent target depth or change geometry,
physical allowances, collision decisions, or task completion. A necessary
approach is not rejected merely because its pixel error increases: a review
requires current boundary proximity, sustained measured deterioration, and a
prospectively validated same-heading prediction of further outward motion.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import deque

import numpy as np


class WristIntentWatchV36:
    version = 'causal-wrist-visibility-intent-review-v36'
    heading_orthogonal_fraction = .15
    minimum_measured_trend_m = .006
    minimum_trend_samples = 5
    recent_window_samples = 5
    maximum_forecast_distance_m = .010

    def __init__(self):
        self.intent_id = None
        self.observed_transitions = 0
        self.status = 'awaiting_intent_observation'
        self._key = self._origin = self._last_state = None
        self._history = deque(maxlen=64)
        self._path_m = 0.
        self._event = None
        self._new_event = False
        self._archive = deque(maxlen=8)
        self._current = None

    @staticmethod
    def _vector(value, length):
        result = np.asarray(value, float)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError('Expected finite measured vector')
        return result.copy()

    def begin_intent(self, intent_id):
        """A new model refresh clears the active event, never its archive.

        Use the caller's monotonically changing refresh sequence, not a model's
        reusable hypothesis ID. Calling again with the same sequence is inert.
        """
        if not isinstance(intent_id, (int, str)) or isinstance(intent_id, bool):
            raise ValueError('intent_id must be a caller-owned integer or string')
        if intent_id != self.intent_id:
            self.intent_id = intent_id
            self._key = self._origin = self._last_state = None
            self._history.clear()
            self._path_m = 0.
            self._event = None
            self._new_event = False
            self._current = None
            self.status = 'awaiting_intent_observation'
        return self.summary()

    def acknowledge_review(self, new_intent_id):
        """After the next model reply, archive remains and its active latch ends.

        The reply itself, geometry and physical budgets belong to the caller.
        A repeated sequence cannot clear a pending review.
        """
        return self.begin_intent(new_intent_id)

    def _clear_trend(self):
        self._key = self._origin = None
        self._history.clear()
        self._path_m = 0.

    @classmethod
    def _capture(cls, after, wrist, track, image_size_wh, target_id):
        if not isinstance(wrist, dict) or not isinstance(track, dict):
            return None, 'current_wrist_reference_unavailable'
        current = wrist.get('current_snapshot')
        if (not isinstance(current, dict) or not isinstance(target_id, str)
                or not target_id or current.get('target_id') != target_id
                or track.get('identity') != target_id or track.get('reliable') is not True
                or track.get('point_currently_observed') is not True):
            return None, 'current_wrist_reference_unavailable'
        try:
            state = cls._vector(current.get('state8'), 8)
            if not np.allclose(state, after, atol=1e-9, rtol=0.):
                return None, 'current_wrist_state_binding_mismatch'
            size = cls._vector(image_size_wh, 2)
            if np.any(size < 24) or np.any(size > 1024) or not np.array_equal(size, np.floor(size)):
                return None, 'current_wrist_image_size_invalid'
            scale = size - 1.
            point = cls._vector(current.get('target_px'), 2)
            desired = cls._vector(current.get('desired_target_uv'), 2)
            error = cls._vector(current.get('error_vector_px'), 2)
            live_point = cls._vector(track.get('tracked_point_uv'), 2) * scale
            if (np.any(point < 0) or np.any(point > scale) or np.any(desired < 0)
                    or np.any(desired > 1) or not np.allclose(live_point, point, atol=1e-7, rtol=0.)
                    or not np.allclose(point - desired * scale, error, atol=1e-7, rtol=0.)):
                return None, 'current_wrist_pixel_binding_mismatch'
            epoch = current.get('epoch')
            if type(epoch) is not int or epoch != wrist.get('epoch'):
                return None, 'current_wrist_domain_binding_mismatch'
            inliers, initial = track.get('inliers'), track.get('initial_features')
            if (type(inliers) is not int or type(initial) is not int
                    or inliers < 3 or initial < inliers):
                return None, 'current_wrist_support_unavailable'
            noise, radius = float(current['noise_px']), float(track['support_radius_px'])
            if not np.isfinite([noise, radius]).all() or noise < 0 or not 4 <= radius <= 16:
                return None, 'current_wrist_support_unavailable'
        except (ValueError, TypeError, KeyError):
            return None, 'current_wrist_summary_invalid'
        margins = np.array([point[0], scale[0] - point[0], point[1], scale[1] - point[1]])
        return {'key': (epoch, target_id, tuple(desired), tuple(size)), 'epoch': epoch,
            'target_id': target_id, 'state8': state, 'point_px': point,
            'desired_target_uv': desired, 'error_vector_px': error,
            'error_px': float(np.linalg.norm(error)), 'scale': scale,
            'noise_px': max(.08, noise), 'margins_px': margins,
            'support_radius_px': radius, 'inliers': inliers, 'initial_features': initial}, None

    @classmethod
    def _response(cls, wrist, direction):
        response = wrist.get('response') or {}
        if response.get('reliable') is not True:
            return None, 'same_heading_response_not_fitted'
        try:
            matrix = np.asarray(wrist.get('response_matrix_px_per_m'), float)
            basis = np.asarray(wrist.get('supported_basis_xyz'), float)
            if (matrix.shape != (3, 2) or basis.ndim != 2 or basis.shape[1] != 3
                    or not 1 <= basis.shape[0] <= 3 or not np.isfinite(matrix).all()
                    or not np.isfinite(basis).all()
                    or not np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-6, rtol=0.)):
                return None, 'same_heading_response_invalid'
            if np.linalg.norm(direction - basis.T @ (basis @ direction)) > cls.heading_orthogonal_fraction:
                return None, 'current_heading_outside_observed_subspace'
            observed = wrist.get('observed_transitions')
            if type(observed) is not int:
                return None, 'same_heading_response_invalid'
            rows = []
            for row in list(wrist.get('held_out_records') or [])[-8:]:
                delta = cls._vector(row.get('delta_m'), 3)
                prediction = cls._vector(row.get('prediction_px'), 2)
                flow = cls._vector(row.get('flow_px'), 2)
                before = row.get('training_samples_before_prediction')
                at = row.get('observed_transition')
                if (type(before) is not int or before < 3 or type(at) is not int
                        or not 0 < at <= observed):
                    continue
                projection = float(delta @ direction)
                if (projection <= 0 or np.linalg.norm(delta - projection * direction)
                        > cls.heading_orthogonal_fraction * np.linalg.norm(delta) + 1e-10):
                    continue
                # Recompute the reported scores, so stale/malformed convenience
                # totals cannot substitute for actual prospective records.
                error2, zero2 = float(np.sum((flow - prediction) ** 2)), float(flow @ flow)
                if (not np.isclose(error2, float(row['error2']), atol=1e-8, rtol=1e-7)
                        or not np.isclose(zero2, float(row['zero2']), atol=1e-8, rtol=1e-7)):
                    continue
                rows.append((projection, error2, zero2, at))
            # Duplicate records are not independent held-out observations.
            if len({row[3] for row in rows}) != len(rows):
                return None, 'same_heading_validation_records_not_distinct'
            error2, zero2 = sum(r[1] for r in rows), sum(r[2] for r in rows)
            if len(rows) < 2 or zero2 <= .5 or error2 >= .75 * zero2:
                return None, 'same_heading_prospective_validation_unavailable'
            residual = float(response['residual'])
            if not np.isfinite(residual) or residual < 0:
                return None, 'same_heading_response_invalid'
        except (ValueError, TypeError, KeyError):
            return None, 'same_heading_response_invalid'
        return {'matrix': matrix, 'basis': basis, 'noise_px': max(.25, residual),
            # A diagnostic forecast at an already measured same-heading step
            # scale. This is never a proposed or authorized robot command.
            'forecast_distance_m': min(cls.maximum_forecast_distance_m,
                                       float(np.median([r[0] for r in rows]))),
            'validation': {'count': len(rows), 'error2': error2, 'zero2': zero2,
                'last_observed_transition': max(r[3] for r in rows),
                'current_observed_transition': observed,
                'same_heading_only': True, 'prediction_precedes_training': True}}, None

    def observe(self, before8, action7, after8, *, wrist_summary, wrist_track_summary,
                image_size_wh, intent_id, target_id, phase, acknowledged=True):
        self.begin_intent(intent_id)
        self.observed_transitions += 1
        self._new_event = False
        before, after = (self._vector(value, 8) for value in (before8, after8))
        action = self._vector(action7, 7)
        continuous = self._last_state is None or np.allclose(before, self._last_state, atol=1e-9, rtol=0.)
        self._last_state = after.copy()
        snap, reason = self._capture(after, wrist_summary, wrist_track_summary, image_size_wh, target_id)
        if snap is None:
            self._clear_trend()
            self._current = {'reference_available': False, 'reason': reason}
            self.status = reason
            return self.summary()
        if snap['key'] != self._key or not continuous:
            self._clear_trend()
            self._key = snap['key']
        self._current = {'reference_available': True, 'target_id': target_id,
            'reference_epoch': snap['epoch'], 'state8': after.tolist(),
            'target_px': snap['point_px'].tolist(), 'desired_target_uv': snap['desired_target_uv'].tolist(),
            'bearing_error_px': snap['error_px'], 'inliers': snap['inliers'],
            'initial_features': snap['initial_features'], 'point_edge_margins_px': dict(
                zip(('left', 'right', 'top', 'bottom'), snap['margins_px'].tolist())),
            'edge_support_band_px': snap['support_radius_px'] + 3.,
            'support_band_is_object_boundary_or_clearance': False}
        delta = after[:3] - before[:3]
        distance, command = float(np.linalg.norm(delta)), float(np.linalg.norm(action[:3]))
        if (acknowledged is not True or command <= 1e-10 or distance <= 1e-6
                or action[6] != 0 or np.linalg.norm(action[3:6]) * .5 > .005):
            self.status = 'observed_without_new_directional_motion_evidence'
            return self.summary()
        direction = delta / distance
        if float(direction @ (action[:3] / command)) < .75:
            self.status = 'actual_motion_does_not_support_command_continuation'
            return self.summary()
        self._path_m += distance
        row = {'error_px': snap['error_px'], 'margins_px': snap['margins_px'].copy(),
            'noise_px': snap['noise_px'], 'inliers': snap['inliers'], 'path_m': self._path_m}
        if self._origin is None:
            self._origin = copy.deepcopy(row)
        self._history.append(row)
        if self._event is not None:
            self.status = 'intent_review_active_until_new_refresh'
            return self.summary()
        response, reason = self._response(wrist_summary, direction)
        self._current['measured_heading_xyz'] = direction.tolist()
        if response is None:
            self.status = reason
            return self.summary()
        edge = int(np.argmin(snap['margins_px']))
        noise = max(snap['noise_px'], self._origin['noise_px'], response['noise_px'])
        forecast_delta = direction * response['forecast_distance_m']
        flow = forecast_delta @ response['matrix']
        predicted_point = snap['point_px'] + flow
        predicted_margins = np.array([predicted_point[0], snap['scale'][0] - predicted_point[0],
                                     predicted_point[1], snap['scale'][1] - predicted_point[1]])
        recent = self._history[-self.recent_window_samples] if len(self._history) >= self.recent_window_samples else self._history[0]
        trend = {'observed_bearing_increase_px': snap['error_px'] - self._origin['error_px'],
            'recent_bearing_increase_px': snap['error_px'] - recent['error_px'],
            'observed_edge_margin_decrease_px': float(self._origin['margins_px'][edge] - snap['margins_px'][edge]),
            'recent_edge_margin_decrease_px': float(recent['margins_px'][edge] - snap['margins_px'][edge]),
            'measured_path_m': self._path_m, 'accepted_motion_samples': len(self._history)}
        prediction = {'forecast_direction_xyz': direction.tolist(),
            'forecast_distance_m': response['forecast_distance_m'], 'predicted_target_px': predicted_point.tolist(),
            'predicted_edge_margin_px': float(predicted_margins[edge]),
            'predicted_edge_margin_decrease_px': float(snap['margins_px'][edge] - predicted_margins[edge]),
            'predicted_bearing_increase_px': float(np.linalg.norm(snap['error_vector_px'] + flow) - snap['error_px']),
            'noise_proxy_px': noise, 'forecast_is_action_permission': False}
        checks = {
            'near_observed_support_boundary': snap['margins_px'][edge] <= snap['support_radius_px'] + 3.,
            'enough_observed_motion': self._path_m >= self.minimum_measured_trend_m,
            'enough_observations': len(self._history) >= self.minimum_trend_samples,
            'sustained_bearing_worsening': trend['observed_bearing_increase_px'] > max(2., 4 * noise),
            'recent_bearing_worsening': trend['recent_bearing_increase_px'] > max(.5, 2 * noise),
            'observed_margin_shrinking': trend['observed_edge_margin_decrease_px'] > max(2., 4 * noise),
            'recent_margin_shrinking': trend['recent_edge_margin_decrease_px'] > max(.5, 2 * noise),
            'continuation_predicts_bearing_worsening': prediction['predicted_bearing_increase_px'] > max(.5, 2 * noise),
            'continuation_predicts_outward_motion': prediction['predicted_edge_margin_decrease_px'] > max(.5, 2 * noise),
        }
        self._current.update(edge=('left', 'right', 'top', 'bottom')[edge],
            current_direction_reliability=response['validation'], trend=trend,
            continuation_prediction=prediction, trigger_checks={k: bool(v) for k, v in checks.items()})
        if not all(checks.values()):
            self.status = 'observing_current_intent_visibility'
            return self.summary()
        event = {'reason': 'wrist_target_visibility_declining_under_current_heading',
            'intent_id': intent_id, 'observed_transition': self.observed_transitions,
            'phase': phase, 'evidence': copy.deepcopy(self._current),
            'instruction': ('Review the still-tracked target, desired bearing and current route. '
                'Choose a justified bounded task-progress intent under existing full geometry checks; '
                'ordinary Field remains available for independent admission. Preserve this current '
                'reference if valid. An unobserved reverse is not validated by the adverse heading.'),
            'automatic_motion': False, 'physical_collision_inferred': False,
            'task_failure_or_success_inferred': False, 'reference_reset_requested': False,
            'physical_allowance_replenished': False, 'permanent_motion_lock': False}
        token = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
        event['event_id'] = 'intent-visibility-review-' + token
        self._event = event
        self._archive.append(copy.deepcopy(event))
        self._new_event = True
        self.status = 'intent_review_required'
        return self.summary()

    def summary(self):
        return {'version': self.version, 'status': self.status, 'intent_id': self.intent_id,
            'observed_transitions': self.observed_transitions,
            'active': self._event is not None, 'review_required': self._event is not None,
            'new_event': self._new_event, 'event': copy.deepcopy(self._event),
            'current_evidence': copy.deepcopy(self._current), 'archive': copy.deepcopy(list(self._archive)),
            'automatic_motion': False, 'physical_allowance_replenished': False,
            'collision_or_task_outcome_certified': False, 'permanent_motion_lock': False}
