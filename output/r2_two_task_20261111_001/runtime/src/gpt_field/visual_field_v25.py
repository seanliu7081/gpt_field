"""Explicit Field route dispatch and checked cone-constrained reverse tracking.

Route labels select an implementation, never permission to move. All existing
current geometry, image, robot response, physical debt and acknowledgment checks
still evaluate the actual command produced here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np

from .tracking import PathTracker
from .visual_field_v24 import MetricFieldExecutorV24


RECOVERY_ROUTES = frozenset(('normal', 'reverse_observed', 'initial_uncertainty_egress'))


def constrained_recovery_delta(desired, route_direction, reverse_axes, tolerance_rad):
    """Keep a desired command in every existing reverse cone without enlarging it.

    The fixed route heading was checked against all observed-motion reversals.
    Intersect the ray from that feasible heading toward the desired correction
    with their convex cones. This retains as much lateral correction as that ray
    admits; it is not an assertion of a globally optimal control projection.
    """
    desired = np.asarray(desired, float)
    route = np.asarray(route_direction, float)
    axes = np.asarray(reverse_axes, float)
    if (desired.shape != (3,) or route.shape != (3,) or axes.ndim != 2
            or axes.shape[1:] != (3,) or not len(axes)
            or not np.isfinite(np.r_[desired, route, axes.ravel()]).all()
            or not 0. < tolerance_rad < math.pi/2):
        return None
    size, route_size = float(np.linalg.norm(desired)), float(np.linalg.norm(route))
    axis_sizes = np.linalg.norm(axes, axis=1)
    if size <= 1e-12 or route_size <= 1e-12 or np.any(axis_sizes <= 1e-12):
        return None
    route, axes = route/route_size, axes/axis_sizes[:, None]
    cosine = math.cos(tolerance_rad)

    def inside(vector):
        return bool(np.all(axes@vector >= cosine*np.linalg.norm(vector)-1e-12))

    if not inside(route):
        return None
    if inside(desired):
        return desired.copy()
    # Stay infinitesimally inside the unchanged boundary so independent acos
    # validation does not turn rounding into a false >30 degree refusal.
    unit = desired/size
    low, high = 0., 1.
    for _ in range(52):
        middle = (low+high)*.5
        vector = route*(1.-middle)+unit*middle
        if inside(vector):
            low = middle
        else:
            high = middle
    weight = max(0., low-1e-9)
    vector = route*(1.-weight)+unit*weight
    return vector/np.linalg.norm(vector)*size


class _RecoveryConeTracker(PathTracker):
    """Constrain before the executor creates and validates a pending action."""

    def command(self, position, field, workspace_min, workspace_max):
        desired, details = super().command(position, field, workspace_min, workspace_max)
        if details['status'] != 'tracking':
            return desired, details
        desired = np.asarray(desired, float)
        selected = constrained_recovery_delta(desired, self.route_direction,
            self.reverse_axes, self.reverse_tolerance_rad)
        point, goal = np.asarray(position, float), self.positions[-1]
        metadata = {'requested_delta_m': desired.tolist(),
            'reverse_axes_xyz': self.reverse_axes.tolist(),
            'maximum_heading_error_rad': self.reverse_tolerance_rad,
            'guard_cone_widened': False, 'command_enlarged': False,
            'fixed_goal_xyz': goal.tolist()}

        def stop(reason):
            metadata['reason'] = reason
            details.update(status=reason, recovery_heading_constraint=metadata,
                command_target_xyz=point.tolist(), command_norm_m=0.)
            return (0., 0., 0.), details

        if selected is None:
            return stop('recovery_cone_has_no_qualified_direction')
        norm = float(np.linalg.norm(selected))
        direction = selected/norm
        goal_projection = float((goal-point)@direction)
        if goal_projection <= 1e-12:
            return stop('recovery_cone_has_no_goal_progress')
        # Do not cross the fixed goal's normal plane merely to satisfy a cone.
        selected *= min(1., goal_projection/norm)
        target = point+selected
        if (np.any(target < np.asarray(workspace_min))
                or np.any(target > np.asarray(workspace_max))
                or not field.segment_valid(point, target)):
            return stop('recovery_constrained_segment_blocked')
        metadata.update(reason='actual_command_inside_observed_reverse_cones',
            changed=not np.allclose(selected, desired, atol=1e-12, rtol=0.),
            selected_delta_m=selected.tolist(),
            heading_errors_rad=[float(math.acos(np.clip(direction@axis, -1., 1.)))
                                for axis in self.reverse_axes])
        details.update(command_target_xyz=target.tolist(),
            command_norm_m=float(np.linalg.norm(selected)),
            recovery_heading_constraint=metadata)
        return tuple(float(x) for x in selected), details


class MetricFieldExecutorV25(MetricFieldExecutorV24):
    def __init__(self):
        super().__init__()
        self.recovery_route_binding = None

    def configure(self, *args, **kwargs):
        self.recovery_route_binding = None
        super().configure(*args, **kwargs)
        prospective = getattr(self, '_prospective_recovery_route', None)
        if prospective is not None:
            self.bind_recovery_route(prospective, self.expected_state)
        return self.summary()

    def bind_recovery_route(self, recovery_route, state8):
        """Bind the model selector to this configured fixed goal; spend no debt."""
        if recovery_route not in RECOVERY_ROUTES:
            return {'accepted': False, 'reason': 'recovery_route_invalid'}
        if (self.intent is None or self.pending is not None
                or not self._same(state8, self.expected_state, (8,))
                or not self._same(state8, getattr(self.translation_guard, 'state', None), (8,))):
            return {'accepted': False, 'reason': 'recovery_route_requires_current_configured_state'}
        previous = self.recovery_route_binding
        if previous is not None:
            if previous['recovery_route'] != recovery_route:
                return {'accepted': False, 'reason': 'recovery_route_change_requires_configure'}
            if recovery_route != 'normal' and not self._same(previous['goal_xyz'], self.goal, (3,)):
                return {'accepted': False, 'reason': 'recovery_route_fixed_goal_changed'}
            return {'accepted': True, 'reason': 'recovery_route_already_bound'}
        self.recovery_route_binding = {'recovery_route': recovery_route,
            'goal_xyz': self.goal.tolist(), 'origin_xyz': self.origin.tolist(),
            'bound_state8': np.asarray(state8, float).tolist(),
            'source': 'explicit_model_route_or_same_prospective_preview',
            'physical_debt_replenished': False}
        return {'accepted': True, 'reason': 'recovery_route_bound'}

    def _route(self):
        return (self.recovery_route_binding or {}).get('recovery_route')

    def _egress_requested(self):
        return self._route() == 'initial_uncertainty_egress' and super()._egress_requested()

    def _recovery_requested(self):
        return self._route() == 'reverse_observed' and super()._recovery_requested()

    def _retreat_geometry_contract(self, state, start, end):
        contract = super()._retreat_geometry_contract(state, start, end)
        binding = self.recovery_route_binding or {}
        if (self._route() != 'reverse_observed' or not self._recovery_requested()
                or self.intent is None or self.checked is None
                or not self._same(state, self.expected_state, (8,))
                or not self._same(state, getattr(self.translation_guard, 'state', None), (8,))
                or not self._same(start, np.asarray(state)[:3], (3,))
                or not self._same(binding.get('origin_xyz'), self.origin, (3,))
                or not self._same(binding.get('goal_xyz'), self.goal, (3,))
                or not contract['full_scene_sweep_accepted']
                or not contract['outside_target_contact_band']
                or not contract['empty_tool'] or contract['target_contact'] != 'none'
                or contract['contact_or_visual_goal_dependency']):
            return contract
        length = float(np.linalg.norm(self.goal-self.origin))
        limit = float(self.intent['max_travel_m'])
        if (not 0. < length <= .008+1e-10 or not 0. < limit <= .008+1e-10
                or contract['remaining_actions'] <= 0
                or contract['remaining_measured_travel_m'] <= 0.
                or not self.checked.segment_valid(start, end)
                or not self.checked.segment_valid(self.origin, self.goal)
                or not self.checked.segment_valid(np.asarray(state)[:3], self.goal)):
            return contract
        positions = (tuple(float(x) for x in self.origin), tuple(float(x) for x in self.goal))
        identity = hashlib.sha256(json.dumps({'positions': positions,
            'geometry_sha256': contract['geometry_sha256']}, sort_keys=True).encode()).hexdigest()
        route = self.heading_recovery
        if route is not None and (route['route_id'] != identity
                or route['geometry_sha256'] != contract['geometry_sha256']):
            return contract
        contract['finite_reverse_route'] = {
            'version': 'v25-current-fixed-observed-reverse',
            'route_family': 'reverse_observed', 'route_id': identity,
            'origin_xyz_m': self.origin.tolist(), 'goal_xyz_m': self.goal.tolist(),
            'length_m': length, 'max_measured_travel_m': limit,
            'measured_travel_m': self.measured_travel_m,
            'remaining_measured_travel_m': contract['remaining_measured_travel_m'],
            'max_actions': self.intent['max_actions'], 'actions_used': self.actions,
            'geometry_sha256': contract['geometry_sha256'],
            'observation_epoch': contract['observation_epoch'],
            'state8': np.asarray(state).tolist(), 'start_xyz_m': np.asarray(start).tolist(),
            'end_xyz_m': np.asarray(end).tolist(),
            'source': 'explicit_model_fixed_route_not_visual_gap_evidence',
            'clearance_certified': False, 'physical_budget_replenished': False}
        return contract

    def _prepare_heading_route(self, state):
        reason = super()._prepare_heading_route(state)
        if reason is not None or self.heading_recovery is None:
            return reason
        guard = self.translation_guard
        identities = sorted(guard.worsening)
        if not identities and getattr(guard, '_retreat_committed', None):
            identities = guard._retreat_committed['episode_ids']
        axes = np.asarray([-np.asarray(guard.worsening_motion[i]['measured_direction_xyz'])
                           for i in identities], float)
        if not isinstance(self.tracker, _RecoveryConeTracker):
            replacement = _RecoveryConeTracker(lookahead_m=self.tracker.lookahead_m,
                max_command_m=self.tracker.max_command_m,
                endpoint_tolerance_m=self.tracker.endpoint_tolerance_m)
            replacement.__dict__.update(copy.deepcopy(self.tracker.__dict__))
            self.tracker = replacement
        self.tracker.route_direction = np.asarray(self.heading_recovery['unit_direction_xyz'])
        self.tracker.reverse_axes = axes/np.linalg.norm(axes, axis=1)[:, None]
        self.tracker.reverse_tolerance_rad = guard.reverse_tolerance_rad
        return None

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        if self.pending is not None:
            raise RuntimeError('observe must acknowledge previous Field action')
        if self.recovery_route_binding is None:
            return self._stop('explicit_recovery_route_unbound')
        route = self._route()
        if route != 'normal' and not self._same(self.recovery_route_binding['goal_xyz'], self.goal, (3,)):
            return self._stop('recovery_route_fixed_goal_changed')
        if route == 'initial_uncertainty_egress' and not self._egress_requested():
            return self._stop('egress_explicit_route_role_mismatch')
        if route == 'reverse_observed' and not self._recovery_requested():
            return self._stop('reverse_explicit_route_evidence_unavailable')
        return super().select(state8, joint_position, command_cap_m=command_cap_m,
                              measured_cap_m=measured_cap_m)

    def _preview_family(self, family, method, *args, **kwargs):
        original = self.__dict__
        working = original.copy()
        working['_prospective_recovery_route'] = family
        try:
            self.__dict__ = working
            return method(*args, **kwargs)
        finally:
            self.__dict__ = original

    def preview_recovery(self, *args, **kwargs):
        return self._preview_family('reverse_observed', super().preview_recovery, *args, **kwargs)

    def preview_egress(self, *args, **kwargs):
        return self._preview_family('initial_uncertainty_egress', super().preview_egress, *args, **kwargs)

    def summary(self):
        result = super().summary()
        result.update(version='explicit-route-cone-tracking-field-v25',
            recovery_route_binding=copy.deepcopy(getattr(self, 'recovery_route_binding', None)))
        return result
