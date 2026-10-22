"""Finite, explicitly selected short reversal under current model geometry.

The original four-millimetre image checkpoint remains diagnostic. It does not
mint travel credit: the fixed route, per-direction eight-millimetre budget and
shared measured debt bound exposure even when image separation is inconclusive.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v24 import ProjectedTranslationGuardV24


class ProjectedTranslationGuardV25(ProjectedTranslationGuardV24):
    version = 'explicit-finite-observed-reverse-exposure-v25'

    def __init__(self):
        super().__init__()
        self.last_finite_reverse = None
        self._finite_reverse_original_checkpoint = None

    def _experiment_for(self, direction, *, nominal_away):
        # Distinct-heading eligibility is a separate inherited review rule.
        # A finite-route exposure must not silently move that threshold.
        active = self.information_checkpoint_m
        original = self._finite_reverse_original_checkpoint
        if original is not None:
            self.information_checkpoint_m = original
        try:
            return super()._experiment_for(direction, nominal_away=nominal_away)
        finally:
            self.information_checkpoint_m = active

    def _finite_reverse_contract(self, records, command, rotation_rad):
        current = self._contract(records, command, rotation_rad)
        if current is None:
            return None
        route = current.get('finite_reverse_route')
        if not isinstance(route, dict):
            return None
        try:
            origin = np.asarray(route['origin_xyz_m'], float)
            goal = np.asarray(route['goal_xyz_m'], float)
            budget = float(route['max_measured_travel_m'])
            spent = float(route['measured_travel_m'])
            remaining = float(route['remaining_measured_travel_m'])
            valid = (route['version'] == 'v25-current-fixed-observed-reverse'
                and route['route_family'] == 'reverse_observed'
                and isinstance(route['route_id'], str) and bool(route['route_id'])
                and origin.shape == goal.shape == (3,)
                and np.isfinite(np.r_[origin, goal, budget, spent, remaining, route['length_m']]).all()
                and 0 < route['length_m'] <= .008+1e-10
                and abs(np.linalg.norm(goal-origin)-route['length_m']) <= 1e-10
                and 0 < budget <= min(.008, self.experiment_limit_m)+1e-10
                and 0 <= spent < budget and 0 < remaining <= budget-spent+1e-10
                and abs(remaining-current['remaining_measured_travel_m']) <= 1e-10
                and route['max_actions']-route['actions_used'] == current['remaining_actions']
                and route['geometry_sha256'] == current['geometry_sha256']
                and route['observation_epoch'] == current['observation_epoch']
                and np.array_equal(route['state8'], current['state8'])
                and np.array_equal(route['start_xyz_m'], current['start_xyz_m'])
                and np.array_equal(route['end_xyz_m'], current['end_xyz_m']))
        except (KeyError, ValueError, TypeError):
            return None
        return copy.deepcopy(route) if valid else None

    def assess(self, delta_xyz, **kwargs):
        command = np.asarray(delta_xyz, float)
        route = self._finite_reverse_contract(kwargs.get('nominal_regions') or [],
            command, float(kwargs.get('rotation_rad', 0.)))
        checkpoint = self.information_checkpoint_m
        previous = self._finite_reverse_original_checkpoint
        # Scope this to one currently proven, explicitly selected proposal.
        # Existing V21 logic still owns reference/domain/worsening checks and
        # all inherited physical accounting. Nothing is cleared on refresh.
        if route is not None:
            self._finite_reverse_original_checkpoint = checkpoint
            self.information_checkpoint_m = route['max_measured_travel_m']
        try:
            result = super().assess(delta_xyz, **kwargs)
        finally:
            self.information_checkpoint_m = checkpoint
            self._finite_reverse_original_checkpoint = previous
        self.last_finite_reverse = None if route is None else {
            'route': route, 'legacy_image_checkpoint_m': checkpoint,
            'finite_measured_limit_m': route['max_measured_travel_m'],
            'positive_image_gap_required_for_travel_credit': False,
            'all_historical_worsening_and_debt_retained': True,
            'current_further_worsening_remains_a_veto': True,
            'physical_budget_replenished': False, 'clearance_certified': False,
            'accepted': result['accepted'], 'reason': result['reason']}
        result.setdefault('diagnostics', {})['finite_reverse_exposure'] = copy.deepcopy(self.last_finite_reverse)
        return result

    def summary(self):
        result = super().summary()
        result['finite_reverse_exposure'] = copy.deepcopy(getattr(self, 'last_finite_reverse', None))
        return result
