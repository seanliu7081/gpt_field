"""Immutable route identity across finite goal serialization roundoff.

The retained subgoal is re-expressed as a relative offset rounded to twelve
decimal places in metres. Reconstructing the same world goal can cross a
rounding-bin boundary. Compare that numerical error to the ORIGINAL raw goal;
never move the binding or grant a new physical allowance.
"""
from __future__ import annotations

import numpy as np

from .visual_egress_evidence_v23 import EgressEvidenceV23
from .visual_route_exposure_v24 import RouteExposureV24


class RouteExposureV27(RouteExposureV24):
    goal_serialization_atol_m = 2e-12

    @staticmethod
    def _finite_goal(proof):
        goal = np.asarray(proof['complete_route_goal_xyz'], float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('Route goal must be three finite metres')
        return goal

    def _geometry_context(self, proof):
        base = EgressEvidenceV23._geometry_context(proof)
        goal = self._finite_goal(proof)
        candidate = base + (tuple(goal),)
        committed = self.committed_geometry_context
        if committed is not None and base == committed[:-1]:
            original = np.asarray(committed[-1], float)
            if (original.shape == (3,) and np.isfinite(original).all()
                    and np.max(np.abs(goal-original)) <= self.goal_serialization_atol_m):
                # Pure canonicalization to an immutable first-acknowledged
                # goal, not a rolling tolerance or a stored-goal revision.
                return committed
        return candidate

    def assess(self, proof, *args, **kwargs):
        try:
            self._finite_goal(proof)
        except (KeyError, TypeError, ValueError, OverflowError):
            return {'accepted': False, 'cap_m': 0.,
                'reason': 'route_exposure_goal_binding_invalid',
                'clearance_certified': False}
        return super().assess(proof, *args, **kwargs)

    def observe(self, before_state8, action7, after_state8, *, pending, **kwargs):
        if pending is not None:
            try:
                self._finite_goal(pending['proof'])
            except (KeyError, TypeError, ValueError, OverflowError):
                # A malformed acknowledgement cannot establish a route.
                # Retain inherited physical accounting of global debt and
                # residual holds, then latch the invalid evidence.
                super().observe(before_state8, action7, after_state8,
                    pending=None, **kwargs)
                self.last_reason = 'route_exposure_goal_binding_invalid'
                self.last_progress = {'worsening': True, 'reason': self.last_reason}
                return self.summary()
        return super().observe(before_state8, action7, after_state8,
            pending=pending, **kwargs)

    def summary(self):
        result = super().summary()
        result.update(version='immutable-numerical-route-goal-exposure-v27',
            goal_binding_absolute_tolerance_m=self.goal_serialization_atol_m,
            goal_binding_relative_tolerance=0.,
            original_goal_mutated_by_roundoff_comparison=False)
        return result
