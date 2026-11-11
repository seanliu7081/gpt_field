"""Select feasible axis-constrained tracking commands before ordinary admission.

An axis corridor permits small measured offsets; it does not require every
command to remove them.  Keep the original path and goal, but omit a locked-axis
return to the centreline when that return conflicts with a warned object's
current model-distance constraint.  All inherited admission and accounting
still evaluate the resulting command.  Model distances are not real clearance.
"""
from __future__ import annotations

import copy
import numpy as np

from .tracking import PathTracker
from .visual_field_v6 import LOCKED_AXIS_TOLERANCE_M
from .visual_field_v27 import MetricFieldExecutorV27


class _AxisCandidateTracker(PathTracker):
    def command(self, position, field, workspace_min, workspace_max):
        nominal, details = super().command(position, field, workspace_min, workspace_max)
        if (details['status'] != 'tracking'
                or details['reference_progress_m'] <= details['progress_m']+1e-12):
            return nominal, details
        selected, metadata = self._candidate_selector(np.asarray(position, float),
            np.asarray(nominal, float), self, field, workspace_min, workspace_max,
            details['reference_xyz'])
        if metadata is not None:
            details['axis_tracking_constraint'] = metadata
        if selected is None:
            return nominal, details
        details.update(command_target_xyz=(np.asarray(position)+selected).tolist(),
            command_norm_m=float(np.linalg.norm(selected)))
        return tuple(float(v) for v in selected), details


class MetricFieldExecutorV29(MetricFieldExecutorV27):
    """Preserve the fixed route while avoiding unnecessary locked-axis return."""

    def __init__(self):
        super().__init__()
        self.axis_tracking_constraint = None

    def configure(self, *args, **kwargs):
        self.axis_tracking_constraint = None
        return super().configure(*args, **kwargs)

    def _axis_tracking_scoped(self):
        guard = self.translation_guard
        return bool(self.intent is not None and self._route() == 'normal'
            and getattr(self, 'locked_axes', ()) and guard is not None
            and getattr(guard, 'worsening', ())
            and (guard.context or {}).get('translation_mode') == 'transit'
            and type(self.tracker) is PathTracker)

    @staticmethod
    def _warned_nondecreasing(proof, identities):
        objects = proof.get('object_proofs') or {}
        return bool(proof.get('eligible') is True and all(
            identity in objects
            and objects[identity].get('gap_nondecreasing') is True
            and objects[identity].get('full_uncertainty_sweep_disjoint') is True
            and objects[identity]['minimum_segment_gap_m'] >= objects[identity]['start_gap_m']-1e-10
            and objects[identity]['end_gap_m'] >= objects[identity]['start_gap_m']-1e-10
            for identity in identities))

    def _axis_candidate(self, point, nominal, tracker, checked, workspace_min, workspace_max,
                        reference_xyz):
        axes = list(self.locked_axes)
        identities = sorted(self.translation_guard.worsening)
        # No work, prediction or changed command for an already planar proposal.
        if not axes or not np.any(nominal[axes] != 0.):
            return None, None
        metadata = {'version': 'v29-locked-axis-tracking-candidate',
            'nominal_delta_m': nominal.tolist(), 'selected_delta_m': None,
            'locked_axes': axes, 'warned_object_ids': identities,
            'fixed_goal_xyz_m': np.asarray(self.goal).tolist(),
            'path_changed': False, 'goal_changed': False, 'command_enlarged': False,
            'physical_budget_replenished': False, 'clearance_certified': False,
            'reason': 'nominal_tracking_retained'}

        def retain(reason):
            metadata['reason'] = reason
            self.axis_tracking_constraint = copy.deepcopy(metadata)
            return None, metadata

        # Use the complete original corridor, including a resumed subgoal's
        # original origin.  A small displacement never excuses existing drift.
        if self._outside_corridor(point):
            return retain('original_axis_corridor_exceeded')
        if (tracker.positions is None or not len(tracker.positions)
                or not np.isfinite(np.r_[point, nominal]).all()):
            return retain('fixed_path_unavailable')
        positions = np.asarray(tracker.positions, float)
        if not np.all(np.abs(positions[:, axes]-positions[0, axes]) <= 1e-12):
            return retain('selected_path_not_axis_constrained')
        offset = float(np.linalg.norm(point[axes]-positions[-1, axes]))
        bound = min(LOCKED_AXIS_TOLERANCE_M, float(tracker.endpoint_tolerance_m))
        metadata.update(locked_offset_norm_m=offset, allowed_offset_norm_m=bound,
            original_corridor_tolerance_m=LOCKED_AXIS_TOLERANCE_M,
            endpoint_tolerance_m=float(tracker.endpoint_tolerance_m))
        if not 0. <= offset < bound:
            return retain('offset_cannot_fit_existing_endpoint_tolerance')
        state = np.asarray(self.expected_state, float)
        nominal_proof = self._transit_reconciliation_contract(state, point, point+nominal)
        if nominal_proof.get('eligible') is not True:
            return retain('nominal_candidate_not_geometry_eligible')
        if self._warned_nondecreasing(nominal_proof, identities):
            return retain('nominal_candidate_already_satisfies_model_constraint')
        candidate = nominal.copy()
        candidate[axes] = 0.
        norm = float(np.linalg.norm(candidate))
        reference_delta = np.asarray(reference_xyz, float)-point
        reference_delta[axes] = 0.
        forward_limit = float(np.linalg.norm(reference_delta))
        size = min(float(np.linalg.norm(nominal)), forward_limit)
        if norm <= 1e-12 or size < self.MIN_USEFUL_COMMAND_M-1e-12:
            return retain('projected_candidate_below_useful_motion_floor')
        # Preserve the requested total norm, not the slightly shorter active
        # component of a mixed recentering command. Never pass its selected
        # reference (which can be an earlier vertex at a masked corner).
        candidate *= size/norm
        target = point+candidate
        # Active-axis path projection is unchanged, so corners are not skipped.
        # Keep strict Euclidean goal progress and the ordinary endpoint check.
        if (float(np.linalg.norm(target-self.goal)) >= float(np.linalg.norm(point-self.goal))
                or np.any(target < np.asarray(workspace_min))
                or np.any(target > np.asarray(workspace_max))
                or self._outside_corridor(target)
                or not checked.segment_valid(point, target)):
            return retain('projected_candidate_lacks_checked_goal_progress')
        proof = self._transit_reconciliation_contract(state, point, target)
        if not self._warned_nondecreasing(proof, identities):
            return retain('projected_candidate_not_geometry_eligible')
        metadata.update(reason='locked_axis_return_suppressed_before_admission',
            selected_delta_m=candidate.tolist(), suppressed_delta_m=(nominal-candidate).tolist(),
            geometry_sha256=proof['geometry_sha256'], observation_epoch=proof['observation_epoch'],
            exact_candidate_proof_checked=True, inherited_admission_still_required=True)
        self.axis_tracking_constraint = copy.deepcopy(metadata)
        return candidate, metadata

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        self.axis_tracking_constraint = None
        original = self.tracker if self.intent is not None else None
        replacement = None
        if self._axis_tracking_scoped():
            # Temporary adapter: retain the original tracker identity and all
            # path/progress state.  Preview and repeated cap selection execute
            # this same path; neither callbacks nor owner cycles are persisted.
            replacement = _AxisCandidateTracker(lookahead_m=original.lookahead_m,
                max_command_m=original.max_command_m,
                endpoint_tolerance_m=original.endpoint_tolerance_m)
            replacement.__dict__.update(original.__dict__)
            replacement._candidate_selector = self._axis_candidate
            self.tracker = replacement
        try:
            result = super().select(state8, joint_position, command_cap_m=command_cap_m,
                                   measured_cap_m=measured_cap_m)
        finally:
            if replacement is not None:
                original.__dict__.update({k:v for k,v in replacement.__dict__.items()
                                         if k != '_candidate_selector'})
                self.tracker = original
        result.setdefault('diagnostics', {})['axis_tracking_constraint'] = copy.deepcopy(self.axis_tracking_constraint)
        return result

    def summary(self):
        result = super().summary()
        result.update(version='constraint-aware-axis-tracking-v29',
            axis_tracking_constraint=copy.deepcopy(getattr(self, 'axis_tracking_constraint', None)))
        return result
