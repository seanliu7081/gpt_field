"""Representation-bound whole-outline evidence for acknowledged reversals.

Use ahead of ProjectedTranslationGuardV27 in the MRO. This mixin changes only
reverse image-evidence interpretation; the inherited observer still owns action
acknowledgement, actual travel, epoch changes and every physical allowance.
"""
from __future__ import annotations

import copy

from .visual_boundary_evidence_v27 import compare_gap_evidence


class ReverseBoundaryEvidenceV28:
    reverse_boundary_minimum_travel_m = .001

    @staticmethod
    def _reverse_boundary_comparison(previous, current, epoch):
        if (previous is None or current is None
                or previous.get('measurement_epoch') != epoch
                or current.get('measurement_epoch') != epoch):
            return {'comparable': False, 'classification': 'unavailable',
                'reason': 'unavailable_current_comparable_snapshot',
                'gap_change_interval_px': None, 'nominal_gap_change_px': None,
                'clearance_certified': False}
        return compare_gap_evidence(previous.get('tool_boundary'),
            current.get('tool_boundary'), previous.get('neighbor_boundary'),
            current.get('neighbor_boundary'))

    def _update_reverse_measurements(self, original, current):
        progress = self.retreat_progress
        if progress is None:
            return
        first = getattr(self, '_v28_first_reverse_observation', None)
        if (first is not None and first['prior_progress'] is None
                and not first['reclassified']):
            # V21 creates the first progress record during acknowledgement and
            # applies its scalar test before V26 invokes this hook. There was
            # NO prior reverse latch to preserve in this case: classify that
            # same first observation once using the new boundary evidence.
            # Existing progress records (including latched ones) never enter.
            progress['further_worsening'] = False
            first['reclassified'] = True
        segments = copy.deepcopy(self._reverse_measurement_segments)
        improving, comparable, evidence = [], [], {}
        for identity in progress['episode_ids']:
            epoch = self.neighbor_measurement_epochs.get(identity, 0)
            segment = segments.get(identity)
            if segment is None:
                segment = self._segment(identity, original, 0.)
            if segment['measurement_epoch'] != epoch:
                segment = self._segment(identity, None, progress['measured_travel_m'])
            previous = segment['baseline']
            now = None if current is None else current['neighbors'].get(identity)
            travel = max(0., progress['measured_travel_m']-segment['start_measured_travel_m'])
            comparison = self._reverse_boundary_comparison(previous, now, epoch)
            enough = travel >= self.reverse_boundary_minimum_travel_m
            opening = bool(comparison['comparable'] and enough
                and comparison['classification'] == 'opening')
            closing = bool(comparison['comparable'] and enough
                and comparison['classification'] == 'closing')
            if closing:
                progress['further_worsening'] = True
            if previous is None and now is not None and now.get('measurement_epoch') == epoch:
                # Current evidence can start a new temporal baseline, but may
                # not borrow old travel or manufacture opening/closing credit.
                segment = self._segment(identity, current, progress['measured_travel_m'])
                travel = 0.
            segment.update(segment_measured_travel_m=travel, improved=opening,
                gap_change_px=comparison.get('nominal_gap_change_px'),
                threshold_px=None, boundary_evidence=comparison,
                boundary_comparable=bool(comparison['comparable']),
                minimum_measured_travel_m=self.reverse_boundary_minimum_travel_m,
                evidence=('unavailable_current_comparable_snapshot'
                    if not comparison['comparable'] else 'worsening' if closing
                    else 'improving' if opening else 'inconclusive'),
                evidence_method='empirical_whole_outline_sensitivity',
                is_clearance_evidence=False)
            segments[identity] = segment
            improving.append(opening)
            comparable.append(bool(comparison['comparable']))
            evidence[identity] = {key: copy.deepcopy(value) for key, value in segment.items()
                                  if key != 'baseline'}
        self._reverse_measurement_segments = segments
        progress['per_neighbor'] = evidence
        progress['all_trigger_gaps_improved'] = bool(improving and all(improving))
        progress['current_boundary_comparable'] = bool(comparable and all(comparable))

    def reverse_boundary_qualification(self):
        """Pure current-support check; its result grants no motion by itself."""
        progress = self.retreat_progress
        if progress is not None and progress.get('further_worsening'):
            return {'qualified': False, 'reason': 'translation_reverse_observed_further_worsening',
                'prior_adverse_latch_preserved': True, 'prior_latch': True,
                'available': False, 'current_closing': False, 'clearance_certified': False}
        identities = (progress['episode_ids'] if progress is not None
                      else sorted(self.worsening))
        current = self._usable_snapshot()
        results, closing = {}, []
        for identity in identities:
            epoch = self.neighbor_measurement_epochs.get(identity, 0)
            now = None if current is None else current['neighbors'].get(identity)
            segment = self._reverse_measurement_segments.get(identity)
            previous = now if progress is None else None if segment is None else segment['baseline']
            results[identity] = self._reverse_boundary_comparison(previous, now, epoch)
            travel = (0. if progress is None or segment is None else max(0.,
                progress['measured_travel_m']-segment['start_measured_travel_m']))
            if (results[identity]['comparable']
                    and results[identity]['classification'] == 'closing'
                    and travel >= self.reverse_boundary_minimum_travel_m):
                closing.append(identity)
        available = bool(results and all(row['comparable'] for row in results.values()))
        qualified = bool(available and not closing)
        return {'qualified': qualified,
            'reason': ('translation_reverse_observed_further_worsening' if closing
                       else 'current_comparable_reverse_boundary' if qualified
                       else 'translation_reverse_boundary_evidence_unavailable'),
            'per_neighbor': results, 'current_closing_ids': closing, 'opening_required': False,
            'available': available, 'current_closing': bool(closing), 'prior_latch': False,
            'physical_allowance_added_m': 0., 'clearance_certified': False}

    def observe(self, *args, **kwargs):
        previous = getattr(self, '_v28_first_reverse_observation', None)
        self._v28_first_reverse_observation = {
            'prior_progress': self.retreat_progress, 'reclassified': False}
        try:
            return super().observe(*args, **kwargs)
        finally:
            if previous is None:
                del self._v28_first_reverse_observation
            else:
                self._v28_first_reverse_observation = previous
