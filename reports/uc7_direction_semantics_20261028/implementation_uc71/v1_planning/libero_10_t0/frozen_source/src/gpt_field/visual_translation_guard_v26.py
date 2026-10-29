"""Compare image-risk trends only within one measurement representation.

An accepted box or feature revision changes a measurement, not the physical
world. Keep its new current risk and every physical ledger, while starting a
new temporal segment for the affected neighbor. The original reverse baseline
is retained as history but is never subtracted across representation epochs.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v25 import ProjectedTranslationGuardV25
from .visual_neighbor_track_v26 import ExactDescriptorNeighborTrackV26


class ProjectedTranslationGuardV26(ProjectedTranslationGuardV25):
    version = 'representation-segmented-risk-trends-v26'

    @staticmethod
    def _seed_neighbor(pixels, box, identity):
        # Same five bounded seed locations and fixed-corner selection as V11.
        # Only exact-descriptor subpixel matching differs; no category or
        # camera-stationarity prior determines the observed obstacle motion.
        candidates = []
        for fx, fy in ((.5, .5), (.25, .25), (.75, .25), (.25, .75), (.75, .75)):
            try:
                candidates.append(ExactDescriptorNeighborTrackV26(
                    pixels, box[:2] + (box[2:]-box[:2])*[fx, fy], identity,
                    support_radius_px=12., maximum_features=8, roi_uv=box))
            except ValueError:
                pass
        return max(candidates, key=lambda track: track.last_inliers) if candidates else None

    def __init__(self):
        super().__init__()
        self.measurement_epoch = 0
        self.neighbor_measurement_epochs = {}
        self._measurement_available = False
        self._reverse_measurement_segments = {}
        self.last_measurement_revision = None

    @staticmethod
    def _same_box(first, second):
        if first is None or second is None:
            return first is second
        return bool(np.allclose(first, second, atol=1e-7, rtol=0.))

    def _representation(self):
        return {'tool': self.tool_track,
            'tool_box': None if self.tool_box is None else self.tool_box.copy(),
            'scale': None if self.scale is None else self.scale.copy(),
            'neighbors': {row['id']: (row['track'], self._transport(row).copy())
                          for row in self.neighbors}}

    def _snapshot(self):
        snapshot = super()._snapshot()
        if snapshot is not None:
            snapshot['measurement_epoch'] = getattr(self, 'measurement_epoch', 0)
            epochs = getattr(self, 'neighbor_measurement_epochs', {})
            for identity, row in snapshot['neighbors'].items():
                row['measurement_epoch'] = epochs.get(identity, 0)
        return snapshot

    def _usable_snapshot(self):
        if self.context_invalid or self.domain_invalid or not self._measurement_available:
            return None
        return self._snapshot()

    def _segment(self, identity, snapshot, start_travel):
        baseline = None if snapshot is None else snapshot['neighbors'].get(identity)
        epoch = self.neighbor_measurement_epochs.get(identity, 0)
        if baseline is not None and baseline.get('measurement_epoch') != epoch:
            baseline = None
        return {'measurement_epoch': epoch, 'baseline': copy.deepcopy(baseline),
            'start_measured_travel_m': float(start_travel),
            'segment_measured_travel_m': 0., 'evidence': 'not_yet_observed',
            'gap_change_px': None, 'threshold_px': None, 'improved': False,
            'is_clearance_evidence': False}

    def configure(self, *args, **kwargs):
        before = self._representation()
        previously_available = self._measurement_available
        super().configure(*args, **kwargs)
        after = self._representation()
        valid = self.context_invalid is None
        tool_changed = (before['tool'] is not after['tool']
            or not self._same_box(before['tool_box'], after['tool_box'])
            or not self._same_box(before['scale'], after['scale']))
        identities = set(before['neighbors']) | set(after['neighbors'])
        changed = {identity for identity in identities
            if (identity not in before['neighbors'] or identity not in after['neighbors']
                or before['neighbors'][identity][0] is not after['neighbors'][identity][0]
                or not self._same_box(before['neighbors'][identity][1], after['neighbors'][identity][1]))}
        anchor_track_changed = any(identity in before['neighbors'] and identity in after['neighbors']
            and before['neighbors'][identity][0] is not after['neighbors'][identity][0]
            for identity in identities)
        # Invalid declarations never become temporal evidence. A later valid
        # declaration starts a fresh comparison instead of bridging that gap.
        if tool_changed or not previously_available or not valid:
            changed = identities
        if changed or (not valid and previously_available):
            self.measurement_epoch += 1
        epochs = dict(self.neighbor_measurement_epochs)
        for identity in changed:
            epochs[identity] = self.measurement_epoch
        self.neighbor_measurement_epochs = epochs
        self._measurement_available = valid
        snapshot = self._usable_snapshot()
        # V12 already restarted the ordinary risk window on configure. Its
        # same-observation snapshot needs the newly committed epoch tags too.
        self._previous_snapshot = copy.deepcopy(snapshot)
        if anchor_track_changed:
            # Cumulative relative flow must not subtract coordinates belonging
            # to a newly seeded neighbor. Box-only revisions keep that origin.
            self._response_origin = None
            self._response_age = 0
        segments = copy.deepcopy(self._reverse_measurement_segments)
        progress = self.retreat_progress
        if progress is not None:
            for identity in progress['episode_ids']:
                if identity in changed or identity not in segments:
                    segments[identity] = self._segment(identity, snapshot, progress['measured_travel_m'])
            # Revision itself supplies no favorable credit. Existing adverse
            # latches and historical worsening are deliberately never cleared.
            if any(identity in changed for identity in progress['episode_ids']):
                progress['all_trigger_gaps_improved'] = False
        self._reverse_measurement_segments = segments
        self.last_measurement_revision = {'accepted_context': valid,
            'epoch': self.measurement_epoch, 'tool_representation_changed': tool_changed,
            'changed_neighbor_ids': sorted(changed), 'is_observed_motion': False,
            'current_risk_extent_preserved': True, 'physical_debt_replenished': False,
            'prior_worsening_or_latch_cleared': False}
        return self.summary()

    def _update_reverse_measurements(self, original, current):
        progress = self.retreat_progress
        if progress is None:
            return
        segments = copy.deepcopy(self._reverse_measurement_segments)
        statuses, evidence = [], {}
        for identity in progress['episode_ids']:
            epoch = self.neighbor_measurement_epochs.get(identity, 0)
            segment = segments.get(identity)
            if segment is None:
                # First matched reverse action: use its actual pre-action
                # snapshot, not an earlier normal-motion configure baseline.
                segment = self._segment(identity, original, 0.)
            if segment['measurement_epoch'] != epoch:
                segment = self._segment(identity, None, progress['measured_travel_m'])
            a = segment['baseline']
            b = None if current is None else current['neighbors'].get(identity)
            segment['segment_measured_travel_m'] = max(0., progress['measured_travel_m']-segment['start_measured_travel_m'])
            comparable = (a is not None and b is not None
                and a.get('measurement_epoch') == b.get('measurement_epoch') == epoch)
            if not comparable:
                segment.update(evidence='unavailable_current_comparable_snapshot', improved=False,
                    gap_change_px=None, threshold_px=None)
                # Establish a baseline only from this genuinely current frame;
                # it supplies no progress credit and spends no travel allowance.
                if a is None and b is not None and b.get('measurement_epoch') == epoch:
                    segment = self._segment(identity, current, progress['measured_travel_m'])
            else:
                change = float(b['gap']-a['gap'])
                noise = float(max(.75, 3*np.hypot(a['noise'], b['noise'])))
                enough = segment['segment_measured_travel_m'] >= .001
                improved = bool(enough and change > noise)
                worse = bool(enough and change < -noise)
                if worse:
                    progress['further_worsening'] = True
                segment.update(gap_change_px=change, threshold_px=noise, improved=improved,
                    evidence='worsening' if worse else 'improving' if improved else 'inconclusive')
            segments[identity] = segment
            statuses.append(segment['improved'])
            evidence[identity] = {key: copy.deepcopy(value) for key, value in segment.items()
                                  if key != 'baseline'}
        self._reverse_measurement_segments = segments
        progress['per_neighbor'] = evidence
        progress['all_trigger_gaps_improved'] = bool(statuses and all(statuses))

    def observe(self, *args, **kwargs):
        progress = self.retreat_progress
        historical_baseline = None if progress is None else progress.get('baseline')
        action = kwargs.get('action', args[2] if len(args) > 2 else None)
        command = np.asarray(action[:3], float)*.05
        size = float(np.linalg.norm(command))
        pending = self._retreat_pending
        matched = bool(pending is not None and size > 1e-8
            and self._angle(command/size, pending['direction']) < 1e-5
            and size <= pending['command_cap_m']+1e-9)
        charged = matched or (size <= 1e-8 and self._retreat_committed is not None
                              and self._charge_uncertain_motion)
        before_available = self.state is not None
        before_state = None if self.state is None else self.state.copy()
        # The inherited observer alone commits actions and all physical debt.
        # Its old baseline subtraction is disabled only during this synchronous
        # call; no assessment runs here. Finally restores history even on error.
        # First-action creation is already same-epoch and is recomputed below.
        if progress is not None:
            progress['baseline'] = None
        try:
            super().observe(*args, **kwargs)
        finally:
            if progress is not None:
                progress['baseline'] = historical_baseline
        progress = self.retreat_progress
        if progress is not None and charged and before_available:
            self._update_reverse_measurements(progress.get('baseline'), self._usable_snapshot())
        elif (progress is not None and before_state is not None
                and not np.allclose(before_state, self.state, atol=1e-12, rtol=0.)):
            # A later matched reverse must not import an intervening Direct or
            # normal movement into its gap trend without matching travel. This
            # ends only the temporal segment, never the physical risk ledger.
            current = self._usable_snapshot()
            self._reverse_measurement_segments = {
                identity: {**self._segment(identity, current, progress['measured_travel_m']),
                           'boundary_reason': 'uncharged_actual_motion_interrupted_reverse_segment'}
                for identity in progress['episode_ids']}
            progress['all_trigger_gaps_improved'] = False
        elif progress is not None and self._usable_snapshot() is None:
            progress['all_trigger_gaps_improved'] = False
        return self.summary()

    def summary(self):
        result = super().summary()
        if not hasattr(self, 'measurement_epoch'):
            return result
        segments = copy.deepcopy(self._reverse_measurement_segments)
        for segment in segments.values():
            baseline = segment.pop('baseline', None)
            segment['baseline_gap_px'] = None if baseline is None else baseline['gap']
        result['measurement_representation'] = {
            'epoch': self.measurement_epoch,
            'per_neighbor_epoch': dict(self.neighbor_measurement_epochs),
            'available': self._measurement_available,
            'last_revision': copy.deepcopy(self.last_measurement_revision),
            'reverse_segments': segments,
            'cross_representation_motion_evidence_allowed': False,
            'historical_worsening_and_physical_debt_preserved': True,
            'current_expanded_extent_is_retained': True,
            'clearance_certified': False}
        return result
