"""Bounded Field transit when projected extent evidence is inconclusive.

Projected warnings remain historical evidence. A candidate-specific model
separation and a prospectively checked local image response can support a
finite experiment; neither is a certificate of physical obstacle clearance.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
import numpy as np

from .visual_field import SupportedLinearModel
from .visual_translation_guard_v26 import ProjectedTranslationGuardV26
from .visual_boundary_evidence_v27 import capture_boundary_evidence, compare_gap_evidence
from .visual_route_exposure_v27 import RouteExposureV27


class _NeighborResponse:
    """Causal, heading-supported relative point motion for one representation."""

    def __init__(self, epoch):
        self.epoch = epoch
        self.origin = None
        self.last_sample_state = None
        self.model = SupportedLinearModel(residual_limit=1.5, min_samples=3,
            min_rank=1, max_samples=24, minimum_excitation=.001)
        self.errors = deque(maxlen=8)
        self.training_prior = None

    def seed_training_prior(self, model, source):
        if self.model.samples or self.errors or model.matrix is None:
            return False
        self.model = copy.deepcopy(model)
        self.training_prior = {'source': source, 'sample_count': len(model.samples),
            'historical_validation_credit': False,
            'per_object_fresh_held_out_validation_required': True}
        return True

    def observe(self, snapshot, identity):
        row = snapshot['neighbors'][identity]
        if row.get('response_representation', row['measurement_epoch']) != self.epoch:
            return
        current = (snapshot['state'].copy(), snapshot['tool']-row['position'])
        if self.origin is None:
            self.origin = current
            return
        delta = current[0]-self.origin[0]
        if np.linalg.norm(delta) < .001-1e-12:
            return
        flow = current[1]-self.origin[1]
        predicted = self.model.predict(delta)
        if predicted is not None:
            # Evaluate against the old fit, before this sample can train it.
            self.errors.append({'delta': delta.copy(),
                'squared_error': float(np.sum((flow-predicted)**2)),
                'squared_zero': float(np.sum(flow**2))})
        self.model.add(delta, flow)
        self.last_sample_state = current[0].copy()
        self.origin = current

    def evidence(self, delta, state):
        delta = np.asarray(delta, float)
        predicted = self.model.predict(delta)
        norm = float(np.linalg.norm(delta))
        # A new fitted axis is training support, not held-out validation.
        # Use the same 15% orthogonal-direction tolerance as the predictor's
        # supported subspace, and never transfer evidence to a reversal.
        matching = [row for row in self.errors if norm > 1e-12 and
            float(np.dot(row['delta'], delta)) > 0. and
            np.linalg.norm(row['delta']-delta*(np.dot(row['delta'], delta)/(norm*norm)))
            <= .15*np.linalg.norm(row['delta'])+1e-12]
        error = sum(row['squared_error'] for row in matching)
        zero = sum(row['squared_zero'] for row in matching)
        fresh = (self.last_sample_state is not None and
            np.linalg.norm(np.asarray(state)-self.last_sample_state) <= .0015+1e-12)
        validated = bool(len(matching) >= 2 and zero > .25 and error < .75*zero
                         and fresh and predicted is not None)
        return {'validated': validated, 'representation_epoch': self.epoch,
            'prediction_px': None if predicted is None else predicted.tolist(),
            'fit': self.model.summary(), 'held_out_count': len(matching),
            'other_heading_held_out_count': len(self.errors)-len(matching),
            'held_out_prediction_sse': float(error), 'held_out_zero_sse': float(zero),
            'training_prior': copy.deepcopy(self.training_prior),
            'current_supported_heading': predicted is not None,
            'recent_measured_sample': bool(fresh), 'clearance_certified': False}


class ProjectedTranslationGuardV27(ProjectedTranslationGuardV26):
    version = 'candidate-bound-projected-metric-reconciliation-v27'
    reconciliation_command_limit_m = .003

    def __init__(self):
        super().__init__()
        self.egress_evidence = RouteExposureV27()
        self._boundary_origins = {}
        self._neighbor_responses = {}
        self.boundary_evidence = {}
        self.boundary_closing_ids = set()
        self.boundary_warning_history = []
        self.last_projected_warning_transit = None

    def _snapshot(self):
        snapshot = super()._snapshot()
        if snapshot is None:
            return None
        # Store sensitivity snapshots alongside the actual measured state.
        # Source corners belong to the same immutable tracking coordinate frame.
        tool_evidence = capture_boundary_evidence(
            self.tool_track, self.tool_source, representation_key='current-tool')
        for identity, row in snapshot['neighbors'].items():
            neighbor = next(n for n in self.neighbors if n['id'] == identity)
            key = str(row['measurement_epoch'])
            row['response_representation'] = (id(self.tool_track), id(neighbor['track']))
            row['tool_boundary'] = {**tool_evidence, 'representation_key': 'tool:'+key}
            row['neighbor_boundary'] = capture_boundary_evidence(
                neighbor['track'], neighbor['source'], representation_key=identity+':'+key)
        return snapshot

    def configure(self, *args, **kwargs):
        super().configure(*args, **kwargs)
        self._update_boundary_evidence(self._usable_snapshot(), configure=True)
        return self.summary()

    def _update_boundary_evidence(self, current, *, configure=False):
        if current is None:
            self.boundary_evidence = {}
            self._neighbor_responses = {}
            self._boundary_origins = {}
            return
        evidence = {}
        for identity, row in current['neighbors'].items():
            epoch = row['measurement_epoch']
            first = self._boundary_origins.get(identity)
            if first is None or first['measurement_epoch'] != epoch:
                first = copy.deepcopy(row)
                self._boundary_origins[identity] = first
            response_key = row['response_representation']
            witness = self._neighbor_responses.get(identity)
            if witness is None or witness.epoch != response_key:
                self._neighbor_responses[identity] = _NeighborResponse(response_key)
            witness = self._neighbor_responses[identity]
            if configure:
                # A declaration never supplies a motion sample. A box-only
                # revision does not discard a still-identical point model.
                witness.origin = (current['state'].copy(), current['tool']-row['position'])
            else:
                witness.observe(current, identity)
            value = compare_gap_evidence(first['tool_boundary'], row['tool_boundary'],
                first['neighbor_boundary'], row['neighbor_boundary'])
            value['measurement_epoch'] = epoch
            evidence[identity] = value
            # A declaration or reseed is never actual boundary movement.
            if not configure and value.get('comparable'):
                if value.get('classification') == 'closing':
                    self.boundary_closing_ids.add(identity)
                elif value.get('classification') == 'opening' and identity in self.improving:
                    self.boundary_closing_ids.discard(identity)
        self.boundary_evidence = evidence

    def observe(self, *args, **kwargs):
        previous = set(self.worsening)
        # Import only pre-action TRAINING from the already checked current
        # candidate. Its shared-anchor response is a prior, not evidence about
        # this particular object. All held-out rows below still come from new
        # actual per-object observations. Do this before the inherited observer
        # can train on the transition being acknowledged now.
        selection = self.reference_selection or {}
        replay = getattr(self, '_v18_replay_applied', None)
        if (replay is not None and selection.get('track_identity') == id(self.tool_track)
                and selection.get('evidence_sha256') == replay[1]
                and self._response_track is self.tool_track):
            for witness in self._neighbor_responses.values():
                witness.seed_training_prior(self.cumulative_response,
                    'checked_current_candidate_pre_action_shared_anchor_training')
        super().observe(*args, **kwargs)
        self._update_boundary_evidence(self._usable_snapshot())
        for identity in sorted(self.worsening-previous):
            self.boundary_warning_history.append({'id': identity,
                'observation_epoch': self.sweep_observation_epoch,
                'evidence': copy.deepcopy(self.boundary_evidence.get(identity)),
                'original_risk_window': copy.deepcopy(self.risk_windows.get(identity)),
                'motion_debt_refunded': False})
        return self.summary()

    def _warned_object_gap_allowed(self, identity, proof, values, contract):
        """Default obstacle semantics; later opt-ins may distinguish attraction."""
        return (proof['gap_nondecreasing'] is True
            and values[1] >= values[0]-1e-10 and values[2] >= values[0]-1e-10)

    def _projected_boundary_allowed(self, identity, evidence, contract):
        """A representation revision cannot erase an observed closing latch."""
        return (evidence.get('comparable')
            and evidence.get('measurement_epoch') == self.neighbor_measurement_epochs.get(identity)
            and evidence.get('classification') in ('inconclusive', 'opening')
            and identity not in self.boundary_closing_ids)

    def _transit_contract(self, records, command, rotation_rad):
        contracts = [row.get('transit_reconciliation_contract') for row in records]
        if not contracts or any(not isinstance(c, dict) for c in contracts):
            return None
        c = contracts[0]
        try:
            ids = c['scene_ids']
            geometry = {row['id']: row['world_geometry'] for row in c['geometry_fingerprint']['objects']}
            valid = (c['version'] == 'v27-current-normal-transit-model-margin'
                and c['eligible'] is True and c['observation_epoch'] == self.sweep_observation_epoch
                and np.array_equal(c['state8'], self.state)
                and np.array_equal(c['start_xyz_m'], self.state[:3])
                and np.allclose(c['end_xyz_m'], self.state[:3]+command, atol=1e-10, rtol=0.)
                and c['field_controller'] is True and c['route_family'] == 'normal'
                and c['empty_tool'] is True and c['target_contact'] == 'none'
                and c['contact_or_visual_goal_dependency'] is False
                and c['full_scene_sweep_accepted'] is True and c['hard_segment_clear'] is True
                and c['outside_target_contact_band'] is True and c['target_id'] == self.target_id
                and c['remaining_actions'] > 0 and np.isfinite(c['remaining_measured_travel_m'])
                and c['remaining_measured_travel_m'] > 0 and len(ids) == len(set(ids))
                and set(ids) == {row['id'] for row in records} == set(c['object_proofs'])
                and len(records) == len(ids) == len(c['geometry_fingerprint']['objects'])
                and set(geometry) == set(ids)
                and ids == c['full_scene_ids']
                and c['full_finger_range'] is True and c['orientation_margin_rad'] >= max(.04, abs(rotation_rad))
                and c['geometry_sha256'] == hashlib.sha256(json.dumps(
                    c['geometry_fingerprint'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                and all(other == c for other in contracts)
                and all(self._proof_status(row, command, rotation_rad) == 'disjoint'
                        for row in records))
            for identity in self.worsening:
                proof = c['object_proofs'][identity]
                values = np.asarray([proof['start_gap_m'], proof['end_gap_m'],
                                     proof['minimum_segment_gap_m']], float)
                valid = valid and (np.isfinite(values).all() and np.min(values) > 0.
                    and self._warned_object_gap_allowed(identity, proof, values, c)
                    and proof['full_uncertainty_sweep_disjoint'] is True
                    and proof['configuration_part_count'] >= 6
                    and proof['world_geometry'] == geometry[identity]
                    and proof['object_geometry_sha256'] == hashlib.sha256(json.dumps(
                        proof['world_geometry'], sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        except (KeyError, TypeError, ValueError):
            valid = False
        return copy.deepcopy(c) if valid else None

    def assess(self, delta_xyz, **kwargs):
        events = self.events
        base = super().assess(delta_xyz, **kwargs)
        self.last_projected_warning_transit = None
        if (base.get('reason') != 'translation_reverse_requires_current_empty_tool_full_sweep'
                or not self.worsening):
            return base
        command = np.asarray(delta_xyz, float)
        size = float(np.linalg.norm(command))
        records = kwargs.get('nominal_regions') or []
        contract = self._transit_contract(records, command, float(kwargs.get('rotation_rad', 0.)))
        info = {'historical_warnings': sorted(self.worsening),
            'geometry_current': contract is not None, 'base_reason': base['reason'],
            'boundary_evidence': {i: copy.deepcopy(self.boundary_evidence.get(i)) for i in self.worsening},
            'physical_budget_replenished': False, 'clearance_certified': False,
            'per_object_response': {}}

        def finish(reason, *, cap=0., experiment=None):
            accepted = experiment is not None
            info.update(accepted=accepted, reason=reason, cap_m=cap)
            self.last_projected_warning_transit = copy.deepcopy(info)
            if accepted:
                self.events = events
                self._motion_classification_pending = {'uncertain': True,
                    'direction': (command/size).tolist(), 'command_cap_m': cap}
                self._experiment_pending = {**experiment, 'command_cap_m': cap}
                self._retreat_pending = None
            else:
                self._motion_classification_pending = self._experiment_pending = None
            self._recovery_reason = reason
            self._v12_last_result = {'accepted': accepted, 'cap_m': cap,
                'need_refresh': not accepted, 'reason': reason}
            result = {**self._v12_last_result, 'diagnostics': self.summary()}
            result['diagnostics']['checked_proposal'] = copy.deepcopy(
                base.get('diagnostics', {}).get('checked_proposal'))
            return result

        if (contract is None or size < 1e-12 or kwargs.get('nominal_clear') is not True
                or self.context.get('translation_mode') != 'transit'):
            return finish('translation_projected_conflict_requires_current_transit_geometry')
        if self.retreat_progress is not None and self.retreat_progress.get('further_worsening'):
            return finish('translation_projected_conflict_prior_reverse_contradiction')
        monitor = kwargs.get('monitor')
        if (monitor is None or (monitor.refs.get('external') or {}).get('moving_kind') != 'tool'
                or not monitor._external_reliable()
                or monitor.cumulative_response_evidence().get('contradicted')
                or monitor.status in ('visual_response_contradiction', 'target_anchor_shifted')):
            return finish('translation_projected_conflict_reference_contradicted')
        for identity in self.worsening:
            evidence = self.boundary_evidence.get(identity) or {}
            if not self._projected_boundary_allowed(identity, evidence, contract):
                return finish('translation_projected_conflict_boundary_unresolved_or_closing')
        response = kwargs.get('translation_response')
        predicted = None if response is None else response.predict(command, self.state)
        if not predicted or not predicted.get('supported') or not predicted.get('responsive'):
            return finish('translation_projected_conflict_robot_response_unavailable')
        for identity in self.worsening:
            witness = self._neighbor_responses.get(identity)
            if witness is None:
                return finish('translation_projected_conflict_image_response_unavailable')
            evidence = witness.evidence(np.asarray(predicted['mean_delta_m']), self.state[:3])
            info['per_object_response'][identity] = evidence
            if not evidence['validated']:
                return finish('translation_projected_conflict_image_response_unavailable')
        experiment, error = self._experiment_for(command/size, nominal_away=False)
        if error:
            return finish(error)
        used = 0. if experiment['index'] is None else self.experiments[experiment['index']]['measured_travel_m']
        remaining = min(max(0., self.observation_limit_m-self.observation_travel_m),
            max(0., self.experiment_limit_m-used), contract['remaining_measured_travel_m'])
        requested = min(size, self.reconciliation_command_limit_m)
        conversion = response.command_cap_for_measured_limit(command/size, remaining, requested, self.state)
        cap = min(requested, float(conversion['command_cap_m']))
        info.update(measured_remaining_m=remaining, conversion=copy.deepcopy(conversion))
        if cap < self.minimum_command_m-1e-9:
            return finish('translation_projected_conflict_measured_allowance_exhausted')
        return finish('projected_ambiguous_extent_bounded_model_separating_transit',
                      cap=cap, experiment=experiment)

    def summary(self):
        result = super().summary()
        if not hasattr(self, 'boundary_evidence'):
            return result
        result['projected_warning_transit'] = {
            'last_admission': copy.deepcopy(self.last_projected_warning_transit),
            'boundary_evidence': copy.deepcopy(self.boundary_evidence),
            'resolved_closing_ids': sorted(self.boundary_closing_ids),
            'warning_history': copy.deepcopy(self.boundary_warning_history),
            'existing_shared_limit_m': self.observation_limit_m,
            'existing_direction_limit_m': self.experiment_limit_m,
            'command_limit_m': self.reconciliation_command_limit_m,
            'physical_budget_replenished': False, 'clearance_certified': False}
        return result
