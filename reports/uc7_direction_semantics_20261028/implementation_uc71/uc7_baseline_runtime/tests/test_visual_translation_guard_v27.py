"""Finite conditional transit must not become a reusable or global exemption."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v26 import ProjectedTranslationGuardV26
from gpt_field.visual_translation_guard_v27 import ProjectedTranslationGuardV27, _NeighborResponse
from test_visual_field_v27 import upgraded
from test_visual_translation_guard_v21 import observe, pixels


def flow_snapshot(i, *, epoch=1, gain=300., stationary=False):
    return {'state': np.array([i*.001, 0., 0.]),
        'tool': np.array([0. if stationary else i*.001*gain, 0.]),
        'neighbors': {'n': {'position': np.zeros(2), 'measurement_epoch': epoch}}}


def test_directional_response_uses_held_out_actual_motion_and_refuses_new_heading():
    witness = _NeighborResponse(1)
    for i in range(4): witness.observe(flow_snapshot(i), 'n')
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([.003, 0, 0]))['validated']
    for i in range(4, 9): witness.observe(flow_snapshot(i), 'n')
    assert witness.evidence(np.array([.001, 0, 0]), np.array([.008, 0, 0]))['validated']
    assert not witness.evidence(np.array([0, .001, 0]), np.array([.008, 0, 0]))['validated']
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([.020, 0, 0]))['validated']
    old = pickle.dumps(witness)
    witness.observe(flow_snapshot(9, epoch=2), 'n')
    assert pickle.dumps(witness) == old


def test_stationary_texture_and_sustained_opposite_response_do_not_validate():
    witness = _NeighborResponse(1)
    for i in range(12): witness.observe(flow_snapshot(i, stationary=True), 'n')
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([.011, 0, 0]))['validated']
    witness = _NeighborResponse(1)
    for i in range(9): witness.observe(flow_snapshot(i), 'n')
    for i in range(9, 14): witness.observe(flow_snapshot(i, gain=-300.), 'n')
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([.013, 0, 0]))['validated']


def test_one_new_axis_training_sample_cannot_borrow_other_heading_validation():
    witness = _NeighborResponse(1)
    for i in range(9): witness.observe(flow_snapshot(i), 'n')
    snapshot = flow_snapshot(8)
    snapshot['state'][1] = .001
    snapshot['tool'][1] = .3
    witness.observe(snapshot, 'n')
    evidence = witness.evidence(np.array([0, .001, 0]), snapshot['state'])
    assert evidence['fit']['rank'] == 2
    assert evidence['current_supported_heading']
    assert evidence['held_out_count'] == 0 and not evidence['validated']
    assert evidence['other_heading_held_out_count'] >= 2
    assert not witness.evidence(np.array([-.001, 0, 0]), snapshot['state'])['validated']


def test_shared_training_prior_supplies_no_historical_validation_and_can_be_disproved():
    source = _NeighborResponse(1)
    for i in range(9): source.observe(flow_snapshot(i), 'n')
    witness = _NeighborResponse(1)
    assert witness.seed_training_prior(source.model, 'pre_action_prior')
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([0.,0,0]))['validated']
    assert len(witness.errors) == 0
    assert not witness.seed_training_prior(source.model, 'repeated_prior')
    for i in range(4): witness.observe(flow_snapshot(i, stationary=True), 'n')
    assert not witness.evidence(np.array([.001, 0, 0]), np.array([.003,0,0]))['validated']
    assert witness.errors and sum(r['squared_error'] for r in witness.errors) > 0
    good = _NeighborResponse(1);good.seed_training_prior(source.model, 'pre_action_prior')
    for i in range(5): good.observe(flow_snapshot(i), 'n')
    assert good.evidence(np.array([.001, 0, 0]), np.array([.004,0,0]))['validated']


def setup(monkeypatch):
    field, old, monitor, state = upgraded()
    old.__class__ = ProjectedTranslationGuardV27
    for key, value in ProjectedTranslationGuardV27().__dict__.items():
        if key not in old.__dict__: old.__dict__[key] = value
    old.context['translation_mode'] = 'transit'
    old.worsening = {'neighbor'}
    old.boundary_evidence = {'neighbor': {'comparable': True, 'classification': 'inconclusive',
        'measurement_epoch': old.neighbor_measurement_epochs.get('neighbor', 0)}}
    old.neighbor_measurement_epochs['neighbor'] = old.boundary_evidence['neighbor']['measurement_epoch']
    old._neighbor_responses['neighbor'] = SimpleNamespace(evidence=lambda *a: {'validated': True})
    monitor.refs['external']['moving_kind'] = 'tool'
    monkeypatch.setattr(monitor, '_external_reliable', lambda: True)
    monkeypatch.setattr(monitor, 'cumulative_response_evidence', lambda: {'contradicted': False})
    monitor.status = 'ready'
    response = SimpleNamespace(predict=lambda *a: {'supported': True, 'responsive': True, 'mean_delta_m':[-.001,0,0]},
        command_cap_for_measured_limit=lambda d, remaining, requested, s: {'command_cap_m': min(remaining,requested)})
    def refused(*a, **kw):
        return {'accepted': False, 'need_refresh': True, 'cap_m': 0.,
            'reason': 'translation_reverse_requires_current_empty_tool_full_sweep', 'diagnostics': {}}
    monkeypatch.setattr(ProjectedTranslationGuardV26, 'assess', refused)
    command = np.array([-.003, 0., 0.])
    records = field.nominal_regions_for_segment(state[:3], state[:3]+command)
    kwargs = dict(monitor=monitor, nominal_clear=True, nominal_away=False,
        nominal_regions=records, translation_response=response)
    return old, field, state, command, kwargs


def test_conditional_acceptance_keeps_warning_and_has_no_selection_time_travel(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    history = copy.deepcopy(guard.worsening_motion)
    result = guard.assess(delta, **kwargs)
    assert result['accepted'], result['reason']
    assert result['cap_m'] <= .003
    assert guard.worsening == {'neighbor'} and guard.worsening_motion == history
    assert guard.observation_travel_m == 0 and guard.experiments == []
    assert guard._motion_classification_pending['uncertain']
    assert guard._experiment_pending is not None
    assert guard._retreat_pending is None


@pytest.mark.parametrize('fault', ['epoch','state','endpoint','role','contact','payload','unknown',
    'decreasing','interior','fingerprint','object_fingerprint','scene','boundary','closing','prior_closing','response',
    'robot_response','contradiction','shared_debt','heading_debt','reverse_worsening'])
def test_ineligible_candidate_or_evidence_cannot_use_reconciliation(monkeypatch, fault):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    c = kwargs['nominal_regions'][0]['transit_reconciliation_contract']
    proof = c['object_proofs']['neighbor']
    if fault == 'epoch': c['observation_epoch'] += 1
    elif fault == 'state': c['state8'][0] += .001
    elif fault == 'endpoint': c['end_xyz_m'][0] += .001
    elif fault == 'role': c['route_family'] = 'reverse_observed'
    elif fault == 'contact': c['target_contact'] = 'fingers'
    elif fault == 'payload': c['empty_tool'] = False
    elif fault == 'unknown': c['hard_segment_clear'] = False
    elif fault == 'decreasing': proof['gap_nondecreasing'] = False
    elif fault == 'interior': proof['minimum_segment_gap_m'] = 0.
    elif fault == 'fingerprint': c['geometry_fingerprint']['intent_uncertainty_m'] += .001
    elif fault == 'object_fingerprint':
        import hashlib,json
        proof['world_geometry']['uncertainty_m'] += .001
        proof['object_geometry_sha256'] = hashlib.sha256(json.dumps(proof['world_geometry'],
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
    elif fault == 'scene': c['full_scene_ids'] = []
    elif fault == 'boundary': guard.boundary_evidence['neighbor']['comparable'] = False
    elif fault == 'closing': guard.boundary_evidence['neighbor']['classification'] = 'closing'
    elif fault == 'prior_closing': guard.boundary_closing_ids.add('neighbor')
    elif fault == 'response': guard._neighbor_responses['neighbor'] = SimpleNamespace(evidence=lambda *a: {'validated': False})
    elif fault == 'robot_response': kwargs['translation_response'].predict = lambda *a: {'supported': False}
    elif fault == 'contradiction': kwargs['monitor'].status = 'visual_response_contradiction'
    elif fault == 'shared_debt': guard.observation_travel_m = guard.observation_limit_m
    elif fault == 'heading_debt':
        guard.experiments = [{'direction':[-1,0,0], 'support_generation':0,
            'measured_travel_m':guard.experiment_limit_m, 'acknowledged_actions':8}]
        guard._last_charged_experiment = 0
    elif fault == 'reverse_worsening': guard.retreat_progress = {'further_worsening': True}
    result = guard.assess(delta, **kwargs)
    assert not result['accepted'], fault
    assert guard.worsening == {'neighbor'}
    assert guard._motion_classification_pending is None and guard._experiment_pending is None


def test_preview_keeps_live_evidence_and_pending_physical_state(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    # Evidence objects include intentionally local stub callables, so compare
    # actual identities and accounting instead of pickling the fixture.
    original = guard.__dict__
    warning = guard.worsening.copy()
    selected = guard.preview_assess(delta, **kwargs)
    assert selected['accepted']
    assert guard.__dict__ is original and guard.worsening == warning
    assert guard.last_projected_warning_transit is None
    assert guard._experiment_pending is None and guard.observation_travel_m == 0


def test_executed_conditional_motion_and_residual_hold_spend_same_existing_debt(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    assert guard.assess(delta, **kwargs)['accepted']
    monkeypatch.undo()  # Actual observer/monitor, after isolated admission setup.
    state = observe(guard, kwargs['monitor'], state, delta, [-.0008,0,0], pixel_x=-1)
    assert guard.observation_travel_m == pytest.approx(.0008)
    assert len(guard.experiments) == 1
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(.0008)
    assert guard.retreat_progress is None
    state = observe(guard, kwargs['monitor'], state, [0,0,0], [-.0002,0,0], pixel_x=-1)
    assert guard.observation_travel_m == pytest.approx(.001)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(.001)
    images = {'external': pixels(-1), 'wrist': pixels(-1)}
    guard.configure(images, guard.context, state, target_id='neighbor', monitor=kwargs['monitor'])
    assert guard.observation_travel_m == pytest.approx(.001)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(.001)
