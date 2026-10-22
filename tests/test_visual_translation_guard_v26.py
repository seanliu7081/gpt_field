"""Measurement changes are not motion; physical risk history is persistent."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v25 import ProjectedTranslationGuardV25
from gpt_field.visual_translation_guard_v26 import ProjectedTranslationGuardV26
from test_visual_translation_guard_v25 import configured
from test_visual_translation_guard_v21 import pixels
from test_visual_field_v4 import Q


def setup():
    guard, monitor, field, state, pixel = configured()
    guard.__class__ = ProjectedTranslationGuardV26
    for key, value in ProjectedTranslationGuardV26().__dict__.items():
        if key not in guard.__dict__:
            guard.__dict__[key] = value
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    guard.configure(images, guard.context, state, target_id='neighbor', monitor=monitor)
    return guard, monitor, field, state, pixel


def exact_context(guard):
    context = copy.deepcopy(guard.context)
    context['tool_external_roi_uv'] = (guard.tool_box/np.tile(guard.scale, 2)).tolist()
    context['nearby_obstacles'] = [{'id': n['id'],
        'external_roi_uv': (guard._transport(n)/np.tile(guard.scale, 2)).tolist()}
        for n in guard.neighbors]
    return context


def step(guard, monitor, field, state, pixel, actual=.0005):
    choice = field.select(state, Q)
    assert choice['accepted'], choice['status']
    after = state.copy(); after[0] -= actual
    field.observe(state, choice['action'], after)
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    monitor.observe(images, after, choice['action'])
    guard.observe(images, after, choice['action'], monitor=monitor)
    return after, images


def seeded_progress(guard, *, travel=.004, worse=False):
    snapshot = guard._snapshot()
    guard.retreat_progress = {'episode_ids': ['neighbor'], 'measured_travel_m': travel,
        'acknowledged_commands': 4, 'baseline': copy.deepcopy(snapshot),
        'all_trigger_gaps_improved': False, 'further_worsening': worse, 'per_neighbor': {}}
    guard._reverse_measurement_segments = {
        'neighbor': guard._segment('neighbor', snapshot, 0.)}
    return snapshot


def test_first_actual_reverse_and_unchanged_epoch_preserve_actions_and_debt():
    guard, monitor, field, state, pixel = setup()
    epoch = guard.measurement_epoch
    for n in range(1, 5):
        state, images = step(guard, monitor, field, state, pixel-n, .001)
    assert guard.measurement_epoch == epoch
    assert guard.retreat_progress['measured_travel_m'] == pytest.approx(.004)
    assert guard.observation_travel_m == pytest.approx(.004)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(.004)
    assert guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.retreat_progress['per_neighbor']['neighbor']['segment_measured_travel_m'] == pytest.approx(.004)
    assert not guard.retreat_progress['further_worsening']


@pytest.mark.parametrize('change', ['tool', 'neighbor'])
def test_same_identity_current_roi_change_starts_no_motion_segment_and_keeps_extent(change):
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    before = guard.measurement_epoch; oldgap = guard._snapshot()['neighbors']['neighbor']['gap']
    context = exact_context(guard)
    if change == 'tool': context['tool_external_roi_uv'][2] = 119/223
    else: context['nearby_obstacles'][0]['external_roi_uv'][0] = 75/223
    debt = guard.observation_travel_m; history = copy.deepcopy(guard.worsening_motion)
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    guard.configure(images, context, state, target_id='neighbor', monitor=monitor)
    current = guard._snapshot(); newgap = current['neighbors']['neighbor']['gap']
    assert newgap < oldgap-5
    assert guard.measurement_epoch == before+1
    assert guard._reverse_measurement_segments['neighbor']['baseline']['gap'] == pytest.approx(newgap)
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    p = guard.retreat_progress
    assert not p['further_worsening'] and not p['all_trigger_gaps_improved']
    assert p['per_neighbor']['neighbor']['gap_change_px'] == 0.
    assert p['per_neighbor']['neighbor']['segment_measured_travel_m'] == 0.
    assert guard.observation_travel_m == debt and guard.worsening_motion == history
    assert guard._previous_snapshot['measurement_epoch'] == guard.measurement_epoch
    assert guard._previous_snapshot['neighbors']['neighbor']['measurement_epoch'] == guard.neighbor_measurement_epochs['neighbor']


@pytest.mark.parametrize('delta_gap', [-8., 8.])
def test_new_epoch_cannot_borrow_old_travel_for_adverse_or_favorable_evidence(delta_gap):
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard, travel=.006)
    context = exact_context(guard); context['tool_external_roi_uv'][2] += .01
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    guard.configure(images, context, state, target_id='neighbor', monitor=monitor)
    current = guard._snapshot(); current['neighbors']['neighbor']['gap'] += delta_gap
    current['neighbors']['neighbor']['noise'] = .1
    guard._reverse_measurement_segments['neighbor']['baseline']['noise'] = .1
    guard.retreat_progress['measured_travel_m'] += .0002
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    assert not guard.retreat_progress['further_worsening']
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    guard.retreat_progress['measured_travel_m'] += .001
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    assert guard.retreat_progress['further_worsening'] is (delta_gap < 0)
    assert guard.retreat_progress['all_trigger_gaps_improved'] is (delta_gap > 0)


def test_genuine_same_epoch_worsening_latches_and_reseeds_cannot_clear_it():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    current = guard._snapshot(); current['neighbors']['neighbor']['gap'] -= 20.
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    assert guard.retreat_progress['further_worsening']
    history = copy.deepcopy(guard.worsening_motion)
    for _ in range(3):
        context = exact_context(guard); context['tool_external_roi_uv'][2] += .001
        guard.configure({'external': pixels(pixel), 'wrist': pixels(pixel)}, context, state,
                        target_id='neighbor', monitor=monitor)
        assert guard.retreat_progress['further_worsening']
        assert guard.retreat_progress['measured_travel_m'] == .004
        assert guard.worsening_motion == history and guard.worsening == {'neighbor'}
    choice = field.select(state, Q)
    assert not choice['accepted'] and choice['status'] == 'translation_reverse_observed_further_worsening'


def test_identical_refresh_keeps_existing_measurement_baseline_and_evidence():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    epoch = guard.measurement_epoch
    baseline = copy.deepcopy(guard._reverse_measurement_segments)
    for _ in range(3):
        guard.configure({'external': pixels(pixel), 'wrist': pixels(pixel)}, exact_context(guard),
                        state, target_id='neighbor', monitor=monitor)
    assert guard.measurement_epoch == epoch
    assert pickle.dumps(guard._reverse_measurement_segments) == pickle.dumps(baseline)


def test_missing_current_snapshot_never_credits_progress_and_reestablishes_current_baseline():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    guard._reverse_measurement_segments['neighbor']['baseline'] = None
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], None)
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    current = guard._snapshot(); current['neighbors']['neighbor']['gap'] += 20
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard._reverse_measurement_segments['neighbor']['start_measured_travel_m'] == .004


def test_malformed_refresh_preserves_historical_latch_and_physical_ledgers():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard, worse=True)
    neighbors = [n['id']for n in guard.neighbors]
    bad = exact_context(guard); bad['nearby_obstacles'][0]['external_roi_uv'] = None
    guard.configure({'external': pixels(pixel), 'wrist': pixels(pixel)}, bad, state,
                    target_id='neighbor', monitor=monitor)
    assert guard.context_invalid and not guard._measurement_available
    assert guard.retreat_progress['further_worsening']
    assert guard.retreat_progress['measured_travel_m'] == .004
    assert [n['id']for n in guard.neighbors] == neighbors
    assert not guard.retreat_progress['all_trigger_gaps_improved']


def test_exception_restores_original_historical_baseline(monkeypatch):
    guard, monitor, field, state, pixel = setup(); baseline = seeded_progress(guard)
    original = guard.retreat_progress['baseline']
    def fail(*args, **kwargs): raise RuntimeError('deliberate observer failure')
    monkeypatch.setattr(ProjectedTranslationGuardV25, 'observe', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        guard.observe({'external': pixels(pixel), 'wrist': pixels(pixel)}, state, [0.]*7, monitor=monitor)
    assert guard.retreat_progress['baseline'] is original
    assert pickle.dumps(original) == pickle.dumps(baseline)


def test_preview_preserves_epochs_physical_debt_and_live_track_identity():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    command = np.array([-.002, 0., 0.]); records = field.nominal_regions_for_segment(state[:3], state[:3]+command)
    original = pickle.dumps(guard); track = guard.tool_track
    choice = guard.preview_assess(command, monitor=monitor, nominal_clear=True,
        nominal_away=False, nominal_regions=records, prospective_translation_mode='separate')
    assert choice['accepted']
    assert pickle.dumps(guard) == original and guard.tool_track is track


def test_uncharged_motion_ends_reverse_measurement_segment_without_refunding_debt():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    historical_baseline = guard.retreat_progress['baseline']
    after = state.copy(); after[1] += .001
    image = pixels(pixel+1); images = {'external': image, 'wrist': image}
    action = [0., .02, 0., 0., 0., 0., 0.]
    # Explicitly no accepted reverse pending: this represents later accepted
    # ordinary/Direct motion, not a reverse command or residual reverse hold.
    assert guard._retreat_pending is None
    monitor.observe(images, after, action)
    guard.observe(images, after, action, monitor=monitor)
    assert guard.retreat_progress['measured_travel_m'] == .004
    assert guard.retreat_progress['baseline'] is historical_baseline
    segment = guard._reverse_measurement_segments['neighbor']
    assert segment['start_measured_travel_m'] == .004
    assert segment['boundary_reason'] == 'uncharged_actual_motion_interrupted_reverse_segment'
    assert not guard.retreat_progress['further_worsening']
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    current = guard._snapshot()
    if current is not None and segment['baseline'] is not None:
        assert segment['baseline']['gap'] == pytest.approx(current['neighbors']['neighbor']['gap'])


def test_neighbor_track_replacement_drops_only_cross_anchor_response_origin():
    guard, monitor, field, state, pixel = setup(); seeded_progress(guard)
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    origin = guard._snapshot(); guard._response_origin = origin; guard._response_age = 7
    model = guard.cumulative_response
    context = exact_context(guard); context['nearby_obstacles'][0]['external_roi_uv'][0] -= .005
    guard.configure(images, context, state, target_id='neighbor', monitor=monitor)
    assert guard._response_origin is origin and guard._response_age == 7
    # A lost neighbor's freshly seeded track defines different anchor
    # coordinates even under its same semantic name and retained whole box.
    prior_track = guard.neighbors[0]['track']; prior_track.reliable = False
    guard.configure(images, exact_context(guard), state, target_id='neighbor', monitor=monitor)
    assert guard.neighbors[0]['track'] is not prior_track
    assert guard._response_origin is None and guard._response_age == 0
    assert guard.cumulative_response is model
    assert guard.retreat_progress['measured_travel_m'] == .004


def test_saved_step24_gap_discontinuity_does_not_become_observed_approach():
    path = Path(__file__).resolve().parents[1]/'runs/visual_field_v25_zero_shot_001/trials/bowl_original_zero_hybrid_v25/trace.jsonl'
    if not path.exists(): pytest.skip('Optional actual V25 archive unavailable')
    rows = [json.loads(x)for x in path.read_text().splitlines()]
    action = next(r['diagnostics']for r in rows if r['event']=='action' and r['step']==24)
    feedback = next(r['feedback']for r in rows if r['event']=='feedback' and r['step']==24)
    prior = next(r['feedback']for r in rows if r['event']=='feedback' and r['step']==23)
    pre = action['selection']['diagnostics']['translation_admission']['diagnostics']
    start_gap = next(r['gap_px']for r in pre['neighbors']if r['id']=='cabinet')
    end_gap = next(r['gap_px']for r in feedback['translation_guard']['neighbors']if r['id']=='cabinet')
    old = feedback['translation_guard']['directional_retreat']['progress']['per_neighbor']['cabinet']
    # Pure recorded-measurement regression: preserve the ACTUAL observed
    # threshold with an equal pair variance decomposition. No tracker replay
    # or task outcome is inferred from this component-level comparison.
    noise = old['threshold_px']/(3*np.sqrt(2))
    guard = ProjectedTranslationGuardV26(); guard.measurement_epoch = 3
    guard.neighbor_measurement_epochs = {'cabinet': 3}
    guard.retreat_progress = copy.deepcopy(feedback['translation_guard']['directional_retreat']['progress'])
    guard.retreat_progress['further_worsening'] = False  # start from recorded pre24 false latch
    assert not prior['translation_guard']['directional_retreat']['progress']['further_worsening']
    initial = {'neighbors': {'cabinet': {'gap': start_gap, 'noise': noise, 'measurement_epoch': 3}}}
    current = {'neighbors': {'cabinet': {'gap': end_gap, 'noise': noise, 'measurement_epoch': 3}}}
    guard._reverse_measurement_segments = {'cabinet': guard._segment('cabinet', initial,
        prior['translation_guard']['directional_retreat']['progress']['measured_travel_m'])}
    guard._update_reverse_measurements(None, current)
    result = guard.retreat_progress['per_neighbor']['cabinet']
    assert result['gap_change_px'] == pytest.approx(-.666214293255223)
    assert result['threshold_px'] == pytest.approx(8.172456360403093)
    assert result['segment_measured_travel_m'] == pytest.approx(.0003937254866551074)
    assert not guard.retreat_progress['further_worsening']
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.retreat_progress['measured_travel_m'] == pytest.approx(.0024696588735342987)
