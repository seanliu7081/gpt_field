"""Finite conditional exposure is bounded independently of 2D gap credits."""
import copy

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v25 import ProjectedTranslationGuardV25
from test_visual_field_v25 import promote
from test_visual_field_v22 import configure_reverse
from test_visual_translation_guard_v21 import triggered, pixels
from test_visual_field_v4 import Q


def configured():
    guard, monitor, field, state, pixel = triggered()
    guard.__class__ = ProjectedTranslationGuardV25
    for key, value in ProjectedTranslationGuardV25().__dict__.items():
        if key not in guard.__dict__:
            guard.__dict__[key] = value
    promote(field)
    configure_reverse(field, guard, state, np.array([-1., 0., 0.]))
    assert field.bind_recovery_route('reverse_observed', state)['accepted']
    return guard, monitor, field, state, pixel


def test_finite_reverse_still_stops_when_static_rgb_contradicts_measured_motion():
    guard, monitor, field, state, pixel = configured()
    goal = field.goal.copy()
    for _ in range(10):
        decision = field.select(state, Q)
        if not decision['accepted']:
            assert decision['status'] == 'translation_reverse_reference_unavailable_or_contradicted'
            break
        info = decision['diagnostics']['translation_admission']['diagnostics']['finite_reverse_exposure']
        assert info is not None and info['finite_measured_limit_m'] == .008
        assert not info['clearance_certified'] and not info['physical_budget_replenished']
        after = state.copy(); after[0] -= .0005
        field.observe(state, decision['action'], after)
        images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
        monitor.observe(images, after, decision['action'])
        guard.observe(images, after, decision['action'], monitor=monitor)
        state = after
    else:
        pytest.fail('Static RGB must not license sustained reference-dependent motion')
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.worsening == {'neighbor'}
    assert len(guard.experiments) == 1
    assert guard.observation_travel_m == pytest.approx(guard.retreat_progress['measured_travel_m'])
    assert guard.information_checkpoint_m == .004
    assert field.goal == pytest.approx(goal)


def test_current_finite_contract_can_admit_after_diagnostic4mm_without_refunding_debt():
    guard, monitor, field, state, pixel = configured()
    guard.retreat_progress = {'episode_ids': ['neighbor'], 'measured_travel_m': .0045,
        'all_trigger_gaps_improved': False, 'further_worsening': False}
    guard.observation_travel_m = .0045
    command = np.array([-.002, 0., 0.])
    records = field.nominal_regions_for_segment(state[:3], state[:3]+command)
    result = guard.assess(command, monitor=monitor, nominal_clear=True, nominal_away=False,
                          nominal_regions=records, orientation_hold=True)
    assert result['accepted'], result['reason']
    assert result['cap_m'] <= .002
    assert result['diagnostics']['finite_reverse_exposure']['finite_measured_limit_m'] == .008
    assert guard.observation_travel_m == .0045
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.information_checkpoint_m == .004


@pytest.mark.parametrize('fault', ['family','goal','budget','remaining','epoch','state','geometry','payload','unknown','further_worsening'])
def test_invalid_finite_provenance_or_current_veto_never_opens_checkpoint(fault):
    guard, monitor, field, state, pixel = configured()
    command = np.array([-.002, 0., 0.])
    records = field.nominal_regions_for_segment(state[:3], state[:3]+command)
    parent = records[0]['directional_retreat_contract']
    finite = parent['finite_reverse_route']
    guard.retreat_progress = {'episode_ids': ['neighbor'], 'measured_travel_m': .0045,
        'all_trigger_gaps_improved': False, 'further_worsening': False}
    if fault == 'family': finite['route_family'] = 'normal'
    elif fault == 'goal': finite['goal_xyz_m'][1] += .1
    elif fault == 'budget': finite['max_measured_travel_m'] = .020
    elif fault == 'remaining': finite['remaining_measured_travel_m'] += .001
    elif fault == 'epoch': finite['observation_epoch'] -= 1
    elif fault == 'state': finite['state8'][0] += .001
    elif fault == 'geometry': finite['geometry_sha256'] = 'changed'
    elif fault == 'payload': parent['empty_tool'] = False
    elif fault == 'unknown': records[0]['kind'] = 'unknown_regions'
    elif fault == 'further_worsening': guard.retreat_progress['further_worsening'] = True
    debt = guard.observation_travel_m
    result = guard.assess(command, monitor=monitor, nominal_clear=True, nominal_away=False,
                          nominal_regions=records, orientation_hold=True)
    assert not result['accepted']
    assert guard.information_checkpoint_m == .004
    assert guard.observation_travel_m == debt


def test_reconfigure_and_new_route_name_cannot_refund_measured_retreat():
    guard, monitor, field, state, pixel = configured()
    guard.retreat_progress = {'episode_ids': ['neighbor'], 'measured_travel_m': .008,
        'all_trigger_gaps_improved': False, 'further_worsening': False}
    guard.observation_travel_m = .008
    for _ in range(2):
        configure_reverse(field, guard, state, np.array([-1., 0., 0.]))
        assert field.bind_recovery_route('reverse_observed', state)['accepted']
        selection = field.select(state, Q)
        assert not selection['accepted']
        assert guard.observation_travel_m == .008
        assert guard.retreat_progress['measured_travel_m'] == .008
    assert guard.information_checkpoint_m == .004


def test_finite_route_scope_does_not_change_distinct_heading_review_threshold():
    guard = ProjectedTranslationGuardV25()
    guard.context = {'translation_mode': 'separate'}
    guard.experiments = [{'direction': [0., 1., 0.], 'measured_travel_m': .005}]
    guard.information_checkpoint_m = .008
    guard._finite_reverse_original_checkpoint = .004
    selected, error = guard._experiment_for(np.array([-1., 0., 0.]), nominal_away=True)
    assert error is None and selected['index'] is None
    assert guard.information_checkpoint_m == .008
    guard.experiments[0]['measured_travel_m'] = .003
    selected, error = guard._experiment_for(np.array([-1., 0., 0.]), nominal_away=True)
    assert selected is None and error == 'translation_recovery_direction_change_requires_review'
