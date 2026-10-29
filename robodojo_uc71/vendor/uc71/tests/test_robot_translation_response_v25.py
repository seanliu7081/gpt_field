"""Ordered robot-feedback response regimes; no simulator or model inference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.robot_translation_response import MeasuredTranslationResponse
from gpt_field.robot_translation_response_v25 import MeasuredTranslationResponseV25
from test_robot_translation_response import JOINTS, feed, state


def recovering(gains=(-.2, -.1, .09, .16), *, joints=None):
    model = MeasuredTranslationResponseV25()
    current = state()
    for gain in gains:
        current, _ = feed(model, current, [.002, 0, 0], [gain*.002, 0, 0], joints=joints)
    return model, current


def test_ordered_recovery_replaces_only_responsiveness_and_retains_negative_tail_envelope():
    model, current = recovering()
    before = copy.deepcopy(model.summary())
    samples = [(s['command'].copy(), s['measured'].copy()) for s in model.samples]
    result = model.predict([.002, 0, 0], current)
    assert result['reason'] == 'robot_response_recent_responsive_regime'
    assert result['supported'] and result['responsive']
    assert result['median_longitudinal_gain'] == pytest.approx(-.005)
    assert result['mean_regime_longitudinal_gain'] == pytest.approx(.125)
    assert result['mean_delta_m'] == pytest.approx([.00025, 0, 0])
    assert result['longitudinal_interval_m'][0] < -.0004
    assert result['max_measured_norm_m'] > .0004  # Earlier opposed tail is retained.
    assert result['envelope_sample_count'] == 4 and result['prior_samples_discarded_for_regime'] == 0
    for old, new in zip(samples, model.samples):
        np.testing.assert_array_equal(old[0], new['command'])
        np.testing.assert_array_equal(old[1], new['measured'])
    for _ in range(10):
        again = model.predict([.002, 0, 0], current)
        assert again == result
    assert model.summary() == before
    assert not result['guaranteed_bound'] and not result['clearance_certified']


def test_inverse_uses_all_samples_and_hold_tail_without_adding_or_amplifying_command():
    model, current = recovering(gains=(-.2, -.1))
    current, _ = feed(model, current, [0, 0, 0], [0, .0008, 0])
    for gain in (.09, .16):
        current, _ = feed(model, current, [.002, 0, 0], [gain*.002, 0, 0])
    result = model.command_cap_for_measured_limit([1, 0, 0], .0014, .002, current)
    assert result['supported'] and result['command_cap_m'] == pytest.approx(.002)
    assert result['prediction']['measurement_and_hold_drift_margin_m'] == pytest.approx(.0008)
    assert result['prediction']['max_measured_norm_m'] <= .0014
    assert result['prediction']['longitudinal_interval_m'][0] < -.0012
    smaller = model.command_cap_for_measured_limit([1, 0, 0], .0013, .0016, current)
    assert smaller['supported'] and smaller['command_cap_m'] <= .0016
    tiny = model.command_cap_for_measured_limit([1, 0, 0], .0001, .002, current)
    assert tiny['command_cap_m'] == 0 and not tiny['cold_fallback']
    assert not result['cold_fallback'] and not result['clearance_certified']


@pytest.mark.parametrize('gains', [
    (0., 0., 0., 0.), (-.2, -.1, -.1, -.1),
    (-.2, -.1, 0., .16), (-.2, -.1, .16, 0.),
    (-.2, -.1, .16, .09), (-.2, -.1, .021, .022),
])
def test_blocked_opposed_one_outlier_declining_and_subresolution_motion_remain_unresponsive(gains):
    model, current = recovering(gains)
    result = model.command_cap_for_measured_limit([1, 0, 0], .01, .002, current)
    assert result['command_cap_m'] == 0 and not result['supported'] and not result['cold_fallback']
    assert result['prediction']['responsive'] is False


def test_hold_breaks_fresh_pair_and_cannot_reuse_pre_hold_recovery():
    model, current = recovering()
    assert model.predict([.002, 0, 0], current)['supported']
    current, _ = feed(model, current, [0, 0, 0], [.00005, 0, 0])
    result = model.predict([.002, 0, 0], current)
    assert not result['supported']
    assert result['recent_response_regime']['reason'] == 'recent_response_interrupted_by_hold_or_invalid_sample'
    current, _ = feed(model, current, [.002, 0, 0], [.00032, 0, 0])
    # Legacy pooled response can eventually recover independently; no recent
    # two-sample override exists until two actual non-holds are acknowledged.
    regime = model._recent_regime(np.array([1., 0, 0]), current, None)
    assert not regime['eligible']


def test_latest_other_heading_cannot_borrow_prior_fresh_directional_response():
    model, current = recovering()
    current, _ = feed(model, current, [0, .002, 0], [0, .0004, 0])
    result = model.predict([.002, 0, 0], current)
    assert not result['supported']
    assert result['recent_response_regime']['reason'] == 'recent_response_direction_changed'


def test_same_domain_stale_query_cannot_use_recent_regime():
    model, current = recovering()
    stale = current.copy(); stale[0] -= .0002
    result = model.predict([.002, 0, 0], stale)
    assert not result['supported']
    assert result['recent_response_regime']['reason'] == 'recent_response_requires_current_acknowledged_endpoint'


@pytest.mark.parametrize('change', ['orientation', 'fingers', 'position', 'joints', 'missing_joints'])
def test_configuration_or_joint_domain_changes_never_transfer_recent_regime(change):
    model, current = recovering(joints=JOINTS)
    changed = current.copy(); joints = JOINTS.copy()
    if change == 'orientation': changed[5] += .016
    elif change == 'fingers': changed[6] -= .003
    elif change == 'position': changed[1] += .041
    elif change == 'joints': joints[2] += .121
    else: joints = None; changed[0] += .00001
    result = model.predict([.002, 0, 0], changed, joint_position=joints)
    assert not result['supported'] and 'domain_changed' in result['reason']
    assert 'recent_response_regime' not in result


def test_observation_discontinuity_and_invalid_transition_drop_recent_pair():
    model, current = recovering()
    jumped = current.copy(); jumped[1] += .001
    after, report = feed(model, jumped, [.002, 0, 0], [.00032, 0, 0])
    assert report['domain_reset_reason'] == 'robot_response_observation_discontinuity'
    assert len(model._ordered_recent) == 1 and len(model.samples) == 1
    assert not model.predict([.002, 0, 0], after)['supported']


def test_recent_recovery_does_not_expand_direction_or_scale_support():
    model, current = recovering()
    for command in ([0, .002, 0], [-.002, 0, 0], [.0031, 0, 0], [.0004, 0, 0]):
        result = model.predict(command, current)
        assert not result['supported'] and 'recent_response_regime' not in result


def test_ordinary_responsive_predictions_and_caps_are_legacy_identical():
    base = MeasuredTranslationResponse(); model = MeasuredTranslationResponseV25(); current = state()
    for gain in (.17, .21, .2, .22, .23):
        after, _ = feed(base, current, [.002, 0, 0], [.002*gain, .00001, 0])
        feed(model, current, [.002, 0, 0], [.002*gain, .00001, 0]); current = after
    assert model.predict([.002, 0, 0], current) == base.predict([.002, 0, 0], current)
    assert model.command_cap_for_measured_limit([1, 0, 0], .0005, .002, current) == base.command_cap_for_measured_limit([1, 0, 0], .0005, .002, current)


def test_delayed_response_plant_can_continue_after_two_positive_acknowledgments():
    model = MeasuredTranslationResponseV25(); current = state()
    gains = [-.22, -.16, -.08, -.03, .07, .12, .16, .19, .2]
    caps = []
    for index, gain in enumerate(gains):
        current, _ = feed(model, current, [.002, 0, 0], [gain*.002, 0, 0])
        if index >= 3:
            cap = model.command_cap_for_measured_limit([1, 0, 0], .001, .002, current)
            caps.append(cap['command_cap_m'])
    assert caps[:2] == [0., 0.]  # Not before two positive acknowledgments.
    assert all(value == pytest.approx(.002) for value in caps[2:])
    assert len(model.samples) == len(gains)


@pytest.fixture(scope='module')
def archive140():
    path = Path('runs/visual_field_v24_zero_shot_001/trials/bowl_original_zero_hybrid_v24/trace.jsonl')
    if not path.exists(): pytest.skip('Optional actual V24 robot archive unavailable')
    models = [MeasuredTranslationResponse(), MeasuredTranslationResponseV25()]
    before = action = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'reset': before = row['observation']
            if row['event'] == 'action':
                if row['step'] == 140:
                    diagnostic = row['diagnostics']['selection']['diagnostics']['translation_admission']['diagnostics']['directional_retreat']['diagnostics']
                    return models, before, diagnostic
                action = row['action']
            if row['event'] == 'executed':
                after = row['observation']
                for model in models:
                    model.observe(before['observation/state'], action, after['observation/state'],
                        before_joints=before['observation/joint_position'], after_joints=after['observation/joint_position'])
                before = after
    raise AssertionError('Actual step140 unavailable')


def test_actual_v24_step140_recovery_cap_preserves_historical_bounds_and_allowance(archive140):
    (legacy, model), before, diagnostic = archive140
    delta = diagnostic['candidate_delta_xyz_m']; remaining = diagnostic['measured_remaining_m']
    kwargs = {'state8': before['observation/state'], 'joint_position': before['observation/joint_position']}
    original = legacy.command_cap_for_measured_limit(delta, remaining, .002, **kwargs)
    revised = model.command_cap_for_measured_limit(delta, remaining, .002, **kwargs)
    assert original['command_cap_m'] == 0
    assert original['prediction']['median_longitudinal_gain'] == pytest.approx(.017547964)
    assert revised['supported'] and revised['command_cap_m'] == pytest.approx(.002)
    assert revised['measured_limit_m'] == remaining == pytest.approx(.0019067850122473366)
    prediction = revised['prediction']
    assert prediction['max_measured_norm_m'] == pytest.approx(.001393798208980382)
    assert prediction['measurement_and_hold_drift_margin_m'] == pytest.approx(.0008092365955809844)
    assert [x['gain'] for x in prediction['recent_response_regime']['samples']] == pytest.approx([.08981469643038328, .1585983009036529])
    assert prediction['longitudinal_interval_m'][0] < 0
    assert len(model.samples) == len(legacy.samples)
    for new, old in zip(model.samples, legacy.samples):
        for key in ('command', 'measured'): np.testing.assert_array_equal(new[key], old[key])
    assert not revised['clearance_certified'] and not revised['guaranteed_bound']


def test_aggregate_pair_can_resolve_motion_even_if_first_sample_is_below_single_floor():
    model, current = recovering((-.2, -.1, .021, .045))
    prediction = model.predict([.002, 0, 0], current)
    regime = prediction['recent_response_regime']
    assert prediction['supported'] and regime['eligible']
    assert regime['samples'][0]['longitudinal_motion_m'] == pytest.approx(.000042)
    assert regime['aggregate_projected_motion_m'] == pytest.approx(.000132)
    assert regime['minimum_aggregate_projected_displacement_m'] == pytest.approx(.0001)
    assert prediction['longitudinal_interval_m'][0] < -.0004


def test_aggregate_uses_each_transition_minimum_of_own_and_query_projection():
    model, current = recovering((-.2, -.1))
    angle = np.deg2rad(14.)
    for gain, sign in ((.021, 1), (.035, -1)):
        direction = np.array([np.cos(angle), sign*np.sin(angle), 0.])
        measured = gain*.002*direction
        current, _ = feed(model, current, .002*direction, measured)
    direction = np.array([1., 0., 0.])
    prediction = model.predict(direction*.002, current)
    regime = prediction['recent_response_regime']
    expected = sum(min(r['longitudinal_motion_m'], r['query_direction_motion_m']) for r in regime['samples'])
    assert regime['aggregate_projected_motion_m'] == pytest.approx(expected)
    assert expected == pytest.approx(.000112*np.cos(angle))
    assert regime['eligible'] and prediction['supported']


def test_actual_v25_diagnostic001_pair_qualifies_but_historical_bound_is_never_removed():
    path = Path('runs/visual_field_v25_design_001/controller_diagnostic_001/trace.jsonl')
    if not path.exists(): pytest.skip('Optional actual V25 diagnostic archive unavailable')
    model = MeasuredTranslationResponseV25(); before = action = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'reset': before = row['observation']
            elif row['event'] == 'action': action = row['action']
            elif row['event'] == 'executed':
                after = row['observation']
                model.observe(before['observation/state'], action, after['observation/state'],
                    before_joints=before['observation/joint_position'], after_joints=after['observation/joint_position'])
                before = after
            elif row['event'] == 'terminal_selection_not_executed':
                diagnostic = row['diagnostics']['selection']['diagnostics']['translation_admission']['diagnostics']['directional_retreat']['diagnostics']
                break
    delta = np.asarray(diagnostic['candidate_delta_xyz_m'])
    kwargs = {'state8': before['observation/state'], 'joint_position': before['observation/joint_position']}
    checkpoint = model.command_cap_for_measured_limit(delta, diagnostic['measured_remaining_m'], .002, **kwargs)
    assert checkpoint['prediction']['supported'] and checkpoint['prediction']['recent_response_regime']['eligible']
    regime = checkpoint['prediction']['recent_response_regime']
    assert regime['samples'][0]['longitudinal_motion_m'] == pytest.approx(.00004024752789689585)
    assert regime['samples'][1]['longitudinal_motion_m'] == pytest.approx(.00008319912017460509)
    assert regime['aggregate_projected_motion_m'] > .0001
    assert checkpoint['command_cap_m'] == 0  # Historic hold tail still exceeds checkpoint.
    assert checkpoint['reason'] == 'measured_limit_below_empirical_command_support'
    assert checkpoint['prediction']['measurement_and_hold_drift_margin_m'] == pytest.approx(.001868677545637717)
    route_remaining = .008-.002436932188084978
    route = model.command_cap_for_measured_limit(delta, route_remaining, .002, **kwargs)
    assert route['supported'] and route['command_cap_m'] == pytest.approx(.002)
    assert route['prediction']['max_measured_norm_m'] == pytest.approx(.002250796622952562)
    assert route['prediction']['max_measured_norm_m'] < route_remaining
    assert not route['clearance_certified'] and not route['guaranteed_bound']
