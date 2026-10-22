"""Intent review preserves information, without claiming an executable reverse."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_intent_watch_v36 import WristIntentWatchV36
from gpt_field.visual_wrist_goal_v35 import WristBearingGoalFieldV35


def system(point=(206., 111.5), desired=(.5, .5)):
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    track = SimpleNamespace(ident='object', position=np.array(point), scale=np.array([223., 223.]),
        reliable=True, last_residual_px=.12, last_inliers=8)
    reference = {'target_id': 'object', 'desired_target_uv': list(desired)}
    prediction = {'target_id': 'object', 'visual_references': {'external': None, 'wrist': reference}}
    monitor = SimpleNamespace(state=state.copy(), refs={'wrist': copy.deepcopy(reference)},
                              tracks={'wrist_target': track})
    learned = WristBearingGoalFieldV35()
    learned.configure(state, monitor, prediction)
    return learned, WristIntentWatchV36(), monitor, prediction, state


def public_track(track):
    return {'identity': track.ident, 'reliable': track.reliable,
        'point_currently_observed': track.reliable,
        'tracked_point_uv': (track.position / track.scale).tolist(),
        'inliers': track.last_inliers, 'initial_features': 8, 'support_radius_px': 16.}


def advance(parts, *, flow=(1., 0.), delta=(0., .0035, 0.), intent=1, accepted=True,
            action=None, watch=True):
    learned, observer, monitor, prediction, state = parts
    before = state.copy(); state[:3] += delta
    command = np.r_[np.array(delta) / .05, np.zeros(4)] if action is None else np.asarray(action)
    monitor.state = state.copy(); monitor.tracks['wrist_target'].position += flow
    learned.observe(before, command, state, monitor=monitor, prediction=prediction,
                    acknowledged=accepted)
    args = {'wrist_summary': learned.summary(), 'wrist_track_summary': public_track(monitor.tracks['wrist_target']),
        'image_size_wh': [224, 224], 'intent_id': intent, 'target_id': prediction['target_id'],
        'phase': 'approach', 'acknowledged': accepted}
    report = observer.observe(before, command, state, **args) if watch else None
    return report, (before, command, state.copy(), args)


def qualified():
    parts = system()
    reports = [advance(parts)[0] for _ in range(6)]
    assert any(r['new_event'] for r in reports)
    return parts, reports


def test_boundary_decline_requires_real_prospective_same_heading_and_sustained_observations():
    parts = system()
    reports = [advance(parts)[0] for _ in range(5)]
    assert not any(r['active'] for r in reports[:4])
    event = reports[-1]['event']
    assert event and reports[-1]['new_event'] and reports[-1]['review_required']
    evidence = event['evidence']
    assert evidence['current_direction_reliability']['count'] == 2
    assert evidence['current_direction_reliability']['error2'] < .75 * evidence['current_direction_reliability']['zero2']
    assert all(evidence['trigger_checks'].values())
    assert evidence['edge'] == 'right'
    assert evidence['continuation_prediction']['predicted_edge_margin_decrease_px'] > .5
    assert event['reason'] == 'wrist_target_visibility_declining_under_current_heading'
    for flag in ('automatic_motion', 'physical_collision_inferred', 'task_failure_or_success_inferred',
                 'reference_reset_requested', 'physical_allowance_replenished', 'permanent_motion_lock'):
        assert event[flag] is False


@pytest.mark.parametrize('case', ['interior_approach', 'inward', 'tangent', 'weak_noise_signal'])
def test_necessary_approach_or_no_visibility_loss_is_not_automatically_failure(case):
    parts = system(point=(130., 111.5) if case == 'interior_approach' else (206., 80.))
    flow = {'interior_approach': (1., 0.), 'inward': (-1., 0.),
            'tangent': (0., -1.), 'weak_noise_signal': (.1, 0.)}[case]
    for _ in range(10):
        report, _ = advance(parts, flow=flow)
        assert not report['active']
    assert parts[0].model.matrix is not None


def test_same_intent_event_latches_once_and_only_new_refresh_clears_active_not_archive():
    parts, reports = qualified()
    observer = parts[1]
    event = copy.deepcopy(observer.summary()['event'])
    assert sum(r['new_event'] for r in reports) == 1
    for _ in range(10):
        report, _ = advance(parts, delta=(0., 0., 0.), flow=(0., 0.), action=np.zeros(7), accepted=False)
        assert report['active'] and not report['new_event'] and report['event'] == event
    assert observer.acknowledge_review(1)['active']
    after = observer.acknowledge_review(2)
    assert not after['active'] and not after['new_event'] and after['event'] is None
    assert after['archive'] == [event]
    assert not observer.acknowledge_review(2)['active']


@pytest.mark.parametrize('case', ['hold', 'rejected', 'gripper', 'rotation'])
def test_unaccepted_or_nontranslation_observations_never_create_review_motion_evidence(case):
    parts = system()
    for _ in range(8):
        command = np.array([0., .07, 0., 0., 0., 0., 0.])
        if case == 'hold': command[:] = 0
        if case == 'gripper': command[6] = 1
        if case == 'rotation': command[3] = .1
        report, _ = advance(parts, action=command, accepted=case != 'rejected')
        assert not report['active']
    assert not parts[1]._history


@pytest.mark.parametrize('case', ['target', 'state', 'point', 'epoch', 'lost', 'size', 'desired'])
def test_stale_or_mismatched_current_observations_cannot_supply_a_review(case):
    parts = system()
    for _ in range(4): advance(parts)
    _, (before, command, after, args) = advance(parts, watch=False)
    if case == 'target': args['target_id'] = 'other'
    if case == 'state': args['wrist_summary']['current_snapshot']['state8'][0] += .01
    if case == 'point': args['wrist_track_summary']['tracked_point_uv'][0] -= .1
    if case == 'epoch': args['wrist_summary']['current_snapshot']['epoch'] += 1
    if case == 'lost': args['wrist_track_summary']['point_currently_observed'] = False
    if case == 'size': args['image_size_wh'] = [225, 224]
    if case == 'desired': args['wrist_summary']['current_snapshot']['desired_target_uv'] = [.4, .5]
    report = parts[1].observe(before, command, after, **args)
    assert not report['active'] and not report['current_evidence']['reference_available']
    assert not parts[1]._history


@pytest.mark.parametrize('case', ['future', 'duplicate', 'in_sample', 'wrong_scores', 'unseen_reverse'])
def test_invalid_or_opposite_heading_predictions_are_not_prospective_evidence(case):
    parts = system()
    for _ in range(4): advance(parts)
    _, (before, command, after, args) = advance(parts, watch=False)
    rows = args['wrist_summary']['held_out_records']
    if case == 'future': rows[0]['observed_transition'] = 999
    if case == 'duplicate': rows[1] = copy.deepcopy(rows[0])
    if case == 'in_sample': rows[0]['training_samples_before_prediction'] = 0
    if case == 'wrong_scores': rows[0]['zero2'] += 10
    if case == 'unseen_reverse':
        for row in rows: row['delta_m'] = (-np.array(row['delta_m'])).tolist()
    report = parts[1].observe(before, command, after, **args)
    assert not report['active']


def test_changed_reference_epoch_does_not_mix_old_trend_into_new_domain():
    parts = system()
    for _ in range(4): advance(parts)
    learned, observer, monitor, prediction, state = parts
    monitor.tracks['wrist_target'] = copy.deepcopy(monitor.tracks['wrist_target'])
    learned.configure(state, monitor, prediction)
    for _ in range(4):
        report, _ = advance(parts)
        assert not report['active']
    assert len(observer._history) == 4


def test_watch_is_detached_and_never_mutates_the_response_or_tracking_inputs():
    parts = system()
    for _ in range(4): advance(parts)
    _, (before, command, after, args) = advance(parts, watch=False)
    snapshots = pickle.dumps((before, command, after, args, parts[0].__dict__, parts[2]))
    report = parts[1].observe(before, command, after, **args)
    assert pickle.dumps((before, command, after, args, parts[0].__dict__, parts[2])) == snapshots
    internal = parts[1].summary()
    report['event']['evidence']['target_px'][0] = 999
    assert parts[1].summary() == internal


def test_actual_v35_saved_observations_trigger_before_support_loss_without_new_actions():
    path = Path(__file__).resolve().parents[1] / 'runs/visual_field_v35_zero_shot_001/trials/bowl_original_zero_hybrid_v35/trace.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    actions = {r['step']: r for r in rows if r['event'] == 'action'}
    observer = WristIntentWatchV36(); intent = 0; triggers = []
    for row in rows:
        if row['event'] != 'feedback' or row['step'] > 37: continue
        step, feedback = row['step'], row['feedback']; action = actions[step]
        if action['diagnostics'].get('refreshed'): intent += 1
        report = observer.observe(feedback['before_state'], action['action'], feedback['after_state'],
            wrist_summary=feedback['visual_goal_field']['wrist_navigation'],
            wrist_track_summary=feedback['visual']['tracking']['wrist_target'],
            image_size_wh=[224, 224], intent_id=intent, target_id=action['diagnostics']['target_id'],
            phase=action['diagnostics']['phase'], acknowledged=action['diagnostics']['selection']['accepted'])
        if report['new_event']: triggers.append((step, report))
    assert len(triggers) == 1 and triggers[0][0] == 18
    event = triggers[0][1]['event']; evidence = event['evidence']
    assert evidence['reference_available'] and evidence['inliers'] == 9
    assert evidence['current_direction_reliability']['count'] == 2
    assert evidence['point_edge_margins_px']['right'] == pytest.approx(11.17183385524617)
    assert evidence['trend']['observed_bearing_increase_px'] == pytest.approx(5.831317939258724)
    # This analysis continues consuming the saved original trace only to inspect
    # the trigger. It never claims the original later actions follow a new policy.
    assert observer.summary()['active'] and not observer.summary()['current_evidence']['reference_available']
    assert observer.summary()['archive'] == [event]
