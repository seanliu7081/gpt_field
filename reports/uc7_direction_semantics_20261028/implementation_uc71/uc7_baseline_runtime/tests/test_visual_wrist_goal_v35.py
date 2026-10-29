"""Pure causal bearing proposals do not grant action or contact permission."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_wrist_goal_v35 import WristBearingGoalFieldV35


def fixture():
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    track = SimpleNamespace(ident='object', position=np.array([165., 110.]),
        scale=np.array([223., 223.]), reliable=True, last_residual_px=.12, last_inliers=8)
    wrist = {'update': 'seed', 'target_id': 'object', 'desired_target_uv': [.5, .5],
             'seed': {'target_anchor_uv': (track.position / track.scale).tolist(), 'target_roi_uv': None}}
    prediction = {'target_id': 'object', 'phase': 'approach',
        'visual_references': {'external': None, 'wrist': copy.deepcopy(wrist)}}
    monitor = SimpleNamespace(state=state.copy(), refs={'wrist': copy.deepcopy(wrist)},
                              tracks={'wrist_target': track})
    helper = WristBearingGoalFieldV35()
    helper.configure(state, monitor, prediction)
    return helper, monitor, prediction, state


def advance(helper, monitor, prediction, state, delta=(0., .0035, 0.), flow=(-1., 0.),
            *, action=None, acknowledged=True, joint_position=None):
    after = state.copy()
    after[:3] += delta
    monitor.state = after.copy()
    monitor.tracks['wrist_target'].position += flow
    command = np.r_[np.asarray(delta) / .05, [0., 0., 0., 0.]] if action is None else action
    helper.observe(state, command, after, monitor=monitor, prediction=prediction,
                   acknowledged=acknowledged, joint_position=joint_position)
    return after


def trained(count=9, **kwargs):
    helper, monitor, prediction, state = fixture()
    for _ in range(count):
        state = advance(helper, monitor, prediction, state, **kwargs)
    return helper, monitor, prediction, state


def test_rank_one_wrist_field_has_real_target_and_declared_objective_without_anchor():
    helper, monitor, prediction, state = trained()
    report = helper.propose(state, motion_axes='xy')
    assert report['accepted'] and report['supported_rank'] == 1
    assert all(row['goal_offset_m'][1] > 0 and row['goal_offset_m'][0] == row['goal_offset_m'][2] == 0
               for row in report['candidates'])
    assert all(.006 <= row['proposal_distance_m'] <= .03 for row in report['candidates'])
    assert all(row['predicted_error_reduction_px'] > 0 and row['trust_limit_m'] <= row['measured_directional_span_m']
               for row in report['candidates'])
    assert set(monitor.tracks) == {'wrist_target'}
    assert not helper.summary()['current_snapshot']['desired_coordinate_is_observed_anchor']
    assert not report['target_pose_estimated'] and not report['anatomical_alignment_certified']
    assert not report['execution_authorized'] and not report['geometry_checked']
    assert all(row['explicit_model_choice_required'] and row['full_route_geometry_and_current_admission_required']
               for row in report['candidates'])


def test_prediction_precedes_training_and_fit_is_not_validation():
    helper, monitor, prediction, state = trained(3)
    assert helper.model.matrix is not None and not helper.held_out
    assert not helper.propose(state)['accepted']
    state = advance(helper, monitor, prediction, state)
    assert len(helper.held_out) == 1 and not helper.propose(state)['accepted']
    state = advance(helper, monitor, prediction, state)
    assert helper.propose(state)['accepted']
    assert [row['training_samples_before_prediction'] for row in helper.held_out] == [3, 4]
    assert all(np.linalg.norm(row['prediction_px'] - row['flow_px']) < 1e-4 for row in helper.held_out)


def test_external_reference_or_role_change_preserves_independent_wrist_evidence():
    helper, monitor, prediction, state = trained()
    before = copy.deepcopy(helper.summary())
    prediction['visual_references']['external'] = {'reference_role': 'rigid_transit_proxy',
        'moving_kind': 'tool', 'moving_id': 'housing', 'target_id': 'object'}
    monitor.tracks['moving'] = SimpleNamespace(reliable=False)
    helper.configure(state, monitor, prediction)
    assert helper.summary() == before
    assert helper.propose(state)['accepted']
    prediction['visual_references']['external'] = None
    helper.configure(state, monitor, prediction)
    assert helper.summary() == before


def test_actuator_scale_uses_measured_translation_not_command_and_does_not_invent_depth():
    helper, monitor, prediction, state = fixture()
    for _ in range(35):
        state = advance(helper, monitor, prediction, state, delta=(0., .0007, 0.),
                        flow=(-.2, 0.), action=[0., .07, 0., 0., 0., 0., 0.])
    assert helper.propose(state)['accepted']
    assert helper.model.matrix[1, 0] == pytest.approx(-1 / .0035, rel=1e-4)
    assert helper.measured_training_path_m <= .0245 + 1e-12
    assert not helper.propose(state, motion_axes='z')['accepted']
    assert all(row['proposal_distance_m'] <= .0245 + 1e-12
               for row in helper.propose(state, max_goal_distance_m=.5)['candidates'])


def test_worsening_heading_does_not_authorize_unobserved_reverse():
    helper, monitor, prediction, state = trained(flow=(1., 0.))
    assert not helper.propose(state)['accepted']
    disagreement = helper.propose(state, metric_goal_xyz=state[:3] + [0., .03, 0.])['metric_image_disagreement']
    assert disagreement['opposes_visual_descent_in_observed_subspace']
    for _ in range(8):
        state = advance(helper, monitor, prediction, state, delta=(0., -.0035, 0.), flow=(-1., 0.))
    assert helper.propose(state)['accepted']
    assert all(row['goal_offset_m'][1] < 0 for row in helper.propose(state)['candidates'])


def test_proposal_and_same_observation_configure_are_pure():
    helper, monitor, prediction, state = trained()
    before = pickle.dumps(helper.__dict__)
    a, b = helper.propose(state), helper.propose(state)
    assert a == b and pickle.dumps(helper.__dict__) == before
    summary = copy.deepcopy(helper.summary())
    helper.configure(state, monitor, prediction)
    assert helper.summary() == summary
    assert not helper.summary()['physical_budget_owned']


@pytest.mark.parametrize('kind', ['target', 'desired', 'track', 'scale', 'loss', 'missing', 'payload'])
def test_reference_identity_objective_and_support_changes_cannot_transfer_evidence(kind):
    helper, monitor, prediction, state = trained()
    if kind == 'target':
        prediction['target_id'] = 'another'
    elif kind == 'desired':
        prediction['visual_references']['wrist']['desired_target_uv'] = [.6, .5]
        monitor.refs['wrist']['desired_target_uv'] = [.6, .5]
    elif kind == 'track':
        monitor.tracks['wrist_target'] = copy.deepcopy(monitor.tracks['wrist_target'])
    elif kind == 'scale':
        monitor.tracks['wrist_target'].scale[:] = 255
    elif kind == 'loss':
        monitor.tracks['wrist_target'].reliable = False
    elif kind == 'missing':
        prediction['visual_references']['wrist'] = None
    else:
        prediction['visual_references']['external'] = {'moving_kind': 'payload', 'moving_id': 'object'}
    helper.configure(state, monitor, prediction)
    assert not helper.propose(state)['accepted']
    assert len(helper.model.samples) == len(helper.held_out) == 0


@pytest.mark.parametrize('kind', ['orientation', 'finger', 'gap', 'pixel', 'stale_monitor'])
def test_robot_domains_and_unacknowledged_state_revisions_reset_evidence(kind):
    helper, monitor, prediction, state = trained()
    if kind == 'orientation':
        state[3] += .04
    elif kind == 'finger':
        state[6] += .003
    elif kind == 'gap':
        state[0] += .001
    elif kind == 'pixel':
        monitor.tracks['wrist_target'].position += [1., 0.]
    if kind != 'stale_monitor':
        monitor.state = state.copy()
    else:
        monitor.state[0] += .001
    helper.configure(state, monitor, prediction)
    assert not helper.propose(state)['accepted'] and not helper.held_out


def test_so3_equivalent_axis_angle_representation_does_not_reset_domain():
    helper, monitor, prediction, state = fixture()
    state[3] = np.pi - .001
    monitor.state = state.copy()
    helper.configure(state, monitor, prediction)
    for _ in range(8):
        state = advance(helper, monitor, prediction, state)
    epoch = helper.epoch
    # Exactly the same rotation, with an equivalent axis-angle representation.
    after = state.copy(); after[3] -= 2 * np.pi
    monitor.state = after.copy()
    helper.observe(state, np.zeros(7), after, monitor=monitor, prediction=prediction)
    assert helper.epoch == epoch and helper.propose(after)['accepted']


def test_joint_feedback_is_bound_to_each_proposal_and_domain():
    helper, monitor, prediction, state = fixture()
    joints = np.zeros(7)
    helper.configure(state, monitor, prediction, joint_position=joints)
    for _ in range(8):
        joints = joints.copy(); joints[1] += .01
        state = advance(helper, monitor, prediction, state, joint_position=joints)
    assert helper.propose(state, joint_position=joints)['accepted']
    missing = helper.propose(state)
    assert not missing['accepted']
    assert missing['reason'] == 'wrist_bearing_proposal_current_joint_feedback_required'
    assert not helper.propose(state, joint_position=np.zeros(7))['accepted']
    joints[1] += .36
    helper.configure(state, monitor, prediction, joint_position=joints)
    assert not helper.propose(state, joint_position=joints)['accepted']
    assert not helper.held_out


@pytest.mark.parametrize('kind', ['zero_hold', 'rejected', 'gripper', 'rotation'])
def test_unacknowledged_or_nontranslation_actions_never_train(kind):
    helper, monitor, prediction, state = fixture()
    for _ in range(8):
        command = np.array([0., .07, 0., 0., 0., 0., 0.])
        if kind == 'zero_hold': command[:] = 0
        if kind == 'gripper': command[6] = 1
        if kind == 'rotation': command[3] = .1
        state = advance(helper, monitor, prediction, state, action=command,
                        acknowledged=kind != 'rejected')
    assert helper.training_updates == 0 and not helper.held_out
    assert not helper.propose(state)['accepted']


@pytest.mark.parametrize('kind', ['zero_hold', 'rejected', 'gripper', 'rotation'])
def test_large_untrained_translation_cannot_transport_a_fitted_local_response(kind):
    helper, monitor, prediction, state = trained()
    assert helper.propose(state)['accepted']
    command = np.array([.1, 0., 0., 0., 0., 0., 0.])
    if kind == 'zero_hold': command[:] = 0
    if kind == 'gripper': command[6] = 1
    if kind == 'rotation': command[3] = .1
    state = advance(helper, monitor, prediction, state, delta=(.1, 0., 0.), flow=(0., 0.),
                    action=command, acknowledged=kind != 'rejected')
    assert helper.status == 'wrist_bearing_untrained_position_domain_changed'
    assert not helper.propose(state)['accepted']
    assert not helper.model.samples and not helper.held_out
    # Fresh real local response can be learned; the invalid old evidence never
    # supplies the two prospective validation records needed in the new epoch.
    for _ in range(4):
        state = advance(helper, monitor, prediction, state)
    assert not helper.propose(state)['accepted']
    state = advance(helper, monitor, prediction, state)
    assert helper.propose(state)['accepted']


def test_small_untrained_steps_cannot_walk_old_response_away_from_its_training_region():
    helper, monitor, prediction, state = trained()
    epoch, training = helper.epoch, helper.training_updates
    for _ in range(5):
        state = advance(helper, monitor, prediction, state, delta=(.001, 0., 0.),
                        flow=(0., 0.), action=np.zeros(7), acknowledged=False)
    assert helper.epoch == epoch and helper.training_updates == training
    assert helper.propose(state)['accepted']
    assert helper.summary()['distance_from_latest_training_endpoint_m'] == pytest.approx(.005)
    for _ in range(2):
        state = advance(helper, monitor, prediction, state, delta=(.001, 0., 0.),
                        flow=(0., 0.), action=np.zeros(7), acknowledged=False)
    assert helper.epoch > epoch and not helper.propose(state)['accepted']
    assert not helper.model.samples and not helper.held_out


def test_stationary_holds_preserve_real_local_response_without_training_new_evidence():
    helper, monitor, prediction, state = trained()
    original = copy.deepcopy(helper.summary())
    for _ in range(80):
        state = advance(helper, monitor, prediction, state, delta=(0., 0., 0.),
                        flow=(0., 0.), action=np.zeros(7), acknowledged=False)
    current = helper.summary()
    assert helper.propose(state)['accepted']
    assert current['training_updates'] == original['training_updates']
    assert current['held_out_records'] == original['held_out_records']
    assert current['distance_from_latest_training_endpoint_m'] == 0


def test_no_motion_does_not_create_training_and_unexplained_target_motion_invalidates():
    helper, monitor, prediction, state = trained()
    training = helper.training_updates
    for _ in range(5):
        state = advance(helper, monitor, prediction, state, delta=(0., 0., 0.), flow=(0., 0.))
    assert helper.training_updates == training
    state = advance(helper, monitor, prediction, state, delta=(0., 0., 0.),
                    flow=(2., 0.), action=np.zeros(7))
    assert not helper.held_out and not helper.propose(state)['accepted']
    assert helper.status == 'wrist_bearing_image_change_requires_new_evidence'


def test_current_live_reference_changes_reject_without_mutating_query():
    helper, monitor, prediction, state = trained()
    monitor.tracks['wrist_target'].position[0] += .1
    before = pickle.dumps(helper.__dict__)
    assert not helper.propose(state)['accepted']
    assert pickle.dumps(helper.__dict__) == before
    monitor.tracks['wrist_target'].position[0] -= .1
    prediction['visual_references']['wrist']['desired_target_uv'] = [.4, .5]
    assert not helper.propose(state)['accepted']


def test_static_bearing_cannot_qualify_and_short_trust_is_not_enlarged():
    helper, monitor, prediction, state = trained(flow=(0., 0.))
    assert not helper.propose(state)['accepted']
    helper, monitor, prediction, state = trained(5)
    report = helper.propose(state, max_goal_distance_m=.5)
    assert report['accepted']
    assert max(row['proposal_distance_m'] for row in report['candidates']) <= .0175 + 1e-12
    assert not helper.propose(state, max_goal_distance_m=.004)['accepted']


@pytest.mark.parametrize('value', [None, [float('nan'), .5], [-.1, .5], [.5, 1.1]])
def test_desired_bearing_is_explicit_finite_model_intent_never_an_implicit_centre(value):
    helper, monitor, prediction, state = fixture()
    prediction['visual_references']['wrist']['desired_target_uv'] = value
    monitor.refs['wrist']['desired_target_uv'] = value
    helper.configure(state, monitor, prediction)
    assert helper.current is None and not helper.propose(state)['accepted']


def test_true_target_at_declared_bearing_is_not_task_or_contact_success():
    helper, monitor, prediction, state = trained()
    monitor.tracks['wrist_target'].position = np.array([111.5, 111.5])
    helper.configure(state, monitor, prediction)
    summary = helper.summary()
    assert summary['current_snapshot']['error_px'] == 0
    assert not summary['task_success_certified'] and not summary['anatomical_alignment_certified']
    assert not summary['target_pose_estimated'] and not helper.propose(state)['accepted']


def test_real_runtime_wrist_bundle_identity_and_pixel_observations_qualify_a_field():
    from gpt_field.visual_monitor_v32 import RoleAwareVisualMonitorV32
    from gpt_field.visual_reference_track_v32 import ExactDescriptorPartTrackV32
    rng = np.random.default_rng(12)
    gray = rng.uniform(20, 230, (112, 112))
    gray = np.lib.stride_tricks.sliding_window_view(np.pad(gray, 1), (3, 3)).mean(axis=(-1, -2))
    image = np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)
    images = {'external': image, 'wrist': image.copy()}
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    joints = np.zeros(7)
    prediction = {'target_id': 'object', 'visual_references': {'external': None,
        'wrist': {'update': 'seed', 'target_id': 'object', 'desired_target_uv': [.5, .5],
                  'seed': {'target_anchor_uv': [.75, .5], 'target_roi_uv': None}}}}
    monitor = RoleAwareVisualMonitorV32()
    monitor.configure(images, prediction['visual_references'], state, intent_id=1)
    track = monitor.tracks['wrist_target']
    assert isinstance(track, ExactDescriptorPartTrackV32)
    assert track.ident == 'object' and not hasattr(track, 'identity')
    helper = WristBearingGoalFieldV35()
    helper.configure(state, monitor, prediction, joint_position=joints)
    assert helper.current is not None
    for index in range(1, 9):
        before = state.copy(); state[1] += .0035
        command = np.array([0., .07, 0., 0., 0., 0., 0.])
        # Render a known image translation and use actual persistent tracking.
        # Do not set tracked positions, response matrices or validation values.
        images['wrist'] = np.roll(image, -index, axis=1)
        monitor.observe(images, state, command)
        assert monitor.tracks['wrist_target'] is track and track.reliable
        helper.observe(before, command, state, monitor=monitor, prediction=prediction,
                       joint_position=joints, acknowledged=True)
    report = helper.propose(state, joint_position=joints)
    assert report['accepted'] and report['supported_rank'] == 1
    assert helper.held_out and all(row['training_samples_before_prediction'] >= 3
                                 for row in helper.held_out)
    assert all(row['goal_offset_m'][1] > 0 for row in report['candidates'])
    # Runtime retain keeps the same actual wrist object across external absence.
    prediction['visual_references']['wrist'].update(update='retain', seed=None)
    epoch, samples = helper.epoch, helper.training_updates
    monitor.configure(images, prediction['visual_references'], state, intent_id=2)
    helper.configure(state, monitor, prediction, joint_position=joints)
    assert monitor.tracks['wrist_target'] is track and helper.epoch == epoch
    assert helper.training_updates == samples and helper.propose(state, joint_position=joints)['accepted']
    # Serialized summary spelling is not an alternative authority. A wrong
    # real track ident must fail even if a fake identity alias matches intent.
    track.ident = 'another_object'
    track.identity = 'object'
    helper.configure(state, monitor, prediction, joint_position=joints)
    refused = helper.propose(state, joint_position=joints)
    assert not refused['accepted']
    assert refused['reason'] == 'wrist_bearing_current_reference_unavailable'
    assert not helper.model.samples and not helper.held_out
