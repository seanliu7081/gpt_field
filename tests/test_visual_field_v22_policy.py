"""Candidate offers use the actual constrained route and current robot state."""
import copy
import pickle

import numpy as np
import pytest

from gpt_field.visual_field_policy_v21 import VisualFieldPolicyV21
from gpt_field.visual_field_policy_v22 import VisualFieldPolicyV22
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v22 import archive_v21_failure, snapshot, assert_unchanged
from test_visual_field_v4 import Q


def archived_policy():
    guard, monitor, field, state, pred, feedback = archive_v21_failure()
    policy = object.__new__(VisualFieldPolicyV22)
    policy.local, policy.translation_guard, policy.monitor = field, guard, monitor
    policy.prediction = pred
    policy.scene_guard = SceneCoverageGuardV17()
    policy.contact_alignment = None
    policy._recovery_joint_position = Q
    policy.step_index = 228
    policy.recovery_options = {'candidates': [], 'rejected_candidates': [],
                              'preview_scope': 'actual archived V21 observation'}
    # The archive supplies a current complete materialization, not object truth.
    material = copy.deepcopy(field._transit_materialization)
    policy._materialize = lambda state8: copy.deepcopy(material)
    policy._scene_assess = field.scene_assessor
    policy.scene_materialization = {'must_remain': 'live policy snapshot'}
    return policy, state


def test_recorded_offer_previews_same_route_that_real_executor_tracks_without_mutation():
    policy, state = archived_policy()
    field, guard = policy.local, policy.translation_guard
    before = snapshot(field, guard)
    original_prediction = copy.deepcopy(policy.prediction)
    for _ in range(2):
        policy._screen_recovery_options(state)
        assert len(policy.recovery_options['candidates']) == 1
        assert_unchanged(before, field, guard)
        assert policy.prediction == original_prediction
        assert policy.scene_materialization == {'must_remain': 'live policy snapshot'}
    offer = policy.recovery_options['candidates'][0]
    selected = field.select(state, Q)
    assert selected['accepted'], selected['status']
    assert np.allclose(np.asarray(selected['action'][:3])*.05,
                       offer['preview_first_command_delta_m'], atol=1e-12)
    assert selected['diagnostics']['heading_recovery']['route_id'] == \
        offer['actual_executor_preview']['diagnostics']['heading_recovery']['route_id']
    assert selected['diagnostics']['heading_recovery']['length_m'] == pytest.approx(.008)
    assert selected['diagnostics']['scene_coverage_admission']['accepted']
    assert guard.observation_travel_m == 0 and guard.experiments == []


def test_missing_or_invalid_current_joints_never_offer_executable_recovery():
    policy, state = archived_policy()
    policy._recovery_joint_position = None
    policy._screen_recovery_options(state)
    assert policy.recovery_options['candidates'] == []
    policy._recovery_joint_position = [0.]*7
    policy._screen_recovery_options(state)
    assert policy.recovery_options['candidates'] == []
    assert policy.recovery_options['rejected_candidates'][-1]['reason'] == 'measured_joint_limit'
    assert policy.local.pending is None


def test_prospective_scene_assessment_does_not_borrow_direct_branch_or_old_field():
    policy, state = archived_policy()
    policy.prediction.update(controller='direct', phase='recover', field_intent=None,
                             direct_actions=[[0.]*7 for _ in range(5)])
    before = pickle.dumps(policy.prediction)
    policy._screen_recovery_options(state)
    offer = policy.recovery_options['candidates'][0]
    scene = offer['actual_executor_preview']['diagnostics']['scene_coverage_admission']
    assert scene['accepted'] and len(scene['diagnostics']['checked_metric_ids']) == 8
    assert scene['diagnostics']['coverage_issues'] == []
    assert pickle.dumps(policy.prediction) == before


def test_normal_field_commands_feedback_and_call_count_match_v21():
    previous, current = VisualFieldPolicyV21(Client(prediction())), VisualFieldPolicyV22(Client(prediction()))
    old_obs, new_obs = observation(), observation()
    for _ in range(12):
        old_obs, old_decision, old_row = fixed_rgb_transition(previous, old_obs)
        new_obs, new_decision, new_row = fixed_rgb_transition(current, new_obs)
        assert new_decision.action == old_decision.action
        assert new_decision.diagnostics['selection']['accepted']
        assert new_row['task_progress'] == old_row['task_progress']
    assert current.client.calls == previous.client.calls == 1
    assert current.local.heading_recovery is None
    assert current.local.pending is current.pending is None
    assert current._recovery_joint_position == new_obs.joint_position
    assert new_decision.diagnostics['policy_version'] == 'visual-field-policy-v22'
