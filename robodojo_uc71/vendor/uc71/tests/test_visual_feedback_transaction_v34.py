"""Final diagnostic assembly preserves the actual observer/control sequence."""
import copy
import numpy as np
import pytest

from gpt_field.visual_feedback_transaction_v34 import FinalTransitionFeedbackMixinV34
from gpt_field.visual_field_policy_v33 import VisualFieldPolicyV33
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


class CountedPolicy(VisualFieldPolicyV33):
    def _observable_feedback(self):
        self.render_count = getattr(self, 'render_count', 0)+1
        return super()._observable_feedback()


class DeferredPolicy(FinalTransitionFeedbackMixinV34, CountedPolicy):
    pass


def make_policy(cls):
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    return cls(Client(value))


def test_full_feedback_is_generated_once_after_every_observer_and_block_row_is_current():
    policy = make_policy(DeferredPolicy)
    obs = observation()
    for _ in range(8):
        decision = policy.act(obs)
        # Same deterministic RGB transition used by the actual policy tests.
        from gpt_field.observation import PolicyObservation
        data = obs.to_dict()
        data['observation/state'][:3] = (
            np.asarray(obs.state[:3])+np.asarray(decision.action[:3])*.05*.8).tolist()
        after = PolicyObservation.from_dict(data)
        before_count = policy.render_count
        row = policy.observe_transition(obs, decision, after)
        assert policy.render_count-before_count == 1
        assert not policy._transition_feedback_deferred
        assert row is policy.block[-1]
        assert row['after_state'] == list(after.state)
        assert row['action'] == list(decision.action)
        assert row['translation_guard']['version'] == policy.translation_guard.summary()['version']
        assert row['visual_goal_field'] == policy.visual_goal_field.summary()
        assert row['task_progress'] == policy.task_progress.summary()
        assert row['reference_repair'] == policy.reference_candidates.summary()
        assert row['efficiency']['field_effective_actions'] == policy.field_effective_actions
        assert policy.feedback == {key: row[key] for key in policy.feedback}
        obs = after
    assert policy.client.calls == 1
    assert policy.field_effective_actions == 8
    policy.finish()


def test_equivalent_actions_and_control_ledgers_under_same_synthetic_observations():
    legacy, deferred = make_policy(CountedPolicy), make_policy(DeferredPolicy)
    before = observation()
    for _ in range(8):
        a = legacy.act(before)
        b = deferred.act(before)
        assert a.action == b.action
        assert a.diagnostics['selection']['status'] == b.diagnostics['selection']['status']
        from gpt_field.observation import PolicyObservation
        data = before.to_dict()
        data['observation/state'][:3] = (
            np.asarray(before.state[:3])+np.asarray(a.action[:3])*.05*.8).tolist()
        after = PolicyObservation.from_dict(data)
        count_a, count_b = legacy.render_count, deferred.render_count
        row_a = legacy.observe_transition(before, a, after)
        row_b = deferred.observe_transition(before, b, after)
        assert legacy.render_count-count_a > 1
        assert deferred.render_count-count_b == 1
        assert legacy.refresh_reason == deferred.refresh_reason
        assert legacy.diagnostic_counters == deferred.diagnostic_counters
        assert legacy.subgoals.summary() == deferred.subgoals.summary()
        assert legacy.translation_guard.experiments == deferred.translation_guard.experiments
        assert legacy.translation_guard.observation_travel_m == deferred.translation_guard.observation_travel_m
        assert legacy.task_progress.summary() == deferred.task_progress.summary()
        assert legacy.robot_response.summary() == deferred.robot_response.summary()
        assert legacy.visual_goal_field.summary() == deferred.visual_goal_field.summary()
        for key in ('before_state', 'after_state', 'action', 'measured_translation_m',
                    'commanded_translation_m', 'visual_goal_check', 'efficiency'):
            assert row_a[key] == row_b[key]
        before = after
    legacy.finish(); deferred.finish()


def test_invalid_acknowledgement_restores_normal_feedback_generation():
    policy = make_policy(DeferredPolicy)
    obs = observation()
    action = policy.act(obs)
    old_feedback = policy.feedback
    with pytest.raises(ValueError, match='pending decision'):
        policy.observe_transition(obs, copy.deepcopy(action), obs)
    assert not policy._transition_feedback_deferred
    assert policy.feedback is old_feedback
    assert policy._observable_feedback()['task_success_certified'] is False
    assert policy.pending is not None
