"""V14 integration tests use synthetic images and robot feedback only."""
from dataclasses import replace
import numpy as np

from gpt_field.visual_field_policy_v14 import VisualFieldPolicyV14, recovery_options
from gpt_field.visual_translation_guard_v14 import ProjectedTranslationGuardV14
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v13_payload_policy import prediction
from test_visual_field_v13_execution import prediction as alignment_prediction, observation as alignment_observation, transition


def test_useful_field_reuse_and_two_view_contact_refinement_remain_available():
    client = Client(alignment_prediction())
    policy = VisualFieldPolicyV14(client)
    obs = alignment_observation()
    accepted = 0
    for _ in range(30):
        obs, decision, row = transition(policy, obs)
        accepted += bool(decision.diagnostics['selection']['accepted'])
        assert decision.diagnostics['policy_version'] == 'visual-field-policy-v14'
        assert decision.action[6] == 0.
        assert not row['task_success_certified']
        if policy.refresh_reason:
            break
    assert accepted > 10 and client.calls == 1
    assert policy.visual_corrections > 0
    assert isinstance(policy.translation_guard, ProjectedTranslationGuardV14)
    assert policy.local.translation_guard is policy.translation_guard
    assert 'reference_support' in row and 'recovery_options' in row
    assert policy.pending is None


def test_gripper_capture_and_measured_feedback_unchanged():
    policy = VisualFieldPolicyV14(Client(prediction()))
    obs = observation()
    decision = policy.act(obs)
    assert decision.action[6] > 0
    after = replace(obs, state=(*obs.state[:6], .037, -.037))
    row = policy.observe_transition(obs, decision, after)
    assert policy.gripper_actions == 1
    assert not row['target_contact_geometry']['carried_payload']['active']
    assert not row['task_success_certified']


def test_nominal_options_never_consume_or_assess_execution_allowance():
    policy = VisualFieldPolicyV14(Client(prediction('visual_field')))
    obs = observation()
    policy.act(obs)
    before = policy.translation_guard.summary()['recovery_experiments']
    for _ in range(4):
        options = recovery_options(policy.local, policy.translation_guard, obs.state)
        assert not options['clearance_certified']
    assert policy.translation_guard.summary()['recovery_experiments'] == before


def test_prospective_recovery_keeps_continuous_obstacle_and_unknown_checks():
    class Guard:
        def summary(self):
            return {'conflicting_obstacle_ids': ['obstacle'], 'recovery_experiments':
                {'trials': [{}], 'shared_remaining_m': .012}}
    class Planner:
        def _valid_segment(self, field, start, end, boxes):
            # A continuous region between grid vertices blocks every +Z proposal.
            assert boxes == ['continuous_hard_and_unknown_boxes']
            return end[2] <= start[2]
    class Local:
        intent = {'motion_axes': 'xyz'}
        field = object()
        boxes = ['continuous_hard_and_unknown_boxes']
        planner = Planner()
        def _nominal_away(self, point, delta):
            return delta[2] > 0 or delta[0] > 0
    choices = recovery_options(Local(), Guard(), (0.,)*8)
    assert len(choices['candidates']) == 1
    assert choices['candidates'][0]['direction_xyz'] == [1., 0., 0.]
    assert choices['candidates'][0]['checked_nominal_segment_m'] == .008
    assert choices['actual_segment_must_be_rechecked']
