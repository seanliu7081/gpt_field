"""Integrated observers and independent objectives with actual local control."""
import copy

import numpy as np

from gpt_field.visual_field_policy_v35 import VisualFieldPolicyV35
from gpt_field.visual_field_proposals_v35 import FieldProposalBankV35
from gpt_field.visual_navigation_fields_v35 import VisualNavigationFieldsV35
from gpt_field.visual_translation_guard_v35 import ProjectedTranslationGuardV35
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit
from test_visual_wrist_goal_v35 import trained


def test_integrated_observers_reuse_field_and_keep_wrist_state_current():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    policy = VisualFieldPolicyV35(Client(value))
    obs = observation()
    try:
        for index in range(8):
            obs, decision, feedback = fixed_rgb_transition(policy, obs)
            assert decision.diagnostics['selection']['accepted']
            assert decision.diagnostics['policy_version'] == 'visual-field-policy-v35'
            assert not policy._transition_feedback_deferred
            wrist = policy.visual_goal_field.wrist_navigation
            assert wrist.observed_transitions == index + 1
            assert np.array_equal(wrist.state, obs.state)
            assert feedback['visual_goal_field']['wrist_navigation'] == wrist.summary()
        assert policy.client.calls == 1
        assert isinstance(policy.translation_guard, ProjectedTranslationGuardV35)
        assert isinstance(policy.field_proposals, FieldProposalBankV35)
        assert policy.local.translation_guard is policy.translation_guard
        limits = feedback['execution_limit_sources']
        assert limits['scene_execution_limits'] == policy.scene_execution_limits
        assert limits['effective_intent_max_travel_m'] == policy.local.intent['max_travel_m']
        assert not feedback['visual_goal_field']['wrist_is_anatomical_or_contact_evidence']
        assert policy.block[-1]['geometry_semantics'] == feedback['geometry_semantics']
    finally:
        policy.finish()


def test_composite_preserves_real_wrist_learning_across_external_proxy_changes():
    wrist, monitor, prediction, state = trained()
    composite = VisualNavigationFieldsV35()
    composite.wrist_navigation = wrist
    before = copy.deepcopy(wrist.summary())
    prediction['visual_references']['external'] = {
        'reference_role': 'rigid_transit_proxy', 'moving_kind': 'tool',
        'moving_id': 'housing', 'target_id': 'object'}
    composite.configure(state, monitor, prediction)
    assert wrist.summary() == before
    report = composite.propose(state, motion_axes='xy')
    assert report['accepted'] and report['objective_source'] == 'wrist_bearing'
    assert not report['available_objectives']['anatomical_relative_pixels']['accepted']
    assert all(row['objective_source'] == 'wrist_bearing'
               and not row['contact_alignment_evidence'] for row in report['candidates'])
    assert not report['task_success_certified']
