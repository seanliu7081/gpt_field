"""Exercise the integrated policy using actual observer/Field implementations."""
import numpy as np

from gpt_field.visual_field_policy_v34 import VisualFieldPolicyV34
from gpt_field.visual_field_proposals_v34 import FieldProposalBankV34
from gpt_field.visual_field_v34 import MetricFieldExecutorV34
from gpt_field.visual_reference_review_v34 import TypedReferenceReviewPolicyMixinV34
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def test_integrated_v34_preserves_field_reuse_and_final_observer_feedback():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    policy = VisualFieldPolicyV34(Client(value))
    obs = observation()
    try:
        for _ in range(8):
            obs, decision, feedback = fixed_rgb_transition(policy, obs)
            assert decision.diagnostics['selection']['accepted']
            assert decision.diagnostics['policy_version'] == 'visual-field-policy-v34'
            assert feedback['rigid_proxy_use']['metric_field_planning_retained']
            assert not policy._transition_feedback_deferred
            assert feedback['reference_review_dispatch'] == policy.reference_review_dispatch
        assert policy.client.calls == 1
        assert type(policy.local) is MetricFieldExecutorV34
        assert type(policy.field_proposals) is FieldProposalBankV34
        assert policy._review_paid_reference.__func__ is TypedReferenceReviewPolicyMixinV34._review_paid_reference
        assert np.linalg.norm(policy.local.goal-np.asarray(obs.state[:3])) > 0
        assert not policy.visual_goal_field.propose(obs.state)['task_success_certified']
        assert policy.block[-1]['geometry_semantics'] == policy.feedback['geometry_semantics']
    finally:
        policy.finish()
