"""Actual policy Field route with deterministic synthetic RGB transitions."""
import numpy as np

from gpt_field.visual_field_policy_v33 import VisualFieldPolicyV33
from gpt_field.visual_field_v33 import MetricFieldExecutorV33
from gpt_field.visual_reference_track_v32 import ExactDescriptorPartTrackV32
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def test_new_policy_reuses_metric_field_and_measured_feedback_without_more_model_calls():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    policy = VisualFieldPolicyV33(Client(value))
    obs = observation()
    try:
        for _ in range(8):
            obs, decision, feedback = fixed_rgb_transition(policy, obs)
            assert decision.diagnostics['selection']['accepted']
            assert decision.diagnostics['policy_version'] == 'visual-field-policy-v33'
            assert feedback['rigid_proxy_use']['metric_field_planning_retained']
            assert not feedback['rigid_proxy_use']['active']
        assert policy.client.calls == 1
        assert type(policy.local) is MetricFieldExecutorV33
        assert type(policy.monitor.tracks['moving']) is ExactDescriptorPartTrackV32
        assert np.linalg.norm(policy.local.goal-np.asarray(obs.state[:3])) > 0
        assert not policy.visual_goal_field.propose(obs.state)['task_success_certified']
    finally:
        policy.finish()
