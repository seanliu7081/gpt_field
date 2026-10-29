"""Real policy integration preserves contact control and state acknowledgement."""
import copy
import numpy as np

from gpt_field.visual_field_policy_v21 import VisualFieldPolicyV21
from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21
from test_visual_field_v6_policy import Client
from test_visual_field_v17_policy import prediction as contact_prediction
from test_visual_field_v13_execution import observation, transition
from test_visual_field_v20_policy import prediction, fixed_rgb_transition


def test_new_tracker_policy_reuses_fixed_goal_beyond30mm_without_clearing_response_debt():
    policy=VisualFieldPolicyV21(Client(prediction()));obs=observation();start=np.array(obs.state[:3])
    for _ in range(35):
        obs,decision,row=fixed_rgb_transition(policy,obs)
        assert decision.diagnostics['selection']['accepted']
        assert not policy.refresh_reason
    assert np.linalg.norm(np.array(obs.state[:3])-start)>.035
    assert policy.client.calls==1
    assert isinstance(policy.monitor.tracks['moving'],PartFeatureTrackV21)
    assert not policy.monitor.response_ready
    assert policy.monitor._active_reference['unvalidated_travel_m']>.035
    assert decision.diagnostics['policy_version']=='visual-field-policy-v21'
    assert row['task_progress']['active']['samples']==35
    assert not row['task_progress']['active']['review_required']
    assert policy.pending is policy.local.pending is None


def test_contact_refinement_remains_observation_driven_with_new_tracker():
    value=contact_prediction();policy=VisualFieldPolicyV21(Client(value));obs=observation()
    accepted=0
    for _ in range(25):
        obs,decision,row=transition(policy,obs)
        accepted+=bool(decision.diagnostics['selection']['accepted'])
        assert decision.action[6]==0
        if policy.refresh_reason:break
    assert accepted>10 and policy.client.calls==1
    assert policy.visual_corrections>0
    assert policy.prediction==value
    assert not row['task_success_certified']


def test_unresolved_same_objective_conflict_does_not_gain_permission_from_retain():
    client=Client(prediction());policy=VisualFieldPolicyV21(client);obs=observation()
    policy._refresh(obs)
    policy.task_progress.active['review_required']=True
    policy.task_progress.active['reason']='metric_goal_progress_disagrees_with_declared_image_objective'
    retained=copy.deepcopy(client.value);retained['subgoal_update']['action']='retain'
    for view in ('external','wrist'):
        if retained['visual_references'].get(view):
            retained['visual_references'][view].update(update='retain',seed=None)
    client.value=retained;policy.refresh_reason='metric_goal_progress_disagrees_with_declared_image_objective'
    decision=policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert decision.diagnostics['selection']['status']=='task_goal_hypothesis_requires_revision'
    assert policy.local.pending is None and decision.action[6]==0
    policy.observe_transition(obs,decision,obs)
    assert policy.pending is None
