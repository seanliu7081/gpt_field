"""Whole-policy ordinary Field reuse and refusal without observed support."""
import copy

import numpy as np

from gpt_field.visual_field_policy_v26 import VisualFieldPolicyV26
from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
from gpt_field.visual_field_v27 import MetricFieldExecutorV27
from gpt_field.visual_translation_guard_v27 import ProjectedTranslationGuardV27
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def test_ordinary_field_commands_acknowledgments_and_one_request_reuse_unchanged():
    old=VisualFieldPolicyV26(Client(explicit(prediction())))
    new=VisualFieldPolicyV27(Client(explicit(prediction())))
    a,b=observation(),observation()
    assert isinstance(new.local,MetricFieldExecutorV27)
    assert isinstance(new.translation_guard,ProjectedTranslationGuardV27)
    for _ in range(12):
        a,da,_=fixed_rgb_transition(old,a)
        b,db,feedback=fixed_rgb_transition(new,b)
        assert db.action==da.action and db.diagnostics['selection']['accepted']
        assert db.diagnostics['policy_version']=='visual-field-policy-v27'
        assert new.local.pending is new.pending is None
        assert not feedback['translation_guard']['projected_warning_transit']['physical_budget_replenished']
        assert not feedback['task_success_certified']
    assert new.client.calls==old.client.calls==1
    assert new.local.actions==12
    assert new.translation_guard.observation_travel_m==old.translation_guard.observation_travel_m
    assert np.array_equal(new.local.goal,old.local.goal)
    old.finish();new.finish()


def test_model_clearance_alone_cannot_override_warning_without_fresh_image_response():
    value=explicit(prediction())
    # Synthetic declared geometry: the selected +X task goal increases model
    # clearance to the named object. This is not supplied physical truth.
    center=[-.3,.3,.3]
    value['scene_inventory'][0]['geometry']['center_offset_m']=center
    value['target_geometry']['center_offset_m']=center
    policy=VisualFieldPolicyV27(Client(value));obs=observation()
    policy._refresh(obs)
    guard=policy.translation_guard;identity=value['target_id']
    guard.worsening.add(identity)
    before=copy.deepcopy((guard.observation_travel_m,guard.experiments))
    decision=policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert np.linalg.norm(decision.action[:3])==0.
    assert identity in guard.worsening
    assert (guard.observation_travel_m,guard.experiments)==before
    assert policy.client.calls==1
    policy.observe_transition(obs,decision,obs)
    assert policy.local.pending is policy.pending is None
    assert identity in guard.worsening
    policy.finish()
