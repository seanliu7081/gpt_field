"""Opt-in policy wiring preserves ordinary Field execution and explicit control."""
import numpy as np
from gpt_field.visual_field_policy_v28 import VisualFieldPolicyV28
from gpt_field.visual_field_policy_v29 import VisualFieldPolicyV29
from gpt_field.visual_reference_candidates_v28 import CausalReferenceCandidatesV28
from gpt_field.visual_translation_guard_v29 import ProjectedTranslationGuardV29
from gpt_field.visual_field_v29 import MetricFieldExecutorV29
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def test_normal_field_reuse_keeps_same_actions_no_extra_model_requests_or_debt():
    old=VisualFieldPolicyV28(Client(explicit(prediction())))
    new=VisualFieldPolicyV29(Client(explicit(prediction())))
    assert isinstance(new.local,MetricFieldExecutorV29)
    assert isinstance(new.translation_guard,ProjectedTranslationGuardV29)
    assert isinstance(new.reference_candidates,CausalReferenceCandidatesV28)
    assert new.client.reference_candidate_provider.__self__ is new.reference_candidates
    a,b=observation(),observation()
    try:
        for _ in range(12):
            a,da,_=fixed_rgb_transition(old,a)
            b,db,feedback=fixed_rgb_transition(new,b)
            assert db.action==da.action and db.diagnostics['selection']['accepted']
            assert db.diagnostics['policy_version']=='visual-field-policy-v29'
            assert new.local.pending is new.pending is None
            assert not feedback['task_success_certified']
        assert new.client.calls==old.client.calls==1
        assert new.translation_guard.observation_travel_m==old.translation_guard.observation_travel_m
        assert np.array_equal(new.local.goal,old.local.goal)
        assert new.reference_candidates.maximum_frames==32
    finally:
        old.finish();new.finish()
