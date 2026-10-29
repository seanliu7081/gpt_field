"""Paid RGB history reaches the actual request without extra simulated motion."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation
from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
from gpt_field.visual_reference_candidates_v24 import CausalReferenceCandidatesV24
from gpt_field.visual_reference_candidates_v28 import CausalReferenceCandidatesV28
from gpt_field.visual_field_subscription_v28 import VisualFieldSubscriptionClientV28, CANDIDATE_LABEL
from test_visual_field_v6_policy import Client
from test_visual_field_v7_protocol import current_record, current_images
from test_visual_field_v25_protocol import prediction
from test_field_history import FakeTransport, envelope


ARCHIVE = Path(__file__).resolve().parents[1]/'runs/visual_field_v27_zero_shot_002/trials/bowl_original_zero_hybrid_v27'


def test_only_explicit_refusal_families_and_unchanged_history_bound():
    assert CausalReferenceCandidatesV28.maximum_frames==CausalReferenceCandidatesV24.maximum_frames==32
    assert CausalReferenceCandidatesV24.handoff_reasons <= CausalReferenceCandidatesV28.handoff_reasons
    for reason in ('translation_projected_conflict_image_response_unavailable',
                   'translation_projected_conflict_robot_response_unavailable',
                   'translation_reverse_observed_further_worsening',
                   'translation_heading_evidence_allowance_exhausted',
                   'translation_heading_evidence_budget_exhausted',
                   'translation_heading_evidence_evidence_contradicted',
                   'translation_reverse_boundary_evidence_unavailable',
                   'translation_reverse_boundary_closing'):
        assert reason in CausalReferenceCandidatesV28.handoff_reasons
    assert 'arbitrary_error' not in CausalReferenceCandidatesV28.handoff_reasons


def test_actual_paid14_transition_handoff_populates_real_request_board_without_guard_change():
    if not (ARCHIVE/'trace.jsonl').exists(): pytest.skip('Local archived RGB evidence unavailable')
    rows=[json.loads(x) for x in (ARCHIVE/'trace.jsonl').read_text().splitlines()]
    initial=next(x['observation'] for x in rows if x['event']=='reset')
    actions={x['step']:x for x in rows if x['event']=='action'}
    executed={x['step']:x['observation'] for x in rows if x['event']=='executed'}
    policy=VisualFieldPolicyV27(Client(actions[0]['diagnostics']['prediction']))
    # Reproduce the paid V27 dynamics; swap only the passive presentation bank.
    # New V28 motion at the divergence is intentionally not replayed against
    # saved observations from a different command.
    bank=CausalReferenceCandidatesV28();policy.reference_candidates=bank
    policy.client.reference_candidate_provider=bank.image
    policy.client.reference_candidate_resolver=policy._resolve_reference_candidate
    policy.translation_guard.bind_reference_witness(bank)
    obs=PolicyObservation.from_dict(initial)
    try:
        for step in range(14):
            decision=policy.act(obs)
            assert np.allclose(decision.action,actions[step]['action'],atol=1e-10,rtol=0.)
            after=PolicyObservation.from_dict(executed[step])
            policy.observe_transition(obs,decision,after);obs=after
        before=(bank.path_m,policy.translation_guard.observation_travel_m,
                copy.deepcopy(policy.translation_guard.experiments))
        decision=policy.act(obs)
        assert decision.diagnostics['selection']['status']=='translation_projected_conflict_image_response_unavailable'
        assert not decision.diagnostics['selection']['accepted']
        review=bank.summary()
        assert review['review_required'] and review['candidates'],review
        assert review['last_handoff']['reason']=='translation_projected_conflict_image_response_unavailable'
        assert review['reason']=='paid_motion_available_at_control_handoff'
        assert not review['control_debt_modified'] and not review['automatic_reference_promotion']
        assert (bank.path_m,policy.translation_guard.observation_travel_m,policy.translation_guard.experiments)==before
        assert policy.client.calls==1 and bank.sequence==14
        transport=FakeTransport(envelope(prediction(),model='gpt-6-astra',reasoning={'effort':'medium'}))
        client=VisualFieldSubscriptionClientV28(token_provider=lambda:'offline-oauth',transport=transport)
        client.reference_candidate_provider=bank.image
        client.reference_candidate_resolver=policy._resolve_reference_candidate
        client.observable_context={'visual_feedback':policy._observable_feedback(),'refresh_reason':policy.refresh_reason,'task_success_certified':False}
        client.predict(obs.to_dict())
        assert len(transport.calls)==1
        payload=transport.calls[0];record=current_record(payload)
        assert record['observable_context']['visual_feedback']['reference_repair']['candidates']==review['candidates']
        images=current_images(payload)
        assert images[:2]==[obs.image,obs.wrist_image] and len(images)==3
        assert any(x.get('text')==CANDIDATE_LABEL for m in payload['input'] for x in m.get('content',[]))
        assert images[-1]==bank.image(obs.to_dict())
        assert not bank.image({**obs.to_dict(),'observation/state':[obs.state[0]+.001,*obs.state[1:]]})
        with pytest.raises(ValueError):bank.resolve_candidate(review['candidates'][0]['candidate_id'],initial)
        assert client.budget_snapshot()['api_key_calls']==0
    finally:
        policy.finish()


@pytest.mark.parametrize('reason',['translation_heading_evidence_allowance_exhausted',
                                  'translation_reverse_boundary_closing'])
def test_new_handoff_changes_presentation_only_and_arbitrary_error_does_not(reason):
    from test_visual_reference_candidates_v24 import setup,advance
    bank,monitor,context,obs=setup(moving=False)
    bank.__class__=CausalReferenceCandidatesV28
    obs=advance(bank,monitor,obs)
    before=(bank.path_m,copy.deepcopy(monitor._active_reference),bank.sequence)
    assert not bank.handoff(obs,monitor,'unrelated_parser_error')['review_required']
    report=bank.handoff(obs,monitor,reason)
    assert report['review_required'] and report['candidates']
    assert report['last_handoff']['reason']==reason
    assert (bank.path_m,monitor._active_reference,bank.sequence)==before
    assert not report['handoff_review_contract']['automatic_identity_or_motion_permission']
