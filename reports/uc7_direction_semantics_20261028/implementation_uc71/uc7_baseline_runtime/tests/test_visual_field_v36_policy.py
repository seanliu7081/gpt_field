"""Real policy observers, early decision handoff and unchanged Field control."""
import copy

import numpy as np

from gpt_field.visual_field_policy_v36 import VisualFieldPolicyV36
from gpt_field.visual_field_proposals_v31 import observation_binding
from gpt_field.visual_intent_replan_v36 import ordinary_replan_previews
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def value():
    result = explicit(prediction())
    result.update(scene_revisions=[], field_proposal_id=None)
    result['visual_references']['external']['reference_role'] = 'tool_anatomy'
    return result


def test_no_adverse_wrist_evidence_reuses_field_without_extra_model_calls():
    policy = VisualFieldPolicyV36(Client(value()))
    obs = observation()
    for index in range(8):
        obs, decision, feedback = fixed_rgb_transition(policy, obs)
        assert decision.diagnostics['selection']['accepted']
        assert decision.diagnostics['policy_version'] == 'visual-field-policy-v36'
        assert policy.visual_goal_field.wrist_navigation.observed_transitions == index + 1
        assert policy.intent_watch.observed_transitions == index + 1
        assert feedback['ongoing_objective_review'] == policy.intent_watch.summary()
        assert not feedback['ongoing_objective_review']['review_required']
        assert feedback['ordinary_replan_previews'] is None
        assert policy.block[-1]['ongoing_objective_review'] == feedback['ongoing_objective_review']
        assert policy.pending is policy.local.pending is None
    assert policy.client.calls == 1
    assert not policy.refresh_reason


def test_review_reaches_next_request_then_explicit_replacement_continues_without_hold(monkeypatch):
    class ReviewClient(Client):
        def predict(self, obs):
            if self.calls:
                self.received = copy.deepcopy(self.observable_context['visual_feedback'])
                result = copy.deepcopy(self.value)
                result['subgoal_update'].update(action='replace', reason='Review test current target direction')
                result['field_intent']['goal_offset_m'] = [.030, 0., 0.]
                result['visual_references']['external'].update(update='retain', seed=None)
                for entry in result['scene_inventory']:
                    entry['geometry'] = None  # Retain world hypotheses at the new robot state.
                self.calls += 1
                return result
            return super().predict(obs)

    client = ReviewClient(value())
    policy = VisualFieldPolicyV36(client)
    obs = observation()
    original_observe = policy.intent_watch.observe
    state = {}

    def event_after_all_observers(before, action, after, **kwargs):
        assert policy.pending is policy.local.pending is None
        assert np.array_equal(policy.visual_goal_field.wrist_navigation.state, after)
        state['guard'] = copy.deepcopy(policy.translation_guard.summary())
        state['local'] = copy.deepcopy(policy.local.summary())
        original_observe(before, action, after, **kwargs)
        # Pure monitor positive case; real-trace trigger validation is separate.
        event = {'reason': 'wrist_target_visibility_declining_under_current_heading',
                 'event_id': 'test-current-causal-visibility-event'}
        policy.intent_watch._event = event
        policy.intent_watch._new_event = True
        return policy.intent_watch.summary()

    monkeypatch.setattr(policy.intent_watch, 'observe', event_after_all_observers)
    obs, first, feedback = fixed_rgb_transition(policy, obs)
    assert first.diagnostics['selection']['accepted']
    assert policy.refresh_reason == 'wrist_target_visibility_declining_under_current_heading'
    assert policy.translation_guard.summary() == state['guard']
    assert policy.local.summary() == state['local']
    assert policy.local.stop_reason is None
    assert not (policy.task_progress.active or {}).get('review_required')
    assert feedback['ongoing_objective_review']['new_event']
    previews = {'observation_sha256': observation_binding(obs), 'candidates': [], 'review_test': True}
    calls = []
    def preview(current, actual):
        calls.append(actual)
        return copy.deepcopy(previews)
    monkeypatch.setattr('gpt_field.visual_field_policy_v36.ordinary_replan_previews', preview)
    second = policy.act(obs)
    assert client.calls == 2 and calls == [obs]
    assert client.received['ongoing_objective_review']['event']['event_id'] == 'test-current-causal-visibility-event'
    assert client.received['ordinary_replan_previews'] == previews
    assert not policy.intent_watch.summary()['review_required']
    assert second.diagnostics['selection']['accepted'], second.diagnostics['selection']
    assert np.linalg.norm(second.action[:3]) > 0
    assert policy.subgoals.active['subgoal_id'] == client.value['subgoal_update']['subgoal_id']
    assert policy.feedback['ordinary_replan_previews'] is None


def test_ordinary_previews_keep_live_state_and_exclude_pending_or_contact():
    policy = VisualFieldPolicyV36(Client(value()))
    obs = observation()
    decision = policy.act(obs)
    result = ordinary_replan_previews(policy, obs)
    assert not result['candidates'] and result['reason'] == 'ordinary_noncontact_approach_context_unavailable'
    policy.observe_transition(obs, decision, obs)
    before_guard, before_local = copy.deepcopy(policy.translation_guard.summary()), copy.deepcopy(policy.local.summary())
    result = ordinary_replan_previews(policy, obs)
    assert len(result['candidates']) == 6
    assert result['observation_sha256'] == observation_binding(obs)
    assert not result['validated_visual_improvement'] and not result['automatic_action']
    assert not result['physical_debt_replenished']
    assert policy.translation_guard.summary() == before_guard
    assert policy.local.summary() == before_local
    policy.prediction['target_contact'] = 'fingers'
    assert not ordinary_replan_previews(policy, obs)['candidates']
