"""Actual policy admission from the recorded V22 initially blocked observation."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from gpt_field.observation import PolicyObservation
from gpt_field.visual_field_policy_v22 import VisualFieldPolicyV22
from gpt_field.visual_field_policy_v23 import VisualFieldPolicyV23
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition


def archive_initial():
    path = Path(__file__).resolve().parents[1] / 'runs/visual_field_v22_zero_shot_001/trials/bowl_original_zero_hybrid_v22/trace.jsonl'
    initial = value = None
    for line in path.open():
        row = json.loads(line)
        if row['event'] == 'reset':
            initial = PolicyObservation.from_dict(row['observation'])
        if row['event'] == 'action' and row['step'] == 0:
            value = row['diagnostics']['prediction']
            break
    assert initial is not None and value is not None
    return initial, value


class OfferClient(Client):
    def predict(self, obs):
        self.calls += 1
        value = copy.deepcopy(self.value)
        if self.calls == 1:
            return value
        candidates = self.observable_context['visual_feedback']['recovery_options']['candidates']
        self.offer = next(c for c in candidates if c.get('source') == 'initial_uncertainty_egress')
        value.update(phase='recover', reason='Explicitly test the current checked uncertainty-egress offer.')
        value['motion_context'].update(purpose='reference_repair', translation_mode='separate')
        value['field_intent'].update(goal_offset_m=self.offer['recommended_goal_offset_m'],
            goal_tolerance_m=self.offer['recommended_goal_tolerance_m'],
            command_limit_m=self.offer['recommended_command_limit_m'],
            max_travel_m=self.offer['recommended_local_max_travel_m'],
            max_actions=self.offer['recommended_max_actions'], motion_axes='xyz')
        value['subgoal_update'].update(action='replace', reason='Initial goal rejected by uncertainty; explicitly select checked egress.')
        return value


def test_archived_initial_refusal_offers_executable_model_selected_field_recovery():
    obs, value = archive_initial()
    client = OfferClient(value)
    policy = VisualFieldPolicyV23(client)
    first = policy.act(obs)
    assert first.diagnostics['selection']['status'] == 'invalid_measured_start'
    policy.observe_transition(obs, first, obs)
    offers = [c for c in policy.recovery_options['candidates']
              if c.get('source') == 'initial_uncertainty_egress']
    assert offers, policy.recovery_options
    before = pickle.dumps(policy.translation_guard.egress_evidence.__dict__)
    for _ in range(2):
        policy._screen_recovery_options(obs.state)
        assert pickle.dumps(policy.translation_guard.egress_evidence.__dict__) == before
        assert policy.local.pending is None
    second = policy.act(obs)
    selected = second.diagnostics['selection']
    assert selected['accepted'], selected['status']
    assert np.allclose(np.asarray(second.action[:3])*.05,
                       client.offer['preview_first_command_delta_m'], atol=1e-10)
    assert second.action[6] == 0
    assert client.calls == 2
    assert len(policy.inventory.objects) == 8
    assert policy.translation_guard.observation_travel_m == 0
    assert pickle.dumps(policy.translation_guard.egress_evidence.__dict__) == before
    assert second.diagnostics['policy_version'] == 'visual-field-policy-v23'


def test_missing_robot_joint_feedback_does_not_offer_or_execute_a_recovery():
    obs, value = archive_initial()
    policy = VisualFieldPolicyV23(Client(value))
    policy.act(obs)
    policy._recovery_joint_position = None
    policy._screen_recovery_options(obs.state)
    assert not any(c.get('source') == 'initial_uncertainty_egress'
                   for c in policy.recovery_options['candidates'])


def test_normal_field_commands_progress_and_call_count_match_v22():
    previous, current = VisualFieldPolicyV22(Client(prediction())), VisualFieldPolicyV23(Client(prediction()))
    old_obs, new_obs = observation(), observation()
    for _ in range(12):
        old_obs, old_decision, old_row = fixed_rgb_transition(previous, old_obs)
        new_obs, new_decision, new_row = fixed_rgb_transition(current, new_obs)
        assert new_decision.action == old_decision.action
        assert new_decision.diagnostics['selection']['accepted']
        assert new_row['task_progress'] == old_row['task_progress']
    assert current.client.calls == previous.client.calls == 1
    assert current.local.pending is current.pending is None

