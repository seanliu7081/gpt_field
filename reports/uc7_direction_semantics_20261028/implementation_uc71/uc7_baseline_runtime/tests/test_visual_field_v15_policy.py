"""V15 measured Field reuse, diagnostic freshness and admission-aware proposals."""
from dataclasses import replace
import copy
import numpy as np

from gpt_field.visual_field_policy_v15 import VisualFieldPolicyV15, recovery_options
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v13_payload_policy import prediction
from test_visual_field_v13_execution import prediction as alignment_prediction, observation as alignment_observation, transition


def test_real_field_reuse_refines_contact_and_collects_motion_without_extra_requests():
    client = Client(alignment_prediction())
    policy = VisualFieldPolicyV15(client)
    obs = alignment_observation()
    accepted = 0
    initial = obs
    for _ in range(30):
        obs, decision, row = transition(policy, obs)
        accepted += bool(decision.diagnostics['selection']['accepted'])
        assert decision.diagnostics['policy_version'] == 'visual-field-policy-v15'
        assert decision.action[6] == 0.
        if policy.refresh_reason:
            break
    assert accepted > 10 and client.calls == 1 and policy.visual_corrections > 0
    assert policy.pending is None and not row['task_success_certified']
    assert 'motion_evidence' in row
    assert client.motion_diagnostic_provider(obs) is not None
    assert client.motion_diagnostic_provider(initial) is None  # Never label an old image current.


def test_explicit_gripper_capture_does_not_invent_attachment_or_motion_diagnostic():
    policy = VisualFieldPolicyV15(Client(prediction()))
    obs = observation()
    decision = policy.act(obs)
    assert decision.action[6] > 0
    after = replace(obs, state=(*obs.state[:6], .037, -.037))
    row = policy.observe_transition(obs, decision, after)
    assert policy.gripper_actions == 1
    assert not row['target_contact_geometry']['carried_payload']['active']
    assert policy.client.motion_diagnostic_provider(after) is None
    assert not row['task_success_certified']


class ProposalPlant:
    intent = {'motion_axes': 'xyz'}
    field = object()
    boxes = ['hard_target_and_unknown_boxes']
    class Planner:
        def _valid_segment(self, field, start, end, boxes):
            assert boxes == ['hard_target_and_unknown_boxes']
            return end[2] <= start[2]  # Positive Z crosses a thin hard region.
    planner = Planner()
    def _nominal_away(self, point, delta):
        return delta[2] > 0 or abs(delta[0]) > 0
    def nominal_regions_for_segment(self, start, end):
        return [{'id': 'neighbor', 'core_distance_m': .04, 'uncertainty_m': .03}]


class Admission:
    def __init__(self):
        self.calls = []
    def summary(self):
        return {'conflicting_obstacle_ids': ['neighbor'], 'recovery_experiments':
            {'trials': [{}], 'shared_remaining_m': .004}}
    def preview_assess(self, delta, **kwargs):
        self.calls.append((np.asarray(delta).copy(), kwargs))
        return {'accepted': delta[0] < 0, 'reason': 'existing_direction_remaining' if delta[0] < 0
                else 'translation_recovery_experiment_limit', 'cap_m': .0015 if delta[0] < 0 else 0.}


def test_geometry_clear_but_admission_refused_direction_is_not_offered():
    guard = Admission()
    result = recovery_options(ProposalPlant(), guard, (0.,)*8)
    assert len(result['candidates']) == 1
    offered = result['candidates'][0]
    assert offered['direction_xyz'] == [-1., 0., 0.]
    assert offered['preview_first_command_delta_m'] == [-.0015, 0., 0.]
    assert offered['preview_reason'] == 'existing_direction_remaining'
    assert any(row['reason'] == 'translation_recovery_experiment_limit' for row in result['rejected_candidates'])
    assert any(row['reason'] == 'nominal_segment_blocked' for row in result['rejected_candidates'])
    assert all(kw['prospective_translation_mode'] == 'separate' for _, kw in guard.calls)
    assert all(kw['orientation_hold'] and kw['nominal_clear'] for _, kw in guard.calls)
    assert result['actual_segment_must_be_rechecked'] and not result['clearance_certified']


def test_proposal_carries_current_response_monitor_and_small_remaining_allowance_without_reset():
    guard = Admission(); monitor = object(); response = object()
    before = copy.deepcopy(guard.summary())
    result = recovery_options(ProposalPlant(), guard, (0.,)*8, monitor=monitor, robot_response=response)
    assert guard.summary() == before
    assert result['shared_remaining_m'] == .004
    assert all(kw['monitor'] is monitor and kw['translation_response'] is response for _, kw in guard.calls)
    assert result['candidates'][0]['local_limit_does_not_override_persistent_guard_allowance']


def test_capped_command_losing_nominal_separation_is_not_offered():
    class Plant(ProposalPlant):
        def _nominal_away(self, point, delta):
            return abs(delta[0]) >= .0018
    class Guard(Admission):
        def preview_assess(self, delta, **kwargs):
            row = super().preview_assess(delta, **kwargs)
            if not kwargs['nominal_away']:
                row.update(accepted=False, reason='changed_heading_requires_nominal_separation')
            return row
    guard = Guard()
    result = recovery_options(Plant(), guard, (0.,)*8)
    assert not result['candidates']
    assert any(row['reason'] == 'changed_heading_requires_nominal_separation'
               for row in result['rejected_candidates'])
    checked = [delta[0] for delta, _ in guard.calls if delta[0] < 0]
    assert checked == [-.002, -.0015]
