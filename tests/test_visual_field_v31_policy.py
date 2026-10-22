"""Actual policy/planner integration: no online model or simulator required."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v30 import VisualFieldPolicyV30
from gpt_field.visual_field_policy_v31 import VisualFieldPolicyV31
from gpt_field.visual_field_proposals_v31 import observation_binding, FieldProposalBankV31
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit


def new_policy():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    return VisualFieldPolicyV31(Client(value))


def snapshot(policy):
    return json.dumps({'local': policy.local.summary(), 'guard': policy.translation_guard.summary(),
        'monitor': policy.monitor.summary(), 'scene': policy.inventory.summary(),
        'ledger': policy.subgoals.summary(), 'robot': policy.robot_response.summary()}, sort_keys=True)


def test_ordinary_field_execution_retains_actual_actions_and_one_model_call():
    old = VisualFieldPolicyV30(Client(explicit(prediction())))
    new = new_policy()
    a, b = observation(), observation()
    try:
        for _ in range(12):
            a, da, _ = fixed_rgb_transition(old, a)
            b, db, feedback = fixed_rgb_transition(new, b)
            assert db.action == da.action
            assert db.diagnostics['selection']['accepted']
            assert db.diagnostics['policy_version'] == 'visual-field-policy-v31'
            assert not feedback['visual_goal_field']['task_success_certified']
        assert new.client.calls == old.client.calls == 1
        assert np.array_equal(new.local.goal, old.local.goal)
        assert new.translation_guard.observation_travel_m == old.translation_guard.observation_travel_m
        assert new.inventory.summary()['entries'][0]['metric_age_observations'] == 12
    finally:
        old.finish(); new.finish()


def test_proposal_planner_preview_is_pure_and_retains_margin_unknowns_and_exact_binding():
    policy = new_policy(); obs = observation()
    try:
        for _ in range(3):
            obs, _, _ = fixed_rgb_transition(policy, obs)
        before = snapshot(policy)
        refs = {name: id(track) for name, track in policy.monitor.tracks.items()}
        report = policy.field_proposals.build(policy, obs,
            {'candidates': [{'goal_offset_m': [.01, 0., 0.]}]})
        assert snapshot(policy) == before
        assert refs == {name: id(track) for name, track in policy.monitor.tracks.items()}
        offer = next(x for x in report['offers'] if x['source'] == 'visual_goal_field')
        assert offer['field_intent']['uncertainty_m'] == policy.local.intent['uncertainty_m']
        assert offer['field_intent']['orientation_command'] == 'hold'
        assert 'obstacles' not in offer['field_intent']
        assert offer['subgoal_update']['action'] == 'replace'
        assert offer['subgoal_update']['subgoal_id'] == policy.subgoals.active['subgoal_id']
        assert offer['binding']['observation_sha256'] == observation_binding(obs)
        assert policy.field_proposals.resolve(offer['proposal_id'], obs.to_dict()) == offer
        changed = list(obs.state); changed[0] += .0001
        with pytest.raises(ValueError, match='observation changed'):
            policy.field_proposals.resolve(offer['proposal_id'], replace(obs, state=tuple(changed)))
        with pytest.raises(ValueError, match='Unknown'):
            policy.field_proposals.resolve('not-a-current-offer', obs)
        assert snapshot(policy) == before
        assert not offer['physical_debt_replenished'] and not offer['clearance_certified']
    finally:
        policy.finish()


def test_full_segment_refuses_blocked_goal_before_any_execution():
    policy = new_policy(); obs = observation()
    try:
        obs, _, _ = fixed_rgb_transition(policy, obs)
        # Existing target box is at(.3,.3,.3); this proposal is outside the
        # bounded local field and cannot acquire an apparently safe first step.
        before = snapshot(policy)
        report = policy.field_proposals.build(policy, obs,
            {'candidates': [{'goal_offset_m': [.3, .3, .3]}]})
        assert not any(x['source'] == 'visual_goal_field' for x in report['offers'])
        assert report['rejected'][0]['reason'] == 'proposal_offset_outside_local_domain'
        assert snapshot(policy) == before
    finally:
        policy.finish()


def test_scene_review_identifies_direction_conflict_without_changing_estimates():
    policy = new_policy(); obs = observation()
    try:
        obs, _, _ = fixed_rgb_transition(policy, obs)
        before = snapshot(policy)
        report = {'candidates': [{'goal_offset_m': [-.01, 0., 0.]}]}
        policy._review_scene(obs, report)
        assert any(x['reason'] == 'image_improving_heading_opposes_metric_target_centre'
                   for x in policy.scene_review['reasons'])
        assert snapshot(policy) == before
        assert not policy.scene_review['automatic_scene_change']
    finally:
        policy.finish()


def test_nonempty_unknowns_roundtrip_native_wire_keeps_world_position():
    from gpt_field.visual_field_subscription import _validate
    from gpt_field.visual_field_subscription_v4 import _runtime_schema
    from gpt_field.visual_field_subscription_v30 import output_schema
    policy = new_policy(); obs = observation()
    try:
        obs, _, _ = fixed_rgb_transition(policy, obs)
        policy.local.intent['unknown_regions'] = [{'id': 'unseen-region',
            'lower_offset_m': [.045, .035, .025], 'upper_offset_m': [.065, .075, .045],
            'uncertainty_m': .012}]
        origin = policy.local.origin.copy()
        state = np.asarray(obs.state)
        internal = FieldProposalBankV31._intent(policy, state, [.01, 0., 0.])
        wire = FieldProposalBankV31._wire_intent(internal)
        _validate(wire, _runtime_schema(output_schema()['properties']['field_intent']['anyOf'][0]))
        box = wire['unknown_regions'][0]
        low = state[:3]+np.asarray(box['center_offset_m'])-box['half_extents_m']
        high = state[:3]+np.asarray(box['center_offset_m'])+box['half_extents_m']
        assert low == pytest.approx(origin+[.045, .035, .025])
        assert high == pytest.approx(origin+[.065, .075, .045])
        assert box['uncertainty_m'] == .012
    finally:
        policy.finish()
