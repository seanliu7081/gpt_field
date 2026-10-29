"""Independent cross-layer checks: real local offers, fake subscription only."""
import copy
from dataclasses import replace

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from test_visual_field_v31_policy import new_policy, snapshot
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import fixed_rgb_transition
from test_visual_field_v31_protocol import client, prediction


def select_offer(offer):
    wire = prediction(controller='visual_field')
    for key in ('phase', 'target_id', 'recovery_route', 'subgoal_update', 'visual_references'):
        wire[key] = copy.deepcopy(offer[key])
    wire['field_intent'] = None
    wire['field_proposal_id'] = offer['proposal_id']
    wire['scene_inventory'] = copy.deepcopy(offer['scene_declarations'])
    wire['motion_context'] = copy.deepcopy(offer['motion_context_overrides'])
    return wire


def test_actual_planner_offer_roundtrips_native_client_without_state_or_debt_mutation():
    policy = new_policy()
    obs = replace(observation(), prompt=resolve_distance_task('bowl_plate').prompt)
    try:
        for _ in range(3):
            obs, _, _ = fixed_rgb_transition(policy, obs)
        report = policy.field_proposals.build(policy, obs,
            {'candidates': [{'goal_offset_m': [.01, 0., 0.]}]})
        offer = next(x for x in report['offers'] if x['source'] == 'visual_goal_field')
        wire = select_offer(offer)
        before = snapshot(policy)
        instance, transport = client(wire)
        instance.field_proposal_resolver = policy.field_proposals.resolve
        instance.scene_revision_validator = policy.inventory.validate_revision_update
        instance.observable_context = {'visual_feedback': policy._observable_feedback()}
        result = instance.predict(obs.to_dict())
        assert result['raw_wire_prediction'] == wire
        assert result['field_proposal_resolution'] == offer
        assert result['field_intent']['goal_offset_m'] == offer['field_intent']['goal_offset_m']
        assert result['field_intent']['uncertainty_m'] == policy.local.intent['uncertainty_m']
        assert snapshot(policy) == before
        assert len(transport.calls) == 1
        assert instance.budget_snapshot()['api_key_calls'] == 0
    finally:
        policy.finish()


def test_real_persistent_inventory_invalid_revision_repairs_before_any_scene_change():
    policy = new_policy()
    obs = replace(observation(), prompt=resolve_distance_task('bowl_plate').prompt)
    try:
        obs, _, _ = fixed_rgb_transition(policy, obs)
        offer_report = policy.field_proposals.build(policy, obs,
            {'candidates': [{'goal_offset_m': [.01, 0., 0.]}]})
        offer = next(x for x in offer_report['offers'] if x['source'] == 'visual_goal_field')
        invalid = select_offer(offer)
        invalid['field_proposal_id'] = None
        invalid['field_intent'] = copy.deepcopy(offer['field_intent'])
        entry = policy.inventory.summary(obs.state)['entries'][0]
        invalid['scene_inventory'][0]['geometry'] = copy.deepcopy(entry['current_geometry'])
        invalid['scene_inventory'][0]['geometry']['center_offset_m'][0] += .005
        good = copy.deepcopy(invalid)
        good['scene_revisions'] = [{'id': entry['id'], 'reason': 'Current view and measured displacement disagree with the old estimate.',
                                    'evidence_kind': 'measured_motion_conflict'}]
        before = snapshot(policy)
        instance, transport = client(invalid, good)
        instance.scene_revision_validator = policy.inventory.validate_revision_update
        instance.observable_context = {'visual_feedback': policy._observable_feedback()}
        result = instance.predict(obs.to_dict())
        assert result['raw_wire_prediction'] == good
        assert len(transport.calls) == 2
        assert snapshot(policy) == before
        # Parsing a revision is not inventory application or a controller step.
        assert policy.inventory.summary(obs.state)['entries'][0]['current_geometry'] == entry['current_geometry']
    finally:
        policy.finish()
