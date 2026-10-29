"""Whole-policy regressions; synthetic motion is not a task success result."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation
from gpt_field.visual_field_policy_v23 import VisualFieldPolicyV23
from gpt_field.visual_field_policy_v24 import VisualFieldPolicyV24
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation as simple_observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v23_policy import archive_initial, OfferClient
from test_visual_field_v23 import synthetic_egress_field, archive_field


def test_normal_field_commands_and_request_count_remain_unchanged():
    old = VisualFieldPolicyV23(Client(prediction()))
    new = VisualFieldPolicyV24(Client(prediction()))
    a, b = simple_observation(), simple_observation()
    for _ in range(12):
        a, da, ra = fixed_rgb_transition(old, a)
        b, db, rb = fixed_rgb_transition(new, b)
        assert db.action == da.action
        assert db.diagnostics['selection']['accepted']
        assert rb['task_progress'] == ra['task_progress']
    assert old.client.calls == new.client.calls == 1
    assert new.local.pending is new.pending is None


def test_actual_initial_egress_offer_passes_full_policy_and_is_not_a_clearance_claim():
    obs, pred = archive_initial()
    client = OfferClient(pred)
    policy = VisualFieldPolicyV24(client)
    first = policy.act(obs)
    assert first.diagnostics['selection']['status'] == 'invalid_measured_start'
    policy.observe_transition(obs, first, obs)
    second = policy.act(obs)
    assert second.diagnostics['selection']['accepted'], second.diagnostics['selection']
    assert np.asarray(second.action[:3])*.05 == pytest.approx(client.offer['preview_first_command_delta_m'])
    assert not client.offer['positive_image_gap_is_required']
    assert not client.offer['clearance_certified']
    assert not client.offer['reference_repair_confirmation_can_renew']
    assert policy.translation_guard.egress_evidence.spent_m == 0.
    assert client.calls == 2 and len(policy.inventory.objects) == 8


def synthetic_policy():
    field, guard, monitor, inventory, response, plant, state, observe, obs = synthetic_egress_field()
    _, _, _, _, _, pred, _ = archive_field()
    pred = copy.deepcopy(pred)
    pred.pop('raw_wire_prediction', None)
    pred.update(phase='recover', target_id='target', field_intent=copy.deepcopy(field.intent),
                motion_context=copy.deepcopy(guard.context), reference_candidate_id=None)
    pred['field_intent']['goal_offset_m'] = [0., .006, .038]
    pred['field_intent']['obstacles'] = []  # Inventory is materialized by the real policy.
    ext = monitor.refs['external']
    pred['visual_references'] = {'external': {
        'update': 'seed', 'moving_id': ext['moving_id'], 'moving_kind': 'tool',
        'target_id': 'target', 'goal_offset_uv': [0., 0.],
        'seed': {k: copy.deepcopy(ext[k]) for k in ('moving_point_uv', 'moving_roi_uv',
            'target_anchor_uv', 'target_roi_uv', 'tool_point_uv', 'tool_roi_uv')}}, 'wrist': None}
    declarations = []
    for ident, obj in inventory.objects.items():
        geom = obj['world_geometry']
        declarations.append({'id': ident,
            'external_roi_uv': obj['views']['external']['current_roi_uv'],
            'wrist_roi_uv': obj['views']['wrist']['current_roi_uv'],
            'geometry': {'center_offset_m': (np.asarray(geom['center_xyz_m'])-state[:3]).tolist(),
                'half_extents_m': geom['half_extents_m'], 'uncertainty_m': geom['uncertainty_m']}})
    pred['scene_inventory'] = declarations
    pred['target_geometry'] = dict(next(c['geometry'] for c in declarations if c['id']=='target'),
                                   id='target', allowed_contact='none')
    return VisualFieldPolicyV24(Client(pred)), plant, state, observe, obs


def test_whole_policy_fixed_route_exceeds_30mm_without_extra_request_or_gap_grants():
    policy, plant, state, observe, obs = synthetic_policy()
    initial = state[:3].copy()
    goal = initial + [0., .006, .038]
    commands = []
    for step in range(120):
        decision = policy.act(obs)
        selected = decision.diagnostics['selection']
        assert selected['accepted'], (step, selected, policy.feedback.get('controlled_reference_witness'))
        delta = np.asarray(decision.action[:3])*.05*.2
        state[:3] += delta
        plant.travel += float(np.linalg.norm(delta))
        after = observe()
        row = policy.observe_transition(obs, decision, after)
        obs = after
        commands.append(decision.action)
        assert policy.goal == pytest.approx(goal)
        assert policy.local.goal == pytest.approx(goal)
        assert policy.pending is policy.local.pending is None
        assert not row['task_success_certified']
        if plant.travel > .032:
            break
        assert not policy.refresh_reason, (step, policy.refresh_reason, row.get('reference_repair'))
    assert plant.travel > .032 and len(commands) > 50
    assert policy.client.calls == 1 and policy.local.plan_calls == 1
    assert policy.local.goal_revisions == 0
    evidence = policy.translation_guard.egress_evidence
    assert evidence.grants == 0 and evidence.spent_m == pytest.approx(plant.travel)
    assert row['controlled_reference_witness']['causal_supported']


def test_recorded_real_static_reference_exposes_board_at_refusal_before_another_action():
    path = Path(__file__).resolve().parents[1]/'runs/visual_field_v23_design_001/controller_diagnostic_003'
    if not path.exists():
        pytest.skip('Optional real controller diagnostic unavailable')
    rows = list(map(json.loads, (path/'trace.jsonl').read_text().splitlines()))
    initial = PolicyObservation.from_dict(next(r['observation'] for r in rows if r['event']=='reset'))
    after = {r['step']: PolicyObservation.from_dict(r['observation']) for r in rows if r['event']=='executed'}
    pred = json.loads((path/'archived_prediction.json').read_text())
    class CurrentHoldClient(OfferClient):
        def predict(self, observation):
            from gpt_field.action import _rotvec_quaternion
            value = super().predict(observation)
            value['field_intent']['orientation_target_xyzw'] = _rotvec_quaternion(
                np.asarray(observation['observation/state'][3:6])).tolist()
            return value
    policy = VisualFieldPolicyV24(CurrentHoldClient(pred))
    before = initial
    for step in range(9):
        decision = policy.act(before)
        assert decision.diagnostics['selection']['accepted'] is (step > 0)
        policy.observe_transition(before, decision, after[step])
        before = after[step]
    debt = policy.translation_guard.egress_evidence.spent_m
    assert not policy.reference_candidates.summary()['candidates']
    decision = policy.act(before)
    assert not decision.diagnostics['selection']['accepted']
    report = decision.diagnostics['feedback']['reference_repair']
    assert report['review_required'] and report['candidates']
    assert len(report['candidates']) <= 6
    assert not decision.diagnostics['feedback']['controlled_reference_witness']['causal_supported']
    assert policy.translation_guard.egress_evidence.spent_m == debt
    assert policy.reference_candidates.current.state == pytest.approx(before.state)
    assert policy.client.calls == 2

