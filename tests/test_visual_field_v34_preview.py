"""Actual saved Direct-to-proxy handoff: prospective versus live admission."""
import copy
import importlib.util
from pathlib import Path
import sys
import json

import numpy as np
import pytest

from gpt_field.visual_field_v33 import MetricFieldExecutorV33
from gpt_field.visual_field_v34 import MetricFieldExecutorV34
from gpt_field.visual_translation_guard_v34 import ProjectedTranslationGuardV34
from gpt_field.visual_field_policy_v15 import recovery_options
from gpt_field.observation import PolicyObservation


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'runs/visual_field_v34_design_001'


@pytest.fixture(scope='module')
def reconstructed():
    trial = ROOT/'runs/visual_field_v34_zero_shot_001/trials/bowl_original_zero_hybrid_v34'
    if not (trial/'trace.jsonl').exists():
        pytest.skip('Optional full saved-state regression needs local trial artifacts')
    spec = importlib.util.spec_from_file_location('v34_preview_test_inputs',
        DESIGN/'proxy_preview_counterfactual.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Boundary observations contain process-local track identity tokens. Replay
    # the frozen policy here instead of treating a deserialized token as proof
    # that a new object is the same observed reference.
    frozen_spec = importlib.util.spec_from_file_location('gpt_field.v34_frozen_test_policy',
        trial/'frozen_source/visual_field_policy_v34.py')
    frozen = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen)
    rows = [json.loads(line) for line in (trial/'trace.jsonl').read_text().splitlines()]
    actions, executed, decisions = {}, {}, []
    for row in rows:
        if row['event'] == 'reset':
            initial = before = row['observation']
        elif row['event'] == 'action':
            actions[row['step']] = row
            if row['diagnostics'].get('prediction') is not None:
                decisions.append(dict(archive_step=row['step'], observation=copy.deepcopy(before),
                    prediction=copy.deepcopy(row['diagnostics']['prediction'])))
        elif row['event'] == 'executed':
            before = executed[row['step']] = row['observation']
    policy = frozen.VisualFieldPolicyV34(module.common.ArchivedPrefixClient(decisions))
    obs = PolicyObservation.from_dict(initial)
    for step in range(52):
        decision = policy.act(obs)
        assert np.allclose(decision.action, actions[step]['action'], atol=1e-10, rtol=0.)
        old = actions[step]['diagnostics'].get('selection') or {}
        current = decision.diagnostics.get('selection') or {}
        assert (old.get('accepted'), old.get('status')) == (current.get('accepted'), current.get('status'))
        after = PolicyObservation.from_dict(executed[step])
        policy.observe_transition(obs, decision, after)
        obs = after
    return policy, obs, module.snapshot


@pytest.fixture
def archived(reconstructed):
    base, obs, snapshot = reconstructed
    tracks = [t for t in base.monitor.tracks.values() if t is not None]
    tracks += [v for row in base.translation_guard.neighbors for v in row.values()
               if hasattr(v, 'position') and hasattr(v, 'reliable')]
    reliability = [(track, track.reliable) for track in tracks]
    # Preserve the actual identity of measured RGB tracks, not only their name.
    # Every mutable executor/guard/control ledger is independently copied.
    policy = copy.deepcopy(base, {id(track): track for track in tracks})
    policy.inventory.revision_context_provider = lambda: (policy.prediction or {}).get('scene_revisions', [])
    policy.local.__class__ = MetricFieldExecutorV34
    policy.translation_guard.__class__ = ProjectedTranslationGuardV34
    offset = np.asarray(policy.translation_guard.recovery_directions()[0]['direction_xyz'])*.008
    try:
        yield policy, obs, offset, snapshot
    finally:
        for track, reliable in reliability:
            track.reliable = reliable


def preview(policy, obs, offset):
    return policy.local.preview_recovery(obs.state, obs.joint_position, offset,
        goal_tolerance_m=.003, command_limit_m=.002, max_travel_m=.008,
        scene_assessor=policy._recovery_scene_assess)


def test_actual_reference_handoff_offers_field_reverse_but_grants_no_live_permission(archived):
    policy, obs, offset, snapshot = archived
    guard, local = policy.translation_guard, policy.local
    original = snapshot(policy)
    guard_dict, local_dict, role = guard.__dict__, local.__dict__, guard.reference_role_context
    assert role['controller'] == 'direct'
    selected = preview(policy, obs, offset)
    assert selected['accepted'], selected['status']
    assert selected['status'] == 'metric_field_tracking'
    assert np.linalg.norm(np.asarray(selected['action'][:3])*.05) == pytest.approx(.002)
    assert selected['requires_explicit_model_choice']
    assert not selected['physical_debt_replenished']
    assert snapshot(policy) == original
    assert guard.__dict__ is guard_dict and local.__dict__ is local_dict
    assert guard.reference_role_context is role
    assert guard._proxy_role_reason(policy.monitor, True) == 'translation_proxy_role_or_contact_ineligible'


def test_ordinary_axis_recovery_also_checks_prospective_role_and_restores_context(archived):
    policy, obs, _, snapshot = archived
    original = copy.deepcopy(policy.translation_guard.summary())
    result = recovery_options(policy.local, policy.translation_guard, obs.state,
        monitor=policy.monitor, robot_response=policy.robot_response)
    assert result['candidates'] or result['rejected_candidates']
    assert all(row.get('reason') != 'translation_proxy_role_or_contact_ineligible'
               for row in result['rejected_candidates'])
    assert policy.translation_guard.summary() == original


@pytest.mark.parametrize('fault', ['stale_state', 'target', 'contact', 'payload',
    'alignment', 'attachment', 'lost_hand', 'rotation', 'unknown_region', 'pending'])
def test_real_geometry_reference_contact_and_acknowledgement_checks_remain(archived, fault):
    policy, obs, offset, snapshot = archived
    guard = policy.translation_guard
    if fault == 'stale_state':
        from dataclasses import replace
        state = list(obs.state); state[0] += .001
        obs = replace(obs, state=tuple(state))
    elif fault == 'target':
        policy.monitor.refs['external']['target_id'] = 'different_target'
    elif fault == 'contact':
        guard.reference_role_context['target_contact'] = 'fingers'
    elif fault == 'payload':
        policy.monitor.refs['external']['moving_kind'] = 'payload'
    elif fault == 'alignment':
        guard.reference_role_context['contact_alignment'] = {'target_id': guard.target_id}
    elif fault == 'attachment':
        policy.monitor.attachment_evidence = True
    elif fault == 'lost_hand':
        policy.monitor.tracks['tool'].reliable = False
    elif fault == 'rotation':
        # Current robot state stays bound; change the retained proxy seed's
        # pose domain so current rotation no longer supports that reference.
        policy.monitor._proxy_origin[3] += .2
    elif fault == 'unknown_region':
        center = np.asarray(obs.state[:3])-policy.local.origin
        policy.local.intent['unknown_regions'].append({'id': 'newly_declared_unknown',
            'lower_offset_m': (center-.02).tolist(),
            'upper_offset_m': (center+.02).tolist(), 'uncertainty_m': .01})
    else:
        policy.local.pending = {'unacknowledged': True}
    before = snapshot(policy)
    role = guard.reference_role_context
    result = preview(policy, obs, offset)
    assert not result['accepted'], (fault, result['status'])
    assert snapshot(policy) == before
    assert guard.reference_role_context is role


def test_exception_restores_original_role_and_runtime_objects(archived, monkeypatch):
    policy, obs, offset, snapshot = archived
    original = snapshot(policy)
    guard_dict, local_dict = policy.translation_guard.__dict__, policy.local.__dict__
    role = policy.translation_guard.reference_role_context
    def failing(self, *args, **kwargs):
        assert self.translation_guard.reference_role_context['controller'] == 'visual_field'
        assert self.__dict__ is not local_dict
        assert self.translation_guard.__dict__ is not guard_dict
        assert self.translation_guard.context['translation_mode'] == 'separate'
        raise RuntimeError('deliberate prospective failure')
    # Fail after configure, inside both inherited dictionary transactions, to
    # verify inner executor/guard restoration as well as the outer role binding.
    monkeypatch.setattr(MetricFieldExecutorV33, 'select', failing)
    with pytest.raises(RuntimeError, match='deliberate prospective failure'):
        preview(policy, obs, offset)
    assert snapshot(policy) == original
    assert policy.translation_guard.__dict__ is guard_dict
    assert policy.local.__dict__ is local_dict
    assert policy.translation_guard.reference_role_context is role
