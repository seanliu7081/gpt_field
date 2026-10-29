"""Source continuity at the saved retry002 handoff, without new robot steps."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation
from gpt_field.visual_field_policy import images
from gpt_field.visual_field_proposals_v35 import FieldProposalBankV35
from gpt_field.visual_translation_guard_v11 import ProjectedTranslationGuardV11
from gpt_field.visual_translation_guard_v34 import ProjectedTranslationGuardV34
from gpt_field.visual_translation_guard_v35 import ProjectedTranslationGuardV35


ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT/'runs/visual_field_v34_zero_shot_002/trials/bowl_original_zero_hybrid_v34'


@pytest.fixture(scope='module')
def saved_prefix():
    if not (TRIAL/'trace.jsonl').exists():
        pytest.skip('Optional saved trial regression requires retry002 artifacts')
    spec = importlib.util.spec_from_file_location('v35_saved_inputs',
        ROOT/'runs/visual_field_v30_design_001/archived_approach_inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location('gpt_field.v35_saved_policy',
        TRIAL/'frozen_source/visual_field_policy_v34.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    actions, executed, decisions = {}, {}, []
    for row in map(json.loads, (TRIAL/'trace.jsonl').read_text().splitlines()):
        if row['event'] == 'reset':
            initial = before = row['observation']
        elif row['event'] == 'action':
            actions[row['step']] = row
            if row['diagnostics']['refreshed']:
                decisions.append(dict(archive_step=row['step'], observation=copy.deepcopy(before),
                    prediction=copy.deepcopy(row['diagnostics']['prediction'])))
        elif row['event'] == 'executed':
            before = executed[row['step']] = row['observation']
    policy = frozen.VisualFieldPolicyV34(module.ArchivedPrefixClient(decisions))
    obs = PolicyObservation.from_dict(initial)
    for step in range(52):
        decision = policy.act(obs)
        assert np.allclose(decision.action, actions[step]['action'], atol=1e-10, rtol=0.)
        actual, prior = decision.diagnostics.get('selection') or {}, actions[step]['diagnostics'].get('selection') or {}
        assert (actual.get('accepted'), actual.get('status')) == (prior.get('accepted'), prior.get('status'))
        after = PolicyObservation.from_dict(executed[step])
        policy.observe_transition(obs, decision, after)
        obs = after
    return policy, obs, actions, executed


@pytest.fixture
def archived(saved_prefix):
    base, obs, actions, executed = saved_prefix
    tracks = list({id(t): t for t in list(base.monitor.tracks.values()) +
        [n['track'] for n in base.translation_guard.neighbors] if t is not None}.values())
    # Keep process-local observation identity tokens valid; restore every track
    # afterwards, including the test that replays later archived observations.
    saved_tracks = [(track, copy.deepcopy(track.__dict__)) for track in tracks]
    policy = copy.deepcopy(base, {id(track): track for track in tracks})
    policy.inventory.revision_context_provider = lambda: (policy.prediction or {}).get('scene_revisions', [])
    policy.translation_guard.__class__ = ProjectedTranslationGuardV35
    try:
        yield policy, obs, actions, executed
    finally:
        for track, state in saved_tracks:
            track.__dict__.clear()
            track.__dict__.update(state)


def persistent(guard):
    return copy.deepcopy({key: getattr(guard, key) for key in (
        'observation_travel_m', 'experiments', 'worsening', 'improving',
        'worsening_motion', 'retreat_progress', 'boundary_closing_ids',
        'boundary_warning_history')})


def test_hook_precedes_scalar_and_boundary_snapshot_capture():
    order = ProjectedTranslationGuardV35.__mro__
    names = [cls.__name__ for cls in order]
    assert names.index('ObservedRiskTranslationGuardV12') < names.index('_RetainHandSourceAfterV11')
    assert names.index('_RetainHandSourceAfterV11') < order.index(ProjectedTranslationGuardV11)


def test_actual_configure52_keeps_original_outline_and_current_cabinet_boundary(archived):
    policy, obs, actions, _ = archived
    guard = policy.translation_guard
    source, box = guard.tool_source.copy(), guard.tool_box.copy()
    old = persistent(guard)
    decision = policy.act(obs)
    assert np.allclose(decision.action, actions[52]['action'], atol=1e-10, rtol=0.)
    assert guard.source_continuity['retained']
    assert np.array_equal(guard.tool_source, source)
    assert np.allclose(guard.tool_box, box, atol=1e-7, rtol=0.)
    assert guard.boundary_evidence['cabinet']['comparable']
    assert guard.neighbor_measurement_epochs['cabinet'] == 7
    assert persistent(guard) == old
    assert not guard.boundary_evidence['cabinet']['clearance_certified']


@pytest.mark.parametrize('fault', ['roi', 'track', 'state', 'rotation', 'gripper',
    'monitor_state', 'image', 'unreliable', 'invalid_domain', 'target'])
def test_changed_or_unbound_context_does_not_reuse_original_source(archived, fault):
    policy, obs, _, _ = archived
    guard = policy.translation_guard
    context, state, current_images = copy.deepcopy(guard.context), np.array(obs.state), images(obs)
    target = guard.target_id
    old = persistent(guard)
    if fault == 'roi':
        context['tool_external_roi_uv'][2] += .02
    elif fault == 'track':
        # Actual hand selection in the current anatomical role.
        policy.monitor.tracks['moving'] = copy.deepcopy(guard.tool_track)
    elif fault == 'state':
        state[0] += .001
    elif fault == 'rotation':
        state[3] += .1
    elif fault == 'gripper':
        state[6] += .01
    elif fault == 'monitor_state':
        policy.monitor.state = np.array(obs.state); policy.monitor.state[1] += .001
    elif fault == 'image':
        current_images['external'] = current_images['external'].copy()
        current_images['external'][0, 0] ^= np.uint8(1)
    elif fault == 'unreliable':
        guard.tool_track.reliable = False
    elif fault == 'invalid_domain':
        guard.domain_invalid = 'translation_tool_configuration_changed'
    elif fault == 'target':
        target = 'different_target'
    guard.configure(current_images, context, state, target_id=target, monitor=policy.monitor)
    assert not guard.source_continuity['retained']
    assert persistent(guard) == old
    if fault in ('roi', 'track'):
        assert guard.measurement_epoch > 7


def test_legacy_refresh_still_exposes_the_original_defect(archived):
    policy, obs, _, _ = archived
    policy.translation_guard.__class__ = ProjectedTranslationGuardV34
    policy.act(obs)
    boundary = policy.translation_guard.boundary_evidence['cabinet']
    assert not boundary['comparable']
    assert boundary['reason'] == 'measurement_representation_changed'


def test_saved_continuation_preserves_scalar_warning_without_claiming_clearance(archived, tmp_path):
    policy, obs, actions, executed = archived
    for step in range(52, 60):
        decision = policy.act(obs)
        assert np.allclose(decision.action, actions[step]['action'], atol=1e-10, rtol=0.)
        after = PolicyObservation.from_dict(executed[step])
        policy.observe_transition(obs, decision, after)
        obs = after
    guard = policy.translation_guard
    assert 'cabinet' in guard.worsening
    assert guard.risk_windows['cabinet']['state'] == 'worsening'
    assert guard.boundary_evidence['cabinet']['comparable']
    assert not guard.boundary_evidence['cabinet']['clearance_certified']
    assert not guard.source_continuity['physical_debt_replenished']
    # A fresh prospective normal transit is a separate question. Keep every
    # current geometry, robot response, per-object response, boundary and debt
    # check; this diagnostic neither changes the live role nor executes motion.
    delta = np.array([0., .002, 0.])
    start = np.asarray(obs.state[:3])
    old_debt = persistent(guard)
    prospective = copy.copy(policy)
    prospective.prediction = copy.deepcopy(policy.prediction)
    prospective.prediction['phase'] = 'approach'
    offer, refusal = FieldProposalBankV35()._preview(
        prospective, obs, np.array([0., .008, 0.]), recovery=False)
    scene = policy._scene_assess(obs.state, delta)
    assert persistent(guard) == old_debt
    report = dict(saved_actions52_through59_match=True,
        scalar_warning=guard.risk_windows['cabinet'],
        boundary=guard.boundary_evidence['cabinet'],
        prospective_goal_offset_m=[0., .008, 0.],
        normal_candidate_offer=offer, normal_candidate_refusal=refusal,
        scene_delta_m=delta.tolist(), scene=scene,
        actual_actions_executed=0, model_calls=0, simulator_calls=0,
        interpretation='Unexecuted candidate only; admission is not a task outcome or clearance certificate.')
    (tmp_path/'normal_candidate.json').write_text(json.dumps(report, indent=2)+'\n')
