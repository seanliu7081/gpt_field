"""Whole-policy saved-RGB admission and acknowledged-only physical accounting."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation
from gpt_field.visual_field_policy import images
from gpt_field.visual_field_policy_v28 import VisualFieldPolicyV28
from gpt_field.visual_translation_guard_v27 import ProjectedTranslationGuardV27
from gpt_field.visual_translation_guard_v28 import ProjectedTranslationGuardV28


def archive_policy():
    root = Path(__file__).parents[1]
    helper = root/'runs/visual_field_v28_design_001/archived_heading_inputs.py'
    if not helper.exists(): pytest.skip('Archived-heading input helper unavailable')
    spec = importlib.util.spec_from_file_location('guard28_archived_heading', helper)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _, initial, decisions, actions, executed = module.replay_data()
    policy = VisualFieldPolicyV28(module.ArchivedPrefixClient(decisions))
    obs = PolicyObservation.from_dict(initial)
    for step in range(14):
        decision = policy.act(obs)
        np.testing.assert_allclose(decision.action, actions[step], atol=1e-10, rtol=0.)
        after = PolicyObservation.from_dict(executed[step])
        policy.observe_transition(obs, decision, after)
        obs = after
    return policy, obs


def accounting(guard):
    return pickle.dumps({
        'global': guard.observation_travel_m, 'experiments': guard.experiments,
        'heading': guard.heading_acquisition.__dict__,
        'heading_pending': guard._heading_pending,
        'physical_pending': guard._motion_classification_pending,
        'experiment_pending': guard._experiment_pending,
        'retreat_pending': guard._retreat_pending,
        'last_heading': guard.last_heading_acquisition,
        'worsening': guard.worsening, 'progress': guard.retreat_progress})


def test_actual_fourteen_transition_prefix_admits_bounded_new_heading_and_preserves_all_gates():
    policy, obs = archive_policy()
    guard, field = policy.translation_guard, policy.local
    assert policy.client.calls == 1
    assert guard.worsening
    before_debt = guard.observation_travel_m
    selected = policy.act(obs)
    assert selected.diagnostics['selection']['accepted'], selected.diagnostics['selection']['status']
    delta = np.asarray(selected.action[:3])*.05
    assert 0. < np.linalg.norm(delta) <= .003+1e-12
    assert guard.last_heading_acquisition['accepted']
    assert guard.last_heading_acquisition['pending'] is not None
    assert guard.heading_acquisition.acknowledged_actions == 0
    assert guard.observation_travel_m == before_debt
    assert policy.client.calls == 1

    records = field.nominal_regions_for_segment(np.asarray(obs.state[:3]),
        np.asarray(obs.state[:3])+delta)
    options = dict(monitor=policy.monitor, nominal_clear=True, nominal_away=False,
        nominal_regions=records, translation_response=policy.robot_response,
        rotation_rad=float(np.linalg.norm(np.asarray(selected.action[3:6])*.5)),
        orientation_hold=True)
    # Prospective checks cannot overwrite the action which is already pending,
    # its physical classification, estimator history or the retained track.
    before = accounting(guard); dictionary = guard.__dict__; track = guard.tool_track
    preview = guard.preview_assess(delta, **options)
    assert preview['accepted'], preview['reason']
    assert guard.__dict__ is dictionary and guard.tool_track is track
    assert accounting(guard) == before

    # Each negative changes only an independently copied candidate proof. No
    # geometry, unknown, target or role failure can obtain cold-heading credit.
    faults = ['state', 'endpoint', 'epoch', 'unknown', 'contact', 'payload',
              'scene', 'fingerprint', 'crossed_halo', 'decreasing_gap']
    for fault in faults:
        bad = copy.deepcopy(records)
        proof = bad[0]['transit_reconciliation_contract']
        identity = next(iter(guard.worsening))
        if fault == 'state': proof['state8'][0] += .001
        elif fault == 'endpoint': proof['end_xyz_m'][0] += .001
        elif fault == 'epoch': proof['observation_epoch'] -= 1
        elif fault == 'unknown': proof['hard_segment_clear'] = False
        elif fault == 'contact': proof['target_contact'] = 'fingers'
        elif fault == 'payload': proof['empty_tool'] = False
        elif fault == 'scene': proof['full_scene_ids'] = []
        elif fault == 'fingerprint': proof['geometry_sha256'] = 'different-world-hypothesis'
        elif fault == 'crossed_halo':
            proof['object_proofs'][identity]['full_uncertainty_sweep_disjoint'] = False
        else: proof['object_proofs'][identity]['gap_nondecreasing'] = False
        rejected = guard.preview_assess(delta, **{**options, 'nominal_regions':bad})
        assert not rejected['accepted'], (fault, rejected['reason'])
        assert accounting(guard) == before

    # A wrong acknowledgement is rejected by the actual pending executor.
    wrong = list(selected.action);wrong[0] += .001
    with pytest.raises(ValueError): field.observe(obs.state, wrong, obs.state)
    assert accounting(guard) == before

    # Accounting-only synthetic endpoint after the exact accepted command;
    # this is NOT replayed robot motion or evidence of task/collision success.
    data = obs.to_dict();after_state = np.asarray(obs.state).copy()
    after_state[:3] += delta*.2
    data['observation/state'] = after_state.tolist()
    after = PolicyObservation.from_dict(data)
    policy.observe_transition(obs, selected, after)
    measured = float(np.linalg.norm(after_state[:3]-np.asarray(obs.state[:3])))
    assert guard.observation_travel_m == pytest.approx(before_debt+measured, abs=1e-12)
    assert guard.heading_acquisition.acknowledged_actions == 1
    assert guard._heading_pending is None
    assert len(guard.experiments) == 1
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(measured, abs=1e-12)
    assert field.pending is None
    rows_before = copy.deepcopy(guard.heading_acquisition.rows)

    # Same-observation context refresh supplies no refund or second action.
    guard.configure(images(after), guard.context, after.state,
        target_id=guard.target_id, monitor=policy.monitor)
    assert guard.heading_acquisition.rows == rows_before
    assert guard.observation_travel_m == pytest.approx(before_debt+measured, abs=1e-12)

    # Explicit zero-command residual motion is also charged to the same row.
    held_state = after_state.copy();held_state[:3] += delta*.05
    policy.monitor.observe(images(after), held_state, [0.]*7)
    guard.observe(images(after), held_state, [0.]*7, monitor=policy.monitor)
    residual = float(np.linalg.norm(held_state[:3]-after_state[:3]))
    assert guard.observation_travel_m == pytest.approx(before_debt+measured+residual, abs=1e-12)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(measured+residual, abs=1e-12)
    assert guard.heading_acquisition.acknowledged_actions == 1
    assert guard.heading_acquisition.residual_hold_transitions == 1


@pytest.mark.parametrize('qualified,reason', [
    (False, 'translation_reverse_boundary_evidence_unavailable'),
    (False, 'translation_reverse_observed_further_worsening'),
    (True, 'current_comparable_reverse_boundary')])
def test_actual_guard_reverse_branch_obeys_comparable_qualification(monkeypatch, qualified, reason):
    # Isolate the new root wiring after an already accepted inherited proposal.
    # Real shape/evidence computation is covered by test_visual_reverse_evidence_v28.
    guard = ProjectedTranslationGuardV28();guard.worsening = {'neighbor'}
    monkeypatch.setattr(ProjectedTranslationGuardV27, 'assess', lambda *a, **k:
        {'accepted':True,'reason':'inherited_checked_reverse','cap_m':.001,'diagnostics':{}})
    monkeypatch.setattr(guard, '_finite_reverse_contract', lambda *a: {'accepted':True})
    monkeypatch.setattr(guard, 'reverse_boundary_qualification', lambda:
        {'qualified':qualified,'reason':reason,'available':qualified,
         'current_closing':False,'prior_latch':reason.endswith('worsening')})
    result = guard.assess([.001,0,0], nominal_regions=[])
    assert result['accepted'] is qualified
    if not qualified:
        assert result['reason'] == reason
        assert guard._heading_pending is None and guard._experiment_pending is None
        assert guard._retreat_pending is None
    assert guard.observation_travel_m == 0. and guard.experiments == []


@pytest.mark.parametrize('cap', [.001, .002])
def test_cap_tolerance_mismatch_forces_reselection_without_pending_credit(monkeypatch, cap):
    from types import SimpleNamespace
    from test_visual_field_v4 import state
    guard = ProjectedTranslationGuardV28();guard.state = state()
    guard.context = {'translation_mode':'transit'};guard.worsening = {'neighbor'}
    norm = cap+5e-11
    command = np.array([-norm, 0., 0.])
    contract = {'remaining_measured_travel_m':.02, 'remaining_actions':20}
    monkeypatch.setattr(guard, '_transit_contract', lambda *a:contract)
    monkeypatch.setattr(guard, '_finite_reverse_contract', lambda *a:None)
    monkeypatch.setattr(guard, '_usable_snapshot', lambda:{'neighbors':{
        'neighbor':{'response_representation':('tool','neighbor')}}})
    monkeypatch.setattr(ProjectedTranslationGuardV27, 'assess', lambda *a, **k:
        {'accepted':False,'reason':'translation_projected_conflict_image_response_unavailable',
         'cap_m':0.,'diagnostics':{}})
    monkeypatch.setattr(guard.heading_acquisition, 'assess', lambda *a, **k:
        {'accepted':False,'reason':'translation_heading_evidence_candidate_requires_cap_reselection',
         'command_cap_m':cap, 'pending':None})
    response = SimpleNamespace(predict=lambda *a:{'supported':False})
    result = guard.assess(command, nominal_regions=[],
        monitor=SimpleNamespace(intent_id=1), translation_response=response)
    assert result['accepted']  # A cap proposal, not an executable admission.
    assert result['cap_m'] < norm-1e-10
    assert guard._heading_pending is None
    assert guard._motion_classification_pending is None
    assert guard._experiment_pending is None
    assert guard.observation_travel_m == 0. and guard.experiments == []
    # Exact floor conservatively hands off: no rounding up or unaccounted
    # command. Above floor the original selector must retry its exact proposal.
    assert (result['cap_m'] < .001) is (cap == .001)


def test_actual_executor_reselects_above_floor_but_refuses_floor_tail(monkeypatch):
    from test_visual_field_v27 import upgraded
    from test_visual_field_v4 import Q
    for cap in (.001, .002):
        field, guard, monitor, state8 = upgraded()
        calls = []
        def admission(delta, **kwargs):
            size = float(np.linalg.norm(delta));calls.append(size)
            if len(calls) == 1:
                return {'accepted':True,'cap_m':cap-2e-10,'reason':'requires_exact_reselection',
                        'diagnostics':{}}
            return {'accepted':True,'cap_m':size,'reason':'exact_smaller_candidate_checked',
                    'diagnostics':{}}
        monkeypatch.setattr(guard, 'assess', admission)
        selected = field.select(state8, Q, command_cap_m=cap+5e-11)
        assert calls[0] == pytest.approx(cap+5e-11, abs=1e-12)
        if cap == .001:
            assert not selected['accepted']
            assert selected['status'] == 'translation_allowance_has_no_useful_supported_command'
            assert field.pending is None and len(calls) == 1
        else:
            assert selected['accepted'] and len(calls) == 2
            assert calls[1] == pytest.approx(cap-2e-10, abs=1e-12)
            assert np.linalg.norm(selected['action'][:3])*.05 == pytest.approx(calls[1], abs=1e-12)
        assert field.actions == 0 and field.measured_travel_m == 0.
