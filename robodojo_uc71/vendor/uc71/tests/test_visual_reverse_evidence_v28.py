"""Comparable reverse outlines preserve uncertainty, latches and actual debt."""
import copy
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_reverse_evidence_v28 import ReverseBoundaryEvidenceV28
from gpt_field.visual_translation_guard_v26 import ProjectedTranslationGuardV26
from gpt_field.visual_translation_guard_v27 import ProjectedTranslationGuardV27
from test_visual_boundary_evidence_v27 import corners, track, move, snapshot


class EvidenceHarness(ReverseBoundaryEvidenceV28, ProjectedTranslationGuardV26):
    def _usable_snapshot(self):
        return self.current


def setup(spread=40., travel=.002, old_travel=0., worse=False):
    guard = EvidenceHarness()
    tool = track(spread=spread)
    neighbor = track(center=(200., 50.), identity='neighbor')
    guard.neighbor_measurement_epochs = {'neighbor': 4}
    guard.worsening = {'neighbor'}
    def capture(epoch=4):
        toolrow = snapshot(tool, epoch=epoch)
        neighborrow = snapshot(neighbor, corners(150, 0, 250, 100), epoch=epoch)
        bounds = np.asarray(toolrow.get('nominal_bounds_px', [0, 0, 100, 100]))
        return {'neighbors': {'neighbor': {'measurement_epoch': epoch,
            'gap': 150.-float(bounds[2]), 'noise': .1,
            'tool_boundary': toolrow, 'neighbor_boundary': neighborrow}}}
    baseline = capture()
    guard.current = baseline
    guard.retreat_progress = {'episode_ids': ['neighbor'],
        'measured_travel_m': old_travel+travel, 'acknowledged_commands': 3,
        'baseline': copy.deepcopy(baseline), 'further_worsening': worse,
        'all_trigger_gaps_improved': False, 'per_neighbor': {}}
    guard._reverse_measurement_segments = {'neighbor': guard._segment('neighbor', baseline, old_travel)}
    # Independent physical rows are deliberately nonzero: evidence updates may
    # not spend, refund, erase or recreate them.
    guard.observation_travel_m = .017
    guard.experiments = [{'measured_travel_m': .008, 'direction': [0, 1, 0]}]
    return guard, tool, neighbor, capture


def update(guard, capture):
    guard.current = capture()
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], guard.current)
    return guard.retreat_progress['per_neighbor']['neighbor']


@pytest.mark.parametrize('shift,classification,latch,improved',
    [(4., 'closing', True, False), (-4., 'opening', False, True)])
def test_well_spread_observed_reverse_resolves_closing_or_opening(shift, classification, latch, improved):
    guard, tool, _, capture = setup()
    move(tool, (shift, 0))
    row = update(guard, capture)
    assert row['boundary_evidence']['classification'] == classification
    assert guard.retreat_progress['further_worsening'] is latch
    assert guard.retreat_progress['all_trigger_gaps_improved'] is improved
    assert guard.reverse_boundary_qualification()['qualified'] is (not latch)
    assert guard.observation_travel_m == .017
    assert guard.experiments == [{'measured_travel_m': .008, 'direction': [0, 1, 0]}]
    assert not row['is_clearance_evidence']


def test_local_support_outline_motion_is_inconclusive_not_opening_or_closing():
    guard, tool, _, capture = setup(spread=2.)
    move(tool, (4., 0))
    row = update(guard, capture)
    assert row['boundary_comparable']
    assert row['boundary_evidence']['classification'] == 'inconclusive'
    assert not guard.retreat_progress['further_worsening']
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.reverse_boundary_qualification()['qualified']
    assert guard.reverse_boundary_qualification()['physical_allowance_added_m'] == 0.


def test_minimum_actual_segment_travel_not_command_or_lifetime_travel():
    guard, tool, _, capture = setup(travel=.0002, old_travel=.006)
    move(tool, (4., 0))
    row = update(guard, capture)
    assert row['boundary_evidence']['classification'] == 'closing'
    assert row['segment_measured_travel_m'] == pytest.approx(.0002)
    assert not guard.retreat_progress['further_worsening']
    guard.retreat_progress['measured_travel_m'] += .001
    update(guard, capture)
    assert guard.retreat_progress['further_worsening']


@pytest.mark.parametrize('case', ['lost', 'epoch', 'new_track', 'changed_extent'])
def test_unavailable_current_boundary_refuses_qualification_without_positive_credit(case):
    guard, tool, _, capture = setup()
    move(tool, (-4., 0))
    current = capture()
    if case == 'lost':
        tool.reliable = False
        current = capture()
    elif case == 'epoch':
        current = capture(epoch=5)
    elif case == 'new_track':
        current['neighbors']['neighbor']['tool_boundary'] = snapshot(copy.deepcopy(tool), epoch=4)
    else:
        current['neighbors']['neighbor']['tool_boundary'] = snapshot(tool, corners(0, 0, 110, 100), epoch=4)
    guard.current = current
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], current)
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    result = guard.reverse_boundary_qualification()
    assert not result['qualified']
    assert result['reason'] == 'translation_reverse_boundary_evidence_unavailable'
    assert guard.retreat_progress['measured_travel_m'] == .002


@pytest.mark.parametrize('current', ['inconclusive', 'opening', 'missing'])
def test_existing_resolved_adverse_latch_cannot_be_cleared(current):
    guard, tool, _, capture = setup(spread=2. if current == 'inconclusive' else 40., worse=True)
    move(tool, (-4., 0))
    guard.current = None if current == 'missing' else capture()
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], guard.current)
    assert guard.retreat_progress['further_worsening']
    assert not guard.reverse_boundary_qualification()['qualified']


def test_new_epoch_cannot_borrow_past_motion_and_qualification_is_pure():
    guard, tool, _, capture = setup(old_travel=.006)
    guard.neighbor_measurement_epochs['neighbor'] = 5
    move(tool, (-4., 0))
    guard.current = capture(epoch=5)
    guard._update_reverse_measurements(guard.retreat_progress['baseline'], guard.current)
    row = guard.retreat_progress['per_neighbor']['neighbor']
    assert row['segment_measured_travel_m'] == 0.
    assert not row['improved']
    assert not guard.retreat_progress['further_worsening']
    before = pickle.dumps(guard)
    assert guard.reverse_boundary_qualification()['qualified']
    assert pickle.dumps(guard) == before


def test_common_camera_flow_does_not_become_gap_progress():
    guard, tool, neighbor, capture = setup()
    move(tool, (5., -3.));move(neighbor, (5., -3.))
    row = update(guard, capture)
    assert row['boundary_evidence']['classification'] == 'inconclusive'
    assert row['gap_change_px'] == pytest.approx(0.)
    assert not row['improved'] and not guard.retreat_progress['further_worsening']


def test_first_inherited_scalar_result_is_reclassified_only_in_synchronous_creation():
    class LegacyFirstObserver:
        def observe(self, *unused, **kwargs):
            if self.retreat_progress is None:
                self.retreat_progress = copy.deepcopy(self.to_create)
                # Emulate V21's first-action scalar decision before the V26
                # replacement hook; the row itself is actual boundary data.
                self.retreat_progress['further_worsening'] = True
            self._update_reverse_measurements(self.retreat_progress['baseline'], self.current)
    class FirstGuard(ReverseBoundaryEvidenceV28, LegacyFirstObserver):
        _segment = staticmethod(ProjectedTranslationGuardV26._segment)
    ready, tool, _, capture = setup(spread=2.)
    move(tool, (4., 0))
    g = FirstGuard(); g.__dict__.update(copy.deepcopy(ready.__dict__))
    g.current = capture();g.to_create = g.retreat_progress;g.retreat_progress = None
    g.observe()
    assert not g.retreat_progress['further_worsening']
    assert not hasattr(g, '_v28_first_reverse_observation')
    g.retreat_progress['further_worsening'] = True
    g.observe()
    assert g.retreat_progress['further_worsening']


def test_first_observation_resolved_closing_still_latches():
    guard, tool, _, capture = setup()
    guard._v28_first_reverse_observation = {'prior_progress': None, 'reclassified': False}
    guard.retreat_progress['further_worsening'] = True
    move(tool, (4., 0))
    update(guard, capture)
    assert guard.retreat_progress['further_worsening']
    assert guard._v28_first_reverse_observation['reclassified']


def replay_retry002():
    """Saved RGB/robot observations only; no simulator or model invocation."""
    root = Path(__file__).parents[1]
    path = root/'runs/visual_field_v27_zero_shot_002/trials/bowl_original_zero_hybrid_v27/trace.jsonl'
    if not path.exists():
        pytest.skip('Actual retry002 archive unavailable')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    initial = next(r['observation'] for r in rows if r['event'] == 'reset')
    actions = {r['step']: r for r in rows if r['event'] == 'action'}
    after = {r['step']: r['observation'] for r in rows if r['event'] == 'executed'}
    feedback = {r['step']: r['feedback'] for r in rows if r['event'] == 'feedback'}
    helper = root/'runs/visual_field_v27_design_001/archived_approach_inputs.py'
    spec = importlib.util.spec_from_file_location('reverse_v28_archive_input', helper)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records = [{'archive_step':i, 'observation':initial if i == 0 else after[i-1],
                'prediction':copy.deepcopy(actions[i]['diagnostics']['prediction'])}
               for i in range(33) if actions[i]['diagnostics'].get('prediction') is not None]
    from gpt_field.observation import PolicyObservation
    from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
    class Guard(ReverseBoundaryEvidenceV28, ProjectedTranslationGuardV27): pass
    class Policy(VisualFieldPolicyV27):
        def reset(self):
            super().reset()
            self.translation_guard.__class__ = Guard
    policy = Policy(module.ArchivedPrefixClient(records))
    before = PolicyObservation.from_dict(initial)
    snapshots = {}
    for i in range(33):
        decision = policy.act(before)
        np.testing.assert_allclose(decision.action, actions[i]['action'], atol=1e-10, rtol=0.)
        if i == 29:
            snapshots['before29'] = copy.deepcopy(policy.translation_guard._usable_snapshot())
        current = PolicyObservation.from_dict(after[i])
        policy.observe_transition(before, decision, current)
        if i >= 29:
            snapshots[f'after{i}'] = copy.deepcopy(policy.translation_guard._usable_snapshot())
        before = current
    return policy, before, feedback, snapshots


def test_actual_retry002_epoch4_reverse_is_inconclusive_with_identical_prefix_and_debt():
    policy, before, old, snapshots = replay_retry002()
    guard = policy.translation_guard
    progress = guard.retreat_progress
    prior = old[32]['translation_guard']['directional_retreat']['progress']
    assert prior['further_worsening']
    assert not progress['further_worsening']
    assert not progress['all_trigger_gaps_improved']
    assert progress['measured_travel_m'] == pytest.approx(prior['measured_travel_m'], abs=1e-12)
    assert guard.observation_travel_m == pytest.approx(old[32]['translation_guard']['observation_travel_m'], abs=1e-12)
    identity = prior['episode_ids'][0]
    row = progress['per_neighbor'][identity]
    assert row['measurement_epoch'] == 4
    assert row['segment_measured_travel_m'] == pytest.approx(.00162368855, abs=1e-10)
    assert row['boundary_evidence']['classification'] == 'inconclusive'
    assert row['boundary_evidence']['nominal_gap_change_px'] == pytest.approx(-8.3347450034, abs=1e-8)
    assert row['boundary_evidence']['gap_change_interval_px'][0] < 0 < row['boundary_evidence']['gap_change_interval_px'][1]
    assert guard.reverse_boundary_qualification()['qualified']
    # Assess a prospective next command only. The saved old hold is NOT fed
    # back as the consequence of a changed command; no trajectory is invented.
    choice = policy.act(before)
    assert choice.diagnostics['selection']['accepted'], choice.diagnostics['selection']['status']
    assert np.linalg.norm(choice.action[:3]) > 0
