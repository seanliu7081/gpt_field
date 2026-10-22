"""Rejected commands and pure previews preserve their actual proposal evidence."""
import copy
import numpy as np

from gpt_field.visual_translation_guard_v17 import ProjectedTranslationGuardV17
from test_visual_translation_guard_v16 import scene, refresh
from test_visual_translation_guard_v11 import plant


def test_refusal_retains_proposed_translation_without_claiming_execution():
    _, monitor, state = plant()
    guard = ProjectedTranslationGuardV17()
    refresh(guard, monitor, state, scene(mode='interaction'))
    delta = np.array([.002, 0., 0.])
    result = guard.assess(delta, monitor=monitor, direct_observation=True)
    assert not result['accepted']
    proposal = result['diagnostics']['checked_proposal']
    assert proposal['delta_xyz_m'] == delta.tolist()
    assert not proposal['executed']
    assert proposal['translation_mode'] == 'interaction'


def test_preview_records_own_proposal_without_mutating_actual_guard():
    _, monitor, state = plant()
    guard = ProjectedTranslationGuardV17()
    refresh(guard, monitor, state, scene())
    before = copy.deepcopy(guard.summary())
    result = guard.preview_assess([0., 0., .002], monitor=monitor,
        nominal_clear=True, nominal_away=True,
        prospective_translation_mode='separate')
    assert result['diagnostics']['checked_proposal']['delta_xyz_m'] == [0., 0., .002]
    assert result['diagnostics']['checked_proposal']['translation_mode'] == 'separate'
    assert guard.summary() == before
