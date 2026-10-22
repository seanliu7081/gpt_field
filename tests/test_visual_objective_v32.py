import copy
from types import SimpleNamespace

from gpt_field.visual_objective_v32 import VisualGoalFieldV32, VisualTaskProgressV32, FieldProposalBankV32
from test_visual_goal_field_v31 import fixture, advance
from test_visual_field_v13_execution import observation


def test_same_anatomical_motion_can_still_produce_a_measured_field():
    _, monitor, prediction, state = fixture()
    prediction['visual_references']['external']['reference_role'] = 'tool_anatomy'
    field = VisualGoalFieldV32()
    field.configure(state, monitor, prediction)
    for _ in range(9):
        state = advance(field, monitor, prediction, state)
    assert field.propose(state)['accepted']


def test_reference_loss_removes_active_training_report_without_refunding_import():
    _, monitor, prediction, state = fixture()
    field = VisualGoalFieldV32()
    field.configure(state, monitor, prediction)
    key = ('candidate', 'digest', 31)
    field._v32_goal_replay_applied = {key}
    field._v32_goal_replay_report = {'accepted': True, 'reference_epoch': field.epoch}
    assert field.summary()['retrospective_training']['accepted']
    monitor.tracks['moving'].reliable = False
    field.configure(state, monitor, prediction)
    assert field.summary()['retrospective_training'] is None
    assert field._v32_goal_replay_applied == {key}
    assert not field.model.samples and not field.held_out


def test_a_responsive_proxy_cannot_be_mistaken_for_anatomical_goal_progress():
    _, monitor, prediction, state = fixture()
    prediction['visual_references']['external']['reference_role'] = 'rigid_transit_proxy'
    field = VisualGoalFieldV32()
    field.configure(state, monitor, prediction)
    progress = VisualTaskProgressV32()
    goal = state[:3] + [.03, 0, 0]
    progress.configure(state, monitor, prediction, goal, step=0)
    for _ in range(9):
        state = advance(field, monitor, prediction, state)
    assert not field.propose(state)['accepted']
    assert not field.model.samples and not field.held_out
    assert progress.active is None
    assert field.status == 'rigid_proxy_has_no_anatomical_image_goal'


def test_proxy_offer_presentation_cannot_run_an_anatomical_planner_or_change_control():
    _, _, prediction, _ = fixture()
    prediction['visual_references']['external']['reference_role'] = 'rigid_transit_proxy'
    policy = SimpleNamespace(prediction=prediction)
    before = copy.deepcopy(prediction)
    bank = FieldProposalBankV32()
    report = bank.build(policy, observation(), {'candidates': [{'goal_offset_m': [.01, 0, 0]}]})
    assert report['offers'] == []
    assert report['ordinary_metric_field_and_recovery_choices_available']
    assert policy.prediction == before
