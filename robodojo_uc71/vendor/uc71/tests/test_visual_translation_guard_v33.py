"""Reviewed reversal may abort adverse acquisition; no extra physical credit."""
import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v33 import ProjectedTranslationGuardV33
from gpt_field.visual_translation_guard_v32 import ProjectedTranslationGuardV32
from test_visual_translation_guard_v25 import configured as old_fixture
from test_visual_translation_guard_v21 import observe, pixels


def configured():
    guard, monitor, field, state, pixel = old_fixture()
    guard.__class__ = ProjectedTranslationGuardV33
    for key, value in ProjectedTranslationGuardV33().__dict__.items():
        if key not in guard.__dict__:
            guard.__dict__[key] = value
    images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    guard.configure(images, guard.context, state, target_id='neighbor', monitor=monitor)
    guard.bind_egress_runtime(None, monitor)
    guard.experiments = [{'direction': [1., 0., 0.], 'measured_travel_m': .002,
        'acknowledged_actions': 2, 'support_generation': 0}]
    guard._last_charged_experiment = 0
    guard._charge_uncertain_motion = True
    guard.observation_travel_m = .002
    prediction = {'controller':'visual_field','phase':'recover','recovery_route':'reverse_observed',
        'target_id':'neighbor','target_contact':'none','contact_alignment':None,
        'motion_context':{'purpose':'reference_repair','translation_mode':'separate','intended_contact_id':None},
        'visual_references':{'external':{'moving_kind':'tool'}},
        'field_intent':{'gripper_command':'hold'},
        'subgoal_update':{'action':'replace','subgoal_id':'original','reason':'Reverse acknowledged adverse motion'}}
    assert guard.bind_reverse_abort_context(prediction, monitor.intent_id, fixed_goal_xyz=field.goal)['bound']
    command = np.array([-.002,0.,0.])
    kwargs = {'monitor':monitor,'nominal_clear':True,'nominal_away':False,'orientation_hold':True,
        'nominal_regions':field.nominal_regions_for_segment(state[:3],state[:3]+command)}
    return guard, monitor, field, state, pixel, prediction, command, kwargs


def test_actual_rgb_worsening_plus_review_can_abort_before_checkpoint():
    g,m,f,s,px,p,d,k=configured()
    prior=copy.deepcopy((g.experiments,g.observation_travel_m,g.worsening,g.worsening_motion))
    g.__class__=ProjectedTranslationGuardV32
    old=g.assess(d,**k)
    assert not old['accepted'] and old['reason']=='translation_recovery_direction_change_requires_review'
    g.__class__=ProjectedTranslationGuardV33
    new=g.assess(d,**k)
    assert new['accepted'],new['reason']
    assert new['reason']=='projected_worsening_bounded_observed_motion_reverse'
    assert (g.experiments,g.observation_travel_m,g.worsening,g.worsening_motion)==prior
    assert g.information_checkpoint_m==.004
    source=copy.deepcopy(g.experiments[0])
    observe(g,m,s,d,[-.0005,0,0],pixel_x=px-1)
    assert len(g.experiments)==2 and g.experiments[0]==source
    assert g.observation_travel_m==pytest.approx(.0025)
    assert g.experiments[1]['measured_travel_m']==pytest.approx(.0005)
    assert len(g.reverse_abort_events)==1
    assert g.worsening=={'neighbor'}


@pytest.mark.parametrize('fault',['no_review','retain','normal','goal','state','epoch','intent','no_adverse',
    'unpaid','source_heading','geometry','unknown','payload','contact','reference_lost','contradicted',
    'further_worsening','full_budget','slots'])
def test_abort_never_relaxes_other_checks(fault):
    g,m,f,s,px,p,d,k=configured()
    c=k['nominal_regions'][0]['directional_retreat_contract']
    if fault=='no_review':g.reverse_abort_context=None
    elif fault in ('retain','normal'):
        if fault=='retain':p['subgoal_update']['action']='retain'
        else:p['recovery_route']='normal'
        assert not g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal)['bound']
    elif fault=='goal':g.reverse_abort_context['goal_xyz_m'][1]+=.001
    elif fault=='state':g.reverse_abort_context['state8'][0]+=.001
    elif fault=='epoch':g.reverse_abort_context['observation_epoch']-=1
    elif fault=='intent':m.intent_id+=1
    elif fault=='no_adverse':g.worsening_motion={}
    elif fault=='unpaid':g.experiments[0]['measured_travel_m']=0.
    elif fault=='source_heading':g.experiments[0]['direction']=[0.,1.,0.]
    elif fault=='geometry':c['geometry_sha256']='new'
    elif fault=='unknown':k['nominal_regions'][0]['kind']='unknown_regions'
    elif fault=='payload':c['empty_tool']=False
    elif fault=='contact':c['target_contact']='fingers'
    elif fault=='reference_lost':m.tracks['moving'].reliable=False
    elif fault=='contradicted':m.cumulative_response_evidence=lambda:{'contradicted':True}
    elif fault=='further_worsening':g.retreat_progress={'further_worsening':True}
    elif fault=='full_budget':g.observation_travel_m=.020
    elif fault=='slots':
        g.experiments.extend([{'direction':[0.,1.,0.],'measured_travel_m':.002},
                              {'direction':[0.,0.,1.],'measured_travel_m':.002}])
    before=copy.deepcopy((g.experiments,g.observation_travel_m))
    result=g.assess(d,**k)
    assert not result['accepted'],(fault,result)
    assert g._reverse_abort_pending is None
    assert (g.experiments,g.observation_travel_m)==before


def test_preview_is_pure_requires_explicit_choice_and_survives_exception():
    g,m,f,s,px,p,d,k=configured();g.reverse_abort_context=None
    before=pickle.dumps(g);track=g.tool_track
    with g.preview_reverse_abort_context(s,f.goal-s[:3]):
        result=g.preview_assess(d,**k)
        assert result['accepted'],result['reason']
        assert result['diagnostics']['precheckpoint_reverse_abort']['last_admission']['requires_explicit_model_choice']
    assert pickle.dumps(g)==before and g.tool_track is track
    assert not g.assess(d,**k)['accepted']
    before=pickle.dumps(g)
    with pytest.raises(RuntimeError):
        with g.preview_reverse_abort_context(s,f.goal-s[:3]):
            assert g.assess(d,**k)['accepted']
            raise RuntimeError('prospective executor failure')
    assert pickle.dumps(g)==before


def test_downstream_hold_keeps_original_paid_direction_and_does_not_commit_abort():
    g,m,f,s,px,p,d,k=configured();assert g.assess(d,**k)['accepted']
    observe(g,m,s,[0,0,0],[.0002,0,0],pixel_x=px)
    assert len(g.experiments)==1 and not g.reverse_abort_events
    assert g.experiments[0]['measured_travel_m']==pytest.approx(.0022)
    assert g.observation_travel_m==pytest.approx(.0022)
    assert g.retreat_progress is None


@pytest.mark.parametrize('mismatch',['different_command','role_rebind','epoch','intent'])
def test_stale_ack_charges_motion_without_committing_abort(mismatch):
    g,m,f,s,px,p,d,k=configured();assert g.assess(d,**k)['accepted']
    if mismatch=='different_command':d=d*.5
    elif mismatch=='role_rebind':g.bind_reverse_abort_context({'controller':'direct'},m.intent_id,fixed_goal_xyz=f.goal)
    elif mismatch=='epoch':g.sweep_observation_epoch+=1
    elif mismatch=='intent':m.intent_id+=1
    observe(g,m,s,d,[-.0003,0,0],pixel_x=px)
    assert not g.reverse_abort_events and len(g.experiments)==1
    assert g.observation_travel_m==pytest.approx(.0023)
    assert g.domain_invalid=='translation_reverse_abort_acknowledgement_mismatch'


def test_repeated_current_bind_uses_fixed_goal_not_relative_offset():
    g,m,f,s,px,p,d,k=configured()
    p['field_intent']['goal_offset_m']=[-.008,0,0]
    original=f.goal.copy()
    assert g.assess(d,**k)['accepted']
    after=observe(g,m,s,d,[-.0005,0,0],pixel_x=px-1)
    assert g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=original)['bound']
    assert g.reverse_abort_context['goal_xyz_m']==original.tolist()
    assert g.reverse_abort_context['state8']==after.tolist()
    assert g.observation_travel_m==pytest.approx(.0025)


@pytest.mark.parametrize('step',[56,68])
def test_actual_archived_refusal_is_admissible_without_executing_or_repaying_motion(step):
    path=Path(__file__).resolve().parents[1]/f'runs/visual_field_v33_design_001/archive_step{step}_checkpoint.pkl'
    if not path.exists():pytest.skip('Optional exact archived policy checkpoint unavailable')
    g,m,p,requests,obs,d,k,goal=pickle.loads(path.read_bytes())
    for key,value in ProjectedTranslationGuardV33().__dict__.items():
        if key not in g.__dict__:g.__dict__[key]=value
    prior=copy.deepcopy((g.experiments,g.observation_travel_m,g.worsening))
    g.__class__=ProjectedTranslationGuardV32
    old=g.assess(d,**k)
    assert old['reason']=='translation_recovery_direction_change_requires_review'
    g.__class__=ProjectedTranslationGuardV33
    assert g.bind_reverse_abort_context(p,requests,fixed_goal_xyz=goal)['bound']
    new=g.assess(d,**k)
    assert new['accepted'],new['reason']
    assert (g.experiments,g.observation_travel_m,g.worsening)==prior
    assert new['cap_m']<=.003


def test_repeated_binding_cannot_claim_review_of_a_later_adverse_event():
    g,m,f,s,px,p,d,k=configured()
    review_epoch=g.reverse_abort_context['review_observation_epoch']
    g.sweep_observation_epoch+=1
    g.worsening_motion['neighbor']['observation_epoch']=g.sweep_observation_epoch
    assert g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal)['bound']
    assert g.reverse_abort_context['review_observation_epoch']==review_epoch
    k['nominal_regions']=f.nominal_regions_for_segment(s[:3],s[:3]+d)
    assert g.assess(d,**k)['reason']=='translation_recovery_direction_change_requires_review'
    m.intent_id+=1
    assert g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal)['bound']
    assert g.assess(d,**k)['accepted']


def test_same_model_request_cannot_rebind_a_different_fixed_goal():
    g,m,f,s,px,p,d,k=configured()
    assert not g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal+[0,.001,0])['bound']
    assert not g.assess(d,**k)['accepted']


def test_same_epoch_history_replacement_requires_a_new_actual_review():
    g,m,f,s,px,p,d,k=configured()
    g.worsening_motion['neighbor']['measured_net_translation_m']+=.001
    assert g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal)['bound']
    assert not g.assess(d,**k)['accepted']
    m.intent_id+=1
    assert g.bind_reverse_abort_context(p,m.intent_id,fixed_goal_xyz=f.goal)['bound']
    assert g.assess(d,**k)['accepted']
