"""Serialization equivalence never moves the first physical route binding."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from gpt_field.visual_route_exposure_v24 import RouteExposureV24
from gpt_field.visual_route_exposure_v27 import RouteExposureV27
from test_visual_route_exposure_v24 import plant


OLD_GOAL=np.array([-.21124162683412046,-.0005946606628817585,1.174249517698727])
REEXPRESSED_GOAL=np.array([-.21124162683441447,-.0005946606632061049,1.1742495176978596])


def current_plant():
    p,status=plant(qualified=True,prior_debt=.003)
    p.ledger=RouteExposureV27()
    original=p.proof
    p.goal=OLD_GOAL.copy()
    p.proof=lambda command:dict(original(command),complete_route_goal_xyz=p.goal.tolist())
    return p,status


def test_actual_rounding_bin_boundary_is_equivalent_without_moving_goal_or_debt():
    p,_=current_plant();first=p.proof(np.array([.002,0,0]))
    changed=copy.deepcopy(first);changed['complete_route_goal_xyz']=REEXPRESSED_GOAL.tolist()
    assert RouteExposureV24._geometry_context(first)!=RouteExposureV24._geometry_context(changed)
    initial=pickle.dumps(p.ledger.__dict__)
    assert p.assess()['accepted'] and pickle.dumps(p.ledger.__dict__)==initial
    p.move(p.assess(),actual=.0005)
    binding=p.ledger.committed_geometry_context
    assert binding[-1]==tuple(OLD_GOAL)  # Raw first acknowledged goal, not rounded.
    debt=p.ledger.spent_m;p.goal=REEXPRESSED_GOAL.copy()
    before=pickle.dumps(p.ledger.__dict__)
    admission=p.assess();assert admission['accepted'],admission
    assert pickle.dumps(p.ledger.__dict__)==before
    assert p.ledger.committed_geometry_context is binding and p.ledger.spent_m==debt
    p.move(admission,actual=.0005)
    assert p.ledger.observations==2 and p.ledger.spent_m==pytest.approx(debt+.0005)
    assert p.ledger.committed_geometry_context is binding


@pytest.mark.parametrize('revision',['geometry','origin','parts','real_goal','numeric_outside'])
def test_only_numeric_goal_equivalence_is_permitted(revision):
    p,_=current_plant();p.move(p.assess())
    proof=p.proof(np.array([.002,0,0]))
    if revision=='geometry':proof['geometry_sha256']='changed-geometry'
    elif revision=='origin':proof['geometry_origin_state8'][0]+=.000001
    elif revision=='parts':proof['initial_halo_parts'][0]['tool_part']='different-part'
    elif revision=='real_goal':proof['complete_route_goal_xyz'][0]+=.001
    else:proof['complete_route_goal_xyz'][0]+=4e-12
    before=pickle.dumps(p.ledger.__dict__)
    result=p.ledger.assess(proof,p.state,[.002,0,0],guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert not result['accepted'] and result['reason']=='route_exposure_original_geometry_or_goal_changed'
    assert pickle.dumps(p.ledger.__dict__)==before


def test_successive_small_goal_changes_cannot_accumulate_a_new_route():
    p,_=current_plant();p.move(p.assess());binding=p.ledger.committed_geometry_context
    for shift in (.8e-12,1.6e-12):
        p.goal=OLD_GOAL+[shift,0,0]
        choice=p.assess();assert choice['accepted']
        p.move(choice,actual=.0002)
        assert p.ledger.committed_geometry_context is binding
    p.goal=OLD_GOAL+[2.4e-12,0,0]
    assert p.assess()['reason']=='route_exposure_original_geometry_or_goal_changed'
    assert p.ledger.committed_geometry_context is binding


@pytest.mark.parametrize('goal',[[float('nan'),0,1],[0,float('inf'),1],[0,1],None])
def test_malformed_goals_never_match_or_commit(goal):
    p,_=current_plant();proof=p.proof(np.array([.002,0,0]));proof['complete_route_goal_xyz']=goal
    before=pickle.dumps(p.ledger.__dict__)
    result=p.ledger.assess(proof,p.state,[.002,0,0],guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert not result['accepted'] and result['reason']=='route_exposure_goal_binding_invalid'
    assert pickle.dumps(p.ledger.__dict__)==before
    assert p.ledger.committed_geometry_context is None


def test_invalid_pending_goal_cannot_commit_but_retains_measured_shared_debt():
    p,_=current_plant();admission=p.assess()
    admission['pending']['proof']['complete_route_goal_xyz']=[float('nan'),0,1]
    p.move(admission,actual=.0005)
    assert p.ledger.committed_geometry_context is None and p.ledger.observations==0
    assert p.ledger.spent_m>=p.guard.observation_travel_m
    assert p.ledger.last_progress['worsening']
    assert p.ledger.last_reason=='route_exposure_goal_binding_invalid'


def test_legacy_ledger_and_guard_remain_unchanged():
    from gpt_field.visual_translation_guard_v26 import ProjectedTranslationGuardV26
    from gpt_field.visual_translation_guard_v27 import ProjectedTranslationGuardV27
    assert type(ProjectedTranslationGuardV26().egress_evidence) is RouteExposureV24
    assert type(ProjectedTranslationGuardV27().egress_evidence) is RouteExposureV27


def test_recorded_step9_candidate_repair_preserves_world_goal_and_admits_current_action():
    root=Path(__file__).resolve().parents[1]
    run=root/'runs/visual_field_v27_zero_shot_001'
    path=run/'trials/bowl_original_zero_hybrid_v27/trace.jsonl'
    if not path.exists():pytest.skip('Optional stopped V27 full trial is unavailable')
    helperpath=root/'runs/visual_field_v27_design_001/archived_approach_inputs.py'
    spec=importlib.util.spec_from_file_location('v27_numeric_goal_cached_client',helperpath)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from gpt_field.observation import PolicyObservation
    from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
    rows=[json.loads(s)for s in path.read_text().splitlines()]
    actions={r['step']:r for r in rows if r['event']=='action'}
    executed={r['step']:r['observation']for r in rows if r['event']=='executed'}
    initial=json.loads((run/'inputs/bowl_original/expected_initial.json').read_text())
    records=[{'archive_step':i,'prediction':copy.deepcopy(actions[i]['diagnostics']['prediction']),
        'observation':initial if i==0 else executed[i-1]}for i in (0,1,9)]
    client=helper.ArchivedPrefixClient(records);policy=VisualFieldPolicyV27(client)
    obs=PolicyObservation.from_dict(initial)
    try:
        for step in range(9):
            decision=policy.act(obs)
            assert np.allclose(decision.action,actions[step]['action'],atol=1e-10,rtol=0.),step
            after=PolicyObservation.from_dict(executed[step])
            policy.observe_transition(obs,decision,after);obs=after
        exposure=policy.translation_guard.egress_evidence
        binding=exposure.committed_geometry_context
        before=copy.deepcopy((exposure.spent_m,exposure.egress_travel_m,
            exposure.confirmation_start_travel_m,policy.translation_guard.observation_travel_m))
        goal=policy.subgoals.summary()['active']['world_goal_xyz']
        decision=policy.act(obs)
        assert decision.diagnostics['selection']['accepted'],decision.diagnostics['selection']['status']
        assert np.linalg.norm(decision.action[:3])*.05<=.003+1e-10
        assert exposure.committed_geometry_context is binding
        assert binding[-1]==tuple(OLD_GOAL)
        assert (exposure.spent_m,exposure.egress_travel_m,exposure.confirmation_start_travel_m,
            policy.translation_guard.observation_travel_m)==before
        assert policy.subgoals.summary()['active']['world_goal_xyz']==goal
        assert client.calls==3 and client.records[-1]['candidate_revalidation']['support_matches_archive']
        assert policy.local.pending is not None
        # Stop at the first new action. Do not replay observations from the
        # archived rejected hold as if they followed this accepted movement.
    finally:policy.finish()
