"""Validated normal transit is a separate, finite, acknowledgement-bound phase."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v29 import ProjectedTranslationGuardV29
from gpt_field.visual_translation_guard_v27 import _NeighborResponse
from gpt_field.robot_translation_response_v25 import MeasuredTranslationResponseV25
from test_visual_translation_guard_v27 import setup as old_setup
from test_visual_translation_guard_v21 import observe, pixels


def setup(monkeypatch):
    guard, field, state, delta, kwargs = old_setup(monkeypatch)
    guard.__class__ = ProjectedTranslationGuardV29
    for key, value in ProjectedTranslationGuardV29().__dict__.items():
        if key not in guard.__dict__: guard.__dict__[key] = value
    return guard, field, state, delta, kwargs


def debt(guard, shared=.020, heading=.008):
    guard.observation_travel_m = shared
    guard.experiments = [{'direction': [-1.,0,0], 'support_generation':0,
        'measured_travel_m': heading, 'acknowledged_actions':8}]
    guard._last_charged_experiment = 0
    guard._charge_uncertain_motion = True


def actual_observer(monkeypatch, guard, kwargs):
    # Keep admission's isolated legacy-warning trigger, but use REAL RGB/robot
    # observers and trackers. They are not bypassed during acknowledgement.
    monkeypatch.delattr(kwargs['monitor'], '_external_reliable')
    monkeypatch.delattr(kwargs['monitor'], 'cumulative_response_evidence')
    guard._neighbor_responses = {}


def proof_for_current(field, guard, state, delta):
    field.expected_state = state.copy()
    return field.nominal_regions_for_segment(state[:3], state[:3]+delta)


def test_fully_validated_transit_can_cross_exhausted_acquisition_without_refund(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    debt(guard)
    old = copy.deepcopy((guard.observation_travel_m, guard.experiments, guard.worsening))
    choice = guard.assess(delta, **kwargs)
    assert choice['accepted'] and choice['reason'] == 'projected_warning_empirically_validated_fixed_intent_execution'
    assert guard._experiment_pending is None
    assert guard._motion_classification_pending['uncertain'] is False
    assert (guard.observation_travel_m, guard.experiments, guard.worsening) == old
    actual_observer(monkeypatch, guard, kwargs)
    state = observe(guard, kwargs['monitor'], state, delta, [-.0008,0,0], pixel_x=-1)
    assert guard.supported_execution_travel_m == pytest.approx(.0008)
    assert guard.supported_execution_actions == 1 and not guard._charge_uncertain_motion
    assert (guard.observation_travel_m, guard.experiments) == old[:2]
    state = observe(guard, kwargs['monitor'], state, [0,0,0], [-.0002,0,0], pixel_x=-1)
    assert guard.supported_execution_travel_m == pytest.approx(.001)
    assert guard.supported_residual_hold_travel_m == pytest.approx(.0002)
    assert guard.supported_execution_actions == 1
    assert (guard.observation_travel_m, guard.experiments) == old[:2]


def test_downstream_veto_hold_keeps_actual_prior_acquisition_class_and_debt(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch)
    debt(guard, shared=.006, heading=.006)
    assert guard.assess(delta, **kwargs)['accepted']
    actual_observer(monkeypatch, guard, kwargs)
    state = observe(guard, kwargs['monitor'], state, [0,0,0], [-.0002,0,0], pixel_x=-1)
    assert guard.supported_execution_actions == 0 and guard._validated_rows == {}
    assert guard.observation_travel_m == pytest.approx(.0062)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(.0062)
    assert guard._charge_uncertain_motion


@pytest.mark.parametrize('fault', ['epoch','state','endpoint','normal_route','payload','contact',
    'unknown','decreasing','minimum_gap','scene_ids','fingerprint','boundary','closing','prior_closing',
    'reference','robot','new_heading','stale_response','reverse_contradiction','domain','context'])
def test_earlier_geometry_or_evidence_veto_never_promoted(monkeypatch, fault):
    guard, field, state, delta, kwargs = setup(monkeypatch); debt(guard)
    c = kwargs['nominal_regions'][0]['transit_reconciliation_contract']
    if fault == 'epoch': c['observation_epoch'] += 1
    elif fault == 'state': c['state8'][0] += .00001
    elif fault == 'endpoint': c['end_xyz_m'][0] += .00001
    elif fault == 'normal_route': c['route_family'] = 'reverse_observed'
    elif fault == 'payload': c['empty_tool'] = False
    elif fault == 'contact': c['target_contact'] = 'fingers'
    elif fault == 'unknown': c['hard_segment_clear'] = False
    elif fault == 'decreasing': c['object_proofs']['neighbor']['gap_nondecreasing'] = False
    elif fault == 'minimum_gap': c['object_proofs']['neighbor']['minimum_segment_gap_m'] = 0.
    elif fault == 'scene_ids': c['full_scene_ids'] = []
    elif fault == 'fingerprint': c['geometry_sha256'] = 'stale'
    elif fault == 'boundary': guard.boundary_evidence['neighbor']['comparable'] = False
    elif fault == 'closing': guard.boundary_evidence['neighbor']['classification'] = 'closing'
    elif fault == 'prior_closing': guard.boundary_closing_ids.add('neighbor')
    elif fault == 'reference': kwargs['monitor'].status = 'visual_response_contradiction'
    elif fault == 'robot': kwargs['translation_response'].predict = lambda *a: {'supported':False}
    elif fault in ('new_heading','stale_response'):
        guard._neighbor_responses['neighbor'] = SimpleNamespace(evidence=lambda *a:{'validated':False})
    elif fault == 'reverse_contradiction': guard.retreat_progress = {'further_worsening':True}
    elif fault == 'domain': guard.domain_invalid = 'translation_tool_correspondence_lost'
    elif fault == 'context': guard.context_invalid = 'invalid'
    old = copy.deepcopy((guard.observation_travel_m, guard.experiments))
    chosen = guard.assess(delta, **kwargs)
    assert not chosen['accepted'], (fault, chosen)
    assert guard._validated_pending is None
    assert (guard.observation_travel_m, guard.experiments) == old


def test_acquisition_failure_with_same_physical_reason_cannot_impersonate_validation(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch); debt(guard)
    guard._neighbor_responses['neighbor'] = SimpleNamespace(evidence=lambda *a: {'validated':False})
    refused = guard.assess(delta, **kwargs)
    assert refused['reason'] == 'translation_recovery_direction_exhausted'
    assert not refused['accepted'] and guard._validated_pending is None
    assert not guard.last_projected_warning_transit['per_object_response']['neighbor']['validated']


def test_preview_and_smaller_candidate_reselection_are_not_execution(monkeypatch):
    guard, field, state, delta, kwargs = setup(monkeypatch); debt(guard)
    original = guard.__dict__
    before = copy.deepcopy((guard.experiments, guard.observation_travel_m, guard._validated_rows))
    preview = guard.preview_assess(delta, **kwargs)
    assert preview['accepted'] and guard.__dict__ is original
    assert guard._validated_pending is None and guard.last_validated_intent_admission is None
    kwargs['translation_response'].command_cap_for_measured_limit = lambda *a:{'command_cap_m':.001}
    first = guard.assess(delta, **kwargs)
    assert first['accepted'] and first['cap_m'] == .001
    assert first['reason'].endswith('requires_cap_reselection')
    assert guard._motion_classification_pending is None and guard._validated_pending is None
    shorter = delta/3
    # Old long-segment proof is not a permission for this new candidate.
    assert not guard.assess(shorter, **kwargs)['accepted']
    kwargs['nominal_regions'] = proof_for_current(field, guard, state, shorter)
    chosen = guard.assess(shorter, **kwargs)
    assert chosen['accepted'] and guard._validated_pending['command_delta_m'] == pytest.approx(shorter)
    assert (guard.experiments, guard.observation_travel_m, guard._validated_rows) == before
    # A preview while an executable action is pending must restore that action.
    pending = copy.deepcopy(guard._validated_pending)
    guard.preview_assess(shorter, **kwargs)
    assert guard._validated_pending == pending


@pytest.mark.parametrize('fault', ['shorter','larger','different_direction','before_state','epoch','closing','rotation'])
def test_exact_acknowledgement_not_only_old_direction_and_cap_is_required(monkeypatch,fault):
    guard, field, state, delta, kwargs = setup(monkeypatch); debt(guard, .006, .006)
    assert guard.assess(delta, **kwargs)['accepted']
    actual_observer(monkeypatch,guard,kwargs)
    action = np.r_[delta/.05,0.,0.,0.,0.]
    if fault == 'shorter': action[0] *= .5
    elif fault == 'larger': action[0] *= 1.01
    elif fault == 'different_direction': action[:3] = [0,.06,0]
    elif fault == 'before_state': guard.state = guard.state.copy(); guard.state[0] += .000001
    elif fault == 'epoch': guard.sweep_observation_epoch += 1
    elif fault == 'closing': action[6] = 1.
    elif fault == 'rotation': action[3] = .2
    after=guard.state.copy();after[0]-=.0002
    images={'external':pixels(-1),'wrist':pixels(-1)}
    kwargs['monitor'].observe(images,after,action)
    guard.observe(images,after,action,monitor=kwargs['monitor'])
    assert guard.domain_invalid == 'translation_validated_intent_acknowledgement_mismatch'
    assert not guard.last_validated_intent_observation['matched']
    assert guard._validated_rows == {} and guard.supported_execution_actions == 0
    assert guard.observation_travel_m == pytest.approx(.0062)


def test_stalled_physical_heading_cannot_be_reset_by_new_validated_intent(monkeypatch):
    guard,field,state,delta,kwargs=setup(monkeypatch);debt(guard)
    guard.heading_acquisition.rows[0]={'adverse_response':{'reason':'observed_opposed_or_nonresponsive_heading'}}
    chosen=guard.assess(delta,**kwargs)
    assert not chosen['accepted'] and chosen['reason'].endswith('prior_heading_contradiction')


def test_budget_replay_and_goal_creep_cannot_extend_committed_intent(monkeypatch):
    guard,field,state,delta,kwargs=setup(monkeypatch)
    c=kwargs['nominal_regions'][0]['transit_reconciliation_contract']
    c['remaining_measured_travel_m']=.0011;c['remaining_actions']=1
    kwargs['translation_response'].command_cap_for_measured_limit=lambda d,r,n,s:{'command_cap_m':min(r,n)}
    first=guard.assess(delta,**kwargs);assert first['cap_m']==.0011
    smaller=np.array([-.0011,0,0]);kwargs['nominal_regions']=proof_for_current(field,guard,state,smaller)
    c=kwargs['nominal_regions'][0]['transit_reconciliation_contract'];c['remaining_measured_travel_m']=.0011;c['remaining_actions']=1
    assert guard.assess(smaller,**kwargs)['accepted']
    actual_observer(monkeypatch,guard,kwargs)
    state=observe(guard,kwargs['monitor'],state,smaller,[-.001,0,0],pixel_x=-1)
    row=next(iter(guard._validated_rows.values()))
    assert row['context']['fixed_goal_xyz_m'] == pytest.approx(field.goal)
    # Restore explicit fresh evidence stubs to isolate immutable intent accounting.
    monkeypatch.setattr(kwargs['monitor'],'_external_reliable',lambda:True)
    monkeypatch.setattr(kwargs['monitor'],'cumulative_response_evidence',lambda:{'contradicted':False})
    guard.worsening={'neighbor'};guard.domain_invalid=None
    guard.boundary_evidence['neighbor']={'comparable':True,'classification':'inconclusive','measurement_epoch':guard.neighbor_measurement_epochs['neighbor']}
    guard._neighbor_responses['neighbor']=SimpleNamespace(evidence=lambda *a:{'validated':True})
    kwargs['nominal_regions']=proof_for_current(field,guard,state,smaller)
    assert guard.assess(smaller,**kwargs)['reason']=='translation_validated_intent_budget_exhausted'
    # Neither a true changed goal nor cumulative serialization creep is accepted.
    for change in [3e-12,.001]:
        c=copy.deepcopy(kwargs['nominal_regions']);c[0]['transit_reconciliation_contract']['fixed_goal_xyz_m'][0]+=change
        assert guard.assess(smaller,**{**kwargs,'nominal_regions':c})['reason']=='translation_validated_intent_binding_changed'
    assert row['context']['fixed_goal_xyz_m'] == pytest.approx(field.goal)


def test_mismatched_command_after_supported_execution_records_unattributed_path(monkeypatch):
    guard,field,state,delta,kwargs=setup(monkeypatch)
    assert guard.assess(delta,**kwargs)['accepted']
    original_pending=copy.deepcopy(guard._validated_pending)
    actual_observer(monkeypatch,guard,kwargs)
    state=observe(guard,kwargs['monitor'],state,delta,[-.0008,0,0],pixel_x=-1)
    assert not guard._charge_uncertain_motion
    # An impossible reused old selection is explicitly rejected at observe;
    # this models an upstream caller bug, never a real policy intervention.
    guard._validated_pending=original_pending
    state=observe(guard,kwargs['monitor'],state,delta,[-.0003,0,0],pixel_x=-1)
    assert guard.domain_invalid=='translation_validated_intent_acknowledgement_mismatch'
    assert guard.supported_execution_travel_m==pytest.approx(.0008)
    assert guard.unmatched_execution_travel_m==pytest.approx(.0003)
    state=observe(guard,kwargs['monitor'],state,[0,0,0],[-.0001,0,0],pixel_x=-1)
    assert guard.unmatched_execution_travel_m==pytest.approx(.0004)
    assert guard.unmatched_residual_hold_travel_m==pytest.approx(.0001)
    assert guard.supported_execution_travel_m==pytest.approx(.0008)


def _snapshot(state,epoch,flow):
    return {'state':state[:3].copy(),'tool':np.array(flow,float),
        'neighbors':{'neighbor':{'position':np.zeros(2),'measurement_epoch':epoch,
                               'response_representation':epoch}}}


@pytest.mark.parametrize('gain',[1.,.2])
def test_causally_validated_intent_continues_beyond_eight_and_twenty_mm_with_real_ack(monkeypatch,gain):
    guard,field,state,delta,kwargs=setup(monkeypatch)
    debt(guard)  # Historical acquisition is exhausted, NOT erased by this test.
    # Generate causally checked observations on this same heading. The image
    # model must predict each held-out flow before training on that transition.
    robot=MeasuredTranslationResponseV25();witness=_NeighborResponse(1)
    command=np.array([-min(.003,.001/gain),0,0])
    measured_step=abs(command[0])*gain
    count=int(np.ceil(.007/measured_step))
    initial=state.copy();initial[0]+=count*measured_step
    witness.observe(_snapshot(initial,1,[(initial[0]-state[0])*500,0]),'neighbor')
    previous=initial
    for i in range(1,count+1):
        now=state.copy();now[0]+=(count-i)*measured_step
        robot.observe(previous,np.r_[command/.05,0.,0.,0.,0.],now)
        witness.observe(_snapshot(now,1,[(now[0]-state[0])*500,0]),'neighbor')
        previous=now
    assert witness.evidence([-.001,0,0],state[:3])['validated']
    assert len(witness.errors)>=2
    kwargs['translation_response']=robot
    delta=command.copy()  # Query only a causally supported command scale.
    # Physical RGB/robot observers run for every subsequent acknowledgement.
    # The dedicated causal witness has a synthetic known 500px/m observation
    # function, isolating lifecycle/accounting from image-render quantization.
    actual_observer(monkeypatch,guard,kwargs)
    old_debt=copy.deepcopy((guard.observation_travel_m,guard.experiments))
    traveled=0.;accepted=0;start=state.copy();original_goal=field.goal.copy()
    while traveled < .021:
        # Honest current evidence, never copying held-out credit or marking an
        # unsupported estimator validated. Only reference reliability is fixed
        # at the declared test-boundary; all actual observers still execute.
        monkeypatch.setattr(kwargs['monitor'],'_external_reliable',lambda:True)
        monkeypatch.setattr(kwargs['monitor'],'cumulative_response_evidence',lambda:{'contradicted':False})
        guard.worsening={'neighbor'}
        assert guard.domain_invalid is None
        guard.boundary_evidence['neighbor']={'comparable':True,'classification':'inconclusive',
            'measurement_epoch':guard.neighbor_measurement_epochs['neighbor']}
        guard._neighbor_responses['neighbor']=witness
        kwargs['nominal_regions']=proof_for_current(field,guard,state,delta)
        c=kwargs['nominal_regions'][0]['transit_reconciliation_contract']
        c['remaining_measured_travel_m']=.04-traveled;c['remaining_actions']=120-accepted
        choice=guard.assess(delta,**kwargs)
        assert choice['accepted'],choice['reason']
        if choice['reason'].endswith('requires_cap_reselection'):
            actual_command=delta/np.linalg.norm(delta)*choice['cap_m']
            kwargs['nominal_regions']=proof_for_current(field,guard,state,actual_command)
            c=kwargs['nominal_regions'][0]['transit_reconciliation_contract']
            c['remaining_measured_travel_m']=.04-traveled;c['remaining_actions']=120-accepted
            choice=guard.assess(actual_command,**kwargs)
            assert choice['accepted'],choice['reason']
        else: actual_command=delta.copy()
        assert choice['reason']=='projected_warning_empirically_validated_fixed_intent_execution'
        # Restore real monitor APIs before acknowledgement. Isolated witness
        # observations are supplied only AFTER the action, so no label leakage.
        monkeypatch.delattr(kwargs['monitor'],'_external_reliable')
        monkeypatch.delattr(kwargs['monitor'],'cumulative_response_evidence')
        guard._neighbor_responses={}
        before=state.copy();actual=actual_command*gain
        state=observe(guard,kwargs['monitor'],state,actual_command,actual,
                      pixel_x=-int(round((traveled+np.linalg.norm(actual))*500)))
        robot.observe(before,np.r_[actual_command/.05,0.,0.,0.,0.],state)
        witness.observe(_snapshot(state,1,[(state[0]-start[0])*500,0]),'neighbor')
        traveled+=float(np.linalg.norm(actual));accepted+=1
        assert (guard.observation_travel_m,guard.experiments)==old_debt
        assert field.goal==pytest.approx(original_goal)
    assert guard.supported_execution_travel_m==pytest.approx(traveled)
    assert traveled > .020 and guard.supported_execution_actions==accepted
    assert len(guard._validated_rows)==1
    assert not guard.summary()['validated_intent_execution']['clearance_certified']
