"""Training from paid-for motion is distinct from new causal confirmation."""
import copy
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_monitor_v12 import RoleAwareVisualMonitorV12
from gpt_field.visual_translation_guard_v17 import ProjectedTranslationGuardV17
from gpt_field.visual_reference_replay_v18 import evidence_digest, warm_start_reference_response
from test_visual_field_v6_policy import observation, prediction, MOVING, TARGET
from test_visual_monitor_v8 import views


def fixture(*, static=False):
    observations=[observation((i*.001,0,0),slip=static) for i in range(7)]
    obs=observations[-1]; monitor=RoleAwareVisualMonitorV12();refs=prediction()['visual_references']
    end=MOVING+(np.zeros(2) if static else [6,0])
    refs['external']['seed']['moving_point_uv']=(end/223).tolist()
    monitor.configure(views(obs),refs,obs.state,intent_id=1)
    guard=ProjectedTranslationGuardV17()
    context={'purpose':'reference_repair','translation_mode':'separate','intended_contact_id':None,
        'obstacle_assessment':'present','tool_external_roi_uv':[40/223,55/223,95/223,95/223],
        'nearby_obstacles':[{'id':'destination','external_roi_uv':[155/223,140/223,175/223,160/223]}]}
    guard.configure(views(obs),context,obs.state,target_id='destination',monitor=monitor)
    records=[]
    for i,item in enumerate(observations):
        rgb=np.asarray(decode_image(item.image))
        records.append({'sequence':i,'state8':list(item.state),'joint_position':list(item.joint_position),
            'candidate_point_px':(MOVING+(np.zeros(2) if static else [i,0])).tolist(),
            'anchor_point_px':TARGET.tolist(),'guard_anchor_point_px':TARGET.tolist(),
            'noise_px':.08,'candidate_reliable':True,'anchor_reliable':True,'guard_anchor_reliable':True,
            'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
    evidence={'kind':'retrospective_candidate_conditioned','chronological_samples':records,
        'current_sequence':6,'image_shape':[224,224], 'guard_anchor_source':'explicit_stationary_anchor',
        'validation_is_prospective':False,'selected_semantic_identity_certified':False}
    resolved={'candidate_id':'new_distal_candidate','current_point_uv':(end/223).tolist(),
        'current_roi_uv':[50/223,60/223,80/223,90/223],'evidence':evidence,'evidence_sha256':evidence_digest(evidence)}
    return monitor,guard,resolved,obs


def physical_state(monitor,guard):
    return pickle.dumps({'monitor_state':monitor.state,'origin':monitor.origin,'status':monitor.status,
        'reference':monitor._active_reference,'references':monitor._reference_records,
        'path_m':monitor.path_m,'intent_path_m':monitor.intent_path_m,'unvalidated_path_m':monitor.unvalidated_path_m,
        'attachment':monitor.attachment.__dict__,'guard_state':guard.state,'observation_debt':guard.observation_travel_m,
        'experiments':guard.experiments,'active_experiment':guard._last_charged_experiment,
        'worsening':guard.worsening,'improving':guard.improving,'context':guard.context,'tool_box':guard.tool_box,
        'domain_invalid':guard.domain_invalid,'context_invalid':guard.context_invalid})


def test_both_models_warm_start_without_granting_validation_or_resetting_physical_state():
    monitor,guard,resolved,obs=fixture()
    guard.observation_travel_m=.014
    guard.experiments=[{'direction':[0,0,1],'measured_travel_m':.008,'acknowledged_actions':12}]
    guard._last_charged_experiment=0
    monitor._active_reference['unvalidated_travel_m']=.014
    monitor.validation_count=9;guard.validations=9
    monitor._held_out_motion=[(1.,0.)]*4;guard._response_error.extend([(0.,1.)]*4)
    original=physical_state(monitor,guard)
    result=warm_start_reference_response(monitor,guard,resolved,obs)
    assert result['accepted'] and result['fresh_live_validation_required']
    assert result['historical_samples_are_held_out'] is False
    assert all(model.matrix is not None for model in (monitor.response,guard.response,guard.cumulative_response))
    assert np.allclose(monitor.response.predict([.001,0,0]),[1,0],atol=1e-4)
    assert not monitor.response_ready and not guard.cumulative_response_ready and not guard.response_ready
    assert monitor.validation_count==0 and guard.validations==0
    assert not monitor._held_out_motion and not monitor._prediction_window and not guard._response_error
    assert physical_state(monitor,guard)==original
    assert not monitor.attachment_evidence


def test_static_candidate_rejected_and_both_observers_unchanged():
    monitor,guard,resolved,obs=fixture(static=True)
    original=pickle.dumps((monitor,guard))
    result=warm_start_reference_response(monitor,guard,resolved,obs)
    assert result['reason']=='candidate_relative_motion_unresolved'
    assert pickle.dumps((monitor,guard))==original


@pytest.mark.parametrize('change,expected',[
    ('digest','candidate_evidence_digest_mismatch'),
    ('sequence','candidate_history_not_chronological'),
    ('orientation','candidate_orientation_domain_changed'),
    ('fingers','candidate_finger_domain_changed'),
    ('joints','candidate_joint_domain_changed'),
    ('position','candidate_position_domain_changed'),
    ('anchor','candidate_stationary_anchor_motion_unresolved'),
    ('reliability','candidate_history_correspondence_unreliable'),
])
def test_changed_or_incompatible_history_never_mutates_models(change,expected):
    monitor,guard,resolved,obs=fixture();evidence=resolved['evidence'];rows=evidence['chronological_samples']
    if change=='digest': rows[0]['candidate_point_px'][0]+=.001
    if change=='sequence':rows[1]['sequence']=rows[0]['sequence']
    if change=='orientation':rows[0]['state8'][3]+=.02
    if change=='fingers':rows[0]['state8'][6]+=.003
    if change=='joints':rows[0]['joint_position'][0]+=.13
    if change=='position':rows[0]['state8'][0]-=.05
    if change=='anchor':rows[0]['anchor_point_px'][0]+=2.
    if change=='reliability':rows[1]['candidate_reliable']=False
    if change!='digest':resolved['evidence_sha256']=evidence_digest(evidence)
    original=pickle.dumps((monitor,guard));result=warm_start_reference_response(monitor,guard,resolved,obs)
    assert result['reason']==expected
    assert pickle.dumps((monitor,guard))==original


def test_current_image_and_actual_seed_binding_are_required():
    monitor,guard,resolved,obs=fixture()
    changed=observation((.007,0,0));result=warm_start_reference_response(monitor,guard,resolved,changed)
    assert result['reason']=='candidate_current_rgb_changed'
    monitor.tracks['moving'].position+=np.array([1,0])
    assert warm_start_reference_response(monitor,guard,resolved,obs)['reason']=='candidate_current_reference_point_changed'


def test_pending_command_cannot_be_silently_invalidated_by_warm_start():
    monitor,guard,resolved,obs=fixture()
    guard._motion_classification_pending={'uncertain':True,'direction':[1,0,0],'command_cap_m':.002}
    original=pickle.dumps((monitor,guard))
    assert warm_start_reference_response(monitor,guard,resolved,obs)['reason']=='candidate_import_during_pending_transition'
    assert pickle.dumps((monitor,guard))==original


def test_only_fresh_real_rgb_observations_can_confirm_predictors_and_duplicate_import_is_refused():
    monitor,guard,resolved,obs=fixture()
    result=warm_start_reference_response(monitor,guard,resolved,obs)
    assert result['accepted']
    for _ in range(3):
        monitor.observe(views(obs),obs.state,[0]*7)
        guard.observe(views(obs),obs.state,[0]*7,monitor=monitor)
    assert not monitor.response_ready and not guard.cumulative_response_ready
    fresh=[]
    for i in range(7,12):
        after=observation((i*.001,0,0))
        monitor.observe(views(after),after.state,[.04,0,0,0,0,0,0])
        guard.observe(views(after),after.state,[.04,0,0,0,0,0,0],monitor=monitor)
        fresh.append((monitor.response_ready,guard.cumulative_response_ready))
    assert all(fresh[-1])
    assert len(monitor._held_out_motion)==5 and len(guard._response_error)>=2
    assert not monitor.attachment_evidence
    # Reapplying at the original endpoint cannot erase fresh validation.
    original=pickle.dumps((monitor,guard))
    retry=warm_start_reference_response(monitor,guard,resolved,obs)
    assert not retry['accepted'] and pickle.dumps((monitor,guard))==original


def test_replay_does_not_unblock_a_previously_rejected_correspondence():
    monitor,guard,resolved,obs=fixture()
    monitor._active_reference['blocked_reason']='controlled_reference_uncertainty_budget_exhausted'
    monitor.status='controlled_reference_uncertainty_budget_exhausted'
    original=physical_state(monitor,guard)
    assert warm_start_reference_response(monitor,guard,resolved,obs)['accepted']
    assert physical_state(monitor,guard)==original and monitor.measured_control_limit()[0]==0.


def test_same_resolved_evidence_is_applied_only_once():
    monitor,guard,resolved,obs=fixture()
    assert warm_start_reference_response(monitor,guard,resolved,obs)['accepted']
    original=pickle.dumps((monitor,guard))
    assert warm_start_reference_response(monitor,guard,resolved,obs)['reason']=='candidate_replay_already_applied'
    assert pickle.dumps((monitor,guard))==original


def archive_fixture():
    """Actual paid-for V17 motion; no simulator or policy call is performed."""
    root = Path(__file__).resolve().parents[1]
    design = root/'runs/visual_field_v18_design_001'
    candidate_file = design/'motion-1-6.resolved.json'
    if not candidate_file.exists():
        pytest.skip('optional saved V17 RGB evidence is not in this checkout')
    resolved = json.loads(candidate_file.read_text())
    obs = PolicyObservation.from_dict(json.loads((design/'candidate_current_observation.json').read_text()))
    trace = root/'runs/visual_field_v17_zero_shot_001/trials/bowl_original_zero_hybrid_v17/trace.jsonl'
    with trace.open() as handle:
        pred = next(row['diagnostics']['prediction'] for line in handle
                    if (row := json.loads(line))['event']=='action' and row['diagnostics'].get('refreshed'))
    refs = copy.deepcopy(pred['visual_references'])
    end = resolved['evidence']['chronological_samples'][-1]
    scale = np.array(resolved['evidence']['image_shape'][::-1])-1
    refs['external']['update'] = 'seed'
    refs['external']['seed'] = {
        'moving_point_uv':resolved['current_point_uv'], 'moving_roi_uv':resolved['current_roi_uv'],
        'target_anchor_uv':(np.asarray(end['anchor_point_px'])/scale).tolist(),
        'target_roi_uv':None, 'tool_point_uv':None, 'tool_roi_uv':None}
    monitor = RoleAwareVisualMonitorV12()
    monitor.configure(views(obs),refs,obs.state,intent_id=999)
    guard = ProjectedTranslationGuardV17()
    guard.configure(views(obs),pred['motion_context'],obs.state,target_id=pred['target_id'],monitor=monitor)
    return monitor,guard,resolved,obs


def test_actual_v17_early_review_candidate_trains_both_without_claiming_causal_validation():
    monitor,guard,resolved,obs = archive_fixture()
    assert guard.context_invalid is None  # Candidate is within the old tool envelope.
    guard.observation_travel_m = .004179377273851864
    original = physical_state(monitor,guard)
    result = warm_start_reference_response(monitor,guard,resolved,obs)
    assert result['accepted']
    assert result['sample_count']==7
    # The seven retained states omit tiny intermediate hold drift; physical
    # debt remains the full nine-transition measured path above.
    assert result['training_measured_path_m']==pytest.approx(.004176665867183138)
    assert result['relative_image_extent_px']==pytest.approx(.8149887969908739)
    assert result['maximum_point_precision_proxy_px']==pytest.approx(.18140603733148103)
    assert result['training_image_motion_threshold_px']==pytest.approx(.5442181119944431)
    assert result['guard_aggregate_training']['rank']==1
    assert result['guard_aggregate_training']['sample_count']==3
    assert result['guard_aggregate_training']['residual'] < .12
    assert result['noise_is_calibrated_bound'] is False
    assert not monitor.response_ready and not guard.cumulative_response_ready
    assert physical_state(monitor,guard)==original
    assert guard.context_invalid is None
    assert not result['semantic_identity_certified'] and not result['clearance_certified']


def test_actual_archive_candidate_cannot_be_imported_at_a_different_robot_state():
    monitor,guard,resolved,obs = archive_fixture()
    state = list(obs.state); state[0] += .001
    altered = replace(obs,state=state)
    original = pickle.dumps((monitor,guard))
    result = warm_start_reference_response(monitor,guard,resolved,altered)
    assert result['reason']=='candidate_current_robot_state_changed'
    assert pickle.dumps((monitor,guard))==original
