"""Selected anatomical response training from paid motion, never validation.

The existing V18 import must already have accepted this exact candidate at the
current observation. This helper imports only its measured three-millimetre
aggregate response samples into a freshly configured anatomical objective.
No observer, motion ledger, current goal, RGB history or held-out record changes.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np

from .visual_field import _gray
from .visual_monitor_v4 import _orientation_distance
from .visual_reference_replay_v18 import _observation, _vector, evidence_digest


def _prepare(field, monitor, candidate, observation, replay):
    ext = (getattr(monitor, 'refs', {}) or {}).get('external') or {}
    if ext.get('reference_role', 'tool_anatomy') != 'tool_anatomy' or ext.get('moving_kind') != 'tool':
        raise ValueError('goal_replay_requires_anatomical_tool_reference')
    if not isinstance(candidate, dict) or not isinstance(candidate.get('evidence'), dict):
        raise ValueError('goal_replay_candidate_evidence_missing')
    evidence = candidate['evidence']; digest = evidence_digest(evidence)
    if candidate.get('evidence_sha256') != digest:
        raise ValueError('goal_replay_evidence_digest_mismatch')
    if (not isinstance(replay, dict) or replay.get('accepted') is not True
            or replay.get('kind') != 'retrospective_candidate_conditioned_training_only'
            or replay.get('candidate_id') != candidate.get('candidate_id')
            or replay.get('evidence_sha256') != digest
            or replay.get('historical_samples_are_held_out') is not False
            or replay.get('fresh_live_validation_required') is not True
            or replay.get('physical_debt_modified') is not False):
        raise ValueError('goal_replay_existing_reference_import_not_accepted')
    if (evidence.get('kind') != 'retrospective_candidate_conditioned'
            or evidence.get('validation_is_prospective') is not False):
        raise ValueError('goal_replay_historical_validation_claim_invalid')
    records = evidence.get('chronological_samples')
    if not isinstance(records, list) or not 4 <= len(records) <= 64:
        raise ValueError('goal_replay_history_sample_count_invalid')
    sequence = [row.get('sequence') for row in records]
    if (any(type(v) is not int for v in sequence) or any(a >= b for a,b in zip(sequence,sequence[1:]))
            or sequence[-1] != evidence.get('current_sequence') or sequence[-1] != replay.get('last_sequence')):
        raise ValueError('goal_replay_history_sequence_invalid')
    key = (candidate.get('candidate_id'), digest, sequence[-1])
    if getattr(monitor, '_v18_replay_applied', None) != key:
        raise ValueError('goal_replay_monitor_import_binding_changed')
    if key in getattr(field, '_v32_goal_replay_applied', set()):
        raise ValueError('goal_replay_already_imported')
    if (field.model.samples or field.held_out or field.sample_positions
            or field.training_updates or field.measured_training_path_m):
        raise ValueError('goal_replay_would_replace_existing_live_evidence')
    rgb, current, joints = _observation(observation)
    binding = hashlib.sha256(rgb.tobytes()+np.asarray(current, '<f8').tobytes()).hexdigest()
    if candidate.get('observation_binding') != binding:
        raise ValueError('goal_replay_current_observation_binding_changed')
    if list(rgb.shape[:2]) != evidence.get('image_shape') or records[-1].get('raw_rgb_sha256') != hashlib.sha256(rgb.tobytes()).hexdigest():
        raise ValueError('goal_replay_current_rgb_changed')
    states = np.asarray([_vector(row.get('state8'),8,'history_state') for row in records])
    if (not np.allclose(states[-1],current,atol=1e-9,rtol=0.)
            or field.state is None or not np.allclose(field.state,current,atol=1e-9,rtol=0.)
            or not np.allclose(monitor.state,current,atol=1e-9,rtol=0.)):
        raise ValueError('goal_replay_current_state_changed')
    if (max(_orientation_distance(s[3:6],current[3:6]) for s in states) > .015+1e-12
            or np.max(abs(states[:,6:]-current[6:])) > .002+1e-12
            or max(np.linalg.norm(s[:3]-current[:3]) for s in states) > .04+1e-12):
        raise ValueError('goal_replay_robot_history_domain_changed')
    if joints is None or field._current_joints is None or not np.allclose(joints,field._current_joints,atol=1e-9,rtol=0.):
        raise ValueError('goal_replay_current_joints_changed')
    history_q = np.asarray([_vector(row.get('joint_position'),7,'history_joints') for row in records])
    if not np.allclose(history_q[-1],joints,atol=1e-9,rtol=0.) or np.max(abs(history_q-joints)) > .12+1e-12:
        raise ValueError('goal_replay_joint_history_domain_changed')
    moving, target = monitor.tracks.get('moving'), monitor.tracks.get('target')
    if (field._track_refs is None or field._track_refs[0] is not moving or field._track_refs[1] is not target
            or moving is None or target is None or moving is target or not moving.reliable or not target.reliable
            or field.current is None or not monitor._external_reliable()):
        raise ValueError('goal_replay_current_track_binding_changed')
    pixels = _gray(rgb)
    if (not np.array_equal(moving.previous_image,pixels) or not np.array_equal(target.previous_image,pixels)
            or not np.array_equal(field.current['moving_px'],moving.position)
            or not np.array_equal(field.current['anchor_px'],target.position)):
        raise ValueError('goal_replay_current_track_observation_changed')
    points = np.asarray([_vector(row.get('candidate_point_px'),2,'candidate_point') for row in records])
    anchors = np.asarray([_vector(row.get('anchor_point_px'),2,'anchor_point') for row in records])
    if (np.any(points < 0) or np.any(anchors < 0) or np.any(points > moving.scale) or np.any(anchors > target.scale)
            or np.linalg.norm(points[-1]-moving.position) > .5
            or np.linalg.norm(anchors[-1]-target.position) > .5
            or np.linalg.norm(_vector(candidate.get('current_point_uv'),2,'candidate_uv')*moving.scale-points[-1]) > .5):
        raise ValueError('goal_replay_candidate_coordinate_binding_changed')
    if any(row.get('candidate_reliable') is not True or row.get('anchor_reliable') is not True for row in records):
        raise ValueError('goal_replay_history_correspondence_unreliable')
    if getattr(monitor,'_pending_prediction',None) is not None:
        raise ValueError('goal_replay_pending_observation_exists')
    model = field._new_model(); positions = []; path = 0.; accumulated = 0.; origin = 0
    relative = points-anchors
    for i in range(1,len(states)):
        accumulated += float(np.linalg.norm(states[i,:3]-states[i-1,:3]))
        delta = states[i,:3]-states[origin,:3]
        if np.linalg.norm(delta) < field.aggregate_distance_m:
            continue
        if np.linalg.norm(delta) >= .65*accumulated:
            model.add(delta,relative[i]-relative[origin])
            positions.append((states[origin,:3].copy(),states[i,:3].copy())); path += accumulated
        origin = i; accumulated = 0.
    if model.matrix is None or len(positions) < model.min_samples:
        raise ValueError('goal_replay_aggregate_training_insufficient')
    details = {'candidate_id':candidate['candidate_id'],'evidence_sha256':digest,
        'reference_epoch':field.epoch,'last_sequence':sequence[-1],'historical_samples':len(records),'training_aggregates':len(positions),
        'training_measured_path_m':path,'model':model.summary(),
        'kind':'retrospective_selected_anatomy_training_only','historical_samples_are_held_out':False,
        'fresh_live_validation_required':True,'anatomical_identity_certified':False,
        'physical_debt_modified':False,'current_goal_or_snapshot_changed':False,
        'observer_observe_called':False,'task_success_certified':False,'clearance_certified':False}
    return key,model,positions,path,details


def warm_start_visual_goal(field, monitor, resolved_candidate, current_obs, reference_replay):
    """Atomically import training only; rejections leave all input objects intact."""
    try:
        key, model, positions, path, report = _prepare(field,monitor,resolved_candidate,current_obs,reference_replay)
    except (ValueError,TypeError,KeyError,AttributeError,IndexError) as error:
        return {'accepted':False,'reason':str(error),'fresh_live_validation_required':True,
                'physical_debt_modified':False,'historical_samples_are_held_out':False,
                'observer_observe_called':False,'clearance_certified':False}
    field.model = model
    field.status = 'visual_objective_retrospective_training_pending_validation'
    field.sample_positions.extend(positions)
    field.training_updates = len(positions)
    field.measured_training_path_m = path
    applied = set(getattr(field,'_v32_goal_replay_applied',set())); applied.add(key)
    field._v32_goal_replay_applied = applied
    report.update(accepted=True,reason='selected_anatomy_goal_response_trained_pending_actual_validation')
    field._v32_goal_replay_report = copy.deepcopy(report)
    return report
