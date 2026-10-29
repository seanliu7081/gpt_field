"""Train current-reference predictors from previously acknowledged RGB motion.

Candidate selection used this history, so it is retrospective training data,
never held-out validation. No observer ``observe`` method is called. Actual
motion debts, goals, attachment, track identity and blocked states are preserved.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .observation import decode_image
from .visual_monitor_v4 import _orientation_distance


def evidence_digest(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _vector(value, count, label):
    result = np.asarray(value, float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("invalid_"+label)
    return result


def _observation(obs):
    if isinstance(obs, dict):
        image, state, joints = obs["observation/image"], obs["observation/state"], obs.get("observation/joint_position")
    else:
        image, state, joints = obs.image, obs.state, getattr(obs, "joint_position", None)
    rgb = np.asarray(decode_image(image))
    return rgb, _vector(state, 8, "current_state"), None if joints is None else _vector(joints, 7, "current_joints")


def _extent(values):
    values = np.asarray(values)
    return float(np.linalg.norm(values[:, None]-values[None, :], axis=-1).max())


def _prepare(monitor, guard, candidate, observation):
    if not isinstance(candidate, dict) or not isinstance(candidate.get("evidence"), dict):
        raise ValueError("candidate_evidence_missing")
    evidence = candidate["evidence"]
    digest = evidence_digest(evidence)
    if candidate.get("evidence_sha256") != digest:
        raise ValueError("candidate_evidence_digest_mismatch")
    if evidence.get("kind") != "retrospective_candidate_conditioned" or evidence.get("validation_is_prospective") is not False:
        raise ValueError("candidate_evidence_semantics_invalid")
    rgb, current, joints = _observation(observation)
    records = evidence.get("chronological_samples")
    if not isinstance(records, list) or not 4 <= len(records) <= 64:
        raise ValueError("candidate_history_sample_count_invalid")
    if list(rgb.shape[:2]) != list(evidence.get("image_shape", [])):
        raise ValueError("candidate_image_shape_changed")
    sequence = [row.get("sequence") for row in records]
    if any(isinstance(v, bool) or not isinstance(v, int) for v in sequence) or any(b <= a for a, b in zip(sequence, sequence[1:])):
        raise ValueError("candidate_history_not_chronological")
    if sequence[-1] != evidence.get("current_sequence"):
        raise ValueError("candidate_history_endpoint_sequence_mismatch")
    if records[-1].get("raw_rgb_sha256") != hashlib.sha256(rgb.tobytes()).hexdigest():
        raise ValueError("candidate_current_rgb_changed")
    states = np.asarray([_vector(row.get("state8"), 8, "history_state") for row in records])
    if not np.allclose(states[-1], current, atol=1e-9, rtol=0.):
        raise ValueError("candidate_current_robot_state_changed")
    if max(_orientation_distance(row[3:6], current[3:6]) for row in states) > .015+1e-12:
        raise ValueError("candidate_orientation_domain_changed")
    if np.max(abs(states[:, 6:]-current[6:])) > .002+1e-12:
        raise ValueError("candidate_finger_domain_changed")
    if max(np.linalg.norm(row[:3]-current[:3]) for row in states) > .04+1e-12:
        raise ValueError("candidate_position_domain_changed")
    historic_joints = [row.get("joint_position") for row in records]
    if any(row is not None for row in historic_joints):
        if joints is None or any(row is None for row in historic_joints):
            raise ValueError("candidate_joint_history_incomplete")
        array = np.asarray([_vector(row, 7, "history_joints") for row in historic_joints])
        if not np.allclose(array[-1], joints, atol=1e-9, rtol=0.) or np.max(abs(array-joints)) > .12+1e-12:
            raise ValueError("candidate_joint_domain_changed")
    moving = np.asarray([_vector(row.get("candidate_point_px"), 2, "candidate_point") for row in records])
    anchor = np.asarray([_vector(row.get("anchor_point_px"), 2, "anchor_point") for row in records])
    guard_anchor = np.asarray([_vector(row.get("guard_anchor_point_px"), 2, "guard_anchor_point") for row in records])
    noises = np.asarray([float(row.get("noise_px")) for row in records])
    if not np.isfinite(noises).all() or np.any(noises < 0):
        raise ValueError("candidate_noise_invalid")
    for row in records:
        if any(row.get(key) is not True for key in ("candidate_reliable", "anchor_reliable", "guard_anchor_reliable")):
            raise ValueError("candidate_history_correspondence_unreliable")
    anchor_source = evidence.get("guard_anchor_source")
    if anchor_source not in ("explicit_stationary_anchor", "median_persistent_scene_anchors"):
        raise ValueError("candidate_guard_anchor_source_missing")
    for points in (moving, anchor, guard_anchor):
        if np.any(points < 0) or np.any(points > np.array(rgb.shape[:2][::-1])-1):
            raise ValueError("candidate_history_point_outside_image")
    # A named stationary anchor is a hypothesis with bounded observed motion;
    # no metric pose or environment calibration is consumed here.
    if max(_extent(anchor), _extent(guard_anchor)) > max(1., 3*float(noises.max())):
        raise ValueError("candidate_stationary_anchor_motion_unresolved")
    ext = (getattr(monitor, "refs", {}) or {}).get("external") or {}
    track = (getattr(monitor, "tracks", {}) or {}).get("moving")
    target = (getattr(monitor, "tracks", {}) or {}).get("target")
    if ext.get("moving_kind") != "tool" or track is None or target is None or not track.reliable or not target.reliable:
        raise ValueError("candidate_current_tool_anchor_unavailable")
    if guard.tool_track is not track or getattr(guard, "_response_track", None) is not track:
        raise ValueError("candidate_guard_reference_not_configured")
    if np.linalg.norm(moving[-1]-track.position) > .5 or np.linalg.norm(anchor[-1]-target.position) > .5:
        raise ValueError("candidate_current_reference_point_changed")
    current_uv = _vector(candidate.get("current_point_uv"), 2, "candidate_current_uv")
    if np.linalg.norm(current_uv*np.asarray(track.scale)-moving[-1]) > .5:
        raise ValueError("candidate_current_seed_changed")
    if not np.allclose(monitor.state, current, atol=1e-9, rtol=0.) or not np.allclose(guard.state, current, atol=1e-9, rtol=0.):
        raise ValueError("candidate_observers_not_at_current_state")
    if (getattr(monitor, "_pending_prediction", None) is not None
            or getattr(guard, "_experiment_pending", None) is not None
            or getattr(guard, "_motion_classification_pending", None) is not None):
        raise ValueError("candidate_import_during_pending_transition")
    key = (candidate.get("candidate_id"), digest, sequence[-1])
    if getattr(monitor, "_v18_replay_applied", None) == key or getattr(guard, "_v18_replay_applied", None) == key:
        raise ValueError("candidate_replay_already_applied")
    relative, guard_relative = moving-anchor, moving-guard_anchor
    # ``noise_px`` is the resolver's consensus point-precision heuristic,
    # matching the live observer (scatter / sqrt(inliers), with a floor).
    # This is a training screen, not a calibrated confidence/clearance bound.
    threshold = max(.35, 3*float(noises.max()))
    if min(_extent(relative), _extent(guard_relative)) < threshold:
        raise ValueError("candidate_relative_motion_unresolved")
    monitor_model, guard_model, cumulative = monitor._new_model(), guard._model(), guard._cumulative_model()
    training_path = 0.
    for i in range(1, len(states)):
        delta = states[i,:3]-states[i-1,:3]
        if np.linalg.norm(delta) <= .00005:
            continue
        training_path += float(np.linalg.norm(delta))
        monitor_model.add(delta, relative[i]-relative[i-1])
        guard_model.add(delta, guard_relative[i]-guard_relative[i-1])
    origin = 0
    for i in range(1, len(states)):
        delta = states[i,:3]-states[origin,:3]
        if np.linalg.norm(delta) >= .001-1e-12:
            cumulative.add(delta, guard_relative[i]-guard_relative[origin])
            origin = i
    if (training_path < .003-1e-12 or any(model.matrix is None or np.linalg.norm(model.matrix) < 20.
            for model in (monitor_model, guard_model, cumulative))):
        raise ValueError("candidate_response_training_insufficient")
    details = {"candidate_id": candidate.get("candidate_id"), "evidence_sha256": digest,
        "kind": "retrospective_candidate_conditioned_training_only", "sample_count": len(records),
        "first_sequence": sequence[0], "last_sequence": sequence[-1], "training_measured_path_m": training_path,
        "relative_image_extent_px": _extent(relative), "guard_relative_image_extent_px": _extent(guard_relative),
        "maximum_point_precision_proxy_px": float(noises.max()),
        "training_image_motion_threshold_px": threshold,
        "noise_is_calibrated_bound": False,
        "guard_anchor_source": anchor_source, "monitor_training": monitor_model.summary(),
        "guard_training": guard_model.summary(), "guard_aggregate_training": cumulative.summary(),
        "historical_samples_are_held_out": False, "fresh_live_validation_required": True,
        "semantic_identity_certified": False, "attachment_certified": False, "clearance_certified": False,
        "physical_debt_modified": False, "observer_observe_called": False, "task_success_certified": False}
    return key, monitor_model, guard_model, cumulative, details


def warm_start_reference_response(monitor, guard, resolved_candidate, current_obs):
    """Atomically install training only; a rejection leaves both objects intact.

    Call AFTER both observers are configured on the model-confirmed current
    seed, BEFORE any action selection. Existing live observation methods alone
    can generate new held-out tests. Blocked references are never unblocked.
    """
    try:
        key, model, guard_model, cumulative, details = _prepare(monitor, guard, resolved_candidate, current_obs)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return {"accepted": False, "reason": str(error), "fresh_live_validation_required": True,
                "physical_debt_modified": False, "observer_observe_called": False,
                "clearance_certified": False, "task_success_certified": False}
    monitor.response, guard.response, guard.cumulative_response = model, guard_model, cumulative
    monitor.validation_count = 0
    monitor._held_out_motion = []
    monitor._prediction_window.clear()
    monitor._pending_prediction = None
    monitor._mapping_had_supported_response = False
    guard.validations = 0
    guard.prediction_errors = []
    guard._response_error.clear()
    # The next actual observation starts a new aggregate. Do not import a
    # historical gap/response snapshot or count its motion a second time.
    guard._response_origin = None
    guard._response_age = 0
    monitor._v18_replay_applied = guard._v18_replay_applied = key
    details.update(accepted=True, reason="candidate_predictors_warm_started_pending_actual_validation")
    return details
