"""Audit checks use actual parser/policy/trackers, with no model or simulator."""
from dataclasses import replace
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "audit_v4", Path(__file__).parents[1] / "scripts/audit_visual_field_v4.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def archived_trace(path, *, static_feature=True, tamper_observation=False):
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    obs = audit.fixture_observation(state)
    prediction = audit.fixture_prediction(state)
    prediction["visual_references"]["external"].update(
        moving_footprint_uv={"lower_uv": [.1, .1], "upper_uv": [.9, .9]},
        blocked_boxes_uv=[{"object_id": "legacy_overlap", "lower_uv": [0., 0.], "upper_uv": [1., 1.]}],
        unknown_boxes_uv=[])
    raw = obs.to_dict()
    if tamper_observation:
        raw["object_poses"] = "must_not_enter_policy"
    records = [{"event": "reset", "observation": raw,
                "sim_status": {"success": True, "object_poses": "evaluator-only-secret"}}]
    for step in range(4):
        action = [.2, 0., 0., 0., 0., 0., 0.]
        records.append({"event": "action", "step": step, "action": action,
            "diagnostics": {"prediction": prediction if step == 0 else None}})
        state[0] += .0025
        after = replace(obs, state=tuple(state)) if static_feature else audit.fixture_observation(state)
        records.append({"event": "executed", "step": step, "action": action,
                        "observation": after.to_dict(), "sim_status": {"contact": "private-contact"}})
    path.write_text("".join(json.dumps(record)+"\n" for record in records))
    return records


def test_twenty_real_policy_field_actions_amortize_one_fake_response_over_direct5():
    result = audit.synthetic_call_amortization()
    assert result["hybrid"]["moving_actions"] == result["hybrid"]["field"]["effective_moving_actions"] == 20
    assert result["hybrid"]["fake_model_requests"] == 1
    assert result["direct5"]["fake_model_requests"] == 4
    assert result["request_reduction_fraction"] == .75
    assert result["hybrid"]["monitor"]["response_validated"]
    assert not result["hybrid"]["monitor"]["wrist_target_reliable"]
    measured = result["hybrid"]["field"]["measured_travel_m"]
    assert .035 < measured < .055
    assert result["direct5"]["measured_distance_m"] == pytest.approx(measured)
    assert result["real_model_calls"] == result["real_robot_or_simulator_steps"] == 0
    assert result["live_speedup_measured"] is False and result["actual_task_success"] is None
    assert result["hybrid"]["local_policy_processing_s"] > 0
    # Includes history flushing through the subscription contract: catches numpy
    # booleans leaking into actual JSON before a live task can hit that failure.
    json.dumps(result, allow_nan=False)


def test_archived_motion_exposes_stationary_wrong_feature_without_evaluator_input(tmp_path):
    path = tmp_path / "trace.jsonl"
    archived_trace(path)
    before = path.read_bytes()
    result = audit.replay_visual_evidence(path)
    assert path.read_bytes() == before
    assert result["transitions"] == 4
    assert result["rows"][0]["before_visual_gate_reason"] is None  # Bounded bootstrap.
    assert result["rows"][-1]["after_visual_gate_reason"] == "controlled_feature_unresponsive"
    assert result["rows"][-1]["monitor"]["controlled_feature_displacement_px"] == 0
    assert not result["rows"][-1]["monitor"]["response_validated"]
    assert result["executed_v4_actions"] == 0
    assert not result["metric_field_geometry_supplied"]
    assert not result["task_success_established"]
    serialized = json.dumps(result, allow_nan=False)
    assert "evaluator-only-secret" not in serialized and "private-contact" not in serialized


def test_reference_conversion_preserves_landmarks_but_makes_no_metric_mask_conversion():
    ref = audit.fixture_prediction(np.zeros(8))["visual_references"]
    ref["external"]["unknown_boxes_uv"] = [{"id": "old_image_only"}]
    ref["wrist"] = {"target_id": "fixture_target", "target_anchor_uv": [.3, .4],
                    "target_point_uv": [.4, .5], "moving_footprint_uv": "cropped"}
    original = copy.deepcopy(ref)
    result = audit.role_references(ref)
    assert ref == original
    assert result["external"]["moving_point_uv"] == ref["external"]["moving_point_uv"]
    assert set(result["external"]) == audit.EXTERNAL_KEYS
    assert result["wrist"] == {"target_id": "fixture_target", "target_anchor_uv": [.3, .4],
                               "desired_target_uv": [.4, .5]}


def test_privileged_observation_is_rejected_not_silently_stripped(tmp_path):
    path = tmp_path / "trace.jsonl"
    archived_trace(path, tamper_observation=True)
    with pytest.raises(ValueError, match="allowlisted"):
        audit.replay_visual_evidence(path)


def test_unacknowledged_or_mismatched_archived_action_is_rejected(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = archived_trace(path)
    records[2]["action"] = [0.] * 7
    path.write_text("".join(json.dumps(record)+"\n" for record in records))
    with pytest.raises(ValueError, match="does not match"):
        audit.replay_visual_evidence(path)


def test_fresh_artifact_hashes_input_and_cannot_overwrite_earlier_evidence(tmp_path):
    run = tmp_path / "archive"; run.mkdir()
    trace = run / "trace.jsonl"; archived_trace(trace)
    digest = audit.sha256(trace)
    output = tmp_path / "audit"
    report = audit.audit_run(run, output)
    assert report["input"]["trace_sha256"] == digest == audit.sha256(trace)
    metadata = json.loads((output / "artifact.json").read_text())
    assert metadata["audit_sha256"] == audit.sha256(output / "audit.json")
    assert report["live_trials_started"] == report["subscription_inference_calls"] == report["api_key_calls"] == 0
    with pytest.raises(FileExistsError):
        audit.audit_run(run, output)
    with pytest.raises(FileExistsError):
        audit.audit_run(run, run / "forbidden-new-data")


@pytest.mark.parametrize("steps", [5, 11, 30, True])
def test_matched_direct5_fixture_requires_complete_chunks(steps):
    with pytest.raises(ValueError):
        audit.synthetic_call_amortization(steps)
