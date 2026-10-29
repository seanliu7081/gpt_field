"""Offline replay audit fixtures; no real robot, authentication or model calls."""

import importlib.util
import json
from pathlib import Path

import pytest

from test_distance_policy import intent
from test_distance_runner import SETTINGS
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_distance_control_tests", ROOT / "scripts/audit_distance_control.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def recorded_fixture(tmp_path, observation, *, stop=False, verify=False, real_video=False,
                     interrupt_request=False):
    """Use the real runner/policy with explicit in-memory transport and simulator."""
    def before_step(step, action):
        if stop and step == 7:
            raise KeyboardInterrupt("synthetic user interruption")

    sim, _ = fake_simulator(observation, success_at=None if stop else 7, before_step=before_step)
    values = ([intent(phase="verify", hold=True), intent(direct=True, phase="recover")]
              if verify else [intent(), intent(direct=True, phase="grasp"), intent(hold=True)])
    responses = [envelope(value, id=f"audit-fixture-{i}", model="gpt-6-astra", reasoning={"effort": "medium"})
                 for i, value in enumerate(values)]
    if interrupt_request:
        responses[1] = KeyboardInterrupt("synthetic interruption waiting for response")
    transport = FakeTransport(*responses)
    run = tmp_path / "run"
    runner.execute_episode(run, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-fixture-oauth", transport=transport,
        sim_factory=sim, recorder_factory=None if real_video else FakeRecorder, **SETTINGS)
    return run


@pytest.fixture
def skip_fake_video(monkeypatch):
    # This isolates controller replay. The recorder is deliberately non-video;
    # the production auditor's full decoder is separately used for real MP4s.
    def check(run, summary, actions, executed, feedback, audit):
        audit.notes.append("Test harness substituted video decoding for the declared mock recorder.")
        return {"synthetic_mock_recorder": True}, 0
    monkeypatch.setattr(auditor._legacy, "_video_audit", check)


def rewrite(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def rewrite_trace(run, records):
    (run / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))


def test_reconstructs_all_distance_direct_and_boundary_history(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    report = auditor.audit_run(run, tmp_path / "audit", allow_offline_smoke=False)
    assert report["passed"], report["issues"]
    assert report["executed_actions"] == report["feedback_records"] == 7
    assert report["replayed_recorded_requests"] == 3
    assert report["distance_event_reasons"] == {"goal_reached": 1, "field_budget_exhausted": 1}
    assert report["shared_code_reconstruction"] and report["synthetic_test_fixture"]
    assert not report["physical_object_state_independently_verified"]
    assert not report["real_task_completion_verified"]
    assert report["live_model_calls_by_audit"] == report["network_requests_by_audit"] == 0
    assert report["simulator_actions_by_audit"] == 0 and not report["credentials_loaded_by_audit"]


def test_replays_local_view_recovery_without_model_or_completion(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation, verify=True)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["replayed_recorded_requests"] == 2
    assert report["distance_event_reasons"] == {"placement_unverified": 1}
    trace = rows(run / "trace.jsonl")
    actions = [row for row in trace if row["event"] == "action"]
    assert actions[5]["diagnostics"]["local_recovery_active"]
    assert actions[5]["diagnostics"]["local_field_revision"] == 1


def test_tampered_command_is_rejected(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    trace = rows(run / "trace.jsonl")
    action = next(row for row in trace if row["event"] == "action")
    action["action"][0] += .01
    rewrite_trace(run, trace)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("exact action replay" in issue for issue in report["issues"])


def test_fake_success_without_evaluator_signal_is_rejected(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    trace = rows(run / "trace.jsonl")
    for row in trace:
        if row["event"] == "executed":
            row["success"] = False
            row["sim_status"]["success"] = False
    rewrite_trace(run, trace)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("success status requires actual evaluator evidence" in issue for issue in report["issues"])


def test_user_stop_remains_incomplete_and_unacknowledged_proposal_is_not_execution(
        tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation, stop=True)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["status"] == "user_stop" and not report["evaluation_completed"]
    assert report["executed_actions"] == 7 and not report["trace_complete"]
    assert report["execution_evidence"]["unacknowledged_action_index"] == 7
    assert not report["libero_success_observed"]
    assert any("not a successful completion time" in note for note in report["notes"])


def test_payload_or_raw_response_tampering_cannot_hide_in_local_diagnostics(
        tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    request = sorted((run / "requests").glob("*.request.json"))[0]
    value = json.loads(request.read_text())
    value["payload"]["input"].append({"role": "user", "content": "hidden object pose"})
    rewrite(request, value)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("exact full RGB/state/history/schema payload" in issue for issue in report["issues"])


def test_new_audit_requires_separate_output_and_explicit_protocol(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    with pytest.raises(ValueError, match="separate"):
        auditor.audit_run(run, run / "audit")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["configuration"]["policy"]["distance_field_control"] = False
    rewrite(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="explicit v7"):
        auditor.audit_run(run, tmp_path / "audit")


def test_frozen_source_drift_is_reported(tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation)
    archived = run / "frozen_source" / "distance_policy.py"
    archived.write_text(archived.read_text() + "\n# changed archived source\n")
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]
    assert any("recorded frozen source hash: distance_policy.py" in issue for issue in report["issues"])


def test_interrupted_next_model_request_is_reconstructed_without_fictitious_action(
        tmp_path, observation, skip_fake_video):
    run = recorded_fixture(tmp_path, observation, interrupt_request=True)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["status"] == "user_stop" and not report["evaluation_completed"]
    assert report["executed_actions"] == 1 and report["trace_complete"]
    assert report["replayed_recorded_requests"] == 2 and report["accepted_recorded_responses"] == 1
    assert report["usage_totals"]["unknown_usage_calls"] == 1


@pytest.mark.parametrize("target", ["plan", "visual", "event", "completion"])
def test_semantic_diagnostics_and_boundary_tampering_are_rejected(
        tmp_path, observation, skip_fake_video, target):
    run = recorded_fixture(tmp_path, observation)
    trace = rows(run / "trace.jsonl")
    action = next(row for row in trace if row["event"] == "action")
    if target == "plan":
        action["diagnostics"]["plan"]["search"]["start_cost_m"] += .001
    elif target == "visual":
        next(row for row in trace if row["event"] == "local_visual_feedback")["feedback"]["reliable"] = True
    elif target == "event":
        next(row for row in trace if row["event"] == "distance_control_event")["distance_control_event"]["reason"] = "no_progress"
    else:
        action["diagnostics"]["task_success_certified"] = True
    rewrite_trace(run, trace)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert not report["passed"]


def test_original_encoded_video_and_full_replay_agree(tmp_path, observation):
    run = recorded_fixture(tmp_path, observation, real_video=True)
    report = auditor.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["video"]["codec_name"] == "h264"
    assert int(report["video"]["nb_read_frames"]) == report["event_frames"] > 0
    dispatched = auditor._legacy.audit_run(run, tmp_path / "legacy_dispatch_audit")
    assert dispatched["passed"], dispatched["issues"]
    assert dispatched["kind"] == "distance_control_recorded_shared_code_audit"
