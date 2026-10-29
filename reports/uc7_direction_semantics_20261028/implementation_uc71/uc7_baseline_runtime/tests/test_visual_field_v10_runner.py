"""V10 fake-client -> real-policy -> runner boundaries and durable timing prefix."""
import copy
import json

import pytest

from test_visual_field_v7_protocol import stream_error, current_record, current_images
from test_visual_field_v8_runner import fixtures, gripper_request as request_v8, runner, run_args
from test_visual_field_v10_protocol import prediction


def gripper_request():
    value = request_v8()
    value["subgoal_update"] = {"action": "none", "subgoal_id": None, "reason": "No active navigation subgoal."}
    return value


def checkpoints(monkeypatch):
    original = runner.write_json
    captured = []

    def write(path, value):
        if path.name != "run_state.json.tmp":
            assert path.name != "run_state.json", "V10 must publish checkpoints by atomic replacement"
            return original(path, value)
        previous_path = path.with_name("run_state.json")
        previous = previous_path.read_bytes() if previous_path.exists() else None
        if previous is not None:
            json.loads(previous)  # The old checkpoint is always a complete JSON record.
        original(path, value)
        assert json.loads(path.read_text()) == value
        if previous is not None:
            assert previous_path.read_bytes() == previous
        captured.append(copy.deepcopy(value))

    monkeypatch.setattr(runner, "write_json", write)
    return captured


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupt"])
def test_v10_evaluator_priority_same_observation_retry_and_finalized_video_excluded_time(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    snapshots = checkpoints(monkeypatch)
    Sim, Recorder, transport, sims = fixtures(clock, [stream_error(), gripper_request()],
        outcome=outcome, terminal_step=8, request_action_counts=[0, 0])
    config = run_args(tmp_path, method_version="v10")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert result["method_version"] == "v10" and result["steps"] == 8
    assert result["gripper_actions"] == 8 and result["field_actions"] == result["direct_actions"] == 0
    assert result["status"] == {"success": "success", "collision": "collision_failure",
        "simultaneous": "collision_failure", "interrupt": "user_stop"}[outcome]
    assert result["task_success"] is result["completion_time_eligible"] is (outcome == "success")
    assert result["libero_success"] is (outcome in ("success", "simultaneous"))
    assert len(transport.calls) == result["budget"]["api_calls"] == result["policy_diagnostic_counters"]["model_requests"] == 2
    assert result["budget"]["unknown_usage_calls"] == 1 and result["budget"]["api_key_calls"] == 0
    records = [current_record(call) for call in transport.calls]
    assert records[0]["observation_index"] == records[1]["observation_index"] == 0
    assert records[0]["observation/state"] == records[1]["observation/state"]
    assert current_images(transport.calls[0]) == current_images(transport.calls[1])
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert all(p["model"] == "gpt-6-astra" and p["reasoning"] == {"effort": "medium"} for p in transport.calls)
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["stage"] == "finished"
    assert final["timing_components"] == result["timing_components"]
    assert final["elapsed_control_s"] == result["control_elapsed_s"] == 20.
    assert final["control_without_video_s"] == result["control_without_video_s"] == 16.
    assert final["timing_components"]["video_render_s"] == 4.
    assert final["timing_components"]["model_pipeline_s"] == 14.
    observed = [item for item in snapshots if item["stage"] == "observed"]
    assert len(observed) == 8 and all(not item["finalized"] for item in observed)
    assert observed[-1]["control_without_video_s"] == 16.
    assert all("observed prefix" in item["timing_scope"] for item in snapshots)
    assert not (config.output / "run_state.json.tmp").exists()
    manifest = json.loads((config.output / "manifest.json").read_text())
    assert manifest["method_version"] == "v10" and manifest["shots"] == 0
    assert manifest["max_policy_steps"] is manifest["max_api_calls"] is manifest["max_wall_time_s"] is None
    assert "persistent world goal" in manifest["subgoal_reuse"]
    assert "atomic run_state" in manifest["timing_checkpoint"]
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("outcome", ["interrupt", "success", "collision", "simultaneous"])
def test_operator_stop_waits_for_complete_feedback_and_yields_to_evaluator_outcome(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, [gripper_request()], outcome=outcome, terminal_step=1)
    config = run_args(tmp_path, method_version="v10")

    class StopAfterStep(Sim):
        def step(self, action):
            value = super().step(action)
            (config.output / "operator_stop.request.json").write_text('{"reason":"operator-reviewed boundary stop"}')
            return value

    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=StopAfterStep, recorder_factory=Recorder)
    rows = [json.loads(line) for line in (config.output / "trace.jsonl").read_text().splitlines()]
    for event in ("action", "executed", "feedback"):
        assert sum(row["event"] == event for row in rows) == 1
    assert len(transport.calls) == result["steps"] == 1
    if outcome == "interrupt":
        assert result["status"] == "user_stop" and result["error"]["type"] == "OperatorStopFile"
        assert rows[-1]["event"] == "operator_stop"
        assert rows[-1]["completed_actions"] == 1
    else:
        assert result["status"] == ("success" if outcome == "success" else "collision_failure")
        assert not any(row["event"] == "operator_stop" for row in rows)
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["step"] == 1
    assert final["control_without_video_s"] == result["control_without_video_s"] == 7.25
    assert final["timing_components"]["video_render_s"] == .5
    assert sims[0].closed and not result["cleanup_errors"]


def test_pending_request_preserves_exact_completed_prefix_until_final_failed_request_accounting(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    snapshots = checkpoints(monkeypatch)
    first = prediction()
    first["visual_references"] = {"external": None, "wrist": None}
    first["direct_actions"] = [[.01, 0., 0., 0., 0., 0., -1.]] * 5
    failure = stream_error(code="invalid_api_key", status=401)
    Sim, Recorder, transport, sims = fixtures(clock, [first, failure], outcome="interrupt",
        terminal_step=99, request_action_counts=[0, 5])
    config = run_args(tmp_path, method_version="v10")
    pending = []

    class InspectPending:
        def __call__(self, *args):
            if len(transport.calls) != 1:
                return transport(*args)
            path = config.output / "run_state.json"
            before = path.read_bytes()
            try:
                return transport(*args)
            finally:
                assert path.read_bytes() == before
                pending.append((json.loads(before), clock[0]))

    class ClosingRecorder(Recorder):
        def close(self):
            clock[0] += 11.  # Final encoding belongs to cleanup, outside control time.

    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=InspectPending(),
                            sim_factory=Sim, recorder_factory=ClosingRecorder)
    assert len(pending) == 1
    prefix, now = pending[0]
    assert prefix["stage"] == "deciding" and prefix["step"] == 5 and not prefix["finalized"]
    assert prefix["elapsed_control_s"] == 10.75
    assert prefix["timing_components"]["video_render_s"] == 2.5
    assert prefix["control_without_video_s"] == 8.25
    assert prefix["timing_components"]["model_pipeline_s"] == 7.
    assert prefix["budget"]["api_calls"] == 1  # Second request began after the checkpoint.
    assert now == 118.25  # 100 + initial video .5 + prefix10.75 + pending model7.
    observed = [item for item in snapshots if item["stage"] == "observed"][-1]
    assert observed["step"] == 5 and observed["control_without_video_s"] == 8.25
    assert result["status"] == "subscription_error" and result["steps"] == 5
    assert len(transport.calls) == result["budget"]["api_calls"] == 2
    assert result["budget"]["api_key_calls"] == 0
    assert result["control_elapsed_s"] == 17.75 and result["control_without_video_s"] == 15.25
    assert result["wall_elapsed_s"] == 29.25
    assert result["timing_components"]["decision_s"] == 14.
    assert result["timing_components"]["failed_decision_s"] == 7.
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["control_without_video_s"] == 15.25
    assert final["timing_components"] == result["timing_components"]
    assert final["budget"]["api_calls"] == 2
    assert not result["completion_time_eligible"] and not (config.output / "run_state.json.tmp").exists()
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("change", [{"controller": "direct"}, {"demo": "unused.json"}])
def test_v10_refuses_unrequested_baseline_or_demonstration(tmp_path, change):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(run_args(tmp_path, method_version="v10", **change),
                       token_provider=lambda: pytest.fail("Must not authorize"))
