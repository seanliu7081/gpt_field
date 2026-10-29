"""V9 preserves evaluator priority, subscription timing and stop boundaries."""
import json

import pytest

from test_visual_field_v8_runner import fixtures, gripper_request, stream_error, runner, run_args


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous"])
def test_subscription_retry_and_independent_evaluator_with_video_exclusion(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, [stream_error(), gripper_request()],
        outcome=outcome, terminal_step=8, request_action_counts=[0, 0])
    config = run_args(tmp_path, method_version="v9")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert result["steps"] == 8 and result["method_version"] == "v9"
    assert result["task_success"] is (outcome == "success")
    assert result["status"] == ("success" if outcome == "success" else "collision_failure")
    assert result["budget"]["api_calls"] == 2 and result["budget"]["api_key_calls"] == 0
    assert result["control_without_video_s"] == 16.
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert all(p["reasoning"] == {"effort": "medium"} for p in transport.calls)
    manifest = json.loads((config.output/"manifest.json").read_text())
    assert "field_geometry_semantics" in manifest and manifest["max_policy_steps"] is None
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("outcome", ["interrupt", "success", "collision", "simultaneous"])
def test_operator_stop_requested_during_execution_waits_for_feedback_and_evaluator(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, [gripper_request()], outcome=outcome, terminal_step=1)
    config = run_args(tmp_path, method_version="v9")

    class StopAfterStep(Sim):
        def step(self, action):
            result = super().step(action)
            (config.output/"operator_stop.request.json").write_text('{"reason":"reviewed test stop"}')
            return result

    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=StopAfterStep, recorder_factory=Recorder)
    rows = [json.loads(line) for line in (config.output/"trace.jsonl").read_text().splitlines()]
    for kind in ("action", "executed", "feedback"):
        assert sum(row["event"] == kind for row in rows) == 1
    assert len(transport.calls) == 1 and result["steps"] == 1
    if outcome == "interrupt":
        assert result["status"] == "user_stop" and result["error"]["type"] == "OperatorStopFile"
        assert rows[-1]["event"] == "operator_stop"
    else:
        assert result["status"] == ("success" if outcome == "success" else "collision_failure")
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("change", [{"controller": "direct"}, {"demo": "unused.json"}])
def test_v9_refuses_unrequested_baseline_or_demonstration(tmp_path, change):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(run_args(tmp_path, method_version="v9", **change),
                       token_provider=lambda: pytest.fail("Must not authorize"))
