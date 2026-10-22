"""Real recorder/rendering integration; only the MP4 byte sink is replaced."""

import hashlib
import json
from pathlib import Path

import pytest

from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.observation import PolicyObservation
from gpt_field.policy import Decision
from gpt_field.video import HEIGHT, WIDTH
from test_field_history_runner import observation, rows
from test_speed_first_v2_runner import run, v2_intent


@pytest.fixture
def writer_sink(monkeypatch):
    class Writer:
        instances = []

        def __init__(self, path, *, fps=20):
            self.path = Path(path)
            self.fps = fps
            self.frame_count = 0
            self.closed = False
            self.calls = []
            self.last_frame = None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"UNIT TEST FRAME SINK; NOT AN ENCODED VIDEO")
            self.instances.append(self)

        def write(self, frame, count=1):
            assert not self.closed
            assert frame.mode == "RGB" and frame.size == (WIDTH, HEIGHT)
            assert type(count) is int and count > 0
            self.calls.append({"count": count, "sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            self.last_frame = frame.copy()
            self.frame_count += count

        def close(self):
            self.closed = True

    # decision(), render_frame(), PIL drawing, event writing, and runner hooks
    # all execute as shipped. No recorder method or renderer is mocked.
    monkeypatch.setattr("gpt_field.video.MP4Stream", Writer)
    return Writer


def test_actual_recorder_mixed_operation_direct_five_field_preserves_event_meaning(
        tmp_path, observation, writer_sink):
    direct = v2_intent(direct=True, phase="verify")
    direct["direct"]["actions"] = [[.01 * (i + 1), -.02, 0., 0., 0., 0., 1.] for i in range(5)]
    output = tmp_path / "mixed"
    result, transport, instances = run(output, observation, [
        v2_intent(operation="close", phase="grasp"), direct,
        v2_intent(phase="recover", goal=(0., -.03, 0.)),
    ], success_at=21, recorder=HybridVideoRecorder)
    assert result["status"] == "success", result["error"]
    assert result["measured_feedback_count"] == result["steps"] == 21
    assert result["local_gripper_actions"] == 15
    assert result["ordinary_direct_actions"] == 5
    assert result["coarse_field_actions"] == 1
    assert len(transport.calls) == 3 and instances[0].closed
    events = rows(output / "episode.events.jsonl")
    trace_actions = [r for r in rows(output / "trace.jsonl") if r["event"] == "action"]
    assert [r["step"] for r in events if r["event"] == "gripper_operation"] == [0]
    assert [r["step"] for r in events if r["event"] == "gripper_action"] == list(range(15))
    assert [r["step"] for r in events if r["event"] == "action_chunk"] == [15]
    assert [r["step"] for r in events if r["event"] == "direct_action"] == list(range(15, 20))
    assert [r["step"] for r in events if r["event"] == "field_refresh"] == [20]
    operation_events = [r for r in events if r.get("execution_kind") == "local_gripper_settle"]
    assert operation_events
    for event in operation_events:
        assert "actions" not in event
        assert event["local_gripper"]["command"] == "close"
        assert not event["local_gripper"]["task_success_certified"]
        assert event["field_center"] is event["field_step"] is event["candidate_count"] is None
        assert event["event"] != "action_chunk"
        if event["event"] == "gripper_action":
            assert event["action"] == trace_actions[event["step"]]["action"]
            assert event["action"][6] == 1.
    direct_chunk = next(r for r in events if r["event"] == "action_chunk")
    assert direct_chunk["actions"] == direct["direct"]["actions"]
    assert direct_chunk["action"] == []
    executed = [r for r in events if r["event"] == "executed"]
    assert len(executed) == 21
    for event, action in zip(executed, trace_actions):
        assert event["action"] == action["action"]
        assert event["feedback_step"] == action["step"]
        assert event["history_stats"]["feedback_records"] == action["step"] + 1
    assert events[-1]["status"] == "success" and events[-1]["success"] is True
    writer = writer_sink.instances[-1]
    assert writer.closed and writer.frame_count == sum(r["frame_count"] for r in events)
    assert len(writer.calls) == len(events)
    assert len({r["sha256"] for r in writer.calls}) > 3


@pytest.mark.parametrize("success_at", [2, 5])
def test_real_runner_and_recorder_stop_on_libero_inside_operation_without_fabricated_completion(
        tmp_path, observation, writer_sink, success_at):
    output = tmp_path / "early_success"
    result, transport, instances = run(output, observation,
        [v2_intent(operation="close", phase="grasp")], success_at=success_at,
        recorder=HybridVideoRecorder)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == success_at
    assert result["libero_success"] is result["task_success"] is True
    assert result["model_declared_complete"] is False
    assert result["local_gripper_actions"] == success_at
    assert len(transport.calls) == 1 and len(instances[0].actions) == success_at
    events = rows(output / "episode.events.jsonl")
    assert len([r for r in events if r["event"] == "gripper_action"]) == success_at
    assert not any(r["event"] in ("action_chunk", "direct_action", "model_completion") for r in events)
    terminal = events[-1]
    assert terminal["event"] == "terminal"
    assert terminal["status"] == "success" and terminal["success"] is True
    assert terminal["execution_kind"] == "local_gripper_settle"
    assert terminal["local_gripper"]["done"] is False
    assert terminal["local_gripper"]["task_success_certified"] is False
    assert writer_sink.instances[-1].closed


def test_real_recorder_open_operation_has_no_stale_direct_action_array(tmp_path, observation, writer_sink):
    output = tmp_path / "open"
    direct = v2_intent(direct=True, phase="align")
    result, _, _ = run(output, observation,
        [direct, v2_intent(operation="open", phase="release")], success_at=7,
        recorder=HybridVideoRecorder)
    assert result["status"] == "success", result["error"]
    events = rows(output / "episode.events.jsonl")
    assert [r["step"] for r in events if r["event"] == "action_chunk"] == [0]
    assert [r["step"] for r in events if r["event"] == "gripper_operation"] == [5]
    for event in events:
        if event.get("execution_kind") != "local_gripper_settle":
            continue
        assert "actions" not in event
        assert event["local_gripper"]["command"] == "open"
        if event["action"]:
            assert event["action"][6] == -1.


def test_archived_pending_step54_decision_renders_without_prediction_actions(tmp_path, writer_sink):
    source = Path(__file__).resolve().parents[1] / "runs/speed_first_v2_bowl_zero_shot_001/trial/trace.jsonl"
    if not source.exists():
        pytest.skip("archived local trial fixture is unavailable")
    records = rows(source)
    before = next(r for r in records if r["event"] == "executed" and r["step"] == 53)
    pending = next(r for r in records if r["event"] == "action" and r["step"] == 54)
    assert pending["diagnostics"]["execution_kind"] == "local_gripper_settle"
    assert "prediction" not in pending["diagnostics"]
    assert pending["diagnostics"]["hybrid_prediction"]["direct"] is None
    actual = PolicyObservation.from_dict(before["observation"])
    recorder = HybridVideoRecorder(tmp_path / "archived_step54.mp4")
    recorder.observe(actual, step=54)
    recorder.decision(actual, Decision(tuple(pending["action"]), pending["diagnostics"]),
                      decision_latency_s=pending["decision_latency_s"])
    assert recorder.chunk == []
    assert recorder.action == tuple(pending["action"])
    frame = recorder.render_frame()
    assert frame.mode == "RGB" and frame.size == (WIDTH, HEIGHT)
    frame.save(tmp_path / "archived_step54.png")
    recorder.close(status="offline_archived_decision_rendered")
    events = rows(tmp_path / "archived_step54.events.jsonl")
    action = next(r for r in events if r["event"] == "gripper_action")
    assert action["step"] == 54 and action["action"] == pending["action"]
    assert action["local_gripper"] == pending["diagnostics"]["local_gripper"]
    assert not any(r["event"] in ("action_chunk", "direct_action", "executed") for r in events)
    assert all("actions" not in r for r in events)
    assert writer_sink.instances[-1].closed
