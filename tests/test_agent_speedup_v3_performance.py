"""Offline contracts for recording overhead; no simulator, encoder or model."""
import importlib.util
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from gpt_field.agent_speedup_v3_performance import (
    LiveSnapshotGate, adapt_private_evaluator, completion_target,
)
from gpt_field.agent_speedup_v3_video import AgentSpeedUpV3VideoRecorder, CANVAS


def test_live_snapshot_gate_throttles_only_status_and_uses_compact_stats():
    clock = SimpleNamespace(now=0.)
    gate = LiveSnapshotGate(clock=lambda: clock.now)
    assert gate.due("starting")
    assert not gate.due("deciding")
    clock.now = 1.
    assert gate.due("observed")
    assert not gate.due("executing")
    assert gate.due("finalizing_video")
    class Policy:
        step_index = 8
        def agent_stats(self):
            raise AssertionError("full history must never be built for live status")
        def fast_stats(self):
            return {"step": 8, "phase": "move"}
    assert gate.policy_stats(Policy()) == {"step": 8, "phase": "move"}
    assert gate.policy_stats(SimpleNamespace(step_index=9))["step"] == 9


def test_private_adapter_preserves_every_action_and_terminal_full_stats(tmp_path):
    source = tmp_path / "fake_evaluator.py"
    source.write_text('''
records = []
def write_json(path, value):
    records.append((path, value))
def _append(log, value):
    log.append(value)
def guarded(label, function):
    return function()
def execute(policy):
    stage = "initial"
    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json("run_state", {"stage": stage, "agent": guarded("stats", policy.agent_stats)})
    state("starting")
    actions = []
    for step in range(5):
        state("deciding")
        _append(actions, {"step": step, "action": [0, 0, 0, 0, 0, 0, -1]})
        state("observed")
    state("finished")
    return actions, policy.agent_stats(), stage
''')
    spec = importlib.util.spec_from_file_location("private_test_evaluator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.execute
    counts = {"full": 0, "fast": 0}
    class Policy:
        def agent_stats(self):
            counts["full"] += 1
            return {"full_history": list(range(5))}
        def fast_stats(self):
            counts["fast"] += 1
            return {"compact": True}
    adapt_private_evaluator(module, clock=lambda: 0.)
    actions, final, stage = module.execute(Policy())
    assert module.execute is not original
    assert [row["step"] for row in actions] == list(range(5))
    assert counts == {"full": 1, "fast": 2}
    assert final == {"full_history": list(range(5))} and stage == "finished"
    assert [row[1]["stage"] for row in module.records] == ["starting", "finished"]
    assert module._v3_recording_timings.counts["trace_appends"] == 5
    assert module._v3_live_snapshot_gate.skipped == 10
    changed = module.execute
    adapt_private_evaluator(module)
    assert module.execute is changed


@pytest.mark.parametrize("seconds,success,expected", [(275., True, True), (276., True, False),
                                                       (1., False, False)])
def test_completion_target_includes_prepare_and_never_confuses_timeout_with_success(seconds, success, expected):
    report = completion_target({"user_check": {"preparation_elapsed_s": 25.},
                                "libero_success": success, "task_success": success}, seconds)
    assert report["total_service_elapsed_s"] == seconds+25.
    assert report["achieved"] is expected and report["hard_stop"] is False
    assert completion_target({"libero_success": True, "task_success": True}, 1.)["achieved"] is False


def test_compact_recorder_indexes_all_decisions_reuses_rgb_and_drains(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.)
    decoded = []
    def decode(url):
        decoded.append(url)
        return Image.new("RGB", (256, 256), "blue")
    monkeypatch.setattr("gpt_field.agent_speedup_v3_video.decode_image", decode)
    class Writer:
        def __init__(self, path, fps):
            self.fps, self.frame_count, self.closed = fps, 0, False
        def write(self, frame, count):
            assert frame.size == CANVAS and count == 1
            self.frame_count += count
        def close(self):
            clock.now += 2.
            self.closed = True
    monkeypatch.setattr(AgentSpeedUpV3VideoRecorder, "writer_class", Writer)
    path = tmp_path / "episode.mp4"
    recorder = AgentSpeedUpV3VideoRecorder(path, clock=lambda: clock.now)
    obs = SimpleNamespace(image="external", wrist_image="wrist", prompt="Move the object")
    decision = SimpleNamespace(action=[0., 0., 0., 0., 0., 0., -1.], diagnostics={
        "step": 0, "primitive": "move_to_pose", "api_calls": [{"kind": "agent", "latency_s": 2.}],
        "routing": {"params": {"motion_context": {"intent": "transit"}}}})
    recorder.observe(obs)
    recorder.decision(obs, decision)
    assert recorder.frame_count == 1
    recorder.executed(obs, step=0, success=True)
    recorder.close(status="success")
    assert recorder.frame_count == 3  # initial + one executed action + terminal
    assert decoded == ["external", "wrist"]
    assert recorder.writer.closed
    events = [json.loads(line) for line in path.with_suffix(".events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["observation", "decision", "executed", "terminal"]
    assert events[1]["action"] == decision.action and events[1]["frame_rendered"] is False
    assert events[2]["success"] is True
    timing = json.loads(path.with_suffix(".timing.json").read_text())
    assert timing["close_drain_s"] == 2. and timing["executed_events"] == 1
    recorder.close(status="success")
    assert clock.now == 2.


def test_encoder_failure_still_records_drain_and_propagates(tmp_path, monkeypatch):
    class Writer:
        def __init__(self, path, fps):
            self.frame_count = 0
        def close(self):
            raise RuntimeError("encoder failed")
    monkeypatch.setattr(AgentSpeedUpV3VideoRecorder, "writer_class", Writer)
    recorder = AgentSpeedUpV3VideoRecorder(tmp_path / "episode.mp4")
    with pytest.raises(RuntimeError, match="encoder failed"):
        recorder.close(status="success")
    assert recorder.closed and (tmp_path / "episode.timing.json").exists()
