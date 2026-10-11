"""Hybrid render tests use synthetic observations and a memory video writer."""

import json

from PIL import ImageDraw
import pytest

from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.video import WIDTH, HEIGHT
from test_field_history_runner import fake_simulator, observation, rows, sim_config
from test_hybrid_subscription_runner import runner, reply


@pytest.fixture
def memory_writer(monkeypatch):
    texts = []
    original_text = ImageDraw.ImageDraw.text
    def capture_text(self, xy, text, *args, **kwargs):
        texts.append(str(text))
        return original_text(self, xy, text, *args, **kwargs)
    class Writer:
        instances = []
        def __init__(self, path, fps=20):
            self.fps, self.frame_count, self.closed = fps, 0, False
            self.labels, self.sizes = [], []
            self.instances.append(self)
        def write(self, frame, count=1):
            self.labels.append(tuple(texts))
            self.sizes.append(frame.size)
            texts.clear()
            self.frame_count += count
        def close(self):
            self.closed = True
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setattr("gpt_field.video.MP4Stream", Writer)
    return Writer


def test_switches_clear_field_and_never_show_fake_local_plans(tmp_path, observation, memory_writer):
    simulator, _ = fake_simulator(observation)
    output = tmp_path / "smoke"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=simulator, recorder_factory=HybridVideoRecorder)
    events = rows(output / "episode.events.jsonl")
    assert result["diagnostic_complete"]
    assert [e["step"] for e in events if e["event"] == "routing_decision"] == [0, 5, 10]
    assert [e["step"] for e in events if e["event"] == "controller_switch"] == [5, 10]
    assert len([e for e in events if e["event"] == "field_refresh"]) == 2
    assert len([e for e in events if e["event"] == "action_chunk"]) == 1
    assert len([e for e in events if e["event"] == "local_replan"]) == 10
    assert len([e for e in events if e["event"] == "direct_action"]) == 5
    assert len([e for e in events if e["event"] == "executed"]) == 15
    direct_events = [e for e in events if e["controller"] == "contact_direct"]
    for event in direct_events:
        assert event["field_center"] is event["field_step"] is event["selected_index"] is event["candidate_count"] is None
        assert "field_prediction" not in event
        assert event["event"] not in ("field_refresh", "local_replan")
    chunk = next(e for e in direct_events if e["event"] == "action_chunk")
    assert chunk["actions"] == [[0, 0, 0, 0, 0, 0, 1]] * 5 and chunk["action"] == []
    for event in events:
        assert event["routing_source"] == "model_prediction"
        assert event["offline_smoke"] and event["offline_mock"]
        if event["event"] in ("routing_decision", "controller_switch", "field_refresh", "action_chunk"):
            assert event["action"] == [] and event["selected_index"] is None
    returning_field = [e for e in events if e["event"] == "field_refresh"][-1]
    assert returning_field["step"] == returning_field["field_step"] == 10
    assert returning_field["field_center"] == returning_field["state"][:3]
    assert returning_field["field_refresh_count"] == 2 and returning_field["direct_chunk_count"] == 1
    writer = memory_writer.instances[-1]
    assert writer.closed and all(size == (WIDTH, HEIGHT) for size in writer.sizes)
    for event, labels in zip(events, writer.labels):
        text = " ".join(labels)
        assert "SCRIPTED SMOKE" in text and "NO LIVE MODEL CALLS" in text
        assert "model prediction, not contact measurement" in text
        if event["controller"] == "contact_direct":
            assert "SCRIPTED RESPONSE" in text and "MEASURED XY PATH" in text
            assert "RECORDED FIELD + ACTUAL LOCAL PLANNER SEARCH" not in text
            assert "SCRIPTED FIELD + ACTUAL LOCAL PLANNER SEARCH" not in text
    cursor = 0
    for event in events:
        assert event["frame_start"] == cursor and event["video_time_s"] == cursor / 20
        cursor += event["frame_count"]
    assert cursor == writer.frame_count


def test_video_feedback_and_routing_match_recorded_trace(tmp_path, observation, memory_writer):
    simulator, _ = fake_simulator(observation)
    output = tmp_path / "feedback"
    runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        offline_smoke=True, sim_factory=simulator, recorder_factory=HybridVideoRecorder)
    trace, events = rows(output / "trace.jsonl"), rows(output / "episode.events.jsonl")
    actions = [item for item in trace if item["event"] == "action"]
    feedback = [item for item in trace if item["event"] == "feedback"]
    executed = [item for item in events if item["event"] == "executed"]
    for index, event in enumerate(executed):
        assert event["feedback"] == feedback[index]["feedback"]
        assert event["action"] == actions[index]["action"]
        assert event["history_stats"]["feedback_records"] == index + 1
        assert event["history_stats_scope"] == "after_feedback"
        assert event["controller"] == actions[index]["diagnostics"]["controller"]
    for event in events:
        if event["event"] == "routing_decision":
            assert event["hybrid_prediction"] == actions[event["step"]]["diagnostics"]["hybrid_prediction"]


def test_live_mode_success_video_retains_direct_mode_without_field_claim(tmp_path, observation, memory_writer):
    simulator, _ = fake_simulator(observation, success_at=2)
    output = tmp_path / "live_fixture"
    result = runner.execute_episode(output, config=runner.configuration(), sim_config=sim_config(),
        token_provider=lambda: "synthetic-oauth", transport=lambda *args: reply(),
        sim_factory=simulator, recorder_factory=HybridVideoRecorder)
    events = rows(output / "episode.events.jsonl")
    assert result["evaluation_completed"] and events[-1]["status"] == "success" and events[-1]["success"]
    assert all(not event["offline_smoke"] for event in events)
    assert not any(event["event"] in ("field_refresh", "local_replan") for event in events)
    assert all("ASTRA / LIBERO | HYBRID" in " ".join(labels) for labels in memory_writer.instances[-1].labels)
