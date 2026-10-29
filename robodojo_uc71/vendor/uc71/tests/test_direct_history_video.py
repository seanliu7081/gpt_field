"""Verify video provenance for direct actions rather than field planning."""

import json

import numpy as np
import pytest

from gpt_field.direct_history_video import DirectHistoryVideoRecorder
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.policy import Decision
from gpt_field.video import HEIGHT, WIDTH


class MemoryWriter:
    def __init__(self, path, fps=20):
        self.fps, self.frame_count, self.closed = fps, 0, False
        self.last_frame = None

    def write(self, frame, count=1):
        self.last_frame = frame
        self.frame_count += count

    def close(self):
        self.closed = True


def observation(x=0.0):
    first = np.full((224, 224, 3), (150, 0, 0), dtype=np.uint8)
    second = np.full((224, 224, 3), (0, 0, 150), dtype=np.uint8)
    return PolicyObservation(image_to_data_url(first, rotate=False),
                             image_to_data_url(second, rotate=False),
                             (x, 0.1, 0.9, 0, 0, 0, 0.04, -0.04),
                             "pick up the black bowl between the plate and the ramekin and place it on the plate")


@pytest.mark.parametrize("offline_smoke", [True, False])
def test_direct_video_records_actual_chunks_actions_and_measured_feedback(tmp_path, monkeypatch, offline_smoke):
    monkeypatch.setattr("gpt_field.video.MP4Stream", MemoryWriter)
    video = DirectHistoryVideoRecorder(tmp_path / "video.mp4", offline_smoke=offline_smoke)
    before, after = observation(), observation(0.002)
    video.observe(before)
    assert video.writer.last_frame.size == (WIDTH, HEIGHT)
    assert video.writer.last_frame.getpixel((210, 350)) == (150, 0, 0)
    assert video.writer.last_frame.getpixel((210, 740)) == (0, 0, 150)
    actions = [[0.2, 0, 0, 0, 0, 0, 1]] * 5
    decision = Decision(tuple(actions[0]), {
        "step": 0, "method": "direct_history", "refreshed": True,
        "prediction": {"actions": actions}, "chunk_step": 0,
        "history_stats": {"completed_chunks": 0},
        "api_latency_s": 2.5, "api_request_key": "request-1",
    })
    video.decision(before, decision, decision_latency_s=2.6)
    feedback = {"commanded_delta_xyz_m": [0.01, 0, 0], "measured_delta_xyz_m": [0.002, 0, 0]}
    video.executed(after, step=0, success=False, feedback=feedback)
    video.close(status="offline_smoke_complete" if offline_smoke else "budget_exceeded")
    events = [json.loads(line) for line in video.events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["observation", "action_chunk", "direct_action", "executed", "terminal"]
    assert all(event["offline_smoke"] == offline_smoke for event in events)
    assert all("field_refresh_count" not in event and "selected_index" not in event for event in events)
    assert events[1]["action"] == []
    assert events[1]["actions"] == actions
    assert events[2]["action"] == actions[0]
    assert events[2]["api_latency_s"] == 2.5
    assert events[3]["feedback"] == feedback
    assert events[3]["state"] == list(after.state)
    assert events[3]["measured_delta_xyz_m"] == pytest.approx([0.002, 0, 0])
    assert not events[-1]["success"]
    assert events[-1]["frame_start"] + events[-1]["frame_count"] == video.frame_count
    assert video.writer.closed


def test_failure_before_first_action_finalizes_observed_video(tmp_path, monkeypatch):
    monkeypatch.setattr("gpt_field.video.MP4Stream", MemoryWriter)
    video = DirectHistoryVideoRecorder(tmp_path / "video.mp4")
    video.observe(observation())
    video.close(status="api_failure", error="request failed")
    video.close(status="api_failure")
    events = [json.loads(line) for line in video.events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["observation", "terminal"]
    assert all(event["chunk_count"] == 0 for event in events)
    assert video.writer.closed
