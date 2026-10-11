"""Video provenance, coordinate alignment, and a real encoder smoke test."""

import json
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.policy import Decision, FieldPolicy
from gpt_field.schema import GridSpec
from gpt_field.video import EpisodeVideoRecorder, HEIGHT, MP4Stream, WIDTH, project_world


GRID = GridSpec(size=3, spacing_m=0.1)


def observation(x=0.0):
    red = np.zeros((224, 224, 3), dtype=np.uint8)
    red[..., 0] = 120
    blue = np.zeros_like(red)
    blue[..., 2] = 120
    return PolicyObservation(image_to_data_url(red, rotate=False), image_to_data_url(blue, rotate=False),
                             (x, 0, 0.5, 0, 0, 0, 0.04, -0.04), "put the bowl on the plate")


class Predictor:
    def predict(self, observation, grid, mode="field"):
        return {"stage": "approach", "orientation_target_xyzw": [0, 0, 0, 1],
                "gripper_command": "hold", "motion_mode": "move",
                "values": (-grid.coordinates((0, 0, 0))[:, 0]).tolist(),
                "blocked_indices": [0], "unknown_indices": [grid.count - 1]}


class MemoryWriter:
    def __init__(self, path, fps=20):
        self.fps, self.frame_count, self.closed = fps, 0, False
        self.last_frame = None

    def write(self, frame, count=1):
        self.last_frame = frame
        self.frame_count += count

    def close(self):
        self.closed = True


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr("gpt_field.video.MP4Stream", MemoryWriter)
    video = EpisodeVideoRecorder(tmp_path / "episode.mp4", grid=GRID, episode_id="test episode")
    yield video
    video.close(status="timeout")


def test_world_projection_matches_grid_vertices_and_flips_only_vertical_axis():
    lower = np.array((-0.08, -0.04, 0.42))
    upper = lower + 0.16
    center = (lower + upper) / 2
    for axes in ((0, 1), (0, 2), (1, 2)):
        actual = project_world([lower, center, upper], lower, upper, axes, 101)
        np.testing.assert_allclose(actual, [[0, 100], [50, 50], [100, 0]])
    shifted = center.copy()
    shifted[0] += 0.01
    actual = project_world([center, shifted], lower, upper, (0, 1), 101)
    np.testing.assert_allclose(actual[1] - actual[0], [6.25, 0])


def test_field_only_appears_from_recorded_refresh_and_stays_world_fixed(recorder):
    policy = FieldPolicy(Predictor(), grid=GRID, planner_config=PlannerConfig(horizon=1))
    recorder.observe(observation())
    assert recorder.field is None
    assert recorder.writer.last_frame.size == (WIDTH, HEIGHT)
    # The two camera panels retain different real observations.
    assert recorder.writer.last_frame.getpixel((210, 350)) == (120, 0, 0)
    assert recorder.writer.last_frame.getpixel((210, 740)) == (0, 0, 120)
    first = policy.act(observation())
    first.diagnostics.update(api_latency_s=2.4, api_request_key="request-1", api_request_log="request-1.json")
    recorder.decision(observation(), first, decision_latency_s=2.6)
    assert recorder.field.center == (0, 0, 0.5)
    assert recorder.refresh_count == 1
    assert recorder.field_step == 0
    recorder.executed(observation(0.004), step=0, success=False)
    second = policy.act(observation(0.004))
    recorder.decision(observation(0.004), second, decision_latency_s=0.01)
    assert recorder.field.center == (0, 0, 0.5)
    assert recorder.refresh_count == 1
    assert recorder.last_api_request_key == "request-1"
    # A real refresh without a field must not be represented using stale data.
    recorder.decision(observation(), Decision((0,) * 7, {"step": 2, "refreshed": True}))
    assert recorder.field is None


def test_event_index_preserves_real_order_actions_latency_and_success(recorder):
    policy = FieldPolicy(Predictor(), grid=GRID, planner_config=PlannerConfig(horizon=1))
    initial, final = observation(), observation(0.004)
    recorder.observe(initial)
    decision = policy.act(initial)
    decision.diagnostics["api_latency_s"] = 2.5
    recorder.decision(initial, decision, decision_latency_s=2.6)
    recorder.executed(final, step=0, success=True, sim_status={"success": True, "terminated": True})
    recorder.close(status="success")
    events = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["observation", "field_refresh", "local_replan", "executed", "terminal"]
    assert events[1]["action"] == []
    assert events[1]["selected_index"] is None
    assert events[2]["action"] == list(decision.action)
    assert events[2]["selected_index"] == decision.diagnostics["plan"]["selected_index"]
    assert events[2]["decision_latency_s"] == 2.6
    assert events[2]["api_latency_s"] == 2.5
    assert events[3]["state"] == list(final.state)
    assert events[3]["success"] is True
    assert events[4]["status"] == "success"
    assert events[4]["frame_count"] == 60  # Three seconds terminal hold.
    assert events[-1]["frame_start"] + events[-1]["frame_count"] == recorder.frame_count
    assert recorder.writer.closed
    count = recorder.frame_count
    recorder.close(status="success")
    assert recorder.frame_count == count


def test_api_failure_after_initial_observation_still_has_terminal_video(recorder):
    recorder.observe(observation())
    recorder.close(status="api_failure", error="recorded transport error")
    assert recorder.field is None
    events = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["observation", "terminal"]
    assert events[-1]["status"] == "api_failure"
    assert events[-1]["api_latency_s"] is None


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg")
def test_real_mp4_encoding_is_finalized_and_frame_count_is_exact(tmp_path):
    path = tmp_path / "episode.mp4"
    writer = MP4Stream(path, fps=20)
    frame = Image.new("RGB", (WIDTH, HEIGHT), (12, 30, 55))
    writer.write(frame, 3)
    writer.close()
    assert path.is_file()
    assert not path.with_suffix(".partial.mp4").exists()
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_name,width,height,nb_read_frames,r_frame_rate", "-of", "json", str(path),
    ]))
    stream = info["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
    assert stream["nb_read_frames"] == "3"
    assert stream["r_frame_rate"] == "20/1"
    with pytest.raises(FileExistsError):
        MP4Stream(path)


def test_success_at_reset_has_true_terminal_success(recorder):
    recorder.observe(observation())
    recorder.close(status="success")
    events = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
    assert events[-1]["event"] == "terminal"
    assert events[-1]["success"] is True
