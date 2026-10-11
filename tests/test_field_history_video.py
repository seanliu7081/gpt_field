"""Field-history video provenance using real local plans and recorded fixtures."""

from copy import deepcopy
import json

import numpy as np
from PIL import ImageDraw
import pytest

from gpt_field.field_history_video import FieldHistoryVideoRecorder
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.policy import Decision, FieldPolicy
from gpt_field.schema import GridSpec
from gpt_field.video import EpisodeVideoRecorder, HEIGHT, WIDTH, YELLOW


GRID = GridSpec(size=3, spacing_m=0.1)


def observation(x=0):
    first = np.full((224, 224, 3), (140, 0, 0), dtype=np.uint8)
    second = np.full((224, 224, 3), (0, 0, 140), dtype=np.uint8)
    return PolicyObservation(image_to_data_url(first, rotate=False), image_to_data_url(second, rotate=False),
                             (x, 0, 0.5, 0, 0, 0, 0.04, -0.04), "put the black bowl on the plate")


class Predictor:
    def predict(self, observation, grid, mode="field"):
        return {"stage": "approach_black_bowl", "orientation_target_xyzw": [0, 0, 0, 1],
                "gripper_command": "hold", "motion_mode": "move",
                "values": (-grid.coordinates((0, 0, 0))[:, 0]).tolist(),
                "blocked_indices": [0], "unknown_indices": [grid.count-1]}


def decision(before, policy=None, feedback=None):
    policy = policy or FieldPolicy(Predictor(), grid=GRID, planner_config=PlannerConfig(horizon=1))
    value = policy.act(before)
    value.diagnostics.update(method="field_history", history_stats={"observations": 1, "feedback_records": 0,
                              "retained_image_pairs": 1, "messages": 2}, feedback=deepcopy(feedback))
    if value.diagnostics["refreshed"]:
        value.diagnostics.update(api_latency_s=2.4, api_request_key="recorded-request", api_request_log="api/recorded.request.json")
    return value


def feedback(before, action, after):
    desired = [0.05*v for v in action[:3]]
    measured = [after.state[i]-before.state[i] for i in range(3)]
    return {"type": "execution_feedback", "step": 0, "before_state": list(before.state),
            "after_state": list(after.state), "normalized_action": list(action),
            "desired_translation_delta_m": desired, "measured_translation_delta_m": measured,
            "translation_tracking_error_m": float(np.linalg.norm(np.asarray(measured)-desired)),
            "gripper_joint_delta": [0, 0], "rolling_progress": {"window_size": 1,
                "low_motion_translation_steps": 0, "translation_command_steps": 1}}


@pytest.fixture
def memory_writer(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        texts.append(str(text))
        return original(self, xy, text, *args, **kwargs)

    class Writer:
        def __init__(self, path, fps=20):
            self.fps, self.frame_count, self.closed = fps, 0, False
            self.frames, self.labels = [], []

        def write(self, frame, count=1):
            self.frames.append(frame.copy())
            self.labels.append(tuple(texts))
            texts.clear()
            self.frame_count += count

        def close(self):
            self.closed = True

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setattr("gpt_field.video.MP4Stream", Writer)
    return Writer


def events(video):
    return [json.loads(line) for line in video.events_path.read_text().splitlines()]


@pytest.mark.parametrize("offline_smoke", [False, True])
def test_records_true_fields_plans_feedback_and_current_history(tmp_path, memory_writer, offline_smoke):
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4", grid=GRID, offline_smoke=offline_smoke)
    before, after = observation(), observation(0.002)
    value = decision(before)
    measured = feedback(before, value.action, after)
    history = {"observations": 1, "feedback_records": 1, "retained_image_pairs": 1, "messages": 3}
    video.observe(before)
    video.decision(before, value, decision_latency_s=2.5)
    video.executed(after, step=0, success=False, feedback=measured, history_stats=history,
                   sim_status={"steps": 1, "success": False})
    video.close(status="offline_smoke_complete" if offline_smoke else "budget_exceeded")
    rows = events(video)
    assert [r["event"] for r in rows] == ["observation", "field_refresh", "local_replan", "executed", "terminal"]
    assert all(r["policy"] == "field_history" and r["offline_smoke"] is offline_smoke and r["offline_mock"] is offline_smoke for r in rows)
    assert rows[1]["field_prediction"] == value.diagnostics["field_prediction"]
    assert rows[1]["action"] == [] and rows[1]["selected_index"] is None
    assert rows[1]["candidate_count"] is None
    assert rows[2]["action"] == list(value.action)
    assert rows[2]["selected_index"] == value.diagnostics["plan"]["selected_index"]
    assert rows[2]["history_stats_scope"] == "decision"
    assert rows[3]["history_stats_scope"] == "after_feedback"
    assert rows[3]["history_stats"] == history and rows[3]["feedback"] == measured
    assert rows[3]["feedback_step"] == 0 and rows[3]["measured_delta_xyz_m"] == pytest.approx([0.002, 0, 0])
    assert rows[3]["state"] == list(after.state)
    assert rows[3]["sim_status"] == {"steps": 1, "success": False}
    assert rows[2]["api_latency_s"] == 2.4 and rows[2]["decision_latency_s"] == 2.5
    assert rows[-1]["frame_count"] == 60
    assert rows[-1]["frame_start"] + rows[-1]["frame_count"] == video.frame_count
    assert video.frame_count == 89  # Event holds only; API waiting time is not fabricated.
    assert video.writer.closed
    assert video.writer.frames[0].size == (WIDTH, HEIGHT)
    assert video.writer.frames[0].getpixel((210, 350)) == (140, 0, 0)
    assert video.writer.frames[0].getpixel((210, 740)) == (0, 0, 140)
    if offline_smoke:
        for frame_text in video.writer.labels:
            joined = " ".join(frame_text)
            assert "SCRIPTED SMOKE" in joined and "NO LIVE MODEL CALLS" in joined
            assert "ASTRA" not in joined.upper()
    else:
        assert all("ASTRA / LIBERO" in " ".join(text) for text in video.writer.labels)
    assert "History after feedback: 1 observations | 1 feedback reports" in " ".join(video.writer.labels[3])
    assert "Latest feedback, executed step 0" in " ".join(video.writer.labels[3])


def test_world_fixed_field_geometry_is_unchanged_and_plan_absent_on_refresh(tmp_path, memory_writer):
    before = observation()
    value = decision(before)
    reference = EpisodeVideoRecorder(tmp_path/"reference.mp4", grid=GRID)
    video = FieldHistoryVideoRecorder(tmp_path/"history.mp4", grid=GRID)
    for item in (reference, video):
        item.observe(before)
        item.decision(before, value)
    # Test actual rendered pixels for all three world-coordinate field planes.
    for box in ((479, 245, 767, 533), (870, 245, 1158, 533), (1261, 245, 1549, 533)):
        assert np.array_equal(np.asarray(reference.writer.frames[1].crop(box)), np.asarray(video.writer.frames[1].crop(box)))
        assert np.array_equal(np.asarray(reference.writer.frames[2].crop(box)), np.asarray(video.writer.frames[2].crop(box)))
    first = np.asarray(video.writer.frames[1].crop((479, 245, 767, 533)))
    planned = np.asarray(video.writer.frames[2].crop((479, 245, 767, 533)))
    assert not np.any(np.all(first == YELLOW, axis=-1))
    assert np.any(np.all(planned == YELLOW, axis=-1))
    assert video.field.center == (0, 0, 0.5)
    for item in (reference, video):
        item.close(status="test_complete")


def test_cached_replan_keeps_recorded_field_and_labels_previous_feedback(tmp_path, memory_writer):
    policy = FieldPolicy(Predictor(), grid=GRID, planner_config=PlannerConfig(horizon=1))
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4", grid=GRID)
    before, after = observation(), observation(0.002)
    video.observe(before)
    first = decision(before, policy)
    video.decision(before, first)
    measured = feedback(before, first.action, after)
    video.executed(after, step=0, success=False, feedback=measured)
    second = decision(after, policy, measured)
    second.diagnostics["history_stats"]["feedback_records"] = 1
    video.decision(after, second)
    assert video.field.center == (0, 0, 0.5) and video.field_step == 0
    assert video.refresh_count == 1 and video.last_api_request_key == "recorded-request"
    assert events(video)[-1]["event"] == "local_replan"
    assert events(video)[-1]["feedback_step"] == 0
    assert events(video)[-1]["measured_delta_xyz_m"] is None
    assert "Latest feedback, executed step 0" in " ".join(video.writer.labels[-1])
    assert "History at decision" in " ".join(video.writer.labels[-1])
    video.close(status="budget_exceeded")


def test_missing_field_on_refresh_clears_stale_geometry(tmp_path, memory_writer):
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4", grid=GRID, offline_mock=True)
    video.observe(observation())
    video.decision(observation(), decision(observation()))
    assert video.field is not None
    video.decision(observation(), Decision((0,)*7, {"method": "field_history", "step": 5, "refreshed": True}))
    assert video.field is None
    assert events(video)[-2]["field_prediction"] is None
    assert events(video)[-2]["field_center"] is None
    assert "No scripted field recorded" in " ".join(video.writer.labels[-1])
    video.close(status="offline_smoke_complete")


def test_feedback_and_history_are_recorded_snapshots_not_live_mutable_references(tmp_path, memory_writer):
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4", grid=GRID)
    before, after = observation(), observation(0.002)
    value = decision(before)
    video.observe(before)
    video.decision(before, value)
    measured = feedback(before, value.action, after)
    history = {"observations": 1, "feedback_records": 1, "retained_image_pairs": 1, "messages": 3}
    video.executed(after, step=0, success=False, feedback=measured, history_stats=history)
    measured["measured_translation_delta_m"][0] = 999
    history["feedback_records"] = 999
    video.close(status="budget_exceeded")
    assert events(video)[-1]["feedback"]["measured_translation_delta_m"][0] == pytest.approx(0.002)
    assert events(video)[-1]["history_stats"]["feedback_records"] == 1


@pytest.mark.parametrize("offline_smoke", [False, True])
def test_terminal_before_any_decision_and_idempotent_close(tmp_path, memory_writer, offline_smoke):
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4", task="recorded task", offline_smoke=offline_smoke)
    assert video.grid == GridSpec()
    video.observe(observation())
    video.close(status="infrastructure_error", error="recorded transport error")
    count = video.frame_count
    video.close(status="infrastructure_error")
    assert video.frame_count == count == 72
    rows = events(video)
    assert [r["event"] for r in rows] == ["observation", "terminal"]
    assert all(r["history_stats"] is None and r["feedback"] is None and r["field_refresh_count"] == 0 for r in rows)
    assert not rows[-1]["success"]
    if offline_smoke:
        for texts in video.writer.labels:
            assert "SCRIPTED SMOKE" in " ".join(texts) and "ASTRA" not in " ".join(texts).upper()


def test_conflicting_provenance_flags_fail_before_creating_artifacts(tmp_path):
    with pytest.raises(ValueError, match="labels conflict"):
        FieldHistoryVideoRecorder(tmp_path/"video.mp4", offline_mock=True, offline_smoke=False)
    assert not list(tmp_path.iterdir())


def test_success_at_initial_observation_is_not_lost(tmp_path, memory_writer):
    video = FieldHistoryVideoRecorder(tmp_path/"video.mp4")
    video.observe(observation())
    video.close(status="success")
    assert events(video)[-1]["success"] is True
