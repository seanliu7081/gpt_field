"""Hybrid render tests use synthetic observations and a memory video writer."""

import json

from PIL import ImageDraw
import pytest

from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.policy import Decision
from gpt_field.schema import GridSpec
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
    assert [e["step"] for e in events if e["event"] == "routing_decision"] == [0, 30, 35]
    assert [e["step"] for e in events if e["event"] == "controller_switch"] == [30, 35]
    assert len([e for e in events if e["event"] == "field_refresh"]) == 2
    assert len([e for e in events if e["event"] == "action_chunk"]) == 1
    assert len([e for e in events if e["event"] == "local_replan"]) == 6
    assert len([e for e in events if e["event"] == "field_action"]) == 54
    assert [e["step"] for e in events if e["event"] == "local_replan"] == [0, 10, 20, 35, 45, 55]
    assert len([e for e in events if e["event"] == "direct_action"]) == 5
    assert len([e for e in events if e["event"] == "executed"]) == 65
    direct_events = [e for e in events if e["controller"] == "contact_direct"]
    for event in direct_events:
        assert event["field_center"] is event["field_step"] is event["selected_index"] is event["candidate_count"] is None
        assert "field_prediction" not in event
        assert event["event"] not in ("field_refresh", "local_replan", "field_action")
    chunk = next(e for e in direct_events if e["event"] == "action_chunk")
    assert chunk["actions"] == [[0, 0, 0, 0, 0, 0, 1]] * 5 and chunk["action"] == []
    for event in events:
        assert event["routing_source"] == "model_prediction"
        assert event["update_steps"] == (5 if event["controller"] == "contact_direct" else 30)
        assert event["grid"] == runner.HYBRID_GRID.to_dict()
        assert event["offline_smoke"] and event["offline_mock"]
        if event["event"] in ("routing_decision", "controller_switch", "field_refresh", "action_chunk"):
            assert event["action"] == [] and event["selected_index"] is None
    returning_field = [e for e in events if e["event"] == "field_refresh"][-1]
    assert returning_field["step"] == returning_field["field_step"] == 35
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
            assert "5 actions, 0.25 s simulated time" in text
            assert "Controller block: 5 actions, 0.25 s simulated time" in text
            start = event["chunk_step"] // 5 * 5
            assert f"showing {start + 1}-{start + 5}" in text
            action_rows = [label for label in labels if label.startswith(tuple(f"{i}   [" for i in range(1, 6)))]
            assert len(action_rows) == 5
            assert any(label.startswith(f"{event['chunk_step'] + 1}   [") for label in action_rows)
            assert "RECORDED FIELD + ACTUAL LOCAL PLANNER SEARCH" not in text
            assert "SCRIPTED FIELD + ACTUAL LOCAL PLANNER SEARCH" not in text
        elif event["controller"] == "coarse_field":
            assert "Controller block: 30 actions, 1.5 s simulated time" in text
            if event["event"] in ("local_replan", "field_action", "executed"):
                assert f"Plan step {event['plan_step'] + 1}/10; planned at step {event['plan_start_step']}" in text
                if event["event"] != "executed":
                    assert "Current planned delta m:" in text
                    assert "First planned delta m:" not in text
                assert event["plan_refreshed"] == (event["plan_step"] == 0)
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
        details = actions[index]["diagnostics"]
        assert event["controller"] == details["controller"]
        if event["controller"] == "coarse_field":
            for key in ("plan_refreshed", "plan_start_step", "plan_step", "plan_execution_steps", "planned_delta_xyz"):
                assert event[key] == details[key]
            plan_events = [e for e in events if e["step"] == index and e["event"] in ("local_replan", "field_action")]
            assert len(plan_events) == 1
            assert plan_events[0]["event"] == ("local_replan" if details["plan_refreshed"] else "field_action")
            assert event["plan_start_step"] == index - event["plan_step"]
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


def test_compressed_search_labels_finalists_separately_from_valid_paths(tmp_path, observation, memory_writer):
    recorder = HybridVideoRecorder(tmp_path / "finalists.mp4", grid=runner.HYBRID_GRID)
    recorder.observe(observation)
    envelope = json.loads(reply("coarse_field")["output"][0]["content"][0]["text"])
    start = list(observation.state[:3])
    points = [start]
    for index in range(10):
        point = points[-1].copy()
        point[0 if index < 8 else 1] += .01
        points.append(point)
    search = {"method": "exact_lattice_dp", "candidate_semantics": "terminal_band_finalists",
              "nominal_candidate_count": 60466176, "represented_valid_sequence_count": 1000000}
    plan = {"candidate_count": 1, "valid_count": 1, "selected_index": 0,
            "delta_xyz": [.01, 0, 0], "search": search,
            "candidates": [{"valid": True, "positions": points}]}
    details = {"step": 0, "controller": "coarse_field", "routing": {
        key: envelope[key] for key in ("controller", "target_region", "near_target", "routing_reason")},
        "refreshed": True, "field_center": start, "field_prediction": envelope["field"],
        "plan": plan, "status": "move", "hybrid_prediction": envelope}
    recorder.decision(observation, Decision((.2, 0, 0, 0, 0, 0, 1), details))
    recorder.close(status="fixture_complete")
    events = rows(tmp_path / "finalists.events.jsonl")
    local = next(event for event in events if event["event"] == "local_replan")
    assert local["planner_search"] == search
    labels = memory_writer.instances[-1].labels[-2]
    assert "search finalists" in labels
    assert "Exact DP: 1 finalists; 1,000,000 valid paths" in labels


@pytest.mark.parametrize("grid, count", [(GridSpec(), 5), (runner.HYBRID_GRID, 30)])
def test_legacy_direct_recordings_infer_interval_and_keep_final_command_visible(
        tmp_path, observation, memory_writer, grid, count):
    recorder = HybridVideoRecorder(tmp_path / "legacy.mp4", grid=grid)
    recorder.observe(observation)
    action = [0, 0, 0, 0, 0, 0, 1]
    routing = {"controller": "contact_direct", "target_region": "black_bowl", "near_target": True}
    details = {"step": 0, "controller": "contact_direct", "routing": routing,
               "refreshed": True, "prediction": {"actions": [action] * count}}
    recorder.decision(observation, Decision(tuple(action), details))
    recorder.decision(observation, Decision(tuple(action), {
        "step": count - 1, "controller": "contact_direct", "routing": routing, "refreshed": False}))
    recorder.close(status="fixture_complete")
    events = rows(tmp_path / "legacy.events.jsonl")
    assert all(event["update_steps"] == count for event in events if event["controller"] is not None)
    labels = memory_writer.instances[-1].labels[-2]
    text = " ".join(labels)
    assert f"{count} actions, {count / 20:g} s simulated time" in text
    assert f"showing {count - 4}-{count}" in text
    assert any(label.startswith(f"{count}   [") for label in labels)


def test_cached_turn_has_field_action_event_and_current_delta_label(tmp_path, observation, memory_writer):
    recorder = HybridVideoRecorder(tmp_path / "cached.mp4", grid=runner.HYBRID_GRID)
    recorder.observe(observation)
    envelope = json.loads(reply("coarse_field")["output"][0]["content"][0]["text"])
    start = list(observation.state[:3])
    points = [start]
    for index in range(10):
        point = points[-1].copy()
        point[0 if index < 8 else 1] += .01
        points.append(point)
    plan = {"candidate_count": 1, "valid_count": 1, "selected_index": 0,
            "delta_xyz": [.01, 0, 0], "candidates": [{"valid": True, "positions": points}]}
    routing = {key: envelope[key] for key in ("controller", "target_region", "near_target", "routing_reason")}
    common = {"controller": "coarse_field", "routing": routing, "field_center": start,
              "plan": plan, "status": "move", "plan_start_step": 0, "plan_execution_steps": 10}
    recorder.decision(observation, Decision((.2, 0, 0, 0, 0, 0, 1), {
        **common, "step": 0, "refreshed": True, "field_prediction": envelope["field"],
        "hybrid_prediction": envelope, "plan_refreshed": True, "plan_step": 0,
        "planned_delta_xyz": [.01, 0, 0]}))
    recorder.decision(observation, Decision((0, .2, 0, 0, 0, 0, 1), {
        **common, "step": 8, "refreshed": False, "plan_refreshed": False,
        "plan_step": 8, "planned_delta_xyz": [0, .01, 0]}))
    recorder.close(status="fixture_complete")
    events = rows(tmp_path / "cached.events.jsonl")
    assert [e["event"] for e in events if e["event"] in ("local_replan", "field_action")] == ["local_replan", "field_action"]
    cached = next(e for e in events if e["event"] == "field_action")
    assert cached["plan_refreshed"] is False
    assert cached["plan_step"] == 8 and cached["plan_start_step"] == 0
    assert cached["plan_execution_steps"] == 10
    assert cached["planned_delta_xyz"] == [0, .01, 0]
    labels = memory_writer.instances[-1].labels[-2]
    assert any("Plan step 9/10; planned at step 0" in label for label in labels)
    delta = next(label for label in labels if label.startswith("Current planned delta m:"))
    assert "+0.0000, +0.0100, +0.0000" in delta
    assert "current EEF / square: current move" in labels
    for event, event_labels in zip(events, memory_writer.instances[-1].labels):
        if event["event"] in ("routing_decision", "field_refresh"):
            assert not any(label.startswith("Current planned delta m:") for label in event_labels)
            assert "plan_step" not in event


@pytest.mark.parametrize("controller", ["coarse_field", "contact_direct"])
def test_model_completion_video_has_no_proposed_or_executed_action(
        tmp_path, observation, memory_writer, controller):
    recorder = HybridVideoRecorder(tmp_path / "complete.mp4")
    recorder.observe(observation)
    reason = "Bowl released on plate, gripper clear."
    details = {
        "controller": controller, "routing": {"controller": controller, "target_region": "plate",
            "near_target": controller == "contact_direct", "routing_reason": reason},
        "history_stats": {}, "api_latency_s": 2., "api_request_key": "terminal-response",
        "api_request_log": "terminal.request.json", "compact_prediction": {"task_complete": True},
        "completion_reason": reason, "status": "model_declared_complete",
    }
    recorder.model_completion(observation, step=0, completion_reason=reason,
                              decision_latency_s=2.1, diagnostics=details)
    recorder.close(status="model_declared_complete")
    events = rows(tmp_path / "complete.events.jsonl")
    assert [e["event"] for e in events] == ["observation", "model_completion", "terminal"]
    final = events[1]
    assert final["task_complete"] is True and final["completion_reason"] == reason
    assert final["action"] == [] and final["success"] is False
    assert final["field_center"] is final["selected_index"] is final["plan_latency_s"] is None
    assert final["hybrid_refresh_count"] == 1
    assert final["api_request_key"] == "terminal-response"
