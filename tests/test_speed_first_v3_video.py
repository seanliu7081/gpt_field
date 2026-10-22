"""Real v3 renderer and event log; only the encoded-video sink is replaced."""
import json
from types import SimpleNamespace

import numpy as np

from gpt_field.fine_distance_planner import FINE_GRID

from gpt_field.speed_first_v3_video import SpeedFirstV3VideoRecorder
from test_field_history_runner import rows
from test_speed_first_v2_video import writer_sink
from test_speed_first_v3_runner import highres_observation, observation, run, v3_intent


def test_v3_real_recorder_adds_color_slices_without_dense_3d_map(
        tmp_path, highres_observation, writer_sink, monkeypatch):
    def forbidden_dense(*args, **kwargs):
        raise AssertionError("The lazy-field recorder must never allocate a dense GridField")

    monkeypatch.setattr("gpt_field.hybrid_video.GridField", forbidden_dense)
    direct = v3_intent(direct=True, phase="verify")
    direct["direct"]["actions"] = [[.01 * (i + 1), -.02, 0., 0., 0., 0., 1.] for i in range(5)]
    output = tmp_path / "mixed"
    result, transport, instances = run(output, highres_observation, [
        v3_intent(operation="close", phase="grasp"), direct,
        v3_intent(phase="recover", goal=(0., -.03, 0.)),
    ], success_at=21, recorder=SpeedFirstV3VideoRecorder)
    assert result["status"] == "success", result["error"]
    assert result["steps"] == result["measured_feedback_count"] == 21
    assert result["local_gripper_actions"] == 15 and result["ordinary_direct_actions"] == 5
    assert result["coarse_field_actions"] == 1 and len(transport.calls) == 3
    assert instances[0].closed
    events = rows(output / "episode.events.jsonl")
    actions = [r for r in rows(output / "trace.jsonl") if r["event"] == "action"]
    assert [r["step"] for r in events if r["event"] == "gripper_operation"] == [0]
    assert [r["step"] for r in events if r["event"] == "gripper_action"] == list(range(15))
    assert [r["step"] for r in events if r["event"] == "action_chunk"] == [15]
    assert [r["step"] for r in events if r["event"] == "direct_action"] == list(range(15, 20))
    assert [r["step"] for r in events if r["event"] == "field_refresh"] == [20]
    for event in events:
        assert event["grid"]["size"] == 161 and event["grid"]["spacing_m"] == .001
        assert not event["dense_distance_map_rendered"]
        assert not event["task_success_certified"]
        assert len(json.dumps(event)) < 100_000
        if event.get("execution_kind") == "local_gripper_settle":
            assert "actions" not in event and event["model_action_array"] is None
    fresh = next(r for r in events if r["event"] == "field_refresh")
    assert fresh["field_display_source"] == "unavailable"  # Fresh routing omits the current plan.
    assert not fresh["distance_slice_heatmaps_rendered"]
    planned = next(r for r in events if r["event"] == "local_replan")
    assert planned["field_display_source"] == "reconstructed_analytic_distance_slices"
    assert planned["distance_slice_heatmaps_rendered"]
    assert planned["distance_slice_shape"] == [161, 161]
    assert planned["field_display_cost_unit"] == "m"
    assert planned["distance_display_goal_tolerance_m"] == planned["planner_search"]["goal_tolerance_m"]
    assert planned["distance_display_goal_xyz"] == planned["planner_search"]["goal_xyz"]
    assert all(not r["distance_slice_heatmaps_rendered"] for r in events if r["controller"] == "contact_direct")
    assert fresh["field_center"] is not None
    assert "values" not in (fresh["field_prediction"] or {})
    for event in (r for r in events if r["event"] == "executed"):
        assert event["action"] == actions[event["step"]]["action"]
        assert event["feedback_step"] == event["step"]
    assert events[-1]["status"] == "success" and events[-1]["success"] is True
    writer = writer_sink.instances[-1]
    assert writer.closed and writer.frame_count == sum(r["frame_count"] for r in events)
    assert len(writer.calls) == len(events)
    assert len({row["sha256"] for row in writer.calls}) > 3


def test_v3_runner_selects_real_v3_recorder_by_default(tmp_path, highres_observation, writer_sink):
    result, _, _ = run(tmp_path / "default", highres_observation,
        [v3_intent(operation="close", phase="grasp")], success_at=2, recorder=None)
    assert result["status"] == "success", result["error"]
    events = rows(tmp_path / "default/episode.events.jsonl")
    assert len([r for r in events if r["event"] == "gripper_action"]) == 2
    assert not any(r["event"] == "model_completion" for r in events)
    assert all("dense_distance_map_rendered" in r for r in events)
    assert writer_sink.instances[-1].closed


def test_reconstructed_slice_units_workspace_mask_and_no_stale_color(tmp_path, writer_sink):
    recorder = SpeedFirstV3VideoRecorder(tmp_path / "slices.mp4", grid=FINE_GRID)
    center = [-.2, 0., .04]
    recorder.controller = "coarse_field"
    recorder.observation = SimpleNamespace(state=center + [0.] * 5)
    compact = v3_intent(goal=(0., 0., 0.))["field"]
    recorder.details = {"field_center": center, "effective_compact_field": compact,
        "plan": {"search": {"method": "analytic_uniform_goal_distance",
                              "goal_xyz": center.copy(), "goal_tolerance_m": .0025}}}
    display = recorder._distance_slices()
    assert display["source"] == "reconstructed_analytic_distance_slices"
    assert sum(p["values"].size for p in display["planes"]) == 3 * 161 ** 2
    xy, xz, yz = (p["values"] for p in display["planes"])
    assert xy[80, 80] == 0.
    # A physical 3-4-5 cm offset minus the actual 2.5 mm goal radius.
    assert np.isclose(xy[40, 110], .0475)
    assert np.isclose(xy[120, 50], .0475)
    assert np.isfinite(xz[120, 80]) and np.isnan(xz[121:, :]).all()
    assert np.isnan(yz[121:, :]).all()
    assert recorder._distance_slices() is display
    # Updating a local goal must replace the cached color, not borrow the old target.
    compact["goal_offset_m"] = [.01, 0., 0.]
    recorder.details["plan"]["search"]["goal_xyz"] = [-.19, 0., .04]
    changed = recorder._distance_slices()
    assert changed is not display and np.isclose(changed["planes"][0]["values"][80, 80], .0075)
    # Even an analytic selected path does not imply exact distances everywhere
    # when off-path boxes or traversal costs exist.
    for key in ("repulsors", "blocked_boxes", "unknown_boxes"):
        compact[key] = [{}]
        fallback = recorder._distance_slices()
        assert fallback["source"] == "unavailable" and "planes" not in fallback
        compact[key] = []
    recorder.details.pop("plan")
    assert recorder._distance_slices()["source"] == "unavailable"
    recorder.controller = "contact_direct"
    assert recorder._distance_slices()["source"] == "unavailable"
    # Close the unit-test sink directly; no terminal frame with our synthetic observation.
    recorder.writer.close()
    recorder._events.close()
