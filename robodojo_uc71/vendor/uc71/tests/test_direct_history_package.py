"""Offline provenance/packaging checks; these tests never start a policy or simulator."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_package", ROOT / "scripts/package_direct_history_continuation.py")
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def ledger():
    return {"configuration": {"model": "gpt-6-astra", "max_cost_usd": 150,
            "input_price_per_million": 10, "cache_write_price_per_million": 12.5,
            "cached_input_price_per_million": 1, "output_price_per_million": 50,
            "max_input_tokens": 1050000, "max_output_tokens": 8192},
            "max_cost_usd": 150, "api_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "reasoning_tokens": 0, "unknown_usage_calls": 0,
            "reservation_per_call_usd": 26.8644, "cost_usd_upper_bound": 0.0, "elapsed_s": 10}


def test_cap_handoff_keeps_every_existing_counter_and_unknown_reservation():
    old = ledger()
    old.update(api_calls=91, unknown_usage_calls=1, cost_usd_upper_bound=150.3845)
    handoff = deepcopy(old)
    handoff["configuration"]["max_cost_usd"] = handoff["max_cost_usd"] = 300
    config = {"original_budget_snapshot": deepcopy(old), "handoff_budget_snapshot": handoff,
              "cost_cap_change": {"old_total_usd": 150, "new_total_usd": 300, "only_configuration_change": "max_cost_usd"},
              "api": handoff["configuration"]}
    package.validate_handoff({"budget": old}, config)
    config["handoff_budget_snapshot"]["unknown_usage_calls"] = 0
    with pytest.raises(package.NotReady, match="beyond the cap"):
        package.validate_handoff({"budget": old}, config)


def test_ledger_counts_full_long_context_premium_and_unknown_call():
    snapshot = ledger()
    usage = {"input_tokens": 300000, "output_tokens": 200,
             "input_tokens_details": {"cached_tokens": 100}, "output_tokens_details": {"reasoning_tokens": 50}}
    metadata = [{"ordinal": 1, "usage": usage}, {"ordinal": 2, "usage": None}]
    snapshot.update(api_calls=2, input_tokens=300000, output_tokens=200, cached_input_tokens=100,
                    reasoning_tokens=50, unknown_usage_calls=1, cost_usd_upper_bound=7.515+26.8644)
    assert package.check_ledger(metadata, snapshot)["unknown_usage_calls"] == 1
    snapshot["cost_usd_upper_bound"] -= 26.8644
    with pytest.raises(package.NotReady, match="Cumulative cost"):
        package.check_ledger(metadata, snapshot)


def trace_fixture(tmp_path):
    state = [0, 0, 1, 0, 0, 0, 0.04, -0.04]
    obs = {"observation/state": state, "observation/image": "actual-camera-fixture", "observation/wrist_image": "actual-wrist-fixture"}
    command = [0, 0, 0, 0, 0, 0, -1]
    actions = [command]*5
    history = {"observations": 91, "feedback_records": 450}
    diagnostics = {"method": "direct_history", "status": "direct_action", "chunk_index": 90, "chunk_step": 0,
                   "refreshed": True, "prediction": {"actions": actions}, "api_request_log": "000091-key.request.json",
                   "api_request_key": "key", "api_latency_s": 1, "history_stats": history}
    feedback = {"type": "execution_feedback", "before_state": state, "after_state": state,
                "normalized_action": command, "measured_translation_delta_m": [0, 0, 0],
                "desired_translation_delta_m": [0, 0, 0]}
    trace = [{"event": "resume", "step": 450, "observation": obs},
             {"event": "action", "step": 450, "state": state, "action": command, "diagnostics": diagnostics},
             {"event": "executed", "step": 450, "next_state": state, "observation": obs, "action": command, "success": False},
             {"event": "feedback", "step": 450, "feedback": feedback}]
    events = [{"event": "observation", "step": 450, "state": state},
              {"event": "action_chunk", "step": 450, "actions": actions, "action": [], "api_request_key": "key", "api_latency_s": 1, "chunk_count": 91},
              {"event": "direct_action", "step": 450, "state": state, "action": command, "history_stats": history},
              {"event": "executed", "step": 450, "state": state, "action": command, "success": False,
               "feedback": feedback, "measured_delta_xyz_m": [0, 0, 0]},
              {"event": "terminal", "state": state}]
    requests = {"000091-key.request.json": {"request": {"request_key": "key", "history_stats_before_request": {"observations": 90, "feedback_records": 450}},
                                           "model_actions": deepcopy(actions), "result_status": "valid"}}
    parent = {"validated_action_chunks": 90, "measured_feedback_count": 450,
              "gripper_close_actions": 82, "gripper_open_actions": 368, "gripper_neutral_actions": 0}
    row = {"base_steps": 450, "new_steps": 1, "trace": "trace.jsonl", "pending_action_index": None,
           **parent, "validated_action_chunks": 91, "measured_feedback_count": 451, "gripper_open_actions": 369}
    (tmp_path/"trace.jsonl").write_text("".join(json.dumps(e)+"\n" for e in trace))
    return row, events, requests, parent


def test_global_action_450_uses_model_chunk_91_and_prior_feedback(tmp_path):
    row, events, requests, parent = trace_fixture(tmp_path)
    result, first, last = package.validate_child_trace(tmp_path, row, events, requests, parent)
    assert result["executed_actions"] == result["feedback_records"] == 1
    assert first == last
    requests["000091-key.request.json"]["request"]["history_stats_before_request"]["feedback_records"] = 0
    with pytest.raises(package.NotReady, match="prior observation/feedback"):
        package.validate_child_trace(tmp_path, row, events, requests, parent)


def test_rejects_action_chunk_forged_after_api_response(tmp_path):
    row, events, requests, parent = trace_fixture(tmp_path)
    requests["000091-key.request.json"]["model_actions"][0][0] = 1
    with pytest.raises(package.NotReady, match="exact validated model response"):
        package.validate_child_trace(tmp_path, row, events, requests, parent)


def test_rejects_replay_in_evaluated_trace(tmp_path):
    row, events, requests, parent = trace_fixture(tmp_path)
    entries = [json.loads(line) for line in (tmp_path/"trace.jsonl").read_text().splitlines()]
    entries.insert(1, {"event": "replay", "step": 0})
    (tmp_path/"trace.jsonl").write_text("".join(json.dumps(e)+"\n" for e in entries))
    with pytest.raises(package.NotReady, match="replay must be excluded"):
        package.validate_child_trace(tmp_path, row, events, requests, parent)


def test_event_index_retains_historical_pause_without_counting_reconstruction():
    data = {"segments": []}
    for label, start in (("original", 0), ("continuation", 450)):
        events = [{"event": "observation", "frame_start": 0, "frame_count": 12, "video_time_s": 0},
                  {"event": "executed", "step": start, "frame_start": 12, "frame_count": 2, "video_time_s": 0.6},
                  {"event": "terminal", "status": "budget_exceeded", "frame_start": 14, "frame_count": 60, "video_time_s": 0.7}]
        data["segments"].append({"label": label, "start_step": start, "events": events, "media": {"frames": 74}})
    events = package.shifted_events(data)
    assert [e["event"] for e in events] == ["observation", "executed", "historical_segment_pause", "editorial_resume_boundary", "observation", "executed", "terminal"]
    assert events[2]["source_event"] == "terminal" and events[2]["status"] == "budget_exceeded"
    assert events[3]["new_evaluated_actions"] == 0 and events[4]["frame_start"] == 174
    assert events[5]["step"] == 450


def test_live_child_rejected_before_reading_parent_or_creating_output(tmp_path):
    child = tmp_path/"live"
    child.mkdir()
    (child/"run_state.json").write_text(json.dumps({"stage": "executing", "status": "running"}))
    with pytest.raises(package.NotReady, match="Continuation is live"):
        package.validate(tmp_path/"nonexistent-parent", child)
    assert sorted(p.name for p in child.iterdir()) == ["run_state.json"]


def test_payload_hash_detects_changed_control_history():
    import hashlib
    config = {"endpoint": "https://api.openai.com/v1/responses"}
    request = {"configuration": config, "payload": {"store": False, "include": ["reasoning.encrypted_content"],
               "input": [{"role": "user", "content": "measured transition"}]},
               "grid": {"size": 9, "spacing_m": 0.02, "frame": "world"}, "mode": "direct",
               "prompt_version": "astra-direct-history-v1", "history_protocol": "responses_output_transcript_v1", "image_history": 2}
    material = {key: request[key] for key in ("payload", "grid", "mode", "prompt_version", "history_protocol", "image_history")}
    material["endpoint"] = config["endpoint"]
    request["request_key"] = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    package.request_integrity(request, config)
    request["payload"]["input"][0]["content"] = "altered control history"
    with pytest.raises(package.NotReady, match="payload does not match"):
        package.request_integrity(request, config)


def test_restore_requires_exact_next_request_after_all_history(tmp_path):
    parent = {"steps": 450, "api_calls": 90, "validated_action_chunks": 90}
    budget = ledger()
    budget["configuration"]["max_cost_usd"] = budget["max_cost_usd"] = 300
    history = {"observations": 90, "feedback_records": 450, "messages": 665}
    verification = {"status": "verified", "steps": 450, "model_api_calls": 0, "policy_changed": False,
                    "simulator_started": False, "history_sha256": "history-hash", "next_request_key": "next-key",
                    "history_stats": history, "feedback_records": 450, "state_observations": 451,
                    "validated_requests": 90, "next_global_request_ordinal": 91}
    config = {"offline_history_verification": verification, "history_digest_at_handoff": "history-hash",
              "next_request_key_at_handoff": "next-key", "handoff_budget_snapshot": budget}
    restored = {"status": "verified", "source_verification": verification, "policy_step_index": 450,
                "new_api_calls": 0, "policy_reset_after_restore": False, "history_sha256": "history-hash",
                "next_request_key": "next-key", "history_stats": history, "budget": budget}
    (tmp_path/"history_restore.json").write_text(json.dumps(restored))
    replay = {"status": "validated", "completed_actions": 450, "state_checks": 451, "image_pairs_checked": 451,
              "status_checks": 451, "state_atol": 0, "max_state_error": 0, "max_pixel_error": 0,
              "astra_requests_during_reconstruction": 0, "same_simulator_retained_for_continuation": True}
    assert package.validate_restore(tmp_path, config, replay, parent, [{"request_key": "next-key"}])["first_actual_request_matches_verified_next_request"]
    with pytest.raises(package.NotReady, match="First actual continuation request"):
        package.validate_restore(tmp_path, config, replay, parent, [{"request_key": "different-key"}])
    replay["max_pixel_error"] = 1
    with pytest.raises(package.NotReady, match="Exact replay proof"):
        package.validate_restore(tmp_path, config, replay, parent, [{"request_key": "next-key"}])


def test_media_package_preserves_source_frames_and_offline_archive(tmp_path):
    """Small colored clips test real ffmpeg/ZIP publication, without robot simulation."""
    import hashlib
    import shutil
    import zipfile
    from PIL import Image
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg tools unavailable")
    segments = []
    for index, color in enumerate(((150, 30, 20), (20, 100, 150))):
        root = tmp_path/f"source{index}"
        root.mkdir()
        Image.new("RGB", (1600, 960), color).save(root/"fixture.png")
        video = root/"episode.mp4"
        package.media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-loop", "1",
                               "-framerate", "20", "-i", str(root/"fixture.png"), "-frames:v", "3", "-an",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-threads", "1", str(video)])
        events = [{"event": "observation", "frame_start": 0, "frame_count": 1, "video_time_s": 0},
                  {"event": "executed", "step": index, "frame_start": 1, "frame_count": 1, "video_time_s": 0.05},
                  {"event": "terminal", "status": "budget_exceeded", "frame_start": 2, "frame_count": 1, "video_time_s": 0.1}]
        events_path = root/"episode.events.jsonl"
        events_path.write_text("".join(json.dumps(e)+"\n" for e in events))
        for name in ("manifest.json", "summary.json", "replay.json"):
            (root/name).write_text("{}")
        segments.append({"root": root, "label": "original" if index == 0 else "continuation", "start_step": index,
                         "summary": {"steps": index+1}, "video": video, "events_path": events_path, "events": events,
                         "media": {"frames": 3}, "api_metadata": [], "manifest": {"identity_hash": "fixture"},
                         "trace_validation": {"executed_actions": 1}, "frozen_source_checks": {}})
    row = {"status": "budget_exceeded", "evaluation_completed": False, "steps": 2, "new_steps": 1,
           "requested_episodes": 1, "scored_episodes": 0, "incomplete_episodes": 1,
           "validated_action_chunks": 2, "api_calls": 2, "measured_feedback_count": 2,
           "budget": {"cost_usd_upper_bound": 1, "unknown_usage_calls": 0}, "task": "offline colored media fixture"}
    source_hashes = [package.sha(s["video"]) for s in segments]
    result = package.package({"segments": segments, "summary": row, "ledger": {}, "restoration": {}}, tmp_path/"package")
    assert result["media"]["frames"] == 106
    proof = json.loads((tmp_path/"package/video_package_manifest.json").read_text())
    assert proof["exact_frame_preservation"]["passed"] is True
    assert proof["events"]["counts"]["executed"] == 2
    assert proof["events"]["counts"]["historical_segment_pause"] == 1
    assert [package.sha(s["video"]) for s in segments] == source_hashes
    with zipfile.ZipFile(result["archive"]) as z:
        for name, meta in proof["files"].items():
            assert hashlib.sha256(z.read("package/"+name)).hexdigest() == meta["sha256"]
