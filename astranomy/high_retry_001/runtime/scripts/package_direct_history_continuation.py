#!/usr/bin/env python3
"""Offline, exact-frame package of one direct-history episode across a pause.

Usage: .venv/bin/python scripts/package_direct_history_continuation.py \
  --parent runs/astra_direct_history_001 \
  --continuation runs/astra_direct_history_continuation_001

Both segments must be terminal. Source runs and existing packages are immutable.
No simulator, policy, API client, network, or paid calls are used. The only new
video material is an explicitly labeled five-second text boundary card.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont

# This sibling contains only offline file/media utilities, never policy imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_direct_history_episode as base

FPS, WIDTH, HEIGHT = base.FPS, base.WIDTH, base.HEIGHT
CARD_FRAMES = 100
NotReady, require, read_json, read_jsonl = base.NotReady, base.require, base.read_json, base.read_jsonl
sha, local_file, media_command = base.sha, base.local_file, base.media_command


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def without_elapsed(snapshot):
    return {key: value for key, value in snapshot.items() if key != "elapsed_s"}


def validate_handoff(parent, config):
    """Only the approved cumulative cap can change; never discard past usage."""
    original = config["original_budget_snapshot"]
    require(without_elapsed(original) == without_elapsed(parent["budget"]), "Original cumulative ledger differs from parent terminal ledger")
    change = config["cost_cap_change"]
    require(change == {"old_total_usd": 150, "new_total_usd": 300,
                       "only_configuration_change": "max_cost_usd"}, "Unrecognized authorized cap handoff")
    expected = deepcopy(original)
    require(expected["configuration"]["max_cost_usd"] == expected["max_cost_usd"] == 150, "Parent cap is not the recorded $150 cap")
    expected["configuration"]["max_cost_usd"] = expected["max_cost_usd"] = 300
    require(config["handoff_budget_snapshot"] == expected, "Handoff changed prior accounting or configuration beyond the cap")
    require(config["api"] == expected["configuration"], "Continuation API configuration differs from cap-only handoff")


def check_ledger(metadata, snapshot):
    """Recompute the frozen upper-bound formula without importing the API client."""
    config = snapshot["configuration"]
    rate = max(config["input_price_per_million"], config["cache_write_price_per_million"],
               config["cached_input_price_per_million"] or 0)
    def cost(inp, out):
        long = config["model"] == "gpt-6-astra" and inp > 272000
        return (inp * rate * (2 if long else 1) + out * config["output_price_per_million"] * (1.5 if long else 1)) / 1000000
    reservation = cost(config["max_input_tokens"], config["max_output_tokens"])
    totals = Counter(api_calls=len(metadata))
    upper = 0.0
    for ordinal, entry in enumerate(metadata, 1):
        require(entry["ordinal"] == ordinal, "Cumulative request ordinals are not contiguous")
        usage = entry.get("usage")
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
            totals["unknown_usage_calls"] += 1
            upper += reservation
            continue
        inp, out = usage["input_tokens"], usage["output_tokens"]
        totals["input_tokens"] += inp
        totals["output_tokens"] += out
        upper += cost(inp, out)
        for key, section, field in (("cached_input_tokens", "input_tokens_details", "cached_tokens"),
                                    ("reasoning_tokens", "output_tokens_details", "reasoning_tokens")):
            value = (usage.get(section) or {}).get(field)
            if type(value) is int and value >= 0:
                totals[key] += value
    for key in ("api_calls", "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "unknown_usage_calls"):
        require(totals[key] == snapshot[key], f"Cumulative response usage differs from ledger: {key}")
    require(math.isclose(upper, snapshot["cost_usd_upper_bound"], rel_tol=1e-12, abs_tol=1e-8), "Cumulative cost does not match response usage plus unknown reservations")
    require(math.isclose(reservation, snapshot["reservation_per_call_usd"], abs_tol=1e-10), "Reservation amount mismatch")
    return {"all_response_usage_checked": True, "cumulative_api_calls": totals["api_calls"],
            "unknown_usage_calls": totals["unknown_usage_calls"], "recomputed_cost_usd_upper_bound": upper,
            "unknown_usage_retains_full_reservation": True}


def request_integrity(request, config):
    require(request.get("configuration") == config, "Request configuration differs from frozen segment configuration")
    payload = request["payload"]
    require(payload.get("store") is False and "reasoning.encrypted_content" in payload.get("include", []),
            "Request changed opaque-history retention protocol")
    material = {"payload": payload, "grid": request["grid"], "mode": request["mode"],
                "prompt_version": request["prompt_version"], "endpoint": config["endpoint"],
                "history_protocol": request["history_protocol"], "image_history": request["image_history"]}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    require(hashlib.sha256(encoded).hexdigest() == request["request_key"], "Request payload does not match its recorded key")


def child_api_index(root, row, parent_calls, config):
    indexed, requests = [], {}
    paths = sorted((root / "api").glob("*.request.json"))
    require(len(paths) == row["new_api_calls"] == row["api_calls"] - parent_calls, "New request count disagrees with cumulative API calls")
    for ordinal, path in enumerate(paths, parent_calls + 1):
        request = read_json(path)
        request_integrity(request, config)
        require(request.get("request_ordinal") == ordinal and request.get("mode") == "direct", "Continuation request ordinal/mode mismatch")
        require(request.get("prompt_version") == "astra-direct-history-v1"
                and request.get("history_protocol") == "responses_output_transcript_v1"
                and request.get("image_history") == 2, "Continuation changed history protocol")
        payload = request.get("payload", {})
        require(payload.get("model") == "gpt-6-astra" and payload.get("text", {}).get("format", {}).get("name") == "astra_direct_history_v1", "Continuation changed model or output schema")
        result_path = path.with_name(path.name.replace(".request.json", ".result.json"))
        response_path = path.with_name(path.name.replace(".request.json", ".response.json"))
        result = read_json(result_path)
        response = read_json(response_path) if response_path.exists() else None
        require(response is None or not response.get("offline_mock"), "Mock response in live continuation")
        require(result.get("budget", {}).get("api_calls") == ordinal, "Per-request cumulative budget ordinal mismatch")
        metadata = {"ordinal": ordinal, "request_key": request["request_key"],
                    "request": path.relative_to(root).as_posix(), "result_status": result.get("status"),
                    "history_stats_before_request": request.get("history_stats_before_request"),
                    "response_id": response.get("id") if response else None,
                    "response_status": response.get("status") if response else None,
                    "usage": response.get("usage") if response else None,
                    "files": {f.relative_to(root).as_posix(): {"sha256": sha(f), "bytes": f.stat().st_size}
                              for f in (path, response_path, result_path) if f.exists()}}
        indexed.append(metadata)
        requests[path.name] = {"request": {"request_key": request["request_key"],
                                           "history_stats_before_request": request.get("history_stats_before_request")},
                              "model_actions": base.model_actions(response) if response else None,
                              "result_status": result.get("status")}
    return indexed, requests


def validate_child_trace(root, row, events, requests, parent):
    cursor = issued = executed = feedback_count = chunks = 0
    offset = row["base_steps"]
    pending = chunk = initial = latest = last_executed = None
    grippers = Counter()
    trace = local_file(root, row["trace"])
    with trace.open() as source:
        for line in source:
            entry = json.loads(line)
            require(cursor < len(events), "Trace exceeds video events")
            kind, step = entry["event"], offset + executed
            if kind == "resume":
                require(cursor == 0 and initial is None and entry["step"] == offset, "Unexpected resume boundary")
                initial = latest = entry["observation"]
                require(events[cursor]["event"] == "observation" and events[cursor]["step"] == offset
                        and events[cursor]["state"] == latest["observation/state"], "Initial continuation video/state mismatch")
                cursor += 1
            elif kind == "action":
                require(pending is None and entry["step"] == step and entry["state"] == latest["observation/state"], "Nonsequential issuance or stale state")
                d = entry["diagnostics"]
                require(d.get("method") == "direct_history" and d.get("status") == "direct_action"
                        and "plan" not in d and "field_prediction" not in d, "Unexpected field or local planner")
                require(d["chunk_index"] == step // 5 and d["chunk_step"] == step % 5, "Continuation chunk indices reset or disagree")
                if d["refreshed"]:
                    require(step % 5 == 0, "Mid-chunk continuation refresh")
                    event = events[cursor]
                    chunk = d["prediction"]["actions"]
                    require(event["event"] == "action_chunk" and event["step"] == step
                            and event["actions"] == chunk and event["action"] == []
                            and event["chunk_count"] == parent["validated_action_chunks"] + chunks + 1,
                            "Recorded chunk differs from decision or cumulative chunk count")
                    require(len(chunk) == 5 and all(len(a) == 7 for a in chunk), "Invalid chunk shape")
                    require(all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1 for a in chunk for v in a),
                            "Action chunk contains nonfinite, boolean or out-of-bounds controls")
                    request = requests.get(Path(d["api_request_log"]).name)
                    require(request is not None and request["result_status"] == "valid"
                            and request["model_actions"] == chunk, "Chunk lacks exact validated model response")
                    require(request["request"]["request_key"] == d["api_request_key"] == event["api_request_key"], "Chunk API request mismatch")
                    history = request["request"]["history_stats_before_request"]
                    require(history["observations"] == parent["validated_action_chunks"] + chunks
                            and history["feedback_records"] == step, "Full prior observation/feedback history was not retained")
                    require(event["api_latency_s"] == d["api_latency_s"], "Recorded API latency mismatch")
                    chunks += 1
                    cursor += 1
                event = events[cursor]
                require(event["event"] == "direct_action" and event["step"] == step
                        and event["state"] == entry["state"] and event["action"] == entry["action"], "Issued action/video mismatch")
                require(chunk is not None and chunk[step % 5] == entry["action"], "Issued action differs from actual API chunk")
                require(event["history_stats"] == d["history_stats"], "Displayed history metadata differs from decision")
                pending = entry
                issued += 1
                cursor += 1
            elif kind == "executed":
                require(pending is not None and entry["step"] == step and entry["action"] == pending["action"], "Execution lacks matching model issuance")
                require(entry["next_state"] == entry["observation"]["observation/state"], "Execution state/observation mismatch")
                latest, last_executed = entry["observation"], entry
                executed += 1
                grip = entry["action"][6]
                grippers["close" if grip > 0 else "open" if grip < 0 else "neutral"] += 1
            elif kind == "feedback":
                require(pending is not None and last_executed is not None and entry["step"] == step - 1, "Feedback without execution")
                feedback, event = entry["feedback"], events[cursor]
                require(feedback["type"] == "execution_feedback" and feedback["before_state"] == pending["state"]
                        and feedback["after_state"] == latest["observation/state"]
                        and feedback["normalized_action"] == pending["action"], "Measured feedback state/action mismatch")
                require(not any(k in feedback for k in ("success", "reward", "object_pose", "contact")), "Evaluator-only data included in policy feedback")
                delta = [feedback["after_state"][i] - feedback["before_state"][i] for i in range(3)]
                base.close_vectors(feedback["measured_translation_delta_m"], delta, "measured translation")
                base.close_vectors(feedback["desired_translation_delta_m"], [0.05*x for x in pending["action"][:3]], "desired translation")
                require(event["event"] == "executed" and event["step"] == entry["step"]
                        and event["state"] == latest["observation/state"] and event["action"] == pending["action"]
                        and event["success"] == last_executed["success"] and event["feedback"] == feedback, "Executed video does not match measured transition")
                base.close_vectors(event["measured_delta_xyz_m"], delta, "video translation")
                pending = None
                feedback_count += 1
                cursor += 1
            else:
                raise NotReady(f"Unexpected continuation trace event (replay must be excluded): {kind}")
    require(initial is not None and cursor == len(events)-1 and events[-1]["event"] == "terminal", "Trace/video does not end together")
    require(events[-1]["state"] == latest["observation/state"], "Final camera/state mismatch")
    require(executed == row["new_steps"] and feedback_count == row["measured_feedback_count"] - parent["measured_feedback_count"], "Continuation execution/feedback count mismatch")
    require(chunks == row["validated_action_chunks"] - parent["validated_action_chunks"], "Cumulative accepted chunk mismatch")
    pending_index = pending["step"] if pending else None
    require(pending_index == row["pending_action_index"], "Unconfirmed action mismatch")
    for label in ("close", "open", "neutral"):
        require(grippers[label] == row[f"gripper_{label}_actions"] - parent[f"gripper_{label}_actions"], "Cumulative gripper counts disagree")
    return {"executed_actions": executed, "issued_actions": issued, "accepted_action_chunks": chunks,
            "feedback_records": feedback_count, "gripper_counts": dict(grippers),
            "unconfirmed_issued_action_index": pending_index, "trace_sha256": sha(trace),
            "all_video_actions_match_recorded_model_chunks": True,
            "all_video_feedback_matches_measured_transitions": True}, initial, latest


def final_observation(root, trace_name):
    observation = None
    with local_file(root, trace_name).open() as source:
        for line in source:
            entry = json.loads(line)
            if entry["event"] in ("reset", "executed"):
                observation = entry["observation"]
    require(observation is not None, "Parent has no actual observations")
    return observation


def validate_restore(root, config, replay, parent, metadata):
    offset = parent["steps"]
    require(replay.get("status") == "validated" and replay["completed_actions"] == offset
            and replay["state_checks"] == replay["image_pairs_checked"] == replay["status_checks"] == offset + 1
            and replay["state_atol"] == replay["max_state_error"] == replay["max_pixel_error"] == 0
            and replay["astra_requests_during_reconstruction"] == 0
            and replay["same_simulator_retained_for_continuation"] is True, "Exact replay proof missing or incomplete")
    restored = read_json(root / "history_restore.json")
    verification = config["offline_history_verification"]
    require(restored["status"] == verification["status"] == "verified" and restored["source_verification"] == verification,
            "Full history restoration proof differs from frozen offline verification")
    require(restored["policy_step_index"] == verification["steps"] == offset
            and restored["new_api_calls"] == verification["model_api_calls"] == 0
            and restored["policy_reset_after_restore"] is False and verification["policy_changed"] is False
            and verification["simulator_started"] is False, "History reset or new model/simulator calls occurred during offline history restoration")
    require(restored["history_sha256"] == verification["history_sha256"] == config["history_digest_at_handoff"]
            and restored["next_request_key"] == verification["next_request_key"] == config["next_request_key_at_handoff"],
            "Restored history digest or next request key differs from verified handoff")
    require(restored["history_stats"] == verification["history_stats"]
            and verification["feedback_records"] == offset
            and verification["state_observations"] == offset + 1
            and verification["validated_requests"] == parent["validated_action_chunks"]
            and verification["next_global_request_ordinal"] == parent["api_calls"] + 1,
            "Restored history counts do not include every original action and accepted response")
    require(without_elapsed(restored["budget"]) == without_elapsed(config["handoff_budget_snapshot"]), "History restoration lost cumulative ledger values")
    if metadata:
        require(metadata[0]["request_key"] == restored["next_request_key"], "First actual continuation request differs from exact restored history")
    return {"exact_replay_actions": offset, "state_and_image_pairs_checked": offset + 1,
            "full_history_sha256": restored["history_sha256"], "next_request_key": restored["next_request_key"],
            "zero_paid_requests_during_reconstruction": True, "policy_reset_after_restore": False,
            "first_actual_request_matches_verified_next_request": bool(metadata)}


def validate(parent_root, child_root):
    state = read_json(child_root / "run_state.json")
    require(state.get("stage") == "finished" and state.get("status") != "running", "Continuation is live; no final package will be written")
    before = base.validate(parent_root)
    parent = before["summary"]
    require(before["manifest"]["identity_hash"] == digest(before["manifest"]["configuration"]), "Parent manifest identity is invalid")
    for metadata in before["api_metadata"]:
        request_integrity(read_json(local_file(parent_root, metadata["request"])), before["manifest"]["configuration"]["api"])
    require(parent["status"] == "budget_exceeded" and parent["steps"] == 450
            and parent["pending_action_index"] is None, "Expected unchanged 450-action parent budget pause")
    row = read_json(child_root / "summary.json")
    require(row == read_json(child_root / "runner_summary.json") and read_jsonl(child_root / "episodes.jsonl") == [row], "Child terminal summaries disagree")
    require(row["status"] == state["status"] and row["mode"] == base.LIVE_MODE, "Child terminal state/mode mismatch")
    require(row["base_steps"] == parent["steps"] and row["steps"] == row["cumulative_steps"] == row["base_steps"] + row["new_steps"], "Replay counted as new or missing actions")
    require(row["replay_verified"] is True and row["reconstruction_actions"] == parent["steps"]
            and row["base_api_calls"] == parent["api_calls"], "Child summary does not certify exact parent continuation")
    for key in ("task", "suite", "task_id", "episode_index", "seed"):
        require(row[key] == parent[key], f"Episode identity changed: {key}")
    success = row["status"] == "success"
    require(row["requested_episodes"] == 1 and row["scored_episodes"] == int(success)
            and row["incomplete_episodes"] == int(not success) and row["evaluation_completed"] is success
            and row["success_rate"] == (1.0 if success else None) and row["libero_success"] == (True if success else None), "Incorrect single-episode scoring")
    manifest = read_json(child_root / "manifest.json")
    config = manifest["configuration"]
    require(manifest["identity_hash"] == digest(config), "Child manifest identity is invalid")
    require(config["kind"] == "same_episode_direct_history_budget_continuation"
            and config["mode"] == base.LIVE_MODE, "Unexpected continuation provenance")
    require(Path(config["parent_run"]).resolve() == parent_root
            and config["parent_identity_hash"] == before["manifest"]["identity_hash"]
            and config["continuation_offset"] == parent["steps"], "Parent linkage differs from source episode")
    for name, expected in config["parent_artifact_sha256"].items():
        require(sha(local_file(parent_root, name)) == expected, f"Parent artifact changed: {name}")
    require(config["policy"] == before["manifest"]["configuration"]["policy"], "Policy changed across continuation")
    validate_handoff(parent, config)
    require(row["budget"]["configuration"] == config["api"] and row["budget"]["max_cost_usd"] == 300, "Terminal cap/config differs from authorized handoff")
    frozen = {}
    for name, expected in config["software"]["source_sha256"].items():
        frozen[name] = sha(base.PROJECT / "src" / "gpt_field" / name) == expected
    if "runner" in config:
        frozen[config["runner"]["path"]] = sha(local_file(base.PROJECT, config["runner"]["path"])) == config["runner"]["sha256"]
    require(all(frozen.values()), "Frozen continuation source changed")
    replay = read_json(child_root / "replay.json")
    require(not list(child_root.glob("*.partial.mp4")), "Child MP4 still partial")
    video = local_file(child_root, row["video"])
    events_path = local_file(child_root, video.with_suffix(".events.jsonl").name)
    events = read_jsonl(events_path)
    require(events and events[0]["event"] == "observation" and events[-1]["event"] == "terminal", "Child observation/terminal event missing")
    position = 0
    for event in events:
        require(event["event"] in base.ALLOWED_EVENTS and event["policy"] == "direct_history"
                and event["offline_smoke"] is False, "Unexpected child video provenance")
        require(event["frame_start"] == position and type(event["frame_count"]) is int and event["frame_count"] > 0
                and abs(event["video_time_s"] - position/FPS) < 1e-7, "Child event frames not contiguous")
        position += event["frame_count"]
    require(events[-1]["status"] == row["status"] and events[-1]["success"] is success, "Child video outcome differs from terminal summary")
    metadata, requests = child_api_index(child_root, row, parent["api_calls"], config["api"])
    restoration = validate_restore(child_root, config, replay, parent, metadata)
    trace, first, last = validate_child_trace(child_root, row, events, requests, parent)
    require(first == final_observation(parent_root, parent["trace"]), "Resume camera images/state differ from exact parent terminal observation")
    media = base.probe(video)
    require(media["frames"] == position, "Child video frame count differs from events")
    before.update(label="original", start_step=0)
    after = {"root": child_root, "label": "continuation", "start_step": parent["steps"], "summary": row,
             "manifest": manifest, "video": video, "events_path": events_path, "events": events, "media": media,
             "api_metadata": metadata, "trace_validation": trace, "frozen_source_checks": frozen}
    ledger = check_ledger(before["api_metadata"] + metadata, row["budget"])
    return {"segments": [before, after], "summary": row, "replay": replay, "ledger": ledger, "restoration": restoration}


def shifted_events(data):
    result, offset = [], 0
    for index, segment in enumerate(data["segments"]):
        if index:
            result.append({"event": "editorial_resume_boundary", "frame_start": offset,
                           "frame_count": CARD_FRAMES, "video_time_s": offset/FPS,
                           "completed_evaluated_actions": segment["start_step"], "new_evaluated_actions": 0,
                           "reconstruction_actions_counted_as_new": 0, "cumulative_api_cap_usd": 300})
            offset += CARD_FRAMES
        for event in segment["events"]:
            updated = dict(event, segment=segment["label"], source_event=event["event"],
                           source_frame_start=event["frame_start"], source_video_time_s=event["video_time_s"],
                           frame_start=offset+event["frame_start"], video_time_s=(offset+event["frame_start"])/FPS)
            if event["event"] == "terminal":
                updated["event"] = "terminal" if index == len(data["segments"])-1 else "historical_segment_pause"
                updated["is_final_outcome_of_combined_episode"] = index == len(data["segments"])-1
            result.append(updated)
        offset += segment["media"]["frames"]
    return result


def boundary_card(path, offset):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (12, 18, 30))
    draw = ImageDraw.Draw(canvas)
    def font(size, bold=False):
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    draw.text((90, 80), "ONE SIMULATION EPISODE / RESUME", font=font(44, True), fill="#e8effa")
    draw.text((90, 185), f"Budget pause after {offset} executed actions", font=font(42, True), fill="#ffd273")
    lines = ["The original recording stopped at its $150 API budget guard.",
             "The same simulation episode was reconstructed exactly.",
             "Full prior model history and cumulative usage were restored.",
             "Historical replay adds zero evaluated actions or API requests.",
             f"New actions begin at index {offset}, with API request 91.",
             "Approved API cap: $300 total, including the original usage."]
    for index, line in enumerate(lines):
        draw.text((90, 310+index*70), line, font=font(29), fill="#d2e2f4")
    draw.text((90, 855), "Editorial text card. API waits and reconstruction footage are omitted.", font=font(25), fill="#8db0c8")
    canvas.save(path)


def decode_hashes(path):
    print(f"Decoding all frames: {path.name}", file=sys.stderr, flush=True)
    text = media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-i", str(path),
                          "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "framemd5", "-"])
    hashes = [line.rsplit(",", 1)[1].strip() for line in text.splitlines() if line and not line.startswith("#")]
    require(hashes, "No video frames decoded")
    return hashes


def outcome(row):
    return "LIBERO success reached" if row["evaluation_completed"] else f"Incomplete: {row['status'].replace('_', ' ')}"


def result_text(data, media):
    row, parent = data["summary"], data["segments"][0]["summary"]
    return (f"**{outcome(row)}.**\n\nOne simulation episode: task 0, initial state 0, seed 7. "
            f"{row['steps']} confirmed actions = {parent['steps']} original + {row['new_steps']} continuation actions. "
            "Exact historical reconstruction contributes zero new evaluated actions or API requests.\n\n"
            f"Episode counts: {row['requested_episodes']} requested, {row['scored_episodes']} scored, {row['incomplete_episodes']} incomplete. "
            + ("No success rate is assigned to this incomplete episode.\n\n" if not row["evaluation_completed"] else "\n\n")
            + f"{row['validated_action_chunks']} accepted action chunks; {row['api_calls']} cumulative API attempts; "
            f"{row['measured_feedback_count']} measured feedback records. "
            f"Cumulative accounting upper bound: ${row['budget']['cost_usd_upper_bound']:.4f}; "
            f"unknown-usage attempts: {row['budget']['unknown_usage_calls']}. The $300 total cap includes prior usage and retained unknown reservations.\n\n"
            "Original budget-stop footage is retained as a historical pause. A labeled five-second editorial card separates the exact source recordings. "
            "API waits and reconstruction footage are omitted. No field generation or local field planning is used.\n\n"
            f"Combined video: {media['frames']:,} frames at 20 fps, {media['duration_s']:.2f} seconds. "
            "Source frames are preserved exactly and every frame is decoded for verification.\n")


def gallery(data, media):
    text = result_text(data, media)
    paragraphs = "".join(f"<p>{html.escape(p).replace('**', '')}</p>" for p in text.strip().split("\n\n"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>One Astra direct-history episode</title><style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#0c121e;color:#e8effa}}a{{color:#79d9ee}}video,img{{width:100%;height:auto}}</style></head><body>
<h1>One Astra / LIBERO episode, continued after a budget pause</h1><p>{html.escape(data['summary']['task'])}</p>{paragraphs}
<video controls preload="metadata" src="full_episode.mp4" poster="video_preview.png"></video>
<p><a href="full_episode.mp4" download>Combined MP4</a> · <a href="full_episode.events.jsonl">Combined events</a> · <a href="RESULTS.md">Results</a> · <a href="video_package_manifest.json">Validation and hashes</a> · <a href="api_log_index.json">API log metadata</a></p>
<p><a href="original/episode.mp4">Original recording</a> · <a href="continuation/episode.mp4">Continuation recording</a> · <a href="continuation/replay.json">Exact reconstruction proof</a> · <a href="continuation/manifest.json">Continuation protocol</a> · <a href="continuation/summary.json">Terminal summary</a></p>
<h2>Actual final recorded frame</h2><img src="video_preview.png" alt="Unedited decoded terminal video frame"></body></html>'''


def package(data, output):
    archive = output.with_name(output.name+".zip")
    require(not output.exists() and not archive.exists(), "Output directory and ZIP must both be fresh")
    require(all(not output.is_relative_to(s["root"]) for s in data["segments"]), "Output must be outside immutable source runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+"-build-", dir=output.parent) as tmp:
        staging, publish = Path(tmp), Path(tmp)/output.name
        publish.mkdir()
        pieces, expected, piece_proofs = [], [], []
        for index, segment in enumerate(data["segments"]):
            target = publish/segment["label"]
            target.mkdir()
            names = {"manifest.json", "summary.json", "runner_summary.json", "episodes.jsonl", "run_state.json",
                     "simulator.json", "authorization.json", "replay.json", "history_restore.json"}
            for name in sorted(names):
                source = segment["root"]/name
                if source.is_file():
                    shutil.copyfile(source, target/name)
            shutil.copyfile(segment["video"], target/"episode.mp4")
            shutil.copyfile(segment["events_path"], target/"episode.events.jsonl")
            if index:
                boundary_card(publish/"resume_boundary.png", segment["start_step"])
                card = staging/"boundary.mp4"
                media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-loop", "1",
                               "-framerate", str(FPS), "-i", str(publish/"resume_boundary.png"), "-frames:v", str(CARD_FRAMES),
                               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                               "-threads", "1", "-movflags", "+faststart", str(card)])
                pieces.append((card, CARD_FRAMES, "editorial_resume_boundary"))
            source_copy = staging/f"source_{index:02d}.mp4"
            shutil.copyfile(segment["video"], source_copy)
            pieces.append((source_copy, segment["media"]["frames"], segment["label"]))
        concat = staging/"concat.txt"
        concat.write_text("".join(f"file '{path.name}'\n" for path, _, _ in pieces))
        combined = publish/"full_episode.mp4"
        media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1",
                       "-i", str(concat), "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(combined)])
        for path, frames, label in pieces:
            hashes = decode_hashes(path)
            require(len(hashes) == frames, f"Unexpected decoded frame count: {label}")
            expected.extend(hashes)
            piece_proofs.append({"label": label, "frames": frames,
                                 "pixel_hash_sequence_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest()})
        actual = decode_hashes(combined)
        require(actual == expected, "Combined decoded source pixels differ; refusing to publish")
        media = base.probe(combined)
        require(media["frames"] == len(actual), "Combined duration/frame count mismatch")
        events = shifted_events(data)
        position = 0
        for event in events:
            require(event["frame_start"] == position, "Combined events are not contiguous")
            position += event["frame_count"]
        require(position == media["frames"] and sum(e["event"] == "executed" for e in events) == data["summary"]["steps"], "Combined event counts include replay or omit source footage")
        (publish/"full_episode.events.jsonl").write_text("".join(json.dumps(e, allow_nan=False)+"\n" for e in events))
        preview = events[-1]["frame_start"]
        media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-i", str(combined),
                       "-vf", f"select=eq(n\\,{preview})", "-frames:v", "1", "-threads", "1", str(publish/"video_preview.png")])
        (publish/"index.html").write_text(gallery(data, media))
        (publish/"RESULTS.md").write_text("# One continued direct-history episode\n\n"+result_text(data, media)+
            "\n[Video](full_episode.mp4) · [Gallery](index.html) · [Events](full_episode.events.jsonl) · [Proof](video_package_manifest.json) · [Terminal summary](continuation/summary.json)\n")
        write = lambda name, value: (publish/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        write("api_log_index.json", {"api_requests_made_by_packager": 0,
              "scope": "Metadata and source-log hashes only. Opaque response history and payloads remain in immutable source runs.",
              "segments": [{"label": s["label"], "requests": s["api_metadata"]} for s in data["segments"]]})
        files = {p.relative_to(publish).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(publish.rglob("*")) if p.is_file()}
        proof = {"kind": "one_direct_history_episode_with_budget_continuation", "created_utc": datetime.now(timezone.utc).isoformat(),
                 "simulation_episode_count": 1, "terminal_status": data["summary"]["status"],
                 "evaluation_completed": data["summary"]["evaluation_completed"], "reconstruction_actions_counted_as_new": 0,
                 "cumulative_actions": data["summary"]["steps"], "cumulative_api_calls": data["summary"]["api_calls"],
                 "approved_cumulative_cap_usd": 300, "ledger_validation": data["ledger"], "restoration_validation": data["restoration"], "media": media,
                 "api_waits_omitted": True, "reconstruction_footage_omitted": True,
                 "source_segments": [{"label": s["label"], "identity_hash": s["manifest"]["identity_hash"],
                                      "video_sha256": sha(s["video"]), "events_sha256": sha(s["events_path"]),
                                      "start_step": s["start_step"], "trace_validation": s["trace_validation"],
                                      "frozen_source_checks": s["frozen_source_checks"]} for s in data["segments"]],
                 "exact_frame_preservation": {"passed": True, "decoded_pieces": piece_proofs,
                      "method": "Every source frame and editorial card frame decoded; concatenated pixel hashes equal combined decode in exact order",
                      "frames": len(actual), "decoder_errors": [], "decoder_exit_codes": [0]*(len(pieces)+1),
                      "combined_pixel_hash_sequence_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest()},
                 "events": {"counts": dict(Counter(e["event"] for e in events)), "contiguous_frames": True,
                            "historical_terminal_is_intermediate_pause": True},
                 "preview": {"frame_index_zero_based": preview, "time_s": preview/FPS, "unedited_decoded_frame": True},
                 "files": files}
        write("video_package_manifest.json", proof)
        base.validate_links(publish)
        tmp_archive = staging/archive.name
        with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(publish.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path(output.name)/path.relative_to(publish)).as_posix(),
                                 compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(tmp_archive) as bundle:
            require(bundle.testzip() is None, "ZIP integrity error")
            for path in (p for p in publish.rglob("*") if p.is_file()):
                content = bundle.read((Path(output.name)/path.relative_to(publish)).as_posix())
                require(hashlib.sha256(content).hexdigest() == sha(path), f"ZIP content differs: {path.name}")
        # Frozen source and copied source videos must still match after packaging.
        for segment in data["segments"]:
            require(sha(segment["video"]) == files[f"{segment['label']}/episode.mp4"]["sha256"], "Source video changed while packaging")
        publish.rename(output)
        tmp_archive.rename(archive)
    return {"output": str(output), "archive": str(archive), "archive_sha256": sha(archive),
            "archive_bytes": archive.stat().st_size, "media": media, "simulation_episode_count": 1,
            "cumulative_actions": data["summary"]["steps"], "terminal_status": data["summary"]["status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--continuation", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    parent, child = args.parent.resolve(), args.continuation.resolve()
    try:
        data = validate(parent, child)
        if args.check_only:
            result = {"ready": True, "writes": 0, "cumulative_actions": data["summary"]["steps"],
                      "terminal_status": data["summary"]["status"], "ledger": data["ledger"]}
        else:
            result = package(data, (args.output or child.with_name(child.name+"_full_episode")).resolve())
    except (NotReady, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
