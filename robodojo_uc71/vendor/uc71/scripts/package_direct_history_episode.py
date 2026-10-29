#!/usr/bin/env python3
"""Validate and package one terminal Astra direct-actions-with-history episode.

No API requests, simulator, policy imports, or source-run mutations. API payloads
and opaque response history stay in the source run; the package contains a hash
and metadata index of those logs. Use --check-only while checking readiness.
Offline smoke is rejected unless explicitly requested and remains unscored.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


FPS, WIDTH, HEIGHT = 20, 1600, 960
LIVE_MODE = "live_astra_direct_history"
SMOKE_MODE = "offline_mock_smoke"
ALLOWED_EVENTS = {"observation", "action_chunk", "direct_action", "executed", "terminal"}
PROJECT = Path(__file__).resolve().parents[1]


class NotReady(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise NotReady(message)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise NotReady(f"Missing or invalid JSON: {path}") from exc


def read_jsonl(path):
    try:
        with path.open() as source:
            return [json.loads(line) for line in source if line.strip()]
    except (OSError, ValueError) as exc:
        raise NotReady(f"Missing or invalid JSONL: {path}") from exc


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_file(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"Artifact is not a relative local file: {name}")
    path = root / relative
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f"Missing or external artifact: {path}")
    return path


def media_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NotReady(f"Media command failed: {command[0]}: {exc}") from exc
    require(not result.stderr.strip(), f"Media decoder errors: {result.stderr.strip()}")
    return result.stdout


def probe(video):
    require(shutil.which("ffprobe") is not None, "ffprobe is required")
    info = json.loads(media_command([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(video),
    ]))
    require(len(info.get("streams", [])) == 1, "Expected one video stream")
    stream = info["streams"][0]
    require(stream.get("codec_type") == "video" and stream.get("codec_name") == "h264"
            and stream.get("pix_fmt") == "yuv420p", "Unexpected video codec")
    require((stream.get("width"), stream.get("height")) == (WIDTH, HEIGHT), "Unexpected video dimensions")
    require(Fraction(stream["r_frame_rate"]) == FPS, "Unexpected video frame rate")
    frames = int(stream["nb_read_frames"])
    require(frames > 0 and abs(float(info["format"]["duration"]) - frames / FPS) <= 1 / FPS,
            "Incomplete video duration/frame count")
    return {"codec": "h264", "width": WIDTH, "height": HEIGHT, "fps": FPS,
            "frames": frames, "duration_s": frames / FPS}


def full_decode(video, frames):
    require(shutil.which("ffmpeg") is not None, "ffmpeg is required")
    print(f"Decoding every direct-history video frame: {video.name}", file=sys.stderr, flush=True)
    decoded = media_command([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-f", "framemd5", "-",
    ])
    hashes = [line.rsplit(",", 1)[1].strip() for line in decoded.splitlines() if line and not line.startswith("#")]
    require(len(hashes) == frames, "Full decoded frame count differs from event/probe count")
    return {"method": "full framemd5 pixel decode", "exit_code": 0, "decoder_errors": [],
            "frames": len(hashes), "pixel_hash_sequence_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest()}


def model_actions(response):
    """Read only structured action answer text; never inspect reasoning items."""
    texts = [part["text"] for item in response.get("output", []) if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    if not texts:
        return None
    try:
        value = json.loads("".join(texts))
    except ValueError:
        return None
    return value.get("actions") if isinstance(value, dict) else None


def api_index(root, summary, smoke):
    indexed, by_name = [], {}
    requests = sorted((root / "api").glob("*.request.json"))
    expected_calls = summary["mock_response_calls"] if smoke else summary["api_calls"]
    require(len(requests) == expected_calls, "Request log count differs from terminal API-attempt count")
    for ordinal, path in enumerate(requests, 1):
        request = read_json(path)
        require(request.get("request_ordinal") == ordinal and request.get("mode") == "direct", "Nonsequential or non-direct API request")
        require(request.get("prompt_version") == "astra-direct-history-v1"
                and request.get("history_protocol") == "responses_output_transcript_v1", "Unexpected direct-history protocol")
        require(request.get("image_history") == 2, "Unexpected image-history retention policy")
        payload = request.get("payload", {})
        require(payload.get("model") == "gpt-6-astra", "Request used an unexpected model")
        require(payload.get("text", {}).get("format", {}).get("name") == "astra_direct_history_v1", "Request used a different output schema")
        result_path = path.with_name(path.name.replace(".request.json", ".result.json"))
        response_path = path.with_name(path.name.replace(".request.json", ".response.json"))
        result = read_json(result_path)
        response = read_json(response_path) if response_path.exists() else None
        if response is not None:
            require(bool(response.get("offline_mock")) == smoke, "Live/model and scripted-mock provenance conflict")
        metadata = {
            "ordinal": ordinal, "request_key": request["request_key"], "request": path.relative_to(root).as_posix(),
            "result_status": result.get("status"), "history_stats_before_request": request.get("history_stats_before_request"),
            "response_id": response.get("id") if response else None,
            "response_status": response.get("status") if response else None,
            "usage": response.get("usage") if response else None,
            "files": {artifact.relative_to(root).as_posix(): {"sha256": sha(artifact), "bytes": artifact.stat().st_size}
                      for artifact in (path, response_path, result_path) if artifact.exists()},
        }
        indexed.append(metadata)
        # Retain compact action provenance only, not every growing history payload.
        by_name[path.name] = {
            "request": {"request_key": request["request_key"],
                        "history_stats_before_request": request.get("history_stats_before_request")},
            "response_present": response is not None,
            "model_actions": model_actions(response) if response is not None else None,
            "result_status": result.get("status"), "metadata": metadata,
        }
    return indexed, by_name


def close_vectors(actual, expected, name):
    require(len(actual) == len(expected) and all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                               for a, b in zip(actual, expected)), f"Measured/commanded feedback mismatch: {name}")


def validate_trace(root, summary, events, requests):
    trace = local_file(root, summary["trace"])
    cursor = executed = issued = feedback_count = chunks = 0
    pending = None
    last_observation = None
    current_chunk = None
    grippers = Counter()
    last_executed = None
    with trace.open() as source:
        for line in source:
            entry = json.loads(line)
            kind = entry.get("event")
            require(cursor < len(events), "More trace events than recorded video events")
            if kind == "reset":
                require(cursor == 0 and events[cursor]["event"] == "observation", "Initial observation mismatch")
                last_observation = entry["observation"]
                require(events[cursor]["state"] == last_observation["observation/state"], "Initial video state mismatch")
                cursor += 1
            elif kind == "action":
                require(pending is None and entry["step"] == executed, "Unmatched or nonsequential issued action")
                require(entry["state"] == last_observation["observation/state"], "Action used a different measured state")
                diagnostics = entry["diagnostics"]
                require(diagnostics.get("method") == "direct_history" and diagnostics.get("status") == "direct_action",
                        "Non-direct policy/planner event in direct recording")
                require("plan" not in diagnostics and "field_prediction" not in diagnostics, "Field planning unexpectedly present")
                require(diagnostics["chunk_index"] == executed // 5 and diagnostics["chunk_step"] == executed % 5,
                        "Chunk position does not match global action index")
                if diagnostics["refreshed"]:
                    require(executed % 5 == 0, "Action chunk refreshed mid-chunk")
                    event = events[cursor]
                    current_chunk = diagnostics["prediction"]["actions"]
                    require(event["event"] == "action_chunk" and event["step"] == executed
                            and event["actions"] == current_chunk and event["action"] == [], "Model chunk/video mismatch")
                    require(len(current_chunk) == 5 and all(len(action) == 7 for action in current_chunk), "Invalid action chunk dimensions")
                    source_request = requests.get(Path(diagnostics["api_request_log"]).name)
                    require(source_request is not None, "Model chunk has no corresponding API request log")
                    require(source_request["request"]["request_key"] == diagnostics["api_request_key"] == event["api_request_key"],
                            "Model chunk request-key mismatch")
                    require(source_request["result_status"] == "valid", "Model chunk points to an unvalidated API result")
                    require(source_request["response_present"] and source_request["model_actions"] == current_chunk,
                            "Recorded action chunk differs from actual structured API answer")
                    before = source_request["request"].get("history_stats_before_request", {})
                    require(before.get("observations") == chunks and before.get("feedback_records") == executed,
                            "Request history counts disagree with preceding observed execution")
                    require(event["api_latency_s"] == diagnostics["api_latency_s"], "Request latency mismatch")
                    chunks += 1
                    cursor += 1
                event = events[cursor]
                require(event["event"] == "direct_action" and event["step"] == executed, "Issued action/video ordering mismatch")
                require(event["state"] == entry["state"] and event["action"] == entry["action"], "Issued action/video data mismatch")
                require(current_chunk is not None and entry["action"] == current_chunk[executed % 5], "Issued command differs from model chunk")
                require(event["history_stats"] == diagnostics["history_stats"], "Displayed decision history metadata mismatch")
                pending = entry
                issued += 1
                cursor += 1
            elif kind == "executed":
                require(pending is not None and entry["step"] == executed, "Executed action has no matching issuance")
                require(entry["action"] == pending["action"], "Executed command differs from issued command")
                require(entry["next_state"] == entry["observation"]["observation/state"], "Executed state/camera observation mismatch")
                last_executed = entry
                last_observation = entry["observation"]
                executed += 1
                grip = entry["action"][6]
                grippers["close" if grip > 0 else "open" if grip < 0 else "neutral"] += 1
            elif kind == "feedback":
                require(last_executed is not None and pending is not None and entry["step"] == executed - 1,
                        "Feedback lacks an observed execution")
                feedback = entry["feedback"]
                require(feedback["type"] == "execution_feedback" and feedback["before_state"] == pending["state"]
                        and feedback["after_state"] == last_executed["next_state"]
                        and feedback["normalized_action"] == pending["action"], "Feedback action/state provenance mismatch")
                require(not any(key in feedback for key in ("success", "reward", "object_pose", "contact")), "Evaluator-only data appears in feedback")
                measured = [feedback["after_state"][i] - feedback["before_state"][i] for i in range(3)]
                close_vectors(feedback["measured_translation_delta_m"], measured, "translation")
                close_vectors(feedback["desired_translation_delta_m"], [0.05 * x for x in pending["action"][:3]], "desired translation")
                event = events[cursor]
                require(event["event"] == "executed" and event["step"] == entry["step"], "Observed-execution video event mismatch")
                require(event["state"] == last_executed["next_state"] and event["action"] == pending["action"]
                        and event["success"] == last_executed["success"] and event["feedback"] == feedback,
                        "Executed video evidence differs from actual measured transition")
                close_vectors(event["measured_delta_xyz_m"], measured, "video translation")
                feedback_count += 1
                pending = None
                cursor += 1
            else:
                raise NotReady(f"Unexpected trace event: {kind}")
    require(cursor == len(events) - 1 and events[cursor]["event"] == "terminal", "Trace/video does not end at the same terminal event")
    require(events[-1]["state"] == last_observation["observation/state"], "Terminal video state is not the final actual observation")
    require(executed == summary["steps"] and feedback_count == summary["measured_feedback_count"]
            and chunks == summary["validated_action_chunks"], "Terminal trace/result counts disagree")
    pending_index = pending["step"] if pending is not None else None
    require(pending_index == summary["pending_action_index"], "Unconfirmed action index disagrees with terminal summary")
    for label in ("close", "open", "neutral"):
        require(grippers[label] == summary[f"gripper_{label}_actions"], f"Gripper count mismatch: {label}")
    return {"executed_actions": executed, "issued_actions": issued, "unconfirmed_issued_action_index": pending_index,
            "accepted_action_chunks": chunks, "feedback_records": feedback_count, "gripper_counts": dict(grippers),
            "trace_sha256": sha(trace), "trace_bytes": trace.stat().st_size,
            "all_video_actions_match_recorded_model_chunks": True, "all_video_feedback_matches_measured_transitions": True}


def validate(root, allow_smoke=False):
    state = read_json(root / "run_state.json")
    require(state.get("stage") == "finished" and state.get("status") != "running", "Episode is still live; no package will be written")
    summary = read_json(root / "summary.json")
    require(summary == read_json(root / "runner_summary.json"), "Runner and published summary disagree")
    require(read_jsonl(root / "episodes.jsonl") == [summary], "Expected one identical terminal episode result")
    require(summary["status"] == state["status"], "Terminal state/summary mismatch")
    smoke = summary.get("mode") == SMOKE_MODE
    require(summary.get("mode") in (LIVE_MODE, SMOKE_MODE) and (not smoke or allow_smoke),
            "Scripted smoke is not a live model evaluation; explicit --allow-offline-smoke is required")
    require(not list(root.glob("*.partial.mp4")), "Unfinalized MP4 remains")
    manifest = read_json(root / "manifest.json")
    config = manifest["configuration"]
    require(config.get("kind") == "fresh_direct_history_baseline" and config.get("mode") == summary["mode"], "Wrong experiment provenance")
    require(config["policy"]["field_generation"] is False and config["policy"]["local_field_planning"] is False,
            "Direct experiment incorrectly claims field planning")
    require(config["policy"]["success_feedback_to_policy"] is False, "Evaluator success fed into policy")
    require(summary["task_id"] == summary["episode_index"] == 0 and summary["seed"] == 7, "Unexpected task/initial-state/seed")
    require(summary["budget_scope"] == "this fresh run only", "Prior field budget inherited by fresh direct experiment")
    completed = summary["status"] == "success" and not smoke
    require(summary["evaluation_completed"] is completed and summary["scored_episodes"] == int(completed), "Incorrect episode scoring")
    require(summary["success_rate"] == (1.0 if completed else None), "Incomplete or diagnostic episode scored as a failure")
    source_checks = {}
    for name, expected in config["software"]["source_sha256"].items():
        path = PROJECT / "src" / "gpt_field" / name
        source_checks[name] = sha(path) == expected
    runner = PROJECT / config["runner"]["path"]
    source_checks[config["runner"]["path"]] = sha(runner) == config["runner"]["sha256"]
    require(all(source_checks.values()), "A source recorded in the frozen run manifest changed")
    video = local_file(root, summary.get("video") or "MISSING.mp4")
    events_path = local_file(root, video.with_suffix(".events.jsonl").name)
    events = read_jsonl(events_path)
    require(events and events[0]["event"] == "observation" and events[-1]["event"] == "terminal", "Missing initial or terminal video event")
    position = 0
    for event in events:
        require(event.get("event") in ALLOWED_EVENTS and event.get("policy") == "direct_history", "Unexpected field/planner video event")
        require(event.get("offline_smoke") is smoke, "Video mock/live label mismatch")
        require(event.get("frame_start") == position and type(event.get("frame_count")) is int and event["frame_count"] > 0,
                "Video events do not span contiguous frames")
        require(abs(event["video_time_s"] - position / FPS) < 1e-7, "Video event timestamp/frame mismatch")
        position += event["frame_count"]
    require(events[-1]["status"] == summary["status"], "Recorded video outcome differs from final summary (including cleanup errors)")
    require(events[-1]["success"] is (summary["status"] == "success"), "Terminal video success flag mismatch")
    metadata, requests = api_index(root, summary, smoke)
    checked = validate_trace(root, summary, events, requests)
    media = probe(video)
    require(media["frames"] == position, "Encoded frame count differs from event index")
    return {"root": root, "summary": summary, "manifest": manifest, "video": video,
            "events_path": events_path, "events": events, "media": media, "api_metadata": metadata,
            "trace_validation": checked, "frozen_source_checks": source_checks, "smoke": smoke}


def gallery(data):
    row, media = data["summary"], data["media"]
    title = "Scripted integration diagnostic" if data["smoke"] else "Astra / LIBERO direct actions + history"
    outcome = "Unscored mock smoke" if data["smoke"] else "LIBERO success reached" if row["evaluation_completed"] else f"Incomplete: {row['status'].replace('_', ' ')}"
    api_calls = row["mock_response_calls"] if data["smoke"] else row["api_calls"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#0c121e;color:#e8effa}}a{{color:#79d9ee}}video,img{{width:100%;height:auto}}.outcome{{padding:18px;background:#243548;border-radius:8px}}small{{color:#adc0d8}}</style></head><body>
<h1>{html.escape(title)}</h1><p>{html.escape(row['task'])}</p>
<p class="outcome"><strong>{html.escape(outcome)}</strong><br>One fresh simulation episode: task 0, initial state 0, seed 7. {row['steps']} confirmed executed actions; {row['validated_action_chunks']} accepted five-action chunks; {api_calls} {'mock responses' if data['smoke'] else 'API attempts'}. Prior field episodes and budgets are separate.</p>
<p>Episode counts: {row['requested_episodes']} requested, {row['scored_episodes']} scored, {row['incomplete_episodes']} incomplete. {'This diagnostic has no task success rate.' if data['smoke'] else 'No success rate is assigned to this incomplete episode.' if not row['evaluation_completed'] else ''}</p>
<p>Terminal accounting upper bound: ${row['budget']['cost_usd_upper_bound']:.4f}. API attempts with unknown usage: {row['budget']['unknown_usage_calls']}. This configured estimate includes retained reservations for unknown usage.</p>
<video controls preload="metadata" poster="video_preview.png" src="episode.mp4"></video>
<p><a href="episode.mp4" download>Download annotated MP4</a> · <a href="episode.events.jsonl">Event index</a> · <a href="RESULTS.md">Result interpretation</a> · <a href="video_package_manifest.json">Validation and hashes</a></p>
<p><strong>API waits are omitted.</strong> Annotated playback shows the actual action chunk, issued action, subsequent camera observations and measured feedback. The path plots show measured robot positions. No field or local field planner is used.</p>
<p>{media['frames']:,} frames, 1600 × 960 at 20 fps; {media['duration_s']:.2f} seconds of annotated playback. Recorded episode wall time: {row['elapsed_s']:.2f} seconds. Confirmed control simulation: {row['steps']/20:.2f} seconds, excluding reset settlement.</p>
<p>History retains observation/feedback text and accepted ordered Responses output items; the latest two observation image pairs are kept. Opaque reasoning items are not displayed. Task success is checked separately by LIBERO.</p>
<p>{'This scripted diagnostic is not model performance.' if data['smoke'] else 'Task success is established only when LIBERO reports success; budget, user-stop and infrastructure outcomes remain unscored.'}</p>
<h2>Actual terminal frame</h2><img src="video_preview.png" alt="Decoded final direct-action episode frame"><p><small>The preview is decoded from the recorded terminal event.</small></p>
<p><a href="summary.json">Terminal summary</a> · <a href="manifest.json">Frozen protocol</a> · <a href="api_log_index.json">API metadata and source-log hashes</a></p>
</body></html>'''


def validate_links(directory):
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links = []
        def handle_starttag(self, tag, attrs):
            self.links.extend(value for key, value in attrs if key in ("href", "src", "poster"))
    links = Links(); links.feed((directory / "index.html").read_text())
    for name in links.links:
        local_file(directory, name)
    return len(links.links)


def package(data, output):
    root = data["root"]
    require(not output.exists() and not output.is_relative_to(root), "Output must be a fresh directory outside the source run")
    archive = output.with_name(output.name + ".zip")
    require(not archive.exists(), "Archive output already exists")
    decoding = full_decode(data["video"], data["media"]["frames"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-build-", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        sources = [data["video"], data["events_path"]]
        sources += [root / name for name in ("manifest.json", "summary.json", "runner_summary.json", "episodes.jsonl",
                    "run_state.json", "simulator.json", "authorization.json", "audit.json") if (root / name).is_file()]
        for source in sources:
            shutil.copyfile(source, staging / source.name)
        preview_index = data["events"][-1]["frame_start"]
        media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
                       "-i", str(data["video"]), "-vf", f"select=eq(n\\,{preview_index})",
                       "-frames:v", "1", "-threads", "1", str(staging / "video_preview.png")])
        (staging / "index.html").write_text(gallery(data))
        (staging / "api_log_index.json").write_text(json.dumps({
            "scope": "Metadata/hashes only. Raw payloads, camera encodings and opaque response items remain in the source run.",
            "api_requests_made_by_packager": 0, "mode": data["summary"]["mode"], "requests": data["api_metadata"],
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        row = data["summary"]
        label = "Unscored scripted diagnostic" if data["smoke"] else "LIBERO success reached" if row["evaluation_completed"] else f"Incomplete: {row['status']}"
        (staging / "RESULTS.md").write_text(
            f"# Direct actions with history\n\n**{label}.**\n\n"
            f"One fresh simulation episode; {row['steps']} confirmed executed actions, {row['validated_action_chunks']} accepted five-action chunks, "
            f"and {row['measured_feedback_count']} measured feedback records. "
            "No field generation or local field planning is used. Earlier field runs and their budgets are separate.\n\n"
            f"Episode counts: {row['requested_episodes']} requested, {row['scored_episodes']} scored, {row['incomplete_episodes']} incomplete. "
            + ("This diagnostic has no task success rate.\n\n" if data['smoke'] else
               "No success rate is assigned to this incomplete episode.\n\n" if not row['evaluation_completed'] else "\n\n")
            + f"Terminal accounting upper bound: ${row['budget']['cost_usd_upper_bound']:.4f}. "
            f"API attempts with unknown usage: {row['budget']['unknown_usage_calls']}. "
            "This configured estimate includes retained reservations for unknown usage.\n\n"
            f"Unconfirmed issued action index: {row['pending_action_index']}. "
            f"Gripper commands: {row['gripper_close_actions']} close, {row['gripper_open_actions']} open, {row['gripper_neutral_actions']} neutral. "
            "A close command alone is not evidence of a grasp.\n\n"
            "API waits are omitted from annotated playback. Incomplete budget/interruption/infrastructure outcomes are unscored.\n\n"
            "[Video](episode.mp4) · [Offline gallery](index.html) · [Events](episode.events.jsonl) · "
            "[Terminal summary](summary.json) · [Protocol](manifest.json) · [API metadata](api_log_index.json) · [Validation](video_package_manifest.json)\n")
        # The manifest link is created below; validate all gallery targets after writing it.
        hashes = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(staging.iterdir()) if path.is_file()}
        proof = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "one_direct_history_episode",
                 "mode": row["mode"], "physical_episode_count": 1, "terminal_status": row["status"],
                 "evaluation_completed": row["evaluation_completed"], "libero_success": row["libero_success"],
                 "media": data["media"], "full_decode": decoding, "api_waits_omitted": True,
                 "field_generation": False, "local_field_planning": False,
                 "event_counts": dict(Counter(event["event"] for event in data["events"])),
                 "trace_validation": data["trace_validation"], "frozen_source_checks": data["frozen_source_checks"],
                 "source_trace": {"name": row["trace"], "included_in_package": False,
                                  "sha256": data["trace_validation"]["trace_sha256"]},
                 "preview": {"frame_index_zero_based": preview_index, "video_time_s": preview_index / FPS,
                             "unedited_decoded_frame": True}, "files": hashes}
        (staging / "video_package_manifest.json").write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n")
        validate_links(staging)
        staged_archive = Path(tmp) / archive.name
        with zipfile.ZipFile(staged_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(staging.iterdir()):
                bundle.write(path, (Path(output.name) / path.name).as_posix(),
                             compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(staged_archive) as bundle:
            require(bundle.testzip() is None, "ZIP integrity check failed")
            for name, meta in hashes.items():
                content = bundle.read((Path(output.name) / name).as_posix())
                require(hashlib.sha256(content).hexdigest() == meta["sha256"] and len(content) == meta["bytes"], f"ZIP hash mismatch: {name}")
        staging.rename(output)
        staged_archive.rename(archive)
    return {"output": str(output), "archive": str(archive), "archive_sha256": sha(archive),
            "archive_bytes": archive.stat().st_size, "terminal_status": row["status"],
            "frames": data["media"]["frames"], "duration_s": data["media"]["duration_s"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-offline-smoke", action="store_true", help="Only for explicit unscored existing mock recordings")
    args = parser.parse_args(argv)
    root = args.run_directory.resolve()
    try:
        data = validate(root, args.allow_offline_smoke)
        if args.check_only:
            result = {"ready": True, "mode": data["summary"]["mode"], "terminal_status": data["summary"]["status"],
                      "media": data["media"], "trace_validation": data["trace_validation"], "writes": 0}
        else:
            output = (args.output or root.with_name(root.name + "_video_package")).resolve()
            result = package(data, output)
    except (NotReady, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
