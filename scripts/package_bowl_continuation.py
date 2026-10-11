#!/usr/bin/env python3
"""Package a terminal bowl continuation as one faithfully joined episode.

This offline helper makes no API requests, imports no policy, and opens no
simulator. It writes only to a fresh output directory and its sibling ZIP.
The source budget-stop ending is retained, followed by a labeled editorial
resume card and the continuation recording. Historical reconstruction never
adds to evaluated action or field counts.

Example:
  .venv/bin/python scripts/package_bowl_continuation.py \
    --parent runs/astra_bowl_unbounded_002 \
    --continuation runs/astra_bowl_continuation_001

Repeat --continuation for each later infrastructure recovery or cumulative-budget
handoff segment, in chronological order. Every source segment must be terminal.
Use --check-only for a read-only terminal-artifact readiness check.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont


FPS, WIDTH, HEIGHT = 20, 1600, 960
CARD_SECONDS = 5
CARD_FRAMES = FPS * CARD_SECONDS
TERMINAL = {"success", "budget_exceeded", "user_stop", "infrastructure_error"}


class NotReady(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise NotReady(message)


def log(message):
    print(message, file=sys.stderr, flush=True)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise NotReady(f"Missing or invalid JSON: {path}") from exc


def jsonl(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise NotReady(f"Missing or invalid JSONL: {path}") from exc


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(root, name):
    path = root / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts,
            f"Artifact path must be relative: {name}")
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f"Missing local artifact: {path}")
    return path


def run(command, timeout=600):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NotReady(f"Media command failed: {command[0]}: {exc}") from exc
    require(not result.stderr.strip(), f"Media decoder reported errors: {result.stderr.strip()}")
    return result.stdout


def probe(path):
    info = json.loads(run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", str(path)]))
    streams = info.get("streams", [])
    require(len(streams) == 1 and streams[0].get("codec_type") == "video", f"Expected one video stream: {path}")
    stream = streams[0]
    require(stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p", f"Unexpected video codec: {path}")
    require((stream.get("width"), stream.get("height")) == (WIDTH, HEIGHT), f"Unexpected video dimensions: {path}")
    require(Fraction(stream["r_frame_rate"]) == FPS, f"Unexpected frame rate: {path}")
    frames = int(stream["nb_read_frames"])
    require(frames > 0 and abs(float(info["format"]["duration"]) - frames / FPS) <= 1 / FPS,
            f"Invalid video frame count/duration: {path}")
    return {"frames": frames, "duration_s": frames / FPS, "fps": FPS,
            "width": WIDTH, "height": HEIGHT, "codec": "h264", "pix_fmt": "yuv420p"}


def decoded_hashes(path):
    """Decode every frame and retain only its pixel hash; timestamps are checked separately."""
    log(f"Decoding every frame: {path.name}")
    output = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
                  "-i", str(path), "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
                  "-f", "framemd5", "-"], timeout=1800)
    result = [line.rsplit(",", 1)[1].strip() for line in output.splitlines() if line and not line.startswith("#")]
    require(result, f"No frames decoded from {path}")
    return result


def validate_trace(path, events, *, initial_event, start_step, expected_actions):
    """Cross-check every rendered action/state/selection with its actual source trace."""
    cursor, actions, refreshes = 0, 0, 0
    first_observation = last_observation = None
    try:
        with gzip.open(path, "rt") as source:
            for line in source:
                entry = json.loads(line)
                require(cursor < len(events), f"More trace records than video events: {path}")
                if entry.get("event") == initial_event:
                    require(first_observation is None, f"Duplicate initial observation: {path}")
                    first_observation = last_observation = entry["observation"]
                    event = events[cursor]
                    require(event["event"] == "observation" and event["step"] == start_step,
                            f"Initial video observation mismatch: {path}")
                    require(event["state"] == first_observation["observation/state"], f"Initial measured state mismatch: {path}")
                    cursor += 1
                elif entry.get("event") == "executed":
                    event = events[cursor]
                    require(event["event"] == "executed" and event["step"] == entry["step"] == start_step + actions,
                            f"Nonsequential action/video event: {path}")
                    require(event["state"] == entry["next_state"] == entry["observation"]["observation/state"],
                            f"Executed measured state mismatch: {path}")
                    require(event["success"] == entry["success"], f"Executed success mismatch: {path}")
                    last_observation = entry["observation"]
                    actions += 1
                    cursor += 1
                else:
                    require(entry["step"] == start_step + actions, f"Nonsequential decision: {path}")
                    require(last_observation is not None and entry["state"] == last_observation["observation/state"],
                            f"Decision does not use latest measured state: {path}")
                    diagnostics = entry["diagnostics"]
                    if diagnostics["refreshed"]:
                        event = events[cursor]
                        require(event["event"] == "field_refresh" and event["step"] == entry["step"],
                                f"Field refresh event mismatch: {path}")
                        require(event["selected_index"] is None and event["action"] == [], f"Field event contains an issued plan: {path}")
                        for key in ("api_request_key", "api_latency_s", "stage"):
                            require(event.get(key) == diagnostics.get(key), f"Field provenance mismatch for {key}: {path}")
                        require(diagnostics.get("field_prediction", {}).get("values"), f"No recorded field on refresh: {path}")
                        refreshes += 1
                        cursor += 1
                    event = events[cursor]
                    require(event["event"] == "local_replan" and event["step"] == entry["step"], f"Replan event mismatch: {path}")
                    require(event["state"] == entry["state"] and event["action"] == entry["action"], f"Recorded plan/action mismatch: {path}")
                    require(event["selected_index"] == diagnostics["plan"]["selected_index"], f"Selected candidate mismatch: {path}")
                    cursor += 1
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise NotReady(f"Trace validation failed for {path}: {exc}") from exc
    require(first_observation is not None and cursor == len(events) - 1, f"Trace/video sequence does not reach a unique terminal event: {path}")
    require(actions == expected_actions and events[-1]["state"] == last_observation["observation/state"], f"Terminal action count/state mismatch: {path}")
    return {"actions": actions, "fields": refreshes, "first_observation": first_observation,
            "last_observation": last_observation, "trace_sha256": sha(path)}


def validate_segment(root, row, *, start_step, actions, initial_event):
    require(not list(root.glob("*.partial.mp4")), f"Partial video remains in {root}")
    video = artifact(root, row.get("video") or "MISSING.mp4")
    require(video.suffix == ".mp4", f"Not an MP4: {video}")
    events_path = artifact(root, video.with_suffix(".events.jsonl").name)
    events = jsonl(events_path)
    require(events and events[0]["event"] == "observation" and events[-1]["event"] == "terminal", f"Missing initial/terminal video events: {root}")
    media = probe(video)
    position = 0
    for event in events:
        require(event.get("frame_start") == position and type(event.get("frame_count")) is int and event["frame_count"] > 0,
                f"Noncontiguous video frame spans: {root}")
        require(abs(event.get("video_time_s", -1) - position / FPS) < 1e-7, f"Frame/time disagreement: {root}")
        position += event["frame_count"]
    require(position == media["frames"], f"Event count disagrees with decoded frame count: {root}")
    require(events[-1]["status"] == row["status"] and events[-1]["success"] is (row["status"] == "success"),
            f"Terminal status mismatch: {root}")
    counts = Counter(event["event"] for event in events)
    require(counts["terminal"] == 1 and counts["executed"] == actions, f"Unexpected terminal/execution count: {root}")
    trace_path = artifact(root, row["log"])
    trace = validate_trace(trace_path, events, initial_event=initial_event, start_step=start_step, expected_actions=actions)
    return {"root": root, "row": row, "video": video, "events_path": events_path,
            "events": events, "media": media, "counts": dict(counts), "trace": trace}


def collect_first(parent, child):
    require(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    state = read_json(child / "run_state.json")
    require(state.get("stage") == "terminal" and state.get("status") in TERMINAL,
            "Continuation is not terminal; no package will be written")
    parent_rows, child_rows = jsonl(parent / "episodes.jsonl"), jsonl(child / "episodes.jsonl")
    require(len(parent_rows) == len(child_rows) == 1, "Expected exactly one episode result per segment")
    original, continued = parent_rows[0], child_rows[0]
    require(original.get("status") == "budget_exceeded", "Original recording must retain its budget-stop outcome")
    require(continued.get("status") == state.get("status"), "Continuation terminal records disagree")
    offset = continued.get("continuation_offset")
    require(type(offset) is int and offset == original.get("steps"), "Continuation offset differs from historical actions")
    require(continued.get("cumulative_actions") == offset + continued.get("new_evaluated_actions", -1), "Cumulative action count double-counts or omits actions")
    p_manifest, c_manifest = read_json(parent / "manifest.json"), read_json(child / "manifest.json")
    config = c_manifest["configuration"]
    require(config.get("kind") == "same_episode_budget_continuation", "Wrong continuation kind")
    require(config.get("parent_identity_hash") == p_manifest["identity_hash"] == continued.get("parent_identity_hash"), "Parent episode identity mismatch")
    for name, expected in config["parent_artifact_sha256"].items():
        require(sha(artifact(parent, name)) == expected, f"Parent source artifact changed: {name}")
    for key in ("suite", "task_id", "episode_index", "seed"):
        require(original.get(key) == continued.get(key) == config.get(key), f"Episode identity differs for {key}")
    require(continued.get("evaluation_completed") is (continued["status"] == "success"), "Incomplete continuation marked as completed evaluation")
    proof = read_json(child / "replay_validation.json")
    require(proof.get("status") == "validated" and proof.get("completed_actions") == offset,
            "Exact historical reconstruction proof is missing")
    require(proof.get("state_checks") == offset + 1 and proof.get("image_pairs_checked") == offset + 1,
            "Historical reconstruction did not check every state/image pair")
    require(proof.get("max_state_error") == 0 and proof.get("images_match") is True,
            "Historical reconstruction was not exact")
    require(proof.get("astra_requests_during_reconstruction") == 0, "Reconstruction unexpectedly made new model requests")
    require(proof.get("same_simulator_retained_for_continuation") is True, "Reconstruction simulator was not retained for continuation")
    log("Checking original and continuation media against trace records")
    before = validate_segment(parent, original, start_step=0, actions=offset, initial_event="reset")
    after = validate_segment(child, continued, start_step=offset, actions=continued["new_evaluated_actions"], initial_event="continuation_start")
    require(before["trace"]["last_observation"] == after["trace"]["first_observation"],
            "Continuation starting cameras/state do not exactly match the original final observation")
    require(before["trace"]["trace_sha256"] == config["source_trace"]["snapshot_sha256"], "Reconstructed trace hash mismatch")
    require(continued["budget_after"]["api_calls"] == original["budget_after"]["api_calls"] + continued["additional_api_calls"], "Cumulative API ledger mismatch")
    before.update(label="parent", start_step=0, manifest=p_manifest)
    after.update(label="continuation_001", start_step=offset, manifest=c_manifest)
    return {"segments": [before, after], "before": before, "after": after,
            "parent_manifest": p_manifest, "child_manifest": c_manifest,
            "replay": proof, "offset": offset,
            "cap": continued["max_total_cost_usd"], "total_actions": continued["cumulative_actions"]}


def collect(parent, children):
    require(children, "At least one continuation recording is required")
    require(len({parent, *children}) == len(children) + 1, "Source segment directories must be distinct")
    data = collect_first(parent, children[0])
    for number, child in enumerate(children[1:], start=2):
        previous = data["segments"][-1]
        state = read_json(child / "run_state.json")
        require(state.get("stage") == "terminal" and state.get("status") in TERMINAL,
                f"Continuation is not terminal: {child}")
        rows = jsonl(child / "episodes.jsonl")
        require(len(rows) == 1, f"Expected one continuation terminal result: {child}")
        row = rows[0]
        require(row.get("status") == state.get("status"), f"Terminal records disagree: {child}")
        offset = data["total_actions"]
        require(row.get("continuation_offset") == offset, f"Continuation action offset is not contiguous: {child}")
        new_actions = row.get("new_evaluated_actions")
        require(type(new_actions) is int and new_actions >= 0 and row.get("cumulative_actions") == offset + new_actions,
                f"Continuation cumulative action count is incorrect: {child}")
        require(row.get("evaluation_completed") is (row["status"] == "success"),
                f"Incomplete continuation marked as completed evaluation: {child}")
        manifest = read_json(child / "manifest.json")
        config = manifest["configuration"]
        kind = config.get("kind")
        require(kind in {"same_episode_infrastructure_recovery", "same_episode_chain_budget_continuation"},
                f"Unexpected chained continuation kind: {child}")
        prior_status = previous["row"]["status"]
        prior_cap = previous["row"]["budget_after"]["max_cost_usd"]
        cap_change = config.get("cost_cap_change", {})
        current_cap = row.get("max_total_cost_usd")
        require(current_cap == config.get("api", {}).get("max_cost_usd"), f"Terminal cap differs from continuation protocol: {child}")
        if kind == "same_episode_chain_budget_continuation":
            require(prior_status == "budget_exceeded", f"Budget handoff lacks a recorded budget pause: {child}")
            require(cap_change.get("old_total_usd") == prior_cap and cap_change.get("new_total_usd") == current_cap
                    and current_cap > prior_cap and cap_change.get("only_configuration_change") == "max_cost_usd",
                    f"Budget handoff cap change is not explicit and cumulative: {child}")
        else:
            require(prior_status == "infrastructure_error" and current_cap == prior_cap,
                    f"Infrastructure recovery changed the approved cumulative cap or prior outcome: {child}")
        require(config.get("parent_identity_hash") == previous["manifest"]["identity_hash"] == row.get("parent_identity_hash"),
                f"Immediate parent segment identity mismatch: {child}")
        require(config.get("root_identity_hash") == data["parent_manifest"]["identity_hash"],
                f"Original physical episode identity mismatch: {child}")
        for key in ("suite", "task_id", "episode_index", "seed"):
            require(row.get(key) == data["before"]["row"].get(key) == config.get(key), f"Episode identity differs for {key}: {child}")
        chain = config.get("chain")
        require(isinstance(chain, list) and len(chain) == len(data["segments"]), f"Ordered history omits or repeats source segments: {child}")
        for entry, historical in zip(chain, data["segments"]):
            require(entry.get("identity_hash") == historical["manifest"]["identity_hash"], f"Historical segment identity mismatch: {child}")
            require(entry.get("action_start") == historical["start_step"] and entry.get("action_count") == historical["trace"]["actions"],
                    f"Historical action span mismatch: {child}")
            require(entry.get("trace", {}).get("snapshot_sha256") == historical["trace"]["trace_sha256"],
                    f"Historical trace hash mismatch: {child}")
            require(entry["trace"].get("available_completed_actions") == historical["trace"]["actions"],
                    f"Historical replay counted incorrect source actions: {child}")
        composite = config.get("source_trace", {})
        require(composite.get("kind") == "ordered_chain" and composite.get("segments") == chain
                and composite.get("available_completed_actions") == offset, f"Composite replay provenance mismatch: {child}")
        for name, expected in config.get("parent_artifact_sha256", {}).items():
            require(sha(artifact(previous["root"], name)) == expected, f"Immediate-parent artifact changed: {name}")
        proof = read_json(child / "replay_validation.json")
        require(proof.get("status") == "validated" and proof.get("completed_actions") == offset
                and proof.get("state_checks") == offset + 1 and proof.get("image_pairs_checked") == offset + 1,
                f"Complete chain reconstruction proof is missing: {child}")
        require(proof.get("max_state_error") == 0 and proof.get("images_match") is True
                and proof.get("astra_requests_during_reconstruction") == 0
                and proof.get("same_simulator_retained_for_continuation") is True,
                f"Reconstruction was not exact or did not retain its simulator: {child}")
        require(proof.get("source_trace") == composite, f"Replay proof does not reference the prepared history: {child}")
        after = validate_segment(child, row, start_step=offset, actions=new_actions, initial_event="continuation_start")
        require(previous["trace"]["last_observation"] == after["trace"]["first_observation"],
                f"Starting images/state do not exactly match the preceding recording: {child}")
        prior_budget = previous["row"]["budget_after"]
        inherited = config.get("original_budget_snapshot", {})
        handoff = config.get("handoff_budget_snapshot", {})
        for key in ("api_calls", "input_tokens", "output_tokens", "cached_input_tokens", "unknown_usage_calls", "cost_usd_upper_bound"):
            require(inherited.get(key) == prior_budget.get(key), f"Original ledger differs for {key}: {child}")
            require(handoff.get(key) == prior_budget.get(key), f"Carried ledger differs for {key}: {child}")
        require(handoff.get("max_cost_usd") == current_cap,
                f"Handoff ledger does not use the recorded cumulative cap: {child}")
        require(row["budget_after"]["api_calls"] == prior_budget["api_calls"] + row["additional_api_calls"],
                f"Cumulative API call count omits the preceding reservation: {child}")
        require(row["additional_api_calls"] >= after["trace"]["fields"], f"Recorded fields exceed actual new API attempts: {child}")
        after.update(label=f"continuation_{number:03d}", start_step=offset, manifest=manifest)
        data["segments"].append(after)
        data.update(after=after, child_manifest=manifest, replay=proof,
                    cap=row["max_total_cost_usd"], total_actions=row["cumulative_actions"])
    return data


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def boundary_card(data, path):
    card = Image.new("RGB", (WIDTH, HEIGHT), (12, 18, 30))
    draw = ImageDraw.Draw(card)
    draw.text((92, 85), "SAME PHYSICAL EPISODE / CONTINUATION", font=font(39, True), fill=(232, 239, 250))
    draw.text((92, 183), f"Paused after {data['offset']} executed actions", font=font(51, True), fill=(255, 210, 115))
    lines = [
        ("The prior recording paused at its API budget guard." if data.get("previous_status", "budget_exceeded") == "budget_exceeded"
         else "The prior recording paused after an infrastructure error."),
        f"All {data['offset']} historical actions were reconstructed exactly.",
        "Reconstruction adds 0 evaluated actions and 0 new model calls.",
        f"Recorded action indices resume at step {data['offset']} (action {data['offset'] + 1}).",
        f"Approved cumulative API cap: ${data['cap']:g}, including prior usage.",
        "Field-refresh counters restart within the continuation recording.",
    ]
    for index, line in enumerate(lines):
        draw.text((92, 308 + index * 74), line, font=font(29), fill=(210, 226, 244))
    draw.text((92, 835), "Editorial boundary card. API waits and reconstruction footage are omitted.",
              font=font(24), fill=(141, 176, 200))
    card.save(path)


def create_card_video(png, mp4):
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-loop", "1",
         "-framerate", str(FPS), "-i", str(png), "-frames:v", str(CARD_FRAMES), "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
         "-threads", "1", "-movflags", "+faststart", str(mp4)])


def shifted_events(data):
    result = []
    frame_offset = cumulative_fields = 0
    for number, source in enumerate(data["segments"]):
        if number:
            result.append({"event": "editorial_resume_boundary", "segment": "editorial",
                           "resuming_segment": source["label"], "frame_start": frame_offset,
                           "frame_count": CARD_FRAMES, "video_time_s": frame_offset / FPS,
                           "completed_evaluated_actions": source["start_step"], "new_evaluated_actions": 0,
                           "reconstruction_actions_counted_as_new": 0,
                           "approved_cumulative_cap_usd": source["row"]["max_total_cost_usd"]})
            frame_offset += CARD_FRAMES
        for event in source["events"]:
            updated = dict(event)
            updated.update(segment=source["label"], source_event=event["event"],
                           source_frame_start=event["frame_start"], source_video_time_s=event["video_time_s"],
                           frame_start=frame_offset + event["frame_start"],
                           video_time_s=(frame_offset + event["frame_start"]) / FPS,
                           segment_field_refresh_count=event["field_refresh_count"],
                           cumulative_field_refresh_count=cumulative_fields + event["field_refresh_count"])
            if event["event"] == "terminal":
                final = number == len(data["segments"]) - 1
                updated["event"] = "terminal" if final else "historical_segment_pause"
                updated["is_final_outcome_of_combined_episode"] = final
            result.append(updated)
        frame_offset += source["media"]["frames"]
        cumulative_fields += source["counts"].get("field_refresh", 0)
    return result


def copy_source_snapshot(source, destination):
    """Copy root-level artifact types only; API payloads and full traces stay in source runs."""
    destination.mkdir()
    copied = []
    allowed = {".json", ".jsonl", ".md", ".html", ".png", ".mp4"}
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in allowed and ".partial." not in path.name:
            require(path.resolve().is_relative_to(source.resolve()), f"External source artifact: {path}")
            target = destination / path.name
            shutil.copyfile(path, target)
            copied.append(target)
    return copied


def gallery(data, duration, boundaries):
    row = data["after"]["row"]
    success = row["status"] == "success"
    outcome = "LIBERO success reached" if success else f"Incomplete: {row['status'].replace('_', ' ')}"
    total, historical = data["total_actions"], data["offset"]
    fresh = total - historical
    fields = sum(segment["counts"].get("field_refresh", 0) for segment in data["segments"])
    boundary_items = "".join(
        f"<li>{html.escape(boundary['preceding_segment'])} pauses at {boundary['source_pause_time_s']:.2f}s "
        f"with <code>{html.escape(boundary['preceding_status'])}</code>; "
        f"the labeled resume card begins at {boundary['card_start_s']:.2f}s, "
        f"and {html.escape(boundary['resuming_segment'])} starts at {boundary['resume_start_s']:.2f}s "
        f"after {boundary['completed_evaluated_actions']} historical actions.</li>"
        for boundary in boundaries)
    source_links = " · ".join(
        f'<a href="{segment["label"]}/{html.escape(segment["video"].name)}">{html.escape(segment["label"])} recording</a>'
        for segment in data["segments"])
    final_label = data["after"]["label"]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One Astra / LIBERO bowl episode with continuation</title><style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#0c121e;color:#e8effa}}a{{color:#79d9ee}}video,img{{width:100%;height:auto}}.outcome{{background:#243548;padding:18px;border-radius:8px}}small{{color:#abc0d8}}</style></head><body>
<h1>One bowl episode, continued across recorded pauses</h1><p>{html.escape(row['task_prompt'])}</p>
<p class="outcome"><strong>{html.escape(outcome)}</strong><br>{total} evaluated actions = {historical} original actions + {fresh} new actions across {len(data['segments'])-1} continuation recordings. Historical reconstruction contributes zero new evaluated actions. This is one physical episode.</p>
<video id="episode" controls preload="metadata" poster="video_preview.png" src="full_episode.mp4"></video>
<p><a href="full_episode.mp4" download>Download full episode MP4</a> · <a href="full_episode.events.jsonl">Combined event index</a> · <a href="video_package_manifest.json">Video proof and hashes</a> · <a href="RESULTS.md">Episode interpretation</a></p>
<p><strong>API waits and reconstruction footage are omitted.</strong> Playback preserves all source recordings. Every historical terminal pause remains visible, followed by a labeled five-second editorial resume card. Action indices continue across recordings; displayed field counters restart within each continuation clip.</p>
<ul>{boundary_items}</ul>
<p>{fields} actual recorded field refreshes; {duration:.2f}s combined playback at 20 fps. The approved ${data['cap']:g} cap covers cumulative usage across every recording. Terminal status: <code>{html.escape(row['status'])}</code>.</p>
<p>Historical pauses remain intermediate events in this combined episode. {'The final continuation records task success.' if success else 'Task completion was not established; no completed-failure benchmark score is assigned.'}</p>
<h2>Actual final recorded frame</h2><img src="video_preview.png" alt="Decoded final frame of the combined episode"><p><small>The preview is decoded directly from the final continuation event.</small></p>
<p>{source_links} · <a href="{final_label}/replay_validation.json">Final reconstruction proof</a> · <a href="{final_label}/manifest.json">Continuation protocol</a> · <a href="{final_label}/summary.json">Terminal continuation result</a></p>
</body></html>'''


def check_local_html(path, root):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []
        def handle_starttag(self, _tag, attrs):
            self.paths.extend(value for key, value in attrs if key in ("href", "src", "poster"))
    links = Links()
    links.feed(path.read_text())
    for relative in links.paths:
        if "://" in relative or relative.startswith("#"):
            continue
        target = (path.parent / relative.split("#", 1)[0]).resolve()
        require(target.is_relative_to(root.resolve()) and target.is_file(), f"Missing offline gallery link: {relative}")
    return len(links.paths)


def package(data, output):
    require(not output.exists(), f"Output must be a fresh directory: {output}")
    archive = output.with_name(output.name + ".zip")
    require(not archive.exists(), f"Archive already exists: {archive}")
    require(all(not output.is_relative_to(segment["root"]) for segment in data["segments"]),
            "Package output must be separate from all immutable source runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-build-", dir=output.parent) as directory:
        staging = Path(directory)
        publish = staging / output.name
        publish.mkdir()
        pieces, boundaries = [], []
        frame_offset = 0
        for number, segment in enumerate(data["segments"]):
            copy_source_snapshot(segment["root"], publish / segment["label"])
            if number:
                previous = data["segments"][number - 1]
                image_name = f"resume_boundary_{number:02d}.png"
                boundary_card({"offset": segment["start_step"], "cap": segment["row"]["max_total_cost_usd"],
                               "previous_status": previous["row"]["status"]}, publish / image_name)
                card_video = staging / f"card_{number:02d}.mp4"
                create_card_video(publish / image_name, card_video)
                pieces.append({"path": card_video, "kind": "editorial", "expected_frames": CARD_FRAMES})
                boundaries.append({"preceding_segment": previous["label"], "preceding_status": previous["row"]["status"],
                                   "resuming_segment": segment["label"],
                                   "source_pause_time_s": (frame_offset - previous["media"]["frames"] + previous["events"][-1]["frame_start"]) / FPS,
                                   "card_start_s": frame_offset / FPS, "card_duration_s": CARD_SECONDS,
                                   "resume_start_s": (frame_offset + CARD_FRAMES) / FPS,
                                   "completed_evaluated_actions": segment["start_step"],
                                   "same_observed_state_and_png_cameras_at_resume": True})
                frame_offset += CARD_FRAMES
            clip = staging / f"source_{number:02d}.mp4"
            shutil.copyfile(segment["video"], clip)
            pieces.append({"path": clip, "kind": "source", "segment": segment["label"],
                           "expected_frames": segment["media"]["frames"]})
            frame_offset += segment["media"]["frames"]
        concat = staging / "concat.txt"
        concat.write_text("".join(f"file '{piece['path'].name}'\n" for piece in pieces))
        combined = publish / "full_episode.mp4"
        log("Joining source videos and labeled resume cards without re-encoding source frames")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1",
             "-i", str(concat), "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(combined)])
        expected_hashes, decoded_pieces = [], []
        for piece in pieces:
            hashes = decoded_hashes(piece["path"])
            require(len(hashes) == piece["expected_frames"], f"Unexpected frame count in {piece['path']}")
            expected_hashes.extend(hashes)
            decoded_pieces.append({"kind": piece["kind"], "segment": piece.get("segment"), "frames": len(hashes),
                                   "pixel_hash_sequence_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest()})
        combined_hashes = decoded_hashes(combined)
        require(combined_hashes == expected_hashes, "Combined decoded pixels do not exactly preserve every source frame")
        media = probe(combined)
        require(media["frames"] == len(expected_hashes), "Combined frame totals disagree")
        events = shifted_events(data)
        position = 0
        for event in events:
            require(event["frame_start"] == position, "Combined event frame spans are not contiguous")
            position += event["frame_count"]
        require(position == media["frames"], "Combined events do not cover every frame")
        require(sum(e["event"] == "executed" for e in events) == data["total_actions"], "Reconstruction was incorrectly counted as new execution")
        event_file = publish / "full_episode.events.jsonl"
        event_file.write_text("".join(json.dumps(event, allow_nan=False) + "\n" for event in events))
        preview_index = events[-1]["frame_start"]
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-i", str(combined),
             "-vf", f"select=eq(n\\,{preview_index})", "-frames:v", "1", "-threads", "1", str(publish / "video_preview.png")])
        (publish / "index.html").write_text(gallery(data, media["duration_s"], boundaries))
        row = data["after"]["row"]
        completed = row["status"] == "success"
        final_label = data["after"]["label"]
        (publish / "RESULTS.md").write_text(
            "# One physical bowl episode with continuation\n\n"
            + ("**LIBERO success reached.**\n\n" if completed else f"**Incomplete: {row['status']}.** Task completion was not established.\n\n")
            + f"{data['total_actions']} evaluated actions comprise {data['offset']} original actions and {data['total_actions']-data['offset']} new actions across {len(data['segments'])-1} continuation recordings. "
            + "Historical reconstruction adds zero evaluated actions or model calls. No second episode is counted.\n\n"
            + f"Every recorded budget or infrastructure pause is retained; the approved final continuation cap is ${data['cap']:g} total across all recordings. "
            + "API waits and reconstruction footage are omitted. Five-second boundary cards are labeled editorial material.\n\n"
            + "[Full video](full_episode.mp4) · [Offline viewer](index.html) · [Combined events](full_episode.events.jsonl) · "
            + f"[Media validation](video_package_manifest.json) · [Exact reconstruction]({final_label}/replay_validation.json) · "
            + f"[Terminal result]({final_label}/summary.json)\n")
        checksums = {path.relative_to(publish).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                     for path in sorted(publish.rglob("*")) if path.is_file()}
        links_checked = check_local_html(publish / "index.html", publish)
        proof = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "kind": "single_episode_with_continuation_chain",
            "physical_episode_count": 1, "evaluation_completed": completed, "terminal_status": row["status"],
            "libero_success": True if completed else None, "incomplete_is_not_scored_as_failure": True,
            "original_actions": data["offset"], "new_continuation_actions": data["total_actions"] - data["offset"],
            "cumulative_evaluated_actions": data["total_actions"], "reconstruction_actions_counted_as_new": 0,
            "cumulative_recorded_field_refreshes": sum(segment["counts"].get("field_refresh", 0) for segment in data["segments"]),
            "cumulative_api_calls": row["budget_after"]["api_calls"], "approved_cumulative_cap_usd": data["cap"],
            "source_identity_hashes": {segment["label"]: segment["manifest"]["identity_hash"] for segment in data["segments"]},
            "media": media, "api_waits_omitted": True, "reconstruction_footage_omitted": True,
            "exact_source_frame_preservation": {
                "method": "full framemd5 decode of every source and editorial card equals full combined decode in exact order",
                "passed": True, "decoded_pieces": decoded_pieces, "combined_frames": len(combined_hashes),
                "combined_pixel_hash_sequence_sha256": hashlib.sha256("\n".join(combined_hashes).encode()).hexdigest(),
                "decoder_exit_codes": [0] * (len(pieces) + 1), "decoder_errors": [],
            },
            "boundaries": boundaries,
            "events": {"count": len(events), "contiguous_frames": True, "evaluated_execution_events": data["total_actions"],
                       "historical_terminal_events_are_intermediate_pauses": True,
                       "source_actions_states_selected_candidates_and_field_provenance_match_trace": True},
            "source_segments": [{"label": segment["label"], "start_step": segment["start_step"],
                                "evaluated_actions": segment["trace"]["actions"], "field_refreshes": segment["trace"]["fields"],
                                "terminal_status": segment["row"]["status"], "trace_sha256": segment["trace"]["trace_sha256"]}
                               for segment in data["segments"]],
            "preview": {"file": "video_preview.png", "frame_index_zero_based": preview_index, "time_s": preview_index / FPS},
            "gallery_links_checked": links_checked, "files": checksums,
        }
        (publish / "video_package_manifest.json").write_text(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n")
        archive_tmp = staging / archive.name
        with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(publish.rglob("*")):
                if path.is_file():
                    bundle.write(path, (Path(output.name) / path.relative_to(publish)).as_posix(),
                                 compress_type=zipfile.ZIP_STORED if path.suffix == ".mp4" else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(archive_tmp) as bundle:
            require(bundle.testzip() is None, "ZIP integrity check failed")
            for name, metadata in checksums.items():
                content = bundle.read((Path(output.name) / name).as_posix())
                require(len(content) == metadata["bytes"] and hashlib.sha256(content).hexdigest() == metadata["sha256"], f"ZIP checksum mismatch: {name}")
        publish.rename(output)
        archive_tmp.rename(archive)
    return {"output": str(output), "archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha(archive), "frames": media["frames"], "duration_s": media["duration_s"],
            "physical_episode_count": 1, "cumulative_actions": data["total_actions"], "terminal_status": row["status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--continuation", required=True, type=Path, action="append",
                        help="Continuation directories in chronological order; repeat for a recovery chain")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    parent = args.parent.resolve()
    children = [path.resolve() for path in args.continuation]
    try:
        data = collect(parent, children)
        if args.check_only:
            result = {"ready": True, "physical_episode_count": 1, "cumulative_actions": data["total_actions"],
                      "terminal_status": data["after"]["row"]["status"], "writes": 0}
        else:
            output = (args.output or children[-1].with_name(children[-1].name + "_full_episode")).resolve()
            result = package(data, output)
    except (NotReady, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
