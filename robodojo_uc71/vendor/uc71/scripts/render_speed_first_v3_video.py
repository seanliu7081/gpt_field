#!/usr/bin/env python3
"""Re-render an archived v3 episode with colored field slices, entirely offline.

Reads saved RGB observations, decisions and feedback. It never calls policy.act,
constructs a simulator, reads credentials, or submits model requests. The source
trial and its timing/result artifacts are immutable; this is presentation only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.observation import PolicyObservation
from gpt_field.planner import PlannerConfig
from gpt_field.schema import GridSpec
from gpt_field.speed_first_v3_video import SpeedFirstV3VideoRecorder


class ReplayError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def rows(path):
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ReplayError(f"Invalid JSON at {path}:{number}") from exc


def probe(path):
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height,duration",
        "-of", "json", str(path),
    ], text=True)
    streams = json.loads(raw)["streams"]
    require(len(streams) == 1, "Expected one video stream")
    return streams[0]


def source_paths(trial):
    names = ("trace.jsonl", "summary.json", "manifest.json", "episode.mp4", "episode.events.jsonl")
    paths = {name: trial / name for name in names}
    for name in ("analysis.json", "independent_terminal_audit.json", "preflight.json"):
        path = trial.parent / name
        if path.exists():
            paths["../" + name] = path
    return paths


def load_archive(trial):
    summary = json.loads((trial / "summary.json").read_text())
    require(summary.get("speed_first_v3") is True, "Expected an archived Speed-first v3 episode")
    require(summary.get("status") not in (None, "running"), "Source episode is not terminal")
    events = list(rows(trial / "episode.events.jsonl"))
    require(events and events[0]["event"] == "observation", "Missing initial video event")
    require(events[-1]["event"] == "terminal", "Missing terminal video event")
    require(events[-1]["status"] == summary["status"], "Terminal result differs from summary")
    video = probe(trial / "episode.mp4")
    rate = Fraction(video["r_frame_rate"])
    require(rate.denominator == 1, "Only original integer playback rates are supported")
    frame = 0
    for event in events:
        require(event["frame_start"] == frame, "Original video event index is discontinuous")
        frame += event["frame_count"]
    require(frame == int(video["nb_frames"]), "Original video/event frame counts differ")
    executed = {event["step"]: event for event in events if event["event"] == "executed"}
    require(len(executed) == summary["steps"], "Original executed video count differs from summary")
    return summary, events, executed, video, int(rate)


def trace_items(trial, summary):
    """Bounded-memory iteration; pair each saved action with its exact outcome."""
    before = action = executed = None
    counts = Counter()
    for row in rows(trial / "trace.jsonl"):
        event = row["event"]
        if event == "reset":
            require(before is None and counts["reset"] == 0, "Unexpected repeated reset")
            before = row["observation"]
            counts[event] += 1
            yield ("reset", row)
        elif event == "action":
            require(before is not None and action is None and executed is None, "Incomplete prior transition")
            require(row["step"] == counts["actions"], "Nonsequential action index")
            require(row["state"] == before["observation/state"], "Action observation chain differs")
            require(row["diagnostics"]["step"] == row["step"], "Decision step differs from trace")
            action = row
        elif event == "executed":
            require(action is not None and executed is None, "Executed event has no unique action")
            require(row["step"] == action["step"] and row["action"] == action["action"], "Executed action differs")
            require(row["next_state"] == row["observation"]["observation/state"], "Executed state differs")
            executed = row
        elif event == "feedback":
            require(action is not None and executed is not None, "Feedback has no complete transition")
            feedback = row["feedback"]
            require(row["step"] == action["step"] == feedback["step"], "Feedback step differs")
            require(feedback["before_state"] == action["state"], "Feedback starting state differs")
            require(feedback["after_state"] == executed["next_state"], "Feedback resulting state differs")
            require(feedback["normalized_action"] == action["action"], "Feedback action differs")
            yield ("transition", (action, executed, feedback))
            before = executed["observation"]
            counts["actions"] += 1
            counts["model_replies"] += bool(action["diagnostics"].get("refreshed"))
            action = executed = None
    require(action is None and executed is None, "Source archive has an incomplete final transition")
    require(counts["reset"] == 1 and counts["actions"] == summary["steps"], "Archived action total differs")
    require(counts["model_replies"] == summary["validated_hybrid_refreshes"], "Archived reply total differs")


class ArchivedTimelineRecorder(SpeedFirstV3VideoRecorder):
    """Retain original event cadence/latencies, with explicit replay wall time."""

    def __init__(self, *args, original_events, **kwargs):
        self.original_events = original_events
        self.original_event_index = 0
        super().__init__(*args, **kwargs)

    def _emit(self, event, repeats, **extra):
        require(self.original_event_index < len(self.original_events), "Replay generated extra events")
        original = self.original_events[self.original_event_index]
        require((event, self.step, self.frame_count, repeats) == (
            original["event"], original["step"], original["frame_start"], original["frame_count"]),
            f"Replay event cadence differs at event {self.original_event_index}")
        require(self.decision_latency_s == original.get("decision_latency_s"), "Original decision latency changed")
        super()._emit(event, repeats,
            archived_episode_replay=True,
            original_wall_elapsed_s=original.get("wall_elapsed_s"),
            wall_elapsed_scope="offline_rendering_only_not_task_control_time",
            live_model_calls_during_render=0,
            simulator_actions_during_render=0,
            **extra)
        self.original_event_index += 1


def compare_events(before, after):
    require(len(before) == len(after), "Replay event total changed")
    # Rendering fields may change; recorded actions, feedback, geometry and time
    # must not. Compare every original field except the explicit display fields.
    display_keys = {"field_display_source", "dense_distance_map_rendered", "wall_elapsed_s"}
    for index, (original, rendered) in enumerate(zip(before, after)):
        for key, value in original.items():
            if key not in display_keys:
                require(rendered.get(key) == value, f"Original event metadata changed: event {index}, key {key}")
        require(rendered["original_wall_elapsed_s"] == original["wall_elapsed_s"], "Original wall time not retained")


def render(trial, output, check_only=False):
    started = time.monotonic()
    sources = source_paths(trial)
    before_hashes = {name: file_record(path) for name, path in sources.items()}
    summary, events, executed_events, source_video, fps = load_archive(trial)
    manifest = json.loads((trial / "manifest.json").read_text())
    planner_parameters = manifest["configuration"]["policy"]["planner"]
    report = {
        "kind": "offline_saved_episode_colored_field_video",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_trial": str(trial), "source_files": before_hashes,
        "source_video_metadata": source_video,
        "recorded_planner_parameters": planner_parameters,
        "archived_actions_displayed": summary["steps"],
        "archived_model_requests": summary["api_calls"],
        "archived_validated_model_replies": summary["validated_hybrid_refreshes"],
        "archived_terminal_status": summary["status"],
        "archived_libero_success": summary["libero_success"],
        "archived_control_elapsed_s": summary["control_elapsed_s"],
        "archived_video_render_s": summary["timing_components"]["video_render_s"],
        "archived_control_without_video_s": summary["control_without_video_s"],
        "live_inference_calls": 0, "api_key_calls": 0,
        "simulator_starts": 0, "simulator_actions": 0,
        "policy_act_calls": 0, "credential_access": False,
        "new_task_result": False, "original_task_timing_modified": False,
        "interpretation": "Presentation-only reconstruction from recorded observations, actions and compact fields. Rendering time is not added to or subtracted from the original task time. No new robot episode or model inference occurred.",
    }
    if check_only:
        for _ in trace_items(trial, summary):
            pass
        report.update(status="validated_without_rendering", original_event_count=len(events),
                      original_frame_count=int(source_video["nb_frames"]), fps=fps)
        return report
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "episode_color.mp4"
    outputs = [video_path, video_path.with_suffix(".partial.mp4"), video_path.with_suffix(".events.jsonl"),
               output / "provenance.json", output / "preview_field.png"]
    require(not any(path.exists() for path in outputs), "Refusing to overwrite an existing render artifact")
    renderer_paths = [ROOT / "src/gpt_field" / name for name in (
        "speed_first_v3_video.py", "hybrid_video.py", "field_history_video.py", "video.py")]
    renderer_paths.append(Path(__file__).resolve())
    renderer_hashes = {str(path.relative_to(ROOT)): file_record(path) for path in renderer_paths}
    recorder = ArchivedTimelineRecorder(video_path, grid=GridSpec(**events[0]["grid"]),
        episode_id=f"OFFLINE RE-RENDER | {trial.parent.name}", fps=fps, original_events=events,
        planner_config=PlannerConfig(**planner_parameters))
    before = None
    count = 0
    try:
        for kind, item in trace_items(trial, summary):
            if kind == "reset":
                before = PolicyObservation.from_dict(item["observation"])
                recorder.observe(before, step=0, history_stats=events[0].get("history_stats"))
                continue
            action, executed, feedback = item
            decision = SimpleNamespace(action=tuple(action["action"]), diagnostics=action["diagnostics"])
            recorder.decision(before, decision, decision_latency_s=action.get("decision_latency_s"))
            if decision.diagnostics["controller"] == "coarse_field" and not (output / "preview_field.png").exists():
                recorder.render_frame().save(output / "preview_field.png")
                report["preview_field_action_index"] = action["step"]
            after = PolicyObservation.from_dict(executed["observation"])
            archived_event = executed_events[action["step"]]
            require(archived_event["feedback"] == feedback, "Trace/video feedback differs")
            recorder.executed(after, step=executed["step"], success=executed["success"],
                sim_status=executed["sim_status"], feedback=feedback,
                history_stats=archived_event.get("history_stats"))
            before = after
            count += 1
            if count % 100 == 0:
                print(json.dumps({"event": "offline_render_progress", "archived_actions_rendered": count,
                                  "total_archived_actions": summary["steps"]}), flush=True)
        error = summary.get("error") or summary.get("cleanup_errors")
        recorder.close(status=summary["status"], error=json.dumps(error) if error else None)
        require(recorder.original_event_index == len(events), "Replay omitted original video events")
        colored_events = list(rows(video_path.with_suffix(".events.jsonl")))
        compare_events(events, colored_events)
        rendered_video = probe(video_path)
        require(rendered_video == source_video, "Encoded playback duration/rate/dimensions/frame count changed")
        require(before_hashes == {name: file_record(path) for name, path in sources.items()}, "Source artifact changed")
        require(renderer_hashes == {str(path.relative_to(ROOT)): file_record(path) for path in renderer_paths}, "Renderer changed during processing")
        report.update(status="passed", output_video=str(video_path), output_video_metadata=rendered_video,
            rendered_event_count=len(colored_events), rendered_frame_count=recorder.frame_count,
            event_sequence_and_original_metadata_preserved=True, original_sources_unchanged=True,
            renderer_sources=renderer_hashes, completed_archived_actions=count,
            field_display_sources=dict(Counter(event.get("field_display_source") for event in colored_events)),
            output_files={path.name: file_record(path) for path in outputs if path.exists()},
            offline_render_wall_s_not_task_latency=time.monotonic() - started)
    except BaseException:
        # Finalize any partial stream without fabricating a successful terminal
        # replay; retain it for inspection and never replace archived artifacts.
        if not recorder.closed:
            recorder.closed = True
            recorder._events.close()
            recorder.writer.close()
        raise
    with (output / "provenance.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to a sibling video_color directory")
    parser.add_argument("--check-only", action="store_true", help="Validate saved inputs without rendering or writing")
    args = parser.parse_args()
    trial = args.trial.resolve()
    output = (args.output or trial.parent / "video_color").resolve()
    if output == trial or trial in output.parents:
        parser.error("Output must be outside the immutable source trial")

    def deny_network(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise ReplayError("Network access is prohibited during offline rendering")

    sys.addaudithook(deny_network)
    try:
        result = render(trial, output, check_only=args.check_only)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "archived_actions_displayed", "archived_model_requests", "archived_terminal_status",
        "live_inference_calls", "simulator_actions")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
