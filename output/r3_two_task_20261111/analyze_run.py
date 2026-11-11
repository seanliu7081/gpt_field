#!/usr/bin/env python3
"""Summarize a completed archived FVH V3 run; no policy or simulator imports.

Use --dry-run to inspect an existing run without changing its artifacts.
The default writes analysis.json and decodes the two saved terminal images.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_BASELINE = REPO / "runs/foundation_vision_harness_V3_t04_can_obstacle_003"


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run, source_baseline):
    trial = run / "trial"
    result = read_json(run / "result.json")
    if result is None:
        raise SystemExit(f"No terminal result.json: {run}")
    stats = result.get("agent_stats", {})
    timings = result.get("performance", {}).get("timings", {})
    online = result.get("online_wall_s")
    overhead = sum(timings.get(k, {}).get("total_s", 0.)
                   for k in ("recording_submit", "status_write", "trace_write"))
    events = read_jsonl(trial / "foundation_vision_harness_V3/events.jsonl")
    actions = [e for e in read_jsonl(trial / "trace.jsonl") if e.get("event") == "action"]
    plans = [e for e in events if e.get("event") == "model_plan"]
    routes, interruptions = [], []
    for event in events:
        if event.get("event") not in ("segment_result", "candidate_failed"):
            continue
        outcome = event.get("outcome", {})
        evidence = outcome.get("evidence", {})
        common = {
            "step": event.get("step"), "event": event["event"],
            "reason": outcome.get("reason"),
            "executed_steps": evidence.get("executed_steps", outcome.get("actions")),
        }
        if outcome.get("status") != "succeeded":
            interruptions.append({**common, "position_error_m": evidence.get("position_error_m"),
                                  "orientation_error_rad": evidence.get("orientation_error_rad")})
        for segment in evidence.get("segments", []):
            route = segment.get("routing")
            if route:
                routes.append({**common, "budget_steps": evidence.get("budget_steps"),
                               "routing": route})

    evaluator = read_json(trial / "terminal_evaluator_status.json", {})
    collision = evaluator.get("collision", {})
    increments, previous = [], {}
    for action in actions:
        counts = action.get("evaluator", {}).get("collision", {}).get("counts", previous)
        delta = counts.get("forbidden_contact_points", 0) - previous.get("forbidden_contact_points", 0)
        if delta > 0:
            increments.append({"step": action["step"], "new_points": delta,
                "by_class": {name: counts.get("points_" + name, 0) - previous.get("points_" + name, 0)
                             for name in collision.get("forbidden_classes", [])
                             if counts.get("points_" + name, 0) > previous.get("points_" + name, 0)}})
        previous = counts

    manifest = read_json(run / "source_manifest.json", {})
    reference = read_json(source_baseline / "source_manifest.json", {})
    names = {name for name in manifest.keys() | reference.keys() if not name.startswith("@")}
    differences = sorted(name for name in names if manifest.get(name) != reference.get(name))
    model_s = stats.get("timing", {}).get("model_s", 0.)
    requests = stats.get("model_requests", [])
    latencies = [x["latency_s"] for x in requests if isinstance(x.get("latency_s"), (int, float))]
    truth = read_json(trial / "truth_terminal.json", {})
    hold = read_json(trial / "truth_after_same_gripper_hold.json", {})
    config = read_json(run / "configuration.json", {})
    analysis = {
        "version": result.get("version"), "revision": stats.get("revision", config.get("revision")),
        "run": str(run), "task": result.get("task"), "status": result.get("status"),
        "steps": result.get("steps"), "online_wall_s": online,
        "measured_video_log_overhead_s": overhead,
        "online_minus_measured_video_log_s": None if online is None else online - overhead,
        "timing_note": "Only measured main-thread recording_submit/status_write/trace_write subtracted; not asynchronous encoding.",
        "completion_time_eligible": result.get("completion_time_eligible"),
        "completion_wall_s": result.get("completion_wall_s"),
        "first_raw_success": result.get("first_raw_success"),
        "raw_success_at_terminal": result.get("raw_success_at_terminal"),
        "raw_success_after_same_gripper_hold": result.get("raw_success_after_same_gripper_hold"),
        "strict_physical_at_terminal": result.get("strict_success_at_terminal"),
        "strict_physical_after_same_gripper_hold": result.get("strict_success_after_same_gripper_hold"),
        "combined_success": result.get("strict_success"),
        "withheld_decisions_after_stop": result.get("withheld_decisions_after_stop", 0),
        "operator_stop": read_json(trial / "operator_stop.request.json"),
        "model_decisions": stats.get("model_decisions"),
        "model_plan_types": dict(Counter(e["answer"]["status"] for e in plans)),
        "timing": stats.get("timing", {}), "performance_timings": timings,
        "model_share_online": model_s / online if online else None,
        "model_latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "model_latency_max_s": max(latencies) if latencies else None,
        "grasp_attempts": stats.get("grasp_attempts"), "completed_segments": stats.get("completed_segments"),
        "interruptions": interruptions, "robot_guard": stats.get("robot_guard", {}),
        "vision_tools": stats.get("vision_tools", {}),
        "map_updates": sum(e["answer"].get("obstacle_map") is not None for e in plans),
        "routes": routes, "route_count": len(routes), "routes_are_finalized_candidates_only": True,
        "route_planning_s": sum(e["routing"].get("planning_s", 0.) for e in routes),
        "collision": collision, "contact_increment_steps": increments,
        "contact_note": "Counts are physics-substep contact samples, not independent collision events; terminal hold collision counts are not inferred.",
        "terminal_atoms": truth.get("atoms", []), "after_hold_atoms": hold.get("atoms", []),
        "code_baseline": str(source_baseline), "source_matches_r3_archive": bool(reference) and not differences,
        "source_differences": differences,
        "terminal_source_audit": read_json(run / "terminal_source_audit.json"),
        "video": result.get("video", {}), "cleanup_errors": result.get("cleanup_errors", []),
    }
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-baseline", type=Path, default=DEFAULT_SOURCE_BASELINE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-output", action="store_true", help="Replace existing analysis/images explicitly")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    analysis = summarize(run, args.source_baseline.expanduser().resolve())
    if not args.dry_run:
        destinations = [run / "analysis.json", run / "terminal_external.png", run / "terminal_wrist.png"]
        if not args.replace_output and any(path.exists() for path in destinations):
            raise SystemExit("Analysis artifacts already exist; use --dry-run or explicit --replace-output.")
        observation = read_json(run / "trial/terminal_observation.json", {})
        images = []
        for key, path in zip(("observation/image", "observation/wrist_image"), destinations[1:]):
            encoded = observation.get(key)
            if isinstance(encoded, str):
                images.append((path, base64.b64decode(encoded.split(",", 1)[-1], validate=True)))
        destinations[0].write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
        for path, image in images:
            path.write_bytes(image)
    compact = {k: analysis[k] for k in (
        "run", "revision", "status", "steps", "online_wall_s", "online_minus_measured_video_log_s",
        "raw_success_at_terminal", "strict_physical_at_terminal", "combined_success",
        "model_decisions", "model_plan_types", "model_share_online", "model_latency_max_s",
        "grasp_attempts", "route_count", "route_planning_s", "source_matches_r3_archive")}
    compact["interruptions"] = dict(Counter(x["reason"] for x in analysis["interruptions"]))
    compact["forbidden_contact_points"] = analysis["collision"].get("counts", {}).get("forbidden_contact_points")
    compact["dry_run"] = args.dry_run
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
