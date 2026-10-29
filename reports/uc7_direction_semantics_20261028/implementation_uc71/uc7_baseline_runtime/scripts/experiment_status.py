#!/usr/bin/env python3
"""Read compact experiment status for a chat conversation; never control a run.

Uses only the standard library and immediate children of the runs directory.
Recorded "running" status is not evidence that its process is still alive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time


DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
MAX_JSON_BYTES = 16 * 1024 * 1024
ACTIVE_STATUSES = {
    "running", "starting", "preparing", "waiting", "review_required",
    "awaiting_user_approval", "waiting_for_approval", "ready",
}


def _number(value):
    return (value if type(value) in (int, float)
            and math.isfinite(value) and value >= 0 else None)


def _first(data, keys, predicate=lambda value: value is not None):
    for key in keys:
        if key in data and predicate(data[key]):
            return data[key]
    return None


def _utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _run_path(root, run_id):
    if (not isinstance(run_id, str) or not run_id or run_id in {".", ".."}
            or "/" in run_id or "\\" in run_id):
        raise ValueError("Run IDs must name one immediate child of the runs directory.")
    path = root / run_id
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Run directory is unavailable: {run_id}") from exc
    if resolved.parent != root or not resolved.is_dir():
        raise ValueError(f"Run directory is outside the runs directory: {run_id}")
    return resolved


def _contained_file(path, run):
    try:
        return path.resolve(strict=True).is_relative_to(run) and path.is_file()
    except (OSError, RuntimeError):
        return False


def _record(path, run):
    result = {"name": path.name, "mtime_ns": None, "data": None, "warning": None}
    if not path.exists() and not path.is_symlink():
        return result
    if not _contained_file(path, run):
        result["warning"] = f"{path.name}: refused a file outside this run."
        return result
    try:
        stat = path.stat()
        result["mtime_ns"] = stat.st_mtime_ns
        if stat.st_size > MAX_JSON_BYTES:
            raise ValueError("too large")
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("too large")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        result["data"] = data
    except (OSError, ValueError, UnicodeError, RecursionError):
        result["warning"] = f"{path.name}: unavailable, invalid, or being written."
    return result


def _steps(data):
    return _first(data, ("steps", "step"), lambda value: type(value) is int and value >= 0)


def _requests(data):
    """Request counters include failed calls; model_calls may count decisions."""
    predicate = lambda value: type(value) is int and value >= 0
    count = _first(data, ("api_calls", "subscription_inference_calls"), predicate)
    if count is not None:
        return count
    for field in ("budget", "execution_budget"):
        budget = data.get(field)
        if isinstance(budget, dict):
            count = _first(budget, ("api_calls", "subscription_inference_calls"), predicate)
            if count is not None:
                return count
    return None


def _consistent_summary(summary, state, *, summary_is_newer):
    """A newer summary can finish an older active state, but cannot regress it."""
    for key in ("task_key", "task_id", "episode_index", "method"):
        if key in summary and key in state and summary[key] != state[key]:
            return False
    ss, rs = _steps(summary), _steps(state)
    if ss is not None and rs is not None:
        if ss < rs or (not summary_is_newer and ss != rs):
            return False
    a, b = summary.get("status"), state.get("status")
    if a is not None and b is not None and a != b:
        return summary_is_newer and b in ACTIVE_STATUSES
    return True


def summarize_run(root, run_id, *, now=None):
    root = Path(root).resolve(strict=True)
    run = _run_path(root, run_id)
    records = [_record(run / name, run) for name in ("run_state.json", "summary.json")]
    warnings = [item["warning"] for item in records if item["warning"]]
    present = [item for item in records if item["mtime_ns"] is not None]
    # State wins equal mtimes: a terminal summary must actually be newer to
    # replace a contradictory state.
    present.sort(key=lambda item: (item["mtime_ns"], item["name"] == "run_state.json"), reverse=True)
    latest = present[0] if present else None
    data = dict(latest["data"]) if latest and latest["data"] is not None else {}
    source = latest["name"] if latest and latest["data"] is not None else None
    summary_record = records[1]
    state_record = records[0]
    summary, state = summary_record["data"], state_record["data"]
    conflict = False
    if summary is not None and state is not None:
        newer = summary_record["mtime_ns"] > state_record["mtime_ns"]
        compatible = _consistent_summary(summary, state, summary_is_newer=newer)
        if source == "summary.json" and not compatible:
            conflict = True
            warnings.append("Summary conflicts with run state; current outcome is unverified.")
        elif source == "run_state.json" and compatible:
            data = {**summary, **state}
        elif source == "run_state.json" and not compatible:
            warnings.append("Older summary does not match the latest run state; ignored its outcome and timing.")
    now = time.time() if now is None else now
    stamp = latest["mtime_ns"] / 1e9 if latest else None
    status = data.get("status") if isinstance(data.get("status"), str) else "unknown"
    if conflict:
        status = "conflicting_records"
    if not data:
        warnings.append("No readable current status record.")
    result = {
        "run_id": run_id,
        "status": status,
        "status_source": source,
        "task": _first(data, ("task_key", "task", "task_id"),
                       lambda value: type(value) in (str, int)),
        "stage": data.get("stage") if isinstance(data.get("stage"), str) else None,
        "steps": _steps(data),
        "model_calls": _requests(latest["data"]) if latest and latest["data"] is not None else None,
        "raw_libero_success": _first(data, ("libero_success",), lambda value: type(value) is bool),
        "raw_robocasa_success": _first(data, ("robocasa_success",), lambda value: type(value) is bool),
        "completion_time_eligible": _first(data, ("completion_time_eligible",), lambda value: type(value) is bool),
        "control_without_video_s": _number(data.get("control_without_video_s")),
        "timing_kind": None,
        "updated_at_utc": _utc(stamp) if stamp is not None else None,
        "age_seconds": round(max(0, now - stamp), 1) if stamp is not None else None,
        "status_note": ("Recorded status; process liveness is unverified."
                        if status in ACTIVE_STATUSES else "Recorded status only."),
        "videos": [],
        "warnings": warnings,
    }
    if result["model_calls"] is None:
        result["model_calls"] = _requests(data)
    if result["model_calls"] is None:
        result["model_calls"] = _first(data, ("model_calls",),
            lambda value: type(value) is int and value >= 0)
    if conflict:
        result["raw_libero_success"] = None
        result["raw_robocasa_success"] = None
        result["completion_time_eligible"] = None
        result["control_without_video_s"] = None
    if result["control_without_video_s"] is not None:
        raw_success = (result["raw_libero_success"] is True
                       or result["raw_robocasa_success"] is True)
        result["timing_kind"] = ("completion" if status == "success" and raw_success
                                 and result["completion_time_eligible"] is True else "attempt")
    if latest and latest["data"] is None:
        readable = next((item for item in present if item["data"] is not None), None)
        if readable:
            result["last_readable_status"] = readable["data"].get("status")
            result["last_readable_source"] = readable["name"]
    for path in sorted(run.glob("episode*.mp4")):
        if path.name.endswith(".partial.mp4") or not _contained_file(path, run):
            continue
        try:
            if path.stat().st_size > 0:
                result["videos"].append(str(path))
        except OSError:
            continue
    return result


def latest_run_ids(root, limit):
    root = Path(root).resolve(strict=True)
    candidates = []
    for child in root.iterdir():
        try:
            run = _run_path(root, child.name)
        except (ValueError, OSError, RuntimeError):
            continue
        stamps = []
        for name in ("run_state.json", "summary.json"):
            path = run / name
            if _contained_file(path, run):
                try:
                    stamps.append(path.stat().st_mtime_ns)
                except OSError:
                    pass
        if stamps:
            candidates.append((max(stamps), child.name))
    return [name for _, name in sorted(candidates, reverse=True)[:limit]]


def _display(value):
    if value is None:
        return "unknown"
    if type(value) is bool:
        return str(value).lower()
    return str(value)


def format_run(run):
    text = (f"{run['run_id']}: {run['status']} | task={_display(run['task'])} | "
            f"steps={_display(run['steps'])} | model calls={_display(run['model_calls'])}")
    text += (f"\n  Raw LIBERO success={_display(run['raw_libero_success'])}; "
             f"completion eligible={_display(run['completion_time_eligible'])}.")
    if run["raw_robocasa_success"] is not None:
        text += f" Raw RoboCasa success={_display(run['raw_robocasa_success'])}."
    if run["control_without_video_s"] is not None:
        text += f" {run['timing_kind'].capitalize()} time excluding video={run['control_without_video_s']:.2f}s."
    text += (f"\n  Updated {_display(run['updated_at_utc'])}; "
             f"age={_display(run['age_seconds'])}s. {run['status_note']}")
    for video in run["videos"]:
        text += f"\n  Video: {video}"
    for warning in run["warnings"]:
        text += f"\n  Note: {warning}"
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="*", help="Immediate run-directory names; defaults to latest runs")
    parser.add_argument("--limit", type=int, default=5, help="Number of latest runs (default: 5)")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--json", action="store_true", help="Print compact structured records")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    try:
        root = args.runs_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("--runs-root must be a directory")
        ids = args.run_ids or latest_run_ids(root, args.limit)
        now = time.time()
        records = [summarize_run(root, run_id, now=now) for run_id in ids]
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps({"schema_version": 1, "generated_at_utc": _utc(now), "runs": records},
                         indent=2, allow_nan=False))
    else:
        print("\n\n".join(format_run(record) for record in records) or "No recorded runs found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
