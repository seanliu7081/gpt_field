"""Exact archived MP4 frame -> event -> action/state mapping, with validation.

Only the current run's frozen recorder sources and archived events/observations
are used. No simulator or policy is imported. Old user-deleted artifacts are not
restored. Intervals are start-inclusive/end-exclusive; time is source video time,
not control wall time. Execute this file to regenerate its three JSON artifacts.
"""
from __future__ import annotations

import base64
from bisect import bisect_right
from collections import Counter
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess


RUN = Path("/workspace/gpt_field/runs/foundation_vision_t08_bottle_obstacle_20261105_001")
OUTPUT = Path(__file__).resolve().parent
FPS = 20
CROPS_XYXY = {"external": (44, 181, 380, 517), "wrist": (44, 577, 380, 913)}
COLUMNS = ["frame_start", "frame_count", "event_line", "event", "action_index", "state_index"]
EXPECTED_POLICY_ACTIONS = 590
EXPECTED_VIDEO_FRAMES = 2542
FROZEN_SOURCE_REFS = [
    ("video.py", 186, "observe: initial 0.6 second repeat"),
    ("agent_speedup_v1_video.py", 213, "decision: refreshed 0.6 second pause; otherwise two frames"),
    ("video.py", 227, "executed: updated observation repeated twice"),
    ("video.py", 235, "close: three seconds of the last recorder observation"),
    ("agent_speedup_v2_video.py", 36, "external/wrist camera panel origins"),
    ("agent_speedup_v2_video.py", 193, "upright camera conversion and resize"),
    ("agent_speedup_v2_video.py", 208, "camera crop placement"),
    ("agent_speedup_v2_user_check_video.py", 284, "planning overlays preserve inherited event timing"),
    ("evaluate_agent_speedup_v2.py", 1210, "diagnostic release/settle does not update recorder observation"),
    ("evaluate_agent_speedup_v2.py", 1657, "diagnostics occur after policy termination before recorder close"),
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_events(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def read_blocks(path=RUN / "trial/episode.events.jsonl"):
    """Read the recorder's actual blocks; never infer a linear action/time ratio."""
    blocks, cursor, executed_count = [], 0, 0
    for line_number, event in enumerate(_read_events(path), 1):
        kind, step = event["event"], event["step"]
        start, count = event["frame_start"], event["frame_count"]
        if type(start) is not int or type(count) is not int or count <= 0 or start != cursor:
            raise ValueError(f"Noncontiguous/invalid frame interval at event line {line_number}")
        if not math.isclose(event["video_time_s"], start / FPS, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"Video timestamp mismatch at event line {line_number}")
        if kind == "executed":
            if step != executed_count:
                raise ValueError("Executed actions must be contiguous and zero-indexed")
            action_index, state_index = step, step + 1
            executed_count += 1
        elif kind in ("skill_decision", "agent_action"):
            if step != executed_count:
                raise ValueError("Pre-action observation is not at the next action boundary")
            action_index, state_index = step, step
        elif kind == "observation":
            if blocks or step != 0:
                raise ValueError("Expected exactly one initial observation")
            action_index, state_index = None, 0
        elif kind == "terminal":
            # The recorder retains last action's step label, not a new action.
            action_index, state_index = None, executed_count
        else:
            raise ValueError("Unexpected recorded event: " + str(kind))
        blocks.append(dict(zip(COLUMNS, (start, count, line_number, kind, action_index, state_index))))
        cursor += count
    if not blocks or blocks[-1]["event"] != "terminal":
        raise ValueError("Expected a finalized episode with terminal video block")
    return blocks


def frame_record(frame_index, blocks=None):
    if type(frame_index) is not int or frame_index < 0:
        raise IndexError(frame_index)
    blocks = read_blocks() if blocks is None else blocks
    index = bisect_right([block["frame_start"] for block in blocks], frame_index) - 1
    if index < 0:
        raise IndexError(frame_index)
    block = blocks[index]
    if frame_index >= block["frame_start"] + block["frame_count"]:
        raise IndexError(frame_index)
    return block.copy()


def time_record(time_s, blocks=None):
    if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < 0:
        raise ValueError("time_s must be a finite nonnegative source-video timestamp")
    return frame_record(math.floor(time_s * FPS + 1e-9), blocks)


def _video_metadata(path):
    result = subprocess.run([
        "/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    return {"frame_count": int(stream["nb_read_frames"]), "container_frame_count": int(stream["nb_frames"]),
            "fps": numerator / denominator, "width": int(stream["width"]),
            "height": int(stream["height"]), "duration_s": float(stream["duration"]),
            "decoder": "ffprobe -count_frames", "decode_stderr": result.stderr}


def _validate(blocks, events, video_metadata):
    import cv2
    import numpy as np
    from PIL import Image

    cv2.setNumThreads(1)
    initial = json.loads((RUN / "trial/initial_observation.json").read_text())
    sample_states = {0, 1, 277, 278, 589, 590}
    sample_states.update(block["state_index"] for block in blocks if block["event"] == "skill_decision")
    sample_states.update(block["state_index"] + 1 for block in blocks if block["event"] == "skill_decision")
    states = {0: initial["observation/state"]}
    images = {0: initial}
    trace_executed = []
    for line in (RUN / "trial/trace.jsonl").open():
        row = json.loads(line)
        if row.get("event") != "executed":
            continue
        state_index = row["step"] + 1
        states[state_index] = row["next_state"]
        trace_executed.append(row["step"])
        if state_index in sample_states:
            images[state_index] = row["observation"]
    mismatches, maximum_error = [], 0.0
    for block, event in zip(blocks, events):
        error = float(np.max(np.abs(np.asarray(event["state"]) - np.asarray(states[block["state_index"]]))))
        maximum_error = max(maximum_error, error)
        if error > 1e-12:
            mismatches.append({"event_line": block["event_line"], "state_index": block["state_index"], "max_abs_error": error})
    boundary_checks = []
    for index, block in enumerate(blocks):
        start, end = block["frame_start"], block["frame_start"] + block["frame_count"]
        ok = (frame_record(start, blocks) == block and frame_record(end - 1, blocks) == block
              and time_record(start / FPS, blocks) == block)
        if index + 1 < len(blocks):
            ok = ok and frame_record(end, blocks) == blocks[index + 1]
        boundary_checks.append(ok)
    out_of_range_rejected = []
    for value in (-1, EXPECTED_VIDEO_FRAMES):
        try:
            frame_record(value, blocks)
        except IndexError:
            out_of_range_rejected.append(True)
        else:
            out_of_range_rejected.append(False)

    frame_by_state = {0: 0, **{block["state_index"]: block["frame_start"]
                             for block in blocks if block["event"] == "executed"}}
    requested_samples = [(state, frame_by_state[state], "control_state") for state in sorted(sample_states)]
    requested_samples.extend((block["state_index"], block["frame_start"], "decision_pause")
                             for block in blocks if block["event"] == "skill_decision")
    requested_samples.extend([(590, blocks[-1]["frame_start"], "terminal_start"),
                              (590, EXPECTED_VIDEO_FRAMES - 1, "terminal_last_frame")])
    capture = cv2.VideoCapture(str(RUN / "trial/episode.mp4"))
    checks = []
    try:
        for state_index, frame_index, purpose in requested_samples:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise RuntimeError(f"Cannot seek source frame {frame_index}")
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode source frame {frame_index}")
            next_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            shown = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            comparisons = {}
            for camera, box in CROPS_XYXY.items():
                key = "observation/image" if camera == "external" else "observation/wrist_image"
                source = Image.open(BytesIO(base64.b64decode(images[state_index][key].split(",", 1)[1]))).convert("RGB")
                expected = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((336, 336), Image.Resampling.LANCZOS)
                difference = np.abs(np.asarray(shown.crop(box)).astype(np.float32)
                                    - np.asarray(expected).astype(np.float32))
                comparisons[camera] = {"mae_rgb_0_255": float(difference.mean()),
                                       "p95_absolute_error": float(np.quantile(difference, 0.95)),
                                       "max_absolute_error": float(difference.max())}
            checks.append({"state_index": state_index, "source_frame_index": frame_index,
                           "source_time_s": frame_index / FPS, "purpose": purpose,
                           "decoder_next_frame": next_frame, "camera_comparison": comparisons})
    finally:
        capture.release()
    truth = json.loads((RUN / "trial/physical_truth.json").read_text())
    diagnostic = truth["after_release_settle"]["settle"]
    predicates = {
        "frame_intervals_cover_decoded_video_exactly": sum(b["frame_count"] for b in blocks) == video_metadata["frame_count"] == EXPECTED_VIDEO_FRAMES,
        "fps_20": video_metadata["fps"] == FPS,
        "resolution_1600x960": (video_metadata["width"], video_metadata["height"]) == (1600, 960),
        "decoded_frame_count_matches_container": video_metadata["frame_count"] == video_metadata["container_frame_count"],
        "all_frame_and_time_boundaries_resolve": all(boundary_checks),
        "outside_video_indices_rejected": all(out_of_range_rejected),
        "all_event_states_equal_trace_states": not mismatches,
        "policy_trace_has_exact_actions_0_through_589": trace_executed == list(range(EXPECTED_POLICY_ACTIONS)),
        "state_indices_cover_0_through_590": sorted(states) == list(range(EXPECTED_POLICY_ACTIONS + 1)),
        "terminal_video_repeats_policy_state_590": blocks[-1]["state_index"] == 590 and events[-1]["state"] == states[590],
        "38_diagnostic_actions_are_not_video_actions": diagnostic["actions"] == 38 and truth["terminal"]["steps"] == 590
            and sum(b["event"] == "executed" for b in blocks) == 590,
        "sample_decode_indices_exact": all(s["decoder_next_frame"] == s["source_frame_index"] + 1 for s in checks),
        "camera_crops_match_saved_observations_with_lossy_encoding": all(
            c["mae_rgb_0_255"] < 6 for s in checks for c in s["camera_comparison"].values()),
    }
    return {"passed": all(predicates.values()), "checks": predicates, "video": video_metadata,
            "event_blocks_checked": len(blocks), "state_count": len(states),
            "max_state_absolute_error": maximum_error, "state_mismatches": mismatches,
            "representative_image_checks": checks,
            "image_comparison_note": "Decoded lossy MP4 versus saved 512x512 observation PNG, flipped left-right and Lanczos resized to 336x336. Six RGB levels is an acceptance threshold, not identity of encoded pixels.",
            "diagnostic_actions_excluded": diagnostic["actions"],
            "diagnostic_protocol": diagnostic["protocol"],
            "no_simulator_or_model_calls": True}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events = _read_events(RUN / "trial/episode.events.jsonl")
    blocks = read_blocks()
    video = _video_metadata(RUN / "trial/episode.mp4")
    event_counts = Counter(b["event"] for b in blocks)
    frame_counts = Counter()
    for block in blocks:
        frame_counts[block["event"]] += block["frame_count"]
    assert event_counts["executed"] == EXPECTED_POLICY_ACTIONS
    validation = _validate(blocks, events, video)
    source_paths = [RUN / "trial" / name for name in (
        "episode.mp4", "episode.events.jsonl", "trace.jsonl", "initial_observation.json", "physical_truth.json", "summary.json")]
    source_paths += [RUN / "trial/frozen_source" / name for name in dict.fromkeys(r[0] for r in FROZEN_SOURCE_REFS)]
    source_hashes = {str(path): _sha256(path) for path in source_paths}
    metadata = {
        "source_events": str(RUN / "trial/episode.events.jsonl"),
        "source_video": str(RUN / "trial/episode.mp4"), "fps": FPS,
        "frame_count": video["frame_count"], "duration_s": video["duration_s"],
        "event_counts": dict(event_counts), "frames_by_event": dict(frame_counts),
        "policy_actions": event_counts["executed"], "contiguous_coverage": True,
        "interval_convention": "frame_start inclusive, frame_start+frame_count exclusive; frame at source video time t is floor(t*20)",
        "state_index_convention": "State 0 is initial settled observation; action t transitions state t to t+1. Pre-action decision shows state t; executed t shows state t+1. Terminal shows state 590 while event.step remains 589.",
        "api_waits": "Actual model and vision waits are omitted from the MP4 timeline. Each refreshed decision is exactly 12 frames / 0.6 s, independently of wall latency.",
        "per_action_display": "Each policy action has a pre-action block (2 frames, or 12 for refreshed decision), followed by 2 frames of its post-action observation.",
        "setup": "Initial 12 frames are the already-settled reset observation. Initial simulator settling is not in the MP4.",
        "terminal": "Frames 2482..2541 repeat policy-terminal state 590. The later 38 diagnostic release/lift/settle actions are absent; terminal rendering uses the recorder's retained policy observation.",
        "diagnostic_actions_in_video": 0, "diagnostic_actions_excluded": 38,
        "camera_crops_xyxy_exclusive": CROPS_XYXY,
        "camera_crops_ffmpeg_whxy": {"external": [336, 336, 44, 181], "wrist": [336, 336, 44, 577]},
        "camera_input": "512x512 observation PNG is flipped left-right to upright, Lanczos resized to 336x336, then encoded in lossy H.264.",
        "columns": COLUMNS, "blocks": [[b[k] for k in COLUMNS] for b in blocks],
        "frozen_source_refs": [{"path": str(RUN / "trial/frozen_source" / name), "line": line, "meaning": meaning}
                               for name, line, meaning in FROZEN_SOURCE_REFS],
        "source_sha256": source_hashes, "validation_passed": validation["passed"],
    }
    frames = [{"state_index": 0, "source_frame_index": 0, "source_time_s": 0.,
               "source_event": "observation", "action_index": None}]
    frames += [{"state_index": b["state_index"], "source_frame_index": b["frame_start"],
                "source_time_s": b["frame_start"] / FPS, "source_event": "executed",
                "action_index": b["action_index"]} for b in blocks if b["event"] == "executed"]
    assert [row["state_index"] for row in frames] == list(range(EXPECTED_POLICY_ACTIONS + 1))
    control = {"source_video": metadata["source_video"], "source_fps": FPS,
               "control_state_count": len(frames), "state_convention": metadata["state_index_convention"],
               "diagnostic_actions_excluded": 38, "camera_crops_xyxy_exclusive": CROPS_XYXY,
               "control_state_frames": frames}
    validation["source_sha256"] = source_hashes
    for filename, data in (("frame_mapping.json", metadata), ("control_state_frames.json", control),
                           ("frame_mapping_validation.json", validation)):
        (OUTPUT / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": validation["passed"], "frames": video["frame_count"],
                      "event_counts": dict(event_counts), "frames_by_event": dict(frame_counts),
                      "checked_states": validation["state_count"], "image_samples": len(validation["representative_image_checks"]),
                      "max_state_absolute_error": validation["max_state_absolute_error"],
                      "failed_checks": [k for k, v in validation["checks"].items() if not v]}, indent=2))
    if not validation["passed"]:
        raise SystemExit("Frame mapping validation failed; do not use this mapping for rendering")


if __name__ == "__main__":
    main()
