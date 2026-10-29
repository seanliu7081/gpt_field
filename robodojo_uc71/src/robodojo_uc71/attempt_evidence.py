"""Finalize operator videos/trajectories without changing native result records."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .approval import file_sha256
from .native_worker_client import write_private_record


def frame_records(attempt):
    attempt = Path(attempt).resolve(strict=True)
    records = []
    for path in sorted((attempt / "parent/video_frames").glob("frame_*.json")):
        value = json.loads(path.read_text())
        if (value.get("schema") != "robodojo-parent-video-frame-1"
                or value.get("native_action_index") != len(records)
                or value.get("capture_is_readonly") is not True
                or set(value.get("images", {})) != {"head", "left_wrist", "right_wrist"}):
            raise ValueError("noncontiguous_or_invalid_video_frames")
        for entry in value["images"].values():
            image = Path(entry["path"])
            if (image.is_symlink() or not image.resolve(strict=True).is_relative_to(attempt / "worker")
                    or any(c in str(image) for c in "'\r\n")
                    or file_sha256(image) != entry["sha256"]):
                raise ValueError("video_image_binding_invalid")
        records.append(value)
    return records


def finalize_video(attempt, output):
    """Encode action-indexed clean RGB at native25Hz, then independently ffprobe."""
    output = Path(output)
    frames = frame_records(attempt)
    if not frames:
        return {"available": False, "reason": "no_acknowledged_action_frames"}
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    for camera in ("head", "left_wrist", "right_wrist"):
        listing = output / f"{camera}_frames.txt"
        with listing.open("x") as stream:
            stream.write("".join("file '" + frame["images"][camera]["path"] + "'\n" for frame in frames))
        args.extend(["-r", "25", "-f", "concat", "-safe", "0", "-i", str(listing)])
    video = output / "three_view.mp4"
    args.extend(["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]", "-map", "[v]",
                 "-frames:v", str(len(frames)), "-r", "25", "-c:v", "libx264", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)])
    subprocess.run(args, check=True, capture_output=True, timeout=180)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(video)],
        text=True, timeout=30))
    stream = probe["streams"][0]
    if (stream["width"] != 1920 or stream["height"] != 480 or stream["avg_frame_rate"] != "25/1"
            or int(stream["nb_read_frames"]) != len(frames)
            or abs(float(stream["duration"]) - len(frames) / 25) > .001):
        raise ValueError("encoded_video_verification_failed")
    result = {"available": True, "path": str(video), "sha256": file_sha256(video),
        "frame_count": len(frames), "native_action_frames": len(frames) - 1,
        "initial_frame_included": True, "fps": 25, "view_order": ["head", "left_wrist", "right_wrist"],
        "ffprobe": probe, "ffmpeg_argv": args, "model_latency_in_video": False}
    write_private_record(output / "video_validation.json", result)
    return result


def finalize_trajectory(attempt, output):
    """Match indexed video images to the same synchronized measured-robot capture."""
    attempt, output = Path(attempt), Path(output)
    frames = frame_records(attempt)
    captures = {}
    for path in sorted((attempt / "parent").glob("capture_*.json")):
        record = json.loads(path.read_text())
        if record.get("worker_readonly_capture_contract_verified") is not True:
            raise ValueError("unverified_trajectory_capture")
        public = record["public"]
        key = tuple(public["images"][c]["sha256"] for c in ("head", "left_wrist", "right_wrist"))
        captures.setdefault(key, []).append((path, public["robot"]))
    rows = []
    for frame in frames:
        key = tuple(frame["images"][c]["sha256"] for c in ("head", "left_wrist", "right_wrist"))
        # File identities disambiguate equal pixel arrays at consecutive holds.
        matches = [(p, robot) for p, robot in captures.get(key, [])
                   if json.loads(p.read_text())["public"]["images"] == frame["images"]]
        if len(matches) != 1:
            raise ValueError("trajectory_capture_binding_ambiguous")
        path, robot = matches[0]
        rows.append({"native_action_index": frame["native_action_index"],
            "policy_simulation_time_s": frame["native_action_index"] * .04,
            "capture": {"path": str(path), "sha256": file_sha256(path)}, "robot": robot})
    write_private_record(output / "trajectory.json", {"schema": "robodojo-both-arm-trajectory-1",
        "scope": "operator_evidence", "rows": rows, "model_input_eligible": False})
    if not rows:
        return {"available": False, "reason": "no_acknowledged_action_frames"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    times = [row["policy_simulation_time_s"] for row in rows]
    for column, arm in enumerate(("left", "right")):
        for component, label in enumerate(("x", "y", "z")):
            axes[0, column].plot(times, [r["robot"][arm]["tool_pose_wxyz"][component] for r in rows], label=label)
        for joint in range(6):
            axes[1, column].plot(times, [r["robot"][arm]["joint_position_rad"][joint] for r in rows], label=str(joint + 1))
        for finger in range(2):
            axes[2, column].plot(times, [r["robot"][arm]["finger_position_m"][finger] for r in rows], label=str(finger + 1))
        axes[0, column].set_title(arm.capitalize() + " arm")
        for row, label in enumerate(("Measured TCP position (m)", "Measured joints (rad)", "Measured fingers (m)")):
            axes[row, column].set_ylabel(label); axes[row, column].grid(alpha=.25)
            axes[row, column].legend(ncol=3, fontsize=8)
        axes[2, column].set_xlabel("Policy simulation time (s)")
    fig.tight_layout()
    plot = output / "both_arm_trajectory.png"
    fig.savefig(plot, dpi=160); plt.close(fig)
    return {"available": True, "path": str(plot), "sha256": file_sha256(plot), "samples": len(rows)}
