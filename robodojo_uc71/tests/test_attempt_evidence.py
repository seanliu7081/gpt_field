import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robodojo_uc71.attempt_evidence import frame_records, finalize_video, finalize_trajectory


def make_evidence(tmp_path):
    attempt = tmp_path / "attempt"
    (attempt / "worker").mkdir(parents=True)
    (attempt / "parent/video_frames").mkdir(parents=True)
    output = attempt / "finalized"; output.mkdir()
    for index in range(3):
        images = {}
        for camera in ("head", "left_wrist", "right_wrist"):
            path = attempt / "worker" / f"{camera}_{index}.png"
            Image.fromarray(np.full((480,640,3), index * 60, dtype=np.uint8)).save(path)
            images[camera] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "bytes": path.stat().st_size, "width":640,"height":480}
        robot = {arm: {"tool_pose_wxyz": [index*.001,0,0,1,0,0,0], "joint_position_rad": [index*.001]*6,
                       "finger_position_m": [.044,.044]} for arm in ("left", "right")}
        (attempt / f"parent/capture_{index:06d}.json").write_text(json.dumps({
            "public": {"images": images, "robot": robot}, "worker_readonly_capture_contract_verified": True}))
        (attempt / f"parent/video_frames/frame_{index:04d}.json").write_text(json.dumps({
            "schema":"robodojo-parent-video-frame-1", "native_action_index":index,
            "images":images, "capture_is_readonly":True}))
    return attempt, output


def test_real_three_view_encode_ffprobe_and_both_arm_plot(tmp_path):
    attempt, output = make_evidence(tmp_path)
    result = finalize_video(attempt, output)
    assert result["frame_count"] == 3
    assert result["native_action_frames"] == 2
    assert result["ffprobe"]["streams"][0]["width"] == 1920
    plot = finalize_trajectory(attempt, output)
    assert plot["samples"] == 3
    rows = json.loads((output / "trajectory.json").read_text())["rows"]
    assert [row["policy_simulation_time_s"] for row in rows] == [0,.04,.08]
    assert set(rows[-1]["robot"]) == {"left","right"}


def test_changed_image_cannot_be_encoded_as_original_evidence(tmp_path):
    attempt, output = make_evidence(tmp_path)
    (attempt / "worker/head_1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image_binding"):
        finalize_video(attempt, output)
    assert not (output / "three_view.mp4").exists()


def test_missing_action_frame_is_not_silently_skipped(tmp_path):
    attempt, _ = make_evidence(tmp_path)
    (attempt / "parent/video_frames/frame_0001.json").unlink()
    with pytest.raises(ValueError, match="noncontiguous"):
        frame_records(attempt)


def test_trajectory_requires_matching_synchronized_capture(tmp_path):
    attempt, output = make_evidence(tmp_path)
    (attempt / "parent/capture_000001.json").unlink()
    with pytest.raises(ValueError, match="capture_binding"):
        finalize_trajectory(attempt, output)
