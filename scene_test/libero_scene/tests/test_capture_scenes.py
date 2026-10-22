from pathlib import Path
import base64
import io
import json
import sys
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_scenes as capture


def test_geometry_export_keeps_full_pose_and_excludes_robot():
    metadata = [dict(id=i, name=f"g{i}", body="body", object="object", role=role,
                     type=kind, size=[.1, .2, .3], contype=1, conaffinity=1, margin_m=0.)
                for i, (role, kind) in enumerate((("scene", "box"), ("robot", "box"), ("scene", "plane")))]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    boxes, planes = capture.geometry_records(metadata, np.array([[1, 2, 3]] * 3), np.array([rotation] * 3))
    assert len(boxes) == len(planes) == 1
    assert boxes[0]["center_m"] == [1, 2, 3]
    assert boxes[0]["half_extents_m"] == [.1, .2, .3]
    assert planes[0]["normal_world"] == [0, 0, 1]
    bounds = capture.object_bounds(boxes)["object"]
    np.testing.assert_allclose(bounds["lower_m"], [.8, 1.9, 2.7])


def test_unknown_geometry_is_not_silently_discarded():
    record = dict(id=0, name="mesh", body="body", object="object", role="scene", type="mesh",
                  size=[1, 1, 1], contype=1, conaffinity=1, margin_m=0.)
    with pytest.raises(ValueError, match="cannot silently omit"):
        capture.geometry_records([record], np.zeros((1, 3)), np.eye(3)[None])


def test_preaction_images_are_not_after_action_images(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [{"event": "reset", "observation": {"label": "initial"}},
            {"event": "executed", "step": 59, "observation": {"label": "before60"}},
            {"event": "executed", "step": 60, "observation": {"label": "after60"}}]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert capture.trace_preaction_observations(path, [0, 60]) == {
        0: {"label": "initial"}, 60: {"label": "before60"}}


def test_observation_image_identity_and_state_mismatch():
    stream = io.BytesIO()
    Image.new("RGB", (3, 3), "red").save(stream, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    expected = {"observation/image": data, "observation/wrist_image": data,
                "observation/state": [0.] * 8, "prompt": "test"}
    assert capture.compare_observation(expected, expected)["complete_observation_exact"]
    changed = {**expected, "observation/state": [1.] * 8}
    with pytest.raises(ValueError, match="nonimage mismatch"):
        capture.compare_observation(changed, expected)


def test_existing_output_is_not_modified(tmp_path):
    marker = tmp_path / "manifest.json"
    marker.write_text("old")
    with pytest.raises(FileExistsError):
        capture.run(tmp_path)
    assert marker.read_text() == "old"
