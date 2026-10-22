"""Input isolation, units/calibration, and refusal checks; no model downloads."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "infer_depth.py"
SPEC = importlib.util.spec_from_file_location("rgb_infer_depth", SOURCE)
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


def make_inputs(tmp_path):
    rows = []
    for name in sorted(infer.EXPECTED_NAMES):
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.full((12, 18, 3), 42, dtype=np.uint8)).save(path)
        rows.append({"name": name, "rgb": path.name, "sha256": infer.sha256(path)})
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({"schema_version": 1, "inputs": rows}))
    return path


def test_rgb_manifest_has_native_dimensions_and_verified_bytes(tmp_path):
    path = make_inputs(tmp_path)
    rows = infer.read_inputs(path)
    assert len(rows) == 3
    assert {(x["width"], x["height"]) for x in rows} == {(18, 12)}
    assert all(Path(x["rgb_path"]).is_absolute() for x in rows)


def test_ground_truth_metadata_is_not_accepted_by_inference(tmp_path):
    path = make_inputs(tmp_path)
    data = json.loads(path.read_text())
    data["K_ground_truth"] = np.eye(3).tolist()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only schema_version"):
        infer.read_inputs(path)


def test_rgb_hash_change_is_rejected_before_inference(tmp_path):
    path = make_inputs(tmp_path)
    data = json.loads(path.read_text())
    with (tmp_path / data["inputs"][0]["rgb"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        infer.read_inputs(path)


def test_duplicate_snapshot_cannot_replace_missing_snapshot(tmp_path):
    path = make_inputs(tmp_path)
    data = json.loads(path.read_text())
    data["inputs"][1] = data["inputs"][0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        infer.read_inputs(path)


def test_predicted_camera_uses_integer_pixel_centers():
    K = infer.predicted_intrinsics(256.0, 512, 512)
    assert np.array_equal(K, [[256, 0, 255.5], [0, 256, 255.5], [0, 0, 1]])
    ray = np.linalg.solve(K, [255.5, 255.5, 1.0])
    assert np.array_equal(ray, [0, 0, 1])
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            infer.predicted_intrinsics(invalid, 512, 512)


def test_depth_units_and_fov_consistency_are_not_silently_repaired():
    depth = np.ones((12, 18), dtype=np.float32)
    infer.validate_prediction(depth, 9.0, 90.0, (12, 18))
    with pytest.raises(ValueError, match="disagree"):
        infer.validate_prediction(depth, 256.0, 90.0, (12, 18))
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite positive"):
        infer.validate_prediction(depth, 9.0, 90.0, (12, 18))


def test_existing_output_refusal_does_not_mutate_old_manifest(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "manifest.json"
    marker.write_text('"unchanged"')
    args = argparse.Namespace(output=output)
    with pytest.raises(FileExistsError, match="Refusing existing output"):
        infer.run(args)
    assert marker.read_text() == '"unchanged"'


def test_terminal_hash_audit_detects_missing_or_changed_artifact(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    hashes = {str(path): infer.sha256(path)}
    assert all(infer.check_unchanged(hashes).values())
    path.write_bytes(b"after")
    assert not any(infer.check_unchanged(hashes).values())
    path.unlink()
    assert not any(infer.check_unchanged(hashes).values())
