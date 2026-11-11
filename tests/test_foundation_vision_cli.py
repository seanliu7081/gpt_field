"""Offline driver preserves old artifacts and records failure/provenance honestly."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from test_foundation_vision_bridge import write_manifest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_foundation_vision.py"
    spec = importlib.util.spec_from_file_location("evaluate_foundation_vision", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Backend:
    def provenance(self):
        return {"backend": "test_double", "real_model": False}


class Snapshot:
    def to_dict(self):
        return {"geometry": {"space": "unknown"}, "events": []}

    def summary(self):
        return {"requires_observation": True}


def test_offline_driver_records_all_frames_without_task_claims(tmp_path):
    manifest, _ = write_manifest(tmp_path, frame_count=2)
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    seen = []
    class Engine:
        def __init__(self, *backends):
            pass
        def process(self, frame, targets, *, now_s):
            seen.append((frame.frame_id, now_s, targets[0].key))
            return Snapshot()
    result = module().run_manifest(args, engine_factory=Engine,
                                   backend_factory=lambda _: (Backend(), Backend()))
    assert seen == [(0, 0., "target"), (1, 1., "target")]
    assert result["status"] == "complete" and result["frames_completed"] == 2
    assert result["robot_actions"] == result["simulator_steps"] == result["astra_calls"] == 0
    assert result["task_success"] is None and result["collision_free"] is None
    records = [json.loads(line) for line in (args.output / "snapshots.jsonl").read_text().splitlines()]
    assert [row["frame_id"] for row in records] == [0, 1]
    assert len(json.loads((args.output / "summary.json").read_text())["input_sha256"]) == 3


def test_existing_output_is_never_reused(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    output = tmp_path / "result"
    output.mkdir()
    sentinel = output / "do_not_change.txt"
    sentinel.write_text("old result")
    with pytest.raises(FileExistsError):
        module().run_manifest(SimpleNamespace(manifest=manifest, output=output),
                              backend_factory=lambda _: pytest.fail("must not load a model"))
    assert sentinel.read_text() == "old result"
    assert list(output.iterdir()) == [sentinel]


def test_backend_failure_is_preserved_as_incomplete_evidence(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    def unavailable(_):
        raise RuntimeError("official checkpoint unavailable")
    with pytest.raises(RuntimeError, match="checkpoint"):
        module().run_manifest(args, backend_factory=unavailable)
    result = json.loads((args.output / "summary.json").read_text())
    assert result["status"] == "error" and result["frames_completed"] == 0
    assert result["task_success"] is None and result["error"]["type"] == "RuntimeError"


def test_input_failure_precedes_model_load_and_output_creation(tmp_path):
    manifest, data = write_manifest(tmp_path)
    data["frames"][0]["cameras"][1]["observed_at_s"] = 7.
    manifest.write_text(json.dumps(data))
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    with pytest.raises(ValueError, match="timestamp"):
        module().run_manifest(args, backend_factory=lambda _: pytest.fail("must not load a model"))
    assert not args.output.exists()


def test_inspect_uses_preflight_without_loading_inference(monkeypatch):
    driver = module()
    class InspectBackend:
        def preflight(self):
            return {"dependencies_ready": False, "checkpoint_exists": False}
    monkeypatch.setattr(driver, "_backends", lambda _: (InspectBackend(), InspectBackend()))
    report = driver.inspect_backends(SimpleNamespace())
    assert report["models_loaded"] is False
    assert report["sam"]["dependencies_ready"] is False


def test_driver_runs_actual_engine_on_injected_sensor_predictions(tmp_path):
    from gpt_field.foundation_vision.types import Detection, DepthEstimate

    manifest, _ = write_manifest(tmp_path, frame_count=2)
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    class Segments(Backend):
        def segment(self, frame, targets):
            return [Detection(camera.name, target.key, np.ones(camera.rgb.shape[:2], dtype=bool), .8)
                    for target in targets for camera in frame.cameras]
    class Depths(Backend):
        def estimate(self, frame):
            return {camera.name: DepthEstimate(camera.name, np.full(camera.rgb.shape[:2], 2.),
                    camera.intrinsics, camera.world_to_camera, source="synthetic_test_plane")
                    for camera in frame.cameras}
    result = module().run_manifest(args, backend_factory=lambda _: (Segments(), Depths()))
    assert result["status"] == "complete" and result["frames_completed"] == 2
    rows = [json.loads(line) for line in (args.output / "snapshots.jsonl").read_text().splitlines()]
    assert rows[0]["executive_summary"]["frame_id"] == 0
    assert rows[1]["snapshot"]["generated_at_s"] >= rows[1]["captured_at_s"]
    assert result["engine_config"]["max_age_s"] > 0


def test_depth_only_records_no_segmentation_and_never_creates_engine(tmp_path):
    from gpt_field.foundation_vision.types import DepthEstimate

    manifest, data = write_manifest(tmp_path)
    data["targets"] = []
    manifest.write_text(json.dumps(data))
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "depth_result")
    class Depths(Backend):
        def estimate(self, frame):
            return {camera.name: DepthEstimate(camera.name, np.full(camera.rgb.shape[:2], 2.),
                    camera.intrinsics, camera.world_to_camera, source="synthetic_test_plane")
                    for camera in frame.cameras}
    result = module().run_manifest(args, depth_only=True,
        engine_factory=lambda *_: pytest.fail("depth-only must not construct the full pipeline"),
        backend_factory=lambda _: Depths())
    assert result["status"] == "complete"
    assert result["evaluation_kind"] == "offline_rgb_depth_only"
    assert result["segmentation"] == result["target_processing"] == "not_run"
    assert set(result["models"]) == {"da3"}
    record = json.loads((args.output / "snapshots.jsonl").read_text())
    assert record["snapshot"]["interaction"] == "unknown"
    assert record["snapshot"]["geometry"]


def test_full_pipeline_requires_target_before_creating_output(tmp_path):
    manifest, data = write_manifest(tmp_path)
    data["targets"] = []
    manifest.write_text(json.dumps(data))
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    with pytest.raises(ValueError, match="at least one target"):
        module().run_manifest(args, backend_factory=lambda _: pytest.fail("must not load model"))
    assert not args.output.exists()


def test_native_predictions_and_backend_timings_are_saved_per_frame(tmp_path):
    from gpt_field.foundation_vision.bridge import file_sha256

    manifest, _ = write_manifest(tmp_path, frame_count=2)
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    class Depths(Backend):
        last_timings = {}
        frame_id = None
        evidence_calls = 0

        def native_evidence(self):
            self.evidence_calls += 1
            return {"raw_depth_model_units": np.array([self.frame_id, np.nan], dtype=np.float32)}

    depth = Depths()
    class Engine:
        def __init__(self, *backends):
            pass

        def process(self, frame, targets, *, now_s):
            depth.frame_id = frame.frame_id
            depth.last_timings = {"inference_s": 0.2 + frame.frame_id, "load_s": 0.1}
            return Snapshot()

    module().run_manifest(args, engine_factory=Engine,
                          backend_factory=lambda _: (Backend(), depth))
    rows = [json.loads(line) for line in (args.output / "snapshots.jsonl").read_text().splitlines()]
    assert depth.evidence_calls == 2
    for frame_id, row in enumerate(rows):
        native = row["native_prediction"]
        assert native["path"] == f"native_prediction_{frame_id:06d}.npz"
        path = args.output / native["path"]
        assert native["sha256"] == file_sha256(path)
        with np.load(path, allow_pickle=False) as prediction:
            np.testing.assert_equal(prediction["raw_depth_model_units"], [frame_id, np.nan])
        assert row["backend_timings"] == {"da3": {"inference_s": 0.2 + frame_id, "load_s": 0.1}}


def test_depth_snapshot_age_includes_geometry_serialization_and_empty_native_is_optional(tmp_path, monkeypatch):
    import gpt_field.foundation_vision.geometry as geometry_module

    manifest, _ = write_manifest(tmp_path)
    args = SimpleNamespace(manifest=manifest, output=tmp_path / "result")
    clock = [0.]
    driver = module()
    monkeypatch.setattr(driver, "perf_counter", lambda: clock[0])
    class Geometry:
        def to_dict(self):
            clock[0] += 3.
            return {"space": "unknown"}

    class Depths(Backend):
        def estimate(self, frame):
            clock[0] += 2.
            return {}

        def native_evidence(self):
            return {}

    monkeypatch.setattr(geometry_module, "build_geometry", lambda *args: Geometry())
    driver.run_manifest(args, depth_only=True, backend_factory=lambda _: Depths())
    row = json.loads((args.output / "snapshots.jsonl").read_text())
    assert row["snapshot"]["age_s"] == 5.
    assert row["snapshot"]["generated_at_s"] == 5.
    assert row["native_prediction"] is None and row["backend_timings"] == {}
    assert not list(args.output.glob("*.npz"))
