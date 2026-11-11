#!/usr/bin/env python3
"""Inspect dependencies or run foundation vision on saved RGB; never move a robot.

Run with PYTHONPATH=src. The only model calls are explicitly requested local
vision inference. Outputs are new artifacts, not LIBERO completion evaluations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from time import perf_counter

from gpt_field.foundation_vision import VERSION
from gpt_field.foundation_vision.bridge import file_sha256, load_manifest


def _dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _backends(args):
    from gpt_field.foundation_vision.backends import Sam31Segmenter, DA3DepthEstimator
    return (Sam31Segmenter(checkpoint_path=args.sam_checkpoint, device=args.device,
                          local_files_only=args.local_files_only),
            DA3DepthEstimator(model_id=args.da3_model, device=args.device,
                              local_files_only=args.local_files_only, process_res=args.process_res))


def _depth_backend(args):
    from gpt_field.foundation_vision.backends import DA3DepthEstimator
    return DA3DepthEstimator(model_id=args.da3_model, device=args.device,
                             local_files_only=args.local_files_only, process_res=args.process_res)


def inspect_backends(args):
    segmenter, depth = _backends(args)
    return {"schema_version": VERSION, "inspection_only": True, "models_loaded": False,
            "sam": segmenter.preflight(), "da3": depth.preflight()}


def _native_evidence(depth, output, frame_id):
    """Preserve optional raw model arrays independently of normalized geometry."""
    method = getattr(depth, "native_evidence", None)
    arrays = method() if callable(method) else {}
    if not arrays:
        return None
    import numpy as np

    arrays = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("native prediction arrays must not require pickle")
    path = output / f"native_prediction_{frame_id:06d}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": path.name, "sha256": file_sha256(path)}


def run_manifest(args, *, engine_factory=None, backend_factory=None, depth_only=False):
    """Offline replay with injectable factories only for ordinary Python tests."""
    started = perf_counter()
    manifest = load_manifest(args.manifest)
    if not depth_only and not manifest.targets:
        raise ValueError("run requires at least one target; use run-depth for geometry-only evaluation")
    output = Path(args.output).resolve()
    # Never overwrite or reuse a partial/complete previous evaluation.
    output.mkdir(parents=True, exist_ok=False)
    base = {"schema_version": VERSION,
            "evaluation_kind": "offline_rgb_depth_only" if depth_only else "offline_rgb_perception",
            "segmentation": "not_run" if depth_only else "requested",
            "target_processing": "not_run" if depth_only else "requested",
            "source_kind": manifest.source_kind, "input_manifest": manifest.manifest_path,
            "input_sha256": manifest.source_hashes, "robot_actions": 0,
            "simulator_steps": 0, "astra_calls": 0, "task_success": None,
            "collision_free": None, "output_directory": str(output),
            "models": {}, "frames_completed": 0,
            "timing_clock": "recorded_sensor_time_plus_measured_inference_latency",
            "interpretation": "Perception outputs are estimates, not task/contact/collision certificates."}
    (output / "provenance.json").write_text(_dump(base) + "\n", encoding="utf-8")
    frame_times = []
    try:
        if depth_only:
            from gpt_field.foundation_vision.geometry import GeometryConfig, build_geometry
            depth = (backend_factory or _depth_backend)(args)
            geometry_config = GeometryConfig()
            base["models"] = {"da3": depth.provenance()}
            base["geometry_config"] = asdict(geometry_config)
        else:
            segmenter, depth = (backend_factory or _backends)(args)
            if engine_factory is None:
                from gpt_field.foundation_vision.engine import VisionEngine
                engine_factory = VisionEngine
            engine = engine_factory(segmenter, depth)
            base["models"] = {"sam": segmenter.provenance(), "da3": depth.provenance()}
            config = getattr(engine, "config", None)
            if config is not None:
                base["engine_config"] = asdict(config) if is_dataclass(config) else str(config)
        with (output / "snapshots.jsonl").open("x", encoding="utf-8") as stream:
            for frame in manifest.frames:
                tick = perf_counter()
                if depth_only:
                    predictions = depth.estimate(frame)
                    geometry = build_geometry(frame, predictions, {}, geometry_config)
                    geometry_data = geometry.to_dict()
                    elapsed = perf_counter() - tick
                    snapshot_data = {"version": VERSION, "frame_id": frame.frame_id,
                        "captured_at_s": frame.captured_at_s,
                        "generated_at_s": frame.captured_at_s + elapsed,
                        "age_s": elapsed, "status": "depth_only",
                        "segmentation": "not_run", "target_processing": "not_run",
                        "interaction": "unknown", "geometry": geometry_data}
                    executive_summary = {"status": "depth_only", "frame_id": frame.frame_id,
                                         "target_processing": "not_run", "interaction": "unknown"}
                else:
                    snapshot = engine.process(frame, manifest.targets, now_s=frame.captured_at_s)
                    snapshot_data, executive_summary = snapshot.to_dict(), snapshot.summary()
                native_prediction = _native_evidence(depth, output, frame.frame_id)
                backend_timings = {}
                for name, backend in (("sam", locals().get("segmenter")), ("da3", depth)):
                    timings = getattr(backend, "last_timings", None)
                    if timings is not None:
                        backend_timings[name] = dict(timings)
                elapsed = perf_counter() - tick
                # Snapshot retains dense geometry; summary is the executive-facing view.
                record = {"frame_id": frame.frame_id, "captured_at_s": frame.captured_at_s,
                          "wall_elapsed_s": elapsed, "snapshot": snapshot_data,
                          "executive_summary": executive_summary,
                          "native_prediction": native_prediction,
                          "backend_timings": backend_timings}
                stream.write(_dump(record) + "\n")
                stream.flush()
                frame_times.append(elapsed)
                base["frames_completed"] += 1
        base["status"] = "complete"
    except Exception as exc:
        base.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        # Refresh after lazy loads so revisions/checkpoint paths reflect what ran.
        for name, backend in (("sam", locals().get("segmenter")), ("da3", locals().get("depth"))):
            if backend is not None:
                try:
                    base["models"][name] = backend.provenance()
                except Exception:
                    pass
        base["frame_wall_times_s"] = frame_times
        base["elapsed_s"] = perf_counter() - started
        (output / "summary.json").write_text(_dump(base) + "\n", encoding="utf-8")
    return base


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run", "run-depth"):
        command = sub.add_parser(name)
        command.add_argument("--sam-checkpoint", help="Path to the official SAM 3.1 checkpoint")
        command.add_argument("--da3-model", default="depth-anything/DA3-LARGE-1.1")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--local-files-only", action="store_true", help="Forbid checkpoint downloads")
        command.add_argument("--process-res", type=int, default=504)
        if name in ("run", "run-depth"):
            command.add_argument("--manifest", required=True, type=Path)
            command.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = inspect_backends(args) if args.command == "inspect" else run_manifest(
            args, depth_only=args.command == "run-depth")
        print(_dump(result))
        return 0
    except Exception as exc:
        print(_dump({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
