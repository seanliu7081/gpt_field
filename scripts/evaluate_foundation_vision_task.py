#!/usr/bin/env python3
"""Run the existing reviewed task with local foundation-vision executive feedback.

Arguments not listed here pass unchanged to evaluate_agent_speedup_v2_user_check.py.
The original runner owns the task, subscription, approval, controller and stop rules.
Only the execution client receives vision feedback; the initial plan is unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = "scripts/evaluate_foundation_vision_task.py"
MODE = "vision_conditioned_executive"


def _positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("vision max age must be finite and positive")
    return seconds


def _query(value):
    if not value.strip() or len(value) > 256:
        raise argparse.ArgumentTypeError("vision query must be nonempty and at most 256 characters")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sam-checkpoint", type=Path, required=True,
                        help="Existing local official SAM 3.1 multiplex checkpoint")
    parser.add_argument("--da3-model", type=Path, required=True,
                        help="Existing local DA3 directory containing config.json and model.safetensors")
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--vision-max-age-s", type=_positive_seconds, default=5.0)
    parser.add_argument("--vision-query", action="append", type=_query, default=[],
                        help="Additional text region query; repeat for multiple descriptions")
    return parser


def _validate_local_models(args):
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.da3_model = args.da3_model.expanduser().resolve()
    if not args.sam_checkpoint.is_file():
        raise ValueError("--sam-checkpoint must name an existing local file")
    if not args.da3_model.is_dir() or any(
            not (args.da3_model / name).is_file()
            for name in ("config.json", "model.safetensors")):
        raise ValueError("--da3-model must be a local directory with config.json and model.safetensors")


def _load_runner():
    path = ROOT / "scripts/evaluate_agent_speedup_v2_user_check.py"
    spec = importlib.util.spec_from_file_location("_foundation_vision_task_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _build_vision_engine(args):
    # No imports, network, credentials, GPU work or checkpoint reads on preflight.
    from gpt_field.foundation_vision.backends import Sam31Segmenter, DA3DepthEstimator
    from gpt_field.foundation_vision.engine import EngineConfig, VisionEngine

    sam = Sam31Segmenter(checkpoint_path=args.sam_checkpoint, device=args.vision_device,
                         local_files_only=True)
    da3 = DA3DepthEstimator(model_id=str(args.da3_model), device=args.vision_device,
                           local_files_only=True)
    timings = {}
    for name, backend in (("sam", sam), ("da3", da3)):
        started = perf_counter()
        backend._load()
        timings[name + "_load_s"] = perf_counter() - started
    engine = VisionEngine(sam, da3, EngineConfig(max_age_s=args.vision_max_age_s))
    return engine, {"load_times": timings, "models": {
        "sam": sam.provenance(), "da3": da3.provenance()}}


def _live_client_class():
    from gpt_field.foundation_vision.live import FoundationVisionAgentClient
    return FoundationVisionAgentClient


def _settings(args, preload):
    # Only immutable model identity belongs in the approval hash. Later model
    # alignment/quality diagnostics and timings must not change that identity.
    identities = {}
    for name, model in preload.get("models", {}).items():
        identities[name] = {key: model.get(key) for key in (
            "backend", "model_id", "revision", "checkpoint_sha256", "precision")}
    return {
        "mode": MODE, "sam_checkpoint": str(args.sam_checkpoint),
        "da3_model": str(args.da3_model), "device": args.vision_device,
        "max_age_s": args.vision_max_age_s, "models": identities,
        "extra_target_texts": list(args.vision_query),
        "local_files_only": True, "initial_plan_vision_feedback": False,
        "automatic_depth_routing": False, "automatic_action_gate": False,
        "routing_geometry_source": "astra_estimated_obstacle_boxes",
    }


def _install(runner, args, engine, preload):
    settings = _settings(args, preload)
    original_options = runner.execution_options
    original_inventory = runner.runtime_inventory
    original_execute = runner.execute_reviewed

    def execution_options(*positional, **kwargs):
        result = dict(original_options(*positional, **kwargs))
        result["foundation_vision"] = deepcopy(settings)
        return result

    def runtime_inventory(*positional, **kwargs):
        result = dict(original_inventory(*positional, **kwargs))
        paths = list((ROOT / "src/gpt_field/foundation_vision").rglob("*.py"))
        paths.append(ROOT / WRAPPER_PATH)
        for path in sorted(paths):
            result[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        # An explicit entry also binds vision options when a test runner's
        # original inventory does not derive @configuration from execution_options.
        result["@foundation_vision"] = sha256(json.dumps(
            settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return result

    def execute_reviewed(output, *positional, **kwargs):
        if engine is None:
            raise RuntimeError("Execution requires successfully preloaded local vision models")
        if kwargs.get("client_factory") is not None:
            raise ValueError("This wrapper owns the foundation-vision execution client")
        client_type = _live_client_class()
        vision_output = Path(output).resolve() / "foundation_vision"

        def client_factory(*client_args, **client_kwargs):
            # The original runner creates trial/ first. The client exclusively
            # creates this fresh child directory and rejects existing artifacts.
            return client_type(*client_args, vision_engine=engine,
                               vision_output_dir=vision_output,
                               vision_extra_targets=tuple(args.vision_query), **client_kwargs)

        kwargs["client_factory"] = client_factory
        return original_execute(output, *positional, **kwargs)

    runner.execution_options = execution_options
    runner.runtime_inventory = runtime_inventory
    runner.execute_reviewed = execute_reviewed


def main(argv=None):
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not forwarded or forwarded[0] not in {"prepare", "run", "start", "preflight", "review"}:
        parser.error("Supply an original runner command: start, prepare, run, preflight or review")
    _validate_local_models(args)
    # Load both local models before runner.main can read credentials, reset a
    # simulator or request the initial model plan. Exceptions stop immediately.
    if forwarded[0] == "preflight":
        engine, preload = None, {}
    else:
        engine, preload = _build_vision_engine(args)
        print(json.dumps({"event": "foundation_vision_preloaded", "mode": MODE,
                          **preload}, allow_nan=False), flush=True)
    runner = _load_runner()
    _install(runner, args, engine, preload)
    return runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
