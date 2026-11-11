#!/usr/bin/env python3
"""Inspect or run one explicitly approved integrated foundation-harness episode.

The default command only inspects local configuration. `run --approved` is an
explicit full-task launch, not a preparation or smoke command. It reuses the
existing evaluator, public simulator adapter, subscription transport and video
timing, without an unused legacy initial-plan call or automatic episode retry.
Astra supplies keyposes; local planning addresses the next keypose. Unknown
geometry uses bounded motion with observation and is not a clearance certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "foundation-harness-runner-2"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("inspect", "run"), nargs="?", default="inspect")
    parser.add_argument("--approved", action="store_true", help="Launch only after explicit user approval of this full task")
    parser.add_argument("--output", type=Path, help="Fresh run root; actual episode is written in trial/")
    parser.add_argument("--task", required=True, help="Existing official task profile/alias")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--da3-model", type=Path, required=True)
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--vision-max-age-s", type=_positive, default=5.0)
    parser.add_argument("--sim-python")
    parser.add_argument("--libero-root")
    parser.add_argument("--robot-xml", type=Path, help="Optional static Panda robot.xml; otherwise discovered from simulator environment")
    parser.add_argument("--expected-initial-observation-sha256")
    parser.add_argument("--initial-state-override", type=Path)
    parser.add_argument("--task-bddl-override", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--no-video", action="store_true")
    return parser


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def _configuration(args, runner):
    base = runner.base
    task = base.resolve_v2_task(args.task)
    base._validate_episode_index(args.episode_index)
    sim, monitor = base.build_sim_config(task.key,
        sim_python=args.sim_python or base.DEFAULT_SIM_PYTHON,
        libero_root=args.libero_root or base.DEFAULT_LIBERO_ROOT)
    for option in ("initial_state_override", "task_bddl_override"):
        path = getattr(args, option)
        if path is not None:
            if not path.expanduser().is_file():
                raise ValueError(f"missing --{option.replace('_', '-')} input")
            sim = replace(sim, **{option: str(path.expanduser().resolve())})
    runner.require_custom_reset_sha(sim, args.expected_initial_observation_sha256)
    config = base.configuration()
    base.validate_configuration(config, sim, task, allow_initial_state_override=True,
                                allow_task_bddl_override=True)
    if any(getattr(config, name, None) is not None for name in ("max_api_calls", "max_wall_time_s")):
        raise ValueError("full task requires uncapped subscription configuration")
    args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.da3_model = args.da3_model.expanduser().resolve()
    if not args.sam_checkpoint.is_file():
        raise ValueError("SAM checkpoint must be an existing local file")
    if not args.da3_model.is_dir() or not all((args.da3_model/name).is_file()
                                             for name in ("config.json", "model.safetensors")):
        raise ValueError("DA3 requires a local directory containing config.json and model.safetensors")
    if monitor.get("oracle", False) or getattr(sim, "stop_on_collision", True):
        raise ValueError("public sensors and record-only evaluator collisions are required")
    settings = {"version": VERSION, "mode": "integrated_foundation_harness", "task": task.key,
        "keypose_owner": "astra", "planning_scope": "next_keypose",
        "unknown_space_policy": "bounded_step_with_observation",
        "geometry_hypotheses_not_collision_certificate": True,
        "task_prompt": task.prompt, "episode_index": args.episode_index,
        "model": config.model, "reasoning_effort": "high", "billing": "chatgpt_subscription",
        "api_key_fallback": False, "automatic_episode_retry": False,
        "unused_initial_model_plan": False, "episode_action_cap": None, "episode_time_cap_s": None,
        "episode_model_call_cap": None, "collision_evaluator": "record_only",
        "task_success": "existing independent raw LIBERO evaluator",
        "sam_checkpoint": str(args.sam_checkpoint), "da3_model": str(args.da3_model),
        "vision_device": args.vision_device, "vision_max_age_s": args.vision_max_age_s,
        "robot_xml": str(args.robot_xml.expanduser().resolve()) if args.robot_xml is not None else None,
        "expected_initial_observation_sha256": args.expected_initial_observation_sha256,
        "video": not args.no_video, "simulator": asdict(sim), "subscription": asdict(config),
        "models_preloaded_before_auth_and_reset": True,
        "geometry_limitation": "predicted visible surfaces; no certified collision freedom",
        "control_timing": "includes first semantic planning, local perception and execution; video measured separately"}
    return config, sim, monitor, task, settings


def _freeze(root, args, runner, config, sim, monitor, settings):
    """Archive current files once; never reset, checkout or edit the dirty tree."""
    inventory = runner.runtime_inventory(config, sim, monitor_options=monitor,
        video=not args.no_video, reasoning_effort="high", stop_after_steps=None,
        smoke=False, smoke_steps=None, planning_video=False)
    inventory["scripts/evaluate_foundation_harness.py"] = sha256(Path(__file__).read_bytes()).hexdigest()
    inventory["@foundation_harness"] = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    for name, digest in inventory.items():
        if name.startswith("@"):
            continue
        source, destination = ROOT / name, root / "source_at_launch" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during freeze: {name}")
    for key in ("initial_state_override", "task_bddl_override"):
        source = getattr(sim, key, None)
        if source:
            path = Path(source)
            dest = root / "inputs" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            inventory[f"@input:{source}"] = sha256(dest.read_bytes()).hexdigest()
    _write(root / "source_manifest.json", inventory)
    _write(root / "configuration.json", settings)
    _write(root / "dependencies.json", {dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    _write(root / "authorization.json", {"explicit_run_approved_flag": True,
        "scope": "one fresh full task only", "automatic_retry": False,
        "user_approval_required_before_command": True, "argv": sys.argv})
    return inventory


def _preload(args):
    from gpt_field.foundation_vision.backends import Sam31Segmenter, DA3DepthEstimator
    from gpt_field.foundation_vision.engine import EngineConfig, VisionEngine
    backends = {
        "sam": Sam31Segmenter(checkpoint_path=args.sam_checkpoint, device=args.vision_device, local_files_only=True),
        "da3": DA3DepthEstimator(model_id=str(args.da3_model), device=args.vision_device, local_files_only=True)}
    times = {}
    for name, backend in backends.items():
        started = perf_counter()
        backend._load()
        times[name+"_load_s"] = perf_counter()-started
    return VisionEngine(backends["sam"], backends["da3"], EngineConfig(max_age_s=args.vision_max_age_s)), {
        "load_times": times, "models": {name: backend.provenance() for name, backend in backends.items()}}


def run(args, runner, config, sim, monitor, task, settings):
    if not args.approved:
        raise ValueError("run requires --approved after explicit user approval of this full task")
    if args.output is None:
        raise ValueError("run requires --output for a fresh single episode")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    inventory = _freeze(root, args, runner, config, sim, monitor, settings)
    # Local model load happens only after explicit run authorization, before auth/reset.
    engine, preload = _preload(args)
    _write(root / "model_preload.json", preload)
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    from gpt_field.foundation_harness.client import FoundationHarnessClient
    from gpt_field.foundation_harness.executor import FoundationHarnessPolicy
    from gpt_field.foundation_harness.video import FoundationHarnessVideoRecorder
    from gpt_field.robodojo_direct_sim import PandaKinematics
    from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
    trial = root / "trial"
    kinematics = PandaKinematics.from_xml(robot_xml=args.robot_xml, python_executable=sim.python_executable)
    _write(root / "robot_kinematics.json", kinematics.metadata)
    stop_path = trial / runner.base.OPERATOR_STOP_FILE
    holder = {}

    class StopSafePolicy(FoundationHarnessPolicy):
        def act(self, observation):
            if stop_path.is_file():
                raise KeyboardInterrupt("Operator stop requested before deciding")
            decision = super().act(observation)
            if stop_path.is_file():
                self.withheld_decisions = getattr(self, "withheld_decisions", 0)+1
                self._pending = None
                self._primitive = None
                raise KeyboardInterrupt("Operator stop during inference; decision withheld before dispatch")
            return decision

    def make_client(*positional, **kwargs):
        client = FoundationHarnessClient(*positional, reasoning_effort="high", **kwargs)
        configure_client_transport(client, request_deadline_s=None, stream_progress=True)
        return client

    def make_policy(client):
        policy = StopSafePolicy(client, vision_engine=engine,
                                output_dir=trial / "foundation_harness", kinematics=kinematics)
        holder["policy"] = policy
        return policy

    def planning_snapshot():
        policy = holder.get("policy")
        return None if policy is None else policy.planning_snapshot()

    def recorder(*positional, **kwargs):
        kwargs["episode_id"] = kwargs.get("episode_id", "").replace("AGENT SPEEDUP V2", "FOUNDATION HARNESS")
        return FoundationHarnessVideoRecorder(*positional, planning_provider=planning_snapshot, **kwargs)

    credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
    result = None
    try:
        result = runner.base.execute(trial, config=config, sim_config=sim,
            task_key=task.key, episode_index=args.episode_index, monitor_options=monitor,
            token_provider=credentials.access_token, client_factory=make_client,
            policy_factory=make_policy, sim_factory=runner.user_check_sim_factory(monitor),
            recorder_factory=recorder if not args.no_video else runner.base.NullRecorder,
            expected_initial_observation_sha256=args.expected_initial_observation_sha256,
            video=not args.no_video, smoke=False, stop_after_steps=None,
            allow_initial_state_override=True, allow_task_bddl_override=True)
        _write(root / "result.json", {"version": VERSION, "base_summary": result,
            "withheld_decisions_after_stop": getattr(holder.get("policy"), "withheld_decisions", 0)})
        return result
    finally:
        changed = [name for name, digest in inventory.items() if not name.startswith("@")
                   and (not (ROOT/name).is_file() or sha256((ROOT/name).read_bytes()).hexdigest() != digest)]
        _write(root / "terminal_source_audit.json", {"changed_sources": changed,
            "unchanged": not changed, "source_count": sum(not name.startswith("@") for name in inventory)})


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect" and args.approved:
        parser.error("--approved is meaningful only with run")
    runner = _load("scripts/evaluate_agent_speedup_v2_user_check.py", "_foundation_harness_uc_runner")
    config, sim, monitor, task, settings = _configuration(args, runner)
    if args.command == "inspect":
        print(json.dumps({"inspection_only": True, "network_calls": 0, "model_loads": 0,
            "simulator_resets": 0, "configuration": settings}, indent=2, allow_nan=False))
        return 0
    result = run(args, runner, config, sim, monitor, task, settings)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
