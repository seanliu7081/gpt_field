#!/usr/bin/env python3
"""Inspect or explicitly launch one foundation_vision_harness_V3 episode.

Vision is an optional on-demand tool; neither inspect nor launch preloads it.
Only run --approved starts a simulator and subscription-backed policy. One fresh
episode is run without automatic retries or episode time/action/request caps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "foundation_vision_harness_V3"
DEFAULT_SAM = ROOT / "artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt"
DEFAULT_DA3 = ROOT / "artifacts/foundation_vision/models/models--depth-anything--DA3-LARGE-1.1/snapshots/0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("inspect", "run"), nargs="?", default="inspect")
    parser.add_argument("--approved", action="store_true", help="Use only after explicit approval of this full task")
    parser.add_argument("--output", type=Path, help="Fresh run root; episode artifacts go in trial/")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--vision", choices=("on-demand", "off"), default="on-demand")
    parser.add_argument("--sam-checkpoint", type=Path, help="Optional local SAM weights; loaded only if a tool request needs them")
    parser.add_argument("--da3-model", type=Path, help="Optional local DA3 directory; loaded only if a tool request needs it")
    parser.add_argument("--vision-device", default="cuda:0")
    parser.add_argument("--vision-max-age-s", type=float, default=60.,
                        help="Maximum age of the immutable RGB source used by an optional vision tool")
    parser.add_argument("--sim-python")
    parser.add_argument("--libero-root")
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--expected-initial-observation-sha256")
    parser.add_argument("--initial-state-override", type=Path)
    parser.add_argument("--task-bddl-override", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--no-video", action="store_true")
    return parser


def _write(path, value):
    from gpt_field.foundation_vision_harness_V3.runner import _write as write
    write(path, value)


def _model_paths(args):
    # Read-only discovery. Missing optional weights do not prevent ordinary
    # RGB-based manipulation or vision=off; a requested tool reports readiness.
    sam = args.sam_checkpoint
    da3 = args.da3_model
    if sam is None and DEFAULT_SAM.is_file():
        sam = DEFAULT_SAM
    if da3 is None and (DEFAULT_DA3 / "config.json").is_file() and (DEFAULT_DA3 / "model.safetensors").is_file():
        da3 = DEFAULT_DA3
    args.sam_checkpoint = None if sam is None else sam.expanduser().resolve()
    args.da3_model = None if da3 is None else da3.expanduser().resolve()
    return {
        "sam_checkpoint": None if args.sam_checkpoint is None else str(args.sam_checkpoint),
        "da3_model": None if args.da3_model is None else str(args.da3_model),
        "sam_local_file_present": args.sam_checkpoint is not None and args.sam_checkpoint.is_file(),
        "da3_local_files_present": args.da3_model is not None and all(
            (args.da3_model / name).is_file() for name in ("config.json", "model.safetensors")),
    }


def configuration(args, runner):
    """Resolve existing task/sensor/auth settings without visual backend loads."""
    if not math.isfinite(args.vision_max_age_s) or args.vision_max_age_s <= 0:
        raise ValueError("--vision-max-age-s must be positive and finite")
    base = runner.base
    task = base.resolve_v2_task(args.task)
    base._validate_episode_index(args.episode_index)
    sim, monitor = base.build_sim_config(task.key,
        sim_python=args.sim_python or base.DEFAULT_SIM_PYTHON,
        libero_root=args.libero_root or base.DEFAULT_LIBERO_ROOT)
    for option in ("initial_state_override", "task_bddl_override"):
        value = getattr(args, option)
        if value is not None:
            path = value.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"missing --{option.replace('_', '-')} input")
            sim = replace(sim, **{option: str(path)})
    runner.require_custom_reset_sha(sim, args.expected_initial_observation_sha256)
    config = base.configuration()
    base.validate_configuration(config, sim, task, allow_initial_state_override=True,
                                allow_task_bddl_override=True)
    if any(getattr(config, name, None) is not None for name in ("max_api_calls", "max_wall_time_s")):
        raise ValueError("full task requires uncapped subscription configuration")
    if monitor.get("oracle", False) or getattr(sim, "stop_on_collision", True):
        raise ValueError("public sensors and record-only evaluator collisions are required")
    sim = replace(sim, stop_on_success=False)
    expected, source = base.resolve_expected_sha(
        task.key, args.episode_index, args.expected_initial_observation_sha256, injected_sim=False)
    from gpt_field.action import USER_CHECK_TRANSIT_ACTION_M, TRANSLATION_SCALE_M
    from gpt_field.foundation_vision_harness_V3 import REVISION
    model_paths = _model_paths(args)
    settings = {
        "version": VERSION, "revision": REVISION, "mode": VERSION,
        "task": task.key, "task_prompt": task.prompt, "episode_index": args.episode_index,
        "model": config.model, "reasoning_effort": "high", "billing": "chatgpt_subscription",
        "api_key_fallback": False, "automatic_episode_retry": False,
        "episode_action_cap": None, "episode_time_cap_s": None, "episode_model_call_cap": None,
        "keypose_owner": "astra", "environment_obstacles": "astra_explicit_approximate_boxes",
        "robot_guard": "known_robot_joint_limits_self_collision_independent_of_vision",
        "vision": {"mode": args.vision, "device": args.vision_device,
                   "max_age_s": args.vision_max_age_s, **model_paths,
                   "preloaded": False, "automatic_per_step_perception": False,
                   "missing_weights_do_not_block_ordinary_actions": True},
        "robot_xml": None if args.robot_xml is None else str(args.robot_xml.expanduser().resolve()),
        "expected_initial_observation_sha256": expected, "expected_sha_source": source,
        "video": not args.no_video, "simulator": asdict(sim), "subscription": asdict(config),
        "collision_evaluator": "record_only",
        "task_success": "autonomous finish plus independent raw and strict physical scoring",
        "terminal_policy": "finish_or_blocked_from_public_evidence", "raw_success_terminates_control": False,
        "terminal_evaluation": "10 unchanged-gripper pose-hold actions; no release or counterfactual",
        "control_timing": "monotonic online wall time; no video/IPC/logging/tool-wait subtraction",
        "video_mode": "asynchronous bounded queue; intermediate sampled frames may be dropped",
        "transit_command_cap_m": USER_CHECK_TRANSIT_ACTION_M, "osc_translation_scale_m": TRANSLATION_SCALE_M,
    }
    return config, sim, monitor, task, settings


def freeze_sources(directory, args, runner, config, sim, monitor, settings):
    inventory = runner.runtime_inventory(config, sim, monitor_options=monitor,
        video=not args.no_video, reasoning_effort="high", stop_after_steps=None,
        smoke=False, smoke_steps=None, planning_video=False)
    inventory = {name: digest for name, digest in inventory.items() if not name.startswith("@")}
    name = "scripts/evaluate_foundation_vision_harness_V3.py"
    inventory[name] = sha256((ROOT / name).read_bytes()).hexdigest()
    inventory["@configuration"] = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    for name, digest in inventory.items():
        if name.startswith("@"):
            continue
        destination = directory / "source_at_launch" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source changed during freeze: {name}")
    for name in ("initial_state_override", "task_bddl_override"):
        source = getattr(sim, name, None)
        if source:
            path = Path(source)
            destination = directory / "inputs" / (name+"_"+path.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            inventory["@input:"+source] = sha256(destination.read_bytes()).hexdigest()
    _write(directory / "source_manifest.json", inventory)
    _write(directory / "configuration.json", settings)
    _write(directory / "authorization.json", {"explicit_run_approved_flag": True,
        "user_approval_required_before_command": True, "scope": "one fresh full task",
        "automatic_retry": False, "argv": sys.argv})
    _write(directory / "dependencies.json", {dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions() if dist.metadata.get("Name")})
    return inventory


def run(args, runner, config, sim_config, monitor, task, settings):
    if not args.approved:
        raise ValueError("run requires --approved after explicit user approval of this full task")
    if args.output is None:
        raise ValueError("run requires --output for a fresh single episode")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    launch_started = time.monotonic()
    inventory = freeze_sources(root, args, runner, config, sim_config, monitor, settings)
    vision_tools = None
    result = None
    try:
        with runner.base.termination_as_interrupt():
            from gpt_field.chatgpt_auth import SubscriptionCredentials
            from gpt_field.foundation_vision_harness_V3.client import FoundationVisionHarnessClient
            from gpt_field.foundation_vision_harness_V3.executor import FoundationVisionHarnessPolicy
            from gpt_field.foundation_vision_harness_V3.vision_tools import OnDemandVisionTools
            from gpt_field.foundation_vision_harness_V3.recording import AsyncVideoRecorder
            from gpt_field.foundation_vision_harness_V3.runner import execute_episode
            from gpt_field.agent_speedup_v2_user_check_sim import UserCheckSimProcess
            from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
            from gpt_field.robodojo_direct_sim import PandaKinematics

            kinematics = PandaKinematics.from_xml(robot_xml=args.robot_xml,
                                                 python_executable=sim_config.python_executable)
            _write(root / "robot_kinematics.json", kinematics.metadata)
            trial = root / "trial"
            vision_tools = OnDemandVisionTools(mode=args.vision, sam_checkpoint=args.sam_checkpoint,
                da3_model=args.da3_model, device=args.vision_device, max_age_s=args.vision_max_age_s,
                output_dir=trial / VERSION / "vision_tools")
            _write(root / "vision_configuration.json", settings["vision"])
            credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)

            def make_client(**kwargs):
                client = FoundationVisionHarnessClient(config, token_provider=credentials.access_token,
                                                        reasoning_effort="high", **kwargs)
                configure_client_transport(client, request_deadline_s=None, stream_progress=True)
                return client

            def make_policy(client):
                return FoundationVisionHarnessPolicy(client, vision_tools=vision_tools,
                    output_dir=trial / VERSION, kinematics=kinematics)

            episode_started = time.monotonic()
            result = execute_episode(trial, task=task, episode_index=args.episode_index,
                sim_factory=lambda: UserCheckSimProcess(sim_config, monitor_options=monitor),
                client_factory=make_client, policy_factory=make_policy,
                expected_initial_observation_sha256=settings["expected_initial_observation_sha256"],
                expected_sha_source=settings["expected_sha_source"],
                recorder_factory=None if args.no_video else AsyncVideoRecorder,
                simulator_start_context=runner.base._simulator_without_credentials)
            result["pre_episode_startup_wall_s"] = episode_started-launch_started
            result["episode_startup_wall_s"] = result["startup_wall_s"]
            result["episode_total_wall_s"] = result["total_wall_s"]
            result["startup_wall_s"] += result["pre_episode_startup_wall_s"]
    except BaseException as error:
        _write(root / "launch_error.json", {"type": type(error).__name__, "error": str(error),
                                            "automatic_retry": False})
        raise
    finally:
        close = getattr(vision_tools, "close", None)
        if callable(close):
            close()
        changed = [name for name, digest in inventory.items() if not name.startswith("@")
            and (not (ROOT / name).is_file() or sha256((ROOT / name).read_bytes()).hexdigest() != digest)]
        _write(root / "terminal_source_audit.json", {"changed_sources": changed,
            "unchanged": not changed, "source_count": sum(not n.startswith("@") for n in inventory)})
        if result is not None:
            result["launch_total_wall_s"] = time.monotonic()-launch_started
            result["total_wall_s"] = result["launch_total_wall_s"]
            result["postprocessing_wall_s"] = max(0., result["total_wall_s"]-
                result["startup_wall_s"]-(result["online_wall_s"] or 0.))
            result["source_unchanged"] = not changed
            _write(root / "result.json", result)
    return result


def main(argv=None):
    # Set the successful v2 actuator scale before importing executor helpers.
    os.environ.setdefault("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M", "0.10")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect" and args.approved:
        parser.error("--approved is meaningful only with run")
    runner = _load("scripts/evaluate_agent_speedup_v2_user_check.py", "_fvh_v3_user_check_helpers")
    config, sim, monitor, task, settings = configuration(args, runner)
    if args.command == "inspect":
        print(json.dumps({"inspection_only": True, "network_calls": 0, "model_loads": 0,
                          "simulator_resets": 0, "configuration": settings}, indent=2, allow_nan=False))
        return 0
    result = run(args, runner, config, sim, monitor, task, settings)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("finish", "blocked", "user_stop") else 1


if __name__ == "__main__":
    raise SystemExit(main())
