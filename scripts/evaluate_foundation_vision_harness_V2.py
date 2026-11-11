#!/usr/bin/env python3
"""Inspect or explicitly launch one foundation_vision_harness_V2 episode.

Inspect reads local configuration only. Run requires --approved after the user
has approved the concrete full task. There is no smoke, retry or task queue.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "foundation_vision_harness_V2"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser():
    legacy = _load("scripts/evaluate_foundation_harness.py", "_fvh_v2_legacy_options")
    parser = legacy.build_parser()
    parser.description = __doc__
    return parser


def _write(path, value):
    from gpt_field.foundation_vision_harness_V2.runner import _write as write
    write(path, value)


def configuration(args, legacy, runner):
    # Reuse checked sensor/task/auth settings, then explicitly replace the old
    # evaluator-driven stop rule. We never invoke its execute() function.
    config, sim, monitor, task, settings = legacy._configuration(args, runner)
    from gpt_field.action import USER_CHECK_TRANSIT_ACTION_M, TRANSLATION_SCALE_M
    from gpt_field.foundation_vision_harness_V2 import REVISION
    sim = replace(sim, stop_on_success=False)
    expected, source = runner.base.resolve_expected_sha(
        task.key, args.episode_index, args.expected_initial_observation_sha256, injected_sim=False)
    settings.update(version=VERSION, revision=REVISION, mode=VERSION,
        keypose_owner="astra_material_and_orientation_with_local_measured_position",
        planning_scope="committed_v2_fragments_with_short_horizon_body_checks", unknown_space_policy="qualified_external_identity_unresolved_and_disputed_measurements",
        task_success="autonomous finish plus independent raw and strict physical scoring",
        terminal_policy="finish_or_blocked_from_public_evidence", raw_success_terminates_control=False,
        terminal_evaluation="10 unchanged-gripper pose-hold actions; no release or counterfactual",
        control_timing="monotonic online wall time; no video/IPC/logging/perception-wait subtraction",
        video_mode="asynchronous bounded queue; intermediate sampled frames may be dropped",
        expected_initial_observation_sha256=expected, expected_sha_source=source,
        transit_command_cap_m=USER_CHECK_TRANSIT_ACTION_M,
        osc_translation_scale_m=TRANSLATION_SCALE_M,
        simulator=asdict(sim))
    return config, sim, monitor, task, settings


def freeze_sources(directory, args, runner, config, sim, monitor, settings):
    # A scoped Python dependency inventory, not a git workspace or runs scan.
    inventory = runner.runtime_inventory(config, sim, monitor_options=monitor,
        video=not args.no_video, reasoning_effort="high", stop_after_steps=None,
        smoke=False, smoke_steps=None, planning_video=False)
    inventory = {name: digest for name, digest in inventory.items() if not name.startswith("@")}
    for name in ("scripts/evaluate_foundation_vision_harness_V2.py", "scripts/evaluate_foundation_harness.py"):
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


def run(args, legacy, runner, config, sim_config, monitor, task, settings):
    if not args.approved:
        raise ValueError("run requires --approved after explicit user approval of this full task")
    if args.output is None:
        raise ValueError("run requires --output for a fresh single episode")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    launch_started = time.monotonic()
    inventory = freeze_sources(root, args, runner, config, sim_config, monitor, settings)
    engine = None
    result = None
    try:
        with runner.base.termination_as_interrupt():
            # No model loading, authentication or simulator construction occurs
            # on inspect, or before the explicit approved launch branch.
            engine, preload = legacy._preload(args)
            _write(root / "model_preload.json", preload)
            from gpt_field.chatgpt_auth import SubscriptionCredentials
            from gpt_field.foundation_vision_harness_V2.client import FoundationVisionHarnessClient
            from gpt_field.foundation_vision_harness_V2.executor import FoundationVisionHarnessPolicy
            from gpt_field.foundation_vision_harness_V2.recording import AsyncVideoRecorder
            from gpt_field.foundation_vision_harness_V2.runner import execute_episode
            from gpt_field.agent_speedup_v2_user_check_sim import UserCheckSimProcess
            from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
            from gpt_field.robodojo_direct_sim import PandaKinematics

            kinematics = PandaKinematics.from_xml(robot_xml=args.robot_xml,
                                                 python_executable=sim_config.python_executable)
            _write(root / "robot_kinematics.json", kinematics.metadata)
            credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
            trial = root / "trial"

            def make_client(**kwargs):
                client = FoundationVisionHarnessClient(config, token_provider=credentials.access_token,
                                                        reasoning_effort="high", **kwargs)
                configure_client_transport(client, request_deadline_s=None, stream_progress=True)
                return client

            def make_policy(client):
                return FoundationVisionHarnessPolicy(client, vision_engine=engine,
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
        close = getattr(engine, "close", None)
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
    # Fresh V2 entrypoint defaults to the successful H1 transit100 actuator
    # configuration. The existing worker applies its matching OSC scale after
    # reset; this does not alter an already imported old executor or old runner.
    os.environ.setdefault("GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M", "0.10")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect" and args.approved:
        parser.error("--approved is meaningful only with run")
    legacy = _load("scripts/evaluate_foundation_harness.py", "_fvh_v2_legacy_helpers")
    runner = _load("scripts/evaluate_agent_speedup_v2_user_check.py", "_fvh_v2_user_check_helpers")
    config, sim, monitor, task, settings = configuration(args, legacy, runner)
    if args.command == "inspect":
        print(json.dumps({"inspection_only": True, "network_calls": 0, "model_loads": 0,
                          "simulator_resets": 0, "configuration": settings}, indent=2, allow_nan=False))
        return 0
    result = run(args, legacy, runner, config, sim, monitor, task, settings)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("finish", "blocked", "user_stop") else 1


if __name__ == "__main__":
    raise SystemExit(main())
