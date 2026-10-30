#!/usr/bin/env python3
"""Prepare or run one separately authorized, fixed37-action robot diagnostic.

No policy/model, object targeting, episode approval or calibration pass is created.
The parent must explicitly launch this child with the frozen program binding.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.native_config import (ResetOnlyClient, add_native_import_paths,
    build_native_preview_config, install_preview_adapters, validate_native_preview_config)
from robodojo_uc71.native_characterization import (FixedRobotCharacterization, capture_characterization_state,
    content_hash, validate_probe_plan, verify_diagnostic_authorization)
from robodojo_uc71.native_measurements import capture_camera_inventory
from new_host_camera import query_cuda_identity, renderer_table_identity, write_durable_json, normalize_pci
from prepare_native_preview import save_rgb_images, verify_manifests
from provision_native_common import digest, pins

RGB_ACTIONS = (0, 5, 14, 23, 32, 37)


def program_files() -> list[Path]:
    scripts = ("prepare_native_characterization.py", "provision_native_characterization.py",
        "prepare_native_preview.py", "provision_native_preview.py", "new_host_camera.py",
        "provision_new_host_camera.py", "provision_native_common.py",
        "provision_native_dependencies.py", "provision_native_compatibility.py")
    modules = ("native_config.py", "native_robot_inventory.py", "native_measurements.py",
               "native_action_observer.py", "native_characterization.py", "host_runtime.py")
    return [*(BASE / "scripts" / name for name in scripts),
            *(BASE / "src/robodojo_uc71" / name for name in modules)]


def verify_program_binding(path: Path) -> dict:
    value = json.loads(path.read_text())
    if set(value) != {str(p) for p in program_files()} or any(digest(Path(p)) != h for p, h in value.items()):
        raise ValueError("Characterization program binding is incomplete or changed")
    return value


def verify_previous_preview(path: Path, plan: dict) -> dict:
    value = json.loads(path.read_text())
    flags = {"diagnostic_render_passed": True, "clean_shutdown_confirmed": True,
        "shutdown_contract_satisfied": True, "gpu_processes_released": True, "gpu_memory_released": True,
        "bindings_unchanged_after_preview": True, "returncode": 0, "pause_seconds": 300,
        "model_calls": 0, "benchmark_actions": 0, "full_test_authorized": False}
    if (any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in flags.items())
            or value.get("termination_signals") != [] or value.get("remaining_owned_pids") != []
            or value.get("binding") != plan["binding"] or "termination" in value or "cleanup_error" in value):
        raise ValueError("A clean single-GPU zero-action native preview with300s pause is required")
    child = path.parent / "preview"
    result = json.loads((child / "preview_result.json").read_text())
    preparation = child / "preparation.json"
    inventory = child / "robot_inventory.json"
    if (digest(child / "preview_result.json") != value.get("preview_result_sha256")
            or digest(preparation) != value.get("preparation_sha256")
            or plan["source_sha256"].get(str(preparation)) != digest(preparation)
            or plan["source_sha256"].get(str(inventory)) != digest(inventory)
            or result.get("robot_inventory", {}).get("status") != "captured"
            or result["robot_inventory"]["sha256"] != digest(inventory)
            or result.get("camera_inventory", {}).get("status") != "captured"
            or result["camera_inventory"]["sha256"] != digest(child / "camera_inventory.json")
            or value.get("pause_validation", {}).get("passed") is not True
            or digest(child / "pause_validation.json") != value["pause_validation"].get("sha256")):
        raise ValueError("Prior native preview/initial inventory/pause evidence differs from the fixed plan")
    return {"path": str(path), "sha256": digest(path), "initial_inventory_sha256": digest(inventory),
            "preparation_sha256": digest(preparation), "pause_validation_sha256": digest(child / "pause_validation.json")}


def check_authorization(path: Path, plan: dict, binding: dict, *, launch: bool, output_parent: Path) -> dict:
    verify_diagnostic_authorization(path, plan, 1)
    auth = json.loads(path.read_text())
    if (plan["binding"] != binding or auth.get("shutdown_mode") != "stock-fast"
            or type(auth.get("maximum_wall_time_s")) is not int or not 30 <= auth["maximum_wall_time_s"] <= 900
            or normalize_pci(auth.get("pci_bus_id", "")) != normalize_pci(binding["pci_bus_id"])
            or Path(auth.get("attempt_output", "")).resolve() != output_parent.resolve()
            or auth.get("rgb_checkpoint_action_indices") != list(RGB_ACTIONS)):
        raise ValueError("Diagnostic authorization must bind this one output/GPU, deadline and fixed RGB checkpoints")
    if launch and os.environ.get("UC71_NATIVE_CHARACTERIZATION_AUTHORIZED_CHILD") != "1":
        raise ValueError("Only the bounded characterization supervisor may launch this child")
    return {"path": str(path), "sha256": digest(path), "plan_sha256": content_hash(plan),
        "maximum_wall_time_s": auth["maximum_wall_time_s"], "diagnostic_native_actions_authorized": True,
        "maximum_native_actions": 37, "benchmark_attempt_started": False, "full_test_authorized": False,
        "model_calls_authorized": False, "rgb_checkpoint_action_indices": list(RGB_ACTIONS)}


def install_characterization_guards(env):
    active = False
    names = []
    def forbidden(*args, **kwargs):
        raise RuntimeError("Policy/task loops are forbidden in fixed robot characterization")
    for name in ("run_eval", "eval_one_episode", "eval_one_episode_batch"):
        if hasattr(env, name):
            setattr(env, name, forbidden)
            names.append(name)
    for name in ("take_action", "take_action_batch", "step"):
        native = getattr(env, name)
        def guarded(*args, _native=native, **kwargs):
            if not active:
                raise RuntimeError("Native policy actions require the fixed-plan diagnostic dispatch scope")
            return _native(*args, **kwargs)
        setattr(env, name, guarded)
        names.append(name)
    @contextmanager
    def permitted():
        nonlocal active
        if active:
            raise RuntimeError("Nested diagnostic dispatch forbidden")
        active = True
        try:
            yield
        finally:
            active = False
    return names, permitted


def capture_stage(env, counters, output: Path, label: str, *, rgb: bool) -> dict:
    directory = output / label
    directory.mkdir(parents=True, exist_ok=False)
    before = (float(env.sim.sim.current_time), env.sim._sim_step_counter, counters["native_sim_step_calls"], tuple(env.take_action_cnt))
    env.obs_manager.render_for_capture()
    camera = capture_camera_inventory(env)
    robot = capture_characterization_state(env)
    write_durable_json(directory / "camera_inventory.json", camera)
    write_durable_json(directory / "robot_measurement.json", robot)
    images = save_rgb_images(env.obs_manager.get_obs()[0], directory) if rgb else []
    after = (float(env.sim.sim.current_time), env.sim._sim_step_counter, counters["native_sim_step_calls"], tuple(env.take_action_cnt))
    if before != after or camera["native_physics_time_s"] != before[0] or robot["native_physics_time_s"] != before[0]:
        raise RuntimeError("Synchronized diagnostic capture advanced physics")
    for image in images:
        image["path"] = label + "/" + image["path"]
    return {"label": label, "captured_monotonic_s": time.monotonic(), "native_physics_time_s": before[0],
        "native_physics_counter": before[1], "native_sim_step_calls": before[2],
        "native_action_counter": list(before[3]),
        "physics_unchanged_during_capture": True,
        "camera_inventory": {"path": label + "/camera_inventory.json", "sha256": digest(directory / "camera_inventory.json")},
        "robot_measurement": {"path": label + "/robot_measurement.json", "sha256": digest(directory / "robot_measurement.json")},
        "images": images}


def run_characterization(args, config, preparation, binding, output, progress) -> int:
    app = env = monitor = None
    failure = None
    cleanup_errors = []
    result = None
    try:
        identity = query_cuda_identity(binding)
        write_durable_json(output / "cuda_identity.json", identity)
        progress("cuda_identity_verified")
        from src.eval_client.physx_warning_monitor import get_monitor
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args, zero_delay_active
        from isaaclab.app import AppLauncher
        monitor = get_monitor()
        monitor.start(enabled=True)
        kit_args = ["--portable", f"--portable-root={output / 'kit-portable'}",
                    f"--/log/file={output / 'kit.log'}", "--/app/settings/persistent=false",
                    "--/app/telemetry/enable=false", "--/telemetry/enableAnonymousData=false",
                    "--/telemetry/enableAnonymousAppName=false", "--/rtx/multiGpu/enabled=false",
                    "--/rtx/multiGpu/maxGpuCount=1"]
        launcher_args = argparse.Namespace(headless=True, enable_cameras=True, livestream=0,
            device="cpu", multi_gpu=False, width=640, height=480,
            fast_shutdown=args.shutdown_mode == "stock-fast", enable_crashreporter=False,
            kit_args=" ".join(kit_args))
        add_zero_delay_kit_args(launcher_args)
        write_durable_json(output / "app_configuration.json", vars(launcher_args))
        progress("before_native_app_startup")
        app = AppLauncher(launcher_args).app
        progress("native_app_started")
        import carb.settings
        settings = carb.settings.get_settings()
        observed = {key: settings.get(key) for key in
                    ("/renderer/activeGpu", "/physics/cudaDevice", "/renderer/multiGpu/enabled")}
        write_durable_json(output / "observed_gpu_settings.json", observed)
        if (observed["/renderer/activeGpu"] != binding["kit_index"]
                or observed["/physics/cudaDevice"] != binding["cuda_ordinal"]
                or observed["/renderer/multiGpu/enabled"] is not False):
            raise RuntimeError("Native app selected unexpected GPU settings")
        renderer = renderer_table_identity(args.parent_log.read_text(errors="replace"), binding)
        write_durable_json(output / "renderer_identity.json", renderer)
        if not renderer["uuid_prefix_and_pci_bus_match"] or not zero_delay_active():
            raise RuntimeError("Native preview GPU identity or zero-delay camera settings did not verify")
        adaptations = install_preview_adapters()
        from src.eval_client.eval_env import create_eval_env
        env = create_eval_env(config, app, resume_state=None)
        if type(env.model_client) is not ResetOnlyClient or env.robot_manager.planner or env.robot_manager.ik_solver:
            raise RuntimeError("Characterization transport/planner exclusion failed")
        guarded_methods, permitted = install_characterization_guards(env)
        selected = Path(env.seed_manager.seed_info[0]["scene_layout"]).resolve()
        if str(selected) != preparation["layout_path"] or digest(selected) != preparation["layout_sha256"]:
            raise RuntimeError("Native numeric layout 0 differs from the pinned selected layout")
        counters = {"native_sim_step_calls": 0, "native_render_calls": 0}
        original_step, original_render = env.sim_step, env.render
        def counted_step(*args, **kwargs):
            counters["native_sim_step_calls"] += 1
            return original_step(*args, **kwargs)
        def counted_render(*args, **kwargs):
            counters["native_render_calls"] += 1
            return original_render(*args, **kwargs)
        env.sim_step, env.render = counted_step, counted_render
        progress("before_native_layout0_reset")
        env.reset(seed=[0])
        progress("native_layout0_reset_returned", **counters)
        if (env.device != "cpu" or env.use_fabric is not False or env.num_envs != 1
                or list(env.env_seeds) != [0] or env.model_client.reset_calls != 1
                or env.take_action_cnt != [0] or env.robot_manager.planner or env.robot_manager.ik_solver):
            raise RuntimeError("Native reset violated the preview scope/configuration")
        # Register exactly the native predicates; never infer success from reset's success=True flag.
        env.run_reward()
        env.get_score()
        env._check_physx_broken_envs()
        if monitor.is_fatal() or monitor.get_broken_envs():
            raise RuntimeError("Native PhysX monitor reported a failure")
        module_provenance = {}
        for name in ("numpy", "torch", "omegaconf", "gymnasium", "shapely", "transforms3d",
                     "h5py", "isaaclab", "isaaclab_tasks", "isaacsim", "websockets",
                     "hid", "setuptools", "pkg_resources", "_distutils_hack"):
            module = sys.modules.get(name)
            if module is not None:
                module_provenance[name] = {"file": str(getattr(module, "__file__", None)),
                                           "version": str(getattr(module, "__version__", None))}
        write_durable_json(output / "loaded_module_provenance.json", module_provenance)
        plan = json.loads(args.plan.read_text())
        captures = [capture_stage(env, counters, output, "before", rgb=True)]
        initial = captures[0]
        actions = []
        program_binding = preparation["source_file_sha256"]
        def verify_runtime_binding(current_plan):
            validate_native_preview_config(config)
            return (content_hash(current_plan) == preparation["plan_sha256"]
                and digest(args.plan) == preparation["plan_file_sha256"]
                and digest(args.authorization) == preparation["authorization"]["sha256"]
                and all(digest(Path(path)) == expected for path, expected in program_binding.items())
                and type(env.model_client) is ResetOnlyClient and env.model_client.reset_calls == 1
                and not env.model_client.closed and not env.robot_manager.planner and not env.robot_manager.ik_solver
                and env.device == "cpu" and env.use_fabric is False and env.num_envs == 1
                and list(env.env_seeds) == [0] and str(selected) == current_plan["layout_path"]
                and digest(selected) == current_plan["layout_sha256"]
                and not monitor.is_fatal() and not monitor.get_broken_envs())
        probe = FixedRobotCharacterization(env, plan, authorization_path=args.authorization,
                                          verify_runtime_binding=verify_runtime_binding)
        previous_capture = initial
        for index in range(1, 38):
            progress("before_fixed_native_probe", diagnostic_native_action_index=index)
            with permitted():
                receipt = probe.dispatch_next()
            path = output / f"native_action_{index:03d}.json"
            write_durable_json(path, receipt)
            # Every action gets synchronized robot/camera metadata. Only fixed
            # predeclared checkpoints get RGB; observations never select targets.
            capture = capture_stage(env, counters, output, f"action_{index:03d}", rgb=index in RGB_ACTIONS)
            if (receipt["diagnostic_native_action_index"] != index
                    or receipt["before"]["native_physics_time_s"] != previous_capture["native_physics_time_s"]
                    or receipt["after"]["native_physics_time_s"] != capture["native_physics_time_s"]
                    or receipt["native_accounting"]["native_counter_after"] != index
                    or env.take_action_cnt != [index]):
                raise RuntimeError("Diagnostic receipt and synchronized capture disagree")
            actions.append({"index": index, "receipt": {"path": path.name, "sha256": digest(path)},
                            "before_capture": previous_capture["label"], "after_capture": capture})
            if capture["images"]:
                captures.append(capture)
            previous_capture = capture
            env._check_physx_broken_envs()
            if monitor.is_fatal() or monitor.get_broken_envs():
                raise RuntimeError("PhysX monitor reported a diagnostic failure")
            progress("fixed_native_probe_recorded", diagnostic_native_action_index=index,
                     native_physics_time_s=capture["native_physics_time_s"])
            if receipt["stop_after_this_action"] and index != 37:
                raise RuntimeError("Native diagnostic ended before the fixed37-action plan completed")
        simulated = previous_capture["native_physics_time_s"] - initial["native_physics_time_s"]
        if (env.take_action_cnt != [37] or len(actions) != 37
                or previous_capture["native_physics_counter"] - initial["native_physics_counter"] != 370
                or counters["native_sim_step_calls"] - initial["native_sim_step_calls"] != 370
                or not math.isclose(simulated, 1.48, abs_tol=1e-7, rel_tol=0.)
                or len(captures) != 6 or env.model_client.reset_calls != 1):
            raise RuntimeError("Fixed diagnostic action/substep/time scope mismatch")
        result = {"schema": "robodojo-native-characterization-result-1", "diagnostic_completed": True,
            "completed_at_monotonic_s": time.monotonic(), "binding": binding, "shutdown_mode": "stock-fast",
            "plan_sha256": preparation["plan_sha256"], "plan_file_sha256": preparation["plan_file_sha256"],
            "authorization_sha256": preparation["authorization"]["sha256"],
            "preparation_sha256": digest(output / "preparation.json"),
            "loaded_module_provenance_sha256": digest(output / "loaded_module_provenance.json"),
            "layout_path": str(selected), "layout_sha256": digest(selected),
            "captures": captures, "actions": actions, "diagnostic_native_actions": 37,
            "diagnostic_physics_substeps": 370, "diagnostic_simulated_time_s": simulated,
            "rgb_checkpoint_action_indices": list(RGB_ACTIONS), "image_count": 18,
            "native_take_action_count": list(env.take_action_cnt), "reset_calls": 1,
            "native_client_reset_calls": 1, "native_setup_and_diagnostic_counts": counters,
            "model_calls": 0, "benchmark_actions": 0, "benchmark_attempt_started": False,
            "full_test_authorized": False, "task_success_assessed": False,
            "standard_ready": False, "calibration_passed": False, "asset_fidelity_passed": False,
            "dependency_closure_complete": preparation["dependency_closure_complete"],
            "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
            "policy_action_methods_guarded": guarded_methods, "adaptations": adaptations,
            "physics_device": env.device, "use_fabric": env.use_fabric,
            "native_planner_instances": 0, "native_ik_solver_instances": 0,
            "reward_predicates_registered": True,
            "limitations": "Fixed transient joint probes only; no settled tracking, TCP/frame calibration pass, object targeting or task outcome classification."}
        write_durable_json(output / "characterization_result.json", result)
        progress("fixed_characterization_saved", diagnostic_native_actions=37, images=18)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_durable_json(output / "characterization_failure.json", failure)
        progress("characterization_failed", error=failure)
    finally:
        for phase, operation in (("native_env_close", env.close if env is not None else None),
                                 ("physx_monitor_shutdown", monitor.shutdown if monitor is not None else None)):
            if operation is not None:
                try:
                    operation()
                    progress(phase + "_returned")
                except BaseException as error:
                    cleanup_errors.append({"phase": phase, "type": type(error).__name__, "message": str(error)})
        no_pending_exception = failure is None and sys.exc_info()[0] is None
        healthy = monitor is not None and not monitor.is_fatal() and not monitor.get_broken_envs()
        ready = bool(result and no_pending_exception and not cleanup_errors and healthy)
        evidence = {"schema": "robodojo-native-characterization-preclose-1", "ready_for_shutdown": ready,
                    "diagnostic_native_actions": env.take_action_cnt[0] if env is not None else 0,
                    "benchmark_attempt_started": False, "full_test_authorized": False,
                    "standard_ready": False, "asset_fidelity_passed": False,
                    "dependency_closure_complete": preparation["dependency_closure_complete"],
                    "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
                    "monotonic_s": time.monotonic(), "child_pid": os.getpid(), "binding": binding,
                    "shutdown_mode": args.shutdown_mode, "no_pending_exception": no_pending_exception,
                    "native_env_close_returned": env is not None and not any(e["phase"] == "native_env_close" for e in cleanup_errors),
                    "physx_monitor_healthy": healthy, "cleanup_errors": cleanup_errors,
                    "characterization_result_sha256": digest(output / "characterization_result.json") if result else None,
                    "preparation_sha256": digest(output / "preparation.json"),
                    "image_count": sum(len(x["images"]) for x in result["captures"]) if result else 0,
                    "model_calls": 0, "benchmark_actions": 0, "full_plugin_cleanup_proven": False,
                    "full_teardown_returned": False}
        write_durable_json(output / "characterization_preclose_ready.json", evidence)
        progress("before_simulation_app_close", ready_for_shutdown=ready)
        # Fast shutdown may exit here; parent must require ready + hashes + clean exit/resources.
        if app is not None:
            try:
                app.close()
                write_durable_json(output / "characterization_close_returned.json", {
                    "app_close_returned": True, "monotonic_s": time.monotonic(),
                    "shutdown_mode": args.shutdown_mode, "full_teardown_returned": args.shutdown_mode == "full"})
            except BaseException as error:
                cleanup_errors.append({"phase": "simulation_app_close", "type": type(error).__name__, "message": str(error)})
                write_durable_json(output / "characterization_close_failure.json", cleanup_errors[-1])
    return 0 if ready and not cleanup_errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "source-manifest", "asset-manifest", "authorization", "output",
                 "plan", "preview-result", "program-binding", "parent-log", "asset-diagnostic-assessment"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--kit-index", type=int, choices=(0,), required=True)
    parser.add_argument("--cuda-ordinal", type=int, choices=(0,), required=True)
    parser.add_argument("--shutdown-mode", choices=("stock-fast",), default="stock-fast")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        parser.error("Characterization requires the reviewed single unmasked GPU")
    for key in ("native_root", "source_manifest", "asset_manifest", "authorization", "plan",
                "preview_result", "program_binding", "asset_diagnostic_assessment", "parent_log"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    output = args.output.resolve()
    if output.exists() or any(c.isspace() for c in str(output)) or not args.parent_log.is_relative_to(output.parent):
        parser.error("Use a fresh child output and the parent's opened private log")
    binding = {"gpu_uuid": args.gpu_uuid, "pci_bus_id": args.pci_bus_id, "kit_index": 0, "cuda_ordinal": 0}
    plan = json.loads(args.plan.read_text())
    validate_probe_plan(plan)
    programs = verify_program_binding(args.program_binding)
    auth = check_authorization(args.authorization, plan, binding, launch=args.launch, output_parent=output.parent)
    prior = verify_previous_preview(args.preview_result, plan)
    preparation = verify_manifests(args.native_root, args.source_manifest, args.asset_manifest,
                                   pins(), args.asset_diagnostic_assessment)
    if preparation["camera_gate"]["gpu_uuid"] != binding["gpu_uuid"]:
        raise ValueError("Characterization GPU differs from reviewed rendering gate")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    os.environ["ROBODOJO_RUN_ID"] = output.name
    preparation["import_paths"] = add_native_import_paths(args.native_root)
    config = build_native_preview_config(args.native_root, output.name)
    from omegaconf import OmegaConf
    write_durable_json(output / "native_configuration.json", OmegaConf.to_container(config, resolve=True))
    preparation.update(schema="robodojo-native-characterization-preparation-1", authorization=auth,
        native_root=str(args.native_root), plan_path=str(args.plan), plan_sha256=content_hash(plan),
        plan_file_sha256=digest(args.plan), prior_preview=prior, source_file_sha256=programs,
        native_configuration_sha256=digest(output / "native_configuration.json"),
        binding=binding, shutdown_mode="stock-fast", launch_requested=args.launch,
        diagnostic_native_actions_planned=37, rgb_checkpoint_action_indices=list(RGB_ACTIONS),
        benchmark_attempt_started=False, full_test_authorized=False, model_calls=0, benchmark_actions=0)
    write_durable_json(output / "preparation.json", preparation)
    if not args.launch:
        print(json.dumps({"prepared": True, "simulator_starts": 0, "output": str(output)}), flush=True)
        return 0
    def progress(phase, **details):
        row = {"phase": phase, "monotonic_s": time.monotonic(), **details}
        with (output / "characterization_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)
    return run_characterization(args, config, preparation, binding, output, progress)


if __name__ == "__main__":
    raise SystemExit(main())
