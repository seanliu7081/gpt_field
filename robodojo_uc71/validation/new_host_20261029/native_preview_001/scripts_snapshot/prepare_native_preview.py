#!/usr/bin/env python3
"""Prepare, or explicitly launch once, a supervised model-free native preview.

The parent owns timeout/process cleanup and GPU measurement. This child never
runs an evaluator policy loop, sends observations, or executes benchmark actions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.native_config import (EXPECTED_CAMERAS, ResetOnlyClient, add_native_import_paths,
    build_native_preview_config, forbid_policy_actions, install_preview_adapters)
from new_host_camera import query_cuda_identity, renderer_table_identity, write_durable_json
from provision_native_common import (PINS, asset_path, digest, pins, require_camera_gate,
    require_native_checkout)


REVIEWED_INCOMPLETE_ASSET_MANIFEST_SHA256 = "d031d8bd55746879907ce36e2e7fe8e55d2864ceac1848a061e2a681bc870419"
MISSING_STAND_TEXTURE = "/home/kaslensu/Desktop/textures/Image_0.png"
STAND_ASSET = "Assets/Object/RoboDojo/Geometry/camera_stand/00000/object.usd"
STAND_ASSET_SHA256 = "6c4b9ec4cd325f6e47d862b87e7dddbe9adc5e9a5aa0f04224c47d0f16a600a8"


def verify_asset_diagnostic_assessment(native: Path, assets_path: Path, assets: dict,
                                       assessment_path: Path) -> dict:
    """One enumerated upstream-asset defect may be inspected, never marked READY."""
    if digest(assets_path) != REVIEWED_INCOMPLETE_ASSET_MANIFEST_SHA256:
        raise ValueError("Only the exact reviewed native_assets_002 manifest permits a diagnostic assessment")
    assessment_path = assessment_path.resolve(strict=True)
    assessment = json.loads(assessment_path.read_text())
    flags = {"schema": "robodojo-native-asset-diagnostic-assessment-1", "diagnostic_authorized": True,
             "standard_ready": False, "asset_fidelity_passed": False, "dependency_closure_complete": False,
             "model_calls_authorized": False, "full_test_authorized": False}
    if any(type(assessment.get(k)) is not type(v) or assessment.get(k) != v for k, v in flags.items()):
        raise ValueError("Asset assessment must authorize only an explicitly incomplete-fidelity diagnostic")
    raw = assessment["raw_asset_manifest"]
    if (Path(raw["path"]).resolve() != assets_path or raw["sha256"] != digest(assets_path)
            or assets.get("complete") is not False or assets.get("scan_errors")
            or assessment.get("permitted_unresolved") != assets["unresolved"] or len(assets["unresolved"]) != 15
            or digest(native / STAND_ASSET) != STAND_ASSET_SHA256):
        raise ValueError("Diagnostic asset exception differs from the exact immutable source defect")
    artifacts = assessment.get("bound_artifacts", [])
    if not artifacts:
        raise ValueError("Diagnostic assessment must hash-bind classification evidence and its resources")
    by_path = {}
    for entry in artifacts:
        path = Path(entry["path"]).resolve(strict=True)
        if (not path.is_relative_to(BASE) or str(path) in by_path
                or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]):
            raise ValueError(f"Diagnostic classification artifact changed: {path}")
        by_path[str(path)] = entry["sha256"]
    evidence = {}
    for key in ("classification_evidence", "static_missing_texture_evidence"):
        spec = assessment[key]
        path = Path(spec["path"]).resolve(strict=True)
        if by_path.get(str(path)) != spec["sha256"]:
            raise ValueError("Assessment evidence is not in the verified artifact closure")
        evidence[key] = json.loads(path.read_text())
    material = evidence["classification_evidence"]
    static = evidence["static_missing_texture_evidence"]
    rows = material.get("unresolved_classification", [])
    if (material.get("schema") != "robodojo-native-material-audit-1"
            or material.get("classification_complete") is not True or material.get("fetch_errors")
            or material.get("source_manifest_unchanged") is not True
            or material.get("source_manifest", {}).get("sha256") != digest(assets_path)
            or material.get("authored_assets_modified") is not False
            or len(rows) != 15
            or sorted(json.dumps(row["raw_reference"], sort_keys=True) for row in rows)
                != sorted(json.dumps(row, sort_keys=True) for row in assets["unresolved"])):
        raise ValueError("Material audit must classify every exact raw unresolved reference")
    engines, remote = material.get("engine_modules", []), material.get("remote_files", [])
    if len(engines) != 3 or len(remote) != 9:
        raise ValueError("Unexpected engine/remote material evidence set")
    for resource in engines + remote:
        path = Path(resource["path"]).resolve(strict=True)
        if by_path.get(str(path)) != resource["sha256"]:
            raise ValueError("A classified material/texture is absent from the bound artifact closure")
    for engine in engines:
        if not engine.get("owners"):
            raise ValueError("Engine material lacks wheel ownership evidence")
        for owner in engine["owners"]:
            if (owner.get("distribution") != "isaacsim-kernel" or owner.get("version") != "5.1.0.0"
                    or owner.get("record_verified") is not True
                    or by_path.get(str(Path(owner["record"]).resolve())) != owner["record_sha256"]):
                raise ValueError("Engine material wheel provenance changed")
    remote_urls = {row["url"] for row in remote}
    for resource in remote:
        if resource.get("http_status") != 200 or resource.get("final_url") != resource["url"]:
            raise ValueError("Remote material evidence is incomplete or redirected")
        if any(ref["url"] not in remote_urls for ref in resource.get("active_texture_2d_references", [])):
            raise ValueError("An active remote material texture lacks fetched evidence")
    for row in rows:
        kind, index = row.get("classification"), row.get("evidence_index")
        if kind == "engine_search_module":
            if type(index) is not int or not 0 <= index < len(engines) or engines[index]["authored"] != row["raw_reference"]["authored"]:
                raise ValueError("Engine classification points to another material")
        elif kind == "remote_material":
            if type(index) is not int or not 0 <= index < len(remote) or remote[index]["url"] != row["raw_reference"]["authored"]:
                raise ValueError("Remote classification points to another material")
    counts = {kind: sum(row.get("classification") == kind for row in rows) for kind in
              ("engine_search_module", "remote_material", "unresolved_external_asset")}
    missing = [row["raw_reference"] for row in rows if row.get("classification") == "unresolved_external_asset"]
    if (counts != {"engine_search_module": 11, "remote_material": 3, "unresolved_external_asset": 1}
            or missing != [{"authored": MISSING_STAND_TEXTURE, "candidate": None,
                           "from": STAND_ASSET, "package_member": None}]
            or static.get("schema") != "robodojo-native-camera-stand-texture-audit-1"
            or static.get("asset", {}).get("sha256") != STAND_ASSET_SHA256
            or static.get("asset_manifest", {}).get("sha256") != digest(assets_path)
            or static.get("missing_reference") != MISSING_STAND_TEXTURE
            or static.get("native_loader_remaps_missing_reference") is not False
            or not static.get("checks") or any(v is not True for v in static["checks"].values())):
        raise ValueError("Only the documented live upstream camera-stand normal-map defect is allowed")
    return {"path": str(assessment_path), "sha256": digest(assessment_path),
            "raw_asset_manifest_sha256": digest(assets_path),
            "classification_evidence_sha256": assessment["classification_evidence"]["sha256"],
            "static_missing_texture_evidence_sha256": assessment["static_missing_texture_evidence"]["sha256"],
            "permitted_unresolved_count": 15, "known_missing_reference": MISSING_STAND_TEXTURE,
            "source_assets_unchanged": True, "standard_ready": False,
            "asset_fidelity_passed": False, "dependency_closure_complete": False}


def verify_manifests(native: Path, source_path: Path, assets_path: Path, pin_data: dict,
                     asset_diagnostic_assessment: Path | None = None) -> dict:
    source = json.loads(source_path.read_text())
    assets = json.loads(assets_path.read_text())
    if (source.get("schema") != "robodojo-native-source-1" or source.get("complete") is not True
            or Path(source.get("root", "")).resolve() != native
            or source.get("pins_sha256") != digest(PINS)):
        raise ValueError("A completed pinned source manifest is required")
    verification = require_native_checkout(native, pin_data)
    if verification != source["verification"]:
        raise ValueError("Native checkout changed since source provisioning")
    gate_info = source["gate"]
    assessment = gate_info.get("assessment")
    gate = require_camera_gate(Path(gate_info["result"]), Path(gate_info["visual_review"]),
                               Path(assessment["path"]) if assessment else None)
    if (assets.get("schema") != "robodojo-native-assets-1" or assets.get("scan_errors")
            or assets.get("revision") != pin_data["assets"]["revision"]
            or assets.get("pins_sha256") != digest(PINS) or assets.get("source") != verification
            or assets.get("full_test_authorized") is not False or not assets.get("files")):
        raise ValueError("Complete hash-bound native assets with no unresolved references are required")
    names = set()
    for entry in assets["files"]:
        name = asset_path(entry["path"])
        path = native / name
        if (name in names or not path.resolve().is_relative_to(native)
                or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]):
            raise ValueError(f"Asset changed or escaped pinned checkout: {name}")
        names.add(name)
    layout = native / pin_data["selected_layout"]["path"]
    if str(layout.relative_to(native)) not in names or digest(layout) != pin_data["selected_layout"]["sha256"]:
        raise ValueError("Selected layout 0 is absent or changed")
    diagnostic = None
    if asset_diagnostic_assessment is not None:
        diagnostic = verify_asset_diagnostic_assessment(native, assets_path, assets, asset_diagnostic_assessment)
    elif assets.get("complete") is not True or assets.get("unresolved"):
        raise ValueError("Strict preview gate requires complete assets with no unresolved references")
    return {"asset_diagnostic_assessment": diagnostic, "standard_ready": False,
            "asset_fidelity_passed": False, "dependency_closure_complete": diagnostic is None,
            "source_manifest": str(source_path), "source_manifest_sha256": digest(source_path),
            "asset_manifest": str(assets_path), "asset_manifest_sha256": digest(assets_path),
            "asset_files_verified": len(names), "source": verification, "camera_gate": gate,
            "layout_path": str(layout), "layout_sha256": digest(layout)}


def check_authorization(path: Path, binding: dict, launch: bool) -> dict:
    auth = json.loads(path.read_text())
    if (auth.get("native_preview_authorized") is not True
            or auth.get("full_test_authorized") is not False
            or auth.get("model_calls_authorized") is not False
            or auth.get("gpu_uuid") != binding["gpu_uuid"] or not auth.get("user_message")):
        raise ValueError("A reset-only native preview authorization bound to this GPU is required")
    if launch and os.environ.get("UC71_NATIVE_PREVIEW_AUTHORIZED_CHILD") != "1":
        raise ValueError("Launch only under the reviewed bounded native preview supervisor")
    return {"path": str(path), "sha256": digest(path), "full_test_authorized": False,
            "model_calls_authorized": False}


def save_rgb_images(observation: dict, output: Path) -> list[dict]:
    import numpy as np
    from PIL import Image
    vision = observation["vision"]
    if set(vision) != EXPECTED_CAMERAS:
        raise RuntimeError(f"Unexpected native camera names: {sorted(vision)}")
    images = []
    for name in sorted(EXPECTED_CAMERAS):
        rgb = np.asarray(vision[name]["color"])
        if (rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8
                or not np.isfinite(rgb).all() or float(rgb.std()) == 0):
            raise RuntimeError(f"Invalid native RGB frame for {name}: {rgb.shape}/{rgb.dtype}")
        path = output / f"{name}.png"
        Image.fromarray(rgb).save(path)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        images.append({"camera": name, "path": path.name, "sha256": digest(path),
                       "shape": list(rgb.shape), "dtype": str(rgb.dtype),
                       "std": float(rgb.std()), "bytes": path.stat().st_size})
    return images


def run_preview(args, config, preparation, binding, output, progress) -> int:
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
            raise RuntimeError("Preview transport/planner exclusion failed")
        guarded_methods = forbid_policy_actions(env)
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
                     "h5py", "isaaclab", "isaaclab_tasks", "isaacsim", "websockets"):
            module = sys.modules.get(name)
            if module is not None:
                module_provenance[name] = {"file": str(getattr(module, "__file__", None)),
                                           "version": str(getattr(module, "__version__", None))}
        write_durable_json(output / "loaded_module_provenance.json", module_provenance)
        physics_before = float(env.sim.sim.current_time)
        steps_before_capture = counters["native_sim_step_calls"]
        env.obs_manager.render_for_capture()
        observation = env.obs_manager.get_obs()[0]
        images = save_rgb_images(observation, output)
        from robodojo_uc71.native_robot_inventory import RobotInventoryError, capture_robot_inventory
        try:
            inventory = capture_robot_inventory(env)
            inventory_name, inventory_status = "robot_inventory.json", "captured"
        except RobotInventoryError as error:
            inventory = {"schema": "robodojo-native-robot-inventory-error-1",
                         "scope": "setup_evaluator_only", "stage": error.stage,
                         "error_type": error.error_type, "model_input_eligible": False,
                         "calibration_status": "uncalibrated_candidate"}
            inventory_name, inventory_status = "robot_inventory_error.json", "extraction_failed"
        write_durable_json(output / inventory_name, inventory)
        inventory_record = {"status": inventory_status, "path": inventory_name,
                            "sha256": digest(output / inventory_name),
                            "calibration_status": "uncalibrated_candidate"}
        progress("robot_inventory_recorded", status=inventory_status)
        physics_after = float(env.sim.sim.current_time)
        if physics_before != physics_after or counters["native_sim_step_calls"] != steps_before_capture:
            raise RuntimeError("Preview capture advanced physics")
        env._check_physx_broken_envs()
        if monitor.is_fatal() or monitor.get_broken_envs() or env.take_action_cnt != [0]:
            raise RuntimeError("Native preview health or zero-action contract failed")
        result = {"schema": "robodojo-native-preview-result-1", "preview_render_passed": True,
                  "diagnostic_render_passed": True, "standard_ready": False, "asset_fidelity_passed": False,
                  "dependency_closure_complete": preparation["dependency_closure_complete"],
                  "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
                  "loaded_module_provenance_sha256": digest(output / "loaded_module_provenance.json"),
                  "completed_at_monotonic_s": time.monotonic(), "preparation_sha256": digest(output / "preparation.json"),
                  "layout_path": str(selected), "layout_sha256": digest(selected), "images": images,
                  "robot_inventory": inventory_record,
                  "binding": binding, "shutdown_mode": args.shutdown_mode, "reset_calls": 1,
                  "native_client_reset_calls": env.model_client.reset_calls, "model_calls": 0,
                  "benchmark_actions": 0, "native_take_action_count": list(env.take_action_cnt),
                  "native_setup_counts": counters, "setup_counts_note": "Counts delegated env.sim_step/render calls; native backend initialization may additionally step internally.",
                  "capture_physics_time_before_s": physics_before, "capture_physics_time_after_s": physics_after,
                  "reward_predicates_registered": True, "task_success_assessed": False,
                  "policy_action_methods_guarded": guarded_methods, "adaptations": adaptations,
                  "physics_device": env.device, "use_fabric": env.use_fabric,
                  "native_planner_instances": 0, "native_ik_solver_instances": 0,
                  "full_test_authorized": False, "gpu_peak_source": "parent gpu_monitor.jsonl; child makes no peak-memory claim"}
        write_durable_json(output / "preview_result.json", result)
        progress("three_native_rgb_images_saved", images=[item["path"] for item in images])
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_durable_json(output / "preview_failure.json", failure)
        progress("preview_failed", error=failure)
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
        evidence = {"schema": "robodojo-native-preview-preclose-1", "ready_for_shutdown": ready,
                    "standard_ready": False, "asset_fidelity_passed": False,
                    "dependency_closure_complete": preparation["dependency_closure_complete"],
                    "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
                    "monotonic_s": time.monotonic(), "child_pid": os.getpid(), "binding": binding,
                    "shutdown_mode": args.shutdown_mode, "no_pending_exception": no_pending_exception,
                    "native_env_close_returned": env is not None and not any(e["phase"] == "native_env_close" for e in cleanup_errors),
                    "physx_monitor_healthy": healthy, "cleanup_errors": cleanup_errors,
                    "preview_result_sha256": digest(output / "preview_result.json") if result else None,
                    "preparation_sha256": digest(output / "preparation.json"),
                    "image_count": len(result["images"]) if result else 0,
                    "model_calls": 0, "benchmark_actions": 0, "full_plugin_cleanup_proven": False,
                    "full_teardown_returned": False}
        write_durable_json(output / "preview_preclose_ready.json", evidence)
        progress("before_simulation_app_close", ready_for_shutdown=ready)
        # Fast shutdown may exit here; parent must require ready + hashes + clean exit/resources.
        if app is not None:
            try:
                app.close()
                write_durable_json(output / "preview_close_returned.json", {
                    "app_close_returned": True, "monotonic_s": time.monotonic(),
                    "shutdown_mode": args.shutdown_mode, "full_teardown_returned": args.shutdown_mode == "full"})
            except BaseException as error:
                cleanup_errors.append({"phase": "simulation_app_close", "type": type(error).__name__, "message": str(error)})
                write_durable_json(output / "preview_close_failure.json", cleanup_errors[-1])
    return 0 if ready and not cleanup_errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "source-manifest", "asset-manifest", "authorization", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--kit-index", type=int, required=True)
    parser.add_argument("--cuda-ordinal", type=int, required=True)
    parser.add_argument("--shutdown-mode", choices=("full", "stock-fast"), default="full")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--asset-diagnostic-assessment", type=Path,
                        help="Exact reviewed native_assets_002 diagnostic assessment; never grants Standard readiness")
    parser.add_argument("--parent-log", type=Path, help="Supervisor stdout/stderr log; required with --launch")
    args = parser.parse_args()
    if args.kit_index != 0 or args.cuda_ordinal != 0 or "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("This preview requires a single unmasked GPU with explicit Kit/CUDA ordinal 0")
    native, output = args.native_root.resolve(strict=True), args.output.resolve()
    if args.launch:
        if args.parent_log is None or not args.parent_log.is_file():
            raise SystemExit("The supervisor must open its stdout/stderr log before --launch")
        args.parent_log = args.parent_log.resolve(strict=True)
        if not args.parent_log.is_relative_to(output.parent):
            raise SystemExit("The supervisor log must be inside the fresh attempt directory")
    if any(char.isspace() for char in str(output)):
        raise SystemExit("Kit argument paths must contain no whitespace")
    binding = {"gpu_uuid": args.gpu_uuid, "pci_bus_id": args.pci_bus_id,
               "kit_index": args.kit_index, "cuda_ordinal": args.cuda_ordinal}
    auth = check_authorization(args.authorization.resolve(strict=True), binding, args.launch)
    preparation = verify_manifests(native, args.source_manifest.resolve(strict=True),
                                   args.asset_manifest.resolve(strict=True), pins(), args.asset_diagnostic_assessment)
    if preparation["camera_gate"]["gpu_uuid"] != binding["gpu_uuid"]:
        raise SystemExit("Preview GPU differs from the reviewed procedural camera gate")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    os.environ["ROBODOJO_RUN_ID"] = output.name
    preparation["import_paths"] = add_native_import_paths(native)
    config = build_native_preview_config(native, output.name)
    from omegaconf import OmegaConf
    write_durable_json(output / "native_configuration.json", OmegaConf.to_container(config, resolve=True))
    preparation.update(schema="robodojo-native-preview-preparation-1", authorization=auth,
        native_root=str(native), native_configuration_sha256=digest(output / "native_configuration.json"),
        source_file_sha256={str(Path(__file__).resolve()): digest(Path(__file__)),
            str(BASE / "src/robodojo_uc71/native_config.py"): digest(BASE / "src/robodojo_uc71/native_config.py"),
            str(BASE / "src/robodojo_uc71/native_robot_inventory.py"): digest(BASE / "src/robodojo_uc71/native_robot_inventory.py"),
            str(BASE / "scripts/new_host_camera.py"): digest(BASE / "scripts/new_host_camera.py"),
            str(BASE / "scripts/provision_native_common.py"): digest(BASE / "scripts/provision_native_common.py")},
        replaced_native_module_sha256={name: digest(native / name) for name in
            ("env/planner_manager/curobo_planner.py", "XPolicyLab/client_server/ws/model_client.py")},
        binding=binding, shutdown_mode=args.shutdown_mode, launch_requested=args.launch,
        model_calls=0, benchmark_actions=0, full_test_authorized=False)
    write_durable_json(output / "preparation.json", preparation)
    if not args.launch:
        print(json.dumps({"prepared": True, "simulator_starts": 0, "output": str(output)}), flush=True)
        return 0
    def progress(phase, **details):
        row = {"phase": phase, "monotonic_s": time.monotonic(), **details}
        with (output / "preview_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)
    return run_preview(args, config, preparation, binding, output, progress)


if __name__ == "__main__":
    raise SystemExit(main())
