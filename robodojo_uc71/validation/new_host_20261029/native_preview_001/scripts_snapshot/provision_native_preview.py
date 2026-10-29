#!/usr/bin/env python3
"""Supervise one reset-only native Make Toast layout-0 camera preview.

This is not a benchmark launcher. It requires fresh evidence, preserved package
metadata, complete source/assets, and a GPU-bound reset-only authorization.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from new_host_camera import normalize_pci, renderer_table_identity
from prepare_native_preview import EXPECTED_CAMERAS, check_authorization, verify_manifests
from provision_native_common import digest, pins
from provision_native_dependencies import METADATA_CODE, PINS as DEPENDENCY_PINS, package_map
from provision_new_host_camera import (child_environment, classify_terminal_exit,
    gpu_process_aliases, gpu_snapshot, owned_processes, process_record,
    terminate_owned, validate_single_gpu, write_json)

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def exact_fields(value: dict, expected: dict) -> bool:
    return all(type(value.get(k)) is type(v) and value.get(k) == v for k, v in expected.items())


def metadata(python: Path, target: Path | None = None) -> dict:
    command = [str(python), "-I", "-B", "-c", METADATA_CODE]
    if target is not None:
        command.append(str(target))
    return json.loads(subprocess.check_output(command, text=True, timeout=30,
        env={"PATH": str(python.parent) + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))


def verify_robot_inventory(child: Path, spec: dict) -> dict:
    states = {"captured": ("robot_inventory.json", "robodojo-native-robot-inventory-1"),
              "extraction_failed": ("robot_inventory_error.json", "robodojo-native-robot-inventory-error-1")}
    if spec.get("status") not in states or spec.get("calibration_status") != "uncalibrated_candidate":
        raise ValueError("Optional robot inventory status/calibration claim is invalid")
    name, schema = states[spec["status"]]
    path = child / name
    if spec.get("path") != name or path.is_symlink() or digest(path) != spec.get("sha256"):
        raise ValueError("Optional robot inventory artifact differs from the bound result")
    value = read_json(path)
    if not exact_fields(value, {"schema": schema, "scope": "setup_evaluator_only",
            "model_input_eligible": False, "calibration_status": "uncalibrated_candidate"}):
        raise ValueError("Optional robot inventory exceeds its uncalibrated setup scope")
    if spec["status"] == "captured":
        if value.get("num_envs") != 1 or len(value.get("robots", [])) != 2:
            raise ValueError("Optional robot inventory does not describe both native articulations")
    elif not value.get("stage") or not value.get("error_type"):
        raise ValueError("Optional robot inventory extraction failure was not surfaced")
    return {"record": spec, "captured": spec["status"] == "captured",
            "calibration_passed": False, "failure": value if spec["status"] == "extraction_failed" else None}


def verify_dependencies(python: Path, install_path: Path, dependency_path: Path) -> dict:
    installed, dependencies = read_json(install_path), read_json(dependency_path)
    base_lock = install_path.with_name("requirements.lock")
    original = read_json(install_path.with_name("package_inventory.json"))
    target = BASE / "runtime/native_overlay"
    if (installed.get("ready_for_diagnostic") is not True
            or Path(installed.get("python", "")).absolute() != python
            or digest(base_lock) != installed.get("lock_sha256")
            or dependencies.get("schema") != "robodojo-native-dependencies-1"
            or dependencies.get("ready_for_native_preview_dependencies") is not True
            or dependencies.get("base_unchanged") is not True
            or dependencies.get("base_lock_sha256") != installed["lock_sha256"]
            or dependencies.get("base_inventory_sha256") != digest(install_path.with_name("package_inventory.json"))
            or Path(dependencies.get("python", "")).absolute() != python
            or Path(dependencies.get("base_install_record", "")).resolve() != install_path
            or Path(dependencies.get("target", "")).resolve() != target.resolve()
            or dependencies.get("expected_overlay_packages") != DEPENDENCY_PINS
            or dependencies.get("full_test_approved") is not False):
        raise ValueError("Completed base and isolated overlay records must bind this interpreter and target")
    required_artifacts = {"base_before.json", "base_after.json", "base_constraints.txt", "requirements.in",
                          "requirements.lock", "resolved_requirements.lock", "overlay_inventory.json", "combined_inventory.json"}
    artifact_hashes = dependencies.get("artifacts_sha256", {})
    if set(artifact_hashes) != required_artifacts:
        raise ValueError("Dependency artifact manifest differs")
    for name, expected_hash in artifact_hashes.items():
        if digest(dependency_path.parent / name) != expected_hash:
            raise ValueError(f"Dependency artifact changed: {name}")
    if digest(dependency_path.with_name("requirements.lock")) != dependencies.get("lock_sha256"):
        raise ValueError("Overlay lock binding differs")
    actual_base, actual_overlay = metadata(python), metadata(python, target)
    base, overlay = package_map(actual_base), package_map(actual_overlay)
    if (actual_base.get("python_version") != [3, 11, 16]
            or actual_base.get("python") != original.get("python")
            or base != package_map(original) or len(base) != 140
            or base.get("isaacsim") != "5.1.0.0" or base.get("torch") != "2.7.0+cu128"
            or overlay != DEPENDENCY_PINS or set(base) & set(overlay)
            or overlay != package_map(read_json(dependency_path.with_name("overlay_inventory.json")))
            or base != package_map(read_json(dependency_path.with_name("base_after.json")))
            or dict(base, **overlay) != package_map(read_json(dependency_path.with_name("combined_inventory.json")))):
        raise ValueError("Actual base/overlay metadata differs from the preserved dependency records")
    return {"base_inventory": actual_base, "overlay_inventory": actual_overlay,
            "overlay_target": str(target), "install_record_sha256": digest(install_path),
            "dependency_record_sha256": digest(dependency_path),
            "base_lock_sha256": installed["lock_sha256"], "overlay_lock_sha256": dependencies["lock_sha256"]}


def verify_preview_evidence(child: Path, binding: dict, identity: dict, preparation_expected: dict,
                            source_hashes: dict, parent_log: Path, python: Path,
                            authorization_expected: dict) -> dict:
    """Validate durable child evidence; exit zero alone is never sufficient."""
    result = read_json(child / "preview_result.json")
    ready = read_json(child / "preview_preclose_ready.json")
    preparation = read_json(child / "preparation.json")
    cuda = read_json(child / "cuda_identity.json")
    renderer = read_json(child / "renderer_identity.json")
    expected_result = {"schema": "robodojo-native-preview-result-1", "preview_render_passed": True,
        "diagnostic_render_passed": True,
        "binding": binding, "shutdown_mode": "stock-fast", "reset_calls": 1,
        "native_client_reset_calls": 1, "model_calls": 0, "benchmark_actions": 0,
        "native_take_action_count": [0], "physics_device": "cpu", "use_fabric": False,
        "native_planner_instances": 0, "native_ik_solver_instances": 0,
        "reward_predicates_registered": True, "task_success_assessed": False, "full_test_authorized": False}
    expected_ready = {"schema": "robodojo-native-preview-preclose-1", "ready_for_shutdown": True,
        "no_pending_exception": True, "native_env_close_returned": True, "physx_monitor_healthy": True,
        "cleanup_errors": [], "image_count": 3, "binding": binding, "child_pid": identity["pid"],
        "shutdown_mode": "stock-fast", "model_calls": 0, "benchmark_actions": 0,
        "full_plugin_cleanup_proven": False, "full_teardown_returned": False}
    fidelity = {key: preparation_expected[key] for key in ("standard_ready", "asset_fidelity_passed",
                "dependency_closure_complete", "asset_diagnostic_assessment")}
    expected_result.update(fidelity)
    expected_ready.update(fidelity)
    if not exact_fields(result, expected_result) or not exact_fields(ready, expected_ready):
        raise ValueError("Native result or durable preclose scope/health contract failed")
    if not exact_fields(preparation, fidelity):
        raise ValueError("Child preparation changed the asset diagnostic/fidelity classification")
    completed, closing = result.get("completed_at_monotonic_s"), ready.get("monotonic_s")
    counts = result.get("native_setup_counts", {})
    if (type(completed) not in (int, float) or not math.isfinite(completed)
            or type(closing) not in (int, float) or not math.isfinite(closing) or not 0 < completed <= closing
            or type(counts.get("native_sim_step_calls")) is not int or counts["native_sim_step_calls"] <= 0
            or type(counts.get("native_render_calls")) is not int or counts["native_render_calls"] < 0):
        raise ValueError("Native setup counters or preclose ordering are invalid")
    preparation_hash, result_hash = digest(child / "preparation.json"), digest(child / "preview_result.json")
    if (ready.get("preview_result_sha256") != result_hash
            or ready.get("preparation_sha256") != preparation_hash
            or result.get("preparation_sha256") != preparation_hash
            or preparation.get("source_manifest_sha256") != preparation_expected["source_manifest_sha256"]
            or preparation.get("asset_manifest_sha256") != preparation_expected["asset_manifest_sha256"]
            or preparation.get("source") != preparation_expected["source"]
            or preparation.get("camera_gate") != preparation_expected["camera_gate"]
            or preparation.get("authorization") != authorization_expected
            or preparation.get("binding") != binding
            or preparation.get("full_test_authorized") is not False
            or preparation.get("model_calls") != 0 or preparation.get("benchmark_actions") != 0
            or preparation.get("native_configuration_sha256") != digest(child / "native_configuration.json")
            or result.get("layout_path") != preparation_expected["layout_path"]
            or result.get("layout_sha256") != preparation_expected["layout_sha256"]):
        raise ValueError("Native result/preparation/source/layout hashes differ")
    expected_helpers = {str(SCRIPTS / name) for name in (
        "prepare_native_preview.py", "new_host_camera.py", "provision_native_common.py")}
    expected_helpers.add(str(BASE / "src/robodojo_uc71/native_config.py"))
    expected_helpers.add(str(BASE / "src/robodojo_uc71/native_robot_inventory.py"))
    if set(preparation.get("source_file_sha256", {})) != expected_helpers:
        raise ValueError("Child helper provenance is incomplete")
    for name, expected_hash in preparation["source_file_sha256"].items():
        if source_hashes.get(name) != expected_hash:
            raise ValueError(f"Child loaded an unbound helper: {name}")
    loaded_modules = read_json(child / "loaded_module_provenance.json")
    if (not loaded_modules or result.get("loaded_module_provenance_sha256")
            != digest(child / "loaded_module_provenance.json")):
        raise ValueError("Loaded-module provenance is missing or differs from the result hash")
    robot_inventory = verify_robot_inventory(child, result.get("robot_inventory", {}))
    capture_before, capture_after = (result.get("capture_physics_time_before_s"), result.get("capture_physics_time_after_s"))
    if (type(capture_before) not in (int, float) or not math.isfinite(capture_before)
            or type(capture_after) not in (int, float) or not math.isfinite(capture_after)
            or capture_before != capture_after):
        raise ValueError("Image capture advanced native physics")
    if any((child / name).exists() for name in ("preview_failure.json", "preview_close_failure.json")):
        raise ValueError("Child recorded an execution or close failure")
    cuda_expected = {"gpu_uuid": binding["gpu_uuid"], "cuda_ordinal": 0, "device_count": 1,
                     "primary_context_created": False, "child_pid": identity["pid"],
                     "child_start_ticks": identity["start_ticks"]}
    if (not exact_fields(cuda, cuda_expected)
            or normalize_pci(cuda.get("pci_bus_id", "")) != normalize_pci(binding["pci_bus_id"])):
        raise ValueError("CUDA metadata identity differs from the owned child/GPU")
    parsed_renderer = renderer_table_identity(parent_log.read_text(errors="replace"), binding)
    if (renderer.get("binding") != binding or renderer.get("uuid_prefix_and_pci_bus_match") is not True
            or parsed_renderer.get("uuid_prefix_and_pci_bus_match") is not True
            or renderer.get("active_rows") != parsed_renderer.get("active_rows")):
        raise ValueError("Kit renderer identity does not match the actual parent log/GPU")
    images = result.get("images", [])
    if len(images) != 3 or {row.get("camera") for row in images} != EXPECTED_CAMERAS:
        raise ValueError("Exactly the three native cameras are required")
    paths = []
    for row in images:
        if (row.get("path") != row["camera"] + ".png" or row.get("shape") != [480, 640, 3]
                or row.get("dtype") != "uint8" or not isinstance(row.get("std"), (int, float))
                or not row["std"] > 0):
            raise ValueError("Native image names, dimensions, or statistics differ")
        path = child / row["path"]
        if path.is_symlink() or digest(path) != row.get("sha256") or path.stat().st_size != row.get("bytes"):
            raise ValueError(f"Native image artifact changed: {row['camera']}")
        paths.append(str(path))
    # Decode independently in the preserved CPU-only Pillow/NumPy packages.
    image_code = """import json,sys,numpy as np
from PIL import Image
rows=[]
for path in sys.argv[1:]:
 with Image.open(path) as image:
  image.load(); array=np.asarray(image)
  rows.append({'path':path,'format':image.format,'mode':image.mode,'shape':list(array.shape),
               'dtype':str(array.dtype),'std':float(array.std())})
print(json.dumps(rows))
"""
    decoded = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", image_code, *paths],
        text=True, timeout=30, env={"PATH": str(python.parent) + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))
    if len(decoded) != 3 or any(row["format"] != "PNG" or row["mode"] != "RGB"
            or row["shape"] != [480, 640, 3] or row["dtype"] != "uint8" or not row["std"] > 0 for row in decoded):
        raise ValueError("Independent PNG decoding failed the native RGB contract")
    return {"preclose_ready_verified": True, "native_render_verified": True,
            "cuda_identity_verified": True, "renderer_identity_verified": True,
            "preview_result_sha256": result_hash, "preparation_sha256": preparation_hash,
            "loaded_module_provenance": loaded_modules,
            "robot_inventory": robot_inventory,
            "images": images, "independent_image_decoding": decoded,
            "image_visual_review_required": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authorization", "install-record", "dependency-record", "source-manifest", "asset-manifest",
                 "output", "python", "library-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pci-bus-id")
    parser.add_argument("--asset-diagnostic-assessment", type=Path,
                        help="Exact reviewed upstream-asset diagnostic assessment; strict closure is the default")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    parser.add_argument("--vk-icd", type=Path, default=Path("/etc/vulkan/icd.d/nvidia_icd.json"))
    parser.add_argument("--eula-record", type=Path, default=BASE / "validation/provisioning/vendor_authorization.json")
    args = parser.parse_args()
    python, output = args.python.absolute(), args.output.resolve()
    if (not python.is_file() or not os.access(python, os.X_OK) or not 30 <= args.timeout_s <= 1200
            or args.minimum_free_mib < 16384 or not args.vk_icd.is_file()):
        parser.error("Valid interpreter/ICD, deadline 30..1200s, and at least 16384MiB free are required")
    library_path = args.library_path.resolve(strict=True)
    if (not library_path.is_dir() or not library_path.is_relative_to(BASE / "runtime")
            or output.exists() or not output.is_relative_to(BASE / "validation")
            or any(char.isspace() for char in str(output))):
        parser.error("Use task-local runtime libraries and a fresh validation output without whitespace")
    auth_path = args.authorization.resolve(strict=True)
    auth = read_json(auth_path)
    binding = {"gpu_uuid": args.gpu_uuid or auth.get("gpu_uuid"),
               "pci_bus_id": args.pci_bus_id or auth.get("pci_bus_id"), "kit_index": 0, "cuda_ordinal": 0}
    if not binding["gpu_uuid"] or not binding["pci_bus_id"]:
        parser.error("GPU UUID and PCI address are required in authorization or explicit flags")
    authorization = check_authorization(auth_path, binding, launch=False)
    authorized_deadline = auth.get("maximum_wall_time_s", 1200)
    if (auth.get("benchmark_actions_authorized") is not False
            or auth.get("automatic_retry", False) is not False
            or auth.get("shutdown_mode", "stock-fast") != "stock-fast"
            or type(authorized_deadline) is not int or args.timeout_s > authorized_deadline
            or (auth.get("pci_bus_id") is not None
                and normalize_pci(auth["pci_bus_id"]) != normalize_pci(binding["pci_bus_id"]))):
        parser.error("Authorization must forbid actions/retries and bind this PCI address, shutdown mode and deadline")
    if read_json(args.eula_record).get("nvidia_eula_accepted") is not True:
        parser.error("Preserved NVIDIA licence acceptance is required")
    source_path, assets_path = args.source_manifest.resolve(strict=True), args.asset_manifest.resolve(strict=True)
    asset_assessment = args.asset_diagnostic_assessment.resolve(strict=True) if args.asset_diagnostic_assessment else None
    native = Path(read_json(source_path)["root"]).resolve(strict=True)
    preparation = verify_manifests(native, source_path, assets_path, pins(), asset_assessment)
    if preparation["camera_gate"]["gpu_uuid"] != binding["gpu_uuid"]:
        parser.error("Native preparation gate is bound to another GPU")
    dependency_info = verify_dependencies(python, args.install_record.resolve(strict=True), args.dependency_record.resolve(strict=True))
    before = gpu_snapshot()
    gpu = validate_single_gpu(before, binding, require_idle=True)
    if int(gpu["memory_free"].split()[0]) < args.minimum_free_mib:
        parser.error("Insufficient free GPU memory for the bounded native preview")
    kernel = Path("/proc/driver/nvidia/version").read_text()
    if not re.search(r"\b" + re.escape(gpu["driver_version"]) + r"\b", kernel):
        parser.error("Injected kernel and NVML driver versions differ")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "config", "data", "scripts_snapshot"):
        (output / name).mkdir()
    child_output = output / "preview"  # Child creates it exclusively.
    helpers = [SCRIPTS / name for name in ("prepare_native_preview.py", "provision_native_preview.py",
        "new_host_camera.py", "provision_new_host_camera.py", "provision_native_common.py",
        "provision_native_dependencies.py")]
    helpers.append(BASE / "src/robodojo_uc71/native_config.py")
    helpers.append(BASE / "src/robodojo_uc71/native_robot_inventory.py")
    source_hashes = {str(path): digest(path) for path in helpers}
    for path in helpers:
        shutil.copyfile(path, output / "scripts_snapshot" / path.name)
    for path, name in ((auth_path, "authorization.json"), (args.install_record, "base_install_status.json"),
                       (args.dependency_record, "dependency_install_status.json"), (source_path, "source_manifest.json"),
                       (assets_path, "asset_manifest.json"), (args.eula_record, "vendor_authorization.json")):
        shutil.copyfile(path, output / name)
    if asset_assessment is not None:
        shutil.copyfile(asset_assessment, output / "asset_diagnostic_assessment.json")
    for name, value in (("gpu_before.json", before), ("preflight_preparation.json", preparation),
                        ("dependency_binding.json", dependency_info)):
        write_json(output / name, value)
    (output / "kernel_driver.txt").write_text(kernel)
    environment = child_environment(python, output, binding, args.vk_icd.resolve(), library_path, "stock-fast")
    environment = {key: value for key, value in environment.items() if not key.startswith("UC71_CAMERA_")}
    environment.update(PYTHONPATH=dependency_info["overlay_target"], UC71_NATIVE_PREVIEW_AUTHORIZED_CHILD="1")
    parent_log = output / "stdout_stderr.log"
    argv = [str(python), "-B", "-X", "faulthandler", "-u", str(SCRIPTS / "prepare_native_preview.py"),
            "--native-root", str(native), "--source-manifest", str(source_path), "--asset-manifest", str(assets_path),
            "--authorization", str(auth_path), "--output", str(child_output), "--gpu-uuid", binding["gpu_uuid"],
            "--pci-bus-id", binding["pci_bus_id"], "--kit-index", "0", "--cuda-ordinal", "0",
            "--shutdown-mode", "stock-fast", "--parent-log", str(parent_log), "--launch"]
    if asset_assessment is not None:
        argv.extend(("--asset-diagnostic-assessment", str(asset_assessment)))
    record = {"schema": "robodojo-native-preview-launch-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": argv, "environment": environment, "binding": binding, "source_file_sha256": source_hashes,
        "authorization_sha256": digest(auth_path), "dependency_binding": dependency_info,
        "preparation": preparation, "wall_time_limit_s": args.timeout_s, "minimum_free_mib": args.minimum_free_mib,
        "local_library_sha256": {path.name: digest(path) for path in library_path.iterdir() if path.is_file()},
        "shutdown_mode": "stock-fast", "full_plugin_cleanup_proven": False, "full_teardown_returned": False,
        "shutdown_limitation": "NVIDIA stock fast exit skips complete plugin/Python teardown; successful preclose and process/GPU release are required.",
        "automatic_retry": False, "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
        "standard_ready": False, "asset_fidelity_passed": False,
        "dependency_closure_complete": preparation["dependency_closure_complete"],
        "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
        "limitations": "Reset/camera preview only; environment filtering is not filesystem/network isolation; peak VRAM is sampled."}
    write_json(output / "launch.json", record)
    known, sent, previous_handlers, observed_libraries = {}, [], {}, set()
    proc = initial = None
    namespace_candidate = None
    peak = int(gpu["memory_used"].split()[0])
    minimum_free = int(gpu["memory_free"].split()[0])
    started = time.monotonic()

    def interrupt(signum, _frame):
        raise InterruptedError(f"Native preview supervisor received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        with parent_log.open("x") as log, (output / "gpu_monitor.jsonl").open("x") as monitor:
            proc = subprocess.Popen(argv, cwd=output, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = proc.pid
            initial = process_record(proc.pid)
            if initial is None:
                raise RuntimeError("Could not capture the directly owned child's kernel identity")
            known[proc.pid] = initial["start_ticks"]
            write_json(output / "launcher_pid.json", {"launcher_pid": os.getpid(), "child": initial})
            while proc.poll() is None:
                remaining = args.timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Native preview wall-time deadline reached")
                members = owned_processes(proc.pid, known)
                snapshot = gpu_snapshot(timeout_s=min(10, max(0.1, remaining)))
                exit_code = proc.poll()
                monitor.write(json.dumps({"elapsed_s": time.monotonic() - started,
                    "child_returncode": exit_code, "owned_processes": members, "gpus": snapshot}) + "\n")
                monitor.flush()
                current_gpu = validate_single_gpu(snapshot, binding)
                peak = max(peak, int(current_gpu["memory_used"].split()[0]))
                minimum_free = min(minimum_free, int(current_gpu["memory_free"].split()[0]))
                if exit_code is not None:
                    break  # Preserve actual exit while NVML bookkeeping drains.
                aliases = gpu_process_aliases(known, child_identity=initial, child_unreaped=proc.returncode is None)
                unknown = [row for row in current_gpu["processes"] if row["pid"] not in aliases]
                if unknown and proc.poll() is None:
                    if (len({row["pid"] for row in current_gpu["processes"]}) != 1
                            or len({row["pid"] for row in unknown}) != 1
                            or process_record(unknown[0]["pid"]) is not None):
                        raise RuntimeError(f"Unexpected or ambiguous GPU process occupancy: {unknown}")
                    candidate = unknown[0]["pid"]
                    if namespace_candidate is not None and namespace_candidate != candidate:
                        raise RuntimeError("Inaccessible NVML process candidate changed")
                    namespace_candidate = candidate
                    record["nvml_namespace_candidate"] = {"pid": candidate, "binding_proven": False,
                        "basis": "Sole inaccessible process on initially idle GPU; final release required"}
                for member in members:
                    try:
                        maps = Path(f"/proc/{member['pid']}/maps").read_text()
                    except OSError:
                        continue
                    for line in maps.splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) == 6 and re.search(r"/(?:libcuda|libnvidia|libGLX|libEGL|libGLU|libvulkan)", fields[5]):
                            observed_libraries.add(fields[5])
                time.sleep(0.5)
            record["returncode"] = proc.wait(timeout=5)
    except BaseException as error:
        record["termination"] = type(error).__name__
        record["termination_detail"] = str(error)
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if proc is not None:
            if proc.poll() is None or owned_processes(proc.pid, known):
                try:
                    sent.extend(terminate_owned(proc, known))
                except Exception as error:
                    record["cleanup_error"] = repr(error)
            record["returncode"] = proc.poll()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    record.update(elapsed_s=time.monotonic() - started, termination_signals=sent,
        remaining_owned_pids=[row["pid"] for row in owned_processes(proc.pid, known)] if proc else [],
        loaded_graphics_library_paths=sorted(observed_libraries), sampled_peak_gpu_used_mib=peak,
        sampled_peak_gpu_used_delta_mib=peak - int(gpu["memory_used"].split()[0]), sampled_minimum_free_mib=minimum_free)
    try:
        after = gpu_snapshot()
        deadline = time.monotonic() + 10
        while any(item["processes"] for item in after) and time.monotonic() < deadline:
            time.sleep(0.5)
            after = gpu_snapshot(timeout_s=3)
        write_json(output / "gpu_after.json", after)
        final_gpu = validate_single_gpu(after, binding)
        record["gpu_processes_released"] = not final_gpu["processes"]
        record["gpu_memory_used_delta_mib"] = int(final_gpu["memory_used"].split()[0]) - int(gpu["memory_used"].split()[0])
        record["gpu_memory_released"] = record["gpu_memory_used_delta_mib"] <= 256
    except Exception as error:
        record["gpu_cleanup_check_error"] = repr(error)
    try:
        if initial is None:
            raise ValueError("Owned child identity is missing")
        record.update(verify_preview_evidence(child_output, binding, initial, preparation, source_hashes,
                                             parent_log, python, authorization))
        if digest(auth_path) != record["authorization_sha256"]:
            raise ValueError("Preview authorization changed during this attempt")
        if any(digest(Path(name)) != value for name, value in source_hashes.items()):
            raise ValueError("A launcher/child/helper changed during this attempt")
        if verify_manifests(native, source_path, assets_path, pins(), asset_assessment) != preparation:
            raise ValueError("Native source/assets changed during this attempt")
        if verify_dependencies(python, args.install_record.resolve(), args.dependency_record.resolve()) != dependency_info:
            raise ValueError("Package metadata or dependency records changed during this attempt")
        record["bindings_unchanged_after_preview"] = True
    except Exception as error:
        record["evidence_validation_error"] = f"{type(error).__name__}: {error}"
    record["shutdown_contract_satisfied"] = bool(record.get("preclose_ready_verified") is True
        and record.get("returncode") == 0 and not sent and "termination" not in record)
    record["clean_shutdown_confirmed"] = bool(record["shutdown_contract_satisfied"]
        and record.get("gpu_processes_released") is True and record.get("gpu_memory_released") is True
        and not record["remaining_owned_pids"] and "cleanup_error" not in record)
    record["diagnostic_render_passed"] = bool(record["clean_shutdown_confirmed"]
        and record.get("native_render_verified") is True and record.get("bindings_unchanged_after_preview") is True
        and "evidence_validation_error" not in record)
    if proc is not None and record.get("returncode") is not None:
        record["terminal_exit"] = classify_terminal_exit(record["returncode"], record.get("termination"), sent, child_pid=proc.pid)
    write_json(output / "launch_result.json", record)
    print(json.dumps({"output": str(output), "diagnostic_render_passed": record["diagnostic_render_passed"],
                      "asset_fidelity_passed": record["asset_fidelity_passed"], "standard_ready": record["standard_ready"],
                      "returncode": record.get("returncode"), "termination": record.get("termination"),
                      "evidence_validation_error": record.get("evidence_validation_error")}), flush=True)
    return 0 if record["diagnostic_render_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
