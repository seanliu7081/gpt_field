#!/usr/bin/env python3
"""Supervise a separately authorized fixed37-action native robot diagnostic.

Default is read-only command review. --launch is required to create an attempt
and start its child. No benchmark attempt or full-test approval is created.
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
import sys
import time

from new_host_camera import normalize_pci, renderer_table_identity
from prepare_native_characterization import (RGB_ACTIONS, check_authorization, program_files,
    verify_previous_preview, content_hash, validate_probe_plan)
from prepare_native_preview import EXPECTED_CAMERAS, verify_manifests
from provision_native_preview import exact_fields, read_json, verify_dependencies
from provision_native_common import digest, pins
from provision_native_compatibility import verify_compatibility_record
from provision_new_host_camera import (child_environment, classify_terminal_exit, gpu_process_aliases,
    gpu_snapshot, owned_processes, process_record, terminate_owned, validate_single_gpu, write_json)

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"


def _artifact(child, spec):
    path = child / spec["path"]
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(child) or digest(path) != spec["sha256"]:
        raise ValueError("Characterization artifact changed or escaped its child directory")
    return path


def _robot_state(value, plan, *, initial=False):
    if not exact_fields(value, {"schema": "robodojo-native-characterization-measurement-1",
            "calibration_status": "uncalibrated_candidate", "standard_ready": False, "model_input_eligible": False}):
        raise ValueError("Diagnostic robot measurement exceeds candidate scope")
    if set(value.get("arms", {})) != {"left", "right"}:
        raise ValueError("Both measured robot arms are required")
    for arm in ("left", "right"):
        item = value["arms"][arm]
        for key, count in (("joint_positions", 6), ("joint_velocities", 6), ("finger_positions", 2),
                           ("link6_world_pose_wxyz", 7)):
            numbers = item[key]
            if len(numbers) != count or any(type(x) not in (int, float) or not math.isfinite(x) for x in numbers):
                raise ValueError("Nonfinite or malformed measured robot vector")
        bound = plan["initial_joint_tolerance_rad"] if initial else plan["measured_joint_envelope_rad"]
        if (max(abs(a-b) for a, b in zip(item["joint_positions"], plan["initial"][arm]["joint_positions"])) > bound
                or max(abs(x) for x in item["joint_velocities"]) > plan["measured_velocity_limit_rad_s"]
                or max(abs(x-.044) for x in item["finger_positions"]) > .001):
            raise ValueError("Diagnostic measured joint/open-gripper envelope changed")
        jacobian = item["link6_actor_world_jacobian"]
        if len(jacobian) != 6 or any(len(row) != 6 or any(type(x) not in (int, float) or not math.isfinite(x) for x in row) for row in jacobian):
            raise ValueError("Malformed finite measured Jacobian")
    raw = value.get("raw_robot_inventory", {})
    if raw.get("num_envs") != 1 or len(raw.get("robots", [])) != 2:
        raise ValueError("Raw dual-articulation evidence is missing")


def _capture(child, capture, label, plan, *, rgb):
    if capture.get("label") != label or capture.get("physics_unchanged_during_capture") is not True:
        raise ValueError("Synchronized capture identity or paused-physics contract differs")
    for name in ("camera_inventory", "robot_measurement"):
        if capture[name]["path"] != label + "/" + name + ".json":
            raise ValueError("Unexpected capture artifact path")
    camera = read_json(_artifact(child, capture["camera_inventory"]))
    robot = read_json(_artifact(child, capture["robot_measurement"]))
    _robot_state(robot, plan, initial=label == "before")
    if (camera.get("schema") != "robodojo-native-camera-inventory-1"
            or camera.get("calibration_status") != "uncalibrated_candidate"
            or camera.get("standard_ready") is not False or camera.get("model_input_eligible") is not False
            or set(camera.get("cameras", {})) != {"head", "left_wrist", "right_wrist"}
            or camera.get("native_physics_time_s") != capture["native_physics_time_s"]
            or robot.get("native_physics_time_s") != capture["native_physics_time_s"]
            or robot.get("native_physics_counter") != capture["native_physics_counter"]
            or robot.get("native_action_counter") != capture["native_action_counter"]):
        raise ValueError("Camera/robot inventory is not synchronized to its native action state")
    images = capture.get("images", [])
    if (len(images) != (3 if rgb else 0)
            or (rgb and {row.get("camera") for row in images} != EXPECTED_CAMERAS)):
        raise ValueError("RGB capture set differs from the fixed instrumentation plan")
    paths = []
    for row in images:
        if (row.get("path") != label + "/" + row["camera"] + ".png"
                or row.get("shape") != [480, 640, 3] or row.get("dtype") != "uint8"
                or type(row.get("std")) not in (int, float) or not row["std"] > 0):
            raise ValueError("Invalid clean native RGB metadata")
        path = _artifact(child, row)
        if path.stat().st_size != row.get("bytes"):
            raise ValueError("RGB byte count changed")
        paths.append(str(path))
    return robot, paths


def verify_characterization_evidence(child, binding, identity, preparation_expected, source_hashes,
                                    parent_log, python, authorization, plan, prior_preview):
    result = read_json(child / "characterization_result.json")
    ready = read_json(child / "characterization_preclose_ready.json")
    preparation = read_json(child / "preparation.json")
    log_text = parent_log.read_text(errors="replace")
    if re.search(r"ModuleNotFoundError:|ImportError:|Failed to import python module|Failed to startup python extension", log_text):
        raise ValueError("Native extension startup/import failure remains")
    scope = {"model_calls": 0, "benchmark_actions": 0, "benchmark_attempt_started": False,
        "full_test_authorized": False, "standard_ready": False, "asset_fidelity_passed": False}
    if not exact_fields(result, {**scope, "schema": "robodojo-native-characterization-result-1",
            "diagnostic_completed": True, "diagnostic_native_actions": 37, "diagnostic_physics_substeps": 370,
            "image_count": 18, "native_take_action_count": [37], "reset_calls": 1, "native_client_reset_calls": 1,
            "calibration_passed": False, "task_success_assessed": False, "reward_predicates_registered": True,
            "binding": binding, "shutdown_mode": "stock-fast", "physics_device": "cpu", "use_fabric": False,
            "native_planner_instances": 0, "native_ik_solver_instances": 0}):
        raise ValueError("Fixed diagnostic scope/result contract differs")
    if not exact_fields(ready, {**scope, "schema": "robodojo-native-characterization-preclose-1",
            "ready_for_shutdown": True, "no_pending_exception": True, "native_env_close_returned": True,
            "physx_monitor_healthy": True, "cleanup_errors": [], "diagnostic_native_actions": 37,
            "child_pid": identity["pid"], "binding": binding, "shutdown_mode": "stock-fast", "image_count": 18,
            "full_plugin_cleanup_proven": False, "full_teardown_returned": False}):
        raise ValueError("Durable preclose evidence is not clean or correctly scoped")
    if any((child / name).exists() for name in ("characterization_failure.json", "characterization_close_failure.json")):
        raise ValueError("Child recorded execution/cleanup failure")
    result_hash, preparation_hash = digest(child / "characterization_result.json"), digest(child / "preparation.json")
    if (ready.get("characterization_result_sha256") != result_hash
            or ready.get("preparation_sha256") != preparation_hash or result.get("preparation_sha256") != preparation_hash
            or preparation.get("authorization") != authorization or preparation.get("source_file_sha256") != source_hashes
            or preparation.get("prior_preview") != prior_preview
            or preparation.get("plan_sha256") != content_hash(plan) or result.get("plan_sha256") != content_hash(plan)
            or result.get("authorization_sha256") != authorization["sha256"]
            or result.get("layout_path") != preparation_expected["layout_path"]
            or result.get("layout_sha256") != preparation_expected["layout_sha256"]
            or result.get("plan_file_sha256") != preparation.get("plan_file_sha256")
            or preparation.get("native_configuration_sha256") != digest(child / "native_configuration.json")
            or result.get("loaded_module_provenance_sha256") != digest(child / "loaded_module_provenance.json")
            or result.get("rgb_checkpoint_action_indices") != list(RGB_ACTIONS)
            or not 0 < result["completed_at_monotonic_s"] <= ready["monotonic_s"]):
        raise ValueError("Diagnostic program/plan/authorization/preparation evidence changed")
    for key in ("source_manifest_sha256", "asset_manifest_sha256", "source", "camera_gate", "layout_path", "layout_sha256",
                "asset_diagnostic_assessment", "dependency_closure_complete"):
        if preparation.get(key) != preparation_expected[key]:
            raise ValueError("Characterization source/assets/prior gates changed")
    for value in (result, ready):
        if any(value.get(key) != preparation_expected[key] for key in
               ("asset_diagnostic_assessment", "dependency_closure_complete")):
            raise ValueError("Diagnostic asset-fidelity limitation differs from the reviewed source evidence")
    cuda, renderer = read_json(child / "cuda_identity.json"), read_json(child / "renderer_identity.json")
    if (not exact_fields(cuda, {"gpu_uuid": binding["gpu_uuid"], "cuda_ordinal": 0,
            "primary_context_created": False, "child_pid": identity["pid"], "child_start_ticks": identity["start_ticks"]})
            or type(cuda.get("device_count")) is not int or cuda["device_count"] < 1
            or normalize_pci(cuda.get("pci_bus_id", "")) != normalize_pci(binding["pci_bus_id"])):
        raise ValueError("Actual CUDA identity does not match owned child/GPU")
    parsed = renderer_table_identity(log_text, binding)
    if (renderer.get("binding") != binding or renderer.get("uuid_prefix_and_pci_bus_match") is not True
            or parsed.get("uuid_prefix_and_pci_bus_match") is not True or renderer.get("active_rows") != parsed.get("active_rows")):
        raise ValueError("Actual Kit renderer identity differs")
    captures, actions = result.get("captures", []), result.get("actions", [])
    if len(captures) != 6 or len(actions) != 37 or captures[0].get("label") != "before":
        raise ValueError("Fixed action/capture evidence count differs")
    previous = captures[0]
    if previous.get("native_action_counter") != [0]:
        raise ValueError("Diagnostic did not begin from the fresh reset native action count")
    previous_robot, paths = _capture(child, previous, "before", plan, rgb=True)
    checkpoints = [previous]
    for index, row in enumerate(actions, 1):
        if row.get("index") != index or row.get("before_capture") != previous["label"] or row["receipt"]["path"] != f"native_action_{index:03d}.json":
            raise ValueError("Diagnostic action order/path differs")
        receipt = read_json(_artifact(child, row["receipt"]))
        expected = {**scope, "schema": "robodojo-native-characterization-action-1", "diagnostic_native_action_index": index,
            "scope": "diagnostic_native_actions",
            "plan_sha256": content_hash(plan), "authorization_sha256": authorization["sha256"],
            "target": plan["actions"][index-1]["action"], "planned_probe": plan["actions"][index-1]["probe"],
            "calibration_passed": False, "task_success_assessed": False, "stop_after_this_action": index == 37}
        expected.pop("benchmark_actions"); expected.pop("asset_fidelity_passed")
        if not exact_fields(receipt, expected):
            raise ValueError("Native diagnostic target/scope differs from fixed plan")
        accounting = receipt["native_accounting"]
        if (not exact_fields(accounting, {"native_counter_before": index-1, "native_counter_after": index,
                "native_sim_step_calls": 10, "physics_dt_s": .004, "native_terminal_callback_calls": 1,
                "native_reward_callback_calls": 1, "native_reward_step_calls": 1, "native_end_observed": False,
                "task_success_assessed": False, "benchmark_attempt_started": False, "model_calls": 0})
                or accounting["native_physics_counter_after"] - accounting["native_physics_counter_before"] != 10
                or not math.isclose(accounting["physics_time_after_s"]-accounting["physics_time_before_s"], .04, abs_tol=1e-7, rel_tol=0)):
            raise ValueError("Real native callback/substep/action accounting differs")
        after = row["after_capture"]
        robot, images = _capture(child, after, f"action_{index:03d}", plan, rgb=index in RGB_ACTIONS)
        if (receipt["before"]["arms"] != previous_robot["arms"] or receipt["after"]["arms"] != robot["arms"]
                or accounting["physics_time_before_s"] != previous["native_physics_time_s"]
                or accounting["physics_time_after_s"] != after["native_physics_time_s"]
                or after["native_action_counter"] != [index]
                or accounting["native_physics_counter_before"] != previous["native_physics_counter"]
                or accounting["native_physics_counter_after"] != after["native_physics_counter"]
                or after["native_physics_counter"] - previous["native_physics_counter"] != 10
                or after["native_sim_step_calls"] - previous["native_sim_step_calls"] != 10):
            raise ValueError("Per-action camera/robot capture is not aligned with actual native physics")
        if images:
            checkpoints.append(after)
        paths.extend(images)
        previous, previous_robot = after, robot
    if (captures != checkpoints or len(paths) != 18
            or not math.isclose(result["diagnostic_simulated_time_s"], 1.48, rel_tol=0., abs_tol=1e-7)):
        raise ValueError("Fixed RGB instrumentation or total simulation time differs")
    code = "import json,sys,numpy as np; from PIL import Image; rows=[]\nfor p in sys.argv[1:]:\n with Image.open(p) as im:\n  im.load(); a=np.asarray(im); rows.append({'path':p,'format':im.format,'mode':im.mode,'shape':list(a.shape),'dtype':str(a.dtype),'std':float(a.std())})\nprint(json.dumps(rows))"
    decoded = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", code, *paths], text=True, timeout=60,
        env={"PATH": str(python.parent)+":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))
    if len(decoded) != 18 or any(not exact_fields(row, {"format": "PNG", "mode": "RGB", "shape": [480,640,3], "dtype": "uint8"}) or not row["std"] > 0 for row in decoded):
        raise ValueError("Independent clean RGB decoding failed")
    return {"preclose_ready_verified": True, "native_render_verified": True, "diagnostic_actions_verified": True,
        "cuda_identity_verified": True, "renderer_identity_verified": True, "extension_startup_clean": True,
        "diagnostic_native_actions": 37, "diagnostic_physics_substeps": 370, "diagnostic_simulated_time_s": 1.48,
        "characterization_result_sha256": result_hash, "preparation_sha256": preparation_hash,
        "independent_image_decoding": decoded, "image_visual_review_required": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authorization", "install-record", "dependency-record", "source-manifest", "asset-manifest",
                 "output", "python", "library-path", "plan", "preview-result"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pci-bus-id")
    parser.add_argument("--launch", action="store_true", help="Execute the explicitly bound diagnostic; default only reviews inputs")
    parser.add_argument("--compatibility-record", type=Path, required=True,
                        help="Reviewed isolated hidapi/setuptools compatibility installation")
    parser.add_argument("--asset-diagnostic-assessment", type=Path, required=True,
                        help="Exact reviewed upstream-asset diagnostic assessment; strict closure is the default")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    parser.add_argument("--vk-icd", type=Path, default=Path("/etc/vulkan/icd.d/nvidia_icd.json"))
    parser.add_argument("--eula-record", type=Path, default=BASE / "validation/provisioning/vendor_authorization.json")
    args = parser.parse_args()
    python, output = args.python.absolute(), args.output.resolve()
    if (not python.is_file() or not os.access(python, os.X_OK) or not 30 <= args.timeout_s <= 900
            or args.minimum_free_mib < 16384 or not args.vk_icd.is_file()):
        parser.error("Valid interpreter/ICD, deadline 30..900s, and at least 16384MiB free are required")
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
    plan_path, preview_path = args.plan.resolve(strict=True), args.preview_result.resolve(strict=True)
    plan = read_json(plan_path)
    validate_probe_plan(plan)
    prior_preview = verify_previous_preview(preview_path, plan)
    authorization = check_authorization(auth_path, plan, binding, launch=False, output_parent=output)
    authorized_deadline = auth["maximum_wall_time_s"]
    if (auth.get("diagnostic_native_actions_authorized") is not True
            or auth.get("automatic_retry", False) is not False
            or auth.get("shutdown_mode", "stock-fast") != "stock-fast"
            or type(authorized_deadline) is not int or args.timeout_s > authorized_deadline
            or (auth.get("pci_bus_id") is not None
                and normalize_pci(auth["pci_bus_id"]) != normalize_pci(binding["pci_bus_id"]))):
        parser.error("Authorization must bind only fixed diagnostic actions and forbid retries and bind this PCI address, shutdown mode and deadline")
    if read_json(args.eula_record).get("nvidia_eula_accepted") is not True:
        parser.error("Preserved NVIDIA licence acceptance is required")
    source_path, assets_path = args.source_manifest.resolve(strict=True), args.asset_manifest.resolve(strict=True)
    asset_assessment = args.asset_diagnostic_assessment.resolve(strict=True) if args.asset_diagnostic_assessment else None
    native = Path(read_json(source_path)["root"]).resolve(strict=True)
    preparation = verify_manifests(native, source_path, assets_path, pins(), asset_assessment)
    if preparation["camera_gate"]["gpu_uuid"] != binding["gpu_uuid"]:
        parser.error("Native preparation gate is bound to another GPU")
    dependency_info = verify_dependencies(python, args.install_record.resolve(strict=True), args.dependency_record.resolve(strict=True))
    compatibility_path = args.compatibility_record.resolve(strict=True) if args.compatibility_record else None
    compatibility_info = verify_compatibility_record(python, compatibility_path, dependency_info) if compatibility_path else None
    before = gpu_snapshot()
    gpu = validate_single_gpu(before, binding, require_idle=True)
    if int(gpu["memory_free"].split()[0]) < args.minimum_free_mib:
        parser.error("Insufficient free GPU memory for the bounded native characterization")
    kernel = Path("/proc/driver/nvidia/version").read_text()
    if not re.search(r"\b" + re.escape(gpu["driver_version"]) + r"\b", kernel):
        parser.error("Injected kernel and NVML driver versions differ")
    helpers = program_files()
    source_hashes = {str(path): digest(path) for path in helpers}
    if not args.launch:
        print(json.dumps({"inputs_verified": True, "simulator_starts": 0, "diagnostic_native_actions": 0,
            "plan_sha256": content_hash(plan), "source_file_sha256": source_hashes,
            "output": str(output), "reviewed_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--launch"],
            "benchmark_attempt_started": False, "full_test_authorized": False}), flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for name in ("home", "cache", "config", "data", "scripts_snapshot"):
        (output / name).mkdir()
    child_output = output / "characterization"  # Child creates it exclusively.
    write_json(output / "program_binding.json", source_hashes)
    for path in helpers:
        shutil.copyfile(path, output / "scripts_snapshot" / path.name)
    for path, name in ((auth_path, "authorization.json"), (args.install_record, "base_install_status.json"),
                       (args.dependency_record, "dependency_install_status.json"), (source_path, "source_manifest.json"),
                       (assets_path, "asset_manifest.json"), (args.eula_record, "vendor_authorization.json"),
                       (plan_path, "plan.json"), (preview_path, "prior_preview_result.json")):
        shutil.copyfile(path, output / name)
    if asset_assessment is not None:
        shutil.copyfile(asset_assessment, output / "asset_diagnostic_assessment.json")
    if compatibility_path is not None:
        shutil.copyfile(compatibility_path, output / "compatibility_install_status.json")
    for name, value in (("gpu_before.json", before), ("preflight_preparation.json", preparation),
                        ("dependency_binding.json", dependency_info)):
        write_json(output / name, value)
    (output / "kernel_driver.txt").write_text(kernel)
    environment = child_environment(python, output, binding, args.vk_icd.resolve(), library_path, "stock-fast")
    environment = {key: value for key, value in environment.items() if not key.startswith("UC71_CAMERA_")}
    import_path = dependency_info["overlay_target"]
    if compatibility_info is not None:
        import_path = compatibility_info["target"] + os.pathsep + import_path
    environment.update(PYTHONPATH=import_path, UC71_NATIVE_CHARACTERIZATION_AUTHORIZED_CHILD="1")
    parent_log = output / "stdout_stderr.log"
    argv = [str(python), "-B", "-X", "faulthandler", "-u", str(SCRIPTS / "prepare_native_characterization.py"),
            "--native-root", str(native), "--source-manifest", str(source_path), "--asset-manifest", str(assets_path),
            "--authorization", str(auth_path), "--output", str(child_output), "--gpu-uuid", binding["gpu_uuid"],
            "--pci-bus-id", binding["pci_bus_id"], "--kit-index", "0", "--cuda-ordinal", "0",
            "--shutdown-mode", "stock-fast", "--parent-log", str(parent_log), "--launch"]
    argv.extend(("--plan", str(plan_path), "--preview-result", str(preview_path), "--program-binding", str(output / "program_binding.json")))
    if asset_assessment is not None:
        argv.extend(("--asset-diagnostic-assessment", str(asset_assessment)))
    record = {"schema": "robodojo-native-characterization-launch-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": argv, "environment": environment, "binding": binding, "source_file_sha256": source_hashes,
        "authorization_sha256": digest(auth_path), "dependency_binding": dependency_info,
        "compatibility_binding": compatibility_info,
        "plan_sha256": content_hash(plan), "plan_file_sha256": digest(plan_path), "prior_preview": prior_preview,
        "diagnostic_native_actions_planned": 37, "benchmark_attempt_started": False, "rgb_checkpoint_action_indices": list(RGB_ACTIONS),
        "preparation": preparation, "wall_time_limit_s": args.timeout_s, "minimum_free_mib": args.minimum_free_mib,
        "local_library_sha256": {path.name: digest(path) for path in library_path.iterdir() if path.is_file()},
        "shutdown_mode": "stock-fast", "full_plugin_cleanup_proven": False, "full_teardown_returned": False,
        "shutdown_limitation": "NVIDIA stock fast exit skips complete plugin/Python teardown; successful preclose and process/GPU release are required.",
        "automatic_retry": False, "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
        "standard_ready": False, "asset_fidelity_passed": False,
        "dependency_closure_complete": preparation["dependency_closure_complete"],
        "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
        "limitations": "Fixed37-action diagnostic only; no calibration/task-success claim; environment filtering is not filesystem/network isolation; peak VRAM is sampled."}
    write_json(output / "launch.json", record)
    known, sent, previous_handlers, observed_libraries = {}, [], {}, set()
    proc = initial = None
    namespace_candidate = None
    peak = int(gpu["memory_used"].split()[0])
    minimum_free = int(gpu["memory_free"].split()[0])
    started = time.monotonic()

    def interrupt(signum, _frame):
        raise InterruptedError(f"Native characterization supervisor received signal {signum}")

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
                    raise TimeoutError("Native characterization wall-time deadline reached")
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
        record.update(verify_characterization_evidence(child_output, binding, initial, preparation, source_hashes,
                                             parent_log, python, authorization, plan, prior_preview))
        if digest(plan_path) != record["plan_file_sha256"] or verify_previous_preview(preview_path, plan) != prior_preview:
            raise ValueError("Fixed diagnostic plan/prior-preview evidence changed")
        if digest(auth_path) != record["authorization_sha256"]:
            raise ValueError("Characterization authorization changed during this attempt")
        if any(digest(Path(name)) != value for name, value in source_hashes.items()):
            raise ValueError("A launcher/child/helper changed during this attempt")
        if verify_manifests(native, source_path, assets_path, pins(), asset_assessment) != preparation:
            raise ValueError("Native source/assets changed during this attempt")
        if verify_dependencies(python, args.install_record.resolve(), args.dependency_record.resolve()) != dependency_info:
            raise ValueError("Package metadata or dependency records changed during this attempt")
        if compatibility_path is not None and verify_compatibility_record(python, compatibility_path, dependency_info) != compatibility_info:
            raise ValueError("Compatibility metadata or installation record changed during this attempt")
        record["bindings_unchanged_after_characterization"] = True
    except Exception as error:
        record["evidence_validation_error"] = f"{type(error).__name__}: {error}"
    record["shutdown_contract_satisfied"] = bool(record.get("preclose_ready_verified") is True
        and record.get("returncode") == 0 and not sent and "termination" not in record)
    record["clean_shutdown_confirmed"] = bool(record["shutdown_contract_satisfied"]
        and record.get("gpu_processes_released") is True and record.get("gpu_memory_released") is True
        and not record["remaining_owned_pids"] and "cleanup_error" not in record)
    record["diagnostic_completed"] = bool(record["clean_shutdown_confirmed"]
        and record.get("native_render_verified") is True and record.get("diagnostic_actions_verified") is True
        and record.get("bindings_unchanged_after_characterization") is True
        and "evidence_validation_error" not in record)
    if proc is not None and record.get("returncode") is not None:
        record["terminal_exit"] = classify_terminal_exit(record["returncode"], record.get("termination"), sent, child_pid=proc.pid)
    write_json(output / "launch_result.json", record)
    print(json.dumps({"output": str(output), "diagnostic_completed": record["diagnostic_completed"],
                      "asset_fidelity_passed": record["asset_fidelity_passed"], "standard_ready": record["standard_ready"],
                      "returncode": record.get("returncode"), "termination": record.get("termination"),
                      "evidence_validation_error": record.get("evidence_validation_error")}), flush=True)
    return 0 if record["diagnostic_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
