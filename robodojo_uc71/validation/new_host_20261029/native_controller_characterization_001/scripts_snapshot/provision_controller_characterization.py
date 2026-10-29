#!/usr/bin/env python3
"""Bounded isolated controller diagnostic; default reviews inputs without launch."""
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
import stat
import subprocess
import sys
import time
import numpy as np

from new_host_camera import normalize_pci, renderer_table_identity
from prepare_controller_characterization import (MAX_ACTIONS, SCOPE, accumulated_time_tolerance,
    check_authorization, program_files, verify_first_probe, content_hash, validate_controller_probe_proposal,
    physical_arm_states, measured_envelope, stage_controller, stage_joint_command, validate_operator_review)
from prepare_native_preview import EXPECTED_CAMERAS, verify_manifests
from provision_native_preview import exact_fields, read_json, verify_dependencies
from provision_native_common import digest, pins
from provision_native_compatibility import verify_compatibility_record
from provision_native_characterization import _artifact
from provision_new_host_camera import (child_environment, classify_terminal_exit, gpu_process_aliases,
    gpu_snapshot, owned_processes, process_record, terminate_owned, validate_single_gpu, write_json)
from robodojo_uc71.worker_isolation import popen_identity, WORKER_UID, WORKER_GID

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"


def prepare_worker_git_config(output, native, pin_data):
    """Permit read-only Git inspection of only the root-owned pinned checkouts."""
    paths = [native, *(native/entry["path"] for entry in pin_data["gitlinks"])]
    if any(any(c in str(path) for c in "\r\n") for path in paths): raise ValueError("Invalid Git source path")
    path = output/"home/.gitconfig"
    with path.open("x") as stream:
        stream.write("[safe]\n"+"".join("\tdirectory = "+str(p)+"\n" for p in paths))
    os.chmod(path, 0o644)
    return {"path":str(path), "sha256":digest(path), "safe_directories":[str(p) for p in paths],
        "scope":"fresh_private_HOME_only", "optional_git_locks":False}


def write_operator_review(request_path, *, expected_request_sha256, decision, reason):
    """Root/operator calls only after inspecting the exact request's fresh RGB."""
    if os.getuid() != 0 or os.geteuid() != 0: raise ValueError("Root/operator review required")
    request_path = Path(request_path).resolve(strict=True)
    if digest(request_path) != expected_request_sha256: raise ValueError("Reviewed request changed")
    request = read_json(request_path)
    arm = request.get("active_arm")
    child = request_path.parent.parent
    if (arm not in ("left", "right") or request_path != child/f"clearance_{arm}/request.json"
            or child.name != "controller" or child.parent.name != "worker"
            or request.get("schema") != "robodojo-controller-clearance-request-1"):
        raise ValueError("Unexpected controller clearance request path/schema")
    for image in request["capture"]["images"]:
        _artifact(child, image)
    reviews = child.parents[1]/"operator_reviews"
    info = reviews.lstat()
    if (not stat.S_ISDIR(info.st_mode) or reviews.is_symlink() or info.st_uid != 0
            or info.st_gid != WORKER_GID or stat.S_IMODE(info.st_mode) != 0o750):
        raise ValueError("Root-owned worker-read-only review directory required")
    value = {"schema":"robodojo-controller-clearance-review-1", "request_sha256":expected_request_sha256,
        "plan_sha256":request["plan_sha256"], "stage_index":request["stage_index"], "active_arm":arm,
        "native_action_counter":request["native_action_counter"], "image_sha256":request["image_sha256"],
        "reviewer":"root/operator", "decision":decision, "reason":reason,
        "full_test_authorized":False, "benchmark_attempt_started":False, "object_truth_used":False}
    validate_operator_review(value, request, expected_request_sha256)
    path = reviews/f"{arm}.json"
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o640)
    with os.fdopen(fd, "w") as stream:
        os.fchown(stream.fileno(), 0, WORKER_GID); os.fchmod(stream.fileno(), 0o640)
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    directory_fd = os.open(reviews, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    return {"path":str(path), "sha256":digest(path), "decision":decision}


def verify_capture(child, capture, label, plan, geometry):
    if capture.get("label") != label or capture.get("physics_unchanged_during_capture") is not True:
        raise ValueError("Controller capture identity/paused-physics contract differs")
    for kind in ("camera_inventory", "robot_measurement"):
        if capture[kind]["path"] != label+"/"+kind+".json": raise ValueError("Unexpected capture path")
    robot = read_json(_artifact(child, capture["robot_measurement"]))
    camera = read_json(_artifact(child, capture["camera_inventory"]))
    states = physical_arm_states(robot, geometry)
    measured_envelope(robot, states, plan, initial=label == "before")
    if (not exact_fields(robot, {"schema": "robodojo-native-characterization-measurement-1",
            "scope": "diagnostic_native_actions", "benchmark_attempt_started": False,
            "full_test_authorized": False, "model_calls": 0, "standard_ready": False})
            or camera.get("schema") != "robodojo-native-camera-inventory-1"
            or camera.get("standard_ready") is not False or camera.get("model_input_eligible") is not False
            or set(camera.get("cameras", {})) != {"head", "left_wrist", "right_wrist"}
            or camera.get("native_physics_time_s") != capture["native_physics_time_s"]
            or robot.get("native_physics_time_s") != capture["native_physics_time_s"]
            or robot.get("native_physics_counter") != capture["native_physics_counter"]
            or robot.get("native_action_counter") != capture["native_action_counter"]):
        raise ValueError("Controller robot/camera capture is not synchronized")
    images = capture.get("images", [])
    if len(images) != 3 or {x.get("camera") for x in images} != EXPECTED_CAMERAS:
        raise ValueError("Every stage boundary requires three clean RGB views")
    paths = []
    for image in images:
        if (image["path"] != label+"/"+image["camera"]+".png"
                or image["shape"] != [480,640,3] or image["dtype"] != "uint8"
                or not math.isfinite(image["std"]) or image["std"] <= 0): raise ValueError("Invalid RGB metadata")
        path = _artifact(child, image)
        if path.stat().st_size != image["bytes"]: raise ValueError("RGB byte count changed")
        paths.append(str(path))
    return robot, paths


def verify_clearance(child, spec, stage, capture, plan):
    directory = child / f"clearance_{stage['active_arm']}"
    path = directory / "wait_result.json"
    if spec.get("path") != str(path) or digest(path) != spec["sha256"]: raise ValueError("Clearance result binding differs")
    wait = read_json(path); request_path = directory / "request.json"
    request = read_json(request_path)
    review_path = child.parents[1] / "operator_reviews" / f"{stage['active_arm']}.json"
    info = review_path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_gid != WORKER_GID
            or stat.S_IMODE(info.st_mode) != 0o640 or info.st_nlink != 1):
        raise ValueError("Operator review is not root-owned and read-only to the worker")
    review = read_json(review_path)
    if (wait.get("schema") != "robodojo-controller-clearance-wait-1" or wait.get("accepted") is not True
            or wait.get("failure") is not None or wait.get("physics_unchanged") is not True
            or wait.get("simulation_updates_during_wait") != 0 or wait.get("before") != wait.get("after")
            or wait.get("request_sha256") != digest(request_path)
            or wait.get("review_path") != str(review_path) or wait.get("review_sha256") != digest(review_path)
            or wait.get("review_consumed_sha256") != digest(directory / "review_consumed.json")
            or read_json(directory / "review_consumed.json") != review
            or spec.get("review_path") != str(review_path) or spec.get("review_sha256") != digest(review_path)
            or request.get("capture") != capture or request.get("plan_sha256") != content_hash(plan)
            or request.get("stage_index") != stage["index"] or request.get("active_arm") != stage["active_arm"]
            or request.get("native_action_counter") != capture["native_action_counter"]
            or request.get("physics_state") != wait["before"]
            or wait["before"]["native_physics_time_s"] != capture["native_physics_time_s"]
            or wait["before"]["native_physics_counter"] != capture["native_physics_counter"]
            or wait["before"]["native_action_counter"] != capture["native_action_counter"]
            or request.get("image_sha256") != {x["path"]:x["sha256"] for x in capture["images"]}
            or not 0 <= wait["end_monotonic_s"]-wait["start_monotonic_s"] <= 121
            or not validate_operator_review(review, request, digest(request_path))):
        raise ValueError("Fresh operator clearance or motionless wait evidence differs")


def verify_controller_evidence(child, binding, identity, preparation_expected, source_hashes,
                               parent_log, python, authorization, plan, first_probe):
    result = read_json(child / "controller_result.json"); ready = read_json(child / "controller_preclose_ready.json")
    preparation = read_json(child / "preparation.json"); log_text = parent_log.read_text(errors="replace")
    if re.search(r"ModuleNotFoundError:|ImportError:|Failed to import python module|Failed to startup python extension", log_text):
        raise ValueError("Native extension startup/import failure remains")
    image_count = (len(plan["stages"])+1)*3
    if not exact_fields(result, {**SCOPE, "schema": "robodojo-controller-characterization-result-1",
            "diagnostic_completed": True, "diagnostic_native_actions": MAX_ACTIONS,
            "diagnostic_physics_substeps": MAX_ACTIONS*10, "image_count": image_count,
            "native_take_action_count": [MAX_ACTIONS], "reset_calls": 1, "native_client_reset_calls": 1,
            "task_success_assessed": False, "reward_predicates_registered": True, "binding": binding,
            "shutdown_mode": "stock-fast", "physics_device": "cpu", "use_fabric": False,
            "native_planner_instances": 0, "native_ik_solver_instances": 0, "automatic_configuration_selection": False}):
        raise ValueError("Controller diagnostic result scope differs")
    if not exact_fields(ready, {"schema": "robodojo-controller-characterization-preclose-1",
            "ready_for_shutdown": True, "no_pending_exception": True, "native_env_close_returned": True,
            "physx_monitor_healthy": True, "cleanup_errors": [], "diagnostic_native_actions": MAX_ACTIONS,
            "child_pid": identity["pid"], "binding": binding, "shutdown_mode": "stock-fast", "image_count": image_count,
            "full_plugin_cleanup_proven": False, "full_teardown_returned": False,
            "benchmark_attempt_started": False, "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
            "standard_ready": False, "asset_fidelity_passed": False}):
        raise ValueError("Durable controller preclose evidence is not clean")
    if any((child/name).exists() for name in ("controller_failure.json", "controller_close_failure.json")):
        raise ValueError("Child recorded a controller execution/cleanup failure")
    result_hash, prep_hash = digest(child/"controller_result.json"), digest(child/"preparation.json")
    if (ready.get("controller_result_sha256") != result_hash or ready.get("preparation_sha256") != prep_hash
            or result.get("preparation_sha256") != prep_hash or preparation.get("authorization") != authorization
            or preparation.get("source_file_sha256") != source_hashes or preparation.get("first_probe") != first_probe
            or result.get("first_probe") != first_probe or result.get("plan_sha256") != content_hash(plan)
            or preparation.get("plan_sha256") != content_hash(plan) or result.get("authorization_sha256") != authorization["sha256"]
            or preparation.get("native_configuration_sha256") != digest(child/"native_configuration.json")
            or result.get("loaded_module_provenance_sha256") != digest(child/"loaded_module_provenance.json")
            or not 0 < result["completed_at_monotonic_s"] <= ready["monotonic_s"]):
        raise ValueError("Controller program/plan/authorization evidence changed")
    for key in ("source_manifest_sha256", "asset_manifest_sha256", "source", "camera_gate", "layout_path", "layout_sha256",
                "asset_diagnostic_assessment", "dependency_closure_complete"):
        if preparation.get(key) != preparation_expected[key]: raise ValueError("Controller preparation binding changed")
        if key in ("asset_diagnostic_assessment", "dependency_closure_complete") and (result.get(key) != preparation_expected[key] or ready.get(key) != preparation_expected[key]):
            raise ValueError("Controller asset limitation differs")
    isolation = read_json(child / "worker_isolation.json")
    if (result.get("worker_isolation_sha256") != digest(child/"worker_isolation.json")
            or not exact_fields(isolation, {"schema": "robodojo-worker-unix-isolation-proof-1", "pid": identity["pid"],
                "uid": WORKER_UID, "gid": WORKER_GID, "supplementary_groups": [], "no_new_privs": True,
                "effective_permitted_inheritable_ambient_capabilities": 0,
                "access_denied": {"credential_directory": True, "parent_environment": True},
                "namespace_isolation": False, "network_isolation": False})):
        raise ValueError("Native worker credential isolation did not verify")
    graphics = read_json(child/"loaded_graphics_library_paths.json")
    if (result.get("loaded_graphics_library_paths_sha256") != digest(child/"loaded_graphics_library_paths.json")
            or graphics.get("pid") != identity["pid"] or graphics.get("source") != "/proc/self/maps"
            or not graphics.get("paths")): raise ValueError("Child graphics-library provenance missing")
    cuda = read_json(child/"cuda_identity.json"); renderer = read_json(child/"renderer_identity.json")
    if (not exact_fields(cuda, {"gpu_uuid": binding["gpu_uuid"], "cuda_ordinal": 0, "device_count": 1,
            "primary_context_created": False, "child_pid": identity["pid"], "child_start_ticks": identity["start_ticks"]})
            or normalize_pci(cuda.get("pci_bus_id", "")) != normalize_pci(binding["pci_bus_id"])):
        raise ValueError("Actual CUDA identity differs")
    parsed = renderer_table_identity(log_text, binding)
    if (renderer.get("binding") != binding or renderer.get("uuid_prefix_and_pci_bus_match") is not True
            or parsed.get("uuid_prefix_and_pci_bus_match") is not True or renderer.get("active_rows") != parsed.get("active_rows")):
        raise ValueError("Actual Kit renderer identity differs")
    captures, actions, stages = result["captures"], result["actions"], result["stages"]
    if len(captures) != len(plan["stages"])+1 or len(actions) != MAX_ACTIONS or len(stages) != len(plan["stages"]):
        raise ValueError("Controller action/stage/capture counts differ")
    geometry = read_json(Path(plan["geometry_candidate_path"]))
    previous, paths = verify_capture(child, captures[0], "before", plan, geometry)
    start_time = previous["native_physics_time_s"]; start_counter = previous["native_physics_counter"]
    index = 0; clearances = {}
    for stage, entry in zip(plan["stages"], stages):
        if not exact_fields(entry, {"stage_index": stage["index"], "label": stage["label"],
                "native_actions": stage["native_actions"], "first_action_index": index+1,
                "last_action_index": index+stage["native_actions"]}): raise ValueError("Controller stage ordering differs")
        states = physical_arm_states(previous, geometry)
        inactive = {arm: np.asarray(states[arm].joint_positions).copy() for arm in ("left", "right") if arm != stage["active_arm"]}
        controller = None if stage["kind"] in ("visual_checkpoint", "joint_hold") else stage_controller(stage, states)
        for local_index in range(1, stage["native_actions"]+1):
            index += 1; row = actions[index-1]
            if row.get("index") != index or row.get("stage_index") != stage["index"] or row["receipt"]["path"] != f"native_action_{index:04d}.json":
                raise ValueError("Controller native action ordering/path differs")
            receipt = read_json(_artifact(child, row["receipt"]))
            if not exact_fields(receipt, {**SCOPE, "schema": "robodojo-controller-native-action-1",
                    "native_action_index": index, "stage_index": stage["index"], "stage_action_index": local_index,
                    "measured_envelope_passed": True, "measurement_validation_error": None,
                    "plan_sha256": content_hash(plan), "authorization_sha256": authorization["sha256"]}):
                raise ValueError("Controller action receipt scope differs")
            before, after = receipt["before"], receipt["after"]
            if (before["arms"] != previous["arms"] or before["native_physics_time_s"] != previous["native_physics_time_s"]
                    or before["native_physics_counter"] != previous["native_physics_counter"]
                    or before["native_action_counter"] != [index-1] or after["native_action_counter"] != [index]):
                raise ValueError("Per-action measured robot chain differs")
            states = physical_arm_states(before, geometry)
            measured_envelope(before, states, plan, inactive_hold=inactive)
            measured_envelope(after, physical_arm_states(after, geometry), plan, inactive_hold=inactive)
            if controller is None:
                expected_action = {f"{arm}_arm_joint_state":plan["home"][arm]["joint_positions"] for arm in ("left", "right")}
                expected_action.update({f"{arm}_ee_joint_state":[1.] for arm in ("left", "right")})
            else:
                expected_action, _ = stage_joint_command(stage, controller, states)
            if (set(receipt["target"]) != set(expected_action)
                    or any(not np.allclose(receipt["target"][k], expected_action[k], atol=1e-12, rtol=1e-10) for k in expected_action)):
                raise ValueError("Native command differs from independently reconstructed controller output")
            accounting = receipt["native_accounting"]
            if (not exact_fields(accounting, {"native_counter_before": index-1, "native_counter_after": index,
                    "native_sim_step_calls": 10, "physics_dt_s": .004, "native_terminal_callback_calls": 1,
                    "native_reward_callback_calls": 1, "native_reward_step_calls": 1, "native_end_observed": False,
                    "task_success_assessed": False, "benchmark_attempt_started": False, "model_calls": 0})
                    or accounting["native_physics_counter_before"] != before["native_physics_counter"]
                    or accounting["native_physics_counter_after"] != after["native_physics_counter"]
                    or after["native_physics_counter"]-before["native_physics_counter"] != 10
                    or accounting["physics_time_before_s"] != before["native_physics_time_s"]
                    or accounting["physics_time_after_s"] != after["native_physics_time_s"]
                    or not math.isclose(after["native_physics_time_s"]-before["native_physics_time_s"], .04, abs_tol=1e-7, rel_tol=0)):
                raise ValueError("Actual native controller cadence/callback accounting differs")
            expected_clearance = clearances.get(stage["active_arm"]) if stage["kind"] == "gripper_close" else None
            if receipt.get("clearance") != expected_clearance or (stage["kind"] == "gripper_close" and expected_clearance is None):
                raise ValueError("Gripper close lacks the prior fresh operator clearance")
            previous = after
        capture = entry["capture"]
        observed, images = verify_capture(child, capture, f"stage_{stage['index']:03d}", plan, geometry)
        if (observed["arms"] != previous["arms"] or observed["native_physics_time_s"] != previous["native_physics_time_s"]
                or capture != captures[stage["index"]+1]): raise ValueError("Stage RGB differs from acknowledged robot state")
        paths.extend(images)
        if stage["kind"] == "visual_checkpoint":
            verify_clearance(child, entry["clearance"], stage, capture, plan)
            clearances[stage["active_arm"]] = entry["clearance"]
        else:
            analysis = read_json(_artifact(child, entry["analysis"]))
            if not exact_fields(analysis, {"schema": "robodojo-controller-characterization-stage-analysis-1",
                    "stage": stage["label"], "calibration_passed": False, "standard_ready": False,
                    "benchmark_attempt_started": False, "automatic_configuration_selection": False}):
                raise ValueError("Stage analysis improperly claims automatic calibration")
        previous = observed
    actual_time = previous["native_physics_time_s"]-start_time
    if (index != MAX_ACTIONS or previous["native_physics_counter"]-start_counter != MAX_ACTIONS*10
            or result["clearances"] != clearances or set(clearances) != {"left", "right"}
            or not math.isclose(actual_time, MAX_ACTIONS*.04, rel_tol=0, abs_tol=accumulated_time_tolerance(MAX_ACTIONS))
            or result["diagnostic_simulated_time_s"] != actual_time
            or result["accumulated_dt_tolerance_s"] != accumulated_time_tolerance(MAX_ACTIONS)):
        raise ValueError("Controller total count/time/clearance evidence differs")
    code = "import json,sys,numpy as np; from PIL import Image; rows=[]\nfor p in sys.argv[1:]:\n with Image.open(p) as im:\n  im.load(); a=np.asarray(im); rows.append({'path':p,'format':im.format,'mode':im.mode,'shape':list(a.shape),'dtype':str(a.dtype),'std':float(a.std())})\nprint(json.dumps(rows))"
    decoded = json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", code, *paths], text=True, timeout=60,
        env={"PATH":str(python.parent)+":/usr/bin:/bin", "PYTHONNOUSERSITE":"1"}))
    if len(decoded) != image_count or any(not exact_fields(row, {"format":"PNG", "mode":"RGB", "shape":[480,640,3], "dtype":"uint8"}) or not row["std"] > 0 for row in decoded):
        raise ValueError("Independent RGB decoding failed")
    return {"preclose_ready_verified": True, "native_render_verified": True, "diagnostic_actions_verified": True,
        "cuda_identity_verified": True, "renderer_identity_verified": True, "extension_startup_clean": True,
        "worker_isolation_verified": True, "operator_clearances_verified": True,
        "diagnostic_native_actions": MAX_ACTIONS, "diagnostic_physics_substeps": MAX_ACTIONS*10,
        "diagnostic_simulated_time_s": actual_time, "controller_result_sha256": result_hash,
        "preparation_sha256": prep_hash, "independent_image_decoding": decoded,
        "child_loaded_graphics_library_paths": graphics["paths"], "image_visual_review_required": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authorization", "install-record", "dependency-record", "source-manifest", "asset-manifest",
                 "output", "python", "library-path", "plan", "first-probe-result", "first-probe-analysis"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pci-bus-id")
    parser.add_argument("--launch", action="store_true", help="Execute the explicitly bound diagnostic; default only reviews inputs")
    parser.add_argument("--compatibility-record", type=Path, required=True,
                        help="Reviewed isolated hidapi/setuptools compatibility installation")
    parser.add_argument("--asset-diagnostic-assessment", type=Path, required=True,
                        help="Exact reviewed upstream-asset diagnostic assessment; strict closure is the default")
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    parser.add_argument("--vk-icd", type=Path, default=Path("/etc/vulkan/icd.d/nvidia_icd.json"))
    parser.add_argument("--eula-record", type=Path, default=BASE / "validation/provisioning/vendor_authorization.json")
    args = parser.parse_args()
    python, output = args.python.absolute(), args.output.resolve()
    if (not python.is_file() or not os.access(python, os.X_OK) or not 30 <= args.timeout_s <= 1800
            or args.minimum_free_mib < 16384 or not args.vk_icd.is_file()):
        parser.error("Valid interpreter/ICD, deadline 30..1800s, and at least 16384MiB free are required")
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
    plan_path = args.plan.resolve(strict=True)
    probe_path, analysis_path = args.first_probe_result.resolve(strict=True), args.first_probe_analysis.resolve(strict=True)
    plan = read_json(plan_path)
    validate_controller_probe_proposal(plan)
    first_probe = verify_first_probe(probe_path, analysis_path, plan)
    authorization = check_authorization(auth_path, plan, binding, first_probe=first_probe,
                                         launch=False, output_parent=output)
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
    os.chown(output, 0, WORKER_GID); os.chmod(output, 0o750)
    for name in ("home", "cache", "config", "data", "worker"):
        path = output / name; path.mkdir(mode=0o700)
        os.chown(path, WORKER_UID, WORKER_GID)
    for name in ("scripts_snapshot", "operator_reviews"):
        path = output / name; path.mkdir(mode=0o750)
        os.chown(path, 0, WORKER_GID); os.chmod(path, 0o750)
    child_output = output / "worker/controller"  # Worker creates only this fresh child directory.
    worker_git = prepare_worker_git_config(output, native, pins())
    write_json(output / "program_binding.json", source_hashes)
    for path in helpers:
        shutil.copyfile(path, output / "scripts_snapshot" / path.name)
    for path, name in ((auth_path, "authorization.json"), (args.install_record, "base_install_status.json"),
                       (args.dependency_record, "dependency_install_status.json"), (source_path, "source_manifest.json"),
                       (assets_path, "asset_manifest.json"), (args.eula_record, "vendor_authorization.json"),
                       (plan_path, "plan.json"), (probe_path, "first_probe_result.json"), (analysis_path, "first_probe_analysis.json")):
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
    environment.update(PYTHONPATH=import_path, UC71_CONTROLLER_CHARACTERIZATION_AUTHORIZED_CHILD="1",
        UC71_CONTROLLER_PARENT_PID=str(os.getpid()), GIT_OPTIONAL_LOCKS="0")
    parent_log = output / "stdout_stderr.log"
    argv = [str(python), "-B", "-X", "faulthandler", "-u", str(SCRIPTS / "prepare_controller_characterization.py"),
            "--native-root", str(native), "--source-manifest", str(source_path), "--asset-manifest", str(assets_path),
            "--authorization", str(auth_path), "--output", str(child_output), "--gpu-uuid", binding["gpu_uuid"],
            "--pci-bus-id", binding["pci_bus_id"], "--kit-index", "0", "--cuda-ordinal", "0",
            "--shutdown-mode", "stock-fast", "--parent-log", str(parent_log), "--launch"]
    argv.extend(("--plan", str(plan_path), "--first-probe-result", str(probe_path), "--first-probe-analysis", str(analysis_path), "--operator-review-directory", str(output / "operator_reviews"), "--program-binding", str(output / "program_binding.json")))
    if asset_assessment is not None:
        argv.extend(("--asset-diagnostic-assessment", str(asset_assessment)))
    record = {"schema": "robodojo-controller-characterization-launch-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": argv, "environment": environment, "binding": binding, "source_file_sha256": source_hashes,
        "authorization_sha256": digest(auth_path), "dependency_binding": dependency_info,
        "compatibility_binding": compatibility_info,
        "plan_sha256": content_hash(plan), "plan_file_sha256": digest(plan_path), "first_probe": first_probe,
        "diagnostic_native_actions_planned": MAX_ACTIONS, "benchmark_attempt_started": False, "operator_visual_clearance_required": True,
        "preparation": preparation, "wall_time_limit_s": args.timeout_s, "minimum_free_mib": args.minimum_free_mib,
        "local_library_sha256": {path.name: digest(path) for path in library_path.iterdir() if path.is_file()},
        "shutdown_mode": "stock-fast", "full_plugin_cleanup_proven": False, "full_teardown_returned": False,
        "shutdown_limitation": "NVIDIA stock fast exit skips complete plugin/Python teardown; successful preclose and process/GPU release are required.",
        "automatic_retry": False, "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
        "standard_ready": False, "asset_fidelity_passed": False,
        "dependency_closure_complete": preparation["dependency_closure_complete"],
        "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
        "worker_identity": {"uid":WORKER_UID,"gid":WORKER_GID,"supplementary_groups":[],"parent_pid":os.getpid()},
        "worker_git_configuration": worker_git,
        "limitations": "Bounded local controller diagnostic; no automatic calibration/task-success claim. Separate Unix UID protects credentials; no filesystem namespace or network isolation; peak VRAM is sampled."}
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
                                    stderr=subprocess.STDOUT, **popen_identity())
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
        record.update(verify_controller_evidence(child_output, binding, initial, preparation, source_hashes,
                                             parent_log, python, authorization, plan, first_probe))
        if digest(plan_path) != record["plan_file_sha256"] or verify_first_probe(probe_path, analysis_path, plan) != first_probe:
            raise ValueError("Controller proposal/first-probe analysis changed")
        if digest(auth_path) != record["authorization_sha256"]:
            raise ValueError("Characterization authorization changed during this attempt")
        if digest(Path(worker_git["path"])) != worker_git["sha256"]:
            raise ValueError("Fresh worker Git configuration changed")
        if any(digest(Path(name)) != value for name, value in source_hashes.items()):
            raise ValueError("A launcher/child/helper changed during this attempt")
        if verify_manifests(native, source_path, assets_path, pins(), asset_assessment) != preparation:
            raise ValueError("Native source/assets changed during this attempt")
        if verify_dependencies(python, args.install_record.resolve(), args.dependency_record.resolve()) != dependency_info:
            raise ValueError("Package metadata or dependency records changed during this attempt")
        if compatibility_path is not None and verify_compatibility_record(python, compatibility_path, dependency_info) != compatibility_info:
            raise ValueError("Compatibility metadata or installation record changed during this attempt")
        record["bindings_unchanged_after_controller_characterization"] = True
    except Exception as error:
        record["evidence_validation_error"] = f"{type(error).__name__}: {error}"
    record["shutdown_contract_satisfied"] = bool(record.get("preclose_ready_verified") is True
        and record.get("returncode") == 0 and not sent and "termination" not in record)
    record["clean_shutdown_confirmed"] = bool(record["shutdown_contract_satisfied"]
        and record.get("gpu_processes_released") is True and record.get("gpu_memory_released") is True
        and not record["remaining_owned_pids"] and "cleanup_error" not in record)
    record["diagnostic_completed"] = bool(record["clean_shutdown_confirmed"]
        and record.get("native_render_verified") is True and record.get("diagnostic_actions_verified") is True
        and record.get("bindings_unchanged_after_controller_characterization") is True
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
