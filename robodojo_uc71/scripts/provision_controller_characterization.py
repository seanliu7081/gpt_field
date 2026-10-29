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
