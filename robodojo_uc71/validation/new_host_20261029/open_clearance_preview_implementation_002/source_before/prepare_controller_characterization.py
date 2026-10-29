#!/usr/bin/env python3
"""One bounded, separately scoped controller diagnostic; no benchmark lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import traceback
import uuid

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.worker_isolation import enforce_child_boundary
BOUNDARY_PROOF = None
if __name__ == "__main__" and "--launch" in sys.argv:
    BOUNDARY_PROOF = enforce_child_boundary(credential_store="/workspace/gpt_field/.private/chatgpt_plan",
        parent_pid=int(os.environ["UC71_CONTROLLER_PARENT_PID"]))
import numpy as np
from robodojo_uc71.controller import measured_tcp_pose, orientation_error_world
from robodojo_uc71.controller_characterization import (validate_controller_probe_proposal,
    validate_open_clearance_preview_proposal,
    physical_arm_states, stage_controller, stage_joint_command, describe_sample, summarize_stage)
from robodojo_uc71.native_action_observer import observe_native_joint_action_once
from robodojo_uc71.native_characterization import capture_characterization_state, content_hash
from robodojo_uc71.native_config import (ResetOnlyClient, add_native_import_paths,
    build_native_preview_config, install_preview_adapters, validate_native_preview_config)
from robodojo_uc71.native_measurements import capture_paused_physics_state
from new_host_camera import query_cuda_identity, renderer_table_identity, write_durable_json, normalize_pci
from prepare_native_characterization import (capture_stage, install_characterization_guards,
    program_files as first_probe_program_files)
from prepare_native_preview import verify_manifests
from provision_native_common import digest, pins

MAX_ACTIONS = 960
SCOPE = {"scope": "diagnostic_native_actions", "benchmark_attempt_started": False,
    "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
    "calibration_passed": False, "standard_ready": False, "asset_fidelity_passed": False}


def diagnostic_contract(kind="controller-characterization"):
    if kind not in ("controller-characterization", "open-clearance-preview"):
        raise ValueError("Unknown native diagnostic kind")
    preview = kind == "open-clearance-preview"
    schema = "robodojo-open-clearance-preview" if preview else "robodojo-controller-characterization"
    prefix = "clearance_preview" if preview else "controller"
    return {"kind": kind, "open_only": preview, "actions": 260 if preview else MAX_ACTIONS,
        "stages": 8 if preview else 39, "images": 27 if preview else 120,
        **{name+"_schema": schema+"-"+name+"-1" for name in
           ("authorization", "launch", "preparation", "result", "preclose")},
        "result_filename": prefix+"_result.json", "preclose_filename": prefix+"_preclose_ready.json",
        "result_hash_key": prefix+"_result_sha256", "failure_filename": prefix+"_failure.json",
        "progress_filename": prefix+"_progress.jsonl",
        "authorized_child_env": "UC71_OPEN_CLEARANCE_PREVIEW_AUTHORIZED_CHILD" if preview else
            "UC71_CONTROLLER_CHARACTERIZATION_AUTHORIZED_CHILD",
        "operator_visual_clearance_required": not preview, "max_wall_time_s": 900 if preview else 1800}


def validate_diagnostic_plan(plan, diagnostic_kind="controller-characterization"):
    contract = diagnostic_contract(diagnostic_kind)
    if contract["open_only"]:
        validate_open_clearance_preview_proposal(plan)
    else:
        validate_controller_probe_proposal(plan)
    if len(plan["stages"]) != contract["stages"] or plan["maximum_native_actions"] != contract["actions"]:
        raise ValueError("Native diagnostic plan accounting differs from its explicit mode")


def require_open_only_action(action):
    """Inspect the actual normalized native action inside its authorization callback."""
    for arm in ("left", "right"):
        values = action.get(arm+"_ee_joint_state")
        if type(values) is not list or len(values) != 1 or type(values[0]) is not float or values[0] != 1.:
            raise RuntimeError("Open-only preview cannot dispatch a closing gripper command")


def verify_preview_endpoint(samples, stage, plan):
    """Accept only the predeclared last five measured states; never add settling."""
    limits = plan["endpoint_validation"]
    if len(samples) < limits["samples"]+1:
        raise RuntimeError("Preview endpoint lacks five acknowledged measured states")
    target = np.asarray(stage["target_tool_pose_wxyz"])
    position_errors, orientation_errors, finger_errors = [], [], []
    for sample in samples[-limits["samples"]:]:
        pose = np.asarray(sample["arms"][stage["active_arm"]]["tool_pose_wxyz"])
        position_errors.append(float(np.linalg.norm(pose[:3]-target[:3])))
        orientation_errors.append(float(np.linalg.norm(orientation_error_world(pose[3:], target[3:]))))
        for arm in ("left", "right"):
            fingers = np.asarray(sample["arms"][arm]["finger_positions"])
            if fingers.shape != (2,) or not np.all(np.isfinite(fingers)):
                raise RuntimeError("Preview endpoint has invalid measured finger positions")
            finger_errors.append(float(np.max(np.abs(fingers-.044))))
    metrics = {"maximum_position_error_m": max(position_errors),
        "maximum_orientation_error_rad": max(orientation_errors),
        "maximum_open_finger_error_m": max(finger_errors)}
    if (not all(np.isfinite(v) for v in metrics.values())
            or metrics["maximum_position_error_m"] > limits["position_tolerance_m"]
            or metrics["maximum_orientation_error_rad"] > limits["orientation_tolerance_rad"]
            or metrics["maximum_open_finger_error_m"] > limits["open_finger_tolerance_m"]):
        raise RuntimeError("Preview endpoint did not satisfy fixed five-state stable/open checks")
    return {"passed": True, "samples_checked": limits["samples"], "extra_settling_actions": 0, **metrics}


def private_runtime_paths(output):
    """Exact writable paths; no installed package or shared /tmp cache is used."""
    root = output / "kit-portable"
    tokens = {}
    for kind in ("cache", "data", "logs"):
        tokens[kind] = str(root / kind / "kit")
        tokens["omni_" + kind] = str(root / kind)
        tokens["omni_global_" + kind] = str(root / kind)
    return {"tmpdir": str(output.parents[1] / "tmp"), "portable_root": str(root),
            "tokens": tokens, "isaaclab_log_directory": str(output.parents[1] / "tmp/isaaclab/logs")}


def prepare_private_runtime_paths(output):
    expected = private_runtime_paths(output)
    tmp = Path(expected["tmpdir"])
    info = tmp.lstat()
    if (os.geteuid() != 65534 or tmp.is_symlink() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 65534 or info.st_gid != 65534 or stat.S_IMODE(info.st_mode) != 0o700
            or os.environ.get("TMPDIR") != str(tmp) or Path(tempfile.gettempdir()) != tmp):
        raise RuntimeError("Worker-private TMPDIR did not verify")
    for path in sorted(set(expected["tokens"].values())):
        Path(path).mkdir(parents=True, mode=0o700, exist_ok=True)
        if Path(path).resolve() != Path(path) or Path(path).stat().st_uid != 65534:
            raise RuntimeError("Worker-private Kit path did not verify")
    # This does not load Kit or configure logging; it proves actual worker writes.
    with tempfile.TemporaryFile(dir=tmp) as stream:
        stream.write(b"private-temp-path-check"); stream.flush()
    return {"schema": "robodojo-private-runtime-path-preparation-1", "pid": os.getpid(),
            "uid": os.geteuid(), **expected, "temporary_file_write_passed": True,
            "kit_runtime_resolution_checked": False}


def private_kit_arguments(output):
    paths = private_runtime_paths(output)
    # Kit107.3 documents a separate portable-root argument and /app/tokens overrides:
    # https://docs.omniverse.nvidia.com/kit/docs/kit-manual/107.3.0/guide/tokens.html
    return ["--portable", "--portable-root", paths["portable_root"],
            *(f"--/app/tokens/{key}={value}" for key, value in paths["tokens"].items())]


def verify_effective_runtime_paths(output, resolve):
    expected = private_runtime_paths(output)
    actual = {key: resolve("${" + key + "}") for key in expected["tokens"]}
    if actual != expected["tokens"] or str(Path(tempfile.gettempdir())) != expected["tmpdir"]:
        raise RuntimeError("Effective Kit token or Python temporary directory escaped the worker paths")
    return {"schema": "robodojo-private-runtime-paths-1", "pid": os.getpid(),
            "uid": os.geteuid(), "monotonic_s": time.monotonic(), **expected,
            "resolved_tokens": actual, "verified_before_native_reset": True}


def verify_native_logging_paths(output, handlers):
    import logging
    expected = Path(private_runtime_paths(output)["isaaclab_log_directory"])
    paths = [Path(h.baseFilename) for h in handlers if isinstance(h, logging.FileHandler)]
    if (not paths or any(p.parent != expected or p.resolve() != p or not p.is_file()
            or p.stat().st_uid != 65534 or not p.name.startswith("isaaclab_") for p in paths)):
        raise RuntimeError("IsaacLab file logging escaped the worker TMPDIR")
    return {"schema": "robodojo-private-native-logging-1", "pid": os.getpid(),
            "uid": os.geteuid(), "monotonic_s": time.monotonic(),
            "directory": str(expected), "file_paths": [str(p) for p in paths],
            "verified_after_native_reset": True}


def program_files():
    return [*first_probe_program_files(), *(BASE / "scripts" / name for name in
        ("prepare_controller_characterization.py", "provision_controller_characterization.py")),
        *(BASE / "src/robodojo_uc71" / name for name in ("controller.py", "controller_characterization.py", "worker_isolation.py"))]


def verify_program_binding(path):
    value = json.loads(path.read_text())
    if set(value) != {str(p) for p in program_files()} or any(digest(Path(p)) != h for p, h in value.items()):
        raise ValueError("Controller diagnostic program binding changed")
    return value


def accumulated_time_tolerance(actions):
    return actions*10*abs(float(np.float32(.004))-.004)+1e-7


def verify_first_probe(result_path, analysis_path, plan):
    """Require completed native accounting and independent source-bound analysis."""
    result = json.loads(result_path.read_text())
    required = {"diagnostic_completed": True, "clean_shutdown_confirmed": True,
        "bindings_unchanged_after_characterization": True, "gpu_processes_released": True,
        "gpu_memory_released": True, "returncode": 0, "diagnostic_native_actions": 37,
        "diagnostic_physics_substeps": 370, "benchmark_attempt_started": False,
        "full_test_authorized": False, "model_calls": 0}
    if (any(type(result.get(k)) is not type(v) or result.get(k) != v for k, v in required.items())
            or result.get("termination_signals") != [] or result.get("remaining_owned_pids") != []
            or result.get("binding") != plan["binding"] or "termination" in result or "cleanup_error" in result):
        raise ValueError("A completed clean first native probe is required")
    fixed = json.loads(Path(plan["fixed_probe_plan_path"]).read_text())
    if result.get("plan_sha256") != content_hash(fixed):
        raise ValueError("First probe used another initial robot plan")
    analysis = json.loads(analysis_path.read_text())
    # Exact independent analysis contract is checked here; no manual READY flag.
    if (analysis.get("schema") != "robodojo-native-joint-probe-analysis-1"
            or analysis.get("source_binding_verified") is not True
            or analysis.get("all_recorded_pose_checks_passed") is not True
            or analysis.get("kinematics_geometry_passed") is not True
            or analysis.get("source_launch_result_sha256") != digest(result_path)
            or analysis.get("geometry_candidate_sha256") != digest(Path(plan["geometry_candidate_path"]))
            or analysis.get("full_calibration_passed") is not False
            or analysis.get("controller_tracking_validated") is not False
            or analysis.get("full_test_authorized") is not False
            or analysis.get("standard_ready") is not False):
        raise ValueError("Reviewed first-probe native accounting/FK/Jacobian analysis is required")
    sources = analysis.get("source_files", [])
    if (not sources or any(not isinstance(row, dict) or set(row) != {"path", "sha256"}
            or digest(Path(row["path"])) != row["sha256"] for row in sources)):
        raise ValueError("First-probe analysis source artifacts changed")
    return {"launch_result_path": str(result_path), "launch_result_sha256": digest(result_path),
        "analysis_path": str(analysis_path), "analysis_sha256": digest(analysis_path),
        "fixed_probe_plan_sha256": content_hash(fixed)}


def check_authorization(path, plan, binding, *, first_probe, launch, output_parent,
                        diagnostic_kind="controller-characterization"):
    contract = diagnostic_contract(diagnostic_kind)
    value = json.loads(path.read_text())
    preparation_only = value.get("preparation_only") is True
    required = {"schema": contract["authorization_schema"],
        "plan_sha256": content_hash(plan), "gpu_uuid": binding["gpu_uuid"],
        "diagnostic_native_actions_authorized": not preparation_only, "maximum_native_actions": contract["actions"],
        "model_calls_authorized": False, "benchmark_attempt_started": False, "full_test_authorized": False,
        "automatic_retry": False, "automatic_tuning": False, "shutdown_mode": "stock-fast",
        "operator_visual_clearance_required": contract["operator_visual_clearance_required"],
        "operator_wait_timeout_s": 0 if contract["open_only"] else 120,
        "checkpoint_position_tolerance_m": .002, "checkpoint_orientation_tolerance_rad": .02,
        "open_finger_tolerance_m": .001,
        "first_probe_result_sha256": first_probe["launch_result_sha256"],
        "first_probe_analysis_sha256": first_probe["analysis_sha256"],
        "program_sha256": {str(p):digest(p) for p in program_files()}}
    if contract["open_only"]:
        required.update(diagnostic_kind=diagnostic_kind, open_only=True,
            endpoint_stability_samples=5, operator_review_after_exit=True,
            candidate_sha256=plan["candidate"]["sha256"])
    if (plan["binding"] != binding or any(type(value.get(k)) is not type(v) or value.get(k) != v for k,v in required.items())
            or type(value.get("maximum_wall_time_s")) is not int or not 30 <= value["maximum_wall_time_s"] <= contract["max_wall_time_s"]
            or normalize_pci(value.get("pci_bus_id", "")) != normalize_pci(binding["pci_bus_id"])
            or Path(value.get("attempt_output", "")).resolve() != output_parent.resolve()
            or not isinstance(value.get("authorization_basis"), str) or not value["authorization_basis"].strip()
            or not isinstance(value.get("user_message"), str) or not value["user_message"].strip()):
        raise ValueError("Exact separate controller diagnostic authorization is required")
    if launch and (preparation_only or os.environ.get(contract["authorized_child_env"]) != "1"):
        raise ValueError("Controller diagnostic requires its bounded parent")
    return {"path": str(path), "sha256": digest(path), "plan_sha256": content_hash(plan),
        "diagnostic_kind": diagnostic_kind,
        "maximum_wall_time_s": value["maximum_wall_time_s"], "maximum_native_actions": contract["actions"],
        "diagnostic_native_actions_authorized": not preparation_only, "full_test_authorized": False,
        "benchmark_attempt_started": False, "model_calls_authorized": False,
        "operator_visual_clearance_required": contract["operator_visual_clearance_required"],
        "operator_wait_timeout_s": 0 if contract["open_only"] else 120,
        "first_probe_result_sha256": first_probe["launch_result_sha256"],
        "first_probe_analysis_sha256": first_probe["analysis_sha256"]}


def measured_envelope(measurement, states, plan, *, initial=False, inactive_hold=None):
    limits = plan["candidate_abort_envelopes"]
    for arm in ("left", "right"):
        state = states[arm]; pose = measured_tcp_pose(state)
        baseline = plan["home"][arm]
        q = np.asarray(state.joint_positions); dq = np.asarray(state.joint_velocities)
        if (max(abs(q-np.asarray(baseline["joint_positions"]))) > (2e-4 if initial else limits["joint_distance_from_home_rad"])
                or max(abs(dq)) > limits["measured_joint_velocity_rad_s"]
                or np.linalg.norm(pose[0]-baseline["tool_pose_wxyz"][:3]) > (.002 if initial else limits["tool_distance_from_home_m"])
                or np.linalg.norm(orientation_error_world(pose[1], baseline["tool_pose_wxyz"][3:])) > (.02 if initial else limits["tool_orientation_from_home_rad"])):
            raise RuntimeError("Measured controller diagnostic robot envelope exceeded")
        finger = np.asarray(measurement["arms"][arm]["finger_positions"])
        if np.any(finger < -limits["observed_finger_limit_excess_m"]) or np.any(finger > .044+limits["observed_finger_limit_excess_m"]):
            raise RuntimeError("Observed finger physical-limit envelope exceeded")
        if inactive_hold and arm in inactive_hold and max(abs(q-inactive_hold[arm])) > limits["inactive_joint_distance_from_latched_hold_rad"]:
            raise RuntimeError("Inactive arm drifted outside its latched diagnostic hold")


def validate_operator_review(review, request, request_sha256):
    expected = {"schema": "robodojo-controller-clearance-review-1", "request_sha256": request_sha256,
        "plan_sha256": request["plan_sha256"], "stage_index": request["stage_index"],
        "active_arm": request["active_arm"], "native_action_counter": request["native_action_counter"],
        "image_sha256": request["image_sha256"], "reviewer": "root/operator",
        "full_test_authorized": False, "benchmark_attempt_started": False,
        "object_truth_used": False}
    if (any(type(review.get(k)) is not type(v) or review.get(k) != v for k,v in expected.items())
            or review.get("decision") not in ("clear_for_empty_gripper_timing", "ambiguous_or_not_clear")
            or not isinstance(review.get("reason"), str) or not review["reason"].strip()):
        raise ValueError("Operator review does not bind the exact fresh clearance request")
    return review["decision"] == "clear_for_empty_gripper_timing"


def paused_state(env, counters):
    result = capture_paused_physics_state(env, counters["native_sim_step_calls"])
    result.update(native_physics_counter=env.sim._sim_step_counter, native_action_counter=list(env.take_action_cnt))
    return result


def wait_for_operator_clearance(env, counters, output, stage, capture, preparation, progress, verify_runtime):
    """Bounded logs-only wait: no app, render, or simulation updates."""
    directory = output / f"clearance_{stage['active_arm']}"
    directory.mkdir(exist_ok=False)
    before = paused_state(env, counters)
    request = {"schema": "robodojo-controller-clearance-request-1", "nonce": uuid.uuid4().hex,
        "plan_sha256": preparation["plan_sha256"], "stage_index": stage["index"], "active_arm": stage["active_arm"],
        "native_action_counter": list(env.take_action_cnt), "capture": capture,
        "image_sha256": {row["path"]: row["sha256"] for row in capture["images"]},
        "physics_state": before, "operator_wait_timeout_s": 120, **SCOPE}
    write_durable_json(directory / "request.json", request)
    request_hash = digest(directory / "request.json")
    review_path = Path(preparation["operator_review_directory"]) / f"{stage['active_arm']}.json"
    started = time.monotonic(); last_log = -30.; accepted = False; review_hash = None; failure = None
    progress("awaiting_operator_clearance", request=str(directory / "request.json"), review=str(review_path), active_arm=stage["active_arm"])
    try:
        while True:
            verify_runtime()
            elapsed = time.monotonic()-started
            if elapsed >= 120: raise TimeoutError("Operator clearance was not provided within120seconds")
            if review_path.exists():
                info = review_path.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_gid != 65534
                        or stat.S_IMODE(info.st_mode) != 0o640 or info.st_nlink != 1):
                    raise ValueError("Root-owned read-only operator review with mode0640/gid65534 required")
                payload = review_path.read_bytes(); review = json.loads(payload)
                accepted = validate_operator_review(review, request, request_hash)
                review_hash = hashlib.sha256(payload).hexdigest()
                write_durable_json(directory / "review_consumed.json", review)
                if not accepted: raise RuntimeError("Operator clearance was ambiguous or not clear; diagnostic stopped")
                break
            if elapsed-last_log >= 30:
                progress("operator_clearance_wait", active_arm=stage["active_arm"], elapsed_s=elapsed)
                last_log = elapsed
            time.sleep(1.)
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        after = paused_state(env, counters)
        unchanged = before == after
        evidence = {"schema": "robodojo-controller-clearance-wait-1", "request_path": str(directory / "request.json"),
            "request_sha256": request_hash, "review_path": str(review_path), "review_sha256": review_hash,
            "review_consumed_sha256": digest(directory / "review_consumed.json") if review_hash else None,
            "start_monotonic_s": started, "end_monotonic_s": time.monotonic(),
            "before": before, "after": after, "physics_unchanged": unchanged,
            "accepted": accepted, "failure": failure, "simulation_updates_during_wait": 0, **SCOPE}
        write_durable_json(directory / "wait_result.json", evidence)
        if not unchanged: raise RuntimeError("Physics moved during the operator clearance wait")
    return {"path": str(directory / "wait_result.json"), "sha256": digest(directory / "wait_result.json"),
        "review_path": str(review_path), "review_sha256": review_hash, "active_arm": stage["active_arm"]}


def execute_stages(env, counters, output, plan, preparation, permitted, progress, verify_runtime,
                   *, diagnostic_kind="controller-characterization"):
    contract = diagnostic_contract(diagnostic_kind)
    maximum = contract["actions"]
    geometry = json.loads(Path(plan["geometry_candidate_path"]).read_text())
    captures = [capture_stage(env, counters, output, "before", rgb=True)]
    initial = captures[0]
    current = capture_characterization_state(env)
    states = physical_arm_states(current, geometry)
    measured_envelope(current, states, plan, initial=True)
    if contract["open_only"]:
        for arm in ("left", "right"):
            fingers = np.asarray(current["arms"][arm]["finger_positions"])
            if fingers.shape != (2,) or not np.all(np.isfinite(fingers)) or np.max(np.abs(fingers-.044)) > .001:
                raise RuntimeError("Open-only preview must begin with both measured grippers open")
    actions, stage_records, clearances = [], [], {}
    for stage in plan["stages"]:
        verify_runtime()
        active = stage["active_arm"]
        states = physical_arm_states(current, geometry)
        entry_index = len(actions)
        inactive = {arm: np.asarray(states[arm].joint_positions).copy() for arm in ("left", "right") if arm != active}
        if contract["open_only"] and (stage["kind"] != "translation"
                or stage["active_gripper_command"] != 1. or stage["other_gripper_command"] != 1.
                or stage["operator_clearance_required_before"] is not False):
            raise RuntimeError("Open-only preview contains a closing or operator-wait stage")
        if stage["kind"] == "visual_checkpoint":
            pose = measured_tcp_pose(states[active]); target = stage["target_tool_pose_wxyz"]
            if np.linalg.norm(pose[0]-target[:3]) > .002 or np.linalg.norm(orientation_error_world(pose[1], target[3:])) > .02:
                raise RuntimeError("Retreat did not reach the declared clearance pose")
            capture = capture_stage(env, counters, output, f"stage_{stage['index']:03d}", rgb=True)
            captures.append(capture)
            clearances[active] = wait_for_operator_clearance(env, counters, output, stage, capture, preparation, progress, verify_runtime)
            stage_records.append({"stage_index": stage["index"], "label": stage["label"], "native_actions": 0,
                "first_action_index": entry_index+1, "last_action_index": entry_index,
                "capture": capture, "clearance": clearances[active]})
            continue
        if stage["kind"] == "gripper_close" and active not in clearances:
            raise RuntimeError("Fresh operator clearance required before closing")
        if "return_from_retreat_open" in stage["label"]:
            if max(abs(np.asarray(current["arms"][active]["finger_positions"])-.044)) > .001:
                raise RuntimeError("Measured fingers are not open before the declared return")
        controller = None if stage["kind"] == "joint_hold" else stage_controller(stage, states)
        samples = [describe_sample(current["native_physics_time_s"], states,
                    {arm: current["arms"][arm]["finger_positions"] for arm in ("left", "right")})]
        for local_index in range(1, stage["native_actions"]+1):
            index = len(actions)+1
            if index > maximum: raise RuntimeError("Controller diagnostic native action ceiling exceeded")
            before = current
            states = physical_arm_states(before, geometry)
            measured_envelope(before, states, plan, inactive_hold=inactive)
            if controller is None:
                action = {f"{arm}_arm_joint_state": list(plan["home"][arm]["joint_positions"]) for arm in ("left", "right")}
                action.update({f"{arm}_ee_joint_state": [1.] for arm in ("left", "right")}); predicted = None
            else:
                action, predicted = stage_joint_command(stage, controller, states)
            for arm in ("left", "right"):
                if max(abs(np.asarray(action[f"{arm}_arm_joint_state"])-states[arm].joint_positions)) > .05+1e-12:
                    raise RuntimeError("Candidate joint target exceeds per-action lead")
            def authorize(actual, native_counter):
                verify_runtime()
                if contract["open_only"]:
                    require_open_only_action(actual)
                if actual != action or native_counter != index-1: raise RuntimeError("Native action identity changed")
                if stage["kind"] == "gripper_close" and digest(Path(clearances[active]["review_path"])) != clearances[active]["review_sha256"]:
                    raise RuntimeError("Operator clearance was changed or revoked")
                return True
            with permitted():
                accounting = observe_native_joint_action_once(env, action, verify_authorization=authorize)
            after = None; measurement_error = None
            try:
                after = capture_characterization_state(env)
                states_after = physical_arm_states(after, geometry)
                measured_envelope(after, states_after, plan, inactive_hold=inactive)
            except Exception as error:
                measurement_error = type(error).__name__
            receipt = {"schema": "robodojo-controller-native-action-1", **SCOPE,
                "native_action_index": index, "stage_index": stage["index"], "stage_action_index": local_index,
                "plan_sha256": preparation["plan_sha256"], "authorization_sha256": preparation["authorization"]["sha256"],
                "target": action, "controller_prediction": predicted, "before": before, "after": after,
                "measured_envelope_passed": measurement_error is None, "measurement_validation_error": measurement_error,
                "native_accounting": accounting, "clearance": clearances.get(active) if stage["kind"] == "gripper_close" else None}
            receipt_path = output / f"native_action_{index:04d}.json"
            write_durable_json(receipt_path, receipt)
            actions.append({"index": index, "stage_index": stage["index"],
                "receipt": {"path": receipt_path.name, "sha256": digest(receipt_path)}})
            if measurement_error: raise RuntimeError("Post-action measured envelope/measurement failed; native receipt preserved")
            verify_runtime()
            current = after
            samples.append(describe_sample(current["native_physics_time_s"], states_after,
                {arm: current["arms"][arm]["finger_positions"] for arm in ("left", "right")}))
            if accounting["native_end_observed"]: raise RuntimeError("Native terminal callback ended the diagnostic early")
            if local_index == 1 or local_index % 10 == 0:
                progress("controller_action_recorded", native_action_index=index, stage_index=stage["index"],
                    native_physics_time_s=current["native_physics_time_s"])
        capture = capture_stage(env, counters, output, f"stage_{stage['index']:03d}", rgb=True)
        captures.append(capture)
        if capture["native_physics_time_s"] != current["native_physics_time_s"]:
            raise RuntimeError("Stage boundary capture differs from acknowledged action")
        analysis = summarize_stage(stage, samples)
        endpoint = verify_preview_endpoint(samples, stage, plan) if contract["open_only"] else None
        if endpoint is not None:
            analysis["endpoint_validation"] = endpoint
        analysis_path = output / f"stage_{stage['index']:03d}_analysis.json"
        write_durable_json(analysis_path, analysis)
        stage_records.append({"stage_index": stage["index"], "label": stage["label"], "native_actions": stage["native_actions"],
            "first_action_index": entry_index+1, "last_action_index": len(actions), "capture": capture,
            "analysis": {"path": analysis_path.name, "sha256": digest(analysis_path)}})
        progress("controller_stage_recorded", stage_index=stage["index"], label=stage["label"], native_action_index=len(actions))
    elapsed = current["native_physics_time_s"]-initial["native_physics_time_s"]
    if (len(actions) != maximum or env.take_action_cnt != [maximum]
            or current["native_physics_counter"]-initial["native_physics_counter"] != maximum*10
            or counters["native_sim_step_calls"]-initial["native_sim_step_calls"] != maximum*10
            or not math.isclose(elapsed, maximum*.04, rel_tol=0, abs_tol=accumulated_time_tolerance(maximum))):
        raise RuntimeError("Controller diagnostic total native cadence/accounting differs")
    return {"actions": actions, "stages": stage_records, "captures": captures, "clearances": clearances,
        "diagnostic_native_actions": maximum, "diagnostic_physics_substeps": maximum*10,
        "diagnostic_simulated_time_s": elapsed, "nominal_simulated_time_s": maximum*.04,
        "accumulated_dt_tolerance_s": accumulated_time_tolerance(maximum),
        "float32_dt_reference_s": float(np.float32(.004)), "image_count": sum(len(x["images"]) for x in captures)}


def run_controller_characterization(args, config, preparation, binding, output, progress) -> int:
    contract = diagnostic_contract(args.diagnostic_kind)
    if BOUNDARY_PROOF is None or os.geteuid() != 65534:
        raise RuntimeError("Isolated native worker boundary must be enforced before startup")
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
        kit_args = [*private_kit_arguments(output),
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
        import carb.tokens
        runtime_paths = verify_effective_runtime_paths(output, carb.tokens.get_tokens_interface().resolve)
        write_durable_json(output / "runtime_paths.json", runtime_paths)
        progress("private_runtime_paths_verified")
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
        import logging
        logging_paths = verify_native_logging_paths(output, logging.getLogger().handlers)
        write_durable_json(output / "native_logging_paths.json", logging_paths)
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
        def verify_runtime():
            validate_native_preview_config(config)
            if (digest(args.plan) != preparation["plan_file_sha256"]
                    or digest(args.authorization) != preparation["authorization"]["sha256"]
                    or digest(args.first_probe_result) != preparation["first_probe"]["launch_result_sha256"]
                    or digest(args.first_probe_analysis) != preparation["first_probe"]["analysis_sha256"]
                    or any(digest(Path(path)) != expected for path, expected in preparation["source_file_sha256"].items())
                    or (contract["open_only"] and digest(Path(plan["candidate"]["path"])) != plan["candidate"]["sha256"])
                    or type(env.model_client) is not ResetOnlyClient or env.model_client.reset_calls != 1
                    or env.model_client.closed or env.robot_manager.planner or env.robot_manager.ik_solver
                    or env.device != "cpu" or env.use_fabric is not False or env.num_envs != 1
                    or list(env.env_seeds) != [0] or digest(selected) != preparation["layout_sha256"]
                    or monitor.is_fatal() or monitor.get_broken_envs()):
                raise RuntimeError("Controller diagnostic runtime/source/authorization binding changed")
            check_authorization(args.authorization, plan, binding, first_probe=preparation["first_probe"],
                                launch=True, output_parent=output.parents[1], diagnostic_kind=args.diagnostic_kind)
        diagnostic = execute_stages(env, counters, output, plan, preparation, permitted, progress, verify_runtime,
                                    diagnostic_kind=args.diagnostic_kind)
        graphics = sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
            if len(line.split(maxsplit=5)) == 6 and re.search(r'/(?:libcuda|libnvidia|libGLX|libEGL|libGLU|libvulkan)', line.split(maxsplit=5)[5])})
        write_durable_json(output / "loaded_graphics_library_paths.json", {"pid": os.getpid(), "paths": graphics, "source": "/proc/self/maps"})
        result = {"schema": contract["result_schema"], **SCOPE, **diagnostic,
            "diagnostic_kind": args.diagnostic_kind,
            "diagnostic_completed": not contract["open_only"],
            "controller_characterization_completed": not contract["open_only"],
            "clearance_preview_completed": contract["open_only"],
            "open_only": contract["open_only"], "completed_at_monotonic_s": time.monotonic(),
            "binding": binding, "shutdown_mode": "stock-fast", "plan_sha256": preparation["plan_sha256"],
            "plan_file_sha256": preparation["plan_file_sha256"],
            "authorization_sha256": preparation["authorization"]["sha256"], "first_probe": preparation["first_probe"],
            "preparation_sha256": digest(output / "preparation.json"),
            "loaded_module_provenance_sha256": digest(output / "loaded_module_provenance.json"),
            "worker_isolation_sha256": digest(output / "worker_isolation.json"),
            "runtime_paths_sha256": digest(output / "runtime_paths.json"),
            "native_logging_paths_sha256": digest(output / "native_logging_paths.json"),
            "loaded_graphics_library_paths_sha256": digest(output / "loaded_graphics_library_paths.json"),
            "layout_path": str(selected), "layout_sha256": digest(selected),
            "native_take_action_count": list(env.take_action_cnt), "reset_calls": 1, "native_client_reset_calls": 1,
            "native_setup_and_diagnostic_counts": counters, "task_success_assessed": False,
            "dependency_closure_complete": preparation["dependency_closure_complete"],
            "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
            "policy_action_methods_guarded": guarded_methods, "adaptations": adaptations,
            "physics_device": env.device, "use_fabric": env.use_fabric,
            "native_planner_instances": 0, "native_ik_solver_instances": 0, "reward_predicates_registered": True,
            "automatic_configuration_selection": False,
            "limitations": "Bounded local controller diagnostic only; sampled traces do not certify loaded contact behavior or workspace-wide tracking."}
        if contract["open_only"]:
            result["candidate"] = plan["candidate"]
            result["endpoint_validation"] = plan["endpoint_validation"]
        write_durable_json(output / contract["result_filename"], result)
        progress("controller_diagnostic_saved", diagnostic_native_actions=contract["actions"], images=result["image_count"])
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_durable_json(output / contract["failure_filename"], failure)
        progress("controller_failed", error=failure)
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
        evidence = {"schema": contract["preclose_schema"], "ready_for_shutdown": ready,
                    "diagnostic_kind": args.diagnostic_kind, "open_only": contract["open_only"],
                    "diagnostic_completed": ready and not contract["open_only"],
                    "controller_characterization_completed": ready and not contract["open_only"],
                    "clearance_preview_completed": ready and contract["open_only"],
                    "diagnostic_native_actions": env.take_action_cnt[0] if env is not None else 0,
                    "native_setup_and_diagnostic_counts": locals().get("counters"),
                    "benchmark_attempt_started": False, "full_test_authorized": False,
                    "standard_ready": False, "asset_fidelity_passed": False,
                    "dependency_closure_complete": preparation["dependency_closure_complete"],
                    "asset_diagnostic_assessment": preparation["asset_diagnostic_assessment"],
                    "monotonic_s": time.monotonic(), "child_pid": os.getpid(), "binding": binding,
                    "shutdown_mode": args.shutdown_mode, "no_pending_exception": no_pending_exception,
                    "native_env_close_returned": env is not None and not any(e["phase"] == "native_env_close" for e in cleanup_errors),
                    "physx_monitor_healthy": healthy, "cleanup_errors": cleanup_errors,
                    contract["result_hash_key"]: digest(output / contract["result_filename"]) if result else None,
                    "preparation_sha256": digest(output / "preparation.json"),
                    "image_count": sum(len(x["images"]) for x in result["captures"]) if result else 0,
                    "model_calls": 0, "benchmark_actions": 0, "full_plugin_cleanup_proven": False,
                    "full_teardown_returned": False}
        write_durable_json(output / contract["preclose_filename"], evidence)
        progress("before_simulation_app_close", ready_for_shutdown=ready)
        # Fast shutdown may exit here; parent must require ready + hashes + clean exit/resources.
        if app is not None:
            try:
                app.close()
                write_durable_json(output / ("clearance_preview_close_returned.json" if contract["open_only"] else "controller_close_returned.json"), {
                    "app_close_returned": True, "monotonic_s": time.monotonic(),
                    "shutdown_mode": args.shutdown_mode, "full_teardown_returned": args.shutdown_mode == "full"})
            except BaseException as error:
                cleanup_errors.append({"phase": "simulation_app_close", "type": type(error).__name__, "message": str(error)})
                write_durable_json(output / ("clearance_preview_close_failure.json" if contract["open_only"] else "controller_close_failure.json"), cleanup_errors[-1])
    return 0 if ready and not cleanup_errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "source-manifest", "asset-manifest", "authorization", "output",
                 "plan", "first-probe-result", "first-probe-analysis", "program-binding", "parent-log", "asset-diagnostic-assessment", "operator-review-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--kit-index", type=int, choices=(0,), required=True)
    parser.add_argument("--cuda-ordinal", type=int, choices=(0,), required=True)
    parser.add_argument("--shutdown-mode", choices=("stock-fast",), default="stock-fast")
    parser.add_argument("--diagnostic-kind", choices=("controller-characterization", "open-clearance-preview"),
                        default="controller-characterization")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    contract = diagnostic_contract(args.diagnostic_kind)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        parser.error("Characterization requires the reviewed single unmasked GPU")
    for key in ("native_root", "source_manifest", "asset_manifest", "authorization", "plan",
                "first_probe_result", "first_probe_analysis", "program_binding", "asset_diagnostic_assessment", "parent_log", "operator_review_directory"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    output = args.output.resolve()
    attempt = output.parents[1]
    if (output.exists() or any(c.isspace() for c in str(output)) or output != attempt / "worker/controller"
            or args.parent_log != attempt / "stdout_stderr.log" or args.operator_review_directory != attempt / "operator_reviews"):
        parser.error("Use a fresh child output and the parent's opened private log")
    binding = {"gpu_uuid": args.gpu_uuid, "pci_bus_id": args.pci_bus_id, "kit_index": 0, "cuda_ordinal": 0}
    plan = json.loads(args.plan.read_text())
    validate_diagnostic_plan(plan, args.diagnostic_kind)
    programs = verify_program_binding(args.program_binding)
    prior = verify_first_probe(args.first_probe_result, args.first_probe_analysis, plan)
    auth = check_authorization(args.authorization, plan, binding, first_probe=prior,
                               launch=args.launch, output_parent=attempt, diagnostic_kind=args.diagnostic_kind)
    preparation = verify_manifests(args.native_root, args.source_manifest, args.asset_manifest,
                                   pins(), args.asset_diagnostic_assessment)
    if preparation["camera_gate"]["gpu_uuid"] != binding["gpu_uuid"]:
        raise ValueError("Characterization GPU differs from reviewed rendering gate")
    output.mkdir(parents=True, exist_ok=False)
    private_paths = prepare_private_runtime_paths(output)
    if args.launch:
        if BOUNDARY_PROOF is None: raise RuntimeError("Native worker isolation proof is missing")
        write_durable_json(output / "worker_isolation.json", BOUNDARY_PROOF)
    os.chdir(output)
    os.environ["ROBODOJO_RUN_ID"] = output.name
    preparation["import_paths"] = add_native_import_paths(args.native_root)
    config = build_native_preview_config(args.native_root, output.name)
    from omegaconf import OmegaConf
    write_durable_json(output / "native_configuration.json", OmegaConf.to_container(config, resolve=True))
    preparation.update(schema=contract["preparation_schema"], authorization=auth,
        diagnostic_kind=args.diagnostic_kind, open_only=contract["open_only"],
        native_root=str(args.native_root), plan_path=str(args.plan), plan_sha256=content_hash(plan),
        operator_review_directory=str(args.operator_review_directory),
        plan_file_sha256=digest(args.plan), first_probe=prior, source_file_sha256=programs,
        native_configuration_sha256=digest(output / "native_configuration.json"),
        private_runtime_paths=private_paths,
        binding=binding, shutdown_mode="stock-fast", launch_requested=args.launch,
        diagnostic_native_actions_planned=contract["actions"],
        operator_visual_clearance_required=contract["operator_visual_clearance_required"],
        benchmark_attempt_started=False, full_test_authorized=False, model_calls=0, benchmark_actions=0)
    if contract["open_only"]:
        preparation.update(candidate=plan["candidate"], endpoint_validation=plan["endpoint_validation"])
    write_durable_json(output / "preparation.json", preparation)
    if not args.launch:
        print(json.dumps({"prepared": True, "simulator_starts": 0, "output": str(output)}), flush=True)
        return 0
    def progress(phase, **details):
        row = {"phase": phase, "monotonic_s": time.monotonic(), **details}
        with (output / contract["progress_filename"]).open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)
    return run_controller_characterization(args, config, preparation, binding, output, progress)


if __name__ == "__main__":
    raise SystemExit(main())
