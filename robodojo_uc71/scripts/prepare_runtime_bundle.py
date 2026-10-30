#!/usr/bin/env python3
"""Assemble a scoped runtime bundle only after completed measured diagnostics.

Default: validate inputs and an ephemeral candidate, including a separate static
UID65534 reader. --assemble publishes a fresh directory by atomic rename. This
does not create human approval, READY, an attempt manifest, or a simulator/model
process. A completed controller launch and a separate independent selection
review are mandatory; incomplete diagnostics cannot produce even a candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from robodojo_uc71.approval import VerifiedFileHashCache, canonical_sha256, file_sha256
from robodojo_uc71.controller import ControllerConfig
from robodojo_uc71.fragment_executor import MonitoringSettings
from robodojo_uc71.native_runtime_bundle import (CONTROLLER_DIAGNOSTIC_ACTIONS,
    CONTROLLER_DIAGNOSTIC_SUBSTEPS, CONTROLLER_PLAN_CONTENT_SHA256, CONTROLLER_PLAN_FILE_SHA256,
    EVIDENCE, PREPARATION_CHECKS, VerifiedRuntimeBundle)
from robodojo_uc71.schema import ExecutionLimits

SCOPE = "measured_robot_geometry_and_free_space_characterization"
DEFAULT_ROOT = BASE / "validation/new_host_20261029"
DEFAULTS = {
    "geometry": "tool_geometry_001/candidate.json",
    "joint_analysis": "joint_probe_analysis_001/analysis.json",
    "joint_projection_review": "joint_probe_projection_001/visual_review.json",
    "reset_reference": "reset_binding_001/reference.json",
    "reset_check": "reset_binding_001/fresh_probe001_check.json",
    "reset_closure": "reset_binding_closure_001/closure.json",
    "pause": "native_preview_002/preview/pause_validation.json",
    "asset_assessment": "native_assets002_diagnostic_assessment.json",
}
PROGRAM_NAMES = ("native_runtime_bundle.py", "approval.py", "controller.py", "fragment_executor.py",
    "native_measurements.py", "native_robot_inventory.py", "reset_binding.py", "schema.py")


class AssemblyError(ValueError):
    pass


def strict_pairs(rows):
    value = {}
    for key, item in rows:
        if key in value:
            raise AssemblyError("duplicate_json_key")
        value[key] = item
    return value


def read(path):
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=strict_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(AssemblyError("nonfinite_json")))
    if not isinstance(value, dict):
        raise AssemblyError("json_record_required")
    return value


def entry(path):
    path = Path(path).absolute()
    info = path.lstat()
    if path.resolve(strict=True) != path or not stat.S_ISREG(info.st_mode):
        raise AssemblyError("canonical_regular_evidence_required")
    return {"path": str(path), "sha256": file_sha256(path)}


def completed_controller(path):
    """Run this before opening future reviews or creating any output directory."""
    launch = read(path)
    positives = ("diagnostic_completed", "diagnostic_actions_verified", "native_render_verified",
        "preclose_ready_verified", "worker_isolation_verified", "operator_clearances_verified",
        "cuda_identity_verified", "renderer_identity_verified", "extension_startup_clean",
        "gpu_processes_released", "gpu_memory_released", "shutdown_contract_satisfied")
    if (launch.get("schema") != "robodojo-controller-characterization-launch-1"
            or any(launch.get(key) is not True for key in positives)
            or type(launch.get("returncode")) is not int or launch["returncode"] != 0
            or launch.get("termination_signals") != [] or launch.get("remaining_owned_pids") != []
            or any(launch.get(key) is not False for key in
                ("automatic_retry", "full_test_authorized", "benchmark_attempt_started", "standard_ready"))
            or any(type(launch.get(key)) is not int or launch[key] != 0 for key in ("model_calls", "benchmark_actions"))
            or launch.get("diagnostic_native_actions") != CONTROLLER_DIAGNOSTIC_ACTIONS
            or launch.get("diagnostic_physics_substeps") != CONTROLLER_DIAGNOSTIC_SUBSTEPS
            or launch.get("diagnostic_native_actions_planned") != CONTROLLER_DIAGNOSTIC_ACTIONS
            or launch.get("plan_file_sha256") != CONTROLLER_PLAN_FILE_SHA256
            or launch.get("plan_sha256") != CONTROLLER_PLAN_CONTENT_SHA256):
        raise AssemblyError("completed_reviewed_controller_diagnostic_required")
    return launch


def checked_parameters(value):
    if not isinstance(value, dict) or set(value) != {"controller", "monitoring", "execution_limits"}:
        raise AssemblyError("exact_selected_parameter_groups_required")
    controller = ControllerConfig(**value["controller"])
    monitoring = MonitoringSettings(**value["monitoring"])
    limits = ExecutionLimits(**value["execution_limits"])
    controller.validate(); monitoring.checked()
    actual = {"controller": asdict(controller), "monitoring": asdict(monitoring), "execution_limits": asdict(limits)}
    if actual != value:
        raise AssemblyError("fully_explicit_selected_parameters_required")
    return actual


def build_document(paths):
    launch = completed_controller(paths["controller_launch"])
    if set(paths) != EVIDENCE or "controller_independent_review" not in EVIDENCE:
        raise AssemblyError("independent_controller_review_gate_not_installed")
    evidence = {key: entry(path) for key, path in sorted(paths.items())}
    review = read(paths["controller_review"])
    independent = read(paths["controller_independent_review"])
    parameters = checked_parameters(review.get("selected_parameters"))
    if (review.get("schema") != "robodojo-controller-characterization-review-1"
            or review.get("preparation_checks_passed") is not True
            or review.get("selection_supported_by_recorded_stages") is not True
            or review.get("launch_result_sha256") != evidence["controller_launch"]["sha256"]
            or review.get("full_test_authorized") is not False
            or independent.get("schema") != "robodojo-controller-selection-independent-review-1"
            or independent.get("selection_supported") is not True
            or independent.get("controller_launch") != evidence["controller_launch"]
            or independent.get("controller_review") != evidence["controller_review"]
            or independent.get("selected_parameters_sha256") != canonical_sha256(parameters)
            or independent.get("full_test_authorized") is not False):
        raise AssemblyError("actual_independent_controller_selection_review_required")
    child = Path(paths["controller_launch"]).parent / "worker/controller"
    required = [evidence["controller_launch"], evidence["native_isolation"],
        entry(child / "controller_result.json"), entry(child / "preparation.json"),
        entry(child / "controller_preclose_ready.json")]
    if (required[2]["sha256"] != launch.get("controller_result_sha256")
            or required[3]["sha256"] != launch.get("preparation_sha256")):
        raise AssemblyError("controller_raw_result_binding_mismatch")
    preparation = read(required[3]["path"])
    plan = entry(preparation["plan_path"])
    if (plan["sha256"] != CONTROLLER_PLAN_FILE_SHA256
            or canonical_sha256(read(plan["path"])) != CONTROLLER_PLAN_CONTENT_SHA256
            or preparation.get("plan_file_sha256") != plan["sha256"]
            or preparation.get("plan_sha256") != CONTROLLER_PLAN_CONTENT_SHA256
            or preparation.get("diagnostic_native_actions_planned") != CONTROLLER_DIAGNOSTIC_ACTIONS
            or any(record.get("controller_plan") != plan
                or record.get("plan_content_sha256") != CONTROLLER_PLAN_CONTENT_SHA256
                for record in (review, independent))
            or parameters["controller"]["angular_speed_rad_s"] != .5):
        raise AssemblyError("reviewed_controller_plan_binding_required")
    required.append(plan)
    cache = VerifiedFileHashCache()
    for record, needed in ((review, required), (independent, [evidence["controller_review"], *required])):
        rows = record.get("source_files")
        if not isinstance(rows, list) or not rows:
            raise AssemblyError("controller_review_source_closure_required")
        pairs = set()
        for row in rows:
            cache.verify(row["path"], row["sha256"])
            pairs.add((row["path"], row["sha256"]))
        if not {(row["path"], row["sha256"]) for row in needed} <= pairs:
            raise AssemblyError("controller_review_source_closure_required")
    programs = [*(BASE / "src/robodojo_uc71" / name for name in PROGRAM_NAMES), Path(__file__).resolve()]
    return {"schema": "robodojo-runtime-bundle-1", "scope": SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence, "parameters": parameters,
        "preparation_checks": {key: True for key in sorted(PREPARATION_CHECKS)},
        "program_files": [entry(path) for path in programs],
        "full_test_authorized": False, "asset_fidelity_passed": False,
        "loaded_contact_validated": False, "workspace_wide_tracking_validated": False,
        "limitations": ["Robot geometry/camera projection measured near home; control settings characterize the recorded free-space stages.",
            "Empty-gripper timing does not certify loaded grasps or contact behavior.",
            "Published asset defect remains unchanged and disclosed.",
            "A fresh reset, actual worker READY, exact human approval and one-shot claim are still required before model actions."]}


def verify_static(path, native_root):
    bundle = VerifiedRuntimeBundle.load(path, native_root)
    bundle.verify()
    forbidden = sorted(name for name in sys.modules if name.split(".")[0] in
        {"omni", "isaacsim", "isaaclab", "carb", "torch"})
    if forbidden:
        raise AssemblyError("native_module_imported_by_static_verifier")
    return {"schema": "robodojo-runtime-bundle-static-check-1", "verified": True,
        "checked_bundle_path": str(Path(path).absolute()), "bundle_sha256": file_sha256(path),
        "uid": os.getuid(), "gid": os.getgid(), "supplementary_groups": os.getgroups(),
        "scope": SCOPE, "controller_capabilities": bundle.controller_capabilities,
        "native_modules_loaded": forbidden, "simulator_starts": 0, "model_calls": 0,
        "policy_actions": 0, "full_test_authorized": False, "worker_ready_issued": False}


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    Path(path).chmod(0o444)


def prepare(paths, *, native_root, python, output=None, assemble=False,
            control_backend="legacy_joint", official_setup=None, official_qualification=None):
    from robodojo_uc71.official_control import validate_control_backend
    validate_control_backend(control_backend)
    if control_backend == "official_ee" and (official_setup is None or official_qualification is None):
        raise AssemblyError("official_setup_and_fresh_native_qualification_required")
    if os.getuid() != 0:
        raise AssemblyError("root_required_to_test_separate_reader_uid")
    if assemble and output is None:
        raise AssemblyError("fresh_output_directory_required")
    output = Path(output).absolute() if output is not None else None
    if output is not None and (output.exists() or output.parent.resolve(strict=True) != output.parent):
        raise AssemblyError("fresh_canonical_output_directory_required")
    # No candidate or pass record exists before completed_controller and reviews.
    document = build_document(paths)
    if control_backend == "official_ee":
        document["control_backend"] = control_backend
        document["official_control"] = {"setup": entry(official_setup), "qualification": entry(official_qualification)}
        names = ("official_control.py", "official_control_setup.py", "official_control_readiness.py",
                 "native_config.py", "native_episode.py", "native_worker_protocol.py", "native_worker_client.py",
                 "native_episode_runner.py", "client.py")
        document["program_files"].extend(entry(BASE/"src/robodojo_uc71"/name) for name in names)
        document["limitations"].append("Legacy diagnostics support robot geometry only; official EEF motion requires its separately bound current-host qualification. Legacy DLS speeds and joint-increment limits do not apply.")
    parent = output.parent if output is not None else Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix=".runtime-bundle-validation-", dir=parent) as temporary:
        staging = Path(temporary); staging.chmod(0o755)
        candidate = staging / "bundle.json"
        write(candidate, document)
        parent_check = verify_static(candidate, native_root)
        environment = {"PATH": str(Path(python).absolute().parent) + ":/usr/bin:/bin",
            "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        argv = [str(Path(python).absolute()), "-I", "-B", str(Path(__file__).resolve()),
            "--verify-existing-bundle", str(candidate), "--native-root", str(Path(native_root).resolve(strict=True))]
        result = subprocess.run(argv, env=environment, cwd="/tmp", stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120, check=True, user=65534, group=65534,
            extra_groups=(), start_new_session=True, close_fds=True)
        child_check = json.loads(result.stdout)
        if (child_check.get("verified") is not True or child_check.get("uid") != 65534
                or child_check.get("gid") != 65534 or child_check.get("supplementary_groups") != []
                or child_check.get("bundle_sha256") != parent_check["bundle_sha256"]
                or child_check.get("native_modules_loaded") != []
                or child_check.get("full_test_authorized") is not False):
            raise AssemblyError("separate_uid_static_validation_failed")
        # Recheck all sources after the independent reader and before publication.
        verify_static(candidate, native_root)
        report = {"schema": "robodojo-runtime-bundle-assembly-1", "bundle_issued": assemble,
            "inputs_verified": True, "scope": SCOPE, "bundle_sha256": file_sha256(candidate),
            "bundle_path": str(output / "bundle.json") if assemble else None,
            "parent_static_check": parent_check, "uid65534_static_check": child_check,
            "atomic_directory_publication": assemble, "assembler": entry(Path(__file__).resolve()),
            "simulator_starts": 0, "model_calls": 0, "policy_actions": 0,
            "full_test_authorized": False, "worker_ready_issued": False}
        if assemble:
            write(staging / "assembly.json", report)
            os.rename(staging, output)
            fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing-bundle", type=Path)
    parser.add_argument("--native-root", type=Path, default=BASE / "runtime/native/RoboDojo")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--control-backend", choices=("official_ee", "legacy_joint"), default="official_ee")
    parser.add_argument("--official-setup", type=Path)
    parser.add_argument("--official-qualification", type=Path)
    for name, relative in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=DEFAULT_ROOT / relative)
    for name in ("controller_launch", "controller_review", "controller_independent_review", "native_isolation"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    if args.verify_existing_bundle is not None:
        if args.assemble:
            parser.error("Static existing-bundle verification cannot assemble output")
        print(json.dumps(verify_static(args.verify_existing_bundle, args.native_root)))
        return 0
    if any(getattr(args, name) is None for name in ("controller_launch", "controller_review", "controller_independent_review")):
        parser.error("Actual completed controller launch, measured review and independent review are required")
    paths = {key: getattr(args, key) for key in DEFAULTS}
    paths.update({key: getattr(args, key) for key in ("controller_launch", "controller_review", "controller_independent_review")})
    paths["native_isolation"] = args.native_isolation or args.controller_launch.parent / "worker/controller/worker_isolation.json"
    print(json.dumps(prepare(paths, native_root=args.native_root, python=args.python,
        output=args.output, assemble=args.assemble, control_backend=args.control_backend,
        official_setup=args.official_setup, official_qualification=args.official_qualification), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
