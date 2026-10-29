#!/usr/bin/env python3
"""Publish one immutable, scoped review from verified completed010 evidence."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]
sys.path[:0] = [str(BASE / "src"), str(BASE / "scripts")]
from prepare_runtime_bundle import checked_parameters, completed_controller
from robodojo_uc71.approval import canonical_sha256
from robodojo_uc71.schema import ExecutionLimits, capabilities

def read(path):
    return json.loads(Path(path).read_text())

def entry(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    root = OUT.parent
    attempt = root / "native_controller_characterization_010"
    child = attempt / "worker/controller"
    launch_path = attempt / "launch_result.json"
    launch = completed_controller(launch_path)
    prep = read(child / "preparation.json")
    selected = checked_parameters(read(OUT / "selected_parameters.json"))
    replay = read(OUT / "monitor_replay.json")
    assert replay["manually_selected_parameters"] == selected
    assert replay["all_recorded_stages_complete_under_selected_monitors"] is True
    assert len(replay["stage_monitor_replays"]) == 36
    public = capabilities(ExecutionLimits(**selected["execution_limits"]))
    assert public["limits"] == selected["execution_limits"]
    closure = read(OUT / "source_closure.json")
    rows = closure["source_files"]
    extra = [entry(path) for path in (
        __file__, OUT / "analyze.py", OUT / "selected_parameters.json", OUT / "monitor_replay.json",
        OUT / "source_closure.json", launch_path, child / "controller_result.json",
        child / "preparation.json", child / "controller_preclose_ready.json",
        child / "worker_isolation.json", prep["plan_path"],
        root / "controller_selection_independent_review_001/measurements.json",
        root / "controller_selection_independent_review_001/monitor_comparison.json")]
    for name in ("controller.py", "controller_characterization.py", "fragment_executor.py", "schema.py",
                 "client.py", "native_runtime_bundle.py", "approval.py"):
        extra.append(entry(BASE / "src/robodojo_uc71" / name))
    for name in ("prepare_runtime_bundle.py", "run_standard_attempt.py"):
        extra.append(entry(BASE / "scripts" / name))
    sources = {}
    for row in [*rows, *extra]:
        assert entry(row["path"]) == row, row["path"]
        assert sources.get(row["path"], row["sha256"]) == row["sha256"]
        sources[row["path"]] = row["sha256"]
    review = {
        "schema": "robodojo-controller-characterization-review-1",
        "scope": "measured_robot_geometry_and_free_space_characterization",
        "preparation_checks_passed": True,
        "selection_supported_by_recorded_stages": True,
        "launch_result_sha256": entry(launch_path)["sha256"],
        "controller_launch": entry(launch_path),
        "controller_plan": entry(prep["plan_path"]),
        "plan_content_sha256": prep["plan_sha256"],
        "selected_parameters": selected,
        "selected_parameters_sha256": canonical_sha256(selected),
        "monitor_replay": entry(OUT / "monitor_replay.json"),
        "independent_measured_geometry": entry(root / "controller_selection_independent_review_001/measurements.json"),
        "independent_monitor_comparison": entry(root / "controller_selection_independent_review_001/monitor_comparison.json"),
        "source_closure": entry(OUT / "source_closure.json"),
        "source_files": [{"path": path, "sha256": digest} for path, digest in sorted(sources.items())],
        "diagnostic_native_actions": 960, "diagnostic_physics_substeps": 9600,
        "observed_measured_maxima": replay["measured_maxima"],
        "selection_basis": {
            "controller": "The explicit plan candidate (.2m/s transit,.05m/s contact,.5rad/s rotation,.05rad lead,.02 damping) completed both-arm trials and returns without changing solver/controller math.",
            "arrival": "The2/1mm and.02/.01rad profile tolerances and five stable samples were present in the predeclared threshold sweep. Every active recorded stage completes under replay.",
            "empty_gripper": "Each arm reached physical zero by action6 and five-sample0.5mm gap stability by10. Opening reached0.5mm endpoint error by10 and five-sample gap stability by16. Selected close15/open20 retains measured margins; open endpoint error at20 is2.653 micrometres.",
            "stall": "A10-sample weighted-error window with.05m/rad orientation weight had minimum5.576mm pre-completion progress versus manually selected0.1mm threshold. This checks absence of false stalls on progressing traces, not obstruction detection.",
            "gap_rise": "The2mm/2-sample trigger is a conservative software monitor above recorded settled empty-jaw fluctuations. Loaded grip loss and true positive response are untested.",
            "execution_limits": "Primitive caps are reduced to30mm transit/10mm contact/.08rad rotation. Duration/count ceilings stay explicit software bounds. Open20 and close15 are runtime minima, not a grasp-success predicate.",
            "public_configuration": "Existing schema.capabilities() returns all asdict(ExecutionLimits) fields; NativeExecutiveClient publishes them. The actual runner receives the identical bundle limits/controller/monitoring. No source change was needed."},
        "operator_clearances": replay["operator_clearances"],
        "shutdown": {key: launch[key] for key in ("returncode", "termination_signals", "remaining_owned_pids",
            "gpu_processes_released", "gpu_memory_released", "shutdown_contract_satisfied")},
        "sampled_peak_gpu_used_mib": launch["sampled_peak_gpu_used_mib"],
        "sampled_minimum_free_mib": launch["sampled_minimum_free_mib"],
        "public_limits_verified": public["limits"],
        "full_test_authorized": False, "benchmark_attempt_started": False, "model_calls": 0,
        "standard_ready": False, "asset_fidelity_passed": False,
        "loaded_contact_validated": False, "workspace_wide_tracking_validated": False,
        "loaded_grip_or_slip_response_validated": False, "genuine_obstruction_response_validated": False,
        "task_success_assessed": False, "automatic_configuration_selection": False,
        "limitations": [*replay["limitations"],
            "Orientation-only pulses caused up to1.676mm transient nominal TCP displacement before settling; orientation commands are not assumed to translate the physical tool by exactly zero.",
            "These manual settings are supported for recorded near-home free-space and empty-gripper behavior only. They do not certify arbitrary directions, carried-object clearance, contact-loaded behavior or full-task success.",
            "A separate independent selection review and runtime-bundle verification remain required. No benchmark approval, READY receipt, attempt claim or policy action is created by this review."]}
    assert review["controller_plan"]["sha256"] == prep["plan_file_sha256"]
    with (OUT / "review.json").open("x") as stream:
        stream.write(json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({**entry(OUT / "review.json"), "source_files": len(sources),
                      "selected_parameters_sha256": canonical_sha256(selected)}))

if __name__ == "__main__":
    main()
