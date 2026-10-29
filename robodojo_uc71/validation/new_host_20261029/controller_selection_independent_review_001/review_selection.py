"""Independently replay measured monitor decisions, then review explicit settings."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import math
import sys

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE/"src"))
from robodojo_uc71.approval import VerifiedFileHashCache, canonical_sha256, file_sha256
from robodojo_uc71.controller import ControllerConfig
from robodojo_uc71.fragment_executor import MonitoringSettings
from robodojo_uc71.schema import ExecutionLimits
from robodojo_uc71.native_runtime_bundle import CONTROLLER_PLAN_FILE_SHA256, CONTROLLER_PLAN_CONTENT_SHA256

HERE = Path(__file__).resolve().parent
V = BASE/"validation/new_host_20261029"
DRIVER = V/"controller_characterization_review_001"
ATTEMPT = V/"native_controller_characterization_010"
cache, sources = VerifiedFileHashCache(), {}

def ref(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = expected or file_sha256(path)
    cache.verify(path, digest)
    if str(path) in sources and sources[str(path)] != digest: raise ValueError("conflicting_source")
    sources[str(path)] = digest
    return {"path":str(path), "sha256":digest}

def read(path, expected=None):
    return json.loads(Path(ref(path,expected)["path"]).read_text())

def write(path, value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    path.chmod(0o444)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--controller-review",type=Path);args=parser.parse_args()
    for path in (Path(__file__),BASE/"src/robodojo_uc71/fragment_executor.py",BASE/"src/robodojo_uc71/schema.py"):
        ref(path)
    measurements=read(HERE/"measurements.json")
    selected=read(DRIVER/"selected_parameters.json")
    driver_replay=read(DRIVER/"monitor_replay.json")
    launch=read(ATTEMPT/"launch_result.json")
    preparation=read(ATTEMPT/"worker/controller/preparation.json",launch["preparation_sha256"])
    plan=read(preparation["plan_path"],CONTROLLER_PLAN_FILE_SHA256)
    assert canonical_sha256(plan)==CONTROLLER_PLAN_CONTENT_SHA256
    assert selected["controller"]==plan["candidate_controller"]
    assert asdict(ControllerConfig(**selected["controller"]))==selected["controller"]
    settings=MonitoringSettings(**selected["monitoring"]).checked()
    limits=ExecutionLimits(**selected["execution_limits"])
    assert asdict(limits)==selected["execution_limits"]
    assert limits.transit_translation_m==limits.direct_translation_m==.03
    assert limits.contact_translation_m==.01
    assert limits.transit_rotation_rad==limits.contact_rotation_rad==limits.direct_rotation_rad==.08
    assert limits.min_open_steps==20 and limits.min_close_steps==15
    assert driver_replay["manually_selected_parameters"]==selected
    assert driver_replay["all_recorded_stages_complete_under_selected_monitors"] is True
    assert measurements["all_source_bound_geometry_checks_passed"] is True
    outcomes=[]; all_progress=[]
    for stage in measurements["stage_metrics"]:
        arm=stage["active_arm"]
        if arm is None: continue
        raw=plan["stages"][stage["stage_index"]]; values=stage["arms"][arm]
        distances,angles=values["position_error_m"],values["orientation_error_rad"]
        p=settings["position_tolerance_m"][raw["profile"]]
        q=settings["orientation_tolerance_rad"][raw["profile"]]
        arrival=int(distances[0]<=p and angles[0]<=q)
        errors=[];gaps=[];completion=None;stalls=[];progresses=[]
        for i in range(1,len(distances)):
            arrival=arrival+1 if distances[i]<=p and angles[i]<=q else 0
            errors.append(distances[i]+settings["orientation_error_weight_m_per_rad"]*angles[i])
            arrived=arrival>=settings["arrival_stable_samples"]
            if raw["kind"] in ("gripper_close","gripper_open"):
                gaps.append(values["gap_sum_proxy_m"][i])
                window=gaps[-settings["gap_stable_samples"]:]
                stable=len(window)==settings["gap_stable_samples"] and max(window)-min(window)<=settings["gap_stable_range_m"]
                minimum=limits.min_close_steps if raw["kind"]=="gripper_close" else limits.min_open_steps
                complete=arrived and stable and i>=minimum
            else: complete=arrived
            if complete:
                completion=i;break
            if raw["kind"] in ("translation","orientation") and i>=settings["stall_min_steps"]:
                window=errors[-settings["stall_window_steps"]:]
                progress=window[0]-min(window[1:]);progresses.append(progress);all_progress.append(progress)
                if progress<settings["stall_progress_m"]:stalls.append(i)
        assert completion is not None and not stalls
        prior=next(s for s in driver_replay["stage_monitor_replays"] if s["label"]==stage["label"])
        assert completion==prior["monitor_first_completion_action"] and prior["false_stall_steps"]==[]
        assert len(progresses)==len(prior["pre_completion_stall_checks"])
        for a,b in zip(progresses,prior["pre_completion_stall_checks"]):
            assert math.isclose(a,b["progress_m"],rel_tol=1e-8,abs_tol=1e-10)
        outcomes.append({"stage_index":stage["stage_index"],"label":stage["label"],
            "first_completion_native_action":completion,"false_precompletion_stalls":[],
            "minimum_checked_precompletion_progress_m":min(progresses) if progresses else None})
    assert len(outcomes)==36
    comparison={"schema":"robodojo-controller-independent-monitor-comparison-1",
        "selected_parameters_sha256":canonical_sha256(selected),"all36_recorded_monitor_outcomes_match":True,
        "minimum_precompletion_progress_m":min(all_progress),"selected_stall_threshold_m":settings["stall_progress_m"],
        "stage_results":outcomes,"full_test_authorized":False,"model_calls":0,"simulator_starts":0,
        "native_actions_executed":0,"source_files":[{"path":p,"sha256":d} for p,d in sorted(sources.items())]}
    comparison_path=HERE/"monitor_comparison.json"
    if comparison_path.exists(): assert read(comparison_path)==comparison
    else: write(comparison_path,comparison)
    if args.controller_review is None:
        print(json.dumps({"monitor_comparison":ref(comparison_path),"review_issued":False}));return
    review_ref=ref(args.controller_review);review=read(args.controller_review)
    assert review["schema"]=="robodojo-controller-characterization-review-1"
    assert review["preparation_checks_passed"] is True and review["selection_supported_by_recorded_stages"] is True
    assert review["selected_parameters"]==selected and review["full_test_authorized"] is False
    launch_ref=ref(ATTEMPT/"launch_result.json")
    plan_ref=ref(preparation["plan_path"],CONTROLLER_PLAN_FILE_SHA256)
    assert review["launch_result_sha256"]==launch_ref["sha256"]
    assert review["controller_plan"]==plan_ref and review["plan_content_sha256"]==CONTROLLER_PLAN_CONTENT_SHA256
    for document in (review,measurements,driver_replay):
        for row in document["source_files"]:ref(row["path"],row["sha256"])
    ref(comparison_path)
    independent={"schema":"robodojo-controller-selection-independent-review-1","selection_supported":True,
        "scope":"measured_robot_geometry_and_free_space_characterization",
        "controller_launch":launch_ref,"controller_review":review_ref,"controller_plan":plan_ref,
        "plan_content_sha256":CONTROLLER_PLAN_CONTENT_SHA256,
        "selected_parameters_sha256":canonical_sha256(selected),"selected_parameters":selected,
        "independent_measurements":ref(HERE/"measurements.json"),"independent_monitor_replay":ref(comparison_path),
        "all3920_actual_geometry_checks_passed":True,"all36_monitor_outcomes_match":True,
        "controller_speed_settings_supported_at_measured_local_poses":True,
        "empty_gripper_minimum_steps_supported":True,
        "full_test_authorized":False,"loaded_contact_validated":False,"workspace_wide_tracking_validated":False,
        "genuine_obstruction_response_validated":False,"loaded_grip_or_slip_response_validated":False,
        "standard_ready":False,"human_approval_created":False,"model_calls":0,"simulator_starts":0,
        "limitations":[
            "Selected speeds/tolerances and3cm transit/1cm contact/0.08rad command amplitudes have local free-space evidence; arbitrary directions, other workspace poses and loaded contact remain unvalidated.",
            "Five-sample0.5mm gap stability occurs by10 close/16 open actions; selected minimum15/20 adds observed margins. This single empty-gripper sequence does not establish loaded grasp or slip behavior.",
            "All36 measured trace monitor outcomes match independently; no false pre-arrival stalls occurred. True obstruction and2mm sustained gap-rise positives were not induced, and early-yield/chained-policy trajectories were not re-executed.",
            "Nominal TCP transiently shifts up to1.676mm during orientation-only pulses before settling; sampled25Hz speeds do not bound physics-substep peaks.",
            "The fixed nominal tool point differs from the instantaneous finger-pad midpoint by up to5.61micrometres during asymmetric finger motion; source-derived symmetric geometry consistency passes without redefining the tool point.",
            "Execution duration/count limits are software ceilings, not measured guarantees. Direct motion remains1–5 native actions and may return pending verification before the five-sample arrival requirement.",
            "The published missing normal texture remains unchanged and disclosed; this review grants neither full-test approval nor live worker READY."],
        "source_files":[{"path":p,"sha256":d} for p,d in sorted(sources.items())]}
    for row in independent["source_files"]:cache.verify(row["path"],row["sha256"])
    output=HERE/"review.json";write(output,independent)
    print(json.dumps({"independent_review":ref(output),"verified_source_files":len(independent["source_files"])}))

if __name__=="__main__":main()
