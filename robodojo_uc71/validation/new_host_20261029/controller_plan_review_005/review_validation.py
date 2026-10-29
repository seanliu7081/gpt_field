"""Offline independent comparison; no simulator, model or execution authority."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from robodojo_uc71 import controller_characterization as helper
from robodojo_uc71 import native_runtime_bundle as bundle

BASE = Path(__file__).resolve().parents[3]
VALIDATION = BASE / "validation/new_host_20261029"
HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

old_path = VALIDATION / "controller_characterization_plan_004/proposal.json"
new_path = VALIDATION / "controller_characterization_plan_005/proposal.json"
design_path = new_path.parent / "design_review.json"
old, new, design = read(old_path), read(new_path), read(design_path)
assert helper.build_controller_probe_proposal(new["fixed_probe_plan_path"]) == old
helper.validate_controller_probe_proposal(new)
assert ref(new_path)["sha256"] == bundle.CONTROLLER_PLAN_FILE_SHA256
assert helper.content_hash(new) == bundle.CONTROLLER_PLAN_CONTENT_SHA256
indices = [i for i, (before, after) in enumerate(zip(old["stages"], new["stages"])) if before != after]
assert indices == [26, 27, 28, 29, 33, 34, 35, 36]
restored = deepcopy(new)
basis = restored.pop("clearance_preview_basis")
assert basis["closing_authorized_by_preview"] is False
assert basis["future_fresh_three_view_gates_required"] is True
assert basis["full_test_authorized"] is False
assert old["candidate_abort_envelopes"]["joint_distance_from_home_rad"] == .35
assert restored["candidate_abort_envelopes"]["joint_distance_from_home_rad"] == .50
restored["candidate_abort_envelopes"]["joint_distance_from_home_rad"] = .35
preview = read(basis["preview_plan"]["path"])
for i in indices:
    before, after = old["stages"][i], new["stages"][i]
    assert set(before) == set(after)
    assert {k for k in before if before[k] != after[k]} == {"target_tool_pose_wxyz"}
    endpoint = preview["stages"][1 if i < 30 else 5]["target_tool_pose_wxyz"]
    assert after["target_tool_pose_wxyz"] == endpoint
    restored["stages"][i] = deepcopy(before)
assert restored == old
assert len(new["stages"]) == 39
assert sum(s["native_actions"] for s in new["stages"]) == 960
assert new["maximum_physics_substeps"] == 9600
assert new["maximum_simulated_time_s"] == 38.4
assert new["visual_clearance_checkpoint"] == old["visual_clearance_checkpoint"]
assert all(s["operator_clearance_required_before"] for s in new["stages"] if s["kind"] == "gripper_close")

source_paths = {Path(__file__).resolve(), old_path, new_path, design_path,
    BASE / "src/robodojo_uc71/native_runtime_bundle.py", BASE / "scripts/prepare_runtime_bundle.py",
    BASE / "tests/test_native_runtime_bundle.py", BASE / "tests/test_prepare_runtime_bundle.py",
    BASE / "tests/test_controller_preview_basis.py"}
for path, digest in design["program_sha256"].items():
    assert ref(path)["sha256"] == digest
    source_paths.add(Path(path))
for key in ("scope_proposal", "command", "focused_tests", "completed_preview_operator_measurements"):
    row = design[key]
    assert ref(row["path"]) == {k: row[k] for k in ("path", "sha256")}
    source_paths.add(Path(row["path"]))
for row in basis.values():
    if isinstance(row, dict) and set(row) == {"path", "sha256"}:
        assert ref(row["path"]) == row
        source_paths.add(Path(row["path"]))
geometry = read(basis["geometry_analysis"]["path"])
assert geometry["geometry_and_endpoints_passed"] is True
assert geometry["arm_pose_checks"] == 1058
assert geometry["native_actions_analyzed"] == 260

review = {
    "schema": "robodojo-controller-plan-independent-review-1",
    "scope": "Offline review of the measured open-preview basis and exact full diagnostic proposal changes",
    "old_plan": ref(old_path), "new_plan": ref(new_path),
    "new_plan_content_sha256": helper.content_hash(new),
    "exact_authorized_transformation_verified": True,
    "changed_stage_target_poses_only": [{"index": i, "label": new["stages"][i]["label"],
        "native_actions": new["stages"][i]["native_actions"]} for i in indices],
    "joint_home_excursion_rad": {"before": .35, "after": .50},
    "all_other_guards_counts_speeds_grippers_profiles_and_targets_unchanged": True,
    "controller_solver_source_unchanged": True,
    "full_default_plan004_unchanged": True,
    "clearance_preview_basis": basis,
    "independent_actual_geometry": basis["geometry_analysis"],
    "stage_count": 39, "native_actions": 960, "physics_substeps": 9600, "simulated_time_s": 38.4,
    "fresh_three_RGB_gates_still_required": True,
    "preview_authorizes_future_closing": False,
    "actual_preview_visual_interpretation": "All six actual endpoint views were inspected. The left endpoint is over clear tabletop; the right head view shows the inner fingertip separated from the toaster silhouette. Wrist perspective still overlaps distant objects, so these views support the measured preparation pose rather than certify depth, a collision envelope, or future closing clearance.",
    "helper_change": "An explicit three-record completed-preview adapter rechecks the clean 260-action launch, archived program bytes, exact six RGBs/root review, independent 1058-arm-state geometry analysis, and source hashes; only the eight held clearance poses and explicit candidate envelope are transferred. The child authorization binds all three evidence hashes and the exact envelope. Default full builder and preview builder remain separately validated.",
    "accepted_bundle_gate": {"plan_file_sha256": bundle.CONTROLLER_PLAN_FILE_SHA256,
        "plan_content_sha256": bundle.CONTROLLER_PLAN_CONTENT_SHA256,
        "minimum_completed_actions": 960, "preview_260_actions_rejected": True,
        "actual_bundle_issued": False},
    "validation": {"bundle_and_assembler_tests_passed": 50, "driver_focused_tests_passed": 83,
        "simulator_starts": 0, "native_actions": 0, "model_calls": 0,
        "bundle_test_command": "PYTHONPATH=robodojo_uc71/src:robodojo_uc71/runtime/new_host_testdeps robodojo_uc71/runtime/new_host_env/bin/python -B -m pytest robodojo_uc71/tests/test_native_runtime_bundle.py robodojo_uc71/tests/test_prepare_runtime_bundle.py -q"},
    "limitations": [
        "The 0.50 rad envelope is explicit and was exercised by the open-only preview; it is not an unchanged guard or a workspace-wide calibration.",
        "A completed 39-stage/960-action characterization with fresh live three-view close gates remains required; the new full sequence is not yet measured.",
        "Open-only endpoint stability supplies no gripper closing/opening timing or loaded-contact calibration.",
        "No runtime bundle, complete controller selection, full-test approval, or Standard readiness follows from this review."],
    "controller_characterization_passed": False, "selection_supported_by_completed_measurements": False,
    "full_calibration_passed": False, "standard_ready": False, "execution_authorized": False,
    "full_test_authorized": False,
    "source_files": [ref(p) for p in sorted(source_paths)]}
output = HERE / "review.json"
with output.open("x") as stream:
    json.dump(review, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps({"review": ref(output), "source_files": len(review["source_files"])}))
