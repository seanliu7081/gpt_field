from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pytest

from robodojo_uc71.approval import canonical_sha256, file_sha256
from robodojo_uc71.controller import ControllerConfig
from robodojo_uc71.fragment_executor import MonitoringSettings
from robodojo_uc71.schema import ExecutionLimits
from robodojo_uc71.reset_binding import verify_fresh_reset
import robodojo_uc71.native_runtime_bundle as module
import robodojo_uc71.approval as approval_module
from test_reset_binding import reset


@pytest.fixture
def bundle_fixture(tmp_path, reset, monkeypatch):
    scripts = Path(module.__file__).resolve().parents[2] / "scripts"
    def save(name, value):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": file_sha256(path)}
    source = save("synthetic_robot_source", {"test_only": True})
    tool = np.eye(4); tool[0, 3] = .149
    parameters = {"controller": asdict(ControllerConfig(transit_translation_speed_m_s=.2,
        contact_translation_speed_m_s=.05, angular_speed_rad_s=.5)),
        "execution_limits": asdict(ExecutionLimits()), "monitoring": asdict(MonitoringSettings(
        {"transit": .003, "contact": .001}, {"transit": .03, "contact": .02},
        2, 3, .0002, .002, 2, 5, 10, .0002, .05))}
    evidence = {}
    evidence["geometry"] = save("geometry", {"schema": "robodojo-physical-tool-candidate-1",
        "link6_from_tool": tool.tolist(), "source_files": [source],
        "script_sha256": file_sha256(scripts / "derive_tool_candidate.py")})
    reference, robot, camera, images, _ = reset
    fresh_robot = save("fresh_robot", {"schema": "robodojo-native-characterization-measurement-1",
        "raw_robot_inventory": robot})
    fresh_camera = save("fresh_camera", camera)
    fresh_images = {}
    for name, path in images.items():
        fresh = tmp_path / ("fresh_" + path.name)
        fresh.write_bytes(path.read_bytes())
        fresh_images[name] = {"path": str(fresh), "sha256": file_sha256(fresh)}
    first_launch = save("first_launch", {"schema": "robodojo-native-characterization-launch-1",
        "diagnostic_completed": True, "returncode": 0, "termination_signals": [],
        "gpu_processes_released": True, "gpu_memory_released": True})
    checkpoints = [{"index": i, "robot_measurement": fresh_robot, "camera_inventory": fresh_camera,
        "images": [{"camera": "cam_" + name, **row} for name,row in fresh_images.items()]}
        for i in [5,14,23,32]]
    input_files = [fresh_robot, fresh_camera, *fresh_images.values()]
    evidence["joint_analysis"] = save("joint_analysis", {"schema": "robodojo-native-joint-probe-analysis-1",
        "geometry_candidate_sha256": evidence["geometry"]["sha256"],
        "geometry_candidate_path": evidence["geometry"]["path"], "source_binding_verified": True,
        "source_launch_result_path": first_launch["path"], "source_launch_result_sha256": first_launch["sha256"],
        "kinematics_geometry_passed": True, "all_recorded_pose_checks_passed": True,
        "thresholds": module.THRESHOLDS, "diagnostic_native_actions_analyzed": 37,
        "full_calibration_passed": False, "controller_tracking_validated": False, "standard_ready": False,
        "full_test_authorized": False, "fixed_projection_checkpoint_indices": [5,14,23,32],
        "checkpoint_inputs": checkpoints, "source_files": [source, evidence["geometry"], first_launch, *input_files],
        "pose_checks": [{"passed": True}], "maximum_errors": {k: 0 for k in module.THRESHOLDS}})
    joint_path = Path(evidence["joint_analysis"]["path"])
    joint = json.loads(joint_path.read_text())
    joint["script_sha256"] = file_sha256(scripts / "analyze_joint_probe.py")
    joint_path.write_text(json.dumps(joint)); evidence["joint_analysis"]["sha256"] = file_sha256(joint_path)
    evidence["reset_reference"] = save("reference", reference)
    comparison = verify_fresh_reset(reference, robot, camera, {name:r["path"] for name,r in fresh_images.items()})
    evidence["reset_check"] = save("reset_check", comparison)
    program = Path(module.__file__).with_name("reset_binding.py")
    program = {"path": str(program), "sha256": file_sha256(program)}
    reset_files = [evidence["reset_reference"], evidence["reset_check"], program,
        reference["robot_file"], reference["camera_file"], *reference["image_files"].values(), *input_files]
    evidence["reset_closure"] = save("closure", {"schema": "robodojo-reset-evidence-closure-1",
        "reference": evidence["reset_reference"], "comparison": evidence["reset_check"],
        "reset_binding_program": program, "fresh_robot_measurement": fresh_robot,
        "robot_inventory_field": "raw_robot_inventory", "fresh_camera_inventory": fresh_camera,
        "fresh_images": fresh_images, "reproduced_comparison_sha256": canonical_sha256(comparison),
        "comparison_reproduced_exactly": True, "source_files": reset_files, "full_test_authorized": False})
    paused_state = {"native_physics_time_s": 3.56, "native_sim_step_calls": 800,
        "articulation_dof_positions": {arm: [[0.]*8] for arm in ("left", "right")}}
    evidence["pause"] = save("pause", {"schema": "robodojo-native-paused-physics-1", "passed": True,
        "elapsed_monotonic_s": 300.1, "clock_unchanged": True, "step_count_unchanged": True,
        "dof_positions_unchanged": True, "monitor_healthy": True, "simulation_updates_during_wait": 0,
        "requested_seconds": 300, "before": paused_state, "after": paused_state,
        "start_monotonic_s": 10., "end_monotonic_s": 310.1, "model_calls": 0, "benchmark_actions": 0})
    evidence["native_isolation"] = save("isolation", {"schema": "robodojo-worker-unix-isolation-proof-1",
        "pid": 123, "uid": 65534, "gid": 65534, "supplementary_groups": [], "no_new_privs": True,
        "effective_permitted_inheritable_ambient_capabilities": 0,
        "access_denied": {"credential_directory": True, "parent_environment": True}})
    child = tmp_path / "worker/controller"
    child.mkdir(parents=True)
    native_stage = save("native_stage", {"test_only": True})
    plan_data = {"test_only": True, "maximum_native_actions": 960, "maximum_physics_substeps": 9600,
        "stages": [{"native_actions": 25 if index < 38 else 10} for index in range(39)]}
    plan = save("controller_plan", plan_data)
    plan_content = canonical_sha256(plan_data)
    # These isolated fixtures exercise binding logic without issuing or rewriting
    # real diagnostic evidence. Production constants bind the reviewed plan004.
    monkeypatch.setattr(module, "CONTROLLER_PLAN_FILE_SHA256", plan["sha256"])
    monkeypatch.setattr(module, "CONTROLLER_PLAN_CONTENT_SHA256", plan_content)
    plan_fields = {"plan_file_sha256": plan["sha256"], "plan_sha256": plan_content,
        "diagnostic_native_actions_planned": 960}
    prep = save("worker/controller/preparation", {"test_only": True, "plan_path": plan["path"], **plan_fields})
    result = save("worker/controller/controller_result", {"schema": "robodojo-controller-characterization-result-1",
        "diagnostic_completed": True, "diagnostic_native_actions": 960, "diagnostic_physics_substeps": 9600,
        "worker_isolation_sha256": evidence["native_isolation"]["sha256"], "stage_evidence": native_stage})
    preclose = save("worker/controller/controller_preclose_ready", {
        "schema": "robodojo-controller-characterization-preclose-1", "controller_result_sha256": result["sha256"],
        "preparation_sha256": prep["sha256"], "child_pid": 123, "ready_for_shutdown": True,
        "no_pending_exception": True, "native_env_close_returned": True, "physx_monitor_healthy": True, "cleanup_errors": []})
    evidence["controller_launch"] = save("launch", {"diagnostic_completed": True, "returncode": 0,
        "pid": 123, "worker_isolation_verified": True, "termination_signals": [],
        "gpu_processes_released": True, "gpu_memory_released": True,
        "diagnostic_native_actions": 960, "diagnostic_physics_substeps": 9600, **plan_fields,
        "controller_result_sha256": result["sha256"], "preparation_sha256": prep["sha256"]})
    controller_sources = [evidence["controller_launch"], evidence["native_isolation"], result, prep, preclose, native_stage, plan]
    evidence["controller_review"] = save("review", {"schema": "robodojo-controller-characterization-review-1",
        "preparation_checks_passed": True, "selected_parameters": parameters,
        "launch_result_sha256": evidence["controller_launch"]["sha256"],
        "selection_supported_by_recorded_stages": True, "full_test_authorized": False,
        "controller_plan": plan, "plan_content_sha256": plan_content,
        "source_files": controller_sources})
    evidence["controller_independent_review"] = save("independent_review", {
        "schema": "robodojo-controller-selection-independent-review-1", "selection_supported": True,
        "controller_launch": evidence["controller_launch"], "controller_review": evidence["controller_review"],
        "controller_plan": plan, "plan_content_sha256": plan_content,
        "selected_parameters_sha256": canonical_sha256(parameters),
        "scope": "measured_robot_geometry_and_free_space_characterization",
        "full_test_authorized": False, "loaded_contact_validated": False, "workspace_wide_tracking_validated": False,
        "limitations": ["Synthetic offline fixture only."], "source_files": [evidence["controller_review"], *controller_sources]})
    evidence["asset_assessment"] = save("asset", {"schema": "robodojo-native-asset-diagnostic-assessment-1",
        "asset_fidelity_passed": False, "full_test_authorized": False})
    figures = [{"index": i, **save("figure_"+str(i), {"test_only": True})} for i in [5,14,23,32]]
    projection_sources = [evidence["geometry"], evidence["joint_analysis"], *input_files]
    projection_analysis = save("projection_analysis", {"schema": "robodojo-joint-probe-projection-analysis-1",
        "kinematics_analysis": evidence["joint_analysis"], "fixed_checkpoint_indices": [5,14,23,32],
        "original_rgb_unchanged": True, "full_test_authorized": False,
        "script_sha256": file_sha256(scripts / "check_joint_probe_projection.py"),
        "projection_helper_sha256": file_sha256(scripts / "check_robot_projection.py"),
        "figures": figures, "source_files": projection_sources})
    evidence["joint_projection_review"] = save("projection", {"schema": "robodojo-joint-probe-projection-review-1",
        "visible_robot_projection_supported": True, "full_test_authorized": False,
        "kinematics_analysis": evidence["joint_analysis"], "geometry_candidate": evidence["geometry"],
        "source_launch_result_sha256": first_launch["sha256"], "fixed_checkpoint_indices": [5,14,23,32],
        "camera_frames_visually_inspected": 12, "original_rgb_unchanged": True,
        "projection_analysis": projection_analysis, "reviewed_figures": figures,
        "source_files": [projection_analysis, *projection_sources, *figures]})
    programs = [Path(module.__file__), *[Path(module.__file__).with_name(name) for name in
        ("approval.py", "controller.py", "fragment_executor.py", "native_measurements.py", "native_robot_inventory.py", "reset_binding.py", "schema.py")]]
    document = {"schema": "robodojo-runtime-bundle-1",
        "scope": "measured_robot_geometry_and_free_space_characterization", "full_test_authorized": False,
        "asset_fidelity_passed": False, "loaded_contact_validated": False, "workspace_wide_tracking_validated": False,
        "evidence": evidence, "preparation_checks": {k: True for k in module.PREPARATION_CHECKS},
        "parameters": parameters, "program_files": [{"path": str(p), "sha256": file_sha256(p)} for p in programs]}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(document))
    return path, document, tmp_path


def test_bundle_preserves_scope_and_explicit_physical_tool(bundle_fixture):
    path, _, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    camera = {"schema": "robodojo-native-camera-inventory-1", "cameras": {name: {
        "width": 640, "height": 480, "intrinsics": np.eye(3).tolist(), "world_from_optical": np.eye(4).tolist()}
        for name in ("head", "left_wrist", "right_wrist")}}
    public = bundle.public_calibration(camera)
    assert public["tools"]["left"]["link6_from_tool"][0][3] == .149
    assert bundle.document["full_test_authorized"] is False
    assert bundle.controller_capabilities["action_translation_m"] == {"transit": .008, "contact": .002}
    signature = bundle.sensor_signature(camera)
    camera["cameras"]["left_wrist"]["world_from_optical"][0][3] = .1
    assert bundle.sensor_signature(camera) == signature


@pytest.mark.parametrize("field", ["full_test_authorized", "asset_fidelity_passed", "loaded_contact_validated",
                                   "workspace_wide_tracking_validated"])
def test_bundle_cannot_expand_authority_or_validation_scope(bundle_fixture, field):
    path, document, root = bundle_fixture
    document[field] = True; path.write_text(json.dumps(document))
    with pytest.raises(module.RuntimeBundleError, match="scoped_preparation"):
        module.VerifiedRuntimeBundle.load(path, root)


def test_changed_evidence_is_rejected_after_load(bundle_fixture):
    path, document, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    Path(document["evidence"]["controller_review"]["path"]).write_text("{}")
    with pytest.raises(module.RuntimeBundleError, match="evidence_or_program_changed"):
        bundle.verify()


def test_unreviewed_controller_values_rejected(bundle_fixture):
    path, document, root = bundle_fixture
    document["parameters"]["controller"]["transit_translation_speed_m_s"] = .4
    path.write_text(json.dumps(document))
    with pytest.raises(module.RuntimeBundleError, match="controller_characterization"):
        module.VerifiedRuntimeBundle.load(path, root)


def rewrite_evidence(bundle_fixture, label, mutate):
    path, document, _ = bundle_fixture
    entry = document["evidence"][label]
    value = json.loads(Path(entry["path"]).read_text())
    mutate(value)
    Path(entry["path"]).write_text(json.dumps(value))
    entry["sha256"] = file_sha256(entry["path"])
    path.write_text(json.dumps(document))


@pytest.mark.parametrize("field", ["kinematics_analysis", "geometry_candidate"])
def test_unrelated_projection_review_cannot_satisfy_current_geometry(bundle_fixture, field):
    rewrite_evidence(bundle_fixture, "joint_projection_review", lambda d: d[field].update(sha256="0"*64))
    with pytest.raises(module.RuntimeBundleError, match="projection_source_binding"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_projection_review_must_bind_actual_plot_files(bundle_fixture):
    rewrite_evidence(bundle_fixture, "joint_projection_review",
        lambda d: d.update(source_files=[r for r in d["source_files"] if "figure_" not in r["path"]]))
    with pytest.raises(module.RuntimeBundleError, match="nested_evidence_closure"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_joint_analysis_cannot_omit_its_actual_rgb_inputs(bundle_fixture):
    rewrite_evidence(bundle_fixture, "joint_analysis",
        lambda d: d.update(source_files=[r for r in d["source_files"] if not r["path"].endswith(".png")]))
    with pytest.raises(module.RuntimeBundleError, match="nested_evidence_closure"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_fragment_executor_is_part_of_program_closure(bundle_fixture):
    path, document, root = bundle_fixture
    document["program_files"] = [r for r in document["program_files"] if not r["path"].endswith("fragment_executor.py")]
    path.write_text(json.dumps(document))
    with pytest.raises(module.RuntimeBundleError, match="program_closure"):
        module.VerifiedRuntimeBundle.load(path, root)


@pytest.mark.parametrize("mutation", [
    lambda d: d["after"].update(native_sim_step_calls=801),
    lambda d: d["after"]["articulation_dof_positions"]["left"][0].__setitem__(0, .001),
    lambda d: d.update(model_calls=1), lambda d: d.update(benchmark_actions=1),
    lambda d: d.update(end_monotonic_s=99.)])
def test_pause_pass_flag_cannot_hide_physics_actions_or_invalid_elapsed_time(bundle_fixture, mutation):
    rewrite_evidence(bundle_fixture, "pause", mutation)
    with pytest.raises(module.RuntimeBundleError, match="paused_physics"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_fresh_reset_comparison_is_recomputed_instead_of_trusting_pass_claim(bundle_fixture):
    rewrite_evidence(bundle_fixture, "reset_check",
        lambda d: d["actual_image_comparisons"]["head"].update(mean_absolute_channel_difference=.1))
    comparison_entry = bundle_fixture[1]["evidence"]["reset_check"]
    comparison = json.loads(Path(comparison_entry["path"]).read_text())
    def bind_changed_report(closure):
        closure["comparison"] = dict(comparison_entry)
        closure["reproduced_comparison_sha256"] = canonical_sha256(comparison)
        for row in closure["source_files"]:
            if row["path"] == comparison_entry["path"]: row["sha256"] = comparison_entry["sha256"]
    rewrite_evidence(bundle_fixture, "reset_closure", bind_changed_report)
    with pytest.raises(module.RuntimeBundleError, match="comparison_not_reproduced"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_deep_reference_image_mutation_rejected_after_initial_verification(bundle_fixture):
    path, document, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    reference = json.loads(Path(document["evidence"]["reset_reference"]["path"]).read_text())
    Path(reference["image_files"]["head"]["path"]).write_bytes(b"changed")
    with pytest.raises(module.RuntimeBundleError, match="evidence_or_program_changed"):
        bundle.verify()


def test_unchanged_closure_uses_stat_cache_but_metadata_change_rehashes(bundle_fixture, monkeypatch):
    path, _, root = bundle_fixture
    calls = []
    def counted(p):
        calls.append(str(p)); return file_sha256(p)
    monkeypatch.setattr(approval_module, "file_sha256", counted)
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    count = len(calls)
    bundle.verify(); bundle.verify()
    assert len(calls) == count
    source = root / "synthetic_robot_source.json"
    source.touch()
    bundle.verify()
    assert len(calls) == count + 1


def test_cache_rejects_same_size_mutation_with_restored_mtime(bundle_fixture):
    path, _, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    source = root / "synthetic_robot_source.json"
    before = source.stat()
    original = source.read_bytes()
    changed = original.replace(b"true", b"True")
    assert changed != original and len(changed) == len(original)
    source.write_bytes(changed)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(module.RuntimeBundleError, match="evidence_or_program_changed"):
        bundle.verify()


def test_changed_controller_stage_evidence_invalidates_review_after_load(bundle_fixture):
    path, _, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    (root / "native_stage.json").write_text('{"changed":true}')
    with pytest.raises(module.RuntimeBundleError, match="evidence_or_program_changed"):
        bundle.verify()


def test_controller_review_cannot_omit_artifact_referenced_by_raw_result(bundle_fixture):
    rewrite_evidence(bundle_fixture, "controller_review",
        lambda d: d.update(source_files=[r for r in d["source_files"] if not r["path"].endswith("native_stage.json")]))
    with pytest.raises(module.RuntimeBundleError, match="nested_evidence_closure"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_independent_controller_review_must_bind_exact_selected_parameters(bundle_fixture):
    rewrite_evidence(bundle_fixture, "controller_independent_review",
        lambda d: d.update(selected_parameters_sha256="0"*64))
    with pytest.raises(module.RuntimeBundleError, match="independent_controller_selection"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_controller_review_must_bind_the_reviewed_plan_content(bundle_fixture):
    rewrite_evidence(bundle_fixture, "controller_review", lambda d: d.update(plan_content_sha256="0"*64))
    with pytest.raises(module.RuntimeBundleError, match="reviewed_controller_plan_binding"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_independent_selection_review_cannot_point_to_a_different_plan(bundle_fixture):
    rewrite_evidence(bundle_fixture, "controller_independent_review",
        lambda d: d["controller_plan"].update(sha256="0"*64))
    with pytest.raises(module.RuntimeBundleError, match="independent_controller_selection"):
        module.VerifiedRuntimeBundle.load(bundle_fixture[0], bundle_fixture[2])


def test_bound_controller_plan_is_reverified_after_bundle_load(bundle_fixture):
    path, _, root = bundle_fixture
    bundle = module.VerifiedRuntimeBundle.load(path, root)
    plan = root / "controller_plan.json"
    data = json.loads(plan.read_text())
    data["stages"][0]["native_actions"] = 24
    plan.write_text(json.dumps(data))
    with pytest.raises(module.RuntimeBundleError, match="evidence_or_program_changed"):
        bundle.verify()
