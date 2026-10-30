"""Evidence-backed runtime measurements and explicitly reviewed control settings.

No bundle is issued by loading this module. A complete, hash-bound preparation
review is required. That review certifies the stated free-space characterization
scope only; it never authorizes an episode, certifies loaded contact behavior,
repairs published assets or turns a model declaration into benchmark success.
The immutable geometry/measurement candidates remain unchanged in their files.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .approval import ApprovalError, VerifiedFileHashCache, canonical_sha256
from .controller import ControllerConfig, measured_tcp_pose
from .official_control import validate_control_backend
from .fragment_executor import MonitoringSettings, RobotMeasurement
from .native_measurements import NativeMeasurements, native_link6_tool_candidate
from .reset_binding import CAMERAS, verify_fresh_reset
from .schema import (CALIBRATION_VERSION, ExecutionLimits, calibration_binding,
                     validate_calibration, validate_controller_capabilities)


EVIDENCE = frozenset({"geometry", "joint_analysis", "joint_projection_review", "controller_launch",
    "controller_review", "controller_independent_review", "reset_reference", "reset_check", "reset_closure",
    "pause", "asset_assessment", "native_isolation"})
PREPARATION_CHECKS = frozenset({"robot_geometry_and_jacobian", "synchronized_camera_projection",
    "selected_free_space_controller_settings", "empty_gripper_timing", "inactive_arm_holds",
    "native_action_accounting", "exact_robot_and_camera_reset", "bounded_rgb_reset",
    "paused_300_seconds", "isolated_native_worker", "clean_process_and_gpu_release"})
THRESHOLDS = {"fk_translation_m": 5e-6, "fk_rotation_rad": 1e-5,
    "jacobian_linear_max_abs": 1e-5, "jacobian_angular_max_abs": 1e-5,
    "symmetric_pad_tcp_consistency_m": 5e-6}
# The independently reviewed plan004 retains 39 stages and adds a 15mm outward
# clearance offset; all action counts, speeds, and diagnostic guards are unchanged.
# These bindings deliberately require another review for any future plan change.
CONTROLLER_DIAGNOSTIC_ACTIONS = 960
CONTROLLER_DIAGNOSTIC_SUBSTEPS = 9600
CONTROLLER_DIAGNOSTIC_STAGES = 39
CONTROLLER_PLAN_FILE_SHA256 = "1b196b2f6f3ebe2455e03d8e578814b7b6831021b5d1dd5e7bcd290cdff0532d"
CONTROLLER_PLAN_CONTENT_SHA256 = "5638ae4e23c51e2f1980b68a03e1e99d5d52f22082cec5b2a6e6b403d9d20528"


class RuntimeBundleError(ValueError):
    pass


class VerifiedRuntimeBundle:
    @classmethod
    def load(cls, path, native_root):
        return cls(path, native_root)

    def __init__(self, path, native_root):
        self.path = Path(path).resolve(strict=True)
        self.native_root = Path(native_root).resolve(strict=True)
        raw = self.path.read_bytes()
        self._hash = hashlib.sha256(raw).hexdigest()
        self.document = json.loads(raw)
        self._content_hash = canonical_sha256(self.document)
        self._file_cache = VerifiedFileHashCache()
        self._json_records = {}
        self._reset_verified = None
        self._readers = {}
        self.control_backend = validate_control_backend(self.document.get("control_backend", "legacy_joint"))
        self.verify()
        geometry = self._read("geometry")
        self._tool_transform = np.asarray(geometry["link6_from_tool"], dtype=float)
        self._tool_hash = self.document["evidence"]["geometry"]["sha256"]
        self.reset_reference = self._read("reset_reference")
        parameters = self.document["parameters"]
        self.controller_config = ControllerConfig(**parameters["controller"])
        self.monitoring_settings = MonitoringSettings(**parameters["monitoring"])
        self.execution_limits = ExecutionLimits(**parameters["execution_limits"])
        self.controller_config.validate()
        self.monitoring_settings.checked()

    def _read(self, label):
        return self._read_json(self.document["evidence"][label])

    def _verify_file(self, entry):
        try:
            self._file_cache.verify(entry["path"], entry["sha256"])
            return (str(entry["path"]), entry["sha256"])
        except (KeyError, ApprovalError) as error:
            raise RuntimeBundleError("runtime_evidence_or_program_changed") from error

    def _read_json(self, entry):
        key = self._verify_file(entry)
        if key not in self._json_records:
            raw = Path(key[0]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != key[1]:
                raise RuntimeBundleError("runtime_json_changed_during_read")
            value = json.loads(raw)
            # Verify again after the read, including a fresh stat/race check.
            self._verify_file(entry)
            self._json_records[key] = value
        return deepcopy(self._json_records[key])

    def _source_files(self, rows, required=()):
        if not isinstance(rows, list) or not rows:
            raise RuntimeBundleError("nested_evidence_closure_missing")
        actual = {self._verify_file(row) for row in rows}
        expected = {self._verify_file(row) for row in required}
        if not expected <= actual:
            raise RuntimeBundleError("nested_evidence_closure_missing")

    @staticmethod
    def _binding(entry):
        return {"path": entry["path"], "sha256": entry["sha256"]}

    def _analysis_script(self, name, expected):
        self._verify_file({"path": str(Path(__file__).resolve().parents[2] / "scripts" / name), "sha256": expected})

    def _joint_closure(self, joint):
        self._analysis_script("analyze_joint_probe.py", joint.get("script_sha256"))
        geometry = self.document["evidence"]["geometry"]
        launch = {"path": joint["source_launch_result_path"], "sha256": joint["source_launch_result_sha256"]}
        if ({"path": joint["geometry_candidate_path"], "sha256": joint["geometry_candidate_sha256"]} != geometry
                or any(joint.get(k) is not False for k in ("full_calibration_passed",
                    "controller_tracking_validated", "standard_ready", "full_test_authorized"))
                or joint.get("fixed_projection_checkpoint_indices") != [5, 14, 23, 32]
                or [row.get("index") for row in joint.get("checkpoint_inputs", [])] != [5, 14, 23, 32]):
            raise RuntimeBundleError("measured_joint_geometry_validation_required")
        required = [geometry, launch]
        for checkpoint in joint["checkpoint_inputs"]:
            images = checkpoint["images"]
            if len(images) != 3 or {row["camera"] for row in images} != set(CAMERAS.values()):
                raise RuntimeBundleError("joint_checkpoint_rgb_closure_missing")
            required.extend([checkpoint["robot_measurement"], checkpoint["camera_inventory"], *images])
        self._source_files(joint.get("source_files"), required)
        raw = self._read_json(launch)
        if (raw.get("schema") != "robodojo-native-characterization-launch-1"
                or raw.get("diagnostic_completed") is not True or raw.get("returncode") != 0
                or raw.get("termination_signals") != []
                or raw.get("gpu_processes_released") is not True or raw.get("gpu_memory_released") is not True
                or not joint.get("pose_checks")
                or any(row.get("passed") is not True for row in joint["pose_checks"])
                or any(type(joint.get("maximum_errors", {}).get(k)) not in (int, float)
                    or not math.isfinite(joint["maximum_errors"][k])
                    or not 0 <= joint["maximum_errors"][k] <= limit for k, limit in THRESHOLDS.items())):
            raise RuntimeBundleError("measured_joint_geometry_validation_required")

    def _projection_closure(self, projection, joint):
        evidence = self.document["evidence"]
        if (projection.get("kinematics_analysis") != evidence["joint_analysis"]
                or projection.get("geometry_candidate") != evidence["geometry"]
                or projection.get("source_launch_result_sha256") != joint["source_launch_result_sha256"]
                or projection.get("fixed_checkpoint_indices") != [5, 14, 23, 32]
                or projection.get("camera_frames_visually_inspected") != 12
                or projection.get("original_rgb_unchanged") is not True):
            raise RuntimeBundleError("moving_projection_source_binding_mismatch")
        analysis_entry = projection["projection_analysis"]
        analysis = self._read_json(analysis_entry)
        self._analysis_script("check_joint_probe_projection.py", analysis.get("script_sha256"))
        self._analysis_script("check_robot_projection.py", analysis.get("projection_helper_sha256"))
        figures = analysis.get("figures", [])
        required = [evidence["joint_analysis"], evidence["geometry"]]
        for row in joint["checkpoint_inputs"]:
            required.extend([row["robot_measurement"], row["camera_inventory"], *row["images"]])
        if (analysis.get("schema") != "robodojo-joint-probe-projection-analysis-1"
                or analysis.get("kinematics_analysis") != evidence["joint_analysis"]
                or analysis.get("fixed_checkpoint_indices") != [5, 14, 23, 32]
                or analysis.get("original_rgb_unchanged") is not True
                or analysis.get("full_test_authorized") is not False
                or [row.get("index") for row in figures] != [5, 14, 23, 32]
                or projection.get("reviewed_figures") != figures):
            raise RuntimeBundleError("moving_projection_source_binding_mismatch")
        self._source_files(analysis.get("source_files"), required)
        self._source_files(projection.get("source_files"), [analysis_entry, *analysis["source_files"], *figures])

    def _reset_closure(self, comparison, reference):
        closure = self._read("reset_closure")
        evidence = self.document["evidence"]
        program_path = str(Path(__file__).with_name("reset_binding.py").resolve())
        program = next(row for row in self.document["program_files"] if row["path"] == program_path)
        if (closure.get("schema") != "robodojo-reset-evidence-closure-1"
                or closure.get("reference") != evidence["reset_reference"]
                or closure.get("comparison") != evidence["reset_check"]
                or closure.get("reset_binding_program") != program
                or closure.get("robot_inventory_field") != "raw_robot_inventory"
                or closure.get("reproduced_comparison_sha256") != canonical_sha256(comparison)
                or closure.get("comparison_reproduced_exactly") is not True
                or closure.get("full_test_authorized") is not False
                or set(closure.get("fresh_images", {})) != set(CAMERAS)
                or set(comparison.get("actual_image_comparisons", {})) != set(CAMERAS)
                or any(closure["fresh_images"][name] != self._binding(comparison["actual_image_comparisons"][name])
                    for name in CAMERAS)):
            raise RuntimeBundleError("fresh_reset_source_binding_mismatch")
        required = [evidence["reset_reference"], evidence["reset_check"], program,
            reference["robot_file"], reference["camera_file"], *reference["image_files"].values(),
            closure["fresh_robot_measurement"], closure["fresh_camera_inventory"], *closure["fresh_images"].values()]
        self._source_files(closure.get("source_files"), required)
        key = (evidence["reset_closure"]["sha256"], evidence["reset_reference"]["sha256"], evidence["reset_check"]["sha256"])
        if self._reset_verified != key:
            state = self._read_json(closure["fresh_robot_measurement"])
            if state.get("schema") != "robodojo-native-characterization-measurement-1":
                raise RuntimeBundleError("fresh_reset_measurement_required")
            result = verify_fresh_reset(reference, state["raw_robot_inventory"],
                self._read_json(closure["fresh_camera_inventory"]),
                {name: row["path"] for name, row in closure["fresh_images"].items()})
            if result != comparison:
                raise RuntimeBundleError("fresh_reset_comparison_not_reproduced")
            # A second closure check prevents accepting a mutation during PNG decoding.
            self._source_files(closure["source_files"], required)
            self._reset_verified = key

    @staticmethod
    def _artifact_references(value, directory):
        rows = []
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                if not path.is_absolute():
                    path = directory / path
                rows.append({"path": str(path.resolve(strict=True)), "sha256": value["sha256"]})
            for item in value.values():
                rows.extend(VerifiedRuntimeBundle._artifact_references(item, directory))
        elif isinstance(value, list):
            for item in value:
                rows.extend(VerifiedRuntimeBundle._artifact_references(item, directory))
        return rows

    def _controller_closure(self, review, launch):
        evidence = self.document["evidence"]
        child = Path(evidence["controller_launch"]["path"]).parent / "worker/controller"
        rows = review.get("source_files")
        self._source_files(rows, [evidence["controller_launch"], evidence["native_isolation"]])
        by_path = {row["path"]: row for row in rows}
        required = []
        records = {}
        for name in ("controller_result.json", "preparation.json", "controller_preclose_ready.json"):
            spec = by_path.get(str(child / name))
            if spec is None:
                raise RuntimeBundleError("controller_raw_artifact_closure_missing")
            required.append(spec)
            records[name] = self._read_json(spec)
        result, ready = records["controller_result.json"], records["controller_preclose_ready.json"]
        preparation = records["preparation.json"]
        plan_entry = {"path": preparation.get("plan_path"), "sha256": CONTROLLER_PLAN_FILE_SHA256}
        if (review.get("controller_plan") != plan_entry
                or review.get("plan_content_sha256") != CONTROLLER_PLAN_CONTENT_SHA256
                or any(record.get("plan_file_sha256") != CONTROLLER_PLAN_FILE_SHA256
                    or record.get("plan_sha256") != CONTROLLER_PLAN_CONTENT_SHA256
                    or record.get("diagnostic_native_actions_planned") != CONTROLLER_DIAGNOSTIC_ACTIONS
                    for record in (launch, preparation))):
            raise RuntimeBundleError("reviewed_controller_plan_binding_required")
        self._source_files(rows, [plan_entry])
        plan = self._read_json(plan_entry)
        if (canonical_sha256(plan) != CONTROLLER_PLAN_CONTENT_SHA256
                or len(plan.get("stages", [])) != CONTROLLER_DIAGNOSTIC_STAGES
                or sum(row["native_actions"] for row in plan["stages"]) != CONTROLLER_DIAGNOSTIC_ACTIONS
                or plan.get("maximum_native_actions") != CONTROLLER_DIAGNOSTIC_ACTIONS
                or plan.get("maximum_physics_substeps") != CONTROLLER_DIAGNOSTIC_SUBSTEPS
                or self.document["parameters"]["controller"]["angular_speed_rad_s"] != .5):
            raise RuntimeBundleError("reviewed_controller_plan_binding_required")
        if (result.get("schema") != "robodojo-controller-characterization-result-1"
                or result.get("diagnostic_completed") is not True
                or result.get("diagnostic_native_actions") != CONTROLLER_DIAGNOSTIC_ACTIONS
                or result.get("diagnostic_physics_substeps") != CONTROLLER_DIAGNOSTIC_SUBSTEPS
                or launch.get("diagnostic_native_actions") != CONTROLLER_DIAGNOSTIC_ACTIONS
                or launch.get("diagnostic_physics_substeps") != CONTROLLER_DIAGNOSTIC_SUBSTEPS
                or launch.get("controller_result_sha256") != required[0]["sha256"]
                or launch.get("preparation_sha256") != required[1]["sha256"]
                or result.get("worker_isolation_sha256") != evidence["native_isolation"]["sha256"]
                or ready.get("schema") != "robodojo-controller-characterization-preclose-1"
                or ready.get("controller_result_sha256") != required[0]["sha256"]
                or ready.get("preparation_sha256") != required[1]["sha256"]
                or ready.get("child_pid") != launch.get("pid")
                or any(ready.get(k) is not True for k in ("ready_for_shutdown", "no_pending_exception",
                    "native_env_close_returned", "physx_monitor_healthy"))
                or ready.get("cleanup_errors") != []):
            raise RuntimeBundleError("completed_controller_raw_evidence_required")
        for record in records.values():
            required.extend(self._artifact_references(record, child))
        self._source_files(rows, required)
        independent = self._read("controller_independent_review")
        if (independent.get("schema") != "robodojo-controller-selection-independent-review-1"
                or independent.get("selection_supported") is not True
                or independent.get("controller_launch") != evidence["controller_launch"]
                or independent.get("controller_review") != evidence["controller_review"]
                or independent.get("controller_plan") != plan_entry
                or independent.get("plan_content_sha256") != CONTROLLER_PLAN_CONTENT_SHA256
                or independent.get("selected_parameters_sha256") != canonical_sha256(self.document["parameters"])
                or independent.get("scope") != "measured_robot_geometry_and_free_space_characterization"
                or any(independent.get(k) is not False for k in ("full_test_authorized",
                    "loaded_contact_validated", "workspace_wide_tracking_validated"))
                or not isinstance(independent.get("limitations"), list) or not independent["limitations"]):
            raise RuntimeBundleError("independent_controller_selection_review_required")
        self._source_files(independent.get("source_files"), [evidence["controller_review"], *rows])

    def verify(self):
        d = self.document
        if validate_control_backend(d.get("control_backend", "legacy_joint")) != self.control_backend:
            raise RuntimeBundleError("runtime_control_backend_changed")
        self._verify_file({"path": str(self.path), "sha256": self._hash})
        if (canonical_sha256(d) != self._content_hash
                or d.get("schema") != "robodojo-runtime-bundle-1"
                or d.get("scope") != "measured_robot_geometry_and_free_space_characterization"
                or d.get("full_test_authorized") is not False or d.get("asset_fidelity_passed") is not False
                or d.get("loaded_contact_validated") is not False or d.get("workspace_wide_tracking_validated") is not False
                or set(d.get("evidence", {})) != EVIDENCE
                or set(d.get("preparation_checks", {})) != PREPARATION_CHECKS
                or any(value is not True for value in d["preparation_checks"].values())):
            raise RuntimeBundleError("complete_scoped_preparation_bundle_required")
        for entry in [*d["evidence"].values(), *d.get("program_files", [])]:
            self._verify_file(entry)
        required_programs = {Path(__file__).resolve(), *[Path(__file__).with_name(name).resolve() for name in
            ("approval.py", "controller.py", "fragment_executor.py", "native_measurements.py", "native_robot_inventory.py", "reset_binding.py", "schema.py")]}
        if not required_programs <= {Path(row["path"]).resolve() for row in d.get("program_files", [])}:
            raise RuntimeBundleError("measurement_program_closure_missing")
        geometry, joint = self._read("geometry"), self._read("joint_analysis")
        transform = np.asarray(geometry.get("link6_from_tool"), dtype=float)
        if (geometry.get("schema") != "robodojo-physical-tool-candidate-1" or transform.shape != (4, 4)
                or not np.isfinite(transform).all() or not np.array_equal(transform[:3, :3], np.eye(3))
                or not np.array_equal(transform[3], [0, 0, 0, 1])
                or not geometry.get("source_files")):
            raise RuntimeBundleError("source_bound_translation_only_physical_tool_required")
        self._source_files(geometry["source_files"])
        self._analysis_script("derive_tool_candidate.py", geometry.get("script_sha256"))
        if (joint.get("schema") != "robodojo-native-joint-probe-analysis-1"
                or joint.get("geometry_candidate_sha256") != d["evidence"]["geometry"]["sha256"]
                or joint.get("source_binding_verified") is not True
                or joint.get("kinematics_geometry_passed") is not True
                or joint.get("all_recorded_pose_checks_passed") is not True
                or joint.get("thresholds") != THRESHOLDS
                or joint.get("diagnostic_native_actions_analyzed") != 37):
            raise RuntimeBundleError("measured_joint_geometry_validation_required")
        self._joint_closure(joint)
        reset, reference = self._read("reset_check"), self._read("reset_reference")
        if (reset.get("schema") != "robodojo-reset-comparison-1" or reset.get("passed") is not True
                or reset.get("start_state_signature", {}).get("reference_sha256") != canonical_sha256(reference)):
            raise RuntimeBundleError("matching_fresh_reset_validation_required")
        self._reset_closure(reset, reference)
        pause = self._read("pause")
        if (pause.get("schema") != "robodojo-native-paused-physics-1"
                or pause.get("passed") is not True or pause.get("requested_seconds") != 300
                or type(pause.get("elapsed_monotonic_s")) not in (int, float)
                or not math.isfinite(pause["elapsed_monotonic_s"]) or pause["elapsed_monotonic_s"] < 300
                or any(pause.get(key) is not True for key in ("clock_unchanged", "step_count_unchanged",
                    "dof_positions_unchanged", "monitor_healthy"))
                or any(type(pause.get(k)) is not int or pause[k] != 0 for k in
                    ("simulation_updates_during_wait", "benchmark_actions", "model_calls"))
                or not isinstance(pause.get("before"), dict) or pause.get("before") != pause.get("after")
                or set(pause["before"]) != {"native_physics_time_s", "native_sim_step_calls", "articulation_dof_positions"}
                or type(pause["before"]["native_sim_step_calls"]) is not int
                or pause["before"]["native_sim_step_calls"] < 0
                or set(pause["before"]["articulation_dof_positions"]) != {"left", "right"}
                or any(np.asarray(v).shape != (1, 8) or not np.isfinite(np.asarray(v, dtype=float)).all()
                    for v in pause["before"]["articulation_dof_positions"].values())
                or not math.isfinite(pause["before"]["native_physics_time_s"])
                or not all(type(pause.get(k)) in (int, float) and math.isfinite(pause[k])
                    for k in ("start_monotonic_s", "end_monotonic_s"))
                or not math.isclose(pause["end_monotonic_s"]-pause["start_monotonic_s"],
                    pause["elapsed_monotonic_s"], abs_tol=1e-6, rel_tol=0)):
            raise RuntimeBundleError("real_paused_physics_validation_required")
        isolation = self._read("native_isolation")
        if (isolation.get("schema") != "robodojo-worker-unix-isolation-proof-1"
                or isolation.get("uid") != 65534 or isolation.get("gid") != 65534
                or isolation.get("supplementary_groups") != [] or isolation.get("no_new_privs") is not True
                or isolation.get("effective_permitted_inheritable_ambient_capabilities") != 0
                or isolation.get("access_denied") != {"credential_directory": True, "parent_environment": True}):
            raise RuntimeBundleError("isolated_native_worker_proof_required")
        review, launch = self._read("controller_review"), self._read("controller_launch")
        if (review.get("schema") != "robodojo-controller-characterization-review-1"
                or review.get("preparation_checks_passed") is not True
                or review.get("selected_parameters") != d.get("parameters")
                or review.get("launch_result_sha256") != d["evidence"]["controller_launch"]["sha256"]
                or review.get("selection_supported_by_recorded_stages") is not True
                or review.get("full_test_authorized") is not False
                or launch.get("diagnostic_completed") is not True or launch.get("returncode") != 0
                or launch.get("termination_signals") != []
                or launch.get("worker_isolation_verified") is not True
                or type(isolation.get("pid")) is not int or isolation["pid"] != launch.get("pid")
                or launch.get("gpu_processes_released") is not True or launch.get("gpu_memory_released") is not True):
            raise RuntimeBundleError("reviewed_native_controller_characterization_required")
        self._controller_closure(review, launch)
        asset = self._read("asset_assessment")
        if (asset.get("schema") != "robodojo-native-asset-diagnostic-assessment-1"
                or asset.get("asset_fidelity_passed") is not False
                or asset.get("full_test_authorized") is not False):
            raise RuntimeBundleError("published_asset_limitation_must_remain_explicit")
        projection = self._read("joint_projection_review")
        if (projection.get("schema") != "robodojo-joint-probe-projection-review-1"
                or projection.get("visible_robot_projection_supported") is not True
                or projection.get("full_test_authorized") is not False):
            raise RuntimeBundleError("actual_moving_camera_projection_review_required")
        self._projection_closure(projection, joint)
        if self.control_backend == "official_ee":
            from .official_control_readiness import verify_official_readiness
            self._official_setup = verify_official_readiness(d.get("official_control"),
                geometry=d["evidence"]["geometry"], reset_reference=d["evidence"]["reset_reference"],
                parameters=d["parameters"])
            required = {Path(__file__).with_name(name).resolve() for name in
                ("official_control.py", "official_control_setup.py", "official_control_readiness.py",
                 "native_config.py", "native_episode.py", "native_worker_protocol.py", "native_worker_client.py",
                 "native_episode_runner.py", "client.py")}
            if not required <= {Path(row["path"]).resolve() for row in d["program_files"]}:
                raise RuntimeBundleError("official_control_program_closure_missing")
        elif "official_control" in d:
            raise RuntimeBundleError("legacy_bundle_cannot_claim_official_qualification")
        return True

    def activate_official_dependencies(self, cache_root):
        """Load only the frozen, verified task-local overlay in the native worker."""
        from .official_control_setup import activate_setup_record
        self.verify()
        if self.control_backend != "official_ee":
            raise RuntimeBundleError("official_backend_required")
        record = activate_setup_record(self.document["official_control"]["setup"]["path"], cache_root)
        return record["overlay_target"]

    def measure(self, env):
        self.verify()
        key = id(env)
        if key not in self._readers:
            self._readers[key] = (env, NativeMeasurements(env,
                tool_candidate=native_link6_tool_candidate(self.native_root)))
        bound_env, reader = self._readers[key]
        if bound_env is not env:
            raise RuntimeBundleError("measurement_environment_identity_changed")
        sample = reader.sample()
        arms = {name: replace(state, tcp_offset_link6=self._tool_transform[:3, 3].copy())
                for name, state in sample.arms.items()}
        for state in arms.values():
            measured_tcp_pose(state)
        return RobotMeasurement(arms, deepcopy(sample.finger_position_m))

    def public_calibration(self, camera_inventory):
        self.verify()
        if camera_inventory.get("schema") != "robodojo-native-camera-inventory-1":
            raise RuntimeBundleError("live_native_camera_inventory_required")
        cameras = {}
        for name in ("head", "left_wrist", "right_wrist"):
            c = camera_inventory["cameras"][name]
            cameras[name] = {key: deepcopy(c[key]) for key in ("width", "height", "intrinsics", "world_from_optical")}
            cameras[name]["pose_source"] = "native Camera.get_view_matrix_ros; synchronized after render_for_capture"
        tools = {name: {"link_name": "link6", "link6_from_tool": self._tool_transform.tolist(),
            "source_sha256": self._tool_hash, "finger_gap_formula": "sum_of_measured_finger_joint_positions",
            "finger_gap_semantics": "joint_separation_proxy"} for name in ("left", "right")}
        return validate_calibration({"version": CALIBRATION_VERSION, "cameras": cameras, "tools": tools})

    def sensor_signature(self, camera_inventory):
        return {"schema": "robodojo-runtime-sensor-binding-1", "bundle_sha256": self._hash,
                "calibration": calibration_binding(self.public_calibration(camera_inventory))}

    @property
    def controller_capabilities(self):
        if self.control_backend == "official_ee":
            return validate_controller_capabilities({"version": "robodojo-uc71-official-ee-controller-1",
                "status": "validated", **self.monitoring_settings.checked()})
        c = self.controller_config
        return validate_controller_capabilities({"version": "robodojo-uc71-robot-only-controller-1",
            "status": "validated", "action_translation_m": {
                "transit": c.transit_translation_speed_m_s/c.control_hz,
                "contact": c.contact_translation_speed_m_s/c.control_hz},
            "action_rotation_rad": c.angular_speed_rad_s/c.control_hz,
            "max_joint_step_rad": c.joint_target_lead_rad,
            "joint_target_lead_rad": c.joint_target_lead_rad, **self.monitoring_settings.checked()})
