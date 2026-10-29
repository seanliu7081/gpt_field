"""Offline evidence-contract tests; no simulator, subprocess, or model runs.

PNG decoding is mocked here; the preserved real PNGs were decoded separately
during the supervisor's implementation checks. These cases test evidence binding.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import provision_native_preview as supervisor


class NativePreviewEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uc71-evidence-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.binding = {"gpu_uuid": "GPU-12345678-0000-0000-0000-000000000000",
                        "pci_bus_id": "0000:01:00.0", "kit_index": 0, "cuda_ordinal": 0}
        self.identity = {"pid": 123, "start_ticks": 456}
        self.authorization = {"path": "authorization", "sha256": "auth-sha",
                              "full_test_authorized": False, "model_calls_authorized": False}
        self.fidelity = {"standard_ready": False, "asset_fidelity_passed": False,
                         "dependency_closure_complete": False,
                         "asset_diagnostic_assessment": {"path": "assessment", "sha256": "assessment-sha"}}
        self.expected = {**self.fidelity, "source_manifest_sha256": "source-sha",
            "asset_manifest_sha256": "assets-sha", "source": {}, "camera_gate": {},
            "layout_path": "layout-0", "layout_sha256": "layout-sha"}
        self.helpers = {str(supervisor.SCRIPTS / name): "helper-sha" for name in (
            "prepare_native_preview.py", "new_host_camera.py", "provision_native_common.py")}
        for name in ("native_config.py", "native_robot_inventory.py", "native_measurements.py"):
            self.helpers[str(supervisor.BASE / "src/robodojo_uc71" / name)] = "helper-sha"
        self.put("native_configuration.json", {})
        self.put("loaded_module_provenance.json", {"numpy": {"version": "fixture"}})
        self.preparation = {**self.expected, "authorization": self.authorization, "binding": self.binding,
            "pause_seconds": 0,
            "full_test_authorized": False, "model_calls": 0, "benchmark_actions": 0,
            "native_configuration_sha256": supervisor.digest(self.path / "native_configuration.json"),
            "source_file_sha256": dict(self.helpers)}
        self.put("robot_inventory_error.json", {"schema": "robodojo-native-robot-inventory-error-1",
            "scope": "setup_evaluator_only", "stage": "synthetic_measurement", "error_type": "ValueError",
            "model_input_eligible": False, "calibration_status": "uncalibrated_candidate"})
        self.put("camera_inventory_error.json", {"schema": "robodojo-native-camera-inventory-error-1",
            "scope": "setup_evaluator_only", "stage": "synthetic_camera", "error_type": "ValueError",
            "model_input_eligible": False, "standard_ready": False, "calibration_status": "uncalibrated_candidate"})
        self.images = []
        for name in sorted(supervisor.EXPECTED_CAMERAS):
            path = self.path / (name + ".png")
            path.write_bytes(b"synthetic-unit-fixture-decoded-by-mock")
            self.images.append({"camera": name, "path": path.name, "sha256": supervisor.digest(path),
                "shape": [480, 640, 3], "dtype": "uint8", "std": 1.0, "bytes": path.stat().st_size})
        self.result = {**self.fidelity, "schema": "robodojo-native-preview-result-1",
            "preview_render_passed": True, "diagnostic_render_passed": True, "binding": self.binding,
            "shutdown_mode": "stock-fast", "reset_calls": 1, "native_client_reset_calls": 1,
            "model_calls": 0, "benchmark_actions": 0, "native_take_action_count": [0],
            "physics_device": "cpu", "use_fabric": False, "native_planner_instances": 0,
            "native_ik_solver_instances": 0, "reward_predicates_registered": True,
            "task_success_assessed": False, "full_test_authorized": False,
            "completed_at_monotonic_s": 1.0, "native_setup_counts": {"native_sim_step_calls": 500, "native_render_calls": 1},
            "pause_seconds": 0,
            "pause_validation": {"requested_seconds": 0, "performed": False, "passed": None, "path": None, "sha256": None},
            "layout_path": "layout-0", "layout_sha256": "layout-sha", "images": self.images,
            "capture_physics_time_before_s": 1.0, "capture_physics_time_after_s": 1.0,
            "loaded_module_provenance_sha256": supervisor.digest(self.path / "loaded_module_provenance.json"),
            "robot_inventory": {"status": "extraction_failed", "path": "robot_inventory_error.json",
                "sha256": supervisor.digest(self.path / "robot_inventory_error.json"),
                "calibration_status": "uncalibrated_candidate"},
            "camera_inventory": {"status": "extraction_failed", "path": "camera_inventory_error.json",
                "sha256": supervisor.digest(self.path / "camera_inventory_error.json"),
                "calibration_status": "uncalibrated_candidate"}}
        self.ready = {**self.fidelity, "schema": "robodojo-native-preview-preclose-1",
            "ready_for_shutdown": True, "no_pending_exception": True, "native_env_close_returned": True,
            "physx_monitor_healthy": True, "cleanup_errors": [], "image_count": 3, "binding": self.binding,
            "child_pid": 123, "shutdown_mode": "stock-fast", "model_calls": 0, "benchmark_actions": 0,
            "full_plugin_cleanup_proven": False, "full_teardown_returned": False, "monotonic_s": 2.0}
        self.ready["pause_seconds"] = 0
        self.log = self.path / "parent.log"
        self.log.write_text("| 0 | NVIDIA GPU | Yes: 0 |\n| | | | | | | 12345678.. |\n| | | | | | 1 |\n")
        self.put("renderer_identity.json", supervisor.renderer_table_identity(self.log.read_text(), self.binding))
        self.put("cuda_identity.json", {"gpu_uuid": self.binding["gpu_uuid"], "pci_bus_id": "0000:01:00.0",
            "cuda_ordinal": 0, "device_count": 1, "primary_context_created": False,
            "child_pid": 123, "child_start_ticks": 456})
        decoded = [{"format": "PNG", "mode": "RGB", "shape": [480, 640, 3], "dtype": "uint8", "std": 1.0}] * 3
        self.decoder = patch.object(supervisor.subprocess, "check_output", return_value=json.dumps(decoded))
        self.decoder.start(); self.addCleanup(self.decoder.stop)

    def put(self, name, value):
        (self.path / name).write_text(json.dumps(value))

    def verify(self, pause_seconds=0):
        self.put("preparation.json", self.preparation)
        self.result["preparation_sha256"] = supervisor.digest(self.path / "preparation.json")
        self.put("preview_result.json", self.result)
        self.ready.update(preparation_sha256=self.result["preparation_sha256"],
                          preview_result_sha256=supervisor.digest(self.path / "preview_result.json"))
        self.put("preview_preclose_ready.json", self.ready)
        return supervisor.verify_preview_evidence(self.path, self.binding, self.identity, self.expected,
            self.helpers, self.log, Path(sys.executable), self.authorization, pause_seconds=pause_seconds)

    def test_optional_extraction_failure_stays_separate(self):
        result = self.verify()
        self.assertTrue(result["native_render_verified"])
        self.assertFalse(result["robot_inventory"]["captured"])
        self.assertFalse(result["robot_inventory"]["calibration_passed"])
        self.assertEqual(result["robot_inventory"]["failure"]["stage"], "synthetic_measurement")

    def test_preclose_must_have_no_pending_exception(self):
        self.ready["no_pending_exception"] = False
        with self.assertRaises(ValueError): self.verify()

    def test_preclose_must_follow_result(self):
        self.ready["monotonic_s"] = 0.5
        with self.assertRaises(ValueError): self.verify()

    def test_policy_calls_and_actions_are_rejected(self):
        original = copy.deepcopy(self.result)
        for key in ("model_calls", "benchmark_actions"):
            with self.subTest(key=key):
                self.result = {**original, key: 1}
                with self.assertRaises(ValueError): self.verify()

    def test_asset_fidelity_cannot_be_upgraded(self):
        self.result["asset_fidelity_passed"] = True
        with self.assertRaises(ValueError): self.verify()

    def test_image_hash_is_bound(self):
        self.images[0]["sha256"] = "changed"
        with self.assertRaises(ValueError): self.verify()

    def test_helper_manifest_must_be_complete(self):
        self.preparation["source_file_sha256"].pop(next(iter(self.helpers)))
        with self.assertRaises(ValueError): self.verify()

    def test_capture_cannot_advance_physics(self):
        self.result["capture_physics_time_after_s"] = 2.0
        with self.assertRaises(ValueError): self.verify()

    def test_inventory_cannot_claim_calibration(self):
        self.result["robot_inventory"]["calibration_status"] = "certified"
        with self.assertRaises(ValueError): self.verify()

    def test_extension_import_failure_cannot_pass(self):
        self.log.write_text(self.log.read_text() + "ModuleNotFoundError: No module named 'hid'\n")
        with self.assertRaisesRegex(ValueError, "startup/import errors"): self.verify()

    def test_complete_pause_binds_before_and_after_images(self):
        for value in (self.preparation, self.result, self.ready):
            value["pause_seconds"] = 300
        self.result["completed_at_monotonic_s"] = 400.0
        self.ready["monotonic_s"] = 410.0
        before_dir = self.path / "before_pause"
        before_dir.mkdir()
        before_images = copy.deepcopy(self.images)
        for row in before_images:
            (before_dir / row["path"]).write_bytes((self.path / row["path"]).read_bytes())
            row["path"] = "before_pause/" + row["path"]
        sample = {"native_physics_time_s": 1.0, "native_sim_step_calls": 500,
                  "articulation_dof_positions": {"left": [[0.0] * 8], "right": [[0.0] * 8]}}
        pause = {"schema": "robodojo-native-paused-physics-1", "requested_seconds": 300,
            "start_monotonic_s": 10.0, "end_monotonic_s": 310.5, "elapsed_monotonic_s": 300.5,
            "before": sample, "after": sample, "clock_unchanged": True, "step_count_unchanged": True,
            "dof_positions_unchanged": True, "monitor_healthy": True, "passed": True,
            "model_calls": 0, "benchmark_actions": 0, "simulation_updates_during_wait": 0,
            "before_pause_images": before_images}
        self.put("pause_validation.json", pause)
        self.result["pause_validation"] = {"requested_seconds": 300, "performed": True, "passed": True,
            "path": "pause_validation.json", "sha256": supervisor.digest(self.path / "pause_validation.json")}
        decoded = [{"format": "PNG", "mode": "RGB", "shape": [480, 640, 3], "dtype": "uint8", "std": 1.0}] * 6
        with patch.object(supervisor.subprocess, "check_output", return_value=json.dumps(decoded)):
            self.assertTrue(self.verify(pause_seconds=300)["pause_validation"]["measurements_unchanged"])
            pause["before_pause_images"].pop()
            self.put("pause_validation.json", pause)
            self.result["pause_validation"]["sha256"] = supervisor.digest(self.path / "pause_validation.json")
            with self.assertRaisesRegex(ValueError, "all three native images"):
                self.verify(pause_seconds=300)


class PhysicsPauseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uc71-pause-test-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        sample = {"native_physics_time_s": 3.2, "native_sim_step_calls": 800,
                  "articulation_dof_positions": {"left": [[0.0] * 8], "right": [[0.0] * 8]}}
        self.value = {"schema": "robodojo-native-paused-physics-1", "requested_seconds": 300,
            "start_monotonic_s": 10.0, "end_monotonic_s": 310.5, "elapsed_monotonic_s": 300.5,
            "before": sample, "after": copy.deepcopy(sample), "clock_unchanged": True,
            "step_count_unchanged": True, "dof_positions_unchanged": True, "monitor_healthy": True,
            "passed": True, "model_calls": 0, "benchmark_actions": 0, "simulation_updates_during_wait": 0}

    def verify(self):
        path = self.path / "pause_validation.json"
        path.write_text(json.dumps(self.value))
        result = {"completed_at_monotonic_s": 320.0, "pause_validation": {"requested_seconds": 300,
            "performed": True, "passed": True, "path": path.name, "sha256": supervisor.digest(path)}}
        return supervisor.verify_pause(self.path, result, 300)

    def test_full_wait_with_unchanged_measurements(self):
        self.assertTrue(self.verify()["measurements_unchanged"])

    def test_short_wait_rejected(self):
        self.value.update(end_monotonic_s=309.0, elapsed_monotonic_s=299.0)
        with self.assertRaises(ValueError): self.verify()

    def test_hidden_physics_step_rejected(self):
        self.value["after"]["native_sim_step_calls"] += 1
        with self.assertRaises(ValueError): self.verify()

    def test_hidden_joint_motion_rejected(self):
        self.value["after"]["articulation_dof_positions"]["left"][0][0] = 0.1
        with self.assertRaises(ValueError): self.verify()

    def test_health_failure_rejected(self):
        self.value["monitor_healthy"] = False
        with self.assertRaises(ValueError): self.verify()

    def test_nonfinite_measurement_rejected(self):
        self.value["before"]["articulation_dof_positions"]["right"][0][0] = float("nan")
        with self.assertRaises(ValueError): self.verify()


if __name__ == "__main__":
    unittest.main()
