"""Offline safety checks. No CUDA/Isaac imports, model calls, or GPU allocation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import new_host_camera as camera
import provision_new_host_camera as launch


class CameraDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"gpu_uuid": "GPU-e1a460e9-e272-b227-222e-0dbb77c248c5",
                        "pci_bus_id": "00000000:81:00.0", "kit_index": 4, "cuda_ordinal": 4}
        self.log = (SCRIPTS.parent / "validation/provisioning/a1_attempt_003/stdout_stderr.log").read_text()

    def test_actual_kit_truncated_uuid_and_hex_bus(self):
        result = camera.renderer_table_identity(self.log, self.binding)
        self.assertTrue(result["uuid_prefix_and_pci_bus_match"])
        self.assertFalse(result["kit_exposes_full_uuid"])

    def test_wrong_uuid_or_bus_rejected(self):
        for key, value in (("gpu_uuid", "GPU-deadbeef-e272-b227-222e-0dbb77c248c5"),
                           ("pci_bus_id", "0000:01:00.0"), ("kit_index", 0)):
            with self.subTest(key=key):
                self.assertFalse(camera.renderer_table_identity(self.log, {**self.binding, key: value})["uuid_prefix_and_pci_bus_match"])

    def test_second_active_gpu_rejected(self):
        log = self.log.replace("| 0   | NVIDIA GeForce RTX 4090          |        |", "| 0   | NVIDIA GeForce RTX 4090          | Yes: 1 |")
        self.assertFalse(camera.renderer_table_identity(log, self.binding)["uuid_prefix_and_pci_bus_match"])

    def test_pci_domain_normalization(self):
        self.assertEqual(camera.normalize_pci("00000000:01:00.0"), "0000:01:00.0")

    def test_single_device_binding_rejects_multi_gpu_and_busy(self):
        binding = {**self.binding, "kit_index": 0, "cuda_ordinal": 0}
        gpu = {"index": 0, "uuid": binding["gpu_uuid"], "pci_bus_id": "0000:81:00.0", "processes": []}
        self.assertEqual(launch.validate_single_gpu([gpu], binding, require_idle=True), gpu)
        with self.assertRaises(ValueError):
            launch.validate_single_gpu([gpu, gpu], binding)
        with self.assertRaises(ValueError):
            launch.validate_single_gpu([{**gpu, "processes": [{"pid": 1}]}], binding, require_idle=True)

    def test_environment_does_not_inherit_credentials_masks_or_python_paths(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "CUDA_VISIBLE_DEVICES": "4", "PYTHONPATH": "/bad", "LD_LIBRARY_PATH": "/bad"}):
            env = launch.child_environment(Path("/test/bin/python"), Path("/attempt"), self.binding, Path("/icd.json"), None)
        for key in ("OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["HOME"], "/attempt/home")

    def test_crash_precedes_stale_monitor_observation(self):
        result = launch.classify_terminal_exit(-11, "ownership_unresolved", [], child_pid=123)
        self.assertEqual(result["kind"], "child_signal_exit")
        self.assertEqual(result["signal"], 11)

    def test_camera_cleanup_continues_after_detach_error(self):
        calls = []
        class Annotator:
            def detach(self):
                calls.append("detach")
                raise ValueError("record this independently")
        class Product:
            def destroy(self):
                calls.append("destroy")
        class App:
            def close(self):
                calls.append("close")
        def progress(phase, **details):
            self.assertIsInstance(phase, str)
        result = camera.cleanup_camera(App(), Annotator(), Product(), True, progress)
        self.assertEqual(calls, ["detach", "destroy", "close"])
        self.assertTrue(result["app_close_returned"])
        self.assertEqual(result["errors"][0]["phase"], "annotator_detach")

    def test_fast_shutdown_exit_zero_rejects_missing_preclose(self):
        result = {"passed": True, "capture_requests": 8, "synchronous_captures_completed": 8,
                  "wait_for_render": True}
        checked = launch.shutdown_evidence("stock-fast", result, {}, {}, returncode=0,
            termination_signals=[], rgb_sha256="rgb", camera_result_sha256="result",
            child_pid=123, gpu_uuid="GPU-test")
        self.assertFalse(checked["shutdown_contract_satisfied"])
        self.assertFalse(checked["full_teardown_returned"])

    def test_fast_shutdown_requires_bound_complete_capture_evidence(self):
        result = {"passed": True, "capture_requests": 8, "synchronous_captures_completed": 8,
                  "wait_for_render": True}
        ready = {"schema": "robodojo-camera-preclose-ready-1", "shutdown_mode": "stock-fast",
            "camera_passed": True, "capture_requests": 8, "synchronous_captures_completed": 8,
            "wait_for_render": True, "no_pending_exception": True, "annotator_detached": True,
            "render_product_destroyed": True, "cleanup_errors": [], "child_pid": 123,
            "gpu_uuid": "GPU-test", "model_calls": 0, "benchmark_actions": 0,
            "durability": "file_and_parent_directory_fsync_before_close",
            "rgb_sha256": "rgb", "camera_result_sha256": "result"}
        def check(record, signals=None):
            return launch.shutdown_evidence("stock-fast", result, {}, record, returncode=0,
                termination_signals=signals or [], rgb_sha256="rgb", camera_result_sha256="result",
                child_pid=123, gpu_uuid="GPU-test")
        self.assertTrue(check(ready)["shutdown_contract_satisfied"])
        self.assertFalse(check(ready)["full_plugin_cleanup_proven"])
        for key, value in (("synchronous_captures_completed", 7), ("no_pending_exception", False),
                           ("annotator_detached", False), ("rgb_sha256", "changed")):
            with self.subTest(key=key):
                self.assertFalse(check({**ready, key: value})["shutdown_contract_satisfied"])
        self.assertFalse(check(ready, [{"signal": 15}])["shutdown_contract_satisfied"])

    def test_preclose_callback_follows_detach_destroy_and_precedes_close(self):
        calls = []
        class Annotator:
            def detach(self): calls.append("detach")
        class Product:
            def destroy(self): calls.append("destroy")
        class App:
            def close(self): calls.append("close")
        def before_close(state):
            self.assertEqual(state["errors"], [])
            self.assertEqual(state["completed"], ["annotator_detach", "render_product_destroy"])
            calls.append("preclose")
        camera.cleanup_camera(App(), Annotator(), Product(), True, lambda *a, **kw: None, before_close)
        self.assertEqual(calls, ["detach", "destroy", "preclose", "close"])

    def test_exiting_child_alias_survives_live_member_filter(self):
        child = {"pid": 10169, "start_ticks": 759390307, "nspid": [10169]}
        known = {10169: child["start_ticks"]}
        zombie = {**child, "state": "Z"}
        aliases = launch.gpu_process_aliases(known, child_identity=child,
            child_unreaped=True, process_lookup=lambda pid: zombie)
        self.assertIn(10169, aliases)
        # The two NVML rows in camera003 refer to this same previously verified PID.
        gpu_rows = [{"pid": 10169, "type": "C+G"}, {"pid": 10169, "type": "C"}]
        self.assertEqual([r for r in gpu_rows if r["pid"] not in aliases], [])
        missing_aliases = launch.gpu_process_aliases(known, child_identity=child,
            child_unreaped=True, process_lookup=lambda pid: None)
        self.assertIn(10169, missing_aliases)

    def test_exit_alias_rejects_reused_new_or_reaped_pid(self):
        child = {"pid": 10169, "start_ticks": 759390307, "nspid": [10169]}
        known = {10169: child["start_ticks"], 10170: 777}
        reused = {**child, "start_ticks": child["start_ticks"] + 1}
        self.assertEqual(launch.gpu_process_aliases(known, child_identity=child,
            child_unreaped=True, process_lookup=lambda pid: reused if pid == 10169 else None), set())
        self.assertEqual(launch.gpu_process_aliases(known, child_identity=child,
            child_unreaped=False, process_lookup=lambda pid: None), set())
        aliases = launch.gpu_process_aliases(known, child_identity=child,
            child_unreaped=True, process_lookup=lambda pid: None)
        self.assertNotIn(10170, aliases)
        self.assertNotIn(12345, aliases)

    def test_owned_process_cleanup_without_proc_identity(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        try:
            signals = launch.terminate_owned(proc, {})
            self.assertIsNotNone(proc.poll())
            self.assertTrue(signals)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def test_owned_process_cleanup_with_stable_identity(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        try:
            identity = launch.process_record(proc.pid)
            self.assertIsNotNone(identity)
            known = {proc.pid: identity["start_ticks"]}
            launch.terminate_owned(proc, known)
            self.assertIsNotNone(proc.poll())
            self.assertEqual(launch.owned_processes(proc.pid, known), [])
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


if __name__ == "__main__":
    unittest.main()
