"""Offline fixtures for the read-only chat status helper."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_status.py"
SPEC = importlib.util.spec_from_file_location("experiment_status", SCRIPT)
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


class ExperimentStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runs"
        self.root.mkdir()

    def write(self, run, name, data, stamp):
        directory = self.root / run
        directory.mkdir(exist_ok=True)
        path = directory / name
        path.write_text(json.dumps(data) if not isinstance(data, str) else data)
        os.utime(path, (stamp, stamp))
        return path

    def read(self, run="one"):
        return status.summarize_run(self.root, run, now=1000)

    def test_false_and_zero_are_preserved_and_incomplete_time_is_attempt(self):
        self.write("one", "summary.json", {"status": "subscription_error", "steps": 0,
            "api_calls": 0, "libero_success": False, "completion_time_eligible": False,
            "control_without_video_s": 0.0}, 100)
        result = self.read()
        self.assertIs(result["raw_libero_success"], False)
        self.assertIs(result["completion_time_eligible"], False)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["timing_kind"], "attempt")
        self.assertIn("Attempt time", status.format_run(result))

    def test_newer_active_state_does_not_inherit_old_terminal_outcome(self):
        self.write("one", "summary.json", {"status": "success", "steps": 50,
            "libero_success": True, "completion_time_eligible": True,
            "control_without_video_s": 12}, 100)
        self.write("one", "run_state.json", {"status": "running", "step": 10,
            "budget": {"api_calls": 2}}, 200)
        result = self.read()
        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["raw_libero_success"])
        self.assertIsNone(result["control_without_video_s"])
        self.assertEqual(result["model_calls"], 2)
        self.assertIn("liveness is unverified", result["status_note"])
        self.assertEqual(result["age_seconds"], 800)

    def test_newer_summary_can_finish_older_running_state(self):
        self.write("one", "run_state.json", {"status": "running", "step": 9,
            "task_key": "t1"}, 100)
        self.write("one", "summary.json", {"status": "success", "steps": 10,
            "task_key": "t1", "libero_success": True, "completion_time_eligible": True,
            "control_without_video_s": 5}, 200)
        self.assertEqual(self.read()["timing_kind"], "completion")

    def test_model_requests_include_failed_call_not_only_accepted_decisions(self):
        self.write("one", "summary.json", {"status": "subscription_error", "steps": 10,
            "api_calls": 43, "model_calls": 42}, 100)
        self.assertEqual(self.read()["model_calls"], 43)
        self.write("one", "run_state.json", {"status": "running", "step": 11,
            "model_calls": 42, "budget": {"api_calls": 44}}, 200)
        self.assertEqual(self.read()["model_calls"], 44)
        self.write("one", "run_state.json", {"status": "subscription_error", "step": 10,
            "model_calls": 42, "budget": {"api_calls": 44}}, 300)
        self.assertEqual(self.read()["model_calls"], 44)

    def test_newer_terminal_state_can_use_matching_summary_timing(self):
        self.write("one", "summary.json", {"status": "success", "steps": 10,
            "libero_success": True, "completion_time_eligible": True,
            "control_without_video_s": 5}, 100)
        self.write("one", "run_state.json", {"status": "success", "step": 10}, 200)
        result = self.read()
        self.assertEqual(result["status_source"], "run_state.json")
        self.assertEqual(result["timing_kind"], "completion")

    def test_conflicting_summary_does_not_claim_success(self):
        self.write("one", "run_state.json", {"status": "running", "step": 20}, 100)
        self.write("one", "summary.json", {"status": "success", "steps": 10,
            "libero_success": True, "completion_time_eligible": True,
            "control_without_video_s": 5}, 200)
        result = self.read()
        self.assertEqual(result["status"], "conflicting_records")
        self.assertIsNone(result["raw_libero_success"])
        self.assertIsNone(result["timing_kind"])

    def test_partial_new_record_does_not_revive_old_success(self):
        self.write("one", "summary.json", {"status": "success", "libero_success": True}, 100)
        self.write("one", "run_state.json", '{"status":', 200)
        result = self.read()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["last_readable_status"], "success")
        self.assertIsNone(result["raw_libero_success"])
        self.assertTrue(result["warnings"])

    def test_missing_records_are_reported_without_writes(self):
        (self.root / "one").mkdir()
        self.assertEqual(self.read()["status"], "unknown")
        self.assertEqual(list((self.root / "one").iterdir()), [])

    def test_latest_discovery_is_immediate_and_ignores_escaped_symlink(self):
        self.write("one", "summary.json", {"status": "success"}, 100)
        self.write("two", "run_state.json", {"status": "running"}, 200)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "summary.json").write_text('{"status":"success"}')
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        nested = self.root / "container" / "nested"
        nested.mkdir(parents=True)
        (nested / "summary.json").write_text('{}')
        self.assertEqual(status.latest_run_ids(self.root, 5), ["two", "one"])
        with self.assertRaises(ValueError):
            status.summarize_run(self.root, "escape")
        for run_id in ("../outside", "one/../two", "/tmp", "..", "one\\two"):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                status.summarize_run(self.root, run_id)

    def test_record_and_video_symlinks_cannot_escape_run(self):
        run = self.root / "one"
        run.mkdir()
        outside = Path(self.temp.name) / "secret"
        outside.write_text('{"status":"secret"}')
        (run / "summary.json").symlink_to(outside)
        (run / "episode.mp4").symlink_to(outside)
        result = self.read()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["videos"], [])

    def test_cli_read_only_and_only_finalized_top_level_video_paths(self):
        self.write("one", "summary.json", {"status": "user_stop", "steps": 10}, 100)
        run = self.root / "one"
        (run / "episode.mp4").write_bytes(b"fixture")
        (run / "episode.partial.mp4").write_bytes(b"unfinished")
        (run / "nested").mkdir()
        (run / "nested" / "episode.mp4").write_bytes(b"nested")
        before = {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        completed = subprocess.run([sys.executable, str(SCRIPT), "--runs-root", str(self.root),
            "--json"], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)["runs"][0]
        self.assertEqual(result["videos"], [str(run / "episode.mp4")])
        after = {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_nonfinite_and_boolean_metrics_are_not_reported_as_numbers(self):
        self.write("one", "summary.json", {"status": "success", "steps": True,
            "api_calls": False, "control_without_video_s": float("nan"),
            "libero_success": False, "completion_time_eligible": True}, 100)
        result = self.read()
        self.assertIsNone(result["steps"])
        self.assertIsNone(result["model_calls"])
        self.assertIsNone(result["control_without_video_s"])
        self.assertIs(result["raw_libero_success"], False)


if __name__ == "__main__":
    unittest.main()
