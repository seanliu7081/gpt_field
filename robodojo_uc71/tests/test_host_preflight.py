"""Offline coverage for host decisions and container memory accounting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_host.py"
SPEC = importlib.util.spec_from_file_location("robodojo_check_host", SCRIPT)
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


class HostPreflightTests(unittest.TestCase):
    def test_driver_comparison_does_not_claim_runtime_compatibility(self):
        result = host.assess_gpu({"driver_version": "580.82.09", "memory_total_mib": 24564,
                                  "memory_free_mib": 24065})
        self.assertTrue(result["driver_at_or_above_5_1_tested_version"])
        self.assertFalse(result["exact_5_1_tested_driver"])
        self.assertTrue(result["currently_free_vram_at_least_16_gib"])
        self.assertIn("unverified", result["runtime_compatibility"])
        result = host.assess_gpu({"driver_version": "575.99.99", "memory_total_mib": 24564,
                                  "memory_free_mib": 8192})
        self.assertFalse(result["driver_at_or_above_5_1_tested_version"])
        self.assertFalse(result["currently_free_vram_at_least_16_gib"])
        self.assertIsNone(host.assess_gpu({"driver_version": "N/A"})["driver_at_or_above_5_1_tested_version"])

    def test_unlimited_and_invalid_cgroup_limits_are_not_finite_capacity(self):
        for value in (None, "max", "9223372036854771712", "N/A", "-1"):
            self.assertIsNone(host.memory_limit(value))
        self.assertEqual(host.memory_limit("0"), 0)

    def test_v1_bind_root_and_hierarchical_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "memory.limit_in_bytes").write_text(str(64 * host.GIB))
            (directory / "memory.usage_in_bytes").write_text(str(3 * host.GIB))
            (directory / "memory.stat").write_text(f"hierarchical_memory_limit {24 * host.GIB}\n")
            result = host.cgroup_memory("11:memory:/docker/this-container\n",
                f"100 99 0:1 /docker/this-container {directory} ro - cgroup cgroup rw,memory\n")
            entry = result["entries"][0]
            self.assertEqual(entry["directory"], str(directory))
            self.assertEqual(entry["effective_limit_bytes"], 24 * host.GIB)
            self.assertEqual(entry["headroom_bytes"], 21 * host.GIB)

    def test_v2_visible_parent_limit_and_shared_usage_bound_available_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            (proc / "self").mkdir(parents=True)
            mount = root / "cgroup"
            child = mount / "child"
            child.mkdir(parents=True)
            (mount / "memory.max").write_text(str(20 * host.GIB))
            (mount / "memory.current").write_text(str(15 * host.GIB))
            (child / "memory.max").write_text("max")
            (child / "memory.current").write_text(str(1 * host.GIB))
            (proc / "meminfo").write_text("MemTotal: 67108864 kB\nMemAvailable: 50331648 kB\n")
            (proc / "self/cgroup").write_text("0::/child\n")
            (proc / "self/mountinfo").write_text(f"100 99 0:1 / {mount} ro - cgroup2 cgroup rw\n")
            result = host.memory_inventory(proc)
            self.assertEqual(result["effective_capacity_upper_bound_bytes"], 20 * host.GIB)
            self.assertEqual(result["effective_available_upper_bound_bytes"], 5 * host.GIB)
            self.assertFalse(result["capacity_at_least_32_gib"])

    def test_private_namespace_root_maps_to_mounted_root(self):
        result = host.cgroup_mounts("0::/\n", "100 99 0:1 /docker/id /sys/fs/cgroup ro - cgroup2 cgroup rw\n")
        self.assertEqual(result[0]["directory"], "/sys/fs/cgroup")

    def test_absent_cgroup_not_reported_as_checked(self):
        self.assertFalse(host.cgroup_memory("", "")["limits_observed"])

    def test_output_cannot_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence.json"
            evidence.write_text("preserved\n")
            with patch.object(host, "collect", return_value={}), patch.object(host.sys, "argv", [str(SCRIPT), "--output", str(evidence)]):
                self.assertEqual(host.main(), 2)
            self.assertEqual(evidence.read_text(), "preserved\n")


if __name__ == "__main__":
    unittest.main()
