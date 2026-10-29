"""Offline setup boundary tests; no network, simulator or GPU contexts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from provision_native_common import asset_path, digest, require_camera_gate
from provision_native_assets import (allowed_file, candidate_path, download, seed_roots,
    text_references, verify_file, inspect_references, material_thumbnail_references)


class SetupBoundaries(unittest.TestCase):
    def test_reject_asset_escape_and_non_assets(self):
        for name in ("../secret", "/Assets/x.usd", "Assets/../secret", "https://example/x.usd",
                     "Assets\\x.usd", "Training/demo.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                asset_path(name)

    def test_dependency_anchor_and_placeholder(self):
        self.assertEqual(candidate_path("Assets/Object/a/object.usd", "../tex/a.png"), "Assets/Object/tex/a.png")
        self.assertEqual(candidate_path("Assets/Robots/x5/robot_config.yml", "$RoboDojo_ASSETS/Sensor/camera.usd"),
                         "Assets/Sensor/camera.usd")
        self.assertIsNone(candidate_path("Assets/a.usd", "../../outside.bin"))
        self.assertIsNone(candidate_path("Assets/a.usd", "/old/Assets/x.usd"))
        self.assertIsNone(candidate_path("Assets/a.usd", "https://example/x.usd"))

    def test_exclude_training_models_and_recordings(self):
        for name in ("Assets/Robots/x5/policy.pt", "Assets/training/a.png", "Assets/checkpoints/a.bin",
                     "Assets/Object/demo.mp4", "Assets/Object/traj.pkl"):
            self.assertFalse(allowed_file(name))
        self.assertTrue(allowed_file("Assets/Object/a/00000/object.usdz"))

    def test_hash_checks_lfs_and_git(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.json"
            path.write_bytes(b"{}")
            entry = {"path": "Assets/a.json", "size": 2,
                     "oid": hashlib.sha1(b"blob 2\0{}").hexdigest()}
            self.assertEqual(verify_file(path, entry)["bytes"], 2)
            entry["lfs"] = {"size": 2, "oid": digest(path)}
            self.assertEqual(verify_file(path, entry)["sha256"], digest(path))
            entry["lfs"]["oid"] = "0" * 64
            with self.assertRaises(ValueError):
                verify_file(path, entry)

    def test_byte_bound_prevents_network(self):
        with tempfile.TemporaryDirectory() as temp:
            entry = {"path": "Assets/a.png", "size": 3, "oid": "a" * 40}
            with patch("urllib.request.urlopen") as network:
                with self.assertRaises(ValueError):
                    download(None, Path(temp), entry, remaining_bytes=2)
                network.assert_not_called()

    def test_camera_frame_with_failed_process_is_not_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = directory / "launch_result.json"
            review = directory / "review.json"
            result.write_text(json.dumps({"schema": "robodojo-new-host-camera-1",
                "camera_render_passed": True, "shutdown_contract_satisfied": True,
                "diagnostic_passed": False, "returncode": -6}))
            review.write_text(json.dumps({"passed": True}))
            with patch("subprocess.check_output") as command:
                with self.assertRaises(ValueError):
                    require_camera_gate(result, review)
                command.assert_not_called()

    def test_visual_review_is_bound_and_gpu_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = directory / "launch_result.json"
            review = directory / "review.json"
            image = directory / "rgb.png"
            image.write_bytes(b"fixture")
            data = {"schema": "robodojo-new-host-camera-1", "returncode": 0,
                    "remaining_owned_pids": [], "termination_signals": [],
                    "shutdown_mode": "stock-fast", "preclose_ready_verified": True,
                    "full_teardown_returned": False, "binding": {"gpu_uuid": "GPU-test"}}
            data.update({key: True for key in ("diagnostic_passed", "camera_render_passed",
                "shutdown_contract_satisfied", "renderer_identity_verified", "cuda_identity_verified",
                "gpu_processes_released", "gpu_memory_released")})
            result.write_text(json.dumps(data))
            review.write_text(json.dumps({"passed": True, "image_sha256": digest(image),
                                          "launch_result_sha256": digest(result)}))
            with patch("subprocess.check_output", return_value="GPU-other\n"):
                with self.assertRaises(ValueError):
                    require_camera_gate(result, review)
            with patch("subprocess.check_output", return_value="GPU-test\n"):
                self.assertFalse(require_camera_gate(result, review)["full_test_authorized"])
            image.write_bytes(b"changed")
            with patch("subprocess.check_output") as command:
                with self.assertRaises(ValueError):
                    require_camera_gate(result, review)
                command.assert_not_called()

    def test_metadata_resource_strings(self):
        refs = text_references('texture_2d("./textures/color.png"); mesh: meshes/link.stl\nname: robot\n')
        self.assertEqual(refs, ["./textures/color.png", "meshes/link.stl"])

    def test_yaml_comments_do_not_create_missing_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "robot.yml"
            path.write_text('urdf: "./robot.urdf"\n# srdf: "./absent.srdf"\n')
            self.assertEqual(inspect_references(path, Path("unused")), [("", "./robot.urdf")])

    def test_mdl_thumbnail_metadata_does_not_hide_shader_texture(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "material.mdl"
            path.write_text('::anno::thumbnail("./preview.png"); texture_2d("./color.png");')
            self.assertEqual(material_thumbnail_references(path), {"./preview.png"})
            path.write_text('::anno::thumbnail("./preview.png"); texture_2d("./preview.png");')
            self.assertEqual(material_thumbnail_references(path), set())

    def test_native_asset_root_interpolation(self):
        self.assertEqual(candidate_path("Assets/Robots/x5/curobo_tmp.yml",
            "${ASSETS_PATH}/Assets/Robots/x5/X5A.urdf"), "Assets/Robots/x5/X5A.urdf")


if __name__ == "__main__":
    unittest.main()
