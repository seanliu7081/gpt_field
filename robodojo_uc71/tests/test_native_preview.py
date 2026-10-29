"""Offline regression checks for preview scope; never construct Isaac Sim."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))
from robodojo_uc71.native_config import (CLIENT_MODULE, PLANNER_MODULE, DisabledPreviewPlanner,
    ResetOnlyClient, forbid_policy_actions, install_preview_adapters)
from prepare_native_preview import (REVIEWED_INCOMPLETE_ASSET_MANIFEST_SHA256, check_authorization,
    save_rgb_images, verify_asset_diagnostic_assessment, verify_manifests)


class NativePreviewGuards(unittest.TestCase):
    def test_reset_only_transport_accepts_one_reset_and_no_observation(self):
        with mock.patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            client = ResetOnlyClient(url="ws://127.0.0.1:1")
            for kwargs in ({"func_name": "get_action"}, {"func_name": "reset", "obs": {}},
                           {"func_name": "reset", "env_idx_list": [0]}):
                with self.assertRaises(RuntimeError):
                    client.call(**kwargs)
            self.assertIsNone(client.call(func_name="reset"))
            self.assertEqual(client.reset_calls, 1)
            with self.assertRaises(RuntimeError):
                client.call(func_name="reset")

    def test_adapters_prevent_eager_imports_and_planner_instantiation(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop(CLIENT_MODULE, None)
            sys.modules.pop(PLANNER_MODULE, None)
            before = set(sys.modules)
            adaptations = install_preview_adapters()
            client_module = importlib.import_module(CLIENT_MODULE)
            planner_module = importlib.import_module(PLANNER_MODULE)
            self.assertIs(client_module.WsModelClient, ResetOnlyClient)
            namespace = {}
            exec("from client_server.ws.model_client import WsModelClient", namespace)
            self.assertIs(namespace["WsModelClient"], ResetOnlyClient)
            self.assertIs(planner_module.CuroboPlanner, DisabledPreviewPlanner)
            with self.assertRaises(RuntimeError):
                planner_module.CuroboPlanner()
            self.assertEqual(set(adaptations), {CLIENT_MODULE, PLANNER_MODULE})
            self.assertFalse(any(name.startswith("curobo") for name in set(sys.modules) - before))
            with self.assertRaises(RuntimeError):
                install_preview_adapters()

    def test_action_guards_leave_reset_and_reward_registration_callable(self):
        env = SimpleNamespace(reset=lambda: "reset", run_reward=lambda: "register",
                              run_eval=lambda: "bad", step=lambda: "bad", take_action=lambda: "bad")
        guarded = forbid_policy_actions(env)
        self.assertEqual(set(guarded), {"run_eval", "step", "take_action"})
        self.assertEqual(env.reset(), "reset")
        self.assertEqual(env.run_reward(), "register")
        for name in guarded:
            with self.assertRaises(RuntimeError):
                getattr(env, name)()

    def test_launch_requires_parent_scope_even_with_setup_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "auth.json"
            auth = {"native_preview_authorized": True, "full_test_authorized": False,
                    "model_calls_authorized": False, "gpu_uuid": "GPU-test", "user_message": "setup"}
            path.write_text(json.dumps(auth))
            binding = {"gpu_uuid": "GPU-test"}
            with mock.patch.dict("os.environ", {}, clear=True):
                check_authorization(path, binding, launch=False)
                with self.assertRaises(ValueError):
                    check_authorization(path, binding, launch=True)
            auth["full_test_authorized"] = True
            path.write_text(json.dumps(auth))
            with self.assertRaises(ValueError):
                check_authorization(path, binding, launch=False)

    def test_only_three_actual_rgb_frames_are_saved(self):
        import numpy as np
        names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, 320:] = 255
        observation = {"vision": {name: {"color": image} for name in names}}
        with tempfile.TemporaryDirectory() as temp:
            images = save_rgb_images(observation, Path(temp))
            self.assertEqual({row["camera"] for row in images}, set(names))
            self.assertTrue(all((Path(temp) / row["path"]).is_file() for row in images))
            observation["vision"].pop("cam_head")
            with self.assertRaises(RuntimeError):
                save_rgb_images(observation, Path(temp))
            observation["vision"]["cam_head"] = {"color": image[:1]}
            with self.assertRaises(RuntimeError):
                save_rgb_images(observation, Path(temp))

    def test_diagnostic_assessment_rejects_any_other_raw_asset_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "asset_manifest.json"
            raw.write_text('{"complete":false}')
            with self.assertRaisesRegex(ValueError, "exact reviewed native_assets_002"):
                verify_asset_diagnostic_assessment(root, raw, {"complete": False}, root / "missing_assessment.json")

    def test_diagnostic_assessment_cannot_claim_standard_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, assessment = root / "asset.json", root / "assessment.json"
            raw.write_text("{}")
            assessment.write_text(json.dumps({"schema": "robodojo-native-asset-diagnostic-assessment-1",
                "diagnostic_authorized": True, "standard_ready": True,
                "asset_fidelity_passed": False, "dependency_closure_complete": False,
                "model_calls_authorized": False, "full_test_authorized": False}))
            with mock.patch("prepare_native_preview.digest", return_value=REVIEWED_INCOMPLETE_ASSET_MANIFEST_SHA256):
                with self.assertRaisesRegex(ValueError, "incomplete-fidelity diagnostic"):
                    verify_asset_diagnostic_assessment(root, raw, {}, assessment)

    def test_incomplete_source_manifest_fails_before_checkout_or_gpu_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, assets = root / "source.json", root / "assets.json"
            source.write_text(json.dumps({"schema": "robodojo-native-source-1", "complete": False}))
            assets.write_text("{}")
            with mock.patch("prepare_native_preview.require_native_checkout", side_effect=AssertionError):
                with self.assertRaises(ValueError):
                    verify_manifests(root, source, assets, {})


if __name__ == "__main__":
    unittest.main()
