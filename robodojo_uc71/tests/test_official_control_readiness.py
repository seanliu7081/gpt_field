"""Offline fabricated evidence tests; these never qualify native physical runs.

The installation verifier is explicitly mocked. No GPU, simulator, cuRobo,
native motion, package installation or model inference is performed here.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from robodojo_uc71.approval import ApprovalError, file_sha256
from robodojo_uc71.native_runtime_bundle import RuntimeBundleError, VerifiedRuntimeBundle
from robodojo_uc71.official_control_readiness import CHECKS, verify_official_readiness
import robodojo_uc71.official_control_setup as official_setup_module
from robodojo_uc71.schema import ExecutionLimits
from test_fragment_executor import monitoring
from test_native_runtime_bundle import bundle_fixture
from test_reset_binding import reset


class SyntheticEvidence:
    def __init__(self, root, monkeypatch, *, geometry=None, reset_reference=None, parameters=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.setup_value = {"synthetic_only": True, "overlay_target": str(root / "fake_overlay")}
        self.setup = self.save("mock_setup", self.setup_value)
        self.geometry = geometry or self.save("geometry", {"synthetic_only": True})
        self.reset_reference = reset_reference or self.save("reset", {"synthetic_only": True})
        self.parameters = parameters or {"monitoring": monitoring().checked(),
                                        "execution_limits": asdict(ExecutionLimits())}
        self.setup_calls = []

        def mock_verify_setup(path):
            self.setup_calls.append(path)
            assert str(path) == self.setup["path"]
            return deepcopy(self.setup_value)

        monkeypatch.setattr(official_setup_module, "verify_setup_record", mock_verify_setup)
        self.receipts = [self.save(f"receipt_{index}", {"schema": "robodojo-native-action-receipt-1",
            "control_backend": "official_ee", "ik_status": {"left": "Success", "right": "Success"},
            "native_action_index": index, "native_sim_step_calls": 10,
            "native_terminal_callback_calls": 1, "native_reward_callback_calls": 1,
            "physics_dt_s": .004, "physics_time_before_s": .04*(index-1),
            "physics_time_after_s": .04*index}) for index in (1, 2)]
        self.proof = {"schema": "robodojo-official-control-qualification-1",
            "synthetic_only": True, "control_backend": "official_ee", "passed": True,
            "scope": "current_host_native_ee_motion_and_gripper_diagnostic",
            "model_calls": 0, "full_test_authorized": False,
            "setup": deepcopy(self.setup), "geometry": deepcopy(self.geometry),
            "reset_reference": deepcopy(self.reset_reference),
            "monitoring": deepcopy(self.parameters["monitoring"]),
            "execution_limits": deepcopy(self.parameters["execution_limits"]),
            "checks": {name: True for name in CHECKS}, "native_receipts": self.receipts,
            "native_actions": len(self.receipts),
            "source_files": [self.setup, self.geometry, self.reset_reference, *self.receipts]}
        self.binding = {"setup": self.setup}
        self.write_proof()

    def save(self, name, value):
        path = self.root / (name + ".json")
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": file_sha256(path)}

    def write_proof(self):
        self.binding["qualification"] = self.save("qualification", self.proof)

    def verify(self):
        return verify_official_readiness(self.binding, geometry=self.geometry,
            reset_reference=self.reset_reference, parameters=self.parameters)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    return SyntheticEvidence(tmp_path / "fabricated_official_evidence", monkeypatch)


def test_offline_synthetic_evidence_schema_accepts_matching_bound_inputs(evidence):
    assert evidence.verify() == evidence.setup_value
    assert evidence.setup_calls == [evidence.setup["path"]]


@pytest.mark.parametrize("installation_only", ["missing_qualification", "installation_as_qualification"])
def test_installation_record_alone_cannot_qualify_native_motion(evidence, installation_only):
    if installation_only == "missing_qualification":
        evidence.binding.pop("qualification")
    else:
        evidence.binding["qualification"] = evidence.setup
    with pytest.raises(ValueError, match="qualification_required"):
        evidence.verify()


@pytest.mark.parametrize("field", ["setup", "geometry", "reset_reference", "monitoring", "execution_limits"])
def test_qualification_must_match_current_inputs_even_if_passed(evidence, field):
    if field in ("setup", "geometry", "reset_reference"):
        evidence.proof[field]["sha256"] = "a" * 64
    else:
        evidence.proof[field][next(iter(evidence.proof[field]))] = "stale"
    evidence.write_proof()
    with pytest.raises(ValueError, match="fresh_measured_official_control"):
        evidence.verify()


@pytest.mark.parametrize("mutation", ["legacy_backend", "missing_ik", "private_ik_status", "wrong_count", "wrong_time"])
def test_pass_flag_cannot_hide_legacy_or_invalid_native_receipts(evidence, mutation):
    row = evidence.receipts[0]
    receipt = json.loads(Path(row["path"]).read_text())
    if mutation == "legacy_backend":
        receipt["control_backend"] = "legacy_joint"
    elif mutation == "missing_ik":
        receipt.pop("ik_status")
    elif mutation == "private_ik_status":
        receipt["ik_status"]["left"] = "collision_with_hidden_object"
    elif mutation == "wrong_count":
        receipt["native_reward_callback_calls"] = 0
    else:
        receipt["physics_time_after_s"] = .4
    Path(row["path"]).write_text(json.dumps(receipt))
    row["sha256"] = file_sha256(row["path"])
    evidence.write_proof()
    with pytest.raises(ValueError, match="native_action_evidence_invalid"):
        evidence.verify()


@pytest.mark.parametrize("mutation", ["unordered_receipts", "missing_source", "missing_check", "false_check"])
def test_complete_sequence_and_source_closure_are_required(evidence, mutation):
    if mutation == "unordered_receipts":
        evidence.proof["native_receipts"] = list(reversed(evidence.receipts))
    elif mutation == "missing_source":
        evidence.proof["source_files"] = evidence.proof["source_files"][:-1]
    elif mutation == "missing_check":
        evidence.proof["checks"].pop("inactive_cartesian_hold")
    else:
        evidence.proof["checks"]["current_host_reset"] = False
    evidence.write_proof()
    with pytest.raises(ValueError):
        evidence.verify()


def test_bound_receipt_mutation_is_rejected_after_previous_verification(evidence):
    evidence.verify()
    Path(evidence.receipts[0]["path"]).write_text('{"synthetic_only": "changed"}')
    with pytest.raises(ApprovalError):
        evidence.verify()


def test_legacy_bundle_cannot_be_relabelled_official_without_new_evidence(bundle_fixture):
    path, document, native_root = bundle_fixture
    document["control_backend"] = "official_ee"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="official_setup_and_native_qualification_required"):
        VerifiedRuntimeBundle.load(path, native_root)


def test_legacy_bundle_cannot_claim_official_qualification_or_activate_it(bundle_fixture, monkeypatch):
    path, document, native_root = bundle_fixture
    bundle = VerifiedRuntimeBundle.load(path, native_root)
    assert bundle.control_backend == "legacy_joint"
    monkeypatch.setattr(official_setup_module, "activate_setup_record",
                        lambda *args: pytest.fail("Legacy bundle must not activate an official overlay"))
    with pytest.raises(RuntimeBundleError, match="official_backend_required"):
        bundle.activate_official_dependencies(native_root / "synthetic_private_cache")
    document["official_control"] = {"synthetic_only": True}
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeBundleError, match="legacy_bundle_cannot_claim"):
        VerifiedRuntimeBundle.load(path, native_root)


def test_offline_official_bundle_binds_capabilities_overlay_and_changed_evidence(bundle_fixture, monkeypatch):
    path, document, native_root = bundle_fixture
    evidence = SyntheticEvidence(path.parent / "fabricated_official_evidence", monkeypatch,
        geometry=document["evidence"]["geometry"], reset_reference=document["evidence"]["reset_reference"],
        parameters=document["parameters"])
    document.update(control_backend="official_ee", official_control=evidence.binding)
    source_root = Path(__file__).resolve().parents[1] / "src/robodojo_uc71"
    for name in ("official_control.py", "official_control_setup.py", "official_control_readiness.py",
                 "native_config.py", "native_episode.py", "native_worker_protocol.py", "native_worker_client.py",
                 "native_episode_runner.py", "client.py"):
        source = source_root / name
        document["program_files"].append({"path": str(source), "sha256": file_sha256(source)})
    path.write_text(json.dumps(document))
    bundle = VerifiedRuntimeBundle.load(path, native_root)
    assert bundle.control_backend == "official_ee"
    caps = bundle.controller_capabilities
    assert caps["version"] == "robodojo-uc71-official-ee-controller-1"
    assert not {"action_translation_m", "action_rotation_rad", "max_joint_step_rad", "joint_target_lead_rad"} & caps.keys()
    calls = []
    private_cache = native_root / "synthetic_private_cache"
    def mock_activate(setup_path, cache_root):
        calls.append((setup_path, cache_root))
        return deepcopy(evidence.setup_value)
    monkeypatch.setattr(official_setup_module, "activate_setup_record", mock_activate)
    assert bundle.activate_official_dependencies(private_cache) == evidence.setup_value["overlay_target"]
    assert calls == [(evidence.setup["path"], private_cache)]
    assert not private_cache.exists()  # The mocked helper does no filesystem setup.
    Path(evidence.receipts[0]["path"]).write_text('{}')
    with pytest.raises(ApprovalError):
        bundle.verify()


def test_activation_helper_rejects_stale_dependency_before_cache_creation(evidence, monkeypatch):
    Path(evidence.setup_value["overlay_target"]).mkdir()
    stale = ModuleType("curobo")
    stale.__file__ = str(evidence.root / "unverified_curobo/__init__.py")
    monkeypatch.setitem(sys.modules, "curobo", stale)
    private_cache = evidence.root / "private_cache"
    with pytest.raises(ValueError, match="already_loaded_from_other_source"):
        official_setup_module.activate_setup_record(evidence.setup["path"], private_cache)
    assert not private_cache.exists()
