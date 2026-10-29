"""Offline rejection tests: no real passing bundle or review is produced."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_runtime_bundle.py"
spec = importlib.util.spec_from_file_location("prepare_runtime_bundle_test_module", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def launch_record():
    return {"schema": "robodojo-controller-characterization-launch-1",
        **{name: True for name in ("diagnostic_completed", "diagnostic_actions_verified", "native_render_verified",
            "preclose_ready_verified", "worker_isolation_verified", "operator_clearances_verified",
            "cuda_identity_verified", "renderer_identity_verified", "extension_startup_clean",
            "gpu_processes_released", "gpu_memory_released", "shutdown_contract_satisfied")},
        "returncode": 0, "termination_signals": [], "remaining_owned_pids": [],
        "automatic_retry": False, "full_test_authorized": False, "benchmark_attempt_started": False,
        "standard_ready": False, "model_calls": 0, "benchmark_actions": 0,
        "diagnostic_native_actions": 960, "diagnostic_physics_substeps": 9600,
        "diagnostic_native_actions_planned": 960,
        "plan_file_sha256": module.CONTROLLER_PLAN_FILE_SHA256,
        "plan_sha256": module.CONTROLLER_PLAN_CONTENT_SHA256}


@pytest.mark.parametrize("changes", [
    {"diagnostic_completed": False, "diagnostic_native_actions": 20},
    {"diagnostic_native_actions": 959}, {"diagnostic_physics_substeps": 9599},
    {"diagnostic_native_actions": 1040, "diagnostic_physics_substeps": 10400,
        "diagnostic_native_actions_planned": 1040},
    {"plan_file_sha256": "8620736e67aa630d8d5b0bf9c260197ecc418856ddeb9c076843be93c565beed"},
    {"plan_sha256": "a76fbc11c220c249a76c6297af7acd1b2e5423c8b69aa5d499440bbfe2550fdc"},
    {"plan_file_sha256": "fa4fd6d3224610adb20ce277c8b5848de25558b1d5b22b2bb2f22c2dfd7048d6"},
    {"plan_sha256": "14e59539c58fd4350f37300d57ece3df664700e8fc28a6814347d58be5767cf8"},
    {"plan_file_sha256": "a75d980c739b87096a1ae1d9278fe88b910ee07bf1cfbdb8d98e90716329a0e6"},
    {"plan_sha256": "0ddc7fd440d6d39aea6f5b0f2a1b0cc4036fbad5d04fd06e68ef633aac2b79ea"},
    {"plan_file_sha256": "1794f9a11ae6e6a24257d36456bd47ea1dcb545a223b4a6205a6f35658993744"},
    {"plan_sha256": "dfa361c4cceeb5bd6eced1b1dc310dcae010896204439e53e22273b662bda11f"},
    {"schema": "robodojo-open-clearance-preview-launch-1", "clearance_preview_completed": True,
        "diagnostic_native_actions": 260, "diagnostic_physics_substeps": 2600,
        "diagnostic_native_actions_planned": 260},
    {"operator_clearances_verified": False}, {"shutdown_contract_satisfied": False},
    {"termination_signals": [15]}, {"gpu_processes_released": False},
    {"full_test_authorized": True}, {"model_calls": 1}, {"returncode": False}])
def test_incomplete_or_out_of_scope_controller_cannot_create_candidate(tmp_path, monkeypatch, changes):
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({**launch_record(), **changes}))
    output = tmp_path / "unissued_bundle"
    monkeypatch.setattr(module.os, "getuid", lambda: 0)
    with pytest.raises(module.AssemblyError, match="completed_reviewed_controller"):
        module.prepare({"controller_launch": launch}, native_root=tmp_path,
            python=Path("/missing/python"), output=output, assemble=True)
    assert not output.exists()
    assert not list(tmp_path.glob(".runtime-bundle-validation-*"))


def test_new_completed_plan_passes_initial_gate_without_issuing_a_bundle(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch_record()))
    assert module.completed_controller(path)["diagnostic_native_actions"] == 960
    assert list(tmp_path.iterdir()) == [path]


def test_duplicate_record_keys_cannot_hide_failed_diagnostic(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text('{"diagnostic_completed":false,"diagnostic_completed":true}')
    with pytest.raises(module.AssemblyError, match="duplicate_json"):
        module.completed_controller(path)


def test_assembler_never_fills_unselected_parameter_defaults():
    with pytest.raises(module.AssemblyError, match="exact_selected_parameter_groups"):
        module.checked_parameters({"controller": {}})
