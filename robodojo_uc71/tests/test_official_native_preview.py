"""Offline official-preview preparation and evidence tests; no simulator starts."""
import argparse
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_native_preview as child
import provision_native_preview as supervisor
from robodojo_uc71 import official_control_setup as setup
import test_native_preview_supervisor as preview_evidence_tests


@pytest.fixture
def setup_record(tmp_path):
    native, overlay = tmp_path / "native", tmp_path / "overlay"
    names = ("env/planner_manager/curobo_planner.py", "env/robot_manager/robot_manager.py",
             "Assets/Robots/x5/robot_config.yml", "Assets/Robots/x5/X5A.urdf", "Assets/Robots/x5/curobo.yml")
    source = []
    for name in names:
        path = native / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture: " + name)
        source.append({"path": str(path), "sha256": child.digest(path)})
    module = overlay / "curobo" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("# No imports or execution in this fixture.\n")
    record = {"schema": setup.SCHEMA, "verified": True, "overlay_target": str(overlay),
        "python": sys.executable, "native_commit": child.pins()["native_repository"]["commit"],
        "curobo_commit": next(row["commit"] for row in child.pins()["gitlinks"] if row["path"] == "third_party/curobo"),
        "source_files": source, "installed_files": [{"path": str(module), "sha256": child.digest(module)}],
        "base_requirements": []}
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(record))
    return native, overlay, path, record


def test_official_setup_binds_native_and_dependency_closure(setup_record):
    native, overlay, path, record = setup_record
    binding = child.verify_preview_control_setup("official_ee", path, native=native, python=Path(sys.executable))
    assert binding["path"] == str(path)
    assert binding["sha256"] == child.digest(path)
    assert binding["overlay_target"] == str(overlay)
    assert binding["native_commit"] == record["native_commit"]
    (overlay / "untracked.py").write_text("# changed closure\n")
    with pytest.raises(ValueError, match="overlay_inventory_changed"):
        child.verify_preview_control_setup("official_ee", path, native=native, python=Path(sys.executable))


@pytest.mark.parametrize("change", ["python", "native_commit", "source"])
def test_official_setup_rejects_mismatched_runtime(setup_record, change):
    native, _overlay, path, record = setup_record
    if change == "source":
        (native / "Assets/Robots/x5/curobo.yml").write_text("changed")
    else:
        record[change] = "/missing/interpreter" if change == "python" else "0" * 40
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        child.verify_preview_control_setup("official_ee", path, native=native, python=Path(sys.executable))


@pytest.mark.parametrize("backend,path", [("official_ee", None), ("legacy_joint", Path("setup.json"))])
def test_cli_requires_setup_exactly_for_official_backend(backend, path):
    with pytest.raises(SystemExit):
        child.preview_control_arguments(argparse.ArgumentParser(),
            argparse.Namespace(control_backend=backend, official_setup=path))


def test_legacy_preparation_does_not_activate_official_setup(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Legacy preview must not verify or activate official dependencies")
    monkeypatch.setattr(setup, "verify_setup_record", forbidden)
    assert child.verify_preview_control_setup("legacy_joint", None, native=tmp_path, python=Path(sys.executable)) is None


def test_preparation_activates_verified_setup_before_native_imports(setup_record, tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    native, _overlay, path, _record = setup_record
    for name in ("source.json", "assets.json", "auth.json"):
        (tmp_path / name).write_text("{}")
    client_path = native / "XPolicyLab/client_server/ws/model_client.py"
    client_path.parent.mkdir(parents=True)
    client_path.write_text("# Unexecuted native source fixture\n")
    output = tmp_path / "preview"
    events = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ROBODOJO_RUN_ID", "original")
    monkeypatch.setattr(child, "check_authorization", lambda *_args, **_kwargs: {"full_test_authorized": False})
    monkeypatch.setattr(child, "verify_manifests", lambda *_args, **_kwargs: {"camera_gate": {"gpu_uuid": "GPU-test"}})
    def activate(record, cache):
        assert record == str(path)
        assert cache == output / "official_control_cache"
        assert child.verify_preview_control_setup("official_ee", path, native=native, python=Path(sys.executable))
        events.append("activate_verified_setup")
        return _record
    monkeypatch.setattr(setup, "activate_setup_record", activate)
    monkeypatch.setattr(child, "add_native_import_paths", lambda _root: events.append("native_import_paths") or [])
    def configuration(_native, _run_id, *, control_backend):
        assert events == ["activate_verified_setup", "native_import_paths"]
        assert control_backend == "official_ee"
        events.append("native_configuration")
        return OmegaConf.create({"robot": {"robots": [{"robot_name": "x5", "need_planner": True}] * 2}})
    monkeypatch.setattr(child, "build_native_preview_config", configuration)
    monkeypatch.setattr(child, "run_preview", lambda *_args: pytest.fail("Preparation must not start simulator"))
    monkeypatch.setattr(sys, "argv", ["prepare_native_preview.py", "--native-root", str(native),
        "--source-manifest", str(tmp_path / "source.json"), "--asset-manifest", str(tmp_path / "assets.json"),
        "--authorization", str(tmp_path / "auth.json"), "--output", str(output), "--gpu-uuid", "GPU-test",
        "--pci-bus-id", "0000:01:00.0", "--kit-index", "0", "--cuda-ordinal", "0",
        "--control-backend", "official_ee", "--official-setup", str(path)])
    assert child.main() == 0
    preparation = json.loads((output / "preparation.json").read_text())
    assert preparation["official_setup"]["sha256"] == child.digest(path)
    assert preparation["control_backend"] == "official_ee"
    assert preparation["native_motion_qualified"] is False
    assert set(preparation["replaced_native_module_sha256"]) == {"XPolicyLab/client_server/ws/model_client.py"}
    assert preparation["launch_requested"] is False
    assert events[-1] == "native_configuration"


@pytest.fixture
def evidence():
    case = preview_evidence_tests.NativePreviewEvidenceTests()
    case.setUp()
    try:
        overlay = case.path / "verified_overlay"
        for name in ("curobo", "warp"):
            module = overlay / name / "__init__.py"
            module.parent.mkdir(parents=True)
            module.write_text("# Synthetic loaded-module provenance fixture\n")
        case.official_setup = {"path": str(case.path / "setup.json"), "sha256": "setup-sha", "overlay_target": str(overlay)}
        for item in (case.preparation, case.result, case.ready):
            item.update(control_backend="official_ee", official_setup=copy.deepcopy(case.official_setup), native_motion_qualified=False)
        case.result.update(native_planner_instances=1, native_ik_solver_instances=1)
        case.helpers = {str(path): "helper-sha" for path in child.preview_source_files("official_ee")}
        case.preparation["source_file_sha256"] = dict(case.helpers)
        case.put("loaded_module_provenance.json", {name: {"file": str(overlay / name / "__init__.py")}
                                                   for name in ("curobo", "warp")})
        case.result["loaded_module_provenance_sha256"] = child.digest(case.path / "loaded_module_provenance.json")
        case.put("native_configuration.json", {"robot": {"robots": [{"robot_name": "x5", "need_planner": True}] * 2}})
        case.preparation["native_configuration_sha256"] = child.digest(case.path / "native_configuration.json")
        yield case
    finally:
        case.doCleanups()


def verify_official_evidence(case):
    case.put("preparation.json", case.preparation)
    case.result["preparation_sha256"] = child.digest(case.path / "preparation.json")
    case.put("preview_result.json", case.result)
    case.ready.update(preparation_sha256=case.result["preparation_sha256"],
                      preview_result_sha256=child.digest(case.path / "preview_result.json"))
    case.put("preview_preclose_ready.json", case.ready)
    return supervisor.verify_preview_evidence(case.path, case.binding, case.identity, case.expected,
        case.helpers, case.log, Path(sys.executable), case.authorization,
        control_backend="official_ee", official_setup_expected=case.official_setup)


def test_official_preview_evidence_verifies_render_without_claiming_motion(evidence):
    result = verify_official_evidence(evidence)
    assert result["native_render_verified"] is True
    assert result["native_motion_qualified"] is False
    assert result["official_setup"] == evidence.official_setup


@pytest.mark.parametrize("record,key,value", [
    ("preparation", "control_backend", "legacy_joint"),
    ("result", "official_setup", None),
    ("ready", "native_motion_qualified", True),
    ("result", "native_planner_instances", 0),
    ("result", "native_ik_solver_instances", 0),
    ("result", "benchmark_actions", 1),
    ("result", "model_calls", 1),
])
def test_official_preview_rejects_scope_or_backend_changes(evidence, record, key, value):
    getattr(evidence, record)[key] = value
    with pytest.raises(ValueError):
        verify_official_evidence(evidence)


def test_official_preview_rejects_bundled_dependency(evidence):
    evidence.put("loaded_module_provenance.json", {name: {"file": "/unverified/" + name + "/__init__.py"}
                                                  for name in ("curobo", "warp")})
    evidence.result["loaded_module_provenance_sha256"] = child.digest(evidence.path / "loaded_module_provenance.json")
    with pytest.raises(ValueError, match="unbound native control dependency"):
        verify_official_evidence(evidence)


def test_official_preview_rejects_disabled_planner_configuration(evidence):
    evidence.put("native_configuration.json", {"robot": {"robots": [{"robot_name": "x5", "need_planner": False}] * 2}})
    evidence.preparation["native_configuration_sha256"] = child.digest(evidence.path / "native_configuration.json")
    with pytest.raises(ValueError, match="enable both native X5 planners"):
        verify_official_evidence(evidence)
