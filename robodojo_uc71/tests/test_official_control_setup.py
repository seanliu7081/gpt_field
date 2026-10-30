import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from robodojo_uc71.official_control_setup import (
    SCHEMA, activate_setup_record, digest, verify_setup_record,
)


@pytest.fixture
def setup_record(tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    module = overlay / "controller.py"
    module.write_text("value = 1\n")
    source = tmp_path / "source.py"
    source.write_text("native = True\n")
    record = {"schema": SCHEMA, "verified": True, "native_commit": "a" * 40,
              "curobo_commit": "b" * 40, "overlay_target": str(overlay),
              "source_files": [{"path": str(source), "sha256": digest(source)}],
              "installed_files": [{"path": str(module), "sha256": digest(module)}],
              "base_requirements": []}
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(record))
    return path, module, source


def test_exact_setup_passes_without_loading_package(setup_record):
    path, module, _ = setup_record
    assert verify_setup_record(path)["overlay_target"] == str(module.parent)


@pytest.mark.parametrize("index", [1, 2])
def test_modified_runtime_or_native_source_rejected(setup_record, index):
    setup_record[index].write_text("changed = True\n")
    with pytest.raises(ValueError, match="file_changed"):
        verify_setup_record(setup_record[0])


def test_unrecorded_importable_file_rejected(setup_record):
    path, module, _ = setup_record
    (module.parent / "injected.py").write_text("pass\n")
    with pytest.raises(ValueError, match="inventory_changed"):
        verify_setup_record(path)


def test_overlay_entry_cannot_be_redirected_by_symlink(setup_record, tmp_path):
    path, module, _ = setup_record
    outside = tmp_path / "outside.py"
    outside.write_bytes(module.read_bytes())
    module.unlink()
    module.symlink_to(outside)
    with pytest.raises(ValueError, match="file_changed"):
        verify_setup_record(path)


def test_changed_base_package_rejected(setup_record):
    path, _, _ = setup_record
    record = json.loads(path.read_text())
    record["base_requirements"] = [{"name": "numpy", "version": "0.0.0"}]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="base_runtime_changed"):
        verify_setup_record(path)


def test_cached_verification_rejects_same_size_write_with_restored_mtime(setup_record):
    path, module, _ = setup_record
    verify_setup_record(path)
    before = module.stat()
    module.write_text("value = 2\n")
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="file_changed"):
        verify_setup_record(path)


@pytest.mark.parametrize("filename", ["new.py", "controller.pyc", "__pycache__/controller.cpython-311.pyc"])
def test_cached_verification_rejects_added_import_code(setup_record, filename):
    path, module, _ = setup_record
    verify_setup_record(path)
    added = module.parent / filename
    added.parent.mkdir(exist_ok=True)
    added.write_bytes(b"unbound code")
    with pytest.raises(ValueError, match="inventory_changed"):
        verify_setup_record(path)


def test_cached_verification_rejects_directory_symlink(setup_record):
    path, module, _ = setup_record
    verify_setup_record(path)
    original = module.parent
    moved = original.with_name("moved")
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="overlay_invalid"):
        verify_setup_record(path)


def test_activation_redirects_real_runtime_api_and_all_caches(setup_record, tmp_path, monkeypatch):
    path, module, _ = setup_record
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    imported = {}
    def load(name):
        imported[name] = SimpleNamespace(__file__=str(module), cache_dir="old")
        return imported[name]
    monkeypatch.setattr("robodojo_uc71.official_control_setup.importlib.import_module", load)
    cache = tmp_path / "cache"
    activate_setup_record(path, cache)
    assert set(imported) == {"curobo.runtime", "curobo._src.runtime"}
    assert all(item.cache_dir == str(cache / "curobo") for item in imported.values())
    assert sys.path[0] == str(module.parent)
    assert sys.dont_write_bytecode
    for variable, dirname in (("WARP_CACHE_PATH", "warp"), ("CUDA_CACHE_PATH", "cuda"),
                              ("XDG_CACHE_HOME", "xdg")):
        assert os.environ[variable] == str(cache / dirname)
        assert (cache / dirname).is_dir()


def test_activation_rejects_already_loaded_other_curobo(setup_record, tmp_path, monkeypatch):
    path, _, source = setup_record
    monkeypatch.setitem(sys.modules, "curobo.foreign", SimpleNamespace(__file__=str(source)))
    with pytest.raises(ValueError, match="already_loaded_from_other_source"):
        activate_setup_record(path, tmp_path / "cache")


def test_activation_rejects_import_hook_loading_unbound_module(setup_record, tmp_path, monkeypatch):
    path, _, source = setup_record
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr("robodojo_uc71.official_control_setup.importlib.import_module",
                        lambda _: SimpleNamespace(__file__=str(source)))
    with pytest.raises(ValueError, match="module_loaded_from_other_source"):
        activate_setup_record(path, tmp_path / "cache")
