"""Current-host paths and hash-bound consolidated Conda installation records.

No simulator, credentials or GPU contexts are opened on import. Historical
base/overlay records keep their existing validators; this is a separate schema.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

BASE = Path(__file__).resolve().parents[2]
CREDENTIAL_STORE = BASE.parent / ".private/robodojo_chatgpt_plan"
CONDA_SCHEMA = "robodojo-conda-runtime-1"
LOCKS = (
    "validation/provisioning/a1_requirements.lock",
    "validation/new_host_20261029/native_dependencies_001/requirements.lock",
    "validation/new_host_20261029/native_compatibility_001/requirements.lock",
    "configs/conda_test_requirements.lock",
)
INVENTORY_CODE = """import importlib.metadata as m,json,sys,sysconfig
print(json.dumps({'python':sys.version,'python_version':list(sys.version_info[:3]),
 'purelib':sysconfig.get_path('purelib'),'packages':sorted(
 [{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda r:r['name'].lower())}))
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(python):
    return json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", INVENTORY_CODE],
        text=True, timeout=30, env={"PATH":str(Path(python).parent)+":/usr/bin:/bin", "PYTHONNOUSERSITE":"1"}))


def package_map(value):
    result = {}
    for row in value["packages"]:
        key = re.sub(r"[-_.]+", "-", row["name"]).lower()
        if key in result:
            raise ValueError("duplicate_distribution_metadata")
        result[key] = row["version"]
    return result


def combined_lock():
    base, count = re.subn(r"^setuptools==[^\n]+(?:\n[ \t]+[^\n]*)*\n?", "",
                          (BASE / LOCKS[0]).read_text(), flags=re.M)
    if count != 1:
        raise ValueError("expected_preserved_setuptools_override")
    return base + "\n" + "\n".join((BASE / path).read_text() for path in LOCKS[1:])


def pinned_versions(lock):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
    result = {}
    for line in lock.replace("\\\n", " ").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line.split(" --hash=", 1)[0].strip())
        if requirement.url:
            version = str(parse_wheel_filename(Path(unquote(urlsplit(requirement.url).path)).name)[1])
        else:
            specifiers = list(requirement.specifier)
            if len(specifiers) != 1 or specifiers[0].operator != "==":
                raise ValueError("exact_requirement_pin_required")
            version = specifiers[0].version
        name = canonicalize_name(requirement.name)
        if name in result and result[name] != version:
            raise ValueError("conflicting_requirement_pins")
        result[name] = version
    return result


def validate_inventory(value, lock):
    if value["python_version"] != [3, 11, 16]:
        raise ValueError("conda_python_3_11_16_required")
    actual = package_map(value)
    for name, expected in pinned_versions(lock).items():
        if actual.get(name) != expected:
            raise ValueError(f"conda_package_pin_mismatch: {name}")
    return actual


def prepare_conda_record(python, output):
    python, output = Path(python).absolute(), Path(output).resolve()
    lock = combined_lock()
    value = inventory(python)
    validate_inventory(value, lock)
    subprocess.run([str(python), "-m", "pip", "check"], check=True, timeout=60,
                   stdout=subprocess.DEVNULL)
    imports = BASE / "runtime/conda_imports"
    imports.mkdir(exist_ok=True)
    if any(imports.iterdir()):
        raise ValueError("conda_import_directory_must_be_empty")
    pth = Path(value["purelib"]) / "robodojo_uc71_source.pth"
    pth_hash = digest(pth)
    output.mkdir(parents=True, exist_ok=False)
    (output / "requirements.lock").write_text(lock)
    (output / "package_inventory.json").write_text(json.dumps(value, indent=2)+"\n")
    record = {
        "schema": CONDA_SCHEMA, "python":str(python), "prefix":str(python.parent.parent),
        "ready_for_diagnostic":True, "lock_sha256":digest(output/"requirements.lock"),
        "inventory_sha256":digest(output/"package_inventory.json"),
        "source_locks":{path:digest(BASE/path) for path in LOCKS},
        "interpreter_sha256":digest(python), "import_path":str(imports),
        "source_pth":{"path":str(pth), "sha256":pth_hash},
        "layout":"consolidated_conda", "model_calls":0, "benchmark_actions":0,
        "full_test_authorized":False,
    }
    path = output / "install_status.json"
    path.write_text(json.dumps(record, indent=2)+"\n")
    verify_conda_record(python, path)
    return path


def verify_conda_record(python, path):
    python, path = Path(python).absolute(), Path(path).resolve(strict=True)
    record = json.loads(path.read_text())
    if (record.get("schema") != CONDA_SCHEMA or record.get("ready_for_diagnostic") is not True
            or record.get("python") != str(python) or record.get("layout") != "consolidated_conda"
            or record.get("source_locks") != {name:digest(BASE/name) for name in LOCKS}
            or digest(python) != record.get("interpreter_sha256")
            or digest(path.with_name("requirements.lock")) != record.get("lock_sha256")
            or digest(path.with_name("package_inventory.json")) != record.get("inventory_sha256")
            or path.with_name("requirements.lock").read_text() != combined_lock()):
        raise ValueError("conda_runtime_record_changed")
    actual = inventory(python)
    recorded = json.loads(path.with_name("package_inventory.json").read_text())
    if actual != recorded:
        raise ValueError("conda_inventory_changed")
    validate_inventory(actual, combined_lock())
    imports = Path(record["import_path"])
    pth = Path(actual["purelib"]) / "robodojo_uc71_source.pth"
    if (imports != BASE/"runtime/conda_imports" or imports.resolve(strict=True) != imports
            or any(imports.iterdir())
            or record.get("source_pth") != {"path":str(pth), "sha256":digest(pth)}):
        raise ValueError("conda_import_paths_changed")
    return {"layout":"consolidated_conda", "record":str(path), "record_sha256":digest(path),
            "base_inventory":actual, "overlay_inventory":{"packages":[]},
            "overlay_target":str(imports), "target":str(imports),
            "install_record_sha256":digest(path), "dependency_record_sha256":digest(path)}


def default_runtime_config():
    icd = next((p for p in (Path('/usr/share/vulkan/icd.d/nvidia_icd.json'),
                            Path('/etc/vulkan/icd.d/nvidia_icd.json')) if p.is_file()), None)
    if icd is None:
        raise ValueError("nvidia_vulkan_icd_required")
    return {"python":sys.executable, "library_path":str(BASE/"runtime/sysroot/usr/lib/x86_64-linux-gnu"),
            "icd":str(icd), "overlay_target":str(BASE/"runtime/conda_imports"),
            "compat_target":str(BASE/"runtime/conda_imports"),
            "worker_script":str(BASE/"scripts/run_native_worker.py"),
            "maximum_wall_time_s":10800, "ready_timeout_s":600, "minimum_free_mib":16384}
