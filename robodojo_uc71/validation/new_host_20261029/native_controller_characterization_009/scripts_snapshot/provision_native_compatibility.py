#!/usr/bin/env python3
"""Install two scoped compatibility packages without modifying prior environments.

Only importlib metadata and hid/setuptools/pkg_resources imports are exercised.
No simulator, Torch, model, HID device, or benchmark is started.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import urllib.request

from provision_native_common import digest
from provision_native_dependencies import METADATA_CODE, package_map

BASE = Path(__file__).resolve().parents[1]
PINS = {"hidapi": "0.14.0.post2", "setuptools": "80.9.0"}
OVERRIDES = {"setuptools": {"base_version": "84.2.1", "effective_version": "80.9.0",
                          "reason": "Kit pipapi requires the removed pkg_resources API"}}
WHEEL_HASHES = {
    "hidapi-0.14.0.post2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl":
        "f95340c0d69245820a2075d516df9a99c466d92db29fa78df5f13d9cfee4716d",
    "setuptools-80.9.0-py3-none-any.whl": "062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata(python: Path, target: Path | None = None) -> dict:
    argv = [str(python), "-I", "-B", "-c", METADATA_CODE]
    if target is not None:
        argv.append(str(target))
    return json.loads(subprocess.check_output(argv, text=True, timeout=30,
        env={"PATH": str(python.parent) + ":/usr/bin:/bin", "PYTHONNOUSERSITE": "1"}))


def baseline(install: Path, dependency: Path) -> dict:
    original, deps = read_json(install), read_json(dependency)
    python = BASE / "runtime/new_host_env/bin/python"
    target = BASE / "runtime/native_overlay"
    if (original.get("ready_for_diagnostic") is not True
            or Path(original.get("python", "")).absolute() != python
            or digest(install.with_name("requirements.lock")) != original.get("lock_sha256")
            or deps.get("ready_for_native_preview_dependencies") is not True
            or deps.get("base_unchanged") is not True
            or Path(deps.get("base_install_record", "")).resolve() != install
            or Path(deps.get("target", "")).resolve() != target
            or deps.get("base_lock_sha256") != original["lock_sha256"]):
        raise ValueError("Completed, unchanged base and native dependency records are required")
    for name, value in deps["artifacts_sha256"].items():
        if Path(name).name != name or digest(dependency.parent / name) != value:
            raise ValueError("Native dependency artifact changed")
    actual_base, actual_overlay = metadata(python), metadata(python, target)
    base, overlay = package_map(actual_base), package_map(actual_overlay)
    if (base != package_map(read_json(install.with_name("package_inventory.json")))
            or overlay != deps["expected_overlay_packages"] or len(base) != 140 or len(overlay) != 12
            or set(base) & set(overlay) or base.get("setuptools") != "84.2.1"
            or base.get("isaacsim") != "5.1.0.0" or base.get("torch") != "2.7.0+cu128"
            or actual_base["python_version"] != [3, 11, 16]):
        raise ValueError("Actual baseline metadata differs from preserved 140+12 installation")
    return {"python": str(python), "base_inventory": actual_base, "overlay_inventory": actual_overlay,
            "install_record": str(install), "install_record_sha256": digest(install),
            "dependency_record": str(dependency), "dependency_record_sha256": digest(dependency)}


def locked_versions(text: str) -> dict:
    result = {}
    for line in text.replace("\\\n", " ").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        match = re.fullmatch(r"([a-z0-9-]+)==([A-Za-z0-9.+]+)", tokens[0])
        if (match is None or len(tokens) < 2 or match[1] in result
                or any(not re.fullmatch(r"--hash=sha256:[a-f0-9]{64}", token) for token in tokens[1:])):
            raise ValueError("Unexpected compatibility lock syntax")
        result[match[1]] = match[2]
    if result != PINS:
        raise ValueError("Compatibility lock must contain exactly the two reviewed pins")
    return result


def verify_compatibility_record(python: Path, path: Path, dependency_info: dict) -> dict:
    """Read-only parent gate; the single setuptools override is explicit."""
    record = read_json(path)
    target = BASE / "runtime/native_compat"
    if (record.get("schema") != "robodojo-native-compatibility-1"
            or record.get("ready_for_native_preview_compatibility") is not True
            or record.get("baseline_unchanged") is not True
            or record.get("pins") != PINS or record.get("overrides") != OVERRIDES
            or record.get("physical_distribution_counts") != {"base": 140, "native_overlay": 12, "compatibility": 2}
            or record.get("effective_distribution_count") != 153
            or record.get("python") != str(python) or record.get("target") != str(target)
            or record["baseline"]["install_record_sha256"] != dependency_info["install_record_sha256"]
            or record["baseline"]["dependency_record_sha256"] != dependency_info["dependency_record_sha256"]):
        raise ValueError("Compatibility installation is not bound to this unchanged native baseline")
    for name, expected in record["artifacts_sha256"].items():
        if Path(name).name != name or digest(path.parent / name) != expected:
            raise ValueError("Compatibility provenance changed")
    locked_versions(path.with_name("requirements.lock").read_text())
    actual = metadata(python, target)
    if package_map(actual) != PINS or actual != read_json(path.with_name("compatibility_inventory.json")):
        raise ValueError("Actual compatibility metadata differs from its recorded two-package inventory")
    probe = read_json(path.with_name("import_check.json"))
    if probe.get("passed") is not True or probe.get("versions") != PINS or probe.get("forbidden_imports") != []:
        raise ValueError("Compatibility import-only check did not pass")
    return {"record": str(path), "record_sha256": digest(path), "target": str(target),
            "pins": PINS, "overrides": OVERRIDES, "physical_distribution_count": 2,
            "effective_distribution_count": 153, "inventory": actual,
            "import_check_sha256": digest(path.with_name("import_check.json"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("install-record", "dependency-record", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output, target = args.output.resolve(), BASE / "runtime/native_compat"
    if output.exists() or not output.is_relative_to(BASE / "validation") or target.exists() or target.is_symlink():
        parser.error("Use a fresh validation directory and previously absent native_compat target")
    install, dependency = args.install_record.resolve(strict=True), args.dependency_record.resolve(strict=True)
    before = baseline(install, dependency)
    python = Path(before["python"])
    native_setup = BASE / "runtime/native/RoboDojo/third_party/IsaacLab/source/isaaclab/setup.py"
    if '"hidapi==0.14.0.post2"' not in native_setup.read_text():
        parser.error("Pinned native hidapi requirement differs")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("home", "tmp", "uv-cache"):
        (output / name).mkdir()
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required")
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output / "home"),
           "TMPDIR": str(output / "tmp"), "UV_CACHE_DIR": str(output / "uv-cache"),
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    state = {"schema": "robodojo-native-compatibility-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": str(python), "target": str(target), "pins": PINS, "overrides": OVERRIDES,
        "baseline": before, "baseline_unchanged": False, "ready_for_native_preview_compatibility": False,
        "physical_distribution_counts": {"base": 140, "native_overlay": 12, "compatibility": 2},
        "effective_distribution_count": 153, "script_sha256": digest(Path(__file__)),
        "native_setup_sha256": digest(native_setup), "events": [], "simulator_starts": 0,
        "model_calls": 0, "benchmark_actions": 0, "full_test_authorized": False,
        "setuptools_note": "80.9.0 is the conservative <81 compatibility pin; pkg_resources was removed in82.0.0, not81.0.0.",
        "primary_sources": ["https://pypi.org/project/hidapi/0.14.0.post2/",
                            "https://pypi.org/project/setuptools/80.9.0/",
                            "https://setuptools.pypa.io/en/latest/history.html#v82-0-0"]}

    def save():
        write_json(output / "install_status.json", state)

    def run(argv, *, custom_env=None, timeout=300):
        index = len(state["events"])
        event = {"argv": argv, "stdout": f"command_{index:02d}.stdout.log", "stderr": f"command_{index:02d}.stderr.log"}
        state["events"].append(event); save()
        print(json.dumps(event), flush=True)
        with (output / event["stdout"]).open("x") as out, (output / event["stderr"]).open("x") as err:
            proc = subprocess.Popen(argv, cwd=output, env=custom_env or env, stdout=out, stderr=err, start_new_session=True)
            try:
                event["returncode"] = proc.wait(timeout=timeout)
            except BaseException:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(); raise
            finally:
                save()
        if event["returncode"]:
            raise RuntimeError(f"Compatibility command failed: {event['stderr']}")
        return output / event["stdout"]

    save()
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    try:
        write_json(output / "baseline_before.json", before)
        for name, version in PINS.items():
            url = f"https://pypi.org/pypi/{name}/{version}/json"
            with urllib.request.urlopen(url, timeout=30) as response:
                info = json.load(response)
            selected = [row for row in info["urls"] if row["filename"] in WHEEL_HASHES]
            if (len(selected) != 1 or selected[0]["yanked"]
                    or selected[0]["digests"]["sha256"] != WHEEL_HASHES[selected[0]["filename"]]):
                raise ValueError(f"Reviewed public wheel metadata differs: {name}")
            write_json(output / f"pypi_{name}.json", {"source": url, "metadata": info})
        combined = {**package_map(before["base_inventory"]), **package_map(before["overlay_inventory"])}
        constraints = output / "baseline_constraints_except_setuptools.txt"
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(combined.items()) if name != "setuptools"))
        requirements = output / "requirements.in"
        requirements.write_text("".join(f"{name}=={version}\n" for name, version in PINS.items()))
        lock = output / "requirements.lock"
        run([uv, "--no-config", "pip", "compile", "--python", str(python), "--only-binary", ":all:",
             "--default-index", "https://pypi.org/simple", "--no-sources", "--generate-hashes", "--no-annotate", "--no-header",
             "--constraints", str(constraints), "--output-file", str(lock), str(requirements)])
        locked_versions(lock.read_text())
        target.mkdir(exist_ok=False)
        run([uv, "--no-config", "pip", "install", "--python", str(python), "--target", str(target),
             "--require-hashes", "--no-deps", "--only-binary", ":all:", "--no-sources",
             "--default-index", "https://pypi.org/simple", "--link-mode", "copy", "-r", str(lock)])
        actual = metadata(python, target)
        if package_map(actual) != PINS:
            raise ValueError("Unexpected compatibility distribution inventory")
        write_json(output / "compatibility_inventory.json", actual)
        probe_env = dict(env, PYTHONPATH=str(target) + os.pathsep + str(BASE / "runtime/native_overlay"))
        probe_code = """import importlib.metadata as m,json,sys
from pathlib import Path
import setuptools,pkg_resources,hid
target=Path(sys.argv[1]).resolve()
versions={name:m.version(name) for name in ('hidapi','setuptools')}
modules={name:str(sys.modules[name].__file__) for name in ('setuptools','pkg_resources','hid','_distutils_hack')}
for path in modules.values():
 assert Path(path).resolve().is_relative_to(target),path
assert versions=={'hidapi':'0.14.0.post2','setuptools':'80.9.0'}
pkg_resources._initialize_master_working_set()
assert pkg_resources.get_distribution('hidapi').version=='0.14.0.post2'
assert pkg_resources.get_distribution('setuptools').version=='80.9.0'
forbidden=[name for name in sys.modules if name.split('.')[0] in ('torch','isaacsim','isaaclab','omni')]
assert not forbidden
print(json.dumps({'passed':True,'versions':versions,'modules':modules,'forbidden_imports':forbidden,
 'device_enumeration_calls':0,'device_open_calls':0,'simulator_starts':0}))
"""
        probe = json.loads(run([str(python), "-B", "-c", probe_code, str(target)], custom_env=probe_env, timeout=30).read_text())
        write_json(output / "import_check.json", probe)
        after = baseline(install, dependency)
        write_json(output / "baseline_after.json", after)
        if after != before:
            raise ValueError("Preserved baseline changed during compatibility setup")
        effective = {**combined, **PINS}
        if len(effective) != 153:
            raise ValueError("Unexpected effective distribution count")
        write_json(output / "effective_inventory.json", {"packages": [{"name": n, "version": v} for n, v in sorted(effective.items())],
            "precedence": [str(target), str(BASE / "runtime/native_overlay"), str(python.parent.parent)], "overrides": OVERRIDES})
        state["baseline_unchanged"] = True
        state["artifacts_sha256"] = {name: digest(output / name) for name in (
            "baseline_before.json", "baseline_after.json", "pypi_hidapi.json", "pypi_setuptools.json",
            "baseline_constraints_except_setuptools.txt", "requirements.in", "requirements.lock",
            "compatibility_inventory.json", "effective_inventory.json", "import_check.json")}
        state["ready_for_native_preview_compatibility"] = True
        state["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(); save()
        print(json.dumps({"ready": True, "target": str(target), "effective_distributions": 153}), flush=True)
        return 0
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        try:
            state["baseline_unchanged"] = baseline(install, dependency) == before
        except Exception as check_error:
            state["baseline_check_error"] = repr(check_error)
        save(); print(json.dumps(state), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
