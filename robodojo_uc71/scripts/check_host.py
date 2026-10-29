#!/usr/bin/env python3
"""Read-only Isaac Sim 5.1 host inventory; never imports or launches the simulator.

Only --output writes a file, exclusively creating a new JSON record. GPU/library
inventory is preliminary evidence, not proof of working rendering or scene fit.
Exit status zero means the inventory was produced, never approval to run a test.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

GIB = 1024**3
TESTED_DRIVER = "580.65.06"
CURRENT_OMNIVERSE_R580_REFERENCE = "580.95.05"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "isaac_sim_5_1_requirements": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html",
    "omniverse_driver_recommendations": "https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html",
}


def driver_version(value: str) -> tuple[int, ...] | None:
    """Accept only a complete dotted numeric driver version, not arbitrary labels."""
    if not re.fullmatch(r"\d+(?:\.\d+){1,3}", value.strip()):
        return None
    return tuple(int(part) for part in value.strip().split("."))


def assess_gpu(gpu: dict) -> dict:
    version = driver_version(gpu.get("driver_version", ""))
    baseline = driver_version(TESTED_DRIVER)
    free_mib = gpu.get("memory_free_mib")
    total_mib = gpu.get("memory_total_mib")
    return {
        "driver_at_or_above_5_1_tested_version": None if version is None else version >= baseline,
        "same_r580_branch_as_5_1_tested_version": None if version is None else version[0] == 580,
        "exact_5_1_tested_driver": None if version is None else version == baseline,
        "total_vram_at_least_16_gib": None if total_mib is None else total_mib >= 16 * 1024,
        "currently_free_vram_at_least_16_gib": None if free_mib is None else free_mib >= 16 * 1024,
        "rt_core_support": "not_determined_by_this_inventory; verify GPU model against NVIDIA requirements",
        "runtime_compatibility": "unverified; a newer driver alone does not prove compatibility",
        "native_layout_peak_vram": "unmeasured",
    }


def run_readonly(argv: list[str]) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "error": str(error)}


def number(value: str) -> int | float | None:
    try:
        parsed = float(value.strip())
        return int(parsed) if parsed.is_integer() else parsed
    except ValueError:
        return None


def gpu_inventory() -> dict:
    command = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,memory.free,compute_cap",
               "--format=csv,noheader,nounits"]
    result = run_readonly(command)
    gpus = []
    if result["returncode"] == 0:
        for row in csv.reader(io.StringIO(result["stdout"]), skipinitialspace=True):
            if len(row) != 9:
                continue
            gpu = dict(zip(("index", "uuid", "pci_bus_id", "name", "driver_version",
                            "memory_total_mib", "memory_used_mib", "memory_free_mib", "compute_capability"),
                           (item.strip() for item in row)))
            for key in ("index", "memory_total_mib", "memory_used_mib", "memory_free_mib"):
                gpu[key] = number(gpu[key])
            gpu["assessment"] = assess_gpu(gpu)
            gpus.append(gpu)
    return {"command": command, "returncode": result["returncode"], "gpus": gpus,
            "error": result.get("error") or result.get("stderr") or None}


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def memory_limit(value: str | None) -> int | None:
    """None means unlimited/absent; v1 uses a near-LONG_MAX sentinel."""
    if value is None or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if 0 <= parsed < 2**60 else None


def unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def cgroup_mounts(cgroups: str, mounts: str) -> list[dict]:
    """Resolve a process group relative to the mounted hierarchy (including bind roots)."""
    groups = {}
    for line in cgroups.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, group = parts
        if "memory" in controllers.split(","):
            groups["v1"] = group
        elif controllers == "":
            groups["v2"] = group
    found = []
    for line in mounts.splitlines():
        before, separator, after = line.partition(" - ")
        fields, tail = before.split(), after.split()
        if not separator or len(fields) < 5 or len(tail) < 3:
            continue
        version = "v2" if tail[0] == "cgroup2" else "v1" if tail[0] == "cgroup" and "memory" in tail[2].split(",") else None
        if version is None or version not in groups:
            continue
        root, mount = Path(unescape_mount(fields[3])), Path(unescape_mount(fields[4]))
        group = Path(groups[version])
        try:
            relative = group.relative_to(root)
        except ValueError:
            # With a private cgroup namespace /proc may already be relative to its root.
            if group != Path("/"):
                continue
            relative = Path(".")
        found.append({"version": version, "mount_root": str(root), "mount_point": str(mount),
                      "process_group": str(group), "directory": str(mount / relative)})
    return found


def cgroup_memory(cgroups: str, mounts: str) -> dict:
    entries = []
    for binding in cgroup_mounts(cgroups, mounts):
        directory, mount = Path(binding["directory"]), Path(binding["mount_point"])
        while True:
            v2 = binding["version"] == "v2"
            limit_path = directory / ("memory.max" if v2 else "memory.limit_in_bytes")
            usage_path = directory / ("memory.current" if v2 else "memory.usage_in_bytes")
            raw_limit, raw_usage = read_text(limit_path), read_text(usage_path)
            limit, usage = memory_limit(raw_limit), memory_limit(raw_usage)
            stat = {}
            if not v2:
                for line in (read_text(directory / "memory.stat") or "").splitlines():
                    fields = line.split()
                    if len(fields) == 2:
                        stat[fields[0]] = fields[1]
            inherited = memory_limit(stat.get("hierarchical_memory_limit"))
            finite = [item for item in (limit, inherited) if item is not None]
            effective_limit = min(finite) if finite else None
            entries.append({**binding, "directory": str(directory), "limit_file": str(limit_path),
                            "limit_raw": raw_limit, "limit_bytes": limit, "usage_bytes": usage,
                            "hierarchical_limit_bytes": inherited, "effective_limit_bytes": effective_limit,
                            "headroom_bytes": None if effective_limit is None or usage is None else max(0, effective_limit - usage),
                            "limit_file_readable": raw_limit is not None})
            if directory == mount:
                break
            directory = directory.parent
    return {"entries": entries, "limits_observed": bool(entries) and any(e["limit_file_readable"] for e in entries),
            "limitation": "Only visible cgroup limits are inspected. Hidden ancestor limits/shared usage may reduce available memory."}


def memory_inventory(proc_root: Path = Path("/proc")) -> dict:
    host = {}
    for line in (read_text(proc_root / "meminfo") or "").splitlines():
        fields = line.split()
        if fields and fields[0] in ("MemTotal:", "MemAvailable:") and len(fields) >= 2:
            host[fields[0][:-1]] = int(fields[1]) * 1024
    cgroup = cgroup_memory(read_text(proc_root / "self/cgroup") or "", read_text(proc_root / "self/mountinfo") or "")
    capacities = [host["MemTotal"]] if "MemTotal" in host else []
    headrooms = [host["MemAvailable"]] if "MemAvailable" in host else []
    capacities.extend(entry["effective_limit_bytes"] for entry in cgroup["entries"] if entry["effective_limit_bytes"] is not None)
    headrooms.extend(entry["headroom_bytes"] for entry in cgroup["entries"] if entry["headroom_bytes"] is not None)
    capacity = min(capacities) if capacities else None
    available = min(headrooms) if headrooms else None
    return {"proc_memtotal_bytes": host.get("MemTotal"), "proc_memavailable_bytes": host.get("MemAvailable"),
            "cgroup": cgroup, "effective_capacity_upper_bound_bytes": capacity,
            "effective_available_upper_bound_bytes": available,
            "capacity_at_least_32_gib": None if capacity is None else capacity >= 32 * GIB,
            "currently_available_at_least_32_gib": None if available is None else available >= 32 * GIB}


def graphics_inventory() -> dict:
    executable = shutil.which("ldconfig") or "/usr/sbin/ldconfig"
    result = run_readonly([executable, "-p"])
    cache = {}
    for line in result["stdout"].splitlines():
        match = re.match(r"\s*(\S+)\s+\([^)]*\)\s+=>\s+(.+)", line)
        if match:
            cache.setdefault(match[1], []).append(match[2])
    names = {"libvulkan.so.1", "libGL.so.1", "libEGL.so.1", "libGLX.so.0",
             "libGLX_nvidia.so.0", "libEGL_nvidia.so.0", "libnvoptix.so.1", "libcuda.so.1"}
    names.update(name for name in cache if name.startswith(("libnvidia-rtcore.so", "libnvidia-glvkspirv.so")))

    def inspect_path(path: str) -> dict:
        item = Path(path)
        return {"path": str(item), "resolved_path": str(item.resolve()), "file_exists": item.is_file()}

    libraries = {name: [inspect_path(path) for path in cache.get(name, [])] for name in sorted(names)}
    icds = []
    for directory in ("/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/usr/local/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"):
        for path in sorted(Path(directory).glob("*.json")):
            try:
                metadata = json.loads(path.read_text()).get("ICD", {})
                library = metadata.get("library_path", "")
                candidates = [str(path.parent / library)] if "/" in library else cache.get(library, [])
                icds.append({"json_path": str(path), "library_path": library,
                             "api_version": metadata.get("api_version"),
                             "resolved_candidates": [inspect_path(candidate) for candidate in candidates]})
            except (OSError, ValueError, TypeError, AttributeError) as error:
                icds.append({"json_path": str(path), "error": str(error)})
    return {"ldconfig_returncode": result["returncode"], "error": result.get("error") or result.get("stderr") or None,
            "libraries": libraries, "vulkan_icds": icds,
            "limitation": "File presence and resolved symlinks only; libraries were not loaded. Vulkan enumeration, RTX rendering and loader override behavior remain untested."}


def collect(disk_path: Path) -> dict:
    disk = shutil.disk_usage(disk_path)
    os_release = {}
    for line in (read_text(Path("/etc/os-release")) or "").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in ("ID", "VERSION_ID", "PRETTY_NAME"):
            os_release[key] = value.strip('"')
    return {
        "schema_version": "robodojo-host-preflight-1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_runtime": {"isaac_sim": "5.1.0.0", "python": "3.11", "bootstrap_python_pin": "3.11.16",
                             "torch": "2.7.0+cu128", "installed_versions_verified": False},
        "documented_requirements": {"isaac_sim_5_1_tested_linux_driver": TESTED_DRIVER,
                                    "current_omniverse_r580_linux_reference": CURRENT_OMNIVERSE_R580_REFERENCE,
                                    "reference_checked_utc_date": "2026-10-29", "minimum_vram_gb": 16, "minimum_ram_gb": 32,
                                    "comparison_policy": "Conservatively compare published GB minima as GiB; driver ordering is an inventory check, not a compatibility guarantee.",
                                    "sources": SOURCES},
        "os": {**os_release, "architecture": platform.machine(), "kernel": platform.release(),
               "preflight_python": platform.python_version()},
        "gpu": gpu_inventory(), "system_memory": memory_inventory(),
        "disk": {"path": str(disk_path.resolve()), "total_bytes": disk.total, "free_bytes": disk.free,
                 "free_at_least_setup_estimate_60_gib": disk.free >= 60 * GIB,
                 "note": "60 GiB is the historical package setup estimate; native asset dependency size remains unresolved."},
        "graphics": graphics_inventory(),
        "decision": {"full_test_authorized": False, "runtime_compatibility_verified": False,
                     "full_test_ready": False, "required_next_evidence": ["Bounded model-free Isaac Sim RGB camera pass",
                     "Pinned native layout 0 scene with measured peak GPU/RAM use", "Controller/native integration checks",
                     "Actual preview and plan, followed by explicit user approval"]},
        "actions": {"simulator_starts": 0, "model_calls": 0, "package_installs": 0, "gpu_contexts_requested": 0,
                    "host_configuration_changes": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Exclusively create a NEW JSON file; parent must exist. Default: stdout.")
    parser.add_argument("--disk-path", type=Path, default=ROOT, help="Existing filesystem path to inspect for free storage.")
    args = parser.parse_args()
    try:
        document = json.dumps(collect(args.disk_path), indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(document)
        else:
            with args.output.open("x") as stream:
                stream.write(document)
    except (OSError, ValueError) as error:
        print(f"Preflight failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
