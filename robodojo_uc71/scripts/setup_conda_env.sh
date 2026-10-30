#!/usr/bin/env bash
# Provision packages and import paths only; never launch a simulator or model.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${1:-robodojo_uc71}"
conda_exe="${CONDA_EXE:-conda}"

if ! "$conda_exe" run -n "$env_name" python --version >/dev/null 2>&1; then
    "$conda_exe" create -n "$env_name" python=3.11.16 pip uv ffmpeg -y
fi
"$conda_exe" run -n "$env_name" python -c 'import sys; assert sys.version_info[:3] == (3, 11, 16), "Expected Python 3.11.16; use a fresh environment name"'
if ! "$conda_exe" run -n "$env_name" ffmpeg -version >/dev/null 2>&1; then
    "$conda_exe" install -n "$env_name" --freeze-installed ffmpeg -y
fi
env_python="$("$conda_exe" run -n "$env_name" python -c 'import sys; print(sys.executable)')"
env_uv="$(dirname -- "$env_python")/uv"
mkdir -p "$task_root/runtime"
setup_dir="$(mktemp -d "$task_root/runtime/conda-setup.XXXXXXXX")"

# Ubuntu's GLU runtime is needed by Isaac Sim's MDL renderer. Extract missing
# libraries locally; no sudo or system package changes are needed.
library_dir="$task_root/runtime/sysroot/usr/lib/x86_64-linux-gnu"
if ! "$env_python" -c 'import ctypes; ctypes.CDLL("libGLU.so.1")' >/dev/null 2>&1 \
        && [[ ! -f "$library_dir/libGLU.so.1" ]]; then
    mkdir -p "$task_root/runtime/system_packages" "$task_root/runtime/sysroot"
    (
        cd "$task_root/runtime/system_packages"
        apt-get download libglu1-mesa libopengl0
        for package in ./*.deb; do
            dpkg-deb -x "$package" "$task_root/runtime/sysroot"
        done
    )
fi

"$env_python" - "$task_root" "$setup_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root, output = map(Path, sys.argv[1:])
licence = root / "validation/provisioning/vendor_authorization.json"
if json.loads(licence.read_text()).get("nvidia_eula_accepted") is not True:
    raise SystemExit("The recorded NVIDIA EULA acceptance is required")
inputs = [
    root / "validation/provisioning/a1_requirements.lock",
    root / "validation/new_host_20261029/native_dependencies_001/requirements.lock",
    root / "validation/new_host_20261029/native_compatibility_001/requirements.lock",
]
# The reviewed compatibility layer replaces setuptools 84.2.1 with 80.9.0
# because Isaac Sim 5.1 uses pkg_resources. Keep every other original pin/hash.
base = inputs[0].read_text()
base, count = re.subn(r"^setuptools==[^\n]+(?:\n[ \t]+[^\n]*)*\n?", "", base, flags=re.M)
if count != 1:
    raise SystemExit("Expected exactly one preserved setuptools requirement")
lock = base + "\n" + inputs[1].read_text() + "\n" + inputs[2].read_text()
(output / "requirements.lock").write_text(lock)
(output / "provenance.json").write_text(json.dumps({
    "python": sys.version,
    "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    "setuptools_override": "80.9.0, from the preserved compatibility lock",
    "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
}, indent=2) + "\n")
PY

export UV_CACHE_DIR="$task_root/runtime/uv-cache"
"$env_uv" pip install --python "$env_python" --require-hashes \
    --default-index https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
    -r "$setup_dir/requirements.lock"
"$env_uv" pip install --python "$env_python" --require-hashes \
    --default-index https://pypi.org/simple -r "$task_root/configs/conda_test_requirements.lock"
"$env_python" - "$task_root" <<'PY'
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1])
paths = [root / "src", root / "vendor/uc71/src"]
native = root / "runtime/native/RoboDojo"
if native.is_dir():
    paths += [native, native / "XPolicyLab"]
    paths += [native / "third_party/IsaacLab/source" / name
              for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks")]
pth = Path(sysconfig.get_path("purelib")) / "robodojo_uc71_source.pth"
pth.write_text("".join(str(path) + "\n" for path in paths))
print("Installed adapter and frozen core import paths:", pth)
PY
"$env_uv" pip check --python "$env_python"
"$conda_exe" env config vars set -n "$env_name" PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -d "$library_dir" ]]; then
    "$conda_exe" env config vars set -n "$env_name" "LD_LIBRARY_PATH=$library_dir"
fi
"$env_python" -m pip freeze > "$setup_dir/pip-freeze.txt"
"$conda_exe" list -n "$env_name" --explicit > "$setup_dir/conda-explicit.txt"
"$env_python" "$task_root/scripts/prepare_conda_runtime.py" --output "$setup_dir/verified-runtime"
printf 'Environment ready: conda activate %s\nRecords: %s\n' "$env_name" "$setup_dir"
