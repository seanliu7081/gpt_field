#!/usr/bin/env bash
# Isolated inference environment; never installs into the repository .venv.
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${FOUNDATION_VISION_ENV:-${root_dir}/.venv-foundation-vision}"
vendor_dir="${root_dir}/artifacts/foundation_vision/vendor"
if [[ "${env_dir}" == "${root_dir}/.venv" ]]; then
  echo 'Refusing to modify the existing repository .venv.' >&2
  exit 2
fi
python3 -m venv "${env_dir}"
"${env_dir}/bin/python" -m pip install --upgrade pip 'setuptools<81' wheel
"${env_dir}/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
mkdir -p "${vendor_dir}"
clone_pinned() {
  local source_url="$1" destination="$2" revision="$3"
  if [[ ! -d "${destination}" ]]; then
    git clone --filter=blob:none "${source_url}" "${destination}"
  fi
  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    echo "Refusing to replace modified vendor source: ${destination}" >&2
    exit 2
  fi
  git -C "${destination}" checkout --detach "${revision}"
}
clone_pinned https://github.com/facebookresearch/sam3.git "${vendor_dir}/sam3" f1647e8d0c1d280991f4d1b7d974c2704537fe66
clone_pinned https://github.com/ByteDance-Seed/Depth-Anything-3.git "${vendor_dir}/depth-anything-3" 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
# Minimal inference dependencies. Optional training/UI/3DGS packages are omitted.
"${env_dir}/bin/python" -m pip install 'numpy>=1.26,<2' pillow scipy pytest timm 'ftfy==6.1.1' regex iopath typing_extensions huggingface_hub einops omegaconf safetensors 'opencv-python-headless<4.12' imageio trimesh evo e3nn plyfile 'moviepy==1.0.3' pillow_heif pycolmap addict pycocotools psutil
"${env_dir}/bin/python" -m pip install --no-deps -e "${vendor_dir}/sam3" -e "${vendor_dir}/depth-anything-3"
echo "Installed isolated environment: ${env_dir}"
echo 'Model weights are downloaded only by an explicit adapter load/offline run.'
echo 'SAM 3.1 additionally requires an approved authenticated Hugging Face account.'
