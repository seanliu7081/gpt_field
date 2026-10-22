#!/usr/bin/env python3
"""Frozen, RGB-only DepthPro front end; no simulator or evaluation imports.

The input JSON contains only ``inputs: [{name, rgb, sha256}]``. Relative RGB
paths are resolved against that JSON. The model receives pixels only, with its
own predicted field of view. Output depth is axial camera Z in meters; no
ground-truth scaling, intrinsics, cropping, or outcome-dependent tuning occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ID = "apple/DepthPro-hf"
REVISION = "de816c8ce7168afcb231f96d501d72b869d0beda"
WEIGHT_SHA256 = "9c6811e3165485b9a94a204329860cb333a79877e757eb795a179a4ea34bbcf7"
EXPECTED_NAMES = {
    "bowl002_initial", "drawer003_closed_step60", "drawer003_open_step552"
}
MODEL_FILES = ("model.safetensors", "config.json", "preprocessor_config.json", "README.md")
PINNED_METADATA_SHA256 = {
    "config.json": "995993be477aefdf41468ca51e171afd26b59aeee4b0afc4ef16957f0b622222",
    "preprocessor_config.json": "88c3e95c10d3780840674bfd52581de03c547cf0a4976b9c4779dd8e2fc5af05",
    "README.md": "9137c19eca509c4e6145ce6d2be0140e059e6221dd88d5937d13ef1c2bce9ee2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def predicted_intrinsics(focal_length_px: float, width: int, height: int) -> np.ndarray:
    if not math.isfinite(focal_length_px) or focal_length_px <= 0:
        raise ValueError("Predicted focal length must be finite and positive")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    # Integer pixel indices denote pixel centers; origin is the top-left center.
    return np.array([[focal_length_px, 0.0, (width - 1) / 2],
                     [0.0, focal_length_px, (height - 1) / 2],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def read_inputs(path: Path, require_three: bool = True) -> list[dict]:
    """Validate a deliberately restricted pixels-only input manifest."""
    path = path.resolve()
    data = json.loads(path.read_text())
    if set(data) - {"schema_version", "inputs"}:
        raise ValueError("Inference input manifest may contain only schema_version and inputs")
    if data.get("schema_version") != 1 or not isinstance(data.get("inputs"), list):
        raise ValueError("Expected schema_version=1 and an inputs list")
    results, seen = [], set()
    for item in data["inputs"]:
        if set(item) != {"name", "rgb", "sha256"}:
            raise ValueError("Each input must contain exactly name, rgb, sha256")
        name = item["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Unsafe or duplicate image name")
        seen.add(name)
        rgb = Path(item["rgb"])
        if not rgb.is_absolute():
            rgb = path.parent / rgb
        rgb = rgb.resolve()
        digest = sha256(rgb)
        if digest != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
        with Image.open(rgb) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"Expected explicit RGB input: {name}, mode={image.mode}")
            width, height = image.size
        results.append({"name": name, "rgb_path": str(rgb), "rgb_sha256": digest,
                        "width": width, "height": height})
    if not results or (require_three and seen != EXPECTED_NAMES):
        raise ValueError("Expected exactly the three preselected LIBERO RGB snapshots")
    return results


def check_unchanged(hashes: dict[str, str]) -> dict[str, bool]:
    return {path: Path(path).is_file() and sha256(Path(path)) == digest
            for path, digest in hashes.items()}


def validate_prediction(depth: np.ndarray, focal: float, fov: float, shape: tuple[int, int]) -> None:
    if depth.shape != shape or not np.isfinite(depth).all() or not (depth > 0).all():
        raise ValueError("Model depth must have native image shape and finite positive metric values")
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("Model field of view must lie strictly between 0 and 180 degrees")
    expected = 0.5 * shape[1] / math.tan(0.5 * math.radians(fov))
    if not math.isclose(focal, expected, rel_tol=2e-6):
        raise ValueError("Focal length and horizontal field of view disagree")


def prepare_rgb_inputs(captures: Path, output: Path) -> dict:
    """Normalize the three archived policy images, without reading GT metadata."""
    captures, output = captures.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing input output: {output}")
    sources = {name: captures / "snapshots" / name / "agentview.png"
               for name in sorted(EXPECTED_NAMES)}
    original_hashes = {str(path): sha256(path) for path in sources.values()}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "running", "inputs": [],
                "source_sha256": original_hashes,
                "export_code": str(Path(__file__).resolve()),
                "export_code_sha256": sha256(Path(__file__).resolve()),
                "transform": "PIL.Image.Transpose.FLIP_LEFT_RIGHT; no resize, crop, interpolation or depth/geometry access",
                "camera_convention": "upright CV; integer pixel centers; x right, y down, z forward",
                "Pillow": importlib.metadata.version("Pillow")}
    try:
        entries = []
        for name, path in sources.items():
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB":
                    raise ValueError("Archived input must already be RGB")
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target = output / f"{name}.png"
                image.save(target)
            digest = sha256(target)
            entries.append({"name": name, "rgb": target.name, "sha256": digest})
            manifest["inputs"].append({"name": name, "file": target.name,
                                       "sha256": digest, "source": str(path),
                                       "source_sha256": original_hashes[str(path)],
                                       "shape": [image.height, image.width, 3]})
        write_json(output / "rgb_inputs.json", {"schema_version": 1, "inputs": entries})
        manifest["sources_unchanged"] = check_unchanged(original_hashes)
        if not all(manifest["sources_unchanged"].values()):
            raise RuntimeError("Archived source RGB changed while preparing inputs")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    # Refusal happens before any mutation, including writing a failed manifest.
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    inputs_path = args.inputs.resolve()
    inputs_hash = sha256(inputs_path)
    inputs = read_inputs(inputs_path)
    model_dir = args.model_dir.resolve()
    download_info_path = args.download_info.resolve()
    download_info_hash = sha256(download_info_path)
    download_info = json.loads(download_info_path.read_text())
    if download_info.get("repo_id") != REPO_ID or download_info.get("revision") != REVISION:
        raise ValueError("Download manifest does not match the fixed official model revision")
    model_hashes = {str(model_dir / name): sha256(model_dir / name) for name in MODEL_FILES}
    if model_hashes[str(model_dir / "model.safetensors")] != WEIGHT_SHA256:
        raise ValueError("Weights do not match official pinned Hugging Face LFS SHA256")
    for name, expected in PINNED_METADATA_SHA256.items():
        if model_hashes[str(model_dir / name)] != expected:
            raise ValueError(f"Pinned official model metadata changed: {name}")
    config = json.loads((model_dir / "config.json").read_text())
    preprocessing = json.loads((model_dir / "preprocessor_config.json").read_text())
    if config.get("use_fov_model") is not True:
        raise ValueError("RGB-only primary requires the model's learned FOV head")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "predictions", "source", "model_metadata"):
        (output / name).mkdir()
    started = time.time()
    manifest = {
        "schema_version": 1, "status": "running", "started_unix": started,
        "input_manifest": str(inputs_path), "input_manifest_sha256": inputs_hash,
        "inputs": inputs, "predictions": [],
        "model": {"repo_id": REPO_ID, "revision": REVISION, "files_sha256": model_hashes,
                  "download_manifest": str(download_info_path),
                  "download_manifest_sha256": download_info_hash,
                  "source_urls": ["https://huggingface.co/apple/DepthPro-hf",
                                  "https://github.com/apple-aiml-research/ml-depth-pro",
                                  "https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/depth_pro"],
                  "license_model_card_metadata": "apple-amlr",
                  "license_note": "Pinned model card metadata says apple-amlr; its older prose says Apple-ASCL. Card preserved verbatim; no relicensing claim."},
        "configuration": {
            "model_inputs": "RGB pixels only; learned FOV; no GT depth, K, geometry, crop, or scale alignment",
            "device": "cuda:0", "physical_gpu_index": 0, "batch_size": 1,
            "model_dtype": "float16", "postprocessing_dtype": "float32",
            "attention": "sdpa", "seed": 0, "eval_mode": True,
            "preprocessing": preprocessing,
            "native_image_handling": "Full upright RGB, no crop or augmentation; official resize to 1536x1536; predictions resized back to original H,W",
            "postprocessing": "Official processor: focal=0.5*W/tan(FOV_deg*pi/360); scale canonical inverse depth by W/focal; bilinear resize; depth=1/clamp(inverse_depth,1e-4,1e4)",
            "depth_convention": "camera_z", "depth_units": "m",
            "intrinsics_assumption": "fx=fy=predicted focal; zero skew; principal point ((W-1)/2,(H-1)/2); integer coordinates denote pixel centers",
            "no_outcome_tuning": True,
        },
    }
    write_json(output / "manifest.json", manifest)
    try:
        # The subprocess restricts visibility before torch is imported; never touch GPU 1.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_HOME"] = str(HERE / "models" / "_hf_home")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import transformers
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        from transformers.models.depth_pro import modeling_depth_pro, image_processing_depth_pro_fast
        from transformers.models.dinov2 import modeling_dinov2
        from transformers import image_processing_utils_fast

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly physical GPU 0 to be visible")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest["environment"] = {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "transformers", "huggingface-hub", "safetensors", "numpy", "Pillow")}
        manifest["environment"].update({"python": sys.version, "cuda_runtime": torch.version.cuda,
                                        "gpu_name": torch.cuda.get_device_name(0),
                                        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory})
        source_paths = [Path(__file__).resolve()] + [Path(inspect.getfile(module)).resolve() for module in (
            modeling_depth_pro, image_processing_depth_pro_fast, modeling_dinov2, image_processing_utils_fast)]
        manifest["source_sha256"] = {str(path): sha256(path) for path in source_paths}
        frozen_hashes = {}
        for path in source_paths:
            target = output / "source" / path.name
            shutil.copy2(path, target)
            frozen_hashes[str(target)] = sha256(target)
        for name in ("config.json", "preprocessor_config.json", "README.md"):
            target = output / "model_metadata" / name
            shutil.copy2(model_dir / name, target)
            frozen_hashes[str(target)] = sha256(target)
        shutil.copy2(download_info_path, output / "model_metadata" / "download_info.json")
        frozen_hashes[str(output / "model_metadata" / "download_info.json")] = download_info_hash
        shutil.copy2(inputs_path, output / "inputs" / "input_manifest.json")
        frozen_hashes[str(output / "inputs" / "input_manifest.json")] = inputs_hash
        for row in inputs:
            target = output / "inputs" / f"{row['name']}.png"
            shutil.copy2(row["rgb_path"], target)
            if sha256(target) != row["rgb_sha256"]:
                raise RuntimeError("Input changed while freezing")
            row["frozen_rgb"] = str(target.relative_to(output))
            frozen_hashes[str(target)] = row["rgb_sha256"]
        write_json(output / "settings.json", manifest["configuration"])
        frozen_hashes[str(output / "settings.json")] = sha256(output / "settings.json")
        manifest["frozen_artifact_sha256"] = frozen_hashes
        write_json(output / "manifest.json", manifest)

        load_start = time.perf_counter()
        processor = DepthProImageProcessorFast.from_pretrained(str(model_dir), local_files_only=True)
        model = DepthProForDepthEstimation.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.float16,
            attn_implementation="sdpa").to("cuda:0").eval()
        torch.cuda.synchronize()
        manifest["model_load_seconds"] = time.perf_counter() - load_start
        torch.cuda.reset_peak_memory_stats()
        for row in inputs:
            rgb_path = output / row["frozen_rgb"]
            with Image.open(rgb_path) as image:
                image.load()
                image = image.copy()
            tick = time.perf_counter()
            tensors = processor(images=image, return_tensors="pt").to(device="cuda:0", dtype=torch.float16)
            torch.cuda.synchronize()
            inference_tick = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**tensors)
            torch.cuda.synchronize()
            inference_ms = 1000 * (time.perf_counter() - inference_tick)
            # Preserve native network precision but avoid float16 rounding in
            # focal conversion, reciprocal, and interpolation arithmetic.
            outputs.predicted_depth = outputs.predicted_depth.float()
            outputs.field_of_view = outputs.field_of_view.float()
            with torch.inference_mode():
                result = processor.post_process_depth_estimation(
                    outputs, target_sizes=[(row["height"], row["width"])])[0]
            depth = result["predicted_depth"].detach().cpu().numpy().astype(np.float32)
            focal = float(result["focal_length"].detach().cpu())
            fov = float(result["field_of_view"].detach().cpu())
            validate_prediction(depth, focal, fov, (row["height"], row["width"]))
            K = predicted_intrinsics(focal, row["width"], row["height"])
            prediction = output / "predictions" / f"{row['name']}.npz"
            np.savez_compressed(prediction, depth_m=depth, K_pred=K,
                                focal_length_px=np.float64(focal), field_of_view_deg=np.float64(fov))
            entry = {"name": row["name"], "file": str(prediction.relative_to(output)),
                     "sha256": sha256(prediction), "depth_units": "m", "depth_convention": "camera_z",
                     "shape": list(depth.shape), "focal_length_px": focal, "field_of_view_deg": fov,
                     "K_pred": K.tolist(), "inference_ms": inference_ms,
                     "preprocess_inference_postprocess_save_ms": 1000 * (time.perf_counter() - tick),
                     "depth_quantiles_m": {str(q): float(np.quantile(depth, q)) for q in (0, .01, .5, .99, 1)},
                     "rgb_sha256": row["rgb_sha256"],
                     "model_revision": REVISION}
            manifest["predictions"].append(entry)
            write_json(output / "manifest.json", manifest)
            print(json.dumps({"name": row["name"], "inference_ms": inference_ms,
                              "focal_length_px": focal, "depth_median_m": float(np.median(depth))}), flush=True)
            del outputs, tensors, result
        manifest["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del model
        torch.cuda.empty_cache()
        manifest["sources_unchanged"] = check_unchanged(manifest["source_sha256"])
        manifest["model_files_unchanged"] = check_unchanged(model_hashes)
        manifest["frozen_artifacts_unchanged"] = check_unchanged(frozen_hashes)
        manifest["inputs_unchanged"] = check_unchanged(
            {str(inputs_path): inputs_hash, str(download_info_path): download_info_hash,
             **{row["rgb_path"]: row["rgb_sha256"] for row in inputs}})
        manifest["predictions_unchanged"] = check_unchanged(
            {str(output / row["file"]): row["sha256"] for row in manifest["predictions"]})
        for key in ("sources_unchanged", "model_files_unchanged", "frozen_artifacts_unchanged",
                    "inputs_unchanged", "predictions_unchanged"):
            if not manifest[key] or not all(manifest[key].values()):
                raise RuntimeError(f"Terminal integrity check failed: {key}")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_unix"] = time.time()
        manifest["wall_seconds"] = time.time() - started
        write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="RGB-only JSON: schema_version=1, inputs=[{name,rgb,sha256}]")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing outputs refused")
    parser.add_argument("--model-dir", type=Path, default=HERE / "models" / "apple_DepthPro_hf")
    parser.add_argument("--download-info", type=Path, default=HERE / "models" / "download_info.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
