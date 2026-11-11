"""Lazy, explicit adapters for SAM 3.1 detection and DA3 geometry hypotheses.

Importing this module requires only NumPy. Model weights are never substituted,
and neither scores nor depth predictions are physical safety certificates.
SAM uses the 3.1 detector, not the asynchronous multiplex video tracker.
"""
from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import math

import numpy as np

from .types import CameraFrame, DepthEstimate, Detection, TargetSpec, VisionFrame, intrinsics

SAM31_REPOSITORY = "facebook/sam3.1"
SAM31_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM31_FILENAME = "sam3.1_multiplex.pt"
DA3_MODEL_ID = "depth-anything/DA3-LARGE-1.1"
DA3_REVISION = "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"


class BackendUnavailable(RuntimeError):
    """A real model dependency/checkpoint is unavailable; no result was invented."""


def _dependency_report(names):
    result = {}
    for name in names:
        try:
            result[name] = find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            result[name] = False
    return result


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value):
    # Duck typing keeps torch out of dependency-free contract tests.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _checkpoint_error(model, error):
    name = type(error).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        return BackendUnavailable(
            f"{model} checkpoint access failed ({name}); use an approved authenticated "
            "Hugging Face account or an explicitly supplied local checkpoint. No model fallback."
        )
    return BackendUnavailable(f"{model} could not load ({name}); no model fallback was used")


def _sam31_detector_state(checkpoint):
    """Extract only the 3.1 detector keys; reject unlabelled or empty checkpoints."""
    if not isinstance(checkpoint, Mapping):
        raise BackendUnavailable("SAM 3.1 checkpoint must contain a state dictionary")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise BackendUnavailable("SAM 3.1 checkpoint.model must be a state dictionary")
    result = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise BackendUnavailable("SAM 3.1 checkpoint has a non-string state key")
        for prefix in ("detector.", "sam3_model."):
            if key.startswith(prefix):
                stripped = key[len(prefix):]
                if stripped in result:
                    raise BackendUnavailable("Duplicate SAM 3.1 detector state key")
                result[stripped] = value
                break
    if not result:
        raise BackendUnavailable("Checkpoint has no SAM 3.1 detector.* weights")
    return result


def _construct_sam31_detector():
    """Same detector construction as the official 3.1 multiplex builder.

    The older build_sam3_image_model uses a dual-neck backbone and must not load
    3.1's tri-neck weights with strict=False. We instantiate the official 3.1
    detector only, and require every detector tensor to match exactly.
    """
    from importlib.resources import files
    from sam3 import model_builder as builder
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
    from sam3.model.vl_combiner import SAM3VLBackboneTri

    bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
    neck = builder._create_multiplex_tri_backbone(
        compile_mode=None, use_fa3=False, use_rope_real=False)
    backbone = SAM3VLBackboneTri(
        scalp=0, visual=neck, text=builder._create_text_encoder(bpe_path))
    detector = Sam3MultiplexDetector(
        num_feature_levels=1, backbone=backbone,
        transformer=builder._create_sam3_transformer(use_fa3=False),
        segmentation_head=builder._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=builder._create_geometry_encoder(),
        use_early_fusion=True, use_dot_prod_scoring=True,
        dot_prod_scoring=builder._create_dot_product_scoring(),
        supervise_joint_box_scores=True, is_multiplex=True,
    )
    return detector


def _build_sam31_detector(torch, checkpoint_path, device):
    detector = _construct_sam31_detector()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    detector.load_state_dict(_sam31_detector_state(checkpoint), strict=True)
    return detector.to(device=device).eval()


class Sam31Segmenter:
    """Return every detector candidate above the explicit score threshold.

    A processor can be injected for contract tests. Its interface is the official
    Sam3Processor (set_image/reset_all_prompts/set_text_prompt). Image features
    and identical text results are reused only for the exact current RGB frame.
    """

    def __init__(self, checkpoint_path=None, device="cuda:0", confidence_threshold=0.5,
                 processor=None, cache_dir=None, local_files_only=False):
        threshold = float(confidence_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.device = str(device)
        self.confidence_threshold = threshold
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self._processor = processor
        self._injected = processor is not None
        self._torch = None
        self._cache_key = None
        self._image_states = {}
        self._prompt_results = {}
        self._checkpoint_sha = None

    def preflight(self):
        deps = _dependency_report(("torch", "torchvision", "sam3", "PIL", "huggingface_hub"))
        local = self.checkpoint_path is not None and self.checkpoint_path.is_file()
        return {"backend": "sam3.1_detector", "injected": self._injected,
                "dependencies": deps, "checkpoint_local": local,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "checkpoint_access": "local" if local else "not_checked_gated_repository",
                "model_id": SAM31_REPOSITORY, "revision": SAM31_REVISION,
                "ready_without_download": self._injected or (all(deps.values()) and local)}

    def provenance(self):
        return {"backend": "sam3.1_detector", "model_id": SAM31_REPOSITORY,
                "revision": SAM31_REVISION, "checkpoint_filename": SAM31_FILENAME,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
                "checkpoint_sha256": self._checkpoint_sha, "device": self.device,
                "precision": "injected" if self._injected else (
                    "float32_weights_cuda_bfloat16_autocast"
                    if self.device.split(":", 1)[0] == "cuda" else "float32"),
                "confidence_threshold": self.confidence_threshold,
                "video_tracking": False, "injected": self._injected}

    def _load(self):
        if self._processor is not None:
            return
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as error:
            raise BackendUnavailable("SAM 3.1 dependencies missing; run scripts/setup_foundation_vision.sh") from error
        if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
            raise BackendUnavailable(f"SAM 3.1 checkpoint does not exist: {self.checkpoint_path}")
        if self.checkpoint_path is None:
            try:
                self.checkpoint_path = Path(hf_hub_download(
                    repo_id=SAM31_REPOSITORY, filename=SAM31_FILENAME, revision=SAM31_REVISION,
                    cache_dir=self.cache_dir, local_files_only=self.local_files_only))
            except Exception as error:
                raise _checkpoint_error("SAM 3.1", error) from error
        try:
            detector = _build_sam31_detector(torch, self.checkpoint_path, self.device)
            self._processor = Sam3Processor(
                detector, device=self.device, confidence_threshold=self.confidence_threshold)
        except Exception as error:
            raise BackendUnavailable("SAM 3.1 detector construction/strict weight loading failed") from error
        self._torch = torch
        self._checkpoint_sha = _file_sha256(self.checkpoint_path)

    def segment(self, frame: VisionFrame, targets: Sequence[TargetSpec]):
        if not isinstance(frame, VisionFrame):
            raise TypeError("frame must be VisionFrame")
        targets = tuple(targets)
        if any(not isinstance(target, TargetSpec) for target in targets):
            raise TypeError("targets must contain TargetSpec")
        if len({target.key for target in targets}) != len(targets):
            raise ValueError("target keys must be unique")
        if not targets:
            return []
        self._load()
        from PIL import Image
        key = (frame.frame_id, frame.captured_at_s, tuple(
            (camera.name, camera.rgb.shape, sha256(camera.rgb.tobytes()).digest())
            for camera in frame.cameras))
        if key != self._cache_key:
            self._cache_key = key
            self._image_states.clear()
            self._prompt_results.clear()
        detections = []
        context = self._torch.inference_mode() if self._torch is not None else nullcontext()
        # The official image examples and multiplex predictor use BF16 autocast
        # across image encoding AND grounding. Some vision blocks already emit
        # BF16, so inference_mode alone leaves subsequent FP32 linear layers
        # incompatible. Scope AMP to this call instead of changing global state.
        autocast = (self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)
                    if self._torch is not None and self.device.split(":", 1)[0] == "cuda"
                    else nullcontext())
        with context, autocast:
            for camera in frame.cameras:
                if camera.name not in self._image_states:
                    # PIL avoids upstream's ndarray shape[-2:] HWC ambiguity.
                    self._image_states[camera.name] = self._processor.set_image(Image.fromarray(camera.rgb))
                state = self._image_states[camera.name]
                for target in targets:
                    prompt_key = (camera.name, target.text)
                    if prompt_key not in self._prompt_results:
                        self._processor.reset_all_prompts(state)
                        output = self._processor.set_text_prompt(state=state, prompt=target.text)
                        score_values = output["scores"]
                        # NumPy cannot represent torch.bfloat16; promote the
                        # returned probabilities only, preserving their values.
                        if hasattr(score_values, "float"):
                            score_values = score_values.float()
                        masks, scores = _numpy(output["masks"]), _numpy(score_values)
                        if masks.ndim == 4 and masks.shape[1] == 1:
                            masks = masks[:, 0]
                        if (masks.ndim != 3 or masks.shape[1:] != camera.rgb.shape[:2]
                                or scores.shape != (len(masks),) or masks.dtype != np.bool_
                                or not np.isfinite(scores).all()):
                            raise ValueError("SAM output must be original-resolution boolean masks and finite scores")
                        self._prompt_results[prompt_key] = (masks.copy(), scores.copy())
                    masks, scores = self._prompt_results[prompt_key]
                    detections.extend(Detection(camera.name, target.key, mask, float(score))
                                      for mask, score in zip(masks, scores)
                                      if score > self.confidence_threshold)
        return detections


def _bilinear_valid(values, u, v, *, positive=False):
    """Conservative ray sampling: no fill across invalid contributing pixels."""
    values = np.asarray(values, dtype=float)
    height, width = values.shape
    inside = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0)
              & (u <= width - 1) & (v <= height - 1))
    safe_u = np.clip(np.where(inside, u, 0), 0, width - 1)
    safe_v = np.clip(np.where(inside, v, 0), 0, height - 1)
    x0, y0 = np.floor(safe_u).astype(int), np.floor(safe_v).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = safe_u - x0, safe_v - y0
    result = np.zeros_like(safe_u)
    valid = inside.copy()
    for x, y, weight in ((x0, y0, (1-dx)*(1-dy)), (x1, y0, dx*(1-dy)),
                         (x0, y1, (1-dx)*dy), (x1, y1, dx*dy)):
        sample = values[y, x]
        good = np.isfinite(sample) & ((sample > 0) if positive else True)
        contributes = weight > 1e-12
        valid &= ~contributes | good
        result += np.where(good, sample, 0) * weight
    return np.where(valid, result, np.nan)


def depth_on_original_rays(depth, native_intrinsics, camera: CameraFrame, score_map=None):
    """Resample using K_native @ inv(K_original); preserve depth's z units.

    This also handles upstream centre-crops and its own resize convention.
    Unobserved border rays and invalid depths stay NaN. We do not reinterpret
    a resized raster using the original K without doing this ray mapping.
    """
    native_k = intrinsics(native_intrinsics)
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("native depth must be H x W")
    height, width = camera.rgb.shape[:2]
    v, u = np.indices((height, width), dtype=float)
    pixels = np.stack((u.ravel(), v.ravel(), np.ones(height*width)))
    native = native_k @ np.linalg.solve(camera.intrinsics, pixels)
    source_u = (native[0] / native[2]).reshape(height, width)
    source_v = (native[1] / native[2]).reshape(height, width)
    normalized_depth = _bilinear_valid(depth, source_u, source_v, positive=True)
    normalized_score = None
    if score_map is not None:
        score_map = np.asarray(score_map, dtype=float)
        if score_map.shape != depth.shape:
            raise ValueError("native score map must match depth")
        normalized_score = _bilinear_valid(score_map, source_u, source_v)
        normalized_score[~np.isfinite(normalized_depth)] = np.nan
    return normalized_depth, normalized_score


def _align_two_view_depth(prediction, extrinsics, processed_intrinsics):
    """Anchor two-view depth scale using baseline lengths, without fitting Sim3.

    Upstream Umeyama fits camera centres and rejects the necessarily rank-one
    covariance of exactly two views. Camera z-depth needs only a scale factor;
    each view is subsequently backprojected through its supplied K and E.
    """
    known = _numpy(extrinsics).astype(float)
    predicted = _numpy(prediction.extrinsics).astype(float)
    if known.shape not in ((2, 3, 4), (2, 4, 4)) or predicted.shape not in ((2, 3, 4), (2, 4, 4)):
        raise ValueError("two-view scale alignment requires two camera poses")
    def baseline(poses):
        centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
        return float(np.linalg.norm(centers[1] - centers[0]))
    known_baseline, predicted_baseline = baseline(known), baseline(predicted)
    if (not math.isfinite(known_baseline) or not math.isfinite(predicted_baseline)
            or min(known_baseline, predicted_baseline) <= 1e-6):
        raise ValueError("two-view depth scale is unavailable: degenerate camera baseline")
    scale = known_baseline / predicted_baseline
    known_relative_r = known[1, :3, :3] @ known[0, :3, :3].T
    predicted_relative_r = predicted[1, :3, :3] @ predicted[0, :3, :3].T
    relative_angle = math.acos(float(np.clip(
        (np.trace(known_relative_r @ predicted_relative_r.T) - 1) / 2, -1, 1)))
    def baseline_direction_in_camera_zero(poses):
        centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
        direction = poses[0, :3, :3] @ (centers[1] - centers[0])
        return direction / np.linalg.norm(direction)
    direction_angle = math.acos(float(np.clip(np.dot(
        baseline_direction_in_camera_zero(known), baseline_direction_in_camera_zero(predicted)), -1, 1)))
    prediction.depth = _numpy(prediction.depth) * scale
    prediction.intrinsics = _numpy(processed_intrinsics).copy()
    prediction.extrinsics = known[:, :3, :].copy()
    return prediction, {"method": "two_view_camera_baseline_ratio",
                        "known_baseline_m": known_baseline,
                        "predicted_baseline_model_units": predicted_baseline,
                        "depth_scale_to_metres": scale,
                        "predicted_relative_rotation_error_rad": relative_angle,
                        "predicted_baseline_direction_error_rad": direction_angle,
                        "predicted_orientation_used_for_backprojection": False,
                        "depth_errors_from_prediction_are_not_corrected": True}


def _calibrated_da3_class(base):
    class CalibratedDA3(base):
        def _preprocess_inputs(self, *args, **kwargs):
            start = perf_counter()
            result = super()._preprocess_inputs(*args, **kwargs)
            self.foundation_vision_stage_timings = {"preprocess_s": perf_counter() - start}
            return result

        def _run_model_forward(self, *args, **kwargs):
            start = perf_counter()
            result = super()._run_model_forward(*args, **kwargs)
            self.foundation_vision_stage_timings["network_forward_s"] = perf_counter() - start
            return result

        def _convert_to_prediction(self, *args, **kwargs):
            start = perf_counter()
            result = super()._convert_to_prediction(*args, **kwargs)
            self.foundation_vision_stage_timings["prediction_conversion_s"] = perf_counter() - start
            return result

        def _align_to_input_extrinsics_intrinsics(
                self, extrinsics, intrinsics, prediction, align_to_input_ext_scale=True, **kwargs):
            start = perf_counter()
            self.foundation_vision_raw = {
                "raw_depth_model_units": _numpy(prediction.depth).copy(),
                "raw_predicted_extrinsics": _numpy(prediction.extrinsics).copy(),
                "raw_predicted_intrinsics": _numpy(prediction.intrinsics).copy(),
                "processed_input_intrinsics": _numpy(intrinsics).copy(),
                "input_extrinsics": _numpy(extrinsics).copy(),
            }
            if getattr(prediction, "conf", None) is not None:
                self.foundation_vision_raw["raw_score_map"] = _numpy(prediction.conf).copy()
            if extrinsics is not None and len(extrinsics) == 2 and align_to_input_ext_scale:
                result, self.foundation_vision_alignment = _align_two_view_depth(
                    prediction, extrinsics, intrinsics)
            else:
                self.foundation_vision_alignment = {"method": "upstream_umeyama"}
                result = super()._align_to_input_extrinsics_intrinsics(
                    extrinsics, intrinsics, prediction, align_to_input_ext_scale, **kwargs)
            self.foundation_vision_stage_timings["metric_alignment_s"] = perf_counter() - start
            return result
    return CalibratedDA3


class DA3DepthEstimator:
    """Synchronous calibrated any-view depth; no temporal static-scene fusion."""

    def __init__(self, model_id=DA3_MODEL_ID, device="cuda:0", model=None, cache_dir=None,
                 local_files_only=False, process_res=504, revision=None):
        if type(process_res) is not int or process_res < 28 or process_res % 14:
            raise ValueError("process_res must be a positive multiple of 14, at least 28")
        self.model_id = str(model_id)
        self.device = str(device)
        self.cache_dir = cache_dir
        self.local_files_only = bool(local_files_only)
        self.process_res = process_res
        self.revision = revision if revision is not None else (DA3_REVISION if self.model_id == DA3_MODEL_ID else None)
        self._model = model
        self._injected = model is not None
        self._torch = None
        self._resolved_path = None
        self._checkpoint_sha = None
        self.last_timings = {}
        self._native_evidence = {}

    def preflight(self):
        deps = _dependency_report(("torch", "torchvision", "depth_anything_3", "huggingface_hub"))
        local = Path(self.model_id).is_dir()
        return {"backend": "depth_anything_3", "dependencies": deps,
                "model_id": self.model_id, "revision": self.revision, "injected": self._injected,
                "checkpoint_local": local,
                "ready_without_download": self._injected or (all(deps.values()) and local)}

    def provenance(self):
        return {"backend": "depth_anything_3", "model_id": self.model_id, "revision": self.revision,
                "checkpoint_path": self._resolved_path, "checkpoint_sha256": self._checkpoint_sha,
                "device": self.device, "precision": "injected" if self._injected else "upstream_autocast",
                "process_res": self.process_res, "pose_conditioned": True,
                "metric_scale_source": "provided_camera_extrinsics",
                "alignment": getattr(self._model, "foundation_vision_alignment", None),
                "raster": "original_rgb_via_native_intrinsics_ray_resampling",
                "temporal_fusion": False, "injected": self._injected}

    def native_evidence(self):
        """Copied current inference arrays, before original-raster resampling."""
        return {key: value.copy() for key, value in self._native_evidence.items()}

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:
            raise BackendUnavailable("DA3 dependencies missing; run scripts/setup_foundation_vision.sh") from error
        try:
            if Path(self.model_id).is_dir():
                path = self.model_id
            else:
                path = snapshot_download(
                    repo_id=self.model_id, revision=self.revision, cache_dir=self.cache_dir,
                    allow_patterns=["config.json", "model.safetensors"],
                    local_files_only=self.local_files_only)
            model = _calibrated_da3_class(DepthAnything3).from_pretrained(
                path, local_files_only=True).to(device=self.device).eval()
        except Exception as error:
            raise _checkpoint_error("DA3", error) from error
        self._model = model
        self._torch = torch
        self._resolved_path = str(path)
        self._checkpoint_sha = _file_sha256(Path(path) / "model.safetensors")

    def estimate(self, frame: VisionFrame):
        if not isinstance(frame, VisionFrame):
            raise TypeError("frame must be VisionFrame")
        centers = [-(camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3])
                   for camera in frame.cameras]
        if max(np.linalg.norm(center - centers[0]) for center in centers) <= 1e-6:
            raise ValueError("DA3 metric alignment requires a nonzero camera baseline")
        self._native_evidence = {}
        self.last_timings = {}
        started = perf_counter()
        self._load()
        self.last_timings = {"load_s": perf_counter() - started}
        expected_ext = np.stack([camera.world_to_camera for camera in frame.cameras])
        context = self._torch.inference_mode() if self._torch is not None else nullcontext()
        started = perf_counter()
        with context:
            prediction = self._model.inference(
                image=[camera.rgb.copy() for camera in frame.cameras],
                intrinsics=np.stack([camera.intrinsics for camera in frame.cameras]),
                extrinsics=expected_ext.copy(), align_to_input_ext_scale=True,
                process_res=self.process_res, process_res_method="upper_bound_resize",
                infer_gs=False, export_dir=None)
        self.last_timings["inference_s"] = perf_counter() - started
        self.last_timings.update(getattr(self._model, "foundation_vision_stage_timings", {}))
        started = perf_counter()
        depth = _numpy(prediction.depth)
        native_k = _numpy(prediction.intrinsics)
        native_ext = _numpy(prediction.extrinsics)
        count = len(frame.cameras)
        if depth.ndim != 3 or depth.shape[0] != count or native_k.shape != (count, 3, 3):
            raise ValueError("DA3 must return one depth and processed intrinsic matrix per camera")
        if native_ext.shape == (count, 4, 4):
            native_ext = native_ext[:, :3]
        if native_ext.shape != (count, 3, 4) or not np.allclose(native_ext, expected_ext[:, :3], atol=1e-5, rtol=1e-5):
            raise ValueError("DA3 did not preserve the supplied camera extrinsics after metric alignment")
        scores = getattr(prediction, "conf", None)
        if scores is not None:
            scores = _numpy(scores)
            if scores.shape != depth.shape:
                raise ValueError("DA3 confidence dimensions must match depth")
        self._native_evidence = dict(getattr(self._model, "foundation_vision_raw", {}))
        self._native_evidence.update({"aligned_depth_m": depth.copy(), "native_intrinsics": native_k.copy(),
                                      "aligned_extrinsics": native_ext.copy()})
        outputs = {}
        for index, camera in enumerate(frame.cameras):
            normalized, score = depth_on_original_rays(
                depth[index], native_k[index], camera, None if scores is None else scores[index])
            outputs[camera.name] = DepthEstimate(
                camera.name, normalized, camera.intrinsics, camera.world_to_camera, score,
                source="da3_pose_conditioned_depth_hypothesis")
        self.last_timings["normalize_rays_s"] = perf_counter() - started
        return outputs
