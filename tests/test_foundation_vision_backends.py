"""Backend contracts using injected predictors; these are not model-quality tests."""
from types import SimpleNamespace
from contextlib import contextmanager
import subprocess
import sys

import numpy as np
import pytest

from gpt_field.foundation_vision.backends import (
    BackendUnavailable, DA3DepthEstimator, Sam31Segmenter,
    _align_two_view_depth, _sam31_detector_state, depth_on_original_rays,
)
from gpt_field.foundation_vision.types import CameraFrame, TargetSpec, VisionFrame


def frame(frame_id=1, value=0):
    k = np.array([[8., 0., 3.5], [0., 8., 2.5], [0., 0., 1.]])
    e = np.eye(4)
    other = e.copy()
    other[0, 3] = -.2
    return VisionFrame(frame_id, float(frame_id), (
        CameraFrame("external", np.full((6, 8, 3), value, np.uint8), k, e, float(frame_id)),
        CameraFrame("wrist", np.full((6, 8, 3), value, np.uint8), k, other, float(frame_id)),
    ))


class Processor:
    def __init__(self):
        self.images = 0
        self.prompts = 0
        self.resets = 0

    def set_image(self, image):
        assert image.mode == "RGB"
        self.images += 1
        return {"shape": (image.height, image.width)}

    def reset_all_prompts(self, state):
        self.resets += 1
        state.pop("prompt", None)

    def set_text_prompt(self, *, state, prompt):
        assert "prompt" not in state
        self.prompts += 1
        state["prompt"] = prompt
        mask = np.zeros((3, 1, *state["shape"]), bool)
        mask[0, 0, 1:3, 1:3] = True
        mask[1, 0, 3:5, 5:7] = True
        mask[2, 0] = True
        return {"masks": mask, "scores": np.array([.9, .8, .2])}


class DepthModel:
    def __init__(self):
        self.calls = []

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            depth=np.full((2, 6, 8), 1.25), conf=np.full((2, 6, 8), 2.5),
            intrinsics=kwargs["intrinsics"].copy(), extrinsics=kwargs["extrinsics"][:, :3].copy())


def test_import_and_preflight_do_not_import_torch_or_model_packages():
    code = """
import sys
from gpt_field.foundation_vision.backends import Sam31Segmenter, DA3DepthEstimator
Sam31Segmenter().preflight()
DA3DepthEstimator().preflight()
assert 'torch' not in sys.modules
assert 'sam3' not in sys.modules
assert 'depth_anything_3' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sam_retains_all_candidates_and_original_mask_shape():
    backend = Sam31Segmenter(processor=Processor())
    results = backend.segment(frame(), [TargetSpec("target", "container")])
    assert len(results) == 4
    assert [result.score for result in results] == [.9, .8, .9, .8]
    assert {result.view for result in results} == {"external", "wrist"}
    assert all(result.mask.shape == (6, 8) for result in results)


def test_sam_shares_image_features_and_exact_prompt_results_only_within_frame():
    processor = Processor()
    backend = Sam31Segmenter(processor=processor)
    targets = [TargetSpec("a", "cup"), TargetSpec("b", "plate")]
    backend.segment(frame(), targets)
    assert (processor.images, processor.prompts, processor.resets) == (2, 4, 4)
    results = backend.segment(frame(), [TargetSpec("new_id", "cup")])
    assert (processor.images, processor.prompts) == (2, 4)
    assert all(result.prompt_key == "new_id" for result in results)
    backend.segment(frame(value=3), targets)
    assert (processor.images, processor.prompts) == (4, 8)
    backend.segment(frame(frame_id=2), targets)
    assert (processor.images, processor.prompts) == (6, 12)


def test_sam_rejects_wrong_raster_instead_of_guessing_mask_resize():
    processor = Processor()
    original = processor.set_text_prompt
    def invalid(**kwargs):
        result = original(**kwargs)
        result["masks"] = result["masks"][:, :, :3, :4]
        return result
    processor.set_text_prompt = invalid
    with pytest.raises(ValueError, match="original-resolution"):
        Sam31Segmenter(processor=processor).segment(frame(), [TargetSpec("a", "cup")])


def test_sam_does_not_load_for_empty_targets():
    backend = Sam31Segmenter(checkpoint_path="/does/not/exist")
    assert backend.segment(frame(), []) == []


@pytest.mark.parametrize("fail_at", [None, "image", "prompt"])
def test_sam_scopes_bfloat16_autocast_over_image_and_text_even_on_failure(fail_at):
    active = set()

    @contextmanager
    def inference_mode():
        active.add("inference")
        try:
            yield
        finally:
            active.remove("inference")

    @contextmanager
    def autocast(*, device_type, dtype):
        assert (device_type, dtype) == ("cuda", "bf16")
        active.add("autocast")
        try:
            yield
        finally:
            active.remove("autocast")

    class RequiresAutocast(Processor):
        def set_image(self, image):
            assert active == {"inference", "autocast"}
            if fail_at == "image":
                raise RuntimeError("image failure")
            return super().set_image(image)

        def set_text_prompt(self, **kwargs):
            assert active == {"inference", "autocast"}
            if fail_at == "prompt":
                raise RuntimeError("prompt failure")
            return super().set_text_prompt(**kwargs)

    backend = Sam31Segmenter(processor=RequiresAutocast(), device="cuda:2")
    backend._torch = SimpleNamespace(inference_mode=inference_mode, autocast=autocast,
                                     bfloat16="bf16")
    if fail_at:
        with pytest.raises(RuntimeError, match=f"{fail_at} failure"):
            backend.segment(frame(), [TargetSpec("target", "container")])
    else:
        assert len(backend.segment(frame(), [TargetSpec("target", "container")])) == 4
    assert active == set()


def test_sam_promotes_bfloat16_scores_before_numpy_without_changing_masks():
    class Bfloat16Scores:
        def float(self):
            return np.array([.875, .75, .25], dtype=np.float32)

        def numpy(self):
            raise TypeError("Got unsupported ScalarType BFloat16")

    class Bfloat16Processor(Processor):
        def set_text_prompt(self, **kwargs):
            result = super().set_text_prompt(**kwargs)
            result["scores"] = Bfloat16Scores()
            return result

    results = Sam31Segmenter(processor=Bfloat16Processor()).segment(
        frame(), [TargetSpec("target", "container")])
    assert [result.score for result in results] == [.875, .75, .875, .75]
    assert all(result.mask.dtype == np.bool_ and result.mask.shape == (6, 8) for result in results)


def test_sam_precision_provenance_distinguishes_cuda_amp_cpu_and_injected():
    assert Sam31Segmenter(device="cuda:2").provenance()["precision"] == (
        "float32_weights_cuda_bfloat16_autocast")
    assert Sam31Segmenter(device="cpu").provenance()["precision"] == "float32"
    assert Sam31Segmenter(processor=Processor()).provenance()["precision"] == "injected"


def test_checkpoint_prefix_extraction_is_exact_and_nonempty():
    state = {"model": {"detector.layer": 1, "tracker.layer": 2, "notdetector.layer": 3}}
    assert _sam31_detector_state(state) == {"layer": 1}
    assert _sam31_detector_state({"sam3_model.layer": 4}) == {"layer": 4}
    with pytest.raises(BackendUnavailable, match="no SAM"):
        _sam31_detector_state({"layer": 1})
    with pytest.raises(BackendUnavailable, match="Duplicate"):
        _sam31_detector_state({"detector.layer": 1, "sam3_model.layer": 2})


def test_da3_passes_synced_metric_calibration_without_mutating_inputs():
    observation = frame()
    model = DepthModel()
    backend = DA3DepthEstimator(model=model)
    results = backend.estimate(observation)
    kwargs = model.calls[0]
    assert kwargs["align_to_input_ext_scale"] is True
    assert kwargs["export_dir"] is None and kwargs["infer_gs"] is False
    assert kwargs["process_res_method"] == "upper_bound_resize"
    for camera in observation.cameras:
        result = results[camera.name]
        np.testing.assert_allclose(result.depth_m, 1.25)
        np.testing.assert_array_equal(result.intrinsics, camera.intrinsics)
        np.testing.assert_array_equal(result.world_to_camera, camera.world_to_camera)
    kwargs["image"][0][:] = 255
    kwargs["intrinsics"][0, 0, 0] = 999
    assert observation.cameras[0].rgb.max() == 0
    assert observation.cameras[0].intrinsics[0, 0] == 8


def test_depth_resampling_uses_returned_intrinsics_and_does_not_scale_metres():
    camera = frame().cameras[0]
    # Native rays are twice as dense; native depth varies linearly with pixel u.
    k = np.diag([2., 2., 1.]) @ camera.intrinsics
    native = np.broadcast_to(1 + np.arange(16)[None, :] / 10, (12, 16))
    result, score = depth_on_original_rays(native, k, camera, np.ones_like(native))
    np.testing.assert_allclose(result, np.broadcast_to(1 + np.arange(8)[None, :] / 5, (6, 8)))
    np.testing.assert_allclose(score, 1)


def test_depth_crop_leaves_unobserved_border_unknown_and_retains_invalid_pixels():
    camera = frame().cameras[0]
    k = camera.intrinsics.copy()
    k[:2, 2] -= [2, 1]
    native = np.full((4, 4), 2.)
    native[1, 1] = np.nan
    native[2, 2] = -1
    result, score = depth_on_original_rays(native, k, camera, np.ones_like(native))
    assert np.isnan(result[:, :2]).all() and np.isnan(result[:, 6:]).all()
    assert np.isnan(result[0]).all() and np.isnan(result[-1]).all()
    assert np.isnan(result[2, 3]) and np.isnan(result[3, 4])
    assert result[1, 2] == 2
    assert np.isnan(score[2, 3]) and np.isnan(score[3, 4])


def test_depth_fractional_rays_never_interpolate_across_invalid_evidence():
    camera = frame().cameras[0]
    k = camera.intrinsics.copy()
    k[0, 2] += .5
    native = np.full((6, 8), 2.)
    native[2, 2] = np.nan
    result, _ = depth_on_original_rays(native, k, camera)
    assert np.isnan(result[2, 1]) and np.isnan(result[2, 2])
    assert np.isnan(result[:, -1]).all()
    assert result[3, 1] == 2


def test_da3_uses_native_crop_K_and_rejects_changed_extrinsics():
    model = DepthModel()
    original = model.inference
    def bad(**kwargs):
        output = original(**kwargs)
        output.extrinsics[1, 0, 3] += .01
        return output
    model.inference = bad
    with pytest.raises(ValueError, match="preserve"):
        DA3DepthEstimator(model=model).estimate(frame())


def test_da3_rejects_zero_baseline_before_loading():
    observation = frame()
    first = observation.cameras[0]
    second = CameraFrame("same", first.rgb, first.intrinsics, first.world_to_camera, 1.)
    with pytest.raises(ValueError, match="baseline"):
        DA3DepthEstimator(model=DepthModel()).estimate(VisionFrame(1, 1., (first, second)))


def test_two_view_scale_uses_metric_baseline_without_rank_deficient_umeyama():
    observation = frame()
    known = np.stack([camera.world_to_camera for camera in observation.cameras])
    predicted = known.copy()
    predicted[:, :3, 3] *= 5
    result = SimpleNamespace(depth=np.full((2, 3, 4), 10.), extrinsics=predicted[:, :3])
    result, alignment = _align_two_view_depth(result, known, np.stack([c.intrinsics for c in observation.cameras]))
    np.testing.assert_allclose(result.depth, 2.)
    np.testing.assert_array_equal(result.extrinsics, known[:, :3])
    assert alignment["depth_scale_to_metres"] == pytest.approx(.2)
    assert alignment["known_baseline_m"] == pytest.approx(.2)


def test_two_view_scale_accounts_for_camera_rotation_and_rejects_zero_prediction():
    known = np.stack([camera.world_to_camera for camera in frame().cameras])
    known[1, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    known[1, :3, 3] = [0, 0, .2]
    predicted = known.copy()
    predicted[:, :3, 3] *= 3
    result = SimpleNamespace(depth=np.full((2, 3, 4), 6.), extrinsics=predicted)
    result, alignment = _align_two_view_depth(result, known, np.stack([np.eye(3)]*2))
    np.testing.assert_allclose(result.depth, 2.)
    assert alignment["known_baseline_m"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="degenerate"):
        _align_two_view_depth(SimpleNamespace(depth=np.ones((2, 2, 2)), extrinsics=np.stack([np.eye(4)]*2)),
                              known, np.stack([np.eye(3)]*2))


def test_provenance_discloses_injected_predictors_and_nontracking_scope():
    sam = Sam31Segmenter(processor=Processor()).provenance()
    da3 = DA3DepthEstimator(model=DepthModel()).provenance()
    assert sam["injected"] and sam["video_tracking"] is False
    assert da3["injected"] and da3["temporal_fusion"] is False
    assert da3["metric_scale_source"] == "provided_camera_extrinsics"
