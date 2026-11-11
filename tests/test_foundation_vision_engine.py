"""State-machine and fail-closed tests; no model, network or simulator calls."""
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.foundation_vision.engine import EngineConfig, VisionEngine
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, Detection, TargetSpec, VisionFrame


def frame(index=0, at=None):
    k = np.array([[20., 0, 15.5], [0, 20., 15.5], [0, 0, 1]])
    cameras = []
    for name, x in (("external", 0.), ("wrist", .2)):
        e = np.eye(4); e[0, 3] = -x
        cameras.append(CameraFrame(name, np.full((32, 32, 3), 120, np.uint8), k, e,
                                   index*.1 if at is None else at))
    return VisionFrame(index, index*.1 if at is None else at, tuple(cameras), np.eye(4))


class Segmenter:
    def __init__(self):
        self.count = 1
        self.fail = False

    def segment(self, frame, targets):
        if self.fail:
            raise RuntimeError("model service unavailable")
        return [Detection(c.name, t.key, np.ones(c.rgb.shape[:2], bool), .95)
                for t in targets for c in frame.cameras for _ in range(self.count)]


class Depth:
    def __init__(self):
        self.value = 1.

    def estimate(self, frame):
        return {c.name: DepthEstimate(c.name, np.full(c.rgb.shape[:2], self.value),
                                     c.intrinsics, c.world_to_camera)
                for c in frame.cameras}


def engine():
    return VisionEngine(Segmenter(), Depth(), EngineConfig(max_age_s=10.))


TARGETS = [TargetSpec("target", "generic rigid item")]


def test_continuous_identity_without_claiming_attachment():
    runtime = engine()
    a = runtime.process(frame(), TARGETS)
    b = runtime.process(frame(1), TARGETS)
    assert a.data["objects"][0]["status"] == "new_candidate"
    assert b.data["objects"][0]["status"] == "tracked_estimate"
    assert a.data["objects"][0]["track_id"] == b.data["objects"][0]["track_id"]
    assert b.data["objects"][0]["interaction"]["attachment"] == "unknown"
    assert b.data["objects"][0]["interaction"]["release"] == "unknown"
    assert b.data["objects"][0]["relative_to_hand_m"] is not None


def test_multiple_matching_objects_never_silently_choose_highest_score():
    runtime = engine()
    runtime.segmenter.count = 2
    result = runtime.process(frame(), TARGETS)
    obj = result.data["objects"][0]
    assert obj["status"] == "identity_ambiguous"
    assert obj["track_id"] is None
    assert any(e["type"] == "identity_ambiguous" for e in result.data["events"])


def test_loss_and_reappearance_do_not_reuse_identity():
    runtime = engine()
    original = runtime.process(frame(), TARGETS).data["objects"][0]["track_id"]
    runtime.segmenter.count = 0
    missing = runtime.process(frame(1), TARGETS)
    assert any(e["type"] == "track_lost" for e in missing.data["events"])
    runtime.segmenter.count = 1
    recovered = runtime.process(frame(2), TARGETS).data["objects"][0]
    assert recovered["track_id"] != original
    assert recovered["status"] == "reacquired_candidate"


def test_identity_expires_across_long_observation_gap():
    runtime = engine()
    first = runtime.process(frame(), TARGETS).data["objects"][0]["track_id"]
    result = runtime.process(frame(1, at=20.), TARGETS)
    assert result.data["objects"][0]["track_id"] != first
    assert result.data["objects"][0]["status"] == "reacquired_candidate"
    assert any(e["type"] == "track_expired" for e in result.data["events"])


def test_same_colour_and_nearby_geometry_cannot_override_disjoint_masks():
    runtime = engine()
    runtime.config = replace(runtime.config, max_identity_displacement_m=2.)
    def segment(f, targets):
        detections = []
        for c in f.cameras:
            mask = np.zeros((32,32), bool)
            left = (10 if f.frame_id == 0 else 18) - (4 if c.name == "wrist" else 0)
            mask[12:16, left:left+4] = True
            detections.append(Detection(c.name, "target", mask, .99))
        return detections
    runtime.segmenter.segment = segment
    assert runtime.process(frame(), TARGETS).data["objects"][0]["track_id"] is not None
    obj = runtime.process(frame(1), TARGETS).data["objects"][0]
    assert obj["status"] == "identity_ambiguous" and obj["track_id"] is None


def test_moving_every_camera_requires_stronger_correspondence():
    runtime = engine()
    runtime.process(frame(), TARGETS)
    moved = frame(1)
    cameras = []
    for c in moved.cameras:
        e = c.world_to_camera.copy(); e[1,3] += .01
        cameras.append(replace(c, world_to_camera=e))
    result = runtime.process(replace(moved, cameras=tuple(cameras)), TARGETS)
    assert result.data["objects"][0]["status"] == "identity_ambiguous"
    assert result.data["objects"][0]["track_id"] is None


def test_slow_association_is_included_in_freshness(monkeypatch):
    from gpt_field.foundation_vision import engine as engine_module
    runtime = engine()
    time = [0.]
    original = engine_module._appearance
    def slow(*args):
        time[0] += 11.
        return original(*args)
    monkeypatch.setattr(engine_module, "perf_counter", lambda: time[0])
    monkeypatch.setattr(engine_module, "_appearance", slow)
    result = runtime.process(frame(), TARGETS)
    assert result.data["status"] == "stale"
    assert result.data["objects"][0]["track_id"] is None
    assert not runtime._tracks["target"]["valid"]
    assert not any(e["type"] == "new_candidate" for e in result.data["events"])


def test_crossview_mixed_instances_are_explicitly_ambiguous():
    runtime = engine()
    def segment(f, targets):
        detections = []
        for c in f.cameras:
            mask = np.zeros((32,32), bool)
            if c.name == "external":
                mask[:, :16] = True
            else:
                mask[:, 16:] = True
            detections.append(Detection(c.name, "target", mask, .99))
        return detections
    runtime.segmenter.segment = segment
    result = runtime.process(frame(), TARGETS)
    obj = result.data["objects"][0]
    assert obj["status"] == "identity_ambiguous"
    assert obj["track_id"] is None
    assert any(e["type"] == "identity_ambiguous" for e in result.data["events"])


def test_stale_observation_cannot_continue_existing_track():
    runtime = engine()
    runtime.process(frame(), TARGETS)
    result = runtime.process(frame(1), TARGETS, now_s=20.)
    assert result.data["status"] == "stale"
    assert result.data["objects"][0]["track_id"] is None
    assert runtime.query_path([[0, 0, .5], [0, 0, .8]], .01)["status"] == "stale"


def test_default_query_clock_advances_after_inference(monkeypatch):
    runtime = engine(); runtime.process(frame(), TARGETS)
    anchor = runtime._clock_anchor[1]
    monkeypatch.setattr("gpt_field.foundation_vision.engine.perf_counter", lambda: anchor + 20.)
    assert runtime.query_path([[0, 0, .5], [0, 0, .8]], .01)["status"] == "stale"


def test_query_cannot_backdate_a_stale_observation():
    runtime = engine(); runtime.process(frame(), TARGETS, now_s=20.)
    with pytest.raises(ValueError, match="predate"):
        runtime.query_path([[0, 0, .5], [0, 0, .8]], .01, now_s=0.)


def test_path_query_time_is_included_in_freshness(monkeypatch):
    runtime = engine(); result = runtime.process(frame(), TARGETS)
    time = [runtime._clock_anchor[1] + 1.]
    monkeypatch.setattr("gpt_field.foundation_vision.engine.perf_counter", lambda: time[0])
    assess_path = result.geometry.assess_path
    def slow(*args):
        output = assess_path(*args)
        assert output["status"] == "estimated_clear"
        time[0] += 20.
        return output
    monkeypatch.setattr(result.geometry, "assess_path", slow)
    assert runtime.query_path([[0, 0, .5], [0, 0, .8]], .01)["status"] == "stale"


def test_invalid_depth_produces_unknown_and_never_attachment():
    runtime = engine(); runtime.depth_estimator.value = np.nan
    result = runtime.process(frame(), TARGETS)
    obj = result.data["objects"][0]
    assert obj["centroid_world_m"] is None
    assert obj["track_id"] is None
    assert any(e["type"] == "need_observation" for e in result.data["events"])


def test_backend_error_invalidates_previous_snapshot():
    runtime = engine(); runtime.process(frame(), TARGETS)
    runtime.segmenter.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.process(frame(1), TARGETS)
    with pytest.raises(RuntimeError, match="no current"):
        runtime.query_path([[0, 0, .2], [0, 0, .8]], .01)


def test_duplicate_frames_and_time_regression_rejected():
    runtime = engine(); runtime.process(frame(), TARGETS)
    with pytest.raises(ValueError, match="increasing"):
        runtime.process(frame(), TARGETS)
    with pytest.raises(ValueError, match="increasing"):
        runtime.process(frame(1, at=0), TARGETS)
    with pytest.raises(ValueError, match="predate"):
        runtime.process(frame(1), TARGETS, now_s=0)


def test_duplicate_target_names_rejected_before_models_run():
    runtime = engine()
    with pytest.raises(ValueError, match="unique"):
        runtime.process(frame(), TARGETS + TARGETS)


def test_wrong_output_coordinates_rejected():
    runtime = engine()
    runtime.segmenter.segment = lambda f, t: [Detection("external", "target", np.ones((16, 16), bool), .9)]
    with pytest.raises(ValueError, match="original camera raster"):
        runtime.process(frame(), TARGETS)


def test_feedback_is_detached_and_does_not_return_dense_geometry():
    result = engine().process(frame(), TARGETS)
    summary = result.summary()
    assert "geometry" not in summary
    summary["objects"][0]["interaction"]["attachment"] = "fake"
    assert result.summary()["objects"][0]["interaction"]["attachment"] == "unknown"
    detached = result.to_dict(); detached["objects"].clear()
    assert len(result.data["objects"]) == 1


def test_reset_clears_state_and_allocates_new_identity():
    runtime = engine()
    first = runtime.process(frame(), TARGETS).data["objects"][0]["track_id"]
    runtime.reset()
    assert runtime.latest is None
    second = runtime.process(frame(), TARGETS).data["objects"][0]["track_id"]
    assert first != second


@pytest.mark.parametrize("config", [{"max_age_s":0},{"min_region_pixels":False},
                                  {"max_appearance_distance":1.1},{"min_mask_iou":np.nan}])
def test_invalid_limits_rejected(config):
    with pytest.raises(ValueError):
        EngineConfig(**config)


def test_target_removal_does_not_erase_obstacle_cloud():
    runtime = engine(); runtime.process(frame(), TARGETS)
    result = runtime.process(frame(1), [])
    assert any(e["type"] == "target_removed" for e in result.data["events"])
    assert runtime.query_path([[0, 0, .8], [0, 0, 1.2]], .03)["status"] == "blocked"


def test_validated_inputs_are_detached_from_callers():
    raw = np.full((8, 8, 3), 1, np.uint8)
    k = np.array([[8.,0,3.5],[0,8.,3.5],[0,0,1]])
    camera = CameraFrame("external", raw, k, np.eye(4), 0)
    raw[:] = 0; k[0,0] = 2
    assert camera.rgb[0,0,0] == 1 and camera.intrinsics[0,0] == 8
    with pytest.raises(ValueError):
        camera.rgb[0,0,0] = 2


def test_unsynchronized_pair_rejected():
    f = frame()
    with pytest.raises(ValueError, match="timestamp"):
        VisionFrame(0,0,(f.cameras[0],replace(f.cameras[1],observed_at_s=.1)))
