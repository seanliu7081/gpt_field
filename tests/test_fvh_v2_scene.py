"""Focused CPU contracts for typed scene evidence and same-source inference."""
from types import SimpleNamespace

import numpy as np

from gpt_field.foundation_vision_harness_V2 import scene
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, VisionFrame


def _fixture():
    k = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    cameras = []
    for name, x in (("external", 0.), ("wrist", .1)):
        pose = np.eye(4)
        pose[0, 3] = -x
        cameras.append(CameraFrame(name, np.zeros((32, 32, 3), np.uint8), k, pose, 20.))
    frame = VisionFrame(7, 20., tuple(cameras))
    calls = {"depth": 0, "segment": 0}
    def depth(f):
        calls["depth"] += 1
        return {c.name: DepthEstimate(c.name, np.full((32, 32), 2.), c.intrinsics, c.world_to_camera)
                for c in f.cameras}
    def segment(f, targets):
        calls["segment"] += 1
        return []
    engine = SimpleNamespace(config=EngineConfig(), segmenter=SimpleNamespace(segment=segment),
                             depth_estimator=SimpleNamespace(estimate=depth))
    cfg = scene.SceneConfig(automatic_robot_queries=False, min_refresh_interval_s=0.)
    state = np.array([0., 0., 1., 0., 0., 0., .04, -.04])
    def job(targets=(), token=1):
        return SimpleNamespace(frame=frame, targets=targets, token=token, target_key=str(token), state=state, joints=None)
    return engine, cfg, job, calls


def _target(identity, role, box):
    return {"id": identity, "query": "surface", "role": role, "frame_id": 7,
            "region_pixels": {"external": box, "wrist": None}}


def test_self_identity_never_deletes_uncorroborated_foreground():
    owners = np.array(["arm", "arm", "arm", "cup", "", "", "arm"])
    kinds, reasons, clean = scene.classify_ray_evidence(
        owners, {"arm"}, owners == "arm", np.array([.003, -.04, -.04, 0., .012, -.08, np.nan]),
        np.array([2, 1, 2, 1, 2, 2, 1]), np.zeros(7), np.array([.005, .02, .005, .02, .02, .005, .02]), .006)
    assert list(kinds) == ["self", "unresolved", "external", "external", "unresolved", "external", "unresolved"]
    assert clean[2] == "" and clean[3] == "cup"
    assert reasons[2] == "corroborated_foreground"
    assert reasons[-1] == "robot_label_outside_model"


def test_a_small_foreground_depth_disagreement_remains_unresolved():
    kinds, _, _ = scene.classify_ray_evidence(["arm"], {"arm"}, [True], [-.015], [1], [0], [.02], .006)
    assert kinds.tolist() == ["unresolved"]


def test_same_source_selection_reuses_depth_and_same_query_segmentation():
    engine, cfg, job, calls = _fixture()
    backend = scene._FramePerception(engine, cfg)
    first = backend.process(job((_target("part", "region", [2, 2, 10, 10]),)))
    second = backend.process(job((_target("part", "region", [4, 4, 12, 12]),), 2))
    requested = []
    base_segment = engine.segmenter.segment
    def tracked_segment(frame, targets):
        requested.append([t.key for t in targets])
        return base_segment(frame, targets)
    engine.segmenter.segment = tracked_segment
    third = backend.process(job((_target("part", "region", [4, 4, 12, 12]),
                                 _target("other", "region", [4, 4, 12, 12])), 3))
    assert calls == {"depth": 1, "segment": 2}
    assert requested == [["other"]]
    assert first.geometry is second.geometry is third.geometry
    assert second.evidence["inference_cache"]["segmentation_hit"]
    assert first.captured_at_s == second.captured_at_s == 20.


def test_external_map_separates_unresolved_without_inflating_position_error(tmp_path):
    engine, cfg, job, calls = _fixture()
    service = scene.SceneService(engine, tmp_path, config=cfg)
    try:
        current = job((_target("arm", "robot", [1, 1, 14, 30]),
                       _target("object", "object", [18, 1, 30, 30])))
        snapshot = service._backend.process(current)
        service._prepare(snapshot, current)
        service._save(snapshot)
        evidence = snapshot.evidence
        assert len(evidence["unresolved_points"]) > 0
        assert not np.any(snapshot.scene_point_object_ids == "arm")
        assert np.any(snapshot.scene_point_object_ids == "object")
        assert np.max(evidence["unresolved_position_error_m"]) <= .02
        assert np.array_equal(evidence["point_sigma_m"], evidence["point_position_error_m"] + evidence["point_representation_radius_m"])
        assert "unresolved_points" not in scene.scene_summary(snapshot)
        data = np.load(next(tmp_path.glob("*.npz")))
        assert "unresolved_source_indices" in data and "camera_0_depth_m" in data
        assert len(data["point_source_indices"]) == len(snapshot.scene_points)
    finally:
        service.close()
