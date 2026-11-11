"""Focused CPU checks for V1 scheduling, dated regions and self identity."""
from threading import Event
from types import SimpleNamespace

import numpy as np

from gpt_field.foundation_vision_harness_V1 import scene
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_vision.geometry import build_geometry
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, VisionFrame
from gpt_field.foundation_harness.types import ObjectGeometry, SceneSnapshot


def frame(view, geometry, frame_id, captured_at_s):
    k = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    cameras = []
    for name, x in (("external", 0.), ("wrist", .1)):
        e = np.eye(4); e[0, 3] = -x
        cameras.append(CameraFrame(name, np.zeros((32, 32, 3), np.uint8), k, e, captured_at_s))
    return VisionFrame(frame_id, captured_at_s, tuple(cameras))


def view():
    return SimpleNamespace(state=np.array([0., 0., 1., 0., 0., 0., .04, -.04]),
                           observation=SimpleNamespace(geometry={}, joint_position=None))


def target(frame_id=0):
    return {"id": "region1", "query": "surface", "role": "region", "parent_id": None,
            "frame_id": frame_id, "region_pixels": {"external": [8, 8, 20, 20], "wrist": None}}


def engine(depth_fn=None):
    def depth(frame):
        return {c.name: DepthEstimate(c.name, np.full((32, 32), 2.+frame.frame_id),
                                     c.intrinsics, c.world_to_camera) for c in frame.cameras}
    return SimpleNamespace(config=EngineConfig(), segmenter=SimpleNamespace(segment=lambda f, t: []),
                           depth_estimator=SimpleNamespace(estimate=depth_fn or depth))


def cfg(**kwargs):
    return scene.SceneConfig(automatic_robot_queries=False, min_refresh_interval_s=0., **kwargs)


def test_slow_worker_coalesces_pending_frames_without_blocking_submit(monkeypatch):
    monkeypatch.setattr(scene, "frame_from_step_view", frame)
    entered, release = Event(), Event()
    calls = []
    base = engine().depth_estimator.estimate
    def slow(frame):
        calls.append(frame.frame_id)
        if frame.frame_id == 0:
            entered.set()
            assert release.wait(3.)
        return base(frame)
    service = scene.SceneService(engine(slow), config=cfg())
    try:
        service.submit(view(), [], force=True, frame_id=0)
        assert entered.wait(1.)
        service.submit(view(), [], force=True, frame_id=1)
        token = service.submit(view(), [], force=True, frame_id=2)
        assert service.latest() is None
        assert service.stats["coalesced"] == 1
        release.set()
        result = service.wait(token, 3.)
        assert result.frame_id == 2 and calls == [0, 2]
        assert result.evidence["request_token"] == token
    finally:
        release.set()
        service.close()


def test_wall_age_does_not_retrigger_cadence_and_resubmit_keeps_capture_time(monkeypatch):
    monkeypatch.setattr(scene, "frame_from_step_view", frame)
    service = scene.SceneService(engine(), config=cfg())
    original = view()
    try:
        first = service.submit(original, [], force=True, frame_id=0)
        before = service.wait(first, 3.)
        # A much older source does not itself request another inference.
        service._last_submitted_s -= 100.
        skipped = service.submit(view(), [], frame_id=1)
        assert skipped == first and service.stats["submitted"] == 1
        selected = service.submit(original, [target()], force=True, frame_id=0)
        after = service.wait(selected, 3.)
        assert after.captured_at_s == before.captured_at_s
        assert len(after.objects["region1"].points) > 0
    finally:
        service.close()


def test_roi_works_without_sam_and_stale_pixels_never_recrop_new_view(monkeypatch):
    monkeypatch.setattr(scene, "frame_from_step_view", frame)
    service = scene.SceneService(engine(), config=cfg())
    try:
        initial = service.wait(service.submit(view(), [target()], force=True, frame_id=0), 3.)
        original_points = initial.objects["region1"].points.copy()
        assert len(original_points) > 0
        later = service.wait(service.submit(view(), [target()], force=True, frame_id=1), 3.)
        # Blank images have no trackable material: retain the dated hypothesis,
        # never use the old pixel rectangle on frame1's now-different depth.
        assert later.objects["region1"].status == "unobserved"
        np.testing.assert_array_equal(later.objects["region1"].points, original_points)
        assert later.objects["region1"].evidence["observed_frame_id"] == 0
        summary = scene.scene_summary(later)
        assert "point_sigma_m" not in str(summary)
        assert "scene_points" not in str(summary)
    finally:
        service.close()


def test_self_filter_includes_robot_owner_but_preserves_object_foreground():
    f = frame(None, None, 0, 1.)
    depths = {c.name: DepthEstimate(c.name, np.ones((32, 32)), c.intrinsics,
                                   c.world_to_camera) for c in f.cameras}
    geometry = build_geometry(f, depths, {}, EngineConfig().geometry)
    source = geometry._views["external"]
    ids = np.array([15*32+15, 16*32+16, 14*32+14])
    points = source.points_world[ids]
    mask = np.ones((32, 32), bool)
    objects = {"robot": ObjectGeometry("robot", "robot", points[[0, 2]], np.zeros((2, 3)), "partial",
                                       {"external": mask}, {"role": "robot"}),
               "item": ObjectGeometry("item", "object", points[1:2], np.zeros((1, 3)), "partial",
                                      {}, {"role": "object"})}
    snapshot = SceneSnapshot(0, 1., 1., objects, points, geometry, f,
                             evidence={"scene_point_view_slices": {"external": [0, 1024]}})
    class Shape:
        def surface_points(self):
            return np.array([[x, y, z] for x in (-.2, .2) for y in (-.2, .2) for z in (1., 1.2)])
        def contains(self, pts, margin):
            return np.zeros(len(pts), bool)
        def ray_entry_distances(self, origin, pts):
            return np.linalg.norm(pts-origin, axis=1)+np.where(pts[:, 0] < -.01, .1, 0.)
    seen = []
    kinematics = SimpleNamespace(bind=lambda state, q: seen.append(q.copy()))
    service = scene.SceneService(engine(), kinematics=kinematics, config=cfg())
    service._robot = SimpleNamespace(shapes=lambda bound, q, fingers: [Shape()])
    try:
        keep, evidence = service._self_filter(snapshot, points, np.array(["robot", "item", "robot"]), ids,
                                               np.zeros(8), np.arange(7.))
        assert keep.tolist() == [False, True, True]
        assert evidence["removed"] == 1
        np.testing.assert_array_equal(seen[0], np.arange(7.))
    finally:
        service.close()


def test_cloud_budget_coarsens_without_erasing_thin_obstacle_evidence():
    # Isolated thin samples must remain covered even when the cloud is coarsened.
    points = np.array([[x, y, .2] for x in np.linspace(0., 1., 19)
                                  for y in np.linspace(0., 1., 17)])
    points = np.r_[points, [[.493, .481, .81]]]
    owners = np.array([""]*len(points))
    ids, radius, inverse, voxel = scene._voxel_sample(points, owners, .005, 16)
    assert len(ids) <= 16 and voxel > .005
    distances = np.linalg.norm(points-points[ids[inverse]], axis=1)
    assert np.all(distances <= radius[inverse]+1e-12)
    sigma = scene._voxel_sigma(np.full(len(points), .01), inverse, radius)
    assert np.all(sigma[inverse] >= distances+.01-1e-12)
