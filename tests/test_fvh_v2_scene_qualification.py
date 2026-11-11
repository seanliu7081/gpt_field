"""Regressions for contradictory obstacle promotion and source qualification."""
from threading import Condition
from types import SimpleNamespace

import numpy as np

from gpt_field.foundation_vision.geometry import GeometryConfig, build_geometry
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, VisionFrame
from gpt_field.foundation_vision_harness_V2 import scene


def _geometry(depth):
    height, width = depth.shape
    k = np.array([[100., 0., (width-1)/2], [0., 100., (height-1)/2], [0., 0., 1.]])
    cameras = tuple(CameraFrame(name, np.zeros((height, width, 3), np.uint8), k, np.eye(4), 10.)
                    for name in ("view", "coincident"))
    frame = VisionFrame(1, 10., cameras)
    # A second calibrated camera satisfies the frame contract, but its equal
    # centre supplies no independent geometric corroboration.
    estimates = {camera.name: DepthEstimate(camera.name, depth, k, np.eye(4)) for camera in cameras}
    return frame, build_geometry(frame, estimates, {})


def test_crossview_conflict_cannot_become_external_or_unresolved_even_with_name():
    kinds, reasons, _ = scene.classify_ray_evidence(
        ["", "cup", "arm", "arm", ""], {"arm"}, [False, False, True, True, False],
        [np.nan, np.nan, .001, .001, np.nan], [1, 1, 2, 1, 1], [1, 1, 1, 0, 0],
        [.02]*5, .006, ["", "", "", "single_view_image_border", ""])
    assert kinds.tolist() == ["disputed", "disputed", "disputed", "self", "external"]
    assert reasons[:3].tolist() == ["crossview_depth_conflict"]*3


def test_single_view_boundary_is_disputed_but_clean_interior_and_supported_edge_remain():
    depth = np.ones((16, 16))
    depth[:, 8:] = 1.3
    _, geometry = _geometry(depth)
    view = geometry._views["view"]
    # Independent agreement supports one edge sample; no all-scene two-view gate.
    supported = np.flatnonzero((view.rows == 8) & (view.cols == 7))[0]
    view.support[supported] = 2
    issues, qualified = scene._depth_measurement_issues(view, scene.SceneConfig(), geometry.config)
    by_pixel = dict(zip(zip(view.rows, view.cols), issues))
    assert by_pixel[8, 0] == "single_view_image_border"
    assert by_pixel[8, 8] == "single_view_depth_discontinuity"
    assert by_pixel[8, 4] == "" and qualified[8, 4]
    assert by_pixel[8, 7] == "" and qualified[8, 7]


def test_same_voxel_conflicting_and_reliable_rays_keep_distinct_provenance(monkeypatch):
    points = np.array([[0., 0., 1.], [.0001, 0., 1.], [.0002, 0., 1.]])
    frame, geometry = _geometry(np.ones((16, 16)))
    geometry = SimpleNamespace(config=GeometryConfig(), _views={
        "view": SimpleNamespace(support=np.array([1, 2, 1]), contradictions=np.array([1, 0, 1]))})
    snapshot = SimpleNamespace(scene_points=points, scene_point_object_ids=np.array(["cup"]*3),
        geometry=geometry, objects={}, frame=frame, frame_id=1, captured_at_s=10., evidence={})
    service = object.__new__(scene.SceneService)
    service.config = scene.SceneConfig()
    service.engine = SimpleNamespace(config=SimpleNamespace(geometry=GeometryConfig()))
    service._cv, service._invalidated, service._objects = Condition(), set(), {}
    service._unlabelled_memory = None
    service._ray_evidence = lambda *args: (np.zeros(len(args[1]), bool), np.full(len(args[1]), np.nan), {})
    monkeypatch.setattr(scene, "_measurement_evidence", lambda *args: (
        np.array(["crossview_depth_conflict", "", "crossview_depth_conflict"]),
        np.zeros(3, bool), np.zeros(3, int), {}))
    job = SimpleNamespace(state=np.zeros(8), joints=None, frame=frame, token=1, target_key="test")
    service._prepare(snapshot, job)
    assert snapshot.evidence["point_source_indices"].tolist() == [1]
    assert snapshot.evidence["disputed_source_indices"].tolist() == [0]
    assert snapshot.evidence["disputed_source_group_indices"].tolist() == [0, 2]
    assert snapshot.evidence["disputed_source_group_offsets"].tolist() == [0, 2]
    assert len(snapshot.evidence["unresolved_points"]) == 0
    assert snapshot.evidence["unknown_space_is_free"] is False
    assert snapshot.evidence["measurement_qualification"]["disputed_is_occupancy"] is False


def test_memory_clearing_requires_a_qualified_ray_and_distinguishes_occlusion():
    _, geometry = _geometry(np.full((16, 16), 2.))
    points = np.array([[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]])
    trusted = {name: np.ones((16, 16), bool) for name in geometry._views}
    disputed = {name: np.zeros((16, 16), bool) for name in geometry._views}
    assert scene._memory_keep(points, geometry, trusted, replace_surface=False).tolist() == [False, True, True]
    assert scene._memory_keep(points, geometry, trusted, replace_surface=True).tolist() == [False, False, True]
    assert scene._memory_keep(points, geometry, disputed, replace_surface=True).tolist() == [True, True, True]


def test_object_extent_cannot_reintroduce_rejected_source_measurements():
    _, geometry = _geometry(np.ones((16, 16)))
    view = geometry._views["view"]
    indices = [np.flatnonzero((view.rows == 8) & (view.cols == col))[0] for col in (0, 8)]
    obj = SimpleNamespace(points=view.points_world[indices], evidence={"point_support_ranges": [
        {"start": 0, "end": 2, "view": "view", "support": "single_view_clean_patch"}]})
    _, qualified = scene._depth_measurement_issues(view, scene.SceneConfig(), geometry.config)
    assert scene._qualified_object_points(obj, geometry, {name: qualified for name in geometry._views}).tolist() == [False, True]
