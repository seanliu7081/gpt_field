"""Offline appearance evidence only; synthetic images are not policy observations."""
from types import SimpleNamespace

import numpy as np

from gpt_field.agent_speedup_v3_visual_watch import VisualGeometryWatch


def rgb():
    y, x = np.indices((128, 128))
    checker = np.where(((x // 4) + (y // 4)) % 2, 185, 65).astype(np.uint8)
    return np.repeat(checker[:, :, None], 3, axis=2)


class View:
    def __init__(self, image=None, geometry=True):
        self.image = rgb() if image is None else image
        self.image_reads = 0
        self.observation = SimpleNamespace(geometry={
            "image": {"width": 128, "height": 128},
            "cameras": {"external": {"pose_source": "fixed_world_camera",
                "position_world_m": [0, 0, 1], "rotation_world_from_camera": np.eye(3).tolist(),
                "focal_length_px": [100., 100.], "principal_point_px": [63.5, 63.5]}}} if geometry else None)

    @property
    def external_rgb_upright(self):
        self.image_reads += 1
        if isinstance(self.image, Exception):
            raise self.image
        return self.image


def box(center=(0, 0, 0), half=(.20, .20, .01), uncertainty=0.):
    return {"center": np.array(center, float), "rotation": np.eye(3), "half": np.array(half, float),
            "uncertainty": uncertainty}


def test_static_texture_has_no_timer_and_never_reanchors_or_certifies_geometry():
    watch, boxes = VisualGeometryWatch(), [box()]
    first = watch.capture("surface", boxes, View(), 3)
    assert first["status"] == "reference"
    for step in (80, 10000):
        result = watch.assess("surface", boxes, View(), step)
        assert result["status"] == "consistent"
        assert result["reference_step"] == 3
        assert result["geometry_verified"] is False
        assert result["free_space_verified"] is False
        assert result["contact_verified"] is False
    watch.forget(["surface"])
    assert watch.assess("surface", boxes, View(), 10001)["reason"] == "no_reference"


def test_local_new_appearance_in_free_region_is_changed_without_obstacle_claim():
    watch, boxes = VisualGeometryWatch(), [box()]
    watch.capture("free_region", boxes, View(), 0)
    image = rgb()
    image[50:76, 50:76] = [220, 20, 20]
    changed = View(image)
    evidence = watch.assess("free_region", boxes, changed, 20)
    assert evidence["status"] == "changed"
    assert evidence["changed_pixel_fraction"] > .18
    assert evidence["unmodeled_obstacle_detection_complete"] is False
    # A change report itself cannot silently install a matching new baseline.
    assert watch.assess("free_region", boxes, changed, 21)["status"] == "changed"
    assert watch.capture("free_region", boxes, changed, 22)["status"] == "reference"
    assert watch.assess("free_region", boxes, changed, 23)["reference_step"] == 22


def test_current_and_reference_tool_payload_masks_suppress_expected_occlusion():
    watch, boxes = VisualGeometryWatch(), [box(half=(.3, .3, .01))]
    # Separate narrow projections cover the changed hand at both endpoints.
    old_hand = [box(center=(-.20, 0, .2), half=(.045, .08, .025))]
    new_hand = [box(center=(.20, 0, .2), half=(.045, .08, .025))]
    before, after = rgb(), rgb()
    before[57:70, 36:43] = [240, 30, 40]
    after[57:70, 84:91] = [20, 240, 40]
    assert watch.capture("table", boxes, View(before), 0, occluder_boxes=old_hand)["status"] == "reference"
    result = watch.assess("table", boxes, View(after), 1, occluder_boxes=new_hand)
    assert result["status"] == "consistent"
    covering_hand = [box(center=(0, 0, .2), half=(.30, .30, .02))]
    blocked = watch.assess("table", boxes, View(after), 2, occluder_boxes=covering_hand)
    assert blocked["status"] == "unverified"
    assert blocked["reason"] == "expected_self_occlusion" and blocked["self_occluded"] is True


def test_low_texture_and_unreadable_or_missing_calibration_stay_unverified():
    watch, boxes = VisualGeometryWatch(), [box()]
    flat = View(np.full((128, 128, 3), 120, np.uint8))
    assert watch.capture("flat", boxes, flat, 0)["reason"] == "low_texture"
    flat_result = watch.assess("flat", boxes, flat, 20)
    assert flat_result["status"] == "unverified" and flat_result["self_occluded"] is False
    cases = [(View(geometry=False), "missing_calibration"),
             (View(OSError("image missing")), "unreadable_image"),
             (View(np.zeros((20, 20, 3), np.uint8)), "image_calibration_mismatch")]
    for view, reason in cases:
        result = watch.capture("missing", boxes, view, 0)
        assert result["status"] == "unverified" and result["reason"] == reason
    # An explicit new estimate with unreadable images discards the old reference.
    watch.capture("valid", boxes, View(), 0)
    watch.capture("valid", boxes, View(geometry=False), 1)
    assert watch.assess("valid", boxes, View(), 2)["reason"] == "no_reference"


def test_camera_motion_missing_calibration_and_world_geometry_change_are_distinct():
    watch, boxes = VisualGeometryWatch(), [box()]
    watch.capture("target", boxes, View(), 0)
    moved = View()
    moved.observation.geometry["cameras"]["external"]["position_world_m"][0] += .01
    assert watch.assess("target", boxes, moved, 1)["reason"] == "camera_moved"
    wrist = View()
    wrist.observation.geometry["cameras"]["external"]["pose_source"] = "observed_robot_pose_fixed_mount"
    assert watch.assess("target", boxes, wrist, 1)["reason"] == "camera_not_fixed"
    changed_boxes = [box(center=(.01, 0, 0))]
    assert watch.assess("target", changed_boxes, View(), 1)["reason"] == "geometry_changed_since_reference"
    # Distinguish broad illumination/camera/scene changes from a localized event.
    global_change = 255 - rgb()
    assert watch.assess("target", boxes, View(global_change), 1)["reason"] == "widespread_scene_or_camera_change"


def test_uncertainty_padding_does_not_change_nominal_visual_shape_signature():
    watch = VisualGeometryWatch()
    assert watch.capture("target", [box()], View(), 3)["status"] == "reference"
    padded = watch.assess("target", [box(uncertainty=.025)], View(), 8)
    assert padded["status"] == "consistent"
    assert padded["reference_step"] == 3
    assert padded["signature_policy"] == "nominal_pose_and_shape_excludes_risk_padding"
    assert not padded["geometry_verified"] and not padded["free_space_verified"]
    changed_shape = watch.assess("target", [box(half=(.225, .20, .01))], View(), 9)
    assert changed_shape["reason"] == "geometry_changed_since_reference"


def test_decode_is_cached_per_view_across_entities_and_results_are_small_json_data():
    watch, view = VisualGeometryWatch(), View()
    boxes = [box()]
    for index in range(12):
        watch.capture(str(index), boxes, view, 0)
    assert view.image_reads == 1
    current = View()
    for index in range(12):
        result = watch.assess(str(index), boxes, current, 1)
        assert result["status"] == "consistent"
        assert not any(isinstance(value, np.ndarray) for value in result.values())
    assert current.image_reads == 1
    assert len(watch._difference_cache) == 1


def test_offscreen_clipping_and_camera_plane_projection_fail_closed():
    watch = VisualGeometryWatch()
    assert watch.capture("outside", [box(center=(4, 0, 0))], View(), 0)["status"] == "unverified"
    assert watch.capture("partial", [box(center=(.7, 0, 0))], View(), 0)["reason"] == "insufficient_projection_visibility"
    assert watch.capture("camera", [box(center=(0, 0, 1), half=(.1, .1, .1))], View(), 0)["reason"] == "projection_crosses_camera_plane"
    # One invalid camera value must never be interpreted as an empty projection.
    invalid = View()
    invalid.observation.geometry["cameras"]["external"]["focal_length_px"] = [float("nan"), 1]
    assert watch.capture("invalid", [box()], invalid, 0)["status"] == "unverified"


def test_reference_occlusion_is_not_forgotten_when_tool_moves_away():
    watch, boxes = VisualGeometryWatch(), [box()]
    cover = [box(center=(0, 0, .2), half=(.25, .25, .01))]
    assert watch.capture("hidden", boxes, View(), 0, occluder_boxes=cover)["self_occluded"] is True
    result = watch.assess("hidden", boxes, View(), 1)
    assert result["reason"] == "expected_self_occlusion"
    assert result["status"] == "unverified"


def test_projection_cache_reuses_masks_but_checks_new_rgb():
    watch, boxes = VisualGeometryWatch(), [box()]
    watch.capture("surface", boxes, View(), 0)
    projections = watch.compute_counts["projections"]
    assert watch.assess("surface", boxes, View(), 1)["status"] == "consistent"
    assert watch.compute_counts["projections"] == projections
    image = rgb(); image[50:76, 50:76] = [220, 20, 20]
    changed = watch.assess("surface", boxes, View(image), 2)
    assert changed["status"] == "changed"
    assert changed["reference_step"] == 0
    assert not changed["observation_age_refreshed"]
