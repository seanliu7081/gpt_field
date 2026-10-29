"""uc7.1 camera semantics and actual observed-region solver, no sim/model calls."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_types import HF_SOURCES, HeightField, Meas, PartRecord
from gpt_field.agent_speedup_v2_user_check_spatial import (
    DIRECTION_RELATIONS, DIRECTION_SEMANTICS, build_view_frame, ground_region,
    install_spatial_spec, region_directions, validate_view_frame,
)


# Camera-only snapshots from recorded, already validated geometry. No task or
# evaluator information is passed into the production helper.
# runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6/trial/requests/000001-*.request.json
LIBERO_R = [[-1.7233905013069872e-6, -0.5287697435529835, 0.8487653140297038],
            [0.9999999999985034, -7.823149652530503e-7, 1.5430955956352577e-6],
            [-1.5194045527300304e-7, 0.848765314031093, 0.5287697435535403]]
# runs/robocasa_coffee_user_check_001/trial/initial_sensors.json
ROBOCASA_R = [[0.9792973845320813, 0.09726115808109848, -0.17752999683813428],
              [-0.20197362855829382, 0.5281292976892554, -0.8247945794494831],
              [0.013538316572630271, 0.8435755520704573, 0.5368398289372741]]


def geometry(rotation=LIBERO_R, *, position=(0.6065773716836134, 0.0, 0.96), robocasa=False):
    fovy = 60.0 if robocasa else 45.0
    return {"image": {"width": 512, "height": 512, "pixel_axes": "u_right_v_down",
                      "pixel_centers": "integer", "source_transform": "raw_mujoco_rgb_vertical_flip_no_crop"},
            "cameras": {"external": {
                "mujoco_name": "robot0_agentview_left" if robocasa else "agentview",
                "pose_source": "synchronized_robot_mounted_camera" if robocasa else "fixed_world_camera",
                "fovy_deg": fovy, "focal_length_px": [256 / math.tan(math.radians(fovy / 2))] * 2,
                "principal_point_px": [255.5, 255.5], "position_world_m": list(position),
                "rotation_world_from_camera": np.asarray(rotation).tolist()}}}


def camera_rotation(yaw, elevation):
    """Independent look-at basis: yaw is the horizontal camera-to-scene heading."""
    forward = np.array([math.cos(yaw) * math.cos(elevation),
                        math.sin(yaw) * math.cos(elevation), -math.sin(elevation)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, forward)
    return np.column_stack([right, up, -forward])


def test_recorded_libero_calibration_and_direction_contract():
    original = geometry()
    frame = build_view_frame(original)
    assert frame["available"] and frame["direction_semantics"] == DIRECTION_SEMANTICS
    assert frame["right_xy"] == pytest.approx([0, 1], abs=3e-6)
    assert frame["away_xy"] == pytest.approx([-1, 0], abs=3e-6)
    assert frame["roll_deg"] == pytest.approx(0, abs=2e-5)
    assert frame["external_camera"] == original["cameras"]["external"]
    assert frame["image"] == original["image"]
    contract = region_directions(frame, [{"id": "E", "relation": "right_of", "anchor": "C"}])
    assert contract["regions"]["E"]["direction_world_xy"] == frame["right_xy"]
    assert contract["regions"]["E"]["anchor"] == "C"
    # No image flip or mutable shared list can change the approved basis.
    original["cameras"]["external"]["rotation_world_from_camera"][0][0] = 9
    contract["frame"]["right_xy"][0] = 9
    contract["regions"]["E"]["direction_world_xy"][1] = 9
    assert validate_view_frame(frame) == frame


def test_recorded_robocasa_rotation_is_not_a_libero_axis_constant():
    frame = build_view_frame(geometry(ROBOCASA_R, position=(0.09, -1.3, 1.75), robocasa=True))
    assert frame["right_xy"] == pytest.approx([0.979, -0.202], abs=5e-4)
    assert frame["away_xy"] == pytest.approx([0.202, 0.979], abs=5e-4)
    assert frame["roll_deg"] == pytest.approx(0.919, abs=0.001)
    assert frame["external_camera"]["pose_source"] == "synchronized_robot_mounted_camera"
    assert validate_view_frame(frame) == frame


@pytest.mark.parametrize("yaw", [-2.1, -0.4, 0.0, 1.3, math.pi])
@pytest.mark.parametrize("elevation", [math.radians(8), math.radians(45), math.radians(88), math.pi / 2])
def test_viewer_basis_tracks_camera_yaw_and_elevation(yaw, elevation):
    rotation = camera_rotation(yaw, elevation)
    frame = build_view_frame(geometry(rotation, position=(1.7, -0.8, 1.3)))
    expected_right = [math.sin(yaw), -math.cos(yaw)]
    expected_away = [math.cos(yaw), math.sin(yaw)]
    assert frame["available"]
    assert frame["right_xy"] == pytest.approx(expected_right, abs=1e-14)
    assert frame["away_xy"] == pytest.approx(expected_away, abs=1e-14)
    assert frame["relation_directions"]["left_of"] == pytest.approx(-np.asarray(expected_right))
    assert frame["relation_directions"]["front_of"] == pytest.approx(-np.asarray(expected_away))
    # Translate the camera without turning it: one global reviewed basis remains.
    assert build_view_frame(geometry(rotation, position=(-0.5, 2.0, 3.0)))["right_xy"] == frame["right_xy"]


@pytest.mark.parametrize("roll", [-45, -15, 15, 45])
def test_rolled_camera_records_roll_and_preserves_centre_word_signs(roll):
    rad = math.radians(roll)
    rotation = camera_rotation(math.pi, math.radians(32))
    optical_roll = np.array([[math.cos(rad), -math.sin(rad), 0],
                             [math.sin(rad), math.cos(rad), 0], [0, 0, 1]])
    rotation = rotation @ optical_roll
    frame = build_view_frame(geometry(rotation))
    assert frame["roll_deg"] == pytest.approx(roll)
    # Word signs are asserted only at an optical-axis intersection on support,
    # not as an unsupported claim of global pixel alignment under roll.
    for relation, pixel_axis, sign in (("right_of", 0, 1), ("left_of", 0, -1),
                                       ("behind", 1, 1), ("front_of", 1, -1)):
        world_direction = np.r_[frame["relation_directions"][relation], 0.0]
        assert sign * float(rotation[:, pixel_axis] @ world_direction) > 0


def test_inverted_view_is_diagnostic_only_and_top_down_has_no_roll():
    inverted = camera_rotation(math.pi, math.radians(30)) @ np.diag([-1.0, -1.0, 1.0])
    frame = build_view_frame(geometry(inverted))
    assert frame["available"] and frame["inverted"]
    assert frame["right_xy"] == pytest.approx([0, -1], abs=1e-14)
    assert frame["away_xy"] == pytest.approx([1, 0], abs=1e-14)
    assert abs(frame["roll_deg"]) == pytest.approx(180)
    top_down = build_view_frame(geometry(np.diag([1.0, -1.0, -1.0])))
    assert top_down["available"] and top_down["roll_deg"] is None
    assert top_down["away_xy"] == [0.0, 1.0]


def test_missing_and_degenerate_frames_do_not_fall_back_to_world_axes():
    vertical_x = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    frame = build_view_frame(geometry(vertical_x))
    assert not frame["available"] and frame["reason"] == "camera_x_vertical"
    assert frame["calibration_sha256"] and frame["relation_directions"] == {}
    assert validate_view_frame(frame) == frame
    for missing in (None, {}, {"cameras": {}}, {"cameras": {"wrist": {}}}):
        absent = build_view_frame(missing)
        assert not absent["available"] and absent["reason"] == "no_external_calibration"
    missing_contract = region_directions(None, [{"id": "E", "relation": "right_of", "anchor": "C"}])
    assert not missing_contract["available"]
    assert missing_contract["regions"]["E"]["direction_world_xy"] is None
    assert validate_view_frame(None) == build_view_frame(None)


def test_frame_always_sent_without_regions_and_nondirectional_relations_unchanged():
    frame = build_view_frame(geometry())
    for regions in ([], [{"id": "A", "relation": "on", "anchor": None}],
                    [{"id": "A", "relation": "near", "anchor": "B"}]):
        compact = region_directions(frame, regions)
        assert compact["frame"]["right_xy"] == frame["right_xy"]
        assert compact["regions"] == {} and compact["available"]


@pytest.mark.parametrize("field,value", [("right_xy", [1, 0]), ("calibration_sha256", "0" * 64),
                                         ("direction_semantics", "world_axes_legacy"), ("roll_deg", 999)])
def test_serialized_frame_rejects_tampered_semantic_evidence(field, value):
    frame = build_view_frame(geometry())
    frame[field] = value
    with pytest.raises(ValueError):
        validate_view_frame(frame)


def test_malformed_camera_is_not_sliced_before_its_shape_is_known():
    with pytest.raises(ValueError, match="3 by 3"):
        build_view_frame(geometry([[1, 0], [0, 1]]))
    with pytest.raises(ValueError):
        build_view_frame(geometry([[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]]))


def region_context(relation="right_of", frame=True, bounds=None):
    hf = HeightField.empty((0.1, 0.0), half_extent_m=0.32)
    hf.z[:] = 0.44
    hf.sigma[:] = 0.001
    hf.known[:] = True
    hf.source[:] = HF_SOURCES.index("plane")
    hf.stamp[:] = 0
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    poly = np.column_stack([0.149 + 0.0732 * np.cos(theta), 0.0057 + 0.0732 * np.sin(theta)])
    # A flat measured anchor; its footprint still supplies directional extent.
    anchor = PartRecord("C", "plate: body", "free", frozenset({"ref"}),
                        xyz=np.array([0.149, 0.0057, 0.444]), sigma_h_m=0.001,
                        top_z=Meas(0.444, 0.001, "dense"), region_xy=poly,
                        status="registered", registration_step=0)
    record = PartRecord("E", "table: right of plate", "fixed", frozenset({"ref"}))
    scene = SimpleNamespace(hf=hf, parts={"C": anchor, "E": record}, planes=[{"z": Meas(0.44, 0.001, "plane")}])
    spec = {"id": "E", "support": "observed_plane", "anchor": "C", "relation": relation,
            "frame": "world", "bounds_xy_m": bounds}
    install_spatial_spec(scene, {"regions": [spec]})
    ctx = SimpleNamespace(scene=scene, step=4, holding=None)
    if frame is True:
        # Exact LIBERO heading: use no floating yaw trig at the boundary.
        rotation = [[0, -0.6, 0.8], [1, 0, 0], [0, 0.8, 0.6]]
        ctx.user_check_view_frame = build_view_frame(geometry(rotation))
    elif isinstance(frame, dict):
        ctx.user_check_view_frame = frame
    return ctx, poly


def region_mask(ctx):
    cells = ctx.scene.parts["E"].region_cells
    mask = np.zeros(ctx.scene.hf.shape, bool)
    mask[cells[:, 0], cells[:, 1]] = True
    return mask


@pytest.mark.parametrize("relation", DIRECTION_RELATIONS)
def test_actual_solver_changes_same_label_to_viewer_half_plane(relation):
    ctx, poly = region_context(relation)
    hf = ctx.scene.hf
    gy, gx = np.indices(hf.shape)
    x, y = hf.origin_xy[0] + gx * hf.cell_m, hf.origin_xy[1] + gy * hf.cell_m
    # Independent old-axis masks: the permutation is the intended semantic change.
    expected = {"right_of": y > poly[:, 1].max(), "left_of": y < poly[:, 1].min(),
                "behind": x < poly[:, 0].min(), "front_of": x > poly[:, 0].max()}[relation]
    result = ground_region(ctx, "E")
    assert result["ready"], result
    np.testing.assert_array_equal(region_mask(ctx), expected)
    assert result["direction_world_xy"] == ctx.user_check_view_frame["relation_directions"][relation]
    assert result["view_frame"] == region_directions(ctx.user_check_view_frame, [])["frame"]


def test_actual_solver_uses_oblique_robocasa_direction_and_ctx_survives_scene_rebuild():
    frame = build_view_frame(geometry(ROBOCASA_R, robocasa=True))
    ctx, poly = region_context(frame=frame)
    # Rebuilding scene must not replace the approved frame with camera/world axes.
    ctx.scene = deepcopy(ctx.scene)
    result = ground_region(ctx, "E")
    assert result["ready"]
    points = ctx.scene.hf.origin_xy + ctx.scene.parts["E"].region_cells[:, ::-1] * ctx.scene.hf.cell_m
    direction = np.asarray(frame["right_xy"])
    assert np.all(points @ direction > np.max(poly @ direction))
    # A world +x test would include a below-oblique-boundary cell; ensure it does not.
    point = np.array([0.24, 0.20])
    assert point[0] > poly[:, 0].max()
    assert not region_mask(ctx)[ctx.scene.hf.index(point)]


def test_actual_solver_without_frame_invalidates_previous_directional_geometry():
    ctx, _ = region_context()
    assert ground_region(ctx, "E")["ready"]
    del ctx.user_check_view_frame
    result = ground_region(ctx, "E")
    assert not result["ready"] and result["reason"] == "view_frame_unavailable"
    assert ctx.scene.parts["E"].region_cells is None
    assert ctx.scene.parts["E"].status == "seeded"


def test_observed_region_keeps_bounds_unknown_cells_and_obstacles():
    ctx, _ = region_context(bounds=[[-0.1, 0.1], [0.08, 0.2]])
    hf = ctx.scene.hf
    unknown = hf.index((0.1, 0.12))
    obstacle_fill = hf.index((0.11, 0.12))
    hf.known[unknown] = False
    hf.source[obstacle_fill] = HF_SOURCES.index("obstacle")
    obstacle_poly = np.array([[0.15, 0.14], [0.18, 0.14], [0.18, 0.17], [0.15, 0.17]])
    ctx.scene.parts["B"] = PartRecord("B", "mug: body", "free", frozenset(),
        xyz=np.array([0.16, 0.15, 0.55]), sigma_h_m=0.001, top_z=Meas(0.55, 0.001, "dense"),
        region_xy=obstacle_poly, status="registered", registration_step=0)
    result = ground_region(ctx, "E")
    assert result["ready"] and result["excluded_objects"] == ["B"]
    mask = region_mask(ctx)
    assert not mask[unknown] and not mask[obstacle_fill]
    assert not mask[hf.index((0.16, 0.15))]
    points = hf.origin_xy + ctx.scene.parts["E"].region_cells[:, ::-1] * hf.cell_m
    assert points[:, 0].min() >= 0.049 and points[:, 0].max() <= 0.249
    assert points[:, 1].min() >= 0.0857 and points[:, 1].max() <= 0.2057


def test_on_region_does_not_require_direction_calibration():
    ctx, _ = region_context("on", frame=False)
    assert ground_region(ctx, "E")["ready"]
