"""Contact reference geometry only; real camera maths, no models or simulator."""
from copy import deepcopy

import numpy as np

from gpt_field.foundation_harness.types import ObjectGeometry, SceneSnapshot
from gpt_field.foundation_vision.geometry import build_geometry, GeometryConfig
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, VisionFrame
from gpt_field.foundation_vision_harness_V2.grounding import ground_candidate
from gpt_field.robodojo_direct_sim import quaternion_matrix


def scene(frame_id=0, z=1., *, pixels=32, depth_pixels=32):
    K = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    camera_pose = np.eye(4)
    second_pose = np.eye(4); second_pose[0, 3] = -.01
    rgb = np.zeros((pixels, pixels, 3), np.uint8)
    cameras = tuple(CameraFrame(name, rgb, K, pose, 10.+frame_id)
                    for name, pose in (("external", camera_pose), ("wrist", second_pose)))
    frame = VisionFrame(frame_id, 10.+frame_id, cameras)
    scale = depth_pixels/pixels
    affine = np.array([[scale, 0., (scale-1)/2], [0., scale, (scale-1)/2], [0., 0., 1.]])
    depths = {c.name: DepthEstimate(c.name, np.full((depth_pixels, depth_pixels), z),
                                  affine@K, c.world_to_camera) for c in cameras}
    geometry = build_geometry(frame, depths, {}, GeometryConfig())
    mask = np.zeros((pixels, pixels), bool); mask[5:27, 5:27] = True
    points = geometry._views['external'].points_world
    obj = ObjectGeometry("contact", "handle surface", points.copy(), np.zeros_like(points), "supported",
        {name: mask.copy() for name in depths}, {"role": "region", "parent_id": "body",
                                               "observed_frame_id": frame_id, "position_error_m": .005})
    return SceneSnapshot(frame_id, 10.+frame_id, 10.1+frame_id, {"contact": obj}, points,
                         geometry, frame)


def inputs(*, xyz=(.025, .005, 1.), offset=(0., 0., 0.), standoff=0., quat=(1., 0., 0., 0.)):
    candidate = {"skill": "move_to_pose", "xyz_m": None if xyz is None else list(xyz),
        "quaternion_wxyz": list(quat), "gripper": "keep", "profile": "contact", "max_steps": 80}
    segment = {"id": "contact", "object": "body", "anchor": "contact", "contact_mode": "grasp",
               "grounding": {"region": "contact", "view": "external", "pixel": [16., 16.],
                             "offset_hand_m": list(offset), "standoff_m": standoff,
                             "reference": "tool", "frame_id": 0}}
    return candidate, segment


def test_stationary_contact_corrects_initial_2cm_error_and_keeps_orientation():
    before, current = scene(), scene(1)
    q = np.array([np.cos(.3), 0., 0., np.sin(.3)])
    candidate, segment = inputs(quat=q)
    result = ground_candidate(candidate, segment, before, current)
    assert result['status'] == 'grounded'
    np.testing.assert_allclose(result['candidate']['xyz_m'], [.005, .005, 1.])
    np.testing.assert_array_equal(result['candidate']['quaternion_wxyz'], q)
    assert abs(result['evidence']['correction_norm_m']-.02) < 1e-9
    assert result['evidence']['correspondence']['method'] == 'fixed_camera_unchanged_local_region'
    assert not result['evidence']['whole_object_centroid_used']


def test_measured_target_preserves_rotated_contact_offset_and_standoff_with_resized_depth():
    source = scene(depth_pixels=16)
    q = np.array([0., 1., 0., 0.])
    offset = np.array([.01, 0., .0036])
    desired = np.array([.005, .005, 1.])-quaternion_matrix(q)@offset-quaternion_matrix(q)[:, 2]*.03
    candidate, segment = inputs(xyz=desired+[.02, 0., 0.], offset=offset, standoff=.03, quat=q)
    result = ground_candidate(candidate, segment, source, source)
    assert result['status'] == 'grounded'
    np.testing.assert_allclose(result['candidate']['xyz_m'], desired)
    np.testing.assert_array_equal(result['candidate']['quaternion_wxyz'], q)
    assert result['evidence']['standoff_m'] == .03


def test_explicit_pixel_can_provide_absolute_position_without_model_metric_guess():
    source = scene()
    candidate, segment = inputs(xyz=None)
    result = ground_candidate(candidate, segment, source, source)
    assert result['status'] == 'grounded'
    np.testing.assert_allclose(result['candidate']['xyz_m'], [.005, .005, 1.])
    assert result['evidence']['position_source'] == 'material_point_projection'
    assert result['evidence']['correction_norm_m'] is None


def test_material_pixel_moves_with_local_appearance_instead_of_body_centroid():
    import cv2
    source, current = scene(), scene(1)
    rng = np.random.default_rng(19)
    rgb = np.repeat(rng.integers(0, 256, (32, 32, 1), dtype=np.uint8), 3, axis=2)
    shifted = cv2.warpAffine(rgb, np.array([[1., 0., 2.], [0., 1., 0.]]), (32, 32))
    def update_rgb(snapshot, image):
        cameras = tuple(CameraFrame(c.name, image, c.intrinsics, c.world_to_camera, c.observed_at_s)
                        for c in snapshot.frame.cameras)
        snapshot.frame = VisionFrame(snapshot.frame_id, snapshot.captured_at_s, cameras)
        for camera in cameras:
            snapshot.geometry._views[camera.name].camera = camera
    update_rgb(source, rgb); update_rgb(current, shifted)
    candidate, segment = inputs(xyz=None)
    result = ground_candidate(candidate, segment, source, current)
    assert result['status'] == 'grounded'
    assert result['evidence']['correspondence']['method'] == 'local_material_affine_flow'
    np.testing.assert_allclose(result['candidate']['xyz_m'], [.025, .005, 1.], atol=.002)


def test_missing_mask_depth_conflict_and_wrong_source_do_not_fabricate_target():
    candidate, segment = inputs()
    source = scene(); source.objects['contact'].masks['external'][16, 16] = False
    result = ground_candidate(candidate, segment, source, source)
    assert result['status'] == 'unresolved' and result['candidate'] is None
    assert result['reason'] == 'contact_pixel_outside_selected_region'
    source = scene(); source.geometry._views['external'].contradiction_map[16, 16] = 1
    assert ground_candidate(candidate, segment, source, source)['reason'] == 'contact_depth_crossview_conflict'
    source = scene(1)
    assert ground_candidate(candidate, segment, source, source)['reason'] == 'contact_source_frame_mismatch'
    source = scene()
    segment['object'] = 'unrelated_body'
    assert ground_candidate(candidate, segment, source, source)['reason'] == 'contact_material_object_mismatch'


def test_large_correction_returns_measured_alternative_for_semantic_review_not_clipped_pose():
    candidate, segment = inputs(xyz=(.15, .005, 1.))
    source = scene()
    result = ground_candidate(candidate, segment, source, source)
    assert result['status'] == 'unresolved' and result['candidate'] is None
    assert result['reason'] == 'contact_correction_exceeds_local_scope'
    np.testing.assert_allclose(result['evidence']['corrected_xyz_m'], [.005, .005, 1.])


def test_placement_uses_actual_observed_payload_support_patch_not_object_centre():
    source = scene()
    source.objects['contact'].evidence.update(role='support', parent_id=None)
    candidate, segment = inputs(xyz=(.025, .005, 1.13), quat=(0., 1., 0., 0.), standoff=.03)
    segment['contact_mode'] = 'place'
    segment['grounding'].update(reference='payload_support', offset_hand_m=None)
    points = np.array([[x, y, z] for x in (-.01, .01) for y in (-.02, .02) for z in (.05, .10)])
    held = {'object_id': 'body', 'points_in_gripper': points, 'extent_unknown': False, 'geometry_complete': False}
    result = ground_candidate(candidate, segment, source, source, held=held)
    assert result['status'] == 'grounded'
    np.testing.assert_allclose(result['candidate']['xyz_m'], [.005, .005, 1.13])
    assert result['evidence']['payload_contact']['points'] == 4
    assert not result['evidence']['payload_contact']['complete_object_shape']
    held['object_id'] = 'unrelated_body'
    assert ground_candidate(candidate, segment, source, source, held=held)['reason'] == 'placement_payload_object_mismatch'
    held['object_id'] = 'body'
    held['extent_unknown'] = True
    assert ground_candidate(candidate, segment, source, source, held=held)['reason'] == 'payload_contact_extent_unavailable'
