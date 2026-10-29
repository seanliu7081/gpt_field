"""Interaction evidence from rendered public RGB, using real feature tracking.

The fixture supplies declared boxes and camera/robot measurements only. It does
not supply measured object poses to InteractionEvidence or mock optical flow.
Known object motion is used solely to render the synthetic sensor images.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gpt_field.agent_speedup_v3_interaction import InteractionEvidence


IMAGE_SIDE = 160
FOCAL = 160.
PRINCIPAL = (IMAGE_SIDE-1)/2
TEXTURE_HALF_WIDTH_M = .18
# Aperiodic material texture avoids ambiguous checkerboard correspondences.
_TEXTURE = cv2.GaussianBlur(np.random.default_rng(41).integers(
    50, 245, size=(80, 80), dtype=np.uint8), (3, 3), 0)


def render_material(object_x, camera_position):
    """Ray/plane intersection for a textured square at world z=0.

    These cameras look along -Z, with image Y pointing down. Independent inverse
    rendering is used instead of the tracker's projection helpers.
    """
    rows, columns = np.indices((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    cx, cy, height = camera_position
    world_x = cx+(columns-PRINCIPAL)*height/FOCAL
    world_y = cy-(rows-PRINCIPAL)*height/FOCAL
    u = (world_x-object_x+TEXTURE_HALF_WIDTH_M)/(2*TEXTURE_HALF_WIDTH_M)*(_TEXTURE.shape[1]-1)
    v = (TEXTURE_HALF_WIDTH_M-world_y)/(2*TEXTURE_HALF_WIDTH_M)*(_TEXTURE.shape[0]-1)
    gray = cv2.remap(_TEXTURE, u.astype(np.float32), v.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=24)
    return np.repeat(gray[:, :, None], 3, axis=2)


class SyntheticView:
    def __init__(self, hand_x=0., object_x=0., *, gap_mm=50., missing_wrist=False, covered=False):
        self.position = np.array([hand_x, 0., .25])
        self.R = np.eye(3)
        self.gap_mm = gap_mm
        self.state = np.array([*self.position, 0., 0., 0., gap_mm/2000, -gap_mm/2000])
        self.covered = covered
        self.missing_wrist = missing_wrist
        origins = {"external": [0., 0., 1.], "wrist": [hand_x+.05, -.08, 1.]}
        self.observation = SimpleNamespace(geometry={
            "image": {"width": IMAGE_SIDE, "height": IMAGE_SIDE},
            "cameras": {name: {"position_world_m": origin,
                "rotation_world_from_camera": np.eye(3).tolist(),
                "focal_length_px": [FOCAL, FOCAL], "principal_point_px": [PRINCIPAL, PRINCIPAL],
                "pose_source": "fixed_world_camera" if name == "external" else "observed_robot_pose_fixed_mount"}
                for name, origin in origins.items()}})
        self.external_rgb_upright = render_material(object_x, origins["external"])
        self._wrist = render_material(object_x, origins["wrist"])
        if covered:
            self.external_rgb_upright[:] = 24
            self._wrist[:] = 24

    @property
    def wrist_rgb_upright(self):
        if self.missing_wrist:
            raise AttributeError("wrist RGB unavailable")
        return self._wrist


class DeclaredGeometry:
    """Only the initial model box and known hand occlusion; no scene truth API."""
    def __init__(self):
        self.entities = {"package": {"position_m": [0., 0., 0.], "attachment": "none"}}

    def _entity_boxes(self, entity, view, suppress_stale_padding=False):
        return [{"center": np.asarray(entity["position_m"]), "rotation": np.eye(3),
                 "half": np.array([.18, .18, .003])}]

    def _visual_occluders(self, view, target_id):
        if not view.covered:
            return []
        return [{"center": np.array([0., 0., .4]), "rotation": np.eye(3),
                 "half": np.array([.8, .8, .03])}]


def test_material_moving_with_hand_supports_hold_in_both_calibrated_views():
    geometry = DeclaredGeometry()
    tracker = InteractionEvidence(geometry)
    assert tracker.begin("package", SyntheticView(), 0)["status"] == "unknown"
    for step in range(1, 5):
        view = SyntheticView(hand_x=.025*step, object_x=.025*step)
        result = tracker.evidence("hold", "package", view, step)
    assert result["status"] == "supported"
    assert set(result["views"]) == {"external", "wrist"}
    assert all(report["status"] == "supported" for report in result["views"].values())
    assert result["conditional_attachment_supported"] is True
    assert result["attachment_verified"] is False and result["geometry_pose_updated"] is False
    assert result["task_success"] == "not_available_to_policy"
    # Visual agreement must not quietly become a measured object pose.
    assert geometry.entities["package"] == {"position_m": [0., 0., 0.], "attachment": "none"}


def test_stationary_object_does_not_pass_when_wrist_camera_moves_with_hand():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(), 0)
    for step in range(1, 5):
        result = tracker.evidence("hold", "package", SyntheticView(hand_x=.025*step), step)
        assert result["status"] == "unknown"
        assert result["conditional_attachment_supported"] is False
    # There is real image motion in the wrist view, but calibration identifies
    # it as camera motion around a stationary object, not object/hand rigidity.
    assert result["views"]["wrist"]["stationary_residual_px"] < result["views"]["wrist"]["held_residual_px"]


def test_finger_closure_without_material_motion_is_not_hold_evidence():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(gap_mm=80.), 0)
    for step, gap in enumerate((60., 40., 20., 5.), 1):
        result = tracker.evidence("hold", "package", SyntheticView(gap_mm=gap), step)
        assert result["status"] == "unknown" and not result["conditional_attachment_supported"]


def test_one_camera_missing_at_reference_cannot_be_replaced_by_one_good_track():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(missing_wrist=True), 0)
    for step in range(1, 5):
        result = tracker.evidence("hold", "package", SyntheticView(hand_x=.025*step, object_x=.025*step,
                                                                  missing_wrist=True), step)
    assert result["status"] == "unknown" and not result["conditional_attachment_supported"]
    assert result["views"]["external"]["status"] == "supported"
    assert "wrist" not in result["views"]


def test_newly_visible_camera_requires_new_motion_after_its_actual_reference():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(missing_wrist=True), 0)
    for step in range(1, 4):
        tracker.evidence("hold", "package", SyntheticView(hand_x=.025*step, object_x=.025*step,
                                                           missing_wrist=True), step)
    # Seeing the camera again cannot certify the preceding invisible lift.
    stopped = SyntheticView(hand_x=.075, object_x=.075)
    first = tracker.evidence("hold", "package", stopped, 4)
    assert first["status"] == "unknown"
    assert first["views"]["wrist"]["reason"] == "baseline_only"
    assert first["views"]["wrist"]["reference_step"] == 4
    still = tracker.evidence("hold", "package", stopped, 5)
    assert still["status"] == "unknown"
    for step in range(6, 10):
        x = .075+.02*(step-5)
        result = tracker.evidence("hold", "package", SyntheticView(hand_x=x, object_x=x), step)
    assert result["status"] == "supported"
    assert result["views"]["wrist"]["reference_step"] == 4
    assert result["views"]["external"]["reference_step"] == 0


@pytest.mark.parametrize("lost", ["wrist", "occluded"])
def test_losing_visibility_revokes_current_support_instead_of_reusing_old_result(lost):
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(), 0)
    for step in range(1, 4):
        result = tracker.evidence("hold", "package", SyntheticView(hand_x=.025*step, object_x=.025*step), step)
    assert result["status"] == "supported"
    now = SyntheticView(hand_x=.10, object_x=.10, missing_wrist=lost == "wrist", covered=lost == "occluded")
    result = tracker.evidence("hold", "package", now, 4)
    assert result["status"] == "unknown" and not result["conditional_attachment_supported"]
    assert result["observed_at_step"] == 4


def test_world_stability_and_retreat_after_opening_support_release():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(gap_mm=80.), 10, kind="release")
    results = []
    for index in range(1, 6):
        view = SyntheticView(hand_x=.025*index, object_x=0., gap_mm=80.)
        results.append(tracker.evidence("release", "package", view, 10+index, since_step=10))
    assert results[0]["status"] == "unknown"
    assert results[-1]["status"] == "supported"
    assert all(report["stable_samples"] >= 3 for report in results[-1]["views"].values())
    assert not results[-1]["attachment_verified"] and not results[-1]["conditional_attachment_supported"]
    assert results[-1]["task_success"] == "not_available_to_policy"


def test_open_gripper_without_separating_from_object_does_not_support_release():
    tracker = InteractionEvidence(DeclaredGeometry())
    tracker.begin("package", SyntheticView(gap_mm=80.), 0, kind="release")
    for step in range(1, 6):
        result = tracker.evidence("release", "package", SyntheticView(gap_mm=80.), step)
        assert result["status"] == "unknown"
