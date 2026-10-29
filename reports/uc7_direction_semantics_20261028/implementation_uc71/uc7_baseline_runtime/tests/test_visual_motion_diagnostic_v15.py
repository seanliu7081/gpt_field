"""RGB evidence remains a passive, bounded, explicitly ambiguous diagnostic."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import decode_image, image_to_data_url
from gpt_field.visual_motion_diagnostic_v15 import VisualMotionDiagnosticV15


def observation(rgb=None, xyz=(0., 0., 0.)):
    if rgb is None:
        rgb = np.full((224, 224, 3), 100, np.uint8)
    return {"observation/image": image_to_data_url(rgb, rotate=False),
            "observation/state": [*xyz, 0., 0., 0., .04, -.04]}


def setup(roi=(.3, .3, .7, .7)):
    track = SimpleNamespace(ident="claimed_tip", reliable=True, position=np.array([112., 112.]),
        points=np.array([[110., 110.], [114., 110.], [112., 114.]]),
        matrix=np.eye(2), translation=np.zeros(2))
    monitor = SimpleNamespace(refs={"external": {"moving_id": "claimed_tip", "moving_kind": "tool"}},
        tracks={"moving": track}, _active_reference={"unvalidated_travel_m": .017},
        attachment_evidence=False)
    diagnostic = VisualMotionDiagnosticV15()
    obs = observation()
    context = {"tool_external_roi_uv": list(roi)}
    diagnostic.configure(obs, monitor, context)
    return diagnostic, monitor, obs, context


def test_no_new_image_or_evidence_from_stationary_robot_holds():
    diagnostic, monitor, obs, _ = setup()
    for i in range(30):
        rgb = np.full((224, 224, 3), 100+i, np.uint8)
        after = observation(rgb)
        report = diagnostic.observe(obs, after, monitor)
        obs = after
    assert report["measured_path_m"] == 0.
    assert not report["diagnostic_available"] and diagnostic.image(obs) is None
    assert report["retained_frame_count"] <= 2


def test_four_mm_acknowledged_motion_displays_raw_changes_and_static_selected_support():
    diagnostic, monitor, obs, _ = setup()
    debt = copy.deepcopy(monitor._active_reference)
    for i in range(1, 5):
        rgb = np.full((224, 224, 3), 100, np.uint8)
        rgb[82:92, 76+i:87+i] = 220
        after = observation(rgb, (i*.001, 0, 0))
        report = diagnostic.observe(obs, after, monitor)
        obs = after
        assert report["diagnostic_available"] == (i == 4)
    change = report["raw_rgb_change"]
    assert change["global_changed_pixel_count"] > 50
    assert change["current_descriptor_support_changed_pixel_count"] == 0
    assert report["pair"]["net_delta_xyz_m"] == pytest.approx([.004, 0, 0])
    assert report["pair"]["acknowledged_path_m"] == pytest.approx(.004)
    assert report["current_descriptor_support_bounds_px"] == [107, 107, 117, 117]
    assert monitor._active_reference == debt
    assert not report["execution_veto"] and not report["control_debt_modified"]
    assert not report["automatic_reference_promotion"] and not report["semantic_correspondence_certified"]
    assert "weak projection" in report["interpretation"]
    image = diagnostic.image(obs)
    assert decode_image(image).size == (1344, 644)
    assert diagnostic.image(obs) == image
    # The diagnostic summary does not duplicate images in text/history.
    assert "data:image" not in json.dumps(report)


@pytest.mark.parametrize("appearance", ["shadow", "background", "global_brightness"])
def test_fixed_camera_appearance_changes_are_not_promoted_to_tool_identity(appearance):
    diagnostic, monitor, obs, _ = setup()
    rgb = np.full((224, 224, 3), 100, np.uint8)
    if appearance == "shadow":
        rgb[70:150, 70:150] = 40
    elif appearance == "background":
        rgb[15:50, 170:210] = 220
    else:
        rgb[:] = 130
    after = observation(rgb, (0, .004, 0))
    report = diagnostic.observe(obs, after, monitor)
    assert report["raw_rgb_change"]["global_changed_pixel_count"] > 100
    assert report["diagnostic_available"]
    for key in ("semantic_correspondence_certified", "attachment_certified", "clearance_certified",
                "automatic_reference_promotion", "automatic_candidate_selection"):
        assert report[key] is False
    assert "shadows" in report["interpretation"] and "proximal" in report["interpretation"]


def test_uint8_small_darkening_does_not_wrap_into_large_change():
    diagnostic, monitor, obs, _ = setup()
    after = observation(np.full((224, 224, 3), 99, np.uint8), (.004, 0, 0))
    report = diagnostic.observe(obs, after, monitor)
    assert report["raw_rgb_change"]["global_changed_pixel_count"] == 0
    assert report["raw_rgb_change"]["current_descriptor_support_mean_absolute_change"] == 1.


def test_reseed_or_inconsistent_roi_preserves_history_and_actual_support_coordinates():
    diagnostic, monitor, obs, context = setup()
    after = observation(xyz=(.004, 0, 0))
    diagnostic.observe(obs, after, monitor)
    monitor.tracks["moving"] = copy.deepcopy(monitor.tracks["moving"])
    monitor.refs["external"]["moving_id"] = "renamed_claim"
    context["tool_external_roi_uv"] = [.1, .1, .2, .2]
    report = diagnostic.configure(after, monitor, context)
    assert report["diagnostic_available"] and report["measured_path_m"] == pytest.approx(.004)
    assert report["history_resets_for_unobserved_gap"] == 0
    assert report["current_descriptor_support_bounds_px"] == [107, 107, 117, 117]
    assert not report["current_moving_reference"]["support_inside_declared_tool_roi"]
    assert diagnostic.image(after) is not None


def test_lost_tracker_still_exposes_last_support_as_unreliable_without_certification():
    diagnostic, monitor, obs, _ = setup()
    monitor.tracks["moving"].reliable = False
    after = observation(xyz=(0, 0, .004))
    report = diagnostic.observe(obs, after, monitor)
    assert report["diagnostic_available"]
    assert not report["current_moving_reference"]["reliable"]
    assert report["current_descriptor_support_bounds_px"] is not None
    assert report["whole_tool_roi"]["status"] == "last_declared_rectangle_not_currently_tracked"


def test_history_is_bounded_by_actual_travel_and_latest_frame_is_current():
    diagnostic, monitor, obs, _ = setup()
    for i in range(1, 81):
        rgb = np.full((224, 224, 3), 100, np.uint8)
        rgb[10, 10] = i
        after = observation(rgb, (i*.00025, 0, 0))
        report = diagnostic.observe(obs, after, monitor)
        obs = after
        assert report["retained_frame_count"] <= 8
    assert report["diagnostic_available"]
    assert report["pair"]["acknowledged_path_m"] <= .005+1e-12
    assert report["pair"]["current_transition"] == 80
    assert diagnostic.current.rgb[10, 10, 0] == 80
    assert len(diagnostic._frames()) <= 8


def test_current_observation_binding_and_duplicate_transition_fail_closed_for_image_only():
    diagnostic, monitor, obs, _ = setup()
    after = observation(xyz=(.004, 0, 0))
    diagnostic.observe(obs, after, monitor)
    assert diagnostic.image(after) is not None and diagnostic.image(obs) is None
    changed = observation(np.full((224, 224, 3), 130, np.uint8), (.004, 0, 0))
    assert diagnostic.image(changed) is None
    report = diagnostic.observe(obs, after, monitor)
    assert report["status"] == "unacknowledged_transition_gap"
    assert report["measured_path_m"] == pytest.approx(.004)
    assert diagnostic.image(after) is None
    assert not report["execution_veto"]


def test_out_and_back_path_reports_zero_net_and_does_not_invent_motion_direction():
    diagnostic, monitor, obs, _ = setup()
    away = observation(xyz=(.002, 0, 0))
    diagnostic.observe(obs, away, monitor)
    report = diagnostic.observe(away, obs, monitor)
    assert report["diagnostic_available"] and report["pair"]["acknowledged_path_m"] == pytest.approx(.004)
    assert report["pair"]["net_direction_xyz"] is None
    assert report["raw_rgb_change"]["global_changed_pixel_count"] == 0
    assert not report["semantic_correspondence_certified"]


def test_axisangle_pi_boundary_does_not_invent_full_rotation():
    diagnostic, monitor, obs, context = setup()
    obs["observation/state"][3] = np.pi-1e-4
    diagnostic.configure(obs, monitor, context)
    after = copy.deepcopy(obs)
    after["observation/state"][0] += .004
    after["observation/state"][3] = -np.pi+1e-4
    report = diagnostic.observe(obs, after, monitor)
    assert report["pair"]["orientation_change_rad"] == pytest.approx(.0002, abs=1e-8)
    assert "rotation_vector_change_rad" not in report["pair"]


def replay_v14_prefix():
    """Only saved allowed observations/model outputs; no environment replay."""
    from gpt_field.visual_monitor_v12 import RoleAwareVisualMonitorV12
    path = Path(__file__).parents[1]/"runs/visual_field_v14_zero_shot_001/trials/bowl_original_zero_hybrid_v14/trace.jsonl"
    if not path.exists():
        pytest.skip("archived V14 RGB trace not installed")
    monitor, diagnostic = RoleAwareVisualMonitorV12(), VisualMotionDiagnosticV15()
    obs = None
    def images(value):
        return {"external": np.asarray(decode_image(value["observation/image"])),
                "wrist": np.asarray(decode_image(value["observation/wrist_image"]))}
    for line in path.open():
        row = json.loads(line)
        if row["event"] == "reset":
            obs = row["observation"]
        elif row["event"] == "action" and row["diagnostics"].get("prediction"):
            prediction = row["diagnostics"]["prediction"]
            monitor.configure(images(obs), prediction["visual_references"], obs["observation/state"],
                              intent_id=row["diagnostics"]["chunk_index"])
            diagnostic.configure(obs, monitor, prediction["motion_context"])
        elif row["event"] == "executed":
            after = row["observation"]
            monitor.observe(images(after), after["observation/state"], row["action"])
            debt = copy.deepcopy(monitor._active_reference)
            diagnostic.observe(obs, after, monitor)
            np.testing.assert_equal(monitor._active_reference, debt)
            obs = after
            if row["step"] == 72:
                return diagnostic, obs
    pytest.fail("archived actual transition72 absent")


def test_archive_v14_actual_image_changes_do_not_make_static_fingertip_a_valid_reference():
    diagnostic, obs = replay_v14_prefix()
    report = diagnostic.summary()
    assert report["diagnostic_available"] and report["pair"]["acknowledged_path_m"] >= .004
    change = report["raw_rgb_change"]
    assert change["global_changed_pixel_count"] > 100
    assert change["current_descriptor_support_changed_pixel_count"] == 0
    assert change["current_descriptor_support_mean_absolute_change"] == 0.
    assert not report["semantic_correspondence_certified"]
    assert diagnostic.image(obs) is not None
