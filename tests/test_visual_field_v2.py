import copy
import json

import numpy as np
import pytest

from gpt_field.visual_field_v2 import (
    FeatureBankTrack, ValidatedResponse, VisualFieldControllerV2,
    PANDA_JOINT_LOWER, PANDA_JOINT_UPPER,
)

RNG = np.random.default_rng(5421)
PATCHES = {name: RNG.integers(20, 240, (19, 19), dtype=np.uint8)
           for name in ("moving", "target", "tool", "obstacle", "impostor")}
BASE = {"moving": np.array([42, 38]), "target": np.array([96, 96]),
        "tool": np.array([18, 38]), "obstacle": np.array([87, 39])}


def frame(eef=(0., 0.), *, missing=(), camera=(0, 0), payload=False, slip=(0, 0), obstacle=(0, 0), impostor=False):
    pixels = np.full((128, 128, 3), 120, dtype=np.uint8)
    movement = np.rint(np.asarray(eef)*400).astype(int)
    for name, base in BASE.items():
        if name in missing or (name == "tool" and not payload):
            continue
        offset = movement if name in ("moving", "tool") else np.zeros(2, int)
        if name == "tool":
            offset = offset+np.asarray(slip)
        if name == "obstacle":
            offset = offset+np.asarray(obstacle)
        point = base+offset+np.asarray(camera)
        x, y = map(int, point)
        patch = PATCHES["impostor"] if impostor and name == "moving" else PATCHES[name]
        pixels[y-9:y+10, x-9:x+10, :] = patch[:, :, None]
    return pixels


def images(eef=(0., 0.), *, missing=(), payload=False, camera=False, slip=(0, 0), obstacle=(0, 0), impostor=False):
    return {name: frame(eef, missing=missing if not isinstance(missing, dict) else missing.get(name, ()),
        payload=payload, camera=(-round(eef[0]*200), 0) if camera and name == "wrist" else (0, 0),
        slip=slip, obstacle=obstacle, impostor=impostor) for name in ("external", "wrist")}


def seeds(*, payload=False, obstacles=False, target=(94, 72)):
    value = {"moving_id": "object" if payload else "tool", "target_id": "target",
        "moving_point_uv": (BASE["moving"]/127).tolist(),
        "target_anchor_uv": (BASE["target"]/127).tolist(), "target_point_uv": (np.asarray(target)/127).tolist(),
        "moving_radius_uv": 0., "moving_kind": "payload" if payload else "tool",
        "tool_point_uv": (BASE["tool"]/127).tolist() if payload else None,
        "blocked_boxes_uv": [{"object_id": "obstacle", "lower_uv": ([76/127, 28/127]),
                               "upper_uv": [98/127, 50/127]}] if obstacles else [], "unknown_boxes_uv": []}
    return {name: copy.deepcopy(value) for name in ("external", "wrist")}


def state(eef=(0, 0), *, angle=0., fingers=(.005, -.005)):
    return [*eef, .6, angle, 0., 0., *fingers]


def train(*, count=4, payload=False, obstacles=False, camera=False, angle_noise=False, bounds=False):
    c = VisualFieldControllerV2(grid_size=16, joint_lower=[-1]*7 if bounds else None,
                               joint_upper=[1]*7 if bounds else None)
    q, eef = np.zeros(7), np.zeros(2)
    c.configure(images(payload=payload), seeds(payload=payload, obstacles=obstacles), q, state8=state())
    for index in range(count):
        eef[0] += .005
        q[0] += .01
        c.observe(images(eef, payload=payload, camera=camera), q,
                  state8=state(eef, angle=.001*(-1)**index if angle_noise else 0.),
                  executed_action=[.1, 0, 0, 0, 0, 0, 1])
    return c, q, eef


def test_virtual_target_tracks_landmark_offset_not_texture_at_goal():
    c = VisualFieldControllerV2(grid_size=16)
    initial = c.configure(images(), seeds(), np.zeros(7), state8=state())
    assert initial["tracks_reliable"]
    assert np.std(frame()[68:77, 90:99]) == 0  # desired target contains no texture
    c.observe({name: frame(camera=(2, 3)) for name in ("external", "wrist")}, np.zeros(7),
              state8=state(), executed_action=[0]*6+[1])
    for view in c.summary()["views"].values():
        assert view["target_anchor_uv"] == pytest.approx(np.array([98, 99])/127)
        assert view["target_point_uv"] == pytest.approx(np.array([96, 75])/127)
        assert view["target_offset_px"] == pytest.approx([-2, -24])
        assert view["target_anchor_track_reliable"]


def test_feature_bank_uses_offsets_and_consensus_with_partial_occlusion():
    pixels = frame()
    track = FeatureBankTrack(pixels.mean(axis=2), BASE["moving"]/127, "tool")
    assert len(track.features) >= 2
    assert any(np.linalg.norm(f["offset"]) > 0 for f in track.features)
    shifted = frame((.005, 0))
    # A tiny damaged corner must not move the semantic seed to the visible edge.
    shifted[30:33, 35:38] = 120
    assert track.track(shifted.mean(axis=2))
    assert track.position == pytest.approx([44, 38], abs=1)
    assert track.consensus_count >= 2


def test_loss_reacquires_original_identity_twice_without_jump_to_impostor():
    track = FeatureBankTrack(frame().mean(axis=2), BASE["moving"]/127, "tool")
    assert not track.track(frame(missing=("moving",)).mean(axis=2))
    assert not track.track(frame(impostor=True).mean(axis=2))
    assert track.position == pytest.approx(BASE["moving"])
    assert not track.track(frame().mean(axis=2))
    assert track.status == "reacquisition_pending"
    assert track.track(frame().mean(axis=2))
    assert track.status == "reacquired_original_features"
    assert track.position == pytest.approx(BASE["moving"])


def test_ambiguous_duplicate_does_not_reassign_identity():
    pixels = np.full((96, 96), 120., dtype=float)
    patch = PATCHES["moving"][:7, :7]
    pixels[27:34, 27:34] = patch
    track = FeatureBankTrack(pixels, np.array([30, 30])/95, "tool", extent=0)
    duplicate = pixels.copy()
    duplicate[27:34, 37:44] = patch
    # Both complete local appearances are visible in the bounded search area.
    assert not track.track(duplicate)
    assert track.position == pytest.approx([30, 30])


def test_response_rank_one_requires_held_out_validation_and_rejects_unseen_axis():
    model = ValidatedResponse(residual_limit=.01, minimum_excitation=.001)
    for _ in range(2):
        model.add([.1, 0, 0], [.005, 0, 0])
    assert model.predict([.1, 0, 0]) is None
    assert model.predict([.1, 0, 0], validated=False) is not None
    for _ in range(2):
        model.add([.1, 0, 0], [.005, 0, 0])
    assert model.summary()["reliable"] and model.rank == 1
    assert model.predict([.1, 0, 0]) == pytest.approx([.005, 0, 0], abs=1e-6)
    assert model.predict([0, .1, 0]) is None
    assert model.predict([.3, 0, 0]) is None
    json.dumps(model.summary(), allow_nan=False)


def test_predicted_response_failure_revokes_readiness_and_is_observable():
    model = ValidatedResponse(residual_limit=.001, minimum_excitation=.001)
    for _ in range(4):
        model.add([.1, 0, 0], [.005, 0, 0])
    model.add([.1, 0, 0], [-.02, 0, 0])
    assert not model.summary()["reliable"]
    assert model.summary()["failure_count"] == 1
    assert model.summary()["last_prediction_error"] > .02
    assert model.predict([.1, 0, 0]) is None


def test_actual_rank_one_motion_activates_local_servo():
    c, q, _ = train(bounds=True)
    info = c.summary()
    assert info["action_to_eef"]["reliable"] and info["eef_to_image"]["reliable"]
    assert info["action_to_joint"]["reliable"] and info["eef_to_image"]["rank"] == 1
    result = c.select_action([.1, 0, 0, 0, 0, 0, 1], q=q)
    assert result["status"] == "visual_distance_descent"
    assert result["accepted"] and not result["need_refresh"] and result["horizon"] == 1
    assert 0 < result["action"][0] <= .1
    assert result["action"][3:] == [0, 0, 0, 1]
    assert result["diagnostics"]["risk_unknown"]  # image checks do not establish 3D clearance
    assert not result["diagnostics"]["projected_risk_unknown"]
    assert not result["diagnostics"]["clearance_certified"]
    json.dumps(result, allow_nan=False)


def test_first_unmeasured_direction_requests_direct_probe_not_blind_bootstrap():
    c = VisualFieldControllerV2(grid_size=16)
    c.configure(images(), seeds(), np.zeros(7), state8=state())
    result = c.select_action([1, 0, 0, 0, 0, 0, 1])
    assert not result["accepted"] and result["status"] == "direct_probe_required"
    assert result["action"][:6] == [0]*6
    assert result["need_refresh"]


def test_provisional_empirical_bootstrap_is_small_checked_and_explicitly_uncertain():
    c, q, _ = train(count=2, bounds=True)
    result = c.select_action([1, 0, 0, 0, 0, 0, 1], q=q)
    assert result["status"] == "bootstrap_empirical_image_checked"
    assert result["accepted"] and result["need_refresh"]
    assert np.linalg.norm(result["action"][:3]) <= .1
    assert result["diagnostics"]["risk_unknown"] and result["diagnostics"]["projected_risk_unknown"]
    unseen = c.select_action([0, 1, 0, 0, 0, 0, 1])
    assert not unseen["accepted"] and unseen["status"] == "direct_probe_required"


def test_missing_tracks_hold_and_offer_specific_reseed_diagnostics():
    c, q, eef = train()
    c.observe(images(eef, missing={"wrist": ("moving",)}), q, state8=state(eef), executed_action=[0]*6+[1])
    result = c.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert not result["accepted"] and result["status"] == "visual_reference_unavailable"
    assert c.summary()["recovery"]["missing_views"] == ["wrist"]
    assert c.summary()["recovery"]["reseed_requests"][0]["features"] == ["moving"]
    c.observe(images(eef), q, state8=state(eef), executed_action=[0]*6+[1])
    assert not c.summary()["tracks_reliable"]
    c.observe(images(eef), q, state8=state(eef), executed_action=[0]*6+[1])
    assert c.summary()["tracks_reliable"]


def test_obstacle_tracks_and_camera_relative_models_preserve_checked_motion():
    c, q, eef = train(obstacles=True, camera=True)
    assert c.summary()["geometry_reliable"]
    assert all(v["obstacle_tracks"][0]["response_reliable"] for v in c.summary()["views"].values())
    assert c.select_action([.1, 0, 0, 0, 0, 0, 1], q=q)["status"] == "visual_distance_descent"
    c.observe(images(eef, camera=True, missing={"wrist": ("obstacle",)}), q, state8=state(eef), executed_action=[0]*6+[1])
    assert not c.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]
    assert c.summary()["recovery"]["missing_views"] == ["wrist"]


def test_obstacle_intersecting_predicted_segment_rejects_bootstrap():
    c, _, _ = train(count=2)
    for view in c.views.values():
        # Thin exclusion after the current point: checked continuously, not just endpoint cells.
        moving = view["tracks"]["moving"].uv
        view["field"].boxes = [(moving+np.array([.004, -.05]), moving+np.array([.005, .05]))]
    result = c.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert not result["accepted"]
    assert result["status"] == "predicted_image_path_blocked"


def test_joint_limits_gate_measured_and_predicted_state_and_stale_observations():
    c, q, _ = train(bounds=True)
    c.joint_upper[0] = q[0]+.003
    result = c.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert not result["accepted"]
    assert "predicted_joint_limit" in result["diagnostics"]["rejection_reasons"]
    c.joint_upper[0] = q[0]-.001
    assert c.select_action([.1, 0, 0, 0, 0, 0, 1])["status"] == "measured_joint_limit"
    with pytest.raises(ValueError, match="last acknowledged"):
        c.select_action([.1, 0, 0, 0, 0, 0, 1], q=q+.01)
    assert len(PANDA_JOINT_LOWER) == len(PANDA_JOINT_UPPER) == 7


def test_tiny_orientation_noise_keeps_measured_eef_response_and_activates_servo():
    c, _, _ = train(angle_noise=True)
    assert c.summary()["action_to_eef"]["reliable"]
    assert c.summary()["eef_to_image"]["reliable"]
    assert c.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]


def test_actual_rotation_resets_projection_only_and_rotation_actions_require_direct():
    c, q, eef = train()
    old_count = c.summary()["action_to_eef"]["sample_count"]
    c.observe(images(eef), q, state8=state(eef, angle=.2), executed_action=[0, 0, 0, .4, 0, 0, 1])
    info = c.summary()
    assert info["action_to_eef"]["sample_count"] == old_count
    assert info["eef_to_image"]["sample_count"] == 0
    assert info["projection_reset_reason"] == "orientation_or_gripper_changed"
    rotated = c.select_action([.1, 0, 0, .1, 0, 0, 1])
    assert not rotated["accepted"] and rotated["status"] == "rotation_requires_direct_probe"


def test_reseed_preserves_actuator_map_but_does_not_preserve_wrong_feature_response():
    c, q, eef = train()
    old_action = c.action_model
    fresh = seeds()
    for name in fresh:
        fresh[name]["moving_point_uv"] = ((BASE["moving"]+np.array([8, 0]))/127).tolist()
        fresh[name]["target_id"] = "new_target"
    c.configure(images(eef), fresh, q, state8=state(eef))
    assert c.action_model is old_action and c.summary()["action_to_eef"]["reliable"]
    assert not c.summary()["eef_to_image"]["reliable"]


def test_same_identity_refresh_retains_response_and_updates_virtual_goal_offset():
    c, q, eef = train(obstacles=True)
    old_feature, old_obstacle = c.feature_model, c.views["external"]["boxes"][0]["response"]
    fresh = seeds(obstacles=True, target=(91, 71))
    for name in fresh:
        fresh[name]["moving_point_uv"] = ((BASE["moving"]+np.array([8, 0]))/127).tolist()
    c.configure(images(eef), fresh, q, state8=state(eef))
    assert c.feature_model is old_feature
    assert c.views["external"]["boxes"][0]["response"] is old_obstacle
    assert c.summary()["views"]["external"]["target_point_uv"] == pytest.approx(np.array([91, 71])/127)


def test_payload_comotion_requires_independent_tool_motion_and_survives_target_loss():
    c, q, eef = train(payload=True)
    info = c.summary()
    assert info["co_motion_evidence"] and info["object_tool_relative_change_px"] <= 1
    assert all(v["tool_track_reliable"] and v["tool_point_uv"] is not None for v in info["views"].values())
    c.observe(images(eef, payload=True, missing=("target",)), q, state8=state(eef), executed_action=[0]*6+[1])
    assert not c.summary()["tracks_reliable"]
    assert all(v["moving_track_reliable"] and v["tool_track_reliable"] for v in c.summary()["views"].values())


def test_closed_fingers_without_motion_are_not_attachment_evidence_and_slip_revokes():
    c = VisualFieldControllerV2(grid_size=16)
    q = np.zeros(7)
    c.configure(images(payload=True), seeds(payload=True), q, state8=state())
    for _ in range(4):
        c.observe(images(payload=True), q, state8=state(), executed_action=[0]*6+[1])
    assert not c.summary()["co_motion_evidence"]
    c, q, eef = train(payload=True)
    q[0] += .01; eef[0] += .005
    c.observe(images(eef, payload=True, slip=(-4, 0)), q, state8=state(eef), executed_action=[.1, 0, 0, 0, 0, 0, 1])
    assert not c.summary()["co_motion_evidence"]
    assert c.summary()["object_tool_relative_change_px"] >= 3


def test_held_out_measured_eef_error_is_reported_and_cannot_keep_old_servo():
    c, q, eef = train()
    eef[0] -= .02; q[0] -= .04
    info = c.observe(images(eef), q, state8=state(eef), executed_action=[.1, 0, 0, 0, 0, 0, 1])
    assert info["action_to_eef"]["failure_count"] == 1
    assert info["observed_prediction_errors"]["eef_prediction"] > .02
    assert not c.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]


def test_no_progress_and_unknown_seed_are_holds_not_blind_motion():
    c = VisualFieldControllerV2(grid_size=16)
    q = np.zeros(7)
    c.configure(images(), {"external": None, "wrist": None}, q, state8=state())
    assert c.select_action([1, 0, 0, 0, 0, 0, 1])["status"] == "visual_reference_unavailable"
    c.configure(images(), seeds(), q, state8=state())
    for _ in range(5):
        c.observe(images(), q, state8=state(), executed_action=[.1, 0, 0, 0, 0, 0, 1])
    assert c.summary()["no_progress"]
    assert c.select_action([.1, 0, 0, 0, 0, 0, 1])["status"] == "no_progress"


@pytest.mark.parametrize("mutate", [
    lambda s: s["external"].pop("target_anchor_uv"),
    lambda s: s["external"].update(target_anchor_uv=[float("nan"), .5]),
    lambda s: s["external"].update(object_pose=[1, 2, 3]),
    lambda s: s["external"].update(moving_radius_uv=True),
    lambda s: s["external"].update(moving_id="target"),
])
def test_invalid_or_privileged_seed_fields_are_rejected(mutate):
    c = VisualFieldControllerV2(grid_size=16)
    value = seeds(); mutate(value)
    with pytest.raises(ValueError):
        c.configure(images(), value, np.zeros(7), state8=state())


@pytest.mark.parametrize("action", [[2, 0, 0, 0, 0, 0, 1], [float("nan")]*7, [0]*6])
def test_nonfinite_or_unnormalized_actions_rejected(action):
    with pytest.raises(ValueError):
        VisualFieldControllerV2(grid_size=16).select_action(action)


def test_attachment_epoch_changes_on_identity_reseed_or_loss_but_not_new_target():
    c, q, eef = train(payload=True)
    epoch = c.summary()["attachment_epoch"]
    fresh = seeds(payload=True)
    for value in fresh.values():
        value["moving_point_uv"] = ((BASE["moving"]+[8, 0])/127).tolist()
        value["tool_point_uv"] = ((BASE["tool"]+[8, 0])/127).tolist()
        value["target_id"] = "new_destination"
    c.configure(images(eef, payload=True), fresh, q, state8=state(eef))
    assert c.summary()["attachment_epoch"] == epoch
    assert c.summary()["co_motion_evidence"]
    c.observe(images(eef, payload=True, missing=("tool",)), q, state8=state(eef), executed_action=[0]*6+[1])
    assert c.summary()["attachment_epoch"] == epoch+1
    assert not c.summary()["co_motion_evidence"]
    c.configure(images(eef, payload=True), fresh, q, state8=state(eef))
    assert c.summary()["attachment_epoch"] == epoch+2


def test_recent_failed_prediction_cannot_fall_back_to_provisional_bootstrap():
    c, _, _ = train(count=2)
    assert c.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]
    c.action_model.last_error = c.action_model.residual_limit*1.5
    result = c.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert not result["accepted"]
    assert result["status"] == "direct_probe_required"


def test_real_adapter_rounded_hold_orientation_is_zeroed_before_local_translation():
    from gpt_field.action import ActionAdapter, _rotvec_quaternion
    c, q, _ = train(angle_noise=True)
    rounded_hold = np.round(_rotvec_quaternion(c.state8[3:6]), 3)
    proposed = ActionAdapter().adapt([.005, 0, 0], rounded_hold, "close", c.state8)
    assert 1e-8 < np.linalg.norm(proposed[3:6]) <= .02
    result = c.select_action(proposed, q=q)
    assert result["accepted"] and result["status"] == "visual_distance_descent"
    assert result["action"][3:6] == [0, 0, 0]
    assert result["action"][6] == proposed[6]
    assert result["diagnostics"]["rotation_deadband_applied"]
    assert not result["diagnostics"]["rotation_gripper_preserved"]
    large = list(proposed); large[3:6] = [.021, 0, 0]
    assert c.select_action(large)["status"] == "rotation_requires_direct_probe"


def test_empirical_bootstrap_also_removes_small_unmodeled_rotation():
    c, _, _ = train(count=2)
    result = c.select_action([.1, 0, 0, 0, .015, 0, 1])
    assert result["accepted"] and result["status"] == "bootstrap_empirical_image_checked"
    assert result["action"][3:6] == [0, 0, 0]
    assert result["diagnostics"]["rotation_deadband_applied"]
