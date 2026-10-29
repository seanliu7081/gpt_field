"""Synthetic tests: no model, simulator, pretrained perception or hidden poses."""
import copy
import json

import numpy as np
import pytest

from gpt_field.visual_field import ImageDistanceField, SupportedLinearModel, VisualFieldController


def box(low, high, ident="obstacle"):
    return {"lower_uv": low, "upper_uv": high, "object_id": ident}


def test_image_distance_detours_around_blocker_and_reports_pixel_cost():
    field = ImageDistanceField((101, 101), [.85, .5], grid_size=21,
        blocked_boxes_uv=[box([.45, 0], [.55, .7])])
    path = field.route([.15, .5])
    assert path and max(p[1] for p in path) > .7
    assert field.cost([.15, .5]) > 70
    assert all(field.segment_valid(a, b) for a, b in zip(path[:-1], path[1:]))
    assert np.linalg.norm(np.asarray(path[-1])-[.85, .5]) < 1e-10


def test_unknown_wall_and_blocked_target_are_unreachable():
    field = ImageDistanceField((101, 101), [.8, .5], grid_size=21,
        unknown_boxes_uv=[box([.45, 0], [.55, 1])])
    assert field.unknown.any()
    assert field.cost([.2, .5]) is None
    assert field.route([.2, .5]) == []
    blocked = ImageDistanceField((101, 101), [.5, .5], grid_size=21,
        blocked_boxes_uv=[box([.45, .45], [.55, .55])])
    assert not np.isfinite(blocked.distance).any()


def test_thin_box_chord_and_payload_inflation_are_checked_continuously():
    empty = ImageDistanceField((101, 101), [.8, .8], grid_size=21,
        blocked_boxes_uv=[box([.51, .4], [.511, .6])])
    assert not empty.segment_valid([.3, .5], [.7, .5])
    assert not empty.segment_valid([-.01, .5], [.1, .5])
    small = ImageDistanceField((101, 101), [.8, .8], grid_size=21,
        blocked_boxes_uv=[box([.5, .4], [.55, .6])], moving_radius_uv=0)
    large = ImageDistanceField((101, 101), [.8, .8], grid_size=21,
        blocked_boxes_uv=[box([.5, .4], [.55, .6])], moving_radius_uv=.15)
    assert small.segment_valid([.3, .2], [.3, .8])
    assert not large.segment_valid([.3, .2], [.3, .8])


def test_unobstructed_distance_is_scaled_by_image_pixels_not_world_meters():
    first = ImageDistanceField((101, 101), [.8, .5], grid_size=21)
    second = ImageDistanceField((201, 201), [.8, .5], grid_size=21)
    assert first.cost([.2, .5]) == pytest.approx(60)
    assert second.cost([.2, .5]) == pytest.approx(120)


def test_tool_envelope_crossing_image_edge_is_unknown_not_free():
    field = ImageDistanceField((101, 101), [.8, .5], grid_size=21, moving_radius_uv=.1)
    assert not field.segment_valid([.1, .5], [.2, .5])
    assert field.cost([.1, .5]) is None
    assert field.unknown[:, 0].all()


def test_supported_linear_fit_accepts_planar_subspace_not_unseen_direction():
    model = SupportedLinearModel(residual_limit=.01)
    mapping = np.array([[2., 1.], [-1., 3.], [0., 0.]])
    for point in ([.1, 0, 0], [0, .1, 0], [-.1, 0, 0], [0, -.1, 0], [.1, .1, 0], [-.1, -.1, 0]):
        model.add(point, np.asarray(point)@mapping)
    assert model.summary()["reliable"]
    assert model.rank == 2
    assert model.predict([.05, .05, 0]) == pytest.approx([.05, .2], abs=1e-5)
    assert model.predict([0, 0, .05]) is None
    assert model.predict([3, 0, 0]) is None


def test_rank_one_or_inconsistent_responses_do_not_enable_prediction():
    rank_one = SupportedLinearModel(residual_limit=.01)
    for _ in range(8):
        rank_one.add([.1, 0, 0], [.2, .1])
    assert rank_one.predict([.1, 0, 0]) is None
    bad = SupportedLinearModel(residual_limit=.01)
    for index in range(8):
        point = [.1, 0] if index % 2 else [0, .1]
        bad.add(point, [index, -index])
    assert bad.status == "inconsistent_response"
    assert bad.predict([.1, 0]) is None


SIZE = 96
PATCHES = {key: np.random.default_rng(seed).integers(10, 246, (9, 9, 3), dtype=np.uint8)
           for key, seed in (("moving", 2), ("target", 18), ("obstacle", 93), ("tool", 136))}


def images(q=None, *, obstacle_shift=(0, 0), missing=None, duplicate=False, payload=False, tool_slip=(0, 0)):
    q = np.zeros(7) if q is None else np.asarray(q)
    result = {}
    for name in ("external", "wrist"):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        delta = np.rint(q[:2]*200).astype(int)
        positions = {"moving": np.array([25, 25])+delta, "target": np.array([75, 75]),
                     "obstacle": np.array([60, 30])+obstacle_shift}
        if payload:
            positions["tool"] = np.array([12, 25])+delta+tool_slip
        for key, point in positions.items():
            if missing == key or (isinstance(missing, tuple) and missing == (name, key)):
                continue
            x, y = map(int, point)
            frame[y-4:y+5, x-4:x+5] = PATCHES[key]
            if duplicate and key == "moving":
                frame[y-4:y+5, x+7:x+16] = PATCHES[key]
        result[name] = frame
    return result


def seeds(*, obstacles=False, payload=False):
    one = {"moving_id": "payload" if payload else "tool", "target_id": "destination",
           "moving_point_uv": (np.array([25, 25])/95).tolist(),
           "target_point_uv": (np.array([75, 75])/95).tolist(), "moving_radius_uv": 0.,
           "blocked_boxes_uv": [box((np.array([52, 22])/95).tolist(), (np.array([68, 38])/95).tolist())] if obstacles else [],
           "unknown_boxes_uv": [], "moving_kind": "payload" if payload else "tool",
           "tool_point_uv": (np.array([12, 25])/95).tolist() if payload else None}
    return {view: copy.deepcopy(one) for view in ("external", "wrist")}


def state(q=None):
    q = np.zeros(7) if q is None else q
    return [q[0]*.5, q[1]*.5, 1., 0., 0., 0., .005, -.005]


def calibrated():
    controller = VisualFieldController(grid_size=16)
    q = np.zeros(7)
    controller.configure(images(), seeds(), q, state8=state())
    actions = [[.1, 0, 0], [0, .1, 0], [-.1, 0, 0], [0, -.1, 0], [.1, 0, 0], [0, .1, 0], [.1, 0, 0], [0, .1, 0]]
    for command in actions:
        q[:3] += np.asarray(command)*.1
        feedback = controller.observe(images(q), q, state8=state(q), executed_action=[*command, 0, 0, 0, 1])
    return controller, q, feedback


def test_two_view_fit_selects_actual_osc_action_and_preserves_gripper():
    controller, q, feedback = calibrated()
    assert feedback["tracks_reliable"] and feedback["joint_to_image"]["reliable"]
    assert feedback["action_to_joint"]["reliable"]
    result = controller.select_action([.1, .1, 0, 0, 0, 0, 1], q=q)
    assert result["accepted"] and result["status"] == "visual_distance_descent"
    assert result["action"][6] == 1
    assert np.linalg.norm(result["action"][:3]) <= .2+1e-10
    assert result["horizon"] == 1
    assert result["diagnostics"]["clearance_certified"] is False
    assert result["diagnostics"]["risk_unknown"] is False
    json.dumps(result, allow_nan=False)


def test_missing_seed_holds_but_reliable_uncalibrated_tracks_allow_bounded_bootstrap():
    controller = VisualFieldController(grid_size=16)
    controller.configure(images(), {"external": None, "wrist": None}, np.zeros(7), state8=state())
    result = controller.select_action([1, 1, 1, .2, 0, 0, -1])
    assert result["action"][:6] == [0]*6
    assert result["status"] == "visual_reference_unavailable"
    assert not result["accepted"]
    assert result["horizon"] == 1 and result["need_refresh"]
    assert result["diagnostics"]["risk_unknown"]
    assert result["action"][6] == -1
    controller.configure(images(), seeds(), np.zeros(7), state8=state())
    bootstrap = controller.select_action([1, 1, 1, .2, 0, 0, -1])
    assert np.linalg.norm(bootstrap["action"][:3]) == pytest.approx(.1)
    assert bootstrap["status"] == "bootstrap_unverified"
    assert bootstrap["action"][3:] == [.2, 0, 0, -1]
    controller, _, _ = calibrated()
    rotated = controller.select_action([.1, .1, 0, 0, .2, 0, 1])
    assert rotated["status"] == "bootstrap_unverified"


def test_visual_tracking_failure_cannot_keep_old_map_certified():
    controller, q, _ = calibrated()
    feedback = controller.observe(images(q, missing=("wrist", "moving")), q,
        state8=state(q), executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert not feedback["tracks_reliable"]
    result = controller.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert result["diagnostics"]["risk_unknown"] and result["need_refresh"]
    assert not result["accepted"]


def test_moving_obstacle_boxes_follow_patch_and_lost_obstacle_invalidates_geometry():
    controller = VisualFieldController(grid_size=16)
    controller.configure(images(), seeds(obstacles=True), np.zeros(7), state8=state())
    before = controller.views["wrist"]["field"].boxes[0][0].copy()
    result = controller.observe(images(obstacle_shift=(3, 4)), np.zeros(7), state8=state(),
                                executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert result["geometry_reliable"]
    after = controller.views["wrist"]["field"].boxes[0][0]
    assert after-before == pytest.approx(np.array([3, 4])/95)
    result = controller.observe(images(missing=("wrist", "obstacle")), np.zeros(7), state8=state(),
                                executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert not result["geometry_reliable"]
    assert controller.views["wrist"]["field"] is None
    chosen = controller.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert chosen["status"] == "obstacle_tracking_unavailable" and not chosen["accepted"]


def test_textureless_obstacle_is_unknown_not_static_known_geometry():
    data = images(missing="obstacle")
    controller = VisualFieldController(grid_size=16)
    result = controller.configure(data, seeds(obstacles=True), np.zeros(7), state8=state())
    assert not result["geometry_reliable"]
    assert result["views"]["external"]["status"] == "obstacle_track_unavailable"


def test_stable_seed_ids_retain_fit_but_new_target_clears_it():
    controller, q, _ = calibrated()
    same = seeds()
    for view in same:
        same[view]["moving_point_uv"] = controller.views[view]["tracks"]["moving"].uv.tolist()
    result = controller.configure(images(q), same, q, state8=state(q))
    assert result["joint_to_image"]["reliable"]
    changed = copy.deepcopy(same)
    for view in changed:
        changed[view]["target_id"] = "different_object"
    result = controller.configure(images(q), changed, q, state8=state(q))
    assert not result["joint_to_image"]["reliable"]


def test_no_progress_and_measured_joint_limit_hold_translation():
    controller = VisualFieldController(grid_size=16, joint_lower=[-1]*7, joint_upper=[1]*7)
    controller.configure(images(), seeds(), np.zeros(7), state8=state())
    for _ in range(5):
        controller.observe(images(), np.zeros(7), state8=state(), executed_action=[.1, 0, 0, 0, 0, 0, 1])
    result = controller.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert result["status"] == "no_progress" and not result["accepted"]
    outside = np.zeros(7); outside[4] = 2.
    controller.configure(images(outside), seeds(), outside, state8=state(outside))
    assert controller.select_action([.1, 0, 0, 0, 0, 0, 1])["status"] == "measured_joint_limit"


def test_payload_comotion_requires_tool_track_motion_and_multiple_frames():
    controller = VisualFieldController(grid_size=16)
    q = np.zeros(7)
    controller.configure(images(payload=True), seeds(payload=True), q, state8=state())
    for _ in range(3):
        q[0] += .01
        feedback = controller.observe(images(q, payload=True), q, state8=state(q),
                                      executed_action=[.1, 0, 0, 0, 0, 0, 1])
    assert feedback["co_motion_evidence"]
    assert feedback["object_tool_relative_change_px"] == pytest.approx(0)
    assert feedback["task_success_certified"] is False
    feedback = controller.observe(images(q, payload=True, missing="tool"), q, state8=state(q),
                                  executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert not feedback["co_motion_evidence"]


def test_closure_or_static_images_alone_do_not_establish_attachment():
    controller = VisualFieldController(grid_size=16)
    controller.configure(images(payload=True), seeds(payload=True), np.zeros(7), state8=state())
    for _ in range(4):
        result = controller.observe(images(payload=True), np.zeros(7), state8=state(),
                                    executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert not result["co_motion_evidence"]


def test_payload_tool_tracks_remain_independent_of_lost_destination_track():
    controller = VisualFieldController(grid_size=16)
    q = np.zeros(7)
    controller.configure(images(payload=True), seeds(payload=True), q, state8=state())
    for _ in range(2):
        q[0] += .01
        feedback = controller.observe(images(q, payload=True, missing="target"), q, state8=state(q),
                                      executed_action=[.1, 0, 0, 0, 0, 0, 1])
    for view in feedback["views"].values():
        assert not view["reliable"]
        assert view["moving_track_reliable"] and view["tool_track_reliable"]
        assert view["moving_point_uv"][0] == pytest.approx(29/95)
    assert not controller.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]


def test_online_joint_prediction_mismatch_requests_refresh_before_refitting_can_hide_it():
    controller, q, _ = calibrated()
    changed = q.copy(); changed[3] += .1
    result = controller.observe(images(changed), changed, state8=state(changed),
                                executed_action=[.1, 0, 0, 0, 0, 0, 1])
    assert result["prediction_failure"] == "joint_prediction_inconsistent"
    decision = controller.select_action([.1, 0, 0, 0, 0, 0, 1])
    assert not decision["accepted"] and decision["need_refresh"]


def test_online_image_response_mismatch_cannot_remain_a_reliable_local_policy():
    controller, q, _ = calibrated()
    unexpected = q.copy(); unexpected[0] += .04
    result = controller.observe(images(unexpected), q, state8=state(q),
                                executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert result["prediction_failure"] == "image_prediction_inconsistent"
    assert not controller.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]


def test_obstacle_motion_response_learns_from_joint_feedback_not_static_boxes():
    controller = VisualFieldController(grid_size=16)
    q = np.zeros(7)
    controller.configure(images(), seeds(obstacles=True), q, state8=state())
    for command in ([.1, 0, 0], [0, .1, 0], [-.1, 0, 0], [0, -.1, 0], [.1, 0, 0], [0, .1, 0]):
        q[:3] += np.array(command)*.1
        shift = np.rint(q[:2]*100).astype(int)
        controller.observe(images(q, obstacle_shift=shift), q, state8=state(q),
                           executed_action=[*command, 0, 0, 0, 1])
    response = controller.views["wrist"]["boxes"][0]["response"]
    assert response.matrix is not None
    predicted = response.predict([.005, 0, 0, 0, 0, 0, 0])
    assert predicted == pytest.approx([.5, 0], abs=.01)
    reference = seeds(obstacles=True)
    for view in reference:
        reference[view]["moving_point_uv"] = controller.views[view]["tracks"]["moving"].uv.tolist()
    controller.configure(images(q, obstacle_shift=shift), reference, q, state8=state(q))
    assert controller.views["wrist"]["boxes"][0]["response"] is response
    assert controller.feedback()["views"]["wrist"]["obstacle_tracks"][0]["response_reliable"]
    feedback = controller.observe(images(q, obstacle_shift=shift+[7, 0]), q, state8=state(q),
        executed_action=[0, 0, 0, 0, 0, 0, 1])
    assert feedback["prediction_failure"] == "obstacle_prediction_inconsistent"
    assert not controller.select_action([.1, 0, 0, 0, 0, 0, 1])["accepted"]


@pytest.mark.parametrize("change", ["hidden_field", "duplicate_id", "out_of_range", "invalid_image", "bad_action"])
def test_strict_inputs_reject_hidden_or_malformed_data(change):
    data, reference = images(), seeds()
    action = [0]*7
    if change == "hidden_field":
        reference["external"]["ground_truth_pose"] = [0, 0, 0]
    elif change == "duplicate_id":
        reference["external"]["target_id"] = "tool"
    elif change == "out_of_range":
        reference["external"]["moving_point_uv"] = [1.1, 0]
    elif change == "invalid_image":
        data["external"] = data["external"].astype(float)
    else:
        action[0] = 1.1
    with pytest.raises(ValueError):
        controller = VisualFieldController(grid_size=16)
        controller.configure(data, reference, np.zeros(7), state8=state())
        controller.select_action(action)
