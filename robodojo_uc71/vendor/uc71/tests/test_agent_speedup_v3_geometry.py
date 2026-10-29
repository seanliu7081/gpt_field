"""Geometry evidence regressions; synthetic values are never live policy inputs."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v3_geometry import (
    GeometryMemory, MAX_SWEEP_INTERVALS, _box, _corners, _rotation, _separations,
    _union_coverage, obb_separation, validate_scene_update,
)


class VisualWatchFake:
    """Deterministic appearance events; does not claim synthetic RGB coverage."""
    def __init__(self):
        self.reports = {}
        self.captures = []

    def capture(self, ident, boxes, view, step, *, occluder_boxes=()):
        self.captures.append((ident, step))
        return {"status": "reference", "reason": "synthetic_reference", "geometry_verified": False}

    def assess(self, ident, boxes, view, step, *, occluder_boxes=()):
        return deepcopy(self.reports.get(ident, {"status": "consistent", "reason": "synthetic_consistency",
                                                "geometry_verified": False}))

    def forget(self, ids):
        pass


def geometry_memory(watch=None):
    return GeometryMemory(visual_watch=watch if watch is not None else VisualWatchFake())


def view(xyz=(0, 0, 0), rotvec=(0, 0, 0), gap=.08):
    state = np.array([*xyz, *rotvec, gap/2, -gap/2], float)
    angle = np.linalg.norm(rotvec)
    q = [1, 0, 0, 0] if angle == 0 else [math.cos(angle/2), *(np.asarray(rotvec)*math.sin(angle/2)/angle)]
    return SimpleNamespace(state=state, position=state[:3], R=_rotation(q), gap_mm=gap*1000,
                           observation=SimpleNamespace(geometry=None))


def entity(ident, role="obstacle", center=(0, 0, 0), half=(.02, .02, .02), **kwargs):
    result = {"id": ident, "role": role, "parent_id": None, "frame": "world", "position_m": list(center),
              "quaternion_wxyz": [1, 0, 0, 0], "boxes": [{"center_local_m": [0, 0, 0],
              "half_extents_m": list(half), "quaternion_wxyz": [1, 0, 0, 0]}], "uncertainty_m": 0.,
              "coverage": "visible_surfaces", "attachment": "none", "related_ids": [], "anchors": None}
    result.update(kwargs)
    return result


def update(*entities, invalidate_ids=(), remove_ids=()):
    return {"entities": list(entities), "invalidate_ids": list(invalidate_ids), "remove_ids": list(remove_ids)}


def context(intent="transit", **kwargs):
    result = {"intent": intent, "target_id": None, "held_object_id": None, "allowed_contact_ids": []}
    result.update(kwargs)
    return result


def test_schema_rejects_truth_fields_and_nonfinite_or_zero_shapes():
    for modification in ({"source": "measured"}, {"uncertainty_m": float("nan")}, {"quaternion_wxyz": [0, 0, 0, 0]}):
        with pytest.raises(ValueError):
            validate_scene_update(update(entity("a", **modification)))
    bad = entity("a")
    bad["boxes"][0]["half_extents_m"][0] = 0
    with pytest.raises(ValueError):
        validate_scene_update(update(bad))
    assert validate_scene_update(update()) == update()


def test_update_is_detached_normalized_and_selectively_invalidated():
    memory, v = geometry_memory(), view()
    incoming = update(entity("static", center=(2, 2, 2)), entity("target", "object", center=(.3, 0, 0)))
    memory.update_from_model(incoming, v, 0)
    incoming["entities"][0]["position_m"][0] = 999
    memory.update_from_model(update(invalidate_ids=["target"]), v, 4)
    assert memory.entities["static"]["position_m"][0] == 2
    assert not memory.entities["static"]["stale"]
    assert memory.entities["target"]["stale"]
    assert memory.entities["target"]["source"] == "model_visual_estimate"


def test_invalidation_uses_directional_physical_parent_not_semantic_cycles():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(
        entity("table", "support_surface", center=(0, 0, -.2), related_ids=["target"]),
        entity("target", "object", center=(.2, 0, 0), parent_id="table", related_ids=["table", "neighbor"]),
        entity("handle", "graspable_part", center=(.21, 0, 0), parent_id="target"),
        entity("neighbor", "object", center=(.5, 0, 0), related_ids=["table"]),
        entity("near_free", "free_region", center=(.2, 0, 0), half=(.06, .06, .06)),
        entity("remote_free", "free_region", center=(2, 0, 0), half=(.1, .1, .1), related_ids=["target"])), v, 0)
    memory._invalidate(["target"], 1, "release_slip_or_possible_contact", "possible_contact")
    assert {ident for ident, item in memory.entities.items() if item["stale"]} == {"target", "handle", "near_free"}
    assert memory.events[-1]["origin_ids"] == ["target"]
    assert memory.entities["handle"]["invalidation_origins"] == ["target"]
    memory._invalidate(["table"], 2, "visual_appearance_changed")
    assert memory.entities["handle"]["stale"]
    assert not memory.entities["neighbor"]["stale"]
    assert not memory.entities["remote_free"]["stale"]


def test_semantic_relation_edit_does_not_revise_spatial_evidence():
    memory, v = geometry_memory(), view()
    target = entity("target", "object", center=(.2, 0, 0))
    memory.update_from_model(update(target, entity("space", "free_region", half=(.5, .5, .5))), v, 0)
    target["related_ids"] = ["space"]
    memory.update_from_model(update(target), v, 10)
    assert not memory.entities["space"]["stale"]
    assert memory.entities["space"]["observation_step"] == 0
    assert not any(event["kind"] == "model_geometry_revision" for event in memory.events)


def test_unknown_geometry_gives_short_uncertain_continuation():
    result = geometry_memory().assess_motion(view(), [.2, 0, 0], None, context(), 0)
    assert result["status"] == "unknown"
    assert result["unknown_fraction"] == 1
    assert result["allowed_fraction"] == pytest.approx(.1)
    assert result["execution_status"] == "unknown"
    assert result["suggested_steps"] == 6
    assert result["translation_limit_m"] == .005
    assert result["missing_evidence_requests"]
    assert not result["clearance_is_physical_certificate"]
    assert not result["full_arm_checked"]


def test_contained_obstacle_and_rotated_sat_batch_agree():
    outer = _box([0, 0, 0], np.eye(3), [.1, .1, .1], "a", "a")
    inner = _box([0, 0, 0], _rotation([.9238795, 0, 0, .3826834]), [.01, .02, .01], "b", "b")
    separate = _box([.5, 0, 0], np.eye(3), [.01, .01, .01], "c", "c")
    batch = _separations(outer, np.array([inner["center"], separate["center"]]),
                         np.array([inner["rotation"], separate["rotation"]]),
                         np.array([inner["half"], separate["half"]]))
    assert batch[0] < 0 < batch[1]
    assert batch == pytest.approx([obb_separation(outer, inner), obb_separation(outer, separate)])


def test_sweep_catches_thin_midpath_obstacle():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("wall", center=(.15, 0, -.06), half=(.0002, .10, .10))), v, 0)
    result = memory.assess_motion(v, [.3, 0, 0], None, context(), 0)
    assert result["status"] == "blocked"
    assert result["allowed_fraction"] < .5
    assert "wall" in result["evidence_ids"]


def test_rotation_checks_whole_payload_not_only_endpoint():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("payload", "object", center=(.16, 0, 0), half=(.15, .01, .01), attachment="held"),
                                    entity("post", center=(.12, .12, 0), half=(.01, .01, .04))), v, 0)
    result = memory.assess_motion(v, [0, 0, 0], [math.sqrt(.5), 0, 0, math.sqrt(.5)],
                                  context(held_object_id="payload"), 0)
    assert result["status"] == "blocked"
    assert "post" in result["evidence_ids"]


def test_free_region_is_explicit_model_hypothesis():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("visible_space", "free_region", half=(.5, .5, .5))), v, 0)
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 0)
    assert result["status"] == "supported"
    assert result["allowed_fraction"] == 1
    assert not memory.summary()["physical_clearance_verified"]
    memory.update_from_model(update(invalidate_ids=["visible_space"]), v, 1)
    assert memory.assess_motion(v, [.1, 0, 0], None, context(), 1)["status"] == "unknown"


def test_held_transform_tracks_measured_hand_without_claiming_grasp():
    memory, before = geometry_memory(), view()
    memory.update_from_model(update(entity("cup", "object", center=(.1, 0, 0), attachment="held")), before, 0)
    after = view((1, 0, 0), (0, 0, math.pi/2))
    memory.observe_transition(before, after, 1, "close")
    assert memory.entities["cup"]["position_m"] == pytest.approx([1, .1, 0])
    assert memory.entities["cup"]["observation_step"] == 0
    assert not memory.entities["cup"]["physically_verified"]
    memory.observe_transition(after, after, 2, "open")
    assert memory.entities["cup"]["release_pending"]
    assert memory.entities["cup"]["stale"]
    assert memory.summary()["entities"][0]["release_pending"]
    result = memory.assess_motion(after, after.position+[.01, 0, 0], None, context(), 2)
    assert result["payload_geometry_unknown"]
    assert result["status"] == "unknown"
    # A new model pose/attachment estimate is not accepted visual release.
    # Keep actual-open provenance and the possible carried extent until the
    # independent interaction evidence passes geometry confirmation.
    memory.update_from_model(update(entity("cup", "object", center=(1, .1, 0))), after, 2)
    assert memory.entities["cup"]["attachment"] == "none"
    assert memory.entities["cup"]["position_m"] == pytest.approx([1, .1, 0])
    assert memory.entities["cup"]["observation_step"] == 2
    assert memory.entities["cup"]["release_pending"]
    assert memory.entities["cup"]["release_actual_command_step"] == 2
    bodies, payload_unknown = memory._local_bodies(after, {})
    assert payload_unknown and any(body["owner"] == "cup" for body in bodies)


def test_allowed_finger_contact_does_not_exempt_palm_or_other_object():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("target", "object", center=(0, 0, -.07), half=(.08, .08, .03))), v, 0)
    result = memory.assess_motion(v, [.001, 0, 0], None,
                                  context("grasp", target_id="target", allowed_contact_ids=["target", "unrelated"]), 0)
    assert result["status"] == "blocked"
    assert result["allowed_contact_ids"] == ["target"]


def test_escape_initial_overlap_requires_improvement_and_keeps_unknown():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("coarse_wall", center=(-.045, 0, -.07), half=(.02, .08, .06))), v, 0)
    result = memory.assess_motion(v, [.08, 0, 0], None, context("retreat"), 0)
    assert result["estimated_overlap_escape"]
    assert result["status"] == "unknown"
    assert result["allowed_fraction"] <= .25


def test_no_pair_correspondence_manufactures_no_triangulation():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("a", anchors={"external_uv": [10, 20], "wrist_uv": [20, 30], "same_feature": False})), v, 0)
    assert memory.entities["a"]["triangulation"]["status"] == "unavailable"


def test_stationary_observation_not_prohibited_by_coarse_overlap():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("coarse", center=(0, 0, -.07), half=(.08, .08, .03))), v, 0)
    result = memory.assess_motion(v, [0, 0, 0], None, context("observe"), 0)
    assert result["status"] == "unknown"
    assert result["allowed_fraction"] == 1
    assert result["reason"] == "stationary_observation_no_clearance_claim"


def test_commanded_gripper_checks_finger_sweep_without_extending_unknown_horizon():
    memory, v = geometry_memory(), view(gap=.08)
    memory.update_from_model(update(entity("inside_jaw", center=(0, .018, -.004), half=(.004, .003, .004))), v, 0)
    hold = memory.assess_motion(v, [0, 0, 0], None, context("observe"), 0)
    closing = memory.assess_motion(v, [0, 0, 0], None, context("grasp", commanded_gripper="close"), 0)
    assert hold["status"] == "unknown"
    assert closing["status"] == "blocked"
    assert closing["allowed_fraction"] == 0
    assert closing["suggested_steps"] <= 6
    assert closing["commanded_gripper_sweep"] == "close"


def test_opengl_pixel_rays_triangulate_without_overwriting_model_pose():
    memory, v = geometry_memory(), view()
    v.observation.geometry = {"image": {"width": 100, "height": 100}, "cameras": {
        "external": {"focal_length_px": [100, 100], "principal_point_px": [50, 50],
                     "position_world_m": [-.1, 0, 1], "rotation_world_from_camera": np.eye(3).tolist()},
        "wrist": {"focal_length_px": [100, 100], "principal_point_px": [50, 50],
                  "position_world_m": [.1, 0, 1], "rotation_world_from_camera": np.eye(3).tolist()},
    }}
    memory.update_from_model(update(entity("feature", center=(.01, .21, 0),
        anchors={"external_uv": [60, 30], "wrist_uv": [40, 30], "same_feature": True})), v, 0)
    record = memory.entities["feature"]
    assert record["triangulation"]["status"] == "conditional_estimate"
    assert record["triangulation"]["position_world_m"] == pytest.approx([0, .2, 0], abs=1e-12)
    assert record["position_m"] == [.01, .21, 0]
    assert not record["triangulation"]["verified_correspondence"]


def test_supported_far_approach_uses_clearance_speed_without_intent_wide_slowdown():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("current_corridor", "free_region", half=(.5, .5, .5))), v, 12)
    result = memory.assess_motion(v, [0, 0, -.20], None,
                                  context("approach", tracking={"status": "normal"}), 12)
    assert result["status"] == "supported"
    assert result["allowed_fraction"] == 1
    assert result["suggested_steps"] is None
    assert result["translation_step_m"] == .02
    assert not result["clearance_is_physical_certificate"]


def test_supported_transit_has_no_fixed_24_step_observation_timer():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("current_corridor", "free_region", half=(.5, .5, .5))), v, 0)
    result = memory.assess_motion(v, [.25, 0, 0], None, context(), 0)
    assert result["status"] == "supported"
    assert result["allowed_fraction"] == 1
    assert result["suggested_steps"] is None
    assert result["translation_step_m"] <= .02


def test_unrelated_old_regions_and_obstacles_do_not_age_the_current_route():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("old_basket_air", "free_region", center=(2, 2, 2)),
                                    entity("old_background", center=(3, 3, 3))), v, 0)
    memory.update_from_model(update(entity("current_corridor", "free_region", half=(.5, .5, .5))), v, 120)
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 120)
    assert result["status"] == "supported"
    assert result["evidence_ids"] == ["current_corridor"]
    assert result["max_relevant_evidence_age_steps"] == 0
    assert memory.entities["old_basket_air"]["observation_step"] == 0


def test_fresh_covering_region_supersedes_redundant_old_cover_without_erasing_it():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("old_corridor", "free_region", half=(.5, .5, .5))), v, 0)
    memory.update_from_model(update(entity("fresh_corridor", "free_region", half=(.5, .5, .5))), v, 100)
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 100)
    assert result["status"] == "supported"
    assert result["support_region_ids"] == ["fresh_corridor"]
    assert "old_corridor" not in result["evidence_ids"]
    assert memory.entities["old_corridor"]["observation_step"] == 0


def test_elapsed_steps_do_not_expire_unchanged_relevant_corridor():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("only_corridor", "free_region", half=(.5, .5, .5))), v, 0)
    memory.observe_transition(v, v, 100, "open")
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 100)
    memory.watch_observation(v, 101)
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 101)
    assert result["status"] == "supported"
    assert result["suggested_steps"] is None
    assert result["allowed_fraction"] == 1
    assert result["max_relevant_evidence_age_steps"] == 101
    assert memory.entities["only_corridor"]["observation_step"] == 0
    assert memory.entities["only_corridor"]["visual_checked_step"] == 101


def test_old_payload_remains_relevant_despite_fresh_free_space_and_robot_updates():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("cup", "object", center=(.08, 0, 0), attachment="held")), v, 0)
    moved = view((.01, 0, 0))
    memory.observe_transition(v, moved, 100, "close")
    memory.update_from_model(update(entity("current_corridor", "free_region", half=(.5, .5, .5))), moved, 100)
    result = memory.assess_motion(moved, [.1, 0, 0], None, context(held_object_id="cup"), 100)
    assert result["status"] == "supported"
    assert result["payload_evidence_ids"] == ["cup"]
    assert result["max_relevant_evidence_age_steps"] == 100
    assert memory.entities["cup"]["observation_step"] == 0


def test_unknown_contact_still_has_short_prefix_and_observation_budget():
    result = geometry_memory().assess_motion(view(), [0, 0, -.06], None, context("approach"), 0)
    assert result["status"] == "unknown"
    assert result["suggested_steps"] == 4
    assert result["allowed_fraction"] == pytest.approx(.25)
    assert result["translation_step_m"] == .005


def test_visual_change_invalidates_only_related_evidence_and_consistency_cannot_repair_it():
    watch, v = VisualWatchFake(), view()
    memory = geometry_memory(watch)
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("remote", center=(3, 3, 3))), v, 0)
    memory.assess_motion(v, [.1, 0, 0], None, context(), 0)
    watch.reports["corridor"] = {"status": "changed", "reason": "local_appearance_changed"}
    report = memory.watch_observation(v, 1)
    assert report["changed_ids"] == ["corridor"]
    assert memory.entities["corridor"]["stale"]
    assert not memory.entities["remote"]["stale"]
    watch.reports["corridor"] = {"status": "consistent", "reason": "appearance_consistent"}
    memory.watch_observation(v, 2)
    assert memory.entities["corridor"]["stale"]
    assert memory.entities["corridor"]["observation_step"] == 0
    assert memory.assess_motion(v, [.1, 0, 0], None, context(), 2)["status"] == "unknown"


@pytest.mark.parametrize("watched_role", ["free_region", "object"])
def test_expected_robot_occlusion_preserves_conditional_pose_prediction(watched_role):
    watch, v = VisualWatchFake(), view()
    memory = geometry_memory(watch)
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    watched_id = "corridor"
    if watched_role == "object":
        memory.update_from_model(update(entity("target", "object", center=(.3, 0, 0))), v, 0)
        watched_id = "target"
    memory.assess_motion(v, [.1, 0, 0], None, context(), 0)
    watch.reports[watched_id] = {"status": "unverified", "reason": "expected_self_occlusion", "self_occluded": True}
    memory.watch_observation(v, 500, [watched_id])
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 500)
    assert result["status"] == "supported"
    assert not memory.entities[watched_id]["monitoring_unknown"]
    assert not memory.entities[watched_id]["stale"]
    assert memory.entities[watched_id]["visual_monitoring"]["status"] == "unverified"
    assert memory.entities[watched_id]["observation_step"] == 0
    assert not result["clearance_is_physical_certificate"]


def test_unexpected_camera_loss_is_unknown_and_recovery_does_not_refresh_shape_age():
    watch, v = VisualWatchFake(), view()
    memory = geometry_memory(watch)
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    memory.assess_motion(v, [.1, 0, 0], None, context(), 0)
    watch.reports["corridor"] = {"status": "unverified", "reason": "missing_calibration"}
    memory.watch_observation(v, 1)
    result = memory.assess_motion(v, [.1, 0, 0], None, context(), 1)
    assert result["status"] == "unknown"
    assert result["monitoring_unknown_ids"] == ["corridor"]
    watch.reports["corridor"] = {"status": "consistent", "reason": "appearance_consistent"}
    memory.watch_observation(v, 2)
    assert memory.assess_motion(v, [.1, 0, 0], None, context(), 2)["status"] == "supported"
    assert memory.entities["corridor"]["observation_step"] == 0


def test_revised_object_invalidates_dependent_free_region_without_refreshing_it():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("object", "object", center=(.3, 0, 0)),
                                    entity("corridor", "free_region", half=(.5, .5, .5), related_ids=["object"])), v, 0)
    memory.update_from_model(update(entity("object", "object", center=(.32, 0, 0))), v, 10)
    assert memory.entities["corridor"]["stale"]
    assert memory.entities["corridor"]["observation_step"] == 0
    assert not memory.entities["object"]["stale"]


def test_tracking_mismatch_revokes_motion_without_rewriting_scene_evidence():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("remote", center=(3, 3, 3))), v, 0)
    result = memory.assess_motion(v, [.2, 0, 0], None, context(tracking={"status": "mismatch"}), 2)
    assert result["status"] == "blocked"
    assert result["reason"] == "measured_tracking_mismatch"
    assert result["allowed_fraction"] == 0
    assert result["motion_support_revoked"]
    assert not result["scene_invalidated_by_tracking"]
    assert not memory.entities["corridor"]["stale"]
    assert not memory.entities["remote"]["stale"]
    assert memory.entities["corridor"]["observation_step"] == 0
    assert memory.events[-1]["kind"] == "motion_support_revoked"
    assert memory.events[-1]["scope"] == "current_motion_only"
    # Observation remains available after a controller anomaly without claiming
    # the same untrusted motion can continue or certifying physical clearance.
    hold = memory.assess_motion(v, v.position, None, context("observe", tracking={"status": "mismatch"}), 3)
    assert hold["status"] == "unknown"
    assert hold["reason"] == "stationary_observation_no_clearance_claim"
    assert not hold["motion_support_revoked"]


def test_speed_changes_with_remaining_distance_and_not_low_normal_response_ratio():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    ctx = context("transit", tracking={"status": "normal", "translation_response_ratio": .12,
                                      "translation_response_estimate": .25,
                                      "translation_response_bounds": [.2, .3]})
    far = memory.assess_motion(v, [.2, 0, 0], None, ctx, 0)
    near = memory.assess_motion(v, [.01, 0, 0], None, ctx, 0)
    assert far["translation_step_m"] == .02
    assert near["translation_step_m"] == .01
    assert far["selected_command_candidate"]["expected_translation_progress_m"] == pytest.approx(.005)
    assert near["selected_command_candidate"]["expected_translation_progress_m"] == pytest.approx(.0025)
    assert all(candidate["command_translation_m"] <= .02 for candidate in far["command_candidates"])
    assert near["speed_evidence"]["command_units"] == "OSC_pose_error_increment_not_promised_physical_displacement"


def test_candidate_speed_obeys_controller_caps_and_empirical_overshoot_bound():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("space", "free_region", half=(.5, .5, .5))), v, 0)
    ctx = context(controller_translation_limit_m=.005, controller_rotation_limit_rad=.1,
                  tracking={"status": "normal", "translation_response_estimate": 1.5,
                            "translation_response_bounds": [1.2, 2.]})
    far = memory.assess_motion(v, [.2, 0, 0], None, ctx, 0)
    near = memory.assess_motion(v, [.006, 0, 0], None, ctx, 0)
    assert far["translation_step_m"] == .005
    assert near["translation_step_m"] == .003
    assert near["selected_command_candidate"]["expected_translation_progress_m"] == pytest.approx(.0045)
    assert near["selected_command_candidate"]["feasible"]


def test_supported_parallel_command_uses_padded_geometry_without_second_uncertainty_penalty():
    memory, v = geometry_memory(), view()
    # Close lateral wall, motion parallel to its face. The solid uncertainty
    # already expands its box; it must not multiply a second speed penalty.
    # The palm envelope reaches about 0.10 m laterally, wider than the fingers.
    # Leave roughly 2 cm after the wall uncertainty and interval sweep margin.
    wall = entity("wall", center=(0, .16, -.06), half=(.4, .005, .15), uncertainty_m=.03)
    memory.update_from_model(update(entity("space", "free_region", half=(.5, .5, .5)), wall,
                                    entity("remote", center=(3, 3, 3), uncertainty_m=.15)), v, 0)
    result = memory.assess_motion(v, [.2, 0, 0], None, context(tracking={"status": "normal"}), 0)
    assert result["status"] == "supported"
    assert 0 < result["speed_evidence"]["current_clearance_estimate_m"] < .04
    assert result["translation_step_m"] == .02
    assert result["limiting_pair"]["obstacle_id"] == "wall"
    assert result["limiting_pair"]["combined_extent_uncertainty_m"] >= .03
    assert result["speed_evidence"]["uncertainty_applied_once_to_shapes"]
    assert result["selected_command_candidate"]["feasible"]
    # The same uncertainty still matters geometrically when the wall is moved
    # into the envelope; removing a speed multiplier must not remove padding.
    wall["position_m"][1] = .065
    memory.update_from_model(update(wall, entity("space", "free_region", half=(.5, .5, .5))), v, 1)
    blocked = memory.assess_motion(v, [.2, 0, 0], None, context(tracking={"status": "normal"}), 1)
    assert blocked["status"] == "blocked"
    assert blocked["allowed_fraction"] == 0


@pytest.mark.parametrize("intent", ["approach", "grasp", "place"])
def test_final_interaction_blocks_missing_inventory_or_support_instead_of_unknown_crawl(intent):
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("target", "object", center=(.15, 0, 0))), v, 0)
    missing = memory.assess_motion(v, [.02, 0, 0], None, context(intent, target_id="target"), 0)
    assert missing["status"] == "blocked"
    assert missing["reason"] == "geometry_inventory_required"
    assert missing["allowed_fraction"] == 0
    assert "nearby_support_surface_for:target" in missing["missing_evidence_requests"]
    reviewed = context("grasp", target_id="target", scene_inventory={"reviewed": True, "unresolved": False,
                       "support_ids": ["corridor"], "nearby_obstacle_ids": []})
    wrong_role = memory.assess_motion(v, [.02, 0, 0], None, reviewed, 0)
    assert wrong_role["allowed_fraction"] == 0
    assert "current_support_ids:corridor" in wrong_role["missing_evidence_requests"]


def test_final_interaction_accepts_nearby_declared_support_but_not_remote_support():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("target", "object", center=(.15, 0, 0)),
                                    entity("support", "support_surface", center=(.15, 0, -.04), half=(.03, .03, .01))), v, 0)
    ctx = context("grasp", target_id="target", scene_inventory={"reviewed": True, "unresolved": False,
                  "support_ids": ["support"], "nearby_obstacle_ids": []})
    valid = memory.assess_motion(v, [.01, 0, 0], None, ctx, 0)
    assert valid["status"] == "supported"
    assert valid["missing_evidence_requests"] == []
    memory.update_from_model(update(entity("support", "support_surface", center=(2, 0, 0))), v, 1)
    remote = memory.assess_motion(v, [.01, 0, 0], None, ctx, 1)
    assert remote["status"] == "blocked"
    assert "nearby_support_surface_for:target" in remote["missing_evidence_requests"]


def test_far_motion_speed_uses_current_clearance_while_sweep_still_checks_narrow_destination():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("wall", center=(.2, 0, -.06), half=(.005, .3, .15))), v, 0)
    result = memory.assess_motion(v, [.15, 0, 0], None, context(tracking={"status": "normal"}), 0)
    assert result["status"] == "supported"
    assert 0 < result["clearance_estimate_m"] < .02
    assert result["speed_evidence"]["current_clearance_estimate_m"] > .1
    assert result["translation_step_m"] == .02


def test_newly_relevant_old_geometry_is_watched_on_segment_entry():
    watch, v = VisualWatchFake(), view()
    memory = geometry_memory(watch)
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("wall", center=(.2, 0, -.06), half=(.005, .3, .15))), v, 0)
    first = memory.assess_motion(v, [.01, 0, 0], None, context(), 0)
    assert "wall" not in first["evidence_ids"]
    watch.reports["wall"] = {"status": "changed", "reason": "local_appearance_changed"}
    # Root has already watched the previous segment at this observation. The
    # new segment must additionally check its own wall before claiming support.
    memory.watch_observation(v, 1)
    result = memory.assess_motion(v, [.15, 0, 0], None, context(), 1)
    # This newly observed change also adds the existing 2cm stale-shape
    # envelope. Recompute the sweep in this same assessment: its endpoint now
    # intersects that envelope. The old unknown verdict used pre-watch boxes.
    assert result["status"] == "blocked"
    assert result["reason"] == "estimated_obstacle_overlap"
    assert result["allowed_fraction"] < 1
    assert memory.entities["wall"]["stale"]
    repeated = memory.assess_motion(v, [.15, 0, 0], None, context(), 1)
    assert repeated["status"] == result["status"]
    assert repeated["allowed_fraction"] == result["allowed_fraction"]
    assert not result["clearance_is_physical_certificate"]


def test_possible_contact_with_declared_obstacle_is_an_invalidation_event():
    memory, before = geometry_memory(), view()
    memory.update_from_model(update(entity("milk", "obstacle", center=(0, 0, -.07), half=(.04, .04, .03))), before, 0)
    memory.observe_transition(before, view((.001, 0, 0)), 1, "open")
    assert memory.entities["milk"]["stale"]
    assert memory.entities["milk"]["stale_reason"] == "release_slip_or_possible_contact"
    assert not memory.entities["milk"]["physically_verified"]


def test_same_sample_tracking_failure_blocks_final_interaction_without_staling_inventory():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("target", "object", center=(.15, 0, 0)),
                                    entity("support", "support_surface", center=(.15, 0, -.04), half=(.03, .03, .01))), v, 0)
    ctx = context("grasp", target_id="target", tracking={"status": "mismatch"},
                  scene_inventory={"reviewed": True, "unresolved": False,
                                   "support_ids": ["support"], "nearby_obstacle_ids": []})
    result = memory.assess_motion(v, [.01, 0, 0], None, ctx, 1)
    assert result["status"] == "blocked"
    assert result["allowed_fraction"] == 0
    assert result["reason"] == "measured_tracking_mismatch"
    assert result["missing_evidence_requests"] == []
    assert all(not item["stale"] for item in memory.entities.values())


def test_transition_is_idempotent_across_primitive_and_policy_observers():
    memory, before = geometry_memory(), view()
    memory.update_from_model(update(entity("cup", "object", center=(.08, 0, 0), attachment="held"),
                                    entity("corridor", "free_region", half=(.5, .5, .5), related_ids=["cup"])), before, 0)
    memory.assess_motion(before, [.1, 0, 0], None, context(held_object_id="cup"), 0)
    after = view((.01, 0, 0))
    # An earlier watcher writes memory.step, but must not consume the transition.
    memory.watch_observation(after, 1)
    first = memory.observe_transition(before, after, 1, "open")
    uncertainty = memory.entities["cup"]["uncertainty_m"]
    events = deepcopy(memory.events)
    assert uncertainty == pytest.approx(.0001)
    assert memory.entities["cup"]["release_pending"]
    assert memory.entities["corridor"]["stale"]
    memory.assess_motion(after, [.1, 0, 0], None, context(held_object_id="cup"), 1)
    second = memory.observe_transition(before, after, 1, "open")
    assert second == first
    assert memory.entities["cup"]["uncertainty_m"] == uncertainty
    assert memory.events == events
    first["invalidated_ids"].clear()
    assert memory.observe_transition(before, after, 1, "open")["invalidated_ids"] == ["cup"]
    memory.observe_transition(after, view((.02, 0, 0)), 2, "open")
    assert memory.entities["cup"]["uncertainty_m"] == pytest.approx(2*uncertainty)


@pytest.mark.parametrize("command", ["close", "open"])
def test_checked_stationary_gripper_can_settle_expected_contact_or_release_only(command):
    memory, before = geometry_memory(), view(gap=.08 if command == "close" else .02)
    target = entity("target", "object" if command == "close" else "destination_region",
                    center=(0, 0, 0) if command == "close" else (.08, 0, 0), half=(.005, .015, .004))
    rows = [entity("corridor", "free_region", half=(.5, .5, .5)), target,
            entity("support", "support_surface", center=(.06, 0, 0), half=(.005, .005, .005)),
            entity("neighbor", center=(.12, 0, 0), half=(.005, .005, .005))]
    if command == "open":
        rows.append(entity("payload", "object", half=(.005, .015, .004), attachment="held"))
    memory.update_from_model(update(*rows), before, 0)
    ctx = context("grasp" if command == "close" else "place", target_id="target",
                  held_object_id="payload" if command == "open" else None,
                  allowed_contact_ids=["target"] if command == "close" else [],
                  gripper_action=True, commanded_gripper=command,
                  scene_inventory={"reviewed": True, "unresolved": False,
                                   "support_ids": ["support"], "nearby_obstacle_ids": ["neighbor"]})
    entry = memory.assess_motion(before, [0, 0, 0], None, ctx, 0)
    assert entry["status"] == "supported"
    authorization = deepcopy(entry["gripper_settling_reference"])
    authorization.update(entry_step=0, max_steps=16)
    ctx["gripper_settling_authorization"] = authorization
    # Maintaining the same hold target requires a correction after measured
    # robot drift. A 0.2mm correction is within the existing pose tolerance.
    after = view((.0002, 0, 0), gap=.02 if command == "close" else .08)
    memory.observe_transition(before, after, 1, command)
    settling = memory.assess_motion(after, [0, 0, 0], None, ctx, 1)
    expected_id = "target" if command == "close" else "payload"
    assert memory.entities[expected_id]["stale"]
    assert expected_id in settling["gripper_settling_expected_ids"]
    assert settling["status"] == "unknown"
    assert settling["allowed_fraction"] == 1
    assert settling["inventory_missing_evidence_requests"] == []
    assert not settling["clearance_is_physical_certificate"]
    actual = memory.verify_command(after, [-.004, 0, 0, 0, 0, 0, 1 if command == "close" else -1], ctx, 1)
    assert actual["command_allowed"]
    assert actual["command_settling_authorized"]
    assert not actual["command_supported"]
    assert not memory.entities["corridor"]["stale"]
    assert expected_id in memory.entities["corridor"].get("local_uncertain_solid_ids", [])
    # This token does not authorize another pose, and does not refresh evidence.
    assert not memory._settling_ids(after, np.array([0, 0, .02]), np.array([1, 0, 0, 0]), ctx, 2)
    assert memory.entities[expected_id]["observation_step"] == 0
    # An unrelated observed change cannot use the expected-contact exception.
    memory._invalidate(["neighbor"], 2, "visual_appearance_changed")
    changed = memory.assess_motion(after, [0, 0, 0], None, ctx, 2)
    assert changed["status"] == "blocked"
    assert changed["allowed_fraction"] == 0
    assert "current_nearby_obstacle_ids:neighbor" in changed["missing_evidence_requests"]
    rejected = memory.verify_command(after, [0, 0, 0, 0, 0, 0, 1 if command == "close" else -1], ctx, 2)
    assert not rejected["command_allowed"]


def test_full_goal_unknown_keeps_actionable_witness_and_fixed_supported_prefix():
    memory, v = geometry_memory(), view()
    bodies, _ = memory._local_bodies(v, context())
    corners = np.concatenate([_corners(body) for body in bodies])
    low, high = corners.min(axis=0)-.02, corners.max(axis=0)+.08
    region = entity("observed_corridor", "free_region", center=(low+high)/2, half=(high-low)/2)
    memory.update_from_model(update(region), v, 0)
    result = memory.assess_motion(v, [.2, 0, 0], None, context(), 0)
    assert result["status"] == result["whole_goal_assessment"]["status"] == "unknown"
    assert result["whole_goal_assessment"]["xyz_m"] == [.2, 0, 0]
    assert result["execution_status"] == result["execution_prefix"]["status"] == "supported"
    assert 0 < result["allowed_fraction"] < 1
    assert result["suggested_steps"] is None
    assert result["supported_prefix_fraction"] < result["first_uncovered_fraction"]
    assert result["coverage_witnesses"]
    witness = result["coverage_witnesses"][0]
    assert witness["body_part"]
    assert witness["region_id"] == "observed_corridor"
    assert witness["axis"] == "x"
    assert witness["face"] == "positive"
    assert len(witness["world_point_m"]) == 3
    assert witness["point_outside_eroded_region_union"]
    assert witness["padded_margin_m"] < 0
    assert result["missing_evidence_requests"][0].startswith("sweep_coverage:")
    assert result["inventory_missing_evidence_requests"] == []
    prefix = memory.preview_motion(v, result["execution_prefix"]["xyz_m"],
                                   result["execution_prefix"]["quaternion_wxyz"], context(), 0)
    assert prefix["status"] == "supported"
    assert not prefix["clearance_is_physical_certificate"]


def test_interpolation_refinement_preserves_body_and_region_uncertainty():
    memory, v = geometry_memory(), view()
    body = _box([0, 0, 0], np.eye(3), [.01, .01, .01], "payload", "payload_box", .002)
    free = _box([0, 0, 0], np.eye(3), [.4, .0237, .1], "free", "free_box", .01)
    result = memory._sweep_geometry(v, np.array([.12, 0, 0]), np.array([1., 0, 0, 0]),
                                   np.array([1., 0, 0, 0]), [body], [], [free], set(), .02)
    assert result["refinement"]["refinements"] >= 1
    assert result["refinement"]["attempts"][0]["uncovered_samples"] > 0
    assert result["unknown_count"] == 0
    assert result["supported_prefix_fraction"] == 1
    assert body["uncertainty"] == .002 and free["uncertainty"] == .01
    assert not result["refinement"]["model_uncertainty_reduced"]
    # Real model uncertainty cannot disappear by using more samples.
    free["half"][1] = .021
    uncertain = memory._sweep_geometry(v, np.array([.12, 0, 0]), np.array([1., 0, 0, 0]),
                                      np.array([1., 0, 0, 0]), [body], [], [free], set(), .02)
    assert uncertain["unknown_count"] > 0
    assert uncertain["supported_prefix_fraction"] == 0
    assert uncertain["coverage_witnesses"][0]["coverage_kind"] == "uncertainty_boundary"
    assert uncertain["coverage_witnesses"][0]["nominal_margin_m"] > 0
    assert uncertain["coverage_witnesses"][0]["uncertainty_margin_m"] < 0


def test_boundary_refinement_has_a_hard_work_bound():
    memory, v = geometry_memory(), view()
    body = _box([0, 0, 0], np.eye(3), [.01, .01, .01], "payload", "payload_box")
    free = _box([0, 0, 0], np.eye(3), [1., .0100001, .1], "free", "free_box")
    result = memory._sweep_geometry(v, np.array([.6, 0, 0]), np.array([1., 0, 0, 0]),
                                   np.array([1., 0, 0, 0]), [body], [], [free], set(), .02)
    assert result["intervals"] <= MAX_SWEEP_INTERVALS
    assert len(result["refinement"]["attempts"]) <= 4
    assert result["unknown_count"] > 0
    assert result["refinement"]["still_interpolation_limited"]
    assert result["coverage_witnesses"][0]["coverage_kind"] == "interpolation_bound"


def test_collision_interpolation_can_refine_but_true_overlap_remains():
    memory, v = geometry_memory(), view()
    body = _box([0, 0, 0], np.eye(3), [.01, .01, .01], "payload", "payload_box", .002)
    free = _box([0, 0, 0], np.eye(3), [1., 1., 1.], "free", "free_box")
    wall = _box([0, .0247, 0], np.eye(3), [.5, .01, .1], "wall", "wall_box", .001)
    args = (v, np.array([.12, 0, 0]), np.array([1., 0, 0, 0]), np.array([1., 0, 0, 0]), [body])
    result = memory._sweep_geometry(*args, [wall], [free], set(), .02)
    assert result["refinement"]["attempts"][0]["overlap_samples"] > 0
    assert not any(row[3] for row in result["rows"])
    assert result["supported_prefix_fraction"] == 1
    wall["center"][1] -= .003
    overlap = memory._sweep_geometry(*args, [wall], [free], set(), .02)
    assert any(row[3] for row in overlap["rows"])
    assert overlap["supported_prefix_fraction"] == 0


def test_preview_update_and_motion_do_not_mutate_live_evidence_or_capture_rgb():
    watch, v = VisualWatchFake(), view()
    memory = geometry_memory(watch)
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    before, captures = memory.snapshot(), deepcopy(watch.captures)
    preview = memory.preview_scene_update(update(entity("new_corridor", "free_region", center=(1, 0, 0),
                                                       half=(.5, .5, .5))), v, 1)
    assert "new_corridor" in preview.entities
    assert "new_corridor" not in memory.entities
    result = memory.preview_motion(v, [.1, 0, 0], None, context(tracking={"status": "mismatch"}), 1)
    assert result["preview_only"]
    assert result["status"] == "blocked"
    assert memory.snapshot() == before
    assert watch.captures == captures


def test_command_verifier_requires_entire_exact_command_not_only_its_prefix():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5)),
                                    entity("wall", center=(.06, 0, -.06), half=(.001, .3, .15))), v, 0)
    safe = memory.verify_command(v, [.05, 0, 0, 0, 0, 0, 0], context(), 0)
    assert safe["command_allowed"] and safe["command_supported"]
    blocked = memory.verify_command(v, [1., 0, 0, 0, 0, 0, 0], context(), 0, allow_unknown=True)
    assert blocked["execution_status"] == "supported"
    assert 0 < blocked["execution_prefix"]["fraction"] < 1
    assert blocked["status"] == "blocked"
    assert not blocked["command_allowed"]
    assert blocked["exact_command_xyz_m"] == pytest.approx([.05, 0, 0])


def test_exact_command_unknown_permission_does_not_claim_support():
    memory, v = geometry_memory(), view()
    action = [.1, 0, 0, 0, 0, 0, 0]
    assert not memory.verify_command(v, action, context(), 0)["command_allowed"]
    result = memory.verify_command(v, action, context(), 0, allow_unknown=True)
    assert result["command_allowed"]
    assert not result["command_supported"]
    assert result["status"] == "unknown"
    assert result["missing_evidence_requests"]
    hold = memory.verify_command(v, [0, 0, 0, 0, 0, 0, 0], context("observe"), 0)
    assert hold["command_allowed"] and hold["command_observation_hold_allowed"]
    assert not hold["command_supported"]


def test_short_command_uses_original_goal_for_final_interaction_guard():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    ctx = context("approach", original_goal_xyz_m=[.2, 0, 0])
    command = memory.verify_command(v, [.1, 0, 0, 0, 0, 0, 0], ctx, 0)
    assert command["command_supported"]
    assert not command["final_interaction_guard"]
    final = memory.verify_command(v, [.1, 0, 0, 0, 0, 0, 0],
                                  context("approach", original_goal_xyz_m=[.01, 0, 0]), 0)
    assert final["final_interaction_guard"]
    assert not final["command_allowed"]
    assert final["inventory_missing_evidence_requests"]


def test_command_residual_is_separate_from_persistent_controller_risk_limit():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    result = memory.assess_motion(v, [.00004, 0, 0], None, context(tracking={"status": "normal"}), 0)
    assert result["translation_step_m"] == pytest.approx(.00004)
    assert result["translation_limit_m"] == .02
    assert result["selected_command_candidate"]["translation_delta_m"] == pytest.approx([.00004, 0, 0])


def test_union_coverage_proves_cells_and_preserves_uncertain_gap():
    body = _box([0, 0, 0], np.eye(3), [.12, .02, .02], "hand", "palm")
    low = _box([0, 0, -.02], np.eye(3), [.2, .1, .0211], "low", "low:0", .001)
    high = _box([0, 0, .02], np.eye(3), [.2, .1, .0211], "high", "high:0", .001)
    covered, owners, cells = _union_coverage(body, [low, high])
    assert covered and owners == {"low", "high"} and cells <= 64
    # Eroding both estimates farther creates a real unknown gap at the seam.
    low["uncertainty"] = high["uncertainty"] = .002
    assert not _union_coverage(body, [low, high])[0]


def test_supported_route_reuse_requires_unused_margin_and_unchanged_scene():
    memory, before = geometry_memory(), view()
    memory.update_from_model(update(entity("air", "free_region", half=(.5, .5, .5)),
                                    entity("remote", center=(4, 0, 0))), before, 0)
    first = memory.assess_motion(before, [.2, 0, 0], None, context(), 0)
    assert first["status"] == "supported"
    assert first["geometry_computation"]["obstacle_boxes_checked"] == 0
    after = view((.02, .001, 0))
    reused = memory.assess_motion(after, [.2, 0, 0], [1, 0, 0, 0], context(), 1)
    assert reused["status"] == "supported"
    assert reused["geometry_computation"]["supported_route_reuse"]["remaining_margin_m"] > 0
    assert memory.entities["air"]["observation_step"] == 0
    # A new obstacle invalidates the cached route; final commands never inherit
    # a proof from a changed scene or merely from a distant original endpoint.
    memory.update_from_model(update(entity("new_wall", center=(.12, 0, -.06), half=(.001, .3, .15))), after, 2)
    changed = memory.assess_motion(after, [.2, 0, 0], [1, 0, 0, 0], context(), 2)
    assert changed["status"] == "blocked"
    assert changed["geometry_computation"]["supported_route_reuse"] is None
    assert not memory.verify_command(view((.08, 0, 0)), [1, 0, 0, 0, 0, 0, 0], context(), 2)["command_allowed"]


def test_free_region_appearance_change_does_not_cascade_to_other_free_regions():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("first", "free_region", half=(.5, .5, .5)),
                                    entity("second", "free_region", half=(.5, .5, .5))), v, 0)
    memory._invalidate(["first"], 1, "visual_appearance_changed")
    assert memory.entities["first"]["stale"]
    assert not memory.entities["second"]["stale"]
    assert memory.entities["second"]["observation_step"] == 0


def test_probe_requires_matching_fresh_two_view_motion_and_keeps_extent_uncertainty():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("target", "object", center=(.08, 0, 0), uncertainty_m=.01),
                                    entity("air", "free_region", half=(.5, .5, .5))), v, 0)
    probe = memory.begin_attachment_probe("target", v, 2)
    assert probe["attachment_probe_shape_usable"]
    assert memory._local_bodies(v, context(held_object_id="target"))[1]
    good = {"source": "public_two_view_feature_motion", "status": "supported", "kind": "hold",
            "target_id": "target", "observed_at_step": 4, "reference_step": 1,
            "views": {"external": {"status": "supported"}, "wrist": {"status": "supported"}}}
    bad = deepcopy(good); bad["views"]["wrist"]["status"] = "unknown"
    assert not memory.confirm_attachment_evidence(bad, v, 4)["accepted"]
    assert memory.confirm_attachment_evidence(good, v, 4)["accepted"]
    assert not memory._local_bodies(v, context(held_object_id="target"))[1]
    assert memory.entities["target"]["uncertainty_m"] == .01
    assert memory.entities["target"]["observation_step"] == 0
    memory.begin_release_probe("target", v, 5)
    # Registering a camera baseline alone must not claim physical opening.
    assert not memory.entities["target"].get("release_pending", False)
    memory.observe_transition(v, v, 6, "open")
    assert memory.entities["target"]["release_pending"]
    release = {**good, "kind": "release", "reference_step": 5, "observed_at_step": 7}
    assert memory.confirm_attachment_evidence(release, v, 7)["accepted"]
    assert memory.entities["target"]["attachment"] == "none"
    assert not memory.entities["target"]["release_pending"]
    assert memory.entities["target"]["stale"]  # No absolute release pose was measured.


def local_displacement(ident, revision, step, delta=(.02, 0, 0)):
    return {"entity_id": ident, "status": "supported", "source": "public_two_view_feature_displacement",
            "observed_at_step": step, "reference_step": 0, "reference_geometry_revision": revision,
            "evidence_id": f"{ident}:{revision}:0:{step}", "reason": "two_view_rigid_material_displacement",
            "world_rotation_delta": np.eye(3).tolist(), "world_translation_delta_m": list(delta),
            "position_uncertainty_m": .003, "orientation_uncertainty_rad": .01,
            "views": {name: {"status": "supported"} for name in ("external", "wrist")}}


def test_fresh_local_pose_updates_preserve_shape_uncertainty_and_localize_invalidity():
    memory, v = geometry_memory(), view()
    memory.update_from_model(update(entity("target", "object", center=(.3, 0, 0), uncertainty_m=.01),
        entity("part", "graspable_part", center=(.31, 0, 0), parent_id="target"),
        entity("corridor", "free_region", half=(.6, .6, .6)),
        entity("neighbor", center=(0, .4, 0))), v, 0)
    previous_boxes = deepcopy(memory.entities["target"]["boxes"])
    result = memory.apply_visual_displacement(local_displacement("target", 1, 4), v, 4)
    assert result["accepted"]
    target = memory.entities["target"]
    assert target["position_m"] == pytest.approx([.32, 0, 0])
    assert target["boxes"] == previous_boxes
    assert target["uncertainty_m"] >= .013
    assert target["observation_step"] == 0
    assert target["pose_observation_step"] == 4 and target["local_pose_revision"] == 1
    assert target["local_pose_evidence"]["accepted"]
    assert not memory.entities["corridor"]["stale"]
    assert not memory.entities["neighbor"]["stale"]
    assert memory.entities["part"]["stale"]
    assert target["local_motion_exclusion_boxes"]
    assert memory.assess_motion(v, [-.1, 0, 0], None, context(), 4)["status"] == "supported"
    assert not memory.apply_visual_displacement(local_displacement("target", 1, 4), v, 4)["accepted"]


def test_local_pose_requires_current_two_view_revision_and_does_not_renew_on_model_timestamp():
    memory, v = geometry_memory(), view()
    obj = entity("target", "object", center=(.3, 0, 0))
    memory.update_from_model(update(obj), v, 0)
    for bad in (local_displacement("target", 2, 1), local_displacement("target", 1, 0),
                {**local_displacement("target", 1, 1), "views": {"external": {"status": "supported"}}},
                {**local_displacement("target", 1, 1), "status": "consistent"}):
        assert not memory.apply_visual_displacement(bad, v, 1)["accepted"]
    assert memory.entities["target"]["local_pose_revision"] == 0
    assert memory.apply_visual_displacement(local_displacement("target", 1, 1), v, 1)["accepted"]
    memory.update_from_model(update(obj), v, 2)
    assert memory.entities["target"]["local_pose_revision"] == 1
    assert memory.entities["target"]["local_pose_evidence"] is None


def test_stationary_open_does_not_require_airborne_payload_to_touch_support():
    memory, v = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("payload", "object", half=(.005, .008, .004), attachment="held"),
        entity("corridor", "free_region", half=(.5, .5, .5)),
        entity("support", "support_surface", center=(0, 0, -.3), half=(.1, .1, .01))), v, 0)
    ctx = context("place", target_id="payload", held_object_id="payload", allowed_contact_ids=["payload"],
                  gripper_action=True, commanded_gripper="open",
                  scene_inventory={"reviewed": True, "unresolved": False,
                                   "support_ids": ["support"], "nearby_obstacle_ids": []})
    opened = memory.assess_motion(v, v.position, None, ctx, 0)
    assert opened["status"] == "supported"
    assert opened["gripper_settling_reference"] is not None
    assert opened["inventory_missing_evidence_requests"] == []
    moving = memory.assess_motion(v, [0, 0, -.02], None, ctx, 0)
    assert "nearby_support_surface_for:payload" in moving["inventory_missing_evidence_requests"]
    palm = next(body for body in memory._local_bodies(v, {})[0] if not body["name"].startswith("finger"))
    memory.update_from_model(update(entity("neighbor", center=palm["center"], half=(.004, .004, .004))), v, 1)
    assert memory.assess_motion(v, v.position, None, ctx, 1)["status"] == "blocked"


def test_opening_detached_bound_target_keeps_finger_contact_exception_before_token():
    memory, v = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("target", "object", half=(.005, .015, .004)),
        entity("destination", "destination_region", center=(.2, 0, 0)),
        entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    ctx = context("place", target_id="destination", held_object_id="target", allowed_contact_ids=["target"],
                  gripper_action=True, commanded_gripper="open",
                  scene_inventory={"reviewed": True, "unresolved": False,
                                   "support_ids": [], "nearby_obstacle_ids": []})
    result = memory.assess_motion(v, v.position, None, ctx, 0)
    assert result["status"] == "unknown"  # Model disagreement about attachment remains explicit.
    assert "target" in result["allowed_contact_ids"]
    assert result["reason"] != "estimated_obstacle_overlap"
    assert result["inventory_missing_evidence_requests"] == []


def test_finite_dual_hypothesis_lift_probe_retains_unknown_attachment_without_forced_handoff():
    memory, v = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("payload", "object", half=(.005, .008, .004)),
        entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    memory.begin_attachment_probe("payload", v, 1)
    ctx = context("retreat", held_object_id="payload", allowed_contact_ids=["payload"],
                  commanded_gripper="close", attachment_probe_authorized=True,
                  attachment_probe_target_id="payload", attachment_probe_start_step=1, attachment_probe_max_steps=20)
    probe = memory.assess_motion(v, [0, 0, -.04], None, ctx, 2)
    assert probe["status"] == "supported"
    assert probe["suggested_steps"] is None
    assert probe["conditional_probe_geometry_supported"]
    assert probe["attachment_state_unknown"] and not probe["attachment_verified"]
    assert probe["attachment_probe_hypotheses"] == ["world_stationary", "hand_following"]
    expired = memory.assess_motion(v, [0, 0, -.04], None, ctx, 21)
    assert expired["status"] == "unknown" and expired["suggested_steps"] is not None
    assert memory.entities["payload"]["attachment_probe_unconfirmed"]


def test_authorized_lift_probe_keeps_public_grasp_intent_but_checks_declared_neighbors():
    memory, v = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("payload", "object", half=(.005, .008, .004)),
        entity("neighbor", center=(.3, .3, 0)), entity("corridor", "free_region", half=(.5, .5, .5))), v, 0)
    memory.begin_attachment_probe("payload", v, 1)
    ctx = context("grasp", target_id="payload", held_object_id="payload", allowed_contact_ids=["payload"],
        commanded_gripper="close", attachment_probe_authorized=True, attachment_probe_target_id="payload",
        attachment_probe_start_step=1, attachment_probe_max_steps=20,
        scene_inventory={"reviewed": True, "unresolved": False, "support_ids": [], "nearby_obstacle_ids": ["neighbor"]})
    checked = memory.assess_motion(v, [0, 0, -.04], None, ctx, 2)
    assert checked["status"] == "supported"
    assert ctx["intent"] == "grasp"
    memory._invalidate(["neighbor"], 3, "visual_appearance_changed")
    blocked = memory.assess_motion(v, [0, 0, -.04], None, ctx, 3)
    assert blocked["status"] == "blocked"
    assert "current_nearby_obstacle_ids:neighbor" in blocked["inventory_missing_evidence_requests"]


def test_actual_release_epoch_and_possible_payload_survive_model_entity_replacement():
    memory, before = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("payload", "object", center=(.08, 0, 0),
                                    half=(.025, .02, .02), attachment="held", uncertainty_m=.01)), before, 0)
    memory.begin_release_probe("payload", before, 3)
    opened = view(gap=.04)
    memory.observe_transition(before, opened, 4, "open")
    original = deepcopy(memory.entities["payload"])
    assert original["release_actual_command_step"] == 4
    assert original["release_requested_step"] == 3  # Actual pre-open image baseline.
    # The model now believes the object is stationary and has revised its pose
    # and size. Its label is not physical release evidence or a new open event.
    revised = entity("payload", "object", center=(.10, 0, -.01),
                     half=(.01, .01, .01), attachment="none", uncertainty_m=.005)
    memory.update_from_model(update(revised), opened, 8)
    pending = memory.entities["payload"]
    assert pending["attachment"] == "none" and pending["release_pending"]
    assert pending["position_m"] == revised["position_m"]
    assert pending["observation_step"] == 8
    assert pending["release_actual_command_step"] == 4
    assert pending["release_requested_step"] == 3
    bodies, unknown = memory._local_bodies(opened, {})
    assert unknown and any(b["owner"] == "payload" for b in bodies)
    carried = pending["release_pending_carried_envelope"]
    assert carried["half"][0] >= original["boxes"][0]["half_extents_m"][0]
    assert carried["uncertainty"] >= original["uncertainty_m"]
    assert pending["release_pending_world_envelope"]
    unsupported = {"kind": "release", "target_id": "payload", "source": "public_two_view_feature_motion",
        "status": "unknown", "reference_step": 3, "observed_at_step": 9,
        "views": {name: {"status": "supported"} for name in ("external", "wrist")}}
    assert not memory.confirm_attachment_evidence(unsupported, opened, 9)["accepted"]
    assert pending["release_pending"]
    accepted = memory.confirm_attachment_evidence({**unsupported, "status": "supported"}, opened, 9)
    assert accepted["accepted"]
    assert not pending["release_pending"]
    assert pending["position_m"] == revised["position_m"]
    assert pending["observation_step"] == 8  # Visual stationarity does not refresh pose.
    assert pending["stale"]  # Current metric pose still only a model hypothesis.
    assert not any(b["owner"] == "payload" for b in memory._local_bodies(opened, {})[0])
    assert pending["local_motion_exclusion_boxes"]  # Keep uncertain world occupancy.


def test_model_detachment_without_physical_open_cannot_manufacture_release_epoch():
    memory, v = geometry_memory(), view(gap=.02)
    memory.update_from_model(update(entity("payload", "object", center=(.08, 0, 0), attachment="held")), v, 0)
    memory.begin_release_probe("payload", v, 2)
    memory.update_from_model(update(entity("payload", "object", center=(.08, 0, 0), attachment="none")), v, 3)
    report = {"kind": "release", "target_id": "payload", "source": "public_two_view_feature_motion",
        "status": "supported", "reference_step": 2, "observed_at_step": 4,
        "views": {name: {"status": "supported"} for name in ("external", "wrist")}}
    assert not memory.confirm_attachment_evidence(report, v, 4)["accepted"]
    assert "release_actual_command_step" not in memory.entities["payload"]
