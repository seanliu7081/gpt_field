from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.contact_skills import StepView
from gpt_field.agent_speedup_v3_geometry import _box, _rotation
from gpt_field.agent_speedup_v3_program import (
    AdaptiveProgramPrimitive, MissingGeometry, derive_program_goal, program_schema, validate_program,
)


def view(x=0., z=.8, gap=.08):
    return StepView(SimpleNamespace(state=[x, 0., z, np.pi, 0., 0., gap/2, -gap/2]))


def stage(kind="hold", *, target=None, held=None, reference=None, gripper="keep", verification="arrival"):
    return {"motion_context": {"intent": "grasp" if gripper == "close" else "transit",
        "target_id": target, "held_object_id": held, "allowed_contact_ids": [target] if target else [],
        "scene_inventory": {"reviewed": True, "support_ids": ["table"], "nearby_obstacle_ids": [], "unresolved": False}},
        "goal": {"kind": kind, "reference_id": reference, "quaternion_wxyz": None},
        "gripper": gripper, "profile": "transit", "verification": verification, "max_steps": 40}


def entity(ident, role, xyz, half):
    return {"id": ident, "role": role, "position_m": xyz, "quaternion_wxyz": [1., 0., 0., 0.],
        "boxes": [{"center_local_m": [0., 0., 0.], "half_extents_m": half, "quaternion_wxyz": [1., 0., 0., 0.]}],
        "uncertainty_m": .001, "stale": False, "attachment": "none"}


class Geometry:
    def __init__(self, unknown=False):
        self.entities = {"object": entity("object", "object", [.04, 0., .83], [.01, .01, .02]),
                         "table": entity("table", "support_surface", [0., 0., .77], [.3, .3, .01])}
        self.unknown = unknown
        self.probes = []
        self.confirmations = []

    def _entity_boxes(self, value, measured=None):
        rotation = _rotation(value["quaternion_wxyz"])
        return [_box(np.asarray(value["position_m"])+rotation @ np.asarray(b["center_local_m"]),
                     rotation @ _rotation(b["quaternion_wxyz"]), b["half_extents_m"], value["id"], value["id"], value["uncertainty_m"])
                for b in value["boxes"]]

    def _local_bodies(self, measured, context):
        return [_box([0., 0., 0.], np.eye(3), [.02, .03, .04], "robot_hand", "palm")], False

    def assess_motion(self, measured, xyz, q, context, step):
        return {"status": "unknown" if self.unknown else "supported", "allowed_fraction": 1.,
                "suggested_steps": 4 if self.unknown else None, "translation_step_m": .005, "rotation_step_rad": .04}

    def begin_attachment_probe(self, ident, measured, step):
        self.probes.append((ident, step))

    def confirm_attachment_evidence(self, evidence, measured, step):
        self.confirmations.append(deepcopy(evidence))
        return {"accepted": True, "target_id": evidence["target_id"]}


class Monitor:
    def __init__(self, status="unknown"):
        self.status = status
        self.begins = []
        self.observed = []
        self.queries = []

    def begin(self, ident, measured, step, kind="hold"):
        self.begins.append((ident, step, kind))

    def observe(self, measured, step):
        self.observed.append(step)

    def evidence(self, kind, ident, measured, step, since_step):
        self.queries.append((kind, ident, step, since_step))
        return {"status": self.status, "kind": kind, "target_id": ident,
                "source": "public_two_view_feature_motion", "reference_step": since_step,
                "views": {"external": {"status": self.status}, "wrist": {"status": self.status}}}


def advance(primitive, before, after):
    action = primitive.next_action(before)
    primitive.observe_transition(before, action, after)
    return action


def test_program_contract_is_semantic_and_rejects_waypoints_or_hidden_stage_fields():
    original = {"stages": [stage()]}
    assert validate_program(original) == original
    assert program_schema()["anyOf"][0]["properties"]["stages"]["maxItems"] == 24
    for field in ("xyz_m", "offset_local_m", "task_id"):
        bad = deepcopy(original)
        bad["stages"][0]["goal"][field] = [0., 0., 0.]
        with pytest.raises(ValueError, match="semantic"):
            validate_program(bad)
    bad = deepcopy(original)
    bad["stages"][0]["motion_context"]["allowed_contact_ids"] = ["object", "object"]
    with pytest.raises(ValueError, match="duplicates"):
        validate_program(bad)


def test_grasp_goal_moves_with_object_geometry_and_uses_public_pad_center():
    geometry = Geometry()
    request = stage("grasp", target="object", reference="object")
    first = derive_program_goal(view(), request, geometry)
    geometry.entities["object"]["position_m"][0] += .02
    second = derive_program_goal(view(.02), request, geometry)
    assert np.asarray(second["xyz_m"])-first["xyz_m"] == pytest.approx([.02, 0., 0.])
    assert first["xyz_m"][2] != pytest.approx(geometry.entities["object"]["position_m"][2])
    geometry.entities["object"]["boxes"][0]["half_extents_m"] = [.2, .2, .2]
    with pytest.raises(MissingGeometry, match="jaw_span"):
        derive_program_goal(view(), request, geometry)


def test_program_advances_authorized_stage_and_gripper_without_a_model_call():
    geometry, monitor = Geometry(), Monitor()
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"stages": [stage(),
        stage(target="object", gripper="close")]}, geometry, interaction_monitor=monitor)
    measured = view(gap=.04)
    first = advance(primitive, measured, measured)
    assert first[-1] == -1. and primitive.stage_index == 1
    assert primitive.current_context["target_id"] == "object"
    assert monitor.begins == [("object", 1, "hold")]
    for _ in range(12):
        assert advance(primitive, measured, measured)[-1] == 1.
    assert primitive.done and primitive.reason == "program_complete"
    assert primitive.actions_executed == 13
    assert len(primitive.completed_stages) == 2
    assert geometry.probes == [("object", 13)] and geometry.confirmations == []
    assert primitive.progress()["task_success"] == "unknown"


def test_mechanical_close_and_model_attachment_do_not_pass_visual_hold_gate():
    geometry, monitor = Geometry(), Monitor()
    geometry.entities["object"]["attachment"] = "held"
    primitive = AdaptiveProgramPrimitive(view(gap=0.), {"stages": [stage("transfer", held="object", reference="destination")]},
                                         geometry, interaction_monitor=monitor)
    assert primitive.done and primitive.reason == "program_visual_hold_required"
    assert primitive.actions_executed == 0
    assert primitive.export_continuation()["remaining_watchdog_steps"] == 40


def test_missing_visual_verification_pauses_before_next_stage_and_preserves_baseline():
    geometry, monitor = Geometry(), Monitor()
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"stages": [
        stage(target="object", gripper="close", verification="visual_hold"), stage()]}, geometry,
        step=30, interaction_monitor=monitor)
    measured = view(gap=.04)
    for _ in range(12):
        advance(primitive, measured, measured)
    assert primitive.done and primitive.reason == "program_visual_verification_required"
    assert primitive.stage_index == 0 and primitive.completed_stages == []
    saved = primitive.export_continuation()
    assert saved["phase"] == "verify" and saved["interaction_starts"]["hold:object"] == 30
    monitor.status = "supported"
    resumed = AdaptiveProgramPrimitive.from_continuation(measured, saved, geometry, step=42, interaction_monitor=monitor)
    assert resumed.stage_index == 1 and not resumed.done
    assert geometry.confirmations[-1]["reference_step"] == 30
    assert monitor.begins == [("object", 30, "hold")]


def test_geometry_pause_retains_absolute_derived_goal_and_remaining_child_watchdog():
    geometry, monitor = Geometry(unknown=True), Monitor()
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [
        stage("grasp", reference="object", target="object")]}, geometry, interaction_monitor=monitor)
    original = deepcopy(primitive.stage_goal)
    before = view()
    for index in range(4):
        after = view((index+1)*.001)
        advance(primitive, before, after)
        before = after
    assert primitive.done and primitive.reason == "program_geometry_observation_horizon"
    saved = primitive.export_continuation()
    assert saved["stage_goal"] == original
    child_budget = saved["child_continuation"]["remaining_watchdog_steps"]
    geometry.unknown = False
    geometry.entities["object"]["position_m"][0] += .1
    resumed = AdaptiveProgramPrimitive.from_continuation(before, saved, geometry, step=4, interaction_monitor=monitor)
    assert resumed.stage_goal == original
    assert resumed.child._watchdog.limit == child_budget
    changed = deepcopy(saved["motion_context"])
    changed["allowed_contact_ids"] = []
    with pytest.raises(ValueError, match="intent"):
        AdaptiveProgramPrimitive.from_continuation(before, saved, geometry, changed, interaction_monitor=monitor)


def test_visual_support_cannot_advance_when_geometry_rejects_confirmation():
    geometry, monitor = Geometry(), Monitor(status="supported")
    geometry.confirm_attachment_evidence = lambda *args: {"accepted": False, "reason": "unresolved_shape_or_neighbor_change"}
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"stages": [
        stage(target="object", gripper="close", verification="visual_hold"), stage()]}, geometry,
        interaction_monitor=monitor)
    measured = view(gap=.04)
    for _ in range(12):
        advance(primitive, measured, measured)
    assert primitive.done and primitive.reason == "program_visual_verification_required"
    assert primitive.stage_index == 0 and not primitive.completed_stages
    assert not primitive.confirmed_interactions
    assert primitive.verification_evidence["geometry_confirmation"]["accepted"] is False


def test_release_verification_keeps_manipulated_object_when_next_stage_targets_container():
    geometry, monitor = Geometry(), Monitor(status="supported")
    opening = stage(target="container", held="object", reference="container", gripper="open")
    verify = stage(target="container", reference="container", verification="visual_release")
    primitive = AdaptiveProgramPrimitive(view(gap=.04), {"stages": [opening, verify]}, geometry,
        keep_gripper="close", interaction_monitor=monitor)
    measured = view(gap=.04)
    for _ in range(9):
        advance(primitive, measured, measured)
    assert primitive.done and primitive.reason == "program_complete"
    assert monitor.begins == [("object", 0, "release")]
    assert [query[1] for query in monitor.queries if query[0] == "release"] == ["object", "object"]
    assert geometry.confirmations[-1]["target_id"] == "object"


class SupportedGeometry(Geometry):
    def preview_motion(self, measured, xyz, q, context, step=None):
        return {"status": "supported", "allowed_fraction": 1., "clearance_estimate_m": .1,
                "missing_evidence_requests": []}


def test_credible_local_pose_update_retargets_saved_goal_without_resetting_budget():
    geometry, monitor = SupportedGeometry(unknown=True), Monitor()
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [stage("grasp", reference="object", target="object")]},
        geometry, interaction_monitor=monitor)
    before = view()
    for index in range(4):
        after = view((index+1)*.001)
        advance(primitive, before, after)
        before = after
    assert primitive.done
    saved = primitive.export_continuation()
    original = deepcopy(saved["stage_goal"])
    geometry.unknown = False
    obj = geometry.entities["object"]
    obj["position_m"][0] += .02
    obj.update(local_pose_revision=1, geometry_revision=4, local_pose_evidence={
        "status": "supported", "accepted": True, "pose_updated": True,
        "source": "public_two_view_feature_displacement", "geometry_revision": 4})
    resumed = AdaptiveProgramPrimitive.from_continuation(before, saved, geometry, step=4, interaction_monitor=monitor)
    assert resumed.stage_goal["xyz_m"] != original["xyz_m"]
    assert resumed.stage_goal["xyz_m"][0]-original["xyz_m"][0] == pytest.approx(.02)
    assert resumed._watchdog.limit == saved["remaining_watchdog_steps"]
    assert resumed._prior_stage_actions == saved["stage_consumed_steps"]
    assert resumed.goal_updates[-1]["fresh_pose_evidence_ids"] == ["object"]


def test_real_stagnation_uses_supported_measured_retreat_then_alternative_without_closing():
    from gpt_field.action import TRANSLATION_SCALE_M
    geometry, monitor = SupportedGeometry(), Monitor()
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [stage("grasp", reference="object", target="object")]},
        geometry, interaction_monitor=monitor)
    measured = view()
    learned_limit = None
    actions = []
    for index in range(100):
        action = primitive.next_action(measured)
        actions.append(action)
        state = np.asarray(measured.state).copy()
        # First establish a measured route, then stall for a complete watchdog
        # window, and finally follow the controller's actual recovery commands.
        if index < 10 or primitive.recovery_events:
            state[:3] += np.asarray(action[:3])*TRANSLATION_SCALE_M*.5
        after = StepView(SimpleNamespace(state=state.tolist()))
        primitive.observe_transition(measured, action, after)
        measured = after
        if index == 9:
            learned_limit = primitive._watchdog.limit
        if primitive.done:
            break
    assert primitive.done and primitive.reason == "program_complete"
    assert primitive.recovery_events and primitive.recovery_events[0]["kind"] == "retreat"
    assert primitive.recovery_events[0]["status"] == "resumed"
    assert primitive._watchdog.limit == learned_limit
    assert primitive.recovery_events[0]["remaining_watchdog_steps"] < learned_limit
    assert all(action[-1] == -1. for action in actions)
    assert primitive.completed_stages[0]["actions"] == len(actions)


def test_recovery_does_not_treat_unknown_geometry_as_supported_retreat():
    from gpt_field.action import TRANSLATION_SCALE_M
    geometry, monitor = SupportedGeometry(), Monitor()
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [stage("grasp", reference="object", target="object")]},
        geometry, interaction_monitor=monitor)
    measured = view()
    for index in range(40):
        action = primitive.next_action(measured)
        state = np.asarray(measured.state).copy()
        if index < 10:
            state[:3] += np.asarray(action[:3])*TRANSLATION_SCALE_M*.5
        else:
            geometry.preview_motion = lambda *args, **kwargs: {"status": "unknown", "allowed_fraction": 1.}
        after = StepView(SimpleNamespace(state=state.tolist()))
        primitive.observe_transition(measured, action, after)
        measured = after
        if primitive.done:
            break
    assert primitive.done and primitive.reason == "program_segment_stall"
    assert primitive.recovery_events == []
    saved = primitive.export_continuation()
    assert saved is not None and saved["remaining_watchdog_steps"] < primitive._watchdog.limit


def test_contact_speed_uses_current_padded_clearance_and_reverts_on_tracking_mismatch():
    from gpt_field.agent_speedup_v3_fragments import _adaptive_contact_cap
    geometry = Geometry()
    context = {**stage("grasp", target="object", reference="object")["motion_context"], "adaptive_contact_speed": True}
    assert _adaptive_contact_cap(geometry, view(z=1.1), context, [.04, 0., .85], "contact", {"status": "normal"}) == pytest.approx(.02)
    assert _adaptive_contact_cap(geometry, view(z=.90), context, [.04, 0., .85], "contact", {"status": "normal"}) == pytest.approx(.005)
    assert _adaptive_contact_cap(geometry, view(z=1.1), context, [.04, 0., .85], "contact", {"status": "mismatch"}) == pytest.approx(.005)
    geometry.entities["object"]["stale"] = True
    assert _adaptive_contact_cap(geometry, view(z=1.1), context, [.04, 0., .85], "contact", {"status": "normal"}) == pytest.approx(.005)


class GoalLocalGeometry(SupportedGeometry):
    execution_policy = "goal_local"

    def preview_motion(self, measured, xyz, q, context, step=None):
        return {"status": "unknown", "allowed_fraction": 1., "execution_allowed": True,
            "blocking_reasons": [], "advisories": ["free_space_not_certified"],
            "clearance_estimate_m": .1, "missing_evidence_requests": []}


def contact_goal_request():
    request = stage("grasp", target="object", reference="object")
    request["profile"] = "contact"
    request["motion_context"]["intent"] = "grasp"
    return request


@pytest.mark.parametrize("orientation", ([0., 1., 0., 0.], [0., 2**-.5, 2**-.5, 0.]))
def test_goal_local_grasp_intersects_whole_hand_support_clearance_and_pad_contact(orientation):
    from gpt_field.robot_tool_geometry import _local_parts
    geometry = GoalLocalGeometry()
    geometry.entities["object"] = entity("object", "object", [.04, 0., .79], [.015, .01, .01])
    request = contact_goal_request()
    request["goal"]["quaternion_wxyz"] = orientation
    goal = derive_program_goal(view(), request, geometry)
    rotation = _rotation(goal["quaternion_wxyz"])
    whole_hand = np.concatenate([vertices for _, vertices, _ in _local_parts(view().state, True)])
    world = whole_hand @ rotation.T+goal["xyz_m"]
    assert world[:, 2].min() >= .78+.002-1e-9
    assert goal["goal_consistency"]["support_correction_world_m"][2] > 0
    assert min(min(row) for row in goal["goal_consistency"]["pad_object_overlap_xz_m"]) > 0
    assert goal["physical_clearance_verified"] is False
    before = deepcopy(geometry.entities)
    geometry.entities["object"]["uncertainty_m"] = .05
    geometry.entities["table"]["uncertainty_m"] = .05
    uncertain_goal = derive_program_goal(view(), request, geometry)
    assert uncertain_goal["xyz_m"] == pytest.approx(goal["xyz_m"])
    assert geometry.entities["object"]["uncertainty_m"] == .05
    assert geometry.entities["table"]["uncertainty_m"] == .05
    assert uncertain_goal["goal_consistency"]["uncertainty_is_advisory"] is True
    assert uncertain_goal["goal_consistency"]["object_uncertainty_m"] == .05
    assert uncertain_goal["geometry_advisories"]
    assert before["object"]["position_m"] == geometry.entities["object"]["position_m"]


def test_goal_local_grasp_does_not_raise_pads_above_an_infeasible_thin_object():
    geometry = GoalLocalGeometry()
    geometry.entities["object"] = entity("object", "object", [.04, 0., .783], [.015, .01, .003])
    with pytest.raises(MissingGeometry, match="support_clearance_and_grasp_contact_inconsistent"):
        derive_program_goal(view(), contact_goal_request(), geometry)


@pytest.mark.parametrize("uncertain", [
    {"stale": True, "stale_reason": "release_slip_or_possible_contact", "invalidation_kind": "possible_contact",
     "invalidation_local_contact": True, "nonsettling_invalidation": False},
    {"monitoring_unknown": True, "visual_monitoring": {"status": "unavailable", "self_occluded": True}},
])
def test_goal_local_expected_contact_or_occlusion_retains_explicitly_uncertain_shape(uncertain):
    geometry = GoalLocalGeometry()
    geometry.entities["object"].update(uncertain)
    before = deepcopy(geometry.entities)
    request = contact_goal_request()
    request["motion_context"]["scene_inventory"].update(reviewed=False, unresolved=True)
    goal = derive_program_goal(view(), request, geometry)
    assert goal["geometry_advisories"]
    assert any(item["reason"] == "conditional_shape_after_contact_or_occlusion" for item in goal["geometry_advisories"])
    assert geometry.entities == before
    geometry.execution_policy = "strict"
    request["motion_context"]["scene_inventory"].update(reviewed=True, unresolved=False)
    with pytest.raises(MissingGeometry, match="fresh_geometry"):
        derive_program_goal(view(), request, geometry)


@pytest.mark.parametrize("change", [
    {"invalidation_kind": "local_observed_displacement"},
    {"visual_monitoring": {"status": "changed", "self_occluded": True}},
    {"nonsettling_invalidation": True},
])
def test_goal_local_does_not_reuse_observed_unresolved_displacement_as_occlusion(change):
    geometry = GoalLocalGeometry()
    geometry.entities["object"].update({"stale": True, "invalidation_kind": "possible_contact", "invalidation_local_contact": True,
        "visual_monitoring": {"self_occluded": True}, **change})
    with pytest.raises(MissingGeometry, match="unresolved_observed_geometry_change"):
        derive_program_goal(view(), contact_goal_request(), geometry)


@pytest.mark.parametrize("permitted", [True, False])
def test_local_recovery_obeys_execution_permission_separately_from_evidence_status(permitted):
    from gpt_field.action import TRANSLATION_SCALE_M
    geometry, monitor = GoalLocalGeometry(), Monitor()
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [stage("grasp", reference="object", target="object")]},
        geometry, interaction_monitor=monitor)
    measured = view()
    learned_limit = None
    actions = []
    for index in range(100):
        action = primitive.next_action(measured)
        actions.append(action)
        state = np.asarray(measured.state).copy()
        if index < 10 or primitive.recovery_events:
            state[:3] += np.asarray(action[:3])*TRANSLATION_SCALE_M*.5
        if index == 10:
            geometry.preview_motion = lambda *args, **kwargs: {
                "status": "unknown" if permitted else "supported", "allowed_fraction": 1.,
                "execution_allowed": permitted, "missing_evidence_requests": []}
        after = StepView(SimpleNamespace(state=state.tolist()))
        primitive.observe_transition(measured, action, after)
        measured = after
        if index == 9:
            learned_limit = primitive._watchdog.limit
        if primitive.done:
            break
    assert primitive.done
    assert primitive._watchdog.limit == learned_limit
    assert all(action[-1] == -1. for action in actions)
    if permitted:
        assert primitive.reason == "program_complete"
        assert primitive.recovery_events[0]["kind"] == "retreat"
        assert primitive.recovery_events[0]["status"] == "resumed"
        assert primitive.completed_stages[0]["actions"] == len(actions)
    else:
        assert primitive.reason == "program_segment_stall"
        assert primitive.recovery_events == []


def test_goal_local_finite_lift_can_derive_conditional_payload_goal_but_not_certify_hold():
    geometry = GoalLocalGeometry()
    geometry.entities["object"].update(attachment="held", attachment_probe_shape_usable=True,
        attachment_probe_unconfirmed=True)
    original = geometry._local_bodies
    geometry._local_bodies = lambda measured, context: (original(measured, context)[0], True)
    request = stage("lift_clear", target="object", held="object", reference="object", verification="visual_hold")
    goal = derive_program_goal(view(), request, geometry)
    assert goal["xyz_m"][2] > view().position[2]
    primitive = AdaptiveProgramPrimitive(view(), {"stages": [stage("transfer", held="object", reference="container")]},
        geometry, interaction_monitor=Monitor())
    assert primitive.done and primitive.reason == "program_visual_hold_required"
    assert geometry.entities["object"]["attachment_probe_unconfirmed"] is True
