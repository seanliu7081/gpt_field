"""Synthetic measured-state contracts; no model or simulator is instantiated."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v3_fragments import (
    AdaptiveFragmentSequencePrimitive, AdaptiveLocalMotionPrimitive,
)
from gpt_field.contact_skills import StepView


def view(x=0., gap=.08):
    return StepView(SimpleNamespace(state=[x, 0., .8, np.pi, 0., 0., gap/2, -gap/2]))


def segment(skill="move_relative", xyz=None, **changes):
    result = {"skill": skill, "xyz_m": [.04, 0., 0.] if xyz is None else xyz,
              "quaternion_wxyz": None, "gripper": "keep", "profile": "transit",
              "max_steps": 40, "observation_after": False}
    if skill in ("set_gripper", "wait"):
        result["xyz_m"] = None
    result.update(changes)
    return result


def motion(**changes):
    value = {"frame": "world", "position_mode": "relative", "xyz_m": [.04, 0., 0.],
             "quaternion_wxyz": None, "gripper": "keep", "steps": 5}
    value.update(changes)
    return value


def context(intent="transit"):
    return {"intent": intent, "target_id": None, "held_object_id": None,
            "allowed_contact_ids": [], "scene_inventory": {
                "reviewed": False, "support_ids": [], "nearby_obstacle_ids": [], "unresolved": True}}


def assessment(**changes):
    value = {"status": "unknown", "allowed_fraction": 1., "suggested_steps": 4,
             "translation_step_m": .005, "rotation_step_rad": .025}
    value.update(changes)
    return value


class Geometry:
    def __init__(self, first=None, later=None, by_step=None):
        self.first = assessment() if first is None else first
        self.later = self.first if later is None else later
        self.by_step = by_step
        self.calls = []

    def assess_motion(self, measured, xyz, wxyz, motion_context, step):
        self.calls.append({"xyz": np.asarray(xyz).copy(), "context": deepcopy(motion_context),
                           "step": step})
        if self.by_step is not None:
            return deepcopy(self.by_step(step))
        return deepcopy(self.first if len(self.calls) == 1 else self.later)


def advance(primitive, before, after):
    command = primitive.next_action(before)
    primitive.observe_transition(before, command, after)
    return command


def test_fragment_observation_budget_is_fixed_at_entry_not_reset_or_extended():
    geometry = Geometry(first=assessment(suggested_steps=4),
                        later=assessment(status="supported", suggested_steps=None))
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]},
                                                  geometry, context(), step=20)
    before = view()
    for index in range(4):
        after = view((index + 1)*.001)
        advance(primitive, before, after)
        before = after
        assert primitive.done is (index == 3)
    progress = primitive.progress()
    assert primitive.actions_executed == 4
    assert progress["status"] == "observation_required"
    assert progress["goal_reached"] is False
    assert progress["geometry_commitment"]["horizon_steps"] == 4
    assert [item["step"] for item in geometry.calls] == [20, 21, 22, 23, 24]


def test_supported_contact_reaches_original_goal_then_honors_explicit_observation():
    geometry = Geometry(assessment(status="supported", suggested_steps=None,
                                   translation_step_m=.003))
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
        segment(xyz=[.035, 0., 0.], profile="contact", observation_after=True),
        segment("set_gripper", gripper="close")]}, geometry, context("approach"))
    before = view()
    for index in range(40):
        after = view(min((index + 1)*.001, .035))
        command = advance(primitive, before, after)
        assert np.linalg.norm(np.asarray(command[:3])*.05) <= .003 + 1e-12
        before = after
        if primitive.done:
            break
    progress = primitive.progress()
    assert primitive.done and primitive.reason == "observation_required"
    assert primitive.actions_executed > 24
    assert progress["completed_segment_indices"] == [0]
    assert progress["pending_segment_indices"] == [1]
    assert progress["goal_reached"] is True
    assert progress["geometry_commitment"]["horizon_steps"] == 40
    assert progress["geometry_commitment"]["original_goal_truncated"] is False
    assert primitive.gripper_command == "open"
    assert len(geometry.calls) == primitive.actions_executed + 1


def test_supported_fragment_degradation_locks_remaining_horizon_without_renewal():
    supported = assessment(status="supported", suggested_steps=None)
    geometry = Geometry(by_step=lambda step: assessment(suggested_steps=4)
                        if 8 <= step < 10 else supported)
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]},
                                                  geometry, context())
    before = view()
    for index in range(12):
        after = view((index + 1)*.001)
        advance(primitive, before, after)
        before = after
        assert primitive.done is (index == 11)
        assert primitive.progress()["geometry_commitment"]["horizon_steps"] == (
            40 if index < 7 else 12)
    assert primitive.reason == "geometry_observation_horizon"
    assert primitive.progress()["remaining_current_goal"]["xyz_m"] == pytest.approx([.04, 0., .8])
    assert len(geometry.calls) == 13


@pytest.mark.parametrize("budget, expected_reason", [(5, "segment_budget_exhausted"),
                                                    (40, "segment_stall")])
def test_supported_geometry_preserves_original_budget_and_stall(budget, expected_reason):
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment(max_steps=budget),
        segment("set_gripper", gripper="close")]},
        Geometry(assessment(status="supported", suggested_steps=None)), context())
    for _ in range(budget):
        advance(primitive, view(), view())
        if primitive.done:
            break
    assert primitive.done and primitive.reason == expected_reason
    assert primitive.gripper_command == "open"
    assert primitive.progress()["completed_segment_indices"] == []
    if expected_reason == "segment_budget_exhausted":
        assert primitive.actions_executed == budget


def test_local_prefix_preserves_original_target_and_unexecuted_gripper_segment():
    geometry = Geometry(first=assessment(allowed_fraction=.5, suggested_steps=30),
                        later=assessment(suggested_steps=30))
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment(),
        segment("set_gripper", gripper="close")]}, geometry, context())
    command = advance(primitive, view(), view(.02))
    assert command[-1] == -1.
    progress = primitive.progress()
    assert primitive.done and primitive.gripper_command == "open"
    assert progress["status"] == "observation_required"
    assert progress["current_segment_status"] == "paused"
    assert progress["goal_reached"] is False and progress["sequence_complete"] is False
    assert progress["completed_segment_indices"] == []
    assert progress["pending_segment_indices"] == [0, 1]
    assert progress["target_xyz_m"] == pytest.approx([.04, 0., .8])
    assert progress["remaining_current_goal"]["xyz_m"] == pytest.approx([.04, 0., .8])
    assert progress["geometry_commitment"]["target_xyz_m"] == pytest.approx([.02, 0., .8])


@pytest.mark.parametrize("requested", [
    segment("move_to_pose", [.3001, 0., .8]),
    segment("move_to_pose", [.0801, 0., .8], profile="contact"),
    segment("move_to_pose", [0., 0., .8], quaternion_wxyz=[0., np.cos(.65), np.sin(.65), 0.]),
    segment("move_to_pose", [0., 0., .8], profile="contact",
            quaternion_wxyz=[0., np.cos(.3), np.sin(.3), 0.]),
])
def test_original_target_envelopes_reject_before_geometry_or_any_action(requested):
    geometry = Geometry()
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [requested]}, geometry, context())
    assert primitive.done and primitive.reason == "target_envelope_rejected"
    assert primitive.actions_executed == 0 and geometry.calls == []


def test_direct_unknown_motion_keeps_original_target_and_observation_budget():
    geometry = Geometry(first=assessment(allowed_fraction=.4, suggested_steps=4,
                                         translation_step_m=.003),
                        later=assessment(status="supported", suggested_steps=None,
                                         translation_step_m=.02))
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    before = view()
    for index in range(4):
        after = view((index + 1)*.002)
        command = advance(primitive, before, after)
        assert np.linalg.norm(np.asarray(command[:3])*.05) <= .003 + 1e-12
        before = after
        assert primitive.done is (index == 3)
    evidence = primitive.outcome()["evidence"]
    assert evidence["executed_steps"] == 4
    assert evidence["original_requested_target"][:3] == pytest.approx([.04, 0., .8])
    assert evidence["requested_target"][:3] == pytest.approx([.016, 0., .8])
    assert evidence["goal_reached"] is False and evidence["automatic_resume"] is False


def test_supported_direct_motion_renews_chunk_without_rewriting_original_goal():
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(),
        Geometry(assessment(status="supported", suggested_steps=None)), context())
    before = view()
    for index in range(5):
        after = view((index + 1)*.001)
        advance(primitive, before, after)
        before = after
        assert not primitive.done
    assert primitive.executor.requested_steps == 10
    assert primitive._local_continuations == 1
    assert primitive.original_target[:3] == pytest.approx([.04, 0., .8])


def test_direct_degradation_counts_remaining_steps_without_renewal_or_recovery_extension():
    supported = assessment(status="supported", suggested_steps=None)
    # A two-step suggestion makes the deadline observable within the direct
    # motion's unchanged five-step outer budget.
    geometry = Geometry(by_step=lambda step: assessment(suggested_steps=2)
                        if 1 <= step < 3 else supported)
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    before = view()
    for index in range(3):
        after = view((index + 1)*.001)
        advance(primitive, before, after)
        before = after
        assert primitive.done is (index == 2)
        assert primitive.executor.requested_steps == 3
    evidence = primitive.outcome()["evidence"]
    assert evidence["executed_steps"] == 3
    assert evidence["original_requested_target"][:3] == pytest.approx([.04, 0., .8])
    assert evidence["automatic_resume"] is False


def test_unknown_motion_horizon_does_not_prevent_gripper_settling():
    geometry = Geometry()
    stationary = view(gap=.04)
    primitive = AdaptiveFragmentSequencePrimitive(stationary, {"segments": [
        segment("set_gripper", gripper="close")]}, geometry, context("grasp"))
    for index in range(12):
        advance(primitive, stationary, stationary)
        assert primitive.done is (index == 11)
    assert primitive.outcome()["evidence"]["grasp_success"] == "unknown"


def test_direct_new_geometry_may_reduce_command_amplitude_without_resetting_progress():
    geometry = Geometry(first=assessment(translation_step_m=.02),
                        later=assessment(translation_step_m=.001))
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    advance(primitive, view(), view(.001))
    command = primitive.next_action(view(.001))
    assert np.linalg.norm(np.asarray(command[:3])*.05) <= .001 + 1e-12
    assert primitive.actions_executed == 1


@pytest.mark.parametrize("kind, profile, outer_cap", [
    ("fragment", "transit", .02), ("fragment", "contact", .005),
    ("direct", None, .02),
])
def test_supported_speed_recovers_after_temporary_reduction_with_normal_tracking(kind, profile, outer_cap):
    geometry = Geometry(by_step=lambda step: assessment(
        status="supported", suggested_steps=None,
        translation_step_m=.003 if step == 1 else .02,
        rotation_step_rad=.025 if step == 1 else .1))
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment(profile=profile)]},
                                                      geometry, context("approach"))
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context("approach"))
    before = view()
    for index in range(3):
        after = view((index + 1)*.001)
        command = advance(primitive, before, after)
        if index == 1:
            assert np.linalg.norm(np.asarray(command[:3])*.05) <= .003 + 1e-12
        before = after
    command = primitive.next_action(before)
    assert np.linalg.norm(np.asarray(command[:3])*.05) == pytest.approx(outer_cap)
    tracking = geometry.calls[-1]["context"]["tracking"]
    assert tracking["status"] == "normal" and tracking["sample_count"] == 3
    assert 0. < tracking["translation_response_ratio"] < 1.
    assert tracking["position_progress_m"] > 0.
    assert tracking["source"] == "measured_robot_proprioception"
    assert primitive.actions_executed == 3
    if kind == "fragment":
        commitment = primitive.progress()["geometry_commitment"]
        assert commitment["rotation_step_rad"] == .1
        assert commitment["horizon_steps"] == 40
        assert commitment["target_xyz_m"] == pytest.approx([.04, 0., .8])
    else:
        assert primitive.executor.config.max_step_rotation_rad == .1
        assert primitive.executor.requested_steps == 5


@pytest.mark.parametrize("kind, remaining, stop_step", [("fragment", 4, 8), ("direct", 1, 5)])
def test_measured_tracking_mismatch_reassesses_geometry_and_locks_bounded_handoff(kind, remaining, stop_step):
    class TrackingGeometry(Geometry):
        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            if motion_context["tracking"]["status"] == "mismatch":
                return assessment(translation_step_m=.001, suggested_steps=remaining)
            return assessment(status="supported", translation_step_m=.02, suggested_steps=None)

    geometry = TrackingGeometry()
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment(),
            segment("set_gripper", gripper="close")]}, geometry, context())
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    for index in range(stop_step):
        command = advance(primitive, view(), view())
        if index >= 4:
            assert np.linalg.norm(np.asarray(command[:3])*.05) <= .001 + 1e-12
        assert primitive.done is (index == stop_step - 1)
    tracking = geometry.calls[-1]["context"]["tracking"]
    assert geometry.calls[2]["context"]["tracking"]["status"] == "unobserved"
    assert geometry.calls[3]["context"]["tracking"]["status"] == "unobserved"
    assert geometry.calls[4]["context"]["tracking"]["status"] == "mismatch"
    assert tracking["translation_response_ratio"] == 0.
    assert tracking["position_progress_m"] == 0.
    assert tracking["consecutive_mismatch_steps"] == stop_step - 3
    assert primitive.gripper_command == "open"
    if kind == "fragment":
        progress = primitive.progress()
        assert progress["geometry_commitment"]["horizon_steps"] == 8
        assert progress["pending_segment_indices"] == [0, 1]
        assert progress["remaining_current_goal"]["xyz_m"] == pytest.approx([.04, 0., .8])
    else:
        assert primitive.outcome()["evidence"]["requested_steps"] == 5
        assert primitive.outcome()["evidence"]["automatic_resume"] is False


def test_rotation_tracking_uses_measured_orientation_when_translation_is_stationary():
    geometry = Geometry(assessment(status="supported", suggested_steps=None,
                                   rotation_step_rad=.1))
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(xyz_m=[0., 0., 0.],
        quaternion_wxyz=[0., np.cos(.1), np.sin(.1), 0.]), geometry, context())
    for _ in range(4):
        advance(primitive, view(), view())
    tracking = geometry.calls[-1]["context"]["tracking"]
    assert tracking["status"] == "mismatch"
    assert tracking["commanded_rotation_rad"] > .09
    assert tracking["rotation_response_ratio"] == 0.
    assert tracking["translation_response_ratio"] is None


@pytest.mark.parametrize("kind", ["fragment", "direct"])
def test_acknowledged_observation_is_watched_before_geometry_reassessment(kind):
    events = []
    class WatchedGeometry(Geometry):
        def watch_observation(self, measured, step):
            events.append(("watch", step, float(measured.position[0])))

        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            events.append(("assess", step, float(measured.position[0])))
            return super().assess_motion(measured, xyz, wxyz, motion_context, step)

    geometry = WatchedGeometry(assessment(status="supported", suggested_steps=None))
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]},
                                                      geometry, context(), step=10)
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context(), step=10)
    advance(primitive, view(), view(.001))
    assert events == [("assess", 10, 0.), ("watch", 11, .001), ("assess", 11, .001)]


def test_watched_change_is_seen_before_entering_explicit_gripper_segment():
    class ChangedGeometry(Geometry):
        changed = False

        def watch_observation(self, measured, step):
            self.changed = True

        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            if self.changed and motion_context["gripper_action"]:
                return assessment(status="blocked", allowed_fraction=0.)
            return assessment(status="supported", suggested_steps=None)

    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
        segment(xyz=[0., 0., 0.]), segment("set_gripper", gripper="close")]},
        ChangedGeometry(), context("grasp"))
    advance(primitive, view(), view())
    assert primitive.done and primitive.reason == "geometry_predicted_block"
    assert primitive.gripper_command == "open"
    assert primitive.progress()["pending_segment_indices"] == [1]


def test_contact_at_arrival_invalidates_before_following_explicit_close_guard():
    class ContactGeometry(Geometry):
        touched = False

        def __init__(self):
            super().__init__()
            self.transitions = []

        def observe_transition(self, before, after, step, gripper):
            self.transitions.append((step, gripper))
            self.touched = after.position[0] >= .003

        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            if self.touched and motion_context["gripper_action"]:
                return assessment(status="blocked", allowed_fraction=0.)
            return assessment(status="supported", suggested_steps=None)

    geometry = ContactGeometry()
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
        segment(xyz=[.003, 0., 0.]), segment("set_gripper", gripper="close")]},
        geometry, context("grasp"), step=10)
    action = advance(primitive, view(), view(.003))
    assert action[-1] == -1. and geometry.transitions == [(11, "open")]
    assert primitive.done and primitive.reason == "geometry_predicted_block"
    assert primitive.gripper_command == "open"
    assert primitive.progress()["completed_segment_indices"] == [0]
    assert primitive.progress()["pending_segment_indices"] == [1]
    assert primitive.actions_executed == 1


def test_direct_transition_updates_contact_state_before_geometry_reassessment():
    events = []
    class TransitionGeometry(Geometry):
        def observe_transition(self, before, after, step, gripper):
            events.append(("transition", step, gripper))

        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            events.append(("assess", step, motion_context["commanded_gripper"]))
            return super().assess_motion(measured, xyz, wxyz, motion_context, step)

    geometry = TransitionGeometry(assessment(status="supported", suggested_steps=None))
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(gripper="close"), geometry,
                                             context("grasp"), step=10)
    advance(primitive, view(), view(.001))
    assert events == [("assess", 10, "close"), ("transition", 11, "close"),
                      ("assess", 11, "close")]


def test_checked_gripper_settling_authorization_is_runtime_only_and_cleared_before_motion():
    reference = {"target_id": "target", "held_object_id": None, "gripper_command": "close",
                 "allowed_contact_ids": ["target"], "support_ids": ["table"],
                 "nearby_obstacle_ids": [], "entity_revisions": {"target": 1, "table": 2},
                 "hold_position_m": [0., 0., .8], "hold_quaternion_wxyz": [0., 1., 0., 0.]}
    expected = {**reference, "entry_step": 50, "max_steps": 20}
    class SettlingGeometry(Geometry):
        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            if len(self.calls) == 1:
                return assessment(status="supported", suggested_steps=None,
                    final_interaction_guard=True, missing_evidence_requests=[],
                    gripper_settling_reference=reference)
            if motion_context["gripper_action"]:
                assert motion_context["gripper_settling_authorization"] == expected
            else:
                assert motion_context["gripper_settling_authorization"] is None
            return assessment(status="supported", suggested_steps=None)

    geometry = SettlingGeometry()
    stationary = view(gap=.04)
    forged_context = {**context("grasp"), "gripper_settling_authorization": {"untrusted": True}}
    primitive = AdaptiveFragmentSequencePrimitive(stationary, {"segments": [
        segment("set_gripper", gripper="close", max_steps=20), segment()]},
        geometry, forged_context, step=50)
    assert geometry.calls[0]["context"]["gripper_settling_authorization"] is None
    for _ in range(12):
        advance(primitive, stationary, stationary)
    assert not primitive.done
    assert primitive.progress()["current_skill"] == "move_relative"
    assert geometry.calls[-1]["context"]["gripper_settling_authorization"] is None
    assert primitive.actions_executed == 12


@pytest.mark.parametrize("initial_status, stationary, authorized", [
    ("supported", True, True), ("unknown", True, False), ("supported", False, False),
])
def test_direct_settling_authorization_requires_checked_stationary_entry(initial_status, stationary, authorized):
    reference = {"target_id": "target", "gripper_command": "close"}
    geometry = Geometry(assessment(status=initial_status, suggested_steps=None,
        final_interaction_guard=True, missing_evidence_requests=[],
        gripper_settling_reference=reference))
    primitive = AdaptiveLocalMotionPrimitive(view(), motion(gripper="close",
        xyz_m=[0., 0., 0.] if stationary else [.04, 0., 0.]), geometry, context("grasp"), step=10)
    advance(primitive, view(), view())
    token = geometry.calls[-1]["context"]["gripper_settling_authorization"]
    assert token == ({**reference, "entry_step": 10, "max_steps": 5} if authorized else None)


def test_supported_progress_renews_chunks_past_240_without_crossing_observation_boundary():
    geometry = Geometry(assessment(status="supported", suggested_steps=None))
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
        segment(xyz=[.2, 0., 0.], max_steps=50, observation_after=True),
        segment("set_gripper", gripper="close")]}, geometry, context())
    before = view()
    for index in range(450):
        after = view(min((index+1)*.0005, .2))
        advance(primitive, before, after)
        before = after
        if index == 239:
            assert not primitive.done
        if primitive.done:
            break
    progress = primitive.progress()
    assert primitive.done and primitive.reason == "observation_required"
    assert primitive.actions_executed > 240 and progress["local_continuations"] > 0
    assert progress["goal_reached"] and progress["pending_segment_indices"] == [1]
    assert primitive.gripper_command == "open"
    assert primitive.export_continuation() is None


@pytest.mark.parametrize("kind", ["fragment", "motion"])
def test_geometry_patch_restores_absolute_goal_response_and_remaining_watchdog(kind):
    geometry = Geometry(assessment(suggested_steps=4, translation_step_m=.005))
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]}, geometry, context())
        cls = AdaptiveFragmentSequencePrimitive
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
        cls = AdaptiveLocalMotionPrimitive
    before = view()
    for index in range(4):
        after = view((index+1)*.001)
        advance(primitive, before, after)
        before = after
    saved = primitive.export_continuation()
    assert saved is not None and saved["current_target"]["xyz_m"] == pytest.approx([.04, 0., .8])
    assert saved["remaining_watchdog_steps"] > saved["remaining_chunk_steps"]
    revised = context()
    revised["scene_inventory"]["reviewed"] = True
    fresh = Geometry(assessment(status="supported", suggested_steps=None))
    resumed = cls.from_continuation(before, saved, fresh, revised, step=4)
    assert fresh.calls[0]["xyz"] == pytest.approx([.04, 0., .8])
    assert fresh.calls[0]["context"]["tracking"]["status"] == "normal"
    assert fresh.calls[0]["context"]["tracking"]["translation_response_estimate"] == pytest.approx(.2)
    limit = resumed._watchdog.limit
    advance(resumed, before, view(.005))
    assert resumed._watchdog.limit == limit == saved["remaining_watchdog_steps"]
    with pytest.raises(ValueError, match="intent"):
        cls.from_continuation(before, saved, fresh, {**revised, "intent": "grasp"}, step=4)
    changed = deepcopy(saved)
    changed["remaining_watchdog_steps"] = 0
    with pytest.raises(ValueError, match="budget"):
        cls.from_continuation(before, changed, fresh, revised, step=4)


def test_normal_response_estimate_is_bounded_and_mismatch_never_learns_low_response_gain():
    geometry = Geometry(assessment(status="supported", suggested_steps=None))
    primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]}, geometry, context())
    before = view()
    for index in range(3):
        after = view((index+1)*.001)
        advance(primitive, before, after)
        before = after
    tracking = primitive._tracking.summary()
    assert tracking["translation_response_estimate"] == pytest.approx(.2)
    assert tracking["translation_response_bounds"] == pytest.approx([.2, .2])
    for _ in range(4):
        advance(primitive, before, before)
    after_mismatch = primitive._tracking.summary()
    assert after_mismatch["status"] == "mismatch"
    assert after_mismatch["translation_response_estimate"] >= .03
    assert after_mismatch["normal_response_samples"]["translation"] <= 12
    assert geometry.calls[-1]["context"]["controller_translation_limit_m"] == .02


@pytest.mark.parametrize("kind", ["fragment", "motion"])
@pytest.mark.parametrize("whole_status", ["unknown", "blocked"])
def test_certified_fixed_prefix_continues_past_short_horizon_without_claiming_whole_goal(kind, whole_status):
    class PrefixGeometry(Geometry):
        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            if np.isclose(xyz[0], .04):
                fraction = max(0., (.03-measured.position[0])/(.04-measured.position[0]))
                return assessment(status=whole_status, execution_status="supported",
                    allowed_fraction=fraction, suggested_steps=None,
                    execution_prefix={"xyz_m": [.03, 0., .8], "quaternion_wxyz": list(wxyz),
                                      "fraction": fraction, "status": "supported", "reason": "coverage_frontier"})
            assert xyz[0] == pytest.approx(.03)
            return assessment(status="supported", suggested_steps=None)

    geometry = PrefixGeometry()
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
            segment(max_steps=10), segment("set_gripper", gripper="close")]}, geometry, context())
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    before = view()
    for index in range(35):
        after = view((index+1)*.001)
        command = advance(primitive, before, after)
        assert before.position[0] + command[0]*.05 <= .03 + 1e-12
        before = after
        if primitive.done:
            break
    assert primitive.done and primitive.reason == "geometry_prefix_observed"
    assert primitive.actions_executed > 6 and primitive._local_continuations > 0
    saved = primitive.export_continuation()
    assert saved["current_target"]["xyz_m"] == pytest.approx([.04, 0., .8])
    assert primitive.gripper_command == "open"
    latest = primitive.progress()["geometry_commitment"]["last_assessment"] if kind == "fragment" else primitive.assessment
    assert latest["whole_goal_assessment"]["status"] == whole_status
    assert latest["committed_prefix_assessment"]["status"] == "supported"
    assert latest["execution_status"] == "supported"
    assert not (primitive.progress()["goal_reached"] if kind == "fragment" else primitive.outcome()["evidence"]["goal_reached"])


@pytest.mark.parametrize("kind", ["fragment", "motion"])
def test_residual_clipped_command_is_not_a_persistent_cap_and_exact_couple_is_verified(kind):
    class VerifiedGeometry(Geometry):
        def __init__(self):
            super().__init__()
            self.verified = []

        def assess_motion(self, measured, xyz, wxyz, motion_context, step):
            super().assess_motion(measured, xyz, wxyz, motion_context, step)
            return assessment(status="supported", suggested_steps=None,
                translation_step_m=float(np.linalg.norm(np.asarray(xyz)-measured.position)),
                translation_limit_m=.02, rotation_step_rad=.1, rotation_limit_rad=.1)

        def verify_command(self, measured, action, motion_context, step, *, allow_unknown=False):
            self.verified.append((step, tuple(action), deepcopy(motion_context)))
            assert not allow_unknown
            return {"command_allowed": True, "command_supported": True}

    geometry = VerifiedGeometry()
    requested_q = [0., np.cos(.1), np.sin(.1), 0.]
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [
            segment(xyz=[.00004, 0., 0.], quaternion_wxyz=requested_q)]}, geometry, context())
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(xyz_m=[.00004, 0., 0.],
            quaternion_wxyz=requested_q), geometry, context())
    first = advance(primitive, view(), view(-.001))
    second = primitive.next_action(view(-.001))
    assert first == geometry.verified[0][1] and second == geometry.verified[1][1]
    assert first[0]*.05 == pytest.approx(.00004)
    assert second[0]*.05 == pytest.approx(.00104)
    assert np.linalg.norm(second[3:6])*.5 == pytest.approx(.1)
    assert geometry.verified[1][2]["controller_translation_limit_m"] == .02
    assert geometry.verified[1][2]["tracking"]["channels"]["translation"]["mode"] == "hold"


@pytest.mark.parametrize("kind", ["fragment", "motion"])
def test_failed_final_coupled_command_check_hands_off_before_another_action(kind):
    class RejectChangedGeometry(Geometry):
        def verify_command(self, measured, action, motion_context, step, *, allow_unknown=False):
            return {"command_allowed": step == 0, "command_supported": step == 0}

    geometry = RejectChangedGeometry(assessment(status="supported", suggested_steps=None))
    if kind == "fragment":
        primitive = AdaptiveFragmentSequencePrimitive(view(), {"segments": [segment()]}, geometry, context())
    else:
        primitive = AdaptiveLocalMotionPrimitive(view(), motion(), geometry, context())
    advance(primitive, view(), view(.001))
    assert primitive.done and primitive.reason == "geometry_predicted_block"
    assert primitive.actions_executed == 1
    assert primitive.export_continuation() is not None
    with pytest.raises(RuntimeError, match="finished"):
        primitive.next_action(view(.001))


def test_tracking_requires_own_valid_samples_and_two_persistent_anomalous_windows():
    from gpt_field.agent_speedup_v3_fragments import _TrackingEvidence

    tracking = _TrackingEvidence()
    q = [0., np.cos(.1), np.sin(.1), 0.]
    for rotation in (.001, .006, .001, .006, .001):
        tracking.observe(view(), [0., 0., 0., 0., 0., rotation/.5, -1.], view(), [0., 0., .8], q)
        assert tracking.summary()["status"] == "unobserved"
        assert tracking.summary()["channels"]["rotation"]["valid_drive_samples"] < 3
    for index in range(4):
        tracking.observe(view(), [0., 0., 0., 0., 0., .1/.5, -1.], view(), [0., 0., .8], q)
        if index < 2:
            assert tracking.summary()["status"] == "unobserved"
    assert tracking.summary()["status"] == "mismatch"
    assert tracking.summary()["normal_response_samples"]["rotation"] == 0


def test_policy_dispatch_keeps_shared_geometry_and_passes_only_action_to_controller():
    from gpt_field.agent_speedup_v3 import AgentSpeedUpV3Policy, POLICY_VERSION
    from gpt_field.agent_speedup_v3_client import validate_v3_agent_decision
    from test_agent_speedup_v3_client import client, decision, PNG

    executive = client()
    entity = {"id": "background", "role": "obstacle", "parent_id": None,
              "frame": "world", "position_m": [1., 1., 1.], "quaternion_wxyz": [1., 0., 0., 0.],
              "boxes": [{"center_local_m": [0., 0., 0.], "half_extents_m": [.05, .05, .05],
                         "quaternion_wxyz": [1., 0., 0., 0.]}],
              "uncertainty_m": .01, "coverage": "visible_surfaces", "attachment": "none",
              "related_ids": [], "anchors": None}
    proposed = validate_v3_agent_decision(decision(type="execute", execution={"segments": [segment()]},
        scene_update={"entities": [entity], "invalidate_ids": [], "remove_ids": []},
        motion_context=context()))
    executive.request_agent = lambda *_: proposed
    policy = AgentSpeedUpV3Policy(executive, lock=executive.plan_lock)
    policy._client_checked = True
    before = view().observation
    before.image = PNG
    action = policy.act(before)
    assert isinstance(policy._primitive, AdaptiveFragmentSequencePrimitive)
    assert action.diagnostics["macro"] == "framework_fragment"
    after = view(.005).observation
    after.image = PNG
    policy.observe_transition(before, action, after)
    assert policy.agent_stats()["counts"]["fragment_actions"] == 1
    assert policy.agent_stats()["policy_version"] == POLICY_VERSION
    summary = policy._context(view(.005))["native_evidence"]["shared_geometry"]
    background = next(item for item in summary["entities"] if item["id"] == "background")
    assert background["source"] == "model_visual_estimate"
    assert background["observation_step"] == 0
    assert policy.plan_lock._world_state["version"] == POLICY_VERSION
    assert policy._context(view(.005))["execution"]["shared_state"]["holding_status"] == "unverified"


@pytest.mark.parametrize("event_step", [27, 37, 41, 181])
def test_recorded_tracking_false_stops_keep_in_tolerance_channel_in_hold_mode(event_step):
    from gpt_field.agent_speedup_v3_fragments import _TrackingEvidence
    from gpt_field.agent_speedup_v2_user_check_fragments import PROFILES

    _, profile_name, held, xyz, q, rows = next(row for row in _RECORDED_HOLD_WINDOWS if row[0] == event_step)
    profile = PROFILES[profile_name]
    tracking = _TrackingEvidence()
    for before, command, after in rows:
        before_view = StepView(SimpleNamespace(state=[*before, .04, -.04]))
        after_view = StepView(SimpleNamespace(state=[*after, .04, -.04]))
        tracking.observe(before_view, [*command, -1.], after_view, xyz, q,
            position_tolerance_m=profile["position_tolerance_m"],
            rotation_tolerance_rad=profile["rotation_tolerance_rad"])
        assert tracking.summary()["status"] != "mismatch"
    evidence = tracking.summary()
    assert evidence["channels"][held]["mode"] == "hold"
    assert evidence[held+"_response_ratio"] is None
    assert evidence["normal_response_samples"][held] == 0
    assert evidence["status"] == "normal"


# Frozen public proprioception/action windows from the four t1_005 false stops.
# These are replay fixtures, not runtime waypoints or task-specific policy rules.
_RECORDED_HOLD_WINDOWS = [
    [27, 'transit', 'rotation', [0.05, 0.06, 0.67], [0.0, 1.0, 0.0, 0.0], [[[-0.031857963351691755, 0.01107284538146283, 0.6684100046358197, -3.1406124207634227, -0.0032256903158820547, -0.0015743924634438976], [0.3432961617086503, 0.20519084150317227, 0.0066681271229842454, -0.0019563637623214274, 0.0020052033698363743, -0.004108356233684259], [-0.028436882846423674, 0.013694580148764514, 0.6675912759642797, -3.1406314303927267, -0.0028285359874709076, -0.008093528861307705]], [[-0.028436882846423674, 0.013694580148764514, 0.6675912759642797, -3.1406314303927267, -0.0028285359874709076, -0.008093528861307705], [0.34433415296477327, 0.20327857180877706, 0.01057418296682696, -0.0018990440044401888, 0.010308128575554345, -0.0036024969008044885], [-0.02492584939371548, 0.016337029340828715, 0.6668658914327128, -3.1406109172525474, -0.0022867641266382023, -0.012245965993199196]], [[-0.02492584939371548, 0.016337029340828715, 0.6668658914327128, -3.1406109172525474, -0.0022867641266382023, -0.012245965993199196], [0.08634348169334007, 0.050316585508114575, 0.0036117020747072908, -0.0004411472984555131, 0.0035947110934055975, -0.0006712623878421211], [-0.022435973494348255, 0.018177232912514427, 0.6664820261971937, -3.14104675913897, -0.0011572591682640482, -0.01120108990411897]]]],
    [37, 'transit', 'rotation', [0.025, 0.045, 0.666], [0.000262854786158813, -0.9999935393341118, -0.0003684286737318243, -0.003566014261019775], [[[-0.011264678744855745, 0.02404288341019199, 0.6669882259317016, -3.140613559777523, -0.002635832205824576, -0.012164268366442778], [0.346232334181401, 0.20008536255488762, -0.00943495938400271, -0.0008511743231580638, 0.0012288682349874392, -0.0018860754636767165], [-0.007847453034013513, 0.026199932216754974, 0.6662331087419362, -3.139922189917157, -0.0036425213950438064, -0.018323377507501216]], [[-0.007847453034013513, 0.026199932216754974, 0.6662331087419362, -3.139922189917157, -0.0036425213950438064, -0.018323377507501216], [0.34715357631769284, 0.1986915320104226, -0.0024636471311872844, -0.0021704082740953367, 0.009078128058768547, -0.003171852942917236], [-0.004265305203074741, 0.0285521296154687, 0.6653909754660818, -3.139510882347308, -0.003886164998189729, -0.022482444176548634]], [[-0.004265305203074741, 0.0285521296154687, 0.6653909754660818, -3.139510882347308, -0.003886164998189729, -0.022482444176548634], [0.3486434102189548, 0.19594675613018575, 0.00725543058371872, -0.0029392214510384606, 0.014379751861114006, -0.0034841947642616088], [-0.00056831543195664, 0.030970879470321806, 0.6646724056701009, -3.1393402353185995, -0.003668846429686522, -0.024881607581617712]]]],
    [41, 'contact', 'rotation', [0.05, 0.06, 0.65], [0.001075836506389505, -0.9999673312482432, -0.0011686298069189533, -0.007925485250257134], [[[0.0021377281057580197, 0.032690148330221616, 0.6642410528688661, -3.1398007381484354, -0.0028348697976095525, -0.022773100869262566], [0.08409280356127531, 0.04798272001857324, -0.02502117041275192, 0.0008819610405846101, -0.0026907623773801724, 0.0010672174950545747], [0.003996711773512043, 0.03370977949409092, 0.6642230440746524, -3.1402855124787954, -0.002240594641862509, -0.01935988467631361]], [[0.003996711773512043, 0.03370977949409092, 0.6642230440746524, -3.1402855124787954, -0.002240594641862509, -0.01935988467631361], [0.08385365351063369, 0.047921162290988646, -0.025925412197288524, 0.0007281894342958178, -0.00284298545720692, 0.0007376885984439104], [0.00565593659103092, 0.034517567423311836, 0.6642626123767534, -3.1404635525297175, -0.0021301969756865125, -0.01697316244156513]], [[0.00565593659103092, 0.034517567423311836, 0.6642626123767534, -3.1404635525297175, -0.0021301969756865125, -0.01697316244156513], [0.08351695773169462, 0.04799323924783961, -0.0268619946716911, 0.002134570725165167, -0.010083447458078647, 0.0019674535115933116], [0.00724061926309175, 0.035236156947615636, 0.6642782857677234, -3.1403964242456013, -0.002355628907353087, -0.015887267513755906]]]],
    [181, 'transit', 'translation', [0.0503, 0.0578, 0.5666], [0.0, 0.9238797538373537, 0.3826828980362601, 0.0], [[[0.05028931263166996, 0.05828407249870605, 0.5658724593707056, -3.1389912150156545, -0.015581839781920912, -0.0023067306072292616], [9.708196600080299e-06, -0.00043972200087118006, 0.0006608836942473406, -0.0004648574580850043, 5.989475787941066e-05, 0.11999908466384918], [0.05021738325951543, 0.05885655521847924, 0.5657312830193268, -3.139197768535797, -0.03555359144468023, -0.0023555834678150433]], [[0.05021738325951543, 0.05885655521847924, 0.5657312830193268, -3.139197768535797, -0.03555359144468023, -0.0023555834678150433], [4.786168732180252e-05, -0.0006120855798531129, 0.0005032667744606045, -0.00041913909020797587, 8.823197993126934e-05, 0.11999923557065176], [0.050176092419074295, 0.05920037750431723, 0.5657121057572968, -3.1391229360748247, -0.05720033861581853, -0.0023786897878668493]], [[0.050176092419074295, 0.05920037750431723, 0.5657121057572968, -3.1391229360748247, -0.05720033861581853, -0.0023786897878668493], [0.0024781516185140573, -0.0280075500863447, 0.01775788485406382, -0.0006504612686855111, 0.00017894480242310425, 0.19999886219400256], [0.05013796915195675, 0.05959226565481101, 0.5657934628336055, -3.138717745657207, -0.08567356876004069, -0.002501185780053596]]]],
]
